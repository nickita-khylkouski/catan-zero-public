"""Modal-distributed PPO ACTOR fleet (IMPALA / SEED-RL style).

Each CPU worker is a *self-play actor*: it loads the current policy from a shared Modal
volume, plays self-play games against league (PFSP) or fixed baseline opponents, and ships
pickled trajectory shards back through the on-disk contract in
``src/catan_zero/rl/ppo_distributed.py``. Actors are decoupled from the GPU learner — they
only ever read/write the shared run directory:

    {VOLUME_ROOT}/{run_name}/
      policy/   version.json + current.pt   (learner WRITES, actors READ + poll)
      trajectories/{worker_id}/shard_*.pkl   (actors WRITE)
      league/   league.json                  (optional; PFSP opponent sampling)

WITHIN-CONTAINER PARALLELISM (mirrors ``tools/modal_teacher_factory.py``): every ``cpu=8``
container splits its ``games_per_worker`` into ``cpu_workers`` (=8) chunks and plays them in
SEPARATE PROCESSES via ``ProcessPoolExecutor(max_workers=cpu_workers,
mp_context=spawn)`` — spawn (not fork) because torch forbids forking a process that has
imported CUDA/torch internals. Each worker process keeps ``TORCH_NUM_THREADS=1`` so the 8
processes saturate the 8 billed cores instead of a single-threaded loop wasting 7/8.

The PARENT coordinates staleness the IMPALA way: between chunk *rounds* it ``volume.reload()``s
and polls ``read_version``; when the learner has published a newer ``version`` it resubmits the
next round of chunks against the NEW checkpoint path (the simplest correct way to refresh
weights across a process pool — each worker reloads the policy at chunk start). Every shard is
stamped with the ``policy_version`` it was played under so the learner can drop over-stale data.

This mirrors ``tools/modal_teacher_factory.py`` (same image, ``@app.function`` with ``cpu=8``,
``Volume.from_name(create_if_missing=True)``, ``_payloads`` builder, ``rollout_worker.map``,
``volume.commit()`` / ``volume.reload()``, ``commit_every_chunks`` batching, and
``@app.local_entrypoint()`` launchers). Build-only: the entrypoints DEFINE the launchers but
nothing is invoked at import time.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import modal


APP_NAME = "catan-zero-ppo-factory"
VOLUME_NAME = "catan-zero-ppo"
REMOTE_ROOT = Path("/root/catan-zero")
VOLUME_ROOT = Path("/data")

# Default fixed opponents for the cold-start (no league yet) — resolved via the named
# baselines in ``policy_pool`` / ``factory_common.make_named_policy``.
DEFAULT_OPPONENTS = "random,heuristic,jsettlers_lite,catanatron_ab3"

# Seat names used by ColonistMultiAgentEnv (must match train_ppo.py).
SEAT_NAMES = ("BLUE", "RED", "ORANGE", "WHITE")

# Max league/opponent checkpoints kept resident per worker process. Each frozen entity_graph
# snapshot is ~140MB; over a 6h run the learner can publish 100+ of them, so an unbounded cache
# OOMs the 16GB container. Cap with an LRU (see ``_OpponentResolver``).
DEFAULT_OPPONENT_CACHE_SIZE = 8

# Cold-start gating: at worker start we block-poll for a published policy version so the first
# wave does NOT flood the volume with version-0 (BC) shards the learner will drop as stale. The
# learner publishes v0 at startup so this is normally a short wait; we cap it to avoid hanging.
DEFAULT_COLD_START_TIMEOUT_SECS = 300.0
COLD_START_POLL_SECS = 2.0

# Backpressure: if the learner's published version hasn't advanced in this many actor rounds we
# sleep before the next round so 75 containers don't flood the volume with shards the learner
# can't drain (the IMPALA actors-too-far-ahead guard).
DEFAULT_MAX_ACTOR_LAG = 8
DEFAULT_LAG_STALL_ROUNDS = 4
DEFAULT_LAG_STALL_SLEEP = 10.0

# Version-poll throttle: don't ``volume.reload()`` + ``read_version`` every single round across
# 75 containers — poll the published version at most this often (overridable via --policy-poll-secs).
DEFAULT_POLICY_POLL_SECS = 15.0

_RUN_MANIFEST_JSON_KEY = "run_manifest_json"
_RUN_MANIFEST_SHA256_KEY = "run_manifest_sha256"


def _actor_manifest_science(manifest: Any) -> dict[str, Any]:
    """Return every actor-science field owned by a v2 run manifest."""
    identity = manifest.spec.identity
    actor = manifest.spec.actor
    return {
        "architecture": identity.architecture,
        "track": identity.track,
        "vps_to_win": identity.vps_to_win,
        "max_decisions": actor.max_decisions,
        "games_per_shard": actor.games_per_shard,
        "gamma": actor.gamma,
        "gae_lambda": actor.gae_lambda,
        "action_temperature": actor.action_temperature,
        "value_shaping_coef": actor.value_shaping_coef,
        "value_shaping_scale": actor.value_shaping_scale,
        "value_shaping_opponent_penalty": actor.value_shaping_opponent_penalty,
        "seed": actor.seed,
        "opponent_mode": actor.opponent_mode,
        # The manifest order is behaviorally significant. Never sort it.
        "opponents": ",".join(actor.opponents),
        "pfsp_mode": actor.pfsp_mode,
    }


def _actor_container_manifest_science(
    manifest: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    """Account for the launcher's deterministic per-container seed partition."""
    science = _actor_manifest_science(manifest)
    worker_id = str(payload.get("worker_id", ""))
    prefix = "actor_"
    suffix = worker_id.removeprefix(prefix)
    if worker_id.startswith(prefix) and suffix.isdigit():
        science["seed"] += int(suffix) * max(1, int(payload["games_per_worker"]))
    return science


def _learner_manifest_science(manifest: Any) -> dict[str, Any]:
    """Return the legacy learner fields exposed by the Modal entrypoints."""
    identity = manifest.spec.identity
    actor = manifest.spec.actor
    learner = manifest.spec.learner
    return {
        "architecture": identity.architecture,
        "max_steps": learner.max_steps,
        "shards_per_step": learner.shards_per_step,
        "max_staleness": learner.max_staleness,
        "lr": learner.lr,
        "clip_ratio": learner.clip_ratio,
        "value_coef": learner.value_coef,
        "entropy_coef": learner.entropy_coef,
        "ppo_epochs": learner.ppo_epochs,
        "minibatch_size": learner.minibatch_size,
        "behavior_temperature": actor.action_temperature,
        "gamma": actor.gamma,
        "gae_lambda": actor.gae_lambda,
        "vtrace_clip_rho": learner.vtrace_clip_rho,
        "vtrace_clip_pg_rho": learner.vtrace_clip_pg_rho,
        "advantage_normalization": learner.advantage_normalization,
        "vtrace_forward_chunk": learner.vtrace_forward_chunk,
        "use_vtrace": learner.use_vtrace,
    }


def _reject_manifest_science_conflicts(
    payload: dict[str, Any], expected: dict[str, Any]
) -> None:
    conflicts = {
        key: (payload[key], expected_value)
        for key, expected_value in expected.items()
        if key in payload and payload[key] != expected_value
    }
    if conflicts:
        details = ", ".join(
            f"{key}={actual!r} (manifest {wanted!r})"
            for key, (actual, wanted) in sorted(conflicts.items())
        )
        raise ValueError(f"run manifest conflicts with legacy science: {details}")


def _verify_manifest_initializer(
    manifest: Any,
    init_checkpoint: str,
    *,
    required: bool,
) -> None:
    """Hash initializer bytes when visible; containers require them to exist."""
    from catan_zero.rl import ppo_distributed as ppd

    path = Path(init_checkpoint)
    if not path.is_file():
        if required:
            raise RuntimeError(f"cannot hash init checkpoint: file not found: {path}")
        return
    actual = f"sha256:{ppd.checkpoint_sha256(path)}"
    expected = manifest.spec.identity.initializer_sha256
    if actual != expected:
        raise ValueError(
            "init checkpoint SHA-256 does not match run manifest identity: "
            f"expected={expected} actual={actual}"
        )


def _bound_manifest_from_payload(
    payload: dict[str, Any],
    *,
    init_checkpoint: str,
    require_initializer: bool,
) -> Any | None:
    """Reconstruct and authenticate an optional canonical manifest envelope."""
    from catan_zero.rl.ppo_run_manifest import PPORunManifest

    raw = payload.get(_RUN_MANIFEST_JSON_KEY)
    claimed_sha256 = payload.get(_RUN_MANIFEST_SHA256_KEY)
    if raw is None and claimed_sha256 is None:
        return None
    if type(raw) is not str or type(claimed_sha256) is not str:
        raise ValueError("run manifest payload requires canonical JSON and SHA-256")
    manifest = PPORunManifest.from_json(raw)
    if manifest.status != "bound":
        raise ValueError("run manifest must have status='bound'; templates cannot run")
    actual_sha256 = manifest.sha256()
    if claimed_sha256 != actual_sha256:
        raise ValueError(
            "run manifest SHA-256 mismatch: "
            f"expected={claimed_sha256} actual={actual_sha256}"
        )
    if raw != manifest.canonical_json():
        raise ValueError("run manifest payload must use canonical JSON")
    _verify_manifest_initializer(
        manifest, init_checkpoint, required=require_initializer
    )
    return manifest


def _manifest_envelope_from_path(
    run_manifest: str,
    *,
    init_checkpoint: str,
) -> tuple[Any, dict[str, str]]:
    """Load a local manifest once and prepare its authenticated Modal payload."""
    from catan_zero.rl.ppo_run_manifest import load_manifest

    manifest = load_manifest(run_manifest)
    if manifest.status != "bound":
        raise ValueError("run manifest must have status='bound'; templates cannot run")
    _verify_manifest_initializer(manifest, init_checkpoint, required=False)
    return manifest, {
        _RUN_MANIFEST_JSON_KEY: manifest.canonical_json(),
        _RUN_MANIFEST_SHA256_KEY: manifest.sha256(),
    }


def _add_learner_manifest(
    payload: dict[str, Any],
    run_manifest: str | None,
) -> dict[str, Any]:
    if run_manifest is None:
        return payload
    manifest, envelope = _manifest_envelope_from_path(
        run_manifest, init_checkpoint=str(payload["init_checkpoint"])
    )
    science = _learner_manifest_science(manifest)
    _reject_manifest_science_conflicts(payload, science)
    payload.update(science)
    payload.update(envelope)
    return payload


def _run_manifest_chunk_fields(payload: dict[str, Any]) -> dict[str, str]:
    """Propagate authenticated v2 identity to every child-process chunk."""
    value = payload.get(_RUN_MANIFEST_SHA256_KEY)
    return {} if value is None else {_RUN_MANIFEST_SHA256_KEY: str(value)}


# The actor fleet needs torch + catanatron on top of the teacher-factory image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "networkx>=3.0",
        "gymnasium>=1.0",
        "zstandard",
        "torch>=2.0",
        "catanatron",
        "modal",
        "protobuf",
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'tools'}",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TORCH_NUM_THREADS": "1",
        }
    )
    .add_local_dir("src", remote_path=str(REMOTE_ROOT / "src"))
    .add_local_dir("tools", remote_path=str(REMOTE_ROOT / "tools"))
    .add_local_dir("vendor", remote_path=str(REMOTE_ROOT / "vendor"))
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_ROOT / "pyproject.toml"))
    .add_local_file("catan_rules_v1.json", remote_path=str(REMOTE_ROOT / "catan_rules_v1.json"))
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- opponents
class _OpponentResolver:
    """Builds the per-episode opponent dict, caching loaded checkpoints by path.

    Two modes:
      * ``league`` — sample frozen league members via PFSP and load their checkpoints with
        ``load_ppo_policy`` (LRU-cached). Falls back to fixed opponents if the league is empty.
      * ``fixed`` — round-robin over named baselines (``make_named_policy``), rebuilt per
        episode (the search/AB baselines are cheap to construct and carry per-game state).

    The checkpoint cache is a bounded LRU (``max_cache``): frozen league snapshots are ~140MB
    each, so an unbounded cache OOMs the 16GB container over a long run. Evicted entries are
    least-recently-used. Every cached policy is ``freeze_in_place``d (eval + requires_grad=False).
    """

    def __init__(
        self,
        *,
        mode: str,
        opponents: str,
        run_root: Path,
        architecture: str,
        device: str,
        pfsp_mode: str,
        max_cache: int = DEFAULT_OPPONENT_CACHE_SIZE,
    ) -> None:
        self._mode = str(mode)
        self._names = [name.strip() for name in str(opponents).split(",") if name.strip()]
        self._run_root = run_root
        self._architecture = str(architecture)
        self._device = str(device)
        self._pfsp_mode = str(pfsp_mode)
        # OrderedDict as an LRU: most-recently-used moved to the end, evict from the front.
        self._checkpoint_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._max_cache = max(1, int(max_cache))
        self._league = None
        self._main_id: str | None = None
        if self._mode == "league":
            self._try_load_league()

    @property
    def effective_mode(self) -> str:
        """``league`` only if a non-empty league actually loaded, else ``fixed``."""
        return "league" if self._league is not None else "fixed"

    def _try_load_league(self) -> None:
        from catan_zero.rl import ppo_distributed as ppd
        from catan_zero.rl.league import League

        league_path = ppd.league_dir(self._run_root)
        if not (league_path / "league.json").exists():
            return
        try:
            league = League.load(str(league_path))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return
        mains = [a for a in league._agents.values() if a.role == "main"]
        if not mains:
            return
        # Pick the most recently created main as the perspective agent.
        self._main_id = max(mains, key=lambda a: a.created_step).id
        self._league = league

    def reload_league(self) -> None:
        """Refresh the league from disk (called after ``volume.reload()``)."""
        if self._mode != "league":
            return
        self._league = None
        self._main_id = None
        self._try_load_league()

    def _load_checkpoint(self, checkpoint: str) -> Any:
        cached = self._checkpoint_cache.get(checkpoint)
        if cached is not None:
            self._checkpoint_cache.move_to_end(checkpoint)  # mark most-recently-used
            return cached
        from catan_zero.rl.ppo_policy_factory import freeze_in_place, load_ppo_policy

        policy = freeze_in_place(
            load_ppo_policy(checkpoint, architecture=self._architecture, device=self._device)
        )
        self._checkpoint_cache[checkpoint] = policy
        self._checkpoint_cache.move_to_end(checkpoint)
        while len(self._checkpoint_cache) > self._max_cache:
            self._checkpoint_cache.popitem(last=False)  # evict least-recently-used
        return policy

    def opponents_for(
        self,
        seats: tuple[str, ...],
        rng,
    ) -> dict[str, Any]:
        if self._league is not None and self._main_id is not None:
            try:
                agent = self._league.sample_opponent(
                    self._main_id, mode=self._pfsp_mode, rng=rng
                )
                opponent = self._load_checkpoint(agent.checkpoint_path)
                return {seat: opponent for seat in seats}
            except (ValueError, OSError):
                pass  # league exists but unusable this draw -> fixed fallback
        return self._fixed_opponents(seats, rng)

    def _fixed_opponents(self, seats: tuple[str, ...], rng) -> dict[str, Any]:
        from factory_common import make_named_policy

        names = self._names or [DEFAULT_OPPONENTS.split(",")[0]]
        opponents: dict[str, Any] = {}
        for seat in seats:
            name = names[int(rng.integers(0, len(names)))]
            opponents[seat] = make_named_policy(name, device=self._device)
        return opponents


# --------------------------------------------------------------------------- rollout speedups
def _maybe_quantize_rollout(policy: Any) -> bool:
    """Dynamic-INT8 quantize the LOADED ROLLOUT policy's Linear layers IN PLACE. Fail-open.

    Applies ``torch.ao.quantization.quantize_dynamic(policy.model, {nn.Linear}, qint8)`` to the
    rollout copy only — NEVER the learner (this worker never owns learner weights). Dynamic
    quantization is the easy first step and needs no calibration data.

    TODO(perf): static INT8 with the x86/fbgemm backend (calibrated activations, prepare/convert)
    is the bigger win — research shows up to ~3x on x86 vs dynamic's ~1.5-2x — but it requires a
    calibration pass over representative observations. Wire that once a calibration set exists.

    Returns True if quantization was applied; on ANY failure logs and returns False so the caller
    keeps the fp32 model (no-op fallback).
    """
    try:
        import torch

        model = getattr(policy, "model", None)
        if model is None:
            return False
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        policy.model = quantized
        return True
    except Exception as exc:  # noqa: BLE001 - fail-open: any quant error -> stay fp32
        print(
            json.dumps({"event": "quantize_rollout_failed", "error": repr(exc)}),
            flush=True,
        )
        return False


def _tune_rollout_threads() -> None:
    """Pin intra-op threads to 1 for the per-process rollout loop.

    The container runs ``cpu_workers`` rollout processes so the 8 billed cores are saturated by
    process-level parallelism; letting torch spin up its own intra-op thread pool per process
    oversubscribes the cores and thrashes. ``TORCH_NUM_THREADS=1`` is already set in the image env,
    but we also call the API directly so a spawned process that didn't inherit it stays single-op.
    """
    try:
        import torch

        torch.set_num_threads(1)
        # interop pool: 1 is fine for a serial rollout loop; guard since it can only be set once.
        try:
            torch.set_num_interop_threads(1)
        except Exception:  # noqa: BLE001 - already set in this process; ignore
            pass
    except Exception:  # noqa: BLE001 - torch missing/odd build: best-effort only
        pass


# --------------------------------------------------------------------------- chunk worker
def _run_actor_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Play ONE chunk of games in a separate (spawned) process and write its trajectory shards.

    Runs under ``ProcessPoolExecutor`` inside the container so the 8 billed cores are saturated
    by 8 processes (``TORCH_NUM_THREADS=1`` each). The chunk loads the policy from the weights
    path the PARENT handed it (``policy_path`` + ``policy_version``); the parent refreshes this
    path between rounds when the learner publishes newer weights, so the workers never reload
    weights mid-chunk — they reload at chunk start.

    Trajectories are CPU-only (``device='cpu'``). Shards are written through the on-disk contract
    and stamped with ``policy_version`` for the learner's staleness filter. The parent owns
    ``volume.commit()`` (batched), so this worker only WRITES shard files.
    """
    import numpy as np
    import torch

    from catan_zero.rl import ppo_distributed as ppd
    from catan_zero.rl.ppo_policy_factory import (
        load_ppo_policy,
        validate_canonical_ppo_actor_contract,
    )
    from catan_zero.rl.torch_ppo import collect_ppo_episode
    from factory_common import parse_track

    run_name = str(chunk["run_name"])
    worker_id = str(chunk["worker_id"])
    architecture = str(chunk.get("architecture", "entity_graph"))
    device = str(chunk.get("device", "cpu"))  # actors run CPU-only inference
    games = int(chunk["games"])
    game_offset = int(chunk["game_offset"])
    shard_base = int(chunk["shard_base"])
    games_per_shard = max(1, int(chunk["games_per_shard"]))
    seed = int(chunk["seed"])
    policy_path = str(chunk["policy_path"])
    policy_version = int(chunk["policy_version"])
    max_cache = int(chunk.get("opponent_cache_size", DEFAULT_OPPONENT_CACHE_SIZE))
    quantize_rollout = bool(chunk.get("quantize_rollout", False))
    action_temperature = float(chunk.get("action_temperature", 1.0))
    validate_canonical_ppo_actor_contract(
        architecture=architecture,
        gamma=float(chunk.get("gamma", 1.0)),
        gae_lambda=float(chunk.get("gae_lambda", 0.95)),
        action_temperature=action_temperature,
    )

    # Pin intra-op threads to 1 so the cpu_workers processes own the cores (no oversubscription).
    _tune_rollout_threads()

    root = ppd.run_root(VOLUME_ROOT, run_name)

    policy = load_ppo_policy(policy_path, architecture=architecture, device=device)
    # Rollout is eval-only: put the model in eval() so dropout/BN are inference-stable.
    model = getattr(policy, "model", None)
    if model is not None:
        model.eval()
    if quantize_rollout:
        _maybe_quantize_rollout(policy)  # fail-open: no-ops + logs on failure, stays fp32

    config = parse_track(str(chunk["track"]), vps_to_win=int(chunk["vps_to_win"]))
    players = int(config.players)
    seat_names = SEAT_NAMES[:players]
    max_decisions = int(chunk["max_decisions"])

    resolver = _OpponentResolver(
        mode=str(chunk.get("opponent_mode", "fixed")),
        opponents=str(chunk.get("opponents", DEFAULT_OPPONENTS)),
        run_root=root,
        architecture=architecture,
        device=device,
        pfsp_mode=str(chunk.get("pfsp_mode", "pfsp")),
        max_cache=max_cache,
    )

    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    buffer: list[Any] = []
    shard_index = shard_base
    shard_paths: list[str] = []
    samples_written = 0
    games_played = 0
    games_in_buffer = 0
    samples_in_buffer = 0

    # ``inference_mode`` is strictly faster than ``no_grad`` for the rollout (no autograd version
    # counters / view tracking); the whole self-play loop is forward-only inference.
    with torch.inference_mode():
        while games_played < games:
            # Round-robin the learner seat over the global game index so seats stay balanced even
            # when a container's games are split across chunks.
            training_seat = seat_names[(game_offset + games_played) % len(seat_names)]
            training_seats = {training_seat}
            opponent_seats = tuple(name for name in seat_names if name != training_seat)
            opponents = resolver.opponents_for(opponent_seats, rng)

            trajectory = collect_ppo_episode(
                policy,
                opponents,
                seed=int(rng.integers(2**31)),
                config=config,
                max_decisions=max_decisions,
                rng=rng,
                training_seats=training_seats,
                gamma=float(chunk.get("gamma", 1.0)),
                gae_lambda=float(chunk.get("gae_lambda", 0.95)),
                value_shaping_coef=float(chunk.get("value_shaping_coef", 0.0)),
                value_shaping_scale=float(chunk.get("value_shaping_scale", 100.0)),
                value_shaping_opponent_penalty=float(
                    chunk.get("value_shaping_opponent_penalty", 0.05)
                ),
                action_temperature=action_temperature,
            )
            buffer.append(trajectory)
            games_played += 1
            games_in_buffer += 1
            samples_in_buffer += len(trajectory.samples)

            if games_in_buffer >= games_per_shard:
                path = ppd.write_trajectory_shard(
                    root,
                    worker_id,
                    shard_index,
                    buffer,
                    policy_version=policy_version,
                    run_manifest_sha256=chunk.get(_RUN_MANIFEST_SHA256_KEY),
                )
                shard_paths.append(str(path))
                samples_written += samples_in_buffer
                shard_index += 1
                buffer = []
                games_in_buffer = 0
                samples_in_buffer = 0

    if buffer:
        path = ppd.write_trajectory_shard(
            root,
            worker_id,
            shard_index,
            buffer,
            policy_version=policy_version,
            run_manifest_sha256=chunk.get(_RUN_MANIFEST_SHA256_KEY),
        )
        shard_paths.append(str(path))
        samples_written += samples_in_buffer
        shard_index += 1

    elapsed = time.perf_counter() - start
    return {
        "worker_id": worker_id,
        "opponent_mode": resolver.effective_mode,
        "games": games_played,
        "shards": len(shard_paths),
        "shard_paths": shard_paths,
        "samples": samples_written,
        "policy_version": policy_version,
        "next_shard_index": shard_index,
        "elapsed_sec": elapsed,
    }


# --------------------------------------------------------------------------- container core
def _cold_start_wait(root, ppd, *, timeout_secs: float, container_id: str) -> Any:
    """BLOCK until the learner publishes a policy version, or ``timeout_secs`` elapses.

    Actors must not waste a big first wave producing version-0 (BC) shards that the learner will
    drop as stale (cold-start gating, FIX H3). We ``volume.reload()`` between polls so we observe
    the learner's freshly-committed ``version.json``. The learner publishes v0 at startup so this
    is normally a SHORT wait.

    Returns the ``PublishedVersion`` if one appeared, else ``None`` (timed out -> caller falls back
    to ``init_checkpoint`` and logs ``cold_start_timeout``).
    """
    deadline = time.perf_counter() + max(0.0, float(timeout_secs))
    while True:
        published = ppd.read_version(root)
        if published is not None:
            return published
        if time.perf_counter() >= deadline:
            return None
        time.sleep(COLD_START_POLL_SECS)
        volume.reload()  # pull the learner's latest commit before re-checking


def _run_actor(payload: dict[str, Any]) -> dict[str, Any]:
    """Container entrypoint: parallelize ``games_per_worker`` across ``cpu_workers`` processes.

    Mirrors ``modal_teacher_factory._run_worker``: a ``ProcessPoolExecutor`` (spawn ctx) fans the
    games out across the 8 cores. We submit work in ROUNDS so the parent can refresh the policy
    checkpoint between rounds when the learner publishes newer weights (the IMPALA staleness
    bound). Shards are committed in batches (``commit_every_shards``) and at most every
    ``commit_min_secs`` seconds to avoid the per-shard ``volume.commit()`` churn that 75
    containers would otherwise generate.
    """
    from catan_zero.rl import ppo_distributed as ppd
    from catan_zero.rl.ppo_policy_factory import (
        canonical_actor_rollout_contract_fields,
        validate_canonical_ppo_actor_contract,
    )

    run_name = str(payload["run_name"])
    container_id = str(payload["worker_id"])
    architecture = str(payload.get("architecture", "entity_graph"))
    device = str(payload.get("device", "cpu"))  # actors run CPU-only inference
    games_per_worker = int(payload["games_per_worker"])
    games_per_shard = max(1, int(payload["games_per_shard"]))
    cpu_workers = max(1, int(payload.get("cpu_workers", 8)))
    seed = int(payload["seed"])
    commit_every_shards = max(1, int(payload.get("commit_every_shards", 4)))
    commit_min_secs = float(payload.get("commit_min_secs", 0.0))
    opponent_cache_size = int(payload.get("opponent_cache_size", DEFAULT_OPPONENT_CACHE_SIZE))
    quantize_rollout = bool(payload.get("quantize_rollout", False))
    cold_start_timeout_secs = float(
        payload.get("cold_start_timeout_secs", DEFAULT_COLD_START_TIMEOUT_SECS)
    )
    policy_poll_secs = float(payload.get("policy_poll_secs", DEFAULT_POLICY_POLL_SECS))
    max_actor_lag = max(1, int(payload.get("max_actor_lag", DEFAULT_MAX_ACTOR_LAG)))
    lag_stall_rounds = max(1, int(payload.get("lag_stall_rounds", DEFAULT_LAG_STALL_ROUNDS)))
    lag_stall_sleep = float(payload.get("lag_stall_sleep", DEFAULT_LAG_STALL_SLEEP))
    rollout_contract = canonical_actor_rollout_contract_fields(payload)
    validate_canonical_ppo_actor_contract(
        architecture=architecture,
        gamma=rollout_contract["gamma"],
        gae_lambda=rollout_contract["gae_lambda"],
        action_temperature=rollout_contract["action_temperature"],
    )

    root = ppd.run_root(VOLUME_ROOT, run_name)
    has_manifest_payload = any(
        key in payload for key in (_RUN_MANIFEST_JSON_KEY, _RUN_MANIFEST_SHA256_KEY)
    )
    if has_manifest_payload:
        volume.reload()  # make initializer and any existing binding visible first
        manifest = _bound_manifest_from_payload(
            payload,
            init_checkpoint=str(payload["init_checkpoint"]),
            require_initializer=True,
        )
        assert manifest is not None
        _reject_manifest_science_conflicts(
            payload, _actor_container_manifest_science(manifest, payload)
        )
        # Bind before any runtime read/write. This preserves the v2 pristine-root guard.
        ppd.bind_run_manifest(root, manifest)
        ppd.ensure_run_dirs(root)
        # Publish the immutable identity before any actor can poll or emit data.
        volume.commit()
    else:
        manifest = None
        ppd.ensure_run_dirs(root)
        volume.reload()  # preserve the historical legacy ordering

    # ---- cold-start gating (FIX H3): block-poll until the learner has published a real version
    #      so the first wave doesn't produce version-0 BC shards the learner will drop as stale. ----
    cold_start_timed_out = False
    published = _cold_start_wait(
        root, ppd, timeout_secs=cold_start_timeout_secs, container_id=container_id
    )

    # ---- resolve the starting policy path + version. FIX H1: load the EXACT bytes the version
    #      stamp refers to via ``read_version().path`` (versioned ``weights_v{N}.pt`` once the
    #      backbone migrates; ``current.pt`` today) so loaded weights match the stamped version. ----
    if published is not None:
        policy_path = str(published.path)
        current_policy_version = int(published.version)
        init_source = "published"
    else:
        # Timed out waiting for a publish: fall back to the BC warm-start and log it.
        cold_start_timed_out = True
        policy_path = str(payload["init_checkpoint"])
        current_policy_version = 0  # version 0 == pre-publish BC warm-start
        init_source = "init_checkpoint"
        print(
            json.dumps(
                {
                    "event": "cold_start_timeout",
                    "run_name": run_name,
                    "worker_id": container_id,
                    "cold_start_timeout_secs": cold_start_timeout_secs,
                    "fallback": "init_checkpoint",
                }
            ),
            flush=True,
        )

    # The learner normally creates this immutable binding before publishing the
    # version observed above. The actor verifies the same initializer and
    # behavior distribution before producing any shard; a timeout fallback may
    # create it, but can never overwrite a conflicting learner contract.
    if manifest is None:
        ppd.bind_run_contract(
            root,
            init_checkpoint=payload["init_checkpoint"],
            architecture=architecture,
            gamma=rollout_contract["gamma"],
            gae_lambda=rollout_contract["gae_lambda"],
            behavior_temperature=rollout_contract["action_temperature"],
        )

    # ---- per-process shard namespaces: each of the cpu_workers processes owns a disjoint
    #      worker_id so their shard files never collide on disk. ----
    sub_worker_ids = [f"{container_id}_p{p:02d}" for p in range(cpu_workers)]
    next_shard_index = {wid: 0 for wid in sub_worker_ids}

    start = time.perf_counter()
    games_done = 0
    total_shards = 0
    total_samples = 0
    policy_reloads = 0
    uncommitted_shards = 0
    last_commit_time = start

    # ---- version-poll throttle (FIX L5) + backpressure (FIX backpressure) state ----
    last_poll_time = 0.0  # 0 => first round always polls (after the cold-start wait)
    last_seen_version = current_policy_version  # learner version observed last poll
    rounds_since_advance = 0  # consecutive rounds the published version hasn't advanced
    throttles = 0

    mp_context = multiprocessing.get_context("spawn")  # torch requires spawn, not fork
    with ProcessPoolExecutor(max_workers=cpu_workers, mp_context=mp_context) as executor:
        round_index = 0
        while games_done < games_per_worker:
            # ---- staleness refresh (FIX L5: rate-limited): only ``volume.reload()`` + read the
            #      version at most every ``policy_poll_secs`` so 75 containers don't hammer the
            #      volume reloading every round. Between polls we keep the last-known weights. ----
            now_poll = time.perf_counter()
            if (now_poll - last_poll_time) >= policy_poll_secs:
                volume.reload()
                latest = ppd.read_version(root)
                last_poll_time = now_poll
                latest_version = int(latest.version) if latest is not None else current_policy_version
                if latest is not None and latest_version > current_policy_version:
                    # FIX H1: track the versioned path so stamped shards match loaded bytes.
                    policy_path = str(latest.path)
                    current_policy_version = latest_version
                    policy_reloads += 1
                # ---- backpressure: has the LEARNER's published version advanced since last poll? ----
                if latest_version > last_seen_version:
                    rounds_since_advance = 0
                    last_seen_version = latest_version
                else:
                    rounds_since_advance += 1
            else:
                rounds_since_advance += 1

            # ---- backpressure (FIX): if the learner isn't draining (version stalled for
            #      ``lag_stall_rounds`` rounds) OR we're piling up uncommitted shards beyond
            #      ``max_actor_lag``, sleep briefly so we don't flood the volume with shards the
            #      learner can't keep up with. ----
            learner_stalled = rounds_since_advance >= lag_stall_rounds
            shards_backed_up = uncommitted_shards >= max_actor_lag
            if games_done < games_per_worker and (learner_stalled or shards_backed_up):
                throttles += 1
                print(
                    json.dumps(
                        {
                            "event": "actor_throttle",
                            "run_name": run_name,
                            "worker_id": container_id,
                            "current_version": current_policy_version,
                            "rounds_since_advance": rounds_since_advance,
                            "uncommitted_shards": uncommitted_shards,
                            "reason": "learner_stalled" if learner_stalled else "shards_backed_up",
                            "sleep_secs": lag_stall_sleep,
                        }
                    ),
                    flush=True,
                )
                time.sleep(max(0.0, lag_stall_sleep))
                rounds_since_advance = 0  # reset so we don't busy-throttle every round

            remaining = games_per_worker - games_done
            # One round = up to cpu_workers chunks; size chunks so a round covers ~all cores.
            per_chunk = max(1, -(-remaining // cpu_workers))  # ceil-div across workers
            chunks: list[dict[str, Any]] = []
            offset = games_done
            for p in range(cpu_workers):
                if offset >= games_per_worker:
                    break
                chunk_games = min(per_chunk, games_per_worker - offset)
                chunks.append(
                    {
                        "run_name": run_name,
                        "worker_id": sub_worker_ids[p],
                        "games": chunk_games,
                        "game_offset": offset,
                        "shard_base": next_shard_index[sub_worker_ids[p]],
                        "games_per_shard": games_per_shard,
                        "seed": seed + round_index * 1_000_003 + p * 9_973,
                        "policy_path": policy_path,
                        "policy_version": current_policy_version,
                        "quantize_rollout": quantize_rollout,
                        "architecture": architecture,
                        "device": device,
                        "track": payload["track"],
                        "vps_to_win": payload["vps_to_win"],
                        "max_decisions": payload["max_decisions"],
                        "opponent_mode": payload.get("opponent_mode", "fixed"),
                        "opponents": payload.get("opponents", DEFAULT_OPPONENTS),
                        "pfsp_mode": payload.get("pfsp_mode", "pfsp"),
                        "opponent_cache_size": opponent_cache_size,
                        **_run_manifest_chunk_fields(payload),
                        **rollout_contract,
                        "value_shaping_coef": payload.get("value_shaping_coef", 0.0),
                        "value_shaping_scale": payload.get("value_shaping_scale", 100.0),
                        "value_shaping_opponent_penalty": payload.get(
                            "value_shaping_opponent_penalty", 0.05
                        ),
                    }
                )
                offset += chunk_games

            futures = [executor.submit(_run_actor_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                games_done += int(result["games"])
                total_shards += int(result["shards"])
                total_samples += int(result["samples"])
                uncommitted_shards += int(result["shards"])
                next_shard_index[str(result["worker_id"])] = int(result["next_shard_index"])

                # ---- batched commit: flush shards to the volume in groups, and at most every
                #      commit_min_secs seconds, instead of after every single shard. ----
                now = time.perf_counter()
                due_by_count = uncommitted_shards >= commit_every_shards
                due_by_time = commit_min_secs > 0.0 and (now - last_commit_time) >= commit_min_secs
                if uncommitted_shards > 0 and (due_by_count or due_by_time):
                    volume.commit()
                    uncommitted_shards = 0
                    last_commit_time = now

            round_index += 1

    # ---- final commit of any straggler shards ----
    if uncommitted_shards > 0:
        volume.commit()
        uncommitted_shards = 0

    elapsed = time.perf_counter() - start
    return {
        "run_name": run_name,
        "worker_id": container_id,
        "init_source": init_source,
        "cpu_workers": cpu_workers,
        "games": games_done,
        "shards": total_shards,
        "samples": total_samples,
        "final_policy_version": current_policy_version,
        "policy_reloads": policy_reloads,
        "cold_start_timed_out": cold_start_timed_out,
        "quantize_rollout": quantize_rollout,
        "throttles": throttles,
        "elapsed_sec": elapsed,
        "games_per_sec": games_done / elapsed if elapsed > 0 else 0.0,
        "samples_per_sec": total_samples / elapsed if elapsed > 0 else 0.0,
    }


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=8,
    memory=16_384,
    max_containers=75,
    timeout=6 * 3_600,
)
def ppo_actor_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    return _run_actor(payload)


# Alias to match the teacher factory's name so launch code reads identically.
rollout_worker = ppo_actor_worker


# --------------------------------------------------------------------------- GPU learner
@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    gpu="A100",
    memory=32_768,
    timeout=24 * 3_600,
)
def ppo_learner(config_payload: dict[str, Any]) -> dict[str, Any]:
    """Run the distributed-PPO GPU learner INSIDE Modal so it can mount the shared volume.

    The learner (``tools/ppo_distributed_learner.py``) must read the same Modal volume the
    actors write to. It cannot run on the separate Oracle box (which can't mount the Modal
    volume), so it runs here as an A100 Modal function. It:

      * chdir to the repo and imports the learner module (lazily — the other agent is editing
        that file in parallel, so we reference ``train``/``main`` defensively),
      * builds a ``LearnerConfig`` from ``config_payload`` (run_base on the volume, init
        checkpoint on the volume, device='cuda'),
      * passes a ``volume_reload_fn=volume.reload`` hook so the learner calls ``volume.reload()``
        before scanning the run dir for new shards (the other agent is adding the
        ``--reload-volume`` / ``volume_reload_fn`` hook),
      * runs the train loop to completion (or ``max_steps``).
    """
    os.chdir(REMOTE_ROOT)

    # Lazy import: the learner module is being edited by another agent; resolve its callable
    # defensively so this wrapper still imports cleanly if names shift.
    import importlib

    learner = importlib.import_module("ppo_distributed_learner")

    run_name = str(config_payload["run_name"])
    init_checkpoint = str(config_payload["init_checkpoint"])
    run_base = str(config_payload.get("run_base", str(VOLUME_ROOT)))
    architecture = str(config_payload.get("architecture", "entity_graph"))
    device = str(config_payload.get("device", "cuda"))

    has_manifest_payload = any(
        key in config_payload
        for key in (_RUN_MANIFEST_JSON_KEY, _RUN_MANIFEST_SHA256_KEY)
    )
    if has_manifest_payload:
        volume.reload()
        manifest = _bound_manifest_from_payload(
            config_payload,
            init_checkpoint=init_checkpoint,
            require_initializer=True,
        )
        assert manifest is not None
        _reject_manifest_science_conflicts(
            config_payload, _learner_manifest_science(manifest)
        )
        # The learner's v2 binder requires a pristine run root. Keep the input
        # manifest in /tmp; train() will bind its canonical contents into the root.
        manifest_path = Path("/tmp") / (
            "catan_zero_ppo_" + manifest.sha256().removeprefix("sha256:") + ".json"
        )
        manifest_path.write_text(manifest.canonical_json(), encoding="utf-8")
        config, _args = learner.resolve_config(
            [
                "--run-manifest",
                str(manifest_path),
                "--run-base",
                run_base,
                "--run-name",
                run_name,
                "--init-checkpoint",
                init_checkpoint,
                "--device",
                device,
            ]
        )
    else:
        # Preserve the historical construction path byte-for-byte in legacy mode.
        from catan_zero.rl import ppo_distributed as ppd

        manifest = None
        ppd.ensure_run_dirs(ppd.run_root(run_base, run_name))
        volume.reload()
        LearnerConfig = getattr(learner, "LearnerConfig")
        known_fields = set(getattr(LearnerConfig, "__dataclass_fields__", {}).keys())
        cfg_kwargs: dict[str, Any] = {
            "run_base": run_base,
            "run_name": run_name,
            "init_checkpoint": init_checkpoint,
            "architecture": architecture,
            "device": device,
        }
        for key, value in config_payload.items():
            if key in known_fields and key not in cfg_kwargs:
                cfg_kwargs[key] = value
        cfg_kwargs = {
            k: v for k, v in cfg_kwargs.items() if not known_fields or k in known_fields
        }
        config = LearnerConfig(**cfg_kwargs)

    # The checked-out learner contract requires both volume hooks. Invoke it once and let any
    # runtime TypeError propagate: retrying after a partial update can duplicate training work.
    reload_fn = volume.reload
    commit_fn = volume.commit
    train_fn = getattr(learner, "train", None)
    main_fn = getattr(learner, "main", None)

    started = time.perf_counter()
    if callable(train_fn):
        train_fn(  # type: ignore[call-arg]
            config, volume_reload_fn=reload_fn, volume_commit_fn=commit_fn
        )
    elif callable(main_fn):
        argv = [
            "--run-base",
            run_base,
            "--run-name",
            run_name,
            "--init-checkpoint",
            init_checkpoint,
            "--device",
            device,
        ]
        if manifest is not None:
            argv[0:0] = ["--run-manifest", str(manifest_path)]
        else:
            argv.extend(["--architecture", architecture])
        main_fn(argv)
    else:  # pragma: no cover - defensive
        raise RuntimeError("ppo_distributed_learner exposes neither train() nor main()")

    return {
        "event": "ppo_learner_done",
        "run_name": run_name,
        "run_base": run_base,
        "device": device,
        "elapsed_sec": time.perf_counter() - started,
        "run_root": str(Path(run_base) / run_name),
    }


# --------------------------------------------------------------------------- payloads
def _payloads(
    *,
    run_name: str,
    init_checkpoint: str,
    containers: int,
    games_per_container: int,
    cpu_workers: int,
    games_per_shard: int,
    commit_every_shards: int,
    commit_min_secs: float,
    opponent_cache_size: int,
    quantize_rollout: bool,
    cold_start_timeout_secs: float,
    policy_poll_secs: float,
    max_actor_lag: int,
    lag_stall_rounds: int,
    lag_stall_sleep: float,
    seed: int,
    architecture: str,
    device: str,
    track: str,
    vps_to_win: int,
    max_decisions: int,
    opponent_mode: str,
    opponents: str,
    pfsp_mode: str,
    gamma: float,
    gae_lambda: float,
    value_shaping_coef: float,
    value_shaping_scale: float,
    value_shaping_opponent_penalty: float,
    action_temperature: float,
    run_manifest: str | None = None,
) -> list[dict[str, Any]]:
    from catan_zero.rl.ppo_policy_factory import validate_canonical_ppo_actor_contract

    manifest_envelope: dict[str, str] = {}
    if run_manifest is not None:
        manifest, manifest_envelope = _manifest_envelope_from_path(
            run_manifest, init_checkpoint=init_checkpoint
        )
        supplied_science = {
            "architecture": architecture,
            "track": track,
            "vps_to_win": vps_to_win,
            "max_decisions": max_decisions,
            "games_per_shard": games_per_shard,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "action_temperature": action_temperature,
            "value_shaping_coef": value_shaping_coef,
            "value_shaping_scale": value_shaping_scale,
            "value_shaping_opponent_penalty": value_shaping_opponent_penalty,
            "seed": seed,
            "opponent_mode": opponent_mode,
            "opponents": opponents,
            "pfsp_mode": pfsp_mode,
        }
        manifest_science = _actor_manifest_science(manifest)
        _reject_manifest_science_conflicts(supplied_science, manifest_science)
        architecture = manifest_science["architecture"]
        track = manifest_science["track"]
        vps_to_win = manifest_science["vps_to_win"]
        max_decisions = manifest_science["max_decisions"]
        games_per_shard = manifest_science["games_per_shard"]
        gamma = manifest_science["gamma"]
        gae_lambda = manifest_science["gae_lambda"]
        action_temperature = manifest_science["action_temperature"]
        value_shaping_coef = manifest_science["value_shaping_coef"]
        value_shaping_scale = manifest_science["value_shaping_scale"]
        value_shaping_opponent_penalty = manifest_science[
            "value_shaping_opponent_penalty"
        ]
        seed = manifest_science["seed"]
        opponent_mode = manifest_science["opponent_mode"]
        opponents = manifest_science["opponents"]
        pfsp_mode = manifest_science["pfsp_mode"]

    validate_canonical_ppo_actor_contract(
        architecture=architecture,
        gamma=gamma,
        gae_lambda=gae_lambda,
        action_temperature=action_temperature,
    )
    return [
        {
            "run_name": run_name,
            "worker_id": f"actor_{index:05d}",
            "init_checkpoint": init_checkpoint,
            "games_per_worker": games_per_container,
            "cpu_workers": cpu_workers,
            "games_per_shard": games_per_shard,
            "commit_every_shards": commit_every_shards,
            "commit_min_secs": commit_min_secs,
            "opponent_cache_size": opponent_cache_size,
            "quantize_rollout": quantize_rollout,
            "cold_start_timeout_secs": cold_start_timeout_secs,
            "policy_poll_secs": policy_poll_secs,
            "max_actor_lag": max_actor_lag,
            "lag_stall_rounds": lag_stall_rounds,
            "lag_stall_sleep": lag_stall_sleep,
            "seed": seed + index * max(1, games_per_container),
            "architecture": architecture,
            "device": device,
            "track": track,
            "vps_to_win": vps_to_win,
            "max_decisions": max_decisions,
            "opponent_mode": opponent_mode,
            "opponents": opponents,
            "pfsp_mode": pfsp_mode,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "value_shaping_coef": value_shaping_coef,
            "value_shaping_scale": value_shaping_scale,
            "value_shaping_opponent_penalty": value_shaping_opponent_penalty,
            "action_temperature": action_temperature,
            **manifest_envelope,
        }
        for index in range(containers)
    ]


def _launch(
    *,
    run_name: str,
    init_checkpoint: str,
    containers: int,
    games_per_container: int,
    cpu_workers: int,
    games_per_shard: int,
    commit_every_shards: int,
    commit_min_secs: float,
    opponent_cache_size: int,
    quantize_rollout: bool,
    cold_start_timeout_secs: float,
    policy_poll_secs: float,
    max_actor_lag: int,
    lag_stall_rounds: int,
    lag_stall_sleep: float,
    seed: int,
    architecture: str,
    device: str,
    track: str,
    vps_to_win: int,
    max_decisions: int,
    opponent_mode: str,
    opponents: str,
    pfsp_mode: str,
    gamma: float,
    gae_lambda: float,
    value_shaping_coef: float,
    value_shaping_scale: float,
    value_shaping_opponent_penalty: float,
    action_temperature: float,
    run_manifest: str | None = None,
) -> None:
    started = time.perf_counter()
    run_id = f"{run_name}-{uuid.uuid4().hex[:12]}"
    payloads = _payloads(
        run_name=run_name,
        init_checkpoint=init_checkpoint,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        games_per_shard=games_per_shard,
        commit_every_shards=commit_every_shards,
        commit_min_secs=commit_min_secs,
        opponent_cache_size=opponent_cache_size,
        quantize_rollout=quantize_rollout,
        cold_start_timeout_secs=cold_start_timeout_secs,
        policy_poll_secs=policy_poll_secs,
        max_actor_lag=max_actor_lag,
        lag_stall_rounds=lag_stall_rounds,
        lag_stall_sleep=lag_stall_sleep,
        seed=seed,
        architecture=architecture,
        device=device,
        track=track,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        opponent_mode=opponent_mode,
        opponents=opponents,
        pfsp_mode=pfsp_mode,
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_shaping_coef=value_shaping_coef,
        value_shaping_scale=value_shaping_scale,
        value_shaping_opponent_penalty=value_shaping_opponent_penalty,
        action_temperature=action_temperature,
        run_manifest=run_manifest,
    )
    print(
        json.dumps(
            {
                "progress": "modal_ppo_launch",
                "run_name": run_name,
                "run_id": run_id,
                "containers": containers,
                "cpu_per_container": 8,
                "cpu_workers_per_container": cpu_workers,
                "max_physical_cpus": containers * 8,
                "target_games": containers * games_per_container,
                "init_checkpoint": init_checkpoint,
                "architecture": architecture,
                "track": track,
                "opponent_mode": opponent_mode,
                "opponents": opponents,
                "pfsp_mode": pfsp_mode,
                "commit_every_shards": commit_every_shards,
                "commit_min_secs": commit_min_secs,
                "opponent_cache_size": opponent_cache_size,
                "quantize_rollout": quantize_rollout,
                "cold_start_timeout_secs": cold_start_timeout_secs,
                "policy_poll_secs": policy_poll_secs,
                "max_actor_lag": max_actor_lag,
                "lag_stall_rounds": lag_stall_rounds,
                "lag_stall_sleep": lag_stall_sleep,
                "action_temperature": action_temperature,
                "volume": VOLUME_NAME,
                "run_root": str(VOLUME_ROOT / run_name),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    reports: list[dict[str, Any]] = []
    total_games = 0
    total_shards = 0
    total_samples = 0
    for report in rollout_worker.map(payloads, order_outputs=False):
        reports.append(report)
        total_games += int(report.get("games", 0))
        total_shards += int(report.get("shards", 0))
        total_samples += int(report.get("samples", 0))
        print(
            json.dumps(
                {
                    "progress": "modal_actor_done",
                    "run_name": run_name,
                    "actors_done": len(reports),
                    "actors_total": containers,
                    "worker_id": report.get("worker_id"),
                    "cpu_workers": report.get("cpu_workers"),
                    "games": report.get("games"),
                    "shards": report.get("shards"),
                    "samples": report.get("samples"),
                    "final_policy_version": report.get("final_policy_version"),
                    "policy_reloads": report.get("policy_reloads"),
                    "cold_start_timed_out": report.get("cold_start_timed_out"),
                    "quantize_rollout": report.get("quantize_rollout"),
                    "throttles": report.get("throttles"),
                    "opponent_mode": report.get("opponent_mode"),
                    "elapsed_sec": report.get("elapsed_sec"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "progress": "modal_ppo_complete",
                "run_name": run_name,
                "run_id": run_id,
                "actors": len(reports),
                "games": total_games,
                "shards": total_shards,
                "samples": total_samples,
                "elapsed_sec": time.perf_counter() - started,
                "run_root": str(VOLUME_ROOT / run_name),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


# --------------------------------------------------------------------------- entrypoints
def _validate_modal_learner_contract(
    *,
    architecture: str,
    behavior_temperature: float,
    gamma: float,
    gae_lambda: float,
    use_vtrace: bool,
    max_staleness: int,
    vtrace_clip_rho: float,
    vtrace_clip_pg_rho: float,
) -> None:
    from catan_zero.rl.ppo_policy_factory import (
        validate_canonical_ppo_actor_contract,
        validate_canonical_ppo_staleness_contract,
    )

    validate_canonical_ppo_actor_contract(
        architecture=architecture,
        gamma=gamma,
        gae_lambda=gae_lambda,
        action_temperature=behavior_temperature,
    )
    validate_canonical_ppo_staleness_contract(
        use_vtrace=use_vtrace,
        max_staleness=max_staleness,
        vtrace_clip_rho=vtrace_clip_rho,
        vtrace_clip_pg_rho=vtrace_clip_pg_rho,
    )


@app.local_entrypoint()
def smoke(
    run_name: str = "ppo_actor_smoke",
    init_checkpoint: str = "/data/bc_warmstart/current.pt",
    containers: int = 1,
    games_per_container: int = 4,
    cpu_workers: int = 8,
    games_per_shard: int = 2,
    commit_every_shards: int = 4,
    commit_min_secs: float = 0.0,
    opponent_cache_size: int = DEFAULT_OPPONENT_CACHE_SIZE,
    quantize_rollout: bool = False,
    cold_start_timeout_secs: float = DEFAULT_COLD_START_TIMEOUT_SECS,
    policy_poll_secs: float = DEFAULT_POLICY_POLL_SECS,
    max_actor_lag: int = DEFAULT_MAX_ACTOR_LAG,
    lag_stall_rounds: int = DEFAULT_LAG_STALL_ROUNDS,
    lag_stall_sleep: float = DEFAULT_LAG_STALL_SLEEP,
    seed: int = 70_628_650,
    architecture: str = "entity_graph",
    device: str = "cpu",
    track: str = "2p_no_trade",
    vps_to_win: int = 10,
    max_decisions: int = 1_200,
    opponent_mode: str = "fixed",
    opponents: str = DEFAULT_OPPONENTS,
    pfsp_mode: str = "pfsp",
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    value_shaping_coef: float = 0.0,
    value_shaping_scale: float = 100.0,
    value_shaping_opponent_penalty: float = 0.05,
    action_temperature: float = 1.0,
) -> None:
    _launch(
        run_name=run_name,
        init_checkpoint=init_checkpoint,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        games_per_shard=games_per_shard,
        commit_every_shards=commit_every_shards,
        commit_min_secs=commit_min_secs,
        opponent_cache_size=opponent_cache_size,
        quantize_rollout=quantize_rollout,
        cold_start_timeout_secs=cold_start_timeout_secs,
        policy_poll_secs=policy_poll_secs,
        max_actor_lag=max_actor_lag,
        lag_stall_rounds=lag_stall_rounds,
        lag_stall_sleep=lag_stall_sleep,
        seed=seed,
        architecture=architecture,
        device=device,
        track=track,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        opponent_mode=opponent_mode,
        opponents=opponents,
        pfsp_mode=pfsp_mode,
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_shaping_coef=value_shaping_coef,
        value_shaping_scale=value_shaping_scale,
        value_shaping_opponent_penalty=value_shaping_opponent_penalty,
        action_temperature=action_temperature,
    )


@app.local_entrypoint()
def launch_ppo_actors(
    run_name: str = "ppo_2p10_actors_600cpu_v1",
    init_checkpoint: str = "/data/bc_warmstart/current.pt",
    containers: int = 75,
    games_per_container: int = 256,
    cpu_workers: int = 8,
    games_per_shard: int = 8,
    commit_every_shards: int = 4,
    commit_min_secs: float = 30.0,
    opponent_cache_size: int = DEFAULT_OPPONENT_CACHE_SIZE,
    quantize_rollout: bool = False,
    cold_start_timeout_secs: float = DEFAULT_COLD_START_TIMEOUT_SECS,
    policy_poll_secs: float = DEFAULT_POLICY_POLL_SECS,
    max_actor_lag: int = DEFAULT_MAX_ACTOR_LAG,
    lag_stall_rounds: int = DEFAULT_LAG_STALL_ROUNDS,
    lag_stall_sleep: float = DEFAULT_LAG_STALL_SLEEP,
    seed: int = 70_628_700,
    architecture: str = "entity_graph",
    device: str = "cpu",
    track: str = "2p_no_trade",
    vps_to_win: int = 10,
    max_decisions: int = 1_200,
    opponent_mode: str = "league",
    opponents: str = DEFAULT_OPPONENTS,
    pfsp_mode: str = "pfsp",
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    value_shaping_coef: float = 0.0,
    value_shaping_scale: float = 100.0,
    value_shaping_opponent_penalty: float = 0.05,
    action_temperature: float = 1.0,
) -> None:
    _launch(
        run_name=run_name,
        init_checkpoint=init_checkpoint,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        games_per_shard=games_per_shard,
        commit_every_shards=commit_every_shards,
        commit_min_secs=commit_min_secs,
        opponent_cache_size=opponent_cache_size,
        quantize_rollout=quantize_rollout,
        cold_start_timeout_secs=cold_start_timeout_secs,
        policy_poll_secs=policy_poll_secs,
        max_actor_lag=max_actor_lag,
        lag_stall_rounds=lag_stall_rounds,
        lag_stall_sleep=lag_stall_sleep,
        seed=seed,
        architecture=architecture,
        device=device,
        track=track,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        opponent_mode=opponent_mode,
        opponents=opponents,
        pfsp_mode=pfsp_mode,
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_shaping_coef=value_shaping_coef,
        value_shaping_scale=value_shaping_scale,
        value_shaping_opponent_penalty=value_shaping_opponent_penalty,
        action_temperature=action_temperature,
    )


@app.local_entrypoint()
def launch_ppo_actors_from_manifest(
    run_manifest: str,
    run_name: str = "ppo_2p10_manifest_v2",
    init_checkpoint: str = "/data/bc_warmstart/current.pt",
    containers: int = 75,
    games_per_container: int = 256,
    cpu_workers: int = 8,
    commit_every_shards: int = 4,
    commit_min_secs: float = 30.0,
    opponent_cache_size: int = DEFAULT_OPPONENT_CACHE_SIZE,
    cold_start_timeout_secs: float = DEFAULT_COLD_START_TIMEOUT_SECS,
    policy_poll_secs: float = DEFAULT_POLICY_POLL_SECS,
    max_actor_lag: int = DEFAULT_MAX_ACTOR_LAG,
    lag_stall_rounds: int = DEFAULT_LAG_STALL_ROUNDS,
    lag_stall_sleep: float = DEFAULT_LAG_STALL_SLEEP,
    device: str = "cpu",
) -> None:
    """Launch actors with runtime wiring only; the manifest owns all science."""
    manifest, _envelope = _manifest_envelope_from_path(
        run_manifest, init_checkpoint=init_checkpoint
    )
    science = _actor_manifest_science(manifest)
    _launch(
        run_name=run_name,
        init_checkpoint=init_checkpoint,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        games_per_shard=science["games_per_shard"],
        commit_every_shards=commit_every_shards,
        commit_min_secs=commit_min_secs,
        opponent_cache_size=opponent_cache_size,
        # Quantization changes policy numerics and is not part of the v2 manifest identity.
        quantize_rollout=False,
        cold_start_timeout_secs=cold_start_timeout_secs,
        policy_poll_secs=policy_poll_secs,
        max_actor_lag=max_actor_lag,
        lag_stall_rounds=lag_stall_rounds,
        lag_stall_sleep=lag_stall_sleep,
        seed=science["seed"],
        architecture=science["architecture"],
        device=device,
        track=science["track"],
        vps_to_win=science["vps_to_win"],
        max_decisions=science["max_decisions"],
        opponent_mode=science["opponent_mode"],
        opponents=science["opponents"],
        pfsp_mode=science["pfsp_mode"],
        gamma=science["gamma"],
        gae_lambda=science["gae_lambda"],
        value_shaping_coef=science["value_shaping_coef"],
        value_shaping_scale=science["value_shaping_scale"],
        value_shaping_opponent_penalty=science["value_shaping_opponent_penalty"],
        action_temperature=science["action_temperature"],
        run_manifest=run_manifest,
    )


@app.local_entrypoint()
def launch_learner(
    run_name: str = "ppo_2p10_actors_600cpu_v1",
    init_checkpoint: str = "/data/bc_warmstart/current.pt",
    run_base: str = str(VOLUME_ROOT),
    gpu: str = "A100",
    architecture: str = "entity_graph",
    device: str = "cuda",
    max_steps: int = 0,
    shards_per_step: int = 16,
    max_staleness: int = 4,
    lr: float = 2.0e-4,
    clip_ratio: float = 0.1,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 2,
    minibatch_size: int = 65536,
    behavior_temperature: float = 1.0,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    vtrace_clip_rho: float = 1.0,
    vtrace_clip_pg_rho: float = 1.0,
    advantage_normalization: str = "global",
    vtrace_forward_chunk: int = 8192,
    no_vtrace: bool = False,
) -> None:
    """Spawn the GPU learner on Modal (reads the same volume the actors write to).

    DEFINE-only: this spawns the ``ppo_learner`` function; it does NOT block on it. The learner
    runs until ``max_steps`` (0 == forever) on the A100, pulling shards off the shared volume,
    publishing weights the actors poll, and checkpointing/eval-ing periodically.
    """
    _validate_modal_learner_contract(
        architecture=architecture,
        behavior_temperature=behavior_temperature,
        gamma=gamma,
        gae_lambda=gae_lambda,
        use_vtrace=not no_vtrace,
        max_staleness=max_staleness,
        vtrace_clip_rho=vtrace_clip_rho,
        vtrace_clip_pg_rho=vtrace_clip_pg_rho,
    )
    payload = {
        "run_name": run_name,
        "init_checkpoint": init_checkpoint,
        "run_base": run_base,
        "architecture": architecture,
        "device": device,
        "max_steps": max_steps,
        "shards_per_step": shards_per_step,
        "max_staleness": max_staleness,
        "lr": lr,
        "clip_ratio": clip_ratio,
        "value_coef": value_coef,
        "entropy_coef": entropy_coef,
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "behavior_temperature": behavior_temperature,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "vtrace_clip_rho": vtrace_clip_rho,
        "vtrace_clip_pg_rho": vtrace_clip_pg_rho,
        "advantage_normalization": advantage_normalization,
        "vtrace_forward_chunk": vtrace_forward_chunk,
        "use_vtrace": not no_vtrace,
    }
    print(
        json.dumps(
            {
                "progress": "modal_ppo_learner_launch",
                "run_name": run_name,
                "run_base": run_base,
                "gpu": gpu,
                "device": device,
                "init_checkpoint": init_checkpoint,
                "max_steps": max_steps,
                "shards_per_step": shards_per_step,
                "minibatch_size": minibatch_size,
                "ppo_epochs": ppo_epochs,
                "behavior_temperature": behavior_temperature,
                "advantage_normalization": advantage_normalization,
                "use_vtrace": not no_vtrace,
                "volume": VOLUME_NAME,
                "run_root": str(Path(run_base) / run_name),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    handle = ppo_learner.spawn(payload)
    print(
        json.dumps(
            {
                "progress": "modal_ppo_learner_spawned",
                "run_name": run_name,
                "object_id": getattr(handle, "object_id", None),
            },
            sort_keys=True,
        ),
        flush=True,
    )


@app.local_entrypoint()
def launch_learner_from_manifest(
    run_manifest: str,
    run_name: str = "ppo_2p10_manifest_v2",
    init_checkpoint: str = "/data/bc_warmstart/current.pt",
    run_base: str = str(VOLUME_ROOT),
    device: str = "cuda",
    blocking: bool = False,
) -> None:
    """Launch the fixed-A100 learner with runtime wiring only."""
    payload = _add_learner_manifest(
        {
            "run_name": run_name,
            "init_checkpoint": init_checkpoint,
            "run_base": run_base,
            "device": device,
        },
        run_manifest,
    )
    science = _learner_manifest_science(
        _bound_manifest_from_payload(
            payload,
            init_checkpoint=init_checkpoint,
            require_initializer=False,
        )
    )
    print(
        json.dumps(
            {
                "progress": "modal_ppo_manifest_learner_launch",
                "run_name": run_name,
                "run_base": run_base,
                "gpu": "A100",
                "device": device,
                "run_manifest_sha256": payload[_RUN_MANIFEST_SHA256_KEY],
                "max_steps": science["max_steps"],
                "lr": science["lr"],
                "minibatch_size": science["minibatch_size"],
                "blocking": blocking,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if blocking:
        result = ppo_learner.remote(payload)
        print(
            json.dumps(
                {"progress": "modal_ppo_manifest_learner_done", "result": result},
                default=str,
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        handle = ppo_learner.spawn(payload)
        print(
            json.dumps(
                {
                    "progress": "modal_ppo_manifest_learner_spawned",
                    "object_id": getattr(handle, "object_id", None),
                },
                sort_keys=True,
            ),
            flush=True,
        )


@app.local_entrypoint()
def run_learner_blocking(
    run_name: str = "ppo_2p10_actors_600cpu_v1",
    init_checkpoint: str = "/data/bc_warmstart/current.pt",
    run_base: str = str(VOLUME_ROOT),
    gpu: str = "A100",
    architecture: str = "entity_graph",
    device: str = "cuda",
    max_steps: int = 1,
    shards_per_step: int = 16,
    max_staleness: int = 4,
    lr: float = 2.0e-4,
    clip_ratio: float = 0.1,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 2,
    minibatch_size: int = 65536,
    behavior_temperature: float = 1.0,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    vtrace_clip_rho: float = 1.0,
    vtrace_clip_pg_rho: float = 1.0,
    advantage_normalization: str = "global",
    vtrace_forward_chunk: int = 8192,
    no_vtrace: bool = False,
) -> None:
    """Run the Modal GPU learner as a blocking smoke job.

    The non-blocking launcher uses ``spawn`` and returns immediately. That is useful for long
    production runs, but in some environments the ephemeral app can stop before the spawned
    learner publishes its initial policy. This entrypoint intentionally blocks so short capped
    Modal actor/learner smokes keep the learner alive while actors feed shards.
    """
    _validate_modal_learner_contract(
        architecture=architecture,
        behavior_temperature=behavior_temperature,
        gamma=gamma,
        gae_lambda=gae_lambda,
        use_vtrace=not no_vtrace,
        max_staleness=max_staleness,
        vtrace_clip_rho=vtrace_clip_rho,
        vtrace_clip_pg_rho=vtrace_clip_pg_rho,
    )
    payload = {
        "run_name": run_name,
        "init_checkpoint": init_checkpoint,
        "run_base": run_base,
        "architecture": architecture,
        "device": device,
        "max_steps": max_steps,
        "shards_per_step": shards_per_step,
        "max_staleness": max_staleness,
        "lr": lr,
        "clip_ratio": clip_ratio,
        "value_coef": value_coef,
        "entropy_coef": entropy_coef,
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "behavior_temperature": behavior_temperature,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "vtrace_clip_rho": vtrace_clip_rho,
        "vtrace_clip_pg_rho": vtrace_clip_pg_rho,
        "advantage_normalization": advantage_normalization,
        "vtrace_forward_chunk": vtrace_forward_chunk,
        "use_vtrace": not no_vtrace,
    }
    print(
        json.dumps(
            {
                "progress": "modal_ppo_learner_blocking_start",
                "run_name": run_name,
                "run_base": run_base,
                "gpu": gpu,
                "device": device,
                "init_checkpoint": init_checkpoint,
                "max_steps": max_steps,
                "shards_per_step": shards_per_step,
                "minibatch_size": minibatch_size,
                "ppo_epochs": ppo_epochs,
                "behavior_temperature": behavior_temperature,
                "advantage_normalization": advantage_normalization,
                "use_vtrace": not no_vtrace,
                "volume": VOLUME_NAME,
                "run_root": str(Path(run_base) / run_name),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result = ppo_learner.remote(payload)
    print(
        json.dumps(
            {
                "progress": "modal_ppo_learner_blocking_complete",
                "run_name": run_name,
                "result": result,
            },
            default=str,
            sort_keys=True,
        ),
        flush=True,
    )
