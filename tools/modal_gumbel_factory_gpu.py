"""Modal GPU self-play factory for Gumbel + true-chance-node MCTS generation.

GPU adaptation of `tools/modal_gumbel_factory.py` (the CPU-only variant). The
CPU factory ran 8 single-threaded torch-INT8 workers per 8-core container
(per-eval ~34ms CPU-bound); on GPU the 35M entity_graph net forwards in
~0.33ms, so ONE fp32 worker per L4 GPU container (device="cuda") is the fast
path. Everything else -- the shard-tree output format, the resume/commit
protocol, the run_id-stamped part manifests, `build_gumbel_gen_manifest.py`
consumption -- is inherited unchanged.

KEY DIFFERENCES vs the CPU factory:
  * Image uses the DEFAULT (CUDA) torch build, NOT the cpu index_url.
  * `@app.function(gpu="L4", max_containers=44)` and ONE inner worker
    (44 = hard GPU cap, 1-GPU margin under the standing "Modal <45 L4s" rule; see
    the `max_containers=44` comment at the decorator).
  * Evaluator runs on device="cuda" in fp32 (NO int8 dynamic quantization --
    quantize_dynamic is a CPU-only speedup and would be a no-op/regression on
    the GPU).
  * `public_observation=True` is WIRED through to EntityGraphRustEvaluatorConfig
    (the CPU factory omits it, defaulting False). The v3a checkpoint is
    masked-trained (mask_hidden_info=True), so the task #76 safety net in
    `_assert_public_observation_matches_checkpoint_training` HARD-FAILS unless
    public_observation=True -- a clean pass IS the masked-regime proof.
  * `lazy_interior_chance=True` and `c_scale=0.03` wired into the search config.
  * Seed base 20_000_001+ (disjoint from A100 gen-1 9.3M-13.8M and B200 H2H
    ~9.3-9.4M); verified with tools/seed_fleet_planner.assert_disjoint_seed_blocks.
  * Output under the DISTINCT volume path prefix `gen1_modal_gpu/`.

Volume layout (volume `catan-zero-gumbel-data`, shared with the CPU factory but
a disjoint run_name prefix):
    /data/checkpoints/<name>/checkpoint.pt              <- `modal volume put`
    /data/gen1_modal_gpu/<run>/parts/part_XXXXX/
        worker_000/                                     <- run_worker_games tree
        manifest.json                                   <- container summary

Operating procedure (Modal authenticated LOCALLY, workspace prox/dev-dennis;
run from a checkout that has src/tools/vendor + the cp311 wheel at
/tmp/catanatron_rs_wheels/ so add_local_dir/add_local_file resolve):

  1. Upload the seed checkpoint once:
       modal volume put catan-zero-gumbel-data \
           <local checkpoint_masked.pt> checkpoints/v3a_masked/checkpoint.pt
  2. SMALL validation (BLOCKS, prints per-part games/hr + aggregate):
       modal run tools/modal_gumbel_factory_gpu.py::launch_gpu_pilot \
           --run-name gen1_modal_gpu/pilot_v1 \
           --checkpoint-rel checkpoints/v3a_masked/checkpoint.pt \
           --containers 4 --games-per-container 4
  3. Full wave (GATED on team-lead go, cap 100 containers):
       modal run tools/modal_gumbel_factory_gpu.py::launch_gpu_gen \
           --run-name gen1_modal_gpu/wave1 \
           --checkpoint-rel checkpoints/v3a_masked/checkpoint.pt \
           --containers 100 --games-per-container 500
     Spawns and exits; poll with ::summarize, then:
       modal volume get catan-zero-gumbel-data gen1_modal_gpu/wave1 <local_dir>
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import uuid

import modal


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resume_semantics_sha256(payload: dict[str, Any], checkpoint: Path) -> str:
    """Bind retry semantics to immutable model bytes and all science fields."""

    operational = {"run_name", "run_id", "part_index", "commit_secs", "resume"}
    science_payload = {
        key: value for key, value in payload.items() if key not in operational
    }
    science_payload["producer_checkpoint_sha256"] = _file_sha256(checkpoint)
    encoded = json.dumps(
        science_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


APP_NAME = "catan-zero-gumbel-factory-gpu"
VOLUME_NAME = "catan-zero-gumbel-data"
REMOTE_ROOT = Path("/root/catan-zero")
VOLUME_ROOT = Path("/data")

# The compiled pyo3 engine wheel (manylinux_2_34, cp311). Image uses python 3.11
# so the cp311 wheel is the ABI match. Rebuilding the wheel? Update BOTH names.
WHEEL_NAME = "catanatron_rs-0.1.2-cp311-cp311-manylinux_2_34_x86_64.whl"
LOCAL_WHEEL_PATH = f"/tmp/catanatron_rs_wheels/{WHEEL_NAME}"

# Fresh high base, disjoint from the A100 gen-1 seed table (9_300_001 ..
# 13_800_001) and the B200 H2H (~9.3-9.4M). game_seed = base_seed + game_index,
# and game_index in [0, total_games), so the whole fleet occupies
# [base_seed, base_seed + total_games). 20M leaves a wide moat.
DEFAULT_BASE_SEED = 20_000_001

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "networkx>=3.0", "gymnasium>=1.0", "zstandard")
    # DEFAULT PyPI torch => the CUDA build (bundles CUDA 12.x runtime, runs on
    # L4/Ada sm_89). The CPU factory pinned index_url=.../whl/cpu; we must NOT.
    .pip_install("torch==2.12.1")
    # modal in-image guarantees the container runtime's client deps (grpclib,
    # protobuf<7) live in site-packages (pilot_v1 postmortem from CPU factory).
    .pip_install("modal==1.5.1")
    .add_local_file(LOCAL_WHEEL_PATH, f"/root/wheels/{WHEEL_NAME}", copy=True)
    .run_commands(f"pip install /root/wheels/{WHEEL_NAME}")
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'tools'}",
            # One GPU worker; let torch/BLAS use the container's CPU cores for
            # the Rust-adjacent featurization + game loop (CPU-bound part).
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

# The resume/wipe/hard-error decision for an existing `part_dir` is pure
# stdlib logic (no modal/torch/CUDA/Rust dependency) split into its own
# module specifically so it's unit-testable in a plain local Python
# environment -- see tests/test_gumbel_resume.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gumbel_factory_resume import resolve_part_resume_action  # noqa: E402


# ------------------------------------------------------------- child process
def _run_gpu_worker(worker_args: dict[str, Any]) -> dict[str, Any]:
    """One inner worker process: fp32 CUDA evaluator + run_worker_games.

    Top-level and picklable (ProcessPoolExecutor spawn ctx). NEVER raises a
    worker-level failure out: a caught error is returned as a zeroed summary so
    one dead worker cannot lose already-written shards from the part manifest.
    """
    worker_index = int(worker_args.get("worker_index", -1))
    try:
        import torch

        from catan_zero.rl.gumbel_self_play import (
            GumbelSelfPlayConfig,
            run_worker_games,
        )
        from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig
        from catan_zero.search.neural_rust_mcts import (
            EntityGraphRustEvaluator,
            EntityGraphRustEvaluatorConfig,
        )

        device = str(worker_args.get("device", "cuda"))
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else ""
        print(
            json.dumps(
                {
                    "event": "worker_start",
                    "worker_index": worker_index,
                    "device": device,
                    "cuda_available": cuda_available,
                    "gpu_name": gpu_name,
                    "public_observation": bool(worker_args["public_observation"]),
                }
            ),
            flush=True,
        )
        if device == "cuda" and not cuda_available:
            raise RuntimeError(
                "device='cuda' requested but torch.cuda.is_available() is False "
                "in-container (image built without the CUDA torch build, or no "
                "GPU attached to the function)."
            )

        # fp32 on GPU. NO int8 quantize_dynamic: that is a CPU-only path and
        # would be a no-op (or a regression) on the GPU. public_observation MUST
        # be True here to clear the task #76 safety net against the masked
        # checkpoint (mask_hidden_info=True) -- the assert fires inside __init__.
        evaluator = EntityGraphRustEvaluator.from_checkpoint(
            worker_args["checkpoint"],
            device=device,
            config=EntityGraphRustEvaluatorConfig(
                value_scale=float(worker_args["value_scale"]),
                prior_temperature=float(worker_args["prior_temperature"]),
                public_observation=bool(worker_args["public_observation"]),
            ),
        )

        config = GumbelSelfPlayConfig(
            track=str(worker_args["track"]),
            vps_to_win=int(worker_args["vps_to_win"]),
            obs_width=int(worker_args["obs_width"]),
            max_decisions=int(worker_args["max_decisions"]),
            temperature_move_fraction=float(worker_args["temperature_move_fraction"]),
            temperature_high=float(worker_args["temperature_high"]),
            temperature_low=float(worker_args["temperature_low"]),
            correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
        )
        search_config = GumbelChanceMCTSConfig(
            max_depth=int(worker_args["max_depth"]),
            seed=int(worker_args["worker_seed"]),
            c_visit=float(worker_args["c_visit"]),
            c_scale=float(worker_args["c_scale"]),
            prior_temperature=float(worker_args["prior_temperature"]),
            n_full=int(worker_args["n_full"]),
            n_fast=int(worker_args["n_fast"]),
            p_full=float(worker_args["p_full"]),
            lazy_interior_chance=bool(worker_args["lazy_interior_chance"]),
            correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
        )
        summary = run_worker_games(
            out_dir=Path(worker_args["out_dir"]),
            games=int(worker_args["games"]),
            game_index_start=int(worker_args["game_index_start"]),
            base_seed=int(worker_args["base_seed"]),
            worker_seed=int(worker_args["worker_seed"]),
            config=config,
            search_config=search_config,
            evaluator=evaluator,
            shard_size=int(worker_args["shard_size"]),
            fmt=str(worker_args["fmt"]),
            run_id=str(worker_args.get("run_id", "")),
            resume=bool(worker_args.get("resume", False)),
            resume_semantics_sha256=str(worker_args["resume_semantics_sha256"]),
        )
        summary["worker_index"] = worker_index
        summary["evaluator_mode"] = "torch_fp32_cuda"
        summary["device"] = device
        summary["public_observation"] = bool(worker_args["public_observation"])
        return summary
    except Exception as error:  # noqa: BLE001 - isolate one worker from the part
        return {
            "worker_index": worker_index,
            "out_dir": str(worker_args.get("out_dir", "")),
            "games_requested": int(worker_args.get("games", 0)),
            "games_completed": 0,
            "games_failed": int(worker_args.get("games", 0)),
            "games_truncated": 0,
            "wins_by_color": {},
            "rows": 0,
            "decisions_total": 0,
            "forced_decisions_total": 0,
            "simulations_used_total": 0,
            "elapsed_sec": 0.0,
            "rows_per_sec": 0.0,
            "shards": [],
            "evaluator_mode": "torch_fp32_cuda",
            "device": str(worker_args.get("device", "cuda")),
            "public_observation": bool(worker_args.get("public_observation", True)),
            "errors": [
                {
                    "worker_index": worker_index,
                    "game_index": None,
                    "game_seed": None,
                    "error": f"worker-level failure before any game ran: {error!r}",
                }
            ],
        }


# --------------------------------------------------------- container function
@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    gpu="L4",
    cpu=4,  # game loop + Rust featurization are CPU-bound; only NN forward is GPU
    memory=16_384,
    # HARD GPU CAP (standing user constraint: Modal must STAY UNDER 45 L4s ALWAYS).
    # max_containers is the per-Function upper limit -- Modal queues excess inputs
    # rather than exceeding it (per Modal scaling docs), and retries re-queue, they
    # do NOT spawn containers beyond this. 44 leaves a 1-GPU safety margin under 45.
    # NOTE: this cap is PER container pool; run ONE wave (one app) at a time --
    # two concurrent ephemeral apps = two pools = 2x this cap (that combo caused a
    # transient 50-container breach on 2026-07-06; root-caused via the docs).
    max_containers=44,
    # 24h: at the validated ~36 games/hr/GPU (~100s/game), 500 games/container
    # is ~14h wall, so the full 50k-game wave fits in ONE fire without tripping
    # the timeout. Periodic volume.commit() + resume=True cover preemption.
    timeout=86_400,
    retries=2,
)
def gpu_part_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Play one part's games with ONE fp32 CUDA worker + periodic commits."""
    os.chdir(REMOTE_ROOT)

    run_name = str(payload["run_name"])
    run_id = str(payload.get("run_id", ""))
    part_index = int(payload["part_index"])
    games = int(payload["games"])
    commit_secs = max(30.0, float(payload.get("commit_secs", 240.0)))
    resume = bool(payload.get("resume", False))

    part_dir = VOLUME_ROOT / run_name / "parts" / f"part_{part_index:05d}"
    manifest_path = part_dir / "manifest.json"
    marker_path = part_dir / ".run_id"

    # Resume / retry / stale-output protocol. A same-run_id reinvocation
    # (Modal preempted and auto-retried this exact task) is NOT wiped
    # anymore -- it falls through to `run_worker_games(resume=True)`, which
    # picks up from the last durably-flushed game instead of replaying from
    # game 0 (see `resolve_part_resume_action`'s docstring for why this is
    # safe and the different-run_id guard still hard-errors unchanged).
    action, complete = resolve_part_resume_action(
        part_dir=part_dir,
        manifest_path=manifest_path,
        marker_path=marker_path,
        run_id=run_id,
        resume=resume,
    )
    if action == "return_complete":
        return complete
    if action == "wipe_and_restart":
        shutil.rmtree(part_dir)
    incremental_resume = action == "incremental_resume"

    part_dir.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(run_id, encoding="utf-8")
    volume.commit()

    # Stage the checkpoint on container-local disk once.
    volume_checkpoint = VOLUME_ROOT / str(payload["checkpoint_rel"])
    if not volume_checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not on volume: {volume_checkpoint} "
            f"(modal volume put {VOLUME_NAME} <local.pt> {payload['checkpoint_rel']})"
        )
    local_checkpoint = Path("/tmp/gumbel_checkpoint.pt")
    shutil.copyfile(volume_checkpoint, local_checkpoint)

    # ONE worker owns the whole GPU and all of this part's games.
    fleet_ordinal = part_index  # one worker per part -> ordinal is the part index
    worker_args = {
        "worker_index": 0,
        "games": games,
        "game_index_start": int(payload["game_index_start"]),
        "out_dir": str(part_dir / "worker_000"),
        "checkpoint": str(local_checkpoint),
        "device": str(payload.get("device", "cuda")),
        "base_seed": int(payload["base_seed"]),
        # Decorrelate search RNG across the whole fleet.
        "worker_seed": int(payload["base_seed"]) + 0x9E3779B9 * (fleet_ordinal + 1),
        "n_full": int(payload["n_full"]),
        "n_fast": int(payload["n_fast"]),
        "p_full": float(payload["p_full"]),
        "c_visit": float(payload["c_visit"]),
        "c_scale": float(payload["c_scale"]),
        "lazy_interior_chance": bool(payload["lazy_interior_chance"]),
        "max_decisions": int(payload["max_decisions"]),
        "max_depth": int(payload["max_depth"]),
        "temperature_move_fraction": float(payload["temperature_move_fraction"]),
        "temperature_high": float(payload["temperature_high"]),
        "temperature_low": float(payload["temperature_low"]),
        "prior_temperature": float(payload["prior_temperature"]),
        "value_scale": float(payload["value_scale"]),
        "public_observation": bool(payload["public_observation"]),
        "track": str(payload["track"]),
        "vps_to_win": int(payload["vps_to_win"]),
        "obs_width": int(payload["obs_width"]),
        "correct_rust_chance_spectra": bool(payload["correct_rust_chance_spectra"]),
        "shard_size": int(payload["shard_size"]),
        "fmt": str(payload["fmt"]),
        "run_id": run_id,
        "resume_semantics_sha256": _resume_semantics_sha256(
            payload, local_checkpoint
        ),
        # Always ask `run_worker_games` to attempt an incremental resume: if
        # `<out_dir>/progress.json` doesn't exist (first launch of this
        # part, or a "wipe_and_restart" that just cleared it), this is a
        # no-op and behavior is identical to before resume support existed.
        "resume": True,
    }

    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "part_start",
                "run_name": run_name,
                "run_id": run_id,
                "part_index": part_index,
                "games": games,
                "gpu": "L4",
                "public_observation": bool(payload["public_observation"]),
                "incremental_resume": incremental_resume,
            }
        ),
        flush=True,
    )

    # torch forbids forking a process with torch/CUDA initialized -> spawn.
    # One worker, but the pool keeps the periodic-commit loop alive so a
    # preemption loses only ~commit_secs of shards.
    mp_context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_context) as pool:
        pending = {pool.submit(_run_gpu_worker, worker_args)}
        last_commit = time.perf_counter()
        while pending:
            done, pending = wait(pending, timeout=commit_secs, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())  # never raises by contract
            if done or (time.perf_counter() - last_commit) >= commit_secs:
                volume.commit()
                last_commit = time.perf_counter()

    elapsed = time.perf_counter() - started
    games_completed = sum(int(r.get("games_completed", 0)) for r in results)
    summary: dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "part_index": part_index,
        "games_requested": games,
        "games_completed": games_completed,
        "games_failed": sum(int(r.get("games_failed", 0)) for r in results),
        "games_truncated": sum(int(r.get("games_truncated", 0)) for r in results),
        "rows": sum(int(r.get("rows", 0)) for r in results),
        "decisions_total": sum(int(r.get("decisions_total", 0)) for r in results),
        "forced_decisions_total": sum(int(r.get("forced_decisions_total", 0)) for r in results),
        "simulations_used_total": sum(int(r.get("simulations_used_total", 0)) for r in results),
        "evaluator_mode": "torch_fp32_cuda",
        "device": str(payload.get("device", "cuda")),
        "public_observation": bool(payload["public_observation"]),
        "base_seed": int(payload["base_seed"]),
        "game_index_start": int(payload["game_index_start"]),
        "elapsed_sec": elapsed,
        "games_per_hour": games_completed / max(elapsed / 3600.0, 1e-9),
        "shards": [
            os.path.relpath(shard, part_dir)
            for r in results
            for shard in r.get("shards", ())
        ],
        "workers": results,
        "errors": [err for r in results for err in r.get("errors", [])],
    }
    tmp_manifest = part_dir / "manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tmp_manifest.replace(manifest_path)
    volume.commit()
    return summary


@app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=300)
def summarize_run(run_name: str, run_id: str = "") -> dict[str, Any]:
    """Aggregate completed part manifests for a run (safe to call mid-wave)."""
    volume.reload()
    parts: list[dict[str, Any]] = []
    for path in sorted((VOLUME_ROOT / run_name / "parts").glob("part_*/manifest.json")):
        part = json.loads(path.read_text(encoding="utf-8"))
        if run_id and str(part.get("run_id", "")) != str(run_id):
            continue
        parts.append(part)
    games = sum(int(p.get("games_completed", 0)) for p in parts)
    rows = sum(int(p.get("rows", 0)) for p in parts)
    max_elapsed = max((float(p.get("elapsed_sec", 0.0)) for p in parts), default=0.0)
    return {
        "run_name": run_name,
        "run_id": run_id,
        "parts_complete": len(parts),
        "games_completed": games,
        "rows": rows,
        "games_failed": sum(int(p.get("games_failed", 0)) for p in parts),
        "games_truncated": sum(int(p.get("games_truncated", 0)) for p in parts),
        "forced_fraction": (
            sum(int(p.get("forced_decisions_total", 0)) for p in parts)
            / max(sum(int(p.get("decisions_total", 0)) for p in parts), 1)
        ),
        "slowest_part_sec": max_elapsed,
        "fleet_games_per_hour_estimate": sum(
            float(p.get("games_per_hour", 0.0)) for p in parts
        ),
        "errors": sum(len(p.get("errors", [])) for p in parts),
    }


# ------------------------------------------------------------------ launchers
def _verify_disjoint_seeds(base_seed: int, containers: int, games_per_container: int) -> None:
    """Fleet-disjointness gate (task #77 tooling). Each part occupies the
    half-open seed block [base_seed + part*gpc, base_seed + part*gpc + gpc)
    because game_seed = base_seed + game_index. Verify no two parts overlap
    with the shared checker before launching."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from seed_fleet_planner import assert_disjoint_seed_blocks

    workers = [
        (f"part_{p:05d}", base_seed + p * games_per_container, games_per_container)
        for p in range(containers)
    ]
    assert_disjoint_seed_blocks(workers)  # raises ValueError on any overlap
    lo = base_seed
    hi = base_seed + containers * games_per_container
    print(
        json.dumps(
            {
                "progress": "seed_verify_ok",
                "base_seed": base_seed,
                "containers": containers,
                "games_per_container": games_per_container,
                "seed_range": [lo, hi],
                "disjoint_from": {
                    "a100_gen1": [9_300_001, 13_800_001],
                    "b200_h2h_approx": [9_300_000, 9_400_000],
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _payloads(
    *,
    run_name: str,
    run_id: str,
    checkpoint_rel: str,
    containers: int,
    games_per_container: int,
    base_seed: int,
    device: str,
    public_observation: bool,
    n_full: int,
    n_fast: int,
    p_full: float,
    lazy_interior_chance: bool,
    max_decisions: int,
    max_depth: int,
    temperature_move_fraction: float,
    temperature_high: float,
    temperature_low: float,
    prior_temperature: float,
    value_scale: float,
    c_visit: float,
    c_scale: float,
    track: str,
    vps_to_win: int,
    obs_width: int,
    correct_rust_chance_spectra: bool,
    shard_size: int,
    fmt: str,
    commit_secs: float,
    resume: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "run_name": run_name,
            "run_id": run_id,
            "part_index": part_index,
            "games": games_per_container,
            "game_index_start": part_index * games_per_container,
            "checkpoint_rel": checkpoint_rel,
            "base_seed": base_seed,
            "device": device,
            "public_observation": public_observation,
            "n_full": n_full,
            "n_fast": n_fast,
            "p_full": p_full,
            "lazy_interior_chance": lazy_interior_chance,
            "c_visit": c_visit,
            "c_scale": c_scale,
            "max_decisions": max_decisions,
            "max_depth": max_depth,
            "temperature_move_fraction": temperature_move_fraction,
            "temperature_high": temperature_high,
            "temperature_low": temperature_low,
            "prior_temperature": prior_temperature,
            "value_scale": value_scale,
            "track": track,
            "vps_to_win": vps_to_win,
            "obs_width": obs_width,
            "correct_rust_chance_spectra": correct_rust_chance_spectra,
            "shard_size": shard_size,
            "fmt": fmt,
            "commit_secs": commit_secs,
            "resume": resume,
        }
        for part_index in range(containers)
    ]


@app.local_entrypoint()
def launch_gpu_pilot(
    run_name: str = "gen1_modal_gpu/pilot_v1",
    checkpoint_rel: str = "checkpoints/v3a_masked/checkpoint.pt",
    containers: int = 4,
    games_per_container: int = 4,
    base_seed: int = DEFAULT_BASE_SEED,
    device: str = "cuda",
    public_observation: bool = True,
    n_full: int = 64,
    n_fast: int = 16,
    p_full: float = 0.25,
    lazy_interior_chance: bool = True,
    max_decisions: int = 600,
    max_depth: int = 80,
    temperature_move_fraction: float = 0.075,
    temperature_high: float = 1.0,
    temperature_low: float = 0.0,
    prior_temperature: float = 1.0,
    value_scale: float = 1.0,
    c_visit: float = 50.0,
    c_scale: float = 0.03,
    track: str = "2p_no_trade",
    vps_to_win: int = 10,
    obs_width: int = 806,
    shard_size: int = 2048,
    fmt: str = "npz_zst",
    commit_secs: float = 240.0,
    resume: bool = False,
) -> None:
    """Small GPU validation run: a few L4 containers x a few games, BLOCKS.

    Confirms: image builds, torch.cuda.is_available() in-container, checkpoint
    loads from the volume, public_observation active (task #76 safety net
    passes against the masked checkpoint), games complete + shards written,
    disjoint seeds. Prints per-part games/hr + aggregate for the $/1k estimate.
    """
    _verify_disjoint_seeds(base_seed, containers, games_per_container)
    run_id = f"{run_name.replace('/', '_')}-{uuid.uuid4().hex[:12]}"
    payloads = _payloads(
        run_name=run_name,
        run_id=run_id,
        checkpoint_rel=checkpoint_rel,
        containers=containers,
        games_per_container=games_per_container,
        base_seed=base_seed,
        device=device,
        public_observation=public_observation,
        n_full=n_full,
        n_fast=n_fast,
        p_full=p_full,
        lazy_interior_chance=lazy_interior_chance,
        max_decisions=max_decisions,
        max_depth=max_depth,
        temperature_move_fraction=temperature_move_fraction,
        temperature_high=temperature_high,
        temperature_low=temperature_low,
        prior_temperature=prior_temperature,
        value_scale=value_scale,
        c_visit=c_visit,
        c_scale=c_scale,
        track=track,
        vps_to_win=vps_to_win,
        obs_width=obs_width,
        correct_rust_chance_spectra=True,
        shard_size=shard_size,
        fmt=fmt,
        commit_secs=commit_secs,
        resume=resume,
    )
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "progress": "pilot_launch",
                "run_name": run_name,
                "run_id": run_id,
                "containers": containers,
                "games_target": containers * games_per_container,
                "gpu": "L4",
                "public_observation": public_observation,
                "volume": VOLUME_NAME,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for report in gpu_part_worker.map(payloads, order_outputs=False):
        print(
            json.dumps(
                {
                    "progress": "pilot_part_done",
                    "part_index": report["part_index"],
                    "games": report["games_completed"],
                    "rows": report["rows"],
                    "truncated": report["games_truncated"],
                    "elapsed_sec": round(report["elapsed_sec"], 1),
                    "games_per_hour": round(report["games_per_hour"], 2),
                    "public_observation": report.get("public_observation"),
                    "errors": len(report["errors"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = summarize_run.remote(run_name, run_id)
    summary["wall_sec"] = time.perf_counter() - started
    print(json.dumps({"progress": "pilot_complete", **summary}, indent=2, sort_keys=True))


@app.local_entrypoint()
def launch_gpu_gen(
    run_name: str,
    checkpoint_rel: str,
    containers: int = 100,
    games_per_container: int = 500,
    base_seed: int = DEFAULT_BASE_SEED,
    device: str = "cuda",
    public_observation: bool = True,
    n_full: int = 64,
    n_fast: int = 16,
    p_full: float = 0.25,
    lazy_interior_chance: bool = True,
    max_decisions: int = 600,
    max_depth: int = 80,
    temperature_move_fraction: float = 0.075,
    temperature_high: float = 1.0,
    temperature_low: float = 0.0,
    prior_temperature: float = 1.0,
    value_scale: float = 1.0,
    c_visit: float = 50.0,
    c_scale: float = 0.03,
    track: str = "2p_no_trade",
    vps_to_win: int = 10,
    obs_width: int = 806,
    shard_size: int = 2048,
    fmt: str = "npz_zst",
    commit_secs: float = 240.0,
    resume: bool = False,
) -> None:
    """Full generation wave. GATED: pilot numbers + team-lead go. Cap 100 L4s.

    Spawn-based (stragglers never block): fires all parts and exits. Poll with
    ::summarize; re-run with resume=True to fill failed/missing parts.
    """
    if containers > 100:
        raise ValueError(f"containers={containers} exceeds the hard cap of 100 L4 GPUs.")
    _verify_disjoint_seeds(base_seed, containers, games_per_container)
    run_id = f"{run_name.replace('/', '_')}-{uuid.uuid4().hex[:12]}"
    payloads = _payloads(
        run_name=run_name,
        run_id=run_id,
        checkpoint_rel=checkpoint_rel,
        containers=containers,
        games_per_container=games_per_container,
        base_seed=base_seed,
        device=device,
        public_observation=public_observation,
        n_full=n_full,
        n_fast=n_fast,
        p_full=p_full,
        lazy_interior_chance=lazy_interior_chance,
        max_decisions=max_decisions,
        max_depth=max_depth,
        temperature_move_fraction=temperature_move_fraction,
        temperature_high=temperature_high,
        temperature_low=temperature_low,
        prior_temperature=prior_temperature,
        value_scale=value_scale,
        c_visit=c_visit,
        c_scale=c_scale,
        track=track,
        vps_to_win=vps_to_win,
        obs_width=obs_width,
        correct_rust_chance_spectra=True,
        shard_size=shard_size,
        fmt=fmt,
        commit_secs=commit_secs,
        resume=resume,
    )
    print(
        json.dumps(
            {
                "progress": "gen_spawn_launch",
                "run_name": run_name,
                "run_id": run_id,
                "containers": containers,
                "games_target": containers * games_per_container,
                "gpu": "L4",
                "public_observation": public_observation,
                "base_seed": base_seed,
                "volume": VOLUME_NAME,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    call_ids = []
    for payload in payloads:
        call = gpu_part_worker.spawn(payload)
        call_ids.append(call.object_id)
    print(
        json.dumps(
            {
                "progress": "gen_spawn_complete",
                "run_name": run_name,
                "run_id": run_id,
                "function_call_ids": call_ids,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@app.local_entrypoint()
def summarize(run_name: str, run_id: str = "") -> None:
    """Print aggregate progress for a run (works mid-wave)."""
    print(json.dumps(summarize_run.remote(run_name, run_id), indent=2, sort_keys=True))
