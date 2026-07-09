"""LEGACY / STALE: superseded by `tools/modal_gumbel_factory_gpu.py`.

This CPU-only factory predates the task #71/#76 public-observation (hidden-info
leak) fix: it has NO `public_observation` knob at all (its
`EntityGraphRustEvaluatorConfig` is always constructed with the default
`public_observation=False`), so pointing it at a masked-trained checkpoint
(e.g. any v3a/v3b arm) silently generates UNMASKED, hidden-info-leaked
training data -- or fails the `_assert_public_observation_matches_checkpoint_
training` safety net deep in evaluator construction with a confusing error
far from this file. It also predates `lazy_interior_chance` and the
corrected `c_scale=0.03` search calibration wired into the GPU factory. Both
`launch_gumbel_pilot` and `launch_gumbel_gen` below refuse to run unless
called with `i_know_this_is_the_legacy_cpu_factory=True` -- use
`modal_gumbel_factory_gpu.py` instead unless you have a specific, understood
reason to run the CPU path (e.g. a HeuristicRustEvaluator smoke test with no
checkpoint at all, where the masking gap doesn't apply).

Modal CPU self-play factory for Gumbel + true-chance-node MCTS generation (gen-2+).

Runs `catan_zero.rl.gumbel_self_play.run_worker_games` on a fleet of 8-core CPU
containers: each container plays its share of games with 8 single-threaded
worker PROCESSES (spawn), each owning a dynamic-INT8-quantized
`EntityGraphRustEvaluator` on device="cpu", and writes standard gumbel shard
trees (`worker_XXX/` dirs with shards + manifest.json) to a Modal volume. The
inner format IS the GPU hosts' format, so `tools/build_gumbel_gen_manifest.py`
consumes a downloaded run unchanged.

Design provenance: optimizer task #45 benchmarks + #51 design (2026-07-03).
Measured basis (B200 Xeon 8592+, 8 physical cores, 8-way contention):
torch-int8 forward ≈ 86ms/leaf-eval all-in, ~234 evals/decision mid-game,
~232 decisions/game -> ~6 games/hr/container; INT8 drift: max prob delta
0.001, argmax stable. ONNX-Runtime int8 (34ms/eval solo) is the documented
+80% follow-up (`evaluator_mode="ort_int8"` raises until built).

Mirrors `tools/modal_teacher_factory.py` / `tools/modal_ppo_factory.py`:
same image recipe (+ torch-cpu + the catanatron_rs 0.1.2 manylinux wheel),
`Volume.from_name(create_if_missing=True)`, run_id-stamped part manifests for
resume, periodic `volume.commit()` so preemption loses minutes not hours, and
a spawn-based launcher for full waves (stragglers never block the harvest).

Volume layout (volume `catan-zero-gumbel-data`):
    /data/checkpoints/<name>/checkpoint.pt      <- `modal volume put` from B200
    /data/<run_name>/parts/part_XXXXX/          <- one container's output
        worker_000/ ... worker_007/             <- run_worker_games shard trees
        manifest.json                           <- container summary (run_id-stamped)

Operating procedure (Modal is authenticated ONLY on the B200; run from
/home/ubuntu/catan-zero so `add_local_dir` picks up the repo):

  1. Upload the generation seed checkpoint (once per generation):
       .venv/bin/modal volume put catan-zero-gumbel-data \
           runs/bc/<...>/checkpoint.pt checkpoints/<gen_name>/checkpoint.pt
  2. MANDATORY $4 pilot (4 containers x 8 games) BEFORE any full wave:
       .venv/bin/modal run tools/modal_gumbel_factory.py::launch_gumbel_pilot \
           --run-name gumbel_pilot_v1 --checkpoint-rel checkpoints/<gen_name>/checkpoint.pt
     Blocks until done; prints per-part reports + aggregate games/hr.
  3. Full wave (GATED: requires team-lead go on the pilot numbers):
       .venv/bin/modal run tools/modal_gumbel_factory.py::launch_gumbel_gen \
           --run-name gumbel_gen2_v1 --checkpoint-rel ... --containers 400 \
           --games-per-container 48
     Spawns and exits; poll with ::summarize, then download:
       .venv/bin/modal volume get catan-zero-gumbel-data <run_name> <local_dir>
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import modal


APP_NAME = "catan-zero-gumbel-factory"
VOLUME_NAME = "catan-zero-gumbel-data"
REMOTE_ROOT = Path("/root/catan-zero")
VOLUME_ROOT = Path("/data")

# The compiled pyo3 engine wheel (manylinux_2_34, cp311). Lives on the B200;
# baked into the image at deploy time. Rebuilding the wheel? Update BOTH names.
WHEEL_NAME = "catanatron_rs-0.1.2-cp311-cp311-manylinux_2_34_x86_64.whl"
LOCAL_WHEEL_PATH = f"/tmp/catanatron_rs_wheels/{WHEEL_NAME}"

# Fleet-disjoint seed space: GPU hosts use base-seed offsets 1 / 1000001 /
# 2000001 (gen-1 runbook), Modal owns 3000001+.
DEFAULT_BASE_SEED = 3_000_001


def _refuse_unless_legacy_cpu_factory_acknowledged(acknowledged: bool) -> None:
    """Hard startup guard: this CPU factory has no `public_observation` knob
    (see the module docstring), so it's an armed footgun against any
    masked-trained checkpoint. Refuse to run unless the caller explicitly
    opts in with `i_know_this_is_the_legacy_cpu_factory=True`."""
    if acknowledged:
        return
    raise SystemExit(
        "modal_gumbel_factory.py is the LEGACY CPU-only factory: it has no "
        "public_observation knob and will silently generate unmasked, "
        "hidden-info-leaked data against a masked-trained checkpoint (or fail "
        "a confusing safety-net assertion far from here). Use "
        "tools/modal_gumbel_factory_gpu.py instead. If you have a specific, "
        "understood reason to run this legacy CPU path anyway, pass "
        "i_know_this_is_the_legacy_cpu_factory=True."
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "networkx>=3.0", "gymnasium>=1.0", "zstandard")
    # CPU wheel index: the default PyPI torch drags the full CUDA stack into
    # a CPU-only container image.
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cpu")
    # modal in-image guarantees the container runtime's client deps (grpclib,
    # protobuf<7) live in site-packages; without it the injected /pkg client
    # crash-looped on `import grpclib` (pilot_v1 postmortem). ort_int8
    # follow-up: onnxruntime MUST be co-installed with "protobuf<7" in the
    # SAME pip_install or it upgrades protobuf to 7.x and re-breaks this.
    .pip_install("modal==1.5.1")
    .add_local_file(LOCAL_WHEEL_PATH, f"/root/wheels/{WHEEL_NAME}", copy=True)
    .run_commands(f"pip install /root/wheels/{WHEEL_NAME}")
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


# ------------------------------------------------------------- child process
def _tune_worker_threads() -> None:
    """Pin torch intra/inter-op threads to 1 (8 processes own the 8 cores)."""
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:  # noqa: BLE001 - already set in this process
            pass
    except Exception:  # noqa: BLE001 - best effort
        pass


def _maybe_quantize_int8(policy: Any) -> bool:
    """Dynamic-INT8 quantize the loaded policy's Linear layers in place. Fail-open.

    Same contract as `modal_ppo_factory._maybe_quantize_rollout`: on ANY
    failure log + return False and the worker keeps fp32 (correct, ~40%
    slower). Measured drift on the 35M entity_graph checkpoint: max logit
    delta 0.086, max prob delta 0.001, value delta 0.001, argmax stable.
    """
    try:
        import torch

        model = getattr(policy, "model", None)
        if model is None:
            return False
        policy.model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return True
    except Exception as exc:  # noqa: BLE001 - fail-open: stay fp32
        print(json.dumps({"event": "quantize_failed", "error": repr(exc)}), flush=True)
        return False


def _run_gumbel_worker(worker_args: dict[str, Any]) -> dict[str, Any]:
    """One inner worker process: quantized CPU evaluator + run_worker_games.

    Top-level and picklable (ProcessPoolExecutor spawn ctx). NEVER raises:
    mirrors `generate_gumbel_selfplay_data._worker_entry` so one dead worker
    can't lose its siblings' already-written shards from the part manifest.
    """
    worker_index = int(worker_args.get("worker_index", -1))
    try:
        _tune_worker_threads()

        from catan_zero.rl.gumbel_self_play import (
            GumbelSelfPlayConfig,
            run_worker_games,
        )
        from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig
        from catan_zero.search.neural_rust_mcts import (
            EntityGraphRustEvaluator,
            EntityGraphRustEvaluatorConfig,
        )

        evaluator_mode = str(worker_args.get("evaluator_mode", "torch_int8"))
        if evaluator_mode == "ort_int8":
            # Documented follow-up (+80% per bench): OrtEntityGraphEvaluator
            # with per-batch-size static ONNX exports. Not built yet.
            raise NotImplementedError(
                "evaluator_mode='ort_int8' is a follow-up; use torch_int8 or torch_fp32"
            )
        if evaluator_mode not in ("torch_int8", "torch_fp32"):
            raise ValueError(f"unknown evaluator_mode: {evaluator_mode!r}")

        # Plain (non-Batched) evaluator on purpose: a single-threaded worker
        # gains nothing from the batch queue/thread, and evaluate_many already
        # batches each ~11-child chance fan-out into one padded forward.
        evaluator = EntityGraphRustEvaluator.from_checkpoint(
            worker_args["checkpoint"],
            device="cpu",
            config=EntityGraphRustEvaluatorConfig(
                value_scale=float(worker_args["value_scale"]),
                prior_temperature=float(worker_args["prior_temperature"]),
            ),
        )
        quantized = False
        if evaluator_mode == "torch_int8":
            quantized = _maybe_quantize_int8(evaluator.policy)

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
        )
        summary["worker_index"] = worker_index
        summary["evaluator_mode"] = evaluator_mode
        summary["quantized"] = quantized
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
            "evaluator_mode": str(worker_args.get("evaluator_mode", "")),
            "quantized": False,
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
    cpu=8,
    memory=16_384,  # measured: 1.0-1.4GB RSS per worker x 8 + headroom
    max_containers=400,
    timeout=21_600,
    retries=2,
)
def gumbel_part_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Play one part's games: 8 single-threaded CPU workers + periodic commits."""
    os.chdir(REMOTE_ROOT)

    run_name = str(payload["run_name"])
    run_id = str(payload.get("run_id", ""))
    part_index = int(payload["part_index"])
    games = int(payload["games"])
    cpu_workers = max(1, int(payload.get("cpu_workers", 8)))
    commit_secs = max(30.0, float(payload.get("commit_secs", 240.0)))
    resume = bool(payload.get("resume", False))

    part_dir = VOLUME_ROOT / run_name / "parts" / f"part_{part_index:05d}"
    manifest_path = part_dir / "manifest.json"

    # Resume / retry / stale-output protocol:
    #   - COMPLETE manifest from our run_id (or ANY run_id under resume=True,
    #     the fill-the-stragglers relaunch) -> return it, play nothing.
    #   - partial output from our OWN run_id (Modal retry after a crash /
    #     preemption; the .run_id marker is committed before any game) -> wipe
    #     and replay this part.
    #   - anything else without resume -> refuse; it's another run's data.
    marker_path = part_dir / ".run_id"
    if part_dir.exists():
        if manifest_path.exists():
            complete = json.loads(manifest_path.read_text(encoding="utf-8"))
            if resume or (run_id and str(complete.get("run_id", "")) == run_id):
                return complete
        own_partial = marker_path.exists() and marker_path.read_text(
            encoding="utf-8"
        ).strip() == run_id
        if resume or own_partial:
            shutil.rmtree(part_dir)
        elif any(part_dir.iterdir()):
            raise RuntimeError(
                f"{part_dir} already contains output from a different run_id; "
                "use a fresh run_name or pass resume=True."
            )
    part_dir.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(run_id, encoding="utf-8")
    volume.commit()

    # Stage the checkpoint on container-local disk once: 8 spawned readers
    # hammering the volume mount for the same 140MB is pure waste.
    volume_checkpoint = VOLUME_ROOT / str(payload["checkpoint_rel"])
    if not volume_checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not on volume: {volume_checkpoint} "
            f"(modal volume put {VOLUME_NAME} <local.pt> {payload['checkpoint_rel']})"
        )
    local_checkpoint = Path("/tmp/gumbel_checkpoint.pt")
    shutil.copyfile(volume_checkpoint, local_checkpoint)

    # Split games across inner workers exactly like generate_gumbel_selfplay_data.
    games_per_worker = [
        games // cpu_workers + (1 if i < games % cpu_workers else 0)
        for i in range(cpu_workers)
    ]
    base_seed = int(payload["base_seed"])
    game_index_start = int(payload["game_index_start"])
    worker_args: list[dict[str, Any]] = []
    offset = game_index_start
    for worker_index, worker_games in enumerate(games_per_worker):
        if worker_games <= 0:
            continue
        fleet_ordinal = part_index * cpu_workers + worker_index
        worker_args.append(
            {
                "worker_index": worker_index,
                "games": worker_games,
                "game_index_start": offset,
                "out_dir": str(part_dir / f"worker_{worker_index:03d}"),
                "checkpoint": str(local_checkpoint),
                "evaluator_mode": str(payload.get("evaluator_mode", "torch_int8")),
                "base_seed": base_seed,
                # Decorrelate search RNG across the whole fleet, not per part.
                "worker_seed": base_seed + 0x9E3779B9 * (fleet_ordinal + 1),
                "n_full": int(payload["n_full"]),
                "n_fast": int(payload["n_fast"]),
                "p_full": float(payload["p_full"]),
                "c_visit": float(payload["c_visit"]),
                "c_scale": float(payload["c_scale"]),
                "max_decisions": int(payload["max_decisions"]),
                "max_depth": int(payload["max_depth"]),
                "temperature_move_fraction": float(payload["temperature_move_fraction"]),
                "temperature_high": float(payload["temperature_high"]),
                "temperature_low": float(payload["temperature_low"]),
                "prior_temperature": float(payload["prior_temperature"]),
                "value_scale": float(payload["value_scale"]),
                "track": str(payload["track"]),
                "vps_to_win": int(payload["vps_to_win"]),
                "obs_width": int(payload["obs_width"]),
                "correct_rust_chance_spectra": bool(payload["correct_rust_chance_spectra"]),
                "shard_size": int(payload["shard_size"]),
                "fmt": str(payload["fmt"]),
            }
        )
        offset += worker_games

    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "part_start",
                "run_name": run_name,
                "run_id": run_id,
                "part_index": part_index,
                "games": games,
                "cpu_workers": len(worker_args),
                "evaluator_mode": str(payload.get("evaluator_mode", "torch_int8")),
            }
        ),
        flush=True,
    )

    # torch forbids forking a process with torch imported -> spawn.
    mp_context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=mp_context) as pool:
        pending = {pool.submit(_run_gumbel_worker, args) for args in worker_args}
        last_commit = time.perf_counter()
        while pending:
            done, pending = wait(pending, timeout=commit_secs, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())  # never raises by contract
            # Periodic commit bounds preemption loss to ~commit_secs of shards.
            # Shard files are written whole and the per-worker manifest is
            # atomic, so a torn mid-write commit self-heals on the next one.
            if done or (time.perf_counter() - last_commit) >= commit_secs:
                volume.commit()
                last_commit = time.perf_counter()

    results.sort(key=lambda summary: int(summary.get("worker_index", 0)))
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
        "evaluator_mode": str(payload.get("evaluator_mode", "torch_int8")),
        "quantized_workers": sum(1 for r in results if r.get("quantized")),
        "cpu_workers": len(worker_args),
        "base_seed": base_seed,
        "game_index_start": game_index_start,
        "elapsed_sec": elapsed,
        "games_per_hour": games_completed / max(elapsed / 3600.0, 1e-9),
        # Part-RELATIVE shard paths make this manifest a valid --gen-input for
        # tools/build_gumbel_gen_manifest.py after `modal volume get` (its
        # relative-to-manifest fallback resolves "worker_00i/shard.npz.zst"),
        # so a wave merges with one input per part instead of one per worker.
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
def _payloads(
    *,
    run_name: str,
    run_id: str,
    checkpoint_rel: str,
    containers: int,
    games_per_container: int,
    cpu_workers: int,
    base_seed: int,
    evaluator_mode: str,
    n_full: int,
    n_fast: int,
    p_full: float,
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
            "cpu_workers": cpu_workers,
            "base_seed": base_seed,
            "evaluator_mode": evaluator_mode,
            "n_full": n_full,
            "n_fast": n_fast,
            "p_full": p_full,
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
def launch_gumbel_pilot(
    run_name: str = "gumbel_modal_pilot_v1",
    checkpoint_rel: str = "checkpoints/gen1_seed/checkpoint.pt",
    containers: int = 4,
    games_per_container: int = 8,
    cpu_workers: int = 8,
    base_seed: int = DEFAULT_BASE_SEED,
    evaluator_mode: str = "torch_int8",
    n_full: int = 64,
    n_fast: int = 16,
    p_full: float = 0.25,
    # Defaults below MUST match the GumbelSelfPlayConfig / GumbelChanceMCTSConfig
    # dataclass defaults (the single source of truth) -- enforced by
    # tests/test_cli_config_drift.py. max_decisions=600 + temperature_move_fraction
    # =0.075 is the adopted cap-600 policy (coupled: 0.075 * 600 == 45 temperature
    # moves); c_scale=0.03 is the banked production winner (mctx value_scale-style
    # rescale; raw-Q c_scale=1.0 was the verified near-one-hot-target trap, F1a/F1b).
    max_decisions: int = 600,
    max_depth: int = 80,
    temperature_move_fraction: float = 0.075,
    temperature_high: float = 1.0,
    temperature_low: float = 0.0,
    prior_temperature: float = 1.0,
    value_scale: float = 1.0,
    c_visit: float = 50.0,
    c_scale: float = 0.1,  # matches GumbelChanceMCTSConfig.c_scale (mctx value_scale) per test_cli_config_drift; prod overrides to the banked 0.03 explicitly. (BUG-2 set 0.03 here but that drifts from the canonical dataclass; legacy path stays guarded.)
    track: str = "2p_no_trade",
    vps_to_win: int = 10,
    obs_width: int = 806,
    shard_size: int = 2048,
    fmt: str = "npz_zst",
    commit_secs: float = 240.0,
    resume: bool = False,
    i_know_this_is_the_legacy_cpu_factory: bool = False,
) -> None:
    """The mandatory ~$4 pilot: 4 containers x 8 games, BLOCKS until done.

    Purpose: calibrate real Modal-CPU games/hr vs the B200 bench, validate
    the shard trees merge cleanly, and inspect forced fraction / errors —
    BEFORE any full wave (full waves are gated on team-lead sign-off).
    """
    _refuse_unless_legacy_cpu_factory_acknowledged(i_know_this_is_the_legacy_cpu_factory)
    run_id = f"{run_name}-{uuid.uuid4().hex[:12]}"
    payloads = _payloads(
        run_name=run_name,
        run_id=run_id,
        checkpoint_rel=checkpoint_rel,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        base_seed=base_seed,
        evaluator_mode=evaluator_mode,
        n_full=n_full,
        n_fast=n_fast,
        p_full=p_full,
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
                "evaluator_mode": evaluator_mode,
                "volume": VOLUME_NAME,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for report in gumbel_part_worker.map(payloads, order_outputs=False):
        print(
            json.dumps(
                {
                    "progress": "pilot_part_done",
                    "part_index": report["part_index"],
                    "games": report["games_completed"],
                    "rows": report["rows"],
                    "elapsed_sec": round(report["elapsed_sec"], 1),
                    "games_per_hour": round(report["games_per_hour"], 2),
                    "quantized_workers": report["quantized_workers"],
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
def launch_gumbel_gen(
    run_name: str,
    checkpoint_rel: str,
    containers: int = 400,
    games_per_container: int = 48,
    cpu_workers: int = 8,
    base_seed: int = DEFAULT_BASE_SEED,
    evaluator_mode: str = "torch_int8",
    n_full: int = 64,
    n_fast: int = 16,
    p_full: float = 0.25,
    # Defaults below MUST match the GumbelSelfPlayConfig / GumbelChanceMCTSConfig
    # dataclass defaults (the single source of truth) -- enforced by
    # tests/test_cli_config_drift.py. max_decisions=600 + temperature_move_fraction
    # =0.075 is the adopted cap-600 policy (coupled: 0.075 * 600 == 45 temperature
    # moves); c_scale=0.1 is mctx's value_scale (raw-Q c_scale=1.0 was the verified
    # near-one-hot-target trap, F1a/F1b).
    max_decisions: int = 600,
    max_depth: int = 80,
    temperature_move_fraction: float = 0.075,
    temperature_high: float = 1.0,
    temperature_low: float = 0.0,
    prior_temperature: float = 1.0,
    value_scale: float = 1.0,
    c_visit: float = 50.0,
    c_scale: float = 0.1,  # matches GumbelChanceMCTSConfig.c_scale (mctx value_scale) per test_cli_config_drift; prod overrides to the banked 0.03 explicitly. (BUG-2 set 0.03 here but that drifts from the canonical dataclass; legacy path stays guarded.)
    track: str = "2p_no_trade",
    vps_to_win: int = 10,
    obs_width: int = 806,
    shard_size: int = 2048,
    fmt: str = "npz_zst",
    commit_secs: float = 240.0,
    resume: bool = False,
    i_know_this_is_the_legacy_cpu_factory: bool = False,
) -> None:
    """Full generation wave. GATED: pilot numbers + explicit team-lead go first.

    Spawn-based (stragglers never block): fires all parts and exits. Poll with
    ::summarize_run; re-run with resume=True to fill failed/missing parts
    (completed run_id-matching parts are skipped, not replayed). Harvest at
    >=95% parts and cancel leftovers via the printed function_call_ids.
    """
    _refuse_unless_legacy_cpu_factory_acknowledged(i_know_this_is_the_legacy_cpu_factory)
    run_id = f"{run_name}-{uuid.uuid4().hex[:12]}"
    payloads = _payloads(
        run_name=run_name,
        run_id=run_id,
        checkpoint_rel=checkpoint_rel,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        base_seed=base_seed,
        evaluator_mode=evaluator_mode,
        n_full=n_full,
        n_fast=n_fast,
        p_full=p_full,
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
                "cpu_per_container": cpu_workers,
                "games_target": containers * games_per_container,
                "evaluator_mode": evaluator_mode,
                "base_seed": base_seed,
                "volume": VOLUME_NAME,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    call_ids = []
    for payload in payloads:
        call = gumbel_part_worker.spawn(payload)
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
