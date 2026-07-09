from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import modal


APP_NAME = "catan-zero-teacher-factory"
VOLUME_NAME = "catan-zero-teacher-data"
REMOTE_ROOT = Path("/root/catan-zero")
VOLUME_ROOT = Path("/data")

DEFAULT_TEACHERS = (
    "catanatron_ab4,catanatron_ab5,value_rollout_search,"
    "catanatron_ab3,catanatron_value,jsettlers_lite"
)
DEFAULT_TEACHER_SAMPLING_WEIGHTS = (
    "catanatron_ab5=3.0,catanatron_ab4=2.5,value_rollout_search=2.5,"
    "catanatron_ab3=1.2,catanatron_value=0.6,jsettlers_lite=0.7"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "networkx>=3.0",
        "gymnasium>=1.0",
        "zstandard",
        "modal>=1.0",
        "protobuf>=4.25",
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'tools'}",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
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


def _run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from generate_teacher_data import ShardWriter, _generate_chunk

    run_name = str(payload["run_name"])
    run_id = str(payload.get("run_id", ""))
    part_index = int(payload["part_index"])
    games = int(payload["games"])
    seed = int(payload["seed"])
    cpu_workers = int(payload["cpu_workers"])
    teachers = [name.strip() for name in str(payload["teachers"]).split(",") if name.strip()]
    out_dir = VOLUME_ROOT / run_name / "parts" / f"part_{part_index:05d}"
    if out_dir.exists() and bool(payload.get("resume", False)):
        complete_manifest = out_dir / "manifest.json"
        if complete_manifest.exists():
            complete = json.loads(complete_manifest.read_text(encoding="utf-8"))
            if run_id and str(complete.get("run_id", "")) == run_id:
                return complete
        shutil.rmtree(out_dir)
    if out_dir.exists() and not bool(payload.get("resume", False)):
        existing = list(out_dir.iterdir())
        if existing:
            complete_manifest = out_dir / "manifest.json"
            partial_manifest = out_dir / "manifest.partial.json"
            if complete_manifest.exists():
                complete = json.loads(complete_manifest.read_text(encoding="utf-8"))
                if run_id and str(complete.get("run_id", "")) == run_id:
                    return complete
            if partial_manifest.exists():
                partial = json.loads(partial_manifest.read_text(encoding="utf-8"))
                if run_id and str(partial.get("run_id", "")) == run_id:
                    shutil.rmtree(out_dir)
                else:
                    raise RuntimeError(
                        f"{out_dir} already exists for a different run_id; use a fresh "
                        "run_name or pass resume=True explicitly."
                    )
            else:
                raise RuntimeError(
                    f"{out_dir} already exists and is not empty; use a fresh run_name "
                    "or pass resume=True explicitly."
                )
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    writer = ShardWriter(out_dir, int(payload["shard_size"]), str(payload["fmt"]))
    decisions = 0
    wins = 0
    completed_games = 0
    teacher_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    score_source_counts: Counter[str] = Counter()
    forced_actions = 0
    invalid_labels = 0
    soft_policy_rows = 0
    soft_score_rows = 0
    final_public_vp_rows = 0
    final_actual_vp_rows = 0
    outcome_rows = 0
    clean_terminal_outcome_rows = 0
    truncated_rows = 0
    legal_counts: list[int] = []
    completed_chunks = 0
    commit_every_chunks = max(1, int(payload.get("commit_every_chunks", 8)))

    chunk_payloads = [
        {
            "start": idx,
            "end": min(games, idx + int(payload["chunk_games"])),
            "track": payload["track"],
            "vps_to_win": payload["vps_to_win"],
            "teachers": teachers,
            "seed": seed,
            "max_decisions": payload["max_decisions"],
            "mixed_seats": bool(payload.get("mixed_seats", False)),
            "mixed_seat_mode": str(payload.get("mixed_seat_mode", "random")),
            "teacher_sampling_weights": str(payload.get("teacher_sampling_weights", "")),
            "graph_history_features": bool(payload.get("graph_history_features", False)),
        }
        for idx in range(0, games, int(payload["chunk_games"]))
    ]

    with ProcessPoolExecutor(max_workers=cpu_workers) as executor:
        futures = [executor.submit(_generate_chunk, chunk) for chunk in chunk_payloads]
        for future in as_completed(futures):
            result = future.result()
            completed_chunks += 1
            wins += int(result["wins"])
            completed_games += int(result["games"])
            for row in result["rows"]:
                valid = np.asarray(row["valid"], dtype=np.int16)
                legal_count = int(np.sum(valid >= 0))
                action = int(row["action"])
                decisions += 1
                teacher_counts[str(row["teacher"])] += 1
                phase_counts[str(row.get("phase", "")) or "unknown"] += 1
                score_source_counts[str(row.get("target_score_source", "")) or "none"] += 1
                forced_actions += int(legal_count <= 1)
                if action not in set(map(int, valid[valid >= 0])):
                    invalid_labels += 1
                    raise ValueError(
                        f"invalid teacher action {action} in {run_name} part={part_index} "
                        f"seed={row.get('seed')} player={row.get('player')} "
                        f"teacher={row.get('teacher')} phase={row.get('phase')} "
                        f"legal_count={legal_count}"
                    )
                policy = np.asarray(row.get("target_policy", ()), dtype=np.float32)
                scores = np.asarray(row.get("target_scores", ()), dtype=np.float32)
                soft_policy_rows += int(
                    policy.size
                    and np.sum(np.where(np.isfinite(policy), np.maximum(policy, 0.0), 0.0)) > 0.0
                )
                soft_score_rows += int(scores.size and np.isfinite(scores).any())
                final_public_vp_rows += int(bool(row.get("has_final_public_vps", False)))
                final_actual_vp_rows += int(bool(row.get("has_final_actual_vps", False)))
                truncated_rows += int(bool(row.get("truncated", False)))
                outcome_rows += int(bool(row.get("winner", "")))
                clean_terminal_outcome_rows += int(
                    bool(row.get("winner", "")) and not bool(row.get("truncated", False))
                )
                legal_counts.append(legal_count)
                writer.add_row(row)
            if completed_chunks % commit_every_chunks == 0:
                writer.flush()
                _write_part_manifest(
                    out_dir / "manifest.partial.json",
                    _part_report(
                        run_name=run_name,
                        run_id=run_id,
                        part_index=part_index,
                        payload=payload,
                        teachers=teachers,
                        games=games,
                        completed_games=completed_games,
                        wins=wins,
                        decisions=decisions,
                        elapsed=time.perf_counter() - start,
                        teacher_counts=teacher_counts,
                        phase_counts=phase_counts,
                        score_source_counts=score_source_counts,
                        forced_actions=forced_actions,
                        invalid_labels=invalid_labels,
                        soft_policy_rows=soft_policy_rows,
                        soft_score_rows=soft_score_rows,
                        final_public_vp_rows=final_public_vp_rows,
                        final_actual_vp_rows=final_actual_vp_rows,
                        outcome_rows=outcome_rows,
                        clean_terminal_outcome_rows=clean_terminal_outcome_rows,
                        truncated_rows=truncated_rows,
                        legal_counts=legal_counts,
                        shards=writer.paths,
                        complete=False,
                    ),
                )
                volume.commit()

    shards = writer.close()
    elapsed = time.perf_counter() - start
    report = _part_report(
        run_name=run_name,
        run_id=run_id,
        part_index=part_index,
        payload=payload,
        teachers=teachers,
        games=games,
        completed_games=completed_games,
        wins=wins,
        decisions=decisions,
        elapsed=elapsed,
        teacher_counts=teacher_counts,
        phase_counts=phase_counts,
        score_source_counts=score_source_counts,
        forced_actions=forced_actions,
        invalid_labels=invalid_labels,
        soft_policy_rows=soft_policy_rows,
        soft_score_rows=soft_score_rows,
        final_public_vp_rows=final_public_vp_rows,
        final_actual_vp_rows=final_actual_vp_rows,
        outcome_rows=outcome_rows,
        clean_terminal_outcome_rows=clean_terminal_outcome_rows,
        truncated_rows=truncated_rows,
        legal_counts=legal_counts,
        shards=shards,
        complete=True,
    )
    _write_part_manifest(out_dir / "manifest.json", report)
    volume.commit()
    return report


def _part_report(
    *,
    run_name: str,
    run_id: str,
    part_index: int,
    payload: dict[str, Any],
    teachers: list[str],
    games: int,
    completed_games: int,
    wins: int,
    decisions: int,
    elapsed: float,
    teacher_counts: Counter[str],
    phase_counts: Counter[str],
    score_source_counts: Counter[str],
    forced_actions: int,
    invalid_labels: int,
    soft_policy_rows: int,
    soft_score_rows: int,
    final_public_vp_rows: int,
    final_actual_vp_rows: int,
    outcome_rows: int,
    clean_terminal_outcome_rows: int,
    truncated_rows: int,
    legal_counts: list[int],
    shards: list[Path],
    complete: bool,
) -> dict[str, Any]:
    import numpy as np

    legal = np.asarray(legal_counts, dtype=np.int64) if legal_counts else np.asarray([0])
    return {
        "run_name": run_name,
        "run_id": run_id,
        "part_index": part_index,
        "track": payload["track"],
        "vps_to_win": int(payload["vps_to_win"]),
        "teachers": teachers,
        "games": games,
        "completed_games": completed_games,
        "wins": wins,
        "samples": decisions,
        "cpu_workers": int(payload["cpu_workers"]),
        "format": payload["fmt"],
        "mixed_seats": bool(payload.get("mixed_seats", False)),
        "mixed_seat_mode": str(payload.get("mixed_seat_mode", "")),
        "graph_history_features": bool(payload.get("graph_history_features", False)),
        "teacher_sampling_weights": str(payload.get("teacher_sampling_weights", "")),
        "tool_provenance": _tool_provenance(),
        "shards": [str(path) for path in shards],
        "elapsed_sec": elapsed,
        "complete": bool(complete),
        "games_per_sec": completed_games / elapsed if elapsed > 0 else 0.0,
        "samples_per_sec": decisions / elapsed if elapsed > 0 else 0.0,
        "teacher_counts": dict(teacher_counts.most_common()),
        "phase_counts": dict(phase_counts.most_common()),
        "score_source_counts": dict(score_source_counts.most_common()),
        "forced_actions": forced_actions,
        "forced_action_fraction": forced_actions / decisions if decisions else 0.0,
        "soft_policy_rows": soft_policy_rows,
        "soft_policy_fraction": soft_policy_rows / decisions if decisions else 0.0,
        "soft_score_rows": soft_score_rows,
        "soft_score_fraction": soft_score_rows / decisions if decisions else 0.0,
        "final_public_vp_rows": final_public_vp_rows,
        "final_public_vp_fraction": final_public_vp_rows / decisions if decisions else 0.0,
        "final_actual_vp_rows": final_actual_vp_rows,
        "final_actual_vp_fraction": final_actual_vp_rows / decisions if decisions else 0.0,
        "outcome_rows": outcome_rows,
        "outcome_fraction": outcome_rows / decisions if decisions else 0.0,
        "clean_terminal_outcome_rows": clean_terminal_outcome_rows,
        "clean_terminal_outcome_fraction": (
            clean_terminal_outcome_rows / decisions if decisions else 0.0
        ),
        "truncated_rows": truncated_rows,
        "truncated_fraction": truncated_rows / decisions if decisions else 0.0,
        "invalid_teacher_actions": invalid_labels,
        "legal_actions": {
            "mean": float(np.mean(legal)),
            "p50": int(np.percentile(legal, 50)),
            "p90": int(np.percentile(legal, 90)),
            "p99": int(np.percentile(legal, 99)),
            "max": int(np.max(legal)),
        },
    }


def _tool_provenance() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    files = [
        "tools/modal_teacher_factory.py",
        "tools/generate_teacher_data.py",
        "catan_rules_v1.json",
        "src/catan_zero/rules.py",
        "src/catan_zero/rl/action_mask.py",
        "src/catan_zero/rl/multiagent_env.py",
        "src/catan_zero/rl/self_play.py",
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/xdim_lite_policy.py",
        "src/catan_zero/rl/policy_pool.py",
        "tools/factory_common.py",
    ]
    hashes = {}
    for name in files:
        path = repo_root / name
        try:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return {
        "file_sha256": hashes,
        "feature_semantics_files": [
            "catan_rules_v1.json",
            "src/catan_zero/rules.py",
            "src/catan_zero/rl/action_mask.py",
            "src/catan_zero/rl/multiagent_env.py",
            "src/catan_zero/rl/self_play.py",
            "src/catan_zero/rl/action_features.py",
            "src/catan_zero/rl/xdim_lite_policy.py",
            "src/catan_zero/rl/policy_pool.py",
        ],
    }


def _write_part_manifest(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=8,
    memory=8_192,
    max_containers=75,
    timeout=21_600,
)
def rollout_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    return _run_worker(payload)


@app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=300)
def summarize_run(run_name: str, run_id: str = "") -> dict[str, Any]:
    run_dir = VOLUME_ROOT / run_name / "parts"
    volume.reload()
    manifests = sorted(run_dir.glob("part_*/manifest.json"))
    partial_manifests = sorted(run_dir.glob("part_*/manifest.partial.json"))
    parts = []
    for path in manifests:
        part = json.loads(path.read_text(encoding="utf-8"))
        if run_id and str(part.get("run_id", "")) != str(run_id):
            continue
        parts.append(part)
    complete_part_dirs = set()
    for path in manifests:
        part = json.loads(path.read_text(encoding="utf-8"))
        if run_id and str(part.get("run_id", "")) != str(run_id):
            continue
        complete_part_dirs.add(path.parent.name)
    partial_parts = []
    for path in partial_manifests:
        if path.parent.name in complete_part_dirs:
            continue
        part = json.loads(path.read_text(encoding="utf-8"))
        if run_id and str(part.get("run_id", "")) != str(run_id):
            continue
        partial_parts.append(part)
    total_games = sum(int(part.get("completed_games", 0)) for part in parts)
    total_samples = sum(int(part.get("samples", 0)) for part in parts)
    partial_games = sum(int(part.get("completed_games", 0)) for part in partial_parts)
    partial_samples = sum(int(part.get("samples", 0)) for part in partial_parts)
    partial_invalid = sum(int(part.get("invalid_teacher_actions", 0)) for part in partial_parts)
    partial_soft_policy = sum(int(part.get("soft_policy_rows", 0)) for part in partial_parts)
    partial_soft_scores = sum(int(part.get("soft_score_rows", 0)) for part in partial_parts)
    partial_final_public_vps = sum(int(part.get("final_public_vp_rows", 0)) for part in partial_parts)
    partial_final_actual_vps = sum(int(part.get("final_actual_vp_rows", 0)) for part in partial_parts)
    partial_truncated = sum(int(part.get("truncated_rows", 0)) for part in partial_parts)
    total_invalid = sum(int(part.get("invalid_teacher_actions", 0)) for part in parts)
    total_forced = sum(int(part.get("forced_actions", 0)) for part in parts)
    total_soft_policy = sum(int(part.get("soft_policy_rows", 0)) for part in parts)
    total_soft_scores = sum(int(part.get("soft_score_rows", 0)) for part in parts)
    total_final_public_vps = sum(int(part.get("final_public_vp_rows", 0)) for part in parts)
    total_final_actual_vps = sum(int(part.get("final_actual_vp_rows", 0)) for part in parts)
    total_truncated = sum(int(part.get("truncated_rows", 0)) for part in parts)
    mixed_seats = sorted(
        {bool(part.get("mixed_seats")) for part in parts if "mixed_seats" in part}
    )
    mixed_seat_modes = sorted(
        {
            str(part.get("mixed_seat_mode"))
            for part in parts
            if str(part.get("mixed_seat_mode", ""))
        }
    )
    graph_history_features = sorted(
        {bool(part.get("graph_history_features")) for part in parts if "graph_history_features" in part}
    )
    teacher_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    score_source_counts: Counter[str] = Counter()
    for part in parts:
        teacher_counts.update(part.get("teacher_counts", {}))
        phase_counts.update(part.get("phase_counts", {}))
        score_source_counts.update(part.get("score_source_counts", {}))
    partial_teacher_counts: Counter[str] = Counter()
    partial_phase_counts: Counter[str] = Counter()
    partial_score_source_counts: Counter[str] = Counter()
    partial_mixed_seats = sorted(
        {bool(part.get("mixed_seats")) for part in partial_parts if "mixed_seats" in part}
    )
    partial_mixed_seat_modes = sorted(
        {
            str(part.get("mixed_seat_mode"))
            for part in partial_parts
            if str(part.get("mixed_seat_mode", ""))
        }
    )
    partial_graph_history_features = sorted(
        {
            bool(part.get("graph_history_features"))
            for part in partial_parts
            if "graph_history_features" in part
        }
    )
    for part in partial_parts:
        partial_teacher_counts.update(part.get("teacher_counts", {}))
        partial_phase_counts.update(part.get("phase_counts", {}))
        partial_score_source_counts.update(part.get("score_source_counts", {}))
    return {
        "run_name": run_name,
        "run_id": run_id,
        "parts_complete": len(parts),
        "parts_partial": len(partial_parts),
        "games": total_games,
        "samples": total_samples,
        "partial_games": partial_games,
        "partial_samples": partial_samples,
        "partial_invalid_teacher_actions": partial_invalid,
        "partial_soft_policy_fraction": (
            partial_soft_policy / partial_samples if partial_samples else 0.0
        ),
        "partial_soft_score_fraction": (
            partial_soft_scores / partial_samples if partial_samples else 0.0
        ),
        "partial_final_public_vp_fraction": (
            partial_final_public_vps / partial_samples if partial_samples else 0.0
        ),
        "partial_final_actual_vp_fraction": (
            partial_final_actual_vps / partial_samples if partial_samples else 0.0
        ),
        "partial_truncated_fraction": partial_truncated / partial_samples if partial_samples else 0.0,
        "partial_teacher_counts": dict(partial_teacher_counts.most_common()),
        "partial_phase_counts": dict(partial_phase_counts.most_common()),
        "partial_score_source_counts": dict(partial_score_source_counts.most_common()),
        "partial_mixed_seats": partial_mixed_seats,
        "partial_mixed_seat_modes": partial_mixed_seat_modes,
        "partial_graph_history_features": partial_graph_history_features,
        "observed_games": total_games + partial_games,
        "observed_samples": total_samples + partial_samples,
        "invalid_teacher_actions": total_invalid,
        "forced_action_fraction": total_forced / total_samples if total_samples else 0.0,
        "soft_policy_fraction": total_soft_policy / total_samples if total_samples else 0.0,
        "soft_score_fraction": total_soft_scores / total_samples if total_samples else 0.0,
        "final_public_vp_fraction": (
            total_final_public_vps / total_samples if total_samples else 0.0
        ),
        "final_actual_vp_fraction": (
            total_final_actual_vps / total_samples if total_samples else 0.0
        ),
        "truncated_fraction": total_truncated / total_samples if total_samples else 0.0,
        "teacher_counts": dict(teacher_counts.most_common()),
        "phase_counts": dict(phase_counts.most_common()),
        "score_source_counts": dict(score_source_counts.most_common()),
        "mixed_seats": mixed_seats,
        "mixed_seat_modes": mixed_seat_modes,
        "graph_history_features": graph_history_features,
        "volume_path": str(VOLUME_ROOT / run_name),
    }


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=8,
    memory=16_384,
    max_containers=75,
    timeout=21_600,
)
def curate_part_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    volume.reload()
    input_runs = [item.strip() for item in str(payload["input_runs"]).split(",") if item.strip()]
    output_run = str(payload["output_run"])
    part_index = int(payload["part_index"])
    out_dir = VOLUME_ROOT / output_run / "parts" / f"part_{part_index:05d}"
    if out_dir.exists():
        if bool(payload.get("resume", False)) and (out_dir / "manifest.json").exists():
            return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_part_dirs = []
    for run_name in input_runs:
        part_dir = VOLUME_ROOT / run_name / "parts" / f"part_{part_index:05d}"
        if list(part_dir.glob("*.npz")) or list(part_dir.glob("*.npz.zst")):
            input_part_dirs.append(part_dir)

    if not input_part_dirs:
        manifest = {
            "output_run": output_run,
            "part_index": part_index,
            "input_runs": input_runs,
            "input_part_dirs": [],
            "raw_samples": 0,
            "kept_samples": 0,
            "empty": True,
        }
        _write_part_manifest(out_dir / "manifest.json", manifest)
        volume.commit()
        return manifest

    cmd = [
        sys.executable,
        str(REMOTE_ROOT / "tools" / "curate_teacher_data.py"),
        "--out",
        str(out_dir),
        "--format",
        str(payload.get("fmt", "npz_zst")),
        "--shard-size",
        str(int(payload.get("shard_size", 100_000))),
        "--seed",
        str(int(payload.get("seed", 1)) + part_index),
        "--production-35m-teacher",
        "--dedupe-keys",
        str(payload.get("dedupe_keys", "exact")),
        "--progress-every",
        str(int(payload.get("progress_every", 250_000))),
    ]
    for part_dir in input_part_dirs:
        cmd.extend(["--data", str(part_dir)])

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REMOTE_ROOT, text=True, capture_output=True)
    if proc.stdout:
        (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.stderr:
        (out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        volume.commit()
        raise RuntimeError(
            f"curation part {part_index} failed with {proc.returncode}; "
            f"stderr tail={proc.stderr[-1000:]}"
        )
    report_path = out_dir / "curation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update(
        {
            "part_index": part_index,
            "input_runs": input_runs,
            "input_part_dirs": [str(path) for path in input_part_dirs],
            "output_run": output_run,
            "elapsed_wall_sec": time.perf_counter() - started,
        }
    )
    _write_part_manifest(out_dir / "manifest.json", report)
    volume.commit()
    return report


@app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=300)
def summarize_curated_run(output_run: str) -> dict[str, Any]:
    volume.reload()
    run_dir = VOLUME_ROOT / output_run / "parts"
    manifests = sorted(run_dir.glob("part_*/manifest.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    nonempty = [report for report in reports if not report.get("empty")]
    return {
        "output_run": output_run,
        "volume_path": str(VOLUME_ROOT / output_run),
        "parts": len(reports),
        "nonempty_parts": len(nonempty),
        "raw_samples": sum(int(report.get("raw_samples", 0)) for report in nonempty),
        "kept_samples": sum(int(report.get("kept_samples", 0)) for report in nonempty),
        "dropped_invalid": sum(int(report.get("dropped_invalid", 0)) for report in nonempty),
        "dropped_truncated": sum(int(report.get("dropped_truncated", 0)) for report in nonempty),
        "dropped_duplicate": sum(int(report.get("dropped_duplicate", 0)) for report in nonempty),
        "kept_value_only_samples": sum(
            int(report.get("kept_value_only_samples", 0)) for report in nonempty
        ),
    }


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    cpu=8,
    memory=16_384,
    max_containers=75,
    timeout=21_600,
)
def entity_convert_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    volume.reload()
    input_runs = [item.strip() for item in str(payload["input_runs"]).split(",") if item.strip()]
    output_run = str(payload["output_run"])
    part_index = int(payload["part_index"])
    partition_count = int(payload["partition_count"])
    out_dir = VOLUME_ROOT / output_run / "parts" / f"part_{part_index:05d}"
    if out_dir.exists():
        if bool(payload.get("resume", False)) and (out_dir / "manifest.json").exists():
            return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REMOTE_ROOT / "tools" / "convert_teacher_to_entity_tokens.py"),
        "--out",
        str(out_dir),
        "--track",
        str(payload.get("track", "2p_no_trade")),
        "--vps-to-win",
        str(int(payload.get("vps_to_win", 10))),
        "--format",
        str(payload.get("fmt", "npz_zst")),
        "--shard-size",
        str(int(payload.get("shard_size", 100_000))),
        "--partition-count",
        str(partition_count),
        "--partition-index",
        str(part_index),
        "--progress-every",
        str(int(payload.get("progress_every", 100_000))),
    ]
    if bool(payload.get("skip_duplicate_conflicts", True)):
        cmd.append("--skip-duplicate-conflicts")
    if bool(payload.get("graph_history_features", True)):
        cmd.append("--graph-history-features")
    for run_name in input_runs:
        cmd.extend(["--data", str(VOLUME_ROOT / run_name)])
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REMOTE_ROOT, text=True, capture_output=True)
    if proc.stdout:
        (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.stderr:
        (out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        volume.commit()
        cmd_text = " ".join(cmd)
        stdout_tail = proc.stdout[-2000:] if proc.stdout else ""
        stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
        raise RuntimeError(
            f"entity conversion part {part_index} failed with {proc.returncode}; "
            f"cmd={cmd_text}; stdout tail={stdout_tail}; stderr tail={stderr_tail}"
        )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "part_index": part_index,
            "partition_count": partition_count,
            "input_runs": input_runs,
            "output_run": output_run,
            "elapsed_wall_sec": time.perf_counter() - started,
        }
    )
    _write_part_manifest(out_dir / "manifest.json", manifest)
    volume.commit()
    return manifest


@app.local_entrypoint()
def debug_entity_convert_part(
    input_runs: str,
    output_run: str = "entity_teacher_debug",
    part_index: int = 0,
    partition_count: int = 1000,
    track: str = "2p_no_trade",
    vps_to_win: int = 10,
    fmt: str = "npz_zst",
    shard_size: int = 50_000,
    graph_history_features: bool = True,
) -> None:
    report = entity_convert_worker.remote(
        {
            "input_runs": input_runs,
            "output_run": output_run,
            "part_index": int(part_index),
            "partition_count": int(partition_count),
            "track": track,
            "vps_to_win": int(vps_to_win),
            "fmt": fmt,
            "shard_size": int(shard_size),
            "resume": False,
            "graph_history_features": bool(graph_history_features),
            "progress_every": 10_000,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


@app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=300)
def summarize_entity_conversion(output_run: str) -> dict[str, Any]:
    volume.reload()
    run_dir = VOLUME_ROOT / output_run / "parts"
    manifests = sorted(run_dir.glob("part_*/manifest.json"))
    parts = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    return {
        "output_run": output_run,
        "volume_path": str(VOLUME_ROOT / output_run),
        "parts": len(parts),
        "converted_rows": sum(int(part.get("converted_rows", 0)) for part in parts),
        "loaded_rows": sum(int(part.get("loaded_rows", 0)) for part in parts),
        "converted_seeds": sum(int(part.get("converted_seeds", 0)) for part in parts),
        "mismatch_parts": sum(1 for part in parts if part.get("mismatches")),
        "duplicate_decision_rows": sum(int(part.get("duplicate_decision_rows", 0)) for part in parts),
        "rows_per_sec_sum": sum(float(part.get("rows_per_sec", 0.0)) for part in parts),
    }


def _payloads(
    *,
    run_name: str,
    containers: int,
    games_per_container: int,
    cpu_workers: int,
    seed: int,
    teachers: str,
    track: str,
    vps_to_win: int,
    max_decisions: int,
    fmt: str,
    shard_size: int,
    chunk_games: int,
    mixed_seats: bool = False,
    mixed_seat_mode: str = "random",
    teacher_sampling_weights: str = "",
    run_id: str = "",
    resume: bool = False,
    commit_every_chunks: int = 8,
    graph_history_features: bool = True,
) -> list[dict[str, Any]]:
    return [
        {
            "run_name": run_name,
            "run_id": run_id,
            "part_index": index,
            "games": games_per_container,
            "cpu_workers": cpu_workers,
            "seed": seed + index * games_per_container,
            "teachers": teachers,
            "track": track,
            "vps_to_win": vps_to_win,
            "max_decisions": max_decisions,
            "fmt": fmt,
            "shard_size": shard_size,
            "chunk_games": chunk_games,
            "mixed_seats": mixed_seats,
            "mixed_seat_mode": mixed_seat_mode,
            "teacher_sampling_weights": teacher_sampling_weights,
            "resume": resume,
            "commit_every_chunks": commit_every_chunks,
            "graph_history_features": graph_history_features,
        }
        for index in range(containers)
    ]


def _launch(
    *,
    run_name: str,
    containers: int,
    games_per_container: int,
    cpu_workers: int,
    seed: int,
    teachers: str,
    track: str,
    vps_to_win: int,
    max_decisions: int,
    fmt: str,
    shard_size: int,
    chunk_games: int,
    mixed_seats: bool = False,
    mixed_seat_mode: str = "random",
    teacher_sampling_weights: str = "",
    resume: bool = False,
    commit_every_chunks: int = 8,
    graph_history_features: bool = True,
) -> None:
    started = time.perf_counter()
    run_id = f"{run_name}-{uuid.uuid4().hex[:12]}"
    payloads = _payloads(
        run_name=run_name,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        seed=seed,
        teachers=teachers,
        track=track,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        fmt=fmt,
        shard_size=shard_size,
        chunk_games=chunk_games,
        mixed_seats=mixed_seats,
        mixed_seat_mode=mixed_seat_mode,
        teacher_sampling_weights=teacher_sampling_weights,
        run_id=run_id,
        resume=resume,
        commit_every_chunks=commit_every_chunks,
        graph_history_features=graph_history_features,
    )
    print(
        json.dumps(
            {
                "progress": "modal_launch",
                "run_name": run_name,
                "run_id": run_id,
                "containers": containers,
                "cpu_per_container": 8,
                "cpu_workers_per_container": cpu_workers,
                "max_physical_cpus": containers * 8,
                "target_games": containers * games_per_container,
                "teachers": teachers,
                "mixed_seats": mixed_seats,
                "mixed_seat_mode": mixed_seat_mode,
                "teacher_sampling_weights": teacher_sampling_weights,
                "commit_every_chunks": commit_every_chunks,
                "graph_history_features": graph_history_features,
                "volume": VOLUME_NAME,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    reports = []
    for report in rollout_worker.map(payloads, order_outputs=False):
        reports.append(report)
        print(
            json.dumps(
                {
                    "progress": "modal_part_done",
                    "run_name": run_name,
                    "parts_done": len(reports),
                    "parts_total": containers,
                    "part_index": report["part_index"],
                    "games": report["completed_games"],
                    "samples": report["samples"],
                    "invalid": report["invalid_teacher_actions"],
                    "elapsed_sec": report["elapsed_sec"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = summarize_run.remote(run_name, run_id)
    summary["elapsed_sec"] = time.perf_counter() - started
    print(json.dumps({"progress": "modal_complete", **summary}, indent=2, sort_keys=True))


def _launch_spawn(
    *,
    run_name: str,
    containers: int,
    games_per_container: int,
    cpu_workers: int,
    seed: int,
    teachers: str,
    track: str,
    vps_to_win: int,
    max_decisions: int,
    fmt: str,
    shard_size: int,
    chunk_games: int,
    mixed_seats: bool = False,
    mixed_seat_mode: str = "random",
    teacher_sampling_weights: str = "",
    resume: bool = False,
    commit_every_chunks: int = 8,
    graph_history_features: bool = True,
) -> None:
    run_id = f"{run_name}-{uuid.uuid4().hex[:12]}"
    payloads = _payloads(
        run_name=run_name,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        seed=seed,
        teachers=teachers,
        track=track,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        fmt=fmt,
        shard_size=shard_size,
        chunk_games=chunk_games,
        mixed_seats=mixed_seats,
        mixed_seat_mode=mixed_seat_mode,
        teacher_sampling_weights=teacher_sampling_weights,
        run_id=run_id,
        resume=resume,
        commit_every_chunks=commit_every_chunks,
        graph_history_features=graph_history_features,
    )
    print(
        json.dumps(
            {
                "progress": "modal_spawn_launch",
                "run_name": run_name,
                "run_id": run_id,
                "containers": containers,
                "cpu_per_container": 8,
                "cpu_workers_per_container": cpu_workers,
                "max_physical_cpus": containers * 8,
                "target_games": containers * games_per_container,
                "teachers": teachers,
                "mixed_seats": mixed_seats,
                "mixed_seat_mode": mixed_seat_mode,
                "teacher_sampling_weights": teacher_sampling_weights,
                "commit_every_chunks": commit_every_chunks,
                "graph_history_features": graph_history_features,
                "volume": VOLUME_NAME,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    call_ids = []
    for payload in payloads:
        call = rollout_worker.spawn(payload)
        call_ids.append(call.object_id)
        print(
            json.dumps(
                {
                    "progress": "modal_part_spawned",
                    "run_name": run_name,
                    "run_id": run_id,
                    "part_index": payload["part_index"],
                    "function_call_id": call.object_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "progress": "modal_spawn_complete",
                "run_name": run_name,
                "run_id": run_id,
                "function_call_ids": call_ids,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@app.local_entrypoint()
def smoke(
    run_name: str = "modal_smoke",
    containers: int = 5,
    games_per_container: int = 32,
    cpu_workers: int = 8,
    seed: int = 60628650,
    teachers: str = DEFAULT_TEACHERS,
    teacher_sampling_weights: str = DEFAULT_TEACHER_SAMPLING_WEIGHTS,
    fmt: str = "npz_zst",
    mixed_seats: bool = True,
    mixed_seat_mode: str = "random",
    resume: bool = False,
    graph_history_features: bool = True,
) -> None:
    _launch(
        run_name=run_name,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        seed=seed,
        teachers=teachers,
        track="2p_no_trade",
        vps_to_win=10,
        max_decisions=1200,
        fmt=fmt,
        shard_size=50_000,
        chunk_games=1,
        mixed_seats=mixed_seats,
        mixed_seat_mode=mixed_seat_mode,
        teacher_sampling_weights=teacher_sampling_weights,
        resume=resume,
        graph_history_features=graph_history_features,
    )


@app.local_entrypoint()
def launch_600(
    run_name: str = "teacher_2p10_searchmix_600cpu_v1",
    containers: int = 75,
    games_per_container: int = 256,
    cpu_workers: int = 8,
    seed: int = 60628700,
    teachers: str = DEFAULT_TEACHERS,
    teacher_sampling_weights: str = DEFAULT_TEACHER_SAMPLING_WEIGHTS,
    fmt: str = "npz_zst",
    mixed_seats: bool = True,
    mixed_seat_mode: str = "random",
    resume: bool = False,
    commit_every_chunks: int = 8,
    graph_history_features: bool = True,
) -> None:
    _launch(
        run_name=run_name,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        seed=seed,
        teachers=teachers,
        track="2p_no_trade",
        vps_to_win=10,
        max_decisions=1200,
        fmt=fmt,
        shard_size=50_000,
        chunk_games=1,
        mixed_seats=mixed_seats,
        mixed_seat_mode=mixed_seat_mode,
        teacher_sampling_weights=teacher_sampling_weights,
        resume=resume,
        commit_every_chunks=commit_every_chunks,
        graph_history_features=graph_history_features,
    )


@app.local_entrypoint()
def launch_600_ab45(
    run_name: str = "teacher_2p10_ab45_search_600cpu_v1",
    containers: int = 75,
    games_per_container: int = 128,
    cpu_workers: int = 8,
    seed: int = 60629700,
    teachers: str = (
        "catanatron_ab4,catanatron_ab5,value_rollout_search,"
        "catanatron_ab3,catanatron_value,jsettlers_lite"
    ),
    teacher_sampling_weights: str = DEFAULT_TEACHER_SAMPLING_WEIGHTS,
    fmt: str = "npz_zst",
    mixed_seats: bool = True,
    mixed_seat_mode: str = "random",
    resume: bool = False,
    commit_every_chunks: int = 8,
    graph_history_features: bool = True,
) -> None:
    _launch(
        run_name=run_name,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        seed=seed,
        teachers=teachers,
        track="2p_no_trade",
        vps_to_win=10,
        max_decisions=1200,
        fmt=fmt,
        shard_size=50_000,
        chunk_games=1,
        mixed_seats=mixed_seats,
        mixed_seat_mode=mixed_seat_mode,
        teacher_sampling_weights=teacher_sampling_weights,
        resume=resume,
        commit_every_chunks=commit_every_chunks,
        graph_history_features=graph_history_features,
    )


@app.local_entrypoint()
def launch_600_ab45_spawn(
    run_name: str = "teacher_2p10_ab45_search_600cpu_spawn_v1",
    containers: int = 75,
    games_per_container: int = 128,
    cpu_workers: int = 8,
    seed: int = 60629700,
    teachers: str = (
        "catanatron_ab4,catanatron_ab5,value_rollout_search,"
        "catanatron_ab3,catanatron_value,jsettlers_lite"
    ),
    teacher_sampling_weights: str = DEFAULT_TEACHER_SAMPLING_WEIGHTS,
    fmt: str = "npz_zst",
    mixed_seats: bool = True,
    mixed_seat_mode: str = "random",
    resume: bool = False,
    commit_every_chunks: int = 8,
    graph_history_features: bool = True,
) -> None:
    _launch_spawn(
        run_name=run_name,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        seed=seed,
        teachers=teachers,
        track="2p_no_trade",
        vps_to_win=10,
        max_decisions=1200,
        fmt=fmt,
        shard_size=50_000,
        chunk_games=1,
        mixed_seats=mixed_seats,
        mixed_seat_mode=mixed_seat_mode,
        teacher_sampling_weights=teacher_sampling_weights,
        resume=resume,
        commit_every_chunks=commit_every_chunks,
        graph_history_features=graph_history_features,
    )


@app.local_entrypoint()
def launch_600_4p(
    run_name: str = "teacher_4p_trade_softmix_600cpu_v1",
    containers: int = 75,
    games_per_container: int = 128,
    cpu_workers: int = 8,
    seed: int = 60630700,
    teachers: str = DEFAULT_TEACHERS,
    teacher_sampling_weights: str = DEFAULT_TEACHER_SAMPLING_WEIGHTS,
    track: str = "4p_bank_trade",
    fmt: str = "npz_zst",
    mixed_seats: bool = True,
    mixed_seat_mode: str = "random",
    resume: bool = False,
    commit_every_chunks: int = 8,
    graph_history_features: bool = True,
) -> None:
    _launch(
        run_name=run_name,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        seed=seed,
        teachers=teachers,
        track=track,
        vps_to_win=10,
        max_decisions=1600,
        fmt=fmt,
        shard_size=50_000,
        chunk_games=1,
        mixed_seats=mixed_seats,
        mixed_seat_mode=mixed_seat_mode,
        teacher_sampling_weights=teacher_sampling_weights,
        resume=resume,
        commit_every_chunks=commit_every_chunks,
        graph_history_features=graph_history_features,
    )


@app.local_entrypoint()
def launch_600_edge_search(
    run_name: str = "teacher_2p10_edge_search_600cpu_v1",
    containers: int = 75,
    games_per_container: int = 64,
    cpu_workers: int = 8,
    seed: int = 60632900,
    teachers: str = (
        "catanatron_mcts100,catanatron_greedy25,catanatron_sab4,"
        "catanatron_ab5,value_rollout_search,jsettlers_lite"
    ),
    teacher_sampling_weights: str = (
        "catanatron_mcts100=1.2,catanatron_greedy25=1.0,"
        "catanatron_sab4=1.3,catanatron_ab5=2.0,"
        "value_rollout_search=2.0,jsettlers_lite=0.6"
    ),
    fmt: str = "npz_zst",
    mixed_seats: bool = True,
    mixed_seat_mode: str = "random",
    resume: bool = False,
    commit_every_chunks: int = 4,
    graph_history_features: bool = True,
) -> None:
    _launch(
        run_name=run_name,
        containers=containers,
        games_per_container=games_per_container,
        cpu_workers=cpu_workers,
        seed=seed,
        teachers=teachers,
        track="2p_no_trade",
        vps_to_win=10,
        max_decisions=1200,
        fmt=fmt,
        shard_size=50_000,
        chunk_games=1,
        mixed_seats=mixed_seats,
        mixed_seat_mode=mixed_seat_mode,
        teacher_sampling_weights=teacher_sampling_weights,
        resume=resume,
        commit_every_chunks=commit_every_chunks,
        graph_history_features=graph_history_features,
    )


@app.local_entrypoint()
def status(run_name: str = "teacher_2p10_searchmix_600cpu_v1", run_id: str = "") -> None:
    print(json.dumps(summarize_run.remote(run_name, run_id), indent=2, sort_keys=True))


@app.local_entrypoint()
def launch_curate_75(
    input_runs: str,
    output_run: str = "curated_teacher_2p10_hq_v1",
    containers: int = 75,
    fmt: str = "npz_zst",
    shard_size: int = 100_000,
    seed: int = 90210,
    resume: bool = False,
    dedupe_keys: str = "exact",
) -> None:
    print(
        json.dumps(
            {
                "progress": "modal_curate_launch",
                "input_runs": input_runs,
                "output_run": output_run,
                "containers": containers,
                "max_physical_cpus": int(containers) * 8,
                "production_35m_teacher": True,
                "dedupe_keys": dedupe_keys,
                "volume": VOLUME_NAME,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    payloads = [
        {
            "input_runs": input_runs,
            "output_run": output_run,
            "part_index": index,
            "fmt": fmt,
            "shard_size": int(shard_size),
            "seed": int(seed),
            "resume": bool(resume),
            "dedupe_keys": dedupe_keys,
        }
        for index in range(int(containers))
    ]
    reports = []
    for report in curate_part_worker.map(payloads, order_outputs=False):
        reports.append(report)
        print(
            json.dumps(
                {
                    "progress": "modal_curate_part_done",
                    "parts_done": len(reports),
                    "parts_total": int(containers),
                    "part_index": report.get("part_index"),
                    "raw_samples": int(report.get("raw_samples", 0)),
                    "kept_samples": int(report.get("kept_samples", 0)),
                    "dropped_invalid": int(report.get("dropped_invalid", 0)),
                    "dropped_truncated": int(report.get("dropped_truncated", 0)),
                    "dropped_duplicate": int(report.get("dropped_duplicate", 0)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = summarize_curated_run.remote(output_run)
    print(
        json.dumps(
            {
                "progress": "modal_curate_complete",
                **summary,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


@app.local_entrypoint()
def status_curated(output_run: str = "curated_teacher_2p10_hq_v1") -> None:
    print(json.dumps(summarize_curated_run.remote(output_run), indent=2, sort_keys=True))


@app.local_entrypoint()
def launch_entity_convert_75(
    input_runs: str,
    output_run: str = "entity_teacher_2p10_hq_v1",
    containers: int = 75,
    track: str = "2p_no_trade",
    vps_to_win: int = 10,
    fmt: str = "npz_zst",
    shard_size: int = 100_000,
    resume: bool = False,
    graph_history_features: bool = True,
) -> None:
    print(
        json.dumps(
            {
                "progress": "modal_entity_convert_launch",
                "input_runs": input_runs,
                "output_run": output_run,
                "partitions": containers,
                "max_active_containers": min(int(containers), 75),
                "max_physical_cpus": min(int(containers), 75) * 8,
                "track": track,
                "vps_to_win": int(vps_to_win),
                "volume": VOLUME_NAME,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    payloads = [
        {
            "input_runs": input_runs,
            "output_run": output_run,
            "part_index": index,
            "partition_count": int(containers),
            "track": track,
            "vps_to_win": int(vps_to_win),
            "fmt": fmt,
            "shard_size": int(shard_size),
            "resume": bool(resume),
            "graph_history_features": bool(graph_history_features),
        }
        for index in range(int(containers))
    ]
    reports = []
    for report in entity_convert_worker.map(payloads, order_outputs=False):
        reports.append(report)
        print(
            json.dumps(
                {
                    "progress": "modal_entity_convert_part_done",
                    "parts_done": len(reports),
                    "parts_total": int(containers),
                    "part_index": report.get("part_index"),
                    "loaded_rows": int(report.get("loaded_rows", 0)),
                    "converted_rows": int(report.get("converted_rows", 0)),
                    "converted_seeds": int(report.get("converted_seeds", 0)),
                    "mismatches": len(report.get("mismatches", [])),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = summarize_entity_conversion.remote(output_run)
    print(
        json.dumps(
            {
                "progress": "modal_entity_convert_complete",
                **summary,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


@app.local_entrypoint()
def status_entity_conversion(output_run: str = "entity_teacher_2p10_hq_v1") -> None:
    print(json.dumps(summarize_entity_conversion.remote(output_run), indent=2, sort_keys=True))
