#!/usr/bin/env python3
"""Seal and run a sample-matched 8-B200 microbatch-quality comparison.

The historical batch probe compared local batches at equal row exposure but
left ``lr_warmup_steps`` fixed.  Because its global batch changed, the larger
batch traversed twice as many warmup *rows* per optimizer step.  This tool
removes that confound without changing the learner: local 512 with two
gradient-accumulation microsteps is compared with local 1024 with one.  Both
arms therefore have global batch 8192, the same optimizer steps, LR schedule,
warmup row dose, and total row dose.

The input is an already-reviewed diagnostic training command, normally the
production-next K0 composite command.  Planning binds the current Git/trainer
runtime, composite descriptor, warm-start checkpoint, and exact argv.  Runs
are diagnostic-only and never promotion eligible.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import a1_b200_batch_probe as batch_probe  # noqa: E402


SCHEMA = "a1-b200-microbatch-quality-plan-v1"
RUN_SCHEMA = "a1-b200-microbatch-quality-run-v1"
WORLD_SIZE = 8
DEFAULT_OPTIMIZER_STEPS = 512
ARMS = (
    ("micro512-accum2", 512, 2),
    ("micro1024-accum1", 1024, 1),
)


class QualityProbeError(RuntimeError):
    """The requested comparison is not matched or not sealed."""


def _value(command: Sequence[str], flag: str) -> str:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise QualityProbeError(f"base command must contain exactly one {flag}")
    return str(command[positions[0] + 1])


def _set(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) > 1:
        raise QualityProbeError(f"base command repeats {flag}")
    if positions:
        command[positions[0] + 1] = value
    else:
        command.extend([flag, value])


def _runtime() -> dict[str, str]:
    runtime = batch_probe._current_runtime()  # noqa: SLF001
    tool = Path(__file__).resolve(strict=True)
    return {
        **runtime,
        "quality_probe": str(tool),
        "quality_probe_sha256": batch_probe._file_sha(tool),  # noqa: SLF001
    }


def _load_base_command(path: Path) -> list[str]:
    source = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualityProbeError(f"cannot load base command: {error}") from error
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) for item in value
    ):
        raise QualityProbeError("base command JSON must be a non-empty string list")
    return list(value)


def _input_binding(command: Sequence[str]) -> dict[str, str]:
    data_format = _value(command, "--data-format")
    if data_format != "memmap":
        raise QualityProbeError("quality comparison requires memmap data")
    descriptor = Path(_value(command, "--data")).expanduser().resolve(strict=True)
    try:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualityProbeError(f"cannot load composite descriptor: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in {
            "memmap_composite_v1",
            "memmap_composite_v2",
        }
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
    ):
        raise QualityProbeError("data is not a diagnostic-only memmap composite")
    checkpoint = Path(_value(command, "--init-checkpoint")).expanduser().resolve(
        strict=True
    )
    return {
        "data": str(descriptor),
        "data_sha256": batch_probe._file_sha(descriptor),  # noqa: SLF001
        "init_checkpoint": str(checkpoint),
        "init_checkpoint_sha256": batch_probe._file_sha(checkpoint),  # noqa: SLF001
    }


def build_plan(
    *,
    base_command_json: Path,
    output_dir: Path,
    optimizer_steps: int = DEFAULT_OPTIMIZER_STEPS,
) -> dict[str, Any]:
    if optimizer_steps <= 100:
        raise QualityProbeError("optimizer steps must exceed the complete warmup")
    runtime = _runtime()
    base = _load_base_command(base_command_json)
    if "torch.distributed.run" not in base or "--nproc-per-node=8" not in base:
        raise QualityProbeError("base command must use torch.distributed.run on 8 ranks")
    if any(item.startswith("--a1-batch-probe") for item in base):
        raise QualityProbeError("base command must not inherit historical batch authority")
    trainer_positions = [
        index for index, item in enumerate(base) if Path(item).name == "train_bc.py"
    ]
    if len(trainer_positions) != 1:
        raise QualityProbeError("base command must name exactly one train_bc.py")
    base[trainer_positions[0]] = runtime["trainer"]
    if int(_value(base, "--lr-warmup-steps")) != 100:
        raise QualityProbeError("matched comparison is sealed to a 100-step warmup")
    if _value(base, "--lr-schedule") != "flat":
        raise QualityProbeError("matched comparison is sealed to the flat LR schedule")
    inputs = _input_binding(base)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise QualityProbeError(f"refusing existing output directory: {output_dir}")

    runs: list[dict[str, Any]] = []
    global_batches: set[int] = set()
    for run_id, local_batch, accumulation in ARMS:
        command = list(base)
        run_dir = output_dir / run_id
        for flag, value in (
            ("--batch-size", str(local_batch)),
            ("--grad-accum-steps", str(accumulation)),
            ("--max-steps", str(optimizer_steps)),
            ("--epochs", "1"),
            ("--train-diagnostics-every-batches", "1"),
            ("--checkpoint", str(run_dir / "candidate.pt")),
            ("--report", str(run_dir / "train.report.json")),
        ):
            _set(command, flag, value)
        global_batch = WORLD_SIZE * local_batch * accumulation
        global_batches.add(global_batch)
        runs.append(
            {
                "run_id": run_id,
                "local_batch_size": local_batch,
                "grad_accum_steps": accumulation,
                "world_size": WORLD_SIZE,
                "global_batch_size": global_batch,
                "lr_warmup_steps": 100,
                "warmup_samples": 100 * global_batch,
                "max_steps": optimizer_steps,
                "planned_samples": optimizer_steps * global_batch,
                "run_dir": str(run_dir),
                "command": command,
                "command_sha256": batch_probe._digest(command),  # noqa: SLF001
            }
        )
    if global_batches != {8192}:
        raise AssertionError("microbatch arms do not have identical global batch")
    plan: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": True,
        "purpose": (
            "Choose the per-GPU microbatch systems knee without changing global "
            "batch, LR trajectory, optimizer steps, warmup samples, or total samples."
        ),
        "runtime": runtime,
        "inputs": inputs,
        "matched_invariants": {
            "global_batch_size": 8192,
            "lr": float(_value(base, "--lr")),
            "lr_schedule": "flat",
            "lr_warmup_steps": 100,
            "warmup_samples": 819_200,
            "optimizer_steps": optimizer_steps,
            "planned_samples": optimizer_steps * 8192,
            "seed": int(_value(base, "--seed")),
        },
        "only_intended_drift": ["batch_size", "grad_accum_steps"],
        "adjudication": {
            "primary": [
                "samples_per_second",
                "active_policy_teacher_gap_closure_per_wall_second",
            ],
            "quality_floor": [
                "active_policy_teacher_gap_closure_per_million_samples",
                "validation_policy_loss",
                "validation_value_loss",
            ],
            "safety": ["clipped_fraction", "preclip_grad_norm_max"],
            "hbm_is_not_an_objective": True,
        },
        "runs": runs,
    }
    plan["plan_sha256"] = batch_probe._digest(plan)  # noqa: SLF001
    output_dir.mkdir(parents=True)
    plan_path = output_dir / "plan.json"
    fd = os.open(plan_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return plan


def _read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    stated = plan.get("plan_sha256")
    actual = batch_probe._digest(  # noqa: SLF001
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("schema_version") != SCHEMA or stated != actual:
        raise QualityProbeError("quality plan schema/digest drift")
    return plan


def _verify(plan: dict[str, Any], run: dict[str, Any]) -> None:
    if plan.get("runtime") != _runtime():
        raise QualityProbeError("quality-probe runtime drift")
    if run.get("command_sha256") != batch_probe._digest(run.get("command")):  # noqa: SLF001
        raise QualityProbeError("quality-probe command drift")
    bindings = plan.get("inputs", {})
    for path_key, sha_key in (
        ("data", "data_sha256"),
        ("init_checkpoint", "init_checkpoint_sha256"),
    ):
        path = Path(str(bindings.get(path_key, ""))).resolve(strict=True)
        if bindings.get(sha_key) != batch_probe._file_sha(path):  # noqa: SLF001
            raise QualityProbeError(f"quality-probe {path_key} bytes drifted")


def run_one(plan_path: Path, run_id: str, *, go: bool) -> dict[str, Any]:
    plan = _read_plan(plan_path)
    matches = [run for run in plan["runs"] if run["run_id"] == run_id]
    if len(matches) != 1:
        raise QualityProbeError(f"unknown run id {run_id!r}")
    run = matches[0]
    _verify(plan, run)
    if not go:
        return {"dry_run": True, **run}
    names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if len(names) != WORLD_SIZE or any("B200" not in name.upper() for name in names):
        raise QualityProbeError(f"--go requires exactly 8 B200 GPUs, got {names}")
    batch_probe._require_no_non_mps_compute()  # noqa: SLF001
    with batch_probe._without_mps():  # noqa: SLF001
        batch_probe._require_no_non_mps_compute()  # noqa: SLF001
        run_dir = Path(run["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=False)
        gpu_file = (run_dir / "gpu.csv").open("w", encoding="utf-8")
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used",
                "--format=csv,nounits",
                "-lms",
                "500",
            ],
            stdout=gpu_file,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        started = time.time_ns()
        completed: subprocess.CompletedProcess[str]
        try:
            with (run_dir / "train.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    run["command"],
                    cwd=Path(plan["runtime"]["repository_root"]),
                    check=False,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        finally:
            finished = time.time_ns()
            monitor.terminate()
            monitor.wait(timeout=10)
            gpu_file.close()
    runtime = {
        "schema_version": RUN_SCHEMA,
        "run_id": run_id,
        "plan_sha256": plan["plan_sha256"],
        "command_sha256": run["command_sha256"],
        "started_unix_ns": started,
        "finished_unix_ns": finished,
        "returncode": completed.returncode,
    }
    (Path(run["run_dir"]) / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise QualityProbeError(
            f"diagnostic run failed with return code {completed.returncode}"
        )
    return batch_probe.summarize(run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--base-command-json", type=Path, required=True)
    plan.add_argument("--out-dir", type=Path, required=True)
    plan.add_argument("--optimizer-steps", type=int, default=DEFAULT_OPTIMIZER_STEPS)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--run-id", choices=[arm[0] for arm in ARMS], required=True)
    run.add_argument("--go", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = build_plan(
                base_command_json=args.base_command_json,
                output_dir=args.out_dir,
                optimizer_steps=args.optimizer_steps,
            )
        elif args.command == "run":
            result = run_one(args.plan, args.run_id, go=args.go)
        else:
            plan = _read_plan(args.plan)
            result = {"runs": [batch_probe.summarize(run) for run in plan["runs"]]}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (QualityProbeError, OSError, ValueError, KeyError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
