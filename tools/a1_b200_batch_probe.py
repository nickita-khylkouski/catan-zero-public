#!/usr/bin/env python3
"""Plan, run, and summarize short 8-B200 batch-efficiency diagnostics.

The probe never uses a promotion executor.  It authenticates a completed n256
midpoint receipt, derives short commands from that exact trainer argv, removes
the production A1 contract flags that correctly forbid topology drift, and
marks every output diagnostic-only.  The underlying corpus/checkpoint bytes and
all objective flags remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import a1_dual_arm_train as dual  # noqa: E402


SCHEMA = "a1-b200-batch-probe-plan-v1"
RUN_SCHEMA = "a1-b200-batch-probe-run-v1"
BATCHES = (512, 768, 1024)
WORLD_SIZE = 8
REFERENCE_LR = 0.00012
REFERENCE_BATCH = 512
STRIP_VALUE_FLAGS = {
    "--validation-game-seed-manifest",
    "--a1-dual-learner-lock",
    "--a1-dual-reviewed-lock-file-sha256",
    "--a1-learner-ablation-id",
    "--a1-effective-learner-recipe-json",
    "--a1-effective-learner-recipe-sha256",
    "--a1-ablation-code-binding-json",
    "--a1-ablation-code-tree-sha256",
    "--a1-reviewed-lock-file-sha256",
    "--a1-curriculum-parent-receipt",
}


class ProbeError(RuntimeError):
    """The requested benchmark is not a matched diagnostic."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _current_runtime() -> dict[str, str]:
    trainer = (_ROOT / "tools" / "train_bc.py").resolve(strict=True)
    commit = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProbeError("probe runtime has no full Git commit")
    dirty = subprocess.run(
        ["git", "-C", str(_ROOT), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if dirty:
        raise ProbeError("probe runtime has tracked modifications")
    return {
        "repository_root": str(_ROOT.resolve()),
        "repository_commit": commit,
        "trainer": str(trainer),
        "trainer_sha256": _file_sha(trainer),
    }


def _bind_current_trainer(command: list[str], runtime: dict[str, str]) -> None:
    indices = [
        index for index, item in enumerate(command) if Path(item).name == "train_bc.py"
    ]
    if len(indices) != 1:
        raise ProbeError("midpoint command does not name exactly one trainer script")
    command[indices[0]] = runtime["trainer"]


def _set_option(command: list[str], flag: str, value: str) -> None:
    indices = [index for index, item in enumerate(command) if item == flag]
    if len(indices) > 1:
        raise ProbeError(f"base command repeats {flag}")
    if indices:
        index = indices[0]
        if index + 1 >= len(command):
            raise ProbeError(f"base command has valueless {flag}")
        command[index + 1] = value
    else:
        command.extend([flag, value])


def _strip_production_authority(command: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item in STRIP_VALUE_FLAGS or item.startswith("--a1-"):
            if index + 1 >= len(command) or command[index + 1].startswith("--"):
                raise ProbeError(f"cannot safely strip valueless production flag {item}")
            index += 2
            continue
        if item in {"--save-each-epoch"}:
            index += 1
            continue
        result.append(item)
        index += 1
    return result


def _scaled_lr(batch: int, policy: str) -> float:
    ratio = batch / REFERENCE_BATCH
    if policy == "fixed":
        return REFERENCE_LR
    if policy == "sqrt":
        return REFERENCE_LR * math.sqrt(ratio)
    if policy == "linear":
        return REFERENCE_LR * ratio
    raise ProbeError(f"unknown LR scaling policy {policy!r}")


def _authenticated_midpoint(path: Path) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve(strict=True)
    receipt = dual.verify_receipt(receipt_path)
    ablation = receipt.get("inputs", {}).get("learner_ablation", {})
    recipe = ablation.get("effective_recipe", {})
    if (
        (receipt.get("arm_id"), receipt.get("subset_id")) != ("n256", "full-56k")
        or ablation.get("ablation_id") != "all-196k-corrective-lr120u-loser1"
        or ablation.get("diagnostic_only") is not True
        or recipe.get("lr") != REFERENCE_LR
        or recipe.get("loser_sample_weight") != 1.0
    ):
        raise ProbeError("midpoint receipt is not the authenticated corrective n256 run")
    command = receipt.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ProbeError("midpoint receipt has no replayable command")
    if "torch.distributed.run" not in command or not any(
        item == "--nproc_per_node=8" for item in command
    ):
        raise ProbeError("midpoint command is not the 8-B200 topology")
    return {"path": str(receipt_path), "sha256": _file_sha(receipt_path), "payload": receipt}


def build_plan(
    *,
    midpoint_receipt: Path,
    output_dir: Path,
    lr_policy: str,
    throughput_steps: int = 24,
    equal_sample_reference_steps: int = 48,
) -> dict[str, Any]:
    if throughput_steps <= 0 or equal_sample_reference_steps <= 0:
        raise ProbeError("step budgets must be positive")
    if equal_sample_reference_steps % 3:
        raise ProbeError("equal-sample reference steps must be divisible by 3")
    midpoint = _authenticated_midpoint(midpoint_receipt)
    runtime = _current_runtime()
    base = _strip_production_authority(list(midpoint["payload"]["command"]))
    _bind_current_trainer(base, runtime)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ProbeError(f"refusing existing output directory: {output_dir}")
    runs: list[dict[str, Any]] = []
    for cohort in ("throughput_fixed_steps", "learning_equal_samples"):
        for batch in BATCHES:
            global_batch = batch * WORLD_SIZE
            steps = (
                throughput_steps
                if cohort == "throughput_fixed_steps"
                else equal_sample_reference_steps * REFERENCE_BATCH // batch
            )
            run_id = f"{cohort}-b{batch}-{lr_policy}"
            run_dir = output_dir / run_id
            command = list(base)
            for flag, value in (
                ("--batch-size", str(batch)),
                ("--max-steps", str(steps)),
                ("--epochs", "1"),
                ("--lr", repr(_scaled_lr(batch, lr_policy))),
                ("--validation-fraction", "0.02"),
                ("--validation-max-samples", "200000"),
                ("--validation-seed", "1701"),
                ("--train-diagnostics-every-batches", "1"),
                ("--checkpoint", str(run_dir / "candidate.pt")),
                ("--report", str(run_dir / "train.report.json")),
            ):
                _set_option(command, flag, value)
            if any(item.startswith("--a1-") for item in command):
                raise AssertionError("production A1 authority leaked into topology probe")
            runs.append(
                {
                    "run_id": run_id,
                    "cohort": cohort,
                    "local_batch_size": batch,
                    "world_size": WORLD_SIZE,
                    "global_batch_size": global_batch,
                    "max_steps": steps,
                    "planned_samples": steps * global_batch,
                    "lr": _scaled_lr(batch, lr_policy),
                    "lr_policy": lr_policy,
                    "run_dir": str(run_dir),
                    "command": command,
                    "command_sha256": _digest(command),
                }
            )
    equal_samples = {
        run["planned_samples"]
        for run in runs
        if run["cohort"] == "learning_equal_samples"
    }
    if len(equal_samples) != 1:
        raise AssertionError("equal-sample cohort does not conserve exposure")
    plan: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "midpoint_receipt": {key: midpoint[key] for key in ("path", "sha256")},
        "runtime": runtime,
        "lr_scaling": {
            "policy": lr_policy,
            "fixed": "isolates batch/system effects at the production LR",
            "sqrt": "optional noise-scale compensation; LR grows with sqrt(global batch)",
            "linear": "optional large-batch rule; highest-risk and never inferred implicitly",
        },
        "cohorts": {
            "throughput_fixed_steps": {
                "purpose": "systems throughput and optimizer-step cost",
                "steps_per_arm": throughput_steps,
                "sample_exposure_varies": True,
            },
            "learning_equal_samples": {
                "purpose": "teacher-gap closure at equal row exposure",
                "planned_samples_per_arm": next(iter(equal_samples)),
                "optimizer_steps_vary": True,
            },
        },
        "ranking_policy": {
            "primary": [
                "samples_per_second",
                "active_teacher_gap_closure_per_wall_second",
                "active_teacher_gap_closure_per_million_samples",
            ],
            "safety": [
                "clipped_fraction",
                "preclip_grad_norm",
                "module_parameter_delta_norms",
            ],
            "diagnostic_only": ["hbm_memory_mib"],
            "note": "HBM occupancy is capacity telemetry, never the optimization objective.",
        },
        "runs": runs,
    }
    plan["plan_sha256"] = _digest(plan)
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
    actual = _digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    if plan.get("schema_version") != SCHEMA or stated != actual:
        raise ProbeError("batch-probe plan schema/digest drift")
    return plan


def _verify_runtime(plan: dict[str, Any], run: dict[str, Any]) -> None:
    runtime = plan.get("runtime")
    if not isinstance(runtime, dict):
        raise ProbeError("batch-probe plan has no runtime binding")
    current = _current_runtime()
    if runtime != current:
        raise ProbeError(f"batch-probe runtime drift: planned={runtime} current={current}")
    trainers = [
        str(Path(item).resolve(strict=False))
        for item in run["command"]
        if Path(item).name == "train_bc.py"
    ]
    if trainers != [runtime["trainer"]]:
        raise ProbeError("batch-probe command does not use its bound current trainer")
    if run.get("command_sha256") != _digest(run.get("command")):
        raise ProbeError("batch-probe command digest drift")


def _require_no_non_mps_compute(*, runner=subprocess.run) -> None:
    result = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    occupied = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "nvidia-cuda-mps-server" not in line
    ]
    if occupied:
        raise ProbeError(f"B200 GPUs have active non-MPS compute: {occupied}")


@contextmanager
def _without_mps(*, runner=subprocess.run):
    """Transfer the host from fleet MPS ownership to DDP and always restore it."""

    status = runner(
        ["systemctl", "is-active", "nvidia-mps.service"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if status.returncode != 0 or status.stdout.strip() != "active":
        raise ProbeError("MPS must be active at the DDP ownership handoff")
    runner(["sudo", "-n", "true"], check=True)
    runner(["sudo", "-n", "systemctl", "stop", "nvidia-mps.service"], check=True)
    stopped = runner(
        ["systemctl", "is-active", "nvidia-mps.service"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if stopped.stdout.strip() == "active":
        runner(
            ["sudo", "-n", "systemctl", "start", "nvidia-mps.service"],
            check=False,
        )
        raise ProbeError("MPS remained active after stop; refusing DDP")
    try:
        yield
    finally:
        runner(
            ["sudo", "-n", "systemctl", "start", "nvidia-mps.service"],
            check=False,
        )


def _gpu_samples(path: Path) -> dict[str, float | int]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    def numeric(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key, "").strip()]

    util = numeric("utilization.gpu [%]")
    power = numeric("power.draw [W]")
    memory = numeric("memory.used [MiB]")
    return {
        "samples": len(rows),
        "sm_util_mean_pct": sum(util) / len(util) if util else 0.0,
        "power_mean_w": sum(power) / len(power) if power else 0.0,
        "hbm_memory_mean_mib": sum(memory) / len(memory) if memory else 0.0,
        "hbm_is_ranking_objective": False,
    }


def _optimizer_log_summary(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("progress") == "bc_optimizer_observability":
            events.append(value)
    if not events:
        raise ProbeError("training log has no per-step optimizer observability")

    def module_mean(field: str) -> dict[str, float]:
        names = sorted(
            {
                name
                for event in events
                for name in (event.get(field) or {})
            }
        )
        return {
            name: sum(float((event.get(field) or {}).get(name, 0.0)) for event in events)
            / len(events)
            for name in names
        }

    norms = [float(event["pre_clip_total_grad_norm"]) for event in events]
    return {
        "observed_steps": len(events),
        "preclip_grad_norm_mean": sum(norms) / len(norms),
        "preclip_grad_norm_max": max(norms),
        "clipped_fraction": sum(bool(event.get("clipped")) for event in events)
        / len(events),
        "module_preclip_grad_norm_mean": module_mean("module_pre_clip_grad_norms"),
        "module_parameter_update_norm_mean": module_mean(
            "module_parameter_delta_norms"
        ),
        "module_norm_scope": sorted(
            {str(event.get("module_norm_scope")) for event in events}
        ),
    }


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run["run_dir"])
    report = json.loads((run_dir / "train.report.json").read_text(encoding="utf-8"))
    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    elapsed = (int(runtime["finished_unix_ns"]) - int(runtime["started_unix_ns"])) / 1e9
    steps = int(report["steps_completed"])
    samples = steps * int(run["global_batch_size"])
    validation = report["metrics"][-1]["validation"]
    closure = float(validation["active_policy_teacher_gap_closure"])
    optimizer = _optimizer_log_summary(run_dir / "train.log")
    return {
        "run_id": run["run_id"],
        "local_batch_size": run["local_batch_size"],
        "global_batch_size": run["global_batch_size"],
        "lr": run["lr"],
        "steps": steps,
        "samples": samples,
        "elapsed_sec": elapsed,
        "samples_per_second": samples / elapsed,
        "mean_wall_step_sec": elapsed / steps,
        "active_teacher_gap_closure": closure,
        "active_teacher_gap_closure_per_wall_second": closure / elapsed,
        "active_teacher_gap_closure_per_million_samples": closure * 1_000_000 / samples,
        "optimizer_observability": optimizer,
        "gpu": _gpu_samples(run_dir / "gpu.csv"),
    }


def run_one(plan_path: Path, run_id: str, *, go: bool) -> dict[str, Any]:
    plan = _read_plan(plan_path)
    matches = [run for run in plan["runs"] if run["run_id"] == run_id]
    if len(matches) != 1:
        raise ProbeError(f"unknown run id {run_id!r}")
    run = matches[0]
    _verify_runtime(plan, run)
    if not go:
        return {"dry_run": True, **run}
    names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if len(names) != WORLD_SIZE or any("B200" not in name.upper() for name in names):
        raise ProbeError(f"--go requires exactly 8 B200 GPUs, got {names}")
    _require_no_non_mps_compute()
    with _without_mps():
        _require_no_non_mps_compute()
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
        try:
            with (run_dir / "train.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    run["command"],
                    cwd=_ROOT,
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
    (run_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise ProbeError(f"diagnostic run failed with return code {completed.returncode}")
    return summarize(run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--midpoint-receipt", type=Path, required=True)
    plan.add_argument("--out-dir", type=Path, required=True)
    plan.add_argument("--lr-policy", choices=("fixed", "sqrt", "linear"), default="fixed")
    plan.add_argument("--throughput-steps", type=int, default=24)
    plan.add_argument("--equal-sample-reference-steps", type=int, default=48)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--go", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = build_plan(
                midpoint_receipt=args.midpoint_receipt,
                output_dir=args.out_dir,
                lr_policy=args.lr_policy,
                throughput_steps=args.throughput_steps,
                equal_sample_reference_steps=args.equal_sample_reference_steps,
            )
        elif args.command == "run":
            result = run_one(args.plan, args.run_id, go=args.go)
        else:
            plan = _read_plan(args.plan)
            result = {"runs": [summarize(run) for run in plan["runs"]]}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ProbeError, dual.DualTrainError, OSError, ValueError, KeyError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
