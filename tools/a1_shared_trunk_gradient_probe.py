#!/usr/bin/env python3
"""Run a bounded, non-promotable shared-trunk gradient diagnostic.

The normal trainer always validates and writes a candidate at ``--max-steps``.
That is the right production contract and the wrong contract for a 32-step
gradient probe.  This runner derives a single-GPU command from an authenticated
training launch receipt, streams the trainer's existing
``bc_optimizer_observability`` events, and terminates the process immediately
after the requested event count.  Consequently the trainer never reaches
validation or checkpoint finalization.

Only diagnostics emitted by :mod:`tools.train_bc` are interpreted here.  The
runner does not reimplement data loading, sampling, losses, optimizer behavior,
or gradient measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from typing import Any, Sequence


SCHEMA = "a1-shared-trunk-gradient-probe-v1"
DEFAULT_STEPS = 32
FORBIDDEN_OUTPUT_NAMES = (
    "candidate.pt",
    "candidate.pt.optimizer.pt",
    "candidate.pt.training-progress.json",
    "train.report.json",
)


class ProbeError(RuntimeError):
    """The requested run is not an exact, bounded diagnostic."""


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


def _option(command: Sequence[str], flag: str) -> str:
    indices = [index for index, item in enumerate(command) if item == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ProbeError(f"source command must contain exactly one valued {flag}")
    return str(command[indices[0] + 1])


def _optional_option(command: Sequence[str], flag: str) -> str | None:
    indices = [index for index, item in enumerate(command) if item == flag]
    if not indices:
        return None
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ProbeError(f"source command has malformed {flag}")
    return str(command[indices[0] + 1])


def _set_option(command: list[str], flag: str, value: str) -> None:
    indices = [index for index, item in enumerate(command) if item == flag]
    if len(indices) > 1:
        raise ProbeError(f"source command repeats {flag}")
    if indices:
        index = indices[0]
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ProbeError(f"source command has valueless {flag}")
        command[index + 1] = value
    else:
        command.extend((flag, value))


def _single_gpu_command(source: Sequence[str]) -> list[str]:
    """Remove only torchrun topology; retain the trainer argv byte-for-byte."""
    command = list(source)
    try:
        module = command.index("-m")
    except ValueError as error:
        raise ProbeError("source command is not a torchrun command") from error
    if command[module + 1 : module + 2] != ["torch.distributed.run"]:
        raise ProbeError("source command does not invoke torch.distributed.run")
    trainer_indices = [
        index for index, item in enumerate(command) if Path(item).name == "train_bc.py"
    ]
    if len(trainer_indices) != 1 or trainer_indices[0] <= module + 1:
        raise ProbeError("source command does not name one trainer after torchrun")
    trainer_index = trainer_indices[0]
    # Keep the exact Python interpreter and exact trainer/argv.  All torchrun
    # launcher flags disappear, which makes the trainer's distributed state
    # world_size=1 without changing any learner flag.
    return [command[0], *command[trainer_index:]]


def _load_source_receipt(path: Path, expected_parent_sha256: str) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve(strict=True)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProbeError("source launch receipt is not a JSON object")
    stated = payload.get("receipt_sha256")
    actual = _digest({key: value for key, value in payload.items() if key != "receipt_sha256"})
    if stated != actual:
        raise ProbeError("source launch receipt digest drift")
    if payload.get("diagnostic_only") is not True or payload.get("promotion_eligible") is not False:
        raise ProbeError("source launch receipt is not diagnostic-only/non-promotable")
    if payload.get("parent_checkpoint_sha256") != expected_parent_sha256:
        raise ProbeError("source launch receipt is not independently initialized from expected f7")
    command = payload.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ProbeError("source launch receipt has no replayable command")
    if payload.get("command_sha256") != _digest(command):
        raise ProbeError("source training command digest drift")
    parent = Path(_option(command, "--init-checkpoint")).resolve(strict=True)
    descriptor = Path(_option(command, "--data")).resolve(strict=True)
    if _file_sha(parent) != expected_parent_sha256:
        raise ProbeError("f7 checkpoint bytes differ from the source receipt")
    if _file_sha(descriptor) != payload.get("descriptor_sha256"):
        raise ProbeError("authenticated composite descriptor bytes drifted")
    if "--no-resume-optimizer" not in command:
        raise ProbeError("probe must independently reload f7 with fresh optimizer state")
    return {"path": receipt_path, "payload": payload}


def _runtime_binding(trainer: Path) -> dict[str, str]:
    trainer = trainer.expanduser().resolve(strict=True)
    root = trainer.parents[1]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if dirty:
        raise ProbeError("trainer checkout has tracked modifications")
    return {
        "repository_root": str(root),
        "repository_commit": commit,
        "trainer": str(trainer),
        "trainer_sha256": _file_sha(trainer),
    }


def build_plan(
    *,
    source_receipt: Path,
    output_dir: Path,
    expected_parent_sha256: str,
    steps: int = DEFAULT_STEPS,
    gpu: int = 0,
) -> dict[str, Any]:
    if steps <= 0 or gpu < 0:
        raise ProbeError("steps must be positive and GPU index nonnegative")
    source = _load_source_receipt(source_receipt, expected_parent_sha256)
    source_command = list(source["payload"]["command"])
    command = _single_gpu_command(source_command)
    trainer = Path(next(item for item in command if Path(item).name == "train_bc.py"))
    runtime = _runtime_binding(trainer)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ProbeError(f"refusing existing output directory: {output_dir}")
    # Keep max_steps beyond the observation horizon.  The runner terminates on
    # event N, before the trainer's end-of-epoch validation/checkpoint path.
    _set_option(command, "--max-steps", str(max(steps + 1, 1024)))
    _set_option(command, "--train-diagnostics-every-batches", "1")
    _set_option(command, "--progress-every-batches", "1")
    _set_option(command, "--checkpoint", str(output_dir / "candidate.pt"))
    _set_option(command, "--report", str(output_dir / "train.report.json"))
    source_args = source_command[
        next(i for i, item in enumerate(source_command) if Path(item).name == "train_bc.py")
        + 1 :
    ]
    probe_args = command[
        next(i for i, item in enumerate(command) if Path(item).name == "train_bc.py")
        + 1 :
    ]
    changed_flags = {
        flag: {
            "source": _optional_option(source_args, flag),
            "probe": _option(probe_args, flag),
        }
        for flag in (
            "--max-steps",
            "--train-diagnostics-every-batches",
            "--progress-every-batches",
            "--checkpoint",
            "--report",
        )
    }
    plan: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "steps": int(steps),
        "gpu": int(gpu),
        "source_launch_receipt": str(source["path"]),
        "source_launch_receipt_sha256": _file_sha(source["path"]),
        "parent_checkpoint": _option(command, "--init-checkpoint"),
        "parent_checkpoint_sha256": expected_parent_sha256,
        "authenticated_composite": _option(command, "--data"),
        "authenticated_composite_sha256": _file_sha(Path(_option(command, "--data"))),
        "sampler_recipe_preserved": True,
        "topology_delta": "8-rank DDP to rank-local single GPU; learner argv otherwise preserved",
        "changed_flags": changed_flags,
        "termination_contract": (
            "SIGTERM process group immediately after N optimizer-observability events; "
            "validation and candidate finalization must remain unreachable"
        ),
        "runtime": runtime,
        "command": command,
        "command_sha256": _digest(command),
    }
    plan["plan_sha256"] = _digest(plan)
    output_dir.mkdir(parents=True)
    (output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(output_dir / "plan.json", 0o444)
    return plan


def _read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    stated = plan.get("plan_sha256")
    actual = _digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    if (
        plan.get("schema_version") != SCHEMA
        or plan.get("diagnostic_only") is not True
        or plan.get("promotion_eligible") is not False
        or stated != actual
    ):
        raise ProbeError("gradient-probe plan schema/digest/authority drift")
    return plan


def _require_gpu_idle(gpu: int) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    occupied = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "nvidia-cuda-mps-server" not in line
    ]
    if occupied:
        raise ProbeError(f"GPU {gpu} has active non-MPS compute: {occupied}")


def _aggregate(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: Sequence[float]) -> dict[str, float]:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return {}
        return {
            "min": min(finite),
            "mean": statistics.fmean(finite),
            "median": statistics.median(finite),
            "max": max(finite),
        }

    interference = [row["objective_gradient_interference"] for row in observations]
    modules = sorted(
        {
            module
            for row in interference
            for module in row.get("modules", {})
        }
    )
    return {
        "policy_trunk_grad_norm": stats(
            [row["policy_trunk_grad_norm"] for row in interference]
        ),
        "value_trunk_grad_norm": stats(
            [row["value_trunk_grad_norm"] for row in interference]
        ),
        "value_to_policy_grad_norm_ratio": stats(
            [row["value_to_policy_grad_norm_ratio"] for row in interference]
        ),
        "trunk_gradient_cosine": stats(
            [row["trunk_gradient_cosine"] for row in interference]
        ),
        "opposing_coordinate_fraction": stats(
            [row["opposing_coordinate_fraction"] for row in interference]
        ),
        "pre_clip_total_grad_norm": stats(
            [row["pre_clip_total_grad_norm"] for row in observations]
        ),
        "clipped_fraction": statistics.fmean(bool(row["clipped"]) for row in observations),
        "module_parameter_delta_norms": {
            module: stats(
                [row["module_parameter_delta_norms"][module] for row in observations]
            )
            for module in sorted(observations[0].get("module_parameter_delta_norms", {}))
        },
        "objective_gradient_modules": {
            module: {
                field: stats(
                    [
                        row.get("modules", {}).get(module, {}).get(field)
                        for row in interference
                        if row.get("modules", {}).get(module, {}).get(field) is not None
                    ]
                )
                for field in ("policy_grad_norm", "value_grad_norm", "cosine")
            }
            for module in modules
        },
    }


def run_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan = _read_plan(plan_path)
    runtime = plan["runtime"]
    if _runtime_binding(Path(runtime["trainer"])) != runtime:
        raise ProbeError("trainer runtime binding drift")
    if _file_sha(Path(plan["parent_checkpoint"])) != plan["parent_checkpoint_sha256"]:
        raise ProbeError("f7 checkpoint drift after planning")
    if _file_sha(Path(plan["authenticated_composite"])) != plan["authenticated_composite_sha256"]:
        raise ProbeError("composite descriptor drift after planning")
    output_dir = plan_path.parent
    forbidden = [output_dir / name for name in FORBIDDEN_OUTPUT_NAMES]
    if any(path.exists() for path in forbidden):
        raise ProbeError("candidate/report artifact exists before diagnostic launch")
    gpu = int(plan["gpu"])
    _require_gpu_idle(gpu)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    process = subprocess.Popen(
        plan["command"],
        cwd=runtime["repository_root"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    observations: list[dict[str, Any]] = []
    log_path = output_dir / "trainer.stdout.log"
    try:
        with log_path.open("x", encoding="utf-8") as log:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("progress") != "bc_optimizer_observability":
                    continue
                interference = event.get("objective_gradient_interference")
                if not isinstance(interference, dict) or interference.get("available") is not True:
                    raise ProbeError(f"objective gradient telemetry unavailable: {interference}")
                observations.append(event)
                if len(observations) == int(plan["steps"]):
                    os.killpg(process.pid, signal.SIGTERM)
                    break
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    if len(observations) != int(plan["steps"]):
        raise ProbeError(
            f"trainer ended after {len(observations)}/{plan['steps']} gradient observations"
        )
    leaked = [str(path) for path in forbidden if path.exists()]
    if leaked:
        raise ProbeError(f"trainer reached forbidden candidate/report finalization: {leaked}")
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "plan": str(plan_path),
        "plan_sha256": _file_sha(plan_path),
        "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
        "authenticated_composite_sha256": plan["authenticated_composite_sha256"],
        "steps_observed": len(observations),
        "elapsed_sec": time.time() - started,
        "trainer_exit": process.returncode,
        "termination": "bounded_before_validation_and_checkpoint",
        "promotion_artifacts_emitted": False,
        "aggregate": _aggregate(observations),
        "observations": observations,
    }
    result["result_sha256"] = _digest(result)
    result_path = output_dir / "gradient-probe.result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(result_path, 0o444)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--source-receipt", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--expected-parent-sha256", required=True)
    plan.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    plan.add_argument("--gpu", type=int, default=0)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command_name == "plan":
        result = build_plan(
            source_receipt=args.source_receipt,
            output_dir=args.output_dir,
            expected_parent_sha256=args.expected_parent_sha256,
            steps=args.steps,
            gpu=args.gpu,
        )
    else:
        result = run_plan(args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
