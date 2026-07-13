#!/usr/bin/env python3
"""Seal or replay one completed non-promotable A1 diagnostic training run.

This finalizer is deliberately smaller than the production one-dose executor: it
does not launch training or confer promotion authority.  It binds an already
completed run to its immutable plan, command, inputs, runtime result, learner
report, and output bytes so downstream diagnostics/evaluation can consume the
checkpoint without treating an ad-hoc path as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "a1-diagnostic-training-receipt-v1"
EXPECTED_ARTIFACTS = (
    "candidate.pt",
    "candidate.pt.optimizer.pt",
    "candidate.pt.training-progress.json",
    "train.report.json",
    "train.report.validation_seeds.json",
    "runtime.json",
    "train.log",
    "gpu.csv",
)


class ReceiptError(ValueError):
    """A completed diagnostic cannot be authenticated."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ReceiptError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().resolve(strict=True).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ReceiptError(f"{label} is not a JSON object")
    return payload


def _load_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = path.expanduser().resolve(strict=True)
    plan = _load_json(plan_path, label="plan")
    stated = plan.get("plan_sha256")
    actual = _digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    if stated != actual:
        raise ReceiptError("plan semantic digest drift")
    if not (
        plan.get("diagnostic_only") is True
        and plan.get("promotion_eligible") is False
        and isinstance(plan.get("runs"), list)
    ):
        raise ReceiptError("plan is not diagnostic-only/non-promotable")
    return plan, _file_ref(plan_path)


def _selected_run(plan: dict[str, Any], run_id: str) -> dict[str, Any]:
    matches = [run for run in plan["runs"] if run.get("run_id") == run_id]
    if len(matches) != 1:
        raise ReceiptError(f"plan does not contain exactly one run {run_id!r}")
    run = matches[0]
    command = run.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ReceiptError("run command is malformed")
    if run.get("command_sha256") != _digest(command):
        raise ReceiptError("run command digest drift")
    return run


def _verify_completion(plan: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run.get("run_dir", ""))).expanduser().resolve(strict=True)
    runtime_path = run_dir / "runtime.json"
    report_path = run_dir / "train.report.json"
    runtime = _load_json(runtime_path, label="runtime")
    report = _load_json(report_path, label="training report")
    if not (
        runtime.get("returncode") == 0
        and runtime.get("run_id") == run["run_id"]
        and runtime.get("plan_sha256") == plan["plan_sha256"]
        and runtime.get("command_sha256") == run["command_sha256"]
        and runtime.get("gpu_ids") == run.get("gpu_ids")
        and int(runtime.get("finished_unix_ns", 0))
        > int(runtime.get("started_unix_ns", 0))
    ):
        raise ReceiptError("runtime does not prove successful execution of the selected run")
    expected_report = {
        "diagnostic_only": True,
        "promotion_eligible": False,
        "checkpoint": str((run_dir / "candidate.pt").resolve()),
        "init_checkpoint_sha256": plan["inputs"]["init_checkpoint_sha256"],
        "world_size": int(run["world_size"]),
        "batch_size": int(run["local_batch_size"]),
        "effective_global_batch_size": int(run["global_batch_size"]),
        "grad_accum_steps": int(run["grad_accum_steps"]),
        "max_steps": int(run["max_steps"]),
        "steps_completed": int(run["max_steps"]),
        "training_row_draws": int(run["planned_samples"]),
        "optimizer_restored": False,
        "resume_optimizer": False,
    }
    drift = {
        key: {"expected": expected, "observed": report.get(key)}
        for key, expected in expected_report.items()
        if report.get(key) != expected
    }
    if drift:
        raise ReceiptError(f"training report invariant drift: {drift}")
    source = plan.get("runtime")
    if not isinstance(source, dict):
        raise ReceiptError("plan has no runtime source binding")
    trainer_ref = _file_ref(Path(str(source.get("trainer", ""))))
    probe_ref = _file_ref(Path(str(source.get("quality_probe", ""))))
    if trainer_ref["sha256"] != source.get("trainer_sha256"):
        raise ReceiptError("trainer bytes drifted from plan")
    if probe_ref["sha256"] != source.get("quality_probe_sha256"):
        raise ReceiptError("quality-probe bytes drifted from plan")
    checkout = report.get("checkout_runtime_binding")
    if not isinstance(checkout, dict) or not (
        checkout.get("trainer") == trainer_ref["path"]
        and checkout.get("trainer_sha256") == trainer_ref["sha256"]
    ):
        raise ReceiptError("training report runtime binding differs from plan")
    inputs = {
        "descriptor": _file_ref(Path(plan["inputs"]["data"])),
        "parent_checkpoint": _file_ref(Path(plan["inputs"]["init_checkpoint"])),
    }
    if inputs["descriptor"]["sha256"] != plan["inputs"]["data_sha256"]:
        raise ReceiptError("descriptor bytes drifted from plan")
    if inputs["parent_checkpoint"]["sha256"] != plan["inputs"]["init_checkpoint_sha256"]:
        raise ReceiptError("parent checkpoint bytes drifted from plan")
    artifacts = {
        name: _file_ref(run_dir / name)
        for name in EXPECTED_ARTIFACTS
    }
    optional_drift = run_dir / "layer_drift.audit.json"
    if optional_drift.exists():
        artifacts[optional_drift.name] = _file_ref(optional_drift)
    return {
        "run_dir": str(run_dir),
        "runtime": runtime,
        "report": report,
        "inputs": inputs,
        "artifacts": artifacts,
        "source": {"trainer": trainer_ref, "quality_probe": probe_ref},
    }


def build_receipt(plan_path: Path, *, run_id: str) -> dict[str, Any]:
    plan, plan_ref = _load_plan(plan_path)
    run = _selected_run(plan, run_id)
    completion = _verify_completion(plan, run)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "complete_nonpromotable",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "plan": plan_ref,
        "plan_sha256": plan["plan_sha256"],
        "run_id": run_id,
        "command": run["command"],
        "command_sha256": run["command_sha256"],
        # Compatibility names consumed by bounded follow-up probes.
        "parent_checkpoint_sha256": completion["inputs"]["parent_checkpoint"]["sha256"],
        "descriptor_sha256": completion["inputs"]["descriptor"]["sha256"],
        "inputs": completion["inputs"],
        "runtime": completion["runtime"],
        "outputs": completion["artifacts"],
        "learner_summary": {
            "steps_completed": completion["report"]["steps_completed"],
            "training_row_draws": completion["report"]["training_row_draws"],
            "world_size": completion["report"]["world_size"],
            "local_batch_size": completion["report"]["batch_size"],
            "effective_global_batch_size": completion["report"][
                "effective_global_batch_size"
            ],
            "lr": completion["report"]["lr"],
            "value_lr_mult": completion["report"]["value_lr_mult"],
        },
        "source_binding": {
            **plan["runtime"],
            "files": completion["source"],
        },
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def verify_receipt(path: Path) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve(strict=True)
    receipt = _load_json(receipt_path, label="receipt")
    stated = receipt.get("receipt_sha256")
    actual = _digest({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    if not (
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == "complete_nonpromotable"
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and stated == actual
    ):
        raise ReceiptError("receipt schema/status/digest drift")
    replay = build_receipt(Path(receipt["plan"]["path"]), run_id=str(receipt["run_id"]))
    if replay != receipt:
        raise ReceiptError("receipt no longer replays from plan and output bytes")
    return receipt


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--receipt", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "finalize":
            payload = build_receipt(args.plan, run_id=args.run_id)
            _write_exclusive(args.receipt, payload)
            result = verify_receipt(args.receipt)
        else:
            result = verify_receipt(args.receipt)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ReceiptError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"REFUSED: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
