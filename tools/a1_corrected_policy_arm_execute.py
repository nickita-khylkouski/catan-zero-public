#!/usr/bin/env python3
"""Explicitly submit one immutable corrected learner manifest to systemd."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_corrected_policy_arm as prepare  # noqa: E402


RECEIPT_SCHEMA = "a1-corrected-policy-arm-execution-receipt-v1"
STATUS_SCHEMA = "a1-corrected-policy-arm-execution-status-v1"


class ExecutionError(RuntimeError):
    """The immutable diagnostic cannot be submitted exactly once."""


def _read_manifest(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload, ref = prepare._load_json(path)  # noqa: SLF001
    stated = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if stated != prepare._digest(unhashed):  # noqa: SLF001
        raise ExecutionError("prepared manifest semantic digest drift")
    if (
        payload.get("schema_version") != prepare.SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("diagnostic_execution_authorized") is not True
    ):
        raise ExecutionError("manifest does not authorize diagnostic execution")
    return payload, ref


def _verify_ref(value: Any, *, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ExecutionError(f"{label} file reference is malformed")
    try:
        path = Path(str(value["path"])).expanduser().resolve(strict=True)
    except OSError as error:
        raise ExecutionError(f"cannot resolve {label}: {error}") from error
    if prepare._file_sha(path) != value["sha256"]:  # noqa: SLF001
        raise ExecutionError(f"{label} bytes drifted")
    return path


def _option(command: Sequence[str], flag: str) -> str:
    try:
        return prepare._option(command, flag)  # noqa: SLF001
    except prepare.ArmError as error:
        raise ExecutionError(str(error)) from error


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutionError(f"cannot identify execution checkout: {error}") from error


def _verify_event_history_training_contract(
    manifest: dict[str, Any], command: list[str], descriptor: Path
) -> None:
    contract = manifest.get("event_history_training_contract")
    if not isinstance(contract, dict) or not (
        contract.get("schema") == prepare.EVENT_HISTORY_COMMAND_CONTRACT_SCHEMA
        and contract.get("crop_authenticated_empty_event_history") is True
    ):
        raise ExecutionError("manifest has no authenticated event-history contract")
    try:
        descriptor_meta, _ = prepare._preflight_descriptor(descriptor)  # noqa: SLF001
        expected_contract = prepare._event_history_training_contract(  # noqa: SLF001
            descriptor_meta
        )
    except prepare.ArmError as error:
        raise ExecutionError(str(error)) from error
    if contract != expected_contract:
        raise ExecutionError("event-history contract differs from descriptor inventories")
    expected = [
        row["payload_inventory_sha256"]
        for row in contract["empty_payload_inventory_acknowledgements"]
    ]
    positions = [
        index
        for index, value in enumerate(command)
        if value == prepare.EVENT_HISTORY_ACK_FLAG
    ]
    observed = [
        command[index + 1]
        for index in positions
        if index + 1 < len(command) and not command[index + 1].startswith("--")
    ]
    if observed != expected or len(positions) != len(expected):
        raise ExecutionError("command lacks the exact event-history inventory ACK set")
    if command.count(prepare.EVENT_HISTORY_CROP_FLAG) != 1:
        raise ExecutionError("command lacks the authenticated empty-history crop flag")


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest, manifest_ref = _read_manifest(manifest_path)
    for field in (
        "source_receipt", "source_descriptor", "descriptor",
        "source_validation_sentinel", "validation_sentinel", "initialization",
    ):
        _verify_ref(manifest.get(field), label=field)
    lineage = manifest.get("failed_retry_lineage", {}).get("artifacts")
    if not isinstance(lineage, list) or len(lineage) != len(prepare.LINEAGE_ROLES):
        raise ExecutionError("failed/retry lineage is incomplete")
    for role, row in zip(prepare.LINEAGE_ROLES, lineage, strict=True):
        if not isinstance(row, dict) or row.get("role") != role:
            raise ExecutionError("failed/retry lineage order drift")
        _verify_ref(row.get("file"), label=f"lineage.{role}")
    source_binding = manifest.get("source_binding")
    if not isinstance(source_binding, dict):
        raise ExecutionError("manifest has no source checkout binding")
    repo = Path(str(source_binding.get("repository_root", ""))).resolve(strict=True)
    if _git_head(repo) != source_binding.get("git_commit"):
        raise ExecutionError("execution checkout commit differs from prepared manifest")
    files = source_binding.get("files")
    if not isinstance(files, dict) or not files:
        raise ExecutionError("source checkout binding has no files")
    for relative, ref in files.items():
        path = _verify_ref(ref, label=f"source.{relative}")
        if path != (repo / relative).resolve(strict=True):
            raise ExecutionError(f"source path escaped checkout: {relative}")
    command = manifest.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ExecutionError("manifest command is malformed")
    if manifest.get("command_sha256") != prepare._digest(command):  # noqa: SLF001
        raise ExecutionError("manifest command digest drift")
    trainer = [Path(value).resolve() for value in command if Path(value).name == "train_bc.py"]
    if trainer != [(repo / "tools/train_bc.py").resolve(strict=True)]:
        raise ExecutionError("manifest trainer is not the bound checkout")
    exact_inputs = {
        "--data": manifest["descriptor"]["path"],
        "--validation-game-sentinel-manifest": manifest["validation_sentinel"]["path"],
        "--init-checkpoint": manifest["initialization"]["path"],
    }
    for flag, expected in exact_inputs.items():
        if _option(command, flag) != expected:
            raise ExecutionError(f"command differs from bound {flag}")
    _verify_event_history_training_contract(
        manifest, command, Path(manifest["descriptor"]["path"])
    )
    if "--validation-game-seed-manifest" in command:
        raise ExecutionError("command contains a second validation control")
    output_root = Path(_option(command, "--checkpoint")).parent.resolve()
    checkpoint = output_root / "candidate.pt"
    report = output_root / "train.report.json"
    if Path(_option(command, "--checkpoint")) != checkpoint or Path(
        _option(command, "--report")
    ) != report:
        raise ExecutionError("command outputs are not canonical corrected-arm paths")
    forbidden = (
        checkpoint, Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"), report,
        output_root / "diagnostic-execution.claim.json",
        output_root / "diagnostic-execution.receipt.json",
        output_root / "diagnostic-execution.status.jsonl",
        output_root / "stdout.log", output_root / "stderr.log",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise ExecutionError(f"corrected-arm output/claim already exists: {existing}")
    return {
        "manifest": manifest, "manifest_ref": manifest_ref, "repo": repo,
        "command": command, "output_root": output_root,
    }


def _probe_conflicting_compute(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    try:
        topology = runner(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        )
        result = runner(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutionError(f"cannot prove B200s are idle: {error}") from error
    names = [line.strip() for line in topology.stdout.splitlines() if line.strip()]
    if len(names) != 8 or any("B200" not in name for name in names):
        raise ExecutionError(f"executor requires exactly eight visible B200s: {names}")
    return [
        line.strip() for line in result.stdout.splitlines()
        if line.strip() and "nvidia-cuda-mps" not in line.lower()
    ]


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise


def _append_status(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def execute(
    manifest_path: Path, *, unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    conflict_probe: Callable[[], list[str]] = _probe_conflicting_compute,
) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,79}", unit) is None:
        raise ExecutionError("systemd unit name is invalid")
    verified = verify(manifest_path)
    return _submit_verified(
        verified,
        unit=unit,
        runner=runner,
        conflict_probe=conflict_probe,
        claim_schema="a1-corrected-policy-arm-execution-claim-v1",
        receipt_schema=RECEIPT_SCHEMA,
        status_schema=STATUS_SCHEMA,
    )


def _submit_verified(
    verified: dict[str, Any],
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    conflict_probe: Callable[[], list[str]],
    claim_schema: str,
    receipt_schema: str,
    status_schema: str,
) -> dict[str, Any]:
    """Submit one already schema-verified diagnostic with append-only evidence."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,79}", unit) is None:
        raise ExecutionError("systemd unit name is invalid")
    conflicts = conflict_probe()
    if conflicts:
        raise ExecutionError(f"B200 compute is not idle: {conflicts}")
    output_root = verified["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    now = time.time_ns()
    claim = {
        "schema_version": claim_schema,
        "created_at_unix_ns": now,
        "manifest": verified["manifest_ref"],
        "unit": unit,
    }
    claim["claim_sha256"] = prepare._digest(claim)  # noqa: SLF001
    claim_path = output_root / "diagnostic-execution.claim.json"
    try:
        _write_exclusive(claim_path, claim)
    except FileExistsError as error:
        raise ExecutionError(
            f"diagnostic execution was already claimed: {claim_path}"
        ) from error
    status_path = output_root / "diagnostic-execution.status.jsonl"
    _append_status(status_path, {
        "schema_version": status_schema, "event": "authorized",
        "created_at_unix_ns": now, "claim_sha256": claim["claim_sha256"],
    })
    stdout = output_root / "stdout.log"
    stderr = output_root / "stderr.log"
    systemd_command = [
        "sudo", "-n", "systemd-run", f"--unit={unit}", "--uid=ubuntu", "--gid=ubuntu",
        "--service-type=exec", "--collect",
        "--property=LimitNOFILE=65536",
        f"--property=WorkingDirectory={verified['repo']}",
        f"--property=StandardOutput=append:{stdout}",
        f"--property=StandardError=append:{stderr}",
        "--setenv=HOME=/home/ubuntu", "--setenv=PYTHONNOUSERSITE=1",
        "--", *verified["command"],
    ]
    try:
        result = runner(systemd_command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        _append_status(status_path, {
            "schema_version": status_schema, "event": "submission_failed",
            "created_at_unix_ns": time.time_ns(), "error": str(error),
        })
        raise ExecutionError(f"systemd submission failed: {error}") from error
    receipt = {
        "schema_version": receipt_schema,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "claim": {"path": str(claim_path), "sha256": prepare._file_sha(claim_path)},  # noqa: SLF001
        "unit": unit,
        "command_sha256": verified["manifest"]["command_sha256"],
        "systemd_command_sha256": prepare._digest(systemd_command),  # noqa: SLF001
        "systemd_stdout": result.stdout.strip(),
    }
    receipt["receipt_sha256"] = prepare._digest(receipt)  # noqa: SLF001
    receipt_path = output_root / "diagnostic-execution.receipt.json"
    _write_exclusive(receipt_path, receipt)
    _append_status(status_path, {
        "schema_version": status_schema, "event": "submitted",
        "created_at_unix_ns": receipt["created_at_unix_ns"],
        "receipt_sha256": receipt["receipt_sha256"], "unit": unit,
    })
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--unit", default="a1-corrected-anchor-k3")
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args(argv)
    if not args.go:
        verified = verify(args.manifest)
        print(json.dumps({"verified": True, "launched": False,
                          "manifest": verified["manifest_ref"]}, sort_keys=True))
        return
    receipt = execute(args.manifest, unit=args.unit)
    print(json.dumps({"submitted": True, "unit": receipt["unit"],
                      "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
