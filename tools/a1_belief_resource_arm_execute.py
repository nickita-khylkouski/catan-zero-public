#!/usr/bin/env python3
"""Verify or explicitly submit one sealed belief-resource diagnostic arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_belief_resource_arm as arm  # noqa: E402


CLAIM_SCHEMA = "a1-belief-resource-arm-execution-claim-v1"
RECEIPT_SCHEMA = "a1-belief-resource-arm-execution-receipt-v1"


class ExecutionError(RuntimeError):
    """The sealed belief diagnostic cannot be safely submitted."""


def execute(
    manifest_path: Path,
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    idle_probe: Callable[[], list[str]] = arm.temperature.base._idle_b200s,  # noqa: SLF001
) -> dict[str, Any]:
    if arm.temperature.base.SAFE_UNIT.fullmatch(unit) is None:
        raise ExecutionError("systemd unit name is invalid")
    try:
        verified = arm.verify(manifest_path)
    except arm.BeliefArmError as error:
        raise ExecutionError(str(error)) from error
    conflicts = idle_probe()
    if conflicts:
        raise ExecutionError(f"B200 compute is not idle: {conflicts}")
    root = verified["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema_version": CLAIM_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "unit": unit,
    }
    claim["claim_sha256"] = arm.temperature.base._digest(claim)  # noqa: SLF001
    claim_path = root / "diagnostic-execution.claim.json"
    try:
        arm.temperature.base._write_exclusive(claim_path, claim)  # noqa: SLF001
    except arm.temperature.base.L1Error as error:
        raise ExecutionError(str(error)) from error
    systemd_command = arm.temperature._systemd_command(  # noqa: SLF001
        unit=unit,
        repo=verified["repo"],
        root=root,
        command=verified["command"],
    )
    try:
        result = runner(systemd_command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutionError(
            f"systemd submission failed after one-shot claim: {error}"
        ) from error
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "claim": {
            "path": str(claim_path),
            "sha256": arm.temperature.base._file_sha(claim_path),  # noqa: SLF001
        },
        "unit": unit,
        "command_sha256": verified["manifest"]["command_sha256"],
        "systemd_command_sha256": arm.temperature.base._digest(  # noqa: SLF001
            systemd_command
        ),
        "systemd_stdout": result.stdout.strip(),
    }
    receipt["receipt_sha256"] = arm.temperature.base._digest(receipt)  # noqa: SLF001
    try:
        arm.temperature.base._write_exclusive(  # noqa: SLF001
            root / "diagnostic-execution.receipt.json", receipt
        )
    except arm.temperature.base.L1Error as error:
        raise ExecutionError(str(error)) from error
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--unit", default="a1-belief-resource-calibrated")
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args(argv)
    if not args.go:
        try:
            verified = arm.verify(args.manifest)
        except arm.BeliefArmError as error:
            raise SystemExit(f"REFUSED: {error}") from error
        print(
            json.dumps(
                {
                    "verified": True,
                    "launched": False,
                    "manifest": verified["manifest_ref"],
                },
                sort_keys=True,
            )
        )
        return
    try:
        receipt = execute(args.manifest, unit=args.unit)
    except ExecutionError as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(
        json.dumps(
            {
                "submitted": True,
                "unit": receipt["unit"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
