#!/usr/bin/env python3
"""Commission a zero-output action adapter on an A1 parent, then train it.

It reuses a completed adapter command, swaps in a new parent/data descriptor,
creates the selected zero-output module with the existing upgrade utility, and
trains only that adapter on all requested DDP ranks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.a1_fast_learner_loop import (
    _checkpoint_config_fields,
    _file_sha,
    _load_json,
    _option,
    _set_option,
    _write_new_json,
)


class GatherAdapterError(RuntimeError):
    """The source gather recipe cannot be reused without changing its meaning."""


ADAPTERS = {
    "gather": {
        "prefix": "target_gather_proj",
        "freeze_modules": "trunk,action_encoder,policy_head,value_heads",
        "upgrade_flag": "gather",
        "upgrade_module": "entity_graph.action_target_gather.v1",
    },
    "static": {
        "prefix": "static_action_residual_proj",
        "freeze_modules": (
            "trunk,action_encoder,policy_head,value_heads,"
            "target_gather,edge_policy,action_cross"
        ),
        "upgrade_flag": "static",
        "upgrade_module": "entity_graph.static_action_residual.v1",
    },
    "cross1": {
        "prefix": "action_cross_blocks",
        "freeze_modules": (
            "trunk,action_encoder,policy_head,value_heads,"
            "target_gather,edge_policy,static_action_residual"
        ),
        "upgrade_flag": "cross:1",
        "upgrade_module": "entity_graph.action_cross_attention.1.v1",
    },
}


def _ensure_nofile_limit(minimum: int = 65_536) -> None:
    """Give every torchrun rank the descriptor limit required by train_bc."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= minimum:
        return
    if hard != resource.RLIM_INFINITY and hard < minimum:
        raise GatherAdapterError(
            f"RLIMIT_NOFILE hard limit {hard} is below required {minimum}"
        )
    resource.setrlimit(resource.RLIMIT_NOFILE, (minimum, hard))


def _replace_trainer(command: list[str], trainer: Path) -> None:
    matches = [i for i, item in enumerate(command) if item.endswith("/tools/train_bc.py")]
    if len(matches) != 1:
        raise GatherAdapterError("gather source command must contain one trainer")
    command[matches[0]] = str(trainer)


def _set_or_append(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        _set_option(command, flag, value)
    else:
        command.extend((flag, value))


def _derive_command(
    *,
    source_manifest: Path,
    trainer: Path,
    gather_init: Path,
    data: Path,
    validation_sentinel: Path,
    checkpoint: Path,
    report: Path,
    soft_target_weight: float,
    adapter: str,
    policy_aux_active_batch_size: int,
) -> list[str]:
    source = _load_json(source_manifest)
    command = source.get("command")
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise GatherAdapterError("source manifest has no command array")
    command = list(command)
    _replace_trainer(command, trainer)
    for flag, value in (
        ("--init-checkpoint", gather_init),
        ("--data", data),
        ("--validation-game-sentinel-manifest", validation_sentinel),
        ("--checkpoint", checkpoint),
        ("--report", report),
    ):
        _set_option(command, flag, str(value))
    _set_option(command, "--soft-target-weight", str(float(soft_target_weight)))
    _set_or_append(
        command,
        "--policy-aux-active-batch-size",
        str(int(policy_aux_active_batch_size)),
    )

    adapter_config = ADAPTERS[adapter]
    _set_or_append(command, "--freeze-modules", adapter_config["freeze_modules"])
    _set_or_append(
        command,
        "--require-only-trainable-prefixes",
        adapter_config["prefix"],
    )
    expected = {
        "--max-steps": "128",
        "--freeze-modules": adapter_config["freeze_modules"],
        "--require-only-trainable-prefixes": adapter_config["prefix"],
    }
    drift = {flag: _option(command, flag) for flag, want in expected.items() if _option(command, flag) != want}
    if drift:
        raise GatherAdapterError(f"source gather recipe drifted: {drift}")
    return command


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gather-manifest", type=Path, required=True)
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), default="gather")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-sentinel", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--upgrade-tool", type=Path, required=True)
    parser.add_argument("--upgrade-receipt-tool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--soft-target-weight", type=float, default=0.9)
    parser.add_argument("--policy-aux-active-batch-size", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.policy_aux_active_batch_size < 0:
        raise GatherAdapterError("policy-active auxiliary batch size must be non-negative")
    if args.execute:
        _ensure_nofile_limit()

    source_manifest = args.source_gather_manifest.expanduser().resolve(strict=True)
    parent = args.parent_checkpoint.expanduser().resolve(strict=True)
    data = args.data.expanduser().resolve(strict=True)
    sentinel = args.validation_sentinel.expanduser().resolve(strict=True)
    trainer = args.trainer.expanduser().resolve(strict=True)
    upgrade_tool = args.upgrade_tool.expanduser().resolve(strict=True)
    upgrade_receipt_tool = args.upgrade_receipt_tool.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    gather_init = output / f"{args.adapter}-init.pt"
    upgrade_receipt = output / f"{args.adapter}-upgrade.receipt.json"
    checkpoint = output / "candidate.pt"
    report = output / "train.report.json"
    if any(path.exists() for path in (gather_init, upgrade_receipt, checkpoint, report)):
        raise GatherAdapterError(f"output is not fresh: {output}")

    upgrade_command = [
        sys.executable,
        str(upgrade_tool),
        "--in-checkpoint",
        str(parent),
        "--out-checkpoint",
        str(gather_init),
        "--flags",
        ADAPTERS[args.adapter]["upgrade_flag"],
    ]
    receipt_command = [
        sys.executable,
        str(upgrade_receipt_tool),
        "--source",
        str(parent),
        "--upgraded",
        str(gather_init),
        "--output",
        str(upgrade_receipt),
        "--module",
        ADAPTERS[args.adapter]["upgrade_module"],
    ]
    if args.execute:
        subprocess.run(upgrade_command, check=True)
        subprocess.run(receipt_command, check=True)
    else:
        # The training command is deterministic even before the upgraded bytes
        # exist; do not require a dry-run caller to manufacture them.
        gather_init.parent.mkdir(parents=True, exist_ok=True)

    command = _derive_command(
        source_manifest=source_manifest,
        trainer=trainer,
        gather_init=gather_init,
        data=data,
        validation_sentinel=sentinel,
        checkpoint=checkpoint,
        report=report,
        soft_target_weight=args.soft_target_weight,
        adapter=args.adapter,
        policy_aux_active_batch_size=args.policy_aux_active_batch_size,
    )
    manifest: dict[str, Any] = {
        "schema_version": "a1-fast-architecture-adapter-v1",
        "adapter": args.adapter,
        "policy_aux_active_batch_size": args.policy_aux_active_batch_size,
        "source_gather_manifest": {"path": str(source_manifest), "sha256": _file_sha(source_manifest)},
        "parent": {"path": str(parent), "sha256": _file_sha(parent)},
        "data": {"path": str(data), "sha256": _file_sha(data)},
        "validation_sentinel": {"path": str(sentinel), "sha256": _file_sha(sentinel)},
        "upgrade_command": upgrade_command,
        "upgrade_receipt_command": receipt_command,
        "upgrade_receipt": str(upgrade_receipt),
        "command": command,
        "output_root": str(output),
    }
    _write_new_json(output / "run.manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.execute:
        return 0

    started = time.monotonic()
    completed = subprocess.run(command, check=False)
    result: dict[str, Any] = {
        "schema_version": "a1-fast-architecture-adapter-result-v1",
        "adapter": args.adapter,
        "returncode": int(completed.returncode),
        "elapsed_sec": time.monotonic() - started,
    }
    if completed.returncode == 0 and checkpoint.is_file() and report.is_file():
        result["gather_init"] = {"path": str(gather_init), "sha256": _file_sha(gather_init)}
        result["upgrade_receipt"] = {
            "path": str(upgrade_receipt),
            "sha256": _file_sha(upgrade_receipt),
        }
        result["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": _file_sha(checkpoint),
            "effective_config": _checkpoint_config_fields(checkpoint),
        }
        result["report"] = {"path": str(report), "sha256": _file_sha(report)}
    _write_new_json(output / "run.result.json", result)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(run())
