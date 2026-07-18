#!/usr/bin/env python3
"""Bind a selected V6 checkpoint as a non-promotable coherent-n128 teacher.

This is a Stage-C reanalysis authority, not a generation or promotion
authority.  It reuses an authenticated coherent-public n128 operator contract
while replacing only the producer checkpoint with explicitly matched V6
feature semantics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
    RUST_ENTITY_ADAPTER_V6,
)
from catan_zero.rl.meaningful_history import (  # noqa: E402
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
)
from tools import a1_stage_c_teacher_alignment as alignment  # noqa: E402
from tools import a1_target_eligibility_inventory as inventory  # noqa: E402
from tools import train_bc  # noqa: E402


class BindingError(RuntimeError):
    """The selected checkpoint cannot become an R&D teacher."""


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BindingError(f"cannot load {where}: {error}") from error
    if not isinstance(payload, dict):
        raise BindingError(f"{where} must contain one JSON object")
    return resolved, payload


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise BindingError(f"refusing non-fresh binding path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_history_contract(checkpoint: Path) -> dict[str, Any]:
    """Read the checkpoint-owned history surface used by native inference."""

    try:
        enabled, limit, schema, _pooling, _target_gather = (
            train_bc._checkpoint_meaningful_public_history(str(checkpoint))  # noqa: SLF001
        )
    except (OSError, RuntimeError, SystemExit, ValueError) as error:
        raise BindingError(f"cannot authenticate checkpoint history: {error}") from error
    return {
        "meaningful_public_history": bool(enabled),
        "meaningful_public_history_schema": str(schema),
        "event_history_limit": int(limit),
    }


def _typed_history_contract(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the typed generator's implicit V6 history schema."""

    enabled = fields.get("meaningful_public_history")
    limit = fields.get("event_history_limit")
    if type(enabled) is not bool or type(limit) is not int:
        raise BindingError(
            "typed V6 generation config must explicitly bind "
            "meaningful_public_history and event_history_limit"
        )
    return {
        "meaningful_public_history": enabled,
        "meaningful_public_history_schema": MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
        "event_history_limit": limit,
    }


def bind(
    *,
    checkpoint: Path,
    base_operator_contract: Path,
    typed_generation_config: Path,
    binding_id: str,
    output: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    base_path, base = _load_json(
        base_operator_contract, where="base coherent operator contract"
    )
    config_path, config = _load_json(
        typed_generation_config, where="V6 teacher typed generation config"
    )
    try:
        base_inspection = inventory.inspect_rd_contract(base_path)
    except (inventory.InventoryError, OSError, ValueError) as error:
        raise BindingError(f"base coherent operator refused: {error}") from error

    fields = config.get("fields")
    operator = base.get("operator")
    if (
        config.get("pipeline") != "generate"
        or config.get("schema_version") != 13
        or not isinstance(fields, Mapping)
        or not isinstance(operator, Mapping)
    ):
        raise BindingError("V6 teacher config/operator is malformed")
    try:
        checkpoint_adapter = train_bc._checkpoint_entity_feature_adapter_version(  # noqa: SLF001
            str(checkpoint)
        )
    except (OSError, RuntimeError, SystemExit, ValueError) as error:
        raise BindingError(f"cannot authenticate checkpoint adapter: {error}") from error
    if (
        checkpoint_adapter != RUST_ENTITY_ADAPTER_V6
        or fields.get("teacher_entity_feature_adapter_version")
        != checkpoint_adapter
        or fields.get("learner_entity_feature_adapter_version")
        != checkpoint_adapter
    ):
        raise BindingError(
            "checkpoint, teacher evaluator, and learner rows must all bind exact V6"
        )
    checkpoint_history = _checkpoint_history_contract(checkpoint)
    typed_history = _typed_history_contract(fields)
    if checkpoint_history != typed_history:
        raise BindingError(
            "checkpoint and typed generator history contracts differ: "
            f"checkpoint={checkpoint_history!r} typed={typed_history!r}"
        )
    drift = {
        key: {"base_contract": value, "typed_config": fields.get(key)}
        for key, value in operator.items()
        if key in fields and fields.get(key) != value
    }
    if drift:
        raise BindingError(
            "typed config changes base coherent operator: "
            + json.dumps(drift, sort_keys=True)
        )
    if not binding_id.strip():
        raise BindingError("binding id must be non-empty")

    payload: dict[str, Any] = {
        "schema_version": alignment.RD_TEACHER_TRANSITION_BINDING_SCHEMA,
        "binding_id": binding_id.strip(),
        "status": "ready_nonpromotable_reanalysis_teacher",
        "purpose": (
            "Use one selected V8/V6 checkpoint as the exact coherent-public n128 "
            "teacher for bounded Stage-C root reanalysis."
        ),
        "diagnostic_only": True,
        "promotion_eligible": False,
        "production_authority": False,
        "producer_checkpoint": {
            "path": str(checkpoint),
            "sha256": alignment._file_sha256(checkpoint),  # noqa: SLF001
        },
        "teacher_feature_contract": {
            "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
            "entity_feature_adapter_version": checkpoint_adapter,
            **checkpoint_history,
        },
        "base_operator_contract": {
            "path": str(base_path),
            "file_sha256": alignment._file_sha256(base_path),  # noqa: SLF001
            "contract_sha256": base_inspection["contract_sha256"],
        },
        "typed_generation_config": {
            "path": str(config_path),
            "file_sha256": alignment._file_sha256(config_path),  # noqa: SLF001
        },
        "target_information_regime": base["target_information_regime"],
        "allowed_use": {
            "stage_c_root_reanalysis": True,
            "self_play_generation": False,
            "production_promotion": False,
            "production_pointer_update": False,
        },
    }
    payload["binding_sha256"] = alignment._value_sha256(payload)  # noqa: SLF001
    _write_immutable(output, payload)
    written, replay = _load_json(output, where="written R&D teacher binding")
    try:
        alignment._rd_teacher_transition_authority(  # noqa: SLF001
            written, replay, checkpoint
        )
    except alignment.AlignmentError as error:
        raise BindingError(f"written R&D teacher binding did not replay: {error}") from error
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("bind")
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--base-operator-contract", required=True, type=Path)
    command.add_argument("--typed-generation-config", required=True, type=Path)
    command.add_argument("--binding-id", required=True)
    command.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "bind":
        raise BindingError(f"unsupported command {args.command!r}")
    result = bind(
        checkpoint=args.checkpoint,
        base_operator_contract=args.base_operator_contract,
        typed_generation_config=args.typed_generation_config,
        binding_id=args.binding_id,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
