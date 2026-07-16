#!/usr/bin/env python3
"""Run one authenticated parent-update R&D delta on eight local B200s.

Production ``tools/train.py`` intentionally accepts only catalogued recipes.
This launcher keeps that boundary intact: it loads the commissioned parent-
update recipe, verifies a checked-in R&D descriptor, and permits exactly one
experimental change (``trunk_lr_mult=0.25``). Runtime artifacts are supplied
through a hash-bound JSON file so all eight DDP ranks consume identical bytes.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_ROOT, _ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from tools import train as canonical_train  # noqa: E402
from catan_zero.rl.production_recipe_catalog import canonical_json_sha256  # noqa: E402


ARM_SCHEMA = "catan-zero-rnd-parent-update-arm-v1"
BINDINGS_SCHEMA = "catan-zero-rnd-parent-update-bindings-v1"
ARM_NAME = "a1-parent-update-35m-b200-trunk25-rnd"
ALLOWED_OVERRIDE = {"trunk_lr_mult": 0.25}
REQUIRED_WORLD_SIZE = 8


class ArmError(ValueError):
    """The experimental arm or its runtime bindings are not exact."""


def _load_json(path: Path, *, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArmError(f"cannot load {what} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArmError(f"{what} must be a JSON object")
    return value


def _absolute(raw: object, *, field: str, must_exist: bool) -> Path:
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise ArmError(f"{field} must be an absolute path")
    source = Path(raw).expanduser()
    if must_exist:
        if source.is_symlink():
            raise ArmError(f"{field} must not be a symlink: {source}")
        try:
            path = source.resolve(strict=True)
        except OSError as error:
            raise ArmError(f"cannot resolve {field}: {error}") from error
        if not path.is_file():
            raise ArmError(f"{field} must be a regular non-symlink file: {path}")
    else:
        path = source.resolve(strict=False)
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _expected_sha(raw: object, *, field: str) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) != 71
        or not raw.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in raw[7:])
    ):
        raise ArmError(f"{field} must be sha256:<64 lowercase hex characters>")
    return raw


def _attest_file(record: object, *, field: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ArmError(f"{field} must contain exactly path and sha256")
    path = _absolute(record["path"], field=f"{field}.path", must_exist=True)
    expected = _expected_sha(record["sha256"], field=f"{field}.sha256")
    actual = canonical_train._sha256(path)  # noqa: SLF001 - shared exact attestation
    if actual != expected:
        raise ArmError(
            f"{field} hash mismatch: expected={expected} actual={actual} path={path}"
        )
    return path


def _load_arm(path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    arm = _load_json(path, what="R&D arm")
    if set(arm) != {
        "schema_version",
        "name",
        "base_recipe",
        "overrides",
        "required_world_size",
    }:
        raise ArmError("R&D arm fields drifted")
    if arm["schema_version"] != ARM_SCHEMA or arm["name"] != ARM_NAME:
        raise ArmError("R&D arm identity drifted")
    if arm["required_world_size"] != REQUIRED_WORLD_SIZE:
        raise ArmError("R&D arm must require exactly eight DDP ranks")
    if arm["overrides"] != ALLOWED_OVERRIDE:
        raise ArmError(
            f"R&D arm must differ only by {ALLOWED_OVERRIDE!r}; "
            f"actual={arm['overrides']!r}"
        )
    base = arm["base_recipe"]
    if not isinstance(base, dict) or set(base) != {"name", "path", "canonical_sha256"}:
        raise ArmError("base_recipe fields drifted")
    if base["name"] != "a1-parent-update-35m-b200":
        raise ArmError("R&D arm must inherit the commissioned parent-update recipe")
    relative = Path(str(base["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ArmError("base_recipe.path must stay inside the repository")
    base_path = (_ROOT / relative).resolve(strict=True)
    config, engine = canonical_train._load_recipe(base_path)  # noqa: SLF001
    payload = _load_json(base_path, what="base recipe")
    actual = canonical_json_sha256(payload)
    if actual != base["canonical_sha256"]:
        raise ArmError(
            "R&D base recipe semantic hash drifted: "
            f"expected={base['canonical_sha256']} actual={actual}"
        )
    return dataclasses.replace(config, trunk_lr_mult=0.25), engine, arm


def _load_bindings(path: Path) -> tuple[argparse.Namespace, dict[str, Any]]:
    value = _load_json(path, what="R&D bindings")
    required = {
        "schema_version",
        "arm",
        "data",
        "parent_checkpoint",
        "init_checkpoint",
        "checkpoint",
        "report",
    }
    optional = {"architecture_upgrade_receipt", "host_lock_file"}
    if set(value) - (required | optional) or required - set(value):
        raise ArmError(
            "R&D bindings fields drifted: "
            f"missing={sorted(required - set(value))} "
            f"unknown={sorted(set(value) - required - optional)}"
        )
    if value["schema_version"] != BINDINGS_SCHEMA or value["arm"] != ARM_NAME:
        raise ArmError("R&D bindings identity drifted")
    data = _attest_file(value["data"], field="data")
    parent = _attest_file(value["parent_checkpoint"], field="parent_checkpoint")
    initializer = _attest_file(value["init_checkpoint"], field="init_checkpoint")
    receipt = ""
    if "architecture_upgrade_receipt" in value:
        receipt = str(
            _attest_file(
                value["architecture_upgrade_receipt"],
                field="architecture_upgrade_receipt",
            )
        )
    checkpoint = _absolute(value["checkpoint"], field="checkpoint", must_exist=False)
    report = _absolute(value["report"], field="report", must_exist=False)
    lock_raw = value.get("host_lock_file", "/tmp/catan_zero_train_bc_trunk25.lock")
    if not isinstance(lock_raw, str) or not Path(lock_raw).is_absolute():
        raise ArmError("host_lock_file must be an absolute path")
    public_args = argparse.Namespace(
        data=str(data),
        checkpoint=str(checkpoint),
        report=str(report),
        init_checkpoint=str(initializer),
        parent_checkpoint=str(parent),
        architecture_upgrade_receipt=receipt,
        device="auto",
        host_lock_file=lock_raw,
        allow_concurrent_bc=False,
    )
    return public_args, value


def _stamp_report(
    report_path: str | Path,
    *,
    arm_path: Path,
    arm: Mapping[str, Any],
    bindings_path: Path,
    bindings: Mapping[str, Any],
) -> None:
    path = Path(report_path).resolve(strict=True)
    report = _load_json(path, what="training report")
    report["rnd_arm"] = {
        "schema_version": ARM_SCHEMA,
        "name": ARM_NAME,
        "arm_config": str(arm_path.resolve(strict=True)),
        "arm_config_sha256": canonical_train._sha256(arm_path),  # noqa: SLF001
        "base_recipe": copy.deepcopy(arm["base_recipe"]),
        "overrides": copy.deepcopy(arm["overrides"]),
        "bindings": str(bindings_path.resolve(strict=True)),
        "bindings_sha256": canonical_train._sha256(bindings_path),  # noqa: SLF001
        "input_bindings": {
            key: copy.deepcopy(bindings[key])
            for key in (
                "data",
                "parent_checkpoint",
                "init_checkpoint",
                "architecture_upgrade_receipt",
            )
            if key in bindings
        },
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-config", required=True)
    parser.add_argument("--bindings", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != REQUIRED_WORLD_SIZE:
        raise SystemExit(
            f"{ARM_NAME} requires WORLD_SIZE={REQUIRED_WORLD_SIZE}; actual={world_size}"
        )
    arm_path = Path(args.arm_config).expanduser().resolve(strict=True)
    bindings_path = Path(args.bindings).expanduser().resolve(strict=True)
    try:
        config, engine, arm = _load_arm(arm_path)
        public_args, bindings = _load_bindings(bindings_path)
    except (OSError, ArmError) as error:
        raise SystemExit(str(error)) from error
    initialization = canonical_train._parent_initializer_binding(public_args)  # noqa: SLF001
    engine_args = canonical_train._engine_namespace(  # noqa: SLF001
        config=config,
        engine_settings=engine,
        public_args=public_args,
    )
    from tools import train_bc

    train_bc.main(engine_args)
    if int(os.environ.get("RANK", "0")) == 0:
        canonical_train._bind_parent_report(  # noqa: SLF001
            public_args.report,
            initialization=initialization,
        )
        _stamp_report(
            public_args.report,
            arm_path=arm_path,
            arm=arm,
            bindings_path=bindings_path,
            bindings=bindings,
        )


if __name__ == "__main__":
    main()
