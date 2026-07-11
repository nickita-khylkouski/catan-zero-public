#!/usr/bin/env python3
"""Seal and verify the reviewed learner authority for one dual-arm candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_pre_wave_contract as generation_contract  # noqa: E402
from tools import train_bc  # noqa: E402


SPEC_SCHEMA = "a1-dual-arm-learner-spec-v1"
LOCK_SCHEMA = "a1-dual-arm-learner-lock-v1"
TOPOLOGIES = {
    2: {
        "world_size": 2,
        "local_batch_size": 512,
        "grad_accum_steps": 4,
        "global_batch_size": 4096,
        "data_format": "memmap",
        "ddp_shard_data": False,
        "fsdp": False,
    },
    8: {
        "world_size": 8,
        "local_batch_size": 512,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
        "data_format": "memmap",
        "ddp_shard_data": False,
        "fsdp": False,
    },
}
# Backwards-compatible name for callers rendering the preferred 8-rank contract.
TOPOLOGY = TOPOLOGIES[8]
EXTRA_RUNTIME = {
    "tools/a1_dual_learner_contract.py",
    "tools/a1_dual_arm_train.py",
    "tools/a1_one_dose_train.py",
}


class LearnerContractError(RuntimeError):
    """A fail-closed learner contract refusal."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ref(path: Path, *, where: str) -> dict[str, str]:
    try:
        path = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise LearnerContractError(f"cannot resolve {where}: {error}") from error
    if not path.is_file() or path.stat().st_size <= 0:
        raise LearnerContractError(f"{where} must be a non-empty file: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _load(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LearnerContractError(f"cannot load {where}: {error}") from error
    if not isinstance(value, dict):
        raise LearnerContractError(f"{where} must be a JSON object")
    return value


def inspect_artifacts(
    *, data: Path, validation: Path, producer_checkpoint: Path
) -> dict[str, Any]:
    try:
        data = data.expanduser().resolve(strict=True)
        validation = validation.expanduser().resolve(strict=True)
        producer_checkpoint = producer_checkpoint.expanduser().resolve(strict=True)
        meta = train_bc._preflight_a1_memmap_metadata(  # noqa: SLF001
            data, validation_manifest_path=validation
        )
        if meta is None:
            raise LearnerContractError("corpus is not an audited A1 memmap")
        holdout = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
            validation,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )
        train_bc._validate_a1_validation_manifest_corpus_binding(  # noqa: SLF001
            meta, holdout
        )
        mapped = train_bc.load_teacher_data_memmap(data)
        bound = train_bc._validate_a1_corpus_artifacts_and_seeds(  # noqa: SLF001
            meta, holdout, np.asarray(mapped["game_seed"], dtype=np.int64)
        )
    except (OSError, SystemExit, ValueError) as error:
        raise LearnerContractError(f"dual learner artifact refusal: {error}") from error
    identity = (bound.get("arm_id"), bound.get("subset_id"))
    if bound.get("dual_arm") is not True or identity not in train_bc.DUAL_ARM_SUBSET_COUNTS:
        raise LearnerContractError(f"unauthorized dual identity: {identity}")
    producer = _ref(producer_checkpoint, where="producer checkpoint")
    if producer["sha256"] != bound.get("producer_checkpoint_sha256"):
        raise LearnerContractError("producer bytes differ from selected-game lineage")
    selected = meta["selected_game_seed_manifest"]
    audit = meta["a1_post_wave_audit"]
    assert isinstance(selected, dict) and isinstance(audit, dict)
    validation_rows = int(holdout["validation_row_count"])
    return {
        "arm_id": identity[0],
        "subset_id": identity[1],
        "contract_sha256": holdout["a1_contract_sha256"],
        "data": data,
        "corpus_meta": _ref(data / "corpus_meta.json", where="corpus metadata"),
        "selected_manifest": _ref(Path(str(selected["path"])), where="selection"),
        "audit": _ref(Path(str(audit["path"])), where="audit"),
        "validation": _ref(validation, where="validation"),
        "producer": producer,
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "data_fingerprint": train_bc._training_data_fingerprint(str(data), "memmap"),  # noqa: SLF001
        "recipe": bound["learner_training_recipe"],
        "objective": bound["learner_value_objective"],
        "learner_code_sha256": bound["learner_code_sha256"],
        "runtime_code_tree_sha256": bound["runtime_code_tree_sha256"],
        "selected_game_seed_set_sha256": bound["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": bound["training_game_seed_set_sha256"],
        "validation_game_seed_set_sha256": holdout["validation_game_seed_set_sha256"],
        "corpus_rows": int(meta["row_count"]),
        "training_rows": int(meta["row_count"]) - validation_rows,
        "validation_rows": validation_rows,
    }


def _runtime_records() -> list[dict[str, str]]:
    paths = set(train_bc.A1_REQUIRED_RUNTIME_CODE_SUFFIXES) | EXTRA_RUNTIME
    records = []
    for relative in sorted(paths):
        path = (_REPO_ROOT / relative).resolve(strict=True)
        records.append({"path": relative, "sha256": _sha256(path)})
    return records


def render_spec(
    *, data: Path, validation: Path, producer_checkpoint: Path, world_size: int = 8
) -> dict[str, Any]:
    artifacts = inspect_artifacts(
        data=data,
        validation=validation,
        producer_checkpoint=producer_checkpoint,
    )
    return {
        "schema_version": SPEC_SCHEMA,
        "arm_id": artifacts["arm_id"],
        "subset_id": artifacts["subset_id"],
        "objective": artifacts["objective"],
        "recipe": artifacts["recipe"],
        "topology": TOPOLOGIES[world_size],
    }


def build_lock(
    *,
    arm_lock: Path,
    learner_spec: Path,
    data: Path,
    validation: Path,
    producer_checkpoint: Path,
    verify_arm_lock=generation_contract.verify_lock,
) -> dict[str, Any]:
    arm_lock = arm_lock.expanduser().resolve(strict=True)
    try:
        arm = verify_arm_lock(arm_lock, require_all_job_claims=True)
    except generation_contract.ContractError as error:
        raise LearnerContractError(f"generation arm lock refused: {error}") from error
    artifacts = inspect_artifacts(
        data=data, validation=validation, producer_checkpoint=producer_checkpoint
    )
    arm_id = arm.get("game_contract", {}).get("arm_id")
    if arm_id != artifacts["arm_id"] or arm.get("contract_sha256") != artifacts[
        "contract_sha256"
    ]:
        raise LearnerContractError("generation arm lock differs from corpus arm/contract")
    spec_path = learner_spec.expanduser().resolve(strict=True)
    spec = _load(spec_path, where="reviewed learner spec")
    if set(spec) != {"schema_version", "arm_id", "subset_id", "objective", "recipe", "topology"}:
        raise LearnerContractError("reviewed learner spec fields drift")
    if (
        spec.get("schema_version") != SPEC_SCHEMA
        or (spec.get("arm_id"), spec.get("subset_id"))
        != (artifacts["arm_id"], artifacts["subset_id"])
        or spec.get("objective") != artifacts["objective"]
        or spec.get("recipe") != artifacts["recipe"]
        or spec.get("topology") not in TOPOLOGIES.values()
    ):
        raise LearnerContractError("reviewed learner spec differs from audited artifacts")
    producer_records = [
        record for record in arm.get("checkpoints", [])
        if isinstance(record, dict) and record.get("role") == "producer"
    ]
    if len(producer_records) != 1 or producer_records[0].get("sha256") != artifacts[
        "producer"
    ]["sha256"]:
        raise LearnerContractError("generation lock producer differs from learner producer")
    value = {
        "schema_version": LOCK_SCHEMA,
        "arm_id": artifacts["arm_id"],
        "subset_id": artifacts["subset_id"],
        "generation_arm_lock": _ref(arm_lock, where="generation arm lock"),
        "generation_contract_sha256": artifacts["contract_sha256"],
        "learner_spec": _ref(spec_path, where="reviewed learner spec"),
        "objective": artifacts["objective"],
        "recipe": artifacts["recipe"],
        "topology": spec["topology"],
        "inputs": {
            key: artifacts[key]
            for key in (
                "corpus_meta", "selected_manifest", "audit", "validation", "producer"
            )
        },
        "payload_inventory_sha256": artifacts["payload_inventory_sha256"],
        "data_fingerprint": artifacts["data_fingerprint"],
        "row_counts": {
            "corpus": artifacts["corpus_rows"],
            "training": artifacts["training_rows"],
            "validation": artifacts["validation_rows"],
        },
        "selected_game_seed_set_sha256": artifacts["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": artifacts["training_game_seed_set_sha256"],
        "validation_game_seed_set_sha256": artifacts["validation_game_seed_set_sha256"],
        "runtime": _runtime_records(),
        "trainer_report_bindings": {
            "learner_code_sha256": artifacts["learner_code_sha256"],
            "runtime_code_tree_sha256": artifacts["runtime_code_tree_sha256"],
        },
    }
    value["runtime_sha256"] = _digest(value["runtime"])
    value["lock_sha256"] = _digest(value)
    return value


def verify_lock(path: Path, *, reviewed_file_sha256: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if reviewed_file_sha256 != _sha256(path):
        raise LearnerContractError("learner lock bytes differ from explicitly reviewed SHA-256")
    value = _load(path, where="dual learner lock")
    expected_fields = {
        "schema_version", "arm_id", "subset_id", "generation_arm_lock",
        "generation_contract_sha256", "learner_spec", "objective", "recipe",
        "topology", "inputs", "payload_inventory_sha256",
        "data_fingerprint", "row_counts",
        "selected_game_seed_set_sha256", "training_game_seed_set_sha256",
        "validation_game_seed_set_sha256", "runtime", "runtime_sha256",
        "trainer_report_bindings", "lock_sha256",
    }
    if set(value) != expected_fields:
        raise LearnerContractError("dual learner lock fields drift")
    stated = value.get("lock_sha256")
    unhashed = dict(value)
    unhashed.pop("lock_sha256", None)
    if value.get("schema_version") != LOCK_SCHEMA or stated != _digest(unhashed):
        raise LearnerContractError("dual learner lock schema/semantic digest drift")
    runtime = value.get("runtime")
    if not isinstance(runtime, list) or value.get("runtime_sha256") != _digest(runtime):
        raise LearnerContractError("dual learner runtime digest drift")
    required = set(train_bc.A1_REQUIRED_RUNTIME_CODE_SUFFIXES) | EXTRA_RUNTIME
    if {record.get("path") for record in runtime if isinstance(record, dict)} != required:
        raise LearnerContractError("dual learner runtime closure path drift")
    for record in runtime:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise LearnerContractError("dual learner runtime record fields drift")
        if _sha256((_REPO_ROOT / record["path"]).resolve(strict=True)) != record["sha256"]:
            raise LearnerContractError(f"dual learner runtime byte drift: {record['path']}")
    for section in ("generation_arm_lock", "learner_spec"):
        ref = value.get(section)
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or _ref(
            Path(str(ref["path"])), where=section
        ) != ref:
            raise LearnerContractError(f"dual learner lock {section} drift")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "corpus_meta", "selected_manifest", "audit", "validation", "producer"
    }:
        raise LearnerContractError("dual learner input bindings drift")
    for name, ref in inputs.items():
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or _ref(
            Path(str(ref["path"])), where=f"inputs.{name}"
        ) != ref:
            raise LearnerContractError(f"dual learner input bytes drift: {name}")
    try:
        arm = generation_contract.verify_lock(
            Path(value["generation_arm_lock"]["path"]),
            require_all_job_claims=True,
        )
    except generation_contract.ContractError as error:
        raise LearnerContractError(f"generation arm lock replay refused: {error}") from error
    if (
        arm.get("contract_sha256") != value.get("generation_contract_sha256")
        or arm.get("game_contract", {}).get("arm_id") != value.get("arm_id")
    ):
        raise LearnerContractError("generation arm lock semantic identity drift")
    spec = _load(Path(value["learner_spec"]["path"]), where="reviewed learner spec")
    if (
        set(spec)
        != {"schema_version", "arm_id", "subset_id", "objective", "recipe", "topology"}
        or
        spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("arm_id") != value.get("arm_id")
        or spec.get("subset_id") != value.get("subset_id")
        or spec.get("objective") != value.get("objective")
        or spec.get("recipe") != value.get("recipe")
        or spec.get("topology") != value.get("topology")
        or value.get("topology") not in TOPOLOGIES.values()
    ):
        raise LearnerContractError("reviewed learner spec/lock semantics drift")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-spec")
    for name in ("data", "validation", "producer-checkpoint"):
        inspect.add_argument(f"--{name}", type=Path, required=True)
    inspect.add_argument("--world-size", type=int, choices=sorted(TOPOLOGIES), default=8)
    seal = sub.add_parser("seal")
    for name in ("arm-lock", "learner-spec", "data", "validation", "producer-checkpoint", "out"):
        seal.add_argument(f"--{name}", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--reviewed-lock-file-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-spec":
            value = render_spec(
                data=args.data,
                validation=args.validation,
                producer_checkpoint=args.producer_checkpoint,
                world_size=args.world_size,
            )
        elif args.command == "seal":
            value = build_lock(
                arm_lock=args.arm_lock,
                learner_spec=args.learner_spec,
                data=args.data,
                validation=args.validation,
                producer_checkpoint=args.producer_checkpoint,
            )
            _write_new(args.out, value)
        else:
            value = verify_lock(
                args.lock, reviewed_file_sha256=args.reviewed_lock_file_sha256
            )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except (LearnerContractError, OSError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
