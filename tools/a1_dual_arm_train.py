#!/usr/bin/env python3
"""Sealed B200 learner transaction for audited dual-arm corpora.

The default is a read-only plan. ``--go`` atomically claims one arm/subset,
owns every GPU in its reviewed topology for the complete torchrun lifetime,
verifies every output, and
publishes a no-clobber receipt.  A valid completed receipt is resumable: a
repeated invocation verifies and returns it without training twice.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools import a1_dual_learner_contract as learner_contract  # noqa: E402
from tools import train_bc  # noqa: E402
from tools import a1_lineage_dose as lineage  # noqa: E402


PLAN_SCHEMA = "a1-dual-arm-training-plan-v1"
CLAIM_SCHEMA = "a1-dual-arm-training-claim-v1"
RECEIPT_SCHEMA = "a1-dual-arm-training-receipt-v1"
REPORT_BINDING_FIELD = "a1_dual_arm_execution_binding"
GLOBAL_BATCH = 4096
ALLOWED_IDENTITIES = frozenset(train_bc.DUAL_ARM_SUBSET_COUNTS)
DUAL_CORRECTIVE_ABLATION_FIELDS = frozenset(
    {"epochs", "lr", "loser_sample_weight"}
)


class DualTrainError(RuntimeError):
    """A fail-closed dual-arm learner refusal."""


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


def _file_ref(path: Path, *, where: str) -> dict[str, str]:
    try:
        path = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise DualTrainError(f"cannot resolve {where}: {error}") from error
    if not path.is_file() or path.stat().st_size <= 0:
        raise DualTrainError(f"{where} must be a non-empty file: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_inputs(
    *,
    learner_lock: Path,
    reviewed_lock_file_sha256: str,
    data: Path,
    validation: Path,
    producer_checkpoint: Path,
    curriculum_parent_receipt: Path | None = None,
) -> dict[str, Any]:
    try:
        authority = learner_contract.verify_lock(
            learner_lock, reviewed_file_sha256=reviewed_lock_file_sha256
        )
    except learner_contract.LearnerContractError as error:
        raise DualTrainError(f"dual learner authority refused: {error}") from error
    try:
        data = data.expanduser().resolve(strict=True)
        validation = validation.expanduser().resolve(strict=True)
        producer_checkpoint = producer_checkpoint.expanduser().resolve(strict=True)
        meta = train_bc._preflight_a1_memmap_metadata(  # noqa: SLF001
            data, validation_manifest_path=validation
        )
        if meta is None:
            raise DualTrainError("corpus is not an audited A1 memmap")
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
        raise DualTrainError(f"dual-arm input verification failed: {error}") from error
    identity = (bound.get("arm_id"), bound.get("subset_id"))
    if bound.get("dual_arm") is not True or identity not in ALLOWED_IDENTITIES:
        raise DualTrainError(f"unauthorized dual-arm identity: {identity}")
    bound_recipe = bound.get("learner_training_recipe")
    topology = authority.get("topology")
    if (
        not isinstance(bound_recipe, dict)
        or topology not in learner_contract.TOPOLOGIES.values()
        or topology.get("global_batch_size") != GLOBAL_BATCH
        or topology.get("data_format") != "memmap"
        or topology.get("ddp_shard_data") is not False
        or topology.get("fsdp") is not False
    ):
        raise DualTrainError("dual learner topology is not an authorized direct memmap DDP shape")
    recipe = dict(bound_recipe)
    recipe.update(
        {
            "world_size": topology["world_size"],
            "batch_size": topology["local_batch_size"],
            "grad_accum_steps": topology["grad_accum_steps"],
            "global_batch_size": topology["global_batch_size"],
            "ddp_shard_data": topology["ddp_shard_data"],
        }
    )
    producer_ref = _file_ref(producer_checkpoint, where="producer checkpoint")
    if producer_ref["sha256"] != bound.get("producer_checkpoint_sha256"):
        raise DualTrainError("producer checkpoint bytes differ from selected-game lineage")
    selected_meta = meta["selected_game_seed_manifest"]
    audit_meta = meta["a1_post_wave_audit"]
    assert isinstance(selected_meta, dict) and isinstance(audit_meta, dict)
    training_rows = int(meta["row_count"]) - int(holdout["validation_row_count"])
    if training_rows <= 0:
        raise DualTrainError("dual-arm corpus has no training rows")
    artifact_refs = {
        "corpus_meta": _file_ref(data / "corpus_meta.json", where="corpus metadata"),
        "selected_manifest": _file_ref(Path(str(selected_meta["path"])), where="selection"),
        "audit": _file_ref(Path(str(audit_meta["path"])), where="audit"),
        "validation": _file_ref(validation, where="validation holdout"),
        "producer": producer_ref,
    }
    expected_authority = {
        "identity": (authority.get("arm_id"), authority.get("subset_id")),
        "contract_sha256": authority.get("generation_contract_sha256"),
        "recipe": authority.get("recipe"),
        "objective": authority.get("objective"),
        "inputs": authority.get("inputs"),
        "payload_inventory_sha256": authority.get("payload_inventory_sha256"),
        "data_fingerprint": authority.get("data_fingerprint"),
        "row_counts": authority.get("row_counts"),
        "selected_game_seed_set_sha256": authority.get("selected_game_seed_set_sha256"),
        "training_game_seed_set_sha256": authority.get("training_game_seed_set_sha256"),
        "validation_game_seed_set_sha256": authority.get("validation_game_seed_set_sha256"),
        "trainer_report_bindings": authority.get("trainer_report_bindings"),
    }
    actual_authority = {
        "identity": identity,
        "contract_sha256": holdout["a1_contract_sha256"],
        "recipe": bound_recipe,
        "objective": bound["learner_value_objective"],
        "inputs": artifact_refs,
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "data_fingerprint": train_bc._training_data_fingerprint(str(data), "memmap"),  # noqa: SLF001
        "row_counts": {
            "corpus": int(meta["row_count"]),
            "training": training_rows,
            "validation": int(holdout["validation_row_count"]),
        },
        "selected_game_seed_set_sha256": bound["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": bound["training_game_seed_set_sha256"],
        "validation_game_seed_set_sha256": holdout["validation_game_seed_set_sha256"],
        "trainer_report_bindings": {
            "learner_code_sha256": bound["learner_code_sha256"],
            "runtime_code_tree_sha256": bound["runtime_code_tree_sha256"],
        },
    }
    if actual_authority != expected_authority:
        raise DualTrainError(
            "audited corpus/runtime differs from reviewed immutable learner lock"
        )
    verified = {
        "identity": identity,
        "arm_id": identity[0],
        "subset_id": identity[1],
        "contract_sha256": holdout["a1_contract_sha256"],
        "data": data,
        "learner_lock": _file_ref(learner_lock, where="reviewed learner lock"),
        "reviewed_lock_file_sha256": reviewed_lock_file_sha256,
        "corpus_meta": artifact_refs["corpus_meta"],
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "data_fingerprint": train_bc._training_data_fingerprint(str(data), "memmap"),  # noqa: SLF001
        "selected_manifest": artifact_refs["selected_manifest"],
        "audit": artifact_refs["audit"],
        "validation": artifact_refs["validation"],
        "producer": producer_ref,
        "recipe": recipe,
        "bound_recipe": bound_recipe,
        "topology": topology,
        "objective": bound["learner_value_objective"],
        "learner_code_sha256": bound["learner_code_sha256"],
        "runtime_code_tree_sha256": bound["runtime_code_tree_sha256"],
        "selected_game_seed_set_sha256": bound["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": bound["training_game_seed_set_sha256"],
        "validation_game_seed_set_sha256": holdout["validation_game_seed_set_sha256"],
        "corpus_rows": int(meta["row_count"]),
        "training_rows": training_rows,
        "validation_rows": int(holdout["validation_row_count"]),
        # Re-express the reviewed dual-learner runtime as the lock-shaped code
        # inventory consumed by the existing diagnostic-ablation binder.  This
        # does not authorize recipe drift by itself; the caller must still bind
        # an allowlisted override to the reviewed raw learner-lock digest.
        "ablation_code_lock": {
            "provenance": {
                "learner_code": [
                    {"path": str((_REPO_ROOT / "tools/train_bc.py").resolve())}
                ],
                "runtime_code_tree": [
                    {"path": str((_REPO_ROOT / record["path"]).resolve())}
                    for record in authority["runtime"]
                ],
            }
        },
    }
    if curriculum_parent_receipt is not None:
        # Reuse train_bc's fail-closed receipt/checkpoint/producer validation so
        # the second curriculum dose cannot name an arbitrary warm start.
        class _Args:
            pass

        parent_args = _Args()
        parent_args.a1_curriculum_parent_receipt = str(curriculum_parent_receipt)
        # Replay the original claim/completion and every referenced input/output
        # byte before allowing train_bc's narrower producer/init check.
        parent_receipt = verify_receipt(curriculum_parent_receipt)
        # The receipt has already replayed its claim/completion and every bound
        # input/output byte. Do not require the current executor checkout to
        # reproduce an older diagnostic executor's source lock; verifier-only
        # upgrades must not invalidate a completed curriculum parent.
        parent_args.init_checkpoint = parent_receipt["outputs"]["checkpoint"]["path"]
        parent_args.init_checkpoint_sha256 = parent_receipt["outputs"]["checkpoint"]["sha256"]
        parent = train_bc._validate_a1_curriculum_parent(parent_args, bound)  # noqa: SLF001
        assert parent is not None
        verified["curriculum_parent"] = parent
        try:
            parent_lineage = lineage.validate_lineage_dose(
                parent_receipt.get("lineage_dose")
            )
        except lineage.LineageDoseError as error:
            raise DualTrainError(
                f"invalid curriculum parent lineage dose: {error}"
            ) from error
        verified["curriculum_declaration"] = {
            "schema_version": lineage.CURRICULUM_DECLARATION_SCHEMA,
            "kind": "sequential_checkpoint_curriculum",
            "parent_receipt_path": parent["receipt_path"],
            "parent_receipt_sha256": parent["receipt_sha256"],
            "parent_arm_id": parent["parent_arm_id"],
            "parent_subset_id": parent["parent_subset_id"],
            "parent_checkpoint": parent["parent_checkpoint"],
            "generation_producer_sha256": parent["generation_producer_sha256"],
            "parent_lineage_dose": parent_lineage,
            "parent_cumulative_sampled_rows": parent_lineage["cumulative_sampled_rows"],
            "parent_cumulative_optimizer_steps": parent_lineage[
                "cumulative_optimizer_steps"
            ],
            "child_arm_id": verified["arm_id"],
            "child_subset_id": verified["subset_id"],
        }
    return verified


def bind_learner_ablation(
    verified: dict[str, Any],
    *,
    ablation_id: str,
    overrides_json: str,
    reviewed_code_tree_sha256: str,
) -> dict[str, Any]:
    """Bind existing learner-only knobs without weakening dual-arm authority."""

    try:
        overrides = json.loads(overrides_json)
    except json.JSONDecodeError as error:
        raise DualTrainError(f"invalid corrective overrides JSON: {error}") from error
    if not isinstance(overrides, dict) or not overrides:
        raise DualTrainError("dual corrective overrides must be a nonempty object")
    forbidden = set(overrides) - DUAL_CORRECTIVE_ABLATION_FIELDS
    if forbidden:
        raise DualTrainError(
            "dual corrective ablation only permits epochs, lr, and loser_sample_weight; "
            f"got forbidden fields {sorted(forbidden)}"
        )

    original_bound_recipe = verified["bound_recipe"]
    ablation_input = dict(verified)
    ablation_input.update(
        {
            "lock": verified["ablation_code_lock"],
            "lock_file_sha256": verified["learner_lock"]["sha256"],
        }
    )
    result = one_dose.bind_learner_ablation(
        ablation_input,
        ablation_id=ablation_id,
        overrides_json=overrides_json,
        reviewed_code_tree_sha256=reviewed_code_tree_sha256,
    )
    # The child report must continue to distinguish the immutable generation
    # recipe from the effective topology+ablation recipe.
    result["bound_recipe"] = original_bound_recipe
    return result


def build_command(
    verified: dict[str, Any], *, python: Path, checkpoint: Path, report: Path
) -> list[str]:
    base = one_dose.build_train_command(
        {
            "recipe": verified["recipe"],
            "producer": verified["producer"],
            "data_path": verified["data"],
            "validation_path": Path(verified["validation"]["path"]),
            "payload_inventory_sha256": verified["payload_inventory_sha256"],
            **(
                {}
                if verified.get("learner_ablation") is None
                else {"learner_ablation": verified["learner_ablation"]}
            ),
        },
        python=python,
        checkpoint=checkpoint,
        report=report,
    )
    trainer = base[1]
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={verified['topology']['world_size']}",
        trainer,
        *base[2:],
        "--a1-dual-learner-lock",
        verified["learner_lock"]["path"],
        "--a1-dual-reviewed-lock-file-sha256",
        verified["reviewed_lock_file_sha256"],
    ]
    if int(verified["recipe"].get("epochs", 1)) > 1:
        # Epoch-curve diagnostics must be one uninterrupted optimizer trajectory.
        # Persist every integer exposure with its exact Adam state, and sample the
        # opt-in optimizer/teacher telemetry without changing training math.
        command.extend(
            [
                "--save-each-epoch",
                "--train-diagnostics-every-batches",
                "100",
            ]
        )
    parent = verified.get("curriculum_parent")
    if parent is not None:
        init_index = command.index("--init-checkpoint") + 1
        command[init_index] = str(parent["parent_checkpoint"]["path"])
        command.extend(
            ["--a1-curriculum-parent-receipt", str(parent["receipt_path"])]
        )
    return command


def _gpu_ids(verified: dict[str, Any]) -> tuple[int, ...]:
    return tuple(range(int(verified["topology"]["world_size"])))


def _environment(verified: dict[str, Any] | None = None) -> dict[str, str]:
    gpu_ids = tuple(range(8)) if verified is None else _gpu_ids(verified)
    environment = one_dose._child_environment(0)  # noqa: SLF001
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    return environment


def _execution_binding(
    command: list[str], verified: dict[str, Any] | None = None
) -> dict[str, Any]:
    topology = learner_contract.TOPOLOGY if verified is None else verified["topology"]
    environment = _environment(verified)
    return {
        "schema_version": "a1-dual-arm-execution-binding-v1",
        "command_sha256": _digest(command),
        "environment": environment,
        "environment_sha256": _digest(environment),
        "gpu_ids": list(range(int(topology["world_size"]))),
        "world_size": topology["world_size"],
        "local_batch_size": topology["local_batch_size"],
        "grad_accum_steps": topology["grad_accum_steps"],
        "global_batch_size": GLOBAL_BATCH,
    }


def _claim_identity(verified: dict[str, Any]) -> str:
    return _digest(
        {
            "contract_sha256": verified["contract_sha256"],
            "arm_id": verified["arm_id"],
            "subset_id": verified["subset_id"],
            "selected_game_seed_set_sha256": verified[
                "selected_game_seed_set_sha256"
            ],
            "learner_training_recipe_sha256": _digest(verified["recipe"]),
            "learner_code_sha256": verified["learner_code_sha256"],
            "runtime_code_tree_sha256": verified["runtime_code_tree_sha256"],
            "executor_sha256": _sha256(Path(__file__).resolve()),
            "learner_lock_sha256": verified["learner_lock"]["sha256"],
            "curriculum_parent_receipt_sha256": (
                None
                if verified.get("curriculum_parent") is None
                else verified["curriculum_parent"]["receipt_sha256"]
            ),
            "learner_ablation_sha256": (
                None
                if verified.get("learner_ablation") is None
                else _digest(verified["learner_ablation"])
            ),
        }
    )


def _claim_path(verified: dict[str, Any]) -> Path:
    digest = _claim_identity(verified).removeprefix("sha256:")
    return Path(verified["data"]).parent / ".a1-dual-arm-training-claims" / f"{digest}.json"


def _validate_output_paths(
    verified: dict[str, Any], *, checkpoint: Path, report: Path, receipt: Path
) -> None:
    claim = _claim_path(verified)
    paths = {
        "checkpoint": checkpoint,
        "optimizer": Path(str(checkpoint) + ".optimizer.pt"),
        "report": report,
        "receipt": receipt,
        "claim": claim,
        "completion": _completion_path(claim),
    }
    canonical = {name: path.expanduser().resolve(strict=False) for name, path in paths.items()}
    if len(set(canonical.values())) != len(canonical):
        raise DualTrainError("dual learner outputs/claims must be distinct paths")
    input_paths = {
        Path(ref["path"]).resolve(strict=True)
        for key in (
            "learner_lock", "corpus_meta", "selected_manifest", "audit",
            "validation", "producer",
        )
        for ref in (verified[key],)
    }
    if verified.get("curriculum_parent") is not None:
        input_paths.add(
            Path(verified["curriculum_parent"]["receipt_path"]).resolve(strict=True)
        )
        input_paths.add(
            Path(
                verified["curriculum_parent"]["parent_checkpoint"]["path"]
            ).resolve(strict=True)
        )
    overlap = input_paths.intersection(canonical.values())
    if overlap:
        raise DualTrainError(f"dual learner output aliases immutable input: {sorted(map(str, overlap))}")


@contextmanager
def _identity_lock(verified: dict[str, Any]):
    path = _claim_path(verified).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "r+b", buffering=0) as handle:
        info = os.fstat(handle.fileno())
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise DualTrainError("dual-arm identity lock ownership/mode is unsafe")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DualTrainError("dual-arm learner identity is already active") from error
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_report_binding(report: Path) -> dict[str, Any] | None:
    if not report.is_file():
        return None
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    binding = payload.get(REPORT_BINDING_FIELD) if isinstance(payload, dict) else None
    return binding if isinstance(binding, dict) else None


def _completion_path(claim: Path) -> Path:
    return claim.with_name(claim.stem + ".complete.json")


def _quarantine_incomplete(
    *, claim: Path, checkpoint: Path, report: Path, receipt: Path
) -> dict[str, Any]:
    completion = _completion_path(claim)
    candidates = [claim, completion, checkpoint, Path(str(checkpoint) + ".optimizer.pt"), report]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {"quarantined": []}
    root = claim.parent / "quarantine" / claim.stem
    root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while (root / f"recovery-{attempt:04d}.json").exists():
        attempt += 1
    moved: list[dict[str, str]] = []
    for index, source in enumerate(existing):
        if source.is_symlink() or not source.is_file():
            raise DualTrainError(f"cannot quarantine unsafe incomplete artifact: {source}")
        destination_root = source.parent / ".a1-dual-quarantine" / claim.stem
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = (
            destination_root / f"attempt-{attempt:04d}-{index:02d}-{source.name}"
        )
        os.replace(source, destination)
        _fsync_dir(source.parent)
        _fsync_dir(destination_root)
        moved.append(
            {
                "original_path": str(source),
                "quarantined_path": str(destination),
                "sha256": _sha256(destination),
            }
        )
    recovery = {
        "schema_version": "a1-dual-arm-training-recovery-v1",
        "status": "quarantined_incomplete_attempt",
        "claim_identity": claim.stem,
        "receipt_target": str(receipt),
        "artifacts": moved,
        "recovered_unix_ns": time.time_ns(),
    }
    recovery["recovery_sha256"] = _digest(recovery)
    _write_new(root / f"recovery-{attempt:04d}.json", recovery)
    return recovery


def _write_attempt_failure(
    claim: Path, *, command: list[str], error: BaseException, returncode: int | None
) -> None:
    root = claim.parent / "attempt-receipts" / claim.stem
    root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while (root / f"failed-{attempt:04d}.json").exists():
        attempt += 1
    value = {
        "schema_version": "a1-dual-arm-training-attempt-receipt-v1",
        "status": "failed",
        "claim_snapshot_sha256": _sha256(claim),
        "command_sha256": _digest(command),
        "returncode": returncode,
        "failure": f"{type(error).__name__}: {error}",
        "finished_unix_ns": time.time_ns(),
    }
    value["attempt_receipt_sha256"] = _digest(value)
    _write_new(root / f"failed-{attempt:04d}.json", value)


def _bind_report(
    report: Path,
    binding: dict[str, Any],
    lineage_dose: dict[str, Any],
    curriculum_declaration: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DualTrainError(f"cannot read training report: {error}") from error
    if (
        not isinstance(payload, dict)
        or REPORT_BINDING_FIELD in payload
        or "a1_lineage_dose" in payload
        or "a1_curriculum_declaration" in payload
    ):
        raise DualTrainError(
            "training report is malformed or pre-populated executor provenance"
        )
    payload[REPORT_BINDING_FIELD] = binding
    payload["a1_lineage_dose"] = lineage.validate_lineage_dose(lineage_dose)
    payload["a1_curriculum_declaration"] = curriculum_declaration
    temporary = report.with_name(f".{report.name}.bind.{os.getpid()}.{time.time_ns()}")
    _write_new(temporary, payload)
    os.chmod(temporary, 0o600)
    os.replace(temporary, report)
    return payload


def _lineage_dose(verified: dict[str, Any]) -> dict[str, Any]:
    if verified["recipe"].get("resume_optimizer") is not False:
        raise DualTrainError(
            "canonical dual-arm lineage requires a fresh optimizer per dose"
        )
    completed_epochs = int(verified["recipe"].get("epochs", 1))
    steps = math.ceil(int(verified["training_rows"]) / GLOBAL_BATCH) * completed_epochs
    sampled_rows = int(verified["training_rows"]) * completed_epochs
    parent = verified.get("curriculum_parent")
    try:
        if parent is None:
            return lineage.direct_lineage_dose(
                declared_producer_sha256=verified["producer"]["sha256"],
                init_checkpoint_sha256=verified["producer"]["sha256"],
                current_sampled_rows=sampled_rows,
                current_optimizer_steps=steps,
            )
        declaration = verified.get("curriculum_declaration")
        if not isinstance(declaration, dict):
            raise lineage.LineageDoseError(
                "curriculum parent is present without a typed declaration"
            )
        return lineage.curriculum_lineage_dose(
            declared_producer_sha256=verified["producer"]["sha256"],
            init_checkpoint_sha256=parent["parent_checkpoint"]["sha256"],
            parent_receipt_sha256=parent["receipt_sha256"],
            parent_lineage_dose=declaration["parent_lineage_dose"],
            current_sampled_rows=sampled_rows,
            current_optimizer_steps=steps,
        )
    except lineage.LineageDoseError as error:
        raise DualTrainError(f"invalid learner lineage dose: {error}") from error


def verify_outputs(
    *,
    verified: dict[str, Any],
    checkpoint: Path,
    report: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    for path in (checkpoint, optimizer, report):
        if not path.is_file() or path.stat().st_size <= 0:
            raise DualTrainError(f"missing training output: {path}")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DualTrainError(f"cannot parse training report: {error}") from error
    recipe = verified["recipe"]
    completed_epochs = int(recipe.get("epochs", 1))
    steps_per_epoch = math.ceil(int(verified["training_rows"]) / GLOBAL_BATCH)
    steps = steps_per_epoch * completed_epochs
    lineage_dose = _lineage_dose(verified)
    expected = {
        "arch": "entity_graph",
        "hidden_size": 640,
        "graph_layers": 6,
        "attention_heads": 8,
        "graph_dropout": 0.05,
        "world_size": verified["topology"]["world_size"],
        "batch_size": verified["topology"]["local_batch_size"],
        "ddp_shard_data": False,
        "steps_completed": steps,
        "total_training_steps": steps,
        "epochs": completed_epochs,
        "max_steps": 0,
        "samples": verified["corpus_rows"],
        "global_samples": verified["corpus_rows"],
        "train_samples": verified["training_rows"],
        "validation_samples": verified["validation_rows"],
        "data": str(verified["data"]),
        "data_format": "memmap",
        "data_fingerprint": verified["data_fingerprint"],
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "checkpoint": str(checkpoint),
        "init_checkpoint": (
            verified["producer"]["path"]
            if verified.get("curriculum_parent") is None
            else verified["curriculum_parent"]["parent_checkpoint"]["path"]
        ),
        "init_checkpoint_sha256": (
            verified["producer"]["sha256"]
            if verified.get("curriculum_parent") is None
            else verified["curriculum_parent"]["parent_checkpoint"]["sha256"]
        ),
        "a1_curriculum_parent": verified.get("curriculum_parent"),
        "a1_curriculum_declaration": verified.get("curriculum_declaration"),
        "a1_lineage_dose": lineage_dose,
        "a1_contract_sha256": verified["contract_sha256"],
        "a1_selected_game_seed_set_sha256": verified[
            "selected_game_seed_set_sha256"
        ],
        "a1_training_game_seed_set_sha256": verified[
            "training_game_seed_set_sha256"
        ],
        "validation_game_seed_set_sha256": verified[
            "validation_game_seed_set_sha256"
        ],
        "input_validation_game_seed_manifest": verified["validation"]["path"],
        "input_validation_game_seed_manifest_sha256": verified["validation"]["sha256"],
        "a1_bound_learner_training_recipe": verified["bound_recipe"],
        "a1_bound_learner_value_objective": verified["objective"],
        # train_bc preserves this legacy field as the immutable corpus-bound
        # recipe digest. Diagnostic overrides are authenticated separately by
        # the effective-recipe and learner-ablation fields below.
        "a1_learner_training_recipe_sha256": _digest(verified["bound_recipe"]),
        "a1_learner_code_sha256": verified["learner_code_sha256"],
        "a1_runtime_code_tree_sha256": verified["runtime_code_tree_sha256"],
        "a1_memmap_payload_inventory_sha256": verified[
            "payload_inventory_sha256"
        ],
        "mask_hidden_info": True,
        "require_35m_model": True,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "amp": "bf16",
        REPORT_BINDING_FIELD: binding,
    }
    if verified["recipe"] != verified["bound_recipe"]:
        topology_authorization = {
            "schema_version": "a1-dual-learner-topology-authorization-v1",
            "learner_lock": verified["learner_lock"]["path"],
            "learner_lock_file_sha256": verified["reviewed_lock_file_sha256"],
            "topology": verified["topology"],
            "effective_recipe": verified["recipe"],
            "effective_recipe_sha256": _digest(verified["recipe"]),
        }
        expected.update(
            {
                "a1_effective_learner_training_recipe": verified["recipe"],
                "a1_effective_learner_training_recipe_sha256": _digest(
                    verified["recipe"]
                ),
                "a1_learner_topology_authorization": topology_authorization,
            }
        )
    learner_ablation = verified.get("learner_ablation")
    if learner_ablation is not None:
        expected.update(
            {
                "a1_learner_ablation": learner_ablation,
                "diagnostic_only": True,
                "promotion_eligible": False,
            }
        )
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if drift:
        raise DualTrainError(f"dual-arm training report invariant drift: {drift}")
    parameter_count = payload.get("parameter_count")
    metrics = payload.get("metrics")
    value_training = payload.get("value_training")
    epoch_metrics = metrics if isinstance(metrics, list) else []
    finite_metrics = len(epoch_metrics) == completed_epochs and all(
        row.get("epoch") == index
        and all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
            for key in ("loss", "policy_loss", "value_loss")
        )
        for index, row in enumerate(epoch_metrics, start=1)
    )
    validation_metrics = [row.get("validation", {}) for row in epoch_metrics]
    valid_validation_metrics = len(validation_metrics) == completed_epochs and all(
        isinstance(value, dict)
        and value.get("samples") == verified["validation_rows"]
        and isinstance(value.get("loss"), (int, float))
        and not isinstance(value.get("loss"), bool)
        and math.isfinite(float(value["loss"]))
        for value in validation_metrics
    )
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or not 30_000_000 <= parameter_count <= 40_000_000
        or not finite_metrics
        or not valid_validation_metrics
        or not isinstance(value_training, dict)
        or value_training.get("optimizer_steps") != steps
        or value_training.get("completed_epochs") != completed_epochs
        or value_training.get("a1_contract_sha256") != verified["contract_sha256"]
        or value_training.get("a1_selected_game_seed_set_sha256")
        != verified["selected_game_seed_set_sha256"]
        or value_training.get("a1_training_game_seed_set_sha256")
        != verified["training_game_seed_set_sha256"]
        or value_training.get("a1_learner_training_recipe_sha256")
        != _digest(verified["bound_recipe"])
        or value_training.get("a1_memmap_payload_inventory_sha256")
        != verified["payload_inventory_sha256"]
        or (
            learner_ablation is not None
            and value_training.get("learner_ablation") != learner_ablation
        )
        or "scalar" not in value_training.get("trained_value_readouts", [])
    ):
        raise DualTrainError(
            "dual-arm report lacks the complete authenticated 35M scalar-MSE trajectory"
        )
    outputs = {
        "checkpoint": _file_ref(checkpoint, where="candidate checkpoint"),
        "optimizer": _file_ref(optimizer, where="optimizer sidecar"),
        "report": _file_ref(report, where="training report"),
        "steps_completed": steps,
        "sampled_rows": lineage_dose["current_sampled_rows"],
        "lineage_dose": lineage_dose,
    }
    if completed_epochs > 1:
        epoch_outputs: dict[str, dict[str, Any]] = {}
        for epoch in range(1, completed_epochs + 1):
            epoch_checkpoint = train_bc._epoch_checkpoint_path(str(checkpoint), epoch)  # noqa: SLF001
            epoch_optimizer = Path(str(epoch_checkpoint) + ".optimizer.pt")
            epoch_outputs[str(epoch)] = {
                "exposures": float(epoch),
                "checkpoint": _file_ref(
                    epoch_checkpoint, where=f"epoch {epoch} candidate checkpoint"
                ),
                "optimizer": _file_ref(
                    epoch_optimizer, where=f"epoch {epoch} optimizer sidecar"
                ),
                "validation": epoch_metrics[epoch - 1]["validation"],
            }
        outputs["epoch_checkpoints"] = epoch_outputs
    return outputs


def verify_receipt(
    path: Path, *, verified: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        path = path.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DualTrainError(f"cannot load training receipt: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != RECEIPT_SCHEMA:
        raise DualTrainError("dual-arm training receipt schema drift")
    stated = payload.get("receipt_sha256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if stated != _digest(unhashed) or payload.get("status") != "complete":
        raise DualTrainError("dual-arm training receipt digest/status drift")
    claim_ref = payload.get("claim")
    completion_ref = payload.get("claim_completion")
    if (
        not isinstance(claim_ref, dict)
        or set(claim_ref) != {"path", "sha256"}
        or not isinstance(completion_ref, dict)
        or set(completion_ref) != {"path", "sha256"}
        or _file_ref(Path(claim_ref["path"]), where="receipt claim") != claim_ref
        or _file_ref(Path(completion_ref["path"]), where="receipt claim completion")
        != completion_ref
    ):
        raise DualTrainError("dual-arm receipt claim provenance drift")
    completion = json.loads(Path(completion_ref["path"]).read_text(encoding="utf-8"))
    if (
        not isinstance(completion, dict)
        or completion.get("schema_version") != CLAIM_SCHEMA
        or completion.get("status") != "complete"
        or completion.get("claim") != claim_ref
        or completion.get("claim_identity_sha256")
        != payload.get("claim_identity_sha256")
        or completion.get("receipt") != str(path)
        or completion.get("outputs") != payload.get("outputs")
        or completion.get("lineage_dose") != payload.get("lineage_dose")
        or completion.get("command_sha256") != payload.get("command_sha256")
        or completion.get("execution_binding") != payload.get("execution_binding")
        or completion.get("gpu_names") != payload.get("gpu_names")
    ):
        raise DualTrainError("dual-arm claim completion disagrees with receipt")
    try:
        receipt_lineage = lineage.validate_lineage_dose(payload.get("lineage_dose"))
    except lineage.LineageDoseError as error:
        raise DualTrainError(f"dual-arm receipt lineage dose drift: {error}") from error
    if payload.get("outputs", {}).get("lineage_dose") != receipt_lineage:
        raise DualTrainError("dual-arm receipt/output lineage dose disagreement")
    for group in ("inputs", "outputs"):
        values = payload.get(group)
        if not isinstance(values, dict):
            raise DualTrainError(f"receipt {group} is malformed")
        for key, ref in values.items():
            if key in {"steps_completed", "payload_inventory_sha256"}:
                continue
            if isinstance(ref, dict) and set(ref) == {"path", "sha256"}:
                if _file_ref(Path(ref["path"]), where=f"receipt {group}.{key}") != ref:
                    raise DualTrainError(f"receipt {group}.{key} drift")
    if verified is not None:
        if (
            payload.get("claim_identity_sha256") != _claim_identity(verified)
            or (payload.get("arm_id"), payload.get("subset_id"))
            != (verified["arm_id"], verified["subset_id"])
            or payload.get("contract_sha256") != verified["contract_sha256"]
        ):
            raise DualTrainError("completed receipt belongs to different verified inputs")
        outputs = payload["outputs"]
        replayed = verify_outputs(
            verified=verified,
            checkpoint=Path(outputs["checkpoint"]["path"]),
            report=Path(outputs["report"]["path"]),
            binding=payload["execution_binding"],
        )
        if replayed != outputs:
            raise DualTrainError("completed receipt output invariants no longer replay")
    return payload


def _publish_completion_and_receipt(
    *,
    verified: dict[str, Any],
    command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    claim: Path,
    binding: dict[str, Any],
    gpu_names: list[str],
) -> dict[str, Any]:
    output_refs = verify_outputs(
        verified=verified,
        checkpoint=checkpoint,
        report=report,
        binding=binding,
    )
    claim_ref = _file_ref(claim, where="training claim")
    completion = _completion_path(claim)
    completion_payload = {
        "schema_version": CLAIM_SCHEMA,
        "status": "complete",
        "claim": claim_ref,
        "claim_identity_sha256": _claim_identity(verified),
        "receipt": str(receipt),
        "command_sha256": _digest(command),
        "execution_binding": binding,
        "gpu_names": gpu_names,
        "outputs": output_refs,
        "lineage_dose": output_refs["lineage_dose"],
        "finished_unix_ns": time.time_ns(),
    }
    if completion.exists():
        existing_completion = json.loads(completion.read_text(encoding="utf-8"))
        stable_fields = {
            key: completion_payload[key]
            for key in (
                "schema_version",
                "status",
                "claim",
                "claim_identity_sha256",
                "receipt",
                "command_sha256",
                "execution_binding",
                "gpu_names",
                "outputs",
                "lineage_dose",
            )
        }
        if any(existing_completion.get(key) != value for key, value in stable_fields.items()):
            raise DualTrainError("existing claim completion disagrees with verified outputs")
        completion_payload = existing_completion
    else:
        _write_new(completion, completion_payload)
    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "claim_identity_sha256": _claim_identity(verified),
        "claim": claim_ref,
        "claim_completion": _file_ref(completion, where="training claim completion"),
        "arm_id": verified["arm_id"],
        "subset_id": verified["subset_id"],
        "contract_sha256": verified["contract_sha256"],
        "inputs": {
            "corpus_meta": verified["corpus_meta"],
            "learner_lock": verified["learner_lock"],
            "selected_manifest": verified["selected_manifest"],
            "audit": verified["audit"],
            "validation": verified["validation"],
            "producer": verified["producer"],
            "executor": _file_ref(Path(__file__), where="dual learner executor"),
            "payload_inventory_sha256": verified["payload_inventory_sha256"],
            **(
                {}
                if verified.get("learner_ablation") is None
                else {"learner_ablation": verified["learner_ablation"]}
            ),
            **(
                {}
                if verified.get("curriculum_parent") is None
                else {
                    "curriculum_parent": verified["curriculum_parent"],
                    "curriculum_declaration": verified["curriculum_declaration"],
                }
            ),
        },
        "execution_binding": binding,
        "gpu_names": gpu_names,
        "command": command,
        "command_sha256": _digest(command),
        "outputs": output_refs,
        "lineage_dose": output_refs["lineage_dose"],
        "finished_unix_ns": completion_payload["finished_unix_ns"],
    }
    receipt_payload["receipt_sha256"] = _digest(receipt_payload)
    if receipt.exists():
        return verify_receipt(receipt, verified=verified)
    _write_new(receipt, receipt_payload)
    return verify_receipt(receipt, verified=verified)


def execute(
    *,
    verified: dict[str, Any],
    command: list[str],
    checkpoint: Path,
    report: Path,
    receipt: Path,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    probe: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    claim = _claim_path(verified)
    _validate_output_paths(
        verified, checkpoint=checkpoint, report=report, receipt=receipt
    )
    claim.parent.mkdir(parents=True, exist_ok=True)
    binding = _execution_binding(command, verified)
    outputs = (checkpoint, Path(str(checkpoint) + ".optimizer.pt"), report)
    # Completed receipt replay is read-only and must remain available even when
    # the training GPUs are occupied.  A second check under the identity lock
    # below closes the publication race for a receipt appearing concurrently.
    if receipt.exists():
        return verify_receipt(receipt, verified=verified)
    with _identity_lock(verified), ExitStack() as stack:
        names: list[str] = []
        for index, gpu in enumerate(_gpu_ids(verified)):
            stack.enter_context(one_dose._physical_gpu_lock(gpu))  # noqa: SLF001
            names.append(
                probe(gpu)
                if probe is not None
                else one_dose._probe_b200(  # noqa: SLF001
                    gpu,
                    mps_probe=(
                        one_dose._active_mps_processes  # noqa: SLF001
                        if index == 0
                        else lambda: []
                    )
                )
            )
        if receipt.exists():
            return verify_receipt(receipt, verified=verified)

        existing_binding = _read_report_binding(report)
        if claim.exists() and all(path.is_file() for path in outputs) and existing_binding:
            if existing_binding != binding:
                raise DualTrainError(
                    "completed child outputs bind a different command/environment"
                )
            try:
                return _publish_completion_and_receipt(
                    verified=verified,
                    command=command,
                    checkpoint=checkpoint,
                    report=report,
                    receipt=receipt,
                    claim=claim,
                    binding=binding,
                    gpu_names=names,
                )
            except DualTrainError:
                _quarantine_incomplete(
                    claim=claim,
                    checkpoint=checkpoint,
                    report=report,
                    receipt=receipt,
                )

        if claim.exists() or _completion_path(claim).exists() or any(
            path.exists() for path in outputs
        ):
            _quarantine_incomplete(
                claim=claim,
                checkpoint=checkpoint,
                report=report,
                receipt=receipt,
            )

        claim_payload = {
            "schema_version": CLAIM_SCHEMA,
            "status": "claimed",
            "claim_identity_sha256": _claim_identity(verified),
            "arm_id": verified["arm_id"],
            "subset_id": verified["subset_id"],
            "command_sha256": _digest(command),
            "receipt": str(receipt),
            "started_unix_ns": time.time_ns(),
        }
        _write_new(claim, claim_payload)
        returncode: int | None = None
        try:
            result = runner(
                command,
                cwd=str(_REPO_ROOT),
                env=_environment(verified),
                check=False,
                preexec_fn=one_dose._raise_nofile_limit,  # noqa: SLF001
            )
            returncode = int(result.returncode)
            if returncode != 0:
                raise DualTrainError(f"torchrun exited nonzero: {returncode}")
            _bind_report(
                report,
                binding,
                _lineage_dose(verified),
                verified.get("curriculum_declaration"),
            )
            return _publish_completion_and_receipt(
                verified=verified,
                command=command,
                checkpoint=checkpoint,
                report=report,
                receipt=receipt,
                claim=claim,
                binding=binding,
                gpu_names=names,
            )
        except BaseException as error:
            _write_attempt_failure(
                claim, command=command, error=error, returncode=returncode
            )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--learner-lock", type=Path, required=True)
    parser.add_argument("--reviewed-lock-file-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--producer-checkpoint", type=Path, required=True)
    parser.add_argument("--curriculum-parent-receipt", type=Path)
    parser.add_argument("--ablation-id", default="")
    parser.add_argument("--recipe-overrides-json", default="")
    parser.add_argument("--ablation-code-tree-sha256", default="")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        python = one_dose._lexical_python_executable(args.python)  # noqa: SLF001
        verified = verify_inputs(
            learner_lock=args.learner_lock,
            reviewed_lock_file_sha256=args.reviewed_lock_file_sha256,
            data=args.data,
            validation=args.validation_manifest,
            producer_checkpoint=args.producer_checkpoint,
            curriculum_parent_receipt=args.curriculum_parent_receipt,
        )
        ablation_values = (
            args.ablation_id,
            args.recipe_overrides_json,
            args.ablation_code_tree_sha256,
        )
        if any(ablation_values) and not all(ablation_values):
            raise DualTrainError(
                "--ablation-id, --recipe-overrides-json, and "
                "--ablation-code-tree-sha256 must be supplied together"
            )
        if all(ablation_values):
            verified = bind_learner_ablation(
                verified,
                ablation_id=args.ablation_id,
                overrides_json=args.recipe_overrides_json,
                reviewed_code_tree_sha256=args.ablation_code_tree_sha256,
            )
        checkpoint = args.checkpoint.expanduser().resolve(strict=False)
        report = args.report.expanduser().resolve(strict=False)
        receipt = args.receipt.expanduser().resolve(strict=False)
        command = build_command(
            verified, python=python, checkpoint=checkpoint, report=report
        )
        _validate_output_paths(
            verified, checkpoint=checkpoint, report=report, receipt=receipt
        )
        plan = {
            "schema_version": PLAN_SCHEMA,
            "mode": "go" if args.go else "dry-run",
            "arm_id": verified["arm_id"],
            "subset_id": verified["subset_id"],
            "claim_identity_sha256": _claim_identity(verified),
            "world_size": verified["topology"]["world_size"],
            "local_batch_size": verified["topology"]["local_batch_size"],
            "grad_accum_steps": verified["topology"]["grad_accum_steps"],
            "global_batch_size": GLOBAL_BATCH,
            "gpu_ids": list(_gpu_ids(verified)),
            "command": command,
            "command_sha256": _digest(command),
            "inputs": {
                "corpus_meta": verified["corpus_meta"],
                "learner_lock": verified["learner_lock"],
                "selected_manifest": verified["selected_manifest"],
                "audit": verified["audit"],
                "validation": verified["validation"],
                "producer": verified["producer"],
                **(
                    {}
                    if verified.get("curriculum_parent") is None
                    else {
                        "curriculum_parent": verified["curriculum_parent"],
                        "curriculum_declaration": verified[
                            "curriculum_declaration"
                        ],
                    }
                ),
                "executor": _file_ref(Path(__file__), where="dual learner executor"),
                **(
                    {}
                    if verified.get("learner_ablation") is None
                    else {"learner_ablation": verified["learner_ablation"]}
                ),
            },
            "outputs": {
                "checkpoint": str(checkpoint),
                "report": str(report),
                "receipt": str(receipt),
            },
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        if args.go:
            execute(
                verified=verified,
                command=command,
                checkpoint=checkpoint,
                report=report,
                receipt=receipt,
            )
        return 0
    except (DualTrainError, one_dose.ExecutorError, OSError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
