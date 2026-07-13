#!/usr/bin/env python3
"""Seal and optionally submit one topology-residual-only learner diagnostic.

The experiment starts from an explicitly selected parent, appends the reviewed
zero-output ``MODULE_TOPOLOGY_RESIDUAL`` upgrade, and trains only the eight
topology-adapter tensors.  Parent selection is a first-class immutable artifact:
the launcher never infers that a gather checkpoint was selected from a filename
or from its mere existence.  Two parent profiles are supported:

``direct_short_d6``
    The authenticated 128-step D6 checkpoint, used only after an explicit
    selection artifact says the gather treatment was rejected.

``sealed_selected_gather``
    A completed D6+gather diagnostic whose completion receipt replays exactly,
    used only after an explicit selection artifact names it as selected.

Preparation and verification do not launch compute.  ``execute --go`` reuses
the existing one-shot A1 systemd executor after every byte and recipe binding
has replayed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_corrected_policy_arm_execute as executor_base  # noqa: E402
from tools import a1_d6_gather_composition_arm as gather_arm  # noqa: E402
from tools import a1_d6_gather_composition_completion as gather_completion  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402


SCHEMA = "a1-topology-only-composition-arm-v1"
PARENT_SELECTION_SCHEMA = "a1-topology-parent-selection-v1"
RECEIPT_SCHEMA = "a1-topology-only-composition-execution-receipt-v1"
STATUS_SCHEMA = "a1-topology-only-composition-execution-status-v1"
CLAIM_SCHEMA = "a1-topology-only-composition-execution-claim-v1"
EXECUTOR_RELATIVE_PATH = "tools/a1_topology_only_composition_arm.py"
COMPLETION_RELATIVE_PATH = "tools/a1_topology_only_composition_completion.py"

PARENT_DIRECT_SHORT_D6 = "direct_short_d6"
PARENT_SELECTED_GATHER = "sealed_selected_gather"
PARENT_PROFILES = (PARENT_DIRECT_SHORT_D6, PARENT_SELECTED_GATHER)

WORLD_SIZE = gather_arm.WORLD_SIZE
LOCAL_BATCH_SIZE = gather_arm.SELECTED_LOCAL_BATCH_SIZE
GLOBAL_BATCH_SIZE = gather_arm.SELECTED_GLOBAL_BATCH_SIZE
ALLOWED_OPTIMIZER_STEPS = (gather_arm.SELECTED_OPTIMIZER_STEPS,)
OPTIMIZER_STEPS = gather_arm.SELECTED_OPTIMIZER_STEPS
TRUNK_LR_MULT = 4.0
ACTION_MODULE_LR_MULT = 1.0
VALUE_LR_MULT = 1.0
FREEZE_MODULES = (
    "trunk_base,action_encoder,policy_head,value_heads,target_gather,"
    "edge_policy,action_cross"
)
TRAINABLE_PREFIX = "topology_residual_adapter"
EXPECTED_TOPOLOGY_PARAMETERS = tuple(
    sorted(
        architecture_upgrade.ALLOWLIST[architecture_upgrade.MODULE_TOPOLOGY_RESIDUAL][
            "new_parameter_initialization"
        ]
    )
)
EXPECTED_TOPOLOGY_PARAMETER_COUNT = 823_040

SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            EXECUTOR_RELATIVE_PATH,
            COMPLETION_RELATIVE_PATH,
            "tools/a1_function_preserving_upgrade.py",
            "src/catan_zero/rl/relational_trunks.py",
        )
        + tuple(gather_arm.SOURCE_FILES)
    )
)


class TopologyCompositionError(RuntimeError):
    """The request is not the sealed topology-only experiment."""


def _digest(value: Any) -> str:
    return gather_arm.gather.corrected._digest(value)  # noqa: SLF001


def _file_ref(path: Path) -> dict[str, str]:
    try:
        return gather_arm.gather.corrected._file_ref(path)  # noqa: SLF001
    except (OSError, gather_arm.gather.corrected.ArmError) as error:
        raise TopologyCompositionError(
            f"cannot bind artifact {path}: {error}"
        ) from error


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        ref = _file_ref(path)
        value = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TopologyCompositionError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise TopologyCompositionError(f"{label} must be a JSON object")
    return value, ref


def _checkpoint_config(path: Path) -> dict[str, Any]:
    try:
        payload = gather_arm.gather._torch_load(path)  # noqa: SLF001
        return gather_arm.gather._effective_config(payload.get("config"))  # noqa: SLF001
    except (OSError, RuntimeError, gather_arm.gather.ArmError) as error:
        raise TopologyCompositionError(
            f"cannot authenticate parent architecture {path}: {error}"
        ) from error


def _require_parent_architecture(
    checkpoint: Path, *, gather_enabled: bool
) -> dict[str, Any]:
    config = _checkpoint_config(checkpoint)
    exact = {
        "state_trunk": "transformer",
        "action_target_gather": bool(gather_enabled),
        "topology_residual_adapter": False,
        "action_cross_attention_layers": 0,
        "edge_policy_head": False,
        "value_attention_pool": False,
    }
    drift = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in exact.items()
        if config.get(key) != expected
    }
    if drift:
        raise TopologyCompositionError(f"selected parent architecture drift: {drift}")
    return {
        **exact,
        "effective_config_sha256": _digest(config),
    }


def _direct_short_d6_parent(
    checkpoint: Path, report: Path, progress: Path
) -> dict[str, Any]:
    try:
        evidence = gather_arm._load_d6_parent(checkpoint, report, progress)  # noqa: SLF001
    except gather_arm.CompositionArmError as error:
        raise TopologyCompositionError(str(error)) from error
    if evidence.get("parent_profile") != "selected_short_d6":
        raise TopologyCompositionError(
            "direct topology parent must be the authenticated short-D6 profile"
        )
    architecture = _require_parent_architecture(
        Path(evidence["checkpoint"]["path"]), gather_enabled=False
    )
    parent = {
        "parent_profile": PARENT_DIRECT_SHORT_D6,
        "checkpoint": evidence["checkpoint"],
        "report": evidence["report"],
        "progress": evidence["progress"],
        "completion_receipt": None,
        "source_d6_evidence": evidence,
        "architecture": architecture,
    }
    parent["parent_sha256"] = _digest(parent)
    return parent


def _selected_gather_parent(receipt_path: Path) -> dict[str, Any]:
    try:
        receipt = gather_completion.verify_completion(receipt_path)
    except gather_completion.CompletionError as error:
        raise TopologyCompositionError(
            f"gather parent completion does not replay: {error}"
        ) from error
    manifest, _ = _load_json(
        Path(receipt["manifest"]["path"]), label="gather parent manifest"
    )
    d6_parent = manifest.get("d6_parent")
    changed = receipt.get("model_delta", {}).get("changed_parameter_tensors")
    if not (
        isinstance(d6_parent, Mapping)
        and d6_parent.get("parent_profile") == "selected_short_d6"
        and changed == list(gather_completion.EXPECTED_CHANGED_PARAMETERS)
    ):
        raise TopologyCompositionError(
            "gather parent is not a completed gather-only child of short D6"
        )
    checkpoint = Path(receipt["checkpoint"]["path"])
    architecture = _require_parent_architecture(checkpoint, gather_enabled=True)
    parent = {
        "parent_profile": PARENT_SELECTED_GATHER,
        "checkpoint": receipt["checkpoint"],
        "report": receipt["report"],
        "progress": receipt["progress"],
        "completion_receipt": _file_ref(receipt_path),
        "source_d6_evidence": dict(d6_parent),
        "architecture": architecture,
        "gather_completion_receipt_sha256": receipt["receipt_sha256"],
    }
    parent["parent_sha256"] = _digest(parent)
    return parent


def _resolve_parent_from_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = value.get("parent_profile")
    if profile == PARENT_DIRECT_SHORT_D6:
        completion = value.get("completion_receipt")
        if completion is not None:
            raise TopologyCompositionError(
                "direct short-D6 selection must not carry a gather completion"
            )
        return _direct_short_d6_parent(
            Path(str(value.get("checkpoint", {}).get("path", ""))),
            Path(str(value.get("report", {}).get("path", ""))),
            Path(str(value.get("progress", {}).get("path", ""))),
        )
    if profile == PARENT_SELECTED_GATHER:
        completion = value.get("completion_receipt")
        if not isinstance(completion, Mapping):
            raise TopologyCompositionError(
                "selected gather profile requires its completion receipt"
            )
        return _selected_gather_parent(Path(str(completion.get("path", ""))))
    raise TopologyCompositionError(f"unknown topology parent profile: {profile!r}")


def issue_parent_selection(
    *,
    parent_profile: str,
    output: Path,
    selection_evidence: Sequence[Path],
    selection_basis: str,
    d6_checkpoint: Path | None = None,
    d6_report: Path | None = None,
    d6_progress: Path | None = None,
    gather_completion_receipt: Path | None = None,
) -> dict[str, Any]:
    """Write one explicit parent decision; this never launches training."""

    basis = str(selection_basis).strip()
    if not basis:
        raise TopologyCompositionError("parent selection requires a nonempty basis")
    evidence = [_file_ref(path) for path in selection_evidence]
    if not evidence:
        raise TopologyCompositionError(
            "parent selection requires at least one bound decision artifact"
        )
    if parent_profile == PARENT_DIRECT_SHORT_D6:
        if not all((d6_checkpoint, d6_report, d6_progress)):
            raise TopologyCompositionError(
                "direct short-D6 selection requires checkpoint/report/progress"
            )
        if gather_completion_receipt is not None:
            raise TopologyCompositionError(
                "direct short-D6 selection cannot also name a gather completion"
            )
        parent = _direct_short_d6_parent(
            d6_checkpoint,
            d6_report,
            d6_progress,  # type: ignore[arg-type]
        )
    elif parent_profile == PARENT_SELECTED_GATHER:
        if gather_completion_receipt is None:
            raise TopologyCompositionError(
                "selected gather profile requires a replayable completion receipt"
            )
        if any((d6_checkpoint, d6_report, d6_progress)):
            raise TopologyCompositionError(
                "selected gather profile is resolved only through its completion receipt"
            )
        parent = _selected_gather_parent(gather_completion_receipt)
    else:
        raise TopologyCompositionError(
            f"unknown topology parent profile: {parent_profile}"
        )

    payload: dict[str, Any] = {
        "schema_version": PARENT_SELECTION_SCHEMA,
        "status": "selected",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "parent_profile": parent_profile,
        "selection_basis": basis,
        "selection_evidence": evidence,
        "parent": parent,
        "issuer": _file_ref(Path(__file__)),
    }
    payload["selection_sha256"] = _digest(payload)
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        executor_base._write_exclusive(output, payload)  # noqa: SLF001
    except FileExistsError as error:
        raise TopologyCompositionError(
            f"refusing to overwrite parent selection: {output}"
        ) from error
    return payload


def verify_parent_selection(path: Path) -> dict[str, Any]:
    payload, ref = _load_json(path, label="topology parent selection")
    unhashed = dict(payload)
    stated = unhashed.pop("selection_sha256", None)
    expected_keys = {
        "schema_version",
        "status",
        "diagnostic_only",
        "promotion_eligible",
        "parent_profile",
        "selection_basis",
        "selection_evidence",
        "parent",
        "issuer",
        "selection_sha256",
    }
    if not (
        set(payload) == expected_keys
        and stated == _digest(unhashed)
        and payload.get("schema_version") == PARENT_SELECTION_SCHEMA
        and payload.get("status") == "selected"
        and payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and isinstance(payload.get("selection_basis"), str)
        and payload["selection_basis"].strip()
        and payload.get("issuer") == _file_ref(Path(__file__))
    ):
        raise TopologyCompositionError(
            "parent selection schema/status/issuer/digest drift"
        )
    evidence = payload.get("selection_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise TopologyCompositionError("parent selection has no decision evidence")
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping) or _file_ref(
            Path(str(item.get("path", "")))
        ) != dict(item):
            raise TopologyCompositionError(
                f"parent selection evidence[{index}] bytes drifted"
            )
    parent_payload = payload.get("parent")
    if not isinstance(parent_payload, Mapping):
        raise TopologyCompositionError("parent selection has no parent payload")
    resolved = _resolve_parent_from_payload(parent_payload)
    if resolved != dict(parent_payload) or payload.get(
        "parent_profile"
    ) != resolved.get("parent_profile"):
        raise TopologyCompositionError("selected parent does not replay exactly")
    return {**payload, "selection_artifact": ref}


def _dose_geometry(optimizer_steps: int) -> dict[str, Any]:
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps not in ALLOWED_OPTIMIZER_STEPS
    ):
        raise TopologyCompositionError(
            "topology optimizer steps must equal the selected short geometry: "
            f"{ALLOWED_OPTIMIZER_STEPS}, got {optimizer_steps!r}"
        )
    warmup_steps = 100
    warmup_area = min(optimizer_steps, warmup_steps)
    integrated = warmup_area * (warmup_area + 1) / (2.0 * warmup_steps) + max(
        0, optimizer_steps - warmup_steps
    )
    return {
        "optimizer_steps": optimizer_steps,
        "global_row_dose": GLOBAL_BATCH_SIZE * optimizer_steps,
        "lr_warmup_steps": warmup_steps,
        "integrated_lr_step_equivalents": integrated,
        "action_integrated_lr_step_equivalents": integrated,
    }


def _source_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES),
            cwd=repo,
            check=True,
        )
        for relative in SOURCE_FILES:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise TopologyCompositionError(
            "topology-only sources must be clean tracked canonical bytes"
        ) from error
    files = {relative: _file_ref(repo / relative) for relative in SOURCE_FILES}
    return {
        "repository_root": str(repo),
        "git_commit": commit,
        "files": files,
        "files_sha256": _digest(files),
    }


def _set_boolean(command: list[str], flag: str, enabled: bool) -> dict[str, str]:
    try:
        return gather_arm._set_boolean(command, flag, enabled)  # noqa: SLF001
    except gather_arm.CompositionArmError as error:
        raise TopologyCompositionError(str(error)) from error


def _derive_command(
    source: Sequence[str],
    *,
    trainer: Path,
    initializer: Path,
    output_root: Path,
    optimizer_steps: int,
) -> tuple[list[str], dict[str, Any]]:
    dose = _dose_geometry(optimizer_steps)
    command = list(source)
    expected = {
        "--max-steps": str(gather_arm.SELECTED_OPTIMIZER_STEPS),
        "--batch-size": str(gather_arm.SELECTED_LOCAL_BATCH_SIZE),
        "--grad-accum-steps": "1",
        "--optimizer": "adam",
        "--lr": "3e-05",
        "--lr-warmup-steps": "100",
        "--soft-target-weight": "0.9",
        "--value-loss-weight": "0.25",
        "--action-module-lr-mult": "1.0",
        "--value-lr-mult": "0.3",
    }
    try:
        observed = {
            flag: gather_arm.gather.corrected._option(command, flag)  # noqa: SLF001
            for flag in expected
        }
    except gather_arm.gather.corrected.ArmError as error:
        raise TopologyCompositionError(str(error)) from error
    if observed != expected:
        raise TopologyCompositionError(
            f"source is not exact selected-dose TEMP geometry: {observed}"
        )
    if command.count("--no-resume-optimizer") != 1 or "--resume-optimizer" in command:
        raise TopologyCompositionError("topology arm requires fresh Adam")
    if "--ddp-shard-data" in command:
        raise TopologyCompositionError("selected TEMP order must remain non-sharded")

    trainer_positions = [
        index
        for index, value in enumerate(command)
        if Path(value).name == "train_bc.py"
    ]
    if len(trainer_positions) != 1:
        raise TopologyCompositionError("source must name exactly one trainer")
    current_trainer = trainer.expanduser().resolve(strict=True)
    changes: dict[str, Any] = {
        "trainer": {
            "source": command[trainer_positions[0]],
            "treatment": str(current_trainer),
        }
    }
    command[trainer_positions[0]] = str(current_trainer)
    updates = {
        "--init-checkpoint": str(initializer.resolve(strict=True)),
        "--checkpoint": str(output_root / "candidate.pt"),
        "--report": str(output_root / "train.report.json"),
        "--batch-size": str(LOCAL_BATCH_SIZE),
        "--max-steps": str(dose["optimizer_steps"]),
        "--action-module-lr-mult": str(ACTION_MODULE_LR_MULT),
        "--value-lr-mult": str(VALUE_LR_MULT),
        "--trunk-lr-mult": str(TRUNK_LR_MULT),
        "--amp": "none",
        "--float32-matmul-precision": "highest",
        "--freeze-modules": FREEZE_MODULES,
        "--require-only-trainable-prefixes": TRAINABLE_PREFIX,
    }
    appendable = {
        "--freeze-modules",
        "--require-only-trainable-prefixes",
        "--trunk-lr-mult",
        "--amp",
        "--float32-matmul-precision",
    }
    for flag, value in updates.items():
        try:
            old = (
                "absent"
                if flag in appendable and flag not in command
                else gather_arm.gather.corrected._option(command, flag)  # noqa: SLF001
            )
            gather_arm.gather.corrected._set_option(command, flag, value)  # noqa: SLF001
        except gather_arm.gather.corrected.ArmError as error:
            raise TopologyCompositionError(str(error)) from error
        changes[flag] = {"source": old, "treatment": value}
    changes["--symmetry-augment"] = _set_boolean(command, "--symmetry-augment", True)
    changes["--symmetry-augment-events"] = _set_boolean(
        command, "--symmetry-augment-events", True
    )
    changes["--fused-optimizer"] = _set_boolean(command, "--fused-optimizer", False)
    return command, changes


def _validate_topology_coverage(path: Path, descriptor_path: Path) -> dict[str, Any]:
    try:
        action_coverage = gather_arm.gather._validate_coverage(  # noqa: SLF001
            path, descriptor_path
        )
        audit, _ = gather_arm.gather.corrected._load_json(path)  # noqa: SLF001
    except (gather_arm.gather.ArmError, OSError) as error:
        raise TopologyCompositionError(str(error)) from error
    rows = audit.get("audits")
    if not isinstance(rows, list) or not rows:
        raise TopologyCompositionError("topology audit has no corpus rows")
    graph_rows = []
    for row in rows:
        graph = row.get("graph_incidence", {}) if isinstance(row, Mapping) else {}
        viability = row.get("viability", {}) if isinstance(row, Mapping) else {}
        if not (
            viability.get("graph_relational_trunk") is True
            and graph.get("missing_columns") == []
            and graph.get("out_of_range_ids") == 0
        ):
            raise TopologyCompositionError(
                "TEMP corpus lacks exact usable graph-incidence columns"
            )
        graph_rows.append(
            {
                "corpus_dir": row.get("corpus_dir"),
                "columns": graph.get("columns"),
                "out_of_range_ids": 0,
            }
        )
    return {
        **action_coverage,
        "graph_incidence": graph_rows,
        "graph_incidence_sha256": _digest(graph_rows),
    }


def _validate_upgrade_receipt(
    path: Path, *, parent_checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        receipt = architecture_upgrade.verify_receipt(path)
    except architecture_upgrade.UpgradeError as error:
        raise TopologyCompositionError(
            f"topology upgrade receipt does not replay: {error}"
        ) from error
    if not (
        receipt.get("module") == architecture_upgrade.MODULE_TOPOLOGY_RESIDUAL
        and receipt.get("source") == dict(parent_checkpoint)
        and receipt.get("new_parameters") == list(EXPECTED_TOPOLOGY_PARAMETERS)
    ):
        raise TopologyCompositionError(
            "upgrade is not MODULE_TOPOLOGY_RESIDUAL on the selected parent bytes"
        )
    return receipt


def _forbidden_outputs(root: Path) -> tuple[Path, ...]:
    checkpoint = root / "candidate.pt"
    return (
        checkpoint,
        Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"),
        root / "train.report.json",
        root / "diagnostic-execution.claim.json",
        root / "diagnostic-execution.receipt.json",
        root / "diagnostic-execution.status.jsonl",
        root / "diagnostic-completion.receipt.json",
        root / "stdout.log",
        root / "stderr.log",
    )


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    try:
        source, source_ref = gather_arm.gather._load_source(  # noqa: SLF001
            args.source_manifest,
            args.selected_dose_plan,
            args.selected_dose_report,
        )
    except gather_arm.gather.ArmError as error:
        raise TopologyCompositionError(str(error)) from error
    selection = verify_parent_selection(args.parent_selection)
    parent = selection["parent"]
    upgrade = _validate_upgrade_receipt(
        args.topology_upgrade_receipt,
        parent_checkpoint=parent["checkpoint"],
    )
    initializer = Path(upgrade["upgraded_initializer"]["path"])
    output_root = args.output_root.expanduser().resolve()
    existing = [str(path) for path in _forbidden_outputs(output_root) if path.exists()]
    if existing:
        raise TopologyCompositionError(
            f"topology-only output already exists: {existing}"
        )

    dose = _dose_geometry(args.optimizer_steps)
    coverage = _validate_topology_coverage(
        args.architecture_audit, Path(source["descriptor"]["path"])
    )
    binding = _source_binding(args.repo)
    files = binding["files"]
    executor_ref = files.get(EXECUTOR_RELATIVE_PATH)
    completion_ref = files.get(COMPLETION_RELATIVE_PATH)
    trainer_ref = files.get("tools/train_bc.py")
    if not all(
        isinstance(ref, Mapping) for ref in (executor_ref, completion_ref, trainer_ref)
    ):
        raise TopologyCompositionError(
            "source binding lacks executor/finalizer/current trainer"
        )
    command, changes = _derive_command(
        source["command"],
        trainer=Path(trainer_ref["path"]),  # type: ignore[index]
        initializer=initializer,
        output_root=output_root,
        optimizer_steps=dose["optimizer_steps"],
    )
    descriptor_meta, _ = gather_arm.gather.corrected._preflight_descriptor(  # noqa: SLF001
        Path(source["descriptor"]["path"])
    )
    event_contract, event_changes = (
        gather_arm.gather.corrected._bind_event_history_training_command(  # noqa: SLF001
            command, descriptor_meta
        )
    )
    changes.update(event_changes)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": f"{EXECUTOR_RELATIVE_PATH} execute --go",
        "diagnostic_executor": dict(executor_ref),  # type: ignore[arg-type]
        "completion_finalizer": dict(completion_ref),  # type: ignore[arg-type]
        "completion_interface_present": (
            f"{COMPLETION_RELATIVE_PATH} finalize --expected-checkpoint-sha256 SHA256"
        ),
        "source_temperature_manifest": source_ref,
        "source_temperature_manifest_sha256": source["manifest_sha256"],
        "selected_geometry_evidence": source["selected_geometry_evidence"],
        "source_recipe": source["recipe"],
        "source_recipe_sha256": source["recipe_sha256"],
        "descriptor": source["descriptor"],
        "validation_sentinel": source["validation_sentinel"],
        "parent_selection": selection["selection_artifact"],
        "selected_parent": parent,
        "selected_parent_profile": selection["parent_profile"],
        "parent_selection_sha256": selection["selection_sha256"],
        "initialization_treatment": upgrade["upgraded_initializer"],
        "function_preserving_upgrade_receipt": upgrade["receipt"],
        "function_preserving_upgrade": {
            key: value for key, value in upgrade.items() if key != "receipt"
        },
        "corpus_topology_target_coverage": coverage,
        "event_history_training_contract": event_contract,
        "source_binding": binding,
        "only_declared_model_delta": (
            "train function-preserving topology_residual_adapter on frozen exact "
            "selected parent"
        ),
        "matched_contract": {
            "reference_checkpoint": parent["checkpoint"],
            "reference_parent_receipt": parent["completion_receipt"],
            "evaluation_reference": "exact_selected_parent",
            "candidate_chaining": False,
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "optimizer_steps": dose["optimizer_steps"],
            "global_row_dose": dose["global_row_dose"],
            "fresh_adam": True,
            "base_lr": 3e-5,
            "trunk_lr_mult": TRUNK_LR_MULT,
            "topology_lr": 3e-5 * TRUNK_LR_MULT,
            "action_module_lr_mult": ACTION_MODULE_LR_MULT,
            "value_lr_mult": VALUE_LR_MULT,
            "freeze_modules": FREEZE_MODULES.split(","),
            "required_trainable_prefixes": [TRAINABLE_PREFIX],
            "new_trainable_parameter_names": list(EXPECTED_TOPOLOGY_PARAMETERS),
            "new_trainable_parameter_tensors": len(EXPECTED_TOPOLOGY_PARAMETERS),
            "new_trainable_parameters": EXPECTED_TOPOLOGY_PARAMETER_COUNT,
            "mature_parameters_trainable": False,
            "trained_gather_frozen": parent["architecture"]["action_target_gather"],
            "amp": "none",
            "float32_matmul_precision": "highest",
            "symmetry_augment": True,
            "symmetry_augment_events": True,
            "selected_TEMP_data_descriptor_and_seed_unchanged": True,
            "sampler_batch_partition_unchanged": True,
            "selected_TEMP_policy_value_losses_unchanged": True,
            "distributed_symmetry_contract": (
                "per_rank_seedsequence_checkpoint_resume_v1"
            ),
        },
        "effective_trainable_objective": {
            "policy_loss_reaches_topology_adapter": True,
            "value_loss_reaches_topology_adapter": True,
            "all_mature_policy_value_tensors_frozen": True,
        },
        "optimizer_geometry_contract": {
            "source_selected_TEMP": {
                "world_size": WORLD_SIZE,
                "local_batch_size": gather_arm.SELECTED_LOCAL_BATCH_SIZE,
                "global_batch_size": gather_arm.SELECTED_GLOBAL_BATCH_SIZE,
                "optimizer_steps": gather_arm.SELECTED_OPTIMIZER_STEPS,
                "global_row_dose": (
                    gather_arm.SELECTED_GLOBAL_BATCH_SIZE
                    * gather_arm.SELECTED_OPTIMIZER_STEPS
                ),
            },
            "treatment_topology_commissioning": {
                "world_size": WORLD_SIZE,
                "local_batch_size": LOCAL_BATCH_SIZE,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                **dose,
                "trunk_integrated_lr_step_equivalents": (
                    dose["integrated_lr_step_equivalents"] * TRUNK_LR_MULT
                ),
            },
            "allowlisted_optimizer_steps": list(ALLOWED_OPTIMIZER_STEPS),
        },
        "allowlisted_command_changes": changes,
        "command": command,
        "command_sha256": _digest(command),
        "output_root": str(output_root),
        "evaluation_contract": {
            "primary_opponent": parent["checkpoint"],
            "comparison": "same-key exact-parent behavior screen after completion",
            "promotion_from_this_diagnostic": False,
        },
        "executor_compatibility": {
            "receipt_schema": RECEIPT_SCHEMA,
            "idle_topology": "exactly_8_visible_B200s",
            "one_shot": True,
        },
    }
    manifest["manifest_sha256"] = _digest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "topology-only-composition.manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise TopologyCompositionError(f"prepared manifest drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    return manifest, path


def _verify_ref(value: Any, *, label: str) -> Path:
    try:
        return executor_base._verify_ref(value, label=label)  # noqa: SLF001
    except executor_base.ExecutionError as error:
        raise TopologyCompositionError(str(error)) from error


def verify(
    manifest_path: Path,
    *,
    expected_executor: Path | None = None,
    require_fresh_outputs: bool = True,
) -> dict[str, Any]:
    payload, manifest_ref = _load_json(manifest_path, label="topology manifest")
    unhashed = dict(payload)
    stated = unhashed.pop("manifest_sha256", None)
    if not (
        stated == _digest(unhashed)
        and payload.get("schema_version") == SCHEMA
        and payload.get("diagnostic_only") is True
        and payload.get("promotion_eligible") is False
        and payload.get("launch_authorized") is False
        and payload.get("diagnostic_execution_authorized") is True
        and payload.get("launch_interface_present")
        == f"{EXECUTOR_RELATIVE_PATH} execute --go"
    ):
        raise TopologyCompositionError(
            "topology manifest schema/authorization/digest drift"
        )

    executor = _verify_ref(payload.get("diagnostic_executor"), label="executor")
    bound_executor = (
        Path(__file__).resolve()
        if expected_executor is None
        else expected_executor.expanduser().resolve(strict=True)
    )
    if executor != bound_executor:
        raise TopologyCompositionError("manifest authorizes a different executor")
    binding = payload.get("source_binding")
    if not isinstance(binding, Mapping):
        raise TopologyCompositionError("manifest lacks source binding")
    repo = Path(str(binding.get("repository_root", ""))).resolve(strict=True)
    if executor_base._git_head(repo) != binding.get("git_commit"):  # noqa: SLF001
        raise TopologyCompositionError("topology checkout commit drift")
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        raise TopologyCompositionError("topology source binding is incomplete")
    if binding.get("files_sha256") != _digest(files):
        raise TopologyCompositionError("topology source-set digest drift")
    for relative, ref in files.items():
        if _verify_ref(ref, label=f"source.{relative}") != (repo / relative).resolve(
            strict=True
        ):
            raise TopologyCompositionError(f"bound source escaped checkout: {relative}")
    if files[EXECUTOR_RELATIVE_PATH] != payload["diagnostic_executor"]:
        raise TopologyCompositionError("executor differs from bound source bytes")
    completion = _verify_ref(
        payload.get("completion_finalizer"), label="completion finalizer"
    )
    if not (
        completion == (repo / COMPLETION_RELATIVE_PATH).resolve(strict=True)
        and files[COMPLETION_RELATIVE_PATH] == payload["completion_finalizer"]
        and payload.get("completion_interface_present")
        == f"{COMPLETION_RELATIVE_PATH} finalize --expected-checkpoint-sha256 SHA256"
    ):
        raise TopologyCompositionError("completion finalizer binding drift")

    source_manifest = _verify_ref(
        payload.get("source_temperature_manifest"), label="source TEMP manifest"
    )
    selected = payload.get("selected_geometry_evidence")
    if not isinstance(selected, Mapping):
        raise TopologyCompositionError("manifest lacks selected TEMP geometry")
    try:
        source, _ = gather_arm.gather._load_source(  # noqa: SLF001
            source_manifest,
            _verify_ref(selected.get("plan"), label="selected geometry plan"),
            _verify_ref(selected.get("report"), label="selected geometry report"),
        )
    except gather_arm.gather.ArmError as error:
        raise TopologyCompositionError(str(error)) from error
    for manifest_key, source_key in (
        ("selected_geometry_evidence", "selected_geometry_evidence"),
        ("source_recipe", "recipe"),
        ("descriptor", "descriptor"),
        ("validation_sentinel", "validation_sentinel"),
    ):
        if payload.get(manifest_key) != source[source_key]:
            raise TopologyCompositionError(f"selected TEMP drift: {manifest_key}")
    if not (
        payload.get("source_temperature_manifest_sha256") == source["manifest_sha256"]
        and payload.get("source_recipe_sha256") == source["recipe_sha256"]
    ):
        raise TopologyCompositionError("selected TEMP digest drift")

    selection = verify_parent_selection(
        _verify_ref(payload.get("parent_selection"), label="parent selection")
    )
    if not (
        payload.get("selected_parent") == selection["parent"]
        and payload.get("selected_parent_profile") == selection["parent_profile"]
        and payload.get("parent_selection_sha256") == selection["selection_sha256"]
    ):
        raise TopologyCompositionError("selected parent binding drift")
    upgrade = _validate_upgrade_receipt(
        _verify_ref(
            payload.get("function_preserving_upgrade_receipt"),
            label="topology upgrade receipt",
        ),
        parent_checkpoint=selection["parent"]["checkpoint"],
    )
    if not (
        payload.get("initialization_treatment") == upgrade["upgraded_initializer"]
        and payload.get("function_preserving_upgrade")
        == {key: value for key, value in upgrade.items() if key != "receipt"}
    ):
        raise TopologyCompositionError("topology upgrade replay drift")

    coverage = payload.get("corpus_topology_target_coverage")
    if not isinstance(coverage, Mapping):
        raise TopologyCompositionError("manifest lacks topology coverage")
    expected_coverage = _validate_topology_coverage(
        _verify_ref(coverage.get("artifact"), label="architecture audit"),
        Path(source["descriptor"]["path"]),
    )
    if expected_coverage != coverage:
        raise TopologyCompositionError("topology coverage replay drift")

    root = Path(str(payload.get("output_root", ""))).resolve()
    matched = payload.get("matched_contract")
    if not isinstance(matched, Mapping):
        raise TopologyCompositionError("manifest lacks matched contract")
    dose = _dose_geometry(matched.get("optimizer_steps"))
    trainer = _verify_ref(files["tools/train_bc.py"], label="trainer")
    expected_command, changes = _derive_command(
        source["command"],
        trainer=trainer,
        initializer=Path(upgrade["upgraded_initializer"]["path"]),
        output_root=root,
        optimizer_steps=dose["optimizer_steps"],
    )
    descriptor_meta, _ = gather_arm.gather.corrected._preflight_descriptor(  # noqa: SLF001
        Path(source["descriptor"]["path"])
    )
    expected_event_contract, event_changes = (
        gather_arm.gather.corrected._bind_event_history_training_command(  # noqa: SLF001
            expected_command, descriptor_meta
        )
    )
    changes.update(event_changes)
    if not (
        payload.get("command") == expected_command
        and payload.get("command_sha256") == _digest(expected_command)
        and payload.get("allowlisted_command_changes") == changes
        and payload.get("event_history_training_contract") == expected_event_contract
    ):
        raise TopologyCompositionError("topology command derivation drift")

    parent = selection["parent"]
    expected_matched = {
        "reference_checkpoint": parent["checkpoint"],
        "reference_parent_receipt": parent["completion_receipt"],
        "evaluation_reference": "exact_selected_parent",
        "candidate_chaining": False,
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "optimizer_steps": dose["optimizer_steps"],
        "global_row_dose": dose["global_row_dose"],
        "fresh_adam": True,
        "base_lr": 3e-5,
        "trunk_lr_mult": TRUNK_LR_MULT,
        "topology_lr": 3e-5 * TRUNK_LR_MULT,
        "action_module_lr_mult": ACTION_MODULE_LR_MULT,
        "value_lr_mult": VALUE_LR_MULT,
        "freeze_modules": FREEZE_MODULES.split(","),
        "required_trainable_prefixes": [TRAINABLE_PREFIX],
        "new_trainable_parameter_names": list(EXPECTED_TOPOLOGY_PARAMETERS),
        "new_trainable_parameter_tensors": len(EXPECTED_TOPOLOGY_PARAMETERS),
        "new_trainable_parameters": EXPECTED_TOPOLOGY_PARAMETER_COUNT,
        "mature_parameters_trainable": False,
        "trained_gather_frozen": parent["architecture"]["action_target_gather"],
        "amp": "none",
        "float32_matmul_precision": "highest",
        "symmetry_augment": True,
        "symmetry_augment_events": True,
        "selected_TEMP_data_descriptor_and_seed_unchanged": True,
        "sampler_batch_partition_unchanged": True,
        "selected_TEMP_policy_value_losses_unchanged": True,
        "distributed_symmetry_contract": "per_rank_seedsequence_checkpoint_resume_v1",
    }
    expected_objective = {
        "policy_loss_reaches_topology_adapter": True,
        "value_loss_reaches_topology_adapter": True,
        "all_mature_policy_value_tensors_frozen": True,
    }
    expected_geometry = {
        "source_selected_TEMP": {
            "world_size": WORLD_SIZE,
            "local_batch_size": gather_arm.SELECTED_LOCAL_BATCH_SIZE,
            "global_batch_size": gather_arm.SELECTED_GLOBAL_BATCH_SIZE,
            "optimizer_steps": gather_arm.SELECTED_OPTIMIZER_STEPS,
            "global_row_dose": (
                gather_arm.SELECTED_GLOBAL_BATCH_SIZE
                * gather_arm.SELECTED_OPTIMIZER_STEPS
            ),
        },
        "treatment_topology_commissioning": {
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            **dose,
            "trunk_integrated_lr_step_equivalents": (
                dose["integrated_lr_step_equivalents"] * TRUNK_LR_MULT
            ),
        },
        "allowlisted_optimizer_steps": list(ALLOWED_OPTIMIZER_STEPS),
    }
    if not (
        payload.get("matched_contract") == expected_matched
        and payload.get("effective_trainable_objective") == expected_objective
        and payload.get("optimizer_geometry_contract") == expected_geometry
        and payload.get("only_declared_model_delta")
        == (
            "train function-preserving topology_residual_adapter on frozen exact "
            "selected parent"
        )
        and payload.get("evaluation_contract")
        == {
            "primary_opponent": parent["checkpoint"],
            "comparison": "same-key exact-parent behavior screen after completion",
            "promotion_from_this_diagnostic": False,
        }
    ):
        raise TopologyCompositionError("topology causal/evaluation contract drift")
    if require_fresh_outputs:
        existing = [str(path) for path in _forbidden_outputs(root) if path.exists()]
        if existing:
            raise TopologyCompositionError(
                f"topology output already exists: {existing}"
            )
    return {
        "manifest": payload,
        "manifest_ref": manifest_ref,
        "repo": repo,
        "preparer_repo": repo,
        "command": expected_command,
        "output_root": root,
    }


def execute(
    manifest_path: Path,
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    conflict_probe: Callable[[], list[str]] = executor_base._probe_conflicting_compute,  # noqa: SLF001
) -> dict[str, Any]:
    verified = verify(manifest_path)
    try:
        return executor_base._submit_verified(  # noqa: SLF001
            verified,
            unit=unit,
            runner=runner,
            conflict_probe=conflict_probe,
            claim_schema=CLAIM_SCHEMA,
            receipt_schema=RECEIPT_SCHEMA,
            status_schema=STATUS_SCHEMA,
        )
    except executor_base.ExecutionError as error:
        raise TopologyCompositionError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    select = actions.add_parser("select-parent")
    select.add_argument("--parent-profile", choices=PARENT_PROFILES, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--selection-evidence", type=Path, nargs="+", required=True)
    select.add_argument("--selection-basis", required=True)
    select.add_argument("--d6-checkpoint", type=Path)
    select.add_argument("--d6-report", type=Path)
    select.add_argument("--d6-progress", type=Path)
    select.add_argument("--gather-completion-receipt", type=Path)

    prep = actions.add_parser("prepare")
    prep.add_argument("--source-manifest", required=True, type=Path)
    prep.add_argument("--selected-dose-plan", required=True, type=Path)
    prep.add_argument("--selected-dose-report", required=True, type=Path)
    prep.add_argument("--parent-selection", required=True, type=Path)
    prep.add_argument("--topology-upgrade-receipt", required=True, type=Path)
    prep.add_argument("--architecture-audit", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument(
        "--optimizer-steps",
        type=int,
        choices=ALLOWED_OPTIMIZER_STEPS,
        default=OPTIMIZER_STEPS,
    )
    prep.add_argument("--repo", default=REPO_ROOT, type=Path)

    check_parent = actions.add_parser("verify-parent")
    check_parent.add_argument("--selection", required=True, type=Path)
    check = actions.add_parser("verify")
    check.add_argument("--manifest", required=True, type=Path)
    run = actions.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-topology-only-composition")
    run.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "select-parent":
            value = issue_parent_selection(
                parent_profile=args.parent_profile,
                output=args.output,
                selection_evidence=args.selection_evidence,
                selection_basis=args.selection_basis,
                d6_checkpoint=args.d6_checkpoint,
                d6_report=args.d6_report,
                d6_progress=args.d6_progress,
                gather_completion_receipt=args.gather_completion_receipt,
            )
            print(json.dumps(value, indent=2, sort_keys=True))
            return
        if args.action == "verify-parent":
            print(
                json.dumps(
                    verify_parent_selection(args.selection), indent=2, sort_keys=True
                )
            )
            return
        if args.action == "prepare":
            manifest, path = prepare(args)
            print(
                json.dumps(
                    {
                        "prepared": str(path),
                        "launched": False,
                        "manifest_sha256": manifest["manifest_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return
        if args.action == "verify" or (args.action == "execute" and not args.go):
            checked = verify(args.manifest)
            print(
                json.dumps(
                    {
                        "verified": True,
                        "launched": False,
                        "manifest": checked["manifest_ref"],
                    },
                    sort_keys=True,
                )
            )
            return
        receipt = execute(args.manifest, unit=args.unit)
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
    except (TopologyCompositionError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
