#!/usr/bin/env python3
"""Prepare (never launch) the function-preserving gather commissioning arm.

This arm bridges the sealed full-dose TEMP recipe/data identity to an executed
selected-dose geometry plan/report and consumes its 524,288-row measure exactly
once.  Unlike the mature TEMP
network, however, the target-gather projection starts with a zero output matrix:
the selected 128-step/100-warmup schedule gives it only 78.5 full-LR-equivalent
updates and cannot fairly reject the representation mechanism.  Commissioning
therefore uses a smaller global batch (more optimizer steps at the same row
dose), the already-reviewed 4x action-module LR, and freezes every mature model
surface.  The initialization must be an exact gather-only upgrade of the same f7
bytes and the bound corpora must prove that topology targets are present and
valid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_corrected_policy_arm as corrected  # noqa: E402
from tools import a1_production_temperature_replication as production_temp  # noqa: E402


SCHEMA = "a1-topology-gather-arm-manifest-v4"
LEGACY_TEMP_SCHEMA = getattr(
    production_temp,
    "LEGACY_MANIFEST_SCHEMA",
    "a1-production-temperature-replication-v2",
)
SOURCE_SCHEMAS = frozenset(
    {
        "a1-production-temperature-replication-v1",
        LEGACY_TEMP_SCHEMA,
    }
)
GEOMETRY_SCHEMA = "a1-b200-microbatch-quality-plan-v1"
GEOMETRY_RUN_ID = "ddp8-b512"
EXECUTOR_RELATIVE_PATH = "tools/a1_topology_gather_arm_execute.py"
WORLD_SIZE = 8
SELECTED_OPTIMIZER_STEPS = 128
SELECTED_GLOBAL_ROW_DOSE = 524_288
LOCAL_BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
OPTIMIZER_STEPS = SELECTED_GLOBAL_ROW_DOSE // GLOBAL_BATCH_SIZE
ACTION_MODULE_LR_MULT = 4.0
FREEZE_MODULES = "trunk,action_encoder,policy_head,value_heads"
TRAINABLE_PREFIX = "target_gather_proj"
SOURCE_FILES = (
    "tools/a1_topology_gather_arm.py",
    EXECUTOR_RELATIVE_PATH,
    "tools/a1_corrected_policy_arm_execute.py",
    "tools/a1_corrected_policy_arm.py",
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/f69_upgrade_checkpoint_config.py",
    "tools/audit_memmap_architecture_targets.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
) + (
    # v3 production TEMP delegates typed dose semantics to this module.  Older
    # checkouts do not contain it, so bind it exactly when that API is present.
    ("tools/a1_learner_dose_contract.py",)
    if hasattr(production_temp, "LEGACY_MANIFEST_SCHEMA")
    else ()
)
EXPECTED_NEW_PARAMETERS = (
    "target_gather_proj.0.bias",
    "target_gather_proj.0.weight",
    "target_gather_proj.1.bias",
    "target_gather_proj.1.weight",
)


class ArmError(RuntimeError):
    """The requested experiment is not the one-axis topology arm."""


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmError(f"cannot authenticate selected-geometry checkout: {error}") from error


def _verify_json_digest(
    payload: Mapping[str, Any], *, digest_field: str, label: str
) -> None:
    stated = payload.get(digest_field)
    unhashed = {key: value for key, value in payload.items() if key != digest_field}
    if stated != corrected._digest(unhashed):  # noqa: SLF001
        raise ArmError(f"{label} semantic digest is invalid")


def _verify_ref(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ArmError(f"{label} reference is malformed")
    try:
        observed = corrected._file_ref(Path(str(value.get("path", ""))))  # noqa: SLF001
    except (OSError, corrected.ArmError) as error:
        raise ArmError(f"{label} reference cannot be verified: {error}") from error
    if observed != dict(value):
        raise ArmError(f"{label} bytes drifted")
    return observed


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    ref = corrected._file_ref(path)  # noqa: SLF001
    try:
        value = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArmError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ArmError(f"{label} is not a JSON object")
    return value, ref


def _selected_recipe() -> dict[str, Any]:
    return {
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "steps": SELECTED_OPTIMIZER_STEPS,
        "base_value_row_dose": SELECTED_GLOBAL_ROW_DOSE,
        "policy_aux_active_batch_size_per_rank": 0,
        "policy_aux_active_row_dose": 0,
        "replay_supervised_policy": True,
        "replay_supervised_value": True,
        "replay_forward_kl_weight": 0.0,
        "soft_target_weight": 0.9,
        "fresh_optimizer": True,
        "independent_f7_initialization": True,
    }


def _load_temperature_source(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Authenticate the historical full-dose TEMP manifest as recipe evidence.

    The old selection receipts and its execution checkout were deliberately
    cleaned.  They are not needed to reuse immutable data/objective identities:
    this verifier binds the manifest digest, live f7/data/sentinel bytes, exact
    TEMP argv, and descriptor semantics without claiming the rejected 4.19M-row
    dose is still launch authority.
    """

    payload, ref = corrected._load_json(path)  # noqa: SLF001
    _verify_json_digest(payload, digest_field="manifest_sha256", label="source TEMP")
    if payload.get("schema_version") not in SOURCE_SCHEMAS or not (
        payload.get("diagnostic_only") is False
        and payload.get("production_eligible") is True
        and payload.get("launch_authorized") is True
    ):
        raise ArmError("source is not a sealed production TEMP manifest")
    if payload.get("selected_dose") != {
        "optimizer_steps": 1024,
        "world_size": 8,
        "per_rank_batch_size": 512,
        "global_samples": 4_194_304,
        "optimizer": "fresh_adam",
        "lr": 3e-5,
        "training_rng_rank_offset": True,
    }:
        raise ArmError("historical TEMP source dose identity drift")
    initialization = _verify_ref(payload.get("f7_parent"), label="source f7")
    descriptor = _verify_ref(payload.get("source_descriptor"), label="source descriptor")
    sentinel = _verify_ref(payload.get("validation_sentinel"), label="source sentinel")
    if initialization["sha256"] != production_temp.F7_SHA256:
        raise ArmError("source TEMP initializer is not exact f7")
    command = payload.get("command")
    if (
        not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or payload.get("command_sha256") != corrected._digest(command)  # noqa: SLF001
    ):
        raise ArmError("source TEMP command binding is invalid")
    try:
        production_temp._validate_recipe(  # noqa: SLF001
            command,
            descriptor=descriptor["path"],
            sentinel=sentinel["path"],
            f7=initialization["path"],
        )
    except production_temp.TemperatureReplicationError as error:
        raise ArmError(f"source TEMP objective drift: {error}") from error
    if "--validation-game-seed-manifest" in command:
        raise ArmError("source TEMP command mixes validation controls")

    descriptor_meta, _ = corrected._preflight_descriptor(  # noqa: SLF001
        Path(descriptor["path"])
    )
    components = descriptor_meta.get("components")
    component_ids = [
        row.get("component_id") for row in components if isinstance(row, Mapping)
    ] if isinstance(components, list) else []
    expected_ids = list(production_temp.COMPONENT_IDS)
    if not (
        component_ids == expected_ids
        and descriptor_meta.get("policy_distillation_component_ids") == expected_ids
        and descriptor_meta.get("value_training_component_ids") == expected_ids
        and descriptor_meta.get("policy_kl_anchor_component_ids") == ["gen3_replay"]
        and descriptor_meta.get("stored_policy_component_temperatures")
        == production_temp.COMPONENT_TEMPERATURES
    ):
        raise ArmError("source TEMP descriptor objective/supervision scope drift")
    bindings = payload.get("component_bindings")
    if not isinstance(bindings, list) or [
        row.get("component_id") for row in bindings if isinstance(row, Mapping)
    ] != expected_ids:
        raise ArmError("source TEMP component binding identity drift")
    inventories = []
    for binding, component in zip(bindings, components, strict=True):
        inventory = binding.get("payload_inventory_sha256")
        if inventory != component.get("payload_inventory_sha256"):
            raise ArmError("source TEMP payload inventory binding drift")
        inventories.append(inventory)
        _verify_ref(binding.get("corpus_meta"), label=f"{binding['component_id']}.corpus_meta")
        _verify_ref(
            binding.get("validation_manifest"),
            label=f"{binding['component_id']}.validation_manifest",
        )
    event_contract = payload.get("event_history_training_contract")
    if not isinstance(event_contract, Mapping) or not (
        event_contract.get("public_observation_masked") is True
        and event_contract.get("graph_history_features") is True
        and event_contract.get("payload_inventory_acknowledgements") == inventories
    ):
        raise ArmError("source TEMP event-history binding drift")
    ack_positions = [
        index for index, item in enumerate(command)
        if item == corrected.EVENT_HISTORY_ACK_FLAG
    ]
    observed_acks = [command[index + 1] for index in ack_positions]
    crop_count = command.count(corrected.EVENT_HISTORY_CROP_FLAG)
    if (
        observed_acks != inventories
        or crop_count not in {0, 1}
        or (
            payload.get("schema_version") == LEGACY_TEMP_SCHEMA
            and crop_count != 1
        )
    ):
        raise ArmError("source TEMP command event-history authority drift")
    return {
        "manifest": payload,
        "manifest_ref": ref,
        "initialization": initialization,
        "descriptor": descriptor,
        "validation_sentinel": sentinel,
        "command": list(command),
        "descriptor_meta": descriptor_meta,
    }, ref


def _geometry_expected_command(
    source: Sequence[str], *, trainer: Path, checkpoint: str, report: str
) -> list[str]:
    command = list(source)
    trainers = [index for index, value in enumerate(command) if Path(value).name == "train_bc.py"]
    if len(trainers) != 1:
        raise ArmError("source TEMP command does not name exactly one trainer")
    command[trainers[0]] = str(trainer)
    for flag, value in (
        ("--max-steps", str(SELECTED_OPTIMIZER_STEPS)),
        ("--checkpoint", checkpoint),
        ("--report", report),
        ("--train-diagnostics-every-batches", "0"),
        ("--objective-gradient-interference-every-batches", "0"),
    ):
        corrected._set_option(command, flag, value)  # noqa: SLF001
    return command


def _load_selected_geometry(
    plan_path: Path,
    report_path: Path,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    plan, plan_ref = corrected._load_json(plan_path)  # noqa: SLF001
    _verify_json_digest(plan, digest_field="plan_sha256", label="selected geometry plan")
    if not (
        plan.get("schema_version") == GEOMETRY_SCHEMA
        and plan.get("diagnostic_only") is True
        and plan.get("promotion_eligible") is False
        and plan.get("launch_authorized") is True
        and plan.get("only_intended_drift") == ["world_size", "batch_size", "gpu_ids"]
    ):
        raise ArmError("selected geometry plan authorization/schema drift")
    expected_invariants = {
        "global_batch_size": 4096,
        "lr": 3e-5,
        "lr_schedule": "flat",
        "lr_warmup_steps": 100,
        "warmup_samples": 409_600,
        "optimizer_steps": SELECTED_OPTIMIZER_STEPS,
        "planned_samples": SELECTED_GLOBAL_ROW_DOSE,
        "seed": 1,
    }
    if plan.get("matched_invariants") != expected_invariants:
        raise ArmError("selected 128-step geometry contract drift")
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping) or inputs != {
        "data": source["descriptor"]["path"],
        "data_sha256": source["descriptor"]["sha256"],
        "init_checkpoint": source["initialization"]["path"],
        "init_checkpoint_sha256": source["initialization"]["sha256"],
    }:
        raise ArmError("selected geometry does not bind the sealed TEMP data/f7")
    runs = plan.get("runs")
    matches = [
        run for run in runs if isinstance(run, Mapping) and run.get("run_id") == GEOMETRY_RUN_ID
    ] if isinstance(runs, list) else []
    if len(matches) != 1:
        raise ArmError("selected geometry plan lacks exactly one ddp8-b512 run")
    run = dict(matches[0])
    expected_run_geometry = {
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "grad_accum_steps": 1,
        "max_steps": SELECTED_OPTIMIZER_STEPS,
        "planned_samples": SELECTED_GLOBAL_ROW_DOSE,
        "warmup_samples": 409_600,
        "lr_warmup_steps": 100,
        "gpu_ids": list(range(8)),
    }
    if any(run.get(key) != value for key, value in expected_run_geometry.items()):
        raise ArmError("selected ddp8-b512 run geometry drift")
    command = run.get("command")
    if (
        not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or run.get("command_sha256") != corrected._digest(command)  # noqa: SLF001
    ):
        raise ArmError("selected geometry command binding drift")
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ArmError("selected geometry runtime binding is malformed")
    try:
        runtime_repo = Path(str(runtime.get("repository_root", ""))).resolve(strict=True)
        trainer = Path(str(runtime.get("trainer", ""))).resolve(strict=True)
    except OSError as error:
        raise ArmError(f"selected geometry runtime is unavailable: {error}") from error
    if not (
        trainer == (runtime_repo / "tools/train_bc.py").resolve(strict=True)
        and runtime.get("trainer_sha256") == corrected._file_sha(trainer)  # noqa: SLF001
        and runtime.get("repository_commit") == _git_head(runtime_repo)
    ):
        raise ArmError("selected geometry trainer checkout/bytes drifted")
    expected_command = _geometry_expected_command(
        source["command"],
        trainer=trainer,
        checkpoint=str(corrected._option(command, "--checkpoint")),  # noqa: SLF001
        report=str(corrected._option(command, "--report")),  # noqa: SLF001
    )
    if command != expected_command:
        raise ArmError("selected geometry command is not an exact short-dose TEMP derivation")

    report, report_ref = _load_json_object(
        report_path.expanduser().resolve(strict=True), label="selected geometry report"
    )
    expected_report_path = Path(str(corrected._option(command, "--report"))).resolve()  # noqa: SLF001
    if Path(report_ref["path"]) != expected_report_path:
        raise ArmError("selected geometry report is not the sealed ddp8-b512 output")
    expected_report = {
        "world_size": 8,
        "batch_size": 512,
        "effective_global_batch_size": 4096,
        "max_steps": SELECTED_OPTIMIZER_STEPS,
        "steps_completed": SELECTED_OPTIMIZER_STEPS,
        "base_training_row_draws": SELECTED_GLOBAL_ROW_DOSE,
        "total_training_row_draws": SELECTED_GLOBAL_ROW_DOSE,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "lr": 3e-5,
        "lr_schedule": "flat",
        "lr_warmup_steps": 100,
        "value_lr_mult": 0.3,
        "action_module_lr_mult": 1.0,
        "soft_target_weight": 0.9,
        "soft_target_temperature": 0.7,
        "value_loss_weight": 0.25,
        "value_target_lambda": 1.0,
        "value_head_type": "mse",
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "policy_kl_anchor_weight": 0.0,
        "q_loss_weight": 0.0,
        "mask_hidden_info": True,
        "graph_history_features": True,
        "training_rng_rank_offset": True,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "freeze_modules": "",
        "require_only_trainable_prefixes": "",
        "init_checkpoint_sha256": source["initialization"]["sha256"],
        "data": source["descriptor"]["path"],
        "input_validation_game_sentinel_manifest": source["validation_sentinel"]["path"],
        "checkpoint": str(corrected._option(command, "--checkpoint")),  # noqa: SLF001
    }
    drift = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected_report.items()
        if report.get(key) != value
    }
    if drift:
        raise ArmError(f"executed selected-geometry report drift: {drift}")
    recipe = _selected_recipe()
    evidence = {
        "plan": plan_ref,
        "plan_sha256": plan["plan_sha256"],
        "run_id": GEOMETRY_RUN_ID,
        "run_command_sha256": run["command_sha256"],
        "report": report_ref,
        "runtime": dict(runtime),
    }
    evidence["evidence_sha256"] = corrected._digest(evidence)  # noqa: SLF001
    return {
        "command": list(command),
        "recipe": recipe,
        "recipe_sha256": corrected._digest(recipe),  # noqa: SLF001
        "evidence": evidence,
        "runtime_repo": runtime_repo,
        "trainer": trainer,
    }


def _load_source(
    source_manifest: Path,
    selected_dose_plan: Path,
    selected_dose_report: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    source, source_ref = _load_temperature_source(source_manifest)
    geometry = _load_selected_geometry(
        selected_dose_plan, selected_dose_report, source=source
    )
    return {
        "manifest_sha256": source["manifest"]["manifest_sha256"],
        "initialization": source["initialization"],
        "descriptor": source["descriptor"],
        "validation_sentinel": source["validation_sentinel"],
        "recipe": geometry["recipe"],
        "recipe_sha256": geometry["recipe_sha256"],
        "command": geometry["command"],
        "selected_geometry_evidence": geometry["evidence"],
        "selected_geometry_runtime_repo": str(geometry["runtime_repo"]),
        "selected_geometry_trainer": str(geometry["trainer"]),
    }, source_ref


def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ArmError(f"checkpoint is not a mapping: {path}")
    return raw


def _config_fields(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        fields = raw.get("fields", raw)
        if isinstance(fields, Mapping):
            return dict(fields)
    if dataclasses.is_dataclass(raw):
        return {
            field.name: getattr(raw, field.name)
            for field in dataclasses.fields(raw)
            if hasattr(raw, field.name)
        }
    raise ArmError("checkpoint config cannot be normalized")


def _effective_config(raw: Any) -> dict[str, Any]:
    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    values = _config_fields(raw)
    known = {field.name for field in dataclasses.fields(EntityGraphConfig)}
    try:
        return dataclasses.asdict(EntityGraphConfig(**{
            key: value for key, value in values.items() if key in known
        }))
    except (TypeError, ValueError) as error:
        raise ArmError(f"checkpoint config cannot instantiate current schema: {error}") from error


def _equal_artifact_value(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_equal_artifact_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right) and len(left) == len(right)
            and all(_equal_artifact_value(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def _validate_upgrade(source: Path, upgraded: Path) -> dict[str, Any]:
    """Prove the checkpoint delta is exactly the inert gather branch."""
    import torch

    source_ref = corrected._file_ref(source)  # noqa: SLF001
    upgraded_ref = corrected._file_ref(upgraded)  # noqa: SLF001
    before = _torch_load(Path(source_ref["path"]))
    after = _torch_load(Path(upgraded_ref["path"]))
    provenance = after.get("upgrade_provenance")
    if not isinstance(provenance, Mapping) or not (
        provenance.get("schema_version") == "entity-graph-upgrade-v1"
        and provenance.get("source_checkpoint_sha256")
        == source_ref["sha256"].removeprefix("sha256:")
        and provenance.get("flags") == {"action_target_gather": True}
        and provenance.get("forward_max_diff") == 0.0
        and provenance.get("forward_identical_at_init") is True
        and provenance.get("trained_value_readouts_added") == []
    ):
        raise ArmError("gather checkpoint lacks exact function-preserving provenance")
    source_config = _effective_config(before.get("config"))
    treatment_config = _effective_config(after.get("config"))
    expected_treatment_config = dict(source_config)
    expected_treatment_config["action_target_gather"] = True
    if treatment_config != expected_treatment_config or not (
        treatment_config.get("state_trunk", "transformer") == "transformer"
        and treatment_config.get("action_target_gather") is True
        and int(treatment_config.get("action_cross_attention_layers", 0)) == 0
        and treatment_config.get("edge_policy_head", False) is False
        and treatment_config.get("value_attention_pool", False) is False
        and treatment_config.get("relational_block_pattern", "") == ""
        and int(treatment_config.get("relational_ff_size", 0)) == 0
    ):
        raise ArmError("upgraded checkpoint effective config delta is not gather-only")
    provenance_drift = [
        key for key in before
        if key not in {"model", "config", "upgrade_provenance"}
        and (key not in after or not _equal_artifact_value(before[key], after[key]))
    ]
    if provenance_drift:
        raise ArmError(f"gather upgrade changed/dropped source provenance: {provenance_drift}")
    before_model, after_model = before.get("model"), after.get("model")
    if not isinstance(before_model, Mapping) or not isinstance(after_model, Mapping):
        raise ArmError("checkpoint model state is malformed")
    removed = sorted(set(before_model) - set(after_model))
    added = sorted(set(after_model) - set(before_model))
    if removed or tuple(added) != EXPECTED_NEW_PARAMETERS:
        raise ArmError(f"gather parameter identity drift: added={added} removed={removed}")
    changed = [
        name for name in before_model
        if not torch.equal(before_model[name], after_model[name])
    ]
    if changed:
        raise ArmError(f"shared f7 parameters changed during gather upgrade: {changed[:8]}")
    expected_init = {
        "target_gather_proj.0.weight": "ones",
        "target_gather_proj.0.bias": "zeros",
        "target_gather_proj.1.weight": "zeros",
        "target_gather_proj.1.bias": "zeros",
    }
    for name, kind in expected_init.items():
        tensor = after_model[name]
        reference = torch.ones_like(tensor) if kind == "ones" else torch.zeros_like(tensor)
        if not torch.equal(tensor, reference):
            raise ArmError(f"new gather parameter is not deterministic {kind}: {name}")
    return {
        "utility": "tools/f69_upgrade_checkpoint_config.py --flags gather",
        "source": source_ref,
        "upgraded": upgraded_ref,
        "flags": {"action_target_gather": True},
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
        "shared_parameter_count": len(before_model),
        "shared_parameters_bit_identical": True,
        "new_parameters": added,
        "new_parameter_initialization": expected_init,
    }


def _validate_coverage(path: Path, descriptor_path: Path) -> dict[str, Any]:
    audit, audit_ref = corrected._load_json(path)  # noqa: SLF001
    if audit.get("schema_version") != "memmap-architecture-target-audit-bundle-v1":
        raise ArmError("architecture target audit schema drift")
    descriptor, _ = corrected._preflight_descriptor(descriptor_path)  # noqa: SLF001
    components = descriptor["components"]
    supervised_ids = set(descriptor.get("policy_distillation_component_ids", ())) | set(
        descriptor.get("value_training_component_ids", ())
    )
    component_by_id = {row.get("component_id"): row for row in components}
    if not supervised_ids or not supervised_ids <= set(component_by_id):
        raise ArmError("TEMP descriptor has an invalid supervised component scope")
    # The exact TEMP control supervises policy and value on every component,
    # including predecessor replay.  The gather treatment must therefore prove
    # valid target bindings for the complete supervised mixture; auditing only
    # current rows would silently change the effective treatment population.
    # Bind by resolved corpus identity; audit ordering is not semantic.
    expected_dirs = {
        str(Path(component_by_id[component_id]["corpus_dir"]).resolve())
        for component_id in supervised_ids
    }
    rows = audit.get("audits")
    audited_dirs = [row.get("corpus_dir") for row in rows] if isinstance(rows, list) else []
    if (
        len(audited_dirs) != len(expected_dirs)
        or len(set(audited_dirs)) != len(audited_dirs)
        or set(audited_dirs) != expected_dirs
    ):
        raise ArmError("coverage audit does not bind exactly the supervised TEMP corpora")
    coverage = []
    for row in rows:
        legal = row.get("legal_action_targets", {})
        graph = row.get("graph_incidence", {})
        viability = row.get("viability", {})
        if not (
            viability.get("action_target_gather") is True
            and graph.get("out_of_range_ids") == 0
            and legal.get("invalid_legal_action_ids") == 0
            and legal.get("out_of_range_target_rows") == 0
            and legal.get("search_active_rows_with_any_target", 0) > 0
            and legal.get("actions_with_any_target", 0) > 0
        ):
            raise ArmError("TEMP corpus lacks valid, learnable topology target coverage")
        coverage.append({
            "corpus_dir": row["corpus_dir"],
            "actions": legal.get("actions"),
            "actions_with_any_target": legal.get("actions_with_any_target"),
            "target_coverage": legal.get("target_coverage"),
            "rows_with_any_target": legal.get("rows_with_any_target"),
            "row_target_coverage": legal.get("row_target_coverage"),
            "search_active_rows_with_any_target": legal["search_active_rows_with_any_target"],
            "chosen_actions_with_any_target": legal.get("chosen_actions_with_any_target"),
        })
    return {"artifact": audit_ref, "components": coverage,
            "coverage_sha256": corrected._digest(coverage)}  # noqa: SLF001


def _source_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=repo, text=True).strip()
        subprocess.run(("git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES), cwd=repo, check=True)
        for relative in SOURCE_FILES:
            subprocess.run(("git", "ls-files", "--error-unmatch", relative), cwd=repo,
                           check=True, stdout=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmError("topology arm sources must be clean tracked canonical bytes") from error
    files = {relative: corrected._file_ref(repo / relative) for relative in SOURCE_FILES}  # noqa: SLF001
    return {"repository_root": str(repo), "git_commit": commit, "files": files,
            "files_sha256": corrected._digest(files)}  # noqa: SLF001


def _derive_command(source: Sequence[str], *, upgraded: Path, output_root: Path) -> tuple[list[str], dict[str, Any]]:
    command = list(source)
    changes: dict[str, Any] = {}
    updates = {
        "--init-checkpoint": str(upgraded),
        "--checkpoint": str(output_root / "candidate.pt"),
        "--report": str(output_root / "train.report.json"),
        # Preserve the selected *row* dose while giving a fresh adapter enough
        # optimizer updates to be an interpretable mechanism test.
        "--batch-size": str(LOCAL_BATCH_SIZE),
        "--max-steps": str(OPTIMIZER_STEPS),
        "--action-module-lr-mult": str(ACTION_MODULE_LR_MULT),
        # Every value readout is frozen below.  Leaving the TEMP control's 0.3
        # multiplier would correctly fail closed because the value group is empty.
        "--value-lr-mult": "1.0",
        "--freeze-modules": FREEZE_MODULES,
        "--require-only-trainable-prefixes": TRAINABLE_PREFIX,
    }
    appendable = {"--freeze-modules", "--require-only-trainable-prefixes"}
    for flag, value in updates.items():
        if flag in appendable and flag not in command:
            old = "absent"
        else:
            old = corrected._option(command, flag)  # noqa: SLF001
        corrected._set_option(command, flag, value)  # noqa: SLF001
        changes[flag] = {"source": old, "treatment": value}
    return command, changes


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    repo = args.repo.expanduser().resolve(strict=True)
    source, source_ref = _load_source(
        args.source_manifest,
        args.selected_dose_plan,
        args.selected_dose_report,
    )
    output_root = args.output_root.expanduser().resolve()
    for name in ("candidate.pt", "candidate.pt.optimizer.pt", "train.report.json"):
        if (output_root / name).exists():
            raise ArmError(f"refusing existing topology-arm output: {output_root / name}")
    source_init = Path(source["initialization"]["path"])
    upgraded = args.gather_checkpoint.expanduser().resolve(strict=True)
    upgrade = _validate_upgrade(source_init, upgraded)
    coverage = _validate_coverage(args.architecture_audit, Path(source["descriptor"]["path"]))
    command, changes = _derive_command(source["command"], upgraded=upgraded, output_root=output_root)
    descriptor_meta, _ = corrected._preflight_descriptor(  # noqa: SLF001
        Path(source["descriptor"]["path"])
    )
    event_history_contract, event_history_changes = (
        corrected._bind_event_history_training_command(  # noqa: SLF001
            command, descriptor_meta
        )
    )
    changes.update(event_history_changes)
    # Data, targets, row dose, initialization ancestry, and validation remain
    # bound to the selected TEMP control.  Optimizer geometry intentionally
    # differs and is declared below: this is adapter commissioning, not a
    # same-optimizer TEMP A/B.
    source_binding = _source_binding(repo)
    executor_ref = source_binding.get("files", {}).get(EXECUTOR_RELATIVE_PATH)
    if not isinstance(executor_ref, dict):
        raise ArmError("source binding does not authenticate the topology executor")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": f"{EXECUTOR_RELATIVE_PATH} --go",
        "diagnostic_executor": executor_ref,
        "source_temperature_manifest": source_ref,
        "source_temperature_manifest_sha256": source["manifest_sha256"],
        "selected_geometry_evidence": source["selected_geometry_evidence"],
        "source_recipe": source["recipe"],
        "source_recipe_sha256": source["recipe_sha256"],
        "descriptor": source["descriptor"],
        "validation_sentinel": source["validation_sentinel"],
        "initialization_source": source["initialization"],
        "initialization_treatment": upgrade["upgraded"],
        "function_preserving_upgrade": upgrade,
        "corpus_topology_target_coverage": coverage,
        "event_history_training_contract": event_history_contract,
        "source_binding": source_binding,
        "only_declared_optimization_delta": (
            "commission function-preserving target_gather_proj only"
        ),
        "matched_contract": {
            "recipe_sha256": source["recipe_sha256"],
            "descriptor": source["descriptor"],
            "validation_sentinel": source["validation_sentinel"],
            "row_dose_and_objective_operator_unchanged": True,
            "sampler_batch_partition_unchanged": False,
            "optimizer_state_reused": False,
            "step0_network_outputs_bit_identical": True,
        },
        "adapter_commissioning_contract": {
            "reference_checkpoint": source["initialization"],
            "candidate_chaining": False,
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "optimizer_steps": OPTIMIZER_STEPS,
            "global_row_dose": SELECTED_GLOBAL_ROW_DOSE,
            "lr_warmup_steps": 100,
            "integrated_lr_step_equivalents": 974.5,
            "action_module_lr_mult": ACTION_MODULE_LR_MULT,
            "action_integrated_lr_step_equivalents": 3898.0,
            "freeze_modules": FREEZE_MODULES.split(","),
            "required_trainable_prefixes": [TRAINABLE_PREFIX],
            "mature_parameters_trainable": False,
            "interpretation": (
                "tests whether fixed f7 target-token features contain useful "
                "action-local signal; it is not a joint learner candidate"
            ),
        },
        "allowlisted_command_changes": changes,
        "command": command,
        "command_sha256": corrected._digest(command),  # noqa: SLF001
        "executor_compatibility": {
            "executor": f"{EXECUTOR_RELATIVE_PATH} --go",
            "receipt_schema": "a1-topology-gather-arm-execution-receipt-v3",
            "compatible_now": True,
            "idle_topology": "exactly_8_visible_B200s",
            "one_shot": True,
        },
    }
    manifest["manifest_sha256"] = corrected._digest(manifest)  # noqa: SLF001
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "topology-gather-arm.manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ArmError(f"prepared manifest drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    return manifest, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--selected-dose-plan", required=True, type=Path)
    parser.add_argument("--selected-dose-report", required=True, type=Path)
    parser.add_argument("--gather-checkpoint", required=True, type=Path)
    parser.add_argument("--architecture-audit", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo", default=REPO_ROOT, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    manifest, path = prepare(build_parser().parse_args(argv))
    print(json.dumps({"prepared": str(path), "launched": False,
                      "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
