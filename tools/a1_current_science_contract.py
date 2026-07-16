#!/usr/bin/env python3
"""One machine-readable authority for the current A1 improvement operator.

Issued PIMC locks remain replayable by their historical verifiers.  This
module governs only the new coherent-public operator and supplies projections
for generation sealing, one-dose learning, fleet evaluation, and promotion.
"""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "configs/operations/a1-next-wave-coherent-public-v3/science.contract.json"
)
TEMPLATE_PATH = REPO_ROOT / "configs/experiments/a1_pre_wave_contract.template.json"
GENERATOR_CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/next_wave/coherent_public_n128_adaptive256_forced_value_v3.schema15.json"
)
GENERATOR_GUARD_PATH = (
    REPO_ROOT
    / "configs/guards/a1_generation_coherent_public_n128_adaptive256_forced_value_v3.json"
)
SCHEMA_VERSION = "a1-current-science-contract-v2"
TEACHER_REPORT_SCHEMA = "teacher-operator-causal-report-v1"
ADOPTION_RECEIPT_SCHEMA = "a1-teacher-operator-adoption-v1"
ADAPTIVE_FIELDS = (
    "n_full_wide",
    "n_full_wide_threshold",
    "wide_roots_always_full",
)
POLICY_TARGET_BLEND_FALLBACK_V2 = "policy_target_fallback_v2"
CURRENT_TEACHER_ENTITY_ADAPTER = (
    "rust_entity_adapter_v2_land_topology_ports_maritime"
)
CURRENT_LEARNER_ENTITY_ADAPTER = (
    "rust_entity_adapter_v5_meaningful_history_v2"
)
CURRENT_ARCHITECTURE_UPGRADE_FLAGS = (
    "structured_action_value,card_count_v2,meaningful_history,"
    "history_target_gather,public_rule_state"
)
CURRENT_ARCHITECTURE_UPGRADE_MODULE = (
    "entity_graph.static_action_residual+legal_action_value_residual+"
    "public_card_count_features+meaningful_public_history+"
    "meaningful_history_target_gather+"
    "actor_public_rule_state.v5"
)
CURRENT_MEANINGFUL_HISTORY_POOLING = "ordered_attention_v2"
PRODUCTION_LEARNER_SIGNAL_CONTRACT = {
    # The checked-in recipe is a representation-learning run from random
    # initialization, not the f7 trust-region commissioning experiment.  It
    # therefore trains by complete corpus coverage rather than inheriting the
    # fine-tune frontier's 32-step cap.
    "world_size": 1,
    "batch_size": 4096,
    "global_batch_size": 4096,
    "grad_accum_steps": 1,
    "epochs": 3,
    "max_steps": 0,
    "resume_optimizer": False,
    "optimizer": "adamw",
    "lr": 6e-5,
    "lr_warmup_steps": 100,
    "lr_schedule": "cosine",
    "weight_decay": 0.01,
    "fused_optimizer": True,
    "amp": "bf16",
    "value_lr_mult": 1.0,
    # Policy/value gradients are frequently opposed in the shared trunk. Keep
    # the value head fully trainable while halving only its shared-trunk
    # contribution.
    "value_trunk_grad_scale": 0.5,
    # Reporting is part of production admission even though it does not alter
    # the optimizer trajectory. The full scratch dose must prove every
    # commissioned v5 path received gradient and an actual update.
    "train_diagnostics_every_batches": 16,
    "objective_gradient_interference_every_batches": 16,
    "require_feature_learning_signal_modules": (
        "event_encoder,legal_action_value_residual_proj,"
        "legal_action_value_static_proj,meaningful_history_residual_gate,"
        "meaningful_history_ordered_gate,meaningful_history_sequence,"
        "meaningful_history_target_proj,"
        "public_card_count_residual,public_rule_state_residual,"
        "static_action_residual_proj"
    ),
    "minimum_feature_learning_signal_observations": 2,
    "final_vp_loss_weight": 0.05,
    "symmetry_augment": True,
    # History action ids and board-entity target ids must rotate with the board.
    # Sealing this separately prevents a future parser-default change from
    # producing geometrically contradictory augmented rows.
    "symmetry_augment_events": True,
    # The coherent corpus's natural policy-active distribution assigns only
    # 34.16% of policy objective mass to ordinary PLAY_TURN decisions; the
    # successful selected-dose corpus assigned 66.08%.  Fourfold PLAY_TURN
    # weighting restores 66.49% after the existing equal-per-game
    # normalization, keeping mandatory prompts supervised without letting
    # them dominate the strategic policy update.
    "phase_weights": "PLAY_TURN=4.0",
    # Policy phase repair must not silently starve opening/robber/discard value
    # calibration.
    "value_phase_weights": "none",
}
PRODUCTION_LEARNER_INITIALIZATION_CONTRACT = {
    "mode": "from_scratch",
    "entity_feature_adapter_version": (
        "rust_entity_adapter_v5_meaningful_history_v2"
    ),
    "checkpoint": None,
    "optimizer_state": "fresh",
}
PRODUCTION_LEARNER_MODEL_CONSTRUCTION_CONTRACT = {
    "arch": "entity_graph",
    "hidden_size": 640,
    "graph_tokens": None,
    "graph_layers": 6,
    "attention_heads": 8,
    "graph_dropout": 0.05,
    "entity_state_trunk": "transformer",
    "static_action_residual": True,
    "legal_action_value_residual": True,
    "public_card_count_features": True,
    "public_card_count_residual_bias": False,
    "public_rule_state_features": True,
    "public_rule_state_feature_schema": "actor_public_rule_state_2p_v1",
    "actor_public_rule_state": (
        "dev_used_road_building_free_roads_discard_remainder_playable_dev_counts"
    ),
    "meaningful_public_history": True,
    "meaningful_public_history_schema": (
        "meaningful_public_history_2p_no_trade_v2"
    ),
    "meaningful_public_history_pooling": CURRENT_MEANINGFUL_HISTORY_POOLING,
    "meaningful_public_history_target_gather": True,
    "event_history_limit": 64,
    "mask_hidden_info": True,
    "entity_feature_adapter_version": (
        "rust_entity_adapter_v5_meaningful_history_v2"
    ),
    "require_35m_model": True,
}
PRODUCTION_LEARNER_EXECUTION_TOPOLOGY_CONTRACT = {
    "schema_version": "a1-scratch-training-topology-v1",
    "launcher": "torch.distributed.run",
    "name": "b200-8gpu-ddp",
    "world_size": 8,
    "physical_gpus": list(range(8)),
    "local_batch_size": 512,
    "grad_accum_steps": 1,
    "global_batch_size": 4096,
    "ddp_shard_data": False,
    "training_rng_rank_offset": True,
    "optimization_schedule_status": "unresolved",
    "go_authorized": False,
    "reviewed_optimizer_schedule_role": (
        "checkpoint_initialized_diagnostic_canary_only"
    ),
}
DIAGNOSTIC_POLICY_AUX_FIELDS = frozenset(
    {"policy_aux_active_batch_size", "policy_aux_loss_weight"}
)
PRODUCTION_TARGET_QUALITY_LEARNER_CONTRACT = {
    # Raw search disagreement is not target correctness. Production may only
    # prioritize it again after the reliability audit has qualified a recipe.
    "policy_surprise_weight": 0.0,
}
PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT = {
    # A stable root hash selects this audit slice without consuming gameplay or
    # primary-search RNG. Five percent is enough to calibrate target stability
    # without turning the duplicate reference search into the dominant cost.
    "target_reliability_audit_fraction": 0.05,
    "target_reliability_audit_seed": 20260716,
    # Completed-Q and visit evidence cost far less than one percent of the
    # corpus and are required to calibrate or reconstruct teacher reliability.
    "preserve_search_evidence": True,
}


class ScienceContractError(ValueError):
    """The current production science contract is malformed or drifted."""


def _load() -> dict[str, Any]:
    try:
        value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceContractError(
            f"cannot load current science contract {CONTRACT_PATH}: {error}"
        ) from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ScienceContractError("current science contract schema drift")
    required = {
        "schema_version",
        "contract_id",
        "operator_selection",
        "target_information_regime",
        "operator",
        "generation",
        "learner",
        "evaluation",
        "promotion",
    }
    if set(value) != required:
        raise ScienceContractError(
            "current science contract top-level fields drifted: "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    operator = value.get("operator")
    if not isinstance(operator, dict) or set(operator) != {"search", "evaluator"}:
        raise ScienceContractError("current operator must bind search and evaluator")
    for key in ("search", "evaluator"):
        if not isinstance(operator[key], dict) or not operator[key]:
            raise ScienceContractError(f"current operator.{key} is empty")
    for key in ("generation", "learner", "evaluation", "promotion"):
        if not isinstance(value[key], dict) or not value[key]:
            raise ScienceContractError(f"current {key} contract is empty")
    selection = value["operator_selection"]
    if not isinstance(selection, dict) or selection.get("status") not in {
        "provisional_pending_teacher_campaign",
        "adopted_teacher_campaign",
    }:
        raise ScienceContractError("current operator selection status is invalid")
    if (
        selection.get("report_schema") != TEACHER_REPORT_SCHEMA
        or selection.get("mutable_fields") != list(ADAPTIVE_FIELDS)
    ):
        raise ScienceContractError("current operator selection authority drifted")
    if selection["status"] == "adopted_teacher_campaign":
        selected_fields = _selected_adaptive_fields(
            str(selection.get("selected_operator"))
        )
        actual_fields = {
            key: operator["search"].get(key) for key in ADAPTIVE_FIELDS
        }
        if actual_fields != selected_fields or not isinstance(
            selection.get("report"), dict
        ):
            raise ScienceContractError("adopted teacher operator evidence drifted")
    learner_value = value["learner"]
    if set(learner_value) != {
        "initialization",
        "architecture_upgrade_flags",
        "architecture_upgrade_module",
        "model_construction",
        "execution_topology",
        "topology",
        "training_recipe",
    } or not isinstance(learner_value["training_recipe"], dict):
        raise ScienceContractError("current learner contract shape drifted")
    if learner_value.get("initialization") != PRODUCTION_LEARNER_INITIALIZATION_CONTRACT:
        raise ScienceContractError(
            "current coherent learner initialization must be native from-scratch "
            "v5 with fresh optimizer state"
        )
    if (
        learner_value.get("model_construction")
        != PRODUCTION_LEARNER_MODEL_CONSTRUCTION_CONTRACT
    ):
        raise ScienceContractError("current scratch model construction drifted")
    if (
        learner_value.get("execution_topology")
        != PRODUCTION_LEARNER_EXECUTION_TOPOLOGY_CONTRACT
        or learner_value.get("topology")
        != PRODUCTION_LEARNER_EXECUTION_TOPOLOGY_CONTRACT["name"]
    ):
        raise ScienceContractError("current scratch execution topology drifted")
    recipe = learner_value["training_recipe"]
    execution = learner_value["execution_topology"]
    if (
        execution["world_size"]
        * execution["local_batch_size"]
        * execution["grad_accum_steps"]
        != execution["global_batch_size"]
        or execution["global_batch_size"] != recipe.get("global_batch_size")
    ):
        raise ScienceContractError(
            "current scratch execution topology changes the logical global dose"
        )
    if (
        learner_value.get("architecture_upgrade_flags")
        != CURRENT_ARCHITECTURE_UPGRADE_FLAGS
        or learner_value.get("architecture_upgrade_module")
        != CURRENT_ARCHITECTURE_UPGRADE_MODULE
    ):
        raise ScienceContractError(
            "current coherent learner architecture upgrade authority drifted"
        )
    if (
        recipe.get("policy_target_blend_semantics")
        != POLICY_TARGET_BLEND_FALLBACK_V2
        or recipe.get("soft_target_weight") != 1.0
        or recipe.get("soft_target_source") != "policy"
        or recipe.get("soft_target_min_legal_coverage") != 1.0
    ):
        raise ScienceContractError(
            "current coherent learner must bind complete authenticated policy CE "
            "with no reachable hard-action fallback"
        )
    learner_signal_drift = {
        key: {
            "expected": expected,
            "actual": recipe.get(key),
        }
        for key, expected in PRODUCTION_LEARNER_SIGNAL_CONTRACT.items()
        if recipe.get(key) != expected
    }
    if learner_signal_drift:
        raise ScienceContractError(
            "current coherent learner inherited a diagnostic/approximate training "
            f"setting: {learner_signal_drift}"
        )
    leaked_aux_fields = sorted(DIAGNOSTIC_POLICY_AUX_FIELDS & set(recipe))
    if leaked_aux_fields:
        raise ScienceContractError(
            "current coherent base learner must not bind diagnostic active-policy "
            f"AUX fields: {leaked_aux_fields}"
        )
    target_quality_learner_drift = {
        key: {"expected": expected, "actual": recipe.get(key)}
        for key, expected in PRODUCTION_TARGET_QUALITY_LEARNER_CONTRACT.items()
        if recipe.get(key) != expected
    }
    if target_quality_learner_drift:
        raise ScienceContractError(
            "current coherent learner target-quality contract drifted: "
            f"{target_quality_learner_drift}"
        )
    if bool(recipe.get("target_reliability_confidence_weighting", False)):
        raise ScienceContractError(
            "the five-percent duplicate-search audit is diagnostic evidence, "
            "not corpus-wide learner weighting"
        )
    if bool(recipe.get("per_game_policy_surprise_weighting", False)):
        raise ScienceContractError(
            "per-game surprise weighting is not part of the production learner"
        )
    if float(recipe.get("public_card_lr_mult", 1.0)) != 1.0:
        raise ScienceContractError(
            "from-scratch public-card features must use the base optimizer LR"
        )
    generation_value = value["generation"]
    if (
        generation_value.get("teacher_entity_feature_adapter_version")
        != CURRENT_TEACHER_ENTITY_ADAPTER
        or generation_value.get("learner_entity_feature_adapter_version")
        != CURRENT_LEARNER_ENTITY_ADAPTER
    ):
        raise ScienceContractError(
            "current coherent generation teacher/learner adapter authority drifted"
        )
    target_quality_generation_drift = {
        key: {"expected": expected, "actual": generation_value.get(key)}
        for key, expected in PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT.items()
        if generation_value.get(key) != expected
    }
    if target_quality_generation_drift:
        raise ScienceContractError(
            "current coherent generation target-quality contract drifted: "
            f"{target_quality_generation_drift}"
        )
    evaluator_value = operator["evaluator"]
    if (
        evaluator_value.get("value_readout") == "scalar"
        and evaluator_value.get("value_squash") == "tanh"
        and (
            recipe.get("scalar_value_loss_readout") != "deployed_tanh"
            or recipe.get("scalar_value_loss_scale")
            != evaluator_value.get("value_scale")
        )
    ):
        raise ScienceContractError(
            "current scalar learner must optimize the exact deployed tanh "
            "search readout and scale"
        )
    _validate_target_quality_artifacts(value)
    return value


def load() -> dict[str, Any]:
    return copy.deepcopy(_load())


def search() -> dict[str, Any]:
    return copy.deepcopy(_load()["operator"]["search"])


def evaluator() -> dict[str, Any]:
    return copy.deepcopy(_load()["operator"]["evaluator"])


def generation() -> dict[str, Any]:
    return copy.deepcopy(_load()["generation"])


def learner() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"])


def learner_training_recipe() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"]["training_recipe"])


def learner_initialization() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"]["initialization"])


def learner_model_construction() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"]["model_construction"])


def learner_execution_topology() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"]["execution_topology"])


def target_information_regime() -> str:
    return str(_load()["target_information_regime"])


def operator_selection_status() -> str:
    return str(_load()["operator_selection"]["status"])


def is_coherent_search(value: Mapping[str, Any]) -> bool:
    return value.get("coherent_public_belief_search") is True


def require_current_operator(
    *,
    search_value: Mapping[str, Any],
    evaluator_value: Mapping[str, Any] | None = None,
    generation_value: Mapping[str, Any] | None = None,
    learner_recipe_value: Mapping[str, Any] | None = None,
    target_regime: str | None = None,
    require_adopted: bool = False,
) -> None:
    """Fail closed when a current coherent-public authority drifts.

    Callers invoke this only for coherent-public locks/drafts.  Historical
    information-set/PIMC objects therefore retain their original semantics.
    """

    expected_search = search()
    if require_adopted and operator_selection_status() != "adopted_teacher_campaign":
        raise ScienceContractError(
            "coherent-public teacher operator is provisional; aggregate and adopt "
            "the causal teacher campaign before sealing a production wave"
        )
    if dict(search_value) != expected_search:
        differing = sorted(
            key
            for key in set(search_value) | set(expected_search)
            if search_value.get(key) != expected_search.get(key)
        )
        raise ScienceContractError(
            f"coherent-public search differs from current science contract: {differing}"
        )
    if evaluator_value is not None:
        expected_evaluator = evaluator()
        actual_evaluator = {
            key: evaluator_value.get(key) for key in expected_evaluator
        }
        if actual_evaluator != expected_evaluator:
            differing = sorted(
                key
                for key in expected_evaluator
                if actual_evaluator.get(key) != expected_evaluator.get(key)
            )
            raise ScienceContractError(
                "coherent-public evaluator differs from current science contract: "
                f"{differing}"
            )
    if generation_value is not None:
        expected_generation = generation()
        actual_generation = {
            key: generation_value.get(key) for key in expected_generation
        }
        if actual_generation != expected_generation:
            differing = sorted(
                key
                for key in expected_generation
                if actual_generation.get(key) != expected_generation.get(key)
            )
            raise ScienceContractError(
                "coherent-public generation differs from current science contract: "
                f"{differing}"
            )
    if learner_recipe_value is not None:
        expected_recipe = learner_training_recipe()
        actual_recipe = dict(learner_recipe_value)
        if actual_recipe != expected_recipe:
            differing = sorted(
                key
                for key in set(actual_recipe) | set(expected_recipe)
                if actual_recipe.get(key) != expected_recipe.get(key)
            )
            raise ScienceContractError(
                "coherent-public learner recipe differs from current science "
                f"contract: {differing}"
            )
    if target_regime is not None and target_regime != target_information_regime():
        raise ScienceContractError(
            "coherent-public target-information regime differs from current "
            f"science contract: {target_regime!r}"
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScienceContractError(f"{path} must contain a JSON object")
    return value


def _validate_target_quality_artifacts(contract: Mapping[str, Any]) -> None:
    """Bind the generator recipe and guard to the target-quality authority."""

    generator = _read_object(GENERATOR_CONFIG_PATH)
    fields = generator.get("fields")
    if not isinstance(fields, dict):
        raise ScienceContractError("coherent generator config fields are missing")
    generation = contract["generation"]
    search_value = contract["operator"]["search"]
    expected_generator = {
        **PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT,
        "exact_budget_sh": search_value.get("exact_budget_sh"),
        "exact_budget_sh_min_n": search_value.get("exact_budget_sh_min_n"),
        "learner_entity_feature_adapter_version": CURRENT_LEARNER_ENTITY_ADAPTER,
    }
    generator_drift = {
        key: {"expected": expected, "actual": fields.get(key)}
        for key, expected in expected_generator.items()
        if fields.get(key) != expected
    }
    if generator_drift:
        raise ScienceContractError(
            "current coherent generator target-quality config drifted: "
            f"{generator_drift}"
        )
    for key, expected in PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT.items():
        if generation.get(key) != expected:
            raise ScienceContractError(
                f"current generation.{key} differs from generator config"
            )

    guard = _read_object(GENERATOR_GUARD_PATH)
    try:
        lint_args = next(
            item["args"]
            for item in guard["guards"]
            if item.get("name") == "cli_flag_lint"
        )
        critical = set(lint_args["critical_flags"])
        expected_values = lint_args["expected_values"]
    except (KeyError, StopIteration, TypeError) as error:
        raise ScienceContractError("coherent generator guard shape drifted") from error
    guarded = {
        "--target-reliability-audit-fraction": expected_generator[
            "target_reliability_audit_fraction"
        ],
        "--target-reliability-audit-seed": expected_generator[
            "target_reliability_audit_seed"
        ],
        "--exact-budget-sh": expected_generator["exact_budget_sh"],
        "--exact-budget-sh-min-n": expected_generator["exact_budget_sh_min_n"],
        "--learner-entity-feature-adapter-version": (
            expected_generator["learner_entity_feature_adapter_version"]
        ),
    }
    guard_drift = {
        flag: {
            "critical": flag in critical,
            "expected": expected,
            "actual": expected_values.get(flag),
        }
        for flag, expected in guarded.items()
        if flag not in critical or expected_values.get(flag) != expected
    }
    if guard_drift:
        raise ScienceContractError(
            "current coherent generator target-quality guard drifted: "
            f"{guard_drift}"
        )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _selected_adaptive_fields(selected: str) -> dict[str, Any]:
    if selected == "base_n128_d6":
        return {
            "n_full_wide": None,
            "n_full_wide_threshold": None,
            "wide_roots_always_full": False,
        }
    if selected in {"adaptive_n256_w20_d6", "adaptive_n256_w40_d6"}:
        return {
            "n_full_wide": 256,
            "n_full_wide_threshold": 20 if "w20" in selected else 40,
            "wide_roots_always_full": True,
        }
    raise ScienceContractError(f"teacher campaign selected unknown operator {selected!r}")


def adopt_teacher_campaign(
    report_path: str | Path,
    *,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Adopt one completed causal campaign into every production authority.

    The current w20 value is intentionally provisional.  This transaction
    changes only the three preregistered adaptive-dose fields in the contract,
    draft template, typed generator config, and guard.  The canonical contract
    is replaced last so an interrupted transaction can be safely rerun.
    """

    report_path = Path(report_path).expanduser().resolve(strict=True)
    receipt_path = Path(receipt_path).expanduser().resolve(strict=False)
    report = _read_object(report_path)
    if report.get("schema_version") != TEACHER_REPORT_SCHEMA:
        raise ScienceContractError("teacher campaign report schema drift")
    report_content = dict(report)
    reported_content_sha = report_content.pop("report_content_sha256", None)
    if reported_content_sha != _content_sha256(report_content):
        raise ScienceContractError("teacher campaign report content digest mismatch")

    contract = _load()
    selection = contract["operator_selection"]
    if selection.get("status") != "provisional_pending_teacher_campaign":
        raise ScienceContractError("teacher operator has already been adopted")
    before_contract_sha = _file_sha256(CONTRACT_PATH)
    report_authority = report.get("science_contract")
    if not isinstance(report_authority, dict) or (
        report_authority.get("sha256") != before_contract_sha
        or report_authority.get("contract_id") != contract["contract_id"]
        or report_authority.get("experimental_dose_fields")
        != list(ADAPTIVE_FIELDS)
    ):
        raise ScienceContractError(
            "teacher campaign was not run against these provisional contract bytes"
        )
    selected = report.get("selection", {}).get("selected_operator")
    adaptive = _selected_adaptive_fields(str(selected))

    template = _read_object(TEMPLATE_PATH)
    generator = _read_object(GENERATOR_CONFIG_PATH)
    guard = _read_object(GENERATOR_GUARD_PATH)
    search_targets = (
        contract["operator"]["search"],
        template["science"]["search"],
        generator["fields"],
    )
    provisional = {key: contract["operator"]["search"].get(key) for key in ADAPTIVE_FIELDS}
    for target in search_targets:
        observed = {key: target.get(key) for key in ADAPTIVE_FIELDS}
        if observed != provisional:
            raise ScienceContractError(
                f"adaptive authority drift before adoption: {observed} != {provisional}"
            )
        target.update(adaptive)

    try:
        lint_args = next(
            item["args"] for item in guard["guards"] if item.get("name") == "cli_flag_lint"
        )
        critical = lint_args["critical_flags"]
        expected = lint_args["expected_values"]
    except (KeyError, StopIteration, TypeError) as error:
        raise ScienceContractError("coherent generator guard shape drifted") from error
    old_guard = {
        "n_full_wide": expected.get("--n-full-wide"),
        "n_full_wide_threshold": expected.get("--n-full-wide-threshold"),
        "wide_roots_always_full": expected.get("--wide-roots-always-full"),
    }
    if old_guard != provisional:
        raise ScienceContractError(
            f"adaptive guard drift before adoption: {old_guard} != {provisional}"
        )
    for flag in ("--n-full-wide", "--n-full-wide-threshold"):
        while flag in critical:
            critical.remove(flag)
        expected.pop(flag, None)
    if adaptive["n_full_wide"] is not None:
        insertion = critical.index("--wide-roots-always-full")
        critical[insertion:insertion] = ["--n-full-wide", "--n-full-wide-threshold"]
        expected["--n-full-wide"] = adaptive["n_full_wide"]
        expected["--n-full-wide-threshold"] = adaptive["n_full_wide_threshold"]
    expected["--wide-roots-always-full"] = adaptive["wide_roots_always_full"]

    contract["operator_selection"] = {
        "status": "adopted_teacher_campaign",
        "report_schema": TEACHER_REPORT_SCHEMA,
        "mutable_fields": list(ADAPTIVE_FIELDS),
        "selected_operator": selected,
        "report": {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
            "content_sha256": reported_content_sha,
            "checkpoint_sha256": report.get("checkpoint_sha256"),
        },
    }
    receipt = {
        "schema_version": ADOPTION_RECEIPT_SCHEMA,
        "selected_operator": selected,
        "adaptive_fields": adaptive,
        "teacher_report": contract["operator_selection"]["report"],
        "contract_before_sha256": before_contract_sha,
        "artifacts": {
            "template": str(TEMPLATE_PATH),
            "generator_config": str(GENERATOR_CONFIG_PATH),
            "generator_guard": str(GENERATOR_GUARD_PATH),
            "science_contract": str(CONTRACT_PATH),
        },
    }
    _atomic_json(GENERATOR_CONFIG_PATH, generator)
    _atomic_json(GENERATOR_GUARD_PATH, guard)
    _atomic_json(TEMPLATE_PATH, template)
    _atomic_json(CONTRACT_PATH, contract)
    receipt["contract_after_sha256"] = _file_sha256(CONTRACT_PATH)
    receipt["receipt_content_sha256"] = _content_sha256(receipt)
    _atomic_json(receipt_path, receipt)
    return copy.deepcopy(receipt)


def fleet_evaluation_science_config() -> dict[str, Any]:
    """Project the current operator into the H100 evaluator's plan schema."""

    contract = _load()
    search_value = contract["operator"]["search"]
    evaluator_value = contract["operator"]["evaluator"]
    evaluation_value = contract["evaluation"]
    return {
        "internal_map_kind": evaluation_value["internal_map_kind"],
        "external_map_kind": evaluation_value["external_map_kind"],
        "n_full": search_value["n_full"],
        "c_scale": search_value["c_scale"],
        "c_visit": search_value["c_visit"],
        "sigma_eval": search_value["sigma_eval"],
        "rescale_noise_floor_c": search_value["rescale_noise_floor_c"],
        "lazy_interior_chance": search_value["lazy_interior_chance"],
        "correct_rust_chance_spectra": search_value["correct_rust_chance_spectra"],
        "public_observation": evaluator_value["public_observation"],
        "information_set_search": search_value["information_set_search"],
        "belief_chance_spectra": search_value["belief_chance_spectra"],
        "coherent_public_belief_search": search_value[
            "coherent_public_belief_search"
        ],
        "determinization_particles": search_value["determinization_particles"],
        "determinization_min_simulations": search_value[
            "determinization_min_simulations"
        ],
        "forced_root_target_mode": search_value["forced_root_target_mode"],
        "n_full_wide": search_value["n_full_wide"],
        "n_full_wide_threshold": search_value["n_full_wide_threshold"],
        "wide_roots_always_full": search_value["wide_roots_always_full"],
        "symmetry_averaged_eval": search_value["symmetry_averaged_eval"],
        "symmetry_averaged_eval_threshold": search_value[
            "symmetry_averaged_eval_threshold"
        ],
        "evaluator_rust_featurize": evaluator_value["rust_featurize"],
        "native_mcts_hot_loop": contract["generation"]["native_mcts_hot_loop"],
        "value_readout": evaluator_value["value_readout"],
        "value_squash": evaluator_value["value_squash"],
        "max_depth": search_value["max_depth"],
        "max_decisions": contract["generation"]["max_decisions"],
        "max_root_candidates": evaluation_value["max_root_candidates"],
        "max_root_candidates_wide": evaluation_value[
            "max_root_candidates_wide"
        ],
        "wide_candidates_threshold": search_value["wide_candidates_threshold"],
        "gate_config": evaluation_value["gate_config"],
        "external_vps_to_win": evaluation_value["external_vps_to_win"],
        "external_max_player_trade_offers_per_turn": evaluation_value[
            "external_max_player_trade_offers_per_turn"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adopt = subparsers.add_parser(
        "adopt-teacher",
        help="adopt a completed causal teacher report into production authorities",
    )
    adopt.add_argument("--report", required=True)
    adopt.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "adopt-teacher":
        receipt = adopt_teacher_campaign(args.report, receipt_path=args.receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
