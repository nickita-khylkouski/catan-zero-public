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
    / "configs/operations/a1-next-wave-coherent-public-v4/science.contract.json"
)
TEMPLATE_PATH = REPO_ROOT / "configs/experiments/a1_pre_wave_contract_v4.template.json"
GENERATOR_CONFIG_PATH = (
    REPO_ROOT
    / "configs/generation/coherent_public_n128.schema22.json"
)
PRODUCTION_RECIPE_CATALOG_PATH = REPO_ROOT / "configs/production_recipes.json"
GENERATOR_GUARD_PATH = (
    REPO_ROOT
    / "configs/guards/a1_generation_coherent_public_n128_v4.json"
)
CANONICAL_PARENT_UPDATE_CONFIG_PATH = (
    REPO_ROOT / "configs/training/a1_parent_update_35m_b200.schema1.json"
)
TRAINING_SCIENCE_ADMISSION_PATH = (
    REPO_ROOT / "configs/production/training_science_admission.json"
)
PRE_V8_SCRATCH_CONTROL_PATH = (
    REPO_ROOT / "configs/experiments/a1_scratch_pre_v8_compat_control.json"
)
SCHEMA_VERSION = "a1-current-science-contract-v4"
TEACHER_REPORT_SCHEMA = "teacher-operator-causal-report-v1"
ADOPTION_RECEIPT_SCHEMA = "a1-teacher-operator-adoption-v1"
TRAINING_SCIENCE_COMMISSIONING_SCHEMA = (
    "a1-selected-parent-update-commissioning-v1"
)
ADAPTIVE_FIELDS = (
    "n_full_wide",
    "n_full_wide_threshold",
    "wide_roots_always_full",
)
POLICY_TARGET_BLEND_FALLBACK_V2 = "policy_target_fallback_v2"
CURRENT_TEACHER_ENTITY_ADAPTER = "rust_entity_adapter_v2_land_topology_ports_maritime"
CURRENT_LEARNER_ENTITY_ADAPTER = (
    "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
)
CURRENT_ARCHITECTURE_UPGRADE_FLAGS = (
    "current_v8_information_migration_topology_split1"
)
CURRENT_ARCHITECTURE_UPGRADE_MODULE = (
    "entity_graph.current_v2_to_v6_information_contract+topology+split1+"
    "public_resource_v8.v1"
)
CURRENT_MEANINGFUL_HISTORY_POOLING = "ordered_attention_v2"
PRODUCTION_LEARNER_SIGNAL_CONTRACT = {
    # The checked-in recipe is a representation-learning run from random
    # initialization, not the f7 trust-region commissioning experiment.  It
    # therefore trains by complete corpus coverage rather than inheriting the
    # fine-tune frontier's 32-step cap.
    "world_size": 1,
    # Random-init representation learning needs optimizer updates, not merely
    # corpus passes. Global 512 preserves the same three-epoch row exposure
    # while providing roughly 8x more Adam updates than global 4096.
    "batch_size": 512,
    "global_batch_size": 512,
    "grad_accum_steps": 1,
    "epochs": 3,
    "max_steps": 0,
    # Preserve a playing-strength selection frontier inside the full scratch
    # run. Promotion selects among these checkpoints using matched H2H evidence,
    # never the training loss.
    "checkpoint_steps": "8,16,32,64,128,256,512,1024",
    "base_sampler": "coverage_importance_v1",
    # The failed Stage-C corpus exposed only 3.649 Kish-effective policy rows
    # per global-512 update. The exact fresh-corpus preflight (after holdout,
    # phase/per-game weights, forced-row exclusion, and coverage importance)
    # measures 41.002. Require 32: 8.77x above the failed regime with a 28%
    # measured margin on the commissioned corpus.
    "minimum_policy_effective_rows_per_global_batch": 32.0,
    # A global policy ESS floor cannot prove that rare strategic phases survive
    # phase weighting, forced-row exclusion, the train/holdout split, and the
    # authenticated coverage measure. Bind the reviewed per-phase floor already
    # commissioned by the canonical parent-update recipe.
    "minimum_initial_settlement_policy_mass_fraction": 0.02,
    "minimum_initial_road_policy_mass_fraction": 0.02,
    "minimum_discard_policy_mass_fraction": 0.02,
    "minimum_move_robber_policy_mass_fraction": 0.02,
    # The authenticated r4-v4 composite carries 10.4546% maritime soft-label
    # objective mass overall and 10.4209%/10.7970%/9.9672% in its three
    # components. Require 8% globally and per component, leaving 1.967 points
    # of measured margin in the weakest hard-negative source.
    "minimum_maritime_trade_policy_objective_mass_fraction": 0.08,
    # The commissioned scratch model has no routed experts.  The generic
    # trainer default is intentionally useful for MoE experiments, but leaving
    # it implicit here makes coverage_importance_v1 reject the production
    # recipe before the first optimizer step.
    "moe_balance_loss_weight": 0.0,
    "resume_optimizer": False,
    "optimizer": "adamw",
    "lr": 6e-5,
    "lr_warmup_steps": 250,
    "lr_schedule": "cosine",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "fused_optimizer": True,
    "amp": "bf16",
    "scalar_value_objective": "binary_win_bce",
    "value_lr_mult": 1.0,
    # Adam can cancel a constant parameter-group LR interpretation through its
    # first/second moments, so this is an explicit autograd-boundary scale. The
    # private final value block protects the policy readout, while a 0.1
    # boundary gradient still lets terminal outcomes train the shared
    # token/history representation from scratch.  A zero boundary made every
    # value-only row useless to the 5 shared blocks and input encoders.  Keep
    # this scale at the shared/private boundary: the private value tower and
    # readouts remain fully trainable.
    "value_trunk_grad_scale": 0.1,
    # Dormant while policy_dose_lr_area is unresolved/zero. Once a positive
    # policy frontier is sealed, terminal value learning continues only through
    # the private value tower/readouts instead of drifting the policy trunk.
    "post_policy_dose_value_trunk_grad_scale": 0.0,
    # Reporting is part of production admission even though it does not alter
    # the optimizer trajectory. The full scratch dose must prove every
    # commissioned native-V8 path received gradient and an actual update.
    "train_diagnostics_every_batches": 16,
    "objective_gradient_interference_every_batches": 16,
    "require_feature_learning_signal_modules": (
        "action_cross_blocks,event_encoder,target_gather_proj,legal_action_value_residual_proj,"
        "legal_action_value_static_proj,legal_action_value_max_proj,"
        "legal_action_value_count_proj,legal_action_value_static_max_proj,"
        "meaningful_history_residual_gate,"
        "meaningful_history_ordered_gate,meaningful_history_sequence,"
        "meaningful_history_target_proj,"
        "public_card_count_residual,public_rule_state_residual,"
        "static_action_residual_proj,topology_residual_adapter,"
        "value_blocks,value_head,"
        "value_state_norm,final_vp_head"
    ),
    "minimum_feature_learning_signal_observations": 2,
    "final_vp_loss_weight": 0.05,
    # Equal-per-game weighting already prevents long trajectories from
    # dominating.  Do not additionally force every source component to a
    # synthetic 50/50 winner/loser prior: the scalar value consumed by MCTS
    # must remain an expected return under the natural trajectory measure.
    "value_player_outcome_balance_mode": "none",
    # Mechanical actions have no policy target, but their states remain valid
    # value evidence.  The coherent commissioning runs retained both typed
    # boundaries at full value weight; no causal result supports silently
    # reducing END_TURN in the production objective.
    "forced_action_weight": 0.0,
    "forced_row_value_weight": 1.0,
    "forced_row_value_action_type_weights": "END_TURN=1.0,ROLL=1.0",
    # Stored coherent search policies are already normalized teacher targets.
    # train_bc's scalar temperature applies only to score-derived targets;
    # sealing 0.7 here previously described an objective the optimizer never
    # received. Per-component policy calibration must be separately
    # authenticated in the corpus descriptor.
    "soft_target_temperature": 1.0,
    "symmetry_augment": True,
    # History action ids and board-entity target ids must rotate with the board.
    # Sealing this separately prevents a future parser-default change from
    # producing geometrically contradictory augmented rows.
    "symmetry_augment_events": True,
    # The coherent corpus's natural policy-active distribution assigns only
    # 34.16% of policy objective mass to ordinary PLAY_TURN decisions; the
    # successful selected-dose corpus assigned 66.08%. Fourfold PLAY_TURN
    # weighting restores ordinary-play representation, while the audited
    # hard-decision multipliers prevent robber, initial-road, and discard
    # choices from being drowned out by routine turns. The sealed r5 measure
    # placed 52.27% of policy mass on PLAY_TURN after this repair.
    "phase_weights": (
        "PLAY_TURN=4.0,MOVE_ROBBER=3.0,"
        "BUILD_INITIAL_ROAD=2.0,DISCARD=1.5"
    ),
    # Policy phase repair must not silently starve opening/robber/discard value
    # calibration.
    "value_phase_weights": "none",
}
PRODUCTION_LEARNER_INITIALIZATION_CONTRACT = {
    "mode": "from_scratch",
    "entity_feature_adapter_version": (
        "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
    ),
    "checkpoint": None,
    "optimizer_state": "fresh",
}
PRODUCTION_LEARNER_MODEL_CONSTRUCTION_CONTRACT = {
    "arch": "entity_graph",
    # H100 BF16 forward/backward profiling on the exact scratch topology found
    # that width 624 falls off the tensor-core-friendly attention path: width
    # 640 is materially faster and uses less peak memory. The resulting 41.7M
    # checkpoint remains in the same model class, so bind its ceiling explicitly
    # instead of silently relying on train_bc's generic 40M default.
    "hidden_size": 640,
    "graph_tokens": None,
    "graph_layers": 6,
    "attention_heads": 8,
    "graph_dropout": 0.05,
    "entity_state_trunk": "transformer",
    "action_target_gather": True,
    "action_cross_attention_layers": 1,
    "action_cross_attention_bottleneck": 80,
    # Native scratch has no inherited V2--V5 encoders to preserve. Feed the
    # physical V6 player/resource scales and corrected initial-road feature
    # directly to their base encoders instead of reconstructing a legacy view.
    "v6_compatibility_preserving_inputs": False,
    "topology_residual_adapter": True,
    "static_action_residual": True,
    "legal_action_value_residual": True,
    "legal_action_value_set_statistics": True,
    "value_tower_split_layers": 1,
    "public_card_count_features": True,
    "public_card_count_residual_bias": False,
    # With compatibility rewriting disabled, the base public-card projection
    # receives the exact public deduction tensor directly. The separate V8
    # residual is only required by checkpoint upgrades that must retain a
    # mature legacy projection.
    "public_card_exact_resource_residual": False,
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
        "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
    ),
    "require_35m_model": True,
    "min_parameter_count": 42_500_000,
    "max_parameter_count": 43_000_000,
}
PRODUCTION_LEARNER_EXECUTION_TOPOLOGY_CONTRACT = {
    "schema_version": "a1-scratch-training-topology-v1",
    "launcher": "torch.distributed.run",
    "name": "b200-8gpu-ddp",
    "world_size": 8,
    "physical_gpus": list(range(8)),
    "local_batch_size": 64,
    "grad_accum_steps": 1,
    "global_batch_size": 512,
    "ddp_shard_data": False,
    "training_rng_rank_offset": True,
    # The smaller batch creates a plausible update horizon, but it does not
    # resolve the untested AdamW/decay/cosine bundle or exact-zero shared value
    # gradient. Keep execution fail-closed until matched scratch evidence
    # adjudicates those choices.
    "optimization_schedule_status": "unresolved",
    "go_authorized": False,
    "reviewed_optimizer_schedule_role": (
        "candidate_horizon_pending_optimizer_and_value_routing_evidence_v1"
    ),
}
PRODUCTION_LEARNER_SELECTION_CONTRACT = {
    # The full random-initialization learner remains a useful R&D contract, but
    # its optimizer horizon is deliberately unresolved and go_authorized=false.
    # The commissioned production turn is the independently initialized,
    # parent-fresh 12-step update.  Keep that routing decision separate from
    # the scratch construction so callers cannot infer "selected" from which
    # recipe happened to be added most recently.
    "mode": "canonical_parent_update",
    "recipe": "a1-parent-update-35m-b200",
    "config_path": "configs/training/a1_parent_update_35m_b200.schema1.json",
    "config_canonical_sha256": (
        "e14e87a45513c8c2eb5fe1d9e5a603e217b29ef7f9fb59c1ddf7d02f9632f6de"
    ),
    "initialization": {
        "mode": "parent_fresh_optimizer",
        "binding_schema": "a1-canonical-parent-initializer-v1",
        "parent_checkpoint": "required_runtime_input",
        "init_checkpoint": "required_runtime_input",
        "information_contract_migration_receipt": (
            "required_if_checkpoint_bytes_differ"
        ),
        "entity_feature_adapter_version": (
            "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
        ),
        "public_card_exact_resource_residual": True,
        "optimizer_state": "fresh",
    },
    "execution_topology": {
        "schema_version": "a1-canonical-parent-update-topology-v1",
        "launcher": "tools/train.py",
        "distributed_launcher": "torch.distributed.run",
        "name": "b200-8gpu-ddp",
        "required_accelerator_model": "NVIDIA B200",
        "world_size": 8,
        "physical_gpus": list(range(8)),
        "local_batch_size": 64,
        "grad_accum_steps": 1,
        "global_batch_size": 512,
        "go_authorized": True,
        "authorization_authority": (
            "configs/production/training_science_admission.json"
        ),
        "authorization_reason": "canonical_parent_update_authorized",
    },
    "research_scratch_status": "research_only_unresolved_not_selected",
}
PRODUCTION_LEARNER_VALUE_OBJECTIVE_CONTRACT = {
    "objective": "binary_win_bce",
    "value_readout": "scalar",
    "value_categorical_bins": None,
    "hlgauss_sigma_ratio": None,
}
MARITIME_BEHAVIORAL_COMPETENCE_GATE_CONTRACT = {
    "schema_version": "a1-maritime-behavioral-competence-gate-v1",
    "required_streams": ["base", "aux"],
    "stream_admission": {
        "mode": "independent",
        "pooling": False,
        "all_required_streams_must_pass": True,
    },
    "required_axis": "teacher_argmax_action_type",
    "required_action_type": "MARITIME_TRADE",
    "minimum_rows": 64,
    "confidence_interval": {
        "method": "wilson_score",
        "sided": "one",
        "confidence_level": 0.95,
        "z": 1.6448536269514722,
    },
    "minimum_teacher_top1_lower_bound": 0.15,
    "maximum_end_turn_confusion_upper_bound": 0.40,
    "maximum_teacher_top1_parent_regression": 0.05,
    "maximum_end_turn_confusion_parent_regression": 0.05,
    "objective_weighted_diagnostic": {
        "required": True,
        "axis": "objective_weighted_teacher_argmax_action_type",
        "selection_authority": False,
    },
    "playing_strength": {
        "candidate_prefilter_only": True,
        "h2h_final_ranking_authority": True,
    },
}
MARITIME_BEHAVIORAL_COMPETENCE_NEGATIVE_OBSERVATION = {
    "audit_role": "retrospective_negative_audit",
    "promotion_authority": False,
    "required_stream": "base",
    "required_axis": "teacher_argmax_action_type",
    "required_action_type": "MARITIME_TRADE",
    "action_catalog_abi": {
        "version": "catanatron-flat-v1",
        "size": 332,
        "ordered_descriptors_sha256": (
            "sha256:3c4dbb523f45376e6de211a8b69b36ba5f3a9be5981e60a6fab320916ed9f741"
        ),
        "action_types_by_id_sha256": (
            "sha256:184a8199061431a538607f58de53f453ad010046d2eed8727bb054a636169920"
        ),
        "identity_sha256": (
            "sha256:bf8203d092568346c28b171e14e6c88b2aa480030c72d1c74c727167afe2dc51"
        ),
    },
    "rows": 94,
    "teacher_top1_correct": 6,
    "teacher_top1_accuracy": 0.06382978723404255,
    "teacher_top1_wilson_lower_bound": 0.03336295494479746,
    "end_turn_confusion_rows": 74,
    "end_turn_confusion_rate": 0.7872340425531915,
    "end_turn_confusion_wilson_upper_bound": 0.8481230261724284,
    "selection_admitted": False,
    "requires_fresh_candidate_evidence": True,
}
DIAGNOSTIC_POLICY_AUX_FIELDS = frozenset(
    {"policy_aux_active_batch_size", "policy_aux_loss_weight"}
)
PRODUCTION_TARGET_QUALITY_LEARNER_CONTRACT = {
    # Raw search disagreement is not target correctness. Production may only
    # prioritize it again after the reliability audit has qualified a recipe.
    "policy_surprise_weight": 0.0,
    # The archived Stage-C audits show that direct root/completed-Q regression
    # is worse than the raw value baseline and usually below the measured
    # search-noise floor. Keep those scalar/Q objectives diagnostic-only. The
    # authorized parent-update policy target remains search-derived; completed-Q
    # and root-value experiments require their own distinct recipe identity.
    "q_loss_weight": 0.0,
    "value_target_lambda": 1.0,
    "value_root_blend_phases": "",
    "value_root_blend_global_compat": False,
}
PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT = {
    # The teacher policy is the full legal-support MCTS distribution.  Bind
    # the pruning threshold even at zero so a future recipe cannot claim a
    # different target operator than workers actually construct.
    "policy_target_min_visits": 0,
    # A stable root hash selects this audit slice without consuming gameplay or
    # primary-search RNG. Five percent is enough to calibrate target stability
    # without turning the duplicate reference search into the dominant cost.
    "target_reliability_audit_fraction": 0.05,
    "target_reliability_audit_seed": 20260716,
    # Completed-Q and visit evidence cost far less than one percent of the
    # corpus and are required to calibrate or reconstruct teacher reliability.
    "preserve_search_evidence": True,
    # The empirical admission baseline is the evaluator output captured at
    # root expansion, before search backups mutate root_value. Generation and
    # memmap materialization must retain this distinct field and mask.
    "preserve_root_prior_value": True,
}
PRODUCTION_GENERATION_RUNTIME_FIELD_MAP = {
    "track": "track",
    "vps_to_win": "vps_to_win",
    "obs_width": "obs_width",
    "max_decisions": "max_decisions",
    "temperature_clock": "temperature_clock",
    "temperature_decisions": "temperature_decisions",
    "temperature_high": "temperature_high",
    "temperature_low": "temperature_low",
    "late_temperature_decisions": "late_temperature_decisions",
    "late_temperature": "late_temperature",
    "record_automatic_transitions": "record_automatic_transitions",
    "meaningful_public_history": "meaningful_public_history",
    "event_history_limit": "event_history_limit",
    "teacher_entity_feature_adapter_version": (
        "teacher_entity_feature_adapter_version"
    ),
    "learner_entity_feature_adapter_version": (
        "learner_entity_feature_adapter_version"
    ),
    "workers_per_gpu": "workers",
    "shard_size": "shard_size",
    "format": "fmt",
    "device": "device",
    "eval_server": "eval_server",
    "eval_server_max_batch": "eval_server_max_batch",
    "eval_server_max_neural_rows": "eval_server_max_neural_rows",
    "eval_server_max_wait_ms": "eval_server_max_wait_ms",
    "eval_server_timeout_ms": "eval_server_timeout_ms",
    "eval_server_batch_timeout_sec": "eval_server_batch_timeout_sec",
    "eval_server_local_fallback": "eval_server_local_fallback",
    "eval_server_matmul_precision": "eval_server_matmul_precision",
    "eval_server_request_collector": "eval_server_request_collector",
    "eval_server_transport": "eval_server_transport",
    "eval_server_shared_memory_slot_bytes": (
        "eval_server_shared_memory_slot_bytes"
    ),
    "eval_server_event_token_limit": "eval_server_event_token_limit",
    "eval_server_cuda_graph": "eval_server_cuda_graph",
    "eval_server_cuda_graph_batch_buckets": (
        "eval_server_cuda_graph_batch_buckets"
    ),
    "eval_server_cuda_graph_warmup_iterations": (
        "eval_server_cuda_graph_warmup_iterations"
    ),
    "native_mcts_hot_loop": "native_mcts_hot_loop",
    "policy_target_min_visits": "policy_target_min_visits",
    "target_reliability_audit_fraction": "target_reliability_audit_fraction",
    "target_reliability_audit_seed": "target_reliability_audit_seed",
    "preserve_search_evidence": "preserve_search_evidence",
    "preserve_root_prior_value": "preserve_root_prior_value",
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
    boundary_particles = operator["search"].get("boundary_value_particles")
    if type(boundary_particles) is not int or boundary_particles != 1:
        raise ScienceContractError(
            "current production search must bind boundary_value_particles=1; "
            "K>1 remains an uncommissioned operator"
        )
    for key in ("generation", "learner", "evaluation", "promotion"):
        if not isinstance(value[key], dict) or not value[key]:
            raise ScienceContractError(f"current {key} contract is empty")
    promotion = value["promotion"]
    if (
        promotion.get("maritime_behavioral_competence_gate")
        != MARITIME_BEHAVIORAL_COMPETENCE_GATE_CONTRACT
        or promotion.get("maritime_behavioral_competence_negative_observation")
        != MARITIME_BEHAVIORAL_COMPETENCE_NEGATIVE_OBSERVATION
    ):
        raise ScienceContractError(
            "current maritime behavioral competence promotion contract drifted"
        )
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
        "production_selection",
        "value_objective",
        "research_scratch_initialization",
        "architecture_upgrade_flags",
        "architecture_upgrade_module",
        "research_scratch_model_construction",
        "research_scratch_execution_topology",
        "research_scratch_topology",
        "research_scratch_training_recipe",
    } or not isinstance(learner_value["research_scratch_training_recipe"], dict):
        raise ScienceContractError("current learner contract shape drifted")
    if (
        learner_value.get("production_selection")
        != PRODUCTION_LEARNER_SELECTION_CONTRACT
    ):
        raise ScienceContractError("current production learner selection drifted")
    if (
        learner_value.get("value_objective")
        != PRODUCTION_LEARNER_VALUE_OBJECTIVE_CONTRACT
    ):
        raise ScienceContractError("current production learner value objective drifted")
    selected_config = _read_object(CANONICAL_PARENT_UPDATE_CONFIG_PATH)
    if (
        _content_sha256(selected_config).removeprefix("sha256:")
        != PRODUCTION_LEARNER_SELECTION_CONTRACT["config_canonical_sha256"]
    ):
        raise ScienceContractError(
            "selected canonical parent-update config content drifted"
        )
    selected_engine = selected_config.get("engine_settings")
    selected_train_envelope = selected_config.get("train_config")
    selected_train = (
        selected_train_envelope.get("fields")
        if isinstance(selected_train_envelope, dict)
        else None
    )
    selected_initialization = PRODUCTION_LEARNER_SELECTION_CONTRACT["initialization"]
    selected_topology = PRODUCTION_LEARNER_SELECTION_CONTRACT["execution_topology"]
    if (
        not isinstance(selected_engine, dict)
        or not isinstance(selected_train, dict)
        or selected_engine.get("initialization_mode")
        != selected_initialization["mode"]
        or selected_engine.get("entity_feature_adapter_version")
        != selected_initialization["entity_feature_adapter_version"]
        or selected_train.get("resume_optimizer") is not False
        or selected_initialization.get("optimizer_state") != "fresh"
        or selected_train.get("batch_size")
        != selected_topology["local_batch_size"]
        or selected_train.get("grad_accum_steps")
        != selected_topology["grad_accum_steps"]
        or selected_topology["world_size"]
        * selected_topology["local_batch_size"]
        * selected_topology["grad_accum_steps"]
        != selected_topology["global_batch_size"]
    ):
        raise ScienceContractError(
            "selected parent-update initialization/topology contradicts its config"
        )
    admission = _read_object(TRAINING_SCIENCE_ADMISSION_PATH)
    admission_recipes = admission.get("recipes")
    admission_recipe = (
        admission_recipes.get(PRODUCTION_LEARNER_SELECTION_CONTRACT["recipe"])
        if isinstance(admission_recipes, dict)
        else None
    )
    admission_observations = (
        admission_recipe.get("observations")
        if isinstance(admission_recipe, dict)
        and isinstance(admission_recipe.get("observations"), dict)
        else {}
    )
    if (
        admission.get("schema_version")
        != "catan-zero-training-science-admission-v1"
        or not isinstance(admission_recipe, dict)
        or admission_recipe.get("recipe_canonical_sha256")
        != PRODUCTION_LEARNER_SELECTION_CONTRACT["config_canonical_sha256"]
        or not isinstance(admission_recipe.get("authorized"), bool)
        or not isinstance(selected_topology.get("go_authorized"), bool)
        or admission_recipe.get("authorized")
        is not selected_topology.get("go_authorized")
        or admission_recipe.get("reason")
        != selected_topology["authorization_reason"]
        or admission_observations.get("maritime_behavioral_competence_gate")
        != MARITIME_BEHAVIORAL_COMPETENCE_GATE_CONTRACT
        or admission_observations.get(
            "maritime_behavioral_competence_negative_observation"
        )
        != MARITIME_BEHAVIORAL_COMPETENCE_NEGATIVE_OBSERVATION
        or (
            admission_recipe.get("authorized") is True
            and (
                not isinstance(admission_recipe.get("commissioning_evidence"), list)
                or not admission_recipe["commissioning_evidence"]
                or not all(
                    isinstance(item, str) and item
                    for item in admission_recipe["commissioning_evidence"]
                )
            )
        )
    ):
        raise ScienceContractError(
            "selected parent-update admission must remain exact and fail-closed "
            "unless its typed commissioning and go authority agree"
        )
    if (
        learner_value.get("research_scratch_initialization")
        != PRODUCTION_LEARNER_INITIALIZATION_CONTRACT
    ):
        raise ScienceContractError(
            "research-only scratch initialization must remain native from-scratch v6 with "
            "fresh optimizer state"
        )
    if (
        learner_value.get("research_scratch_model_construction")
        != PRODUCTION_LEARNER_MODEL_CONSTRUCTION_CONTRACT
    ):
        raise ScienceContractError("research-only scratch model construction drifted")
    if (
        learner_value.get("research_scratch_execution_topology")
        != PRODUCTION_LEARNER_EXECUTION_TOPOLOGY_CONTRACT
        or learner_value.get("research_scratch_topology")
        != PRODUCTION_LEARNER_EXECUTION_TOPOLOGY_CONTRACT["name"]
    ):
        raise ScienceContractError("research-only scratch execution topology drifted")
    recipe = learner_value["research_scratch_training_recipe"]
    execution = learner_value["research_scratch_execution_topology"]
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
            recipe.get("scalar_value_objective") != "binary_win_bce"
            or learner_value.get("value_objective")
            != PRODUCTION_LEARNER_VALUE_OBJECTIVE_CONTRACT
            or recipe.get("scalar_value_loss_readout") != "deployed_tanh"
            or recipe.get("scalar_value_loss_scale")
            != evaluator_value.get("value_scale")
        )
    ):
        raise ScienceContractError(
            "current scalar learner must optimize stable binary win BCE over "
            "the exact deployed tanh search readout and scale"
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


def learner_production_selection() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"]["production_selection"])


def learner_value_objective() -> dict[str, Any]:
    return copy.deepcopy(_load()["learner"]["value_objective"])


def maritime_behavioral_competence_gate() -> dict[str, Any]:
    """Return the diagnostic prefilter applied before matched H2H ranking."""

    return copy.deepcopy(
        _load()["promotion"]["maritime_behavioral_competence_gate"]
    )


def maritime_behavioral_competence_negative_observation() -> dict[str, Any]:
    """Return the retrospective observation that requires a fresh candidate."""

    return copy.deepcopy(
        _load()["promotion"][
            "maritime_behavioral_competence_negative_observation"
        ]
    )


def require_selected_parent_update(config_path: str | Path) -> Path:
    """Authenticate the one selected production learner route.

    This does not authorize a corpus or a training execution.  The one-dose
    executor still has to prove the production composite, direct architecture
    upgrade, DDP canary, and fresh optimizer transaction independently.
    """

    selection = learner_production_selection()
    if selection.get("mode") != "canonical_parent_update":
        raise ScienceContractError(
            "current production learner does not select a parent update"
        )
    try:
        supplied = Path(config_path).expanduser().resolve(strict=True)
        expected = (REPO_ROOT / str(selection["config_path"])).resolve(strict=True)
    except OSError as error:
        raise ScienceContractError(
            f"cannot resolve selected parent-update config: {error}"
        ) from error
    if supplied != expected or supplied.is_symlink():
        raise ScienceContractError(
            "production learner must use the selected canonical parent-update config"
        )
    actual = _content_sha256(_read_object(supplied)).removeprefix("sha256:")
    if actual != selection["config_canonical_sha256"]:
        raise ScienceContractError(
            "selected canonical parent-update config content drifted"
        )
    return supplied


def selected_parent_update_commissioning() -> dict[str, Any]:
    """Return the typed live commissioning state for the selected treatment."""

    selection = learner_production_selection()
    admission = _read_object(TRAINING_SCIENCE_ADMISSION_PATH)
    recipes = admission.get("recipes")
    record = recipes.get(selection["recipe"]) if isinstance(recipes, dict) else None
    if (
        admission.get("schema_version")
        != "catan-zero-training-science-admission-v1"
        or not isinstance(record, dict)
        or record.get("recipe_canonical_sha256")
        != selection["config_canonical_sha256"]
        or not isinstance(record.get("authorized"), bool)
        or not isinstance(selection["execution_topology"].get("go_authorized"), bool)
    ):
        raise ScienceContractError("selected parent-update commissioning is malformed")
    value = {
        "schema_version": TRAINING_SCIENCE_COMMISSIONING_SCHEMA,
        "recipe": selection["recipe"],
        "config_canonical_sha256": selection["config_canonical_sha256"],
        "admission": {
            "path": str(TRAINING_SCIENCE_ADMISSION_PATH.resolve(strict=True)),
            "sha256": _file_sha256(TRAINING_SCIENCE_ADMISSION_PATH),
        },
        "authorized": bool(record["authorized"]),
        "go_authorized": bool(selection["execution_topology"]["go_authorized"]),
        "reason": record.get("reason"),
        "commissioning_evidence": copy.deepcopy(
            record.get("commissioning_evidence", [])
        ),
    }
    value["commissioning_sha256"] = _content_sha256(value)
    return value


def require_selected_parent_update_go_authorized() -> dict[str, Any]:
    """Refuse GPU execution until the selected treatment is commissioned."""

    value = selected_parent_update_commissioning()
    if (
        value["authorized"] is not True
        or value["go_authorized"] is not True
        or not isinstance(value["commissioning_evidence"], list)
        or not value["commissioning_evidence"]
    ):
        raise ScienceContractError(
            "selected parent-update training science is not authorized for --go"
        )
    return value


def learner_training_recipe() -> dict[str, Any]:
    """Return the unresolved research-only scratch recipe."""

    return copy.deepcopy(_load()["learner"]["research_scratch_training_recipe"])


def learner_initialization() -> dict[str, Any]:
    """Return the unresolved research-only scratch initializer."""

    return copy.deepcopy(_load()["learner"]["research_scratch_initialization"])


def learner_model_construction() -> dict[str, Any]:
    """Return the unresolved research-only scratch model construction."""

    return copy.deepcopy(
        _load()["learner"]["research_scratch_model_construction"]
    )


def learner_execution_topology() -> dict[str, Any]:
    """Return the unresolved research-only scratch execution topology."""

    return copy.deepcopy(
        _load()["learner"]["research_scratch_execution_topology"]
    )


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
    learner_value_objective_value: Mapping[str, Any] | None = None,
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
    if learner_value_objective_value is not None:
        expected_objective = learner_value_objective()
        actual_objective = dict(learner_value_objective_value)
        if actual_objective != expected_objective:
            differing = sorted(
                key
                for key in set(actual_objective) | set(expected_objective)
                if actual_objective.get(key) != expected_objective.get(key)
            )
            raise ScienceContractError(
                "coherent-public learner value objective differs from current "
                f"science contract: {differing}"
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
    catalog = _read_object(PRODUCTION_RECIPE_CATALOG_PATH)
    recipes = catalog.get("recipes", {}).get("generate", [])
    expected_relative_path = str(GENERATOR_CONFIG_PATH.relative_to(REPO_ROOT))
    catalog_records = [
        item
        for item in recipes
        if isinstance(item, dict) and item.get("path") == expected_relative_path
    ]
    if len(catalog_records) != 1:
        raise ScienceContractError(
            "current coherent generator is not uniquely authenticated by the "
            "production recipe catalog"
        )
    canonical_sha256 = _content_sha256(generator).removeprefix("sha256:")
    if (
        generator.get("pipeline") != "generate"
        or generator.get("schema_version") != 22
        or catalog_records[0].get("canonical_sha256") != canonical_sha256
    ):
        raise ScienceContractError(
            "current coherent generator catalog authentication drifted"
        )
    fields = generator.get("fields")
    if not isinstance(fields, dict):
        raise ScienceContractError("coherent generator config fields are missing")
    generation = contract["generation"]
    runtime_drift = {
        science_field: {
            "config_field": config_field,
            "science": generation.get(science_field),
            "catalog": fields.get(config_field),
        }
        for science_field, config_field in PRODUCTION_GENERATION_RUNTIME_FIELD_MAP.items()
        if generation.get(science_field) != fields.get(config_field)
    }
    if runtime_drift:
        raise ScienceContractError(
            "current science generation runtime differs from authenticated "
            f"catalog schema22 recipe: {runtime_drift}"
        )
    search_value = contract["operator"]["search"]
    expected_generator = {
        **PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT,
        "exact_budget_sh": search_value.get("exact_budget_sh"),
        "exact_budget_sh_min_n": search_value.get("exact_budget_sh_min_n"),
        "boundary_value_particles": search_value["boundary_value_particles"],
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
        "--boundary-value-particles": expected_generator[
            "boundary_value_particles"
        ],
        "--learner-entity-feature-adapter-version": (
            expected_generator["learner_entity_feature_adapter_version"]
        ),
        "--teacher-entity-feature-adapter-version": generation[
            "teacher_entity_feature_adapter_version"
        ],
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
    category_variant = {
        "--workers",
        "--eval-server",
    }
    overconstrained = sorted(category_variant & critical)
    if overconstrained:
        raise ScienceContractError(
            "shared coherent generator guard overconstrains category-variant "
            f"runtime flags: {overconstrained}"
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
        "rng_stream_separation": bool(
            search_value.get("rng_stream_separation", False)
        ),
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
        "boundary_value_particles": search_value["boundary_value_particles"],
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
