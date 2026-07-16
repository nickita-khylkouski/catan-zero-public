from __future__ import annotations

import copy
import json

import pytest

from tools import a1_current_science_contract as current_science


def test_current_production_learner_binds_full_value_and_exact_dose() -> None:
    recipe = current_science.learner_training_recipe()

    assert (
        current_science.learner_initialization()
        == current_science.PRODUCTION_LEARNER_INITIALIZATION_CONTRACT
    )
    assert (
        current_science.learner_model_construction()
        == current_science.PRODUCTION_LEARNER_MODEL_CONSTRUCTION_CONTRACT
    )
    model = current_science.learner_model_construction()
    assert model["graph_tokens"] is None
    assert model["hidden_size"] == 640
    assert model["min_parameter_count"] == 41_700_000
    assert model["max_parameter_count"] == 42_000_000
    assert model["public_card_count_residual_bias"] is False
    assert model["public_rule_state_features"] is True
    assert model["value_tower_split_layers"] == 1
    assert model["action_target_gather"] is True
    assert model["legal_action_value_set_statistics"] is True
    assert model["actor_public_rule_state"].startswith("dev_used_")
    assert recipe["value_trunk_grad_scale"] == 0.25
    assert recipe["post_policy_dose_value_trunk_grad_scale"] == 0.0
    execution = current_science.learner_execution_topology()
    assert execution == current_science.PRODUCTION_LEARNER_EXECUTION_TOPOLOGY_CONTRACT
    assert execution["go_authorized"] is False
    assert execution["optimization_schedule_status"] == "unresolved"
    assert (
        execution["reviewed_optimizer_schedule_role"]
        == "candidate_horizon_pending_optimizer_and_value_routing_evidence_v1"
    )
    assert (
        execution["world_size"]
        * execution["local_batch_size"]
        * execution["grad_accum_steps"]
        == recipe["global_batch_size"]
    )
    for key, expected in current_science.PRODUCTION_LEARNER_SIGNAL_CONTRACT.items():
        assert recipe[key] == expected
    for (
        key,
        expected,
    ) in current_science.PRODUCTION_TARGET_QUALITY_LEARNER_CONTRACT.items():
        assert recipe[key] == expected
    assert not current_science.DIAGNOSTIC_POLICY_AUX_FIELDS & set(recipe)
    assert recipe["phase_weights"] == "PLAY_TURN=4.0"
    assert recipe["base_sampler"] == "coverage_importance_v1"
    assert recipe["minimum_policy_effective_rows_per_global_batch"] == 32.0
    assert recipe["moe_balance_loss_weight"] == 0.0
    assert recipe["soft_target_source"] == "policy"
    assert recipe["soft_target_weight"] == 1.0
    assert recipe["soft_target_temperature"] == 1.0
    assert recipe["soft_target_min_legal_coverage"] == 1.0
    assert recipe["train_diagnostics_every_batches"] == 16
    assert recipe["objective_gradient_interference_every_batches"] == 16
    assert recipe["minimum_feature_learning_signal_observations"] == 2
    assert set(
        recipe["require_feature_learning_signal_modules"].split(",")
    ) == {
        "event_encoder",
        "final_vp_head",
        "target_gather_proj",
        "legal_action_value_residual_proj",
        "legal_action_value_static_proj",
        "legal_action_value_max_proj",
        "legal_action_value_count_proj",
        "legal_action_value_static_max_proj",
        "meaningful_history_residual_gate",
        "meaningful_history_ordered_gate",
        "meaningful_history_sequence",
        "meaningful_history_target_proj",
        "public_card_count_residual",
        "public_rule_state_residual",
        "static_action_residual_proj",
        "value_blocks",
        "value_head",
        "value_state_norm",
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("mode", "checkpoint"),
        (
            "entity_feature_adapter_version",
            "rust_entity_adapter_v2_land_topology_ports_maritime",
        ),
        ("checkpoint", "/tmp/legacy.pt"),
        ("optimizer_state", "resume"),
    ),
)
def test_current_contract_rejects_non_scratch_v5_initialization(
    tmp_path, monkeypatch, field: str, bad_value
) -> None:
    contract = copy.deepcopy(current_science.load())
    contract["learner"]["initialization"][field] = bad_value
    path = tmp_path / "science.contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(current_science, "CONTRACT_PATH", path)

    with pytest.raises(
        current_science.ScienceContractError,
        match="native from-scratch v5",
    ):
        current_science.load()


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    (
        ("model_construction", "static_action_residual", False),
        ("model_construction", "action_target_gather", False),
        ("model_construction", "max_parameter_count", 40_000_000),
        ("model_construction", "entity_feature_adapter_version", "legacy"),
        ("execution_topology", "world_size", 4),
        ("execution_topology", "local_batch_size", 1024),
    ),
)
def test_current_contract_rejects_scratch_construction_or_topology_drift(
    tmp_path,
    monkeypatch,
    section: str,
    field: str,
    bad_value,
) -> None:
    contract = copy.deepcopy(current_science.load())
    contract["learner"][section][field] = bad_value
    path = tmp_path / "science.contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(current_science, "CONTRACT_PATH", path)

    with pytest.raises(current_science.ScienceContractError, match="scratch"):
        current_science.load()


def test_current_target_quality_generation_is_bound_to_config_and_guard() -> None:
    generation = current_science.generation()
    learner = current_science.learner()
    assert current_science.search()["boundary_value_particles"] == 1
    assert (
        current_science.fleet_evaluation_science_config()[
            "boundary_value_particles"
        ]
        == 1
    )
    assert generation["teacher_entity_feature_adapter_version"] == (
        current_science.CURRENT_TEACHER_ENTITY_ADAPTER
    )
    assert generation["learner_entity_feature_adapter_version"] == (
        current_science.CURRENT_LEARNER_ENTITY_ADAPTER
    )
    assert learner["architecture_upgrade_flags"] == (
        current_science.CURRENT_ARCHITECTURE_UPGRADE_FLAGS
    )
    assert learner["architecture_upgrade_module"] == (
        current_science.CURRENT_ARCHITECTURE_UPGRADE_MODULE
    )
    for (
        key,
        expected,
    ) in current_science.PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT.items():
        assert generation[key] == expected

    generator = json.loads(
        current_science.GENERATOR_CONFIG_PATH.read_text(encoding="utf-8")
    )["fields"]
    assert generator["boundary_value_particles"] == 1
    assert generator["exact_budget_sh"] is False
    assert generator["exact_budget_sh_min_n"] == 0
    for (
        key,
        expected,
    ) in current_science.PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT.items():
        assert generator[key] == expected

    guard = json.loads(
        current_science.GENERATOR_GUARD_PATH.read_text(encoding="utf-8")
    )
    lint = next(item["args"] for item in guard["guards"] if item["name"] == "cli_flag_lint")
    assert lint["expected_values"]["--exact-budget-sh"] is False
    assert lint["expected_values"]["--exact-budget-sh-min-n"] == 0
    assert lint["expected_values"]["--target-reliability-audit-fraction"] == 0.05
    assert lint["expected_values"]["--target-reliability-audit-seed"] == 20260716
    assert lint["expected_values"]["--boundary-value-particles"] == 1
    assert lint["expected_values"][
        "--learner-entity-feature-adapter-version"
    ] == current_science.CURRENT_LEARNER_ENTITY_ADAPTER


@pytest.mark.parametrize("bad_value", (None, 0, 2))
def test_current_contract_rejects_uncommissioned_boundary_particles(
    tmp_path, monkeypatch, bad_value
) -> None:
    contract = copy.deepcopy(current_science.load())
    contract["operator"]["search"]["boundary_value_particles"] = bad_value
    path = tmp_path / "science.contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(current_science, "CONTRACT_PATH", path)

    with pytest.raises(
        current_science.ScienceContractError,
        match="boundary_value_particles=1",
    ):
        current_science.load()


@pytest.mark.parametrize(
    ("field", "diagnostic_value"),
    (
        ("value_lr_mult", 0.3),
        ("value_trunk_grad_scale", 0.1),
        ("grad_accum_steps", 4),
        ("max_steps", 1024),
        ("phase_weights", ""),
    ),
)
def test_current_contract_rejects_diagnostic_training_settings(
    tmp_path, monkeypatch, field: str, diagnostic_value
) -> None:
    contract = copy.deepcopy(current_science.load())
    contract["learner"]["training_recipe"][field] = diagnostic_value
    path = tmp_path / "science.contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(current_science, "CONTRACT_PATH", path)

    with pytest.raises(
        current_science.ScienceContractError,
        match="diagnostic/approximate training setting",
    ):
        current_science.load()


def test_current_contract_rejects_diagnostic_policy_aux_leak(
    tmp_path, monkeypatch
) -> None:
    contract = copy.deepcopy(current_science.load())
    contract["learner"]["training_recipe"]["policy_aux_active_batch_size"] = 128
    contract["learner"]["training_recipe"]["policy_aux_loss_weight"] = 0.25
    path = tmp_path / "science.contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(current_science, "CONTRACT_PATH", path)

    with pytest.raises(
        current_science.ScienceContractError,
        match="diagnostic active-policy AUX fields",
    ):
        current_science.load()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("per_game_policy_surprise_weighting", True),
        ("target_reliability_confidence_weighting", True),
    ),
)
def test_current_contract_rejects_unsafe_target_quality_learner_drift(
    tmp_path, monkeypatch, field: str, bad_value
) -> None:
    contract = copy.deepcopy(current_science.load())
    contract["learner"]["training_recipe"][field] = bad_value
    path = tmp_path / "science.contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(current_science, "CONTRACT_PATH", path)

    with pytest.raises(
        current_science.ScienceContractError,
        match="weighting|surprise",
    ):
        current_science.load()
