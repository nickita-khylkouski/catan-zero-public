from __future__ import annotations

import copy
import json

import pytest

from tools import a1_current_science_contract as current_science


def test_current_production_learner_binds_full_value_and_exact_dose() -> None:
    recipe = current_science.learner_training_recipe()

    for key, expected in current_science.PRODUCTION_LEARNER_SIGNAL_CONTRACT.items():
        assert recipe[key] == expected
    for (
        key,
        expected,
    ) in current_science.PRODUCTION_TARGET_QUALITY_LEARNER_CONTRACT.items():
        assert recipe[key] == expected
    assert not current_science.DIAGNOSTIC_POLICY_AUX_FIELDS & set(recipe)
    assert recipe["phase_weights"] == "PLAY_TURN=4.0"


def test_current_target_quality_generation_is_bound_to_config_and_guard() -> None:
    generation = current_science.generation()
    for (
        key,
        expected,
    ) in current_science.PRODUCTION_TARGET_QUALITY_GENERATION_CONTRACT.items():
        assert generation[key] == expected

    generator = json.loads(
        current_science.GENERATOR_CONFIG_PATH.read_text(encoding="utf-8")
    )["fields"]
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
        ("target_reliability_confidence_weighting", False),
        ("target_reliability_confidence_floor", 0.0),
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
        match="learner target-quality contract drifted",
    ):
        current_science.load()
