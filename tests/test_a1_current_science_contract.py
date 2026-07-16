from __future__ import annotations

import copy
import json

import pytest

from tools import a1_current_science_contract as current_science


def test_current_production_learner_binds_full_value_and_exact_dose() -> None:
    recipe = current_science.learner_training_recipe()

    for key, expected in current_science.PRODUCTION_LEARNER_SIGNAL_CONTRACT.items():
        assert recipe[key] == expected
    assert not current_science.DIAGNOSTIC_POLICY_AUX_FIELDS & set(recipe)


@pytest.mark.parametrize(
    ("field", "diagnostic_value"),
    (
        ("value_lr_mult", 0.3),
        ("value_trunk_grad_scale", 0.1),
        ("grad_accum_steps", 4),
        ("max_steps", 1024),
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
