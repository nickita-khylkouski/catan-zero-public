from __future__ import annotations

import copy

import pytest

from tools import a1_value_trunk_scale_adjudication as adjudication


def test_contract_exposes_authority_conflict_without_changing_production() -> None:
    contract = adjudication.build_contract()

    assert contract["diagnostic_only"] is True
    assert contract["promotion_eligible"] is False
    assert contract["production_contract_mutated"] is False
    assert contract["authority_conflict"] == {
        "current_production_scale": 1.0,
        "stage_c_diagnostic_scale": 0.1,
        "status": "unresolved_requires_broad_adjudication",
    }
    control = contract["arms"][adjudication.CONTROL_ARM]["recipe"]
    treatment = contract["arms"][adjudication.TREATMENT_ARM]["recipe"]
    differing = {
        key for key in control if control.get(key) != treatment.get(key)
    }
    assert differing == {"value_trunk_grad_scale"}
    assert control["value_trunk_grad_scale"] == pytest.approx(1.0)
    assert treatment["value_trunk_grad_scale"] == pytest.approx(0.1)
    assert contract["narrow_probe_evidence"]["decision_authority"] is False
    assert len(contract["narrow_probe_evidence"]["matched_16_step_seeds"]) == 2
    assert contract["adjudication_result_boundary"] == {
        "implementation": "plan_only_no_adjudicator",
        "maximum_result": "none_until_artifact_replay_is_implemented",
        "automatic_production_flip": False,
        "automatic_promotion": False,
    }
    adjudication.verify_contract(contract)


def test_contract_refuses_silent_production_or_stage_c_authority_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = adjudication.current_science.learner_training_recipe()
    recipe["value_trunk_grad_scale"] = 0.1
    monkeypatch.setattr(
        adjudication.current_science,
        "learner_training_recipe",
        lambda: recipe,
    )

    with pytest.raises(adjudication.ValueTrunkScaleError, match="authority conflict"):
        adjudication.build_contract()


def test_self_asserted_claims_have_no_adjudication_entrypoint() -> None:
    assert not hasattr(adjudication, "adjudicate")
    with pytest.raises(SystemExit):
        adjudication.build_parser().parse_args(
            ["adjudicate", "--receipt", "forged.json"]
        )


def test_forged_boolean_and_count_claims_cannot_be_added_to_contract() -> None:
    contract = adjudication.build_contract()
    forged = copy.deepcopy(contract)
    forged["self_asserted_receipt"] = {
        "authenticated_metric_checkpoint_binding": True,
        "target_manifest_verified": True,
        "optimizer_exclusion_receipt_verified": True,
        "validation_split_receipt_verified": True,
        "selected_root_count": 65_536,
        "complete_pairs": 600,
        "full_promotion_gate_passed": True,
    }

    with pytest.raises(adjudication.ValueTrunkScaleError, match="contract drift"):
        adjudication.verify_contract(forged)


def test_plan_requires_artifact_replay_and_derived_evidence() -> None:
    contract = adjudication.build_contract()
    learner = contract["required_learner_evidence"]
    gameplay = contract["required_gameplay_evidence"]

    assert learner["required_artifact_reference_fields"] == [
        "path",
        "file_sha256",
        "schema_version",
        "semantic_digest",
    ]
    assert "selected_root_count" in learner["derived_not_asserted"]
    assert "checkpoint_metric_binding" in learner["derived_not_asserted"]
    assert "tools.a1_stage_c_final_replication.verify_final_authority" in (
        learner["validators_to_replay"]
    )
    assert "sprt_decision" in gameplay["derived_not_asserted"]
    assert "tools.a1_promotion_transaction._verify_promotion_evidence" in (
        gameplay["validators_to_replay"]
    )


def test_contract_digest_tamper_is_rejected() -> None:
    contract = adjudication.build_contract()
    tampered = copy.deepcopy(contract)
    tampered["required_gameplay_evidence"]["minimum_complete_pairs"] = 1

    with pytest.raises(adjudication.ValueTrunkScaleError, match="contract drift"):
        adjudication.verify_contract(tampered)
