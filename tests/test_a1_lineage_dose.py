from __future__ import annotations

import pytest

from tools import a1_lineage_dose as lineage

PRODUCER = "sha256:" + "1" * 64
PARENT = "sha256:" + "2" * 64
RECEIPT = "sha256:" + "3" * 64


def test_direct_dose_requires_init_to_equal_declared_producer() -> None:
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PRODUCER,
        current_sampled_rows=100,
        current_optimizer_steps=5,
    )
    assert dose["cumulative_sampled_rows"] == 100
    assert dose["cumulative_optimizer_steps"] == 5
    assert dose["optimizer_state_continuity"] == "fresh_optimizer_per_dose"
    with pytest.raises(lineage.LineageDoseError, match="untyped checkpoint chaining"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=PARENT,
            current_sampled_rows=100,
            current_optimizer_steps=5,
        )


def test_typed_curriculum_adds_parent_and_current_dose() -> None:
    parent = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PRODUCER,
        current_sampled_rows=56_000,
        current_optimizer_steps=14,
    )
    child = lineage.curriculum_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PARENT,
        parent_receipt_sha256=RECEIPT,
        parent_lineage_dose=parent,
        current_sampled_rows=140_000,
        current_optimizer_steps=35,
    )
    assert child["mode"] == "typed_curriculum"
    assert child["prior_sampled_rows"] == 56_000
    assert child["cumulative_sampled_rows"] == 196_000
    assert child["cumulative_optimizer_steps"] == 49


def test_validator_rejects_forged_cumulative_arithmetic() -> None:
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PRODUCER,
        current_sampled_rows=100,
        current_optimizer_steps=5,
    )
    dose["cumulative_sampled_rows"] = 101
    with pytest.raises(lineage.LineageDoseError, match="arithmetic drift"):
        lineage.validate_lineage_dose(dose)


def test_direct_dose_binds_exact_objective_exposure() -> None:
    exposure = {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 4_194_304,
        "policy_base_active_sampled_rows": 515_337,
        "policy_aux_active_sampled_rows": 1_048_576,
        "policy_active_sampled_rows": 1_563_913,
        "value_active_sampled_rows": 4_194_304,
        "anchor_eligible_sampled_rows": 0,
    }
    dose = lineage.direct_lineage_dose(
        declared_producer_sha256=PRODUCER,
        init_checkpoint_sha256=PRODUCER,
        current_sampled_rows=4_194_304,
        current_optimizer_steps=1_024,
        objective_exposure=exposure,
    )

    assert dose["objective_exposure"] == exposure


def test_exact_objective_exposure_rejects_policy_arithmetic_drift() -> None:
    exposure = {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 100,
        "policy_base_active_sampled_rows": 20,
        "policy_aux_active_sampled_rows": 10,
        "policy_active_sampled_rows": 29,
        "value_active_sampled_rows": 100,
        "anchor_eligible_sampled_rows": 0,
    }
    with pytest.raises(lineage.LineageDoseError, match="exposure arithmetic drift"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=PRODUCER,
            current_sampled_rows=100,
            current_optimizer_steps=1,
            objective_exposure=exposure,
        )


def test_exact_objective_exposure_must_match_current_dose() -> None:
    exposure = {
        "measurement_status": "bound_exactly",
        "measurement_scope": "current_dose",
        "base_sampled_rows": 99,
        "policy_base_active_sampled_rows": 20,
        "policy_aux_active_sampled_rows": 10,
        "policy_active_sampled_rows": 30,
        "value_active_sampled_rows": 99,
        "anchor_eligible_sampled_rows": 0,
    }
    with pytest.raises(
        lineage.LineageDoseError, match="does not match current sampled rows"
    ):
        lineage.direct_lineage_dose(
            declared_producer_sha256=PRODUCER,
            init_checkpoint_sha256=PRODUCER,
            current_sampled_rows=100,
            current_optimizer_steps=1,
            objective_exposure=exposure,
        )
