from __future__ import annotations

import pytest

from tools import a1_learner_dose_contract as dose


def _report(selected: dose.LearnerDose) -> dict[str, int]:
    return {
        "max_steps": selected.optimizer_steps,
        "steps_completed": selected.optimizer_steps,
        "world_size": selected.world_size,
        "batch_size": selected.per_rank_batch_size,
        "grad_accum_steps": selected.grad_accum_steps,
        "effective_global_batch_size": selected.effective_global_batch_size,
        "training_row_draws": selected.global_samples,
        "base_training_row_draws": selected.global_samples,
        "policy_aux_training_row_draws": 0,
        "total_training_row_draws": selected.global_samples,
    }


def test_pareto_and_historical_doses_are_distinct_typed_identities() -> None:
    assert dose.PARETO_SELECTED_DOSE.global_samples == 524_288
    assert dose.PARETO_SELECTED_DOSE.optimizer_steps == 128
    assert dose.HISTORICAL_FULL_DOSE.global_samples == 4_194_304
    assert dose.HISTORICAL_FULL_DOSE.optimizer_steps == 1024
    assert dose.PARETO_SELECTED_DOSE.payload()["schema_version"] == dose.SCHEMA_VERSION
    assert (
        dose.PARETO_SELECTED_DOSE.payload()["dose_id"]
        != dose.HISTORICAL_FULL_DOSE.payload()["dose_id"]
    )


def test_payload_rejects_relabelling_full_dose_as_selected_short() -> None:
    forged = dose.PARETO_SELECTED_DOSE.payload()
    forged["optimizer_steps"] = 1024
    forged["global_samples"] = 4_194_304

    with pytest.raises(dose.LearnerDoseError, match="learner dose drifted"):
        dose.assert_payload(forged, dose.PARETO_SELECTED_DOSE)


def test_report_contract_binds_actual_draws_not_only_planned_steps() -> None:
    report = _report(dose.PARETO_SELECTED_DOSE)
    dose.assert_report(report, dose.PARETO_SELECTED_DOSE)

    report["training_row_draws"] -= 4096
    with pytest.raises(dose.LearnerDoseError, match="training_row_draws"):
        dose.assert_report(report, dose.PARETO_SELECTED_DOSE)


def test_legacy_projection_is_verifiable_but_not_a_typed_new_contract() -> None:
    legacy = {
        "optimizer_steps": 1024,
        "world_size": 8,
        "per_rank_batch_size": 512,
        "global_samples": 4_194_304,
        "policy_aux_active_batch_size": 0,
    }
    dose.assert_legacy_payload(legacy, dose.HISTORICAL_FULL_DOSE)
    with pytest.raises(dose.LearnerDoseError, match="fields drifted"):
        dose.assert_payload(legacy, dose.HISTORICAL_FULL_DOSE)
