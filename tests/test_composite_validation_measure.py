from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    _combine_policy_aux_validation_metrics,
    _canonical_json_sha256,
    _empty_policy_target_behavior_parts,
    _empty_policy_target_metric_parts,
    _IndexedValidationWeights,
    _objective_measure_validation_aggregate,
    _policy_aux_validation_objective_weights,
    _reduce_common_uniform_clean_outcome_scalar_mse,
    evaluate_composite_validation_measure,
    objective_matched_validation_component_metrics,
    objective_matched_validation_evaluation_identity,
    objective_matched_validation_metrics,
)


def test_policy_aux_validation_applies_sampling_and_loss_measures() -> None:
    # q conditions on policy-active rows; w equalizes the two games' objective
    # mass despite the second game contributing three active roots.
    q = np.full(4, 0.25, dtype=np.float64)
    w = np.asarray([1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    effective = _policy_aux_validation_objective_weights(q, w)

    assert effective == pytest.approx(q * w)
    assert effective[0] == pytest.approx(effective[1:].sum())


def test_policy_aux_validation_reconstructs_the_training_objective() -> None:
    combined = _combine_policy_aux_validation_metrics(
        {
            "loss": 2.0,
            "raw_batch_mean_loss": 2.5,
            "policy_loss": 0.5,
            "value_loss": 1.5,
            "policy_kl_anchor_loss": 0.1,
            "loss_denominators": {"policy_kl_anchor_loss": 10.0},
        },
        {
            "samples": 40,
            "policy_loss": 1.2,
            "policy_kl_anchor_loss": 0.5,
            "loss_denominators": {
                "policy_loss": 3.0,
                "policy_kl_anchor_loss": 6.0,
            },
        },
        policy_loss_weight=1.0,
        policy_aux_loss_weight=0.25,
        policy_kl_anchor_weight=0.4,
    )

    assert combined["policy_base_loss"] == 0.5
    assert combined["policy_aux_loss"] == 1.2
    assert combined["policy_loss"] == pytest.approx(0.8)
    assert combined["policy_kl_anchor_loss"] == pytest.approx(0.225)
    assert combined["policy_kl_anchor_base_loss"] == 0.1
    assert combined["policy_kl_anchor_aux_loss"] == 0.5
    assert combined["policy_kl_anchor_controller_measure"] == pytest.approx(
        1.75 / 11.5
    )
    assert combined["loss"] == pytest.approx(2.35)
    assert combined["component_reconstructed_loss"] == pytest.approx(2.35)
    assert combined["base_raw_batch_mean_loss"] == 2.5
    assert combined["policy_aux_validation_effective_weight_sum"] == 3.0


def test_policy_aux_validation_preserves_base_common_uniform_value_metric() -> None:
    common = {
        "schema_version": "common-uniform-clean-outcome-scalar-mse-v1",
        "measure": "uniform_clean_terminal_outcome_rows",
        "target": "actor_perspective_terminal_outcome_pm1",
        "prediction_readout": "raw",
        "prediction_scale": 1.0,
        "training_value_sample_weights_applied": False,
        "outcome_confidence_applied": False,
        "truncated_rows_included": False,
        "root_value_blend_applied": False,
        "available": True,
        "eligible_rows": 10,
        "squared_error_sum": 4.0,
        "mse": 0.4,
    }
    combined = _combine_policy_aux_validation_metrics(
        {
            "loss": 1.0,
            "policy_loss": 0.5,
            "common_uniform_clean_outcome_scalar_mse": common,
        },
        {
            "samples": 2,
            "policy_loss": 0.25,
            "common_uniform_clean_outcome_scalar_mse": {
                **common,
                "eligible_rows": 2,
                "squared_error_sum": 20.0,
                "mse": 10.0,
            },
        },
        policy_loss_weight=1.0,
        policy_aux_loss_weight=0.5,
    )

    assert combined["common_uniform_clean_outcome_scalar_mse"] == common


def test_indexed_validation_weights_avoid_a_corpus_sized_allocation() -> None:
    weights = _IndexedValidationWeights(
        np.asarray([100, 5, 42], dtype=np.int64),
        np.asarray([0.1, 0.2, 0.7], dtype=np.float64),
    )

    assert weights[np.asarray([42, 5], dtype=np.int64)].tolist() == [0.7, 0.2]
    with pytest.raises(KeyError, match="outside"):
        weights[np.asarray([6], dtype=np.int64)]


class _Composite:
    component_ids = ("n128", "replay")
    component_game_sampling_ratios = (0.75, 0.25)
    corpora = (object(), object())

    def __init__(self) -> None:
        # n128: one one-row game and one three-row game. replay: one two-row
        # game. This deliberately makes raw-row and game-uniform measures differ.
        self._game_seed = np.asarray([11, 12, 12, 12, 21, 21], dtype=np.int64)
        self.component_offsets = np.asarray([0, 4, 6], dtype=np.int64)
        self.meta = {
            "schema": "memmap_composite_v2",
            "descriptor_fingerprint": "sha256:" + "d" * 64,
            "payload_inventory_sha256": "sha256:" + "e" * 64,
            "source_authority_semantic_sha256": "sha256:" + "a" * 64,
        }

    def __getitem__(self, key: str):
        if key != "game_seed":
            raise KeyError(key)
        return self._game_seed

    def component_indices_for_rows(self, rows) -> np.ndarray:
        return (
            np.searchsorted(self.component_offsets, np.asarray(rows), side="right") - 1
        )


def _evaluation_identity() -> dict[str, object]:
    return objective_matched_validation_evaluation_identity(
        model_state_sha256="sha256:" + "1" * 64,
        runtime_binding={"schema_version": "test-runtime-v1", "tree": "sha256:" + "2" * 64},
        epoch=1,
        optimizer_step=32,
    )


def test_objective_matched_validation_is_component_then_game_then_row() -> None:
    data = _Composite()
    per_row_loss = np.asarray([1.0, 3.0, 3.0, 3.0, 10.0, 10.0])
    calls: list[tuple[int, ...]] = []

    def evaluate(indices: np.ndarray) -> dict:
        calls.append(tuple(map(int, indices)))
        value = float(per_row_loss[indices].mean())
        return {
            "loss": value,
            "policy_loss": value + 1.0,
            "accuracy": value / 10.0,
            "samples": int(len(indices)),
        }

    report = evaluate_composite_validation_measure(
        data,
        np.arange(6, dtype=np.int64),
        evaluate,
        evaluation_identity=_evaluation_identity(),
    )

    # n128 game-uniform=(1+3)/2=2, replay=10; authenticated aggregate
    # .75*2 + .25*10 = 4. A raw-row concat would be 5 and is intentionally not
    # the reported objective-matched value.
    assert report["metrics"]["loss"] == 4.0
    assert report["metrics"]["policy_loss"] == 5.0
    assert report["components"]["n128"]["metrics"]["loss"] == 2.0
    assert report["components"]["n128"]["min_rows_per_game"] == 1
    assert report["components"]["n128"]["max_rows_per_game"] == 3
    assert report["components"]["replay"]["metrics"]["loss"] == 10.0
    assert report["component_sampling_ratios"] == {"n128": 0.75, "replay": 0.25}
    assert report["schema_version"] == "composite-validation-measure-v3"
    assert report["validation_key"] == "validation_objective_matched"
    assert report["provenance"]["schema_version"] == (
        "composite-validation-provenance-v3"
    )
    assert report["provenance"]["validation_key"] == (
        "validation_objective_matched"
    )
    assert report["provenance"]["objective_match_sha256"].startswith("sha256:")
    assert report["provenance"]["descriptor_fingerprint"] == "sha256:" + "d" * 64
    assert report["provenance"]["payload_inventory_sha256"] == "sha256:" + "e" * 64
    assert report["provenance"]["source_authority_semantic_sha256"] == (
        "sha256:" + "a" * 64
    )
    assert report["provenance"]["validation_game_seed_set_sha256"].startswith(
        "sha256:"
    )
    assert report["provenance"]["component_coverage_sha256"].startswith("sha256:")
    assert report["provenance"]["promotion_metrics_sha256"].startswith("sha256:")
    assert report["provenance"]["evaluated_model_state_sha256"] == (
        "sha256:" + "1" * 64
    )
    assert report["provenance"]["evaluation_epoch"] == 1
    assert report["provenance"]["evaluation_optimizer_step"] == 32
    assert report["provenance_sha256"].startswith("sha256:")
    assert calls == [(0,), (1, 2, 3), (4, 5)]


def test_natural_outcome_validation_is_not_labeled_sampler_balanced() -> None:
    data = _Composite()
    # Natural holdout mass is deliberately outcome-skewed: the high-loss
    # positive game contributes three rows, while the negative game contributes
    # one. Component/game aggregation remains authenticated, but the validation
    # outcome weights do not match sampler_balanced_v1 fitted on training.
    per_row_loss = np.asarray([1.0, 9.0, 9.0, 9.0, 4.0, 4.0])

    def evaluate(indices: np.ndarray) -> dict:
        value = float(per_row_loss[indices].mean())
        return {
            "samples": int(len(indices)),
            "loss": value,
            "value_loss": value,
            "loss_denominators": {"value_loss": float(len(indices))},
            "objective_coefficients": {"value_loss": 1.0},
        }

    report = evaluate_composite_validation_measure(
        data,
        np.arange(6, dtype=np.int64),
        evaluate,
        evaluation_identity=_evaluation_identity(),
        training_value_player_outcome_balance_mode="sampler_balanced_v1",
        validation_value_player_outcome_balance_mode="none",
    )

    assert report["metrics"]["value_loss"] == pytest.approx(4.75)
    assert report["objective_matched"] is False
    assert report["schema_version"] == "composite-validation-measure-v3"
    assert report["validation_key"] == "validation_natural_composite"
    assert report["objective_match"] == {
        "component_game_row_sampling_matched": True,
        "training_value_player_outcome_balance_mode": "sampler_balanced_v1",
        "validation_value_player_outcome_balance_mode": "none",
        "value_player_outcome_balance_matched": False,
        "validation_outcome_measure": "natural_holdout_v1",
    }
    assert report["provenance"][
        "value_player_outcome_balance_matched"
    ] is False
    epoch = {"validation_natural_composite": report}
    with pytest.raises(ValueError, match="natural composite validation"):
        objective_matched_validation_metrics(
            epoch
        )
    report["objective_matched"] = True
    with pytest.raises(ValueError, match="key-role provenance"):
        objective_matched_validation_metrics(
            {"validation_objective_matched": report}
        )


def test_natural_v3_cannot_be_redeclared_as_objective_matched_without_invalidating_provenance() -> None:
    report = evaluate_composite_validation_measure(
        _Composite(),
        np.arange(6, dtype=np.int64),
        lambda indices: {
            "samples": int(len(indices)),
            "loss": float(len(indices)),
            "value_loss": float(len(indices)),
            "loss_denominators": {"value_loss": float(len(indices))},
            "objective_coefficients": {"value_loss": 1.0},
        },
        evaluation_identity=_evaluation_identity(),
        training_value_player_outcome_balance_mode="sampler_balanced_v1",
        validation_value_player_outcome_balance_mode="none",
    )
    report["objective_matched"] = True
    report["validation_key"] = "validation_objective_matched"
    report["objective_match"] = {
        **report["objective_match"],
        "training_value_player_outcome_balance_mode": "none",
        "value_player_outcome_balance_matched": True,
    }

    with pytest.raises(ValueError, match="key-role provenance"):
        objective_matched_validation_metrics(
            {"validation_objective_matched": report},
            require_matched=True,
            require_provenance=True,
        )


def test_objective_matched_validation_rejects_missing_component_holdout() -> None:
    data = _Composite()

    try:
        evaluate_composite_validation_measure(
            data,
            np.arange(4, dtype=np.int64),
            lambda indices: {"loss": float(len(indices))},
            evaluation_identity=_evaluation_identity(),
        )
    except SystemExit as error:
        assert "replay" in str(error)
        assert "no rows" in str(error)
    else:  # pragma: no cover - fail-closed contract assertion
        raise AssertionError("missing authenticated validation component was accepted")


def _matched_epoch() -> dict:
    data = _Composite()
    wrapper = evaluate_composite_validation_measure(
        data,
        np.arange(6, dtype=np.int64),
        lambda indices: {
            "loss": float(len(indices)),
            "policy_loss": float(len(indices)),
            "samples": int(len(indices)),
        },
        evaluation_identity=_evaluation_identity(),
    )
    return {
        "validation": {"loss": 99.0},
        "validation_objective_matched": wrapper,
    }


def test_downstream_metric_selector_rejects_tampered_matched_metrics() -> None:
    epoch = _matched_epoch()
    epoch["validation_objective_matched"]["metrics"]["loss"] = 4.0

    with pytest.raises(ValueError, match="provenance"):
        objective_matched_validation_metrics(epoch, require_provenance=True)


def test_downstream_metric_selector_rejects_tampered_component_metrics() -> None:
    epoch = _matched_epoch()
    epoch["validation_objective_matched"]["components"]["replay"]["metrics"][
        "loss"
    ] = 4.0

    with pytest.raises(ValueError, match="provenance"):
        objective_matched_validation_metrics(epoch, require_provenance=True)


def test_downstream_metric_selector_rejects_replayed_step_identity() -> None:
    epoch = _matched_epoch()
    matched = epoch["validation_objective_matched"]
    matched["provenance"]["evaluation_optimizer_step"] = 31
    matched["provenance_sha256"] = _canonical_json_sha256(matched["provenance"])

    with pytest.raises(ValueError, match="checkpoint identity"):
        objective_matched_validation_metrics(epoch, require_provenance=True)


def test_downstream_metric_selector_falls_back_only_for_explicit_legacy_use() -> None:
    assert objective_matched_validation_metrics({"validation": {"loss": 2.0}}) == {
        "loss": 2.0
    }


def test_downstream_component_selector_requires_authenticated_wrapper() -> None:
    epoch = _matched_epoch()
    components = objective_matched_validation_component_metrics(epoch)
    assert set(components) == {"n128", "replay"}
    with pytest.raises(ValueError, match="objective-matched"):
        objective_matched_validation_component_metrics({}, require_matched=True)


def test_downstream_metric_selector_does_not_trust_unmarked_wrapper() -> None:
    epoch = {
        "validation": {"loss": 2.0},
        "validation_objective_matched": {
            "objective_matched": False,
            "metrics": {"loss": 1.0},
        },
    }
    with pytest.raises(ValueError, match="wrapper is malformed"):
        objective_matched_validation_metrics(epoch)
    with pytest.raises(ValueError, match="wrapper is malformed"):
        objective_matched_validation_metrics(epoch, require_matched=True)


def test_downstream_metric_selector_rejects_malformed_component_coverage() -> None:
    epoch = _matched_epoch()
    epoch["validation_objective_matched"]["components"]["replay"]["rows"] += 1

    with pytest.raises(ValueError, match="component coverage"):
        objective_matched_validation_metrics(epoch, require_matched=True)


def test_downstream_metric_selector_can_require_bound_provenance() -> None:
    epoch = _matched_epoch()
    assert objective_matched_validation_metrics(
        epoch, require_matched=True, require_provenance=True
    )
    del epoch["validation_objective_matched"]["provenance"]
    del epoch["validation_objective_matched"]["provenance_sha256"]

    with pytest.raises(ValueError, match="provenance"):
        objective_matched_validation_metrics(
            epoch, require_matched=True, require_provenance=True
        )


def test_objective_measure_aggregates_weight_density_before_dividing() -> None:
    data = _Composite()

    def evaluate(indices: np.ndarray) -> dict:
        # Game 11 has one active row with loss 1. Game 12 has one active row out
        # of three with loss 9. Equal-game averaging of normalized losses gives
        # 5, but the actual component->game->row training objective is
        # ((1/1 + 9/3)/2) / ((1/1 + 1/3)/2) = 3.
        seed = int(data["game_seed"][indices[0]])
        policy_loss = {11: 1.0, 12: 9.0, 21: 5.0}[seed]
        denominator = {11: 1.0, 12: 1.0, 21: 2.0}[seed]
        return {
            "samples": int(len(indices)),
            "loss": policy_loss,
            "raw_batch_mean_loss": policy_loss,
            "component_reconstructed_loss": policy_loss,
            "policy_loss": policy_loss,
            "loss_denominators": {"policy_loss": denominator},
            "objective_coefficients": {"policy_loss": 1.0},
        }

    report = evaluate_composite_validation_measure(
        data,
        np.arange(6, dtype=np.int64),
        evaluate,
        evaluation_identity=_evaluation_identity(),
    )

    assert report["components"]["n128"]["metrics"]["policy_loss"] == 3.0
    # Overall: n128 numerator density=2, denominator density=2/3; replay both
    # densities are 5 and 1. Apply .75/.25, then divide: 2.75 / .75.
    assert report["metrics"]["policy_loss"] == pytest.approx(11.0 / 3.0)
    assert report["metrics"]["loss"] == pytest.approx(11.0 / 3.0)
    assert report["metrics"]["raw_batch_mean_loss"] == 5.0


def test_objective_measure_density_scales_behavioral_sufficient_stats() -> None:
    abi = {
        "version": "synthetic-v1",
        "size": 2,
        "ordered_descriptors_sha256": "sha256:" + "1" * 64,
        "action_types_by_id_sha256": "sha256:" + "2" * 64,
        "identity_sha256": "sha256:" + "3" * 64,
    }

    def one_report(
        *,
        samples: int,
        rows: float,
        confusion: float,
        weighted_rows: float,
        weighted_confusion: float,
    ) -> dict:
        uniform = _empty_policy_target_behavior_parts()
        uniform.update(
            {
                "rows": rows,
                "teacher_top1_correct": rows - confusion,
                "teacher_top3_correct": rows,
                "teacher_top3_mass_sum": rows,
                "end_turn_confusion_rows": confusion,
                "end_turn_confusion_teacher_probability_regret_sum": (
                    confusion * 0.5
                ),
            }
        )
        weighted = _empty_policy_target_behavior_parts()
        weighted.update(
            {
                "rows": weighted_rows,
                "teacher_top1_correct": weighted_rows - weighted_confusion,
                "teacher_top3_correct": weighted_rows,
                "teacher_top3_mass_sum": weighted_rows,
                "end_turn_confusion_rows": weighted_confusion,
                "end_turn_confusion_teacher_probability_regret_sum": (
                    weighted_confusion * 0.5
                ),
            }
        )
        return {
            "samples": samples,
            "loss": 0.0,
            "policy_loss": 0.0,
            "loss_denominators": {},
            "policy_target_distribution_sufficient_statistics": {
                "schema_version": (
                    "policy-target-distribution-sufficient-stats-v1"
                ),
                "overall": _empty_policy_target_metric_parts(),
                "objective_weighted_overall": (
                    _empty_policy_target_metric_parts()
                ),
                "phase": {},
                "opening_decision_index": {},
                "behavioral_competence": {
                    "schema_version": (
                        "policy-target-behavior-sufficient-stats-v1"
                    ),
                    "action_catalog_abi": dict(abi),
                    "teacher_argmax_action_type": {
                        "MARITIME_TRADE": uniform
                    },
                    "objective_weighted_teacher_argmax_action_type": {
                        "MARITIME_TRADE": weighted
                    },
                },
            },
        }

    metrics, sufficient = _objective_measure_validation_aggregate(
        [
            one_report(
                samples=10,
                rows=5.0,
                confusion=5.0,
                weighted_rows=2.0,
                weighted_confusion=2.0,
            ),
            one_report(
                samples=20,
                rows=10.0,
                confusion=0.0,
                weighted_rows=8.0,
                weighted_confusion=0.0,
            ),
        ],
        np.asarray([0.7, 0.3], dtype=np.float64),
    )

    behavior = metrics["policy_target_distribution_metrics"][
        "behavioral_competence"
    ]
    uniform = behavior["teacher_argmax_action_type"]["MARITIME_TRADE"]
    assert uniform["row_probability"] == pytest.approx(0.5)
    assert "rows" not in uniform
    assert uniform["end_turn_confusion_rate"] == pytest.approx(0.7)
    assert uniform[
        "end_turn_confusion_teacher_probability_regret_per_row"
    ] == pytest.approx(0.35)
    assert uniform[
        "end_turn_confusion_teacher_probability_regret_conditional_mean"
    ] == pytest.approx(0.5)
    weighted = behavior[
        "objective_weighted_teacher_argmax_action_type"
    ]["MARITIME_TRADE"]
    assert weighted["row_probability"] == pytest.approx(0.26)
    assert weighted["end_turn_confusion_rate"] == pytest.approx(7.0 / 13.0)
    assert sufficient is not None
    nested = sufficient["policy_target_distribution_metrics"][
        "behavioral_competence"
    ]
    assert nested["teacher_argmax_action_type"]["MARITIME_TRADE"][
        "rows"
    ] == pytest.approx(0.5)


def test_common_uniform_value_metric_ignores_composite_objective_ratios() -> None:
    reports = [
        {
            "samples": 1,
            "loss": 1.0,
            "common_uniform_clean_outcome_scalar_mse": {
                "schema_version": "common-uniform-clean-outcome-scalar-mse-v1",
                "measure": "uniform_clean_terminal_outcome_rows",
                "target": "actor_perspective_terminal_outcome_pm1",
                "prediction_readout": "raw",
                "prediction_scale": 1.0,
                "training_value_sample_weights_applied": False,
                "outcome_confidence_applied": False,
                "truncated_rows_included": False,
                "root_value_blend_applied": False,
                "available": True,
                "eligible_rows": 1,
                "squared_error_sum": 9.0,
                "mse": 9.0,
            },
        },
        {
            "samples": 9,
            "loss": 3.0,
            "common_uniform_clean_outcome_scalar_mse": {
                "schema_version": "common-uniform-clean-outcome-scalar-mse-v1",
                "measure": "uniform_clean_terminal_outcome_rows",
                "target": "actor_perspective_terminal_outcome_pm1",
                "prediction_readout": "raw",
                "prediction_scale": 1.0,
                "training_value_sample_weights_applied": False,
                "outcome_confidence_applied": False,
                "truncated_rows_included": False,
                "root_value_blend_applied": False,
                "available": True,
                "eligible_rows": 9,
                "squared_error_sum": 9.0,
                "mse": 1.0,
            },
        },
    ]

    left, _ = _objective_measure_validation_aggregate(
        reports, np.asarray([0.9, 0.1], dtype=np.float64)
    )
    right, _ = _objective_measure_validation_aggregate(
        reports, np.asarray([0.1, 0.9], dtype=np.float64)
    )

    assert left["loss"] != right["loss"]
    assert left["common_uniform_clean_outcome_scalar_mse"] == right[
        "common_uniform_clean_outcome_scalar_mse"
    ]
    assert left["common_uniform_clean_outcome_scalar_mse"]["mse"] == pytest.approx(
        1.8
    )


def test_common_uniform_value_reduction_uses_global_sse_and_row_count(
    monkeypatch,
) -> None:
    def reduce_named(values, _ddp):
        assert values == {"squared_error_sum": 5.0, "eligible_rows": 2.0}
        return {"squared_error_sum": 14.0, "eligible_rows": 5.0}

    monkeypatch.setattr("tools.train_bc._reduce_named_sums", reduce_named)
    report = _reduce_common_uniform_clean_outcome_scalar_mse(
        5.0,
        2,
        {"enabled": True, "world_size": 2, "rank": 0, "local_rank": 0},
        prediction_readout="raw",
        prediction_scale=1.0,
    )

    assert report["eligible_rows"] == 5
    assert report["squared_error_sum"] == 14.0
    assert report["mse"] == pytest.approx(2.8)


def test_common_uniform_value_metric_rejects_mismatched_component_contracts() -> None:
    common = {
        "schema_version": "common-uniform-clean-outcome-scalar-mse-v1",
        "measure": "uniform_clean_terminal_outcome_rows",
        "target": "actor_perspective_terminal_outcome_pm1",
        "prediction_readout": "raw",
        "prediction_scale": 1.0,
        "training_value_sample_weights_applied": False,
        "outcome_confidence_applied": False,
        "truncated_rows_included": False,
        "root_value_blend_applied": False,
        "available": True,
        "eligible_rows": 1,
        "squared_error_sum": 1.0,
        "mse": 1.0,
    }
    reports = [
        {"samples": 1, "loss": 1.0, "common_uniform_clean_outcome_scalar_mse": common},
        {
            "samples": 1,
            "loss": 1.0,
            "common_uniform_clean_outcome_scalar_mse": {
                **common,
                "prediction_readout": "deployed_search",
            },
        },
    ]

    with pytest.raises(SystemExit, match="prediction contracts differ"):
        _objective_measure_validation_aggregate(
            reports, np.asarray([0.5, 0.5], dtype=np.float64)
        )


@pytest.mark.parametrize("eligible_rows", [True, 1.5, "1"])
def test_common_uniform_value_metric_rejects_noninteger_row_counts(
    eligible_rows,
) -> None:
    common = {
        "schema_version": "common-uniform-clean-outcome-scalar-mse-v1",
        "measure": "uniform_clean_terminal_outcome_rows",
        "target": "actor_perspective_terminal_outcome_pm1",
        "prediction_readout": "raw",
        "prediction_scale": 1.0,
        "training_value_sample_weights_applied": False,
        "outcome_confidence_applied": False,
        "truncated_rows_included": False,
        "root_value_blend_applied": False,
        "available": True,
        "eligible_rows": eligible_rows,
        "squared_error_sum": 1.0,
        "mse": 1.0,
    }

    with pytest.raises(SystemExit, match="values are malformed"):
        _objective_measure_validation_aggregate(
            [
                {
                    "samples": 1,
                    "loss": 1.0,
                    "common_uniform_clean_outcome_scalar_mse": common,
                }
            ],
            np.asarray([1.0], dtype=np.float64),
        )


def test_exact_total_loss_reconstruction_includes_zero_weight_belief_term() -> None:
    """An always-reported zero-weight term must not disable reconstruction.

    Both reports contain 100 rows.  The first game has one active policy row at
    loss 10; the second has 100 active rows at loss 0.  Averaging their already
    normalised total losses gives 5.25, while the configured population
    objective is 10 / 101 + 0.25 = 0.3490099.
    """

    coefficient_keys = (
        "policy_loss",
        "value_loss",
        "final_vp_loss",
        "q_loss",
        "policy_kl_anchor_loss",
        "value_uncertainty_loss",
        "aux_subgoal_loss",
        "belief_resource_loss",
        "moe_balance_loss",
        "value_categorical_loss",
    )
    coefficients = {key: 0.0 for key in coefficient_keys}
    coefficients.update({"policy_loss": 1.0, "value_loss": 0.25})

    def report(*, policy_loss: float, policy_rows: float) -> dict:
        losses = {key: 0.0 for key in coefficient_keys}
        losses.update({"policy_loss": policy_loss, "value_loss": 1.0})
        denominators = {key: 0.0 for key in coefficient_keys}
        denominators.update({"policy_loss": policy_rows, "value_loss": 100.0})
        return {
            "samples": 100,
            "loss": policy_loss + 0.25,
            "raw_batch_mean_loss": policy_loss + 0.25,
            "component_reconstructed_loss": policy_loss + 0.25,
            "scalar_value_mse_diagnostic": 1.0,
            "primary_value_loss": 1.0,
            "loss_denominators": denominators,
            "objective_coefficients": coefficients,
            **losses,
        }

    metrics, sufficient = _objective_measure_validation_aggregate(
        [
            report(policy_loss=10.0, policy_rows=1.0),
            report(policy_loss=0.0, policy_rows=100.0),
        ],
        np.asarray([0.5, 0.5]),
    )

    expected_policy = 10.0 / 101.0
    expected_total = expected_policy + 0.25
    assert metrics["raw_batch_mean_loss"] == pytest.approx(5.25)
    assert metrics["policy_loss"] == pytest.approx(expected_policy)
    assert metrics["value_loss"] == pytest.approx(1.0)
    assert metrics["scalar_value_mse_diagnostic"] == pytest.approx(1.0)
    assert metrics["primary_value_loss"] == pytest.approx(1.0)
    assert metrics["component_reconstructed_loss"] == pytest.approx(expected_total)
    assert metrics["loss"] == pytest.approx(expected_total)
    assert sufficient is not None
    assert "belief_resource_loss" in sufficient


def test_objective_matched_validation_excludes_value_only_replay_policy_ce() -> None:
    data = _Composite()
    data.policy_distillation_scope_authenticated = True
    data.policy_distillation_component_indices = (0,)

    def evaluate(indices: np.ndarray) -> dict:
        replay = bool(np.all(indices >= 4))
        return {
            "samples": int(len(indices)),
            "loss": 0.0 if replay else 2.0,
            "policy_loss": 0.0 if replay else 2.0,
            "loss_denominators": {
                "policy_loss": 0.0 if replay else float(len(indices))
            },
            "objective_coefficients": {"policy_loss": 1.0},
        }

    report = evaluate_composite_validation_measure(
        data,
        np.arange(6, dtype=np.int64),
        evaluate,
        evaluation_identity=_evaluation_identity(),
    )
    assert report["policy_distillation_component_ids"] == ["n128"]
    assert report["components"]["n128"]["policy_distillation_enabled"] is True
    assert report["components"]["replay"]["policy_distillation_enabled"] is False
    assert report["components"]["replay"]["metrics"]["policy_loss"] == 0.0
    assert report["metrics"]["policy_loss"] == 2.0


def test_teacher_gap_closure_uses_aggregate_kl_mass_not_mean_of_game_ratios() -> None:
    data = _Composite()

    def evaluate(indices: np.ndarray) -> dict:
        seed = int(data["game_seed"][indices[0]])
        # Game 11: one active row, prior gap 10, model gap 0 (closure 1).
        # Game 12: one active row among three, prior/model gap 1 (closure 0).
        # The correct n128 closure is 1 - (0/1 + 1/3)/(10/1 + 1/3),
        # not mean(1, 0)=.5.
        prior = {11: 10.0, 12: 1.0, 21: 2.0}[seed]
        model = {11: 0.0, 12: 1.0, 21: 1.0}[seed]
        return {
            "samples": int(len(indices)),
            "active_policy_teacher_gap_rows": 1,
            "active_policy_kl_target_model_mean": model,
            "active_policy_kl_target_prior_mean": prior,
            "active_policy_teacher_gap_closure": 1.0 - model / prior,
        }

    report = evaluate_composite_validation_measure(
        data,
        np.arange(6, dtype=np.int64),
        evaluate,
        evaluation_identity=_evaluation_identity(),
    )

    assert report["components"]["n128"]["metrics"][
        "active_policy_teacher_gap_closure"
    ] == pytest.approx(1.0 - (1.0 / 3.0) / (10.0 + 1.0 / 3.0))
