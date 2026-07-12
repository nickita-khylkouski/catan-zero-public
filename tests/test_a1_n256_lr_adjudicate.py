from __future__ import annotations

import math

from tools import a1_n256_lr_adjudicate as adjudicator


def _row(
    *,
    external: float,
    external_ci: tuple[float, float],
    internal: float,
    internal_ci: tuple[float, float],
    teacher_gap: float,
    value_mse: float,
    validation_loss: float = 1.0,
    clipped_fraction: float | None = 0.1,
    max_grad: float | None = 2.0,
) -> dict[str, object]:
    return {
        "external_delta": external,
        "external_delta_ci": list(external_ci),
        "internal_win_rate": internal,
        "internal_win_rate_ci": list(internal_ci),
        "teacher_gap_closure": teacher_gap,
        "value_mse": value_mse,
        "validation_loss": validation_loss,
        "clipped_fraction": clipped_fraction,
        "max_pre_clip_grad_norm": max_grad,
    }


def test_external_clear_win_outweighs_conflicting_secondary_metrics() -> None:
    rows = {
        "external": _row(
            external=0.09,
            external_ci=(0.06, 0.12),
            internal=0.48,
            internal_ci=(0.44, 0.52),
            teacher_gap=0.10,
            value_mse=0.30,
        ),
        "internal": _row(
            external=0.00,
            external_ci=(-0.02, 0.02),
            internal=0.62,
            internal_ci=(0.58, 0.66),
            teacher_gap=0.30,
            value_mse=0.20,
        ),
    }

    result = adjudicator.adjudicate_metrics(rows)

    assert result["winner"] == "external"
    assert result["decision"] == "diagnostic_winner"
    assert result["ranking"][0] == "external"
    assert set(result["pareto_frontier"]) == {"external", "internal"}


def test_overlapping_uncertainty_and_tradeoffs_allow_no_winner() -> None:
    rows = {
        "a": _row(
            external=0.02,
            external_ci=(-0.04, 0.08),
            internal=0.54,
            internal_ci=(0.47, 0.61),
            teacher_gap=0.20,
            value_mse=0.21,
        ),
        "b": _row(
            external=0.01,
            external_ci=(-0.05, 0.07),
            internal=0.53,
            internal_ci=(0.46, 0.60),
            teacher_gap=0.21,
            value_mse=0.20,
        ),
    }

    result = adjudicator.adjudicate_metrics(rows)

    assert result["winner"] is None
    assert result["decision"] == "no_winner"
    assert result["decision_reason"] == "uncertainty_or_tradeoff_unresolved"


def test_exact_ties_do_not_invent_a_winner() -> None:
    tied = _row(
        external=0.03,
        external_ci=(-0.02, 0.08),
        internal=0.55,
        internal_ci=(0.49, 0.61),
        teacher_gap=0.25,
        value_mse=0.20,
    )

    result = adjudicator.adjudicate_metrics(
        {"lr60u": dict(tied), "lr120u": dict(tied), "lr240u": dict(tied)}
    )

    assert result["winner"] is None
    assert result["pareto_frontier"] == ["lr120u", "lr240u", "lr60u"]


def test_clipping_pathology_rejects_apparent_best_arm() -> None:
    rows = {
        "unsafe": _row(
            external=0.20,
            external_ci=(0.15, 0.25),
            internal=0.70,
            internal_ci=(0.65, 0.75),
            teacher_gap=0.40,
            value_mse=0.10,
            clipped_fraction=0.75,
        ),
        "safe": _row(
            external=0.04,
            external_ci=(0.02, 0.06),
            internal=0.57,
            internal_ci=(0.52, 0.62),
            teacher_gap=0.25,
            value_mse=0.20,
        ),
    }

    result = adjudicator.adjudicate_metrics(rows)

    assert result["winner"] == "safe"
    assert result["safety"]["unsafe"]["eligible"] is False
    assert "clipping_fraction_pathology" in result["safety"]["unsafe"]["reasons"]


def test_nonfinite_arm_is_rejected_and_dominated_arm_is_not_pareto() -> None:
    rows = {
        "best": _row(
            external=0.05,
            external_ci=(0.03, 0.07),
            internal=0.58,
            internal_ci=(0.54, 0.62),
            teacher_gap=0.30,
            value_mse=0.18,
        ),
        "dominated": _row(
            external=0.01,
            external_ci=(-0.01, 0.03),
            internal=0.54,
            internal_ci=(0.50, 0.58),
            teacher_gap=0.20,
            value_mse=0.24,
        ),
        "nan": _row(
            external=math.nan,
            external_ci=(-0.01, 0.03),
            internal=0.70,
            internal_ci=(0.66, 0.74),
            teacher_gap=0.50,
            value_mse=0.10,
        ),
    }

    result = adjudicator.adjudicate_metrics(rows)

    assert result["winner"] == "best"
    assert result["pareto_frontier"] == ["best"]
    assert result["safety"]["nan"]["eligible"] is False
    assert "nonfinite_or_missing:external_delta" in result["safety"]["nan"]["reasons"]


def test_missing_optimizer_telemetry_is_explicit_uncertainty_not_fake_zero() -> None:
    row = _row(
        external=0.04,
        external_ci=(0.02, 0.06),
        internal=0.56,
        internal_ci=(0.52, 0.60),
        teacher_gap=0.25,
        value_mse=0.20,
        clipped_fraction=None,
        max_grad=None,
    )

    result = adjudicator.adjudicate_metrics({"only": row})

    assert result["winner"] == "only"
    assert result["safety"]["only"]["optimizer_telemetry_available"] is False
