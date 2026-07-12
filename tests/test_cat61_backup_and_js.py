"""CAT-61 search-side: capped uncertainty backup weighting + closed-form
James-Stein D2 shrinkage. All flag-gated, default OFF = bit-identical search.

These exercise the pure MCTS logic (weight formula/cap, weighted-Q backup,
completed-Q selection, shrinkage math) without the rust engine, by bypassing
`GumbelChanceMCTS.__init__` (which requires the engine) via `__new__` and
constructing `_GNode`/`_GAction` directly.
"""

from __future__ import annotations

import random

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    _GAction,
    _GNode,
    _split_evaluation,
)


class _FakeGame:
    def __init__(self, color: str) -> None:
        self._color = color

    def current_color(self) -> str:
        return self._color


def _mcts(**config_kwargs) -> GumbelChanceMCTS:
    mcts = GumbelChanceMCTS.__new__(GumbelChanceMCTS)
    mcts.config = GumbelChanceMCTSConfig(**config_kwargs)
    mcts.rng = random.Random(0)
    return mcts


def _visited_action(prior, values):
    """A _GAction that has been backed up once per value in `values`."""
    stats = _GAction(prior=prior)
    for v in values:
        stats.visits += 1
        stats.value_sum += v
        stats.value_sq_sum += v * v
    return stats


# --- evaluator-result unpacking ----------------------------------------------
def test_split_evaluation_handles_2_and_3_tuples():
    priors = {1: 0.5, 2: 0.5}
    assert _split_evaluation((priors, 0.3)) == (priors, 0.3, 0.0)
    assert _split_evaluation((priors, 0.3, 0.7)) == (priors, 0.3, 0.7)


# --- _GAction.weighted_q -----------------------------------------------------
def test_weighted_q_falls_back_to_plain_q_without_weights():
    stats = _visited_action(0.5, [0.2, 0.4])  # q = 0.3, no weights recorded
    assert stats.weight_sum == 0.0
    assert stats.weighted_q == pytest.approx(stats.q) == pytest.approx(0.3)


def test_weighted_q_is_the_weight_weighted_mean():
    stats = _GAction(prior=0.5, visits=2, value_sum=0.6)
    stats.weight_sum = 3.0
    stats.weighted_value_sum = 3.0 * 0.9  # a single dominant, heavy backup
    assert stats.weighted_q == pytest.approx(0.9)
    assert stats.weighted_q != pytest.approx(stats.q)


# --- backup weight formula + cap ---------------------------------------------
def test_backup_weight_formula_matches_inverse_uncertainty_operator():
    mcts = _mcts(
        uncertainty_backup_a=0.25,
        uncertainty_backup_exp=2.0,
        uncertainty_backup_cap=10.0,
    )
    # The head predicts squared error: sigma=sqrt(3), then
    # weight = 0.25 / (sigma**2 + 0.25/10).
    assert mcts._backup_weight(3.0) == pytest.approx(0.25 / 3.025)


def test_backup_weight_cap_engages():
    mcts = _mcts(
        uncertainty_backup_a=0.25,
        uncertainty_backup_exp=1.0,
        uncertainty_backup_cap=0.5,
    )
    assert mcts._backup_weight(0.0) == pytest.approx(0.5)
    assert mcts._backup_weight(1.0) == pytest.approx(1.0 / 6.0)
    assert mcts._backup_weight(4.0) == pytest.approx(0.1)
    assert mcts._backup_weight(0.0) > mcts._backup_weight(1.0) > mcts._backup_weight(4.0)


def test_backup_weight_clamps_negative_err():
    mcts = _mcts(uncertainty_backup_a=0.25, uncertainty_backup_exp=0.5, uncertainty_backup_cap=10.0)
    # Squared-error predictions are clamped to 0 before sqrt/fractional power.
    assert mcts._backup_weight(-2.0) == pytest.approx(10.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("uncertainty_backup_a", 0.0),
        ("uncertainty_backup_exp", 0.0),
        ("uncertainty_backup_cap", 0.0),
    ],
)
def test_backup_weight_rejects_invalid_operator_parameters(field, value):
    mcts = _mcts(**{field: value})
    with pytest.raises(ValueError, match="finite and > 0"):
        mcts._backup_weight(1.0)


def test_accumulate_backup_weight_records_distribution_and_cap_hits():
    """Certain leaves hit the cap; uncertain leaves receive less influence."""
    mcts = _mcts(
        uncertainty_backup_weighting=True,
        uncertainty_backup_a=0.25,
        uncertainty_backup_exp=1.0,
        uncertainty_backup_cap=0.5,
    )
    mcts._last_backup_weights = []
    stats = _GAction(prior=1.0)
    for err, value in [(0.0, 0.1), (4.0, -0.2), (8.0, 0.3)]:
        mcts._accumulate_backup_weight(stats, value, err)
    weights = mcts._last_backup_weights
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(0.1)
    assert weights[2] == pytest.approx(0.25 / (8.0**0.5 + 0.5))
    assert weights[0] > weights[1] > weights[2]
    assert stats.weight_sum == pytest.approx(sum(weights))
    assert stats.weighted_value_sum == pytest.approx(
        weights[0] * 0.1 + weights[1] * -0.2 + weights[2] * 0.3
    )


# --- completed-Q uses weighted_q only when the flag is on --------------------
def _node_two_actions():
    node = _GNode(game=_FakeGame("RED"), root_color="RED", prior_value=0.0)
    node.actions = {
        1: _visited_action(0.6, [0.2, 0.2, 0.2]),  # q = 0.2
        2: _visited_action(0.4, [0.8, 0.8]),  # q = 0.8
    }
    return node


def test_completed_q_default_path_is_unweighted_and_bit_identical():
    node = _node_two_actions()
    off = _mcts(uncertainty_backup_weighting=False)
    completed = off._completed_q(node)
    # visited actions keep their plain q (sign +1, root to act)
    assert completed[1] == pytest.approx(0.2)
    assert completed[2] == pytest.approx(0.8)


def test_completed_q_uses_weighted_q_when_weighting_on():
    node = _node_two_actions()
    # Give action 1 a heavy-weighted backup that shifts its weighted_q well
    # above its plain q; action 2 keeps weighted_q == q (no weights recorded).
    node.actions[1].weight_sum = 4.0
    node.actions[1].weighted_value_sum = 4.0 * 0.5  # weighted_q = 0.5, not 0.2
    on = _mcts(uncertainty_backup_weighting=True)
    completed = on._completed_q(node)
    assert completed[1] == pytest.approx(0.5)  # weighted, not 0.2
    assert completed[2] == pytest.approx(0.8)  # falls back to q


# --- D2 closed-form James-Stein shrinkage ------------------------------------
def _node_three_visited():
    node = _GNode(game=_FakeGame("RED"), root_color="RED", prior_value=0.0)
    node.actions = {
        1: _visited_action(0.4, [0.9, 0.7, 0.8, 0.6]),
        2: _visited_action(0.4, [-0.5, -0.6, -0.4, -0.5]),
        3: _visited_action(0.2, [0.1, 0.0, -0.1, 0.05]),
    }
    return node


def test_closed_form_js_applies_one_lambda_in_unit_interval():
    node = _node_three_visited()
    mcts = _mcts(variance_aware_q=True, variance_aware_closed_form_js=True)
    completed = mcts._completed_q(node)  # runs the shrinkage internally

    # Recompute lambda* = v2 / (v2 + mean_se2) from the same inputs and confirm
    # every visited candidate was shrunk toward v_mix by that ONE coefficient.
    sign = 1.0
    total_visits = sum(s.visits for s in node.actions.values())
    vp_sum = sum(s.prior for s in node.actions.values() if s.visits > 0)
    vq_sum = sum(s.prior * (sign * s.q) for s in node.actions.values() if s.visits > 0)
    weighted_q = vq_sum / vp_sum
    v_mix = (sign * node.prior_value + total_visits * weighted_q) / (1.0 + total_visits)

    raw = {aid: sign * s.q for aid, s in node.actions.items()}
    mean_q = sum(raw.values()) / len(raw)
    signal_var = sum((q - mean_q) ** 2 for q in raw.values()) / len(raw)
    se_sqs = [s.q_variance / s.visits for s in node.actions.values()]
    mean_se_sq = sum(se_sqs) / len(se_sqs)
    lam = signal_var / (signal_var + mean_se_sq)

    assert 0.0 <= lam <= 1.0
    for aid in node.actions:
        assert completed[aid] == pytest.approx(v_mix + lam * (raw[aid] - v_mix))


def test_closed_form_js_off_uses_per_arm_k_formula():
    """With the closed-form flag OFF, D2 still uses the per-candidate k-tuned
    shrinkage (each arm gets its own coefficient), not the single lambda*."""
    node = _node_three_visited()
    mcts = _mcts(variance_aware_q=True, variance_aware_closed_form_js=False, variance_aware_k=1.0)
    completed = mcts._completed_q(node)

    sign = 1.0
    total_visits = sum(s.visits for s in node.actions.values())
    vp_sum = sum(s.prior for s in node.actions.values() if s.visits > 0)
    vq_sum = sum(s.prior * (sign * s.q) for s in node.actions.values() if s.visits > 0)
    v_mix = (sign * node.prior_value + total_visits * (vq_sum / vp_sum)) / (1.0 + total_visits)
    raw = {aid: sign * s.q for aid, s in node.actions.items()}
    mean_q = sum(raw.values()) / len(raw)
    signal_var = sum((q - mean_q) ** 2 for q in raw.values()) / len(raw)
    for aid, s in node.actions.items():
        se_sq = s.q_variance / s.visits
        shrink = signal_var / (signal_var + 1.0 * se_sq)  # per-arm coefficient
        assert completed[aid] == pytest.approx(v_mix + shrink * (raw[aid] - v_mix))


def test_d2_completely_off_leaves_completed_q_unshrunk():
    node = _node_three_visited()
    mcts = _mcts(variance_aware_q=False, variance_aware_closed_form_js=True)  # gate is variance_aware_q
    completed = mcts._completed_q(node)
    for aid, s in node.actions.items():
        assert completed[aid] == pytest.approx(s.q)  # untouched raw Q
