from __future__ import annotations

import numpy as np

from tools.generate_rust_mcts_reanalysis import _target_scores_and_mask


def test_unvisited_action_is_excluded_even_though_q_defaults_to_a_finite_zero() -> None:
    """FIX (Q-mask): RustMCTSResult.q_values defaults an unvisited action's Q to 0.0 (finite,
    not NaN), so isfinite alone can't tell an unvisited action from a genuinely-scored one at
    Q==0. The mask must also require visits > 0."""
    legal_rust = (10, 20, 30)
    q_by_rust = {10: 0.4, 20: 0.0, 30: -0.2}  # action 20 has the "looks legal" finite 0.0 Q
    visits_by_rust = {10: 5, 20: 0, 30: 3}  # action 20 was never actually visited

    target_scores, mask = _target_scores_and_mask(q_by_rust, visits_by_rust, legal_rust)

    np.testing.assert_allclose(target_scores, [0.4, 0.0, -0.2])
    assert mask.tolist() == [True, False, True]


def test_missing_q_value_stays_excluded_via_nan() -> None:
    legal_rust = (10, 20)
    q_by_rust = {10: 0.4}  # action 20 never got a q entry at all
    visits_by_rust = {10: 5, 20: 5}

    target_scores, mask = _target_scores_and_mask(q_by_rust, visits_by_rust, legal_rust)

    assert np.isnan(target_scores[1])
    assert mask.tolist() == [True, False]


def test_all_visited_matches_plain_isfinite_behavior() -> None:
    legal_rust = (1, 2, 3)
    q_by_rust = {1: 0.1, 2: 0.2, 3: 0.3}
    visits_by_rust = {1: 1, 2: 4, 3: 2}

    target_scores, mask = _target_scores_and_mask(q_by_rust, visits_by_rust, legal_rust)

    assert mask.tolist() == [True, True, True]


def test_full_coverage_requires_every_legal_action_visited() -> None:
    """soft_score_legal_coverage == 1.0 requires every legal action to be BOTH scored and
    visited; this documents that a single unvisited action breaks full coverage."""
    legal_rust = tuple(range(18))
    q_by_rust = {action: 0.0 for action in legal_rust}
    visits_by_rust = {action: 1 for action in legal_rust}
    visits_by_rust[7] = 0  # one root child never got expanded

    _, mask = _target_scores_and_mask(q_by_rust, visits_by_rust, legal_rust)

    assert mask.sum() == 17
    assert not mask[7]
