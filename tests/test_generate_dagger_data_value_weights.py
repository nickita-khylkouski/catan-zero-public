from __future__ import annotations

from tools.generate_dagger_data import (
    _canonical_episode_status,
    _effective_value_weight_multiplier,
)


def test_completed_games_use_value_weight_multiplier() -> None:
    assert _effective_value_weight_multiplier(
        truncated=False,
        value_weight_multiplier=1.0,
        truncated_value_weight=0.0,
    ) == 1.0


def test_truncated_games_use_truncated_value_weight_not_value_weight_multiplier() -> None:
    """FIX A6: truncated rows must NOT silently inherit --value-weight-multiplier."""
    assert _effective_value_weight_multiplier(
        truncated=True,
        value_weight_multiplier=1.0,
        truncated_value_weight=0.0,
    ) == 0.0


def test_effective_value_weight_multiplier_respects_custom_values() -> None:
    assert _effective_value_weight_multiplier(
        truncated=True,
        value_weight_multiplier=1.0,
        truncated_value_weight=0.3,
    ) == 0.3
    assert _effective_value_weight_multiplier(
        truncated=False,
        value_weight_multiplier=0.7,
        truncated_value_weight=0.3,
    ) == 0.7


def test_terminal_outcome_takes_precedence_over_simultaneous_truncation() -> None:
    assert _canonical_episode_status(terminated=True, truncated=True) == (
        True,
        False,
    )


def test_nonterminal_truncation_is_preserved() -> None:
    assert _canonical_episode_status(terminated=False, truncated=True) == (
        False,
        True,
    )
