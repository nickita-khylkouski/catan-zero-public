from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from diag_common import (  # type: ignore  # noqa: E402
    argmax_agreement,
    kl_divergence,
    line_concentration,
    normalized_entropy,
)


def test_kl_divergence_zero_for_identical_dicts():
    p = {0: 0.9, 1: 0.1}
    assert kl_divergence(p, dict(p)) == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_known_value():
    p = {0: 0.9, 1: 0.1}
    q = {0: 0.5, 1: 0.5}
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert kl_divergence(p, q) == pytest.approx(expected, abs=1e-6)


def test_kl_divergence_is_not_symmetric():
    p = {0: 0.9, 1: 0.1}
    q = {0: 0.5, 1: 0.5}
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


def test_kl_divergence_handles_disjoint_keys_without_crashing():
    p = {0: 1.0}
    q = {1: 1.0}
    result = kl_divergence(p, q)
    assert result >= 0.0
    assert math.isfinite(result)


def test_kl_divergence_nonnegative_for_random_like_dicts():
    p = {0: 0.2, 1: 0.3, 2: 0.5}
    q = {0: 0.4, 1: 0.4, 2: 0.2}
    assert kl_divergence(p, q) >= -1e-9


def test_normalized_entropy_uniform_four_way_is_one():
    probs = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    assert normalized_entropy(probs) == pytest.approx(1.0, abs=1e-9)


def test_normalized_entropy_one_hot_is_near_zero():
    probs = {0: 1.0, 1: 0.0, 2: 0.0}
    assert normalized_entropy(probs) == pytest.approx(0.0, abs=1e-6)


def test_normalized_entropy_single_key_is_none():
    assert normalized_entropy({0: 1.0}) is None


def test_normalized_entropy_empty_is_none():
    assert normalized_entropy({}) is None


def test_normalized_entropy_renormalizes_when_not_summing_to_one():
    probs = {0: 2.0, 1: 2.0}  # unnormalized but uniform
    assert normalized_entropy(probs) == pytest.approx(1.0, abs=1e-9)


def test_argmax_agreement_true_when_same_argmax():
    a = {0: 0.1, 1: 0.9}
    b = {0: 0.2, 1: 0.8}
    assert argmax_agreement(a, b) is True


def test_argmax_agreement_false_when_different_argmax():
    a = {0: 0.9, 1: 0.1}
    b = {0: 0.1, 1: 0.9}
    assert argmax_agreement(a, b) is False


def test_argmax_agreement_tie_broken_by_lowest_key():
    a = {0: 0.5, 1: 0.5}
    b = {0: 0.5, 1: 0.5}
    # Both dicts tie-break to key 0; agreement should hold.
    assert argmax_agreement(a, b) is True


def test_line_concentration_basic_counts():
    lines = [
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
    ]
    result = line_concentration(lines)
    assert result["n_games"] == 5
    assert result["n_unique_lines"] == 3
    assert result["top1_fraction"] == pytest.approx(3 / 5)
    # Fewer than 10 distinct lines -> top10_fraction covers all games.
    assert result["top10_fraction"] == pytest.approx(1.0)
    expected_herfindahl = (3 / 5) ** 2 + (1 / 5) ** 2 + (1 / 5) ** 2
    assert result["herfindahl_index"] == pytest.approx(expected_herfindahl)


def test_line_concentration_empty_list():
    result = line_concentration([])
    assert result["n_games"] == 0
    assert result["n_unique_lines"] == 0
    assert result["top1_fraction"] is None
