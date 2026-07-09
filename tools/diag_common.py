"""Shared pure-math helpers for CAT-25 plateau-mechanism diagnostics.

No GPU/network/rust dependency at all -- stdlib + numpy only, so every
function here is directly unit-testable. Used by tools/search_snr_probe.py
(measurement 1) and tools/corpus_diversity_scan.py (measurement 3).
"""

from __future__ import annotations

import math
from typing import Any


def kl_divergence(p: dict[int, float], q: dict[int, float], *, eps: float = 1e-12) -> float:
    """KL(p‖q) = sum_{k in union(p,q)} p[k] * log((p[k]+eps)/(q[k]+eps)).

    A key missing from either dict is treated as probability 0 for that dict.
    NOT SYMMETRIC: kl_divergence(p, q) != kl_divergence(q, p) in general --
    callers that want a symmetric comparison should compute (and typically
    average) both directions explicitly, as tools/search_snr_probe.py does.
    Non-negative up to eps-induced float slop; 0.0 when p == q.
    """
    keys = set(p) | set(q)
    total = 0.0
    for key in keys:
        pk = float(p.get(key, 0.0))
        qk = float(q.get(key, 0.0))
        if pk <= 0.0:
            continue
        total += pk * math.log((pk + eps) / (qk + eps))
    return total


def normalized_entropy(probs: dict[int, float], *, eps: float = 1e-12) -> float | None:
    """Shannon entropy of `probs` (renormalized if it doesn't sum to ~1),
    divided by log(len(probs)) so uniform -> 1.0 and one-hot -> 0.0.

    Returns None when len(probs) <= 1 (max entropy is degenerate/zero, so
    the ratio is undefined) -- mirrors the None-for-degenerate convention
    used by tools/opening_panel.py's `_kendall_tau_b` and `_mean` helpers.
    """
    n = len(probs)
    if n <= 1:
        return None
    values = [max(0.0, float(v)) for v in probs.values()]
    total = sum(values)
    if total <= 0.0:
        return None
    normalized = [v / total for v in values]
    entropy = -sum(v * math.log(v + eps) for v in normalized if v > 0.0)
    max_entropy = math.log(n)
    if max_entropy <= 0.0:
        return None
    return entropy / max_entropy


def argmax_agreement(a: dict[int, float], b: dict[int, float]) -> bool:
    """True iff argmax(a) == argmax(b), ties broken by lowest key -- matches
    the `max(..., key=lambda k: (value, -k))` convention used elsewhere in
    tools/ (e.g. `_select_raw_action` in tools/gumbel_search_vs_raw_h2h.py)."""
    argmax_a = max(a, key=lambda k: (float(a[k]), -int(k)))
    argmax_b = max(b, key=lambda k: (float(b[k]), -int(k)))
    return argmax_a == argmax_b


def line_concentration(lines: list[tuple[int, ...]]) -> dict[str, Any]:
    """Concentration statistics over a list of "opening line" signatures
    (one tuple per game).

    Returns n_games, n_unique_lines, top1_fraction (fraction of games whose
    line is the single most frequent line), top10_fraction (fraction of
    games whose line is among the 10 most frequent lines, or all distinct
    lines if fewer than 10 exist), and herfindahl_index = sum((count_i/N)**2)
    over distinct lines.
    """
    n_games = len(lines)
    counts: dict[tuple[int, ...], int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    n_unique = len(counts)
    if n_games == 0:
        return {
            "n_games": 0,
            "n_unique_lines": 0,
            "top1_fraction": None,
            "top10_fraction": None,
            "herfindahl_index": None,
        }
    ordered_counts = sorted(counts.values(), reverse=True)
    top1_fraction = ordered_counts[0] / n_games
    top10_fraction = sum(ordered_counts[:10]) / n_games
    herfindahl_index = sum((c / n_games) ** 2 for c in ordered_counts)
    return {
        "n_games": n_games,
        "n_unique_lines": n_unique,
        "top1_fraction": top1_fraction,
        "top10_fraction": top10_fraction,
        "herfindahl_index": herfindahl_index,
    }
