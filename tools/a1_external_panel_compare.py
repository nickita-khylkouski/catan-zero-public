#!/usr/bin/env python3
"""Paired comparison of two matched neutral-harness panel reports.

Candidate-vs-bot and champion-vs-bot panels deliberately reuse the exact same
``(game_seed, orientation)`` cohort.  Treating their two win rates as
independent throws away that design and, more importantly, answers the wrong
uncertainty question.  This module compares the per-game outcomes directly and
uses the paired seed (the two color-swapped games) as the independent cluster.

The exact McNemar result is retained as a familiar diagnostic.  The cluster
interval is the primary uncertainty estimate because the two orientations of
one seed are not independent observations.
"""

from __future__ import annotations

import math
from typing import Any


class ExternalPanelComparisonError(RuntimeError):
    """The two reports do not describe one matched external cohort."""


def _outcomes(report: dict[str, Any], *, where: str) -> dict[tuple[int, str], bool]:
    games = report.get("games")
    if not isinstance(games, list) or not games:
        raise ExternalPanelComparisonError(f"{where} has no retained raw games")
    result: dict[tuple[int, str], bool] = {}
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise ExternalPanelComparisonError(
                f"{where}.games[{index}] is not an object"
            )
        try:
            key = (int(game["game_seed"]), str(game["orientation"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ExternalPanelComparisonError(
                f"{where}.games[{index}] has no seed/orientation identity"
            ) from error
        outcome = game.get("candidate_won")
        # Promotion replay validates the full clean-game contract before it
        # calls us.  Standalone pooled reports retain these fields, so enforce
        # them whenever present while remaining usable on the compact replay
        # fixtures and historical authenticated reports.
        if (
            not isinstance(outcome, bool)
            or ("terminated" in game and game.get("terminated") is not True)
            or game.get("truncated") is True
            or game.get("error") not in {None, ""}
            or bool(game.get("engine_divergence", False))
        ):
            raise ExternalPanelComparisonError(
                f"{where}.games[{index}] is not a complete clean outcome"
            )
        if key in result:
            raise ExternalPanelComparisonError(f"{where} duplicates cohort row {key}")
        result[key] = outcome
    return result


def _exact_mcnemar_p(candidate_only: int, champion_only: int) -> float:
    discordant = candidate_only + champion_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(candidate_only, champion_only) + 1)
    )
    return min(1.0, 2.0 * tail / (1 << discordant))


def compare_matched_external_panels(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    *,
    noninferiority_margin: float = 0.02,
) -> dict[str, Any]:
    """Return paired common-opponent delta and uncertainty diagnostics.

    Each seed contributes the mean of its two color-swapped game differences,
    so the reported standard error is robust to arbitrary within-seed
    correlation.  This is a common-opponent estimand; it must not be presented
    as a replacement for direct candidate-vs-champion H2H strength.
    """

    candidate_outcomes = _outcomes(candidate, where="candidate external panel")
    champion_outcomes = _outcomes(champion, where="champion external panel")
    if set(candidate_outcomes) != set(champion_outcomes):
        missing_candidate = sorted(set(champion_outcomes) - set(candidate_outcomes))[:3]
        missing_champion = sorted(set(candidate_outcomes) - set(champion_outcomes))[:3]
        raise ExternalPanelComparisonError(
            "external panels do not retain the same seed/orientation cohort: "
            f"missing_candidate={missing_candidate}, missing_champion={missing_champion}"
        )

    both_win = both_loss = candidate_only = champion_only = 0
    by_seed: dict[int, list[int]] = {}
    by_orientation: dict[str, list[int]] = {}
    for key in sorted(candidate_outcomes):
        seed, orientation = key
        candidate_won = candidate_outcomes[key]
        champion_won = champion_outcomes[key]
        difference = int(candidate_won) - int(champion_won)
        by_seed.setdefault(seed, []).append(difference)
        by_orientation.setdefault(orientation, []).append(difference)
        if candidate_won and champion_won:
            both_win += 1
        elif candidate_won:
            candidate_only += 1
        elif champion_won:
            champion_only += 1
        else:
            both_loss += 1

    malformed = {
        seed: len(values) for seed, values in by_seed.items() if len(values) != 2
    }
    if malformed:
        raise ExternalPanelComparisonError(
            f"external cohort is not exactly two color-swapped games per seed: {malformed}"
        )
    seed_deltas = [sum(values) / 2.0 for _, values in sorted(by_seed.items())]
    clusters = len(seed_deltas)
    delta = sum(seed_deltas) / clusters
    if clusters > 1:
        variance = sum((value - delta) ** 2 for value in seed_deltas) / (clusters - 1)
        standard_error = math.sqrt(variance / clusters)
    else:
        standard_error = 0.0
    lower = max(-1.0, delta - 1.96 * standard_error)
    upper = min(1.0, delta + 1.96 * standard_error)
    lower_one_sided = max(-1.0, delta - 1.6448536269514722 * standard_error)
    if standard_error > 0.0:
        z_noninferiority = (delta + float(noninferiority_margin)) / standard_error
        noninferiority_p = 0.5 * math.erfc(z_noninferiority / math.sqrt(2.0))
    else:
        z_noninferiority = None
        noninferiority_p = 0.0 if delta >= -float(noninferiority_margin) else 1.0

    return {
        "schema_version": "a1-matched-external-comparison-v1",
        "estimand": "paired common-opponent win-rate delta; not direct H2H strength",
        "games": len(candidate_outcomes),
        "seed_clusters": clusters,
        "candidate_win_rate": sum(candidate_outcomes.values())
        / len(candidate_outcomes),
        "champion_win_rate": sum(champion_outcomes.values()) / len(champion_outcomes),
        "candidate_minus_champion": delta,
        "paired_seed_cluster_standard_error": standard_error,
        "paired_seed_cluster_95ci": [lower, upper],
        "noninferiority": {
            "margin": float(noninferiority_margin),
            "alpha": 0.05,
            "null": "candidate_minus_champion <= -margin",
            "z": z_noninferiority,
            "p": noninferiority_p,
            "one_sided_95_lower": lower_one_sided,
            "passed": bool(lower_one_sided >= -float(noninferiority_margin)),
        },
        "contingency": {
            "both_win": both_win,
            "candidate_only_win": candidate_only,
            "champion_only_win": champion_only,
            "both_loss": both_loss,
            "discordant": candidate_only + champion_only,
        },
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(candidate_only, champion_only),
        "orientation_deltas": {
            orientation: sum(values) / len(values)
            for orientation, values in sorted(by_orientation.items())
        },
        "seed_delta_counts": {
            str(value): seed_deltas.count(value)
            for value in (-1.0, -0.5, 0.0, 0.5, 1.0)
        },
    }


__all__ = [
    "ExternalPanelComparisonError",
    "compare_matched_external_panels",
]
