"""Public-only joint belief over an opponent's hidden development cards.

The belief consumes only two sufficient public quantities:

* ``unknown_pool``: remaining counts by development-card type after removing
  the observer's known cards and every publicly played card; and
* ``opponent_count``: the opponent's public face-down development-card count.

It never accepts deck order or hidden card identities.  With zero tilt the
distribution is the exact multivariate-hypergeometric rule prior.  A learned
caller may optionally provide one public-history-derived logit per card type;
this exponentially tilts the rule prior without assigning mass to an
impossible hidden hand.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Mapping, Sequence

import numpy as np

from catan_zero.deduction_tracker import DEV_CARD_TYPES, STARTING_DEV_DECK


DEV_CARD_TYPE_TO_INDEX = {card: index for index, card in enumerate(DEV_CARD_TYPES)}


def _public_pool_vector(unknown_pool: Mapping[str, int]) -> np.ndarray:
    if not isinstance(unknown_pool, Mapping):
        raise TypeError("unknown_pool must be a mapping of public card counts")
    if set(unknown_pool) != set(DEV_CARD_TYPES):
        raise ValueError(
            "unknown_pool must contain exactly the five canonical dev-card types"
        )

    counts: list[int] = []
    for card in DEV_CARD_TYPES:
        raw = unknown_pool[card]
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"unknown_pool[{card!r}] must be an integer count")
        try:
            count = operator.index(raw)
        except TypeError as exc:
            raise ValueError(
                f"unknown_pool[{card!r}] must be an integer count"
            ) from exc
        if count < 0 or count > STARTING_DEV_DECK[card]:
            raise ValueError(
                f"unknown_pool[{card!r}]={count} is outside the public deck bounds "
                f"[0, {STARTING_DEV_DECK[card]}]"
            )
        counts.append(count)
    return np.asarray(counts, dtype=np.int16)


def _public_opponent_count(opponent_count: int, pool_total: int) -> int:
    if isinstance(opponent_count, (bool, np.bool_)):
        raise ValueError("opponent_count must be an integer count")
    try:
        count = operator.index(opponent_count)
    except TypeError as exc:
        raise ValueError("opponent_count must be an integer count") from exc
    if count < 0 or count > pool_total:
        raise ValueError(
            f"opponent_count={count} cannot be drawn from public pool of {pool_total}"
        )
    return count


def enumerate_feasible_hidden_counts(
    unknown_pool: Mapping[str, int], opponent_count: int
) -> np.ndarray:
    """Enumerate every feasible five-type hidden hand in canonical order.

    Returned rows follow ``DEV_CARD_TYPES`` and satisfy both componentwise
    ``0 <= row <= unknown_pool`` and ``row.sum() == opponent_count``.  Invalid
    public conservation raises instead of silently clipping to a different
    game state.
    """

    pool = _public_pool_vector(unknown_pool)
    draws = _public_opponent_count(opponent_count, int(pool.sum()))
    rows = [
        counts
        for counts in np.ndindex(*(int(value) + 1 for value in pool))
        if sum(counts) == draws
    ]
    # Conservation validation guarantees at least one feasible row, including
    # the all-zero row when draws == 0.
    return np.asarray(rows, dtype=np.int16).reshape(-1, len(DEV_CARD_TYPES))


def _tilt_vector(theta: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if theta is None:
        return np.zeros(len(DEV_CARD_TYPES), dtype=np.float64)
    logits = np.asarray(theta, dtype=np.float64)
    if logits.shape != (len(DEV_CARD_TYPES),):
        raise ValueError(
            f"theta must have shape ({len(DEV_CARD_TYPES)},), got {logits.shape}"
        )
    if not bool(np.isfinite(logits).all()):
        raise ValueError("theta must contain only finite logits")
    return logits


@dataclass(frozen=True, slots=True)
class PublicDevBelief:
    """Finite joint posterior over feasible opponent development-card hands."""

    count_vectors: np.ndarray
    probabilities: np.ndarray

    @property
    def posterior_probabilities(self) -> np.ndarray:
        """Normalized posterior mass aligned with ``count_vectors``."""

        return self.probabilities

    @property
    def expected_counts(self) -> np.ndarray:
        """Posterior expected counts in ``DEV_CARD_TYPES`` order."""

        return self.probabilities @ self.count_vectors.astype(np.float64)

    def expected_count(self, card: str) -> float:
        try:
            index = DEV_CARD_TYPE_TO_INDEX[card]
        except KeyError as exc:
            raise ValueError(f"unknown development-card type: {card!r}") from exc
        return float(self.expected_counts[index])

    def probability_at_least_one(self, card: str) -> float:
        try:
            index = DEV_CARD_TYPE_TO_INDEX[card]
        except KeyError as exc:
            raise ValueError(f"unknown development-card type: {card!r}") from exc
        return float(self.probabilities[self.count_vectors[:, index] > 0].sum())

    @property
    def victory_point_probability(self) -> float:
        return self.probability_at_least_one("VICTORY_POINT")

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Sample feasible hidden count vectors using the supplied RNG only."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a supplied numpy.random.Generator")
        indices = rng.choice(
            self.count_vectors.shape[0], size=size, p=self.probabilities
        )
        return np.array(self.count_vectors[indices], copy=True)


def build_public_dev_belief(
    unknown_pool: Mapping[str, int],
    opponent_count: int,
    theta: Sequence[float] | np.ndarray | None = None,
) -> PublicDevBelief:
    """Build the exact rule prior, optionally tilted by public-history logits.

    For feasible hidden count vector ``c`` and public pool ``u``:

    ``P(c) proportional to product_i comb(u_i, c_i) * exp(theta dot c)``.

    Setting ``theta`` to zero therefore reproduces the normalized joint
    multivariate-hypergeometric prior exactly (up to floating-point storage).
    """

    pool = _public_pool_vector(unknown_pool)
    draws = _public_opponent_count(opponent_count, int(pool.sum()))
    support = enumerate_feasible_hidden_counts(unknown_pool, draws)
    logits = _tilt_vector(theta)

    # The combinatorial numerators are small for Catan's 25-card deck. Compute
    # them exactly as Python integers, then normalize in log space so arbitrary
    # finite learned tilts remain numerically stable.
    log_weights = np.empty(support.shape[0], dtype=np.float64)
    for row_index, counts in enumerate(support):
        numerator = math.prod(
            math.comb(int(available), int(held))
            for available, held in zip(pool, counts, strict=True)
        )
        log_weights[row_index] = math.log(numerator) + float(counts @ logits)
    log_weights -= float(log_weights.max())
    weights = np.exp(log_weights)
    normalizer = float(weights.sum())
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("public dev-card posterior failed to normalize")
    probabilities = weights / normalizer
    return PublicDevBelief(
        count_vectors=support,
        probabilities=probabilities,
    )


__all__ = [
    "DEV_CARD_TYPES",
    "PublicDevBelief",
    "build_public_dev_belief",
    "enumerate_feasible_hidden_counts",
]
