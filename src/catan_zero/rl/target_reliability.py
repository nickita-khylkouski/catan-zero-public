"""Typed duplicate-search evidence for policy-target reliability.

This module is intentionally independent of the self-play driver and trainer:
both sides consume the same schema, selector, and confidence formula without
reimplementing the math.  A duplicate search is diagnostic only; its selected
action must never be applied to the live game.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from catan_zero.search.rng_streams import SEARCH_RNG_STREAM_SCHEMA


TARGET_RELIABILITY_SCHEMA = "coherent_n128_duplicate_search_reliability_v1"
TARGET_RELIABILITY_VERSION = 1
TARGET_RELIABILITY_SELECTOR_SCHEMA = "coherent-n128-root-audit-selector-v1"
TARGET_RELIABILITY_ROOT_SEED_SCHEMA = "coherent-n128-root-audit-seed-v1"
TARGET_RELIABILITY_CONFIDENCE_FORMULA = (
    "(1-js_divergence/ln(2))*policy_top1_agreement"
)
TARGET_RELIABILITY_COLUMNS = (
    "target_reliability_version",
    "target_reliability_audited",
    "target_reliability_js_divergence",
    "target_reliability_policy_top1_agreement",
    "target_reliability_q_top1_agreement",
    "target_reliability_q_margin_primary",
    "target_reliability_q_margin_duplicate",
    "target_reliability_confidence",
)


def _normalized_policy(policy: Mapping[int, float]) -> dict[int, float]:
    if not policy:
        raise ValueError("target reliability requires a non-empty policy")
    values: dict[int, float] = {}
    for action, raw in policy.items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "target reliability policy probabilities must be finite and "
                f"non-negative; action={action!r} value={raw!r}"
            )
        values[int(action)] = value
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("target reliability policy must have positive finite mass")
    return {action: value / total for action, value in values.items()}


def jensen_shannon_divergence(
    first: Mapping[int, float],
    second: Mapping[int, float],
    *,
    eps: float = 1.0e-12,
) -> float:
    """Symmetric Jensen-Shannon divergence in nats over action union."""

    left = _normalized_policy(first)
    right = _normalized_policy(second)
    support = set(left) | set(right)
    midpoint = {
        action: 0.5 * (left.get(action, 0.0) + right.get(action, 0.0))
        for action in support
    }

    def _kl(policy: Mapping[int, float]) -> float:
        return sum(
            probability
            * math.log((probability + float(eps)) / (midpoint[action] + float(eps)))
            for action, probability in policy.items()
            if probability > 0.0
        )

    return min(math.log(2.0), max(0.0, 0.5 * (_kl(left) + _kl(right))))


def _top1(values: Mapping[int, float]) -> int:
    if not values:
        raise ValueError("target reliability requires non-empty action evidence")
    normalized: dict[int, float] = {}
    for action, raw in values.items():
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(
                "target reliability action evidence must be finite; "
                f"action={action!r} value={raw!r}"
            )
        normalized[int(action)] = value
    return int(max(normalized, key=lambda action: (normalized[action], -action)))


def completed_q_top_margin(values: Mapping[int, float]) -> float:
    """Return the non-negative gap between the two largest completed-Q values."""

    if not values:
        raise ValueError("completed-Q evidence must not be empty")
    ordered = sorted((float(value) for value in values.values()), reverse=True)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("completed-Q evidence must be finite")
    return max(0.0, ordered[0] - ordered[1]) if len(ordered) >= 2 else 0.0


def target_reliability_confidence(
    js_divergence: float, *, policy_top1_agreement: bool
) -> float:
    """Scale-free policy reliability score in ``[0, 1]``.

    Q margins are deliberately persisted as evidence but are not folded into
    this score: their useful scale is phase/operator dependent, and inventing a
    global margin threshold would turn an audit field into an uncalibrated
    learner policy.  The trainer owns an explicit confidence floor.
    """

    divergence = float(js_divergence)
    if not math.isfinite(divergence) or not 0.0 <= divergence <= math.log(2.0) + 1e-9:
        raise ValueError(
            "Jensen-Shannon divergence must be finite and in [0, ln(2)]"
        )
    stability = max(0.0, min(1.0, 1.0 - divergence / math.log(2.0)))
    return stability if bool(policy_top1_agreement) else 0.0


def unaudited_target_reliability_fields() -> dict[str, Any]:
    """Neutral, typed fields for a row outside the deterministic audit slice."""

    return {
        "target_reliability_version": np.uint8(TARGET_RELIABILITY_VERSION),
        "target_reliability_audited": np.bool_(False),
        "target_reliability_js_divergence": np.float32(np.nan),
        "target_reliability_policy_top1_agreement": np.bool_(False),
        "target_reliability_q_top1_agreement": np.bool_(False),
        "target_reliability_q_margin_primary": np.float32(np.nan),
        "target_reliability_q_margin_duplicate": np.float32(np.nan),
        # Unobserved reliability is neutral in the learner, never zero weight.
        "target_reliability_confidence": np.float32(1.0),
    }


def duplicate_search_reliability_fields(
    *,
    primary_policy: Mapping[int, float],
    duplicate_policy: Mapping[int, float],
    primary_completed_q: Mapping[int, float],
    duplicate_completed_q: Mapping[int, float],
) -> dict[str, Any]:
    """Build one versioned row record from two independent coherent searches."""

    supports = (
        set(map(int, primary_policy)),
        set(map(int, duplicate_policy)),
        set(map(int, primary_completed_q)),
        set(map(int, duplicate_completed_q)),
    )
    if any(support != supports[0] for support in supports[1:]):
        raise ValueError("duplicate-search action supports differ")
    js = jensen_shannon_divergence(primary_policy, duplicate_policy)
    policy_agreement = _top1(primary_policy) == _top1(duplicate_policy)
    q_agreement = _top1(primary_completed_q) == _top1(duplicate_completed_q)
    return {
        "target_reliability_version": np.uint8(TARGET_RELIABILITY_VERSION),
        "target_reliability_audited": np.bool_(True),
        "target_reliability_js_divergence": np.float32(js),
        "target_reliability_policy_top1_agreement": np.bool_(policy_agreement),
        "target_reliability_q_top1_agreement": np.bool_(q_agreement),
        "target_reliability_q_margin_primary": np.float32(
            completed_q_top_margin(primary_completed_q)
        ),
        "target_reliability_q_margin_duplicate": np.float32(
            completed_q_top_margin(duplicate_completed_q)
        ),
        "target_reliability_confidence": np.float32(
            target_reliability_confidence(
                js, policy_top1_agreement=policy_agreement
            )
        ),
    }


def _hash_u64(schema: str, *values: int) -> int:
    payload = ":".join((schema, *(str(int(value)) for value in values))).encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def target_reliability_root_selected(
    *,
    game_seed: int,
    decision_index: int,
    audit_seed: int,
    audit_fraction: float,
) -> bool:
    """Select a stable root slice without consuming any gameplay/search RNG."""

    fraction = float(audit_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("target reliability audit fraction must be in [0, 1]")
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    draw = _hash_u64(
        TARGET_RELIABILITY_SELECTOR_SCHEMA,
        audit_seed,
        game_seed,
        decision_index,
    )
    return draw < int(fraction * (1 << 64))


def target_reliability_root_seed(
    *, game_seed: int, decision_index: int, audit_seed: int
) -> int:
    """Derive the duplicate search's base seed independently per audited root."""

    return _hash_u64(
        TARGET_RELIABILITY_ROOT_SEED_SCHEMA,
        audit_seed,
        game_seed,
        decision_index,
    )


def target_reliability_contract(
    *, audit_fraction: float, audit_seed: int
) -> dict[str, Any] | None:
    fraction = float(audit_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("target reliability audit fraction must be in [0, 1]")
    if fraction == 0.0:
        return None
    return {
        "schema_version": TARGET_RELIABILITY_SCHEMA,
        "version": TARGET_RELIABILITY_VERSION,
        "eligible_roots": "recorded_policy_active_coherent_exact_n128",
        "audit_fraction": fraction,
        "audit_seed": int(audit_seed),
        "selector_schema": TARGET_RELIABILITY_SELECTOR_SCHEMA,
        "duplicate_root_seed_schema": TARGET_RELIABILITY_ROOT_SEED_SCHEMA,
        "rng_stream_schema": SEARCH_RNG_STREAM_SCHEMA,
        "rng_streams": ["gumbel", "chance", "belief"],
        "confidence_formula": TARGET_RELIABILITY_CONFIDENCE_FORMULA,
        "duplicate_selected_action_applied": False,
        "columns": list(TARGET_RELIABILITY_COLUMNS),
    }
