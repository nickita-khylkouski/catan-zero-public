"""Public decision classes for the two-player no-trade search teacher.

The engine exposes several UI-shaped prompts that are not decisions at all.
When there is exactly one legal transition, the transition should be applied
directly: there is no policy simplex to improve and no reason to invoke the
network or MCTS.  Conversely, compulsory prompts with multiple public choices
are unusually consequential and always receive the full teacher budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DECISION_TAXONOMY_SCHEMA_VERSION = "public_decision_taxonomy_2p_no_trade_v1"

AUTOMATIC_TRANSITION = "automatic_transition"
MANDATORY_CHOICE = "mandatory_choice"
WIDE_CHOICE = "wide_choice"
NORMAL_CHOICE = "normal_choice"

MANDATORY_PUBLIC_PROMPTS = frozenset(
    {
        "BUILD_INITIAL_SETTLEMENT",
        "BUILD_INITIAL_ROAD",
        "DISCARD",
        "MOVE_ROBBER",
    }
)


def classify_public_decision(
    snapshot: Mapping[str, Any] | None,
    *,
    legal_action_count: int,
    wide_threshold: int = 20,
) -> str:
    """Classify a root using public prompt state only.

    ``is_road_building`` is public because the Road Building card has already
    been revealed.  We deliberately do not inspect hand-dependent action
    identities or authoritative hidden state to choose a search budget.
    """

    width = int(legal_action_count)
    threshold = int(wide_threshold)
    if width < 1:
        raise ValueError("decision taxonomy requires at least one legal action")
    if threshold < 2:
        raise ValueError("wide decision threshold must be at least 2")
    if width == 1:
        return AUTOMATIC_TRANSITION

    public = snapshot if isinstance(snapshot, Mapping) else {}
    prompt = str(public.get("current_prompt", ""))
    if prompt in MANDATORY_PUBLIC_PROMPTS or bool(public.get("is_road_building", False)):
        return MANDATORY_CHOICE
    if width >= threshold:
        return WIDE_CHOICE
    return NORMAL_CHOICE


def decision_requires_full_search(decision_class: str) -> bool:
    """Whether the public class overrides playout-cap randomization."""

    return str(decision_class) == MANDATORY_CHOICE
