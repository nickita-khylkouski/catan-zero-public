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


DECISION_TAXONOMY_SCHEMA_VERSION = "public_decision_taxonomy_2p_no_trade_v2"

AUTOMATIC_TRANSITION = "automatic_transition"
MANDATORY_SEQUENCE_START = "mandatory_sequence_start"
MANDATORY_SEQUENCE_CONTINUATION = "mandatory_sequence_continuation"
# Compatibility import for callers that used the v1 name. New rows use the
# more precise v2 value above.
MANDATORY_CHOICE = MANDATORY_SEQUENCE_START
WIDE_CHOICE = "wide_choice"
NORMAL_CHOICE = "normal_choice"
WIDE_CHOICE_MIN_LEGAL_ACTIONS = 20

OPENING_PUBLIC_PROMPTS = frozenset(
    {
        "BUILD_INITIAL_SETTLEMENT",
        "BUILD_INITIAL_ROAD",
    }
)

SEARCH_BUDGET_AUTOMATIC = "automatic_transition_no_search"
SEARCH_BUDGET_OPENING_FULL = "opening_full_search"
SEARCH_BUDGET_ROBBER_FULL = "robber_full_search"
SEARCH_BUDGET_MANDATORY_START_FULL = "mandatory_sequence_start_full_search"
SEARCH_BUDGET_WIDE_FULL = "wide_choice_full_search"
SEARCH_BUDGET_MANDATORY_CONTINUATION_PCR = (
    "mandatory_sequence_continuation_playout_cap_randomization"
)
SEARCH_BUDGET_NORMAL_PCR = "normal_choice_playout_cap_randomization"
SEARCH_BUDGET_EVALUATION_FULL = "evaluation_override_full_search"


def _last_public_action(snapshot: Mapping[str, Any]) -> tuple[str, str] | None:
    records = snapshot.get("action_records")
    if not isinstance(records, (list, tuple)) or not records:
        return None
    record = records[-1]
    if not isinstance(record, Mapping):
        return None
    action = record.get("action")
    if not isinstance(action, (list, tuple)) or len(action) < 2:
        return None
    return str(action[0]), str(action[1])


def _is_same_actor_continuation(
    snapshot: Mapping[str, Any], *, previous_action_type: str
) -> bool:
    previous = _last_public_action(snapshot)
    if previous is None:
        return False
    previous_actor, previous_type = previous
    return (
        previous_actor == str(snapshot.get("current_color", ""))
        and previous_type == str(previous_action_type)
    )


def classify_public_decision(
    snapshot: Mapping[str, Any] | None,
    *,
    legal_action_count: int,
    wide_threshold: int = WIDE_CHOICE_MIN_LEGAL_ACTIONS,
) -> str:
    """Classify a root using public prompt state only.

    ``is_road_building`` and the actor/action type of prior public actions are
    public because the Road Building card and each discard occurrence have
    already been revealed. We deliberately do not inspect resource identities,
    hand-dependent legal-action identities, or authoritative hidden state to
    choose a search budget.
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
    if prompt in OPENING_PUBLIC_PROMPTS or prompt == "MOVE_ROBBER":
        return MANDATORY_SEQUENCE_START
    if prompt == "DISCARD":
        if _is_same_actor_continuation(
            public, previous_action_type="DISCARD_RESOURCE"
        ):
            return MANDATORY_SEQUENCE_CONTINUATION
        return MANDATORY_SEQUENCE_START
    if bool(public.get("is_road_building", False)):
        return MANDATORY_SEQUENCE_START
    if width >= threshold:
        return WIDE_CHOICE
    return NORMAL_CHOICE


def decision_requires_full_search(decision_class: str) -> bool:
    """Whether the public class overrides playout-cap randomization."""

    return str(decision_class) in {MANDATORY_SEQUENCE_START, WIDE_CHOICE}


def search_budget_reason(
    snapshot: Mapping[str, Any] | None,
    *,
    decision_class: str,
    eval_override: bool = False,
) -> str:
    """Return the typed public reason for the root's search-budget route."""

    public = snapshot if isinstance(snapshot, Mapping) else {}
    classified = str(decision_class)
    if classified == AUTOMATIC_TRANSITION:
        return SEARCH_BUDGET_AUTOMATIC
    if eval_override:
        return SEARCH_BUDGET_EVALUATION_FULL
    if classified == WIDE_CHOICE:
        return SEARCH_BUDGET_WIDE_FULL
    if classified == MANDATORY_SEQUENCE_CONTINUATION:
        return SEARCH_BUDGET_MANDATORY_CONTINUATION_PCR
    if classified == MANDATORY_SEQUENCE_START:
        prompt = str(public.get("current_prompt", ""))
        if prompt in OPENING_PUBLIC_PROMPTS:
            return SEARCH_BUDGET_OPENING_FULL
        if prompt == "MOVE_ROBBER":
            return SEARCH_BUDGET_ROBBER_FULL
        return SEARCH_BUDGET_MANDATORY_START_FULL
    if classified == NORMAL_CHOICE:
        return SEARCH_BUDGET_NORMAL_PCR
    raise ValueError(f"unsupported public decision class: {decision_class!r}")
