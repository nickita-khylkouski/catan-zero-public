from types import SimpleNamespace

import numpy as np

from catan_zero.rl.decision_taxonomy import (
    AUTOMATIC_TRANSITION,
    MANDATORY_CHOICE,
    MANDATORY_SEQUENCE_CONTINUATION,
    NORMAL_CHOICE,
    SEARCH_BUDGET_MANDATORY_CONTINUATION_PCR,
    SEARCH_BUDGET_WIDE_FULL,
    WIDE_CHOICE,
    classify_public_decision,
    decision_requires_full_search,
    search_budget_reason,
)
from catan_zero.rl.gumbel_self_play import _build_decision_row


def test_public_decision_taxonomy_matches_coherent_wave_budgeting() -> None:
    assert (
        classify_public_decision({}, legal_action_count=1)
        == AUTOMATIC_TRANSITION
    )
    for prompt in (
        "BUILD_INITIAL_SETTLEMENT",
        "BUILD_INITIAL_ROAD",
        "DISCARD",
        "MOVE_ROBBER",
    ):
        decision = classify_public_decision(
            {"current_prompt": prompt}, legal_action_count=2
        )
        assert decision == MANDATORY_CHOICE
        assert decision_requires_full_search(decision)
    assert (
        classify_public_decision(
            {"current_prompt": "PLAY_TURN", "is_road_building": True},
            legal_action_count=3,
        )
        == MANDATORY_CHOICE
    )
    assert classify_public_decision({}, legal_action_count=20) == WIDE_CHOICE
    assert decision_requires_full_search(WIDE_CHOICE)
    assert (
        search_budget_reason({}, decision_class=WIDE_CHOICE)
        == SEARCH_BUDGET_WIDE_FULL
    )
    assert classify_public_decision({}, legal_action_count=19) == NORMAL_CHOICE


def _discard_snapshot(
    *,
    current_actor: str,
    previous_actor: str,
    previous_value: str = "ORE",
    hidden_resources: dict[str, int] | None = None,
) -> dict:
    return {
        "current_prompt": "DISCARD",
        "current_color": current_actor,
        "action_records": [
            {
                "action": [
                    previous_actor,
                    "DISCARD_RESOURCE",
                    previous_value,
                ],
                "result": previous_value,
            }
        ],
        # Authoritative snapshots contain this field, but v2 budgeting must
        # never inspect it: only the public actor/action occurrence matters.
        "player_state": [{"resources": hidden_resources or {}}],
    }


def test_repeated_same_actor_discard_is_randomized_continuation() -> None:
    snapshot = _discard_snapshot(current_actor="RED", previous_actor="RED")
    decision = classify_public_decision(snapshot, legal_action_count=4)

    assert decision == MANDATORY_SEQUENCE_CONTINUATION
    assert not decision_requires_full_search(decision)
    assert (
        search_budget_reason(snapshot, decision_class=decision)
        == SEARCH_BUDGET_MANDATORY_CONTINUATION_PCR
    )


def test_first_discard_for_next_actor_remains_full_search() -> None:
    snapshot = _discard_snapshot(current_actor="BLUE", previous_actor="RED")
    decision = classify_public_decision(snapshot, legal_action_count=4)

    assert decision == MANDATORY_CHOICE
    assert decision_requires_full_search(decision)


def test_discard_budget_is_invariant_to_hidden_resource_truth() -> None:
    first = _discard_snapshot(
        current_actor="RED",
        previous_actor="RED",
        previous_value="BRICK",
        hidden_resources={"ORE": 9, "WHEAT": 0},
    )
    second = _discard_snapshot(
        current_actor="RED",
        previous_actor="RED",
        previous_value="SHEEP",
        hidden_resources={"ORE": 0, "WHEAT": 9},
    )

    assert classify_public_decision(first, legal_action_count=4) == (
        classify_public_decision(second, legal_action_count=4)
    )


def test_both_public_road_building_placements_remain_full_search() -> None:
    second_placement = {
        "current_prompt": "PLAY_TURN",
        "current_color": "RED",
        "is_road_building": True,
        "action_records": [
            {"action": ["RED", "BUILD_ROAD", [1, 2]], "result": None}
        ],
    }
    decision = classify_public_decision(second_placement, legal_action_count=3)

    assert decision == MANDATORY_CHOICE
    assert decision_requires_full_search(decision)


def test_self_play_row_persists_typed_budget_reason(monkeypatch) -> None:
    import catan_zero.rl.gumbel_self_play as gsp

    snapshot = {"current_prompt": "PLAY_TURN", "current_color": "RED"}
    monkeypatch.setattr(
        gsp,
        "_build_public_learner_features",
        lambda *_args, **_kwargs: (
            (10, 11),
            {},
            np.zeros((2, 1), dtype=np.float32),
            snapshot,
            {1: ["RED", "BUILD_ROAD", [1, 2]], 2: ["RED", "END_TURN", None]},
        ),
    )
    result = SimpleNamespace(
        improved_policy={1: 0.75, 2: 0.25},
        priors={1: 0.5, 2: 0.5},
        q_values={1: 0.2, 2: 0.1},
        afterstate_values={1: 0.2, 2: 0.1},
        completed_q_values={1: 0.2, 2: 0.1},
        visit_counts={1: 96, 2: 32},
        selected_action=1,
        used_full_search=True,
        simulations_used=128,
        root_value=0.15,
        root_prior_value=0.05,
    )
    game = SimpleNamespace(current_color=lambda: "RED")

    row, _features = _build_decision_row(
        game,
        result=result,
        action_size=12,
        colors=("RED", "BLUE"),
        game_seed=7,
        decision_index=3,
        obs_width=4,
        snapshot=snapshot,
        action_by_id={1: ["RED", "BUILD_ROAD", [1, 2]], 2: ["RED", "END_TURN", None]},
        decision_class=WIDE_CHOICE,
        search_budget_reason_value=SEARCH_BUDGET_WIDE_FULL,
    )

    assert row["search_budget_reason"] == SEARCH_BUDGET_WIDE_FULL
    assert row["decision_taxonomy_schema"].endswith("_v2")
