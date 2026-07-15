from catan_zero.rl.decision_taxonomy import (
    AUTOMATIC_TRANSITION,
    MANDATORY_CHOICE,
    NORMAL_CHOICE,
    WIDE_CHOICE,
    classify_public_decision,
    decision_requires_full_search,
)


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
    assert classify_public_decision({}, legal_action_count=19) == NORMAL_CHOICE
