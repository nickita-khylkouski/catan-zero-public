from __future__ import annotations

import pytest

from catan_zero.rl.policy_pool import PolicySpec, make_policy
from catan_zero.rl.self_play import (
    STYLE_SPECIALIST_WEIGHTS,
    CatanatronValuePolicy,
    CatanatronWeightedRandomPolicy,
)


def test_default_value_policy_uses_contender_fn_no_params() -> None:
    policy = CatanatronValuePolicy()
    assert policy.value_fn_builder_name == "contender_fn"
    assert policy.params is None
    assert policy.name == "catanatron_value"


def test_style_specialist_weight_sets_are_complete_and_distinct() -> None:
    # Every specialist must carry the full key set the value function reads
    # (a missing key would KeyError inside base_fn), and each must differ from
    # the others on at least one weight.
    from catan_zero.rl.self_play import _CONTENDER_BASE_WEIGHTS

    keys = set(_CONTENDER_BASE_WEIGHTS)
    for weights in STYLE_SPECIALIST_WEIGHTS.values():
        assert set(weights) == keys
    seen = [tuple(sorted(w.items())) for w in STYLE_SPECIALIST_WEIGHTS.values()]
    assert len(set(seen)) == len(seen)
    # public_vps must stay dominant so winning always beats any style term.
    for weights in STYLE_SPECIALIST_WEIGHTS.values():
        assert weights["public_vps"] == _CONTENDER_BASE_WEIGHTS["public_vps"]


@pytest.mark.parametrize(
    "kind,style",
    [
        ("catanatron_value_ore_city", "ore_city"),
        ("catanatron_value_road_race", "road_race"),
        ("catanatron_value_robber", "robber"),
    ],
)
def test_make_policy_builds_named_style_specialist(kind: str, style: str) -> None:
    policy = make_policy(PolicySpec(kind=kind))
    assert isinstance(policy, CatanatronValuePolicy)
    assert policy.name == kind
    assert policy.value_fn_builder_name == "base_fn"
    assert policy.params == STYLE_SPECIALIST_WEIGHTS[style]


def test_specialist_signature_matches_intended_style() -> None:
    # Sanity that the reweightings actually encode the intended emphasis.
    road = STYLE_SPECIALIST_WEIGHTS["road_race"]
    ore = STYLE_SPECIALIST_WEIGHTS["ore_city"]
    robber = STYLE_SPECIALIST_WEIGHTS["robber"]
    assert road["longest_road"] > ore["longest_road"]
    assert ore["hand_devs"] > road["hand_devs"]
    assert robber["enemy_production"] < ore["enemy_production"]


@pytest.mark.parametrize("kind", ["weighted_random", "catanatron_weighted_random"])
def test_make_policy_builds_weighted_random_floor(kind: str) -> None:
    policy = make_policy(PolicySpec(kind=kind))
    assert isinstance(policy, CatanatronWeightedRandomPolicy)
    assert policy.name == "catanatron_weighted_random"
    # The weight table must skew toward building (cities > settlements > devs).
    from catanatron.models.actions import ActionType

    weights = policy._weights_by_action_type
    assert weights[ActionType.BUILD_CITY] > weights[ActionType.BUILD_SETTLEMENT]
    assert weights[ActionType.BUILD_SETTLEMENT] > weights[ActionType.BUY_DEVELOPMENT_CARD]


def test_default_gate_roster_entries_are_all_resolvable() -> None:
    # Every DEFAULT_ROSTER entry must be either a known bot kind or a checkpoint
    # path -- a specialist misspelled here would be silently treated as a
    # (missing) checkpoint path by is_bot_kind, so lock the routing down.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import promotion_gate_runner as gate  # type: ignore

    specialists = {
        "catanatron_value_ore_city",
        "catanatron_value_road_race",
        "catanatron_value_robber",
    }
    assert specialists <= set(gate.DEFAULT_ROSTER)
    for name in specialists:
        assert gate.is_bot_kind(name), f"{name} not routed as a bot kind"
