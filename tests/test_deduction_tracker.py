from __future__ import annotations

import random

import numpy as np
import pytest

from catan_zero.deduction_tracker import (
    DEDUCTION_FEATURE_SIZE,
    DeductionTracker,
    ResourceBounds,
    RESOURCES,
    STARTING_DEV_DECK,
    _hypergeom_at_least_one,
    _hypergeom_pmf,
    compute_roll_production,
    true_state_label,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv


def _zero_hand() -> dict[str, int]:
    return {r: 0 for r in RESOURCES}


def _players_payload(**per_player: dict) -> dict:
    """Build a minimal `players` sub-dict. Each kwarg value may supply
    `resource_card_count`, `resources` (exact, only for the acting
    perspective), `development_card_count`, `development_cards`,
    `played_development_cards`."""
    players = {}
    for name, fields in per_player.items():
        entry = {
            "resource_card_count": fields.get(
                "resource_card_count",
                sum(fields.get("resources", {}).values()) if "resources" in fields else 0,
            ),
            "development_card_count": fields.get(
                "development_card_count",
                sum(fields.get("development_cards", {}).values())
                if "development_cards" in fields
                else 0,
            ),
            "played_development_cards": fields.get("played_development_cards", {}),
        }
        if "resources" in fields:
            entry["resources"] = fields["resources"]
        if "development_cards" in fields:
            entry["development_cards"] = fields["development_cards"]
        players[name] = entry
    return players


def _frame(
    event: dict | None,
    players: dict,
    board: dict | None = None,
    bank: dict | None = None,
    trade_panel: dict | None = None,
) -> dict:
    return {
        "event": event or {"event_type": "reset"},
        "observations": {
            "SELF": {
                "players": players,
                "board": board or {"tiles": (), "buildings": ()},
                "bank": bank or {"resources": {r: 19 for r in RESOURCES}},
                "trade_panel": trade_panel or {"current_board_trade": None},
            }
        },
    }


def _board_action(actor: str, action_type: str, value=None, result=None) -> dict:
    return {
        "event_type": "board_action",
        "actor": actor,
        "payload": {"action": {"action_type": action_type, "value": value}, "result": result},
    }


def _grant_frame(hand_before: dict, actor: str, other_name: str, other_hand: dict, vector: dict) -> tuple[dict, dict]:
    """Build a frame that grants `vector` to `actor` via a YEAR_OF_PLENTY-
    shaped event (a convenient generic "exact gain" mechanism for test setup
    -- the tracker doesn't check affordability/legality, only event shape).
    Returns (frame, new_hand) so callers can chain multiple grants."""
    picks: list[str] = []
    for resource, amount in vector.items():
        picks.extend([resource] * amount)
    new_hand = dict(hand_before)
    for resource, amount in vector.items():
        new_hand[resource] = new_hand.get(resource, 0) + amount
    frame = _frame(
        _board_action(actor, "PLAY_YEAR_OF_PLENTY", value=tuple(picks)),
        _players_payload(**{actor: {"resources": new_hand}, other_name: {"resources": other_hand}}),
    )
    return frame, new_hand


# --------------------------------------------------------------------------
# Hypergeometric math
# --------------------------------------------------------------------------


def test_hypergeom_pmf_sums_to_one_and_matches_manual_case():
    pmf = _hypergeom_pmf(population=25, successes=5, draws=7)
    assert pytest.approx(sum(pmf.values()), rel=1e-9) == 1.0
    # Mean of a hypergeometric is n*K/N.
    mean = sum(k * p for k, p in pmf.items())
    assert pytest.approx(mean, rel=1e-9) == 7 * 5 / 25


def test_hypergeom_degenerate_population_forces_certainty():
    # If the "unknown pool" is entirely held by the opponent (deck exhausted,
    # nobody else holds any), their exact composition is certain.
    pmf = _hypergeom_pmf(population=5, successes=2, draws=5)
    assert pmf == {2: 1.0}


def test_hypergeom_at_least_one_matches_pmf_tail():
    at_least_one = _hypergeom_at_least_one(population=25, successes=5, draws=7)
    pmf = _hypergeom_pmf(population=25, successes=5, draws=7)
    tail = sum(p for k, p in pmf.items() if k >= 1)
    assert pytest.approx(at_least_one, rel=1e-9) == tail


def test_hypergeom_guaranteed_when_draws_exceed_failures():
    # 25 total, 5 successes -> 20 failures. Drawing 21 forces >=1 success.
    assert _hypergeom_at_least_one(population=25, successes=5, draws=21) == 1.0


# --------------------------------------------------------------------------
# Dev-card posterior (stateless, computed from a public snapshot)
# --------------------------------------------------------------------------


def test_dev_card_posterior_matches_full_deck_at_game_start():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    payload = {
        "players": _players_payload(
            SELF={"development_cards": {c: 0 for c in STARTING_DEV_DECK}},
            OPP={"development_card_count": 0},
        )
    }
    posterior = tracker.dev_card_posterior_for("OPP", payload)
    assert posterior.unknown_pool == STARTING_DEV_DECK
    assert posterior.opponent_hidden_count == 0
    assert posterior.victory_point_probability() == 0.0


def test_dev_card_posterior_excludes_played_and_self_held_cards():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    self_dev = {c: 0 for c in STARTING_DEV_DECK}
    self_dev["KNIGHT"] = 2
    payload = {
        "players": _players_payload(
            SELF={"development_cards": self_dev},
            OPP={
                "development_card_count": 3,
                "played_development_cards": {"KNIGHT": 1, "MONOPOLY": 1},
            },
        )
    }
    posterior = tracker.dev_card_posterior_for("OPP", payload)
    # 14 - 1 played - 2 self-held = 11 knights left unknown.
    assert posterior.unknown_pool["KNIGHT"] == 11
    assert posterior.unknown_pool["MONOPOLY"] == 1  # 2 - 1 played
    assert posterior.unknown_pool["VICTORY_POINT"] == 5
    assert posterior.opponent_hidden_count == 3


def test_dev_card_posterior_full_certainty_when_pool_equals_opponent_hand():
    # If every VP card is unaccounted for except by this one opponent, and
    # the opponent's hidden count equals the entire remaining pool, their
    # exact composition (including VP count) is fully determined.
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    self_dev = {c: 0 for c in STARTING_DEV_DECK}
    payload = {
        "players": _players_payload(
            SELF={"development_cards": self_dev},
            OPP={
                "development_card_count": STARTING_DEV_DECK["VICTORY_POINT"],
                "played_development_cards": {
                    "KNIGHT": 14,
                    "YEAR_OF_PLENTY": 2,
                    "MONOPOLY": 2,
                    "ROAD_BUILDING": 2,
                },
            },
        )
    }
    posterior = tracker.dev_card_posterior_for("OPP", payload)
    assert posterior.pool_total == STARTING_DEV_DECK["VICTORY_POINT"]
    assert posterior.victory_point_probability() == 1.0
    assert posterior.expected_count("VICTORY_POINT") == STARTING_DEV_DECK["VICTORY_POINT"]


# --------------------------------------------------------------------------
# Resource running-count fold: scripted event sequences w/ known ground truth
# --------------------------------------------------------------------------


def test_build_road_paid_debits_exact_recipe():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    baseline = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": _zero_hand()}))
    grant, opp_hand = _grant_frame(_zero_hand(), "OPP", "SELF", _zero_hand(), {"WOOD": 2, "BRICK": 2})
    after = _frame(
        _board_action("OPP", "BUILD_ROAD", value=(0, 1)),
        _players_payload(OPP={"resources": {**opp_hand, "WOOD": 1, "BRICK": 1}}, SELF={"resources": _zero_hand()}),
    )
    tracker.observe_frames([baseline, grant, after])
    bounds = tracker.bounds_for("OPP")
    assert bounds.exact() == {"WOOD": 1, "BRICK": 1, "SHEEP": 0, "WHEAT": 0, "ORE": 0}


def test_build_road_free_during_setup_causes_no_change():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    hand = {**_zero_hand(), "WOOD": 0, "BRICK": 0}
    before = _frame(None, _players_payload(OPP={"resources": hand}, SELF={"resources": _zero_hand()}))
    after = _frame(
        _board_action("OPP", "BUILD_ROAD", value=(2, 3)),
        _players_payload(OPP={"resources": hand}, SELF={"resources": _zero_hand()}),
    )
    tracker.observe_frames([before, after])
    assert tracker.bounds_for("OPP").exact() == hand


def test_build_settlement_second_initial_yields_exact_adjacent_resources():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    board = {
        "tiles": (
            {"resource": "WHEAT", "number": 8, "has_robber": False, "nodes": {"N": 5, "E": 6, "SE": 7, "SW": 8, "W": 9, "NW": 10}},
            {"resource": "ORE", "number": 4, "has_robber": False, "nodes": {"N": 11, "E": 12, "SE": 5, "SW": 13, "W": 14, "NW": 15}},
        ),
        "buildings": (),
    }
    before = _frame(
        None,
        _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": _zero_hand()}),
        board=board,
    )
    after_hand = {**_zero_hand(), "WHEAT": 1, "ORE": 1}
    after = _frame(
        _board_action("OPP", "BUILD_SETTLEMENT", value=5),
        _players_payload(OPP={"resources": after_hand}, SELF={"resources": _zero_hand()}),
        board=board,
    )
    tracker.observe_frames([before, after])
    assert tracker.bounds_for("OPP").exact() == after_hand
    assert tracker.anomalies == []


def test_monopoly_resolves_victim_to_exact_zero_and_credits_actor():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    before = _frame(
        None,
        _players_payload(
            OPP={"resources": {**_zero_hand(), "SHEEP": 3}},
            SELF={"resources": {**_zero_hand(), "SHEEP": 2}},
        ),
    )
    after = _frame(
        _board_action("SELF", "PLAY_MONOPOLY", value="SHEEP"),
        _players_payload(
            OPP={"resources": {**_zero_hand(), "SHEEP": 0}},
            SELF={"resources": {**_zero_hand(), "SHEEP": 5}},
        ),
    )
    tracker.observe_frames([before, after])
    bounds = tracker.bounds_for("OPP")
    assert bounds.lower["SHEEP"] == 0 and bounds.upper["SHEEP"] == 0


def test_monopoly_by_opponent_learns_actor_gain_from_our_own_loss():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    self_hand = {**_zero_hand(), "ORE": 4}
    baseline = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": self_hand}))
    grant, opp_hand = _grant_frame(_zero_hand(), "OPP", "SELF", self_hand, {"ORE": 1})
    after = _frame(
        _board_action("OPP", "PLAY_MONOPOLY", value="ORE"),
        _players_payload(
            OPP={"resources": {**opp_hand, "ORE": opp_hand["ORE"] + 4}},
            SELF={"resources": {**_zero_hand(), "ORE": 0}},
        ),
    )
    tracker.observe_frames([baseline, grant, after])
    # OPP's established prior (1, via the grant) plus our own known contribution (4).
    assert tracker.bounds_for("OPP").exact()["ORE"] == 1 + 4


def test_year_of_plenty_credits_exact_picks():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    before = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": _zero_hand()}))
    after = _frame(
        _board_action("OPP", "PLAY_YEAR_OF_PLENTY", value=("WOOD", "WOOD")),
        _players_payload(OPP={"resources": {**_zero_hand(), "WOOD": 2}}, SELF={"resources": _zero_hand()}),
    )
    tracker.observe_frames([before, after])
    assert tracker.bounds_for("OPP").exact()["WOOD"] == 2


def test_maritime_trade_debits_and_credits_exactly():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    baseline = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": _zero_hand()}))
    grant, opp_hand = _grant_frame(_zero_hand(), "OPP", "SELF", _zero_hand(), {"WOOD": 4})
    after = _frame(
        _board_action("OPP", "MARITIME_TRADE", value=("WOOD", "WOOD", "WOOD", "WOOD", "ORE")),
        _players_payload(OPP={"resources": {**opp_hand, "WOOD": 0, "ORE": 1}}, SELF={"resources": _zero_hand()}),
    )
    tracker.observe_frames([baseline, grant, after])
    exact = tracker.bounds_for("OPP").exact()
    assert exact == {**_zero_hand(), "WOOD": 0, "ORE": 1}


def test_confirm_trade_moves_cards_between_both_named_parties():
    # The `confirm_trade` EVENT itself only logs the responder's color -- the
    # give/want bundle is public via the PRECEDING frame's
    # `trade_panel.current_board_trade.trade` (mirroring
    # `catanatron.state.State.current_trade`, cleared once confirmed).
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    self_hand = {**_zero_hand(), "ORE": 2}
    baseline = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": self_hand}))
    opp_hand = {**_zero_hand(), "WOOD": 3}
    # RESOURCES order is (WOOD, BRICK, SHEEP, WHEAT, ORE).
    give_freqdeck = [1, 0, 0, 0, 0]  # OPP (offer actor) gives 1 wood
    want_freqdeck = [0, 0, 0, 0, 1]  # OPP wants 1 ore
    pending = _frame(
        _board_action("OPP", "PLAY_YEAR_OF_PLENTY", value=("WOOD", "WOOD", "WOOD")),
        _players_payload(OPP={"resources": opp_hand}, SELF={"resources": self_hand}),
        trade_panel={"current_board_trade": {"trade": (*give_freqdeck, *want_freqdeck, 0)}},
    )
    after = _frame(
        _board_action("OPP", "confirm_trade", value="SELF"),
        _players_payload(
            OPP={"resources": {**opp_hand, "WOOD": 2, "ORE": 1}},
            SELF={"resources": {**_zero_hand(), "ORE": 1, "WOOD": 1}},
        ),
    )
    tracker.observe_frames([baseline, pending, after])
    exact = tracker.bounds_for("OPP").exact()
    assert exact == {**_zero_hand(), "WOOD": 2, "ORE": 1}


def test_discard_by_opponent_widens_bounds_but_stays_correct():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    exact_hand = {"WOOD": 3, "BRICK": 2, "SHEEP": 0, "WHEAT": 0, "ORE": 0}
    baseline = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": _zero_hand()}))
    grant, opp_hand = _grant_frame(
        _zero_hand(), "OPP", "SELF", _zero_hand(), {"WOOD": exact_hand["WOOD"], "BRICK": exact_hand["BRICK"]}
    )
    after_total = 4
    after = _frame(
        _board_action("OPP", "DISCARD_RESOURCE", value=None, result=None),
        _players_payload(
            OPP={"resource_card_count": after_total},
            SELF={"resources": _zero_hand()},
        ),
    )
    tracker.observe_frames([baseline, grant, after])
    bounds = tracker.bounds_for("OPP")
    # True post-discard hand is one of {WOOD:2,BRICK:2} or {WOOD:3,BRICK:1}.
    assert bounds.exact() is None
    assert bounds.contains({"WOOD": 2, "BRICK": 2, "SHEEP": 0, "WHEAT": 0, "ORE": 0})
    assert bounds.contains({"WOOD": 3, "BRICK": 1, "SHEEP": 0, "WHEAT": 0, "ORE": 0})
    # But NOT an impossible hand (total must be 4, and can't exceed the prior count per-resource).
    assert not bounds.contains({"WOOD": 3, "BRICK": 2, "SHEEP": 0, "WHEAT": 0, "ORE": 0}, total=after_total)


def test_discard_by_self_does_not_affect_opponent_bounds():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    self_hand = {**_zero_hand(), "WHEAT": 3}
    baseline = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": self_hand}))
    grant, opp_hand = _grant_frame(_zero_hand(), "OPP", "SELF", self_hand, {"WOOD": 2})
    after = _frame(
        _board_action("SELF", "DISCARD_RESOURCE", value="hidden_resource", result="hidden_resource"),
        _players_payload(OPP={"resources": opp_hand}, SELF={"resources": {**_zero_hand(), "WHEAT": 2}}),
    )
    tracker.observe_frames([baseline, grant, after])
    assert tracker.bounds_for("OPP").exact() == opp_hand


def test_move_robber_self_is_thief_recovers_exact_stolen_card():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    baseline = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": _zero_hand()}))
    grant, opp_hand = _grant_frame(_zero_hand(), "OPP", "SELF", _zero_hand(), {"SHEEP": 2})
    after = _frame(
        _board_action("SELF", "MOVE_ROBBER", value=((1, 1, -2), "OPP"), result="hidden_stolen_resource"),
        _players_payload(
            OPP={"resources": {**opp_hand, "SHEEP": 1}},
            SELF={"resources": {**_zero_hand(), "SHEEP": 1}},
        ),
    )
    tracker.observe_frames([baseline, grant, after])
    assert tracker.bounds_for("OPP").exact()["SHEEP"] == 1


def test_move_robber_self_is_victim_recovers_exact_gain_for_opponent():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    before = _frame(
        None,
        _players_payload(
            OPP={"resources": _zero_hand()},
            SELF={"resources": {**_zero_hand(), "ORE": 3}},
        ),
    )
    after = _frame(
        _board_action("OPP", "MOVE_ROBBER", value=((0, 0, 0), "SELF"), result="hidden_stolen_resource"),
        _players_payload(
            OPP={"resources": {**_zero_hand(), "ORE": 1}},
            SELF={"resources": {**_zero_hand(), "ORE": 2}},
        ),
    )
    tracker.observe_frames([before, after])
    assert tracker.bounds_for("OPP").exact()["ORE"] == 1


def test_move_robber_no_steal_result_none_is_a_no_op():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    baseline = _frame(None, _players_payload(OPP={"resources": _zero_hand()}, SELF={"resources": _zero_hand()}))
    grant, hand = _grant_frame(_zero_hand(), "OPP", "SELF", _zero_hand(), {"WOOD": 1})
    after = _frame(
        _board_action("SELF", "MOVE_ROBBER", value=((0, 0, 0), None), result=None),
        _players_payload(OPP={"resources": hand}, SELF={"resources": _zero_hand()}),
    )
    tracker.observe_frames([baseline, grant, after])
    assert tracker.bounds_for("OPP").exact() == hand


def test_feature_vector_has_documented_fixed_size_and_range():
    tracker = DeductionTracker(self_name="SELF", opponent_names=("OPP",))
    payload = {
        "players": _players_payload(
            SELF={"development_cards": {c: 0 for c in STARTING_DEV_DECK}},
            OPP={"development_card_count": 2, "resource_card_count": 3},
        )
    }
    vector = tracker.feature_vector_for("OPP", payload)
    assert vector.shape == (DEDUCTION_FEATURE_SIZE,)
    assert vector.dtype == np.float32
    assert np.all(vector >= 0.0) and np.all(vector <= 1.0)

    table = tracker.feature_table(payload)
    assert table.shape == (4, DEDUCTION_FEATURE_SIZE)


def test_true_state_label_reads_omniscient_fields():
    payload = {
        "players": _players_payload(
            OPP={"resources": {**_zero_hand(), "ORE": 2}, "development_cards": {c: 0 for c in STARTING_DEV_DECK}}
        )
    }
    label = true_state_label(payload, "OPP")
    assert label is not None
    assert label["resources"]["ORE"] == 2


def test_true_state_label_none_when_perspective_masked():
    payload = {"players": _players_payload(OPP={"resource_card_count": 5, "development_card_count": 1})}
    assert true_state_label(payload, "OPP") is None


# --------------------------------------------------------------------------
# End-to-end: real engine replay, cross-checked against omniscient ground truth
# --------------------------------------------------------------------------


def _run_random_game(seed: int, *, max_steps: int = 400):
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    rng = random.Random(seed)
    _, info = env.reset(seed=seed)
    steps = 0
    while steps < max_steps:
        valid = tuple(int(a) for a in info.get("valid_actions", ()))
        if not valid:
            break
        action = rng.choice(valid)
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    return env


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_tracker_never_violates_ground_truth_on_real_games(seed):
    env = _run_random_game(seed)
    frames = env.replay_trace(actor="BLUE")
    tracker = DeductionTracker(self_name="BLUE", opponent_names=("RED",))

    violations = 0
    exact_checks = 0
    exact_hits = 0
    for i in range(1, len(frames)):
        tracker.observe_frames([frames[i - 1], frames[i]] if i == 1 else [frames[i]])
        omniscient = frames[i]["observations"]["RED"]
        true_hand = true_state_label(omniscient, "RED")
        if true_hand is None:
            continue
        bounds = tracker.bounds_for("RED")
        true_total = sum(true_hand["resources"].values())
        if not bounds.contains(true_hand["resources"], total=true_total):
            violations += 1
        exact_checks += 1
        if bounds.exact() == true_hand["resources"]:
            exact_hits += 1

    assert violations == 0
    # Sanity: the tracker should resolve to full exactness most of the time
    # in a 2-player game (only opponent discards / dev-card timing widen it).
    assert exact_checks > 0
    assert exact_hits / exact_checks > 0.5
