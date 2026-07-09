from __future__ import annotations

import random

from catan_zero.rl._catanatron import import_catanatron_module


def _make_two_player_game(seed: int):
    game_module = import_catanatron_module("catanatron.game")
    player_module = import_catanatron_module("catanatron.models.player")
    Game = game_module.Game
    Color = player_module.Color
    RandomPlayer = player_module.RandomPlayer

    rng = random.Random(seed)
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players)
    while game.state.is_initial_build_phase:
        game.execute(rng.choice(game.playable_actions))
    return game, rng


def test_roll_spectrum_enumerated_outcomes_apply_the_requested_dice() -> None:
    """FIX A2: each of the 11 enumerated ROLL outcomes must resolve with THAT dice total, not
    a freshly re-rolled one (apply_roll ignored the enumeration when no action_record was
    passed)."""
    enums_module = import_catanatron_module("catanatron.models.enums")
    tree_search_utils = import_catanatron_module("catanatron.players.tree_search_utils")
    ActionType = enums_module.ActionType

    game, _ = _make_two_player_game(seed=0)
    roll_action = next(a for a in game.playable_actions if a.action_type == ActionType.ROLL)

    results = tree_search_utils.execute_spectrum(game, roll_action)
    assert len(results) == 11
    for (result_game, _proba), roll in zip(results, range(2, 13)):
        dices = result_game.state.action_records[-1].result
        assert dices[0] + dices[1] == roll


def test_roll_spectrum_is_deterministic_across_repeated_expansions() -> None:
    """Expanding the ROLL spectrum twice with the same inputs must yield identical child
    states per outcome -- the enumerated outcome must be APPLIED, not resampled."""
    enums_module = import_catanatron_module("catanatron.models.enums")
    tree_search_utils = import_catanatron_module("catanatron.players.tree_search_utils")
    ActionType = enums_module.ActionType

    game, _ = _make_two_player_game(seed=1)
    roll_action = next(a for a in game.playable_actions if a.action_type == ActionType.ROLL)

    results_a = tree_search_utils.execute_spectrum(game, roll_action)
    results_b = tree_search_utils.execute_spectrum(game, roll_action)

    assert len(results_a) == len(results_b) == 11
    for (game_a, proba_a), (game_b, proba_b) in zip(results_a, results_b):
        assert proba_a == proba_b
        assert (
            game_a.state.action_records[-1].result
            == game_b.state.action_records[-1].result
        )
        # The full deterministic consequence of the roll (bank resource freqdeck) must match
        # too, not just the raw dice tuple.
        assert game_a.state.resource_freqdeck == game_b.state.resource_freqdeck


def test_buy_development_card_spectrum_applies_the_enumerated_card() -> None:
    """FIX A2: each enumerated card option must draw THAT card (or fail/skip if it isn't
    actually available), never silently collapse every branch onto one fresh random draw."""
    enums_module = import_catanatron_module("catanatron.models.enums")
    tree_search_utils = import_catanatron_module("catanatron.players.tree_search_utils")
    ActionType = enums_module.ActionType

    game, rng = _make_two_player_game(seed=0)
    buy_actions: list = []
    for _ in range(500):
        buy_actions = [
            a for a in game.playable_actions if a.action_type == ActionType.BUY_DEVELOPMENT_CARD
        ]
        if buy_actions:
            break
        game.execute(rng.choice(game.playable_actions))
    assert buy_actions, "no BUY_DEVELOPMENT_CARD action found within step budget"

    results = tree_search_utils.execute_spectrum(game, buy_actions[0])
    executed = [
        result_game.state.action_records[-1]
        for result_game, _proba in results
        if result_game.state.action_records[-1].action.action_type
        == ActionType.BUY_DEVELOPMENT_CARD
    ]
    drawn_cards = {record.result for record in executed}
    for record in executed:
        # The Action itself is rewritten in apply_buy_development_card to carry the drawn card
        # -- it must match the ActionRecord.result we requested, proving the enumerated card
        # (not a fresh random one) was applied.
        assert record.action.value == record.result
    # If enumeration were ignored (the A2 bug), every successful draw would collapse onto ONE
    # freshly-random card regardless of which of the 5 branches produced it.
    assert len(drawn_cards) >= 2
