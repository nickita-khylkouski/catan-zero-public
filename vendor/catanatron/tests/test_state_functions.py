import pytest

from catanatron.state import State
from catanatron.apply_action import apply_action
from catanatron.state_functions import (
    buy_dev_card,
    get_actual_victory_points,
    get_largest_army,
    maintain_longest_road,
    player_key,
    play_dev_card,
    player_deck_random_select,
    player_deck_replenish,
)
from catanatron.models.enums import (
    KNIGHT,
    ORE,
    SHEEP,
    WHEAT,
    Action,
    ActionType,
)
from catanatron.models.player import Color, SimplePlayer


def test_longest_road_is_revoked_when_no_player_still_qualifies():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    red_key = player_key(state, Color.RED)
    state.player_state[f"{red_key}_HAS_ROAD"] = True
    state.player_state[f"{red_key}_VICTORY_POINTS"] = 2
    state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] = 2

    maintain_longest_road(
        state,
        previous_road_color=Color.RED,
        road_color=None,
        road_lengths={Color.RED: 4, Color.BLUE: 3},
    )

    assert state.player_state[f"{red_key}_LONGEST_ROAD_LENGTH"] == 4
    assert state.player_state[f"{red_key}_HAS_ROAD"] is False
    assert state.player_state[f"{red_key}_VICTORY_POINTS"] == 0
    assert state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] == 0


def test_cant_steal_devcards():
    # Arrange: Have RED buy 1 dev card (and have no resource cards)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    player_deck_replenish(state, Color.RED, WHEAT)
    player_deck_replenish(state, Color.RED, ORE)
    player_deck_replenish(state, Color.RED, SHEEP)
    buy_dev_card(state, Color.RED, KNIGHT)

    # Act: Attempt to steal a resource
    with pytest.raises(IndexError):  # no resource cards in hand
        player_deck_random_select(state, Color.RED)


def test_defeating_your_own_largest_army_doesnt_give_more_vps():
    # Arrange: Buy all dev cards
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    state = State(players)
    player_deck_replenish(state, players[0].color, SHEEP, 26)
    player_deck_replenish(state, players[0].color, WHEAT, 26)
    player_deck_replenish(state, players[0].color, ORE, 26)
    for i in range(25):
        apply_action(
            state, Action(players[0].color, ActionType.BUY_DEVELOPMENT_CARD, None)
        )
    assert get_largest_army(state) == (None, None)
    assert get_actual_victory_points(state, Color.RED) == 5

    # Act - Assert
    play_dev_card(state, Color.RED, KNIGHT)
    play_dev_card(state, Color.RED, KNIGHT)
    play_dev_card(state, Color.RED, KNIGHT)
    assert get_largest_army(state) == (Color.RED, 3)
    assert get_actual_victory_points(state, Color.RED) == 7

    # Act - Assert
    play_dev_card(state, Color.RED, KNIGHT)
    assert get_largest_army(state) == (Color.RED, 4)
    assert get_actual_victory_points(state, Color.RED) == 7
