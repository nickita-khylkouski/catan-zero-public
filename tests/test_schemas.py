import pytest

from catan_zero.schemas import Action, ActionKind, Observation, Phase, ResourceBundle


def test_resource_bundle_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        ResourceBundle(brick=-1)


def test_action_round_trip() -> None:
    action = Action(
        kind=ActionKind.OFFER_TRADE,
        actor=0,
        target_player=2,
        give=ResourceBundle(brick=1),
        receive=ResourceBundle(ore=1),
    )

    assert Action.from_dict(action.to_dict()) == action


def test_observation_hidden_field_guard() -> None:
    observation = Observation(
        ruleset_id="CatanBench-4P-Full-v1",
        acting_seat=0,
        phase=Phase.MAIN,
        public_board={"tiles": []},
        public_players=(),
        own_resources=ResourceBundle(),
        own_development_cards=(),
        public_event_history=(),
        legal_actions=(),
    )

    observation.assert_no_hidden_opponent_fields()

