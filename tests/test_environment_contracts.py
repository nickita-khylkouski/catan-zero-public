from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, is_dataclass, replace
import importlib
from typing import Any

import pytest

from catan_zero.engine import CatanEngine
from catan_zero.schemas import (
    Action,
    ActionKind,
    Event,
    Observation,
    Phase,
    PublicPlayerState,
    ResourceBundle,
    SeedBundle,
)


RULESET_ID = "CatanBench-4P-Full-v1"
FORBIDDEN_HIDDEN_KEYS = {
    "opponent_resources",
    "opponent_development_cards",
    "development_deck_order",
    "future_dice",
    "future_steals",
}


def _public_player(seat: int) -> PublicPlayerState:
    return PublicPlayerState(
        seat=seat,
        public_victory_points=0,
        resource_count=0,
        development_card_count=0,
        roads_remaining=15,
        settlements_remaining=5,
        cities_remaining=4,
        knights_played=0,
    )


class _ContractProbeEngine(CatanEngine):
    """Small deterministic engine double used to pin the CatanEngine contract."""

    def __init__(self) -> None:
        self._seed_bundle: SeedBundle | None = None
        self._ply = 0
        self._events: tuple[Event, ...] = ()

    def reset(self, seed_bundle: SeedBundle) -> dict[str, int]:
        self._seed_bundle = seed_bundle
        self._ply = 0
        self._events = ()
        return asdict(seed_bundle)

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        self._require_reset()
        actions = [Action(kind=ActionKind.END_TURN, actor=player_id)]
        if (self._seed_bundle.board + self._ply) % 2 == 0:  # type: ignore[union-attr]
            actions.insert(0, Action(kind=ActionKind.ROLL_DICE, actor=player_id))
        return tuple(actions)

    def observe(self, player_id: int) -> Observation:
        self._require_reset()
        seed = self._seed_bundle
        assert seed is not None
        return Observation(
            ruleset_id=RULESET_ID,
            acting_seat=player_id,
            phase=Phase.MAIN,
            public_board={
                "board_seed": seed.board,
                "robber_tile": seed.robber_steal % 19,
            },
            public_players=tuple(_public_player(seat) for seat in range(4)),
            own_resources=ResourceBundle(brick=seed.dice % 3),
            own_development_cards=(),
            public_event_history=self.event_log(),
            legal_actions=self.legal_actions(player_id),
        )

    def step(self, action: Action) -> dict[str, int | str]:
        self._require_reset()
        self._ply += 1
        self._events = (
            *self._events,
            Event(
                ply=self._ply,
                event_type="action",
                actor=action.actor,
                public={"kind": action.kind.value},
            ),
        )
        return {"ply": self._ply, "kind": action.kind.value}

    def clone(self) -> tuple[SeedBundle | None, int, tuple[Event, ...]]:
        return self._seed_bundle, self._ply, self._events

    def restore(self, snapshot: Any) -> None:
        seed_bundle, ply, events = snapshot
        self._seed_bundle = seed_bundle
        self._ply = ply
        self._events = tuple(events)

    def event_log(self) -> tuple[Event, ...]:
        return self._events

    def _require_reset(self) -> None:
        if self._seed_bundle is None:
            raise RuntimeError("engine must be reset before use")


class _DiscreteActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n


class _MaskProbeEnv:
    action_space_size = 5

    def __init__(self) -> None:
        self.action_space = _DiscreteActionSpace(self.action_space_size)
        self._valid_actions = (1, 3)

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, int | None], dict[str, list[int]]]:
        return {"seed": seed}, {"valid_actions": list(self._valid_actions)}

    def action_masks(self) -> list[bool]:
        return [index in self._valid_actions for index in range(self.action_space_size)]


def _engine_signature(engine: CatanEngine, player_id: int = 0) -> dict[str, Any]:
    observation = engine.observe(player_id)
    _assert_observation_has_no_hidden_fields(observation)
    return {
        "observation": observation,
        "legal_actions": tuple(
            action.to_dict() for action in engine.legal_actions(player_id)
        ),
        "event_log": engine.event_log(),
    }


def _assert_observation_has_no_hidden_fields(observation: Observation) -> None:
    observation.assert_no_hidden_opponent_fields()
    leaked_keys = sorted(FORBIDDEN_HIDDEN_KEYS.intersection(_walk_keys(observation)))
    assert leaked_keys == []


def _walk_keys(value: Any) -> set[str]:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_walk_keys(item))
        return keys
    if isinstance(value, (list, tuple, set, frozenset)):
        keys: set[str] = set()
        for item in value:
            keys.update(_walk_keys(item))
        return keys
    return set()


def _assert_action_mask_contract(env: Any, info: dict[str, Any]) -> None:
    action_space_size = getattr(getattr(env, "action_space", None), "n", None)
    if action_space_size is None:
        action_space_size = getattr(env, "action_space_size")

    mask = env.action_masks() if hasattr(env, "action_masks") else info["action_mask"]

    assert len(mask) == action_space_size
    assert all(_is_bool_like(flag) for flag in mask)

    valid_actions = info.get("valid_actions")
    if valid_actions is not None:
        assert all(isinstance(action, int) for action in valid_actions)
        assert all(0 <= action < action_space_size for action in valid_actions)
        assert {index for index, flag in enumerate(mask) if flag} == set(valid_actions)


def _is_bool_like(value: Any) -> bool:
    return value is True or value is False or type(value).__name__ == "bool_"


@pytest.mark.parametrize(
    "module_name",
    [
        "catan_zero",
        "catan_zero.schemas",
        "catan_zero.engine",
        "catan_zero.rules",
        "catan_zero.benchmark",
        "catan_zero.adapters.catanatron",
    ],
)
def test_catan_zero_modules_are_importable(module_name: str) -> None:
    importlib.import_module(module_name)


def test_adapter_constructs_without_optional_catanatron_dependency() -> None:
    from catan_zero.adapters.catanatron import CatanatronAdapter

    adapter = CatanatronAdapter()
    seed_bundle = SeedBundle(
        board=1, dice=2, development_deck=3, robber_steal=4, seat_order=5
    )

    try:
        reset_result = adapter.reset(seed_bundle)
    except (RuntimeError, NotImplementedError):
        return

    assert reset_result.ruleset_id == RULESET_ID


def test_seed_bundle_is_immutable_and_value_based() -> None:
    seed_bundle = SeedBundle(
        board=10, dice=20, development_deck=30, robber_steal=40, seat_order=50
    )

    assert seed_bundle == SeedBundle(
        board=10,
        dice=20,
        development_deck=30,
        robber_steal=40,
        seat_order=50,
    )
    assert seed_bundle != SeedBundle(
        board=10,
        dice=21,
        development_deck=30,
        robber_steal=40,
        seat_order=50,
    )
    with pytest.raises(FrozenInstanceError):
        seed_bundle.dice = 99  # type: ignore[misc]


def test_engine_contract_is_deterministic_for_same_seed_and_actions() -> None:
    seed_bundle = SeedBundle(
        board=10, dice=20, development_deck=30, robber_steal=40, seat_order=50
    )
    left = _ContractProbeEngine()
    right = _ContractProbeEngine()

    assert left.reset(seed_bundle) == right.reset(seed_bundle)
    assert _engine_signature(left) == _engine_signature(right)

    action = left.legal_actions(player_id=0)[0]
    assert left.step(action) == right.step(action)
    assert _engine_signature(left) == _engine_signature(right)


def test_engine_contract_changes_when_seed_changes_public_state() -> None:
    left = _ContractProbeEngine()
    right = _ContractProbeEngine()

    left.reset(
        SeedBundle(board=10, dice=20, development_deck=30, robber_steal=40, seat_order=50)
    )
    right.reset(
        SeedBundle(board=11, dice=20, development_deck=30, robber_steal=40, seat_order=50)
    )

    assert _engine_signature(left)["observation"] != _engine_signature(right)["observation"]


def test_action_mask_shape_matches_action_space_and_valid_actions() -> None:
    env = _MaskProbeEnv()
    first_observation, first_info = env.reset(seed=123)
    second_observation, second_info = env.reset(seed=123)

    assert first_observation == second_observation
    assert first_info == second_info
    _assert_action_mask_contract(env, first_info)


def test_observation_contract_rejects_hidden_information_keys() -> None:
    engine = _ContractProbeEngine()
    engine.reset(
        SeedBundle(board=10, dice=20, development_deck=30, robber_steal=40, seat_order=50)
    )
    clean_observation = engine.observe(player_id=0)

    _assert_observation_has_no_hidden_fields(clean_observation)

    leaky_observation = replace(
        clean_observation,
        public_board={
            **clean_observation.public_board,
            "development_deck_order": ["knight", "victory_point"],
        },
    )
    with pytest.raises(AssertionError):
        _assert_observation_has_no_hidden_fields(leaky_observation)


def test_optional_rl_environment_contract_if_available() -> None:
    factory = _load_optional_env_factory()
    if factory is None:
        pytest.skip("No CatanZero RL environment implementation is exposed yet")

    env = factory()
    first_observation, first_info = env.reset(seed=123)
    second_observation, second_info = env.reset(seed=123)

    assert repr(first_observation) == repr(second_observation)
    assert first_info == second_info
    _assert_action_mask_contract(env, first_info)


def _load_optional_env_factory() -> Any | None:
    for module_name, attribute in (
        ("catan_zero.env", "CatanZeroEnv"),
        ("catan_zero.environment", "CatanZeroEnv"),
        ("catan_zero.rl_env", "CatanZeroEnv"),
        ("catan_zero.gym_env", "CatanZeroEnv"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue
        if hasattr(module, attribute):
            return getattr(module, attribute)
    return None
