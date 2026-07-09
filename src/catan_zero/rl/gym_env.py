from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from catan_zero.rl.chat import ColonistChatConfig, ColonistChatState
from catan_zero.rl.negotiation import (
    ColonistNegotiationState,
    RESOURCE_NAMES,
    TradeSide,
    exact_side,
    open_side,
    wildcard_side,
)
from catan_zero.rl.timers import (
    COLONIST_TIMER_PROFILES,
    TimerProfileName,
    timer_phase_from_prompt,
)

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised only without gym extras.
    gym = None


@dataclass(frozen=True, slots=True)
class CatanZeroGymConfig:
    """Configuration for the first CatanZero RL environment.

    This is intentionally a one-learning-player wrapper around Catanatron's Gym
    environment. It is enough to start PPO/action-mask work while the full
    four-policy multi-agent environment is mapped.
    """

    map_type: str = "BASE"
    players: int = 4
    vps_to_win: int = 10
    representation: str = "mixed"
    enemy_policy: str = "random"
    invalid_action_reward: float = -1.0
    enable_player_trading: bool = True
    max_trade_resources_per_side: int = 2
    max_player_trade_offers_per_turn: int = 3
    enable_table_chat: bool = True
    max_chat_messages_per_turn: int = 4
    max_chat_chars: int = 180
    allow_free_text_chat: bool = True
    enable_timers: bool = True
    timer_profile: TimerProfileName = "normal"


class CatanZeroGymEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Stable CatanZero wrapper around Catanatron's Gymnasium env."""

    metadata = {"render_modes": ["rgb_array", "db"], "render_fps": 30}

    def __init__(self, config: CatanZeroGymConfig | None = None) -> None:
        self.config = config or CatanZeroGymConfig()
        if self.config.players not in (2, 3, 4):
            raise ValueError("Catanatron supports 2-4 player configs through enemies")
        if self.config.max_player_trade_offers_per_turn < 0:
            raise ValueError("max_player_trade_offers_per_turn must be non-negative")
        if self.config.timer_profile not in COLONIST_TIMER_PROFILES:
            raise ValueError(f"unknown timer_profile: {self.config.timer_profile}")
        self._last_turn_key: tuple[int, int] | None = None
        self._trade_offers_this_turn = 0
        self.chat = ColonistChatState(
            ColonistChatConfig(
                enabled=self.config.enable_table_chat,
                max_chars=self.config.max_chat_chars,
                max_messages_per_turn=self.config.max_chat_messages_per_turn,
                allow_free_text=self.config.allow_free_text_chat,
            )
        )
        self.negotiation = ColonistNegotiationState()
        self._timer_profile = COLONIST_TIMER_PROFILES[self.config.timer_profile]
        self._env = self._make_env()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        return self._env.observation_space

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    @property
    def supports_player_trading(self) -> bool:
        return self.config.enable_player_trading

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        observation, info = self._env.reset(seed=seed)
        self._last_turn_key = self._current_turn_key()
        self._trade_offers_this_turn = 0
        self.chat.reset()
        self.negotiation.reset()
        return observation, self._normalize_info(info)

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.config.enable_player_trading and action >= self._base_action_space_n:
            return self._step_extended_action(action)
        observation, reward, terminated, truncated, info = self._env.step(action)
        return observation, float(reward), bool(terminated), bool(truncated), self._normalize_info(info)

    def valid_actions(self) -> tuple[int, ...]:
        self._sync_turn_counters()
        if not self.config.enable_player_trading:
            return tuple(int(action) for action in self._env.unwrapped.get_valid_actions())

        return tuple(sorted(self._extended_valid_actions()))

    def action_mask(self) -> list[bool]:
        if not self.config.enable_player_trading and hasattr(self._env.unwrapped, "action_masks"):
            return [bool(value) for value in self._env.unwrapped.action_masks()]
        valid = set(self.valid_actions())
        return [idx in valid for idx in range(self.action_space.n)]

    def action_masks(self) -> list[bool]:
        return self.action_mask()

    def timer_info(self) -> dict[str, Any]:
        """Return virtual Colonist-style timer metadata for the current prompt."""

        unwrapped = self._env.unwrapped
        prompt_name = getattr(unwrapped.game.state.current_prompt, "name", "unknown")
        action_types = tuple(action.action_type.name for action in unwrapped.game.playable_actions)
        phase = timer_phase_from_prompt(prompt_name, action_types)
        timeout_action = self.timeout_action()
        return {
            "enabled": self.config.enable_timers,
            "profile": self.config.timer_profile,
            "prompt": prompt_name,
            "phase": phase,
            "budget_seconds": self._timer_profile.budget_for_phase(phase),
            "timeout_action": timeout_action,
            "timeout_action_description": self.describe_action(timeout_action)
            if timeout_action is not None
            else None,
        }

    def timeout_action(self) -> int | None:
        """Choose a deterministic Colonist-like fallback for virtual timeout."""

        valid = set(self.valid_actions())
        if not valid:
            return None

        for preferred in (
            ("reject_trade", None),
            ("cancel_trade", None),
        ):
            action = self._extended_action_index(*preferred)
            if action in valid:
                return action

        priority = (
            "ROLL",
            "DISCARD_RESOURCE",
            "MOVE_ROBBER",
            "END_TURN",
        )
        for action_type in priority:
            matches = [
                index
                for index in valid
                if index < self._base_action_space_n
                and self._base_action_array[index][0].name == action_type
            ]
            if matches:
                return min(matches)

        return min(valid)

    def step_timeout(self) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply the deterministic fallback action for the current timer phase."""

        action = self.timeout_action()
        if action is None:
            raise RuntimeError("no timeout action is available")
        return self.step(action)

    def describe_action(self, action: int | None) -> dict[str, Any] | None:
        if action is None:
            return None
        if action < self._base_action_space_n:
            action_type, value = self._base_action_array[action]
            return {"index": action, "kind": action_type.name, "value": _serialize_value(value)}
        try:
            kind, value = self._extended_actions[action - self._base_action_space_n]
        except IndexError:
            return None
        return {"index": action, "kind": kind, "value": _serialize_value(value)}

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def post_chat(
        self,
        text: str,
        *,
        actor: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Post a public table-chat message without consuming a board action."""

        message = self.chat.post_text(
            actor=actor or self._actor_name(),
            text=text,
            turn_key=self._current_turn_key(),
            metadata=metadata,
        )
        return message.to_dict()

    def post_chat_template(
        self,
        template_id: str,
        values: dict[str, Any] | None = None,
        *,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """Post a structured Colonist-style negotiation intent."""

        message = self.chat.post_template(
            actor=actor or self._actor_name(),
            template_id=template_id,
            values=values,
            turn_key=self._current_turn_key(),
        )
        return message.to_dict()

    def propose_trade(
        self,
        *,
        give: dict[str, Any],
        want: dict[str, Any],
        target: str | None = None,
        actor: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a Colonist-style open, wildcard, or exact trade proposal."""

        offer = self.negotiation.create_offer(
            actor=actor or self._actor_name(),
            turn_key=self._current_turn_key(),
            target=target,
            give=_coerce_trade_side(give),
            want=_coerce_trade_side(want),
            metadata=metadata,
        )
        self._post_negotiation_chat(offer.to_dict())
        return offer.to_dict()

    def respond_to_trade(
        self,
        offer_id: int,
        status: str,
        *,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """Record a visible Colonist-style response status for an open offer."""

        offer = self.negotiation.respond(
            offer_id=offer_id,
            actor=actor or self._actor_name(),
            status=status,  # type: ignore[arg-type]
        )
        return offer.to_dict()

    def counter_trade(
        self,
        offer_id: int,
        *,
        give: dict[str, Any],
        want: dict[str, Any],
        actor: str | None = None,
        target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a counteroffer linked to an existing open offer."""

        actor_name = actor or self._actor_name()
        self.negotiation.respond(
            offer_id=offer_id,
            actor=actor_name,
            status="countered",
        )
        counter = self.negotiation.create_offer(
            actor=actor_name,
            turn_key=self._current_turn_key(),
            target=target,
            give=_coerce_trade_side(give),
            want=_coerce_trade_side(want),
            parent_offer_id=offer_id,
            metadata=metadata,
        )
        self._post_negotiation_chat(counter.to_dict())
        return counter.to_dict()

    def trade_action_for_offer(
        self,
        offer_id: int,
        *,
        give: dict[str, Any] | None = None,
        want: dict[str, Any] | None = None,
    ) -> int | None:
        """Return the concrete board-action index for a resolved offer.

        Open and wildcard offers are Colonist negotiation workflow. They become
        Catan resource exchanges only after exact bundles are selected.
        """

        offer = self.negotiation.get_offer(offer_id)
        exact_give = _resolve_trade_side(offer.give, give)
        exact_want = _resolve_trade_side(offer.want, want)
        action_value = (
            *_trade_side_freqdeck(exact_give),
            *_trade_side_freqdeck(exact_want),
        )
        for offset, (kind, value) in enumerate(self._extended_actions):
            action_index = self._base_action_space_n + offset
            if kind == "offer_trade" and value == action_value:
                return action_index if action_index in set(self.valid_actions()) else None
        return None

    def step_negotiated_trade(
        self,
        offer_id: int,
        *,
        give: dict[str, Any] | None = None,
        want: dict[str, Any] | None = None,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Resolve a negotiation offer into a concrete Catan trade offer."""

        action = self.trade_action_for_offer(offer_id, give=give, want=want)
        if action is None:
            raise ValueError("negotiation offer does not resolve to a valid trade action")
        return self.step(action)

    def _normalize_info(self, info: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(info)
        if self.config.enable_player_trading:
            normalized["valid_actions"] = self.valid_actions()
        else:
            normalized["valid_actions"] = tuple(int(action) for action in info.get("valid_actions", ()))
        normalized["action_mask"] = self.action_mask()
        normalized["player_trade_offers_this_turn"] = self._trade_offers_this_turn
        normalized["max_player_trade_offers_per_turn"] = self.config.max_player_trade_offers_per_turn
        normalized["chat_log"] = self.chat.log()
        normalized["valid_chat_templates"] = self.chat.valid_template_ids(
            self._actor_name(),
            self._current_turn_key(),
        )
        normalized["chat_messages_remaining"] = self.chat.remaining_messages(
            self._actor_name(),
            self._current_turn_key(),
        )
        normalized["negotiation_offers"] = self.negotiation.offers()
        normalized["open_negotiation_offers"] = self.negotiation.open_offers_for(
            self._actor_name()
        )
        normalized["timer"] = self.timer_info()
        return normalized

    def _make_env(self) -> Any:
        try:
            import gymnasium
            import catanatron.gym  # noqa: F401
            from catanatron.models.player import Color, RandomPlayer
            from catanatron.players.value import ValueFunctionPlayer
        except ImportError as exc:
            raise RuntimeError(
                "Install Catanatron with gym extras before constructing CatanZeroGymEnv."
            ) from exc

        colors = [Color.RED, Color.ORANGE, Color.WHITE]
        enemy_count = self.config.players - 1
        enemy_colors = colors[:enemy_count]
        enemies: Sequence[Any]
        if self.config.enemy_policy == "random":
            enemies = [RandomPlayer(color) for color in enemy_colors]
        elif self.config.enemy_policy == "value":
            enemies = [ValueFunctionPlayer(color) for color in enemy_colors]
        else:
            raise ValueError(f"unknown enemy_policy: {self.config.enemy_policy}")

        env = gymnasium.make(
            "catanatron/Catanatron-v0",
            config={
                "map_type": self.config.map_type,
                "vps_to_win": self.config.vps_to_win,
                "representation": self.config.representation,
                "enemies": list(enemies),
                "invalid_action_reward": self.config.invalid_action_reward,
            },
        )
        self._init_extended_action_space(env)
        return env

    def _init_extended_action_space(self, env: Any) -> None:
        try:
            from gymnasium import spaces
            from catanatron.gym.envs.action_space import get_action_array
        except ImportError as exc:
            raise RuntimeError("Catanatron Gym dependencies are required") from exc

        unwrapped = env.unwrapped
        self._base_action_space_n = int(unwrapped.action_space.n)
        self._base_action_array = tuple(
            get_action_array(unwrapped.player_colors, self.config.map_type)
        )
        self._offer_trade_values = tuple(
            _generate_trade_values(self.config.max_trade_resources_per_side)
        )
        # Dynamic response actions depend on current_trade, so the static action
        # space stores semantic placeholders.
        self._extended_actions: tuple[tuple[str, Any], ...] = (
            *(("offer_trade", value) for value in self._offer_trade_values),
            ("accept_trade", None),
            ("reject_trade", None),
            ("cancel_trade", None),
            *(("confirm_trade", color) for color in unwrapped.player_colors if color != unwrapped.p0.color),
        )
        if self.config.enable_player_trading:
            env.action_space = spaces.Discrete(self._base_action_space_n + len(self._extended_actions))
            unwrapped.action_space = env.action_space

    def _extended_valid_actions(self) -> list[int]:
        from catanatron.gym.envs.action_space import to_action_space
        from catanatron.models.enums import ActionType

        unwrapped = self._env.unwrapped
        valid: list[int] = []

        for action in unwrapped.game.playable_actions:
            try:
                valid.append(
                    to_action_space(action, unwrapped.player_colors, self.config.map_type)
                )
            except ValueError:
                valid.extend(self._trade_response_indices_for(action))

        if self._can_offer_trade():
            valid.extend(self._valid_offer_trade_indices())

        return valid

    def _trade_response_indices_for(self, catan_action: Any) -> list[int]:
        from catanatron.models.enums import ActionType

        result: list[int] = []
        for offset, (kind, value) in enumerate(self._extended_actions):
            idx = self._base_action_space_n + offset
            if catan_action.action_type == ActionType.ACCEPT_TRADE and kind == "accept_trade":
                result.append(idx)
            elif catan_action.action_type == ActionType.REJECT_TRADE and kind == "reject_trade":
                result.append(idx)
            elif catan_action.action_type == ActionType.CANCEL_TRADE and kind == "cancel_trade":
                result.append(idx)
            elif (
                catan_action.action_type == ActionType.CONFIRM_TRADE
                and kind == "confirm_trade"
                and len(catan_action.value) == 11
                and catan_action.value[10] == value
            ):
                result.append(idx)
        return result

    def _can_offer_trade(self) -> bool:
        from catanatron.game import is_valid_action
        from catanatron.models.enums import Action, ActionType

        if self._trade_offers_this_turn >= self.config.max_player_trade_offers_per_turn:
            return False
        unwrapped = self._env.unwrapped
        state = unwrapped.game.state
        if state.current_color() != unwrapped.p0.color:
            return False
        return any(
            is_valid_action(
                unwrapped.game.playable_actions,
                state,
                Action(unwrapped.p0.color, ActionType.OFFER_TRADE, value),
            )
            and self._player_has_resources(value[:5])
            for value in self._offer_trade_values
        )

    def _valid_offer_trade_indices(self) -> list[int]:
        from catanatron.game import is_valid_action
        from catanatron.models.enums import Action, ActionType

        if self._trade_offers_this_turn >= self.config.max_player_trade_offers_per_turn:
            return []
        unwrapped = self._env.unwrapped
        state = unwrapped.game.state
        valid: list[int] = []
        for offset, (kind, value) in enumerate(self._extended_actions):
            if kind != "offer_trade":
                continue
            catan_action = Action(unwrapped.p0.color, ActionType.OFFER_TRADE, value)
            if is_valid_action(unwrapped.game.playable_actions, state, catan_action) and self._player_has_resources(value[:5]):
                valid.append(self._base_action_space_n + offset)
        return valid

    def _player_has_resources(self, freqdeck: tuple[int, int, int, int, int]) -> bool:
        from catanatron.state_functions import player_resource_freqdeck_contains

        unwrapped = self._env.unwrapped
        return bool(player_resource_freqdeck_contains(unwrapped.game.state, unwrapped.p0.color, freqdeck))

    def _step_extended_action(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        from catanatron.game import TURNS_LIMIT
        from catanatron.models.enums import Action, ActionType

        unwrapped = self._env.unwrapped
        offset = action - self._base_action_space_n
        try:
            kind, value = self._extended_actions[offset]
        except IndexError:
            return self._invalid_action_response()

        current_trade = unwrapped.game.state.current_trade
        if kind == "offer_trade":
            catan_action = Action(unwrapped.p0.color, ActionType.OFFER_TRADE, value)
        elif kind == "accept_trade":
            catan_action = Action(unwrapped.p0.color, ActionType.ACCEPT_TRADE, current_trade)
        elif kind == "reject_trade":
            catan_action = Action(unwrapped.p0.color, ActionType.REJECT_TRADE, current_trade)
        elif kind == "cancel_trade":
            catan_action = Action(unwrapped.p0.color, ActionType.CANCEL_TRADE, None)
        elif kind == "confirm_trade":
            catan_action = Action(unwrapped.p0.color, ActionType.CONFIRM_TRADE, (*current_trade[:10], value))
        else:
            return self._invalid_action_response()

        if action not in set(self.valid_actions()):
            return self._invalid_action_response()

        unwrapped.game.execute(catan_action)
        if kind == "offer_trade":
            self._trade_offers_this_turn += 1
        unwrapped._advance_until_p0_decision()
        self._sync_turn_counters()
        observation = unwrapped._get_observation()
        winning_color = unwrapped.game.winning_color()
        terminated = winning_color is not None
        truncated = unwrapped.game.state.num_turns >= TURNS_LIMIT
        reward = unwrapped.reward_function(catan_action, unwrapped.game, unwrapped.p0.color)
        info = self._normalize_info({"valid_actions": self.valid_actions()})
        return observation, float(reward), bool(terminated), bool(truncated), info

    def _invalid_action_response(self) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        from catanatron.game import TURNS_LIMIT

        unwrapped = self._env.unwrapped
        unwrapped.invalid_actions_count += 1
        observation = unwrapped._get_observation()
        winning_color = unwrapped.game.winning_color()
        terminated = winning_color is not None
        truncated = (
            unwrapped.invalid_actions_count > unwrapped.max_invalid_actions
            or unwrapped.game.state.num_turns >= TURNS_LIMIT
        )
        return (
            observation,
            float(self.config.invalid_action_reward),
            bool(terminated),
            bool(truncated),
            self._normalize_info({"valid_actions": self.valid_actions()}),
        )

    def _current_turn_key(self) -> tuple[int, int]:
        state = self._env.unwrapped.game.state
        return int(state.num_turns), int(state.current_turn_index)

    def _actor_name(self) -> str:
        return getattr(self._env.unwrapped.p0.color, "name", str(self._env.unwrapped.p0.color))

    def _extended_action_index(self, kind: str, value: Any) -> int | None:
        for offset, candidate in enumerate(self._extended_actions):
            if candidate == (kind, value):
                return self._base_action_space_n + offset
        return None

    def _post_negotiation_chat(self, offer: dict[str, Any]) -> None:
        if not self.config.enable_table_chat:
            return
        try:
            self.chat.post_text(
                actor=offer["actor"],
                text=_negotiation_chat_text(offer),
                turn_key=self._current_turn_key(),
                intent="trade_negotiation",
                metadata={"offer_id": offer["offer_id"]},
            )
        except ValueError:
            # Chat caps should not prevent the structured negotiation state from
            # being recorded for training.
            return

    def _sync_turn_counters(self) -> None:
        if not hasattr(self, "_env"):
            return
        turn_key = self._current_turn_key()
        if self._last_turn_key != turn_key:
            self._last_turn_key = turn_key
            self._trade_offers_this_turn = 0


def _generate_trade_values(max_resources_per_side: int) -> list[tuple[int, int, int, int, int, int, int, int, int, int]]:
    give_bundles = _resource_freqdecks(max_resources_per_side)
    receive_bundles = _resource_freqdecks(max_resources_per_side)
    values: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for give in give_bundles:
        for receive in receive_bundles:
            if any(g > 0 and r > 0 for g, r in zip(give, receive)):
                continue
            values.append((*give, *receive))
    return values


def _resource_freqdecks(max_total: int) -> list[tuple[int, int, int, int, int]]:
    results: list[tuple[int, int, int, int, int]] = []

    def rec(position: int, remaining: int, current: list[int]) -> None:
        if position == 5:
            total = sum(current)
            if 1 <= total <= max_total:
                results.append(tuple(current))  # type: ignore[arg-type]
            return
        for count in range(remaining + 1):
            current.append(count)
            rec(position + 1, remaining - count, current)
            current.pop()

    rec(0, max_total, [])
    return results


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, tuple):
        return tuple(_serialize_value(item) for item in value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return repr(value)


def _coerce_trade_side(spec: dict[str, Any]) -> TradeSide:
    kind = spec.get("kind")
    if kind is None:
        return exact_side(**{key: int(value) for key, value in spec.items()})
    if kind == "exact":
        resources = spec.get("resources", {})
        return exact_side(**{key: int(value) for key, value in resources.items()})
    if kind == "wildcard":
        return wildcard_side(
            tuple(spec.get("options", ())),
            int(spec.get("count", 1)),
        )
    if kind == "open":
        return open_side(int(spec.get("count", 1)))
    raise ValueError(f"unknown trade side kind: {kind}")


def _resolve_trade_side(side: TradeSide, exact: dict[str, Any] | None) -> TradeSide:
    if side.kind == "exact":
        if exact is None:
            return side
        candidate = _coerce_trade_side(exact)
        if candidate != side:
            raise ValueError("exact override does not match the original offer side")
        return candidate

    if exact is None:
        raise ValueError(f"{side.kind} trade side requires exact resources")
    candidate = _coerce_trade_side(exact)
    if candidate.kind != "exact":
        raise ValueError("resolved trade side must be exact")
    if not _side_accepts_exact(side, candidate):
        raise ValueError("exact resources do not satisfy original trade side")
    return candidate


def _side_accepts_exact(side: TradeSide, candidate: TradeSide) -> bool:
    total = sum(candidate.resources.values())
    if side.kind == "open":
        return total == side.count
    if side.kind == "wildcard":
        return total == side.count and set(candidate.resources).issubset(side.options)
    return candidate == side


def _trade_side_freqdeck(side: TradeSide) -> tuple[int, int, int, int, int]:
    if side.kind != "exact":
        raise ValueError("trade side must be exact")
    return tuple(int(side.resources.get(resource, 0)) for resource in RESOURCE_NAMES)


def _negotiation_chat_text(offer: dict[str, Any]) -> str:
    give = _format_trade_side(offer["give"])
    want = _format_trade_side(offer["want"])
    target = f" @{offer['target']}" if offer["target"] else ""
    return f"Trade{target}: give {give} for {want}."


def _format_trade_side(side: dict[str, Any]) -> str:
    kind = side["kind"]
    if kind == "open":
        return f"any {side['count']}"
    if kind == "wildcard":
        options = "/".join(side["options"])
        return f"{side['count']} of {options}"
    resources = side["resources"]
    return ", ".join(f"{count} {name}" for name, count in resources.items())
