from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any

try:  # pragma: no cover - fallback exists for dependency-light imports.
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from catan_zero.rl._catanatron import import_catanatron_module
from catan_zero.rl.action_mask import ActionCatalog, MapType
from catan_zero.rl.chat import ColonistChatConfig, ColonistChatState
from catan_zero.rl.gym_env import (
    _coerce_trade_side,
    _format_trade_side,
    _generate_trade_values,
    _resolve_trade_side,
    _serialize_value,
    _trade_side_freqdeck,
)
from catan_zero.rl.negotiation import ColonistNegotiationState
from catan_zero.rl.spaces import make_box, make_discrete
from catan_zero.rl.timers import (
    COLONIST_TIMER_PROFILES,
    TimerProfileName,
    timer_phase_from_prompt,
)


@dataclass(frozen=True, slots=True)
class ColonistMultiAgentConfig:
    map_type: MapType = "BASE"
    players: int = 4
    vps_to_win: int = 10
    dtype: Any = None
    invalid_action_reward: float = -1.0
    max_invalid_actions: int = 10
    max_trade_resources_per_side: int = 2
    max_player_trade_offers_per_turn: int = 3
    enable_table_chat: bool = True
    max_chat_messages_per_turn: int = 4
    max_chat_chars: int = 180
    allow_free_text_chat: bool = True
    enable_timers: bool = True
    timer_profile: TimerProfileName = "normal"
    use_graph_history_features: bool = False


class ColonistMultiAgentEnv:
    """Four-seat Colonist-like env for self-play.

    Unlike `CatanZeroGymEnv`, this does not auto-play opponent seats. Every
    current player decision is surfaced to the caller.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: ColonistMultiAgentConfig | None = None) -> None:
        if np is None:
            raise RuntimeError("numpy is required to instantiate ColonistMultiAgentEnv")
        self.config = config or ColonistMultiAgentConfig()
        if self.config.players not in (2, 3, 4):
            raise ValueError("players must be between 2 and 4")
        if self.config.timer_profile not in COLONIST_TIMER_PROFILES:
            raise ValueError(f"unknown timer_profile: {self.config.timer_profile}")

        self.dtype = np.dtype(self.config.dtype or np.float32)
        self._load_symbols()
        self.players = self._build_players()
        self.player_colors = tuple(player.color for player in self.players)
        self.player_names = tuple(_color_name(color) for color in self.player_colors)
        self.features = tuple(
            self.features_module.get_feature_ordering(
                len(self.players),
                self.config.map_type,
            )
        )
        self.action_catalog = ActionCatalog(self.player_colors, self.config.map_type)
        self._base_action_space_n = self.action_catalog.size
        self._offer_trade_values = tuple(
            _generate_trade_values(self.config.max_trade_resources_per_side)
        )
        self._extended_actions = self._build_extended_actions()
        self.action_space = make_discrete(self._base_action_space_n + len(self._extended_actions))
        observation_size = len(self.features)
        if self.config.use_graph_history_features:
            from catan_zero.rl.graph_history_features import GRAPH_HISTORY_FEATURE_SIZE

            observation_size += GRAPH_HISTORY_FEATURE_SIZE
        self.observation_space = make_box(
            low=0,
            high=19 * 5,
            shape=(observation_size,),
            dtype=self.dtype,
        )

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
        self.game: Any | None = None
        self.invalid_actions_count = 0
        self._trade_offer_counts: dict[tuple[str, tuple[int, int]], int] = {}
        self._event_log: list[dict[str, Any]] = []
        self._replay_frames: list[dict[str, Any]] = []
        self._next_event_id = 1
        self._pending_trade_allowed_responders: tuple[str, ...] | None = None
        self._current_trade_allowed_responders: tuple[str, ...] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = options or {}
        seed = options.get("seed", seed)
        if seed is not None:
            random.seed(seed)
        catan_map = self.map_module.build_map(self.config.map_type)
        for player in self.players:
            player.reset_state()
        self.game = self.Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=int(options.get("vps_to_win", self.config.vps_to_win)),
        )
        self.invalid_actions_count = 0
        self._trade_offer_counts = {}
        self._event_log = []
        self._replay_frames = []
        self._next_event_id = 1
        self._pending_trade_allowed_responders = None
        self._current_trade_allowed_responders = None
        self.chat.reset()
        self.negotiation.reset()
        self._record_event(
            "reset",
            actor=None,
            payload={
                "seed": seed,
                "map_type": self.config.map_type,
                "players": self.player_names,
                "vps_to_win": int(options.get("vps_to_win", self.config.vps_to_win)),
            },
        )
        return self.observations(), self.info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], dict[str, float], bool, bool, dict[str, Any]]:
        self._require_game()
        actor = self.current_player_name()
        catan_action = self._decode_action(int(action))
        valid_actions_before = tuple(self.valid_actions())
        public_legal_action_count_before = self._public_legal_action_count_before(
            valid_actions_before
        )
        if catan_action is None or int(action) not in set(valid_actions_before):
            self.invalid_actions_count += 1
            self._record_event(
                "invalid_action",
                actor=actor,
                payload={"action": int(action)},
            )
            rewards = {
                name: self.config.invalid_action_reward if name == actor else 0.0
                for name in self.player_names
            }
            return (
                self.observations(),
                rewards,
                self.terminated(),
                self.truncated(),
                self.info(),
            )

        action_record = self.game.execute(catan_action)
        if catan_action.action_type.name == "OFFER_TRADE":
            key = (actor, self._turn_key_for_action_record())
            self._trade_offer_counts[key] = self._trade_offer_counts.get(key, 0) + 1
            self._current_trade_allowed_responders = self._pending_trade_allowed_responders
            self._pending_trade_allowed_responders = None
        elif catan_action.action_type.name in ("CONFIRM_TRADE", "CANCEL_TRADE"):
            self._current_trade_allowed_responders = None
        elif not getattr(self.game.state, "is_resolving_trade", False):
            self._current_trade_allowed_responders = None
        self._record_event(
            "board_action",
            actor=actor,
            payload={
                "action_index": int(action),
                "action": self.describe_action(int(action)),
                # Never publish regular turn/discard widths: those depend on
                # hidden hand/dev-card contents. Public prompt widths can
                # safely remove sole-action plumbing from meaningful history.
                **(
                    {
                        "public_legal_action_count_before": (
                            public_legal_action_count_before
                        ),
                        "public_was_sole_legal_action": (
                            public_legal_action_count_before == 1
                        ),
                    }
                    if public_legal_action_count_before is not None
                    else {}
                ),
                "result": _serialize_value(action_record.result),
                "next_player": self.current_player_name()
                if not (self.terminated() or self.truncated())
                else None,
            },
        )
        return (
            self.observations(),
            self.rewards(),
            self.terminated(),
            self.truncated(),
            self.info(),
        )

    def observations(self) -> dict[str, Any]:
        self._require_game()
        return {
            _color_name(color): self._observation_for(color)
            for color in self.player_colors
        }

    def observation_payload(
        self,
        actor: str | None = None,
        *,
        include_event_log: bool = True,
    ) -> dict[str, Any]:
        """Serializable, Colonist-like observation for one seat.

        This is the human/API-friendly surface. It intentionally includes exact
        hand details only for the observing player; opponents expose public
        points, piece state, played development cards, and hidden-card counts.
        """
        self._require_game()
        actor_name = actor or self.current_player_name()
        actor_color = self._color_for_name(actor_name)
        if actor_color is None:
            raise ValueError(f"unknown actor: {actor_name}")
        valid = self.valid_actions(actor_name)
        state = self.game.state
        payload = {
            "actor": actor_name,
            "current_player": self.current_player_name(),
            "current_prompt": state.current_prompt.name,
            # Entity adapter v4 consumes these public rule-state values.  They
            # must be present on the Python payload just as they are in the
            # native Rust snapshot path; omission silently zeroed slots 9:12.
            "is_road_building": bool(state.is_road_building),
            "free_roads_available": int(state.free_roads_available),
            "current_discard_count": int(
                state.discard_counts[state.current_player_index]
            ),
            "players": self._player_payloads(actor_color),
            "board": self._board_payload(),
            "bank": self._bank_payload(),
            "legal_actions": valid,
            "legal_action_descriptions": tuple(
                self.describe_action(action) for action in valid
            ),
            "structured_legal_actions": self.structured_valid_actions(actor_name),
            "action_mask": self.action_mask(actor_name),
            "chat_log": self.chat.log(),
            "negotiation_offers": self.negotiation.offers(),
            "open_negotiation_offers": self.negotiation.open_offers_for(actor_name),
            "trade_panel": self.trade_panel(actor_name),
            "timer": self.timer_info(),
            "replay_frame_count": len(self._replay_frames),
        }
        if include_event_log:
            payload["event_log"] = self.event_log(actor=actor_name)
        return payload

    def observation_payloads(
        self,
        *,
        include_event_log: bool = True,
    ) -> dict[str, dict[str, Any]]:
        self._require_game()
        return {
            name: self.observation_payload(
                name,
                include_event_log=include_event_log,
            )
            for name in self.player_names
        }

    def current_player_name(self) -> str:
        self._require_game()
        return _color_name(self.game.state.current_color())

    def current_player_color(self) -> Any:
        self._require_game()
        return self.game.state.current_color()

    def valid_actions(self, actor: str | None = None) -> tuple[int, ...]:
        self._require_game()
        if actor is not None and actor != self.current_player_name():
            return ()
        valid = list(self.action_catalog.valid_actions(self.game.playable_actions))
        for playable in self.game.playable_actions:
            if self.action_catalog.try_encode(playable) is None:
                valid.extend(self._trade_response_indices_for(playable))
        if self._can_offer_trade():
            valid.extend(self._valid_offer_trade_indices())
        return tuple(sorted(set(valid)))

    def action_mask(self, actor: str | None = None) -> list[bool]:
        valid = set(self.valid_actions(actor))
        return [index in valid for index in range(self.action_space.n)]

    def action_masks(self, actor: str | None = None) -> list[bool]:
        return self.action_mask(actor)

    def sample_valid_action(self, rng: random.Random | None = None) -> int:
        valid = self.valid_actions()
        if not valid:
            raise RuntimeError("no valid actions available")
        chooser = rng.choice if rng else random.choice
        return int(chooser(valid))

    def structured_action(self, action: int | None) -> dict[str, Any] | None:
        description = self.describe_action(action)
        if description is None:
            return None
        action_type = description["action_type"]
        value = description["value"]
        structured = {
            "index": description["index"],
            "action_type": action_type,
            "category": _action_category(action_type),
            "args": self._structured_action_args(action_type, value),
            "raw": description,
        }
        structured["label"] = _structured_action_label(structured)
        return structured

    def structured_valid_actions(
        self,
        actor: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            action
            for index in self.valid_actions(actor)
            for action in [self.structured_action(index)]
            if action is not None
        )

    def action_index_from_structured(self, action: dict[str, Any]) -> int:
        if "index" not in action:
            raise ValueError("structured action must include an index")
        index = int(action["index"])
        if index not in set(self.valid_actions()):
            raise ValueError(f"structured action index is not currently legal: {index}")
        return index

    def step_structured_action(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float], bool, bool, dict[str, Any]]:
        return self.step(self.action_index_from_structured(action))

    def describe_action(self, action: int | None) -> dict[str, Any] | None:
        if action is None:
            return None
        if action < self._base_action_space_n:
            return self.action_catalog.describe(action)
        try:
            kind, value = self._extended_actions[action - self._base_action_space_n]
        except IndexError:
            return None
        return {"index": action, "action_type": kind, "value": _serialize_value(value)}

    def info(self) -> dict[str, Any]:
        self._require_game()
        actor = self.current_player_name()
        valid = self.valid_actions()
        mask = self.action_mask()
        return {
            "current_player": actor,
            "current_prompt": self.game.state.current_prompt.name,
            "valid_actions": valid,
            "legal_action_descriptions": tuple(
                self.describe_action(action) for action in valid
            ),
            "structured_legal_actions": self.structured_valid_actions(actor),
            "action_mask": mask,
            "action_mask_version": "colonist-multiagent-v1",
            "player_names": self.player_names,
            "invalid_actions_count": self.invalid_actions_count,
            "chat_log": self.chat.log(),
            "valid_chat_templates": self.chat.valid_template_ids(actor, self._current_turn_key()),
            "chat_messages_remaining": self.chat.remaining_messages(actor, self._current_turn_key()),
            "negotiation_offers": self.negotiation.offers(),
            "open_negotiation_offers": self.negotiation.open_offers_for(actor),
            "trade_panel": self.trade_panel(actor),
            "player_trade_offers_this_turn": self._trade_offer_count(actor),
            "max_player_trade_offers_per_turn": self.config.max_player_trade_offers_per_turn,
            "timer": self.timer_info(),
            "event_log": self.event_log(),
            "replay_frame_count": len(self._replay_frames),
        }

    def trade_panel(self, actor: str | None = None) -> dict[str, Any]:
        """Colonist-like public trade UI state.

        The real site surfaces who accepted, rejected, countered, or has not
        responded yet. This snapshot gives agents the same public strategic
        signal without making trade responses private side effects.
        """
        self._require_game()
        actor_name = actor or self.current_player_name()
        offers = tuple(
            self._trade_panel_offer(offer, actor_name)
            for offer in self.negotiation.raw_offers()
        )
        return {
            "actor": actor_name,
            "offers": offers,
            "open_offers": tuple(
                offer for offer in offers if offer["status"] == "open"
            ),
            "current_board_trade": self._current_board_trade_payload(),
            "offers_remaining_this_turn": max(
                0,
                self.config.max_player_trade_offers_per_turn
                - self._trade_offer_count(actor_name),
            ),
        }

    def timer_info(self) -> dict[str, Any]:
        prompt_name = self.game.state.current_prompt.name
        action_types = tuple(action.action_type.name for action in self.game.playable_actions)
        phase = timer_phase_from_prompt(prompt_name, action_types)
        timeout_action = self.timeout_action()
        return {
            "enabled": self.config.enable_timers,
            "profile": self.config.timer_profile,
            "prompt": prompt_name,
            "phase": phase,
            "budget_seconds": self._timer_profile.budget_for_phase(phase),
            "timeout_action": timeout_action,
            "timeout_action_description": self.describe_action(timeout_action),
        }

    def timeout_action(self) -> int | None:
        valid = set(self.valid_actions())
        if not valid:
            return None
        for preferred in (("reject_trade", None), ("cancel_trade", None)):
            action = self._extended_action_index(*preferred)
            if action in valid:
                return action
        for action_type in ("ROLL", "DISCARD_RESOURCE", "MOVE_ROBBER", "END_TURN"):
            matches = [
                index
                for index in valid
                if index < self._base_action_space_n
                and self.action_catalog.raw_entry(index)[0].name == action_type
            ]
            if matches:
                return min(matches)
        return min(valid)

    def step_timeout(self) -> tuple[dict[str, Any], dict[str, float], bool, bool, dict[str, Any]]:
        action = self.timeout_action()
        if action is None:
            raise RuntimeError("no timeout action available")
        self._record_event(
            "timeout",
            actor=self.current_player_name(),
            payload={"timer": self.timer_info(), "action": self.describe_action(action)},
        )
        return self.step(action)

    def post_chat(
        self,
        text: str,
        *,
        actor: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        actor_name = actor or self.current_player_name()
        message = self.chat.post_text(
            actor=actor_name,
            text=text,
            turn_key=self._current_turn_key(),
            metadata=metadata,
        )
        self._record_event("chat", actor=actor_name, payload=message.to_dict())
        return message.to_dict()

    def post_chat_template(
        self,
        template_id: str,
        values: dict[str, Any] | None = None,
        *,
        actor: str | None = None,
    ) -> dict[str, Any]:
        actor_name = actor or self.current_player_name()
        message = self.chat.post_template(
            actor=actor_name,
            template_id=template_id,
            values=values,
            turn_key=self._current_turn_key(),
        )
        self._record_event("chat", actor=actor_name, payload=message.to_dict())
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
        actor_name = actor or self.current_player_name()
        offer = self.negotiation.create_offer(
            actor=actor_name,
            turn_key=self._current_turn_key(),
            target=target,
            give=_coerce_trade_side(give),
            want=_coerce_trade_side(want),
            metadata=metadata,
        )
        self._post_negotiation_chat(offer.to_dict())
        self._record_event("trade_proposal", actor=actor_name, payload=offer.to_dict())
        return offer.to_dict()

    def respond_to_trade(
        self,
        offer_id: int,
        status: str,
        *,
        actor: str | None = None,
    ) -> dict[str, Any]:
        offer = self.negotiation.respond(
            offer_id=offer_id,
            actor=actor or self.current_player_name(),
            status=status,  # type: ignore[arg-type]
        )
        self._record_event(
            "trade_response",
            actor=actor or self.current_player_name(),
            payload=offer.to_dict(),
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
        actor_name = actor or self.current_player_name()
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
        self._record_event(
            "trade_counteroffer",
            actor=actor_name,
            payload=counter.to_dict(),
        )
        return counter.to_dict()

    def trade_action_for_offer(
        self,
        offer_id: int,
        *,
        give: dict[str, Any] | None = None,
        want: dict[str, Any] | None = None,
    ) -> int | None:
        offer = self.negotiation.get_offer(offer_id)
        exact_give = _resolve_trade_side(offer.give, give)
        exact_want = _resolve_trade_side(offer.want, want)
        action_value = (
            *_trade_side_freqdeck(exact_give),
            *_trade_side_freqdeck(exact_want),
        )
        for offset, (kind, value) in enumerate(self._extended_actions):
            action = self._base_action_space_n + offset
            if kind == "offer_trade" and value == action_value:
                return action if action in set(self.valid_actions()) else None
        return None

    def step_negotiated_trade(
        self,
        offer_id: int,
        *,
        give: dict[str, Any] | None = None,
        want: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float], bool, bool, dict[str, Any]]:
        action = self.trade_action_for_offer(offer_id, give=give, want=want)
        if action is None:
            raise ValueError("negotiation offer does not resolve to a valid trade action")
        offer = self.negotiation.get_offer(offer_id)
        self._pending_trade_allowed_responders = (offer.target,) if offer.target else None
        return self.step(action)

    def rewards(self) -> dict[str, float]:
        winner = self.game.winning_color()
        if winner is None:
            return {name: 0.0 for name in self.player_names}
        return {
            _color_name(color): 1.0
            if color == winner
            else -1.0 / (len(self.players) - 1)
            for color in self.player_colors
        }

    def terminated(self) -> bool:
        self._require_game()
        return self.game.winning_color() is not None

    def truncated(self) -> bool:
        self._require_game()
        return (
            self.game.state.num_turns >= self.turns_limit
            or self.invalid_actions_count > self.config.max_invalid_actions
        )

    def close(self) -> None:
        self.game = None

    def event_log(self, *, actor: str | None = None) -> tuple[dict[str, Any], ...]:
        return tuple(self._redact_event(event, actor) for event in self._event_log)

    def replay_trace(self, *, actor: str | None = None) -> tuple[dict[str, Any], ...]:
        """Replay frames with redacted events and safe per-seat observations."""
        return tuple(self._redact_replay_frame(frame, actor) for frame in self._replay_frames)

    def write_replay_jsonl(
        self,
        path: str | Path,
        *,
        actor: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        from catan_zero.rl.replay import dump_replay_jsonl

        return dump_replay_jsonl(
            self.replay_trace(actor=actor),
            path,
            metadata={
                "map_type": self.config.map_type,
                "players": self.player_names,
                **(metadata or {}),
            },
        )

    def _decode_action(self, action: int) -> Any | None:
        color = self.current_player_color()
        if action < self._base_action_space_n:
            try:
                return self.action_catalog.decode(action, color)
            except (IndexError, ValueError):
                return None
        offset = action - self._base_action_space_n
        try:
            kind, value = self._extended_actions[offset]
        except IndexError:
            return None
        current_trade = self.game.state.current_trade
        if kind == "offer_trade":
            return self.Action(color, self.ActionType.OFFER_TRADE, value)
        if kind == "accept_trade":
            return self.Action(color, self.ActionType.ACCEPT_TRADE, current_trade)
        if kind == "reject_trade":
            return self.Action(color, self.ActionType.REJECT_TRADE, current_trade)
        if kind == "cancel_trade":
            return self.Action(color, self.ActionType.CANCEL_TRADE, None)
        if kind == "confirm_trade":
            return self.Action(
                color,
                self.ActionType.CONFIRM_TRADE,
                (*current_trade[:10], value),
            )
        return None

    def _can_offer_trade(self) -> bool:
        if (
            self._trade_offer_count(self.current_player_name())
            >= self.config.max_player_trade_offers_per_turn
        ):
            return False
        state = self.game.state
        color = self.current_player_color()
        if state.current_color() != color:
            return False
        return any(self._is_valid_offer_value(value) for value in self._offer_trade_values)

    def _valid_offer_trade_indices(self) -> list[int]:
        valid: list[int] = []
        for offset, (kind, value) in enumerate(self._extended_actions):
            if kind == "offer_trade" and self._is_valid_offer_value(value):
                valid.append(self._base_action_space_n + offset)
        return valid

    def _is_valid_offer_value(self, value: tuple[int, ...]) -> bool:
        action = self.Action(
            self.current_player_color(),
            self.ActionType.OFFER_TRADE,
            value,
        )
        return bool(
            self.is_valid_action(self.game.playable_actions, self.game.state, action)
            and self.player_resource_freqdeck_contains(
                self.game.state,
                self.current_player_color(),
                value[:5],
            )
        )

    def _trade_response_indices_for(self, catan_action: Any) -> list[int]:
        result: list[int] = []
        current_name = self.current_player_name()
        is_targeted_out = (
            self._current_trade_allowed_responders is not None
            and current_name not in self._current_trade_allowed_responders
        )
        for offset, (kind, value) in enumerate(self._extended_actions):
            index = self._base_action_space_n + offset
            if (
                catan_action.action_type == self.ActionType.ACCEPT_TRADE
                and kind == "accept_trade"
            ):
                if not is_targeted_out:
                    result.append(index)
            elif (
                catan_action.action_type == self.ActionType.REJECT_TRADE
                and kind == "reject_trade"
            ):
                result.append(index)
            elif (
                catan_action.action_type == self.ActionType.CANCEL_TRADE
                and kind == "cancel_trade"
            ):
                result.append(index)
            elif (
                catan_action.action_type == self.ActionType.CONFIRM_TRADE
                and kind == "confirm_trade"
                and len(catan_action.value) == 11
                and catan_action.value[10] == value
            ):
                result.append(index)
        return result

    def _build_extended_actions(self) -> tuple[tuple[str, Any], ...]:
        return (
            *(("offer_trade", value) for value in self._offer_trade_values),
            ("accept_trade", None),
            ("reject_trade", None),
            ("cancel_trade", None),
            *(("confirm_trade", color) for color in self.player_colors),
        )

    def _observation_for(self, color: Any) -> Any:
        sample = self.features_module.create_sample(self.game, color)
        base = np.asarray([sample[name] for name in self.features], dtype=self.dtype)
        if not self.config.use_graph_history_features:
            return base
        from catan_zero.rl.graph_history_features import build_graph_history_feature_vector

        suffix = build_graph_history_feature_vector(
            self,
            _color_name(color),
        ).astype(self.dtype, copy=False)
        return np.concatenate((base, suffix)).astype(self.dtype, copy=False)

    def _structured_action_args(self, action_type: str, value: Any) -> dict[str, Any]:
        if action_type in ("ROLL", "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING", "END_TURN"):
            return {}
        if action_type == "DISCARD_RESOURCE":
            return {"resource": _resource_api_name(value)}
        if action_type == "BUILD_ROAD":
            return {"edge": tuple(value)}
        if action_type in ("BUILD_SETTLEMENT", "BUILD_CITY"):
            return {"node": value}
        if action_type == "PLAY_YEAR_OF_PLENTY":
            return {"resources": tuple(_resource_api_name(resource) for resource in value)}
        if action_type == "PLAY_MONOPOLY":
            return {"resource": _resource_api_name(value)}
        if action_type == "MOVE_ROBBER":
            coordinate, victim = value
            return {
                "tile_coordinate": tuple(coordinate),
                "victim": victim,
            }
        if action_type == "MARITIME_TRADE":
            return {
                "trade_kind": "bank_or_port",
                "give": _resource_list_to_bundle(value[:4]),
                "want": _resource_list_to_bundle(value[4:]),
            }
        if action_type == "offer_trade":
            return {
                "trade_kind": "player_offer",
                "give": _freqdeck_to_resource_bundle(value[:5], self.RESOURCES),
                "want": _freqdeck_to_resource_bundle(value[5:10], self.RESOURCES),
            }
        if action_type in ("accept_trade", "reject_trade", "cancel_trade"):
            return {"trade_kind": "player_response"}
        if action_type == "confirm_trade":
            return {
                "trade_kind": "player_confirm",
                "target": _color_name(value),
            }
        return {"value": value}

    def _trade_panel_offer(self, offer: Any, actor_name: str) -> dict[str, Any]:
        eligible = self._eligible_trade_responders(offer)
        responder_statuses = {
            responder: offer.responses.get(
                responder,
                "waiting" if offer.status == "open" else offer.status,
            )
            for responder in eligible
        }
        accepted = tuple(
            player for player, status in responder_statuses.items() if status == "accepted"
        )
        rejected = tuple(
            player for player, status in responder_statuses.items() if status == "rejected"
        )
        countered = tuple(
            player for player, status in responder_statuses.items() if status == "countered"
        )
        actor_response = responder_statuses.get(actor_name)
        return {
            **offer.to_dict(),
            "eligible_responders": eligible,
            "responder_statuses": responder_statuses,
            "accepted_players": accepted,
            "rejected_players": rejected,
            "countered_players": countered,
            "waiting_players": tuple(
                player
                for player, status in responder_statuses.items()
                if status == "waiting"
            ),
            "can_accept": actor_response == "waiting",
            "can_reject": actor_response == "waiting",
            "can_counter": (
                offer.status == "open"
                and actor_name != offer.actor
                and actor_name in eligible
            ),
            "can_confirm": (
                offer.status == "open"
                and actor_name == offer.actor
                and bool(accepted)
            ),
            "can_cancel": offer.status == "open" and actor_name == offer.actor,
        }

    def _eligible_trade_responders(self, offer: Any) -> tuple[str, ...]:
        if offer.target is not None:
            return (offer.target,) if offer.target != offer.actor else ()
        return tuple(player for player in self.player_names if player != offer.actor)

    def _current_board_trade_payload(self) -> dict[str, Any] | None:
        current_trade = getattr(self.game.state, "current_trade", None)
        if not current_trade:
            return None
        return {
            "trade": _serialize_value(current_trade),
            "allowed_responders": self._current_trade_allowed_responders,
            "is_resolving_trade": bool(
                getattr(self.game.state, "is_resolving_trade", False)
            ),
        }

    def _player_payloads(self, actor_color: Any) -> dict[str, dict[str, Any]]:
        state = self.game.state
        payloads: dict[str, dict[str, Any]] = {}
        for color in self.player_colors:
            name = _color_name(color)
            key = self.player_key(state, color)
            player_payload: dict[str, Any] = {
                "public_victory_points": state.player_state[f"{key}_VICTORY_POINTS"],
                "resource_card_count": self.player_num_resource_cards(state, color),
                "development_card_count": self.player_num_dev_cards(state, color),
                "has_largest_army": state.player_state[f"{key}_HAS_ARMY"],
                "has_longest_road": state.player_state[f"{key}_HAS_ROAD"],
                "roads_left": state.player_state[f"{key}_ROADS_AVAILABLE"],
                "settlements_left": state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"],
                "cities_left": state.player_state[f"{key}_CITIES_AVAILABLE"],
                "has_rolled": state.player_state[f"{key}_HAS_ROLLED"],
                "longest_road_length": state.player_state[
                    f"{key}_LONGEST_ROAD_LENGTH"
                ],
                "played_development_cards": {
                    card: state.player_state[f"{key}_PLAYED_{card}"]
                    for card in self.DEVELOPMENT_CARDS
                    if card != self.VICTORY_POINT
                },
            }
            if color == actor_color:
                player_payload["actual_victory_points"] = state.player_state[
                    f"{key}_ACTUAL_VICTORY_POINTS"
                ]
                player_payload["resources"] = {
                    resource: state.player_state[f"{key}_{resource}_IN_HAND"]
                    for resource in self.RESOURCES
                }
                player_payload["development_cards"] = {
                    card: state.player_state[f"{key}_{card}_IN_HAND"]
                    for card in self.DEVELOPMENT_CARDS
                }
                # Keep this actor-private/public-rule surface identical to the
                # native Rust snapshot adapter consumed by entity adapter v4.
                # The old ``..._this_turn`` spelling was not read by
                # ``_global_tokens`` and silently forced slot 8 to zero on the
                # Python feature path.
                player_payload["has_played_development_card_in_turn"] = (
                    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
                )
                # The vendored and native engines both retain the exact number
                # held at the turn boundary. Cards bought later increase the
                # hand without aging into this count. Exclude hidden VP cards,
                # which have no play action and are intentionally not part of
                # the four-slot public-rule feature contract.
                player_payload["playable_development_cards"] = {
                    card: int(state.player_state[f"{key}_{card}_OWNED_AT_START"])
                    for card in self.DEVELOPMENT_CARDS
                    if card != self.VICTORY_POINT
                }
            payloads[name] = player_payload
        return payloads

    def _board_payload(self) -> dict[str, Any]:
        board = self.game.state.board
        return {
            "robber_coordinate": _serialize_value(board.robber_coordinate),
            "tiles": tuple(
                {
                    "tile_id": tile_id,
                    "coordinate": _serialize_value(coordinate),
                    "resource": tile.resource,
                    "number": tile.number,
                    "has_robber": board.map.tiles[board.robber_coordinate] == tile,
                    "nodes": {
                        node_ref.name: node_id
                        for node_ref, node_id in tile.nodes.items()
                    },
                    "edges": {
                        edge_ref.name: tuple(sorted(edge))
                        for edge_ref, edge in tile.edges.items()
                    },
                }
                for coordinate, tile in sorted(
                    board.map.land_tiles.items(), key=lambda item: item[1].id
                )
                for tile_id in [tile.id]
            ),
            "ports": tuple(
                {
                    "port_id": port_id,
                    "resource": port.resource,
                    "nodes": tuple(sorted(port.nodes.values())),
                }
                for port_id, port in sorted(board.map.ports_by_id.items())
            ),
            "buildings": tuple(
                {
                    "node": node_id,
                    "player": _color_name(building[0]),
                    "building_type": building[1],
                }
                for node_id, building in sorted(board.buildings.items())
            ),
            "roads": tuple(
                {
                    "edge": tuple(sorted(edge)),
                    "player": _color_name(color),
                }
                for edge, color in sorted(board.roads.items())
                if edge[0] < edge[1]
            ),
        }

    def _bank_payload(self) -> dict[str, Any]:
        return {
            "resources": {
                resource: self.freqdeck_count(
                    self.game.state.resource_freqdeck, resource
                )
                for resource in self.RESOURCES
            },
            "development_cards_remaining": len(self.game.state.development_listdeck),
        }

    def _trade_offer_count(self, actor: str) -> int:
        return self._trade_offer_counts.get((actor, self._current_turn_key()), 0)

    def _current_turn_key(self) -> tuple[int, int]:
        state = self.game.state
        return int(state.num_turns), int(state.current_turn_index)

    def _public_legal_action_count_before(
        self,
        valid_actions: tuple[int, ...],
    ) -> int | None:
        """Return an exact width only when it cannot reveal hidden cards."""

        state = self.game.state
        prompt = state.current_prompt.name
        if prompt in {
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "MOVE_ROBBER",
        } or (prompt == "PLAY_TURN" and bool(state.is_road_building)):
            return len(valid_actions)
        return None

    def _turn_key_for_action_record(self) -> tuple[int, int]:
        state = self.game.state
        return int(state.num_turns), int(state.current_turn_index)

    def _extended_action_index(self, kind: str, value: Any) -> int | None:
        for offset, candidate in enumerate(self._extended_actions):
            if candidate == (kind, value):
                return self._base_action_space_n + offset
        return None

    def _post_negotiation_chat(self, offer: dict[str, Any]) -> None:
        if not self.config.enable_table_chat:
            return
        try:
            message = self.chat.post_text(
                actor=offer["actor"],
                text=_negotiation_chat_text(offer),
                turn_key=self._current_turn_key(),
                intent="trade_negotiation",
                metadata={"offer_id": offer["offer_id"]},
            )
            self._record_event("chat", actor=offer["actor"], payload=message.to_dict())
        except ValueError:
            return

    def _record_event(self, event_type: str, *, actor: str | None, payload: dict[str, Any]) -> None:
        event = {
            "event_id": self._next_event_id,
            "event_type": event_type,
            "turn_key": self._current_turn_key() if self.game is not None else None,
            "actor": actor,
            "payload": payload,
        }
        self._event_log.append(event)
        self._next_event_id += 1
        if self.game is not None:
            self._replay_frames.append(
                {
                    "frame_id": len(self._replay_frames) + 1,
                    "event": event,
                    "observations": self.observation_payloads(
                        include_event_log=False,
                    ),
                    "rewards": self.rewards(),
                    "terminated": self.terminated(),
                    "truncated": self.truncated(),
                }
            )

    def _redact_event(
        self,
        event: dict[str, Any],
        actor: str | None,
    ) -> dict[str, Any]:
        redacted = dict(event)
        payload = dict(redacted.get("payload", {}))
        action = payload.get("action")
        action_type = action.get("action_type") if isinstance(action, dict) else None
        if action_type == "BUY_DEVELOPMENT_CARD":
            payload["result"] = "hidden_development_card"
        elif action_type == "MOVE_ROBBER" and payload.get("result") is not None:
            payload["result"] = "hidden_stolen_resource"
        elif action_type == "DISCARD_RESOURCE":
            if isinstance(action, dict):
                action = dict(action)
                # The flat action index is itself private here: each
                # DISCARD_RESOURCE catalog entry is keyed by the discarded
                # resource. Redacting only ``value`` while preserving
                # ``index`` leaks the exact card through entity event-token
                # slot 35 (``_event_action_id``).
                action["index"] = None
                action["value"] = "hidden_resource"
                payload["action"] = action
            payload["action_index"] = None
            payload["result"] = "hidden_resource"
        redacted["payload"] = payload
        return redacted

    def _redact_replay_frame(
        self,
        frame: dict[str, Any],
        actor: str | None,
    ) -> dict[str, Any]:
        redacted = dict(frame)
        redacted["event"] = self._redact_event(frame["event"], actor)
        redacted["observations"] = {
            player: dict(observation)
            for player, observation in frame["observations"].items()
        }
        return redacted

    def _load_symbols(self) -> None:
        game_module = import_catanatron_module("catanatron.game")
        map_module = import_catanatron_module("catanatron.models.map")
        player_module = import_catanatron_module("catanatron.models.player")
        features_module = import_catanatron_module("catanatron.features")
        state_functions = import_catanatron_module("catanatron.state_functions")
        enums_module = import_catanatron_module("catanatron.models.enums")
        decks_module = import_catanatron_module("catanatron.models.decks")

        self.Game = game_module.Game
        self.is_valid_action = game_module.is_valid_action
        self.turns_limit = int(game_module.TURNS_LIMIT)
        self.map_module = map_module
        self.Player = player_module.Player
        self.Color = player_module.Color
        self.features_module = features_module
        self.player_resource_freqdeck_contains = state_functions.player_resource_freqdeck_contains
        self.player_key = state_functions.player_key
        self.player_num_resource_cards = state_functions.player_num_resource_cards
        self.player_num_dev_cards = state_functions.player_num_dev_cards
        self.freqdeck_count = decks_module.freqdeck_count
        self.Action = enums_module.Action
        self.ActionType = enums_module.ActionType
        self.RESOURCES = tuple(enums_module.RESOURCES)
        self.DEVELOPMENT_CARDS = tuple(enums_module.DEVELOPMENT_CARDS)
        self.VICTORY_POINT = enums_module.VICTORY_POINT

    def _build_players(self) -> list[Any]:
        colors = [self.Color.BLUE, self.Color.RED, self.Color.ORANGE, self.Color.WHITE]
        return [self.Player(color) for color in colors[: self.config.players]]

    def _require_game(self) -> None:
        if self.game is None:
            raise RuntimeError("environment has not been reset")

    def _color_for_name(self, name: str) -> Any | None:
        for color in self.player_colors:
            if _color_name(color) == name:
                return color
        return None


def _color_name(color: Any) -> str:
    return getattr(color, "name", str(color))


def _negotiation_chat_text(offer: dict[str, Any]) -> str:
    give = _format_trade_side(offer["give"])
    want = _format_trade_side(offer["want"])
    target = f" @{offer['target']}" if offer["target"] else ""
    return f"Trade{target}: give {give} for {want}."


def _action_category(action_type: str) -> str:
    if action_type in ("offer_trade", "accept_trade", "reject_trade", "cancel_trade", "confirm_trade", "MARITIME_TRADE"):
        return "trade"
    if action_type in ("BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"):
        return "build"
    if action_type in ("BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING"):
        return "development"
    if action_type in ("MOVE_ROBBER", "DISCARD_RESOURCE"):
        return "robber"
    return "turn"


def _structured_action_label(action: dict[str, Any]) -> str:
    action_type = action["action_type"]
    args = action["args"]
    if action_type == "BUILD_ROAD":
        return f"Build road on edge {args['edge']}"
    if action_type == "BUILD_SETTLEMENT":
        return f"Build settlement on node {args['node']}"
    if action_type == "BUILD_CITY":
        return f"Build city on node {args['node']}"
    if action_type == "DISCARD_RESOURCE":
        return f"Discard {args['resource']}"
    if action_type == "MOVE_ROBBER":
        victim = args["victim"] or "no victim"
        return f"Move robber to {args['tile_coordinate']} and steal from {victim}"
    if action_type in ("MARITIME_TRADE", "offer_trade"):
        return f"Trade give {args['give']} for {args['want']}"
    if action_type == "confirm_trade":
        return f"Confirm trade with {args['target']}"
    return action_type.lower().replace("_", " ")


def _resource_api_name(resource: Any) -> str | None:
    if resource is None:
        return None
    return str(resource).lower()


def _resource_list_to_bundle(resources: Any) -> dict[str, int]:
    bundle: dict[str, int] = {}
    for resource in resources:
        name = _resource_api_name(resource)
        if name is None:
            continue
        bundle[name] = bundle.get(name, 0) + 1
    return bundle


def _freqdeck_to_resource_bundle(
    freqdeck: Any,
    resource_order: tuple[Any, ...],
) -> dict[str, int]:
    return {
        name: count
        for name, count in (
            (_resource_api_name(resource), int(amount))
            for resource, amount in zip(resource_order, freqdeck)
        )
        if name is not None and count > 0
    }
