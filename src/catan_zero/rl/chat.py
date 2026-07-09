from __future__ import annotations

from dataclasses import dataclass, field
from string import Formatter
from typing import Any, Literal

ChatChannel = Literal["table", "system"]


@dataclass(frozen=True, slots=True)
class ColonistChatTemplate:
    """A safe, structured table-chat utterance.

    Colonist-style chat is strategically important, but it should be modeled as
    a bounded side channel. Templates give RL/LLM layers stable intents while
    still allowing a free-text adapter for human-facing experiments.
    """

    template_id: str
    intent: str
    text: str
    required_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ColonistChatConfig:
    enabled: bool = True
    max_chars: int = 180
    max_messages_per_turn: int = 4
    allow_free_text: bool = True


@dataclass(frozen=True, slots=True)
class ColonistChatMessage:
    message_id: int
    turn_key: tuple[int, int]
    actor: str
    channel: ChatChannel
    text: str
    intent: str = "free_text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "turn_key": self.turn_key,
            "actor": self.actor,
            "channel": self.channel,
            "text": self.text,
            "intent": self.intent,
            "metadata": dict(self.metadata),
        }


DEFAULT_CHAT_TEMPLATES: tuple[ColonistChatTemplate, ...] = (
    ColonistChatTemplate("gl", "social", "gl hf"),
    ColonistChatTemplate("gg", "social", "gg"),
    ColonistChatTemplate("nice", "reaction", "Nice."),
    ColonistChatTemplate("sorry", "reaction", "Sorry."),
    ColonistChatTemplate("oops", "reaction", "Oops."),
    ColonistChatTemplate("no_thanks", "trade_response", "No thanks."),
    ColonistChatTemplate("accept_trade", "trade_response", "I accept."),
    ColonistChatTemplate("reject_trade", "trade_response", "I reject."),
    ColonistChatTemplate("good_trade", "trade_response", "Good trade."),
    ColonistChatTemplate(
        "same_offer",
        "trade_response",
        "I will take that trade too.",
    ),
    ColonistChatTemplate(
        "cannot_trade",
        "trade_response",
        "I cannot make that trade.",
    ),
    ColonistChatTemplate(
        "trade_request",
        "trade_request",
        "Looking for {want}; can offer {give}.",
        ("want", "give"),
    ),
    ColonistChatTemplate(
        "open_offer",
        "open_ended_trade",
        "I can give {give} for anything useful.",
        ("give",),
    ),
    ColonistChatTemplate(
        "open_request",
        "open_ended_trade",
        "I need {want}; what do you want for it?",
        ("want",),
    ),
    ColonistChatTemplate(
        "counteroffer",
        "counteroffer",
        "Counter: {give} for {want}.",
        ("give", "want"),
    ),
    ColonistChatTemplate(
        "leader_block",
        "table_strategy",
        "We should slow down {player}.",
        ("player",),
    ),
    ColonistChatTemplate(
        "do_not_trade_leader",
        "table_strategy",
        "Do not trade with {player}.",
        ("player",),
    ),
    ColonistChatTemplate(
        "robber_spare",
        "robber_negotiation",
        "Please do not block me; I can trade {offer}.",
        ("offer",),
    ),
    ColonistChatTemplate(
        "robber_extort",
        "robber_negotiation",
        "If you can trade {offer}, I can avoid blocking {tile}.",
        ("offer", "tile"),
    ),
    ColonistChatTemplate(
        "robber_no_steal",
        "robber_negotiation",
        "I can avoid stealing from {player} for {offer}.",
        ("player", "offer"),
    ),
)


class ColonistChatState:
    """Bounded public chat log for Colonist-like negotiation."""

    def __init__(
        self,
        config: ColonistChatConfig | None = None,
        templates: tuple[ColonistChatTemplate, ...] = DEFAULT_CHAT_TEMPLATES,
    ) -> None:
        self.config = config or ColonistChatConfig()
        self.templates = {template.template_id: template for template in templates}
        self._messages: list[ColonistChatMessage] = []
        self._counts_by_actor_turn: dict[tuple[str, tuple[int, int]], int] = {}
        self._next_message_id = 1

    def reset(self) -> None:
        self._messages = []
        self._counts_by_actor_turn = {}
        self._next_message_id = 1

    def log(self) -> tuple[dict[str, Any], ...]:
        return tuple(message.to_dict() for message in self._messages)

    def remaining_messages(self, actor: str, turn_key: tuple[int, int]) -> int:
        used = self._counts_by_actor_turn.get((actor, turn_key), 0)
        return max(0, self.config.max_messages_per_turn - used)

    def valid_template_ids(self, actor: str, turn_key: tuple[int, int]) -> tuple[str, ...]:
        if not self.config.enabled or self.remaining_messages(actor, turn_key) <= 0:
            return ()
        return tuple(sorted(self.templates))

    def post_text(
        self,
        *,
        actor: str,
        text: str,
        turn_key: tuple[int, int],
        channel: ChatChannel = "table",
        intent: str = "free_text",
        metadata: dict[str, Any] | None = None,
    ) -> ColonistChatMessage:
        if not self.config.enabled:
            raise ValueError("chat is disabled")
        if not self.config.allow_free_text and intent == "free_text":
            raise ValueError("free-text chat is disabled")
        clean_text = self._sanitize(text)
        self._enforce_rate_limit(actor, turn_key)
        return self._append(
            actor=actor,
            text=clean_text,
            turn_key=turn_key,
            channel=channel,
            intent=intent,
            metadata=metadata or {},
        )

    def post_template(
        self,
        *,
        actor: str,
        template_id: str,
        turn_key: tuple[int, int],
        values: dict[str, Any] | None = None,
    ) -> ColonistChatMessage:
        try:
            template = self.templates[template_id]
        except KeyError as exc:
            raise ValueError(f"unknown chat template: {template_id}") from exc

        values = values or {}
        missing = [field for field in template.required_fields if field not in values]
        if missing:
            raise ValueError(f"missing template fields: {', '.join(missing)}")
        allowed_fields = {field_name for _, field_name, _, _ in Formatter().parse(template.text) if field_name}
        safe_values = {
            key: self._sanitize(str(value))
            for key, value in values.items()
            if key in allowed_fields
        }
        text = template.text.format(**safe_values)
        return self.post_text(
            actor=actor,
            text=text,
            turn_key=turn_key,
            intent=template.intent,
            metadata={"template_id": template.template_id, "values": safe_values},
        )

    def post_system(
        self,
        *,
        text: str,
        turn_key: tuple[int, int],
        metadata: dict[str, Any] | None = None,
    ) -> ColonistChatMessage:
        return self.post_text(
            actor="system",
            text=text,
            turn_key=turn_key,
            channel="system",
            intent="system_log",
            metadata=metadata,
        )

    def _append(
        self,
        *,
        actor: str,
        text: str,
        turn_key: tuple[int, int],
        channel: ChatChannel,
        intent: str,
        metadata: dict[str, Any],
    ) -> ColonistChatMessage:
        message = ColonistChatMessage(
            message_id=self._next_message_id,
            turn_key=turn_key,
            actor=actor,
            channel=channel,
            text=text,
            intent=intent,
            metadata=metadata,
        )
        self._next_message_id += 1
        self._messages.append(message)
        key = (actor, turn_key)
        self._counts_by_actor_turn[key] = self._counts_by_actor_turn.get(key, 0) + 1
        return message

    def _enforce_rate_limit(self, actor: str, turn_key: tuple[int, int]) -> None:
        if self.remaining_messages(actor, turn_key) <= 0:
            raise ValueError("chat message limit reached for this turn")

    def _sanitize(self, text: str) -> str:
        clean = " ".join(text.strip().split())
        if not clean:
            raise ValueError("chat message cannot be empty")
        if len(clean) > self.config.max_chars:
            clean = clean[: self.config.max_chars].rstrip()
        return clean
