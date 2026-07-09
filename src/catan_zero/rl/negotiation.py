from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ResourceName = Literal["wood", "brick", "sheep", "wheat", "ore"]
TradeSideKind = Literal["exact", "wildcard", "open"]
TradeStatus = Literal[
    "open",
    "accepted",
    "rejected",
    "countered",
    "expired",
    "confirmed",
    "cancelled",
]

RESOURCE_NAMES: tuple[ResourceName, ...] = ("wood", "brick", "sheep", "wheat", "ore")


@dataclass(frozen=True, slots=True)
class TradeSide:
    """One side of a Colonist-style trade proposal.

    `exact` is a concrete resource bundle. `wildcard` means the proposer accepts
    one of several resource alternatives. `open` represents Colonist's
    open-ended "make me an offer" workflow before a concrete board trade exists.
    """

    kind: TradeSideKind
    resources: dict[ResourceName, int] = field(default_factory=dict)
    options: tuple[ResourceName, ...] = ()
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "resources": dict(self.resources),
            "options": self.options,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class NegotiationOffer:
    offer_id: int
    turn_key: tuple[int, int]
    actor: str
    target: str | None
    give: TradeSide
    want: TradeSide
    status: TradeStatus = "open"
    responses: dict[str, TradeStatus] = field(default_factory=dict)
    parent_offer_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offer_id": self.offer_id,
            "turn_key": self.turn_key,
            "actor": self.actor,
            "target": self.target,
            "give": self.give.to_dict(),
            "want": self.want.to_dict(),
            "status": self.status,
            "responses": dict(self.responses),
            "parent_offer_id": self.parent_offer_id,
            "metadata": dict(self.metadata),
        }


class ColonistNegotiationState:
    """Public trade-workflow state for Colonist-like RL training.

    This does not mutate Catan resources. It models the negotiation layer that
    produces concrete trades, counteroffers, and training targets.
    """

    def __init__(self) -> None:
        self._offers: list[NegotiationOffer] = []
        self._next_offer_id = 1

    def reset(self) -> None:
        self._offers = []
        self._next_offer_id = 1

    def offers(self) -> tuple[dict[str, Any], ...]:
        return tuple(offer.to_dict() for offer in self._offers)

    def raw_offers(self) -> tuple[NegotiationOffer, ...]:
        return tuple(self._offers)

    def get_offer(self, offer_id: int) -> NegotiationOffer:
        return self._find_offer(offer_id)

    def open_offers_for(self, actor: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            offer.to_dict()
            for offer in self._offers
            if offer.status == "open" and (offer.target is None or offer.target == actor)
        )

    def create_offer(
        self,
        *,
        actor: str,
        turn_key: tuple[int, int],
        give: TradeSide,
        want: TradeSide,
        target: str | None = None,
        parent_offer_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NegotiationOffer:
        offer = NegotiationOffer(
            offer_id=self._next_offer_id,
            turn_key=turn_key,
            actor=actor,
            target=target,
            give=give,
            want=want,
            status="open",
            parent_offer_id=parent_offer_id,
            metadata=metadata or {},
        )
        self._next_offer_id += 1
        self._offers.append(offer)
        return offer

    def respond(
        self,
        *,
        offer_id: int,
        actor: str,
        status: TradeStatus,
    ) -> NegotiationOffer:
        if status not in ("accepted", "rejected", "countered"):
            raise ValueError(f"invalid response status: {status}")
        offer = self._find_offer(offer_id)
        responses = dict(offer.responses)
        responses[actor] = status
        return self._replace_offer(offer, responses=responses)

    def update_status(self, offer_id: int, status: TradeStatus) -> NegotiationOffer:
        if status not in ("expired", "confirmed", "cancelled"):
            raise ValueError(f"invalid terminal status: {status}")
        offer = self._find_offer(offer_id)
        return self._replace_offer(offer, status=status)

    def _find_offer(self, offer_id: int) -> NegotiationOffer:
        for offer in self._offers:
            if offer.offer_id == offer_id:
                return offer
        raise ValueError(f"unknown offer_id: {offer_id}")

    def _replace_offer(self, offer: NegotiationOffer, **changes: Any) -> NegotiationOffer:
        updated = NegotiationOffer(
            offer_id=changes.get("offer_id", offer.offer_id),
            turn_key=changes.get("turn_key", offer.turn_key),
            actor=changes.get("actor", offer.actor),
            target=changes.get("target", offer.target),
            give=changes.get("give", offer.give),
            want=changes.get("want", offer.want),
            status=changes.get("status", offer.status),
            responses=changes.get("responses", offer.responses),
            parent_offer_id=changes.get("parent_offer_id", offer.parent_offer_id),
            metadata=changes.get("metadata", offer.metadata),
        )
        self._offers = [updated if item.offer_id == offer.offer_id else item for item in self._offers]
        return updated


def exact_side(**resources: int) -> TradeSide:
    return TradeSide(
        kind="exact",
        resources=_clean_resources(resources),
    )


def wildcard_side(options: tuple[ResourceName, ...], count: int = 1) -> TradeSide:
    if count <= 0:
        raise ValueError("wildcard count must be positive")
    unknown = sorted(set(options).difference(RESOURCE_NAMES))
    if unknown:
        raise ValueError(f"unknown resources: {', '.join(unknown)}")
    return TradeSide(kind="wildcard", options=tuple(options), count=count)


def open_side(count: int = 1) -> TradeSide:
    if count <= 0:
        raise ValueError("open side count must be positive")
    return TradeSide(kind="open", count=count)


def _clean_resources(resources: dict[str, int]) -> dict[ResourceName, int]:
    cleaned: dict[ResourceName, int] = {}
    for name, count in resources.items():
        if name not in RESOURCE_NAMES:
            raise ValueError(f"unknown resource: {name}")
        if count < 0:
            raise ValueError("resource counts must be non-negative")
        if count:
            cleaned[name] = count
    if not cleaned:
        raise ValueError("exact trade side must contain at least one resource")
    return cleaned
