"""Public-information belief primitives for imperfect-information search.

This module is deliberately engine independent.  A search implementation may
construct a :class:`PublicBelief` from an omniscient engine snapshot, but the
constructor copies only information available to ``perspective``:

* the perspective player's own resource and development-card composition;
* every player's public hand sizes and played development cards; and
* the public remaining development-deck size.

The resulting chance distributions and keyed sampler therefore cannot change
when an opponent's hidden hand (or the hidden deck order) is permuted while
those public facts are held fixed.

This is a chance-belief API, not a complete information-set game solver.  In
particular, it does not yet determinize opponents' hands before asking the
engine for their legal actions.  Callers must not describe enabling this
module alone as fixing opponent-action leakage; see ``OPPONENT_ACTION_SCOPE``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

RESOURCES: tuple[str, ...] = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVELOPMENT_CARDS: tuple[str, ...] = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)
BASE_DEVELOPMENT_DECK: dict[str, int] = {
    "KNIGHT": 14,
    "VICTORY_POINT": 5,
    "YEAR_OF_PLENTY": 2,
    "MONOPOLY": 2,
    "ROAD_BUILDING": 2,
}

OPPONENT_ACTION_SCOPE = (
    "Chance beliefs cover robber-steal identity and development-card draws. "
    "Opponent legal actions are still produced by the authoritative engine "
    "state and can therefore reveal opponent resources or playable cards. A "
    "full fix requires root-consistent determinizations or an information-set "
    "search engine; this module intentionally does not claim that fix."
)


def _named_counts(value: Any, names: Sequence[str]) -> tuple[int, ...]:
    """Normalize Rust dict/list count encodings into canonical tuple order."""
    if isinstance(value, Mapping):
        return tuple(max(0, int(value.get(name, 0) or 0)) for name in names)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(max(0, int(value[index] or 0)) if index < len(value) else 0 for index in range(len(names)))
    return (0,) * len(names)


def _player_states(snapshot: Mapping[str, Any], colors: tuple[str, ...]) -> dict[str, Mapping[str, Any]]:
    raw = snapshot.get("player_state", ())
    if isinstance(raw, Mapping):
        return {
            color: raw.get(color, {}) if isinstance(raw.get(color, {}), Mapping) else {}
            for color in colors
        }
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {
            color: raw[index] if index < len(raw) and isinstance(raw[index], Mapping) else {}
            for index, color in enumerate(colors)
        }
    return {color: {} for color in colors}


@dataclass(frozen=True, slots=True)
class PublicBelief:
    """Immutable public information state from one player's perspective."""

    perspective: str
    colors: tuple[str, ...]
    own_resources: tuple[int, ...]
    own_development_cards: tuple[int, ...]
    resource_card_counts: tuple[int, ...]
    development_card_counts: tuple[int, ...]
    played_development_cards: tuple[tuple[int, ...], ...]
    development_deck_count: int

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        perspective: str,
    ) -> "PublicBelief":
        """Extract only perspective-visible facts from a Rust JSON snapshot.

        ``snapshot`` may remain omniscient.  Opponent ``resources`` and
        ``dev_cards`` are never copied, and hidden deck-order fields are never
        inspected.  This narrow extraction boundary is what makes downstream
        hidden-truth invariance testable.
        """
        colors = tuple(str(color) for color in snapshot.get("colors", ()))
        perspective = str(perspective)
        if perspective not in colors:
            raise ValueError(f"perspective {perspective!r} is not in colors {colors!r}")
        states = _player_states(snapshot, colors)
        own = states[perspective]

        resource_counts: list[int] = []
        development_counts: list[int] = []
        played: list[tuple[int, ...]] = []
        for color in colors:
            state = states[color]
            resources = _named_counts(state.get("resources"), RESOURCES)
            dev_cards = _named_counts(state.get("dev_cards"), DEVELOPMENT_CARDS)
            resource_counts.append(int(state.get("resource_card_count", sum(resources)) or 0))
            development_counts.append(
                int(state.get("development_card_count", sum(dev_cards)) or 0)
            )
            played.append(
                _named_counts(
                    state.get("played_dev_cards", state.get("played_development_cards")),
                    DEVELOPMENT_CARDS,
                )
            )

        return cls(
            perspective=perspective,
            colors=colors,
            own_resources=_named_counts(own.get("resources"), RESOURCES),
            own_development_cards=_named_counts(own.get("dev_cards"), DEVELOPMENT_CARDS),
            resource_card_counts=tuple(resource_counts),
            development_card_counts=tuple(development_counts),
            played_development_cards=tuple(played),
            development_deck_count=max(0, int(snapshot.get("development_deck_count", 0) or 0)),
        )
    def _player_index(self, color: str) -> int:
        try:
            return self.colors.index(str(color))
        except ValueError as error:
            raise ValueError(f"unknown player color {color!r}") from error

    def robber_steal_probabilities(self, victim: str) -> dict[str, float]:
        """Belief over the identity of a card stolen from ``victim``.

        The perspective player's own hand is known, so if they are the victim
        the exact count-weighted Catan distribution is returned.  For an
        opponent only hand size is public; absent a richer history posterior,
        the symmetric maximum-entropy prior is uniform over the five resource
        identities.  This intentionally gives positive mass to types the true
        hidden hand might not contain--conditioning on that truth would leak it.
        """
        victim_index = self._player_index(victim)
        total = self.resource_card_counts[victim_index]
        if total <= 0:
            return {}
        if str(victim) == self.perspective:
            own_total = sum(self.own_resources)
            if own_total <= 0:
                return {}
            return {
                resource: count / own_total
                for resource, count in zip(RESOURCES, self.own_resources)
                if count > 0
            }
        probability = 1.0 / len(RESOURCES)
        return {resource: probability for resource in RESOURCES}

    def development_draw_probabilities(self) -> dict[str, float]:
        """Posterior predictive distribution for the next dev-card draw.

        The unknown pool is the base deck minus all publicly played cards and
        the perspective player's own unplayed cards.  Opponents' hidden cards
        and the remaining deck are exchangeable allocations of that pool, so
        its normalized composition is the marginal next-draw distribution.
        """
        if self.development_deck_count <= 0:
            return {}
        remaining: list[int] = []
        for card_index, card in enumerate(DEVELOPMENT_CARDS):
            public_played = sum(row[card_index] for row in self.played_development_cards)
            known_own = self.own_development_cards[card_index]
            remaining.append(max(0, BASE_DEVELOPMENT_DECK[card] - public_played - known_own))
        total = sum(remaining)
        if total <= 0:
            return {}
        return {
            card: count / total
            for card, count in zip(DEVELOPMENT_CARDS, remaining)
            if count > 0
        }

    def fingerprint(self) -> str:
        """Stable digest containing no opponent hidden composition."""
        payload = {
            "perspective": self.perspective,
            "colors": self.colors,
            "own_resources": self.own_resources,
            "own_development_cards": self.own_development_cards,
            "resource_card_counts": self.resource_card_counts,
            "development_card_counts": self.development_card_counts,
            "played_development_cards": self.played_development_cards,
            "development_deck_count": self.development_deck_count,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.blake2b(encoded, digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicBeliefSampler:
    """Deterministic, call-order-independent sampler keyed by public state.

    A sample is a pure function of ``seed``, the public-belief fingerprint,
    event namespace/context and ``sample_index``.  Parallel workers therefore
    reproduce the same samples without sharing mutable RNG state.
    """

    seed: int

    def _rng(
        self,
        belief: PublicBelief,
        *,
        namespace: str,
        context: str,
        sample_index: int,
    ) -> random.Random:
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        material = "|".join(
            (str(int(self.seed)), belief.fingerprint(), namespace, context, str(sample_index))
        ).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(material, digest_size=16).digest(), "big")
        return random.Random(value)

    @staticmethod
    def _choice(probabilities: Mapping[str, float], rng: random.Random) -> str | None:
        positive = [(name, float(weight)) for name, weight in probabilities.items() if weight > 0.0]
        total = sum(weight for _name, weight in positive)
        if total <= 0.0:
            return None
        threshold = rng.random() * total
        cumulative = 0.0
        for name, weight in positive:
            cumulative += weight
            if threshold < cumulative:
                return name
        return positive[-1][0]

    def sample_robber_steal(
        self,
        belief: PublicBelief,
        *,
        victim: str,
        sample_index: int = 0,
    ) -> str | None:
        return self._choice(
            belief.robber_steal_probabilities(victim),
            self._rng(
                belief,
                namespace="robber-steal",
                context=str(victim),
                sample_index=sample_index,
            ),
        )

    def sample_development_draw(
        self,
        belief: PublicBelief,
        *,
        sample_index: int = 0,
    ) -> str | None:
        return self._choice(
            belief.development_draw_probabilities(),
            self._rng(
                belief,
                namespace="development-draw",
                context="deck",
                sample_index=sample_index,
            ),
        )
