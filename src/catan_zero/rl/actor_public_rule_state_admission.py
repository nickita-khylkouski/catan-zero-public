"""Fail-closed checks for actor playable-development-card features.

The 2p/no-trade policy catalog assigns one contiguous action range to each
playable development-card type.  Whenever one of those actions is legal (or
was selected), the authoritative pre-action state must encode a positive
actor playable-card count in the corresponding global-token slot.  Checking
that implication catches replay/runtime drift that ordinary tensor-shape and
adapter-version admission cannot detect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ADMISSION_SCHEMA = "actor-playable-development-card-admission-v1"


class ActorPublicRuleStateAdmissionError(RuntimeError):
    """The corpus contradicts the actor public-rule-state contract."""


@dataclass(frozen=True)
class _CardContract:
    card: str
    global_slot: int
    first_action_id: int
    last_action_id: int


# Stable 2p/no-trade ActionCatalog policy IDs.  The four global-token slots
# are defined by actor_public_rule_state_2p_v1 in the same semantic order.
_CARD_CONTRACTS = (
    _CardContract("KNIGHT", 12, 304, 304),
    _CardContract("YEAR_OF_PLENTY", 13, 311, 330),
    _CardContract("MONOPOLY", 14, 305, 309),
    _CardContract("ROAD_BUILDING", 15, 310, 310),
)


def _column(data: Any, name: str) -> Any:
    try:
        return data[name]
    except (KeyError, TypeError) as error:
        raise ActorPublicRuleStateAdmissionError(
            f"actor public-rule-state admission requires column {name!r}"
        ) from error


def audit_actor_playable_development_cards(
    data: Mapping[str, Any] | Any,
    *,
    where: str,
    chunk_rows: int = 65_536,
) -> dict[str, Any]:
    """Prove legal/selected play-dev actions have matching positive features.

    ``data`` may be an in-memory shard dictionary or a memmap corpus.  The
    scan is chunked so startup admission does not materialize the full legal
    action matrix in RAM.
    """

    global_tokens = _column(data, "global_tokens")
    action_taken = _column(data, "action_taken")
    legal_action_ids = _column(data, "legal_action_ids")
    legal_action_mask = _column(data, "legal_action_mask")
    shape = tuple(int(value) for value in getattr(global_tokens, "shape", ()))
    if len(shape) != 3 or shape[1] != 1 or shape[2] <= 15:
        raise ActorPublicRuleStateAdmissionError(
            f"{where} has invalid global_tokens shape {shape}; expected (rows,1,>=16)"
        )
    rows = int(shape[0])
    peer_shapes = {
        "action_taken": tuple(
            int(value) for value in getattr(action_taken, "shape", ())
        ),
        "legal_action_ids": tuple(
            int(value) for value in getattr(legal_action_ids, "shape", ())
        ),
        "legal_action_mask": tuple(
            int(value) for value in getattr(legal_action_mask, "shape", ())
        ),
    }
    if (
        peer_shapes["action_taken"] != (rows,)
        or len(peer_shapes["legal_action_ids"]) != 2
        or peer_shapes["legal_action_ids"][0] != rows
        or peer_shapes["legal_action_mask"]
        != peer_shapes["legal_action_ids"]
    ):
        raise ActorPublicRuleStateAdmissionError(
            f"{where} has incompatible action/feature shapes: {peer_shapes}"
        )

    counts = {
        contract.card: {
            "global_slot": contract.global_slot,
            "action_id_range": [
                contract.first_action_id,
                contract.last_action_id,
            ],
            "selected_rows": 0,
            "legal_rows": 0,
            "required_rows": 0,
            "positive_feature_rows": 0,
        }
        for contract in _CARD_CONTRACTS
    }
    width = max(1, int(chunk_rows))
    for start in range(0, rows, width):
        stop = min(rows, start + width)
        features = np.asarray(global_tokens[start:stop])[:, 0, 12:16]
        selected = np.asarray(action_taken[start:stop])
        legal_ids = np.asarray(legal_action_ids[start:stop])
        legal_mask = np.asarray(legal_action_mask[start:stop], dtype=np.bool_)
        if not np.isfinite(features).all() or bool(np.any(features < 0)):
            raise ActorPublicRuleStateAdmissionError(
                f"{where} has non-finite or negative actor playable-card features"
            )
        for contract in _CARD_CONTRACTS:
            selected_rows = (selected >= contract.first_action_id) & (
                selected <= contract.last_action_id
            )
            legal_rows = np.any(
                legal_mask
                & (legal_ids >= contract.first_action_id)
                & (legal_ids <= contract.last_action_id),
                axis=1,
            )
            required_rows = selected_rows | legal_rows
            positive_rows = features[:, contract.global_slot - 12] > 0
            invalid = required_rows & ~positive_rows
            if bool(np.any(invalid)):
                local = np.flatnonzero(invalid)
                examples = (local[:8] + start).astype(np.int64).tolist()
                raise ActorPublicRuleStateAdmissionError(
                    f"{where} has {int(np.count_nonzero(invalid))} {contract.card} "
                    "legal/selected rows with a non-positive actor playable-card "
                    f"slot {contract.global_slot}; example rows={examples}"
                )
            card_counts = counts[contract.card]
            card_counts["selected_rows"] += int(np.count_nonzero(selected_rows))
            card_counts["legal_rows"] += int(np.count_nonzero(legal_rows))
            card_counts["required_rows"] += int(np.count_nonzero(required_rows))
            card_counts["positive_feature_rows"] += int(
                np.count_nonzero(positive_rows)
            )

    observed_cards = [
        card for card, values in counts.items() if values["required_rows"] > 0
    ]
    for card in observed_cards:
        if counts[card]["positive_feature_rows"] <= 0:
            raise ActorPublicRuleStateAdmissionError(
                f"{where} has {card} action support but no positive feature support"
            )
    return {
        "schema_version": ADMISSION_SCHEMA,
        "authenticated": True,
        "where": str(where),
        "rows": rows,
        "implication": (
            "selected_or_legal_play_development_card_implies_"
            "positive_actor_playable_card_feature"
        ),
        "observed_cards": observed_cards,
        "cards": counts,
    }
