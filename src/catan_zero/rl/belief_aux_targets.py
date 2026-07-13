"""Training-only privileged labels for public-information belief supervision.

The model input must be passed through ``mask_player_tokens_public``.  This
module is deliberately label-only: it extracts opponent hand composition from
the omniscient banked token block *before* that transform and never reconstructs
hidden information from a public observation.
"""

from __future__ import annotations

import numpy as np

from catan_zero.rl.entity_token_features import PLAYER_ACTOR_FLAG_SLOT


RESOURCE_TOTAL_SLOT = 6
RESOURCE_LABEL_PRESENT_SLOT = 15
RESOURCE_COMPOSITION_SLICE = slice(16, 21)
PLAYER_PRESENT_SLOT = 0


def resource_belief_targets(
    player_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(composition, public_total, valid)`` for opponent player rows.

    ``composition`` is an unnormalised count vector with shape ``(..., P, 5)``;
    ``public_total`` and ``valid`` have shape ``(..., P)``.  The featurizer
    stores resource counts divided by 10 and the public total divided by 20.
    A row is valid only when it is a present non-actor player, omniscient labels
    are present, the public total is positive, and the private counts sum to the
    public total. Saturated or non-integral banked encodings are rejected too:
    clipped feature slots cannot prove the exact hidden hand. These checks
    prevent malformed or lossy provenance from becoming a privileged signal.
    """

    tokens = np.asarray(player_tokens)
    if tokens.ndim not in (2, 3) or tokens.shape[-1] < 21:
        raise ValueError(
            "player_tokens must have shape (P,F) or (B,P,F) with F >= 21; "
            f"got {tokens.shape}"
        )
    single = tokens.ndim == 2
    if single:
        tokens = tokens[None, ...]

    composition_scaled = (
        tokens[..., RESOURCE_COMPOSITION_SLICE].astype(np.float32) * 10.0
    )
    total_scaled = tokens[..., RESOURCE_TOTAL_SLOT].astype(np.float32) * 20.0
    numeric = (
        np.isfinite(composition_scaled).all(axis=-1)
        & np.isfinite(total_scaled)
        & (composition_scaled >= 0.0).all(axis=-1)
        & (total_scaled >= 0.0)
    )
    # The source featurizer clips every resource slot at 10 cards and the
    # public total at 20.  Once either encoded value reaches its ceiling we
    # cannot distinguish the exact boundary from a larger, clipped hand.  In
    # particular, an actual 21-card hand such as [11, 3, 2, 2, 3] is banked as
    # [10, 3, 2, 2, 3] with total 20 and would otherwise pass the sum check as a
    # silently incomplete label.  Reject the (rare) ambiguous boundary rather
    # than train on invented hidden truth.
    unsaturated = (composition_scaled < 10.0 - 0.01).all(axis=-1) & (
        total_scaled < 20.0 - 0.01
    )
    # Valid feature-bank counts are integer-valued before scaling.  Checking
    # integrality before rounding prevents malformed fractional encodings from
    # being silently coerced into plausible privileged labels.  The tolerance
    # covers float16 storage error (e.g. 0.3 * 10) without admitting a half-card.
    integral = np.isclose(
        composition_scaled,
        np.rint(composition_scaled),
        rtol=0.0,
        atol=0.01,
    ).all(axis=-1) & np.isclose(
        total_scaled,
        np.rint(total_scaled),
        rtol=0.0,
        atol=0.01,
    )
    composition = np.clip(
        np.rint(
            np.nan_to_num(
                composition_scaled, nan=0.0, posinf=0.0, neginf=0.0
            )
        ),
        0.0,
        None,
    )
    public_total = np.clip(
        np.rint(
            np.nan_to_num(total_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        ),
        0.0,
        None,
    )
    present = tokens[..., PLAYER_PRESENT_SLOT] > 0.5
    actor = tokens[..., PLAYER_ACTOR_FLAG_SLOT] > 0.5
    labels_present = tokens[..., RESOURCE_LABEL_PRESENT_SLOT] > 0.5
    totals_match = np.isclose(
        composition.sum(axis=-1), public_total, rtol=0.0, atol=0.01
    )
    valid = (
        present
        & ~actor
        & labels_present
        & numeric
        & unsaturated
        & integral
        & (public_total > 0.0)
        & totals_match
    )

    composition = composition.astype(np.float32, copy=False)
    public_total = public_total.astype(np.float32, copy=False)
    valid = valid.astype(np.bool_, copy=False)
    if single:
        return composition[0], public_total[0], valid[0]
    return composition, public_total, valid
