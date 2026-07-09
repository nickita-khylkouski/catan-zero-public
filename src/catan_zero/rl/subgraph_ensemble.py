"""Subgraph-sampling ensemble for the entity-graph net (CAT-97, ScalableAlphaZero).

ScalableAlphaZero ("Train on Small, Play the Large", Ben-Assayag & El-Yaniv,
arXiv 2107.08387) evaluates a few randomly sampled SUBGRAPHS of the board with
the *same* GNN and blends their outputs to reduce prediction uncertainty. This
module implements that as a pure INFERENCE procedure over an existing
``EntityGraphPolicy`` -- it adds NO parameters, so it warm-starts trivially from
any checkpoint (including champion_v0) and is a no-op unless explicitly called.

Two things, kept deliberately distinct because the paper is precise about them:

  * POLICY blend (paper-faithful, §3.2): the paper combines ONLY the policy
    prior, leaving the value backup untouched. The published rule is
    ``P = (p_full + p_full ∘ p_sub_mean) / 2`` (elementwise, then renormalised),
    where ``p_sub_mean`` is the mean of the per-subgraph policy distributions.
    We reproduce that exactly and expose it as ``policy_probs`` / ``logits``.

  * VALUE uncertainty (OUR extension, clearly labelled): the paper does NOT
    combine subgraph value heads (it explicitly keeps ``v(s')`` from the full
    graph). Averaging the value head across subgraphs to obtain a variance
    estimate is our own extension, motivated by CAT-97's ask for a
    value-uncertainty signal. We therefore return the full-graph value as
    ``value`` (the faithful MCTS-backup quantity) AND, separately,
    ``value_ensemble_mean`` / ``value_std`` from the {full + subgraph} set.
    Nothing here silently changes the value used for search.

A "subgraph" is realised by dropping a random fraction of the currently-valid
board tokens (hex / vertex / edge -- the intersection/path/hex graph) from the
attention mask, which is exactly how the transformer represents an induced
subgraph: the dropped tokens stop being attended to. Context tokens (CLS,
player, global, event) are always kept, analogous to ScalableAlphaZero's
dummy node that stays connected to preserve long-range/global information.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# Board token fields that form the spatial graph we sample subgraphs from. Their
# masks are [B, N] with 1 == valid; dropping a token = setting its mask to 0.
_BOARD_TOKEN_MASKS = ("hex_mask", "vertex_mask", "edge_mask")


def _drop_board_tokens(
    entity_batch: Mapping[str, np.ndarray],
    drop_fraction: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return a COPY of ``entity_batch`` with a random induced board-subgraph.

    For each board mask, independently per batch row, ``drop_fraction`` of the
    currently-valid tokens are flipped to invalid (dropped from attention). The
    caller's arrays are never mutated. At least one valid token per mask is kept
    so a row never collapses to an all-padded board.
    """
    out: dict[str, np.ndarray] = dict(entity_batch)
    for mask_key in _BOARD_TOKEN_MASKS:
        if mask_key not in entity_batch:
            continue
        mask = np.array(entity_batch[mask_key], copy=True)
        original_dtype = mask.dtype
        valid = mask.astype(bool)
        new_mask = valid.copy()
        batch_size = valid.shape[0]
        for row in range(batch_size):
            valid_idx = np.flatnonzero(valid[row])
            if valid_idx.size <= 1:
                continue
            n_drop = int(np.floor(drop_fraction * valid_idx.size))
            # Never drop every valid token in the row.
            n_drop = min(n_drop, valid_idx.size - 1)
            if n_drop <= 0:
                continue
            drop_idx = rng.choice(valid_idx, size=n_drop, replace=False)
            new_mask[row, drop_idx] = False
        out[mask_key] = new_mask.astype(original_dtype)
    return out


def _legal_mask(legal_action_ids: np.ndarray) -> "Any":
    import torch

    ids = torch.as_tensor(np.asarray(legal_action_ids, dtype=np.int64))
    return ids >= 0  # [B, A]


def _masked_softmax(logits: "Any", legal: "Any") -> "Any":
    import torch

    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(legal, logits, torch.full_like(logits, neg_inf))
    return torch.softmax(masked, dim=-1) * legal.to(logits.dtype)


def subgraph_ensemble_forward(
    policy: Any,
    entity_batch: Mapping[str, np.ndarray],
    legal_action_ids: np.ndarray,
    legal_action_context: np.ndarray,
    *,
    num_samples: int = 4,
    drop_fraction: float = 0.25,
    seed: int | None = None,
    return_q: bool = False,
) -> dict[str, "Any"]:
    """Run the subgraph-sampling ensemble over ``policy.forward_legal_np``.

    Returns a dict with the same top-level keys the caller expects plus the
    ensemble extras:

      * ``logits`` / ``policy_probs`` : the ScalableAlphaZero-blended policy
        (paper rule ``(p_full + p_full∘p_sub_mean)/2``, renormalised over legal
        actions). ``logits`` are log-probs with invalid actions at ``-inf`` so
        existing argmax / softmax consumers behave identically.
      * ``value`` : the FULL-graph value (paper-faithful MCTS backup); unchanged.
      * ``value_ensemble_mean`` / ``value_std`` : mean and std of the value head
        across {full graph + subgraphs} -- OUR value-uncertainty extension.
      * ``policy_full`` : the un-blended full-graph policy, for diagnostics.
      * any other keys the base forward produced (``final_vp``, ``q_values`` ...).

    ``num_samples <= 0`` or ``drop_fraction <= 0`` is an EXACT no-op: it returns
    the full-graph outputs verbatim with ``value_std`` all zero. This is the
    warm-start / default-off guarantee.
    """
    import torch

    full = policy.forward_legal_np(
        entity_batch, legal_action_ids, legal_action_context, return_q=return_q
    )
    legal = _legal_mask(legal_action_ids).to(full["logits"].device)
    p_full = _masked_softmax(full["logits"], legal)

    result: dict[str, Any] = dict(full)
    result["policy_full"] = p_full

    if int(num_samples) <= 0 or float(drop_fraction) <= 0.0:
        result["policy_probs"] = p_full
        result["value_ensemble_mean"] = full["value"]
        result["value_std"] = torch.zeros_like(full["value"])
        return result

    rng = np.random.default_rng(seed)
    sub_policies = []
    sub_values = [full["value"]]
    for _ in range(int(num_samples)):
        masked_batch = _drop_board_tokens(entity_batch, float(drop_fraction), rng)
        sub = policy.forward_legal_np(
            masked_batch, legal_action_ids, legal_action_context, return_q=False
        )
        sub_policies.append(_masked_softmax(sub["logits"], legal))
        sub_values.append(sub["value"])

    p_sub_mean = torch.stack(sub_policies, dim=0).mean(dim=0)  # [B, A]
    # ScalableAlphaZero Eq.: P = (p1 + p1 ∘ p2) / 2, then renormalise over legal.
    blended = 0.5 * (p_full + p_full * p_sub_mean)
    blended = blended * legal.to(blended.dtype)
    blended = blended / blended.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    neg_inf = torch.finfo(blended.dtype).min
    blended_logits = torch.where(
        legal, torch.log(blended.clamp_min(1.0e-12)), torch.full_like(blended, neg_inf)
    )

    values = torch.stack(sub_values, dim=0)  # [num_samples+1, B]
    result["logits"] = blended_logits
    result["policy_probs"] = blended
    result["value_ensemble_mean"] = values.mean(dim=0)
    result["value_std"] = values.std(dim=0, unbiased=False)
    # `value` deliberately stays the full-graph value (paper-faithful backup).
    result["value"] = full["value"]
    return result
