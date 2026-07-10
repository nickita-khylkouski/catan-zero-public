#!/usr/bin/env python3
"""f74 test-time symmetry evaluation on real placement roots.

Two measurements, both grounded on the fact that the true value/policy of a
Catan position is INVARIANT under the 12 D6 board symmetries:

1. Symmetry-inconsistency of the CURRENT checkpoint. For each root we evaluate
   all 12 orientations and report the spread of the value head (and of the
   per-candidate prior / q) across them. A perfectly symmetric model would give
   identical outputs, so this spread quantifies how much the model violates a
   known invariant -- and how much headroom symmetry augmentation has.

2. Test-time noise reduction from 12-fold averaging. The orientation spread is
   pure noise (the target is invariant), so averaging over orientations removes
   it. We measure this truth-free with a finite-population denoising curve: the
   RMS deviation of a random k-orientation subset mean from the full 12-mean
   ("consensus"), for k = 1..6. Under independent orientation noise this
   shrinks as ``sqrt(1/k - 1/N)`` (sampling without replacement), so the
   measured ``dev_1 / dev_6`` ratio compared to the independent ideal
   ``sqrt(11) ~= 3.32`` reveals how much of the noise is genuinely independent
   across orientations (and thus averageable toward the ``sqrt(12) ~= 3.46x``
   ceiling). NOTE: the common-mode component -- an identical error in all 12
   orientations -- is not identifiable from a single position and is not
   removed by averaging; we deliberately do not fabricate a factor for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.hex_symmetry import N_SYMMETRIES, build_hex_symmetry
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluatorConfig,
    _assert_public_observation_matches_checkpoint_training,
    rust_action_context_batch,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)
from catan_zero.search.rust_mcts import _require_rust_module
from factory_common import write_json
from sigma_trace_placement_root import COLORS, find_placement_roots


def _root_entity(
    policy: EntityGraphPolicy, game: Any, *, public_observation: bool = False
) -> dict[str, Any]:
    acting = str(game.current_color())
    legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
    pids = rust_policy_action_ids(game, legal, colors=COLORS, action_size=int(policy.action_size))
    entity = rust_game_to_entity_batch(
        game, legal, actor=acting, colors=COLORS,
        action_size=int(policy.action_size), policy_action_ids=pids,
        public_observation=bool(public_observation),
    )
    context = rust_action_context_batch(
        game, legal, actor=acting, colors=COLORS,
        action_size=int(policy.action_size), policy_action_ids=pids,
        public_observation=bool(public_observation),
    )
    legal_ids = np.asarray(pids, dtype=np.int64)[None, :]
    return {"entity": entity, "context": context, "legal_ids": legal_ids}


def _denoising_curve(per_orient: np.ndarray, ks=(1, 2, 3, 4, 6), n_draws=200, seed=0):
    """RMS deviation of a random k-orientation subset mean from the full
    N-orientation consensus mean, aggregated over samples and random subsets.

    ``per_orient`` has shape (S, N): S independent samples (roots, or
    root x candidate), N orientations. Truth-free: the reference is the
    12-mean, and deviations from it for k < N are genuine observables. Under
    independent orientation noise, dev_k ~ sigma * sqrt(1/k - 1/N)."""
    per_orient = np.asarray(per_orient, dtype=np.float64)
    s, n = per_orient.shape
    consensus = per_orient.mean(axis=1)  # (S,)
    rng = np.random.default_rng(seed)
    curve = {}
    for k in ks:
        if k >= n:
            curve[str(k)] = 0.0
            continue
        acc = 0.0
        cnt = 0
        for _ in range(n_draws):
            idx = rng.choice(n, size=k, replace=False)
            sub = per_orient[:, idx].mean(axis=1)  # (S,)
            acc += float(((sub - consensus) ** 2).sum())
            cnt += s
        curve[str(k)] = float(np.sqrt(acc / cnt))
    return curve


def _ideal_ratio(k1=1, k6=6, n=N_SYMMETRIES) -> float:
    return float(np.sqrt((1.0 / k1 - 1.0 / n) / (1.0 / k6 - 1.0 / n)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-states", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=500001)
    parser.add_argument(
        "--public-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask hidden opponent information during featurization. Must match "
        "the checkpoint's recorded training regime.",
    )
    parser.add_argument("--relabel-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    rs = _require_rust_module()
    games = find_placement_roots(rs, n_states=int(args.n_states), base_seed=int(args.base_seed))
    policy = EntityGraphPolicy.load(args.checkpoint, device=args.device)
    policy.model.eval()
    _assert_public_observation_matches_checkpoint_training(
        policy,
        EntityGraphRustEvaluatorConfig(
            public_observation=bool(args.public_observation)
        ),
    )
    sym = build_hex_symmetry()

    def forward_fn(entity_n, legal_n, ctx_n, return_q):
        with torch.no_grad():
            out = policy.forward_legal_np(entity_n, legal_n, ctx_n, return_q=return_q)
        res = {
            "logits": out["logits"].detach().float().cpu().numpy(),
            "value": out["value"].detach().float().cpu().numpy().reshape(-1),
        }
        if return_q and out.get("q_values") is not None:
            res["q_values"] = out["q_values"].detach().float().cpu().numpy()
        return res

    value_per_root = []          # (R, N) value across orientations
    _prior_raw_rows = []         # stacked (candidate, N) raw prior logits
    _q_raw_rows = []
    per_root = []

    for game in games:
        r = _root_entity(
            policy,
            game.copy(),
            public_observation=bool(args.public_observation),
        )
        avg = sym.average_forward(
            r["entity"], r["legal_ids"], r["context"], forward_fn,
            return_q=True, relabel_events=bool(args.relabel_events),
        )
        vpo = np.asarray(avg["value_per_orientation"], dtype=np.float64)  # (N,)
        value_per_root.append(vpo)

        # per-candidate prior/q, restricted to legal candidates (mask sentinel).
        legal_mask = (r["legal_ids"][0] >= 0)
        lpo = np.asarray(avg["logits_per_orientation"], dtype=np.float64)[:, legal_mask]  # (N, C)
        qpo = np.asarray(avg["q_values_per_orientation"], dtype=np.float64)[:, legal_mask]
        prior_dev = (lpo - lpo.mean(axis=0, keepdims=True)).T   # (C, N) for inconsistency std
        q_dev = (qpo - qpo.mean(axis=0, keepdims=True)).T
        _prior_raw_rows.append(lpo.T)   # (C, N) raw, for the denoising curve
        _q_raw_rows.append(qpo.T)

        per_root.append({
            "n_candidates": int(legal_mask.sum()),
            "value_orientation_std": float(vpo.std()),
            "value_orientation_range": float(vpo.max() - vpo.min()),
            "value_mean": float(vpo.mean()),
            "prior_cand_orientation_std_mean": float(prior_dev.std(axis=1).mean()),
            "q_cand_orientation_std_mean": float(q_dev.std(axis=1).mean()),
        })

    value_per_root = np.asarray(value_per_root)                  # (R, N) raw values
    prior_raw = np.concatenate(_prior_raw_rows, axis=0)          # (sum C, N)
    q_raw = np.concatenate(_q_raw_rows, axis=0)

    value_curve = _denoising_curve(value_per_root)
    prior_curve = _denoising_curve(prior_raw)
    q_curve = _denoising_curve(q_raw)
    ideal_ratio_1_6 = _ideal_ratio(1, 6)

    def _ratio(curve):
        d1, d6 = curve.get("1", 0.0), curve.get("6", 0.0)
        return float(d1 / d6) if d6 > 1e-12 else float("nan")

    def _stat(vals):
        vals = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "max": float(vals.max()),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "n_roots": len(per_root),
        "n_symmetries": N_SYMMETRIES,
        "relabel_events": bool(args.relabel_events),
        "public_observation": bool(args.public_observation),
        "symmetry_inconsistency": {
            "value_orientation_std": _stat([r["value_orientation_std"] for r in per_root]),
            "value_orientation_range": _stat([r["value_orientation_range"] for r in per_root]),
            "prior_candidate_orientation_std": _stat([r["prior_cand_orientation_std_mean"] for r in per_root]),
            "q_candidate_orientation_std": _stat([r["q_cand_orientation_std_mean"] for r in per_root]),
        },
        "noise_reduction": {
            "note": (
                "dev_k = RMS deviation of a random k-orientation mean from the "
                "12-consensus. dev_1_over_dev_6 vs independent ideal sqrt(11)~=3.32 "
                "shows how averageable the orientation noise is; 12-fold averaging "
                "removes this reducible dispersion (toward the sqrt(12)~=3.46x "
                "ceiling on the independent component)."
            ),
            "ideal_dev1_over_dev6_independent": ideal_ratio_1_6,
            "value": {
                "denoising_curve_rms": value_curve,
                "dev1_over_dev6": _ratio(value_curve),
            },
            "prior": {
                "denoising_curve_rms": prior_curve,
                "dev1_over_dev6": _ratio(prior_curve),
            },
            "q": {
                "denoising_curve_rms": q_curve,
                "dev1_over_dev6": _ratio(q_curve),
            },
        },
    }
    write_json(args.out, {"summary": summary, "per_root": per_root})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
