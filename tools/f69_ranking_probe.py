#!/usr/bin/env python3
"""f69 ranking-discrimination probe on real 54-wide placement roots.

Measures, per real initial-placement root, how much the model's prior and
q_head SPREAD candidates apart -- the quantity the project's measured failure
is about (a ~0.06-nat prior spread over 54 near-tied opening placements, with
value noise dominating). For each root it reports the prior-logit spread and
the q_head spread (range and std across the legal candidates), plus the
softmax top1-top2 gap, then aggregates across roots.

It evaluates a BASE policy (all f69 upgrades off) and an UPGRADED policy
warm-started from the SAME checkpoint (flags on, new params at init). Because
every upgrade's output path is zero-initialised, an UNTRAINED upgrade must
reproduce the base numbers exactly -- so this script is also the warm-start
equivalence proof against the real 35M checkpoint (see `warm_start_max_diff`
in the output; it must be 0.0). After a finetune of the upgraded config, rerun
with the finetuned checkpoint to see whether the spreads separate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.entity_feature_adapter import (
    policy_entity_feature_adapter_version,
)
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.search.neural_rust_mcts import (
    _policy_history_options,
    rust_action_context_batch,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)
from catan_zero.search.rust_mcts import _require_rust_module
from factory_common import write_json
from sigma_trace_placement_root import COLORS, find_placement_roots

UPGRADE_FLAGS = ("action_target_gather", "action_cross_attention_layers", "value_attention_pool")


def _parse_flags(raw: str) -> dict[str, Any]:
    """"gather,cross:2,value" -> config overrides for the upgraded policy."""
    overrides: dict[str, Any] = {}
    for entry in (piece.strip() for piece in raw.split(",") if piece.strip()):
        if entry in ("gather", "action_target_gather"):
            overrides["action_target_gather"] = True
        elif entry in ("value", "value_attention_pool"):
            overrides["value_attention_pool"] = True
        elif entry.startswith("cross"):
            n = entry.split(":", 1)[1] if ":" in entry else "2"
            overrides["action_cross_attention_layers"] = int(n)
        else:
            raise SystemExit(f"unknown upgrade flag: {entry!r}")
    return overrides


def _upgraded_policy_from(base: EntityGraphPolicy, overrides: dict[str, Any]) -> EntityGraphPolicy:
    """Clone `base` with the upgrade flags on, warm-starting all shared weights
    from the base model and leaving the new (zero-init) params at init."""
    upgraded_config = dataclasses.replace(base.config, **overrides)
    static = base.static_action_features.detach().cpu().numpy()
    upgraded = EntityGraphPolicy(
        upgraded_config,
        static,
        device=str(base.device),
        entity_feature_adapter_version=policy_entity_feature_adapter_version(base),
    )
    # This is a function-preserving in-memory clone of the loaded checkpoint,
    # not a freshly trained policy. Preserve the checkpoint-bound information
    # surface so the base/upgraded comparison consumes identical observations.
    upgraded.trained_with_masked_hidden_info = bool(
        getattr(base, "trained_with_masked_hidden_info", False)
    )
    upgraded.public_award_feature_contract = str(
        base.public_award_feature_contract
    )
    upgraded.entity_feature_adapter_binding_source = str(
        getattr(base, "entity_feature_adapter_binding_source", "legacy_policy")
    )
    missing, unexpected = upgraded.model.load_state_dict(base.model.state_dict(), strict=False)
    disallowed = [k for k in missing if not k.startswith(
        ("target_gather_proj.", "action_cross_blocks.", "value_probe", "value_pool_head.", "q_head.")
    )]
    if disallowed or unexpected:
        raise RuntimeError(f"warm-start mismatch: missing={disallowed[:8]} unexpected={unexpected[:8]}")
    upgraded.model.eval()
    return upgraded


def _feature_contract(
    policy: EntityGraphPolicy, *, context_fill: float = 0.0
) -> dict[str, Any]:
    history_enabled, history_limit, history_schema = _policy_history_options(policy)
    return {
        "entity_feature_adapter_version": policy_entity_feature_adapter_version(policy),
        "public_observation": bool(
            getattr(policy, "trained_with_masked_hidden_info", False)
        ),
        "meaningful_public_history": bool(history_enabled),
        "meaningful_public_history_schema": str(history_schema),
        "event_history_limit": int(history_limit),
        "action_context_fill": float(context_fill),
        "public_award_feature_contract": str(
            policy.public_award_feature_contract
        ),
    }


def _root_outputs(
    policy: EntityGraphPolicy, game: Any, *, context_fill: float = 0.0
) -> dict[str, np.ndarray]:
    import torch

    feature_contract = _feature_contract(policy, context_fill=context_fill)
    acting_color = str(game.current_color())
    legal_actions = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
    policy_action_ids = rust_policy_action_ids(
        game, legal_actions, colors=COLORS, action_size=int(policy.action_size)
    )
    entity = rust_game_to_entity_batch(
        game, legal_actions, actor=acting_color, colors=COLORS,
        action_size=int(policy.action_size), policy_action_ids=policy_action_ids,
        public_observation=feature_contract["public_observation"],
        meaningful_public_history=feature_contract["meaningful_public_history"],
        history_limit=feature_contract["event_history_limit"],
        meaningful_public_history_schema=feature_contract[
            "meaningful_public_history_schema"
        ],
        entity_feature_adapter_version=feature_contract[
            "entity_feature_adapter_version"
        ],
    )
    context = rust_action_context_batch(
        game, legal_actions, actor=acting_color, colors=COLORS,
        action_size=int(policy.action_size), policy_action_ids=policy_action_ids,
        fill=feature_contract["action_context_fill"],
        public_observation=feature_contract["public_observation"],
        entity_feature_adapter_version=feature_contract[
            "entity_feature_adapter_version"
        ],
    )
    legal_ids = np.asarray(policy_action_ids, dtype=np.int64)[None, :]
    with torch.no_grad():
        outputs = policy.forward_legal_np(entity, legal_ids, context, return_q=True)
    logits = outputs["logits"].detach().float().cpu().numpy()[0]
    q_values = outputs["q_values"].detach().float().cpu().numpy()[0]
    return {"logits": logits, "q_values": q_values}


def _spread(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    probs = np.exp(values - values.max())
    probs = probs / probs.sum()
    top2 = np.sort(probs)[::-1][:2]
    return {
        "range": float(values.max() - values.min()),
        "std": float(values.std()),
        "softmax_top1_top2_gap": float(top2[0] - top2[1]) if top2.size > 1 else 0.0,
        "softmax_top1": float(top2[0]),
    }


def _aggregate(per_root: list[dict[str, Any]], field: str) -> dict[str, float]:
    ranges = [r[field]["range"] for r in per_root]
    stds = [r[field]["std"] for r in per_root]
    gaps = [r[field]["softmax_top1_top2_gap"] for r in per_root]
    return {
        "mean_range": float(np.mean(ranges)),
        "mean_std": float(np.mean(stds)),
        "mean_softmax_top1_top2_gap": float(np.mean(gaps)),
    }


def probe(
    policy: EntityGraphPolicy,
    games: list[Any],
    label: str,
    *,
    context_fill: float = 0.0,
) -> dict[str, Any]:
    per_root = []
    for game in games:
        out = _root_outputs(policy, game.copy(), context_fill=context_fill)
        per_root.append(
            {
                "n_candidates": int(out["logits"].shape[0]),
                "prior": _spread(out["logits"]),
                "q": _spread(out["q_values"]),
                "_logits": out["logits"],
                "_q": out["q_values"],
            }
        )
    return {
        "label": label,
        "n_roots": len(per_root),
        "prior_spread": _aggregate(per_root, "prior"),
        "q_spread": _aggregate(per_root, "q"),
        "per_root": [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in per_root
        ],
        "_raw": per_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-states", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=500001)
    parser.add_argument(
        "--context-fill",
        type=float,
        default=0.0,
        help="Action-context padding fill used for both compared policies.",
    )
    parser.add_argument(
        "--flags",
        default="gather,cross:2,value",
        help="upgrade flags for the compared policy, e.g. 'gather,cross:2,value'",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    catanatron_rs = _require_rust_module()
    games = find_placement_roots(
        catanatron_rs, n_states=int(args.n_states), base_seed=int(args.base_seed)
    )

    base = EntityGraphPolicy.load(args.checkpoint, device=args.device)
    base.model.eval()
    overrides = _parse_flags(args.flags)
    upgraded = _upgraded_policy_from(base, overrides)

    base_contract = _feature_contract(base, context_fill=float(args.context_fill))
    upgraded_contract = _feature_contract(
        upgraded, context_fill=float(args.context_fill)
    )
    if upgraded_contract != base_contract:
        raise RuntimeError(
            "function-preserving upgrade changed the feature contract: "
            f"base={base_contract!r} upgraded={upgraded_contract!r}"
        )
    base_result = probe(
        base, games, "base", context_fill=float(args.context_fill)
    )
    up_result = probe(
        upgraded, games, "upgraded", context_fill=float(args.context_fill)
    )

    warm_start_max_diff = 0.0
    for br, ur in zip(base_result["_raw"], up_result["_raw"]):
        warm_start_max_diff = max(
            warm_start_max_diff,
            float(np.abs(br["_logits"] - ur["_logits"]).max()),
            float(np.abs(br["_q"] - ur["_q"]).max()),
        )

    summary = {
        "checkpoint": args.checkpoint,
        "upgrade_flags": overrides,
        "feature_contract": base_contract,
        "n_roots": base_result["n_roots"],
        "base": {"prior_spread": base_result["prior_spread"], "q_spread": base_result["q_spread"]},
        "upgraded": {"prior_spread": up_result["prior_spread"], "q_spread": up_result["q_spread"]},
        "warm_start_max_diff": warm_start_max_diff,
        "warm_start_equivalent": warm_start_max_diff == 0.0,
    }
    for result in (base_result, up_result):
        result.pop("_raw", None)
    write_json(args.out, {"summary": summary, "base": base_result, "upgraded": up_result})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
