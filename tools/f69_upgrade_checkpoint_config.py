#!/usr/bin/env python3
"""Flip the f69 action-attention flags ON in an entity_graph checkpoint's
config, warm-starting all shared weights and zero-initialising the new params,
and re-save it as a new checkpoint.

This is the *mechanical enabler* for the v3b finetune: `tools/train_bc.py` has
no CLI argument for the new EntityGraphConfig flags, and its `--init-checkpoint`
path rebuilds the module from the checkpoint's own pickled config. So instead of
touching train_bc, we produce an upgraded-config checkpoint here and point the
IDENTICAL v3a command at it via `--init-checkpoint`. train_bc's
`EntityGraphPolicy.load` then reads the upgraded config (flags ON), builds the
upgraded module, and loads these weights strictly (the new zero-init params are
already present in this checkpoint, so nothing is missing).

The output is behaviourally identical to the input at init (every upgrade path
is zero-initialised on its output), which this script asserts on a real
placement root before writing -- see `docs/f69_v3b_finetune_launch.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_SRC = (_TOOLS_DIR.parent / "src").resolve(strict=True)
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _REPO_SRC]
sys.path.insert(0, str(_REPO_SRC))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)
from catan_zero.rl.meaningful_history import (  # noqa: E402
    MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
)

_ENTITY_POLICY_MODULE = sys.modules[EntityGraphPolicy.__module__]
_ENTITY_POLICY_PATH = Path(str(_ENTITY_POLICY_MODULE.__file__)).resolve(strict=True)
if _REPO_SRC not in _ENTITY_POLICY_PATH.parents:
    raise RuntimeError(
        "checkpoint upgrader imported catan_zero outside its checkout: "
        f"{_ENTITY_POLICY_PATH}"
    )

# The exact param prefixes introduced by the three upgrades (see
# entity_token_policy.EntityGraphNet). Must equal the load() allow-list.
NEW_PARAM_PREFIXES = (
    "target_gather_proj.",
    "action_cross_blocks.",
    "value_probe",
    "value_pool_head.",
    # CAT-97 edge-feature policy head + CAT-100 aux subgoal heads.
    "edge_policy_mlp.",
    "aux_longest_road_head.",
    "aux_largest_army_head.",
    "aux_vp_in_n_head.",
    "aux_next_settlement_head.",
    "aux_robber_target_head.",
    "aux_next_settlement_pointer_head.",
    "belief_resource_head.",
    "value_categorical_head.",
    "topology_residual_adapter.",
    "static_action_residual_proj.",
    "public_card_count_residual.",
    "meaningful_history_residual_gate",
)


def _build_upgraded_config(
    base_config, overrides: dict[str, object]
) -> EntityGraphConfig:
    """Reconstruct an EntityGraphConfig from a possibly-STALE base config.

    `dataclasses.replace(base_config, **overrides)` reads EVERY current field
    off `base_config`, so a config pickled before a field existed makes it
    raise AttributeError. This is exactly what happens to a seed checkpoint
    whose config predates both the f69 flags AND other later fields (e.g.
    f67's value_uncertainty_head): replace tries to read a field the stale
    object never had. Instead: copy the fields that DO exist, let the dataclass
    fill any the stale pickle lacks from its current defaults, then apply the
    flag overrides. Correct for arbitrary past-or-future config drift.
    """
    base_dict = {
        f.name: getattr(base_config, f.name)
        for f in fields(EntityGraphConfig)
        if hasattr(base_config, f.name)
    }
    base_dict.update(overrides)
    return EntityGraphConfig(**base_dict)


def _parse_flags(raw: str) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for entry in (piece.strip() for piece in raw.split(",") if piece.strip()):
        if entry in ("gather", "action_target_gather"):
            overrides["action_target_gather"] = True
        elif entry in ("value", "value_attention_pool"):
            overrides["value_attention_pool"] = True
        elif entry.startswith("cross"):
            n = entry.split(":", 1)[1] if ":" in entry else "2"
            overrides["action_cross_attention_layers"] = int(n)
        elif entry in ("edge", "edge_policy_head"):
            overrides["edge_policy_head"] = True
        elif entry in ("aux", "aux_subgoal_heads"):
            overrides["aux_subgoal_heads"] = True
        elif entry in (
            "aux_settlement_pointer",
            "aux_settlement_pointer_head",
        ):
            overrides["aux_subgoal_heads"] = True
            overrides["aux_settlement_pointer_head"] = True
        elif entry in ("topology", "topology_residual_adapter"):
            overrides["topology_residual_adapter"] = True
        elif entry in ("belief", "belief_resource_head"):
            overrides["belief_resource_head"] = True
        elif entry in ("static", "static_action_residual"):
            overrides["static_action_residual"] = True
        elif entry in ("card_count", "public_card_count_features"):
            overrides["public_card_count_features"] = True
        elif entry in (
            "card_count_v2",
            "public_card_count_features_v2",
            "bias_free_card_count",
        ):
            overrides["public_card_count_features"] = True
            overrides["public_card_count_residual_bias"] = False
        elif entry in (
            "history",
            "meaningful_history",
            "meaningful_public_history",
        ):
            overrides["meaningful_public_history"] = True
            overrides["meaningful_public_history_schema"] = (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            )
            overrides["event_history_limit"] = MEANINGFUL_PUBLIC_HISTORY_LIMIT
        elif entry.startswith("catbins"):
            # CAT-39: build the HL-Gauss categorical value head with N win-loss
            # bins (plus the truncation class, which the config enables by
            # default). Zero-initialised? No -- but purely ADDITIVE: the scalar
            # value/final_vp/q outputs stay bit-identical (the new head only adds
            # value_categorical* outputs), so the forward-identity assertion below
            # still holds on those keys.
            n = entry.split(":", 1)[1] if ":" in entry else "33"
            overrides["value_categorical_bins"] = int(n)
        else:
            raise SystemExit(f"unknown upgrade flag: {entry!r}")
    return overrides


def _verify_forward_identical(
    base: EntityGraphPolicy, upgraded: EntityGraphPolicy, device: str
) -> float:
    """Max abs diff of logits/q over one real 54-wide placement root."""
    import torch

    from catan_zero.search.neural_rust_mcts import (
        rust_action_context_batch,
        rust_game_to_entity_batch,
        rust_policy_action_ids,
    )
    from catan_zero.search.rust_mcts import _require_rust_module
    from sigma_trace_placement_root import COLORS, find_placement_roots

    catanatron_rs = _require_rust_module()
    game = find_placement_roots(catanatron_rs, n_states=1, base_seed=500001)[0]
    acting_color = str(game.current_color())
    legal_actions = tuple(
        int(a) for a in game.playable_action_indices(list(COLORS), None)
    )
    pids = rust_policy_action_ids(
        game, legal_actions, colors=COLORS, action_size=int(base.action_size)
    )
    base_entity = rust_game_to_entity_batch(
        game,
        legal_actions,
        actor=acting_color,
        colors=COLORS,
        action_size=int(base.action_size),
        policy_action_ids=pids,
    )
    upgraded_entity = rust_game_to_entity_batch(
        game,
        legal_actions,
        actor=acting_color,
        colors=COLORS,
        action_size=int(base.action_size),
        policy_action_ids=pids,
        meaningful_public_history=bool(
            getattr(upgraded.config, "meaningful_public_history", False)
        ),
        history_limit=int(getattr(upgraded.config, "event_history_limit", 64)),
    )
    context = rust_action_context_batch(
        game,
        legal_actions,
        actor=acting_color,
        colors=COLORS,
        action_size=int(base.action_size),
        policy_action_ids=pids,
    )
    legal_ids = np.asarray(pids, dtype=np.int64)[None, :]
    max_diff = 0.0
    with torch.no_grad():
        ob = base.forward_legal_np(base_entity, legal_ids, context, return_q=True)
        ou = upgraded.forward_legal_np(
            upgraded_entity, legal_ids, context, return_q=True
        )
        for key in ("logits", "value", "final_vp", "q_values"):
            max_diff = max(max_diff, float((ob[key] - ou[key]).abs().max().item()))
    return max_diff


def _preserve_source_top_level_keys(
    in_checkpoint: str,
    out_checkpoint: str,
    *,
    mutated_keys: tuple[str, ...] = ("model", "config"),
) -> list[str]:
    """CAT-80: ``EntityGraphPolicy.save()`` rebuilds the checkpoint from the
    freshly-constructed ``upgraded`` policy and does NOT carry over the source
    checkpoint's top-level provenance keys (``mask_hidden_info``,
    ``action_mask_version``, ``static_action_features*``, ``policy_type`` ...).
    In particular ``mask_hidden_info`` silently reset True->False, mislabeling a
    masked net as omniscient -- the exact #71 hidden-info-leak class the #76
    masked-regime guard exists to catch.

    Re-open both checkpoints and restore every top-level key from the SOURCE
    except the ones this upgrade intentionally mutates (``model`` weights and the
    ``config`` flags). Returns the sorted list of preserved source keys.
    """
    import torch

    in_raw = torch.load(in_checkpoint, map_location="cpu", weights_only=False)
    out_raw = torch.load(out_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(in_raw, dict) or not isinstance(out_raw, dict):
        return []
    merged = dict(in_raw)
    for key in mutated_keys:
        if key in out_raw:
            merged[key] = out_raw[key]
    torch.save(merged, out_checkpoint)
    return sorted(k for k in in_raw if k not in mutated_keys)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_upgrade_provenance(
    out_checkpoint: str,
    *,
    in_checkpoint: str,
    flags: dict[str, object],
    seed: int,
    forward_max_diff: float | None,
) -> None:
    """Atomically attest how freshly initialized upgrade modules were built."""

    import torch

    output = Path(out_checkpoint)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": _sha256_file(in_checkpoint),
        "flags": dict(flags),
        "initialization_seed": int(seed),
        "trained_value_readouts_added": [],
        "forward_max_diff": forward_max_diff,
        "forward_identical_at_init": (
            forward_max_diff == 0.0 if forward_max_diff is not None else False
        ),
    }
    tmp = output.with_name(f".{output.name}.upgrade.tmp.{os.getpid()}")
    try:
        torch.save(raw, tmp)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-checkpoint", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    parser.add_argument("--flags", default="gather,cross:2,value")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deterministic initialization seed for every newly added module.",
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    overrides = _parse_flags(args.flags)
    import torch

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    base = EntityGraphPolicy.load(args.in_checkpoint, device=args.device)
    base.model.eval()

    upgraded_config = _build_upgraded_config(base.config, overrides)
    static = base.static_action_features.detach().cpu().numpy()
    # EntityGraphPolicy owns model initialization and resets Torch's RNG from
    # its ``seed`` argument.  Passing no seed here silently reset every upgrade
    # to seed 0 even though the CLI/provenance recorded ``--seed``.
    upgraded = EntityGraphPolicy(
        upgraded_config,
        static,
        seed=int(args.seed),
        device=args.device,
    )
    missing, unexpected = upgraded.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    disallowed = [
        k for k in missing if not k.startswith(NEW_PARAM_PREFIXES + ("q_head.",))
    ]
    if disallowed or unexpected:
        raise SystemExit(
            f"warm-start mismatch: missing={disallowed[:8]} unexpected={unexpected[:8]}"
        )
    upgraded.model.eval()

    max_diff = None
    if not args.no_verify:
        max_diff = _verify_forward_identical(base, upgraded, args.device)
        if max_diff != 0.0:
            raise SystemExit(
                f"forward not identical at init: max_diff={max_diff} (expected 0.0)"
            )

    upgraded.save(args.out_checkpoint)
    # CAT-80: restore top-level provenance keys the fresh-policy save() drops
    # (mask_hidden_info et al.); only model weights + config flags are mutated.
    preserved_source_keys = _preserve_source_top_level_keys(
        args.in_checkpoint, args.out_checkpoint
    )
    _record_upgrade_provenance(
        args.out_checkpoint,
        in_checkpoint=args.in_checkpoint,
        flags=overrides,
        seed=int(args.seed),
        forward_max_diff=max_diff,
    )
    print(
        json.dumps(
            {
                "in_checkpoint": args.in_checkpoint,
                "out_checkpoint": args.out_checkpoint,
                "flags": overrides,
                "new_params_added": sorted(
                    set(k for k in missing if k.startswith(NEW_PARAM_PREFIXES))
                ),
                "forward_max_diff": max_diff,
                "forward_identical_at_init": (max_diff == 0.0)
                if max_diff is not None
                else "skipped",
                "preserved_source_keys": preserved_source_keys,
                "initialization_seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
