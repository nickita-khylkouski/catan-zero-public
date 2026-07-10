#!/usr/bin/env python3
"""Phase-sliced calibration on a TRUE held-out game-seed range (task #78).

`tools/phase_sliced_value_calibration.py`'s `collect_rows()` reads every
*.npz under a shard-dir with NO game-seed filtering -- fine for a quick
diagnostic, but not valid for a train-vs-holdout comparison, since a
directory a checkpoint trained on will include the exact rows it was
fitted to (calibration on those rows is optimistic, not a genuine
held-out read). `tools/value_repair_calibration_probe.py`'s
`collect_holdout_rows()` does the correct game-seed-range filtering (the
same DEFAULT_HOLDOUT_BLOCKS already used for the v3a/v3b masked-training
verification), but doesn't carry the `phase`/`is_forced`/legal-count
columns phase_sliced_value_calibration needs for slicing.

This module is the join of the two: the probe's held-out filtering logic,
extended to also carry the slicing columns, feeding directly into
phase_sliced_value_calibration's existing (imported, not duplicated) readout,
provenance, calibration, and slicing machinery. The scalar readout remains the
default; categorical calibration uses the same fail-closed checkpoint
provenance contract as the non-holdout tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from value_repair_calibration_probe import ENTITY_KEYS, _iter_holdout_shards
from phase_sliced_value_calibration import _PHASE_LABELS


def collect_holdout_rows_with_slices(
    holdout_blocks: tuple[tuple[str, int, int], ...], *, max_rows: int | None = None
) -> list[dict[str, np.ndarray]]:
    """Same held-out (manifest_dir, start, end) filtering as
    `value_repair_calibration_probe.collect_holdout_rows`, but each returned
    group ALSO carries `phase_label`, `forced`, and `legal_count` so the
    result can feed `phase_sliced_value_calibration`'s slicing/stats
    functions directly."""
    groups: list[dict[str, np.ndarray]] = []
    total = 0
    for manifest_dir, start, end in holdout_blocks:
        for shard_path in _iter_holdout_shards(manifest_dir):
            data = np.load(shard_path)
            game_seed = data["game_seed"]
            in_range = (game_seed >= start) & (game_seed < end)
            if not np.any(in_range):
                continue
            terminated = data["terminated"] & ~data["truncated"]
            keep = in_range & terminated
            if not np.any(keep):
                continue
            idx = np.where(keep)[0]
            winner = data["winner"][idx]
            player = data["player"][idx]
            z = np.where(winner == player, 1.0, -1.0).astype(np.float32)

            phases = data["phase"][idx]
            phase_labels = np.array([_PHASE_LABELS.get(str(p), str(p)) for p in phases])
            forced = (
                data["is_forced"][idx].astype(bool)
                if "is_forced" in data.files
                else np.zeros(len(idx), dtype=bool)
            )
            legal_count = data["legal_action_mask"][idx].sum(axis=1).astype(int)

            group = {key: data[key][idx] for key in ENTITY_KEYS}
            group["legal_action_ids"] = data["legal_action_ids"][idx]
            group["legal_action_context"] = data["legal_action_context"][idx]
            group["z"] = z
            group["game_seed"] = game_seed[idx]
            group["phase_label"] = phase_labels
            group["forced"] = forced
            group["legal_count"] = legal_count
            groups.append(group)
            total += len(idx)
            if max_rows is not None and total >= max_rows:
                return groups
    if not groups:
        raise SystemExit("no held-out rows found -- check holdout ranges/manifests")
    return groups


def resolve_use_masking(explicit: bool | None, policy) -> bool:
    """Decide whether to mask player_tokens before the forward pass.

    `explicit` (the --mask-hidden-info / --no-mask-hidden-info CLI flag)
    always wins when set. Otherwise auto-detect from
    `policy.trained_with_masked_hidden_info` -- same fail-closed-to-False
    default as the #76 safety net, so an unannotated legacy checkpoint is
    evaluated unmasked (its actual training regime) rather than guessed."""
    if explicit is not None:
        return explicit
    return bool(getattr(policy, "trained_with_masked_hidden_info", False))


def main() -> None:
    import argparse
    import json

    from value_repair_calibration_probe import DEFAULT_HOLDOUT_BLOCKS
    from phase_sliced_value_calibration import (
        build_calibration_summary,
        compute_readout,
    )
    from catan_zero.rl.entity_token_features import mask_player_tokens_public
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from factory_common import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--value-readout",
        choices=("scalar", "categorical"),
        default="scalar",
        help=(
            "Value expectation to calibrate. Categorical fails closed unless "
            "the checkpoint has positive value-training-v1 provenance."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-slice-rows", type=int, default=30)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument(
        "--mask-hidden-info",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Mask player_tokens to public-only before the forward pass, matching "
            "the checkpoint's OWN training regime (f72). Default: auto-detect from "
            "policy.trained_with_masked_hidden_info (fails closed to unmasked for "
            "legacy checkpoints, same as the #76 safety net) -- override only if "
            "you have independently verified the checkpoint's real training regime, "
            "since evaluating a masked-trained net on unmasked inputs (or vice "
            "versa) is off-distribution and produces a systematically WORSE "
            "corr(q,z), exactly the confound the masked-vs-unmasked control test "
            "(f28d10e) was built to catch."
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    groups = collect_holdout_rows_with_slices(
        DEFAULT_HOLDOUT_BLOCKS, max_rows=args.max_rows
    )
    policy = EntityGraphPolicy.load(args.checkpoint, device=args.device)

    use_masking = resolve_use_masking(args.mask_hidden_info, policy)
    if use_masking:
        for group in groups:
            group["player_tokens"] = mask_player_tokens_public(group["player_tokens"])

    predictions = compute_readout(policy, groups, value_readout=args.value_readout)
    summary = build_calibration_summary(
        predictions,
        groups,
        min_slice_rows=args.min_slice_rows,
        reliability_bin_count=args.reliability_bins,
    )
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "mask_hidden_info_applied": bool(use_masking),
            "holdout_blocks": list(DEFAULT_HOLDOUT_BLOCKS),
        }
    )
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
