#!/usr/bin/env python3
"""Post-repair calibration probe (protocol step 1, 2026-07-04).

Evaluates a checkpoint's value head, via a direct forward pass (no search,
no live game replay -- the entity-token features are already stored per row
in the raw-selfplay shards), over the HELD-OUT game_seed ranges of the
value-repair-v2 raw-selfplay corpus, and reports corr(q, z) against the
true game outcome plus E[q|win]/E[q|loss], matching the original pilot
probe's methodology (corr=0.61 baseline, E[q|win]=+0.41, E[q|loss]=-0.16).

Held-out ranges are the LAST 332 games (of 6667) from each of the 4
raw-selfplay generator directories that fed the training subset -- these
tail slices were excluded from the training subset by construction (see
docs/catan_postrepair_revalidation_protocol_20260704.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from factory_common import write_json

# (manifest_dir, base_seed, holdout_start_inclusive, holdout_end_exclusive)
DEFAULT_HOLDOUT_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("runs/selfplay/raw_selfplay_gen_20260704/b200_gpu0", 5006335, 5006667),
    ("runs/selfplay/raw_selfplay_gen_20260704/b200_gpu1", 5106335, 5106667),
    ("runs/selfplay/raw_selfplay_gen_20260704/a100b_gpu0", 7006335, 7006667),
    ("runs/selfplay/raw_selfplay_gen_20260704/a100b_gpu1", 7106335, 7106667),
)

ENTITY_KEYS = (
    "hex_tokens",
    "hex_vertex_ids",
    "hex_edge_ids",
    "vertex_tokens",
    "edge_tokens",
    "edge_vertex_ids",
    "player_tokens",
    "global_tokens",
    "legal_action_tokens",
    "legal_action_target_ids",
    "event_tokens",
    "event_target_ids",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "legal_action_mask",
    "event_mask",
)


def _iter_holdout_shards(manifest_dir: str) -> list[str]:
    manifest_path = Path(manifest_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(manifest["shards"])


def collect_holdout_rows(
    holdout_blocks: tuple[tuple[str, int, int], ...], *, max_rows: int | None = None
) -> list[dict[str, np.ndarray]]:
    """Return one shard-group per (manifest_dir, start, end) block: the
    stored ENTITY_KEYS + legal_action_ids/legal_action_context (needed
    verbatim -- the model asserts legal_action_tokens' width matches
    legal_action_ids' width, and that padding width varies per shard) for
    rows whose game_seed falls in [start, end) and whose game TERMINATED
    naturally (not truncated -- truncated games have no clean +-1 outcome
    for this probe). Grouping by shard (not one global concat) avoids
    having to re-pad legal_action_ids/context to a common width across
    shards with different legal-action-count paddings."""
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
            group = {key: data[key][idx] for key in ENTITY_KEYS}
            group["legal_action_ids"] = data["legal_action_ids"][idx]
            group["legal_action_context"] = data["legal_action_context"][idx]
            group["z"] = z
            group["game_seed"] = game_seed[idx]
            groups.append(group)
            total += len(idx)
            if max_rows is not None and total >= max_rows:
                return groups
    if not groups:
        raise SystemExit("no held-out rows found -- check holdout ranges/manifests")
    return groups


def compute_q(policy: EntityGraphPolicy, groups: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Forward-pass each shard-group (already a consistent legal-action
    padding width within a group) and return concatenated (q, z)."""
    import torch

    q_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    for group in groups:
        entity_batch = {key: group[key] for key in ENTITY_KEYS}
        with torch.no_grad():
            outputs = policy.forward_legal_np(
                entity_batch,
                group["legal_action_ids"],
                group["legal_action_context"],
            )
        q_chunks.append(outputs["value"].detach().cpu().numpy().reshape(-1))
        z_chunks.append(group["z"])
    return np.concatenate(q_chunks, axis=0), np.concatenate(z_chunks, axis=0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    groups = collect_holdout_rows(DEFAULT_HOLDOUT_BLOCKS, max_rows=args.max_rows)
    policy = EntityGraphPolicy.load(args.checkpoint, device=args.device)

    q, z = compute_q(policy, groups)

    corr = float(np.corrcoef(q, z)[0, 1])
    win_mask = z > 0
    e_q_win = float(q[win_mask].mean()) if np.any(win_mask) else None
    e_q_loss = float(q[~win_mask].mean()) if np.any(~win_mask) else None

    summary = {
        "checkpoint": args.checkpoint,
        "n_rows": int(len(z)),
        "n_win": int(win_mask.sum()),
        "n_loss": int((~win_mask).sum()),
        "corr_q_z": corr,
        "e_q_given_win": e_q_win,
        "e_q_given_loss": e_q_loss,
        "q_mean": float(q.mean()),
        "q_std": float(q.std()),
        "holdout_blocks": [
            {"manifest_dir": d, "start": s, "end": e} for d, s, e in DEFAULT_HOLDOUT_BLOCKS
        ],
    }
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
