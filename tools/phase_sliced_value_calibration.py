#!/usr/bin/env python3
"""Phase-sliced value-head calibration (f70 D4).

Takes any self-play shard directory (the `*.npz` shards written by
`gumbel_self_play` / the raw-selfplay generators) plus a checkpoint, and
reports value-head calibration -- corr(q, z) and the Brier score -- GLOBALLY
and SLICED by game phase and by legal-action-count bucket. The Gate-A
post-mortem showed global corr(q, z) hides the failure that actually matters
for search: the value head can be well-calibrated on average yet rank
candidates by noise at wide placement roots. Per-phase / per-legal-count
calibration is the diagnostic that exposes that.

`q` is the value-head prediction from a direct forward pass over the entity
features stored per row (no search, no game replay), exactly as
`tools/value_repair_calibration_probe.py` does; `z` is the true terminal
outcome (+1 win / -1 loss) for the row's acting player. Only rows whose game
TERMINATED naturally (not truncated) are used -- truncated games have no
clean +-1 label.

PHASE SLICING NOTE: the stored `phase` column is coarse -- ROLL, dev-card
plays, builds, trades and END_TURN all live under a single "PLAY_TURN"
value, so a true dev-vs-build split is NOT recoverable from the shard alone
(it would require decoding `action_taken` back to an action type by replaying
from `game_seed`, out of scope for a lightweight analysis tool). We therefore
slice by the real stored `phase` vocab (opening placement / robber / discard /
play-turn), plus a cross-cutting `forced` slice (the stored `is_forced`
flag), plus legal-action-count buckets. If a finer dev/build split is needed
later, decode `action_taken` per row.
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

# Same entity feature keys the model consumes, matching
# `tools/value_repair_calibration_probe.py`.
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

# Map the coarse stored `phase` values to friendly slice labels.
_PHASE_LABELS = {
    "BUILD_INITIAL_SETTLEMENT": "opening_placement",
    "BUILD_INITIAL_ROAD": "opening_placement",
    "MOVE_ROBBER": "robber",
    "DISCARD": "discard",
    "PLAY_TURN": "play_turn",
}

# Legal-action-count buckets (upper bound inclusive; the widest placement
# roots are 54-wide, per F8).
_LEGAL_BUCKETS = ((1, "1"), (4, "2-4"), (12, "5-12"), (30, "13-30"), (53, "31-53"), (54, "54"))


def _legal_bucket(count: int) -> str:
    for upper, label in _LEGAL_BUCKETS:
        if count <= upper:
            return label
    return f">{_LEGAL_BUCKETS[-1][0]}"


def _iter_shards(shard_dir: str) -> list[str]:
    root = Path(shard_dir)
    shards = sorted(str(p) for p in root.rglob("*.npz"))
    if not shards:
        raise SystemExit(f"no .npz shards found under {shard_dir}")
    return shards


def collect_rows(shard_dir: str, *, max_rows: int | None = None) -> list[dict[str, np.ndarray]]:
    """One group per shard (each shard has a self-consistent legal-action
    padding width -- see the calibration probe's grouping rationale), holding
    the entity features + legal-action arrays needed for the forward pass and
    the per-row slice keys (phase label, forced flag, legal count) and z."""
    groups: list[dict[str, np.ndarray]] = []
    total = 0
    for shard_path in _iter_shards(shard_dir):
        data = np.load(shard_path)
        if "terminated" not in data.files:
            continue
        terminated = data["terminated"] & ~data["truncated"]
        if not np.any(terminated):
            continue
        idx = np.where(terminated)[0]
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

        group: dict[str, np.ndarray] = {key: data[key][idx] for key in ENTITY_KEYS}
        group["legal_action_ids"] = data["legal_action_ids"][idx]
        group["legal_action_context"] = data["legal_action_context"][idx]
        group["z"] = z
        group["phase_label"] = phase_labels
        group["forced"] = forced
        group["legal_count"] = legal_count
        groups.append(group)
        total += len(idx)
        if max_rows is not None and total >= max_rows:
            break
    if not groups:
        raise SystemExit("no naturally-terminated rows found in shard dir")
    return groups


def compute_q(policy: EntityGraphPolicy, groups: list[dict[str, np.ndarray]]) -> np.ndarray:
    import torch

    q_chunks: list[np.ndarray] = []
    for group in groups:
        entity_batch = {key: group[key] for key in ENTITY_KEYS}
        with torch.no_grad():
            outputs = policy.forward_legal_np(
                entity_batch, group["legal_action_ids"], group["legal_action_context"]
            )
        q_chunks.append(outputs["value"].detach().cpu().numpy().reshape(-1))
    return np.concatenate(q_chunks, axis=0)


def _calibration_stats(q: np.ndarray, z: np.ndarray, *, min_rows: int) -> dict[str, Any]:
    n = int(len(z))
    win_mask = z > 0
    n_win = int(win_mask.sum())
    n_loss = int((~win_mask).sum())
    stats: dict[str, Any] = {
        "n": n,
        "n_win": n_win,
        "n_loss": n_loss,
        "win_rate": (n_win / n) if n else None,
        "q_mean": float(q.mean()) if n else None,
        "q_std": float(q.std()) if n else None,
    }
    if n < min_rows or n_win == 0 or n_loss == 0:
        # corr is undefined without both classes / enough rows.
        stats["corr_q_z"] = None
    else:
        stats["corr_q_z"] = float(np.corrcoef(q, z)[0, 1])
    stats["e_q_given_win"] = float(q[win_mask].mean()) if n_win else None
    stats["e_q_given_loss"] = float(q[~win_mask].mean()) if n_loss else None
    # Brier: outcome in {0,1}, predicted prob p = (q+1)/2 clipped to [0,1].
    if n:
        outcome = (z + 1.0) / 2.0
        p = np.clip((q + 1.0) / 2.0, 0.0, 1.0)
        stats["brier"] = float(np.mean((p - outcome) ** 2))
        # Value-space residual RMSE (q vs the +-1 outcome). This is the
        # recommended per-checkpoint / per-phase estimate for the search's
        # `sigma_eval` noise-floor knob (D1): the opening_placement slice's
        # value_rmse is the relevant sigma for the noise floor at wide
        # placement roots. It is an UPPER bound on the pure estimator noise
        # (it also absorbs the irreducible outcome variance given a state),
        # but it is the standard, directly-usable practical proxy.
        stats["value_rmse"] = float(np.sqrt(np.mean((q - z) ** 2)))
    else:
        stats["brier"] = None
        stats["value_rmse"] = None
    return stats


def _slice_by(
    q: np.ndarray, z: np.ndarray, keys: np.ndarray, *, min_rows: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        out[str(key)] = _calibration_stats(q[mask], z[mask], min_rows=min_rows)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True, help="dir searched recursively for *.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--min-slice-rows",
        type=int,
        default=30,
        help="minimum rows in a slice before corr(q,z) is reported (else null)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    groups = collect_rows(args.shard_dir, max_rows=args.max_rows)
    policy = EntityGraphPolicy.load(args.checkpoint, device=args.device)

    q = compute_q(policy, groups)
    z = np.concatenate([g["z"] for g in groups], axis=0)
    phase = np.concatenate([g["phase_label"] for g in groups], axis=0)
    forced = np.concatenate([g["forced"] for g in groups], axis=0)
    legal_count = np.concatenate([g["legal_count"] for g in groups], axis=0)
    legal_bucket = np.array([_legal_bucket(int(c)) for c in legal_count])
    forced_label = np.where(forced, "forced", "unforced")

    summary = {
        "checkpoint": args.checkpoint,
        "shard_dir": args.shard_dir,
        "global": _calibration_stats(q, z, min_rows=args.min_slice_rows),
        "by_phase": _slice_by(q, z, phase, min_rows=args.min_slice_rows),
        "by_forced": _slice_by(q, z, forced_label, min_rows=args.min_slice_rows),
        "by_legal_count_bucket": _slice_by(q, z, legal_bucket, min_rows=args.min_slice_rows),
    }
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
