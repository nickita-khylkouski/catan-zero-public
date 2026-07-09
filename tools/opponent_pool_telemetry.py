#!/usr/bin/env python3
"""CAT-54 step 5: per-opponent telemetry (win rate, KL divergence, policy
entropy, value calibration), TRACKED SEPARATELY PER OPPONENT -- never
aggregated across the mix, per the ticket's explicit ask (mirrors
`tools/phase_sliced_value_calibration.py`'s per-slice-not-pooled convention,
just sliced by the CAT-54 `opponent_tag` shard column instead of phase).

Reads directly from a generation `--out-dir` (or any directory of `*.npz`
gumbel-self-play shards) -- no replay, no re-running search. Four independent
statistics per opponent tag:

  win_rate            fraction of that tag's DISTINCT games the producer/
                       champion won (naturally-terminated games only). Not
                       meaningful for "self"-sourced (mirror) tags -- both
                       seats are the producer there, so there is no single
                       "champion side" to grade a win rate against; reported
                       as null with an explicit note for those tags.
  policy_entropy       mean Shannon entropy (nats) of `target_policy` (the
                       post-search improved policy) over legal actions.
  kl_to_prior          mean KL(target_policy || prior_policy) -- how much the
                       search moved the policy away from the raw network
                       prior, per opponent. Requires shards written with
                       `prior_policy` (already true for every gumbel_self_play
                       shard; see `test_shard_prior_policy_round_trips_...`).
  value_calibration    OPTIONAL (only computed when --checkpoint is given):
                       corr(q, z) / Brier / value RMSE from a direct forward
                       pass over each row's stored entity features, exactly
                       `tools/phase_sliced_value_calibration.py`'s `compute_q`
                       -- restricted to naturally-terminated, non-truncated
                       games (the only rows with a clean +-1 label).

Rows from a shard directory with NO `opponent_tag` column at all (i.e. a run
that did not pass --opponent-mix-manifest) are bucketed under the single tag
``"untagged"`` -- this tool still works on ordinary (non-mix) self-play data,
it just reports one slice instead of several.
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

from factory_common import write_json

UNTAGGED = "untagged"

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


def _iter_shards(shard_dir: str) -> list[str]:
    shards = sorted(str(p) for p in Path(shard_dir).rglob("*.npz"))
    if not shards:
        raise SystemExit(f"no .npz shards found under {shard_dir}")
    return shards


def _row_entropy(target_policy: np.ndarray, mask: np.ndarray) -> float:
    p = target_policy[mask].astype(np.float64)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _row_kl_to_prior(target_policy: np.ndarray, prior_policy: np.ndarray, mask: np.ndarray) -> float:
    p = target_policy[mask].astype(np.float64)
    q = prior_policy[mask].astype(np.float64)
    valid = p > 0.0
    if not np.any(valid):
        return 0.0
    p = p[valid]
    q = np.clip(q[valid], 1e-8, None)
    return float(np.sum(p * np.log(p / q)))


def collect_rows(shard_dir: str, *, max_rows: int | None = None) -> dict[str, np.ndarray]:
    """Concatenate the per-row columns this tool needs across every shard.
    Entity features are collected too (kept per-shard-group implicitly via
    equal padding within one call, same rationale as
    `phase_sliced_value_calibration.collect_rows`) -- only actually forwarded
    through the network when `--checkpoint` is given."""
    tags: list[np.ndarray] = []
    is_pool: list[np.ndarray] = []
    versions: list[np.ndarray] = []
    game_seeds: list[np.ndarray] = []
    winners: list[np.ndarray] = []
    players: list[np.ndarray] = []
    terminated: list[np.ndarray] = []
    truncated: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    kls: list[np.ndarray] = []
    entity_groups: list[dict[str, np.ndarray]] = []
    legal_ids: list[np.ndarray] = []
    legal_ctx: list[np.ndarray] = []

    total = 0
    for shard_path in _iter_shards(shard_dir):
        data = np.load(shard_path)
        n = len(data["action_taken"]) if "action_taken" in data.files else 0
        if n == 0:
            continue
        legal_mask = data["legal_action_mask"].astype(bool)
        target_policy = data["target_policy"]
        prior_policy = (
            data["prior_policy"] if "prior_policy" in data.files else np.zeros_like(target_policy)
        )
        has_prior = "prior_policy" in data.files

        shard_tags = (
            np.array([str(t) for t in data["opponent_tag"]])
            if "opponent_tag" in data.files
            else np.array([UNTAGGED] * n)
        )
        shard_tags = np.array([t if t else UNTAGGED for t in shard_tags])
        shard_is_pool = (
            data["is_pool_game"].astype(bool) if "is_pool_game" in data.files else np.zeros(n, dtype=bool)
        )
        shard_versions = (
            data["opponent_version"].astype(int) if "opponent_version" in data.files else np.full(n, -1)
        )

        row_entropy = np.array([_row_entropy(target_policy[i], legal_mask[i]) for i in range(n)])
        row_kl = (
            np.array([_row_kl_to_prior(target_policy[i], prior_policy[i], legal_mask[i]) for i in range(n)])
            if has_prior
            else np.full(n, np.nan)
        )

        tags.append(shard_tags)
        is_pool.append(shard_is_pool)
        versions.append(shard_versions)
        game_seeds.append(data["game_seed"].astype(np.int64))
        winners.append(data["winner"])
        players.append(data["player"])
        terminated.append(data["terminated"].astype(bool))
        truncated.append(data["truncated"].astype(bool))
        entropies.append(row_entropy)
        kls.append(row_kl)
        legal_ids.append(data["legal_action_ids"])
        legal_ctx.append(data["legal_action_context"])
        entity_groups.append({key: data[key] for key in ENTITY_KEYS if key in data.files})

        total += n
        if max_rows is not None and total >= max_rows:
            break

    if not tags:
        raise SystemExit(f"no usable rows found under {shard_dir}")

    return {
        "opponent_tag": np.concatenate(tags),
        "is_pool_game": np.concatenate(is_pool),
        "opponent_version": np.concatenate(versions),
        "game_seed": np.concatenate(game_seeds),
        "winner": np.concatenate(winners),
        "player": np.concatenate(players),
        "terminated": np.concatenate(terminated),
        "truncated": np.concatenate(truncated),
        "policy_entropy": np.concatenate(entropies),
        "kl_to_prior": np.concatenate(kls),
        "_entity_groups": entity_groups,  # list of per-shard dicts, for optional value calibration
        "_legal_ids_by_shard": legal_ids,
        "_legal_ctx_by_shard": legal_ctx,
    }


def _win_rate_for_tag(rows: dict[str, np.ndarray], mask: np.ndarray) -> tuple[float | None, int]:
    """One row per DISTINCT game_seed (own-side-row filter guarantees every
    recorded row within one pool game shares the same `player`, so any row
    is representative), among naturally-terminated (not truncated) games."""
    terminal_mask = mask & rows["terminated"] & ~rows["truncated"]
    if not np.any(terminal_mask):
        return None, 0
    seeds = rows["game_seed"][terminal_mask]
    winners = rows["winner"][terminal_mask]
    players = rows["player"][terminal_mask]
    _, first_index = np.unique(seeds, return_index=True)
    n_games = len(first_index)
    wins = int(np.sum(winners[first_index] == players[first_index]))
    return wins / n_games, n_games


def _value_calibration_for_tag(
    rows: dict[str, np.ndarray], mask: np.ndarray, *, policy: Any, min_rows: int
) -> dict[str, Any]:
    import torch

    terminal_mask = mask & rows["terminated"] & ~rows["truncated"]
    n = int(np.sum(terminal_mask))
    if n == 0:
        return {"n": 0, "corr_q_z": None, "brier": None, "value_rmse": None}

    # Re-walk the per-shard entity groups, selecting only rows in this tag's
    # terminal mask -- entity features are stored per-shard (equal padding
    # within a shard), so the forward pass runs shard-by-shard exactly like
    # phase_sliced_value_calibration.compute_q.
    q_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    offset = 0
    for shard_index, entity_group in enumerate(rows["_entity_groups"]):
        shard_n = len(rows["_legal_ids_by_shard"][shard_index])
        shard_mask = terminal_mask[offset : offset + shard_n]
        offset += shard_n
        if not np.any(shard_mask):
            continue
        entity_batch = {key: value[shard_mask] for key, value in entity_group.items()}
        legal_ids = rows["_legal_ids_by_shard"][shard_index][shard_mask]
        legal_ctx = rows["_legal_ctx_by_shard"][shard_index][shard_mask]
        with torch.no_grad():
            outputs = policy.forward_legal_np(entity_batch, legal_ids, legal_ctx)
        q_chunks.append(outputs["value"].detach().cpu().numpy().reshape(-1))
        winner_shard = rows["winner"][offset - shard_n : offset][shard_mask]
        player_shard = rows["player"][offset - shard_n : offset][shard_mask]
        z_chunks.append(np.where(winner_shard == player_shard, 1.0, -1.0).astype(np.float32))

    if not q_chunks:
        return {"n": 0, "corr_q_z": None, "brier": None, "value_rmse": None}
    q = np.concatenate(q_chunks)
    z = np.concatenate(z_chunks)
    stats: dict[str, Any] = {"n": int(len(q))}
    win_mask = z > 0
    if len(q) < min_rows or not np.any(win_mask) or np.all(win_mask):
        stats["corr_q_z"] = None
    else:
        stats["corr_q_z"] = float(np.corrcoef(q, z)[0, 1])
    outcome = (z + 1.0) / 2.0
    p = np.clip((q + 1.0) / 2.0, 0.0, 1.0)
    stats["brier"] = float(np.mean((p - outcome) ** 2))
    stats["value_rmse"] = float(np.sqrt(np.mean((q - z) ** 2)))
    return stats


def compute_telemetry(
    shard_dir: str, *, checkpoint: str | None = None, device: str = "cpu", min_calibration_rows: int = 30
) -> dict[str, Any]:
    rows = collect_rows(shard_dir)
    tags = rows["opponent_tag"]

    policy = None
    if checkpoint:
        from catan_zero.rl.entity_token_policy import EntityGraphPolicy

        policy = EntityGraphPolicy.load(checkpoint, device=device)

    per_tag: dict[str, Any] = {}
    for tag in sorted(set(tags.tolist())):
        mask = tags == tag
        is_pool_tag = bool(np.any(rows["is_pool_game"][mask]))
        win_rate, n_games = (
            _win_rate_for_tag(rows, mask) if is_pool_tag else (None, 0)
        )
        entry: dict[str, Any] = {
            "n_rows": int(np.sum(mask)),
            "n_games": n_games,
            "is_pool_tag": is_pool_tag,
            "win_rate": win_rate,
            "win_rate_note": (
                None
                if is_pool_tag
                else "not meaningful for a mirror/self-play tag -- both seats are the producer"
            ),
            "policy_entropy_mean": float(np.mean(rows["policy_entropy"][mask])),
            "kl_to_prior_mean": (
                float(np.nanmean(rows["kl_to_prior"][mask]))
                if np.any(~np.isnan(rows["kl_to_prior"][mask]))
                else None
            ),
        }
        if policy is not None:
            entry["value_calibration"] = _value_calibration_for_tag(
                rows, mask, policy=policy, min_rows=min_calibration_rows
            )
        per_tag[tag] = entry

    return {"shard_dir": shard_dir, "checkpoint": checkpoint, "per_opponent": per_tag}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True, help="dir searched recursively for *.npz")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional: compute value calibration too (forward pass, needs torch). Omit for a "
        "cheap win-rate/entropy/KL-only report.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-calibration-rows", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    summary = compute_telemetry(
        args.shard_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        min_calibration_rows=args.min_calibration_rows,
    )
    if args.out:
        write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
