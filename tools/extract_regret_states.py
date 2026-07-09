#!/usr/bin/env python3
"""Extract high-regret archived states from self-play shards (task #64).

Streams a shard corpus one shard at a time (NEVER concatenates -- the raw
corpus is ~48M rows / 680 GB), scores every non-forced row with a configurable
regret score (see `regret_common.RegretConfig` and
`docs/regret_restart_mixing_recipe.md`), and writes:

  * a regret MANIFEST (.npz of parallel arrays) of the top-K rows sorted by
    score, each identified by (shard_path, row_index, game_seed,
    decision_index) plus the per-component breakdown -- enough for
    `tools/reconstruct_state.py` to replay each state, and
  * a JSON SUMMARY sidecar: score distribution (histogram + quantiles),
    candidate/forced counts, phase mix over the whole corpus AND over the
    top-K, opening-placement fraction, and legal-count / VP-context stats.

Memory is bounded: a size-K min-heap holds the current best rows; streaming
histograms/counters accumulate global stats. Raw shards (no searched Q) get
value_surprise from a batched value-head pass over their STORED entity tokens
(`--value-checkpoint`), so no Rust re-featurisation and no full-corpus load.
"""

from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from regret_common import (
    RegretConfig,
    StoredFeatureValuer,
    discover_shards,
    load_shard,
    score_shard,
    write_json_atomic,
)

# Fixed histogram support for the streaming score distribution. Scores are a
# small non-negative sum of ~[0,1] components; 0..4 covers the weighted range
# comfortably (out-of-range values are clipped into the end bins).
_HIST_LO, _HIST_HI, _HIST_BINS = 0.0, 4.0, 80


def _teacher_of(shard: dict[str, np.ndarray]) -> str:
    if "teacher_name" not in shard:
        return ""
    return str(np.asarray(shard["teacher_name"]).reshape(-1)[0])


def _is_raw_shard(shard: dict[str, np.ndarray]) -> bool:
    """A shard with no searched Q anywhere (raw_selfplay) needs the value pass."""
    if "target_scores_mask" not in shard:
        return True
    return not bool(np.asarray(shard["target_scores_mask"]).any())


class _TopK:
    """Bounded min-heap of the highest-scoring rows seen so far."""

    def __init__(self, k: int) -> None:
        self.k = int(k)
        self._heap: list[tuple[float, int, dict[str, Any]]] = []
        self._counter = 0

    def offer(self, score: float, record: dict[str, Any]) -> None:
        self._counter += 1
        item = (float(score), self._counter, record)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, item)
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def sorted_desc(self) -> list[dict[str, Any]]:
        return [rec for _score, _c, rec in sorted(self._heap, key=lambda t: t[0], reverse=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract high-regret states from self-play shards.")
    parser.add_argument("--shard-root", action="append", required=True, help="repeatable")
    parser.add_argument("--out", required=True, help="output .npz manifest path")
    parser.add_argument("--top-k", type=int, default=200_000)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--teacher-filter", default=None, help="only score shards whose teacher_name matches")
    # Value pass for raw shards.
    parser.add_argument("--value-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--value-squash", choices=("tanh", "clip"), default="tanh")
    parser.add_argument("--value-batch-size", type=int, default=4096)
    # Regret weights.
    parser.add_argument("--w-value-surprise", type=float, default=1.0)
    parser.add_argument("--w-phase-bonus", type=float, default=0.4)
    parser.add_argument("--w-legal-count", type=float, default=0.2)
    parser.add_argument("--w-kl-disagreement", type=float, default=0.5)
    parser.add_argument("--w-argmax-mismatch-lost", type=float, default=0.4)
    parser.add_argument("--include-forced", action="store_true")
    parser.add_argument("--max-shards", type=int, default=0, help="0 = all (debug cap)")
    args = parser.parse_args()

    config = RegretConfig(
        value_surprise_weight=args.w_value_surprise,
        phase_bonus_weight=args.w_phase_bonus,
        legal_count_weight=args.w_legal_count,
        kl_disagreement_weight=args.w_kl_disagreement,
        argmax_mismatch_lost_weight=args.w_argmax_mismatch_lost,
        include_forced=bool(args.include_forced),
    )

    shards = discover_shards([Path(r) for r in args.shard_root])
    if args.max_shards > 0:
        shards = shards[: args.max_shards]

    valuer: StoredFeatureValuer | None = None
    if args.value_checkpoint:
        valuer = StoredFeatureValuer(
            args.value_checkpoint,
            device=args.device,
            value_scale=args.value_scale,
            value_squash=args.value_squash,
            batch_size=args.value_batch_size,
        )

    topk = _TopK(args.top_k)
    shard_paths: list[str] = []
    hist = np.zeros(_HIST_BINS, dtype=np.int64)
    edges = np.linspace(_HIST_LO, _HIST_HI, _HIST_BINS + 1)

    total_rows = 0
    candidate_rows = 0
    forced_rows = 0
    raw_shards = 0
    searched_shards = 0
    value_surprise_rows = 0
    phase_counts: dict[str, int] = {}
    phase_counts_candidates: dict[str, int] = {}
    sum_score = 0.0
    sum_sq = 0.0
    started = time.perf_counter()

    for shard_index, path in enumerate(shards):
        if args.sample_frac < 1.0:
            h = (hash((str(path), int(args.sample_seed))) & 0xFFFFFFFF) / 0xFFFFFFFF
            if h >= args.sample_frac:
                continue
        shard = load_shard(path)
        teacher = _teacher_of(shard)
        if args.teacher_filter and teacher != args.teacher_filter:
            continue
        shard_paths.append(str(path))
        this_shard_id = len(shard_paths) - 1

        is_raw = _is_raw_shard(shard)
        values = None
        if is_raw:
            raw_shards += 1
            if valuer is not None:
                values = valuer.values(shard)
        else:
            searched_shards += 1

        scored = score_shard(shard, config, values=values)
        n = int(scored["regret_score"].shape[0])
        total_rows += n
        forced_rows += int(scored["is_forced"].sum())
        value_surprise_rows += int(scored["has_value_surprise"].sum())

        phases = np.asarray(shard.get("phase", np.full(n, ""))).astype(str)
        uniq, counts = np.unique(phases, return_counts=True)
        for ph, c in zip(uniq, counts):
            phase_counts[str(ph)] = phase_counts.get(str(ph), 0) + int(c)

        cand = scored["is_candidate"]
        candidate_rows += int(cand.sum())
        cand_idx = np.nonzero(cand)[0]
        if cand_idx.size == 0:
            continue

        scores_c = scored["regret_score"][cand_idx]
        hist += np.histogram(scores_c, bins=edges)[0]
        sum_score += float(scores_c.sum())
        sum_sq += float(np.square(scores_c.astype(np.float64)).sum())
        for ph, c in zip(*np.unique(phases[cand_idx], return_counts=True)):
            phase_counts_candidates[str(ph)] = phase_counts_candidates.get(str(ph), 0) + int(c)

        seeds = np.asarray(shard["game_seed"]).reshape(-1)
        didx = np.asarray(shard["decision_index"]).reshape(-1)
        legal_count = scored["legal_count"]
        # Only offer rows that could plausibly enter the heap (cheap gate).
        floor = topk._heap[0][0] if len(topk._heap) >= topk.k else -1.0
        for i in cand_idx:
            score = float(scored["regret_score"][i])
            if score <= floor:
                continue
            topk.offer(
                score,
                {
                    "shard_id": this_shard_id,
                    "row_index": int(i),
                    "game_seed": int(seeds[i]),
                    "decision_index": int(didx[i]),
                    "phase": str(phases[i]),
                    "teacher": teacher,
                    "regret_score": score,
                    "value_surprise": float(scored["value_surprise"][i]),
                    "phase_bonus": float(scored["phase_bonus"][i]),
                    "legal_count_bonus": float(scored["legal_count_bonus"][i]),
                    "kl_disagreement": float(scored["kl_disagreement"][i]),
                    "argmax_mismatch_lost": float(scored["argmax_mismatch_lost"][i]),
                    "legal_count": int(legal_count[i]),
                    "z": float(scored["z"][i]),
                    "has_value_surprise": bool(scored["has_value_surprise"][i]),
                },
            )

    records = topk.sorted_desc()
    _write_manifest(Path(args.out), records, shard_paths, config, args)
    summary = _build_summary(
        records=records,
        shard_paths=shard_paths,
        hist=hist,
        edges=edges,
        total_rows=total_rows,
        candidate_rows=candidate_rows,
        forced_rows=forced_rows,
        raw_shards=raw_shards,
        searched_shards=searched_shards,
        value_surprise_rows=value_surprise_rows,
        phase_counts=phase_counts,
        phase_counts_candidates=phase_counts_candidates,
        sum_score=sum_score,
        sum_sq=sum_sq,
        elapsed=time.perf_counter() - started,
        config=config,
        args=args,
    )
    write_json_atomic(Path(args.out).with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


def _write_manifest(
    out: Path,
    records: list[dict[str, Any]],
    shard_paths: list[str],
    config: RegretConfig,
    args: argparse.Namespace,
) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        np.savez(out, shard_paths=np.asarray(shard_paths))
        return
    cols = {
        "shard_id": np.asarray([r["shard_id"] for r in records], dtype=np.int32),
        "row_index": np.asarray([r["row_index"] for r in records], dtype=np.int32),
        "game_seed": np.asarray([r["game_seed"] for r in records], dtype=np.int64),
        "decision_index": np.asarray([r["decision_index"] for r in records], dtype=np.int32),
        "regret_score": np.asarray([r["regret_score"] for r in records], dtype=np.float32),
        "value_surprise": np.asarray([r["value_surprise"] for r in records], dtype=np.float32),
        "phase_bonus": np.asarray([r["phase_bonus"] for r in records], dtype=np.float32),
        "legal_count_bonus": np.asarray([r["legal_count_bonus"] for r in records], dtype=np.float32),
        "kl_disagreement": np.asarray([r["kl_disagreement"] for r in records], dtype=np.float32),
        "argmax_mismatch_lost": np.asarray([r["argmax_mismatch_lost"] for r in records], dtype=np.float32),
        "legal_count": np.asarray([r["legal_count"] for r in records], dtype=np.int32),
        "z": np.asarray([r["z"] for r in records], dtype=np.float32),
        "phase": np.asarray([r["phase"] for r in records]),
        "teacher": np.asarray([r["teacher"] for r in records]),
        "shard_paths": np.asarray(shard_paths),
    }
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **cols)
    tmp.replace(out)


def _pct(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    total = int(hist.sum())
    if total == 0:
        return 0.0
    target = q * total
    cum = 0
    for i, count in enumerate(hist):
        cum += int(count)
        if cum >= target:
            return float(0.5 * (edges[i] + edges[i + 1]))
    return float(edges[-1])


def _top_n_dict(counts: dict[str, int], n: int) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n])


def _is_opening_placement(phase: str) -> bool:
    up = str(phase).upper()
    return "BUILD_INITIAL_SETTLEMENT" in up or "BUILD_INITIAL_ROAD" in up


def _build_summary(**kw: Any) -> dict[str, Any]:
    records = kw["records"]
    hist = kw["hist"]
    edges = kw["edges"]
    candidate_rows = kw["candidate_rows"]
    top_phase_counts: dict[str, int] = {}
    opening_top = 0
    legal_counts_top: list[int] = []
    for r in records:
        top_phase_counts[r["phase"]] = top_phase_counts.get(r["phase"], 0) + 1
        if _is_opening_placement(r["phase"]):
            opening_top += 1
        legal_counts_top.append(int(r["legal_count"]))
    mean = kw["sum_score"] / max(candidate_rows, 1)
    var = max(kw["sum_sq"] / max(candidate_rows, 1) - mean * mean, 0.0)
    top_scores = [r["regret_score"] for r in records]
    return {
        "config_weights": {
            "value_surprise": kw["config"].value_surprise_weight,
            "phase_bonus": kw["config"].phase_bonus_weight,
            "legal_count": kw["config"].legal_count_weight,
            "kl_disagreement": kw["config"].kl_disagreement_weight,
            "argmax_mismatch_lost": kw["config"].argmax_mismatch_lost_weight,
            "include_forced": kw["config"].include_forced,
        },
        "shard_roots": list(kw["args"].shard_root),
        "value_checkpoint": kw["args"].value_checkpoint,
        "sample_frac": kw["args"].sample_frac,
        "shards_scored": len(kw["shard_paths"]),
        "raw_shards": kw["raw_shards"],
        "searched_shards": kw["searched_shards"],
        "total_rows": kw["total_rows"],
        "forced_rows": kw["forced_rows"],
        "candidate_rows": candidate_rows,
        "value_surprise_rows": kw["value_surprise_rows"],
        "elapsed_sec": kw["elapsed"],
        "rows_per_sec": kw["total_rows"] / max(kw["elapsed"], 1e-9),
        "score_distribution": {
            "mean": mean,
            "std": float(var**0.5),
            "p50": _pct(hist, edges, 0.50),
            "p90": _pct(hist, edges, 0.90),
            "p99": _pct(hist, edges, 0.99),
            "histogram_counts": hist.tolist(),
            "histogram_edges": edges.tolist(),
        },
        "phase_mix_all": _top_n_dict(kw["phase_counts"], 20),
        "phase_mix_candidates": _top_n_dict(kw["phase_counts_candidates"], 20),
        "top_k": {
            "kept": len(records),
            "min_score": float(min(top_scores)) if top_scores else 0.0,
            "max_score": float(max(top_scores)) if top_scores else 0.0,
            "phase_mix": _top_n_dict(top_phase_counts, 20),
            "opening_placement_count": opening_top,
            "opening_placement_fraction": opening_top / max(len(records), 1),
            "legal_count_mean": float(np.mean(legal_counts_top)) if legal_counts_top else 0.0,
            "legal_count_median": float(np.median(legal_counts_top)) if legal_counts_top else 0.0,
        },
        "top_100_preview": [
            {
                "game_seed": r["game_seed"],
                "decision_index": r["decision_index"],
                "phase": r["phase"],
                "regret_score": round(r["regret_score"], 4),
                "value_surprise": round(r["value_surprise"], 4),
                "kl_disagreement": round(r["kl_disagreement"], 4),
                "legal_count": r["legal_count"],
                "z": r["z"],
            }
            for r in records[:100]
        ],
    }


if __name__ == "__main__":
    main()
