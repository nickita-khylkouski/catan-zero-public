#!/usr/bin/env python3
"""CLI: generate raw-policy self-play training shards (value-head repair v2, task #65).

Plays full 2p self-play games with `catan_zero.rl.raw_selfplay.play_one_raw_selfplay_game`
(both seats select actions from the checkpoint's RAW policy priors -- no search,
no `GumbelChanceMCTS` anywhere), writing entity-token shards compatible with
`tools/train_bc.py`'s loader. Every row is written with
`policy_weight_multiplier=0.0` (do not imitate raw argmax) and
`value_weight_multiplier=1.0` (every row is a real, true-outcome value sample).
See `src/catan_zero/rl/raw_selfplay.py` for the driver/schema rationale.

Throughput target: ~100x the searched generator (no MCTS tree per decision),
expected on the order of 2000+ games/hr/GPU.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

from catan_zero.rl.raw_selfplay import (
    COLORS,
    RawSelfPlayConfig,
    run_raw_selfplay_worker_games,
)
from catan_zero.rl.gumbel_self_play import (
    TARGET_INFORMATION_REGIME_AUTHORITATIVE,
    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
)
from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
)
from catan_zero.rl.meaningful_history import MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
from catan_zero.search.gumbel_chance_mcts import HeuristicRustEvaluator
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from factory_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate entity-token raw-policy self-play shards (no search)."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="entity_graph policy checkpoint; omit to use HeuristicRustEvaluator.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument(
        "--temperature-decisions",
        type=int,
        default=45,
        help="Sample from (temperature-reweighted) raw priors for the first N "
        "decisions of each game (trajectory diversity); argmax thereafter. "
        "Absolute decision count, NOT a fraction of --max-decisions.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--value-squash", choices=("tanh", "clip"), default="tanh")
    parser.add_argument(
        "--correct-rust-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mitigate verified Rust engine chance-spectrum bugs (A19/A20) in the "
        "live-game chance resolution. Set --no-correct-rust-chance-spectra to "
        "trust the engine's native spectrum_json directly (A/B against a fixed wheel).",
    )
    parser.add_argument(
        "--public-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same public observation available to the learner for behavior "
        "selection. Persisted rows are always public-masked regardless.",
    )
    parser.add_argument(
        "--meaningful-public-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--event-history-limit", type=int, default=64)
    parser.add_argument(
        "--learner-entity-feature-adapter-version",
        default=RUST_ENTITY_ADAPTER_V5,
    )
    parser.add_argument(
        "--belief-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Accepted for launcher symmetry with the search-based generators, but "
        "INERT here: raw self-play has no MCTS planner, so there is no internal chance "
        "spectrum to de-leak (the live game already samples chance from true state, "
        "which is correct). Setting it True logs a warning and has no effect.",
    )
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--obs-width", type=int, default=806)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--format", choices=("npz", "npz_zst"), default="npz")
    parser.add_argument(
        "--score-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HeuristicRustEvaluator only: score actions via chance-weighted lookahead "
        "(slower, meaningful priors) vs uniform priors (fast, for smoke tests).",
    )
    args = parser.parse_args()

    if bool(getattr(args, "belief_chance_spectra", False)):
        print(
            "WARNING: --belief-chance-spectra has no effect in raw self-play (no MCTS "
            "planner); the live game samples chance from true state. Flag accepted for "
            "launcher symmetry only.",
            flush=True,
        )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("worker_*")) or (output / "manifest.json").exists():
        raise SystemExit(f"{output} already contains self-play output; use a fresh --out-dir")

    workers = max(1, int(args.workers))
    games = max(0, int(args.games))
    games_per_worker = [games // workers + (1 if i < games % workers else 0) for i in range(workers)]

    worker_args = []
    game_index_start = 0
    for worker_index, worker_games in enumerate(games_per_worker):
        if worker_games <= 0:
            continue
        worker_args.append(
            {
                "worker_index": worker_index,
                "games": worker_games,
                "game_index_start": game_index_start,
                "out_dir": str(output / f"worker_{worker_index:03d}"),
                "checkpoint": args.checkpoint,
                "device": args.device,
                "max_decisions": int(args.max_decisions),
                "temperature_decisions": int(args.temperature_decisions),
                "temperature": float(args.temperature),
                "prior_temperature": float(args.prior_temperature),
                "value_scale": float(args.value_scale),
                "value_squash": str(args.value_squash),
                "track": args.track,
                "vps_to_win": int(args.vps_to_win),
                "obs_width": int(args.obs_width),
                "base_seed": int(args.base_seed),
                "worker_seed": int(args.base_seed) + 0x9E3779B9 * (worker_index + 1),
                "shard_size": int(args.shard_size),
                "format": args.format,
                "score_actions": bool(args.score_actions),
                "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
                "public_observation": bool(args.public_observation),
                "meaningful_public_history": bool(args.meaningful_public_history),
                "meaningful_public_history_schema": (
                    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
                    if str(args.learner_entity_feature_adapter_version)
                    in {RUST_ENTITY_ADAPTER_V5, RUST_ENTITY_ADAPTER_V6}
                    else RawSelfPlayConfig().meaningful_public_history_schema
                ),
                "event_history_limit": int(args.event_history_limit),
                "learner_entity_feature_adapter_version": str(
                    args.learner_entity_feature_adapter_version
                ),
            }
        )
        game_index_start += worker_games

    started = time.perf_counter()
    if len(worker_args) <= 1:
        results = [_worker_entry(worker_args[0])] if worker_args else []
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_entry, worker_args)

    summary = _merge_worker_summaries(
        results,
        out_dir=output,
        elapsed_sec=time.perf_counter() - started,
        args=args,
    )
    write_json(output / "manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _worker_entry(worker_args: dict[str, Any]) -> dict[str, Any]:
    """Top-level, picklable per-worker entry point (safe for multiprocessing spawn).

    Must NEVER raise -- see `generate_gumbel_selfplay_data.py`'s identical
    docstring rationale: a raised exception here aborts `pool.map` for every
    OTHER worker too, discarding their already-written shards before the
    top-level manifest is ever written.
    """
    worker_index = int(worker_args.get("worker_index", -1))
    try:
        return _run_worker(worker_args)
    except Exception as error:  # noqa: BLE001 - isolate one worker from the whole batch.
        return {
            "worker_index": worker_index,
            "out_dir": str(worker_args.get("out_dir", "")),
            "games_requested": int(worker_args.get("games", 0)),
            "games_completed": 0,
            "games_failed": int(worker_args.get("games", 0)),
            "games_truncated": 0,
            "wins_by_color": {},
            "rows": 0,
            "decisions_total": 0,
            "forced_decisions_total": 0,
            "elapsed_sec": 0.0,
            "rows_per_sec": 0.0,
            "shards": [],
            "errors": [
                {
                    "worker_index": worker_index,
                    "game_index": None,
                    "game_seed": None,
                    "error": f"worker-level failure before any game ran: {error!r}",
                }
            ],
        }


def _run_worker(worker_args: dict[str, Any]) -> dict[str, Any]:
    checkpoint = worker_args["checkpoint"]
    colors = COLORS
    if checkpoint:
        evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
            checkpoint,
            device=worker_args["device"],
            config=EntityGraphRustEvaluatorConfig(
                value_scale=float(worker_args["value_scale"]),
                prior_temperature=float(worker_args["prior_temperature"]),
                value_squash=str(worker_args.get("value_squash", "tanh")),
                public_observation=bool(worker_args.get("public_observation", False)),
            ),
        )
    else:
        evaluator = HeuristicRustEvaluator(score_actions=bool(worker_args["score_actions"]))

    config = RawSelfPlayConfig(
        colors=colors,
        track=str(worker_args["track"]),
        vps_to_win=int(worker_args["vps_to_win"]),
        obs_width=int(worker_args["obs_width"]),
        max_decisions=int(worker_args["max_decisions"]),
        temperature_decisions=int(worker_args["temperature_decisions"]),
        temperature=float(worker_args["temperature"]),
        correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
        meaningful_public_history=bool(
            worker_args.get("meaningful_public_history", False)
        ),
        meaningful_public_history_schema=str(
            worker_args.get(
                "meaningful_public_history_schema",
                RawSelfPlayConfig().meaningful_public_history_schema,
            )
        ),
        event_history_limit=int(worker_args.get("event_history_limit", 64)),
        entity_feature_adapter_version=str(
            worker_args.get(
                "learner_entity_feature_adapter_version",
                RawSelfPlayConfig().entity_feature_adapter_version,
            )
        ),
        target_information_regime=(
            TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
            if checkpoint and bool(worker_args.get("public_observation", False))
            else TARGET_INFORMATION_REGIME_AUTHORITATIVE
        ),
    )
    summary = run_raw_selfplay_worker_games(
        out_dir=Path(worker_args["out_dir"]),
        games=int(worker_args["games"]),
        game_index_start=int(worker_args["game_index_start"]),
        base_seed=int(worker_args["base_seed"]),
        worker_seed=int(worker_args["worker_seed"]),
        config=config,
        evaluator=evaluator,
        shard_size=int(worker_args["shard_size"]),
        fmt=str(worker_args["format"]),
    )
    summary["worker_index"] = int(worker_args["worker_index"])
    return summary


def _merge_worker_summaries(
    results: list[dict[str, Any]],
    *,
    out_dir: Path,
    elapsed_sec: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    shards: list[str] = []
    games_completed = 0
    games_failed = 0
    games_truncated = 0
    rows = 0
    decisions_total = 0
    forced_decisions_total = 0
    wins_by_color: dict[str, int] = {color: 0 for color in COLORS}
    errors: list[dict[str, Any]] = []
    worker_summaries: list[str] = []
    for result in sorted(results, key=lambda item: int(item.get("worker_index", 0))):
        # Defensive: only list shards that actually exist on disk (mirrors
        # generate_gumbel_selfplay_data.py's identical guard).
        shards.extend(path for path in result.get("shards", ()) if Path(path).exists())
        games_completed += int(result.get("games_completed", 0))
        games_failed += int(result.get("games_failed", 0))
        games_truncated += int(result.get("games_truncated", 0))
        rows += int(result.get("rows", 0))
        decisions_total += int(result.get("decisions_total", 0))
        forced_decisions_total += int(result.get("forced_decisions_total", 0))
        for color, count in dict(result.get("wins_by_color", {})).items():
            wins_by_color[color] = wins_by_color.get(color, 0) + int(count)
        for error in result.get("errors", ()):
            error = dict(error)
            error["worker_index"] = int(result.get("worker_index", -1))
            errors.append(error)
        out_dir_value = result.get("out_dir")
        if out_dir_value:
            manifest_candidate = Path(out_dir_value) / "manifest.json"
            if manifest_candidate.exists():
                worker_summaries.append(str(manifest_candidate))

    return {
        "out_dir": str(out_dir),
        "track": args.track,
        "vps_to_win": int(args.vps_to_win),
        "colors": list(COLORS),
        "games_requested": int(args.games),
        "games_completed": int(games_completed),
        "games_failed": int(games_failed),
        "games_truncated": int(games_truncated),
        "wins_by_color": wins_by_color,
        "rows": int(rows),
        "decisions_total": int(decisions_total),
        "forced_decisions_total": int(forced_decisions_total),
        "workers": len(results),
        "checkpoint": args.checkpoint,
        "base_seed": int(args.base_seed),
        "temperature_decisions": int(args.temperature_decisions),
        "temperature": float(args.temperature),
        # Complete CLI-argument provenance so a shard batch is auditable after
        # the process exits (982d344 pattern).
        "cli_args": {key: value for key, value in vars(args).items()},
        "elapsed_sec": elapsed_sec,
        "rows_per_sec": rows / max(elapsed_sec, 1.0e-9),
        "shards": shards,
        "worker_summaries": worker_summaries,
        "errors": errors,
    }


if __name__ == "__main__":
    main()
