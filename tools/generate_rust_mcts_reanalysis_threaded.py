#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

from catan_zero.search import RustMCTS, RustMCTSConfig
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from convert_teacher_to_entity_tokens import EntityShardWriter
from generate_rust_mcts_reanalysis import (
    COLORS,
    _apply_action_with_sampled_chance,
    _csv_patterns,
    _evaluator_feature_contract,
    _feature_contract_manifest_fields,
    _game_outcome_fields,
    _mcts_row,
    _phase_selected,
    _raw_policy_action,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threaded Rust MCTS reanalysis generator with batched neural inference."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opponent", default="value_function")
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--search-temperature", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-wait-ms", type=float, default=3.0)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--min-legal-actions", type=int, default=2)
    parser.add_argument("--record-after-decisions", type=int, default=0)
    parser.add_argument("--phase-include", default="")
    parser.add_argument("--phase-exclude", default="")
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--format", choices=("npz", "npz_zst"), default="npz_zst")
    parser.add_argument("--obs-width", type=int, default=806)
    parser.add_argument("--progress-seconds", type=float, default=20.0)
    args = parser.parse_args()

    try:
        import catanatron_rs  # type: ignore
    except ImportError as error:
        raise SystemExit("catanatron_rs is not installed in this Python environment") from error

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("entity_teacher_shard_*.npz*")):
        raise SystemExit(f"{output} already contains entity shards; use a fresh output directory")

    writer = EntityShardWriter(output, shard_size=int(args.shard_size), fmt=args.format)
    lock = threading.Lock()
    counters: dict[str, int | float] = {
        "next_game": 0,
        "reserved_rows": 0,
        "written_rows": 0,
        "games_started": 0,
        "games_completed": 0,
        "candidate_wins": 0,
        "skipped_forced": 0,
        "skipped_window": 0,
        "skipped_phase": 0,
        "terminal_rows": 0,
        "truncated_rows": 0,
        "errors": 0,
    }
    started = time.perf_counter()
    stop = threading.Event()
    phase_include = _csv_patterns(args.phase_include)
    phase_exclude = _csv_patterns(args.phase_exclude)

    evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
        args.checkpoint,
        device=args.device,
        config=EntityGraphRustEvaluatorConfig(
            value_scale=float(args.value_scale),
            prior_temperature=float(args.prior_temperature),
        ),
        max_batch_size=int(args.batch_size),
        max_wait_ms=float(args.batch_wait_ms),
    )
    feature_contract = _evaluator_feature_contract(evaluator)

    def reserve_game() -> int | None:
        with lock:
            if int(counters["next_game"]) >= int(args.games):
                return None
            if int(counters["reserved_rows"]) >= int(args.samples):
                return None
            game_index = int(counters["next_game"])
            counters["next_game"] = game_index + 1
            counters["games_started"] = int(counters["games_started"]) + 1
            return game_index

    def reserve_row() -> bool:
        with lock:
            if int(counters["reserved_rows"]) >= int(args.samples):
                return False
            counters["reserved_rows"] = int(counters["reserved_rows"]) + 1
            return True

    def add_counter(key: str, value: int) -> None:
        with lock:
            counters[key] = int(counters[key]) + int(value)

    def write_pending(game: Any, pending: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
        terminal = game.winning_color() is not None
        outcome = _game_outcome_fields(game, terminal=terminal)
        with lock:
            for row, features in pending:
                row.update(outcome)
                writer.add(row, features)
            counters["written_rows"] = int(counters["written_rows"]) + len(pending)
            if terminal:
                counters["terminal_rows"] = int(counters["terminal_rows"]) + len(pending)
            else:
                counters["truncated_rows"] = int(counters["truncated_rows"]) + len(pending)

    def worker(worker_id: int) -> None:
        search = RustMCTS(
            RustMCTSConfig(
                colors=COLORS,
                simulations=max(1, int(args.simulations)),
                c_puct=float(args.c_puct),
                max_depth=int(args.max_depth),
                seed=int(args.seed) + worker_id * 100_003,
                temperature=float(args.search_temperature),
            ),
            evaluator=evaluator,
        )
        rng = random.Random(int(args.seed) ^ (worker_id * 0x9E3779B1))
        while not stop.is_set():
            game_index = reserve_game()
            if game_index is None:
                return
            candidate_color = COLORS[game_index % 2]
            player_kinds = [
                "random" if color == candidate_color else str(args.opponent)
                for color in COLORS
            ]
            game = catanatron_rs.Game(
                colors=list(COLORS),
                seed=int(args.seed) + game_index,
                player_kinds=player_kinds,
                vps_to_win=10,
            )
            decisions = 0
            pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
            try:
                while game.winning_color() is None and decisions < int(args.max_decisions):
                    current = str(game.current_color())
                    if current == candidate_color:
                        legal_rust = tuple(
                            int(action)
                            for action in game.playable_action_indices(list(COLORS), None)
                        )
                        if not legal_rust:
                            break
                        phase = str(json.loads(game.json_snapshot()).get("current_prompt", ""))
                        should_record = True
                        if decisions < int(args.record_after_decisions):
                            add_counter("skipped_window", 1)
                            should_record = False
                        elif len(legal_rust) < int(args.min_legal_actions):
                            add_counter("skipped_forced", 1)
                            should_record = False
                        elif not _phase_selected(phase, include=phase_include, exclude=phase_exclude):
                            add_counter("skipped_phase", 1)
                            should_record = False
                        elif not reserve_row():
                            should_record = False

                        if should_record:
                            row = _mcts_row(
                                game,
                                search=search,
                                evaluator=evaluator,
                                legal_rust=legal_rust,
                                candidate_color=candidate_color,
                                game_seed=int(args.seed) + game_index,
                                decision_index=decisions,
                                obs_width=int(args.obs_width),
                            )
                            features = row.pop("_features")
                            action = int(row.pop("_rust_action"))
                            pending.append((row, features))
                        else:
                            action = _raw_policy_action(evaluator, game, legal_rust)
                        game = _apply_action_with_sampled_chance(
                            game,
                            int(action),
                            colors=COLORS,
                            rng=rng,
                        )
                    else:
                        game.play_tick()
                    decisions += 1
                if game.winning_color() is not None and str(game.winning_color()) == candidate_color:
                    add_counter("candidate_wins", 1)
                write_pending(game, pending)
                add_counter("games_completed", 1)
            except BaseException as error:
                add_counter("errors", 1)
                print(
                    json.dumps(
                        {
                            "error": "threaded_rust_mcts_worker",
                            "worker": worker_id,
                            "game_index": game_index,
                            "message": repr(error),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                stop.set()
                return

    def progress() -> None:
        while not stop.wait(float(args.progress_seconds)):
            with lock:
                payload = dict(counters)
            payload.update(
                {
                    "progress": "threaded_rust_mcts_reanalysis",
                    "elapsed_sec": time.perf_counter() - started,
                    "rows_per_sec": float(payload["written_rows"]) / max(time.perf_counter() - started, 1.0e-9),
                    "adapter_version": feature_contract[
                        "entity_feature_adapter_version"
                    ],
                    "threads": int(args.threads),
                    "batch_size": int(args.batch_size),
                }
            )
            print(json.dumps(payload, sort_keys=True), flush=True)

    progress_thread = threading.Thread(target=progress, name="mcts-progress", daemon=True)
    progress_thread.start()
    threads = [
        threading.Thread(target=worker, args=(idx,), name=f"mcts-worker-{idx}", daemon=True)
        for idx in range(max(1, int(args.threads)))
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        stop.set()
        evaluator.close()
        with lock:
            writer.close()

    summary = dict(counters)
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "out": str(output),
            "opponent": args.opponent,
            "games_requested": int(args.games),
            "simulations": int(args.simulations),
            "c_puct": float(args.c_puct),
            "search_temperature": float(args.search_temperature),
            "max_depth": int(args.max_depth),
            "prior_temperature": float(args.prior_temperature),
            "value_scale": float(args.value_scale),
            **_feature_contract_manifest_fields(feature_contract),
            "seed": int(args.seed),
            "threads": int(args.threads),
            "batch_size": int(args.batch_size),
            "batch_wait_ms": float(args.batch_wait_ms),
            "elapsed_sec": time.perf_counter() - started,
            "rows_per_sec": int(counters["written_rows"]) / max(time.perf_counter() - started, 1.0e-9),
            "shards": [str(path) for path in writer.paths],
        }
    )
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
