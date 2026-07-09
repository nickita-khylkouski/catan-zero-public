from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from catan_zero.rl import evaluate_policy
from catan_zero.rl.policy_pool import assert_policy_compatible_with_env
from factory_common import confidence_interval, make_named_policy, parse_track, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate checkpoints against baselines.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-kind", default="checkpoint")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--tracks", default="2p_no_trade")
    parser.add_argument(
        "--opponents",
        default=(
            "random,heuristic,value,jsettlers_lite,catanatron_ab3,"
            "catanatron_ab4,catanatron_ab5,catanatron_search"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel CPU worker processes. Use 0 to auto-select a sensible "
            "count from available CPUs and matchup/game count."
        ),
    )
    parser.add_argument(
        "--chunk-games",
        type=int,
        default=0,
        help="Games per worker chunk. Defaults to games/workers so one matchup uses all workers.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--paired-seeds",
        action="store_true",
        help=(
            "Use the same seed stream for every opponent within a track. This "
            "makes AB3/AB4/value comparisons much less noisy because each "
            "opponent sees the same board/game schedule."
        ),
    )
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device for checkpoint policies. Defaults to cpu so multi-worker "
            "scoreboards cannot silently steal training GPUs. Pass cuda:0 or "
            "auto explicitly only when the box is dedicated to evaluation."
        ),
    )
    parser.add_argument(
        "--allow-gpu-workers",
        action="store_true",
        help=(
            "Allow --workers > 1 with a CUDA/auto device. Without this guard, "
            "multi-process scoreboards are CPU-only unless explicitly approved."
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    matchups = [
        (track.strip(), _parse_opponent_spec(opponent.strip()))
        for track in args.tracks.split(",")
        for opponent in args.opponents.split(",")
        if track.strip() and opponent.strip()
    ]
    args.workers = _resolve_workers(args.workers, games=args.games, matchups=len(matchups))
    _validate_device_args(args)
    jobs = [
        payload
        for matchup in matchups
        for payload in _chunked_job_payloads(args, matchup)
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_job, job) for job in jobs]
            chunk_results = []
            for future in as_completed(futures):
                result = future.result()
                chunk_results.append(result)
                print(
                    {
                        "progress": "scoreboard_chunk",
                        "track": result["track"],
                        "opponent": result["opponent"],
                        "chunk_index": result["chunk_index"],
                        "chunk_count": result["chunk_count"],
                        "games": result["games"],
                        "wins": result["wins"],
                    },
                    flush=True,
                )
    else:
        chunk_results = [_run_job(job) for job in jobs]
    results = _aggregate_results(chunk_results)
    report = {
        "candidate": args.candidate,
        "candidate_kind": args.candidate_kind,
        "games_per_matchup": args.games,
        "chunk_games": _chunk_games(args),
        "chunks": len(jobs),
        "elapsed_sec": time.perf_counter() - start,
        "device": args.device,
        "workers": args.workers,
        # FIX A8: base seed for this whole report. Two reports with equal
        # "seed", equal "paired_seeds" (True), and equal per-opponent
        # "leg_seed" faced identical per-game-index board/dice schedules, so
        # compare_scoreboards.py can pair their game_outcomes for McNemar.
        "seed": args.seed,
        "paired_seeds": bool(args.paired_seeds),
        "results": results,
    }
    write_json(args.out, report)
    print(report)


def _validate_device_args(args: argparse.Namespace) -> None:
    device = str(args.device).strip().lower()
    if int(args.workers) <= 1:
        return
    if device == "cpu":
        return
    if args.allow_gpu_workers:
        return
    raise SystemExit(
        "Refusing to run a multi-worker scoreboard on a GPU/auto device. "
        "Use --device cpu, lower --workers to 1, or pass --allow-gpu-workers "
        "when the GPUs are intentionally dedicated to evaluation."
    )


def _parse_opponent_spec(spec: str) -> dict:
    if spec.startswith("checkpoint:"):
        checkpoint = spec[len("checkpoint:") :].strip()
        if not checkpoint:
            raise SystemExit("--opponents checkpoint: requires a checkpoint path")
        return {
            "opponent": f"checkpoint:{checkpoint}",
            "opponent_kind": "checkpoint",
            "opponent_checkpoint": checkpoint,
            "opponent_label": f"checkpoint:{checkpoint}",
        }
    return {
        "opponent": spec,
        "opponent_kind": spec,
        "opponent_checkpoint": None,
        "opponent_label": spec,
    }


def _chunked_job_payloads(args: argparse.Namespace, job: tuple[str, dict]) -> list[dict]:
    track, opponent_spec = job
    chunk_games = _chunk_games(args)
    chunk_count = max(1, math.ceil(args.games / chunk_games))
    leg_seed = args.seed + _seed_offset(
        track, opponent_spec["opponent_label"], paired=bool(args.paired_seeds)
    )
    payloads = []
    for chunk_index in range(chunk_count):
        start_game = chunk_index * chunk_games
        games = min(chunk_games, args.games - start_game)
        if games <= 0:
            continue
        payloads.append(
            {
                "candidate": args.candidate,
                "candidate_kind": args.candidate_kind,
                "games": games,
                "track": track,
                **opponent_spec,
                "seed": leg_seed + chunk_index * 100_000_003,
                # Constant across chunks of the same opponent leg; used to
                # verify two reports' game_outcomes are actually comparable
                # game-for-game before compare_scoreboards.py pairs them.
                "leg_seed": leg_seed,
                "vps_to_win": args.vps_to_win,
                "max_decisions": args.max_decisions,
                "device": args.device,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "start_game_index": start_game,
            }
        )
    return payloads


def _resolve_workers(requested: int, *, games: int, matchups: int) -> int:
    if requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    # Do not create more workers than useful game chunks. Leave one core for
    # the OS / logging, but prefer saturating large CPU boxes during scoreboards.
    useful_jobs = max(1, int(games) * max(1, int(matchups)))
    return max(1, min(max(1, cpu_count - 1), useful_jobs))


def _chunk_games(args: argparse.Namespace) -> int:
    if args.chunk_games > 0:
        return max(1, int(args.chunk_games))
    if args.workers > 1:
        return max(1, math.ceil(args.games / args.workers))
    return max(1, int(args.games))


def _run_job(payload: dict) -> dict:
    _cap_torch_threads()
    candidate = (
        make_named_policy(payload["candidate_kind"], device=payload["device"])
        if payload["candidate_kind"] != "checkpoint"
        else make_named_policy("checkpoint", payload["candidate"], device=payload["device"])
    )
    opponent = (
        make_named_policy("checkpoint", payload["opponent_checkpoint"], device=payload["device"])
        if payload.get("opponent_kind") == "checkpoint"
        else make_named_policy(payload["opponent_kind"], device=payload["device"])
    )
    config = parse_track(payload["track"], vps_to_win=int(payload["vps_to_win"]))
    assert_policy_compatible_with_env(candidate, config)
    assert_policy_compatible_with_env(opponent, config)
    # FIX A8: record per-game win/loss in game order so compare_scoreboards.py
    # can build exact per-game-index pairs (McNemar) instead of only aggregate
    # win counts. evaluate_policy calls progress_callback once per completed
    # game, in order, so appending here reproduces that order exactly.
    #
    # FIX (adversarial review, truncation-as-loss bias): a truncated game
    # (episode hit max_decisions with no winner) is a missing data point, not
    # a candidate loss. `event["winner"] == event["candidate_seat"]` would
    # silently evaluate False for it (None != seat name), biasing every SPRT/
    # McNemar pairing against whichever policy truncates more often. Record
    # None for truncated games instead; sprt_gate.py and compare_scoreboards.py
    # exclude any pair containing a None from the paired analysis and report
    # the excluded count separately.
    game_outcomes: list[bool | None] = []
    truncated_games = 0

    def _record_outcome(event: dict[str, Any]) -> None:
        nonlocal truncated_games
        if event.get("truncated"):
            truncated_games += 1
            game_outcomes.append(None)
        else:
            game_outcomes.append(bool(event["winner"] == event["candidate_seat"]))

    result = evaluate_policy(
        candidate,
        opponent,
        games=payload["games"],
        seed=payload["seed"],
        config=config,
        max_decisions=payload["max_decisions"],
        start_game_index=int(payload.get("start_game_index", 0)),
        progress_callback=_record_outcome,
    )
    low, high = confidence_interval(int(result["wins"]), int(result["games"]))
    result.update(
        {
            "track": payload["track"],
            "opponent": payload.get("opponent_label", payload["opponent"]),
            "confidence_interval_95": [low, high],
            "illegal_action_count": int(result.get("illegal_action_count", 0)),
            "timeouts_or_stuck_games": int(result.get("timeouts_or_stuck_games", 0)),
            "chunk_index": payload["chunk_index"],
            "chunk_count": payload["chunk_count"],
            "start_game_index": int(payload.get("start_game_index", 0)),
            "leg_seed": payload.get("leg_seed"),
            "game_outcomes": game_outcomes,
            "truncated_games": truncated_games,
        }
    )
    return result


def _cap_torch_threads() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _aggregate_results(chunk_results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in chunk_results:
        groups[(str(result["track"]), str(result["opponent"]))].append(result)

    results = []
    for (track, opponent), unordered_chunks in sorted(groups.items()):
        # Multi-worker runs complete out of submission order (as_completed),
        # so game_outcomes must be re-sorted by chunk_index before
        # concatenation or the per-game-index pairing used by
        # compare_scoreboards.py would silently scramble game order.
        chunks = sorted(unordered_chunks, key=lambda chunk: int(chunk.get("chunk_index", 0)))
        games = sum(int(chunk["games"]) for chunk in chunks)
        wins = sum(int(chunk["wins"]) for chunk in chunks)
        total_decisions = sum(
            float(chunk.get("avg_decisions", 0.0)) * int(chunk["games"])
            for chunk in chunks
        )
        total_candidate_vp = sum(
            float(chunk.get("avg_candidate_vp", 0.0)) * int(chunk["games"])
            for chunk in chunks
        )
        total_best_opponent_vp = sum(
            float(chunk.get("avg_best_opponent_vp", 0.0)) * int(chunk["games"])
            for chunk in chunks
        )
        total_vp_margin = sum(
            float(chunk.get("avg_vp_margin", 0.0)) * int(chunk["games"])
            for chunk in chunks
        )
        total_candidate_win_decisions = sum(
            float(chunk.get("avg_candidate_win_decisions", 0.0) or 0.0)
            * int(chunk.get("wins", 0))
            for chunk in chunks
        )
        seat_wins: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            for seat, count in dict(chunk.get("seat_wins", {})).items():
                seat_wins[str(seat)] += int(count)
        win_rate = wins / games if games else 0.0
        low, high = confidence_interval(wins, games)
        first = chunks[0]
        leg_seeds = {chunk.get("leg_seed") for chunk in chunks if chunk.get("leg_seed") is not None}
        game_outcomes: list[bool | None] | None = None
        if all("game_outcomes" in chunk for chunk in chunks):
            # NOTE: `outcome` may be None (truncated game) -- do NOT coerce
            # with bool(), that silently turns a truncated game back into a
            # candidate loss (the exact adversarial-review bug this schema
            # exists to avoid).
            game_outcomes = [
                outcome if outcome is None else bool(outcome)
                for chunk in chunks
                for outcome in chunk.get("game_outcomes", [])
            ]
            if len(game_outcomes) != games:
                # Chunk(s) were re-run from a stale timeout report or similar;
                # game_outcomes would no longer line up with `games`, so drop
                # it rather than hand compare_scoreboards.py a silently wrong
                # per-game-index array.
                game_outcomes = None
        truncated_games = sum(int(chunk.get("truncated_games", 0)) for chunk in chunks)
        results.append(
            {
                "games": games,
                "candidate": first.get("candidate"),
                "opponent": opponent,
                "wins": wins,
                "win_rate": win_rate,
                "elo_vs_opponent": first.get("elo_vs_opponent")
                if games == int(first.get("games", 0))
                else _elo_difference(win_rate),
                "seat_wins": dict(sorted(seat_wins.items())),
                "avg_decisions": total_decisions / games if games else 0.0,
                "moves_to_win": total_candidate_win_decisions / wins if wins else None,
                "avg_candidate_win_decisions": total_candidate_win_decisions / wins if wins else None,
                "avg_candidate_vp": total_candidate_vp / games if games else 0.0,
                "avg_best_opponent_vp": total_best_opponent_vp / games if games else 0.0,
                "avg_vp_margin": total_vp_margin / games if games else 0.0,
                "track": track,
                "confidence_interval_95": [low, high],
                "illegal_action_count": sum(
                    int(chunk.get("illegal_action_count", 0)) for chunk in chunks
                ),
                "timeouts_or_stuck_games": sum(
                    int(chunk.get("timeouts_or_stuck_games", 0)) for chunk in chunks
                ),
                "chunks": len(chunks),
                # FIX A8 pairing metadata: leg_seed is only set (single value)
                # when every chunk of this opponent leg agreed on the base
                # seed; game_outcomes is the ordered per-game win/loss array.
                "leg_seed": next(iter(leg_seeds)) if len(leg_seeds) == 1 else None,
                "game_outcomes": game_outcomes,
                "truncated_games": truncated_games,
            }
        )
    return results


def _stable_offset(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 10_000_000


def _seed_offset(track: str, opponent: str, *, paired: bool) -> int:
    if paired:
        return _stable_offset(track)
    return _stable_offset(track, opponent)


def _elo_difference(score: float) -> float:
    clipped = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10((1.0 / clipped) - 1.0)


if __name__ == "__main__":
    main()
