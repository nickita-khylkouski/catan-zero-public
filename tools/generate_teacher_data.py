from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from catan_zero.rl import collect_imitation_game
from factory_common import (
    classical_teacher_hard_action_target_information,
    make_named_policy,
    parse_track,
    write_json,
)

PLAYER_NAMES = ("BLUE", "RED", "ORANGE", "WHITE")
DEFAULT_PRODUCTION_TEACHERS = (
    "catanatron_ab4,catanatron_ab5,value_rollout_search,"
    "catanatron_ab3,catanatron_value,jsettlers_lite"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact synthetic teacher shards.")
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--teachers", default=DEFAULT_PRODUCTION_TEACHERS)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--format", choices=("npz", "npz_zst"), default="npz")
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-games", type=int, default=16)
    parser.add_argument(
        "--graph-history-features",
        action="store_true",
        help="Append the public graph/history feature suffix to observations.",
    )
    parser.add_argument(
        "--teacher-sampling-weights",
        default="",
        help=(
            "Optional comma-separated teacher sampling weights for generation, "
            "e.g. catanatron_ab5=3,value_rollout_search=2,catanatron_value=0.5. "
            "Applies to random mixed-seat assignment and single-policy game assignment."
        ),
    )
    parser.add_argument(
        "--mixed-seats",
        action="store_true",
        help="Use different teachers in different seats within each game.",
    )
    parser.add_argument(
        "--mixed-seat-mode",
        choices=("cycle", "random"),
        default="random",
        help="Seat assignment strategy when --mixed-seats is enabled.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    _prepare_output_dir(output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    teachers = [name.strip() for name in args.teachers.split(",") if name.strip()]
    policies = [make_named_policy(name) for name in teachers]
    teacher_sampling_weights = _teacher_sampling_probabilities(
        teachers,
        args.teacher_sampling_weights,
    )
    config = parse_track(
        args.track,
        vps_to_win=args.vps_to_win,
        use_graph_history_features=args.graph_history_features,
    )
    start = time.perf_counter()
    writer = ShardWriter(output, args.shard_size, args.format)
    decisions = 0
    wins = 0
    workers = max(1, int(args.workers))
    if workers == 1:
        for game_index in range(args.games):
            policy = _teacher_assignment(
                policies,
                game_index=game_index,
                players=config.players,
                mixed_seats=args.mixed_seats,
                mixed_seat_mode=args.mixed_seat_mode,
                teacher_sampling_weights=teacher_sampling_weights,
                seed=args.seed,
            )
            episode = collect_imitation_game(
                policy,
                seed=args.seed + game_index,
                config=config,
                max_decisions=args.max_decisions,
                rng=np.random.default_rng(args.seed + game_index * 9973),
            )
            wins += int(episode.result.winner is not None)
            for player, samples in episode.samples_by_player.items():
                for sample in samples:
                    writer.add(
                        sample,
                        teacher=_sample_teacher_name(sample, policy),
                        game_seed=episode.result.seed,
                        winner=episode.result.winner,
                        terminated=episode.result.terminated,
                        truncated=episode.result.truncated,
                        final_public_vps=episode.result.final_public_vps,
                        final_actual_vps=episode.result.final_actual_vps,
                    )
                    decisions += 1
    else:
        chunk_games = max(1, int(args.chunk_games))
        payloads = [
            {
                "start": start_index,
                "end": min(args.games, start_index + chunk_games),
                "track": args.track,
                "vps_to_win": args.vps_to_win,
                "teachers": teachers,
                "seed": args.seed,
                "max_decisions": args.max_decisions,
                "mixed_seats": args.mixed_seats,
                "mixed_seat_mode": args.mixed_seat_mode,
                "teacher_sampling_weights": args.teacher_sampling_weights,
                "graph_history_features": args.graph_history_features,
            }
            for start_index in range(0, args.games, chunk_games)
        ]
        completed_games = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_chunk, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
                wins += int(result["wins"])
                for row in result["rows"]:
                    writer.add_row(row)
                    decisions += 1
                completed_games += int(result["games"])
                print(
                    json.dumps(
                        {
                            "progress": "teacher_data",
                            "games": completed_games,
                            "samples": decisions,
                            "workers": workers,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    shards = writer.close()
    report = {
        "track": args.track,
        "teachers": teachers,
        "games": args.games,
        "seed": args.seed,
        "vps_to_win": args.vps_to_win,
        "max_decisions": args.max_decisions,
        "workers": workers,
        "chunk_games": args.chunk_games,
        "shard_size": args.shard_size,
        "wins": wins,
        "samples": decisions,
        "format": args.format,
        "mixed_seats": args.mixed_seats,
        "mixed_seat_mode": args.mixed_seat_mode,
        "teacher_sampling_weights": args.teacher_sampling_weights,
        "graph_history_features": args.graph_history_features,
        "hard_action_target_information": (
            classical_teacher_hard_action_target_information()
        ),
        "tool_provenance": _tool_provenance(),
        "shards": [str(path) for path in shards],
        "elapsed_sec": time.perf_counter() - start,
        **writer.summary(),
    }
    write_json(output / "manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


class ShardWriter:
    def __init__(self, output: Path, shard_size: int, fmt: str) -> None:
        self.output = output
        self.shard_size = max(1, int(shard_size))
        self.format = fmt
        existing_shards = list(output.glob("teacher_shard_*.npz")) + list(
            output.glob("teacher_shard_*.npz.zst")
        )
        if existing_shards:
            raise RuntimeError(
                f"{output} already contains teacher shards; use a fresh output "
                "directory so old shards cannot contaminate this run."
            )
        self.rows: list[dict] = []
        self.index = 0
        self.paths: list[Path] = []
        self.teacher_counts: dict[str, int] = {}
        self.phase_counts: dict[str, int] = {}
        self.score_source_counts: dict[str, int] = {}
        self.forced_actions = 0
        self.invalid_teacher_actions = 0
        self.soft_policy_rows = 0
        self.soft_score_rows = 0
        self.outcome_rows = 0
        self.clean_terminal_outcome_rows = 0
        self.final_public_vp_rows = 0
        self.final_actual_vp_rows = 0
        self.truncated_rows = 0
        self.legal_counts: list[int] = []

    def add(
        self,
        sample,
        *,
        teacher: str,
        game_seed: int,
        winner: str | None,
        terminated: bool,
        truncated: bool,
        final_public_vps: dict[str, int],
        final_actual_vps: dict[str, int] | None = None,
    ) -> None:
        valid = np.asarray(sample.valid_actions, dtype=np.int16)
        context = np.asarray(sample.action_context_features, dtype=np.float16)[valid]
        target_policy, target_scores, target_policy_mask, target_scores_mask = _target_arrays(
            valid,
            getattr(sample, "target_policy", None),
            getattr(sample, "target_scores", None),
        )
        self.add_row(
            {
                "obs": np.asarray(sample.observation, dtype=np.float16),
                "valid": valid,
                "context": context,
                "action": np.int16(sample.action),
                "target_policy": target_policy,
                "target_scores": target_scores,
                "target_policy_mask": target_policy_mask,
                "target_scores_mask": target_scores_mask,
                "target_score_source": sample.target_score_source or "",
                "teacher": teacher,
                "seed": np.int64(game_seed),
                "player": sample.player,
                "seat": np.int8(_seat_index(sample.player)),
                "phase": sample.phase or "",
                "decision_index": np.int32(
                    -1 if sample.decision_index is None else sample.decision_index
                ),
                "action_mask_version": sample.action_mask_version or "",
                "winner": winner or "",
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_public_vps": np.asarray(
                    [int(final_public_vps.get(name, 0)) for name in PLAYER_NAMES],
                    dtype=np.int16,
                ),
                "has_final_public_vps": final_public_vps is not None,
                "final_actual_vps": np.asarray(
                    [int((final_actual_vps or {}).get(name, 0)) for name in PLAYER_NAMES],
                    dtype=np.int16,
                ),
                "has_final_actual_vps": final_actual_vps is not None,
            }
        )

    def add_row(self, row: dict) -> None:
        valid_raw = np.asarray(row["valid"], dtype=np.int16)
        action_raw = int(row["action"])
        valid_actions = set(map(int, valid_raw[valid_raw >= 0]))
        if action_raw not in valid_actions:
            raise ValueError(
                f"invalid teacher action {action_raw} for seed={row.get('seed')} "
                f"player={row.get('player')} teacher={row.get('teacher')} "
                f"phase={row.get('phase')} legal_count={len(valid_actions)}"
            )
        record = {
            "obs": np.asarray(row["obs"], dtype=np.float16),
            "valid": valid_raw,
            "context": np.asarray(row["context"], dtype=np.float16),
            "action": np.int16(action_raw),
            "target_policy": np.asarray(
                row.get("target_policy", np.zeros(len(row["valid"]), dtype=np.float16)),
                dtype=np.float16,
            ),
            "target_scores": np.asarray(
                row.get("target_scores", np.full(len(row["valid"]), np.nan, dtype=np.float32)),
                dtype=np.float32,
            ),
            "target_policy_mask": np.asarray(
                row.get("target_policy_mask", np.asarray(row.get("target_policy", ()), dtype=np.float32) > 0.0),
                dtype=np.bool_,
            ),
            "target_scores_mask": np.asarray(
                row.get("target_scores_mask", np.isfinite(np.asarray(row.get("target_scores", ()), dtype=np.float32))),
                dtype=np.bool_,
            ),
            "target_score_source": str(row.get("target_score_source", "")),
            "teacher": str(row["teacher"]),
            "seed": np.int64(row["seed"]),
            "player": str(row.get("player", "")),
            "seat": np.int8(row.get("seat", _seat_index(str(row.get("player", ""))))),
            "phase": str(row.get("phase", "")),
            "decision_index": np.int32(row.get("decision_index", -1)),
            "action_mask_version": str(row.get("action_mask_version", "")),
            "winner": str(row.get("winner", "")),
            "terminated": bool(row.get("terminated", True)),
            "truncated": bool(row.get("truncated", False)),
            "final_public_vps": np.asarray(
                row.get("final_public_vps", np.zeros(len(PLAYER_NAMES), dtype=np.int16)),
                dtype=np.int16,
            ),
            "has_final_public_vps": bool(row.get("has_final_public_vps", False)),
            "final_actual_vps": np.asarray(
                row.get("final_actual_vps", np.zeros(len(PLAYER_NAMES), dtype=np.int16)),
                dtype=np.int16,
            ),
            "has_final_actual_vps": bool(row.get("has_final_actual_vps", False)),
        }
        if record["target_policy_mask"].shape != record["target_policy"].shape:
            record["target_policy_mask"] = (
                np.asarray(record["target_policy"], dtype=np.float32) > 0.0
            )
        if record["target_scores_mask"].shape != record["target_scores"].shape:
            record["target_scores_mask"] = np.isfinite(
                np.asarray(record["target_scores"], dtype=np.float32)
            )
        self.rows.append(record)
        self._update_summary(record)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def close(self) -> list[Path]:
        self.flush()
        return self.paths

    def summary(self) -> dict:
        legal = np.asarray(self.legal_counts, dtype=np.int64) if self.legal_counts else np.asarray([0])
        rows = len(self.legal_counts)
        return {
            "teacher_counts": dict(sorted(self.teacher_counts.items(), key=lambda item: -item[1])),
            "phase_counts": dict(sorted(self.phase_counts.items(), key=lambda item: -item[1])),
            "score_source_counts": dict(sorted(self.score_source_counts.items(), key=lambda item: -item[1])),
            "forced_actions": int(self.forced_actions),
            "forced_action_fraction": self.forced_actions / rows if rows else 0.0,
            "invalid_teacher_actions": int(self.invalid_teacher_actions),
            "soft_policy_rows": int(self.soft_policy_rows),
            "soft_policy_fraction": self.soft_policy_rows / rows if rows else 0.0,
            "soft_score_rows": int(self.soft_score_rows),
            "soft_score_fraction": self.soft_score_rows / rows if rows else 0.0,
            "outcome_rows": int(self.outcome_rows),
            "outcome_fraction": self.outcome_rows / rows if rows else 0.0,
            "clean_terminal_outcome_rows": int(self.clean_terminal_outcome_rows),
            "clean_terminal_outcome_fraction": (
                self.clean_terminal_outcome_rows / rows if rows else 0.0
            ),
            "final_public_vp_rows": int(self.final_public_vp_rows),
            "final_public_vp_fraction": self.final_public_vp_rows / rows if rows else 0.0,
            "final_actual_vp_rows": int(self.final_actual_vp_rows),
            "final_actual_vp_fraction": self.final_actual_vp_rows / rows if rows else 0.0,
            "truncated_rows": int(self.truncated_rows),
            "truncated_fraction": self.truncated_rows / rows if rows else 0.0,
            "legal_actions": {
                "mean": float(np.mean(legal)),
                "p50": int(np.percentile(legal, 50)),
                "p90": int(np.percentile(legal, 90)),
                "p99": int(np.percentile(legal, 99)),
                "max": int(np.max(legal)),
            },
        }

    def _update_summary(self, row: dict) -> None:
        teacher = str(row["teacher"])
        phase = str(row.get("phase", "")) or "unknown"
        score_source = str(row.get("target_score_source", "")) or "none"
        legal = np.asarray(row["valid"], dtype=np.int16)
        legal_count = int(np.sum(legal >= 0))
        action = int(row["action"])
        self.teacher_counts[teacher] = self.teacher_counts.get(teacher, 0) + 1
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        self.score_source_counts[score_source] = self.score_source_counts.get(score_source, 0) + 1
        self.forced_actions += int(legal_count <= 1)
        self.invalid_teacher_actions += int(action not in set(map(int, legal[legal >= 0])))
        self.soft_policy_rows += int(_has_soft_policy(row["target_policy"]))
        self.soft_score_rows += int(_has_soft_scores(row["target_scores"]))
        winner = str(row.get("winner", ""))
        truncated = bool(row.get("truncated", False))
        self.outcome_rows += int(bool(winner))
        self.clean_terminal_outcome_rows += int(bool(winner) and not truncated)
        self.final_public_vp_rows += int(bool(row.get("has_final_public_vps", False)))
        self.final_actual_vp_rows += int(bool(row.get("has_final_actual_vps", False)))
        self.truncated_rows += int(truncated)
        self.legal_counts.append(legal_count)

    def flush(self) -> None:
        if not self.rows:
            return
        max_valid = max(len(row["valid"]) for row in self.rows)
        obs = np.stack([row["obs"] for row in self.rows], axis=0)
        valid = np.full((len(self.rows), max_valid), -1, dtype=np.int16)
        context = np.zeros((len(self.rows), max_valid, self.rows[0]["context"].shape[-1]), dtype=np.float16)
        target_policy = np.zeros((len(self.rows), max_valid), dtype=np.float16)
        target_scores = np.full((len(self.rows), max_valid), np.nan, dtype=np.float32)
        target_policy_mask = np.zeros((len(self.rows), max_valid), dtype=np.bool_)
        target_scores_mask = np.zeros((len(self.rows), max_valid), dtype=np.bool_)
        for idx, row in enumerate(self.rows):
            count = len(row["valid"])
            valid[idx, :count] = row["valid"]
            context[idx, :count, :] = row["context"]
            target_policy[idx, :count] = row["target_policy"]
            target_scores[idx, :count] = row["target_scores"]
            target_policy_mask[idx, :count] = row["target_policy_mask"]
            target_scores_mask[idx, :count] = row["target_scores_mask"]
        actions = np.asarray([row["action"] for row in self.rows], dtype=np.int16)
        seeds = np.asarray([row["seed"] for row in self.rows], dtype=np.int64)
        teachers = np.asarray([row["teacher"] for row in self.rows])
        score_sources = np.asarray([row["target_score_source"] for row in self.rows])
        players = np.asarray([row["player"] for row in self.rows])
        seats = np.asarray([row["seat"] for row in self.rows], dtype=np.int8)
        phases = np.asarray([row["phase"] for row in self.rows])
        decision_indices = np.asarray(
            [row["decision_index"] for row in self.rows],
            dtype=np.int32,
        )
        action_mask_versions = np.asarray([row["action_mask_version"] for row in self.rows])
        winners = np.asarray([row["winner"] for row in self.rows])
        terminated = np.asarray([row["terminated"] for row in self.rows], dtype=np.bool_)
        truncated = np.asarray([row["truncated"] for row in self.rows], dtype=np.bool_)
        final_public_vps = np.stack(
            [row["final_public_vps"] for row in self.rows],
            axis=0,
        ).astype(np.int16, copy=False)
        has_final_public_vps = np.asarray(
            [row["has_final_public_vps"] for row in self.rows],
            dtype=np.bool_,
        )
        final_actual_vps = np.stack(
            [row["final_actual_vps"] for row in self.rows],
            axis=0,
        ).astype(np.int16, copy=False)
        has_final_actual_vps = np.asarray(
            [row["has_final_actual_vps"] for row in self.rows],
            dtype=np.bool_,
        )
        path = self.output / f"teacher_shard_{self.index:05d}.npz"
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                obs=obs,
                legal_action_ids=valid,
                legal_action_context=context,
                action_taken=actions,
                target_policy=target_policy,
                target_scores=target_scores,
                target_policy_mask=target_policy_mask,
                target_scores_mask=target_scores_mask,
                target_score_source=score_sources,
                game_seed=seeds,
                teacher_name=teachers,
                player=players,
                seat=seats,
                phase=phases,
                decision_index=decision_indices,
                action_mask_version=action_mask_versions,
                winner=winners,
                terminated=terminated,
                truncated=truncated,
                final_public_vps=final_public_vps,
                has_final_public_vps=has_final_public_vps,
                final_actual_vps=final_actual_vps,
                has_final_actual_vps=has_final_actual_vps,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        if self.format == "npz_zst":
            path = _try_zstd(path)
        self.paths.append(path)
        self.index += 1
        self.rows = []


def _try_zstd(path: Path) -> Path:
    try:
        import zstandard as zstd
    except ImportError:
        return path
    compressed = path.with_suffix(path.suffix + ".zst")
    tmp_compressed = compressed.with_name(compressed.name + ".tmp")
    compressor = zstd.ZstdCompressor(level=3)
    with tmp_compressed.open("wb") as handle:
        handle.write(compressor.compress(path.read_bytes()))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_compressed, compressed)
    path.unlink()
    return compressed


def _generate_chunk(payload: dict) -> dict:
    random.seed(int(payload["seed"]) + int(payload["start"]) * 104729)
    teachers = list(payload["teachers"])
    policies = [make_named_policy(name) for name in teachers]
    teacher_sampling_weights = _teacher_sampling_probabilities(
        teachers,
        str(payload.get("teacher_sampling_weights", "")),
    )
    config = parse_track(
        str(payload["track"]),
        vps_to_win=int(payload["vps_to_win"]),
        use_graph_history_features=bool(payload.get("graph_history_features", False)),
    )
    rows = []
    wins = 0
    for game_index in range(int(payload["start"]), int(payload["end"])):
        policy = _teacher_assignment(
            policies,
            game_index=game_index,
            players=config.players,
            mixed_seats=bool(payload.get("mixed_seats", False)),
            mixed_seat_mode=str(payload.get("mixed_seat_mode", "random")),
            teacher_sampling_weights=teacher_sampling_weights,
            seed=int(payload["seed"]),
        )
        episode = collect_imitation_game(
            policy,
            seed=int(payload["seed"]) + game_index,
            config=config,
            max_decisions=int(payload["max_decisions"]),
            rng=np.random.default_rng(int(payload["seed"]) + game_index * 9973),
        )
        wins += int(episode.result.winner is not None)
        for samples in episode.samples_by_player.values():
            for sample in samples:
                valid = np.asarray(sample.valid_actions, dtype=np.int16)
                target_policy, target_scores, target_policy_mask, target_scores_mask = _target_arrays(
                    valid,
                    getattr(sample, "target_policy", None),
                    getattr(sample, "target_scores", None),
                )
                rows.append(
                    {
                        "obs": np.asarray(sample.observation, dtype=np.float16),
                        "valid": valid,
                        "context": np.asarray(sample.action_context_features, dtype=np.float16)[valid],
                        "action": np.int16(sample.action),
                        "target_policy": target_policy,
                        "target_scores": target_scores,
                        "target_policy_mask": target_policy_mask,
                        "target_scores_mask": target_scores_mask,
                        "target_score_source": sample.target_score_source or "",
                        "teacher": _sample_teacher_name(sample, policy),
                        "seed": np.int64(episode.result.seed),
                        "player": sample.player,
                        "seat": np.int8(_seat_index(sample.player)),
                        "phase": sample.phase or "",
                        "decision_index": np.int32(
                            -1 if sample.decision_index is None else sample.decision_index
                        ),
                        "action_mask_version": sample.action_mask_version or "",
                        "winner": episode.result.winner or "",
                        "terminated": bool(episode.result.terminated),
                        "truncated": bool(episode.result.truncated),
                        "final_public_vps": np.asarray(
                            [
                                int(episode.result.final_public_vps.get(name, 0))
                                for name in PLAYER_NAMES
                            ],
                            dtype=np.int16,
                        ),
                        "has_final_public_vps": True,
                        "final_actual_vps": np.asarray(
                            [
                                int(episode.result.final_actual_vps.get(name, 0))
                                for name in PLAYER_NAMES
                            ],
                            dtype=np.int16,
                        ),
                        "has_final_actual_vps": True,
                    }
                )
    return {"games": int(payload["end"]) - int(payload["start"]), "wins": wins, "rows": rows}


def _teacher_assignment(
    policies: list,
    *,
    game_index: int,
    players: int,
    mixed_seats: bool,
    mixed_seat_mode: str = "random",
    teacher_sampling_weights: np.ndarray | None = None,
    seed: int = 0,
):
    if not mixed_seats:
        if teacher_sampling_weights is not None:
            rng = np.random.default_rng(int(seed) + int(game_index) * 13007 + int(players))
            index = int(rng.choice(len(policies), p=teacher_sampling_weights))
            return policies[index]
        return policies[game_index % len(policies)]
    names = PLAYER_NAMES[:players]
    if mixed_seat_mode == "random":
        rng = np.random.default_rng(int(seed) + int(game_index) * 13007 + int(players))
        indices = rng.choice(
            len(policies),
            size=len(names),
            replace=len(policies) < len(names),
            p=teacher_sampling_weights,
        )
        return {player: policies[int(indices[seat])] for seat, player in enumerate(names)}
    return {
        player: policies[(game_index + seat) % len(policies)]
        for seat, player in enumerate(names)
    }


def _teacher_sampling_probabilities(
    teachers: list[str],
    raw: str,
) -> np.ndarray | None:
    weights = _parse_weight_map(raw)
    if not weights:
        return None
    values = np.asarray([float(weights.get(name, 1.0)) for name in teachers], dtype=np.float64)
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    if float(values.sum()) <= 0.0:
        raise SystemExit("--teacher-sampling-weights produced zero total weight")
    return values / float(values.sum())


def _parse_weight_map(raw: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for part in str(raw or "").split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid weight entry: {item}")
        name, value = item.split("=", 1)
        result[name.strip()] = float(value)
    return result


def _sample_teacher_name(sample, policy) -> str:
    teacher_name = getattr(sample, "teacher_name", None)
    if teacher_name:
        return str(teacher_name)
    if isinstance(policy, dict):
        player = getattr(sample, "player", None)
        seat_policy = policy.get(player)
        if seat_policy is not None:
            return str(getattr(seat_policy, "name", type(seat_policy).__name__))
    return str(getattr(policy, "name", "mixed_teachers"))


def _target_arrays(
    valid: np.ndarray,
    target_policy: dict[int, float] | None,
    target_scores: dict[int, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    policy = np.zeros(len(valid), dtype=np.float16)
    scores = np.full(len(valid), np.nan, dtype=np.float32)
    policy_mask = np.zeros(len(valid), dtype=np.bool_)
    score_mask = np.zeros(len(valid), dtype=np.bool_)
    if target_policy:
        for idx, action in enumerate(valid):
            if int(action) in target_policy:
                value = float(target_policy[int(action)])
                if np.isfinite(value) and value > 0.0:
                    policy[idx] = np.float16(value)
                    policy_mask[idx] = True
        total = float(np.sum(policy.astype(np.float32)))
        if total > 0.0:
            policy = (policy.astype(np.float32) / total).astype(np.float16)
    if target_scores:
        for idx, action in enumerate(valid):
            if int(action) in target_scores:
                value = float(target_scores[int(action)])
                if np.isfinite(value):
                    scores[idx] = np.float32(value)
                    score_mask[idx] = True
    return policy, scores, policy_mask, score_mask


def _has_soft_policy(policy: np.ndarray) -> bool:
    values = np.asarray(policy, dtype=np.float32)
    if values.size == 0:
        return False
    return bool(np.sum(np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)) > 0.0)


def _has_soft_scores(scores: np.ndarray) -> bool:
    values = np.asarray(scores, dtype=np.float32)
    return bool(values.size and np.isfinite(values).any())


def _seat_index(player: str) -> int:
    try:
        return PLAYER_NAMES.index(player)
    except ValueError:
        return -1


def _prepare_output_dir(output: Path) -> None:
    if not output.exists():
        return
    stale = (
        list(output.glob("teacher_shard_*.npz"))
        + list(output.glob("teacher_shard_*.npz.zst"))
    )
    if stale or any(output.iterdir()):
        raise SystemExit(
            f"{output} already exists and is not empty; use a fresh --out so "
            "old shards cannot contaminate this teacher-data run."
        )


def _tool_provenance() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    files = [
        "tools/generate_teacher_data.py",
        "catan_rules_v1.json",
        "src/catan_zero/rules.py",
        "src/catan_zero/rl/action_mask.py",
        "src/catan_zero/rl/multiagent_env.py",
        "src/catan_zero/rl/self_play.py",
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/xdim_lite_policy.py",
        "src/catan_zero/rl/policy_pool.py",
        "tools/factory_common.py",
    ]
    hashes = {}
    for name in files:
        path = repo_root / name
        try:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return {
        "file_sha256": hashes,
        "feature_semantics_files": [
            "catan_rules_v1.json",
            "src/catan_zero/rules.py",
            "src/catan_zero/rl/action_mask.py",
            "src/catan_zero/rl/multiagent_env.py",
            "src/catan_zero/rl/self_play.py",
            "src/catan_zero/rl/action_features.py",
            "src/catan_zero/rl/xdim_lite_policy.py",
            "src/catan_zero/rl/policy_pool.py",
        ],
        "ab_target_scores_note": (
            "catanatron_ab* target_scores are root alpha-beta action values when "
            "generated with a CatanatronAlphaBetaPolicy that exposes root search "
            "scores; older manifests may contain fallback value-policy scores"
        ),
    }


if __name__ == "__main__":
    main()
