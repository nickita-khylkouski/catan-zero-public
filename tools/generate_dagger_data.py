from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

from catan_zero.rl.action_features import build_action_context_feature_table
from catan_zero.rl.entity_token_features import (
    ENTITY_TOKEN_SCHEMA_VERSION,
    build_entity_token_features,
)
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
from catan_zero.rl.self_play import StepSample, _phase_from_info

# Make the sibling ``tools/`` modules importable whether this module is run as a script
# (``python tools/generate_dagger_data.py``) or imported as a package submodule
# (``from tools.generate_dagger_data import ...``, e.g. from tests) -- mirrors the same
# bootstrap already used by ``tools/ppo_distributed_learner.py`` and ``tools/train_bc.py``.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from convert_teacher_to_entity_tokens import EntityShardWriter
from factory_common import make_named_policy, parse_track, write_json
from generate_teacher_data import (
    PLAYER_NAMES,
    ShardWriter,
    _seat_index,
    _target_arrays,
    _teacher_sampling_probabilities,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DAgger teacher shards: student executes actions, "
            "teachers label learner-visited states."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument(
        "--graph-history-features",
        action="store_true",
        help=(
            "Use the graph/history observation suffix. Enable this when mixing "
            "DAgger rows with the existing 35M entity corpus, whose obs width is 806."
        ),
    )
    parser.add_argument(
        "--label-teachers",
        default="catanatron_ab5,value_rollout_search,catanatron_ab4",
    )
    parser.add_argument(
        "--label-teacher-weights",
        default="catanatron_ab5=2,value_rollout_search=2,catanatron_ab4=1",
    )
    parser.add_argument("--opponents", default="catanatron_ab3,catanatron_value,heuristic")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--learner-seats", choices=("one", "all"), default="one")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-games", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--policy-weight-multiplier",
        type=float,
        default=1.0,
        help="Per-row policy sample multiplier written into DAgger shards.",
    )
    parser.add_argument(
        "--value-weight-multiplier",
        type=float,
        default=1.0,
        help=(
            "Per-row value sample multiplier written into DAgger shards for COMPLETED "
            "(non-truncated) games. FIX A6: previously defaulted to 0, so DAgger repair "
            "rounds could not train the value head at all even on games that finished "
            "cleanly. See --truncated-value-weight for the (still-unreliable) truncated case."
        ),
    )
    parser.add_argument(
        "--truncated-value-weight",
        type=float,
        default=0.0,
        help=(
            "Per-row value sample multiplier for rows from TRUNCATED games only (hit "
            "--max-decisions without a winner, so the terminal outcome/value target is "
            "unreliable). Defaults to 0 -- excluded from the value loss -- while completed "
            "games use --value-weight-multiplier."
        ),
    )
    parser.add_argument(
        "--entity-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write entity-token fields directly from the live learner-visited "
            "state. Keep this enabled for entity_graph training; replay-based "
            "conversion is invalid for DAgger rows because action_taken is the "
            "teacher label, not necessarily the student action that advanced the env."
        ),
    )
    parser.add_argument("--format", choices=("npz", "npz_zst"), default="npz")
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("teacher_shard_*.npz")) or any(out.glob("teacher_shard_*.npz.zst")):
        raise SystemExit(f"{out} already has teacher shards; use a fresh output dir")

    label_teacher_names = [x.strip() for x in args.label_teachers.split(",") if x.strip()]
    opponent_names = [x.strip() for x in args.opponents.split(",") if x.strip()]
    if not label_teacher_names:
        raise SystemExit("--label-teachers is empty")
    if not opponent_names:
        raise SystemExit("--opponents is empty")

    writer = (
        DaggerEntityShardWriter(out, args.shard_size, args.format)
        if args.entity_tokens
        else ShardWriter(out, args.shard_size, args.format)
    )
    start = time.perf_counter()
    total_games = 0
    total_samples = 0
    terminal_games = 0
    workers = max(1, int(args.workers))
    chunk_games = max(1, int(args.chunk_games))
    payloads = [
        {
            "start": i,
            "end": min(args.games, i + chunk_games),
            "checkpoint": args.checkpoint,
            "track": args.track,
            "graph_history_features": bool(args.graph_history_features),
            "vps_to_win": args.vps_to_win,
            "label_teachers": label_teacher_names,
            "label_teacher_weights": args.label_teacher_weights,
            "opponents": opponent_names,
            "seed": args.seed,
            "max_decisions": args.max_decisions,
            "learner_seats": args.learner_seats,
            "device": args.device,
            "policy_weight_multiplier": float(args.policy_weight_multiplier),
            "value_weight_multiplier": float(args.value_weight_multiplier),
            "truncated_value_weight": float(args.truncated_value_weight),
        }
        for i in range(0, args.games, chunk_games)
    ]

    if workers == 1:
        for payload in payloads:
            total_games, terminal_games, total_samples = _merge_result(
                _generate_chunk(payload),
                writer,
                total_games=total_games,
                terminal_games=terminal_games,
                total_samples=total_samples,
                workers=workers,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_chunk, payload) for payload in payloads]
            for future in as_completed(futures):
                total_games, terminal_games, total_samples = _merge_result(
                    future.result(),
                    writer,
                    total_games=total_games,
                    terminal_games=terminal_games,
                    total_samples=total_samples,
                    workers=workers,
                )

    shards = writer.close()
    report = {
        "schema": "dagger_entity_tokens_v1" if args.entity_tokens else "dagger_teacher_shards_v1",
        "entity_token_schema": ENTITY_TOKEN_SCHEMA_VERSION if args.entity_tokens else "",
        "checkpoint": args.checkpoint,
        "track": args.track,
        "graph_history_features": bool(args.graph_history_features),
        "vps_to_win": args.vps_to_win,
        "max_decisions": args.max_decisions,
        "learner_seats": args.learner_seats,
        "label_teachers": label_teacher_names,
        "label_teacher_weights": args.label_teacher_weights,
        "opponents": opponent_names,
        "games": int(args.games),
        "completed_games": int(total_games),
        "terminal_games": int(terminal_games),
        "samples": int(total_samples),
        "workers": workers,
        "chunk_games": chunk_games,
        "device": args.device,
        "entity_tokens": bool(args.entity_tokens),
        "policy_weight_multiplier": float(args.policy_weight_multiplier),
        "value_weight_multiplier": float(args.value_weight_multiplier),
        "truncated_value_weight": float(args.truncated_value_weight),
        "format": args.format,
        "shards": [str(path) for path in shards],
        "elapsed_sec": time.perf_counter() - start,
        **writer.summary(),
    }
    write_json(out / "manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _merge_result(
    result: dict[str, Any],
    writer: ShardWriter,
    *,
    total_games: int,
    terminal_games: int,
    total_samples: int,
    workers: int,
) -> tuple[int, int, int]:
    total_games += int(result["games"])
    terminal_games += int(result["terminal_games"])
    for row in result["rows"]:
        writer.add_row(row)
        total_samples += 1
    print(
        json.dumps(
            {
                "progress": "dagger_data",
                "games": total_games,
                "samples": total_samples,
                "workers": workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return total_games, terminal_games, total_samples


def _generate_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    seed = int(payload["seed"])
    start = int(payload["start"])
    random.seed(seed + start * 104729)
    config = parse_track(
        str(payload["track"]),
        vps_to_win=int(payload["vps_to_win"]),
        use_graph_history_features=bool(payload.get("graph_history_features", False)),
    )
    student = EntityGraphPolicy.load(
        str(payload["checkpoint"]),
        device=str(payload.get("device", "cpu")),
    )
    teachers = [make_named_policy(name) for name in payload["label_teachers"]]
    teacher_probs = _teacher_sampling_probabilities(
        list(payload["label_teachers"]),
        str(payload.get("label_teacher_weights", "")),
    )
    opponents = [make_named_policy(name) for name in payload["opponents"]]
    rows: list[dict[str, Any]] = []
    terminal_games = 0
    for game_index in range(start, int(payload["end"])):
        game_rows, terminal = _collect_game(
            student=student,
            teachers=teachers,
            teacher_names=list(payload["label_teachers"]),
            teacher_probs=teacher_probs,
            opponents=opponents,
            seed=seed + game_index,
            config=config,
            max_decisions=int(payload["max_decisions"]),
            learner_seats=str(payload.get("learner_seats", "one")),
            rng=np.random.default_rng(seed + game_index * 13007),
            policy_weight_multiplier=float(payload.get("policy_weight_multiplier", 1.0)),
            value_weight_multiplier=float(payload.get("value_weight_multiplier", 1.0)),
            truncated_value_weight=float(payload.get("truncated_value_weight", 0.0)),
        )
        rows.extend(game_rows)
        terminal_games += int(terminal)
    return {"games": int(payload["end"]) - start, "terminal_games": terminal_games, "rows": rows}


def _collect_game(
    *,
    student,
    teachers,
    teacher_names,
    teacher_probs,
    opponents,
    seed: int,
    config,
    max_decisions: int,
    learner_seats: str,
    rng: np.random.Generator,
    policy_weight_multiplier: float,
    value_weight_multiplier: float,
    truncated_value_weight: float = 0.0,
):
    env = ColonistMultiAgentEnv(config)
    captured: list[tuple[StepSample, str]] = []
    try:
        observations, info = env.reset(seed=int(seed))
        rewards = {name: 0.0 for name in env.player_names}
        terminated = False
        truncated = False
        decisions = 0
        player_names = tuple(env.player_names)
        learner_set = set(player_names) if learner_seats == "all" else {player_names[int(seed) % len(player_names)]}
        while not (terminated or truncated) and decisions < max_decisions:
            player = str(info["current_player"])
            observation = np.asarray(observations[player], dtype=np.float64)
            valid_actions = tuple(int(a) for a in info["valid_actions"])
            if player in learner_set:
                if teacher_probs is not None:
                    teacher_index = int(rng.choice(len(teachers), p=teacher_probs))
                else:
                    teacher_index = int(rng.integers(len(teachers)))
                teacher = teachers[teacher_index]
                teacher_name = str(teacher_names[teacher_index])
                teacher_action = int(teacher.select_action(env, observation, info, rng, training=False))
                target_policy_fn = getattr(teacher, "target_policy", None)
                target_scores_fn = getattr(teacher, "target_scores", None)
                target_policy = target_policy_fn(env, info, rng) if callable(target_policy_fn) else None
                target_scores = target_scores_fn(env, info, rng) if callable(target_scores_fn) else None
                target_score_source = _target_score_source(teacher, target_scores)
                entity_features = build_entity_token_features(env, player)
                captured.append(
                    (
                        StepSample(
                            observation=observation.copy(),
                            valid_actions=valid_actions,
                            action=teacher_action,
                            player=player,
                            action_context_features=build_action_context_feature_table(env, info),
                            phase=_phase_from_info(info),
                            target_policy=target_policy,
                            target_scores=target_scores,
                            target_score_source=target_score_source,
                            decision_index=decisions,
                            teacher_name=teacher_name,
                            action_mask_version=str(info.get("action_mask_version", "")),
                        ),
                        teacher_name,
                        entity_features,
                    )
                )
                action = int(student.select_action(env, observation, info, rng, training=True))
            else:
                opp_index = (int(seed) + decisions + _seat_index(player)) % len(opponents)
                action = int(opponents[opp_index].select_action(env, observation, info, rng, training=False))
            observations, rewards, terminated, truncated, info = env.step(action)
            decisions += 1

        if not terminated and decisions >= max_decisions:
            truncated = True
        terminated, truncated = _canonical_episode_status(
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        winner = _winner_from_rewards(rewards)
        final_public_vps = _final_vps(env, actual=False)
        final_actual_vps = _final_vps(env, actual=True)
        effective_value_weight_multiplier = _effective_value_weight_multiplier(
            truncated=bool(truncated),
            value_weight_multiplier=value_weight_multiplier,
            truncated_value_weight=truncated_value_weight,
        )
        rows = [
            _row_from_sample(
                sample,
                teacher=teacher_name,
                entity_features=entity_features,
                game_seed=int(seed),
                winner=winner,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_public_vps=final_public_vps,
                final_actual_vps=final_actual_vps,
                policy_weight_multiplier=policy_weight_multiplier,
                value_weight_multiplier=effective_value_weight_multiplier,
            )
            for sample, teacher_name, entity_features in captured
        ]
        return rows, bool(terminated and winner)
    finally:
        env.close()


def _effective_value_weight_multiplier(
    *,
    truncated: bool,
    value_weight_multiplier: float,
    truncated_value_weight: float,
) -> float:
    """FIX A6: truncated games (hit --max-decisions without a winner) have an unreliable
    terminal value target, so they get their OWN (separately configurable) multiplier instead
    of silently sharing --value-weight-multiplier with completed games."""
    return float(truncated_value_weight if truncated else value_weight_multiplier)


def _canonical_episode_status(
    *,
    terminated: bool,
    truncated: bool,
) -> tuple[bool, bool]:
    """Give a realized terminal outcome precedence over a simultaneous limit.

    Gym-style environments may report both flags on the action that wins at
    the turn limit.  DAgger rows carry an exact winner in that case, so marking
    them truncated would make ``train_bc._value_targets`` discard the clean
    +/-1 outcome and route the rows through the lower-confidence truncation
    path instead.
    """

    terminal = bool(terminated)
    return terminal, bool(truncated) and not terminal


def _row_from_sample(
    sample: StepSample,
    *,
    teacher: str,
    entity_features: dict[str, np.ndarray] | None = None,
    game_seed: int,
    winner: str | None,
    terminated: bool,
    truncated: bool,
    final_public_vps: dict[str, int],
    final_actual_vps: dict[str, int],
    policy_weight_multiplier: float = 1.0,
    value_weight_multiplier: float = 0.0,
) -> dict[str, Any]:
    valid = np.asarray(sample.valid_actions, dtype=np.int16)
    target_policy, target_scores, target_policy_mask, target_scores_mask = _target_arrays(
        valid,
        sample.target_policy,
        sample.target_scores,
    )
    row = {
        "obs": np.asarray(sample.observation, dtype=np.float16),
        "valid": valid,
        "context": np.asarray(sample.action_context_features, dtype=np.float16)[valid],
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
        "decision_index": np.int32(-1 if sample.decision_index is None else sample.decision_index),
        "action_mask_version": sample.action_mask_version or "",
        "winner": winner or "",
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_public_vps": np.asarray(
            [int(final_public_vps.get(name, 0)) for name in PLAYER_NAMES],
            dtype=np.int16,
        ),
        "has_final_public_vps": True,
        "final_actual_vps": np.asarray(
            [int(final_actual_vps.get(name, 0)) for name in PLAYER_NAMES],
            dtype=np.int16,
        ),
        "has_final_actual_vps": True,
        "policy_weight_multiplier": np.float32(policy_weight_multiplier),
        "value_weight_multiplier": np.float32(value_weight_multiplier),
    }
    if entity_features is not None:
        row["_entity_features"] = entity_features
    return row


class DaggerEntityShardWriter:
    """Entity-token writer with the same summary contract as ShardWriter."""

    def __init__(self, output: Path, shard_size: int, fmt: str) -> None:
        self._writer = EntityShardWriter(output, shard_size=shard_size, fmt=fmt)
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

    def add_row(self, row: dict[str, Any]) -> None:
        features = row.get("_entity_features")
        if features is None:
            raise ValueError("DaggerEntityShardWriter requires _entity_features")
        record = self._base_record(row)
        self._writer.add(record, features)
        self._update_summary(row)

    def close(self) -> list[Path]:
        self._writer.close()
        return self._writer.paths

    def summary(self) -> dict[str, Any]:
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

    def _base_record(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "obs": np.asarray(row["obs"], dtype=np.float16),
            "legal_action_ids": np.asarray(row["valid"], dtype=np.int16),
            "legal_action_context": np.asarray(row["context"], dtype=np.float16),
            "action_taken": np.int16(row["action"]),
            "target_policy": np.asarray(row["target_policy"], dtype=np.float16),
            "target_scores": np.asarray(row["target_scores"], dtype=np.float32),
            "target_policy_mask": np.asarray(row["target_policy_mask"], dtype=np.bool_),
            "target_scores_mask": np.asarray(row["target_scores_mask"], dtype=np.bool_),
            "target_score_source": str(row.get("target_score_source", "")),
            "game_seed": np.int64(row["seed"]),
            "teacher_name": str(row["teacher"]),
            "player": str(row.get("player", "")),
            "seat": np.int8(row.get("seat", _seat_index(str(row.get("player", ""))))),
            "phase": str(row.get("phase", "")),
            "decision_index": np.int32(row.get("decision_index", -1)),
            "winner": str(row.get("winner", "")),
            "terminated": bool(row.get("terminated", True)),
            "truncated": bool(row.get("truncated", False)),
            "final_public_vps": np.asarray(row["final_public_vps"], dtype=np.int16),
            "has_final_public_vps": bool(row.get("has_final_public_vps", False)),
            "final_actual_vps": np.asarray(row["final_actual_vps"], dtype=np.int16),
            "has_final_actual_vps": bool(row.get("has_final_actual_vps", False)),
            "action_mask_version": str(row.get("action_mask_version", "")),
            "policy_weight_multiplier": np.float32(row.get("policy_weight_multiplier", 1.0)),
            "value_weight_multiplier": np.float32(row.get("value_weight_multiplier", 1.0)),
        }

    def _update_summary(self, row: dict[str, Any]) -> None:
        teacher = str(row["teacher"])
        phase = str(row.get("phase", "")) or "unknown"
        score_source = str(row.get("target_score_source", "")) or "none"
        legal = np.asarray(row["valid"], dtype=np.int16)
        legal_count = int(np.sum(legal >= 0))
        action = int(row["action"])
        legal_set = set(map(int, legal[legal >= 0]))
        self.teacher_counts[teacher] = self.teacher_counts.get(teacher, 0) + 1
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        self.score_source_counts[score_source] = self.score_source_counts.get(score_source, 0) + 1
        self.forced_actions += int(legal_count <= 1)
        self.invalid_teacher_actions += int(action not in legal_set)
        self.soft_policy_rows += int(_has_soft_policy(row["target_policy"]))
        self.soft_score_rows += int(_has_soft_scores(row["target_scores"]))
        winner = str(row.get("winner", ""))
        truncated = bool(row.get("truncated", False))
        player = str(row.get("player", ""))
        has_outcome = bool(winner)
        self.outcome_rows += int(has_outcome)
        self.clean_terminal_outcome_rows += int(has_outcome and not truncated)
        self.final_public_vp_rows += int(bool(row.get("has_final_public_vps", False)))
        self.final_actual_vp_rows += int(bool(row.get("has_final_actual_vps", False)))
        self.truncated_rows += int(truncated)
        self.legal_counts.append(legal_count)


def _has_soft_policy(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=np.float32)
    return bool(arr.size and np.any(arr > 0.0))


def _has_soft_scores(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=np.float32)
    return bool(arr.size and np.any(np.isfinite(arr)))


def _target_score_source(teacher, target_scores) -> str:
    fn = getattr(teacher, "target_score_source", None)
    if callable(fn):
        return str(fn())
    if target_scores:
        return str(getattr(teacher, "name", type(teacher).__name__))
    return ""


def _winner_from_rewards(rewards: dict[str, float]) -> str | None:
    if not rewards:
        return None
    best = max(rewards.items(), key=lambda item: item[1])
    return str(best[0]) if float(best[1]) > 0.0 else None


def _final_vps(env: ColonistMultiAgentEnv, *, actual: bool) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in env.player_names:
        try:
            payload = env.observation_payload(name)
            pdata = payload.get("players", {}).get(name, {})
            key = "actual_victory_points" if actual else "public_victory_points"
            out[name] = int(pdata.get(key, pdata.get("victory_points", 0)))
        except Exception:
            out[name] = 0
    return out


if __name__ == "__main__":
    main()
