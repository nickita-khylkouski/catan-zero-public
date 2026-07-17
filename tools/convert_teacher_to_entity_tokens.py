from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from catan_zero.deduction_tracker import (
    DEDUCTION_FEATURE_SIZE,
    RESOURCES,
    DeductionTracker,
)
from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    RUST_ENTITY_ADAPTER_V6,
)
from catan_zero.rl.entity_token_features import (
    ENTITY_TOKEN_SCHEMA_VERSION,
    build_entity_token_features,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv

# Make the sibling ``tools/`` modules importable whether this module is run as a script
# (``python tools/convert_teacher_to_entity_tokens.py``) or imported as a package
# submodule (``from tools.convert_teacher_to_entity_tokens import ...``, e.g. from
# tests) -- mirrors the bootstrap already used by ``tools/train_bc.py`` and
# ``tools/generate_dagger_data.py``.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from factory_common import (  # noqa: E402
    parse_track,
    propagated_hard_action_target_information,
    write_json,
)
from curate_teacher_data import (  # noqa: E402
    TOOL_PROVENANCE_SCHEMA,
    _hash_required_files,
    _input_manifests,
)

# Additive-only (CAT-59): included in shards only when --emit-deduction-features
# is passed. `(4, DEDUCTION_FEATURE_SIZE)` per row, same per-player-slot
# ordering as `player_tokens` -- see `catan_zero.deduction_tracker.
# DeductionTracker.feature_table`. Model wiring (near-zero-init projection
# onto the player-token rows) is left to the consuming ticket (CAT-23).
DEDUCTION_FEATURES_KEY = "deduction_features"


BASE_KEYS = (
    "obs",
    "legal_action_ids",
    "legal_action_context",
    "action_taken",
    "target_policy",
    "target_scores",
    "target_policy_mask",
    "target_scores_mask",
    "target_score_source",
    "target_information_regime",
    "game_seed",
    "teacher_name",
    "player",
    "seat",
    "phase",
    "decision_index",
    "winner",
    "terminated",
    "truncated",
    "final_public_vps",
    "has_final_public_vps",
    "final_actual_vps",
    "has_final_actual_vps",
    "action_mask_version",
    "policy_weight_multiplier",
    "value_weight_multiplier",
    "adapter_version",
    "actor_resource_counts",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay teacher shards and add typed Catan entity-token tensors."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--graph-history-features", action="store_true")
    parser.add_argument(
        "--emit-deduction-features",
        action="store_true",
        help=(
            "Add an additive per-row 'deduction_features' tensor "
            f"(4, {DEDUCTION_FEATURE_SIZE}) from the CAT-59 exact-deduction "
            "tracker, replayed incrementally per game/player. Off by default; "
            "does not touch player_tokens or any other existing key."
        ),
    )
    parser.add_argument("--format", choices=("npz", "npz_zst"), default="npz_zst")
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=0)
    parser.add_argument(
        "--partition-count",
        type=int,
        default=1,
        help="Only convert seeds where seed %% partition-count == partition-index.",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=0,
        help="Partition index used with --partition-count for parallel conversion.",
    )
    parser.add_argument("--require-complete-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-duplicate-conflicts",
        action="store_true",
        help=(
            "Skip an entire seed if duplicated rows for the same decision/player "
            "disagree on action or legal actions. Default is strict failure."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    entity_keys = ENTITY_KEYS + (DEDUCTION_FEATURES_KEY,) if args.emit_deduction_features else ENTITY_KEYS
    writer = EntityShardWriter(
        output, shard_size=int(args.shard_size), fmt=args.format, entity_keys=entity_keys
    )
    if int(args.partition_count) <= 0:
        raise SystemExit("--partition-count must be positive")
    if int(args.partition_index) < 0 or int(args.partition_index) >= int(args.partition_count):
        raise SystemExit("--partition-index must be in [0, partition-count)")
    config = parse_track(
        args.track,
        vps_to_win=int(args.vps_to_win),
        use_graph_history_features=bool(args.graph_history_features),
    )
    input_manifests = _input_manifests(args.data)
    summary: dict[str, Any] = {
        "inputs": args.data,
        "input_manifests": input_manifests,
        "hard_action_target_information": propagated_hard_action_target_information(
            input_manifests
        ),
        "out": str(output),
        "track": args.track,
        "vps_to_win": int(args.vps_to_win),
        "graph_history_features": bool(args.graph_history_features),
        "emit_deduction_features": bool(args.emit_deduction_features),
        "schema": ENTITY_TOKEN_SCHEMA_VERSION,
        "entity_feature_adapter_version": CURRENT_RUST_ENTITY_ADAPTER_VERSION,
        "tool_provenance": _tool_provenance(),
        "partition_count": int(args.partition_count),
        "partition_index": int(args.partition_index),
        "loaded_rows": 0,
        "candidate_seeds": 0,
        "converted_rows": 0,
        "converted_seeds": 0,
        "skipped_incomplete_prefix": 0,
        "skipped_duplicate_conflict_seeds": 0,
        "skipped_duplicate_conflict_rows": 0,
        "skipped_wrong_player_duplicate_rows": 0,
        "duplicate_decision_rows": 0,
        "noncontiguous_seed_repeats": 0,
        "mismatches": [],
        "shards": [],
    }
    started = time.perf_counter()
    next_progress = int(args.progress_every) if int(args.progress_every) > 0 else 0
    seed_iter = _iter_seed_rows(
        [Path(path) for path in args.data],
        max_rows=int(args.max_rows),
        partition_count=int(args.partition_count),
        partition_index=int(args.partition_index),
    )
    for seed_index, (seed, seed_rows) in enumerate(seed_iter, start=1):
        if int(args.max_seeds) > 0 and seed_index > int(args.max_seeds):
            break
        summary["candidate_seeds"] += 1
        summary["loaded_rows"] += len(seed_rows)
        by_decision: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in seed_rows:
            by_decision[int(row["decision_index"])].append(row)
        summary["duplicate_decision_rows"] += sum(
            max(0, len(decision_rows) - 1)
            for decision_rows in by_decision.values()
        )
        if args.skip_duplicate_conflicts:
            duplicate_conflict = _first_duplicate_decision_error(seed, by_decision)
            if duplicate_conflict is not None:
                summary["skipped_duplicate_conflict_seeds"] += 1
                summary["skipped_duplicate_conflict_rows"] += len(seed_rows)
                continue
        if args.require_complete_prefix:
            max_decision = max(by_decision) if by_decision else -1
            missing = [idx for idx in range(max_decision + 1) if idx not in by_decision]
            if missing:
                summary["skipped_incomplete_prefix"] += 1
                continue
        result = _convert_seed(
            seed,
            by_decision,
            config,
            writer,
            emit_deduction_features=bool(args.emit_deduction_features),
        )
        summary["converted_rows"] += int(result["converted_rows"])
        summary["skipped_wrong_player_duplicate_rows"] += int(
            result.get("skipped_wrong_player_duplicate_rows", 0)
        )
        summary["converted_seeds"] += 1
        summary["mismatches"].extend(result["mismatches"])
        if summary["mismatches"]:
            break
        if next_progress and int(summary["converted_rows"]) >= next_progress:
            print(
                json.dumps(
                    {
                        "progress": "entity_convert",
                        "converted_rows": int(summary["converted_rows"]),
                        "converted_seeds": int(summary["converted_seeds"]),
                        "elapsed_sec": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            next_progress += int(args.progress_every)
    writer.close()
    summary["shards"] = [str(path) for path in writer.paths]
    summary["elapsed_sec"] = time.perf_counter() - started
    summary["rows_per_sec"] = (
        float(summary["converted_rows"]) / max(float(summary["elapsed_sec"]), 1.0e-9)
    )
    summary["unconverted_rows_after_replay"] = max(
        0,
        int(summary["loaded_rows"])
        - int(summary["converted_rows"])
        - int(summary.get("skipped_duplicate_conflict_rows", 0))
        - int(summary.get("skipped_wrong_player_duplicate_rows", 0)),
    )
    if (
        not summary["mismatches"]
        and int(args.max_rows) <= 0
        and int(args.max_seeds) <= 0
        and int(summary["skipped_incomplete_prefix"]) == 0
        and int(summary["converted_rows"]) > int(summary["loaded_rows"])
    ):
        summary["mismatches"].append(
            {
                "kind": "converted_row_count_mismatch",
                "loaded_rows": int(summary["loaded_rows"]),
                "converted_rows": int(summary["converted_rows"]),
            }
        )
    write_json(output / "manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["mismatches"]:
        raise SystemExit(1)


def _tool_provenance() -> dict[str, Any]:
    """Bind the replay/entity transformation and its feature semantics.

    The input manifests preserve the authenticated teacher-generation lineage;
    these hashes additionally identify the conversion code that produced the
    entity-token tensors consumed by training.
    """

    repo_root = Path(__file__).resolve().parents[1]
    files = [
        "tools/convert_teacher_to_entity_tokens.py",
        "src/catan_zero/rl/entity_token_features.py",
        "src/catan_zero/rl/entity_feature_adapter.py",
        "src/catan_zero/rl/multiagent_env.py",
        "src/catan_zero/rl/action_features.py",
    ]
    hashes = _hash_required_files(repo_root, files)
    return {
        "schema_version": TOOL_PROVENANCE_SCHEMA,
        "file_sha256": hashes,
        "feature_semantics_files": files,
    }


def _iter_seed_rows(
    roots: list[Path],
    *,
    max_rows: int,
    partition_count: int = 1,
    partition_index: int = 0,
):
    current_key: tuple[int, int] | None = None
    current_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    emitted_rows = 0
    for row in _iter_rows(
        roots,
        max_rows=max_rows,
        partition_count=partition_count,
        partition_index=partition_index,
    ):
        seed = int(row["game_seed"])
        source_index = int(row.pop("__source_index", 0))
        key = (source_index, seed)
        if current_key is None:
            current_key = key
        elif key != current_key:
            if current_key in seen_keys:
                raise SystemExit(
                    f"non-contiguous repeated source/game_seed {current_key[0]}/{current_key[1]}"
                )
            seen_keys.add(current_key)
            yield current_key[1], current_rows
            current_key = key
            current_rows = []
        current_rows.append(row)
        emitted_rows += 1
    if current_key is not None:
        if current_key in seen_keys:
            raise SystemExit(
                f"non-contiguous repeated source/game_seed {current_key[0]}/{current_key[1]}"
            )
        yield current_key[1], current_rows


def _iter_rows(
    roots: list[Path],
    *,
    max_rows: int,
    partition_count: int = 1,
    partition_index: int = 0,
):
    emitted = 0
    for source_index, root in enumerate(roots):
        for shard in _shards(root):
            arrays = _load_npz(shard)
            n = len(arrays["action_taken"])
            for idx in range(n):
                seed = int(arrays["game_seed"][idx])
                if partition_count > 1 and seed % partition_count != partition_index:
                    continue
                row = {key: _field(arrays, key, idx) for key in BASE_KEYS if key in arrays}
                row["__source_index"] = source_index
                yield row
                emitted += 1
                if max_rows > 0 and emitted >= max_rows:
                    return


def _convert_seed(
    seed: int,
    by_decision: dict[int, list[dict[str, Any]]],
    config,
    writer,
    *,
    emit_deduction_features: bool = False,
) -> dict[str, Any]:
    env = ColonistMultiAgentEnv(config)
    mismatches: list[dict[str, Any]] = []
    converted_rows = 0
    skipped_wrong_player_duplicate_rows = 0
    # One DeductionTracker per seat, fed incrementally (only the frames
    # produced since its own last call) as the single game-length env is
    # stepped forward -- O(decisions) total per game, not O(decisions^2).
    trackers: dict[str, DeductionTracker] = {}
    tracker_cursor: dict[str, int] = {}
    try:
        _, info = env.reset(seed=int(seed))
        decision = 0
        while decision in by_decision:
            rows = by_decision[decision]
            row = rows[0]
            valid = tuple(int(value) for value in info.get("valid_actions", ()))
            runtime_player = str(info.get("current_player", ""))
            matching_rows = [candidate for candidate in rows if str(candidate["player"]) == runtime_player]
            if matching_rows:
                skipped_wrong_player_duplicate_rows += len(rows) - len(matching_rows)
                rows = matching_rows
                row = rows[0]
            player = str(row["player"])
            action = int(row["action_taken"])
            row_valid = tuple(int(value) for value in row["legal_action_ids"] if int(value) >= 0)
            duplicate_error = _duplicate_decision_error(seed, decision, rows)
            if duplicate_error:
                mismatches.append(duplicate_error)
                break
            if player != runtime_player:
                mismatches.append(_mismatch(seed, decision, "player", row, runtime_player))
                break
            if row_valid != valid:
                mismatches.append(_mismatch(seed, decision, "valid_actions", row, valid[:20]))
                break
            features = build_entity_token_features(env, player)
            actor_resource_counts: np.ndarray | None = None
            if CURRENT_RUST_ENTITY_ADAPTER_VERSION == RUST_ENTITY_ADAPTER_V6:
                # `observation_payload(player)` exposes exact resources only
                # for that same acting player. It is therefore authoritative
                # own-private evidence, not an opponent hidden-state leak.
                actor_payload = env.observation_payload(
                    player, include_event_log=False
                ).get("players", {}).get(player, {})
                actor_resources = actor_payload.get("resources")
                if not isinstance(actor_resources, dict):
                    mismatches.append(
                        _mismatch(
                            seed,
                            decision,
                            "authoritative_actor_resource_counts_missing",
                            row,
                            actor_resources,
                        )
                    )
                    break
                actor_resource_counts = np.asarray(
                    [
                        int(actor_resources.get(resource, -1))
                        for resource in RESOURCES
                    ],
                    dtype=np.int16,
                )
                if np.any(actor_resource_counts < 0) or np.any(
                    actor_resource_counts > 19
                ):
                    mismatches.append(
                        _mismatch(
                            seed,
                            decision,
                            "authoritative_actor_resource_counts_invalid",
                            row,
                            actor_resource_counts,
                        )
                    )
                    break
                # `rows` contains only duplicates of this exact decision, not
                # earlier/later decisions from the game.
                stale_witness = None
                for candidate in rows:
                    if "actor_resource_counts" not in candidate:
                        continue
                    candidate_witness = np.asarray(
                        candidate["actor_resource_counts"]
                    )
                    if (
                        candidate_witness.shape != (5,)
                        or candidate_witness.dtype.kind not in {"i", "u"}
                        or not np.array_equal(
                            candidate_witness, actor_resource_counts
                        )
                    ):
                        stale_witness = candidate_witness
                        break
                if stale_witness is not None:
                    mismatches.append(
                        _mismatch(
                            seed,
                            decision,
                            "actor_resource_counts_replay_mismatch",
                            row,
                            actor_resource_counts,
                        )
                    )
                    break
            if emit_deduction_features:
                if player not in trackers:
                    opponents = tuple(name for name in env.player_names if name != player)
                    trackers[player] = DeductionTracker(self_name=player, opponent_names=opponents)
                    tracker_cursor[player] = 0
                tracker = trackers[player]
                # `env.replay_trace()` re-redacts every frame from game start
                # on each call; redact only the NEW frames directly to keep
                # this O(decisions) per game rather than O(decisions^2).
                new_frames = [
                    env._redact_replay_frame(frame, player)
                    for frame in env._replay_frames[tracker_cursor[player] :]
                ]
                tracker.observe_frames(new_frames)
                tracker_cursor[player] = len(env._replay_frames)
                features[DEDUCTION_FEATURES_KEY] = tracker.feature_table(
                    env.observation_payload(player, include_event_log=False)
                )
            legal_action_ids = np.asarray(row["legal_action_ids"], dtype=np.int16)
            runtime_action_ids = np.asarray(
                [
                    int(action.get("index", -1))
                    for action in env.observation_payload(player).get("structured_legal_actions", ())
                ],
                dtype=np.int16,
            )
            runtime_action_ids = _pad_1d(runtime_action_ids, legal_action_ids.shape[0], fill=-1)
            if not np.array_equal(runtime_action_ids, legal_action_ids):
                mismatches.append(
                    _mismatch(seed, decision, "entity_legal_action_order", row, runtime_action_ids[:20])
                )
                break
            for duplicate_row in rows:
                converted_row = {
                    **duplicate_row,
                    "adapter_version": CURRENT_RUST_ENTITY_ADAPTER_VERSION,
                }
                if actor_resource_counts is None:
                    # The current converter still emits adapter v3. Do not
                    # preserve an input v6 witness beside downgraded v3 token
                    # semantics; the pair would be internally contradictory.
                    converted_row.pop("actor_resource_counts", None)
                else:
                    converted_row["actor_resource_counts"] = actor_resource_counts
                writer.add(
                    converted_row,
                    features,
                )
            _, _, terminated, truncated, info = env.step(action)
            converted_rows += len(rows)
            decision += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "converted_rows": converted_rows,
        "mismatches": mismatches,
        "skipped_wrong_player_duplicate_rows": skipped_wrong_player_duplicate_rows,
    }


def _duplicate_decision_error(
    seed: int,
    decision: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(rows) <= 1:
        return None
    first = rows[0]
    first_legal = np.asarray(first["legal_action_ids"], dtype=np.int16)
    for row in rows[1:]:
        if str(row["player"]) != str(first["player"]):
            return _mismatch(seed, decision, "duplicate_player_conflict", row, str(first["player"]))
        if int(row["action_taken"]) != int(first["action_taken"]):
            return _mismatch(seed, decision, "duplicate_action_conflict", row, int(first["action_taken"]))
        if not np.array_equal(np.asarray(row["legal_action_ids"], dtype=np.int16), first_legal):
            return _mismatch(
                seed,
                decision,
                "duplicate_legal_actions_conflict",
                row,
                first_legal[:20],
            )
    return None


def _first_duplicate_decision_error(
    seed: int,
    by_decision: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    for decision, rows in by_decision.items():
        duplicate_error = _duplicate_decision_error(seed, decision, rows)
        if duplicate_error is not None:
            return duplicate_error
    return None


def _mismatch(seed: int, decision: int, kind: str, row: dict[str, Any], runtime: Any) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "decision_index": int(decision),
        "kind": kind,
        "row_player": str(row.get("player", "")),
        "row_teacher": str(row.get("teacher_name", "")),
        "row_action": int(row.get("action_taken", -1)),
        "row_phase": str(row.get("phase", "")),
        "runtime": _jsonable(runtime),
    }


class EntityShardWriter:
    def __init__(
        self,
        output: Path,
        *,
        shard_size: int,
        fmt: str,
        entity_keys: tuple[str, ...] = ENTITY_KEYS,
    ) -> None:
        self.output = output
        self.shard_size = max(1, int(shard_size))
        self.format = fmt
        self.entity_keys = entity_keys
        self.rows: list[dict[str, Any]] = []
        self.paths: list[Path] = []
        self.index = 0

    def add(self, row: dict[str, Any], features: dict[str, np.ndarray]) -> None:
        payload = {key: row[key] for key in BASE_KEYS if key in row}
        for key in self.entity_keys:
            payload[key] = features[key]
        self.rows.append(payload)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def close(self) -> None:
        self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        arrays = _rows_to_arrays(self.rows, entity_keys=self.entity_keys)
        path = self.output / f"entity_teacher_shard_{self.index:05d}.npz"
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if self.format == "npz_zst":
            path = _try_zstd(path)
        self.paths.append(path)
        self.rows = []
        self.index += 1


def _rows_to_arrays(
    rows: list[dict[str, Any]],
    *,
    entity_keys: tuple[str, ...] = ENTITY_KEYS,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    legal_width = max(int(np.asarray(row["legal_action_ids"]).shape[0]) for row in rows)
    for key in BASE_KEYS:
        if key not in rows[0]:
            continue
        values = [row[key] for row in rows]
        if key in {"legal_action_ids", "target_policy", "target_scores", "target_policy_mask", "target_scores_mask"}:
            fill = -1 if key == "legal_action_ids" else np.nan if key == "target_scores" else False if key.endswith("_mask") else 0.0
            out[key] = np.stack([_pad_1d(np.asarray(value), legal_width, fill=fill) for value in values], axis=0)
        elif key == "legal_action_context":
            feature_size = int(np.asarray(values[0]).shape[1])
            out[key] = np.stack(
                [_pad_2d(np.asarray(value), legal_width, feature_size, fill=0.0) for value in values],
                axis=0,
            )
        else:
            out[key] = np.asarray(values)
    for key in entity_keys:
        values = [row[key] for row in rows]
        if key in {"legal_action_tokens", "legal_action_target_ids", "legal_action_mask"}:
            if key == "legal_action_tokens":
                out[key] = np.stack(
                    [
                        _pad_2d(np.asarray(value), legal_width, np.asarray(value).shape[1], fill=0.0)
                        for value in values
                    ],
                    axis=0,
                ).astype(np.float16, copy=False)
            elif key == "legal_action_target_ids":
                out[key] = np.stack(
                    [_pad_2d(np.asarray(value), legal_width, 4, fill=-1) for value in values],
                    axis=0,
                ).astype(np.int16, copy=False)
            else:
                out[key] = np.stack(
                    [_pad_1d(np.asarray(value), legal_width, fill=False) for value in values],
                    axis=0,
                ).astype(np.bool_, copy=False)
        else:
            out[key] = np.stack(values, axis=0)
    return out


def _pad_1d(value: np.ndarray, width: int, *, fill: Any) -> np.ndarray:
    value = np.asarray(value)
    out = np.full((int(width),), fill, dtype=value.dtype)
    count = min(int(width), int(value.shape[0]))
    out[:count] = value[:count]
    return out


def _pad_2d(value: np.ndarray, width: int, feature_size: int, *, fill: Any) -> np.ndarray:
    value = np.asarray(value)
    out = np.full((int(width), int(feature_size)), fill, dtype=value.dtype)
    rows = min(int(width), int(value.shape[0]))
    cols = min(int(feature_size), int(value.shape[1]))
    out[:rows, :cols] = value[:rows, :cols]
    return out


def _load_rows(
    roots: list[Path],
    *,
    max_rows: int,
    partition_count: int = 1,
    partition_index: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for shard in _shards(root):
            arrays = _load_npz(shard)
            n = len(arrays["action_taken"])
            for idx in range(n):
                seed = int(arrays["game_seed"][idx])
                if partition_count > 1 and seed % partition_count != partition_index:
                    continue
                rows.append({key: _field(arrays, key, idx) for key in BASE_KEYS if key in arrays})
                if max_rows > 0 and len(rows) >= max_rows:
                    return rows
    return rows


def _shards(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    manifest = root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        files = []
        for value in payload.get("shards", ()):
            path = Path(value)
            if path.is_absolute():
                files.append(path)
            elif path.exists():
                files.append(path)
            elif (manifest.parent / path).exists():
                files.append(manifest.parent / path)
            elif (manifest.parent / path.name).exists():
                files.append(manifest.parent / path.name)
            else:
                files.append(manifest.parent / path)
        if files:
            return files
    parts = root / "parts"
    if parts.exists():
        files = sorted(parts.glob("part_*/*.npz")) + sorted(parts.glob("part_*/*.npz.zst"))
        if files:
            return files
    files = sorted(root.glob("*.npz")) + sorted(root.glob("*.npz.zst"))
    if files:
        return files
    return sorted(root.glob("*.npz")) + sorted(root.glob("*.npz.zst"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".zst":
        import zstandard as zstd

        data = zstd.ZstdDecompressor().decompress(path.read_bytes())
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _field(arrays: dict[str, np.ndarray], key: str, idx: int) -> Any:
    return arrays[key][idx]


def _try_zstd(path: Path) -> Path:
    try:
        import zstandard as zstd
    except ImportError:
        return path
    compressed = path.with_suffix(path.suffix + ".zst")
    tmp = compressed.with_name(compressed.name + ".tmp")
    tmp.write_bytes(zstd.ZstdCompressor(level=3).compress(path.read_bytes()))
    os.replace(tmp, compressed)
    path.unlink()
    return compressed


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
