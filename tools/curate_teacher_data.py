from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.action_mask import ActionCatalog

from factory_common import propagated_hard_action_target_information, write_json


# ``phase`` is the engine's public ``current_prompt`` in current teacher shards.
# Keep the old aliases because legacy teacher data used the coarser lowercase
# names, but never make production labels pass through the legacy spelling
# accidentally.  In particular, a forced ROLL row has phase=PLAY_TURN; ROLL is
# an action type, not a prompt, and is handled separately below.
IMPORTANT_PHASES = frozenset(
    {
        "BUILD_INITIAL_ROAD",
        "BUILD_INITIAL_SETTLEMENT",
        "DISCARD",
        "MOVE_ROBBER",
        "PLAY_TURN",
        "INITIAL_BUILD",
        "MAIN_TURN",
        "ROBBER",
    }
)
PLAYER_NAMES = ("BLUE", "RED", "ORANGE", "WHITE")
TARGET_INFORMATION_REGIME_UNKNOWN = "unknown"
SHA256_PREFIX = "sha256:"
SOURCE_MANIFEST_BINDING_SCHEMA = "teacher-source-manifest-binding-v1"
TOOL_PROVENANCE_SCHEMA = "teacher-tool-provenance-v1"


def _action_ids_for_type(action_type: str) -> frozenset[int]:
    """Resolve stable flat-catalog ids instead of guessing from ``phase``."""

    catalog = ActionCatalog(PLAYER_NAMES[:2])
    ids = frozenset(
        index
        for index in range(catalog.size)
        if catalog.describe(index)["action_type"] == action_type
    )
    if len(ids) != 1:
        raise RuntimeError(
            f"flat action catalog must have exactly one {action_type} id; got "
            f"{sorted(ids)}"
        )
    return ids


ROLL_ACTION_IDS = _action_ids_for_type("ROLL")
ROLL_BASE_CATALOG_ACTION_MASK_VERSIONS = frozenset(
    {
        ActionCatalog.version,
        # ColonistMultiAgentEnv allocates ActionCatalog first and appends its
        # trade/negotiation actions after ``_base_action_space_n``.  Base ids,
        # including ROLL, are therefore byte-for-byte the flat-v1 catalog.
        "colonist-multiagent-v1",
    }
)


def _roll_row_mask(
    action: np.ndarray,
    phases: np.ndarray,
    action_mask_versions: np.ndarray,
    *,
    require_supported_version: bool = False,
) -> np.ndarray:
    """Identify ROLL only where the row's action-id schema proves its meaning."""

    normalized_phases = np.char.upper(np.asarray(phases).astype(str))
    versions = np.asarray(action_mask_versions).astype(str)
    if versions.shape != np.asarray(action).shape:
        raise SystemExit(
            "action_mask_version must be row-aligned before decoding ROLL ids: "
            f"versions={versions.shape} actions={np.asarray(action).shape}"
        )
    nonempty_versions = set(versions[versions != ""].tolist())
    unsupported = nonempty_versions - ROLL_BASE_CATALOG_ACTION_MASK_VERSIONS
    if require_supported_version and (unsupported or len(nonempty_versions) > 1):
        raise SystemExit(
            "--roll-keep-prob cannot decode a mixed or unsupported nonempty "
            "action_mask_version set: "
            f"observed={sorted(nonempty_versions)} "
            f"supported={sorted(ROLL_BASE_CATALOG_ACTION_MASK_VERSIONS)}"
        )
    supported_rows = np.isin(
        versions, tuple(ROLL_BASE_CATALOG_ACTION_MASK_VERSIONS)
    )
    id_roll = supported_rows & np.isin(action, tuple(ROLL_ACTION_IDS))
    # Old shards sometimes stored the action type itself as the phase.  That
    # label is self-describing and does not rely on a numeric action schema.
    return id_roll | (normalized_phases == "ROLL")


KEYS = (
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
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and repack raw teacher shards.")
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help="Teacher data directory. Can be passed multiple times.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", choices=("npz", "npz_zst"), default="npz_zst")
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--forced-keep-prob",
        type=float,
        default=0.15,
        help="Probability of keeping one-legal-action samples outside important phases.",
    )
    parser.add_argument(
        "--drop-forced-in-important-phases",
        action="store_true",
        help=(
            "Apply --forced-keep-prob to every phase, including initial_build, "
            "main_turn, robber, and discard. Use for strict 35M BC corpora where "
            "forced moves should not dominate the policy loss."
        ),
    )
    parser.add_argument(
        "--roll-keep-prob",
        type=float,
        default=0.35,
        help="Probability of keeping roll-phase samples.",
    )
    parser.add_argument(
        "--teacher-keep",
        default="",
        help="Optional comma-separated teacher keep probabilities, e.g. random=0.05.",
    )
    parser.add_argument(
        "--drop-truncated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows from truncated/stuck games. Use --no-drop-truncated only for diagnostics.",
    )
    parser.add_argument(
        "--production-35m-teacher",
        action="store_true",
        help=(
            "Fail-closed curation preset for the next 35M BC corpus: remove roll "
            "and one-legal-action rows from policy training, preserve clean ones "
            "for value-only training, and drop truncated games. This prevents "
            "accidental production runs with noisy policy defaults."
        ),
    )
    parser.add_argument(
        "--preserve-value-only-filtered-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep clean forced/roll rows that production policy curation would "
            "otherwise drop, but write policy_weight_multiplier=0 so they only "
            "train value/final-VP heads."
        ),
    )
    parser.add_argument(
        "--dedupe-keys",
        choices=("none", "exact", "state"),
        default="none",
        help=(
            "Drop duplicate rows during curation without touching raw shards. "
            "'exact' uses seed, decision, player, teacher, and action; 'state' "
            "uses only seed, decision, and player."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="Print JSON progress every this many raw rows. Use 0 to disable.",
    )
    args = parser.parse_args()
    if args.production_35m_teacher:
        args.forced_keep_prob = 0.0
        args.drop_forced_in_important_phases = True
        args.roll_keep_prob = 0.0
        args.drop_truncated = True
        args.preserve_value_only_filtered_rows = True
        if args.dedupe_keys == "none":
            args.dedupe_keys = "exact"

    rng = np.random.default_rng(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(output, args.shard_size, args.format)
    teacher_keep = _parse_keep_map(args.teacher_keep)
    input_manifests = _input_manifests(args.data)

    report: dict[str, Any] = {
        "inputs": args.data,
        "format": args.format,
        "forced_keep_prob": args.forced_keep_prob,
        "drop_forced_in_important_phases": bool(args.drop_forced_in_important_phases),
        "roll_keep_prob": args.roll_keep_prob,
        "teacher_keep": teacher_keep,
        "drop_truncated": bool(args.drop_truncated),
        "production_35m_teacher": bool(args.production_35m_teacher),
        "preserve_value_only_filtered_rows": bool(args.preserve_value_only_filtered_rows),
        "dedupe_keys": str(args.dedupe_keys),
        "input_manifests": input_manifests,
        "hard_action_target_information": propagated_hard_action_target_information(
            input_manifests
        ),
        "tool_provenance": _tool_provenance(),
        "raw_samples": 0,
        "kept_samples": 0,
        "dropped_invalid": 0,
        "dropped_forced": 0,
        "dropped_roll": 0,
        "dropped_teacher": 0,
        "dropped_truncated": 0,
        "dropped_duplicate": 0,
        "raw_teachers": Counter(),
        "kept_teachers": Counter(),
        "raw_score_sources": Counter(),
        "kept_score_sources": Counter(),
        "raw_target_information_regimes": Counter(),
        "kept_target_information_regimes": Counter(),
        "raw_phases": Counter(),
        "kept_phases": Counter(),
        "raw_legal_counts": [],
        "kept_legal_counts": [],
        "raw_soft_policy": 0,
        "kept_soft_policy": 0,
        "raw_soft_scores": 0,
        "kept_soft_scores": 0,
        "raw_final_public_vps": 0,
        "kept_final_public_vps": 0,
        "raw_final_actual_vps": 0,
        "kept_final_actual_vps": 0,
        "raw_truncated": 0,
        "kept_truncated": 0,
        "kept_policy_weight_zero": 0,
        "kept_value_weight_zero": 0,
        "kept_policy_weight_positive": 0,
        "kept_value_weight_positive": 0,
        "kept_policy_effective_forced": 0,
        "kept_policy_effective_roll": 0,
        "kept_value_only_samples": 0,
    }
    started = time.perf_counter()
    next_progress = max(1, int(args.progress_every)) if args.progress_every else 0
    seen_dedupe_keys: set[tuple[Any, ...]] = set()

    for data_dir in args.data:
        shard_paths = _shard_files(Path(data_dir))
        print(
            json.dumps(
                {
                    "progress": "curate_input",
                    "data": data_dir,
                    "shards": len(shard_paths),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for shard_number, shard_path in enumerate(shard_paths, start=1):
            shard_started = time.perf_counter()
            shard = _materialize_shard(_load_npz(shard_path))
            n = int(len(shard["action_taken"]))
            raw_before = int(report["raw_samples"])
            kept_before = int(report["kept_samples"])
            keep_mask, shard_report, policy_weight_multiplier, value_weight_multiplier = _curate_shard_mask(
                shard,
                rng=rng,
                teacher_keep=teacher_keep,
                forced_keep_prob=float(args.forced_keep_prob),
                drop_forced_in_important_phases=bool(args.drop_forced_in_important_phases),
                roll_keep_prob=float(args.roll_keep_prob),
                drop_truncated=bool(args.drop_truncated),
                preserve_value_only_filtered_rows=bool(args.preserve_value_only_filtered_rows),
            )
            target_information_regimes = np.asarray(
                shard.get(
                    "target_information_regime",
                    np.full(n, TARGET_INFORMATION_REGIME_UNKNOWN),
                )
            ).astype(str)
            report["raw_target_information_regimes"].update(
                target_information_regimes.tolist()
            )
            if args.dedupe_keys != "none":
                dedupe_keep, dropped_duplicate = _dedupe_keep_mask(
                    shard,
                    keep_mask,
                    seen=seen_dedupe_keys,
                    mode=str(args.dedupe_keys),
                )
                duplicate_mask = keep_mask & ~dedupe_keep
                keep_mask &= dedupe_keep
                policy_weight_multiplier[duplicate_mask] = 0.0
                value_weight_multiplier[duplicate_mask] = 0.0
                shard_report = _subtract_dropped_duplicates(
                    shard_report,
                    shard,
                    duplicate_mask=duplicate_mask,
                    final_keep_mask=keep_mask,
                    policy_weight_multiplier=policy_weight_multiplier,
                    value_weight_multiplier=value_weight_multiplier,
                )
                shard_report["dropped_duplicate"] = int(dropped_duplicate)
            else:
                shard_report["dropped_duplicate"] = 0
            report["kept_target_information_regimes"].update(
                target_information_regimes[keep_mask].tolist()
            )
            shard["policy_weight_multiplier"] = policy_weight_multiplier
            shard["value_weight_multiplier"] = value_weight_multiplier
            _merge_shard_report(report, shard_report)
            writer.add_batch(_slice_shard(shard, keep_mask))
            if next_progress:
                while int(report["raw_samples"]) >= next_progress:
                    print(
                        json.dumps(
                            {
                                "progress": "curate_rows",
                                "raw_samples": int(report["raw_samples"]),
                                "kept_samples": int(report["kept_samples"]),
                                "dropped_invalid": int(report["dropped_invalid"]),
                                "dropped_forced": int(report["dropped_forced"]),
                                "dropped_roll": int(report["dropped_roll"]),
                                "dropped_duplicate": int(report["dropped_duplicate"]),
                                "elapsed_sec": time.perf_counter() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    next_progress += max(1, int(args.progress_every))
            print(
                json.dumps(
                    {
                        "progress": "curate_shard_done",
                        "data": data_dir,
                        "shard": str(shard_path),
                        "shard_index": shard_number,
                        "shards_total": len(shard_paths),
                        "raw_rows": int(report["raw_samples"]) - raw_before,
                        "kept_rows": int(report["kept_samples"]) - kept_before,
                        "elapsed_sec": time.perf_counter() - shard_started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    shards = writer.close()
    final_input_manifests = _input_manifests(args.data)
    manifest = {
        "inputs": args.data,
        "shards": [str(path) for path in shards],
        "samples": int(report["kept_samples"]),
        "format": args.format,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "forced_keep_prob": args.forced_keep_prob,
        "drop_forced_in_important_phases": bool(args.drop_forced_in_important_phases),
        "roll_keep_prob": args.roll_keep_prob,
        "teacher_keep": teacher_keep,
        "drop_truncated": bool(args.drop_truncated),
        "production_35m_teacher": bool(args.production_35m_teacher),
        "preserve_value_only_filtered_rows": bool(args.preserve_value_only_filtered_rows),
        "dedupe_keys": str(args.dedupe_keys),
        "input_manifests": final_input_manifests,
        "hard_action_target_information": propagated_hard_action_target_information(
            final_input_manifests
        ),
        "tool_provenance": _tool_provenance(),
        "score_source_counts": dict(report["kept_score_sources"].most_common()),
        "target_information_regime_counts": dict(
            report["kept_target_information_regimes"].most_common()
        ),
        "raw_samples": int(report["raw_samples"]),
        "kept_fraction": (
            float(report["kept_samples"]) / float(report["raw_samples"])
            if report["raw_samples"]
            else 0.0
        ),
    }
    write_json(output / "manifest.json", manifest)

    final_report = {
        **{
            key: value
            for key, value in report.items()
            if key not in {
                "raw_teachers",
                "kept_teachers",
                "raw_score_sources",
                "kept_score_sources",
                "raw_target_information_regimes",
                "kept_target_information_regimes",
                "raw_phases",
                "kept_phases",
                "raw_legal_counts",
                "kept_legal_counts",
            }
        },
        "raw_teachers": dict(report["raw_teachers"].most_common()),
        "kept_teachers": dict(report["kept_teachers"].most_common()),
        "raw_score_sources": dict(report["raw_score_sources"].most_common()),
        "kept_score_sources": dict(report["kept_score_sources"].most_common()),
        "raw_target_information_regimes": dict(
            report["raw_target_information_regimes"].most_common()
        ),
        "kept_target_information_regimes": dict(
            report["kept_target_information_regimes"].most_common()
        ),
        "raw_phases": dict(report["raw_phases"].most_common()),
        "kept_phases": dict(report["kept_phases"].most_common()),
        "raw_legal_actions": _legal_stats(np.asarray(report["raw_legal_counts"], dtype=np.int64)),
        "kept_legal_actions": _legal_stats(np.asarray(report["kept_legal_counts"], dtype=np.int64)),
        "raw_soft_policy_fraction": (
            float(report["raw_soft_policy"]) / float(report["raw_samples"])
            if report["raw_samples"]
            else 0.0
        ),
        "kept_soft_policy_fraction": (
            float(report["kept_soft_policy"]) / float(report["kept_samples"])
            if report["kept_samples"]
            else 0.0
        ),
        "raw_soft_score_fraction": (
            float(report["raw_soft_scores"]) / float(report["raw_samples"])
            if report["raw_samples"]
            else 0.0
        ),
        "kept_soft_score_fraction": (
            float(report["kept_soft_scores"]) / float(report["kept_samples"])
            if report["kept_samples"]
            else 0.0
        ),
        "raw_final_public_vp_fraction": (
            float(report["raw_final_public_vps"]) / float(report["raw_samples"])
            if report["raw_samples"]
            else 0.0
        ),
        "kept_final_public_vp_fraction": (
            float(report["kept_final_public_vps"]) / float(report["kept_samples"])
            if report["kept_samples"]
            else 0.0
        ),
        "raw_final_actual_vp_fraction": (
            float(report["raw_final_actual_vps"]) / float(report["raw_samples"])
            if report["raw_samples"]
            else 0.0
        ),
        "kept_final_actual_vp_fraction": (
            float(report["kept_final_actual_vps"]) / float(report["kept_samples"])
            if report["kept_samples"]
            else 0.0
        ),
        "kept_truncated_fraction": (
            float(report["kept_truncated"]) / float(report["kept_samples"])
            if report["kept_samples"]
            else 0.0
        ),
        "kept_policy_weight_zero_fraction": (
            float(report["kept_policy_weight_zero"]) / float(report["kept_samples"])
            if report["kept_samples"]
            else 0.0
        ),
        "kept_policy_weight_positive_fraction": (
            float(report["kept_policy_weight_positive"]) / float(report["kept_samples"])
            if report["kept_samples"]
            else 0.0
        ),
        "kept_policy_effective_forced_fraction": (
            float(report["kept_policy_effective_forced"])
            / float(max(report["kept_policy_weight_positive"], 1))
        ),
        "kept_policy_effective_roll_fraction": (
            float(report["kept_policy_effective_roll"])
            / float(max(report["kept_policy_weight_positive"], 1))
        ),
        "kept_value_only_samples": int(report["kept_value_only_samples"]),
        "dropped_duplicate": int(report["dropped_duplicate"]),
        "kept_value_only_fraction": (
            float(report["kept_value_only_samples"]) / float(report["kept_samples"])
            if report["kept_samples"]
            else 0.0
        ),
        "kept_fraction": (
            float(report["kept_samples"]) / float(report["raw_samples"])
            if report["raw_samples"]
            else 0.0
        ),
        "shards": [str(path) for path in shards],
    }
    write_json(output / "curation_report.json", final_report)
    print(json.dumps(final_report, indent=2, sort_keys=True))


class ShardWriter:
    def __init__(self, output: Path, shard_size: int, fmt: str) -> None:
        self.output = output
        self.shard_size = max(1, int(shard_size))
        self.format = fmt
        self.rows: list[dict[str, Any]] = []
        self.pending: dict[str, np.ndarray] | None = None
        self.pending_count = 0
        self.index = 0
        self.paths: list[Path] = []

    def add_row(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def add_batch(self, batch: dict[str, np.ndarray]) -> None:
        if self.rows:
            self.flush()
        n = int(len(batch.get("action_taken", ())))
        if n == 0:
            return
        start = 0
        while start < n:
            space = max(1, self.shard_size - self.pending_count)
            end = min(n, start + space)
            piece = _slice_shard(batch, np.arange(start, end, dtype=np.int64))
            self._append_pending(piece)
            if self.pending_count >= self.shard_size:
                self._flush_pending()
            start = end

    def close(self) -> list[Path]:
        self.flush()
        return self.paths

    def flush(self) -> None:
        self._flush_pending()
        if not self.rows:
            return
        max_valid = max(len(row["legal_action_ids"]) for row in self.rows)
        context_size = int(self.rows[0]["legal_action_context"].shape[-1])
        obs = np.stack([row["obs"].astype(np.float16, copy=False) for row in self.rows], axis=0)
        legal = np.full((len(self.rows), max_valid), -1, dtype=np.int16)
        context = np.zeros((len(self.rows), max_valid, context_size), dtype=np.float16)
        target_policy = np.zeros((len(self.rows), max_valid), dtype=np.float16)
        target_scores = np.full((len(self.rows), max_valid), np.nan, dtype=np.float32)
        target_policy_mask = np.zeros((len(self.rows), max_valid), dtype=np.bool_)
        target_scores_mask = np.zeros((len(self.rows), max_valid), dtype=np.bool_)
        for idx, row in enumerate(self.rows):
            valid = row["legal_action_ids"].astype(np.int16, copy=False)
            count = len(valid)
            legal[idx, :count] = valid
            context[idx, :count, :] = row["legal_action_context"].astype(np.float16, copy=False)
            if "target_policy" in row:
                target_policy[idx, :count] = row["target_policy"].astype(np.float16, copy=False)
            if "target_scores" in row:
                target_scores[idx, :count] = row["target_scores"].astype(np.float32, copy=False)
            if "target_policy_mask" in row:
                target_policy_mask[idx, :count] = row["target_policy_mask"].astype(np.bool_, copy=False)
            else:
                policy = np.asarray(row.get("target_policy", ()), dtype=np.float32)
                if policy.shape[:1] == (count,):
                    target_policy_mask[idx, :count] = policy > 0.0
            if "target_scores_mask" in row:
                target_scores_mask[idx, :count] = row["target_scores_mask"].astype(np.bool_, copy=False)
            else:
                scores = np.asarray(row.get("target_scores", ()), dtype=np.float32)
                if scores.shape[:1] == (count,):
                    target_scores_mask[idx, :count] = np.isfinite(scores)

        arrays = {
            "obs": obs,
            "legal_action_ids": legal,
            "legal_action_context": context,
            "action_taken": np.asarray([row["action_taken"] for row in self.rows], dtype=np.int16),
            "target_policy": target_policy,
            "target_scores": target_scores,
            "target_policy_mask": target_policy_mask,
            "target_scores_mask": target_scores_mask,
            "target_score_source": np.asarray(
                [str(row.get("target_score_source", "")) for row in self.rows]
            ),
            "target_information_regime": np.asarray(
                [
                    str(
                        row.get(
                            "target_information_regime",
                            TARGET_INFORMATION_REGIME_UNKNOWN,
                        )
                    )
                    for row in self.rows
                ]
            ),
            "game_seed": np.asarray([row.get("game_seed", 0) for row in self.rows], dtype=np.int64),
            "teacher_name": np.asarray([str(row.get("teacher_name", "")) for row in self.rows]),
            "player": np.asarray([str(row.get("player", "")) for row in self.rows]),
            "seat": np.asarray([row.get("seat", -1) for row in self.rows], dtype=np.int8),
            "phase": np.asarray([str(row.get("phase", "")) for row in self.rows]),
            "decision_index": np.asarray([row.get("decision_index", -1) for row in self.rows], dtype=np.int32),
            "winner": np.asarray([str(row.get("winner", "")) for row in self.rows]),
            "terminated": np.asarray([bool(row.get("terminated", True)) for row in self.rows], dtype=np.bool_),
            "truncated": np.asarray([bool(row.get("truncated", False)) for row in self.rows], dtype=np.bool_),
            "final_public_vps": np.stack(
                [
                    row.get("final_public_vps", np.zeros(len(PLAYER_NAMES), dtype=np.int16)).astype(np.int16, copy=False)
                    for row in self.rows
                ],
                axis=0,
            ),
            "has_final_public_vps": np.asarray(
                [
                    bool(row.get("has_final_public_vps", False))
                    for row in self.rows
                ],
                dtype=np.bool_,
            ),
            "final_actual_vps": np.stack(
                [
                    row.get("final_actual_vps", np.zeros(len(PLAYER_NAMES), dtype=np.int16)).astype(np.int16, copy=False)
                    for row in self.rows
                ],
                axis=0,
            ),
            "has_final_actual_vps": np.asarray(
                [
                    bool(row.get("has_final_actual_vps", False))
                    for row in self.rows
                ],
                dtype=np.bool_,
            ),
            "action_mask_version": np.asarray(
                [str(row.get("action_mask_version", "")) for row in self.rows]
            ),
            "policy_weight_multiplier": np.asarray(
                [float(row.get("policy_weight_multiplier", 1.0)) for row in self.rows],
                dtype=np.float32,
            ),
            "value_weight_multiplier": np.asarray(
                [float(row.get("value_weight_multiplier", 1.0)) for row in self.rows],
                dtype=np.float32,
            ),
        }
        self._write_arrays(arrays)
        self.rows = []

    def _append_pending(self, batch: dict[str, np.ndarray]) -> None:
        n = int(len(batch.get("action_taken", ())))
        if n == 0:
            return
        if self.pending is None:
            self.pending = {key: np.asarray(value) for key, value in batch.items()}
            self.pending_count = n
            return
        self.pending = _concat_batches(self.pending, batch)
        self.pending_count += n

    def _flush_pending(self) -> None:
        if self.pending is None or self.pending_count <= 0:
            return
        self._write_arrays(self.pending)
        self.pending = None
        self.pending_count = 0

    def _write_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        arrays = _trim_legal_width(arrays)
        path = self.output / f"teacher_shard_{self.index:05d}.npz"
        np.savez_compressed(path, **arrays)
        if self.format == "npz_zst":
            path = _try_zstd(path)
        self.paths.append(path)
        self.index += 1


def _concat_batches(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    left_n = _batch_len(left)
    right_n = _batch_len(right)
    ordered_keys = [
        key
        for key in KEYS
        if key in left or key in right
    ]
    ordered_keys.extend(
        sorted((set(left) | set(right)) - set(ordered_keys))
    )
    out = {}
    for key in ordered_keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            left_value = _default_array_for_missing_key(key, right_value, left_n, source=left)
        if right_value is None:
            right_value = _default_array_for_missing_key(key, left_value, right_n, source=right)
        out[key] = _concat_array(key, left_value, right_value)
    return out


def _batch_len(batch: dict[str, np.ndarray]) -> int:
    if "action_taken" in batch:
        return int(len(batch["action_taken"]))
    for value in batch.values():
        array = np.asarray(value)
        if array.ndim > 0:
            return int(array.shape[0])
    return 0


def _default_array_for_missing_key(
    key: str,
    template: np.ndarray,
    n: int,
    *,
    source: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    shape = list(np.asarray(template).shape)
    if not shape:
        shape = [n]
    else:
        shape[0] = n
    dtype = np.asarray(template).dtype
    source = source or {}
    if key == "target_policy_mask" and "target_policy" in source:
        policy = np.asarray(source["target_policy"], dtype=np.float32)
        return np.where(np.isfinite(policy), policy, 0.0) > 0.0
    if key == "target_scores_mask" and "target_scores" in source:
        return np.isfinite(np.asarray(source["target_scores"], dtype=np.float32))
    if key == "legal_action_ids":
        return np.full(shape, -1, dtype=dtype)
    if key == "target_scores":
        return np.full(shape, np.nan, dtype=dtype)
    if key in {"target_policy_mask", "target_scores_mask", "has_final_public_vps", "has_final_actual_vps"}:
        return np.zeros(shape, dtype=np.bool_)
    if key == "target_information_regime":
        return np.full(shape, TARGET_INFORMATION_REGIME_UNKNOWN, dtype=dtype)
    if key in {"target_score_source", "teacher_name", "player", "phase", "winner", "action_mask_version"}:
        return np.full(shape, "", dtype=dtype)
    if key in {"seat", "decision_index"}:
        return np.full(shape, -1, dtype=dtype)
    if key in {"terminated", "truncated"}:
        return np.zeros(shape, dtype=np.bool_)
    if key in {"policy_weight_multiplier", "value_weight_multiplier"}:
        return np.ones(shape, dtype=np.float32)
    return np.zeros(shape, dtype=dtype)


def _concat_array(key: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if key in {
        "legal_action_ids",
        "target_policy",
        "target_scores",
        "target_policy_mask",
        "target_scores_mask",
    }:
        width = max(int(left.shape[1]), int(right.shape[1]))
        fill: Any
        if key == "legal_action_ids":
            fill = -1
        elif key == "target_scores":
            fill = np.nan
        elif key in {"target_policy_mask", "target_scores_mask"}:
            fill = False
        else:
            fill = 0.0
        return np.concatenate(
            (_pad_axis1(left, width, fill), _pad_axis1(right, width, fill)),
            axis=0,
        )
    if key == "legal_action_context":
        width = max(int(left.shape[1]), int(right.shape[1]))
        return np.concatenate(
            (_pad_axis1(left, width, 0.0), _pad_axis1(right, width, 0.0)),
            axis=0,
        )
    return np.concatenate((left, right), axis=0)


def _pad_axis1(value: np.ndarray, width: int, fill: Any) -> np.ndarray:
    if int(value.shape[1]) == width:
        return value
    shape = list(value.shape)
    shape[1] = width
    out = np.full(shape, fill, dtype=value.dtype)
    slices = [slice(None)] * value.ndim
    slices[1] = slice(0, value.shape[1])
    out[tuple(slices)] = value
    return out


def _row(shard: Any, index: int) -> dict[str, Any]:
    row = {}
    for key in KEYS:
        if key in shard:
            row[key] = shard[key][index]
    return row


def _materialize_shard(shard: Any) -> dict[str, np.ndarray]:
    return {key: np.asarray(shard[key]) for key in KEYS if key in shard}


def _curate_shard_mask(
    shard: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    teacher_keep: dict[str, float],
    forced_keep_prob: float,
    drop_forced_in_important_phases: bool,
    roll_keep_prob: float,
    drop_truncated: bool,
    preserve_value_only_filtered_rows: bool,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    action = np.asarray(shard["action_taken"], dtype=np.int16)
    legal = np.asarray(shard["legal_action_ids"], dtype=np.int16)
    n = int(len(action))
    legal_counts = np.sum(legal >= 0, axis=1)
    valid_action = np.any(legal == action[:, None], axis=1)
    teachers = np.asarray(shard.get("teacher_name", np.full(n, "", dtype="<U1"))).astype(str)
    score_sources = np.asarray(shard.get("target_score_source", np.full(n, "", dtype="<U1"))).astype(str)
    score_source_labels = np.where(score_sources == "", "none", score_sources)
    phases = np.asarray(shard.get("phase", np.full(n, "", dtype="<U1"))).astype(str)
    action_mask_versions = np.asarray(
        shard.get("action_mask_version", np.full(n, "", dtype="<U1"))
    ).astype(str)
    phase_labels = np.where(phases == "", "unknown", phases)
    target_policy = np.asarray(shard.get("target_policy", np.zeros((n, 0))), dtype=np.float32)
    target_scores = np.asarray(shard.get("target_scores", np.full((n, 0), np.nan)), dtype=np.float32)
    truncated = np.asarray(shard.get("truncated", np.zeros(n, dtype=np.bool_)), dtype=np.bool_)
    has_final_public_vps = np.asarray(
        shard.get("has_final_public_vps", np.zeros(n, dtype=np.bool_)),
        dtype=np.bool_,
    )
    has_final_actual_vps = np.asarray(
        shard.get("has_final_actual_vps", np.zeros(n, dtype=np.bool_)),
        dtype=np.bool_,
    )
    winners = np.asarray(shard.get("winner", np.full(n, "", dtype="<U1"))).astype(str)

    keep = valid_action.copy()
    truncated_drop = truncated & bool(drop_truncated)
    keep &= ~truncated_drop
    teacher_keep_prob = np.ones(n, dtype=np.float64)
    if teacher_keep:
        for teacher, probability in teacher_keep.items():
            teacher_keep_prob[teachers == teacher] = float(probability)
        teacher_drop = rng.random(n) > teacher_keep_prob
        keep &= ~teacher_drop
    else:
        teacher_drop = np.zeros(n, dtype=np.bool_)

    # Current shards label a ROLL decision with the public prompt PLAY_TURN.
    # The former ``phases == 'roll'`` predicate therefore selected zero rows
    # and made --roll-keep-prob a silent no-op.  Retain the legacy phase alias,
    # but use the recorded action id for the production schema.
    normalized_phases = np.char.upper(phases)
    is_roll = _roll_row_mask(
        action,
        phases,
        action_mask_versions,
        require_supported_version=float(roll_keep_prob) < 1.0,
    )
    roll_drop = is_roll & (rng.random(n) > float(roll_keep_prob))
    important = np.isin(normalized_phases, tuple(IMPORTANT_PHASES))
    protect_forced = important & (not bool(drop_forced_in_important_phases))
    forced_drop = (
        (legal_counts <= 1)
        & ~protect_forced
        & (rng.random(n) > float(forced_keep_prob))
    )
    policy_filtered = roll_drop | forced_drop
    policy_keep = keep & ~policy_filtered
    has_clean_value_target = (
        (winners != "")
        & ~truncated
        & (has_final_actual_vps | has_final_public_vps)
    )
    value_only_keep = (
        keep
        & policy_filtered
        & has_clean_value_target
        & bool(preserve_value_only_filtered_rows)
    )
    keep = policy_keep | value_only_keep
    policy_weight_multiplier = np.zeros(n, dtype=np.float32)
    policy_weight_multiplier[policy_keep] = 1.0
    value_weight_multiplier = np.zeros(n, dtype=np.float32)
    value_weight_multiplier[keep & has_clean_value_target] = 1.0

    has_policy = np.sum(np.where(np.isfinite(target_policy), np.maximum(target_policy, 0.0), 0.0), axis=1) > 0.0
    has_scores = np.isfinite(target_scores).any(axis=1)
    kept = keep
    kept_policy_positive = kept & (policy_weight_multiplier > 0.0)
    kept_value_positive = kept & (value_weight_multiplier > 0.0)
    return keep, {
        "raw_samples": n,
        "kept_samples": int(np.sum(kept)),
        "dropped_invalid": int(np.sum(~valid_action)),
        "dropped_forced": int(np.sum(valid_action & forced_drop & ~value_only_keep)),
        "dropped_roll": int(np.sum(valid_action & ~teacher_drop & roll_drop & ~value_only_keep)),
        "dropped_teacher": int(np.sum(valid_action & teacher_drop)),
        "dropped_truncated": int(np.sum(valid_action & truncated_drop)),
        "raw_teachers": _counter_from_values(teachers),
        "kept_teachers": _counter_from_values(teachers[kept]) if np.any(kept) else Counter(),
        "raw_score_sources": _counter_from_values(score_source_labels),
        "kept_score_sources": _counter_from_values(score_source_labels[kept]) if np.any(kept) else Counter(),
        "raw_phases": _counter_from_values(phase_labels),
        "kept_phases": _counter_from_values(phase_labels[kept]) if np.any(kept) else Counter(),
        "raw_legal_counts": legal_counts.astype(np.int16, copy=False),
        "kept_legal_counts": legal_counts[kept].astype(np.int16, copy=False),
        "raw_soft_policy": int(np.sum(has_policy)),
        "kept_soft_policy": int(np.sum(has_policy[kept])),
        "raw_soft_scores": int(np.sum(has_scores)),
        "kept_soft_scores": int(np.sum(has_scores[kept])),
        "raw_final_public_vps": int(np.sum(has_final_public_vps)),
        "kept_final_public_vps": int(np.sum(has_final_public_vps[kept])),
        "raw_final_actual_vps": int(np.sum(has_final_actual_vps)),
        "kept_final_actual_vps": int(np.sum(has_final_actual_vps[kept])),
        "raw_truncated": int(np.sum(truncated)),
        "kept_truncated": int(np.sum(truncated[kept])),
        "kept_policy_weight_zero": int(np.sum(kept & (policy_weight_multiplier <= 0.0))),
        "kept_value_weight_zero": int(np.sum(kept & (value_weight_multiplier <= 0.0))),
        "kept_policy_weight_positive": int(np.sum(kept_policy_positive)),
        "kept_value_weight_positive": int(np.sum(kept_value_positive)),
        "kept_policy_effective_forced": int(np.sum(kept_policy_positive & (legal_counts <= 1))),
        "kept_policy_effective_roll": int(np.sum(kept_policy_positive & is_roll)),
        "kept_value_only_samples": int(np.sum(kept & (policy_weight_multiplier <= 0.0) & (value_weight_multiplier > 0.0))),
    }, policy_weight_multiplier, value_weight_multiplier


def _merge_shard_report(report: dict[str, Any], shard_report: dict[str, Any]) -> None:
    for key in (
        "raw_samples",
        "kept_samples",
        "dropped_invalid",
        "dropped_forced",
        "dropped_roll",
        "dropped_teacher",
        "dropped_truncated",
        "dropped_duplicate",
        "raw_soft_policy",
        "kept_soft_policy",
        "raw_soft_scores",
        "kept_soft_scores",
        "raw_final_public_vps",
        "kept_final_public_vps",
        "raw_final_actual_vps",
        "kept_final_actual_vps",
        "raw_truncated",
        "kept_truncated",
        "kept_policy_weight_zero",
        "kept_value_weight_zero",
        "kept_policy_weight_positive",
        "kept_value_weight_positive",
        "kept_policy_effective_forced",
        "kept_policy_effective_roll",
        "kept_value_only_samples",
    ):
        report[key] += int(shard_report[key])
    report["raw_teachers"].update(shard_report["raw_teachers"])
    report["kept_teachers"].update(shard_report["kept_teachers"])
    report["raw_score_sources"].update(shard_report["raw_score_sources"])
    report["kept_score_sources"].update(shard_report["kept_score_sources"])
    report["raw_phases"].update(shard_report["raw_phases"])
    report["kept_phases"].update(shard_report["kept_phases"])
    report["raw_legal_counts"].extend(np.asarray(shard_report["raw_legal_counts"], dtype=np.int16).tolist())
    report["kept_legal_counts"].extend(np.asarray(shard_report["kept_legal_counts"], dtype=np.int16).tolist())


def _dedupe_keep_mask(
    shard: dict[str, np.ndarray],
    keep_mask: np.ndarray,
    *,
    seen: set[tuple[Any, ...]],
    mode: str,
) -> tuple[np.ndarray, int]:
    if mode not in {"exact", "state"}:
        raise ValueError(f"unsupported dedupe mode: {mode}")
    n = int(len(shard["action_taken"]))
    out = np.zeros(n, dtype=np.bool_)
    game_seed = np.asarray(shard.get("game_seed", np.arange(n, dtype=np.int64)), dtype=np.int64)
    decision_index = np.asarray(shard.get("decision_index", np.arange(n, dtype=np.int32)), dtype=np.int32)
    player = np.asarray(shard.get("player", np.full(n, "", dtype="<U1"))).astype(str)
    teacher = np.asarray(shard.get("teacher_name", np.full(n, "", dtype="<U1"))).astype(str)
    action = np.asarray(shard["action_taken"], dtype=np.int16)
    dropped = 0
    for idx in range(n):
        if not bool(keep_mask[idx]):
            continue
        if mode == "state":
            key = (int(game_seed[idx]), int(decision_index[idx]), str(player[idx]))
        else:
            key = (
                int(game_seed[idx]),
                int(decision_index[idx]),
                str(player[idx]),
                str(teacher[idx]),
                int(action[idx]),
            )
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out[idx] = True
    return out, dropped


def _subtract_dropped_duplicates(
    shard_report: dict[str, Any],
    shard: dict[str, np.ndarray],
    *,
    duplicate_mask: np.ndarray,
    final_keep_mask: np.ndarray,
    policy_weight_multiplier: np.ndarray,
    value_weight_multiplier: np.ndarray,
) -> dict[str, Any]:
    duplicate_mask = np.asarray(duplicate_mask, dtype=np.bool_)
    if not np.any(duplicate_mask):
        return dict(shard_report)
    final_keep_mask = np.asarray(final_keep_mask, dtype=np.bool_)
    out = dict(shard_report)
    n = int(len(duplicate_mask))
    teachers = np.asarray(shard.get("teacher_name", np.full(n, "", dtype="<U1"))).astype(str)
    score_sources = np.asarray(shard.get("target_score_source", np.full(n, "", dtype="<U1"))).astype(str)
    score_source_labels = np.where(score_sources == "", "none", score_sources)
    phases = np.asarray(shard.get("phase", np.full(n, "", dtype="<U1"))).astype(str)
    action = np.asarray(shard["action_taken"], dtype=np.int16)
    action_mask_versions = np.asarray(
        shard.get("action_mask_version", np.full(n, "", dtype="<U1"))
    ).astype(str)
    is_roll = _roll_row_mask(action, phases, action_mask_versions)
    phase_labels = np.where(phases == "", "unknown", phases)
    legal = np.asarray(shard["legal_action_ids"], dtype=np.int16)
    legal_counts = np.sum(legal >= 0, axis=1)
    target_policy = np.asarray(shard.get("target_policy", np.zeros((n, 0))), dtype=np.float32)
    target_scores = np.asarray(shard.get("target_scores", np.full((n, 0), np.nan)), dtype=np.float32)
    has_final_public_vps = np.asarray(
        shard.get("has_final_public_vps", np.zeros(n, dtype=np.bool_)),
        dtype=np.bool_,
    )
    has_final_actual_vps = np.asarray(
        shard.get("has_final_actual_vps", np.zeros(n, dtype=np.bool_)),
        dtype=np.bool_,
    )
    truncated = np.asarray(shard.get("truncated", np.zeros(n, dtype=np.bool_)), dtype=np.bool_)
    policy_weight = np.asarray(policy_weight_multiplier, dtype=np.float32)
    value_weight = np.asarray(value_weight_multiplier, dtype=np.float32)
    has_policy = (
        np.sum(np.where(np.isfinite(target_policy), np.maximum(target_policy, 0.0), 0.0), axis=1)
        > 0.0
    )
    has_scores = np.isfinite(target_scores).any(axis=1)
    kept = final_keep_mask
    out["kept_samples"] = int(np.sum(kept))
    out["kept_teachers"] = _counter_from_values(teachers[kept]) if np.any(kept) else Counter()
    out["kept_score_sources"] = _counter_from_values(score_source_labels[kept]) if np.any(kept) else Counter()
    out["kept_phases"] = _counter_from_values(phase_labels[kept]) if np.any(kept) else Counter()
    out["kept_legal_counts"] = legal_counts[kept].astype(np.int16, copy=False)
    out["kept_soft_policy"] = int(np.sum(has_policy[kept]))
    out["kept_soft_scores"] = int(np.sum(has_scores[kept]))
    out["kept_final_public_vps"] = int(np.sum(has_final_public_vps[kept]))
    out["kept_final_actual_vps"] = int(np.sum(has_final_actual_vps[kept]))
    out["kept_truncated"] = int(np.sum(truncated[kept]))
    out["kept_policy_weight_zero"] = int(np.sum(kept & (policy_weight <= 0.0)))
    out["kept_value_weight_zero"] = int(np.sum(kept & (value_weight <= 0.0)))
    out["kept_policy_weight_positive"] = int(np.sum(kept & (policy_weight > 0.0)))
    out["kept_value_weight_positive"] = int(np.sum(kept & (value_weight > 0.0)))
    out["kept_policy_effective_forced"] = int(np.sum(kept & (policy_weight > 0.0) & (legal_counts <= 1)))
    out["kept_policy_effective_roll"] = int(
        np.sum(kept & (policy_weight > 0.0) & is_roll)
    )
    out["kept_value_only_samples"] = int(
        np.sum(
            kept
            & (policy_weight <= 0.0)
            & (value_weight > 0.0)
        )
    )
    return out


def _counter_from_values(values: np.ndarray) -> Counter[str]:
    if values.size == 0:
        return Counter()
    keys, counts = np.unique(values.astype(str), return_counts=True)
    return Counter({str(key): int(count) for key, count in zip(keys, counts)})


def _slice_shard(shard: dict[str, np.ndarray], selector: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[selector] for key, value in shard.items()}


def _trim_legal_width(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    legal = np.asarray(arrays["legal_action_ids"], dtype=np.int16)
    if legal.ndim != 2 or legal.shape[0] == 0:
        return arrays
    legal_counts = np.sum(legal >= 0, axis=1)
    width = max(1, int(np.max(legal_counts)) if legal_counts.size else 1)
    if width >= legal.shape[1]:
        return arrays
    trimmed = dict(arrays)
    trimmed["legal_action_ids"] = legal[:, :width]
    for key in ("legal_action_context", "target_policy", "target_scores", "target_policy_mask", "target_scores_mask"):
        if key in trimmed:
            trimmed[key] = np.asarray(trimmed[key])[:, :width, ...]
    return trimmed


def _shard_files(path: Path) -> list[Path]:
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        files = _manifest_shard_files(manifest_path)
        if files:
            return files

    if (path / "manifest.partial.json").exists():
        files = sorted(path.glob("*.npz")) + sorted(path.glob("*.npz.zst"))
        if files:
            return files
        raise SystemExit(
            f"{path} looks like a partial Modal/raw teacher part without shard files; "
            "refusing recursive shard glob."
        )

    if (path / "parts").exists():
        raise SystemExit(
            f"{path} looks like a partial Modal/raw teacher root without a completed "
            "top-level manifest.json; refusing recursive shard glob. Summarize or "
            "finish the run before curation."
        )

    files = sorted(path.glob("*.npz")) + sorted(path.glob("*.npz.zst"))
    if files:
        return files

    child_manifests = sorted(
        candidate
        for candidate in path.glob("**/manifest.json")
        if candidate.parent != path
    )
    if len(child_manifests) == 1:
        files = _manifest_shard_files(child_manifests[0])
        if files:
            return files
    if len(child_manifests) > 1:
        files = _modal_part_manifest_shards(path, child_manifests)
        if files:
            return files
        previews = ", ".join(str(candidate.parent) for candidate in child_manifests[:5])
        raise SystemExit(
            f"{path} contains multiple nested teacher manifests; pass one raw/curated "
            f"leaf directory instead of recursively mixing runs. Examples: {previews}"
        )

    files = sorted(path.glob("**/*.npz")) + sorted(path.glob("**/*.npz.zst"))
    if files:
        return files
    raise SystemExit(f"no teacher shards found in {path}")


def _modal_part_manifest_shards(path: Path, manifests: list[Path]) -> list[Path]:
    files: list[Path] = []
    for manifest in manifests:
        try:
            relative = manifest.relative_to(path)
        except ValueError:
            return []
        parts = relative.parts
        if len(parts) != 3 or parts[0] != "parts" or not parts[1].startswith("part_"):
            return []
        files.extend(_manifest_shard_files(manifest))
    return sorted(files)


def _manifest_shard_files(manifest_path: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: list[Path] = []
    missing: list[str] = []
    for value in manifest.get("shards", ()):
        raw = Path(value)
        candidates = [raw] if raw.is_absolute() else [raw, manifest_path.parent / raw]
        if raw.is_absolute():
            candidates.append(manifest_path.parent / raw.name)
        chosen = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        files.append(chosen)
        if not chosen.exists():
            missing.append(str(chosen))
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            f"{manifest_path} points to missing teacher shards. "
            f"First missing paths: {preview}"
        )
    return files


def _load_npz(path: Path):
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as error:
            raise SystemExit("zstandard is required to read .npz.zst shards") from error
        data = zstd.ZstdDecompressor().decompress(path.read_bytes())
        return np.load(io.BytesIO(data), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def _try_zstd(path: Path) -> Path:
    try:
        import zstandard as zstd
    except ImportError:
        return path
    compressed = path.with_suffix(path.suffix + ".zst")
    compressed.write_bytes(zstd.ZstdCompressor(level=3).compress(path.read_bytes()))
    path.unlink()
    return compressed


def _parse_keep_map(raw: str) -> dict[str, float]:
    result = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid keep entry: {item}")
        name, value = item.split("=", 1)
        result[name.strip()] = float(value)
    return result


def _input_manifests(data_dirs: list[str]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for raw in data_dirs:
        path = Path(raw)
        payload: dict[str, Any] = {"path": str(path)}
        found_direct = False
        for candidate in (path / "manifest.json", path / "manifest.partial.json", path / "curation_report.json"):
            if not candidate.exists():
                continue
            found_direct = True
            loaded, binding = _load_bound_manifest(candidate)
            compact = {
                key: loaded.get(key)
                for key in (
                    "track",
                    "vps_to_win",
                    "teachers",
                    "games",
                    "samples",
                    "raw_samples",
                    "kept_fraction",
                    "seed",
                    "mixed_seats",
                    "mixed_seat_mode",
                    "graph_history_features",
                    "drop_forced_in_important_phases",
                    "forced_action_fraction",
                    "soft_policy_fraction",
                    "soft_score_fraction",
                    "score_source_counts",
                    "truncated_fraction",
                    "invalid_teacher_actions",
                    "hard_action_target_information",
                    "tool_provenance",
                )
                if key in loaded
            }
            compact.update(_manifest_metadata_summary(loaded))
            payload[candidate.name] = compact
            payload["source_manifest"] = binding
            break
        if not found_direct:
            modal_summary = _modal_parts_summary(path)
            if modal_summary:
                payload["modal_parts_summary"] = modal_summary
            else:
                raise SystemExit(
                    f"input teacher-data root lacks a readable source manifest: {path}"
                )
        manifests.append(payload)
    return manifests


def _sha256_bytes(payload: bytes) -> str:
    return SHA256_PREFIX + hashlib.sha256(payload).hexdigest()


def _load_bound_manifest(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        before = resolved.read_bytes()
        loaded = json.loads(before.decode("utf-8"))
        after = resolved.read_bytes()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot authenticate source manifest {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise SystemExit(f"source manifest must contain a JSON object: {resolved}")
    before_sha256 = _sha256_bytes(before)
    if _sha256_bytes(after) != before_sha256:
        raise SystemExit(f"source manifest changed while being authenticated: {resolved}")
    return loaded, {
        "schema_version": SOURCE_MANIFEST_BINDING_SCHEMA,
        "path": str(resolved),
        "file_sha256": before_sha256,
    }


def _manifest_metadata_summary(value: Any) -> dict[str, Any]:
    tracks: set[str] = set()
    vps_values: set[int] = set()
    mixed_seats: set[bool] = set()
    mixed_modes: set[str] = set()
    graph_history_values: set[bool] = set()
    _collect_manifest_metadata(
        value,
        tracks=tracks,
        vps_values=vps_values,
        mixed_seats=mixed_seats,
        mixed_modes=mixed_modes,
        graph_history_values=graph_history_values,
    )
    summary: dict[str, Any] = {}
    if len(tracks) == 1:
        summary.setdefault("track", next(iter(tracks)))
    if len(vps_values) == 1:
        summary.setdefault("vps_to_win", next(iter(vps_values)))
    if len(mixed_seats) == 1:
        summary.setdefault("mixed_seats", next(iter(mixed_seats)))
    if len(mixed_modes) == 1:
        summary.setdefault("mixed_seat_mode", next(iter(mixed_modes)))
    if len(graph_history_values) == 1:
        summary.setdefault("graph_history_features", next(iter(graph_history_values)))
    if tracks:
        summary["tracks"] = sorted(tracks)
    if vps_values:
        summary["vps_to_win_values"] = sorted(vps_values)
    if mixed_seats:
        summary["mixed_seats_values"] = sorted(mixed_seats)
    if mixed_modes:
        summary["mixed_seat_modes"] = sorted(mixed_modes)
    if graph_history_values:
        summary["graph_history_features_values"] = sorted(graph_history_values)
    return summary


def _collect_manifest_metadata(
    value: Any,
    *,
    tracks: set[str],
    vps_values: set[int],
    mixed_seats: set[bool],
    mixed_modes: set[str],
    graph_history_values: set[bool],
) -> None:
    if isinstance(value, dict):
        track = value.get("track")
        if isinstance(track, str) and track:
            tracks.add(track)
        raw_vps = value.get("vps_to_win")
        if raw_vps not in (None, ""):
            try:
                vps_values.add(int(raw_vps))
            except (TypeError, ValueError):
                pass
        raw_mixed = value.get("mixed_seats")
        if isinstance(raw_mixed, bool):
            mixed_seats.add(raw_mixed)
        raw_mode = value.get("mixed_seat_mode")
        if isinstance(raw_mode, str) and raw_mode:
            mixed_modes.add(raw_mode)
        raw_graph_history = value.get("graph_history_features")
        if isinstance(raw_graph_history, bool):
            graph_history_values.add(raw_graph_history)
        for child in value.values():
            _collect_manifest_metadata(
                child,
                tracks=tracks,
                vps_values=vps_values,
                mixed_seats=mixed_seats,
                mixed_modes=mixed_modes,
                graph_history_values=graph_history_values,
            )
    elif isinstance(value, list):
        for child in value:
            _collect_manifest_metadata(
                child,
                tracks=tracks,
                vps_values=vps_values,
                mixed_seats=mixed_seats,
                mixed_modes=mixed_modes,
                graph_history_values=graph_history_values,
            )


def _modal_parts_summary(path: Path) -> dict[str, Any]:
    manifests = sorted((path / "parts").glob("part_*/manifest.json"))
    if not manifests:
        return {}
    tracks: set[str] = set()
    vps_values: set[int] = set()
    teachers: Counter[str] = Counter()
    score_sources: Counter[str] = Counter()
    part_tool_provenance: list[dict[str, Any]] = []
    games = 0
    samples = 0
    invalid = 0
    mixed_seats: set[bool] = set()
    mixed_modes: set[str] = set()
    part_manifests: list[dict[str, str]] = []
    for manifest in manifests:
        loaded, binding = _load_bound_manifest(manifest)
        part_manifests.append(binding)
        tool_provenance = loaded.get("tool_provenance")
        if isinstance(tool_provenance, dict):
            file_sha256 = tool_provenance.get("file_sha256")
            if isinstance(file_sha256, dict):
                part_tool_provenance.append({"file_sha256": file_sha256})
        track = loaded.get("track")
        if isinstance(track, str) and track:
            tracks.add(track)
        raw_vps = loaded.get("vps_to_win")
        if raw_vps not in (None, ""):
            try:
                vps_values.add(int(raw_vps))
            except (TypeError, ValueError):
                pass
        teachers.update({str(k): int(v) for k, v in dict(loaded.get("teacher_counts", {})).items()})
        score_sources.update(
            {str(k): int(v) for k, v in dict(loaded.get("score_source_counts", {})).items()}
        )
        games += int(loaded.get("completed_games", loaded.get("games", 0)) or 0)
        samples += int(loaded.get("samples", 0) or 0)
        invalid += int(loaded.get("invalid_teacher_actions", 0) or 0)
        mixed = loaded.get("mixed_seats")
        if isinstance(mixed, bool):
            mixed_seats.add(mixed)
        mode = loaded.get("mixed_seat_mode")
        if isinstance(mode, str) and mode:
            mixed_modes.add(mode)
    summary: dict[str, Any] = {
        "parts": len(manifests),
        "tracks": sorted(tracks),
        "vps_to_win_values": sorted(vps_values),
        "games": games,
        "samples": samples,
        "invalid_teacher_actions": invalid,
        "teacher_counts": dict(teachers.most_common()),
        "score_source_counts": dict(score_sources.most_common()),
        "mixed_seats_values": sorted(mixed_seats),
        "mixed_seat_modes": sorted(mixed_modes),
        "part_tool_provenance": part_tool_provenance,
        "part_manifests": part_manifests,
    }
    if len(tracks) == 1:
        summary["track"] = next(iter(tracks))
    if len(vps_values) == 1:
        summary["vps_to_win"] = next(iter(vps_values))
    if len(mixed_seats) == 1:
        summary["mixed_seats"] = next(iter(mixed_seats))
    if len(mixed_modes) == 1:
        summary["mixed_seat_mode"] = next(iter(mixed_modes))
    return summary


def _tool_provenance() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    files = [
        "tools/curate_teacher_data.py",
        "tools/generate_teacher_data.py",
        "tools/train_bc.py",
        "tools/report_teacher_data_quality.py",
        "catan_rules_v1.json",
        "src/catan_zero/rules.py",
        "src/catan_zero/rl/action_mask.py",
        "src/catan_zero/rl/multiagent_env.py",
        "src/catan_zero/rl/self_play.py",
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/xdim_lite_policy.py",
        "src/catan_zero/rl/policy_pool.py",
    ]
    hashes = _hash_required_files(repo_root, files)
    return {
        "schema_version": TOOL_PROVENANCE_SCHEMA,
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
    }


def _hash_required_files(repo_root: Path, files: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in files:
        path = repo_root / name
        try:
            hashes[name] = _sha256_bytes(path.read_bytes())
        except OSError as error:
            raise RuntimeError(
                f"required provenance file is unreadable or missing: {path}"
            ) from error
    return hashes


def _has_soft_policy(row: dict[str, Any]) -> bool:
    policy = np.asarray(row.get("target_policy", ()), dtype=np.float32)
    if policy.size == 0:
        return False
    return bool(np.sum(np.where(np.isfinite(policy), np.maximum(policy, 0.0), 0.0)) > 0.0)


def _has_soft_scores(row: dict[str, Any]) -> bool:
    scores = np.asarray(row.get("target_scores", ()), dtype=np.float32)
    return bool(scores.size and np.isfinite(scores).any())


def _legal_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"mean": 0.0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    return {
        "mean": float(np.mean(values)),
        "p50": int(np.percentile(values, 50)),
        "p90": int(np.percentile(values, 90)),
        "p99": int(np.percentile(values, 99)),
        "max": int(np.max(values)),
    }


if __name__ == "__main__":
    main()
