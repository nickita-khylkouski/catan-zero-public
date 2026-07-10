#!/usr/bin/env python3
"""Build a pre-training, topology-sensitive validation-decision mask.

The mask is deliberately derived only from immutable decision metadata:
validation game membership, phase, legal-action type, and typed topology target
IDs.  It never reads teacher probabilities, rewards, winners, model outputs, or
losses.  A decision qualifies only when it presents at least two distinct legal
targets for one Catan board operation, making the mask useful for evaluating a
model's sensitivity to board relationships rather than forced actions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from train_bc import MemmapCorpus, _validate_memmap_payload_inventory  # noqa: E402


SCHEMA_VERSION = "catan-zero-topology-sensitive-mask/v1"
CONFIG_SCHEMA_VERSION = "catan-zero-topology-sensitive-mask-config/v1"
VALIDATION_SCHEMA_VERSION = "train-validation-game-seeds-v1"
ACTION_TYPES = (
    "BUILD_SETTLEMENT",
    "BUILD_ROAD",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
    "MARITIME_TRADE",
    "offer_trade",
    "accept_trade",
    "reject_trade",
    "cancel_trade",
    "confirm_trade",
    "MOVE_ROBBER",
    "DISCARD_RESOURCE",
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "ROLL",
    "END_TURN",
)
_TYPE_TOKEN_START = 2
_TARGET_COLUMN = {
    "MOVE_ROBBER": 0,
    "BUILD_SETTLEMENT": 1,
    "BUILD_CITY": 1,
    "BUILD_ROAD": 2,
}
_CATEGORY_PRECEDENCE = (
    ("MOVE_ROBBER", "robber_hex_target"),
    ("BUILD_SETTLEMENT", "settlement_vertex_target"),
    ("BUILD_CITY", "city_vertex_target"),
    ("BUILD_ROAD", "road_edge_target"),
)
_SHA256_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")


class MaskBuildError(ValueError):
    """Input metadata cannot support an auditable mask."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _metadata_sha256(value: Mapping[str, Any]) -> str:
    """Hash parsed corpus metadata while preserving JSON NaN/Infinity tokens.

    Corpus statistics may legitimately contain non-finite sentinels. They are
    forbidden in emitted mask artifacts, but must not make authenticated
    metadata comparison crash after the canonical payload-inventory check.
    """

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MaskBuildError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise MaskBuildError(f"{label} must be a JSON object")
    return value


def _load_validation_seeds(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = _load_json_object(path, label="validation manifest")
    if payload.get("schema_version") != VALIDATION_SCHEMA_VERSION:
        raise MaskBuildError(
            "validation manifest schema_version must be "
            f"{VALIDATION_SCHEMA_VERSION!r}"
        )
    raw = payload.get("game_seeds")
    if (
        not isinstance(raw, list)
        or not raw
        or any(type(seed) is not int for seed in raw)
    ):
        raise MaskBuildError(
            "validation manifest game_seeds must be a non-empty integer list"
        )
    seeds = np.asarray(raw, dtype="<i8")
    if np.any(seeds[1:] <= seeds[:-1]):
        raise MaskBuildError("validation manifest game_seeds must be sorted and unique")
    declared_count = payload.get("validation_game_seed_count")
    if type(declared_count) is not int or declared_count != int(seeds.size):
        raise MaskBuildError("validation manifest game count does not match game_seeds")
    digest = "sha256:" + hashlib.sha256(seeds.tobytes()).hexdigest()
    declared_digest = payload.get("validation_game_seed_set_sha256")
    if declared_digest != digest:
        raise MaskBuildError("validation manifest game-seed digest is invalid")
    return seeds, {
        "file_sha256": _file_sha256(path),
        "manifest_sha256": _value_sha256(payload),
        "game_seed_set_sha256": digest,
    }


def _decode_types(tokens: np.ndarray, live: np.ndarray) -> np.ndarray:
    type_slice = np.asarray(
        tokens[:, _TYPE_TOKEN_START : _TYPE_TOKEN_START + len(ACTION_TYPES)],
        dtype=np.float32,
    )
    if type_slice.shape[1] != len(ACTION_TYPES):
        raise MaskBuildError(
            "legal_action_tokens does not contain the v1 action-type slots"
        )
    active = type_slice > 0.5
    counts = active.sum(axis=1)
    if np.any(counts[live] != 1):
        raise MaskBuildError(
            "every live legal action must have exactly one action-type token"
        )
    decoded = np.full(tokens.shape[0], -1, dtype=np.int16)
    decoded[live] = np.argmax(active[live], axis=1).astype(np.int16)
    return decoded


def _category_for_row(
    tokens: np.ndarray, targets: np.ndarray, live: np.ndarray, phase: str
) -> tuple[str, str, int] | None:
    decoded = _decode_types(tokens, live)
    for action_type, base_category in _CATEGORY_PRECEDENCE:
        type_index = ACTION_TYPES.index(action_type)
        column = _TARGET_COLUMN[action_type]
        chosen = live & (decoded == type_index) & (targets[:, column] >= 0)
        distinct_targets = np.unique(targets[chosen, column])
        if distinct_targets.size < 2:
            continue
        normalized_phase = phase.strip().lower()
        category = base_category
        if "initial" in normalized_phase and action_type == "BUILD_SETTLEMENT":
            category = "initial_settlement_vertex_target"
        elif "initial" in normalized_phase and action_type == "BUILD_ROAD":
            category = "initial_road_edge_target"
        return category, action_type, int(distinct_targets.size)
    return None


def _parse_category_minimums(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    valid = {category for _, category in _CATEGORY_PRECEDENCE} | {
        "initial_settlement_vertex_target",
        "initial_road_edge_target",
    }
    for raw in values:
        name, separator, count_raw = raw.partition("=")
        if not separator or name not in valid:
            raise MaskBuildError(
                f"invalid category minimum {raw!r}; expected known_category=positive_integer"
            )
        try:
            count = int(count_raw)
        except ValueError as error:
            raise MaskBuildError(f"invalid category minimum {raw!r}") from error
        if count <= 0 or name in result:
            raise MaskBuildError(f"invalid or duplicate category minimum {raw!r}")
        result[name] = count
    return result


def build_mask(
    corpus_dir: Path,
    validation_manifest: Path,
    *,
    min_games: int,
    min_decisions: int,
    min_category_decisions: Mapping[str, int] | None = None,
    batch_size: int = 4096,
) -> dict[str, Any]:
    if min_games <= 0 or min_decisions <= 0 or batch_size <= 0:
        raise MaskBuildError(
            "min_games, min_decisions, and batch_size must be positive"
        )
    category_minimums = dict(min_category_decisions or {})
    known_categories = {category for _, category in _CATEGORY_PRECEDENCE} | {
        "initial_settlement_vertex_target",
        "initial_road_edge_target",
    }
    invalid_minimums = sorted(set(category_minimums) - known_categories)
    if invalid_minimums or any(
        type(value) is not int or value <= 0 for value in category_minimums.values()
    ):
        raise MaskBuildError(
            f"invalid category minimums: unknown={invalid_minimums}, values={category_minimums}"
        )
    seeds, validation_source = _load_validation_seeds(validation_manifest)
    corpus_meta_path = corpus_dir / "corpus_meta.json"
    corpus_meta = _load_json_object(corpus_meta_path, label="corpus metadata")
    corpus_meta_file_sha256 = _file_sha256(corpus_meta_path)
    try:
        verified_inventory_sha = _validate_memmap_payload_inventory(
            corpus_dir, corpus_meta
        )
        # Open memmap views only after every schema-required payload has been
        # authenticated by the canonical inventory validator.
        corpus = MemmapCorpus(corpus_dir)
    except SystemExit as error:
        raise MaskBuildError(f"memmap corpus validation failed: {error}") from error
    if _file_sha256(corpus_meta_path) != corpus_meta_file_sha256:
        raise MaskBuildError(
            "corpus metadata file changed after payload inventory authentication"
        )
    loaded_meta = getattr(corpus, "meta", None)
    if not isinstance(loaded_meta, Mapping) or _metadata_sha256(
        dict(loaded_meta)
    ) != _metadata_sha256(corpus_meta):
        raise MaskBuildError(
            "loaded memmap corpus metadata differs from authenticated corpus metadata"
        )
    payload_inventory_sha = corpus_meta.get("payload_inventory_sha256")
    if (
        not isinstance(payload_inventory_sha, str)
        or _SHA256_RE.fullmatch(payload_inventory_sha) is None
    ):
        raise MaskBuildError(
            "corpus metadata must carry a valid payload_inventory_sha256"
        )
    if verified_inventory_sha != payload_inventory_sha:
        raise MaskBuildError("verified payload inventory digest differs from metadata")
    selected_meta = corpus_meta.get("selected_game_seed_manifest")
    if not isinstance(selected_meta, Mapping):
        raise MaskBuildError(
            "corpus metadata must bind an audited selected-game seed manifest"
        )
    declared_validation_digest = selected_meta.get(
        "validation_game_seed_set_sha256"
    )
    if declared_validation_digest != validation_source["game_seed_set_sha256"]:
        raise MaskBuildError(
            "validation manifest seed set differs from the holdout bound into corpus metadata"
        )
    required = {
        "game_seed",
        "decision_index",
        "phase",
        "legal_action_tokens",
        "legal_action_target_ids",
        "legal_action_mask",
    }
    missing = sorted(required - set(corpus.keys()))
    if missing:
        raise MaskBuildError(
            f"memmap corpus lacks required metadata columns: {missing}"
        )
    corpus_seeds = np.asarray(corpus["game_seed"], dtype=np.int64)
    selected_rows = np.flatnonzero(np.isin(corpus_seeds, seeds)).astype(np.int64)
    observed_seeds = np.unique(corpus_seeds[selected_rows])
    if not np.array_equal(observed_seeds, seeds):
        missing_seeds = sorted(set(map(int, seeds)) - set(map(int, observed_seeds)))
        raise MaskBuildError(
            f"validation games missing from corpus: {missing_seeds[:10]}"
        )

    records: list[dict[str, Any]] = []
    seen_decision_ids: set[str] = set()
    for start in range(0, selected_rows.size, batch_size):
        indices = selected_rows[start : start + batch_size]
        tokens_batch = np.asarray(corpus["legal_action_tokens"][indices])
        targets_batch = np.asarray(corpus["legal_action_target_ids"][indices])
        live_batch = np.asarray(corpus["legal_action_mask"][indices], dtype=np.bool_)
        if (
            tokens_batch.ndim != 3
            or targets_batch.ndim != 3
            or live_batch.ndim != 2
            or targets_batch.shape[-1] != 4
            or tokens_batch.shape[:2] != targets_batch.shape[:2]
            or tokens_batch.shape[:2] != live_batch.shape
        ):
            raise MaskBuildError(
                "legal action token, target, and mask metadata shapes are incompatible"
            )
        phases = np.asarray(corpus["phase"][indices]).astype(str)
        decisions = np.asarray(corpus["decision_index"][indices], dtype=np.int64)
        for offset, row_index in enumerate(indices):
            result = _category_for_row(
                tokens_batch[offset],
                targets_batch[offset],
                live_batch[offset],
                phases[offset],
            )
            if result is None:
                continue
            decision_index = int(decisions[offset])
            if decision_index < 0:
                raise MaskBuildError(
                    "selected topology-sensitive row has no decision_index"
                )
            game_seed = int(corpus_seeds[row_index])
            game_id = f"seed:{game_seed}"
            decision_id = f"{game_id}:decision:{decision_index}"
            if decision_id in seen_decision_ids:
                raise MaskBuildError(f"duplicate decision identity {decision_id}")
            seen_decision_ids.add(decision_id)
            category, action_type, target_count = result
            records.append(
                {
                    "decision_id": decision_id,
                    "game_id": game_id,
                    "game_seed": game_seed,
                    "decision_index": decision_index,
                    "category": category,
                    "action_type": action_type,
                    "distinct_legal_topology_targets": target_count,
                    "source_row_index": int(row_index),
                }
            )

    records.sort(
        key=lambda row: (row["game_seed"], row["decision_index"], row["category"])
    )
    game_count = len({record["game_id"] for record in records})
    counts = Counter(record["category"] for record in records)
    failures: list[str] = []
    if game_count < min_games:
        failures.append(f"selected games {game_count} < required {min_games}")
    if len(records) < min_decisions:
        failures.append(f"selected decisions {len(records)} < required {min_decisions}")
    for category, minimum in sorted(category_minimums.items()):
        if counts[category] < minimum:
            failures.append(
                f"category {category} has {counts[category]} < required {minimum}"
            )
    if failures:
        raise MaskBuildError("; ".join(failures))

    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "selection_rule": "at_least_two_distinct_legal_same_kind_topology_targets",
        "category_precedence": [
            {
                "action_type": action_type,
                "category": category,
                "target_column": _TARGET_COLUMN[action_type],
            }
            for action_type, category in _CATEGORY_PRECEDENCE
        ],
        "initial_phase_override": "case_insensitive_phase_contains_initial",
        "action_type_token_offset": _TYPE_TOKEN_START,
        "action_types": list(ACTION_TYPES),
        "minimums": {
            "games": min_games,
            "decisions": min_decisions,
            "category_decisions": dict(sorted(category_minimums.items())),
        },
        "port_category": {
            "included": False,
            "reason": "port location is not encoded in action type, typed target IDs, phase, or fixed board incidence; state features are intentionally excluded",
        },
    }
    source = {
        "corpus": {
            "corpus_meta_file_sha256": corpus_meta_file_sha256,
            "payload_inventory_sha256": payload_inventory_sha,
            "row_count": int(corpus.row_count),
        },
        "validation_manifest": validation_source,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_sha256_scope": "canonical_json_without_artifact_sha256",
        "config": config,
        "config_sha256": _value_sha256(config),
        "source": source,
        "source_sha256": _value_sha256(source),
        "summary": {
            "decision_count": len(records),
            "game_count": game_count,
            "category_counts": dict(sorted(counts.items())),
        },
        "members": records,
        "members_sha256": _value_sha256(records),
    }
    payload["artifact_sha256"] = _value_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-games", required=True, type=int)
    parser.add_argument("--min-decisions", required=True, type=int)
    parser.add_argument("--min-category-decisions", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    try:
        payload = build_mask(
            args.corpus,
            args.validation_manifest,
            min_games=args.min_games,
            min_decisions=args.min_decisions,
            min_category_decisions=_parse_category_minimums(
                args.min_category_decisions
            ),
            batch_size=args.batch_size,
        )
    except MaskBuildError as error:
        raise SystemExit(
            f"topology-sensitive mask build failed closed: {error}"
        ) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {args.out}")
    try:
        with args.out.open("xb") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
    except FileExistsError as error:
        raise SystemExit(
            f"refusing to overwrite existing artifact: {args.out}"
        ) from error
    print(
        json.dumps(
            {
                "out": str(args.out),
                **payload["summary"],
                "artifact_sha256": payload["artifact_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
