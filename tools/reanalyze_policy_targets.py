#!/usr/bin/env python3
"""Refresh policy-search targets without changing an authenticated trajectory.

This is deliberately a *target reanalyzer*, not another self-play producer.
The source game's actions, outcome, observations, and auxiliary supervision are
immutable.  For every admitted policy-active root we reconstruct the original
public state, prove that its public features and ordered legal actions still
match the shard, and then run the current public-conservation PIMC search with
a separately authenticated checkpoint.

The workflow is sealed in three stages::

    plan  ->  run-chunk (parallel, immutable claims)  ->  merge

The plan hashes the producer manifest, every trajectory shard, the trajectory
producer checkpoint, the target-reanalyzer checkpoint, the exact search
configuration, and the ordered row identities.  Merge replays all those hashes
and refuses missing, duplicated, foreign, or stale claims.  Output shards are
rebuilt from the authenticated source arrays and only the five search-target
columns in ``REWRITTEN_COLUMNS`` may change; the payload inventory is computed
from the new bytes rather than inherited from the source corpus.

Initial supported scope is intentionally narrow: complete two-player A1
producer-mirror games and rows that already carry authenticated, non-forced,
full-search policy supervision under ``public_conservation_pimc_v1``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TOOLS_DIR.parent / "src"
for _import_root in (_TOOLS_DIR, _SRC_DIR):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from regret_common import discover_shards, load_shard  # noqa: E402
from reconstruct_state import (  # noqa: E402
    GameActionSequence,
    action_size_for_colors,
    featurize_state,
    reconstruct_state,
    round_trip_row,
)

from catan_zero.rl.gumbel_self_play import (  # noqa: E402
    TARGET_INFORMATION_REGIME_PUBLIC,
)
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig  # noqa: E402
from catan_zero.search.native_gumbel_mcts import create_gumbel_search  # noqa: E402
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
    rust_policy_action_ids,
)


PLAN_SCHEMA = "a1-policy-target-reanalysis-plan-v1"
CLAIM_SCHEMA = "a1-policy-target-reanalysis-claim-v1"
MERGE_SCHEMA = "a1-policy-target-reanalysis-merged-v1"
PAYLOAD_INVENTORY_SCHEMA = "reanalysis-payload-inventory-v1"
COLORS = ("RED", "BLUE")
REWRITTEN_COLUMNS = frozenset(
    {
        "target_policy",
        "target_scores",
        "target_scores_mask",
        "root_value",
        "prior_policy",
    }
)
RECONSTRUCTION_COLUMNS = frozenset(
    {
        "legal_action_ids",
        "legal_action_context",
        "hex_tokens",
        "vertex_tokens",
        "edge_tokens",
        "player_tokens",
        "global_tokens",
        "event_tokens",
        "hex_mask",
        "vertex_mask",
        "edge_mask",
        "player_mask",
        "event_mask",
        "legal_action_tokens",
        "legal_action_mask",
        "legal_action_target_ids",
    }
)


class ReanalysisError(RuntimeError):
    """Fail-closed contract violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("xb") as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a byte-deterministic, NumPy-compatible uncompressed NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("xb") as handle:
        with zipfile.ZipFile(
            handle, mode="w", compression=zipfile.ZIP_STORED
        ) as archive:
            for key in sorted(arrays):
                if "/" in key or "\\" in key:
                    raise ReanalysisError(f"unsafe NPZ column name: {key!r}")
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[key]), allow_pickle=True
                )
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Exact equality with NaN equivalence, including string/object columns."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReanalysisError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReanalysisError(f"expected JSON object in {path}")
    return value


def _row_count(shard: Mapping[str, np.ndarray]) -> int:
    if "action_taken" not in shard:
        raise ReanalysisError("source shard lacks action_taken")
    return int(np.asarray(shard["action_taken"]).reshape(-1).shape[0])


def _scalar(shard: Mapping[str, np.ndarray], key: str, row: int, default: Any) -> Any:
    if key not in shard:
        return default
    value = np.asarray(shard[key])
    return value.reshape(-1)[row].item()


def _is_producer_mirror(shard: Mapping[str, np.ndarray], row: int) -> bool:
    if bool(_scalar(shard, "is_pool_game", row, False)):
        return False
    if int(_scalar(shard, "opponent_version", row, -1)) >= 0:
        return False
    tag = str(_scalar(shard, "opponent_tag", row, ""))
    opponent_md5 = str(_scalar(shard, "opponent_checkpoint_md5", row, ""))
    opponent_type = str(_scalar(shard, "opponent_type", row, ""))
    return tag in {"", "producer_self_play"} and not opponent_md5 and not opponent_type


def _eligible_policy_row(shard: Mapping[str, np.ndarray], row: int) -> bool:
    """The narrow admission rule for policy-target refresh."""
    required = {
        "policy_weight_multiplier",
        "used_full_search",
        "is_forced",
        "target_information_regime",
    }
    missing = required - set(shard)
    if missing:
        raise ReanalysisError(
            "source shard cannot authenticate policy-active rows; missing "
            + ", ".join(sorted(missing))
        )
    regime = str(_scalar(shard, "target_information_regime", row, ""))
    if regime != TARGET_INFORMATION_REGIME_PUBLIC:
        # Hidden-state roots are never silently upgraded into safe targets: we
        # cannot prove their stored policy supervision was admissible.
        return False
    return (
        float(_scalar(shard, "policy_weight_multiplier", row, 0.0)) > 0.0
        and bool(_scalar(shard, "used_full_search", row, False))
        and not bool(_scalar(shard, "is_forced", row, True))
        and _is_producer_mirror(shard, row)
    )


def _manifest_shards(manifest_path: Path, manifest: Mapping[str, Any]) -> list[Path]:
    raw = manifest.get("shards")
    if not isinstance(raw, list) or not raw:
        raise ReanalysisError("source manifest must contain a non-empty shards list")
    paths: list[Path] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ReanalysisError("source manifest shard entries must be paths")
        path = Path(item)
        if not path.is_absolute():
            path = manifest_path.parent / path
        paths.append(path.resolve())
    discovered = discover_shards(paths)
    if sorted(paths) != discovered:
        raise ReanalysisError(
            "source manifest must list exact shard files (no directories, missing files, or duplicates)"
        )
    return discovered


def _assert_complete_games(
    shards: Sequence[tuple[Path, Mapping[str, np.ndarray]]],
) -> dict[int, GameActionSequence]:
    rows: dict[int, list[tuple[int, int, str, str, bool, bool]]] = defaultdict(list)
    for _path, shard in shards:
        n = _row_count(shard)
        for row in range(n):
            seed = int(_scalar(shard, "game_seed", row, -1))
            rows[seed].append(
                (
                    int(_scalar(shard, "decision_index", row, -1)),
                    int(_scalar(shard, "action_taken", row, -1)),
                    str(_scalar(shard, "phase", row, "")),
                    str(_scalar(shard, "player", row, "")),
                    bool(_scalar(shard, "terminated", row, False)),
                    bool(_scalar(shard, "truncated", row, False)),
                )
            )
    sequences: dict[int, GameActionSequence] = {}
    for seed, game_rows in rows.items():
        game_rows.sort(key=lambda value: value[0])
        indices = [value[0] for value in game_rows]
        if indices != list(range(len(indices))):
            raise ReanalysisError(
                f"game_seed={seed} is not a complete root trajectory: decisions={indices[:12]}"
            )
        if not game_rows[-1][4] and not game_rows[-1][5]:
            raise ReanalysisError(
                f"game_seed={seed} has no authenticated terminal/truncated completion"
            )
        # Outcome fields are copied to every row by the producer; disagreement
        # on completion means shards from different/partial trajectories mixed.
        terminal_pairs = {(row[4], row[5]) for row in game_rows}
        if len(terminal_pairs) != 1:
            raise ReanalysisError(
                f"game_seed={seed} has inconsistent completion fields"
            )
        sequences[seed] = GameActionSequence(
            game_seed=seed,
            colors=COLORS,
            actions=[row[1] for row in game_rows],
            decision_indices=indices,
            phases=[row[2] for row in game_rows],
            players=[row[3] for row in game_rows],
        )
    return sequences


def default_search_config(*, seed: int = 1, n_full: int = 128) -> dict[str, Any]:
    """Exact, JSON-safe reanalysis operator. Search is always forced full."""
    return {
        "colors": list(COLORS),
        "max_depth": 80,
        "seed": int(seed),
        "c_visit": 50.0,
        "c_scale": 0.1,
        "prior_temperature": 1.0,
        "n_full": int(n_full),
        "n_fast": int(n_full),
        "p_full": 1.0,
        "lazy_interior_chance": True,
        "correct_rust_chance_spectra": True,
        "exact_budget_sh": True,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 24,
        "information_set_search": True,
        "determinization_particles": 4,
        "determinization_min_simulations": 32,
        "target_information_regime": TARGET_INFORMATION_REGIME_PUBLIC,
    }


def build_plan(
    *,
    source_manifest: Path,
    trajectory_producer_checkpoint: Path,
    target_checkpoint: Path,
    chunks: int,
    search_config: Mapping[str, Any],
) -> dict[str, Any]:
    if chunks < 1:
        raise ReanalysisError("chunks must be >= 1")
    manifest = _load_json(source_manifest)
    producer_sha = _sha256(trajectory_producer_checkpoint)
    if manifest.get("producer_checkpoint_sha256") != producer_sha:
        raise ReanalysisError(
            "trajectory producer checkpoint does not match source manifest: "
            f"declared={manifest.get('producer_checkpoint_sha256')!r}, actual={producer_sha!r}"
        )
    shard_paths = _manifest_shards(source_manifest, manifest)
    loaded = [(path, load_shard(path)) for path in shard_paths]
    sequences = _assert_complete_games(loaded)
    inventory: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for shard_index, (path, shard) in enumerate(loaded):
        n = _row_count(shard)
        inventory.append(
            {
                "index": shard_index,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": n,
            }
        )
        for row in range(n):
            if not _eligible_policy_row(shard, row):
                continue
            seed = int(_scalar(shard, "game_seed", row, -1))
            decision = int(_scalar(shard, "decision_index", row, -1))
            if seed not in sequences:
                raise ReanalysisError(
                    f"eligible row references incomplete game_seed={seed}"
                )
            identities.append(
                {
                    "shard_index": shard_index,
                    "row_index": row,
                    "game_seed": seed,
                    "decision_index": decision,
                }
            )
    identities.sort(
        key=lambda item: (
            item["game_seed"],
            item["decision_index"],
            item["shard_index"],
            item["row_index"],
        )
    )
    if not identities:
        regimes = sorted(
            {
                str(value)
                for _path, shard in loaded
                for value in np.asarray(
                    shard.get("target_information_regime", [])
                ).reshape(-1)
            }
        )
        raise ReanalysisError(
            "no authenticated policy-active producer-mirror rows admitted; "
            f"observed target_information_regime={regimes}"
        )
    for ordinal, identity in enumerate(identities):
        identity["ordinal"] = ordinal
        identity["chunk_index"] = ordinal % chunks
        identity["identity_sha256"] = _value_sha256(
            {
                key: identity[key]
                for key in ("shard_index", "row_index", "game_seed", "decision_index")
            }
        )
    config = dict(search_config)
    if config.get("target_information_regime") != TARGET_INFORMATION_REGIME_PUBLIC:
        raise ReanalysisError(
            "reanalyzer target_information_regime must be public_conservation_pimc_v1"
        )
    if config.get("information_set_search") is not True:
        raise ReanalysisError("reanalyzer must enable information_set_search")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "source_manifest": {
            "path": str(Path(source_manifest).resolve()),
            "sha256": _sha256(source_manifest),
        },
        "trajectory_producer": {
            "checkpoint_path": str(Path(trajectory_producer_checkpoint).resolve()),
            "checkpoint_sha256": producer_sha,
        },
        "target_reanalyzer": {
            "checkpoint_path": str(Path(target_checkpoint).resolve()),
            "checkpoint_sha256": _sha256(target_checkpoint),
        },
        "target_information_regime": TARGET_INFORMATION_REGIME_PUBLIC,
        "search_config": config,
        "search_config_sha256": _value_sha256(config),
        "source_shards": inventory,
        "source_inventory_sha256": _value_sha256(inventory),
        "chunks": int(chunks),
        "eligible_rows": identities,
        "eligible_rows_sha256": _value_sha256(identities),
        "rewritten_columns": sorted(REWRITTEN_COLUMNS),
    }
    plan["plan_sha256"] = _value_sha256(plan)
    return plan


def _verify_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ReanalysisError("unsupported reanalysis plan schema")
    expected = _value_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected:
        raise ReanalysisError("plan semantic hash mismatch")
    source_manifest = plan["source_manifest"]
    if _sha256(Path(source_manifest["path"])) != source_manifest["sha256"]:
        raise ReanalysisError("source manifest hash drift")
    for role in ("trajectory_producer", "target_reanalyzer"):
        checkpoint = plan[role]
        if (
            _sha256(Path(checkpoint["checkpoint_path"]))
            != checkpoint["checkpoint_sha256"]
        ):
            raise ReanalysisError(f"{role} checkpoint hash drift")
    manifest_payload = _load_json(Path(source_manifest["path"]))
    if (
        manifest_payload.get("producer_checkpoint_sha256")
        != plan["trajectory_producer"]["checkpoint_sha256"]
    ):
        raise ReanalysisError(
            "plan trajectory producer no longer matches source manifest"
        )
    if plan.get("target_information_regime") != TARGET_INFORMATION_REGIME_PUBLIC:
        raise ReanalysisError("plan target information regime is not public PIMC")
    config = plan.get("search_config")
    if not isinstance(config, dict) or _value_sha256(config) != plan.get(
        "search_config_sha256"
    ):
        raise ReanalysisError("search configuration hash mismatch")
    if (
        config.get("information_set_search") is not True
        or config.get("target_information_regime") != TARGET_INFORMATION_REGIME_PUBLIC
    ):
        raise ReanalysisError("search configuration is not public-conservation PIMC")
    identities = plan.get("eligible_rows")
    if not isinstance(identities, list) or _value_sha256(identities) != plan.get(
        "eligible_rows_sha256"
    ):
        raise ReanalysisError("eligible-row identity hash mismatch")
    chunks = int(plan["chunks"])
    for ordinal, identity in enumerate(identities):
        core = {
            key: identity[key]
            for key in ("shard_index", "row_index", "game_seed", "decision_index")
        }
        if (
            int(identity["ordinal"]) != ordinal
            or int(identity["chunk_index"]) != ordinal % chunks
            or identity["identity_sha256"] != _value_sha256(core)
        ):
            raise ReanalysisError("eligible-row identity/chunk assignment mismatch")
    inventory = plan["source_shards"]
    if _value_sha256(inventory) != plan["source_inventory_sha256"]:
        raise ReanalysisError("source inventory semantic hash mismatch")
    for item in inventory:
        path = Path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise ReanalysisError(f"source shard hash drift: {path}")
        if _row_count(load_shard(path)) != int(item["rows"]):
            raise ReanalysisError(f"source shard row-count drift: {path}")


def _stored_features(
    shard: Mapping[str, np.ndarray], row: int
) -> dict[str, np.ndarray]:
    # round_trip_row ignores unknown keys and checks its full public surface.
    return {
        key: np.asarray(value)[row]
        for key, value in shard.items()
        if np.asarray(value).ndim > 0
    }


def _verify_reconstruction(
    *, shard: Mapping[str, np.ndarray], row: int, sequence: GameActionSequence
) -> tuple[Any, dict[str, Any]]:
    missing = RECONSTRUCTION_COLUMNS - set(shard)
    if missing:
        raise ReanalysisError(
            "source row lacks the complete public reconstruction surface: "
            + ", ".join(sorted(missing))
        )
    decision = int(_scalar(shard, "decision_index", row, -1))
    result = round_trip_row(
        sequence,
        decision,
        _stored_features(shard, row),
        np.asarray(shard["legal_action_ids"])[row],
        correct_rust_chance_spectra=True,
        action_size=action_size_for_colors(COLORS),
    )
    if not result.ok:
        raise ReanalysisError(
            "reconstructed public root mismatch before search: "
            f"game_seed={sequence.game_seed} decision={decision} "
            f"legal={result.legal_ids_match} worst={result.worst_key} diff={result.max_abs_diff}"
        )
    game = reconstruct_state(
        sequence.game_seed,
        sequence.actions,
        decision,
        colors=COLORS,
        correct_rust_chance_spectra=True,
    )
    return game, featurize_state(game, colors=COLORS)


def _search_patch(search: Any, game: Any, feature: Mapping[str, Any]) -> dict[str, Any]:
    result = search.search(game, force_full=True)
    legal_rust = tuple(
        int(action) for action in game.playable_action_indices(list(COLORS), None)
    )
    mapped = tuple(
        int(value)
        for value in rust_policy_action_ids(
            game,
            legal_rust,
            colors=COLORS,
            action_size=action_size_for_colors(COLORS),
        )
    )
    if mapped != tuple(int(value) for value in feature["legal_policy_ids"]):
        raise ReanalysisError(
            "legal action order changed between reconstruction and search"
        )
    if set(result.improved_policy) != set(legal_rust):
        raise ReanalysisError("search result does not cover the exact legal root")
    target = [float(result.improved_policy[action]) for action in legal_rust]
    raw_scores = [
        float(result.q_values.get(action, float("nan"))) for action in legal_rust
    ]
    score_mask = [bool(np.isfinite(value)) for value in raw_scores]
    # Claims are strict JSON (NaN is intentionally forbidden). Masked slots
    # carry a harmless zero and become NaN padding only beyond legal width at
    # merge time; consumers must consult target_scores_mask.
    scores = [value if valid else 0.0 for value, valid in zip(raw_scores, score_mask)]
    priors = [float(result.priors[action]) for action in legal_rust]
    if not result.used_full_search or not np.isfinite(result.root_value):
        raise ReanalysisError(
            "forced-full reanalysis returned no full-search root value"
        )
    if not np.isclose(sum(target), 1.0, atol=1e-5) or not np.isclose(
        sum(priors), 1.0, atol=1e-5
    ):
        raise ReanalysisError("search target/prior is not normalized")
    return {
        "target_policy": target,
        "target_scores": scores,
        "target_scores_mask": score_mask,
        "root_value": float(result.root_value),
        "prior_policy": priors,
    }


def _search_from_plan(plan: Mapping[str, Any], *, device: str) -> Any:
    evaluator = EntityGraphRustEvaluator.from_checkpoint(
        plan["target_reanalyzer"]["checkpoint_path"],
        device=device,
        config=EntityGraphRustEvaluatorConfig(
            value_scale=1.0,
            prior_temperature=float(plan["search_config"]["prior_temperature"]),
            public_observation=True,
            rust_featurize=True,
        ),
    )
    allowed = {field.name for field in dataclasses.fields(GumbelChanceMCTSConfig)}
    kwargs = {
        key: value for key, value in plan["search_config"].items() if key in allowed
    }
    kwargs["colors"] = tuple(kwargs.get("colors", COLORS))
    config = GumbelChanceMCTSConfig(**kwargs)
    return create_gumbel_search(config, evaluator, native_hot_loop=False)


def run_chunk(
    *,
    plan: Mapping[str, Any],
    chunk_index: int,
    output: Path,
    device: str = "cpu",
    search: Any = None,
) -> dict[str, Any]:
    _verify_plan(plan)
    chunks = int(plan["chunks"])
    if not 0 <= chunk_index < chunks:
        raise ReanalysisError(f"chunk_index must be in [0,{chunks})")
    entries = [
        row for row in plan["eligible_rows"] if int(row["chunk_index"]) == chunk_index
    ]
    loaded = {
        int(item["index"]): load_shard(Path(item["path"]))
        for item in plan["source_shards"]
    }
    sequences = _assert_complete_games(
        [
            (Path(item["path"]), loaded[int(item["index"])])
            for item in plan["source_shards"]
        ]
    )
    search = search or _search_from_plan(plan, device=device)
    patches: list[dict[str, Any]] = []
    for identity in entries:
        shard = loaded[int(identity["shard_index"])]
        row = int(identity["row_index"])
        if not _eligible_policy_row(shard, row):
            raise ReanalysisError("planned row is no longer policy-active/admissible")
        sequence = sequences[int(identity["game_seed"])]
        game, feature = _verify_reconstruction(shard=shard, row=row, sequence=sequence)
        patch = _search_patch(search, game, feature)
        patches.append(
            {
                "identity_sha256": identity["identity_sha256"],
                "shard_index": identity["shard_index"],
                "row_index": identity["row_index"],
                "values": patch,
            }
        )
    claim = {
        "schema_version": CLAIM_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "chunk_index": int(chunk_index),
        "expected_rows": len(entries),
        "patches": patches,
        "patches_sha256": _value_sha256(patches),
        "target_reanalyzer_checkpoint_sha256": plan["target_reanalyzer"][
            "checkpoint_sha256"
        ],
        "target_information_regime": TARGET_INFORMATION_REGIME_PUBLIC,
    }
    claim["claim_sha256"] = _value_sha256(claim)
    _write_json_atomic(output, claim)
    return claim


def _coerce_row_value(original: np.ndarray, row: int, value: Any) -> np.ndarray | Any:
    dtype = original.dtype
    if original.ndim == 1:
        return np.asarray(value, dtype=dtype).reshape(()).item()
    output = original[row].copy()
    raw = np.asarray(value, dtype=dtype)
    if raw.ndim != 1 or raw.shape[0] > output.shape[0]:
        raise ReanalysisError("patch vector cannot fit source shard column")
    fill: Any = (
        False
        if dtype == np.bool_
        else np.nan
        if np.issubdtype(dtype, np.floating)
        else 0
    )
    output[...] = fill
    output[: raw.shape[0]] = raw
    return output


def _apply_patch(
    arrays: dict[str, np.ndarray], row: int, values: Mapping[str, Any]
) -> None:
    if set(values) != REWRITTEN_COLUMNS:
        raise ReanalysisError(
            f"claim attempted wrong columns: got={sorted(values)}, expected={sorted(REWRITTEN_COLUMNS)}"
        )
    for key, value in values.items():
        if key not in arrays:
            raise ReanalysisError(f"source shard lacks rewrite column {key}")
        arrays[key][row] = _coerce_row_value(arrays[key], row, value)


def _verify_claim(claim: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if (
        claim.get("schema_version") != CLAIM_SCHEMA
        or claim.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise ReanalysisError("foreign or unsupported chunk claim")
    expected = _value_sha256(
        {key: value for key, value in claim.items() if key != "claim_sha256"}
    )
    if claim.get("claim_sha256") != expected:
        raise ReanalysisError("chunk claim hash mismatch")
    if claim.get("patches_sha256") != _value_sha256(claim.get("patches")):
        raise ReanalysisError("chunk patch inventory hash mismatch")
    if (
        claim.get("target_reanalyzer_checkpoint_sha256")
        != plan["target_reanalyzer"]["checkpoint_sha256"]
    ):
        raise ReanalysisError("chunk target checkpoint mismatch")
    if claim.get("target_information_regime") != TARGET_INFORMATION_REGIME_PUBLIC:
        raise ReanalysisError("chunk target information regime mismatch")
    chunk_index = int(claim["chunk_index"])
    expected = {
        row["identity_sha256"]: row
        for row in plan["eligible_rows"]
        if int(row["chunk_index"]) == chunk_index
    }
    if int(claim.get("expected_rows", -1)) != len(expected):
        raise ReanalysisError(f"chunk {chunk_index} expected-row count mismatch")
    for patch in claim.get("patches", []):
        identity = expected.get(patch.get("identity_sha256"))
        if identity is None:
            raise ReanalysisError(
                f"chunk {chunk_index} contains a foreign row identity"
            )
        if int(patch.get("shard_index", -1)) != int(identity["shard_index"]) or int(
            patch.get("row_index", -1)
        ) != int(identity["row_index"]):
            raise ReanalysisError(
                "chunk row identity points at the wrong source location"
            )


def merge_claims(
    *, plan: Mapping[str, Any], claim_paths: Sequence[Path], output: Path
) -> dict[str, Any]:
    _verify_plan(plan)
    claims = [_load_json(path) for path in claim_paths]
    for claim in claims:
        _verify_claim(claim, plan)
    by_chunk: dict[int, Mapping[str, Any]] = {}
    for claim in claims:
        index = int(claim["chunk_index"])
        if index in by_chunk:
            raise ReanalysisError(f"duplicate claim for chunk {index}")
        by_chunk[index] = claim
    expected_chunks = set(range(int(plan["chunks"])))
    if set(by_chunk) != expected_chunks:
        raise ReanalysisError(
            f"incomplete claims: missing={sorted(expected_chunks - set(by_chunk))} "
            f"extra={sorted(set(by_chunk) - expected_chunks)}"
        )
    expected_ids = {row["identity_sha256"] for row in plan["eligible_rows"]}
    patches: dict[str, Mapping[str, Any]] = {}
    for index in sorted(by_chunk):
        claim = by_chunk[index]
        if len(claim["patches"]) != int(claim["expected_rows"]):
            raise ReanalysisError(f"chunk {index} incomplete")
        for patch in claim["patches"]:
            identity = str(patch["identity_sha256"])
            if identity in patches:
                raise ReanalysisError(f"duplicate row patch {identity}")
            patches[identity] = patch
    if set(patches) != expected_ids:
        raise ReanalysisError(
            f"row claims incomplete/foreign: missing={len(expected_ids - set(patches))} "
            f"extra={len(set(patches) - expected_ids)}"
        )

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ReanalysisError(f"merge output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    by_shard: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for patch in patches.values():
        by_shard[int(patch["shard_index"])].append(patch)
    payload_inventory: list[dict[str, Any]] = []
    output_shards: list[str] = []
    for source in plan["source_shards"]:
        shard_index = int(source["index"])
        original = load_shard(Path(source["path"]))
        arrays = {key: value.copy() for key, value in original.items()}
        before = {
            key: value.copy()
            for key, value in original.items()
            if key not in REWRITTEN_COLUMNS
        }
        for patch in sorted(
            by_shard.get(shard_index, []), key=lambda value: int(value["row_index"])
        ):
            _apply_patch(arrays, int(patch["row_index"]), patch["values"])
        for key, expected in before.items():
            if not _array_equal(arrays[key], expected):
                raise ReanalysisError(f"non-target column changed during merge: {key}")
        destination = output / f"reanalyzed_shard_{shard_index:05d}.npz"
        _write_npz_atomic(destination, arrays)
        record = {
            "path": destination.name,
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
            "rows": _row_count(arrays),
        }
        payload_inventory.append(record)
        output_shards.append(record["path"])
    manifest = {
        "schema_version": MERGE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "trajectory_producer": plan["trajectory_producer"],
        "target_reanalyzer": plan["target_reanalyzer"],
        "target_information_regime": TARGET_INFORMATION_REGIME_PUBLIC,
        "search_config": plan["search_config"],
        "search_config_sha256": plan["search_config_sha256"],
        "rewritten_columns": sorted(REWRITTEN_COLUMNS),
        "reanalyzed_rows": len(patches),
        "shards": output_shards,
        "payload_inventory_schema": PAYLOAD_INVENTORY_SCHEMA,
        "payload_inventory": payload_inventory,
        "payload_inventory_sha256": _value_sha256(payload_inventory),
        "claim_sha256s": [
            by_chunk[index]["claim_sha256"] for index in sorted(by_chunk)
        ],
    }
    manifest["manifest_sha256"] = _value_sha256(manifest)
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _command_plan(args: argparse.Namespace) -> None:
    config = default_search_config(seed=args.seed, n_full=args.n_full)
    plan = build_plan(
        source_manifest=Path(args.source_manifest),
        trajectory_producer_checkpoint=Path(args.trajectory_producer_checkpoint),
        target_checkpoint=Path(args.target_checkpoint),
        chunks=args.chunks,
        search_config=config,
    )
    _write_json_atomic(Path(args.output), plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


def _command_chunk(args: argparse.Namespace) -> None:
    claim = run_chunk(
        plan=_load_json(Path(args.plan)),
        chunk_index=args.chunk_index,
        output=Path(args.output),
        device=args.device,
    )
    print(json.dumps(claim, indent=2, sort_keys=True))


def _command_merge(args: argparse.Namespace) -> None:
    manifest = merge_claims(
        plan=_load_json(Path(args.plan)),
        claim_paths=[Path(path) for path in args.claim],
        output=Path(args.output),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--source-manifest", required=True)
    plan.add_argument("--trajectory-producer-checkpoint", required=True)
    plan.add_argument("--target-checkpoint", required=True)
    plan.add_argument("--chunks", type=int, required=True)
    plan.add_argument("--n-full", type=int, default=128)
    plan.add_argument("--seed", type=int, default=1)
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=_command_plan)
    chunk = commands.add_parser("run-chunk")
    chunk.add_argument("--plan", required=True)
    chunk.add_argument("--chunk-index", type=int, required=True)
    chunk.add_argument("--device", default="cuda")
    chunk.add_argument("--output", required=True)
    chunk.set_defaults(func=_command_chunk)
    merge = commands.add_parser("merge")
    merge.add_argument("--plan", required=True)
    merge.add_argument("--claim", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=_command_merge)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
