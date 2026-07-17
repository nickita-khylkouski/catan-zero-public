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
import hmac
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TOOLS_DIR.parent / "src"
for _import_root in (_TOOLS_DIR, _SRC_DIR):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from regret_common import discover_shards, load_shard  # noqa: E402
from reconstruct_state import (  # noqa: E402
    GameActionSequence,
    featurize_state,
    reconstruct_state,
    round_trip_row,
)

from catan_zero.rl.action_mask import ActionCatalog  # noqa: E402
from catan_zero.rl.gumbel_self_play import (  # noqa: E402
    ACTION_MASK_VERSION,
    TARGET_INFORMATION_REGIME_PUBLIC,
)
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
    require_known_entity_feature_adapter,
    resolve_checkpoint_entity_feature_adapter,
)
from catan_zero.rl.meaningful_history import (  # noqa: E402
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
    meaningful_public_history_limit,
)
from catan_zero.rl.target_reliability import (  # noqa: E402
    TARGET_RELIABILITY_COLUMNS,
    unaudited_target_reliability_fields,
)
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig  # noqa: E402
from catan_zero.search.native_gumbel_mcts import create_gumbel_search  # noqa: E402
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
    rust_policy_action_ids,
)


PLAN_SCHEMA = "a1-policy-target-reanalysis-plan-v3"
CLAIM_SCHEMA = "a1-policy-target-reanalysis-claim-v2"
MERGE_SCHEMA = "a1-policy-target-reanalysis-merged-v2"
PAYLOAD_INVENTORY_SCHEMA = "reanalysis-payload-inventory-v1"
PRODUCER_INPUT_ABI_SCHEMA = "a1-trajectory-producer-input-abi-v1"
COLORS = ("RED", "BLUE")
KNOWN_COMPATIBLE_ACTION_MASK_VERSIONS = frozenset(
    {str(ActionCatalog.version), str(ACTION_MASK_VERSION)}
)
REWRITTEN_COLUMNS = frozenset(
    {
        "teacher_name",
        "target_policy",
        "target_policy_mask",
        "target_scores",
        "target_scores_mask",
        "root_value",
        "root_value_mask",
        "root_prior_value",
        "root_prior_value_mask",
        "prior_policy",
        "simulations_used",
        "used_full_search",
        "search_evidence_version",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
        "search_prior_policy_flat",
        "trajectory_producer_checkpoint_sha256",
        "target_reanalyzer_checkpoint_sha256",
        "target_reanalysis_search_config_sha256",
        "target_reanalysis_plan_sha256",
        *TARGET_RELIABILITY_COLUMNS,
    }
)
SEARCH_PATCH_COLUMNS = frozenset(
    {
        "target_policy",
        "target_policy_mask",
        "target_scores",
        "target_scores_mask",
        "root_value",
        "root_value_mask",
        "root_prior_value",
        "root_prior_value_mask",
        "prior_policy",
        "simulations_used",
        "used_full_search",
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


class _LegacyInputABIMetadataUnavailable(ReanalysisError):
    """Checkpoint predates the explicit ABI or cannot be decoded."""

    def __init__(
        self,
        message: str,
        *,
        explicit_input_fields: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.explicit_input_fields = dict(explicit_input_fields or {})


def _runtime_attestation() -> dict[str, Any]:
    """Bind the exact code and native engine that define the search operator."""
    repo = _TOOLS_DIR.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        import catanatron_rs  # type: ignore
    except (OSError, subprocess.CalledProcessError, ImportError) as error:
        raise ReanalysisError(f"cannot attest reanalysis runtime: {error}") from error
    native_path = Path(catanatron_rs.__file__).resolve()
    source_paths = (
        Path(__file__).resolve(),
        _TOOLS_DIR / "reconstruct_state.py",
        repo / "src/catan_zero/search/gumbel_chance_mcts.py",
        repo / "src/catan_zero/search/native_gumbel_mcts.py",
        repo / "src/catan_zero/search/neural_rust_mcts.py",
        repo / "src/catan_zero/rl/gumbel_self_play.py",
    )
    sources = [
        {"path": str(path.relative_to(repo)), "sha256": _sha256(path)}
        for path in source_paths
    ]
    try:
        wheel_version = importlib.metadata.version("catanatron-rs")
    except importlib.metadata.PackageNotFoundError:
        wheel_version = str(getattr(catanatron_rs, "__version__", "unknown"))
    value = {
        "repo_commit": commit,
        "source_files": sources,
        "catanatron_rs": {
            "path": str(native_path),
            "sha256": _sha256(native_path),
            "version": wheel_version,
        },
    }
    value["runtime_sha256"] = _value_sha256(value)
    return value


def _claim_hmac(value: Mapping[str, Any], key: bytes) -> str:
    return (
        "sha256:" + hmac.new(key, _canonical_bytes(value), hashlib.sha256).hexdigest()
    )


def _load_auth_key(path: Path, expected_sha256: str) -> bytes:
    key = Path(path).read_bytes()
    if len(key) < 32 or _sha256(path) != expected_sha256:
        raise ReanalysisError(
            "claim authentication key is too short or hash-mismatched"
        )
    return key


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


def _seal_producer_input_abi(
    raw: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    binding_source: str,
) -> dict[str, Any]:
    """Validate and hash the exact feature contract used by source rows."""

    action_size = raw.get("action_size")
    if (
        not isinstance(action_size, (int, np.integer))
        or isinstance(action_size, (bool, np.bool_))
        or int(action_size) < 1
    ):
        raise ReanalysisError("producer input ABI action_size must be a positive integer")
    history_enabled = raw.get("meaningful_public_history")
    if not isinstance(history_enabled, (bool, np.bool_)):
        raise ReanalysisError(
            "producer input ABI meaningful_public_history must be an explicit boolean"
        )
    history_limit = raw.get("event_history_limit")
    if (
        not isinstance(history_limit, (int, np.integer))
        or isinstance(history_limit, (bool, np.bool_))
        or int(history_limit) < 1
    ):
        raise ReanalysisError(
            "producer input ABI event_history_limit must be a positive integer"
        )
    history_schema = str(raw.get("meaningful_public_history_schema", "") or "")
    try:
        schema_history_limit = meaningful_public_history_limit(history_schema)
    except ValueError as error:
        raise ReanalysisError(
            "producer input ABI has an unsupported meaningful-public-history schema"
        ) from error
    if bool(history_enabled):
        if int(history_limit) > int(schema_history_limit):
            raise ReanalysisError(
                "producer input ABI event_history_limit exceeds its enabled schema "
                f"cap: limit={int(history_limit)}, cap={int(schema_history_limit)}"
            )
    action_mask_version = str(raw.get("action_mask_version", "") or "")
    if not action_mask_version:
        raise ReanalysisError(
            "producer input ABI action_mask_version must be nonempty"
        )
    if action_mask_version not in KNOWN_COMPATIBLE_ACTION_MASK_VERSIONS:
        raise ReanalysisError(
            "producer input ABI has an unsupported action_mask_version "
            f"{action_mask_version!r}; known compatible versions are "
            f"{sorted(KNOWN_COMPATIBLE_ACTION_MASK_VERSIONS)!r}"
        )
    try:
        adapter_version = require_known_entity_feature_adapter(
            raw.get("entity_feature_adapter_version")
        )
    except ValueError as error:
        raise ReanalysisError(
            "producer input ABI has an unsupported entity feature adapter"
        ) from error
    if bool(history_enabled):
        adapter_requires_v2 = adapter_version in {
            RUST_ENTITY_ADAPTER_V5,
            RUST_ENTITY_ADAPTER_V6,
        }
        expected_history_schema = (
            MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
            if adapter_requires_v2
            else MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1
        )
        if history_schema != expected_history_schema:
            raise ReanalysisError(
                "producer input ABI meaningful-public-history schema/adapter "
                "mismatch: "
                f"adapter={adapter_version!r} requires "
                f"schema={expected_history_schema!r} when history is enabled"
            )
    abi = {
        "schema_version": PRODUCER_INPUT_ABI_SCHEMA,
        "checkpoint_sha256": str(checkpoint_sha256),
        "binding_source": str(binding_source),
        "action_size": int(action_size),
        "meaningful_public_history": bool(history_enabled),
        "meaningful_public_history_schema": history_schema,
        "event_history_limit": int(history_limit),
        "entity_feature_adapter_version": adapter_version,
        "action_mask_version": action_mask_version,
    }
    abi["input_abi_sha256"] = _value_sha256(abi)
    return abi


def _checkpoint_config_fields(config: Any) -> tuple[Mapping[str, Any], bool]:
    if isinstance(config, Mapping) and isinstance(config.get("fields"), Mapping):
        return config["fields"], True
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return (
            {
                field.name: getattr(config, field.name)
                for field in dataclasses.fields(config)
                if hasattr(config, field.name)
            },
            False,
        )
    raise ReanalysisError(
        "trajectory producer checkpoint lacks an authenticated name-keyed or "
        "legacy dataclass config"
    )


def _producer_input_abi_from_checkpoint(
    checkpoint: Path,
    checkpoint_sha256: str,
    *,
    binding_source: str = "trajectory_producer_checkpoint",
    checkpoint_role: str = "trajectory producer",
) -> dict[str, Any]:
    try:
        import torch
        from catan_zero.rl.entity_token_policy import EntityGraphConfig

        try:
            from numpy._core.multiarray import scalar as numpy_scalar
        except ImportError:  # NumPy 1.x compatibility.
            from numpy.core.multiarray import scalar as numpy_scalar

        safe_globals = [
            numpy_scalar,
            np.dtype,
            type(np.dtype(np.int64)),
            EntityGraphConfig,
        ]
        with torch.serialization.safe_globals(safe_globals):
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
    except Exception as error:
        raise _LegacyInputABIMetadataUnavailable(
            f"cannot safely read {checkpoint_role} checkpoint input ABI"
        ) from error
    if not isinstance(payload, Mapping) or payload.get("policy_type") != "entity_graph":
        raise ReanalysisError(
            f"{checkpoint_role} checkpoint is not an authenticated entity_graph policy"
        )
    fields, modern_name_keyed_config = _checkpoint_config_fields(
        payload.get("config")
    )
    required = {
        "action_size",
        "action_mask_version",
        "meaningful_public_history",
        "meaningful_public_history_schema",
        "event_history_limit",
    }
    explicit_input_fields = {
        key: fields[key] for key in required if key in fields
    }
    config_action_mask_version = (
        str(fields["action_mask_version"] or "")
        if "action_mask_version" in fields
        else None
    )
    if "action_mask_version" in payload:
        top_level_action_mask_version = str(payload["action_mask_version"] or "")
        if config_action_mask_version is not None and (
            not top_level_action_mask_version
            or top_level_action_mask_version != config_action_mask_version
        ):
            raise ReanalysisError(
                f"{checkpoint_role} checkpoint action_mask_version conflicts "
                "between top-level metadata and config"
            )
        explicit_input_fields["action_mask_version"] = (
            top_level_action_mask_version
        )
    adapter_version: str | None = None
    if "entity_feature_adapter" in payload:
        try:
            adapter_version, _source = resolve_checkpoint_entity_feature_adapter(
                payload["entity_feature_adapter"], metadata_present=True
            )
        except ValueError as error:
            raise ReanalysisError(
                f"{checkpoint_role} checkpoint has invalid entity feature adapter "
                "metadata"
            ) from error
        explicit_input_fields["entity_feature_adapter_version"] = adapter_version
    missing = required - set(fields)
    if missing:
        error_type = (
            ReanalysisError
            if modern_name_keyed_config
            else _LegacyInputABIMetadataUnavailable
        )
        message = (
            f"{checkpoint_role} checkpoint config lacks explicit input ABI fields: "
            + ", ".join(sorted(missing))
        )
        if error_type is _LegacyInputABIMetadataUnavailable:
            raise error_type(
                message, explicit_input_fields=explicit_input_fields
            )
        raise error_type(message)
    if adapter_version is None:
        error_type = (
            ReanalysisError
            if modern_name_keyed_config
            else _LegacyInputABIMetadataUnavailable
        )
        message = (
            f"{checkpoint_role} checkpoint lacks explicit entity feature adapter "
            "metadata"
        )
        if error_type is _LegacyInputABIMetadataUnavailable:
            raise error_type(
                message, explicit_input_fields=explicit_input_fields
            )
        raise error_type(message)
    return _seal_producer_input_abi(
        {
            **{key: fields[key] for key in required},
            "entity_feature_adapter_version": adapter_version,
        },
        checkpoint_sha256=checkpoint_sha256,
        binding_source=binding_source,
    )


def _producer_input_abi_from_manifest(
    manifest: Mapping[str, Any],
    checkpoint_sha256: str,
    *,
    explicit_checkpoint_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit legacy checkpoints only through an explicit, self-hashed contract."""

    raw = manifest.get("producer_input_abi")
    if not isinstance(raw, Mapping):
        raise ReanalysisError(
            "legacy trajectory producer checkpoint has no explicit producer_input_abi "
            "contract in its authenticated source manifest"
        )
    if raw.get("schema_version") != PRODUCER_INPUT_ABI_SCHEMA:
        raise ReanalysisError("unsupported manifest producer_input_abi schema")
    if raw.get("checkpoint_sha256") != checkpoint_sha256:
        raise ReanalysisError(
            "manifest producer_input_abi is not bound to the trajectory producer "
            "checkpoint"
        )
    expected_hash = _value_sha256(
        {key: value for key, value in raw.items() if key != "input_abi_sha256"}
    )
    if raw.get("input_abi_sha256") != expected_hash:
        raise ReanalysisError("manifest producer_input_abi semantic hash mismatch")
    sealed = _seal_producer_input_abi(
        raw,
        checkpoint_sha256=checkpoint_sha256,
        binding_source="source_manifest_explicit_legacy_contract",
    )
    # The manifest may not choose a different provenance label after hashing.
    if raw.get("binding_source") != sealed["binding_source"]:
        raise ReanalysisError("manifest producer_input_abi binding source mismatch")
    if raw.get("input_abi_sha256") != sealed["input_abi_sha256"]:
        raise ReanalysisError("manifest producer_input_abi is not canonical")
    explicit_fields = dict(explicit_checkpoint_fields or {})
    if explicit_fields:
        completed_from_checkpoint = _seal_producer_input_abi(
            {**sealed, **explicit_fields},
            checkpoint_sha256=checkpoint_sha256,
            binding_source="source_manifest_explicit_legacy_contract",
        )
        if completed_from_checkpoint != sealed:
            conflicts = sorted(
                key
                for key in explicit_fields
                if completed_from_checkpoint.get(key) != sealed.get(key)
            )
            raise ReanalysisError(
                "manifest producer_input_abi conflicts with explicit legacy "
                "checkpoint fields: "
                + ", ".join(conflicts)
            )
    return sealed


def _resolve_producer_input_abi(
    checkpoint: Path,
    *,
    checkpoint_sha256: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return _producer_input_abi_from_checkpoint(checkpoint, checkpoint_sha256)
    except _LegacyInputABIMetadataUnavailable as checkpoint_error:
        try:
            return _producer_input_abi_from_manifest(
                manifest,
                checkpoint_sha256,
                explicit_checkpoint_fields=checkpoint_error.explicit_input_fields,
            )
        except ReanalysisError as manifest_error:
            raise ReanalysisError(
                "cannot authenticate trajectory producer input ABI from checkpoint "
                f"or source manifest: checkpoint={checkpoint_error}; "
                f"manifest={manifest_error}"
            ) from manifest_error


def _assert_policy_catalog_compatible(
    producer_input_abi: Mapping[str, Any],
    target_input_abi: Mapping[str, Any],
) -> None:
    producer_action_size = int(producer_input_abi["action_size"])
    target_action_size = int(target_input_abi["action_size"])
    producer_mask_version = str(producer_input_abi["action_mask_version"])
    target_mask_version = str(target_input_abi["action_mask_version"])
    if (
        producer_action_size != target_action_size
        or producer_mask_version != target_mask_version
    ):
        raise ReanalysisError(
            "trajectory producer and target reanalyzer policy catalogs are "
            "incompatible: "
            f"producer action_size={producer_action_size}, "
            f"target action_size={target_action_size}, "
            f"producer action_mask_version={producer_mask_version!r}, "
            f"target action_mask_version={target_mask_version!r}. "
            "Refusing reanalysis until "
            "an explicit authenticated policy-ID mapping exists."
        )


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
    required = {
        "is_pool_game",
        "opponent_version",
        "opponent_tag",
        "opponent_checkpoint_md5",
    }
    missing = required - set(shard)
    if missing:
        raise ReanalysisError(
            "source shard lacks explicit producer-mirror provenance: "
            + ", ".join(sorted(missing))
        )
    if bool(_scalar(shard, "is_pool_game", row, True)):
        return False
    if int(_scalar(shard, "opponent_version", row, 0)) != -1:
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
        "target_policy_mask",
        "root_value_mask",
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
        and bool(_scalar(shard, "root_value_mask", row, False))
        and bool(np.asarray(shard["target_policy_mask"])[row].any())
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
    claim_auth_key: Path,
    runtime_attestation: Mapping[str, Any] | None = None,
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
    producer_input_abi = _resolve_producer_input_abi(
        trajectory_producer_checkpoint,
        checkpoint_sha256=producer_sha,
        manifest=manifest,
    )
    target_sha = _sha256(target_checkpoint)
    target_input_abi = _producer_input_abi_from_checkpoint(
        target_checkpoint,
        target_sha,
        binding_source="target_reanalyzer_checkpoint",
        checkpoint_role="target reanalyzer",
    )
    _assert_policy_catalog_compatible(producer_input_abi, target_input_abi)
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
    runtime = dict(runtime_attestation or _runtime_attestation())
    if runtime.get("runtime_sha256") != _value_sha256(
        {key: value for key, value in runtime.items() if key != "runtime_sha256"}
    ):
        raise ReanalysisError("runtime attestation semantic hash mismatch")
    key_sha = _sha256(claim_auth_key)
    if Path(claim_auth_key).stat().st_size < 32:
        raise ReanalysisError("claim authentication key must contain at least 32 bytes")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "source_manifest": {
            "path": str(Path(source_manifest).resolve()),
            "sha256": _sha256(source_manifest),
        },
        "trajectory_producer": {
            "checkpoint_path": str(Path(trajectory_producer_checkpoint).resolve()),
            "checkpoint_sha256": producer_sha,
            "input_abi": producer_input_abi,
        },
        "target_reanalyzer": {
            "checkpoint_path": str(Path(target_checkpoint).resolve()),
            "checkpoint_sha256": target_sha,
            "input_abi": target_input_abi,
        },
        "target_information_regime": TARGET_INFORMATION_REGIME_PUBLIC,
        "claim_auth_key_sha256": key_sha,
        "claim_auth_key_path": str(Path(claim_auth_key).resolve()),
        "runtime_attestation": runtime,
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
    trajectory_producer = plan["trajectory_producer"]
    expected_input_abi = _resolve_producer_input_abi(
        Path(trajectory_producer["checkpoint_path"]),
        checkpoint_sha256=str(trajectory_producer["checkpoint_sha256"]),
        manifest=manifest_payload,
    )
    if trajectory_producer.get("input_abi") != expected_input_abi:
        raise ReanalysisError(
            "plan trajectory producer input ABI does not match authenticated "
            "checkpoint/source-manifest metadata"
        )
    target_reanalyzer = plan["target_reanalyzer"]
    expected_target_input_abi = _producer_input_abi_from_checkpoint(
        Path(target_reanalyzer["checkpoint_path"]),
        str(target_reanalyzer["checkpoint_sha256"]),
        binding_source="target_reanalyzer_checkpoint",
        checkpoint_role="target reanalyzer",
    )
    if target_reanalyzer.get("input_abi") != expected_target_input_abi:
        raise ReanalysisError(
            "plan target reanalyzer input ABI does not match authenticated "
            "checkpoint metadata"
        )
    _assert_policy_catalog_compatible(expected_input_abi, expected_target_input_abi)
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
    runtime = plan.get("runtime_attestation")
    if not isinstance(runtime, dict) or runtime.get("runtime_sha256") != _value_sha256(
        {key: value for key, value in runtime.items() if key != "runtime_sha256"}
    ):
        raise ReanalysisError("runtime attestation hash mismatch")
    if _runtime_attestation() != runtime:
        raise ReanalysisError("runtime code/native-engine drift from sealed plan")
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
    *,
    shard: Mapping[str, np.ndarray],
    row: int,
    sequence: GameActionSequence,
    producer_input_abi: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    missing = RECONSTRUCTION_COLUMNS - set(shard)
    if missing:
        raise ReanalysisError(
            "source row lacks the complete public reconstruction surface: "
            + ", ".join(sorted(missing))
        )
    decision = int(_scalar(shard, "decision_index", row, -1))
    action_size = int(producer_input_abi["action_size"])
    history_enabled = bool(producer_input_abi["meaningful_public_history"])
    history_schema = str(
        producer_input_abi["meaningful_public_history_schema"]
    )
    history_limit = int(producer_input_abi["event_history_limit"])
    adapter_version = str(producer_input_abi["entity_feature_adapter_version"])
    result = round_trip_row(
        sequence,
        decision,
        _stored_features(shard, row),
        np.asarray(shard["legal_action_ids"])[row],
        correct_rust_chance_spectra=True,
        action_size=action_size,
        meaningful_public_history=history_enabled,
        meaningful_public_history_schema=history_schema,
        history_limit=history_limit,
        entity_feature_adapter_version=adapter_version,
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
        decision_indices=sequence.decision_indices,
        colors=COLORS,
        correct_rust_chance_spectra=True,
        action_size=action_size,
    )
    return game, featurize_state(
        game,
        colors=COLORS,
        action_size=action_size,
        meaningful_public_history=history_enabled,
        meaningful_public_history_schema=history_schema,
        history_limit=history_limit,
        entity_feature_adapter_version=adapter_version,
    )


def _search_patch(
    search: Any,
    game: Any,
    feature: Mapping[str, Any],
    *,
    target_input_abi: Mapping[str, Any],
) -> dict[str, Any]:
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
            action_size=int(target_input_abi["action_size"]),
        )
    )
    if mapped != tuple(int(value) for value in feature["legal_policy_ids"]):
        raise ReanalysisError(
            "legal action order changed between reconstruction and search"
        )
    if set(result.improved_policy) != set(legal_rust):
        raise ReanalysisError("search result does not cover the exact legal root")
    target = [float(result.improved_policy[action]) for action in legal_rust]
    # Coverage records that the teacher supplied a label for a legal action.
    # An exact zero is a valid soft target, not missing supervision.
    target_mask = [True] * len(target)
    raw_scores = [
        float(result.q_values.get(action, float("nan"))) for action in legal_rust
    ]
    score_mask = [bool(np.isfinite(value)) for value in raw_scores]
    # Claims are strict JSON (NaN is intentionally forbidden). Masked slots
    # carry a harmless zero and become NaN padding only beyond legal width at
    # merge time; consumers must consult target_scores_mask.
    scores = [value if valid else 0.0 for value, valid in zip(raw_scores, score_mask)]
    priors = [float(result.priors[action]) for action in legal_rust]
    if (
        not result.used_full_search
        or not np.isfinite(result.root_value)
        or not -1.0 <= float(result.root_value) <= 1.0
        or not np.isfinite(result.root_prior_value)
        or not -1.0 <= float(result.root_prior_value) <= 1.0
    ):
        raise ReanalysisError(
            "forced-full reanalysis returned invalid root search/prior value evidence"
        )
    if not np.isclose(sum(target), 1.0, atol=1e-5) or not np.isclose(
        sum(priors), 1.0, atol=1e-5
    ):
        raise ReanalysisError("search target/prior is not normalized")
    return {
        "target_policy": target,
        "target_policy_mask": target_mask,
        "target_scores": scores,
        "target_scores_mask": score_mask,
        "root_value": float(result.root_value),
        "root_value_mask": True,
        "root_prior_value": float(result.root_prior_value),
        "root_prior_value_mask": True,
        "prior_policy": priors,
        "simulations_used": int(result.simulations_used),
        "used_full_search": True,
    }


def _evaluator_from_plan(plan: Mapping[str, Any], *, device: str) -> Any:
    return EntityGraphRustEvaluator.from_checkpoint(
        plan["target_reanalyzer"]["checkpoint_path"],
        device=device,
        config=EntityGraphRustEvaluatorConfig(
            value_scale=1.0,
            prior_temperature=float(plan["search_config"]["prior_temperature"]),
            public_observation=True,
            rust_featurize=True,
        ),
    )


def _search_from_plan(plan: Mapping[str, Any], *, evaluator: Any, row_seed: int) -> Any:
    allowed = {field.name for field in dataclasses.fields(GumbelChanceMCTSConfig)}
    kwargs = {
        key: value for key, value in plan["search_config"].items() if key in allowed
    }
    kwargs["colors"] = tuple(kwargs.get("colors", COLORS))
    kwargs["seed"] = int(row_seed)
    config = GumbelChanceMCTSConfig(**kwargs)
    return create_gumbel_search(config, evaluator, native_hot_loop=False)


def run_chunk(
    *,
    plan: Mapping[str, Any],
    chunk_index: int,
    output: Path,
    claim_auth_key: Path,
    device: str = "cpu",
    search_factory: Any = None,
) -> dict[str, Any]:
    _verify_plan(plan)
    auth_key = _load_auth_key(claim_auth_key, str(plan["claim_auth_key_sha256"]))
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
    evaluator = (
        None
        if search_factory is not None
        else _evaluator_from_plan(plan, device=device)
    )
    patches: list[dict[str, Any]] = []
    for identity in entries:
        shard = loaded[int(identity["shard_index"])]
        row = int(identity["row_index"])
        if not _eligible_policy_row(shard, row):
            raise ReanalysisError("planned row is no longer policy-active/admissible")
        sequence = sequences[int(identity["game_seed"])]
        game, feature = _verify_reconstruction(
            shard=shard,
            row=row,
            sequence=sequence,
            producer_input_abi=plan["trajectory_producer"]["input_abi"],
        )
        row_seed = int(
            str(identity["identity_sha256"]).removeprefix("sha256:")[:16], 16
        )
        search = (
            search_factory(row_seed)
            if search_factory is not None
            else _search_from_plan(plan, evaluator=evaluator, row_seed=row_seed)
        )
        patch = _search_patch(
            search,
            game,
            feature,
            target_input_abi=plan["target_reanalyzer"]["input_abi"],
        )
        patches.append(
            {
                "identity_sha256": identity["identity_sha256"],
                "shard_index": identity["shard_index"],
                "row_index": identity["row_index"],
                "search_seed": row_seed,
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
        "runtime_attestation_sha256": plan["runtime_attestation"]["runtime_sha256"],
    }
    claim["claim_sha256"] = _value_sha256(claim)
    claim["claim_hmac_sha256"] = _claim_hmac(claim, auth_key)
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
    if set(values) != SEARCH_PATCH_COLUMNS:
        raise ReanalysisError(
            f"claim attempted wrong columns: got={sorted(values)}, expected={sorted(SEARCH_PATCH_COLUMNS)}"
        )
    for key, value in values.items():
        if key not in arrays:
            raise ReanalysisError(f"source shard lacks rewrite column {key}")
        arrays[key][row] = _coerce_row_value(arrays[key], row, value)


def _ensure_reanalysis_provenance_columns(
    arrays: dict[str, np.ndarray], plan: Mapping[str, Any]
) -> None:
    rows = _row_count(arrays)
    arrays.setdefault("root_prior_value", np.full(rows, np.nan, dtype=np.float32))
    arrays.setdefault("root_prior_value_mask", np.zeros(rows, dtype=np.bool_))
    for key, scalar in unaudited_target_reliability_fields().items():
        if key not in arrays:
            arrays[key] = np.full(rows, scalar, dtype=np.asarray(scalar).dtype)
    original_teacher = np.asarray(arrays.get("teacher_name", np.full(rows, ""))).astype(
        "U64"
    )
    arrays["teacher_name"] = original_teacher
    for key in (
        "trajectory_producer_checkpoint_sha256",
        "target_reanalyzer_checkpoint_sha256",
        "target_reanalysis_search_config_sha256",
        "target_reanalysis_plan_sha256",
    ):
        if key in arrays:
            raise ReanalysisError(
                f"source shard already contains reserved column {key}"
            )
        arrays[key] = np.full(rows, "", dtype="U71")


def _stamp_reanalysis_provenance(
    arrays: dict[str, np.ndarray], row: int, plan: Mapping[str, Any]
) -> None:
    arrays["teacher_name"][row] = "policy_target_reanalysis"
    arrays["trajectory_producer_checkpoint_sha256"][row] = plan["trajectory_producer"][
        "checkpoint_sha256"
    ]
    arrays["target_reanalyzer_checkpoint_sha256"][row] = plan["target_reanalyzer"][
        "checkpoint_sha256"
    ]
    arrays["target_reanalysis_search_config_sha256"][row] = plan["search_config_sha256"]
    arrays["target_reanalysis_plan_sha256"][row] = plan["plan_sha256"]
    for key, scalar in unaudited_target_reliability_fields().items():
        arrays[key][row] = scalar


def _columns_sha256(arrays: Mapping[str, np.ndarray], keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = np.asarray(arrays[key])
        digest.update(key.encode("utf-8") + b"\0")
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, value, allow_pickle=True)
        digest.update(buffer.getvalue())
    return "sha256:" + digest.hexdigest()


def _verify_claim(
    claim: Mapping[str, Any], plan: Mapping[str, Any], *, auth_key: bytes
) -> None:
    if (
        claim.get("schema_version") != CLAIM_SCHEMA
        or claim.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise ReanalysisError("foreign or unsupported chunk claim")
    unsigned = {
        key: value for key, value in claim.items() if key != "claim_hmac_sha256"
    }
    if not hmac.compare_digest(
        str(claim.get("claim_hmac_sha256", "")), _claim_hmac(unsigned, auth_key)
    ):
        raise ReanalysisError("chunk claim authentication failed")
    expected = _value_sha256(
        {
            key: value
            for key, value in claim.items()
            if key not in {"claim_sha256", "claim_hmac_sha256"}
        }
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
    if (
        claim.get("runtime_attestation_sha256")
        != plan["runtime_attestation"]["runtime_sha256"]
    ):
        raise ReanalysisError("chunk runtime attestation mismatch")
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
        row_seed = int(
            str(identity["identity_sha256"]).removeprefix("sha256:")[:16], 16
        )
        if int(patch.get("search_seed", -1)) != row_seed:
            raise ReanalysisError("chunk row search seed mismatch")


def merge_claims(
    *,
    plan: Mapping[str, Any],
    claim_paths: Sequence[Path],
    output: Path,
    claim_auth_key: Path,
) -> dict[str, Any]:
    _verify_plan(plan)
    auth_key = _load_auth_key(claim_auth_key, str(plan["claim_auth_key_sha256"]))
    claims = [_load_json(path) for path in claim_paths]
    for claim in claims:
        _verify_claim(claim, plan, auth_key=auth_key)
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
    if output.exists():
        raise ReanalysisError(f"merge output must not already exist: {output}")
    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    by_shard: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for patch in patches.values():
        by_shard[int(patch["shard_index"])].append(patch)
    payload_inventory: list[dict[str, Any]] = []
    output_shards: list[str] = []
    preservation_receipts: list[dict[str, Any]] = []
    for source in plan["source_shards"]:
        shard_index = int(source["index"])
        original = load_shard(Path(source["path"]))
        arrays = {key: value.copy() for key, value in original.items()}
        _ensure_reanalysis_provenance_columns(arrays, plan)
        # Compact search evidence belongs to the exact checkpoint/search that
        # produced it. Until this operator rebuilds that ragged bundle from the
        # new SearchResult, remove it instead of pairing new targets with stale Q
        # and visit evidence. The empirical quality gate then fails closed.
        for key in (
            "search_evidence_version",
            "search_evidence_offsets",
            "search_visit_counts_flat",
            "search_completed_q_flat",
            "search_prior_policy_flat",
        ):
            arrays.pop(key, None)
        before = {
            key: value.copy()
            for key, value in original.items()
            if key not in REWRITTEN_COLUMNS
        }
        for patch in sorted(
            by_shard.get(shard_index, []), key=lambda value: int(value["row_index"])
        ):
            _apply_patch(arrays, int(patch["row_index"]), patch["values"])
            _stamp_reanalysis_provenance(arrays, int(patch["row_index"]), plan)
        for key, expected in before.items():
            if not _array_equal(arrays[key], expected):
                raise ReanalysisError(f"non-target column changed during merge: {key}")
        preserved_sha = _columns_sha256(original, before)
        if _columns_sha256(arrays, before) != preserved_sha:
            raise ReanalysisError("preserved-column semantic digest changed")
        preservation_receipts.append(
            {
                "shard_index": shard_index,
                "source_sha256": source["sha256"],
                "preserved_columns_sha256": preserved_sha,
            }
        )
        destination = staging / f"reanalyzed_shard_{shard_index:05d}.npz"
        _write_npz_atomic(destination, arrays)
        record = {
            "path": destination.name,
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
            "rows": _row_count(arrays),
        }
        payload_inventory.append(record)
        output_shards.append(record["path"])
    plan_path = staging / "plan.json"
    _write_json_atomic(plan_path, plan)
    row_identities = [
        {
            "game_seed": row["game_seed"],
            "decision_index": row["decision_index"],
            "shard_index": row["shard_index"],
            "row_index": row["row_index"],
        }
        for row in plan["eligible_rows"]
    ]
    manifest = {
        "schema_version": MERGE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "plan": {"path": plan_path.name, "file_sha256": _sha256(plan_path)},
        "trajectory_producer": plan["trajectory_producer"],
        "target_reanalyzer": plan["target_reanalyzer"],
        "target_information_regime": TARGET_INFORMATION_REGIME_PUBLIC,
        "search_config": plan["search_config"],
        "search_config_sha256": plan["search_config_sha256"],
        "rewritten_columns": sorted(REWRITTEN_COLUMNS),
        "reanalyzed_rows": len(patches),
        "search_evidence_invalidated": True,
        "shards": output_shards,
        "payload_inventory_schema": PAYLOAD_INVENTORY_SCHEMA,
        "payload_inventory": payload_inventory,
        "payload_inventory_sha256": _value_sha256(payload_inventory),
        "preservation_receipts": preservation_receipts,
        "preserved_columns_sha256": _value_sha256(preservation_receipts),
        "row_identity_sha256": _value_sha256(row_identities),
        "runtime_attestation": plan["runtime_attestation"],
        "claim_sha256s": [
            by_chunk[index]["claim_sha256"] for index in sorted(by_chunk)
        ],
    }
    manifest["manifest_sha256"] = _value_sha256(manifest)
    manifest["manifest_hmac_sha256"] = _claim_hmac(manifest, auth_key)
    _write_json_atomic(staging / "manifest.json", manifest)
    try:
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _command_plan(args: argparse.Namespace) -> None:
    config = default_search_config(seed=args.seed, n_full=args.n_full)
    plan = build_plan(
        source_manifest=Path(args.source_manifest),
        trajectory_producer_checkpoint=Path(args.trajectory_producer_checkpoint),
        target_checkpoint=Path(args.target_checkpoint),
        chunks=args.chunks,
        search_config=config,
        claim_auth_key=Path(args.claim_auth_key),
    )
    _write_json_atomic(Path(args.output), plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


def _command_chunk(args: argparse.Namespace) -> None:
    claim = run_chunk(
        plan=_load_json(Path(args.plan)),
        chunk_index=args.chunk_index,
        output=Path(args.output),
        claim_auth_key=Path(args.claim_auth_key),
        device=args.device,
    )
    print(json.dumps(claim, indent=2, sort_keys=True))


def _command_merge(args: argparse.Namespace) -> None:
    manifest = merge_claims(
        plan=_load_json(Path(args.plan)),
        claim_paths=[Path(path) for path in args.claim],
        output=Path(args.output),
        claim_auth_key=Path(args.claim_auth_key),
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
    plan.add_argument("--claim-auth-key", required=True)
    plan.set_defaults(func=_command_plan)
    chunk = commands.add_parser("run-chunk")
    chunk.add_argument("--plan", required=True)
    chunk.add_argument("--chunk-index", type=int, required=True)
    chunk.add_argument("--device", default="cuda")
    chunk.add_argument("--output", required=True)
    chunk.add_argument("--claim-auth-key", required=True)
    chunk.set_defaults(func=_command_chunk)
    merge = commands.add_parser("merge")
    merge.add_argument("--plan", required=True)
    merge.add_argument("--claim", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--claim-auth-key", required=True)
    merge.set_defaults(func=_command_merge)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
