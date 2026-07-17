#!/usr/bin/env python3
"""Build the exact fresh-only post-wave training composite.

The post-wave audit authorizes whole games, not whole shard files.  This tool
materializes three source-pure fresh components by filtering every audited NPZ
on the signed ``(job_id, category, game_seed)`` selection before memmap
expansion and emits the promotion-eligible .80/.15/.05 descriptor consumed by
``train_bc``.

The resulting tree is host-portable at an identical canonical install path.
Absolute paths are deliberately authenticated; transfer tooling must rsync the
whole tree to the same path on each learner rather than silently rebasing it.

This is intentionally a builder only.  It never launches generation or a
learner and it refuses an existing/non-empty output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catan_zero.rl.flywheel.composite_contract import (  # noqa: E402
    FRESH_SOURCE_GAME_RATIOS,
    build_sampling_receipt,
    canonical_sha256,
    measure_memmap_component,
)
from catan_zero.rl.aux_subgoal_targets import (  # noqa: E402
    AUX_SUBGOAL_TARGET_SEMANTIC,
    AUX_SUBGOAL_TARGET_VERSION,
    AUX_SUBGOAL_TARGET_VERSION_KEY,
)
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    require_known_entity_feature_adapter,
)
from tools import a1_pre_wave_contract as contract  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_frozen_lock_verifier as frozen_lock_verifier  # noqa: E402
from tools import a1_target_eligibility_inventory as target_inventory  # noqa: E402
from tools import build_memmap_corpus as memmap_builder  # noqa: E402
from tools import train_bc  # noqa: E402


SOURCE_AUTHORITY_SCHEMA = "a1-post-wave-composite-source-authority-v4"
BUILD_RECEIPT_SCHEMA = "a1-post-wave-composite-build-v3"
EFFECTIVE_COMPONENT_RATIOS = {
    "current_producer": 0.80,
    "recent_history": 0.15,
    "hard_negative": 0.05,
}
# The fresh rows in this recovery wave are all produced by the same n128
# search teacher.  The winning TEMP experiment established the n128 policy
# target at T=1.0.  ``soft_target_temperature=0.7`` is deliberately inert for
# stored-policy targets, so bind the source temperatures in the descriptor
# instead of relying on that easy-to-misread global score-target flag.
STORED_POLICY_COMPONENT_TEMPERATURES = {
    "current_producer": 1.0,
    "recent_history": 1.0,
    "hard_negative": 1.0,
}

# The production baseline trains policy only from fresh, same-operator n128
# components and value only from current-producer self-play.
POLICY_KL_ANCHOR_WEIGHT = 0.0
_CURRENT_LEARNER_RECIPE = current_science.learner_training_recipe()
_CURRENT_PER_GAME_VALUE_WEIGHT = _CURRENT_LEARNER_RECIPE.get(
    "per_game_value_weight"
)
_CURRENT_PER_GAME_VALUE_WEIGHT_MODE = _CURRENT_LEARNER_RECIPE.get(
    "per_game_value_weight_mode", "equal"
)
_CURRENT_VALUE_PLAYER_OUTCOME_BALANCE_MODE = _CURRENT_LEARNER_RECIPE.get(
    "value_player_outcome_balance_mode", "none"
)
if (
    _CURRENT_PER_GAME_VALUE_WEIGHT is not True
    or _CURRENT_PER_GAME_VALUE_WEIGHT_MODE != "equal"
    or _CURRENT_VALUE_PLAYER_OUTCOME_BALANCE_MODE != "none"
):
    raise RuntimeError(
        "current science contract must bind equal per-game value weighting and "
        "the natural terminal-outcome prior"
    )


def _single_adapter_version(values: Sequence[object], *, source: str) -> str:
    """Resolve one nonempty, registry-known adapter from authenticated bytes."""

    normalized = {str(value or "") for value in values}
    if "" in normalized or len(normalized) != 1:
        raise CompositeBuildError(
            f"{source} does not bind exactly one nonempty entity adapter: "
            f"{sorted(normalized)}"
        )
    version = next(iter(normalized))
    try:
        return require_known_entity_feature_adapter(version)
    except ValueError as error:
        raise CompositeBuildError(
            f"{source} binds an unknown entity adapter {version!r}"
        ) from error


def _memmap_adapter_version(corpus_dir: Path, *, component_id: str) -> str:
    """Read the adapter identity preserved by one freshly built memmap."""

    try:
        meta = _load_json(corpus_dir / "corpus_meta.json")
        schema = meta["columns"]["adapter_version"]
        categories = schema["categories"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise CompositeBuildError(
            f"cannot inspect {component_id} memmap adapter identity: {error}"
        ) from error
    if schema.get("kind") != "string" or not isinstance(categories, list):
        raise CompositeBuildError(
            f"fresh component {component_id} lost adapter_version during conversion"
        )
    return _single_adapter_version(
        categories, source=f"fresh component {component_id}"
    )


LEARNER_RECIPE_OVERRIDES: dict[str, object] = {
    "forced_action_weight": 0.0,
    "forced_row_value_weight": 1.0,
    "loser_sample_weight": 1.0,
    "per_game_policy_weight": True,
    "per_game_policy_weight_mode": "equal",
    "per_game_value_weight": _CURRENT_PER_GAME_VALUE_WEIGHT,
    "per_game_value_weight_mode": _CURRENT_PER_GAME_VALUE_WEIGHT_MODE,
    "value_player_outcome_balance_mode": (
        _CURRENT_VALUE_PLAYER_OUTCOME_BALANCE_MODE
    ),
    "policy_kl_anchor_direction": "forward",
    "policy_kl_anchor_weight": POLICY_KL_ANCHOR_WEIGHT,
    "policy_loss_weight": 1.0,
    "q_loss_weight": 0.0,
    "soft_target_source": "policy",
    "soft_target_temperature": 0.7,
    "soft_target_weight": 1.0,
    "policy_target_blend_semantics": "policy_target_fallback_v2",
    "truncated_vp_margin_value_weight": 0.25,
    "value_target_lambda": 1.0,
}


class CompositeBuildError(RuntimeError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _artifact_ref(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise CompositeBuildError(f"authority artifact is not a file: {resolved}")
    return {"path": str(resolved), "file_sha256": _file_sha256(resolved)}


def _validated_lock_verifier_authority(
    raw: Mapping[str, Any],
    *,
    lock_path: Path,
    lock: Mapping[str, Any],
    require_all_job_claims: bool,
) -> dict[str, Any]:
    """Bind one exact frozen verifier invocation to one exact lock.

    The frozen-verifier helper already executes the verifier.  This second,
    local check prevents a caller from pairing that result with the other
    generation's lock or silently changing the job-claim completeness mode
    before the composite source authority is hashed.
    """

    expected = {
        "schema_version",
        "lock",
        "lock_file_sha256",
        "contract_sha256",
        "frozen_repo",
        "verifier",
        "verifier_sha256",
        "require_all_job_claims",
        "verified_lock_sha256",
        "authority_sha256",
    }
    authority = dict(raw)
    if (
        set(authority) != expected
        or authority.get("schema_version")
        != frozen_lock_verifier.AUTHORITY_SCHEMA
    ):
        raise CompositeBuildError("lock-verifier authority fields/schema drift")
    unhashed = dict(authority)
    declared = unhashed.pop("authority_sha256", None)
    if declared != _digest(unhashed):
        raise CompositeBuildError("lock-verifier authority digest drift")
    try:
        resolved_lock = lock_path.expanduser().resolve(strict=True)
        frozen_repo = Path(str(authority["frozen_repo"])).resolve(strict=True)
        verifier = Path(str(authority["verifier"])).resolve(strict=True)
    except OSError as error:
        raise CompositeBuildError(
            f"lock-verifier authority path is unavailable: {error}"
        ) from error
    if (
        authority["lock"] != str(resolved_lock)
        or authority["lock_file_sha256"] != _file_sha256(resolved_lock)
        or authority["contract_sha256"] != lock.get("contract_sha256")
        or authority["verified_lock_sha256"] != _digest(lock)
        or authority["require_all_job_claims"] is not require_all_job_claims
        or not frozen_repo.is_dir()
        or verifier != frozen_repo / frozen_lock_verifier.VERIFIER_RELATIVE_PATH
        or not verifier.is_file()
        or authority["verifier_sha256"] != _file_sha256(verifier)
    ):
        raise CompositeBuildError("lock-verifier authority/lock binding drift")
    return authority


def _binding_source_id(binding: Mapping[str, Any]) -> str:
    return _digest(dict(binding))


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompositeBuildError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise CompositeBuildError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Durably copy one immutable authority artifact without partial visibility."""

    source = source.expanduser().resolve(strict=True)
    if not source.is_file():
        raise CompositeBuildError(f"authority artifact is not a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


_SEARCH_EVIDENCE_FLAT_COLUMNS = (
    "search_visit_counts_flat",
    "search_completed_q_flat",
)
_SEARCH_EVIDENCE_PRIOR_COLUMN = "search_prior_policy_flat"
_SEARCH_EVIDENCE_COLUMNS = {
    "search_evidence_version",
    "search_evidence_offsets",
    *_SEARCH_EVIDENCE_FLAT_COLUMNS,
    _SEARCH_EVIDENCE_PRIOR_COLUMN,
}


def _filter_selected_source_arrays(
    payload: Mapping[str, np.ndarray],
    *,
    selected_mask: np.ndarray,
    rows: int,
    source: Path,
) -> dict[str, np.ndarray]:
    """Filter row data and its policy-active compact search evidence together.

    Search evidence is deliberately not row aligned: offsets index only rows
    with positive policy weight, while the flat arrays concatenate the legal
    actions for those active rows.  Filtering physical rows therefore requires
    rebuilding that compact index rather than slicing or copying it unchanged.
    """

    keep = np.asarray(selected_mask, dtype=bool)
    if keep.shape != (rows,):
        raise CompositeBuildError(
            f"selected mask is not row-aligned for {source}: {keep.shape}"
        )
    names = set(payload)
    evidence_names = names.intersection(_SEARCH_EVIDENCE_COLUMNS)
    ragged: dict[str, np.ndarray] = {}
    if evidence_names:
        required = {
            "search_evidence_version",
            "search_evidence_offsets",
            *_SEARCH_EVIDENCE_FLAT_COLUMNS,
        }
        if not required.issubset(names):
            missing = sorted(required - names)
            raise CompositeBuildError(
                f"incomplete compact search evidence in {source}: missing {missing}"
            )
        version_array = np.asarray(payload["search_evidence_version"])
        if version_array.ndim != 0:
            raise CompositeBuildError(
                f"search_evidence_version is not scalar in {source}"
            )
        version = int(version_array)
        if version not in (1, 2):
            raise CompositeBuildError(
                f"unsupported search_evidence_version={version} in {source}"
            )
        has_prior = _SEARCH_EVIDENCE_PRIOR_COLUMN in names
        if has_prior != (version == 2):
            requirement = "requires" if version == 2 else "forbids"
            raise CompositeBuildError(
                f"search evidence v{version} {requirement} "
                f"{_SEARCH_EVIDENCE_PRIOR_COLUMN} in {source}"
            )
        if "policy_weight_multiplier" not in names:
            raise CompositeBuildError(
                f"compact search evidence lacks policy weights in {source}"
            )
        weights = np.asarray(payload["policy_weight_multiplier"])
        if weights.ndim < 1 or weights.shape[0] != rows:
            raise CompositeBuildError(
                f"policy_weight_multiplier is not row-aligned in {source}"
            )
        active_rows = np.flatnonzero(weights > 0.0)
        offsets = np.asarray(payload["search_evidence_offsets"])
        flat_names = [*_SEARCH_EVIDENCE_FLAT_COLUMNS]
        if has_prior:
            flat_names.append(_SEARCH_EVIDENCE_PRIOR_COLUMN)
        flat_arrays = {name: np.asarray(payload[name]) for name in flat_names}
        if (
            offsets.ndim != 1
            or not np.issubdtype(offsets.dtype, np.integer)
            or offsets.shape != (active_rows.size + 1,)
            or offsets.size == 0
            or int(offsets[0]) != 0
            or np.any(np.diff(offsets.astype(np.int64, copy=False)) < 0)
            or any(values.ndim != 1 for values in flat_arrays.values())
        ):
            raise CompositeBuildError(
                f"malformed compact search evidence index in {source}"
            )
        flat_size = int(offsets[-1])
        if any(values.size != flat_size for values in flat_arrays.values()):
            raise CompositeBuildError(
                f"compact search evidence flat lengths disagree in {source}"
            )

        kept_active_indices = np.flatnonzero(keep[active_rows])
        lengths = np.diff(offsets.astype(np.int64, copy=False))
        kept_lengths = lengths[kept_active_indices]
        new_offsets = np.empty(kept_active_indices.size + 1, dtype=offsets.dtype)
        new_offsets[0] = 0
        if kept_lengths.size:
            np.cumsum(kept_lengths, dtype=np.int64, out=new_offsets[1:])
        segments = [
            slice(int(offsets[index]), int(offsets[index + 1]))
            for index in kept_active_indices
        ]
        ragged["search_evidence_version"] = version_array
        ragged["search_evidence_offsets"] = new_offsets
        for name, values in flat_arrays.items():
            ragged[name] = (
                np.concatenate([values[segment] for segment in segments])
                if segments
                else np.empty(0, dtype=values.dtype)
            )

    filtered: dict[str, np.ndarray] = {}
    for name in payload:
        values = np.asarray(payload[name])
        if name in ragged:
            filtered[name] = ragged[name]
        elif name in _SEARCH_EVIDENCE_COLUMNS:
            raise CompositeBuildError(
                f"unhandled compact search evidence column {name!r} in {source}"
            )
        elif values.ndim > 0 and values.shape[0] == rows:
            filtered[name] = values[keep]
        elif values.ndim == 0:
            filtered[name] = values
        else:
            raise CompositeBuildError(
                f"unknown non-row-aligned source column {name!r} in {source}"
            )
    return filtered


def _prepare_output_root(path: Path) -> Path:
    root = path.expanduser().absolute()
    if root.exists():
        if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
            raise CompositeBuildError(
                f"output root must be absent or an empty real directory: {root}"
            )
    else:
        root.mkdir(parents=True)
    if root.resolve(strict=True) != root:
        raise CompositeBuildError(f"output root is not canonical: {root}")
    return root


def _finalize_component_payloads_read_only(
    components: Sequence[Mapping[str, Any]],
) -> None:
    """Seal every inventory payload before publishing the composite descriptor.

    The descriptor is the composite's atomic publication boundary.  Keep it
    absent until every referenced memmap payload has been opened without
    following symlinks, changed to exactly ``0444`` through that open inode,
    durably synced, and re-authenticated against the existing byte inventory.
    Chmod does not alter corpus metadata or payload bytes, so all pre-existing
    hashes and descriptor semantics remain unchanged.  Holding every file
    descriptor through the final hash pass also makes pathname replacement a
    hard failure rather than accidentally sealing a different inode.

    A failed seal may leave an incomplete build tree containing read-only
    payloads, but it can never publish ``memmap_composite.json``.  This is the
    intended fail-closed state; the builder already refuses to reuse a nonempty
    output root.
    """

    opened: list[tuple[str, Path, int, tuple[int, int, int]]] = []
    corpora: list[tuple[str, Path, Path, dict[str, Any], str]] = []
    seen_roots: set[Path] = set()
    try:
        for component in components:
            component_id = str(component.get("component_id", ""))
            try:
                corpus_dir = Path(str(component["corpus_dir"])).resolve(strict=True)
            except (KeyError, OSError) as error:
                raise CompositeBuildError(
                    f"cannot resolve {component_id or 'unnamed'} payload corpus: {error}"
                ) from error
            if not component_id or corpus_dir in seen_roots:
                raise CompositeBuildError("component payload corpus identity is ambiguous")
            seen_roots.add(corpus_dir)

            meta_path = corpus_dir / "corpus_meta.json"
            meta = _load_json(meta_path)
            meta_sha = _file_sha256(meta_path)
            inventory = meta.get("payload_inventory")
            if (
                meta_sha != component.get("corpus_meta_sha256")
                or not isinstance(inventory, list)
                or not inventory
                or train_bc._canonical_json_sha256(inventory)  # noqa: SLF001
                != meta.get("payload_inventory_sha256")
                or meta.get("payload_inventory_sha256")
                != component.get("payload_inventory_sha256")
            ):
                raise CompositeBuildError(
                    f"{component_id} payload inventory binding drift"
                )
            try:
                expected_names = sorted(
                    train_bc._expected_memmap_payload_filenames(meta)  # noqa: SLF001
                )
            except SystemExit as error:
                raise CompositeBuildError(
                    f"{component_id} payload schema is not finalizable: {error}"
                ) from error
            inventory_names = [
                record.get("filename") if isinstance(record, Mapping) else None
                for record in inventory
            ]
            if inventory_names != expected_names or any(
                not isinstance(name, str) or Path(name).name != name
                for name in inventory_names
            ):
                raise CompositeBuildError(
                    f"{component_id} payload inventory differs from its column schema"
                )

            for filename in expected_names:
                path = corpus_dir / filename
                descriptor = -1
                try:
                    path_stat = path.lstat()
                    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(
                        path_stat.st_mode
                    ):
                        raise OSError("not a non-symlink regular file")
                    descriptor = os.open(
                        path,
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    )
                    opened_stat = os.fstat(descriptor)
                except OSError as error:
                    if descriptor >= 0:
                        os.close(descriptor)
                    raise CompositeBuildError(
                        f"cannot bind {component_id} payload {path}: {error}"
                    ) from error
                identity = (
                    int(opened_stat.st_dev),
                    int(opened_stat.st_ino),
                    int(opened_stat.st_size),
                )
                if identity != (
                    int(path_stat.st_dev),
                    int(path_stat.st_ino),
                    int(path_stat.st_size),
                ):
                    os.close(descriptor)
                    raise CompositeBuildError(
                        f"{component_id} payload changed while opening: {path}"
                    )
                opened.append((component_id, path, descriptor, identity))
            corpora.append((component_id, corpus_dir, meta_path, meta, meta_sha))

        errors: list[str] = []
        for component_id, path, descriptor, _identity in opened:
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            except OSError as error:
                # Attempt every sibling even after one failure; no descriptor
                # is published unless the complete set succeeds.
                errors.append(f"{component_id}:{path.name}: {error}")
        if errors:
            raise CompositeBuildError(
                "payload read-only finalization failed: " + "; ".join(errors)
            )

        for component_id, corpus_dir, meta_path, meta, meta_sha in corpora:
            try:
                authenticated = train_bc._validate_memmap_payload_inventory(  # noqa: SLF001
                    corpus_dir, meta
                )
            except (OSError, SystemExit, ValueError) as error:
                raise CompositeBuildError(
                    f"{component_id} finalized payload authentication failed: {error}"
                ) from error
            if (
                authenticated != meta["payload_inventory_sha256"]
                or _file_sha256(meta_path) != meta_sha
            ):
                raise CompositeBuildError(
                    f"{component_id} payload finalization changed bound bytes"
                )

        # Authentication re-opened the pathnames. Match them back to the
        # still-open inodes and exact final mode before publishing a descriptor.
        for component_id, path, descriptor, identity in opened:
            opened_stat = os.fstat(descriptor)
            path_stat = path.lstat()
            if (
                (
                    int(opened_stat.st_dev),
                    int(opened_stat.st_ino),
                    int(opened_stat.st_size),
                )
                != identity
                or (
                    int(path_stat.st_dev),
                    int(path_stat.st_ino),
                    int(path_stat.st_size),
                )
                != identity
                or stat.S_IMODE(opened_stat.st_mode) != 0o444
                or stat.S_IMODE(path_stat.st_mode) != 0o444
            ):
                raise CompositeBuildError(
                    f"{component_id} payload did not finalize read-only: {path}"
                )
    finally:
        for _component_id, _path, descriptor, _identity in opened:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _validated_wave_inputs(
    lock_path: Path,
    selected_path: Path,
    audit_path: Path,
    *,
    verify_lock_fn: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        lock = verify_lock_fn(lock_path, require_all_job_claims=True)
        locked_counts = lock.get("game_contract", {}).get("category_games")
        if (
            not isinstance(locked_counts, dict)
            or set(locked_counts) != set(FRESH_SOURCE_GAME_RATIOS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in locked_counts.values()
            )
        ):
            raise contract.ContractError(
                "sealed wave lock lacks valid category-game quota authority"
            )
        selected = memmap_builder._load_a1_selected_game_manifest(  # noqa: SLF001
            selected_path,
            expected_selected_game_count=sum(locked_counts.values()),
            expected_category_game_counts=locked_counts,
        )
        audit = memmap_builder._load_a1_post_wave_audit(  # noqa: SLF001
            audit_path, selected
        )
    except (OSError, SystemExit, contract.ContractError) as error:
        raise CompositeBuildError(f"wave input verification failed: {error}") from error
    if selected["a1_contract_sha256"] != lock["contract_sha256"]:
        raise CompositeBuildError("selected-game manifest binds a different lock")
    if audit["contract_sha256"] != lock["contract_sha256"]:
        raise CompositeBuildError("post-wave audit binds a different lock")
    raw_selected = _load_json(Path(selected["path"]))
    if raw_selected.get("records_sha256") != selected["records_sha256"]:
        raise CompositeBuildError("selected-game record digest drift")
    raw_audit = _load_json(Path(audit["path"]))
    try:
        search_operator = lock["science"]["search_operator"]
        # The production decision taxonomy deterministically routes wide and
        # mandatory choices to full search before the p_full draw.  This is
        # independent of the retired n_full_wide budget override.  Mirror the
        # post-wave auditor exactly so a valid adaptive-complete activation
        # report remains replayable during composite materialization.
        wide_full_threshold = int(contract.WIDE_CHOICE_MIN_LEGAL_ACTIONS)
        audit["target_activation"] = contract._validate_target_activation_report(  # noqa: SLF001
            raw_audit.get("target_activation"),
            categories=tuple(lock["game_contract"]["category_games"]),
            sealed_p_full=float(search_operator["p_full"]),
            wide_full_threshold=wide_full_threshold,
        )
    except (KeyError, TypeError, ValueError, contract.ContractError) as error:
        raise CompositeBuildError(
            f"post-wave target-activation authority failed: {error}"
        ) from error
    return lock, selected, audit, raw_selected


def _selection_by_job(
    lock: Mapping[str, Any],
    raw_selected: Mapping[str, Any],
    *,
    expected_games: Mapping[str, int],
) -> tuple[dict[str, set[int]], dict[int, tuple[str, str]], list[dict[str, Any]]]:
    jobs = {str(job["job_id"]): job for job in lock["fleet"]["jobs"]}
    producer = contract._producer(dict(lock))  # noqa: SLF001
    selections: dict[str, set[int]] = defaultdict(set)
    owners: dict[int, tuple[str, str]] = {}
    records = raw_selected.get("records")
    if not isinstance(records, list):
        raise CompositeBuildError("selected-game manifest has no record list")
    normalized: list[dict[str, Any]] = []
    for record in records:
        job_id = str(record.get("job_id", ""))
        category = str(record.get("category", ""))
        seed = record.get("game_seed")
        job = jobs.get(job_id)
        expected_semantic = (
            None
            if job is None or category != job.get("category")
            else contract._sealed_category_semantic(lock, category)  # noqa: SLF001
        )
        semantic_matches = (
            "category_semantic" not in record
            if expected_semantic is None
            else record.get("category_semantic") == expected_semantic
        )
        if (
            job is None
            or category != job.get("category")
            or not semantic_matches
            or record.get("worker_id") != job.get("worker_id")
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not int(job["base_seed"]) <= seed < int(job["seed_end"])
            or record.get("producer_checkpoint_sha256") != producer["sha256"]
            or record.get("opponent_checkpoint_sha256")
            != contract._category_opponent_sha256(dict(lock), category)  # noqa: SLF001
        ):
            raise CompositeBuildError(
                f"selected game does not bind its sealed job/category: {record!r}"
            )
        if seed in owners or seed in selections[job_id]:
            raise CompositeBuildError(f"selected game seed is duplicated: {seed}")
        owners[seed] = (job_id, category)
        selections[job_id].add(seed)
        normalized.append(dict(record))
    counts = Counter(record["category"] for record in normalized)
    if dict(counts) != dict(expected_games):
        raise CompositeBuildError(
            f"selected category quotas differ: actual={dict(counts)} "
            f"expected={dict(expected_games)}"
        )
    return dict(selections), owners, normalized


_SOURCE_BINDING_FIELDS = {
    "source_id",
    "contract_sha256",
    "audit_file_sha256",
    "audit_sha256",
    "selected_manifest_file_sha256",
    "selected_records_sha256",
    "job_id",
    "category",
    "source_path",
    "source_sha256",
    "generation_manifest_path",
    "generation_manifest_sha256",
}
_TARGET_ACTIVATION_BINDING_FIELDS = {
    "target_activation_chunk_sha256",
    "target_activation_counts_sha256",
}


def _validate_source_bindings(
    bindings: Any,
    *,
    lock: Mapping[str, Any],
    selected_file_sha256: str,
    selected_records_sha256: str,
    audit_file_sha256: str,
    audit_sha256: str,
    require_target_activation: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(bindings, list) or not bindings:
        raise CompositeBuildError("source authority has no source bindings")
    jobs = {str(job["job_id"]): job for job in lock["fleet"]["jobs"]}
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(bindings):
        if not isinstance(raw, dict):
            raise CompositeBuildError(
                f"source authority binding {index} fields differ from schema"
            )
        category = str(raw.get("category", ""))
        job_id = str(raw.get("job_id", ""))
        job = jobs.get(job_id)
        expected_semantic = (
            None
            if job is None or category != job.get("category")
            else contract._sealed_category_semantic(lock, category)  # noqa: SLF001
        )
        expected_fields = set(_SOURCE_BINDING_FIELDS)
        if require_target_activation:
            expected_fields.update(_TARGET_ACTIVATION_BINDING_FIELDS)
        if expected_semantic is not None:
            expected_fields.add("category_semantic")
        if set(raw) != expected_fields:
            raise CompositeBuildError(
                f"source authority binding {index} fields differ from schema"
            )
        value = dict(raw)
        source_id = value.pop("source_id")
        job_id = str(value.get("job_id", ""))
        category = str(value.get("category", ""))
        job = jobs.get(job_id)
        if (
            not isinstance(source_id, str)
            or source_id in seen_ids
            or source_id != _binding_source_id(value)
            or job is None
            or category != job.get("category")
            or (
                value.get("category_semantic") != expected_semantic
                if expected_semantic is not None
                else "category_semantic" in value
            )
            or value.get("contract_sha256") != lock.get("contract_sha256")
            or value.get("selected_manifest_file_sha256")
            != selected_file_sha256
            or value.get("selected_records_sha256") != selected_records_sha256
            or value.get("audit_file_sha256") != audit_file_sha256
            or value.get("audit_sha256") != audit_sha256
        ):
            raise CompositeBuildError(
                f"source authority binding {index} identity/digest drift"
            )
        try:
            source = Path(str(value["source_path"])).expanduser().resolve(strict=True)
            generation_manifest = Path(
                str(value["generation_manifest_path"])
            ).expanduser().resolve(strict=True)
        except OSError as error:
            raise CompositeBuildError(
                f"source authority binding {index} artifact is missing: {error}"
            ) from error
        if (
            str(source) != value["source_path"]
            or str(generation_manifest) != value["generation_manifest_path"]
            or _file_sha256(source) != value["source_sha256"]
            or _file_sha256(generation_manifest)
            != value["generation_manifest_sha256"]
        ):
            raise CompositeBuildError(
                f"source authority binding {index} artifact bytes drifted"
            )
        seen_ids.add(source_id)
        normalized.append({"source_id": source_id, **value})
    return normalized


def _filter_wave_shards(
    *,
    lock: dict[str, Any],
    selected: dict[str, Any],
    audit: dict[str, Any],
    raw_selected: dict[str, Any],
    output_root: Path,
    expected_games: Mapping[str, int],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    selected_by_job, owner_by_seed, _records = _selection_by_job(
        lock, raw_selected, expected_games=expected_games
    )
    jobs = {str(job["job_id"]): job for job in lock["fleet"]["jobs"]}
    checkpoint_by_id = {str(record["id"]): record for record in lock["checkpoints"]}
    category_specs = {
        str(record["name"]): record for record in lock["source_categories"]
    }
    selfplay_colors = tuple(contract._expected_selfplay_config(lock)["colors"])  # noqa: SLF001
    producer = contract._producer(lock)  # noqa: SLF001
    producer_path = Path(str(producer["path"])).expanduser().resolve(strict=True)
    if _file_sha256(producer_path) != producer["sha256"]:
        raise CompositeBuildError("current producer checkpoint bytes drifted")

    audited_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in audit["data_shards"]:
        job_id = str(record.get("job_id", ""))
        category = str(record.get("category", ""))
        if job_id not in jobs or category != jobs[job_id].get("category"):
            raise CompositeBuildError("audited shard has unknown job/category")
        audited_by_job[job_id].append(record)
    if set(audited_by_job) != set(selected_by_job):
        raise CompositeBuildError(
            "audited shard jobs differ from selected-game jobs: "
            f"missing={sorted(set(selected_by_job) - set(audited_by_job))} "
            f"unexpected={sorted(set(audited_by_job) - set(selected_by_job))}"
        )

    raw_audit = _load_json(Path(str(audit["path"])))
    generation_manifest_by_job: dict[str, dict[str, Any]] = {}
    public_award_by_category: dict[str, dict[str, Any]] = {}
    for record in raw_audit.get("shards", []):
        if not isinstance(record, dict) or record.get("kind") != "generation_manifest":
            continue
        job_id = str(record.get("job_id", ""))
        category = str(record.get("category", ""))
        if job_id not in jobs or category != jobs[job_id].get("category"):
            raise CompositeBuildError(
                "audited generation manifest has unknown job/category"
            )
        manifest_path = Path(str(record.get("path", ""))).resolve(strict=True)
        if _file_sha256(manifest_path) != record.get("sha256"):
            raise CompositeBuildError(
                f"audited generation manifest bytes drifted: {manifest_path}"
            )
        manifest = _load_json(manifest_path)
        public_award = manifest.get("public_award_feature_provenance")
        if not isinstance(public_award, dict):
            raise CompositeBuildError(
                f"audited generation manifest lacks public-award provenance: {manifest_path}"
            )
        prior = public_award_by_category.setdefault(category, dict(public_award))
        if prior != public_award:
            raise CompositeBuildError(
                f"category {category} has multiple public-award feature contracts"
            )
        if job_id in generation_manifest_by_job:
            raise CompositeBuildError(
                f"audit repeats generation manifest for job {job_id}"
            )
        generation_manifest_by_job[job_id] = {
            "path": str(manifest_path),
            "sha256": record["sha256"],
        }
    if set(generation_manifest_by_job) != set(selected_by_job):
        raise CompositeBuildError(
            "audited generation manifests do not cover every selected job"
        )

    filtered_records: dict[str, list[dict[str, Any]]] = {
        category: [] for category in expected_games
    }
    source_bindings: list[dict[str, Any]] = []
    activation_chunks: dict[str, list[dict[str, Any]]] = {
        category: [] for category in expected_games
    }
    observed_by_job: dict[str, set[int]] = defaultdict(set)
    order_by_category: Counter[str] = Counter()
    for job_id in [str(job["job_id"]) for job in lock["fleet"]["jobs"]]:
        if job_id not in selected_by_job:
            continue
        job = jobs[job_id]
        category = str(job["category"])
        job_selected = selected_by_job[job_id]
        for source_record in audited_by_job[job_id]:
            source = Path(str(source_record["path"])).resolve(strict=True)
            before_sha = _file_sha256(source)
            if before_sha != source_record["sha256"]:
                raise CompositeBuildError(
                    f"audited source shard bytes drifted: {source}"
                )
            try:
                with np.load(source, allow_pickle=False) as payload:
                    if "game_seed" not in payload.files:
                        raise CompositeBuildError(
                            f"source shard lacks game_seed: {source}"
                        )
                    seeds = np.asarray(payload["game_seed"], dtype=np.int64)
                    if seeds.ndim != 1:
                        raise CompositeBuildError(
                            f"game_seed is not one-dimensional: {source}"
                        )
                    selected_mask = np.isin(
                        seeds, np.asarray(sorted(job_selected), dtype=np.int64)
                    )
                    for seed in set(map(int, seeds.tolist())).intersection(
                        owner_by_seed
                    ):
                        if owner_by_seed[seed] != (job_id, category):
                            raise CompositeBuildError(
                                "selected seed appears in the wrong audited job/category: "
                                f"seed={seed} source={job_id}/{category} "
                                f"owner={owner_by_seed[seed]}"
                            )
                    if not np.any(selected_mask):
                        continue
                    for status, expected in (
                        ("terminated", True),
                        ("truncated", False),
                    ):
                        if status not in payload.files or np.any(
                            np.asarray(payload[status], dtype=bool)[selected_mask]
                            != expected
                        ):
                            raise CompositeBuildError(
                                f"selected {job_id} rows are not complete: {status}"
                            )
                    if "policy_weight_multiplier" not in payload.files:
                        raise CompositeBuildError(
                            f"selected source lacks policy_weight_multiplier: {source}"
                        )
                    activation = contract._selected_target_activation_chunk(  # noqa: SLF001
                        payload,
                        game_seeds=seeds,
                        selected_mask=selected_mask,
                        where=f"{job_id}:{source.name}",
                        require_policy_target_completeness=True,
                        wide_full_threshold=int(
                            contract.WIDE_CHOICE_MIN_LEGAL_ACTIONS
                        ),
                    )
                    activation_chunk = {
                        "schema_version": activation["schema_version"],
                        "job_id": job_id,
                        "source_sha256": before_sha,
                        "counts": activation["counts"],
                        "counts_sha256": activation["counts_sha256"],
                        "row_activation_sha256": activation[
                            "row_activation_sha256"
                        ],
                    }
                    activation_chunk["chunk_sha256"] = _digest(activation_chunk)
                    expected_activation = source_record.get("target_activation")
                    if not isinstance(expected_activation, dict) or expected_activation != {
                        "counts": activation_chunk["counts"],
                        "counts_sha256": activation_chunk["counts_sha256"],
                        "row_activation_sha256": activation_chunk[
                            "row_activation_sha256"
                        ],
                        "chunk_sha256": activation_chunk["chunk_sha256"],
                    }:
                        raise CompositeBuildError(
                            f"audited target activation differs from source rows: {source}"
                        )
                    if category != "current_producer":
                        allowed_versions = {
                            int(checkpoint_by_id[checkpoint_id].get("version", -1))
                            for checkpoint_id in category_specs[category][
                                "checkpoint_ids"
                            ]
                        }
                        contract._validate_selected_opponent_rows(  # noqa: SLF001
                            payload,
                            selected_mask=selected_mask,
                            game_seeds=seeds,
                            job=job,
                            allowed_versions=allowed_versions,
                            colors=selfplay_colors,
                        )
                    arrays = _filter_selected_source_arrays(
                        payload,
                        selected_mask=selected_mask,
                        rows=int(seeds.size),
                        source=source,
                    )
            except (KeyError, OSError, ValueError, contract.ContractError) as error:
                raise CompositeBuildError(
                    f"cannot filter source shard {source}: {error}"
                ) from error

            observed = set(map(int, np.asarray(arrays["game_seed"]).tolist()))
            observed_by_job[job_id].update(observed)
            filtered_dir = output_root / "filtered_sources" / category
            filtered_path = filtered_dir / (
                f"{order_by_category[category]:05d}-{job_id}.npz"
            )
            _atomic_npz(filtered_path, arrays)
            filtered_path = filtered_path.resolve(strict=True)
            binding = {
                "contract_sha256": lock["contract_sha256"],
                "audit_file_sha256": audit["file_sha256"],
                "audit_sha256": audit["audit_sha256"],
                "selected_manifest_file_sha256": selected["file_sha256"],
                "selected_records_sha256": selected["records_sha256"],
                "job_id": job_id,
                "category": category,
                **(
                    {}
                    if contract._sealed_category_semantic(  # noqa: SLF001
                        lock, category
                    )
                    is None
                    else {
                        "category_semantic": contract._sealed_category_semantic(  # noqa: SLF001
                            lock, category
                        )
                    }
                ),
                "source_path": str(source),
                "source_sha256": before_sha,
                "generation_manifest_path": generation_manifest_by_job[job_id]["path"],
                "generation_manifest_sha256": generation_manifest_by_job[job_id][
                    "sha256"
                ],
                "target_activation_chunk_sha256": activation_chunk["chunk_sha256"],
                "target_activation_counts_sha256": activation_chunk[
                    "counts_sha256"
                ],
            }
            source_id = _digest(binding)
            filtered_record = {
                "path": str(filtered_path),
                "rows": int(np.asarray(arrays["game_seed"]).size),
                "order": int(order_by_category[category]),
                "size_bytes": filtered_path.stat().st_size,
                "sha256": _file_sha256(filtered_path),
                "checkpoint_version": int(producer["version"]),
                "producer_checkpoint_path": str(producer_path),
                "producer_checkpoint_sha256": producer["sha256"],
                "source_id": source_id,
                "source_category": category,
            }
            filtered_records[category].append(filtered_record)
            source_bindings.append({"source_id": source_id, **binding})
            activation_chunks[category].append(activation_chunk)
            order_by_category[category] += 1
            if _file_sha256(source) != before_sha:
                raise CompositeBuildError(
                    f"source shard changed during filtering: {source}"
                )

    for job_id, selected_seeds in selected_by_job.items():
        if observed_by_job[job_id] != selected_seeds:
            raise CompositeBuildError(
                f"filtered rows do not exactly cover selected games for {job_id}: "
                f"missing={len(selected_seeds - observed_by_job[job_id])} "
                f"unexpected={len(observed_by_job[job_id] - selected_seeds)}"
            )
    for category, records in filtered_records.items():
        source_root = output_root / "filtered_sources" / category
        _atomic_json(
            source_root / "manifest.json",
            {
                "shards": [record["path"] for record in records],
                "public_award_feature_provenance": public_award_by_category[category],
            },
        )
    try:
        rebuilt_activation = contract._build_target_activation_report(  # noqa: SLF001
            activation_chunks,
            categories=tuple(expected_games),
            sealed_p_full=float(lock["science"]["search_operator"]["p_full"]),
            wide_full_threshold=int(contract.WIDE_CHOICE_MIN_LEGAL_ACTIONS),
        )
    except (KeyError, TypeError, ValueError, contract.ContractError) as error:
        raise CompositeBuildError(
            f"cannot reconstruct selected target activation: {error}"
        ) from error
    if rebuilt_activation != audit.get("target_activation"):
        raise CompositeBuildError(
            "composite target-activation replay differs from post-wave audit"
        )
    return filtered_records, source_bindings, rebuilt_activation


def _build_fresh_component(
    *,
    category: str,
    records: list[dict[str, Any]],
    producer: Mapping[str, Any],
    output_root: Path,
    expected_games: int,
    source_authority: Mapping[str, str],
    policy_target_identity: Mapping[str, Any],
    policy_target_completeness: Mapping[str, Any],
    build_memmap_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise CompositeBuildError(f"fresh component {category} has no filtered shards")
    source_root = output_root / "filtered_sources" / category
    corpus_dir = output_root / "corpora" / category
    try:
        meta = build_memmap_fn(source_root, corpus_dir, progress_every=0)
        mass = measure_memmap_component(corpus_dir, meta)
    except (OSError, SystemExit, ValueError) as error:
        raise CompositeBuildError(f"cannot build {category} memmap: {error}") from error
    if mass["game_count"] != expected_games or mass["policy_active_row_count"] <= 0:
        raise CompositeBuildError(
            f"fresh {category} mass differs from selected whole-game quota: {mass}"
        )
    version = int(producer["version"])
    policy_target_identity_sha256 = str(policy_target_identity["sha256"])
    provenance = {
        "schema_version": "flywheel-replay-component-v2",
        "component_id": category,
        "source_category": category,
        "role": "fresh",
        "current_checkpoint_version": version,
        "checkpoint_versions": [version],
        "producer_checkpoints": [
            {
                "version": version,
                "path": records[0]["producer_checkpoint_path"],
                "sha256": producer["sha256"],
            }
        ],
        "row_count": sum(int(record["rows"]) for record in records),
        "shards": records,
        "shard_inventory_sha256": canonical_sha256(records),
        "component_mass": mass,
        "source_authority_manifest": dict(source_authority),
        "policy_target_identity": dict(policy_target_identity["payload"]),
        "policy_target_identity_sha256": policy_target_identity_sha256,
        "policy_target_completeness": dict(policy_target_completeness),
    }
    provenance_path = output_root / "provenance" / f"{category}.json"
    _atomic_json(provenance_path, provenance)
    provenance_path = provenance_path.resolve(strict=True)
    meta_path = corpus_dir / "corpus_meta.json"
    meta = _load_json(meta_path)
    provenance_ref = {
        "path": str(provenance_path),
        "file_sha256": _file_sha256(provenance_path),
    }
    meta["flywheel_component_provenance"] = provenance_ref
    meta["policy_target_identity"] = dict(policy_target_identity["payload"])
    meta["policy_target_identity_sha256"] = policy_target_identity_sha256
    meta["policy_target_completeness"] = dict(policy_target_completeness)
    _atomic_json(meta_path, meta)
    return {
        "component_id": category,
        "source_category": category,
        "game_sampling_ratio": EFFECTIVE_COMPONENT_RATIOS[category],
        "corpus_dir": str(corpus_dir.resolve(strict=True)),
        "corpus_meta_sha256": _file_sha256(meta_path),
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "provenance_manifest": str(provenance_path),
        "provenance_manifest_sha256": provenance_ref["file_sha256"],
        "component_mass": mass,
        "source_authority_manifest": source_authority["path"],
        "source_authority_manifest_sha256": source_authority["file_sha256"],
    }


def _fresh_policy_target_identities(
    source_authority: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Resolve one exact search-teacher identity for every fresh component."""

    payload = _load_json(Path(str(source_authority["path"])))
    try:
        groups = target_inventory._manifest_operator_groups(payload)  # noqa: SLF001
    except (OSError, ValueError, target_inventory.InventoryError) as error:
        raise CompositeBuildError(
            f"cannot resolve fresh policy-target identities: {error}"
        ) from error
    by_category: dict[str, dict[str, dict[str, Any]]] = {
        category: {} for category in FRESH_SOURCE_GAME_RATIOS
    }
    for group in groups:
        category = str(group.get("category", ""))
        if group.get("scope") == "fresh" and category in by_category:
            identity = str(group.get("operator_sha256", ""))
            if not train_bc._is_sha256(identity):  # noqa: SLF001
                raise CompositeBuildError(
                    f"fresh category {category!r} has malformed target identity"
                )
            identity_payload = group.get("policy_target_identity")
            if not isinstance(identity_payload, Mapping):
                raise CompositeBuildError(
                    f"fresh category {category!r} has no versioned target identity"
                )
            by_category[category][identity] = dict(identity_payload)
    malformed = {
        category: sorted(identity_payloads)
        for category, identity_payloads in by_category.items()
        if len(identity_payloads) != 1
    }
    if malformed:
        raise CompositeBuildError(
            "fresh categories do not bind exactly one policy-target operator: "
            f"{malformed}"
        )
    realized = {
        next(iter(identity_payloads))
        for identity_payloads in by_category.values()
    }
    if len(realized) != 1:
        raise CompositeBuildError(
            "fresh policy components were produced by different search operators"
        )
    return {
        category: {
            "sha256": next(iter(identity_payloads)),
            "payload": next(iter(identity_payloads.values())),
        }
        for category, identity_payloads in by_category.items()
    }


def _build_source_authority(
    *,
    lock_path: Path,
    lock: Mapping[str, Any],
    selected: Mapping[str, Any],
    audit: Mapping[str, Any],
    source_bindings: list[dict[str, Any]],
    current_lock_verifier_authority: Mapping[str, Any],
    output_root: Path,
) -> dict[str, str]:
    """Materialize the complete, portable authority before the descriptor.

    Raw generation shards are verified while filtering/sealing, but are much
    larger than the learner input and deliberately are not copied.  Their
    immutable path/hash preimages remain in ``source_bindings``.  Every small
    semantic artifact needed to interpret those preimages is copied into the
    composite root, so a second B200 never has to re-open an unstaged source
    path merely to authenticate the already-filtered learner corpus.
    """

    normalized_bindings = _validate_source_bindings(
        source_bindings,
        lock=lock,
        selected_file_sha256=str(selected["file_sha256"]),
        selected_records_sha256=str(selected["records_sha256"]),
        audit_file_sha256=str(audit["file_sha256"]),
        audit_sha256=str(audit["audit_sha256"]),
        require_target_activation=True,
    )
    authority_root = output_root / "authority"

    def staged_ref(source: Path, relative: str) -> dict[str, str]:
        destination = authority_root / relative
        _atomic_copy(source, destination)
        return _artifact_ref(destination)

    def staged_manifests(
        bindings: Sequence[Mapping[str, Any]], *, namespace: str
    ) -> list[dict[str, Any]]:
        unique: dict[tuple[str, str, str], dict[str, Any]] = {}
        for binding in bindings:
            identity = (
                str(binding["job_id"]),
                str(binding["generation_manifest_path"]),
                str(binding["generation_manifest_sha256"]),
            )
            unique.setdefault(identity, dict(binding))
        records: list[dict[str, Any]] = []
        for index, (identity, binding) in enumerate(sorted(unique.items())):
            job_id, original_path, original_sha256 = identity
            safe_job = "".join(
                value if value.isalnum() or value in "._-" else "_"
                for value in job_id
            )
            artifact = staged_ref(
                Path(original_path),
                f"{namespace}/generation_manifests/{index:05d}-{safe_job}.json",
            )
            if artifact["file_sha256"] != original_sha256:
                raise CompositeBuildError(
                    f"staged generation manifest changed bytes for {job_id}"
                )
            records.append(
                {
                    "job_id": job_id,
                    "category": binding["category"],
                    "original_path": original_path,
                    "original_file_sha256": original_sha256,
                    "artifact": artifact,
                }
            )
        return records

    lock_ref = staged_ref(lock_path, "current/contract.lock.json")
    selected_ref = staged_ref(
        Path(str(selected["path"])), "current/selected_games.json"
    )
    audit_ref = staged_ref(Path(str(audit["path"])), "current/post_wave_audit.json")
    current_manifests = staged_manifests(normalized_bindings, namespace="current")

    lock_verifier_authorities = {
        "current_wave": _validated_lock_verifier_authority(
            current_lock_verifier_authority,
            lock_path=lock_path,
            lock=lock,
            require_all_job_claims=True,
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": SOURCE_AUTHORITY_SCHEMA,
        "canonical_composite_root": str(output_root.resolve(strict=True)),
        **(
            {}
            if lock.get("category_semantics") is None
            else {"category_semantics": lock["category_semantics"]}
        ),
        "current_contract": {
            **lock_ref,
            "contract_sha256": lock["contract_sha256"],
        },
        "selected_game_manifest": {
            **selected_ref,
            "manifest_sha256": selected["manifest_sha256"],
            "records_sha256": selected["records_sha256"],
            "selected_game_seed_set_sha256": selected[
                "selected_game_seed_set_sha256"
            ],
        },
        "post_wave_audit": {
            **audit_ref,
            "audit_sha256": audit["audit_sha256"],
            "shard_inventory_sha256": audit["shard_inventory_sha256"],
            "target_activation_sha256": audit["target_activation"][
                "target_activation_sha256"
            ],
        },
        "fresh_target_activation": audit["target_activation"],
        "fresh_source_bindings": normalized_bindings,
        "fresh_source_bindings_sha256": canonical_sha256(normalized_bindings),
        "fresh_generation_manifests": current_manifests,
        "fresh_generation_manifests_sha256": canonical_sha256(current_manifests),
        "lock_verifier_authorities": lock_verifier_authorities,
    }
    payload["authority_sha256"] = _digest(payload)
    path = output_root / "source_authority.json"
    _atomic_json(path, payload)
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": _file_sha256(path),
        "authority_sha256": payload["authority_sha256"],
    }


def _build_descriptor(
    *,
    components: list[dict[str, Any]],
    producer_path: Path,
    producer_sha256: str,
    current_version: int,
    source_authority: Mapping[str, str],
    category_semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    component_ids = [str(component["component_id"]) for component in components]
    expected_ids = list(FRESH_SOURCE_GAME_RATIOS)
    if component_ids != expected_ids:
        raise CompositeBuildError(
            f"component order/identity drift: {component_ids} != {expected_ids}"
        )
    effective = {
        str(component["component_id"]): float(component["game_sampling_ratio"])
        for component in components
    }
    if effective != EFFECTIVE_COMPONENT_RATIOS:
        raise CompositeBuildError(
            "effective component ratios differ from .80/.15/.05"
        )
    provenance_payloads = [
        _load_json(Path(str(component["provenance_manifest"])))
        for component in components
    ]
    checkpoint_versions = sorted(
        {
            int(version)
            for provenance in provenance_payloads
            for version in provenance["checkpoint_versions"]
        }
    )
    provenance_binding = [
        {
            "component_id": component["component_id"],
            "provenance_manifest_sha256": component["provenance_manifest_sha256"],
        }
        for component in components
    ]
    sampling_receipt = build_sampling_receipt(components)
    adapter_versions: dict[str, str] = {}
    for component in components:
        component_id = str(component["component_id"])
        version = _memmap_adapter_version(
            Path(str(component["corpus_dir"])), component_id=component_id
        )
        adapter_versions[component_id] = version
    if len(set(adapter_versions.values())) != 1:
        raise CompositeBuildError(
            "post-wave components mix incompatible entity adapters: "
            f"{adapter_versions}"
        )
    aux_subgoal_component_ids: list[str] = []
    for component in components:
        component_id = str(component["component_id"])
        corpus_dir = component.get("corpus_dir")
        if not isinstance(corpus_dir, str):
            continue
        meta = _load_json(Path(corpus_dir) / "corpus_meta.json")
        aux_contract = meta.get("aux_subgoal_target_contract")
        expected_counts = {
            str(AUX_SUBGOAL_TARGET_VERSION): int(meta.get("row_count", -1))
        }
        if (
            isinstance(aux_contract, dict)
            and aux_contract.get("version_key")
            == AUX_SUBGOAL_TARGET_VERSION_KEY
            and aux_contract.get("supported_version")
            == AUX_SUBGOAL_TARGET_VERSION
            and aux_contract.get("semantic") == AUX_SUBGOAL_TARGET_SEMANTIC
            and aux_contract.get("realized_version_counts") == expected_counts
            and aux_contract.get("all_rows_semantically_eligible") is True
        ):
            aux_subgoal_component_ids.append(component_id)
    expected_aux_subgoal_component_ids = list(FRESH_SOURCE_GAME_RATIOS)
    if aux_subgoal_component_ids != expected_aux_subgoal_component_ids:
        missing = [
            component_id
            for component_id in expected_aux_subgoal_component_ids
            if component_id not in aux_subgoal_component_ids
        ]
        raise CompositeBuildError(
            "fresh component aux-subgoal target contract is not uniformly "
            f"strict-future v{AUX_SUBGOAL_TARGET_VERSION}; missing={missing}"
        )
    fresh_component_ids = list(FRESH_SOURCE_GAME_RATIOS)
    replay_contract = {
        "schema_version": "flywheel-replay-composite-v3",
        "current_checkpoint_version": int(current_version),
        "initializer_checkpoint_path": str(producer_path),
        "initializer_checkpoint_sha256": producer_sha256,
        "fresh_component_ids": fresh_component_ids,
        "replay_component_ids": [],
        "fresh_source_game_ratios": dict(FRESH_SOURCE_GAME_RATIOS),
        "effective_component_sampling_ratios": effective,
        "minimum_replay_ratio": 0.0,
        "realized_replay_ratio": 0.0,
        "checkpoint_versions": checkpoint_versions,
        "component_provenance_sha256": canonical_sha256(provenance_binding),
        "sampling_receipt": sampling_receipt,
        "sampling_receipt_sha256": canonical_sha256(sampling_receipt),
    }
    recipe = dict(LEARNER_RECIPE_OVERRIDES)
    descriptor_components = [
        {
            key: value
            for key, value in component.items()
            if key != "entity_feature_adapter_version"
        }
        for component in components
    ]
    return {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": False,
        "promotion_eligible": True,
        **(
            {}
            if category_semantics is None
            else {"category_semantics": dict(category_semantics)}
        ),
        "components": descriptor_components,
        "learner_recipe_overrides": recipe,
        "learner_recipe_overrides_sha256": canonical_sha256(recipe),
        "policy_kl_anchor_component_ids": [],
        "policy_distillation_component_ids": fresh_component_ids,
        "stored_policy_component_temperatures": dict(
            STORED_POLICY_COMPONENT_TEMPERATURES
        ),
        "entity_feature_adapter_component_versions": adapter_versions,
        # Terminal outcomes estimate the return of the complete continuation
        # policy that generated each trajectory.  In opponent-pool games only
        # producer-seat roots are recorded, but the other seat is controlled by
        # an archived/hard-negative policy and that identity is provenance, not
        # a neural input.  Those rows retain current-producer search-policy
        # targets, but their terminal z belongs to a different, unobserved
        # continuation operator.  Canonical search-value learning therefore
        # uses producer self-play outcomes only.  Mixed-opponent rows remain
        # policy/state evidence; a future population value must be a separately
        # conditioned head rather than aliasing this one.
        "value_training_component_ids": ["current_producer"],
        "aux_subgoal_component_ids": aux_subgoal_component_ids,
        "flywheel_replay_contract": replay_contract,
        "source_authority_manifest": source_authority["path"],
        "source_authority_manifest_sha256": source_authority["file_sha256"],
        "source_authority_sha256": source_authority["authority_sha256"],
    }


def build_post_wave_composite(
    *,
    lock_path: Path,
    selected_path: Path,
    audit_path: Path,
    output_root: Path,
    verify_lock_fn: Callable[..., dict[str, Any]],
    current_lock_verifier_authority: Mapping[str, Any] | None = None,
    build_memmap_fn: Callable[..., dict[str, Any]] = memmap_builder.build_memmap_corpus,
    verify_descriptor_fn: Callable[[Path], dict[str, Any]] = (
        train_bc._preflight_memmap_composite_descriptor  # noqa: SLF001
    ),
    expected_games: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if current_lock_verifier_authority is None:
        raise CompositeBuildError(
            "post-wave composite requires a current frozen lock-verifier authority"
        )
    root = _prepare_output_root(output_root)
    lock, selected, audit, raw_selected = _validated_wave_inputs(
        lock_path,
        selected_path,
        audit_path,
        verify_lock_fn=verify_lock_fn,
    )
    # The lock is the quota authority.  The original recovery wave selected
    # 12k games, while the scale profile selects 64k; retaining a module-level
    # 12k default here would make an otherwise valid scale wave fail during
    # corpus construction (or tempt an operator to pass an unauthenticated
    # override).  An explicit value remains available to focused callers, but
    # it must agree exactly with the sealed contract.
    raw_locked_games = lock.get("game_contract", {}).get("category_games")
    if not isinstance(raw_locked_games, dict):
        raise CompositeBuildError(
            "sealed wave lock has no game_contract.category_games quota authority"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_locked_games.values()
    ):
        raise CompositeBuildError(
            "sealed wave lock category-game quotas must be exact integers"
        )
    try:
        locked_games = {
            str(category): int(raw_locked_games[category])
            for category in FRESH_SOURCE_GAME_RATIOS
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CompositeBuildError(
            "sealed wave lock has malformed category-game quotas"
        ) from error
    if (
        set(raw_locked_games) != set(FRESH_SOURCE_GAME_RATIOS)
        or any(value <= 0 for value in locked_games.values())
        or sum(locked_games.values())
        != int(lock["game_contract"].get("total_complete_games", -1))
    ):
        raise CompositeBuildError(
            f"sealed wave lock has inconsistent category-game quotas: {raw_locked_games}"
        )
    if expected_games is not None and dict(expected_games) != locked_games:
        raise CompositeBuildError(
            "caller category-game quotas differ from the sealed wave lock: "
            f"caller={dict(expected_games)} lock={locked_games}"
        )
    expected_games = locked_games
    producer = contract._producer(lock)  # noqa: SLF001
    if isinstance(producer.get("version"), bool) or not isinstance(
        producer.get("version"), int
    ):
        raise CompositeBuildError("current producer has no authenticated version")
    producer_path = Path(str(producer["path"])).expanduser().resolve(strict=True)
    if _file_sha256(producer_path) != producer["sha256"]:
        raise CompositeBuildError("current producer checkpoint bytes drifted")

    records_by_category, source_bindings, target_activation = _filter_wave_shards(
        lock=lock,
        selected=selected,
        audit=audit,
        raw_selected=raw_selected,
        output_root=root,
        expected_games=expected_games,
    )
    source_authority = _build_source_authority(
        lock_path=lock_path,
        lock=lock,
        selected=selected,
        audit=audit,
        source_bindings=source_bindings,
        current_lock_verifier_authority=current_lock_verifier_authority,
        output_root=root,
    )
    policy_target_identities = _fresh_policy_target_identities(source_authority)
    components = [
        _build_fresh_component(
            category=category,
            records=records_by_category[category],
            producer=producer,
            output_root=root,
            expected_games=int(expected_games[category]),
            source_authority=source_authority,
            policy_target_identity=policy_target_identities[category],
            policy_target_completeness=target_activation["categories"][category],
            build_memmap_fn=build_memmap_fn,
        )
        for category in FRESH_SOURCE_GAME_RATIOS
    ]
    descriptor = _build_descriptor(
        components=components,
        producer_path=producer_path,
        producer_sha256=str(producer["sha256"]),
        current_version=int(producer["version"]),
        source_authority=source_authority,
        category_semantics=lock.get("category_semantics"),
    )
    # This is the last mutation before the descriptor's atomic publication.
    # Sealing all three component payload inventories here makes the builder's
    # own descriptor preflight publish an authenticated identity cache, so the
    # one-dose trainer can reuse it instead of rehashing every payload byte.
    _finalize_component_payloads_read_only(components)
    descriptor_path = root / "memmap_composite.json"
    _atomic_json(descriptor_path, descriptor)
    try:
        verified = verify_descriptor_fn(descriptor_path)
    except (OSError, SystemExit, ValueError) as error:
        raise CompositeBuildError(
            f"final composite preflight failed: {error}"
        ) from error
    receipt = {
        "schema_version": BUILD_RECEIPT_SCHEMA,
        "contract": {
            "path": str(lock_path.expanduser().resolve(strict=True)),
            "file_sha256": _file_sha256(lock_path.expanduser().resolve(strict=True)),
            "contract_sha256": lock["contract_sha256"],
        },
        "selected_game_manifest": {
            "path": str(selected["path"]),
            "file_sha256": selected["file_sha256"],
            "records_sha256": selected["records_sha256"],
            "category_game_counts": dict(expected_games),
        },
        "post_wave_audit": {
            "path": str(audit["path"]),
            "file_sha256": audit["file_sha256"],
            "audit_sha256": audit["audit_sha256"],
            "shard_inventory_sha256": audit["shard_inventory_sha256"],
            "target_activation_sha256": target_activation[
                "target_activation_sha256"
            ],
        },
        "fresh_target_activation": target_activation,
        "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_sha256(source_bindings),
        "source_authority": source_authority,
        "descriptor": {
            "path": str(descriptor_path.resolve(strict=True)),
            "file_sha256": _file_sha256(descriptor_path),
            "fingerprint": canonical_sha256(descriptor),
        },
        "sampling_receipt": descriptor["flywheel_replay_contract"]["sampling_receipt"],
        "verified_descriptor_fingerprint": verified.get("descriptor_fingerprint"),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    _atomic_json(root / "build_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--selected-game-manifest", type=Path, required=True)
    parser.add_argument("--post-wave-audit", type=Path, required=True)
    parser.add_argument("--frozen-repo", type=Path, required=True)
    parser.add_argument("--frozen-verifier-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        try:
            verify_lock_fn, current_verifier_authority = (
                frozen_lock_verifier.build_frozen_lock_verifier(
                    frozen_repo=args.frozen_repo,
                    expected_verifier_sha256=args.frozen_verifier_sha256,
                    lock_path=args.lock,
                    require_all_job_claims=True,
                )
            )
        except frozen_lock_verifier.FrozenVerifierError as error:
            raise CompositeBuildError(str(error)) from error
        receipt = build_post_wave_composite(
            lock_path=args.lock,
            selected_path=args.selected_game_manifest,
            audit_path=args.post_wave_audit,
            output_root=args.out,
            verify_lock_fn=verify_lock_fn,
            current_lock_verifier_authority=current_verifier_authority,
        )
    except CompositeBuildError as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
