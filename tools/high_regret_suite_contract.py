"""Shared fail-closed validation for sealed high-regret suite sources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import regret_common  # noqa: E402


SUITE_SCHEMA = "a1-held-out-high-regret-suite-v4"
REPLAY_CONTRACT = "authoritative-shard-parent-hashed-unique-contiguous-trajectory-v3"


def _seed_set_sha256(seeds: np.ndarray) -> str:
    values = np.sort(np.asarray(seeds, dtype=np.int64).reshape(-1))
    return "sha256:" + hashlib.sha256(
        values.astype("<i8", copy=False).tobytes()
    ).hexdigest()


def load_validation_seed_manifest(
    path: Path,
) -> tuple[set[int], dict[str, Any]]:
    """Load and bind the trainer's exact game-level held-out set.

    High-regret's own hash partition is not a training holdout.  Promotion
    evidence must first be restricted to a trainer-authenticated validation
    manifest, then may rank/stratify within that fixed set.
    """

    path = path.expanduser().absolute()
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("validation-seed manifest is not a canonical regular file")
    before = _stable_file_record(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load validation-seed manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("validation-seed manifest is not a JSON object")
    schema = payload.get("schema_version")
    if schema == "train-validation-game-seeds-v1":
        raw_seeds = payload.get("game_seeds")
        declared_count = payload.get("validation_game_seed_count")
        declared_digest = payload.get("validation_game_seed_set_sha256")
    elif schema == "value-calibration-validation-seeds-v1":
        raw_seeds = payload.get("validation_game_seeds")
        declared_count = payload.get("validation_game_seed_count")
        declared_digest = payload.get("validation_game_seed_set_sha256")
    else:
        raise ValueError(
            f"unsupported validation-seed manifest schema {schema!r}"
        )
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("validation-seed manifest contains no game seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise ValueError("validation-seed manifest game seeds must be integers")
    values = np.asarray(raw_seeds, dtype=np.int64)
    if len(np.unique(values)) != len(values):
        raise ValueError("validation-seed manifest contains duplicate game seeds")
    digest = _seed_set_sha256(values)
    if declared_count is not None and (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(values)
    ):
        raise ValueError("validation-seed manifest count mismatch")
    if declared_digest is not None and declared_digest != digest:
        raise ValueError("validation-seed manifest seed-set digest mismatch")
    after = _stable_file_record(path)
    if after != before:
        raise ValueError("validation-seed manifest changed while loading")
    return set(map(int, values)), {
        "path": str(path),
        "sha256": before["sha256"],
        "schema_version": schema,
        "game_seed_count": len(values),
        "game_seed_set_sha256": digest,
    }


def load_source_validation_binding(
    source_manifest: Path,
) -> tuple[set[int], dict[str, Any]]:
    """Replay the validation binding embedded by the regret extractor."""

    required = {
        "held_out_only",
        "validation_seed_manifest_path",
        "validation_seed_manifest_sha256",
        "validation_seed_manifest_schema_version",
        "validation_game_seed_count",
        "validation_game_seed_set_sha256",
        "game_seed",
    }
    try:
        with np.load(source_manifest, allow_pickle=False) as data:
            if not required.issubset(data.files):
                raise ValueError(
                    "regret manifest lacks an authenticated validation-seed binding"
                )

            def scalar(name: str) -> Any:
                values = np.asarray(data[name]).reshape(-1)
                if len(values) != 1:
                    raise ValueError(f"regret manifest {name} must be scalar")
                return values[0].item() if hasattr(values[0], "item") else values[0]

            held_out_only = scalar("held_out_only")
            path = Path(str(scalar("validation_seed_manifest_path")))
            stated = {
                "sha256": str(scalar("validation_seed_manifest_sha256")),
                "schema_version": str(
                    scalar("validation_seed_manifest_schema_version")
                ),
                "game_seed_count": int(scalar("validation_game_seed_count")),
                "game_seed_set_sha256": str(
                    scalar("validation_game_seed_set_sha256")
                ),
            }
            source_seeds = np.asarray(data["game_seed"], dtype=np.int64).reshape(-1)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"cannot replay regret validation binding: {error}") from error
    if held_out_only is not True and held_out_only != np.bool_(True):
        raise ValueError("regret manifest is not held-out-only")
    if not path.is_absolute():
        path = source_manifest.parent / path
    allowed, actual = load_validation_seed_manifest(path)
    comparable = {key: actual[key] for key in stated}
    if stated != comparable:
        raise ValueError("regret validation-seed binding drifted")
    observed = set(map(int, source_seeds))
    if not observed:
        raise ValueError("held-out regret manifest contains no rows")
    leaked = observed - allowed
    if leaked:
        raise ValueError(
            f"held-out regret manifest contains {len(leaked)} non-validation game seeds"
        )
    return allowed, actual


def _stable_file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"held-out replay shard is a symlink: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"held-out replay shard is not regular: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError(f"held-out replay shard changed while hashing: {path}")
        named = path.stat(follow_symlinks=False)
        named_identity = (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        )
        if named_identity != identity_after or not stat.S_ISREG(named.st_mode):
            raise ValueError(f"held-out replay shard pathname changed while hashing: {path}")
        return {
            "path": str(path),
            "size_bytes": int(before.st_size),
            "sha256": "sha256:" + digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def scope_inventory_sha256(scope: Path) -> tuple[str, int]:
    """Hash the exact regular shard inventory scanned by replay."""

    scope = scope.expanduser().absolute()
    try:
        canonical_scope = scope.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"held-out replay scope is missing: {scope}: {error}") from error
    if scope != canonical_scope or not scope.is_dir() or scope.is_symlink():
        raise ValueError(f"held-out replay scope is not a canonical directory: {scope}")
    paths: list[Path] = []
    for pattern in ("*.npz", "*.npz.zst"):
        paths.extend(scope.rglob(pattern))
    canonical_paths: list[Path] = []
    for path in sorted(set(paths)):
        if path.is_symlink():
            raise ValueError(f"held-out replay inventory contains symlink {path}")
        canonical = path.resolve(strict=True)
        if canonical != path.absolute() or scope not in canonical.parents:
            raise ValueError(f"held-out replay shard escapes its scope: {path}")
        canonical_paths.append(canonical)
    if not canonical_paths:
        raise ValueError(f"held-out replay scope contains no shards: {scope}")
    records = [_stable_file_record(path) for path in canonical_paths]
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest(), len(records)


@dataclass
class PinnedReplayScope:
    original_scope: Path
    snapshot_scope: Path
    descriptors: list[int]

    def close(self) -> None:
        for descriptor in self.descriptors:
            os.close(descriptor)
        self.descriptors.clear()
        shutil.rmtree(self.snapshot_scope, ignore_errors=True)


def pin_replay_scope(
    scope: Path, *, expected_sha256: str, expected_count: int
) -> PinnedReplayScope:
    """Snapshot exact bytes from the same held fds used for inventory hashing."""

    scope = scope.expanduser().resolve(strict=True)
    paths: list[Path] = []
    for pattern in ("*.npz", "*.npz.zst"):
        paths.extend(scope.rglob(pattern))
    paths = sorted(set(paths))
    snapshot = Path(tempfile.mkdtemp(prefix="a1-high-regret-replay-"))
    os.chmod(snapshot, 0o700)
    descriptors: list[int] = []
    records: list[dict[str, Any]] = []
    try:
        for path in paths:
            canonical = path.resolve(strict=True)
            if path.is_symlink() or canonical != path.absolute() or scope not in canonical.parents:
                raise ValueError(f"held-out replay shard is unsafe while pinning: {path}")
            descriptor = os.open(
                canonical, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptors.append(descriptor)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"held-out replay shard is not regular: {canonical}")
            relative = canonical.relative_to(scope)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            output = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            digest = hashlib.sha256()
            try:
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(output, view)
                        if written <= 0:
                            raise OSError("short write while pinning replay shard")
                        view = view[written:]
                os.fsync(output)
            finally:
                os.close(output)
            after = os.fstat(descriptor)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if identity != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError(f"held-out replay shard changed while pinning: {canonical}")
            named = canonical.stat(follow_symlinks=False)
            if identity != (
                named.st_dev,
                named.st_ino,
                named.st_size,
                named.st_mtime_ns,
                named.st_ctime_ns,
            ):
                raise ValueError(
                    f"held-out replay shard pathname changed while pinning: {canonical}"
                )
            records.append(
                {
                    "path": str(canonical),
                    "size_bytes": int(before.st_size),
                    "sha256": "sha256:" + digest.hexdigest(),
                }
            )
        encoded = json.dumps(
            records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        actual = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if actual != expected_sha256 or len(records) != expected_count:
            raise ValueError("held-out worker pinned scope inventory drifted")
        return PinnedReplayScope(scope, snapshot, descriptors)
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def validate_replay_metadata(selection: Any, states: Any) -> None:
    if not isinstance(selection, dict) or not isinstance(states, list):
        raise ValueError("held-out suite replay metadata is malformed")
    preflight = selection.get("replay_preflight")
    expected = {
        "contract",
        "candidate_states",
        "replay_complete_states",
        "rejected_bad_source",
        "rejected_noncontiguous",
    }
    if not isinstance(preflight, dict) or set(preflight) != expected:
        raise ValueError("held-out suite lacks required replay preflight")
    counts = [preflight[key] for key in expected - {"contract"}]
    if (
        preflight["contract"] != REPLAY_CONTRACT
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts)
        or preflight["candidate_states"]
        != preflight["replay_complete_states"]
        + preflight["rejected_bad_source"]
        + preflight["rejected_noncontiguous"]
        or preflight["replay_complete_states"] < len(states)
    ):
        raise ValueError("held-out suite replay preflight is inconsistent")


def load_source_manifest(path: Path) -> tuple[list[str], set[tuple[int, int, int, int]]]:
    required = {"shard_paths", "shard_id", "row_index", "game_seed", "decision_index"}
    try:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                raise ValueError("held-out source manifest lacks replay identity fields")
            shard_paths = [str(item) for item in np.asarray(data["shard_paths"]).reshape(-1)]
            columns = [
                np.asarray(data[name]).reshape(-1)
                for name in ("shard_id", "row_index", "game_seed", "decision_index")
            ]
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load held-out source manifest: {error}") from error
    if len({len(column) for column in columns}) != 1:
        raise ValueError("held-out source manifest replay columns are misaligned")
    identities = {
        tuple(int(column[index]) for column in columns)
        for index in range(len(columns[0]))
    }
    if len(identities) != len(columns[0]):
        raise ValueError("held-out source manifest has duplicate replay identities")
    return shard_paths, identities


def bind_state_to_manifest(
    raw_state: Any,
    *,
    suite_base: Path,
    manifest_path: Path,
    shard_paths: Sequence[str],
    identities: set[tuple[int, int, int, int]],
    inventory_cache: dict[Path, tuple[str, int]] | None = None,
    source_row_cache: dict[Path, tuple[np.ndarray, np.ndarray, int]] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_state, dict):
        raise ValueError("held-out suite state is malformed")
    state = dict(raw_state)
    required_ints = ("shard_id", "row_index", "game_seed", "decision_index")
    values: list[int] = []
    for name in required_ints:
        value = state.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"held-out suite state has invalid {name}")
        values.append(value)
    identity = tuple(values)
    if identity not in identities:
        raise ValueError("held-out suite state is not bound to source manifest row")
    shard_id = values[0]
    if shard_id >= len(shard_paths):
        raise ValueError("held-out suite state shard_id is outside source manifest")
    authoritative_path = Path(shard_paths[shard_id]).expanduser()
    if not authoritative_path.is_absolute():
        authoritative_path = manifest_path.parent / authoritative_path
    authoritative_path = authoritative_path.resolve()
    if not authoritative_path.name.endswith((".npz", ".npz.zst")):
        raise ValueError(
            "held-out suite authoritative shard is outside replay inventory namespace"
        )
    declared_path = Path(str(state.get("shard_path", ""))).expanduser()
    if not declared_path.is_absolute():
        declared_path = suite_base / declared_path
    if declared_path.resolve() != authoritative_path:
        raise ValueError("held-out suite state shard_path differs from source manifest")
    if not authoritative_path.is_file() or authoritative_path.is_symlink():
        raise ValueError("held-out suite authoritative shard is missing or is a symlink")
    replay_source = state.get("replay_source")
    scope_path = (
        Path(str(replay_source.get("scope", ""))).expanduser()
        if isinstance(replay_source, dict)
        else Path()
    )
    if (
        not isinstance(replay_source, dict)
        or set(replay_source)
        != {"contract", "scope", "scope_inventory_sha256", "scope_shard_count"}
        or replay_source.get("contract") != REPLAY_CONTRACT
        or not scope_path.is_absolute()
        or scope_path.resolve() != authoritative_path.parent
    ):
        raise ValueError("held-out suite state replay_source is invalid")
    cache = inventory_cache if inventory_cache is not None else {}
    expected_inventory = cache.get(authoritative_path.parent)
    if expected_inventory is None:
        expected_inventory = scope_inventory_sha256(authoritative_path.parent)
        cache[authoritative_path.parent] = expected_inventory
    declared_inventory = replay_source.get("scope_inventory_sha256")
    declared_count = replay_source.get("scope_shard_count")
    if (
        declared_inventory != expected_inventory[0]
        or isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != expected_inventory[1]
    ):
        raise ValueError("held-out suite replay scope inventory drifted")
    row_cache = source_row_cache if source_row_cache is not None else {}
    source_rows = row_cache.get(authoritative_path)
    if source_rows is None:
        try:
            shard = regret_common.load_shard(authoritative_path)
            source_rows = (
                np.asarray(shard["game_seed"]).reshape(-1),
                np.asarray(shard["decision_index"]).reshape(-1),
                len(np.asarray(shard["action_taken"]).reshape(-1)),
            )
        except Exception as error:  # noqa: BLE001 - malformed sealed bytes fail closed.
            raise ValueError(
                f"held-out suite authoritative shard row cannot be loaded: {error}"
            ) from error
        row_cache[authoritative_path] = source_rows
        if scope_inventory_sha256(authoritative_path.parent) != expected_inventory:
            raise ValueError("held-out suite replay scope changed while loading source row")
    row = values[1]
    if (
        row >= len(source_rows[0])
        or row >= len(source_rows[1])
        or row >= source_rows[2]
        or int(source_rows[0][row]) != values[2]
        or int(source_rows[1][row]) != values[3]
    ):
        raise ValueError("held-out suite state differs from authoritative shard row")
    state["shard_path"] = str(authoritative_path)
    state["shard_id"], state["row_index"], state["game_seed"], state["decision_index"] = identity
    return state


def validate_replay_trajectories(states: Sequence[dict[str, Any]]) -> None:
    """Replay the v3 contiguous/unique claim from immutable scope bytes."""

    requested: dict[Path, dict[int, int]] = {}
    for state in states:
        replay_source = state["replay_source"]
        scope = Path(str(replay_source["scope"]))
        seed = int(state["game_seed"])
        target = int(state["decision_index"])
        targets = requested.setdefault(scope, {})
        targets[seed] = max(targets.get(seed, -1), target)
    for scope, targets in requested.items():
        declared_inventory = {
            (
                state["replay_source"]["scope_inventory_sha256"],
                state["replay_source"]["scope_shard_count"],
            )
            for state in states
            if Path(str(state["replay_source"]["scope"])) == scope
        }
        if len(declared_inventory) != 1 or scope_inventory_sha256(scope) not in declared_inventory:
            raise ValueError("held-out suite replay scope inventory is inconsistent")
        counts: dict[int, dict[int, int]] = {seed: {} for seed in targets}
        malformed: set[int] = set()
        for shard_path in regret_common.discover_shards([scope]):
            try:
                shard = regret_common.load_shard(shard_path)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"held-out replay scope contains unreadable shard {shard_path}: {error}"
                ) from error
            if "game_seed" not in shard:
                raise ValueError(
                    f"held-out replay scope shard lacks game_seed: {shard_path}"
                )
            seeds = np.asarray(shard["game_seed"]).reshape(-1)
            if "decision_index" not in shard or "action_taken" not in shard:
                affected = {int(value) for value in seeds if int(value) in targets}
                malformed.update(affected)
                continue
            decisions = np.asarray(shard["decision_index"]).reshape(-1)
            actions = np.asarray(shard["action_taken"]).reshape(-1)
            if len(seeds) != len(decisions) or len(seeds) != len(actions):
                malformed.update(int(value) for value in seeds if int(value) in targets)
                continue
            for row, raw_seed in enumerate(seeds):
                seed = int(raw_seed)
                if seed not in targets:
                    continue
                decision = int(decisions[row])
                if decision < 0:
                    malformed.add(seed)
                    continue
                per_seed = counts[seed]
                per_seed[decision] = per_seed.get(decision, 0) + 1
        for seed, target in targets.items():
            per_seed = counts[seed]
            max_recorded = max(per_seed, default=-1)
            if (
                seed in malformed
                or max_recorded < target
                or any(
                    per_seed.get(decision) != 1
                    for decision in range(max_recorded + 1)
                )
            ):
                raise ValueError(
                    "held-out suite replay trajectory is not one exact contiguous "
                    f"0..N sequence for game_seed={seed} under {scope}"
                )
        if scope_inventory_sha256(scope) not in declared_inventory:
            raise ValueError("held-out suite replay scope changed during trajectory replay")
