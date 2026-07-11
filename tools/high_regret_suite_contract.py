"""Shared fail-closed validation for sealed high-regret suite sources."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SUITE_SCHEMA = "a1-held-out-high-regret-suite-v3"
REPLAY_CONTRACT = "authoritative-shard-parent-hashed-unique-contiguous-trajectory-v3"


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
            from tools.regret_common import load_shard

            shard = load_shard(authoritative_path)
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

    from tools.regret_common import discover_shards, load_shard

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
        for shard_path in discover_shards([scope]):
            try:
                shard = load_shard(shard_path)
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
