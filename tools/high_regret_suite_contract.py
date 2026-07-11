"""Shared fail-closed validation for sealed high-regret suite sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


SUITE_SCHEMA = "a1-held-out-high-regret-suite-v2"
REPLAY_CONTRACT = "authoritative-shard-parent-unique-contiguous-trajectory-v2"


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
        with np.load(path, allow_pickle=True) as data:
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
        or set(replay_source) != {"contract", "scope"}
        or replay_source.get("contract") != REPLAY_CONTRACT
        or not scope_path.is_absolute()
        or scope_path.resolve() != authoritative_path.parent
    ):
        raise ValueError("held-out suite state replay_source is invalid")
    state["shard_path"] = str(authoritative_path)
    state["shard_id"], state["row_index"], state["game_seed"], state["decision_index"] = identity
    return state
