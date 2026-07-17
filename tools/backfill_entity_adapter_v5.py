#!/usr/bin/env python3
"""Losslessly replay adapter-v2 self-play shards into adapter-v5 tensors.

This is deliberately a trajectory replayer, not a tensor guesser.  Adapter-v2
did not store ``owned_at_start`` or turn-local development-card state, so the
actor public-rule slots cannot all be recovered from one row in isolation.
Production self-play shards do retain the game seed, every selected strategic
action, and the absolute decision index. Historical producers could omit
single-legal automatic transitions while still incrementing that index. The
native game's chance stream is derived
solely from ``game_seed ^ 0xA17E``; replay therefore reconstructs the exact
authoritative pre-action state without running MCTS again.

The tool replays an index gap only when every missing state has exactly one
legal action; a missing multi-action choice remains unrecoverable and fails
closed. Before emitting any
row it proves replay identity against the immutable adapter-v2 board/player/
global tensors, legal action IDs, actor, and phase.  It then regenerates every
adapter-owned tensor whose semantics changed through v5.  Input files are
never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V5,
)
from catan_zero.rl.actor_public_rule_state_admission import (
    audit_actor_playable_development_cards,
)
from catan_zero.rl.gumbel_self_play import (
    _apply_selected_action,
    _build_public_learner_features,
)
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
)


BACKFILL_SCHEMA = "entity-adapter-v2-to-v5-trajectory-replay-v1"
COLORS = ("RED", "BLUE")
ACTION_SIZE = 332
CHANCE_SEED_XOR = 0xA17E
SEALED_MAX_DECISIONS = 600

# These are unchanged across v2 and v5.  Exact equality proves that replay
# reached the source row's complete board, hands, bank, player, and prompt
# state before we trust the newly reconstructed turn-local fields.
REPLAY_IDENTITY_KEYS = (
    "hex_tokens",
    "hex_vertex_ids",
    "hex_edge_ids",
    "vertex_tokens",
    "edge_tokens",
    "edge_vertex_ids",
    "player_tokens",
    "global_tokens",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
)

# v3 changed structured-action resources, v4 added public awards/rule state,
# and v5 added ordered public history.  Regenerate the complete semantic
# surface rather than patching global slots 8:16 and leaving a hybrid adapter.
ADAPTER_OWNED_KEYS = (
    "player_tokens",
    "global_tokens",
    "legal_action_tokens",
    "legal_action_target_ids",
    "legal_action_mask",
    "event_tokens",
    "event_target_ids",
    "event_mask",
)


class BackfillError(RuntimeError):
    """A source shard cannot be proven replayable without guessing."""


@dataclass(frozen=True)
class ShardReceipt:
    source: str
    source_relative_path: str
    source_sha256: str
    output: str | None
    output_relative_path: str | None
    output_sha256: str | None
    rows: int
    games: int
    v5_rule_slot_nonzero_counts: tuple[int, ...]
    live_event_rows: int
    actor_playable_development_card_admission: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_relative_path": self.source_relative_path,
            "source_sha256": self.source_sha256,
            "output": self.output,
            "output_relative_path": self.output_relative_path,
            "output_sha256": self.output_sha256,
            "rows": self.rows,
            "games": self.games,
            "v5_rule_slot_nonzero_counts": list(
                self.v5_rule_slot_nonzero_counts
            ),
            "live_event_rows": self.live_event_rows,
            "actor_playable_development_card_admission": (
                self.actor_playable_development_card_admission
            ),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _game_row_spans(
    seeds: np.ndarray,
    decisions: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    *,
    max_decisions: int = SEALED_MAX_DECISIONS,
) -> list[slice]:
    seeds = np.asarray(seeds)
    decisions = np.asarray(decisions)
    terminated = np.asarray(terminated, dtype=np.bool_)
    truncated = np.asarray(truncated, dtype=np.bool_)
    if (
        seeds.ndim != 1
        or decisions.shape != seeds.shape
        or terminated.shape != seeds.shape
        or truncated.shape != seeds.shape
        or seeds.size == 0
    ):
        raise BackfillError(
            "game_seed/decision_index/terminated/truncated must be non-empty 1-D peers"
        )
    starts = np.r_[0, 1 + np.flatnonzero(seeds[1:] != seeds[:-1])]
    stops = np.r_[starts[1:], len(seeds)]
    seen: set[int] = set()
    spans: list[slice] = []
    for start, stop in zip(starts.tolist(), stops.tolist()):
        seed = int(seeds[start])
        if seed in seen:
            raise BackfillError(f"game seed {seed} reappears non-contiguously")
        seen.add(seed)
        actual = decisions[start:stop]
        if int(actual[0]) != 0 or bool(np.any(actual[1:] <= actual[:-1])):
            raise BackfillError(
                f"game seed {seed} decision indices must start at zero and "
                f"increase strictly; got "
                f"{actual[:4].tolist()}..{actual[-4:].tolist()}"
            )
        game_terminated = terminated[start:stop]
        game_truncated = truncated[start:stop]
        if not bool(np.all(game_terminated == game_terminated[0])) or not bool(
            np.all(game_truncated == game_truncated[0])
        ):
            raise BackfillError(
                f"game seed {seed} has non-constant game outcome fields"
            )
        # Outcome fields are copied onto every row after the game completes;
        # they are not per-step done flags.
        if bool(game_terminated[0]) == bool(game_truncated[0]):
            raise BackfillError(
                f"game seed {seed} must be exactly one of terminated/truncated"
            )
        if bool(game_truncated[0]) and int(actual[-1]) >= int(max_decisions):
            raise BackfillError(
                f"truncated game seed {seed} ends at invalid decision index "
                f"{int(actual[-1])} for sealed cutoff {max_decisions}"
            )
        spans.append(slice(start, stop))
    return spans


def _require_exact(name: str, source: np.ndarray, replayed: np.ndarray, row: int) -> None:
    source = np.asarray(source)
    replayed = np.asarray(replayed)
    if source.shape != replayed.shape or not np.array_equal(
        source, replayed, equal_nan=True
    ):
        detail = f"shape {source.shape} != {replayed.shape}"
        if source.shape == replayed.shape and source.dtype.kind in "biufc":
            detail = f"max_abs_diff={float(np.max(np.abs(source.astype(np.float64) - replayed.astype(np.float64))))}"
        raise BackfillError(f"row {row} replay identity failed for {name}: {detail}")


def _padded(value: np.ndarray, target_shape: tuple[int, ...], *, fill: int | float | bool) -> np.ndarray:
    value = np.asarray(value)
    result = np.full(target_shape, fill, dtype=value.dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, target_shape))
    result[slices] = value[slices]
    return result


def _adapter_target_shape(
    key: str, value: np.ndarray, source_row: np.ndarray
) -> tuple[int, ...]:
    """Choose the storage shape for a regenerated adapter-v5 tensor.

    Legal-action tensors remain padded to the source corpus' fixed legal
    width.  Event tensors are different: adapter-v5 deliberately expands the
    meaningful-history window from 32 to 64 events.  Reusing the adapter-v2
    row shape here silently discarded half of the commissioned history signal.
    """

    if key in {"event_tokens", "event_target_ids", "event_mask"}:
        return tuple(np.asarray(value).shape)
    return tuple(np.asarray(source_row).shape)


def _apply_missing_automatic_transition(
    game: Any,
    *,
    chance_rng: random.Random,
    seed: int,
    decision_index: int,
) -> Any:
    """Replay one omitted row only when its action is mathematically forced."""

    legal = tuple(
        int(action)
        for action in game.playable_action_indices(list(COLORS), None)
    )
    if len(legal) != 1:
        raise BackfillError(
            f"game seed {seed} omits decision {decision_index} with "
            f"{len(legal)} legal actions; only single-legal automatic gaps "
            "are replayable"
        )
    raw_ids = [
        int(action)
        for action in game.playable_action_indices(list(COLORS), None)
    ]
    action_by_id = dict(zip(raw_ids, json.loads(game.playable_actions_json())))
    selected = int(legal[0])
    if selected not in action_by_id:
        raise BackfillError(
            f"game seed {seed} automatic decision {decision_index} lost action JSON"
        )
    return _apply_selected_action(
        game,
        selected,
        colors=COLORS,
        rng=chance_rng,
        correct_rust_chance_spectra=True,
        action_json=action_by_id[selected],
    )


def _replay_archive(
    source: Path,
) -> tuple[dict[str, np.ndarray], int, tuple[int, ...], int]:
    import catanatron_rs

    with np.load(source, allow_pickle=False) as archive:
        required = {
            "game_seed",
            "decision_index",
            "action_taken",
            "player",
            "phase",
            "winner",
            "terminated",
            "truncated",
            "legal_action_ids",
            "legal_action_mask",
            "adapter_version",
            *REPLAY_IDENTITY_KEYS,
            *ADAPTER_OWNED_KEYS,
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise BackfillError(f"{source}: missing required columns {missing}")
        versions = set(map(str, np.asarray(archive["adapter_version"]).tolist()))
        if versions != {RUST_ENTITY_ADAPTER_V2}:
            raise BackfillError(
                f"{source}: expected only {RUST_ENTITY_ADAPTER_V2}, got {sorted(versions)}"
            )
        spans = _game_row_spans(
            archive["game_seed"],
            archive["decision_index"],
            archive["terminated"],
            archive["truncated"],
        )
        output = {key: np.asarray(archive[key]).copy() for key in archive.files}
        replacements: dict[str, list[np.ndarray]] = {
            key: [] for key in ADAPTER_OWNED_KEYS
        }
        rule_nonzero = np.zeros(8, dtype=np.int64)
        live_event_rows = 0

        for span in spans:
            first = int(span.start or 0)
            seed = int(archive["game_seed"][first])
            game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
            chance_rng = random.Random(seed ^ CHANCE_SEED_XOR)
            live_decision = 0
            rust_topology_cache: dict[str, Any] = {}
            for row in range(first, int(span.stop)):
                expected_decision = int(archive["decision_index"][row])
                while live_decision < expected_decision:
                    game = _apply_missing_automatic_transition(
                        game,
                        chance_rng=chance_rng,
                        seed=seed,
                        decision_index=live_decision,
                    )
                    live_decision += 1
                if live_decision != expected_decision:
                    raise BackfillError(
                        f"game seed {seed} replay index {live_decision} passed "
                        f"stored decision {expected_decision}"
                    )
                legal_rust = tuple(
                    sorted(
                        int(action)
                        for action in game.playable_action_indices(
                            list(COLORS), None
                        )
                    )
                )
                snapshot = json.loads(game.json_snapshot())
                raw_ids = [
                    int(action)
                    for action in game.playable_action_indices(list(COLORS), None)
                ]
                raw_actions = json.loads(game.playable_actions_json())
                action_by_id = dict(zip(raw_ids, raw_actions))
                actor = str(game.current_color())

                mapped, v2, _context, snapshot, action_by_id = (
                    _build_public_learner_features(
                        game,
                        legal_rust,
                        colors=COLORS,
                        action_size=ACTION_SIZE,
                        actor=actor,
                        snapshot=snapshot,
                        action_by_id=action_by_id,
                        meaningful_public_history=False,
                        event_history_limit=64,
                        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2,
                    )
                )
                stored_mask = np.asarray(archive["legal_action_mask"][row], dtype=np.bool_)
                stored_legal = tuple(
                    int(value)
                    for value in np.asarray(archive["legal_action_ids"][row])[
                        stored_mask
                    ]
                )
                if mapped != stored_legal:
                    raise BackfillError(
                        f"row {row} legal action identity drift: replay={mapped} "
                        f"source={stored_legal}"
                    )
                if actor != str(archive["player"][row]):
                    raise BackfillError(
                        f"row {row} actor drift: replay={actor} "
                        f"source={archive['player'][row]}"
                    )
                if str(snapshot.get("current_prompt", "")) != str(
                    archive["phase"][row]
                ):
                    raise BackfillError(f"row {row} phase drift")
                for key in REPLAY_IDENTITY_KEYS:
                    _require_exact(key, archive[key][row], v2[key], row)

                _mapped_v5, v5, _ctx_v5, _snap_v5, _actions_v5 = (
                    _build_public_learner_features(
                        game,
                        legal_rust,
                        colors=COLORS,
                        action_size=ACTION_SIZE,
                        actor=actor,
                        snapshot=snapshot,
                        action_by_id=action_by_id,
                        meaningful_public_history=True,
                        meaningful_public_history_schema=(
                            MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
                        ),
                        event_history_limit=64,
                        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
                        rust_topology_cache=rust_topology_cache,
                    )
                )
                for key in ADAPTER_OWNED_KEYS:
                    value = np.asarray(v5[key])
                    target_shape = _adapter_target_shape(
                        key, value, np.asarray(archive[key][row])
                    )
                    if value.shape != target_shape:
                        fill: int | float | bool = 0
                        if key in {"legal_action_target_ids", "event_target_ids"}:
                            fill = -1
                        value = _padded(value, target_shape, fill=fill)
                    replacements[key].append(value)
                rule_nonzero += (
                    np.asarray(v5["global_tokens"])[0, 8:16] != 0
                ).astype(np.int64)
                live_event_rows += int(bool(np.any(v5["event_mask"])))

                selected_policy = int(archive["action_taken"][row])
                hits = [
                    action
                    for action, policy_id in zip(legal_rust, mapped)
                    if int(policy_id) == selected_policy
                ]
                if len(hits) != 1:
                    raise BackfillError(
                        f"row {row} selected action {selected_policy} maps to "
                        f"{len(hits)} native actions"
                    )
                selected_native = int(hits[0])
                game = _apply_selected_action(
                    game,
                    selected_native,
                    colors=COLORS,
                    rng=chance_rng,
                    correct_rust_chance_spectra=True,
                    action_json=action_by_id[selected_native],
                )
                live_decision += 1

            final_row = int(span.stop) - 1
            if bool(archive["terminated"][final_row]):
                while game.winning_color() is None and live_decision < SEALED_MAX_DECISIONS:
                    game = _apply_missing_automatic_transition(
                        game,
                        chance_rng=chance_rng,
                        seed=seed,
                        decision_index=live_decision,
                    )
                    live_decision += 1
                native_winner = game.winning_color()
                expected_winner = str(archive["winner"][final_row])
                if native_winner is None or str(native_winner) != expected_winner:
                    raise BackfillError(
                        f"game seed {seed} terminal replay winner drift: "
                        f"replay={native_winner} source={expected_winner}"
                    )
            else:
                while live_decision < SEALED_MAX_DECISIONS:
                    game = _apply_missing_automatic_transition(
                        game,
                        chance_rng=chance_rng,
                        seed=seed,
                        decision_index=live_decision,
                    )
                    live_decision += 1
                native_winner = game.winning_color()
                if native_winner is not None:
                    raise BackfillError(
                        f"game seed {seed} is labeled truncated but replay "
                        f"terminates with winner {native_winner}"
                    )

        for key, values in replacements.items():
            output[key] = np.stack(values, axis=0).astype(
                archive[key].dtype, copy=False
            )
        output["adapter_version"] = np.full(
            (len(output["game_seed"]),), RUST_ENTITY_ADAPTER_V5
        )
        return output, len(spans), tuple(map(int, rule_nonzero)), live_event_rows


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _process_one(args: tuple[str, str, str | None, str | None]) -> ShardReceipt:
    raw_source, source_relative, raw_output, output_relative = args
    source = Path(raw_source)
    arrays, games, counts, live_event_rows = _replay_archive(source)
    actor_playable_admission = audit_actor_playable_development_cards(
        arrays,
        where=str(source),
    )
    output = Path(raw_output) if raw_output else None
    output_sha = None
    if output is not None:
        if output.resolve() == source.resolve():
            raise BackfillError("refusing to overwrite source shard")
        _write_npz_atomic(output, arrays)
        output_sha = _sha256(output)
    return ShardReceipt(
        source=str(source),
        source_relative_path=source_relative,
        source_sha256=_sha256(source),
        output=str(output) if output else None,
        output_relative_path=output_relative,
        output_sha256=output_sha,
        rows=int(len(arrays["game_seed"])),
        games=int(games),
        v5_rule_slot_nonzero_counts=counts,
        live_event_rows=int(live_event_rows),
        actor_playable_development_card_admission=actor_playable_admission,
    )


def _source_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    paths = sorted(source.rglob("*.npz"))
    if not paths:
        raise BackfillError(f"no .npz shards found in {source}")
    return paths


def _write_json_atomic(path: Path, body: dict[str, Any]) -> None:
    encoded = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    body = dict(payload)
    body["receipt_sha256"] = _canonical_sha256(payload)
    _write_json_atomic(path, body)


def _write_output_manifest(output_dir: Path, receipt_path: Path) -> Path:
    """Bind adapter-v5 public-award authority for memmap admission.

    Deliberately omit ``shards``: the rebuilt tree preserves nested source
    paths and the memmap builder's recursive discovery remains authoritative.
    """

    payload: dict[str, Any] = {
        "public_award_feature_provenance": {
            "schema_version": "public-award-feature-provenance-v1",
            "contract": "authoritative_v1",
            "feature_producer": "catanatron_rs_public_award_v1",
            "native_capability": "public_award_feature_parity",
        },
        "entity_adapter_backfill": {
            "schema": BACKFILL_SCHEMA,
            "source_adapter": RUST_ENTITY_ADAPTER_V2,
            "output_adapter": RUST_ENTITY_ADAPTER_V5,
            "receipt": receipt_path.name,
            "receipt_file_sha256": _sha256(receipt_path),
        },
        "actor_playable_development_card_admission": {
            "schema_version": "actor-playable-development-card-admission-v1",
            "status": "authenticated",
            "receipt": receipt_path.name,
            "receipt_file_sha256": _sha256(receipt_path),
        },
    }
    body = dict(payload)
    body["manifest_sha256"] = _canonical_sha256(payload)
    path = output_dir / "manifest.json"
    _write_json_atomic(path, body)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-shards", type=int)
    args = parser.parse_args(argv)
    if args.audit_only == (args.output_dir is not None):
        parser.error("choose exactly one of --audit-only or --output-dir")
    paths = _source_paths(args.source)
    if args.limit_shards is not None:
        paths = paths[: max(0, int(args.limit_shards))]
    if not paths:
        parser.error("no shards selected")
    jobs = [
        (
            str(path),
            str(path.name if args.source.is_file() else path.relative_to(args.source)),
            None
            if args.audit_only
            else str(
                Path(args.output_dir)
                / (path.name if args.source.is_file() else path.relative_to(args.source))
            ),
            None
            if args.audit_only
            else str(path.name if args.source.is_file() else path.relative_to(args.source)),
        )
        for path in paths
    ]
    output_paths = [job[2] for job in jobs if job[2] is not None]
    if len(output_paths) != len(set(output_paths)):
        raise BackfillError("recursive source paths collide under output directory")
    if int(args.workers) == 1:
        receipts = [_process_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            receipts = list(pool.map(_process_one, jobs))
    shard_receipts = [receipt.as_dict() for receipt in receipts]
    aggregate = {
        "schema": BACKFILL_SCHEMA,
        "source_adapter": RUST_ENTITY_ADAPTER_V2,
        "output_adapter": RUST_ENTITY_ADAPTER_V5,
        "chance_seed_contract": "python_random_game_seed_xor_0xA17E",
        "identity_keys": list(REPLAY_IDENTITY_KEYS),
        "regenerated_keys": list(ADAPTER_OWNED_KEYS),
        "audit_only": bool(args.audit_only),
        "rows": sum(item.rows for item in receipts),
        "games": sum(item.games for item in receipts),
        "shards": shard_receipts,
    }
    if args.audit_only:
        print(json.dumps(aggregate, indent=2, sort_keys=True))
    else:
        receipt_path = Path(args.output_dir) / "adapter_v5_backfill_receipt.json"
        _write_receipt(receipt_path, aggregate)
        manifest_path = _write_output_manifest(Path(args.output_dir), receipt_path)
        print(json.dumps({"receipt": str(receipt_path), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
