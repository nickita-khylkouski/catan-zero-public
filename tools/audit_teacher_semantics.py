from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from factory_common import parse_track, write_json
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay teacher shards by game_seed/decision_index and verify that "
            "saved action IDs still match the runtime action catalog."
        )
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-seeds", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--require-complete-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-history-features", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = _load_rows([Path(path) for path in args.data], max_rows=int(args.max_rows))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["game_seed"])].append(row)

    config = parse_track(
        args.track,
        vps_to_win=int(args.vps_to_win),
        use_graph_history_features=bool(args.graph_history_features),
    )
    summary = {
        "data": args.data,
        "track": args.track,
        "vps_to_win": int(args.vps_to_win),
        "graph_history_features": bool(args.graph_history_features),
        "loaded_rows": len(rows),
        "candidate_seeds": len(grouped),
        "audited_seeds": 0,
        "audited_rows": 0,
        "skipped_incomplete_prefix": 0,
        "mismatches": [],
    }

    for seed, seed_rows in sorted(grouped.items())[: int(args.max_seeds)]:
        by_decision = {int(row["decision_index"]): row for row in seed_rows}
        if args.require_complete_prefix:
            max_decision = max(by_decision) if by_decision else -1
            missing = [idx for idx in range(max_decision + 1) if idx not in by_decision]
            if missing:
                summary["skipped_incomplete_prefix"] += 1
                continue
        result = _audit_seed(seed, by_decision, config)
        summary["audited_seeds"] += 1
        summary["audited_rows"] += int(result["audited_rows"])
        summary["mismatches"].extend(result["mismatches"])
        if summary["mismatches"]:
            break

    if args.out:
        write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["mismatches"]:
        raise SystemExit(1)


def _audit_seed(seed: int, by_decision: dict[int, dict[str, Any]], config) -> dict[str, Any]:
    env = ColonistMultiAgentEnv(config)
    mismatches: list[dict[str, Any]] = []
    audited_rows = 0
    try:
        _, info = env.reset(seed=int(seed))
        decision = 0
        while decision in by_decision:
            row = by_decision[decision]
            player = str(row["player"])
            action = int(row["action_taken"])
            valid = tuple(int(value) for value in info.get("valid_actions", ()))
            runtime_player = str(info.get("current_player", ""))
            row_valid = tuple(int(value) for value in row["legal_action_ids"] if int(value) >= 0)
            row_version = str(row.get("action_mask_version", ""))
            runtime_version = str(info.get("action_mask_version", ""))
            if player != runtime_player:
                mismatches.append(_mismatch(seed, decision, "player", row, runtime_player))
                break
            if tuple(row_valid) != tuple(valid):
                mismatches.append(_mismatch(seed, decision, "valid_actions", row, valid[:20]))
                break
            if row_version and runtime_version and row_version != runtime_version:
                mismatches.append(_mismatch(seed, decision, "action_mask_version", row, runtime_version))
                break
            if action not in set(valid):
                mismatches.append(_mismatch(seed, decision, "action_not_runtime_legal", row, valid[:20]))
                break
            description = env.describe_action(action)
            if not isinstance(description, dict) or int(description.get("index", -1)) != action:
                mismatches.append(_mismatch(seed, decision, "description", row, description))
                break
            _, _, terminated, truncated, info = env.step(action)
            audited_rows += 1
            decision += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    return {"audited_rows": audited_rows, "mismatches": mismatches}


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


def _load_rows(roots: list[Path], *, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for shard in _shards(root):
            arrays = _load_npz(shard)
            n = len(arrays["action_taken"])
            for idx in range(n):
                rows.append(
                    {
                        "game_seed": int(_field(arrays, ("game_seed", "seed"), idx, 0)),
                        "decision_index": int(_field(arrays, ("decision_index",), idx, -1)),
                        "player": str(_field(arrays, ("player",), idx, "")),
                        "teacher_name": str(_field(arrays, ("teacher_name", "teacher"), idx, "")),
                        "phase": str(_field(arrays, ("phase",), idx, "")),
                        "action_taken": int(_field(arrays, ("action_taken", "action"), idx, -1)),
                        "legal_action_ids": np.asarray(
                            _field(arrays, ("legal_action_ids", "valid"), idx, ()),
                            dtype=np.int64,
                        ),
                        "action_mask_version": str(_field(arrays, ("action_mask_version",), idx, "")),
                    }
                )
                if len(rows) >= max_rows:
                    return rows
    return rows


def _shards(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("teacher_shard_*.npz")) + sorted(root.glob("teacher_shard_*.npz.zst"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".zst":
        import zstandard as zstd

        data = zstd.ZstdDecompressor().decompress(path.read_bytes())
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _field(arrays: dict[str, np.ndarray], keys: tuple[str, ...], idx: int, default: Any) -> Any:
    for key in keys:
        if key in arrays:
            return arrays[key][idx]
    return default


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
