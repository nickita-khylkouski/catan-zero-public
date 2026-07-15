from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np

from tools import a1_target_eligibility_inventory as inventory

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools import prelaunch_guard  # noqa: E402


def _write_corpus(tmp_path: Path, decisions: list[int]) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    rows = len(decisions)
    columns: dict[str, dict[str, object]] = {}

    fixed = {
        "game_seed": np.asarray([7] * rows, dtype=np.int64),
        "decision_index": np.asarray(decisions, dtype=np.int32),
        "action_taken": np.arange(rows, dtype=np.int16),
        "phase": None,
        "player": None,
        "terminated": np.asarray([True] * rows, dtype=np.bool_),
        "truncated": np.asarray([False] * rows, dtype=np.bool_),
        "policy_weight_multiplier": np.asarray([1.0] * rows, dtype=np.float32),
        "used_full_search": np.asarray([True] * rows, dtype=np.bool_),
        "is_forced": np.asarray([False] * rows, dtype=np.bool_),
    }
    for name in sorted(inventory.ROUND_TRIP_COLUMNS):
        fixed.setdefault(name, np.zeros((rows,), dtype=np.float16))
    for name, value in fixed.items():
        if value is None:
            continue
        value.tofile(root / f"{name}.dat")
        columns[name] = {
            "kind": "fixed",
            "dtype": value.dtype.str,
            "inner_shape": list(value.shape[1:]),
        }
    for name, values in {
        "phase": ["play_turn"] * rows,
        "player": ["RED"] * rows,
        "target_information_regime": [inventory.PIMC_REGIME] * rows,
    }.items():
        np.zeros(rows, dtype=np.int32).tofile(root / f"{name}.codes.dat")
        columns[name] = {"kind": "string", "categories": [values[0]]}

    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": rows,
        "columns": columns,
        "payload_inventory_sha256": "sha256:" + "0" * 64,
    }
    (root / "corpus_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


def test_inventory_exposes_active_pimc_targets_and_complete_trace(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path, [0, 1, 2])
    result = inventory.inspect_memmap(
        label="old", corpus_dir=root, required_regime=inventory.COHERENT_REGIME
    )
    assert result["policy_active_target_regime_rows"] == {
        inventory.PIMC_REGIME: 3
    }
    assert result["incompatible_policy_active_rows"] == 3
    assert result["exact_root_reanalysis"]["full_corpus_replayable"] is True


def test_missing_opponent_decisions_blocks_full_reanalysis(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path, [0, 1, 6])
    result = inventory.inspect_memmap(
        label="partial", corpus_dir=root, required_regime=inventory.COHERENT_REGIME
    )
    replay = result["exact_root_reanalysis"]
    assert replay["complete_action_trace_game_count"] == 0
    assert replay["incomplete_action_trace_game_count"] == 1
    assert replay["full_corpus_replayable"] is False
    assert "noncontiguous_or_incomplete_action_trajectory" in replay["blockers"]


def test_sealed_rd_contract_and_nullable_override_guard() -> None:
    repo = Path(__file__).resolve().parents[1]
    contract = (
        repo
        / "configs/operations/a1-target-identity-coherent-n128-rd-v1/contract.json"
    )
    result = inventory.inspect_rd_contract(contract)
    assert result["contract_eligible_to_launch"] is True
    assert result["target_information_regime"] == inventory.COHERENT_REGIME
    assert result["total_games"] == 8192

    parser = argparse.ArgumentParser()
    parser.add_argument("--required")
    parser.add_argument("--nullable")
    guarded = prelaunch_guard.guard_cli_flag_lint(
        ["--required", "yes", "--nullable", "surprise"],
        ["--required"],
        parser=parser,
        expected_values={"--required": "yes"},
        forbidden_flags=["--nullable"],
    )
    assert guarded.passed is False
    assert guarded.details["forbidden_flags"] == ["--nullable"]
