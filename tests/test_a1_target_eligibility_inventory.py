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


def _write_corpus(
    tmp_path: Path,
    decisions: list[int],
    *,
    policy_weights: list[float] | None = None,
    used_full_search: list[bool] | None = None,
    is_forced: list[bool] | None = None,
    simulations_used: list[int] | None = None,
) -> Path:
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
        "policy_weight_multiplier": np.asarray(
            policy_weights if policy_weights is not None else [1.0] * rows,
            dtype=np.float32,
        ),
        "used_full_search": np.asarray(
            used_full_search if used_full_search is not None else [True] * rows,
            dtype=np.bool_,
        ),
        "is_forced": np.asarray(
            is_forced if is_forced is not None else [False] * rows,
            dtype=np.bool_,
        ),
    }
    if simulations_used is not None:
        fixed["simulations_used"] = np.asarray(simulations_used, dtype=np.int32)
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


def test_inventory_accepts_authenticated_bounded_fast_policy_activity(
    tmp_path: Path,
) -> None:
    root = _write_corpus(
        tmp_path,
        [0, 1, 2],
        policy_weights=[1.0, 0.125, 0.0],
        used_full_search=[True, False, False],
        simulations_used=[64, 16, 0],
    )
    result = inventory.inspect_memmap(
        label="bounded-fast",
        corpus_dir=root,
        required_regime=inventory.PIMC_REGIME,
    )

    assert result["policy_active_rows"] == 2
    assert result["fast_search_policy_active_rows"] == 1
    assert result["policy_active_rule_mismatch_rows"] == 0
    assert result["policy_targets_eligible_for_requested_learner"] is True
    assert "bounded_fast_simulation_confidence" in result[
        "policy_activation_evidence"
    ]


def test_inventory_rejects_fast_policy_activity_without_matching_provenance(
    tmp_path: Path,
) -> None:
    root = _write_corpus(
        tmp_path,
        [0, 1],
        policy_weights=[1.0, 0.5],
        used_full_search=[True, False],
        simulations_used=[64, 16],
    )
    result = inventory.inspect_memmap(
        label="bad-fast",
        corpus_dir=root,
        required_regime=inventory.PIMC_REGIME,
    )

    assert result["policy_active_rule_mismatch_rows"] == 1
    assert result["policy_targets_eligible_for_requested_learner"] is False
    aggregate = inventory.build_inventory(
        corpora=(("bad-fast", root),),
        composite=None,
        rd_contract=None,
        required_regime=inventory.PIMC_REGIME,
    )["aggregate"]
    assert aggregate["policy_activation_invalid_components"] == ["bad-fast"]
    assert aggregate["policy_targets_eligible_for_requested_learner"] is False
    assert aggregate["decision"] == "generate_new_coherent_targets"


def test_inventory_preserves_legacy_zero_fast_policy_compatibility(
    tmp_path: Path,
) -> None:
    root = _write_corpus(
        tmp_path,
        [0, 1],
        policy_weights=[1.0, 0.0],
        used_full_search=[True, False],
    )
    result = inventory.inspect_memmap(
        label="legacy-fast-zero",
        corpus_dir=root,
        required_regime=inventory.PIMC_REGIME,
    )

    assert result["policy_active_rule_mismatch_rows"] == 0
    assert result["fast_search_policy_active_rows"] == 0
    assert result["policy_targets_eligible_for_requested_learner"] is True


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


def test_policy_operator_inventory_rejects_mixed_search_teachers() -> None:
    groups = [
        {
            "scope": "fresh",
            "category": "n128",
            "operator_sha256": "sha256:" + "1" * 64,
        },
        {
            "scope": "fresh",
            "category": "n256",
            "operator_sha256": "sha256:" + "2" * 64,
        },
    ]

    result = inventory._policy_operator_identity_inventory(  # noqa: SLF001
        groups=groups,
        policy_distillation_component_ids={"n128", "n256"},
        policy_active_component_ids={"n128", "n256"},
    )

    assert result["mixed_policy_target_operators"] is True
    assert result["policy_operator_uniform"] is False
    assert result["realized_operator_sha256"] == [
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
    ]


def test_manifest_operator_identity_includes_checkpoint_and_cli_search_fields(
    tmp_path: Path,
) -> None:
    records = []
    for index, (checkpoint, n_full) in enumerate(
        (("sha256:" + "a" * 64, 128), ("sha256:" + "b" * 64, 256))
    ):
        path = tmp_path / f"manifest-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "producer_checkpoint_sha256": checkpoint,
                    "target_information_regime": inventory.COHERENT_REGIME,
                    "cli_args": {
                        "n_full": n_full,
                        "n_fast": 16,
                        "p_full": 0.25,
                        "c_scale": 0.03,
                        "coherent_public_belief_search": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "category": f"arm-{index}",
                "artifact": {
                    "path": str(path),
                    "file_sha256": inventory._file_sha256(path),  # noqa: SLF001
                },
            }
        )

    groups = inventory._manifest_operator_groups(  # noqa: SLF001
        {"fresh_generation_manifests": records}
    )

    assert len(groups) == 2
    assert {group["operator"]["n_full"] for group in groups} == {128, 256}
    assert {
        group["operator"]["producer_checkpoint_sha256"] for group in groups
    } == {"sha256:" + "a" * 64, "sha256:" + "b" * 64}
    assert len({group["operator_sha256"] for group in groups}) == 2
