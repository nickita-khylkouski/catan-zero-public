from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import a1_stage_c_learner_overlay as overlay
from tools import train_bc


def _write(path: Path, values: np.ndarray) -> None:
    values.tofile(path)


def test_policy_projection_disables_old_targets_and_maps_action_ids(tmp_path: Path) -> None:
    base = tmp_path / "base"
    derived = tmp_path / "derived"
    base.mkdir()
    derived.mkdir()
    offsets = np.asarray([0, 2, 5, 7], dtype=np.int64)
    legal = np.asarray([1, 2, 10, 20, 30, 4, 5], dtype=np.int64)
    _write(base / "row_offsets.dat", offsets)
    _write(base / "game_seed.dat", np.asarray([100, 200, 300], dtype=np.int64))
    _write(base / "decision_index.dat", np.asarray([1, 2, 3], dtype=np.int64))
    _write(base / "legal_action_ids.dat", legal)
    _write(base / "value_target.dat", np.asarray([0.1, -0.2, 0.3], dtype=np.float32))
    _write(base / "teacher_name.codes.dat", np.zeros(3, dtype=np.int32))

    meta = {
        "row_count": 3,
        "flat_count": 7,
        "columns": {
            "game_seed": {"kind": "fixed", "dtype": "int64", "inner_shape": []},
            "decision_index": {
                "kind": "fixed",
                "dtype": "int64",
                "inner_shape": [],
            },
            "legal_action_ids": {"kind": "ragged2d", "dtype": "int64"},
            "value_target": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
            "policy_weight_multiplier": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
            "prior_policy": {"kind": "ragged2d", "dtype": "float32"},
            "target_policy": {"kind": "ragged2d", "dtype": "float32"},
            "target_policy_mask": {"kind": "ragged2d", "dtype": "bool"},
            "target_scores": {"kind": "ragged2d", "dtype": "float32"},
            "target_scores_mask": {"kind": "ragged2d", "dtype": "bool"},
            "teacher_name": {"kind": "string", "categories": ["historical"]},
        },
    }
    overlay._hardlink_payloads(base, derived, meta["columns"])  # noqa: SLF001
    patch = {
        "row_index": np.asarray([1], dtype=np.int64),
        "game_seed": np.asarray([200], dtype=np.int64),
        "decision_index": np.asarray([2], dtype=np.int64),
        "legal_action_offsets": np.asarray([0, 3], dtype=np.int64),
        # Deliberately differs from base order [10, 20, 30].
        "legal_action_ids_flat": np.asarray([30, 10, 20], dtype=np.int64),
        # Exact-zero probability remains authenticated support; it must not be
        # reinterpreted as a missing label and routed to historical action_taken.
        "target_policy_flat": np.asarray([0.6, 0.0, 0.4], dtype=np.float32),
        "target_policy_mask_flat": np.asarray([True, True, True]),
        "prior_policy_flat": np.asarray([0.5, 0.2, 0.3], dtype=np.float32),
        "target_scores_flat": np.asarray([3.0, 1.0, 2.0], dtype=np.float32),
        "target_scores_mask_flat": np.asarray([True, True, True]),
    }

    evidence = overlay._project_policy_patch(  # noqa: SLF001
        base_root=base,
        output_root=derived,
        meta=meta,
        patch=patch,
    )

    assert evidence["selected_rows"] == 1
    assert evidence["base_value_rows_retained"] == 3
    assert (base / "value_target.dat").stat().st_ino == (
        derived / "value_target.dat"
    ).stat().st_ino
    weights = np.fromfile(derived / "policy_weight_multiplier.dat", dtype=np.float32)
    targets = np.fromfile(derived / "target_policy.dat", dtype=np.float32)
    target_mask = np.fromfile(derived / "target_policy_mask.dat", dtype=np.bool_)
    priors = np.fromfile(derived / "prior_policy.dat", dtype=np.float32)
    scores = np.fromfile(derived / "target_scores.dat", dtype=np.float32)
    teacher_codes = np.fromfile(derived / "teacher_name.codes.dat", dtype=np.int32)
    assert weights.tolist() == [0.0, 1.0, 0.0]
    assert not target_mask[:2].any() and not target_mask[5:].any()
    assert targets[2:5] == pytest.approx([0.0, 0.4, 0.6])
    assert target_mask[2:5].all()
    assert priors[2:5] == pytest.approx([0.2, 0.3, 0.5])
    assert scores[2:5] == pytest.approx([1.0, 2.0, 3.0])
    assert np.all(targets[:2] == 0.0) and np.all(targets[5:] == 0.0)
    assert np.isnan(scores[:2]).all() and np.isnan(scores[5:]).all()
    assert teacher_codes.tolist() == [0, 1, 0]
    assert meta["columns"]["teacher_name"]["categories"] == [
        "historical",
        overlay.POLICY_TEACHER,
    ]


def test_unique_source_row_count_is_exact_and_fail_closed() -> None:
    local = {7, 2, 7, 4}
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    assert train_bc._reduce_unique_row_count(local, total_rows=8, ddp=ddp) == 3
    with pytest.raises(ValueError, match="outside the corpus"):
        train_bc._reduce_unique_row_count({8}, total_rows=8, ddp=ddp)


def test_stage_c_one_hot_full_support_is_soft_and_fail_closed() -> None:
    torch = pytest.importorskip("torch")
    data = {
        "legal_action_ids": np.asarray([[10, 20, 30]], dtype=np.int64),
        "target_policy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "target_policy_mask": np.asarray([[True, True, True]]),
        "teacher_name": np.asarray([overlay.POLICY_TEACHER]),
        "policy_weight_multiplier": np.asarray([1.0], dtype=np.float32),
    }
    target, support = train_bc._soft_target_array(  # noqa: SLF001
        data, np.asarray([0], dtype=np.int64), 1.0, "policy"
    )
    assert target.tolist() == [[1.0, 0.0, 0.0]]
    assert support.tolist() == [[True, True, True]]
    usable = train_bc._has_distillation_distribution(  # noqa: SLF001
        target,
        support,
        legal_action_ids=data["legal_action_ids"],
        min_legal_coverage=1.0,
    )
    assert usable.tolist() == [True]
    train_bc._require_stage_c_soft_targets(  # noqa: SLF001
        data, np.asarray([0]), torch.as_tensor(usable), source="policy"
    )
    with pytest.raises(ValueError, match="action_taken fallback"):
        train_bc._require_stage_c_soft_targets(  # noqa: SLF001
            data,
            np.asarray([0]),
            torch.as_tensor([False]),
            source="policy",
        )


def test_two_stage_c_sampling_arms_share_roots_but_not_measure() -> None:
    subset = {
        "row_index": np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
        "stratum": np.asarray(["a", "a", "b", "b", "b"]),
        "phase": np.asarray(["P", "P", "Q", "Q", "Q"]),
        "legal_width": np.asarray([2, 2, 8, 8, 8], dtype=np.int64),
    }
    patch = {"row_index": subset["row_index"]}
    export = {
        "sampling_population": {
            "candidate_counts_by_stratum": {"a": 20, "b": 300},
            "selected_counts_by_stratum": {"a": 2, "b": 3},
        }
    }
    validation = np.asarray([False, False, False, False, True])
    balanced, balanced_report = overlay._selected_sampling_weights(  # noqa: SLF001
        export=export,
        subset=subset,
        patch=patch,
        selected_validation=validation,
        arm="STRATEGIC_BALANCED",
        production_weight_cap=4.0,
    )
    production, production_report = overlay._selected_sampling_weights(  # noqa: SLF001
        export=export,
        subset=subset,
        patch=patch,
        selected_validation=validation,
        arm="PRODUCTION_WEIGHTED",
        production_weight_cap=4.0,
    )
    assert balanced.tolist() == [1.0] * 5
    assert np.mean(production[~validation]) == pytest.approx(1.0)
    assert np.max(production[~validation]) <= 4.0
    assert production[2] > production[0]
    assert np.mean(production[validation]) == pytest.approx(1.0)
    assert balanced_report["arm"] == "STRATEGIC_BALANCED"
    assert production_report["arm"] == "PRODUCTION_WEIGHTED"
    assert production_report["normalization_scope"] == (
        "training_and_validation_roots_independently"
    )


def test_production_stage_c_validation_preserves_its_weighted_measure() -> None:
    subset = {
        "row_index": np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64),
        "stratum": np.asarray(["a", "b", "a", "b", "a", "b"]),
        "phase": np.asarray(["P", "Q", "P", "Q", "P", "Q"]),
        "legal_width": np.asarray([2, 8, 2, 8, 2, 8], dtype=np.int64),
    }
    export = {
        "sampling_population": {
            "candidate_counts_by_stratum": {"a": 20, "b": 300},
            "selected_counts_by_stratum": {"a": 3, "b": 3},
        }
    }
    validation = np.asarray([False, False, False, False, True, True])
    weights, report = overlay._selected_sampling_weights(  # noqa: SLF001
        export=export,
        subset=subset,
        patch={"row_index": subset["row_index"]},
        selected_validation=validation,
        arm="PRODUCTION_WEIGHTED",
        production_weight_cap=4.0,
    )

    assert np.mean(weights[validation]) == pytest.approx(1.0)
    assert weights[5] > weights[4]
    assert report["final_validation_weights"]["max"] > 1.0


def test_clean_stage_c_recipe_freezes_only_new_adapters() -> None:
    from tools import a1_b200_stage_c_learner_campaign as campaign

    recipe = campaign._recipe()  # noqa: SLF001
    # The authenticated Stage-C overlay triggers this freeze inside train_bc;
    # it is not an unsealed generic one-dose recipe override.
    assert "freeze_modules" not in recipe
    assert recipe["value_trunk_grad_scale"] == pytest.approx(0.1)
    assert recipe["soft_target_min_legal_coverage"] == pytest.approx(1.0)
    assert train_bc.ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS[
        "public_card_residual"
    ] == ("public_card_count_residual",)
    assert train_bc.ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS[
        "meaningful_history_gate"
    ] == (
        "meaningful_history_residual_gate",
        "meaningful_history_ordered_gate",
        "meaningful_history_sequence",
    )
