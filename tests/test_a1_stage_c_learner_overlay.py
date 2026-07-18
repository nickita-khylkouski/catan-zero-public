from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import a1_stage_c_learner_overlay as overlay
from tools import train_bc


def _write(path: Path, values: np.ndarray) -> None:
    values.tofile(path)


def _completed_q_binding_fixture() -> tuple[dict, dict[str, np.ndarray], str]:
    target_identity = "sha256:" + "a" * 64
    arrays = {
        "row_index": np.asarray([7], dtype=np.int64),
        "game_seed": np.asarray([70], dtype=np.int64),
        "decision_index": np.asarray([3], dtype=np.int64),
        "identity_sha256": np.asarray(["sha256:" + "b" * 64]),
        "legal_action_offsets": np.asarray([0, 2], dtype=np.int64),
        "legal_action_ids_flat": np.asarray([10, 20], dtype=np.int64),
        "completed_q_values_flat": np.asarray([0.25, -0.10], dtype=np.float32),
        "completed_q_mask_flat": np.asarray([True, True]),
        "target_policy_target_identity_sha256": np.asarray([target_identity]),
        "target_reliability_version": np.asarray(
            [overlay.TARGET_RELIABILITY_VERSION], dtype=np.uint8
        ),
        "target_reliability_audited": np.asarray([False]),
        "target_reliability_js_divergence": np.asarray(
            [np.nan], dtype=np.float32
        ),
        "target_reliability_policy_top1_agreement": np.asarray([False]),
        "target_reliability_q_top1_agreement": np.asarray([False]),
        "target_reliability_q_margin_primary": np.asarray(
            [np.nan], dtype=np.float32
        ),
        "target_reliability_q_margin_duplicate": np.asarray(
            [np.nan], dtype=np.float32
        ),
        "target_reliability_confidence": np.asarray([1.0], dtype=np.float32),
    }
    merge = {
        "patch_schema_version": overlay.stage_c.PATCH_SCHEMA,
        "target_policy_target_identity_sha256": target_identity,
        "target_operator_contract": {
            "path": "/sealed/operator.json",
            "file_sha256": "sha256:" + "c" * 64,
        },
        "reliability": {
            "schema_version": overlay.TARGET_RELIABILITY_SCHEMA,
            "audited_rows": 0,
            "unaudited_rows": 1,
            "duplicate_selected_action_applied": False,
        },
    }
    row_identity = overlay._value_sha256(  # noqa: SLF001
        [
            {
                "row_index": 7,
                "game_seed": 70,
                "decision_index": 3,
                "identity_sha256": "sha256:" + "b" * 64,
            }
        ]
    )
    return merge, arrays, row_identity


def _broad_root_inventory(
    *,
    omitted_games: set[int] | None = None,
    short_game: int | None = None,
    omit_phase: str | None = None,
    omit_decision_bin: str | None = None,
) -> dict:
    training_games = np.arange(100, 120, dtype=np.int64)
    validation_games = np.arange(200, 204, dtype=np.int64)
    all_games = np.concatenate((training_games, validation_games))
    phases = np.asarray(overlay.ROOT_BREADTH_REQUIRED_PHASES)
    decision_values = {
        "d000_009": 5,
        "d010_029": 15,
        "d030_059": 35,
        "d060_099": 65,
        "d100_149": 105,
        "d150_199": 155,
        "d200_plus": 205,
    }
    decision_cycle = list(decision_values.values()) + [7]
    selected_games: list[int] = []
    selected_decisions: list[int] = []
    selected_phases: list[str] = []
    for game in all_games.tolist():
        if omitted_games and game in omitted_games:
            continue
        roots = 7 if game == short_game else 8
        for ordinal in range(roots):
            phase = str(phases[ordinal % len(phases)])
            decision = int(decision_cycle[ordinal])
            if omit_phase is not None and phase == omit_phase:
                phase = "PLAY_TURN"
            if (
                omit_decision_bin is not None
                and decision == decision_values[omit_decision_bin]
            ):
                decision = decision_values["d000_009"]
            selected_games.append(game)
            selected_decisions.append(decision)
            selected_phases.append(phase)
    return overlay._stage_c_root_breadth_inventory(  # noqa: SLF001
        corpus_game_seeds=all_games,
        validation_game_seeds=validation_games,
        selected_game_seeds=np.asarray(selected_games, dtype=np.int64),
        selected_decision_indices=np.asarray(selected_decisions, dtype=np.int64),
        selected_phases=np.asarray(selected_phases),
    )


def _inventory_selected_rows(inventory: dict) -> int:
    return sum(
        int(scope["selected_root_count"]) for scope in inventory["scopes"].values()
    )


def test_stage_c_root_breadth_inventory_passes_only_broad_realized_roots() -> None:
    inventory = _broad_root_inventory()

    assert inventory["passed"] is True
    assert inventory["failures"] == []
    verified = overlay._verify_stage_c_root_breadth_inventory(  # noqa: SLF001
        inventory,
        selected_rows=_inventory_selected_rows(inventory),
    )
    assert verified == inventory
    assert verified["scopes"]["training"]["unique_game_fraction"] == 1.0
    assert verified["scopes"]["training"]["roots_per_represented_game"]["minimum"] == 8


@pytest.mark.parametrize(
    ("kwargs", "failure"),
    [
        ({"omitted_games": {100, 101}}, "training:unique_game_fraction"),
        ({"short_game": 100}, "training:minimum_roots_per_represented_game"),
        ({"omit_phase": "DISCARD"}, "training:phase:DISCARD"),
        ({"omit_decision_bin": "d200_plus"}, "training:decision_bin:d200_plus"),
    ],
)
def test_stage_c_root_breadth_inventory_fails_closed(
    kwargs: dict, failure: str
) -> None:
    inventory = _broad_root_inventory(**kwargs)

    assert inventory["passed"] is False
    assert failure in inventory["failures"]
    with pytest.raises(overlay.OverlayError, match="failed or drifted"):
        overlay._verify_stage_c_root_breadth_inventory(  # noqa: SLF001
            inventory,
            selected_rows=_inventory_selected_rows(inventory),
        )


def test_stage_c_root_breadth_verifier_recomputes_semantic_failures() -> None:
    inventory = _broad_root_inventory(omitted_games={100, 101})
    inventory["passed"] = True
    inventory["failures"] = []
    inventory["inventory_sha256"] = overlay._value_sha256(  # noqa: SLF001
        {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    )

    with pytest.raises(overlay.OverlayError, match="failed or drifted"):
        overlay._verify_stage_c_root_breadth_inventory(  # noqa: SLF001
            inventory,
            selected_rows=_inventory_selected_rows(inventory),
        )


def test_policy_projection_disables_old_targets_and_maps_action_ids(
    tmp_path: Path,
) -> None:
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
    _write(base / "root_value.dat", np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    _write(base / "root_value_mask.dat", np.ones(3, dtype=np.bool_))
    _write(
        base / "root_prior_value.dat",
        np.asarray([-0.1, -0.2, -0.3], dtype=np.float32),
    )
    _write(base / "root_prior_value_mask.dat", np.ones(3, dtype=np.bool_))
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
            "root_value": {"kind": "fixed", "dtype": "float32", "inner_shape": []},
            "root_value_mask": {"kind": "fixed", "dtype": "bool", "inner_shape": []},
            "root_prior_value": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
            "root_prior_value_mask": {
                "kind": "fixed",
                "dtype": "bool",
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
    paired = {
        "root_value",
        "root_value_mask",
        "root_prior_value",
        "root_prior_value_mask",
    }
    overlay._hardlink_payloads(  # noqa: SLF001
        base,
        derived,
        meta["columns"],
        rewritten_columns=set(overlay.REWRITTEN_COLUMNS) | paired,
    )
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
        # Completed-Q covers every action and remains distinct from sparse/raw
        # visited-Q target_scores.
        "completed_q_values_flat": np.asarray([0.3, -0.1, 0.2], dtype=np.float32),
        "completed_q_mask_flat": np.asarray([True, True, True]),
        "root_value": np.asarray([0.75], dtype=np.float32),
        "root_value_mask": np.asarray([True]),
        "root_prior_value": np.asarray([0.25], dtype=np.float32),
        "root_prior_value_mask": np.asarray([True]),
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
    completed_q = np.fromfile(
        derived / f"{overlay.COMPLETED_Q_VALUE_COLUMN}.dat", dtype=np.float32
    )
    completed_q_mask = np.fromfile(
        derived / f"{overlay.COMPLETED_Q_MASK_COLUMN}.dat", dtype=np.bool_
    )
    teacher_codes = np.fromfile(derived / "teacher_name.codes.dat", dtype=np.int32)
    root_values = np.fromfile(derived / "root_value.dat", dtype=np.float32)
    root_priors = np.fromfile(derived / "root_prior_value.dat", dtype=np.float32)
    assert weights.tolist() == [0.0, 1.0, 0.0]
    assert not target_mask[:2].any() and not target_mask[5:].any()
    assert targets[2:5] == pytest.approx([0.0, 0.4, 0.6])
    assert target_mask[2:5].all()
    assert priors[2:5] == pytest.approx([0.2, 0.3, 0.5])
    assert scores[2:5] == pytest.approx([1.0, 2.0, 3.0])
    assert completed_q[2:5] == pytest.approx([-0.1, 0.2, 0.3])
    assert completed_q_mask[2:5].all()
    assert np.all(targets[:2] == 0.0) and np.all(targets[5:] == 0.0)
    assert np.isnan(scores[:2]).all() and np.isnan(scores[5:]).all()
    assert np.isnan(completed_q[:2]).all() and np.isnan(completed_q[5:]).all()
    assert not completed_q_mask[:2].any() and not completed_q_mask[5:].any()
    assert not np.array_equal(scores[2:5], completed_q[2:5])
    assert teacher_codes.tolist() == [0, 1, 0]
    assert root_values.tolist() == pytest.approx([0.1, 0.75, 0.3])
    assert root_priors.tolist() == pytest.approx([-0.1, 0.25, -0.3])
    assert set(evidence["authoritative_search_fixed_columns"]) >= paired
    assert evidence["completed_q_rows"] == 1
    assert evidence["completed_q_legal_actions"] == 3
    assert evidence["completed_q_target_scores_separate"] is True
    assert meta["columns"][overlay.COMPLETED_Q_VALUE_COLUMN] == {
        "kind": "ragged2d",
        "dtype": "float32",
    }
    assert meta["columns"]["teacher_name"]["categories"] == [
        "historical",
        overlay.POLICY_TEACHER,
    ]


def test_completed_q_binding_is_operator_bound_and_objective_inert() -> None:
    merge, arrays, row_identity = _completed_q_binding_fixture()

    binding = overlay._completed_q_binding(  # noqa: SLF001
        merge=merge,
        arrays=arrays,
        row_identity_sha256=row_identity,
    )

    assert binding["columns"] == {
        "values": overlay.COMPLETED_Q_VALUE_COLUMN,
        "mask": overlay.COMPLETED_Q_MASK_COLUMN,
    }
    assert binding["row_identity"]["ordered_row_identity_sha256"] == row_identity
    assert binding["operator_identity"][
        "target_policy_target_identity_sha256"
    ] == merge["target_policy_target_identity_sha256"]
    assert binding["operator_identity"]["legacy_or_unbound_q_allowed"] is False
    assert binding["reliability_identity"]["schema_version"] == (
        overlay.TARGET_RELIABILITY_SCHEMA
    )
    assert binding["semantics"]["target_scores_relation"] == (
        "separate_raw_visited_q_column_never_overwritten"
    )
    assert binding["semantics"]["default_learner_objective"] == (
        "none_evidence_only"
    )


@pytest.mark.parametrize("drift", ["operator", "mask", "legacy_patch"])
def test_completed_q_binding_rejects_unbound_or_incomplete_q(drift: str) -> None:
    merge, arrays, row_identity = _completed_q_binding_fixture()
    if drift == "operator":
        arrays["target_policy_target_identity_sha256"] = np.asarray(
            ["sha256:" + "e" * 64]
        )
    elif drift == "mask":
        arrays["completed_q_mask_flat"] = np.asarray([True, False])
    else:
        merge["patch_schema_version"] = overlay.stage_c.PATCH_SCHEMA_V2

    with pytest.raises(
        overlay.OverlayError,
        match="row/operator/reliability authority",
    ):
        overlay._completed_q_binding(  # noqa: SLF001
            merge=merge,
            arrays=arrays,
            row_identity_sha256=row_identity,
        )


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


def test_clean_stage_c_recipe_commissions_new_adapters() -> None:
    from tools import a1_b200_stage_c_learner_campaign as campaign

    recipe = campaign._recipe()  # noqa: SLF001
    # Dataset metadata must not silently mutate the optimizer surface. Legacy
    # isolation remains available through the explicit freeze groups below.
    assert "freeze_modules" not in recipe
    assert recipe["value_trunk_grad_scale"] == pytest.approx(0.1)
    assert recipe["soft_target_min_legal_coverage"] == pytest.approx(1.0)
    assert campaign.TRAINABLE_ADAPTER_MODULES == {
        "legal_action_value_residual_proj",
        "legal_action_value_static_proj",
        "meaningful_history_residual_gate",
        "public_card_count_residual",
        "static_action_residual_proj",
    }
    assert campaign.FEATURE_SIGNAL_MODULES == {
        "event_encoder",
        "legal_action_value_residual_proj",
        "legal_action_value_static_proj",
        "meaningful_history_residual_gate",
        "public_card_count_residual",
        "static_action_residual_proj",
    }
    assert campaign.EFFECTIVE_FEATURE_CONTRACT["static_action_residual"] is True
    assert campaign.EFFECTIVE_FEATURE_CONTRACT["legal_action_value_residual"] is True
    assert campaign.EFFECTIVE_FEATURE_CONTRACT["meaningful_public_history"] is True
    assert (
        campaign.MINIMUM_FEATURE_SIGNAL_OBSERVATIONS
        * campaign.TRAIN_DIAGNOSTIC_CADENCE_BATCHES
        == campaign.MAX_STEPS
    )
    assert train_bc.ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["public_card_residual"] == (
        "public_card_count_residual",
    )
    assert train_bc.ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["meaningful_history_gate"] == (
        "meaningful_history_residual_gate",
        "meaningful_history_ordered_gate",
        "meaningful_history_sequence",
        "meaningful_history_target_proj",
    )
