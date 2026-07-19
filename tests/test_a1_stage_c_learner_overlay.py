from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools import a1_stage_c_learner_overlay as overlay
from tools import train_bc


def _write(path: Path, values: np.ndarray) -> None:
    values.tofile(path)


def test_seal_rewritten_payloads_makes_only_named_files_read_only(
    tmp_path: Path,
) -> None:
    rewritten = tmp_path / "target_policy.dat"
    preserved = tmp_path / "obs.dat"
    rewritten.write_bytes(b"new-policy")
    preserved.write_bytes(b"shared-observation")
    os.chmod(rewritten, 0o644)
    os.chmod(preserved, 0o644)

    overlay._seal_rewritten_payloads(  # noqa: SLF001
        tmp_path, {rewritten.name}
    )

    assert rewritten.read_bytes() == b"new-policy"
    assert rewritten.stat().st_mode & 0o222 == 0
    assert preserved.stat().st_mode & 0o222 != 0


def test_materialized_payload_auth_cache_is_ready_for_first_learner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "overlay"
    corpus.mkdir()
    payloads = {
        "obs.dat": b"authenticated-observations",
        "row_offsets.dat": np.asarray([0, 1], dtype=np.int64).tobytes(),
    }
    for filename, content in payloads.items():
        path = corpus / filename
        path.write_bytes(content)
        os.chmod(path, 0o444)
    inventory = [
        {
            "filename": filename,
            "size_bytes": len(content),
            "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        }
        for filename, content in sorted(payloads.items())
    ]
    meta = {
        "columns": {
            "obs": {
                "kind": "fixed",
                "dtype": "uint8",
                "inner_shape": [],
            },
            "row_offsets": {
                "kind": "row_offsets",
                "dtype": "int64",
                "inner_shape": [],
            },
        },
        "payload_inventory_schema": train_bc.MEMMAP_PAYLOAD_INVENTORY_SCHEMA,
        "payload_inventory": inventory,
        "payload_inventory_sha256": train_bc._canonical_json_sha256(  # noqa: SLF001
            inventory
        ),
    }
    (corpus / "corpus_meta.json").write_text(
        json.dumps(meta, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setenv(
        "TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR", str(tmp_path / "payload-auth-cache")
    )

    assert overlay._prime_materialized_payload_auth_cache(  # noqa: SLF001
        corpus, meta
    )
    first = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["status"] for event in first] == ["miss"]

    train_bc._validate_memmap_payload_inventory(corpus, meta)  # noqa: SLF001
    second = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["status"] for event in second] == ["hit"]
    assert second[0]["bytes_avoided"] == sum(map(len, payloads.values()))


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
    population_omitted_games: set[int] | None = None,
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
        corpus_game_seeds=np.asarray(
            [
                game
                for game in all_games.tolist()
                if not population_omitted_games or game not in population_omitted_games
            ],
            dtype=np.int64,
        ),
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


def _trace_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    games = np.asarray([10, 10, 11, 11, 12, 12, 13, 13], dtype=np.int64)
    decisions = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    validation = np.asarray([12, 13], dtype=np.int64)
    _qualified, receipt = overlay.alignment._qualify_stage_c_game_traces(  # noqa: SLF001
        game_seeds=games,
        decision_indices=decisions,
    )
    return games, decisions, validation, receipt


def test_post_wave_trace_population_uses_only_replayable_games() -> None:
    games, decisions, validation, receipt = _trace_fixture()

    qualified, qualified_validation, replayed = (
        overlay._trace_qualified_game_populations(  # noqa: SLF001
            admission_schema=overlay.post_wave_admission.ADMISSION_SCHEMA,
            game_seeds=games,
            decision_indices=decisions,
            validation_game_seeds=validation,
            expected_qualification=receipt,
        )
    )
    binding = overlay._trace_population_binding(  # noqa: SLF001
        game_seeds=games,
        qualified_game_seeds=qualified,
        qualified_validation_game_seeds=qualified_validation,
        qualification=replayed,
    )

    assert qualified.tolist() == [10, 12]
    assert qualified_validation.tolist() == [12]
    assert replayed == receipt
    assert binding is not None
    assert binding["qualified_games"] == 2
    assert binding["excluded_games"] == 2
    assert binding["excluded_corpus_rows"] == 4
    assert binding["excluded_trace_rows_policy_reanalysis_eligible"] is False
    assert binding["excluded_trace_rows_value_state_evidence_retained"] is True


def test_post_wave_trace_population_replays_pool_game_exclusion() -> None:
    games = np.asarray([10, 10, 11, 11, 12, 12], dtype=np.int64)
    decisions = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    pool_rows = np.asarray([False, False, True, True, False, False], dtype=np.bool_)
    validation = np.asarray([11, 12], dtype=np.int64)
    _qualified, receipt = overlay.alignment._qualify_stage_c_game_traces(  # noqa: SLF001
        game_seeds=games,
        decision_indices=decisions,
        pool_game_rows=pool_rows,
    )

    qualified, qualified_validation, replayed = (
        overlay._trace_qualified_game_populations(  # noqa: SLF001
            admission_schema=overlay.post_wave_admission.ADMISSION_SCHEMA,
            game_seeds=games,
            decision_indices=decisions,
            pool_game_rows=pool_rows,
            validation_game_seeds=validation,
            expected_qualification=receipt,
        )
    )

    assert qualified.tolist() == [10, 12]
    assert qualified_validation.tolist() == [12]
    assert replayed == receipt


@pytest.mark.parametrize("mutation", ["missing", "digest", "corpus"])
def test_post_wave_trace_population_fails_closed_on_drift(
    mutation: str,
) -> None:
    games, decisions, validation, receipt = _trace_fixture()
    expected: object = receipt
    if mutation == "missing":
        expected = None
    elif mutation == "digest":
        expected = {**receipt, "qualified_games": 3}
    else:
        decisions = decisions.copy()
        decisions[0] = 2

    with pytest.raises(
        overlay.OverlayError,
        match="lost game-trace qualification|qualification drifted",
    ):
        overlay._trace_qualified_game_populations(  # noqa: SLF001
            admission_schema=overlay.post_wave_admission.ADMISSION_SCHEMA,
            game_seeds=games,
            decision_indices=decisions,
            validation_game_seeds=validation,
            expected_qualification=expected,
        )


def test_legacy_trace_population_preserves_full_population_only() -> None:
    games, decisions, validation, receipt = _trace_fixture()

    qualified, qualified_validation, replayed = (
        overlay._trace_qualified_game_populations(  # noqa: SLF001
            admission_schema=overlay.LEGACY_ADMISSION_SCHEMA,
            game_seeds=games,
            decision_indices=decisions,
            validation_game_seeds=validation,
            expected_qualification=None,
        )
    )

    assert qualified.tolist() == [10, 11, 12, 13]
    assert qualified_validation.tolist() == [12, 13]
    assert replayed is None
    with pytest.raises(
        overlay.OverlayError,
        match="legacy Stage-C authority unexpectedly carries",
    ):
        overlay._trace_qualified_game_populations(  # noqa: SLF001
            admission_schema=overlay.LEGACY_ADMISSION_SCHEMA,
            game_seeds=games,
            decision_indices=decisions,
            validation_game_seeds=validation,
            expected_qualification=receipt,
        )


def test_trace_qualified_population_is_the_root_breadth_denominator() -> None:
    excluded = {100, 101}
    full_population = _broad_root_inventory(omitted_games=excluded)
    qualified_population = _broad_root_inventory(
        omitted_games=excluded,
        population_omitted_games=excluded,
    )

    assert full_population["passed"] is False
    assert "training:unique_game_fraction" in full_population["failures"]
    assert qualified_population["passed"] is True
    assert qualified_population["scopes"]["training"]["unique_game_fraction"] == 1.0


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
    completed_q_schema = meta["columns"][overlay.COMPLETED_Q_VALUE_COLUMN]
    assert set(completed_q_schema) == {"kind", "dtype", "fill"}
    assert completed_q_schema["kind"] == "ragged2d"
    assert np.dtype(completed_q_schema["dtype"]) == np.dtype(np.float32)
    assert np.isnan(completed_q_schema["fill"])
    assert meta["columns"]["teacher_name"]["categories"] == [
        "historical",
        overlay.POLICY_TEACHER,
    ]


def test_completed_q_column_schema_loads_through_production_memmap(
    tmp_path: Path,
) -> None:
    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": 1,
        "flat_count": 2,
        "legal_width": 2,
        "implicit_zero_columns": [],
        "columns": {},
    }
    overlay._ensure_completed_q_columns(meta)  # noqa: SLF001
    overlay._ensure_completed_q_columns(meta)  # noqa: SLF001
    _write(tmp_path / "row_offsets.dat", np.asarray([0, 2], dtype=np.int64))
    _write(
        tmp_path / f"{overlay.COMPLETED_Q_VALUE_COLUMN}.dat",
        np.asarray([0.25, -0.5], dtype=np.float32),
    )
    _write(
        tmp_path / f"{overlay.COMPLETED_Q_MASK_COLUMN}.dat",
        np.asarray([True, True], dtype=np.bool_),
    )
    (tmp_path / "corpus_meta.json").write_text(
        json.dumps(meta, allow_nan=True), encoding="utf-8"
    )

    corpus = train_bc.MemmapCorpus(tmp_path)

    assert np.asarray(corpus[overlay.COMPLETED_Q_VALUE_COLUMN][0]).tolist() == (
        pytest.approx([0.25, -0.5])
    )
    assert np.asarray(corpus[overlay.COMPLETED_Q_MASK_COLUMN][0]).tolist() == [
        True,
        True,
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


@pytest.mark.parametrize("arm", sorted(overlay.SAMPLING_ARMS))
def test_every_materializer_sampling_arm_is_accepted_by_trainer(arm: str) -> None:
    class _Corpus(dict):
        meta = {
            "stage_c_policy_overlay": {
                "sampling_distribution": {
                    "schema_version": overlay.SAMPLING_SCHEMA,
                    "column": overlay.SAMPLING_COLUMN,
                    "arm": arm,
                }
            }
        }

    data = _Corpus(
        stage_c_policy_sampling_weight=np.asarray(
            [1.0, 0.0, 1.0], dtype=np.float32
        ),
        policy_weight_multiplier=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    )

    weights, source = train_bc._stage_c_policy_aux_base_measure(  # noqa: SLF001
        data, np.asarray([0, 1, 2], dtype=np.int64)
    )

    assert weights.tolist() == [1.0, 0.0, 1.0]
    assert source == f"stage_c_{arm.lower()}"


def test_rare_action_balanced_sampling_raises_observed_teacher_mass() -> None:
    rows = 100
    subset = {
        "row_index": np.arange(rows, dtype=np.int64),
        "stratum": np.asarray(["low_inclusion"] * 50 + ["high_inclusion"] * 50),
        "phase": np.asarray(["PLAY_TURN"] * rows),
        "legal_width": np.full(rows, 2, dtype=np.int64),
    }
    # Action ids 0/1 are common BUILD_ROAD choices. Action ids 2..5 are the
    # four rare strategic development-card plays, one teacher argmax each.
    action_types = (
        "BUILD_ROAD",
        "BUILD_ROAD",
        *overlay.RARE_STRATEGIC_ACTION_TYPES,
    )
    legal_ids: list[int] = []
    target: list[float] = []
    offsets = [0]
    for row in range(rows):
        rare_id = row + 2 if row < len(overlay.RARE_STRATEGIC_ACTION_TYPES) else 1
        legal_ids.extend((0, rare_id))
        target.extend((0.1, 0.9))
        offsets.append(len(legal_ids))
    patch = {
        "row_index": subset["row_index"],
        "legal_action_offsets": np.asarray(offsets, dtype=np.int64),
        "legal_action_ids_flat": np.asarray(legal_ids, dtype=np.int64),
        "target_policy_flat": np.asarray(target, dtype=np.float32),
        "target_policy_mask_flat": np.ones(len(target), dtype=np.bool_),
    }
    export = {
        "sampling_population": {
            "candidate_counts_by_stratum": {
                "low_inclusion": 100,
                "high_inclusion": 500,
            },
            "selected_counts_by_stratum": {
                "low_inclusion": 50,
                "high_inclusion": 50,
            },
        }
    }
    validation = np.zeros(rows, dtype=np.bool_)
    validation[-10:] = True

    weights, report = overlay._selected_sampling_weights(  # noqa: SLF001
        export=export,
        subset=subset,
        patch=patch,
        selected_validation=validation,
        arm="RARE_ACTION_BALANCED",
        production_weight_cap=4.0,
        action_types_by_id=action_types,
    )

    assert np.mean(weights[~validation]) == pytest.approx(1.0)
    assert np.max(weights[~validation]) <= 4.0
    # Rare emphasis composes on the production inverse-inclusion measure; it
    # must not silently turn the rest of the arm into uniform strategic mass.
    assert weights[60] > weights[10]
    for row, action_type in enumerate(overlay.RARE_STRATEGIC_ACTION_TYPES):
        assert weights[row] > weights[10]
        balance = report["rare_action_balance"]["training"]
        assert (
            balance["row_counts"][action_type] == 1
        )
        assert (
            report["training_mass_by_teacher_argmax_action_type"][action_type]
            > balance["base_weight_mass"][action_type]
        )
    assert report["rare_action_balance"]["training"]["composition"] == (
        "production_inverse_inclusion_weight_times_rare_action_multiplier"
    )
    assert report["rare_action_balance"]["training"][
        "missing_types_are_not_synthesized"
    ]


def test_wide_choice_balanced_sampling_raises_width_20_mass() -> None:
    rows = 100
    subset = {
        "row_index": np.arange(rows, dtype=np.int64),
        "stratum": np.asarray(["ordinary"] * rows),
        "phase": np.asarray(
            ["BUILD_INITIAL_SETTLEMENT"] + ["PLAY_TURN"] * (rows - 1)
        ),
        "legal_width": np.asarray(
            [54] + [4] * 89 + [24] + [4] * 7 + [24] * 2,
            dtype=np.int64,
        ),
    }
    patch = {"row_index": subset["row_index"]}
    export = {
        "sampling_population": {
            "candidate_counts_by_stratum": {"ordinary": rows},
            "selected_counts_by_stratum": {"ordinary": rows},
        }
    }
    validation = np.zeros(rows, dtype=np.bool_)
    validation[-2:] = True

    weights, report = overlay._selected_sampling_weights(  # noqa: SLF001
        export=export,
        subset=subset,
        patch=patch,
        selected_validation=validation,
        arm="WIDE_CHOICE_BALANCED",
        production_weight_cap=4.0,
    )

    assert np.mean(weights[~validation]) == pytest.approx(1.0)
    assert np.max(weights[~validation]) <= 4.0
    assert weights[90] > weights[0]
    assert weights[0] == pytest.approx(weights[1])
    balance = report["wide_choice_balance"]["training"]
    assert balance["row_count"] == 1
    assert balance["phase"] == "PLAY_TURN"
    assert balance["realized_mass_fraction"] == pytest.approx(0.03)
    assert balance["requested_multiplier"] > 1.0
    assert balance["composition"] == (
        "production_inverse_inclusion_weight_times_wide_choice_multiplier"
    )


def test_teacher_action_type_projection_rejects_unbound_catalog_ids() -> None:
    patch = {
        "row_index": np.asarray([4], dtype=np.int64),
        "legal_action_offsets": np.asarray([0, 2], dtype=np.int64),
        "legal_action_ids_flat": np.asarray([0, 9], dtype=np.int64),
        "target_policy_flat": np.asarray([0.1, 0.9], dtype=np.float32),
        "target_policy_mask_flat": np.asarray([True, True]),
    }
    with pytest.raises(overlay.OverlayError, match="projection is malformed"):
        overlay._teacher_argmax_action_types(  # noqa: SLF001
            patch,
            action_types_by_id=("BUILD_ROAD",),
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
