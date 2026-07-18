from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_stage_c_learner_overlay as overlay


def _post_wave_admission() -> dict:
    return {
        "schema_version": overlay.post_wave_admission.ADMISSION_SCHEMA,
        "admission_sha256": "sha256:" + "a" * 64,
        "corpus": {
            "stored_policy_target_distillation_eligible": False,
            "state_reanalysis_eligible": True,
        },
        "policy_target_policy": {
            "stored_targets_are_historical_operator_only": True,
            "current_teacher_requires_reanalysis": True,
            "legacy_pimc_rows_allowed": False,
        },
    }


def _augmented_source_semantics(admission: dict) -> dict:
    semantics = overlay._source_policy_semantics(admission)  # noqa: SLF001
    semantics.pop("semantics_sha256")
    semantics.update(
        {
            "target_identity_matches_stored_policy": False,
            "stored_policy_active_rows": 11,
            "stored_policy_eligible_rows": 0,
            "stored_policy_quarantined_rows": 11,
            "derived_overlay_historical_policy_targets_active": False,
        }
    )
    semantics["semantics_sha256"] = overlay._value_sha256(semantics)  # noqa: SLF001
    return semantics


def test_post_wave_source_semantics_require_explicit_quarantine() -> None:
    admission = _post_wave_admission()

    semantics = overlay._source_policy_semantics(admission)  # noqa: SLF001

    assert semantics["stored_policy_target_distillation_eligible"] is False
    assert semantics["current_teacher_requires_reanalysis"] is True
    assert semantics["legacy_pimc_rows_allowed"] is False
    admission["corpus"]["stored_policy_target_distillation_eligible"] = True
    with pytest.raises(overlay.OverlayError, match="policy quarantine drifted"):
        overlay._source_policy_semantics(admission)  # noqa: SLF001


def test_base_admission_dispatch_keeps_schema_verifiers_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "admission.json"
    admission = _post_wave_admission()
    path.write_text(json.dumps(admission), encoding="utf-8")
    calls: list[str] = []

    def verify_post_wave(candidate: Path) -> tuple[Path, dict]:
        calls.append("post-wave")
        return candidate.resolve(), admission

    def reject_legacy(_candidate: Path) -> tuple[Path, dict]:
        raise AssertionError("legacy verifier must not see a post-wave admission")

    monkeypatch.setattr(
        overlay.post_wave_admission, "verify_admission", verify_post_wave
    )
    from tools import a1_b200_active_policy_campaign as active_campaign

    monkeypatch.setattr(
        active_campaign, "_load_admission", reject_legacy
    )

    _resolved, loaded, semantics = overlay._load_base_admission(path)  # noqa: SLF001

    assert calls == ["post-wave"]
    assert loaded is admission
    assert semantics["source_admission_schema"] == admission["schema_version"]


def test_plan_source_admission_binds_row_level_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admission_path = tmp_path / "admission.json"
    admission_path.write_text("{}", encoding="utf-8")
    admission = _post_wave_admission()
    base_semantics = overlay._source_policy_semantics(admission)  # noqa: SLF001
    eligibility = {
        "overlay_sha256": "sha256:" + "b" * 64,
        "policy_quarantine_changes_value_eligibility": False,
        "policy_quarantine_changes_state_reanalysis_eligibility": False,
        "counts": {
            "stored_policy_active_rows": 11,
            "stored_policy_eligible_rows": 0,
            "stored_policy_quarantined_rows": 11,
        },
    }
    eligibility_path = tmp_path / "eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility), encoding="utf-8")

    monkeypatch.setattr(
        overlay,
        "_load_base_admission",
        lambda _path: (admission_path.resolve(), admission, base_semantics),
    )
    plan = {
        "source_corpus_admission": {
            "path": str(admission_path),
            "file_sha256": overlay._file_sha256(admission_path),  # noqa: SLF001
            "admission_sha256": admission["admission_sha256"],
        },
        "eligibility_overlay": {
            "path": str(eligibility_path),
            "file_sha256": overlay._file_sha256(eligibility_path),  # noqa: SLF001
            "overlay_sha256": eligibility["overlay_sha256"],
        },
        "target_identity_matches_stored_policy": False,
    }

    _path, _admission, semantics = overlay._load_plan_source_admission(  # noqa: SLF001
        plan
    )

    assert semantics["stored_policy_eligible_rows"] == 0
    assert semantics["stored_policy_quarantined_rows"] == 11
    assert semantics["derived_overlay_historical_policy_targets_active"] is False
    plan["target_identity_matches_stored_policy"] = True
    with pytest.raises(overlay.OverlayError, match="quarantine counts drifted"):
        overlay._load_plan_source_admission(plan)  # noqa: SLF001


def test_post_wave_derived_contract_authorizes_only_stage_c_rows() -> None:
    admission = _post_wave_admission()
    semantics = _augmented_source_semantics(admission)
    target = "sha256:" + "c" * 64

    contract = overlay._derived_policy_distillation_contract(  # noqa: SLF001
        base_admission=admission,
        source_policy_semantics=semantics,
        selected_rows=65_536,
        root_breadth_inventory_sha256="sha256:" + "d" * 64,
        target_policy_target_identity_sha256=target,
    )

    assert contract["policy_active_rows"] == 65_536
    assert contract["stage_c_reanalysis_only"] is True
    assert contract["historical_policy_targets_active"] is False
    assert contract["source_stored_policy_target_distillation_eligible"] is False
    assert contract["source_stored_policy_quarantined_rows"] == 11
    assert contract["target_policy_target_identity_sha256"] == target


def test_overlay_verifier_accepts_bound_post_wave_derived_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.admission.json"
    source_path.write_text("{}", encoding="utf-8")
    source = _post_wave_admission()
    base_semantics = overlay._source_policy_semantics(source)  # noqa: SLF001
    semantics = _augmented_source_semantics(source)
    target = "sha256:" + "e" * 64
    target_reanalyzer = {
        "path": "/sealed/v15-target-reanalyzer.pt",
        "sha256": "sha256:" + "d" * 64,
    }
    completed_q = {
        "schema_version": overlay.COMPLETED_Q_BINDING_SCHEMA,
        "semantics": {"default_learner_objective": "none_evidence_only"},
        "operator_identity": {"legacy_or_unbound_q_allowed": False},
    }
    root_breadth = {"inventory_sha256": "sha256:" + "f" * 64}
    game_seeds = np.arange(100, 116, dtype=np.int64)
    decision_indices = np.zeros(16, dtype=np.int64)
    validation_game_seeds = np.arange(112, 116, dtype=np.int64)
    qualified_games, trace_qualification = (
        overlay.alignment._qualify_stage_c_game_traces(  # noqa: SLF001
            game_seeds=game_seeds,
            decision_indices=decision_indices,
        )
    )
    trace_population = overlay._trace_population_binding(  # noqa: SLF001
        game_seeds=game_seeds,
        qualified_game_seeds=qualified_games,
        qualified_validation_game_seeds=validation_game_seeds,
        qualification=trace_qualification,
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    meta = {
        "payload_inventory_sha256": "sha256:" + "1" * 64,
        "stage_c_policy_overlay": {
            "root_breadth": root_breadth,
            "paired_root_value_patch_consumed": True,
            "completed_q_patch_consumed": True,
            "completed_q_binding": completed_q,
            "target_reanalyzer_checkpoint": target_reanalyzer,
            "source_game_trace_qualification": trace_qualification,
            "trace_population": trace_population,
        },
        "columns": dict(overlay.COMPLETED_Q_COLUMN_SCHEMAS),
    }
    meta_path = corpus / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    receipt = {
        "schema_version": overlay.MATERIALIZATION_SCHEMA,
        "target_policy_target_identity_sha256": target,
        "target_reanalyzer_checkpoint": target_reanalyzer,
        "root_breadth": root_breadth,
        "paired_root_value_patch_consumed": True,
        "completed_q_patch_consumed": True,
        "completed_q_binding": completed_q,
        "source_game_trace_qualification": trace_qualification,
        "trace_population": trace_population,
        "overlay_corpus": {
            "payload_inventory_sha256": meta["payload_inventory_sha256"]
        },
    }
    receipt["receipt_sha256"] = overlay._value_sha256(receipt)  # noqa: SLF001
    receipt_path = corpus / "stage_c_policy_overlay.receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    source_ref = {
        "path": str(source_path),
        "file_sha256": overlay._file_sha256(source_path),  # noqa: SLF001
        "admission_sha256": source["admission_sha256"],
        "schema_version": source["schema_version"],
    }
    admission = {
        "schema_version": overlay.post_wave_admission.ADMISSION_SCHEMA,
        "status": "admitted_for_diagnostic_policy_distillation",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "corpus": {
            "data_path": str(corpus),
            "corpus_meta_file_sha256": overlay._file_sha256(meta_path),  # noqa: SLF001
            "payload_inventory_sha256": meta["payload_inventory_sha256"],
            "stored_policy_target_distillation_eligible": True,
            "incompatible_policy_active_rows": 0,
            "validation_manifest": {"path": "/sealed/validation.json"},
        },
        "policy_distillation_contract": {
            "coherent_public_n128_only": True,
            "stage_c_reanalysis_only": True,
            "historical_policy_targets_active": False,
            "legacy_pimc_rows_allowed": False,
            "policy_active_rows": 16,
            "source_admission_schema": source["schema_version"],
            "root_breadth_inventory_sha256": root_breadth["inventory_sha256"],
            "target_policy_target_identity_sha256": target,
        },
        "source_policy_target_policy": source["policy_target_policy"],
        "policy_target_policy": {
            "stored_targets_are_current_stage_c_operator_only": True,
            "historical_policy_targets_active": False,
            "legacy_pimc_rows_allowed": False,
            "target_policy_target_identity_sha256": target,
        },
        "stage_c_policy_overlay": {
            "schema_version": overlay.ADMISSION_OVERLAY_SCHEMA,
            "historical_policy_targets_active": False,
            "base_value_rows_retained": True,
            "paired_root_value_patch_consumed": True,
            "completed_q_patch_consumed": True,
            "completed_q_binding": completed_q,
            "selected_policy_rows": 16,
            "selected_training_policy_rows": 12,
            "selected_validation_policy_rows": 4,
            "sampling_distribution": {
                "schema_version": overlay.SAMPLING_SCHEMA,
                "arm": "STRATEGIC_BALANCED",
            },
            "target_policy_target_identity_sha256": target,
            "target_reanalyzer_checkpoint": target_reanalyzer,
            "root_breadth": root_breadth,
            "materialization_receipt": {
                "path": str(receipt_path),
                "file_sha256": overlay._file_sha256(receipt_path),  # noqa: SLF001
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "source_admission": source_ref,
            "source_policy_semantics": semantics,
            "source_game_trace_qualification": trace_qualification,
            "trace_population": trace_population,
        },
    }
    admission["admission_sha256"] = overlay._value_sha256(admission)  # noqa: SLF001
    admission_path = corpus / "overlay.admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    monkeypatch.setattr(
        overlay,
        "_load_base_admission",
        lambda _path: (source_path.resolve(), source, base_semantics),
    )
    monkeypatch.setattr(
        overlay,
        "_verify_stage_c_root_breadth_inventory",
        lambda value, *, selected_rows: value,
    )
    monkeypatch.setattr(
        overlay,
        "_stage_c_root_breadth_inventory",
        lambda **_kwargs: root_breadth,
    )
    monkeypatch.setattr(
        overlay.train_bc,
        "_load_validation_game_seed_manifest_for_training",
        lambda *_args, **_kwargs: {"game_seeds": validation_game_seeds},
    )
    monkeypatch.setattr(
        overlay.train_bc,
        "MemmapCorpus",
        lambda _path: {
            "game_seed": game_seeds,
            "decision_index": decision_indices,
            "policy_weight_multiplier": np.ones(16, dtype=np.float32),
            "phase": np.asarray(["PLAY_TURN"] * 16),
        },
    )

    verified = overlay.verify_overlay_admission(admission_path)

    assert verified["admission"]["schema_version"] == source["schema_version"]
    admission["policy_target_policy"]["historical_policy_targets_active"] = True
    admission["admission_sha256"] = overlay._value_sha256(  # noqa: SLF001
        {key: value for key, value in admission.items() if key != "admission_sha256"}
    )
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    with pytest.raises(overlay.OverlayError, match="policy authority drifted"):
        overlay.verify_overlay_admission(admission_path)
