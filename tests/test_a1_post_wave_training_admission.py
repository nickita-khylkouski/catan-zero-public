from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_b200_stage_c_learner_campaign as campaign
from tools import a1_one_dose_train as executor


def _legacy_population(*, games: int = 8_192) -> dict:
    return {
        "schema_version": "a1-coherent-n128-corpus-admission-v1",
        "corpus": {"selected_games": games},
        "policy_distillation_contract": {"policy_active_rows": 8_192},
    }


def _post_wave_population(*, games: int = 12_000, roots: int = 65_536) -> dict:
    return {
        "schema_version": executor.stage_c_final.overlay.post_wave_admission.ADMISSION_SCHEMA,
        "corpus": {"selected_games": games},
        "policy_distillation_contract": {"policy_active_rows": roots},
        "stage_c_policy_overlay": {"selected_policy_rows": roots},
    }


def test_population_authority_preserves_legacy_8192_exactly() -> None:
    result = executor._coherent_admission_population_authority(  # noqa: SLF001
        _legacy_population(),
        overlay_evidence=None,
    )
    assert result == {
        "selected_games": 8_192,
        "selected_policy_roots": 8_192,
    }
    with pytest.raises(executor.ExecutorError, match="legacy coherent corpus"):
        executor._coherent_admission_population_authority(  # noqa: SLF001
            _legacy_population(games=12_000),
            overlay_evidence=None,
        )


def test_population_authority_uses_authenticated_post_wave_counts() -> None:
    admission = _post_wave_population()
    evidence = {
        "admission": admission,
        "receipt": {"projection": {"selected_rows": 65_536}},
    }

    result = executor._coherent_admission_population_authority(  # noqa: SLF001
        admission,
        overlay_evidence=evidence,
    )

    assert result == {
        "selected_games": 12_000,
        "selected_policy_roots": 65_536,
    }
    evidence["receipt"]["projection"]["selected_rows"] = 65_535
    with pytest.raises(executor.ExecutorError, match="game/root authority drifted"):
        executor._coherent_admission_population_authority(  # noqa: SLF001
            admission,
            overlay_evidence=evidence,
        )


def test_stage_c_campaign_uses_overlay_verifier_as_schema_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "overlay.admission.json"
    path.write_text("{}", encoding="utf-8")
    admission = _post_wave_population()
    evidence = {
        "path": str(path),
        "admission": admission,
        "receipt": {},
    }
    monkeypatch.setattr(
        campaign.overlay,
        "verify_overlay_admission",
        lambda _path: evidence,
    )

    loaded_evidence, loaded_path, loaded_admission = (
        campaign._load_overlay_admission(path)  # noqa: SLF001
    )

    assert loaded_evidence is evidence
    assert loaded_path == path.resolve()
    assert loaded_admission is admission


def test_one_dose_accepts_post_wave_overlay_and_derives_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "corpus"
    data_path.mkdir()
    meta_path = data_path / "corpus_meta.json"
    meta = {"payload_inventory_sha256": "sha256:" + "1" * 64}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    validation_path = tmp_path / "validation.json"
    validation_path.write_text("{}", encoding="utf-8")
    producer_sha = "sha256:" + "2" * 64
    selected = np.asarray([10, 11, 12], dtype=np.int64)
    selected_sha = executor.train_bc._game_seed_set_sha256(selected)  # noqa: SLF001
    target_contract = "sha256:" + "3" * 64
    validation_manifest_sha = "sha256:" + "4" * 64
    target_identity = "sha256:" + "5" * 64
    admission = {
        "schema_version": executor.stage_c_final.overlay.post_wave_admission.ADMISSION_SCHEMA,
        "status": "admitted_for_diagnostic_policy_distillation",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "contract": {"contract_sha256": "sha256:" + "6" * 64},
        "post_wave_evidence": {
            "validation_manifest_sha256": validation_manifest_sha
        },
        "corpus": {
            "data_path": str(data_path),
            "corpus_meta_path": str(meta_path),
            "corpus_meta_file_sha256": executor._file_sha256(meta_path),  # noqa: SLF001
            "payload_inventory_sha256": meta["payload_inventory_sha256"],
            "validation_manifest": {
                "path": str(validation_path),
                "file_sha256": executor._file_sha256(validation_path),  # noqa: SLF001
            },
            "producer_checkpoint_sha256": producer_sha,
            "selected_games": 3,
            "selected_game_seed_set_sha256": selected_sha,
            "target_information_regime": "public_belief_single_tree_v1",
            "search_evidence_schema": executor.SEARCH_EVIDENCE_V2_SCHEMA,
            "search_evidence_storage": "training_memmap",
            "incompatible_policy_active_rows": 0,
        },
        "policy_distillation_contract": {
            "coherent_public_n128_only": True,
            "legacy_pimc_rows_allowed": False,
            "policy_active_rows": 2,
            "stage_c_reanalysis_only": True,
            "target_policy_target_identity_sha256": target_identity,
        },
        "stage_c_policy_overlay": {
            "selected_policy_rows": 2,
            "target_policy_target_identity_sha256": target_identity,
        },
    }
    admission["admission_sha256"] = executor._value_sha256(admission)  # noqa: SLF001
    admission_path = tmp_path / "overlay.admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    observed = np.asarray([10, 10, 11, 11, 12, 12], dtype=np.int64)
    policy_weight = np.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    validation = {
        "path": str(validation_path),
        "file_sha256": executor._file_sha256(validation_path),  # noqa: SLF001
        "manifest_sha256": validation_manifest_sha,
        "a1_contract_sha256": target_contract,
        "game_seeds": np.asarray([12], dtype=np.int64),
        "validation_game_seed_count": 1,
        "validation_game_seed_set_sha256": (
            executor.train_bc._game_seed_set_sha256(  # noqa: SLF001
                np.asarray([12], dtype=np.int64)
            )
        ),
        "validation_row_count": 2,
    }
    overlay_evidence = {
        "path": str(admission_path),
        "admission": admission,
        "receipt": {"projection": {"selected_rows": 2}},
    }
    monkeypatch.setattr(
        executor.stage_c_final.overlay,
        "verify_overlay_admission",
        lambda _path: overlay_evidence,
    )
    monkeypatch.setattr(
        executor.train_bc,
        "_validate_memmap_payload_inventory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        executor.train_bc,
        "_load_validation_game_seed_manifest_for_training",
        lambda *_args, **_kwargs: validation,
    )
    monkeypatch.setattr(
        executor.train_bc,
        "load_teacher_data_memmap",
        lambda *_args, **_kwargs: {
            "game_seed": observed,
            "policy_weight_multiplier": policy_weight,
        },
    )
    monkeypatch.setattr(
        executor.train_bc,
        "_training_data_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "7" * 64,
    )
    monkeypatch.setattr(
        executor,
        "_verify_coherent_search_evidence_memmap",
        lambda **_kwargs: None,
    )

    result = executor._verify_coherent_direct_training_inputs(  # noqa: SLF001
        admission_path=admission_path,
        lock={
            "contract_sha256": "sha256:" + "8" * 64,
            "checkpoints": [{"role": "producer", "sha256": producer_sha}],
        },
        lock_path=lock_path,
        lock_verifier_authority=None,
        reviewed_lock_file_sha256=None,
        recipe={},
        objective={},
        data_path=data_path,
        validation_path=validation_path,
    )

    binding = result["coherent_direct_corpus_binding"]
    assert binding["corpus"]["selected_game_count"] == 3
    assert binding["corpus"]["selected_policy_root_count"] == 2
    assert binding["corpus"]["seed_start"] == 10
    assert binding["corpus"]["seed_end"] == 13
    assert binding["target_contract_sha256"] == target_contract
