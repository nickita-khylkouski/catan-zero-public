from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_post_wave_stage_c_admission as admission
from tools import a1_stage_c_teacher_alignment as alignment


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "producer.pt"
    checkpoint.write_bytes(b"historical producer")
    checkpoint_sha = admission._file_sha256(checkpoint)  # noqa: SLF001
    contract_sha = "sha256:" + "1" * 64
    contract = _write(
        tmp_path / "contract.json",
        {"schema_version": "test", "contract_sha256": contract_sha},
    )
    validation = {
        "schema_version": "train-validation-game-seeds-v1",
        "a1_contract_sha256": contract_sha,
        "game_seeds": [2],
        "validation_game_seed_count": 1,
        "validation_row_count": 3,
        "validation_game_seed_set_sha256": "sha256:" + "2" * 64,
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "validation_game_seed_ranges": [],
    }
    validation_path = _write(tmp_path / "validation.json", validation)
    records = [
        {"game_seed": 1, "split": "train"},
        {"game_seed": 2, "split": "validation"},
    ]
    selected = {
        "schema_version": "a1-selected-training-games-v1",
        "a1_contract_sha256": contract_sha,
        "records": records,
        "records_sha256": admission._value_sha256(records),  # noqa: SLF001
        "selected_game_count": 2,
        "training_game_count": 1,
        "validation_game_count": 1,
        "selected_game_seed_set_sha256": "sha256:" + "3" * 64,
    }
    selected_path = _write(tmp_path / "selected.json", selected)
    provenance_row = {
        "producer_checkpoint_sha256": checkpoint_sha,
        "search_operator_sha256": "sha256:" + "4" * 64,
        "effective_search_config_sha256": "sha256:" + "5" * 64,
        "evaluator_sha256": "sha256:" + "6" * 64,
        "entity_feature_adapter_version": (
            "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
        ),
        "event_history_semantic": "meaningful_public_history_2p_no_trade_v2",
    }
    audit = {
        "schema_version": "a1-post-wave-audit-v3",
        "passed": True,
        "errors": [],
        "contract_path": str(contract),
        "contract_sha256": contract_sha,
        "total_unique_games": 2,
        "rows": 10,
        "target_information_regime": admission.COHERENT_REGIME,
        "target_activation": {"passed": True},
        "selected_training_games": {
            "manifest_file_sha256": admission._file_sha256(selected_path),  # noqa: SLF001
            "manifest_sha256": admission._value_sha256(selected),  # noqa: SLF001
        },
        "validation_holdout": {
            "manifest_file_sha256": admission._file_sha256(validation_path),  # noqa: SLF001
            "manifest_sha256": admission._value_sha256(validation),  # noqa: SLF001
        },
        "source_provenance": {
            "current_producer": provenance_row,
            "hard_negative": provenance_row,
        },
    }
    audit["audit_sha256"] = admission._value_sha256(audit)  # noqa: SLF001
    audit_path = _write(tmp_path / "audit.json", audit)
    fields = {
        name: None
        for name in (
            *alignment.SEARCH_FIELDS,
            *alignment.BELIEF_FIELDS,
            *alignment.CHANCE_FIELDS,
            *alignment.SYMMETRY_FIELDS,
            *alignment.TARGET_SEMANTIC_FIELDS,
        )
    }
    fields.update({
        "n_full": 128,
        "n_fast": 16,
        "p_full": 0.25,
        "c_scale": 0.1,
        "c_visit": 50.0,
        "prior_temperature": 1.0,
        "sigma_eval": 0.79,
        "max_depth": 80,
        "value_scale": 1.0,
        "value_squash": "tanh",
        "value_readout": "scalar",
        "rust_featurize": True,
        "coherent_public_belief_search": True,
        "boundary_value_particles": 1,
        "information_set_search": False,
        "determinization_particles": 1,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "symmetry_averaged_eval": True,
        "public_observation": True,
        "meaningful_public_history": True,
        "record_automatic_transitions": True,
        "preserve_search_evidence": True,
        "learner_entity_feature_adapter_version": provenance_row[
            "entity_feature_adapter_version"
        ],
    })
    manifest_path = _write(
        tmp_path / "manifest.json",
        {
            "checkpoint": str(checkpoint),
            "producer_checkpoint_sha256": checkpoint_sha,
            "target_information_regime": admission.COHERENT_REGIME,
            "search_evidence_schema": admission.SEARCH_EVIDENCE_SCHEMA,
            "full_config_hash": "sha256:test",
            "cli_args": fields,
        },
    )
    column_names = set(admission.REQUIRED_COLUMNS)
    column_names.add("target_information_regime")
    meta = {
        "row_count": 10,
        "payload_inventory_sha256": "sha256:" + "7" * 64,
        "source_shard_inventory": [{"path": "shard.npz"}],
        "search_evidence": {
            "schema": admission.SEARCH_EVIDENCE_SCHEMA,
            "active_row_count": 4,
        },
        "columns": {
            name: (
                {"categories": [admission.COHERENT_REGIME]}
                if name == "target_information_regime"
                else {}
            )
            for name in column_names
        },
        "a1_post_wave_audit": {
            "path": str(audit_path),
            "file_sha256": admission._file_sha256(audit_path),  # noqa: SLF001
            "audit_sha256": audit["audit_sha256"],
            "validation_holdout": {
                "path": str(validation_path),
                "file_sha256": admission._file_sha256(validation_path),  # noqa: SLF001
                "validation_game_seed_count": 1,
                "validation_game_seed_set_sha256": validation[
                    "validation_game_seed_set_sha256"
                ],
            },
        },
        "selected_game_seed_manifest": {
            "path": str(selected_path),
            "file_sha256": admission._file_sha256(selected_path),  # noqa: SLF001
            "selected_game_count": 2,
        },
    }
    meta_path = _write(tmp_path / "corpus" / "corpus_meta.json", meta)
    return {
        "meta": meta_path,
        "manifest": manifest_path,
        "checkpoint": checkpoint,
    }


def test_post_wave_admission_reuses_existing_rows_and_replays_source_identity(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    binding = tmp_path / "source.binding.json"
    output = tmp_path / "admission.json"
    built = admission.build(
        corpus_meta=fixture["meta"],
        representative_manifest=fixture["manifest"],
        source_binding_write=binding,
        admission_write=output,
    )
    _path, verified = admission.verify_admission(output)
    assert verified == built
    assert verified["corpus"]["state_reanalysis_eligible"] is True
    assert verified["corpus"]["stored_policy_target_distillation_eligible"] is False

    identity = alignment._operator_identity(  # noqa: SLF001
        binding,
        fixture["checkpoint"],
    )
    assert identity["search"]["n_full"] == 128
    assert identity["target_information_regime"] == admission.COHERENT_REGIME


def test_post_wave_admission_rejects_audit_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    audit_path = Path(
        json.loads(fixture["meta"].read_text())["a1_post_wave_audit"]["path"]
    )
    audit = json.loads(audit_path.read_text())
    audit["passed"] = False
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(admission.AdmissionError, match="binding drifted"):
        admission.build(
            corpus_meta=fixture["meta"],
            representative_manifest=fixture["manifest"],
            source_binding_write=tmp_path / "binding.json",
            admission_write=tmp_path / "admission.json",
        )
