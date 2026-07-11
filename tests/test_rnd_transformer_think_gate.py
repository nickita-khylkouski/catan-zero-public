from __future__ import annotations

import hashlib
import json

import pytest

from tools import rnd_e3_holdout_export as engine
from tools import rnd_transformer_think_holdout_export as exporter
from tools import rnd_transformer_think_learning_gate as gate


def _canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _experiment():
    payload = {
        "schema_version": "catan-zero-transformer-think-a1-screen/v1",
        "status": "registered_ready",
        "common": {
            "hidden_size": 640,
            "state_layers": 6,
            "attention_heads": 8,
            "state_trunk": "transformer",
            "latent_deliberation_slots": 8,
            "identity_initialization_required": True,
            "frozen_incumbent_checkpoint_sha256": (
                exporter.FROZEN_INCUMBENT_CHECKPOINT_SHA256
            ),
            "information_regime": "public_sampled_belief",
        },
        "arms": [
            {
                "arm_id": arm,
                "latent_deliberation_steps": steps,
                "expected_parameters": parameters,
                "capacity_class": capacity,
            }
            for arm, (steps, parameters, capacity) in exporter.ARMS.items()
        ],
        "run_matrix": {
            "seeds": list(exporter.SEEDS),
            "required_run_count": exporter.EXPECTED_RUNS,
        },
    }
    payload["config_sha256"] = _canonical(payload)
    return payload


def test_exporter_registration_freezes_exact_four_arm_twelve_run_design():
    exporter._validate_registered_experiment(_experiment(), registered=True)
    drifted = _experiment()
    drifted["arms"][2]["expected_parameters"] += 1
    semantic = dict(drifted)
    semantic.pop("config_sha256")
    drifted["config_sha256"] = _canonical(semantic)
    with pytest.raises(ValueError, match="architecture drift"):
        exporter._validate_registered_experiment(drifted, registered=True)


def test_exporter_specialization_is_process_local_and_restored():
    original = {
        "ARMS": engine.ARMS,
        "schema": engine.EVIDENCE_SCHEMA,
        "validator": engine._validate_contract,
    }
    with exporter._specialized_engine():
        assert engine.ARMS is exporter.ARMS
        assert engine.EVIDENCE_SCHEMA == exporter.EVIDENCE_SCHEMA
        assert engine._validate_contract is exporter._validate_registered_experiment
    assert engine.ARMS is original["ARMS"]
    assert engine.EVIDENCE_SCHEMA == original["schema"]
    assert engine._validate_contract is original["validator"]


def _resolved(arm):
    return {
        "pipeline": "train",
        "fields": {
            "hidden_size": 640,
            "graph_layers": 6,
            "attention_heads": 8,
            "entity_state_trunk": "transformer",
            "latent_deliberation_steps": gate.EXPECTED_STEPS[arm],
            "latent_deliberation_slots": 8,
            "max_steps": 250,
            "batch_size": 1024,
            "grad_accum_steps": 4,
            "mask_hidden_info": True,
        },
    }


def _records(experiment_sha, registration, contract, *, forced_outlier=False):
    losses = {
        "transformer-k0": 1.02,
        "think-transformer-k1": 1.00,
        "think-transformer-k2": 0.96,
        "think-transformer-k4": 0.97,
    }
    for arm in gate.ARMS:
        for seed in gate.SEEDS:
            resolved = _resolved(arm)
            provenance = {
                "schema_version": "catan-zero-transformer-think-run-provenance/v1",
                "initial_checkpoint_sha256": registration[
                    "initial_checkpoint_sha256_by_arm_seed"
                ][f"{arm}@{seed}"],
                "resolved_train_config": resolved,
                "resolved_train_config_sha256": _canonical(resolved),
                "parameter_count": gate.EXPECTED_PARAMETERS[arm],
                "optimizer_steps": 250,
                "global_batch_size": 4096,
                "sample_presentations": 1_024_000,
                "checkpoint_sha256": hashlib.sha256(f"{arm}-{seed}".encode()).hexdigest(),
                "evidence_export_contract_sha256": contract["evidence_file_sha"],
                "evidence_export_contract_semantic_sha256": contract["evidence_semantic_sha"],
                "exporter_source_sha256": contract["exporter_source_sha256"],
                "exporter_engine_source_sha256": contract["exporter_engine_source_sha256"],
                "exporter_helper_source_sha256": contract["exporter_helper_source_sha256"],
            }
            for game in ("game:a", "game:b"):
                for decision in ("d:0", "d:1"):
                    forced = forced_outlier and decision == "d:1"
                    loss = losses[arm]
                    if forced and arm in gate.PRIMARY:
                        loss = 100.0
                    yield {
                        "schema_version": gate.EVIDENCE_SCHEMA,
                        "arm_id": arm,
                        "training_seed": seed,
                        "game_id": game,
                        "decision_id": f"{game}:{decision}",
                        "forced": forced,
                        "soft_target_policy_ce": loss,
                        "public_masked": True,
                        "evaluation_split": "holdout",
                        "is_training_game": False,
                        "experiment_config_sha256": experiment_sha,
                        "corpus_fingerprint": registration["corpus_fingerprint"],
                        "training_manifest_sha256": registration["training_manifest_sha256"],
                        "validation_manifest_sha256": registration["validation_manifest_sha256"],
                        "run_provenance": provenance,
                    }


def test_gate_scores_only_k2_k4_against_capacity_matched_k1(monkeypatch):
    digest = "a" * 64
    registration = {
        "corpus_fingerprint": "b" * 64,
        "training_manifest_sha256": "c" * 64,
        "validation_manifest_sha256": "d" * 64,
        "initial_checkpoint_sha256_by_arm_seed": {
            f"{arm}@{seed}": hashlib.sha256(f"init-{arm}-{seed}".encode()).hexdigest()
            for arm in gate.ARMS
            for seed in gate.SEEDS
        },
    }
    contract = {
        "arms": {},
        "registration": registration,
        "gate_sha": "e" * 64,
        "evidence_file_sha": "f" * 64,
        "evidence_semantic_sha": "1" * 64,
        "exporter_source_sha256": "2" * 64,
        "exporter_engine_source_sha256": "3" * 64,
        "exporter_helper_source_sha256": "4" * 64,
    }
    monkeypatch.setattr(gate, "FROZEN_DECISIONS_PER_RUN", 4)
    monkeypatch.setattr(gate, "FROZEN_HOLDOUT_GAMES", 2)
    monkeypatch.setattr(gate, "_validate_contract", lambda *args, **kwargs: contract)
    report = gate.score_learning_gate(
        _records(digest, registration, contract),
        {},
        {},
        experiment_config_sha256=digest,
    )
    assert report["status"] == "pass"
    assert report["promotion_eligible_passed_arms"] == [
        "think-transformer-k2",
        "think-transformer-k4",
    ]
    assert set(report["comparisons"]) == {
        "think-transformer-k2",
        "think-transformer-k4",
    }
    assert report["descriptive_compute_control"]["arm"] == "transformer-k0"
    assert report["registered_thresholds"] == {
        "minimum_relative_improvement_vs_k1": 0.02,
        "maximum_nonforced_decision_micro_ce_regression": 0.005,
    }


def test_resolved_config_rejects_wrong_transformer_depth():
    resolved = _resolved("think-transformer-k2")
    resolved["fields"]["graph_layers"] = 7
    with pytest.raises(gate.GateInputError, match="graph_layers drifted"):
        gate._validate_resolved_config(resolved, arm="think-transformer-k2", row=1)


def test_nonforced_safety_ignores_forced_action_outliers(monkeypatch):
    digest = "a" * 64
    registration = {
        "corpus_fingerprint": "b" * 64,
        "training_manifest_sha256": "c" * 64,
        "validation_manifest_sha256": "d" * 64,
        "initial_checkpoint_sha256_by_arm_seed": {
            f"{arm}@{seed}": hashlib.sha256(f"init-{arm}-{seed}".encode()).hexdigest()
            for arm in gate.ARMS
            for seed in gate.SEEDS
        },
    }
    contract = {
        "arms": {}, "registration": registration, "gate_sha": "e" * 64,
        "evidence_file_sha": "f" * 64, "evidence_semantic_sha": "1" * 64,
        "exporter_source_sha256": "2" * 64,
        "exporter_engine_source_sha256": "3" * 64,
        "exporter_helper_source_sha256": "4" * 64,
    }
    monkeypatch.setattr(gate, "FROZEN_DECISIONS_PER_RUN", 4)
    monkeypatch.setattr(gate, "FROZEN_HOLDOUT_GAMES", 2)
    monkeypatch.setattr(gate, "_validate_contract", lambda *args, **kwargs: contract)
    report = gate.score_learning_gate(
        _records(digest, registration, contract, forced_outlier=True), {}, {},
        experiment_config_sha256=digest,
    )
    assert report["status"] == "pass"
    assert report["support"]["nonforced_decisions_per_run"] == 2
    assert report["comparisons"]["think-transformer-k2"][
        "nonforced_decision_micro_safety"
    ]["relative_regression"] < 0
