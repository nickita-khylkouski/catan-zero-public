from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tools.rnd_e3_holdout_export as exporter
import tools.rnd_e3_learning_gate as gate_module
from catan_zero.rl.entity_token_features import (
    PLAYER_ACTOR_FLAG_SLOT,
    PLAYER_FEATURE_SIZE,
)
from tools.rnd_e3_a1_admission import ARMS, _validate_contract
from tools.rnd_e3_learning_gate import score_learning_gate


def _canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _support_fixture(tmp_path: Path):
    seeds = np.asarray([10, 11], dtype="<i8")
    seed_sha = "sha256:" + hashlib.sha256(seeds.tobytes()).hexdigest()
    contract = "sha256:" + "c" * 64
    validation = _write(
        tmp_path / "validation.json",
        {
            "schema_version": "train-validation-game-seeds-v1",
            "a1_contract_sha256": contract,
            "validation_game_seed_count": 2,
            "validation_game_seed_set_sha256": seed_sha,
            "game_seeds": [10, 11],
        },
    )
    records = [
        {
            "game_seed": seed,
            "job_id": f"job-{seed}",
            "worker_id": "gpu00",
            "category": "current_producer",
            "producer_checkpoint_sha256": "sha256:" + "a" * 64,
            "opponent_checkpoint_sha256": ["sha256:" + "b" * 64],
            "split": "train" if seed == 9 else "validation",
        }
        for seed in (9, 10, 11)
    ]
    training = _write(
        tmp_path / "training.json",
        {
            "schema_version": "a1-selected-training-games-v1",
            "a1_contract_sha256": contract,
            "selection_rule": "lowest_seed_complete_per_job",
            "selected_game_count": 3,
            "selected_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(
                np.asarray([9, 10, 11], dtype="<i8").tobytes()
            ).hexdigest(),
            "category_game_counts": {"current_producer": 3},
            "training_game_count": 1,
            "training_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(np.asarray([9], dtype="<i8").tobytes()).hexdigest(),
            "validation_game_count": 2,
            "validation_game_seed_set_sha256": seed_sha,
            "records": records,
            "records_sha256": "sha256:" + _canonical_sha(records),
        },
    )
    return validation, training


class TinyCorpus:
    def __init__(self, regime: str = "public_conservation_pimc_v1") -> None:
        self.meta = {"payload_inventory_sha256": "sha256:" + "d" * 64}
        self.data = {
            "game_seed": np.asarray([10, 11], dtype=np.int64),
            "decision_index": np.asarray([7, 8], dtype=np.int64),
            "target_information_regime": np.asarray([regime, regime]),
        }

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return 2


def test_checked_in_registered_experiment_is_valid() -> None:
    path = Path("configs/rnd/e3_a1_screen_20260710/experiment.registered.json")
    experiment = json.loads(path.read_text())
    _validate_contract(experiment, registered=True)


def test_checked_in_export_contract_binds_exporter_and_helper_sources() -> None:
    experiment_path = Path(
        "configs/rnd/e3_a1_screen_20260710/experiment.registered.json"
    ).resolve()
    contract_path = Path(
        "configs/rnd/e3_a1_screen_20260710/evidence_export.v1.json"
    ).resolve()
    experiment = json.loads(experiment_path.read_text())
    contract = json.loads(contract_path.read_text())
    provenance = exporter._validate_export_contract(
        contract,
        contract_path=contract_path,
        experiment=experiment,
        experiment_path=experiment_path,
    )
    assert provenance["exporter_source_sha256"] == exporter._sha256_file(
        Path(exporter.__file__)
    )
    tampered = dict(contract)
    tampered["exporter_helper_source_sha256"] = "0" * 64
    semantic = dict(tampered)
    semantic.pop("config_sha256")
    tampered["config_sha256"] = _canonical_sha(semantic)
    with pytest.raises(exporter.ExportError, match="source drift"):
        exporter._validate_export_contract(
            tampered,
            contract_path=contract_path,
            experiment=experiment,
            experiment_path=experiment_path,
        )


def test_validation_support_is_exact_and_public(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(exporter, "EXPECTED_HOLDOUT_GAMES", 2)
    monkeypatch.setattr(exporter, "EXPECTED_HOLDOUT_ROWS", 2)
    monkeypatch.setattr(exporter, "EXPECTED_CORPUS_ROWS", 2)
    validation, training = _support_fixture(tmp_path)
    experiment = {"common": {"information_regime": "public_conservation_pimc_v1"}}
    corpus = TinyCorpus()
    rows = exporter._validation_rows(
        corpus,
        experiment=experiment,
        validation_manifest=validation,
        training_manifest=training,
        payload_inventory_sha=corpus.meta["payload_inventory_sha256"],
    )
    assert [(row["game_seed"], row["decision_index"]) for row in rows] == [
        (10, 7),
        (11, 8),
    ]

    with pytest.raises(exporter.ExportError, match="public-information"):
        exporter._validation_rows(
            TinyCorpus("omniscient"),
            experiment=experiment,
            validation_manifest=validation,
            training_manifest=training,
            payload_inventory_sha=corpus.meta["payload_inventory_sha256"],
        )


def test_validation_support_rejects_wrong_row_count(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(exporter, "EXPECTED_HOLDOUT_GAMES", 2)
    monkeypatch.setattr(exporter, "EXPECTED_HOLDOUT_ROWS", 3)
    monkeypatch.setattr(exporter, "EXPECTED_CORPUS_ROWS", 2)
    validation, training = _support_fixture(tmp_path)
    with pytest.raises(exporter.ExportError, match="exactly 3 rows"):
        exporter._validation_rows(
            TinyCorpus(),
            experiment={
                "common": {"information_regime": "public_conservation_pimc_v1"}
            },
            validation_manifest=validation,
            training_manifest=training,
            payload_inventory_sha="sha256:" + "d" * 64,
        )


@dataclass
class PolicyConfig:
    hidden_size: int = 384
    state_layers: int = 9
    attention_heads: int = 6
    state_trunk: str = "rrt"
    relational_block_pattern: str = "RRTRRTRRT"
    relational_ff_size: int = 1024
    relational_bases: int = 4
    relational_action_cross_layers: int = 1
    latent_deliberation_slots: int = 8
    latent_deliberation_steps: int = 2


class Policy:
    trained_with_masked_hidden_info = True
    config = PolicyConfig()
    model = torch.nn.Linear(1, 1, bias=False)


def test_loaded_policy_must_match_registered_k_and_public_mask() -> None:
    experiment = {
        "common": {
            "hidden_size": 384,
            "state_layers": 9,
            "attention_heads": 6,
            "state_trunk": "rrt",
            "relational_block_pattern": "RRTRRTRRT",
            "relational_ff_size": 1024,
            "relational_bases": 4,
            "relational_action_cross_layers": 1,
            "latent_deliberation_slots": 8,
        }
    }
    arm = {"latent_deliberation_steps": 2}
    exporter._validate_loaded_policy(Policy(), experiment=experiment, arm=arm)
    Policy.config.latent_deliberation_steps = 4
    with pytest.raises(exporter.ExportError, match="latent_deliberation_steps"):
        exporter._validate_loaded_policy(Policy(), experiment=experiment, arm=arm)
    Policy.config.latent_deliberation_steps = 2
    Policy.trained_with_masked_hidden_info = False
    with pytest.raises(exporter.ExportError, match="public-masked"):
        exporter._validate_loaded_policy(Policy(), experiment=experiment, arm=arm)
    Policy.trained_with_masked_hidden_info = True


def test_atomic_jsonl_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "evidence.jsonl"
    records = [{"arm_id": "think-rrt-k2", "soft_target_policy_ce": 0.25}]
    exporter._publish_jsonl_atomic(output, records)
    assert json.loads(output.read_text()) == records[0]
    with pytest.raises(exporter.ExportError, match="overwrite"):
        exporter._publish_jsonl_atomic(output, records)


def test_admission_flag_binding_rejects_duplicate_and_missing_value() -> None:
    with pytest.raises(exporter.ExportError, match="exactly one --seed"):
        exporter._flag_value(["--seed", "11", "--seed", "29"], "--seed")
    with pytest.raises(exporter.ExportError, match="exactly one --seed"):
        exporter._flag_value(["--seed"], "--seed")


def test_tampered_training_report_source_is_rejected() -> None:
    experiment = json.loads(
        Path("configs/rnd/e3_a1_screen_20260710/experiment.registered.json").read_text()
    )
    reported = {
        relative: experiment["registration"]["executing_learner_source_sha256"][
            relative
        ]
        for relative in exporter._REPORT_SOURCE_FILES
    }
    reported["tools/train_bc.py"] = "0" * 64
    with pytest.raises(exporter.ExportError, match="training report source"):
        exporter._validate_sources(
            {"rnd_executing_learner_source_sha256": reported}, experiment
        )


def test_tampered_admission_experiment_binding_is_rejected(tmp_path: Path) -> None:
    experiment = {"config_sha256": "1" * 64}
    experiment_path = _write(tmp_path / "experiment.json", experiment)
    admission_path = _write(tmp_path / "admission.json", {})
    admission = {
        "schema_version": exporter.ADMISSION_SCHEMA,
        "experiment_config_sha256": "0" * 64,
    }
    with pytest.raises(exporter.ExportError, match="registered experiment bytes"):
        exporter._validate_admission(
            admission,
            admission_path=admission_path,
            experiment=experiment,
            experiment_path=experiment_path,
            checkpoint=tmp_path / "checkpoint.pt",
            report=tmp_path / "report.json",
            corpus_dir=tmp_path,
            validation_manifest=tmp_path / "validation.json",
        )


def test_logical_validation_path_need_not_exist_under_rnd_relocation(
    tmp_path: Path,
) -> None:
    logical = "/retired/production/a1_post_wave.audit.validation_seeds.json"
    assert not Path(logical).exists()
    corpus = tmp_path / "corpus"
    _write(
        corpus / "corpus_meta.json",
        {"a1_post_wave_audit": {"validation_holdout": {"path": logical}}},
    )
    report = {
        "input_validation_game_seed_manifest": logical,
        "rnd_a1_artifact_relocation": {
            "files": {"validation_manifest": {"logical_path": logical}}
        },
    }
    assert (
        exporter._validate_logical_validation_binding(report, corpus_dir=corpus)
        == logical
    )
    report["input_validation_game_seed_manifest"] = "/wrong/logical/path.json"
    with pytest.raises(exporter.ExportError, match="logical path differs"):
        exporter._validate_logical_validation_binding(report, corpus_dir=corpus)


def test_registered_parameter_contract_uses_current_shared_latent_size() -> None:
    assert ARMS["rrt-k0"][1] == 20_070_932
    assert {ARMS[arm][1] for arm in ARMS if arm != "rrt-k0"} == {22_146_068}


class InferenceCorpus(TinyCorpus):
    def __init__(self) -> None:
        super().__init__()
        for key in exporter.ENTITY_BATCH_KEYS:
            self.data[key] = np.zeros((2, 1), dtype=np.float32)
        self.data["player_tokens"] = np.zeros(
            (2, 4, PLAYER_FEATURE_SIZE), dtype=np.float32
        )
        self.data["player_tokens"][:, 0, PLAYER_ACTOR_FLAG_SLOT] = 1
        self.data["legal_action_mask"] = np.asarray([[True, True], [True, True]])
        self.data["legal_action_ids"] = np.asarray([[1, 2], [1, 2]], dtype=np.int16)
        self.data["legal_action_context"] = np.zeros((2, 2, 3), dtype=np.float32)
        self.data["action_taken"] = np.asarray([1, 2], dtype=np.int16)
        self.data["target_policy"] = np.asarray([[0.75, 0.25], [0.25, 0.75]])
        self.data["target_policy_mask"] = self.data["legal_action_mask"].copy()


class InferencePolicy:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False)

    def forward_legal_np(self, entity, legal_ids, contexts, *, return_q):
        del entity, contexts, return_q
        return {"logits": torch.zeros(legal_ids.shape, dtype=torch.float32)}


def test_mocked_export_schema_is_accepted_by_learning_gate(
    monkeypatch, tmp_path: Path
) -> None:
    """Exercise JSONL emission, then use an emitted row as the scorer schema template."""

    checkpoint = tmp_path / "checkpoint.pt"
    report = tmp_path / "report.json"
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    admission = tmp_path / "admission.json"
    for path in (checkpoint, report, training, validation):
        path.write_bytes(path.name.encode())
    _write(report, {})
    experiment = {
        "registration": {
            "corpus_fingerprint": "f" * 64,
            "training_manifest_sha256": exporter._sha256_file(training),
            "validation_manifest_sha256": exporter._sha256_file(validation),
        },
        "common": {"information_regime": "public_conservation_pimc_v1"},
    }
    experiment_path = _write(tmp_path / "experiment.json", experiment)
    evidence_contract = _write(tmp_path / "evidence_contract.json", {})
    _write(admission, {"checkpoint": str(checkpoint), "report": str(report)})
    arm = {
        "arm_id": "think-rrt-k2",
        "latent_deliberation_steps": 2,
        "capacity_class": "shared_think_22146068",
        "expected_parameters": 1,
    }
    provenance = {
        "checkpoint_sha256": "1" * 64,
        "training_report_sha256": "2" * 64,
        "admission_manifest_sha256": "3" * 64,
        "initial_checkpoint_sha256": "4" * 64,
        "resolved_train_config": {
            "schema_version": 6,
            "pipeline": "train",
            "fields": {},
        },
        "resolved_train_config_sha256": "5" * 64,
        "graph_history_features": True,
        "parameter_count": 1,
        "optimizer_steps": 250,
        "global_batch_size": 4096,
        "sample_presentations": 1_024_000,
        "experiment_config_sha256": exporter._sha256_file(experiment_path),
        "corpus_fingerprint": "f" * 64,
        "training_manifest_sha256": exporter._sha256_file(training),
        "validation_manifest_sha256": exporter._sha256_file(validation),
    }
    monkeypatch.setattr(exporter, "EXPECTED_HOLDOUT_ROWS", 2)
    monkeypatch.setattr(exporter, "_validate_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        exporter,
        "_validate_export_contract",
        lambda *args, **kwargs: {
            "evidence_export_contract_sha256": "6" * 64,
            "evidence_export_contract_semantic_sha256": "7" * 64,
            "exporter_source_sha256": "8" * 64,
            "exporter_helper_source_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        exporter,
        "_validate_admission",
        lambda *args, **kwargs: ("think-rrt-k2", 11, arm),
    )
    monkeypatch.setattr(exporter, "_corpus_fingerprint", lambda _path: "f" * 64)
    monkeypatch.setattr(
        exporter, "_validate_corpus_payloads", lambda _path: "sha256:" + "d" * 64
    )
    monkeypatch.setattr(
        exporter, "_validate_report", lambda *args, **kwargs: provenance
    )
    monkeypatch.setattr(
        exporter,
        "_validation_rows",
        lambda *args, **kwargs: [
            {"row_index": 0, "game_seed": 10, "decision_index": 7},
            {"row_index": 1, "game_seed": 11, "decision_index": 8},
        ],
    )
    monkeypatch.setattr(
        exporter, "_validate_loaded_policy", lambda *args, **kwargs: None
    )
    corpus = InferenceCorpus()
    output = tmp_path / "evidence.jsonl"
    assert (
        exporter.export_holdout_evidence(
            experiment_config=experiment_path,
            evidence_contract=evidence_contract,
            admission_manifest=admission,
            corpus_dir=tmp_path,
            training_manifest=training,
            validation_manifest=validation,
            checkpoint=checkpoint,
            training_report=report,
            output=output,
            corpus_loader=lambda _path: corpus,
            policy_loader=lambda *args, **kwargs: InferencePolicy(),
        )
        == 2
    )
    emitted = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(emitted) == 2

    registered_path = Path(
        "configs/rnd/e3_a1_screen_20260710/experiment.registered.json"
    )
    registered = json.loads(registered_path.read_text())
    gate = json.loads(
        Path("configs/rnd/e3_a1_screen_20260710/learning_gate.v1.json").read_text()
    )
    gate["minimum_holdout_games"] = 1
    gate["minimum_nonforced_decisions"] = 2
    gate["bootstrap_samples"] = 100
    monkeypatch.setattr(gate_module, "FROZEN_MINIMUM_HOLDOUT_GAMES", 1)
    monkeypatch.setattr(gate_module, "FROZEN_MINIMUM_NONFORCED_DECISIONS", 2)
    monkeypatch.setattr(gate_module, "FROZEN_BOOTSTRAP_SAMPLES", 100)
    semantic = dict(gate)
    semantic.pop("config_sha256")
    gate["config_sha256"] = _canonical_sha(semantic)
    registered_sha = exporter._sha256_file(registered_path)
    arms = {item["arm_id"]: item for item in registered["arms"]}
    ce = {
        "rrt-k0": 0.8,
        "think-rrt-k1": 1.0,
        "think-rrt-k2": 0.95,
        "think-rrt-k4": 0.96,
        "think-rrt-k8": 0.97,
    }
    records = []
    for arm_id, arm_config in arms.items():
        for seed in (11, 29, 47):
            resolved = {
                "schema_version": 6,
                "pipeline": "train",
                "fields": {
                    "hidden_size": 384,
                    "graph_layers": 9,
                    "attention_heads": 6,
                    "entity_state_trunk": "rrt",
                    "relational_block_pattern": "RRTRRTRRT",
                    "relational_ff_size": 1024,
                    "relational_bases": 4,
                    "relational_action_cross_layers": 1,
                    "latent_deliberation_slots": 8,
                    "latent_deliberation_steps": arm_config[
                        "latent_deliberation_steps"
                    ],
                    "max_steps": 250,
                    "batch_size": 1024,
                    "grad_accum_steps": 4,
                    "mask_hidden_info": True,
                },
            }
            run = {
                "checkpoint_sha256": hashlib.sha256(
                    f"{arm_id}:{seed}".encode()
                ).hexdigest(),
                "training_report_sha256": hashlib.sha256(
                    f"r:{arm_id}:{seed}".encode()
                ).hexdigest(),
                "admission_manifest_sha256": hashlib.sha256(
                    f"a:{arm_id}:{seed}".encode()
                ).hexdigest(),
                "initial_checkpoint_sha256": registered["registration"][
                    "initial_checkpoint_sha256_by_arm_seed"
                ][f"{arm_id}@{seed}"],
                "resolved_train_config": resolved,
                "resolved_train_config_sha256": _canonical_sha(resolved),
                "graph_history_features": True,
                "parameter_count": arm_config["expected_parameters"],
                "optimizer_steps": 250,
                "global_batch_size": 4096,
                "sample_presentations": 1_024_000,
                "evidence_export_contract_sha256": gate[
                    "evidence_export_contract_file_sha256"
                ],
                "evidence_export_contract_semantic_sha256": gate[
                    "evidence_export_contract_semantic_sha256"
                ],
                "exporter_source_sha256": gate["exporter_source_sha256"],
                "exporter_helper_source_sha256": gate["exporter_helper_source_sha256"],
            }
            for base in emitted:
                row = dict(base)
                row.update(
                    {
                        "arm_id": arm_id,
                        "training_seed": seed,
                        "soft_target_policy_ce": ce[arm_id],
                        "experiment_config_sha256": registered_sha,
                        "corpus_fingerprint": registered["registration"][
                            "corpus_fingerprint"
                        ],
                        "training_manifest_sha256": registered["registration"][
                            "training_manifest_sha256"
                        ],
                        "validation_manifest_sha256": registered["registration"][
                            "validation_manifest_sha256"
                        ],
                        "run_provenance": run,
                    }
                )
                records.append(row)
    result = score_learning_gate(
        records,
        registered,
        gate,
        experiment_config_sha256=registered_sha,
        bootstrap_samples=100,
    )
    assert result["status"] == "pass"
