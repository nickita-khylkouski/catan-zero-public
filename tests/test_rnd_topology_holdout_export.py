from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tools.rnd_build_topology_sensitive_mask as mask_builder
import tools.rnd_topology_holdout_export as holdout_exporter
from catan_zero.rl.entity_token_features import (
    PLAYER_ACTOR_FLAG_SLOT,
    PLAYER_FEATURE_SIZE,
    PUBLIC_MASK_PLAYER_SLOTS,
)
from catan_zero.rl.pipeline_configs import TrainConfig
from tools.train_bc import _training_data_fingerprint
from tools.rnd_topology_holdout_export import (
    ENTITY_BATCH_KEYS,
    ExportError,
    MASK_SCHEMA,
    RUN_SCHEMA,
    _DYNAMIC_TRAIN_CONFIG_FIELDS,
    _EXECUTING_LEARNER_SOURCE_FILES,
    _a1_legacy_canonical_sha,
    _resolve_report_repo_path,
    _validate_experiment_self_hash,
    export_holdout_evidence,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


class FakeCorpus:
    def __init__(self, path: Path | None = None) -> None:
        n = 2
        self.row_count = n
        self.meta = (
            json.loads((path / "corpus_meta.json").read_text())
            if path is not None
            else {"payload_inventory_sha256": "sha256:" + "a" * 64}
        )
        self.data = {key: np.zeros((n, 1), dtype=np.float32) for key in ENTITY_BATCH_KEYS}
        players = np.full((n, 4, PLAYER_FEATURE_SIZE), 9.0, dtype=np.float32)
        players[:, 0, PLAYER_ACTOR_FLAG_SLOT] = 1.0
        players[:, 1:, PLAYER_ACTOR_FLAG_SLOT] = 0.0
        self.data["player_tokens"] = players
        legal_mask = np.asarray([[True, True], [True, False]])
        self.data["legal_action_mask"] = legal_mask
        action_tokens = np.zeros((n, 2, 50), dtype=np.float16)
        settlement = mask_builder.ACTION_TYPES.index("BUILD_SETTLEMENT")
        road = mask_builder.ACTION_TYPES.index("BUILD_ROAD")
        action_tokens[0, :, 2 + settlement] = 1
        action_tokens[1, 0, 2 + road] = 1
        action_targets = np.full((n, 2, 4), -1, dtype=np.int16)
        action_targets[0, :, 1] = [4, 8]
        action_targets[1, 0, 2] = 3
        self.data["legal_action_tokens"] = action_tokens
        self.data["legal_action_target_ids"] = action_targets
        self.data.update(
            {
                "legal_action_ids": np.asarray([[1, 2], [3, -1]], dtype=np.int16),
                "legal_action_context": np.zeros((n, 2, 3), dtype=np.float32),
                "action_taken": np.asarray([2, 3], dtype=np.int16),
                "target_policy": np.asarray([[0.25, 0.75], [1.0, 0.0]], dtype=np.float32),
                "target_policy_mask": legal_mask.copy(),
                "game_seed": np.asarray([10, 11], dtype=np.int64),
                "decision_index": np.asarray([7, 8], dtype=np.int64),
                "phase": np.asarray(["main", "main"]),
                "target_information_regime": np.asarray(
                    ["public_conservation_pimc_v1"] * n
                ),
            }
        )

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str):
        return self.data[key]

    def __len__(self) -> int:
        return 2

    def keys(self):
        return self.data.keys()


@dataclass
class FakeConfig:
    hidden_size: int = 16
    state_layers: int = 2
    attention_heads: int = 2
    topology_adapter_layers: str = "2"
    topology_adapter_width: int = 8
    topology_adapter_bases: int = 4
    topology_adapter_heads: int = 2
    topology_adapter_share_weights: bool = True
    topology_adapter_edge_control: str = "true_topology"
    topology_adapter_kind: str = "local_attention_v2"


class FakePolicy:
    trained_with_masked_hidden_info = True

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.config = FakeConfig()

    def forward_legal_np(self, entity, legal_ids, contexts, *, return_q):
        del contexts, return_q
        players = entity["player_tokens"]
        for slot in PUBLIC_MASK_PLAYER_SLOTS:
            assert np.all(players[:, 1:, slot] == 0)
            assert np.all(players[:, 0, slot] == 9)
        logits = torch.tensor([[0.0, np.log(3.0)], [2.0, -100.0]])
        return {"logits": logits[: len(legal_ids)]}


def _fixtures(tmp_path: Path) -> dict[str, Path | dict]:
    corpus_dir = tmp_path / "corpus"
    seeds = np.asarray([10, 11], dtype="<i8")
    validation_seed_sha = "sha256:" + hashlib.sha256(seeds.tobytes()).hexdigest()
    payload = corpus_dir / "row_offsets.dat"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"locked memmap fixture")
    dummy_payload = corpus_dir / "dummy.dat"
    dummy_payload.write_bytes(b"dummy column fixture")
    inventory = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + _sha(path),
        }
        for path in sorted((payload, dummy_payload), key=lambda item: item.name)
    ]
    inventory_sha = "sha256:" + _canonical_sha(inventory)
    corpus_meta = _write(
        corpus_dir / "corpus_meta.json",
        {
            "schema": "memmap_corpus_v1",
            "columns": {"dummy": {"kind": "fixed"}},
            "payload_inventory_schema": "memmap-payload-inventory-v1",
            "payload_inventory": inventory,
            "payload_inventory_sha256": inventory_sha,
            "selected_game_seed_manifest": {
                "validation_game_seed_set_sha256": validation_seed_sha
            },
        },
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    contract_sha = "sha256:" + "c" * 64
    holdout = _write(
        tmp_path / "validation.json",
        {
            "schema_version": "train-validation-game-seeds-v1",
            "a1_contract_sha256": contract_sha,
            "validation_game_seed_count": 2,
            "validation_game_seed_set_sha256": validation_seed_sha,
            "game_seeds": [10, 11],
        },
    )
    producer_sha = "sha256:" + "a" * 64
    records = [
        {
            "game_seed": seed,
            "job_id": f"job-{seed}",
            "worker_id": "gpu00",
            "category": "current_producer",
            "producer_checkpoint_sha256": producer_sha,
            "opponent_checkpoint_sha256": [producer_sha],
            "split": "train" if seed == 9 else "validation",
        }
        for seed in (9, 10, 11)
    ]
    training = _write(
        tmp_path / "training.json",
        {
            "schema_version": "a1-selected-training-games-v1",
            "a1_contract_sha256": contract_sha,
            "selection_rule": "lowest_seed_complete_per_job",
            "selected_game_count": 3,
            "selected_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(np.asarray([9, 10, 11], dtype="<i8").tobytes()).hexdigest(),
            "category_game_counts": {"current_producer": 3},
            "training_game_count": 1,
            "training_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(np.asarray([9], dtype="<i8").tobytes()).hexdigest(),
            "validation_game_count": 2,
            "validation_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(seeds.tobytes()).hexdigest(),
            "records": records,
            "records_sha256": "sha256:" + _canonical_sha(records),
        },
    )
    contract_semantic = {
        "schema_version": "fixture-a1-contract/v1",
        "non_ascii_note": "Café",
    }
    contract_sha = "sha256:" + _a1_legacy_canonical_sha(contract_semantic)
    holdout_payload = json.loads(holdout.read_text())
    holdout_payload["a1_contract_sha256"] = contract_sha
    _write(holdout, holdout_payload)
    training_payload = json.loads(training.read_text())
    training_payload["a1_contract_sha256"] = contract_sha
    _write(training, training_payload)
    contract = _write(
        tmp_path / "contract.json",
        {**contract_semantic, "contract_sha256": contract_sha},
    )
    audit = _write(
        tmp_path / "audit.json",
        {
            "schema_version": "fixture-a1-audit/v1",
            "contract_path": str(contract.resolve()),
            "contract_sha256": contract_sha,
        },
    )
    meta_payload = json.loads(corpus_meta.read_text())
    meta_payload["selected_game_seed_manifest"].update(
        {"path": str(training.resolve()), "file_sha256": "sha256:" + _sha(training)}
    )
    meta_payload["a1_post_wave_audit"] = {
        "path": str(audit.resolve()),
        "file_sha256": "sha256:" + _sha(audit),
        "contract_sha256": contract_sha,
        "validation_holdout": {
            "path": str(holdout.resolve()),
            "file_sha256": "sha256:" + _sha(holdout),
        },
    }
    _write(corpus_meta, meta_payload)
    relocation_dir = tmp_path / "relocated-a1"
    relocation_dir.mkdir()
    relocation_sources = {
        "selected_game_manifest": training,
        "post_wave_audit": audit,
        "validation_manifest": holdout,
        "contract_lock": contract,
    }
    relocation_files = {}
    for role, source_path in relocation_sources.items():
        copied = relocation_dir / source_path.name
        copied.write_bytes(source_path.read_bytes())
        relocation_files[role] = {
            "logical_path": str(source_path.resolve()),
            "filename": source_path.name,
            "sha256": "sha256:" + _sha(source_path),
        }
    relocation = {
        "schema_version": "catan-zero-rnd-a1-artifact-relocation/v1",
        "files": relocation_files,
    }
    members = [
        {
            "decision_id": "seed:10:decision:7",
            "game_id": "seed:10",
            "game_seed": 10,
            "decision_index": 7,
            "category": "settlement_vertex_target",
            "action_type": "BUILD_SETTLEMENT",
            "distinct_legal_topology_targets": 2,
            "source_row_index": 0,
        }
    ]
    source = {
        "corpus": {
            "corpus_meta_file_sha256": "sha256:" + _sha(corpus_meta),
            "payload_inventory_sha256": inventory_sha,
            "row_count": 2,
        },
        "validation_manifest": {
            "file_sha256": "sha256:" + _sha(holdout),
            "manifest_sha256": "sha256:" + _canonical_sha(json.loads(holdout.read_text())),
            "game_seed_set_sha256": json.loads(holdout.read_text())["validation_game_seed_set_sha256"],
        },
    }
    mask_payload = {
        "schema_version": MASK_SCHEMA,
        "artifact_sha256_scope": "canonical_json_without_artifact_sha256",
        "config": {"schema_version": "catan-zero-topology-sensitive-mask-config/v1"},
        "config_sha256": "sha256:"
        + _canonical_sha({"schema_version": "catan-zero-topology-sensitive-mask-config/v1"}),
        "source": source,
        "source_sha256": "sha256:" + _canonical_sha(source),
        "summary": {
            "decision_count": 1,
            "game_count": 1,
            "category_counts": {"settlement_vertex_target": 1},
        },
        "members": members,
        "members_sha256": "sha256:" + _canonical_sha(members),
    }
    mask_payload["artifact_sha256"] = "sha256:" + _canonical_sha(mask_payload)
    mask = _write(tmp_path / "mask.json", mask_payload)
    resolved = {
        "hidden_size": 16,
        "state_layers": 2,
        "attention_heads": 2,
        "adapter_layers": "2",
        "information_regime": "public_only",
        "adapter_kind": "local_attention_v2",
        "adapter_width": 8,
        "adapter_heads": 2,
        "share_weights": True,
        "edge_control": "true_topology",
    }
    resolved["adapter_bases"] = 4
    init_sha = "b" * 64
    repo_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        relative: _sha(repo_root / relative)
        for relative in _EXECUTING_LEARNER_SOURCE_FILES
    }
    data_fingerprint = _training_data_fingerprint(str(corpus_dir), "memmap")
    recipe_overrides = {
        "data_format": "memmap",
        "mask_hidden_info": True,
        "resume_optimizer": False,
        "epochs": 1,
        "max_steps": 5,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "optimizer": "adam",
        "hidden_size": 16,
        "graph_layers": 2,
        "attention_heads": 2,
        "graph_dropout": 0.05,
        "symmetry_augment": False,
        "soft_target_temperature": 0.7,
        "soft_target_weight": 0.7,
        "soft_target_source": "policy",
        "soft_target_min_legal_coverage": 0.5,
        "policy_loss_weight": 1.0,
        "value_loss_weight": 0.1,
        "final_vp_loss_weight": 0.05,
        "q_loss_weight": 0.0,
        "policy_kl_anchor_weight": 0.0,
        "value_uncertainty_loss_weight": 0.0,
        "aux_subgoal_loss_weight": 0.0,
        "value_lr_mult": 1.0,
        "value_target_lambda": 1.0,
        "truncated_vp_margin_value_weight": 0.25,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 0.3,
        "forced_action_weight": 0.1,
        "fresh_optimizer": True,
        "information_regime": "public_only",
    }
    recipe_defaults = TrainConfig(
        arch="entity_graph", value_categorical_bins=0
    ).field_values()
    training_recipe = {
        key: value
        for key, value in recipe_defaults.items()
        if key not in _DYNAMIC_TRAIN_CONFIG_FIELDS
    }
    training_recipe.update(recipe_overrides)
    experiment_payload = {
            "config_sha256_scope": "canonical_json_without_config_sha256",
            "common": {
                "hidden_size": 16,
                "state_layers": 2,
                "attention_heads": 2,
                "adapter_layers": "2",
                "information_regime": "public_conservation_pimc_v1",
            },
            "arms": [
                {
                    "arm_id": "candidate",
                    "role": "primary_candidate",
                    "adapter_kind": "local_attention_v2",
                    "adapter_width": 8,
                    "adapter_bases": 4,
                    "adapter_heads": 2,
                    "share_weights": True,
                    "edge_control": "true_topology",
                    "expected_parameters": 1,
                }
            ],
            "training_recipe": training_recipe,
            "frozen_inputs": {
                "warm_start_checkpoint_sha256_by_arm": {"candidate": init_sha},
                "executing_learner_source_sha256": source_hashes,
                "a1_artifact_relocation": relocation,
            },
            "learning_gate": {
                "seeds": [1, 2],
                "optimizer_steps": 5,
                "global_batch_size": 8,
                "sample_presentations_per_arm_seed": 40,
                "training_manifest_sha256": _sha(training),
                "holdout_manifest_sha256": _sha(holdout),
                "topology_mask_registration_artifact_sha256": _sha(mask),
                "training_data_sha256": inventory_sha.removeprefix("sha256:"),
            },
        }
    experiment_payload["config_sha256"] = _canonical_sha(experiment_payload)
    experiment = _write(tmp_path / "experiment.json", experiment_payload)

    train_config = TrainConfig(
        arch="entity_graph",
        data=str(corpus_dir),
        data_fingerprint=data_fingerprint,
        init_checkpoint="warm-start.pt",
        init_checkpoint_sha256="sha256:" + init_sha,
        resume_optimizer=False,
        data_format="memmap",
        mask_hidden_info=True,
        seed=1,
        epochs=1,
        max_steps=5,
        batch_size=2,
        hidden_size=16,
        graph_layers=2,
        attention_heads=2,
        value_categorical_bins=0,
        topology_adapter_layers="2",
        topology_adapter_width=8,
        topology_adapter_bases=4,
        topology_adapter_kind="local_attention_v2",
        topology_adapter_heads=2,
        topology_adapter_share_weights=True,
        topology_adapter_edge_control="true_topology",
        grad_accum_steps=4,
        validation_game_seed_manifest=str(holdout),
        a1_memmap_payload_inventory_sha256=inventory_sha,
        rnd_a1_artifact_dir=str(relocation_dir),
    )
    sidecar = Path(str(checkpoint) + ".optimizer.pt")
    sidecar.write_bytes(b"optimizer fixture")
    report_payload = {
        **train_config.field_values(),
        "arch": "entity_graph",
        "config_hash": train_config.config_hash(),
        "full_config_hash": train_config.full_config_hash(),
        "resolved_train_config": train_config.canonical_payload(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": "sha256:" + _sha(checkpoint),
        "optimizer_sidecar": str(sidecar.resolve()),
        "optimizer_sidecar_sha256": "sha256:" + _sha(sidecar),
        "steps_completed": 5,
        "global_batch_size": 8,
        "sample_presentations": 40,
        "world_size": 1,
        "parameter_count": 1,
        "graph_tokens": None,
        "graph_layers": 2,
        "topology_adapter_layers": "2",
        "topology_adapter_width": 8,
        "topology_adapter_bases": 4,
        "topology_adapter_kind": "local_attention_v2",
        "topology_adapter_heads": 2,
        "topology_adapter_share_weights": True,
        "topology_adapter_edge_control": "true_topology",
        "input_validation_game_seed_manifest_sha256": "sha256:" + _sha(holdout),
        "validation_game_seed_set_sha256": validation_seed_sha,
        "a1_memmap_payload_inventory_sha256": inventory_sha,
        "optimizer_restored": False,
        "rnd_executing_learner_source_sha256": source_hashes,
        "rnd_a1_artifact_relocation": relocation,
    }
    report = _write(tmp_path / "report.json", report_payload)
    run = _write(
        tmp_path / "run.json",
        {
            "schema_version": RUN_SCHEMA,
            "arm": "candidate",
            "training_seed": 1,
            "training_manifest_sha256": _sha(training),
            "training_report": {"path": str(report), "file_sha256": _sha(report)},
            "experiment_config": {
                "path": str(experiment),
                "file_sha256": _sha(experiment),
            },
            "checkpoint": {"path": str(checkpoint), "file_sha256": _sha(checkpoint)},
            "optimizer_sidecar": {"path": str(sidecar), "file_sha256": _sha(sidecar)},
        },
    )
    return {
        "checkpoint": checkpoint,
        "corpus_dir": corpus_dir,
        "training_manifest": training,
        "validation_manifest": holdout,
        "topology_mask": mask,
        "run_manifest": run,
        "experiment_config_path": experiment,
        "output": tmp_path / "evidence.jsonl",
    }


def _export(paths: dict, **overrides) -> int:
    arguments = {
        **paths,
        "arm": "candidate",
        "training_seed": 1,
        "batch_size": 2,
        "device": "cpu",
        "corpus_loader": lambda path: FakeCorpus(path),
        "policy_loader": lambda *_args, **_kwargs: FakePolicy(),
        **overrides,
    }
    return export_holdout_evidence(**arguments)


def test_nested_report_relative_artifact_path_resolves_from_trainer_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "trainer-repo"
    nested_report = repo / "runs" / "reports" / "nested" / "report.json"
    checkpoint = repo / "runs" / "arm-1" / "checkpoint.pt"
    nested_report.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    nested_report.write_text("{}")
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(holdout_exporter, "_ROOT", repo)

    resolved = _resolve_report_repo_path(
        "runs/arm-1/checkpoint.pt", label="checkpoint"
    )
    assert resolved == checkpoint.resolve()
    assert resolved != (nested_report.parent / "runs/arm-1/checkpoint.pt").resolve()


def test_report_relative_artifact_path_rejects_parent_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "trainer-repo"
    repo.mkdir()
    monkeypatch.setattr(holdout_exporter, "_ROOT", repo)
    with pytest.raises(ExportError, match="contains traversal"):
        _resolve_report_repo_path("../outside/checkpoint.pt", label="checkpoint")


def test_a1_contract_digest_uses_legacy_ascii_escaping() -> None:
    payload = {"label": "Café", "nested": {"snowman": "☃"}}
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    assert _a1_legacy_canonical_sha(payload) == expected
    assert _a1_legacy_canonical_sha(payload) != holdout_exporter._canonical_sha(payload)


def test_real_preregistration_self_hash_and_recipe_fields_are_valid() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/rnd/topology_real_train_20260710/experiment.json"
    )
    config = json.loads(path.read_text())
    _validate_experiment_self_hash(config)
    train_fields = {field.name for field in dataclasses.fields(TrainConfig)}
    assert set(config["training_recipe"]) == (
        train_fields - _DYNAMIC_TRAIN_CONFIG_FIELDS
    ) | {"fresh_optimizer", "information_regime"}
    assert config["training_recipe"]["fresh_optimizer"] is True
    assert config["training_recipe"]["information_regime"] == "public_only"
    assert set(config["frozen_inputs"]["a1_artifact_relocation"]["files"]) == {
        "selected_game_manifest",
        "post_wave_audit",
        "validation_manifest",
        "contract_lock",
    }


def _rebind_training_manifest(paths: dict, payload: dict) -> None:
    _write(paths["training_manifest"], payload)
    digest = _sha(paths["training_manifest"])
    run = json.loads(paths["run_manifest"].read_text())
    run["training_manifest_sha256"] = digest
    _write(paths["run_manifest"], run)
    experiment = json.loads(paths["experiment_config_path"].read_text())
    experiment["learning_gate"]["training_manifest_sha256"] = digest
    _rebind_experiment(paths, experiment)


def _rebind_experiment(paths: dict, payload: dict) -> None:
    payload.pop("config_sha256", None)
    payload["config_sha256"] = _canonical_sha(payload)
    _write(paths["experiment_config_path"], payload)
    run = json.loads(paths["run_manifest"].read_text())
    run["experiment_config"]["file_sha256"] = _sha(paths["experiment_config_path"])
    _write(paths["run_manifest"], run)


def _rebind_report(paths: dict, payload: dict) -> None:
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    _write(report_path, payload)
    run = json.loads(paths["run_manifest"].read_text())
    run["training_report"]["file_sha256"] = _sha(report_path)
    _write(paths["run_manifest"], run)


def test_exports_exact_gate_rows_with_public_mask_and_frozen_labels(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    assert _export(paths) == 2

    records = [json.loads(line) for line in paths["output"].read_text().splitlines()]
    assert records[0]["policy_ce"] == pytest.approx(
        -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    )
    assert records[0]["hard_action_ce"] == pytest.approx(-np.log(0.75))
    assert records[0]["game_id"] == "seed:10"
    assert records[0]["decision_id"] == "seed:10:decision:7"
    assert records[0]["topology_sensitive"] is True
    assert records[1]["topology_sensitive"] is False
    assert records[0]["forced"] is False
    assert records[1]["forced"] is True
    assert records[0]["evaluation_split"] == "holdout"
    assert records[0]["is_training_game"] is False
    assert records[0]["run_provenance"]["checkpoint_sha256"] == _sha(paths["checkpoint"])


def test_accepts_artifact_emitted_by_real_mask_builder_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixtures(tmp_path)
    monkeypatch.setattr(mask_builder, "MemmapCorpus", FakeCorpus)
    monkeypatch.setattr(
        mask_builder,
        "_validate_memmap_payload_inventory",
        lambda _path, meta: meta["payload_inventory_sha256"],
    )
    artifact = mask_builder.build_mask(
        paths["corpus_dir"],
        paths["validation_manifest"],
        min_games=1,
        min_decisions=1,
        batch_size=1,
    )
    _write(paths["topology_mask"], artifact)
    experiment = json.loads(paths["experiment_config_path"].read_text())
    experiment["learning_gate"]["topology_mask_registration_artifact_sha256"] = _sha(
        paths["topology_mask"]
    )
    _rebind_experiment(paths, experiment)

    assert _export(paths) == 2
    records = [json.loads(line) for line in paths["output"].read_text().splitlines()]
    assert [record["topology_sensitive"] for record in records] == [True, False]


def test_refuses_mask_member_outside_validation_support(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    mask = json.loads(paths["topology_mask"].read_text())
    mask["members"][0].update(
        {
            "game_seed": 99,
            "game_id": "seed:99",
            "decision_index": 1,
            "decision_id": "seed:99:decision:1",
            "source_row_index": 0,
        }
    )
    mask["members_sha256"] = "sha256:" + _canonical_sha(mask["members"])
    mask.pop("artifact_sha256")
    mask["artifact_sha256"] = "sha256:" + _canonical_sha(mask)
    _write(paths["topology_mask"], mask)
    experiment = json.loads(paths["experiment_config_path"].read_text())
    experiment["learning_gate"]["topology_mask_registration_artifact_sha256"] = _sha(
        paths["topology_mask"]
    )
    _rebind_experiment(paths, experiment)

    with pytest.raises(ExportError, match="outside validation support"):
        _export(paths)
    assert not paths["output"].exists()


def test_refuses_checkpoint_or_manifest_hash_drift_before_inference(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    paths["checkpoint"].write_bytes(b"tampered")
    called = False

    def policy_loader(*_args, **_kwargs):
        nonlocal called
        called = True
        return FakePolicy()

    with pytest.raises(ExportError, match="checkpoint file hash mismatch"):
        _export(paths, policy_loader=policy_loader)
    assert called is False


def test_refuses_training_seed_injected_into_holdout_split(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    training = json.loads(paths["training_manifest"].read_text())
    overlap = dict(training["records"][1])
    overlap["split"] = "train"
    training["records"].insert(1, overlap)
    training["records_sha256"] = "sha256:" + _canonical_sha(training["records"])
    training["selected_game_count"] = 4
    training["training_game_count"] = 2
    training["category_game_counts"] = {"current_producer": 4}
    training["training_game_seed_set_sha256"] = "sha256:" + hashlib.sha256(
        np.asarray([9, 10], dtype="<i8").tobytes()
    ).hexdigest()
    _rebind_training_manifest(paths, training)

    with pytest.raises(ExportError, match="training and holdout game seeds overlap"):
        _export(paths)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report.__setitem__("steps_completed", 4), "budget differs"),
        (
            lambda report: report.__setitem__("data_fingerprint", "sha256:" + "f" * 64),
            "data_fingerprint",
        ),
        (
            lambda report: report.__setitem__("checkpoint_sha256", "sha256:" + "f" * 64),
            "checkpoint SHA",
        ),
        (
            lambda report: report.__setitem__("optimizer_restored", True),
            "fresh optimizer",
        ),
    ],
)
def test_authenticated_report_rejects_forged_budget_data_checkpoint_or_optimizer(
    tmp_path: Path, mutation, match: str
) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    mutation(report)
    _rebind_report(paths, report)
    with pytest.raises(ExportError, match=match):
        _export(paths)


def test_registered_recipe_rejects_forged_resolved_train_config(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    fields = report["resolved_train_config"]["fields"]
    fields["soft_target_weight"] = 0.1
    forged = TrainConfig(**fields)
    report["resolved_train_config"] = forged.canonical_payload()
    report["config_hash"] = forged.config_hash()
    report["full_config_hash"] = forged.full_config_hash()
    report["soft_target_weight"] = 0.1
    _rebind_report(paths, report)
    with pytest.raises(ExportError, match="registered recipe"):
        _export(paths)


def test_entity_graph_report_requires_null_graph_tokens_telemetry(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    assert report["resolved_train_config"]["fields"]["graph_tokens"] == 32
    assert report["graph_tokens"] is None
    report["graph_tokens"] = 32
    _rebind_report(paths, report)
    with pytest.raises(ExportError, match="entity_graph.*graph_tokens.*must be null"):
        _export(paths)


def test_xdim_graph_report_requires_resolved_graph_token_count(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    fields = report["resolved_train_config"]["fields"]
    fields["arch"] = "xdim_graph"
    forged = TrainConfig(**fields)
    report["resolved_train_config"] = forged.canonical_payload()
    report["config_hash"] = forged.config_hash()
    report["full_config_hash"] = forged.full_config_hash()
    report["arch"] = "xdim_graph"
    report["graph_tokens"] = None
    _rebind_report(paths, report)
    with pytest.raises(ExportError, match="xdim_graph.*graph_tokens differs"):
        _export(paths)


def test_registered_recipe_rejects_forged_freeze_modules(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    fields = report["resolved_train_config"]["fields"]
    fields["freeze_modules"] = "state_encoder"
    forged = TrainConfig(**fields)
    report["resolved_train_config"] = forged.canonical_payload()
    report["config_hash"] = forged.config_hash()
    report["full_config_hash"] = forged.full_config_hash()
    report["freeze_modules"] = "state_encoder"
    _rebind_report(paths, report)
    with pytest.raises(ExportError, match="freeze_modules differs from registered recipe"):
        _export(paths)


@pytest.mark.parametrize("mutation", ["omitted", "extra"])
def test_training_recipe_key_coverage_is_exact(tmp_path: Path, mutation: str) -> None:
    paths = _fixtures(tmp_path)
    experiment = json.loads(paths["experiment_config_path"].read_text())
    if mutation == "omitted":
        experiment["training_recipe"].pop("freeze_modules")
    else:
        experiment["training_recipe"]["unregistered_knob"] = False
    _rebind_experiment(paths, experiment)
    with pytest.raises(ExportError, match="incomplete or unexpected"):
        _export(paths)


def test_rejects_forged_registered_executing_source_hash(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    experiment = json.loads(paths["experiment_config_path"].read_text())
    experiment["frozen_inputs"]["executing_learner_source_sha256"][
        "tools/train_bc.py"
    ] = "f" * 64
    _rebind_experiment(paths, experiment)
    with pytest.raises(ExportError, match="report source SHA differs for tools/train_bc.py"):
        _export(paths)


def test_rejects_forged_reported_executing_source_hash(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    report["rnd_executing_learner_source_sha256"]["tools/train_bc.py"] = "f" * 64
    _rebind_report(paths, report)
    with pytest.raises(ExportError, match="report source SHA differs for tools/train_bc.py"):
        _export(paths)


def test_rejects_live_executing_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixtures(tmp_path)
    original = holdout_exporter._sha256_file
    target = (
        Path(__file__).resolve().parents[1] / "tools/train_bc.py"
    ).resolve()

    def drifted(path: Path) -> str:
        return "f" * 64 if Path(path).resolve() == target else original(path)

    monkeypatch.setattr(holdout_exporter, "_sha256_file", drifted)
    with pytest.raises(ExportError, match="live executing learner source SHA differs"):
        _export(paths)


def test_rejects_missing_reported_a1_relocation_role(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    report["rnd_a1_artifact_relocation"]["files"].pop("contract_lock")
    _rebind_report(paths, report)
    with pytest.raises(ExportError, match="relocation differs from registration"):
        _export(paths)


def test_rejects_tampered_relocated_a1_artifact(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    directory = Path(report["resolved_train_config"]["fields"]["rnd_a1_artifact_dir"])
    filename = report["rnd_a1_artifact_relocation"]["files"]["validation_manifest"][
        "filename"
    ]
    (directory / filename).write_bytes(b"tampered relocation")
    with pytest.raises(ExportError, match="physical file hash mismatch"):
        _export(paths)


def test_rejects_extra_file_in_a1_relocation_directory(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report_path = Path(json.loads(paths["run_manifest"].read_text())["training_report"]["path"])
    report = json.loads(report_path.read_text())
    directory = Path(report["resolved_train_config"]["fields"]["rnd_a1_artifact_dir"])
    (directory / "extra.json").write_text("{}")
    with pytest.raises(ExportError, match="directory file set is not exact"):
        _export(paths)


def test_refuses_memmap_payload_drift_before_opening_corpus(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    (paths["corpus_dir"] / "row_offsets.dat").write_bytes(b"tampered payload")
    opened = False

    def corpus_loader(_path):
        nonlocal opened
        opened = True
        return FakeCorpus()

    with pytest.raises(ExportError, match="payload inventory validation failed"):
        _export(paths, corpus_loader=corpus_loader)
    assert opened is False


@pytest.mark.parametrize("mutation", ["omission", "extra"])
def test_canonical_inventory_rejects_omitted_or_extra_payload(
    tmp_path: Path, mutation: str
) -> None:
    paths = _fixtures(tmp_path)
    meta_path = paths["corpus_dir"] / "corpus_meta.json"
    meta = json.loads(meta_path.read_text())
    if mutation == "omission":
        meta["payload_inventory"] = [
            row for row in meta["payload_inventory"] if row["filename"] != "dummy.dat"
        ]
    else:
        extra = paths["corpus_dir"] / "extra.dat"
        extra.write_bytes(b"unregistered payload")
        meta["payload_inventory"].append(
            {
                "filename": "extra.dat",
                "size_bytes": extra.stat().st_size,
                "sha256": "sha256:" + _sha(extra),
            }
        )
        meta["payload_inventory"].sort(key=lambda row: row["filename"])
    meta["payload_inventory_sha256"] = "sha256:" + _canonical_sha(
        meta["payload_inventory"]
    )
    _write(meta_path, meta)

    with pytest.raises(ExportError, match="filenames differ"):
        _export(paths)


def test_refuses_unattested_public_training_and_overwrite(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)

    class UnmaskedPolicy(FakePolicy):
        trained_with_masked_hidden_info = False

    with pytest.raises(ExportError, match="public-masked training"):
        _export(paths, policy_loader=lambda *_args, **_kwargs: UnmaskedPolicy())

    paths["output"].write_text("existing\n")
    with pytest.raises(ExportError, match="refusing to overwrite"):
        _export(paths)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda corpus: corpus.data["target_policy_mask"].__setitem__((0, 0), False),
            "full live-action coverage",
        ),
        (
            lambda corpus: corpus.data["target_policy"].__setitem__((0, 0), 0.5),
            "normalize to one",
        ),
    ],
)
def test_refuses_incomplete_or_unnormalized_soft_targets(
    tmp_path: Path, mutation, match: str
) -> None:
    paths = _fixtures(tmp_path)

    def corpus_loader(_path):
        corpus = FakeCorpus(_path)
        mutation(corpus)
        return corpus

    with pytest.raises(ExportError, match=match):
        _export(paths, corpus_loader=corpus_loader)
