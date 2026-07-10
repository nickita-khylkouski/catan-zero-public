from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import catan_zero.rl.pipeline_configs as pipeline_configs_module
import tools.train_bc as train_bc_module
from catan_zero.rl.pipeline_configs import TrainConfig
from tools.train_bc import (
    RND_EXECUTING_LEARNER_SOURCE_MODULES,
    _checkpoint_config_mismatches,
    _rnd_executing_learner_source_sha256,
    build_parser,
)


TOPOLOGY_EXECUTION_FIELDS = (
    "topology_adapter_layers",
    "topology_adapter_width",
    "topology_adapter_bases",
    "topology_adapter_kind",
    "topology_adapter_heads",
    "topology_adapter_share_weights",
    "topology_adapter_edge_control",
)


def test_topology_v2_cli_is_explicit_and_defaults_to_no_behavior_change() -> None:
    parser = build_parser()
    defaults = parser.parse_args(
        ["--data", "data", "--checkpoint", "checkpoint.pt", "--report", "report.json"]
    )
    assert defaults.topology_adapter_layers == ""
    assert defaults.topology_adapter_kind == "basis_mean_v1"
    assert defaults.topology_adapter_heads == 4
    assert defaults.topology_adapter_share_weights is False
    assert defaults.topology_adapter_edge_control == "true_topology"
    assert defaults.rnd_a1_artifact_dir == ""

    configured = parser.parse_args(
        [
            "--data",
            "data",
            "--checkpoint",
            "checkpoint.pt",
            "--report",
            "report.json",
            "--topology-adapter-layers",
            "2,4",
            "--topology-adapter-kind",
            "local_attention_v2",
            "--topology-adapter-width",
            "192",
            "--topology-adapter-heads",
            "4",
            "--topology-adapter-share-weights",
            "--topology-adapter-edge-control",
            "self_message",
        ]
    )
    assert configured.topology_adapter_layers == "2,4"
    assert configured.topology_adapter_kind == "local_attention_v2"
    assert configured.topology_adapter_width == 192
    assert configured.topology_adapter_heads == 4
    assert configured.topology_adapter_share_weights is True
    assert configured.topology_adapter_edge_control == "self_message"
    effective_config = TrainConfig.from_namespace(configured)
    for field in TOPOLOGY_EXECUTION_FIELDS:
        assert getattr(effective_config, field) == getattr(configured, field)


def test_topology_execution_contract_is_science_hashed() -> None:
    base = TrainConfig()
    overrides = {
        "topology_adapter_layers": "2,4",
        "topology_adapter_width": 192,
        "topology_adapter_bases": 8,
        "topology_adapter_kind": "local_attention_v2",
        "topology_adapter_heads": 8,
        "topology_adapter_share_weights": True,
        "topology_adapter_edge_control": "self_message",
    }

    assert all(hasattr(base, field) for field in TOPOLOGY_EXECUTION_FIELDS)
    hashes = {
        dataclasses.replace(base, **{field: value}).config_hash()
        for field, value in overrides.items()
    }
    assert len(hashes) == len(TOPOLOGY_EXECUTION_FIELDS)
    assert base.config_hash() not in hashes


@pytest.mark.parametrize(
    ("field", "value"),
    [("grad_accum_steps", 4), ("amp", "bf16"), ("fused_optimizer", True)],
)
def test_effective_update_execution_is_science_hashed(
    field: str, value: object
) -> None:
    base = TrainConfig()
    changed = dataclasses.replace(base, **{field: value})
    assert changed.config_hash() != base.config_hash()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_game_seed_manifest", "validation.json"),
        ("a1_memmap_payload_inventory_sha256", "sha256:" + "a" * 64),
        ("rnd_a1_artifact_dir", "/isolated/a1-artifacts"),
    ],
)
def test_training_input_bindings_are_science_hashed(
    field: str, value: object
) -> None:
    base = TrainConfig()
    changed = dataclasses.replace(base, **{field: value})
    assert changed.config_hash() != base.config_hash()


def test_a1_learner_override_is_explicit_hashed_and_default_off() -> None:
    parser = build_parser()
    defaults = parser.parse_args(
        ["--data", "data", "--checkpoint", "checkpoint.pt", "--report", "report.json"]
    )
    enabled = parser.parse_args(
        [
            "--data",
            "data",
            "--checkpoint",
            "checkpoint.pt",
            "--report",
            "report.json",
            "--rnd-allow-a1-learner-override",
        ]
    )
    assert defaults.rnd_allow_a1_learner_override is False
    assert enabled.rnd_allow_a1_learner_override is True
    assert TrainConfig.from_namespace(defaults).config_hash() != TrainConfig.from_namespace(
        enabled
    ).config_hash()


def test_rnd_override_binds_exact_imported_learner_sources() -> None:
    assert _rnd_executing_learner_source_sha256(enabled=False) is None
    source_hashes = _rnd_executing_learner_source_sha256(enabled=True)
    assert source_hashes is not None
    assert tuple(source_hashes) == tuple(
        relative for relative, _module in RND_EXECUTING_LEARNER_SOURCE_MODULES
    )
    root = Path(__file__).resolve().parents[1]
    assert source_hashes == {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative, _module in RND_EXECUTING_LEARNER_SOURCE_MODULES
    }


def test_rnd_override_rejects_substituted_imported_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    substituted = tmp_path / "pipeline_configs.py"
    substituted.write_text("# substituted module\n", encoding="utf-8")
    monkeypatch.setattr(pipeline_configs_module, "__file__", str(substituted))
    with pytest.raises(SystemExit, match="outside the active checkout"):
        _rnd_executing_learner_source_sha256(enabled=True)


def test_rnd_source_binding_occurs_after_a1_corpus_proofs() -> None:
    source = inspect.getsource(train_bc_module.main)
    proof = source.index("_validate_a1_corpus_artifacts_and_seeds(")
    source_binding = source.index("_rnd_executing_learner_source_sha256(enabled=True)")
    assert proof < source_binding
    assert '"a1_lock_scope": "data_generation_provenance_only"' in source


@pytest.mark.parametrize(
    ("field", "checkpoint_value", "cli_value"),
    [
        ("topology_adapter_layers", "1,3", "2,4"),
        ("topology_adapter_width", 128, 192),
        ("topology_adapter_bases", 2, 4),
        ("topology_adapter_kind", "basis_mean_v1", "local_attention_v2"),
        ("topology_adapter_heads", 2, 4),
        ("topology_adapter_share_weights", False, True),
        ("topology_adapter_edge_control", "self_message", "true_topology"),
    ],
)
def test_topology_execution_contract_is_checkpoint_preflighted(
    field: str, checkpoint_value: object, cli_value: object
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--data", "data", "--checkpoint", "checkpoint.pt", "--report", "report.json"]
    )
    args.arch = "entity_graph"
    setattr(args, field, cli_value)
    checkpoint_config = SimpleNamespace(**{field: checkpoint_value})

    mismatches = _checkpoint_config_mismatches(
        policy_type="entity_graph", config=checkpoint_config, args=args
    )
    assert any(item.startswith(f"{field} checkpoint=") for item in mismatches)


def test_training_report_records_complete_topology_execution_contract() -> None:
    train_bc_path = Path(__file__).resolve().parents[1] / "tools" / "train_bc.py"
    tree = ast.parse(train_bc_path.read_text(), filename=str(train_bc_path))
    report_keys: set[str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "report"
            for target in node.targets
        ) or not isinstance(node.value, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "config_hash" in keys and "steps_completed" in keys:
            report_keys = keys
            break

    assert report_keys is not None
    assert set(TOPOLOGY_EXECUTION_FIELDS) <= report_keys
    assert {
        "full_config_hash",
        "resolved_train_config",
        "grad_accum_steps",
        "global_batch_size",
        "sample_presentations",
        "checkpoint_sha256",
        "optimizer_sidecar",
        "optimizer_sidecar_sha256",
        "rnd_allow_a1_learner_override",
        "rnd_executing_learner_source_sha256",
        "a1_contract_provenance_scope",
        "rnd_a1_artifact_relocation",
    } <= report_keys
