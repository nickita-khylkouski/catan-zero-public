from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from catan_zero.rl.pipeline_configs import TrainConfig
from tools import a1_mixed_architecture_probe as probe
from tools.train_bc import build_parser as build_train_parser


def _sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _corpus(root: Path, name: str) -> tuple[Path, Path]:
    corpus = root / name
    corpus.mkdir()
    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": 10,
        "legal_width": 4,
        "flat_count": 20,
        "columns": {},
        "payload_inventory_sha256": "sha256:" + name[1] * 64,
    }
    (corpus / "corpus_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    validation = root / f"{name}.validation.json"
    validation.write_text("{}\n", encoding="utf-8")
    return corpus.resolve(), validation.resolve()


def _audit(path: Path, corpora: list[Path], *, runnable: bool = True) -> Path:
    rows = []
    for corpus in corpora:
        rows.append(
            {
                "corpus_dir": str(corpus),
                "legal_action_targets": {
                    "out_of_range_target_rows": 0,
                    "invalid_legal_action_ids": 0,
                    "search_active_rows_with_any_target": 100,
                },
                "graph_incidence": {"out_of_range_ids": 0},
                "event_targets": {"masked_events": 0, "events_with_any_target": 0},
                "viability": {
                    "action_target_gather": runnable,
                    "action_cross_attention": True,
                    "graph_relational_trunk": True,
                    "event_target_relations": False,
                },
            }
        )
    payload = {
        "schema_version": "memmap-architecture-target-audit-bundle-v1",
        "audits": rows,
        "verdict": {
            "architecture_action_probe_runnable": runnable,
            "requires_generator_changes_for_action_probe": not runnable,
            "event_relation_probe_runnable": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _argv(tmp_path: Path, *, runnable: bool = True, max_steps: int = 1000) -> list[str]:
    n256, n256_validation = _corpus(tmp_path, "n256")
    n128, n128_validation = _corpus(tmp_path, "n128")
    initialization = tmp_path / "init.pt"
    initialization.write_bytes(b"shared warm-start source")
    gather = tmp_path / "gather.pt"
    torch.save(
        {
            "config": {
                "__config_dataclass__": "EntityGraphConfig",
                "__config_schema__": 1,
                "fields": {
                    "state_trunk": "transformer",
                    "action_target_gather": True,
                    "action_cross_attention_layers": 0,
                    "edge_policy_head": False,
                    "value_attention_pool": False,
                },
            },
            "upgrade_provenance": {
                "schema_version": "entity-graph-upgrade-v1",
                "source_checkpoint_sha256": hashlib.sha256(
                    initialization.read_bytes()
                ).hexdigest(),
                "flags": {"action_target_gather": True},
                "forward_max_diff": 0.0,
                "forward_identical_at_init": True,
            },
        },
        gather,
    )
    audit = _audit(tmp_path / "audit.json", [n256, n128], runnable=runnable)
    return [
        "--lr",
        "1.2e-4",
        "--max-steps",
        str(max_steps),
        "--n256-corpus",
        str(n256),
        "--n256-validation",
        str(n256_validation),
        "--n128-corpus",
        str(n128),
        "--n128-validation",
        str(n128_validation),
        "--initialization-checkpoint",
        str(initialization),
        "--gather-checkpoint",
        str(gather),
        "--architecture-audit",
        str(audit),
        "--output-root",
        str(tmp_path / "out"),
    ]


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_prepare_seals_matched_action_only_architecture_ab_without_launch(
    tmp_path, monkeypatch
):
    called = False

    def refuse(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("preparation launched training")

    monkeypatch.setattr(probe, "_launch", refuse)
    probe.main(_argv(tmp_path))
    assert called is False
    manifest = json.loads((tmp_path / "out/experiment.manifest.json").read_text())
    assert manifest["diagnostic_only"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["event_path"]["included"] is False
    assert manifest["only_declared_arm_delta"] == "zero-init action_target_gather"
    assert manifest["topology"] == {
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "global_row_shuffle": True,
        "no_copy": True,
    }

    baseline = manifest["arms"]["baseline"]
    treatment = manifest["arms"]["target_gather"]
    assert baseline["training_recipe"] == treatment["training_recipe"]
    assert baseline["initialization"] != treatment["initialization"]
    assert baseline["initialization_source"] == treatment["initialization_source"]
    assert treatment["upgrade_evidence"]["forward_identical_at_init"] is True
    assert baseline["descriptor"] == treatment["descriptor"]
    assert baseline["architecture_audit"] == treatment["architecture_audit"]
    assert baseline["architecture"]["entity_state_trunk"] == "transformer"
    assert baseline["architecture"]["effective_action_target_gather"] is False
    assert treatment["architecture"] == {
        "entity_state_trunk": "transformer",
        "relational_block_pattern": "",
        "relational_ff_size": 0,
        "relational_bases": 4,
        "relational_action_cross_layers": 1,
        "effective_action_target_gather": True,
        "effective_action_cross_attention_layers": 0,
        "effective_graph_relational_encoding": False,
        "effective_edge_policy_head": False,
    }
    assert manifest["manifest_sha256"] == _sha(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def test_commands_bind_same_source_optimizer_split_steps_and_topology(tmp_path):
    args = probe.build_parser().parse_args(_argv(tmp_path, max_steps=777))
    manifest, _ = probe.prepare(args)
    commands = [manifest["arms"][arm]["command"] for arm in probe.ARMS]
    for command in commands:
        assert "--nproc-per-node=8" in command
        assert _option(command, "--batch-size") == "512"
        assert _option(command, "--max-steps") == "777"
        assert _option(command, "--optimizer") == "adam"
        assert _option(command, "--lr") == "0.00012"
        assert _option(command, "--seed") == "1"
        assert _option(command, "--validation-max-samples") == "0"
        assert "--init-checkpoint" in command
        assert "--no-resume-optimizer" in command
        assert "--no-fused-optimizer" in command
        assert "--no-relational-edge-policy-head" in command
        assert "--symmetry-augment-events" not in command
    assert _option(commands[0], "--entity-state-trunk") == "transformer"
    assert _option(commands[1], "--entity-state-trunk") == "transformer"
    assert "--relational-block-pattern" not in commands[1]
    assert _option(commands[1], "--relational-action-cross-layers") == "1"


def test_nonviable_or_eventful_audit_is_refused(tmp_path):
    args = probe.build_parser().parse_args(_argv(tmp_path, runnable=False))
    with pytest.raises(SystemExit, match="does not authorize"):
        probe.prepare(args)


@pytest.mark.parametrize("steps", [0, -1])
def test_nonpositive_step_budget_is_refused(tmp_path, steps):
    args = probe.build_parser().parse_args(_argv(tmp_path, max_steps=steps))
    with pytest.raises(SystemExit, match="max-steps must be positive"):
        probe.prepare(args)


def test_prepare_preserves_lexical_virtualenv_python(tmp_path: Path) -> None:
    lexical = tmp_path / "venv-python"
    lexical.symlink_to(Path(sys.executable))
    args = probe.build_parser().parse_args(
        [*_argv(tmp_path), "--python", str(lexical)]
    )
    manifest, _ = probe.prepare(args)
    assert all(arm["command"][0] == str(lexical) for arm in manifest["arms"].values())
    assert str(lexical) != str(lexical.resolve())


def test_existing_seal_is_idempotent_and_recipe_drift_fails_closed(tmp_path):
    args = probe.build_parser().parse_args(_argv(tmp_path))
    first, path = probe.prepare(args)
    second, second_path = probe.prepare(args)
    assert first == second
    assert path == second_path
    descriptor = tmp_path / "out/memmap_composite.json"
    descriptor.chmod(0o644)
    payload = json.loads(descriptor.read_text())
    payload["learner_recipe_overrides"]["loser_sample_weight"] = 0.3
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="prepared artifact drift"):
        probe.prepare(args)


def test_relational_edge_head_switch_preserves_historical_default():
    parser = build_train_parser()
    assert parser.get_default("relational_edge_policy_head") is True
    assert TrainConfig().relational_edge_policy_head is True
    parsed = parser.parse_args(
        [
            "--data",
            "data",
            "--checkpoint",
            "checkpoint.pt",
            "--report",
            "report.json",
            "--no-relational-edge-policy-head",
        ]
    )
    assert parsed.relational_edge_policy_head is False
