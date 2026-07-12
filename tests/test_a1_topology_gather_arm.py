from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools import a1_topology_gather_arm as arm


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "f7.pt"
    upgraded = tmp_path / "f7-gather.pt"
    base_model = {
        "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "policy.weight": torch.ones(2, 2),
    }
    base_config = {"state_trunk": "transformer", "action_size": 567,
                   "static_action_feature_size": 1}
    torch.save({"config": {"fields": base_config},
                "model": base_model, "mask_hidden_info": True}, source)
    model = dict(base_model)
    model.update({
        "target_gather_proj.0.weight": torch.ones(3),
        "target_gather_proj.0.bias": torch.zeros(3),
        "target_gather_proj.1.weight": torch.zeros(3, 3),
        "target_gather_proj.1.bias": torch.zeros(3),
    })
    torch.save({
        "config": {"fields": {**base_config, "action_target_gather": True,
            "action_cross_attention_layers": 0, "edge_policy_head": False,
            "value_attention_pool": False,
        }},
        "model": model,
        "mask_hidden_info": True,
        "upgrade_provenance": {
            "schema_version": "entity-graph-upgrade-v1",
            "source_checkpoint_sha256": arm.corrected._file_sha(source).removeprefix("sha256:"),
            "flags": {"action_target_gather": True},
            "initialization_seed": 1,
            "trained_value_readouts_added": [],
            "forward_max_diff": 0.0,
            "forward_identical_at_init": True,
        },
    }, upgraded)
    return source, upgraded


def _source_manifest(tmp_path: Path, source: Path, descriptor: Path,
                     validation: Path) -> Path:
    effective = {"lr": 3e-5, "soft_target_weight": 1.0}
    command = [
        "python", "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=8", "tools/train_bc.py",
        "--data", str(descriptor.resolve()),
        "--validation-game-sentinel-manifest", str(validation.resolve()),
        "--init-checkpoint", str(source.resolve()),
        "--checkpoint", str(tmp_path / "source-candidate.pt"),
        "--report", str(tmp_path / "source-report.json"),
        "--a1-learner-ablation-id", "corrected-anchor-K3",
        "--a1-effective-learner-recipe-json", arm.corrected._canonical(effective).decode(),
        "--a1-effective-learner-recipe-sha256", arm.corrected._digest(effective),
        "--no-resume-optimizer", "--fsdp", "--mask-hidden-info",
    ]
    recipe = {
        "world_size": 8, "local_batch_size": 512, "global_batch_size": 4096,
        "steps": 1024, "base_value_row_dose": 4_194_304,
        "policy_aux_active_batch_size_per_rank": 128,
        "policy_aux_active_row_dose": 1_048_576,
        "replay_supervised_policy": False, "replay_supervised_value": False,
        "replay_forward_kl_weight": 0.006, "soft_target_weight": 1.0,
        "fresh_optimizer": True, "independent_f7_initialization": True,
    }
    payload = {
        "schema_version": arm.SOURCE_SCHEMA,
        "diagnostic_only": True, "promotion_eligible": False,
        "launch_authorized": False, "diagnostic_execution_authorized": True,
        "launch_interface_present": "tools/a1_corrected_policy_arm_execute.py --go",
        "recipe": recipe, "recipe_sha256": arm.corrected._digest(recipe),
        "initialization": arm.corrected._file_ref(source),
        "descriptor": arm.corrected._file_ref(descriptor),
        "validation_sentinel": arm.corrected._file_ref(validation),
        "validation_sentinel_selection_sha256": "sha256:selection",
        "command": command, "command_sha256": arm.corrected._digest(command),
    }
    payload["manifest_sha256"] = arm.corrected._digest(payload)
    return _write_json(tmp_path / "corrected.manifest.json", payload)


def _audit(tmp_path: Path, corpora: list[Path]) -> Path:
    rows = []
    for index, corpus in enumerate(corpora, start=1):
        rows.append({
            "corpus_dir": str(corpus.resolve()),
            "legal_action_targets": {
                "actions": 1000 * index, "actions_with_any_target": 400 * index,
                "target_coverage": 0.4, "rows_with_any_target": 200 * index,
                "row_target_coverage": 0.2,
                "search_active_rows_with_any_target": 150 * index,
                "chosen_actions_with_any_target": 100 * index,
                "invalid_legal_action_ids": 0, "out_of_range_target_rows": 0,
            },
            "graph_incidence": {"out_of_range_ids": 0},
            "viability": {"action_target_gather": True},
        })
    return _write_json(tmp_path / "audit.json", {
        "schema_version": "memmap-architecture-target-audit-bundle-v1",
        "audits": rows,
        "verdict": {"architecture_action_probe_runnable": True},
    })


def _args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source, gather = _checkpoints(tmp_path)
    descriptor = _write_json(tmp_path / "descriptor.json", {"schema_version": "memmap_composite_v2"})
    validation = _write_json(tmp_path / "validation.json", {"schema_version": "validation-v1"})
    corpora = [tmp_path / name for name in ("n128", "n256", "replay")]
    for corpus in corpora:
        corpus.mkdir()
    manifest = _source_manifest(tmp_path, source, descriptor, validation)
    monkeypatch.setattr(arm.corrected, "_preflight_descriptor", lambda _path: ({
        "components": [{"corpus_dir": str(path.resolve())} for path in corpora]
    }, arm.corrected._file_ref(descriptor)))
    monkeypatch.setattr(arm, "_source_binding", lambda repo: {
        "repository_root": str(repo), "git_commit": "abc", "files": {}
    })
    return type("Args", (), {
        "source_manifest": manifest,
        "gather_checkpoint": gather,
        "architecture_audit": _audit(tmp_path, corpora),
        "output_root": tmp_path / "out",
        "repo": tmp_path,
    })()


def test_prepares_one_axis_gather_k3_without_launch(tmp_path, monkeypatch):
    manifest, path = arm.prepare(_args(tmp_path, monkeypatch))
    assert path.is_file()
    assert manifest["launch_authorized"] is False
    assert manifest["diagnostic_execution_authorized"] is False
    assert manifest["launch_interface_present"] is False
    assert manifest["only_declared_optimization_delta"] == "action_target_gather=true"
    assert manifest["matched_contract"]["dose_sampler_objective_operator_unchanged"] is True
    assert manifest["matched_contract"]["step0_network_outputs_bit_identical"] is True
    assert manifest["function_preserving_upgrade"]["shared_parameters_bit_identical"] is True
    assert manifest["function_preserving_upgrade"]["new_parameters"] == list(
        arm.EXPECTED_NEW_PARAMETERS
    )
    assert len(manifest["corpus_topology_target_coverage"]["components"]) == 3
    assert manifest["executor_compatibility"]["compatible_now"] is False
    assert "exact a1-corrected-policy-arm-manifest-v1" in (
        manifest["executor_compatibility"]["reason"]
    )
    command = manifest["command"]
    assert arm.corrected._option(command, "--a1-learner-ablation-id") == (
        "corrected-anchor-K3-topology-gather"
    )
    assert arm.corrected._option(command, "--init-checkpoint") == str(
        _args_checkpoint(manifest)
    )


def _args_checkpoint(manifest: dict) -> Path:
    return Path(manifest["initialization_treatment"]["path"])


def test_upgrade_refuses_any_shared_parameter_change(tmp_path):
    source, gather = _checkpoints(tmp_path)
    raw = torch.load(gather, map_location="cpu", weights_only=False)
    raw["model"]["policy.weight"][0, 0] = 7
    torch.save(raw, gather)
    with pytest.raises(arm.ArmError, match="shared f7 parameters changed"):
        arm._validate_upgrade(source, gather)


def test_upgrade_refuses_nonzero_residual_output(tmp_path):
    source, gather = _checkpoints(tmp_path)
    raw = torch.load(gather, map_location="cpu", weights_only=False)
    raw["model"]["target_gather_proj.1.weight"][0, 0] = 0.01
    torch.save(raw, gather)
    with pytest.raises(arm.ArmError, match="deterministic zeros"):
        arm._validate_upgrade(source, gather)


def test_upgrade_refuses_unrelated_effective_config_or_provenance_drift(tmp_path):
    source, gather = _checkpoints(tmp_path)
    raw = torch.load(gather, map_location="cpu", weights_only=False)
    raw["config"]["fields"]["dropout"] = 0.2
    torch.save(raw, gather)
    with pytest.raises(arm.ArmError, match="effective config delta"):
        arm._validate_upgrade(source, gather)

    source, gather = _checkpoints(tmp_path)
    raw = torch.load(gather, map_location="cpu", weights_only=False)
    raw["mask_hidden_info"] = False
    torch.save(raw, gather)
    with pytest.raises(arm.ArmError, match="source provenance"):
        arm._validate_upgrade(source, gather)


def test_coverage_refuses_zero_search_active_topology_rows(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    payload = json.loads(args.architecture_audit.read_text())
    payload["audits"][1]["legal_action_targets"]["search_active_rows_with_any_target"] = 0
    args.architecture_audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="learnable topology target coverage"):
        arm.prepare(args)


def test_source_manifest_refuses_recipe_drift(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    payload = json.loads(args.source_manifest.read_text())
    payload["recipe"]["base_value_row_dose"] += 1
    payload["recipe_sha256"] = arm.corrected._digest(payload["recipe"])
    payload["manifest_sha256"] = arm.corrected._digest(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    args.source_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="exact corrected anchor-only K3"):
        arm.prepare(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("diagnostic_execution_authorized", False),
        ("launch_interface_present", False),
        ("launch_interface_present", "some_other_executor.py --go"),
        ("promotion_eligible", True),
    ],
)
def test_source_requires_exact_finalized_diagnostic_executor_shape(
    tmp_path, monkeypatch, field, value
):
    args = _args(tmp_path, monkeypatch)
    payload = json.loads(args.source_manifest.read_text())
    payload[field] = value
    payload["manifest_sha256"] = arm.corrected._digest(
        {key: item for key, item in payload.items() if key != "manifest_sha256"}
    )
    args.source_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="exact diagnostic executor"):
        arm.prepare(args)
