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


def _temp_command(
    tmp_path: Path,
    *,
    trainer: Path,
    source: Path,
    descriptor: Path,
    validation: Path,
    inventories: list[str],
) -> list[str]:
    command = [
        "python", "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=8", str(trainer.resolve()),
        "--data", str(descriptor.resolve()), "--data-format", "memmap",
        "--init-checkpoint", str(source.resolve()),
        "--arch", "entity_graph", "--hidden-size", "640",
        "--graph-layers", "6", "--attention-heads", "8",
        "--graph-dropout", "0.05", "--entity-state-trunk", "transformer",
        "--track", "2p_no_trade", "--vps-to-win", "10",
        "--graph-history-features", "--mask-hidden-info", "--epochs", "1",
        "--max-steps", "1024", "--batch-size", "512",
        "--grad-accum-steps", "1", "--seed", "1",
        "--training-rng-rank-offset", "--optimizer", "adam",
        "--no-resume-optimizer", "--no-fused-optimizer", "--lr", "3e-05",
        "--lr-warmup-steps", "100", "--lr-schedule", "flat",
        "--weight-decay", "0.0", "--value-lr-mult", "0.3",
        "--action-module-lr-mult", "1.0", "--policy-loss-weight", "1.0",
        "--soft-target-source", "policy", "--soft-target-weight", "0.9",
        "--soft-target-min-legal-coverage", "0.5",
        "--value-loss-weight", "0.25", "--value-target-lambda", "1.0",
        "--value-head-type", "mse", "--truncated-vp-margin-value-weight", "0.0",
        "--final-vp-loss-weight", "0.0", "--q-loss-weight", "0.0",
        "--policy-kl-anchor-weight", "0.0", "--policy-kl-anchor-direction", "forward",
        "--forced-action-weight", "0.0", "--forced-row-value-weight", "1.0",
        "--winner-sample-weight", "1.0", "--loser-sample-weight", "1.0",
        "--validation-max-samples", "0", "--skip-teacher-quality-gate",
        "--trust-curated-data-quality", "--data-loader-workers", "4",
        "--data-loader-prefetch", "4",
    ]
    for inventory in inventories:
        command.extend((arm.corrected.EVENT_HISTORY_ACK_FLAG, inventory))
    command.extend((
        arm.corrected.EVENT_HISTORY_CROP_FLAG,
        "--validation-game-sentinel-manifest", str(validation.resolve()),
        "--checkpoint", str(tmp_path / "source-candidate.pt"),
        "--report", str(tmp_path / "source-report.json"),
    ))
    return command


def _source_manifest(
    tmp_path: Path,
    source: Path,
    descriptor: Path,
    validation: Path,
    corpora: list[Path],
) -> tuple[Path, list[str], list[str]]:
    source_repo = tmp_path / "cleaned-source-checkout"
    trainer = source_repo / "tools/train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# historical full-dose TEMP trainer\n", encoding="utf-8")
    inventories = ["sha256:" + str(index) * 64 for index in range(1, 4)]
    bindings = []
    for component_id, corpus, inventory in zip(
        arm.production_temp.COMPONENT_IDS, corpora, inventories, strict=True
    ):
        meta = _write_json(corpus / "corpus_meta.json", {"schema_version": "corpus-v1"})
        holdout = _write_json(corpus / "validation.json", {"schema_version": "holdout-v1"})
        bindings.append({
            "component_id": component_id,
            "corpus_meta": arm.corrected._file_ref(meta),
            "payload_inventory_sha256": inventory,
            "validation_manifest": arm.corrected._file_ref(holdout),
        })
    command = _temp_command(
        tmp_path,
        trainer=trainer,
        source=source,
        descriptor=descriptor,
        validation=validation,
        inventories=inventories,
    )
    payload = {
        "schema_version": "a1-production-temperature-replication-v1",
        "diagnostic_only": False, "production_eligible": True,
        "launch_authorized": True,
        "source_descriptor": arm.corrected._file_ref(descriptor),
        "validation_sentinel": arm.corrected._file_ref(validation),
        "f7_parent": arm.corrected._file_ref(source),
        "component_bindings": bindings,
        "stored_policy_component_temperatures": arm.production_temp.COMPONENT_TEMPERATURES,
        "event_history_training_contract": {
            "public_observation_masked": True,
            "graph_history_features": True,
            "payload_inventory_acknowledgements": inventories,
        },
        "selected_dose": {
            "optimizer_steps": 1024, "world_size": 8,
            "per_rank_batch_size": 512, "global_samples": 4_194_304,
            "optimizer": "fresh_adam", "lr": 3e-5,
            "training_rng_rank_offset": True,
        },
        "command": command, "command_sha256": arm.corrected._digest(command),
    }
    payload["manifest_sha256"] = arm.corrected._digest(payload)
    return _write_json(tmp_path / "temp.manifest.json", payload), command, inventories


def _selected_geometry(
    tmp_path: Path,
    *,
    source_command: list[str],
    source: Path,
    descriptor: Path,
    validation: Path,
) -> tuple[Path, Path]:
    repo = tmp_path / "geometry-checkout"
    trainer = repo / "tools/train_bc.py"
    probe = repo / "tools/a1_b200_microbatch_quality.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# selected short-dose trainer\n", encoding="utf-8")
    probe.write_text("# geometry planner\n", encoding="utf-8")
    run_dir = tmp_path / "geometry" / arm.GEOMETRY_RUN_ID
    report_path = run_dir / "train.report.json"
    command = arm._geometry_expected_command(
        source_command,
        trainer=trainer.resolve(),
        checkpoint=str(run_dir / "candidate.pt"),
        report=str(report_path),
    )
    runtime = {
        "repository_root": str(repo.resolve()),
        "repository_commit": "geometry-test-head",
        "trainer": str(trainer.resolve()),
        "trainer_sha256": arm.corrected._file_sha(trainer),
        "quality_probe": str(probe.resolve()),
        "quality_probe_sha256": arm.corrected._file_sha(probe),
    }
    run = {
        "run_id": arm.GEOMETRY_RUN_ID,
        "world_size": 8, "local_batch_size": 512, "global_batch_size": 4096,
        "grad_accum_steps": 1, "max_steps": 128, "planned_samples": 524_288,
        "warmup_samples": 409_600, "lr_warmup_steps": 100,
        "gpu_ids": list(range(8)), "run_dir": str(run_dir),
        "command": command, "command_sha256": arm.corrected._digest(command),
    }
    plan = {
        "schema_version": arm.GEOMETRY_SCHEMA,
        "diagnostic_only": True, "promotion_eligible": False,
        "launch_authorized": True,
        "inputs": {
            "data": str(descriptor.resolve()),
            "data_sha256": arm.corrected._file_sha(descriptor),
            "init_checkpoint": str(source.resolve()),
            "init_checkpoint_sha256": arm.corrected._file_sha(source),
        },
        "matched_invariants": {
            "global_batch_size": 4096, "lr": 3e-5, "lr_schedule": "flat",
            "lr_warmup_steps": 100, "warmup_samples": 409_600,
            "optimizer_steps": 128, "planned_samples": 524_288, "seed": 1,
        },
        "only_intended_drift": ["world_size", "batch_size", "gpu_ids"],
        "runtime": runtime, "runs": [run],
    }
    plan["plan_sha256"] = arm.corrected._digest(plan)
    plan_path = _write_json(tmp_path / "geometry-plan.json", plan)
    run_dir.mkdir(parents=True)
    report = {
        "world_size": 8, "batch_size": 512, "effective_global_batch_size": 4096,
        "max_steps": 128, "steps_completed": 128,
        "base_training_row_draws": 524_288, "total_training_row_draws": 524_288,
        "optimizer": "adam", "resume_optimizer": False,
        "optimizer_restored": False, "lr": 3e-5, "lr_schedule": "flat",
        "lr_warmup_steps": 100, "value_lr_mult": 0.3,
        "action_module_lr_mult": 1.0, "soft_target_weight": 0.9,
        "soft_target_temperature": 0.7, "value_loss_weight": 0.25,
        "value_target_lambda": 1.0, "value_head_type": "mse",
        "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
        "policy_kl_anchor_weight": 0.0, "q_loss_weight": 0.0,
        "mask_hidden_info": True, "graph_history_features": True,
        "training_rng_rank_offset": True, "diagnostic_only": True,
        "promotion_eligible": False, "freeze_modules": "",
        "require_only_trainable_prefixes": "",
        "init_checkpoint_sha256": arm.corrected._file_sha(source),
        "data": str(descriptor.resolve()),
        "input_validation_game_sentinel_manifest": str(validation.resolve()),
        "checkpoint": str(run_dir / "candidate.pt"),
    }
    return plan_path, _write_json(report_path, report)


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
    monkeypatch.setattr(
        arm.production_temp, "F7_SHA256", arm.corrected._file_sha(source)
    )
    descriptor = _write_json(tmp_path / "descriptor.json", {"schema_version": "memmap_composite_v2"})
    validation = _write_json(tmp_path / "validation.json", {"schema_version": "validation-v1"})
    corpora = [tmp_path / name for name in ("n128", "n256", "replay")]
    for corpus in corpora:
        corpus.mkdir()
    manifest, source_command, inventories = _source_manifest(
        tmp_path, source, descriptor, validation, corpora
    )
    plan, report = _selected_geometry(
        tmp_path,
        source_command=source_command,
        source=source,
        descriptor=descriptor,
        validation=validation,
    )
    monkeypatch.setattr(arm.corrected, "_preflight_descriptor", lambda _path: ({
        "components": [
            {
                "component_id": component_id,
                "corpus_dir": str(path.resolve()),
                "payload_inventory_sha256": inventories[index - 1],
            }
            for index, (component_id, path) in enumerate(zip(
                ("n128_current", "n256_current", "gen3_replay"), corpora
            ), start=1)
        ],
        "policy_distillation_component_ids": [
            "n128_current", "n256_current", "gen3_replay"
        ],
        "value_training_component_ids": [
            "n128_current", "n256_current", "gen3_replay"
        ],
        "policy_kl_anchor_component_ids": ["gen3_replay"],
        "stored_policy_component_temperatures": (
            arm.production_temp.COMPONENT_TEMPERATURES
        ),
    }, arm.corrected._file_ref(descriptor)))
    monkeypatch.setattr(arm, "_git_head", lambda _repo: "geometry-test-head")
    executor = tmp_path / arm.EXECUTOR_RELATIVE_PATH
    executor.parent.mkdir(exist_ok=True)
    executor.write_text("# sealed topology executor\n", encoding="utf-8")
    monkeypatch.setattr(arm, "_source_binding", lambda repo: {
        "repository_root": str(repo), "git_commit": "abc",
        "files": {
            arm.EXECUTOR_RELATIVE_PATH: arm.corrected._file_ref(executor),
        },
    })
    return type("Args", (), {
        "source_manifest": manifest,
        "selected_dose_plan": plan,
        "selected_dose_report": report,
        "gather_checkpoint": gather,
        # Ordering is not semantic, but every TEMP-supervised component must be
        # represented, including predecessor replay.
        "architecture_audit": _audit(tmp_path, list(reversed(corpora))),
        "output_root": tmp_path / "out",
        "repo": tmp_path,
    })()


def test_prepares_one_axis_gather_from_selected_temp_without_launch(tmp_path, monkeypatch):
    manifest, path = arm.prepare(_args(tmp_path, monkeypatch))
    assert path.is_file()
    assert manifest["launch_authorized"] is False
    assert manifest["diagnostic_execution_authorized"] is True
    assert manifest["launch_interface_present"] == (
        "tools/a1_topology_gather_arm_execute.py --go"
    )
    assert manifest["diagnostic_executor"] == manifest["source_binding"]["files"][
        arm.EXECUTOR_RELATIVE_PATH
    ]
    assert manifest["only_declared_optimization_delta"] == (
        "commission function-preserving target_gather_proj only"
    )
    assert manifest["matched_contract"]["row_dose_and_objective_operator_unchanged"] is True
    assert manifest["matched_contract"]["sampler_batch_partition_unchanged"] is False
    assert manifest["matched_contract"]["step0_network_outputs_bit_identical"] is True
    assert manifest["function_preserving_upgrade"]["shared_parameters_bit_identical"] is True
    assert manifest["function_preserving_upgrade"]["new_parameters"] == list(
        arm.EXPECTED_NEW_PARAMETERS
    )
    assert len(manifest["corpus_topology_target_coverage"]["components"]) == 3
    assert manifest["executor_compatibility"]["compatible_now"] is True
    assert manifest["executor_compatibility"]["one_shot"] is True
    command = manifest["command"]
    assert command.count(arm.corrected.EVENT_HISTORY_ACK_FLAG) == 3
    assert command.count(arm.corrected.EVENT_HISTORY_CROP_FLAG) == 1
    assert manifest["event_history_training_contract"][
        "crop_authenticated_empty_event_history"
    ] is True
    assert "--a1-learner-ablation-id" not in command
    assert arm.corrected._option(command, "--init-checkpoint") == str(
        _args_checkpoint(manifest)
    )
    assert arm.corrected._option(command, "--batch-size") == "64"
    assert arm.corrected._option(command, "--max-steps") == "1024"
    assert arm.corrected._option(command, "--action-module-lr-mult") == "4.0"
    assert arm.corrected._option(command, "--value-lr-mult") == "1.0"
    assert arm.corrected._option(command, "--freeze-modules") == arm.FREEZE_MODULES
    assert arm.corrected._option(
        command, "--require-only-trainable-prefixes"
    ) == arm.TRAINABLE_PREFIX
    assert manifest["adapter_commissioning_contract"] == {
        "reference_checkpoint": manifest["initialization_source"],
        "candidate_chaining": False,
        "world_size": 8,
        "local_batch_size": 64,
        "global_batch_size": 512,
        "optimizer_steps": 1024,
        "global_row_dose": 524_288,
        "lr_warmup_steps": 100,
        "integrated_lr_step_equivalents": 974.5,
        "action_module_lr_mult": 4.0,
        "action_integrated_lr_step_equivalents": 3898.0,
        "freeze_modules": ["trunk", "action_encoder", "policy_head", "value_heads"],
        "required_trainable_prefixes": ["target_gather_proj"],
        "mature_parameters_trainable": False,
        "interpretation": (
            "tests whether fixed f7 target-token features contain useful "
            "action-local signal; it is not a joint learner candidate"
        ),
    }


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


def test_coverage_refuses_missing_or_duplicate_supervised_audit_rows(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    payload = json.loads(args.architecture_audit.read_text())
    payload["audits"].append(dict(payload["audits"][0]))
    args.architecture_audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="exactly the supervised TEMP corpora"):
        arm.prepare(args)


def test_source_manifest_refuses_historical_dose_drift(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    payload = json.loads(args.source_manifest.read_text())
    payload["selected_dose"]["global_samples"] += 1
    payload["manifest_sha256"] = arm.corrected._digest(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    args.source_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="historical TEMP source dose"):
        arm.prepare(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("diagnostic_only", True),
        ("production_eligible", False),
        ("launch_authorized", False),
    ],
)
def test_source_requires_exact_sealed_production_shape(
    tmp_path, monkeypatch, field, value
):
    args = _args(tmp_path, monkeypatch)
    payload = json.loads(args.source_manifest.read_text())
    payload[field] = value
    payload["manifest_sha256"] = arm.corrected._digest(
        {key: item for key, item in payload.items() if key != "manifest_sha256"}
    )
    args.source_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="sealed production TEMP"):
        arm.prepare(args)


def test_bridge_never_misreads_current_typed_short_schema_as_legacy_full(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, monkeypatch)
    payload = json.loads(args.source_manifest.read_text())
    payload["schema_version"] = "a1-production-temperature-replication-v3"
    payload["manifest_sha256"] = arm.corrected._digest(
        {key: item for key, item in payload.items() if key != "manifest_sha256"}
    )
    args.source_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="sealed production TEMP"):
        arm.prepare(args)


def test_selected_geometry_refuses_objective_or_report_drift(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch)
    plan = json.loads(args.selected_dose_plan.read_text())
    command = plan["runs"][0]["command"]
    command[command.index("--soft-target-weight") + 1] = "1.0"
    plan["runs"][0]["command_sha256"] = arm.corrected._digest(command)
    plan["plan_sha256"] = arm.corrected._digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    args.selected_dose_plan.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="exact short-dose TEMP derivation"):
        arm.prepare(args)

    report_root = tmp_path / "report-drift"
    report_root.mkdir()
    args = _args(report_root, monkeypatch)
    report = json.loads(args.selected_dose_report.read_text())
    report["steps_completed"] = 127
    args.selected_dose_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="executed selected-geometry report drift"):
        arm.prepare(args)


def test_commissioning_appends_absent_freeze_contract(tmp_path):
    source = [
        "python", "train_bc.py", "--init-checkpoint", "f7.pt",
        "--checkpoint", "old.pt", "--report", "old.json",
        "--batch-size", "512", "--max-steps", "128",
        "--action-module-lr-mult", "1.0", "--value-lr-mult", "0.3",
    ]
    upgraded = tmp_path / "gather.pt"
    upgraded.write_bytes(b"gather")
    command, changes = arm._derive_command(
        source, upgraded=upgraded, output_root=tmp_path / "out"
    )
    assert arm.corrected._option(command, "--freeze-modules") == arm.FREEZE_MODULES
    assert changes["--freeze-modules"]["source"] == "absent"
