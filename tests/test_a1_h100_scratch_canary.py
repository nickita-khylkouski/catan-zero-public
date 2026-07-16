from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import a1_current_science_contract as current_science
from tools import a1_h100_scratch_canary as canary
from tools import a1_scratch_train as scratch
from tools import train_bc


def _verified(tmp_path: Path) -> dict:
    recipe = current_science.learner_training_recipe()
    topology = current_science.learner_execution_topology()
    recipe.update(
        world_size=topology["world_size"],
        batch_size=topology["local_batch_size"],
        grad_accum_steps=topology["grad_accum_steps"],
        global_batch_size=topology["global_batch_size"],
    )
    return {
        "recipe": recipe,
        "logical_recipe": current_science.learner_training_recipe(),
        "initialization": current_science.learner_initialization(),
        "model_construction": current_science.learner_model_construction(),
        "execution_topology": topology,
        "trainer_authority": scratch.one_dose._current_production_trainer_authority(),  # noqa: SLF001
        "data_path": tmp_path / "composite.json",
        "event_history_training_contract": {
            "empty_payload_inventory_acknowledgements": [],
            "training_event_history_trainable": True,
        },
        "accepted_policy_target_identity_sha256": "sha256:" + "a" * 64,
    }


def _valid_report_payload() -> dict:
    recipe = current_science.learner_training_recipe()
    modules = sorted(
        name.strip()
        for name in str(
            recipe["require_feature_learning_signal_modules"]
        ).split(",")
        if name.strip()
    )
    cadence = int(recipe["train_diagnostics_every_batches"])
    minimum = int(recipe["minimum_feature_learning_signal_observations"])
    observability = {
        "schema_version": "module-optimizer-observability-v1",
        "norm_scope": "global_replicated",
        "cadence_batches": cadence,
        "observed_steps": minimum,
        "modules": {
            name: {
                "mean_pre_clip_grad_norm": 1.0,
                "max_pre_clip_grad_norm": 1.25,
                "mean_parameter_delta_norm": 0.5,
                "mean_parameter_update_rms": 0.25,
                "parameter_count": 1,
            }
            for name in modules
        },
    }
    objective = {
        "schema_version": "objective-gradient-dose-observations-v1",
        "cadence_batches": int(
            recipe["objective_gradient_interference_every_batches"]
        ),
        "observed_steps": minimum,
        "observations": [
            {
                "optimizer_step": (index + 1) * cadence,
                "available": True,
                "scope": "global_ddp_microbatch",
                "aggregation": (
                    "manual_all_reduce_then_world_average_of_ddp_scaled_gradients"
                ),
                "world_size": 8,
                "scalar_value_trunk_grad_scale": 0.25,
                "policy_trunk_grad_norm": 1.0,
                "value_trunk_grad_norm": 0.5,
                "combined_trunk_grad_norm": 1.1,
                "value_to_policy_grad_norm_ratio": 0.5,
                "trunk_gradient_cosine": 0.1,
                "opposing_coordinate_fraction": 0.2,
            }
            for index in range(minimum)
        ],
    }
    contract = canary.feature_signal.contract_from_cli(
        module_names=modules,
        cadence_batches=cadence,
        minimum_observations=minimum,
    )
    feature_admission = canary.feature_signal.verify_observability(
        observability, contract=contract, where="test"
    )
    objective_admission = canary.feature_signal.verify_objective_interference(
        objective,
        cadence_batches=int(
            recipe["objective_gradient_interference_every_batches"]
        ),
        minimum_observations=minimum,
        expected_world_size=8,
        expected_value_trunk_grad_scale=0.25,
        where="test",
    )
    return {
        "parameter_count": 41_708_233,
        "trainable_parameter_count": 41_708_233,
        "forward_active_parameter_count": 41_708_233,
        "max_steps": 128,
        "exact_max_steps": True,
        "steps_completed": 128,
        "module_optimizer_observability": observability,
        "feature_learning_signal_admission": feature_admission,
        "objective_gradient_signal_admission": objective_admission,
        "objective_gradient_interference": objective,
    }


def _cuda_inventory() -> list[dict]:
    return [
        {
            "index": index,
            "uuid": f"gpu-{index}",
            "name": "NVIDIA H100 80GB HBM3",
            "total_memory_bytes": 80 * 1024**3,
            "compute_capability": [9, 0],
        }
        for index in range(8)
    ]


def test_required_arms_are_current_c640_and_single_delta_t640() -> None:
    arms = canary.build_arm_contracts(max_steps=128)
    control = arms["C640"]
    treatment = arms["T640"]

    assert set(arms) == {"C640", "T640"}
    assert control["model_construction"]["hidden_size"] == 640
    assert control["model_construction"]["action_target_gather"] is True
    assert control["recipe"]["value_trunk_grad_scale"] == 0.25
    assert control["model_construction"]["topology_residual_adapter"] is False
    assert treatment["model_construction"]["topology_residual_adapter"] is True
    assert canary.arm_drift(control, treatment) == {
        "model_construction.topology_residual_adapter": {"C640": False, "T640": True}
    }
    assert control["matched_identity_sha256"] == treatment["matched_identity_sha256"]


@pytest.mark.parametrize("steps", [0, 127, 257, 10_000])
def test_canary_hard_refuses_out_of_range_step_dose(steps: int) -> None:
    with pytest.raises(canary.CanaryError, match="128..256"):
        canary.build_arm_contracts(max_steps=steps)


def test_arm_commands_differ_only_declared_architecture_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["data_path"].write_text("{}\n", encoding="utf-8")
    verified["lock_path"] = tmp_path / "lock.json"
    verified["lock_path"].write_text("{}\n", encoding="utf-8")
    base = [
        "/usr/bin/python3",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=8",
        "tools/train_bc.py",
        "--hidden-size",
        "640",
        "--max-35m-params",
        "42000000",
        "--max-steps",
        "0",
        "--checkpoint-steps",
        "8,16,32,64,128,256,512,1024",
        "--checkpoint",
        "old.pt",
        "--report",
        "old.json",
        "--a1-scratch-authority-json",
        "{}",
    ]
    def fake_build(bound, **kwargs):
        command = list(base)
        command[command.index("--max-steps") + 1] = str(bound["recipe"]["max_steps"])
        command[command.index("--checkpoint-steps") + 1] = str(
            bound["recipe"]["checkpoint_steps"]
        )
        command[command.index("--checkpoint") + 1] = str(kwargs["checkpoint"])
        command[command.index("--report") + 1] = str(kwargs["report"])
        return command

    monkeypatch.setattr(canary.scratch, "build_train_command", fake_build)
    monkeypatch.setattr(
        canary,
        "_effective_recipe_from_command",
        lambda _command: {"epochs": 1, "max_steps": 128},
    )
    commands = canary.build_commands(
        verified,
        python=Path("/usr/bin/python3"),
        output_dir=tmp_path / "out",
        max_steps=128,
    )

    control = canary.normalized_matched_command(commands["C640"])
    treatment = canary.normalized_matched_command(commands["T640"])
    assert control == treatment
    assert "--no-topology-residual-adapter" in commands["C640"]
    assert "--topology-residual-adapter" in commands["T640"]
    for command in commands.values():
        assert command[command.index("--max-steps") + 1] == "128"
        assert command[command.index("--checkpoint-steps") + 1] == "8,16,32,64"
        assert command.count("--exact-max-steps") == 1
        assert command.count("--a1-scratch-authority-json") == 1
        assert command.count("--a1-scratch-diagnostic-authority-json") == 1


def test_real_scratch_command_renderer_accepts_both_topology_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["data_path"].write_text("{}\n", encoding="utf-8")
    verified["lock_path"] = tmp_path / "lock.json"
    verified["lock_path"].write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        canary.scratch,
        "_scratch_plan_authority",
        lambda _verified: {
            "schema_version": "a1-coherent-scratch-plan-authority-v2",
            "test": True,
        },
    )

    commands = canary.build_commands(
        verified,
        python=Path("/usr/bin/python3"),
        output_dir=tmp_path / "out",
        max_steps=128,
    )

    assert set(commands) == {"C640", "T640"}
    science = {
        "learner_training_recipe": current_science.learner_training_recipe(),
        "learner_model_construction": current_science.learner_model_construction(),
        "learner_execution_topology": current_science.learner_execution_topology(),
    }
    for arm_id, command in commands.items():
        parsed = train_bc.build_parser().parse_args(
            command[canary._trainer_index(command) + 1 :]  # noqa: SLF001
        )
        assert parsed.max_steps == 128
        assert parsed.exact_max_steps is True
        assert parsed.max_35m_params == canary.CANARY_MAX_PARAMETER_COUNT
        assert parsed.topology_residual_adapter is (arm_id == "T640")
        validated = train_bc._validate_a1_scratch_diagnostic_authority(  # noqa: SLF001
            parsed.a1_scratch_diagnostic_authority_json,
            args=parsed,
            science=science,
        )
        assert validated["arm_id"] == arm_id


def test_dry_run_writes_non_promotable_receipt_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["data_path"].write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(canary.scratch, "verify_inputs", lambda **_kwargs: verified)
    monkeypatch.setattr(
        canary,
        "build_commands",
        lambda *_args, **_kwargs: {"C640": ["control"], "T640": ["treatment"]},
    )
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("dry run executed")

    receipt_path = tmp_path / "plan.json"
    result = canary.run(
        SimpleNamespace(
            lock=tmp_path / "lock.json",
            data=verified["data_path"],
            composite_build_receipt=tmp_path / "build.json",
            output_dir=tmp_path / "out",
            receipt=receipt_path,
            python=Path("/usr/bin/python3"),
            max_steps=128,
            go=False,
        ),
        runner=forbidden_runner,
    )

    assert called is False
    assert result["status"] == "planned"
    assert result["diagnostic_only"] is True
    assert result["promotion_eligible"] is False
    assert result["production_admission"] == "forbidden"
    assert json.loads(receipt_path.read_text())["receipt_sha256"].startswith("sha256:")


def test_completion_summary_binds_counts_throughput_and_gradient_telemetry(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _valid_report_payload()
        ),
        encoding="utf-8",
    )
    summary = canary.summarize_report(
        report, max_steps=128, global_batch_size=512, elapsed_seconds=64.0
    )

    assert summary["parameter_counts"]["total"] == 41_708_233
    assert summary["throughput"]["optimizer_steps_per_second"] == 2.0
    assert summary["throughput"]["rows_per_second"] == 1024.0
    assert summary["gradient_telemetry"]["feature_learning_signal_admission"][
        "authenticated"
    ] is True


@pytest.mark.parametrize("observed", [None, True, 127, 129])
def test_completion_summary_rejects_unproved_optimizer_dose(
    tmp_path: Path, observed: object
) -> None:
    payload = _valid_report_payload()
    payload["steps_completed"] = observed
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(canary.CanaryError, match="exact optimizer-step"):
        canary.summarize_report(
            report, max_steps=128, global_batch_size=512, elapsed_seconds=1.0
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_completion_summary_rejects_nonfinite_gradient_telemetry(
    tmp_path: Path, bad_value: float
) -> None:
    payload = _valid_report_payload()
    payload["objective_gradient_interference"]["cosine"] = bad_value
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(canary.CanaryError, match="finite structured"):
        canary.summarize_report(
            report, max_steps=128, global_batch_size=512, elapsed_seconds=1.0
        )


def test_completion_summary_rejects_finite_but_malformed_gradient_telemetry(
    tmp_path: Path,
) -> None:
    payload = _valid_report_payload()
    modules = payload["module_optimizer_observability"]["modules"]
    modules.pop(next(iter(modules)))
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(canary.CanaryError, match="failed admission"):
        canary.summarize_report(
            report, max_steps=128, global_batch_size=512, elapsed_seconds=1.0
        )


def test_inventory_uses_authenticated_python_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_run(command, **_kwargs):
        seen.extend(command)
        return SimpleNamespace(
            stdout=json.dumps(
                {"cuda_available": True, "records": _cuda_inventory()}
            )
        )

    monkeypatch.setattr(canary.subprocess, "run", fake_run)
    inventory = canary._h100_inventory(Path("/sealed/python"))  # noqa: SLF001
    assert seen[0] == "/sealed/python"
    assert len(inventory) == 8


def test_inventory_rejects_host_with_only_seven_cuda_visible_h100s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        canary.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(
                {"cuda_available": True, "records": _cuda_inventory()[:7]}
            )
        ),
    )
    with pytest.raises(canary.CanaryError, match="CUDA-visible H100"):
        canary._h100_inventory(Path("/sealed/python"))  # noqa: SLF001


def test_production_scratch_authority_remains_uncommissioned_and_canary_is_rejected(
    tmp_path: Path,
) -> None:
    topology = current_science.learner_execution_topology()
    assert topology["name"] == "b200-8gpu-ddp"
    assert topology["go_authorized"] is False
    assert topology["optimization_schedule_status"] == "unresolved"
    with pytest.raises(SystemExit, match="schedule is unresolved"):
        train_bc._require_a1_scratch_execution_schedule(topology)  # noqa: SLF001
    with pytest.raises(canary.CanaryError, match="never production-admissible"):
        canary.require_production_admission({"promotion_eligible": False})


def test_production_runtime_projection_rejects_topology_treatment() -> None:
    model = current_science.learner_model_construction()
    topology = current_science.learner_execution_topology()
    args = SimpleNamespace(
        init_checkpoint="",
        grow_from_checkpoint="",
        resume_optimizer=False,
        arch=model["arch"],
        hidden_size=model["hidden_size"],
        graph_layers=model["graph_layers"],
        attention_heads=model["attention_heads"],
        graph_dropout=model["graph_dropout"],
        entity_state_trunk=model["entity_state_trunk"],
        action_target_gather=model["action_target_gather"],
        topology_residual_adapter=True,
        static_action_residual=model["static_action_residual"],
        legal_action_value_residual=model["legal_action_value_residual"],
        legal_action_value_set_statistics=model["legal_action_value_set_statistics"],
        value_tower_split_layers=model["value_tower_split_layers"],
        public_card_count_features=model["public_card_count_features"],
        public_card_count_residual_bias=model["public_card_count_residual_bias"],
        public_rule_state_features=model["public_rule_state_features"],
        entity_feature_adapter_version=model["entity_feature_adapter_version"],
        meaningful_public_history=model["meaningful_public_history"],
        meaningful_public_history_pooling=model["meaningful_public_history_pooling"],
        meaningful_public_history_target_gather=model[
            "meaningful_public_history_target_gather"
        ],
        event_history_limit=model["event_history_limit"],
        mask_hidden_info=model["mask_hidden_info"],
        require_35m_model=model["require_35m_model"],
        max_35m_params=model["max_parameter_count"],
        batch_size=topology["local_batch_size"],
        grad_accum_steps=topology["grad_accum_steps"],
        ddp_shard_data=topology["ddp_shard_data"],
        training_rng_rank_offset=topology["training_rng_rank_offset"],
    )
    with pytest.raises(SystemExit, match="topology_residual_adapter"):
        train_bc._validate_a1_scratch_runtime_projection(  # noqa: SLF001
            args, {"world_size": 8}, model, topology
        )


def test_topology_diagnostic_authority_is_exact_and_bounded() -> None:
    authority = canary.diagnostic_authority(
        arm_id="T640",
        max_steps=128,
        checkpoint_steps=(8, 16, 32, 64),
        code_tree_sha256="sha256:" + "a" * 64,
    )
    science = {
        "learner_training_recipe": current_science.learner_training_recipe(),
        "learner_model_construction": current_science.learner_model_construction(),
        "learner_execution_topology": current_science.learner_execution_topology(),
    }
    validated = train_bc._validate_a1_scratch_diagnostic_authority(  # noqa: SLF001
        json.dumps(authority), science=science
    )
    assert validated["topology_residual_adapter"] is True

    authority["max_steps"] = 257
    with pytest.raises(SystemExit, match="value drift"):
        train_bc._validate_a1_scratch_diagnostic_authority(  # noqa: SLF001
            json.dumps(authority), science=science
        )


def test_topology_diagnostic_is_the_only_runtime_projection_exception() -> None:
    model = current_science.learner_model_construction()
    topology = current_science.learner_execution_topology()
    args = SimpleNamespace(
        init_checkpoint="",
        grow_from_checkpoint="",
        resume_optimizer=False,
        arch=model["arch"],
        hidden_size=model["hidden_size"],
        graph_layers=model["graph_layers"],
        attention_heads=model["attention_heads"],
        graph_dropout=model["graph_dropout"],
        entity_state_trunk=model["entity_state_trunk"],
        action_target_gather=model["action_target_gather"],
        topology_residual_adapter=True,
        static_action_residual=model["static_action_residual"],
        legal_action_value_residual=model["legal_action_value_residual"],
        legal_action_value_set_statistics=model["legal_action_value_set_statistics"],
        value_tower_split_layers=model["value_tower_split_layers"],
        public_card_count_features=model["public_card_count_features"],
        public_card_count_residual_bias=model["public_card_count_residual_bias"],
        public_rule_state_features=model["public_rule_state_features"],
        entity_feature_adapter_version=model["entity_feature_adapter_version"],
        meaningful_public_history=model["meaningful_public_history"],
        meaningful_public_history_pooling=model["meaningful_public_history_pooling"],
        meaningful_public_history_target_gather=model[
            "meaningful_public_history_target_gather"
        ],
        event_history_limit=model["event_history_limit"],
        mask_hidden_info=model["mask_hidden_info"],
        require_35m_model=model["require_35m_model"],
        max_35m_params=model["max_parameter_count"],
        batch_size=topology["local_batch_size"],
        grad_accum_steps=topology["grad_accum_steps"],
        ddp_shard_data=topology["ddp_shard_data"],
        training_rng_rank_offset=topology["training_rng_rank_offset"],
    )
    diagnostic = canary.diagnostic_authority(
        arm_id="T640",
        max_steps=128,
        checkpoint_steps=(8, 16, 32, 64),
        code_tree_sha256="sha256:" + "a" * 64,
    )
    args.max_35m_params = canary.CANARY_MAX_PARAMETER_COUNT
    train_bc._validate_a1_scratch_runtime_projection(  # noqa: SLF001
        args,
        {"world_size": 8},
        model,
        topology,
        diagnostic_authority=diagnostic,
    )
    diagnostic["topology_residual_adapter"] = False
    with pytest.raises(SystemExit, match="topology_residual_adapter"):
        train_bc._validate_a1_scratch_runtime_projection(  # noqa: SLF001
            args,
            {"world_size": 8},
            model,
            topology,
            diagnostic_authority=diagnostic,
        )


def test_topology_authority_rejects_self_declared_recipe_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["data_path"].write_text("{}\n", encoding="utf-8")
    verified["lock_path"] = tmp_path / "lock.json"
    verified["lock_path"].write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        canary.scratch,
        "_scratch_plan_authority",
        lambda _verified: {
            "schema_version": "a1-coherent-scratch-plan-authority-v2",
            "test": True,
        },
    )
    command = canary.build_commands(
        verified,
        python=Path("/usr/bin/python3"),
        output_dir=tmp_path / "out",
        max_steps=128,
    )["C640"]
    canary._replace_value(command, "--lr", 0.123)  # noqa: SLF001
    forged_effective = canary._effective_recipe_from_command(command)  # noqa: SLF001
    forged_sha = canary._value_sha256(forged_effective)  # noqa: SLF001
    canary._replace_value(  # noqa: SLF001
        command,
        "--a1-effective-learner-recipe-json",
        canary._canonical_bytes(forged_effective).decode("ascii"),  # noqa: SLF001
    )
    canary._replace_value(  # noqa: SLF001
        command, "--a1-effective-learner-recipe-sha256", forged_sha
    )
    authority_index = command.index("--a1-scratch-diagnostic-authority-json") + 1
    forged_authority = json.loads(command[authority_index])
    forged_authority["effective_recipe_sha256"] = forged_sha
    command[authority_index] = canary._canonical_bytes(forged_authority).decode(  # noqa: SLF001
        "ascii"
    )
    args = train_bc.build_parser().parse_args(
        command[canary._trainer_index(command) + 1 :]  # noqa: SLF001
    )
    science = {
        "learner_training_recipe": current_science.learner_training_recipe(),
        "learner_model_construction": current_science.learner_model_construction(),
        "learner_execution_topology": current_science.learner_execution_topology(),
    }
    with pytest.raises(SystemExit, match="matched V25 recipe drift"):
        train_bc._validate_a1_scratch_diagnostic_authority(  # noqa: SLF001
            args.a1_scratch_diagnostic_authority_json,
            args=args,
            science=science,
        )


def test_topology_authority_rejects_self_selected_code_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["data_path"].write_text("{}\n", encoding="utf-8")
    verified["lock_path"] = tmp_path / "lock.json"
    verified["lock_path"].write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        canary.scratch,
        "_scratch_plan_authority",
        lambda _verified: {
            "schema_version": "a1-coherent-scratch-plan-authority-v2",
            "test": True,
        },
    )
    command = canary.build_commands(
        verified,
        python=Path("/usr/bin/python3"),
        output_dir=tmp_path / "out",
        max_steps=128,
    )["T640"]
    binding_index = command.index("--a1-ablation-code-binding-json") + 1
    binding = json.loads(command[binding_index])
    binding["records"] = binding["records"][:-1]
    unhashed = dict(binding)
    unhashed.pop("code_tree_sha256")
    forged_code_sha = canary._value_sha256(unhashed)  # noqa: SLF001
    binding["code_tree_sha256"] = forged_code_sha
    command[binding_index] = canary._canonical_bytes(binding).decode("ascii")  # noqa: SLF001
    canary._replace_value(  # noqa: SLF001
        command, "--a1-ablation-code-tree-sha256", forged_code_sha
    )
    authority_index = command.index("--a1-scratch-diagnostic-authority-json") + 1
    authority = json.loads(command[authority_index])
    authority["code_tree_sha256"] = forged_code_sha
    command[authority_index] = canary._canonical_bytes(authority).decode("ascii")  # noqa: SLF001
    args = train_bc.build_parser().parse_args(
        command[canary._trainer_index(command) + 1 :]  # noqa: SLF001
    )
    science = {
        "learner_training_recipe": current_science.learner_training_recipe(),
        "learner_model_construction": current_science.learner_model_construction(),
        "learner_execution_topology": current_science.learner_execution_topology(),
    }
    with pytest.raises(SystemExit, match="code surface drift"):
        train_bc._validate_a1_scratch_diagnostic_authority(  # noqa: SLF001
            args.a1_scratch_diagnostic_authority_json,
            args=args,
            science=science,
        )


@pytest.mark.parametrize("device_count", [7, 8])
def test_trainer_refuses_non_eight_h100_runtime(
    monkeypatch: pytest.MonkeyPatch, device_count: int
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            uuid=f"gpu-{index}",
            name="NVIDIA B200" if device_count == 8 else "NVIDIA H100 80GB HBM3",
            total_memory=80 * 1024**3,
            major=9,
            minor=0,
        ),
    )
    diagnostic = canary.diagnostic_authority(
        arm_id="T640",
        max_steps=128,
        checkpoint_steps=(8, 16, 32, 64),
        code_tree_sha256="sha256:" + "a" * 64,
    )
    with pytest.raises(SystemExit, match="CUDA-visible H100"):
        train_bc._require_a1_scratch_topology_h100_runtime(  # noqa: SLF001
            diagnostic,
            {"local_rank": 0},
        )


@pytest.mark.parametrize("failure_mode", ["returncode", "missing_checkpoint"])
def test_failed_execution_receipt_binds_plan_and_preserves_completed_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    verified = _verified(tmp_path)
    verified["data_path"].write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(canary.scratch, "verify_inputs", lambda **_kwargs: verified)
    monkeypatch.setattr(
        canary.scratch,
        "_executable_ref",
        lambda *_args, **_kwargs: {
            "path": "/sealed/python",
            "sha256": "sha256:" + "a" * 64,
        },
    )
    monkeypatch.setattr(
        canary,
        "build_commands",
        lambda *_args, **_kwargs: {"C640": ["C640"], "T640": ["T640"]},
    )
    monkeypatch.setattr(canary, "_h100_inventory", lambda _python: _cuda_inventory())

    output_dir = tmp_path / "out"
    plan_path = tmp_path / "plan.json"
    base_args = dict(
        lock=tmp_path / "lock.json",
        data=verified["data_path"],
        composite_build_receipt=tmp_path / "build.json",
        output_dir=output_dir,
        python=Path("/sealed/python"),
        max_steps=128,
    )
    canary.run(
        SimpleNamespace(
            **base_args,
            receipt=plan_path,
            plan_receipt=None,
            go=False,
        )
    )

    def fake_runner(command, **_kwargs):
        arm_id = command[0]
        arm_dir = output_dir / arm_id
        if arm_id == "T640" and failure_mode == "returncode":
            return SimpleNamespace(returncode=9)
        (arm_dir / "training-report.json").write_text(
            json.dumps(_valid_report_payload()), encoding="utf-8"
        )
        if arm_id == "C640" or failure_mode != "missing_checkpoint":
            (arm_dir / "candidate.pt").write_bytes(arm_id.encode("ascii"))
        return SimpleNamespace(returncode=0)

    execution_path = tmp_path / "execution.json"
    with pytest.raises(canary.CanaryError):
        canary.run(
            SimpleNamespace(
                **base_args,
                receipt=execution_path,
                plan_receipt=plan_path,
                go=True,
            ),
            runner=fake_runner,
        )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    unsigned = dict(execution)
    receipt_sha = unsigned.pop("receipt_sha256")
    assert receipt_sha == canary._value_sha256(unsigned)  # noqa: SLF001
    assert execution["status"] == "failed"
    assert execution["failed_arm"] == "T640"
    assert set(execution["results"]) == {"C640"}
    assert execution["results"]["C640"]["checkpoint"]["file_sha256"].startswith(
        "sha256:"
    )
    assert execution["plan_binding"]["plan_receipt"][
        "plan_identity_sha256"
    ] == execution["plan_binding"]["plan"]["plan_identity_sha256"]
    assert len(execution["gpu_inventory"]) == 8


def test_execution_refuses_arguments_that_differ_from_reviewed_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    verified["data_path"].write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(canary.scratch, "verify_inputs", lambda **_kwargs: verified)
    monkeypatch.setattr(
        canary.scratch,
        "_executable_ref",
        lambda *_args, **_kwargs: {
            "path": "/sealed/python",
            "sha256": "sha256:" + "a" * 64,
        },
    )
    monkeypatch.setattr(
        canary,
        "build_commands",
        lambda _verified, **kwargs: {
            "C640": ["C640", str(kwargs["output_dir"])],
            "T640": ["T640", str(kwargs["output_dir"])],
        },
    )
    plan_path = tmp_path / "plan.json"
    common = dict(
        lock=tmp_path / "lock.json",
        data=verified["data_path"],
        composite_build_receipt=tmp_path / "build.json",
        python=Path("/sealed/python"),
        max_steps=128,
    )
    canary.run(
        SimpleNamespace(
            **common,
            output_dir=tmp_path / "planned-output",
            receipt=plan_path,
            plan_receipt=None,
            go=False,
        )
    )
    with pytest.raises(canary.CanaryError, match="differ from immutable"):
        canary.run(
            SimpleNamespace(
                **common,
                output_dir=tmp_path / "different-output",
                receipt=tmp_path / "execution.json",
                plan_receipt=plan_path,
                go=True,
            )
        )
