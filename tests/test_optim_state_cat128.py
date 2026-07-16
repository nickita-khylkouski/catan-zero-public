"""CAT-128 patch #8: optimizer-state sidecar round-trip + fail-safe, the memmap-
incompatible teacher-guard fail-fast, and a GPU-gated 2-rank FSDP-collective verify.

NOTE: FSDP requires a non-CPU accelerator on our torch (2.11 and 2.13 both raise "FSDP
needs a non-CPU accelerator device" on CPU), so the FSDP collective gather CANNOT be
exercised on pure CPU. The FSDP test therefore runs on >=2 GPUs and is skipped otherwise
(freeze-safe). Single-proc + DDP coverage (here + audit-fixer's CAT-126 tests) plus the
GPU verify give optim_state.py: single / DDP / real-FSDP-collective coverage.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl import optim_state as optim_state_module  # noqa: E402
from catan_zero.rl.optim_state import (  # noqa: E402
    TERMINAL_ADMITTED_CHECKPOINT_ROLE,
    TrainingProgressError,
    is_fsdp,
    load_training_progress,
    load_optimizer_state,
    optimizer_sidecar_path,
    save_training_progress,
    save_optimizer_state,
    training_progress_sidecar_path,
)

_DDP_SINGLE = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}


def _stepped_adam():
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(5, 8)).sum().backward()
        opt.step()
    return model, opt


def test_sidecar_path_convention(tmp_path):
    assert optimizer_sidecar_path(tmp_path / "ckpt.pt").name == "ckpt.pt.optimizer.pt"


def test_is_fsdp_false_for_plain_module():
    assert is_fsdp(torch.nn.Linear(2, 2)) is False


def test_save_load_roundtrip_restores_adam_moments(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    model, opt = _stepped_adam()
    exp_avg = opt.state[model.weight]["exp_avg"].clone()
    assert save_optimizer_state(ckpt, model, opt, _DDP_SINGLE) is not None
    assert optimizer_sidecar_path(ckpt).exists()

    fresh = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert len(fresh.state) == 0
    assert load_optimizer_state(ckpt, model, fresh, _DDP_SINGLE) is True
    assert torch.allclose(fresh.state[model.weight]["exp_avg"], exp_avg)


def test_load_missing_sidecar_is_failsafe(tmp_path):
    model, opt = _stepped_adam()
    assert load_optimizer_state(tmp_path / "absent.pt", model, opt, _DDP_SINGLE) is False


def test_load_corrupt_sidecar_never_raises(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    optimizer_sidecar_path(ckpt).write_bytes(b"not a torch pickle")
    model, opt = _stepped_adam()
    assert load_optimizer_state(ckpt, model, opt, _DDP_SINGLE) is False


def _committed_progress(
    tmp_path: Path,
    *,
    recipe=None,
    controller_state=None,
    checkpoint_role="resumable_epoch",
    policy_aux_global_draw_offset=0,
):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"model-v1")
    model, optimizer = _stepped_adam()
    save_optimizer_state(ckpt, model, optimizer, _DDP_SINGLE)
    rng = __import__("numpy").random.default_rng(17)
    rng.random(9)
    identity = recipe or {
        "schema_version": "recipe-v1",
        "lr": 3e-5,
        "world_size": 1,
    }
    save_training_progress(
        ckpt,
        optimizer_step=713,
        completed_epochs=2,
        recipe_identity=identity,
        rng_state=rng.bit_generator.state,
        rank_numpy_rng_states=[rng.bit_generator.state],
        symmetry_rng_state=None,
        rank_torch_rng_states=[
            {"rank": 0, "cpu": torch.get_rng_state().tolist(), "cuda": None}
        ],
        scalar_training_weight_sum=123.5,
        categorical_training_weight_sum=44.0,
        checkpoint_role=checkpoint_role,
        policy_aux_global_draw_offset=policy_aux_global_draw_offset,
        policy_kl_controller_state=controller_state,
        ddp=_DDP_SINGLE,
    )
    return ckpt, identity, rng.bit_generator.state


def test_progress_commit_binds_model_optimizer_recipe_and_exact_step(tmp_path):
    ckpt, identity, rng_state = _committed_progress(
        tmp_path, policy_aux_global_draw_offset=37
    )
    loaded = load_training_progress(ckpt, expected_recipe_identity=identity)
    assert loaded["optimizer_step"] == 713
    assert loaded["completed_epochs"] == 2
    assert loaded["checkpoint_role"] == "resumable_epoch"
    assert loaded["rng_state"] == rng_state
    assert loaded["policy_aux_global_draw_offset"] == 37
    assert training_progress_sidecar_path(ckpt).name.endswith(
        ".training-progress.json"
    )


def test_progress_role_separates_resume_from_terminal_admission(tmp_path):
    resumable, identity, _ = _committed_progress(tmp_path)
    with pytest.raises(TrainingProgressError, match="required checkpoint role"):
        load_training_progress(
            resumable,
            expected_recipe_identity=identity,
            required_checkpoint_role=TERMINAL_ADMITTED_CHECKPOINT_ROLE,
        )

    terminal_dir = tmp_path / "terminal"
    terminal_dir.mkdir()
    terminal, identity, _ = _committed_progress(
        terminal_dir,
        checkpoint_role=TERMINAL_ADMITTED_CHECKPOINT_ROLE,
    )
    loaded = load_training_progress(
        terminal,
        expected_recipe_identity=identity,
        required_checkpoint_role=TERMINAL_ADMITTED_CHECKPOINT_ROLE,
    )
    assert loaded["checkpoint_role"] == TERMINAL_ADMITTED_CHECKPOINT_ROLE


def test_progress_digest_authenticates_checkpoint_role(tmp_path):
    checkpoint, identity, _ = _committed_progress(tmp_path)
    path = training_progress_sidecar_path(checkpoint)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint_role"] = TERMINAL_ADMITTED_CHECKPOINT_ROLE
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingProgressError, match="digest mismatch"):
        load_training_progress(
            checkpoint,
            expected_recipe_identity=identity,
            required_checkpoint_role=TERMINAL_ADMITTED_CHECKPOINT_ROLE,
        )


def test_legacy_progress_remains_resumable_but_cannot_attest_terminal_role(tmp_path):
    checkpoint, identity, _ = _committed_progress(tmp_path)
    path = training_progress_sidecar_path(checkpoint)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "train-bc-progress-v1"
    del payload["checkpoint_role"]
    payload.pop("progress_sha256")
    payload["progress_sha256"] = optim_state_module._canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_training_progress(
        checkpoint, expected_recipe_identity=identity
    )["schema_version"] == "train-bc-progress-v1"
    with pytest.raises(TrainingProgressError, match="required checkpoint role"):
        load_training_progress(
            checkpoint,
            expected_recipe_identity=identity,
            required_checkpoint_role=TERMINAL_ADMITTED_CHECKPOINT_ROLE,
        )


def test_progress_round_trips_adaptive_policy_kl_controller_state(tmp_path):
    state = {
        "schema_version": "adaptive-parent-policy-kl-controller-v1",
        "target_kl": 0.05,
        "coefficient": 0.031,
        "updates": 17,
    }
    ckpt, identity, _ = _committed_progress(
        tmp_path, controller_state=state
    )
    loaded = load_training_progress(ckpt, expected_recipe_identity=identity)
    assert loaded["policy_kl_controller_state"] == state


@pytest.mark.parametrize("mutated", ["model", "optimizer"])
def test_progress_rejects_mixed_checkpoint_set(tmp_path, mutated):
    ckpt, identity, _ = _committed_progress(tmp_path)
    path = ckpt if mutated == "model" else optimizer_sidecar_path(ckpt)
    path.write_bytes(path.read_bytes() + b"different generation")
    with pytest.raises(TrainingProgressError, match="binding mismatch"):
        load_training_progress(ckpt, expected_recipe_identity=identity)


def test_progress_rejects_schedule_recipe_drift(tmp_path):
    ckpt, identity, _ = _committed_progress(tmp_path)
    with pytest.raises(TrainingProgressError, match="recipe/schedule"):
        load_training_progress(
            ckpt, expected_recipe_identity={**identity, "lr": 1.2e-4}
        )


def test_progress_rejects_tampered_counter_even_when_json_parses(tmp_path):
    ckpt, identity, _ = _committed_progress(tmp_path)
    path = training_progress_sidecar_path(ckpt)
    payload = json.loads(path.read_text())
    payload["optimizer_step"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(TrainingProgressError, match="digest mismatch"):
        load_training_progress(ckpt, expected_recipe_identity=identity)


def test_fsdp_optim_state_roundtrip_2gpu():
    """Real 2-rank FSDP-collective save/restore verify (FSDP.optim_state_dict /
    optim_state_dict_to_load). GPU-gated: FSDP needs an accelerator (no CPU/gloo path on
    our torch), so this SKIPS unless CAT128_FSDP_GPU_TEST=1 and >=2 CUDA devices are free.
    """
    import subprocess
    import sys

    if os.environ.get("CAT128_FSDP_GPU_TEST") != "1":
        pytest.skip("set CAT128_FSDP_GPU_TEST=1 (needs >=2 free GPUs) to run the FSDP verify")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("FSDP collective verify needs >=2 CUDA devices (CPU/gloo FSDP unsupported)")

    root = Path(__file__).resolve().parents[1]
    worker = Path(__file__).with_name("_fsdp_optim_worker.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1", "--nproc_per_node=2", "--tee=3", "--master_port=29579", str(worker),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    assert "FSDP_OPTIM_OK" in out and proc.returncode == 0, (
        f"2-rank FSDP optim round-trip failed (rc={proc.returncode}):\n{out[-2000:]}"
    )


def _load_train_bc():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("train_bc_cat128", root / "tools" / "train_bc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memmap_incompatible_with_strict_teacher_gate_fails_fast():
    tb = _load_train_bc()
    with pytest.raises(SystemExit) as exc:
        tb.main(
            [
                "--data", "d", "--data-format", "memmap", "--require-strict-35m-teacher",
                "--checkpoint", "c", "--report", "r", "--skip-guards",
            ]
        )
    assert "incompatible with --data-format memmap" in str(exc.value)


def test_train_bc_restores_schedule_counter_epoch_and_sampler_rng():
    tb = _load_train_bc()
    np = __import__("numpy")
    source_rng = np.random.default_rng(91)
    source_rng.random(11)
    expected_next = source_rng.random(4)
    # Recreate the state before consuming the expected continuation.
    source_rng = np.random.default_rng(91)
    source_rng.random(11)
    resumed_rng = np.random.default_rng(999)
    progress = {
        "optimizer_step": 713,
        "completed_epochs": 2,
        "rng_state": source_rng.bit_generator.state,
        "symmetry_rng_state": None,
        "rank_torch_rng_states": [
            {"rank": 0, "cpu": torch.get_rng_state().tolist(), "cuda": None}
        ],
        "scalar_training_weight_sum": 123.5,
        "categorical_training_weight_sum": 44.0,
    }
    restored = tb._restore_training_progress_state(
        progress,
        epochs=4,
        rng=resumed_rng,
        symmetry_rng=None,
        ddp=_DDP_SINGLE,
    )
    assert restored == (713, 2, 123.5, 44.0, 0.0)
    assert np.array_equal(resumed_rng.random(4), expected_next)


def test_train_bc_restores_rank_local_numpy_sampler_rng() -> None:
    tb = _load_train_bc()
    np = __import__("numpy")
    rank0_rng = np.random.default_rng(10)
    rank1_rng = np.random.default_rng(20)
    rank0_rng.random(3)
    rank1_rng.random(7)
    rank1_expected = rank1_rng.random(4)
    rank1_rng = np.random.default_rng(20)
    rank1_rng.random(7)
    resumed_rng = np.random.default_rng(999)
    progress = {
        "optimizer_step": 7,
        "completed_epochs": 1,
        "rng_state": rank0_rng.bit_generator.state,
        "rank_numpy_rng_states": [
            rank0_rng.bit_generator.state,
            rank1_rng.bit_generator.state,
        ],
        "symmetry_rng_state": None,
        "rank_torch_rng_states": [
            {"rank": 0, "cpu": torch.get_rng_state().tolist(), "cuda": None},
            {"rank": 1, "cpu": torch.get_rng_state().tolist(), "cuda": None},
        ],
        "scalar_training_weight_sum": 0.0,
        "categorical_training_weight_sum": 0.0,
        "recipe_identity": {"ddp_shard_data": True},
    }

    tb._restore_training_progress_state(
        progress,
        epochs=2,
        rng=resumed_rng,
        symmetry_rng=None,
        ddp={"enabled": True, "world_size": 2, "rank": 1, "local_rank": 1},
    )

    assert np.array_equal(resumed_rng.random(4), rank1_expected)


def test_train_bc_restores_consumed_policy_lr_area() -> None:
    tb = _load_train_bc()
    np = __import__("numpy")
    rng = np.random.default_rng(12)
    progress = {
        "optimizer_step": 7,
        "completed_epochs": 1,
        "rng_state": rng.bit_generator.state,
        "symmetry_rng_state": None,
        "rank_torch_rng_states": [
            {"rank": 0, "cpu": torch.get_rng_state().tolist(), "cuda": None}
        ],
        "scalar_training_weight_sum": 0.0,
        "categorical_training_weight_sum": 0.0,
        "policy_objective_lr_area": 0.0125,
    }

    restored = tb._restore_training_progress_state(
        progress,
        epochs=2,
        rng=np.random.default_rng(999),
        symmetry_rng=None,
        ddp=_DDP_SINGLE,
        require_policy_objective_lr_area=True,
    )

    assert restored[-1] == pytest.approx(0.0125)


def test_weighted_aux_resume_without_cumulative_offset_fails_closed() -> None:
    tb = _load_train_bc()

    with pytest.raises(SystemExit, match="lacks cumulative global draw offset"):
        tb._restore_policy_aux_global_draw_offset(  # noqa: SLF001
            {"optimizer_step": 7},
            required=True,
        )


def test_policy_dose_resume_without_consumed_area_fails_closed() -> None:
    tb = _load_train_bc()
    np = __import__("numpy")
    rng = np.random.default_rng(12)
    progress = {
        "optimizer_step": 7,
        "completed_epochs": 1,
        "rng_state": rng.bit_generator.state,
        "symmetry_rng_state": None,
        "rank_torch_rng_states": [
            {"rank": 0, "cpu": torch.get_rng_state().tolist(), "cuda": None}
        ],
        "scalar_training_weight_sum": 0.0,
        "categorical_training_weight_sum": 0.0,
    }

    with pytest.raises(SystemExit, match="lacks consumed policy LR-area"):
        tb._restore_training_progress_state(
            progress,
            epochs=2,
            rng=np.random.default_rng(999),
            symmetry_rng=None,
            ddp=_DDP_SINGLE,
            require_policy_objective_lr_area=True,
        )


def test_legacy_sharded_ddp_resume_without_rank_sampler_state_fails() -> None:
    tb = _load_train_bc()
    np = __import__("numpy")
    rng = np.random.default_rng(1)
    progress = {
        "optimizer_step": 7,
        "completed_epochs": 1,
        "rng_state": rng.bit_generator.state,
        "symmetry_rng_state": None,
        "rank_torch_rng_states": [
            {"rank": 0, "cpu": torch.get_rng_state().tolist(), "cuda": None},
            {"rank": 1, "cpu": torch.get_rng_state().tolist(), "cuda": None},
        ],
        "scalar_training_weight_sum": 0.0,
        "categorical_training_weight_sum": 0.0,
        "recipe_identity": {"ddp_shard_data": True},
    }

    with pytest.raises(SystemExit, match="per-rank numpy RNG"):
        tb._restore_training_progress_state(
            progress,
            epochs=2,
            rng=np.random.default_rng(999),
            symmetry_rng=None,
            ddp={"enabled": True, "world_size": 2, "rank": 1, "local_rank": 1},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("amp", "bf16"),
        ("fused_optimizer", True),
        ("graph_history_features", True),
        ("public_award_feature_contract", "authoritative_v1"),
        ("allow_mixed_public_award_feature_contracts", True),
        ("validation_contract_file_sha256", "sha256:contract"),
        ("validation_game_seed_set_sha256", "sha256:validation"),
        ("training_excluded_game_seed_set_sha256", "sha256:excluded"),
        ("teacher_weights", "mcts=2.0"),
        ("phase_weights", "robber=3.0"),
        ("value_phase_weights", "robber=8.0"),
        ("q_skip_teacher_prefixes", ""),
        ("value_root_blend_phases", "PLAY_TURN"),
        ("value_root_blend_global_compat", True),
    ],
)
def test_resume_identity_rejects_gradient_or_precision_recipe_drift(
    field: str, value
) -> None:
    """Adam/RNG/LR state may continue only under the exact learner objective.

    These fields used to be absent from TrainConfig, so changing any one of
    them produced the same resume identity and silently mixed two trajectories.
    """
    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    args = SimpleNamespace(
        grad_accum_steps=1,
        ddp_shard_data=False,
        fsdp=False,
        policy_aux_active_batch_size=0,
    )
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    baseline = TrainConfig(init_checkpoint="first.pt", init_checkpoint_sha256="sha256:a")
    changed = __import__("dataclasses").replace(baseline, **{field: value})

    baseline_identity = tb._training_resume_recipe_identity(baseline, args, ddp)
    changed_identity = tb._training_resume_recipe_identity(changed, args, ddp)

    assert changed_identity != baseline_identity


def _resume_identity_args(**overrides):
    values = {
        "arch": "entity_graph",
        "amp": "none",
        "float32_matmul_precision": None,
        "seed": 1,
        "sampler_seed": None,
        "grad_accum_steps": 1,
        "ddp_shard_data": False,
        "fsdp": False,
        "policy_aux_active_batch_size": 0,
        "policy_dose_lr_area": 0.0,
        "policy_dose_reference_global_batch_size": 0,
        "public_card_lr_mult": 1.0,
        "scalar_value_loss_readout": "raw",
        "scalar_value_loss_scale": 1.0,
        "value_player_outcome_balance_mode": "none",
        "base_sampler": "weighted_replacement_v1",
        "minimum_policy_effective_rows_per_global_batch": 0.0,
        "entity_feature_adapter_version": None,
        "public_rule_state_features": False,
        "value_tower_split_layers": 0,
        "meaningful_public_history": False,
        "event_history_limit": 64,
        "meaningful_public_history_pooling": "masked_mean_v1",
        "meaningful_public_history_target_gather": False,
        "require_feature_learning_signal_modules": "",
        "minimum_feature_learning_signal_observations": 0,
        "train_diagnostics_every_batches": 0,
        "objective_gradient_interference_every_batches": 0,
        "require_only_trainable_prefixes": "",
        "accepted_policy_target_identity_sha256": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("policy_dose_lr_area", 0.0125),
        ("policy_dose_reference_global_batch_size", 2048),
        ("public_card_lr_mult", 4.0),
        ("scalar_value_loss_readout", "deployed_tanh"),
        ("scalar_value_loss_scale", 0.75),
        ("value_player_outcome_balance_mode", "sampler_balanced_v1"),
        ("base_sampler", "coverage_importance_v1"),
        ("minimum_policy_effective_rows_per_global_batch", 32.0),
        (
            "entity_feature_adapter_version",
            "rust_entity_adapter_v5_meaningful_history_v2",
        ),
        ("public_rule_state_features", True),
        ("value_tower_split_layers", 1),
        ("meaningful_public_history", True),
        ("event_history_limit", 32),
        ("meaningful_public_history_pooling", "ordered_attention_v2"),
        ("meaningful_public_history_target_gather", True),
        ("require_feature_learning_signal_modules", "event_encoder,value_head"),
        ("minimum_feature_learning_signal_observations", 2),
        ("train_diagnostics_every_batches", 16),
        ("objective_gradient_interference_every_batches", 16),
        ("require_only_trainable_prefixes", "public_card_count_residual"),
        (
            "accepted_policy_target_identity_sha256",
            ["sha256:" + "a" * 64],
        ),
        ("float32_matmul_precision", "medium"),
    ],
)
def test_resume_identity_rejects_untyped_trajectory_or_admission_drift(
    field: str, changed_value
) -> None:
    """Every live trajectory/admission flag must participate in optimizer resume."""

    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    config = TrainConfig(
        init_checkpoint="epoch1.pt", init_checkpoint_sha256="sha256:a"
    )
    baseline = tb._training_resume_recipe_identity(
        config, _resume_identity_args(), ddp
    )
    changed = tb._training_resume_recipe_identity(
        config, _resume_identity_args(**{field: changed_value}), ddp
    )

    assert changed != baseline


def test_resume_identity_rejects_checkout_code_drift() -> None:
    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    first_binding = {
        "trainer_sha256": "sha256:" + "a" * 64,
        "modules": {
            "catan_zero": {"sha256": "sha256:" + "b" * 64},
            "catan_zero.rl.optim_state": {"sha256": "sha256:" + "c" * 64},
        },
    }
    second_binding = {
        **first_binding,
        "modules": {
            **first_binding["modules"],
            "catan_zero.rl.optim_state": {"sha256": "sha256:" + "d" * 64},
        },
    }

    first = tb._training_resume_recipe_identity(
        TrainConfig(),
        _resume_identity_args(checkout_runtime_binding=first_binding),
        ddp,
    )
    second = tb._training_resume_recipe_identity(
        TrainConfig(),
        _resume_identity_args(checkout_runtime_binding=second_binding),
        ddp,
    )

    assert first["checkout_runtime_code_sha256"] != second[
        "checkout_runtime_code_sha256"
    ]
    assert first != second


def test_resume_progress_rejects_public_card_group_multiplier_4x_to_2x(
    tmp_path: Path,
) -> None:
    """Same optimizer group topology may not silently restore a different group LR."""

    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    config = TrainConfig(
        init_checkpoint="epoch1.pt", init_checkpoint_sha256="sha256:a"
    )
    four_x = tb._training_resume_recipe_identity(
        config, _resume_identity_args(public_card_lr_mult=4.0), ddp
    )
    two_x = tb._training_resume_recipe_identity(
        config, _resume_identity_args(public_card_lr_mult=2.0), ddp
    )
    checkpoint, _, _ = _committed_progress(tmp_path, recipe=four_x)

    assert four_x["public_card_lr_mult"] == 4.0
    assert two_x["public_card_lr_mult"] == 2.0
    with pytest.raises(TrainingProgressError, match="recipe/schedule"):
        load_training_progress(checkpoint, expected_recipe_identity=two_x)


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        (
            "require_feature_learning_signal_modules",
            "value_head,event_encoder,value_head",
            "event_encoder,value_head",
        ),
        (
            "require_only_trainable_prefixes",
            "value_head,public_card_count_residual",
            "public_card_count_residual,value_head",
        ),
        (
            "accepted_policy_target_identity_sha256",
            ["sha256:" + "b" * 64, "sha256:" + "a" * 64],
            [
                "sha256:" + "a" * 64,
                "sha256:" + "b" * 64,
                "sha256:" + "a" * 64,
            ],
        ),
    ],
)
def test_resume_identity_canonicalizes_order_insensitive_admission_sets(
    field: str, first, second
) -> None:
    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    config = TrainConfig()

    assert tb._training_resume_recipe_identity(
        config, _resume_identity_args(**{field: first}), ddp
    ) == tb._training_resume_recipe_identity(
        config, _resume_identity_args(**{field: second}), ddp
    )


def test_resume_identity_infers_v5_history_schema_from_resolved_checkpoint_fields() -> (
    None
):
    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    identity = tb._training_resume_recipe_identity(
        TrainConfig(),
        _resume_identity_args(
            entity_feature_adapter_version=None,
            public_rule_state_features=True,
            meaningful_public_history=True,
            event_history_limit=64,
        ),
        ddp,
    )

    assert (
        identity["entity_feature_adapter_version"]
        == "rust_entity_adapter_v5_meaningful_history_v2"
    )
    assert (
        identity["meaningful_public_history_schema"]
        == "meaningful_public_history_2p_no_trade_v2"
    )


def test_resume_identity_reads_legacy_v2_adapter_from_checkpoint(
    tmp_path: Path,
) -> None:
    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    checkpoint = tmp_path / "legacy-entity.pt"
    torch.save({"config": {}}, checkpoint)
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    identity = tb._training_resume_recipe_identity(
        TrainConfig(init_checkpoint=str(checkpoint)),
        _resume_identity_args(init_checkpoint=str(checkpoint)),
        ddp,
    )

    assert (
        identity["entity_feature_adapter_version"]
        == "rust_entity_adapter_v2_land_topology_ports_maritime"
    )
    assert (
        identity["meaningful_public_history_schema"]
        == "meaningful_public_history_2p_no_trade_v1"
    )


def test_non_entity_resume_does_not_infer_legacy_entity_adapter(
    tmp_path: Path,
) -> None:
    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    checkpoint = tmp_path / "xdim.pt"
    torch.save({"config": {}, "policy_type": "xdim_lite"}, checkpoint)
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    scratch = tb._training_resume_recipe_identity(
        TrainConfig(arch="xdim_lite"),
        _resume_identity_args(arch="xdim_lite"),
        ddp,
    )
    resumed = tb._training_resume_recipe_identity(
        TrainConfig(arch="xdim_lite", init_checkpoint=str(checkpoint)),
        _resume_identity_args(
            arch="xdim_lite", init_checkpoint=str(checkpoint)
        ),
        ddp,
    )

    assert scratch == resumed
    assert resumed["entity_feature_adapter_version"] == ""
    assert resumed["meaningful_public_history_schema"] == ""


def test_resume_identity_normalizes_only_the_checkpoint_being_resumed() -> None:
    """Changing the checkpoint file is expected; changing the recipe is not."""
    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    args = SimpleNamespace(
        grad_accum_steps=1,
        ddp_shard_data=False,
        fsdp=False,
        policy_aux_active_batch_size=0,
    )
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    first = TrainConfig(init_checkpoint="epoch1.pt", init_checkpoint_sha256="sha256:a")
    second = __import__("dataclasses").replace(
        first, init_checkpoint="epoch2.pt", init_checkpoint_sha256="sha256:b"
    )

    assert tb._training_resume_recipe_identity(
        first, args, ddp
    ) == tb._training_resume_recipe_identity(second, args, ddp)


def test_grow_checkpoint_can_resume_through_mutually_exclusive_init_path() -> None:
    """The first grown epoch and its continuation describe one trajectory.

    A continuation cannot repeat --grow-from-checkpoint because train_bc rejects
    combining it with the required --init-checkpoint.  The old recipe identity
    nevertheless retained grow_from_checkpoint, making every such sidecar
    impossible to resume.
    """
    from catan_zero.rl.pipeline_configs import TrainConfig

    tb = _load_train_bc()
    args = SimpleNamespace(
        grad_accum_steps=1,
        ddp_shard_data=False,
        fsdp=False,
        policy_aux_active_batch_size=0,
    )
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    first_epoch = TrainConfig(
        grow_from_checkpoint="f7.pt",
        grow_from_checkpoint_sha256="sha256:f7",
    )
    resumed_epoch = TrainConfig(
        init_checkpoint="grown-epoch1.pt",
        init_checkpoint_sha256="sha256:grown",
    )

    assert tb._training_resume_recipe_identity(
        first_epoch, args, ddp
    ) == tb._training_resume_recipe_identity(resumed_epoch, args, ddp)
