from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import train_bc


REPO_ROOT = Path(__file__).resolve().parents[1]


def _dry_run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in ("src", env.get("PYTHONPATH", "")) if value
    )
    return subprocess.run(
        [
            sys.executable,
            "tools/start_training_factory.py",
            "--run-dir",
            str(tmp_path),
            "--dry-run",
            *extra,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _manifest(path: Path) -> dict:
    return json.loads((path / "pipeline_manifest.json").read_text(encoding="utf-8"))


def _train_command(manifest: dict) -> list[str]:
    return next(
        command for command in manifest["commands"] if "tools/train_bc.py" in command
    )


@pytest.mark.parametrize(
    ("world_size", "expected_local_batch"),
    ((1, 4096), (2, 2048), (8, 512)),
)
def test_factory_keeps_default_global_batch_constant_across_world_sizes(
    tmp_path: Path, world_size: int, expected_local_batch: int
) -> None:
    result = _dry_run(
        tmp_path,
        "--torchrun-nproc-per-node",
        str(world_size),
    )

    assert result.returncode == 0, result.stderr
    manifest = _manifest(tmp_path)
    assert manifest["production_hard_action_admission"] == {
        "blocked": True,
        "producer": "classical_policy_on_authoritative_game_v1",
        "student_information_scope": "public_information_set",
    }
    topology = manifest["bc_training_topology"]
    assert topology == {
        "world_size": world_size,
        "rank_local_batch_size": expected_local_batch,
        "grad_accum_steps": 1,
        "effective_global_batch_size": 4096,
        "batch_size_source": "derived_from_global_batch",
        "training_rng_rank_offset": world_size > 1,
        "max_optimizer_steps": 128,
        "exact_max_optimizer_steps": True,
        "optimizer": "adam",
        "value_lr_mult": 1.0,
        "trunk_lr_mult": 1.0,
        "value_trunk_grad_scale": 1.0,
        "policy_kl_target": None,
    }
    command = _train_command(manifest)
    assert command.count("--training-rng-rank-offset") == int(world_size > 1)
    assert command[command.index("--batch-size") + 1] == str(expected_local_batch)
    assert command[command.index("--grad-accum-steps") + 1] == "1"
    assert command[command.index("--max-steps") + 1] == "128"
    assert command[command.index("--value-lr-mult") + 1] == "1.0"
    assert command[command.index("--soft-target-weight") + 1] == "0.0"
    assert command[
        command.index("--policy-target-blend-semantics") + 1
    ] == "policy_target_fallback_v2"


def test_factory_can_render_explicit_legacy_replay_semantics(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        "--soft-target-weight",
        "0.9",
        "--policy-target-blend-semantics",
        "legacy_interpolate_v1",
    )

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    assert command[command.index("--soft-target-weight") + 1] == "0.9"
    assert command[
        command.index("--policy-target-blend-semantics") + 1
    ] == "legacy_interpolate_v1"


def test_factory_keeps_forced_policy_rows_out_of_fresh_training(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    assert command[command.index("--forced-action-weight") + 1] == "0.0"


def test_factory_converts_entity_tokens_and_masks_hidden_inputs(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = _manifest(tmp_path)
    commands = manifest["commands"]
    conversion = next(
        command
        for command in commands
        if "tools/convert_teacher_to_entity_tokens.py" in command
    )
    training = _train_command(manifest)
    conversion_index = commands.index(conversion)
    training_index = commands.index(training)

    assert conversion_index < training_index
    assert conversion[conversion.index("--data") + 1] == str(
        tmp_path / "teacher_data"
    )
    assert conversion[conversion.index("--out") + 1] == str(
        tmp_path / "teacher_data_entity"
    )
    assert "--graph-history-features" in conversion
    assert training[training.index("--data") + 1] == str(
        tmp_path / "teacher_data_entity"
    )
    assert "--graph-history-features" in training
    assert "--mask-hidden-info" in training
    assert manifest["bc_training_data"] == {
        "curated": str(tmp_path / "teacher_data"),
        "entity_converted": str(tmp_path / "teacher_data_entity"),
        "effective": str(tmp_path / "teacher_data_entity"),
        "graph_history_features": True,
        "mask_hidden_info": True,
        "acknowledge_authoritative_hard_action_targets": False,
        "soft_target_weight": 0.0,
        "target_reliability_confidence_weighting": False,
        "target_reliability_confidence_floor": 0.25,
    }


def test_factory_can_render_explicit_omniscient_soft_target_replay(
    tmp_path: Path,
) -> None:
    result = _dry_run(
        tmp_path,
        "--no-mask-hidden-info",
        "--soft-target-weight",
        "1.0",
    )

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    assert "--mask-hidden-info" not in command
    assert command[command.index("--soft-target-weight") + 1] == "1.0"


def test_factory_forwards_explicit_diagnostic_hard_target_acknowledgement(
    tmp_path: Path,
) -> None:
    result = _dry_run(
        tmp_path,
        "--quality-gate",
        "none",
        "--acknowledge-authoritative-hard-action-targets",
    )

    assert result.returncode == 0, result.stderr
    manifest = _manifest(tmp_path)
    assert (
        manifest["bc_training_data"][
            "acknowledge_authoritative_hard_action_targets"
        ]
        is True
    )
    assert "--acknowledge-authoritative-hard-action-targets" in _train_command(
        manifest
    )


def test_factory_rejects_hard_target_acknowledgement_in_production(
    tmp_path: Path,
) -> None:
    result = _dry_run(
        tmp_path,
        "--acknowledge-authoritative-hard-action-targets",
    )

    assert result.returncode != 0
    assert "diagnostic-only" in result.stderr


def test_factory_rejects_masked_classical_production_before_any_subprocess(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "blocked"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in ("src", env.get("PYTHONPATH", "")) if value
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/start_training_factory.py",
            "--run-dir",
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode != 0
    assert "refused before data generation" in result.stderr
    assert not run_dir.exists()


def test_non_entity_factory_skips_entity_conversion_and_masking(
    tmp_path: Path,
) -> None:
    result = _dry_run(tmp_path, "--arch", "xdim_lite")

    assert result.returncode == 0, result.stderr
    manifest = _manifest(tmp_path)
    assert not any(
        "tools/convert_teacher_to_entity_tokens.py" in command
        for command in manifest["commands"]
    )
    command = _train_command(manifest)
    assert command[command.index("--data") + 1] == str(tmp_path / "teacher_data")
    assert "--mask-hidden-info" not in command


def test_factory_trains_the_scalar_readout_deployed_by_search(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    train_argv = command[command.index("tools/train_bc.py") + 1 :]
    resolved = train_bc.build_parser().parse_args(train_argv)

    assert resolved.scalar_value_loss_readout == "deployed_tanh"
    assert resolved.scalar_value_loss_scale == pytest.approx(1.0)


def test_factory_uses_public_equal_game_training_contract(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    train_argv = command[command.index("tools/train_bc.py") + 1 :]
    resolved = train_bc.build_parser().parse_args(train_argv)

    assert resolved.mask_hidden_info is True
    assert resolved.per_game_policy_weight is True
    assert resolved.per_game_policy_weight_mode == "equal"
    assert resolved.per_game_value_weight is True
    assert resolved.per_game_value_weight_mode == "equal"
    assert resolved.lr_warmup_steps == 16
    assert resolved.max_steps == 128
    assert resolved.exact_max_steps is True
    assert resolved.value_lr_mult == pytest.approx(1.0)
    assert resolved.trunk_lr_mult == pytest.approx(1.0)
    assert resolved.value_trunk_grad_scale == pytest.approx(1.0)
    assert resolved.resume_optimizer is False


def test_factory_forwards_explicit_trust_and_value_controls(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        "--value-lr-mult",
        "1.5",
        "--trunk-lr-mult",
        "0.25",
        "--value-trunk-grad-scale",
        "0.5",
        "--policy-kl-anchor-weight",
        "0.02",
        "--policy-kl-target",
        "0.03",
        "--policy-kl-dual-lr",
        "0.5",
        "--policy-kl-max-weight",
        "0.8",
    )

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    train_argv = command[command.index("tools/train_bc.py") + 1 :]
    resolved = train_bc.build_parser().parse_args(train_argv)
    assert resolved.value_lr_mult == pytest.approx(1.5)
    assert resolved.trunk_lr_mult == pytest.approx(0.25)
    assert resolved.value_trunk_grad_scale == pytest.approx(0.5)
    assert resolved.policy_kl_anchor_weight == pytest.approx(0.02)
    assert resolved.policy_kl_target == pytest.approx(0.03)
    assert resolved.policy_kl_dual_lr == pytest.approx(0.5)
    assert resolved.policy_kl_max_weight == pytest.approx(0.8)


def test_factory_forwards_target_reliability_weighting(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        "--target-reliability-confidence-weighting",
        "--target-reliability-confidence-floor",
        "0.4",
    )

    assert result.returncode == 0, result.stderr
    manifest = _manifest(tmp_path)
    command = _train_command(manifest)
    train_argv = command[command.index("tools/train_bc.py") + 1 :]
    resolved = train_bc.build_parser().parse_args(train_argv)
    assert resolved.target_reliability_confidence_weighting is True
    assert resolved.target_reliability_confidence_floor == pytest.approx(0.4)
    assert manifest["bc_training_data"][
        "target_reliability_confidence_weighting"
    ] is True
    assert manifest["bc_training_data"][
        "target_reliability_confidence_floor"
    ] == pytest.approx(0.4)


def test_production_factory_refuses_unbounded_epoch_only_training(
    tmp_path: Path,
) -> None:
    result = _dry_run(tmp_path, "--bc-max-steps", "0")

    assert result.returncode != 0
    assert "positive --bc-max-steps" in result.stderr

    replay = _dry_run(
        tmp_path / "replay",
        "--quality-gate",
        "none",
        "--bc-max-steps",
        "0",
    )
    assert replay.returncode == 0, replay.stderr
    command = _train_command(_manifest(tmp_path / "replay"))
    assert command[command.index("--max-steps") + 1] == "0"


def test_factory_explicitly_binds_guarded_optimizer_and_hard_target_recipe(
    tmp_path: Path,
) -> None:
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    expected = {
        "--optimizer": "adam",
        "--weight-decay": "0.0",
        "--truncated-vp-margin-value-weight": "0.25",
        "--lr-schedule": "flat",
        "--soft-target-weight": "0.0",
    }
    for flag, value in expected.items():
        assert command.count(flag) == 1
        assert command[command.index(flag) + 1] == value

    train_argv = command[command.index("tools/train_bc.py") + 1 :]
    resolved = train_bc.build_parser().parse_args(train_argv)
    unknown_teacher_rows = {
        "action_taken": np.asarray([0, 1], dtype=np.int16),
        "target_policy": np.asarray([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32),
        "target_scores": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "target_information_regime": np.asarray(["unknown", "unknown"]),
    }
    admission = train_bc._validate_target_information_admission(
        unknown_teacher_rows,
        mask_hidden_info=resolved.mask_hidden_info,
        soft_target_weight=resolved.soft_target_weight,
        policy_target_blend_semantics=resolved.policy_target_blend_semantics,
        policy_loss_weight=resolved.policy_loss_weight,
        q_loss_weight=resolved.q_loss_weight,
        value_target_lambda=resolved.value_target_lambda,
        policy_kl_anchor_weight=resolved.policy_kl_anchor_weight,
        policy_surprise_weight=resolved.policy_surprise_weight,
        required_target_information_regime=(
            resolved.required_target_information_regime
        ),
    )
    assert admission["unsafe_or_unknown_rows"] == 2
    assert admission["search_target_objectives"] == []


def test_factory_none_quality_gate_disables_implicit_35m_teacher_gate(
    tmp_path: Path,
) -> None:
    result = _dry_run(tmp_path, "--quality-gate", "none")

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    assert "--skip-teacher-quality-gate" in command
    assert "--require-strict-35m-teacher" not in command
    assert "--require-production-35m-teacher" not in command
    assert "--exact-max-steps" not in command
    assert "--require-35m-model" in command


def test_factory_converts_teacher_rows_before_entity_training(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = _manifest(tmp_path)
    converter_index = next(
        index
        for index, command in enumerate(manifest["commands"])
        if "tools/convert_teacher_to_entity_tokens.py" in command
    )
    train_index = next(
        index
        for index, command in enumerate(manifest["commands"])
        if "tools/train_bc.py" in command
    )
    converter = manifest["commands"][converter_index]
    command = manifest["commands"][train_index]

    assert converter_index < train_index
    assert converter[converter.index("--data") + 1] == str(tmp_path / "teacher_data")
    assert converter[converter.index("--out") + 1] == str(
        tmp_path / "teacher_data_entity"
    )
    assert command[command.index("--data") + 1] == str(
        tmp_path / "teacher_data_entity"
    )


def test_factory_can_replay_the_legacy_raw_scalar_readout(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        "--scalar-value-loss-readout",
        "raw",
        "--scalar-value-loss-scale",
        "2.0",
    )

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    assert command[command.index("--scalar-value-loss-readout") + 1] == "raw"
    assert command[command.index("--scalar-value-loss-scale") + 1] == "2.0"


def test_factory_defaults_to_unbiased_phase_mass(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    configured = train_bc._parse_weight_map(
        command[command.index("--phase-weights") + 1]
    )
    phases = np.asarray(
        [
            "MOVE_ROBBER",
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "DISCARD",
            "PLAY_TURN",
        ]
    )
    data = {
        "action_taken": np.arange(len(phases), dtype=np.int16),
        "legal_action_ids": np.tile(
            np.asarray([[0, 1]], dtype=np.int16), (len(phases), 1)
        ),
        "phase": phases,
    }

    policy_weights = train_bc.build_sample_weights(
        data,
        teacher_weights={},
        phase_weights=configured,
        forced_action_weight=1.0,
        winner_sample_weight=1.0,
        loser_sample_weight=1.0,
        vp_margin_weight=0.0,
        vps_to_win=10,
    )
    value_weights = train_bc.build_value_sample_weights(
        data, phase_weights=configured
    )
    expected_relative_weights = np.ones(len(phases))

    assert configured == {}
    assert policy_weights / policy_weights[-1] == pytest.approx(
        expected_relative_weights
    )
    assert value_weights / value_weights[-1] == pytest.approx(
        expected_relative_weights
    )


def test_factory_accepts_explicit_current_prompt_phase_treatment(
    tmp_path: Path,
) -> None:
    treatment = (
        "MOVE_ROBBER=3.0,BUILD_INITIAL_SETTLEMENT=2.0,"
        "BUILD_INITIAL_ROAD=2.0,DISCARD=1.5"
    )
    result = _dry_run(tmp_path, "--phase-weights", treatment)

    assert result.returncode == 0, result.stderr
    command = _train_command(_manifest(tmp_path))
    assert command[command.index("--phase-weights") + 1] == treatment


def test_factory_none_quality_gate_accounts_for_accumulation_in_global_batch(
    tmp_path: Path,
) -> None:
    result = _dry_run(
        tmp_path,
        "--quality-gate",
        "none",
        "--torchrun-nproc-per-node",
        "2",
        "--bc-grad-accum-steps",
        "4",
    )

    assert result.returncode == 0, result.stderr
    topology = _manifest(tmp_path)["bc_training_topology"]
    assert topology["rank_local_batch_size"] == 512
    assert topology["effective_global_batch_size"] == 4096


def test_factory_refuses_approximate_gradient_accumulation_in_production(
    tmp_path: Path,
) -> None:
    result = _dry_run(tmp_path, "--bc-grad-accum-steps", "2")

    assert result.returncode != 0
    assert "exact union-weighted gradient accumulation" in result.stderr
    assert "--quality-gate none" in result.stderr
    assert not (tmp_path / "pipeline_manifest.json").exists()


def test_factory_refuses_non_divisible_global_batch(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        "--torchrun-nproc-per-node",
        "8",
        "--bc-global-batch-size",
        "4097",
    )

    assert result.returncode != 0
    assert "must be divisible" in result.stderr
    assert not (tmp_path / "pipeline_manifest.json").exists()


def test_explicit_local_batch_override_is_manifested_without_reinterpretation(
    tmp_path: Path,
) -> None:
    result = _dry_run(
        tmp_path,
        "--quality-gate",
        "none",
        "--torchrun-nproc-per-node",
        "2",
        "--bc-grad-accum-steps",
        "2",
        "--bc-batch-size",
        "1024",
    )

    assert result.returncode == 0, result.stderr
    topology = _manifest(tmp_path)["bc_training_topology"]
    assert topology == {
        "world_size": 2,
        "rank_local_batch_size": 1024,
        "grad_accum_steps": 2,
        "effective_global_batch_size": 4096,
        "batch_size_source": "explicit_rank_local_override",
        "training_rng_rank_offset": True,
        "max_optimizer_steps": 128,
        "exact_max_optimizer_steps": False,
        "optimizer": "adam",
        "value_lr_mult": 1.0,
        "trunk_lr_mult": 1.0,
        "value_trunk_grad_scale": 1.0,
        "policy_kl_target": None,
    }
