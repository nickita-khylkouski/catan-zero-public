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
    topology = manifest["bc_training_topology"]
    assert topology == {
        "world_size": world_size,
        "rank_local_batch_size": expected_local_batch,
        "grad_accum_steps": 1,
        "effective_global_batch_size": 4096,
        "batch_size_source": "derived_from_global_batch",
        "training_rng_rank_offset": world_size > 1,
    }
    command = _train_command(manifest)
    assert command.count("--training-rng-rank-offset") == int(world_size > 1)
    assert command[command.index("--batch-size") + 1] == str(expected_local_batch)
    assert command[command.index("--grad-accum-steps") + 1] == "1"
    assert command[command.index("--soft-target-weight") + 1] == "1.0"
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


def test_factory_phase_weights_match_production_prompt_vocabulary(tmp_path: Path) -> None:
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
    expected_relative_weights = np.asarray([3.0, 2.0, 2.0, 1.5, 1.0])

    assert policy_weights / policy_weights[-1] == pytest.approx(
        expected_relative_weights
    )
    assert value_weights / value_weights[-1] == pytest.approx(
        expected_relative_weights
    )


def test_factory_accounts_for_gradient_accumulation_in_global_batch(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        "--torchrun-nproc-per-node",
        "2",
        "--bc-grad-accum-steps",
        "4",
    )

    assert result.returncode == 0, result.stderr
    topology = _manifest(tmp_path)["bc_training_topology"]
    assert topology["rank_local_batch_size"] == 512
    assert topology["effective_global_batch_size"] == 4096


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
    }
