from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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
