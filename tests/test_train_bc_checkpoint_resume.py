from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools import train_bc  # noqa: E402
from catan_zero.rl.optim_state import optimizer_sidecar_path  # noqa: E402


_RECIPE_IDENTITY = {
    "schema_version": "train-bc-resume-recipe-v1",
    "normalized_train_config_sha256": "sha256:" + "1" * 64,
    "world_size": 1,
}


def _write_intermediate_snapshot(
    checkpoint: Path,
    *,
    step: int,
    recipe_identity: dict[str, object] = _RECIPE_IDENTITY,
) -> Path:
    path = train_bc._step_checkpoint_path(checkpoint, step)
    torch.save(
        {
            "policy_type": "entity_graph",
            "model": {"weight": torch.tensor([float(step)])},
            "value_training": {
                "optimizer_steps": step,
                "intermediate_checkpoint": {
                    "schema_version": train_bc.INTERMEDIATE_CHECKPOINT_SCHEMA,
                    "optimizer_step": step,
                    "same_training_trajectory": True,
                    "optimizer_sidecar_intentionally_omitted": True,
                },
                "intermediate_checkpoint_trajectory": (
                    train_bc._intermediate_checkpoint_trajectory_binding(
                        optimizer_step=step,
                        resume_recipe_identity=recipe_identity,
                    )
                ),
            },
        },
        path,
    )
    return path


def test_fresh_frontier_retains_immutable_paths_and_step_zero(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    initialization = train_bc._prepare_checkpoint_frontier_start(
        checkpoint,
        checkpoint_steps=(8, 16),
        init_checkpoint=str(tmp_path / "parent.pt"),
        resume_optimizer=False,
    )

    assert initialization == train_bc._step_checkpoint_path(checkpoint, 0)

    train_bc._step_checkpoint_path(checkpoint, 8).write_bytes(b"occupied")
    with pytest.raises(SystemExit, match="intermediate checkpoint path already exists"):
        train_bc._prepare_checkpoint_frontier_start(
            checkpoint,
            checkpoint_steps=(8, 16),
            init_checkpoint=str(tmp_path / "parent.pt"),
            resume_optimizer=False,
        )


def test_optimizer_resume_defers_frontier_and_never_allocates_step_zero(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    train_bc._step_checkpoint_path(checkpoint, 8).write_bytes(b"past evidence")

    assert (
        train_bc._prepare_checkpoint_frontier_start(
            checkpoint,
            checkpoint_steps=(8, 16),
            init_checkpoint=str(tmp_path / "epoch0001.pt"),
            resume_optimizer=True,
        )
        is None
    )
    assert not train_bc._step_checkpoint_path(checkpoint, 0).exists()


def test_resume_authenticates_past_steps_and_reserves_future_paths(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    _write_intermediate_snapshot(checkpoint, step=8)
    _write_intermediate_snapshot(checkpoint, step=16)

    verified = train_bc._verify_resumed_checkpoint_frontier(
        checkpoint,
        checkpoint_steps=(8, 16, 32, 64),
        restored_global_step=20,
        resume_recipe_identity=_RECIPE_IDENTITY,
    )

    assert verified == {8, 16}


def test_resume_refuses_missing_or_wrong_trajectory_past_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    with pytest.raises(SystemExit, match="missing an immutable past"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
        )

    _write_intermediate_snapshot(
        checkpoint,
        step=8,
        recipe_identity={**_RECIPE_IDENTITY, "world_size": 2},
    )
    with pytest.raises(SystemExit, match="step/trajectory metadata mismatch"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
        )


def test_resume_refuses_occupied_future_snapshot(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    future = train_bc._step_checkpoint_path(checkpoint, 32)
    future.write_bytes(b"unrelated")

    with pytest.raises(SystemExit, match="future intermediate checkpoint path"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(32,),
            restored_global_step=16,
            resume_recipe_identity=_RECIPE_IDENTITY,
        )

    future.unlink()
    optimizer_sidecar_path(future).write_bytes(b"stale optimizer sidecar")
    with pytest.raises(SystemExit, match="future intermediate checkpoint path"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(32,),
            restored_global_step=16,
            resume_recipe_identity=_RECIPE_IDENTITY,
        )
