from __future__ import annotations

from collections import Counter
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools import train_bc  # noqa: E402
from catan_zero.rl.optim_state import (  # noqa: E402
    checkpoint_frontier_sidecar_path,
    load_training_progress,
    optimizer_sidecar_path,
    save_training_progress,
    training_progress_sidecar_path,
)


_RECIPE_IDENTITY = {
    "schema_version": "train-bc-resume-recipe-v1",
    "normalized_train_config_sha256": "sha256:" + "1" * 64,
    "world_size": 1,
}


def _write_intermediate_snapshot(
    checkpoint: Path,
    *,
    step: int,
    trajectory_root: dict[str, str],
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
                        trajectory_root_sha256=trajectory_root[
                            "trajectory_root_sha256"
                        ],
                    )
                ),
            },
        },
        path,
    )
    return path


def _write_model_checkpoint(path: Path, value: float) -> None:
    torch.save({"model": {"weight": torch.tensor([value])}}, path)


def _frontier_evidence(
    checkpoint: Path,
    *,
    steps: tuple[int, ...],
    trajectory_root: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    inventory: list[dict[str, object]] = []
    doses: list[dict[str, object]] = []
    holdouts: list[dict[str, object]] = []
    for step in steps:
        snapshot = _write_intermediate_snapshot(
            checkpoint,
            step=step,
            trajectory_root=trajectory_root,
        )
        snapshot_sha256 = train_bc._sha256_existing_file(snapshot)
        inventory.append(
            {
                "schema_version": train_bc.INTERMEDIATE_CHECKPOINT_SCHEMA,
                "optimizer_step": step,
                "checkpoint": str(snapshot),
                "checkpoint_sha256": snapshot_sha256,
                "size_bytes": snapshot.stat().st_size,
                "same_training_trajectory": True,
                "optimizer_sidecar": None,
            }
        )
        doses.append(
            {
                "schema_version": train_bc.CHECKPOINT_DOSE_TELEMETRY_SCHEMA,
                "optimizer_step": step,
                "effective_policy_lr_area": float(step),
            }
        )
        holdouts.append(
            {
                "schema_version": train_bc.CHECKPOINT_HOLDOUT_SCHEMA,
                "optimizer_step": step,
                "checkpoint": str(snapshot),
                "checkpoint_sha256": snapshot_sha256,
                "measure": "report_bound_raw_validation_rows",
                "validation_game_seed_set_sha256": None,
                "metrics": {"policy_loss": 1.0 / step},
            }
        )
    return inventory, doses, holdouts


def _committed_frontier(
    checkpoint: Path,
    *,
    owner_checkpoint: Path,
    requested_steps: tuple[int, ...],
    passed_steps: tuple[int, ...],
    owner_step: int,
    resume_telemetry_state: dict[str, object] | None = None,
    with_initialization_reference: bool = False,
) -> tuple[dict[str, object], dict[str, str]]:
    initialization_reference = None
    initial_model_state_sha256 = "sha256:" + "0" * 64
    if with_initialization_reference:
        initialization_path = train_bc._step_checkpoint_path(checkpoint, 0)
        _write_model_checkpoint(initialization_path, -1.0)
        initial_model_state_sha256 = (
            train_bc._checkpoint_model_tensor_state_sha256(initialization_path)
        )
        initialization_reference = {
            "schema_version": train_bc.EFFECTIVE_INITIALIZATION_REFERENCE_SCHEMA,
            "optimizer_step": 0,
            "checkpoint": str(initialization_path),
            "checkpoint_sha256": train_bc._sha256_existing_file(
                initialization_path
            ),
            "size_bytes": initialization_path.stat().st_size,
            "public_award_feature_contract": "fixture-contract",
            "same_training_trajectory": True,
            "holdout_metrics": {"policy_loss": 0.75, "accuracy": 0.25},
        }
    trajectory_root = train_bc._checkpoint_frontier_trajectory_root(
        resume_recipe_identity=_RECIPE_IDENTITY,
        initial_model_state_sha256=initial_model_state_sha256,
    )
    inventory, doses, holdouts = _frontier_evidence(
        checkpoint,
        steps=passed_steps,
        trajectory_root=trajectory_root,
    )
    _write_model_checkpoint(owner_checkpoint, float(owner_step))
    journal_path = train_bc._save_checkpoint_frontier_journal(
        owner_checkpoint,
        checkpoint_steps=requested_steps,
        intermediate_checkpoints=inventory,
        checkpoint_dose_snapshots=doses,
        checkpoint_holdout_snapshots=holdouts,
        effective_initialization_reference=initialization_reference,
        trajectory_root=trajectory_root,
        resume_recipe_identity=_RECIPE_IDENTITY,
        validation_game_seed_set_sha256=None,
        resume_telemetry_state=(
            resume_telemetry_state
            or train_bc._empty_checkpoint_resume_telemetry_state()
        ),
        owner_checkpoint=owner_checkpoint,
        owner_optimizer_step=owner_step,
        ddp={"rank": 0},
    )
    assert journal_path is not None
    progress = {
        "checkpoint_frontier": {
            "path": journal_path.name,
            "sha256": train_bc._sha256_existing_file(journal_path),
        }
    }
    return progress, trajectory_root


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
    owner = tmp_path / "candidate.epoch0001.pt"
    progress, trajectory_root = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8, 16, 32, 64),
        passed_steps=(8, 16),
        owner_step=20,
    )

    verified = train_bc._verify_resumed_checkpoint_frontier(
        checkpoint,
        checkpoint_steps=(8, 16, 32, 64),
        restored_global_step=20,
        resume_recipe_identity=_RECIPE_IDENTITY,
        resume_progress=progress,
        resume_checkpoint=owner,
        validation_game_seed_set_sha256=None,
    )

    assert verified["saved_checkpoint_steps"] == {8, 16}
    assert [
        record["optimizer_step"]
        for record in verified["intermediate_checkpoints"]
    ] == [8, 16]
    assert [
        record["optimizer_step"]
        for record in verified["checkpoint_dose_snapshots"]
    ] == [8, 16]
    assert [
        record["optimizer_step"]
        for record in verified["checkpoint_holdout_snapshots"]
    ] == [8, 16]
    assert verified["trajectory_root"] == trajectory_root


def test_resume_to_different_output_reseeds_authenticated_snapshot_paths(
    tmp_path: Path,
) -> None:
    old_output = tmp_path / "old-candidate.pt"
    new_output = tmp_path / "new-candidate.pt"
    owner = tmp_path / "old-candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        old_output,
        owner_checkpoint=owner,
        requested_steps=(8, 16, 32),
        passed_steps=(8, 16),
        owner_step=20,
    )

    verified = train_bc._verify_resumed_checkpoint_frontier(
        new_output,
        checkpoint_steps=(8, 16, 32),
        restored_global_step=20,
        resume_recipe_identity=_RECIPE_IDENTITY,
        resume_progress=progress,
        resume_checkpoint=owner,
        validation_game_seed_set_sha256=None,
    )

    assert verified["saved_checkpoint_steps"] == {8, 16}
    assert [
        Path(record["checkpoint"]).name
        for record in verified["intermediate_checkpoints"]
    ] == [
        train_bc._step_checkpoint_path(new_output, step).name
        for step in (8, 16)
    ]
    for step in (8, 16):
        old_snapshot = train_bc._step_checkpoint_path(old_output, step)
        new_snapshot = train_bc._step_checkpoint_path(new_output, step)
        assert new_snapshot.is_file()
        assert new_snapshot.read_bytes() == old_snapshot.read_bytes()
    new_owner = tmp_path / "new-candidate.epoch0002.pt"
    _write_model_checkpoint(new_owner, 32.0)
    assert train_bc._save_checkpoint_frontier_journal(
        new_owner,
        checkpoint_steps=(8, 16, 32),
        intermediate_checkpoints=verified["intermediate_checkpoints"],
        checkpoint_dose_snapshots=verified["checkpoint_dose_snapshots"],
        checkpoint_holdout_snapshots=verified["checkpoint_holdout_snapshots"],
        effective_initialization_reference=None,
        trajectory_root=verified["trajectory_root"],
        resume_recipe_identity=_RECIPE_IDENTITY,
        validation_game_seed_set_sha256=None,
        resume_telemetry_state=verified["resume_telemetry_state"],
        owner_checkpoint=new_owner,
        owner_optimizer_step=20,
        ddp={"rank": 0},
    ) == checkpoint_frontier_sidecar_path(new_owner)


def test_failed_source_admission_does_not_reseed_new_output(
    tmp_path: Path,
) -> None:
    old_output = tmp_path / "old-candidate.pt"
    new_output = tmp_path / "new-candidate.pt"
    owner = tmp_path / "old-candidate.epoch0001.pt"
    trajectory_root = train_bc._checkpoint_frontier_trajectory_root(
        resume_recipe_identity=_RECIPE_IDENTITY,
        initial_model_state_sha256="sha256:" + "0" * 64,
    )
    inventory, doses, holdouts = _frontier_evidence(
        old_output,
        steps=(8,),
        trajectory_root=trajectory_root,
    )
    source = train_bc._step_checkpoint_path(old_output, 8)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["value_training"]["intermediate_checkpoint_trajectory"][
        "trajectory_root_sha256"
    ] = "sha256:" + "f" * 64
    torch.save(payload, source)
    source_sha256 = train_bc._sha256_existing_file(source)
    inventory[0]["checkpoint_sha256"] = source_sha256
    inventory[0]["size_bytes"] = source.stat().st_size
    holdouts[0]["checkpoint_sha256"] = source_sha256
    _write_model_checkpoint(owner, 8.0)
    journal = train_bc._save_checkpoint_frontier_journal(
        owner,
        checkpoint_steps=(8, 16),
        intermediate_checkpoints=inventory,
        checkpoint_dose_snapshots=doses,
        checkpoint_holdout_snapshots=holdouts,
        effective_initialization_reference=None,
        trajectory_root=trajectory_root,
        resume_recipe_identity=_RECIPE_IDENTITY,
        validation_game_seed_set_sha256=None,
        resume_telemetry_state=(
            train_bc._empty_checkpoint_resume_telemetry_state()
        ),
        owner_checkpoint=owner,
        owner_optimizer_step=8,
        ddp={"rank": 0},
    )
    assert journal is not None
    progress = {
        "checkpoint_frontier": {
            "path": journal.name,
            "sha256": train_bc._sha256_existing_file(journal),
        }
    }

    with pytest.raises(SystemExit, match="step/trajectory metadata mismatch"):
        train_bc._verify_resumed_checkpoint_frontier(
            new_output,
            checkpoint_steps=(8, 16),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )

    assert not train_bc._step_checkpoint_path(new_output, 8).exists()


def test_resume_owner_identity_is_stable_across_relative_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
    )
    monkeypatch.chdir(tmp_path)

    verified = train_bc._verify_resumed_checkpoint_frontier(
        Path("candidate.pt"),
        checkpoint_steps=(8,),
        restored_global_step=8,
        resume_recipe_identity=_RECIPE_IDENTITY,
        resume_progress=progress,
        resume_checkpoint=Path(owner.name),
        validation_game_seed_set_sha256=None,
    )

    assert verified["saved_checkpoint_steps"] == {8}


def test_resume_rehydrates_complete_final_report_inventory(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8, 16, 20),
        passed_steps=(8, 16),
        owner_step=18,
    )
    resumed = train_bc._verify_resumed_checkpoint_frontier(
        checkpoint,
        checkpoint_steps=(8, 16, 20),
        restored_global_step=18,
        resume_recipe_identity=_RECIPE_IDENTITY,
        resume_progress=progress,
        resume_checkpoint=owner,
        validation_game_seed_set_sha256=None,
    )

    terminal_step = 20
    terminal_dose = {
        "schema_version": train_bc.CHECKPOINT_DOSE_TELEMETRY_SCHEMA,
        "optimizer_step": terminal_step,
    }
    terminal_holdout = {
        "schema_version": train_bc.CHECKPOINT_HOLDOUT_SCHEMA,
        "optimizer_step": terminal_step,
    }
    final_report_inventory = {
        "intermediate_checkpoints": resumed["intermediate_checkpoints"],
        "checkpoint_dose_trajectory": [
            *resumed["checkpoint_dose_snapshots"],
            terminal_dose,
        ],
        "checkpoint_holdout_frontier": [
            *resumed["checkpoint_holdout_snapshots"],
            terminal_holdout,
        ],
    }

    assert [
        record["optimizer_step"]
        for record in final_report_inventory["intermediate_checkpoints"]
    ] == [8, 16]
    assert [
        record["optimizer_step"]
        for record in final_report_inventory["checkpoint_dose_trajectory"]
    ] == [8, 16, 20]
    assert [
        record["optimizer_step"]
        for record in final_report_inventory["checkpoint_holdout_frontier"]
    ] == [8, 16, 20]


def test_resume_telemetry_keeps_future_dose_cumulative(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    prior_metric = {
        "samples": 160,
        "policy_base_active_rows": 150,
        "policy_aux_active_rows": 0,
        "value_active_rows": 120,
    }
    telemetry = train_bc._empty_checkpoint_resume_telemetry_state()
    reuse_state = train_bc._policy_aux_source_reuse_resume_state(  # noqa: SLF001
        Counter({2: 2, 9: 1}),
        game_identities={(0, 101), (0, 102)},
        ddp={"enabled": False, "world_size": 1, "rank": 0},
        data_sharded=False,
    )
    epoch_cycle = {
        "epoch": 1,
        **train_bc._policy_aux_sampling_cycle_report(  # noqa: SLF001
            np.asarray([1.0, 2.0, 3.0]),
            local_draws=3,
            ddp={"enabled": False, "world_size": 1, "rank": 0},
            mode=train_bc.POLICY_AUX_SAMPLING_WEIGHTED_CYCLES_V1,
            global_draw_offset=0,
        ),
    }
    telemetry.update(
        {
            "metrics": [prior_metric],
            "optimizer_observed_steps": 16,
            "dose_microbatch_number": 16,
            "policy_dose_cutoff_optimizer_step": 8,
            "policy_aux_source_reuse_state": reuse_state,
            "policy_aux_epoch_cycles": [epoch_cycle],
        }
    )
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8, 16, 20),
        passed_steps=(8, 16),
        owner_step=16,
        resume_telemetry_state=telemetry,
    )
    resumed = train_bc._verify_resumed_checkpoint_frontier(
        checkpoint,
        checkpoint_steps=(8, 16, 20),
        restored_global_step=16,
        resume_recipe_identity=_RECIPE_IDENTITY,
        resume_progress=progress,
        resume_checkpoint=owner,
        validation_game_seed_set_sha256=None,
    )
    prior_dose = train_bc._checkpoint_dose_telemetry(
        [prior_metric],
        optimizer_step=16,
        optimizer_observed_steps=16,
        optimizer_clipped_steps=0,
        optimizer_zero_objective_steps=0,
        optimizer_pre_clip_grad_norm_sum=0.0,
        optimizer_pre_clip_grad_norm_max=0.0,
        objective_gradient_cadence_batches=0,
        train_diagnostic_cadence_batches=0,
        public_card_enabled=False,
        meaningful_history_enabled=False,
    )
    resumed_segment = {
        "samples": 40,
        "policy_base_active_rows": 38,
        "policy_aux_active_rows": 0,
        "value_active_rows": 30,
    }
    restored_metrics = [
        *resumed["resume_telemetry_state"]["metrics"],
        resumed_segment,
    ]
    assert resumed["resume_telemetry_state"]["dose_microbatch_number"] == 16
    assert (
        resumed["resume_telemetry_state"][
            "policy_dose_cutoff_optimizer_step"
        ]
        == 8
    )
    assert (
        resumed["resume_telemetry_state"]["policy_aux_source_reuse_state"]
        == reuse_state
    )
    assert resumed["resume_telemetry_state"]["policy_aux_epoch_cycles"] == [
        epoch_cycle
    ]
    restored_counts, restored_games = (
        train_bc._restore_policy_aux_source_reuse_resume_state(  # noqa: SLF001
            resumed["resume_telemetry_state"][
                "policy_aux_source_reuse_state"
            ],
            ddp={"enabled": False, "world_size": 1, "rank": 0},
            data_sharded=False,
            expected_global_draws=3,
        )
    )
    assert restored_counts == Counter({2: 2, 9: 1})
    assert restored_games == {(0, 101), (0, 102)}
    assert not any(
        key.startswith("optimizer_lr_")
        or key == "optimizer_schedule_multiplier_sum"
        for key in resumed["resume_telemetry_state"]
    )
    future_dose = train_bc._checkpoint_dose_telemetry(
        restored_metrics,
        optimizer_step=20,
        optimizer_observed_steps=20,
        optimizer_clipped_steps=0,
        optimizer_zero_objective_steps=0,
        optimizer_pre_clip_grad_norm_sum=0.0,
        optimizer_pre_clip_grad_norm_max=0.0,
        objective_gradient_cadence_batches=0,
        train_diagnostic_cadence_batches=0,
        public_card_enabled=False,
        meaningful_history_enabled=False,
    )

    assert prior_dose["training_row_draws"]["total_training_row_draws"] == 160
    assert future_dose["training_row_draws"]["total_training_row_draws"] == 200
    assert (
        future_dose["training_row_draws"]["total_training_row_draws"]
        > prior_dose["training_row_draws"]["total_training_row_draws"]
    )


def test_frontier_refuses_policy_dose_cutoff_after_owner_step(
    tmp_path: Path,
) -> None:
    telemetry = train_bc._empty_checkpoint_resume_telemetry_state()
    telemetry["policy_dose_cutoff_optimizer_step"] = 17

    with pytest.raises(
        RuntimeError,
        match="checkpoint frontier resume telemetry is malformed",
    ):
        _committed_frontier(
            tmp_path / "candidate.pt",
            owner_checkpoint=tmp_path / "candidate.epoch0001.pt",
            requested_steps=(),
            passed_steps=(),
            owner_step=16,
            resume_telemetry_state=telemetry,
        )


def test_checkpoint_bundle_relocation_preserves_exact_frontier(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-host"
    destination = tmp_path / "destination-host"
    source.mkdir()
    checkpoint = source / "candidate.pt"
    owner = source / "candidate.epoch0001.pt"
    _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8, 16, 32),
        passed_steps=(8, 16),
        owner_step=20,
    )
    optimizer_sidecar_path(owner).write_bytes(b"authenticated optimizer fixture")
    rng_state = {"bit_generator": "fixture"}
    save_training_progress(
        owner,
        optimizer_step=20,
        completed_epochs=1,
        recipe_identity=_RECIPE_IDENTITY,
        rng_state=rng_state,
        rank_numpy_rng_states=[rng_state],
        symmetry_rng_state=None,
        rank_torch_rng_states=[
            {"rank": 0, "cpu": torch.get_rng_state().tolist(), "cuda": None}
        ],
        scalar_training_weight_sum=1.0,
        categorical_training_weight_sum=2.0,
        checkpoint_role="resumable_epoch",
        checkpoint_frontier_path=checkpoint_frontier_sidecar_path(owner),
        ddp={"rank": 0},
    )
    shutil.copytree(source, destination)

    relocated_owner = destination / owner.name
    relocated_progress = load_training_progress(
        relocated_owner,
        expected_recipe_identity=_RECIPE_IDENTITY,
    )
    verified = train_bc._verify_resumed_checkpoint_frontier(
        destination / checkpoint.name,
        checkpoint_steps=(8, 16, 32),
        restored_global_step=20,
        resume_recipe_identity=_RECIPE_IDENTITY,
        resume_progress=relocated_progress,
        resume_checkpoint=relocated_owner,
        validation_game_seed_set_sha256=None,
    )

    assert verified["saved_checkpoint_steps"] == {8, 16}
    assert all(
        Path(record["checkpoint"]).parent == destination
        for record in verified["intermediate_checkpoints"]
    )


def test_resume_preserves_and_reseeds_effective_initialization_reference(
    tmp_path: Path,
) -> None:
    old_output = tmp_path / "old" / "candidate.pt"
    new_output = tmp_path / "new" / "candidate.pt"
    old_output.parent.mkdir()
    new_output.parent.mkdir()
    owner = old_output.parent / "candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        old_output,
        owner_checkpoint=owner,
        requested_steps=(8, 16),
        passed_steps=(8,),
        owner_step=12,
        with_initialization_reference=True,
    )
    source_reference = train_bc._step_checkpoint_path(old_output, 0)
    source_sha256 = train_bc._sha256_existing_file(source_reference)

    resumed = train_bc._verify_resumed_checkpoint_frontier(
        new_output,
        checkpoint_steps=(8, 16),
        restored_global_step=12,
        resume_recipe_identity=_RECIPE_IDENTITY,
        resume_progress=progress,
        resume_checkpoint=owner,
        validation_game_seed_set_sha256=None,
    )

    reference = resumed["effective_initialization_reference"]
    assert reference is not None
    reseeded = train_bc._step_checkpoint_path(new_output, 0).resolve()
    assert reference["checkpoint"] == str(reseeded)
    assert reference["checkpoint_sha256"] == source_sha256
    assert reference["holdout_metrics"] == {
        "policy_loss": 0.75,
        "accuracy": 0.25,
    }
    assert reseeded.read_bytes() == source_reference.read_bytes()


def test_resume_rejects_missing_or_malformed_initialization_evidence(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
        with_initialization_reference=True,
    )
    train_bc._step_checkpoint_path(checkpoint, 0).unlink()
    with pytest.raises(SystemExit, match="cannot authenticate"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )

    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
        with_initialization_reference=True,
    )
    journal = checkpoint_frontier_sidecar_path(owner)
    payload = json.loads(journal.read_text(encoding="utf-8"))
    payload["effective_initialization_reference"]["evidence"][
        "holdout_metrics"
    ] = "not-a-metric-object"
    journal.write_text(json.dumps(payload), encoding="utf-8")
    progress["checkpoint_frontier"]["sha256"] = train_bc._sha256_existing_file(
        journal
    )
    with pytest.raises(SystemExit, match="cannot authenticate"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )


def test_empty_frontier_without_initialization_reference_remains_none(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0000.pt"
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(),
        passed_steps=(),
        owner_step=0,
    )

    resumed = train_bc._verify_resumed_checkpoint_frontier(
        checkpoint,
        checkpoint_steps=(),
        restored_global_step=0,
        resume_recipe_identity=_RECIPE_IDENTITY,
        resume_progress=progress,
        resume_checkpoint=owner,
        validation_game_seed_set_sha256=None,
    )

    assert resumed["effective_initialization_reference"] is None
    assert not train_bc._step_checkpoint_path(checkpoint, 0).exists()


def test_resume_rejects_traversal_locator_and_symlinked_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
    )
    journal = checkpoint_frontier_sidecar_path(owner)
    journal_payload = json.loads(journal.read_text(encoding="utf-8"))
    journal_payload["entries"][0]["intermediate_checkpoint"][
        "checkpoint"
    ] = "../outside.pt"
    journal_payload["entries"][0]["holdout"]["checkpoint"] = "../outside.pt"
    journal.write_text(json.dumps(journal_payload), encoding="utf-8")
    progress["checkpoint_frontier"]["sha256"] = train_bc._sha256_existing_file(
        journal
    )
    with pytest.raises(SystemExit, match="cannot authenticate"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )

    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
    )
    snapshot = train_bc._step_checkpoint_path(checkpoint, 8)
    real_snapshot = tmp_path / "outside.pt"
    snapshot.replace(real_snapshot)
    snapshot.symlink_to(real_snapshot)
    with pytest.raises(SystemExit, match="cannot authenticate"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )
def test_resume_refuses_missing_or_wrong_trajectory_past_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    _write_model_checkpoint(owner, 8.0)
    with pytest.raises(SystemExit, match="does not bind an authenticated"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress={},
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )

    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
    )
    snapshot = train_bc._step_checkpoint_path(checkpoint, 8)
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    payload["value_training"]["intermediate_checkpoint_trajectory"][
        "trajectory_root_sha256"
    ] = "sha256:" + "f" * 64
    torch.save(payload, snapshot)
    with pytest.raises(SystemExit, match="cannot authenticate"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )


def test_resume_refuses_occupied_future_snapshot(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(32,),
        passed_steps=(),
        owner_step=16,
    )
    future = train_bc._step_checkpoint_path(checkpoint, 32)
    future.write_bytes(b"unrelated")

    with pytest.raises(SystemExit, match="future intermediate checkpoint path"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(32,),
            restored_global_step=16,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )

    future.unlink()
    optimizer_sidecar_path(future).write_bytes(b"stale optimizer sidecar")
    with pytest.raises(SystemExit, match="future intermediate checkpoint path"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(32,),
            restored_global_step=16,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )


def test_resume_rejects_missing_or_tampered_journal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
    )
    journal = checkpoint_frontier_sidecar_path(owner)
    journal.unlink()
    with pytest.raises(SystemExit, match="progress binding mismatch"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )

    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
    )
    payload = json.loads(journal.read_text(encoding="utf-8"))
    payload["entries"][0]["dose_telemetry"]["effective_policy_lr_area"] = 999.0
    journal.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="progress binding mismatch"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )


def test_resume_rejects_arbitrary_model_bytes_with_copied_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    owner = tmp_path / "candidate.epoch0001.pt"
    progress, _ = _committed_frontier(
        checkpoint,
        owner_checkpoint=owner,
        requested_steps=(8,),
        passed_steps=(8,),
        owner_step=8,
    )
    snapshot = train_bc._step_checkpoint_path(checkpoint, 8)
    copied = torch.load(snapshot, map_location="cpu", weights_only=False)
    copied["model"] = {"weight": torch.tensor([123456.0])}
    torch.save(copied, snapshot)

    with pytest.raises(SystemExit, match="cannot authenticate"):
        train_bc._verify_resumed_checkpoint_frontier(
            checkpoint,
            checkpoint_steps=(8,),
            restored_global_step=8,
            resume_recipe_identity=_RECIPE_IDENTITY,
            resume_progress=progress,
            resume_checkpoint=owner,
            validation_game_seed_set_sha256=None,
        )


def test_nonzero_rank_does_not_write_frontier_journal(tmp_path: Path) -> None:
    owner = tmp_path / "epoch.pt"
    _write_model_checkpoint(owner, 0.0)
    root = train_bc._checkpoint_frontier_trajectory_root(
        resume_recipe_identity=_RECIPE_IDENTITY,
        initial_model_state_sha256="sha256:" + "0" * 64,
    )

    result = train_bc._save_checkpoint_frontier_journal(
        owner,
        checkpoint_steps=(),
        intermediate_checkpoints=(),
        checkpoint_dose_snapshots=(),
        checkpoint_holdout_snapshots=(),
        effective_initialization_reference=None,
        trajectory_root=root,
        resume_recipe_identity=_RECIPE_IDENTITY,
        validation_game_seed_set_sha256=None,
        resume_telemetry_state=(
            train_bc._empty_checkpoint_resume_telemetry_state()
        ),
        owner_checkpoint=owner,
        owner_optimizer_step=0,
        ddp={"rank": 1},
    )

    assert result is None
    assert not checkpoint_frontier_sidecar_path(owner).exists()


def test_ddp_frontier_save_failure_is_broadcast_before_progress_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = tmp_path / "epoch.pt"
    _write_model_checkpoint(owner, 0.0)
    root = train_bc._checkpoint_frontier_trajectory_root(
        resume_recipe_identity=_RECIPE_IDENTITY,
        initial_model_state_sha256="sha256:" + "0" * 64,
    )
    broadcast: dict[str, object] = {}

    def _broadcast_object_list(status: list[object], *, src: int) -> None:
        assert src == 0
        if status[0] is not None:
            broadcast["status"] = status[0]
        else:
            status[0] = broadcast["status"]

    def _refuse_payload(**_kwargs) -> dict[str, object]:
        raise OSError("injected journal write refusal")

    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        _broadcast_object_list,
    )
    monkeypatch.setattr(
        train_bc,
        "_checkpoint_frontier_journal_payload",
        _refuse_payload,
    )
    kwargs = {
        "checkpoint_steps": (),
        "intermediate_checkpoints": (),
        "checkpoint_dose_snapshots": (),
        "checkpoint_holdout_snapshots": (),
        "effective_initialization_reference": None,
        "trajectory_root": root,
        "resume_recipe_identity": _RECIPE_IDENTITY,
        "validation_game_seed_set_sha256": None,
        "resume_telemetry_state": (
            train_bc._empty_checkpoint_resume_telemetry_state()
        ),
        "owner_checkpoint": owner,
        "owner_optimizer_step": 0,
    }

    for rank in (0, 1):
        with pytest.raises(
            RuntimeError,
            match="injected journal write refusal",
        ):
            train_bc._save_checkpoint_frontier_journal(
                owner,
                **kwargs,
                ddp={
                    "enabled": True,
                    "world_size": 2,
                    "rank": rank,
                    "local_rank": rank,
                },
            )

    assert not checkpoint_frontier_sidecar_path(owner).exists()
    assert not training_progress_sidecar_path(owner).exists()
