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

from catan_zero.rl.optim_state import (  # noqa: E402
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


def _committed_progress(tmp_path: Path, *, recipe=None):
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
        ddp=_DDP_SINGLE,
    )
    return ckpt, identity, rng.bit_generator.state


def test_progress_commit_binds_model_optimizer_recipe_and_exact_step(tmp_path):
    ckpt, identity, rng_state = _committed_progress(tmp_path)
    loaded = load_training_progress(ckpt, expected_recipe_identity=identity)
    assert loaded["optimizer_step"] == 713
    assert loaded["completed_epochs"] == 2
    assert loaded["rng_state"] == rng_state
    assert training_progress_sidecar_path(ckpt).name.endswith(
        ".training-progress.json"
    )


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
    assert restored == (713, 2, 123.5, 44.0)
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


def test_symmetry_rng_is_deterministic_rank_distinct_and_topology_bound() -> None:
    tb = _load_train_bc()
    np = __import__("numpy")

    streams = [
        tb._initialize_symmetry_rng(
            17,
            {"enabled": True, "world_size": 4, "rank": rank, "local_rank": rank},
        )
        for rank in range(4)
    ]
    draws = [stream.integers(0, 12, size=64) for stream in streams]

    assert all(not np.array_equal(draws[0], other) for other in draws[1:])
    replay = tb._initialize_symmetry_rng(
        17, {"enabled": True, "world_size": 4, "rank": 0, "local_rank": 0}
    )
    assert np.array_equal(draws[0], replay.integers(0, 12, size=64))
    different_topology = tb._initialize_symmetry_rng(
        17, {"enabled": True, "world_size": 8, "rank": 0, "local_rank": 0}
    )
    assert not np.array_equal(
        draws[0], different_topology.integers(0, 12, size=64)
    )


def test_train_bc_restores_exact_rank_local_symmetry_rng() -> None:
    tb = _load_train_bc()
    np = __import__("numpy")
    rank_rngs = [
        tb._initialize_symmetry_rng(
            23,
            {"enabled": True, "world_size": 2, "rank": rank, "local_rank": rank},
        )
        for rank in range(2)
    ]
    rank_rngs[0].integers(0, 12, size=3)
    rank_rngs[1].integers(0, 12, size=9)
    states = [rng.bit_generator.state for rng in rank_rngs]
    expected_rank1 = rank_rngs[1].integers(0, 12, size=8)
    restored_symmetry_rng = np.random.default_rng(999)
    sampler_rng = np.random.default_rng(1)
    torch_state = torch.get_rng_state().tolist()
    progress = {
        "optimizer_step": 7,
        "completed_epochs": 1,
        "rng_state": sampler_rng.bit_generator.state,
        "rank_numpy_rng_states": [
            sampler_rng.bit_generator.state,
            sampler_rng.bit_generator.state,
        ],
        "symmetry_rng_state": tb._rank_symmetry_rng_progress_payload(
            states, world_size=2
        ),
        "rank_torch_rng_states": [
            {"rank": 0, "cpu": torch_state, "cuda": None},
            {"rank": 1, "cpu": torch_state, "cuda": None},
        ],
        "scalar_training_weight_sum": 0.0,
        "categorical_training_weight_sum": 0.0,
    }

    tb._restore_training_progress_state(
        progress,
        epochs=2,
        rng=sampler_rng,
        symmetry_rng=restored_symmetry_rng,
        ddp={"enabled": True, "world_size": 2, "rank": 1, "local_rank": 1},
    )

    assert np.array_equal(
        restored_symmetry_rng.integers(0, 12, size=8), expected_rank1
    )


def test_train_bc_checkpoint_gathers_every_rank_symmetry_rng(monkeypatch) -> None:
    tb = _load_train_bc()
    np = __import__("numpy")
    import catan_zero.rl.optim_state as optim_state
    import torch.distributed as dist

    sampler_rng = np.random.default_rng(31)
    rank0_symmetry_rng = tb._initialize_symmetry_rng(
        41, {"enabled": True, "world_size": 2, "rank": 0, "local_rank": 0}
    )
    rank1_symmetry_rng = tb._initialize_symmetry_rng(
        41, {"enabled": True, "world_size": 2, "rank": 1, "local_rank": 1}
    )
    rank0_symmetry_rng.integers(0, 12, size=2)
    rank1_symmetry_rng.integers(0, 12, size=7)
    expected_rank_states = [
        rank0_symmetry_rng.bit_generator.state,
        rank1_symmetry_rng.bit_generator.state,
    ]
    gather_calls = 0

    def fake_all_gather_object(outputs, local) -> None:
        nonlocal gather_calls
        if gather_calls == 0:
            outputs[0] = local
            outputs[1] = {**local, "rank": 1}
        elif gather_calls == 1:
            outputs[0] = local
            outputs[1] = local
        elif gather_calls == 2:
            outputs[:] = expected_rank_states
        else:  # pragma: no cover - catches accidental extra collectives
            raise AssertionError("unexpected RNG collective")
        gather_calls += 1

    captured: dict[str, object] = {}

    def fake_save_training_progress(_checkpoint_path, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        optim_state, "save_training_progress", fake_save_training_progress
    )

    tb._save_training_progress_sidecar(
        "candidate.pt",
        optimizer_saved=Path("candidate.pt.optimizer.pt"),
        optimizer_step=11,
        completed_epochs=1,
        recipe_identity={"world_size": 2},
        rng=sampler_rng,
        symmetry_rng=rank0_symmetry_rng,
        scalar_training_weight_sum=2.0,
        categorical_training_weight_sum=0.0,
        ddp={"enabled": True, "world_size": 2, "rank": 0, "local_rank": 0},
    )

    assert gather_calls == 3
    assert captured["symmetry_rng_state"] == {
        "schema_version": "train-bc-rank-symmetry-rng-v1",
        "world_size": 2,
        "rank_states": expected_rank_states,
    }


@pytest.mark.parametrize("failure", ["legacy", "world_size", "missing_rank"])
def test_distributed_symmetry_resume_fails_closed(failure: str) -> None:
    tb = _load_train_bc()
    np = __import__("numpy")
    source = np.random.default_rng(7)
    states = [source.bit_generator.state, source.bit_generator.state]
    if failure == "legacy":
        saved_symmetry_state = states[0]
    else:
        saved_symmetry_state = tb._rank_symmetry_rng_progress_payload(
            states, world_size=2
        )
        if failure == "world_size":
            saved_symmetry_state["world_size"] = 3
        else:
            saved_symmetry_state["rank_states"] = states[:1]
    sampler_rng = np.random.default_rng(1)
    torch_state = torch.get_rng_state().tolist()
    progress = {
        "optimizer_step": 7,
        "completed_epochs": 1,
        "rng_state": sampler_rng.bit_generator.state,
        "rank_numpy_rng_states": [
            sampler_rng.bit_generator.state,
            sampler_rng.bit_generator.state,
        ],
        "symmetry_rng_state": saved_symmetry_state,
        "rank_torch_rng_states": [
            {"rank": 0, "cpu": torch_state, "cuda": None},
            {"rank": 1, "cpu": torch_state, "cuda": None},
        ],
        "scalar_training_weight_sum": 0.0,
        "categorical_training_weight_sum": 0.0,
    }

    with pytest.raises(SystemExit, match="per-rank RNG|per-rank symmetry RNG"):
        tb._restore_training_progress_state(
            progress,
            epochs=2,
            rng=sampler_rng,
            symmetry_rng=np.random.default_rng(9),
            ddp={"enabled": True, "world_size": 2, "rank": 1, "local_rank": 1},
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
