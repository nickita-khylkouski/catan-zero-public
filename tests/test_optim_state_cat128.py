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
