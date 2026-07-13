from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from catan_zero.rl.pipeline_configs import TrainConfig
from tools import train_bc


_DDP_1 = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}


def _parsed(*extra: str):
    return train_bc.build_parser().parse_args(
        [
            "--data",
            "teacher.npz",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "train.report.json",
            *extra,
        ]
    )


def test_float32_matmul_precision_is_explicit_and_baseline_preserving() -> None:
    baseline = _parsed()
    treatment = _parsed("--float32-matmul-precision", "high")

    assert baseline.float32_matmul_precision == "highest"
    assert treatment.float32_matmul_precision == "high"
    assert TrainConfig.from_namespace(baseline).float32_matmul_precision == "highest"
    assert TrainConfig.from_namespace(treatment).float32_matmul_precision == "high"


@pytest.mark.parametrize(
    ("amp", "requested", "effective"),
    [
        ("none", "highest", "highest"),
        ("none", "high", "high"),
        # Preserve the historical BF16 behavior: it always selected high.
        ("bf16", "highest", "high"),
        ("bf16", "high", "high"),
    ],
)
def test_effective_float32_matmul_precision(
    amp: str, requested: str, effective: str
) -> None:
    assert (
        train_bc._effective_float32_matmul_precision(  # noqa: SLF001
            amp=amp, requested=requested
        )
        == effective
    )


def test_precision_contract_records_requested_and_effective_modes() -> None:
    args = SimpleNamespace(
        amp="bf16",
        float32_matmul_precision="highest",
        effective_float32_matmul_precision="high",
    )

    assert train_bc._floating_point_execution_contract(args) == {  # noqa: SLF001
        "schema_version": "train-bc-floating-point-execution-v1",
        "amp": "bf16",
        "requested_float32_matmul_precision": "highest",
        "effective_float32_matmul_precision": "high",
    }


def test_tf32_axis_changes_typed_and_resume_identity() -> None:
    baseline = TrainConfig(float32_matmul_precision="highest")
    treatment = dataclasses.replace(baseline, float32_matmul_precision="high")
    args = SimpleNamespace(
        grad_accum_steps=1,
        ddp_shard_data=False,
        fsdp=False,
        policy_aux_active_batch_size=0,
    )

    assert baseline.config_hash() != treatment.config_hash()
    assert train_bc._training_resume_recipe_identity(  # noqa: SLF001
        baseline, args, _DDP_1
    ) != train_bc._training_resume_recipe_identity(treatment, args, _DDP_1)  # noqa: SLF001


def test_tf32_axis_is_additively_a1_recipe_bound() -> None:
    baseline = _parsed()
    treatment = _parsed("--float32-matmul-precision", "high")

    baseline_recipe = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        baseline, _DDP_1
    )
    treatment_recipe = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        treatment, _DDP_1
    )

    # Old seals predate the knob and describe the historical highest mode.
    assert "float32_matmul_precision" not in baseline_recipe
    # A TF32 treatment cannot pass under that old seal.
    assert treatment_recipe["float32_matmul_precision"] == "high"
    assert treatment_recipe != baseline_recipe
