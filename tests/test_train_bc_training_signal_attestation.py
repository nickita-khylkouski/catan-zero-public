from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from tools import train_bc


def test_policy_signal_attestation_exposes_sparse_realized_dose() -> None:
    # Shape copied from the B200 fresh-policy 1024-step report: policy learning
    # was real, but only 9.87% of base draw events were policy-active.
    attestation = train_bc._policy_training_signal_attestation(
        [
            {
                "samples": 4_194_304,
                "policy_base_active_rows": 413_917,
                "policy_aux_active_rows": 0,
                "loss_denominators": {"policy_loss": 3_800_000.0},
            }
        ],
        policy_loss_weight=1.0,
        optimizer_steps=1_024,
        train_value_only=False,
    )

    assert attestation["status"] == "trained"
    assert attestation["trained_policy_objective"] is True
    assert attestation["policy_active_draw_fraction"] == pytest.approx(
        413_917 / 4_194_304
    )
    assert attestation["policy_effective_weight_sum"] == 3_800_000.0


def test_policy_signal_attestation_refuses_value_only_steps_masquerading_as_policy() -> None:
    with pytest.raises(RuntimeError, match="no realized policy training signal"):
        train_bc._policy_training_signal_attestation(
            [
                {
                    "samples": 4_096,
                    "policy_base_active_rows": 0,
                    "policy_aux_active_rows": 0,
                    "loss_denominators": {
                        "policy_loss": 0.0,
                        "value_loss": 4_096.0,
                    },
                }
            ],
            policy_loss_weight=1.0,
            optimizer_steps=128,
            train_value_only=False,
        )


def test_policy_signal_attestation_allows_explicit_value_only_training() -> None:
    attestation = train_bc._policy_training_signal_attestation(
        [{"samples": 4_096, "loss_denominators": {"policy_loss": 0.0}}],
        policy_loss_weight=1.0,
        optimizer_steps=128,
        train_value_only=True,
    )

    assert attestation["status"] == "disabled_value_only"
    assert attestation["policy_objective_enabled"] is False
    assert attestation["trained_policy_objective"] is False


@pytest.mark.parametrize("coefficient", [-1.0, math.nan, math.inf, -math.inf])
def test_policy_signal_attestation_rejects_unsafe_coefficients(
    coefficient: float,
) -> None:
    with pytest.raises(
        SystemExit, match="policy-loss-weight must be finite and non-negative"
    ):
        train_bc._policy_training_signal_attestation(
            [
                {
                    "samples": 32,
                    "policy_base_active_rows": 32,
                    "loss_denominators": {"policy_loss": 32.0},
                }
            ],
            policy_loss_weight=coefficient,
            optimizer_steps=1,
            train_value_only=False,
        )


def test_optimizer_lr_dose_attests_integrated_area_per_group() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(
        [
            {
                "params": [parameter],
                "lr": 2.0e-4,
                "base_lr": 2.0e-4,
                "group_name": "base",
                "weight_decay_role": "no_decay",
            }
        ]
    )
    multipliers = [0.25, 0.5, 0.75, 1.0]
    dose = train_bc._optimizer_lr_dose_attestation(
        applied_updates=4,
        schedule_multiplier_sum=sum(multipliers),
        lr_area_by_group=[2.0e-4 * sum(multipliers)],
        optimizer=optimizer,
    )

    assert dose["applied_updates"] == 4
    assert dose["integrated_schedule_multiplier_area"] == pytest.approx(2.5)
    assert dose["mean_schedule_multiplier"] == pytest.approx(0.625)
    assert dose["parameter_groups"] == [
        {
            "semantic_group_name": "base",
            "optimizer_group_indices": [0],
            "optimizer_group_count": 1,
            "parameter_tensors": 1,
            "parameters": 1,
            "base_lr": pytest.approx(2.0e-4),
            "integrated_lr_area": pytest.approx(5.0e-4),
            "mean_applied_lr": pytest.approx(1.25e-4),
        }
    ]


def test_optimizer_lr_dose_rejects_ambiguous_semantic_groups() -> None:
    import torch

    left = torch.nn.Parameter(torch.tensor(1.0))
    right = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.Adam(
        [
            {
                "params": [left],
                "lr": 1.0e-4,
                "base_lr": 1.0e-4,
                "group_name": "value",
                "weight_decay_role": "no_decay",
            },
            {
                "params": [right],
                "lr": 1.0e-4,
                "base_lr": 1.0e-4,
                "group_name": "value",
                "weight_decay_role": "no_decay",
            },
        ],
        lr=1.0e-4,
    )

    with pytest.raises(ValueError, match="identities must be unique"):
        train_bc._optimizer_lr_dose_attestation(
            applied_updates=1,
            schedule_multiplier_sum=1.0,
            lr_area_by_group=[1.0e-4, 1.0e-4],
            optimizer=optimizer,
        )


@pytest.mark.parametrize("field", ["base_lr", "lr", "area"])
def test_optimizer_lr_dose_rejects_nonfinite_group_values(field: str) -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(
        [
            {
                "params": [parameter],
                "lr": 2.0e-4,
                "base_lr": 2.0e-4,
                "group_name": "value",
                "weight_decay_role": "no_decay",
            }
        ]
    )
    area = 2.0e-4
    if field == "area":
        area = math.nan
    else:
        optimizer.param_groups[0][field] = math.nan

    with pytest.raises(ValueError, match="finite"):
        train_bc._optimizer_lr_dose_attestation(
            applied_updates=1,
            schedule_multiplier_sum=1.0,
            lr_area_by_group=[area],
            optimizer=optimizer,
        )


def test_policy_signal_attestation_is_wired_to_report_and_final_checkpoint() -> None:
    """Source contract: computed evidence must survive both durable outputs."""

    source_path = Path(train_bc.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    early_validation = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "args"
            and target.attr == "policy_loss_weight"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_validate_policy_loss_weight"
    )
    report_assignment = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "report"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    )
    assert early_validation.lineno < report_assignment.lineno
    report_fields = {
        key.value: value
        for key, value in zip(
            report_assignment.value.keys,
            report_assignment.value.values,
            strict=True,
        )
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert isinstance(report_fields["policy_training_signal"], ast.Name)
    assert report_fields["policy_training_signal"].id == "policy_training_signal"

    surface_assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "training_information_surface"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    assert any(
        any(
            isinstance(key, ast.Constant) and key.value == "policy_training_signal"
            for key in assignment.value.keys
        )
        for assignment in surface_assignments
    )

    final_save = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_save_policy"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Attribute)
        and isinstance(node.args[1].value, ast.Name)
        and node.args[1].value.id == "args"
        and node.args[1].attr == "checkpoint"
    )
    checkpoint_surface = next(
        keyword.value
        for keyword in final_save.keywords
        if keyword.arg == "training_information_surface"
    )
    assert isinstance(checkpoint_surface, ast.Call)
    assert isinstance(checkpoint_surface.func, ast.Name)
    assert checkpoint_surface.func.id == "_policy_kl_controller_surface"
    assert isinstance(checkpoint_surface.args[0], ast.Name)
    assert checkpoint_surface.args[0].id == "training_information_surface"
