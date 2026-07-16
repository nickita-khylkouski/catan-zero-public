from __future__ import annotations

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


def test_optimizer_lr_dose_attests_integrated_area_per_group() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(
        [{"params": [parameter], "lr": 2.0e-4, "base_lr": 2.0e-4}]
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
            "group_index": 0,
            "base_lr": pytest.approx(2.0e-4),
            "integrated_lr_area": pytest.approx(5.0e-4),
            "mean_applied_lr": pytest.approx(1.25e-4),
        }
    ]
