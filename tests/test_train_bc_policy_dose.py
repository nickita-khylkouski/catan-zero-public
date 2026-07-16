from types import SimpleNamespace

import numpy as np
import pytest

from tools import train_bc


def test_policy_lr_area_hits_exact_boundary_and_then_stops() -> None:
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.0,
        target_lr_area=0.025,
    ) == pytest.approx(1.0)
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.02,
        target_lr_area=0.025,
    ) == pytest.approx(0.5)
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        1.0,
        scheduled_base_lr=0.01,
        consumed_lr_area=0.025,
        target_lr_area=0.025,
    ) == 0.0


def test_zero_policy_dose_preserves_historical_constant_weight() -> None:
    assert train_bc._policy_weight_for_lr_area(  # noqa: SLF001
        0.75,
        scheduled_base_lr=0.0,
        consumed_lr_area=0.0,
        target_lr_area=0.0,
    ) == pytest.approx(0.75)


def test_policy_objective_fraction_preserves_fractional_boundary() -> None:
    assert train_bc._policy_objective_fraction(  # noqa: SLF001
        0.25, 1.0
    ) == pytest.approx(0.25)
    assert train_bc._policy_objective_fraction(  # noqa: SLF001
        0.0, 1.0
    ) == 0.0
    with pytest.raises(ValueError, match="exceeds"):
        train_bc._policy_objective_fraction(1.01, 1.0)  # noqa: SLF001


def test_policy_dose_requires_matching_global_batch_topology() -> None:
    assert train_bc._validate_policy_dose_topology(  # noqa: SLF001
        target_lr_area=0.01,
        reference_global_batch_size=512,
        local_batch_size=64,
        grad_accum_steps=1,
        world_size=8,
    ) == 512
    with pytest.raises(SystemExit, match="cannot cross optimizer topology"):
        train_bc._validate_policy_dose_topology(  # noqa: SLF001
            target_lr_area=0.01,
            reference_global_batch_size=4096,
            local_batch_size=64,
            grad_accum_steps=1,
            world_size=8,
        )


def test_positive_policy_dose_requires_explicit_reference_topology() -> None:
    with pytest.raises(SystemExit, match="requires.*reference-global-batch-size"):
        train_bc._validate_policy_dose_topology(  # noqa: SLF001
            target_lr_area=0.01,
            reference_global_batch_size=0,
            local_batch_size=64,
            grad_accum_steps=1,
            world_size=8,
        )


def test_uncapped_training_can_request_an_early_checkpoint_frontier() -> None:
    assert train_bc._parse_checkpoint_steps(  # noqa: SLF001
        "8,16,32,64,128",
        max_steps=0,
    ) == (8, 16, 32, 64, 128)


def test_policy_only_gradient_suppression_keeps_shared_value_paths() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_bias = torch.nn.Linear(3, 1)
            self.edge_policy_mlp = torch.nn.Linear(3, 1)
            self.logit_scale = torch.nn.Parameter(torch.ones(()))
            self.state_norm = torch.nn.LayerNorm(3)
            self.value_head = torch.nn.Linear(3, 1)
            self.value_tower_split_layers = 1

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in model.action_bias.parameters():
        optimizer.state[parameter]["stale_momentum"] = torch.ones_like(parameter)
    optimizer.state[model.logit_scale]["stale_momentum"] = torch.ones_like(
        model.logit_scale
    )
    suppressed = train_bc._suppress_inactive_policy_only_gradients(  # noqa: SLF001
        SimpleNamespace(model=model),
        optimizer,
    )

    assert "logit_scale" in suppressed
    assert all(parameter.grad is None for parameter in model.action_bias.parameters())
    assert all(
        parameter.grad is None for parameter in model.edge_policy_mlp.parameters()
    )
    assert all(parameter.grad is None for parameter in model.state_norm.parameters())
    assert all(parameter.grad is not None for parameter in model.value_head.parameters())
    assert all(parameter not in optimizer.state for parameter in model.action_bias.parameters())
    assert model.logit_scale not in optimizer.state


def test_policy_signal_attestation_uses_scheduled_objective_mass() -> None:
    report = train_bc._policy_training_signal_attestation(  # noqa: SLF001
        [
            {
                "samples": 100,
                "policy_base_active_rows": 40,
                "policy_aux_active_rows": 0,
                "policy_objective_active_rows": 10,
                "policy_objective_equivalent_active_rows": 7.5,
                "policy_objective_effective_weight_sum": 7.5,
                "policy_objective_equivalent_effective_weight_sum": 7.5,
                "policy_objective_optimizer_updates": 4,
                "policy_objective_equivalent_optimizer_updates": 3.5,
                "loss_denominators": {"policy_loss": 40.0},
            }
        ],
        policy_loss_weight=1.0,
        optimizer_steps=4,
        train_value_only=False,
    )

    assert report["policy_active_rows"] == 10
    assert report["policy_equivalent_active_rows"] == pytest.approx(7.5)
    assert report["policy_effective_weight_sum"] == pytest.approx(7.5)
    assert report["policy_optimizer_updates"] == 4
    assert report["policy_equivalent_optimizer_updates"] == pytest.approx(3.5)
    assert report["trained_policy_objective"] is True


def test_policy_signal_attestation_recovers_full_dose_equivalent_weight() -> None:
    report = train_bc._policy_training_signal_attestation(  # noqa: SLF001
        [
            {
                "samples": 8,
                "policy_base_active_rows": 8,
                "policy_objective_active_rows": 8,
                "policy_objective_equivalent_active_rows": 4.0,
                "policy_objective_effective_weight_sum": 2.0,
                "policy_objective_optimizer_updates": 1,
                "policy_objective_equivalent_optimizer_updates": 0.5,
            }
        ],
        policy_loss_weight=0.5,
        optimizer_steps=1,
        train_value_only=False,
    )

    assert report["policy_effective_weight_sum"] == pytest.approx(2.0)
    assert report["policy_equivalent_effective_weight_sum"] == pytest.approx(
        4.0
    )


def test_fractional_policy_strata_report_full_dose_equivalent_rows() -> None:
    data = {
        "legal_action_ids": np.asarray([[1, 2], [1, 2]], dtype=np.int16),
        "phase": np.asarray(["opening", "main"]),
    }
    dose = train_bc._training_strata_dose_for_batch(  # noqa: SLF001
        data,
        np.arange(2, dtype=np.int64),
        policy_weights=np.ones(2, dtype=np.float32),
        value_weights=np.ones(2, dtype=np.float32),
        value_active_mask=np.ones(2, dtype=np.bool_),
        policy_objective_fraction=0.25,
    )
    report = train_bc._nest_training_strata_dose(  # noqa: SLF001
        train_bc._flatten_training_strata_dose(dose)  # noqa: SLF001
    )

    assert report["policy_active_row_draws"] == 2
    assert report["policy_objective_active_row_draws"] == 2
    assert report["policy_objective_equivalent_row_draws"] == pytest.approx(
        0.5
    )
    assert report["dimensions"]["phase"]["opening"][
        "policy_objective_equivalent_rows"
    ] == pytest.approx(0.25)


def test_component_dose_counts_only_rows_passed_to_objective() -> None:
    class Composite(dict):
        component_ids = ("fresh", "replay")

        @staticmethod
        def component_indices_for_rows(rows):
            return np.asarray(rows, dtype=np.int64) % 2

    data = Composite(
        phase=np.asarray(["opening", "opening", "main", "main"])
    )
    component, phase = train_bc._policy_component_dose_for_batch(  # noqa: SLF001
        data,
        np.asarray([0, 3], dtype=np.int64),
        suffix="base",
        phase_names=("opening", "main"),
    )

    assert component == {"fresh.base": 1.0, "replay.base": 1.0}
    assert phase["fresh\0opening\0base"] == 1.0
    assert phase["replay\0main\0base"] == 1.0
