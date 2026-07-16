from types import SimpleNamespace

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
                "policy_objective_effective_weight_sum": 7.5,
                "loss_denominators": {"policy_loss": 40.0},
            }
        ],
        policy_loss_weight=1.0,
        optimizer_steps=4,
        train_value_only=False,
    )

    assert report["policy_active_rows"] == 10
    assert report["policy_effective_weight_sum"] == pytest.approx(7.5)
    assert report["trained_policy_objective"] is True
