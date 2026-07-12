from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import train_bc


def test_optimizer_observability_reuses_default_off_diagnostics_cadence() -> None:
    parser = train_bc.build_parser()
    assert parser.get_default("train_diagnostics_every_batches") == 0


def test_optimizer_observability_reports_preclip_norm_clip_and_module_updates() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Linear(2, 2, bias=False)
            self.value_head = torch.nn.Linear(2, 1, bias=False)

    model = TinyModel()
    policy = SimpleNamespace(model=model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)

    expected_total = sum(parameter.numel() * 4.0 for parameter in model.parameters()) ** 0.5
    state = train_bc._capture_optimizer_observability(policy)
    pre_clip = train_bc._clip_grad_norm(policy, 1.0)
    optimizer.step()
    observed = train_bc._finish_optimizer_observability(
        policy,
        state,
        pre_clip_total_grad_norm=pre_clip,
        max_grad_norm=1.0,
    )

    assert observed["pre_clip_total_grad_norm"] == pytest.approx(expected_total)
    assert observed["max_grad_norm"] == pytest.approx(1.0)
    assert observed["clipped"] is True
    assert observed["module_norm_scope"] == "global_replicated"
    assert set(observed["module_pre_clip_grad_norms"]) == {"trunk", "value_head"}
    assert observed["module_pre_clip_grad_norms"]["trunk"] == pytest.approx(4.0)
    assert observed["module_pre_clip_grad_norms"]["value_head"] == pytest.approx(
        2.0 * (2.0**0.5)
    )
    assert observed["module_parameter_delta_norms"]["trunk"] > 0.0
    assert observed["module_parameter_delta_norms"]["value_head"] > 0.0


def test_optimizer_observability_name_normalizes_ddp_and_fsdp_prefixes() -> None:
    normalize = train_bc._optimizer_observability_module_name
    assert normalize("module.blocks.0.weight") == "blocks"
    assert normalize("_fsdp_wrapped_module.value_head.weight") == "value_head"
    assert normalize("module._fsdp_wrapped_module.action_bias") == "action_bias"


def test_objective_gradient_interference_measures_shared_trunk_not_heads() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 1, bias=False)])
            self.policy_head = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)

    model = TinyModel()
    with torch.no_grad():
        model.blocks[0].weight.copy_(torch.tensor([[1.0, 1.0]]))
        model.policy_head.weight.fill_(1.0)
        model.value_head.weight.fill_(1.0)
    policy = SimpleNamespace(model=model)
    shared = model.blocks[0](torch.tensor([[1.0, -2.0]]))
    policy_objective = model.policy_head(shared).sum()
    value_objective = -2.0 * model.value_head(shared).sum()

    observed = train_bc._objective_gradient_interference(
        policy,
        policy_objective=policy_objective,
        value_objective=value_objective,
    )

    assert observed["available"] is True
    assert observed["scope"] == "single_process_microbatch"
    assert observed["value_lr_mult_scales_shared_trunk"] is False
    assert observed["policy_trunk_grad_norm"] == pytest.approx(5.0**0.5)
    assert observed["value_trunk_grad_norm"] == pytest.approx(2.0 * 5.0**0.5)
    assert observed["value_to_policy_grad_norm_ratio"] == pytest.approx(2.0)
    assert observed["trunk_gradient_cosine"] == pytest.approx(-1.0)
    assert observed["opposing_coordinate_fraction"] == pytest.approx(1.0)
    assert observed["combined_trunk_grad_norm"] == pytest.approx(5.0**0.5)
    assert observed["modules"]["blocks.0"]["cosine"] == pytest.approx(-1.0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_objective_gradient_interference_is_explicit_when_objective_inactive() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

    model = TinyModel()
    active = model.blocks[0](torch.ones(1, 1)).sum()
    observed = train_bc._objective_gradient_interference(
        SimpleNamespace(model=model),
        policy_objective=active,
        value_objective=torch.zeros(()),
    )
    assert observed == {
        "available": False,
        "reason": "inactive_policy_or_value_objective",
    }
