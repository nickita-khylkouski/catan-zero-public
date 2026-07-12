from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from tools import train_bc


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(3, 4)
        self.event_encoder = nn.Sequential(
            nn.Linear(2, 4), nn.LayerNorm(4), nn.GELU(), nn.Linear(4, 4)
        )


def test_authenticated_empty_event_freeze_touches_only_event_encoder() -> None:
    model = _Model()
    trunk_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    report = train_bc._freeze_authenticated_empty_event_encoder(model)

    expected = sorted(
        f"event_encoder.{name}" for name, _ in model.event_encoder.named_parameters()
    )
    assert report["frozen_parameter_names"] == expected
    assert report["frozen_parameter_tensors"] == 6
    assert report["unexpected_frozen_parameter_tensors"] == 0
    assert report["optimizer_excluded_parameter_tensors"] == 6
    assert (
        report["trainable_parameters_before"] - report["trainable_parameters_after"]
        == report["frozen_parameters"]
    )
    assert all(not parameter.requires_grad for parameter in model.event_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.trunk.parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, trunk_before[name], rtol=0, atol=0)


def test_authenticated_empty_event_freeze_refuses_ambiguous_second_freeze() -> None:
    model = _Model()
    train_bc._freeze_authenticated_empty_event_encoder(model)
    with pytest.raises(SystemExit, match="fully trainable event_encoder"):
        train_bc._freeze_authenticated_empty_event_encoder(model)


def test_authenticated_empty_event_freeze_requires_named_encoder() -> None:
    with pytest.raises(SystemExit, match="named event_encoder"):
        train_bc._freeze_authenticated_empty_event_encoder(nn.Linear(2, 2))


def test_frozen_event_encoder_is_excluded_from_optimizer_groups() -> None:
    model = _Model()
    train_bc._freeze_authenticated_empty_event_encoder(model)

    groups = train_bc._build_optimizer_param_groups(
        model,
        base_lr=3e-5,
        value_lr_mult=1.0,
        action_module_lr_mult=1.0,
        trunk_lr_mult=1.0,
        architecture="entity_graph",
    )
    optimized = {id(parameter) for parameter in groups}

    assert optimized
    assert not optimized.intersection(id(p) for p in model.event_encoder.parameters())
    assert optimized.issuperset(id(p) for p in model.trunk.parameters())


class _CroppedForwardModel(_Model):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Width-zero authenticated crop bypasses event_encoder exactly like the
        # production entity model's encode_state path.
        return self.trunk(inputs)


def test_two_ddp_iterations_succeed_with_cropped_event_encoder_frozen(tmp_path) -> None:
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the default process group")
    rendezvous = tmp_path / "gloo-rendezvous"
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1
    )
    try:
        model = _CroppedForwardModel()
        train_bc._freeze_authenticated_empty_event_encoder(model)
        ddp = DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.Adam(
            [parameter for parameter in ddp.parameters() if parameter.requires_grad],
            lr=3e-5,
        )
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            ddp(torch.ones(8, 3)).square().mean().backward()
            optimizer.step()
    finally:
        torch.distributed.destroy_process_group()
