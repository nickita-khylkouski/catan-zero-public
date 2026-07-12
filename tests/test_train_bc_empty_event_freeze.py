from __future__ import annotations

import pytest
import torch
from torch import nn

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
