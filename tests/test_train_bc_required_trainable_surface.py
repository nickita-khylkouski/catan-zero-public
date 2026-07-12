from __future__ import annotations

import pytest
import torch

from tools import train_bc


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_gather_proj = torch.nn.Linear(3, 3)
        self.trunk = torch.nn.Linear(3, 3)


def test_required_trainable_surface_accepts_only_exact_adapter() -> None:
    model = _Tiny()
    for parameter in model.trunk.parameters():
        parameter.requires_grad = False
    value = train_bc._require_only_trainable_prefixes(  # noqa: SLF001
        model, ("target_gather_proj",)
    )
    assert value["prefixes"] == ["target_gather_proj"]
    assert value["parameter_tensors"] == 2
    assert value["parameters"] == 12


def test_required_trainable_surface_rejects_escape_or_missing_prefix() -> None:
    model = _Tiny()
    with pytest.raises(SystemExit, match="escaped"):
        train_bc._require_only_trainable_prefixes(  # noqa: SLF001
            model, ("target_gather_proj",)
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    with pytest.raises(SystemExit, match="no parameters"):
        train_bc._require_only_trainable_prefixes(  # noqa: SLF001
            model, ("target_gather_proj",)
        )
