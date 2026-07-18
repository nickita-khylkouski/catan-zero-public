from __future__ import annotations

import argparse

import pytest
import torch

from tools.train_bc import _build_optimizer_param_groups, _make_optimizer


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "optimizer": "adam",
        "weight_decay": 0.0,
        "lr": 1e-3,
        "fused_optimizer": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _params():
    model = torch.nn.Linear(2, 2)
    return list(model.parameters())


class _Args:
    optimizer = "adamw"
    weight_decay = 0.1
    fused_optimizer = False
    lr = 2e-4


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.LayerNorm(8),
        )
        self.value_head = torch.nn.Linear(8, 1)
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.frozen_matrix = torch.nn.Parameter(
            torch.ones(2, 2),
            requires_grad=False,
        )


def _parameter_group_by_id(optimizer) -> dict[int, dict]:
    by_id: dict[int, dict] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert id(parameter) not in by_id
            by_id[id(parameter)] = group
    return by_id


def test_weight_decay_zero_with_adam_is_unaffected() -> None:
    """Backward compatibility: default Adam without decay remains unchanged."""
    optimizer = _make_optimizer(
        _params(),
        _args(optimizer="adam", weight_decay=0.0),
        "cpu",
    )
    assert isinstance(optimizer, torch.optim.Adam)
    assert not isinstance(optimizer, torch.optim.AdamW)


def test_nonzero_weight_decay_with_adam_raises_instead_of_silently_dropping() -> None:
    with pytest.raises(SystemExit, match="weight-decay"):
        _make_optimizer(
            _params(),
            _args(optimizer="adam", weight_decay=0.05),
            "cpu",
        )


def test_nonzero_weight_decay_with_adamw_is_applied_to_matrix_only() -> None:
    params = _params()
    optimizer = _make_optimizer(
        params,
        _args(optimizer="adamw", weight_decay=0.05),
        "cpu",
    )
    by_id = _parameter_group_by_id(optimizer)
    for parameter in params:
        assert by_id[id(parameter)]["weight_decay"] == pytest.approx(
            0.05 if parameter.ndim >= 2 else 0.0
        )


def test_zero_weight_decay_with_adamw_is_still_allowed() -> None:
    optimizer = _make_optimizer(
        _params(),
        _args(optimizer="adamw", weight_decay=0.0),
        "cpu",
    )
    for group in optimizer.param_groups:
        assert group["weight_decay"] == pytest.approx(0.0)


def test_adamw_excludes_bias_norm_and_scalar_parameters_from_weight_decay() -> None:
    model = _TinyPolicy()
    optimizer = _make_optimizer(list(model.parameters()), _Args(), "cpu")
    by_id = _parameter_group_by_id(optimizer)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert set(by_id) == {id(parameter) for parameter in trainable}
    assert id(model.frozen_matrix) not in by_id
    for parameter in trainable:
        group = by_id[id(parameter)]
        expected_decay = 0.1 if parameter.ndim >= 2 else 0.0
        assert group["weight_decay"] == pytest.approx(expected_decay)


def test_adamw_decay_split_preserves_each_lr_group_and_has_no_overlap() -> None:
    model = _TinyPolicy()
    source_groups = _build_optimizer_param_groups(
        model,
        base_lr=2e-4,
        value_lr_mult=0.3,
    )
    optimizer = _make_optimizer(source_groups, _Args(), "cpu")
    by_id = _parameter_group_by_id(optimizer)

    value_ids = {
        id(parameter)
        for parameter in model.value_head.parameters()
        if parameter.requires_grad
    }
    all_trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert set(by_id) == all_trainable_ids
    for parameter in model.parameters():
        if not parameter.requires_grad:
            assert id(parameter) not in by_id
            continue
        group = by_id[id(parameter)]
        expected_lr = 2e-4 * 0.3 if id(parameter) in value_ids else 2e-4
        assert group["lr"] == pytest.approx(expected_lr)
        assert group["base_lr"] == pytest.approx(expected_lr)
        assert group["weight_decay"] == pytest.approx(
            0.1 if parameter.ndim >= 2 else 0.0
        )
        assert group["group_name"] == (
            "value" if id(parameter) in value_ids else "base"
        )
        assert group["weight_decay_role"] == (
            "decay" if parameter.ndim >= 2 else "no_decay"
        )
    assert {
        (group["group_name"], group["weight_decay_role"])
        for group in optimizer.param_groups
    } == {
        ("base", "decay"),
        ("base", "no_decay"),
        ("value", "decay"),
        ("value", "no_decay"),
    }
