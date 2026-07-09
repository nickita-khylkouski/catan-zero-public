from __future__ import annotations

import argparse

import pytest

from tools.train_bc import _make_optimizer


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
    import torch

    model = torch.nn.Linear(2, 2)
    return list(model.parameters())


def test_weight_decay_zero_with_adam_is_unaffected() -> None:
    """Backward compatibility: the default (adam, weight_decay=0.0) must keep working
    exactly as before -- this is the pre-fix, no-op path."""
    import torch

    optimizer = _make_optimizer(_params(), _args(optimizer="adam", weight_decay=0.0), "cpu")
    assert isinstance(optimizer, torch.optim.Adam)
    assert not isinstance(optimizer, torch.optim.AdamW)


def test_nonzero_weight_decay_with_adam_raises_instead_of_silently_dropping() -> None:
    """AUDIT FIX: --weight-decay > 0 with --optimizer adam (the default) used to be
    silently ignored (plain Adam was constructed without a weight_decay kwarg at all).
    It must now fail loud instead of training with a config the user didn't get."""
    with pytest.raises(SystemExit, match="weight-decay"):
        _make_optimizer(_params(), _args(optimizer="adam", weight_decay=0.05), "cpu")


def test_nonzero_weight_decay_with_adamw_is_applied() -> None:
    optimizer = _make_optimizer(_params(), _args(optimizer="adamw", weight_decay=0.05), "cpu")
    for group in optimizer.param_groups:
        assert group["weight_decay"] == pytest.approx(0.05)


def test_zero_weight_decay_with_adamw_is_still_allowed() -> None:
    """adamw with weight_decay=0.0 is a legitimate, non-error configuration."""
    optimizer = _make_optimizer(_params(), _args(optimizer="adamw", weight_decay=0.0), "cpu")
    for group in optimizer.param_groups:
        assert group["weight_decay"] == pytest.approx(0.0)
