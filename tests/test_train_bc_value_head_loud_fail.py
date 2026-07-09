"""Loud-fail guard: a requested value-head objective on a model that lacks the
head must SystemExit, not silently train with the loss stuck at 0.0 for a
multi-hundred-GPU-hour run (RUN-6 hardening, silent-no-op class)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
for p in (str(_TOOLS_DIR), str(_SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_bc  # type: ignore  # noqa: E402

_assert = train_bc._assert_value_heads_present_for_losses


def _args(**kw):
    base = dict(value_head_type="scalar", value_uncertainty_loss_weight=0.0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_hlgauss_on_scalar_model_fails_loud():
    model = types.SimpleNamespace(value_categorical_bins=0, value_uncertainty_head=None)
    with pytest.raises(SystemExit, match="value_categorical_bins"):
        _assert(model, _args(value_head_type="hlgauss"))


def test_hlgauss_on_catbins_model_ok():
    model = types.SimpleNamespace(value_categorical_bins=51, value_uncertainty_head=None)
    _assert(model, _args(value_head_type="hlgauss"))  # no raise


def test_uncertainty_loss_without_head_fails_loud():
    model = types.SimpleNamespace(value_categorical_bins=0, value_uncertainty_head=None)
    with pytest.raises(SystemExit, match="value_uncertainty_head"):
        _assert(model, _args(value_uncertainty_loss_weight=0.5))


def test_uncertainty_loss_with_head_ok():
    model = types.SimpleNamespace(value_categorical_bins=0, value_uncertainty_head=object())
    _assert(model, _args(value_uncertainty_loss_weight=0.5))  # no raise


def test_scalar_defaults_are_noop():
    model = types.SimpleNamespace(value_categorical_bins=0, value_uncertainty_head=None)
    _assert(model, _args())  # neither objective requested -> no raise
