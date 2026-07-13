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


def test_requested_auxiliary_loss_without_heads_fails_before_first_batch():
    model = types.SimpleNamespace(value_categorical_bins=0, value_uncertainty_head=None)
    with pytest.raises(SystemExit, match="complete auxiliary head set"):
        _assert(model, _args(aux_subgoal_loss_weight=0.1))


def test_requested_final_vp_and_belief_losses_require_named_heads():
    model = types.SimpleNamespace(value_categorical_bins=0, value_uncertainty_head=None)
    with pytest.raises(SystemExit, match="final_vp_head"):
        _assert(model, _args(final_vp_loss_weight=0.1))
    with pytest.raises(SystemExit, match="belief_resource_head"):
        _assert(model, _args(belief_resource_loss_weight=0.1))


def test_zero_weight_optional_heads_are_frozen_before_optimizer_construction():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.final_vp_head = torch.nn.Linear(4, 1)
    model.value_uncertainty_head = torch.nn.Linear(4, 1)
    model.value_categorical_head = torch.nn.Linear(4, 7)
    model.aux_longest_road_head = torch.nn.Linear(4, 1)
    model.belief_resource_head = torch.nn.Linear(4, 5)
    model.deliberation_halt_head = torch.nn.Linear(4, 1)

    report = train_bc._freeze_inactive_training_heads(
        model,
        final_vp_loss_weight=0.0,
        value_uncertainty_loss_weight=0.5,
        value_categorical_loss_weight=0.0,
        aux_subgoal_loss_weight=0.0,
        belief_resource_loss_weight=0.0,
    )

    assert all(not p.requires_grad for p in model.final_vp_head.parameters())
    assert all(p.requires_grad for p in model.value_uncertainty_head.parameters())
    assert all(not p.requires_grad for p in model.value_categorical_head.parameters())
    assert all(not p.requires_grad for p in model.aux_longest_road_head.parameters())
    assert all(not p.requires_grad for p in model.belief_resource_head.parameters())
    assert all(not p.requires_grad for p in model.deliberation_halt_head.parameters())
    assert report["active_optional_submodules"] == ["value_uncertainty_head"]
    assert "deliberation_halt_head" in report["frozen_submodules"]
    assert model._inactive_training_head_modules == frozenset(
        report["frozen_submodules"]
    )
    assert report["zero_weight_skips_forward"] is True
