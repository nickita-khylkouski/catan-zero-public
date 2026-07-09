"""Tests for the HL-Gauss distributional (categorical) value head (CAT-39).

Gated by EntityGraphConfig.value_categorical_bins (default 0 = OFF). The head
must be a pure no-op by default (bit-identical parameter set and forward
outputs), and ADDITIVE when enabled: existing output keys stay bit-identical to
the same weights run with the flag off (warm-start safety), while the new
distribution is emitted under "value_categorical_logits" / "value_categorical".

Per the CAT-39 R9 ruling the primary support is win/loss ONLY plus one distinct
TRUNCATION class (VP-margin routes to a separate aux head). Targets are built
with the HL-Gauss projection (Farebrother et al. 2024, arXiv:2403.03950), NOT
two-hot: two-hot underperforms MSE, HL-Gauss beats it. The scalar readout is the
support-expectation over the win-loss bins renormalised to exclude truncation
mass -- the calibrated win-value the search backup consumes.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
from catan_zero.rl.config_serialization import config_from_dict, config_to_dict
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet


def _load_train_bc():
    tools_dir = pathlib.Path(__file__).resolve().parents[1] / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location("train_bc", tools_dir / "train_bc.py")
    train_bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_bc)
    return train_bc


def _config(*, bins: int, truncation_class: bool = True) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        value_categorical_bins=bins,
        value_categorical_truncation_class=truncation_class,
    )


def _synthetic_batch(batch_size: int = 3, num_actions: int = 5) -> dict:
    counts = {
        "hex": (19, HEX_FEATURE_SIZE),
        "vertex": (54, VERTEX_FEATURE_SIZE),
        "edge": (72, EDGE_FEATURE_SIZE),
        "player": (4, PLAYER_FEATURE_SIZE),
        "global": (1, GLOBAL_FEATURE_SIZE),
        "event": (64, EVENT_FEATURE_SIZE),
    }
    torch.manual_seed(11)
    batch: dict = {}
    for name, (count, feat) in counts.items():
        batch[f"{name}_tokens"] = torch.randn(batch_size, count, feat)
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(batch_size, count, dtype=torch.bool)
    batch["legal_action_tokens"] = torch.randn(batch_size, num_actions, LEGAL_ACTION_FEATURE_SIZE)
    batch["legal_action_context"] = torch.randn(batch_size, num_actions, CONTEXT_ACTION_FEATURE_SIZE)
    return batch


# --------------------------------------------------------------------------
# Model: default no-op + additive warm-start safety
# --------------------------------------------------------------------------
def test_default_config_omits_the_head():
    """MSE-arm regression: with bins=0 (the default) the model has no
    categorical head, no new params, and no new outputs -- the scalar value head
    is exactly what it was before CAT-39."""
    default = EntityGraphConfig(action_size=1, static_action_feature_size=1)
    assert default.value_categorical_bins == 0
    assert default.value_categorical_truncation_class is True
    model = EntityGraphNet(_config(bins=0))
    assert model.value_categorical_head is None
    model.eval()
    outputs = model(_synthetic_batch())
    assert "value_categorical_logits" not in outputs
    assert "value_categorical" not in outputs
    assert "value" in outputs


def test_enabled_head_is_purely_additive_on_shared_weights():
    """Warm-start safety: copy flag-off weights into a flag-on model; every
    pre-existing output key must stay bit-identical. The head emits bins+1 logits
    (win-loss bins + the truncation class) and a scalar readout in [-1, 1]."""
    base = EntityGraphNet(_config(bins=0))
    upgraded = EntityGraphNet(_config(bins=9))
    missing, unexpected = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("value_categorical_head.") for k in missing)
    base.eval()
    upgraded.eval()
    batch = _synthetic_batch()
    out_base = base(batch, return_q=True)
    out_up = upgraded(batch, return_q=True)
    for key in ("logits", "value", "final_vp", "q_values"):
        assert torch.equal(out_base[key], out_up[key]), key
    assert out_up["value_categorical_logits"].shape == (3, 10)  # 9 bins + truncation
    assert out_up["value_categorical"].shape == (3,)
    assert torch.all(out_up["value_categorical"] >= -1.0)
    assert torch.all(out_up["value_categorical"] <= 1.0)
    assert "value_categorical_truncation_prob" in out_up


def test_truncation_class_can_be_disabled():
    model = EntityGraphNet(_config(bins=9, truncation_class=False))
    model.eval()
    out = model(_synthetic_batch())
    assert out["value_categorical_logits"].shape == (3, 9)  # no truncation column
    assert "value_categorical_truncation_prob" not in out


def test_scalar_readout_excludes_truncation_mass():
    """R9: the scalar readout is the calibrated win-value -- expectation over the
    win-loss bins renormalised to drop truncation-class mass, never a blend that
    lets truncation probability pull the value."""
    model = EntityGraphNet(_config(bins=9))
    model.eval()
    out = model(_synthetic_batch())
    logits = out["value_categorical_logits"].float()
    bins = 9
    support = torch.linspace(-1.0, 1.0, bins)
    probs = torch.softmax(logits, dim=-1)
    win_probs = probs[:, :bins]
    expected = (win_probs / win_probs.sum(dim=-1, keepdim=True) * support).sum(dim=-1)
    assert torch.allclose(out["value_categorical"], expected, atol=1e-5)


# --------------------------------------------------------------------------
# HL-Gauss projection math
# --------------------------------------------------------------------------
def test_hl_gauss_projection_round_trip():
    """Project scalar -> distribution -> expectation recovers the target within
    the bin resolution (CAT-39 verification (a)/(b)); rows are row-stochastic."""
    train_bc = _load_train_bc()
    bins = 31
    bin_width = 2.0 / (bins - 1)
    support = torch.linspace(-1.0, 1.0, bins)
    targets = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5, 0.73])
    out = train_bc._hl_gauss_value_targets(
        targets, bins, sigma_ratio=0.75, add_truncation_class=False
    )
    assert out.shape == (6, bins)
    assert torch.allclose(out.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert torch.all(out >= 0.0)
    expectation = (out * support).sum(dim=-1)
    # Interior targets recover to within the bin resolution.
    assert torch.allclose(expectation, targets, atol=bin_width)


def test_hl_gauss_mass_concentrates_near_target():
    """Mass is concentrated near the true scalar value: the nearest atom carries
    the most mass and probability falls off with distance."""
    train_bc = _load_train_bc()
    bins = 31
    out = train_bc._hl_gauss_value_targets(
        torch.tensor([0.0]), bins, sigma_ratio=0.75, add_truncation_class=False
    )[0]
    peak = int(torch.argmax(out).item())
    assert peak == bins // 2  # atom at 0.0
    # Monotonic falloff moving away from the peak on both sides.
    left = out[: peak + 1]
    right = out[peak:]
    assert torch.all(left[1:] - left[:-1] >= -1e-6)
    assert torch.all(right[1:] - right[:-1] <= 1e-6)


def test_hl_gauss_sigma_controls_spread():
    """sigma = sigma_ratio * bin_width: smaller ratio -> spikier (two-hot limit),
    larger ratio -> broader target (higher entropy)."""
    train_bc = _load_train_bc()
    bins = 31
    target = torch.tensor([0.0])

    def entropy(sigma_ratio):
        p = train_bc._hl_gauss_value_targets(
            target, bins, sigma_ratio=sigma_ratio, add_truncation_class=False
        )[0]
        return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())

    tiny = train_bc._hl_gauss_value_targets(
        target, bins, sigma_ratio=1e-4, add_truncation_class=False
    )[0]
    # Two-hot limit: essentially all mass on the single aligned atom.
    assert tiny[bins // 2].item() > 0.999
    assert entropy(0.5) < entropy(1.0) < entropy(2.0)


def test_truncation_class_routing():
    """Truncated rows go ENTIRELY to the truncation class (one-hot), leaving the
    win-loss bins at 0; non-truncated rows carry 0 truncation mass (R9 support)."""
    train_bc = _load_train_bc()
    bins = 9
    targets = torch.tensor([0.5, -1.0, 0.3])
    truncated = torch.tensor([False, True, False])
    out = train_bc._hl_gauss_value_targets(
        targets, bins, sigma_ratio=0.75, truncated=truncated, add_truncation_class=True
    )
    assert out.shape == (3, bins + 1)
    # Truncated row: one-hot on the truncation class.
    assert out[1, bins].item() == 1.0
    assert out[1, :bins].abs().sum().item() == 0.0
    # Non-truncated rows: zero truncation mass, win-loss bins sum to 1.
    assert out[0, bins].item() == 0.0
    assert out[2, bins].item() == 0.0
    assert torch.allclose(out[[0, 2], :bins].sum(dim=-1), torch.ones(2), atol=1e-5)


# --------------------------------------------------------------------------
# Distribution-space lambda blend vs scalar blend consistency
# --------------------------------------------------------------------------
def test_lambda_distribution_blend_matches_scalar_blend():
    """Blending two HL-Gauss distributions at lambda, then taking the
    expectation, matches the scalar blend lambda*z + (1-lambda)*V within the bin
    resolution -- because HL-Gauss preserves the expectation and blending is
    linear. This is why distribution-space blending is legitimate."""
    train_bc = _load_train_bc()
    bins = 31
    bin_width = 2.0 / (bins - 1)
    support = torch.linspace(-1.0, 1.0, bins)
    z = torch.tensor([0.6, -0.4, 0.1])
    v = torch.tensor([-0.2, 0.5, -0.8])
    lam = 0.7
    dz = train_bc._hl_gauss_value_targets(z, bins, sigma_ratio=0.75, add_truncation_class=False)
    dv = train_bc._hl_gauss_value_targets(v, bins, sigma_ratio=0.75, add_truncation_class=False)
    dist_blend = lam * dz + (1.0 - lam) * dv
    assert torch.allclose(dist_blend.sum(dim=-1), torch.ones(3), atol=1e-5)
    e_dist = (dist_blend * support).sum(dim=-1)
    scalar_blend = lam * z + (1.0 - lam) * v
    assert torch.allclose(e_dist, scalar_blend, atol=bin_width)


# --------------------------------------------------------------------------
# Checkpoint interface: config + state_dict round-trip
# --------------------------------------------------------------------------
def test_config_round_trip_preserves_head_fields():
    cfg = _config(bins=33, truncation_class=True)
    restored = config_from_dict(EntityGraphConfig, config_to_dict(cfg))
    assert restored.value_categorical_bins == 33
    assert restored.value_categorical_truncation_class is True


def test_state_dict_round_trip_with_head():
    """A model with the head strict-loads its own state_dict (the support buffer
    is non-persistent, so it never appears in the dict), and the head params are
    present under value_categorical_head.*."""
    model = EntityGraphNet(_config(bins=33))
    sd = model.state_dict()
    assert any(k.startswith("value_categorical_head.") for k in sd)
    assert "value_categorical_support" not in sd  # non-persistent
    fresh = EntityGraphNet(_config(bins=33))
    missing, unexpected = fresh.load_state_dict(sd, strict=True)
    assert not missing and not unexpected


def test_head_gradient_flows():
    model = EntityGraphNet(_config(bins=9))
    model.train()
    outputs = model(_synthetic_batch())
    logits = outputs["value_categorical_logits"]
    target = torch.zeros(3, 10)
    target[:, -1] = 1.0  # truncation class
    ce = -(target * torch.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
    ce.backward()
    head_grads = [p.grad for p in model.value_categorical_head.parameters() if p.grad is not None]
    assert head_grads
    assert any(g.abs().sum().item() > 0.0 for g in head_grads)
