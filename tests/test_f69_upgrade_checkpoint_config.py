"""Regression: tools/f69_upgrade_checkpoint_config.py must upgrade a config
pickled BEFORE a field existed.

The seed checkpoint's EntityGraphConfig predates both the f69 flags and f67's
value_uncertainty_head, so the original `dataclasses.replace(base.config, ...)`
raised AttributeError (replace reads every current field off the stale object).
`_build_upgraded_config` copies the fields that exist, fills the rest from the
dataclass defaults, then applies the flag overrides.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import fields
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import f69_upgrade_checkpoint_config as upgrade_tool  # noqa: E402
from catan_zero.rl.entity_token_policy import EntityGraphConfig  # noqa: E402

# Fields absent from a seed config pickled before these landed.
_LATER_FIELDS = (
    "value_uncertainty_head",
    "action_target_gather",
    "action_cross_attention_layers",
    "value_attention_pool",
)
_OVERRIDES = {
    "action_target_gather": True,
    "action_cross_attention_layers": 2,
    "value_attention_pool": True,
}


def _stale_config():
    """A real EntityGraphConfig instance with `_LATER_FIELDS` slots UNSET --
    exactly how a frozen+slots dataclass pickled before those fields existed
    deserialises under the newer class definition (only the pickled slots are
    set; accessing an unset slot raises AttributeError). Using a real instance
    (not a SimpleNamespace) is what makes the replace() failure the genuine
    AttributeError, not a spurious TypeError."""
    full = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    stale = object.__new__(EntityGraphConfig)
    for f in fields(EntityGraphConfig):
        if f.name not in _LATER_FIELDS:
            object.__setattr__(stale, f.name, getattr(full, f.name))
    return stale


def test_replace_on_stale_config_reproduces_the_bug():
    """Documents the failure the fix addresses."""
    stale = _stale_config()
    assert not hasattr(stale, "value_uncertainty_head")
    with pytest.raises(AttributeError):
        dataclasses.replace(stale, **_OVERRIDES)


def test_build_upgraded_config_tolerates_missing_field():
    stale = _stale_config()
    upgraded = upgrade_tool._build_upgraded_config(stale, _OVERRIDES)

    assert isinstance(upgraded, EntityGraphConfig)
    # f69 overrides applied
    assert upgraded.action_target_gather is True
    assert upgraded.action_cross_attention_layers == 2
    assert upgraded.value_attention_pool is True
    # a field the stale pickle lacked is filled from the current default
    assert upgraded.value_uncertainty_head is False
    # pre-existing fields copied through unchanged
    assert upgraded.action_size == 607
    assert upgraded.hidden_size == 640
    assert upgraded.state_layers == 6


def test_build_upgraded_config_preserves_a_full_config():
    """A current (non-stale) config round-trips with only the overrides changed."""
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1, hidden_size=512)
    upgraded = upgrade_tool._build_upgraded_config(base, _OVERRIDES)
    assert upgraded.hidden_size == 512
    assert upgraded.action_cross_attention_layers == 2
    assert dataclasses.replace(base, **_OVERRIDES) == upgraded


def test_preserve_source_top_level_keys_restores_mask_hidden_info(tmp_path):
    """CAT-80 regression: upgrading a masked checkpoint must NOT drop top-level
    provenance keys. EntityGraphPolicy.save() rebuilds the checkpoint from the
    fresh upgraded policy, resetting mask_hidden_info True->False (mislabeling a
    masked net as omniscient). _preserve_source_top_level_keys restores every
    source top-level key except the intentionally-mutated model+config."""
    import torch

    in_ckpt = tmp_path / "in.pt"
    out_ckpt = tmp_path / "out.pt"
    torch.save(
        {
            "policy_type": "entity_graph",
            "mask_hidden_info": True,
            "action_mask_version": "colonist-multiagent-v1",
            "static_action_features_sha256": "abc",
            "config": {"flags": "OLD"},
            "model": {"w": torch.zeros(2)},
        },
        in_ckpt,
    )
    # What EntityGraphPolicy.save() would have written: model+config mutated,
    # mask_hidden_info silently reset to the fresh-policy default False (the bug).
    torch.save(
        {
            "policy_type": "entity_graph",
            "mask_hidden_info": False,
            "action_mask_version": "colonist-multiagent-v1",
            "static_action_features_sha256": "abc",
            "config": {"flags": "NEW"},
            "model": {"w": torch.ones(2)},
        },
        out_ckpt,
    )

    preserved = upgrade_tool._preserve_source_top_level_keys(str(in_ckpt), str(out_ckpt))

    merged = torch.load(out_ckpt, map_location="cpu", weights_only=False)
    # provenance restored from source
    assert merged["mask_hidden_info"] is True
    assert "mask_hidden_info" in preserved
    # intentionally-mutated keys keep the UPGRADED values
    assert merged["config"] == {"flags": "NEW"}
    assert merged["model"]["w"].tolist() == [1.0, 1.0]
    assert "model" not in preserved and "config" not in preserved
