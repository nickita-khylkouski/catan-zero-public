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
    "topology_residual_adapter",
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
    base = EntityGraphConfig(
        action_size=607, static_action_feature_size=1, hidden_size=512
    )
    upgraded = upgrade_tool._build_upgraded_config(base, _OVERRIDES)
    assert upgraded.hidden_size == 512
    assert upgraded.action_cross_attention_layers == 2
    assert dataclasses.replace(base, **_OVERRIDES) == upgraded


def test_topology_upgrade_flag_is_explicit_and_default_off():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    assert base.topology_residual_adapter is False
    overrides = upgrade_tool._parse_flags("gather,topology")
    assert overrides == {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.action_target_gather is True
    assert upgraded.topology_residual_adapter is True


def test_public_card_count_upgrade_flag_is_explicit_and_default_off():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    assert base.public_card_count_features is False
    overrides = upgrade_tool._parse_flags("card_count")
    assert overrides == {"public_card_count_features": True}
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.public_card_count_features is True
    assert upgraded.public_card_count_residual_bias is True


def test_bias_free_public_card_count_upgrade_is_explicit_v2():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    overrides = upgrade_tool._parse_flags("card_count_v2")
    assert overrides == {
        "public_card_count_features": True,
        "public_card_count_residual_bias": False,
    }
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.public_card_count_features is True
    assert upgraded.public_card_count_residual_bias is False


def test_meaningful_history_upgrade_is_bounded_and_explicit():
    base = EntityGraphConfig(action_size=607, static_action_feature_size=1)
    assert base.meaningful_public_history is False
    overrides = upgrade_tool._parse_flags("meaningful_history")
    assert overrides == {
        "meaningful_public_history": True,
        "meaningful_public_history_schema": "meaningful_public_history_2p_no_trade_v1",
        "event_history_limit": 32,
    }
    upgraded = upgrade_tool._build_upgraded_config(base, overrides)
    assert upgraded.meaningful_public_history is True
    assert upgraded.event_history_limit == 32


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
            # A config-only cat-head upgrade must not manufacture evidence that
            # the new random readout was optimized.
            "trained_value_readouts": ["categorical"],
            "value_training": {"primary_readout": "categorical"},
        },
        out_ckpt,
    )

    preserved = upgrade_tool._preserve_source_top_level_keys(
        str(in_ckpt), str(out_ckpt)
    )

    merged = torch.load(out_ckpt, map_location="cpu", weights_only=False)
    # provenance restored from source
    assert merged["mask_hidden_info"] is True
    assert "mask_hidden_info" in preserved
    # intentionally-mutated keys keep the UPGRADED values
    assert merged["config"] == {"flags": "NEW"}
    assert merged["model"]["w"].tolist() == [1.0, 1.0]
    assert "model" not in preserved and "config" not in preserved
    assert "trained_value_readouts" not in merged
    assert "value_training" not in merged


def test_upgrade_seed_is_deterministic_and_durably_attested(
    tmp_path, monkeypatch
) -> None:
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "source.pt"
    out_a = tmp_path / "a.pt"
    out_b = tmp_path / "b.pt"
    out_c = tmp_path / "c.pt"
    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    policy.save(source)

    for output in (out_a, out_b):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "f69_upgrade_checkpoint_config.py",
                "--in-checkpoint",
                str(source),
                "--out-checkpoint",
                str(output),
                "--flags",
                "catbins:9",
                "--seed",
                "73",
                "--device",
                "cpu",
                "--no-verify",
            ],
        )
        upgrade_tool.main()

    raw_a = torch.load(out_a, map_location="cpu", weights_only=False)
    raw_b = torch.load(out_b, map_location="cpu", weights_only=False)
    cat_keys = sorted(
        key for key in raw_a["model"] if key.startswith("value_categorical_head.")
    )
    assert cat_keys
    assert all(
        torch.equal(raw_a["model"][key], raw_b["model"][key]) for key in cat_keys
    )
    assert raw_a["upgrade_provenance"]["initialization_seed"] == 73
    assert raw_a["upgrade_provenance"]["trained_value_readouts_added"] == []
    assert (
        raw_a["upgrade_provenance"]["source_checkpoint_sha256"]
        == raw_b["upgrade_provenance"]["source_checkpoint_sha256"]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(source),
            "--out-checkpoint",
            str(out_c),
            "--flags",
            "catbins:9",
            "--seed",
            "74",
            "--device",
            "cpu",
            "--no-verify",
        ],
    )
    upgrade_tool.main()
    raw_c = torch.load(out_c, map_location="cpu", weights_only=False)
    assert any(
        not torch.equal(raw_a["model"][key], raw_c["model"][key]) for key in cat_keys
    )


def test_combined_topology_gather_upgrade_verifies_exact_real_root(
    tmp_path, monkeypatch
) -> None:
    import torch

    pytest.importorskip("catanatron_rs")

    # Some developer environments retain an older importable wheel that
    # predates the native MCTS snapshot/copy surface.  That environment cannot
    # construct the real-root parity fixture, and is equivalent to the binding
    # being unavailable for this test (the unit/synthetic parity tests still
    # exercise the upgrade below).  Do not misreport a stale wheel as an
    # architecture-upgrade failure.
    from catan_zero.search.rust_mcts import _require_rust_module

    try:
        _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "source.pt"
    output = tmp_path / "topology-gather.pt"
    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    policy.save(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(source),
            "--out-checkpoint",
            str(output),
            "--flags",
            "gather,topology",
            "--seed",
            "73",
            "--device",
            "cpu",
        ],
    )
    upgrade_tool.main()

    raw = torch.load(output, map_location="cpu", weights_only=False)
    assert raw["upgrade_provenance"]["flags"] == {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    assert raw["upgrade_provenance"]["forward_max_diff"] == 0.0
    assert raw["upgrade_provenance"]["forward_identical_at_init"] is True
