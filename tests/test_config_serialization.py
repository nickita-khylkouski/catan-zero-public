"""Tests for durable name-keyed config serialization (task #74).

The positional-pickle hazard being killed: frozen+slots dataclasses pickle as
a bare positional list, so (1) shorter/stale pickles leave slots unset and
crash the next re-pickle, and (2) a mid-list field-order divergence silently
shifts every later value into the wrong slot. New checkpoints store configs
as a name-keyed dict; loaders accept both forms.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.config_serialization import (
    CONFIG_CLASS_KEY,
    CONFIG_FIELDS_KEY,
    CONFIG_SCHEMA_KEY,
    config_attr_view,
    config_from_dict,
    config_to_dict,
    is_config_dict,
)
from catan_zero.rl.entity_token_policy import EntityGraphConfig
from catan_zero.rl.xdim_lite_policy import XDimLiteConfig
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig
from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig


def _entity_config(**overrides) -> EntityGraphConfig:
    return EntityGraphConfig(action_size=567, static_action_feature_size=45, **overrides)


@pytest.mark.parametrize(
    "config",
    [
        _entity_config(hidden_size=64, action_mask_version="colonist-multiagent-v1"),
        GumbelChanceMCTSConfig(seed=7, n_full=32, rescale_noise_floor_c=0.5),
        EntityGraphRustEvaluatorConfig(value_squash="clip", public_observation=True),
        XDimLiteConfig(observation_size=10, action_size=5, static_action_feature_size=3),
    ],
    ids=["entity_graph", "gumbel_chance_mcts", "rust_evaluator", "xdim_lite"],
)
def test_roundtrip_every_persisted_config_class(config):
    payload = config_to_dict(config)
    assert is_config_dict(payload)
    assert payload[CONFIG_CLASS_KEY] == type(config).__name__
    rebuilt = config_from_dict(type(config), payload)
    assert rebuilt == config


def test_field_order_is_irrelevant_in_dict_form():
    """The whole point: reconstruction is by NAME, so a reordered fields dict
    (the analogue of a mid-list branch divergence) reconstructs identically."""
    config = _entity_config(value_uncertainty_head=True, action_cross_attention_layers=2)
    payload = config_to_dict(config)
    reordered = {
        CONFIG_CLASS_KEY: payload[CONFIG_CLASS_KEY],
        CONFIG_SCHEMA_KEY: payload[CONFIG_SCHEMA_KEY],
        CONFIG_FIELDS_KEY: dict(reversed(list(payload[CONFIG_FIELDS_KEY].items()))),
    }
    assert config_from_dict(EntityGraphConfig, reordered) == config


def test_missing_fields_take_current_defaults():
    payload = config_to_dict(_entity_config())
    del payload[CONFIG_FIELDS_KEY]["value_attention_pool"]
    del payload[CONFIG_FIELDS_KEY]["dropout"]
    rebuilt = config_from_dict(EntityGraphConfig, payload)
    assert rebuilt.value_attention_pool is False
    assert rebuilt.dropout == 0.05


def test_action_cross_bottleneck_is_appended_for_positional_pickle_safety():
    assert tuple(EntityGraphConfig.__dataclass_fields__)[-1] == (
        "action_cross_attention_bottleneck"
    )


def test_unknown_fields_warn_and_drop():
    payload = config_to_dict(_entity_config())
    payload[CONFIG_FIELDS_KEY]["field_from_the_future"] = 42
    warnings: list[str] = []
    rebuilt = config_from_dict(EntityGraphConfig, payload, warn=warnings.append)
    assert rebuilt == _entity_config()
    assert any("field_from_the_future" in w for w in warnings)


def test_class_name_mismatch_warns_but_reconstructs():
    payload = config_to_dict(_entity_config())
    payload[CONFIG_CLASS_KEY] = "RenamedConfig"
    warnings: list[str] = []
    rebuilt = config_from_dict(EntityGraphConfig, payload, warn=warnings.append)
    assert rebuilt == _entity_config()
    assert any("mismatch" in w for w in warnings)


def test_legacy_stale_dataclass_instance_reconstructs_with_defaults():
    """Mimics an unpickled pre-merge config: later slots UNSET. Reconstruction
    must fill them from defaults instead of crashing (subsumes a413df8)."""
    stale = EntityGraphConfig.__new__(EntityGraphConfig)
    for name, value in (
        ("action_size", 567),
        ("static_action_feature_size", 45),
        ("context_action_feature_size", 18),
        ("legal_action_feature_size", 50),
        ("hidden_size", 640),
        ("state_layers", 6),
        ("attention_heads", 8),
        ("dropout", 0.05),
        ("action_mask_version", "colonist-multiagent-v1"),
        ("schema_version", "entity_graph_policy_v1"),
    ):
        object.__setattr__(stale, name, value)
    with pytest.raises(AttributeError):
        _ = stale.value_uncertainty_head
    rebuilt = config_from_dict(EntityGraphConfig, stale)
    assert rebuilt.value_uncertainty_head is False
    assert rebuilt.hidden_size == 640
    # And the rebuilt instance re-pickles cleanly (the v3a crash scenario).
    import pickle

    assert pickle.loads(pickle.dumps(rebuilt)) == rebuilt


def test_garbage_payload_raises():
    with pytest.raises(TypeError):
        config_from_dict(EntityGraphConfig, ["not", "a", "config"])


def test_attr_view_passthrough_and_dict_adaptation():
    config = _entity_config(hidden_size=64)
    assert config_attr_view(config) is config
    view = config_attr_view(config_to_dict(config))
    assert view.hidden_size == 64
    assert getattr(view, "not_a_field", "sentinel") == "sentinel"


def test_policy_save_load_roundtrip_and_legacy_checkpoint(tmp_path):
    """End-to-end: EntityGraphPolicy new-format save -> load; and a hand-built
    LEGACY checkpoint (config pickled as the dataclass) loads identically."""
    import numpy as np

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    config = EntityGraphConfig(
        action_size=64,
        static_action_feature_size=50,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        action_mask_version="colonist-multiagent-v1",
    )
    static = np.random.default_rng(3).normal(size=(64, 50)).astype(np.float32)
    policy = EntityGraphPolicy(config, static, device="cpu")

    new_path = tmp_path / "new_format.pt"
    policy.save(new_path)
    raw = torch.load(new_path, map_location="cpu", weights_only=False)
    assert is_config_dict(raw["config"]), "save() must write the name-keyed dict form"
    loaded = EntityGraphPolicy.load(new_path, device="cpu")
    assert loaded.config == config

    legacy_path = tmp_path / "legacy_format.pt"
    legacy = dict(raw)
    legacy["config"] = config  # the old pickled-dataclass form
    torch.save(legacy, legacy_path)
    loaded_legacy = EntityGraphPolicy.load(legacy_path, device="cpu")
    assert loaded_legacy.config == config
    for key_new, key_old in zip(
        loaded.model.state_dict().values(), loaded_legacy.model.state_dict().values()
    ):
        assert torch.equal(key_new, key_old)


def test_train_bc_mismatch_checker_accepts_dict_config():
    import importlib.util
    import pathlib
    import sys
    from argparse import Namespace

    tools_dir = pathlib.Path(__file__).resolve().parents[1] / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location("train_bc", tools_dir / "train_bc.py")
    train_bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_bc)

    config = _entity_config(hidden_size=640, state_layers=6, attention_heads=8, dropout=0.05)
    args = Namespace(arch="entity_graph", hidden_size=640, graph_layers=6, attention_heads=8, graph_dropout=0.05)
    for form in (config, config_to_dict(config)):
        assert train_bc._checkpoint_config_mismatches(policy_type="entity_graph", config=form, args=args) == []
    bad = Namespace(arch="entity_graph", hidden_size=320, graph_layers=6, attention_heads=8, graph_dropout=0.05)
    for form in (config, config_to_dict(config)):
        mismatches = train_bc._checkpoint_config_mismatches(policy_type="entity_graph", config=form, args=bad)
        assert mismatches and "hidden_size" in mismatches[0]
