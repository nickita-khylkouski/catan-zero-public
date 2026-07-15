"""Fail-closed checkpoint loading for config-enabled entity-graph modules."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy


def _config(**overrides) -> EntityGraphConfig:
    config = EntityGraphConfig(
        action_size=16,
        static_action_feature_size=45,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    return replace(config, **overrides)


def _write_checkpoint(tmp_path, config: EntityGraphConfig):
    path = tmp_path / "checkpoint.pt"
    policy = EntityGraphPolicy(
        config,
        np.zeros(
            (config.action_size, config.static_action_feature_size), dtype=np.float32
        ),
        seed=7,
        device="cpu",
    )
    policy.save(path, mask_hidden_info=True)
    return path


def _remove_first_parameter(path, prefix: str, *, output):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    key = next(key for key in raw["model"] if key.startswith(prefix))
    del raw["model"][key]
    torch.save(raw, output)
    return key


@pytest.mark.parametrize(
    ("overrides", "missing_prefix"),
    (
        ({"action_target_gather": True}, "target_gather_proj."),
        ({"action_cross_attention_layers": 1}, "action_cross_blocks."),
        ({"edge_policy_head": True}, "edge_policy_mlp."),
        ({"static_action_residual": True}, "static_action_residual_proj."),
        (
            {"public_card_count_features": True},
            "public_card_count_residual.",
        ),
        (
            {"meaningful_public_history": True, "event_history_limit": 32},
            "meaningful_history_residual_gate",
        ),
        ({"topology_residual_adapter": True}, "topology_residual_adapter."),
        ({"aux_subgoal_heads": True}, "aux_longest_road_head."),
        ({"belief_resource_head": True}, "belief_resource_head."),
        ({"value_uncertainty_head": True}, "value_uncertainty_head."),
        ({"value_categorical_bins": 7}, "value_categorical_head."),
        ({"value_attention_pool": True}, "value_probe"),
    ),
)
def test_default_load_rejects_missing_enabled_optional_parameter_but_explicit_warmstart_allows_it(
    tmp_path, overrides, missing_prefix
) -> None:
    complete = _write_checkpoint(tmp_path, _config(**overrides))
    incomplete = tmp_path / "incomplete.pt"
    removed = _remove_first_parameter(complete, missing_prefix, output=incomplete)

    with pytest.raises(RuntimeError, match="checkpoint state mismatch"):
        EntityGraphPolicy.load(incomplete, device="cpu")

    warmstart = EntityGraphPolicy.load(
        incomplete,
        device="cpu",
        allow_missing_optional_parameters=True,
    )
    assert removed in warmstart._checkpoint_missing_state_keys


def test_explicit_optional_warmstart_does_not_allow_missing_base_tensor(
    tmp_path,
) -> None:
    complete = _write_checkpoint(tmp_path, _config(static_action_residual=True))
    incomplete = tmp_path / "missing-base.pt"
    removed = _remove_first_parameter(complete, "value_head.", output=incomplete)

    with pytest.raises(RuntimeError, match=removed):
        EntityGraphPolicy.load(
            incomplete,
            device="cpu",
            allow_missing_optional_parameters=True,
        )


def test_legacy_checkpoint_with_all_new_flags_off_keeps_q_head_compatibility(
    tmp_path,
) -> None:
    complete = _write_checkpoint(tmp_path, _config())
    raw = torch.load(complete, map_location="cpu", weights_only=False)
    removed = sorted(key for key in raw["model"] if key.startswith("q_head."))
    assert removed
    for key in removed:
        del raw["model"][key]
    legacy = tmp_path / "legacy.pt"
    torch.save(raw, legacy)

    loaded = EntityGraphPolicy.load(legacy, device="cpu")
    assert sorted(loaded._checkpoint_missing_state_keys) == removed
    assert loaded.config.action_target_gather is False
    assert loaded.config.aux_subgoal_heads is False
    assert loaded.config.static_action_residual is False
