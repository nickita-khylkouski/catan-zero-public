from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.entity_token_policy import EntityGraphPolicy  # noqa: E402
from catan_zero.rl.self_play import make_env_config  # noqa: E402


def test_config_only_legacy_upgrade_clones_loaded_policy_suffix(tmp_path) -> None:
    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=2,
        attention_heads=2,
        dropout=0.0,
        seed=17,
        device="cpu",
    )
    source = tmp_path / "source.pt"
    upgraded_path = tmp_path / "config-only-upgrade.pt"
    base.save(source)

    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["config"]["fields"]["value_tower_split_layers"] = 1
    torch.save(payload, upgraded_path)

    upgraded = EntityGraphPolicy.load(upgraded_path, device="cpu")

    assert upgraded.config.value_tower_split_layers == 1
    assert upgraded._checkpoint_value_tower_cloned_from_policy is True
    policy_suffix = upgraded.model.blocks[-1].state_dict()
    value_suffix = upgraded.model.value_blocks[0].state_dict()
    assert policy_suffix.keys() == value_suffix.keys()
    for key in policy_suffix:
        assert torch.equal(policy_suffix[key], value_suffix[key]), key
    policy_norm = upgraded.model.state_norm.state_dict()
    value_norm = upgraded.model.value_state_norm.state_dict()
    for key in policy_norm:
        assert torch.equal(policy_norm[key], value_norm[key]), key
