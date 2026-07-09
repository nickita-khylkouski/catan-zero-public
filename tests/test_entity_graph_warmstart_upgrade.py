"""Warm-start / config-upgrade tests for the CAT-97 + CAT-100 heads.

The whole point of these upgrades is to keep champion_v0's healthy torso and add
heads: enabling edge_policy_head (CAT-97) and aux_subgoal_heads (CAT-100) on a
checkpoint trained WITHOUT them must reproduce the original value/policy function
bit-for-bit at init, and the new params must be exactly the allowed-missing set.
This mirrors tools/f69_upgrade_checkpoint_config.py + EntityGraphPolicy.load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
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

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import f69_upgrade_checkpoint_config as upgrade_tool  # noqa: E402


def _config(**flags) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        **flags,
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
    batch: dict = {}
    for name, (count, feat) in counts.items():
        batch[f"{name}_tokens"] = torch.randn(batch_size, count, feat)
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(batch_size, count, dtype=torch.bool)
    batch["legal_action_tokens"] = torch.randn(batch_size, num_actions, LEGAL_ACTION_FEATURE_SIZE)
    batch["legal_action_context"] = torch.randn(batch_size, num_actions, CONTEXT_ACTION_FEATURE_SIZE)
    target_ids = -torch.ones(batch_size, num_actions, 4, dtype=torch.long)
    target_ids[:, :, 1] = torch.arange(num_actions).remainder(54).view(1, -1)
    batch["legal_action_target_ids"] = target_ids
    return batch


def test_parse_flags_supports_edge_and_aux():
    overrides = upgrade_tool._parse_flags("edge,aux")
    assert overrides == {"edge_policy_head": True, "aux_subgoal_heads": True}


def test_build_upgraded_config_applies_new_heads():
    base = _config()
    upgraded = upgrade_tool._build_upgraded_config(
        base, {"edge_policy_head": True, "aux_subgoal_heads": True}
    )
    assert upgraded.edge_policy_head is True
    assert upgraded.aux_subgoal_heads is True
    # Torso fields preserved.
    assert upgraded.hidden_size == 16
    assert upgraded.state_layers == 1


def test_warmstart_champion_to_both_heads_is_bit_identical():
    """champion (heads off) -> upgraded (edge + aux on): value/policy identical."""
    torch.manual_seed(0)
    champion = EntityGraphNet(_config())
    upgraded = EntityGraphNet(_config(edge_policy_head=True, aux_subgoal_heads=True))

    missing, unexpected = upgraded.load_state_dict(champion.state_dict(), strict=False)
    assert unexpected == []
    # Every missing key is a brand-new head param (nothing torso-level dropped).
    allowed = ("edge_policy_mlp.", "aux_")
    assert all(k.startswith(allowed) for k in missing), missing

    champion.eval()
    upgraded.eval()
    batch = _synthetic_batch()
    out_c = champion(batch)
    out_u = upgraded(batch)
    for key in ("logits", "value", "final_vp"):
        assert torch.equal(out_c[key], out_u[key]), key
    # The upgraded model additionally exposes the new outputs.
    assert "aux_next_settlement" in out_u
    assert "aux_next_settlement" not in out_c


def test_load_allow_list_covers_new_head_prefixes():
    """Guard: EntityGraphPolicy.load must tolerate the new heads as missing so a
    legacy checkpoint warm-starts into an upgraded config."""
    import inspect

    from catan_zero.rl import entity_token_policy

    src = inspect.getsource(entity_token_policy.EntityGraphPolicy.load)
    for prefix in (
        "edge_policy_mlp.",
        "aux_longest_road_head.",
        "aux_largest_army_head.",
        "aux_vp_in_n_head.",
        "aux_next_settlement_head.",
        "aux_robber_target_head.",
    ):
        assert prefix in src, prefix
