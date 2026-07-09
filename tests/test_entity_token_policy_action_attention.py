"""f69 action-attention upgrade: warm-start / flag-off equivalence.

Guards the load-time contract for the three config-gated upgrades in
`entity_token_policy.EntityGraphNet` (action_target_gather,
action_cross_attention_layers, value_attention_pool):

* flags OFF -> the module has exactly the pre-upgrade parameter set, so old
  checkpoints load with strict=True;
* flags ON  -> every new sub-module is zero-initialised on its output path,
  so loading a pre-upgrade checkpoint (strict=False) and running a forward
  reproduces the un-upgraded outputs bit-for-bit (the warm-start guarantee).

The batch is built from a real env decision state so the target-id gather path
actually gathers board tokens rather than exercising an empty code path.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


def _real_entity_batch(n_states: int = 3):
    """A collated entity batch from real placement/early decision states."""
    from catan_zero.rl._catanatron import import_catanatron_module

    import_catanatron_module("catanatron")
    from catan_zero.rl.action_features import (
        CONTEXT_ACTION_FEATURE_SIZE,
        build_action_context_feature_table,
    )
    from catan_zero.rl.entity_token_features import build_entity_token_features
    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
    from catan_zero.rl.self_play import make_env_config

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    rows = []
    try:
        observations, info = env.reset(seed=7)
        del observations
        for _ in range(n_states):
            player = str(info["current_player"])
            valid_actions = tuple(int(a) for a in info["valid_actions"])
            entity = {
                key: np.asarray(value)
                for key, value in build_entity_token_features(env, player).items()
                if key != "schema"
            }
            context_table = np.asarray(
                build_action_context_feature_table(env, info), dtype=np.float32
            )
            legal_context = context_table[list(valid_actions), :]
            rows.append((entity, legal_context))
            # advance with the first legal action to reach a new state
            _obs, _r, term, trunc, info = env.step(int(valid_actions[0]))
            if term or trunc:
                _obs, info = env.reset(seed=7)
    finally:
        env.close()

    max_actions = max(int(entity["legal_action_tokens"].shape[0]) for entity, _ in rows)

    def _pad_actions(arr: np.ndarray, fill) -> np.ndarray:
        out = np.full((max_actions,) + arr.shape[1:], fill, dtype=arr.dtype)
        out[: arr.shape[0]] = arr
        return out

    batch: dict[str, np.ndarray] = {}
    action_keyed = {
        "legal_action_tokens": 0.0,
        "legal_action_target_ids": -1,
        "legal_action_mask": False,
    }
    keys = list(rows[0][0].keys())
    for key in keys:
        if key in action_keyed:
            batch[key] = np.stack(
                [_pad_actions(entity[key], action_keyed[key]) for entity, _ in rows]
            )
        else:
            batch[key] = np.stack([entity[key] for entity, _ in rows])
    context = np.stack(
        [_pad_actions(ctx, 0.0) for _, ctx in rows]
    ).astype(np.float32)
    batch["legal_action_context"] = context
    assert context.shape[2] == CONTEXT_ACTION_FEATURE_SIZE
    return batch


def _to_torch(batch: dict[str, np.ndarray]):
    import torch

    out = {}
    for key, value in batch.items():
        if value.dtype == np.bool_:
            out[key] = torch.as_tensor(value)
        elif np.issubdtype(value.dtype, np.integer):
            out[key] = torch.as_tensor(value.astype(np.int64))
        else:
            out[key] = torch.as_tensor(value.astype(np.float32))
    return out


def _base_config():
    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    # small net -> fast test, structure identical to production shape
    return EntityGraphConfig(
        action_size=607,
        static_action_feature_size=1,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=64,
        state_layers=2,
        attention_heads=4,
        dropout=0.0,
    )


ALL_FLAGS_ON = dict(
    action_target_gather=True,
    action_cross_attention_layers=2,
    value_attention_pool=True,
)


def test_flags_off_parameter_set_is_identical_to_base():
    """No new parameters exist when every flag is off (strict-load contract)."""
    from catan_zero.rl.entity_token_policy import EntityGraphNet

    base = EntityGraphNet(_base_config())
    off = EntityGraphNet(dataclasses.replace(_base_config()))
    assert set(base.state_dict().keys()) == set(off.state_dict().keys())
    # strict load must succeed between two flag-off modules
    off.load_state_dict(base.state_dict(), strict=True)


def test_flags_on_adds_only_new_params_and_warm_start_is_exact():
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphNet

    base = EntityGraphNet(_base_config())
    up = EntityGraphNet(dataclasses.replace(_base_config(), **ALL_FLAGS_ON))

    base_keys = set(base.state_dict().keys())
    up_keys = set(up.state_dict().keys())
    # flags-on is a strict superset: only new upgrade params are added
    assert base_keys.issubset(up_keys)
    new_keys = up_keys - base_keys
    assert new_keys, "flags-on must introduce new parameters"
    allowed_prefixes = (
        "target_gather_proj.",
        "action_cross_blocks.",
        "value_probe",
        "value_pool_head.",
    )
    assert all(k.startswith(allowed_prefixes) for k in new_keys), sorted(new_keys)

    # warm-start: load pre-upgrade weights into the upgraded module
    missing, unexpected = up.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert set(missing) == new_keys

    base.eval()
    up.eval()
    batch = _to_torch(_real_entity_batch())
    with torch.no_grad():
        ob = base(batch, return_q=True)
        ou = up(batch, return_q=True)

    diffs = {}
    for key in ("logits", "value", "final_vp", "q_values"):
        d = (ob[key] - ou[key]).abs().max().item()
        diffs[key] = d
    print("WARMSTART_MAXDIFF", diffs)
    for key, d in diffs.items():
        assert d == 0.0, f"{key} diff {d} != 0 (zero-init warm-start broken)"


@pytest.mark.parametrize(
    "flags,changed",
    [
        (dict(action_target_gather=True), ("logits", "q_values")),
        (dict(action_cross_attention_layers=1), ("logits", "q_values")),
        (dict(value_attention_pool=True), ("value",)),
    ],
)
def test_each_upgrade_path_is_live(flags, changed):
    """Perturbing the zero-init output param of each upgrade must move the
    outputs it feeds -- proves the path is wired, not a permanent no-op."""
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphNet

    base = EntityGraphNet(_base_config())
    up = EntityGraphNet(dataclasses.replace(_base_config(), **flags))
    up.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    up.eval()

    with torch.no_grad():
        for name, param in up.named_parameters():
            if name.endswith((
                "target_gather_proj.1.weight",
                "out_proj.weight",
                "ff.3.weight",
                "value_pool_head.4.weight",
            )):
                param.add_(torch.randn_like(param))

    batch = _to_torch(_real_entity_batch(n_states=2))
    with torch.no_grad():
        ob = base(batch, return_q=True)
        ou = up(batch, return_q=True)
    for key in changed:
        assert (ob[key] - ou[key]).abs().max().item() > 0.0, f"{key} did not move"


@pytest.mark.parametrize(
    "flags",
    [
        dict(action_target_gather=True),
        dict(action_cross_attention_layers=1),
        dict(value_attention_pool=True),
    ],
)
def test_each_flag_alone_warm_starts_exactly(flags):
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphNet

    base = EntityGraphNet(_base_config())
    up = EntityGraphNet(dataclasses.replace(_base_config(), **flags))
    up.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    up.eval()
    batch = _to_torch(_real_entity_batch(n_states=2))
    with torch.no_grad():
        ob = base(batch, return_q=True)
        ou = up(batch, return_q=True)
    for key in ("logits", "value", "final_vp", "q_values"):
        assert (ob[key] - ou[key]).abs().max().item() == 0.0, key
