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


def test_action_cross_cold_commissioning_opens_inner_gradients_first_backward():
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphNet
    from tools.train_bc import _initialize_cold_start_action_cross_attention_path

    model = EntityGraphNet(
        dataclasses.replace(_base_config(), action_cross_attention_layers=1)
    ).train()
    block = model.action_cross_blocks[0]
    assert torch.count_nonzero(block.attn.out_proj.weight).item() == 0
    assert torch.count_nonzero(block.ff[3].weight).item() == 0

    report = _initialize_cold_start_action_cross_attention_path(model)

    assert report["initialization"] == "cold_start_small_nonzero_identity"
    assert report["initial_scale"] == 0.01
    assert report["upgrade_artifact_zero_step_parity_preserved"] is True
    assert report["training_start_function_preserving"] is False
    assert torch.count_nonzero(block.attn.out_proj.weight).item() > 0
    assert torch.count_nonzero(block.ff[3].weight).item() > 0

    outputs = model(_to_torch(_real_entity_batch(n_states=2)))
    outputs["logits"].square().mean().backward()

    for projection in (block.attn.q_proj, block.attn.k_proj, block.attn.v_proj):
        assert projection.weight.grad is not None
        assert float(projection.weight.grad.abs().sum()) > 0.0
    assert block.ff[0].weight.grad is not None
    assert float(block.ff[0].weight.grad.abs().sum()) > 0.0


def test_transformer_action_cross_is_a_budgeted_adapter_not_a_second_tower():
    from catan_zero.rl.entity_token_policy import EntityGraphNet

    config = dataclasses.replace(_base_config(), action_cross_attention_layers=1)
    model = EntityGraphNet(config)
    cross_parameters = sum(
        parameter.numel() for parameter in model.action_cross_blocks.parameters()
    )

    # One full Transformer decoder block is roughly 12*h^2 parameters.  The
    # action-local join is deliberately a <=h^2 adapter so production V7 stays
    # inside the inherited 42.5-43M learner contract.
    assert cross_parameters <= config.hidden_size**2


def test_action_cross_cold_commissioning_preserves_moved_checkpoint():
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphNet
    from tools.train_bc import _initialize_cold_start_action_cross_attention_path

    model = EntityGraphNet(
        dataclasses.replace(_base_config(), action_cross_attention_layers=1)
    )
    block = model.action_cross_blocks[0]
    with torch.no_grad():
        block.attn.out_proj.weight[0, 0] = 0.125
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.action_cross_blocks.named_parameters()
    }

    report = _initialize_cold_start_action_cross_attention_path(model)

    assert report["initialization"] == "checkpoint_preserved"
    for name, parameter in model.action_cross_blocks.named_parameters():
        assert torch.equal(parameter, before[name])


def test_target_gather_uses_disjoint_local_id_namespaces_exactly():
    """Hex/vertex/edge/player ids are local, then receive sequence offsets."""

    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphNet

    model = EntityGraphNet(
        dataclasses.replace(_base_config(), action_target_gather=True)
    )
    # [CLS | 19 hex | 54 vertex | 72 edge | 4 player | 1 global].  Put the
    # absolute sequence index in the first channel so an incorrect namespace
    # offset is directly observable.
    sequence_length = 1 + 19 + 54 + 72 + 4 + 1
    tokens = torch.zeros(1, sequence_length, 64)
    tokens[0, :, 0] = torch.arange(sequence_length)
    targets = -torch.ones(1, 6, 4, dtype=torch.long)
    targets[0, 0, 0] = 3   # hex token 1 + 3
    targets[0, 1, 1] = 7   # vertex token 1 + 19 + 7
    targets[0, 2, 2] = 11  # edge token 1 + 19 + 54 + 11
    targets[0, 3, 3] = 1   # player token 1 + 19 + 54 + 72 + 1
    targets[0, 4, 0] = 2   # mean-pool a robber hex and victim player
    targets[0, 4, 3] = 0
    batch = {
        "hex_tokens": torch.zeros(1, 19, 1),
        "vertex_tokens": torch.zeros(1, 54, 1),
        "edge_tokens": torch.zeros(1, 72, 1),
        "legal_action_target_ids": targets,
    }

    pooled = model._gather_target_tokens(tokens, batch)
    expected = torch.tensor(
        [4.0, 27.0, 85.0, 147.0, (3.0 + 146.0) / 2.0, 0.0]
    )
    torch.testing.assert_close(pooled[0, :, 0], expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("column", "bad_value", "namespace_width"),
    [(0, 19, 19), (1, 54, 54), (2, 72, 72), (3, 4, 4), (1, -2, 54)],
)
def test_target_aware_policy_rejects_out_of_range_local_ids(
    column, bad_value, namespace_width
):
    from catan_zero.rl.entity_token_policy import _assert_entity_batch_shapes

    batch = _real_entity_batch(n_states=1)
    context = batch.pop("legal_action_context")
    legal_mask = np.asarray(batch["legal_action_mask"], dtype=np.bool_)
    legal_ids = np.full(legal_mask.shape, -1, dtype=np.int64)
    legal_ids[legal_mask] = np.arange(int(legal_mask.sum()), dtype=np.int64)
    batch["legal_action_target_ids"][0, 0, column] = bad_value
    config = dataclasses.replace(_base_config(), action_target_gather=True)

    with pytest.raises(
        ValueError,
        match=rf"column={column}.*namespace_width={namespace_width}",
    ):
        _assert_entity_batch_shapes(batch, legal_ids, context, config)


def test_target_aware_policy_rejects_target_on_padded_action():
    from catan_zero.rl.entity_token_policy import _assert_entity_batch_shapes

    batch = _real_entity_batch(n_states=2)
    context = batch.pop("legal_action_context")
    legal_mask = np.asarray(batch["legal_action_mask"], dtype=np.bool_)
    legal_ids = np.full(legal_mask.shape, -1, dtype=np.int64)
    for row in range(int(legal_mask.shape[0])):
        legal_ids[row, legal_mask[row]] = np.arange(
            int(legal_mask[row].sum()), dtype=np.int64
        )
    padded = np.argwhere(~legal_mask)
    if not len(padded):
        pytest.skip("fixture rows happened to have identical legal widths")
    row, action = padded[0]
    batch["legal_action_target_ids"][row, action, 0] = 0
    config = dataclasses.replace(_base_config(), action_target_gather=True)

    with pytest.raises(ValueError, match="padded legal action carries a target id"):
        _assert_entity_batch_shapes(batch, legal_ids, context, config)


def test_transport_filtered_entity_fields_are_exact_noop_with_all_optional_heads():
    """EvalServer may omit only fields no current model variant consumes.

    Exercise every optional forward branch together (including the target-id
    consumers and Q head) so a future field dependency cannot silently make
    the transport/device deny-lists unsafe.
    """
    import torch

    from catan_zero.rl.entity_token_policy import (
        EntityGraphPolicy,
        _NON_MODEL_ENTITY_KEYS,
    )
    from catan_zero.search.eval_server import _NON_FORWARD_ENTITY_KEYS

    assert _NON_FORWARD_ENTITY_KEYS == _NON_MODEL_ENTITY_KEYS - {
        "legal_action_mask"
    }

    full_entity = _real_entity_batch(n_states=3)
    context = full_entity.pop("legal_action_context")
    assert _NON_FORWARD_ENTITY_KEYS <= full_entity.keys()
    transport_entity = {
        key: value
        for key, value in full_entity.items()
        if key not in _NON_FORWARD_ENTITY_KEYS
    }

    legal_mask = np.asarray(full_entity["legal_action_mask"], dtype=np.bool_)
    legal_ids = np.full(legal_mask.shape, -1, dtype=np.int64)
    for row in range(int(legal_mask.shape[0])):
        width = int(legal_mask[row].sum())
        legal_ids[row, :width] = np.arange(width, dtype=np.int64)

    config = dataclasses.replace(
        _base_config(),
        value_uncertainty_head=True,
        action_target_gather=True,
        action_cross_attention_layers=2,
        value_attention_pool=True,
        value_categorical_bins=17,
        value_categorical_truncation_class=True,
        edge_policy_head=True,
        aux_subgoal_heads=True,
    )
    policy = EntityGraphPolicy(
        config,
        np.zeros(
            (int(config.action_size), int(config.static_action_feature_size)),
            dtype=np.float32,
        ),
        seed=123,
        device="cpu",
    )
    policy.model.eval()

    with torch.inference_mode():
        full_outputs = policy.forward_legal_np(
            full_entity, legal_ids, context, return_q=True
        )
        filtered_outputs = policy.forward_legal_np(
            transport_entity, legal_ids, context, return_q=True
        )

    assert full_outputs.keys() == filtered_outputs.keys()
    for key in full_outputs:
        assert torch.equal(full_outputs[key], filtered_outputs[key]), key


def test_entity_graph_policy_create_and_reload_preserve_action_cross_depth(
    tmp_path,
):
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    policy = EntityGraphPolicy.create(
        hidden_size=64,
        state_layers=2,
        attention_heads=4,
        dropout=0.0,
        action_cross_attention_layers=1,
        meaningful_public_history=True,
        event_history_limit=32,
        meaningful_public_history_target_gather=True,
        device="cpu",
    )
    assert policy.config.action_cross_attention_layers == 1
    assert policy.model.action_cross_attention_layers == 1
    assert len(policy.model.action_cross_blocks) == 1

    checkpoint = tmp_path / "cross-one.pt"
    policy.save(checkpoint)
    reloaded = EntityGraphPolicy.load(checkpoint, device="cpu")
    assert reloaded.config.action_cross_attention_layers == 1
    assert reloaded.model.action_cross_attention_layers == 1
    assert len(reloaded.model.action_cross_blocks) == 1
    assert set(reloaded.model.state_dict()) == set(policy.model.state_dict())
