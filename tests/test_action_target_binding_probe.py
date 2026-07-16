from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import action_target_binding_probe as probe  # noqa: E402

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V5,
)
from catan_zero.rl.meaningful_history import (  # noqa: E402
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
)
from catan_zero.rl.entity_token_features import (  # noqa: E402
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)


def _config(*, gather: bool = False) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=607,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=32,
        state_layers=1,
        attention_heads=4,
        dropout=0.0,
        action_target_gather=gather,
    )


def _entity(action_type: str, width: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260716)
    result: dict[str, np.ndarray] = {}
    for name, count, feature_width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", 8, EVENT_FEATURE_SIZE),
    ):
        result[f"{name}_tokens"] = rng.normal(
            size=(1, count, feature_width)
        ).astype(np.float32)
        if name != "global":
            result[f"{name}_mask"] = np.ones((1, count), dtype=np.bool_)
    result["legal_action_tokens"] = rng.normal(
        size=(1, width, LEGAL_ACTION_FEATURE_SIZE)
    ).astype(np.float32)
    result["legal_action_mask"] = np.ones((1, width), dtype=np.bool_)
    targets = np.full((1, width, 4), -1, dtype=np.int64)
    column = probe._TARGET_COLUMN_BY_ACTION_TYPE[action_type]
    namespace_width = (19, 54, 72, 4)[column]
    targets[0, :, column] = np.arange(width) % namespace_width
    result["legal_action_target_ids"] = targets
    return result


@pytest.mark.parametrize(
    "action_type",
    ["BUILD_SETTLEMENT", "BUILD_ROAD", "MOVE_ROBBER"],
)
def test_target_permutation_changes_only_same_namespace_targets(action_type):
    entity = _entity(action_type)
    permuted, changed = probe.permute_action_targets(
        entity, [action_type] * 6
    )

    assert changed == 6
    np.testing.assert_array_equal(
        permuted["legal_action_target_ids"][0, :, probe._TARGET_COLUMN_BY_ACTION_TYPE[action_type]],
        np.roll(
            entity["legal_action_target_ids"][
                0, :, probe._TARGET_COLUMN_BY_ACTION_TYPE[action_type]
            ],
            1,
        ),
    )
    for key in entity:
        if key != "legal_action_target_ids":
            np.testing.assert_array_equal(permuted[key], entity[key])


@pytest.mark.parametrize(
    ("action_type", "expected_key"),
    [
        ("BUILD_SETTLEMENT", "vertex_tokens"),
        ("BUILD_ROAD", "edge_tokens"),
        ("MOVE_ROBBER", "hex_tokens"),
    ],
)
def test_topology_permutation_changes_only_addressed_spatial_tokens(
    action_type, expected_key
):
    entity = _entity(action_type)
    permuted, keys = probe.permute_target_tokens(
        entity, [action_type] * 6
    )

    assert keys == (expected_key,)
    np.testing.assert_array_equal(
        permuted[expected_key], entity[expected_key][:, ::-1, :]
    )
    np.testing.assert_array_equal(
        permuted["legal_action_target_ids"],
        entity["legal_action_target_ids"],
    )


def test_warmstart_gather_is_zero_output_then_admits_target_gradient():
    base = EntityGraphPolicy(
        _config(),
        np.zeros((607, LEGAL_ACTION_FEATURE_SIZE), dtype=np.float32),
        seed=3,
        device="cpu",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V2,
    )
    base.trained_with_masked_hidden_info = True
    base.public_award_feature_contract = "authoritative_v1"
    base.entity_feature_adapter_binding_source = "checkpoint_metadata"
    base.model.eval()
    treatment = probe._warmstart_gather(base, seed=5)
    assert treatment.entity_feature_adapter_version == RUST_ENTITY_ADAPTER_V2
    assert treatment.trained_with_masked_hidden_info is True
    assert treatment.public_award_feature_contract == "authoritative_v1"
    assert treatment.entity_feature_adapter_binding_source == "checkpoint_metadata"
    entity = _entity("BUILD_SETTLEMENT")
    action_types = ["BUILD_SETTLEMENT"] * 6
    permuted, changed = probe.permute_action_targets(entity, action_types)
    assert changed == 6
    legal_ids = np.arange(6, dtype=np.int64)[None, :]
    context = np.zeros((1, 6, CONTEXT_ACTION_FEATURE_SIZE), dtype=np.float32)

    with torch.no_grad():
        base_logits = probe._forward(base, entity, legal_ids, context)
        warm_logits = probe._forward(treatment, entity, legal_ids, context)
        warm_permuted = probe._forward(treatment, permuted, legal_ids, context)
    assert torch.equal(base_logits, warm_logits)
    assert torch.equal(warm_logits, warm_permuted)

    target = probe._preferred_action_index(base_logits, action_types)
    loss = probe._cross_entropy(
        probe._forward(treatment, entity, legal_ids, context), target
    )
    loss.backward()
    gradient_l2 = probe._l2_norm(
        parameter.grad
        for parameter in treatment.model.target_gather_proj.parameters()
    )
    assert gradient_l2 > 0.0

    optimizer = torch.optim.SGD(
        treatment.model.target_gather_proj.parameters(), lr=0.1
    )
    optimizer.step()
    with torch.no_grad():
        learned = probe._forward(treatment, entity, legal_ids, context)
        learned_permuted = probe._forward(
            treatment, permuted, legal_ids, context
        )
    assert not torch.equal(learned, learned_permuted)


def test_warmstart_rejects_checkpoint_that_already_has_gather():
    policy = EntityGraphPolicy(
        dataclasses.replace(_config(), action_target_gather=True),
        np.zeros((607, LEGAL_ACTION_FEATURE_SIZE), dtype=np.float32),
        seed=7,
        device="cpu",
    )
    with pytest.raises(probe.ProbeError, match="already enables"):
        probe._warmstart_gather(policy, seed=11)


def test_root_inputs_use_checkpoint_feature_contract(monkeypatch):
    config = dataclasses.replace(
        _config(),
        meaningful_public_history=True,
        meaningful_public_history_schema=MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
        event_history_limit=19,
    )
    policy = EntityGraphPolicy(
        config,
        np.zeros((607, LEGAL_ACTION_FEATURE_SIZE), dtype=np.float32),
        seed=13,
        device="cpu",
        entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V5,
    )
    policy.trained_with_masked_hidden_info = True

    class Game:
        @staticmethod
        def current_color():
            return "BLUE"

        @staticmethod
        def playable_action_indices(_colors, _catalog):
            return [1, 2]

        @staticmethod
        def playable_actions_json():
            return '[["BLUE", "BUILD_ROAD"], ["BLUE", "BUILD_ROAD"]]'

    observed = {}
    monkeypatch.setattr(
        probe, "rust_policy_action_ids", lambda *_args, **_kwargs: (10, 20)
    )

    def entity(*_args, **kwargs):
        observed["entity"] = kwargs
        return {"tokens": np.zeros((1, 1, 1), dtype=np.float32)}

    def context(*_args, **kwargs):
        observed["context"] = kwargs
        return np.zeros((1, 2, CONTEXT_ACTION_FEATURE_SIZE), dtype=np.float32)

    monkeypatch.setattr(probe, "rust_game_to_entity_batch", entity)
    monkeypatch.setattr(probe, "rust_action_context_batch", context)

    _entity_batch, legal_ids, _context, action_types = probe._root_inputs(
        policy, Game(), context_fill=-4.0
    )

    assert legal_ids.tolist() == [[10, 20]]
    assert action_types == ("BUILD_ROAD", "BUILD_ROAD")
    assert observed["entity"]["public_observation"] is True
    assert observed["entity"]["meaningful_public_history"] is True
    assert observed["entity"]["history_limit"] == 19
    assert (
        observed["entity"]["meaningful_public_history_schema"]
        == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
    )
    assert (
        observed["entity"]["entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V5
    )
    assert observed["context"]["fill"] == -4.0
    assert observed["context"]["public_observation"] is True
    assert (
        observed["context"]["entity_feature_adapter_version"]
        == RUST_ENTITY_ADAPTER_V5
    )


def test_target_identity_distinguishes_representation_identical_opening_edges():
    from catan_zero.rl.entity_token_policy import EntityGraphNet

    model = EntityGraphNet(dataclasses.replace(_config(), action_target_gather=True))
    targets = torch.full((1, 3, 4), -1, dtype=torch.long)
    targets[0, :, 2] = torch.tensor([4, 17, 63])

    identity = model._action_target_local_identity(
        targets,
        width=int(model.config.hidden_size),
        dtype=torch.float32,
    )

    assert not torch.equal(identity[0, 0], identity[0, 1])
    assert not torch.equal(identity[0, 1], identity[0, 2])
    # D6 passes relabelled target ids to this function. A changed canonical
    # edge id therefore changes the identity immediately; there is no cached
    # pre-symmetry id that could leave the action bound to the old edge.
    relabelled = targets.clone()
    relabelled[0, :, 2] = torch.tensor([8, 29, 50])
    relabelled_identity = model._action_target_local_identity(
        relabelled,
        width=int(model.config.hidden_size),
        dtype=torch.float32,
    )
    assert not torch.equal(identity, relabelled_identity)
