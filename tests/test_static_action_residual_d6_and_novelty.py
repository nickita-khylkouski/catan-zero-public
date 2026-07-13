"""Independent adversarial proofs for the static-action residual treatment."""

from __future__ import annotations

import numpy as np
import torch

from catan_zero.rl.action_features import build_action_context_feature_table
from catan_zero.rl.entity_token_features import _legal_action_tokens
from catan_zero.rl.entity_token_policy import (
    STATIC_ACTION_RESIDUAL_SLICE,
    EntityGraphConfig,
    EntityGraphPolicy,
)
from catan_zero.rl.hex_symmetry import HexSymmetry, N_SYMMETRIES
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv
from catan_zero.rl.torch_ppo import build_action_feature_table
from tools.bench_entity_graph_stages import _synthetic_batch


def _add_topology_arrays(entity: dict[str, np.ndarray]) -> None:
    """Add valid-shaped incidence arrays required by HexSymmetry."""
    batch_size = int(entity["hex_tokens"].shape[0])
    entity["hex_vertex_ids"] = np.broadcast_to(
        np.arange(19 * 6, dtype=np.int16).reshape(1, 19, 6) % 54,
        (batch_size, 19, 6),
    ).copy()
    entity["hex_edge_ids"] = np.broadcast_to(
        np.arange(19 * 6, dtype=np.int16).reshape(1, 19, 6) % 72,
        (batch_size, 19, 6),
    ).copy()
    edges = np.arange(72, dtype=np.int16).reshape(1, 72, 1)
    entity["edge_vertex_ids"] = np.broadcast_to(
        np.concatenate((edges % 54, (edges + 1) % 54), axis=2),
        (batch_size, 72, 2),
    ).copy()
    event_width = int(entity["event_tokens"].shape[1])
    entity["event_target_ids"] = np.full(
        (batch_size, event_width, 4), -1, dtype=np.int16
    )


def _rank(value: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.asarray(value, dtype=np.float64), tol=1e-8))


def test_missing_static_slice_is_the_pareto_minimal_novel_catalog_surface():
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig())
    try:
        env.reset(seed=1)
        static = build_action_feature_table(env)
        structured = []
        for action_id in range(env.action_space.n):
            action = env.structured_action(action_id)
            structured.append(
                action
                if action is not None
                else {
                    "index": action_id,
                    "action_type": "",
                    "category": "",
                    "args": {},
                }
            )
        legal = _legal_action_tokens(
            env,
            {
                "structured_legal_actions": structured,
                "current_prompt": "PLAY_TURN",
                "trade_panel": {},
            },
            {},
        ).astype(np.float32)
        context = build_action_context_feature_table(
            env, {"valid_actions": tuple(range(env.action_space.n))}
        )
    finally:
        env.close()

    legacy = np.concatenate((legal, context), axis=1)
    missing = static[:, STATIC_ACTION_RESIDUAL_SLICE]
    full = np.concatenate((legacy, static), axis=1)
    repaired = np.concatenate((legacy, missing), axis=1)

    assert static.shape == (607, 45)
    assert len(np.unique(static, axis=0)) == 607
    assert _rank(static) == 41
    assert missing.shape == (607, 22)
    assert len(np.unique(missing, axis=0)) == 535
    assert _rank(repaired) - _rank(legacy) == 20
    assert _rank(full) - _rank(repaired) == 1
    # The one excluded rank is only the float32 catalog-id scalar versus its
    # existing fp16 legal-token copy; spending a full-table adapter on it would
    # confound the causal test with redundant features.
    assert _rank(full) - _rank(legacy) == 21


def _config(*, static_action_residual: bool) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=607,
        static_action_feature_size=45,
        hidden_size=32,
        state_layers=2,
        attention_heads=4,
        dropout=0.0,
        static_action_residual=static_action_residual,
    )


def _adversarial_symmetry() -> HexSymmetry:
    def identities(width: int) -> np.ndarray:
        return np.broadcast_to(
            np.arange(width, dtype=np.int64), (N_SYMMETRIES, width)
        ).copy()

    pi_act = identities(332)
    # A concrete nonidentity spatial action permutation is sufficient to catch
    # accidentally indexing the original legal IDs; the production D6 tables'
    # group/geometry laws are independently covered by test_hex_symmetry.py.
    pi_act[1, 0], pi_act[1, 1] = 1, 0
    return HexSymmetry(
        fwd_hex=identities(19),
        inv_hex=identities(19),
        fwd_vertex=identities(54),
        inv_vertex=identities(54),
        fwd_edge=identities(72),
        inv_edge=identities(72),
        pi_act=pi_act,
        canonical_hex_coord=np.zeros((19, 3), dtype=np.float32),
        op_names=tuple(str(index) for index in range(N_SYMMETRIES)),
    )


def test_d6_static_catalog_gather_uses_mapped_not_original_action_ids():
    symmetry = _adversarial_symmetry()
    orientation = 1
    spatial_id = int(
        np.flatnonzero(
            symmetry.pi_act[orientation]
            != np.arange(symmetry.pi_act.shape[1], dtype=np.int64)
        )[0]
    )
    mapped_id = int(symmetry.pi_act[orientation, spatial_id])
    nonspatial_id = int(symmetry.pi_act.shape[1] + 7)
    legal_ids = np.asarray([[spatial_id, nonspatial_id, -1]], dtype=np.int64)

    entity, _unused_ids, context = _synthetic_batch(
        batch_size=1,
        legal_width=3,
        valid_legal_fraction=1.0,
        event_width=0,
        valid_players=2,
        seed=17,
    )
    _add_topology_arrays(entity)
    entity["legal_action_mask"] = legal_ids >= 0
    entity["legal_action_target_ids"][0, 2] = -1
    rotated = symmetry.permute_entity_batch(
        entity,
        np.asarray([orientation]),
        legal_action_ids=legal_ids,
        action_size=607,
    )
    np.testing.assert_array_equal(
        rotated["_symmetry_legal_action_ids"],
        np.asarray([[mapped_id, nonspatial_id, -1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(legal_ids, [[spatial_id, nonspatial_id, -1]])

    catalog = np.arange(607 * 45, dtype=np.float32).reshape(607, 45)
    policy = EntityGraphPolicy(
        _config(static_action_residual=True), catalog, seed=3, device="cpu"
    )
    policy.model.eval()
    captured: list[torch.Tensor] = []
    hook = policy.model.static_action_residual_proj.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        policy.forward_legal_np(rotated, legal_ids, context, return_q=True)
    finally:
        hook.remove()

    expected = torch.from_numpy(
        catalog[[mapped_id, nonspatial_id]][:, STATIC_ACTION_RESIDUAL_SLICE]
    )
    assert len(captured) == 1
    assert torch.equal(captured[0][0, :2], expected)
    assert torch.count_nonzero(captured[0][0, 2]).item() == 0
    assert not torch.equal(
        captured[0][0, 0],
        torch.from_numpy(catalog[spatial_id, STATIC_ACTION_RESIDUAL_SLICE]),
    )


def test_static_projection_preserves_absolute_numeric_separation():
    policy = EntityGraphPolicy(
        _config(static_action_residual=True),
        np.zeros((607, 45), dtype=np.float32),
        seed=5,
        device="cpu",
    )
    projection = policy.model.static_action_residual_proj
    assert isinstance(projection, torch.nn.Linear)
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.zero_()
        projection.weight[0, 0] = 1.0
    # Adjacent high node IDs are deliberately used: a per-row LayerNorm made
    # these almost collinear (about 861x less separated), reducing the catalog
    # repair to epsilon-scale artifacts instead of an absolute target signal.
    features = torch.zeros(2, 22)
    features[:, 0] = torch.tensor([52.0 / 54.0, 53.0 / 54.0])
    output = projection(features)
    torch.testing.assert_close(
        output[1, 0] - output[0, 0],
        torch.tensor(1.0 / 54.0),
        rtol=1e-6,
        atol=1e-7,
    )


def test_legacy_policy_remains_invariant_to_arbitrary_static_catalog_bytes():
    entity, legal_ids, context = _synthetic_batch(
        batch_size=2,
        legal_width=7,
        valid_legal_fraction=0.8,
        event_width=0,
        valid_players=2,
        seed=19,
    )
    zeros = np.zeros((607, 45), dtype=np.float32)
    random = np.random.default_rng(23).normal(size=(607, 45)).astype(np.float32)
    left = EntityGraphPolicy(
        _config(static_action_residual=False), zeros, seed=29, device="cpu"
    )
    right = EntityGraphPolicy(
        _config(static_action_residual=False), random, seed=29, device="cpu"
    )
    left.model.eval()
    right.model.eval()

    with torch.no_grad():
        expected = left.forward_legal_np(entity, legal_ids, context, return_q=True)
        observed = right.forward_legal_np(entity, legal_ids, context, return_q=True)
    assert expected.keys() == observed.keys()
    for name in expected:
        assert torch.equal(observed[name], expected[name]), name
