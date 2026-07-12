"""Correctness tests for the D6 hex-symmetry tables (f74).

These prove the permutations are a valid group action that commutes with the
entity featurization, without needing to rotate a live game: the board
incidence graph is an automorphism target, and the model's per-action
target-token gather is shown to be invariant under the relabelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("catanatron_rs")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from catan_zero.rl.hex_symmetry import (  # noqa: E402
    N_SYMMETRIES,
    build_hex_symmetry,
)
from catan_zero.rl import entity_token_features as etf  # noqa: E402


@pytest.fixture(scope="module")
def sym():
    return build_hex_symmetry()


@pytest.fixture(scope="module")
def real_entity():
    """A real full-width opening placement root, rust-featurized (B=1)."""
    pytest.importorskip("catanatron_rs")
    from catan_zero.search.rust_mcts import _require_rust_module
    from catan_zero.search.neural_rust_mcts import (
        rust_game_to_entity_batch,
        rust_policy_action_ids,
    )

    try:
        rs = _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))
    colors = ("RED", "BLUE")
    game = rs.Game.simple(list(colors), seed=1)
    legal = tuple(int(a) for a in game.playable_action_indices(list(colors), None))
    pids = rust_policy_action_ids(game, legal, colors=colors, action_size=332)
    entity = rust_game_to_entity_batch(
        game, legal, actor=str(game.current_color()), colors=colors,
        action_size=332, policy_action_ids=pids,
    )
    return {k: np.asarray(v) for k, v in entity.items()}


# --- layout constants agree with the featurizer -----------------------------

def test_feature_layout_constants():
    assert etf.HEX_FEATURE_SIZE == 13
    assert etf.EVENT_FEATURE_SIZE == 41
    # coord occupies dims 1:4, event action-id is dim 35 (see featurizer source).
    from catan_zero.rl.hex_symmetry import (
        _EVENT_ACTION_ID_DIM,
        _HEX_COORD_SLICE,
        _LEGAL_ACTION_ID_DIM,
    )
    assert (_HEX_COORD_SLICE.start, _HEX_COORD_SLICE.stop) == (1, 4)
    assert _EVENT_ACTION_ID_DIM == 35
    assert _LEGAL_ACTION_ID_DIM == 1


# --- group laws --------------------------------------------------------------

def test_identity_present(sym):
    assert list(sym.fwd_hex[0]) == list(range(19))
    assert list(sym.fwd_vertex[0]) == list(range(54))
    assert list(sym.fwd_edge[0]) == list(range(72))
    assert list(sym.pi_act[0]) == list(range(sym.pi_act.shape[1]))


@pytest.mark.parametrize("table_name", ["fwd_hex", "fwd_vertex", "fwd_edge", "pi_act"])
def test_all_permutations_are_bijections(sym, table_name):
    table = getattr(sym, table_name)
    n = table.shape[1]
    for g in range(N_SYMMETRIES):
        assert sorted(table[g].tolist()) == list(range(n)), (table_name, g)


def test_inverse_tables(sym):
    for g in range(N_SYMMETRIES):
        assert sym.fwd_hex[g][sym.inv_hex[g]].tolist() == list(range(19))
        assert sym.fwd_vertex[g][sym.inv_vertex[g]].tolist() == list(range(54))
        assert sym.fwd_edge[g][sym.inv_edge[g]].tolist() == list(range(72))


def test_group_closed_under_composition(sym):
    """Composing any two elements yields another element of the 12-set, for
    every permutation table simultaneously (a genuine group representation)."""
    def as_tuple(g):
        return (
            tuple(sym.fwd_hex[g]),
            tuple(sym.fwd_vertex[g]),
            tuple(sym.fwd_edge[g]),
            tuple(sym.pi_act[g]),
        )

    members = {as_tuple(g) for g in range(N_SYMMETRIES)}
    assert len(members) == N_SYMMETRIES  # all 12 distinct
    for g1 in range(N_SYMMETRIES):
        for g2 in range(N_SYMMETRIES):
            comp = (
                tuple(sym.fwd_hex[g2][sym.fwd_hex[g1]]),
                tuple(sym.fwd_vertex[g2][sym.fwd_vertex[g1]]),
                tuple(sym.fwd_edge[g2][sym.fwd_edge[g1]]),
                tuple(sym.pi_act[g2][sym.pi_act[g1]]),
            )
            assert comp in members, (g1, g2)


def test_non_spatial_actions_are_fixed_points(sym):
    """ROLL / END_TURN / dev cards / trades must map to themselves."""
    from catan_zero.rl.hex_symmetry import _build_geometry
    geom = _build_geometry()
    ActionType = geom["ActionType"]
    spatial = {
        ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY,
        ActionType.BUILD_ROAD, ActionType.MOVE_ROBBER,
    }
    for i, (atype, _value) in enumerate(geom["action_array"]):
        if atype not in spatial:
            for g in range(N_SYMMETRIES):
                assert sym.pi_act[g, i] == i, (i, atype, g)


# --- incidence automorphism (core correctness) ------------------------------

def test_incidence_automorphism(sym, real_entity):
    """Applying a symmetry to a featurized state preserves the board incidence:
    the permuted (hex,vertex),(hex,edge),(edge,vertex) relations equal the
    originals relabelled -- i.e. each element is an automorphism of the graph."""
    hv0 = real_entity["hex_vertex_ids"][0]
    he0 = real_entity["hex_edge_ids"][0]
    ev0 = real_entity["edge_vertex_ids"][0]

    def incidence_set(hex_vertex, hex_edge, edge_vertex):
        hv = {(a, int(v)) for a in range(19) for v in hex_vertex[a] if v >= 0}
        he = {(a, int(e)) for a in range(19) for e in hex_edge[a] if e >= 0}
        ev = {(int(e), int(v)) for e in range(72) for v in edge_vertex[e] if v >= 0}
        return hv, he, ev

    base = incidence_set(hv0, he0, ev0)
    for g in range(N_SYMMETRIES):
        out = sym.permute_entity_batch(real_entity, g)
        got = incidence_set(
            out["hex_vertex_ids"][0], out["hex_edge_ids"][0], out["edge_vertex_ids"][0]
        )
        # Relabel the base incidence by the forward permutations and compare.
        exp_hv = {(int(sym.fwd_hex[g, a]), int(sym.fwd_vertex[g, v])) for (a, v) in base[0]}
        exp_he = {(int(sym.fwd_hex[g, a]), int(sym.fwd_edge[g, e])) for (a, e) in base[1]}
        exp_ev = {(int(sym.fwd_edge[g, e]), int(sym.fwd_vertex[g, v])) for (e, v) in base[2]}
        assert got == (exp_hv, exp_he, exp_ev), g


# --- featurization commutes with the target-token gather --------------------

def test_target_token_gather_is_invariant(sym, real_entity):
    """The model gathers, per legal action, the token at its target id. Under a
    symmetry that token content must be identical (same intrinsic features),
    which is exactly what makes the per-row policy logit invariant."""
    tids0 = real_entity["legal_action_target_ids"][0]  # (A,4)
    vtok0 = real_entity["vertex_tokens"][0]
    etok0 = real_entity["edge_tokens"][0]
    htok0 = real_entity["hex_tokens"][0]
    for g in range(N_SYMMETRIES):
        out = sym.permute_entity_batch(real_entity, g)
        tids = out["legal_action_target_ids"][0]
        vtok = out["vertex_tokens"][0]
        etok = out["edge_tokens"][0]
        htok = out["hex_tokens"][0]
        for row in range(tids0.shape[0]):
            # vertex target
            v0 = int(tids0[row, 1])
            if v0 >= 0:
                v1 = int(tids[row, 1])
                assert np.array_equal(vtok[v1], vtok0[v0]), (g, row, "vertex")
            e0 = int(tids0[row, 2])
            if e0 >= 0:
                e1 = int(tids[row, 2])
                assert np.array_equal(etok[e1], etok0[e0]), (g, row, "edge")
            h0 = int(tids0[row, 0])
            if h0 >= 0:
                h1 = int(tids[row, 0])
                # hex intrinsic dims (all but the coord slice) must match.
                intr = [d for d in range(13) if d not in (1, 2, 3)]
                assert np.array_equal(htok[h1][intr], htok0[h0][intr]), (g, row, "hex")


def test_legal_action_id_feature_moves_with_board(sym, real_entity):
    """Rows stay aligned, but each row must encode its transformed action id."""

    action_size = sym.pi_act.shape[1]
    tokens = real_entity["legal_action_tokens"]
    ids = np.rint(tokens[:, :, 1].astype(np.float64) * action_size).astype(np.int64)
    assert np.all((ids >= 0) & (ids < action_size))
    for g in range(N_SYMMETRIES):
        out = sym.permute_entity_batch(
            real_entity,
            g,
            legal_action_ids=ids,
            action_size=action_size,
        )
        expected = sym.pi_act[g, ids] / float(action_size)
        assert np.allclose(
            out["legal_action_tokens"][:, :, 1].astype(np.float64),
            expected,
            atol=5e-4,
        )


def test_legal_action_relabel_refuses_wrong_action_space(sym, real_entity):
    ids = np.zeros(real_entity["legal_action_tokens"].shape[:2], dtype=np.int64)
    with pytest.raises(ValueError, match="permutation exceeds action_size"):
        sym.permute_entity_batch(
            real_entity,
            0,
            legal_action_ids=ids,
            action_size=sym.pi_act.shape[1] - 1,
        )


def test_nonspatial_extended_action_id_stays_fixed(sym, real_entity):
    extended_size = sym.pi_act.shape[1] + 235  # production 332 + 235 = 567
    extended_id = sym.pi_act.shape[1]
    entity = {key: np.array(value, copy=True) for key, value in real_entity.items()}
    ids = np.full(entity["legal_action_tokens"].shape[:2], extended_id, dtype=np.int64)
    entity["legal_action_tokens"][:, :, 1] = extended_id / float(extended_size)
    out = sym.permute_entity_batch(
        entity,
        1,
        legal_action_ids=ids,
        action_size=extended_size,
    )
    assert np.allclose(
        out["legal_action_tokens"][:, :, 1],
        extended_id / float(extended_size),
        atol=5e-4,
    )


def test_event_action_zero_relabels_when_masked_present(sym, real_entity):
    entity = {key: np.array(value, copy=True) for key, value in real_entity.items()}
    entity["event_tokens"][0, -1, 35] = 0.0  # exact action id 0
    entity["event_mask"][0, -1] = True
    g = next(index for index in range(N_SYMMETRIES) if sym.pi_act[index, 0] != 0)
    out = sym.permute_entity_batch(entity, g, relabel_events=True)
    assert out["event_tokens"][0, -1, 35] == pytest.approx(
        sym.pi_act[g, 0] / 607.0,
        abs=5e-4,
    )


def test_intrinsic_features_are_a_pure_permutation(sym, real_entity):
    """Vertex/edge token rows are only reordered (multiset preserved); hex rows
    likewise except the slot-fixed coordinate is restored to canonical."""
    for g in range(N_SYMMETRIES):
        out = sym.permute_entity_batch(real_entity, g)
        assert np.array_equal(
            np.sort(out["vertex_tokens"][0], axis=0),
            np.sort(real_entity["vertex_tokens"][0], axis=0),
        )
        assert np.array_equal(
            np.sort(out["edge_tokens"][0], axis=0),
            np.sort(real_entity["edge_tokens"][0], axis=0),
        )
        # coordinate feature stays canonical (equal to the pre-permute board's).
        assert np.allclose(
            out["hex_tokens"][0][:, 1:4].astype(np.float32),
            sym.canonical_hex_coord,
            atol=1e-3,
        )


def test_round_trip_inverse(sym, real_entity):
    """g followed by g^{-1} recovers the original tensors exactly."""
    # inverse of g is the element h with fwd[h] = inv[g]; find it per table via act.
    for g in range(N_SYMMETRIES):
        # locate inverse index by matching pi_act.
        inv_g = next(
            h for h in range(N_SYMMETRIES)
            if np.array_equal(sym.pi_act[h][sym.pi_act[g]], np.arange(sym.pi_act.shape[1]))
        )
        once = sym.permute_entity_batch(real_entity, g)
        back = sym.permute_entity_batch(once, inv_g)
        for key in ("vertex_tokens", "edge_tokens", "hex_tokens",
                    "hex_vertex_ids", "edge_vertex_ids", "legal_action_target_ids"):
            assert np.array_equal(back[key], real_entity[key]), (g, key)


def test_per_row_random_g_matches_scalar(sym, real_entity):
    """Per-row g application equals applying each g individually (used by
    training augmentation to give each sample its own orientation)."""
    tiled = {k: np.repeat(v, N_SYMMETRIES, axis=0) for k, v in real_entity.items()}
    g_per_row = np.arange(N_SYMMETRIES)
    batched = sym.permute_entity_batch(tiled, g_per_row)
    for g in range(N_SYMMETRIES):
        single = sym.permute_entity_batch(real_entity, g)
        for key in ("hex_tokens", "vertex_tokens", "edge_tokens",
                    "hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids",
                    "legal_action_target_ids"):
            assert np.array_equal(batched[key][g], single[key][0]), (g, key)
