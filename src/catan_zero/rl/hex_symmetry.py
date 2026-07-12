"""D6 hex-board symmetry for the entity_graph model (f74).

The catanatron 2p board is a regular hexagon: its tile/node/edge incidence
graph is invariant under the dihedral group **D6** (6 rotations x 2 reflections
= 12 elements) acting on the hex cube-coordinate lattice. The entity_graph
trunk is a set transformer whose only positional signal is the per-type token
embedding, so a board symmetry acts on a *featurized* state as a pure
permutation of the hex/vertex/edge token rows plus a relabelling of the id
tables -- with exactly two orientation-dependent scalar features to fix up:

  * ``hex_tokens[:, 1:4]`` -- the tile cube-coordinate ``/4``. It is a fixed
    function of the tile *slot*, so after permuting rows we simply restore the
    canonical per-slot coordinate (no matrix multiply needed).
  * ``event_tokens[:, 35]`` -- a scaled past-action id encoding a board target.
    Optionally relabelled through the action permutation ``pi_act``.

Every other feature is intrinsic to its entity (resource, ownership, pips,
ports, ...) and rides along with the permuted token row unchanged.

The permutation tables are derived once from the vendored catanatron BASE map
geometry and cached. They are verified (see ``tests/test_hex_symmetry.py``) to
(1) form a group closed under composition, (2) be automorphisms of the board
incidence tables, and (3) round-trip against the live rust featurizer id space.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_SYMMETRIES = 12

# Feature layout constants mirrored from entity_token_features (kept local to
# avoid a heavy import at table-build time; asserted against it in tests).
_HEX_COORD_SLICE = slice(1, 4)
_LEGAL_ACTION_ID_DIM = 1
_EVENT_ACTION_ID_DIM = 35
_EVENT_ACTION_ID_SCALE = 607.0


def _import_catanatron():
    """Import pure-python catanatron, falling back to the vendored copy.

    The repo ships catanatron under ``vendor/`` and the editable install may
    point elsewhere, so we add the vendored path if the plain import fails.
    """
    try:
        from catanatron.models import map as _map  # noqa: F401
        from catanatron.models import coordinate_system as _cs  # noqa: F401
        return
    except ImportError:
        pass
    repo_root = Path(__file__).resolve().parents[3]
    vendored = repo_root / "vendor" / "catanatron" / "catanatron"
    if vendored.is_dir() and str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))


# ---- D6 action on cube coordinates -----------------------------------------
# x + y + z = 0. A 60-degree rotation is (x, y, z) -> (-z, -x, -y); a reflection
# is (x, y, z) -> (x, z, y). The 12 group elements are r^k and r^k . s.

def _rot60(c):
    x, y, z = c
    return (-z, -x, -y)


def _refl(c):
    x, y, z = c
    return (x, z, y)


def _cube_ops():
    ops = []
    for k in range(6):
        def rot_k(c, k=k):
            for _ in range(k):
                c = _rot60(c)
            return c
        ops.append(rot_k)
    reflected = []
    for f in list(ops):
        reflected.append(lambda c, f=f: _refl(f(c)))
    return ops + reflected


# Corner (NodeRef) -> the two tile-neighbour directions flanking that corner.
# Derived from catanatron.models.map.get_nodes_and_edges neighbour logic: the
# three hexes meeting at a corner are the tile itself plus these two neighbours.
_NODE_FLANK = {
    "NORTH": ("NORTHWEST", "NORTHEAST"),
    "NORTHEAST": ("NORTHEAST", "EAST"),
    "SOUTHEAST": ("EAST", "SOUTHEAST"),
    "SOUTH": ("SOUTHEAST", "SOUTHWEST"),
    "SOUTHWEST": ("SOUTHWEST", "WEST"),
    "NORTHWEST": ("WEST", "NORTHWEST"),
}


@dataclass(frozen=True)
class HexSymmetry:
    """Precomputed D6 permutation tables for the entity_graph feature space.

    Arrays are ``int64`` with a leading axis of length ``N_SYMMETRIES`` (12).

      * ``fwd_*[g, a] = b``  -- entity ``a`` moves to slot ``b`` under g.
      * ``inv_*[g, b] = a``  -- slot ``b`` after g came from entity ``a`` (row gather).

    ``pi_act`` maps global policy-action ids; non-spatial actions are fixed
    points. ``canonical_hex_coord`` is the per-slot ``coordinate/4`` restored
    into ``hex_tokens[:, 1:4]`` after a row permutation.
    """

    fwd_hex: np.ndarray
    inv_hex: np.ndarray
    fwd_vertex: np.ndarray
    inv_vertex: np.ndarray
    fwd_edge: np.ndarray
    inv_edge: np.ndarray
    pi_act: np.ndarray
    canonical_hex_coord: np.ndarray
    op_names: tuple

    # -- id-value remap helpers (guard the -1 sentinel) -----------------------
    @staticmethod
    def _remap_values(ids: np.ndarray, table_g: np.ndarray) -> np.ndarray:
        """Map id *values* in ``ids`` (any shape, values in [-1, K)) through the
        per-row lookup ``table_g`` of shape ``(B, K)``; -1 is preserved."""
        b = ids.shape[0]
        clipped = np.where(ids < 0, 0, ids)
        rows = np.arange(b).reshape((b,) + (1,) * (ids.ndim - 1))
        mapped = table_g[rows, clipped]
        return np.where(ids < 0, -1, mapped).astype(ids.dtype)

    @staticmethod
    def _gather_rows(arr: np.ndarray, inv_sel: np.ndarray) -> np.ndarray:
        """Reorder axis-1 of ``arr`` (B, K, ...) by per-row ``inv_sel`` (B, K)."""
        b = arr.shape[0]
        return arr[np.arange(b)[:, None], inv_sel]

    def permute_entity_batch(
        self,
        entity: dict,
        g,
        *,
        relabel_events: bool = True,
        legal_action_ids: np.ndarray | None = None,
        action_size: int | None = None,
    ) -> dict:
        """Return a new entity dict with symmetry ``g`` applied.

        ``g`` is an int (whole batch) or an int array of shape ``(B,)`` (per
        row). Legal-action rows are NOT reordered -- only their target ids are
        relabelled -- so any per-legal-row target/weight arrays kept elsewhere
        stay aligned and need no change. Value/outcome targets are invariant.
        """
        out = dict(entity)
        b = int(np.asarray(entity["hex_tokens"]).shape[0])
        g_arr = np.broadcast_to(np.asarray(g, dtype=np.int64), (b,))

        inv_hex = self.inv_hex[g_arr]        # (B,19)
        inv_vertex = self.inv_vertex[g_arr]  # (B,54)
        inv_edge = self.inv_edge[g_arr]      # (B,72)
        fwd_hex = self.fwd_hex[g_arr]
        fwd_vertex = self.fwd_vertex[g_arr]
        fwd_edge = self.fwd_edge[g_arr]

        # Token rows: gather by inverse permutation.
        hex_tokens = self._gather_rows(np.asarray(entity["hex_tokens"]), inv_hex).copy()
        # Restore canonical (slot-fixed) coordinate feature.
        hex_tokens[:, :, _HEX_COORD_SLICE] = self.canonical_hex_coord[None].astype(hex_tokens.dtype)
        out["hex_tokens"] = hex_tokens
        out["vertex_tokens"] = self._gather_rows(np.asarray(entity["vertex_tokens"]), inv_vertex)
        out["edge_tokens"] = self._gather_rows(np.asarray(entity["edge_tokens"]), inv_edge)

        # Incidence id tables: rows permuted, values relabelled.
        hv = self._gather_rows(np.asarray(entity["hex_vertex_ids"]), inv_hex)
        out["hex_vertex_ids"] = self._remap_values(hv, fwd_vertex)
        he = self._gather_rows(np.asarray(entity["hex_edge_ids"]), inv_hex)
        out["hex_edge_ids"] = self._remap_values(he, fwd_edge)
        ev = self._gather_rows(np.asarray(entity["edge_vertex_ids"]), inv_edge)
        out["edge_vertex_ids"] = self._remap_values(ev, fwd_vertex)

        # Masks ride with their tokens.
        for key, inv_sel in (
            ("hex_mask", inv_hex),
            ("vertex_mask", inv_vertex),
            ("edge_mask", inv_edge),
        ):
            if key in entity:
                out[key] = self._gather_rows(np.asarray(entity[key]), inv_sel)

        # Legal-action rows stay fixed so row-wise policy targets remain
        # aligned, but the action's GLOBAL spatial identity must move with the
        # board.  Before this correction only target ids moved; the incumbent
        # ignores those target ids, so D6 evaluated a rotated board against the
        # original node/edge action-id feature.  Use the exact integer ids
        # supplied by the caller rather than reverse-engineering fp16-scaled
        # tokens.  Current D6 tables are the 2p BASE action space and therefore
        # fail closed on a different action-space width.
        if legal_action_ids is not None:
            ids = np.asarray(legal_action_ids, dtype=np.int64)
            if ids.shape[:2] != np.asarray(entity["legal_action_tokens"]).shape[:2]:
                raise ValueError(
                    "legal_action_ids shape does not match legal_action_tokens: "
                    f"{ids.shape} vs {np.asarray(entity['legal_action_tokens']).shape}"
                )
            resolved_action_size = int(action_size or self.pi_act.shape[1])
            if resolved_action_size != int(self.pi_act.shape[1]):
                raise ValueError(
                    "D6 action permutation width does not match action_size: "
                    f"{self.pi_act.shape[1]} != {resolved_action_size}"
                )
            valid = ids >= 0
            if np.any(ids[valid] >= self.pi_act.shape[1]):
                raise ValueError("legal_action_ids contain an id outside D6 action space")
            clipped = np.where(valid, ids, 0)
            mapped = self.pi_act[g_arr[:, None], clipped]
            action_tokens = np.asarray(entity["legal_action_tokens"]).copy()
            action_tokens[:, :, _LEGAL_ACTION_ID_DIM] = np.where(
                valid,
                mapped / float(resolved_action_size),
                action_tokens[:, :, _LEGAL_ACTION_ID_DIM],
            ).astype(action_tokens.dtype)
            out["legal_action_tokens"] = action_tokens

        # Legal-action target ids: rows fixed, columns relabelled by kind.
        if "legal_action_target_ids" in entity:
            tids = np.asarray(entity["legal_action_target_ids"]).copy()
            # cols: 0 hex, 1 vertex, 2 edge, 3 player (unchanged).
            tids[:, :, 0] = self._remap_values(tids[:, :, 0], fwd_hex)
            tids[:, :, 1] = self._remap_values(tids[:, :, 1], fwd_vertex)
            tids[:, :, 2] = self._remap_values(tids[:, :, 2], fwd_edge)
            out["legal_action_target_ids"] = tids

        # Event action-id scalar (optional; history feature only).
        if relabel_events and "event_tokens" in entity:
            ev_tok = np.asarray(entity["event_tokens"]).copy()
            scaled = ev_tok[:, :, _EVENT_ACTION_ID_DIM].astype(np.float32)
            ids = np.rint(scaled * _EVENT_ACTION_ID_SCALE).astype(np.int64)
            # Action id 0 is a valid spatial action, not an absence sentinel.
            # Event occupancy is carried by event_mask; using ``ids > 0``
            # silently left action 0 unrotated once real history is enabled.
            present = np.asarray(entity.get("event_mask", ids >= 0)).astype(bool)
            if present.shape != ids.shape:
                raise ValueError(
                    f"event_mask shape {present.shape} != event action ids {ids.shape}"
                )
            ids_c = np.clip(ids, 0, self.pi_act.shape[1] - 1)
            new_ids = self.pi_act[g_arr[:, None], ids_c]
            new_scaled = np.where(present, new_ids / _EVENT_ACTION_ID_SCALE, scaled)
            ev_tok[:, :, _EVENT_ACTION_ID_DIM] = new_scaled.astype(ev_tok.dtype)
            out["event_tokens"] = ev_tok

        return out

    def orientations_entity(
        self,
        entity: dict,
        *,
        relabel_events: bool = True,
        legal_action_ids: np.ndarray | None = None,
        action_size: int | None = None,
    ) -> dict:
        """Expand a single-state (B=1) entity dict into all ``N_SYMMETRIES``
        orientations, returning a B=N_SYMMETRIES batch (orientation 0 is the
        canonical/identity state). Used by the test-time value/prior denoiser."""
        b = int(np.asarray(entity["hex_tokens"]).shape[0])
        if b != 1:
            raise ValueError(f"orientations_entity expects B=1, got B={b}")
        tiled = {
            k: (np.repeat(np.asarray(v), N_SYMMETRIES, axis=0)
                if np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] == 1
                else np.asarray(v))
            for k, v in entity.items()
        }
        return self.permute_entity_batch(
            tiled,
            np.arange(N_SYMMETRIES),
            relabel_events=relabel_events,
            legal_action_ids=(
                None
                if legal_action_ids is None
                else np.repeat(np.asarray(legal_action_ids), N_SYMMETRIES, axis=0)
            ),
            action_size=action_size,
        )

    def average_forward(
        self,
        entity: dict,
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
        forward_fn,
        *,
        return_q: bool = False,
        relabel_events: bool = True,
    ) -> dict:
        """Denoise a single-state evaluation by averaging the net over all 12
        board orientations (the ``sqrt(12)`` value-noise reducer).

        ``entity`` is a B=1 entity dict. ``forward_fn(entity_n, legal_ids_n,
        context_n, return_q)`` must return a dict of numpy arrays with keys
        ``logits`` (N, A), ``value`` (N,), and optionally ``q_values`` (N, A).

        Because legal-action rows keep their order under every symmetry, the
        per-candidate outputs are already aligned across orientations, so the
        averaged prior/q is a direct column mean -- no inverse action
        permutation is required. Returns the averaged ``value`` (scalar),
        ``logits`` (A,), ``q_values`` (A,), plus the raw per-orientation arrays.
        """
        ent_n = self.orientations_entity(
            entity,
            relabel_events=relabel_events,
            legal_action_ids=legal_action_ids,
            action_size=self.pi_act.shape[1],
        )
        legal_n = np.repeat(np.asarray(legal_action_ids), N_SYMMETRIES, axis=0)
        ctx_n = np.repeat(np.asarray(legal_action_context), N_SYMMETRIES, axis=0)
        out = forward_fn(ent_n, legal_n, ctx_n, return_q)
        logits = np.asarray(out["logits"], dtype=np.float64)      # (N, A)
        value = np.asarray(out["value"], dtype=np.float64)        # (N,)
        result = {
            "value": float(value.mean()),
            "logits": logits.mean(axis=0),
            "value_per_orientation": value,
            "logits_per_orientation": logits,
        }
        if return_q and out.get("q_values") is not None:
            q = np.asarray(out["q_values"], dtype=np.float64)     # (N, A)
            result["q_values"] = q.mean(axis=0)
            result["q_values_per_orientation"] = q
        return result


def _build_geometry():
    _import_catanatron()
    from catanatron.models.coordinate_system import Direction, UNIT_VECTORS, add
    from catanatron.models.map import build_map
    from catanatron.gym.envs.action_space import get_action_array
    from catanatron.models.enums import ActionType
    from catanatron.models.player import Color

    uv = {d.value: UNIT_VECTORS[d] for d in Direction}
    catan_map = build_map("BASE")
    land = catan_map.land_tiles
    id2coord = {t.id: c for c, t in land.items()}
    coord2id = {c: i for i, c in id2coord.items()}

    node_triples = {}
    for c, tile in land.items():
        for noderef, nid in tile.nodes.items():
            d1, d2 = _NODE_FLANK[noderef.value]
            node_triples[int(nid)] = frozenset({c, add(c, uv[d1]), add(c, uv[d2])})
    trip2node = {v: k for k, v in node_triples.items()}

    edge_pairs = set()
    for c, tile in land.items():
        for edge in tile.edges.values():
            edge_pairs.add(tuple(sorted(edge)))
    edge_to_id = {e: i for i, e in enumerate(sorted(edge_pairs))}

    action_array = get_action_array((Color.RED, Color.BLUE), "BASE")
    action_to_index = {a: i for i, a in enumerate(action_array)}

    return {
        "id2coord": id2coord,
        "coord2id": coord2id,
        "node_triples": node_triples,
        "trip2node": trip2node,
        "edge_to_id": edge_to_id,
        "action_array": action_array,
        "action_to_index": action_to_index,
        "ActionType": ActionType,
    }


@functools.lru_cache(maxsize=1)
def build_hex_symmetry() -> HexSymmetry:
    geom = _build_geometry()
    ops = _cube_ops()
    op_names = tuple([f"r{k}" for k in range(6)] + [f"r{k}s" for k in range(6)])
    id2coord = geom["id2coord"]
    coord2id = geom["coord2id"]
    node_triples = geom["node_triples"]
    trip2node = geom["trip2node"]
    edge_to_id = geom["edge_to_id"]
    action_array = geom["action_array"]
    action_to_index = geom["action_to_index"]
    ActionType = geom["ActionType"]

    n_hex, n_vertex, n_edge = 19, 54, 72
    n_act = len(action_array)
    fwd_hex = np.zeros((N_SYMMETRIES, n_hex), np.int64)
    fwd_vertex = np.zeros((N_SYMMETRIES, n_vertex), np.int64)
    fwd_edge = np.zeros((N_SYMMETRIES, n_edge), np.int64)
    pi_act = np.zeros((N_SYMMETRIES, n_act), np.int64)

    def apply_trip(trip, f):
        return frozenset(f(c) for c in trip)

    for g, f in enumerate(ops):
        for a in range(n_hex):
            fwd_hex[g, a] = coord2id[f(id2coord[a])]
        for v in range(n_vertex):
            fwd_vertex[g, v] = trip2node[apply_trip(node_triples[v], f)]
        for e_pair, eid in edge_to_id.items():
            u, w = e_pair
            new_pair = tuple(sorted((int(fwd_vertex[g, u]), int(fwd_vertex[g, w]))))
            fwd_edge[g, eid] = edge_to_id[new_pair]
        for i, action in enumerate(action_array):
            atype, value = action
            if atype in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
                new = (atype, int(fwd_vertex[g, int(value)]))
            elif atype == ActionType.BUILD_ROAD:
                u, w = value
                new = (atype, tuple(sorted((int(fwd_vertex[g, u]), int(fwd_vertex[g, w])))))
            elif atype == ActionType.MOVE_ROBBER:
                coord, victim = value
                new = (atype, (f(coord), victim))
            else:
                new = action
            pi_act[g, i] = action_to_index[new]

    inv_hex = np.argsort(fwd_hex, axis=1)
    inv_vertex = np.argsort(fwd_vertex, axis=1)
    inv_edge = np.argsort(fwd_edge, axis=1)

    canonical_hex_coord = np.zeros((n_hex, 3), np.float32)
    for a in range(n_hex):
        canonical_hex_coord[a] = np.asarray(id2coord[a], np.float32) / 4.0

    return HexSymmetry(
        fwd_hex=fwd_hex,
        inv_hex=inv_hex,
        fwd_vertex=fwd_vertex,
        inv_vertex=inv_vertex,
        fwd_edge=fwd_edge,
        inv_edge=inv_edge,
        pi_act=pi_act,
        canonical_hex_coord=canonical_hex_coord,
        op_names=op_names,
    )
