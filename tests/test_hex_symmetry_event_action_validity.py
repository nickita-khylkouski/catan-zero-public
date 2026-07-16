from __future__ import annotations

import numpy as np

from catan_zero.rl.hex_symmetry import HexSymmetry, N_SYMMETRIES


def _synthetic_symmetry() -> HexSymmetry:
    def identities(width: int) -> np.ndarray:
        return np.tile(np.arange(width, dtype=np.int64), (N_SYMMETRIES, 1))

    # Make action zero visibly move under every non-identity orientation.  The
    # board permutations can remain identities because this regression targets
    # only the event action-id validity contract.
    pi_act = np.tile(np.arange(2, dtype=np.int64), (N_SYMMETRIES, 1))
    pi_act[1:] = np.array([1, 0], dtype=np.int64)
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


def _synthetic_entity() -> dict[str, np.ndarray]:
    return {
        "hex_tokens": np.zeros((1, 19, 13), dtype=np.float32),
        "vertex_tokens": np.zeros((1, 54, 8), dtype=np.float32),
        "edge_tokens": np.zeros((1, 72, 8), dtype=np.float32),
        "hex_vertex_ids": np.full((1, 19, 6), -1, dtype=np.int64),
        "hex_edge_ids": np.full((1, 19, 6), -1, dtype=np.int64),
        "edge_vertex_ids": np.full((1, 72, 2), -1, dtype=np.int64),
        "legal_action_tokens": np.zeros((1, 1, 8), dtype=np.float32),
        "event_tokens": np.zeros((1, 1, 41), dtype=np.float32),
        "event_mask": np.ones((1, 1), dtype=bool),
        "event_target_ids": np.full((1, 1, 4), -1, dtype=np.int64),
    }


def test_targetless_event_action_zero_stays_zero_for_every_symmetry() -> None:
    symmetry = _synthetic_symmetry()
    entity = _synthetic_entity()

    for orientation in range(N_SYMMETRIES):
        out = symmetry.permute_entity_batch(
            entity,
            orientation,
            relabel_events=True,
        )
        assert out["event_tokens"][0, 0, 35] == 0.0
        assert np.array_equal(
            out["event_target_ids"][0, 0],
            np.array([-1, -1, -1, -1], dtype=np.int64),
        )
