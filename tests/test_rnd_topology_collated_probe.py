from __future__ import annotations

import hashlib

import numpy as np

from tools.rnd_topology_collated_probe import (
    _pad_rows,
    build_collated_public_batch,
)


def test_pad_rows_uses_safe_fill_and_requested_side() -> None:
    source = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
    assert _pad_rows(source, 2).tolist() == [[1, 2], [3, 4]]
    assert _pad_rows(source, 2, take_last=True).tolist() == [[3, 4], [5, 6]]
    assert _pad_rows(source[:1], 2, fill=-1).tolist() == [[1, 2], [-1, -1]]


def test_real_public_state_batch_is_aligned_padded_and_deterministic() -> None:
    kwargs = {"batch_size": 5, "legal_actions": 7, "events": 11, "seed": 1234}
    batch, provenance = build_collated_public_batch(**kwargs)
    repeated, repeated_provenance = build_collated_public_batch(**kwargs)

    assert batch["legal_action_tokens"].shape == (5, 7, 50)
    assert batch["legal_action_context"].shape == (5, 7, 18)
    assert batch["legal_action_target_ids"].shape == (5, 7, 4)
    assert batch["legal_action_ids"].shape == (5, 7)
    assert batch["event_tokens"].shape[:2] == (5, 11)
    assert batch["event_target_ids"].shape == (5, 11, 4)
    assert batch["event_mask"].shape == (5, 11)
    assert batch["hex_vertex_ids"].shape == (5, 19, 6)
    assert batch["hex_edge_ids"].shape == (5, 19, 6)
    assert batch["edge_vertex_ids"].shape == (5, 72, 2)

    live = batch["legal_action_mask"]
    assert live.any(dim=1).all()
    assert (batch["legal_action_ids"][~live] == -1).all()
    assert (batch["legal_action_target_ids"][~live] == -1).all()
    assert (batch["legal_action_tokens"][~live] == 0).all()
    assert (batch["legal_action_context"][~live] == 0).all()
    assert provenance["mask_utilization"]["legal_actions"]["live"] == int(
        live.sum()
    )
    assert provenance["generator"].startswith("ColonistMultiAgentEnv")
    assert all(
        record["legal_action_ids_sha256"]
        == hashlib.sha256(
            batch["legal_action_ids"][row][live[row]].numpy().astype(np.int64).tobytes()
        ).hexdigest()
        for row, record in enumerate(provenance["state_records"])
    )
    assert provenance["legal_action_ids_sha256"] == repeated_provenance[
        "legal_action_ids_sha256"
    ]
    assert batch.keys() == repeated.keys()
    for key in batch:
        assert np.array_equal(batch[key].numpy(), repeated[key].numpy())
