from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import a1_coherent_rd_materialize as materialize


def _v2_shard(path: Path) -> None:
    np.savez_compressed(
        path,
        game_seed=np.asarray([901, 901], dtype=np.uint64),
        decision_index=np.asarray([0, 1], dtype=np.int32),
        seat=np.asarray([0, 1], dtype=np.int8),
        terminated=np.asarray([True, True]),
        truncated=np.asarray([False, False]),
        policy_weight_multiplier=np.asarray([1.0, 0.0], dtype=np.float32),
        target_information_regime=np.asarray(["coherent", "coherent"]),
        legal_action_mask=np.asarray([[True, True], [True, False]]),
        simulations_used=np.asarray([8, 0], dtype=np.uint16),
        search_evidence_version=np.asarray(2, dtype=np.uint8),
        search_evidence_offsets=np.asarray([0, 2], dtype=np.uint32),
        search_visit_counts_flat=np.asarray([5, 3], dtype=np.uint16),
        search_completed_q_flat=np.asarray([0.1, -0.2], dtype=np.float32),
        search_prior_policy_flat=np.asarray([0.7, 0.3], dtype=np.float32),
    )


def test_materializer_accepts_v2_fp32_prior_shard(tmp_path: Path) -> None:
    shard = tmp_path / "shard.npz"
    _v2_shard(shard)
    trace = {
        "seen": set(),
        "current_seed": None,
        "last_decision": None,
        "current_complete": False,
        "current_seats": set(),
    }

    assert materialize._verify_shard(  # noqa: SLF001
        shard, regime="coherent", trace=trace
    ) == {"rows": 2, "policy_active_rows": 1}


@pytest.mark.parametrize(
    "filename",
    (
        "../gumbel_self_play_shard_00000.npz",
        "/tmp/gumbel_self_play_shard_00000.npz",
        "gumbel_self_play_shard_00001.npz",
        "renamed.npz",
    ),
)
def test_confirmed_shard_path_rejects_escape_and_noncanonical_names(
    tmp_path: Path,
    filename: str,
) -> None:
    worker = tmp_path / "worker_000"
    worker.mkdir()

    with pytest.raises(
        materialize.MaterializationError,
        match="filename is not canonical",
    ):
        materialize._confirmed_worker_shard_path(
            worker,
            filename=filename,
            index=0,
        )


def test_confirmed_shard_path_rejects_symlink_alias(tmp_path: Path) -> None:
    worker = tmp_path / "worker_000"
    worker.mkdir()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"not a shard")
    alias = worker / "gumbel_self_play_shard_00000.npz"
    alias.symlink_to(outside)

    with pytest.raises(
        materialize.MaterializationError,
        match="escapes or aliases",
    ):
        materialize._confirmed_worker_shard_path(
            worker,
            filename=alias.name,
            index=0,
        )


def test_confirmed_shard_path_accepts_canonical_regular_file(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker_000"
    worker.mkdir()
    shard = worker / "gumbel_self_play_shard_00000.npz"
    shard.write_bytes(b"regular shard placeholder")

    assert (
        materialize._confirmed_worker_shard_path(
            worker,
            filename=shard.name,
            index=0,
        )
        == shard
    )
