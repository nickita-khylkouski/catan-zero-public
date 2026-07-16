from __future__ import annotations

from pathlib import Path

import pytest

from tools import a1_coherent_rd_materialize as materialize


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
