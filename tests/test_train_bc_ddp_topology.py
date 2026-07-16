from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import train_bc  # type: ignore  # noqa: E402


def _ddp(
    *, enabled: bool = True, world_size: int = 8, rank: int = 0
) -> dict[str, int | bool]:
    return {
        "enabled": enabled,
        "world_size": world_size,
        "rank": rank,
        "local_rank": rank,
    }


@pytest.mark.parametrize(
    "local_world_size",
    [None, "", "invalid", "0", "1", "4"],
)
def test_shared_storage_requires_explicit_matching_local_world_size(
    monkeypatch: pytest.MonkeyPatch,
    local_world_size: str | None,
) -> None:
    if local_world_size is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)

    assert not train_bc._single_node_ddp_shared_storage_enabled(_ddp())
    assert not train_bc._derived_array_cache_enabled(
        _ddp(), data_format="memmap"
    )


def test_shared_storage_accepts_explicit_single_node_ddp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")

    assert train_bc._single_node_ddp_shared_storage_enabled(_ddp())
    assert train_bc._single_node_ddp_preflight_enabled(_ddp())
    assert train_bc._derived_array_cache_enabled(
        _ddp(), data_format="memmap"
    )
    assert not train_bc._derived_array_cache_enabled(
        _ddp(), data_format="npz"
    )
    assert not train_bc._single_node_ddp_shared_storage_enabled(
        _ddp(enabled=False, world_size=1)
    )


def test_rank0_authority_runs_locally_when_topology_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    import torch.distributed as dist

    monkeypatch.setattr(
        dist,
        "broadcast_object_list",
        lambda *_args, **_kwargs: pytest.fail(
            "unknown topology must not use global rank 0 as local authority"
        ),
    )

    result = train_bc._rank0_authoritative_call(
        _ddp(rank=7),
        "per-node corpus audit",
        lambda: {"verified_by_rank": 7},
    )

    assert result == {"verified_by_rank": 7}


def test_rank0_authority_runs_locally_on_multinode_ddp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    import torch.distributed as dist

    monkeypatch.setattr(
        dist,
        "broadcast_object_list",
        lambda *_args, **_kwargs: pytest.fail(
            "global rank 0 cannot attest another node's local corpus"
        ),
    )

    result = train_bc._rank0_authoritative_call(
        _ddp(rank=5),
        "per-node corpus audit",
        lambda: {"verified_by_rank": 5},
    )

    assert result == {"verified_by_rank": 5}
