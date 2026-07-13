from __future__ import annotations

import torch

from tools import train_bc


def _reweight(policy_weights, values, targets):
    return train_bc._advantage_reweighted_policy_weights(  # noqa: SLF001
        policy_weights,
        {"value": values},
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        "outcome_value",
        1.0,
        10.0,
        0.0,
    )


def test_advantage_normalizer_uses_global_weighted_ddp_mean(monkeypatch) -> None:
    # Local rank: two rows with raw multipliers exp(+1), exp(-1).
    weights = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, -1.0])
    # Remote rank contributes one weight-2 row with raw multiplier exp(0)=1.
    remote = torch.tensor([2.0, 2.0])
    collectives = 0

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def all_reduce(value, op=None):
        nonlocal collectives
        collectives += 1
        assert op == torch.distributed.ReduceOp.SUM
        value.add_(remote)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    updated, _stats = _reweight(weights, values, targets)

    global_mean = (torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(-1.0)) + 2.0) / 4.0
    assert collectives == 1
    torch.testing.assert_close(
        updated,
        torch.tensor(
            [
                torch.exp(torch.tensor(1.0)) / global_mean,
                torch.exp(torch.tensor(-1.0)) / global_mean,
            ]
        ),
    )
    # A local normalizer would force the local weighted mean to one; the global
    # objective correctly does not impose that per-rank constraint.
    assert not torch.isclose(updated.mean(), torch.tensor(1.0))


def test_empty_local_rank_still_participates_in_advantage_collective(
    monkeypatch,
) -> None:
    weights = torch.zeros(3)
    values = torch.zeros(3)
    targets = torch.zeros(3)
    collectives = 0

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def all_reduce(value, op=None):
        nonlocal collectives
        collectives += 1
        value.add_(torch.tensor([2.0, 2.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    updated, stats = _reweight(weights, values, targets)

    assert collectives == 1
    assert torch.equal(updated, weights)
    assert stats["advantage_weight_rows"] == 0
