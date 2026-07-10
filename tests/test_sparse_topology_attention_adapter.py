from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.sparse_topology_adapter import (  # noqa: E402
    SparseTopologyAdapter,
    SparseTopologyAttentionAdapter,
    _scatter_destination_softmax,
    create_sparse_topology_adapter,
)


def _edges(*, device: torch.device | str = "cpu"):
    return (
        torch.tensor([[0, 1, 2, 0]], device=device),
        torch.tensor([[2, 2, 3, 0]], device=device),
        torch.tensor([[0, 1, 2, 0]], device=device),
        torch.tensor([[True, True, True, False]], device=device),
    )


def test_destination_softmax_is_sparse_normalized_and_masks_invalid_edges() -> None:
    logits = torch.tensor([[[1.0, -1.0], [2.0, 3.0], [7.0, -4.0], [100.0, 100.0]]])
    _, destination, _, valid = _edges()
    weights = _scatter_destination_softmax(logits, destination, valid, token_count=4)

    assert torch.allclose(weights[:, :2].sum(dim=1), torch.ones(1, 2))
    assert torch.allclose(weights[:, 2], torch.ones(1, 2))
    assert torch.equal(weights[:, 3], torch.zeros(1, 2))
    assert torch.isfinite(weights).all()


def test_v2_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(7)
    adapter = SparseTopologyAttentionAdapter(
        width=24, bottleneck=12, heads=3, dropout=0.0
    )
    x = torch.randn(1, 4, 24)

    result = adapter(x, edges=_edges())

    assert torch.equal(result, x)
    assert torch.count_nonzero(adapter.up.weight) == 0
    assert torch.count_nonzero(adapter.up.bias) == 0


def test_v2_delta_is_limited_to_live_destinations() -> None:
    adapter = SparseTopologyAttentionAdapter(
        width=8, bottleneck=8, heads=2, dropout=0.0
    )
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        adapter.up.bias.fill_(1.0)

    result = adapter(x, edges=_edges())

    assert torch.equal(result[:, :2], x[:, :2])
    assert torch.allclose(result[:, 2:], x[:, 2:] + 1.0)


def test_v2_padding_removes_source_and_destination_edges() -> None:
    adapter = SparseTopologyAttentionAdapter(
        width=8, bottleneck=8, heads=2, dropout=0.0
    )
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        adapter.up.bias.fill_(1.0)
    # Token 1 can no longer send to token 2, and token 3 cannot receive.  The
    # remaining 0 -> 2 edge keeps token 2 live.
    padding = torch.tensor([[False, True, False, True]])

    result = adapter(x, key_padding_mask=padding, edges=_edges())

    assert torch.equal(result[:, :2], x[:, :2])
    assert torch.allclose(result[:, 2], x[:, 2] + 1.0)
    assert torch.equal(result[:, 3], x[:, 3])


def test_v2_zero_init_allows_output_projection_to_learn() -> None:
    torch.manual_seed(11)
    adapter = SparseTopologyAttentionAdapter(
        width=16, bottleneck=8, heads=2, dropout=0.0
    )
    x = torch.randn(1, 4, 16, requires_grad=True)

    adapter(x, edges=_edges()).square().sum().backward()

    assert adapter.up.weight.grad is not None
    assert torch.count_nonzero(adapter.up.weight.grad) > 0
    assert torch.isfinite(adapter.up.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 regression")
def test_v2_bfloat16_autocast_scatter_dtype_matches() -> None:
    adapter = SparseTopologyAttentionAdapter(
        width=16, bottleneck=8, heads=2, dropout=0.0
    ).cuda()
    x = torch.randn(1, 4, 16, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = adapter(x, edges=_edges(device="cuda"))
        loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert adapter.up.weight.grad is not None
    assert torch.isfinite(adapter.up.weight.grad).all()


def test_factory_preserves_v1_parameter_schema_and_selects_v2() -> None:
    direct_v1 = SparseTopologyAdapter(16, 8, 2, 0.0)
    factory_v1 = create_sparse_topology_adapter(
        kind="v1", width=16, bottleneck=8, bases=2, heads=2, dropout=0.0
    )
    factory_v2 = create_sparse_topology_adapter(
        kind="local_attention_v2",
        width=16,
        bottleneck=8,
        bases=2,
        heads=2,
        dropout=0.0,
    )

    assert direct_v1.state_dict().keys() == factory_v1.state_dict().keys()
    assert hasattr(factory_v1, "basis_transforms")
    assert hasattr(factory_v2, "relation_key")

    canonical_v1 = create_sparse_topology_adapter(
        kind="basis_mean_v1",
        width=16,
        bottleneck=8,
        bases=2,
        heads=2,
        dropout=0.0,
    )
    assert canonical_v1.state_dict().keys() == direct_v1.state_dict().keys()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bottleneck": 0, "heads": 1}, "bottleneck"),
        ({"bottleneck": 8, "heads": 0}, "heads"),
        ({"bottleneck": 10, "heads": 3}, "divisible"),
    ],
)
def test_v2_rejects_invalid_dimensions(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SparseTopologyAttentionAdapter(width=16, dropout=0.0, **kwargs)


def test_factory_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown sparse topology adapter kind"):
        create_sparse_topology_adapter(
            kind="mystery",
            width=16,
            bottleneck=8,
            bases=2,
            heads=2,
            dropout=0.0,
        )
