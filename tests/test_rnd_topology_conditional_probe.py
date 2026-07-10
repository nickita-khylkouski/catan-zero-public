from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.relational_trunks import REL_VERTEX_TO_HEX  # noqa: E402
from catan_zero.rl.sparse_topology_adapter import (  # noqa: E402
    create_sparse_topology_adapter,
)
from tools.rnd_topology_conditional_probe import (  # noqa: E402
    conditional_targets_from_inputs,
    construct_conditional_examples,
    run,
)


def _one_vertex_edges(batch_size: int):
    source = torch.tensor([[1, 2]]).expand(batch_size, -1).clone()
    destination = torch.tensor([[20, 20]]).expand(batch_size, -1).clone()
    relation = torch.full_like(source, REL_VERTEX_TO_HEX)
    valid = torch.ones_like(source, dtype=torch.bool)
    return source, destination, relation, valid


def test_conditional_target_is_receiver_query_dependent_soft_selection() -> None:
    x = torch.zeros(2, 151, 4)
    value = 3.0**0.5
    x[:, 1, 0] = value
    x[:, 2, 0] = -value
    x[:, 1, 1] = 1.0
    x[:, 2, 1] = -1.0
    x[0, 20, 1] = 2.0
    x[1, 20, 1] = -2.0

    target, live = conditional_targets_from_inputs(
        x, _one_vertex_edges(2), key_dims=1, temperature=20.0
    )

    assert live[:, 20].tolist() == [True, True]
    assert target[0, 20].item() == pytest.approx(value, abs=1e-5)
    assert target[1, 20].item() == pytest.approx(-value, abs=1e-5)
    assert torch.count_nonzero(target[:, :20]) == 0


def test_paired_examples_close_residual_leakage_path() -> None:
    edges = _one_vertex_edges(6)
    x, target, live = construct_conditional_examples(
        edges,
        batch_size=6,
        width=8,
        key_dims=2,
        temperature=4.0,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )

    assert torch.equal(x[:3, 1:20], x[3:, 1:20])
    assert torch.equal(x[:3, 20:74, 1:3], -x[3:, 20:74, 1:3])
    assert torch.count_nonzero(x[:, 20:74, 0]) == 0
    assert live[:, 20].all()
    assert (target[:3, 20] - target[3:, 20]).abs().mean().item() > 0.05


def test_basis_mean_prediction_is_invariant_to_paired_receiver_query() -> None:
    edges = _one_vertex_edges(6)
    x, _target, _live = construct_conditional_examples(
        edges,
        batch_size=6,
        width=8,
        key_dims=2,
        temperature=4.0,
        generator=torch.Generator().manual_seed(13),
        device=torch.device("cpu"),
    )
    adapter = create_sparse_topology_adapter(
        kind="basis_mean_v1",
        width=8,
        bottleneck=4,
        bases=2,
        heads=2,
        dropout=0.0,
    )
    with torch.no_grad():
        adapter.up.weight.normal_()
        adapter.up.bias.normal_()

    prediction = adapter(x, edges=edges)[:, 20, 0]

    assert torch.equal(prediction[:3], prediction[3:])


def test_local_attention_prediction_can_depend_on_paired_receiver_query() -> None:
    torch.manual_seed(17)
    edges = _one_vertex_edges(6)
    x, _target, _live = construct_conditional_examples(
        edges,
        batch_size=6,
        width=8,
        key_dims=2,
        temperature=0.25,
        generator=torch.Generator().manual_seed(19),
        device=torch.device("cpu"),
    )
    adapter = create_sparse_topology_adapter(
        kind="local_attention_v2",
        width=8,
        bottleneck=4,
        bases=2,
        heads=2,
        dropout=0.0,
    )
    with torch.no_grad():
        adapter.up.weight.normal_()
        adapter.up.bias.normal_()

    prediction = adapter(x, edges=edges)[:, 20, 0]

    assert (prediction[:3] - prediction[3:]).abs().max().item() > 1e-5


def test_conditional_probe_small_cpu_smoke() -> None:
    report = run(
        argparse.Namespace(
            kind="local_attention_v2",
            edge_control="true_topology",
            device="cpu",
            width=8,
            bottleneck=4,
            heads=2,
            key_dims=2,
            temperature=0.25,
            batch_size=4,
            steps=2,
            lr=3e-3,
            seed=11,
            output="",
        )
    )

    assert report["schema_version"] == "catan-zero-topology-conditional-probe/v1"
    assert report["finite"] is True
    assert report["device"] == "cpu"
    assert report["tail_mean_target_pair_gap"] > 0
    assert report["tail_mean_zero_predictor_loss"] > 0
    assert report["source_sha256"]
