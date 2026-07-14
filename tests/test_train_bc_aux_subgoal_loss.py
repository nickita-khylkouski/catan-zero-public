"""Unit tests for tools/train_bc.py::_aux_subgoal_loss (CAT-100 loss wiring).

The helper must combine only the aux heads that have BOTH a model output and a
corpus target field, mask -1 rows for categorical heads, be a pure no-op (0.0, 0)
when NO aux head is built, and (CAT-105) RAISE loud when heads ARE built but the
corpus carries none of the aux target columns (requested-but-inert footgun).
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import train_bc  # noqa: E402
from catan_zero.rl.aux_subgoal_targets import (  # noqa: E402
    AUX_SUBGOAL_TARGET_VERSION,
    AUX_SUBGOAL_TARGET_VERSION_KEY,
)


_N = 4
_DEVICE = torch.device("cpu")


def _versioned(data, *, versions=None, eligible=None):
    data[AUX_SUBGOAL_TARGET_VERSION_KEY] = np.asarray(
        [AUX_SUBGOAL_TARGET_VERSION] * _N if versions is None else versions,
        dtype=np.uint8,
    )
    if eligible is not None:
        data["_aux_subgoal_eligible"] = np.asarray(eligible, dtype=np.bool_)
    return data


def _outputs(all_heads=True):
    out = {
        "logits": torch.randn(_N, 5),
        "value": torch.randn(_N),
    }
    out["aux_longest_road"] = torch.randn(_N, requires_grad=True)
    out["aux_largest_army"] = torch.randn(_N, requires_grad=True)
    out["aux_vp_in_n"] = torch.randn(_N, requires_grad=True)
    out["aux_next_settlement"] = torch.randn(_N, 54, requires_grad=True)
    out["aux_robber_target"] = torch.randn(_N, 19, requires_grad=True)
    return out


def test_heads_built_but_corpus_lacks_aux_fields_raises_loud():
    """CAT-105: heads built (aux outputs present) + weight>0 (this fn is only
    called then) but the corpus carries NONE of the aux columns => the objective
    would silently train as a no-op. Must RAISE, not return (0.0, 0)."""
    out = _outputs()
    data = {}  # corpus without any aux target column
    with pytest.raises(ValueError, match="CAT-105"):
        train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)


def test_heads_absent_is_noop():
    """Heads-off model (no aux outputs): fn is a pure no-op, never raises."""
    out = {"logits": torch.randn(_N, 5), "value": torch.randn(_N)}
    data = {}
    loss, active = train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)
    assert active == 0
    assert float(loss.detach()) == 0.0


def test_unversioned_aux_targets_fail_closed():
    with pytest.raises(ValueError, match="unversioned targets"):
        train_bc._aux_subgoal_loss(
            _outputs(),
            {"aux_vp_in_n": np.zeros(_N, dtype=np.float32)},
            np.arange(_N),
            _DEVICE,
        )


def test_all_heads_present_counts_five_and_is_differentiable():
    out = _outputs()
    data = _versioned({
        "aux_longest_road": np.array([0, 1, 1, 0], dtype=np.float32),
        "aux_largest_army": np.array([0, 0, 1, 0], dtype=np.float32),
        "aux_vp_in_n": np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32),
        "aux_next_settlement": np.array([5, 12, -1, 40], dtype=np.int64),
        "aux_robber_target": np.array([3, -1, -1, 7], dtype=np.int64),
    })
    loss, active = train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)
    assert active == 5
    assert torch.isfinite(loss)
    loss.backward()
    # Categorical head with some valid rows receives gradient.
    assert out["aux_next_settlement"].grad is not None
    assert out["aux_next_settlement"].grad.abs().sum().item() > 0.0


def test_all_ignored_categorical_head_is_zero_but_enters_global_reduction(monkeypatch):
    out = _outputs()
    data = _versioned({
        "aux_next_settlement": np.array([-1, -1, -1, -1], dtype=np.int64),
    })
    calls = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def tracked_all_reduce(denominator, *, op):
        assert op == torch.distributed.ReduceOp.SUM
        calls.append(denominator.detach().clone())
        # Simulate one peer rank carrying two eligible labels.  This rank's
        # numerator remains zero, but it must enter the same collective.
        denominator.fill_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", tracked_all_reduce)
    loss, active = train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)
    # A rank-local empty mask still enters the DDP-global denominator helper;
    # another rank may carry valid labels for this same head.
    assert active == 0
    assert float(loss.detach()) == 0.0
    assert len(calls) == 1
    assert float(calls[0]) == 0.0


def test_partial_head_subset_counted():
    out = _outputs()
    data = _versioned({
        "aux_longest_road": np.array([1, 0, 1, 1], dtype=np.float32),
        "aux_vp_in_n": np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32),
    })
    loss, active = train_bc._aux_subgoal_loss(out, data, np.arange(_N), _DEVICE)
    assert active == 2
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0


def test_spatial_aux_labels_follow_the_exact_sampled_d6_orientation():
    """D6-augmented inputs must not retain absolute-coordinate aux labels."""
    class _FakeSymmetry:
        fwd_vertex = np.stack(
            [np.arange(54), np.roll(np.arange(54), -1)], axis=0
        )
        fwd_hex = np.stack(
            [np.arange(19), np.roll(np.arange(19), -1)], axis=0
        )

        @staticmethod
        def _remap_values(ids, table_g):
            rows = np.arange(ids.shape[0])
            return np.where(ids < 0, -1, table_g[rows, np.maximum(ids, 0)])

    symmetry = _FakeSymmetry()
    batch = np.arange(2)
    g = np.array([1, 1], dtype=np.int64)
    settlement = np.array([5, 31], dtype=np.int64)
    robber = np.array([3, 14], dtype=np.int64)
    mapped_settlement = symmetry.fwd_vertex[g, settlement]
    mapped_robber = symmetry.fwd_hex[g, robber]

    settlement_logits = torch.full((2, 54), -20.0, requires_grad=True)
    robber_logits = torch.full((2, 19), -20.0, requires_grad=True)
    with torch.no_grad():
        settlement_logits[torch.arange(2), torch.as_tensor(mapped_settlement)] = 20.0
        robber_logits[torch.arange(2), torch.as_tensor(mapped_robber)] = 20.0
    outputs = {
        "aux_next_settlement": settlement_logits,
        "aux_robber_target": robber_logits,
    }
    data = _versioned({
        "aux_next_settlement": settlement,
        "aux_robber_target": robber,
    })

    loss, active = train_bc._aux_subgoal_loss(
        outputs,
        data,
        batch,
        _DEVICE,
        symmetry=symmetry,
        symmetry_ids=g,
    )
    assert active == 2
    assert float(loss.detach()) < 1.0e-6


def test_historical_version_zero_is_ignored_only_outside_authenticated_scope():
    outputs = {"aux_vp_in_n": torch.zeros(_N, requires_grad=True)}
    data = _versioned(
        {"aux_vp_in_n": np.asarray([1.0, 2.0, 100.0, 100.0], dtype=np.float32)},
        versions=[1, 1, 0, 0],
        eligible=[True, True, False, False],
    )
    loss, active = train_bc._aux_subgoal_loss(
        outputs, data, np.arange(_N), _DEVICE
    )
    assert active == 1
    # Only version-1 eligible rows enter MSE: mean(1^2, 2^2) = 2.5.
    assert float(loss.detach()) == pytest.approx(2.5)

    data["_aux_subgoal_eligible"][:] = True
    with pytest.raises(ValueError, match="version mismatch"):
        train_bc._aux_subgoal_loss(outputs, data, np.arange(_N), _DEVICE)


class _CompositeAuxData(dict):
    component_ids = ("fresh", "historical_replay")
    aux_subgoal_scope_authenticated = True
    aux_subgoal_component_indices = (0,)

    @staticmethod
    def component_indices_for_rows(rows):
        return np.where(np.asarray(rows) < 2, 0, 1)


def _composite_aux_data():
    return _CompositeAuxData(
        {
            "aux_longest_road": np.asarray([0, 1, 0, 1], dtype=np.float32),
            "aux_largest_army": np.asarray([0, 0, 1, 0], dtype=np.float32),
            "aux_vp_in_n": np.asarray([0, 1, 2, 3], dtype=np.float32),
            "aux_next_settlement": np.asarray([5, 6, 7, 8], dtype=np.int16),
            "aux_robber_target": np.asarray([1, 2, 3, 4], dtype=np.int16),
            AUX_SUBGOAL_TARGET_VERSION_KEY: np.asarray(
                [AUX_SUBGOAL_TARGET_VERSION, AUX_SUBGOAL_TARGET_VERSION, 0, 0],
                dtype=np.uint8,
            ),
        }
    )


def test_aux_contract_keeps_unversioned_replay_policy_value_only():
    data = _composite_aux_data()
    report = train_bc._validate_aux_subgoal_training_contract(
        data, np.arange(_N), loss_weight=0.02
    )
    assert report["component_ids"] == ["fresh"]
    assert report["components"]["fresh"]["version_counts"] == {"1": 2}
    assert report["components"]["historical_replay"]["version_counts"] == {
        "0": 2
    }
    assert (
        report["components"]["historical_replay"]["aux_training_enabled"]
        is False
    )


def test_aux_contract_refuses_including_unversioned_replay():
    data = _composite_aux_data()
    data.aux_subgoal_component_indices = (0, 1)
    with pytest.raises(SystemExit, match="semantic version mismatch"):
        train_bc._validate_aux_subgoal_training_contract(
            data, np.arange(_N), loss_weight=0.02
        )


def test_zero_aux_weight_does_not_require_versioned_data():
    report = train_bc._validate_aux_subgoal_training_contract(
        {}, np.arange(_N), loss_weight=0.0
    )
    assert report == {
        "schema_version": "aux-subgoal-training-contract-v1",
        "enabled": False,
        "loss_weight": 0.0,
    }


class _GeometryPolicy:
    def __init__(self) -> None:
        self.model = torch.nn.Module()
        self.model.hex_encoder = torch.nn.Linear(2, 1, bias=False)


def _ddp_geometry_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    try:
        policy = _GeometryPolicy()
        policy.model = torch.nn.parallel.DistributedDataParallel(policy.model)
        weight = policy.model.module.hex_encoder.weight[0]
        # The two rank-local objectives have deliberately different directions.
        # The helper must reconstruct the global gradient by all-reducing the
        # autograd.grad results (DDP does not do that for us).
        if rank == 0:
            main = 2.0 * weight[0]
            aux = 2.0 * weight[0]
        else:
            main = 2.0 * weight[1]
            aux = -2.0 * weight[1]
        result = train_bc._aux_shared_trunk_gradient_geometry(
            policy,
            main_objective=main,
            unit_aux_objective=aux,
        )
        Path(out_dir, f"rank-{rank}.json").write_text(
            json.dumps(result, sort_keys=True), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


def test_aux_geometry_is_exact_no_update_on_authenticated_trunk_surface():
    policy = _GeometryPolicy()
    weight = policy.model.hex_encoder.weight[0]
    main = 3.0 * weight[0] + 4.0 * weight[1]
    aux = -4.0 * weight[0] + 3.0 * weight[1]

    result = train_bc._aux_shared_trunk_gradient_geometry(
        policy,
        main_objective=main,
        unit_aux_objective=aux,
    )

    assert result["schema_version"] == "a1-aux-global-gradient-geometry-batch-v1"
    assert result["updates_weights"] is False
    assert result["same_forward"] is True
    assert result["world_size"] == 1
    assert result["main_gradient_norm"] == pytest.approx(5.0)
    assert result["unit_aux_gradient_norm"] == pytest.approx(5.0)
    assert result["main_gradient_sq_sum"] == pytest.approx(25.0)
    assert result["unit_aux_gradient_sq_sum"] == pytest.approx(25.0)
    assert result["gradient_dot_product"] == pytest.approx(0.0)
    assert result["gradient_cosine"] == pytest.approx(0.0)
    assert result["opposing_coordinate_fraction"] == pytest.approx(0.5)
    assert result["parameter_surface"] == [
        {
            "name": "hex_encoder.weight",
            "shape": [1, 2],
            "dtype": "torch.float32",
        }
    ]
    assert result["parameter_surface_sha256"].startswith("sha256:")
    assert policy.model.hex_encoder.weight.grad is None


def test_aux_geometry_refuses_preexisting_gradients():
    policy = _GeometryPolicy()
    weight = policy.model.hex_encoder.weight[0]
    weight.sum().backward()

    with pytest.raises(RuntimeError, match="empty Parameter.grad"):
        train_bc._aux_shared_trunk_gradient_geometry(
            policy,
            main_objective=weight.sum(),
            unit_aux_objective=weight.square().sum(),
        )


def test_aux_geometry_manually_reconstructs_global_ddp_gradient(tmp_path: Path):
    import torch.multiprocessing as mp

    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _ddp_geometry_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    results = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert results[0] == results[1]
    result = results[0]
    assert result["world_size"] == 2
    assert result["aggregation"] == (
        "manual_all_reduce_then_world_average_of_ddp_scaled_gradients"
    )
    assert result["main_gradient_sq_sum"] == pytest.approx(2.0)
    assert result["unit_aux_gradient_sq_sum"] == pytest.approx(2.0)
    assert result["gradient_dot_product"] == pytest.approx(0.0)
    assert result["gradient_cosine"] == pytest.approx(0.0)


def test_five_batch_geometry_rng_transaction_restores_all_streams_exactly():
    numpy_rng = np.random.default_rng(424242)
    random.seed(37)
    torch.manual_seed(91)

    python_before = random.getstate()
    numpy_before = json.loads(json.dumps(numpy_rng.bit_generator.state))
    torch_before = torch.get_rng_state().clone()
    observed_batches = []
    with train_bc._isolated_aux_geometry_rng_transaction(
        numpy_generators={"probe_order": numpy_rng}
    ) as evidence:
        for batch_index in range(5):
            observed_batches.append(
                (
                    batch_index,
                    random.random(),
                    numpy_rng.integers(0, 2**31, size=8).tolist(),
                    torch.rand(8).tolist(),
                )
            )

    assert len({tuple(row[2]) for row in observed_batches}) == 5
    assert len({tuple(row[3]) for row in observed_batches}) == 5
    assert random.getstate() == python_before
    assert numpy_rng.bit_generator.state == numpy_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert evidence["scope"] == "one_complete_ordered_five_batch_probe"
    assert evidence["restore_frequency"] == "once_after_all_five_batches"
    assert evidence["restored_exactly"] is True
    assert (
        evidence["before"]["state_sha256"]
        != evidence["after_probe"]["state_sha256"]
    )
    assert (
        evidence["before"]["state_sha256"]
        == evidence["after_restore"]["state_sha256"]
    )


def test_geometry_rng_transaction_restores_after_probe_exception():
    numpy_rng = np.random.Generator(np.random.MT19937(123))
    random.seed(17)
    torch.manual_seed(29)
    python_before = random.getstate()
    numpy_before = copy.deepcopy(numpy_rng.bit_generator.state)
    torch_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="synthetic probe failure"):
        with train_bc._isolated_aux_geometry_rng_transaction(
            numpy_generators={"mt19937": numpy_rng}
        ):
            random.random()
            numpy_rng.random(16)
            torch.rand(16)
            raise ValueError("synthetic probe failure")

    assert random.getstate() == python_before
    assert np.array_equal(
        numpy_rng.bit_generator.state["state"]["key"],
        numpy_before["state"]["key"],
    )
    assert torch.equal(torch.get_rng_state(), torch_before)
