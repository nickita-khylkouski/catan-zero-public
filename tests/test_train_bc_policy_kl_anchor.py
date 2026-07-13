"""Tests for the differentiable policy-KL anchor loss (contingency f67, D1).

The recovery-default anchor is forward KL(prior_policy || model); historical
reverse KL remains an explicit ablation. Both exclude forced rows. It must be
differentiable through the logits, and (c) be a clean None when no prior rows
exist so a caller adds nothing to the loss.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest
import torch

from tools.train_bc import (
    _policy_kl_anchor_loss,
    _policy_kl_anchor_loss_parts,
    _prior_kl_telemetry,
    _q_score_loss_parts,
)


def _base_data(**overrides):
    data = {
        "legal_action_ids": np.asarray(
            [
                [5, 6, 7, -1],  # row 0: 3 legal, 1 padded
                [5, 6, 7, -1],  # row 1: has a prior
                [5, 6, 7, -1],  # row 2: teacher row, no prior recorded
            ],
            dtype=np.int16,
        ),
        "target_policy": np.asarray(
            [
                [0.7, 0.2, 0.1, 0.0],
                [0.5, 0.3, 0.2, 0.0],
                [0.6, 0.3, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
        "prior_policy": np.asarray(
            [
                [0.5, 0.3, 0.2, 0.0],
                [0.4, 0.4, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0],  # no recorded prior
            ],
            dtype=np.float32,
        ),
    }
    data.update(overrides)
    return data


def _logits(n=3, width=4, requires_grad=False):
    logits = torch.zeros((n, width), dtype=torch.float32)
    logits[:, 3] = float("-inf")  # padded slot masked, as the model emits
    if requires_grad:
        # -inf is not differentiable; build a grad-carrying tensor for legal slots
        base = torch.zeros((n, width), dtype=torch.float32, requires_grad=True)
        masked = base + torch.tensor([[0.0, 0.0, 0.0, float("-inf")]])
        return base, masked
    return logits


def _sparse_ddp_anchor_worker(rank: int, rendezvous: str, output) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        data = _base_data()
        row = np.asarray([2 if rank == 0 else 0], dtype=np.int64)
        base, masked = _logits(n=1, requires_grad=True)
        parts = _policy_kl_anchor_loss_parts(
            data, row, masked, torch.device("cpu")
        )
        if parts is None:
            raise AssertionError("global DDP anchor unexpectedly has no terms")
        loss, _weighted_sum, denominator = parts
        loss.backward()
        q_values = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        q_data = {
            "action_taken": np.asarray([5], dtype=np.int16),
            "legal_action_ids": np.asarray([[5, 6, 7]], dtype=np.int16),
            "target_scores": np.asarray(
                [[np.nan, np.nan, np.nan] if rank == 0 else [0.0, 1.0, 2.0]],
                dtype=np.float32,
            ),
            "teacher_name": np.asarray(["value_rollout_search"]),
            "target_score_source": np.asarray(["value_rollout_search"]),
        }
        q_loss, _q_sum, q_denominator = _q_score_loss_parts(
            q_values,
            q_data,
            np.asarray([0], dtype=np.int64),
            torch.ones(1, dtype=torch.float32),
            torch.device("cpu"),
            q_skip_teacher_prefixes=(),
        )
        q_loss.backward()
        output.put(
            {
                "rank": rank,
                "denominator": float(denominator.item()),
                "gradient_l1": float(base.grad.abs().sum().item()),
                "q_denominator": float(q_denominator.item()),
                "q_gradient_l1": float(q_values.grad.abs().sum().item()),
            }
        )
    except BaseException as error:  # pragma: no cover - surfaced in parent
        output.put({"rank": rank, "error": repr(error)})
    finally:
        torch.distributed.destroy_process_group()


def test_returns_none_without_prior_rows():
    data = _base_data()
    del data["prior_policy"]
    assert _policy_kl_anchor_loss(data, np.arange(3), _logits(), torch.device("cpu")) is None


def test_returns_none_when_all_rows_lack_a_prior():
    # Only the teacher row (no prior) is in the batch.
    data = _base_data()
    result = _policy_kl_anchor_loss(data, np.asarray([2]), _logits(n=1), torch.device("cpu"))
    assert result is None


def test_default_matches_forward_kl_over_prior_rows():
    data = _base_data()
    batch = np.arange(3)
    logits = _logits()
    anchor = _policy_kl_anchor_loss(data, batch, logits, torch.device("cpu"))
    terms = _prior_kl_telemetry(data, batch, logits, torch.device("cpu"))

    has_prior = terms["has_prior"]
    expected = terms["kl_prior_model"][has_prior].mean()
    assert anchor is not None
    assert torch.allclose(anchor, expected, atol=1e-6)
    # Only rows 0 and 1 carry a prior; row 2 (teacher) is excluded.
    assert int(has_prior.sum().item()) == 2


def test_legacy_reverse_direction_is_explicit_and_matches_telemetry():
    data = _base_data()
    batch = np.arange(3)
    logits = _logits()
    anchor = _policy_kl_anchor_loss(
        data, batch, logits, torch.device("cpu"), direction="reverse"
    )
    terms = _prior_kl_telemetry(data, batch, logits, torch.device("cpu"))
    assert anchor is not None
    assert torch.allclose(
        anchor, terms["kl_model_prior"][terms["has_prior"]].mean(), atol=1e-6
    )


def test_is_differentiable_through_logits():
    data = _base_data()
    batch = np.arange(3)
    base, masked = _logits(requires_grad=True)
    anchor = _policy_kl_anchor_loss(data, batch, masked, torch.device("cpu"))
    assert anchor is not None
    anchor.backward()
    assert base.grad is not None
    assert torch.isfinite(base.grad).all()
    # Gradient must be nonzero somewhere (the anchor actually pulls the policy).
    assert base.grad.abs().sum().item() > 0.0


def test_sparse_ddp_objective_batches_use_one_collective_path(tmp_path):
    """Locally empty anchor/Q rows must not skip peers' DDP collectives."""

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    context = mp.get_context("spawn")
    output = context.Queue()
    rendezvous = str(tmp_path / "policy-kl-anchor-rendezvous")
    processes = [
        context.Process(
            target=_sparse_ddp_anchor_worker,
            args=(rank, rendezvous, output),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
        process.join(timeout=5)
    assert not hung, "rank-local sparse-objective early return deadlocked DDP"
    assert [process.exitcode for process in processes] == [0, 0]

    observed = sorted((output.get(timeout=5) for _ in range(2)), key=lambda row: row["rank"])
    assert all("error" not in row for row in observed)
    assert [row["denominator"] for row in observed] == [0.0, 1.0]
    assert observed[0]["gradient_l1"] == pytest.approx(0.0)
    assert observed[1]["gradient_l1"] > 0.0
    assert [row["q_denominator"] for row in observed] == [0.0, 1.0]
    assert observed[0]["q_gradient_l1"] == pytest.approx(0.0)
    assert observed[1]["q_gradient_l1"] > 0.0


def test_is_near_zero_when_model_equals_prior():
    # Make the model distribution equal the row-0/row-1 prior so KL ~ 0.
    data = _base_data()
    batch = np.asarray([0])
    prior = data["prior_policy"][0, :3]
    logits = torch.tensor([[float(np.log(p)) for p in prior] + [float("-inf")]], dtype=torch.float32)
    anchor = _policy_kl_anchor_loss(data, batch, logits, torch.device("cpu"))
    assert anchor is not None
    assert abs(float(anchor.item())) < 1e-5


def test_forced_prior_rows_do_not_dilute_anchor_denominator():
    data = _base_data()
    # Row 0 is forced and therefore has identically-zero KL. Row 1 is the sole
    # meaningful anchor row. The result must equal row 1, not half of row 1.
    data["legal_action_ids"][0] = np.asarray([5, -1, -1, -1], dtype=np.int16)
    data["prior_policy"][0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    logits = _logits()
    anchor = _policy_kl_anchor_loss(data, np.arange(3), logits, torch.device("cpu"))
    row_one = _policy_kl_anchor_loss(
        data, np.asarray([1]), logits[1:2], torch.device("cpu")
    )
    assert anchor is not None and row_one is not None
    assert torch.allclose(anchor, row_one, atol=1e-6)


def test_anchor_parts_expose_the_same_conditional_denominator_used_by_the_loss():
    data = _base_data()
    data["legal_action_ids"][0] = np.asarray([5, -1, -1, -1], dtype=np.int16)
    data["prior_policy"][0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    parts = _policy_kl_anchor_loss_parts(
        data, np.arange(3), _logits(), torch.device("cpu")
    )
    assert parts is not None
    loss, weighted_sum, denominator = parts
    assert denominator.item() == 1.0
    assert torch.allclose(loss, weighted_sum / denominator, atol=1e-6)


def test_multi_action_zero_policy_weight_row_remains_anchor_eligible():
    data = _base_data(policy_weight_multiplier=np.zeros(3, dtype=np.float32))
    anchor = _policy_kl_anchor_loss(
        data, np.asarray([0]), _logits(n=1), torch.device("cpu")
    )
    assert anchor is not None
    assert float(anchor.item()) > 0.0


def test_composite_anchor_scope_excludes_current_producer_priors():
    class ScopedData(dict):
        policy_kl_anchor_component_indices = (1,)
        policy_kl_anchor_scope_authenticated = True

        @staticmethod
        def component_indices_for_rows(rows):
            # Row 0 is the current/regressed producer; row 1 is gen3 replay.
            return np.asarray([0 if int(row) == 0 else 1 for row in rows])

    data = ScopedData(_base_data())
    logits = _logits()
    scoped = _policy_kl_anchor_loss(
        data, np.asarray([0, 1]), logits[:2], torch.device("cpu")
    )
    gen3_only = _policy_kl_anchor_loss(
        data, np.asarray([1]), logits[1:2], torch.device("cpu")
    )
    assert scoped is not None and gen3_only is not None
    assert torch.allclose(scoped, gen3_only, atol=1e-6)


def test_materialized_anchor_scope_excludes_ineligible_components():
    data = _base_data()
    data["_policy_kl_anchor_eligible"] = np.asarray(
        [False, True, False], dtype=np.bool_
    )
    logits = _logits()

    scoped = _policy_kl_anchor_loss(
        data, np.asarray([0, 1, 2]), logits, torch.device("cpu")
    )
    eligible_only = _policy_kl_anchor_loss(
        _base_data(), np.asarray([1]), logits[1:2], torch.device("cpu")
    )

    assert scoped is not None and eligible_only is not None
    assert torch.allclose(scoped, eligible_only, atol=1e-6)


def test_unknown_anchor_direction_is_rejected():
    with pytest.raises(ValueError, match="unknown policy KL anchor direction"):
        _policy_kl_anchor_loss(
            _base_data(),
            np.asarray([0]),
            _logits(n=1),
            torch.device("cpu"),
            direction="sideways",
        )
