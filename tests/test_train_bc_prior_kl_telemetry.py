from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.train_bc import (
    _active_policy_teacher_gap_report,
    _active_policy_teacher_gap_telemetry,
    _prior_kl_telemetry,
)


def _base_data(**overrides):
    data = {
        "legal_action_ids": np.asarray(
            [
                [5, 6, 7, -1],  # row 0: 3 legal actions, 1 padded slot
                [5, 6, 7, -1],  # row 1: same shape
                [5, 6, 7, -1],  # row 2: no prior_policy recorded (teacher row)
            ],
            dtype=np.int16,
        ),
        # Search's improved (post-search) policy -- the training target.
        "target_policy": np.asarray(
            [
                [0.7, 0.2, 0.1, 0.0],
                [0.5, 0.3, 0.2, 0.0],
                [0.6, 0.3, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
        # Root prior (pre-search network policy).
        "prior_policy": np.asarray(
            [
                [0.5, 0.3, 0.2, 0.0],
                [0.4, 0.4, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0],  # never recorded for this (teacher) row
            ],
            dtype=np.float32,
        ),
    }
    data.update(overrides)
    return data


def _uniform_logits(n=3, width=4):
    # Uniform model distribution over the 3 legal slots (padded slot masked to -inf).
    logits = torch.zeros((n, width), dtype=torch.float32)
    logits[:, 3] = float("-inf")
    return logits


def test_returns_none_without_prior_policy_field():
    data = _base_data()
    del data["prior_policy"]
    batch = np.arange(3)

    result = _prior_kl_telemetry(data, batch, _uniform_logits(), torch.device("cpu"))

    assert result is None


def test_scopes_to_rows_with_a_recorded_prior_only():
    """FIX (success telemetry): rows with no recorded prior_policy (e.g. teacher/
    replay rows, which never populate this field) must be excluded -- this is
    what naturally scopes the computation to a held-out GEN slice without
    needing an explicit teacher_name check."""
    data = _base_data()
    batch = np.arange(3)

    result = _prior_kl_telemetry(data, batch, _uniform_logits(), torch.device("cpu"))

    assert result is not None
    assert result["has_prior"].tolist() == [True, True, False]


def test_kl_target_prior_matches_manual_computation():
    """KL(target_policy || prior_policy) is a pure data-derived quantity (no model
    involved) -- verify it matches a hand-computed KL divergence for row 0."""
    data = _base_data()
    batch = np.arange(3)

    result = _prior_kl_telemetry(data, batch, _uniform_logits(), torch.device("cpu"))

    # Row 0: target=[0.7,0.2,0.1], prior=[0.5,0.3,0.2] (already normalized, sum=1).
    expected = 0.7 * np.log(0.7 / 0.5) + 0.2 * np.log(0.2 / 0.3) + 0.1 * np.log(0.1 / 0.2)
    assert float(result["kl_target_prior"][0]) == pytest.approx(expected, abs=1e-4)


def test_kl_model_prior_is_zero_when_model_equals_prior():
    """If the model's own distribution exactly matches the recorded prior, KL(model
    || prior) must be ~0 -- sanity check for the direction/normalization of the
    computation."""
    data = _base_data()
    batch = np.arange(3)
    # Row 0 prior (already sums to 1): [0.5, 0.3, 0.2, 0.0].
    logits = torch.full((3, 4), float("-inf"))
    logits[0, 0] = float(np.log(0.5))
    logits[0, 1] = float(np.log(0.3))
    logits[0, 2] = float(np.log(0.2))

    result = _prior_kl_telemetry(data, batch, logits, torch.device("cpu"))

    assert float(result["kl_model_prior"][0]) == pytest.approx(0.0, abs=1e-3)
    assert float(result["kl_prior_model"][0]) == pytest.approx(0.0, abs=1e-3)


def test_unnormalized_prior_or_target_are_renormalized_before_kl():
    """A stored prior_policy/target_policy that doesn't sum to exactly 1 (float16
    rounding in real shards) must be renormalized over the legal slots before
    computing KL, not silently treated as an improper distribution."""
    data = _base_data(
        prior_policy=np.asarray(
            [
                [1.0, 0.6, 0.4, 0.0],  # sums to 2.0, not 1.0
                [0.4, 0.4, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    batch = np.arange(3)

    result = _prior_kl_telemetry(data, batch, _uniform_logits(), torch.device("cpu"))

    # Renormalized row-0 prior is [0.5, 0.3, 0.2] (halved) -- same expected KL as
    # the properly-normalized test above.
    expected = 0.7 * np.log(0.7 / 0.5) + 0.2 * np.log(0.2 / 0.3) + 0.1 * np.log(0.1 / 0.2)
    assert float(result["kl_target_prior"][0]) == pytest.approx(expected, abs=1e-4)


def _active_teacher_gap(
    data,
    logits,
    *,
    soft_targets=None,
    has_soft=None,
    policy_active=None,
):
    batch = np.arange(len(data["legal_action_ids"]))
    if soft_targets is None:
        soft_targets = torch.as_tensor(data["target_policy"], dtype=torch.float32)
    if has_soft is None:
        has_soft = torch.ones(len(batch), dtype=torch.bool)
    if policy_active is None:
        policy_active = torch.ones(len(batch), dtype=torch.bool)
    return _active_policy_teacher_gap_telemetry(
        data,
        batch,
        logits,
        torch.device("cpu"),
        soft_targets=soft_targets,
        has_soft=has_soft,
        policy_active=policy_active,
    )


def test_active_teacher_gap_excludes_zero_weight_single_action_and_missing_prior_rows():
    data = _base_data()
    # Row 1 is a one-action target even though has_soft is deliberately true;
    # the helper must independently enforce the multi-action contract. Row 2
    # has no recorded prior. Only row 0 is eligible.
    soft_targets = torch.as_tensor(data["target_policy"], dtype=torch.float32).clone()
    soft_targets[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    result = _active_teacher_gap(
        data,
        _uniform_logits(),
        soft_targets=soft_targets,
        has_soft=torch.tensor([True, True, True]),
        policy_active=torch.tensor([True, True, True]),
    )

    assert result is not None
    assert result["eligible"].tolist() == [True, False, False]

    # Positive target/prior rows are still ineligible when policy loss weight is zero.
    result = _active_teacher_gap(
        data,
        _uniform_logits(),
        policy_active=torch.tensor([False, True, True]),
    )
    assert result is not None
    assert result["eligible"].tolist() == [False, True, False]


def test_active_teacher_gap_is_fully_closed_when_model_matches_target():
    data = _base_data()
    logits = torch.full((3, 4), float("-inf"))
    logits[0, :3] = torch.log(torch.tensor([0.7, 0.2, 0.1]))
    logits[1, :3] = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    result = _active_teacher_gap(data, logits)

    assert result is not None
    eligible = result["eligible"]
    target_model = float(result["kl_target_model"][eligible].sum())
    target_prior = float(result["kl_target_prior"][eligible].sum())
    closure = 1.0 - target_model / target_prior
    assert target_model == pytest.approx(0.0, abs=1e-6)
    assert target_prior > 0.0
    assert closure == pytest.approx(1.0, abs=1e-6)


def test_active_teacher_gap_is_zero_closed_when_model_matches_prior():
    data = _base_data()
    logits = torch.full((3, 4), float("-inf"))
    logits[0, :3] = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    logits[1, :3] = torch.log(torch.tensor([0.4, 0.4, 0.2]))
    result = _active_teacher_gap(data, logits)

    assert result is not None
    eligible = result["eligible"]
    target_model = float(result["kl_target_model"][eligible].sum())
    target_prior = float(result["kl_target_prior"][eligible].sum())
    closure = 1.0 - target_model / target_prior
    assert target_model == pytest.approx(target_prior, abs=1e-6)
    assert closure == pytest.approx(0.0, abs=1e-6)


def test_active_teacher_gap_is_unavailable_without_soft_targets_or_prior():
    data = _base_data()
    batch = np.arange(3)
    assert (
        _active_policy_teacher_gap_telemetry(
            data,
            batch,
            _uniform_logits(),
            torch.device("cpu"),
            soft_targets=None,
            has_soft=torch.ones(3, dtype=torch.bool),
            policy_active=torch.ones(3, dtype=torch.bool),
        )
        is None
    )
    del data["prior_policy"]
    assert _active_teacher_gap(data, _uniform_logits()) is None


def test_active_teacher_gap_report_uses_additive_sums_and_handles_empty_input():
    report = _active_policy_teacher_gap_report(
        rows=4,
        kl_target_model_sum=0.5,
        kl_target_prior_sum=2.0,
    )
    assert report == {
        "active_policy_teacher_gap_rows": 4,
        "active_policy_kl_target_model_mean": pytest.approx(0.125),
        "active_policy_kl_target_prior_mean": pytest.approx(0.5),
        "active_policy_teacher_gap_closure": pytest.approx(0.75),
    }

    assert _active_policy_teacher_gap_report(
        rows=0,
        kl_target_model_sum=0.0,
        kl_target_prior_sum=0.0,
    ) == {
        "active_policy_teacher_gap_rows": 0,
        "active_policy_kl_target_model_mean": 0.0,
        "active_policy_kl_target_prior_mean": 0.0,
        "active_policy_teacher_gap_closure": 0.0,
    }
