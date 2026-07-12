from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import sample_weight_quality


def test_sparse_policy_ess_separates_coverage_from_positive_weight_variance() -> None:
    # All-row ESS is 2/5 solely because three rows have no policy target.  Among
    # active policy rows the weights are uniform and the conditional ESS is 1.
    quality = sample_weight_quality(
        {"action_taken": np.arange(5)},
        np.asarray([0, 2, 0, 2, 0], dtype=np.float32),
    )
    assert quality["effective_sample_fraction"] == pytest.approx(0.4)
    assert quality["positive_sample_count"] == 2
    assert quality["positive_sample_fraction"] == pytest.approx(0.4)
    assert quality["positive_effective_sample_size"] == pytest.approx(2.0)
    assert quality["positive_effective_sample_fraction"] == pytest.approx(1.0)


def test_positive_policy_ess_exposes_variance_after_sparse_coverage() -> None:
    quality = sample_weight_quality(
        {"action_taken": np.arange(4)}, np.asarray([0, 1, 0, 3], dtype=np.float32)
    )
    assert quality["positive_sample_fraction"] == pytest.approx(0.5)
    assert quality["positive_effective_sample_fraction"] == pytest.approx(0.8)
