from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import build_value_sample_weights


def test_value_sample_weights_applies_phase_weights() -> None:
    """FIX A5: the value head previously had no way to be phase-weighted at all -- only the
    policy loss (build_sample_weights) consumed --phase-weights. A "robber=8.0" repair pass
    must actually raise the relative weight of robber-phase rows for the VALUE head too."""
    data = {
        "action_taken": np.asarray([1, 2, 3], dtype=np.int16),
        "phase": np.asarray(["robber", "initial_build", "robber"]),
    }

    weights = build_value_sample_weights(data, phase_weights={"robber": 8.0})

    # Row 1 (initial_build, untouched) must be weighted less than either robber row.
    assert weights[0] > weights[1]
    assert weights[2] > weights[1]
    # The two robber rows get the identical multiplier.
    assert weights[0] == pytest.approx(weights[2])
    # Still renormalized to mean 1 (same contract as build_sample_weights).
    assert float(np.mean(weights)) == pytest.approx(1.0, abs=1e-5)


def test_value_sample_weights_without_phase_weights_is_unchanged() -> None:
    """Backward compatibility: omitting phase_weights (the pre-fix call signature) must behave
    exactly as before."""
    data = {
        "action_taken": np.asarray([1, 2], dtype=np.int16),
        "phase": np.asarray(["robber", "initial_build"]),
        "value_weight_multiplier": np.asarray([1.0, 0.0], dtype=np.float32),
    }

    weights = build_value_sample_weights(data)

    assert weights.tolist() == pytest.approx([2.0, 0.0])


def test_value_sample_weights_combines_with_value_weight_multiplier() -> None:
    data = {
        "action_taken": np.asarray([1, 2], dtype=np.int16),
        "phase": np.asarray(["robber", "initial_build"]),
        "value_weight_multiplier": np.asarray([1.0, 1.0], dtype=np.float32),
    }

    weights = build_value_sample_weights(data, phase_weights={"robber": 4.0})

    assert weights[0] > weights[1]
    assert float(np.mean(weights)) == pytest.approx(1.0, abs=1e-5)
