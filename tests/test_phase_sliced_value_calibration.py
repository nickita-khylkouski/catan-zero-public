from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from phase_sliced_value_calibration import (  # type: ignore  # noqa: E402
    _calibration_stats,
    _legal_bucket,
    _slice_by,
)


def test_legal_bucket_boundaries():
    assert _legal_bucket(1) == "1"
    assert _legal_bucket(2) == "2-4"
    assert _legal_bucket(4) == "2-4"
    assert _legal_bucket(5) == "5-12"
    assert _legal_bucket(30) == "13-30"
    assert _legal_bucket(53) == "31-53"
    assert _legal_bucket(54) == "54"


def test_calibration_stats_perfectly_calibrated():
    # q equal to z -> corr 1, Brier 0.
    z = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    q = z.copy()
    stats = _calibration_stats(q, z, min_rows=1)
    assert stats["n"] == 4
    assert stats["corr_q_z"] == pytest.approx(1.0)
    assert stats["brier"] == pytest.approx(0.0)
    assert stats["value_rmse"] == pytest.approx(0.0)
    assert stats["e_q_given_win"] == pytest.approx(1.0)
    assert stats["e_q_given_loss"] == pytest.approx(-1.0)


def test_value_rmse_measures_residual_std():
    # q constant 0 vs z in {+1,-1} -> residual is +-1 -> RMSE 1.0.
    z = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    q = np.zeros(4, dtype=np.float32)
    assert _calibration_stats(q, z, min_rows=1)["value_rmse"] == pytest.approx(1.0)


def test_calibration_stats_single_class_reports_null_corr():
    # All wins -> corr undefined (guarded to None), but Brier still defined.
    z = np.ones(5, dtype=np.float32)
    q = np.full(5, 0.5, dtype=np.float32)
    stats = _calibration_stats(q, z, min_rows=1)
    assert stats["corr_q_z"] is None
    assert stats["n_loss"] == 0
    # outcome=1, p=(0.5+1)/2=0.75 -> Brier=(0.75-1)^2=0.0625.
    assert stats["brier"] == pytest.approx(0.0625)


def test_calibration_stats_respects_min_rows():
    z = np.array([1.0, -1.0], dtype=np.float32)
    q = np.array([0.9, -0.9], dtype=np.float32)
    assert _calibration_stats(q, z, min_rows=5)["corr_q_z"] is None
    assert _calibration_stats(q, z, min_rows=2)["corr_q_z"] is not None


def test_brier_clips_out_of_range_q():
    # q outside [-1,1] must be clipped so p stays a valid probability.
    z = np.array([1.0, -1.0], dtype=np.float32)
    q = np.array([5.0, -5.0], dtype=np.float32)  # -> p clipped to 1.0 / 0.0
    stats = _calibration_stats(q, z, min_rows=1)
    assert stats["brier"] == pytest.approx(0.0)


def test_slice_by_partitions_rows():
    q = np.array([0.9, 0.8, -0.9, -0.8], dtype=np.float32)
    z = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    keys = np.array(["opening_placement", "robber", "opening_placement", "robber"])
    sliced = _slice_by(q, z, keys, min_rows=1)
    assert set(sliced.keys()) == {"opening_placement", "robber"}
    assert sliced["opening_placement"]["n"] == 2
    assert sliced["robber"]["n"] == 2
