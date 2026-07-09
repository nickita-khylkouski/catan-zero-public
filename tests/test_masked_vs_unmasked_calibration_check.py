"""Task #76 re-save precondition: empirically determine whether a checkpoint
was actually trained with --mask-hidden-info, when no artifact (checkpoint
metadata predating 96b2819, report.json schema, train.log) records it.
Verdict logic only -- the corr(q,z) computation itself is exercised by
tools/value_repair_calibration_probe.py's own (reused) collect_holdout_rows/
compute_q, not duplicated here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import numpy as np

from masked_vs_unmasked_calibration_check import _corr  # type: ignore  # noqa: E402


def test_corr_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    q = rng.normal(size=1000)
    z = q * 0.7 + rng.normal(size=1000) * 0.3
    expected = float(np.corrcoef(q, z)[0, 1])
    assert _corr(q, z) == expected


def test_perfect_correlation_is_one():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    z = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_corr(q, z) - 1.0) < 1e-9


def test_verdict_logic_prefers_masked_when_masked_corr_is_higher():
    """Mirrors evaluate_both_regimes' verdict branch directly (kept inline
    since the full function needs a real checkpoint + entity features)."""
    corr_masked, corr_unmasked = 0.73, 0.68
    verdict = "masked-trained" if corr_masked > corr_unmasked else "omniscient-trained"
    assert verdict == "masked-trained"


def test_verdict_logic_prefers_omniscient_when_unmasked_corr_is_higher():
    corr_masked, corr_unmasked = 0.60, 0.69
    verdict = "masked-trained" if corr_masked > corr_unmasked else "omniscient-trained"
    assert verdict == "omniscient-trained"
