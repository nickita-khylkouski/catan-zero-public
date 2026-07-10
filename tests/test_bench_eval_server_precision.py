"""Precision-experiment telemetry regressions for bench_eval_server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import bench_eval_server  # type: ignore  # noqa: E402


def test_parity_metrics_report_policy_value_and_action_drift() -> None:
    metrics = bench_eval_server._parity_metrics(
        [({3: 0.8, 5: 0.2}, 0.25), ({7: 0.4, 9: 0.6}, -0.1)],
        [({3: 0.3, 5: 0.7}, 0.20), ({7: 0.45, 9: 0.55}, -0.1)],
        candidate_dtype="bf16",
        matmul_precision="highest",
    )

    assert metrics["reference_dtype"] == "fp32"
    assert metrics["candidate_dtype"] == "bf16"
    assert metrics["max_prior_absdiff"] == pytest.approx(0.5)
    assert metrics["mean_prior_absdiff"] == pytest.approx(0.275)
    assert metrics["max_policy_l1"] == pytest.approx(1.0)
    assert metrics["mean_policy_l1"] == pytest.approx(0.55)
    assert metrics["max_value_absdiff"] == pytest.approx(0.05)
    assert metrics["top_action_disagreements"] == 1
    assert metrics["top_action_disagreement_rate"] == pytest.approx(0.5)
    assert metrics["within_1e-5"] is False


def test_parity_metrics_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="counts differ"):
        bench_eval_server._parity_metrics(
            [],
            [({1: 1.0}, 0.0)],
            candidate_dtype="fp16",
            matmul_precision="highest",
        )
    with pytest.raises(ValueError, match="legal-action keys differ"):
        bench_eval_server._parity_metrics(
            [({1: 1.0}, 0.0)],
            [({2: 1.0}, 0.0)],
            candidate_dtype="fp16",
            matmul_precision="highest",
        )
