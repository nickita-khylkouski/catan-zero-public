from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _subprocess_env() -> dict[str, str]:
    """subprocess.run does not inherit pytest's `pythonpath = ["src"]` ini
    setting, so a subprocess-level CLI test must set PYTHONPATH explicitly
    or the `catan_zero` import inside factory_common.py fails."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    parts = [str(_SRC_DIR)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env

from noise_vs_spread_trend import (  # type: ignore  # noqa: E402
    _pearson,
    _slope,
    build_trend_report,
    extract_generation_metrics,
)


def _opening_panel_json(mean_raw_q_spread: float, mean_spread_over_floor: float) -> dict:
    """Shape mirrors tools/opening_panel.py's `eval` --out JSON (see its
    `main()`/`aggregate()`): top-level "aggregate" key with these fields
    among others."""
    return {
        "checkpoint": "ckpt.pt",
        "panel": "runs/panels/opening_200.json",
        "n_roots_evaluated": 200,
        "aggregate": {
            "n_roots": 200,
            "flip_rate": 0.1,
            "mean_raw_q_spread": mean_raw_q_spread,
            "mean_spread_over_floor": mean_spread_over_floor,
            "mean_kendall_tau": 0.5,
            "mean_top1_regret": 0.05,
            "mean_top3_coverage": 0.9,
        },
        "per_root": [],
    }


def _f74_json(q_orientation_std_mean: float) -> dict:
    """Shape mirrors tools/f74_symmetry_eval.py's --out JSON (see its
    `main()`: `write_json(args.out, {"summary": summary, "per_root": per_root})`)."""
    return {
        "summary": {
            "checkpoint": "ckpt.pt",
            "n_roots": 50,
            "n_symmetries": 12,
            "symmetry_inconsistency": {
                "value_orientation_std": {"mean": 0.02, "median": 0.02, "p90": 0.03, "max": 0.05},
                "value_orientation_range": {"mean": 0.05, "median": 0.05, "p90": 0.08, "max": 0.1},
                "prior_candidate_orientation_std": {"mean": 0.01, "median": 0.01, "p90": 0.02, "max": 0.03},
                "q_candidate_orientation_std": {
                    "mean": q_orientation_std_mean,
                    "median": q_orientation_std_mean,
                    "p90": q_orientation_std_mean * 1.2,
                    "max": q_orientation_std_mean * 1.5,
                },
            },
            "noise_reduction": {},
        },
        "per_root": [],
    }


def test_extract_generation_metrics_pulls_expected_fields():
    op = _opening_panel_json(mean_raw_q_spread=0.3, mean_spread_over_floor=1.5)
    f74 = _f74_json(q_orientation_std_mean=0.05)
    metrics = extract_generation_metrics(op, f74)
    assert metrics["top5_q_spread_proxy"] == pytest.approx(0.3)
    assert metrics["top5_q_spread_over_floor_proxy"] == pytest.approx(1.5)
    assert metrics["orientation_noise_std"] == pytest.approx(0.05)


def test_slope_perfectly_increasing_series():
    assert _slope([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_slope_constant_series_is_zero():
    assert _slope([2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_slope_insufficient_points_is_none():
    assert _slope([1.0]) is None
    assert _slope([]) is None


def test_pearson_perfectly_correlated_series_is_near_one():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert _pearson(a, b) == pytest.approx(1.0, abs=1e-9)


def test_pearson_perfectly_anticorrelated_series_is_near_negative_one():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert _pearson(a, b) == pytest.approx(-1.0, abs=1e-9)


def test_pearson_handles_none_entries_by_dropping_pairs():
    a = [1.0, None, 3.0, 4.0]
    b = [2.0, 5.0, 6.0, 8.0]
    result = _pearson(a, b)
    assert result is not None
    assert -1.0 <= result <= 1.0


def test_pearson_insufficient_pairs_is_none():
    assert _pearson([1.0], [2.0]) is None
    assert _pearson([1.0, None], [None, 2.0]) is None


def test_build_trend_report_per_generation_and_series_shape():
    generations = {
        "v3a": {
            "opening_panel_json": _opening_panel_json(0.2, 1.0),
            "f74_json": _f74_json(0.04),
        },
        "gen-1": {
            "opening_panel_json": _opening_panel_json(0.3, 1.3),
            "f74_json": _f74_json(0.05),
        },
        "gen-2": {
            "opening_panel_json": _opening_panel_json(0.4, 1.6),
            "f74_json": _f74_json(0.06),
        },
    }
    report = build_trend_report(generations)
    assert report["measurement"] == "noise_vs_spread_trend"
    assert report["generations_ordered"] == ["v3a", "gen-1", "gen-2"]
    assert report["series"]["top5_q_spread_proxy"] == pytest.approx([0.2, 0.3, 0.4])
    assert report["series"]["orientation_noise_std"] == pytest.approx([0.04, 0.05, 0.06])
    assert report["trend"]["top5_q_spread_proxy_slope"] == pytest.approx(0.1)
    # Both series increase monotonically and proportionally -> near-perfect correlation.
    corr = report["pearson_correlation"]["top5_q_spread_proxy_vs_orientation_noise_std"]
    assert corr == pytest.approx(1.0, abs=1e-6)
    assert "v3a" in report["per_generation"]
    assert report["per_generation"]["gen-2"]["top5_q_spread_proxy"] == pytest.approx(0.4)


def test_cli_help_does_not_crash():
    script = _TOOLS_DIR / "noise_vs_spread_trend.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=_subprocess_env(),
    )
    assert result.returncode == 0


def test_cli_end_to_end_with_config_file(tmp_path):
    op_v3a = tmp_path / "op_v3a.json"
    f74_v3a = tmp_path / "f74_v3a.json"
    op_gen1 = tmp_path / "op_gen1.json"
    f74_gen1 = tmp_path / "f74_gen1.json"
    op_v3a.write_text(json.dumps(_opening_panel_json(0.2, 1.0)))
    f74_v3a.write_text(json.dumps(_f74_json(0.04)))
    op_gen1.write_text(json.dumps(_opening_panel_json(0.35, 1.4)))
    f74_gen1.write_text(json.dumps(_f74_json(0.055)))

    config = {
        "v3a": {"opening_panel_json": str(op_v3a), "f74_json": str(f74_v3a)},
        "gen-1": {"opening_panel_json": str(op_gen1), "f74_json": str(f74_gen1)},
    }
    config_path = tmp_path / "generations.json"
    config_path.write_text(json.dumps(config))
    out_path = tmp_path / "trend_out.json"

    from noise_vs_spread_trend import main  # local import to patch sys.argv cleanly

    old_argv = sys.argv
    sys.argv = ["noise_vs_spread_trend.py", "--config", str(config_path), "--out", str(out_path)]
    try:
        main()
    finally:
        sys.argv = old_argv

    data = json.loads(out_path.read_text())
    assert data["generations_ordered"] == ["v3a", "gen-1"]
    assert data["measurement"] == "noise_vs_spread_trend"
