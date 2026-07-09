from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# `tools/perf_report.py` does a bare sibling import (`import perf_common`), so
# it only works with `tools/` itself on sys.path -- see test_perf_snapshot.py.
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import perf_common  # type: ignore  # noqa: E402
import perf_report  # type: ignore  # noqa: E402


def _leaf_row(mean_ms: float, timestamp: str, *, device: str = "cpu", checkpoint_label: str = "ckptA") -> dict:
    return {
        "kind": "leaf",
        "timestamp": timestamp,
        "device": device,
        "checkpoint_label": checkpoint_label,
        "stages": {
            "total": {"mean_ms": mean_ms, "p50_ms": mean_ms, "p95_ms": mean_ms, "total_ms": mean_ms, "n": 1.0}
        },
        "baseline_check": {"matches_baseline": True},
        "key": perf_common.stable_key("leaf", device, checkpoint_label, timestamp),
    }


def _generation_row(rows_per_hr: float, timestamp: str, *, hostname: str = "gpu0") -> dict:
    return {
        "kind": "generation",
        "timestamp": timestamp,
        "hostname": hostname,
        "rows_per_hr": rows_per_hr,
        "key": perf_common.stable_key("generation", hostname, timestamp),
    }


def test_build_report_flags_leaf_latency_regression() -> None:
    rows = [_leaf_row(3.5, "2026-07-01T00:00:00Z"), _leaf_row(6.0, "2026-07-02T00:00:00Z")]
    report = perf_report.build_report(rows, regression_threshold=0.25)
    assert len(report["findings"]) == 1
    assert report["findings"][0]["severity"] == "regression"


def test_build_report_no_regression_within_threshold() -> None:
    rows = [_leaf_row(3.5, "2026-07-01T00:00:00Z"), _leaf_row(3.6, "2026-07-02T00:00:00Z")]
    report = perf_report.build_report(rows, regression_threshold=0.25)
    assert report["findings"] == []


def test_build_report_flags_generation_throughput_drop() -> None:
    rows = [_generation_row(90000, "2026-07-01T00:00:00Z"), _generation_row(50000, "2026-07-02T00:00:00Z")]
    report = perf_report.build_report(rows, regression_threshold=0.25)
    assert len(report["findings"]) == 1
    assert "rows_per_hr" in report["findings"][0]["message"]


def test_build_report_no_regression_for_throughput_increase() -> None:
    rows = [_generation_row(50000, "2026-07-01T00:00:00Z"), _generation_row(90000, "2026-07-02T00:00:00Z")]
    report = perf_report.build_report(rows, regression_threshold=0.25)
    assert report["findings"] == []


def test_build_report_flags_gpu_context_thrash_regardless_of_history() -> None:
    row = {
        "kind": "gpu_util",
        "timestamp": "2026-07-01T00:00:00Z",
        "gpu_index": 0,
        "sm_util_pct": 92.0,
        "mem_util_pct": 1.0,
        "context_thrash_flagged": True,
        "key": "k1",
    }
    report = perf_report.build_report([row])
    assert any(f["severity"] == "anomaly" for f in report["findings"])


def test_build_report_does_not_flag_healthy_gpu() -> None:
    row = {
        "kind": "gpu_util",
        "timestamp": "2026-07-01T00:00:00Z",
        "gpu_index": 0,
        "sm_util_pct": 90.0,
        "mem_util_pct": 40.0,
        "context_thrash_flagged": False,
        "key": "k1",
    }
    report = perf_report.build_report([row])
    assert report["findings"] == []


def test_build_report_flags_leaf_baseline_mismatch() -> None:
    row = _leaf_row(3.5, "2026-07-01T00:00:00Z")
    row["baseline_check"] = {"matches_baseline": False}
    report = perf_report.build_report([row])
    assert any(f["severity"] == "anomaly" for f in report["findings"])


def test_build_report_groups_by_identity_not_just_kind() -> None:
    rows = [
        _leaf_row(3.5, "2026-07-01T00:00:00Z", checkpoint_label="ckptA"),
        _leaf_row(100.0, "2026-07-01T00:00:00Z", checkpoint_label="ckptB"),
    ]
    report = perf_report.build_report(rows)
    # Two distinct checkpoints, each with only one snapshot -- no history to
    # compare against, so no regression finding despite wildly different values.
    assert report["findings"] == []
    assert len(report["groups"]) == 2


def test_build_report_groups_leaf_rows_by_hostname_not_just_device() -> None:
    # CAT-71 review finding 3: leaf/gate/gpu_util rows used to group with no
    # hostname, so snapshots from different fleet hosts (which have
    # genuinely different per-leaf wall-clock floors) got compared against
    # each other as if they were the same time series, producing false
    # regressions. Same device/checkpoint, different host, wildly different
    # latency -- must NOT be flagged as a regression because they're
    # different groups (only one snapshot per group, no history to compare).
    b200_row = _leaf_row(3.5, "2026-07-01T00:00:00Z")
    b200_row["hostname"] = "b200"
    a100_row = _leaf_row(20.0, "2026-07-01T00:00:01Z")
    a100_row["hostname"] = "a100a"
    report = perf_report.build_report([b200_row, a100_row])
    assert report["findings"] == []
    assert len(report["groups"]) == 2


def test_build_report_still_flags_regression_within_same_hostname() -> None:
    rows = [
        _leaf_row(3.5, "2026-07-01T00:00:00Z"),
        _leaf_row(6.0, "2026-07-02T00:00:00Z"),
    ]
    for row in rows:
        row["hostname"] = "b200"
    report = perf_report.build_report(rows, regression_threshold=0.25)
    assert len(report["findings"]) == 1
    assert report["findings"][0]["severity"] == "regression"


def test_render_markdown_includes_findings_section() -> None:
    rows = [_leaf_row(3.5, "2026-07-01T00:00:00Z"), _leaf_row(10.0, "2026-07-02T00:00:00Z")]
    report = perf_report.build_report(rows)
    markdown = perf_report.render_markdown(report)
    assert "# CAT-71 Performance Snapshot Report" in markdown
    assert "REGRESSION" in markdown


def test_render_markdown_no_findings_is_explicit() -> None:
    report = perf_report.build_report([])
    markdown = perf_report.render_markdown(report)
    assert "No regressions or anomalies flagged." in markdown


def test_verify_known_anomalies_passes() -> None:
    # CAT-71's decision rule: the two historical ground-truth incidents
    # (GPU context-thrash pmon signature, SH-floor overruns) must both be
    # retroactively flagged.
    assert perf_report._verify_known_anomalies() is True


def test_cli_verify_known_anomalies_exit_code() -> None:
    result = subprocess.run(
        [sys.executable, str(_TOOLS_DIR / "perf_report.py"), "--verify-known-anomalies"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_cli_report_end_to_end_writes_markdown(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(_leaf_row(3.5, "2026-07-01T00:00:00Z")) + "\n" + json.dumps(_leaf_row(10.0, "2026-07-02T00:00:00Z")) + "\n",
        encoding="utf-8",
    )
    out_md = tmp_path / "report.md"
    result = subprocess.run(
        [sys.executable, str(_TOOLS_DIR / "perf_report.py"), "--ledger", str(ledger), "--out", str(out_md)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out_md.exists()
    assert "CAT-71" in out_md.read_text(encoding="utf-8")
    assert result.returncode == 0
