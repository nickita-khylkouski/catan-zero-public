#!/usr/bin/env python3
"""CAT-71: read the perf ledger (tools/perf_snapshot.py's output) and render a
Markdown report comparing snapshots over time -- the regression-catcher.

For each (kind, identity) group -- e.g. (leaf, device=cpu, checkpoint=X),
(generation, host=gpu0), (gate, gate_name=gen2_vs_gen1), (gpu_util, gpu_index=0)
-- compares the latest snapshot against the previous one and flags a
regression when the primary metric moves against the project by more than
`--regression-threshold` (default 25%). gpu_util rows are also checked
directly against the context-thrash anomaly signature regardless of history.

Usage:
    tools/perf_report.py --ledger runs/perf/perf_ledger.jsonl
    tools/perf_report.py --ledger runs/perf/perf_ledger.jsonl --out runs/perf/latest_report.md

Verification mode (CAT-71's decision rule -- retroactively confirm the
profiler/dashboard would have flagged the two known historical incidents):
    tools/perf_report.py --verify-known-anomalies
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import perf_common


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Grouping identity for the regression-vs-previous-snapshot comparison.

    CAT-71 review finding 3: `leaf`/`gate`/`gpu_util` rows used to group by
    metric-identity alone, with no `hostname` -- snapshots taken on
    different fleet hosts (B200 vs A100A vs A100B, which have genuinely
    different per-leaf/per-gate wall-clock floors) landed in the same time
    series and got compared against each other as if they were repeat
    samples from the same host, producing false regressions on ordinary
    cross-host variance. `hostname` is now part of every kind's identity
    (matching `generation`, which already had it).
    """
    kind = row.get("kind")
    if kind == "leaf":
        return (kind, row.get("hostname"), row.get("device"), row.get("checkpoint_label"))
    if kind == "generation":
        return (kind, row.get("hostname"))
    if kind == "gate":
        return (kind, row.get("hostname"), row.get("gate_name"))
    if kind == "gpu_util":
        return (kind, row.get("hostname"), row.get("gpu_index"))
    return (kind,)


def _primary_metric(kind: str, row: dict[str, Any]) -> tuple[str, float | None]:
    """(metric_name, value) for a row, where a HIGHER value is worse for
    "leaf"/"gate" and a LOWER value is worse for "generation". gpu_util has no
    single scalar trend metric here (see the anomaly flag instead)."""
    if kind == "leaf":
        return "total_mean_ms", row.get("stages", {}).get("total", {}).get("mean_ms")
    if kind == "generation":
        return "rows_per_hr", row.get("rows_per_hr")
    if kind == "gate":
        games = row.get("games") or 0
        elapsed = row.get("elapsed_sec") or 0.0
        return "elapsed_sec_per_game", (elapsed / games) if games else elapsed
    return "value", None


_HIGHER_IS_WORSE = {"leaf": True, "gate": True, "generation": False}


def build_report(
    rows: list[dict[str, Any]],
    *,
    regression_threshold: float = 0.25,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    findings: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []

    for group_key, group_rows in groups.items():
        kind = group_key[0]
        group_rows = sorted(group_rows, key=lambda r: r.get("timestamp") or "")
        latest = group_rows[-1]
        previous = group_rows[-2] if len(group_rows) > 1 else None

        summary: dict[str, Any] = {
            "kind": kind,
            "identity": group_key[1:],
            "n_snapshots": len(group_rows),
            "latest_timestamp": latest.get("timestamp"),
        }

        if kind == "gpu_util":
            summary["sm_util_pct"] = latest.get("sm_util_pct")
            summary["mem_util_pct"] = latest.get("mem_util_pct")
            summary["context_thrash_flagged"] = latest.get("context_thrash_flagged")
            if latest.get("context_thrash_flagged"):
                findings.append(
                    {
                        "severity": "anomaly",
                        "kind": kind,
                        "identity": group_key[1:],
                        "message": (
                            f"GPU context-thrash signature flagged on gpu {group_key[1:]}: "
                            f"sm={latest.get('sm_util_pct')}% mem={latest.get('mem_util_pct')}% "
                            "(90%+ SM / near-0% memory-bandwidth = tiny-kernel thrash, not real compute load)"
                        ),
                    }
                )
            group_summaries.append(summary)
            continue

        if kind == "leaf" and latest.get("baseline_check", {}).get("matches_baseline") is False:
            findings.append(
                {
                    "severity": "anomaly",
                    "kind": kind,
                    "identity": group_key[1:],
                    "message": (
                        f"leaf profiler stage split does not match the documented baseline for "
                        f"{group_key[1:]}: {latest['baseline_check']}"
                    ),
                }
            )

        metric_name, metric_value = _primary_metric(kind, latest)
        summary["metric"] = metric_name
        summary["value"] = metric_value

        if previous is not None and metric_value is not None:
            _, previous_value = _primary_metric(kind, previous)
            if previous_value:
                delta_ratio = (metric_value - previous_value) / previous_value
                summary["delta_ratio_vs_previous"] = delta_ratio
                higher_is_worse = _HIGHER_IS_WORSE.get(kind, True)
                regressed = (
                    delta_ratio > regression_threshold
                    if higher_is_worse
                    else delta_ratio < -regression_threshold
                )
                if regressed:
                    findings.append(
                        {
                            "severity": "regression",
                            "kind": kind,
                            "identity": group_key[1:],
                            "message": (
                                f"{kind} {group_key[1:]}: {metric_name} moved from "
                                f"{previous_value:.4g} to {metric_value:.4g} "
                                f"({delta_ratio * 100:+.1f}%, threshold {regression_threshold * 100:.0f}%)"
                            ),
                        }
                    )

        group_summaries.append(summary)

    return {"groups": group_summaries, "findings": findings, "n_rows": len(rows)}


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# CAT-71 Performance Snapshot Report", ""]
    lines.append(f"Ledger rows: {report['n_rows']}")
    lines.append("")

    findings = report["findings"]
    lines.append(f"## Findings ({len(findings)})")
    lines.append("")
    if not findings:
        lines.append("No regressions or anomalies flagged.")
    else:
        for finding in findings:
            lines.append(f"- **[{finding['severity'].upper()}]** {finding['message']}")
    lines.append("")

    lines.append("## Snapshot groups")
    lines.append("")
    lines.append("| kind | identity | snapshots | metric | value | delta vs previous |")
    lines.append("|---|---|---|---|---|---|")
    for group in report["groups"]:
        delta = group.get("delta_ratio_vs_previous")
        delta_str = f"{delta * 100:+.1f}%" if delta is not None else "-"
        if group["kind"] == "gpu_util":
            metric = "context_thrash_flagged"
            value = group.get("context_thrash_flagged")
        else:
            metric = group.get("metric")
            value = group.get("value")
        lines.append(
            f"| {group['kind']} | {group['identity']} | {group['n_snapshots']} | "
            f"{metric} | {value} | {delta_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def _verify_known_anomalies() -> bool:
    """CAT-71's decision rule: cross-check the detectors against at least two
    known historical anomalies with ground truth, and confirm both would be
    flagged automatically. See the two incidents' provenance notes in
    tools/perf_common.py (HISTORICAL_SH_FLOOR_OVERRUN_MS,
    DOCUMENTED_GPU_PER_LEAF_MS)."""
    all_ok = True

    # Incident 1: GPU context-thrash pmon signature. Ground truth documented
    # in-repo at docs/plans/CATAN_ZERO_RESEARCH_CHRONICLE.md section 10.1:
    # "~90% SM time-occupancy with ~0-1% memory bandwidth".
    thrash_flagged = perf_common.check_gpu_context_thrash(90.0, 1.0)
    print(f"[gpu_context_thrash] sm=90% mem=1% -> flagged={thrash_flagged} (must be True)")
    all_ok = all_ok and thrash_flagged

    # Control: genuine heavy compute load (high SM AND high memory-bandwidth
    # utilization) must NOT be flagged as thrash -- otherwise the detector is
    # just "GPU is busy", not the specific anomaly signature.
    not_thrash = perf_common.check_gpu_context_thrash(90.0, 40.0)
    print(f"[gpu_context_thrash control] sm=90% mem=40% -> flagged={not_thrash} (must be False)")
    all_ok = all_ok and (not not_thrash)

    # Incident 2: SH-floor overruns measured at 32ms/105ms/119ms (CAT-71 issue
    # text). The exact originating floor value is not present in this
    # checkout (see tools/perf_common.py's HISTORICAL_SH_FLOOR_OVERRUN_MS
    # docstring) -- demonstrated here against the documented in-repo GPU
    # per-leaf benchmark (3.4ms) as an illustrative floor. Re-run this check
    # against the live per-host floor once tools/perf_snapshot.py's `leaf`
    # mode has produced same-host samples to derive it.
    floor_ms = perf_common.DOCUMENTED_GPU_PER_LEAF_MS
    for observed in perf_common.HISTORICAL_SH_FLOOR_OVERRUN_MS:
        flagged = perf_common.check_sh_floor_overrun(observed, floor_ms)
        print(f"[sh_floor_overrun] observed={observed}ms floor={floor_ms}ms -> flagged={flagged} (must be True)")
        all_ok = all_ok and flagged

    return bool(all_ok)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", default=str(perf_common.DEFAULT_LEDGER_PATH))
    parser.add_argument("--out", default=None, help="also write the markdown report to this path")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.25,
        help="relative change vs the previous snapshot (e.g. 0.25 = 25%%) that counts as a regression",
    )
    parser.add_argument(
        "--verify-known-anomalies",
        action="store_true",
        help="retroactively check the two CAT-71 ground-truth incidents; exits nonzero if either isn't flagged",
    )
    args = parser.parse_args(argv)

    if args.verify_known_anomalies:
        ok = _verify_known_anomalies()
        raise SystemExit(0 if ok else 1)

    rows = perf_common.load_ledger(Path(args.ledger))
    report = build_report(rows, regression_threshold=args.regression_threshold)
    markdown = render_markdown(report)
    print(markdown)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"findings": len(report["findings"]), "groups": len(report["groups"])}, sort_keys=True))


if __name__ == "__main__":
    main()
