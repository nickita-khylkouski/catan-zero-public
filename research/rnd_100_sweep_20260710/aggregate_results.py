#!/usr/bin/env python3
"""Validate and aggregate the isolated 100+ trial R&D sweep.

This script is intentionally independent of production experiment registries.
It consumes one or more JSONL files, rejects ambiguous records, writes a flat
CSV and a compact Markdown summary, and optionally renders screening plots when
matplotlib is installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_COMMIT = "c807874940fd5b3e4c51775f33a64279786504da"
FAMILIES = {"architecture", "learner", "systems"}
STATUSES = {"passed", "failed", "invalid"}
MINIMUMS = {"architecture": 36, "learner": 36, "systems": 34}
REQUIRED = {
    "trial_id",
    "family",
    "status",
    "source_commit",
    "host",
    "gpu",
    "seed",
    "config",
    "input_id",
    "wall_seconds",
    "metrics",
    "failure",
}


def _canonical_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{line_number}: record must be an object")
            missing = REQUIRED - set(record)
            if missing:
                raise SystemExit(f"{path}:{line_number}: missing fields {sorted(missing)}")
            trial_id = record["trial_id"]
            if not isinstance(trial_id, str) or not trial_id or trial_id in seen_ids:
                raise SystemExit(f"{path}:{line_number}: invalid/duplicate trial_id {trial_id!r}")
            seen_ids.add(trial_id)
            if record["family"] not in FAMILIES:
                raise SystemExit(f"{path}:{line_number}: invalid family")
            if record["status"] not in STATUSES:
                raise SystemExit(f"{path}:{line_number}: invalid status")
            if record["source_commit"] != SOURCE_COMMIT:
                raise SystemExit(f"{path}:{line_number}: source commit drift")
            if not isinstance(record["config"], dict) or not isinstance(record["metrics"], dict):
                raise SystemExit(f"{path}:{line_number}: config/metrics must be objects")
            if record["status"] == "passed" and record["failure"] is not None:
                raise SystemExit(f"{path}:{line_number}: passed trial has failure text")
            if record["status"] != "passed" and not record["failure"]:
                raise SystemExit(f"{path}:{line_number}: failed/invalid trial lacks reason")
            record = dict(record)
            record["config_sha256"] = _canonical_digest(record["config"])
            record["source_file"] = str(path)
            records.append(record)
    return records


def _flatten(records: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    metric_keys = sorted({key for row in records for key in row["metrics"]})
    fields = [
        "trial_id",
        "family",
        "status",
        "host",
        "gpu",
        "seed",
        "wall_seconds",
        "input_id",
        "config_sha256",
        "failure",
        "source_file",
        *[f"metric.{key}" for key in metric_keys],
    ]
    flat: list[dict[str, Any]] = []
    for row in records:
        value = {key: row.get(key) for key in fields if not key.startswith("metric.")}
        value.update({f"metric.{key}": row["metrics"].get(key) for key in metric_keys})
        flat.append(value)
    return fields, flat


def _write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    counts = Counter(row["family"] for row in records)
    status = Counter((row["family"], row["status"]) for row in records)
    gpu_counts = Counter(row["gpu"] for row in records)
    config_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        config_groups[(row["family"], row["config_sha256"])].append(row)
    repeated = sum(1 for rows in config_groups.values() if len(rows) > 1)
    lines = [
        "# 100+ Trial Sweep Summary",
        "",
        f"Registered trials: **{len(records)}**",
        "",
        "| Family | Registered | Passed | Failed | Invalid | Minimum met |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for family in sorted(FAMILIES):
        lines.append(
            f"| {family} | {counts[family]} | {status[(family, 'passed')]} | "
            f"{status[(family, 'failed')]} | {status[(family, 'invalid')]} | "
            f"{'yes' if counts[family] >= MINIMUMS[family] else 'no'} |"
        )
    lines.extend(["", f"Repeated configurations with multiple seeds/runs: **{repeated}**", ""])
    lines.extend(["## Hardware coverage", "", "| GPU | Trials |", "|---|---:|"])
    lines.extend(f"| {gpu} | {count} |" for gpu, count in sorted(gpu_counts.items()))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = _records(args.inputs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields, rows = _flatten(records)
    with (args.out_dir / "all_trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(args.out_dir / "summary.md", records)
    counts = Counter(row["family"] for row in records)
    unmet = {family: minimum - counts[family] for family, minimum in MINIMUMS.items() if counts[family] < minimum}
    print(json.dumps({"trials": len(records), "counts": counts, "unmet": unmet}, sort_keys=True))
    return 1 if unmet else 0


if __name__ == "__main__":
    raise SystemExit(main())
