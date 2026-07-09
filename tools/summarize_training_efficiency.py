from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PHASES = (
    ("collect", "collect_seconds"),
    ("ppo_update", "ppo_update_seconds"),
    ("anchor_collect", "anchor_collect_seconds"),
    ("anchor_update", "anchor_update_seconds"),
    ("checkpoint", "checkpoint_seconds"),
    ("other", "other_seconds"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize CatanZero training efficiency from train_ppo JSON reports "
            "or JSONL logs. Optionally emit folded-stack rows for flamegraph.pl."
        )
    )
    parser.add_argument("paths", nargs="+", help="Report JSON or log files.")
    parser.add_argument("--output", help="Write JSON summary to this path.")
    parser.add_argument(
        "--folded-output",
        help=(
            "Write folded stack timing rows. Values are milliseconds, so the "
            "file can be rendered by FlameGraph/flamegraph.pl."
        ),
    )
    args = parser.parse_args()

    runs = [_summarize_path(Path(path)) for path in args.paths]
    payload = {
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.folded_output:
        output = Path(args.folded_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_folded_stacks(runs), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _summarize_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "missing": True,
            "iterations": 0,
            "total_seconds": 0.0,
            "phase_seconds": {name: 0.0 for name, _ in PHASES},
            "phase_fractions": {name: 0.0 for name, _ in PHASES},
            "ppo_samples": 0.0,
            "anchor_samples": 0.0,
            "ppo_samples_per_second": 0.0,
            "anchor_samples_per_second": 0.0,
            "dominant_phase": None,
        }
    iterations = _load_iterations(path)
    phase_seconds = {name: 0.0 for name, _ in PHASES}
    total_seconds = 0.0
    ppo_samples = 0.0
    anchor_samples = 0.0
    for row in iterations:
        timing = row.get("timing") or {}
        total_seconds += float(timing.get("iteration_seconds") or 0.0)
        ppo_samples += float(row.get("samples") or 0.0)
        anchor = row.get("anchor") or {}
        anchor_samples += float(anchor.get("samples") or 0.0)
        for name, key in PHASES:
            phase_seconds[name] += float(timing.get(key) or 0.0)

    total = max(total_seconds, 1e-9)
    phase_fractions = {
        name: phase_seconds[name] / total
        for name, _ in PHASES
    }
    return {
        "path": str(path),
        "iterations": len(iterations),
        "total_seconds": total_seconds,
        "phase_seconds": phase_seconds,
        "phase_fractions": phase_fractions,
        "ppo_samples": ppo_samples,
        "anchor_samples": anchor_samples,
        "ppo_samples_per_second": ppo_samples / max(
            phase_seconds["collect"] + phase_seconds["ppo_update"],
            1e-9,
        ),
        "anchor_samples_per_second": anchor_samples / max(
            phase_seconds["anchor_collect"] + phase_seconds["anchor_update"],
            1e-9,
        ),
        "dominant_phase": max(phase_seconds, key=phase_seconds.get)
        if total_seconds > 0.0
        else None,
    }


def _load_iterations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload.get("iterations"), list):
            return [row for row in payload["iterations"] if isinstance(row, dict)]
    iterations: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        ppo = payload.get("ppo")
        if isinstance(ppo, dict):
            iterations.append(ppo)
    return iterations


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    phase_seconds = {name: 0.0 for name, _ in PHASES}
    total_seconds = 0.0
    ppo_samples = 0.0
    anchor_samples = 0.0
    iterations = 0
    for run in runs:
        total_seconds += float(run.get("total_seconds") or 0.0)
        ppo_samples += float(run.get("ppo_samples") or 0.0)
        anchor_samples += float(run.get("anchor_samples") or 0.0)
        iterations += int(run.get("iterations") or 0)
        for name in phase_seconds:
            phase_seconds[name] += float((run.get("phase_seconds") or {}).get(name) or 0.0)
    total = max(total_seconds, 1e-9)
    return {
        "runs": len(runs),
        "iterations": iterations,
        "total_seconds": total_seconds,
        "phase_seconds": phase_seconds,
        "phase_fractions": {
            name: phase_seconds[name] / total
            for name in phase_seconds
        },
        "ppo_samples": ppo_samples,
        "anchor_samples": anchor_samples,
        "ppo_samples_per_second": ppo_samples / max(
            phase_seconds["collect"] + phase_seconds["ppo_update"],
            1e-9,
        ),
        "anchor_samples_per_second": anchor_samples / max(
            phase_seconds["anchor_collect"] + phase_seconds["anchor_update"],
            1e-9,
        ),
        "dominant_phase": max(phase_seconds, key=phase_seconds.get)
        if total_seconds > 0.0
        else None,
    }


def _folded_stacks(runs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for run in runs:
        label = Path(str(run["path"])).stem
        for phase, seconds in (run.get("phase_seconds") or {}).items():
            millis = int(round(float(seconds) * 1000.0))
            if millis > 0:
                lines.append(f"train;{label};{phase} {millis}")
    return "\n".join(lines) + ("\n" if lines else "")


if __name__ == "__main__":
    main()
