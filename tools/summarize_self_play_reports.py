from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize completed self-play JSON reports."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=("runs/self_play",),
        help=(
            "Report files, directories, or glob patterns. Directories are "
            "searched recursively for JSON reports."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="If >0, print only the top N rows after ranking.",
    )
    parser.add_argument(
        "--min-heuristic-win-rate",
        type=float,
        default=0.50,
        help="Promotion gate for heuristic evaluation.",
    )
    parser.add_argument(
        "--min-value-win-rate",
        type=float,
        default=0.25,
        help="Promotion gate for direct value-policy evaluation.",
    )
    args = parser.parse_args()

    reports = [_load_summary(path, args) for path in _iter_report_paths(args.paths)]

    reports.sort(
        key=lambda row: (
            row["value_win_rate"] if row["value_win_rate"] is not None else -1.0,
            row["heuristic_win_rate"] if row["heuristic_win_rate"] is not None else -1.0,
            row["random_win_rate"] if row["random_win_rate"] is not None else -1.0,
            row["value_games"],
            row["heuristic_games"],
        ),
        reverse=True,
    )
    if args.top > 0:
        reports = reports[: args.top]

    print(
        "status\theur\tvalue\trandom\tgames_h\tgames_v\tselected\treport",
        flush=True,
    )
    for row in reports:
        print(
            "\t".join(
                (
                    row["status"],
                    _fmt(row["heuristic_win_rate"]),
                    _fmt(row["value_win_rate"]),
                    _fmt(row["random_win_rate"]),
                    str(row["heuristic_games"]),
                    str(row["value_games"]),
                    row["selected_checkpoint"],
                    row["path"],
                )
            ),
            flush=True,
        )


def _iter_report_paths(paths: Iterable[str]) -> list[Path]:
    report_paths: list[Path] = []
    for pattern in paths:
        if any(ch in pattern for ch in "*?["):
            matched = sorted(Path().glob(pattern))
        else:
            path = Path(pattern)
            matched = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        report_paths.extend(path for path in matched if path.is_file())
    return sorted(set(report_paths))


def _load_summary(path: Path, args: argparse.Namespace) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    heuristic, value, random = _extract_evals(report)
    heuristic_wr = _as_float(heuristic.get("win_rate"))
    value_wr = _as_float(value.get("win_rate"))
    status = "hold"
    if heuristic_wr is not None and value_wr is not None:
        if (
            heuristic_wr >= args.min_heuristic_win_rate
            and value_wr >= args.min_value_win_rate
        ):
            status = "promote"
        elif heuristic_wr < args.min_heuristic_win_rate:
            status = "reject"
    elif heuristic_wr is not None and heuristic_wr < args.min_heuristic_win_rate:
        status = "reject"
    return {
        "path": str(path),
        "status": status,
        "heuristic_win_rate": heuristic_wr,
        "value_win_rate": value_wr,
        "random_win_rate": _as_float(random.get("win_rate")),
        "heuristic_games": heuristic.get("games", 0),
        "value_games": value.get("games", 0),
        "selected_checkpoint": _selected_checkpoint(report),
    }


def _extract_evals(report: dict) -> tuple[dict, dict, dict]:
    heuristic = report.get("eval_vs_heuristic") or {}
    value = report.get("eval_vs_value") or {}
    random = report.get("eval_vs_random") or {}
    if "win_rate" in report and "opponent" in report:
        opponent = str(report["opponent"])
        if opponent == "heuristic":
            heuristic = report
        elif opponent == "catanatron_value":
            value = report
        elif opponent == "random":
            random = report
    return heuristic, value, random


def _selected_checkpoint(report: dict) -> str:
    selected = (
        report.get("selected_training_checkpoint")
        or report.get("selected_warmup_checkpoint")
        or report.get("best_checkpoint")
    )
    if isinstance(selected, dict):
        return str(selected.get("path") or "-")
    if selected:
        return str(selected)
    return "-"


def _as_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
