#!/usr/bin/env python3
"""CLI: noise-vs-spread trend (CAT-25 measurement 4).

Pure glue over two ALREADY-PRODUCED report families, no new evaluation runs:

  - tools/opening_panel.py `eval` output (its top-level `"aggregate"` key,
    see `aggregate()` in that file), which carries `mean_raw_q_spread` and
    `mean_spread_over_floor`.
  - tools/f74_symmetry_eval.py output (its top-level `"summary"` key, see
    `main()` in that file: `write_json(args.out, {"summary": summary,
    "per_root": per_root})`), which carries
    `summary["symmetry_inconsistency"]["q_candidate_orientation_std"]["mean"]`
    (a `_stat()` dict with mean/median/p90/max).

FIELD-AVAILABILITY NOTE: the CAT-25 ticket asks for "top-5 candidate
Q-spread", but `opening_panel.py`'s aggregate only reports `mean_raw_q_spread`
over ALL visited candidates (not specifically the top 5) -- there is no
existing top-5-specific field in either source JSON. This tool uses
`mean_raw_q_spread` and `mean_spread_over_floor` as the best available
proxies and says so explicitly here (and in its output) rather than silently
mislabeling a different quantity as "top-5".

For an ordered list of generations (oldest -> newest), this tool extracts,
per generation:
  - `top5_q_spread_proxy`: opening-panel `aggregate.mean_raw_q_spread`
  - `top5_q_spread_over_floor_proxy`: opening-panel `aggregate.mean_spread_over_floor`
  - `orientation_noise_std`: f74 `summary.symmetry_inconsistency
    .q_candidate_orientation_std.mean`

and computes, across the generation-ordered series: a simple least-squares
slope (index vs value) for each series, and the Pearson correlation
coefficient between the spread series and the noise series (implemented
directly with numpy -- f74_symmetry_eval.py already imports numpy, so this
adds no new dependency). A full statistical trend test is deliberately not
attempted; slope + correlation is the "sane, inspectable" level of rigor this
diagnostic bundle calls for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from factory_common import write_json  # noqa: E402


def extract_generation_metrics(opening_panel_json: dict[str, Any], f74_json: dict[str, Any]) -> dict[str, Any]:
    """Pull the spread and orientation-noise fields out of one generation's
    already-produced opening_panel.py `eval` output and f74_symmetry_eval.py
    output."""
    aggregate = opening_panel_json.get("aggregate", {})
    summary = f74_json.get("summary", {})
    orientation_std = summary.get("symmetry_inconsistency", {}).get("q_candidate_orientation_std", {})
    return {
        "top5_q_spread_proxy": aggregate.get("mean_raw_q_spread"),
        "top5_q_spread_over_floor_proxy": aggregate.get("mean_spread_over_floor"),
        "orientation_noise_std": orientation_std.get("mean"),
    }


def _slope(values: list[float | None]) -> float | None:
    """Least-squares slope of value vs index, ignoring None entries. None if
    fewer than 2 non-None points exist."""
    pairs = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pairs) < 2:
        return None
    xs = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ys = np.asarray([p[1] for p in pairs], dtype=np.float64)
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = float(((xs - x_mean) ** 2).sum())
    if denom <= 0.0:
        return None
    return float(((xs - x_mean) * (ys - y_mean)).sum() / denom)


def _pearson(a: list[float | None], b: list[float | None]) -> float | None:
    """Pearson correlation coefficient between two same-length series,
    computed directly with numpy over the positions where BOTH are non-None.
    None if fewer than 2 usable pairs, or either series has zero variance."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ys = np.asarray([p[1] for p in pairs], dtype=np.float64)
    x_std = xs.std()
    y_std = ys.std()
    if x_std <= 0.0 or y_std <= 0.0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def build_trend_report(generations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`generations` is an ORDERED (oldest -> newest) mapping of generation
    label -> {"opening_panel_json": {...}, "f74_json": {...}} (already
    loaded dicts). Returns the per-generation extraction, series, slopes, and
    the spread-vs-noise Pearson correlation."""
    labels = list(generations.keys())
    per_generation: dict[str, Any] = {}
    for label, paths in generations.items():
        per_generation[label] = extract_generation_metrics(
            paths["opening_panel_json"], paths["f74_json"]
        )

    spread_series = [per_generation[label]["top5_q_spread_proxy"] for label in labels]
    spread_over_floor_series = [
        per_generation[label]["top5_q_spread_over_floor_proxy"] for label in labels
    ]
    noise_series = [per_generation[label]["orientation_noise_std"] for label in labels]

    return {
        "measurement": "noise_vs_spread_trend",
        "field_availability_note": (
            "opening_panel.py has no top-5-specific Q-spread field; "
            "top5_q_spread_proxy/top5_q_spread_over_floor_proxy are "
            "mean_raw_q_spread/mean_spread_over_floor over ALL visited "
            "candidates, used as the best available proxy."
        ),
        "generations_ordered": labels,
        "per_generation": per_generation,
        "series": {
            "top5_q_spread_proxy": spread_series,
            "top5_q_spread_over_floor_proxy": spread_over_floor_series,
            "orientation_noise_std": noise_series,
        },
        "trend": {
            "top5_q_spread_proxy_slope": _slope(spread_series),
            "top5_q_spread_over_floor_proxy_slope": _slope(spread_over_floor_series),
            "orientation_noise_std_slope": _slope(noise_series),
        },
        "pearson_correlation": {
            "top5_q_spread_proxy_vs_orientation_noise_std": _pearson(spread_series, noise_series),
            "top5_q_spread_over_floor_proxy_vs_orientation_noise_std": _pearson(
                spread_over_floor_series, noise_series
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help=(
            'JSON config file: {"<gen_label>": {"opening_panel_json": "path", '
            '"f74_json": "path"}, ...}, in oldest-to-newest order (JSON object key '
            "order is preserved on load in Python 3.7+)."
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    generations: dict[str, dict[str, Any]] = {}
    for label, paths in config.items():
        opening_panel_json = json.loads(Path(paths["opening_panel_json"]).read_text(encoding="utf-8"))
        f74_json = json.loads(Path(paths["f74_json"]).read_text(encoding="utf-8"))
        generations[label] = {"opening_panel_json": opening_panel_json, "f74_json": f74_json}

    report = build_trend_report(generations)
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
