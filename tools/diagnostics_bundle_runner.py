#!/usr/bin/env python3
"""CLI: CAT-25 diagnostics bundle one-pager synthesis.

Glue-only: assembles the already-produced JSON outputs of the four CAT-25
measurements --

  1. tools/search_snr_probe.py            (mechanism A: SNR-decay)
  2. tools/rollout_doubling_probe.py       (mechanism B: ExIt fixed-point)
  3. tools/corpus_diversity_scan.py        (mechanism C: distribution narrowing)
  4. tools/noise_vs_spread_trend.py        (cross-cutting: noise vs spread)

-- into one JSON (and a printed summary) with the four measurements' key
numbers side by side, plus a `mechanism_weight_conclusion` section.

CRITICAL: this runner does NOT have real host-generated diagnostic data
available in this dev sandbox. Its job is to be the CORRECT PIPELINE, not to
produce a real verdict. Whenever a required input JSON is missing/absent,
the corresponding weight field is left `null` and `rationale` explains which
input(s) are missing and why no number was fabricated -- this mirrors the
CAT-25 ticket's explicit requirement that every number be sane and
inspectable before it's trusted.

Heuristic (documented, NOT a validated formula -- a STARTING POINT for human
judgment, see `_mechanism_weights`): when ALL FOUR inputs are present, this
runner computes three raw "evidence scores" (higher = more evidence for that
mechanism), then normalizes them to sum to 1:
  - score_A (SNR-decay): the search-SNR probe's per-checkpoint decay in
    mean_argmax_agreement from the oldest to the newest checkpoint in its
    lineage sweep, MINUS the change in mean_kl_pi_vs_prior over the same
    span (the ticket's own diagnostic signature: agreement decaying while
    kl-vs-prior stays flat is evidence FOR mechanism A).
  - score_B (ExIt fixed-point): `abs(candidate_win_rate - 0.5)` from the
    rollout-doubling summary, INVERTED (`0.5 - abs(win_rate - 0.5)`) -- a
    win rate near 50% under doubled search is evidence FOR a fixed point
    (mechanism B), a win rate well above 50% is evidence AGAINST it.
  - score_C (distribution narrowing): the corpus-diversity scan's
    herfindahl_index (opening-line concentration) directly -- higher
    concentration is evidence FOR mechanism C.
Each score is clamped to [0, 1] before normalizing; if all three scores are
non-positive after clamping, weights are left null with a rationale instead
of dividing by zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from factory_common import write_json  # noqa: E402


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _search_snr_summary(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    per_checkpoint = data.get("per_checkpoint", {})
    checkpoints = data.get("checkpoints", list(per_checkpoint.keys()))
    aggregates = {
        checkpoint: per_checkpoint.get(checkpoint, {}).get("aggregate", {}) for checkpoint in checkpoints
    }
    return {
        "checkpoints": checkpoints,
        "aggregate_by_checkpoint": aggregates,
    }


def _rollout_doubling_summary(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    summary = data.get("rollout_doubling_summary")
    if summary is not None:
        return summary
    # Also accept a raw H2H-style dict for flexibility (e.g. hand-built test fixtures).
    return {
        "candidate_win_rate": data.get("candidate_win_rate"),
        "pentanomial_sprt": data.get("pentanomial_sprt"),
        "pair_diagnostics": data.get("pair_diagnostics"),
    }


def _diversity_summary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for entry in entries:
        result.append(
            {
                "generation_label": entry.get("generation_label"),
                "unique_state_fraction_cheap": entry.get("unique_state_fraction_cheap", {}).get(
                    "unique_fraction"
                ),
                "unique_state_fraction_content": (
                    (entry.get("unique_state_fraction_content") or {}).get("unique_fraction")
                ),
                "opening_line_herfindahl": entry.get("opening_line_concentration", {}).get(
                    "herfindahl_index"
                ),
                "opening_line_top1_fraction": entry.get("opening_line_concentration", {}).get(
                    "top1_fraction"
                ),
                "mean_normalized_entropy": entry.get("opening_entropy", {}).get(
                    "mean_normalized_entropy"
                ),
            }
        )
    return result


def _noise_spread_summary(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    return {
        "generations_ordered": data.get("generations_ordered"),
        "trend": data.get("trend"),
        "pearson_correlation": data.get("pearson_correlation"),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mechanism_weights(
    search_snr: dict[str, Any] | None,
    rollout_doubling: dict[str, Any] | None,
    diversity: list[dict[str, Any]],
    missing: list[str],
) -> dict[str, Any]:
    if missing:
        return {
            "weight_A_snr_decay": None,
            "weight_B_exit_fixed_point": None,
            "weight_C_distribution_narrowing": None,
            "rationale": (
                "insufficient data: missing " + ", ".join(missing) + " output(s) -- "
                "never substituting a guess for a missing input; rerun the missing probe(s) "
                "and re-invoke this bundle runner."
            ),
        }

    # score_A: agreement-decay minus kl-vs-prior-drift, across the search-SNR
    # probe's checkpoint lineage (oldest -> newest, per its `checkpoints` list order).
    checkpoints = search_snr["checkpoints"]
    aggregates = search_snr["aggregate_by_checkpoint"]
    if len(checkpoints) >= 2:
        first, last = checkpoints[0], checkpoints[-1]
        agreement_first = aggregates.get(first, {}).get("mean_argmax_agreement")
        agreement_last = aggregates.get(last, {}).get("mean_argmax_agreement")
        kl_first = aggregates.get(first, {}).get("mean_kl_pi_vs_prior")
        kl_last = aggregates.get(last, {}).get("mean_kl_pi_vs_prior")
        agreement_decay = (
            (agreement_first - agreement_last)
            if agreement_first is not None and agreement_last is not None
            else 0.0
        )
        kl_drift = (kl_last - kl_first) if kl_first is not None and kl_last is not None else 0.0
        score_a = _clamp01(agreement_decay - abs(kl_drift))
    else:
        score_a = 0.0

    # score_B: how close to 50% the rollout-doubling win rate is (near-50% ==
    # more evidence for an ExIt fixed point).
    win_rate = rollout_doubling.get("candidate_win_rate")
    score_b = _clamp01(0.5 - abs(win_rate - 0.5)) if win_rate is not None else 0.0

    # score_C: mean opening-line herfindahl index across generations (higher
    # concentration == more evidence for distribution narrowing).
    herfindahls = [
        entry["opening_line_herfindahl"]
        for entry in diversity
        if entry.get("opening_line_herfindahl") is not None
    ]
    score_c = _clamp01(sum(herfindahls) / len(herfindahls)) if herfindahls else 0.0

    total = score_a + score_b + score_c
    if total <= 0.0:
        return {
            "weight_A_snr_decay": None,
            "weight_B_exit_fixed_point": None,
            "weight_C_distribution_narrowing": None,
            "rationale": (
                "all three evidence scores were non-positive after clamping -- inputs were "
                "present but uninformative (e.g. a single-checkpoint SNR sweep, a missing win "
                "rate, or no opening-line data); leaving weights null rather than dividing by zero."
            ),
        }

    return {
        "weight_A_snr_decay": score_a / total,
        "weight_B_exit_fixed_point": score_b / total,
        "weight_C_distribution_narrowing": score_c / total,
        "raw_scores": {"score_A": score_a, "score_B": score_b, "score_C": score_c},
        "rationale": (
            "heuristic combination of: (A) search-SNR argmax-agreement decay net of "
            "kl-vs-prior drift across the checkpoint lineage; (B) closeness of the "
            "rollout-doubling win rate to 50%; (C) mean opening-line Herfindahl "
            "concentration across generations. This is a STARTING POINT for human "
            "judgment, not a validated formula -- see tools/diagnostics_bundle_runner.py "
            "module docstring for the exact definitions."
        ),
    }


def build_bundle(
    *,
    search_snr_json: dict[str, Any] | None,
    rollout_doubling_json: dict[str, Any] | None,
    diversity_jsons: list[dict[str, Any]],
    noise_spread_json: dict[str, Any] | None,
    search_snr_path: str | None = None,
    rollout_doubling_path: str | None = None,
    diversity_paths: list[str] | None = None,
    noise_spread_path: str | None = None,
) -> dict[str, Any]:
    missing: list[str] = []
    if search_snr_json is None:
        missing.append(f"search_snr_probe (path={search_snr_path!r})")
    if rollout_doubling_json is None:
        missing.append(f"rollout_doubling_probe (path={rollout_doubling_path!r})")
    if not diversity_jsons:
        missing.append(f"corpus_diversity_scan (paths={diversity_paths!r})")
    if noise_spread_json is None:
        missing.append(f"noise_vs_spread_trend (path={noise_spread_path!r})")

    search_snr = _search_snr_summary(search_snr_json)
    rollout_doubling = _rollout_doubling_summary(rollout_doubling_json)
    diversity = _diversity_summary(diversity_jsons)
    noise_spread = _noise_spread_summary(noise_spread_json)

    conclusion = _mechanism_weights(search_snr, rollout_doubling, diversity, missing)

    return {
        "bundle": "cat25_diagnostics_bundle",
        "inputs_present": {
            "search_snr_probe": search_snr_json is not None,
            "rollout_doubling_probe": rollout_doubling_json is not None,
            "corpus_diversity_scan": bool(diversity_jsons),
            "noise_vs_spread_trend": noise_spread_json is not None,
        },
        "measurement_1_search_snr_probe": search_snr,
        "measurement_2_rollout_doubling_probe": rollout_doubling,
        "measurement_3_corpus_diversity_scan": diversity,
        "measurement_4_noise_vs_spread_trend": noise_spread,
        "mechanism_weight_conclusion": conclusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-snr-json", default=None)
    parser.add_argument("--rollout-doubling-json", default=None)
    parser.add_argument(
        "--diversity-json",
        action="append",
        default=None,
        help="One corpus_diversity_scan.py --out JSON path per generation; repeatable.",
    )
    parser.add_argument("--noise-spread-json", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    search_snr_json = _load_json(args.search_snr_json)
    rollout_doubling_json = _load_json(args.rollout_doubling_json)
    diversity_paths = args.diversity_json or []
    diversity_jsons = [d for d in (_load_json(p) for p in diversity_paths) if d is not None]
    noise_spread_json = _load_json(args.noise_spread_json)

    bundle = build_bundle(
        search_snr_json=search_snr_json,
        rollout_doubling_json=rollout_doubling_json,
        diversity_jsons=diversity_jsons,
        noise_spread_json=noise_spread_json,
        search_snr_path=args.search_snr_json,
        rollout_doubling_path=args.rollout_doubling_json,
        diversity_paths=diversity_paths,
        noise_spread_path=args.noise_spread_json,
    )
    write_json(args.out, bundle)
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
