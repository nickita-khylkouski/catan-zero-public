#!/usr/bin/env python3
"""Synthetic validation: pentanomial GSPRT vs concordant-only Bernoulli SPRT.

Confirms the claim behind task #1 -- the trinomial (no-draw pentanomial)
GSPRT extracts MORE evidence from the SAME color-swapped pairs than the
incumbent concordant-only rule (which maps WW->win, LL->loss and DISCARDS
every split), at matched nominal error control.

Model. A color-swapped pair is a trinomial over the candidate's pair score
{0 (LL), 1/2 (split), 1 (WW)}. Given a per-game win probability `p` (== the
pair-score mean) and a `split_rate` = P(split), the concordant categories are
    p_WW = p - split_rate/2,   p_LL = 1 - p - split_rate/2.
Both must be >= 0, which bounds the admissible split_rate for a given p.

For each truth (p, split_rate) we draw pairs one at a time and feed the SAME
stream to both sequential tests, stopping each at its Wald boundary (or a
--max-pairs cap). Over many trials we report the accept-H1 / accept-H0 /
unresolved rates and the mean pairs-to-decision. Pure stdlib; no numpy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from sprt_gate import (  # noqa: E402
    GATE_CONFIGS,
    elo_to_score,
    evaluate_pentanomial_sprt,
    evaluate_sprt,
    resolve_gate_config,
    sprt_bounds,
    pentanomial_llr,
)


def pair_probs(p: float, split_rate: float) -> tuple[float, float, float]:
    """(p_LL, p_split, p_WW) for a per-game win prob `p` and split rate."""
    p_ww = p - split_rate / 2.0
    p_ll = 1.0 - p - split_rate / 2.0
    if p_ww < -1e-12 or p_ll < -1e-12:
        raise ValueError(f"infeasible: p={p}, split_rate={split_rate} -> p_WW={p_ww}, p_LL={p_ll}")
    return max(p_ll, 0.0), split_rate, max(p_ww, 0.0)


def _draw_pair(probs: tuple[float, float, float], rng: random.Random) -> float:
    r = rng.random()
    p_ll, p_split, _p_ww = probs
    if r < p_ll:
        return 0.0
    if r < p_ll + p_split:
        return 0.5
    return 1.0


def run_one_trial(
    probs: tuple[float, float, float],
    *,
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
    max_pairs: int,
    rng: random.Random,
) -> dict[str, Any]:
    lower, upper = sprt_bounds(alpha, beta)
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    n_ll = n_split = n_ww = 0
    pent_done: dict[str, Any] | None = None
    conc_done: dict[str, Any] | None = None
    for i in range(1, max_pairs + 1):
        score = _draw_pair(probs, rng)
        if score == 0.0:
            n_ll += 1
        elif score == 0.5:
            n_split += 1
        else:
            n_ww += 1

        if pent_done is None:
            llr = pentanomial_llr([n_ll, n_split, n_ww], s0=s0, s1=s1)
            if llr >= upper:
                pent_done = {"decision": "H1", "pairs": i}
            elif llr <= lower:
                pent_done = {"decision": "H0", "pairs": i}
        if conc_done is None:
            # concordant Bernoulli LLR over decisive pairs only
            wins, losses = n_ww, n_ll
            llr_c = wins * _bern_inc(True, s0, s1) + losses * _bern_inc(False, s0, s1)
            if llr_c >= upper:
                conc_done = {"decision": "H1", "pairs": i}
            elif llr_c <= lower:
                conc_done = {"decision": "H0", "pairs": i}
        if pent_done and conc_done:
            break
    return {
        "pent": pent_done or {"decision": "continue", "pairs": max_pairs},
        "conc": conc_done or {"decision": "continue", "pairs": max_pairs},
    }


def _bern_inc(win: bool, s0: float, s1: float) -> float:
    import math

    return math.log(s1 / s0) if win else math.log((1.0 - s1) / (1.0 - s0))


def summarize(trials: list[dict[str, Any]], key: str) -> dict[str, Any]:
    decs = [t[key]["decision"] for t in trials]
    n = len(decs)
    resolved = [t[key]["pairs"] for t in trials if t[key]["decision"] != "continue"]
    return {
        "p_H1": sum(1 for d in decs if d == "H1") / n,
        "p_H0": sum(1 for d in decs if d == "H0") / n,
        "p_unresolved": sum(1 for d in decs if d == "continue") / n,
        "mean_pairs_to_decision": (mean(resolved) if resolved else None),
        "resolved_fraction": len(resolved) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--max-pairs", type=int, default=4000)
    parser.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default="flywheel",
        help="Named SPRT gate config (CAT-7) providing --elo0/--elo1/--alpha/--beta defaults; explicit flags override.",
    )
    parser.add_argument("--elo0", type=float, default=None, help="Override --gate-config's elo0.")
    parser.add_argument("--elo1", type=float, default=None, help="Override --gate-config's elo1.")
    parser.add_argument("--alpha", type=float, default=None, help="Override --gate-config's alpha.")
    parser.add_argument("--beta", type=float, default=None, help="Override --gate-config's beta.")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()
    _gate_cfg, _gate_params = resolve_gate_config(
        args.gate_config, elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta
    )
    args.elo0, args.elo1, args.alpha, args.beta = (
        _gate_params["elo0"],
        _gate_params["elo1"],
        _gate_params["alpha"],
        _gate_params["beta"],
    )

    scenarios = [
        # (label, per-game p, split_rate)
        ("H0_true_low_split", 0.50, 0.30),
        ("H0_true_high_split", 0.50, 0.70),
        ("H1_boundary_low_split", elo_to_score(args.elo1), 0.30),
        ("H1_boundary_high_split", elo_to_score(args.elo1), 0.70),
        ("clear_H1_low_split", 0.60, 0.30),
        ("clear_H1_high_split", 0.60, 0.60),
    ]
    report: dict[str, Any] = {"config": vars(args), "scenarios": {}}
    for label, p, split_rate in scenarios:
        probs = pair_probs(p, split_rate)
        rng = random.Random(hash((args.seed, label)) & 0xFFFFFFFF)
        trials = [
            run_one_trial(
                probs,
                elo0=args.elo0,
                elo1=args.elo1,
                alpha=args.alpha,
                beta=args.beta,
                max_pairs=args.max_pairs,
                rng=rng,
            )
            for _ in range(args.trials)
        ]
        report["scenarios"][label] = {
            "per_game_p": p,
            "split_rate": split_rate,
            "pair_probs_LL_split_WW": probs,
            "pentanomial": summarize(trials, "pent"),
            "concordant": summarize(trials, "conc"),
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
