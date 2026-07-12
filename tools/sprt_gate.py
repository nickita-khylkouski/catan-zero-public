from __future__ import annotations

"""fishtest-style Sequential Probability Ratio Test (SPRT) for promotion gates.

Ports the elo0/elo1 -> log-likelihood-ratio (LLR) math used by fishtest /
cutechess-cli SPRT gates to our paired-seed scoreboards. Catan games always
have exactly one winner (no draws), so the exact two-outcome Wald SPRT
applies directly -- this is the exact MLE-optimal test for distinguishing
H0: win probability == elo_to_score(elo0) from H1: win probability ==
elo_to_score(elo1), and it coincides with fishtest's draw-model LLR in the
no-draw limit. Pure stdlib: no numpy/scipy dependency.

CAUTION on sample size: elo1=5 (the fishtest STC-style default) is a small
effect size. Resolving it at alpha=beta=0.05 typically needs several hundred
to several thousand paired games; a 60%-over-200-games sample will usually
land in "continue", not "H1" (see tests for the exact math + a demonstration
with a coarser elo1 that does resolve quickly).

CHOOSING elo0/elo1 for THIS project: our promotion gates target >=55% win
rate, which is only about +35 Elo (elo_to_score(35) ~= 0.548). Use
--elo0 0 --elo1 30 for standard promotion gates -- it resolves a genuine
~55% candidate in the low hundreds to ~1000 paired games, which matches our
gate budgets. Reserve the tighter --elo0 0 --elo1 5 (fishtest's small-effect
default) for fine-grained regression checks where thousands of paired games
are actually available (e.g. accumulated ladder history), not single gate
runs.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from catan_zero.rl.config_cli import add_config_flags, resolve_config
from catan_zero.rl.pipeline_configs import GateConfig


# ---------------------------------------------------------------------------
# Named gate configs (CAT-7, master plan R7's "two instruments, two jobs").
#
# Every promotion decision needs one of exactly two SPRT bars, picked
# up-front and recorded verbatim in the run artifact -- never a single
# mutable elo0/elo1 flag whose value has to be inferred after the fact:
#   - FLYWHEEL_GATE_CONFIG: the day-to-day producer/flywheel gate. Tight
#     elo0=-10/elo1=+15 gap (candidate must not be a clear regression, not
#     necessarily a big jump) at a smaller game budget -- this is the gate
#     used for every ordinary promotion decision going forward.
#   - CERTIFICATION_GATE_CONFIG: the pre-existing wide elo0=0/elo1=+30 gap,
#     reserved for discrete "gen-N" turn announcements (public capability
#     claims). NOT deleted -- just no longer the default for routine
#     promotions. base_games/max_games (1000/3000) match the historical
#     promotion_gate_runner.py defaults (--games-per-leg 1000, 1x/2x/3x
#     extension tiers) so certification-config gates reproduce exactly the
#     game budget every gen-N announcement to date was measured against.
@dataclass(frozen=True)
class SprtGateConfig:
    # The named SPRT-bar config (elo0/elo1/alpha/beta/budget). Distinct from
    # catan_zero.rl.pipeline_configs.GateConfig (the CAT-66 typed CLI config used
    # for config-hash provenance in main()); this used to be named GateConfig too
    # and shadowed that import, breaking `GateConfig.from_namespace(...)` in the
    # CLI (CAT-119). Keep the names distinct.
    name: str
    elo0: float
    elo1: float
    alpha: float
    beta: float
    n_sims: int
    base_games: int
    max_games: int


FLYWHEEL_GATE_CONFIG = SprtGateConfig(
    name="flywheel", elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05, n_sims=16, base_games=300, max_games=600
)

CERTIFICATION_GATE_CONFIG = SprtGateConfig(
    name="certification", elo0=0.0, elo1=30.0, alpha=0.05, beta=0.05, n_sims=16, base_games=1000, max_games=3000
)

GATE_CONFIGS: dict[str, SprtGateConfig] = {
    FLYWHEEL_GATE_CONFIG.name: FLYWHEEL_GATE_CONFIG,
    CERTIFICATION_GATE_CONFIG.name: CERTIFICATION_GATE_CONFIG,
}


def resolve_gate_config(
    gate_config_name: str,
    *,
    elo0: float | None = None,
    elo1: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> tuple[SprtGateConfig, dict[str, Any]]:
    """Resolve the effective elo0/elo1/alpha/beta for a gate run from a named
    GATE_CONFIGS entry. Any explicitly-passed elo0/elo1/alpha/beta overrides
    that individual field (for one-off diagnostic runs) without losing the
    base config's name for the fields that were NOT overridden -- callers
    should record the returned params dict verbatim in run artifacts so
    promotion provenance is recorded, not inferred."""
    if gate_config_name not in GATE_CONFIGS:
        raise SystemExit(f"unknown --gate-config {gate_config_name!r}; choices: {sorted(GATE_CONFIGS)}")
    config = GATE_CONFIGS[gate_config_name]
    params = {
        "gate_config": config.name,
        "elo0": config.elo0 if elo0 is None else float(elo0),
        "elo1": config.elo1 if elo1 is None else float(elo1),
        "alpha": config.alpha if alpha is None else float(alpha),
        "beta": config.beta if beta is None else float(beta),
        "n_sims": config.n_sims,
        "base_games": config.base_games,
        "max_games": config.max_games,
    }
    return config, params


def elo_to_score(elo: float) -> float:
    """Expected score (win probability) for an Elo difference, logistic model."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    score = min(max(float(score), 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10((1.0 / score) - 1.0)


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Wald decision boundaries for the cumulative log-likelihood ratio."""
    if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0):
        raise ValueError(f"alpha/beta must be in (0, 1); got alpha={alpha}, beta={beta}")
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def game_llr_increment(outcome: bool, *, p0: float, p1: float) -> float:
    """LLR contribution of a single win/loss game under H1 vs H0."""
    return math.log(p1 / p0) if outcome else math.log((1.0 - p1) / (1.0 - p0))


def llr_trajectory(outcomes: Sequence[bool], *, elo0: float, elo1: float) -> list[float]:
    """Cumulative LLR after each game, in game order."""
    p0 = elo_to_score(elo0)
    p1 = elo_to_score(elo1)
    trajectory: list[float] = []
    running = 0.0
    for outcome in outcomes:
        running += game_llr_increment(bool(outcome), p0=p0, p1=p1)
        trajectory.append(running)
    return trajectory


def llr_from_counts(wins: int, losses: int, *, elo0: float, elo1: float) -> float:
    p0 = elo_to_score(elo0)
    p1 = elo_to_score(elo1)
    return wins * math.log(p1 / p0) + losses * math.log((1.0 - p1) / (1.0 - p0))


def sprt_decision(llr: float, *, alpha: float = 0.05, beta: float = 0.05) -> str:
    lower, upper = sprt_bounds(alpha, beta)
    if llr >= upper:
        return "H1"
    if llr <= lower:
        return "H0"
    return "continue"


def evaluate_sprt(
    outcomes: Sequence[bool] | None = None,
    *,
    wins: int | None = None,
    losses: int | None = None,
    elo0: float = 0.0,
    elo1: float = 5.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict[str, Any]:
    """Run the SPRT over either an ordered list of per-game outcomes (True =
    candidate won that game) or aggregate win/loss counts. Returns the LLR
    trajectory (per-game if outcomes were given, else a single terminal
    value), the decision boundaries, and the H0/H1/continue decision."""
    lower, upper = sprt_bounds(alpha, beta)
    if outcomes is not None:
        outcomes = [bool(outcome) for outcome in outcomes]
        trajectory = llr_trajectory(outcomes, elo0=elo0, elo1=elo1)
        llr = trajectory[-1] if trajectory else 0.0
        games = len(outcomes)
        wins_total = sum(1 for outcome in outcomes if outcome)
    else:
        if wins is None or losses is None:
            raise ValueError("evaluate_sprt requires either `outcomes` or `wins` and `losses`")
        wins_total = int(wins)
        losses = int(losses)
        games = wins_total + losses
        llr = llr_from_counts(wins_total, losses, elo0=elo0, elo1=elo1)
        trajectory = [llr]
    return {
        "elo0": float(elo0),
        "elo1": float(elo1),
        "alpha": float(alpha),
        "beta": float(beta),
        "lower_bound": lower,
        "upper_bound": upper,
        "games": games,
        "wins": wins_total,
        "losses": games - wins_total,
        "llr": llr,
        "llr_trajectory": trajectory,
        "decision": sprt_decision(llr, alpha=alpha, beta=beta),
    }


def _load_paired_outcomes(
    candidate_path: Path, baseline_path: Path, opponent: str
) -> tuple[list[bool], int]:
    """Load per-game candidate-vs-baseline outcomes for one opponent from a
    pair of tools/evaluate_scoreboard.py --paired-seeds reports.

    A "win" here means the candidate report's win on a given game index and
    the baseline report's loss on that same index (discordant pairs only are
    informative for a paired comparison); games where both won or both lost
    are dropped since they carry no information about which is stronger --
    exactly the McNemar convention used by tools/compare_scoreboards.py.

    FIX (adversarial review, truncation-as-loss bias): a game_outcomes entry
    is None when that game was truncated (no winner). Coercing it with
    bool() would silently count it as a loss and bias the LLR against
    whichever side truncates more often. Any pair where EITHER side is None
    is excluded (it carries no win/loss information); the excluded count is
    returned alongside the paired outcomes so callers can report it.
    """
    candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not candidate_data.get("paired_seeds") or not baseline_data.get("paired_seeds"):
        raise SystemExit(
            "sprt_gate.py --candidate-scoreboard/--baseline-scoreboard requires both "
            "reports to have been generated with evaluate_scoreboard.py --paired-seeds"
        )
    if candidate_data.get("seed") != baseline_data.get("seed"):
        raise SystemExit(
            "candidate/baseline scoreboards used different --seed values; "
            "their game_outcomes are not aligned game-for-game"
        )

    def _find(data: dict[str, Any]) -> dict[str, Any]:
        for result in data.get("results", []):
            if str(result.get("opponent")) == opponent:
                return result
        raise SystemExit(f"opponent {opponent!r} not found in {data.get('candidate')!r} report")

    candidate_result = _find(candidate_data)
    baseline_result = _find(baseline_data)
    if candidate_result.get("leg_seed") != baseline_result.get("leg_seed"):
        raise SystemExit(f"leg_seed mismatch for opponent {opponent!r}; reports are not paired")
    candidate_outcomes = candidate_result.get("game_outcomes")
    baseline_outcomes = baseline_result.get("game_outcomes")
    if not isinstance(candidate_outcomes, list) or not isinstance(baseline_outcomes, list):
        raise SystemExit(
            f"opponent {opponent!r} is missing game_outcomes; regenerate scoreboards "
            "with the updated evaluate_scoreboard.py"
        )
    if len(candidate_outcomes) != len(baseline_outcomes):
        raise SystemExit(f"game_outcomes length mismatch for opponent {opponent!r}")
    paired: list[bool] = []
    truncated_excluded = 0
    for candidate_won, baseline_won in zip(candidate_outcomes, baseline_outcomes):
        if candidate_won is None or baseline_won is None:
            truncated_excluded += 1
            continue  # truncated game on at least one side: no result to pair
        if bool(candidate_won) == bool(baseline_won):
            continue  # concordant pair: no information about relative strength
        paired.append(bool(candidate_won))
    return paired, truncated_excluded


# ---------------------------------------------------------------------------
# Pentanomial (trinomial, no-draw) generalized SPRT.
#
# DERIVATION.
# A color-swapped pair plays the SAME seed twice with the candidate (search)
# on each color once, cancelling first-move/color bias. Because Catan has no
# draws, the candidate's SCORE over a pair -- (games won)/2 -- can only be
# 0 (lost both = "LL"), 1/2 (split: won one, lost the other) or 1 (won both
# = "WW"). So a pair is a draw from a TRINOMIAL over the value set
# a = (0, 1/2, 1) with probabilities (p_LL, p_split, p_WW). The per-GAME win
# probability is exactly the pair's expected score:
#     mu == E[score] == 0*p_LL + 1/2*p_split + 1*p_WW == p_WW + p_split/2 == p.
# We test the same hypotheses as the game-level Bernoulli SPRT, expressed on
# the pair-score mean:
#     H0: mu == s0 == elo_to_score(elo0),   H1: mu == s1 == elo_to_score(elo1).
#
# H0 and H1 each fix only the MEAN of the trinomial, not its shape, so they
# are composite and we use the GENERALIZED SPRT (GSPRT). Michel Van den
# Bergh's practical GSPRT (the form fishtest / fastchess deploy for the
# pentanomial) plugs the sample mean and sample variance of the per-pair
# scores into the normal-approximation LLR. Writing mu_hat for the sample
# mean of the pair scores and sigma2_hat for their (population) sample
# variance, the sample mean is asymptotically Normal(mu, sigma2/N), so
#     LLR = log N(mu_hat; s1, sigma2_hat/N) / N(mu_hat; s0, sigma2_hat/N)
#         = N/(2*sigma2_hat) * (s1 - s0) * (2*mu_hat - s0 - s1).
# Decision boundaries are the same Wald bounds as the Bernoulli SPRT
# (sprt_bounds), and the GSPRT is asymptotically valid with them.
#
# WHY THIS BEATS THE CONCORDANT-ONLY RULE. The incumbent rule maps each
# color-swapped pair to WW->win / LL->loss and DISCARDS every split, then
# runs the plain Bernoulli SPRT on the decisive pairs. That throws away (a)
# the split pairs entirely and (b) the variance reduction from pairing. The
# GSPRT keeps every pair: splits sit at the mean (0.5) and so they SHRINK
# sigma2_hat, and pairing cancels the shared color/seed variance -- both make
# sigma2_hat smaller than the maximal Bernoulli variance the concordant rule
# implicitly assumes, so the same games yield a larger |LLR| per pair and
# resolve faster at the same error rates (verified in
# tools/pentanomial_power_sim.py).
#
# TERMINAL VALUES ARE NOT ADDITIVE.  This is a generalized statistic whose
# nuisance variance is estimated from the complete sample (and regularized by
# one shared pseudo-sample below), not a sum of per-pair log-probability
# increments.  Therefore a continuation made of several fresh seed cohorts
# must concatenate/replay the raw pair outcomes and call this function once.
# Summing the terminal ``llr`` fields from separately finalized cohorts would
# estimate the variance independently and apply the prior once per cohort;
# it is not the LLR of their union.  ``a1_evaluation_pool.py`` deliberately
# recomputes from the retained raw games for this reason.
#
# WHY THE MEAN/VARIANCE FORM (not the exact empirical-likelihood MLE tilt).
# The exact multinomial GSPRT (constrained-MLE "tilt" of the empirical pdf)
# is degenerate on the small samples our gates actually collect: when the
# observed support does not bracket a hypothesis mean (e.g. all decided pairs
# are splits, or counts like LL=1/split=20/WW=0 with s0=0.5 sitting on the
# hull boundary) a category probability is driven to ~0 and the LLR explodes
# to nonsense (|LLR| in the tens after a handful of pairs). The mean/variance
# GSPRT agrees with the exact tilt to second order but stays finite and
# well-behaved, which is what matters at gate budgets of tens-to-hundreds of
# pairs.
#
# REGULARIZATION. With very few pairs sigma2_hat can be an unreliable (or
# zero, if all pairs are identical) estimate of the true variance, which
# would let a lucky run cross a boundary prematurely. We add PRIOR_PAIRS of
# mean-neutral, variance-maximizing pseudo-mass -- half a WW and half an LL
# per prior pair -- before computing mu_hat and sigma2_hat. This keeps the
# prior mean at 0.5 (it never biases toward H0 or H1), guarantees
# sigma2_hat > 0, and is deliberately CONSERVATIVE (it inflates the variance,
# so it can only slow a decision, never manufacture one). It washes out after
# a few dozen real pairs. The default (2 prior pairs) is validated to hold
# the nominal type-I rate across split rates in tools/pentanomial_power_sim.py.

PAIR_VALUES: tuple[float, float, float] = (0.0, 0.5, 1.0)  # LL, split, WW
_PAIR_LABELS: tuple[str, str, str] = ("ll", "split", "ww")
PENTANOMIAL_PRIOR_PAIRS: float = 2.0


def pentanomial_llr(
    counts: Sequence[float],
    *,
    values: Sequence[float] = PAIR_VALUES,
    s0: float,
    s1: float,
    prior_pairs: float = PENTANOMIAL_PRIOR_PAIRS,
) -> float:
    """Van den Bergh mean/variance GSPRT log-likelihood ratio.

    `counts` are per-category observed pair counts aligned with `values`
    (default the (0, 1/2, 1) no-draw pair value set). `s0`/`s1` are the
    per-game (== per-pair-mean) win probabilities under H0/H1. Adds
    `prior_pairs` of mean-neutral variance-maximizing pseudo-mass (split
    between the extreme categories) for stability at small N, then returns
    LLR = N/(2*var) * (s1 - s0) * (2*mean - s0 - s1)."""
    counts = [float(c) for c in counts]
    values = [float(v) for v in values]
    reg = list(counts)
    if prior_pairs > 0.0 and len(values) >= 2:
        lo_idx = min(range(len(values)), key=lambda i: values[i])
        hi_idx = max(range(len(values)), key=lambda i: values[i])
        reg[lo_idx] += prior_pairs / 2.0
        reg[hi_idx] += prior_pairs / 2.0
    total = sum(reg)
    if total <= 0.0:
        return 0.0
    mean = sum(c * v for c, v in zip(reg, values)) / total
    var = sum(c * (v - mean) ** 2 for c, v in zip(reg, values)) / total
    if var <= 0.0:
        # Only reachable with prior_pairs == 0 and all mass on one category;
        # fall back to a tiny variance so the LLR keeps the correct sign.
        var = 1e-12
    return total / (2.0 * var) * (s1 - s0) * (2.0 * mean - s0 - s1)


def pair_score_counts(pair_scores: Sequence[float]) -> tuple[int, int, int]:
    """Bin a list of pair scores (each in {0.0, 0.5, 1.0}) into (LL, split, WW)
    counts. Scores are matched to the nearest of the three legal values."""
    counts = [0, 0, 0]
    for score in pair_scores:
        idx = min(range(len(PAIR_VALUES)), key=lambda i: abs(PAIR_VALUES[i] - float(score)))
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def evaluate_pentanomial_sprt(
    pair_scores: Sequence[float] | None = None,
    *,
    counts: Sequence[float] | None = None,
    elo0: float = 0.0,
    elo1: float = 30.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict[str, Any]:
    """Trinomial (no-draw pentanomial) GSPRT over color-swapped pair outcomes.

    Pass either `pair_scores` (each 0.0=LL / 0.5=split / 1.0=WW) or aggregate
    `counts` = (n_LL, n_split, n_WW). Tests H0: per-game win prob == elo0 vs
    H1: == elo1 using the pair trinomial, so split pairs contribute evidence
    instead of being discarded. Returns category counts, the observed mean
    pair score (== estimated per-game win rate), the GSPRT LLR, the Wald
    bounds, and the H0/H1/continue decision -- shape-compatible with
    evaluate_sprt() so callers can report them side by side."""
    if counts is None:
        if pair_scores is None:
            raise ValueError("evaluate_pentanomial_sprt requires either `pair_scores` or `counts`")
        counts = pair_score_counts(pair_scores)
    counts = [int(c) for c in counts]
    n_ll, n_split, n_ww = counts
    total = n_ll + n_split + n_ww
    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)
    lower, upper = sprt_bounds(alpha, beta)
    llr = pentanomial_llr(counts, s0=s0, s1=s1)
    mean_score = ((0.0 * n_ll) + (0.5 * n_split) + (1.0 * n_ww)) / total if total else None
    return {
        "model": "pentanomial",
        "elo0": float(elo0),
        "elo1": float(elo1),
        "s0": s0,
        "s1": s1,
        "alpha": float(alpha),
        "beta": float(beta),
        "lower_bound": lower,
        "upper_bound": upper,
        "pairs": total,
        "ll_pairs": n_ll,
        "split_pairs": n_split,
        "ww_pairs": n_ww,
        "mean_pair_score": mean_score,
        "llr": llr,
        "decision": sprt_decision(llr, alpha=alpha, beta=beta),
    }


# ---------------------------------------------------------------------------
# R9 timeout rule (master plan / expert review R9).
#
# The SPRT (Bernoulli or pentanomial) is a strict three-way test: H1
# (promote), H0 (reject), or continue. But a flywheel gate that exhausts its
# largest extension tier still "continue" is not necessarily useless --
# fishtest/GSPRT theory gives us the same sample mean/variance used for the
# LLR, and that normal approximation doubles as a posterior over the
# per-game win probability mu (flat prior). R9 uses that posterior for a
# one-sided question the two-sided LLR doesn't answer: is the candidate
# probably not a regression, even if it isn't (yet, or ever) a proven
# improvement? If the posterior median Elo is positive AND P(Elo < elo0) is
# small, the candidate is safe enough to seed the NEXT round of self-play
# generation -- a "canary_promote" verdict. This is explicitly NOT a
# promotion: it must never be read as (or feed) a public gen-N announcement,
# only the generator's own next iteration.


def _normal_cdf(x: float, *, mean: float, std: float) -> float:
    """Standard normal CDF via math.erf (stdlib-only, no scipy dependency)."""
    if std <= 0.0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))


def score_posterior_stats(
    counts: Sequence[float],
    values: Sequence[float],
    *,
    prior_mass: float = PENTANOMIAL_PRIOR_PAIRS,
) -> dict[str, float]:
    """Normal-approximation posterior over the per-game win-probability mean
    mu, from categorical counts -- the same sample mean/variance
    pentanomial_llr computes, reported directly instead of turned into a
    likelihood ratio. Shares its mean-neutral, variance-maximizing
    `prior_mass` regularization (split between the two extreme categories in
    `values`) so it stays well-behaved at small N. Works for two-outcome
    Bernoulli win/loss counts (values=(0.0, 1.0)) as well as three-outcome
    pentanomial LL/split/WW counts (values=PAIR_VALUES)."""
    counts = [float(c) for c in counts]
    values = [float(v) for v in values]
    reg = list(counts)
    if prior_mass > 0.0 and len(values) >= 2:
        lo_idx = min(range(len(values)), key=lambda i: values[i])
        hi_idx = max(range(len(values)), key=lambda i: values[i])
        reg[lo_idx] += prior_mass / 2.0
        reg[hi_idx] += prior_mass / 2.0
    total = sum(reg)
    if total <= 0.0:
        return {"mu_hat": 0.5, "se": float("inf"), "median_elo": 0.0, "n_effective": 0.0}
    mean = sum(c * v for c, v in zip(reg, values)) / total
    var = sum(c * (v - mean) ** 2 for c, v in zip(reg, values)) / total
    se = math.sqrt(var / total)
    return {"mu_hat": mean, "se": se, "median_elo": score_to_elo(mean), "n_effective": total}


def r9_timeout_verdict(
    counts: Sequence[float],
    values: Sequence[float] = PAIR_VALUES,
    *,
    elo_floor: float = -10.0,
    max_prob_below_floor: float = 0.05,
    prior_mass: float = PENTANOMIAL_PRIOR_PAIRS,
) -> dict[str, Any]:
    """R9 timeout rule: when a gate's game cap is reached with the SPRT
    still 'continue' (neither H0 nor H1 resolved), a candidate that is
    nonetheless probably not harmful can still seed the next round of
    self-play generation. Eligible ("canary_promote") iff the posterior
    median Elo is positive AND P(Elo < elo_floor) is at or below
    `max_prob_below_floor`.

    The caller is responsible for only invoking this once the gate has
    genuinely exhausted its extension schedule (this function has no notion
    of tiers or a game cap) and for keeping the resulting verdict
    generator-only -- never a public gen-N promotion."""
    stats = score_posterior_stats(counts, values, prior_mass=prior_mass)
    floor_score = elo_to_score(elo_floor)
    prob_below_floor = _normal_cdf(floor_score, mean=stats["mu_hat"], std=stats["se"])
    eligible = stats["median_elo"] > 0.0 and prob_below_floor <= max_prob_below_floor
    return {
        **stats,
        "elo_floor": elo_floor,
        "prob_elo_below_floor": prob_below_floor,
        "max_prob_below_floor": max_prob_below_floor,
        "canary_eligible": eligible,
        "verdict": "canary_promote" if eligible else "continue",
    }


def pair_scores_from_h2h_games(games: Sequence[dict[str, Any]]) -> tuple[list[float], dict[str, int]]:
    """Reduce raw gumbel_search_vs_raw_h2h.py game records to per-pair scores
    for the pentanomial test: group the two color-swapped orientations by
    pair_id and map WW->1.0, split->0.5, LL->0.0. A pair missing an
    orientation or with a truncated game (search_won is None) is excluded.
    Unlike the concordant-only rule, splits are KEPT (as 0.5)."""
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        by_pair.setdefault(int(game["pair_id"]), []).append(game)
    scores: list[float] = []
    diagnostics = {"ww_pairs": 0, "split_pairs": 0, "ll_pairs": 0, "incomplete_pairs": 0}
    for pair_games in by_pair.values():
        if len(pair_games) != 2 or any(g.get("search_won") is None for g in pair_games):
            diagnostics["incomplete_pairs"] += 1
            continue
        wins = sum(1 for g in pair_games if bool(g["search_won"]))
        if wins == 2:
            scores.append(1.0)
            diagnostics["ww_pairs"] += 1
        elif wins == 0:
            scores.append(0.0)
            diagnostics["ll_pairs"] += 1
        else:
            scores.append(0.5)
            diagnostics["split_pairs"] += 1
    return scores, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "fishtest-style SPRT: decide promote (H1) / reject (H0) / continue "
            "collecting games, from paired win/loss game results."
        ),
    )
    parser.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default=None,
        help=(
            "Named gate config (flywheel: elo0=-10/elo1=15; certification: "
            "elo0=0/elo1=30) providing defaults for --elo0/--elo1/--alpha/"
            "--beta. Explicit flags override individual fields. Omit for "
            "this CLI's legacy standalone defaults (elo0=0/elo1=5)."
        ),
    )
    parser.add_argument("--elo0", type=float, default=None)
    parser.add_argument(
        "--elo1",
        type=float,
        default=None,
        help=(
            "Alternative-hypothesis Elo gap. With no --gate-config, defaults to "
            "5.0 (fishtest's tight small-effect setting, needs thousands of "
            "paired games to resolve). For standard >=55%%-win-rate promotion "
            "gates (~+35 Elo), pass --elo1 30 instead, or --gate-config "
            "certification -- it resolves in the low hundreds to ~1000 paired "
            "games, matching our gate budgets."
        ),
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--wins", type=int, help="Aggregate win count (paired with --losses).")
    parser.add_argument("--losses", type=int, help="Aggregate loss count (paired with --wins).")
    parser.add_argument(
        "--outcomes-json",
        help="Path to a JSON file containing a list of booleans (True = candidate won).",
    )
    parser.add_argument(
        "--candidate-scoreboard",
        help="Paired --paired-seeds evaluate_scoreboard.py report for the candidate.",
    )
    parser.add_argument(
        "--baseline-scoreboard",
        help="Paired --paired-seeds evaluate_scoreboard.py report for the baseline.",
    )
    parser.add_argument("--opponent", help="Opponent name to pull paired outcomes for.")
    parser.add_argument(
        "--h2h-summary",
        help=(
            "Path to a gumbel_search_vs_raw_h2h.py output JSON. Runs the "
            "pentanomial GSPRT over its color-swapped pairs and prints it "
            "next to the concordant-only Bernoulli SPRT for comparison."
        ),
    )
    parser.add_argument(
        "--pair-scores-json",
        help="Path to a JSON list of pair scores (each 0.0/0.5/1.0) for the pentanomial GSPRT.",
    )
    parser.add_argument("--out", help="Optional JSON output path.")
    add_config_flags(parser, default_purpose="sprt_gate")
    args = parser.parse_args()

    # CAT-7: resolve the named gate config (elo0/elo1/alpha/beta) FIRST, so the
    # CAT-66 typed config-hash below captures the actually-used values.
    if args.gate_config is not None:
        _gate_cfg, gate_params = resolve_gate_config(
            args.gate_config, elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta
        )
    else:
        # No named config: preserve this CLI's historical standalone
        # defaults (elo0=0/elo1=5/alpha=beta=0.05) rather than silently
        # switching every bare invocation to the flywheel config.
        gate_params = {
            "gate_config": None,
            "elo0": 0.0 if args.elo0 is None else float(args.elo0),
            "elo1": 5.0 if args.elo1 is None else float(args.elo1),
            "alpha": 0.05 if args.alpha is None else float(args.alpha),
            "beta": 0.05 if args.beta is None else float(args.beta),
        }
    args.elo0, args.elo1, args.alpha, args.beta = (
        gate_params["elo0"],
        gate_params["elo1"],
        gate_params["alpha"],
        gate_params["beta"],
    )

    # CAT-66 typed config + config-hash. ``test_kind`` records which branch runs
    # (selected implicitly by which input flag is present), and the gate echoes
    # the masking regime (public_observation) from the h2h summary it consumes so
    # a gate result can be checked for consistency against the generation/eval
    # regime that produced its inputs. No-op to the decision itself.
    _gate_test_kind = "pentanomial" if (args.h2h_summary or args.pair_scores_json) else "bernoulli"
    _gate_public_obs: bool | None = None
    if args.h2h_summary:
        try:
            _gate_public_obs = json.loads(Path(args.h2h_summary).read_text(encoding="utf-8")).get(
                "public_observation"
            )
        except (OSError, ValueError):
            _gate_public_obs = None
    gate_config = resolve_config(
        args,
        lambda a: GateConfig.from_namespace(
            a, test_kind=_gate_test_kind, generation_public_observation=_gate_public_obs
        ),
        parser=parser,
    )
    gate_config_hash = gate_config.config_hash()

    # Pentanomial / trinomial GSPRT path (task #1): consume color-swapped
    # pairs directly and report it alongside the concordant-only rule.
    if args.h2h_summary or args.pair_scores_json:
        if args.h2h_summary:
            summary = json.loads(Path(args.h2h_summary).read_text(encoding="utf-8"))
            games = summary.get("games", [])
            pair_scores, pair_diag = pair_scores_from_h2h_games(games)
        else:
            pair_scores = [float(s) for s in json.loads(Path(args.pair_scores_json).read_text(encoding="utf-8"))]
            n_ll, n_split, n_ww = pair_score_counts(pair_scores)
            pair_diag = {"ww_pairs": n_ww, "split_pairs": n_split, "ll_pairs": n_ll, "incomplete_pairs": 0}
        pentanomial = evaluate_pentanomial_sprt(
            pair_scores, elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta
        )
        # Concordant-only Bernoulli SPRT on the same pairs (WW->win, LL->loss,
        # splits discarded) -- the incumbent rule, for a side-by-side view.
        concordant = [True] * pair_diag["ww_pairs"] + [False] * pair_diag["ll_pairs"]
        concordant_sprt = evaluate_sprt(
            outcomes=concordant, elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta
        )
        report = {
            "gate_config": gate_params["gate_config"],
            "pair_diagnostics": pair_diag,
            "pentanomial_sprt": pentanomial,
            "concordant_sprt": concordant_sprt,
            "decisions": {
                "pentanomial": pentanomial["decision"],
                "concordant": concordant_sprt["decision"],
                "changed": pentanomial["decision"] != concordant_sprt["decision"],
            },
            "config_hash": gate_config_hash,
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        print(text)
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
        return

    outcomes = None
    truncated_excluded = 0
    if args.outcomes_json:
        outcomes = json.loads(Path(args.outcomes_json).read_text(encoding="utf-8"))
    elif args.candidate_scoreboard and args.baseline_scoreboard:
        if not args.opponent:
            raise SystemExit("--opponent is required with --candidate/--baseline-scoreboard")
        outcomes, truncated_excluded = _load_paired_outcomes(
            Path(args.candidate_scoreboard), Path(args.baseline_scoreboard), args.opponent
        )

    if outcomes is not None:
        report = evaluate_sprt(outcomes, elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta)
        report["truncated_pairs_excluded"] = truncated_excluded
    elif args.wins is not None and args.losses is not None:
        report = evaluate_sprt(
            wins=args.wins, losses=args.losses, elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta
        )
    else:
        raise SystemExit(
            "provide one of: --outcomes-json, --wins/--losses, or "
            "--candidate-scoreboard/--baseline-scoreboard/--opponent"
        )
    report["gate_config"] = gate_params["gate_config"]

    report["config_hash"] = gate_config_hash
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
