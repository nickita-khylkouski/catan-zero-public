from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sprt_gate import (
    CERTIFICATION_GATE_CONFIG,
    FLYWHEEL_GATE_CONFIG,
    GATE_CONFIGS,
    _load_paired_outcomes,
    elo_to_score,
    evaluate_pentanomial_sprt,
    evaluate_sprt,
    game_llr_increment,
    llr_from_counts,
    llr_trajectory,
    pair_score_counts,
    pair_scores_from_h2h_games,
    pentanomial_llr,
    r9_timeout_verdict,
    resolve_gate_config,
    score_posterior_stats,
    score_to_elo,
    sprt_bounds,
    sprt_decision,
)


def test_elo_to_score_and_back_round_trip() -> None:
    for elo in (-200.0, -5.0, 0.0, 5.0, 50.0, 400.0):
        assert score_to_elo(elo_to_score(elo)) == pytest.approx(elo, abs=1e-6)


def test_elo_to_score_zero_is_even() -> None:
    assert elo_to_score(0.0) == pytest.approx(0.5)


def test_sprt_bounds_symmetric_alpha_beta() -> None:
    lower, upper = sprt_bounds(0.05, 0.05)
    assert upper == pytest.approx(-lower)
    assert upper > 0.0


def test_sprt_bounds_rejects_invalid_alpha_beta() -> None:
    with pytest.raises(ValueError):
        sprt_bounds(0.0, 0.05)
    with pytest.raises(ValueError):
        sprt_bounds(0.05, 1.0)


def test_game_llr_increment_win_vs_loss_have_opposite_signs_above_50pct() -> None:
    # elo1 > elo0 means H1 predicts a higher win rate than H0; a win should
    # push the LLR toward H1 (positive) and a loss toward H0 (negative).
    win = game_llr_increment(True, p0=0.5, p1=0.55)
    loss = game_llr_increment(False, p0=0.5, p1=0.55)
    assert win > 0.0
    assert loss < 0.0


def test_llr_trajectory_is_cumulative_and_matches_from_counts() -> None:
    outcomes = [True, True, False, True, False, False, True]
    trajectory = llr_trajectory(outcomes, elo0=0.0, elo1=5.0)
    assert len(trajectory) == len(outcomes)
    wins = sum(outcomes)
    losses = len(outcomes) - wins
    assert trajectory[-1] == pytest.approx(llr_from_counts(wins, losses, elo0=0.0, elo1=5.0))
    # Cumulative trajectory must be monotonically built from increments, i.e.
    # trajectory[i] - trajectory[i-1] equals that single game's increment.
    running = 0.0
    for value, outcome in zip(trajectory, outcomes):
        running += game_llr_increment(outcome, p0=elo_to_score(0.0), p1=elo_to_score(5.0))
        assert value == pytest.approx(running)


def test_sprt_decision_boundaries() -> None:
    lower, upper = sprt_bounds(0.05, 0.05)
    assert sprt_decision(upper + 0.001, alpha=0.05, beta=0.05) == "H1"
    assert sprt_decision(lower - 0.001, alpha=0.05, beta=0.05) == "H0"
    assert sprt_decision(0.0, alpha=0.05, beta=0.05) == "continue"


def test_evaluate_sprt_clear_h1_case_coarse_elo1() -> None:
    # NOTE: elo1=5 (the fishtest STC-style default) is a genuinely small
    # effect size and 200 games at 60% is nowhere near enough evidence to
    # resolve it at alpha=beta=0.05 (it lands in "continue" -- see the next
    # test). A larger elo1 demonstrates the H1-acceptance mechanics cleanly
    # with a small, easy-to-verify sample.
    report = evaluate_sprt(wins=120, losses=80, elo0=0.0, elo1=50.0, alpha=0.05, beta=0.05)
    assert report["decision"] == "H1"
    assert report["llr"] >= report["upper_bound"]
    assert report["games"] == 200
    assert report["wins"] == 120
    assert report["losses"] == 80


def test_evaluate_sprt_default_elo_config_needs_larger_sample() -> None:
    # 200 games at 60% does NOT resolve the tight default elo0=0/elo1=5 gate.
    small = evaluate_sprt(wins=120, losses=80, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05)
    assert small["decision"] == "continue"
    # But the same 60% win rate over a realistically large paired sample does.
    large = evaluate_sprt(wins=3000, losses=2000, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05)
    assert large["decision"] == "H1"


def test_evaluate_sprt_promotion_gate_config_resolves_within_gate_budget() -> None:
    # Our promotion gates target >=55% win rate (~+35 Elo). --elo1 30 is the
    # recommended config (see sprt_gate.py's module docstring / --elo1 help):
    # it should resolve a genuine 55% candidate within a realistic gate
    # budget (low hundreds to ~1000 paired games), unlike the tight elo1=5
    # fishtest default which needs thousands.
    at_gate_budget = evaluate_sprt(wins=330, losses=270, elo0=0.0, elo1=30.0, alpha=0.05, beta=0.05)
    assert at_gate_budget["games"] == 600
    assert at_gate_budget["decision"] == "H1"


def test_evaluate_sprt_accepts_h0_for_a_losing_record() -> None:
    report = evaluate_sprt(wins=200, losses=800, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05)
    assert report["decision"] == "H0"
    assert report["llr"] <= report["lower_bound"]


def test_evaluate_sprt_from_outcomes_matches_from_counts() -> None:
    outcomes = [True] * 60 + [False] * 40
    from_outcomes = evaluate_sprt(outcomes, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05)
    from_counts = evaluate_sprt(wins=60, losses=40, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05)
    assert from_outcomes["llr"] == pytest.approx(from_counts["llr"])
    assert from_outcomes["decision"] == from_counts["decision"]
    assert len(from_outcomes["llr_trajectory"]) == 100


def test_evaluate_sprt_requires_outcomes_or_counts() -> None:
    with pytest.raises(ValueError):
        evaluate_sprt()


def _write_scoreboard(path: Path, *, seed: int, opponent: str, leg_seed: int, outcomes: list[bool]) -> None:
    wins = sum(1 for outcome in outcomes if outcome)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate": str(path),
                "seed": seed,
                "paired_seeds": True,
                "results": [
                    {
                        "opponent": opponent,
                        "wins": wins,
                        "games": len(outcomes),
                        "win_rate": wins / len(outcomes),
                        "leg_seed": leg_seed,
                        "game_outcomes": outcomes,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_paired_outcomes_keeps_only_discordant_games(tmp_path: Path) -> None:
    # game 0: both win (concordant, dropped); game 1: candidate wins only;
    # game 2: baseline wins only; game 3: both lose (concordant, dropped).
    _write_scoreboard(
        tmp_path / "candidate.json",
        seed=7,
        opponent="catanatron_ab3",
        leg_seed=1007,
        outcomes=[True, True, False, False],
    )
    _write_scoreboard(
        tmp_path / "baseline.json",
        seed=7,
        opponent="catanatron_ab3",
        leg_seed=1007,
        outcomes=[True, False, True, False],
    )
    paired, truncated_excluded = _load_paired_outcomes(
        tmp_path / "candidate.json", tmp_path / "baseline.json", "catanatron_ab3"
    )
    assert paired == [True, False]
    assert truncated_excluded == 0


def test_load_paired_outcomes_rejects_mismatched_seed(tmp_path: Path) -> None:
    _write_scoreboard(
        tmp_path / "candidate.json", seed=7, opponent="catanatron_ab3", leg_seed=1007, outcomes=[True, False]
    )
    _write_scoreboard(
        tmp_path / "baseline.json", seed=8, opponent="catanatron_ab3", leg_seed=1007, outcomes=[True, False]
    )
    with pytest.raises(SystemExit):
        _load_paired_outcomes(tmp_path / "candidate.json", tmp_path / "baseline.json", "catanatron_ab3")


def test_load_paired_outcomes_excludes_any_pair_with_a_truncated_side(tmp_path: Path) -> None:
    # Regression test for the adversarial-review truncation-as-loss bug:
    # game 0: candidate truncated (None) while baseline won -- must be
    # EXCLUDED, not counted as a candidate loss. game 1: baseline truncated
    # while candidate won -- also excluded. game 2/3: normal discordant pair.
    _write_scoreboard(
        tmp_path / "candidate.json",
        seed=7,
        opponent="catanatron_ab3",
        leg_seed=1007,
        outcomes=[None, True, True, False],
    )
    _write_scoreboard(
        tmp_path / "baseline.json",
        seed=7,
        opponent="catanatron_ab3",
        leg_seed=1007,
        outcomes=[True, None, False, True],
    )
    paired, truncated_excluded = _load_paired_outcomes(
        tmp_path / "candidate.json", tmp_path / "baseline.json", "catanatron_ab3"
    )
    assert truncated_excluded == 2
    assert paired == [True, False]  # only the two non-truncated discordant games


# ---------------------------------------------------------------------------
# Pentanomial (trinomial, no-draw) GSPRT.

_S0 = elo_to_score(0.0)
_S1 = elo_to_score(30.0)


def test_pentanomial_llr_sign_ww_positive_ll_negative() -> None:
    # WW-heavy pairs are evidence the candidate is stronger (toward H1);
    # LL-heavy pairs are evidence it is weaker (toward H0).
    assert pentanomial_llr([2, 3, 40], s0=_S0, s1=_S1) > 0.0
    assert pentanomial_llr([40, 3, 2], s0=_S0, s1=_S1) < 0.0


def test_pentanomial_small_all_split_sample_favors_h0_without_resolving() -> None:
    # A few splits mean an observed win rate of exactly 0.5 (== s0), so the
    # LLR leans toward H0 -- but a small sample must NOT blow up or resolve
    # (the exact empirical-likelihood tilt DID blow up here: it drove a
    # category to ~0 and produced |LLR| in the tens; regression guard).
    llr = pentanomial_llr([0, 10, 0], s0=_S0, s1=_S1)
    assert llr < 0.0
    assert abs(llr) < sprt_bounds(0.05, 0.05)[1]  # nowhere near a decision

def test_pentanomial_many_pure_splits_resolve_h0() -> None:
    # A candidate that always splits color-swapped pairs is genuinely ~50%,
    # i.e. definitively NOT +30 Elo, so a large pure-split sample SHOULD
    # accept H0 (tight variance around 0.5 is strong evidence against H1).
    assert evaluate_pentanomial_sprt(counts=(0, 200, 0), elo0=0.0, elo1=30.0)["decision"] == "H0"


def test_pentanomial_boundary_counts_do_not_explode() -> None:
    # counts=(LL=1, split=20, WW=0): s0=0.5 sits on the observed hull edge --
    # the fragile exact tilt returned ~37 here. The robust GSPRT stays small.
    llr = pentanomial_llr([1, 20, 0], s0=_S0, s1=_S1)
    assert abs(llr) < 5.0


def test_pentanomial_mean_pair_score_equals_win_rate() -> None:
    report = evaluate_pentanomial_sprt(counts=(10, 20, 30), elo0=0.0, elo1=30.0)
    # mean = (0*10 + 0.5*20 + 1*30)/60 = 40/60
    assert report["mean_pair_score"] == pytest.approx(40.0 / 60.0)
    assert report["pairs"] == 60
    assert (report["ll_pairs"], report["split_pairs"], report["ww_pairs"]) == (10, 20, 30)


def test_evaluate_pentanomial_sprt_resolves_clear_h1() -> None:
    # A candidate winning ~2/3 of pairs outright is well past the +30 Elo bar
    # and should be accepted within a realistic pair budget.
    report = evaluate_pentanomial_sprt(counts=(20, 60, 220), elo0=0.0, elo1=30.0)
    assert report["decision"] == "H1"
    assert report["llr"] >= report["upper_bound"]


def test_evaluate_pentanomial_sprt_rejects_clear_h0() -> None:
    report = evaluate_pentanomial_sprt(counts=(220, 60, 20), elo0=0.0, elo1=30.0)
    assert report["decision"] == "H0"
    assert report["llr"] <= report["lower_bound"]


def test_evaluate_pentanomial_sprt_small_sample_is_conservative() -> None:
    # A handful of pairs must never resolve a ~4% effect size, no matter how
    # lopsided -- the conservative prior keeps small samples in "continue".
    report = evaluate_pentanomial_sprt(counts=(0, 0, 5), elo0=0.0, elo1=30.0)
    assert report["decision"] == "continue"


def test_pentanomial_extracts_more_than_concordant_from_same_pairs() -> None:
    # Same pairs, split-heavy: 60 WW, 30 LL, 300 splits. The concordant rule
    # sees only 60 wins / 30 losses (90 decisive) and cannot resolve; the
    # pentanomial uses all 390 pairs (splits shrink the variance) and does.
    counts = (30, 300, 60)  # (LL, split, WW)
    pent = evaluate_pentanomial_sprt(counts=counts, elo0=0.0, elo1=30.0)
    concordant = evaluate_sprt(outcomes=[True] * 60 + [False] * 30, elo0=0.0, elo1=30.0)
    assert pent["decision"] == "H1"
    assert concordant["decision"] == "continue"


def test_pentanomial_continuations_recompute_union_instead_of_summing_llrs() -> None:
    """Regression for sequential fleet continuations.

    The statistic estimates variance over its complete input and applies one
    regularizing prior, so independently finalized cohort LLRs are not
    increments.  These are the exact three continuation cohorts from the
    topology-gather audit: every cohort remains unresolved, while their raw
    union resolves H1.  The raw-union replay is authoritative.
    """
    cohorts = ((10, 38, 12), (19, 64, 37), (33, 144, 43))
    reports = [
        evaluate_pentanomial_sprt(counts=counts, elo0=-10.0, elo1=15.0)
        for counts in cohorts
    ]
    pooled_counts = tuple(sum(values) for values in zip(*cohorts))
    pooled = evaluate_pentanomial_sprt(
        counts=pooled_counts, elo0=-10.0, elo1=15.0
    )

    assert pooled_counts == (62, 246, 92)
    assert [report["decision"] for report in reports] == [
        "continue",
        "continue",
        "continue",
    ]
    assert sum(report["llr"] for report in reports) == pytest.approx(
        4.73476002203981
    )
    assert pooled["llr"] == pytest.approx(5.097807900340944)
    assert pooled["decision"] == "H1"
    assert pooled["llr"] != pytest.approx(
        sum(report["llr"] for report in reports)
    )


def test_evaluate_pentanomial_sprt_requires_pairs_or_counts() -> None:
    with pytest.raises(ValueError):
        evaluate_pentanomial_sprt()


def test_pair_score_counts_bins_to_nearest_legal_value() -> None:
    assert pair_score_counts([0.0, 0.5, 1.0, 1.0, 0.5]) == (1, 2, 2)


def _h2h_game(pair_id: int, search_won: bool | None) -> dict:
    return {"pair_id": pair_id, "search_won": search_won}


def test_pair_scores_from_h2h_games_keeps_splits_as_half() -> None:
    games = [
        _h2h_game(0, True), _h2h_game(0, True),    # WW -> 1.0
        _h2h_game(1, False), _h2h_game(1, False),  # LL -> 0.0
        _h2h_game(2, True), _h2h_game(2, False),   # split -> 0.5 (KEPT)
        _h2h_game(3, True), _h2h_game(3, None),     # incomplete -> excluded
    ]
    scores, diagnostics = pair_scores_from_h2h_games(games)
    assert sorted(scores) == [0.0, 0.5, 1.0]
    assert diagnostics == {"ww_pairs": 1, "split_pairs": 1, "ll_pairs": 1, "incomplete_pairs": 1}


# ---------------------------------------------------------------------------
# CAT-7: named gate configs (FLYWHEEL_GATE_CONFIG / CERTIFICATION_GATE_CONFIG).


def test_flywheel_gate_config_matches_cat7_spec() -> None:
    assert FLYWHEEL_GATE_CONFIG.name == "flywheel"
    assert FLYWHEEL_GATE_CONFIG.elo0 == -10.0
    assert FLYWHEEL_GATE_CONFIG.elo1 == 15.0
    assert FLYWHEEL_GATE_CONFIG.alpha == 0.05
    assert FLYWHEEL_GATE_CONFIG.beta == 0.05
    assert FLYWHEEL_GATE_CONFIG.n_sims == 16
    assert FLYWHEEL_GATE_CONFIG.base_games == 300
    assert FLYWHEEL_GATE_CONFIG.max_games == 600


def test_certification_gate_config_is_the_pre_cat7_wide_gap_gate() -> None:
    assert CERTIFICATION_GATE_CONFIG.name == "certification"
    assert CERTIFICATION_GATE_CONFIG.elo0 == 0.0
    assert CERTIFICATION_GATE_CONFIG.elo1 == 30.0
    assert CERTIFICATION_GATE_CONFIG.alpha == 0.05
    assert CERTIFICATION_GATE_CONFIG.beta == 0.05


def test_gate_configs_registry_has_exactly_the_two_named_configs() -> None:
    assert set(GATE_CONFIGS) == {"flywheel", "certification"}
    assert GATE_CONFIGS["flywheel"] is FLYWHEEL_GATE_CONFIG
    assert GATE_CONFIGS["certification"] is CERTIFICATION_GATE_CONFIG


def test_resolve_gate_config_returns_config_defaults_when_no_overrides() -> None:
    config, params = resolve_gate_config("flywheel")
    assert config is FLYWHEEL_GATE_CONFIG
    assert params == {
        "gate_config": "flywheel",
        "elo0": -10.0,
        "elo1": 15.0,
        "alpha": 0.05,
        "beta": 0.05,
        "n_sims": 16,
        "base_games": 300,
        "max_games": 600,
    }


def test_resolve_gate_config_explicit_override_wins_for_that_field_only() -> None:
    _config, params = resolve_gate_config("flywheel", elo1=99.0)
    assert params["gate_config"] == "flywheel"  # provenance name is preserved
    assert params["elo1"] == 99.0  # overridden field
    assert params["elo0"] == -10.0  # untouched field still comes from the config


def test_resolve_gate_config_rejects_unknown_name() -> None:
    with pytest.raises(SystemExit):
        resolve_gate_config("not_a_real_gate_config")


# ---------------------------------------------------------------------------
# R9 timeout rule: score_posterior_stats / r9_timeout_verdict.
#
# Synthetic pentanomial (LL, split, WW) counts at the FLYWHEEL_GATE_CONFIG
# bounds (elo0=-10, elo1=+15), covering the three CAT-7-mandated scenarios:
# a clear pass (H1), a clear fail (H0), and the timeout/canary path (SPRT
# still "continue" at the game cap, but the posterior clears the R9 bar).


def test_r9_clear_pass_resolves_h1_at_flywheel_bounds() -> None:
    # Heavy-WW pentanomial sample: mean pair score ~0.85, way past elo1=+15.
    report = evaluate_pentanomial_sprt(counts=(10, 40, 150), elo0=-10.0, elo1=15.0)
    assert report["decision"] == "H1"
    assert report["llr"] >= report["upper_bound"]


def test_r9_clear_fail_resolves_h0_at_flywheel_bounds() -> None:
    # Mirror image: heavy-LL sample, mean pair score ~0.15.
    report = evaluate_pentanomial_sprt(counts=(150, 40, 10), elo0=-10.0, elo1=15.0)
    assert report["decision"] == "H0"
    assert report["llr"] <= report["lower_bound"]


def test_r9_timeout_path_is_still_continue_on_the_two_sided_sprt() -> None:
    # 600 pairs (the flywheel gate's max_games), no splits, 52% win rate
    # (WW=312, LL=288): a genuine "reached the game cap, still undecided"
    # scenario -- the two-sided GSPRT correctly stays in "continue" since
    # 52% is between the elo0=-10 and elo1=+15 hypotheses, not decisively
    # past either one.
    report = evaluate_pentanomial_sprt(counts=(288, 0, 312), elo0=-10.0, elo1=15.0)
    assert report["decision"] == "continue"


def test_r9_timeout_verdict_promotes_canary_when_posterior_clears_the_floor() -> None:
    # Same (288, 0, 312) counts as the "still continue" test above: the R9
    # one-sided question (is the candidate probably not a regression?) is
    # answered independently of the two-sided LLR, and DOES clear the bar
    # here -- median Elo is positive and P(Elo < elo0=-10) is small.
    r9 = r9_timeout_verdict((288, 0, 312), elo_floor=-10.0, max_prob_below_floor=0.05)
    assert r9["median_elo"] > 0.0
    assert r9["prob_elo_below_floor"] <= 0.05
    assert r9["canary_eligible"] is True
    assert r9["verdict"] == "canary_promote"


def test_r9_timeout_verdict_stays_continue_when_median_elo_is_not_positive() -> None:
    # Exactly 50/50 (mean pair score == 0.5): median Elo is 0.0, which is
    # NOT strictly positive, so R9 must not fire regardless of variance.
    r9 = r9_timeout_verdict((300, 0, 300), elo_floor=-10.0, max_prob_below_floor=0.05)
    assert r9["median_elo"] == pytest.approx(0.0, abs=1e-9)
    assert r9["canary_eligible"] is False
    assert r9["verdict"] == "continue"


def test_r9_timeout_verdict_stays_continue_when_prob_below_floor_is_too_high() -> None:
    # A small, noisy sample can have a positive median Elo yet still carry
    # too much probability mass below the floor to be trusted -- R9 must
    # not fire just because the point estimate is positive.
    r9 = r9_timeout_verdict((9, 0, 11), elo_floor=-10.0, max_prob_below_floor=0.05)
    assert r9["median_elo"] > 0.0
    assert r9["prob_elo_below_floor"] > 0.05
    assert r9["canary_eligible"] is False
    assert r9["verdict"] == "continue"


def test_score_posterior_stats_mean_matches_pentanomial_llr_mean() -> None:
    # score_posterior_stats and pentanomial_llr must agree on mu_hat (they
    # share the same prior-regularized mean) -- cross-check against
    # evaluate_pentanomial_sprt's own reported mean_pair_score.
    counts = (20, 60, 220)
    report = evaluate_pentanomial_sprt(counts=counts, elo0=-10.0, elo1=15.0)
    stats = score_posterior_stats(counts, values=(0.0, 0.5, 1.0))
    # Both apply the same PENTANOMIAL_PRIOR_PAIRS regularization, so the
    # posterior mean should closely track the reported mean_pair_score
    # (which does NOT include the prior) at this sample size.
    assert stats["mu_hat"] == pytest.approx(report["mean_pair_score"], abs=0.01)


def test_score_posterior_stats_degenerate_zero_total_is_safe() -> None:
    stats = score_posterior_stats((0, 0, 0), values=(0.0, 0.5, 1.0), prior_mass=0.0)
    assert stats["n_effective"] == 0.0
    assert stats["se"] == float("inf")
