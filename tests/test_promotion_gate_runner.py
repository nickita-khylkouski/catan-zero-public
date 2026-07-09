from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.promotion_gate_runner import (
    DEFAULT_ROSTER,
    build_scoreboard_command,
    check_min_leg_games,
    compute_roster_deltas,
    decide_promotion,
    derive_seed,
    finalize_verdict,
    h2h_extend_tiers,
    is_bot_kind,
    opponent_spec,
    resolve_subject,
    run_h2h_leg,
    subject_label,
    wrap_remote,
)
from tools.sprt_gate import FLYWHEEL_GATE_CONFIG


def test_is_bot_kind_recognizes_known_names_and_paths() -> None:
    assert is_bot_kind("catanatron_value")
    assert is_bot_kind("CATANATRON_AB3")  # case-insensitive
    assert not is_bot_kind("runs/bc/some_checkpoint/checkpoint.pt")


def test_resolve_subject_checkpoint_vs_bot_kind() -> None:
    assert resolve_subject("catanatron_ab3") == ("catanatron_ab3", "catanatron_ab3")
    assert resolve_subject("runs/bc/x/checkpoint.pt") == ("checkpoint", "runs/bc/x/checkpoint.pt")


def test_opponent_spec_prefixes_checkpoints_only() -> None:
    assert opponent_spec("catanatron_value") == "catanatron_value"
    assert opponent_spec("runs/bc/x/checkpoint.pt") == "checkpoint:runs/bc/x/checkpoint.pt"


def test_subject_label_is_filesystem_safe_and_distinguishes_checkpoints() -> None:
    assert subject_label("catanatron_value") == "catanatron_value"
    label_a = subject_label("runs/bc/run_a/checkpoint.pt")
    label_b = subject_label("runs/bc/run_b/checkpoint.pt")
    assert label_a != label_b  # same filename, different run dirs must not collide


def test_default_roster_uses_traced_hardtarget_checkpoint() -> None:
    assert "catanatron_value" in DEFAULT_ROSTER
    assert "catanatron_ab3" in DEFAULT_ROSTER
    assert any("hardtarget" in item for item in DEFAULT_ROSTER)


def test_build_scoreboard_command_checkpoint_candidate_bot_opponent(tmp_path: Path) -> None:
    command = build_scoreboard_command(
        subject="runs/bc/x/checkpoint.pt",
        opponent="catanatron_ab3",
        games=1000,
        seed=42,
        vps_to_win=10,
        max_decisions=1000,
        workers=8,
        device="cpu",
        out=tmp_path / "leg.json",
    )
    assert command[1] == "tools/evaluate_scoreboard.py"
    assert command[command.index("--candidate") + 1] == "runs/bc/x/checkpoint.pt"
    assert command[command.index("--candidate-kind") + 1] == "checkpoint"
    assert command[command.index("--opponents") + 1] == "catanatron_ab3"
    assert "--paired-seeds" in command
    assert command[command.index("--games") + 1] == "1000"
    assert command[command.index("--seed") + 1] == "42"


def test_build_scoreboard_command_checkpoint_opponent_gets_prefixed(tmp_path: Path) -> None:
    command = build_scoreboard_command(
        subject="catanatron_value",
        opponent="runs/bc/baseline/checkpoint.pt",
        games=1000,
        seed=42,
        vps_to_win=10,
        max_decisions=1000,
        workers=8,
        device="cpu",
        out=tmp_path / "leg.json",
    )
    assert command[command.index("--candidate") + 1] == "catanatron_value"
    assert command[command.index("--candidate-kind") + 1] == "catanatron_value"
    assert command[command.index("--opponents") + 1] == "checkpoint:runs/bc/baseline/checkpoint.pt"


def test_wrap_remote_produces_ssh_invocation_with_cd_into_repo() -> None:
    command = ["python", "tools/evaluate_scoreboard.py", "--out", "some path/with space.json"]
    wrapped = wrap_remote(
        command, host="ubuntu@1.2.3.4", ssh_key="~/.ssh/key", remote_repo_dir="/home/ubuntu/catan-zero"
    )
    assert wrapped[0] == "ssh"
    assert "ubuntu@1.2.3.4" in wrapped
    joined = wrapped[-1]
    assert "cd /home/ubuntu/catan-zero" in joined
    assert "'some path/with space.json'" in joined


# --------------------------------------------------------------------------- decision logic
# This is the core thing the team lead asked to unit-test: promote iff SPRT
# accepts H1 on H2H AND no roster leg regresses beyond -20 Elo equivalent.


def _roster_delta(delta_elo: float, *, p_value: float | None = 0.001) -> dict:
    """p_value=0.001 by default (clearly significant) so existing
    veto-fires tests exercise the F2b significance gate as intended;
    pass p_value=None to simulate an unpaired leg (no McNemar available),
    or a larger p_value to simulate a non-significant delta."""
    return {
        "candidate_elo_vs_opponent": delta_elo,
        "baseline_elo_vs_opponent": 0.0,
        "delta_elo": delta_elo,
        "candidate_win_rate": None,
        "baseline_win_rate": None,
        "paired_mcnemar": {"test": "mcnemar_exact", "p_value": p_value} if p_value is not None else None,
    }


def test_decide_promotion_promotes_on_clear_h2h_win_and_no_regression() -> None:
    outcomes = [True] * 330 + [False] * 270  # 55% over 600 games, resolves elo1=30
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(5.0), "catanatron_ab3": _roster_delta(-3.0)},
    )
    assert decision["verdict"] == "promote"
    assert decision["h2h_sprt"]["decision"] == "H1"
    assert decision["roster_regressions"] == {}


def test_decide_promotion_rejects_on_roster_regression_even_with_h2h_win() -> None:
    outcomes = [True] * 330 + [False] * 270  # would promote on H2H alone
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(-25.0), "catanatron_ab3": _roster_delta(2.0)},
    )
    assert decision["verdict"] == "reject"
    assert "catanatron_value" in decision["roster_regressions"]
    assert "regression" in decision["reason"]


def test_decide_promotion_rejects_on_h2h_h0_with_healthy_roster() -> None:
    outcomes = [True] * 200 + [False] * 800  # clearly losing H2H
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(1.0)},
    )
    assert decision["verdict"] == "reject"
    assert decision["h2h_sprt"]["decision"] == "H0"
    assert decision["roster_regressions"] == {}


def test_decide_promotion_continues_when_h2h_is_still_undecided() -> None:
    outcomes = [True] * 55 + [False] * 45  # 55% but too few games to resolve elo1=30
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(0.0)},
    )
    assert decision["verdict"] == "continue"
    assert decision["h2h_sprt"]["decision"] == "continue"


def test_decide_promotion_roster_regression_threshold_is_exclusive_at_boundary() -> None:
    outcomes = [True] * 55 + [False] * 45
    at_threshold = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(-20.0)},
        max_roster_regression_elo=20.0,
    )
    assert at_threshold["roster_regressions"] == {}  # exactly -20 does not trigger reject
    beyond_threshold = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(-20.01)},
        max_roster_regression_elo=20.0,
    )
    assert "catanatron_value" in beyond_threshold["roster_regressions"]


def test_decide_promotion_custom_max_regression_threshold() -> None:
    outcomes = [True] * 330 + [False] * 270
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(-8.0)},
        max_roster_regression_elo=5.0,
    )
    assert decision["verdict"] == "reject"
    assert "catanatron_value" in decision["roster_regressions"]


# --------------------------------------------------------------------------- F2b: significance-gated roster veto
# The quant audit found the raw-Elo-threshold veto false-positives 13-31% of
# the time under the null of zero regression at typical per-leg game counts
# (per-leg Elo SE ~12-17 at 1000 games). The veto must require BOTH a large
# delta_elo AND a significant paired McNemar p-value.


def test_decide_promotion_large_delta_without_significance_does_not_veto() -> None:
    outcomes = [True] * 330 + [False] * 270  # would promote on H2H alone
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(-25.0, p_value=0.5)},  # large but NOT significant
    )
    assert decision["verdict"] == "promote"  # veto did not fire
    assert decision["roster_regressions"] == {}


def test_decide_promotion_large_delta_with_significance_does_veto() -> None:
    outcomes = [True] * 330 + [False] * 270
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(-25.0, p_value=0.01)},  # large AND significant
    )
    assert decision["verdict"] == "reject"
    assert "catanatron_value" in decision["roster_regressions"]


def test_decide_promotion_large_delta_with_no_mcnemar_is_unverifiable_not_vetoed() -> None:
    outcomes = [True] * 330 + [False] * 270
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(-25.0, p_value=None)},  # unpaired leg
    )
    assert decision["verdict"] == "promote"  # no significance test available -> does not veto
    assert decision["roster_regressions"] == {}
    assert decision["roster_regressions_large_but_unverifiable"] == {"catanatron_value": -25.0}


def test_decide_promotion_significance_threshold_is_configurable() -> None:
    outcomes = [True] * 330 + [False] * 270
    decision = decide_promotion(
        h2h_outcomes=outcomes,
        roster_deltas={"catanatron_value": _roster_delta(-25.0, p_value=0.03)},
        roster_significance_alpha=0.01,  # stricter than the p_value -> does not veto
    )
    assert decision["verdict"] == "promote"
    assert decision["roster_regressions"] == {}


# --------------------------------------------------------------------------- F9: seed derandomization


def test_derive_seed_is_deterministic() -> None:
    a = derive_seed("candidate.pt", "baseline.pt", "2026-07-03")
    b = derive_seed("candidate.pt", "baseline.pt", "2026-07-03")
    assert a == b


def test_derive_seed_differs_across_candidate_baseline_or_date() -> None:
    base = derive_seed("candidate.pt", "baseline.pt", "2026-07-03")
    assert derive_seed("other_candidate.pt", "baseline.pt", "2026-07-03") != base
    assert derive_seed("candidate.pt", "other_baseline.pt", "2026-07-03") != base
    assert derive_seed("candidate.pt", "baseline.pt", "2026-07-04") != base


# --------------------------------------------------------------------------- F2a: extend-on-continue tiers


def test_h2h_extend_tiers_default_schedule_is_1x_2x_3x() -> None:
    assert h2h_extend_tiers(1000, None) == [1000, 2000, 3000]
    assert h2h_extend_tiers(500, None) == [500, 1000, 1500]


def test_h2h_extend_tiers_explicit_override_is_sorted_and_deduplicated() -> None:
    assert h2h_extend_tiers(1000, "3000,1000,2000,2000") == [1000, 2000, 3000]


def test_h2h_extend_tiers_rejects_empty_override() -> None:
    with pytest.raises(SystemExit):
        h2h_extend_tiers(1000, "  ,  ")


def _write_h2h_report(path: Path, *, outcomes: list, seed: int = 1) -> None:
    wins = sum(1 for outcome in outcomes if outcome)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate": "candidate.pt",
                "seed": seed,
                "paired_seeds": True,
                "results": [
                    {
                        "opponent": "checkpoint:baseline.pt",
                        "wins": wins,
                        "games": len(outcomes),
                        "win_rate": wins / len(outcomes) if outcomes else 0.0,
                        "leg_seed": 999,
                        "game_outcomes": outcomes,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_run_h2h_leg_stops_at_first_tier_that_resolves(tmp_path: Path) -> None:
    h2h_out = tmp_path / "h2h.json"
    dispatch_calls: list[int] = []

    def fake_dispatch(command: list[str], index: int) -> None:
        dispatch_calls.append(index)
        # A clear +35-Elo-style win rate that resolves H1 immediately at tier 0.
        _write_h2h_report(h2h_out, outcomes=[True] * 330 + [False] * 270)

    report, sprt = run_h2h_leg(
        candidate="candidate.pt",
        baseline="baseline.pt",
        tiers=[600, 1200, 1800],
        seed=1,
        vps_to_win=10,
        max_decisions=1000,
        workers=8,
        device="cpu",
        h2h_out=h2h_out,
        dispatch=fake_dispatch,
        elo0=0.0,
        elo1=30.0,
        alpha=0.05,
        beta=0.05,
        min_leg_games=50,
    )
    assert dispatch_calls == [0]  # only the first tier was ever dispatched
    assert sprt["decision"] == "H1"
    assert sprt["tier_index"] == 0
    assert sprt["tier_games"] == 600


def test_run_h2h_leg_extends_through_tiers_when_undecided(tmp_path: Path) -> None:
    h2h_out = tmp_path / "h2h.json"
    dispatch_calls: list[int] = []

    def fake_dispatch(command: list[str], index: int) -> None:
        dispatch_calls.append(index)
        games = [100, 200, 300][index]
        # Undecided (50/50) at every tier -- forces extension through all of them.
        outcomes = [True] * (games // 2) + [False] * (games - games // 2)
        _write_h2h_report(h2h_out, outcomes=outcomes)

    report, sprt = run_h2h_leg(
        candidate="candidate.pt",
        baseline="baseline.pt",
        tiers=[100, 200, 300],
        seed=1,
        vps_to_win=10,
        max_decisions=1000,
        workers=8,
        device="cpu",
        h2h_out=h2h_out,
        dispatch=fake_dispatch,
        elo0=0.0,
        elo1=30.0,
        alpha=0.05,
        beta=0.05,
        min_leg_games=50,
    )
    assert dispatch_calls == [0, 1, 2]  # extended through every tier
    assert sprt["decision"] == "continue"  # F2c: exhausted tiers, still honestly "continue"
    assert sprt["tier_games"] == 300


def test_run_h2h_leg_does_not_redispatch_a_tier_already_satisfied(tmp_path: Path) -> None:
    h2h_out = tmp_path / "h2h.json"
    # Pre-seed the leg file as if a prior invocation already reached tier 0.
    _write_h2h_report(h2h_out, outcomes=[True] * 55 + [False] * 45)
    dispatch_calls: list[int] = []

    def fake_dispatch(command: list[str], index: int) -> None:
        dispatch_calls.append(index)

    report, sprt = run_h2h_leg(
        candidate="candidate.pt",
        baseline="baseline.pt",
        tiers=[100],
        seed=1,
        vps_to_win=10,
        max_decisions=1000,
        workers=8,
        device="cpu",
        h2h_out=h2h_out,
        dispatch=fake_dispatch,
        elo0=0.0,
        elo1=30.0,
        alpha=0.05,
        beta=0.05,
        min_leg_games=50,
    )
    assert dispatch_calls == []  # existing file already satisfies the only tier
    assert sprt["tier_games"] == 100


def test_run_h2h_leg_finalize_verdict_never_reports_reject_for_continue(tmp_path: Path) -> None:
    # F2c, end-to-end: combine an exhausted-tiers "continue" H2H result with
    # a clean roster to confirm the final verdict is "continue", not reject.
    h2h_out = tmp_path / "h2h.json"

    def fake_dispatch(command: list[str], index: int) -> None:
        _write_h2h_report(h2h_out, outcomes=[True] * 50 + [False] * 50)

    _, sprt = run_h2h_leg(
        candidate="candidate.pt",
        baseline="baseline.pt",
        tiers=[100],
        seed=1,
        vps_to_win=10,
        max_decisions=1000,
        workers=8,
        device="cpu",
        h2h_out=h2h_out,
        dispatch=fake_dispatch,
        elo0=0.0,
        elo1=30.0,
        alpha=0.05,
        beta=0.05,
        min_leg_games=50,
    )
    decision = finalize_verdict(sprt, {"catanatron_value": _roster_delta(0.0)})
    assert decision["verdict"] == "continue"
    assert decision["verdict"] != "reject"


# --------------------------------------------------------------------------- compute_roster_deltas


def _scoreboard_report(*, seed, paired_seeds, opponent, wins, games, elo_vs_opponent, game_outcomes=None) -> dict:
    return {
        "candidate": "some_checkpoint.pt",
        "seed": seed,
        "paired_seeds": paired_seeds,
        "results": [
            {
                "opponent": opponent,
                "wins": wins,
                "games": games,
                "win_rate": wins / games,
                "elo_vs_opponent": elo_vs_opponent,
                "leg_seed": 12345,
                "game_outcomes": game_outcomes,
            }
        ],
    }


def test_compute_roster_deltas_extracts_elo_diff_and_paired_mcnemar_when_available() -> None:
    outcomes_candidate = [True, True, False, True, False, False, True, True, False, True]
    outcomes_baseline = [True, False, False, False, False, True, False, True, False, True]
    candidate_reports = {
        "catanatron_ab3": _scoreboard_report(
            seed=7,
            paired_seeds=True,
            opponent="catanatron_ab3",
            wins=sum(outcomes_candidate),
            games=10,
            elo_vs_opponent=40.0,
            game_outcomes=outcomes_candidate,
        )
    }
    baseline_reports = {
        "catanatron_ab3": _scoreboard_report(
            seed=7,
            paired_seeds=True,
            opponent="catanatron_ab3",
            wins=sum(outcomes_baseline),
            games=10,
            elo_vs_opponent=10.0,
            game_outcomes=outcomes_baseline,
        )
    }
    deltas = compute_roster_deltas(candidate_reports, baseline_reports, ("catanatron_ab3",))
    assert deltas["catanatron_ab3"]["delta_elo"] == pytest.approx(30.0)
    assert deltas["catanatron_ab3"]["paired_mcnemar"] is not None
    assert deltas["catanatron_ab3"]["paired_mcnemar"]["test"] == "mcnemar_exact"


def test_compute_roster_deltas_marks_diagnostic_unavailable_when_unpaired() -> None:
    candidate_reports = {
        "catanatron_ab3": _scoreboard_report(
            seed=None, paired_seeds=False, opponent="catanatron_ab3", wins=6, games=10, elo_vs_opponent=40.0
        )
    }
    baseline_reports = {
        "catanatron_ab3": _scoreboard_report(
            seed=None, paired_seeds=False, opponent="catanatron_ab3", wins=4, games=10, elo_vs_opponent=10.0
        )
    }
    deltas = compute_roster_deltas(candidate_reports, baseline_reports, ("catanatron_ab3",))
    assert deltas["catanatron_ab3"]["paired_mcnemar"] is None
    assert deltas["catanatron_ab3"]["pairing_unavailable_reason"] is not None


# --------------------------------------------------------------------------- adversarial-review fixes


def test_check_min_leg_games_passes_at_or_above_threshold() -> None:
    check_min_leg_games("h2h", 50, 50)  # no raise
    check_min_leg_games("h2h", 1000, 50)  # no raise


def test_check_min_leg_games_raises_below_threshold() -> None:
    with pytest.raises(SystemExit):
        check_min_leg_games("roster/catanatron_value/candidate", 49, 50)


def test_check_min_leg_games_raises_on_zero_games() -> None:
    # The exact scenario the adversarial review flagged: a 0-games leg must
    # never silently read as delta_elo=0.0 "no regression".
    with pytest.raises(SystemExit):
        check_min_leg_games("roster/catanatron_value/candidate", 0, 50)


def test_decide_promotion_excludes_truncated_h2h_games_from_sprt() -> None:
    # Regression test for the adversarial-review truncation-as-loss bias:
    # a None entry (truncated game) must be excluded from the SPRT input,
    # not coerced to False (a candidate loss), and the excluded count must
    # be reported.
    outcomes_with_truncations: list = [True] * 330 + [False] * 270 + [None] * 50
    decision = decide_promotion(
        h2h_outcomes=outcomes_with_truncations,
        roster_deltas={"catanatron_value": _roster_delta(0.0)},
    )
    # Same 330/270 split as the clean-promote test -> same H1 verdict, the
    # 50 truncated games must not have diluted or biased the LLR.
    assert decision["verdict"] == "promote"
    assert decision["h2h_sprt"]["games"] == 600
    assert decision["h2h_sprt"]["truncated_games_excluded"] == 50


def test_decide_promotion_all_h2h_games_truncated_does_not_crash() -> None:
    decision = decide_promotion(
        h2h_outcomes=[None] * 10,
        roster_deltas={"catanatron_value": _roster_delta(0.0)},
    )
    assert decision["h2h_sprt"]["games"] == 0
    assert decision["h2h_sprt"]["truncated_games_excluded"] == 10
    assert decision["verdict"] == "continue"


# --------------------------------------------------------------------------- CAT-7: flywheel gate config + R9 timeout rule


def test_h2h_extend_tiers_max_games_derives_flywheel_two_tier_schedule() -> None:
    # flywheel gate config: base_games=300, max_games=600 -> exactly two
    # tiers, NOT the certification-style 1x/2x/3x.
    assert h2h_extend_tiers(300, None, max_games=600) == [300, 600]


def test_h2h_extend_tiers_max_games_reproduces_certification_schedule() -> None:
    # certification gate config: base_games=1000, max_games=3000 -> the
    # historical 1x/2x/3x tiers, unchanged.
    assert h2h_extend_tiers(1000, None, max_games=3000) == [1000, 2000, 3000]


def test_h2h_extend_tiers_no_max_games_preserves_legacy_1x2x3x_default() -> None:
    # Direct function callers that don't pass max_games (e.g. the existing
    # unit tests above) must see exactly the pre-CAT-7 behavior.
    assert h2h_extend_tiers(1000, None) == [1000, 2000, 3000]


def test_run_h2h_leg_resolves_canary_promote_at_flywheel_bounds_when_timed_out(tmp_path: Path) -> None:
    # 600 games (the flywheel gate's max_games) at a 52% win rate (312/288):
    # verified in test_sprt_gate.py to leave the two-sided SPRT at
    # elo0=-10/elo1=15 in "continue", while the R9 posterior check (median
    # Elo > 0, P(Elo < -10) <= 0.05) clears the canary bar. A single tier
    # means run_h2h_leg's loop hits tier_index == len(tiers) - 1 on its only
    # iteration, so the R9 check fires immediately.
    h2h_out = tmp_path / "h2h.json"

    def fake_dispatch(command: list[str], index: int) -> None:
        _write_h2h_report(h2h_out, outcomes=[True] * 312 + [False] * 288)

    _, sprt = run_h2h_leg(
        candidate="candidate.pt",
        baseline="baseline.pt",
        tiers=[600],
        seed=1,
        vps_to_win=10,
        max_decisions=1000,
        workers=8,
        device="cpu",
        h2h_out=h2h_out,
        dispatch=fake_dispatch,
        elo0=FLYWHEEL_GATE_CONFIG.elo0,
        elo1=FLYWHEEL_GATE_CONFIG.elo1,
        alpha=FLYWHEEL_GATE_CONFIG.alpha,
        beta=FLYWHEEL_GATE_CONFIG.beta,
        min_leg_games=50,
    )
    assert sprt["decision"] == "canary_promote"
    assert sprt["r9_timeout"]["canary_eligible"] is True
    assert sprt["r9_timeout"]["median_elo"] > 0.0


def test_run_h2h_leg_r9_check_does_not_fire_before_the_final_tier(tmp_path: Path) -> None:
    # Same win rate as above, but reached at tier 0 of a TWO-tier schedule --
    # R9 must not fire early and short-circuit the extend-on-continue
    # machinery (F2a); the leg must still extend to the next tier.
    h2h_out = tmp_path / "h2h.json"
    dispatch_calls: list[int] = []

    def fake_dispatch(command: list[str], index: int) -> None:
        dispatch_calls.append(index)
        games = [600, 1200][index]
        wins = round(games * 0.52)
        _write_h2h_report(h2h_out, outcomes=[True] * wins + [False] * (games - wins))

    _, sprt = run_h2h_leg(
        candidate="candidate.pt",
        baseline="baseline.pt",
        tiers=[600, 1200],
        seed=1,
        vps_to_win=10,
        max_decisions=1000,
        workers=8,
        device="cpu",
        h2h_out=h2h_out,
        dispatch=fake_dispatch,
        elo0=FLYWHEEL_GATE_CONFIG.elo0,
        elo1=FLYWHEEL_GATE_CONFIG.elo1,
        alpha=FLYWHEEL_GATE_CONFIG.alpha,
        beta=FLYWHEEL_GATE_CONFIG.beta,
        min_leg_games=50,
    )
    assert dispatch_calls == [0, 1]  # did NOT stop after tier 0's "continue"
    assert sprt["tier_index"] == 1  # R9 (if it fires) only evaluated on the last tier


def test_finalize_verdict_maps_canary_promote_decision_to_canary_promote_verdict() -> None:
    sprt_report = {"decision": "canary_promote", "r9_timeout": {"canary_eligible": True}}
    decision = finalize_verdict(sprt_report, {"catanatron_value": _roster_delta(0.0)})
    assert decision["verdict"] == "canary_promote"
    assert "generator-only" in decision["reason"].lower() or "generator" in decision["reason"].lower()


def test_finalize_verdict_roster_regression_still_vetoes_canary_promote() -> None:
    # The roster regression veto must take priority over a canary_promote
    # verdict exactly as it does over "promote" -- a candidate that quietly
    # regressed against an anchor must never be waved through as a
    # generator-only canary either.
    sprt_report = {"decision": "canary_promote", "r9_timeout": {"canary_eligible": True}}
    decision = finalize_verdict(sprt_report, {"catanatron_value": _roster_delta(-25.0)})
    assert decision["verdict"] == "reject"
    assert "catanatron_value" in decision["roster_regressions"]


# --- BUG-3: promotion-gate verdict must be written atomically ------------------

def test_verdict_write_is_atomic_and_byte_identical(tmp_path: Path) -> None:
    """The gate verdict decides promotion vs rollback; a killed process must
    never leave a truncated verdict.json. It is now written via
    write_json_atomic (temp + fsync + os.replace). Assert the atomic helper
    produces byte-identical content to the old json.dumps(...indent=2,
    sort_keys=True) + "\\n" path and leaves no stray .tmp files."""
    from tools.atomic_io import write_json_atomic

    verdict = {"decision": "promote", "elo": 12.5, "nested": {"b": 2, "a": 1}}
    out = tmp_path / "sub" / "verdict.json"
    write_json_atomic(out, verdict)

    expected = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    assert out.read_text(encoding="utf-8") == expected
    # No leftover temp files from the atomic write.
    assert not list(tmp_path.glob("**/*.tmp"))
    assert not list(tmp_path.glob("**/.*.tmp"))


def test_gate_runner_uses_atomic_verdict_write() -> None:
    """Regression guard: the gate runner must not revert to a bare
    Path.write_text for its terminal verdict."""
    import inspect

    from tools import promotion_gate_runner

    src = inspect.getsource(promotion_gate_runner)
    assert "write_json_atomic(output, verdict)" in src
    assert "output.write_text(" not in src
