from __future__ import annotations

"""The ONE code path for every promotion decision (kills the old ad-hoc
multi-gate process).

For a candidate checkpoint vs a baseline (checkpoint or bot), runs:
  1. A direct H2H paired-seed leg (candidate vs baseline) via
     tools/evaluate_scoreboard.py, and feeds its per-game outcomes into
     tools/sprt_gate.py under a named --gate-config (CAT-7): 'flywheel'
     (elo0=-10, elo1=15, 300 games extending to 600) is the default and is
     used for every ordinary promotion decision; 'certification' (elo0=0,
     elo1=30, 1000 games extending to 3000, the pre-CAT-7 default) is
     reserved for discrete public gen-N announcements. Pass
     --gate-config certification to reproduce the historical behavior.
  2. A frozen-anchor roster leg for BOTH candidate and baseline against each
     roster member (same seed => paired), to catch a candidate that beats
     the baseline head-to-head but has quietly regressed against the
     anchors.

Decision rule: promote iff (a) SPRT accepts H1 on the H2H leg AND (b) no
roster opponent's Elo-vs-opponent regressed by more than
--max-roster-regression-elo (default 20) relative to the baseline. A clear
roster regression rejects immediately regardless of the H2H SPRT state;
otherwise H1 -> promote, H0 -> reject, continue -> continue (collect more
H2H games).

R9 TIMEOUT RULE (CAT-7): if the H2H leg is still "continue" at the largest
extension tier (the game cap is reached without H0/H1), tools/sprt_gate.py's
r9_timeout_verdict checks the posterior over the per-game win probability.
If the median Elo is positive AND P(Elo < elo0) is at or below
--r9-max-prob-below-floor (default 0.05), the leg resolves to
"canary_promote" instead of "continue" -- a candidate that is probably not
a regression even though it isn't (yet) a proven improvement. This verdict
is GENERATOR-ONLY: it is meant to seed the next round of self-play
generation and must never be treated as, or feed, a public gen-N
announcement. It is still subject to the roster regression veto above.

After the legs land, refreshes the Elo ladder (tools/elo_ladder.py) over the
scoreboards directory (which now includes this run's new leg files) and
writes one verdict JSON with every number: H2H SPRT trajectory, per-opponent
roster deltas (+ a diagnostic paired McNemar p-value reusing
tools/compare_scoreboards.py's exact test), and the refreshed ladder.

Safety nets (from an adversarial review of this tool):
  - A leg with fewer than --min-leg-games (default 50) games is a hard
    error, not a silent pass -- a 0-games leg would otherwise read as
    delta_elo=0.0 "no regression".
  - Truncated games (no winner) are recorded as None and excluded from the
    SPRT/McNemar pairing rather than silently coerced into a loss.
  - Re-running with a larger --games-per-leg after an SPRT "continue"
    verdict is a HARD ERROR if a ROSTER leg file already exists with a
    different game count (roster legs do not auto-extend; reusing one would
    silently cap it forever). Point --leg-dir at a fresh directory, or
    delete the stale leg files, to collect more roster games.

Fixes from the quant/power audit (gate was underpowered enough to mis-price
gen-1's verdict -- these gate the VALIDITY of any promotion decision, not
just its plumbing):
  - F2a EXTEND-ON-CONTINUE: the H2H leg does NOT terminate at a single
    --games-per-leg sample. It pre-commits to a tier schedule
    (--games-per-leg, 2x, 3x by default -- see --h2h-extend-tiers) and
    automatically plays more games (same seed, so the first N games
    reproduce byte-for-byte) whenever the SPRT is still "continue", up to
    the largest tier. At N=1000 a genuinely-successful +35 Elo candidate
    only resolves H1 ~77% of the time; at 2000 that rises to ~96%.
  - F2b SIGNIFICANCE-GATED ROSTER VETO: a roster leg only vetoes promotion
    if delta_elo < -max-roster-regression-elo AND the paired McNemar
    p-value (already computed per-leg) is below --roster-significance-alpha
    (default 0.05). Per-leg Elo standard error at 1000 games (~12-17 Elo) is
    large enough that raw-Elo-threshold vetoes false-positive 13-31% of the
    time under the null of zero regression. A large delta with no
    significance test available (e.g. unpaired legs) is surfaced under
    "roster_regressions_large_but_unverifiable" but does NOT veto by itself.
  - F2c CONTINUE IS NEVER REJECT: exhausting every extension tier still
    "continue" is reported as verdict="continue", never coerced to
    "reject" -- an inconclusive gate must never be mistaken for a rejection.
  - F9 SEED DERANDOMIZATION: --seed defaults to
    hash(candidate, baseline, today's date), not a fixed panel -- reusing
    the same seed panel across every gate run correlates outcomes across
    runs (family-wise error inflation) and risks overfitting to specific
    boards. Pass --seed explicitly only to reproduce a past run exactly.

CLI:
  python tools/promotion_gate_runner.py \\
      --candidate runs/.../candidate.pt --baseline runs/.../baseline.pt \\
      --roster catanatron_value,catanatron_ab3,runs/.../hardtarget.pt \\
      --games-per-leg 1000 --out runs/scoreboards/promotion_gate_.../verdict.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from compare_scoreboards import _mcnemar_exact  # noqa: E402
from elo_ladder import _load_matchups, build_ladder_report  # noqa: E402
from sprt_gate import GATE_CONFIGS, evaluate_sprt, r9_timeout_verdict, resolve_gate_config  # noqa: E402
from atomic_io import write_json_atomic  # noqa: E402

# Mirrors catan_zero.rl.policy_pool.make_policy's recognized "kind" strings.
# Anything NOT in this set is treated as a checkpoint path.
KNOWN_BOT_KINDS = frozenset(
    {
        "random",
        "heuristic",
        "catanatron_heuristic",
        "jsettlers_lite",
        "value",
        "catanatron_value",
        "alphabeta",
        "alpha_beta",
        "ab3",
        "catanatron_ab3",
        "catanatron_alphabeta",
        "ab4",
        "catanatron_ab4",
        "ab5",
        "catanatron_ab5",
        "sab3",
        "catanatron_sab3",
        "same_turn_ab3",
        "sab4",
        "catanatron_sab4",
        "same_turn_ab4",
        "mcts100",
        "catanatron_mcts100",
        "mcts50",
        "catanatron_mcts50",
        "greedy25",
        "catanatron_greedy25",
        "greedy_playouts25",
        "search",
        "value_rollout",
        "value_rollout_search",
        "catanatron_search",
        "weighted_random",
        "catanatron_weighted_random",
        "catanatron_value_ore_city",
        "catanatron_value_road_race",
        "catanatron_value_robber",
    }
)

# The frozen anchor roster: the two most stable bot anchors plus the
# canonical hard-target BC checkpoint (traced via report.json init_checkpoint
# lineage: every hard-target repair branch -- vrs_only_diag, dagger_repair,
# phase_repair -- warm-starts from this checkpoint; it is the base the repair
# branches are measured against, not a repair branch itself).
#
# G2 roster expansion (task #3, lead-authorized for future gates): add the
# three value-function style specialists so promotions must beat a spread of
# playing styles (ore/city + dev engine, road race, robber-aggressive), not
# just the single contender-weighted value bot. The frozen anchors above are
# kept unchanged so cross-gate Elo history stays comparable.
DEFAULT_ROSTER = (
    "catanatron_value",
    "catanatron_ab3",
    "runs/bc/entity_graph_35m_oldbase_hardtarget_ab45_robber_opening_20260630_220320/checkpoint.pt",
    "catanatron_value_ore_city",
    "catanatron_value_road_race",
    "catanatron_value_robber",
)


def derive_seed(candidate: str, baseline: str, run_date: str) -> int:
    """FIX (F9, seed derandomization): a fixed seed panel reused across every
    gate run correlates every run's board sample -- inflating family-wise
    error and risking overfitting to that specific panel. Derive a fresh
    seed per (candidate, baseline, date) instead of a hardcoded constant."""
    digest = hashlib.sha256(f"{candidate}::{baseline}::{run_date}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def h2h_extend_tiers(games_per_leg: int, extend_tiers: str | None, *, max_games: int | None = None) -> list[int]:
    """FIX (F2a, extend-on-continue): resolve the H2H leg's tier schedule.
    With no `max_games` bound, the default is (N, 2N, 3N) -- e.g.
    1000/2000/3000 -- since P(promote | truly +35 Elo) rises from ~77% at
    N=1000 to ~96% at N=2000 (the certification-config game budget).

    `max_games` (CAT-7): when given, generates consecutive N, 2N, 3N, ...
    tiers only up through max_games -- e.g. games_per_leg=300,
    max_games=600 -> [300, 600], matching the flywheel gate config's
    smaller two-tier budget instead of the certification config's 3x
    schedule."""
    if extend_tiers:
        tiers = sorted({int(item.strip()) for item in extend_tiers.split(",") if item.strip()})
        if not tiers:
            raise SystemExit("--h2h-extend-tiers parsed to an empty tier list")
        return tiers
    if max_games is not None:
        tiers = []
        current = games_per_leg
        while current < max_games:
            tiers.append(current)
            current += games_per_leg
        tiers.append(max_games)
        return sorted(set(tiers))
    return [games_per_leg, games_per_leg * 2, games_per_leg * 3]


def is_bot_kind(name: str) -> bool:
    return name.strip().lower() in KNOWN_BOT_KINDS


def resolve_subject(name: str) -> tuple[str, str]:
    """Return (candidate_kind, candidate) for evaluate_scoreboard.py's
    --candidate/--candidate-kind, for a roster/candidate/baseline identifier
    that may be a bot kind name or a checkpoint path."""
    if is_bot_kind(name):
        return name, name
    return "checkpoint", name


def opponent_spec(name: str) -> str:
    """Format an identifier as an evaluate_scoreboard.py --opponents entry."""
    if is_bot_kind(name):
        return name
    return f"checkpoint:{name}"


def subject_label(name: str) -> str:
    """Short, filesystem-safe label for a subject (checkpoint path or bot kind)."""
    if is_bot_kind(name):
        return name.strip().lower()
    path = Path(name)
    parent = path.parent.name
    return f"{parent}_{path.stem}" if parent else path.stem


def build_scoreboard_command(
    *,
    subject: str,
    opponent: str,
    games: int,
    seed: int,
    vps_to_win: int,
    max_decisions: int,
    workers: int,
    device: str,
    out: Path,
) -> list[str]:
    candidate_kind, candidate = resolve_subject(subject)
    return [
        sys.executable,
        "tools/evaluate_scoreboard.py",
        "--candidate",
        candidate,
        "--candidate-kind",
        candidate_kind,
        "--games",
        str(games),
        "--tracks",
        "2p_no_trade",
        "--opponents",
        opponent_spec(opponent),
        "--seed",
        str(seed),
        "--paired-seeds",
        "--vps-to-win",
        str(vps_to_win),
        "--max-decisions",
        str(max_decisions),
        "--workers",
        str(workers),
        "--device",
        device,
        "--out",
        str(out),
    ]


def wrap_remote(command: list[str], *, host: str, ssh_key: str, remote_repo_dir: str) -> list[str]:
    remote_command = " ".join(_shell_quote(part) for part in command)
    return [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath=/tmp/ssh-mux-promotion-gate-{host}",
        "-o",
        "ControlPersist=900",
        host,
        f"cd {_shell_quote(remote_repo_dir)} && {remote_command}",
    ]


def _shell_quote(value: str) -> str:
    if not value or any(c in value for c in " \t\n\"'$`\\"):
        return "'" + value.replace("'", "'\\''") + "'"
    return value


def run_command(command: list[str], *, timeout: float | None = None) -> None:
    print(json.dumps({"command": command}, sort_keys=True), flush=True)
    subprocess.run(command, check=True, timeout=timeout)


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_result(report: dict[str, Any], opponent: str) -> dict[str, Any]:
    spec = opponent_spec(opponent)
    for result in report.get("results", []):
        if str(result.get("opponent")) == spec:
            return result
    raise SystemExit(f"opponent {spec!r} not found in report for {report.get('candidate')!r}")


def check_min_leg_games(label: str, games: int, min_leg_games: int) -> None:
    """FIX (adversarial review, zero-games roster leg passes): a leg with 0
    games produces delta_elo=0.0 (both sides' win rate clips to the same
    floor) and silently reads as "no regression" -- a hard error is much
    safer than a quiet false pass. Applies to the H2H leg too, not just
    roster legs, since a degenerate H2H sample is exactly as unsafe to feed
    into the SPRT."""
    if games < min_leg_games:
        raise SystemExit(
            f"leg {label!r} has only {games} games, below --min-leg-games "
            f"{min_leg_games}. Refusing to treat this as a valid pass/fail "
            f"signal -- investigate why the leg produced so few games "
            f"(crashed early? timed out?) rather than trusting the result."
        )


def compute_roster_deltas(
    candidate_reports: dict[str, dict[str, Any]],
    baseline_reports: dict[str, dict[str, Any]],
    roster: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Per roster opponent: candidate_elo_vs_opponent - baseline_elo_vs_opponent,
    plus a diagnostic paired McNemar test when both legs carry pairing metadata."""
    deltas: dict[str, dict[str, Any]] = {}
    for opponent in roster:
        candidate_result = extract_result(candidate_reports[opponent], opponent)
        baseline_result = extract_result(baseline_reports[opponent], opponent)
        candidate_elo = float(candidate_result["elo_vs_opponent"])
        baseline_elo = float(baseline_result["elo_vs_opponent"])
        entry: dict[str, Any] = {
            "candidate_elo_vs_opponent": candidate_elo,
            "baseline_elo_vs_opponent": baseline_elo,
            "delta_elo": candidate_elo - baseline_elo,
            "candidate_win_rate": candidate_result.get("win_rate"),
            "baseline_win_rate": baseline_result.get("win_rate"),
        }
        candidate_paired = dict(candidate_result)
        candidate_paired["_report_seed"] = candidate_reports[opponent].get("seed")
        candidate_paired["_report_paired_seeds"] = bool(candidate_reports[opponent].get("paired_seeds"))
        baseline_paired = dict(baseline_result)
        baseline_paired["_report_seed"] = baseline_reports[opponent].get("seed")
        baseline_paired["_report_paired_seeds"] = bool(baseline_reports[opponent].get("paired_seeds"))
        from compare_scoreboards import _pairing_eligible

        eligible, reason = _pairing_eligible(candidate_paired, baseline_paired)
        if eligible:
            entry["paired_mcnemar"] = _mcnemar_exact(
                candidate_result["game_outcomes"], baseline_result["game_outcomes"]
            )
        else:
            entry["paired_mcnemar"] = None
            entry["pairing_unavailable_reason"] = reason
        deltas[opponent] = entry
    return deltas


def finalize_verdict(
    sprt_report: dict[str, Any],
    roster_deltas: dict[str, dict[str, Any]],
    *,
    max_roster_regression_elo: float = 20.0,
    roster_significance_alpha: float = 0.05,
) -> dict[str, Any]:
    """Combine an already-computed H2H SPRT report with roster deltas into
    the final promote/reject/continue verdict: promote iff SPRT accepts H1
    on the H2H leg AND no roster leg significantly regressed; H0 -> reject;
    continue -> continue (never coerced to reject -- F2c).

    FIX (F2b, adversarial review): the roster veto used to fire on delta_elo
    alone. At ~1000 games/leg, per-leg Elo standard error (~12-17 Elo) means
    13-31% of GENUINELY NULL regressions would false-veto a good candidate.
    Require BOTH delta_elo below -max_roster_regression_elo AND the paired
    McNemar p-value (already computed by compute_roster_deltas) below
    roster_significance_alpha. A large delta with no significance test
    available (unpaired legs) is reported separately, but does not veto.
    """
    regressions: dict[str, float] = {}
    unverifiable_large_deltas: dict[str, float] = {}
    for opponent, entry in roster_deltas.items():
        if entry["delta_elo"] >= -max_roster_regression_elo:
            continue
        mcnemar = entry.get("paired_mcnemar")
        if mcnemar is not None and mcnemar["p_value"] < roster_significance_alpha:
            regressions[opponent] = entry["delta_elo"]
        elif mcnemar is None:
            unverifiable_large_deltas[opponent] = entry["delta_elo"]
    if regressions:
        verdict = "reject"
        reason = (
            f"roster regression beyond -{max_roster_regression_elo:.1f} Elo AND "
            f"McNemar p<{roster_significance_alpha} vs baseline: "
            + ", ".join(f"{opp}={delta:.1f}" for opp, delta in sorted(regressions.items()))
        )
    elif sprt_report["decision"] == "H1":
        verdict = "promote"
        reason = "SPRT accepted H1 on the H2H leg with no significant roster regression"
    elif sprt_report["decision"] == "H0":
        verdict = "reject"
        reason = "SPRT accepted H0 on the H2H leg (candidate not meaningfully stronger)"
    elif sprt_report["decision"] == "canary_promote":
        verdict = "canary_promote"
        reason = (
            "R9 timeout rule: the H2H SPRT was still undecided at the largest "
            "extension tier, but the posterior favors the candidate strongly "
            "enough (median Elo > 0, P(Elo < elo0) below threshold) to seed "
            "the next self-play generation. Generator-only -- NOT a public "
            "gen-N promotion."
        )
    else:
        verdict = "continue"
        reason = "SPRT undecided on the H2H leg even at the largest extension tier; collect more games"
    return {
        "verdict": verdict,
        "reason": reason,
        "h2h_sprt": sprt_report,
        "roster_deltas": roster_deltas,
        "roster_regressions": regressions,
        "roster_regressions_large_but_unverifiable": unverifiable_large_deltas,
        "max_roster_regression_elo": max_roster_regression_elo,
        "roster_significance_alpha": roster_significance_alpha,
    }


def decide_promotion(
    *,
    h2h_outcomes: list[bool | None],
    roster_deltas: dict[str, dict[str, Any]],
    elo0: float = 0.0,
    elo1: float = 30.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_roster_regression_elo: float = 20.0,
    roster_significance_alpha: float = 0.05,
) -> dict[str, Any]:
    """Single-shot convenience wrapper (no tiered H2H extension): compute
    the H2H SPRT from a raw outcomes list and hand off to finalize_verdict.
    Used directly by unit tests and any one-shot caller; run_promotion_gate
    itself uses the tiered run_h2h_leg + finalize_verdict path (F2a).

    FIX (adversarial review, truncation-as-loss bias): h2h_outcomes entries
    may be None (truncated game, no winner). Filtered here -- defense in
    depth even if a caller forgets to pre-filter -- rather than letting
    sprt_gate.evaluate_sprt's bool(outcome) coerce None to a candidate loss.
    """
    truncated_h2h_games = sum(1 for outcome in h2h_outcomes if outcome is None)
    usable_h2h_outcomes = [bool(outcome) for outcome in h2h_outcomes if outcome is not None]
    sprt_report = evaluate_sprt(usable_h2h_outcomes, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)
    sprt_report["truncated_games_excluded"] = truncated_h2h_games
    return finalize_verdict(
        sprt_report,
        roster_deltas,
        max_roster_regression_elo=max_roster_regression_elo,
        roster_significance_alpha=roster_significance_alpha,
    )


def run_h2h_leg(
    *,
    candidate: str,
    baseline: str,
    tiers: list[int],
    seed: int,
    vps_to_win: int,
    max_decisions: int,
    workers: int,
    device: str,
    h2h_out: Path,
    dispatch,
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
    min_leg_games: int,
    r9_max_prob_below_floor: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """FIX (F2a, extend-on-continue): rather than terminating at one
    --games-per-leg sample, pre-commit to extending the SAME H2H leg (same
    --seed, so the first N games reproduce byte-for-byte -- evaluate_scoreboard
    .py's seed formula depends only on chunk_index, not the total game count)
    up through each tier in `tiers` until the SPRT resolves to H1/H0, or the
    largest tier is exhausted -- in which case the honest terminal verdict is
    "continue" (F2c), never silently coerced to "reject"."""
    report: dict[str, Any] = {}
    sprt_report: dict[str, Any] = {}
    for tier_index, games in enumerate(tiers):
        existing_games = None
        if h2h_out.exists():
            existing_games = int(extract_result(load_report(h2h_out), baseline).get("games", 0))
        if existing_games is None or existing_games < games:
            command = build_scoreboard_command(
                subject=candidate,
                opponent=baseline,
                games=games,
                seed=seed,
                vps_to_win=vps_to_win,
                max_decisions=max_decisions,
                workers=workers,
                device=device,
                out=h2h_out,
            )
            dispatch(command, tier_index)
        report = load_report(h2h_out)
        result = extract_result(report, baseline)
        actual_games = int(result.get("games", 0))
        check_min_leg_games("h2h", actual_games, min_leg_games)
        outcomes = result.get("game_outcomes")
        if not isinstance(outcomes, list):
            raise SystemExit("H2H leg is missing game_outcomes; is evaluate_scoreboard.py up to date?")
        truncated = sum(1 for outcome in outcomes if outcome is None)
        usable = [bool(outcome) for outcome in outcomes if outcome is not None]
        sprt_report = evaluate_sprt(usable, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)
        sprt_report["truncated_games_excluded"] = truncated
        sprt_report["tier_games"] = actual_games
        sprt_report["tier_index"] = tier_index
        sprt_report["tiers"] = list(tiers)
        if sprt_report["decision"] == "continue" and tier_index == len(tiers) - 1:
            # R9 timeout rule: the largest tier is exhausted and the H2H
            # SPRT is still undecided. Check whether the posterior
            # nonetheless clears the "probably not a regression" bar --
            # generator-only, never a public promotion (see finalize_verdict).
            wins_final = sum(1 for outcome in usable if outcome)
            losses_final = len(usable) - wins_final
            r9 = r9_timeout_verdict(
                (losses_final, wins_final),
                values=(0.0, 1.0),
                elo_floor=elo0,
                max_prob_below_floor=r9_max_prob_below_floor,
            )
            sprt_report["r9_timeout"] = r9
            if r9["canary_eligible"]:
                sprt_report["decision"] = "canary_promote"
        if sprt_report["decision"] != "continue":
            break
    return report, sprt_report


def refresh_ladder(
    *, scoreboards_dir: Path, anchor: str, bootstrap_samples: int, track: str = "2p_no_trade"
) -> dict[str, Any]:
    matchups, skipped_tracks, era_file_counts = _load_matchups(
        scoreboards_dir, repo_root=Path("."), track=track
    )
    report = build_ladder_report(matchups, anchor=anchor, bootstrap_samples=bootstrap_samples)
    report["skipped_other_tracks"] = sorted(skipped_tracks)
    report["era_file_counts"] = era_file_counts
    return report


def run_promotion_gate(
    *,
    candidate: str,
    baseline: str,
    roster: tuple[str, ...],
    games_per_leg: int,
    h2h_tiers: list[int],
    seed: int,
    vps_to_win: int,
    max_decisions: int,
    workers: int,
    device: str,
    hosts: tuple[str, ...],
    ssh_key: str,
    remote_repo_dir: str,
    leg_dir: Path,
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
    max_roster_regression_elo: float,
    roster_significance_alpha: float,
    min_leg_games: int,
    scoreboards_dir: Path,
    ladder_anchor: str,
    ladder_bootstrap_samples: int,
    gate_config_params: dict[str, Any] | None = None,
    r9_max_prob_below_floor: float = 0.05,
) -> dict[str, Any]:
    leg_dir.mkdir(parents=True, exist_ok=True)
    host_cycle = list(hosts)

    def _dispatch(command: list[str], index: int) -> None:
        if host_cycle:
            host = host_cycle[index % len(host_cycle)]
            command = wrap_remote(command, host=host, ssh_key=ssh_key, remote_repo_dir=remote_repo_dir)
        run_command(command)

    # F2a: the H2H leg is NOT part of the generic fixed-N jobs list below --
    # it auto-extends through h2h_tiers (see run_h2h_leg) rather than
    # terminating at one games_per_leg sample.
    jobs: list[tuple[str, list[str], Path, str]] = []

    candidate_outs: dict[str, Path] = {}
    baseline_outs: dict[str, Path] = {}
    for opponent in roster:
        candidate_out = leg_dir / f"roster_{subject_label(candidate)}_vs_{subject_label(opponent)}.json"
        baseline_out = leg_dir / f"roster_{subject_label(baseline)}_vs_{subject_label(opponent)}.json"
        candidate_outs[opponent] = candidate_out
        baseline_outs[opponent] = baseline_out
        jobs.append(
            (
                f"roster/{opponent}/candidate",
                build_scoreboard_command(
                    subject=candidate,
                    opponent=opponent,
                    games=games_per_leg,
                    seed=seed,
                    vps_to_win=vps_to_win,
                    max_decisions=max_decisions,
                    workers=workers,
                    device=device,
                    out=candidate_out,
                ),
                candidate_out,
                opponent,
            )
        )
        jobs.append(
            (
                f"roster/{opponent}/baseline",
                build_scoreboard_command(
                    subject=baseline,
                    opponent=opponent,
                    games=games_per_leg,
                    seed=seed,
                    vps_to_win=vps_to_win,
                    max_decisions=max_decisions,
                    workers=workers,
                    device=device,
                    out=baseline_out,
                ),
                baseline_out,
                opponent,
            )
        )

    for index, (label, command, out_path, opponent) in enumerate(jobs):
        if out_path.exists():
            # FIX (adversarial review, continue-verdict footgun): reusing a
            # leg file that has fewer games than requested would silently
            # cap this leg forever, no-op'ing exactly the "collect more
            # games" action an SPRT "continue" verdict calls for. Only skip
            # when the existing file already has the requested game count.
            existing_games = int(extract_result(load_report(out_path), opponent).get("games", 0))
            if existing_games != games_per_leg:
                raise SystemExit(
                    f"leg {label!r} at {out_path} already has {existing_games} games, "
                    f"but --games-per-leg requested {games_per_leg}. Reusing it would "
                    f"silently cap this leg at {existing_games} games forever -- if an "
                    f"earlier run's SPRT verdict was 'continue' and you're trying to "
                    f"collect more games, either (a) delete {out_path} and rerun (loses "
                    f"the {existing_games} already-played games), or (b) pass a fresh "
                    f"--leg-dir for a clean {games_per_leg}-game run. Incremental "
                    f"accumulation (reuse existing games, only play the delta) is not "
                    f"yet implemented."
                )
            print(json.dumps({"skip_existing": label, "path": str(out_path)}))
            continue
        _dispatch(command, index)

    h2h_out = leg_dir / f"h2h_{subject_label(candidate)}_vs_{subject_label(baseline)}.json"
    h2h_report, h2h_sprt = run_h2h_leg(
        candidate=candidate,
        baseline=baseline,
        tiers=h2h_tiers,
        seed=seed,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        workers=workers,
        device=device,
        h2h_out=h2h_out,
        dispatch=_dispatch,
        elo0=elo0,
        elo1=elo1,
        alpha=alpha,
        beta=beta,
        min_leg_games=min_leg_games,
        r9_max_prob_below_floor=r9_max_prob_below_floor,
    )
    h2h_result = extract_result(h2h_report, baseline)

    candidate_reports = {opponent: load_report(candidate_outs[opponent]) for opponent in roster}
    baseline_reports = {opponent: load_report(baseline_outs[opponent]) for opponent in roster}
    for opponent in roster:
        check_min_leg_games(
            f"roster/{opponent}/candidate",
            int(extract_result(candidate_reports[opponent], opponent).get("games", 0)),
            min_leg_games,
        )
        check_min_leg_games(
            f"roster/{opponent}/baseline",
            int(extract_result(baseline_reports[opponent], opponent).get("games", 0)),
            min_leg_games,
        )
    roster_deltas = compute_roster_deltas(candidate_reports, baseline_reports, roster)

    decision = finalize_verdict(
        h2h_sprt,
        roster_deltas,
        max_roster_regression_elo=max_roster_regression_elo,
        roster_significance_alpha=roster_significance_alpha,
    )

    ladder = refresh_ladder(
        scoreboards_dir=scoreboards_dir, anchor=ladder_anchor, bootstrap_samples=ladder_bootstrap_samples
    )

    return {
        "candidate": candidate,
        "baseline": baseline,
        "roster": list(roster),
        "games_per_leg": games_per_leg,
        "h2h_tiers": list(h2h_tiers),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "h2h_win_rate": h2h_result.get("win_rate"),
        "h2h_games": h2h_result.get("games"),
        # CAT-7: record which named gate config (+ its resolved elo0/elo1/
        # alpha/beta/n_sims/base_games/max_games) produced this verdict, so
        # WHR ingest and future audits read it from the artifact instead of
        # inferring it from elo0/elo1 alone.
        "gate_config_params": gate_config_params,
        **decision,
        "ladder": ladder,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, help="Candidate checkpoint path or bot kind name.")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path or bot kind name (or a named champion path).")
    parser.add_argument(
        "--roster",
        default=",".join(DEFAULT_ROSTER),
        help="Comma-separated frozen anchor roster (checkpoint paths and/or bot kind names).",
    )
    parser.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default="flywheel",
        help=(
            "Named SPRT gate config (CAT-7): 'flywheel' (elo0=-10/elo1=15, "
            "300 games extending to 600) is the day-to-day producer gate "
            "used for every promotion decision going forward. "
            "'certification' (elo0=0/elo1=30, 1000 games extending to 3000) "
            "is reserved for discrete public gen-N announcements. Sets "
            "defaults for --games-per-leg/--elo0/--elo1/--alpha/--beta; "
            "explicit flags override individual fields."
        ),
    )
    parser.add_argument(
        "--games-per-leg",
        type=int,
        default=None,
        help="Base H2H tier size. Default: the --gate-config's base_games.",
    )
    parser.add_argument(
        "--h2h-extend-tiers",
        default=None,
        help=(
            "Comma-separated game-count tiers for the H2H leg (F2a, "
            "extend-on-continue). Default: derived from the --gate-config's "
            "base_games/max_games (flywheel: 300,600; certification: "
            "1000,2000,3000). The leg auto-extends through these tiers "
            "(same seed, so earlier games reproduce byte-for-byte) whenever "
            "the SPRT is still 'continue', stopping at H1/H0/canary_promote "
            "or the largest tier (an honest terminal 'continue', never "
            "coerced to reject)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base seed for every leg. Default (F9, seed derandomization): "
            "derived from hash(candidate, baseline, today's date) -- a fixed "
            "seed panel reused across every gate run correlates the board "
            "sample and inflates family-wise error. Pass an explicit value "
            "only to reproduce a specific past run."
        ),
    )
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--hosts",
        default="",
        help="Comma-separated ssh targets (user@host) to round-robin legs across. Empty = run locally.",
    )
    parser.add_argument("--ssh-key", default="~/.ssh/gpu_access_ed25519")
    parser.add_argument("--remote-repo-dir", default="/home/ubuntu/catan-zero")
    parser.add_argument("--leg-dir", required=True, help="Directory to write per-leg scoreboard JSONs into.")
    parser.add_argument("--elo0", type=float, default=None, help="Override --gate-config's elo0.")
    parser.add_argument("--elo1", type=float, default=None, help="Override --gate-config's elo1.")
    parser.add_argument("--alpha", type=float, default=None, help="Override --gate-config's alpha.")
    parser.add_argument("--beta", type=float, default=None, help="Override --gate-config's beta.")
    parser.add_argument(
        "--r9-max-prob-below-floor",
        type=float,
        default=0.05,
        help=(
            "R9 timeout rule: if the H2H SPRT is still 'continue' at the "
            "largest extension tier, the leg resolves to 'canary_promote' "
            "(generator-only, never a public promotion) when the posterior "
            "median Elo is positive AND P(Elo < elo0) is at or below this "
            "threshold."
        ),
    )
    parser.add_argument("--max-roster-regression-elo", type=float, default=20.0)
    parser.add_argument(
        "--roster-significance-alpha",
        type=float,
        default=0.05,
        help=(
            "F2b: a roster leg only vetoes promotion if delta_elo is beyond "
            "-max-roster-regression-elo AND its paired McNemar p-value is "
            "below this threshold. Guards against the raw-Elo-threshold "
            "veto false-positiving on noise at typical per-leg game counts."
        ),
    )
    parser.add_argument(
        "--min-leg-games",
        type=int,
        default=50,
        help=(
            "Hard error (not a silent pass) if the H2H leg or any roster leg "
            "has fewer than this many games -- guards against a 0-games leg "
            "reading as delta_elo=0.0 'no regression'."
        ),
    )
    parser.add_argument("--scoreboards-dir", default="runs/scoreboards")
    parser.add_argument("--ladder-anchor", default="catanatron_value")
    parser.add_argument("--ladder-bootstrap-samples", type=int, default=100)
    parser.add_argument("--out", required=True, help="Verdict JSON output path.")
    args = parser.parse_args()

    roster = tuple(item.strip() for item in args.roster.split(",") if item.strip())
    hosts = tuple(item.strip() for item in args.hosts.split(",") if item.strip())
    seed = (
        args.seed
        if args.seed is not None
        else derive_seed(args.candidate, args.baseline, time.strftime("%Y-%m-%d", time.gmtime()))
    )
    gate_config, gate_params = resolve_gate_config(
        args.gate_config, elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta
    )
    games_per_leg = args.games_per_leg if args.games_per_leg is not None else gate_config.base_games
    h2h_tiers = h2h_extend_tiers(games_per_leg, args.h2h_extend_tiers, max_games=gate_config.max_games)

    verdict = run_promotion_gate(
        candidate=args.candidate,
        baseline=args.baseline,
        roster=roster,
        games_per_leg=games_per_leg,
        h2h_tiers=h2h_tiers,
        seed=seed,
        vps_to_win=args.vps_to_win,
        max_decisions=args.max_decisions,
        workers=args.workers,
        device=args.device,
        hosts=hosts,
        ssh_key=args.ssh_key,
        remote_repo_dir=args.remote_repo_dir,
        leg_dir=Path(args.leg_dir),
        elo0=gate_params["elo0"],
        elo1=gate_params["elo1"],
        alpha=gate_params["alpha"],
        beta=gate_params["beta"],
        max_roster_regression_elo=args.max_roster_regression_elo,
        roster_significance_alpha=args.roster_significance_alpha,
        min_leg_games=args.min_leg_games,
        scoreboards_dir=Path(args.scoreboards_dir),
        ladder_anchor=args.ladder_anchor,
        ladder_bootstrap_samples=args.ladder_bootstrap_samples,
        gate_config_params=gate_params,
        r9_max_prob_below_floor=args.r9_max_prob_below_floor,
    )
    output = Path(args.out)
    text = json.dumps(verdict, indent=2, sort_keys=True)
    # BUG-3: atomic write -- a killed/crashed process must not leave a
    # truncated verdict.json. write_json_atomic writes temp+fsync+os.replace
    # and produces byte-identical content (indent=2, sort_keys=True, trailing \n).
    write_json_atomic(output, verdict)
    print(text)


if __name__ == "__main__":
    main()
