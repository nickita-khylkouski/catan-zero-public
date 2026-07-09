#!/usr/bin/env python3
"""CLI: rollout-doubling probe (CAT-25 measurement 2, mechanism B: ExIt
fixed-point).

Thin config generator + subprocess wrapper around
tools/gumbel_search_cross_net_h2h.py -- reuses the existing H2H tool's
interface (emit the command, parse its output) rather than reimplementing
match-playing logic. Plays the SAME checkpoint against itself, with the
candidate role at a DOUBLED search budget (n_full_b, default 128) against
the baseline role at the original budget (n_full_a, default 64). If the ExIt
fixed-point mechanism (B) dominates the plateau, doubling rollouts at fixed
policy/value net should yield a candidate win rate near 50% (the network has
already converged to what more search of the SAME net can extract); if it
still wins meaningfully above 50%, search headroom remains and the plateau
is likelier attributable to mechanisms A or C instead.

Pairs/games convention: `--pairs` (default 200) is the underlying H2H tool's
own `--pairs` flag, which means "paired seeds; total games = 2x this" (each
pair is played TWICE, color-swapped). So `--pairs 200` yields 400 total
games. This resolves the CAT-25 ticket's "400 paired games" language as
200 pairs / 400 games (not 400 pairs / 800 games) -- documented here
explicitly because the ticket phrasing is ambiguous.

Candidate/baseline convention: "candidate" conventionally means "the thing
being evaluated" -- here, the DOUBLED-budget arm (n_full_b) is `--candidate`
and the original-budget arm (n_full_a) is `--baseline`, both pointed at the
SAME checkpoint. `candidate_win_rate` in the H2H output is therefore the
doubled-budget arm's win rate against the original-budget arm.

Modes:
  (no --run, no --dry-run)  Just BUILD the command (list[str]) and return it
                             -- no subprocess call, no side effects. Safe to
                             import/call in a test.
  --dry-run                 Print the constructed command AND a JSON dict
                             describing the invocation, without ever
                             importing multiprocessing/torch/rust. Exits 0.
  --run                     Actually invoke the H2H tool via subprocess.run,
                             then load+parse its --out JSON into a compact
                             rollout-doubling summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from factory_common import write_json

H2H_TOOL_RELPATH = "tools/gumbel_search_cross_net_h2h.py"
# Repo root, so `--run`'s actual subprocess call works regardless of the
# caller's cwd (H2H_TOOL_RELPATH itself stays relative/repo-root-relative,
# since it's also what gets printed for a human to copy-paste onto a GPU
# host -- mirrors the cwd=str(REPO_ROOT) convention in
# tools/continuous_flywheel.py's subprocess.run call).
_REPO_ROOT = _TOOLS_DIR.parent


def build_h2h_command(
    *,
    champion_checkpoint: str,
    n_full_a: int,
    n_full_b: int,
    pairs: int,
    h2h_out_path: str,
    workers: int = 8,
    base_seed: int = 1,
) -> list[str]:
    """The literal H2H command this probe evaluates: candidate = the n_full_b
    (doubled-budget) arm, baseline = the n_full_a (original-budget) arm, BOTH
    pointed at `champion_checkpoint` (the same checkpoint played against
    itself at two budgets)."""
    return [
        sys.executable,
        H2H_TOOL_RELPATH,
        "--candidate",
        champion_checkpoint,
        "--baseline",
        champion_checkpoint,
        "--candidate-n-full",
        str(int(n_full_b)),
        "--baseline-n-full",
        str(int(n_full_a)),
        "--pairs",
        str(int(pairs)),
        "--workers",
        str(int(workers)),
        "--base-seed",
        str(int(base_seed)),
        "--out",
        str(h2h_out_path),
    ]


def build_invocation_descriptor(
    *,
    champion_checkpoint: str,
    n_full_a: int,
    n_full_b: int,
    pairs: int,
    h2h_out_path: str,
    workers: int = 8,
    base_seed: int = 1,
) -> dict[str, Any]:
    """JSON-describable summary of what `build_h2h_command` would run,
    independent of whether it's actually invoked -- used by --dry-run and by
    tests."""
    command = build_h2h_command(
        champion_checkpoint=champion_checkpoint,
        n_full_a=n_full_a,
        n_full_b=n_full_b,
        pairs=pairs,
        h2h_out_path=h2h_out_path,
        workers=workers,
        base_seed=base_seed,
    )
    return {
        "measurement": "rollout_doubling_probe",
        "mechanism": "B_exit_fixed_point",
        "champion_checkpoint": champion_checkpoint,
        "n_full_a": int(n_full_a),
        "n_full_b": int(n_full_b),
        "pairs": int(pairs),
        "games_total": int(pairs) * 2,
        "workers": int(workers),
        "base_seed": int(base_seed),
        "h2h_out_path": str(h2h_out_path),
        "command": command,
    }


def extract_rollout_doubling_summary(h2h_report: dict[str, Any]) -> dict[str, Any]:
    """Pull the rollout-doubling-specific fields out of a full
    gumbel_search_cross_net_h2h.py --out JSON (or an equivalent hand-built
    fake dict in tests)."""
    return {
        "candidate_win_rate": h2h_report.get("candidate_win_rate"),
        "candidate_wins": h2h_report.get("candidate_wins"),
        "baseline_wins": h2h_report.get("baseline_wins"),
        "games_played": h2h_report.get("games_played"),
        "games_with_winner": h2h_report.get("games_with_winner"),
        "candidate_n_full": h2h_report.get("candidate_n_full"),
        "baseline_n_full": h2h_report.get("baseline_n_full"),
        "pentanomial_sprt": h2h_report.get("pentanomial_sprt"),
        "pair_diagnostics": h2h_report.get("pair_diagnostics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rollout-doubling probe: same checkpoint at n_full vs 2x n_full search "
            "budget. Wraps tools/gumbel_search_cross_net_h2h.py rather than "
            "reimplementing match play."
        )
    )
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument(
        "--pairs",
        type=int,
        default=200,
        help="paired seeds passed to the underlying H2H tool; total games = 2x this (default 400 games)",
    )
    parser.add_argument("--n-full-a", type=int, default=64, help="baseline (original) search budget")
    parser.add_argument("--n-full-b", type=int, default=128, help="candidate (doubled) search budget")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument(
        "--h2h-out",
        default=None,
        help="Path for the underlying H2H tool's --out JSON. Defaults to "
        "<--out with .h2h.json suffix> when --out is given.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the constructed command + invocation descriptor JSON; no subprocess call.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually invoke the H2H tool via subprocess.run and parse its --out JSON.",
    )
    args = parser.parse_args()

    h2h_out_path = args.h2h_out or (str(Path(args.out).with_suffix("")) + ".h2h.json")

    descriptor = build_invocation_descriptor(
        champion_checkpoint=args.champion_checkpoint,
        n_full_a=int(args.n_full_a),
        n_full_b=int(args.n_full_b),
        pairs=int(args.pairs),
        h2h_out_path=h2h_out_path,
        workers=int(args.workers),
        base_seed=int(args.base_seed),
    )

    if args.run:
        # H2H_TOOL_RELPATH is repo-root-relative (intentionally, so the
        # printed command is copy-paste-able onto a GPU host); cwd=_REPO_ROOT
        # makes the actual subprocess call correct regardless of the
        # directory this script itself was invoked from.
        result = subprocess.run(descriptor["command"], check=True, cwd=str(_REPO_ROOT))
        h2h_report = json.loads(Path(h2h_out_path).read_text(encoding="utf-8"))
        summary = {
            **descriptor,
            "ran": True,
            "subprocess_returncode": result.returncode,
            "rollout_doubling_summary": extract_rollout_doubling_summary(h2h_report),
        }
        write_json(args.out, summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "command"}, indent=2, sort_keys=True))
        return

    # Default (no --run) and --dry-run both stop here: build+report the
    # command, no subprocess/torch/rust import. --dry-run is explicit about
    # printing; the plain "just build the command" mode still writes the
    # descriptor to --out for provenance, so a human can re-run the printed
    # command on a GPU host with --run afterward.
    summary = {**descriptor, "ran": False}
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        print("\n# To actually run this probe on a GPU host:")
        print(" ".join(descriptor["command"]))


if __name__ == "__main__":
    main()
