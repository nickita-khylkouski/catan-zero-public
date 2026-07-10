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

The same wrapper also supports the minimum adaptive-opening probe without
changing that role convention: set both normal budgets to 128, candidate/B's
wide budget to 256, and leave baseline/A's wide budget disabled. The shared
``n_full_wide`` option remains a backwards-compatible fallback for both roles;
role-specific A/B values override only their own side.

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
import hashlib
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


def _resolve_wide_budgets(
    *,
    n_full_wide: int | None,
    n_full_wide_a: int | None,
    n_full_wide_b: int | None,
) -> tuple[int | None, int | None]:
    """Return effective (baseline/A, candidate/B) wide-root budgets."""
    baseline = int(n_full_wide_a) if n_full_wide_a is not None else n_full_wide
    candidate = int(n_full_wide_b) if n_full_wide_b is not None else n_full_wide
    return baseline, candidate


def _resolve_wide_thresholds(
    *,
    n_full_wide_threshold: int | None,
    n_full_wide_threshold_a: int | None,
    n_full_wide_threshold_b: int | None,
) -> tuple[int | None, int | None]:
    """Return effective (baseline/A, candidate/B) inclusive width gates."""
    baseline = (
        int(n_full_wide_threshold_a)
        if n_full_wide_threshold_a is not None
        else n_full_wide_threshold
    )
    candidate = (
        int(n_full_wide_threshold_b)
        if n_full_wide_threshold_b is not None
        else n_full_wide_threshold
    )
    return baseline, candidate


def _hash_payload(payload: dict[str, Any]) -> tuple[str, str]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}", f"sha256:{digest}"


def build_h2h_command(
    *,
    champion_checkpoint: str,
    n_full_a: int,
    n_full_b: int,
    pairs: int,
    h2h_out_path: str,
    workers: int = 8,
    base_seed: int = 1,
    devices: str | None = None,
    max_decisions: int = 600,
    max_depth: int = 80,
    c_visit: float = 50.0,
    c_scale: float = 0.03,
    rescale_noise_floor_c: float = 0.0,
    sigma_eval: float = 0.79,
    public_observation: bool = True,
    lazy_interior_chance: bool = True,
    correct_rust_chance_spectra: bool = True,
    symmetry_averaged_eval: bool = False,
    n_full_wide: int | None = None,
    n_full_wide_a: int | None = None,
    n_full_wide_b: int | None = None,
    n_full_wide_threshold: int | None = None,
    n_full_wide_threshold_a: int | None = None,
    n_full_wide_threshold_b: int | None = None,
    wide_candidates_threshold: int = 24,
    symmetry_averaged_eval_threshold: int | None = None,
) -> list[str]:
    """The literal H2H command this probe evaluates: candidate = the n_full_b
    (doubled-budget) arm, baseline = the n_full_a (original-budget) arm, BOTH
    pointed at `champion_checkpoint` (the same checkpoint played against
    itself at two budgets)."""
    command = [
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
        "--max-decisions",
        str(int(max_decisions)),
        "--max-depth",
        str(int(max_depth)),
        "--c-visit",
        str(float(c_visit)),
        "--c-scale",
        str(float(c_scale)),
        "--rescale-noise-floor-c",
        str(float(rescale_noise_floor_c)),
        "--sigma-eval",
        str(float(sigma_eval)),
        "--wide-candidates-threshold",
        str(int(wide_candidates_threshold)),
        ("--public-observation" if public_observation else "--no-public-observation"),
        ("--lazy-interior-chance" if lazy_interior_chance else "--no-lazy-interior-chance"),
        (
            "--correct-rust-chance-spectra"
            if correct_rust_chance_spectra
            else "--no-correct-rust-chance-spectra"
        ),
        (
            "--symmetry-averaged-eval"
            if symmetry_averaged_eval
            else "--no-symmetry-averaged-eval"
        ),
        "--out",
        str(h2h_out_path),
    ]
    if n_full_wide is not None:
        command.extend(["--n-full-wide", str(int(n_full_wide))])
    if n_full_wide_threshold is not None:
        command.extend(["--n-full-wide-threshold", str(int(n_full_wide_threshold))])
    if n_full_wide_b is not None:
        command.extend(["--candidate-n-full-wide", str(int(n_full_wide_b))])
    if n_full_wide_a is not None:
        command.extend(["--baseline-n-full-wide", str(int(n_full_wide_a))])
    if n_full_wide_threshold_b is not None:
        command.extend(
            ["--candidate-n-full-wide-threshold", str(int(n_full_wide_threshold_b))]
        )
    if n_full_wide_threshold_a is not None:
        command.extend(
            ["--baseline-n-full-wide-threshold", str(int(n_full_wide_threshold_a))]
        )
    if symmetry_averaged_eval_threshold is not None:
        command.extend(
            [
                "--symmetry-averaged-eval-threshold",
                str(int(symmetry_averaged_eval_threshold)),
            ]
        )
    if devices:
        command.extend(["--devices", str(devices)])
    return command


def build_invocation_descriptor(
    *,
    champion_checkpoint: str,
    n_full_a: int,
    n_full_b: int,
    pairs: int,
    h2h_out_path: str,
    workers: int = 8,
    base_seed: int = 1,
    devices: str | None = None,
    max_decisions: int = 600,
    max_depth: int = 80,
    c_visit: float = 50.0,
    c_scale: float = 0.03,
    rescale_noise_floor_c: float = 0.0,
    sigma_eval: float = 0.79,
    public_observation: bool = True,
    lazy_interior_chance: bool = True,
    correct_rust_chance_spectra: bool = True,
    symmetry_averaged_eval: bool = False,
    n_full_wide: int | None = None,
    n_full_wide_a: int | None = None,
    n_full_wide_b: int | None = None,
    n_full_wide_threshold: int | None = None,
    n_full_wide_threshold_a: int | None = None,
    n_full_wide_threshold_b: int | None = None,
    wide_candidates_threshold: int = 24,
    symmetry_averaged_eval_threshold: int | None = None,
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
        devices=devices,
        max_decisions=max_decisions,
        max_depth=max_depth,
        c_visit=c_visit,
        c_scale=c_scale,
        rescale_noise_floor_c=rescale_noise_floor_c,
        sigma_eval=sigma_eval,
        public_observation=public_observation,
        lazy_interior_chance=lazy_interior_chance,
        correct_rust_chance_spectra=correct_rust_chance_spectra,
        symmetry_averaged_eval=symmetry_averaged_eval,
        n_full_wide=n_full_wide,
        n_full_wide_a=n_full_wide_a,
        n_full_wide_b=n_full_wide_b,
        n_full_wide_threshold=n_full_wide_threshold,
        n_full_wide_threshold_a=n_full_wide_threshold_a,
        n_full_wide_threshold_b=n_full_wide_threshold_b,
        wide_candidates_threshold=wide_candidates_threshold,
        symmetry_averaged_eval_threshold=symmetry_averaged_eval_threshold,
    )
    resolved_wide_a, resolved_wide_b = _resolve_wide_budgets(
        n_full_wide=n_full_wide,
        n_full_wide_a=n_full_wide_a,
        n_full_wide_b=n_full_wide_b,
    )
    resolved_threshold_a, resolved_threshold_b = _resolve_wide_thresholds(
        n_full_wide_threshold=n_full_wide_threshold,
        n_full_wide_threshold_a=n_full_wide_threshold_a,
        n_full_wide_threshold_b=n_full_wide_threshold_b,
    )
    probe_config = {
        "measurement": "rollout_doubling_probe",
        "mechanism": "B_exit_fixed_point",
        "champion_checkpoint": champion_checkpoint,
        "n_full_a": int(n_full_a),
        "n_full_b": int(n_full_b),
        "pairs": int(pairs),
        "games_total": int(pairs) * 2,
        "workers": int(workers),
        "base_seed": int(base_seed),
        "devices": devices,
        "n_full_wide": (int(n_full_wide) if n_full_wide is not None else None),
        "n_full_wide_a": resolved_wide_a,
        "n_full_wide_b": resolved_wide_b,
        "n_full_wide_threshold": (
            int(n_full_wide_threshold) if n_full_wide_threshold is not None else None
        ),
        "n_full_wide_threshold_a": resolved_threshold_a,
        "n_full_wide_threshold_b": resolved_threshold_b,
        "search_budgets_by_role": {
            "candidate": {
                "n_full": int(n_full_b),
                "n_full_wide": resolved_wide_b,
                "n_full_wide_threshold": resolved_threshold_b,
            },
            "baseline": {
                "n_full": int(n_full_a),
                "n_full_wide": resolved_wide_a,
                "n_full_wide_threshold": resolved_threshold_a,
            },
        },
        "search_config": {
            "max_decisions": int(max_decisions),
            "max_depth": int(max_depth),
            "c_visit": float(c_visit),
            "c_scale": float(c_scale),
            "rescale_noise_floor_c": float(rescale_noise_floor_c),
            "sigma_eval": float(sigma_eval),
            "public_observation": bool(public_observation),
            "lazy_interior_chance": bool(lazy_interior_chance),
            "correct_rust_chance_spectra": bool(correct_rust_chance_spectra),
            "symmetry_averaged_eval": bool(symmetry_averaged_eval),
            "symmetry_averaged_eval_threshold": (
                int(symmetry_averaged_eval_threshold)
                if symmetry_averaged_eval_threshold is not None
                else None
            ),
            "wide_candidates_threshold": int(wide_candidates_threshold),
        },
    }
    config_hash, full_config_hash = _hash_payload(probe_config)
    return {
        **probe_config,
        "config_hash": config_hash,
        "full_config_hash": full_config_hash,
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
        "candidate_n_full_wide": h2h_report.get("candidate_n_full_wide"),
        "baseline_n_full_wide": h2h_report.get("baseline_n_full_wide"),
        "candidate_n_full_wide_threshold": h2h_report.get(
            "candidate_n_full_wide_threshold"
        ),
        "baseline_n_full_wide_threshold": h2h_report.get(
            "baseline_n_full_wide_threshold"
        ),
        "symmetry_averaged_eval_threshold": h2h_report.get(
            "symmetry_averaged_eval_threshold"
        ),
        "h2h_config_hash": h2h_report.get("config_hash"),
        "h2h_full_config_hash": h2h_report.get("full_config_hash"),
        "search_telemetry": h2h_report.get("search_telemetry"),
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
    parser.add_argument(
        "--n-full-wide",
        type=int,
        default=None,
        help="Shared wide-root budget fallback for both roles (default: disabled).",
    )
    parser.add_argument(
        "--n-full-wide-a",
        "--baseline-n-full-wide",
        dest="n_full_wide_a",
        type=int,
        default=None,
        help="Baseline/A-only wide-root budget (default: inherit --n-full-wide).",
    )
    parser.add_argument(
        "--n-full-wide-b",
        "--candidate-n-full-wide",
        dest="n_full_wide_b",
        type=int,
        default=None,
        help="Candidate/B-only wide-root budget (default: inherit --n-full-wide).",
    )
    parser.add_argument(
        "--n-full-wide-threshold",
        type=int,
        default=None,
        help="Shared inclusive n_full_wide width gate (default: legacy shared gate).",
    )
    parser.add_argument(
        "--n-full-wide-threshold-a",
        "--baseline-n-full-wide-threshold",
        dest="n_full_wide_threshold_a",
        type=int,
        default=None,
        help="Baseline/A-only inclusive wide-budget gate (default: inherit shared).",
    )
    parser.add_argument(
        "--n-full-wide-threshold-b",
        "--candidate-n-full-wide-threshold",
        dest="n_full_wide_threshold_b",
        type=int,
        default=None,
        help="Candidate/B-only inclusive wide-budget gate (default: inherit shared).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--devices", default=None, help="Comma-separated CUDA devices for H2H workers.")
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument("--c-scale", type=float, default=0.03)
    parser.add_argument("--rescale-noise-floor-c", type=float, default=0.0)
    parser.add_argument("--sigma-eval", type=float, default=0.79)
    parser.add_argument(
        "--public-observation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--lazy-interior-chance", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--correct-rust-chance-spectra", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--symmetry-averaged-eval", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--wide-candidates-threshold", type=int, default=24)
    parser.add_argument("--symmetry-averaged-eval-threshold", type=int, default=None)
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
        devices=args.devices,
        max_decisions=int(args.max_decisions),
        max_depth=int(args.max_depth),
        c_visit=float(args.c_visit),
        c_scale=float(args.c_scale),
        rescale_noise_floor_c=float(args.rescale_noise_floor_c),
        sigma_eval=float(args.sigma_eval),
        public_observation=bool(args.public_observation),
        lazy_interior_chance=bool(args.lazy_interior_chance),
        correct_rust_chance_spectra=bool(args.correct_rust_chance_spectra),
        symmetry_averaged_eval=bool(args.symmetry_averaged_eval),
        n_full_wide=(int(args.n_full_wide) if args.n_full_wide is not None else None),
        n_full_wide_a=(
            int(args.n_full_wide_a) if args.n_full_wide_a is not None else None
        ),
        n_full_wide_b=(
            int(args.n_full_wide_b) if args.n_full_wide_b is not None else None
        ),
        n_full_wide_threshold=(
            int(args.n_full_wide_threshold)
            if args.n_full_wide_threshold is not None
            else None
        ),
        n_full_wide_threshold_a=(
            int(args.n_full_wide_threshold_a)
            if args.n_full_wide_threshold_a is not None
            else None
        ),
        n_full_wide_threshold_b=(
            int(args.n_full_wide_threshold_b)
            if args.n_full_wide_threshold_b is not None
            else None
        ),
        wide_candidates_threshold=int(args.wide_candidates_threshold),
        symmetry_averaged_eval_threshold=(
            int(args.symmetry_averaged_eval_threshold)
            if args.symmetry_averaged_eval_threshold is not None
            else None
        ),
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
