"""Bulk CLI runner for the Rust <-> Python engine equivalence harness.

Plays N games through both `catanatron_rs` and the vendored Python reference
engine (`vendor/catanatron`) side by side -- forcing identical dice rolls,
robber steals, and development-card draws via each engine's chance API -- and
reports every legal-action-set mismatch or state divergence found, with full
repro info (seed, step index, action, both engines' states).

See `src/catan_zero/adapters/engine_equivalence.py` for how map/seating
alignment and chance-outcome forcing work, and `tests/test_engine_equivalence.py`
for genuine engine bugs already found this way. Divergences are bucketed by
`topic`: anything touching longest-road length/ownership or BUILD_ROAD
legality near an enemy-occupied node is tagged `rules_adjudication_needed_*`
rather than blamed on one engine, per the 2026-07-02 rules audit that found
pre-existing Python-side longest-road bugs (upstream issues #376, #378) --
either engine could be right/wrong there. Dice, resources, dev cards, robber,
bank, current player, and win detection are unaffected by that audit.

Usage:
    .venv/bin/python tools/engine_equivalence_sweep.py --games 1000 --out runs/engine_equivalence_report.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factory_common import write_json  # noqa: E402

from catan_zero.adapters.engine_equivalence import (  # noqa: E402
    EquivalenceConfig,
    RustModuleUnavailable,
    require_rust_module,
    run_sweep,
)


def _git_snapshot() -> dict[str, Any]:
    """Capture the repo's commit hash + a hash of its working-tree diff, to
    catch the mid-sweep-commit staleness failure mode: a long sweep imports
    Python source once at process start and never re-imports it, so if
    someone commits (or otherwise edits tracked files) while the sweep is
    still running, the back half of the sweep silently runs against stale
    code with no error -- only a start/end mismatch reveals it. Hashing the
    diff (not just a dirty bool) means an *already*-dirty-but-unchanging
    worktree doesn't trigger a false "changed mid-sweep" warning.
    """
    import hashlib

    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        diff_text = subprocess.run(
            ["git", "diff", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=30
        ).stdout
        status_text = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, timeout=10
        ).stdout
        dirty = bool(diff_text.strip()) or bool(status_text.strip())
        diff_hash = hashlib.sha256((diff_text + status_text).encode("utf-8")).hexdigest()
        return {"commit": commit or "unknown", "dirty": dirty, "diff_hash": diff_hash}
    except Exception as error:  # pragma: no cover - defensive, git may be unavailable.
        return {"commit": "unknown", "dirty": False, "diff_hash": "unknown", "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--colors", default="RED,BLUE")
    parser.add_argument("--map-kind", default="TOURNAMENT")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--discard-limit", type=int, default=7)
    parser.add_argument("--friendly-robber", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--out", default="runs/engine_equivalence_report.json")
    parser.add_argument("--max-divergences-shown", type=int, default=25)
    args = parser.parse_args()

    try:
        require_rust_module()
    except RustModuleUnavailable as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

    config = EquivalenceConfig(
        colors=tuple(c.strip() for c in args.colors.split(",")),
        map_kind=args.map_kind,
        vps_to_win=args.vps_to_win,
        discard_limit=args.discard_limit,
        friendly_robber=bool(args.friendly_robber),
        max_steps=args.max_steps,
    )

    git_at_start = _git_snapshot()
    if git_at_start["dirty"]:
        print(
            "WARNING: repo has uncommitted changes at sweep start -- results may not "
            "correspond to any single reviewable commit.",
            file=sys.stderr,
        )

    start = time.perf_counter()
    report = run_sweep(
        num_games=args.games,
        start_seed=args.start_seed,
        config=config,
        progress_every=args.progress_every,
    )
    elapsed = time.perf_counter() - start

    git_at_end = _git_snapshot()
    stale_mid_sweep = (
        git_at_start["commit"] != git_at_end["commit"]
        or git_at_start["diff_hash"] != git_at_end["diff_hash"]
    )
    if stale_mid_sweep:
        print(
            f"WARNING: git state changed during the sweep (start={git_at_start!r} "
            f"end={git_at_end!r}). This process never re-imports already-loaded Python "
            "modules, so if anyone committed or edited tracked files while this sweep was "
            "running, some prefix of these games ran against the start-of-sweep code and "
            "the rest may have run against something different (only matters if edits "
            "touched files this sweep actually imports, e.g. vendor/catanatron). Treat "
            "this report's results with that in mind and consider rerunning on a clean "
            "worktree.",
            file=sys.stderr,
        )

    payload = report.to_dict()
    payload["elapsed_seconds"] = elapsed
    payload["git_at_start"] = git_at_start
    payload["git_at_end"] = git_at_end
    payload["git_stale_mid_sweep"] = stale_mid_sweep
    payload["config"] = {
        "colors": list(config.colors),
        "map_kind": config.map_kind,
        "vps_to_win": config.vps_to_win,
        "discard_limit": config.discard_limit,
        "friendly_robber": config.friendly_robber,
        "max_steps": config.max_steps,
    }
    write_json(args.out, payload)

    print(f"\nEngine equivalence sweep: {args.games} games in {elapsed:.1f}s "
          f"({args.games / max(elapsed, 1e-9):.2f} games/sec)")
    print(f"Git at start: {git_at_start}")
    print(f"Git at end:   {git_at_end}")
    print(f"Outcomes: {report.outcomes}")
    print(f"Topics: {report.topics}")
    print(f"Completed-game winners: {report.completed_winners}")
    print(f"Divergences found: {len(report.divergences)}")
    for item in report.divergences[: args.max_divergences_shown]:
        print(
            f"  seed={item.seed} step={item.steps} outcome={item.outcome} "
            f"topic={item.topic} detail={item.detail}"
        )
    if len(report.divergences) > args.max_divergences_shown:
        print(f"  ... and {len(report.divergences) - args.max_divergences_shown} more (see {args.out})")
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
