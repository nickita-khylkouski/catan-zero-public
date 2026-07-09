"""Disjoint base-seed planning and verification for multi-host fleet
launches (task #77).

Root cause this fixes: fleet launches (H2H arms, gen-1 generation) have
historically assigned each worker's --base-seed via independent per-host
arithmetic authored by hand (e.g. one formula for host A's GPUs, a
DIFFERENT formula for host B's GPUs). Two independently-authored formulas
can silently overlap -- this is exactly what happened to the v3b_base
confirmation-H2H arm (A100A gpu4-7 and A100B gpu0-3 both got seeds
314000-317015, a bit-for-bit duplicate 64-pair sample instead of 128
independent pairs) and, separately, to the *staged but never-fired*
gen-1 generation base-seed scheme in the #76 refresh doc (A100A's
9_100_001+i*100_000 collides with A100B's 9_200_001+i*100_000 at seven of
eight GPU indices).

Use `assert_disjoint_seed_blocks` to validate ANY existing/proposed
per-worker seed assignment before firing a fleet, regardless of how the
seeds were derived. Use `plan_disjoint_seed_blocks` to GENERATE a
provably-disjoint assignment from a single global counter, which is
structurally incapable of the copy-paste per-host-formula bug because
there is only one formula, applied once per worker in sequence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def assert_disjoint_seed_blocks(workers: list[tuple[str, int, int]]) -> None:
    """Raise ValueError if any two workers' [base_seed, base_seed+games)
    half-open ranges overlap.

    `workers` is a list of (worker_id, base_seed, games) triples. A worker
    with games=0 occupies no seed range and is ignored. Adjacent blocks
    that touch at a boundary (one ends exactly where the next begins) are
    NOT overlapping.
    """
    intervals = [
        (worker_id, base_seed, base_seed + games)
        for worker_id, base_seed, games in workers
        if games > 0
    ]
    # Sort by start so overlap detection is a single adjacent-pair scan
    # (O(n log n) instead of O(n^2), and finds the actual offending pair).
    intervals.sort(key=lambda item: item[1])
    for (id_a, start_a, end_a), (id_b, start_b, end_b) in zip(intervals, intervals[1:]):
        if start_b < end_a:
            raise ValueError(
                f"seed range collision: {id_a!r} covers [{start_a}, {end_a}) and "
                f"{id_b!r} covers [{start_b}, {end_b}) -- these overlap. "
                "Every worker in a fleet launch must get a disjoint base-seed "
                "block; use plan_disjoint_seed_blocks() to generate one instead "
                "of hand-deriving per-host formulas."
            )


def plan_disjoint_seed_blocks(
    worker_ids: list[str], *, games_per_worker: int, base: int, block_size: int
) -> dict[str, int]:
    """Assign each worker_id a base-seed from a SINGLE global counter, so the
    result is disjoint by construction rather than by having to verify two
    independently-authored per-host formulas against each other.

    Returns {worker_id: base_seed}, in the same order as `worker_ids`.
    """
    if block_size < games_per_worker:
        raise ValueError(
            f"block_size ({block_size}) must be >= games_per_worker "
            f"({games_per_worker}), or workers would run out of their own "
            "seed block and spill into the next worker's range."
        )
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError(f"duplicate worker_ids in {worker_ids!r}")

    plan = {worker_id: base + i * block_size for i, worker_id in enumerate(worker_ids)}
    return plan


def _parse_worker_ids(raw: str) -> list[str]:
    return [w.strip() for w in raw.split(",") if w.strip()]


def _cmd_plan(args) -> int:
    worker_ids = _parse_worker_ids(args.worker_ids)
    plan = plan_disjoint_seed_blocks(
        worker_ids,
        games_per_worker=args.games_per_worker,
        base=args.base,
        block_size=args.block_size,
    )
    # Defense-in-depth: verify the plan we just constructed, even though
    # plan_disjoint_seed_blocks() is disjoint by construction. Catches any
    # future bug in the generator itself before it reaches a launcher.
    assert_disjoint_seed_blocks([(wid, seed, args.games_per_worker) for wid, seed in plan.items()])
    out_path = Path(args.out) if args.out else None
    payload = json.dumps(plan, indent=2, sort_keys=False)
    if out_path is not None:
        out_path.write_text(payload)
    print(payload)
    return 0


def _cmd_verify(args) -> int:
    table = json.loads(Path(args.seeds_json).read_text())
    workers = [(worker_id, int(seed), args.games_per_worker) for worker_id, seed in table.items()]
    try:
        assert_disjoint_seed_blocks(workers)
    except ValueError as exc:
        print(f"SEED COLLISION: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(workers)} workers, all seed blocks disjoint.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan", help="generate a provably-disjoint base-seed table from a single global counter"
    )
    plan_parser.add_argument("--worker-ids", required=True, help="comma-separated worker ids")
    plan_parser.add_argument("--games-per-worker", type=int, required=True)
    plan_parser.add_argument("--base", type=int, required=True)
    plan_parser.add_argument("--block-size", type=int, required=True)
    plan_parser.add_argument("--out", default=None, help="optional path to write the {worker_id: base_seed} JSON table")

    verify_parser = subparsers.add_parser(
        "verify", help="assert an existing {worker_id: base_seed} JSON table is pairwise-disjoint"
    )
    verify_parser.add_argument("--seeds-json", required=True)
    verify_parser.add_argument("--games-per-worker", type=int, required=True, help="uniform games count assumed for every worker in the table")

    args = parser.parse_args()
    if args.command == "plan":
        return _cmd_plan(args)
    return _cmd_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
