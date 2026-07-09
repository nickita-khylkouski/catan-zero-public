#!/usr/bin/env python3
"""CAT-72: profile-driven optimization sweep harness (post-Rust-featurizer).

Governing rule (CAT-72's decision rule, restated): "profiler-confirmed
before build -- no speculative optimization." This tool exists so that rule
is actually checkable, not just asserted: it runs the CAT-71 leaf profiler
(tools/perf_snapshot.py `leaf` mode) TWICE, back-to-back, on the identical
deterministic leaf-state sequence -- once through the legacy Python
featurize path, once through the native Rust path (task #81/CAT-65,
`config.rust_featurize`) -- and reports the before/after stage breakdown so
the new top-cost item can be named from data instead of assumed.

Why this had to be a new tool rather than just re-running `perf_snapshot.py
leaf` twice by hand: `perf_snapshot.py leaf` had no `--rust-featurize` flag
before CAT-72 (it always profiled the legacy path), and its stage split
left `_fetch_leaf_decision_inputs`/`rust_policy_action_ids`/
`_resolve_entity_adapter` uncounted in an opaque "postprocess" residual --
exactly the kind of hidden cost the Rust featurizer's speedup could expose
as the new bottleneck. Both gaps are fixed in `perf_snapshot.profile_leaf_eval`
itself (shared by this tool); this tool adds the A/B diff + ranking on top.

Usage (dev smoke test, tiny fast policy, fast, CPU):
    tools/opt_sweep.py --device cpu --num-evals 200

Usage (real baseline reproduction, host-only, needs a real checkpoint):
    tools/opt_sweep.py --device cuda --checkpoint runs/self_play/champions/current_best.pt --num-evals 200 --out runs/perf/opt_sweep_report.json

Interpreting the output:
    `ranked_stages_after` orders the non-nn_forward, non-total named stages
    by their AFTER (rust_featurize=True) share of total per-leaf cost --
    the top entry is the fresh profile's answer to "what's the new
    bottleneck", which is exactly what CAT-72 step 3/4 requires before any
    of the four candidate optimizations get built.

    `nn_forward` is excluded from the ranking on purpose whenever
    --checkpoint is not given: the tiny fast policy's forward pass is
    deliberately unrepresentative (see `perf_snapshot._baseline_check`'s
    docstring), so ranking it against real per-leaf costs would be
    comparing a real measurement to a known-fake one. Pass --checkpoint for
    a run where nn_forward belongs in the ranking too.
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

import perf_common  # noqa: E402
import perf_snapshot  # noqa: E402


def run_ab_sweep(
    *,
    num_evals: int,
    seed: int,
    device: str,
    checkpoint: str | None,
    public_observation: bool,
) -> dict[str, Any]:
    """Run the leaf profiler once per path and return a before/after report.

    Both runs use the same `seed`/`num_evals`, and `_collect_leaf_states` is
    deterministic given those, so both profile the identical leaf-state
    sequence -- an apples-to-apples A/B, not two independent samples.
    """
    before = perf_snapshot.profile_leaf_eval(
        num_evals=num_evals,
        seed=seed,
        device=device,
        checkpoint=checkpoint,
        public_observation=public_observation,
        rust_featurize=False,
    )
    after = perf_snapshot.profile_leaf_eval(
        num_evals=num_evals,
        seed=seed,
        device=device,
        checkpoint=checkpoint,
        public_observation=public_observation,
        rust_featurize=True,
    )

    rank_excludes = {"total", "postprocess"}
    has_real_checkpoint = bool(checkpoint)
    if not has_real_checkpoint:
        # tiny fast policy's forward pass is not a real measurement -- see
        # module docstring. Keep it in the raw stage dump, just not the
        # ranking used to name "the new bottleneck".
        rank_excludes.add("nn_forward")

    ranked_after = sorted(
        (
            (name, pct)
            for name, pct in after["stage_pct_of_total"].items()
            if name not in rank_excludes
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    stage_deltas = {}
    for name in perf_snapshot._NAMED_STAGES:
        before_ms = before["stages"][name]["total_ms"]
        after_ms = after["stages"][name]["total_ms"]
        speedup = (before_ms / after_ms) if after_ms > 1.0e-9 else float("inf")
        stage_deltas[name] = {
            "before_total_ms": before_ms,
            "after_total_ms": after_ms,
            "speedup": speedup,
            "before_pct_of_total": before["stage_pct_of_total"][name],
            "after_pct_of_total": after["stage_pct_of_total"][name],
        }

    total_before = before["stages"]["total"]["mean_ms"]
    total_after = after["stages"]["total"]["mean_ms"]

    return {
        "kind": "opt_sweep_ab",
        "num_evals": num_evals,
        "seed": seed,
        "device": device,
        "checkpoint": checkpoint,
        "has_real_checkpoint": has_real_checkpoint,
        "before": before,
        "after": after,
        "stage_deltas": stage_deltas,
        "ranked_stages_after": ranked_after,
        "top_cost_post_rust": ranked_after[0][0] if ranked_after else None,
        "total_mean_ms_before": total_before,
        "total_mean_ms_after": total_after,
        "total_speedup": (total_before / total_after) if total_after > 1.0e-9 else float("inf"),
        "topology_bootstrap_ms": after.get("topology_bootstrap_ms"),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"device={report['device']} num_evals={report['num_evals']} "
          f"checkpoint={report['checkpoint'] or '(fast_policy, dev smoke test)'}")
    print()
    print(f"{'stage':20s} {'before ms':>12s} {'after ms':>12s} {'speedup':>10s} "
          f"{'before %':>10s} {'after %':>10s}")
    for name, d in report["stage_deltas"].items():
        speedup_str = f"{d['speedup']:.1f}x" if d["speedup"] != float("inf") else "inf"
        print(f"{name:20s} {d['before_total_ms']:12.2f} {d['after_total_ms']:12.2f} "
              f"{speedup_str:>10s} {d['before_pct_of_total']:9.2f}% {d['after_pct_of_total']:9.2f}%")
    print()
    print(f"total mean ms/leaf: before={report['total_mean_ms_before']:.3f}  "
          f"after={report['total_mean_ms_after']:.3f}  speedup={report['total_speedup']:.2f}x")
    if report["topology_bootstrap_ms"]:
        print(f"topology bootstrap (one-time, rust path only): "
              f"{report['topology_bootstrap_ms']['mean_ms']:.3f}ms")
    print()
    print("ranked AFTER stages (excludes total/postprocess"
          + ("" if report["has_real_checkpoint"] else "/nn_forward [fast_policy, unrepresentative]")
          + "):")
    for name, pct in report["ranked_stages_after"]:
        print(f"  {name:20s} {pct:6.2f}%")
    print()
    print(f"=> new top non-NN cost post-Rust-featurizer: {report['top_cost_post_rust']!r}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-evals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--public-observation", action="store_true")
    ap.add_argument("--out", default=None, help="write the full JSON report to this path")
    ap.add_argument("--ledger", default=str(perf_common.DEFAULT_LEDGER_PATH),
                     help="also append both the before/after leaf rows to the standing perf ledger")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args(argv)

    report = run_ab_sweep(
        num_evals=args.num_evals,
        seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
        public_observation=args.public_observation,
    )

    if not args.no_ledger:
        perf_common.append_ledger_rows(Path(args.ledger), [report["before"], report["after"]])

    _print_report(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()
