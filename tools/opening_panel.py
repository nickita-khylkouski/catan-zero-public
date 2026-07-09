#!/usr/bin/env python3
"""Frozen 200-root held-out opening panel (f70 D3).

The standing pre-H2H diagnostic for ANY (checkpoint, search-config) pair.
The Gate-A post-mortem established that a strength H2H is the only binding
ship signal, but H2H is expensive and slow; this panel is the cheap
early-warning screen that runs FIRST, on a frozen set of 200 wide
initial-placement roots (the widest, most noise-amplifying decisions in the
game -- 54 legal BUILD_SETTLEMENT candidates, per F8), drawn from a fresh
base-seed block (600001) never used by any other tool (the sigma trace uses
500001, the calibration holdout uses the 5.x/7.x million ranges).

Two subcommands:

  build   Scan seeds from --base-seed upward, keep the first --n-roots whose
          opening decision is a near-full-width settlement placement, and
          persist their RECONSTRUCTION SEEDS (not snapshots -- `Game.simple`
          is deterministic in the seed, so a seed list reconstructs the exact
          roots at zero storage cost and survives engine-wheel serialization
          changes) to runs/panels/opening_200.json.

  eval    Reconstruct the panel and evaluate one (checkpoint, search-config)
          pair, reporting per root and in aggregate:
            * harmful-flip proxy: does the search's argmax differ from the
              prior's argmax (the Gate-A failure signature)?
            * per-candidate raw-Q spread vs the evaluation noise floor
              (sigma_eval / sqrt(mean_visits)) -- is the spread the search is
              acting on real signal or sampling noise?
            * action-ranking quality vs a deeper-eval ORACLE over the top-K
              prior candidates: Kendall tau-b of the shallow ranking vs the
              oracle ranking, top-1 regret (oracle-value gap between the
              shallow pick and the oracle's best), and top-3 coverage
              (fraction of the oracle's top 3 also in the shallow top 3).

Oracle (`--oracle`): `deep_search` (default) applies each top-K candidate and
runs an n=`--oracle-sims` search from the resulting afterstate, taking its
root value (sign-adjusted to the root player) as the candidate's value -- a
genuine deeper-lookahead estimate that reuses the exact search machinery and
is cheaper/lower-variance than rollouts. `rollout` instead plays
`--oracle-rollouts` raw-policy games (sampling from the network's prior, with
our own seeded chance resolution) to terminal and averages the outcome -- an
independent Monte-Carlo signal, higher variance, more expensive. Both
evaluate each candidate independently.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.gumbel_self_play import _apply_selected_action
from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    _GNode,
)
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module
from factory_common import write_json

COLORS: tuple[str, ...] = ("RED", "BLUE")
PANEL_VERSION = 1


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_panel(
    catanatron_rs: Any,
    *,
    n_roots: int,
    base_seed: int,
    min_settlement_candidates: int,
) -> dict[str, Any]:
    """Return the panel descriptor: the reconstruction seeds of the first
    `n_roots` seeds (from `base_seed` upward) whose opening decision is a
    near-full-width settlement placement."""
    seeds: list[int] = []
    seed = base_seed
    scanned = 0
    while len(seeds) < n_roots:
        game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
        legal = json.loads(game.playable_actions_json())
        settlement_candidates = [a for a in legal if a[1] == "BUILD_SETTLEMENT"]
        if len(settlement_candidates) >= min_settlement_candidates:
            seeds.append(seed)
        seed += 1
        scanned += 1
    return {
        "version": PANEL_VERSION,
        "colors": list(COLORS),
        "base_seed": base_seed,
        "n_roots": n_roots,
        "min_settlement_candidates": min_settlement_candidates,
        "seeds": seeds,
        "seeds_scanned": scanned,
        "reconstruct": "catanatron_rs.Game.simple(colors, seed=<seed>)",
        "created_unix": int(time.time()),
    }


def reconstruct_roots(catanatron_rs: Any, panel: dict[str, Any]) -> list[Any]:
    colors = tuple(panel["colors"])
    return [catanatron_rs.Game.simple(list(colors), seed=int(s)) for s in panel["seeds"]]


# ---------------------------------------------------------------------------
# eval helpers
# ---------------------------------------------------------------------------
def _kendall_tau_b(x: list[float], y: list[float], *, eps: float = 1.0e-12) -> float | None:
    """Kendall tau-b (tie-corrected) between two score lists over the same
    items. None when a denominator is degenerate (e.g. all-tied)."""
    n = len(x)
    if n < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            tie_x = abs(dx) <= eps
            tie_y = abs(dy) <= eps
            if tie_x:
                ties_x += 1
            if tie_y:
                ties_y += 1
            if tie_x or tie_y:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - ties_x) * (n0 - ties_y))
    if denom <= 0.0:
        return None
    return (concordant - discordant) / denom


def _shallow_root_trace(mcts: GumbelChanceMCTS, game: Any) -> dict[str, Any]:
    """Run one shallow search and return the per-candidate selection basis
    (mirrors tools/sigma_trace_placement_root.trace_one_root but also returns
    the objects the panel's oracle comparison needs)."""
    root_color = str(game.current_color())
    root = _GNode(game=game.copy(), root_color=root_color)
    mcts._expand(root)
    mcts._run_root_search(root, int(mcts.config.n_full))

    completed_q = mcts._completed_q(root)
    rescaled_q = mcts._rescaled_completed_q(root, completed_q)
    scale = mcts._sigma_scale(root)
    logits = root.action_logits

    per_candidate: dict[int, dict[str, float]] = {}
    for action_id, stats in root.actions.items():
        per_candidate[action_id] = {
            "prior": float(stats.prior),
            "logit": float(logits.get(action_id, 0.0)),
            "visits": int(stats.visits),
            "raw_q": float(completed_q.get(action_id, 0.0)),
            "rescaled_q": float(rescaled_q.get(action_id, 0.0)),
            "ranking_score": float(logits.get(action_id, 0.0) + scale * rescaled_q.get(action_id, 0.0)),
        }
    return {"root_color": root_color, "per_candidate": per_candidate}


def _oracle_value_deep_search(
    game: Any, action_id: int, root_color: str, *, evaluator: Any, oracle_sims: int, seed: int
) -> float:
    """Apply `action_id`, run an n=`oracle_sims` search from the afterstate,
    and return its value from `root_color`'s perspective."""
    child = game.copy()
    child = _apply_selected_action(
        child, int(action_id), colors=COLORS, rng=random.Random(seed)
    )
    winner = child.winning_color()
    if winner is not None:
        return 1.0 if str(winner) == root_color else -1.0
    config = GumbelChanceMCTSConfig(
        colors=COLORS,
        seed=seed,
        n_full=oracle_sims,
        n_fast=oracle_sims,
        p_full=1.0,
        temperature=0.0,
    )
    oracle = GumbelChanceMCTS(config, evaluator)
    result = oracle.search(child, force_full=True)
    value = float(result.root_value)
    # root_value is from the afterstate's acting player's perspective.
    return value if str(child.current_color()) == root_color else -value


def _oracle_value_rollout(
    game: Any,
    action_id: int,
    root_color: str,
    *,
    evaluator: Any,
    n_rollouts: int,
    max_steps: int,
    seed: int,
) -> float:
    """Apply `action_id`, then average the terminal outcome of `n_rollouts`
    raw-policy rollouts (sampling from the network prior) with seeded chance
    resolution."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_rollouts):
        g = game.copy()
        g = _apply_selected_action(g, int(action_id), colors=COLORS, rng=rng)
        steps = 0
        while g.winning_color() is None and steps < max_steps:
            legal = tuple(int(a) for a in g.playable_action_indices(list(COLORS), None))
            if not legal:
                break
            acting = str(g.current_color())
            priors, _value = evaluator.evaluate(g, legal, root_color=acting, colors=COLORS)
            chosen = _sample_from_priors(priors, legal, rng)
            g = _apply_selected_action(g, chosen, colors=COLORS, rng=rng)
            steps += 1
        winner = g.winning_color()
        total += (1.0 if str(winner) == root_color else -1.0) if winner is not None else 0.0
    return total / float(n_rollouts)


def _sample_from_priors(priors: dict[int, float], legal: tuple[int, ...], rng: random.Random) -> int:
    items = [(a, float(priors.get(a, 0.0))) for a in legal]
    total = sum(p for _a, p in items)
    if total <= 0.0:
        return int(rng.choice(legal))
    draw = rng.random() * total
    cumulative = 0.0
    for action_id, p in items:
        cumulative += p
        if draw <= cumulative:
            return int(action_id)
    return int(items[-1][0])


def evaluate_root(
    mcts: GumbelChanceMCTS,
    evaluator: Any,
    game: Any,
    *,
    top_k: int,
    oracle: str,
    oracle_sims: int,
    oracle_rollouts: int,
    rollout_max_steps: int,
    seed: int,
) -> dict[str, Any]:
    trace = _shallow_root_trace(mcts, game)
    root_color = trace["root_color"]
    per = trace["per_candidate"]

    prior_argmax = max(per, key=lambda a: per[a]["prior"])
    search_argmax = max(per, key=lambda a: per[a]["ranking_score"])

    # Raw-Q spread vs noise floor (over VISITED candidates -- unvisited carry
    # v_mix, which would understate the true acted-on spread).
    visited = [a for a in per if per[a]["visits"] > 0]
    raw_qs = [per[a]["raw_q"] for a in visited] or [0.0]
    raw_spread = max(raw_qs) - min(raw_qs)
    mean_visits = (sum(per[a]["visits"] for a in per) / len(per)) if per else 0.0
    noise_floor = (
        float(mcts.config.sigma_eval) / math.sqrt(mean_visits) if mean_visits > 0 else float("inf")
    )
    spread_over_floor = (raw_spread / noise_floor) if noise_floor > 0 and math.isfinite(noise_floor) else None

    # Oracle ranking over the top-K prior candidates.
    top_candidates = sorted(per, key=lambda a: per[a]["prior"], reverse=True)[:top_k]
    oracle_values: dict[int, float] = {}
    for i, action_id in enumerate(top_candidates):
        if oracle == "rollout":
            oracle_values[action_id] = _oracle_value_rollout(
                game,
                action_id,
                root_color,
                evaluator=evaluator,
                n_rollouts=oracle_rollouts,
                max_steps=rollout_max_steps,
                seed=seed * 1_000_003 + i,
            )
        else:
            oracle_values[action_id] = _oracle_value_deep_search(
                game,
                action_id,
                root_color,
                evaluator=evaluator,
                oracle_sims=oracle_sims,
                seed=seed * 1_000_003 + i,
            )

    shallow_scores = [per[a]["ranking_score"] for a in top_candidates]
    oracle_scores = [oracle_values[a] for a in top_candidates]
    tau = _kendall_tau_b(shallow_scores, oracle_scores)

    shallow_best = max(top_candidates, key=lambda a: per[a]["ranking_score"])
    oracle_best = max(top_candidates, key=lambda a: oracle_values[a])
    top1_regret = oracle_values[oracle_best] - oracle_values[shallow_best]

    k3 = min(3, len(top_candidates))
    shallow_top3 = set(sorted(top_candidates, key=lambda a: per[a]["ranking_score"], reverse=True)[:k3])
    oracle_top3 = set(sorted(top_candidates, key=lambda a: oracle_values[a], reverse=True)[:k3])
    top3_coverage = len(shallow_top3 & oracle_top3) / float(k3) if k3 else None

    return {
        "n_candidates": len(per),
        "n_visited": len(visited),
        "prior_argmax": int(prior_argmax),
        "search_argmax": int(search_argmax),
        "flipped": bool(prior_argmax != search_argmax),
        "raw_q_spread": float(raw_spread),
        "noise_floor": None if math.isinf(noise_floor) else float(noise_floor),
        "spread_over_floor": spread_over_floor,
        "mean_visits": float(mean_visits),
        "kendall_tau": tau,
        "top1_regret": float(top1_regret),
        "top3_coverage": top3_coverage,
        "oracle_best_in_shallow_top3": bool(oracle_best in shallow_top3),
    }


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def aggregate(root_reports: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(root_reports)
    return {
        "n_roots": n,
        "flip_rate": _mean([1.0 if r["flipped"] else 0.0 for r in root_reports]),
        "mean_raw_q_spread": _mean([r["raw_q_spread"] for r in root_reports]),
        "mean_spread_over_floor": _mean([r["spread_over_floor"] for r in root_reports]),
        "mean_kendall_tau": _mean([r["kendall_tau"] for r in root_reports]),
        "mean_top1_regret": _mean([r["top1_regret"] for r in root_reports]),
        "mean_top3_coverage": _mean([r["top3_coverage"] for r in root_reports]),
    }


def _build_config(args: Any) -> GumbelChanceMCTSConfig:
    return GumbelChanceMCTSConfig(
        colors=COLORS,
        n_full=int(args.n_full),
        n_fast=int(args.n_full),
        p_full=1.0,
        temperature=0.0,
        c_visit=float(args.c_visit),
        c_scale=float(args.c_scale),
        prior_temperature=float(args.prior_temperature),
        rescale_noise_floor_c=float(args.rescale_noise_floor_c),
        sigma_eval=float(args.sigma_eval),
        variance_aware_q=bool(args.variance_aware_q),
        variance_aware_k=float(args.variance_aware_k),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build and persist the frozen panel")
    p_build.add_argument("--out", default="runs/panels/opening_200.json")
    p_build.add_argument("--n-roots", type=int, default=200)
    p_build.add_argument("--base-seed", type=int, default=600001)
    p_build.add_argument("--min-settlement-candidates", type=int, default=40)

    p_eval = sub.add_parser("eval", help="evaluate a (checkpoint, search-config) pair on the panel")
    p_eval.add_argument("--panel", default="runs/panels/opening_200.json")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--out", required=True)
    p_eval.add_argument("--max-roots", type=int, default=None, help="subsample the panel for a quick run")
    p_eval.add_argument("--seed", type=int, default=600001)
    # search-config knobs
    p_eval.add_argument("--n-full", type=int, default=64)
    p_eval.add_argument("--c-visit", type=float, default=50.0)
    p_eval.add_argument("--c-scale", type=float, default=0.1)
    p_eval.add_argument("--prior-temperature", type=float, default=1.0)
    p_eval.add_argument("--rescale-noise-floor-c", type=float, default=0.0)
    p_eval.add_argument("--sigma-eval", type=float, default=0.79)
    p_eval.add_argument("--variance-aware-q", action="store_true")
    p_eval.add_argument("--variance-aware-k", type=float, default=1.0)
    # oracle knobs
    p_eval.add_argument("--oracle", choices=("deep_search", "rollout"), default="deep_search")
    p_eval.add_argument("--oracle-sims", type=int, default=256)
    p_eval.add_argument("--oracle-rollouts", type=int, default=32)
    p_eval.add_argument("--rollout-max-steps", type=int, default=400)
    p_eval.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args()
    catanatron_rs = _require_rust_module()

    if args.command == "build":
        panel = build_panel(
            catanatron_rs,
            n_roots=int(args.n_roots),
            base_seed=int(args.base_seed),
            min_settlement_candidates=int(args.min_settlement_candidates),
        )
        write_json(args.out, panel)
        print(
            json.dumps(
                {k: v for k, v in panel.items() if k != "seeds"}
                | {"seeds_head": panel["seeds"][:5], "seeds_len": len(panel["seeds"])},
                indent=2,
                sort_keys=True,
            )
        )
        return

    # eval
    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    roots = reconstruct_roots(catanatron_rs, panel)
    if args.max_roots is not None:
        roots = roots[: int(args.max_roots)]

    evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
        args.checkpoint, device=args.device, config=EntityGraphRustEvaluatorConfig()
    )
    config = _build_config(args)
    t0 = time.time()
    try:
        root_reports = []
        for i, game in enumerate(roots):
            mcts = GumbelChanceMCTS(
                dataclasses.replace(config, seed=int(args.seed) + i),
                evaluator,
            )
            root_reports.append(
                evaluate_root(
                    mcts,
                    evaluator,
                    game.copy(),
                    top_k=int(args.top_k),
                    oracle=args.oracle,
                    oracle_sims=int(args.oracle_sims),
                    oracle_rollouts=int(args.oracle_rollouts),
                    rollout_max_steps=int(args.rollout_max_steps),
                    seed=int(args.seed) + i,
                )
            )
    finally:
        evaluator.close()
    elapsed = time.time() - t0

    summary = {
        "checkpoint": args.checkpoint,
        "panel": args.panel,
        "n_roots_evaluated": len(root_reports),
        "n_roots_in_panel": len(panel["seeds"]),
        "oracle": args.oracle,
        "oracle_sims": int(args.oracle_sims),
        "oracle_rollouts": int(args.oracle_rollouts),
        "search_config": {
            "n_full": int(args.n_full),
            "c_visit": float(args.c_visit),
            "c_scale": float(args.c_scale),
            "prior_temperature": float(args.prior_temperature),
            "rescale_noise_floor_c": float(args.rescale_noise_floor_c),
            "sigma_eval": float(args.sigma_eval),
            "variance_aware_q": bool(args.variance_aware_q),
            "variance_aware_k": float(args.variance_aware_k),
        },
        "elapsed_seconds": elapsed,
        "seconds_per_root": elapsed / len(root_reports) if root_reports else None,
        "aggregate": aggregate(root_reports),
        "per_root": root_reports,
    }
    write_json(args.out, summary)
    printable = {k: v for k, v in summary.items() if k != "per_root"}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
