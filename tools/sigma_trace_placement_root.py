#!/usr/bin/env python3
"""Diagnostic tool (landed permanently, Gate A mechanism analysis 2026-07-04):
controlled sigma trace on real 54-wide placement roots.

For each of several real initial-placement decisions (54 legal
BUILD_SETTLEMENT candidates -- the widest root in the game, per F8), runs a
full n_full-simulation Gumbel-Top-k + Sequential Halving search and records,
per candidate: prior, visits, raw completed-Q, rescaled completed-Q, and the
final ranking score (logits + sigma(rescaled_q)) used by
_improved_policy/argmax. Compares argmax(ranking score) against argmax(prior
alone) to quantify how often the search's Q term overrides the prior, across
a sweep of (c_visit, c_scale) configs.

Findings on the shipped (c_visit=50, c_scale=0.1) config (2026-07-04, 40
states, n_full=64): 72.5% of placement roots flip the argmax away from the
prior, dropping to 62.5% at c_scale=0.03 (the arm that removed Gate A's net
harm) -- with avg_n_visited == n_candidates (54), i.e. ~1.2 sims/candidate.
Root mechanism: the min-max rescale in `_rescale_completed_q` stretches
WHATEVER spread it observes to fill [0, 1], so a raw completed-Q spread
that's pure sampling noise (e.g. -0.03 to +0.01 across 2-4 samples/candidate)
gets treated identically to a genuinely separated true-value spread -- it
manufactures false confidence that then swamps a near-flat prior among the
top few candidates. Re-run this tool after any change intended to fix that
(value-head repair, SNR-budget changes, a noise-floor-aware rescale) to
verify the raw-Q spreads at these roots become real signal rather than
noise, and that the flip rate drops meaningfully below the noise-driven
62-72% baseline above.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTS, GumbelChanceMCTSConfig, _GNode
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module
from factory_common import write_json

COLORS: tuple[str, ...] = ("RED", "BLUE")


def find_placement_roots(catanatron_rs, *, n_states: int, base_seed: int) -> list[Any]:
    """Real initial-placement game states (54 legal BUILD_SETTLEMENT
    candidates), one per seed, at the very first placement decision."""
    states = []
    seed = base_seed
    while len(states) < n_states:
        game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
        legal = json.loads(game.playable_actions_json())
        settlement_candidates = [a for a in legal if a[1] == "BUILD_SETTLEMENT"]
        if len(settlement_candidates) >= 40:  # near-full-width placement root
            states.append(game)
        seed += 1
    return states


def trace_one_root(mcts: GumbelChanceMCTS, game: Any) -> dict[str, Any]:
    """Run an explicitly authoritative diagnostic and extract private internals.

    This tool studies one tree's completed-Q rescaling and therefore cannot
    represent a cross-determinization information-set aggregate.  Refuse a
    public-observation evaluator rather than letting the private ``_GNode``
    path masquerade as public-information evidence.  ``opening_panel.py`` is
    the information-set-safe replacement for public checkpoint diagnostics.
    """
    evaluator_config = getattr(mcts.evaluator, "config", None)
    if evaluator_config is not None and bool(
        getattr(evaluator_config, "public_observation", False)
    ):
        raise RuntimeError(
            "sigma_trace_placement_root uses authoritative private-tree internals "
            "and refuses public_observation; use opening_panel.py for public evidence"
        )
    root_color = str(game.current_color())
    root = _GNode(game=game.copy(), root_color=root_color)
    mcts._expand(root)

    sh_winner_action, used = mcts._run_root_search(root, int(mcts.config.n_full))

    completed_q = mcts._completed_q(root)
    rescaled_q = mcts._rescale_completed_q(completed_q)
    scale = mcts._sigma_scale(root)
    logits = root.action_logits

    per_candidate = []
    for action_id, stats in root.actions.items():
        ranking_score = logits.get(action_id, 0.0) + scale * rescaled_q.get(action_id, 0.0)
        per_candidate.append(
            {
                "action_id": action_id,
                "prior": stats.prior,
                "logit": logits.get(action_id, 0.0),
                "visits": stats.visits,
                "raw_q": completed_q.get(action_id, 0.0),
                "rescaled_q": rescaled_q.get(action_id, 0.0),
                "ranking_score": ranking_score,
            }
        )

    prior_argmax = max(per_candidate, key=lambda c: c["logit"])["action_id"]
    search_argmax = max(per_candidate, key=lambda c: c["ranking_score"])["action_id"]
    visited = [c for c in per_candidate if c["visits"] > 0]

    return {
        "information_regime": "authoritative_hidden_state_diagnostic",
        "admissible_for_public_information_evidence": False,
        "n_candidates": len(per_candidate),
        "n_visited": len(visited),
        "sh_winner_action": int(sh_winner_action),
        "simulations_used": int(used),
        "prior_argmax": prior_argmax,
        "search_argmax": search_argmax,
        "flipped": prior_argmax != search_argmax,
        "per_candidate": per_candidate,
    }


def run_sweep(
    states: list[Any],
    evaluator: Any,
    *,
    configs: tuple[tuple[float, float], ...],
    n_full: int,
    seed_base: int,
) -> dict[str, Any]:
    """`configs` is a tuple of (c_visit, c_scale) pairs to sweep -- both are
    knobs on the SAME sigma transform (sigma = (c_visit + max_visits) *
    c_scale * rescaled_q), so both are relevant to "does this root's argmax
    get flipped by noise" and worth sweeping independently, not just
    c_scale."""
    results: dict[str, Any] = {}
    for c_visit, c_scale in configs:
        traces = []
        for i, game in enumerate(states):
            config = GumbelChanceMCTSConfig(
                colors=COLORS,
                seed=seed_base + i,
                n_full=n_full,
                n_fast=n_full,
                p_full=1.0,
                c_visit=c_visit,
                c_scale=c_scale,
                max_depth=80,
                temperature=0.0,
            )
            mcts = GumbelChanceMCTS(config, evaluator)
            traces.append(trace_one_root(mcts, game.copy()))
        flips = sum(1 for t in traces if t["flipped"])
        avg_visited = sum(t["n_visited"] for t in traces) / len(traces)
        avg_candidates = sum(t["n_candidates"] for t in traces) / len(traces)
        key = f"cv{c_visit}_cs{c_scale}"
        results[key] = {
            "information_regime": "authoritative_hidden_state_diagnostic",
            "admissible_for_public_information_evidence": False,
            "c_visit": c_visit,
            "c_scale": c_scale,
            "n_states": len(traces),
            "flips": flips,
            "flip_rate": flips / len(traces),
            "avg_n_candidates": avg_candidates,
            "avg_n_visited": avg_visited,
            "traces": traces,
        }
    return results


def _parse_configs(raw: str) -> tuple[tuple[float, float], ...]:
    """Parse "cv:cs,cv:cs,..." into a tuple of (c_visit, c_scale) pairs."""
    configs = []
    for entry in raw.split(","):
        cv_str, cs_str = entry.split(":")
        configs.append((float(cv_str), float(cs_str)))
    return tuple(configs)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-states", type=int, default=20)
    parser.add_argument("--n-full", type=int, default=64)
    parser.add_argument(
        "--configs",
        default="50:0.1,50:0.03,1:0.1",
        help="comma-separated c_visit:c_scale pairs to sweep, e.g. "
        "'50:0.1,50:0.03,1:0.1' (default: shipped, the cs=0.03 diagnostic "
        "arm that removed Gate A's net harm, and cv=1 -- an in-mctx knob "
        "that grows sigma with actual visits instead of being 50-dominated)",
    )
    parser.add_argument("--base-seed", type=int, default=500001)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    catanatron_rs = _require_rust_module()
    configs = _parse_configs(args.configs)

    evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
        args.checkpoint, device=args.device, config=EntityGraphRustEvaluatorConfig()
    )
    try:
        states = find_placement_roots(catanatron_rs, n_states=int(args.n_states), base_seed=int(args.base_seed))
        results = run_sweep(
            states, evaluator, configs=configs, n_full=int(args.n_full), seed_base=int(args.base_seed) * 7
        )
    finally:
        evaluator.close()

    write_json(args.out, results)
    summary = {
        key: {
            "c_visit": value["c_visit"],
            "c_scale": value["c_scale"],
            "n_states": value["n_states"],
            "flips": value["flips"],
            "flip_rate": value["flip_rate"],
            "avg_n_candidates": value["avg_n_candidates"],
            "avg_n_visited": value["avg_n_visited"],
        }
        for key, value in results.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
