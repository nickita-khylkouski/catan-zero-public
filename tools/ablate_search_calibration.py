#!/usr/bin/env python3
"""CLI: joint c_visit x c_scale search-calibration ablation vs the production
baseline (task: "top-ranked science experiment from three independent
reviews").

MOTIVATION. Production search runs c_visit=50, c_scale=0.03 -- 33x below the
Gumbel paper's validated (c_visit=50, c_scale=1.0) pair -- and c_scale was
only ever ablated ALONE (see docs/f70_search_reliability_arms.md /
tools/h2h_v3conf_aggregate.py's v3a-vs-v3b base decision). It may be
compensating for the mctx min-max completed-Q rescale
(`GumbelChanceMCTS._rescale_completed_q`) stretching low-visit sampling noise
into a full [0, 1] spread at wide (54-action) placement roots -- the verified
Gate-A mechanism. This tool holds the CHECKPOINT fixed and plays each of a
grid of (c_visit, c_scale) cells, plus a few structural qtransform variants,
head-to-head against that exact production baseline, using the SAME paired
seed / color-swap / pentanomial-SPRT protocol as
tools/gumbel_search_cross_net_h2h.py -- except both sides here use the SAME
checkpoint (this experiment isolates the SEARCH CONFIG's contribution, not
the checkpoint's), so only one evaluator per worker is needed.

ARMS.
  Grid: c_visit in {5, 25, 50} x c_scale in {0.03, 0.1, 0.3, 1.0}, naming
  each cell "cv<c_visit>_cs<c_scale>" (e.g. "cv5_cs0.03"), MINUS the
  baseline cell itself (cv50_cs0.03 -- that IS the baseline, not a separate
  arm to test against itself).

  Special arms:
    fixed-bounds     Baseline c_visit/c_scale, but replaces the min-max
                      completed-Q rescale with a FIXED affine map from the
                      network's known bounded value range [-1, 1] to [0, 1]
                      (see FixedBoundsGumbelChanceMCTS below) instead of
                      re-stretching whatever raw-Q spread a node happens to
                      observe -- the direct fix for the manufactured-noise
                      mechanism, as opposed to the noise-FLOOR (D1) or
                      variance-SHRINKAGE (D2) mitigations.
    D1               rescale_noise_floor_c > 0 (flag-gated, task #67).
                      NEVER validated in a real H2H before now (f70 doc
                      leaves the default at 0.0/disabled and calls
                      calibration future work) -- --d1-c/--d1-sigma-eval
                      default to this tool's own starting values, not a
                      previously-blessed constant.
    D2               variance_aware_q=True (flag-gated, task #68), also
                      never before validated in an H2H.
    D2+fixed-bounds  D2's shrinkage AND the fixed-bounds rescale together.

IMPLEMENTING fixed-bounds WITHOUT EDITING gumbel_chance_mcts.py. The rescale
is invoked as `self._rescale_completed_q(completed_q)` (an instance-method
call resolved via MRO, not a hardcoded classmethod reference), so a plain
Python subclass overriding just that one staticmethod is a fully general,
process-local extension: only the MCTS instance built with
`FixedBoundsGumbelChanceMCTS` uses the fixed map, and it never touches the
plain `GumbelChanceMCTS` instance playing the other H2H role in the same
process (no global monkeypatching, no shared class-level state). If this
qtransform variant should eventually become a first-class, non-experimental
config knob, the equivalent source-level change is small: add a
`qtransform: Literal["minmax", "fixed_bounds"] = "minmax"` field to
`GumbelChanceMCTSConfig` and branch on it inside `_rescale_completed_q`.

ROOT-WIDTH STRATIFICATION. The task asks for win-rate/decision stats split
by wide (>=40 legal actions, --wide-threshold) placement roots vs mid-game.
A per-GAME win rate has no clean width split -- EVERY game contains both wide
placement roots and mid-game decisions, so there is no natural "games with
wide roots" vs "games without" partition. Instead this tool computes, from
every played game's own decision loop (a "per-game placement info in the
H2H game records" reading of the task, and free -- SearchResult.priors is
already computed for every decision, no extra evaluator calls), a per-ROLE,
per-WIDTH-BUCKET "deviate-from-prior rate": does the arm's/baseline's search
pick a DIFFERENT action than argmax(raw prior) would have, at wide roots
vs mid-game roots? This directly measures whether a search-config change
alters HOW MUCH search overrides the raw policy specifically at wide roots
-- exactly the mechanism this whole experiment is testing -- and is strictly
more informative than a legal-action-count histogram from a separate small
instrumented sample (the task's documented fallback), which is why this
tool computes it from the real games instead. Single-legal-action decisions
(forced ROLLs etc.) are excluded from the denominator (trivially "agree").

SEEDING. Each arm gets a disjoint base-seed BLOCK via
`tools/seed_fleet_planner.plan_disjoint_seed_blocks` (one global counter,
one formula, applied once per arm -- the exact fix for the class of
collision bug documented in that module and in task #77), verified with
`assert_disjoint_seed_blocks` before any arm runs. Within an arm, worker
seeds stride off that arm's block exactly as
`gumbel_search_cross_net_h2h.py` strides off `--base-seed`.

USAGE.
  python tools/ablate_search_calibration.py \\
      --checkpoint runs/bc/gen1_20260705/checkpoint.pt \\
      --arms all --pairs 100 --n-full 8 --workers 16 \\
      --devices cuda:0,cuda:1 --out-dir runs/ablate_search_calibration/<tag>

  CPU smoke (no GPU touched): --arms cv5_cs0.03 --pairs 2 --n-full 4 \\
      --workers 1 --device cpu --out-dir /tmp/ablate_smoke
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.gumbel_self_play import _apply_selected_action
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTS, GumbelChanceMCTSConfig
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module
from factory_common import write_json
from gumbel_search_cross_net_h2h import _concordant_pair_outcomes, _write_worker_progress
from seed_fleet_planner import assert_disjoint_seed_blocks, plan_disjoint_seed_blocks
from sprt_gate import GATE_CONFIGS, evaluate_pentanomial_sprt, evaluate_sprt, pair_scores_from_h2h_games, resolve_gate_config

COLORS: tuple[str, ...] = ("RED", "BLUE")

# --- Baseline + grid definition ---------------------------------------------
BASELINE_C_VISIT: float = 50.0
BASELINE_C_SCALE: float = 0.03
GRID_C_VISIT: tuple[float, ...] = (5.0, 25.0, 50.0)
GRID_C_SCALE: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
SPECIAL_ARMS: tuple[str, ...] = (
    "fixed-bounds",
    "D1",
    "D2",
    "D2+fixed-bounds",
    # CAT-61 retry arms. "D2-js" swaps D2's hand-tuned-k shrinkage for the
    # closed-form James-Stein coefficient; "backup-weight" is the capped
    # uncertainty backup weighting ALONE (no D2 shrinkage); "D2-js+backup-weight"
    # is both, so the two fixes can be attributed independently (ticket step 5).
    "D2-js",
    "backup-weight",
    "D2-js+backup-weight",
)

_GRID_ARM_RE = re.compile(r"^cv(?P<c_visit>[0-9.]+)_cs(?P<c_scale>[0-9.]+)$")


def _grid_arm_name(c_visit: float, c_scale: float) -> str:
    return f"cv{c_visit:g}_cs{c_scale:g}"


def grid_arm_names() -> list[str]:
    names: list[str] = []
    for c_visit in GRID_C_VISIT:
        for c_scale in GRID_C_SCALE:
            if c_visit == BASELINE_C_VISIT and c_scale == BASELINE_C_SCALE:
                continue  # the baseline cell itself, not a separate arm
            names.append(_grid_arm_name(c_visit, c_scale))
    return names


def all_arm_names() -> list[str]:
    return grid_arm_names() + list(SPECIAL_ARMS)


def _parse_requested_arms(spec: str) -> list[str]:
    spec = spec.strip()
    if spec in ("all", ""):
        return all_arm_names()
    if spec == "grid":
        return grid_arm_names()
    if spec == "special":
        return list(SPECIAL_ARMS)
    return [a.strip() for a in spec.split(",") if a.strip()]


def _arm_slug(arm_name: str) -> str:
    """Filesystem-safe stem for an arm's output file (keeps the literal arm
    name, with "+", in the JSON `arm` field itself)."""
    return arm_name.replace("+", "-")


# --- fixed-bounds qtransform variant -----------------------------------------
class FixedBoundsGumbelChanceMCTS(GumbelChanceMCTS):
    """Candidate-only qtransform variant (arms "fixed-bounds" and
    "D2+fixed-bounds"): map completed-Q from the network's known bounded
    value range [-1, 1] straight to [0, 1], instead of the mctx min-max
    rescale re-stretching whatever raw-Q spread the node happens to observe
    (see module docstring). Per-visit backup values are clamped into
    [-1, 1] by `GumbelChanceMCTS._finish_expand`'s prior_value clamp and by
    terminal +-1.0, and a node's Q is a probability-weighted average of
    such values, so it never leaves that range; clamping here is only a
    defensive guard against float round-off, not an expected excursion."""

    @staticmethod
    def _rescale_completed_q(
        completed_q: dict[int, float], *, epsilon: float = 1.0e-8
    ) -> dict[int, float]:
        del epsilon  # unused: the fixed map has no degenerate-spread division
        if not completed_q:
            return {}
        return {
            action_id: min(1.0, max(0.0, (value + 1.0) / 2.0))
            for action_id, value in completed_q.items()
        }


_MCTS_CLASS_BY_KEY: dict[str, type[GumbelChanceMCTS]] = {
    "stock": GumbelChanceMCTS,
    "fixed_bounds": FixedBoundsGumbelChanceMCTS,
}


class ArmSpec:
    __slots__ = ("name", "config_overrides", "mcts_cls_key", "description")

    def __init__(
        self,
        name: str,
        config_overrides: dict[str, Any],
        *,
        mcts_cls_key: str = "stock",
        description: str = "",
    ) -> None:
        self.name = name
        self.config_overrides = config_overrides
        self.mcts_cls_key = mcts_cls_key
        self.description = description


def resolve_arm(arm_name: str, args: argparse.Namespace) -> ArmSpec:
    match = _GRID_ARM_RE.match(arm_name)
    if match:
        c_visit = float(match.group("c_visit"))
        c_scale = float(match.group("c_scale"))
        return ArmSpec(
            arm_name,
            {"c_visit": c_visit, "c_scale": c_scale},
            description=f"grid cell: c_visit={c_visit:g}, c_scale={c_scale:g}",
        )
    if arm_name == "fixed-bounds":
        return ArmSpec(
            arm_name,
            {"c_visit": BASELINE_C_VISIT, "c_scale": BASELINE_C_SCALE},
            mcts_cls_key="fixed_bounds",
            description="baseline c_visit/c_scale; fixed [-1,1]->[0,1] Q-normalization "
            "instead of the min-max rescale",
        )
    if arm_name == "D1":
        return ArmSpec(
            arm_name,
            {
                "c_visit": BASELINE_C_VISIT,
                "c_scale": BASELINE_C_SCALE,
                "rescale_noise_floor_c": float(args.d1_c),
                "sigma_eval": float(args.d1_sigma_eval),
            },
            description=f"D1 noise-floor rescale attenuation, c={args.d1_c:g}, "
            f"sigma_eval={args.d1_sigma_eval:g}",
        )
    if arm_name == "D2":
        return ArmSpec(
            arm_name,
            {
                "c_visit": BASELINE_C_VISIT,
                "c_scale": BASELINE_C_SCALE,
                "variance_aware_q": True,
                "variance_aware_k": float(args.d2_k),
            },
            description=f"D2 variance-aware completed-Q, k={args.d2_k:g}",
        )
    if arm_name == "D2+fixed-bounds":
        return ArmSpec(
            arm_name,
            {
                "c_visit": BASELINE_C_VISIT,
                "c_scale": BASELINE_C_SCALE,
                "variance_aware_q": True,
                "variance_aware_k": float(args.d2_k),
            },
            mcts_cls_key="fixed_bounds",
            description=f"D2 (k={args.d2_k:g}) + fixed [-1,1] Q-normalization",
        )
    if arm_name == "D2-js":
        return ArmSpec(
            arm_name,
            {
                "c_visit": BASELINE_C_VISIT,
                "c_scale": BASELINE_C_SCALE,
                "variance_aware_q": True,
                "variance_aware_closed_form_js": True,
            },
            description="D2 with the closed-form James-Stein shrinkage "
            "coefficient lambda*=v2/(v2+s2) (no hand-tuned k) [CAT-61]",
        )
    if arm_name == "backup-weight":
        return ArmSpec(
            arm_name,
            {
                "c_visit": BASELINE_C_VISIT,
                "c_scale": BASELINE_C_SCALE,
                "uncertainty_backup_weighting": True,
                "uncertainty_backup_a": float(args.backup_weight_a),
                "uncertainty_backup_exp": float(args.backup_weight_exp),
                "uncertainty_backup_cap": float(args.backup_weight_cap),
            },
            description="CAT-61 KataGo capped uncertainty backup weighting "
            f"(a={args.backup_weight_a:g}, exp={args.backup_weight_exp:g}, "
            f"cap={args.backup_weight_cap:g}); requires an uncertainty head",
        )
    if arm_name == "D2-js+backup-weight":
        return ArmSpec(
            arm_name,
            {
                "c_visit": BASELINE_C_VISIT,
                "c_scale": BASELINE_C_SCALE,
                "variance_aware_q": True,
                "variance_aware_closed_form_js": True,
                "uncertainty_backup_weighting": True,
                "uncertainty_backup_a": float(args.backup_weight_a),
                "uncertainty_backup_exp": float(args.backup_weight_exp),
                "uncertainty_backup_cap": float(args.backup_weight_cap),
            },
            description="CAT-61 closed-form James-Stein D2 AND capped backup "
            f"weighting (a={args.backup_weight_a:g}, exp={args.backup_weight_exp:g}, "
            f"cap={args.backup_weight_cap:g})",
        )
    raise SystemExit(f"unknown arm {arm_name!r}; choose from {all_arm_names()}")


# --- per-game play, instrumented with root-width stratification -------------
def _new_width_stats() -> dict[str, int]:
    return {"decisions": 0, "wide_decisions": 0, "wide_deviate": 0, "mid_deviate": 0}


def play_one_h2h_game(
    mcts_by_role: dict[str, GumbelChanceMCTS],
    *,
    role_by_color: dict[str, str],
    game_seed: int,
    max_decisions: int,
    wide_threshold: int,
) -> dict[str, Any]:
    """Adapted from gumbel_search_cross_net_h2h.play_one_h2h_game: same
    paired-seed protocol (both sides use IDENTICAL correct_rust_chance_spectra
    handling and the shared `_apply_selected_action` chance resolution), with
    root-width-stratified deviate-from-prior bookkeeping added per decision
    (see module docstring)."""
    import random

    catanatron_rs = _require_rust_module()
    game = catanatron_rs.Game.simple(list(COLORS), seed=int(game_seed))
    chance_rng = random.Random(int(game_seed) ^ 0xA17E)

    width_stats = {"candidate": _new_width_stats(), "baseline": _new_width_stats()}

    decision_index = 0
    terminal = False
    while decision_index < int(max_decisions):
        if game.winning_color() is not None:
            terminal = True
            break
        legal_rust = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
        if not legal_rust:
            break

        acting_color = str(game.current_color())
        role = role_by_color[acting_color]
        result = mcts_by_role[role].search(game, force_full=True)
        selected = int(result.selected_action)

        n_legal = len(legal_rust)
        if n_legal > 1 and result.priors:
            stats = width_stats[role]
            stats["decisions"] += 1
            prior_argmax = max(result.priors, key=lambda a: (result.priors[a], -int(a)))
            deviated = int(prior_argmax) != selected
            if n_legal >= int(wide_threshold):
                stats["wide_decisions"] += 1
                if deviated:
                    stats["wide_deviate"] += 1
            elif deviated:
                stats["mid_deviate"] += 1

        game = _apply_selected_action(
            game,
            selected,
            colors=COLORS,
            rng=chance_rng,
            correct_rust_chance_spectra=True,
        )
        decision_index += 1

    if not terminal:
        terminal = game.winning_color() is not None
    truncated = not terminal
    winner = str(game.winning_color()) if terminal else None
    final_vps: dict[str, int] = {}
    for color in COLORS:
        state = json.loads(game.player_state_json(color))
        final_vps[color] = int(state.get("victory_points", 0) or 0)

    candidate_color = next(color for color, role in role_by_color.items() if role == "candidate")
    baseline_color = next(color for color, role in role_by_color.items() if role == "baseline")
    candidate_won = (winner == candidate_color) if terminal else None

    return {
        "game_seed": int(game_seed),
        "candidate_color": candidate_color,
        "baseline_color": baseline_color,
        "winner": winner,
        "terminated": bool(terminal),
        "truncated": bool(truncated),
        "decisions": int(decision_index),
        "final_vps": final_vps,
        "candidate_won": candidate_won,
        "search_won": candidate_won,  # reused by sprt_gate's pair reducers
        "width_stats": width_stats,
    }


# --- worker ------------------------------------------------------------------
def _build_evaluator(checkpoint: str, worker_args: dict[str, Any]) -> Any:
    return BatchedEntityGraphRustEvaluator.from_checkpoint(
        checkpoint,
        device=worker_args["device"],
        config=EntityGraphRustEvaluatorConfig(
            value_scale=float(worker_args["value_scale"]),
            prior_temperature=float(worker_args["prior_temperature"]),
            value_squash=str(worker_args.get("value_squash", "tanh")),
            public_observation=bool(worker_args.get("masked", True)),
            emit_uncertainty=bool(worker_args.get("emit_uncertainty", False)),
        ),
    )


def _worker_entry(worker_args: dict[str, Any]) -> dict[str, Any]:
    worker_index = int(worker_args.get("worker_index", -1))
    try:
        return _run_worker(worker_args)
    except Exception as error:  # noqa: BLE001 - isolate one worker from the whole batch.
        return {
            "worker_index": worker_index,
            "games": [],
            "width_totals": {},
            "error": f"worker-level failure before any game ran: {error!r}",
            "pair_errors": [],
        }


def _run_worker(worker_args: dict[str, Any]) -> dict[str, Any]:
    threads_per_worker = int(worker_args.get("threads_per_worker", 0))
    if threads_per_worker > 0:
        import torch

        torch.set_num_threads(threads_per_worker)
        torch.set_num_interop_threads(1)

    evaluator = _build_evaluator(worker_args["checkpoint"], worker_args)
    worker_seed = int(worker_args["worker_seed"])

    arm_mcts_cls = _MCTS_CLASS_BY_KEY[worker_args["arm_mcts_cls_key"]]
    arm_config = GumbelChanceMCTSConfig(seed=worker_seed, **worker_args["arm_config_kwargs"])
    baseline_config = GumbelChanceMCTSConfig(seed=worker_seed, **worker_args["baseline_config_kwargs"])
    mcts_by_role = {
        "candidate": arm_mcts_cls(arm_config, evaluator),
        "baseline": GumbelChanceMCTS(baseline_config, evaluator),
    }

    games: list[dict[str, Any]] = []
    pair_errors: list[dict[str, Any]] = []
    width_totals = {"candidate": _new_width_stats(), "baseline": _new_width_stats()}
    try:
        for pair in worker_args["pairs"]:
            game_seed = int(pair["game_seed"])
            pair_games: list[dict[str, Any]] = []
            try:
                for orientation, role_by_color in (
                    ("candidate_red", {"RED": "candidate", "BLUE": "baseline"}),
                    ("candidate_blue", {"RED": "baseline", "BLUE": "candidate"}),
                ):
                    record = play_one_h2h_game(
                        mcts_by_role,
                        role_by_color=role_by_color,
                        game_seed=game_seed,
                        max_decisions=int(worker_args["max_decisions"]),
                        wide_threshold=int(worker_args["wide_threshold"]),
                    )
                    record["orientation"] = orientation
                    record["pair_id"] = int(pair["pair_id"])
                    for role in ("candidate", "baseline"):
                        for key, value in record["width_stats"][role].items():
                            width_totals[role][key] += value
                    pair_games.append(record)
            except Exception as error:  # noqa: BLE001 - keep the worker's other pairs.
                pair_errors.append({
                    "pair_id": int(pair["pair_id"]),
                    "game_seed": game_seed,
                    "error": repr(error),
                })
                continue
            games.extend(pair_games)
            _wins = sum(1 for g in games if g.get("candidate_won"))
            _write_worker_progress(worker_args.get("progress_dir", ""),
                                    int(worker_args["worker_index"]), len(games), _wins)
    finally:
        evaluator.close()

    return {
        "worker_index": int(worker_args["worker_index"]),
        "games": games,
        "width_totals": width_totals,
        "error": None,
        "pair_errors": pair_errors,
    }


# --- per-arm run ---------------------------------------------------------
def _base_search_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return dict(
        colors=COLORS,
        n_full=int(args.n_full),
        n_fast=int(args.n_full),  # unused: force_full=True always selects n_full.
        p_full=1.0,
        max_depth=int(args.max_depth),
        temperature=0.0,
        correct_rust_chance_spectra=True,
        lazy_interior_chance=bool(args.lazy),
        c_visit=BASELINE_C_VISIT,
        c_scale=BASELINE_C_SCALE,
        max_root_candidates=16,
        max_root_candidates_wide=54,
    )


def run_one_arm(arm: ArmSpec, args: argparse.Namespace, *, seed_block: int) -> dict[str, Any]:
    pairs = [{"pair_id": i, "game_seed": int(seed_block) + i} for i in range(max(1, int(args.pairs)))]
    workers = max(1, int(args.workers))
    threads_per_worker = int(args.threads_per_worker)
    if threads_per_worker <= 0:
        import os as _os

        threads_per_worker = max(1, (_os.cpu_count() or workers) // workers)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        import os as _os

        _os.environ[name] = str(threads_per_worker)

    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for i, pair in enumerate(pairs):
        shards[i % workers].append(pair)

    devices = [d.strip() for d in args.devices.split(",")] if args.devices else [args.device]
    progress_dir = Path(args.out_dir) / "progress" / _arm_slug(arm.name)
    progress_dir.mkdir(parents=True, exist_ok=True)

    base_kwargs = _base_search_config_kwargs(args)
    baseline_kwargs = dict(base_kwargs)
    arm_kwargs = {**base_kwargs, **arm.config_overrides}

    worker_args_list: list[dict[str, Any]] = []
    for worker_index, pair_shard in enumerate(shards):
        if not pair_shard:
            continue
        worker_args_list.append({
            "worker_index": worker_index,
            "pairs": pair_shard,
            "checkpoint": args.checkpoint,
            "device": devices[worker_index % len(devices)],
            "progress_dir": str(progress_dir),
            "value_scale": float(args.value_scale),
            "prior_temperature": float(args.prior_temperature),
            "value_squash": str(args.value_squash),
            "masked": bool(args.masked),
            "max_decisions": int(args.max_decisions),
            "wide_threshold": int(args.wide_threshold),
            "arm_config_kwargs": arm_kwargs,
            "baseline_config_kwargs": baseline_kwargs,
            # CAT-61: the evaluator must surface the value-error head for the
            # capped backup-weighting arm to have any signal. Enabled whenever
            # the arm turns on backup weighting; the baseline shares this
            # evaluator but ignores the extra scalar (its config leaves
            # uncertainty_backup_weighting off), so the baseline stays identical.
            "emit_uncertainty": bool(arm_kwargs.get("uncertainty_backup_weighting", False)),
            "arm_mcts_cls_key": arm.mcts_cls_key,
            "threads_per_worker": threads_per_worker,
            "worker_seed": int(seed_block) + 0x9E3779B9 * (worker_index + 1),
        })

    started = time.perf_counter()
    if len(worker_args_list) <= 1:
        results = [_worker_entry(worker_args_list[0])] if worker_args_list else []
    else:
        ctx = multiprocessing.get_context("spawn")
        results = []
        with ctx.Pool(processes=len(worker_args_list)) as pool:
            for done, result in enumerate(pool.imap_unordered(_worker_entry, worker_args_list), start=1):
                results.append(result)
                _g = sum(len(r.get("games", ())) for r in results)
                _w = sum(1 for r in results for gm in r.get("games", ()) if gm.get("candidate_won"))
                print(json.dumps({
                    "arm": arm.name, "progress": "worker_done", "workers_done": done,
                    "workers_total": len(worker_args_list), "games_so_far": _g,
                    "candidate_wins_so_far": _w,
                    "running_winrate": round(_w / _g, 4) if _g else None,
                }), flush=True)
    elapsed = time.perf_counter() - started

    all_games: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    width_totals = {"candidate": _new_width_stats(), "baseline": _new_width_stats()}
    for result in results:
        all_games.extend(result.get("games", ()))
        for role in width_totals:
            role_totals = result.get("width_totals", {}).get(role, {})
            for key in width_totals[role]:
                width_totals[role][key] += int(role_totals.get(key, 0))
        if result.get("error"):
            errors.append({"worker_index": result.get("worker_index"), "error": result["error"]})
        for pair_error in result.get("pair_errors") or ():
            errors.append({"worker_index": result.get("worker_index"), **pair_error})

    outcomes = [bool(g["candidate_won"]) for g in all_games if g["candidate_won"] is not None]
    truncated_count = sum(1 for g in all_games if g["truncated"])
    sprt = evaluate_sprt(outcomes=outcomes, elo0=float(args.elo0), elo1=float(args.elo1))
    pair_outcomes, pair_diag = _concordant_pair_outcomes(all_games)
    pair_sprt = evaluate_sprt(outcomes=pair_outcomes, elo0=float(args.elo0), elo1=float(args.elo1))
    pair_scores, _pent_diag = pair_scores_from_h2h_games(all_games)
    pentanomial_sprt = evaluate_pentanomial_sprt(
        pair_scores, elo0=float(args.elo0), elo1=float(args.elo1)
    )

    complete_pairs = pair_diag["ww_pairs"] + pair_diag["ll_pairs"] + pair_diag["split_pairs"]
    decisive_pairs = pair_diag["ww_pairs"] + pair_diag["ll_pairs"]

    def _rate(numerator: float, denominator: float) -> float | None:
        return (numerator / denominator) if denominator else None

    width_summary = {
        role: {
            **stats,
            "mid_decisions": stats["decisions"] - stats["wide_decisions"],
            "wide_deviate_rate": _rate(stats["wide_deviate"], stats["wide_decisions"]),
            "mid_deviate_rate": _rate(
                stats["mid_deviate"], stats["decisions"] - stats["wide_decisions"]
            ),
        }
        for role, stats in width_totals.items()
    }

    summary = {
        "arm": arm.name,
        "arm_description": arm.description,
        "arm_config_overrides": arm.config_overrides,
        "arm_mcts_cls_key": arm.mcts_cls_key,
        "checkpoint": args.checkpoint,
        "n_full": int(args.n_full),
        "masked": bool(args.masked),
        "lazy_interior_chance": bool(args.lazy),
        "wide_threshold": int(args.wide_threshold),
        "seed_block_base": int(seed_block),
        "seed_block_size": int(args.seed_block_size),
        "pairs_requested": len(pairs),
        "games_played": len(all_games),
        "games_with_winner": len(outcomes),
        "games_truncated": truncated_count,
        "candidate_wins": sum(1 for outcome in outcomes if outcome),
        "baseline_wins": sum(1 for outcome in outcomes if not outcome),
        "candidate_win_rate": _rate(sum(1 for outcome in outcomes if outcome), len(outcomes)),
        "sprt": sprt,
        "pair_sprt": pair_sprt,
        "pentanomial_sprt": pentanomial_sprt,
        "pair_diagnostics": pair_diag,
        "complete_pairs": complete_pairs,
        "decisive_pair_yield": _rate(decisive_pairs, complete_pairs),
        "split_rate": _rate(pair_diag["split_pairs"], complete_pairs),
        "root_width_stratified": width_summary,
        "elapsed_sec": elapsed,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "errors": errors,
    }
    return {"summary": summary, "games": all_games}


# --- ranked reporting ---------------------------------------------------
def _fmt_pct(value: float | None) -> str:
    return f"{100.0 * value:.1f}" if value is not None else "n/a"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def _write_markdown_table(path: Path, ranked: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Joint c_visit x c_scale search-calibration ablation",
        "",
        f"checkpoint={args.checkpoint}  n_full={args.n_full}  pairs/arm={args.pairs}  "
        f"masked={args.masked}  lazy={args.lazy}  elo0={args.elo0}  elo1={args.elo1}  "
        f"baseline=(c_visit={BASELINE_C_VISIT:g}, c_scale={BASELINE_C_SCALE:g})",
        "",
        "Ranked by pentanomial mean pair score (candidate win rate estimate) vs the "
        "production baseline. `wide%`/`mid%` are the candidate's (arm's) deviate-from-"
        f"prior rate at >={args.wide_threshold}-legal-action roots vs mid-game roots.",
        "",
        "| arm | win% | pairs (WW/split/LL) | pentanomial LLR | decision | wide-deviate% | mid-deviate% |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in ranked:
        pent = summary.get("pentanomial_sprt") or {}
        pdiag = summary.get("pair_diagnostics") or {}
        candidate_width = (summary.get("root_width_stratified") or {}).get("candidate") or {}
        lines.append(
            f"| {summary['arm']} | {_fmt_pct(summary.get('candidate_win_rate'))} | "
            f"{summary.get('complete_pairs')} "
            f"({pdiag.get('ww_pairs')}/{pdiag.get('split_pairs')}/{pdiag.get('ll_pairs')}) | "
            f"{_fmt_num(pent.get('llr'))} | {pent.get('decision', 'n/a')} | "
            f"{_fmt_pct(candidate_width.get('wide_deviate_rate'))} | "
            f"{_fmt_pct(candidate_width.get('mid_deviate_rate'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _arm_output_path(out_dir: Path, arm_name: str) -> Path:
    return out_dir / f"{_arm_slug(arm_name)}.json"


def _load_completed_arm(out_dir: Path, arm_name: str, *, pairs: int) -> dict[str, Any] | None:
    path = _arm_output_path(out_dir, arm_name)
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(existing.get("pairs_requested", -1)) != int(pairs):
        return None
    if int(existing.get("games_played", -1)) != 2 * int(pairs):
        return None
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint c_visit x c_scale search-calibration ablation vs the production "
        "baseline (c_visit=50, c_scale=0.03), same checkpoint both sides.",
    )
    parser.add_argument("--checkpoint", default="runs/bc/gen1_20260705/checkpoint.pt")
    parser.add_argument(
        "--arms", default="all",
        help="comma-separated arm names, or 'all' (default) / 'grid' / 'special'. "
        f"Grid arms: {grid_arm_names()}. Special arms: {list(SPECIAL_ARMS)}.",
    )
    parser.add_argument("--pairs", type=int, default=100, help="paired seeds/arm; games/arm = 2x this")
    parser.add_argument("--n-full", type=int, default=8,
                        help="full-search simulation budget (the sensitive n8 regime).")
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--wide-threshold", type=int, default=40,
                        help="legal-action-count threshold for the root-width-stratified "
                        "deviate-from-prior diagnostic. Independent of "
                        "GumbelChanceMCTSConfig.wide_candidates_threshold (=24), which "
                        "governs simulation-budget allocation, not this diagnostic.")
    parser.add_argument("--masked", action=argparse.BooleanOptionalAction, default=True,
                        help="EntityGraphRustEvaluatorConfig.public_observation, for BOTH "
                        "sides (production default: masked).")
    parser.add_argument("--lazy", action=argparse.BooleanOptionalAction, default=True,
                        help="GumbelChanceMCTSConfig.lazy_interior_chance, for BOTH sides "
                        "(production default per task #52).")
    parser.add_argument("--prior-temperature", type=float, default=1.0,
                        help="Evaluator prior temperature (EntityGraphRustEvaluatorConfig).")
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--value-squash", choices=("tanh", "clip"), default="tanh")
    parser.add_argument("--d1-c", type=float, default=1.0,
                        help="D1 arm's rescale_noise_floor_c. NOT a previously-validated "
                        "constant (f70 doc leaves the default disabled at 0.0 and flags "
                        "calibration as future work) -- override to sweep it.")
    parser.add_argument("--d1-sigma-eval", type=float, default=0.79,
                        help="D1 arm's sigma_eval (f70's placeholder default; not yet "
                        "per-checkpoint calibrated from phase_sliced_value_calibration.py).")
    parser.add_argument("--d2-k", type=float, default=1.0, help="D2 arm's variance_aware_k.")
    parser.add_argument("--backup-weight-a", type=float, default=0.25,
                        help="CAT-61 capped backup weighting coefficient a in "
                        "min(cap, a*err**exp) (KataGo default 0.25).")
    parser.add_argument("--backup-weight-exp", type=float, default=1.0,
                        help="CAT-61 capped backup weighting exponent (KataGo default 1.0).")
    parser.add_argument("--backup-weight-cap", type=float, default=1.0,
                        help="CAT-61 capped backup weighting cap (the R8 lesson: KataGo's "
                        "uncertainty-weighted playouts required a cap to work).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--devices", default=None,
                        help="comma-list of devices to spread workers across per arm, e.g. "
                        "cuda:0,cuda:1 (round-robin per worker; overrides --device).")
    parser.add_argument("--threads-per-worker", type=int, default=0,
                        help="0 = auto: floor(os.cpu_count() / workers).")
    parser.add_argument("--base-seed", type=int, default=81_000_001,
                        help="global counter for plan_disjoint_seed_blocks; each arm gets "
                        "one disjoint block of --seed-block-size seeds.")
    parser.add_argument("--seed-block-size", type=int, default=10_000,
                        help="must be >= --pairs; disjoint per-arm seed block size.")
    parser.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default="flywheel",
        help="Named SPRT gate config (CAT-7) providing --elo0/--elo1 defaults; explicit flags override.",
    )
    parser.add_argument("--elo0", type=float, default=None, help="Override --gate-config's elo0.")
    parser.add_argument("--elo1", type=float, default=None, help="Override --gate-config's elo1.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="skip any arm whose <out-dir>/<arm>.json already reports the "
                        "requested pairs/games count complete.")
    args = parser.parse_args()
    _gate_cfg, _gate_params = resolve_gate_config(args.gate_config, elo0=args.elo0, elo1=args.elo1)
    args.elo0, args.elo1 = _gate_params["elo0"], _gate_params["elo1"]

    requested_arms = _parse_requested_arms(args.arms)
    unknown = [name for name in requested_arms if name not in all_arm_names()]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown!r}; choose from {all_arm_names()}")

    seed_plan = plan_disjoint_seed_blocks(
        requested_arms,
        games_per_worker=int(args.pairs),
        base=int(args.base_seed),
        block_size=int(args.seed_block_size),
    )
    assert_disjoint_seed_blocks(
        [(name, seed, int(args.pairs)) for name, seed in seed_plan.items()]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_summaries: list[dict[str, Any]] = []
    for arm_name in requested_arms:
        existing = _load_completed_arm(out_dir, arm_name, pairs=args.pairs) if args.resume else None
        if existing is not None:
            print(f"[resume] arm {arm_name!r} already complete "
                  f"({existing['games_played']} games), skipping", flush=True)
            arm_summaries.append({k: v for k, v in existing.items() if k != "games"})
            continue

        arm_spec = resolve_arm(arm_name, args)
        print(f"[arm] {arm_name!r}: {arm_spec.description} "
              f"(seed_block={seed_plan[arm_name]})", flush=True)
        result = run_one_arm(arm_spec, args, seed_block=seed_plan[arm_name])
        out_path = _arm_output_path(out_dir, arm_name)
        write_json(out_path, {**result["summary"], "games": result["games"]})
        arm_summaries.append(result["summary"])
        print(json.dumps(result["summary"], indent=2, sort_keys=True, default=str))

    def _rank_key(summary: dict[str, Any]) -> float:
        pentanomial = summary.get("pentanomial_sprt") or {}
        mean_pair_score = pentanomial.get("mean_pair_score")
        return float(mean_pair_score) if mean_pair_score is not None else -1.0

    ranked = sorted(arm_summaries, key=_rank_key, reverse=True)
    summary_payload = {
        "checkpoint": args.checkpoint,
        "gate_config": args.gate_config,
        "n_full": int(args.n_full),
        "pairs_per_arm": int(args.pairs),
        "masked": bool(args.masked),
        "lazy_interior_chance": bool(args.lazy),
        "wide_threshold": int(args.wide_threshold),
        "elo0": float(args.elo0),
        "elo1": float(args.elo1),
        "baseline": {"c_visit": BASELINE_C_VISIT, "c_scale": BASELINE_C_SCALE},
        "base_seed": int(args.base_seed),
        "seed_block_size": int(args.seed_block_size),
        "seed_plan": seed_plan,
        "arms_run": [summary["arm"] for summary in arm_summaries],
        "ranked_arms": ranked,
    }
    write_json(out_dir / "summary.json", summary_payload)
    _write_markdown_table(out_dir / "summary.md", ranked, args)
    print(json.dumps({
        "arms_run": summary_payload["arms_run"],
        "out_dir": str(out_dir),
        "summary_json": str(out_dir / "summary.json"),
        "summary_md": str(out_dir / "summary.md"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
