#!/usr/bin/env python3
"""CLI: generate Gumbel self-play training shards.

Plays full 2p self-play games with `catan_zero.rl.gumbel_self_play.play_one_game`
(both seats searched with `GumbelChanceMCTS`), writing entity-token shards
compatible with `tools/train_bc.py`'s loader. See
`src/catan_zero/rl/gumbel_self_play.py` for the driver/schema details.

Note: `tools/build_combined_entity_manifest.py` (referenced as the tool this
script's output manifest should be compatible with) does not exist in this
checkout. The top-level manifest this script writes instead follows the same
`{"shards": [...]}` convention `tools/train_bc.py`'s own `_teacher_shard_files`
loader already reads directly (see `tools/generate_rust_mcts_reanalysis.py`'s
manifest for the established precedent) -- no separate merge tool is required
to consume this output.

Worker execution is multiprocessing-only BY DESIGN (CAT-120). A threaded /
shared-batched-evaluator generation path was built and benched TWICE
independently (branches ``threaded-gen``/``--worker-mode thread`` and
``threaded-gen-batched``/``--executor thread``, plus a local ``--use-threads``
patch) and every variant is a ~4x THROUGHPUT REGRESSION, not a speedup:
per-leaf featurization is ~96% GIL-bound Python, so N worker threads serialize
on one core while N processes use N cores -- the GPU sits ~97% idle, so bf16 /
larger batches cannot help. ``allow_threads`` + ``--rust-featurize`` is
necessary-but-not-sufficient (also needs the chance-node ``evaluate_many`` path
routed through the batch queue) and still caps below the eval-server. The
throughput lever is the eval-server (CAT-67): a separate GPU process batching
many SEPARATE worker processes = batching AND full CPU parallelism, no shared
GIL. Do NOT reintroduce a ``--use-threads`` / ``--executor thread`` /
``--worker-mode thread`` path here. Data-parity was never the problem (the
threaded shards were schema- and distribution-valid); throughput was.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from catan_zero.rl.gumbel_self_play import (
    COLORS,
    GumbelSelfPlayConfig,
    MixRuntime,
    OpponentPoolRuntime,
    read_opponent_pool_manifest,
    run_worker_games,
)
from catan_zero.rl.flywheel.opponent_mix import (
    EXTERNAL_ENGINE_FRACTION_CAP,
    OpponentMixConfig,
    external_engine_effective_fraction,
    scale_external_engine_fraction,
    validate_external_engine_fraction,
)
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig, HeuristicRustEvaluator
from catan_zero.rl.config_cli import add_config_flags, resolve_config
from catan_zero.rl.pipeline_configs import GenerateConfig
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from factory_common import write_json
from opponent_mix_registry import resolve_opponent_mix_manifest
from seed_fleet_planner import assert_disjoint_seed_blocks

import launcher_guards


def _resolve_mix_with_exploiter(
    manifest_path: str, exploiter_fraction: float | None
) -> OpponentMixConfig:
    """Resolve an opponent-mix manifest (expanding any CAT-9 registry categories),
    apply the CAT-56 --exploiter-fraction rescale when given, and always enforce
    the external-engine (exploiter-lane) cap. Called identically in the main
    process (fail-fast, before workers spawn) and per-worker, so both agree on the
    exact sampled mix."""
    config = resolve_opponent_mix_manifest(manifest_path)
    if exploiter_fraction is not None:
        if external_engine_effective_fraction(config) <= 0.0:
            raise SystemExit(
                "--exploiter-fraction was set but the opponent-mix manifest has no effective "
                "external_engine (catanatron_value/ab3/ab4/ab5) category to scale. Add one "
                '(source="external_engine"), or drop --exploiter-fraction.'
            )
        try:
            config = scale_external_engine_fraction(config, float(exploiter_fraction))
        except ValueError as error:
            raise SystemExit(f"--exploiter-fraction rescale failed: {error}") from error
    try:
        validate_external_engine_fraction(config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return config


def _claim_seed_range(out_dir: Path, *, base_seed: int, games: int) -> None:
    """Filesystem-local guard against two same-host launches colliding on
    --base-seed (the #77 seed-collision class), cheaper than requiring every
    caller to go through seed_fleet_planner.py's cross-host planning.

    Claims [base_seed, base_seed + games) into a JSON file per out-dir under
    out_dir's parent's `.seed_claims/` directory, then hard-fails if that
    range overlaps a LIVE claim (still present on disk) filed by a DIFFERENT
    out-dir. A claim for the SAME out-dir is a resume and is simply
    overwritten -- not a collision.
    """
    import getpass
    import os
    import socket
    import time as _time

    claims_dir = out_dir.parent / ".seed_claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    resolved_out_dir = str(out_dir.resolve())
    claim_path = claims_dir / f"{out_dir.name}.json"

    others: list[tuple[str, int, int]] = []
    for candidate in sorted(claims_dir.glob("*.json")):
        if candidate == claim_path:
            continue  # this out-dir's own prior claim -- a resume, not a peer.
        try:
            payload = json.loads(candidate.read_text())
            other_out_dir = str(payload["out_dir"])
            other_base_seed = int(payload["base_seed"])
            other_games = int(payload["games"])
        except (OSError, ValueError, KeyError, TypeError):
            continue  # stale/malformed claim file -- ignore rather than block launches.
        if other_out_dir == resolved_out_dir:
            continue  # same out-dir under a different claim filename -- still a resume.
        others.append((other_out_dir, other_base_seed, other_games))

    try:
        assert_disjoint_seed_blocks(
            [(resolved_out_dir, base_seed, games)] + others
        )
    except ValueError as error:
        raise SystemExit(
            f"seed-claim conflict: {error} Pass a disjoint --base-seed, use "
            "seed_fleet_planner.py to plan the fleet, or (if this really is an "
            "intentional replay) pass --no-seed-claim to bypass this guard."
        ) from error

    write_json(
        claim_path,
        {
            "out_dir": resolved_out_dir,
            "base_seed": int(base_seed),
            "games": int(games),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        },
    )


def _auto_shard_size(n_full: int) -> int:
    # CAT-126 #4: smaller shards for slow (high-n) generation so the first
    # shards flush in ~minutes instead of ~an hour, unblocking corpus builds.
    if int(n_full) >= 256:
        return 256
    if int(n_full) >= 128:
        return 512
    return 2048


def _shard_size_was_explicit(raw_argv: Sequence[str]) -> bool:
    return any(a == "--shard-size" or a.startswith("--shard-size=") for a in raw_argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate entity-token self-play shards via Gumbel + true-chance-node MCTS."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="entity_graph policy checkpoint; omit to use HeuristicRustEvaluator.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-full", type=int, default=64)
    parser.add_argument("--n-fast", type=int, default=16)
    parser.add_argument("--p-full", type=float, default=0.25)
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument("--c-scale", type=float, default=0.1)
    parser.add_argument(
        "--rescale-noise-floor-c",
        type=float,
        default=0.0,
        help="CAT-12/D1 (task #67): GumbelChanceMCTSConfig.rescale_noise_floor_c. "
        "0.0 (default) is the exact no-op every generation run before this flag "
        "existed used (the min-max completed-Q rescale is untouched). >0 attenuates "
        "the rescale toward neutral 0.5 when the raw Q spread is within the "
        "estimated per-candidate sampling noise (James-Stein/Kalman-gain reliability "
        "coefficient) -- see docs/VALUE_UNCERTAINTY_HEAD_AND_SEARCH_DESIGN_20260704.md "
        "Section 3. NOT a previously-validated production constant as of CAT-12 "
        "(ablate_search_calibration.py's own --d1-c default of 1.0 carries the same "
        "caveat) -- calibrate via ablate_search_calibration.py before generation use.",
    )
    parser.add_argument(
        "--sigma-eval",
        type=float,
        default=0.79,
        help="Per-eval value-estimate noise stdev feeding --rescale-noise-floor-c's "
        "noise floor (GumbelChanceMCTSConfig.sigma_eval). Only matters when "
        "--rescale-noise-floor-c > 0. 0.79 is a rough placeholder from corr(q, z) on "
        "the BC corpus, not yet re-calibrated per checkpoint -- see "
        "phase_sliced_value_calibration.py / sigma_trace_placement_root.py before "
        "trusting a noise-floor arm in production.",
    )
    parser.add_argument(
        "--n-full-wide",
        type=int,
        default=None,
        help="Placement-budget-asymmetry arm: full-search simulations to spend at "
        "roots wider than the config's wide_candidates_threshold (e.g. 512). "
        "Default None = use --n-full everywhere (disabled). Mirrors "
        "tools/gumbel_search_vs_raw_h2h.py's identical flag -- pre-wired so gen-1 "
        "generation can replicate whichever confirmation-H2H arm wins.",
    )
    parser.add_argument(
        "--raw-policy-above-width",
        type=int,
        default=None,
        help="Phase-gated-search arm: at roots wider than this many legal actions, "
        "skip search and play argmax(prior). Default None = always search "
        "(disabled). Mirrors tools/gumbel_search_vs_raw_h2h.py's identical flag.",
    )
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument(
        "--temperature-decisions",
        type=int,
        default=45,
        help="Opening decisions per game played at --temperature-high (sampled from "
        "the improved policy for trajectory diversity); --temperature-low thereafter. "
        "ABSOLUTE decision count -- internally converted to GumbelSelfPlayConfig's "
        "temperature_move_fraction = temperature_decisions / max_decisions, so it stays "
        "invariant when --max-decisions changes. Mirrors "
        "generate_raw_selfplay_data.py's --temperature-decisions.",
    )
    parser.add_argument(
        "--temperature-move-fraction",
        type=float,
        default=None,
        help="DEPRECATED -- use --temperature-decisions. Fraction OF --max-decisions "
        "played at --temperature-high. This fraction-of-cap coupling silently mis-fired "
        "twice (it must be hand-recomputed whenever the cap changes). When set, it "
        "OVERRIDES --temperature-decisions.",
    )
    parser.add_argument("--temperature-high", type=float, default=1.0)
    parser.add_argument("--temperature-low", type=float, default=0.0)
    parser.add_argument(
        "--late-temperature-decisions",
        type=int,
        default=None,
        help="CAT-12 (roadmap R8 diversity-strangulation, queue #16): decision index "
        "(ABSOLUTE count, same convention as --temperature-decisions) at which the "
        "late-temperature window ENDS and the schedule drops to --temperature-low "
        "(argmax). Default None = disabled, exact two-stage schedule unchanged (a "
        "pure no-op -- matches every generation run before this flag existed). When "
        "set, decisions in [--temperature-decisions, this) play at --late-temperature "
        "instead of jumping straight to argmax. Internally converted to "
        "GumbelSelfPlayConfig.late_temperature_move_fraction = "
        "late_temperature_decisions / max_decisions, mirroring --temperature-decisions' "
        "cap-invariant conversion.",
    )
    parser.add_argument(
        "--late-temperature",
        type=float,
        default=0.0,
        help="Nonzero temperature to use inside the --late-temperature-decisions "
        "window (GumbelSelfPlayConfig.late_temperature). Only has an effect when "
        "--late-temperature-decisions is set; a small value (e.g. 0.2-0.3) is the "
        "intended CAT-12 use, extending trajectory diversity into the midgame "
        "instead of hard argmax right after the opening.",
    )
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument(
        "--correct-rust-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mitigate verified Rust engine chance-spectrum bugs (A19/A20) in both "
        "the search and the live-game chance resolution. Set --no-correct-rust-chance-spectra "
        "to trust the engine's native spectrum_json directly (A/B against a fixed wheel).",
    )
    parser.add_argument(
        "--lazy-interior-chance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Single-sample INTERIOR chance-node (ROLL) traversal instead of full "
        "11-outcome enumeration (#52). Root ROLL enumeration and the forced-single-"
        "action fast path stay full in both modes. ~65x fewer leaf evals per full "
        "search at the cost of noisier interior backups; default off — generation "
        "use is gated on a strength-based H2H A/B.",
    )
    parser.add_argument(
        "--exact-budget-sh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exact-budget Sequential Halving (task #61, mctx conformance): root "
        "simulations equal the configured budget EXACTLY, where the legacy "
        "schedule's >=1-sim floor overruns (n_fast=16/m=16 -> 32 sims; fast "
        "search at a 54-wide placement root -> 105 sims; n_full=64/m=54 -> 119). "
        "Threads to GumbelChanceMCTSConfig.exact_budget_sh. Default off; a "
        "search-semantics change, so generation use is gated on a pentanomial "
        "H2H non-inferiority A/B vs the legacy schedule.",
    )
    parser.add_argument(
        "--exact-budget-sh-min-n",
        type=int,
        default=0,
        help="Budget threshold for --exact-budget-sh (gate-informed 2026-07-07: "
        "exact-64 gated non-inferior, exact-16 gated decisively worse): the "
        "exact schedule applies only to searches with n >= this value; smaller "
        "budgets keep the legacy schedule. 0 = exact everywhere. Adoption "
        "pairing: --exact-budget-sh --exact-budget-sh-min-n 48 ships the "
        "full-64 half while fast-16 stays legacy.",
    )
    parser.add_argument(
        "--public-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Public-observation featurization (hidden-info leak fix, f72): mask "
        "every opponent's hand composition, unplayed dev-card identities, and actual "
        "VP from the model input (keep public counts/VP + the actor's own hand). "
        "Threads to EntityGraphRustEvaluatorConfig.public_observation. Default off; "
        "pair with a checkpoint retrained via train_bc --mask-hidden-info.",
    )
    parser.add_argument(
        "--rust-featurize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build the ENTITY-TOKEN arrays via the native Rust featurizer "
        "(catanatron_rs.build_entity_features_flat) instead of the Python "
        "per-token loops (task #81). Bit-exact parity-gated (see "
        "entity_token_features_rust.py + tests/test_rust_featurize_parity.py); "
        "fails loudly, no silent fallback, if the installed wheel lacks the "
        "Rust featurizer. Threads to EntityGraphRustEvaluatorConfig.rust_featurize. "
        "Default off = exact current behavior. Requires the gen-3 Rust wheel "
        "(see docs/GEN3_WHEEL_SYNC_RUNBOOK.md) -- passing --rust-featurize "
        "against an older wheel that predates this function raises an error at "
        "the first leaf eval rather than silently falling back.",
    )
    parser.add_argument(
        "--eval-server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CAT-67: route every worker's NN forward through ONE shared "
        "cross-game EvalServer process (single CUDA context) instead of each "
        "worker holding its own evaluator/context. Removes the GPU context-thrash "
        "that caps per-GPU throughput at ~8 workers (measured 1.87x rows/hr at 8 "
        "workers, up to 3.6x at 32; see CAT-67). Featurization + all postprocess "
        "stay per-worker; only forward_legal_np is centralized, so outputs match "
        "the local path within batched-matmul tolerance. Each worker keeps a "
        "lazy LOCAL fallback: if the server times out/crashes the worker degrades "
        "to an in-process evaluator rather than hanging. Default OFF (behavior "
        "byte-unchanged). Not compatible with --opponent-pool-manifest/"
        "--opponent-mix-manifest in this prototype.",
    )
    parser.add_argument(
        "--eval-server-max-batch",
        type=int,
        default=64,
        help="EvalServer window max batch size (CAT-67). Only used with --eval-server.",
    )
    parser.add_argument(
        "--eval-server-max-wait-ms",
        type=float,
        default=3.0,
        help="EvalServer window straggler timeout in ms (CAT-67). Only used with --eval-server.",
    )
    parser.add_argument(
        "--eval-server-timeout-ms",
        type=float,
        default=20000.0,
        help="Per-request client wait before a worker degrades to its local "
        "fallback evaluator (CAT-67). Only used with --eval-server.",
    )
    parser.add_argument(
        "--eval-cache-size",
        type=int,
        default=100_000,
        help="Size of the per-worker EntityGraphRustEvaluator result cache. The "
        "cache keys every leaf by blake2b(json_snapshot) -- but self-play states "
        "are unique (transpositions are measure-zero over full Catan state), so "
        "the cache never hits and the key work is pure overhead. Pass 0 to "
        "disable the cache ENTIRELY (skips the per-leaf snapshot+blake2b key "
        "build, not just the store; see neural_rust_mcts.evaluate). Default "
        "100000 preserves prior behavior; 0 is the self-play optimization (OPT-1).",
    )
    parser.add_argument(
        "--belief-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Planner-only public-belief chance spectra (hidden-info leak fix, f72): "
        "uniform robber steals + belief-deck dev draws inside the search (the live "
        "game keeps true-state chance resolution). Threads to "
        "GumbelChanceMCTSConfig.belief_chance_spectra. Default off; a search-semantics "
        "change, so shipping is gated on a strength-based H2H A/B.",
    )
    parser.add_argument(
        "--opponent-pool-manifest",
        default=None,
        help="Archived-opponent pool (anti-forgetting, H2): JSON manifest "
        '\'{"opponents": [{"checkpoint": <path>, "version": <int>}, ...], '
        '"pool_fraction": <float in [0,1]>}\'. Default None = OFF (pure mirror '
        "self-play vs --checkpoint, exact current behavior; byte-identical "
        "shard schema). When set, requires --checkpoint (the champion net): a "
        "deterministic per-game-seed fraction of games play the champion "
        "against one of the manifest's archived checkpoints instead of "
        "champion-vs-champion (catan_zero.rl.flywheel.opponent_pool's "
        "hash-based choose_opponent, resume-safe -- not a global RNG). Only "
        "the CHAMPION seat's decisions become training rows; rows are "
        "stamped with the extra is_pool_game/opponent_version columns "
        "(catan_zero.rl.gumbel_self_play.read_opponent_pool_manifest).",
    )
    parser.add_argument(
        "--opponent-mix-manifest",
        default=None,
        help="Arbitrary-category opponent mix (CAT-54, generalizes --opponent-pool-manifest's "
        "binary fraction): JSON manifest with a \"categories\" list, each "
        '{"name": <tag>, "weight": <float>, "source": "self"|"checkpoint_list"|'
        '"external_engine"|"registry_role"|"registry_pool", ...} -- see '
        "tools/opponent_mix_registry.py's docstring for the full schema (registry_role/"
        "registry_pool categories reference a CAT-9 champion-registry JSON via the "
        'manifest\'s top-level "registry" path). Default None = OFF (pure mirror '
        "self-play vs --checkpoint, exact current behavior; byte-identical shard schema). "
        "Requires --checkpoint. Mutually exclusive with --opponent-pool-manifest (both "
        "resolve the same per-game opponent assignment; pass at most one). Only the "
        "producer's own-side decision rows become training rows on non-mirror games; rows "
        "are stamped with the extra opponent_tag/opponent_checkpoint_md5 columns (plus the "
        "existing is_pool_game/opponent_version) "
        "(catan_zero.rl.gumbel_self_play.MixRuntime).",
    )
    parser.add_argument(
        "--exploiter-fraction",
        type=float,
        default=None,
        help="Exploiter lane (CAT-56): rescale the --opponent-mix-manifest's external_engine "
        "(catanatron_value/ab3/ab4/ab5) categories so they together take this fraction of the "
        "effective mix, preserving the manifest's other weights and the external categories' "
        "relative ratios. Lets you dial the exploiter share with one number (R9 ramp: start "
        "0.02-0.03, raise to 0.05 only once neutral-harness parity is proven). Default None = use "
        "the manifest's own external weights unchanged. Whether or not this flag is set, the "
        f"effective external share is hard-capped at {EXTERNAL_ENGINE_FRACTION_CAP} (R9 ceiling); "
        "the run aborts if it would exceed that. Requires at least one external_engine category "
        "in the manifest.",
    )
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--obs-width", type=int, default=806)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument(
        "--shard-size", type=int, default=2048,
        help="Rows per shard. CAT-126 #4: if NOT passed explicitly, main() auto-scales by --n-full (n>=256->256, n>=128->512, else 2048) so slow teacher/probe runs flush first shards sooner. Passing --shard-size overrides the auto default.",
    )
    parser.add_argument("--format", choices=("npz", "npz_zst"), default="npz")
    parser.add_argument(
        "--score-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HeuristicRustEvaluator only: score actions via chance-weighted lookahead "
        "(slower, meaningful priors) vs uniform priors (fast, for smoke tests).",
    )
    parser.add_argument(
        "--seed-claim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filesystem-local same-host seed-collision guard (cheap complement to "
        "seed_fleet_planner.py's cross-host planning, #77 class): before generating, "
        "claim [--base-seed, --base-seed+--games) in <out-dir>/../.seed_claims/<name>.json "
        "and hard-fail if it overlaps a live claim from a DIFFERENT --out-dir on this host "
        "(same --out-dir is treated as a resume and allowed). Default on; "
        "--no-seed-claim opts out.",
    )
    parser.add_argument(
        "--ledger-claim-label",
        default=None,
        help=(
            "CAT-124: unique id of THIS launch's own row in the cross-host seed "
            "ledger (the canonical launcher writes a `claim=<id>` row before running "
            "guards, claim-then-verify). Passing it lets the ledger_overlap guard "
            "recognise our own just-written claim and exclude it from the collision "
            "check (peers still collide), so a legitimate fresh claim passes WITHOUT "
            "--skip-guards. Falls back to $CATAN_LEDGER_CLAIM_ID; unset -> prior "
            "behavior (any overlap, including our own row, fails closed)."
        ),
    )
    parser.add_argument(
        "--skip-guards",
        action="store_true",
        help=(
            "Skip tools/prelaunch_guard.py's pre-launch checks (CLI-default-override "
            "trap, seed-collision/VAL-ONLY range, fd-limit; CAT-69/CAT-75). Logs a loud "
            "WARNING and proceeds anyway -- use only for a known false positive or an "
            "intentional smoke test, never as a routine habit."
        ),
    )
    add_config_flags(parser, default_purpose="generate_gumbel_selfplay")
    return parser


def _build_guard_specs(
    args: argparse.Namespace, argv: Sequence[str], parser: argparse.ArgumentParser
) -> list[dict]:
    import os  # local: only used here to read the launcher-set claim id (CAT-124)

    static_specs = launcher_guards.load_static_guard_specs("generate_gumbel_selfplay_data")
    # CAT-124: the canonical launcher claims the seed range in the ledger BEFORE this tool
    # runs its guards (claim-then-verify), so by guard time our OWN row is already present.
    # own_claim_label is the unique id the launcher wrote into that row; passing it lets
    # ledger_overlap exclude our own claim (peers still collide), so a legitimate fresh claim
    # passes without --skip-guards. Precedence: explicit --ledger-claim-label > the launcher's
    # $CATAN_LEDGER_CLAIM_ID export > None (prior fail-closed-on-any-overlap behavior).
    own_claim_label = args.ledger_claim_label or os.environ.get("CATAN_LEDGER_CLAIM_ID")
    return launcher_guards.merge_dynamic_args(
        static_specs,
        {
            "cli_flag_lint": {"argv": list(argv), "parser": parser},
            "seed_ledger": {
                "out_dir": args.out_dir,
                "base_seed": int(args.base_seed),
                "games": max(0, int(args.games)),
            },
            "ledger_overlap": {
                "base_seed": int(args.base_seed),
                "games": max(0, int(args.games)),
                "own_claim_label": own_claim_label,
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # CAT-126 #4: auto-scale shard size by n-full unless the caller pinned it.
    _raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if not _shard_size_was_explicit(_raw_argv):
        _auto = _auto_shard_size(int(args.n_full))
        if _auto != int(args.shard_size):
            print(json.dumps({"progress": "auto_shard_size", "n_full": int(args.n_full), "shard_size": _auto}), flush=True)
        args.shard_size = _auto

    launcher_guards.run_or_refuse(
        _build_guard_specs(args, argv if argv is not None else sys.argv[1:], parser),
        launcher="generate_gumbel_selfplay_data",
        skip=bool(args.skip_guards),
    )

    # Resolve the temperature schedule to the single fraction the driver consumes.
    # Prefer the absolute --temperature-decisions (cap-invariant); honor an explicit
    # --temperature-move-fraction only for backward compatibility, loudly.
    if args.temperature_move_fraction is not None:
        temperature_decisions_effective = round(
            float(args.max_decisions) * float(args.temperature_move_fraction)
        )
        print(
            "WARNING: --temperature-move-fraction is deprecated; prefer the absolute "
            f"--temperature-decisions. Using fraction {args.temperature_move_fraction} "
            f"(= {temperature_decisions_effective} of {args.max_decisions} decisions).",
            file=sys.stderr,
        )
    else:
        args.temperature_move_fraction = float(args.temperature_decisions) / float(
            max(1, args.max_decisions)
        )
        temperature_decisions_effective = int(args.temperature_decisions)
    # Record the effective absolute count in the provenance dumped to manifest.json.
    args.temperature_decisions_effective = temperature_decisions_effective

    # CAT-12: same absolute-count -> fraction-of-cap conversion as above, for the
    # optional late-temperature window. None stays None (disabled, no-op).
    if args.late_temperature_decisions is not None:
        args.late_temperature_move_fraction = float(args.late_temperature_decisions) / float(
            max(1, args.max_decisions)
        )
    else:
        args.late_temperature_move_fraction = None

    # CAT-66 typed config + config-hash. Built once in the main process from the
    # fully-resolved args (the same source the per-worker dicts flatten from), so
    # the recorded hash reflects the values actually used, not any dataclass
    # default. No-op to the run when no --config* flag is passed.
    generate_config = resolve_config(args, GenerateConfig.from_namespace, parser=parser)
    generate_config_hash = generate_config.config_hash()

    # Opponent-pool (H2): validate the manifest ONCE in the main process, before
    # spawning workers -- a malformed manifest would otherwise be caught only
    # per-worker by `_worker_entry`'s catch-all (which turns it into a silent
    # all-workers-failed summary rather than a loud, immediate error). Pure
    # stdlib JSON parsing (read_opponent_pool_manifest), so this is cheap and
    # needs no torch/device.
    opponent_pool_fraction_configured: float | None = None
    if args.opponent_pool_manifest:
        if not args.checkpoint:
            raise SystemExit(
                "--opponent-pool-manifest requires --checkpoint (a neural champion "
                "net to play one seat); omit both for heuristic-evaluator smoke runs."
            )
        policy, _champion, _archive = read_opponent_pool_manifest(args.opponent_pool_manifest)
        opponent_pool_fraction_configured = float(policy.pool_fraction)

    # Opponent-MIX (CAT-54): same fail-fast-in-main-process validation as the
    # H2 binary pool above, plus the resolve step (registry_role/registry_pool
    # categories get expanded against the CAT-9 registry here, once) -- a bad
    # manifest or an unresolvable registry reference is caught before any
    # worker spawns, not per-worker.
    opponent_mix_effective_weights: dict[str, float] | None = None
    opponent_mix_exploiter_fraction: float | None = None
    if args.opponent_mix_manifest:
        if not args.checkpoint:
            raise SystemExit(
                "--opponent-mix-manifest requires --checkpoint (a neural producer "
                "net to play one seat); omit both for heuristic-evaluator smoke runs."
            )
        if args.opponent_pool_manifest:
            raise SystemExit(
                "--opponent-mix-manifest and --opponent-pool-manifest are mutually exclusive "
                "(both resolve the same per-game opponent assignment) -- pass at most one."
            )
        mix_config = _resolve_mix_with_exploiter(args.opponent_mix_manifest, args.exploiter_fraction)
        opponent_mix_effective_weights = mix_config.effective_weights()
        opponent_mix_exploiter_fraction = external_engine_effective_fraction(mix_config)
    elif args.exploiter_fraction is not None:
        raise SystemExit(
            "--exploiter-fraction requires --opponent-mix-manifest (the exploiter lane is an "
            "external_engine category of the opponent mix)."
        )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("worker_*")) or (output / "manifest.json").exists():
        raise SystemExit(f"{output} already contains self-play output; use a fresh --out-dir")

    if args.seed_claim:
        _claim_seed_range(output, base_seed=int(args.base_seed), games=max(0, int(args.games)))

    workers = max(1, int(args.workers))
    games = max(0, int(args.games))
    games_per_worker = [games // workers + (1 if i < games % workers else 0) for i in range(workers)]

    worker_args = []
    game_index_start = 0
    for worker_index, worker_games in enumerate(games_per_worker):
        if worker_games <= 0:
            continue
        worker_args.append(
            {
                "worker_index": worker_index,
                "games": worker_games,
                "game_index_start": game_index_start,
                "out_dir": str(output / f"worker_{worker_index:03d}"),
                "checkpoint": args.checkpoint,
                "device": args.device,
                "n_full": int(args.n_full),
                "n_fast": int(args.n_fast),
                "p_full": float(args.p_full),
                "c_visit": float(args.c_visit),
                "c_scale": float(args.c_scale),
                "n_full_wide": (int(args.n_full_wide) if args.n_full_wide is not None else None),
                "raw_policy_above_width": (
                    int(args.raw_policy_above_width)
                    if args.raw_policy_above_width is not None
                    else None
                ),
                "max_decisions": int(args.max_decisions),
                "max_depth": int(args.max_depth),
                "temperature_move_fraction": float(args.temperature_move_fraction),
                "temperature_high": float(args.temperature_high),
                "temperature_low": float(args.temperature_low),
                "late_temperature_move_fraction": args.late_temperature_move_fraction,
                "late_temperature": float(args.late_temperature),
                "rescale_noise_floor_c": float(args.rescale_noise_floor_c),
                "sigma_eval": float(args.sigma_eval),
                "prior_temperature": float(args.prior_temperature),
                "value_scale": float(args.value_scale),
                "track": args.track,
                "vps_to_win": int(args.vps_to_win),
                "obs_width": int(args.obs_width),
                "base_seed": int(args.base_seed),
                "worker_seed": int(args.base_seed) + 0x9E3779B9 * (worker_index + 1),
                "shard_size": int(args.shard_size),
                "format": args.format,
                "score_actions": bool(args.score_actions),
                "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
                "lazy_interior_chance": bool(args.lazy_interior_chance),
                "exact_budget_sh": bool(args.exact_budget_sh),
                "exact_budget_sh_min_n": int(args.exact_budget_sh_min_n),
                "public_observation": bool(args.public_observation),
                "rust_featurize": bool(args.rust_featurize),
                "eval_cache_size": int(args.eval_cache_size),
                "belief_chance_spectra": bool(args.belief_chance_spectra),
                "opponent_pool_manifest": args.opponent_pool_manifest,
                "opponent_mix_manifest": args.opponent_mix_manifest,
                "exploiter_fraction": (
                    float(args.exploiter_fraction) if args.exploiter_fraction is not None else None
                ),
            }
        )
        game_index_start += worker_games

    started = time.perf_counter()
    if bool(getattr(args, "eval_server", False)) and worker_args:
        # CAT-67 cross-game eval server: one shared GPU-resident policy, workers
        # are RemoteEvalClients. Explicit Processes (not a Pool) so the raw
        # mp.Queues can be handed to each worker by inheritance. Guarded to the
        # neural-checkpoint, no-opponent-pool/mix case.
        if not args.checkpoint:
            raise SystemExit("--eval-server requires --checkpoint (no server for the heuristic evaluator)")
        if args.opponent_pool_manifest or args.opponent_mix_manifest:
            raise SystemExit(
                "--eval-server is not compatible with --opponent-pool-manifest/"
                "--opponent-mix-manifest in this prototype (CAT-67)"
            )
        results = _run_eval_server_batch(worker_args, args)
    elif len(worker_args) <= 1:
        results = [_worker_entry(worker_args[0])] if worker_args else []
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_entry, worker_args)

    summary = _merge_worker_summaries(
        results,
        out_dir=output,
        elapsed_sec=time.perf_counter() - started,
        args=args,
        opponent_pool_fraction_configured=opponent_pool_fraction_configured,
        opponent_mix_effective_weights=opponent_mix_effective_weights,
        opponent_mix_exploiter_fraction=opponent_mix_exploiter_fraction,
    )
    summary["config_hash"] = generate_config_hash
    write_json(output / "manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _worker_entry(worker_args: dict[str, Any]) -> dict[str, Any]:
    """Top-level, picklable per-worker entry point (safe for multiprocessing spawn).

    Must NEVER raise: `pool.map` propagates any worker exception straight to
    the caller, aborting the whole batch and losing every OTHER worker's
    already-written shards/results (they'd never get merged into the
    top-level manifest). A worker-level failure (e.g. checkpoint load
    failure) is instead caught and returned as an error-flagged summary, so
    surviving workers still merge normally and the top-level manifest lists
    only shards that actually exist.
    """
    try:
        return _run_worker(worker_args)
    except Exception as error:  # noqa: BLE001 - isolate one worker from the whole batch.
        return _worker_level_error_summary(worker_args, error)


def _worker_level_error_summary(worker_args: dict[str, Any], error: BaseException) -> dict[str, Any]:
    """The error-flagged summary a failed worker returns so surviving workers
    still merge (mirrors `_worker_entry`'s catch-all, shared with the
    eval-server path)."""
    worker_index = int(worker_args.get("worker_index", -1))
    return {
        "worker_index": worker_index,
        "out_dir": str(worker_args.get("out_dir", "")),
        "games_requested": int(worker_args.get("games", 0)),
        "games_completed": 0,
        "games_failed": int(worker_args.get("games", 0)),
        "games_truncated": 0,
        "wins_by_color": {},
        "rows": 0,
        "decisions_total": 0,
        "forced_decisions_total": 0,
        "simulations_used_total": 0,
        "elapsed_sec": 0.0,
        "rows_per_sec": 0.0,
        "shards": [],
        "errors": [
            {
                "worker_index": worker_index,
                "game_index": None,
                "game_seed": None,
                "error": f"worker-level failure before any game ran: {error!r}",
            }
        ],
    }


def _server_worker_entry(
    worker_args: dict[str, Any],
    request_queue: Any,
    response_queue: Any,
    client_id: int,
    action_size: int,
    trained_with_masked_hidden_info: bool,
    client_timeout_ms: float,
    result_queue: Any,
) -> None:
    """CAT-67 eval-server per-worker entry (explicit-Process, not Pool). Builds a
    RemoteEvalClient wired to the shared server queues (with a lazy local
    fallback on server failure) and runs the normal self-play games through it,
    putting the same summary dict on `result_queue` that `_worker_entry` would
    return. Never raises across the process boundary."""
    try:
        from catan_zero.search.eval_server import RemoteEvalClient

        client = RemoteEvalClient(
            request_queue,
            response_queue,
            int(client_id),
            action_size=int(action_size),
            trained_with_masked_hidden_info=bool(trained_with_masked_hidden_info),
            config=EntityGraphRustEvaluatorConfig(
                value_scale=float(worker_args["value_scale"]),
                prior_temperature=float(worker_args["prior_temperature"]),
                public_observation=bool(worker_args["public_observation"]),
                rust_featurize=bool(worker_args["rust_featurize"]),
            ),
            client_timeout_ms=float(client_timeout_ms),
            fallback_checkpoint=worker_args["checkpoint"],
            fallback_device=worker_args["device"],
        )
        result_queue.put(_run_worker(worker_args, champion_evaluator=client))
    except Exception as error:  # noqa: BLE001 - isolate one worker from the batch.
        result_queue.put(_worker_level_error_summary(worker_args, error))


def _run_eval_server_batch(
    worker_args: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Launch one shared EvalServer + N RemoteEvalClient worker processes (CAT-67),
    collect their summaries, and stop the server. Results are returned ordered by
    worker_index to match the Pool path."""
    from catan_zero.search.eval_server import EvalServer, EvalServerConfig

    ctx = multiprocessing.get_context("spawn")
    server = EvalServer(
        args.checkpoint,
        num_clients=len(worker_args),
        config=EvalServerConfig(
            max_batch_size=int(args.eval_server_max_batch),
            max_wait_ms=float(args.eval_server_max_wait_ms),
            device=str(args.device),
        ),
        public_observation=bool(args.public_observation),
        mp_context=ctx,
    )
    server.start()
    meta = server.wait_ready(timeout=300.0)
    print(
        json.dumps({"progress": "eval_server_ready", "num_clients": len(worker_args), **meta}),
        flush=True,
    )
    result_queue: Any = ctx.Queue()
    procs = []
    for client_id, wargs in enumerate(worker_args):
        proc = ctx.Process(
            target=_server_worker_entry,
            args=(
                wargs,
                server.request_queue,
                server.response_queues[client_id],
                client_id,
                int(meta["action_size"]),
                bool(meta["trained_with_masked_hidden_info"]),
                float(args.eval_server_timeout_ms),
                result_queue,
            ),
            daemon=False,
            name=f"cat67-gen-worker-{client_id}",
        )
        proc.start()
        procs.append(proc)

    results: list[dict[str, Any]] = []
    for _ in range(len(worker_args)):
        results.append(result_queue.get())
    for proc in procs:
        proc.join(timeout=120.0)
    stats = server.stop()
    print(json.dumps({"progress": "eval_server_stopped", "server_stats": stats}), flush=True)
    results.sort(key=lambda summary: int(summary.get("worker_index", 0)))
    return results


def _run_worker(
    worker_args: dict[str, Any],
    *,
    champion_evaluator: Any | None = None,
) -> dict[str, Any]:
    checkpoint = worker_args["checkpoint"]
    colors = COLORS
    if champion_evaluator is not None:
        # CAT-67 --eval-server path: the champion evaluator (a RemoteEvalClient)
        # is built + injected by the caller so it can be wired to the shared
        # EvalServer's queues. Opponent pool/mix (below) are unsupported on this
        # path and guarded off in main(), so they never build their own
        # evaluators here.
        evaluator = champion_evaluator
    elif checkpoint:
        evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
            checkpoint,
            device=worker_args["device"],
            config=EntityGraphRustEvaluatorConfig(
                value_scale=float(worker_args["value_scale"]),
                prior_temperature=float(worker_args["prior_temperature"]),
                public_observation=bool(worker_args["public_observation"]),
                rust_featurize=bool(worker_args["rust_featurize"]),
                cache_size=int(worker_args.get("eval_cache_size", 100_000)),
            ),
        )
    else:
        evaluator = HeuristicRustEvaluator(score_actions=bool(worker_args["score_actions"]))

    # Opponent pool (H2): re-parse the manifest in-process (each worker is a
    # separate spawned process; cheap pure-JSON re-parse rather than trying to
    # pickle a ChampionRef/OpponentPolicy tuple across the spawn boundary).
    # `evaluator_factory` mirrors the champion evaluator's own construction
    # (same device/value_scale/prior_temperature/public_observation) so an
    # archived opponent checkpoint gets the identical fail-closed
    # public_observation/checkpoint-training-regime guard
    # (`_assert_public_observation_matches_checkpoint_training`) that already
    # protects the champion evaluator above.
    opponent_pool_manifest = worker_args.get("opponent_pool_manifest")
    opponent_pool: OpponentPoolRuntime | None = None
    if opponent_pool_manifest:
        pool_policy, pool_champion, pool_archive = read_opponent_pool_manifest(opponent_pool_manifest)
        opponent_eval_config = EntityGraphRustEvaluatorConfig(
            value_scale=float(worker_args["value_scale"]),
            prior_temperature=float(worker_args["prior_temperature"]),
            public_observation=bool(worker_args["public_observation"]),
            rust_featurize=bool(worker_args["rust_featurize"]),
            cache_size=int(worker_args.get("eval_cache_size", 100_000)),
        )
        opponent_device = worker_args["device"]

        def _load_opponent_evaluator(
            opponent_checkpoint: str,
            *,
            _config: EntityGraphRustEvaluatorConfig = opponent_eval_config,
            _device: str = opponent_device,
        ) -> BatchedEntityGraphRustEvaluator:
            return BatchedEntityGraphRustEvaluator.from_checkpoint(
                opponent_checkpoint, device=_device, config=_config
            )

        opponent_pool = OpponentPoolRuntime(
            policy=pool_policy,
            champion=pool_champion,
            archive=pool_archive,
            evaluator_factory=_load_opponent_evaluator,
        )

    # Opponent MIX (CAT-54): same in-process re-parse-per-worker + evaluator
    # factory construction as the H2 binary pool above (identical
    # public_observation/value_scale/prior_temperature/device parity guard).
    opponent_mix_manifest = worker_args.get("opponent_mix_manifest")
    opponent_mix: MixRuntime | None = None
    if opponent_mix_manifest:
        mix_config = _resolve_mix_with_exploiter(
            opponent_mix_manifest, worker_args.get("exploiter_fraction")
        )
        mix_eval_config = EntityGraphRustEvaluatorConfig(
            value_scale=float(worker_args["value_scale"]),
            prior_temperature=float(worker_args["prior_temperature"]),
            public_observation=bool(worker_args["public_observation"]),
            cache_size=int(worker_args.get("eval_cache_size", 100_000)),
        )
        mix_device = worker_args["device"]

        def _load_mix_evaluator(
            opponent_checkpoint: str,
            *,
            _config: EntityGraphRustEvaluatorConfig = mix_eval_config,
            _device: str = mix_device,
        ) -> BatchedEntityGraphRustEvaluator:
            return BatchedEntityGraphRustEvaluator.from_checkpoint(
                opponent_checkpoint, device=_device, config=_config
            )

        opponent_mix = MixRuntime(config=mix_config, evaluator_factory=_load_mix_evaluator)

    config = GumbelSelfPlayConfig(
        colors=colors,
        track=str(worker_args["track"]),
        vps_to_win=int(worker_args["vps_to_win"]),
        obs_width=int(worker_args["obs_width"]),
        max_decisions=int(worker_args["max_decisions"]),
        temperature_move_fraction=float(worker_args["temperature_move_fraction"]),
        temperature_high=float(worker_args["temperature_high"]),
        temperature_low=float(worker_args["temperature_low"]),
        late_temperature_move_fraction=(
            float(worker_args["late_temperature_move_fraction"])
            if worker_args.get("late_temperature_move_fraction") is not None
            else None
        ),
        late_temperature=float(worker_args.get("late_temperature", 0.0)),
        correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
    )
    search_config = GumbelChanceMCTSConfig(
        colors=colors,
        max_depth=int(worker_args["max_depth"]),
        seed=int(worker_args["worker_seed"]),
        c_visit=float(worker_args["c_visit"]),
        c_scale=float(worker_args["c_scale"]),
        prior_temperature=float(worker_args["prior_temperature"]),
        n_full=int(worker_args["n_full"]),
        n_fast=int(worker_args["n_fast"]),
        p_full=float(worker_args["p_full"]),
        n_full_wide=(
            int(worker_args["n_full_wide"]) if worker_args.get("n_full_wide") is not None else None
        ),
        raw_policy_above_width=(
            int(worker_args["raw_policy_above_width"])
            if worker_args.get("raw_policy_above_width") is not None
            else None
        ),
        correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
        lazy_interior_chance=bool(worker_args["lazy_interior_chance"]),
        exact_budget_sh=bool(worker_args.get("exact_budget_sh", False)),
        exact_budget_sh_min_n=int(worker_args.get("exact_budget_sh_min_n", 0)),
        belief_chance_spectra=bool(worker_args["belief_chance_spectra"]),
        rescale_noise_floor_c=float(worker_args.get("rescale_noise_floor_c", 0.0)),
        sigma_eval=float(worker_args.get("sigma_eval", 0.79)),
    )
    summary = run_worker_games(
        out_dir=Path(worker_args["out_dir"]),
        games=int(worker_args["games"]),
        game_index_start=int(worker_args["game_index_start"]),
        base_seed=int(worker_args["base_seed"]),
        worker_seed=int(worker_args["worker_seed"]),
        config=config,
        search_config=search_config,
        evaluator=evaluator,
        shard_size=int(worker_args["shard_size"]),
        fmt=str(worker_args["format"]),
        opponent_pool=opponent_pool,
        opponent_mix=opponent_mix,
    )
    summary["worker_index"] = int(worker_args["worker_index"])
    return summary


def _merge_worker_summaries(
    results: list[dict[str, Any]],
    *,
    out_dir: Path,
    elapsed_sec: float,
    args: argparse.Namespace,
    opponent_pool_fraction_configured: float | None = None,
    opponent_mix_effective_weights: dict[str, float] | None = None,
    opponent_mix_exploiter_fraction: float | None = None,
) -> dict[str, Any]:
    shards: list[str] = []
    games_completed = 0
    games_failed = 0
    games_truncated = 0
    rows = 0
    decisions_total = 0
    forced_decisions_total = 0
    simulations_used_total = 0
    wins_by_color: dict[str, int] = {color: 0 for color in COLORS}
    errors: list[dict[str, Any]] = []
    worker_summaries: list[str] = []
    opponent_pool_enabled = bool(getattr(args, "opponent_pool_manifest", None))
    opponent_pool_games = 0
    # Raw (games, champion_wins) per opponent version, summed across workers
    # BEFORE dividing -- averaging each worker's own pre-divided win-rate
    # would mis-weight workers that happened to draw fewer games against a
    # given opponent version.
    opponent_pool_version_stats: dict[str, dict[str, int]] = {}
    opponent_mix_enabled = bool(getattr(args, "opponent_mix_manifest", None))
    opponent_mix_pool_games = 0
    # Raw (games, champion_wins) per CATEGORY TAG, summed across workers
    # BEFORE dividing -- same sum-then-divide reasoning as
    # opponent_pool_version_stats above.
    opponent_mix_tag_stats: dict[str, dict[str, int]] = {}
    # Exploiter lane (CAT-56): raw per-engine (games, champion_wins, divergences)
    # summed across workers before dividing, same sum-then-divide reasoning as
    # opponent_mix_tag_stats; divergence topics summed too.
    exploiter_games = 0
    exploiter_engine_stats: dict[str, dict[str, int]] = {}
    exploiter_divergence_topics: dict[str, int] = {}
    for result in sorted(results, key=lambda item: int(item.get("worker_index", 0))):
        # Defensive: only list shards that actually exist on disk, even
        # though `run_worker_games` only ever reports paths it just
        # successfully flushed -- guards against a worker reporting a path
        # that was later removed or never fully written.
        shards.extend(path for path in result.get("shards", ()) if Path(path).exists())
        games_completed += int(result.get("games_completed", 0))
        games_failed += int(result.get("games_failed", 0))
        games_truncated += int(result.get("games_truncated", 0))
        rows += int(result.get("rows", 0))
        decisions_total += int(result.get("decisions_total", 0))
        forced_decisions_total += int(result.get("forced_decisions_total", 0))
        simulations_used_total += int(result.get("simulations_used_total", 0))
        for color, count in dict(result.get("wins_by_color", {})).items():
            wins_by_color[color] = wins_by_color.get(color, 0) + int(count)
        for error in result.get("errors", ()):
            error = dict(error)
            error["worker_index"] = int(result.get("worker_index", -1))
            errors.append(error)
        if opponent_pool_enabled:
            opponent_pool_games += int(result.get("opponent_pool_games", 0))
            for version_str, stats in dict(result.get("opponent_pool_per_version_stats", {})).items():
                agg = opponent_pool_version_stats.setdefault(
                    version_str, {"games": 0, "champion_wins": 0}
                )
                agg["games"] += int(stats.get("games", 0))
                agg["champion_wins"] += int(stats.get("champion_wins", 0))
        if opponent_mix_enabled:
            opponent_mix_pool_games += int(result.get("opponent_mix_pool_games", 0))
            for tag, stats in dict(result.get("opponent_mix_per_tag_stats", {})).items():
                agg = opponent_mix_tag_stats.setdefault(tag, {"games": 0, "champion_wins": 0})
                agg["games"] += int(stats.get("games", 0))
                agg["champion_wins"] += int(stats.get("champion_wins", 0))
            exploiter_games += int(result.get("exploiter_games", 0))
            for engine, stats in dict(result.get("exploiter_per_engine_stats", {})).items():
                agg = exploiter_engine_stats.setdefault(
                    engine, {"games": 0, "champion_wins": 0, "divergences": 0}
                )
                agg["games"] += int(stats.get("games", 0))
                agg["champion_wins"] += int(stats.get("champion_wins", 0))
                agg["divergences"] += int(stats.get("divergences", 0))
            for topic, count in dict(result.get("exploiter_divergence_topics", {})).items():
                exploiter_divergence_topics[topic] = exploiter_divergence_topics.get(topic, 0) + int(count)
        # A worker that crashed in _worker_entry's except-block (before, or
        # without, reaching run_worker_games's atomic manifest write) never
        # wrote a manifest.json -- referencing that nonexistent path here
        # would silently break any downstream tool that tries to read it.
        out_dir_value = result.get("out_dir")
        if out_dir_value:
            manifest_candidate = Path(out_dir_value) / "manifest.json"
            if manifest_candidate.exists():
                worker_summaries.append(str(manifest_candidate))

    return {
        "out_dir": str(out_dir),
        "track": args.track,
        "vps_to_win": int(args.vps_to_win),
        "colors": list(COLORS),
        "games_requested": int(args.games),
        "games_completed": int(games_completed),
        "games_failed": int(games_failed),
        "games_truncated": int(games_truncated),
        "wins_by_color": wins_by_color,
        "rows": int(rows),
        "decisions_total": int(decisions_total),
        "forced_decisions_total": int(forced_decisions_total),
        "simulations_used_total": int(simulations_used_total),
        "workers": len(results),
        "n_full": int(args.n_full),
        "n_fast": int(args.n_fast),
        "p_full": float(args.p_full),
        "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
        "lazy_interior_chance": bool(args.lazy_interior_chance),
        # getattr-defaulted (like opponent_pool_manifest below) so summaries built
        # from partial arg objects in tests predating this integration still emit
        # the field rather than AttributeError-ing.
        "exact_budget_sh": bool(getattr(args, "exact_budget_sh", False)),
        "exact_budget_sh_min_n": int(getattr(args, "exact_budget_sh_min_n", 0)),
        "rust_featurize": bool(args.rust_featurize),
        "checkpoint": args.checkpoint,
        "base_seed": int(args.base_seed),
        # Complete CLI-argument provenance so a shard batch is auditable after
        # the process exits (per build-equiv pilot-audit finding 2026-07-04).
        "cli_args": {key: value for key, value in vars(args).items()},
        "elapsed_sec": elapsed_sec,
        "rows_per_sec": rows / max(elapsed_sec, 1.0e-9),
        "shards": shards,
        "worker_summaries": worker_summaries,
        "errors": errors,
        "opponent_pool_enabled": opponent_pool_enabled,
        "opponent_pool_manifest": getattr(args, "opponent_pool_manifest", None),
        "opponent_pool_fraction_configured": opponent_pool_fraction_configured,
        "opponent_pool_games": int(opponent_pool_games) if opponent_pool_enabled else 0,
        "opponent_pool_fraction_realized": (
            (opponent_pool_games / games_completed) if opponent_pool_enabled and games_completed else 0.0
        ),
        "opponent_pool_versions_used": (
            sorted(int(v) for v in opponent_pool_version_stats) if opponent_pool_enabled else []
        ),
        "opponent_pool_per_version_champion_winrate": (
            {
                version_str: (stats["champion_wins"] / stats["games"] if stats["games"] else 0.0)
                for version_str, stats in sorted(
                    opponent_pool_version_stats.items(), key=lambda item: int(item[0])
                )
            }
            if opponent_pool_enabled
            else {}
        ),
        "opponent_mix_enabled": opponent_mix_enabled,
        "opponent_mix_manifest": getattr(args, "opponent_mix_manifest", None),
        "opponent_mix_effective_weights": (opponent_mix_effective_weights or {}) if opponent_mix_enabled else {},
        "opponent_mix_pool_games": int(opponent_mix_pool_games) if opponent_mix_enabled else 0,
        "opponent_mix_pool_fraction_realized": (
            (opponent_mix_pool_games / games_completed) if opponent_mix_enabled and games_completed else 0.0
        ),
        "opponent_mix_tags_used": (
            sorted(opponent_mix_tag_stats) if opponent_mix_enabled else []
        ),
        "opponent_mix_per_tag_champion_winrate": (
            {
                tag: (stats["champion_wins"] / stats["games"] if stats["games"] else 0.0)
                for tag, stats in sorted(opponent_mix_tag_stats.items())
            }
            if opponent_mix_enabled
            else {}
        ),
        # Exploiter lane (CAT-56).
        "exploiter_fraction_arg": getattr(args, "exploiter_fraction", None),
        "exploiter_fraction_cap": EXTERNAL_ENGINE_FRACTION_CAP,
        "opponent_mix_exploiter_fraction_effective": (
            opponent_mix_exploiter_fraction if opponent_mix_enabled else None
        ),
        "exploiter_enabled": bool(exploiter_engine_stats),
        "exploiter_games": int(exploiter_games),
        "exploiter_fraction_realized": (
            (exploiter_games / games_completed) if games_completed else 0.0
        ),
        "exploiter_engines_used": sorted(exploiter_engine_stats),
        "exploiter_per_engine_stats": {
            engine: dict(stats) for engine, stats in sorted(exploiter_engine_stats.items())
        },
        "exploiter_per_engine_champion_winrate": {
            engine: (stats["champion_wins"] / stats["games"] if stats["games"] else 0.0)
            for engine, stats in sorted(exploiter_engine_stats.items())
        },
        "exploiter_divergence_topics": dict(sorted(exploiter_divergence_topics.items())),
        # Raw summed (games, champion_wins) per opponent version -- the un-divided
        # counts behind the winrate dict above. Exposed so downstream tools
        # (e.g. tools/run_exploit_probe.py's exploit-rate report) get exact
        # numerator/denominator rather than re-deriving them from a rounded
        # ratio. Additive, pool-path only; the normal (non-pool) summary is
        # unchanged.
        "opponent_pool_per_version_stats": (
            {
                version_str: dict(stats)
                for version_str, stats in sorted(
                    opponent_pool_version_stats.items(), key=lambda item: int(item[0])
                )
            }
            if opponent_pool_enabled
            else {}
        ),
    }


if __name__ == "__main__":
    main()
