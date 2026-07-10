#!/usr/bin/env python3
"""CLI: search-SNR probe (CAT-25 measurement 1, mechanism A: SNR-decay).

Runs `GumbelChanceMCTS.search(state.copy(), force_full=True)` TWICE per fixed
root state, with two different seeds but an otherwise IDENTICAL search config
(same n_full), and measures how much the resulting `improved_policy` (pi')
disagrees between the two runs. Root states are sampled the same way
tools/gumbel_search_self_agreement_sweep.py does: real games played out with
the checkpoint's own raw (argmax) policy, snapshotted at a spread of decision
indices -- this IS the "root-sampling" logic referenced in the CAT-25 ticket,
reused directly via `collect_fixed_states` rather than re-derived.

The observable this probe targets: if mechanism A (SNR-decay in the search's
min-max Q-rescale) dominates the plateau, `kl_pi_vs_prior` (how much the
search improves on the raw prior) should stay roughly FLAT across a
checkpoint lineage while `argmax_agreement` (the seed-noise floor of the
search's own root decision) DECAYS -- i.e. the search is spending the same
"visible" amount of improvement-over-prior, but an increasing fraction of
that improvement is unstable sampling noise rather than real signal.

Base-seed choice: 610001. Existing base-seed blocks already in use elsewhere
in tools/ are 500001 (f74_symmetry_eval / sigma trace), 600001 (opening_panel
panel roots), and 70001 (gumbel_search_self_agreement_sweep) -- 610001 is
clearly outside all of those.

--dry-run runs the KL/agreement aggregation over synthetic per-state
(pi1, pi2, prior) dicts constructed in-process, producing a real JSON output
of the same shape as a live run, with no checkpoint/GPU/rust dependency --
this is what makes the probe's math unit-testable in CI.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from diag_common import argmax_agreement, kl_divergence
from factory_common import write_json

COLORS: tuple[str, ...] = ("RED", "BLUE")

# Base-seed block for this probe -- see module docstring for why 610001 is
# clear of the other base-seed blocks already in use in tools/.
DEFAULT_BASE_SEED = 610001


def compute_state_metrics(
    pi1: dict[int, float],
    priors1: dict[int, float],
    pi2: dict[int, float],
    priors2: dict[int, float],
    *,
    prior_tol: float = 1e-6,
) -> dict[str, Any]:
    """Per-state metrics comparing two independent search() calls at the same
    fixed root state and n_full, differing only in RNG seed.

    `kl_pi1_pi2` / `kl_pi2_pi1` are the two (asymmetric) directions; their
    mean is the aggregate-friendly symmetrized measure. `kl_pi_vs_prior` is
    the mean of KL(pi1‖priors1) and KL(pi2‖priors2) -- how much each run's
    search improved on the raw prior. `priors_match` is a correctness check:
    both runs use the same checkpoint/state, so priors1 and priors2 (the raw
    network prior, unaffected by search RNG) should be identical.
    """
    kl_12 = kl_divergence(pi1, pi2)
    kl_21 = kl_divergence(pi2, pi1)
    kl_mean = (kl_12 + kl_21) / 2.0

    agree = argmax_agreement(pi1, pi2)

    kl_prior1 = kl_divergence(pi1, priors1)
    kl_prior2 = kl_divergence(pi2, priors2)
    kl_vs_prior_mean = (kl_prior1 + kl_prior2) / 2.0

    prior_keys = set(priors1) | set(priors2)
    priors_match = all(
        abs(float(priors1.get(k, 0.0)) - float(priors2.get(k, 0.0))) <= prior_tol
        for k in prior_keys
    )

    return {
        "argmax_agreement": bool(agree),
        "kl_pi1_pi2": float(kl_12),
        "kl_pi2_pi1": float(kl_21),
        "kl_pi1_pi2_mean": float(kl_mean),
        "kl_pi_vs_prior_1": float(kl_prior1),
        "kl_pi_vs_prior_2": float(kl_prior2),
        "kl_pi_vs_prior_mean": float(kl_vs_prior_mean),
        "priors_match": bool(priors_match),
    }


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def aggregate_per_state_records(per_state: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of `compute_state_metrics` outputs (one per state,
    for a single checkpoint) into mean/median summary statistics."""
    n = len(per_state)
    agreement = [1.0 if r["argmax_agreement"] else 0.0 for r in per_state]
    kl_pi = [r["kl_pi1_pi2_mean"] for r in per_state]
    kl_prior = [r["kl_pi_vs_prior_mean"] for r in per_state]
    priors_mismatch_count = sum(1 for r in per_state if not r["priors_match"])
    return {
        "n_states": n,
        "mean_argmax_agreement": _mean(agreement),
        "median_argmax_agreement": _median(agreement),
        "mean_kl_pi1_pi2": _mean(kl_pi),
        "median_kl_pi1_pi2": _median(kl_pi),
        "mean_kl_pi_vs_prior": _mean(kl_prior),
        "median_kl_pi_vs_prior": _median(kl_prior),
        "priors_mismatch_count": priors_mismatch_count,
    }


def build_checkpoint_report(checkpoint: str, per_state: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "aggregate": aggregate_per_state_records(per_state),
        "per_state": per_state,
    }


def run_probe_on_checkpoint(
    checkpoint: str,
    *,
    n_states: int,
    decisions_per_game: tuple[int, ...],
    n_full: int,
    max_depth: int,
    base_seed: int,
    device: str,
    correct_rust_chance_spectra: bool = True,
    public_observation: bool = False,
    information_set_search: bool = False,
    determinization_particles: int = 4,
    determinization_min_simulations: int = 32,
    lazy_interior_chance: bool = False,
    c_visit: float = 50.0,
    c_scale: float = 0.1,
    prior_temperature: float = 1.0,
    value_scale: float = 1.0,
    rust_featurize: bool = False,
    symmetry_averaged_eval: bool = False,
    wide_candidates_threshold: int = 24,
) -> dict[str, Any]:
    """Live (GPU/rust-dependent) path: load the checkpoint, sample fixed
    states, run two seeded searches per state, and return this checkpoint's
    report (same shape `build_checkpoint_report` produces in --dry-run)."""
    from gumbel_search_self_agreement_sweep import collect_fixed_states
    from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTS, GumbelChanceMCTSConfig
    from catan_zero.search.neural_rust_mcts import (
        BatchedEntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )

    evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
        checkpoint,
        device=device,
        config=EntityGraphRustEvaluatorConfig(
            public_observation=bool(public_observation),
            prior_temperature=float(prior_temperature),
            value_scale=float(value_scale),
            rust_featurize=bool(rust_featurize),
        ),
    )
    try:
        states = collect_fixed_states(
            evaluator,
            n_states=int(n_states),
            decisions_per_game=decisions_per_game,
            base_seed=int(base_seed),
        )
        per_state: list[dict[str, Any]] = []
        for state_index, state in enumerate(states):
            runs: list[dict[str, dict[int, float]]] = []
            for run_index in range(2):
                seed = int(base_seed) + int(n_full) * 1000 + state_index * 2 + run_index
                config = GumbelChanceMCTSConfig(
                    colors=COLORS,
                    seed=seed,
                    n_full=int(n_full),
                    n_fast=int(n_full),
                    p_full=1.0,
                    max_depth=int(max_depth),
                    temperature=0.0,
                    correct_rust_chance_spectra=bool(correct_rust_chance_spectra),
                    lazy_interior_chance=bool(lazy_interior_chance),
                    c_visit=float(c_visit),
                    c_scale=float(c_scale),
                    symmetry_averaged_eval=bool(symmetry_averaged_eval),
                    wide_candidates_threshold=int(wide_candidates_threshold),
                    information_set_search=bool(information_set_search),
                    determinization_particles=int(determinization_particles),
                    determinization_min_simulations=int(
                        determinization_min_simulations
                    ),
                )
                mcts = GumbelChanceMCTS(config, evaluator)
                result = mcts.search(state.copy(), force_full=True)
                runs.append({"pi": dict(result.improved_policy), "priors": dict(result.priors)})
            metrics = compute_state_metrics(
                runs[0]["pi"], runs[0]["priors"], runs[1]["pi"], runs[1]["priors"]
            )
            metrics["state_index"] = state_index
            per_state.append(metrics)
    finally:
        evaluator.close()

    return build_checkpoint_report(checkpoint, per_state)


def _synthetic_checkpoint_report(checkpoint: str, n_states: int, *, seed_offset: int = 0) -> dict[str, Any]:
    """Deterministic synthetic per-state (pi1, pi2, prior) construction for
    --dry-run: no checkpoint/GPU/rust import at all. A small hand-built
    3-action support per state, with a seed-dependent perturbation so
    different states/checkpoints produce varied (but reproducible) metrics."""
    per_state: list[dict[str, Any]] = []
    for state_index in range(n_states):
        base = 0.6 + 0.05 * ((state_index + seed_offset) % 3)
        rest = (1.0 - base) / 2.0
        prior = {0: base, 1: rest, 2: rest}
        # pi1/pi2 perturb the prior slightly, with the perturbation direction
        # alternating by state_index so some states agree and some don't.
        wiggle = 0.05 if (state_index + seed_offset) % 2 == 0 else -0.05
        pi1 = {0: max(0.01, base + wiggle), 1: rest, 2: rest - wiggle}
        pi2 = {0: max(0.01, base - wiggle), 1: rest, 2: rest + wiggle}
        # Renormalize (cheap, avoids fussy negative probabilities).
        def _norm(d: dict[int, float]) -> dict[int, float]:
            total = sum(d.values())
            return {k: v / total for k, v in d.items()}

        pi1 = _norm(pi1)
        pi2 = _norm(pi2)
        metrics = compute_state_metrics(pi1, prior, pi2, dict(prior))
        metrics["state_index"] = state_index
        per_state.append(metrics)
    return build_checkpoint_report(checkpoint, per_state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search-SNR probe: does argmax_agreement decay across checkpoints "
            "while kl_pi_vs_prior stays flat (mechanism A: SNR-decay)?"
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint path; repeatable for a multi-checkpoint lineage sweep.",
    )
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Comma-separated checkpoint paths (alternative to repeated --checkpoint).",
    )
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument(
        "--decisions-per-game",
        default="20,50,80,110",
        help="comma-separated decision indices to snapshot per generating game",
    )
    parser.add_argument("--n-full", type=int, default=64, help="Fixed search budget (not a sweep).")
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument("--c-scale", type=float, default=0.1)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument(
        "--public-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask hidden opponent information in the evaluator. Pass this for masked champions.",
    )
    parser.add_argument(
        "--information-set-search", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--determinization-particles", type=int, default=4)
    parser.add_argument("--determinization-min-simulations", type=int, default=32)
    parser.add_argument(
        "--lazy-interior-chance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the production lazy interior-chance search path when requested.",
    )
    parser.add_argument(
        "--rust-featurize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the parity-gated native Rust entity featurizer.",
    )
    parser.add_argument(
        "--symmetry-averaged-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable 12-way D6 averaging at wide roots; leave off to measure the pre-denoise baseline.",
    )
    parser.add_argument("--wide-candidates-threshold", type=int, default=24)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip checkpoint/evaluator/rust loading; aggregate synthetic per-state records instead.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.public_observation) != bool(args.information_set_search):
        parser.error(
            "--public-observation and --information-set-search must be enabled together"
        )
    if int(args.determinization_particles) < 1:
        parser.error("--determinization-particles must be >= 1")
    if int(args.determinization_min_simulations) < 1:
        parser.error("--determinization-min-simulations must be >= 1")

    checkpoints: list[str] = list(args.checkpoint or [])
    if args.checkpoints:
        checkpoints.extend(c.strip() for c in args.checkpoints.split(",") if c.strip())
    if not checkpoints:
        if args.dry_run:
            checkpoints = ["synthetic-checkpoint-0", "synthetic-checkpoint-1"]
        else:
            parser.error("at least one --checkpoint (or --checkpoints) is required")

    decisions_per_game = tuple(int(x) for x in args.decisions_per_game.split(","))

    started = time.perf_counter()
    per_checkpoint: dict[str, Any] = {}
    if args.dry_run:
        for offset, checkpoint in enumerate(checkpoints):
            per_checkpoint[checkpoint] = _synthetic_checkpoint_report(
                checkpoint, int(args.n_states), seed_offset=offset
            )
    else:
        for checkpoint in checkpoints:
            per_checkpoint[checkpoint] = run_probe_on_checkpoint(
                checkpoint,
                n_states=int(args.n_states),
                decisions_per_game=decisions_per_game,
                n_full=int(args.n_full),
                max_depth=int(args.max_depth),
                base_seed=int(args.base_seed),
                device=args.device,
                public_observation=bool(args.public_observation),
                information_set_search=bool(args.information_set_search),
                determinization_particles=int(args.determinization_particles),
                determinization_min_simulations=int(
                    args.determinization_min_simulations
                ),
                lazy_interior_chance=bool(args.lazy_interior_chance),
                c_visit=float(args.c_visit),
                c_scale=float(args.c_scale),
                prior_temperature=float(args.prior_temperature),
                value_scale=float(args.value_scale),
                rust_featurize=bool(args.rust_featurize),
                symmetry_averaged_eval=bool(args.symmetry_averaged_eval),
                wide_candidates_threshold=int(args.wide_candidates_threshold),
            )
    elapsed = time.perf_counter() - started

    summary = {
        "measurement": "search_snr_probe",
        "mechanism": "A_snr_decay",
        "dry_run": bool(args.dry_run),
        "checkpoints": checkpoints,
        "n_states": int(args.n_states),
        "decisions_per_game": list(decisions_per_game),
        "n_full": int(args.n_full),
        "max_depth": int(args.max_depth),
        "base_seed": int(args.base_seed),
        "search_config": {
            "c_visit": float(args.c_visit),
            "c_scale": float(args.c_scale),
            "correct_rust_chance_spectra": True,
            "lazy_interior_chance": bool(args.lazy_interior_chance),
            "public_observation": bool(args.public_observation),
            "information_set_search": bool(args.information_set_search),
            "determinization_particles": int(args.determinization_particles),
            "determinization_min_simulations": int(
                args.determinization_min_simulations
            ),
            "prior_temperature": float(args.prior_temperature),
            "value_scale": float(args.value_scale),
            "rust_featurize": bool(args.rust_featurize),
            "symmetry_averaged_eval": bool(args.symmetry_averaged_eval),
            "wide_candidates_threshold": int(args.wide_candidates_threshold),
        },
        "elapsed_sec": elapsed,
        "per_checkpoint": per_checkpoint,
    }
    write_json(args.out, summary)
    print(
        json.dumps(
            {
                "measurement": summary["measurement"],
                "checkpoints": checkpoints,
                "aggregate_by_checkpoint": {
                    checkpoint: report["aggregate"] for checkpoint, report in per_checkpoint.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
