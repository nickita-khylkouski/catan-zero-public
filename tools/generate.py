#!/usr/bin/env python3
"""Canonical config-first self-play launcher.

The implementation executor still accepts the historical flag surface so old
sealed commands remain replayable.  New runs must come through this entrypoint:
science lives in one schema-versioned JSON config, while the command line is
limited to run identity and placement.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from catan_zero.rl.pipeline_configs import (  # noqa: E402
    CONFIG_SCHEMA_VERSION,
    GenerateConfig,
)
from catan_zero.rl.production_recipe_catalog import (  # noqa: E402
    require_production_recipe,
)

from generate_gumbel_selfplay_data import main as _legacy_executor_main  # noqa: E402


CANONICAL_OPTION_COUNT = 9
REQUIRED_NOFILE_SOFT = 65_536

# These are not tuning knobs on the production path anymore.  Historical
# configs remain replayable through generate_gumbel_selfplay_data.py, while the
# canonical launcher fails closed if an old experiment is accidentally revived.
RETIRED_NOOP_FIELDS = {
    "belief_chance_spectra": False,
    "information_set_search": False,
    "information_set_target_aggregation": "mean_improved_policy",
    "n_full_wide": None,
    "n_full_wide_threshold": None,
    "wide_roots_always_full": False,
    "raw_policy_above_width": None,
    "sigma_reference_visits": None,
    "rescale_noise_floor_c": 0.0,
    "rescale_noise_floor_initial_road_only": False,
    "value_readout": "scalar",
    "exact_budget_sh": False,
    "exact_budget_sh_min_n": 0,
    "root_wave_batching": False,
    "opponent_pool_manifest": None,
}

REQUIRED_SCIENCE_FIELDS = {
    "track": "2p_no_trade",
    "vps_to_win": 10,
    "obs_width": 806,
    "n_full": 128,
    "n_fast": 16,
    "p_full": 0.25,
    "c_visit": 50.0,
    "c_scale": 0.1,
    "sigma_eval": 0.79,
    "max_decisions": 600,
    "max_depth": 80,
    "temperature_decisions": 40,
    "temperature_clock": "nonforced_choice",
    "temperature_high": 1.0,
    "temperature_low": 0.0,
    "late_temperature_decisions": 100,
    "late_temperature": 0.1,
    "prior_temperature": 1.0,
    "value_scale": 1.0,
    "public_observation": True,
    "coherent_public_belief_search": True,
    "determinization_particles": 1,
    "determinization_min_simulations": 32,
    "forced_root_target_mode": "trajectory_only",
    "boundary_value_particles": 1,
    "correct_rust_chance_spectra": True,
    "lazy_interior_chance": True,
    "symmetry_averaged_eval": True,
    "symmetry_averaged_eval_threshold": 20,
    "wide_candidates_threshold": 24,
    "native_mcts_hot_loop": True,
    "rust_featurize": True,
    # Retained H100 frontier: one strict-FP32 policy process batches leaves
    # across game workers. TF32, CUDA graphs, and shared-memory transport were
    # all measured and rejected. Do not set event_token_limit=0 here: unlike
    # the historical empty-event checkpoint, the canonical learner consumes
    # meaningful public history.
    "eval_server": True,
    "eval_server_max_batch": 96,
    "eval_server_max_wait_ms": 0.0,
    "eval_server_matmul_precision": "highest",
    "eval_server_request_collector": True,
    "eval_server_transport": "mp_queue",
    "eval_server_event_token_limit": None,
    "eval_server_cuda_graph": False,
    "eval_server_local_fallback": False,
    "record_automatic_transitions": True,
    "meaningful_public_history": True,
    "event_history_limit": 64,
    "learner_entity_feature_adapter_version": (
        "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
    ),
    "teacher_entity_feature_adapter_version": (
        "rust_entity_adapter_v2_land_topology_ports_maritime"
    ),
    "public_card_count_feature_schema": "public_card_state_v2",
    "preserve_search_evidence": True,
    "target_reliability_audit_fraction": 0.05,
    "target_reliability_audit_seed": 20260716,
    "opponent_mix_manifest": None,
    "exploiter_fraction": None,
    "fmt": "npz",
    "shard_size": 512,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Placement override; defaults to the 8xH100 config's commissioned 24 "
            "cross-game EvalServer workers."
        ),
    )
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--claim-label", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Executor-managed retry after the incomplete output was quarantined.",
    )
    return parser


def _validate_config(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load generation config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("generation config must contain a JSON object")
    require_production_recipe(entrypoint="generate", path=path, payload=payload)
    if payload.get("pipeline") != GenerateConfig.PIPELINE:
        raise ValueError(
            f"generation config pipeline must be {GenerateConfig.PIPELINE!r}"
        )
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "generation config schema mismatch: "
            f"expected={CONFIG_SCHEMA_VERSION} actual={payload.get('schema_version')!r}"
        )
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("generation config fields must be a JSON object")
    for name, expected in RETIRED_NOOP_FIELDS.items():
        if fields.get(name) != expected:
            raise ValueError(
                f"retired generation experiment {name!r} must remain "
                f"{expected!r} on the canonical path"
            )
    for name, expected in REQUIRED_SCIENCE_FIELDS.items():
        if fields.get(name) != expected:
            raise ValueError(
                f"canonical generation field {name!r} must be {expected!r}, "
                f"got {fields.get(name)!r}"
            )


def _executor_argv(args: argparse.Namespace) -> list[str]:
    output = args.out_dir.expanduser()
    forwarded = [
        "--config",
        str(args.config.expanduser()),
        "--prelaunch-guard-config",
        str(args.guard.expanduser()),
        "--checkpoint",
        str(args.checkpoint.expanduser()),
        "--out-dir",
        str(output),
        "--games",
        str(args.games),
        "--base-seed",
        str(args.base_seed),
        "--ledger-claim-label",
        str(args.claim_label),
        "--dump-config",
        str(output / "config.registry.jsonl"),
        "--config-purpose",
        args.config.stem,
    ]
    if args.workers is not None:
        forwarded.extend(("--workers", str(args.workers)))
    if args.resume:
        forwarded.append("--resume")
    return forwarded


def _ensure_runtime_limits() -> None:
    """Raise the worker FD budget before the legacy prelaunch guard runs."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    unlimited = resource.RLIM_INFINITY
    if hard != unlimited and hard < REQUIRED_NOFILE_SOFT:
        raise RuntimeError(
            f"hard RLIMIT_NOFILE {hard} is below required {REQUIRED_NOFILE_SOFT}"
        )
    if soft != unlimited and soft < REQUIRED_NOFILE_SOFT:
        resource.setrlimit(resource.RLIMIT_NOFILE, (REQUIRED_NOFILE_SOFT, hard))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    public_actions = {
        id(action)
        for action in parser._actions  # noqa: SLF001
        if action.option_strings and action.dest != "help"
    }
    if len(public_actions) != CANONICAL_OPTION_COUNT:
        parser.error("canonical generation CLI exceeded its ten-option budget")
    if args.games < 1:
        parser.error("--games must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.base_seed < 0:
        parser.error("--base-seed must be non-negative")
    try:
        _validate_config(args.config.expanduser())
        _ensure_runtime_limits()
    except ValueError as error:
        parser.error(str(error))
    except (OSError, RuntimeError) as error:
        parser.error(f"cannot prepare generation runtime: {error}")
    _legacy_executor_main(_executor_argv(args))


if __name__ == "__main__":
    main(sys.argv[1:])
