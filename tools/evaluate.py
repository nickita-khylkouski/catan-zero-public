#!/usr/bin/env python3
"""Canonical config-first candidate-versus-champion evaluator.

All search and evaluator science is supplied by one schema-versioned EvalConfig.
The command line contains only matchup identity and execution placement.  The
large historical H2H CLI remains available solely for sealed replay and R&D.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from catan_zero.rl.pipeline_configs import CONFIG_SCHEMA_VERSION, EvalConfig


CANONICAL_OPTION_COUNT = 9
CANONICAL_CONFIG_SHA256 = (
    "ccfb45ab4b5b3842f0601f5145d2c3f9f7c28ba935361107d2094a6e2222df9a"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--held-out-suite", type=Path)
    return parser


def _validate_config(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load evaluation config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must contain a JSON object")
    payload_sha256 = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    if payload_sha256 != CANONICAL_CONFIG_SHA256:
        raise ValueError(
            "evaluation config is not the exact commissioned canonical payload: "
            f"expected_sha256={CANONICAL_CONFIG_SHA256} "
            f"actual_sha256={payload_sha256}"
        )
    if payload.get("pipeline") != EvalConfig.PIPELINE:
        raise ValueError(f"evaluation config pipeline must be {EvalConfig.PIPELINE!r}")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "evaluation config schema mismatch: "
            f"expected={CONFIG_SCHEMA_VERSION} actual={payload.get('schema_version')!r}"
        )
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("evaluation config fields must be a JSON object")
    required = {
        "mode": "cross_net",
        "public_observation": True,
        "belief_chance_spectra": False,
        "information_set_search": False,
        "coherent_public_belief_search": True,
        "forced_root_target_mode": "trajectory_only",
        "boundary_value_particles": 1,
        "native_mcts_hot_loop": True,
        "n_full": 128,
        "candidate_n_full": 128,
        "baseline_n_full": 128,
        "c_visit": 50.0,
        "c_scale": 0.1,
        "candidate_c_scale": 0.1,
        "baseline_c_scale": 0.1,
        "max_depth": 80,
        "max_decisions": 600,
        "gameplay_policy_aggregation": "mean_improved_policy",
        "candidate_gameplay_policy_aggregation": "mean_improved_policy",
        "baseline_gameplay_policy_aggregation": "mean_improved_policy",
        "n_full_wide": None,
        "candidate_n_full_wide": None,
        "baseline_n_full_wide": None,
        "raw_policy_above_width": None,
        "candidate_raw_policy_above_width": None,
        "baseline_raw_policy_above_width": None,
        "sigma_reference_visits": None,
        "rescale_noise_floor_c": 0.0,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
        "wide_candidates_threshold": 24,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "prior_temperature": 1.0,
        "value_scale": 1.0,
        "value_readout": "scalar",
        "candidate_value_readout": "scalar",
        "baseline_value_readout": "scalar",
        "value_squash": "tanh",
        "candidate_value_squash": "tanh",
        "baseline_value_squash": "tanh",
        "exact_budget_sh": False,
        "root_wave_batching": False,
        "uncertainty_backup_weighting": False,
        "variance_aware_q": False,
        "evaluator_rust_featurize": True,
        "evaluator_cache_size": 0,
        "force_full_every_decision": True,
        "use_batch_api": True,
        "map_kind": "BASE",
    }
    for name, expected in required.items():
        if fields.get(name) != expected:
            raise ValueError(
                f"canonical evaluation field {name!r} must be {expected!r}, "
                f"got {fields.get(name)!r}"
            )


def _executor_argv(args: argparse.Namespace) -> list[str]:
    forwarded = [
        str(Path(__file__).with_name("gumbel_search_cross_net_h2h.py")),
        "--config",
        str(args.config.expanduser()),
        "--candidate",
        str(args.candidate.expanduser()),
        "--baseline",
        str(args.champion.expanduser()),
        "--out",
        str(args.out.expanduser()),
        "--pairs",
        str(args.pairs),
        "--workers",
        str(args.workers),
        "--devices",
        args.devices,
        "--base-seed",
        str(args.base_seed),
        "--dump-config",
        str(args.out.expanduser().with_suffix(".config.json")),
        "--config-purpose",
        args.config.stem,
    ]
    if args.held_out_suite is not None:
        forwarded.extend(
            ["--held-out-high-regret-suite", str(args.held_out_suite.expanduser())]
        )
    return forwarded


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    public_actions = {
        id(action)
        for action in parser._actions  # noqa: SLF001
        if action.option_strings and action.dest != "help"
    }
    if len(public_actions) != CANONICAL_OPTION_COUNT:
        parser.error("canonical evaluation CLI exceeded its ten-option budget")
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.base_seed < 0:
        parser.error("--base-seed must be non-negative")
    try:
        _validate_config(args.config.expanduser())
    except ValueError as error:
        parser.error(str(error))
    os.execv(sys.executable, [sys.executable, *_executor_argv(args)])


if __name__ == "__main__":
    main(sys.argv[1:])
