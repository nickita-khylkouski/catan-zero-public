#!/usr/bin/env python3
"""Freeze, verify, render, and postflight-audit the A1 40-GPU data handoff.

This tool is deliberately *not* a launcher.  It turns the winning bounded-R&D
artifacts into an immutable contract and renders argv/environment records for
the data-production lane.  There is no subprocess/exec path: the 40-GPU wave
remains an explicit operator boundary.

The contract uses category-specific jobs rather than a probabilistic opponent
mix.  Each of 40 workers attempts 245 current, 47 history, and 16 hard-negative
games, then the postflight deterministically selects the lowest-seed complete
240/45/15 per job.  This gives exactly 9,600/1,800/600 selected games before
row expansion while tolerating only the bounded, predeclared reserve.  The
audit rejects an insufficient complete quota, duplicate or VAL-ONLY seeds,
invalid selected actions, config drift, and missing shard provenance.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TOOLS = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig  # noqa: E402
from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig  # noqa: E402
from catan_zero.rl.gumbel_self_play import GumbelSelfPlayConfig  # noqa: E402
from catan_zero.rl.pipeline_configs import GenerateConfig  # noqa: E402
from tools import legacy_scalar_readout_attestation as legacy_scalar  # noqa: E402
from tools import a0_binding_verdict as a0_binding  # noqa: E402
from tools import search_teacher_adjudicator as search_adjudicator  # noqa: E402
from tools import search_operator_binding as operator_binding  # noqa: E402
from tools.prelaunch_guard import VAL_ONLY_SEED_RANGE, parse_seed_ledger  # noqa: E402
from tools.seed_fleet_planner import assert_disjoint_seed_blocks  # noqa: E402

DRAFT_SCHEMA = "a1-pre-wave-contract-draft-v2"
LOCK_SCHEMA = "a1-pre-wave-contract-lock-v2"
RENDER_SCHEMA = "a1-pre-wave-render-v2"
AUDIT_SCHEMA = "a1-post-wave-audit-v2"
GUARD_SYNC_SCHEMA = "a1-pre-wave-generation-guard-sync-v1"
GUARD_SYNC_KEY = "a1_pre_wave_guard_sync"
GUARD_SYNC_TOOL = "tools/a1_pre_wave_contract.py"
DEFAULT_GENERATION_C_SCALE = 0.03
UNRESOLVED = "__UNRESOLVED__"
EXPECTED_GAMES = {
    "current_producer": 9_600,
    "recent_history": 1_800,
    "hard_negative": 600,
}
EXPECTED_WORKER_COUNT = 40
EXPECTED_PER_WORKER = {
    "current_producer": 240,
    "recent_history": 45,
    "hard_negative": 15,
}
# A bounded reserve makes the selected zero-truncation quota achievable when
# otherwise healthy production has rare max-decision truncations.  Postflight
# deterministically selects the lowest-seed complete games from each job.
EXPECTED_ATTEMPTS_PER_WORKER = {
    "current_producer": 245,
    "recent_history": 47,
    "hard_negative": 16,
}
EXPECTED_ATTEMPTS = {
    category: attempts * EXPECTED_WORKER_COUNT
    for category, attempts in EXPECTED_ATTEMPTS_PER_WORKER.items()
}

# One deliberately short, single-B200 learner dose.  A search/data contract is
# not scientifically reproducible if the optimizer or loss mixture can drift
# after the wave, so these are effective values rather than a partial CLI
# overlay.  Keep the keys aligned with train_bc's resolved Namespace; the two
# derived topology fields make the batch semantics explicit.
EXPECTED_LEARNER_TRAINING_RECIPE: dict[str, Any] = {
    "track": "2p_no_trade",
    "vps_to_win": 10,
    "graph_history_features": True,
    "seed": 1,
    "epochs": 1,
    "max_steps": 0,
    "batch_size": 4096,
    "grad_accum_steps": 1,
    "world_size": 1,
    "global_batch_size": 4096,
    "optimizer": "adam",
    "resume_optimizer": False,
    "lr": 3e-5,
    "lr_warmup_steps": 100,
    "lr_schedule": "flat",
    "weight_decay": 0.0,
    "fused_optimizer": False,
    "value_lr_mult": 0.3,
    "action_module_lr_mult": 1.0,
    "policy_loss_weight": 1.0,
    "soft_target_source": "policy",
    "soft_target_weight": 0.9,
    "soft_target_temperature": 0.7,
    "soft_target_min_legal_coverage": 0.5,
    "value_loss_weight": 0.25,
    "value_target_lambda": 1.0,
    "value_categorical_loss_weight": 0.0,
    "hlgauss_scalar_aux_loss_weight": 0.0,
    "final_vp_loss_weight": 0.0,
    "q_loss_weight": 0.0,
    "policy_kl_anchor_weight": 0.0,
    "value_uncertainty_loss_weight": 0.0,
    "aux_subgoal_loss_weight": 0.0,
    "train_value_only": False,
    "freeze_modules": "",
    "policy_surprise_weight": 0.0,
    "advantage_policy_weighting": "none",
    "per_game_value_weight": False,
    "vp_margin_weight": 0.0,
    "truncated_vp_margin_value_weight": 0.25,
    "amp": "bf16",
    "mask_hidden_info": True,
    "symmetry_augment": False,
    "forced_action_weight": 0.1,
    "forced_row_value_weight": 1.0,
    "winner_sample_weight": 1.0,
    "loser_sample_weight": 0.3,
    "teacher_weights": "",
    "phase_weights": "",
    "value_phase_weights": "",
    "ddp_shard_data": False,
}
REQUIRED_EVIDENCE = {"a0", "s1", "s2", "s3"}
A0_EVIDENCE_SCHEMA = "a0-binding-verdict-v1"
SEARCH_STAGE_EVIDENCE_SCHEMA = "rl-rnd-stage-decision-v1"
REQUIRED_REPORTS = {
    "truncation",
    "forced_fraction",
    "phase_mix",
    "decision_index_mix",
    "legal_width",
    "target_entropy",
    "full_search_policy_mass",
}
REQUIRED_SELECTED_TELEMETRY_COLUMNS = {
    "is_forced",
    "used_full_search",
    "phase",
    "decision_index",
    "target_policy",
    "target_policy_mask",
}
REQUIRED_GENERATOR_CODE_SUFFIXES = {
    "tools/generate_gumbel_selfplay_data.py",
    "tools/opponent_mix_registry.py",
    "src/catan_zero/rl/gumbel_self_play.py",
    "src/catan_zero/rl/flywheel/opponent_mix.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
    "src/catan_zero/rl/hex_symmetry.py",
    "src/catan_zero/search/gumbel_chance_mcts.py",
    "src/catan_zero/search/neural_rust_mcts.py",
    "src/catan_zero/rl/pipeline_configs.py",
}
REQUIRED_LEARNER_CODE_SUFFIXES = {
    "configs/guards/train_bc.json",
    "tools/factory_common.py",
    "tools/launcher_guards.py",
    "tools/prelaunch_guard.py",
    "tools/train_bc.py",
    "tools/build_memmap_corpus.py",
    "src/catan_zero/rl/config_cli.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
    "src/catan_zero/rl/pipeline_configs.py",
    "src/catan_zero/rl/aux_subgoal_targets.py",
    "src/catan_zero/rl/config_serialization.py",
    "src/catan_zero/rl/hex_symmetry.py",
    "src/catan_zero/rl/multiagent_env.py",
    "src/catan_zero/rl/optim_state.py",
    "src/catan_zero/rl/torch_ppo.py",
    "src/catan_zero/rl/xdim_lite_policy.py",
}
REQUIRED_RUNTIME_CODE_SUFFIXES = (
    REQUIRED_GENERATOR_CODE_SUFFIXES
    | REQUIRED_LEARNER_CODE_SUFFIXES
    | {
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/action_mask.py",
        "src/catan_zero/search/rust_mcts.py",
    }
)

_SEARCH_INPUT_KEYS = {
    "max_depth",
    "c_visit",
    "c_scale",
    "prior_temperature",
    "n_full",
    "n_fast",
    "p_full",
    "n_full_wide",
    "n_full_wide_threshold",
    "wide_roots_always_full",
    "raw_policy_above_width",
    "symmetry_averaged_eval",
    "symmetry_averaged_eval_threshold",
    "wide_candidates_threshold",
    "correct_rust_chance_spectra",
    "lazy_interior_chance",
    "exact_budget_sh",
    "exact_budget_sh_min_n",
    "belief_chance_spectra",
    "rescale_noise_floor_c",
    "sigma_eval",
}
_EVALUATOR_INPUT_KEYS = {
    "value_scale",
    "prior_temperature",
    "context_fill",
    "cache_size",
    "value_squash",
    "value_readout",
    "public_observation",
    "rust_featurize",
    "emit_uncertainty",
}
_GENERATION_KEYS = {
    "track",
    "vps_to_win",
    "obs_width",
    "max_decisions",
    "temperature_decisions",
    "temperature_high",
    "temperature_low",
    "late_temperature_decisions",
    "late_temperature",
    "workers_per_gpu",
    "shard_size",
    "format",
    "device",
    "eval_server",
}


class ContractError(ValueError):
    """A fail-closed contract validation error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - compatibility identity, SHA-256 is authoritative.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _absolute_ref(raw: str, *, base: Path) -> Path:
    path = Path(raw).expanduser()
    candidate = base / path if not path.is_absolute() else path
    # ``absolute()`` preserves ``..`` components and makes a logically correct
    # checked-in relative path fail later canonical-path provenance checks.
    # resolve(strict=False) normalizes traversal/symlinks while downstream
    # file-specific validators still produce the useful missing-file error.
    return candidate.resolve(strict=False)


def _require_exact_keys(
    value: dict[str, Any], allowed: set[str], *, where: str
) -> None:
    extra = set(value) - allowed
    missing = allowed - set(value)
    if extra or missing:
        raise ContractError(
            f"{where} fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _find_unresolved(value: Any, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if value == UNRESOLVED:
        found.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_find_unresolved(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_unresolved(child, path=f"{path}[{index}]"))
    return found


def _assert_no_unresolved(value: Any) -> None:
    unresolved = _find_unresolved(value)
    if unresolved:
        preview = ", ".join(unresolved[:12])
        raise ContractError(
            f"contract still has unresolved science fields ({preview}); finish A0/S1-S3 "
            "and replace every __UNRESOLVED__ value before seal/render"
        )


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _file_record(
    path: Path, *, kind: str, artifact_id: str | None = None
) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"{kind} is missing or not a file: {path}")
    record = {"kind": kind, "path": str(path), "sha256": _sha256(path)}
    if artifact_id is not None:
        record["id"] = artifact_id
    return record


def _runtime_code_tree_records() -> list[dict[str, Any]]:
    """Content-address the complete local Python runtime used by gen/A1.

    Explicit role lists document the expected load-bearing entry points, but
    cannot safely approximate a transitive import closure.  Hashing every
    Python module under ``src/catan_zero`` and ``tools`` closes that gap; the
    two static guard configs are included because they alter launch semantics.
    """

    paths = {
        path.resolve(strict=True)
        for pattern in ("src/catan_zero/**/*.py", "tools/**/*.py")
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    }
    paths.update(
        {
            (REPO_ROOT / "configs/guards/generate_gumbel_selfplay_data.json").resolve(
                strict=True
            ),
            (REPO_ROOT / "configs/guards/train_bc.json").resolve(strict=True),
        }
    )
    records = [_file_record(path, kind="runtime_code") for path in sorted(paths)]
    record_paths = {Path(record["path"]).as_posix() for record in records}
    missing = {
        suffix
        for suffix in REQUIRED_RUNTIME_CODE_SUFFIXES
        if not any(path.endswith(suffix) for path in record_paths)
    }
    if missing:
        raise ContractError(
            f"runtime code tree omits required transitive files: {sorted(missing)}"
        )
    return records


def _checkpoint_metadata(
    path: Path,
    *,
    checkpoint_sha256: str,
    value_readout: str,
    require_trained_readout: bool,
    legacy_scalar_attestation: Path | None,
) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - a checkpoint must be inspectable to seal.
        raise ContractError(f"cannot inspect checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"checkpoint {path} is not a mapping")
    if payload.get("mask_hidden_info") is not True:
        raise ContractError(
            f"checkpoint {path} does not attest mask_hidden_info=true; public-observation "
            "generation may not use it"
        )
    value_training = payload.get("value_training")
    positive_provenance = (
        isinstance(value_training, dict)
        and value_training.get("schema_version") == "value-training-v1"
        and value_readout
        in set(map(str, value_training.get("trained_value_readouts", [])))
    )
    if value_readout == "categorical":
        if legacy_scalar_attestation is not None:
            raise ContractError(
                "a legacy scalar-readout attestation cannot authorize categorical readout"
            )
        if not positive_provenance:
            raise ContractError(
                f"checkpoint {path} does not prove teacher readout 'categorical' was trained"
            )
        if (
            not isinstance(value_training, dict)
            or value_training.get("schema_version") != "value-training-v1"
        ):
            raise ContractError(
                f"checkpoint {path} lacks positive value-training-v1 provenance required "
                "for categorical readout"
            )
        if "categorical" not in set(
            map(str, value_training.get("trained_value_readouts", []))
        ):
            raise ContractError(
                f"checkpoint {path} categorical readout was requested but categorical training "
                "provenance is not positive"
            )
        if float(value_training.get("resolved_categorical_ce_weight", 0.0)) <= 0.0:
            raise ContractError(
                f"checkpoint {path} categorical CE weight is not positive"
            )
        if int(value_training.get("hlgauss_bins", 0)) != 33:
            raise ContractError(
                f"checkpoint {path} must attest the selected 33-bin HL-Gauss head"
            )
    elif positive_provenance:
        if float(value_training.get("resolved_scalar_mse_weight", 0.0)) <= 0.0:
            raise ContractError(f"checkpoint {path} scalar MSE weight is not positive")
    elif require_trained_readout and legacy_scalar_attestation is None:
        raise ContractError(
            f"checkpoint {path} does not prove teacher readout {value_readout!r} was trained; "
            "supply a typed legacy scalar-readout attestation"
        )

    attestation_record: dict[str, Any] | None = None
    if legacy_scalar_attestation is not None:
        if value_readout != "scalar":
            raise ContractError("legacy scalar-readout attestation is scalar-only")
        try:
            attestation = legacy_scalar.verify_attestation(
                legacy_scalar_attestation,
                expected_checkpoint_path=path,
                expected_checkpoint_sha256=checkpoint_sha256,
            )
        except legacy_scalar.AttestationError as error:
            raise ContractError(
                f"invalid legacy scalar-readout attestation: {error}"
            ) from error
        attestation_record = {
            "kind": "legacy_scalar_readout_attestation",
            "path": str(legacy_scalar_attestation),
            "sha256": _sha256(legacy_scalar_attestation),
            "schema_version": legacy_scalar.SCHEMA_VERSION,
            "attestation_sha256": attestation["attestation_sha256"],
            "checkpoint": dict(attestation["checkpoint"]),
            "report": dict(attestation["report"]),
        }
    record = {
        "mask_hidden_info": True,
        "value_training_schema": (
            "value-training-v1"
            if isinstance(value_training, dict)
            and value_training.get("schema_version") == "value-training-v1"
            else (
                legacy_scalar.SCHEMA_VERSION
                if attestation_record is not None
                else "legacy-scalar-unattested-opponent"
            )
        ),
    }
    if isinstance(value_training, dict):
        record["value_training_sha256"] = _digest_value(value_training)
    if attestation_record is not None:
        record["legacy_scalar_readout_attestation"] = attestation_record
    return record


def _effective_search(raw: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(raw, _SEARCH_INPUT_KEYS, where="science.search")
    bool_keys = {
        "wide_roots_always_full",
        "symmetry_averaged_eval",
        "correct_rust_chance_spectra",
        "lazy_interior_chance",
        "exact_budget_sh",
        "belief_chance_spectra",
    }
    int_keys = {
        "max_depth",
        "n_full",
        "n_fast",
        "wide_candidates_threshold",
        "exact_budget_sh_min_n",
    }
    optional_int_keys = {
        "n_full_wide",
        "n_full_wide_threshold",
        "raw_policy_above_width",
        "symmetry_averaged_eval_threshold",
    }
    numeric_keys = {
        "c_visit",
        "c_scale",
        "prior_temperature",
        "p_full",
        "rescale_noise_floor_c",
        "sigma_eval",
    }
    for key in bool_keys:
        if type(raw[key]) is not bool:
            raise ContractError(f"science.search.{key} must be a JSON boolean")
    for key in int_keys:
        if type(raw[key]) is not int:
            raise ContractError(f"science.search.{key} must be an integer")
    for key in optional_int_keys:
        if raw[key] is not None and type(raw[key]) is not int:
            raise ContractError(f"science.search.{key} must be an integer or null")
    for key in numeric_keys:
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
            raise ContractError(f"science.search.{key} must be numeric")
    config = GumbelChanceMCTSConfig(colors=("RED", "BLUE"), seed=0, **raw)
    effective = dataclasses.asdict(config)
    effective.pop("seed")
    return effective


def _search_operator(raw: dict[str, Any]) -> dict[str, Any]:
    """The explicit, adjudicated operator (separate from code-default fields)."""
    effective = _effective_search(raw)
    return {key: effective[key] for key in sorted(_SEARCH_INPUT_KEYS)}


def _effective_evaluator(raw: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(raw, _EVALUATOR_INPUT_KEYS, where="science.evaluator")
    for key in ("public_observation", "rust_featurize", "emit_uncertainty"):
        if type(raw[key]) is not bool:
            raise ContractError(f"science.evaluator.{key} must be a JSON boolean")
    if type(raw["cache_size"]) is not int:
        raise ContractError("science.evaluator.cache_size must be an integer")
    for key in ("value_scale", "prior_temperature", "context_fill"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
            raise ContractError(f"science.evaluator.{key} must be numeric")
    for key in ("value_squash", "value_readout"):
        if not isinstance(raw[key], str):
            raise ContractError(f"science.evaluator.{key} must be a string")
    config = EntityGraphRustEvaluatorConfig(**raw)
    return dataclasses.asdict(config)


def _guard_cli_flag_lint(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    matches = [
        spec
        for spec in payload.get("guards", [])
        if isinstance(spec, dict) and spec.get("name") == "cli_flag_lint"
    ]
    if len(matches) != 1:
        raise ContractError(
            f"{path} must have exactly one cli_flag_lint guard, found {len(matches)}"
        )
    args = matches[0].get("args")
    if not isinstance(args, dict):
        raise ContractError(f"{path} cli_flag_lint args must be an object")
    return args


def _guard_expected_values(path: Path) -> tuple[dict[str, Any], set[str]]:
    args = _guard_cli_flag_lint(_load_json(path), path=path)
    return dict(args.get("expected_values", {})), set(args.get("critical_flags", []))


def _validate_guard_sync_provenance(
    payload: dict[str, Any],
    *,
    path: Path,
    selected_c_scale: float,
    s1_evidence: dict[str, Any] | None,
) -> None:
    """Require a self-contained receipt when S1 moves the static guard.

    The default ``.03`` selection intentionally leaves the checked-in guard
    byte-for-byte untouched.  A non-default S1 selection is different: the
    guard itself must carry the exact S1 artifact and synchronizer identities,
    so the later guard/runtime hashes preserve why that mutable config changed.
    """

    receipt = payload.get(GUARD_SYNC_KEY)
    if selected_c_scale == DEFAULT_GENERATION_C_SCALE:
        if receipt is not None:
            raise ContractError(
                f"guard {path} carries stale {GUARD_SYNC_KEY} metadata for the "
                "default c_scale=0.03 selection"
            )
        return
    if not isinstance(receipt, dict):
        raise ContractError(
            f"guard {path} selects non-default c_scale={selected_c_scale!r} without "
            f"a {GUARD_SYNC_SCHEMA} receipt; run sync-generation-guard before seal"
        )
    required = {
        "schema_version",
        "selected_c_scale",
        "source_s1_evidence",
        "previous_guard_sha256",
        "synchronizer",
    }
    _require_exact_keys(receipt, required, where=f"guard {GUARD_SYNC_KEY}")
    if receipt["schema_version"] != GUARD_SYNC_SCHEMA:
        raise ContractError(f"guard {path} has an unsupported guard-sync receipt")
    if receipt["selected_c_scale"] != selected_c_scale:
        raise ContractError(f"guard {path} guard-sync selected_c_scale drift")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt["previous_guard_sha256"])):
        raise ContractError(f"guard {path} guard-sync previous SHA-256 is malformed")
    if not isinstance(s1_evidence, dict):
        raise ContractError(
            f"guard {path} cannot validate non-default c_scale without typed S1 evidence"
        )
    expected_s1 = {
        "path": str(Path(str(s1_evidence["path"])).resolve(strict=True)),
        "sha256": str(s1_evidence["sha256"]),
    }
    if receipt["source_s1_evidence"] != expected_s1:
        raise ContractError(f"guard {path} guard-sync S1 provenance drift")
    synchronizer = receipt["synchronizer"]
    if not isinstance(synchronizer, dict) or set(synchronizer) != {"path", "sha256"}:
        raise ContractError(f"guard {path} has malformed synchronizer provenance")
    if synchronizer["path"] != GUARD_SYNC_TOOL:
        raise ContractError(f"guard {path} was synchronized by an unexpected tool")
    synchronizer_path = (REPO_ROOT / GUARD_SYNC_TOOL).resolve(strict=True)
    if synchronizer["sha256"] != _sha256(synchronizer_path):
        raise ContractError(f"guard {path} synchronizer implementation drift")


def _validate_guard(
    path: Path,
    *,
    search: dict[str, Any],
    evaluator: dict[str, Any],
    generation: dict[str, Any],
    s1_evidence: dict[str, Any] | None = None,
) -> None:
    _validate_guard_payload(
        _load_json(path),
        path=path,
        search=search,
        evaluator=evaluator,
        generation=generation,
        s1_evidence=s1_evidence,
    )


def _validate_guard_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    search: dict[str, Any],
    evaluator: dict[str, Any],
    generation: dict[str, Any],
    s1_evidence: dict[str, Any] | None = None,
) -> None:
    """Validate prospective or on-disk guard bytes with identical semantics."""

    args = _guard_cli_flag_lint(payload, path=path)
    expected = dict(args.get("expected_values", {}))
    critical = set(args.get("critical_flags", []))
    required_critical = {
        "--c-scale",
        "--c-visit",
        "--n-full",
        "--n-fast",
        "--base-seed",
        "--games",
    }
    if not required_critical.issubset(critical):
        raise ContractError(
            f"guard {path} is missing critical flags {sorted(required_critical - critical)}"
        )
    comparisons = {
        "--c-scale": search["c_scale"],
        "--temperature-decisions": generation["temperature_decisions"],
        "--public-observation": evaluator["public_observation"],
        "--lazy-interior-chance": search["lazy_interior_chance"],
    }
    for flag, contract_value in comparisons.items():
        if flag not in expected or expected[flag] != contract_value:
            raise ContractError(
                f"guard drift: {path} expected_values[{flag!r}]={expected.get(flag)!r}, "
                f"winning contract requires {contract_value!r}"
            )
    _validate_guard_sync_provenance(
        payload,
        path=path,
        selected_c_scale=float(search["c_scale"]),
        s1_evidence=s1_evidence,
    )


def _atomic_replace_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a mutable config without exposing a partial JSON file."""

    path = path.resolve(strict=True)
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.sync-{os.getpid()}.tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sync_generation_guard(draft_path: Path) -> dict[str, Any]:
    """Synchronize the static generation guard from replayable typed S1.

    This is intentionally a narrow pre-seal mutation.  It cannot infer a
    choice from defaults or informal output, and it never edits the draft or
    launches work.  The common c_scale=.03 decision is a byte-for-byte no-op.
    Non-default decisions embed their S1/tool receipt inside the guard itself,
    which the subsequent guard record and runtime-tree hashes then seal.
    """

    draft_path = draft_path.resolve(strict=True)
    draft = _load_json(draft_path)
    if draft.get("schema_version") != DRAFT_SCHEMA:
        raise ContractError(f"draft schema must be {DRAFT_SCHEMA!r}")
    science = draft.get("science")
    if not isinstance(science, dict) or not isinstance(science.get("search"), dict):
        raise ContractError("draft has no science.search object")
    raw_search = dict(science["search"])
    s1_keys = {
        "c_scale",
        "symmetry_averaged_eval",
        "symmetry_averaged_eval_threshold",
        "rescale_noise_floor_c",
        "sigma_eval",
    }
    unresolved_s1 = sorted(
        key for key in s1_keys if raw_search.get(key, UNRESOLVED) == UNRESOLVED
    )
    if unresolved_s1:
        raise ContractError(
            f"cannot synchronize generation guard before typed S1 resolves {unresolved_s1}"
        )
    final_s1 = {key: raw_search[key] for key in s1_keys}
    selected_c_scale = final_s1["c_scale"]
    if isinstance(selected_c_scale, bool) or not isinstance(
        selected_c_scale, (int, float)
    ):
        raise ContractError("science.search.c_scale must be numeric")
    selected_c_scale = float(selected_c_scale)
    if selected_c_scale not in {0.03, 0.1, 0.3}:
        raise ContractError(
            "typed S1 c_scale must be one of the predeclared {.03,.1,.3} arms"
        )

    evidence_items = science.get("evidence")
    if not isinstance(evidence_items, list):
        raise ContractError("science.evidence must be a list")
    s1_items = [
        item
        for item in evidence_items
        if isinstance(item, dict) and item.get("kind") == "s1"
    ]
    if len(s1_items) != 1 or set(s1_items[0]) != {"kind", "path"}:
        raise ContractError("science.evidence must contain exactly one typed S1 path")
    if s1_items[0]["path"] == UNRESOLVED:
        raise ContractError("cannot synchronize generation guard before typed S1 exists")
    s1_path = _absolute_ref(str(s1_items[0]["path"]), base=draft_path.parent)
    s1_payload = _load_json(s1_path)
    _validate_search_stage_evidence(
        s1_payload,
        path=s1_path,
        expected_stage="s1",
        final_search=final_s1,
        final_evaluator={},
    )
    s1_record = _file_record(s1_path, kind="s1")

    provenance = draft.get("provenance")
    if not isinstance(provenance, dict) or "guard_config" not in provenance:
        raise ContractError("draft provenance has no guard_config")
    guard_path = _absolute_ref(
        str(provenance["guard_config"]), base=draft_path.parent
    ).resolve(strict=True)
    guard_payload = _load_json(guard_path)
    args = _guard_cli_flag_lint(guard_payload, path=guard_path)
    expected = args.get("expected_values")
    if not isinstance(expected, dict) or "--c-scale" not in expected:
        raise ContractError(f"{guard_path} has no expected --c-scale value")
    current_c_scale = expected["--c-scale"]

    guard_search = {
        "c_scale": selected_c_scale,
        "lazy_interior_chance": raw_search.get("lazy_interior_chance"),
    }
    evaluator = science.get("evaluator")
    generation = draft.get("generation")
    if not isinstance(evaluator, dict) or not isinstance(generation, dict):
        raise ContractError("draft evaluator/generation objects are missing")
    guard_evaluator = {"public_observation": evaluator.get("public_observation")}
    guard_generation = {
        "temperature_decisions": generation.get("temperature_decisions")
    }

    before_sha256 = _sha256(guard_path)
    if selected_c_scale == DEFAULT_GENERATION_C_SCALE:
        if current_c_scale != DEFAULT_GENERATION_C_SCALE:
            raise ContractError(
                "typed S1 retained c_scale=0.03 but the guard is not pristine; "
                "refusing to hide manual guard drift"
            )
        _validate_guard(
            guard_path,
            search=guard_search,
            evaluator=guard_evaluator,
            generation=guard_generation,
            s1_evidence=s1_record,
        )
        return {
            "status": "already_synchronized",
            "changed": False,
            "selected_c_scale": selected_c_scale,
            "guard": str(guard_path),
            "before_sha256": before_sha256,
            "after_sha256": before_sha256,
            "s1_evidence": {"path": str(s1_path), "sha256": s1_record["sha256"]},
        }

    if current_c_scale == selected_c_scale:
        _validate_guard(
            guard_path,
            search=guard_search,
            evaluator=guard_evaluator,
            generation=guard_generation,
            s1_evidence=s1_record,
        )
        return {
            "status": "already_synchronized",
            "changed": False,
            "selected_c_scale": selected_c_scale,
            "guard": str(guard_path),
            "before_sha256": before_sha256,
            "after_sha256": before_sha256,
            "s1_evidence": {"path": str(s1_path), "sha256": s1_record["sha256"]},
        }
    if current_c_scale != DEFAULT_GENERATION_C_SCALE or GUARD_SYNC_KEY in guard_payload:
        raise ContractError(
            f"refusing to overwrite unexplained guard c_scale={current_c_scale!r}; "
            "expected the pristine c_scale=0.03 guard"
        )

    expected["--c-scale"] = selected_c_scale
    guard_payload[GUARD_SYNC_KEY] = {
        "schema_version": GUARD_SYNC_SCHEMA,
        "selected_c_scale": selected_c_scale,
        "source_s1_evidence": {
            "path": str(s1_path.resolve(strict=True)),
            "sha256": s1_record["sha256"],
        },
        "previous_guard_sha256": before_sha256,
        "synchronizer": {
            "path": GUARD_SYNC_TOOL,
            "sha256": _sha256((REPO_ROOT / GUARD_SYNC_TOOL).resolve(strict=True)),
        },
    }
    # Validate the exact prospective payload before the atomic replacement.
    # A semantic error therefore leaves the original guard bytes untouched.
    _validate_guard_payload(
        guard_payload,
        path=guard_path,
        search=guard_search,
        evaluator=guard_evaluator,
        generation=guard_generation,
        s1_evidence=s1_record,
    )
    if _sha256(guard_path) != before_sha256:
        raise ContractError(
            f"guard {guard_path} changed concurrently during synchronization"
        )
    _atomic_replace_json(guard_path, guard_payload)
    _validate_guard(
        guard_path,
        search=guard_search,
        evaluator=guard_evaluator,
        generation=guard_generation,
        s1_evidence=s1_record,
    )
    return {
        "status": "synchronized",
        "changed": True,
        "selected_c_scale": selected_c_scale,
        "guard": str(guard_path),
        "before_sha256": before_sha256,
        "after_sha256": _sha256(guard_path),
        "s1_evidence": {"path": str(s1_path), "sha256": s1_record["sha256"]},
    }


def _validate_generation(generation: dict[str, Any]) -> None:
    _require_exact_keys(generation, _GENERATION_KEYS, where="generation")
    for key in ("eval_server",):
        if type(generation[key]) is not bool:
            raise ContractError(f"generation.{key} must be a JSON boolean")
    for key in (
        "vps_to_win",
        "obs_width",
        "max_decisions",
        "temperature_decisions",
        "workers_per_gpu",
        "shard_size",
    ):
        if type(generation[key]) is not int:
            raise ContractError(f"generation.{key} must be an integer")
    if (
        generation["late_temperature_decisions"] is not None
        and type(generation["late_temperature_decisions"]) is not int
    ):
        raise ContractError(
            "generation.late_temperature_decisions must be an integer or null"
        )
    for key in ("temperature_high", "temperature_low", "late_temperature"):
        if isinstance(generation[key], bool) or not isinstance(
            generation[key], (int, float)
        ):
            raise ContractError(f"generation.{key} must be numeric")
    if generation["track"] != "2p_no_trade" or int(generation["vps_to_win"]) != 10:
        raise ContractError("A1 supports only the locked 2p_no_trade, 10-VP regime")
    if int(generation["max_decisions"]) <= 0:
        raise ContractError("max_decisions must be positive")
    opening = int(generation["temperature_decisions"])
    late = generation["late_temperature_decisions"]
    if opening < 0 or opening > int(generation["max_decisions"]):
        raise ContractError("temperature_decisions is outside the game cap")
    if late is not None and not (
        opening <= int(late) <= int(generation["max_decisions"])
    ):
        raise ContractError(
            "late_temperature_decisions must be between opening and max_decisions"
        )
    if late is None and float(generation["late_temperature"]) != 0.0:
        raise ContractError(
            "disabled late-temperature window must use late_temperature=0"
        )
    if late is not None and float(generation["late_temperature"]) <= 0.0:
        raise ContractError(
            "enabled late-temperature window requires positive late_temperature"
        )
    if generation["format"] != "npz":
        raise ContractError("the A1 postflight scanner currently requires format=npz")
    if generation["device"] != "cuda" or bool(generation["eval_server"]):
        raise ContractError(
            "A1 renders per-GPU CUDA jobs with eval_server=false (MPS is host-managed)"
        )
    if int(generation["workers_per_gpu"]) <= 0 or int(generation["shard_size"]) <= 0:
        raise ContractError("workers_per_gpu and shard_size must be positive")


def _validate_post_wave(value: dict[str, Any]) -> None:
    required = {
        "require_complete_games",
        "selected_truncations_max",
        "invalid_teacher_actions_max",
        "require_public_observation",
        "require_unique_game_seeds",
        "require_no_val_only_overlap",
        "selection_before_row_expansion",
        "required_reports",
        "require_shard_sha256",
        "require_contract_attestation",
        "validation_holdout",
    }
    _require_exact_keys(value, required, where="post_wave_acceptance")
    if not all(
        bool(value[key])
        for key in (
            "require_complete_games",
            "require_public_observation",
            "require_unique_game_seeds",
            "require_no_val_only_overlap",
            "selection_before_row_expansion",
            "require_shard_sha256",
            "require_contract_attestation",
        )
    ):
        raise ContractError("all fail-closed A1 post-wave requirements must be true")
    if (
        int(value["selected_truncations_max"]) != 0
        or int(value["invalid_teacher_actions_max"]) != 0
    ):
        raise ContractError(
            "selected truncations and invalid teacher actions must both be zero"
        )
    reports = set(map(str, value["required_reports"]))
    if reports != REQUIRED_REPORTS:
        raise ContractError(
            f"required_reports must be exactly {sorted(REQUIRED_REPORTS)}, got {sorted(reports)}"
        )
    validation = dict(value["validation_holdout"])
    if validation != {
        "split_unit": "game_seed",
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
    }:
        raise ContractError(
            "validation_holdout must be the shared game-level 5%, seed=17, max_samples=0 contract"
        )


def _build_jobs(
    workers: list[dict[str, Any]],
    *,
    seed_base: int,
    block_size: int,
    per_worker: dict[str, int],
    output_root: str,
    contract_id: str,
) -> list[dict[str, Any]]:
    if len(workers) != EXPECTED_WORKER_COUNT:
        raise ContractError(
            "the pre-wave handoff requires exactly "
            f"{EXPECTED_WORKER_COUNT} workers, got {len(workers)}"
        )
    if per_worker != EXPECTED_PER_WORKER:
        raise ContractError(
            f"per_worker_games must be exactly {EXPECTED_PER_WORKER}, got {per_worker}"
        )
    worker_ids = [str(worker["id"]) for worker in workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise ContractError("fleet.workers contains duplicate ids")
    for worker in workers:
        if set(worker) != {"id", "host_alias", "gpu"}:
            raise ContractError(
                "each fleet worker must have exactly id, host_alias, gpu"
            )
        if int(worker["gpu"]) < 0:
            raise ContractError(f"worker {worker['id']} has a negative GPU index")
    placements = [(str(worker["host_alias"]), int(worker["gpu"])) for worker in workers]
    if len(set(placements)) != len(placements):
        raise ContractError(
            "fleet.workers assigns the same host/GPU placement more than once"
        )
    attempts_per_worker = sum(EXPECTED_ATTEMPTS_PER_WORKER.values())
    if block_size < attempts_per_worker:
        raise ContractError(
            f"seed block_size={block_size} is smaller than "
            f"{attempts_per_worker} attempts/worker"
        )
    jobs: list[dict[str, Any]] = []
    category_order = tuple(EXPECTED_GAMES)
    for worker_index, worker in enumerate(workers):
        cursor = seed_base + worker_index * block_size
        for category in category_order:
            games = int(per_worker[category])
            attempts = int(EXPECTED_ATTEMPTS_PER_WORKER[category])
            job_id = f"{worker['id']}__{category}"
            jobs.append(
                {
                    "job_id": job_id,
                    "worker_id": str(worker["id"]),
                    "host_alias": str(worker["host_alias"]),
                    "gpu": int(worker["gpu"]),
                    "category": category,
                    "base_seed": cursor,
                    "games": games,
                    "attempts": attempts,
                    "seed_end": cursor + attempts,
                    "output_dir": str(Path(output_root) / contract_id / job_id),
                    "claim_label": f"{contract_id}:{job_id}",
                }
            )
            cursor += attempts
    assert_disjoint_seed_blocks(
        [
            (job["job_id"], int(job["base_seed"]), int(job["attempts"]))
            for job in jobs
        ]
    )
    totals = Counter()
    for job in jobs:
        totals[job["category"]] += int(job["games"])
        interval = (int(job["base_seed"]), int(job["seed_end"]))
        if _ranges_overlap(interval, VAL_ONLY_SEED_RANGE):
            raise ContractError(
                f"job {job['job_id']} seed range {interval} overlaps VAL-ONLY {VAL_ONLY_SEED_RANGE}"
            )
    if dict(totals) != EXPECTED_GAMES:
        raise ContractError(
            f"job category totals are {dict(totals)}, expected {EXPECTED_GAMES}"
        )
    return jobs


def _read_strict_ledger(ledger: Path) -> tuple[str, list[tuple[int, int, str]]]:
    try:
        ledger_text = ledger.read_text(encoding="utf-8")
        ledger_lines = ledger_text.splitlines()
        claims = parse_seed_ledger(ledger)
    except Exception as error:  # noqa: BLE001 - malformed/unreadable ledger blocks sealing.
        raise ContractError(f"cannot parse seed ledger {ledger}: {error}") from error
    # The shared parser intentionally skips malformed rows so routine generator
    # guards do not crash every launch.  A one-time immutable handoff must be
    # stricter: any line that looks like a range row but was not parsed could
    # conceal a claim, so refuse instead of treating it as free space.
    range_like = [
        line
        for line in ledger_lines
        if re.match(r"^\s*\|?\s*\[.*\)\s*\|", line) is not None
    ]
    if len(range_like) != len(claims):
        raise ContractError(
            f"seed ledger {ledger} has {len(range_like)} range-like row(s) but only "
            f"{len(claims)} parsed claim(s); repair malformed ledger rows before sealing"
        )
    return ledger_text, claims


def _seed_ledger_snapshot(ledger: Path) -> dict[str, Any]:
    ledger_text, claims = _read_strict_ledger(ledger)
    if ledger_text and not ledger_text.endswith("\n"):
        raise ContractError(
            f"seed ledger {ledger} must end with a newline so later claims append safely"
        )
    record = _file_record(ledger, kind="seed_ledger_snapshot")
    record.update(
        {
            "snapshot_text": ledger_text,
            "snapshot_size_bytes": len(ledger_text.encode("utf-8")),
            "claims": [
                {"start": int(start), "end": int(end), "label": str(label)}
                for start, end, label in claims
            ],
            "claims_sha256": _digest_value(
                [
                    {"start": int(start), "end": int(end), "label": str(label)}
                    for start, end, label in claims
                ]
            ),
        }
    )
    return record


def _validate_against_ledger(jobs: list[dict[str, Any]], ledger: Path) -> None:
    _ledger_text, claims = _read_strict_ledger(ledger)
    for job in jobs:
        requested = (int(job["base_seed"]), int(job["seed_end"]))
        for start, end, label in claims:
            if _ranges_overlap(requested, (int(start), int(end))):
                raise ContractError(
                    f"job {job['job_id']} range {requested} overlaps ledger claim "
                    f"[{start}, {end}) {label!r}"
                )


def _ledger_claim_label(contract_sha256: str, job: dict[str, Any]) -> str:
    return (
        f"claim={job['claim_label']} contract={contract_sha256} "
        f"job={job['job_id']} |"
    )


def _verify_live_seed_ledger(
    snapshot: dict[str, Any],
    jobs: list[dict[str, Any]],
    *,
    contract_sha256: str,
    require_all_job_claims: bool,
) -> None:
    """Verify an immutable pre-claim prefix plus append-only live claims.

    The shared seed ledger is intentionally mutable after sealing.  Its sealed
    bytes must remain an exact prefix, while appended rows may contain unrelated
    disjoint work and this contract's own exact job claims.  Any overlapping peer
    claim, mislabeled range, or missing required post-wave claim fails closed.
    """

    expected_fields = {
        "kind",
        "path",
        "sha256",
        "snapshot_text",
        "snapshot_size_bytes",
        "claims",
        "claims_sha256",
    }
    if set(snapshot) != expected_fields:
        raise ContractError(
            "seed-ledger snapshot fields mismatch; "
            f"missing={sorted(expected_fields - set(snapshot))} "
            f"extra={sorted(set(snapshot) - expected_fields)}"
        )
    if snapshot.get("kind") != "seed_ledger_snapshot":
        raise ContractError("seed-ledger record is not an immutable snapshot")
    snapshot_text = snapshot.get("snapshot_text")
    if not isinstance(snapshot_text, str):
        raise ContractError("seed-ledger snapshot_text is not a string")
    snapshot_bytes = snapshot_text.encode("utf-8")
    if int(snapshot.get("snapshot_size_bytes", -1)) != len(snapshot_bytes):
        raise ContractError("seed-ledger snapshot byte count drift")
    if "sha256:" + hashlib.sha256(snapshot_bytes).hexdigest() != snapshot.get(
        "sha256"
    ):
        raise ContractError("seed-ledger snapshot hash drift inside contract")
    locked_claims = snapshot.get("claims")
    if not isinstance(locked_claims, list) or snapshot.get(
        "claims_sha256"
    ) != _digest_value(locked_claims):
        raise ContractError("seed-ledger snapshot claim digest drift")

    ledger = Path(str(snapshot["path"]))
    try:
        live_bytes = ledger.read_bytes()
    except OSError as error:
        raise ContractError(f"cannot read live seed ledger {ledger}: {error}") from error
    if not live_bytes.startswith(snapshot_bytes):
        raise ContractError(
            f"live seed ledger {ledger} is not an append-only extension of the sealed snapshot"
        )
    try:
        live_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"live seed ledger {ledger} is not UTF-8: {error}") from error
    _live_text, live_claims = _read_strict_ledger(ledger)
    normalized_locked = [
        (int(item["start"]), int(item["end"]), str(item["label"]))
        for item in locked_claims
        if isinstance(item, dict)
        and set(item) == {"start", "end", "label"}
    ]
    if len(normalized_locked) != len(locked_claims) or live_claims[
        : len(normalized_locked)
    ] != normalized_locked:
        raise ContractError("seed-ledger sealed claim prefix drift")

    own_matches: dict[str, int] = {str(job["job_id"]): 0 for job in jobs}
    for start, end, label in live_claims:
        claim = (int(start), int(end))
        for job in jobs:
            requested = (int(job["base_seed"]), int(job["seed_end"]))
            claim_token = f"claim={job['claim_label']}"
            expected_label = _ledger_claim_label(contract_sha256, job)
            names_us = claim_token in str(label)
            exact_own_claim = claim == requested and str(label) == expected_label
            if names_us and not exact_own_claim:
                raise ContractError(
                    "ledger claim does not exactly match its rendered contract/job row: "
                    f"label={label!r} range={claim}, expected_label={expected_label!r} "
                    f"expected_range={requested}"
                )
            if not _ranges_overlap(requested, claim):
                continue
            if not exact_own_claim:
                raise ContractError(
                    f"job {job['job_id']} range {requested} overlaps live ledger claim "
                    f"[{start}, {end}) {label!r}"
                )
            own_matches[str(job["job_id"])] += 1
    duplicates = sorted(job_id for job_id, count in own_matches.items() if count > 1)
    if duplicates:
        raise ContractError(
            "live ledger repeats exact own claim(s); resume without appending a second row: "
            f"{duplicates[:8]}"
        )
    if require_all_job_claims:
        missing = sorted(job_id for job_id, count in own_matches.items() if count == 0)
        if missing:
            raise ContractError(
                "post-wave ledger is missing exact own claim(s) for "
                f"{len(missing)} job(s): {missing[:8]}"
            )


def _checkpoint_records(
    raw: list[dict[str, Any]], *, base: Path, value_readout: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    sha_to_id: dict[str, str] = {}
    for entry in raw:
        role = str(entry.get("role", ""))
        expected_fields = {"id", "role", "path"}
        if role == "producer":
            expected_fields.add("legacy_scalar_readout_attestation")
        if set(entry) != expected_fields:
            raise ContractError(
                f"checkpoint role={role!r} fields mismatch; expected {sorted(expected_fields)}"
            )
        artifact_id = str(entry["id"])
        if artifact_id in ids:
            raise ContractError(f"duplicate checkpoint id {artifact_id!r}")
        ids.add(artifact_id)
        path = _absolute_ref(str(entry["path"]), base=base)
        record = _file_record(path, kind="checkpoint", artifact_id=artifact_id)
        record["role"] = role
        record["md5"] = _md5(path)
        raw_attestation = entry.get("legacy_scalar_readout_attestation")
        attestation_path = (
            _absolute_ref(str(raw_attestation), base=base)
            if raw_attestation is not None
            else None
        )
        # Every neural opponent is constructed with the same evaluator config
        # and readout as the producer.  A categorical producer therefore makes
        # positive HL-Gauss provenance mandatory for history/hard-negative
        # checkpoints too; otherwise those jobs would fail only after launch.
        record["metadata"] = _checkpoint_metadata(
            path,
            checkpoint_sha256=str(record["sha256"]),
            value_readout=value_readout,
            require_trained_readout=(
                role == "producer" or value_readout == "categorical"
            ),
            legacy_scalar_attestation=attestation_path,
        )
        previous = sha_to_id.get(record["sha256"])
        if previous is not None:
            raise ContractError(
                f"checkpoint ids {previous!r} and {artifact_id!r} contain identical bytes"
            )
        sha_to_id[record["sha256"]] = artifact_id
        records.append(record)
    producers = [record for record in records if record["role"] == "producer"]
    if len(producers) != 1:
        raise ContractError("checkpoints must contain exactly one role=producer")
    return records


def _verify_declared_source_artifacts(
    raw: Any, *, envelope_path: Path
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ContractError(
            f"evidence {envelope_path} must bind at least one source artifact"
        )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ContractError(
                f"evidence {envelope_path} source_artifacts[{index}] must have path and sha256"
            )
        path = _absolute_ref(str(item["path"]), base=envelope_path.parent)
        actual = _sha256(path) if path.is_file() else "<missing>"
        if actual != item["sha256"]:
            raise ContractError(
                f"evidence {envelope_path} source artifact drift at {path}: "
                f"declared {item['sha256']}, actual {actual}"
            )
        records.append({"path": str(path), "sha256": actual})
    return records


def _validate_learner_objective(value: dict[str, Any]) -> None:
    required = {
        "objective",
        "value_readout",
        "value_categorical_bins",
        "hlgauss_sigma_ratio",
    }
    _require_exact_keys(value, required, where="science.learner_value_objective")
    objective = str(value["objective"])
    readout = str(value["value_readout"])
    if objective == "hlgauss":
        if readout != "categorical" or value["value_categorical_bins"] != 33:
            raise ContractError(
                "HL-Gauss learner objective requires categorical readout and 33 bins"
            )
        if float(value["hlgauss_sigma_ratio"]) != 0.75:
            raise ContractError(
                "HL-Gauss learner objective requires the locked sigma ratio 0.75"
            )
    elif objective == "mse":
        if readout != "scalar" or value["value_categorical_bins"] is not None:
            raise ContractError(
                "MSE learner objective requires scalar readout and null categorical bins"
            )
        if value["hlgauss_sigma_ratio"] is not None:
            raise ContractError(
                "MSE learner objective requires null HL-Gauss sigma ratio"
            )
    else:
        raise ContractError("learner objective must be 'hlgauss' or 'mse'")


def _validate_learner_training_recipe(value: dict[str, Any]) -> None:
    """Require the exact effective one-dose recipe, including JSON types."""

    _require_exact_keys(
        value,
        set(EXPECTED_LEARNER_TRAINING_RECIPE),
        where="science.learner_training_recipe",
    )
    for key, expected in EXPECTED_LEARNER_TRAINING_RECIPE.items():
        actual = value[key]
        if type(actual) is not type(expected):
            raise ContractError(
                f"science.learner_training_recipe.{key} must have JSON type "
                f"{type(expected).__name__}, got {type(actual).__name__}"
            )
        if actual != expected:
            raise ContractError(
                f"science.learner_training_recipe.{key} must equal the locked "
                f"pre-wave value {expected!r}, got {actual!r}"
            )
    expected_global_batch = (
        int(value["batch_size"])
        * int(value["grad_accum_steps"])
        * int(value["world_size"])
    )
    if int(value["global_batch_size"]) != expected_global_batch:
        raise ContractError(
            "science.learner_training_recipe.global_batch_size does not equal "
            "batch_size * grad_accum_steps * world_size"
        )


def _validate_a0_evidence(
    payload: dict[str, Any], *, path: Path, learner_objective: dict[str, Any]
) -> dict[str, Any]:
    # A typed-looking JSON object is not evidence.  Replay the exact A0
    # adjudicator in-process from its sealed lock/result (and, on the adoption
    # path, its calibration/policy inputs) and require byte-semantic equality.
    sealed_for_replay = dict(payload.get("sealed_inputs", {}))
    lock_for_replay = _absolute_ref(
        str(sealed_for_replay.get("lock", "")), base=path.parent
    )
    result_for_replay = _absolute_ref(
        str(sealed_for_replay.get("training_result", "")), base=path.parent
    )
    try:
        a0_lock = _load_json(lock_for_replay)
        repo_root = Path(
            str(
                a0_lock.get("repo_root_at_seal")
                or a0_lock.get("artifact_root_at_seal")
                or REPO_ROOT
            )
        ).expanduser().absolute()
        calibration = payload.get("calibration_artifacts")
        scalar_calibration: Path | None = None
        hl_calibration: Path | None = None
        if isinstance(calibration, dict):
            scalar_raw = dict(calibration.get("scalar") or {}).get("calibration")
            hl_raw = dict(calibration.get("hlgauss33") or {}).get("calibration")
            if scalar_raw is not None:
                scalar_calibration = _absolute_ref(
                    str(scalar_raw), base=path.parent
                )
            if hl_raw is not None:
                hl_calibration = _absolute_ref(str(hl_raw), base=path.parent)
        policy = payload.get("policy_drift")
        policy_path = (
            _absolute_ref(str(policy["artifact"]), base=path.parent)
            if isinstance(policy, dict) and policy.get("artifact")
            else None
        )
        replayed = a0_binding.build_binding_verdict(
            lock_path=lock_for_replay,
            result_path=result_for_replay,
            scalar_calibration_path=scalar_calibration,
            hl_calibration_path=hl_calibration,
            policy_drift_path=policy_path,
            repo_root=repo_root,
        )
    except Exception as error:  # noqa: BLE001 - any replay failure blocks sealing.
        raise ContractError(f"A0 evidence {path} failed semantic replay: {error}") from error
    if payload != replayed:
        raise ContractError(
            f"A0 evidence {path} does not equal the replayed binding verdict"
        )

    if payload.get("schema_version") != A0_EVIDENCE_SCHEMA:
        raise ContractError(
            f"A0 evidence {path} schema must be {A0_EVIDENCE_SCHEMA!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("a0_interpretable") is not True:
        raise ContractError(f"A0 evidence {path} has a0_interpretable != true")
    if payload.get("a0_stage_complete") is not True:
        raise ContractError(f"A0 evidence {path} has a0_stage_complete != true")
    if payload.get("a0_binding_pass") is not True:
        raise ContractError(f"A0 evidence {path} has a0_binding_pass != true")
    if type(payload.get("hlgauss_adoption_pass")) is not bool:
        raise ContractError(
            f"A0 evidence {path} must carry boolean hlgauss_adoption_pass"
        )
    gates = dict(payload.get("gates", {}))
    required_gates = {
        "scalar_reproduction",
        "hl_training_stability",
        "exact_validation_seeds",
        "categorical_readout_provenance",
        "calibration",
        "policy_drift",
    }
    if set(gates) != required_gates or any(
        value is not None and type(value) is not bool for value in gates.values()
    ):
        raise ContractError(f"A0 evidence {path} has malformed binding gates")
    if gates["scalar_reproduction"] is not True:
        raise ContractError(f"A0 evidence {path} failed scalar reproduction")
    # A failed HL calibration/policy/readout gate is a legitimate, informative
    # retain-scalar decision.  The binding-verdict producer owns the stronger
    # cross-artifact checks; this consumer requires its explicit stage-complete
    # and interpretable decision plus immutable source bytes below.
    decision = dict(payload.get("decision", {}))
    required_decision = {
        "status",
        "learner_objective",
        "learner_value_readout",
        "mechanism_checkpoint_sha256",
        "mechanism_checkpoint_is_production_candidate",
    }
    _require_exact_keys(
        decision, required_decision, where=f"A0 evidence {path} decision"
    )
    expected_statuses = (
        {"adopt_hlgauss_for_a1", "adopt_hlgauss"}
        if learner_objective["objective"] == "hlgauss"
        else {"retain_scalar_for_a1", "retain_scalar"}
    )
    if decision["status"] not in expected_statuses:
        raise ContractError(
            f"A0 evidence selected {decision['status']!r}, contract learner objective requires "
            f"one of {sorted(expected_statuses)!r}"
        )
    if (
        decision["learner_objective"] != learner_objective["objective"]
        or decision["learner_value_readout"] != learner_objective["value_readout"]
    ):
        raise ContractError("A0 learner objective/readout does not match the contract")
    if decision["mechanism_checkpoint_is_production_candidate"] is not False:
        raise ContractError(
            "A0 mechanism checkpoint must remain evidence-only, not production"
        )
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(decision["mechanism_checkpoint_sha256"])
    ):
        raise ContractError("A0 mechanism checkpoint SHA-256 is malformed")
    should_adopt_hl = learner_objective["objective"] == "hlgauss"
    if payload["hlgauss_adoption_pass"] is not should_adopt_hl:
        raise ContractError(
            "A0 HL-adoption verdict does not match the selected learner objective"
        )
    records: list[dict[str, Any]] = []
    sealed = dict(payload.get("sealed_inputs", {}))
    for path_key, sha_key in (
        ("lock", "lock_sha256"),
        ("training_result", "training_result_sha256"),
    ):
        source = _absolute_ref(str(sealed.get(path_key, "")), base=path.parent)
        actual = _sha256(source) if source.is_file() else "<missing>"
        declared = str(sealed.get(sha_key, ""))
        if actual.removeprefix("sha256:") != declared.removeprefix("sha256:"):
            raise ContractError(f"A0 evidence {path} {path_key} artifact hash drift")
        records.append({"path": str(source), "sha256": actual})
    for arm, raw in dict(payload.get("calibration_artifacts") or {}).items():
        artifact = dict(raw)
        for path_key, sha_key in (
            ("calibration", "calibration_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("report", "report_sha256"),
            ("manifest", "manifest_sha256"),
        ):
            if path_key not in artifact or sha_key not in artifact:
                continue
            source = _absolute_ref(str(artifact[path_key]), base=path.parent)
            actual = _sha256(source) if source.is_file() else "<missing>"
            if actual.removeprefix("sha256:") != str(artifact[sha_key]).removeprefix(
                "sha256:"
            ):
                raise ContractError(
                    f"A0 evidence {path} {arm}.{path_key} artifact hash drift"
                )
            records.append({"path": str(source), "sha256": actual})
    policy = dict(payload.get("policy_drift") or {})
    if "artifact" in policy and "artifact_sha256" in policy:
        source = _absolute_ref(str(policy["artifact"]), base=path.parent)
        actual = _sha256(source) if source.is_file() else "<missing>"
        if actual.removeprefix("sha256:") != str(
            policy["artifact_sha256"]
        ).removeprefix("sha256:"):
            raise ContractError(f"A0 evidence {path} policy-drift artifact hash drift")
        records.append({"path": str(source), "sha256": actual})
    if not records:
        raise ContractError(f"A0 evidence {path} binds no source artifacts")
    # Bind the adjudicator implementation as well as its inputs.  The A0
    # verdict schema predates the generic stage envelope's explicit
    # ``adjudicator`` field, so the consumer supplies this known path.
    records.append(
        _file_record(
            REPO_ROOT / "tools" / "a0_binding_verdict.py",
            kind="a0_adjudicator",
        )
    )
    return {"decision": decision, "source_artifacts": records}


def _validate_operator_binding_reference(
    raw: Any, *, owner_path: Path, where: str
) -> tuple[Path, dict[str, str]]:
    _require_exact_keys(raw, {"path", "sha256"}, where=where)
    path = _absolute_ref(str(raw["path"]), base=owner_path.parent)
    actual = _sha256(path) if path.is_file() else "<missing>"
    if actual != raw["sha256"]:
        raise ContractError(
            f"{where} hash drift at {path}: declared {raw['sha256']}, actual {actual}"
        )
    return path, {"path": str(path), "sha256": actual}


def _validate_operator_binding_evidence(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_stage: str,
    final_search: dict[str, Any],
    final_evaluator: dict[str, Any],
) -> dict[str, Any]:
    """Validate the narrow no-S2/no-S3 operator-choice bridge.

    This path cannot accept experimental verdicts and the ordinary
    search-adjudication path cannot accept these bindings.  The exact constants
    below make the exception specific to the current n128/no-adaptive operator
    directive rather than a general mechanism for bypassing evidence.
    """

    if expected_stage not in {"s2", "s3"}:
        raise ContractError(
            f"{expected_stage.upper()} cannot use the operator-binding schema"
        )
    common_keys = {
        "schema_version",
        "artifact_kind",
        "stage",
        "operator",
        "passed",
        "decision",
        "reason",
        "binding_time_utc",
        "statement",
        "selected_fields",
        "selected_fields_sha256",
        "source_s1",
        "source_s1_selected_fields_sha256",
        "emitter",
        "artifact_content_sha256",
    }
    expected_keys = set(common_keys)
    if expected_stage == "s3":
        expected_keys.add("source_s2_binding")
    _require_exact_keys(
        payload,
        expected_keys,
        where=f"{expected_stage.upper()} operator binding {path}",
    )
    if payload["schema_version"] != operator_binding.SCHEMA:
        raise ContractError(
            f"{expected_stage.upper()} operator-binding schema mismatch"
        )
    if payload["artifact_kind"] != operator_binding.ARTIFACT_KIND:
        raise ContractError(
            f"{expected_stage.upper()} artifact_kind must explicitly deny strength evidence"
        )
    if payload["stage"] != expected_stage or payload["passed"] is not True:
        raise ContractError(
            f"{expected_stage.upper()} operator binding has wrong stage/passed state"
        )
    expected_decision = "operator_bind" if expected_stage == "s2" else "operator_hold"
    expected_operator = (
        operator_binding.S2_OPERATOR
        if expected_stage == "s2"
        else operator_binding.S3_OPERATOR
    )
    expected_reason = (
        operator_binding.S2_REASON
        if expected_stage == "s2"
        else operator_binding.S3_REASON
    )
    expected_selected = (
        operator_binding.S2_SELECTED
        if expected_stage == "s2"
        else operator_binding.S3_SELECTED
    )
    if payload["decision"] != expected_decision:
        raise ContractError(
            f"{expected_stage.upper()} operator-binding decision mismatch"
        )
    if payload["operator"] != expected_operator:
        raise ContractError(
            f"{expected_stage.upper()} operator-binding operator mismatch"
        )
    if payload["reason"] != expected_reason:
        raise ContractError(f"{expected_stage.upper()} operator-binding reason mismatch")
    if payload["statement"] != operator_binding.STATEMENT:
        raise ContractError(
            f"{expected_stage.upper()} must state that the binding is not strength evidence"
        )
    timestamp = payload["binding_time_utc"]
    if not isinstance(timestamp, str):
        raise ContractError(f"{expected_stage.upper()} binding_time_utc must be a string")
    try:
        parsed_time = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(
            f"{expected_stage.upper()} binding_time_utc is not ISO-8601"
        ) from error
    if parsed_time.tzinfo is None or parsed_time.utcoffset() != dt.timedelta(0):
        raise ContractError(
            f"{expected_stage.upper()} binding_time_utc must carry an explicit UTC offset"
        )
    selected = payload["selected_fields"]
    if selected != expected_selected:
        raise ContractError(
            f"{expected_stage.upper()} operator binding must select exactly {expected_selected}"
        )
    if payload["selected_fields_sha256"] != _digest_value(selected):
        raise ContractError(
            f"{expected_stage.upper()} operator-binding selected_fields_sha256 mismatch"
        )
    mismatches = {
        key: (selected[key], final_search[key])
        for key in selected
        if selected[key] != final_search[key]
    }
    if mismatches:
        raise ContractError(
            f"{expected_stage.upper()} selected search fields mismatch final contract: {mismatches}"
        )
    unhashed = dict(payload)
    declared_content_digest = unhashed.pop("artifact_content_sha256")
    if declared_content_digest != _digest_value(unhashed):
        raise ContractError(
            f"{expected_stage.upper()} operator-binding self digest mismatch"
        )

    emitter_path, emitter_record = _validate_operator_binding_reference(
        payload["emitter"],
        owner_path=path,
        where=f"{expected_stage.upper()} operator-binding emitter",
    )
    if not emitter_path.as_posix().endswith("tools/search_operator_binding.py"):
        raise ContractError(
            f"{expected_stage.upper()} operator binding has an untrusted emitter"
        )
    s1_path, s1_record = _validate_operator_binding_reference(
        payload["source_s1"],
        owner_path=path,
        where=f"{expected_stage.upper()} source S1",
    )
    s1_payload = _load_json(s1_path)
    s1_semantic = _validate_search_stage_evidence(
        s1_payload,
        path=s1_path,
        expected_stage="s1",
        final_search=final_search,
        final_evaluator=final_evaluator,
    )
    if (
        payload["source_s1_selected_fields_sha256"]
        != s1_semantic["selected_fields_sha256"]
    ):
        raise ContractError(
            f"{expected_stage.upper()} source S1 selected-fields digest mismatch"
        )

    if expected_stage == "s2":
        try:
            replayed_s2, _ = operator_binding.build_bindings(
                s1_path,
                s2_output_path=path,
                binding_time_utc=timestamp,
            )
        except operator_binding.BindingError as error:
            raise ContractError(f"S2 operator-binding replay failed: {error}") from error
        if payload != replayed_s2:
            raise ContractError("S2 operator binding does not equal semantic replay")

    records = [s1_record, emitter_record]
    semantic: dict[str, Any] = {
        "decision": payload["decision"],
        "evidence_class": operator_binding.ARTIFACT_KIND,
        "selected_fields": dict(selected),
        "selected_fields_sha256": payload["selected_fields_sha256"],
        "checkpoint": dict(s1_semantic["checkpoint"]),
        "source_artifacts": records,
        "artifact_content_sha256": declared_content_digest,
        "binding_time_utc": timestamp,
    }
    if expected_stage == "s3":
        s2_path, s2_record = _validate_operator_binding_reference(
            payload["source_s2_binding"],
            owner_path=path,
            where="S3 source S2 operator binding",
        )
        s2_payload = _load_json(s2_path)
        s2_semantic = _validate_operator_binding_evidence(
            s2_payload,
            path=s2_path,
            expected_stage="s2",
            final_search=final_search,
            final_evaluator=final_evaluator,
        )
        if s2_payload["source_s1"] != payload["source_s1"]:
            raise ContractError("S3 and S2 operator bindings do not share exact S1 lineage")
        if s2_payload["binding_time_utc"] != timestamp:
            raise ContractError("S3 and S2 operator bindings must share one binding time")
        if s2_semantic["evidence_class"] != operator_binding.ARTIFACT_KIND:
            raise ContractError("S3 predecessor is not an operator binding")
        try:
            replayed_s2, replayed_s3 = operator_binding.build_bindings(
                s1_path,
                s2_output_path=s2_path,
                binding_time_utc=timestamp,
            )
        except operator_binding.BindingError as error:
            raise ContractError(f"S3 operator-binding replay failed: {error}") from error
        if s2_payload != replayed_s2 or payload != replayed_s3:
            raise ContractError("S3 operator binding does not equal semantic replay")
        records.append(s2_record)
    return semantic
def _validate_search_stage_evidence(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_stage: str,
    final_search: dict[str, Any],
    final_evaluator: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") == operator_binding.SCHEMA:
        return _validate_operator_binding_evidence(
            payload,
            path=path,
            expected_stage=expected_stage,
            final_search=final_search,
            final_evaluator=final_evaluator,
        )
    if payload.get("schema_version") != SEARCH_STAGE_EVIDENCE_SCHEMA:
        raise ContractError(
            f"{expected_stage.upper()} evidence {path} schema must be "
            f"{SEARCH_STAGE_EVIDENCE_SCHEMA!r}"
        )
    if payload.get("stage") != expected_stage:
        raise ContractError(
            f"evidence {path} stage={payload.get('stage')!r}, expected {expected_stage!r}"
        )
    if payload.get("passed") is not True:
        raise ContractError(f"{expected_stage.upper()} evidence {path} passed != true")
    if payload.get("decision") not in {"adopt", "hold"}:
        raise ContractError(
            f"{expected_stage.upper()} evidence must decide adopt or hold"
        )
    source_records = _verify_declared_source_artifacts(
        payload.get("source_artifacts", []), envelope_path=path
    )
    manifest_candidates: list[Path] = []
    for record in source_records:
        candidate = Path(record["path"])
        if candidate.suffix.lower() != ".json":
            continue
        try:
            candidate_payload = _load_json(candidate)
        except ContractError:
            continue
        if (
            candidate_payload.get("schema_version")
            == search_adjudicator.MANIFEST_SCHEMA
            and candidate_payload.get("stage") == expected_stage
        ):
            manifest_candidates.append(candidate)
    if len(manifest_candidates) != 1:
        raise ContractError(
            f"{expected_stage.upper()} evidence must bind exactly one replayable "
            f"adjudication manifest, found {len(manifest_candidates)}"
        )
    try:
        replayed = search_adjudicator.adjudicate(manifest_candidates[0])
    except search_adjudicator.AdjudicationError as error:
        raise ContractError(
            f"{expected_stage.upper()} evidence failed semantic replay: {error}"
        ) from error
    if payload != replayed:
        raise ContractError(
            f"{expected_stage.upper()} evidence does not equal the replayed adjudication"
        )
    selected = dict(payload.get("selected_fields", {}))
    expected_keys = {
        "s1": {
            "c_scale",
            "symmetry_averaged_eval",
            "symmetry_averaged_eval_threshold",
            "rescale_noise_floor_c",
            "sigma_eval",
        },
        "s2": {"n_full", "n_fast", "p_full"},
        "s3": {"n_full_wide", "n_full_wide_threshold", "wide_roots_always_full"},
    }[expected_stage]
    if set(selected) != expected_keys:
        raise ContractError(
            f"{expected_stage.upper()} selected_fields must be exactly {sorted(expected_keys)}"
        )
    mismatches = {
        key: (selected[key], final_search[key])
        for key in expected_keys
        if selected[key] != final_search[key]
    }
    if mismatches:
        raise ContractError(
            f"{expected_stage.upper()} selected search fields mismatch final contract: {mismatches}"
        )
    declared_selected_hash = payload.get("selected_fields_sha256")
    if declared_selected_hash != _digest_value(selected):
        raise ContractError(f"{expected_stage.upper()} selected_fields_sha256 mismatch")
    records = source_records
    adjudicator = payload.get("adjudicator")
    adjudicator_records = _verify_declared_source_artifacts(
        [adjudicator] if isinstance(adjudicator, dict) else [], envelope_path=path
    )
    if (
        not Path(adjudicator_records[0]["path"])
        .as_posix()
        .endswith("tools/search_teacher_adjudicator.py")
    ):
        raise ContractError(
            f"{expected_stage.upper()} evidence was not emitted by search_teacher_adjudicator.py"
        )
    records.extend(adjudicator_records)
    manifest_payload = _load_json(manifest_candidates[0])
    manifest_checkpoint = dict(manifest_payload.get("checkpoint", {}))
    _require_exact_keys(
        manifest_checkpoint,
        {"path", "sha256"},
        where=f"{expected_stage.upper()} manifest checkpoint",
    )
    manifest_checkpoint_path = _absolute_ref(
        str(manifest_checkpoint["path"]), base=manifest_candidates[0].parent
    )
    actual_checkpoint_sha = (
        _sha256(manifest_checkpoint_path)
        if manifest_checkpoint_path.is_file()
        else "<missing>"
    )
    if actual_checkpoint_sha != manifest_checkpoint["sha256"]:
        raise ContractError(
            f"{expected_stage.upper()} manifest checkpoint hash drift"
        )
    semantic: dict[str, Any] = {
        "decision": payload["decision"],
        "selected_fields": selected,
        "selected_fields_sha256": declared_selected_hash,
        "manifest": next(
            record
            for record in source_records
            if Path(record["path"]) == manifest_candidates[0]
        ),
        "checkpoint": {
            "path": str(manifest_checkpoint_path),
            "sha256": actual_checkpoint_sha,
        },
        "source_artifacts": records,
    }
    if expected_stage == "s3":
        if payload.get("final_search_operator_sha256") != _digest_value(final_search):
            raise ContractError(
                "S3 final search-operator hash does not match the contract"
            )
        if payload.get("teacher_evaluator_sha256") != _digest_value(final_evaluator):
            raise ContractError("S3 teacher-evaluator hash does not match the contract")
        if _digest_value(payload.get("final_search_operator")) != _digest_value(
            final_search
        ):
            raise ContractError(
                "S3 final search-operator fields do not match the contract"
            )
        if _digest_value(payload.get("teacher_evaluator")) != _digest_value(
            final_evaluator
        ):
            raise ContractError("S3 teacher-evaluator fields do not match the contract")
        semantic["final_search_operator_sha256"] = payload[
            "final_search_operator_sha256"
        ]
        semantic["teacher_evaluator_sha256"] = payload["teacher_evaluator_sha256"]
    return semantic


def _validate_categories(
    categories: list[dict[str, Any]], checkpoint_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in checkpoint_records}
    if {str(category.get("name")) for category in categories} != set(EXPECTED_GAMES):
        raise ContractError(
            f"source categories must be exactly {sorted(EXPECTED_GAMES)}"
        )
    normalized: list[dict[str, Any]] = []
    for category in categories:
        if set(category) != {"name", "mode", "checkpoint_ids"}:
            raise ContractError(
                "each source category must have exactly name, mode, checkpoint_ids"
            )
        name = str(category["name"])
        mode = str(category["mode"])
        ids = list(map(str, category["checkpoint_ids"]))
        if name == "current_producer":
            if mode != "self" or ids:
                raise ContractError(
                    "current_producer must use mode=self and no opponent checkpoint"
                )
        else:
            if mode != "checkpoint_list" or not ids:
                raise ContractError(f"{name} must use a non-empty checkpoint_list")
            for artifact_id in ids:
                if artifact_id not in by_id:
                    raise ContractError(
                        f"{name} references unknown checkpoint id {artifact_id!r}"
                    )
                if by_id[artifact_id]["role"] == "producer":
                    raise ContractError(
                        f"{name} may not disguise the producer as an opponent"
                    )
                required_role = (
                    "history" if name == "recent_history" else "hard_negative"
                )
                if by_id[artifact_id]["role"] != required_role:
                    raise ContractError(
                        f"{name} checkpoint {artifact_id!r} has role={by_id[artifact_id]['role']!r}; "
                        f"expected {required_role!r}"
                    )
        normalized.append({"name": name, "mode": mode, "checkpoint_ids": ids})
    return sorted(normalized, key=lambda item: list(EXPECTED_GAMES).index(item["name"]))


def _verify_artifact_records(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        path = Path(record["path"])
        actual = _sha256(path) if path.is_file() else "<missing>"
        if actual != record["sha256"]:
            raise ContractError(
                f"artifact drift for {record.get('id', record.get('kind'))}: "
                f"expected {record['sha256']}, got {actual} at {path}"
            )


def _verify_checkpoint_provenance_records(
    records: Iterable[dict[str, Any]], *, value_readout: str
) -> None:
    """Reconstruct checkpoint provenance, including legacy sidecar sources."""

    for record in records:
        metadata = dict(record.get("metadata", {}))
        attestation_record = metadata.get("legacy_scalar_readout_attestation")
        attestation_path: Path | None = None
        if attestation_record is not None:
            if not isinstance(attestation_record, dict):
                raise ContractError(
                    "legacy scalar-readout attestation lock record is malformed"
                )
            _verify_artifact_records([attestation_record])
            attestation_path = Path(str(attestation_record["path"]))
        reconstructed = _checkpoint_metadata(
            Path(str(record["path"])),
            checkpoint_sha256=str(record["sha256"]),
            value_readout=value_readout,
            require_trained_readout=(
                str(record.get("role")) == "producer" or value_readout == "categorical"
            ),
            legacy_scalar_attestation=attestation_path,
        )
        if metadata != reconstructed:
            raise ContractError(
                f"checkpoint provenance drift for {record.get('id', record['path'])}"
            )


def build_lock(
    draft_path: Path,
    *,
    seed_ledger_snapshot: dict[str, Any] | None = None,
    seed_ledger_contract_sha256: str | None = None,
) -> dict[str, Any]:
    draft_path = draft_path.absolute()
    draft = _load_json(draft_path)
    if draft.get("schema_version") != DRAFT_SCHEMA:
        raise ContractError(f"draft schema must be {DRAFT_SCHEMA!r}")
    _assert_no_unresolved(draft)
    required_top = {
        "schema_version",
        "contract_id",
        "science",
        "generation",
        "checkpoints",
        "source_categories",
        "fleet",
        "provenance",
        "post_wave_acceptance",
    }
    _require_exact_keys(draft, required_top, where="draft")
    contract_id = str(draft["contract_id"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", contract_id):
        raise ContractError("contract_id must be a stable lowercase identifier")

    base = draft_path.parent
    science = dict(draft["science"])
    if set(science) != {
        "search",
        "evaluator",
        "learner_value_objective",
        "learner_training_recipe",
        "evidence",
    }:
        raise ContractError(
            "science must have exactly search, evaluator, learner_value_objective, "
            "learner_training_recipe, evidence"
        )
    raw_search = dict(science["search"])
    search = _search_operator(raw_search)
    effective_search = _effective_search(raw_search)
    evaluator = _effective_evaluator(dict(science["evaluator"]))
    learner_objective = dict(science["learner_value_objective"])
    _validate_learner_objective(learner_objective)
    learner_training_recipe = dict(science["learner_training_recipe"])
    _validate_learner_training_recipe(learner_training_recipe)
    if evaluator["public_observation"] is not True:
        raise ContractError("science.evaluator.public_observation must be true")
    if evaluator["value_readout"] not in {"scalar", "categorical"}:
        raise ContractError("value_readout must be scalar or categorical")
    if evaluator["value_squash"] != "tanh":
        raise ContractError(
            "A1 generator currently implements only value_squash=tanh"
        )
    if float(evaluator["context_fill"]) != 0.0:
        raise ContractError(
            "A1 generator does not expose context_fill; it must remain 0"
        )
    if int(evaluator["cache_size"]) < 0:
        raise ContractError("evaluator cache_size must be non-negative")
    if evaluator["prior_temperature"] != search["prior_temperature"]:
        raise ContractError("search/evaluator prior_temperature values must match")
    if bool(evaluator["emit_uncertainty"]):
        raise ContractError(
            "A1 isolation does not enable the unselected uncertainty path"
        )
    if search["n_full"] < search["n_fast"] or not 0.0 <= float(search["p_full"]) <= 1.0:
        raise ContractError("invalid n_fast/n_full/p_full search budget")
    if int(search["n_fast"]) != 16 or int(search["n_full"]) not in {64, 128}:
        raise ContractError(
            "A1 permits n_fast=16 and global n_full in {64,128}; never global n256"
        )
    if search["wide_roots_always_full"] and search["n_full_wide"] is None:
        raise ContractError("wide_roots_always_full requires n_full_wide")
    if search["n_full_wide"] is None:
        if (
            search["n_full_wide_threshold"] is not None
            or search["wide_roots_always_full"]
        ):
            raise ContractError(
                "disabled adaptive n256 must use null threshold and always_full=false"
            )
    elif (
        search["n_full_wide_threshold"] is None or not search["wide_roots_always_full"]
    ):
        raise ContractError(
            "selected adaptive n256 requires an explicit threshold and always_full=true"
        )
    elif (
        int(search["n_full_wide"]) != 256 or int(search["n_full_wide_threshold"]) != 40
    ):
        raise ContractError(
            "the only permitted n256 arm is adaptive n_full_wide=256 at >=40"
        )
    if (
        search["symmetry_averaged_eval"]
        and search["symmetry_averaged_eval_threshold"] is None
    ):
        raise ContractError("selected D6 requires an explicit independent threshold")
    if (
        not search["symmetry_averaged_eval"]
        and search["symmetry_averaged_eval_threshold"] is not None
    ):
        raise ContractError("disabled D6 must use a null D6 threshold")
    if (
        search["symmetry_averaged_eval"]
        and int(search["symmetry_averaged_eval_threshold"]) < 20
    ):
        raise ContractError("selected D6 threshold must remain independent and >=20")
    if not search["exact_budget_sh"] and int(search["exact_budget_sh_min_n"]) != 0:
        raise ContractError("disabled exact-budget SH must use exact_budget_sh_min_n=0")
    if (
        float(search["rescale_noise_floor_c"]) > 0.0
        and float(search["sigma_eval"]) <= 0.0
    ):
        raise ContractError("selected D1 requires positive sigma_eval")

    evidence_raw = list(science["evidence"])
    evidence_kinds = {str(item.get("kind")) for item in evidence_raw}
    if evidence_kinds != REQUIRED_EVIDENCE or len(evidence_raw) != len(
        REQUIRED_EVIDENCE
    ):
        raise ContractError(
            f"science.evidence must contain exactly {sorted(REQUIRED_EVIDENCE)}"
        )
    evidence = []
    for item in sorted(evidence_raw, key=lambda item: str(item["kind"])):
        if set(item) != {"kind", "path"}:
            continue
        evidence_path = _absolute_ref(str(item["path"]), base=base)
        evidence_payload = _load_json(evidence_path)
        record = _file_record(evidence_path, kind=str(item["kind"]))
        record["document_schema"] = evidence_payload.get("schema_version")
        record["document_digest"] = _digest_value(evidence_payload)
        if item["kind"] == "a0":
            record["semantic_decision"] = _validate_a0_evidence(
                evidence_payload,
                path=evidence_path,
                learner_objective=learner_objective,
            )
        else:
            record["semantic_decision"] = _validate_search_stage_evidence(
                evidence_payload,
                path=evidence_path,
                expected_stage=str(item["kind"]),
                final_search=search,
                final_evaluator=evaluator,
            )
        evidence.append(record)
    if len(evidence) != len(evidence_raw):
        raise ContractError("each evidence entry must have exactly kind and path")

    generation = dict(draft["generation"])
    _validate_generation(generation)
    if (
        learner_training_recipe["track"] != generation["track"]
        or learner_training_recipe["vps_to_win"] != generation["vps_to_win"]
    ):
        raise ContractError(
            "learner track/vps_to_win must match the locked generation regime"
        )
    if int(generation["obs_width"]) != 806 or not bool(
        learner_training_recipe["graph_history_features"]
    ):
        raise ContractError(
            "A1 fixes obs_width=806 and requires graph_history_features=true "
            "to match the gen3 learner regime"
        )
    checkpoint_records = _checkpoint_records(
        list(draft["checkpoints"]),
        base=base,
        value_readout=str(evaluator["value_readout"]),
    )
    evidence_by_kind = {str(record["kind"]): record for record in evidence}
    for stage, predecessors in {"s2": ("s1",), "s3": ("s1", "s2")}.items():
        stage_sources = {
            (str(record["path"]), str(record["sha256"]))
            for record in evidence_by_kind[stage]["semantic_decision"][
                "source_artifacts"
            ]
        }
        for predecessor in predecessors:
            expected = evidence_by_kind[predecessor]
            identity = (str(expected["path"]), str(expected["sha256"]))
            if identity not in stage_sources:
                raise ContractError(
                    f"{stage.upper()} does not inherit the exact sealed "
                    f"{predecessor.upper()} decision artifact"
                )
    producer_record = next(
        record for record in checkpoint_records if record["role"] == "producer"
    )
    for stage in ("s1", "s2", "s3"):
        stage_checkpoint = evidence_by_kind[stage]["semantic_decision"]["checkpoint"]
        if (
            stage_checkpoint["sha256"] != producer_record["sha256"]
            or Path(stage_checkpoint["path"]).resolve(strict=True)
            != Path(producer_record["path"]).resolve(strict=True)
        ):
            raise ContractError(
                f"{stage.upper()} adjudicated a different teacher checkpoint than "
                "the production contract"
            )
    categories = _validate_categories(
        list(draft["source_categories"]), checkpoint_records
    )

    fleet = dict(draft["fleet"])
    if set(fleet) != {
        "workers",
        "per_worker_games",
        "seed_base",
        "seed_block_size",
        "seed_ledger",
        "output_root",
    }:
        raise ContractError("fleet fields are not the exact pre-wave schema")
    ledger_path = _absolute_ref(str(fleet["seed_ledger"]), base=base)
    output_root = Path(str(fleet["output_root"])).expanduser()
    if not output_root.is_absolute():
        raise ContractError(
            "fleet.output_root must be an absolute path shared by the data lane"
        )
    jobs = _build_jobs(
        list(fleet["workers"]),
        seed_base=int(fleet["seed_base"]),
        block_size=int(fleet["seed_block_size"]),
        per_worker={str(k): int(v) for k, v in dict(fleet["per_worker_games"]).items()},
        output_root=str(output_root),
        contract_id=contract_id,
    )
    if seed_ledger_snapshot is None:
        _validate_against_ledger(jobs, ledger_path)
        ledger_record = _seed_ledger_snapshot(ledger_path)
    else:
        if seed_ledger_contract_sha256 is None:
            raise ContractError(
                "rebuilding from a seed-ledger snapshot requires the sealed contract SHA-256"
            )
        ledger_record = dict(seed_ledger_snapshot)
        try:
            locked_ledger_path = Path(str(ledger_record["path"])).resolve(strict=True)
        except (KeyError, OSError) as error:
            raise ContractError(f"invalid locked seed-ledger snapshot: {error}") from error
        if locked_ledger_path != ledger_path.resolve(strict=True):
            raise ContractError(
                "source draft seed-ledger path differs from its immutable snapshot"
            )
        _verify_live_seed_ledger(
            ledger_record,
            jobs,
            contract_sha256=seed_ledger_contract_sha256,
            require_all_job_claims=False,
        )

    provenance = dict(draft["provenance"])
    if set(provenance) != {
        "guard_config",
        "generator_code_files",
        "learner_code_files",
    }:
        raise ContractError(
            "provenance must have exactly guard_config, generator_code_files, "
            "and learner_code_files"
        )
    guard_path = _absolute_ref(str(provenance["guard_config"]), base=base)
    _validate_guard(
        guard_path,
        search=search,
        evaluator=evaluator,
        generation=generation,
        s1_evidence=evidence_by_kind["s1"],
    )
    guard_record = _file_record(guard_path, kind="guard_config")
    code_records = [
        _file_record(_absolute_ref(str(raw), base=base), kind="generator_code")
        for raw in provenance["generator_code_files"]
    ]
    code_paths = {Path(record["path"]).as_posix() for record in code_records}
    missing_code = {
        suffix
        for suffix in REQUIRED_GENERATOR_CODE_SUFFIXES
        if not any(path.endswith(suffix) for path in code_paths)
    }
    if missing_code:
        raise ContractError(
            f"generator_code_files omits required semantics files: {sorted(missing_code)}"
        )
    learner_code_records = [
        _file_record(_absolute_ref(str(raw), base=base), kind="learner_code")
        for raw in provenance["learner_code_files"]
    ]
    learner_code_paths = {
        Path(record["path"]).as_posix() for record in learner_code_records
    }
    missing_learner_code = {
        suffix
        for suffix in REQUIRED_LEARNER_CODE_SUFFIXES
        if not any(path.endswith(suffix) for path in learner_code_paths)
    }
    if missing_learner_code:
        raise ContractError(
            "learner_code_files omits required learner semantics files: "
            f"{sorted(missing_learner_code)}"
        )
    runtime_code_tree = _runtime_code_tree_records()
    _validate_post_wave(dict(draft["post_wave_acceptance"]))

    lock: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "contract_id": contract_id,
        "source_draft": {"path": str(draft_path), "sha256": _sha256(draft_path)},
        "science": {
            "search_operator": search,
            "search_operator_sha256": _digest_value(search),
            "effective_search_config": effective_search,
            "effective_search_config_sha256": _digest_value(effective_search),
            "evaluator": evaluator,
            "evaluator_sha256": _digest_value(evaluator),
            "value_readout": evaluator["value_readout"],
            "learner_value_objective": learner_objective,
            "learner_value_objective_sha256": _digest_value(learner_objective),
            "learner_training_recipe": learner_training_recipe,
            "learner_training_recipe_sha256": _digest_value(
                learner_training_recipe
            ),
            "evidence": evidence,
        },
        "generation": generation,
        "checkpoints": checkpoint_records,
        "source_categories": categories,
        "game_contract": {
            "total_complete_games": sum(EXPECTED_GAMES.values()),
            "category_games": dict(EXPECTED_GAMES),
            "total_attempts": sum(EXPECTED_ATTEMPTS.values()),
            "category_attempts": dict(EXPECTED_ATTEMPTS),
            "selection_rule": "lowest_seed_complete_per_job",
            "selection_before_row_expansion": True,
        },
        "fleet": {
            "workers": list(fleet["workers"]),
            "per_worker_games": dict(EXPECTED_PER_WORKER),
            "seed_base": int(fleet["seed_base"]),
            "seed_block_size": int(fleet["seed_block_size"]),
            "seed_ledger": ledger_record,
            "val_only_range": list(VAL_ONLY_SEED_RANGE),
            "output_root": str(output_root),
            "jobs": jobs,
            "seed_plan_sha256": _digest_value(jobs),
        },
        "provenance": {
            "guard_config": guard_record,
            "generator_code": code_records,
            "learner_code": learner_code_records,
            "learner_code_sha256": _digest_value(learner_code_records),
            "runtime_code_tree": runtime_code_tree,
            "runtime_code_tree_sha256": _digest_value(runtime_code_tree),
        },
        "post_wave_acceptance": dict(draft["post_wave_acceptance"]),
    }
    lock["contract_sha256"] = _digest_value(lock)
    return lock


def verify_lock(
    lock_path: Path, *, require_all_job_claims: bool = False
) -> dict[str, Any]:
    lock = _load_json(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ContractError(f"lock schema must be {LOCK_SCHEMA!r}")
    _assert_no_unresolved(lock)
    expected_digest = str(lock.get("contract_sha256", ""))
    unhashed = dict(lock)
    unhashed.pop("contract_sha256", None)
    actual_digest = _digest_value(unhashed)
    if expected_digest != actual_digest:
        raise ContractError(
            f"contract digest mismatch: expected {expected_digest or '<missing>'}, got {actual_digest}"
        )
    _verify_artifact_records([lock["source_draft"]])
    _verify_artifact_records(lock["science"]["evidence"])
    for evidence in lock["science"]["evidence"]:
        _verify_artifact_records(evidence["semantic_decision"]["source_artifacts"])
    _verify_artifact_records(lock["checkpoints"])
    _verify_artifact_records([lock["provenance"]["guard_config"]])
    _verify_artifact_records(lock["provenance"]["generator_code"])
    learner_code_records = lock["provenance"].get("learner_code")
    if not isinstance(learner_code_records, list) or not learner_code_records:
        raise ContractError("lock does not bind the learner implementation")
    if lock["provenance"].get("learner_code_sha256") != _digest_value(
        learner_code_records
    ):
        raise ContractError("learner-code provenance digest drift")
    _verify_artifact_records(learner_code_records)
    runtime_code_tree = lock["provenance"].get("runtime_code_tree")
    if not isinstance(runtime_code_tree, list) or not runtime_code_tree:
        raise ContractError("lock does not bind the transitive runtime code tree")
    if lock["provenance"].get("runtime_code_tree_sha256") != _digest_value(
        runtime_code_tree
    ):
        raise ContractError("runtime-code-tree provenance digest drift")
    _verify_artifact_records(runtime_code_tree)
    search = dict(lock["science"]["search_operator"])
    effective_search = dict(lock["science"]["effective_search_config"])
    evaluator = dict(lock["science"]["evaluator"])
    if _digest_value(search) != lock["science"]["search_operator_sha256"]:
        raise ContractError("search operator digest mismatch")
    if (
        _digest_value(effective_search)
        != lock["science"]["effective_search_config_sha256"]
    ):
        raise ContractError("effective search-config digest mismatch")
    reconstructed_effective = _effective_search(search)
    if _digest_value(reconstructed_effective) != _digest_value(effective_search):
        raise ContractError(
            "effective search config does not reconstruct from selected operator"
        )
    if _digest_value(evaluator) != lock["science"]["evaluator_sha256"]:
        raise ContractError("evaluator digest mismatch")
    if evaluator["value_readout"] != lock["science"]["value_readout"]:
        raise ContractError("value readout attestation drift")
    _verify_checkpoint_provenance_records(
        lock["checkpoints"], value_readout=str(evaluator["value_readout"])
    )
    learner_objective = dict(lock["science"]["learner_value_objective"])
    _validate_learner_objective(learner_objective)
    if (
        _digest_value(learner_objective)
        != lock["science"]["learner_value_objective_sha256"]
    ):
        raise ContractError("learner value-objective digest mismatch")
    learner_training_recipe = dict(lock["science"]["learner_training_recipe"])
    _validate_learner_training_recipe(learner_training_recipe)
    if (
        _digest_value(learner_training_recipe)
        != lock["science"]["learner_training_recipe_sha256"]
    ):
        raise ContractError("learner training-recipe digest mismatch")
    jobs = list(lock["fleet"]["jobs"])
    if _digest_value(jobs) != lock["fleet"]["seed_plan_sha256"]:
        raise ContractError("seed plan digest mismatch")
    assert_disjoint_seed_blocks(
        [
            (job["job_id"], int(job["base_seed"]), int(job["attempts"]))
            for job in jobs
        ]
    )
    if Counter({category: 0 for category in EXPECTED_GAMES}) + Counter(
        {
            category: sum(int(j["games"]) for j in jobs if j["category"] == category)
            for category in EXPECTED_GAMES
        }
    ) != Counter(EXPECTED_GAMES):
        raise ContractError("job category totals drifted from 9600/1800/600")
    if Counter(
        {
            category: sum(
                int(j["attempts"]) for j in jobs if j["category"] == category
            )
            for category in EXPECTED_ATTEMPTS
        }
    ) != Counter(EXPECTED_ATTEMPTS):
        raise ContractError("job attempt totals drifted from the bounded reserve")
    for job in jobs:
        if _ranges_overlap(
            (int(job["base_seed"]), int(job["seed_end"])),
            tuple(lock["fleet"]["val_only_range"]),
        ):
            raise ContractError(f"job {job['job_id']} overlaps VAL-ONLY")
    _verify_live_seed_ledger(
        dict(lock["fleet"]["seed_ledger"]),
        jobs,
        contract_sha256=expected_digest,
        require_all_job_claims=require_all_job_claims,
    )
    _validate_guard(
        Path(lock["provenance"]["guard_config"]["path"]),
        search=search,
        evaluator=evaluator,
        generation=dict(lock["generation"]),
        s1_evidence=next(
            record for record in lock["science"]["evidence"] if record["kind"] == "s1"
        ),
    )
    _validate_post_wave(dict(lock["post_wave_acceptance"]))
    rebuilt = build_lock(
        Path(str(lock["source_draft"]["path"])),
        seed_ledger_snapshot=dict(lock["fleet"]["seed_ledger"]),
        seed_ledger_contract_sha256=expected_digest,
    )
    if _digest_value(rebuilt) != _digest_value(lock):
        raise ContractError(
            "lock does not exactly reconstruct from its immutable source draft"
        )
    return lock


def _bool_flag(name: str, value: bool) -> str:
    return name if value else "--no-" + name.removeprefix("--")


def _producer(lock: dict[str, Any]) -> dict[str, Any]:
    return next(
        record for record in lock["checkpoints"] if record["role"] == "producer"
    )


def _category_by_name(lock: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        category for category in lock["source_categories"] if category["name"] == name
    )


def _category_opponent_sha256(lock: dict[str, Any], name: str) -> list[str]:
    """Exact opponent bytes; self-play names the producer on the second seat."""

    if name == "current_producer":
        return [_producer(lock)["sha256"]]
    category = _category_by_name(lock, name)
    checkpoints = {record["id"]: record for record in lock["checkpoints"]}
    return sorted(
        checkpoints[checkpoint_id]["sha256"]
        for checkpoint_id in category["checkpoint_ids"]
    )


def _render_mix_manifest(lock: dict[str, Any], category: str) -> dict[str, Any]:
    spec = _category_by_name(lock, category)
    by_id = {record["id"]: record for record in lock["checkpoints"]}
    return {
        "_a1_contract": {
            "contract_sha256": lock["contract_sha256"],
            "category": category,
        },
        "categories": [
            {
                "name": category,
                "weight": 1.0,
                "source": "checkpoint_list",
                "pending": False,
                "engine": None,
                "checkpoints": [
                    {
                        "path": by_id[checkpoint_id]["path"],
                        "version": -1,
                        "md5": by_id[checkpoint_id]["md5"],
                    }
                    for checkpoint_id in spec["checkpoint_ids"]
                ],
            }
        ],
    }


def _generator_argv(
    lock: dict[str, Any], job: dict[str, Any], *, mix_paths: dict[str, Path]
) -> list[str]:
    search = lock["science"]["search_operator"]
    evaluator = lock["science"]["evaluator"]
    generation = lock["generation"]
    producer = _producer(lock)
    argv = [
        "tools/generate_gumbel_selfplay_data.py",
        "--out-dir",
        job["output_dir"],
        "--games",
        str(job["attempts"]),
        "--workers",
        str(generation["workers_per_gpu"]),
        "--checkpoint",
        producer["path"],
        "--device",
        generation["device"],
        "--n-full",
        str(search["n_full"]),
        "--n-fast",
        str(search["n_fast"]),
        "--p-full",
        str(search["p_full"]),
        "--c-visit",
        str(search["c_visit"]),
        "--c-scale",
        str(search["c_scale"]),
        "--rescale-noise-floor-c",
        str(search["rescale_noise_floor_c"]),
        "--sigma-eval",
        str(search["sigma_eval"]),
        "--wide-candidates-threshold",
        str(search["wide_candidates_threshold"]),
        "--max-depth",
        str(search["max_depth"]),
        "--max-decisions",
        str(generation["max_decisions"]),
        "--temperature-decisions",
        str(generation["temperature_decisions"]),
        "--temperature-high",
        str(generation["temperature_high"]),
        "--temperature-low",
        str(generation["temperature_low"]),
        "--late-temperature",
        str(generation["late_temperature"]),
        "--prior-temperature",
        str(evaluator["prior_temperature"]),
        "--value-scale",
        str(evaluator["value_scale"]),
        "--value-readout",
        str(evaluator["value_readout"]),
        "--eval-cache-size",
        str(evaluator["cache_size"]),
        "--track",
        generation["track"],
        "--vps-to-win",
        str(generation["vps_to_win"]),
        "--obs-width",
        str(generation["obs_width"]),
        "--base-seed",
        str(job["base_seed"]),
        "--shard-size",
        str(generation["shard_size"]),
        "--format",
        generation["format"],
        "--ledger-claim-label",
        job["claim_label"],
        _bool_flag("--symmetry-averaged-eval", bool(search["symmetry_averaged_eval"])),
        _bool_flag("--wide-roots-always-full", bool(search["wide_roots_always_full"])),
        _bool_flag(
            "--correct-rust-chance-spectra", bool(search["correct_rust_chance_spectra"])
        ),
        _bool_flag("--lazy-interior-chance", bool(search["lazy_interior_chance"])),
        _bool_flag("--exact-budget-sh", bool(search["exact_budget_sh"])),
        "--exact-budget-sh-min-n",
        str(search["exact_budget_sh_min_n"]),
        _bool_flag("--belief-chance-spectra", bool(search["belief_chance_spectra"])),
        _bool_flag("--public-observation", bool(evaluator["public_observation"])),
        _bool_flag("--rust-featurize", bool(evaluator["rust_featurize"])),
        _bool_flag("--eval-server", bool(generation["eval_server"])),
        "--seed-claim",
    ]
    optional = (
        ("--n-full-wide", search["n_full_wide"]),
        ("--n-full-wide-threshold", search["n_full_wide_threshold"]),
        ("--raw-policy-above-width", search["raw_policy_above_width"]),
        (
            "--symmetry-averaged-eval-threshold",
            search["symmetry_averaged_eval_threshold"],
        ),
        ("--late-temperature-decisions", generation["late_temperature_decisions"]),
    )
    for flag, value in optional:
        if value is not None:
            argv.extend((flag, str(value)))
    if job["category"] != "current_producer":
        argv.extend(("--opponent-mix-manifest", str(mix_paths[job["category"]])))
    return argv


def _job_attestation(lock: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "a1-generation-job-attestation-v2",
        "contract_sha256": lock["contract_sha256"],
        "seed_plan_sha256": lock["fleet"]["seed_plan_sha256"],
        "job_id": job["job_id"],
        "worker_id": job["worker_id"],
        "category": job["category"],
        "base_seed": job["base_seed"],
        "games": job["games"],
        "attempts": job["attempts"],
        "seed_end": job["seed_end"],
        "producer_checkpoint_sha256": _producer(lock)["sha256"],
        "opponent_checkpoint_sha256": _category_opponent_sha256(
            lock, job["category"]
        ),
        "search_operator_sha256": lock["science"]["search_operator_sha256"],
        "effective_search_config_sha256": lock["science"][
            "effective_search_config_sha256"
        ],
        "evaluator_sha256": lock["science"]["evaluator_sha256"],
        "runtime_code_tree_sha256": lock["provenance"][
            "runtime_code_tree_sha256"
        ],
        "teacher_value_readout": lock["science"]["value_readout"],
    }


def _ledger_claim_row(lock: dict[str, Any], job: dict[str, Any]) -> str:
    return (
        f"[{int(job['base_seed'])} – {int(job['seed_end'])}) | "
        f"{_ledger_claim_label(str(lock['contract_sha256']), job)}"
    )


def _create_readonly(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise ContractError(
            f"refusing to overwrite immutable artifact {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def render(lock_path: Path, out_dir: Path) -> dict[str, Any]:
    lock = verify_lock(lock_path)
    out_dir = out_dir.absolute()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ContractError(f"render output must be absent or empty: {out_dir}")
    mix_paths: dict[str, Path] = {}
    for category in ("recent_history", "hard_negative"):
        path = out_dir / "opponent_mix" / f"{category}.json"
        _create_readonly(path, _render_mix_manifest(lock, category))
        mix_paths[category] = path
    commands = []
    for job in lock["fleet"]["jobs"]:
        argv = _generator_argv(lock, job, mix_paths=mix_paths)
        attestation = _job_attestation(lock, job)
        attestation_source = out_dir / "job_attestations" / f"{job['job_id']}.json"
        _create_readonly(attestation_source, attestation)
        commands.append(
            {
                "job_id": job["job_id"],
                "worker_id": job["worker_id"],
                "host_alias": job["host_alias"],
                "gpu": job["gpu"],
                "category": job["category"],
                "environment": {
                    "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                    "CATAN_SEED_LEDGER": lock["fleet"]["seed_ledger"]["path"],
                    "CATAN_A1_CONTRACT_SHA256": lock["contract_sha256"],
                },
                "python": "python",
                "argv": argv,
                "argv_sha256": _digest_value(argv),
                "ledger_claim": {
                    "path": lock["fleet"]["seed_ledger"]["path"],
                    "row": _ledger_claim_row(lock, job),
                    "row_sha256": _digest_value(_ledger_claim_row(lock, job)),
                },
                "output_attestation": {
                    "source": str(attestation_source),
                    "source_file_sha256": _sha256(attestation_source),
                    "destination": str(Path(job["output_dir"]) / "a1_contract.json"),
                    "payload_sha256": _digest_value(attestation),
                },
                "must_run_after": (
                    []
                    if job["category"] == "current_producer"
                    else [
                        f"{job['worker_id']}__{list(EXPECTED_GAMES)[list(EXPECTED_GAMES).index(job['category']) - 1]}"
                    ]
                ),
            }
        )
    payload = {
        "schema_version": RENDER_SCHEMA,
        "contract_path": str(lock_path.absolute()),
        "contract_sha256": lock["contract_sha256"],
        "required_artifacts": {
            "checkpoints": [
                {key: record[key] for key in ("id", "path", "sha256", "md5")}
                for record in lock["checkpoints"]
            ],
            "seed_ledger": lock["fleet"]["seed_ledger"],
            "guard_config": lock["provenance"]["guard_config"],
            "generator_code": lock["provenance"]["generator_code"],
            "learner_code": lock["provenance"]["learner_code"],
            "learner_code_sha256": lock["provenance"]["learner_code_sha256"],
            "runtime_code_tree": lock["provenance"]["runtime_code_tree"],
            "runtime_code_tree_sha256": lock["provenance"][
                "runtime_code_tree_sha256"
            ],
            "rendered_opponent_mix": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in mix_paths.values()
            ],
        },
        "execution_policy": {
            "execute": False,
            "category_jobs_are_sequential_per_gpu": True,
            "operator_must_claim_ledger_before_each_job": True,
            "operator_must_copy_output_attestation_before_each_job": True,
            "operator_must_run_post_wave_audit_before_ingest": True,
        },
        "commands": commands,
    }
    payload["render_sha256"] = _digest_value(payload)
    _create_readonly(out_dir / "commands.json", payload)
    return payload


def _resolve_shard(manifest: Path, raw: str) -> Path:
    path = Path(raw)
    # A frozen handoff may not guess by basename or process cwd.  Such fallback
    # made a stale absolute path silently bind unrelated bytes with the same
    # filename.  Relative shard paths have exactly one owner: the manifest.
    resolved = path if path.is_absolute() else manifest.parent / path
    if not resolved.is_file():
        raise ContractError(f"manifest {manifest} points to missing shard {raw}")
    return resolved.absolute()


def _row_legal(action: int, legal: np.ndarray, mask: np.ndarray | None) -> bool:
    if mask is not None:
        legal = legal[np.asarray(mask, dtype=bool)]
    return bool(np.any(np.asarray(legal, dtype=np.int64) == int(action)))


def _selected_telemetry_arrays(
    payload: Any,
    *,
    game_seeds: np.ndarray,
    selected_mask: np.ndarray,
    max_decisions: int,
    where: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load mandatory report inputs without inventing fallback telemetry."""

    missing = sorted(
        column
        for column in REQUIRED_SELECTED_TELEMETRY_COLUMNS
        if column not in payload
    )
    if missing:
        raise ContractError(f"{where}: missing selected telemetry columns {missing}")

    row_aligned: dict[str, np.ndarray] = {}
    for column in ("is_forced", "used_full_search", "phase", "decision_index"):
        raw = np.asarray(payload[column])
        if raw.shape != game_seeds.shape:
            raise ContractError(
                f"{where}: telemetry column {column} is not row-aligned"
            )
        row_aligned[column] = raw[selected_mask]

    phase = row_aligned["phase"].astype(str)
    if np.any(np.char.str_len(phase) == 0) or np.any(phase == "<missing>"):
        raise ContractError(f"{where}: selected phase telemetry is empty")
    decision = np.asarray(row_aligned["decision_index"], dtype=np.int64)
    if np.any(decision < 0) or np.any(decision >= int(max_decisions)):
        raise ContractError(
            f"{where}: selected decision_index is outside [0,{max_decisions})"
        )

    raw_policy = np.asarray(payload["target_policy"], dtype=np.float64)
    raw_policy_mask = np.asarray(payload["target_policy_mask"], dtype=bool)
    if (
        raw_policy.ndim != 2
        or raw_policy.shape[0] != game_seeds.size
        or raw_policy.shape[1] == 0
        or raw_policy_mask.shape != raw_policy.shape
    ):
        raise ContractError(
            f"{where}: target_policy/target_policy_mask are empty or not row-aligned"
        )
    policy = raw_policy[selected_mask]
    policy_mask = raw_policy_mask[selected_mask]
    if np.any(~np.any(policy_mask, axis=1)):
        raise ContractError(f"{where}: selected target policy has no active entries")
    active_values = policy[policy_mask]
    if np.any(~np.isfinite(active_values)) or np.any(active_values < 0.0):
        raise ContractError(
            f"{where}: selected target policy has invalid active probabilities"
        )
    mass = np.where(policy_mask, policy, 0.0).sum(axis=1)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ContractError(f"{where}: selected target policy has non-positive mass")

    return (
        np.asarray(row_aligned["is_forced"], dtype=bool),
        np.asarray(row_aligned["used_full_search"], dtype=bool),
        phase,
        decision,
        policy,
        policy_mask,
    )


def _advance_game_seed_runs(
    game_seeds: np.ndarray,
    *,
    active_seed: int | None,
    closed_seeds: set[int],
    where: str,
) -> int | None:
    """Track one contiguous raw row run per game across ordered shards.

    Keeping ``active_seed`` across calls permits a game to span adjacent shard
    files.  Once another seed begins, the old run is closed forever; seeing it
    again would duplicate a whole/partial game under the same selected seed.
    """

    flat = np.asarray(game_seeds, dtype=np.int64).reshape(-1)
    current = active_seed
    for raw_seed in flat:
        seed = int(raw_seed)
        if current is None:
            if seed in closed_seeds:
                raise ContractError(
                    f"{where}: game_seed {seed} starts a second non-contiguous raw run"
                )
            current = seed
            continue
        if seed == current:
            continue
        closed_seeds.add(current)
        if seed in closed_seeds:
            raise ContractError(
                f"{where}: game_seed {seed} starts a second non-contiguous raw run"
            )
        current = seed
    return current


def _expected_cli_fields(lock: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    search = lock["science"]["search_operator"]
    evaluator = lock["science"]["evaluator"]
    generation = lock["generation"]
    late_decisions = generation["late_temperature_decisions"]
    return {
        "out_dir": job["output_dir"],
        "games": int(job["attempts"]),
        "workers": int(generation["workers_per_gpu"]),
        "checkpoint": _producer(lock)["path"],
        "device": generation["device"],
        "n_full": search["n_full"],
        "n_fast": search["n_fast"],
        "p_full": search["p_full"],
        "c_visit": search["c_visit"],
        "c_scale": search["c_scale"],
        "rescale_noise_floor_c": search["rescale_noise_floor_c"],
        "sigma_eval": search["sigma_eval"],
        "n_full_wide": search["n_full_wide"],
        "n_full_wide_threshold": search["n_full_wide_threshold"],
        "wide_roots_always_full": search["wide_roots_always_full"],
        "raw_policy_above_width": search["raw_policy_above_width"],
        "symmetry_averaged_eval": search["symmetry_averaged_eval"],
        "symmetry_averaged_eval_threshold": search["symmetry_averaged_eval_threshold"],
        "wide_candidates_threshold": search["wide_candidates_threshold"],
        "correct_rust_chance_spectra": search["correct_rust_chance_spectra"],
        "lazy_interior_chance": search["lazy_interior_chance"],
        "exact_budget_sh": search["exact_budget_sh"],
        "exact_budget_sh_min_n": search["exact_budget_sh_min_n"],
        "belief_chance_spectra": search["belief_chance_spectra"],
        "max_depth": search["max_depth"],
        "prior_temperature": evaluator["prior_temperature"],
        "value_scale": evaluator["value_scale"],
        "value_readout": evaluator["value_readout"],
        "public_observation": evaluator["public_observation"],
        "rust_featurize": evaluator["rust_featurize"],
        "eval_cache_size": evaluator["cache_size"],
        "track": generation["track"],
        "vps_to_win": generation["vps_to_win"],
        "obs_width": generation["obs_width"],
        "max_decisions": generation["max_decisions"],
        "temperature_decisions": generation["temperature_decisions"],
        "temperature_decisions_effective": generation["temperature_decisions"],
        "temperature_move_fraction": float(generation["temperature_decisions"])
        / float(generation["max_decisions"]),
        "temperature_high": generation["temperature_high"],
        "temperature_low": generation["temperature_low"],
        "late_temperature_decisions": generation["late_temperature_decisions"],
        "late_temperature_move_fraction": (
            None
            if late_decisions is None
            else float(late_decisions) / float(generation["max_decisions"])
        ),
        "late_temperature": generation["late_temperature"],
        "base_seed": int(job["base_seed"]),
        "shard_size": generation["shard_size"],
        "format": generation["format"],
        "eval_server": generation["eval_server"],
        "opponent_pool_manifest": None,
        "exploiter_fraction": None,
        "seed_claim": True,
        "ledger_claim_label": job["claim_label"],
        "skip_guards": False,
    }


def _expected_selfplay_config(lock: dict[str, Any]) -> dict[str, Any]:
    generation = lock["generation"]
    search = lock["science"]["search_operator"]
    opening_fraction = float(generation["temperature_decisions"]) / float(
        generation["max_decisions"]
    )
    late = generation["late_temperature_decisions"]
    late_fraction = (
        None if late is None else float(late) / float(generation["max_decisions"])
    )
    effective = dataclasses.asdict(
        GumbelSelfPlayConfig(
            colors=("RED", "BLUE"),
            track=str(generation["track"]),
            vps_to_win=int(generation["vps_to_win"]),
            obs_width=int(generation["obs_width"]),
            max_decisions=int(generation["max_decisions"]),
            temperature_move_fraction=opening_fraction,
            temperature_high=float(generation["temperature_high"]),
            temperature_low=float(generation["temperature_low"]),
            late_temperature_move_fraction=late_fraction,
            late_temperature=float(generation["late_temperature"]),
            correct_rust_chance_spectra=bool(search["correct_rust_chance_spectra"]),
        )
    )
    # Match JSON-loaded worker manifests (tuples become lists).
    return json.loads(json.dumps(effective))


def audit_outputs(lock_path: Path, out_path: Path) -> dict[str, Any]:
    """Deep post-wave audit.  This is callable only after generation; it never launches it."""
    try:
        lock_path = lock_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve contract lock {lock_path}: {error}") from error
    lock = verify_lock(lock_path, require_all_job_claims=True)
    all_seeds: set[int] = set()
    category_seeds: dict[str, set[int]] = {name: set() for name in EXPECTED_GAMES}
    rows_by_seed: Counter[int] = Counter()
    shard_records: list[dict[str, Any]] = []
    invalid_actions = 0
    rows = 0
    forced = 0
    full_active = 0
    target_entropy_sum = 0.0
    target_entropy_count = 0
    phases: Counter[str] = Counter()
    decision_bins: Counter[str] = Counter()
    legal_widths: Counter[str] = Counter()
    errors: list[str] = []
    seen_shards: set[Path] = set()
    job_selections: list[dict[str, Any]] = []
    selected_game_records: list[dict[str, Any]] = []
    producer = _producer(lock)
    checkpoint_by_id = {record["id"]: record for record in lock["checkpoints"]}
    category_specs = {item["name"]: item for item in lock["source_categories"]}
    for job in lock["fleet"]["jobs"]:
        attestation_path = Path(job["output_dir"]) / "a1_contract.json"
        expected_attestation = _job_attestation(lock, job)
        if not attestation_path.is_file():
            errors.append(f"missing contract attestation: {attestation_path}")
        else:
            try:
                actual_attestation = _load_json(attestation_path)
                if actual_attestation != expected_attestation:
                    errors.append(f"{job['job_id']}: output contract attestation drift")
                shard_records.append(
                    {
                        "kind": "contract_attestation",
                        "path": str(attestation_path),
                        "sha256": _sha256(attestation_path),
                        "job_id": job["job_id"],
                        "category": job["category"],
                    }
                )
            except ContractError as error:
                errors.append(f"{job['job_id']}: {error}")
        manifest_path = Path(job["output_dir"]) / "manifest.json"
        if not manifest_path.is_file():
            errors.append(f"missing manifest: {manifest_path}")
            continue
        manifest = _load_json(manifest_path)
        shard_records.append(
            {
                "kind": "generation_manifest",
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "job_id": job["job_id"],
                "category": job["category"],
                "producer_checkpoint_sha256": producer["sha256"],
                "opponent_checkpoint_sha256": _category_opponent_sha256(
                    lock, job["category"]
                ),
            }
        )
        if int(manifest.get("games_requested", -1)) != int(job["attempts"]):
            errors.append(f"{job['job_id']}: games_requested drift")
        if int(manifest.get("games_completed", -1)) != int(job["attempts"]):
            errors.append(f"{job['job_id']}: incomplete attempts")
        if int(manifest.get("games_failed", -1)) != 0 or manifest.get("errors"):
            errors.append(f"{job['job_id']}: failures/errors present")
        manifest_truncated = int(manifest.get("games_truncated", -1))
        if manifest_truncated < 0:
            errors.append(f"{job['job_id']}: invalid games_truncated")
        if int(manifest.get("base_seed", -1)) != int(job["base_seed"]):
            errors.append(f"{job['job_id']}: base_seed drift")
        if str(manifest.get("checkpoint")) != producer["path"]:
            errors.append(f"{job['job_id']}: producer checkpoint path drift")
        cli = dict(manifest.get("cli_args", {}))
        for key, expected in _expected_cli_fields(lock, job).items():
            if cli.get(key) != expected:
                errors.append(
                    f"{job['job_id']}: cli_args.{key}={cli.get(key)!r}, expected {expected!r}"
                )
        try:
            actual_config_hash = GenerateConfig.from_namespace(
                argparse.Namespace(**cli)
            ).config_hash()
        except Exception as error:  # noqa: BLE001 - malformed config provenance blocks ingest.
            errors.append(f"{job['job_id']}: cannot reconstruct config_hash: {error}")
        else:
            if manifest.get("config_hash") != actual_config_hash:
                errors.append(
                    f"{job['job_id']}: config_hash={manifest.get('config_hash')!r}, "
                    f"reconstructed={actual_config_hash!r}"
                )
        worker_summaries = list(manifest.get("worker_summaries", []))
        expected_workers = min(
            int(lock["generation"]["workers_per_gpu"]), int(job["attempts"])
        )
        if len(worker_summaries) != expected_workers:
            errors.append(
                f"{job['job_id']}: worker_summaries={len(worker_summaries)}, "
                f"expected {expected_workers}"
            )
        for raw_worker_summary in worker_summaries:
            worker_manifest_path = Path(raw_worker_summary)
            if not worker_manifest_path.is_file():
                errors.append(
                    f"{job['job_id']}: missing worker manifest {worker_manifest_path}"
                )
                continue
            worker_manifest = _load_json(worker_manifest_path)
            actual_search = dict(worker_manifest.get("search_config", {}))
            actual_search.pop("seed", None)
            if actual_search != lock["science"]["effective_search_config"]:
                errors.append(f"{job['job_id']}: worker effective search config drift")
            if worker_manifest.get("selfplay_config") != _expected_selfplay_config(
                lock
            ):
                errors.append(
                    f"{job['job_id']}: worker effective self-play config drift"
                )
        expected_category = category_specs[job["category"]]
        allowed_md5 = {
            checkpoint_by_id[checkpoint_id]["md5"]
            for checkpoint_id in expected_category["checkpoint_ids"]
        }
        mix_manifest_raw = cli.get("opponent_mix_manifest")
        if job["category"] == "current_producer":
            if mix_manifest_raw not in (None, ""):
                errors.append(
                    f"{job['job_id']}: current-producer job unexpectedly used a mix"
                )
        elif not mix_manifest_raw:
            errors.append(f"{job['job_id']}: missing category-specific opponent mix")
        else:
            try:
                mix_manifest = _load_json(Path(str(mix_manifest_raw)))
                attestation = dict(mix_manifest.get("_a1_contract", {}))
                if attestation != {
                    "contract_sha256": lock["contract_sha256"],
                    "category": job["category"],
                }:
                    errors.append(
                        f"{job['job_id']}: opponent-mix contract attestation drift"
                    )
            except ContractError as error:
                errors.append(f"{job['job_id']}: {error}")
        # Pass 1 inventories every attempt and hashes every shard.  Selection is
        # game-level and deterministic; no reserve/truncated row contributes to
        # metrics, the holdout, or the accepted shard-row inventory below.
        job_shards: list[Path] = []
        seed_status: dict[int, tuple[bool, bool]] = {}
        active_seed_run: int | None = None
        closed_seed_runs: set[int] = set()
        for raw_shard in manifest.get("shards", []):
            try:
                shard = _resolve_shard(manifest_path, str(raw_shard))
                canonical_shard = shard.resolve(strict=True)
                if canonical_shard in seen_shards:
                    raise ContractError(f"duplicate shard reference {canonical_shard}")
                seen_shards.add(canonical_shard)
                job_shards.append(canonical_shard)
                shard_records.append(
                    {
                        "kind": "data_shard",
                        "path": str(canonical_shard),
                        "sha256": _sha256(canonical_shard),
                        "job_id": job["job_id"],
                        "category": job["category"],
                        "producer_checkpoint_sha256": producer["sha256"],
                        "opponent_checkpoint_sha256": _category_opponent_sha256(
                            lock, job["category"]
                        ),
                        "search_operator_sha256": lock["science"][
                            "search_operator_sha256"
                        ],
                        "effective_search_config_sha256": lock["science"][
                            "effective_search_config_sha256"
                        ],
                        "evaluator_sha256": lock["science"]["evaluator_sha256"],
                    }
                )
                with np.load(canonical_shard, allow_pickle=False) as payload:
                    game_seeds = np.asarray(payload["game_seed"], dtype=np.int64)
                    terminated = np.asarray(payload["terminated"], dtype=bool)
                    truncated = np.asarray(payload["truncated"], dtype=bool)
                    if game_seeds.ndim != 1 or not (
                        terminated.shape == truncated.shape == game_seeds.shape
                    ):
                        raise ContractError("game status arrays are not row-aligned")
                    active_seed_run = _advance_game_seed_runs(
                        game_seeds,
                        active_seed=active_seed_run,
                        closed_seeds=closed_seed_runs,
                        where=job["job_id"],
                    )
                    for seed in np.unique(game_seeds):
                        seed_int = int(seed)
                        if not int(job["base_seed"]) <= seed_int < int(job["seed_end"]):
                            errors.append(
                                f"{job['job_id']}: out-of-range seed {seed_int}"
                            )
                        mask = game_seeds == seed
                        statuses = set(
                            zip(
                                map(bool, terminated[mask].tolist()),
                                map(bool, truncated[mask].tolist()),
                            )
                        )
                        if len(statuses) != 1:
                            errors.append(
                                f"{job['job_id']}: inconsistent row status for seed {seed_int}"
                            )
                            continue
                        status = next(iter(statuses))
                        prior = seed_status.get(seed_int)
                        if prior is not None and prior != status:
                            errors.append(
                                f"{job['job_id']}: cross-shard status drift for seed {seed_int}"
                            )
                        seed_status[seed_int] = status
            except (ContractError, KeyError, OSError, ValueError) as error:
                errors.append(f"{job['job_id']}: {error}")

        expected_attempt_seeds = set(
            range(int(job["base_seed"]), int(job["seed_end"]))
        )
        observed_attempt_seeds = set(seed_status)
        if observed_attempt_seeds != expected_attempt_seeds:
            errors.append(
                f"{job['job_id']}: attempted seed set drift; "
                f"missing={len(expected_attempt_seeds - observed_attempt_seeds)}, "
                f"extra={len(observed_attempt_seeds - expected_attempt_seeds)}"
            )
        observed_truncated = sum(
            1 for terminated, truncated in seed_status.values() if truncated or not terminated
        )
        if manifest_truncated != observed_truncated:
            errors.append(
                f"{job['job_id']}: games_truncated={manifest_truncated}, "
                f"row evidence={observed_truncated}"
            )
        complete = sorted(
            seed
            for seed, (terminated, truncated) in seed_status.items()
            if terminated and not truncated
        )
        selected = set(complete[: int(job["games"])])
        if len(selected) != int(job["games"]):
            errors.append(
                f"{job['job_id']}: only {len(complete)} complete attempts for "
                f"selected quota {job['games']}"
            )
        category_seeds[job["category"]].update(selected)
        selected_game_records.extend(
            {
                "game_seed": int(seed),
                "job_id": job["job_id"],
                "worker_id": job["worker_id"],
                "category": job["category"],
                "producer_checkpoint_sha256": producer["sha256"],
                "opponent_checkpoint_sha256": _category_opponent_sha256(
                    lock, job["category"]
                ),
            }
            for seed in sorted(selected)
        )
        job_selections.append(
            {
                "job_id": job["job_id"],
                "category": job["category"],
                "attempts": int(job["attempts"]),
                "complete_attempts": len(complete),
                "truncated_attempts": observed_truncated,
                "selected_games": len(selected),
                "selected_seed_sha256": _digest_value(sorted(selected)),
            }
        )

        # Pass 2 computes all acceptance metrics from selected complete games
        # only.  Reserve rows remain hashed in shard_records but are excluded.
        for shard in job_shards:
            try:
                with np.load(shard, allow_pickle=False) as payload:
                    game_seeds = np.asarray(payload["game_seed"], dtype=np.int64)
                    selected_mask = np.isin(
                        game_seeds, np.asarray(sorted(selected), dtype=np.int64)
                    )
                    selected_indices = np.flatnonzero(selected_mask)
                    n = int(selected_indices.size)
                    if n == 0:
                        continue
                    rows += n
                    selected_seeds = game_seeds[selected_mask]
                    rows_by_seed.update(map(int, selected_seeds.tolist()))
                    actions = np.asarray(payload["action_taken"])[selected_mask]
                    legal_ids = np.asarray(payload["legal_action_ids"])[selected_mask]
                    raw_legal_mask = payload.get("legal_action_mask")
                    legal_mask = (
                        None
                        if raw_legal_mask is None
                        else np.asarray(raw_legal_mask)[selected_mask]
                    )
                    for index in range(n):
                        mask = None if legal_mask is None else legal_mask[index]
                        invalid_actions += int(
                            not _row_legal(int(actions[index]), legal_ids[index], mask)
                        )
                    if np.any(np.asarray(payload["truncated"], dtype=bool)[selected_mask]):
                        errors.append(f"{job['job_id']}: selected truncation leaked")
                    if not np.all(
                        np.asarray(payload["terminated"], dtype=bool)[selected_mask]
                    ):
                        errors.append(f"{job['job_id']}: selected incomplete game leaked")
                    (
                        is_forced,
                        used_full,
                        phase,
                        decision,
                        policy,
                        policy_mask,
                    ) = _selected_telemetry_arrays(
                        payload,
                        game_seeds=game_seeds,
                        selected_mask=selected_mask,
                        max_decisions=int(lock["generation"]["max_decisions"]),
                        where=job["job_id"],
                    )
                    forced += int(is_forced.sum())
                    full_active += int(np.sum(used_full & ~is_forced))
                    phases.update(phase.tolist())
                    for value in decision:
                        key = (
                            f"{(int(value) // 25) * 25:03d}-"
                            f"{(int(value) // 25) * 25 + 24:03d}"
                        )
                        decision_bins[key] += 1
                    for index in range(n):
                        probs = policy[index][policy_mask[index]]
                        probs = probs[np.isfinite(probs) & (probs > 0)]
                        legal_widths[str(int(probs.size))] += 1
                        total = float(probs.sum())
                        if total > 0:
                            normalized = probs / total
                            target_entropy_sum += float(
                                -np.sum(normalized * np.log(normalized))
                            )
                            target_entropy_count += 1
                    if job["category"] != "current_producer":
                        tags = np.asarray(
                            payload.get(
                                "opponent_tag", np.full(game_seeds.size, "")
                            )
                        ).astype(str)[selected_mask]
                        md5s = np.asarray(
                            payload.get(
                                "opponent_checkpoint_md5",
                                np.full(game_seeds.size, ""),
                            )
                        ).astype(str)[selected_mask]
                        if np.any(tags != job["category"]):
                            errors.append(
                                f"{job['job_id']}: opponent source label drift"
                            )
                        if any(md5 not in allowed_md5 for md5 in md5s.tolist()):
                            errors.append(
                                f"{job['job_id']}: opponent checkpoint identity drift"
                            )
                    elif "opponent_tag" in payload:
                        tags = np.asarray(payload["opponent_tag"]).astype(str)[
                            selected_mask
                        ]
                        if np.any(tags != ""):
                            errors.append(
                                f"{job['job_id']}: current-producer source label drift"
                            )
            except (ContractError, KeyError, OSError, ValueError, IndexError) as error:
                errors.append(f"{job['job_id']}: {error}")
    for category, seeds in category_seeds.items():
        if len(seeds) != EXPECTED_GAMES[category]:
            errors.append(
                f"{category}: unique complete games={len(seeds)}, expected {EXPECTED_GAMES[category]}"
            )
        duplicate = all_seeds.intersection(seeds)
        if duplicate:
            errors.append(
                f"{category}: {len(duplicate)} game seeds overlap another category"
            )
        all_seeds.update(seeds)
    if invalid_actions:
        errors.append(f"invalid_teacher_actions={invalid_actions}, expected 0")
    if any(
        _ranges_overlap((seed, seed + 1), VAL_ONLY_SEED_RANGE) for seed in all_seeds
    ):
        errors.append("selected corpus overlaps VAL-ONLY seeds")
    record_seeds = [int(record["game_seed"]) for record in selected_game_records]
    if len(record_seeds) != len(all_seeds) or set(record_seeds) != all_seeds:
        errors.append("selected game/source records do not bijectively cover selected seeds")
    validation_contract = lock["post_wave_acceptance"]["validation_holdout"]
    validation_seed_manifest_path = out_path.with_suffix(".validation_seeds.json")
    selected_game_manifest_path = out_path.with_suffix(".selected_games.json")
    validation_manifest: dict[str, Any] | None = None
    selected_game_manifest: dict[str, Any] | None = None
    if not errors:
        unique_seeds = np.asarray(sorted(all_seeds), dtype=np.int64)
        rng = np.random.default_rng(int(validation_contract["validation_seed"]))
        shuffled = rng.permutation(unique_seeds)
        target_rows = max(
            1, int(round(rows * float(validation_contract["validation_fraction"])))
        )
        selected: list[int] = []
        selected_rows = 0
        for seed in shuffled:
            selected.append(int(seed))
            selected_rows += int(rows_by_seed[int(seed)])
            if selected_rows >= target_rows:
                break
        held_out = np.asarray(sorted(selected), dtype="<i8")
        seed_set_digest = "sha256:" + hashlib.sha256(held_out.tobytes()).hexdigest()
        validation_manifest = {
            "schema_version": "train-validation-game-seeds-v1",
            "a1_contract_sha256": lock["contract_sha256"],
            "validation_fraction": float(validation_contract["validation_fraction"]),
            "validation_seed": int(validation_contract["validation_seed"]),
            "validation_max_samples": int(
                validation_contract["validation_max_samples"]
            ),
            "validation_game_seed_ranges": [],
            "validation_game_seed_count": int(held_out.size),
            "validation_row_count": int(selected_rows),
            "validation_game_seed_set_sha256": seed_set_digest,
            "game_seeds": held_out.tolist(),
        }
        held_out_set = set(map(int, held_out.tolist()))
        selected_records = [
            {**record, "split": "validation" if record["game_seed"] in held_out_set else "train"}
            for record in sorted(
                selected_game_records,
                key=lambda item: (int(item["game_seed"]), str(item["job_id"])),
            )
        ]
        selected_seed_array = np.asarray(
            [record["game_seed"] for record in selected_records], dtype="<i8"
        )
        training_seed_array = np.asarray(
            [
                record["game_seed"]
                for record in selected_records
                if record["split"] == "train"
            ],
            dtype="<i8",
        )
        selected_game_manifest = {
            "schema_version": "a1-selected-training-games-v1",
            "a1_contract_sha256": lock["contract_sha256"],
            "selection_rule": "lowest_seed_complete_per_job",
            "selected_game_count": len(selected_records),
            "selected_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(selected_seed_array.tobytes()).hexdigest(),
            "category_game_counts": dict(EXPECTED_GAMES),
            "training_game_count": int(training_seed_array.size),
            "training_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(training_seed_array.tobytes()).hexdigest(),
            "validation_game_count": int(held_out.size),
            "validation_game_seed_set_sha256": seed_set_digest,
            "records": selected_records,
            "records_sha256": _digest_value(selected_records),
        }
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "contract_path": str(lock_path.absolute()),
        "contract_sha256": lock["contract_sha256"],
        "passed": not errors,
        "errors": errors,
        "games": {category: len(seeds) for category, seeds in category_seeds.items()},
        "attempts": dict(EXPECTED_ATTEMPTS),
        "total_unique_games": len(all_seeds),
        "selection_rule": "lowest_seed_complete_per_job",
        "job_selections": job_selections,
        "job_selection_sha256": _digest_value(job_selections),
        "rows": rows,
        "invalid_teacher_actions": invalid_actions,
        "reports": {
            "truncation": {
                "selected_truncated_games": 0,
                "reserve_truncated_or_incomplete_attempts": sum(
                    int(item["truncated_attempts"]) for item in job_selections
                ),
            },
            "forced_fraction": forced / rows if rows else None,
            "phase_mix": dict(phases),
            "decision_index_mix": dict(decision_bins),
            "legal_width": dict(legal_widths),
            "target_entropy": target_entropy_sum / target_entropy_count
            if target_entropy_count
            else None,
            "full_search_policy_mass": full_active / rows if rows else None,
        },
        "shards": shard_records,
        "shard_inventory_sha256": _digest_value(shard_records),
        "source_provenance": {
            category: {
                "producer_checkpoint_sha256": producer["sha256"],
                "opponent_checkpoint_sha256": _category_opponent_sha256(
                    lock, category
                ),
                "search_operator_sha256": lock["science"]["search_operator_sha256"],
                "effective_search_config_sha256": lock["science"][
                    "effective_search_config_sha256"
                ],
                "evaluator_sha256": lock["science"]["evaluator_sha256"],
            }
            for category in EXPECTED_GAMES
        },
        "selected_training_games": (
            {
                "manifest": str(selected_game_manifest_path),
                "manifest_sha256": _digest_value(selected_game_manifest),
                "selected_game_count": selected_game_manifest[
                    "selected_game_count"
                ],
                "selected_game_seed_set_sha256": selected_game_manifest[
                    "selected_game_seed_set_sha256"
                ],
                "records_sha256": selected_game_manifest["records_sha256"],
            }
            if selected_game_manifest is not None
            else None
        ),
        "validation_holdout": (
            {
                "manifest": str(validation_seed_manifest_path),
                "manifest_sha256": _digest_value(validation_manifest),
                "validation_game_seed_count": validation_manifest[
                    "validation_game_seed_count"
                ],
                "validation_game_seed_set_sha256": validation_manifest[
                    "validation_game_seed_set_sha256"
                ],
            }
            if validation_manifest is not None
            else None
        ),
    }
    report["audit_sha256"] = _digest_value(report)
    if validation_manifest is not None:
        _create_readonly(validation_seed_manifest_path, validation_manifest)
        assert selected_game_manifest is not None
        _create_readonly(selected_game_manifest_path, selected_game_manifest)
        report["validation_holdout"]["manifest_file_sha256"] = _sha256(
            validation_seed_manifest_path
        )
        report["selected_training_games"]["manifest_file_sha256"] = _sha256(
            selected_game_manifest_path
        )
        # The on-disk file digest is added after writing; refresh the report
        # digest so the immutable report binds the exact sidecar bytes too.
        report["audit_sha256"] = _digest_value(
            {key: value for key, value in report.items() if key != "audit_sha256"}
        )
    _create_readonly(out_path.absolute(), report)
    if errors:
        raise ContractError(
            f"post-wave audit failed with {len(errors)} error(s); see {out_path}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser(
        "inspect-template", help="report unresolved fields; never seals"
    )
    inspect_parser.add_argument("--draft", required=True)
    sync_guard_parser = sub.add_parser(
        "sync-generation-guard",
        help=(
            "replay typed S1 and synchronize only the provenance-declared static "
            "generation guard; never seals or launches"
        ),
    )
    sync_guard_parser.add_argument("--draft", required=True)
    seal_parser = sub.add_parser(
        "seal", help="freeze a fully resolved draft exactly once"
    )
    seal_parser.add_argument("--draft", required=True)
    seal_parser.add_argument("--out", required=True)
    verify_parser = sub.add_parser(
        "verify", help="rehash and semantically verify a frozen lock"
    )
    verify_parser.add_argument("--lock", required=True)
    render_parser = sub.add_parser(
        "render", help="render immutable commands; never execute"
    )
    render_parser.add_argument("--lock", required=True)
    render_parser.add_argument("--out-dir", required=True)
    audit_parser = sub.add_parser(
        "audit", help="deep post-wave audit before corpus ingest"
    )
    audit_parser.add_argument("--lock", required=True)
    audit_parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-template":
            draft = _load_json(Path(args.draft))
            unresolved = _find_unresolved(draft)
            print(
                json.dumps(
                    {
                        "schema_version": draft.get("schema_version"),
                        "unresolved": unresolved,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "sync-generation-guard":
            print(
                json.dumps(
                    sync_generation_guard(Path(args.draft)), indent=2, sort_keys=True
                )
            )
            return 0
        if args.command == "seal":
            payload = build_lock(Path(args.draft))
            _create_readonly(Path(args.out).absolute(), payload)
            print(
                json.dumps(
                    {
                        "out": str(Path(args.out).absolute()),
                        "contract_sha256": payload["contract_sha256"],
                    }
                )
            )
            return 0
        if args.command == "verify":
            payload = verify_lock(Path(args.lock).absolute())
            print(
                json.dumps(
                    {"status": "PASS", "contract_sha256": payload["contract_sha256"]}
                )
            )
            return 0
        if args.command == "render":
            payload = render(Path(args.lock).absolute(), Path(args.out_dir))
            print(
                json.dumps(
                    {
                        "out": str(Path(args.out_dir).absolute()),
                        "jobs": len(payload["commands"]),
                    }
                )
            )
            return 0
        payload = audit_outputs(Path(args.lock).absolute(), Path(args.out).absolute())
        print(json.dumps({"status": "PASS", "audit_sha256": payload["audit_sha256"]}))
        return 0
    except ContractError as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
