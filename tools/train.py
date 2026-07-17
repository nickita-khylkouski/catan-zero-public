#!/usr/bin/env python3
"""Canonical config-first learner launcher.

This is the supported human/orchestrator entrypoint for new training runs.
``train_bc.py`` remains the internal engine while historical sealed launch
receipts are migrated; callers must not construct its experimental CLI.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = _REPO_ROOT / "src"
for import_root in (_REPO_ROOT, _REPO_SRC):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from catan_zero.rl.pipeline_configs import (  # noqa: E402
    CONFIG_SCHEMA_VERSION,
    TrainConfig,
    config_from_payload,
)
from catan_zero.rl.production_recipe_catalog import (  # noqa: E402
    ProductionRecipeError,
    require_production_recipe,
)


CANONICAL_TRAIN_LAUNCH_SCHEMA = 1
REQUIRED_NOFILE_SOFT = 65_536
CANONICAL_CONFIG_ROLES_BY_CATALOG_NAME = {
    "a1-current-35m-b200": "scratch_fresh_optimizer",
    "a1-parent-update-35m-b200": "parent_fresh_optimizer",
    "a1-parent-update-active-p25-35m-b200": "parent_fresh_optimizer",
}
_ENGINE_SETTING_KEYS = frozenset(
    {
        "base_sampler",
        "checkpoint_steps",
        "data_loader_prefetch",
        "data_loader_workers",
        "entity_feature_adapter_version",
        "initialization_mode",
        "minimum_discard_policy_mass_fraction",
        "minimum_feature_learning_signal_observations",
        "minimum_initial_road_policy_mass_fraction",
        "minimum_initial_settlement_policy_mass_fraction",
        "minimum_move_robber_policy_mass_fraction",
        "objective_gradient_interference_every_batches",
        "public_rule_state_features",
        "require_35m_model",
        "require_feature_learning_signal_modules",
        "require_production_35m_teacher",
        "required_target_information_regime",
        "scalar_value_loss_readout",
        "scalar_value_loss_scale",
        "scalar_value_objective",
        "train_diagnostics_every_batches",
        "skip_teacher_quality_gate",
        "trust_curated_data_quality",
        "value_tower_split_layers",
        "min_35m_params",
        "max_35m_params",
    }
)


# Operational/authority fields intentionally excluded from TrainConfig and the
# checked-in engine_settings envelope. Canonical launches must not inherit
# these values by copying the internal experimental parser: doing so makes a
# newly added train_bc flag silently become production behavior. Keep this
# small baseline explicit and fail closed below when the internal engine grows
# a field that is not bound by the recipe, public launcher, or this mapping.
_CANONICAL_RUNTIME_DEFAULTS: dict[str, Any] = {
    "acknowledge_authoritative_hard_action_targets": False,
    "validation_game_seed_manifest": "",
    "validation_game_sentinel_manifest": "",
    "accepted_policy_target_identity_sha256": [],
    "require_only_trainable_prefixes": "",
    "allow_teacher_score_q_loss": False,
    "allow_legacy_action_mask_upgrade": False,
    "acknowledge_diagnostic_outcome_conditioned_policy_distillation": False,
    "a1_learner_ablation_id": "",
    "a1_scratch_authority_json": "",
    "a1_canonical_parent_update_authority_json": "",
    "a1_scratch_diagnostic_authority_json": "",
    "a1_effective_learner_recipe_json": "",
    "a1_effective_learner_recipe_sha256": "",
    "a1_ablation_code_binding_json": "",
    "a1_ablation_code_tree_sha256": "",
    "a1_reviewed_lock_file_sha256": "",
    "a1_aux_regularization_binding_json": "",
    "a1_central_learner_binding_json": "",
    "a1_coherent_corpus_binding_json": "",
    "a1_central_executor_authority": "",
    "a1_central_executor_authority_sha256": "",
    "a1_aux_stage_binding_json": "",
    "a1_aux_stage_executor_authority": "",
    "a1_aux_stage_executor_authority_sha256": "",
    "a1_dual_learner_lock": "",
    "a1_dual_reviewed_lock_file_sha256": "",
    "a1_curriculum_parent_receipt": "",
    "a1_batch_probe_plan": "",
    "a1_batch_probe_run_id": "",
    "save_each_epoch": False,
    "progress_every_batches": 50,
    "ddp_find_unused_parameters": False,
    "float32_matmul_precision": None,
    "require_strict_35m_teacher": False,
    "minimum_maritime_trade_policy_objective_mass_fraction": None,
    "skip_guards": False,
    "config": None,
    "dump_config": None,
    "print_config_hash": False,
    "config_purpose": "train_bc",
}


_HARD_DECISION_POLICY_MASS_MINIMUM_KEYS = (
    "minimum_initial_settlement_policy_mass_fraction",
    "minimum_initial_road_policy_mass_fraction",
    "minimum_discard_policy_mass_fraction",
    "minimum_move_robber_policy_mass_fraction",
)


def _require_production_hard_decision_policy_mass_contract(
    engine_settings: Mapping[str, Any],
) -> dict[str, float]:
    """Refuse production unless every commissioned hard phase retains signal.

    These are realized policy-objective fractions, not raw row-count floors.
    Values live in ``engine_settings`` so the production catalog hashes the
    recipe semantics and train_bc additionally binds them into resume identity.
    """

    missing = [
        key
        for key in _HARD_DECISION_POLICY_MASS_MINIMUM_KEYS
        if key not in engine_settings
    ]
    if missing:
        raise SystemExit(
            "production training remains fail-closed until reviewed hard-decision "
            "policy-mass minima are commissioned for initial settlement, initial "
            f"road, discard, and robber movement; missing={missing}"
        )
    minima: dict[str, float] = {}
    for key in _HARD_DECISION_POLICY_MASS_MINIMUM_KEYS:
        raw = engine_settings[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise SystemExit(f"canonical engine setting {key} must be numeric")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise SystemExit(
                f"canonical engine setting {key} must be a finite fraction in (0, 1]"
            )
        minima[key] = value
    if sum(minima.values()) > 1.0:
        raise SystemExit(
            "canonical hard-decision policy-mass minima cannot sum above one"
        )
    return minima


def _require_exact_cap_feature_observability(
    config: TrainConfig,
    engine_settings: Mapping[str, Any],
) -> None:
    """Reject an exact dose that cannot produce its required observations."""

    if not bool(config.exact_max_steps) or int(config.max_steps) <= 0:
        return
    required_modules = tuple(
        name.strip()
        for name in str(
            engine_settings.get("require_feature_learning_signal_modules", "")
        ).split(",")
        if name.strip()
    )
    if not required_modules:
        return
    minimum = engine_settings.get("minimum_feature_learning_signal_observations", 0)
    cadence = engine_settings.get("train_diagnostics_every_batches", 0)
    for name, value in (
        ("minimum_feature_learning_signal_observations", minimum),
        ("train_diagnostics_every_batches", cadence),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SystemExit(f"canonical engine setting {name} must be an integer >= 0")
    if minimum == 0:
        return

    # max_steps counts applied optimizer updates. Diagnostics are dose-global
    # micro-batch events, but a retained observation must also close an
    # accumulation group. The eligible indices are therefore the common
    # multiples of cadence and grad_accum_steps, not every cadence hit.
    accumulation = int(config.grad_accum_steps)
    eligible_period = math.lcm(accumulation, cadence) if cadence else 0
    maximum_observations = (
        0
        if cadence == 0
        else int(config.max_steps) * accumulation // eligible_period
    )
    if maximum_observations < minimum:
        raise SystemExit(
            "canonical exact-capped training cannot satisfy feature-learning-signal "
            "observability before GPU launch: "
            f"max_steps={config.max_steps} "
            f"grad_accum_steps={config.grad_accum_steps} "
            f"train_diagnostics_every_batches={cadence} "
            f"maximum_feature_learning_signal_observations={maximum_observations} "
            f"minimum_feature_learning_signal_observations={minimum}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the canonical Catan model from one checked-in recipe. "
            "Science and architecture knobs belong in the recipe, not the CLI."
        )
    )
    parser.add_argument("--config", required=True, help="Canonical train recipe JSON.")
    parser.add_argument("--data", required=True, help="Authenticated corpus or composite.")
    parser.add_argument("--checkpoint", required=True, help="Terminal checkpoint path.")
    parser.add_argument("--report", required=True, help="Training report path.")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help=(
            "Exact learner initializer. The selected v6 parent treatment is a "
            "measured non-promotable V2->V6 information-contract migration, "
            "not a function-preserving upgrade."
        ),
    )
    parser.add_argument(
        "--parent-checkpoint",
        default="",
        help=(
            "Exact incumbent checkpoint being updated. Parent-update runs must "
            "bind this separately from the architecture initializer."
        ),
    )
    parser.add_argument(
        "--information-contract-migration-receipt",
        default="",
        help=(
            "Reviewed non-promotable information-contract migration connecting "
            "--parent-checkpoint to --init-checkpoint. Required for the v6 treatment."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--host-lock-file",
        default="/tmp/catan_zero_train_bc.lock",
        help="Host-local single-learner lock.",
    )
    parser.add_argument(
        "--allow-concurrent-bc",
        action="store_true",
        help="Permit multiple learners on one host when a scheduler isolates them.",
    )
    return parser


def _load_recipe(path: str | Path) -> tuple[TrainConfig, dict[str, Any]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid canonical train config {source}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit("canonical train config must be a JSON object")
    try:
        recipe_name = require_production_recipe(
            entrypoint="train", path=source, payload=payload
        )
    except ProductionRecipeError as error:
        raise SystemExit(str(error)) from error
    engine = payload.get("engine_settings")
    if not isinstance(engine, dict):
        raise SystemExit("canonical train config engine_settings must be an object")
    expected_role = CANONICAL_CONFIG_ROLES_BY_CATALOG_NAME.get(recipe_name)
    if expected_role is None:
        raise SystemExit(
            "production train recipe lacks an initialization-role binding: "
            f"catalog_name={recipe_name!r}"
        )
    recipe_role = str(engine.get("initialization_mode", "") or "")
    if recipe_role != expected_role:
        raise SystemExit(
            "canonical train config role does not match its commissioned payload: "
            f"expected_role={expected_role!r} actual_role={recipe_role!r}"
        )
    if payload.get("launcher_schema") != CANONICAL_TRAIN_LAUNCH_SCHEMA:
        raise SystemExit(
            "canonical train config launcher_schema must be "
            f"{CANONICAL_TRAIN_LAUNCH_SCHEMA}"
        )
    train_payload = payload.get("train_config")
    if not isinstance(train_payload, dict):
        raise SystemExit("canonical train config requires train_config object")
    if train_payload.get("pipeline") != TrainConfig.PIPELINE:
        raise SystemExit("canonical train config train_config.pipeline must be 'train'")
    if train_payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(
            "canonical train config schema is stale: "
            f"expected {CONFIG_SCHEMA_VERSION}, got "
            f"{train_payload.get('schema_version')!r}"
        )
    fields = train_payload.get("fields")
    if not isinstance(fields, dict):
        raise SystemExit("canonical train config train_config.fields must be an object")
    expected_fields = {field.name for field in dataclasses.fields(TrainConfig)}
    missing_fields = sorted(expected_fields - set(fields))
    unknown_fields = sorted(set(fields) - expected_fields)
    if missing_fields or unknown_fields:
        raise SystemExit(
            "canonical train config must explicitly bind every TrainConfig field; "
            f"missing={missing_fields} unknown={unknown_fields}"
        )
    config = config_from_payload(train_payload)
    if not isinstance(config, TrainConfig):
        raise SystemExit("canonical train config did not decode as TrainConfig")
    unknown = sorted(set(engine) - _ENGINE_SETTING_KEYS)
    if unknown:
        raise SystemExit(f"unknown canonical engine setting(s): {unknown}")
    for name, minimum in (
        ("data_loader_workers", 0),
        ("data_loader_prefetch", 1),
    ):
        if name not in engine:
            continue
        value = engine[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise SystemExit(
                f"canonical engine setting {name} must be an integer >= {minimum}"
            )
    return config, dict(engine)


def _preferred_option(action: argparse.Action, *, negative: bool = False) -> str:
    candidates = [
        option
        for option in action.option_strings
        if option.startswith("--") and option.startswith("--no-") is negative
    ]
    if not candidates:
        candidates = [option for option in action.option_strings if option.startswith("--")]
    if not candidates:
        raise SystemExit(f"internal trainer setting {action.dest!r} has no long option")
    return candidates[0]


def _encode_setting(action: argparse.Action, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(action, argparse.BooleanOptionalAction):
        return [_preferred_option(action, negative=not bool(value))]
    if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
        return [_preferred_option(action)] if bool(value) else []
    if isinstance(action, argparse._StoreFalseAction):  # noqa: SLF001
        return [_preferred_option(action)] if not bool(value) else []
    if value == "" and action.default == "":
        return []
    option = _preferred_option(action)
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        return [option, *[str(item) for item in value]]
    return [option, str(value)]


def _engine_default_namespace(
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Materialize legacy diagnostic defaults without parsing command text.

    Historical R&D helpers still import this compatibility function. Canonical
    production launches do not: :func:`_engine_namespace` starts from the
    explicit runtime baseline above and obtains all science values from the
    typed recipe.
    """

    values: dict[str, Any] = {}
    for action in parser._actions:  # noqa: SLF001
        if action.dest == argparse.SUPPRESS or action.default is argparse.SUPPRESS:
            continue
        values[action.dest] = copy.deepcopy(action.default)
    return argparse.Namespace(**values)


def _engine_namespace(
    *,
    config: TrainConfig,
    engine_settings: dict[str, Any],
    public_args: argparse.Namespace,
) -> argparse.Namespace:
    # Importing the large engine is intentionally delayed until after the
    # compact public CLI and recipe envelope have been validated.
    from tools import train_bc

    internal_parser = train_bc.build_parser()
    actions = {
        action.dest: action
        for action in internal_parser._actions  # noqa: SLF001
        if action.option_strings
    }
    args = argparse.Namespace(**copy.deepcopy(_CANONICAL_RUNTIME_DEFAULTS))
    settings = dict(config.field_values())
    engine_settings = dict(engine_settings)
    initialization_mode = str(
        engine_settings.pop("initialization_mode", "") or ""
    )
    if initialization_mode not in {
        "scratch_fresh_optimizer",
        "parent_fresh_optimizer",
    }:
        raise SystemExit(
            "canonical train recipe must bind initialization_mode to "
            "scratch_fresh_optimizer or parent_fresh_optimizer"
        )
    requested_parent = str(public_args.init_checkpoint or config.init_checkpoint or "")
    requested_growth = str(config.grow_from_checkpoint or "")
    if initialization_mode == "scratch_fresh_optimizer":
        if requested_parent or requested_growth or bool(config.resume_optimizer):
            raise SystemExit(
                "scratch_fresh_optimizer recipe forbids parent/grow checkpoints "
                "and optimizer resume; use a distinct parent recipe instead of "
                "silently changing the experiment initializer"
            )
    elif not requested_parent:
        raise SystemExit(
            "parent_fresh_optimizer recipe requires --init-checkpoint (or a "
            "recipe-bound init_checkpoint)"
        )
    elif requested_growth or bool(config.resume_optimizer):
        raise SystemExit(
            "parent_fresh_optimizer requires an exact parent checkpoint and fresh "
            "optimizer; grow-from and optimizer resume are different experiments"
        )
    settings.update(engine_settings)
    settings.update(
        {
            "data": public_args.data,
            "checkpoint": public_args.checkpoint,
            "report": public_args.report,
            "device": public_args.device,
            "host_lock_file": public_args.host_lock_file,
            "allow_concurrent_bc": bool(public_args.allow_concurrent_bc),
        }
    )
    validation_manifest = _validation_manifest_from_memmap(public_args.data)
    if validation_manifest:
        settings["validation_game_seed_manifest"] = validation_manifest
    if public_args.init_checkpoint:
        settings["init_checkpoint"] = public_args.init_checkpoint

    unsupported: list[str] = []
    for name, value in settings.items():
        action = actions.get(name)
        if action is None:
            # The V8 exact-resource residual is checkpoint-owned architecture,
            # deliberately not a public train.py knob.  Still materialize the
            # typed recipe value on the internal namespace so train_bc can
            # verify it against the initializer and record the effective
            # topology in its immutable TrainConfig.
            if name == "public_card_exact_resource_residual":
                setattr(args, name, value)
                continue
            # Derived identities (corpus/checkpoint hashes, effective holdout
            # hashes) are filled by the engine after authentication.
            if name in {
                "data_fingerprint",
                "grow_from_checkpoint_sha256",
                "init_checkpoint_sha256",
                "meaningful_public_history_schema",
                "public_card_count_feature_schema",
                "training_excluded_game_seed_set_sha256",
                "validation_contract_file_sha256",
                "validation_game_seed_set_sha256",
            }:
                continue
            unsupported.append(name)
            continue
        if value is None and action.default is None:
            # ``None`` is an explicit typed sentinel for several canonical
            # settings (for example categorical value and adaptive KL). Bind it
            # into the namespace instead of relying on the parser to supply it.
            setattr(args, name, None)
            continue
        # Reuse the internal parser's exact type/choice contract without
        # reconstructing a giant synthetic command line.
        from catan_zero.rl.config_cli import _coerce_config_value

        coerced = _coerce_config_value(action, value, internal_parser)
        setattr(args, name, coerced)
    if unsupported:
        raise SystemExit(
            "canonical TrainConfig contains settings the internal trainer does "
            f"not expose: {sorted(unsupported)}"
        )
    missing = sorted(set(actions) - {"help"} - set(vars(args)))
    if missing:
        raise SystemExit(
            "canonical train launch leaves internal engine settings unbound; "
            "bind them in TrainConfig, engine_settings, the public launcher, or "
            f"the explicit runtime baseline: {missing}"
        )
    guard_dests = (
        "optimizer",
        "weight_decay",
        "truncated_vp_margin_value_weight",
        "lr_schedule",
        "mask_hidden_info",
    )
    guard_argv: list[str] = [
        "--data",
        str(args.data),
        "--checkpoint",
        str(args.checkpoint),
        "--report",
        str(args.report),
    ]
    for name in guard_dests:
        guard_argv.extend(_encode_setting(actions[name], getattr(args, name)))
    args._canonical_guard_argv = tuple(guard_argv)
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _validation_manifest_from_memmap(raw_data: str | Path) -> str:
    """Bind the exact validation split already authenticated by a memmap.

    The corpus builder records the immutable validation-only manifest and its
    file digest under the audited holdout in ``corpus_meta.json``. Canonical
    training must consume that exact sidecar; the broader selected-game
    envelope is a different schema and cannot substitute for it. Leaving the
    internal flag blank makes every valid A1 corpus fail preflight, while
    recomputing a fractional split would change the experiment.
    Non-directory/composite inputs retain their existing routing.
    """

    raw_path = Path(raw_data).expanduser()
    if not raw_path.is_dir():
        return ""
    meta_path = raw_path / "corpus_meta.json"
    if not meta_path.exists():
        return ""
    if meta_path.is_symlink() or not meta_path.is_file():
        raise SystemExit(f"memmap corpus metadata must be a regular file: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid memmap corpus metadata {meta_path}: {error}") from error
    audit = meta.get("a1_post_wave_audit")
    if audit is None:
        return ""
    if not isinstance(audit, Mapping):
        raise SystemExit("memmap a1_post_wave_audit must be an object")
    validation = audit.get("validation_holdout")
    if not isinstance(validation, Mapping):
        raise SystemExit("memmap audit lacks an authenticated validation holdout")
    path_raw = validation.get("path")
    digest = validation.get("file_sha256")
    if not isinstance(path_raw, str) or not path_raw:
        raise SystemExit("memmap selected-game manifest path is missing")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise SystemExit("memmap selected-game manifest SHA256 is missing")
    candidate = Path(path_raw).expanduser()
    if not candidate.is_absolute():
        candidate = meta_path.parent / candidate
    if candidate.is_symlink():
        raise SystemExit(
            f"memmap selected-game manifest must not be a symlink: {candidate}"
        )
    try:
        manifest = candidate.resolve(strict=True)
    except OSError as error:
        raise SystemExit(
            f"cannot resolve memmap selected-game manifest: {error}"
        ) from error
    if not manifest.is_file():
        raise SystemExit(
            f"memmap selected-game manifest must be a regular file: {manifest}"
        )
    actual = _sha256(manifest)
    if actual != digest:
        raise SystemExit(
            "memmap selected-game manifest digest mismatch: "
            f"declared={digest} actual={actual}"
        )
    return str(manifest)


def _checkpoint_ref(raw: str, *, where: str) -> dict[str, str]:
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as error:
        raise SystemExit(f"cannot resolve {where}: {error}") from error
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{where} must be a regular non-symlink file: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _same_checkpoint_bytes(left: Any, right: Any) -> bool:
    """Compare immutable checkpoint identity without binding deployment paths."""

    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and isinstance(left.get("sha256"), str)
        and left["sha256"] == right.get("sha256")
    )


def _parent_initializer_binding(
    public_args: argparse.Namespace,
) -> dict[str, Any]:
    """Replay the exact incumbent -> initializer edge before optimizer launch."""

    if not public_args.parent_checkpoint:
        raise SystemExit(
            "parent_fresh_optimizer requires --parent-checkpoint separately from "
            "--init-checkpoint"
        )
    if not public_args.init_checkpoint:
        raise SystemExit("parent_fresh_optimizer requires --init-checkpoint")
    parent = _checkpoint_ref(public_args.parent_checkpoint, where="learner parent")
    initializer = _checkpoint_ref(
        public_args.init_checkpoint, where="learner initializer"
    )
    receipt_raw = str(public_args.information_contract_migration_receipt or "")
    if parent["sha256"] == initializer["sha256"]:
        if receipt_raw:
            raise SystemExit(
                "exact-parent initialization must not claim a migration receipt"
            )
        return {
            "schema_version": "a1-canonical-parent-initializer-v1",
            "mode": "exact_parent",
            "parent": parent,
            "initializer": initializer,
            "information_contract_migration": None,
        }
    if not receipt_raw:
        raise SystemExit(
            "initializer bytes differ from the incumbent; "
            "--information-contract-migration-receipt is required"
        )
    from tools import a1_information_contract_migration as migration

    try:
        replayed = migration.verify_receipt(Path(receipt_raw))
    except (OSError, migration.MigrationError) as error:
        raise SystemExit(f"information migration receipt refused: {error}") from error
    migration_kind = replayed.get("migration")
    expected_forward_identity = {
        migration.MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1: False,
        migration.MIGRATION_CURRENT_V2_TO_V6_TOPOLOGY_SPLIT1_PUBLIC_RESOURCE_V8: False,
        migration.MIGRATION_V5_TO_V7_INPUT_COMPATIBILITY: True,
        migration.MIGRATION_V5_TO_V8_PUBLIC_RESOURCE_COMPATIBILITY: True,
    }.get(migration_kind)
    if (
        expected_forward_identity is None
        or not _same_checkpoint_bytes(replayed.get("source"), parent)
        or not _same_checkpoint_bytes(
            replayed.get("migrated_initializer"), initializer
        )
        or replayed.get("forward_identical") is not expected_forward_identity
        or replayed.get("promotion_eligible") is not False
    ):
        raise SystemExit(
            "migration receipt must connect the exact incumbent directly to an "
            "allowlisted non-promotable architecture treatment"
        )
    receipt = replayed["receipt"]
    lineage_binding = {
        "schema_version": "a1-lineage-information-contract-migration-v1",
        "migration": migration_kind,
        "receipt": receipt["path"],
        "receipt_sha256": receipt["sha256"],
        "source_checkpoint_sha256": parent["sha256"],
        "migrated_initializer_sha256": initializer["sha256"],
        "forward_identical": expected_forward_identity,
        "promotion_eligible": False,
    }
    return {
        "schema_version": "a1-canonical-parent-initializer-v1",
        "mode": "information_contract_migration",
        "parent": parent,
        "initializer": initializer,
        "information_contract_migration": lineage_binding,
    }


def _bind_parent_report(
    report_path: str | Path,
    *,
    initialization: Mapping[str, Any],
) -> None:
    """Stamp diagnostic canonical runs with exact lineage, never eligibility.

    Promotion requires the sealed receipt emitted by ``a1_one_dose_train.py``;
    this report binding makes standalone commissioning scientifically legible
    without creating a second promotion receipt format.
    """

    from tools import a1_lineage_dose as lineage

    path = Path(report_path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot bind canonical parent report: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit("canonical parent report must be a JSON object")
    steps = payload.get("steps_completed")
    sampled_rows = payload.get("base_training_row_draws")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
        or isinstance(sampled_rows, bool)
        or not isinstance(sampled_rows, int)
        or sampled_rows <= 0
    ):
        raise SystemExit(
            "canonical parent report lacks exact optimizer-step/base-draw counters"
        )
    try:
        dose = lineage.direct_lineage_dose(
            declared_producer_sha256=initialization["parent"]["sha256"],
            init_checkpoint_sha256=initialization["initializer"]["sha256"],
            current_sampled_rows=sampled_rows,
            current_optimizer_steps=steps,
            information_contract_migration=initialization.get(
                "information_contract_migration"
            ),
        )
    except lineage.LineageDoseError as error:
        raise SystemExit(f"canonical parent lineage refused: {error}") from error
    payload["a1_lineage_dose"] = dose
    payload["a1_parent_update_initialization"] = dict(initialization)
    payload["promotion_eligible"] = False
    payload["promotion_block_reason"] = (
        "information_contract_migration_uncommissioned"
        if initialization.get("mode") == "information_contract_migration"
        else "requires_sealed_a1_one_dose_execution_receipt"
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    public_args = build_parser().parse_args(argv)
    _ensure_runtime_limits()
    config, engine_settings = _load_recipe(public_args.config)
    _require_exact_cap_feature_observability(config, engine_settings)
    if engine_settings.get("initialization_mode") == "scratch_fresh_optimizer":
        raise SystemExit(
            "tools/train.py is not launch authority for the checked-in "
            "scratch_fresh_optimizer recipe. Use tools/a1_scratch_train.py "
            "with --lock, --data, --composite-build-receipt, --checkpoint, "
            "--report, and --receipt to create the authenticated plan; rerun "
            "that same command with --go only after the plan is commissioned."
        )
    _require_production_hard_decision_policy_mass_contract(engine_settings)
    initialization = _parent_initializer_binding(public_args)
    engine_args = _engine_namespace(
        config=config,
        engine_settings=engine_settings,
        public_args=public_args,
    )
    # train_bc independently replays the sealed corpus contract. Preserve the
    # already-verified parent -> migrated initializer edge so that replay can
    # distinguish a function-preserving architecture expansion from candidate
    # chaining. This is an internal attribute, not a public bypass flag.
    engine_args.a1_parent_update_initialization = initialization
    config_path = Path(str(public_args.config)).expanduser().resolve(strict=True)
    engine_args.a1_canonical_parent_update_authority = {
        "schema_version": "a1-canonical-parent-update-runtime-authority-v1",
        "config": str(config_path),
        "config_file_sha256": _sha256(config_path),
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    from tools import train_bc

    train_bc.main(engine_args)
    # Under torchrun every rank executes this wrapper. train_bc owns report
    # emission on rank zero, so only rank zero may seal the post-run binding.
    if int(os.environ.get("RANK", "0")) == 0:
        _bind_parent_report(public_args.report, initialization=initialization)


def _ensure_runtime_limits() -> None:
    """Make the canonical learner satisfy its own FD admission contract."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    unlimited = resource.RLIM_INFINITY
    if hard != unlimited and hard < REQUIRED_NOFILE_SOFT:
        raise SystemExit(
            f"hard RLIMIT_NOFILE {hard} is below required {REQUIRED_NOFILE_SOFT}"
        )
    if soft != unlimited and soft < REQUIRED_NOFILE_SOFT:
        resource.setrlimit(resource.RLIMIT_NOFILE, (REQUIRED_NOFILE_SOFT, hard))


if __name__ == "__main__":
    main()
