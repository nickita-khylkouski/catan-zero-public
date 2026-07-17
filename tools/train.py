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
CANONICAL_CONFIG_ROLES_BY_CATALOG_NAME = {
    "a1-current-35m-b200": "scratch_fresh_optimizer",
    "a1-parent-update-35m-b200": "parent_fresh_optimizer",
}
_ENGINE_SETTING_KEYS = frozenset(
    {
        "base_sampler",
        "checkpoint_steps",
        "data_loader_prefetch",
        "data_loader_workers",
        "entity_feature_adapter_version",
        "initialization_mode",
        "minimum_feature_learning_signal_observations",
        "minimum_initial_road_policy_mass_fraction",
        "minimum_initial_settlement_policy_mass_fraction",
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


_OPENING_POLICY_MASS_MINIMUM_KEYS = (
    "minimum_initial_settlement_policy_mass_fraction",
    "minimum_initial_road_policy_mass_fraction",
)


def _require_production_opening_policy_mass_contract(
    engine_settings: Mapping[str, Any],
) -> dict[str, float]:
    """Refuse a production learner until both opening minima are reviewed.

    The checked-in science admission currently names these values as unresolved,
    so there is intentionally no fallback threshold. Once commissioned, values
    live in ``engine_settings``: the production catalog hashes those recipe
    semantics and train_bc additionally binds them into resume identity.
    """

    missing = [
        key
        for key in _OPENING_POLICY_MASS_MINIMUM_KEYS
        if key not in engine_settings
    ]
    if missing:
        raise SystemExit(
            "production training remains fail-closed until reviewed opening "
            "policy-mass minima are commissioned for both initial settlement "
            f"and initial road; missing={missing}"
        )
    minima: dict[str, float] = {}
    for key in _OPENING_POLICY_MASS_MINIMUM_KEYS:
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
        raise SystemExit("canonical opening policy-mass minima cannot sum above one")
    return minima


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
            "Exact learner initializer. For a legacy incumbent this is the "
            "function-preserving upgraded checkpoint, not the incumbent bytes."
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
        "--architecture-upgrade-receipt",
        default="",
        help=(
            "Reviewed zero-diff receipt connecting --parent-checkpoint to "
            "--init-checkpoint. Required when their bytes differ."
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
    """Materialize engine defaults without invoking its legacy CLI parser.

    ``train_bc`` still owns parser actions temporarily because historical
    receipts import them, but canonical launches must not translate a typed
    recipe back into command-line text.  Copying the action defaults produces
    the same starting namespace as ``parse_args`` while keeping the handoff
    entirely in process.
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
    args = _engine_default_namespace(internal_parser)
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
    if public_args.init_checkpoint:
        settings["init_checkpoint"] = public_args.init_checkpoint

    unsupported: list[str] = []
    for name, value in settings.items():
        action = actions.get(name)
        if action is None:
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


def _checkpoint_ref(raw: str, *, where: str) -> dict[str, str]:
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as error:
        raise SystemExit(f"cannot resolve {where}: {error}") from error
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{where} must be a regular non-symlink file: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


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
    receipt_raw = str(public_args.architecture_upgrade_receipt or "")
    if parent["sha256"] == initializer["sha256"]:
        if receipt_raw:
            raise SystemExit(
                "exact-parent initialization must not claim an architecture receipt"
            )
        return {
            "schema_version": "a1-canonical-parent-initializer-v1",
            "mode": "exact_parent",
            "parent": parent,
            "initializer": initializer,
            "function_preserving_upgrade": None,
        }
    if not receipt_raw:
        raise SystemExit(
            "initializer bytes differ from the incumbent; "
            "--architecture-upgrade-receipt is required"
        )
    from tools import a1_function_preserving_upgrade as upgrade

    try:
        replayed = upgrade.verify_receipt(Path(receipt_raw))
    except (OSError, upgrade.UpgradeError) as error:
        raise SystemExit(f"architecture upgrade receipt refused: {error}") from error
    if (
        replayed.get("module")
        != upgrade.MODULE_CURRENT_V5_TOPOLOGY_VALUE_TOWER_SPLIT_1
        or replayed.get("source") != parent
        or replayed.get("upgraded_initializer") != initializer
    ):
        raise SystemExit(
            "architecture receipt must connect the exact incumbent directly to "
            "the reviewed current-v5+topology+split1 initializer"
        )
    receipt = replayed["receipt"]
    lineage_binding = {
        "schema_version": "a1-lineage-function-preserving-upgrade-v1",
        "module": replayed["module"],
        "receipt": receipt["path"],
        "receipt_sha256": receipt["sha256"],
        "source_checkpoint_sha256": parent["sha256"],
        "upgraded_initializer_sha256": initializer["sha256"],
    }
    return {
        "schema_version": "a1-canonical-parent-initializer-v1",
        "mode": "function_preserving_upgrade",
        "parent": parent,
        "initializer": initializer,
        "function_preserving_upgrade": lineage_binding,
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
            function_preserving_upgrade=initialization[
                "function_preserving_upgrade"
            ],
        )
    except lineage.LineageDoseError as error:
        raise SystemExit(f"canonical parent lineage refused: {error}") from error
    payload["a1_lineage_dose"] = dose
    payload["a1_parent_update_initialization"] = dict(initialization)
    payload["promotion_eligible"] = False
    payload["promotion_block_reason"] = (
        "requires_sealed_a1_one_dose_execution_receipt"
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
    config, engine_settings = _load_recipe(public_args.config)
    if engine_settings.get("initialization_mode") == "scratch_fresh_optimizer":
        raise SystemExit(
            "tools/train.py is not launch authority for the checked-in "
            "scratch_fresh_optimizer recipe. Use tools/a1_scratch_train.py "
            "with --lock, --data, --composite-build-receipt, --checkpoint, "
            "--report, and --receipt to create the authenticated plan; rerun "
            "that same command with --go only after the plan is commissioned."
        )
    _require_production_opening_policy_mass_contract(engine_settings)
    initialization = _parent_initializer_binding(public_args)
    engine_args = _engine_namespace(
        config=config,
        engine_settings=engine_settings,
        public_args=public_args,
    )
    from tools import train_bc

    train_bc.main(engine_args)
    # Under torchrun every rank executes this wrapper. train_bc owns report
    # emission on rank zero, so only rank zero may seal the post-run binding.
    if int(os.environ.get("RANK", "0")) == 0:
        _bind_parent_report(public_args.report, initialization=initialization)


if __name__ == "__main__":
    main()
