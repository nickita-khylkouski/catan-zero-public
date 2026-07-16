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
import sys
from pathlib import Path
from typing import Any, Sequence

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


CANONICAL_TRAIN_LAUNCH_SCHEMA = 1
CANONICAL_CONFIG_SHA256 = (
    "e31e2fab6467530a1f057d5e0c05d28dc1b0c96d8e2d76fec477d0bd209559d3"
)
_ENGINE_SETTING_KEYS = frozenset(
    {
        "base_sampler",
        "checkpoint_steps",
        "entity_feature_adapter_version",
        "minimum_feature_learning_signal_observations",
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
        help="Optional parent checkpoint; omitted means the recipe's initialization.",
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
    payload_sha256 = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    if payload_sha256 != CANONICAL_CONFIG_SHA256:
        raise SystemExit(
            "canonical train config is not the exact commissioned payload: "
            f"expected_sha256={CANONICAL_CONFIG_SHA256} "
            f"actual_sha256={payload_sha256}"
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
    engine = payload.get("engine_settings", {})
    if not isinstance(engine, dict):
        raise SystemExit("canonical train config engine_settings must be an object")
    unknown = sorted(set(engine) - _ENGINE_SETTING_KEYS)
    if unknown:
        raise SystemExit(f"unknown canonical engine setting(s): {unknown}")
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


def main(argv: Sequence[str] | None = None) -> None:
    public_args = build_parser().parse_args(argv)
    config, engine_settings = _load_recipe(public_args.config)
    engine_args = _engine_namespace(
        config=config,
        engine_settings=engine_settings,
        public_args=public_args,
    )
    from tools import train_bc

    train_bc.main(engine_args)


if __name__ == "__main__":
    main()
