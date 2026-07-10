"""Additive typed-config CLI wiring (task CAT-66).

Gives every pipeline CLI three new, opt-in flags without changing any existing
behavior when they are unused:

    --config PATH         Load a typed config (canonical JSON, as written by
                          --dump-config) and use it to fill any flag the caller
                          left at its default. Explicitly-passed flags always
                          win, so this only supplies values the argv did not.
    --dump-config PATH    After building the resolved typed config, write its
                          canonical JSON to PATH and register it. Combine with a
                          normal run, or use it alone to materialize a config.
    --config-hash         Print the resolved config's hash to stderr and keep
                          running (does not change the run).
    --config-purpose STR  Free-text label recorded in the registry.

GUARANTEE: when none of these flags are passed, ``add_config_flags`` only adds
argument definitions (which cannot affect parsing of other flags) and
``resolve_config`` still builds+registers the config but makes ZERO observable
change to the run -- argv-only invocations are byte-identical to pre-CAT-66
behavior. ``tests/test_pipeline_configs.py`` proves this for train and generate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from catan_zero.rl import config_registry
from catan_zero.rl.pipeline_configs import (
    CONFIG_SCHEMA_VERSION,
    PipelineConfig,
    config_from_payload,
)


def add_config_flags(
    parser: argparse.ArgumentParser, *, default_purpose: str = ""
) -> None:
    """Add the four opt-in typed-config flags to ``parser`` (additive, no-op
    when unused)."""
    group = parser.add_argument_group("typed config / config-hash (CAT-66)")
    group.add_argument(
        "--config",
        default=None,
        help=(
            "Load a typed config (canonical JSON from --dump-config) and use it "
            "to fill any flag left at its default. Explicitly-passed flags win."
        ),
    )
    group.add_argument(
        "--dump-config",
        default=None,
        help="Write the fully-resolved typed config (canonical JSON) to this path and register it.",
    )
    group.add_argument(
        "--config-hash",
        dest="print_config_hash",
        action="store_true",
        help="Print the resolved config hash to stderr and continue.",
    )
    group.add_argument(
        "--config-purpose",
        default=default_purpose,
        help="Free-text purpose recorded alongside the config in the registry.",
    )


def _explicit_cli_dests(
    parser: argparse.ArgumentParser, argv: list[str] | tuple[str, ...]
) -> set[str]:
    """Return parser destinations explicitly present in ``argv``.

    Comparing a parsed value to its parser default cannot distinguish an omitted
    flag from an explicit flag whose value happens to equal that default.  That
    distinction matters when a config file supplies a different value: explicit
    command-line input must always win.
    """
    explicit: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)  # noqa: SLF001
        if action is not None:
            explicit.add(action.dest)
    return explicit


def _validated_payload(
    config_path: str | Path, *, expected_pipeline: str | None
) -> dict[str, Any]:
    """Load an executable config file and validate its envelope.

    ``config_from_payload`` is deliberately migration-friendly for offline
    registry reads. Launch-time config application has a stricter contract:
    wrong-pipeline and stale-schema payloads must never be interpreted as CLI
    values for a different executable.
    """
    payload = json.loads(Path(config_path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("config payload must be a JSON object")
    pipeline = payload.get("pipeline")
    if expected_pipeline is not None and pipeline != expected_pipeline:
        raise ValueError(
            f"config pipeline {pipeline!r} does not match expected "
            f"{expected_pipeline!r}"
        )
    schema = payload.get("schema_version")
    if schema != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"config schema_version {schema!r} does not match current "
            f"{CONFIG_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("fields"), dict):
        raise ValueError("config fields must be a JSON object")
    return payload


def _coerce_config_value(
    action: argparse.Action, value: Any, parser: argparse.ArgumentParser
) -> Any:
    """Apply an argparse action's type/choice contract to one JSON value."""
    option = action.option_strings[0] if action.option_strings else action.dest
    if isinstance(
        action,
        (
            argparse.BooleanOptionalAction,
            argparse._StoreTrueAction,  # noqa: SLF001
            argparse._StoreFalseAction,  # noqa: SLF001
        ),
    ):
        if not isinstance(value, bool):
            parser.error(f"config field {action.dest!r} for {option} must be boolean")
        return value

    def convert(item: Any) -> Any:
        converter = action.type
        if converter is None:
            default = action.default
            if isinstance(default, str) or default is None:
                if not isinstance(item, str):
                    parser.error(
                        f"config field {action.dest!r} for {option} must be a string"
                    )
                converted = item
            else:
                converted = item
        else:
            if converter is int and (
                isinstance(item, bool) or not isinstance(item, int)
            ):
                parser.error(
                    f"config field {action.dest!r} for {option} must be an integer"
                )
            if converter is float and (
                isinstance(item, bool) or not isinstance(item, (int, float))
            ):
                parser.error(
                    f"config field {action.dest!r} for {option} must be numeric"
                )
            try:
                converted = converter(item)
            except (TypeError, ValueError, argparse.ArgumentTypeError) as error:
                parser.error(
                    f"invalid config value for {option}: {item!r} ({error})"
                )
        if action.choices is not None and converted not in action.choices:
            parser.error(
                f"invalid config value for {option}: {converted!r}; "
                f"choose from {tuple(action.choices)!r}"
            )
        return converted

    if action.nargs in ("+", "*") or isinstance(action.nargs, int):
        if not isinstance(value, (list, tuple)):
            parser.error(f"config field {action.dest!r} for {option} must be a list")
        converted_items = [convert(item) for item in value]
        if action.nargs == "+" and not converted_items:
            parser.error(f"config field {action.dest!r} for {option} cannot be empty")
        if isinstance(action.nargs, int) and len(converted_items) != action.nargs:
            parser.error(
                f"config field {action.dest!r} for {option} requires "
                f"{action.nargs} values"
            )
        return converted_items
    return convert(value)


def apply_config_file(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    argv: list[str] | tuple[str, ...] | None = None,
    expected_pipeline: str | None = None,
) -> list[str]:
    """If ``--config`` was given, fill args left at their parser default from it.

    Returns the list of field names that were filled from the file (empty when
    ``--config`` was not passed). An explicitly-passed flag -- one whose value
    differs from the parser default -- is never overwritten, so the file only
    supplies values the argv omitted. Fields in the file with no matching CLI
    dest are ignored.
    """
    config_path = getattr(args, "config", None)
    if not config_path:
        return []
    try:
        payload = _validated_payload(
            config_path, expected_pipeline=expected_pipeline
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"invalid --config {config_path}: {error}")
    fields = dict(payload.get("fields", {}))
    filled: list[str] = []
    # Real CLI callers normally omit ``argv`` after ``parse_args()``. In that
    # case inspect the process command line so an explicit value equal to the
    # parser default still wins over the config file. Programmatic callers that
    # parsed a custom sequence should pass that same sequence explicitly.
    effective_argv = sys.argv[1:] if argv is None else argv
    explicit_dests = _explicit_cli_dests(parser, effective_argv)
    for name, value in fields.items():
        # ``format`` field is stored as ``fmt`` in GenerateConfig; map back.
        dest = "format" if name == "fmt" else name
        if not hasattr(args, dest) or dest in explicit_dests:
            continue
        default = parser.get_default(dest)
        # Optional dataclass fields serialize as JSON null. When the CLI also
        # defaults to None this represents omission, not a value to feed
        # through an argparse int/string converter.
        if value is None and default is None:
            continue
        if getattr(args, dest) == default:
            action = parser._option_string_actions.get(  # noqa: SLF001
                next(
                    (
                        option
                        for option, candidate in parser._option_string_actions.items()  # noqa: SLF001
                        if candidate.dest == dest
                    ),
                    "",
                )
            )
            if action is None:
                continue
            setattr(args, dest, _coerce_config_value(action, value, parser))
            filled.append(dest)
    return filled


def resolve_config(
    args: argparse.Namespace,
    build_config: Callable[[argparse.Namespace], PipelineConfig],
    *,
    parser: argparse.ArgumentParser | None = None,
    argv: list[str] | tuple[str, ...] | None = None,
    register: bool = True,
) -> PipelineConfig:
    """Build (and optionally register) the typed config, honoring the flags.

    Order: apply ``--config`` (if a parser is supplied) so file-supplied
    defaults are in place, build the typed config from the resolved namespace,
    register it (idempotent), then honor ``--dump-config`` / ``--config-hash``.
    Safe to call unconditionally: with no CAT-66 flag set it returns the config
    and registers it but makes no other change.
    """
    if parser is not None:
        apply_config_file(args, parser, argv=argv)
    config = build_config(args)
    if getattr(args, "config", None):
        try:
            _validated_payload(args.config, expected_pipeline=config.PIPELINE)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            if parser is not None:
                parser.error(f"invalid --config {args.config}: {error}")
            raise ValueError(f"invalid config {args.config}: {error}") from error
    config_hash = config.config_hash()

    if register:
        try:
            config_registry.register(
                config, purpose=getattr(args, "config_purpose", "") or ""
            )
        except OSError as exc:  # never let registry I/O sink a real run
            print(
                json.dumps(
                    {"progress": "config_registry_warning", "message": str(exc)},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    if getattr(args, "print_config_hash", False):
        print(
            json.dumps(
                {
                    "progress": "config_hash",
                    "pipeline": config.PIPELINE,
                    "config_hash": config_hash,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    dump_path = getattr(args, "dump_config", None)
    if dump_path:
        out = Path(dump_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(config.canonical_payload(), indent=2, sort_keys=True) + "\n"
        )

    return config


def load_config(path: str | Path) -> PipelineConfig:
    """Reconstruct a typed config from a canonical-JSON file written by --dump-config."""
    payload = json.loads(Path(path).read_text())
    return config_from_payload(payload)
