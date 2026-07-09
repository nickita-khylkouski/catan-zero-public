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
from catan_zero.rl.pipeline_configs import PipelineConfig, config_from_payload


def add_config_flags(parser: argparse.ArgumentParser, *, default_purpose: str = "") -> None:
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


def apply_config_file(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
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
    payload = json.loads(Path(config_path).read_text())
    fields = dict(payload.get("fields", {}))
    filled: list[str] = []
    for name, value in fields.items():
        # ``format`` field is stored as ``fmt`` in GenerateConfig; map back.
        dest = "format" if name == "fmt" else name
        if not hasattr(args, dest):
            continue
        default = parser.get_default(dest)
        if getattr(args, dest) == default:
            setattr(args, dest, value)
            filled.append(dest)
    return filled


def resolve_config(
    args: argparse.Namespace,
    build_config: Callable[[argparse.Namespace], PipelineConfig],
    *,
    parser: argparse.ArgumentParser | None = None,
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
        apply_config_file(args, parser)
    config = build_config(args)
    config_hash = config.config_hash()

    if register:
        try:
            config_registry.register(config, purpose=getattr(args, "config_purpose", "") or "")
        except OSError as exc:  # never let registry I/O sink a real run
            print(
                json.dumps({"progress": "config_registry_warning", "message": str(exc)}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )

    if getattr(args, "print_config_hash", False):
        print(
            json.dumps({"progress": "config_hash", "pipeline": config.PIPELINE, "config_hash": config_hash}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )

    dump_path = getattr(args, "dump_config", None)
    if dump_path:
        out = Path(dump_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(config.canonical_payload(), indent=2, sort_keys=True) + "\n")

    return config


def load_config(path: str | Path) -> PipelineConfig:
    """Reconstruct a typed config from a canonical-JSON file written by --dump-config."""
    payload = json.loads(Path(path).read_text())
    return config_from_payload(payload)
