#!/usr/bin/env python3
"""Shared prelaunch-guard wiring for the real launcher CLIs (CAT-75).

``tools/prelaunch_guard.py`` (CAT-69) is a pure guard *library*: every guard
is a callable that returns a ``GuardResult`` and never touches ``sys.exit``
or a real launch's argv. Without this module, that library is "build and
shelve" -- reviewed, tested, and never actually consulted by a real launch,
the exact pattern CAT-69's own review flagged as the standing risk.

This module is the thin, launcher-facing wiring layer on top of it: load a
launcher's static guard config from ``configs/guards/<launcher>.json``, merge
in the per-invocation dynamic values (argv, the real parser, a seed range, a
checkpoint path, ...) that can only be known inside ``main()``, run every
guard, and refuse to proceed (``SystemExit``) on any FAIL -- unless the
launcher's own ``--skip-guards`` escape hatch was passed, in which case a
loud WARNING is logged and the launch proceeds anyway.

Each retained internal launcher (``tools/generate_gumbel_selfplay_data.py``
and ``tools/train_bc.py``) calls
:func:`run_or_refuse` once, at the very top of ``main()``, immediately after
``build_parser().parse_args()`` -- before any other side effect (file I/O,
subprocess, torch import). Guards never fire on ``--help`` (argparse exits
during ``parse_args`` itself, before this module is even reached) and
callers skip this module's call entirely under a launcher's own
``--dry-run`` flag (dry-run exercises control flow only and legitimately
points at fixture paths a real guard would correctly refuse).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import prelaunch_guard  # type: ignore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD_CONFIG_DIR = REPO_ROOT / "configs" / "guards"


def argv_with_config_values(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    raw_argv: Sequence[str],
    config_filled: Sequence[str],
) -> list[str]:
    """Represent config-filled values as explicit inputs to CLI guards.

    The guard rejects silent parser defaults, but a validated typed config is
    also explicit operator input.  Synthesize only fields that
    ``apply_config_file`` actually filled, then let argparse re-parse them for
    the guard's exact-value checks.
    """

    effective = list(raw_argv)
    actions_by_dest = {action.dest: action for action in parser._actions}  # noqa: SLF001
    for dest in config_filled:
        action = actions_by_dest.get(dest)
        if action is None or not action.option_strings:
            continue
        value = getattr(args, dest)
        if value is None:
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            prefix = "--no-" if not bool(value) else "--"
            option = next(
                (item for item in action.option_strings if item.startswith(prefix)),
                action.option_strings[0],
            )
            effective.append(option)
        elif isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
            if bool(value):
                effective.append(action.option_strings[0])
        elif isinstance(action, argparse._StoreFalseAction):  # noqa: SLF001
            if not bool(value):
                effective.append(action.option_strings[0])
        elif isinstance(action, argparse._AppendAction):  # noqa: SLF001
            for item in value:
                effective.extend((action.option_strings[0], str(item)))
        elif isinstance(value, (list, tuple)):
            if not value:
                continue
            effective.append(action.option_strings[0])
            effective.extend(str(item) for item in value)
        else:
            effective.extend((action.option_strings[0], str(value)))
    return effective


def load_static_guard_specs(
    launcher: str, *, config_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Load the ``{"guards": [{"name": ..., "args": {...}}, ...]}`` config
    committed at ``configs/guards/<launcher>.json``.
    """
    path = (
        GUARD_CONFIG_DIR / f"{launcher}.json"
        if config_path is None
        else Path(config_path).expanduser()
    )
    lexical = Path(path).absolute()
    resolved = lexical.resolve(strict=True)
    if lexical != resolved or resolved.parent != GUARD_CONFIG_DIR.resolve(strict=True):
        raise ValueError(
            f"static guard config must be one canonical file in {GUARD_CONFIG_DIR}: {lexical}"
        )
    payload = json.loads(resolved.read_text())
    return list(payload["guards"])


def merge_dynamic_args(
    static_specs: Sequence[Mapping[str, Any]],
    dynamic_args: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay per-invocation ``dynamic_args`` (keyed by guard name -- the
    real ``argv``/``parser``/seed range/checkpoint path, values that cannot
    be committed to a static JSON config) onto a static guard-spec list.
    Dynamic values win on key conflicts; guards not named in
    ``dynamic_args`` pass through with only their static args.
    """
    merged: list[dict[str, Any]] = []
    for spec in static_specs:
        name = spec["name"]
        args = {**spec.get("args", {}), **dynamic_args.get(name, {})}
        merged.append({"name": name, "args": args})
    return merged


def run_or_refuse(
    guard_specs: Sequence[Mapping[str, Any]],
    *,
    launcher: str,
    skip: bool,
) -> None:
    """Run every guard spec and refuse to launch (``SystemExit``) if any of
    them FAILs.

    ``skip=True`` (the launcher's ``--skip-guards`` flag) never silently
    bypasses this: it logs one loud WARNING naming every guard that would
    have run and returns without running anything.
    """
    guard_names = sorted({spec["name"] for spec in guard_specs})
    if skip:
        print(
            f"WARNING: --skip-guards was passed; SKIPPING {len(guard_specs)} prelaunch "
            f"guard(s) for {launcher} ({guard_names}). This bypasses automated checks "
            "for every documented incident class the guard library encodes (CAT-69/"
            "CAT-75) -- use only for a known false positive or an intentional smoke test.",
            file=sys.stderr,
        )
        return

    results = prelaunch_guard.run_guards(guard_specs)
    for result in results:
        marker = "host-only, " if result.host_only else ""
        print(
            f"[{result.status}] prelaunch guard ({marker}{result.guard}) [{launcher}]: {result.reason}",
            file=sys.stderr,
        )
    failures = [result for result in results if not result.passed]
    if failures:
        failed_names = sorted({result.guard for result in failures})
        raise SystemExit(
            f"{launcher}: refusing to launch -- {len(failures)} prelaunch guard(s) FAILED "
            f"{failed_names}. Fix the underlying issue, or pass --skip-guards to proceed "
            "anyway (logs a loud warning and runs none of them)."
        )


def discover_generation_seed_ranges(data_path: str | Path) -> list[tuple[int, int]]:
    """Best-effort discovery of the ``[base_seed, base_seed + games)`` range(s)
    a training corpus at ``data_path`` was generated from, by reading
    ``base_seed``/``games_requested`` back out of any
    ``generate_gumbel_selfplay_data.py``-style ``manifest.json`` reachable
    from ``data_path`` (a top-level manifest, or one level of nested
    manifests -- mirrors ``train_bc._teacher_shard_files``'s own manifest
    discovery, without needing to import train_bc for it).

    Returns an empty list (the val-only-never-trains guard then has nothing
    to check and trivially passes) rather than raising when no manifest with
    both fields is found -- this is a defense-in-depth check layered on top
    of the existing seed_fleet_planner/seed-claim machinery at generation
    time, not the sole seed-collision guarantee.
    """
    data_path = Path(data_path)
    candidates: list[Path] = []
    top = data_path / "manifest.json"
    if top.exists():
        candidates.append(top)
    elif data_path.is_dir():
        candidates.extend(sorted(data_path.glob("*/manifest.json")))

    ranges: list[tuple[int, int]] = []
    for candidate in candidates:
        try:
            manifest = json.loads(candidate.read_text())
            base_seed = int(manifest["base_seed"])
            games = int(manifest["games_requested"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        ranges.append((base_seed, base_seed + games))
    return ranges
