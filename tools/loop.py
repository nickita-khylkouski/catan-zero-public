#!/usr/bin/env python3
"""Run one canonical config-driven RL improvement loop.

The CLI intentionally exposes only configuration identity, durable state, and
the execution switch.  Search, learner, evaluator, and fleet placement settings
belong to the stage configs referenced by the loop document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from catan_zero.rl.production_loop import (  # noqa: E402
    ProductionLoopError,
    execute,
    load_config,
    plan,
)


CANONICAL_OPTION_COUNT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--go", action="store_true", help="execute; default is dry-run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    if len([a for a in parser._actions if a.option_strings and a.dest != "help"]) != CANONICAL_OPTION_COUNT:  # noqa: SLF001
        parser.error("canonical loop CLI exceeded its three-option budget")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, state_dir=args.state_dir)
        result = (
            execute(config, state_dir=args.state_dir)
            if args.go
            else plan(config, state_dir=args.state_dir)
        )
    except (ProductionLoopError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
