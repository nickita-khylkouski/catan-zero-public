#!/usr/bin/env python3
"""Checkout-local wrapper for the single CatanZero production CLI."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    while str(value) in sys.path:
        sys.path.remove(str(value))
    sys.path.insert(0, str(value))

from catan_zero.production_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

