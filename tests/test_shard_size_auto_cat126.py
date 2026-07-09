"""CAT-126 #4: --shard-size auto-scales by --n-full unless explicitly passed.

Data-bit-identical (only shard granularity changes, not rows). Default stays 2048
(so explicit callers + n64 volume are unchanged); slow high-n teacher/probe runs
that omit --shard-size get smaller shards so first shards flush sooner.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import generate_gumbel_selfplay_data as cli  # type: ignore


def test_auto_shard_size_thresholds():
    assert cli._auto_shard_size(64) == 2048
    assert cli._auto_shard_size(100) == 2048
    assert cli._auto_shard_size(128) == 512
    assert cli._auto_shard_size(200) == 512
    assert cli._auto_shard_size(256) == 256
    assert cli._auto_shard_size(512) == 256


def test_explicit_detection():
    assert cli._shard_size_was_explicit(["--n-full", "128", "--shard-size", "2048"])
    assert cli._shard_size_was_explicit(["--shard-size=512"])
    assert not cli._shard_size_was_explicit(["--n-full", "128", "--games", "1"])


def _resolve(argv):
    """Mirror main()'s resolution on real parsed args."""
    args = cli.build_parser().parse_args(argv)
    if not cli._shard_size_was_explicit(argv):
        args.shard_size = cli._auto_shard_size(int(args.n_full))
    return int(args.shard_size)


def _min(extra):
    return ["--out-dir", "/tmp/x", "--games", "1", "--checkpoint", "/c.pt", *extra]


def test_parser_default_unchanged():
    # Direct parse (no resolution) still yields 2048 — protects direct-parse tests.
    assert cli.build_parser().parse_args(_min([])).shard_size == 2048


def test_resolution_end_to_end():
    assert _resolve(_min(["--n-full", "64"])) == 2048     # volume unchanged
    assert _resolve(_min(["--n-full", "128"])) == 512     # teacher auto-shrinks
    assert _resolve(_min(["--n-full", "256"])) == 256     # probe auto-shrinks
    # explicit always wins:
    assert _resolve(_min(["--n-full", "128", "--shard-size", "2048"])) == 2048
    assert _resolve(_min(["--n-full", "256", "--shard-size", "1024"])) == 1024
