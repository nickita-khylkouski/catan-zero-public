#!/usr/bin/env python3
"""Atomic JSON writes (CAT-runsix bug-a fix).

Gate/aggregation tooling wrote its terminal verdict with a bare
``Path.write_text`` -- a non-atomic write that leaves a truncated file if the
process dies mid-write, and (for the aggregators) only ran at all when a human
remembered to pass ``--out``. Both are how the sharded H2H gate "never wrote a
durable verdict.json". This module is the one small stdlib-only helper the
aggregators/monitor use so a verdict either lands whole or not at all.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> Path:
    """Serialize ``payload`` to ``path`` atomically: write a temp file in the
    same directory, ``fsync`` it, then ``os.replace`` it into place (atomic on
    a POSIX filesystem). A reader either sees the previous file or the complete
    new one, never a half-written one. Returns the resolved path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(output.parent), prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return output
