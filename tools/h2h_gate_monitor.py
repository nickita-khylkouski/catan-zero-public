#!/usr/bin/env python3
"""Sharded-H2H-gate monitor + ladder handoff (CAT-runsix bug-a fix).

The fleet runs an H2H (head-to-head) gate as N parallel shards, each writing
one JSON into a shard directory. Before this module there was no monitor at
all: the two aggregators (``h2h_postrepair_aggregate.py`` /
``h2h_v3conf_aggregate.py``) are one-shot CLIs a human had to remember to run,
they defaulted ``--out`` to ``None`` (stdout only, nothing durable on disk),
and nothing ever invoked a next ladder step -- so the pipeline went idle after
the shards finished and a verdict had to be hand-aggregated from the shard
JSONs.

This monitor closes that gap: poll the shard directory until every expected
shard has landed as complete JSON, run the aggregator to write ``verdict.json``
atomically, then (optionally) exec a handoff command -- the "next ladder step"
-- with ``$VERDICT_PATH`` pointing at the verdict. On timeout it refuses
silently succeeding: it writes a ``monitor_timeout.json`` marker and exits
non-zero so an operator/parent process notices the stall instead of a pipeline
that looks done but isn't.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import atomic_io  # type: ignore  # noqa: E402

# Aggregator presets: shard glob + the default command template. {dir} and
# {out} are substituted at run time. The aggregators write --out atomically.
_AGGREGATORS: dict[str, dict[str, str]] = {
    "postrepair": {
        "glob": "arm*_*.json",
        "cmd": f"{sys.executable} {_TOOLS_DIR / 'h2h_postrepair_aggregate.py'} --dir {{dir}} --out {{out}}",
    },
    "v3conf": {
        "glob": "*.json",
        "cmd": f"{sys.executable} {_TOOLS_DIR / 'h2h_v3conf_aggregate.py'} --dir {{dir}} --out {{out}}",
    },
}


def count_ready_shards(shard_dir: Path, glob: str, *, out_name: str, min_bytes: int = 2) -> list[Path]:
    """Return the shard files that are fully written (valid, non-trivial JSON).

    A shard mid-write (opened but not yet flushed) is not counted -- we parse it
    and skip it on JSONDecodeError -- so the monitor never aggregates a partial
    shard. The verdict file itself (``out_name``) is excluded so a re-run that
    globs ``*.json`` doesn't count its own output as a shard.
    """
    ready: list[Path] = []
    for path in sorted(shard_dir.glob(glob)):
        if path.name == out_name:
            continue
        try:
            if path.stat().st_size < min_bytes:
                continue
            json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        ready.append(path)
    return ready


def wait_for_shards(
    shard_dir: Path,
    glob: str,
    expected: int,
    *,
    out_name: str,
    poll_seconds: float,
    timeout_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> list[Path]:
    """Poll ``shard_dir`` until at least ``expected`` complete shards exist or
    ``timeout_seconds`` elapses. Returns the shard list (which may be shorter
    than ``expected`` on timeout -- the caller checks length)."""
    deadline = clock() + timeout_seconds
    while True:
        ready = count_ready_shards(shard_dir, glob, out_name=out_name)
        if len(ready) >= expected:
            log(f"[monitor] all {len(ready)}/{expected} shards ready in {shard_dir}")
            return ready
        if clock() >= deadline:
            log(f"[monitor] TIMEOUT: {len(ready)}/{expected} shards after {timeout_seconds}s in {shard_dir}")
            return ready
        log(f"[monitor] {len(ready)}/{expected} shards ready; sleeping {poll_seconds}s")
        sleep(poll_seconds)


def run_command(cmd: str, *, env_extra: dict[str, str] | None = None) -> int:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, shell=True, env=env).returncode


def run_monitor(
    shard_dir: Path,
    *,
    glob: str,
    aggregator_cmd: str,
    expected: int,
    out_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    on_complete: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> int:
    ready = wait_for_shards(
        shard_dir, glob, expected,
        out_name=out_path.name, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds,
        clock=clock, sleep=sleep, log=log,
    )
    if len(ready) < expected:
        marker = atomic_io.write_json_atomic(
            shard_dir / "monitor_timeout.json",
            {"status": "timeout", "expected": expected, "ready": len(ready),
             "shard_dir": str(shard_dir), "timeout_seconds": timeout_seconds},
        )
        log(f"[monitor] wrote timeout marker {marker}; exiting non-zero")
        return 2

    cmd = aggregator_cmd.replace("{dir}", str(shard_dir)).replace("{out}", str(out_path))
    log(f"[monitor] aggregating: {cmd}")
    rc = run_command(cmd)
    if rc != 0:
        log(f"[monitor] aggregator exited {rc}; NOT running handoff")
        return rc
    if not out_path.exists():
        log(f"[monitor] aggregator returned 0 but {out_path} was not written; refusing handoff")
        return 3
    try:
        json.loads(out_path.read_text())
    except (OSError, ValueError) as error:
        log(f"[monitor] verdict {out_path} is not valid JSON ({error}); refusing handoff")
        return 3
    log(f"[monitor] durable verdict written: {out_path}")

    if on_complete:
        log(f"[monitor] ladder handoff: {on_complete}")
        return run_command(on_complete, env_extra={"VERDICT_PATH": str(out_path)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="shard directory the per-shard JSONs land in")
    parser.add_argument("--aggregator", choices=sorted(_AGGREGATORS), default="postrepair",
                        help="which aggregator preset (shard glob + command) to use")
    parser.add_argument("--aggregator-cmd", default=None,
                        help="override the aggregator command template ({dir}/{out} substituted)")
    parser.add_argument("--glob", default=None, help="override the shard glob for the chosen aggregator")
    parser.add_argument("--expected-shards", type=int, required=True, help="number of shards to wait for")
    parser.add_argument("--out", default=None, help="verdict path (default: <dir>/verdict.json)")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    parser.add_argument("--on-complete", default=None,
                        help="handoff command run after a durable verdict lands ($VERDICT_PATH set)")
    args = parser.parse_args(argv)

    preset = _AGGREGATORS[args.aggregator]
    shard_dir = Path(args.dir)
    out_path = Path(args.out) if args.out else shard_dir / "verdict.json"
    return run_monitor(
        shard_dir,
        glob=args.glob or preset["glob"],
        aggregator_cmd=args.aggregator_cmd or preset["cmd"],
        expected=int(args.expected_shards),
        out_path=out_path,
        poll_seconds=float(args.poll_seconds),
        timeout_seconds=float(args.timeout_seconds),
        on_complete=args.on_complete,
    )


if __name__ == "__main__":
    raise SystemExit(main())
