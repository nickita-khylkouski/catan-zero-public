"""Sharded-H2H-gate monitor + atomic verdict writes (CAT-runsix bug-a).

Before this, the gate had no monitor: aggregators defaulted --out=None (nothing
durable landed) and nothing invoked a next ladder step, so the pipeline stalled
after the shards finished. These tests pin: shards are only counted when fully
written, the monitor writes a durable verdict, hands off to the next step, and
fails LOUD (marker + non-zero) on timeout instead of silently succeeding."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import atomic_io  # type: ignore  # noqa: E402
import h2h_gate_monitor as mon  # type: ignore  # noqa: E402


# --- atomic_io ---------------------------------------------------------------

def test_atomic_write_lands_whole_and_leaves_no_tmp(tmp_path):
    out = tmp_path / "verdict.json"
    atomic_io.write_json_atomic(out, {"verdict": "PASS", "n": 3})
    assert json.loads(out.read_text()) == {"verdict": "PASS", "n": 3}
    assert list(tmp_path.glob(".verdict.json.*.tmp")) == []  # temp cleaned up


# --- shard readiness ---------------------------------------------------------

def test_count_ready_shards_skips_partial_and_verdict(tmp_path):
    (tmp_path / "armA_h_g0.json").write_text(json.dumps({"games": []}))
    (tmp_path / "armA_h_g1.json").write_text(json.dumps({"games": []}))
    (tmp_path / "armA_h_g2.json").write_text("{ not valid json")  # mid-write shard
    (tmp_path / "verdict.json").write_text(json.dumps({"verdict": "PASS"}))  # own output
    ready = mon.count_ready_shards(tmp_path, "arm*_*.json", out_name="verdict.json")
    assert len(ready) == 2


# --- polling / timeout -------------------------------------------------------

def test_wait_for_shards_returns_when_complete(tmp_path):
    for i in range(3):
        (tmp_path / f"armA_h_g{i}.json").write_text(json.dumps({"games": []}))
    fake = iter([0.0, 0.0, 0.0])
    ready = mon.wait_for_shards(
        tmp_path, "arm*_*.json", 3, out_name="verdict.json",
        poll_seconds=1, timeout_seconds=100,
        clock=lambda: next(fake), sleep=lambda _s: None, log=lambda _m: None,
    )
    assert len(ready) == 3


def test_wait_for_shards_times_out(tmp_path):
    (tmp_path / "armA_h_g0.json").write_text(json.dumps({"games": []}))
    ticks = iter([0.0, 5.0, 999.0])  # third read is past the deadline
    ready = mon.wait_for_shards(
        tmp_path, "arm*_*.json", 3, out_name="verdict.json",
        poll_seconds=1, timeout_seconds=10,
        clock=lambda: next(ticks), sleep=lambda _s: None, log=lambda _m: None,
    )
    assert len(ready) < 3


# --- end-to-end monitor ------------------------------------------------------

_STUB_AGG = (
    f"{sys.executable} -c "
    "\"import json,sys; json.dump({'verdict':'PASS','arms':1}, open(sys.argv[1],'w'))\" {out}"
)


def _make_shards(d: Path, n: int) -> None:
    for i in range(n):
        (d / f"armA_h_g{i}.json").write_text(json.dumps({"games": []}))


def test_run_monitor_aggregates_and_hands_off(tmp_path):
    _make_shards(tmp_path, 3)
    out = tmp_path / "verdict.json"
    sentinel = tmp_path / "handoff.txt"
    on_complete = (
        f"{sys.executable} -c "
        f"\"import os; open(r'{sentinel}','w').write(os.environ['VERDICT_PATH'])\""
    )
    rc = mon.run_monitor(
        tmp_path, glob="arm*_*.json", aggregator_cmd=_STUB_AGG, expected=3,
        out_path=out, poll_seconds=0, timeout_seconds=10,
        on_complete=on_complete, clock=lambda: 0.0, sleep=lambda _s: None, log=lambda _m: None,
    )
    assert rc == 0
    assert json.loads(out.read_text())["verdict"] == "PASS"
    assert sentinel.read_text() == str(out)  # handoff saw $VERDICT_PATH


def test_run_monitor_timeout_writes_marker_and_refuses_handoff(tmp_path):
    _make_shards(tmp_path, 1)  # fewer than expected
    out = tmp_path / "verdict.json"
    sentinel = tmp_path / "handoff.txt"
    ticks = iter([0.0, 999.0, 999.0])
    rc = mon.run_monitor(
        tmp_path, glob="arm*_*.json", aggregator_cmd=_STUB_AGG, expected=3,
        out_path=out, poll_seconds=0, timeout_seconds=10,
        on_complete=f"{sys.executable} -c \"open(r'{sentinel}','w').write('x')\"",
        clock=lambda: next(ticks), sleep=lambda _s: None, log=lambda _m: None,
    )
    assert rc == 2
    assert (tmp_path / "monitor_timeout.json").exists()
    assert not out.exists()
    assert not sentinel.exists()  # handoff must NOT run on timeout


def test_run_monitor_refuses_handoff_when_verdict_not_written(tmp_path):
    _make_shards(tmp_path, 3)
    out = tmp_path / "verdict.json"
    sentinel = tmp_path / "handoff.txt"
    # aggregator returns 0 but writes nothing
    noop_agg = f"{sys.executable} -c \"pass\" {{out}}"
    rc = mon.run_monitor(
        tmp_path, glob="arm*_*.json", aggregator_cmd=noop_agg, expected=3,
        out_path=out, poll_seconds=0, timeout_seconds=10,
        on_complete=f"{sys.executable} -c \"open(r'{sentinel}','w').write('x')\"",
        clock=lambda: 0.0, sleep=lambda _s: None, log=lambda _m: None,
    )
    assert rc == 3
    assert not sentinel.exists()
