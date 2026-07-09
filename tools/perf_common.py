"""Shared helpers for CAT-71 standing performance measurement
(tools/perf_snapshot.py, tools/perf_report.py): JSONL ledger I/O,
latency-summary stats, and named regression detectors for known anomaly
signatures (GPU context-thrash, SH-floor overrun).

Kept dependency-free (stdlib only) so it can be imported and unit-tested
without torch/catan_zero installed.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_PATH = Path("runs/perf/perf_ledger.jsonl")

# Historical ground truth cited in CAT-71's issue text / the
# catan-speed-czar-program precedent: three SH-floor overrun incidents
# measured at 32ms/105ms/119ms against "an expected floor". The exact
# production floor value that made these overruns lives on whatever host
# captured them (not in this checkout -- grepped exhaustively for "SH floor",
# "32ms", "105ms", "119ms" across docs/ and tools/ with zero hits beyond the
# issue text itself). Treat these as illustrative inputs for the retroactive
# detector self-check (see perf_report.py --verify-known-anomalies), not as
# a verified floor derivation -- swap in the live per-host floor once
# tools/perf_snapshot.py's `leaf` mode has produced enough same-host samples.
HISTORICAL_SH_FLOOR_OVERRUN_MS: tuple[float, ...] = (32.0, 105.0, 119.0)

# Documented in-repo GPU per-leaf benchmark (docs/plans/
# CATAN_ZERO_RESEARCH_CHRONICLE.md section 10.1: "GPU per-leaf ~3.4ms").
DOCUMENTED_GPU_PER_LEAF_MS: float = 3.4


def stable_key(*parts: Any) -> str:
    return "|".join(str(part) for part in parts)


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Mean/p50/p95/total/n over a list of millisecond latencies.

    Same nearest-rank percentile convention as tools/bench_leaf_eval_batching.py's
    `_summarize` helper (kept compatible on purpose so a reader who knows one
    trusts the other) -- this version returns the dict only, no print, so
    callers can print/format however their subcommand wants.
    """
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "total_ms": 0.0, "n": 0.0}
    ordered = sorted(latencies_ms)
    n = len(ordered)
    p50 = ordered[n // 2]
    p95 = ordered[min(n - 1, int(n * 0.95))]
    mean = statistics.mean(ordered)
    total = sum(ordered)
    return {"mean_ms": mean, "p50_ms": p50, "p95_ms": p95, "total_ms": total, "n": float(n)}


# --- append-only JSONL ledger, modeled on tools/update_population_payoffs.py -----


def append_ledger_rows(
    output: str | Path,
    rows: list[dict[str, Any]],
    *,
    dedupe_existing: bool = True,
) -> int:
    """Append `rows` to a JSONL ledger, one JSON object per line.

    Dedupes by each row's `key` field against keys already present in the
    file (mirrors tools/update_population_payoffs.py's
    `append_payoff_entries`). Every row must have a `key`. CAT-71 review
    finding 2: a fired dedupe is logged to stderr (not silently skipped) so a
    row that looked identical on its dedupe key but actually mattered (e.g. a
    key collision) leaves a trace instead of vanishing without a sign.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[str] = set()
    if dedupe_existing and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if key:
                existing_keys.add(str(key))
    written = 0
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = str(row["key"])
            if dedupe_existing and key in existing_keys:
                print(
                    f"[perf_common] dedupe: skipping ledger row with duplicate key={key!r} "
                    f"(kind={row.get('kind')!r}) -- already present in {output}",
                    file=sys.stderr,
                )
                continue
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            existing_keys.add(key)
            written += 1
    return written


def load_ledger(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# --- rows/hr generation-log parsing ---------------------------------------


def parse_generation_manifest(
    payload: dict[str, Any],
    *,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Extract a rows/hr-per-host row from a
    tools/generate_gumbel_selfplay_data.py manifest.json (the
    `_merge_worker_summaries` schema: rows, elapsed_sec, rows_per_sec,
    workers, out_dir)."""
    rows = int(payload.get("rows", 0))
    elapsed_sec = float(payload.get("elapsed_sec", 0.0))
    rows_per_sec = float(
        payload.get("rows_per_sec", rows / elapsed_sec if elapsed_sec > 0 else 0.0)
    )
    games_completed = int(payload.get("games_completed", 0))
    out_dir = payload.get("out_dir")
    return {
        "kind": "generation",
        "out_dir": out_dir,
        "hostname": hostname,
        "workers": int(payload.get("workers", 0)),
        "rows": rows,
        "games_completed": games_completed,
        "elapsed_sec": elapsed_sec,
        "rows_per_sec": rows_per_sec,
        "rows_per_hr": rows_per_sec * 3600.0,
        "games_per_hr": (games_completed / elapsed_sec * 3600.0) if elapsed_sec > 0 else 0.0,
        "key": stable_key("generation", out_dir, hostname, rows, round(elapsed_sec, 3)),
    }


# --- gate cost tracking -----------------------------------------------------


def parse_gate_summary(
    payload: dict[str, Any],
    *,
    gate_name: str,
    summary_path: str,
    wall_clock_sec: float | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Extract wall-clock + game-count + extension-tier info from a gate run.

    Reads both tools/promotion_gate_runner.py's per-leg `sprt_report` fields
    (`tier_games` / `tier_index` / `tiers`) and the real H2H summary writers'
    schema (tools/gumbel_search_vs_bot_h2h.py, gumbel_search_vs_raw_h2h.py,
    gumbel_search_cross_net_h2h.py all agree: `games_played` is the int game
    count, `games` is the *list* of per-game result dicts -- NOT a count).
    `games_completed` is also accepted for older/other callers. `games` is
    only used as a count fallback when it isn't a list, so pointing this at a
    real H2H summary can't crash on `int(games)` seeing a list.
    `wall_clock_sec` overrides the payload's own `elapsed_sec` when the
    caller measured the gate invocation itself (see perf_snapshot.py's
    `gate` subcommand with `--cmd`, which wraps a gate command with
    time.perf_counter()).

    CAT-71 review finding 2: the dedupe key used to be metrics-only
    (gate_name/summary_path/elapsed/games), so re-parsing the SAME summary
    file after its SPRT decision flipped (e.g. CONTINUE -> H1, same
    games/elapsed at the moment of the flip) produced an identical key and
    was silently dropped by `append_ledger_rows`'s dedupe -- a decision
    reversal must never be lost. `decision` and a fetch-time `timestamp` are
    now both part of the key (matching how the `leaf`/`gpu_util` row kinds
    already fold a timestamp into their own keys), so a decision reversal
    always lands as a new row; `append_ledger_rows` also now logs (instead of
    silently skipping) whenever a dedupe actually fires.
    """
    games = payload.get("games_played")
    if games is None:
        games = payload.get("games_completed")
    if games is None and not isinstance(payload.get("games"), list):
        games = payload.get("games")
    if games is None:
        games = payload.get("tier_games")
    games = int(games) if games is not None else 0
    elapsed = wall_clock_sec if wall_clock_sec is not None else float(payload.get("elapsed_sec", 0.0))
    tiers = payload.get("tiers")
    tier_index = payload.get("tier_index")
    extended = bool(tiers) and tier_index is not None and int(tier_index) > 0
    decision = payload.get("decision") or payload.get("sprt_decision")
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "kind": "gate",
        "gate_name": gate_name,
        "timestamp": timestamp,
        "hostname": hostname,
        "summary_path": summary_path,
        "games": games,
        "elapsed_sec": elapsed,
        "games_per_sec": games / elapsed if elapsed > 0 else 0.0,
        "tiers": tiers,
        "tier_index": tier_index,
        "extended": extended,
        "decision": decision,
        "key": stable_key("gate", gate_name, summary_path, round(elapsed, 3), games, decision, timestamp),
    }


# --- named anomaly-signature detectors --------------------------------------


def check_gpu_context_thrash(
    sm_util_pct: float,
    mem_util_pct: float,
    *,
    sm_threshold: float = 85.0,
    mem_threshold: float = 5.0,
) -> bool:
    """The pmon/dmon "context-thrash" signature (docs/plans/
    CATAN_ZERO_RESEARCH_CHRONICLE.md section 10.1): ~90% SM time-occupancy
    with ~0-1% memory-bandwidth utilization means many tiny kernel launches
    are thrashing the GPU context, not genuine compute load (which would
    show non-trivial memory-bandwidth utilization too). This is the smoking
    gun distinguishing "GPU is busy because it's doing real work" from
    "GPU is busy doing nothing useful, one kernel launch at a time"."""
    return sm_util_pct >= sm_threshold and mem_util_pct <= mem_threshold


def check_sh_floor_overrun(
    observed_ms: float,
    floor_ms: float,
    *,
    overrun_ratio: float = 1.5,
) -> bool:
    """Sequential-Halving per-decision wall-time floor overrun: `observed_ms`
    exceeding `overrun_ratio`x the expected floor (derived from the per-leaf
    profile times evals-per-decision) signals a scheduling/thrash
    regression -- the class of incident behind CAT-71's cited 32ms/105ms/
    119ms overruns (see HISTORICAL_SH_FLOOR_OVERRUN_MS docstring for the
    floor-provenance caveat)."""
    return floor_ms > 0 and observed_ms > floor_ms * overrun_ratio


def parse_dmon_pucm(text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi dmon -s pucm -c 1` fixed-width text output into one
    dict per GPU row.

    dmon prints two '#'-prefixed header lines (names, then units) followed
    by one data line per GPU, e.g.::

        # gpu   pwr gtemp mtemp    sm   mem   enc   dec   jpg   ofa  mclk  pclk
        # Idx     W     C     C     %     %     %     %     %     %   MHz   MHz
            0   250    45     -    92     1     0     0     0     0  2619  1980

    Column names come from the first header line; the second (units) is
    skipped. `-` is treated as missing (None). Handles multiple GPUs (one
    data line each) and tolerates driver versions with fewer/more columns
    (e.g. no jpg/ofa) since it zips names to values positionally.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            tokens = stripped.lstrip("#").split()
            if header is None:
                header = tokens
            # second '#' line is the units header -- ignored.
            continue
        if header is None:
            continue
        values = stripped.split()
        row: dict[str, Any] = {}
        for name, value in zip(header, values):
            if value == "-":
                row[name] = None
                continue
            try:
                row[name] = int(value)
            except ValueError:
                try:
                    row[name] = float(value)
                except ValueError:
                    row[name] = value
        rows.append(row)
    return rows
