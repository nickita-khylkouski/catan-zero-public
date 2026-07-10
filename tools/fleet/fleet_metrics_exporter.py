#!/usr/bin/env python3
"""Prometheus exporter for Catan-Zero generation progress.

DCGM and node-exporter remain authoritative for GPU and host telemetry. This
small exporter adds the application layer they cannot see: generator process
health, durable game/row/simulation/shard progress, failures/truncations,
typed config hash, seed range, output-disk capacity, and progress staleness.
It reads existing config/manifest/progress files and ``/proc`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import socket
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

METRIC_PREFIX = "catan_fleet_"
GPU_DIR_RE = re.compile(
    r"^gpu(?P<gpu>[0-9]+)(?:_pipeline(?P<pipeline>[01]))?$"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _config_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()[:16]


def _flag_value(argv: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def discover_generators(proc_root: Path = Path("/proc")) -> dict[str, set[int]]:
    """Map resolved ``--out-dir`` paths to live top-level generator PIDs."""
    found: dict[str, set[int]] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if not any(value.endswith("generate_gumbel_selfplay_data.py") for value in argv):
            continue
        out_dir = _flag_value(argv, "--out-dir")
        if not out_dir:
            continue
        found.setdefault(str(Path(out_dir).expanduser().resolve()), set()).add(int(entry.name))
    return found


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


@dataclass(frozen=True)
class RunSnapshot:
    host: str
    gpu: str
    pipeline: str
    run: str
    role: str
    config_hash: str
    n_full: int
    p_full: float
    seed_start: int
    seed_end: int
    games_requested: int
    games_completed: int
    rows: int
    simulations: int
    shards: int
    failures: int
    truncations: int
    process_count: int
    complete: bool
    stale_seconds: float
    healthy: bool
    output_dir: Path


def _aggregate_progress(gpu_dir: Path) -> tuple[dict[str, float], float]:
    totals = {
        "games_requested": 0.0,
        "games_completed": 0.0,
        "rows": 0.0,
        "simulations": 0.0,
        "shards": 0.0,
        "failures": 0.0,
        "truncations": 0.0,
    }
    newest = 0.0
    for path in sorted(gpu_dir.glob("worker_*/progress.json")):
        payload = _load_json(path)
        if payload is None:
            continue
        newest = max(newest, path.stat().st_mtime)
        totals["games_requested"] += _number(payload.get("games_requested"))
        # games_completed_local is the durable, shard-confirmed counter.
        totals["games_completed"] += _number(payload.get("games_completed_local"))
        totals["rows"] += _number(payload.get("rows"))
        totals["simulations"] += _number(payload.get("simulations_used_total"))
        totals["shards"] += _number(payload.get("shard_count_confirmed"))
        totals["failures"] += _number(payload.get("games_failed"))
        totals["truncations"] += _number(payload.get("games_truncated"))
    return totals, newest


def snapshot_run(
    gpu_dir: Path,
    *,
    host: str,
    processes: Mapping[str, set[int]],
    now: float,
    stale_after_seconds: float,
) -> RunSnapshot | None:
    match = GPU_DIR_RE.match(gpu_dir.name)
    if match is None:
        return None
    config_path = gpu_dir / "config.json"
    manifest_path = gpu_dir / "manifest.json"
    config = _load_json(config_path) or {}
    fields = config.get("fields") if isinstance(config.get("fields"), dict) else {}
    manifest = _load_json(manifest_path)
    progress, progress_mtime = _aggregate_progress(gpu_dir)
    if not config and manifest is None and progress_mtime == 0.0:
        return None

    if manifest is not None:
        values = {
            "games_requested": _number(manifest.get("games_requested")),
            "games_completed": _number(manifest.get("games_completed")),
            "rows": _number(manifest.get("rows")),
            "simulations": _number(manifest.get("simulations_used_total")),
            "shards": float(len(manifest.get("shards", [])))
            if isinstance(manifest.get("shards"), list)
            else 0.0,
            "failures": _number(manifest.get("games_failed")),
            "truncations": _number(manifest.get("games_truncated")),
        }
    else:
        values = progress

    mtimes = [value for value in (progress_mtime,) if value > 0]
    for path in (config_path, manifest_path):
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    newest = max(mtimes, default=0.0)
    age = max(0.0, now - newest) if newest else float("inf")
    resolved = str(gpu_dir.resolve())
    pids = processes.get(resolved, set())
    games_requested = int(values["games_requested"] or _number(fields.get("games")))
    games_completed = int(values["games_completed"])
    errors = manifest.get("errors", []) if manifest is not None else []
    complete = bool(
        manifest is not None
        and games_requested > 0
        and games_completed == games_requested
        and values["failures"] == 0
        and not errors
        and manifest.get("fatal_execution_error") in (None, {})
    )
    healthy = bool(
        (len(pids) > 0 and age <= stale_after_seconds and values["failures"] == 0)
        or complete
    )
    config_hash = (
        str(manifest.get("config_hash"))
        if manifest is not None and manifest.get("config_hash")
        else _config_hash(config)
        if config
        else "pending"
    )
    seed_start = int(
        _number(
            manifest.get("base_seed") if manifest is not None else fields.get("base_seed")
        )
    )
    if not seed_start:
        seed_start = int(_number(fields.get("base_seed")))
    run = gpu_dir.parent.name
    n_full = int(
        _number(manifest.get("n_full") if manifest is not None else fields.get("n_full"))
    )
    p_full = _number(
        manifest.get("p_full") if manifest is not None else fields.get("p_full")
    )
    role = "teacher" if n_full >= 128 else "volume"
    return RunSnapshot(
        host=host,
        gpu=match.group("gpu"),
        pipeline=match.group("pipeline") or "0",
        run=run,
        role=role,
        config_hash=config_hash,
        n_full=n_full,
        p_full=p_full,
        seed_start=seed_start,
        seed_end=seed_start + games_requested,
        games_requested=games_requested,
        games_completed=games_completed,
        rows=int(values["rows"]),
        simulations=int(values["simulations"]),
        shards=int(values["shards"]),
        failures=int(values["failures"]),
        truncations=int(values["truncations"]),
        process_count=len(pids),
        complete=complete,
        stale_seconds=age,
        healthy=healthy,
        output_dir=gpu_dir,
    )


def collect_snapshots(
    roots: Iterable[Path],
    *,
    host: str,
    processes: Mapping[str, set[int]],
    now: float,
    stale_after_seconds: float,
    max_run_age_seconds: float,
) -> list[RunSnapshot]:
    snapshots: list[RunSnapshot] = []
    for root in roots:
        for gpu_dir in sorted(root.expanduser().glob("*/gpu*")):
            snapshot = snapshot_run(
                gpu_dir,
                host=host,
                processes=processes,
                now=now,
                stale_after_seconds=stale_after_seconds,
            )
            if snapshot is None:
                continue
            if snapshot.process_count == 0 and snapshot.stale_seconds > max_run_age_seconds:
                continue
            snapshots.append(snapshot)
    # Avoid unbounded label cardinality: expose one current run per physical
    # GPU/pipeline slot. Production supports at most two pipelines per GPU.
    by_slot: dict[tuple[str, str], RunSnapshot] = {}
    for snapshot in snapshots:
        slot = (snapshot.gpu, snapshot.pipeline)
        prior = by_slot.get(slot)
        score = (snapshot.process_count > 0, -snapshot.stale_seconds)
        prior_score = (
            (prior.process_count > 0, -prior.stale_seconds) if prior is not None else None
        )
        if prior_score is None or score > prior_score:
            by_slot[slot] = snapshot
    return [by_slot[slot] for slot in sorted(by_slot, key=lambda item: tuple(map(int, item)))]


def _escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _sample(name: str, value: float | int, labels: Mapping[str, Any]) -> str:
    rendered = ",".join(
        f'{key}="{_escape_label(labels[key])}"' for key in sorted(labels)
    )
    return f"{METRIC_PREFIX}{name}{{{rendered}}} {value}"


def render_metrics(
    snapshots: Sequence[RunSnapshot],
    *,
    host: str,
    roots: Sequence[Path],
    now: float,
    scrape_success: bool = True,
) -> str:
    lines = [
        "# HELP catan_fleet_exporter_up Exporter collection succeeded.",
        "# TYPE catan_fleet_exporter_up gauge",
        _sample("exporter_up", int(scrape_success), {"host": host}),
        "# HELP catan_fleet_exporter_scrape_timestamp_seconds Unix scrape time.",
        "# TYPE catan_fleet_exporter_scrape_timestamp_seconds gauge",
        _sample("exporter_scrape_timestamp_seconds", now, {"host": host}),
    ]
    metric_fields = {
        "generator_processes": "process_count",
        "generator_healthy": "healthy",
        "generator_complete": "complete",
        "generator_progress_age_seconds": "stale_seconds",
        "generator_games_requested": "games_requested",
        "generator_games_completed": "games_completed",
        "generator_rows": "rows",
        "generator_simulations": "simulations",
        "generator_shards": "shards",
        "generator_failures": "failures",
        "generator_truncations": "truncations",
        "generator_seed_start": "seed_start",
        "generator_seed_end": "seed_end",
    }
    for snapshot in snapshots:
        labels = {
            "host": snapshot.host,
            "gpu": snapshot.gpu,
            "pipeline": snapshot.pipeline,
            "run": snapshot.run,
            "role": snapshot.role,
            "config_hash": snapshot.config_hash,
        }
        info_labels = {
            **labels,
            "n_full": snapshot.n_full,
            "p_full": snapshot.p_full,
            "seed_range": f"[{snapshot.seed_start},{snapshot.seed_end})",
        }
        lines.append(_sample("generator_info", 1, info_labels))
        for metric, field in metric_fields.items():
            value = getattr(snapshot, field)
            if isinstance(value, bool):
                value = int(value)
            lines.append(_sample(metric, value, labels))
    for root in roots:
        resolved = root.expanduser().resolve()
        try:
            usage = shutil.disk_usage(resolved if resolved.exists() else resolved.parent)
        except OSError:
            continue
        labels = {"host": host, "path": str(resolved)}
        lines.append(_sample("output_disk_free_bytes", usage.free, labels))
        lines.append(_sample("output_disk_total_bytes", usage.total, labels))
    return "\n".join(lines) + "\n"


def collect_metrics(args: argparse.Namespace) -> str:
    now = time.time()
    processes = discover_generators(Path(args.proc_root))
    roots = [Path(value) for value in args.run_root]
    snapshots = collect_snapshots(
        roots,
        host=args.host_label,
        processes=processes,
        now=now,
        stale_after_seconds=float(args.stale_after_seconds),
        max_run_age_seconds=float(args.max_run_age_seconds),
    )
    return render_metrics(snapshots, host=args.host_label, roots=roots, now=now)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9500)
    parser.add_argument("--host-label", default=socket.gethostname())
    parser.add_argument("--run-root", action="append", default=None)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--stale-after-seconds", type=float, default=300.0)
    parser.add_argument("--max-run-age-seconds", type=float, default=86400.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_root is None:
        args.run_root = [str(Path.home() / "gen_out")]
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be in 1..65535")
    if args.stale_after_seconds <= 0 or args.max_run_age_seconds <= 0:
        parser.error("staleness windows must be positive")
    if args.once:
        print(collect_metrics(args), end="")
        return 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path not in {"/", "/metrics"}:
                self.send_error(404)
                return
            try:
                body = collect_metrics(args).encode("utf-8")
                status = 200
            except Exception as error:  # exporter must remain scrapeable on bad input
                body = render_metrics(
                    [],
                    host=args.host_label,
                    roots=[],
                    now=time.time(),
                    scrape_success=False,
                ).encode("utf-8")
                body += f"# collection error: {type(error).__name__}\n".encode()
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
