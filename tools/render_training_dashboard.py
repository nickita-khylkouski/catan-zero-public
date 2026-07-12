from __future__ import annotations

import argparse
import ast
import html
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a tiny HTML/JSON training dashboard.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--target-games", type=int, default=0)
    parser.add_argument("--refresh-seconds", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    while True:
        status = build_status(run_dir, pid=args.pid, target_games=args.target_games)
        write_outputs(run_dir, status, refresh_seconds=args.refresh_seconds)
        if args.once:
            print(json.dumps(status, indent=2, sort_keys=True))
            return
        time.sleep(max(1, args.refresh_seconds))


def build_status(run_dir: Path, *, pid: int, target_games: int) -> dict[str, Any]:
    log_path = run_dir / "train.log"
    log_objects = _parse_json_stream(log_path)
    teacher_progress = _parse_teacher_progress(log_objects)
    bc_events = _parse_bc_events(log_objects)
    bc_progress = _parse_bc_progress(log_objects)
    manifest = _read_json(run_dir / "teacher_data" / "manifest.json")
    candidate_report = _read_json(run_dir / "candidate_strong_bc.json")
    xdim_report = _read_json(run_dir / "xdim_lite_strong_bc.json")
    candidate_scoreboard = _read_json(run_dir / "scoreboard_candidate.json")
    xdim_scoreboard = _read_json(run_dir / "scoreboard_xdim_lite.json")
    generic_reports = _generic_reports(run_dir)
    generic_scoreboards = _generic_scoreboards(run_dir)
    gpu = _gpu_snapshot()
    process = _process_snapshot(pid) if pid else None
    current_phase = _phase(
        manifest=manifest,
        candidate_report=candidate_report,
        xdim_report=xdim_report,
        candidate_scoreboard=candidate_scoreboard,
        xdim_scoreboard=xdim_scoreboard,
        process=process,
        training_progress=bool(bc_events),
        generic_reports=generic_reports,
    )
    games = int(teacher_progress.get("games") or manifest.get("games") or 0)
    samples = int(teacher_progress.get("samples") or manifest.get("samples") or 0)
    progress = (games / target_games) if target_games > 0 else None
    status = {
        "run_dir": str(run_dir),
        "phase": current_phase,
        "process": process,
        "gpu": gpu,
        "teacher": {
            "games": games,
            "target_games": target_games,
            "progress": progress,
            "samples": samples,
            "workers": int(teacher_progress.get("workers") or manifest.get("workers") or 0),
            "teachers": manifest.get("teachers", []),
            "elapsed_sec": manifest.get("elapsed_sec"),
            "shards": _shards(run_dir),
        },
        "bc": {
            "candidate": _bc_summary(candidate_report),
            "xdim_lite": _bc_summary(xdim_report),
        },
        "training": {
            "latest": bc_events[-1] if bc_events else {},
            "curve": bc_events,
        },
        "scoreboards": {
            "candidate": _scoreboard_summary(candidate_scoreboard),
            "xdim_lite": _scoreboard_summary(xdim_scoreboard),
        },
        "artifacts": _artifact_summary(run_dir),
        "reports": generic_reports,
        "scoreboard_files": generic_scoreboards,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    for arch, progress_item in bc_progress.items():
        if arch in status["bc"] and not status["bc"][arch].get("accuracy"):
            status["bc"][arch].update(progress_item)
    return status


def write_outputs(run_dir: Path, status: dict[str, Any], *, refresh_seconds: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dashboard.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "dashboard.html").write_text(
        _html(status, refresh_seconds=refresh_seconds),
        encoding="utf-8",
    )


def _parse_teacher_progress(lines: list[Any]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for line in lines:
        parsed = _parse_line_object(line)
        if isinstance(parsed, dict) and parsed.get("progress") == "teacher_data":
            latest = parsed
    return latest


def _parse_bc_progress(lines: list[Any]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in lines:
        parsed = _parse_line_object(line)
        if isinstance(parsed, dict) and parsed.get("progress") in {"bc", "bc_batch"}:
            arch = str(parsed.get("arch") or "")
            if arch:
                latest[arch] = {
                    "last_epoch": parsed.get("epoch"),
                    "loss": parsed.get("loss"),
                    "accuracy": parsed.get("accuracy"),
                    "samples": parsed.get("samples"),
                }
    return latest


def _parse_bc_events(lines: list[Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        parsed = _parse_line_object(line)
        if isinstance(parsed, dict) and parsed.get("progress") in {"bc", "bc_batch"}:
            events.append(parsed)
    return events


def _parse_line_object(line: str) -> Any:
    if isinstance(line, (dict, list)):
        return line
    line = line.strip()
    if not line or not line.startswith(("{", "[")):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None


def _parse_json_stream(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    decoder = json.JSONDecoder()
    values: list[Any] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        try:
            value, offset = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            # A writer may be appending the final object. Preserve every
            # complete event already decoded and retry on the next refresh.
            break
        values.append(value)
    return values


def _bc_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") or []
    last = metrics[-1] if metrics else {}
    return {
        "checkpoint": report.get("checkpoint"),
        "samples": report.get("samples"),
        "epochs": report.get("epochs"),
        "last_epoch": last.get("epoch"),
        "loss": last.get("loss"),
        "accuracy": last.get("accuracy"),
        "elapsed_sec": report.get("elapsed_sec"),
    }


def _scoreboard_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for item in report.get("results", []) if isinstance(report, dict) else []:
        output.append(
            {
                "opponent": item.get("opponent"),
                "track": item.get("track"),
                "games": item.get("games"),
                "wins": item.get("wins"),
                "win_rate": item.get("win_rate"),
                "avg_decisions": item.get("avg_decisions"),
            }
        )
    return output


def _phase(**values: Any) -> str:
    if values.get("generic_reports"):
        return "complete"
    if values["xdim_scoreboard"]:
        return "complete"
    if values["candidate_scoreboard"] and values["xdim_report"]:
        return "evaluating_xdim_lite"
    if values["xdim_report"]:
        return "evaluating_candidate"
    if values["candidate_report"]:
        return "training_xdim_lite_bc"
    if values.get("training_progress"):
        return "training_bc"
    if values["manifest"]:
        return "training_candidate_bc"
    if values["process"] and values["process"].get("alive"):
        return "generating_teacher_data"
    return "idle_or_failed"


def _gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return {}
    lines = result.stdout.strip().splitlines()
    if not lines:
        return {}
    gpus = []
    for index, line in enumerate(lines):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            gpus.append({"index": index, "raw": line})
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[0],
                "memory_mib": _int(parts[1]),
                "utilization_pct": _int(parts[2]),
            }
        )
    return {"gpus": gpus}


def _process_snapshot(pid: int) -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,etime=,pcpu=,pmem=,cmd="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    line = result.stdout.strip()
    return {"pid": pid, "alive": bool(line), "ps": line}


def _artifact_summary(run_dir: Path) -> list[dict[str, Any]]:
    names = (
        "teacher_data/manifest.json",
        "candidate_strong_bc.pt",
        "candidate_strong_bc.json",
        "xdim_lite_strong_bc.pt",
        "xdim_lite_strong_bc.json",
        "scoreboard_candidate.json",
        "scoreboard_xdim_lite.json",
        "train.log",
    )
    output = []
    for name in names:
        path = run_dir / name
        if path.exists():
            output.append({"path": name, "bytes": path.stat().st_size})
    return output


def _generic_reports(run_dir: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in {
            "dashboard.json",
            "candidate_strong_bc.json",
            "xdim_lite_strong_bc.json",
            "scoreboard_candidate.json",
            "scoreboard_xdim_lite.json",
        }:
            continue
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        metrics = data.get("metrics") or []
        if not isinstance(metrics, list) or not metrics:
            continue
        last = metrics[-1] if metrics else {}
        output.append(
            {
                "path": path.name,
                "arch": data.get("arch"),
                "samples": data.get("samples"),
                "epochs": data.get("epochs"),
                "last_epoch": last.get("epoch"),
                "loss": last.get("loss"),
                "accuracy": last.get("accuracy"),
                "top3_accuracy": last.get("top3_accuracy"),
            }
        )
    return output


def _generic_scoreboards(run_dir: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(run_dir.glob("*.json")):
        data = _read_json(path)
        if not isinstance(data, dict) or "results" not in data:
            continue
        output.append({"path": path.name, "results": _scoreboard_summary(data)})
    return output


def _shards(run_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.name, "bytes": path.stat().st_size}
        for path in sorted((run_dir / "teacher_data").glob("teacher_shard_*.npz*"))
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _sparkline(events: list[dict[str, Any]], key: str) -> str:
    values = [float(event[key]) for event in events if isinstance(event.get(key), (int, float))]
    if not values:
        return '<span class="muted">no samples yet</span>'
    width, height, pad = 520, 120, 6
    low, high = min(values), max(values)
    span = high - low or 1.0
    count = max(1, len(values) - 1)
    points = " ".join(
        f"{pad + (width - 2 * pad) * index / count:.1f},"
        f"{height - pad - (height - 2 * pad) * (value - low) / span:.1f}"
        for index, value in enumerate(values)
    )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(key)} curve"><polyline points="{points}"/></svg>'
        f'<div class="muted">min {_fmt(low)} · max {_fmt(high)} · {len(values)} points</div>'
    )


def _html(status: dict[str, Any], *, refresh_seconds: int) -> str:
    teacher = status["teacher"]
    bc = status["bc"]
    scoreboards = status["scoreboards"]
    artifacts = status["artifacts"]
    gpu = status["gpu"]
    process = status["process"] or {}
    training = status.get("training") or {}
    latest = training.get("latest") or {}
    curve = training.get("curve") or []
    batch = int(latest.get("batch") or 0)
    batches = int(latest.get("batches") or 0)
    gpu_rows = "".join(
        "<tr>"
        f"<td>{_fmt(item.get('index'))}</td>"
        f"<td>{html.escape(str(item.get('name', item.get('raw', 'n/a'))))}</td>"
        f"<td>{_fmt(item.get('utilization_pct'))}%</td>"
        f"<td>{_fmt(item.get('memory_mib'))}</td>"
        "</tr>"
        for item in gpu.get("gpus", [])
    )
    rows = []
    for model_name, summary in bc.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(model_name)}</td>"
            f"<td>{_fmt(summary.get('samples'))}</td>"
            f"<td>{_fmt(summary.get('last_epoch'))}/{_fmt(summary.get('epochs'))}</td>"
            f"<td>{_fmt(summary.get('accuracy'))}</td>"
            f"<td>{_fmt(summary.get('loss'))}</td>"
            f"<td>{html.escape(str(summary.get('checkpoint') or ''))}</td>"
            "</tr>"
        )
    scoreboard_rows = []
    for model_name, items in scoreboards.items():
        for item in items:
            scoreboard_rows.append(
                "<tr>"
                f"<td>{html.escape(model_name)}</td>"
                f"<td>{html.escape(str(item.get('opponent')))}</td>"
                f"<td>{_fmt(item.get('games'))}</td>"
                f"<td>{_fmt(item.get('wins'))}</td>"
                f"<td>{_fmt(item.get('win_rate'))}</td>"
                "</tr>"
            )
    for scoreboard in status.get("scoreboard_files", []):
        for item in scoreboard.get("results", []):
            scoreboard_rows.append(
                "<tr>"
                f"<td>{html.escape(str(scoreboard.get('path')))}</td>"
                f"<td>{html.escape(str(item.get('opponent')))}</td>"
                f"<td>{_fmt(item.get('games'))}</td>"
                f"<td>{_fmt(item.get('wins'))}</td>"
                f"<td>{_fmt(item.get('win_rate'))}</td>"
                "</tr>"
            )
    report_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('path')))}</td>"
        f"<td>{html.escape(str(item.get('arch') or ''))}</td>"
        f"<td>{_fmt(item.get('samples'))}</td>"
        f"<td>{_fmt(item.get('last_epoch'))}/{_fmt(item.get('epochs'))}</td>"
        f"<td>{_fmt(item.get('accuracy'))}</td>"
        f"<td>{_fmt(item.get('top3_accuracy'))}</td>"
        f"<td>{_fmt(item.get('loss'))}</td>"
        "</tr>"
        for item in status.get("reports", [])
    )
    artifact_items = "".join(
        f"<li>{html.escape(item['path'])} ({item['bytes']:,} bytes)</li>"
        for item in artifacts
    )
    shard_items = "".join(
        f"<li>{html.escape(item['path'])} ({item['bytes']:,} bytes)</li>"
        for item in teacher["shards"][-8:]
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{int(refresh_seconds)}">
  <title>CatanZero Teacher Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #17202a; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; background: #fff; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    h2 {{ font-size: 15px; margin: 0 0 10px; }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .muted {{ color: #57606a; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 7px; text-align: left; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
    .spark {{ width: 100%; height: 120px; background: #f6f8fa; border-radius: 6px; }}
    .spark polyline {{ fill: none; stroke: #0969da; stroke-width: 2; vector-effect: non-scaling-stroke; }}
    progress {{ width: 100%; }}
  </style>
</head>
<body>
  <h1>CatanZero Teacher Training Dashboard</h1>
  <p class="muted">Updated {html.escape(status['updated_at'])}. Run: <code>{html.escape(status['run_dir'])}</code></p>
  <div class="grid">
    <div class="card"><h2>Phase</h2><div class="metric">{html.escape(status['phase'])}</div></div>
    <div class="card"><h2>Teacher Games</h2><div class="metric">{teacher['games']:,} / {teacher['target_games']:,}</div><div class="muted">{_pct(teacher['progress'])}</div></div>
    <div class="card"><h2>Teacher Samples</h2><div class="metric">{teacher['samples']:,}</div><div class="muted">{teacher['workers']} workers</div></div>
    <div class="card"><h2>GPU</h2><table><thead><tr><th>#</th><th>Name</th><th>Util</th><th>MiB</th></tr></thead><tbody>{gpu_rows}</tbody></table></div>
    <div class="card"><h2>Process</h2><div class="metric">{'alive' if process.get('alive') else 'not running'}</div><div class="muted">{html.escape(str(process.get('ps', '')))}</div></div>
    <div class="card"><h2>Optimizer Progress</h2><div class="metric">{batch:,} / {batches:,}</div><progress max="{max(1, batches)}" value="{batch}"></progress><div class="muted">{_pct(batch / batches if batches else None)}</div></div>
    <div class="card"><h2>Latest Batch</h2><div class="metric">loss {_fmt(latest.get('loss'))}</div><div class="muted">accuracy {_fmt(latest.get('accuracy'))} · samples {_fmt(latest.get('samples'))}</div></div>
  </div>
  <div class="grid">
    <div class="card"><h2>Batch Loss</h2>{_sparkline(curve, 'loss')}</div>
    <div class="card"><h2>Batch Accuracy</h2>{_sparkline(curve, 'accuracy')}</div>
  </div>
  <h2>Behavior Cloning</h2>
  <table><thead><tr><th>Model</th><th>Samples</th><th>Epoch</th><th>Accuracy</th><th>Loss</th><th>Checkpoint</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>Reports</h2>
  <table><thead><tr><th>File</th><th>Arch</th><th>Samples</th><th>Epoch</th><th>Top1</th><th>Top3</th><th>Loss</th></tr></thead><tbody>{report_rows}</tbody></table>
  <h2>Scoreboards</h2>
  <table><thead><tr><th>Model</th><th>Opponent</th><th>Games</th><th>Wins</th><th>Win Rate</th></tr></thead><tbody>{''.join(scoreboard_rows)}</tbody></table>
  <h2>Teacher Shards</h2>
  <ul>{shard_items}</ul>
  <h2>Artifacts</h2>
  <ul>{artifact_items}</ul>
</body>
</html>
"""


if __name__ == "__main__":
    main()
