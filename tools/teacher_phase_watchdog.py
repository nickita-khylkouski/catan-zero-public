from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any


COUNT_RE = re.compile(
    r"counts\s+bc=(?P<bc>\d+)\s+eval=(?P<eval>\d+)\s+gen=(?P<gen>\d+)\s+curate=(?P<curate>\d+)\s+ppo=(?P<ppo>\d+)"
)
GPU_RE = re.compile(r"^\s*(?P<index>\d+),\s*(?P<util>\d+),\s*(?P<used>\d+),\s*(?P<total>\d+),")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast watchdog for the pre-PPO 35M teacher-training phase."
    )
    parser.add_argument("--modal-run", action="append", default=[])
    parser.add_argument("--modal-timeout", type=int, default=90)
    parser.add_argument("--status-timeout", type=int, default=25)
    parser.add_argument(
        "--state",
        default="runs/ops/teacher_phase_watchdog_state.json",
        help="Local state file used to detect stalled Modal progress.",
    )
    parser.add_argument(
        "--modal-stale-sec",
        type=int,
        default=900,
        help="Warn if a Modal run's observed_samples does not advance for this long.",
    )
    parser.add_argument(
        "--b200-low-gpu-stale-sec",
        type=int,
        default=300,
        help=(
            "Escalate if B200 has useful BC/eval/curation processes but all GPUs "
            "remain below 10 percent utilization for this many seconds."
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    status = _cluster_status(args)
    previous = _read_state(Path(args.state))
    report = _build_report(status, previous, args)
    _write_state(Path(args.state), _state_from_status(status, previous))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    if report["critical"]:
        raise SystemExit(2)


def _cluster_status(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        "python3",
        "tools/cluster_teacher_status.py",
        "--json",
        "--timeout",
        str(args.status_timeout),
        "--modal-timeout",
        str(args.modal_timeout),
    ]
    for run_name in args.modal_run:
        command.extend(["--modal-run", run_name])
    env = os.environ.copy()
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(args.status_timeout + args.modal_timeout + 30, 60),
        env=env,
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": f"invalid JSON from cluster_teacher_status.py: {error}",
        }
    payload["ok"] = True
    return payload


def _build_report(
    status: dict[str, Any],
    previous: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    now = int(time.time())
    critical: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    host_summaries: dict[str, Any] = {}

    if not status.get("ok", False):
        critical.append(
            "cluster status command failed: "
            f"rc={status.get('returncode')} stderr={str(status.get('stderr', '')).strip()}"
        )
        return {
            "ok": False,
            "critical": critical,
            "warnings": warnings,
            "notes": notes,
            "hosts": host_summaries,
            "modal": {},
        }

    for name, item in dict(status.get("hosts", {})).items():
        stdout = str(item.get("stdout", ""))
        counts = _parse_counts(stdout)
        gpus = _parse_gpus(stdout)
        host_summaries[name] = {"counts": counts, "gpus": gpus, "ok": item.get("ok", False)}
        if not item.get("ok", False):
            warnings.append(f"{name}: status failed rc={item.get('returncode')}")
        if counts.get("ppo", 0) > 0:
            critical.append(f"{name}: PPO process count is {counts['ppo']}; teacher phase requires 0")
        if name == "b200":
            useful = counts.get("bc", 0) + counts.get("eval", 0) + counts.get("curate", 0)
            fresh_activity = _has_fresh_activity(stdout, max_age_sec=180)
            missing_pointers = {
                line.split(" ", 1)[1].strip()
                for line in stdout.splitlines()
                if line.startswith("pointer_missing ")
            }
            for pointer in sorted(missing_pointers):
                if pointer in {
                    "runs/teacher/current_35m_training_dataset.txt",
                    "runs/teacher/current_best_35m_checkpoint.txt",
                }:
                    warnings.append(f"b200: missing production pointer {pointer}")
            train_gate_warning_keys: set[str] = set()
            for line in stdout.splitlines():
                if "tools/train_bc.py" not in line:
                    continue
                command_key = _train_command_key(line)
                if command_key in train_gate_warning_keys:
                    continue
                train_gate_warning_keys.add(command_key)
                if "--require-production-35m-teacher" not in line and "--require-strict-35m-teacher" not in line:
                    critical.append(
                        "b200: active train_bc.py is missing a strict/production teacher-data gate"
                    )
                elif "--require-production-35m-teacher" not in line and "diagnostic" not in line.lower():
                    warnings.append(
                        "b200: active train_bc.py is not using --require-production-35m-teacher "
                        "and is not labeled diagnostic"
                    )
            for line in stdout.splitlines():
                if line.startswith("scoreboard_error "):
                    critical.append(f"b200: failed scoreboard detected: {line}")
                elif line.startswith("scoreboard_no_json_dead "):
                    warnings.append(f"b200: scoreboard has no JSON and no live pid: {line}")
                elif line.startswith("train_error "):
                    critical.append(f"b200: failed training log detected: {line}")
            if useful <= 0:
                warnings.append("b200: no BC/eval/curation process detected")
            previous_host = dict(previous.get("hosts", {}).get("b200", {}))
            if gpus and useful > 0 and all(int(gpu["util"]) < 10 for gpu in gpus):
                low_seen_at = int(previous_host.get("low_gpu_all_seen_at", now))
                if low_seen_at <= 0:
                    low_seen_at = now
                low_for = now - low_seen_at
                message = (
                    "b200: useful process exists but all GPU utilization is under 10% "
                    f"for {low_for}s"
                )
                if fresh_activity:
                    warnings.append(message + "; recent train/scoreboard logs are still moving")
                elif low_for >= int(args.b200_low_gpu_stale_sec):
                    critical.append(message)
                else:
                    warnings.append(message)
            elif gpus and useful > 0:
                idle = [str(gpu["index"]) for gpu in gpus if int(gpu["util"]) < 10]
                if idle:
                    idle_seen = dict(previous_host.get("idle_gpu_seen_at", {}))
                    stale: list[str] = []
                    fresh: list[str] = []
                    for gpu_index in idle:
                        seen_at = int(idle_seen.get(gpu_index, now))
                        if seen_at <= 0:
                            seen_at = now
                        if now - seen_at >= int(args.b200_low_gpu_stale_sec):
                            stale.append(f"{gpu_index} for {now - seen_at}s")
                        else:
                            fresh.append(f"{gpu_index} for {now - seen_at}s")
                    if stale:
                        critical.append(
                            "b200: useful process exists but GPU(s) "
                            + ", ".join(stale)
                            + " are under 10% utilization"
                        )
                    if fresh:
                        warnings.append(
                            "b200: useful process exists but GPU(s) "
                            + ", ".join(fresh)
                            + " are under 10% utilization"
                        )
        if name in {"a100", "gh200"} and counts.get("gen", 0) <= 0:
            warnings.append(f"{name}: no teacher generation process detected")
        stale_empty = [line for line in stdout.splitlines() if line.startswith("empty_stale")]
        if stale_empty:
            warnings.append(f"{name}: {len(stale_empty)} empty stale logs reported")

    modal_report: dict[str, Any] = {}
    for run_name, item in dict(status.get("modal", {})).items():
        summary = _extract_json_object(str(item.get("stdout", "")))
        modal_report[run_name] = summary or {"ok": False}
        if not item.get("ok", False):
            warnings.append(f"modal {run_name}: status failed rc={item.get('returncode')}")
            continue
        if not summary:
            warnings.append(f"modal {run_name}: no JSON summary parsed")
            continue
        if int(summary.get("partial_invalid_teacher_actions", 0)) > 0:
            critical.append(
                f"modal {run_name}: partial invalid teacher actions="
                f"{summary.get('partial_invalid_teacher_actions')}"
            )
        if int(summary.get("parts_partial", 0)) > 0:
            partial_mixed = set(bool(value) for value in summary.get("partial_mixed_seats", ()))
            if partial_mixed and partial_mixed != {True}:
                warnings.append(
                    f"modal {run_name}: partial mixed_seats is {sorted(partial_mixed)}, expected [True]"
                )
            partial_modes = set(str(value) for value in summary.get("partial_mixed_seat_modes", ()))
            if partial_modes and partial_modes != {"random"}:
                warnings.append(
                    f"modal {run_name}: partial mixed_seat_modes is {sorted(partial_modes)}, expected ['random']"
                )
            if float(summary.get("partial_soft_score_fraction", 0.0)) < 0.50:
                warnings.append(
                    f"modal {run_name}: partial soft score fraction is "
                    f"{summary.get('partial_soft_score_fraction')}"
                )
            if float(summary.get("partial_final_actual_vp_fraction", 0.0)) not in (0.0, 1.0):
                warnings.append(
                    f"modal {run_name}: partial final actual VP fraction is "
                    f"{summary.get('partial_final_actual_vp_fraction')}"
                )
        if int(summary.get("invalid_teacher_actions", 0)) > 0:
            critical.append(
                f"modal {run_name}: invalid teacher actions={summary.get('invalid_teacher_actions')}"
            )
        if int(summary.get("parts_complete", 0)) > 0:
            completed_mixed = set(bool(value) for value in summary.get("mixed_seats", ()))
            if completed_mixed and completed_mixed != {True}:
                warnings.append(
                    f"modal {run_name}: completed mixed_seats is {sorted(completed_mixed)}, expected [True]"
                )
            elif not completed_mixed:
                warnings.append(
                    f"modal {run_name}: completed parts do not expose mixed_seats metadata"
                )
            completed_modes = set(str(value) for value in summary.get("mixed_seat_modes", ()))
            if completed_modes and completed_modes != {"random"}:
                warnings.append(
                    f"modal {run_name}: completed mixed_seat_modes is {sorted(completed_modes)}, expected ['random']"
                )
            elif not completed_modes:
                warnings.append(
                    f"modal {run_name}: completed parts do not expose mixed_seat_mode metadata"
                )
            if (
                int(summary.get("samples", 0)) > 0
                and float(summary.get("soft_score_fraction", 0.0)) < 0.50
            ):
                warnings.append(
                    f"modal {run_name}: completed soft score fraction is "
                    f"{summary.get('soft_score_fraction')}"
                )
        observed = int(summary.get("observed_samples", 0))
        observed_games = int(summary.get("observed_games", 0))
        complete_parts = int(summary.get("parts_complete", 0))
        partial_parts = int(summary.get("parts_partial", 0))
        if complete_parts == 0 and partial_parts == 0 and observed == 0 and observed_games == 0:
            warnings.append(
                f"modal {run_name}: listed as active but has no completed or partial work"
            )
        if complete_parts >= 75 and partial_parts == 0:
            notes.append(
                f"modal {run_name}: complete parts={complete_parts} observed_samples={observed}"
            )
            continue
        previous_run = dict(previous.get("modal", {}).get(run_name, {}))
        previous_observed = int(previous_run.get("observed_samples", -1))
        previous_seen_at = int(previous_run.get("seen_at", now))
        if observed == previous_observed and now - previous_seen_at >= int(args.modal_stale_sec):
            warnings.append(
                f"modal {run_name}: observed_samples stuck at {observed} for "
                f"{now - previous_seen_at}s"
            )
        elif observed != previous_observed:
            notes.append(f"modal {run_name}: observed_samples={observed}")

    return {
        "ok": not critical,
        "critical": critical,
        "warnings": warnings,
        "notes": notes,
        "hosts": host_summaries,
        "modal": modal_report,
    }


def _parse_counts(stdout: str) -> dict[str, int]:
    match = COUNT_RE.search(stdout)
    if not match:
        return {}
    return {key: int(value) for key, value in match.groupdict().items()}


def _train_command_key(line: str) -> str:
    for flag in ("--report", "--checkpoint", "--data"):
        match = re.search(rf"{re.escape(flag)}\s+(\S+)", line)
        if match:
            return f"{flag}={match.group(1)}"
    return line


def _has_fresh_activity(stdout: str, *, max_age_sec: int) -> bool:
    for line in stdout.splitlines():
        if not line.startswith("age_sec="):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            age = int(parts[0].split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        if age > max_age_sec:
            continue
        if any("path=runs/scoreboards/" in part or "path=runs/teacher/" in part for part in parts):
            return True
    return False


def _parse_gpus(stdout: str) -> list[dict[str, int]]:
    gpus: list[dict[str, int]] = []
    for line in stdout.splitlines():
        match = GPU_RE.match(line)
        if not match:
            continue
        gpus.append({key: int(value) for key, value in match.groupdict().items()})
    return gpus


def _extract_json_object(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return None


def _read_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _state_from_status(status: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    now = int(time.time())
    modal: dict[str, Any] = {}
    hosts: dict[str, Any] = {}
    for name, item in dict(status.get("hosts", {})).items():
        if name != "b200":
            continue
        stdout = str(item.get("stdout", ""))
        counts = _parse_counts(stdout)
        gpus = _parse_gpus(stdout)
        useful = counts.get("bc", 0) + counts.get("eval", 0) + counts.get("curate", 0)
        previous_host = dict(previous.get("hosts", {}).get(name, {}))
        low_now = bool(gpus and useful > 0 and all(int(gpu["util"]) < 10 for gpu in gpus))
        if low_now:
            low_seen_at = int(previous_host.get("low_gpu_all_seen_at", now))
            if low_seen_at <= 0:
                low_seen_at = now
            hosts[name] = {"low_gpu_all_seen_at": low_seen_at, "idle_gpu_seen_at": {}}
        else:
            previous_idle = dict(previous_host.get("idle_gpu_seen_at", {}))
            idle_now = {
                str(gpu["index"]): int(previous_idle.get(str(gpu["index"]), now)) or now
                for gpu in gpus
                if useful > 0 and int(gpu["util"]) < 10
            }
            hosts[name] = {"low_gpu_all_seen_at": 0, "idle_gpu_seen_at": idle_now}
    for run_name, item in dict(status.get("modal", {})).items():
        summary = _extract_json_object(str(item.get("stdout", "")))
        if not summary:
            continue
        observed_samples = int(summary.get("observed_samples", 0))
        observed_games = int(summary.get("observed_games", 0))
        previous_run = dict(previous.get("modal", {}).get(run_name, {}))
        previous_samples = int(previous_run.get("observed_samples", -1))
        previous_games = int(previous_run.get("observed_games", -1))
        if observed_samples == previous_samples and observed_games == previous_games:
            seen_at = int(previous_run.get("seen_at", now))
        else:
            seen_at = now
        modal[run_name] = {
            "observed_samples": observed_samples,
            "observed_games": observed_games,
            "parts_complete": int(summary.get("parts_complete", 0)),
            "seen_at": seen_at,
        }
    return {"updated_at": now, "hosts": hosts, "modal": modal}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_report(report: dict[str, Any]) -> None:
    print("teacher phase watchdog")
    for label in ("critical", "warnings", "notes"):
        values = list(report.get(label, []))
        print(f"{label}: {len(values)}")
        for value in values:
            print(f"  - {value}")
    print("hosts:")
    for name, summary in sorted(dict(report.get("hosts", {})).items()):
        print(f"  - {name}: counts={summary.get('counts', {})} gpus={summary.get('gpus', [])}")


if __name__ == "__main__":
    main()
