from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


DEFAULT_HOSTS_CONFIG = Path("configs/gpu_cluster_hosts.json")
DEFAULT_MODAL_RUNS_FILE = Path("runs/ops/current_modal_teacher_runs.txt")


REMOTE_STATUS = r"""
cd /home/ubuntu/catan-zero 2>/dev/null || exit 2
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "uptime=$(uptime)"
count_pattern() {
  pattern="$1"
  pgrep -af "$pattern" | awk -v self="$$" '$1 != self' | grep -v pgrep | wc -l
}
bc_count="$(count_pattern 'tools/train_bc.py|torchrun.*train_bc.py|torch.distributed.run.*tools/train_bc.py')"
eval_count="$(count_pattern 'tools/evaluate_scoreboard.py')"
gen_count="$(count_pattern 'tools/generate_teacher_data.py')"
curate_count="$(count_pattern 'tools/curate_teacher_data.py')"
ppo_count="$(count_pattern 'train_selfplay_gpu.py|train_ppo.py')"
printf "counts bc=%s eval=%s gen=%s curate=%s ppo=%s\n" \
  "$bc_count" \
  "$eval_count" \
  "$gen_count" \
  "$curate_count" \
  "$ppo_count"
echo "gpu:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null || true
echo "active_processes:"
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | awk -v self="$$" '$1 != self' | grep -E 'tools/train_bc.py|torchrun.*train_bc.py|torch.distributed.run.*tools/train_bc.py|tools/evaluate_scoreboard.py|tools/generate_teacher_data.py|tools/curate_teacher_data.py|train_selfplay_gpu.py|train_ppo.py' | grep -v grep | head -20 || true
echo "latest_teacher:"
find runs/teacher -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5 | cut -d' ' -f2-
echo "latest_scoreboards:"
find runs/scoreboards -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5 | cut -d' ' -f2-
echo "pointers:"
for pointer in \
  runs/teacher/current_35m_training_dataset.txt \
  runs/teacher/current_best_35m_checkpoint.txt \
  runs/teacher/current_b200_35m_teacher_run.txt \
  runs/scoreboards/current_b200_scoreboard_run.txt
do
  if [ -f "$pointer" ]; then
    printf 'pointer %s=' "$pointer"
    head -n 1 "$pointer"
  else
    echo "pointer_missing $pointer"
  fi
done
echo "latest_train_tail:"
train_run="$(cat runs/teacher/current_b200_35m_teacher_run.txt 2>/dev/null || true)"
if { [ -n "$train_run" ] && [ ! -f "$train_run/train.log" ]; } || [ "$bc_count" -gt 0 ]; then
  train_run="$(find runs/teacher -mindepth 2 -maxdepth 2 -type f -name 'train*.log' -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [ -n "$train_run" ] && [ -d "$train_run" ]; then
  train_log="$(find "$train_run" -maxdepth 1 -type f -name 'train*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  [ -n "$train_log" ] || train_log="$train_run/train.log"
  echo "train_run=$train_run"
  echo "train_log=$train_log"
  [ -f "$train_log" ] && tail -n 3 "$train_log"
fi
echo "latest_scoreboard_tail:"
score_run="$(cat runs/scoreboards/current_b200_scoreboard_run.txt 2>/dev/null || true)"
if { [ -n "$score_run" ] && [ ! -d "$score_run" ]; } || [ "$eval_count" -gt 0 ]; then
  score_run="$(find runs/scoreboards -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [ -n "$score_run" ] && [ -d "$score_run" ]; then
  echo "scoreboard_run=$score_run"
  for log in "$score_run"/*.log; do [ -f "$log" ] && echo "log=$log" && tail -n 2 "$log"; done
fi
echo "scoreboard_health:"
active_eval_args="$(ps -eo args | grep 'tools/evaluate_scoreboard.py' | grep -v grep || true)"
find runs/scoreboards -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -8 | cut -d' ' -f2- | while read -r d; do
  [ -n "$d" ] || continue
  if [ -f "$d/quarantined_by_watchdog.txt" ]; then
    printf 'scoreboard_quarantined path=%s reason=%s\n' "$d" "$(head -n 1 "$d/quarantined_by_watchdog.txt" 2>/dev/null)"
    continue
  fi
  json_count="$(find "$d" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
  running=0
  for pid_file in "$d"/*.pid "$d"/pid "$d"/watcher.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(head -n 1 "$pid_file" 2>/dev/null | tr -cd '0-9')"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && running=1
  done
  if printf '%s\n' "$active_eval_args" | grep -F -- "$d/" >/dev/null; then
    running=1
  fi
  error_line="$(grep -Eihm1 'Traceback|unbound variable|unrecognized arguments|checkpoint_missing|No such file|ModuleNotFoundError|RuntimeError|ValueError|error:' "$d"/*.log 2>/dev/null || true)"
  if [ -n "$error_line" ]; then
    printf 'scoreboard_error path=%s json=%s running=%s error=%s\n' "$d" "$json_count" "$running" "$error_line"
  elif [ "$json_count" -eq 0 ] && [ "$running" -eq 0 ]; then
    printf 'scoreboard_no_json_dead path=%s json=%s running=%s\n' "$d" "$json_count" "$running"
  else
    printf 'scoreboard_ok path=%s json=%s running=%s\n' "$d" "$json_count" "$running"
  fi
done
echo "train_health:"
active_train_args="$(ps -eo args | grep -E 'tools/train_bc.py|torchrun.*train_bc.py|torch.distributed.run.*tools/train_bc.py' | grep -v grep || true)"
find runs/teacher -mindepth 2 -maxdepth 2 -type f -name 'train*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -8 | cut -d' ' -f2- | while read -r log; do
  [ -n "$log" ] || continue
  run_dir="$(dirname "$log")"
  running=0
  if printf '%s\n' "$active_train_args" | grep -F -- "$run_dir/" >/dev/null; then
    running=1
  fi
  error_line="$(grep -Eihm1 'teacher data quality failed|Traceback|ChildFailedError|CUDA out of memory|RuntimeError|ValueError|error:' "$log" 2>/dev/null || true)"
  if [ -n "$error_line" ] && [ "$running" -eq 0 ]; then
    printf 'train_error path=%s running=%s error=%s\n' "$log" "$running" "$error_line"
  else
    printf 'train_ok path=%s running=%s\n' "$log" "$running"
  fi
done
echo "recent_logs:"
find runs/teacher runs/scoreboards -maxdepth 2 -type f \( -name '*.log' -o -name '*.json' \) -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -12 | awk '{age=systime()-int($1); printf "age_sec=%s size=%s path=%s\n", age, $2, $3}'
echo "empty_or_stale_logs:"
find runs/teacher runs/scoreboards -maxdepth 2 -type f -name '*.log' -mmin +10 -printf '%T@ %s %p\n' 2>/dev/null | awk '$2 == 0 {printf "empty_stale size=%s path=%s\n", $2, $3} $2 > 0 {printf "stale size=%s path=%s\n", $2, $3}' | head -12
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot teacher-phase cluster status.")
    parser.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/gpu_access_ed25519"))
    parser.add_argument("--hosts-config", default=str(DEFAULT_HOSTS_CONFIG))
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--modal-run",
        action="append",
        default=[],
        help=(
            "Modal teacher run name to summarize. Can be repeated. If omitted, "
            "CATAN_ZERO_MODAL_RUNS or --modal-runs-file may provide names."
        ),
    )
    parser.add_argument(
        "--modal-runs-file",
        default=str(DEFAULT_MODAL_RUNS_FILE),
        help="Text file with one active Modal run name per line.",
    )
    parser.add_argument("--modal-timeout", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    started = time.perf_counter()
    hosts = _load_hosts(Path(args.hosts_config))
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {
            executor.submit(_host_status, name, spec, ssh_key=args.ssh_key, timeout=args.timeout): name
            for name, spec in hosts.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as error:
                results[name] = {"ok": False, "error": str(error)}

    modal_runs = _modal_run_names(args.modal_run, Path(args.modal_runs_file))
    modal_results: dict[str, Any] = {}
    if modal_runs:
        with ThreadPoolExecutor(max_workers=min(4, len(modal_runs))) as executor:
            futures = {
                executor.submit(_modal_status, run_name, timeout=args.modal_timeout): run_name
                for run_name in modal_runs
            }
            for future in as_completed(futures):
                run_name = futures[future]
                try:
                    modal_results[run_name] = future.result()
                except Exception as error:
                    modal_results[run_name] = {"ok": False, "error": str(error)}

    payload = {
        "elapsed_sec": time.perf_counter() - started,
        "hosts": results,
        "host_names": list(hosts),
        "modal": modal_results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(payload)


def _host_status(name: str, spec: dict[str, str], *, ssh_key: str, timeout: int) -> dict[str, Any]:
    command = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if spec.get("known_hosts"):
        command.extend(["-o", f"UserKnownHostsFile={spec['known_hosts']}"])
    command.extend([spec["host"], REMOTE_STATUS])
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(1, timeout),
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "role": name,
    }


def _load_hosts(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    user = str(data.get("user", "ubuntu"))
    hosts: dict[str, dict[str, str]] = {}
    for entry in data.get("hosts", []):
        raw_name = str(entry["name"])
        name = _display_host_name(raw_name)
        host = str(entry["host"])
        if "@" not in host:
            host = f"{user}@{host}"
        hosts[name] = {
            "host": host,
            "known_hosts": str(entry.get("known_hosts", "")),
        }
    if not hosts:
        raise SystemExit(f"no hosts found in {path}")
    return hosts


def _display_host_name(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("b200"):
        return "b200"
    if lowered.startswith("a100"):
        return "a100"
    if lowered.startswith("gh200"):
        return "gh200"
    return name


def _modal_run_names(values: list[str], runs_file: Path) -> list[str]:
    import os

    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    env_value = os.environ.get("CATAN_ZERO_MODAL_RUNS", "")
    if env_value:
        names.extend(part.strip() for part in env_value.split(",") if part.strip())
    try:
        for line in runs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.extend(part.strip() for part in line.split(",") if part.strip())
    except FileNotFoundError:
        pass
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def _modal_status(run_name: str, *, timeout: int) -> dict[str, Any]:
    command = [
        "python3",
        "-m",
        "modal",
        "run",
        "tools/modal_teacher_factory.py::status",
        "--run-name",
        run_name,
    ]
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(1, timeout),
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "role": "modal",
    }


def _print_text(payload: dict[str, Any]) -> None:
    print(f"teacher cluster status ({payload['elapsed_sec']:.1f}s)")
    for name in payload.get("host_names", []):
        item = payload["hosts"].get(name, {})
        print(f"\n== {name} ==")
        if not item.get("ok"):
            print(f"ERROR rc={item.get('returncode')} {item.get('error', '')}")
            if item.get("stderr"):
                print(item["stderr"].strip())
        print((item.get("stdout") or "").strip())
    if payload.get("modal"):
        print("\n== modal ==")
        for run_name, item in sorted(payload["modal"].items()):
            print(f"-- {run_name} --")
            if not item.get("ok"):
                print(f"ERROR rc={item.get('returncode')} {item.get('error', '')}")
                if item.get("stderr"):
                    print(item["stderr"].strip())
            print((item.get("stdout") or "").strip())


if __name__ == "__main__":
    main()
