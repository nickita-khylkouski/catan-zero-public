from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


DEFAULT_HOSTS_CONFIG = Path("configs/gpu_cluster_hosts.json")
DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/gpu_access_ed25519")


REMOTE_COMMANDS = {
    "b200-train": r"""
	cd /home/ubuntu/catan-zero || exit 2
	run="$(cat runs/teacher/current_b200_35m_teacher_run.txt 2>/dev/null || true)"
	bc_count="$(pgrep -af 'tools/train_bc.py|torchrun.*train_bc.py|torch.distributed.run.*tools/train_bc.py' | grep -v pgrep | wc -l)"
	if [ -z "$run" ] || [ ! -d "$run" ] || [ "$bc_count" -gt 0 ]; then
	  run="$(find runs/teacher -mindepth 2 -maxdepth 2 -type f -name 'train*.log' -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
	fi
	test -n "$run" || { echo "no train.log found"; exit 1; }
	log="$(find "$run" -maxdepth 1 -type f -name 'train*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
	[ -n "$log" ] || log="$run/train.log"
	echo "train_run=$run"
	echo "train_log=$log"
	tail __TAIL_FLAGS__ "$log"
	""",
    "b200-scoreboard": r"""
	cd /home/ubuntu/catan-zero || exit 2
	run="$(cat runs/scoreboards/current_b200_scoreboard_run.txt 2>/dev/null || true)"
	eval_count="$(pgrep -af 'tools/evaluate_scoreboard.py' | grep -v pgrep | wc -l)"
	if [ -z "$run" ] || [ ! -d "$run" ] || [ "$eval_count" -gt 0 ]; then
	  run="$(find runs/scoreboards -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
	fi
test -n "$run" || { echo "no scoreboard run found"; exit 1; }
echo "scoreboard_run=$run"
for log in "$run"/*.log; do
  [ -f "$log" ] && echo "---- $log" && tail -n __LINES__ "$log"
done
for report in "$run"/*.json; do
  [ -f "$report" ] && echo "---- $report" && cat "$report"
done
true
""",
    "a100-generate": r"""
cd /home/ubuntu/catan-zero || exit 2
run="$(find runs/teacher -mindepth 2 -maxdepth 2 -type f -name generate.log -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$run" || { echo "no generate.log found"; exit 1; }
echo "teacher_run=$run"
tail __TAIL_FLAGS__ "$run/generate.log"
""",
    "gh200-generate": r"""
cd /home/ubuntu/catan-zero || exit 2
run="$(find runs/teacher -mindepth 2 -maxdepth 2 -type f -name generate.log -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$run" || { echo "no generate.log found"; exit 1; }
echo "teacher_run=$run"
tail __TAIL_FLAGS__ "$run/generate.log"
""",
}


TARGET_HOST = {
    "b200-train": "b200",
    "b200-scoreboard": "b200",
    "a100-generate": "a100",
    "gh200-generate": "gh200",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tail the active 35M teacher-training logs on the remote boxes."
    )
    parser.add_argument(
        "target",
        choices=sorted(REMOTE_COMMANDS),
        help="Which active log to tail.",
    )
    parser.add_argument("--lines", type=int, default=80)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--hosts-config", default=str(DEFAULT_HOSTS_CONFIG))
    args = parser.parse_args()

    hosts = _load_hosts(Path(args.hosts_config))
    host = hosts[TARGET_HOST[args.target]]
    tail_flags = f"-n {max(1, int(args.lines))}"
    if args.follow:
        tail_flags += " -f"
    remote = (
        REMOTE_COMMANDS[args.target]
        .replace("__LINES__", str(max(1, int(args.lines))))
        .replace("__TAIL_FLAGS__", tail_flags)
    )
    command = [
        "ssh",
        "-i",
        args.ssh_key,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if host.get("known_hosts"):
        command.extend(["-o", f"UserKnownHostsFile={host['known_hosts']}"])
    command.extend([host["host"], remote])
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    raise SystemExit(subprocess.call(command))


def _load_hosts(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    user = str(data.get("user", "ubuntu"))
    hosts: dict[str, dict[str, str]] = {}
    for entry in data.get("hosts", []):
        name = _display_host_name(str(entry["name"]))
        host = str(entry["host"])
        if "@" not in host:
            host = f"{user}@{host}"
        hosts[name] = {
            "host": host,
            "known_hosts": str(entry.get("known_hosts", "")),
        }
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


if __name__ == "__main__":
    main()
