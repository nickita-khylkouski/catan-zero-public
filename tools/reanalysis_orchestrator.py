from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.league_orchestrator import gate_checkpoints, read_manifest, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll, pull, and gate search-reanalysis CatanZero branches.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    poll_parser = subparsers.add_parser("poll")
    poll_parser.add_argument(
        "--manifest",
        default="runs/self_play/reanalysis_v1/manifest.json",
    )

    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument(
        "--manifest",
        default="runs/self_play/reanalysis_v1/manifest.json",
    )
    pull_parser.add_argument("--output-dir", default="runs/self_play/box_imports")
    pull_parser.add_argument(
        "--include-jsonl",
        action="store_true",
        help="Also pull generated search JSONL files. Usually not needed for gating.",
    )

    gate_parser = subparsers.add_parser("gate-ready")
    gate_parser.add_argument(
        "--manifest",
        default="runs/self_play/reanalysis_v1/manifest.json",
    )
    gate_parser.add_argument("--output-dir", default="runs/self_play/box_imports")
    gate_parser.add_argument("--eval-dir", default="runs/self_play/box_eval_imports")
    gate_parser.add_argument(
        "--champion",
        default="runs/self_play/champions/current_best_s9752_iter0002.pt",
    )
    gate_parser.add_argument("--games", type=int, default=200)
    gate_parser.add_argument("--workers", type=int, default=24)
    gate_parser.add_argument("--vps-to-win", type=int, default=10)
    gate_parser.add_argument("--max-decisions", type=int, default=1200)
    gate_parser.add_argument("--common-heuristic-seed", type=int, default=85001)
    gate_parser.add_argument("--common-value-seed", type=int, default=85002)
    gate_parser.add_argument("--verify-heuristic-seed", type=int, default=85101)
    gate_parser.add_argument("--verify-value-seed", type=int, default=85102)
    gate_parser.add_argument("--champion-heuristic-wins", type=int, default=18)
    gate_parser.add_argument("--champion-value-wins", type=int, default=11)
    gate_parser.add_argument(
        "--skip-evaluate-champion",
        dest="evaluate_champion",
        action="store_false",
    )
    gate_parser.set_defaults(evaluate_champion=True)
    gate_parser.add_argument("--promote-if-better", action="store_true")
    gate_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    manifest = read_manifest(Path(args.manifest))
    if args.command == "poll":
        print(json.dumps(poll_reanalysis_manifest(manifest), indent=2, sort_keys=True))
    elif args.command == "pull":
        pulled = pull_reanalysis_manifest(
            manifest,
            Path(args.output_dir),
            include_jsonl=args.include_jsonl,
        )
        print(json.dumps({"pulled": pulled}, indent=2, sort_keys=True))
    elif args.command == "gate-ready":
        pulled = pull_reanalysis_manifest(
            manifest,
            Path(args.output_dir),
            include_jsonl=False,
        )
        gate_args = argparse.Namespace(
            checkpoint=pulled["checkpoints"],
            eval_dir=args.eval_dir,
            champion=args.champion,
            games=args.games,
            workers=args.workers,
            vps_to_win=args.vps_to_win,
            max_decisions=args.max_decisions,
            common_heuristic_seed=args.common_heuristic_seed,
            common_value_seed=args.common_value_seed,
            verify_heuristic_seed=args.verify_heuristic_seed,
            verify_value_seed=args.verify_value_seed,
            champion_heuristic_wins=args.champion_heuristic_wins,
            champion_value_wins=args.champion_value_wins,
            evaluate_champion=args.evaluate_champion,
            promote_if_better=args.promote_if_better,
            dry_run=args.dry_run,
        )
        gated = gate_checkpoints(gate_args) if pulled["checkpoints"] else []
        print(json.dumps({"pulled": pulled, "gated": gated}, indent=2, sort_keys=True))
    else:  # pragma: no cover - argparse enforces subcommands.
        raise ValueError(args.command)


def poll_reanalysis_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for branch in manifest.get("branches", []):
        result = subprocess.run(
            ["box", "ssh", branch["box"], build_remote_poll_command(branch)],
            check=False,
            text=True,
            capture_output=True,
        )
        rows.append(
            {
                "name": branch["name"],
                "box": branch["box"],
                "returncode": result.returncode,
                **parse_reanalysis_poll_stdout(result.stdout),
                "stderr": result.stderr,
            }
        )
    return rows


def build_remote_poll_command(branch: dict[str, Any]) -> str:
    remote_dir = shlex.quote(branch["remote_dir"])
    checkpoint = shlex.quote(branch["checkpoint"])
    report = shlex.quote(branch["report"])
    log = shlex.quote(branch["log"])
    jsonl = shlex.quote(branch["checkpoint"].removesuffix(".pt") + ".jsonl")
    process_pattern = shlex.quote(branch["name"])
    return (
        f"cd {remote_dir} && "
        f"(pgrep -af {process_pattern} || true); "
        "printf '__ARTIFACTS__\\n'; "
        f"for f in {checkpoint} {report} {jsonl}; do "
        "test -f \"$f\" && printf '%s %s %s\\n' \"$f\" \"$(wc -c < \"$f\")\" \"$(wc -l < \"$f\")\"; "
        "done; "
        "printf '__LOG__\\n'; "
        f"tail -n 5 {log} 2>/dev/null || true"
    )


def parse_reanalysis_poll_stdout(stdout: str) -> dict[str, Any]:
    process_text, _, rest = stdout.partition("__ARTIFACTS__\n")
    artifact_text, _, log_text = rest.partition("__LOG__\n")
    processes = [
        line
        for line in process_text.splitlines()
        if (
            ("generate_reanalysis.py" in line or "train_ppo.py" in line)
            and "pgrep -af" not in line
            and is_live_python_worker(line)
        )
    ]
    artifacts = []
    for line in artifact_text.splitlines():
        parts = line.rsplit(" ", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            artifacts.append(
                {
                    "file": parts[0],
                    "bytes": int(parts[1]),
                    "lines": int(parts[2]),
                }
            )
    return {
        "running": any(".venv/bin/python" in line for line in processes),
        "phase": infer_phase(processes),
        "processes": processes,
        "artifacts": artifacts,
        "checkpoints": [item for item in artifacts if item["file"].endswith(".pt")],
        "reports": [item for item in artifacts if item["file"].endswith(".json")],
        "jsonl": [item for item in artifacts if item["file"].endswith(".jsonl")],
        "log_tail": log_text.splitlines(),
    }


def infer_phase(processes: list[str]) -> str:
    if any("train_ppo.py" in line and is_live_python_worker(line) for line in processes):
        return "train"
    if any("generate_reanalysis.py" in line and is_live_python_worker(line) for line in processes):
        return "generate"
    return "idle"


def is_live_python_worker(line: str) -> bool:
    fields = line.strip().split(maxsplit=1)
    if len(fields) != 2 or not fields[0].isdigit():
        return False
    command = fields[1]
    return command.startswith(".venv/bin/python ") or command.startswith(
        "/tmp/",
    ) and "/.venv/bin/python " in command


def pull_reanalysis_manifest(
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    include_jsonl: bool,
) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled = {"checkpoints": [], "reports": [], "jsonl": []}
    poll_rows = poll_reanalysis_manifest(manifest)
    branches = {branch["name"]: branch for branch in manifest.get("branches", [])}
    for row in poll_rows:
        branch = branches[row["name"]]
        for checkpoint in row["checkpoints"]:
            pulled["checkpoints"].append(
                _pull_remote_artifact(branch, checkpoint["file"], output_dir)
            )
        for report in row["reports"]:
            pulled["reports"].append(
                _pull_remote_artifact(branch, report["file"], output_dir)
            )
        if include_jsonl:
            for jsonl in row["jsonl"]:
                pulled["jsonl"].append(
                    _pull_remote_artifact(branch, jsonl["file"], output_dir)
                )
    return {key: sorted(set(value)) for key, value in pulled.items()}


def _pull_remote_artifact(
    branch: dict[str, Any],
    remote_file: str,
    output_dir: Path,
) -> str:
    destination = output_dir / Path(remote_file).name
    if destination.exists():
        return str(destination)
    remote_path = f"{branch['box']}:{branch['remote_dir']}/{remote_file}"
    run(["box", "scp", remote_path, str(destination)], dry_run=False)
    return str(destination)


if __name__ == "__main__":
    main()
