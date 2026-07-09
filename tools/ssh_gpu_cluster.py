from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/gpu_cluster_hosts.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate SSH GPU workers without running local training."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=6)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inventory")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("remote_command")

    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("--delete", action="store_true")
    sync_parser.add_argument("--dry-run", action="store_true")

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("--dry-run", action="store_true")

    launch_parser = subparsers.add_parser("launch-train")
    launch_parser.add_argument("--champion", default="runs/self_play/champions/current_best_s9752_iter0002.pt")
    launch_parser.add_argument("--recipe", action="append", default=[])
    launch_parser.add_argument("--seed-base", type=int, default=20000)
    launch_parser.add_argument("--iterations", type=int, default=8)
    launch_parser.add_argument("--episodes-per-iteration", type=int, default=10)
    launch_parser.add_argument("--checkpoint-every", type=int, default=2)
    launch_parser.add_argument("--max-gpus-per-host", type=int, default=0)
    launch_parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=1,
        help=(
            "Launch this many independent train_ppo.py workers per visible GPU. "
            "Useful while game simulation is CPU-bound and the GPU is mostly idle."
        ),
    )
    launch_parser.add_argument(
        "--label-prefix",
        default="gpu",
        help="Prefix for remote checkpoint/report/log labels.",
    )
    launch_parser.add_argument("--dry-run", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--log-lines", type=int, default=20)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    if args.command == "inventory":
        payload = parallel_hosts(config, args.workers, inventory_host)
    elif args.command == "run":
        payload = parallel_hosts(
            config,
            args.workers,
            lambda cfg, host: run_remote(cfg, host, args.remote_command),
        )
    elif args.command == "sync":
        payload = parallel_hosts(
            config,
            args.workers,
            lambda cfg, host: sync_host(cfg, host, delete=args.delete, dry_run=args.dry_run),
        )
    elif args.command == "setup":
        payload = parallel_hosts(
            config,
            args.workers,
            lambda cfg, host: setup_host(cfg, host, dry_run=args.dry_run),
        )
    elif args.command == "launch-train":
        payload = parallel_hosts(
            config,
            args.workers,
            lambda cfg, host: launch_train_host(cfg, host, args),
        )
    elif args.command == "status":
        payload = parallel_hosts(
            config,
            args.workers,
            lambda cfg, host: status_host(cfg, host, log_lines=args.log_lines),
        )
    else:
        raise SystemExit(f"unknown command {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["ssh_key"] = str(Path(config["ssh_key"]).expanduser())
    for index, host in enumerate(config.get("hosts") or []):
        host.setdefault("ordinal", index)
    return config


def parallel_hosts(config: dict[str, Any], workers: int, fn) -> dict[str, Any]:
    hosts = list(config.get("hosts") or [])
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(fn, config, host): host for host in hosts}
        for future in as_completed(futures):
            host = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report per-host failures.
                rows.append(
                    {
                        "name": host.get("name"),
                        "host": host.get("host"),
                        "ok": False,
                        "error": str(exc),
                    }
                )
    rows.sort(key=lambda row: str(row.get("name") or row.get("host")))
    return {"hosts": rows}


def inventory_host(config: dict[str, Any], host: dict[str, Any]) -> dict[str, Any]:
    command = (
        "python3 --version; "
        "printf 'HOSTNAME='; hostname; "
        "printf 'CPU='; nproc; "
        "printf 'RAM_MB='; awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo; "
        "printf 'DISK='; df -h / | tail -1; "
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    row = run_remote(config, host, command)
    row["expected_gpus"] = host.get("expected_gpus")
    row["role"] = host.get("role")
    row["gpu_count"] = sum(
        1
        for line in row.get("stdout", "").splitlines()
        if "NVIDIA" in line and "," in line
    )
    return row


def setup_host(config: dict[str, Any], host: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    remote_repo = shlex.quote(str(config["remote_repo"]))
    command = (
        f"mkdir -p {remote_repo} && cd {remote_repo} && "
        "python3 -m venv --system-site-packages .venv && "
        ". .venv/bin/activate && "
        "python -m pip install --upgrade pip && "
        "python -m pip install 'numpy>=1.26,<2.0' 'gymnasium>=1.0,<1.1' 'networkx>=3.0,<4.0' 'pytest>=8,<9' && "
        "python -m pip install --no-deps --ignore-requires-python -e . && "
        "python - <<'PY'\n"
        "import gymnasium, networkx, numpy, torch\n"
        "print('numpy', numpy.__version__)\n"
        "print('gymnasium', gymnasium.__version__)\n"
        "print('networkx', networkx.__version__)\n"
        "print('torch', torch.__version__)\n"
        "print('cuda_devices', torch.cuda.device_count())\n"
        "raise SystemExit(0 if torch.cuda.is_available() else 1)\n"
        "PY"
    )
    if dry_run:
        return {
            "name": host.get("name"),
            "host": host.get("host"),
            "ok": True,
            "dry_run": True,
            "command": command,
        }
    return run_remote(config, host, command)


def launch_train_host(
    config: dict[str, Any],
    host: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    recipes = args.recipe or [
        "bulk_self_play_guard",
        "bulk_self_play_antireg",
        "bulk_jsettlers_value_repair",
        "bulk_tactical_rollout_guard",
    ]
    expected_gpus = int(host.get("expected_gpus") or 1)
    gpu_count = expected_gpus
    if int(args.max_gpus_per_host) > 0:
        gpu_count = min(gpu_count, int(args.max_gpus_per_host))
    remote_repo = shlex.quote(str(config["remote_repo"]))
    lines = [
        "set -euo pipefail",
        f"cd {remote_repo}",
        "mkdir -p runs/self_play/gpu_logs",
        ". .venv/bin/activate",
    ]
    launched = []
    process_count = max(1, int(args.processes_per_gpu))
    for gpu_index in range(gpu_count):
        for worker_index in range(process_count):
            recipe_index = host_index(host) * process_count + gpu_index * process_count + worker_index
            recipe = recipes[recipe_index % len(recipes)]
            seed = int(args.seed_base) + int(host_index(host)) * 1000 + gpu_index * 100 + worker_index
            label = (
                f"{safe_name(str(args.label_prefix))}_{safe_name(str(host['name']))}"
                f"_g{gpu_index}_w{worker_index}_s{seed}_{recipe}"
            )
            checkpoint = f"runs/self_play/{label}.pt"
            report = f"runs/self_play/{label}.json"
            log = f"runs/self_play/gpu_logs/{label}.log"
            train_args = gpu_train_args(
                seed=seed,
                champion=args.champion,
                recipe=recipe,
                iterations=args.iterations,
                episodes_per_iteration=args.episodes_per_iteration,
                checkpoint_every=args.checkpoint_every,
                checkpoint=checkpoint,
                report=report,
            )
            shell_args = " ".join(shlex.quote(part) for part in train_args)
            lines.append(
                " ".join(
                    [
                        "nohup",
                        "env",
                        f"CUDA_VISIBLE_DEVICES={gpu_index}",
                        "PYTHONUNBUFFERED=1",
                        ".venv/bin/python",
                        "tools/train_ppo.py",
                        shell_args,
                        f">{shlex.quote(log)}",
                        "2>&1",
                        "&",
                        f"echo {shlex.quote(label)}:$!",
                    ]
                )
            )
            launched.append(
                {
                    "gpu": gpu_index,
                    "worker": worker_index,
                    "recipe": recipe,
                    "seed": seed,
                    "label": label,
                    "checkpoint": checkpoint,
                    "report": report,
                    "log": log,
                }
            )
    command = "\n".join(lines)
    if args.dry_run:
        return {
            "name": host.get("name"),
            "host": host.get("host"),
            "ok": True,
            "dry_run": True,
            "launches": launched,
            "remote_command": command,
        }
    row = run_remote(config, host, command)
    row["launches"] = launched
    return row


def gpu_train_args(
    *,
    seed: int,
    champion: str,
    recipe: str,
    iterations: int,
    episodes_per_iteration: int,
    checkpoint_every: int,
    checkpoint: str,
    report: str,
) -> list[str]:
    args = [
        "--seed",
        str(seed),
        "--vps-to-win",
        "4",
        "--max-decisions",
        "300",
        "--init-checkpoint",
        champion,
        "--opponent-checkpoints",
        champion,
        "--device",
        "cuda",
        "--teacher",
        "tactical_rollout_mixed",
        "--warmup-games",
        "4",
        "--warmup-epochs",
        "1",
        "--warmup-value-coef",
        "0.25",
        "--iterations",
        str(iterations),
        "--episodes-per-iteration",
        str(episodes_per_iteration),
        "--learner-seats",
        "one",
        "--opponents",
        "strict_gate_repair_mixed",
        "--training-value-candidate-limit",
        "32",
        "--training-value-opponent-penalty",
        "0.07",
        "--ppo-epochs",
        "2",
        "--minibatch-size",
        "512",
        "--learning-rate",
        "0.00004",
        "--clip-ratio",
        "0.055",
        "--value-coef",
        "0.75",
        "--q-value-coef",
        "0.55",
        "--q-advantage-mix",
        "0.05",
        "--q-expected-sarsa-mix",
        "0.65",
        "--entropy-coef",
        "0.007",
        "--old-policy-kl-coef",
        "0.075",
        "--ema-policy-kl-coef",
        "0.120",
        "--ema-policy-decay",
        "0.9992",
        "--target-kl",
        "0.0045",
        "--anchor-games-per-iteration",
        "4",
        "--dagger-games-per-iteration",
        "4",
        "--anchor-replay-size",
        "8192",
        "--anchor-learning-rate-multiplier",
        "0.35",
        "--teacher-candidate-limit",
        "24",
        "--teacher-presearch-candidate-limit",
        "48",
        "--teacher-rollout-decisions",
        "2",
        "--teacher-rollout-samples",
        "1",
        "--teacher-root-value-weight",
        "0.35",
        "--teacher-temperature",
        "0.38",
        "--imitation-score-coef",
        "0.05",
        "--imitation-hard-target-weight",
        "0.18",
        "--ppo-top-advantage-fraction",
        "0.30",
        "--ppo-min-advantage-samples",
        "64",
        "--q-advantage-warmup-iterations",
        "2",
        "--q-advantage-ramp-iterations",
        "5",
        "--q-advantage-min-sign-agreement",
        "0.64",
        "--q-advantage-min-return-corr",
        "0.14",
        "--dagger-sample-weight",
        "5.0",
        "--dagger-low-return-multiplier",
        "3.5",
        "--dagger-low-return-threshold",
        "0.0",
        "--gae-lambda",
        "0.88",
        "--checkpoint-every",
        str(checkpoint_every),
        "--checkpoint-eval-games",
        "0",
        "--checkpoint-eval-value-games",
        "0",
        "--eval-games",
        "0",
        "--eval-value-games",
        "0",
        "--checkpoint",
        checkpoint,
        "--report",
        report,
    ]
    if recipe == "strict_gate_antireg":
        replace_arg(args, "--warmup-games", "0")
        replace_arg(args, "--learning-rate", "0.000052")
        replace_arg(args, "--clip-ratio", "0.065")
        replace_arg(args, "--q-advantage-mix", "0.07")
        replace_arg(args, "--q-advantage-min-sign-agreement", "0.62")
        replace_arg(args, "--q-advantage-min-return-corr", "0.12")
    elif recipe == "vrpo_jsettlers_value_repair":
        replace_arg(args, "--teacher", "baseline_rollout_mixed")
        replace_arg(args, "--warmup-games", "0")
        replace_arg(args, "--opponents", "jsettlers_value_repair_mixed")
        replace_arg(args, "--learning-rate", "0.000055")
        replace_arg(args, "--clip-ratio", "0.07")
        replace_arg(args, "--q-advantage-mix", "0.08")
        replace_arg(args, "--q-expected-sarsa-mix", "0.60")
    elif recipe == "tactical_rollout_guard_repair":
        replace_arg(args, "--opponents", "anti_regression_mixed")
        replace_arg(args, "--learning-rate", "0.000055")
        replace_arg(args, "--old-policy-kl-coef", "0.05")
        replace_arg(args, "--ema-policy-kl-coef", "0.08")
    elif recipe == "bulk_self_play_guard":
        apply_bulk_self_play_args(args)
        replace_arg(args, "--opponents", "strict_gate_repair_mixed")
    elif recipe == "bulk_self_play_antireg":
        apply_bulk_self_play_args(args)
        replace_arg(args, "--opponents", "anti_regression_mixed")
        replace_arg(args, "--learning-rate", "0.000052")
        replace_arg(args, "--clip-ratio", "0.065")
    elif recipe == "bulk_jsettlers_value_repair":
        apply_bulk_self_play_args(args)
        replace_arg(args, "--teacher", "baseline_rollout_mixed")
        replace_arg(args, "--opponents", "jsettlers_value_repair_mixed")
        replace_arg(args, "--learning-rate", "0.000055")
        replace_arg(args, "--clip-ratio", "0.07")
        replace_arg(args, "--q-expected-sarsa-mix", "0.60")
    elif recipe == "bulk_tactical_rollout_guard":
        apply_bulk_self_play_args(args)
        replace_arg(args, "--opponents", "anti_regression_mixed")
        replace_arg(args, "--learning-rate", "0.000055")
        replace_arg(args, "--old-policy-kl-coef", "0.05")
        replace_arg(args, "--ema-policy-kl-coef", "0.08")
    return args


def apply_bulk_self_play_args(args: list[str]) -> None:
    """Trade expensive teacher collection for more independent self-play workers."""
    replace_arg(args, "--warmup-games", "0")
    replace_arg(args, "--anchor-games-per-iteration", "1")
    replace_arg(args, "--dagger-games-per-iteration", "1")
    replace_arg(args, "--teacher-candidate-limit", "12")
    replace_arg(args, "--teacher-presearch-candidate-limit", "16")
    replace_arg(args, "--teacher-rollout-decisions", "1")
    replace_arg(args, "--teacher-root-value-weight", "0.25")
    replace_arg(args, "--anchor-replay-size", "4096")
    replace_arg(args, "--q-advantage-mix", "0.04")
    replace_arg(args, "--q-expected-sarsa-mix", "0.55")
    replace_arg(args, "--old-policy-kl-coef", "0.045")
    replace_arg(args, "--ema-policy-kl-coef", "0.075")
    replace_arg(args, "--entropy-coef", "0.006")


def replace_arg(args: list[str], key: str, value: str) -> None:
    args[args.index(key) + 1] = value


def host_index(host: dict[str, Any]) -> int:
    if "ordinal" in host:
        return int(host["ordinal"])
    name = str(host.get("name") or "host")
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return int(digits)
    return abs(hash(name)) % 1000


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def sync_host(
    config: dict[str, Any],
    host: dict[str, Any],
    *,
    delete: bool,
    dry_run: bool,
) -> dict[str, Any]:
    remote = f"{config['user']}@{host['host']}:{config['remote_repo'].rstrip('/')}/"
    command = [
        "rsync",
        "-az",
        "--exclude",
        ".git/",
        "--exclude",
        "data/",
        "--exclude",
        ".venv/",
        "--exclude",
        "runs/self_play/",
        "--exclude",
        ".pytest_cache/",
        "--exclude",
        "__pycache__/",
        "-e",
        " ".join(ssh_base(config)),
    ]
    if delete:
        command.append("--delete")
    if dry_run:
        command.append("--dry-run")
    command.extend(["./", remote])
    result = subprocess.run(
        command,
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout = result.stdout
    stderr = result.stderr
    returncode = result.returncode
    champion = Path("runs/self_play/champions/current_best_s9752_iter0002.pt")
    if result.returncode == 0 and champion.exists():
        mkdir_result = subprocess.run(
            [
                *ssh_base(config),
                f"{config['user']}@{host['host']}",
                f"mkdir -p {shlex.quote(config['remote_repo'].rstrip('/'))}/runs/self_play/champions",
            ],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout += mkdir_result.stdout
        stderr += mkdir_result.stderr
        returncode = mkdir_result.returncode
        if returncode != 0:
            return {
                "name": host.get("name"),
                "host": host.get("host"),
                "ok": False,
                "returncode": returncode,
                "stdout": stdout[-4000:],
                "stderr": stderr[-4000:],
            }
        champion_remote_dir = (
            f"{config['user']}@{host['host']}:"
            f"{config['remote_repo'].rstrip('/')}/runs/self_play/champions/"
        )
        champion_command = [
            "rsync",
            "-az",
            "-e",
            " ".join(ssh_base(config)),
        ]
        if dry_run:
            champion_command.append("--dry-run")
        champion_command.extend([str(champion), champion_remote_dir])
        champion_result = subprocess.run(
            champion_command,
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout += champion_result.stdout
        stderr += champion_result.stderr
        returncode = champion_result.returncode
    return {
        "name": host.get("name"),
        "host": host.get("host"),
        "ok": returncode == 0,
        "returncode": returncode,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


def status_host(config: dict[str, Any], host: dict[str, Any], *, log_lines: int) -> dict[str, Any]:
    remote_repo = shlex.quote(str(config["remote_repo"]))
    lines = [
        f"cd {remote_repo}",
        "echo PROCS",
        "ps -eo pid,pcpu,pmem,etime,cmd | grep 'tools/train_ppo.py' | grep -v grep || true",
        "echo GPU",
        "nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true",
        "echo ARTIFACTS",
        (
            "find runs/self_play -maxdepth 1 \\( -name '*.pt' -o -name '*.json' \\) "
            "-printf '%TY-%Tm-%Td %TH:%TM %s %p\\n' 2>/dev/null | sort | tail -20"
        ),
        "echo LOGS",
        (
            "find runs/self_play/gpu_logs -maxdepth 1 -type f -name '*.log' "
            "-printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -12 | "
            "while read _ f; do echo ===$f===; tail "
            f"-{max(1, int(log_lines))} \"$f\"; done"
        ),
    ]
    return run_remote(config, host, " && ".join(lines))


def run_remote(
    config: dict[str, Any],
    host: dict[str, Any],
    remote_command: str,
) -> dict[str, Any]:
    destination = f"{config['user']}@{host['host']}"
    command = [*ssh_base(config), destination, remote_command]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "name": host.get("name"),
        "host": host.get("host"),
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
    }


def ssh_base(config: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-i",
        str(config["ssh_key"]),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]


if __name__ == "__main__":
    main()
