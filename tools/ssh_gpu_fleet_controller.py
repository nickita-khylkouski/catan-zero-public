from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HOSTS = (
    "gpu-v100:ubuntu@gpu-v100",
    "gpu-h100:ubuntu@gpu-h100",
)
DEFAULT_KEY = Path.home() / ".ssh" / "gpu_access_ed25519"
DEFAULT_REMOTE_REPO = "~/catan-zero-gpu"
DEFAULT_CHAMPION = "runs/self_play/champions/current_best_s9752_iter0002.pt"
REFILL_MANIFEST = "runs/self_play/gpu_manifests/launch_refill.jsonl"


@dataclass(frozen=True, slots=True)
class HostCapacity:
    max_cuda_trainers: int
    max_cpu_trainers: int
    gpu_indices: tuple[int, ...]
    max_new_cuda_memory_used_mib: int


DEFAULT_CAPACITY = {
    "gpu-v100": HostCapacity(
        max_cuda_trainers=14,
        max_cpu_trainers=12,
        gpu_indices=tuple(range(8)),
        max_new_cuda_memory_used_mib=2500,
    ),
    "gpu-h100": HostCapacity(
        max_cuda_trainers=4,
        max_cpu_trainers=4,
        gpu_indices=(0,),
        max_new_cuda_memory_used_mib=62_000,
    ),
}


@dataclass(frozen=True, slots=True)
class Host:
    label: str
    target: str


@dataclass(frozen=True, slots=True)
class RefillSpec:
    host_label: str
    target: str
    seed: int
    label: str
    recipe: str
    device: str
    gpu_index: int | None
    checkpoint: str
    report: str
    log: str
    command: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll and pull CatanZero artifacts from plain SSH GPU hosts."
    )
    parser.add_argument("--host", action="append", default=[])
    parser.add_argument("--key", default=str(DEFAULT_KEY))
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--run-prefix", default="s20")
    subparsers = parser.add_subparsers(dest="command", required=True)

    poll_parser = subparsers.add_parser("poll")
    poll_parser.add_argument("--output")

    pull_parser = subparsers.add_parser("pull-finished")
    pull_parser.add_argument("--poll")
    pull_parser.add_argument("--output-dir", default="runs/self_play/gpu_imports")
    pull_parser.add_argument("--include-interim", action="store_true")
    pull_parser.add_argument("--dry-run", action="store_true")

    refill_plan_parser = subparsers.add_parser("refill-plan")
    refill_plan_parser.add_argument("--poll")
    refill_plan_parser.add_argument("--output")
    refill_plan_parser.add_argument("--max-launches", type=int, default=2)
    refill_plan_parser.add_argument("--seed-floor", type=int, default=20400)
    refill_plan_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    refill_plan_parser.add_argument(
        "--recipe",
        choices=(
            "auto",
            "selfplay_vrpo_guard",
            "selfplay_ema_guard",
            "jsettlers_value_repair",
            "large_graph_distill",
        ),
        default="auto",
    )
    refill_plan_parser.add_argument("--disable-cuda", action="store_true")
    refill_plan_parser.add_argument("--disable-cpu", action="store_true")

    refill_parser = subparsers.add_parser("refill")
    refill_parser.add_argument("--poll")
    refill_parser.add_argument("--plan")
    refill_parser.add_argument("--output")
    refill_parser.add_argument("--max-launches", type=int, default=2)
    refill_parser.add_argument("--seed-floor", type=int, default=20400)
    refill_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    refill_parser.add_argument(
        "--recipe",
        choices=(
            "auto",
            "selfplay_vrpo_guard",
            "selfplay_ema_guard",
            "jsettlers_value_repair",
            "large_graph_distill",
        ),
        default="auto",
    )
    refill_parser.add_argument("--disable-cuda", action="store_true")
    refill_parser.add_argument("--disable-cpu", action="store_true")
    refill_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    hosts = parse_hosts(args.host or list(DEFAULT_HOSTS))
    if args.command == "poll":
        payload = poll_hosts(
            hosts,
            key=Path(args.key),
            remote_repo=args.remote_repo,
            run_prefix=args.run_prefix,
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "pull-finished":
        poll_payload = (
            json.loads(Path(args.poll).read_text(encoding="utf-8"))
            if args.poll
            else poll_hosts(
                hosts,
                key=Path(args.key),
                remote_repo=args.remote_repo,
                run_prefix=args.run_prefix,
            )
        )
        pulled = pull_finished_artifacts(
            poll_payload,
            key=Path(args.key),
            output_dir=Path(args.output_dir),
            include_interim=args.include_interim,
            dry_run=args.dry_run,
        )
        print(json.dumps({"pulled": pulled}, indent=2, sort_keys=True))
    elif args.command == "refill-plan":
        poll_payload = _read_or_poll(
            args.poll,
            hosts=hosts,
            key=Path(args.key),
            remote_repo=args.remote_repo,
            run_prefix=args.run_prefix,
        )
        payload = plan_refill(
            poll_payload,
            remote_repo=args.remote_repo,
            champion=args.champion,
            recipe=args.recipe,
            seed_floor=args.seed_floor,
            max_launches=args.max_launches,
            allow_cuda=not args.disable_cuda,
            allow_cpu=not args.disable_cpu,
        )
        _write_json_if_requested(payload, args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "refill":
        if args.plan:
            payload = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        else:
            poll_payload = _read_or_poll(
                args.poll,
                hosts=hosts,
                key=Path(args.key),
                remote_repo=args.remote_repo,
                run_prefix=args.run_prefix,
            )
            payload = plan_refill(
                poll_payload,
                remote_repo=args.remote_repo,
                champion=args.champion,
                recipe=args.recipe,
                seed_floor=args.seed_floor,
                max_launches=args.max_launches,
                allow_cuda=not args.disable_cuda,
                allow_cpu=not args.disable_cpu,
            )
        launched = launch_refill_plan(payload, key=Path(args.key), dry_run=args.dry_run)
        result = {**payload, "launched": launched}
        _write_json_if_requested(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))


def _read_or_poll(
    poll_path: str | None,
    *,
    hosts: list[Host],
    key: Path,
    remote_repo: str,
    run_prefix: str,
) -> dict[str, Any]:
    if poll_path:
        return json.loads(Path(poll_path).read_text(encoding="utf-8"))
    return poll_hosts(hosts, key=key, remote_repo=remote_repo, run_prefix=run_prefix)


def _write_json_if_requested(payload: dict[str, Any], output: str | None) -> None:
    if not output:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_hosts(values: list[str]) -> list[Host]:
    hosts = []
    for value in values:
        label, sep, target = value.partition(":")
        if sep != ":" or not label or not target:
            raise SystemExit(f"invalid host {value!r}; expected label:user@host")
        hosts.append(Host(label=label, target=target))
    return hosts


def poll_hosts(
    hosts: list[Host],
    *,
    key: Path,
    remote_repo: str,
    run_prefix: str,
) -> dict[str, Any]:
    rows = [
        poll_host(host, key=key, remote_repo=remote_repo, run_prefix=run_prefix)
        for host in hosts
    ]
    return {
        "hosts": rows,
        "running_train_processes": sum(
            int(row.get("running_train_processes", 0) or 0) for row in rows
        ),
        "finished_checkpoints": sum(
            len(row.get("finished_checkpoints", []) or []) for row in rows
        ),
    }


def poll_host(
    host: Host,
    *,
    key: Path,
    remote_repo: str,
    run_prefix: str,
) -> dict[str, Any]:
    command = remote_poll_command(remote_repo=remote_repo, run_prefix=run_prefix)
    try:
        result = subprocess.run(
            [
                "ssh",
                "-i",
                str(key),
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                host.target,
                command,
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
        payload = json.loads(result.stdout)
        payload.update({"label": host.label, "target": host.target, "ok": True})
        return payload
    except Exception as exc:  # noqa: BLE001 - one bad host should not hide the other.
        return {
            "label": host.label,
            "target": host.target,
            "ok": False,
            "error": str(exc),
        }


def remote_poll_command(*, remote_repo: str, run_prefix: str) -> str:
    repo_expr = json.dumps(remote_repo)
    prefix_expr = json.dumps(run_prefix)
    return f"""python3 - <<'PY'
import json, os, pathlib, re, subprocess
repo=os.path.expanduser({repo_expr})
prefix={prefix_expr}
os.chdir(repo)
ps=subprocess.run(['pgrep','-af','tools/train_ppo.py'], text=True, stdout=subprocess.PIPE).stdout
processes=[]
for line in ps.splitlines():
    if (
        'pgrep -af' in line
        or 'python3 - <<' in line
        or '/bin/sh -c' in line
        or 'bash -c' in line
    ):
        continue
    seed=re.search(r'--seed (\\d+)', line)
    checkpoint=re.search(r'--checkpoint ([^ ]+)', line)
    device=re.search(r'--device ([^ ]+)', line)
    processes.append({{
        'seed': seed.group(1) if seed else None,
        'checkpoint': checkpoint.group(1) if checkpoint else None,
        'device': device.group(1) if device else None,
        'command': line[:1000],
    }})
gpu=subprocess.run(
    ['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
files=[]
for path in sorted(pathlib.Path('runs/self_play').glob(prefix + '*')):
    if path.suffix not in ('.pt', '.json'):
        continue
    files.append({{'name': path.name, 'size': path.stat().st_size}})
logs=[]
for path in sorted(pathlib.Path('runs/self_play/logs').glob(prefix + '*.log')):
    text=path.read_text(errors='replace').splitlines()
    logs.append({{'name': path.name, 'lines': len(text), 'tail': text[-1][:500] if text else ''}})
manifest_rows=[]
for path in sorted(pathlib.Path('runs/self_play/gpu_manifests').glob('launch*.jsonl')):
    for line in path.read_text(errors='replace').splitlines():
        try:
            row=json.loads(line)
        except Exception:
            continue
        if str(row.get('label','')).startswith(prefix):
            manifest_rows.append(row)
finished=[]
for row in manifest_rows:
    checkpoint=pathlib.Path(str(row.get('checkpoint',''))).name
    report=pathlib.Path(str(row.get('report',''))).name
    if checkpoint and pathlib.Path('runs/self_play', checkpoint).exists():
        finished.append({{**row, 'checkpoint_name': checkpoint, 'report_name': report}})
print(json.dumps({{
    'repo': repo,
    'running_train_processes': len(processes),
    'processes': processes,
    'gpu': gpu.stdout.splitlines(),
    'files': files,
    'finished_checkpoints': finished,
    'logs': logs,
}}, sort_keys=True))
PY"""


def pull_finished_artifacts(
    poll_payload: dict[str, Any],
    *,
    key: Path,
    output_dir: Path,
    include_interim: bool,
    dry_run: bool,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[str] = []
    for host in poll_payload.get("hosts", []):
        if not host.get("ok"):
            continue
        target = str(host["target"])
        label = str(host["label"])
        repo = str(host.get("repo") or DEFAULT_REMOTE_REPO)
        names = artifact_names(host, include_interim=include_interim)
        for name in sorted(names):
            destination = output_dir / f"{label}_{name}"
            if destination.exists():
                continue
            remote_path = f"{target}:{repo}/runs/self_play/{name}"
            command = [
                "rsync",
                "-az",
                "-e",
                f"ssh -i {key} -o IdentitiesOnly=yes",
                remote_path,
                str(destination),
            ]
            print(json.dumps({"command": command, "dry_run": dry_run}), flush=True)
            if not dry_run:
                subprocess.run(command, check=True)
            pulled.append(str(destination))
    return pulled


def artifact_names(host_payload: dict[str, Any], *, include_interim: bool) -> set[str]:
    existing = {str(row.get("name", "")) for row in host_payload.get("files", [])}
    names: set[str] = set()
    for row in host_payload.get("finished_checkpoints", []):
        checkpoint = str(row.get("checkpoint_name", ""))
        if checkpoint:
            names.add(checkpoint)
        report = str(row.get("report_name", ""))
        if report in existing:
            names.add(report)
    if include_interim:
        for name in existing:
            if ".iter" in name and name.endswith(".pt"):
                names.add(name)
    return names


def plan_refill(
    poll_payload: dict[str, Any],
    *,
    remote_repo: str,
    champion: str,
    recipe: str,
    seed_floor: int,
    max_launches: int,
    allow_cuda: bool,
    allow_cpu: bool,
) -> dict[str, Any]:
    planned: list[RefillSpec] = []
    skipped: dict[str, list[dict[str, Any]]] = {
        "host_error": [],
        "unknown_capacity": [],
        "cuda_full": [],
        "cpu_full": [],
        "no_safe_gpu": [],
        "disabled": [],
    }
    seed = next_refill_seed(poll_payload, seed_floor=seed_floor)
    used_labels = existing_labels(poll_payload)
    for host_payload in poll_payload.get("hosts", []):
        label = str(host_payload.get("label", ""))
        if not host_payload.get("ok"):
            skipped["host_error"].append({"host": label, "error": host_payload.get("error")})
            continue
        capacity = DEFAULT_CAPACITY.get(label)
        if capacity is None:
            skipped["unknown_capacity"].append({"host": label})
            continue
        if allow_cuda and len(planned) < max(0, int(max_launches)):
            cuda_spec, seed = _plan_cuda_refill(
                host_payload,
                capacity=capacity,
                remote_repo=remote_repo,
                champion=champion,
                requested_recipe=recipe,
                seed=seed,
                used_labels=used_labels,
                skipped=skipped,
            )
            if cuda_spec is not None:
                planned.append(cuda_spec)
        elif not allow_cuda:
            skipped["disabled"].append({"host": label, "kind": "cuda"})
        if allow_cpu and len(planned) < max(0, int(max_launches)):
            cpu_spec, seed = _plan_cpu_refill(
                host_payload,
                capacity=capacity,
                remote_repo=remote_repo,
                champion=champion,
                requested_recipe=recipe,
                seed=seed,
                used_labels=used_labels,
                skipped=skipped,
            )
            if cpu_spec is not None:
                planned.append(cpu_spec)
        elif not allow_cpu:
            skipped["disabled"].append({"host": label, "kind": "cpu"})
        if len(planned) >= max(0, int(max_launches)):
            break
    return {
        "planned_count": len(planned),
        "planned": [spec_to_dict(spec) for spec in planned],
        "skipped": skipped,
        "next_seed": seed,
        "running_train_processes": poll_payload.get("running_train_processes", 0),
        "strategy": (
            "remote-only conservative refill: never restarts OOM labels, "
            "uses active process list plus GPU memory, and leaves promotion to strict gates"
        ),
    }


def _plan_cuda_refill(
    host_payload: dict[str, Any],
    *,
    capacity: HostCapacity,
    remote_repo: str,
    champion: str,
    requested_recipe: str,
    seed: int,
    used_labels: set[str],
    skipped: dict[str, list[dict[str, Any]]],
) -> tuple[RefillSpec | None, int]:
    host_label = str(host_payload["label"])
    cuda_count = active_cuda_count(host_payload)
    if cuda_count >= capacity.max_cuda_trainers:
        skipped["cuda_full"].append(
            {"host": host_label, "active_cuda": cuda_count, "target": capacity.max_cuda_trainers}
        )
        return None, seed
    gpu_index = choose_safe_gpu(host_payload, capacity)
    if gpu_index is None:
        skipped["no_safe_gpu"].append(
            {
                "host": host_label,
                "active_cuda": cuda_count,
                "active_gpus": sorted(active_gpu_indices(host_payload)),
                "gpu": host_payload.get("gpu", []),
            }
        )
        return None, seed
    selected_recipe = select_refill_recipe(
        requested_recipe,
        host_label=host_label,
        kind="cuda",
        ordinal=seed,
    )
    spec = make_refill_spec(
        host_payload,
        seed=seed,
        recipe=selected_recipe,
        device="cuda",
        gpu_index=gpu_index,
        remote_repo=remote_repo,
        champion=champion,
        used_labels=used_labels,
    )
    return spec, seed + 1


def _plan_cpu_refill(
    host_payload: dict[str, Any],
    *,
    capacity: HostCapacity,
    remote_repo: str,
    champion: str,
    requested_recipe: str,
    seed: int,
    used_labels: set[str],
    skipped: dict[str, list[dict[str, Any]]],
) -> tuple[RefillSpec | None, int]:
    host_label = str(host_payload["label"])
    cpu_count = active_cpu_count(host_payload)
    if cpu_count >= capacity.max_cpu_trainers:
        skipped["cpu_full"].append(
            {"host": host_label, "active_cpu": cpu_count, "target": capacity.max_cpu_trainers}
        )
        return None, seed
    selected_recipe = select_refill_recipe(
        requested_recipe,
        host_label=host_label,
        kind="cpu",
        ordinal=seed,
    )
    if selected_recipe == "large_graph_distill":
        selected_recipe = "selfplay_ema_guard"
    spec = make_refill_spec(
        host_payload,
        seed=seed,
        recipe=selected_recipe,
        device="cpu",
        gpu_index=None,
        remote_repo=remote_repo,
        champion=champion,
        used_labels=used_labels,
    )
    return spec, seed + 1


def select_refill_recipe(
    requested_recipe: str,
    *,
    host_label: str,
    kind: str,
    ordinal: int,
) -> str:
    if requested_recipe != "auto":
        return requested_recipe
    if kind == "cuda" and host_label == "gpu-h100" and ordinal % 5 == 0:
        return "large_graph_distill"
    if kind == "cuda":
        return "selfplay_vrpo_guard"
    return "selfplay_vrpo_guard" if ordinal % 2 == 0 else "selfplay_ema_guard"


def make_refill_spec(
    host_payload: dict[str, Any],
    *,
    seed: int,
    recipe: str,
    device: str,
    gpu_index: int | None,
    remote_repo: str,
    champion: str,
    used_labels: set[str],
) -> RefillSpec:
    host_label = str(host_payload["label"])
    label = unique_refill_label(
        seed=seed,
        recipe=recipe,
        host_label=host_label,
        device=device,
        gpu_index=gpu_index,
        used_labels=used_labels,
    )
    checkpoint = f"runs/self_play/{label}.pt"
    report = f"runs/self_play/{label}.json"
    log = f"runs/self_play/logs/{label}.log"
    args = build_refill_training_args(
        seed=seed,
        recipe=recipe,
        device=device,
        champion=champion,
        checkpoint=checkpoint,
        report=report,
    )
    command = build_remote_refill_command(
        remote_repo=remote_repo,
        label=label,
        args=args,
        checkpoint=checkpoint,
        report=report,
        log=log,
        device=device,
        gpu_index=gpu_index,
        host_label=host_label,
        seed=seed,
        recipe=recipe,
    )
    used_labels.add(label)
    return RefillSpec(
        host_label=host_label,
        target=str(host_payload["target"]),
        seed=seed,
        label=label,
        recipe=recipe,
        device=device,
        gpu_index=gpu_index,
        checkpoint=checkpoint,
        report=report,
        log=log,
        command=command,
    )


def unique_refill_label(
    *,
    seed: int,
    recipe: str,
    host_label: str,
    device: str,
    gpu_index: int | None,
    used_labels: set[str],
) -> str:
    safe_host = host_label.replace("gpu-", "").replace("-", "_")
    if device == "cuda":
        suffix = f"{safe_host}g{gpu_index}"
    else:
        suffix = f"{safe_host}cpu"
    base = f"s{seed}_{recipe}_{suffix}"
    label = base
    counter = 2
    while label in used_labels:
        label = f"{base}_r{counter}"
        counter += 1
    return label


def build_refill_training_args(
    *,
    seed: int,
    recipe: str,
    device: str,
    champion: str,
    checkpoint: str,
    report: str,
) -> list[str]:
    args = [
        ".venv/bin/python",
        "-u",
        "tools/train_ppo.py",
        "--seed",
        str(seed),
        "--vps-to-win",
        "4",
        "--max-decisions",
        "300",
    ]
    if recipe == "large_graph_distill":
        args.extend(
            [
                "--architecture",
                "graph_history_candidate",
                "--hidden-size",
                "384",
                "--teacher",
                "tactical_rollout_mixed",
                "--teacher-candidate-limit",
                "32",
                "--teacher-presearch-candidate-limit",
                "64",
                "--teacher-rollout-decisions",
                "3",
                "--teacher-rollout-samples",
                "1",
                "--teacher-root-value-weight",
                "0.35",
                "--warmup-games",
                "32",
                "--warmup-epochs",
                "2",
                "--warmup-value-coef",
                "0.45",
                "--warmup-replay-size",
                "8192",
                "--warmup-checkpoint-every",
                "16",
                "--warmup-checkpoint-eval-games",
                "2",
                "--warmup-checkpoint-eval-value-games",
                "2",
                "--iterations",
                "16",
                "--episodes-per-iteration",
                "12",
                "--learner-seats",
                "all",
                "--opponents",
                "self",
                "--ppo-epochs",
                "2",
                "--minibatch-size",
                "512",
                "--learning-rate",
                "0.00005",
                "--clip-ratio",
                "0.07",
                "--value-coef",
                "0.70",
                "--q-value-coef",
                "0.50",
                "--q-advantage-mix",
                "0.06",
                "--q-expected-sarsa-mix",
                "0.55",
            ]
        )
    else:
        args.extend(
            [
                "--init-checkpoint",
                champion,
                "--teacher",
                "baseline_rollout_mixed",
                "--warmup-games",
                "0",
                "--warmup-epochs",
                "0",
                "--warmup-value-coef",
                "0.5",
                "--anchor-value-coef",
                "0.0",
                "--iterations",
                "24",
                "--episodes-per-iteration",
                "18",
            ]
        )
        if recipe == "jsettlers_value_repair":
            args.extend(
                [
                    "--opponent-checkpoints",
                    champion,
                    "--learner-seats",
                    "one",
                    "--opponents",
                    "jsettlers_value_repair_mixed",
                    "--training-value-candidate-limit",
                    "24",
                    "--training-value-opponent-penalty",
                    "0.05",
                    "--ppo-epochs",
                    "2",
                    "--minibatch-size",
                    "256",
                    "--learning-rate",
                    "0.000055",
                    "--clip-ratio",
                    "0.07",
                    "--value-coef",
                    "0.70",
                    "--q-value-coef",
                    "0.50",
                    "--q-advantage-mix",
                    "0.08",
                    "--q-expected-sarsa-mix",
                    "0.60",
                    "--entropy-coef",
                    "0.009",
                    "--old-policy-kl-coef",
                    "0.045",
                    "--ema-policy-kl-coef",
                    "0.090",
                    "--ema-policy-decay",
                    "0.9985",
                    "--target-kl",
                    "0.007",
                    "--anchor-games-per-iteration",
                    "4",
                    "--dagger-games-per-iteration",
                    "4",
                    "--dagger-low-return-multiplier",
                    "1.5",
                    "--anchor-replay-size",
                    "6144",
                    "--anchor-epochs",
                    "1",
                    "--anchor-learning-rate-multiplier",
                    "0.5",
                ]
            )
        else:
            q_mix = "0.08" if recipe == "selfplay_vrpo_guard" else "0.06"
            esarsa_mix = "0.62" if recipe == "selfplay_vrpo_guard" else "0.50"
            q_coef = "0.55" if recipe == "selfplay_vrpo_guard" else "0.45"
            args.extend(
                [
                    "--learner-seats",
                    "all",
                    "--opponents",
                    "self",
                    "--ppo-epochs",
                    "2",
                    "--minibatch-size",
                    "384" if device == "cuda" else "256",
                    "--learning-rate",
                    "0.000045" if recipe == "selfplay_vrpo_guard" else "0.00005",
                    "--clip-ratio",
                    "0.065" if recipe == "selfplay_vrpo_guard" else "0.075",
                    "--value-coef",
                    "0.70",
                    "--q-value-coef",
                    q_coef,
                    "--q-advantage-mix",
                    q_mix,
                    "--q-expected-sarsa-mix",
                    esarsa_mix,
                    "--entropy-coef",
                    "0.009" if recipe == "selfplay_vrpo_guard" else "0.010",
                    "--old-policy-kl-coef",
                    "0.070" if recipe == "selfplay_vrpo_guard" else "0.060",
                    "--ema-policy-kl-coef",
                    "0.115" if recipe == "selfplay_vrpo_guard" else "0.095",
                    "--ema-policy-decay",
                    "0.9985",
                    "--target-kl",
                    "0.006" if recipe == "selfplay_vrpo_guard" else "0.007",
                    "--anchor-games-per-iteration",
                    "4",
                    "--dagger-games-per-iteration",
                    "2",
                    "--dagger-low-return-multiplier",
                    "1.35",
                    "--anchor-replay-size",
                    "8192" if device == "cuda" else "4096",
                    "--anchor-epochs",
                    "1",
                    "--anchor-learning-rate-multiplier",
                    "0.45",
                ]
            )
    args.extend(
        [
            "--q-advantage-warmup-iterations",
            "2",
            "--q-advantage-ramp-iterations",
            "4",
            "--q-advantage-min-sign-agreement",
            "0.52",
            "--q-advantage-min-return-corr",
            "-0.05",
            "--value-clip-range",
            "0.25",
            "--checkpoint-every",
            "4",
            "--checkpoint-eval-games",
            "4",
            "--checkpoint-eval-value-games",
            "4",
            "--eval-games",
            "0",
            "--eval-value-games",
            "0",
            "--select-best-checkpoint",
            "--select-best-min-value-win-rate",
            "0.18",
            "--device",
            device,
            "--checkpoint",
            checkpoint,
            "--report",
            report,
        ]
    )
    return args


def build_remote_refill_command(
    *,
    remote_repo: str,
    label: str,
    args: list[str],
    checkpoint: str,
    report: str,
    log: str,
    device: str,
    gpu_index: int | None,
    host_label: str,
    seed: int,
    recipe: str,
) -> str:
    manifest = {
        "label": label,
        "seed": seed,
        "host": host_label,
        "recipe": recipe,
        "device": device,
        "gpu_index": gpu_index,
        "checkpoint": checkpoint,
        "report": report,
        "log": log,
        "command": args,
        "source": "ssh_gpu_fleet_controller.refill",
    }
    env = {}
    if device == "cuda" and gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    payload = {
        "repo": remote_repo,
        "manifest_path": REFILL_MANIFEST,
        "manifest": manifest,
        "args": args,
        "log": log,
        "env": env,
    }
    payload_json = repr(json.dumps(payload, sort_keys=True))
    return f"""python3 - <<'PY'
import json, os, pathlib, subprocess
payload = json.loads({payload_json})
repo = os.path.expanduser(payload["repo"])
os.chdir(repo)
pathlib.Path("runs/self_play/logs").mkdir(parents=True, exist_ok=True)
pathlib.Path("runs/self_play/gpu_manifests").mkdir(parents=True, exist_ok=True)
with open(payload["manifest_path"], "a", encoding="utf-8") as manifest_file:
    manifest_file.write(json.dumps(payload["manifest"], sort_keys=True) + "\\n")
env = os.environ.copy()
env.update({{str(key): str(value) for key, value in payload["env"].items()}})
log_file = open(payload["log"], "ab", buffering=0)
try:
    process = subprocess.Popen(
        payload["args"],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
finally:
    log_file.close()
print(json.dumps({{"label": payload["manifest"]["label"], "pid": process.pid}}, sort_keys=True))
PY"""


def launch_refill_plan(
    plan_payload: dict[str, Any],
    *,
    key: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    launched = []
    for row in plan_payload.get("planned", []):
        target = str(row["target"])
        command = str(row["command"])
        ssh_command = [
            "ssh",
            "-n",
            "-i",
            str(key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "StrictHostKeyChecking=accept-new",
            target,
            command,
        ]
        launched_row = {
            "host": row.get("host_label"),
            "label": row.get("label"),
            "target": target,
            "dry_run": dry_run,
            "ssh_command": ssh_command,
        }
        print(json.dumps(launched_row, sort_keys=True), flush=True)
        if not dry_run:
            subprocess.run(ssh_command, check=True, timeout=30)
        launched.append(launched_row)
    return launched


def spec_to_dict(spec: RefillSpec) -> dict[str, Any]:
    return {
        "host_label": spec.host_label,
        "target": spec.target,
        "seed": spec.seed,
        "label": spec.label,
        "recipe": spec.recipe,
        "device": spec.device,
        "gpu_index": spec.gpu_index,
        "checkpoint": spec.checkpoint,
        "report": spec.report,
        "log": spec.log,
        "command": spec.command,
    }


def existing_labels(poll_payload: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for host in poll_payload.get("hosts", []):
        for process in host.get("processes", []) or []:
            checkpoint = Path(str(process.get("checkpoint", ""))).name
            if checkpoint.endswith(".pt"):
                labels.add(checkpoint[:-3])
        for row in host.get("files", []) or []:
            name = str(row.get("name", ""))
            if name.endswith((".pt", ".json")):
                labels.add(strip_checkpoint_suffix(name))
        for row in host.get("logs", []) or []:
            name = str(row.get("name", ""))
            if name.endswith(".log"):
                labels.add(name[:-4])
    return labels


def strip_checkpoint_suffix(name: str) -> str:
    if re.search(r"\.iter\d+\.pt$", name):
        return name.rsplit(".iter", 1)[0]
    for suffix in (".pt", ".json", ".log"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def next_refill_seed(poll_payload: dict[str, Any], *, seed_floor: int) -> int:
    seeds = [int(seed_floor)]
    for host in poll_payload.get("hosts", []):
        for process in host.get("processes", []) or []:
            seed = _parse_int(process.get("seed"))
            if seed is not None:
                seeds.append(seed + 1)
        for row in host.get("files", []) or []:
            match = re.match(r"s(\d+)_", str(row.get("name", "")))
            if match:
                seeds.append(int(match.group(1)) + 1)
    return max(seeds)


def active_cpu_count(host_payload: dict[str, Any]) -> int:
    return sum(1 for process in host_payload.get("processes", []) or [] if is_cpu_process(process))


def active_cuda_count(host_payload: dict[str, Any]) -> int:
    return sum(1 for process in host_payload.get("processes", []) or [] if is_cuda_process(process))


def is_cpu_process(process: dict[str, Any]) -> bool:
    device = str(process.get("device") or "")
    checkpoint = str(process.get("checkpoint") or "")
    return device == "cpu" or "_cpu_" in checkpoint


def is_cuda_process(process: dict[str, Any]) -> bool:
    device = str(process.get("device") or "")
    return device.startswith("cuda") and not is_cpu_process(process)


def active_gpu_indices(host_payload: dict[str, Any]) -> set[int]:
    indices: set[int] = set()
    for process in host_payload.get("processes", []) or []:
        if not is_cuda_process(process):
            continue
        checkpoint = Path(str(process.get("checkpoint", ""))).name
        match = re.search(r"g(\d+)(?=\.|_|$)", checkpoint)
        if match:
            indices.add(int(match.group(1)))
    return indices


def choose_safe_gpu(host_payload: dict[str, Any], capacity: HostCapacity) -> int | None:
    active = active_gpu_indices(host_payload)
    memory_by_gpu = parse_gpu_memory_rows(host_payload.get("gpu", []) or [])
    for index in capacity.gpu_indices:
        if index in active:
            continue
        used = memory_by_gpu.get(index, 0)
        if used <= capacity.max_new_cuda_memory_used_mib:
            return index
    return None


def parse_gpu_memory_rows(rows: list[str]) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for row in rows:
        parts = [part.strip() for part in str(row).split(",")]
        if len(parts) < 3:
            continue
        index = _parse_int(parts[0])
        used = _parse_int(parts[2].replace("MiB", "").strip())
        if index is not None and used is not None:
            parsed[index] = used
    return parsed


def _parse_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
