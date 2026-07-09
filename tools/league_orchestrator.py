from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1
DEFAULT_BRANCHES = (
    "adaptive_ema_qoff",
    "search_ema_dagger_qoff",
    "allseat_lowkl_qoff",
)


@dataclass(frozen=True, slots=True)
class BranchSpec:
    name: str
    seed: int
    box: str
    remote_dir: str
    checkpoint_prefix: str
    command: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch and manage longer box-based CatanZero league branches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    _add_common_args(plan_parser)
    plan_parser.add_argument("--boxes", nargs="+", required=True)
    plan_parser.add_argument("--branch", choices=DEFAULT_BRANCHES, action="append")
    plan_parser.add_argument("--write-manifest", action="store_true")

    launch_parser = subparsers.add_parser("launch")
    _add_common_args(launch_parser)
    launch_parser.add_argument("--boxes", nargs="+", required=True)
    launch_parser.add_argument("--branch", choices=DEFAULT_BRANCHES, action="append")
    launch_parser.add_argument("--dry-run", action="store_true")

    poll_parser = subparsers.add_parser("poll")
    poll_parser.add_argument("--manifest", default="runs/self_play/league_v2/manifest.json")

    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("--manifest", default="runs/self_play/league_v2/manifest.json")
    pull_parser.add_argument("--output-dir", default="runs/self_play/box_imports")

    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--checkpoint", action="append", required=True)
    gate_parser.add_argument("--eval-dir", default="runs/self_play/box_eval_imports")
    gate_parser.add_argument("--champion", default="runs/self_play/champions/current_best_s9752_iter0002.pt")
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
        help=(
            "Use stored champion win counts instead of evaluating the champion "
            "on the same legs. This is only for quick debugging."
        ),
    )
    gate_parser.set_defaults(evaluate_champion=True)
    gate_parser.add_argument("--promote-if-better", action="store_true")
    gate_parser.add_argument("--dry-run", action="store_true")

    cleanup_parser = subparsers.add_parser("cleanup-local-wrappers")
    cleanup_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "plan":
        specs = build_branch_specs(args)
        manifest = build_manifest(args, specs)
        if args.write_manifest:
            write_manifest(manifest, Path(args.manifest))
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "launch":
        specs = build_branch_specs(args)
        manifest = build_manifest(args, specs)
        write_manifest(manifest, Path(args.manifest))
        launch_specs(specs, bundle=Path(args.bundle), dry_run=args.dry_run)
        print(json.dumps({"manifest": args.manifest, "branches": len(specs)}, sort_keys=True))
    elif args.command == "poll":
        manifest = read_manifest(Path(args.manifest))
        print(json.dumps(poll_manifest(manifest), indent=2, sort_keys=True))
    elif args.command == "pull":
        manifest = read_manifest(Path(args.manifest))
        pulled = pull_manifest_checkpoints(manifest, Path(args.output_dir))
        print(json.dumps({"pulled": pulled}, indent=2, sort_keys=True))
    elif args.command == "gate":
        print(json.dumps(gate_checkpoints(args), indent=2, sort_keys=True))
    elif args.command == "cleanup-local-wrappers":
        print(json.dumps(cleanup_local_wrappers(dry_run=args.dry_run), sort_keys=True))
    else:  # pragma: no cover - argparse enforces subcommands.
        raise ValueError(args.command)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="runs/self_play/league_v2/manifest.json")
    parser.add_argument("--bundle", default="/tmp/catan-zero-s4920-bundle.tar.gz")
    parser.add_argument(
        "--init-checkpoint",
        default="runs/self_play/champions/current_best_s9752_iter0002.pt",
    )
    parser.add_argument("--base-seed", type=int, default=6000)
    parser.add_argument("--vps-to-win", type=int, default=6)
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-eval-games", type=int, default=0)
    parser.add_argument("--checkpoint-eval-value-games", type=int, default=0)


def build_branch_specs(args: argparse.Namespace) -> list[BranchSpec]:
    branches = tuple(args.branch or DEFAULT_BRANCHES)
    if len(args.boxes) < len(branches):
        raise SystemExit("provide at least one box per requested branch")
    specs = []
    for index, branch in enumerate(branches):
        seed = int(args.base_seed) + index + 1
        name = f"s{seed}_{branch}"
        remote_dir = f"/tmp/catan-zero-{name}"
        checkpoint_prefix = f"runs/self_play/{name}.pt"
        command = build_train_command(args, branch=branch, seed=seed, checkpoint=checkpoint_prefix)
        specs.append(
            BranchSpec(
                name=name,
                seed=seed,
                box=args.boxes[index],
                remote_dir=remote_dir,
                checkpoint_prefix=checkpoint_prefix,
                command=command,
            )
        )
    return specs


def build_train_command(
    args: argparse.Namespace,
    *,
    branch: str,
    seed: int,
    checkpoint: str,
) -> list[str]:
    common = [
        ".venv/bin/python",
        "-u",
        "tools/train_ppo.py",
        "--seed",
        str(seed),
        "--vps-to-win",
        str(args.vps_to_win),
        "--max-decisions",
        str(args.max_decisions),
        "--init-checkpoint",
        str(args.init_checkpoint),
        "--teacher",
        "value",
        "--teacher-candidate-limit",
        "48",
        "--warmup-games",
        "0",
        "--iterations",
        str(args.iterations),
        "--episodes-per-iteration",
        str(args.episodes_per_iteration),
        "--ppo-epochs",
        "1",
        "--minibatch-size",
        "256",
        "--gamma",
        "0.995",
        "--gae-lambda",
        "0.95",
        "--value-clip-range",
        "0.15",
        "--anchor-replay-size",
        "120000",
        "--anchor-epochs",
        "1",
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--checkpoint-eval-games",
        str(args.checkpoint_eval_games),
        "--checkpoint-eval-value-games",
        str(args.checkpoint_eval_value_games),
        "--eval-games",
        "0",
        "--eval-value-games",
        "0",
        "--checkpoint",
        checkpoint,
        "--report",
        checkpoint.removesuffix(".pt") + ".json",
    ]
    if branch == "adaptive_ema_qoff":
        return common + [
            "--opponents",
            "adaptive_league",
            "--learner-seats",
            "one",
            "--league-snapshot-every",
            "10",
            "--league-max-snapshots",
            "8",
            "--learning-rate",
            "0.000014",
            "--clip-ratio",
            "0.07",
            "--value-coef",
            "0.50",
            "--q-value-coef",
            "0.0",
            "--q-advantage-mix",
            "0.0",
            "--q-advantage-warmup-iterations",
            "0",
            "--q-advantage-ramp-iterations",
            "1",
            "--entropy-coef",
            "0.006",
            "--old-policy-kl-coef",
            "0.35",
            "--ema-policy-kl-coef",
            "0.06",
            "--ema-policy-decay",
            "0.97",
            "--target-kl",
            "0.005",
            "--anchor-games-per-iteration",
            "4",
            "--dagger-games-per-iteration",
            "1",
        ]
    if branch == "search_ema_dagger_qoff":
        return common + [
            "--opponents",
            "search_mixed",
            "--learner-seats",
            "one",
            "--learning-rate",
            "0.000010",
            "--clip-ratio",
            "0.055",
            "--value-coef",
            "0.55",
            "--q-value-coef",
            "0.0",
            "--q-advantage-mix",
            "0.0",
            "--q-advantage-warmup-iterations",
            "0",
            "--q-advantage-ramp-iterations",
            "1",
            "--entropy-coef",
            "0.004",
            "--old-policy-kl-coef",
            "0.55",
            "--ema-policy-kl-coef",
            "0.08",
            "--ema-policy-decay",
            "0.97",
            "--target-kl",
            "0.0035",
            "--anchor-games-per-iteration",
            "6",
            "--dagger-games-per-iteration",
            "2",
        ]
    if branch == "allseat_lowkl_qoff":
        return common + [
            "--opponents",
            "self",
            "--learner-seats",
            "all",
            "--league-snapshot-every",
            "10",
            "--league-max-snapshots",
            "8",
            "--learning-rate",
            "0.000010",
            "--clip-ratio",
            "0.06",
            "--value-coef",
            "0.45",
            "--q-value-coef",
            "0.0",
            "--q-advantage-mix",
            "0.0",
            "--q-advantage-warmup-iterations",
            "0",
            "--q-advantage-ramp-iterations",
            "1",
            "--entropy-coef",
            "0.008",
            "--old-policy-kl-coef",
            "0.45",
            "--ema-policy-kl-coef",
            "0.04",
            "--ema-policy-decay",
            "0.97",
            "--target-kl",
            "0.004",
            "--anchor-games-per-iteration",
            "4",
            "--dagger-games-per-iteration",
            "1",
        ]
    raise ValueError(branch)


def build_manifest(args: argparse.Namespace, specs: list[BranchSpec]) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "bundle": str(args.bundle),
        "init_checkpoint": str(args.init_checkpoint),
        "vps_to_win": int(args.vps_to_win),
        "max_decisions": int(args.max_decisions),
        "iterations": int(args.iterations),
        "episodes_per_iteration": int(args.episodes_per_iteration),
        "checkpoint_every": int(args.checkpoint_every),
        "branches": [asdict(spec) for spec in specs],
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def launch_specs(specs: list[BranchSpec], *, bundle: Path, dry_run: bool) -> None:
    for spec in specs:
        run(["box", "scp", str(bundle), f"{spec.box}:/tmp/{bundle.name}"], dry_run=dry_run)
        remote = build_remote_launch_command(spec, bundle_name=bundle.name)
        run(["box", "ssh", spec.box, remote], dry_run=dry_run)


def build_remote_launch_command(spec: BranchSpec, *, bundle_name: str) -> str:
    train = " ".join(shlex.quote(part) for part in spec.command)
    log = f"runs/self_play/logs/{spec.name}.log"
    detached_train = (
        "env PYTHONPATH=src:vendor/catanatron OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        f"{train} > {shlex.quote(log)} 2>&1 < /dev/null"
    )
    return " && ".join(
        [
            f"rm -rf {shlex.quote(spec.remote_dir)}",
            f"mkdir -p {shlex.quote(spec.remote_dir)}",
            f"tar -xzf /tmp/{shlex.quote(bundle_name)} -C {shlex.quote(spec.remote_dir)}",
            f"cd {shlex.quote(spec.remote_dir)}",
            "python3 -m venv .venv",
            ".venv/bin/python -m pip install -q -e '.[rl]'",
            "mkdir -p runs/self_play/logs",
            f"setsid sh -c {shlex.quote(detached_train)} >/dev/null 2>&1 & exit 0",
        ]
    )


def poll_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for branch in manifest.get("branches", []):
        remote_dir = branch["remote_dir"]
        name = branch["name"]
        command = (
            f"cd {shlex.quote(remote_dir)} && "
            f"(pgrep -af {shlex.quote('train_ppo.py --seed ' + str(branch['seed']))} || true); "
            "printf '__CHECKPOINTS__\\n'; "
            f"find runs/self_play -maxdepth 1 -name {shlex.quote(name + '*.pt')} "
            "-printf '%f %s\\n' | sort; "
            "printf '__LOG__\\n'; "
            f"tail -n 3 runs/self_play/logs/{shlex.quote(name)}.log 2>/dev/null || true"
        )
        result = subprocess.run(
            ["box", "ssh", branch["box"], command],
            check=False,
            text=True,
            capture_output=True,
        )
        rows.append(
            {
                "name": name,
                "box": branch["box"],
                "returncode": result.returncode,
                **_parse_poll_stdout(result.stdout),
                "stderr": result.stderr,
            }
        )
    return rows


def _parse_poll_stdout(stdout: str) -> dict[str, Any]:
    process_text, _, rest = stdout.partition("__CHECKPOINTS__\n")
    checkpoint_text, _, log_text = rest.partition("__LOG__\n")
    processes = [
        line
        for line in process_text.splitlines()
        if "train_ppo.py" in line and "pgrep -af" not in line
    ]
    checkpoints = []
    for line in checkpoint_text.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            checkpoints.append({"file": parts[0], "bytes": int(parts[1])})
    return {
        "running": any(".venv/bin/python" in line for line in processes),
        "processes": processes,
        "checkpoints": checkpoints,
        "log_tail": log_text.splitlines(),
    }


def pull_manifest_checkpoints(manifest: dict[str, Any], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled = []
    for branch in manifest.get("branches", []):
        remote_dir = branch["remote_dir"]
        name = branch["name"]
        list_command = (
            f"find {shlex.quote(remote_dir)}/runs/self_play -maxdepth 1 "
            f"-name {shlex.quote(name + '*.pt')} -printf '%f\\n' | sort"
        )
        listed = subprocess.run(
            ["box", "ssh", branch["box"], list_command],
            check=False,
            text=True,
            capture_output=True,
        )
        if listed.returncode != 0:
            continue
        for filename in listed.stdout.splitlines():
            destination = output_dir / filename
            if destination.exists():
                continue
            remote_path = f"{branch['box']}:{remote_dir}/runs/self_play/{filename}"
            run(["box", "scp", remote_path, str(destination)], dry_run=False)
            pulled.append(str(destination))
    return pulled


def gate_checkpoints(args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for checkpoint in args.checkpoint:
        summary = gate_checkpoint(
            checkpoint=Path(checkpoint),
            eval_dir=Path(args.eval_dir),
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
            champion=Path(args.champion),
            evaluate_champion=args.evaluate_champion,
            dry_run=args.dry_run,
        )
        if summary["promote"] and args.promote_if_better and not args.dry_run:
            champion = Path(args.champion)
            champion.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, champion)
            summary["promoted_to"] = str(champion)
        results.append(summary)
    return results


def gate_checkpoint(
    *,
    checkpoint: Path,
    eval_dir: Path,
    games: int,
    workers: int,
    vps_to_win: int,
    max_decisions: int,
    common_heuristic_seed: int,
    common_value_seed: int,
    verify_heuristic_seed: int,
    verify_value_seed: int,
    champion_heuristic_wins: int,
    champion_value_wins: int,
    champion: Path,
    evaluate_champion: bool,
    dry_run: bool,
) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.stem
    legs = [
        ("common", "catanatron_ab3", common_heuristic_seed),
        ("common", "catanatron_search", common_value_seed),
        ("verify2", "jsettlers_lite", verify_heuristic_seed),
        ("verify2", "value", verify_value_seed),
    ]
    reports = {}
    for prefix, opponent, seed in legs:
        output = eval_dir / f"{prefix}_{stem}_vs_{opponent}{games}_s{seed}.json"
        if dry_run:
            reports[f"{prefix}_{opponent}"] = {"output": str(output), "dry_run": True}
            continue
        if not output.exists():
            run(
                build_eval_command(
                    checkpoint=checkpoint,
                    opponent=opponent,
                    games=games,
                    seed=seed,
                    vps_to_win=vps_to_win,
                    max_decisions=max_decisions,
                    workers=workers,
                    output=output,
                ),
                dry_run=False,
            )
        reports[f"{prefix}_{opponent}"] = json.loads(output.read_text(encoding="utf-8"))
    if dry_run:
        return {"checkpoint": str(checkpoint), "reports": reports, "promote": False}
    champion_reports = {}
    if evaluate_champion:
        champion_stem = champion.stem
        for prefix, opponent, seed in legs:
            output = eval_dir / f"{prefix}_{champion_stem}_vs_{opponent}{games}_s{seed}.json"
            if not output.exists():
                run(
                    build_eval_command(
                        checkpoint=champion,
                        opponent=opponent,
                        games=games,
                        seed=seed,
                        vps_to_win=vps_to_win,
                        max_decisions=max_decisions,
                        workers=workers,
                        output=output,
                    ),
                    dry_run=False,
                )
            champion_reports[f"{prefix}_{opponent}"] = json.loads(
                output.read_text(encoding="utf-8")
            )
        champion_heuristic_wins = int(champion_reports["common_catanatron_ab3"]["wins"]) + int(
            champion_reports["verify2_jsettlers_lite"]["wins"]
        )
        champion_value_wins = int(champion_reports["common_catanatron_search"]["wins"]) + int(
            champion_reports["verify2_value"]["wins"]
        )
    candidate_heuristic_wins = int(reports["common_catanatron_ab3"]["wins"]) + int(
        reports["verify2_jsettlers_lite"]["wins"]
    )
    candidate_value_wins = int(reports["common_catanatron_search"]["wins"]) + int(
        reports["verify2_value"]["wins"]
    )
    promote, reason = should_promote_gate(
        candidate_heuristic_wins=candidate_heuristic_wins,
        candidate_value_wins=candidate_value_wins,
        champion_heuristic_wins=champion_heuristic_wins,
        champion_value_wins=champion_value_wins,
    )
    return {
        "checkpoint": str(checkpoint),
        "games_per_leg": games,
        "candidate_heuristic_wins": candidate_heuristic_wins,
        "candidate_value_wins": candidate_value_wins,
        "champion_heuristic_wins": champion_heuristic_wins,
        "champion_value_wins": champion_value_wins,
        "promote": promote,
        "reason": reason,
        "reports": reports,
        "champion_reports": champion_reports,
    }


def build_eval_command(
    *,
    checkpoint: Path,
    opponent: str,
    games: int,
    seed: int,
    vps_to_win: int,
    max_decisions: int,
    workers: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        "tools/evaluate_self_play.py",
        "--candidate",
        "ppo",
        "--checkpoint",
        str(checkpoint),
        "--opponent",
        opponent,
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--vps-to-win",
        str(vps_to_win),
        "--max-decisions",
        str(max_decisions),
        "--workers",
        str(workers),
        "--output",
        str(output),
    ]


def should_promote_gate(
    *,
    candidate_heuristic_wins: int,
    candidate_value_wins: int,
    champion_heuristic_wins: int,
    champion_value_wins: int,
) -> tuple[bool, str]:
    if candidate_value_wins < champion_value_wins:
        return False, "value regression"
    if candidate_heuristic_wins < champion_heuristic_wins:
        return False, "heuristic regression"
    if (
        candidate_value_wins == champion_value_wins
        and candidate_heuristic_wins == champion_heuristic_wins
    ):
        return False, "candidate tied champion aggregate"
    return True, "candidate improved aggregate without regression"


def cleanup_local_wrappers(*, dry_run: bool) -> dict[str, Any]:
    before = _wrapper_count()
    commands = [
        ["pkill", "-f", "^box ssh .*train_ppo.py"],
        ["pkill", "-f", "/bin/ssh .*train_ppo.py"],
    ]
    for command in commands:
        run(command, dry_run=dry_run, check=False)
    after = before if dry_run else _wrapper_count()
    return {"before": before, "after": after, "dry_run": dry_run}


def _wrapper_count() -> int:
    result = subprocess.run(
        "ps -eo command | egrep 'box ssh|/bin/ssh .*train_ppo.py' | grep -v egrep | wc -l",
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )
    return int(result.stdout.strip() or "0")


def run(command: list[str], *, dry_run: bool, check: bool = True) -> subprocess.CompletedProcess:
    print(json.dumps({"command": command, "dry_run": dry_run}, sort_keys=True), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(command, check=check)


if __name__ == "__main__":
    main()
