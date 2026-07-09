from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from factory_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the teacher -> BC bootstrap pipeline on this machine."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument(
        "--teachers",
        default=(
            "catanatron_ab4,catanatron_ab5,value_rollout_search,"
            "catanatron_ab3,catanatron_value,jsettlers_lite"
        ),
    )
    parser.add_argument(
        "--teacher-sampling-weights",
        default=(
            "catanatron_ab5=3.0,catanatron_ab4=2.5,value_rollout_search=2.5,"
            "catanatron_ab3=1.2,catanatron_value=0.6,jsettlers_lite=0.7"
        ),
        help="Generation-time teacher sampling weights for mixed-seat teacher games.",
    )
    parser.add_argument(
        "--scoreboard-opponents",
        default=(
            "random,heuristic,value,jsettlers_lite,catanatron_ab3,"
            "catanatron_ab4,catanatron_ab5,catanatron_search"
        ),
    )
    parser.add_argument("--verify-games", type=int, default=128)
    parser.add_argument("--benchmark-games", type=int, default=256)
    parser.add_argument("--scoreboard-games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--teacher-games", type=int, default=2000)
    parser.add_argument("--teacher-max-decisions", type=int, default=1200)
    parser.add_argument(
        "--quality-gate",
        choices=("none", "strict", "production"),
        default="production",
        help=(
            "Teacher-data quality gate before BC. Use production for real B200 "
            "35M runs; strict is useful for smaller smoke runs."
        ),
    )
    parser.add_argument("--bc-epochs", type=int, default=2)
    parser.add_argument("--bc-batch-size", type=int, default=4096)
    parser.add_argument(
        "--bc-amp",
        choices=("none", "bf16"),
        default="bf16",
        help="Mixed precision mode passed to train_bc.py. B200/A100 should use bf16.",
    )
    parser.add_argument(
        "--torchrun-nproc-per-node",
        type=int,
        default=2,
        help="Launch BC with this many DDP ranks. Use 2 on the B200 box.",
    )
    parser.add_argument(
        "--arch",
        choices=("candidate", "xdim_lite", "xdim_graph", "entity_graph"),
        default="entity_graph",
        help="Teacher-phase default is the 35M entity_graph model, not the old candidate MLP.",
    )
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--graph-tokens", type=int, default=32)
    parser.add_argument("--graph-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--soft-target-temperature", type=float, default=0.7)
    parser.add_argument("--soft-target-weight", type=float, default=0.7)
    parser.add_argument("--forced-action-weight", type=float, default=0.1)
    parser.add_argument(
        "--phase-weights",
        default="robber=3.0,initial_build=2.0,discard=1.5",
    )
    parser.add_argument("--winner-sample-weight", type=float, default=1.0)
    parser.add_argument("--loser-sample-weight", type=float, default=0.3)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--final-vp-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--teacher-weights",
        default=(
            "catanatron_ab5=1.8,catanatron_ab4=1.6,"
            "value_rollout_search=1.5,catanatron_value=1.1,"
            "jsettlers_lite=0.8,catanatron_ab3=1.0"
        ),
    )
    parser.add_argument("--min-35m-params", type=int, default=30_000_000)
    parser.add_argument("--max-35m-params", type=int, default=40_000_000)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--ppo-iterations", type=int, default=20)
    parser.add_argument("--ppo-decisions-per-rank", type=int, default=4096)
    parser.add_argument("--ppo-rollout-workers", type=int, default=1)
    parser.add_argument("--ppo-max-decisions", type=int, default=1200)
    parser.add_argument("--ppo-opponent", default="random")
    parser.add_argument(
        "--allow-ppo",
        action="store_true",
        help=(
            "Also launch PPO after BC. Keep unset during the 35M teacher-training "
            "phase; PPO should only start after the promotion gate passes."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write/print the pipeline manifest without running commands.",
    )
    args = parser.parse_args()
    if args.allow_ppo:
        raise SystemExit(
            "--allow-ppo is disabled for the 35M teacher-only phase. "
            "Promote a BC checkpoint by scoreboard first, then use the separate "
            "PPO runbook after explicit approval."
        )

    run_dir = Path(args.run_dir)
    raw_data_dir = run_dir / "teacher_data_raw"
    data_dir = run_dir / "teacher_data"
    teacher_quality_report = run_dir / "teacher_data_quality.json"
    bc_checkpoint = run_dir / f"bc_{args.arch}.pt"
    bc_report = run_dir / f"bc_{args.arch}.json"
    ppo_config = run_dir / "ppo_config.json"
    scoreboard = run_dir / "scoreboard_bc.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            args.python,
            "tools/verify_fast_env.py",
            "--games",
            str(args.verify_games),
            "--workers",
            str(args.workers),
            "--track",
            args.track,
            "--vps-to-win",
            str(args.vps_to_win),
            "--max-decisions",
            str(args.ppo_max_decisions),
            "--out",
            str(run_dir / "verify_fast_env.json"),
        ],
        [
            args.python,
            "tools/benchmark_gh200.py",
            "--games",
            str(args.benchmark_games),
            "--workers",
            str(args.workers),
            "--players",
            "random",
            "--track",
            args.track,
            "--vps-to-win",
            str(args.vps_to_win),
            "--max-decisions",
            str(args.ppo_max_decisions),
            "--out",
            str(run_dir / "benchmark_random.json"),
        ],
        [
            args.python,
            "tools/generate_teacher_data.py",
            "--track",
            args.track,
            "--vps-to-win",
            str(args.vps_to_win),
            "--teachers",
            args.teachers,
            "--teacher-sampling-weights",
            args.teacher_sampling_weights,
            "--games",
            str(args.teacher_games),
            "--seed",
            str(args.seed),
            "--max-decisions",
            str(args.teacher_max_decisions),
            "--format",
            "npz",
            "--shard-size",
            "50000",
            "--workers",
            str(args.workers),
            "--chunk-games",
            "8",
            "--mixed-seats",
            "--mixed-seat-mode",
            "random",
            "--out",
            str(raw_data_dir),
        ],
        [
            args.python,
            "tools/curate_teacher_data.py",
            "--data",
            str(raw_data_dir),
            "--out",
            str(data_dir),
            "--format",
            "npz_zst",
            "--production-35m-teacher",
        ],
    ]
    if args.quality_gate != "none":
        commands.append(
            [
                args.python,
                "tools/report_teacher_data_quality.py",
                "--data",
                str(data_dir),
                "--track",
                args.track,
                "--vps-to-win",
                str(args.vps_to_win),
                "--out",
                str(teacher_quality_report),
                "--production-35m-teacher"
                if args.quality_gate == "production"
                else "--strict-35m-teacher",
            ]
        )
    bc_launcher = [args.python]
    if int(args.torchrun_nproc_per_node) > 1:
        bc_launcher = [
            args.python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={int(args.torchrun_nproc_per_node)}",
        ]
    commands.extend(
        [
        bc_launcher
        + [
            "tools/train_bc.py",
            "--arch",
            args.arch,
            "--data",
            str(data_dir),
            "--track",
            args.track,
            "--vps-to-win",
            str(args.vps_to_win),
            "--epochs",
            str(args.bc_epochs),
            "--batch-size",
            str(args.bc_batch_size),
            "--amp",
            args.bc_amp,
            "--hidden-size",
            str(args.hidden_size),
            "--lr",
            str(args.lr),
            "--soft-target-temperature",
            str(args.soft_target_temperature),
            "--soft-target-weight",
            str(args.soft_target_weight),
            "--forced-action-weight",
            str(args.forced_action_weight),
            "--phase-weights",
            args.phase_weights,
            "--winner-sample-weight",
            str(args.winner_sample_weight),
            "--loser-sample-weight",
            str(args.loser_sample_weight),
            "--value-loss-weight",
            str(args.value_loss_weight),
            "--final-vp-loss-weight",
            str(args.final_vp_loss_weight),
            "--teacher-weights",
            args.teacher_weights,
        ]
        + (
            ["--graph-tokens", str(args.graph_tokens), "--graph-layers", str(args.graph_layers)]
            if args.arch == "xdim_graph"
            else ["--graph-layers", str(args.graph_layers)]
            if args.arch == "entity_graph"
            else []
        )
        + (["--init-checkpoint", args.init_checkpoint] if args.init_checkpoint else [])
        + (["--save-each-epoch"] if args.save_each_epoch else [])
        + (
            ["--require-production-35m-teacher"]
            if args.quality_gate == "production"
            else ["--require-strict-35m-teacher"]
            if args.quality_gate == "strict"
            else []
        )
        + (
            [
                "--require-35m-model",
                "--min-35m-params",
                str(args.min_35m_params),
                "--max-35m-params",
                str(args.max_35m_params),
            ]
            if args.arch in {"xdim_graph", "entity_graph"}
            else []
        )
        + [
            "--checkpoint",
            str(bc_checkpoint),
            "--report",
            str(bc_report),
            "--device",
            args.device,
        ],
        [
            args.python,
            "tools/evaluate_scoreboard.py",
            "--candidate",
            str(bc_checkpoint),
            "--candidate-kind",
            "checkpoint",
            "--games",
            str(args.scoreboard_games),
            "--tracks",
            args.track,
            "--vps-to-win",
            str(args.vps_to_win),
            "--opponents",
            args.scoreboard_opponents,
            "--workers",
            str(max(1, args.workers // 2)),
            "--max-decisions",
            str(args.ppo_max_decisions),
            "--out",
            str(scoreboard),
        ],
    ])
    if args.allow_ppo:
        write_json(
            ppo_config,
            {
                "track": args.track,
                "algorithm": "ppo",
                "arch": "candidate",
                "seed": args.seed + 10_000,
                "run_dir": str(run_dir / "ppo"),
                "hidden_size": args.hidden_size,
                "vps_to_win": args.vps_to_win,
                "iterations": args.ppo_iterations,
                "init_checkpoint": str(bc_checkpoint),
                "rollout": {
                    "decisions_per_rank": args.ppo_decisions_per_rank,
                    "workers": args.ppo_rollout_workers,
                    "max_decisions_per_game": args.ppo_max_decisions,
                    "gamma": 0.997,
                    "gae_lambda": 0.95,
                },
                "ppo": {
                    "epochs": 2,
                    "minibatch_size": min(65536, max(1024, args.ppo_decisions_per_rank)),
                    "clip_ratio": 0.15,
                    "entropy_coef": 0.01,
                    "value_coef": 0.5,
                    "max_grad_norm": 1.0,
                    "lr": 0.0002,
                    "value_clip_range": 0.2,
                },
                "opponents": {"baseline": args.ppo_opponent},
            },
        )
        commands.append([args.python, "tools/train_selfplay_gpu.py", "--config", str(ppo_config)])

    manifest = {"run_dir": str(run_dir), "commands": commands, "allow_ppo": bool(args.allow_ppo)}
    write_json(run_dir / "pipeline_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return
    for command in commands:
        print(json.dumps({"running": command}), flush=True)
        subprocess.run(command, check=True)
    print(json.dumps({"complete": str(run_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
