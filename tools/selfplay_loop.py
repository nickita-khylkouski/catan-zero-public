"""Continuous self-play training loop (Step 7 orchestrator).

Runs generation -> merge -> train -> gate -> promote/rollback repeatedly,
journaling every step so the loop is resumable after a crash or host outage.

v1 scope (deliberate):
  - Single-host generation (the B200 pilot topology). Multi-host generation is
    launched manually per the gen-1 runbook and can be folded in later by
    swapping `run_generation` for a fan-out implementation.
  - Gateless-with-safety-net promotion: every completed generation's checkpoint
    becomes the next actor by default (AlphaZero-style throughput economics);
    the SPRT gate runs as the safety net and triggers ROLLBACK on a confirmed
    regression rather than gating every promotion on a win.
  - Stop conditions: --max-generations, or an Elo plateau (< --min-elo-gain
    over the last --plateau-window gated generations), or a STOP file.

Usage (pilot-scale):
  .venv/bin/python tools/selfplay_loop.py \
    --seed-checkpoint runs/bc/<hard-target>/checkpoint.pt \
    --loop-dir runs/selfplay_loop/loop1 \
    --games-per-gen 4000 --workers 42 --device cuda:1 \
    --teacher-replay runs/teacher/<curated_dir> --replay-fraction 0.15 \
    --max-generations 20

Every artifact lives under <loop-dir>/gen_NNN/ and the loop state journal at
<loop-dir>/loop_state.json records, per generation: the actor checkpoint used,
generation/merge/train/gate outcomes, wall times, and the promote/rollback
decision — the single source of truth for resume.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from sprt_gate import GATE_CONFIGS  # noqa: E402


# ----------------------------------------------------------------------- state
def load_state(loop_dir: Path) -> dict:
    path = loop_dir / "loop_state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"generations": [], "champion": None, "started_at": None}


def save_state(loop_dir: Path, state: dict) -> None:
    path = loop_dir / "loop_state.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(state, indent=2, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run(cmd: list[str], *, log_path: Path, timeout: int | None = None) -> int:
    """Run a subprocess, teeing output to a log file. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} $ {shlex.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), timeout=timeout
        )
    return proc.returncode


# ------------------------------------------------------------------- phases
def _build_generation_cmd(
    out_dir: Path, actor_checkpoint: str, gen_index: int, args: argparse.Namespace
) -> list[str]:
    # BUG-4 (CAT-88 silent-default class): pass the FULL production search/config
    # recipe EXPLICITLY. The gen script's bare defaults are WRONG for production --
    # c-scale 0.1 (not 0.03); public-observation OFF (=> omniscient/hidden-info-
    # leaked data, the f72 leak class); temperature-decisions 45 (not 90); no
    # lazy-interior-chance (=> ~65x more leaf evals). These values mirror the live
    # H100 fleet generation command exactly.
    return [
        PYTHON, "tools/generate_gumbel_selfplay_data.py",
        "--out-dir", str(out_dir),
        "--games", str(args.games_per_gen),
        "--workers", str(args.workers),
        "--checkpoint", actor_checkpoint,
        "--device", args.device,
        "--base-seed", str(args.base_seed + gen_index * 10_000_019),
        # --- production recipe (explicit; never rely on gen-script defaults) ---
        "--n-full", "64", "--n-fast", "16", "--p-full", "0.25",
        "--c-visit", "50.0", "--c-scale", "0.03",
        "--max-decisions", "600", "--max-depth", "80",
        "--temperature-decisions", "90",
        "--correct-rust-chance-spectra", "--lazy-interior-chance",
        "--public-observation",
        "--information-set-search", "--determinization-particles", "4",
        "--determinization-min-simulations", "32",
        "--track", "2p_no_trade", "--vps-to-win", "10",
        "--shard-size", "2048", "--format", "npz",
        "--score-actions",
    ]


def run_generation(gen_dir: Path, actor_checkpoint: str, gen_index: int, args: argparse.Namespace) -> dict:
    # gen_index is passed explicitly (audit fix: a mid-function load_state() re-read
    # raced with the in-memory state held by main and was one resume away from
    # reusing a generation's seed block).
    out_dir = gen_dir / "selfplay"
    code = run(
        _build_generation_cmd(out_dir, actor_checkpoint, gen_index, args),
        log_path=gen_dir / "generation.log",
    )
    manifest = out_dir / "manifest.json"
    ok = code == 0 and manifest.exists()
    rows = None
    if ok:
        try:
            rows = json.loads(manifest.read_text()).get("rows")
        except Exception:
            ok = False
    return {"ok": ok, "exit_code": code, "manifest": str(manifest), "rows": rows}


def run_merge(gen_dir: Path, gen_manifest_dir: str, args: argparse.Namespace) -> dict:
    out_dir = gen_dir / "combined"
    cmd = [
        PYTHON, "tools/build_gumbel_gen_manifest.py",
        "--gen-input", gen_manifest_dir,
        "--out", str(out_dir),
        "--seed", str(args.base_seed),
        "--replay-fraction", str(args.replay_fraction),
    ]
    if args.teacher_replay:
        cmd += ["--teacher-input", args.teacher_replay]
    code = run(cmd, log_path=gen_dir / "merge.log")
    manifest = out_dir / "manifest.json"
    return {"ok": code == 0 and manifest.exists(), "exit_code": code, "manifest": str(out_dir)}


def run_training(gen_dir: Path, data_dir: str, init_checkpoint: str, args: argparse.Namespace) -> dict:
    ckpt_dir = gen_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    code = run(
        [
            PYTHON, "tools/train_bc.py",
            "--arch", "entity_graph",
            "--hidden-size", "640", "--graph-layers", "6", "--attention-heads", "8",
            "--graph-dropout", "0.05",
            "--data", data_dir,
            "--init-checkpoint", init_checkpoint,
            "--checkpoint", str(ckpt_dir / "checkpoint.pt"),
            "--report", str(ckpt_dir / "report.json"),
            "--soft-target-source", "policy",
            "--soft-target-weight", "0.9",
            "--value-loss-weight", "1.0",
            "--final-vp-loss-weight", "0.1",
            "--policy-loss-weight", "1.0",
            "--lr", str(args.lr),
            "--lr-warmup-steps", str(args.lr_warmup_steps),
            "--trust-curated-data-quality",
            "--batch-size", str(args.batch_size),
            "--amp", "bf16",
            "--epochs", "1",
            "--device", args.device,
        ],
        log_path=gen_dir / "train.log",
    )
    ckpt = ckpt_dir / "checkpoint.pt"
    return {"ok": code == 0 and ckpt.exists(), "exit_code": code, "checkpoint": str(ckpt)}


def run_gate(gen_dir: Path, candidate: str, baseline: str, args: argparse.Namespace) -> dict:
    """SPRT safety-net gate. Returns verdict dict; 'promote'/'reject'/'continue'/'canary_promote'."""
    out = gen_dir / "gate" / "verdict.json"
    cmd = [
        PYTHON, "tools/promotion_gate_runner.py",
        "--candidate", candidate,
        "--baseline", baseline,
        "--roster", args.gate_roster,
        "--gate-config", args.gate_config,
        "--leg-dir", str(gen_dir / "gate" / "legs"),
        "--out", str(out),
    ]
    # CAT-7: --gate-games is now an override, not a default -- omit both flags
    # so promotion_gate_runner.py derives games-per-leg/min-leg-games from
    # --gate-config's base_games/max_games (flywheel: 300/600). Passing the
    # historical --gate-games 1000 default unconditionally alongside the new
    # flywheel gate-config default (max_games=600) used to derive a single
    # truncated 600-game tier that then failed the 950-game --min-leg-games
    # floor computed from the old --gate-games -- crashing every gate call.
    if args.gate_games is not None:
        cmd += ["--games-per-leg", str(args.gate_games), "--min-leg-games", str(max(50, int(args.gate_games * 0.95)))]
    code = run(
        cmd,
        log_path=gen_dir / "gate.log",
    )
    verdict = None
    h2h_elo = None
    if out.exists():
        try:
            verdict = json.loads(out.read_text())
            # Derive an H2H Elo estimate from the SPRT report (audit fix: the payload
            # has no "h2h_elo" field — keys are verdict/reason/h2h_sprt{games,wins,llr,
            # decision}/roster_deltas/...; the old code read a phantom field).
            sprt = verdict.get("h2h_sprt") or {}
            games, wins = sprt.get("games"), sprt.get("wins")
            if games and wins is not None and 0 < wins < games:
                import math
                h2h_elo = -400.0 * math.log10(games / wins - 1.0)
        except Exception:
            pass
    return {"ok": code == 0 and verdict is not None, "exit_code": code,
            "verdict": verdict, "h2h_elo": h2h_elo}


# --------------------------------------------------------------------- loop
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-checkpoint", required=True)
    p.add_argument("--loop-dir", required=True)
    p.add_argument("--games-per-gen", type=int, default=4000)
    p.add_argument("--workers", type=int, default=42)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--teacher-replay", default=None,
                   help="curated teacher corpus dir for the replay mix; annealed off after --replay-anneal-gens")
    p.add_argument("--replay-fraction", type=float, default=0.15)
    p.add_argument("--replay-anneal-gens", type=int, default=3,
                   help="teacher replay fraction is used for this many generations, then dropped to 0")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--lr-warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--gate-roster", default="catanatron_value,catanatron_ab3")
    p.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default="flywheel",
        help=(
            "Named SPRT gate config (CAT-7) passed through to "
            "promotion_gate_runner.py: 'flywheel' (elo0=-10/elo1=15, 300 "
            "games extending to 600) is the day-to-day producer gate; "
            "'certification' (elo0=0/elo1=30, 1000 games extending to 3000) "
            "is reserved for discrete public gen-N announcements."
        ),
    )
    p.add_argument(
        "--gate-games",
        type=int,
        default=None,
        help="Override --gate-config's base_games/min-leg-games. Default: derived from --gate-config.",
    )
    p.add_argument("--gate-every", type=int, default=1,
                   help="run the SPRT safety-net gate every N generations (1 = every generation)")
    p.add_argument("--max-generations", type=int, default=50)
    p.add_argument("--min-elo-gain", type=float, default=15.0)
    p.add_argument("--plateau-window", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=20260703)
    args = p.parse_args()

    loop_dir = Path(args.loop_dir)
    loop_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(loop_dir)
    if state["champion"] is None:
        state["champion"] = args.seed_checkpoint
        state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_state(loop_dir, state)

    while len(state["generations"]) < args.max_generations:
        if (loop_dir / "STOP").exists():
            print("STOP file present — exiting cleanly.", flush=True)
            return 0

        gen_index = len(state["generations"])
        gen_dir = loop_dir / f"gen_{gen_index:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        actor = state["champion"]
        record: dict = {"index": gen_index, "actor": actor,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(f"[gen {gen_index}] actor={actor}", flush=True)

        t0 = time.time()
        record["generation"] = run_generation(gen_dir, actor, gen_index, args)
        record["generation"]["wall_sec"] = round(time.time() - t0, 1)
        if not record["generation"]["ok"]:
            record["decision"] = "abort_generation_failed"
            state["generations"].append(record); save_state(loop_dir, state)
            print(f"[gen {gen_index}] GENERATION FAILED — stopping loop for operator attention.", flush=True)
            return 1

        use_replay = args.teacher_replay and gen_index < args.replay_anneal_gens
        merge_args = argparse.Namespace(**{**vars(args),
                                           "teacher_replay": args.teacher_replay if use_replay else None})
        t0 = time.time()
        record["merge"] = run_merge(gen_dir, str(gen_dir / "selfplay"), merge_args)
        record["merge"]["wall_sec"] = round(time.time() - t0, 1)
        if not record["merge"]["ok"]:
            record["decision"] = "abort_merge_failed"
            state["generations"].append(record); save_state(loop_dir, state)
            return 1

        t0 = time.time()
        record["train"] = run_training(gen_dir, record["merge"]["manifest"], actor, args)
        record["train"]["wall_sec"] = round(time.time() - t0, 1)
        if not record["train"]["ok"]:
            record["decision"] = "abort_train_failed"
            state["generations"].append(record); save_state(loop_dir, state)
            return 1
        candidate = record["train"]["checkpoint"]

        # Gateless promotion by default; SPRT safety net every N gens.
        if gen_index % args.gate_every == 0:
            t0 = time.time()
            record["gate"] = run_gate(gen_dir, candidate, actor, args)
            record["gate"]["wall_sec"] = round(time.time() - t0, 1)
            # Audit fix: the payload's promote/reject/continue lives at key "verdict",
            # not "decision" ("decision" only exists inside h2h_sprt as H0/H1/continue).
            decision = (record["gate"].get("verdict") or {}).get("verdict", "unknown")
            if decision == "reject":
                # Confirmed regression: rollback (champion unchanged), flag for operator.
                record["decision"] = "rollback_regression"
                state["generations"].append(record); save_state(loop_dir, state)
                print(f"[gen {gen_index}] SAFETY NET TRIPPED — candidate rejected, champion unchanged. "
                      f"Stopping for operator review (config change needed, not another identical run).", flush=True)
                return 2
            record["decision"] = f"promote({decision})"
        else:
            record["gate"] = None
            record["decision"] = "promote(ungated)"

        state["champion"] = candidate
        state["generations"].append(record)
        save_state(loop_dir, state)
        print(f"[gen {gen_index}] promoted → {candidate}", flush=True)

        # Plateau stop: needs gated generations with elo estimates in verdicts.
        gated = [g for g in state["generations"] if g.get("gate") and g["gate"].get("h2h_elo") is not None]
        if len(gated) >= args.plateau_window:
            recent = [g["gate"]["h2h_elo"] for g in gated[-args.plateau_window:]]
            if all(e < args.min_elo_gain for e in recent):
                print(f"[loop] PLATEAU: last {args.plateau_window} gated gens all < {args.min_elo_gain} Elo — "
                      f"stopping for config escalation (raise n_full / grow net), not more identical runs.", flush=True)
                return 3
    print("[loop] max generations reached.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
