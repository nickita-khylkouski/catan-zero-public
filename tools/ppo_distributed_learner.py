"""GPU LEARNER for distributed PPO (the actor-learner split's learner half).

The actor fleet plays games with the latest
*published* weights and writes pickled trajectory shards into a shared run directory; this
process is the single GPU learner that:

  1. pulls *bounded-staleness* trajectory shards from the run dir,
  2. (optionally) re-weights them off-policy with IMPALA **V-trace** under the *current*
     policy, otherwise keeps the actor-side GAE targets (bounded-staleness PPO),
  3. runs a PPO update with a **KL-to-BC anchor** (a frozen behavior-cloned copy of the
     policy, delivered for free via ``ppo_update``'s ``ema_policy`` / ``ema_policy_kl_coef``),
  4. **publishes** the new weights so actors pick them up,
  5. periodically checkpoints, runs a held-out **scoreboard eval**, and updates the **League**
     (AlphaStar-style PFSP) + a cheap cycling check.

It coordinates with the actors ONLY through the on-disk contract in
``catan_zero.rl.ppo_distributed`` (paths, atomic weight versioning, shard read/consume).

This file is ADDITIVE: it imports existing interfaces and does not modify them. In particular
the BC-anchor KL term reuses the *existing* ``ema_policy``/``ema_policy_kl_coef`` args of
``ppo_update`` with a FROZEN anchor (never EMA-updated) — no edit to ``ppo_update`` is needed.

NOTE (reward shaping): v1 relies on the value-shaping baked into the actor-side
``collect_ppo_episode`` (``trajectory.shaped_rewards``). A potential-based VP shaping variant
lives in ``src/catan_zero/rl/ppo_reward_shaping.py`` and will be wired in via a shaped-collect
variant later — see the TODO in :func:`_vtrace_rewards`.

Build-only: importing or ``--help``-ing this module must not start training.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import numbers
import os
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

# Make the sibling ``tools/`` modules importable (factory_common, evaluate_scoreboard) whether
# the learner is launched as ``python tools/ppo_distributed_learner.py`` or ``-m``.
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
for _checkout_path in (_REPO_ROOT, _SRC_DIR, _TOOLS_DIR):
    while str(_checkout_path) in sys.path:
        sys.path.remove(str(_checkout_path))
    sys.path.insert(0, str(_checkout_path))

from catan_zero.rl import ppo_distributed as dist  # noqa: E402
from catan_zero.rl.config_cli import _explicit_cli_dests  # noqa: E402
from catan_zero.rl.league import League  # noqa: E402
from catan_zero.rl.ppo_policy_factory import (  # noqa: E402
    CANONICAL_PPO_ARCHITECTURE,
    load_exact_parent_and_frozen_anchor,
    load_frozen_bc_anchor,
    load_ppo_policy,
    validate_canonical_ppo_actor_contract,
    validate_canonical_ppo_staleness_contract,
)
from catan_zero.rl.ppo_run_manifest import (  # noqa: E402
    ManifestError,
    PPORunManifest,
    load_manifest,
)
from catan_zero.rl.torch_ppo import make_ppo_optimizer, ppo_update  # noqa: E402
from catan_zero.rl.vtrace import vtrace_from_log_probs  # noqa: E402


LEARNER_CHECKPOINT_SCHEMA = "ppo-distributed-learner-checkpoint-v1"


# --------------------------------------------------------------------------- config / CLI
@dataclass
class LearnerConfig:
    """Flat learner configuration. Defaults match ``configs/selfplay/ppo_2p_v1.yaml`` intent.

    Populated from argparse, then overlaid with ``--config`` YAML/JSON keys (CLI wins only
    where the user explicitly set a non-default — see :func:`_apply_config_defaults`).
    """

    # run wiring
    run_base: str
    run_name: str
    init_checkpoint: str
    architecture: str = CANONICAL_PPO_ARCHITECTURE
    device: str = "auto"
    # Initial learner RNG. In v2 this is derived from the manifest-bound actor
    # seed so the first minibatch order/dropout stream is part of run identity.
    seed: int = 1

    # pull / staleness
    shards_per_step: int = 16
    max_staleness: int = 4
    poll_secs: float = 5.0
    stable_secs: float = 0.0
    max_steps: int = 0  # 0 == run forever

    # resume (FIX C1): survive 24h Modal restarts by reloading the freshest trainable
    # checkpoint + optimizer state instead of restarting from the frozen BC.
    resume: bool = True

    # optimization
    lr: float = 2.0e-4
    trunk_lr_mult: float = 0.1
    clip_ratio: float = 0.1
    value_coef: float = 0.5
    value_trunk_grad_scale: float = 0.1
    legacy_value_trunk_grad_scale_compat: bool = False
    value_clip_range: float = 0.0
    entropy_coef: float = 0.01
    ppo_epochs: int = 2
    minibatch_size: int = 65536
    target_kl: float = 0.0075
    top_advantage_fraction: float = 1.0
    min_advantage_samples: int = 1
    behavior_temperature: float = 1.0
    advantage_normalization: str = "global"
    advantage_group_weights: str = ""

    # KL-to-BC anchor anneal
    kl_to_bc_init: float = 1.0
    kl_to_bc_final: float = 0.1
    kl_to_bc_anneal_steps: int = 2000

    # V-trace off-policy correction
    use_vtrace: bool = True
    vtrace_clip_rho: float = 1.0
    vtrace_clip_pg_rho: float = 1.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    vtrace_use_current_values: bool = (
        True  # recompute critic values vs reuse traj.old_values
    )
    # FIX H4: chunk the V-trace recompute forward into sub-batches of this many rows to bound
    # peak memory (one forward over 25k-150k rows x [action_size x ctx] would OOM the GPU).
    vtrace_forward_chunk: int = 8192

    # checkpoint / eval / league
    checkpoint_every: int = 50
    # FIX C4: checkpoint rotation. Keep the last N checkpoints, every league-referenced one, and
    # every `checkpoint_milestone_every`-th step; delete the rest (+ their .opt.pt) so the volume
    # does not fill with TBs of step_{N}.pt files over a multi-day run.
    keep_last_checkpoints: int = 5
    checkpoint_milestone_every: int = 500
    eval_games: int = 200
    eval_tracks: str = "2p_no_trade"
    eval_opponents: str = "random,heuristic,jsettlers_lite,catanatron_ab3,catanatron_ab4,catanatron_ab5,catanatron_value"
    eval_workers: int = 8
    eval_max_decisions: int = 1000
    eval_device: str = "cpu"
    # FIX C3: hard wall-clock cap on the scoreboard subprocess so a hung eval cannot stall the
    # learner indefinitely (eval is best-effort).
    eval_timeout_secs: float = 1200.0
    league_snapshot_interval: int = 200
    league_promote_winrate: float = 0.7
    run_manifest_sha256: str | None = None
    run_manifest_path: str | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GPU learner for distributed PPO (pull shards, KL-to-BC + V-trace, publish).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=None,
        help="JSON or YAML config (e.g. configs/selfplay/ppo_2p_v1.yaml). "
        "Keys overlay the defaults below; explicit CLI flags still win.",
    )
    p.add_argument(
        "--run-manifest",
        default=None,
        help=(
            "Bound canonical_entity_ppo_run_v2 manifest. In manifest mode it is "
            "the sole science authority and legacy science flags are rejected."
        ),
    )

    g = p.add_argument_group("run wiring")
    g.add_argument(
        "--run-base",
        default="runs/distributed",
        help="Base dir holding the shared run directory (a Modal volume or NFS path).",
    )
    g.add_argument(
        "--run-name",
        default="ppo_distributed_v1",
        help="Run name; the shared dir is {run-base}/{run-name}.",
    )
    g.add_argument(
        "--init-checkpoint",
        default=None,
        help="BC warm-start checkpoint; also the frozen KL-to-BC anchor.",
    )
    g.add_argument(
        "--architecture",
        choices=(CANONICAL_PPO_ARCHITECTURE,),
        default=CANONICAL_PPO_ARCHITECTURE,
        help="Canonical W7 policy architecture (legacy architectures fail closed).",
    )
    g.add_argument(
        "--device", default="auto", help="auto|cpu|cuda|cuda:N for the learner."
    )
    g.add_argument("--seed", type=int, default=1)

    g = p.add_argument_group("pull / staleness")
    g.add_argument("--shards-per-step", type=int, default=16)
    g.add_argument(
        "--max-staleness",
        type=int,
        default=4,
        help="Drop shards whose policy_version < current_version - max_staleness.",
    )
    g.add_argument(
        "--poll-secs",
        type=float,
        default=5.0,
        help="Sleep when no shards are available.",
    )
    g.add_argument(
        "--stable-secs",
        type=float,
        default=0.0,
        help="Skip shards modified within the last N seconds (mid-write guard).",
    )
    g.add_argument(
        "--max-steps", type=int, default=0, help="Stop after N steps (0 = forever)."
    )
    g.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from the freshest checkpoints/step_{N}.pt + optimizer state "
        "(default). Essential on Modal: every ~24h container restart would "
        "otherwise wipe the run back to the BC warm-start.",
    )
    g.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Always cold-start from --init-checkpoint (step 0); ignore prior progress.",
    )

    g = p.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=2.0e-4)
    g.add_argument(
        "--trunk-lr-mult",
        type=float,
        default=0.1,
        help="Shared entity-state trunk LR multiplier; policy/value heads keep --lr.",
    )
    g.add_argument("--clip-ratio", type=float, default=0.1)
    g.add_argument("--value-coef", type=float, default=0.5)
    g.add_argument(
        "--value-trunk-grad-scale",
        type=float,
        default=0.1,
        help=(
            "Scale critic gradients entering the shared entity trunk; the private "
            "value tower and head remain fully trainable."
        ),
    )
    g.add_argument(
        "--value-clip-range",
        type=float,
        default=0.0,
        help=(
            "Clip critic updates around actor old_values before computing the PPO "
            "value loss. 0 disables clipping. Useful when early PPO preserves "
            "policy KL but damages the value-trained seed."
        ),
    )
    g.add_argument("--entropy-coef", type=float, default=0.01)
    g.add_argument("--ppo-epochs", type=int, default=2)
    g.add_argument("--minibatch-size", type=int, default=65536)
    g.add_argument(
        "--target-kl",
        type=float,
        default=0.0075,
        help="Early-stop PPO epochs once approx_kl exceeds this.",
    )
    g.add_argument(
        "--behavior-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature used by actor behavior log-probs. Set this to the "
            "same value passed to run_local_entity_ppo_shards.py "
            "--action-temperature so PPO ratios compare matching distributions."
        ),
    )
    g.add_argument(
        "--advantage-normalization",
        choices=("global", "per_opponent", "none"),
        default="global",
        help=(
            "How to normalize advantages before PPO. per_opponent standardizes "
            "advantages separately for each trajectory opponent mix so value/AB3/AB4 "
            "rollouts do not distort each other's update signal."
        ),
    )
    g.add_argument(
        "--advantage-group-weights",
        default="",
        help=(
            "Optional comma-separated opponent advantage weights, e.g. "
            "catanatron_ab4=1.5,catanatron_value=1.2,catanatron_ab3=1.0. "
            "Applied after advantage normalization and before top-advantage filtering."
        ),
    )
    g.add_argument(
        "--top-advantage-fraction",
        type=float,
        default=1.0,
        help=(
            "If <1, keep only the top fraction of positive-advantage samples before "
            "normalization. This prevents a mostly losing PPO batch from reinforcing "
            "least-bad negative-outcome actions after mean-centering."
        ),
    )
    g.add_argument(
        "--min-advantage-samples",
        type=int,
        default=1,
        help="Minimum positive-advantage samples to keep when --top-advantage-fraction < 1.",
    )

    g = p.add_argument_group("KL-to-BC anchor")
    g.add_argument("--kl-to-bc-init", type=float, default=1.0)
    g.add_argument("--kl-to-bc-final", type=float, default=0.1)
    g.add_argument("--kl-to-bc-anneal-steps", type=int, default=2000)

    g = p.add_argument_group("V-trace")
    g.add_argument("--use-vtrace", dest="use_vtrace", action="store_true", default=True)
    g.add_argument(
        "--no-vtrace",
        dest="use_vtrace",
        action="store_false",
        help="Disable V-trace; keep actor-side GAE returns/advantages (bounded-staleness PPO).",
    )
    g.add_argument("--vtrace-clip-rho", type=float, default=1.0)
    g.add_argument("--vtrace-clip-pg-rho", type=float, default=1.0)
    g.add_argument(
        "--vtrace-forward-chunk",
        type=int,
        default=8192,
        help="Rows per sub-batch for the V-trace recompute forward (caps peak memory).",
    )
    g.add_argument("--gamma", type=float, default=1.0)
    g.add_argument("--gae-lambda", type=float, default=0.95)
    g.add_argument(
        "--vtrace-reuse-old-values",
        dest="vtrace_use_current_values",
        action="store_false",
        default=True,
        help="Use the stored traj.old_values as the V-trace baseline instead of "
        "recomputing the current critic's values.",
    )

    g = p.add_argument_group("checkpoint / eval / league")
    g.add_argument("--checkpoint-every", type=int, default=50)
    g.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=5,
        help="Rotation: keep the last N step_{N}.pt checkpoints (plus league-referenced "
        "and milestone ones); delete older ones and their .opt.pt.",
    )
    g.add_argument(
        "--checkpoint-milestone-every",
        type=int,
        default=500,
        help="Rotation: always keep every Nth-step checkpoint as a permanent milestone.",
    )
    g.add_argument("--eval-games", type=int, default=200)
    g.add_argument("--eval-tracks", default="2p_no_trade")
    g.add_argument(
        "--eval-opponents",
        default="random,heuristic,jsettlers_lite,catanatron_ab3,catanatron_ab4",
    )
    g.add_argument(
        "--eval-workers",
        type=int,
        default=8,
        help="Scoreboard eval workers; eval runs inside the learner container so a "
        "higher value finishes faster.",
    )
    g.add_argument("--eval-max-decisions", type=int, default=1000)
    g.add_argument(
        "--eval-timeout-secs",
        type=float,
        default=1200.0,
        help="Hard cap on the scoreboard subprocess; on timeout the process group is "
        "killed and eval returns None (best-effort, never stalls training).",
    )
    g.add_argument(
        "--eval-device",
        default="cpu",
        help="Device for the held-out scoreboard (cpu by default so eval cannot "
        "steal the training GPU).",
    )
    g.add_argument("--league-snapshot-interval", type=int, default=200)
    g.add_argument("--league-promote-winrate", type=float, default=0.7)
    return p


# Map of config-file (nested) keys -> LearnerConfig attribute. The selfplay YAML/JSON files are
# nested (rollout/ppo/eval blocks); we flatten the keys we care about and ignore the rest.
def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def take(value: Any, *attrs: str) -> None:
        if value is None:
            return
        for attr in attrs:
            flat[attr] = value

    take(raw.get("arch"), "architecture")
    take(raw.get("architecture"), "architecture")
    take(raw.get("device"), "device")
    take(raw.get("run_dir"), "run_name")  # informational; real wiring via --run-name

    rollout = raw.get("rollout", {}) or {}
    take(rollout.get("gamma"), "gamma")
    take(rollout.get("gae_lambda"), "gae_lambda")
    take(rollout.get("max_decisions_per_game"), "eval_max_decisions")

    ppo = raw.get("ppo", {}) or {}
    take(ppo.get("lr"), "lr")
    take(ppo.get("trunk_lr_mult"), "trunk_lr_mult")
    take(ppo.get("clip_ratio"), "clip_ratio")
    take(ppo.get("value_coef"), "value_coef")
    take(ppo.get("value_trunk_grad_scale"), "value_trunk_grad_scale")
    take(ppo.get("value_clip_range"), "value_clip_range")
    take(ppo.get("entropy_coef"), "entropy_coef")
    take(ppo.get("epochs"), "ppo_epochs")
    take(ppo.get("minibatch_size"), "minibatch_size")
    take(ppo.get("target_kl"), "target_kl")
    take(ppo.get("top_advantage_fraction"), "top_advantage_fraction")
    take(ppo.get("min_advantage_samples"), "min_advantage_samples")
    take(ppo.get("behavior_temperature"), "behavior_temperature")
    take(ppo.get("advantage_normalization"), "advantage_normalization")
    take(ppo.get("advantage_group_weights"), "advantage_group_weights")

    kl = raw.get("kl_to_bc", {}) or {}
    take(kl.get("init"), "kl_to_bc_init")
    take(kl.get("final"), "kl_to_bc_final")
    take(kl.get("anneal_steps"), "kl_to_bc_anneal_steps")

    vt = raw.get("vtrace", {}) or {}
    take(vt.get("enabled"), "use_vtrace")
    take(vt.get("clip_rho"), "vtrace_clip_rho")
    take(vt.get("clip_pg_rho"), "vtrace_clip_pg_rho")

    distributed = raw.get("distributed", {}) or {}
    take(distributed.get("shards_per_step"), "shards_per_step")
    take(distributed.get("max_staleness"), "max_staleness")
    take(distributed.get("poll_secs"), "poll_secs")

    ckpt = raw.get("checkpoint", {}) or {}
    take(ckpt.get("every_iterations"), "checkpoint_every")
    take(ckpt.get("keep_last"), "keep_last_checkpoints")
    take(ckpt.get("milestone_every"), "checkpoint_milestone_every")

    ev = raw.get("eval", {}) or {}
    take(ev.get("dev_games"), "eval_games")
    take(ev.get("opponents"), "eval_opponents")
    take(ev.get("tracks"), "eval_tracks")
    take(ev.get("workers"), "eval_workers")
    take(ev.get("timeout_secs"), "eval_timeout_secs")

    league = raw.get("league", {}) or {}
    take(league.get("snapshot_interval"), "league_snapshot_interval")
    take(league.get("promote_winrate"), "league_promote_winrate")

    take(
        raw.get("track"), "eval_tracks"
    )  # selfplay configs name the track at top level
    return flat


def load_config_file(path: str | os.PathLike) -> dict[str, Any]:
    """Load a JSON or (JSON-compatible) YAML config and flatten it to LearnerConfig keys."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:  # pragma: no cover - depends on optional dep
            raise SystemExit(
                f"{path} is not JSON and PyYAML is not installed; use JSON-compatible YAML"
            ) from error
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise SystemExit(
            f"config {path} must be a mapping at top level, got {type(raw).__name__}"
        )
    return _flatten_config(raw)


def _manifest_config_values(manifest: PPORunManifest) -> dict[str, Any]:
    """Translate every v2 field consumed by this learner without default fallbacks."""
    spec = manifest.spec
    actor = spec.actor
    learner = spec.learner
    checkpoint = spec.checkpoint
    evaluation = spec.evaluation
    league = spec.league
    return {
        "architecture": spec.identity.architecture,
        "gamma": actor.gamma,
        "gae_lambda": actor.gae_lambda,
        "behavior_temperature": actor.action_temperature,
        "seed": actor.seed,
        "shards_per_step": learner.shards_per_step,
        "max_staleness": learner.max_staleness,
        "max_steps": learner.max_steps,
        "resume": learner.resume,
        "lr": learner.lr,
        "trunk_lr_mult": learner.trunk_lr_mult,
        "clip_ratio": learner.clip_ratio,
        "value_coef": learner.value_coef,
        "value_trunk_grad_scale": (
            1.0
            if learner.value_trunk_grad_scale is None
            else learner.value_trunk_grad_scale
        ),
        "legacy_value_trunk_grad_scale_compat": (
            learner.value_trunk_grad_scale is None
        ),
        "value_clip_range": learner.value_clip_range,
        "entropy_coef": learner.entropy_coef,
        "ppo_epochs": learner.ppo_epochs,
        "minibatch_size": learner.minibatch_size,
        "target_kl": learner.target_kl,
        "top_advantage_fraction": learner.top_advantage_fraction,
        "min_advantage_samples": learner.min_advantage_samples,
        "advantage_normalization": learner.advantage_normalization,
        "advantage_group_weights": ",".join(learner.advantage_group_weights),
        "kl_to_bc_init": learner.kl_to_bc_init,
        "kl_to_bc_final": learner.kl_to_bc_final,
        "kl_to_bc_anneal_steps": learner.kl_to_bc_anneal_steps,
        "use_vtrace": learner.use_vtrace,
        "vtrace_clip_rho": learner.vtrace_clip_rho,
        "vtrace_clip_pg_rho": learner.vtrace_clip_pg_rho,
        "vtrace_use_current_values": learner.vtrace_use_current_values,
        "vtrace_forward_chunk": learner.vtrace_forward_chunk,
        "checkpoint_every": checkpoint.every_steps,
        "keep_last_checkpoints": checkpoint.keep_last,
        "checkpoint_milestone_every": checkpoint.milestone_every,
        "eval_games": evaluation.dev_games,
        "eval_tracks": ",".join(evaluation.tracks),
        "eval_opponents": ",".join(evaluation.opponents),
        "eval_workers": evaluation.workers,
        "eval_max_decisions": evaluation.max_decisions,
        "eval_timeout_secs": evaluation.timeout_secs,
        "eval_device": evaluation.device,
        "league_snapshot_interval": league.snapshot_interval,
        "league_promote_winrate": league.promote_winrate,
        "run_manifest_sha256": manifest.sha256(),
    }


_MANIFEST_RUNTIME_DESTS = {
    "run_manifest",
    "run_base",
    "run_name",
    "init_checkpoint",
    "device",
    "poll_secs",
    "stable_secs",
}


def resolve_config(
    argv: list[str] | None = None,
) -> tuple[LearnerConfig, argparse.Namespace]:
    """Build a LearnerConfig: argparse defaults < config file < explicit CLI flags."""
    parser = build_arg_parser()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(effective_argv)

    # Presence in argv, rather than inequality with the parser default, is the
    # only reliable way to distinguish omission from an explicit value that
    # happens to equal the default.
    explicit_dests = _explicit_cli_dests(parser, effective_argv)
    explicit = {
        dest: getattr(args, dest)
        for dest in explicit_dests
        if dest not in ("help", "config")
    }

    cfg_kwargs: dict[str, Any] = {
        "run_base": args.run_base,
        "run_name": args.run_name,
        "init_checkpoint": args.init_checkpoint or "",
        "architecture": args.architecture,
        "device": args.device,
        "seed": args.seed,
        "shards_per_step": args.shards_per_step,
        "max_staleness": args.max_staleness,
        "poll_secs": args.poll_secs,
        "stable_secs": args.stable_secs,
        "max_steps": args.max_steps,
        "resume": args.resume,
        "lr": args.lr,
        "trunk_lr_mult": args.trunk_lr_mult,
        "clip_ratio": args.clip_ratio,
        "value_coef": args.value_coef,
        "value_trunk_grad_scale": args.value_trunk_grad_scale,
        "value_clip_range": args.value_clip_range,
        "entropy_coef": args.entropy_coef,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "target_kl": args.target_kl,
        "top_advantage_fraction": args.top_advantage_fraction,
        "min_advantage_samples": args.min_advantage_samples,
        "behavior_temperature": args.behavior_temperature,
        "advantage_normalization": args.advantage_normalization,
        "advantage_group_weights": args.advantage_group_weights,
        "kl_to_bc_init": args.kl_to_bc_init,
        "kl_to_bc_final": args.kl_to_bc_final,
        "kl_to_bc_anneal_steps": args.kl_to_bc_anneal_steps,
        "use_vtrace": args.use_vtrace,
        "vtrace_clip_rho": args.vtrace_clip_rho,
        "vtrace_clip_pg_rho": args.vtrace_clip_pg_rho,
        "vtrace_forward_chunk": args.vtrace_forward_chunk,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "vtrace_use_current_values": args.vtrace_use_current_values,
        "checkpoint_every": args.checkpoint_every,
        "keep_last_checkpoints": args.keep_last_checkpoints,
        "checkpoint_milestone_every": args.checkpoint_milestone_every,
        "eval_games": args.eval_games,
        "eval_tracks": args.eval_tracks,
        "eval_opponents": args.eval_opponents,
        "eval_workers": args.eval_workers,
        "eval_max_decisions": args.eval_max_decisions,
        "eval_timeout_secs": args.eval_timeout_secs,
        "eval_device": args.eval_device,
        "league_snapshot_interval": args.league_snapshot_interval,
        "league_promote_winrate": args.league_promote_winrate,
    }

    if args.run_manifest:
        conflicts = sorted(explicit_dests - _MANIFEST_RUNTIME_DESTS)
        if conflicts:
            parser.error(
                "--run-manifest cannot be combined with legacy science/config "
                f"flags: {', '.join(conflicts)}"
            )
        if not args.init_checkpoint:
            parser.error(
                "--init-checkpoint is required with --run-manifest so its bytes "
                "can be verified"
            )
        try:
            manifest = load_manifest(args.run_manifest)
        except (OSError, ManifestError) as error:
            parser.error(f"invalid --run-manifest: {error}")
        if manifest.status != "bound":
            parser.error(
                "--run-manifest must have status='bound'; templates cannot run"
            )
        try:
            actual_initializer_sha256 = (
                f"sha256:{dist.checkpoint_sha256(args.init_checkpoint)}"
            )
        except OSError as error:
            parser.error(f"cannot hash --init-checkpoint: {error}")
        expected_initializer_sha256 = manifest.spec.identity.initializer_sha256
        if actual_initializer_sha256 != expected_initializer_sha256:
            parser.error(
                "--init-checkpoint SHA-256 does not match run manifest identity: "
                f"expected={expected_initializer_sha256} "
                f"actual={actual_initializer_sha256}"
            )
        cfg_kwargs.update(_manifest_config_values(manifest))
        cfg_kwargs["run_manifest_path"] = str(args.run_manifest)
    elif args.config:
        file_cfg = load_config_file(args.config)
        for key, value in file_cfg.items():
            if key in cfg_kwargs:
                cfg_kwargs[key] = value
        # explicit CLI flags win over the config file
        for dest, value in explicit.items():
            if dest in cfg_kwargs:
                cfg_kwargs[dest] = value

    config = LearnerConfig(**cfg_kwargs)
    if not config.init_checkpoint:
        parser.error(
            "--init-checkpoint is required (BC warm-start + frozen KL-to-BC anchor)"
        )
    try:
        _validate_w7_config(config)
    except ValueError as error:
        parser.error(str(error))
    return config, args


def _validate_w7_config(config: LearnerConfig) -> None:
    """Enforce the canonical on-policy contract before loading a checkpoint."""
    validate_canonical_ppo_actor_contract(
        architecture=config.architecture,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        action_temperature=config.behavior_temperature,
    )
    if not 2 <= int(config.ppo_epochs) <= 4:
        raise ValueError("canonical PPO requires 2-4 update epochs")
    if not 0.005 <= float(config.target_kl) <= 0.01:
        raise ValueError("canonical PPO requires target_kl in [0.005, 0.01]")
    if float(config.clip_ratio) != 0.1:
        raise ValueError(
            f"canonical PPO requires clip_ratio=0.1, got {config.clip_ratio}"
        )
    legacy_scale_compat = bool(config.legacy_value_trunk_grad_scale_compat)
    if legacy_scale_compat:
        if config.run_manifest_sha256 is None or float(config.value_trunk_grad_scale) != 1.0:
            raise ValueError(
                "legacy value-trunk scale compatibility requires an immutable "
                "pre-field v2 run manifest and value_trunk_grad_scale=1.0"
            )
    elif float(config.value_trunk_grad_scale) != 0.1:
        raise ValueError(
            "canonical PPO requires value_trunk_grad_scale=0.1, got "
            f"{config.value_trunk_grad_scale}"
        )
    if not math.isfinite(float(config.trunk_lr_mult)) or not (
        0.0 < float(config.trunk_lr_mult) <= 0.1
    ):
        raise ValueError("canonical PPO requires trunk_lr_mult in (0, 0.1]")
    validate_canonical_ppo_staleness_contract(
        use_vtrace=config.use_vtrace,
        max_staleness=config.max_staleness,
        vtrace_clip_rho=config.vtrace_clip_rho,
        vtrace_clip_pg_rho=config.vtrace_clip_pg_rho,
    )


# --------------------------------------------------------------------------- KL-to-BC anneal
def kl_to_bc_beta(step: int, *, init: float, final: float, anneal_steps: int) -> float:
    """Linearly anneal the KL-to-BC coefficient β from ``init`` to ``final``.

    β is the ``ema_policy_kl_coef`` passed to ``ppo_update`` against the FROZEN BC anchor, so a
    large β early keeps π_θ near the behavior-cloned policy (AlphaStar distillation / Cicero
    piKL) and a small late β lets RL refine it. Clamped to [min(init,final), max(init,final)].
    """
    init = float(init)
    final = float(final)
    if anneal_steps <= 0:
        return final
    frac = min(max(float(step) / float(anneal_steps), 0.0), 1.0)
    beta = init + (final - init) * frac
    lo, hi = (final, init) if init >= final else (init, final)
    return float(min(max(beta, lo), hi))


# --------------------------------------------------------------------------- V-trace re-weight
def _recompute_target_logp_and_values_batched(
    policy: Any,
    trajectories: list[Any],
    *,
    forward_chunk: int = 8192,
    behavior_temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute, under the CURRENT policy, the taken-action log-prob and critic value for
    every step of EVERY trajectory in CHUNKED batched forward passes.

    FIX 3 (efficiency): the previous implementation ran one tiny forward per trajectory
    (~128 forwards/step). Here we concatenate all trajectories' samples and process them in
    sub-batches, returning flat ``target_logp`` / ``current_values`` arrays in concatenation
    order. Callers split the results back per trajectory by length.

    FIX H4 (memory): a step can carry 25k-150k samples; one forward over ALL of them would
    allocate a [N, action_size, ctx] logits tensor of multi-GB and OOM the GPU. We slice the
    samples into sub-batches of ``forward_chunk`` rows, run ``policy.forward`` per chunk, and keep
    only the CHEAP per-row outputs (log_probs, values) — the big logits tensor for each chunk is
    freed before the next chunk. Numerically identical to the single-forward path (no cross-row
    interaction in the masking / Categorical log-prob).

    Mirrors how ``ppo_update`` recomputes log-probs: build the normalized observation batch and
    the per-sample action-context tensor, run ``policy.forward`` to get full-action logits +
    value, mask to the legal set, take ``Categorical(logits).log_prob(action)``.

    FIX A1 (V-trace behavior-distribution mismatch): entity actors always temperature-scale
    and clamp legal logits, including at T=1.  The current-policy recompute must use that exact
    transform and preserve the mixed-width padding mask or V-trace sees policy drift where none
    exists.  The legacy flat path retains its historical T=1 no-op and applies its established
    scale-and-clamp only when a non-unit behavior temperature is requested.
    """
    import torch

    # Reuse the exact batching helpers ppo_update uses so the recomputed numbers line up with
    # the actor's stored log-probs (same normalization, same masking, same context handling).
    from catan_zero.rl.torch_ppo import (
        _behavior_policy_logits,
        _entity_action_column,
        _entity_behavior_valid_mask,
        _entity_graph_outputs,
        _action_context_features_tensor,
        _masked_logits,
        _policy_observation_array,
    )

    samples = [sample for trajectory in trajectories for sample in trajectory.samples]
    n_total = len(samples)
    if n_total == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    behavior_temperature = max(float(behavior_temperature), 1.0e-6)

    entity_mode = callable(getattr(policy, "forward_legal_np", None)) and all(
        getattr(sample, "entity_features", None) is not None for sample in samples
    )
    if entity_mode:
        actions_all = torch.as_tensor(
            [_entity_action_column(sample) for sample in samples],
            dtype=torch.long,
            device=policy.device,
        )
        obs_all = None
        context_all = None
        valid_actions = None
    else:
        observations = _policy_observation_array(policy, samples)
        actions = np.asarray([s.action for s in samples], dtype=np.int64)
        valid_actions = [s.valid_actions for s in samples]

        obs_all = torch.as_tensor(
            observations, dtype=torch.float32, device=policy.device
        )
        context_all = _action_context_features_tensor(samples, policy)
        actions_all = torch.as_tensor(actions, dtype=torch.long, device=policy.device)

    chunk = int(forward_chunk) if forward_chunk and forward_chunk > 0 else n_total
    logp_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            actions_t = actions_all[start:end]
            if entity_mode:
                outputs = _entity_graph_outputs(
                    policy,
                    samples[start:end],
                    return_q=False,
                )
                logits = outputs["logits"]
                values = outputs["value"]
            else:
                obs_t = obs_all[start:end]
                context_t = context_all[start:end] if context_all is not None else None
                chunk_valid = valid_actions[start:end]
                logits, values = policy.forward(obs_t, context_t)
                logits = _masked_logits(logits, chunk_valid, policy.action_size)
            if entity_mode:
                logits = _behavior_policy_logits(
                    logits,
                    behavior_temperature,
                    valid_mask=_entity_behavior_valid_mask(
                        samples[start:end],
                        logits,
                    ),
                )
            elif behavior_temperature != 1.0:
                logits = torch.clamp(
                    logits / behavior_temperature,
                    min=-50.0,
                    max=50.0,
                )
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            # Pull the cheap [chunk]-shaped outputs to host; the big ``logits``/``masked`` tensors
            # for this chunk are released as the loop iterates (never all held at once).
            logp_parts.append(
                log_probs.detach().to(dtype=torch.float64).cpu().numpy().reshape(-1)
            )
            value_parts.append(
                values.detach().to(dtype=torch.float64).cpu().numpy().reshape(-1)
            )

    target_logp = (
        np.concatenate(logp_parts) if logp_parts else np.zeros(0, dtype=np.float64)
    )
    current_values = (
        np.concatenate(value_parts) if value_parts else np.zeros(0, dtype=np.float64)
    )
    return target_logp, current_values


def _vtrace_rewards(trajectory: Any) -> np.ndarray:
    """Per-step rewards fed to V-trace.

    FIX 1 (CRITICAL): use the REAL per-step environment reward
    (``collect_ppo_episode``'s ``trajectory.rewards``), which INCLUDES the terminal win/loss
    reward folded into the seat's final decision. The old code used ``shaped_rewards``, which is
    all-zeros when ``value_shaping_coef==0`` (the default) AND never carries the terminal
    outcome — so V-trace ``vs`` collapsed to a function of values only, with no outcome signal.

    Fallback: older shards (written before this field existed) have an empty ``rewards`` list;
    in that case we fall back to ``shaped_rewards`` and log a warning so the caller can flag it.
    Shaped rewards, when present, are layered on top of the real env reward.
    """
    n = len(trajectory.samples)
    real = np.asarray(getattr(trajectory, "rewards", []) or [], dtype=np.float64)
    shaped = np.asarray(
        getattr(trajectory, "shaped_rewards", []) or [], dtype=np.float64
    )

    if real.shape[0] == 0:
        # Older shard: no real env reward recorded. Fall back to shaped rewards (may be zeros).
        print(
            {
                "event": "vtrace_reward_fallback",
                "reason": "trajectory.rewards empty (older shard); using shaped_rewards",
                "n": int(n),
            },
            flush=True,
        )
        if shaped.shape[0] != n:
            fixed = np.zeros(n, dtype=np.float64)
            fixed[: min(n, shaped.shape[0])] = shaped[: min(n, shaped.shape[0])]
            return fixed
        return shaped

    # The real env reward already folds in the terminal outcome AND shaped per-step rewards are
    # a SEPARATE additive stream (when value-shaping is on). ``trajectory.rewards`` is built from
    # ``shaped_rewards`` + terminal env reward in ``collect_ppo_episode``, so it is complete on
    # its own; we do NOT double-add shaped rewards here.
    if real.shape[0] != n:
        fixed = np.zeros(n, dtype=np.float64)
        fixed[: min(n, real.shape[0])] = real[: min(n, real.shape[0])]
        return fixed
    return real


def _discounts_for_trajectory(trajectory: Any, *, gamma: float) -> np.ndarray:
    """``discounts[t] = gamma * (1 - done[t])`` over a flat single-player sequence.

    The actor stores one episode per trajectory. The LAST step's discount controls whether
    V-trace bootstraps past the episode boundary:

      * TERMINAL game  -> done=1 -> ``discounts[-1] = 0`` (no future credit; bootstrap_value=0).
      * TRUNCATED game -> not done -> ``discounts[-1] = gamma`` so V-trace bootstraps on
        ``trajectory.bootstrap_value`` (FIX 2: truncated episodes were previously underestimated
        by forcing a terminal-0 bootstrap at the cutoff).
    """
    n = len(trajectory.samples)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    discounts = np.full(n, float(gamma), dtype=np.float64)
    discounts[-1] = (
        float(gamma) if bool(getattr(trajectory, "truncated", False)) else 0.0
    )
    return discounts


# A V-trace step is SKIPPED if more than this fraction of trajectories fail the shape/finite
# checks (FIX 4): silently keeping stale actor-GAE returns for a large minority of bad
# trajectories contaminates the batch, so we bail on the whole step instead.
VTRACE_BAD_TRAJECTORY_SKIP_FRACTION = 0.10


@contextmanager
def _temporary_policy_eval_mode(policy: Any):
    """Make V-trace recomputation deterministic and restore caller state."""

    model = getattr(policy, "model", None)
    if model is None or not callable(getattr(model, "eval", None)):
        yield
        return
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def apply_vtrace_in_place(
    policy: Any, trajectories: list[Any], config: LearnerConfig
) -> dict[str, float]:
    """Off-policy correct each trajectory with V-trace under the CURRENT policy, OVERWRITING
    ``traj.returns`` (<- vs), ``traj.advantages`` (<- pg_advantages), and the learner-only PPO
    policy/value references (<- the current learner snapshot before the update).

    Behavior log-probs are the actor's stored ``old_log_probs`` (the stale μ); target log-probs
    and (optionally) values are recomputed under the live π. V-trace already applies the
    actor-to-learner importance ratio to ``pg_advantages``. Rebasing PPO's ``old_log_probs`` to
    the pre-update learner policy makes PPO optimize π_θ / π_preupdate instead of applying
    π_preupdate / μ a second time. Rebasing the learner-only value reference likewise makes
    value clipping measure the learner update rather than actor staleness. The actor's
    ``old_log_probs``/``old_values`` remain immutable evidence, so the correction is auditable
    and idempotent. NaN/inf guards protect the recursion from bad shards.

    FIX 3: a SINGLE batched forward pass over all trajectories' samples computes target log-probs
    and current values for every step at once; results are split back per trajectory by length,
    then the (necessarily sequential) V-trace recursion runs per sequence.

    FIX 4: bad trajectories (shape mismatch / non-finite V-trace output) are counted. If more than
    ``VTRACE_BAD_TRAJECTORY_SKIP_FRACTION`` of trajectories fail, the WHOLE step is skipped
    (``"vtrace_skipped": 1.0``) and the caller consumes the shards without an update, rather than
    mixing stale actor-GAE returns into the batch.
    """
    non_empty = [traj for traj in trajectories if len(traj.samples) > 0]
    total = len(non_empty)
    if total == 0:
        return {
            "vtrace_steps": 0.0,
            "vtrace_bad_trajectories": 0.0,
            "vtrace_total_trajectories": 0.0,
            "vtrace_skipped": 0.0,
            "vtrace_missing_current_bootstrap": 0.0,
        }

    # FIX 4: trajectories are read here — assert the actor-side alignment invariant up front.
    for traj in non_empty:
        assert len(traj.old_log_probs) == len(traj.samples), (
            "PPOTrajectory.old_log_probs must align with samples "
            f"({len(traj.old_log_probs)} != {len(traj.samples)})"
        )

    # FIX 3 + H4: chunked batched forward over every step of every trajectory, then split by len.
    with _temporary_policy_eval_mode(policy):
        batched_target_logp, batched_current_values = (
            _recompute_target_logp_and_values_batched(
                policy,
                non_empty,
                forward_chunk=getattr(config, "vtrace_forward_chunk", 8192),
                behavior_temperature=getattr(config, "behavior_temperature", 1.0),
            )
        )
        current_bootstrap_values: dict[int, float] = {}
        missing_current_bootstrap = 0
        if config.vtrace_use_current_values:
            truncated_trajectories = [
                trajectory
                for trajectory in non_empty
                if bool(getattr(trajectory, "truncated", False))
            ]
            bootstrap_samples = [
                getattr(trajectory, "bootstrap_sample", None)
                for trajectory in truncated_trajectories
            ]
            missing_current_bootstrap = sum(
                sample is None for sample in bootstrap_samples
            )
            if missing_current_bootstrap:
                return {
                    "vtrace_steps": 0.0,
                    "vtrace_bad_trajectories": float(missing_current_bootstrap),
                    "vtrace_total_trajectories": float(total),
                    "vtrace_skipped": 1.0,
                    "vtrace_missing_current_bootstrap": float(
                        missing_current_bootstrap
                    ),
                }
            if bootstrap_samples:
                _, recomputed_bootstrap_values = (
                    _recompute_target_logp_and_values_batched(
                        policy,
                        [SimpleNamespace(samples=bootstrap_samples)],
                        forward_chunk=getattr(config, "vtrace_forward_chunk", 8192),
                        behavior_temperature=getattr(
                            config, "behavior_temperature", 1.0
                        ),
                    )
                )
            else:
                recomputed_bootstrap_values = np.asarray([], dtype=np.float32)
        else:
            truncated_trajectories = []
            recomputed_bootstrap_values = np.asarray([], dtype=np.float32)
    if config.vtrace_use_current_values:
        if bootstrap_samples:
            if (
                len(recomputed_bootstrap_values) != len(truncated_trajectories)
                or not np.isfinite(recomputed_bootstrap_values).all()
            ):
                return {
                    "vtrace_steps": 0.0,
                    "vtrace_bad_trajectories": float(len(truncated_trajectories)),
                    "vtrace_total_trajectories": float(total),
                    "vtrace_skipped": 1.0,
                    "vtrace_missing_current_bootstrap": 0.0,
                }
            current_bootstrap_values = {
                id(trajectory): float(value)
                for trajectory, value in zip(
                    truncated_trajectories,
                    recomputed_bootstrap_values,
                    strict=True,
                )
            }

    n_steps = 0
    n_bad = 0
    offset = 0
    pending: list[tuple[Any, list[float], list[float], list[float], list[float]]] = []
    for trajectory in non_empty:
        n = len(trajectory.samples)
        target_logp = batched_target_logp[offset : offset + n]
        current_values = batched_current_values[offset : offset + n]
        offset += n

        behavior_logp = np.asarray(trajectory.old_log_probs, dtype=np.float64)
        if (
            behavior_logp.shape[0] != n
            or target_logp.shape[0] != n
            or current_values.shape[0] != n
            or not np.isfinite(target_logp).all()
            or not np.isfinite(current_values).all()
        ):
            n_bad += 1
            continue

        if config.vtrace_use_current_values:
            values = current_values
        else:
            values = np.asarray(trajectory.old_values, dtype=np.float64)
            if values.shape[0] != n:
                values = current_values

        rewards = _vtrace_rewards(trajectory)
        discounts = _discounts_for_trajectory(trajectory, gamma=config.gamma)

        # FIX 2: bootstrap on the seat's cutoff value for truncated games (0.0 for terminal).
        bootstrap_value = (
            (
                current_bootstrap_values[id(trajectory)]
                if config.vtrace_use_current_values
                else float(getattr(trajectory, "bootstrap_value", 0.0) or 0.0)
            )
            if bool(getattr(trajectory, "truncated", False))
            else 0.0
        )
        if not math.isfinite(bootstrap_value):
            bootstrap_value = 0.0

        # NaN/inf guards: sanitize every V-trace input; a single bad value would poison the
        # whole reverse recursion otherwise.
        behavior_logp = np.nan_to_num(behavior_logp, nan=0.0, posinf=50.0, neginf=-50.0)
        target_logp = np.nan_to_num(target_logp, nan=0.0, posinf=50.0, neginf=-50.0)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        discounts = np.nan_to_num(discounts, nan=0.0, posinf=0.0, neginf=0.0)

        out = vtrace_from_log_probs(
            behavior_log_probs=behavior_logp,
            target_log_probs=target_logp,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=config.vtrace_clip_rho,
            clip_pg_rho_threshold=config.vtrace_clip_pg_rho,
        )
        vs = np.asarray(_to_numpy(out.vs), dtype=np.float64).reshape(-1)
        pg = np.asarray(_to_numpy(out.pg_advantages), dtype=np.float64).reshape(-1)
        if (
            vs.shape[0] != n
            or pg.shape[0] != n
            or not np.isfinite(vs).all()
            or not np.isfinite(pg).all()
        ):
            n_bad += 1
            continue
        # Stage the result; only commit once we know the whole step is healthy (FIX 4).
        pending.append(
            (
                trajectory,
                [float(x) for x in vs],
                [float(x) for x in pg],
                [float(x) for x in target_logp],
                [float(x) for x in current_values],
            )
        )
        n_steps += n

    bad_fraction = (n_bad / total) if total else 0.0
    if bad_fraction > VTRACE_BAD_TRAJECTORY_SKIP_FRACTION:
        # FIX 4: too many bad trajectories — SKIP the whole step. Do NOT overwrite returns; the
        # caller consumes the shards and continues without a PPO update.
        print(
            {
                "event": "vtrace_step_skipped",
                "reason": "bad-trajectory fraction exceeds threshold",
                "n_bad": int(n_bad),
                "total": int(total),
                "bad_fraction": float(bad_fraction),
                "threshold": VTRACE_BAD_TRAJECTORY_SKIP_FRACTION,
            },
            flush=True,
        )
        return {
            "vtrace_steps": 0.0,
            "vtrace_bad_trajectories": float(n_bad),
            "vtrace_total_trajectories": float(total),
            "vtrace_skipped": 1.0,
            "vtrace_missing_current_bootstrap": float(missing_current_bootstrap),
        }

    for trajectory, vs_list, pg_list, target_logp_list, current_values_list in pending:
        trajectory.returns = vs_list
        trajectory.advantages = pg_list
        trajectory.ppo_reference_log_probs = target_logp_list
        trajectory.ppo_reference_values = current_values_list

    return {
        "vtrace_steps": float(n_steps),
        "vtrace_bad_trajectories": float(n_bad),
        "vtrace_total_trajectories": float(total),
        "vtrace_skipped": 0.0,
        "vtrace_missing_current_bootstrap": 0.0,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):  # torch tensor
        return value.detach().cpu().numpy()
    return np.asarray(value)


# --------------------------------------------------------------------------- shard ingest
def read_shards(root: str | os.PathLike, shards: Iterable[Any]) -> list[Any]:
    """Concatenate the PPOTrajectory lists from ``shards``.

    FIX 5 (efficiency): ``shards`` items may be either bare shard ``Path`` objects (legacy) or
    ``(path, envelope)`` tuples produced by ``iter_unconsumed_shards(..., with_envelope=True)``.
    When an envelope is already provided we REUSE it instead of deserializing the shard a second
    time (the staleness check in ``iter_unconsumed_shards`` already paid that cost). Unreadable
    shards are skipped, never crashing the learner.
    """
    trajectories: list[Any] = []
    for item in shards:
        envelope: Any = None
        if isinstance(item, tuple):
            shard, envelope = item
        else:
            shard = item
        if envelope is None:
            try:
                envelope = dist.read_trajectory_shard(shard)
            except (
                Exception
            ) as exc:  # corrupt / mid-write shard: skip, do not crash the learner
                # FIX M4 (quarantine): mark the bad shard consumed so it isn't re-read every poll
                # forever. Mid-write shards are rare here (atomic rename), so a persistent failure
                # means a genuinely corrupt file — drop it. Log once.
                print(
                    {
                        "event": "shard_read_error",
                        "shard": str(shard),
                        "error": repr(exc),
                    },
                    flush=True,
                )
                try:
                    dist.mark_consumed(root, shard)
                except Exception:
                    pass
                continue
        shard_trajectories = envelope.get("trajectories") or []
        trajectories.extend(shard_trajectories)
    return trajectories


# --------------------------------------------------------------------------- eval + league
def run_scoreboard_eval(
    checkpoint_path: str, out_path: str, config: LearnerConfig
) -> dict[str, Any] | None:
    """Run the held-out scoreboard (``tools/evaluate_scoreboard.py``) on ``checkpoint_path`` and
    return the parsed JSON it writes to ``out_path``. Shells out so the heavyweight eval runs in
    its own process (and on ``eval_device``, cpu by default, so it can't steal the training GPU).
    """
    import subprocess

    cmd = [
        sys.executable,
        str(_TOOLS_DIR / "evaluate_scoreboard.py"),
        "--candidate",
        str(checkpoint_path),
        "--candidate-kind",
        "checkpoint",
        "--games",
        str(int(config.eval_games)),
        "--tracks",
        str(config.eval_tracks),
        "--opponents",
        str(config.eval_opponents),
        "--workers",
        str(int(config.eval_workers)),
        "--device",
        str(config.eval_device),
        "--max-decisions",
        str(int(config.eval_max_decisions)),
        "--out",
        str(out_path),
    ]
    eval_env = os.environ.copy()
    # Scoreboard workers are CPU-bound game/eval processes. If PyTorch/BLAS inherits the
    # default thread count inside every worker, a small eval can oversubscribe the host by
    # hundreds of runnable threads. Keep learner-triggered evals predictable; callers that
    # want wider CPU parallelism should increase --eval-workers instead.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        eval_env[key] = "1"
    timeout = float(getattr(config, "eval_timeout_secs", 1200.0) or 0.0)
    # FIX C3: a hung scoreboard must NEVER stall the learner. Run in its own process GROUP so a
    # timeout kills the whole subprocess tree (the scoreboard fans out to worker procs), then
    # log eval_timeout and return None — eval is best-effort.
    proc = None
    try:
        proc = subprocess.Popen(cmd, start_new_session=True, env=eval_env)
        proc.wait(timeout=timeout if timeout > 0 else None)
        if proc.returncode != 0:
            print(
                {"event": "eval_error", "returncode": proc.returncode, "cmd": cmd},
                flush=True,
            )
            return None
    except subprocess.TimeoutExpired:
        # Kill the entire process group, then reap so we don't leak a zombie.
        try:
            os.killpg(os.getpgid(proc.pid), 15)  # SIGTERM the group
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), 9)  # SIGKILL if it won't die
                proc.wait(timeout=10)
        except (ProcessLookupError, OSError):
            pass
        print(
            {"event": "eval_timeout", "timeout_secs": timeout, "cmd": cmd}, flush=True
        )
        return None
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, OSError):
                pass
        print({"event": "eval_error", "error": repr(exc), "cmd": cmd}, flush=True)
        return None
    try:
        return json.loads(Path(out_path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            {"event": "eval_parse_error", "error": repr(exc), "out": str(out_path)},
            flush=True,
        )
        return None


def record_eval_into_league(
    league: League,
    main_id: str,
    report: dict[str, Any],
    *,
    baseline_ids: dict[str, str],
    step: int,
) -> None:
    """Record scoreboard win-rates as League matches (main vs each fixed baseline).

    Baselines (random/heuristic/jsettlers_lite/...) are registered as ``frozen`` league agents
    keyed by opponent name so the payoff matrix tracks the main's progress against each. The
    scoreboard ``win_rate`` is recorded directly as ``a_score`` (already in [0,1]).
    """
    for entry in report.get("results", []):
        opponent = str(entry.get("opponent", ""))
        if not opponent:
            continue
        win_rate = entry.get("win_rate")
        if win_rate is None:
            continue
        win_rate = float(min(max(win_rate, 0.0), 1.0))
        opp_id = baseline_ids.get(opponent)
        if opp_id is None:
            agent = league.snapshot(main_id, f"baseline::{opponent}", step=step)
            opp_id = agent.id
            baseline_ids[opponent] = opp_id
        league.record_match(main_id, opp_id, win_rate)


def cycling_check(
    league: League, baseline_ids: dict[str, str], main_id: str
) -> dict[str, Any]:
    """Cheap non-transitivity / regression check on the payoff matrix.

    Reports:
      * ``dominant``: True if some row beats every column it has played (a dominant agent — the
        opposite of a rock-paper-scissors cycle).
      * ``min_baseline_winrate`` / ``worst_baseline``: the main's weakest matchup, to catch a
        win-rate regression against the structured bots (jsettlers/value/AB-k).
    """
    ids, matrix = league.payoff_matrix()
    info: dict[str, Any] = {
        "dominant": False,
        "min_baseline_winrate": None,
        "worst_baseline": None,
    }
    if matrix.size:
        for i in range(matrix.shape[0]):
            row = matrix[i]
            played = ~np.isnan(row)
            if played.any() and np.all(row[played] > 0.5):
                info["dominant"] = True
                break
    worst_name = None
    worst_rate = None
    index = {agent_id: i for i, agent_id in enumerate(ids)}
    main_i = index.get(main_id)
    if main_i is not None and matrix.size:
        for name, opp_id in baseline_ids.items():
            j = index.get(opp_id)
            if j is None:
                continue
            rate = matrix[main_i, j]
            if np.isnan(rate):
                continue
            if worst_rate is None or rate < worst_rate:
                worst_rate = float(rate)
                worst_name = name
    info["min_baseline_winrate"] = worst_rate
    info["worst_baseline"] = worst_name
    return info


# --------------------------------------------------------------------------- resume helpers
def _checkpoint_step(path: Path) -> int | None:
    """Parse N from a ``step_{N}.pt`` checkpoint filename (ignores ``.opt.pt`` sidecars)."""
    name = path.name
    if (
        not name.startswith("step_")
        or not name.endswith(".pt")
        or name.endswith(".opt.pt")
    ):
        return None
    try:
        return int(name[len("step_") : -len(".pt")])
    except ValueError:
        return None


def _list_checkpoints(root: str | os.PathLike) -> list[tuple[int, Path]]:
    """All ``step_{N}.pt`` trainable checkpoints in checkpoints_dir, ascending by step."""
    cdir = dist.checkpoints_dir(root)
    if not Path(cdir).exists():
        return []
    out: list[tuple[int, Path]] = []
    for path in Path(cdir).glob("step_*.pt"):
        step = _checkpoint_step(path)
        if step is not None:
            out.append((step, path))
    out.sort(key=lambda pair: pair[0])
    return out


def _opt_path_for(ckpt_path: Path) -> Path:
    """The optimizer-state sidecar path for a ``step_{N}.pt`` checkpoint: ``step_{N}.opt.pt``."""
    return ckpt_path.with_name(ckpt_path.name[: -len(".pt")] + ".opt.pt")


def _capture_rng_state() -> dict[str, Any]:
    """Capture every process-global RNG used by PPO minibatch/update code."""
    import torch

    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "torch_cuda_device_count": torch.cuda.device_count(),
    }


def _restore_rng_state(state: Any) -> None:
    """Restore a validated checkpoint RNG bundle or fail closed."""
    import torch

    if not isinstance(state, dict):
        raise RuntimeError("refusing PPO resume: RNG state is missing or malformed")
    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "torch_cuda_device_count",
    }
    if not required.issubset(state):
        raise RuntimeError("refusing PPO resume: RNG state is incomplete")
    saved_cuda_count = state["torch_cuda_device_count"]
    cuda_states = state["torch_cuda"]
    if not isinstance(saved_cuda_count, int) or not isinstance(cuda_states, list):
        raise RuntimeError("refusing PPO resume: CUDA RNG state is malformed")
    current_cuda_count = torch.cuda.device_count()
    if saved_cuda_count != current_cuda_count or len(cuda_states) != current_cuda_count:
        raise RuntimeError(
            "refusing PPO resume: CUDA device count does not match checkpoint RNG state"
        )
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if cuda_states:
            torch.cuda.set_rng_state_all(
                [cuda_state.cpu() for cuda_state in cuda_states]
            )
    except Exception as error:
        raise RuntimeError(
            f"refusing PPO resume: cannot restore checkpoint RNG state: {error}"
        ) from error


def _relative_shard_frontier(
    root: str | os.PathLike,
    shards: Iterable[str | os.PathLike],
) -> list[str]:
    """Bind consumed inputs to paths below this run's trajectories directory."""
    base = dist.trajectories_dir(root).resolve()
    frontier: list[str] = []
    for shard in shards:
        try:
            relative = Path(shard).resolve().relative_to(base)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"shard is outside trajectories directory: {shard}"
            ) from error
        if not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe shard frontier path: {shard}")
        frontier.append(relative.as_posix())
    return frontier


def _validate_consumed_frontier(frontier: Any) -> list[str]:
    if not isinstance(frontier, list) or not all(
        isinstance(item, str) for item in frontier
    ):
        raise RuntimeError("refusing PPO resume: consumed-shard frontier is malformed")
    for item in frontier:
        path = Path(item)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise RuntimeError(
                "refusing PPO resume: unsafe consumed-shard frontier path"
            )
    return frontier


def _finalize_consumed_frontier(
    root: str | os.PathLike,
    frontier: Iterable[str],
) -> None:
    """Idempotently finish the shard deletions committed by a recovery checkpoint."""
    base = dist.trajectories_dir(root).resolve()
    for relative_text in _validate_consumed_frontier(list(frontier)):
        shard = (base / relative_text).resolve()
        try:
            shard.relative_to(base)
        except ValueError as error:
            raise RuntimeError(
                "refusing PPO resume: unsafe consumed-shard frontier path"
            ) from error
        dist.mark_consumed(root, shard)
        marker = dist.consumed_dir(root) / relative_text.replace("/", "__")
        if shard.exists() or not marker.exists():
            raise RuntimeError(f"failed to finalize consumed PPO shard: {shard}")


def _save_checkpoint_set(
    *,
    policy: Any,
    optimizer: Any,
    root: str | os.PathLike,
    step: int,
    consumed_shards: Iterable[str | os.PathLike] = (),
) -> tuple[Path, Path]:
    """Atomically expose a model and its exact learner-state sidecar as one generation."""
    import torch

    completed_step = int(step)
    if completed_step <= 0:
        raise ValueError("PPO checkpoint step must be positive")
    checkpoint_dir = dist.checkpoints_dir(root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step_{completed_step}.pt"
    optimizer_path = _opt_path_for(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(
            f"refusing to overwrite PPO checkpoint: {checkpoint_path}"
        )
    # An optimizer-only file is unreachable because resume discovery keys off the model.
    # It can only be an interrupted attempt at this same step, so clear it before retrying.
    if optimizer_path.exists():
        optimizer_path.unlink()

    frontier = _relative_shard_frontier(root, consumed_shards)
    rng_state = _capture_rng_state()
    unique = f"{os.getpid()}.{time.time_ns()}"
    checkpoint_tmp = checkpoint_path.with_name(f".{checkpoint_path.name}.{unique}.tmp")
    optimizer_tmp = optimizer_path.with_name(f".{optimizer_path.name}.{unique}.tmp")
    try:
        policy.save(str(checkpoint_tmp))
        if not checkpoint_tmp.is_file() or checkpoint_tmp.stat().st_size <= 0:
            raise RuntimeError("PPO policy did not write a non-empty checkpoint")
        checkpoint_sha256 = dist.checkpoint_sha256(checkpoint_tmp)
        torch.save(
            {
                "schema": LEARNER_CHECKPOINT_SCHEMA,
                "step": completed_step,
                "checkpoint_sha256": checkpoint_sha256,
                "optimizer_state": optimizer.state_dict(),
                "rng_state": rng_state,
                "consumed_shards": frontier,
            },
            optimizer_tmp,
        )
        if not optimizer_tmp.is_file() or optimizer_tmp.stat().st_size <= 0:
            raise RuntimeError("PPO optimizer did not write a non-empty sidecar")
        # The model is the discovery/commit point. An interrupted new-step write
        # leaves at most an ignored optimizer-only sidecar.
        os.replace(optimizer_tmp, optimizer_path)
        os.replace(checkpoint_tmp, checkpoint_path)
    finally:
        checkpoint_tmp.unlink(missing_ok=True)
        optimizer_tmp.unlink(missing_ok=True)
    return checkpoint_path, optimizer_path


def _restore_optimizer_checkpoint(
    *,
    policy: Any,
    optimizer: Any,
    checkpoint_path: Path,
    step: int,
    map_location: str,
) -> dict[str, Any]:
    """Restore the optimizer paired with ``checkpoint_path`` or refuse resume."""
    import torch

    optimizer_path = _opt_path_for(checkpoint_path)
    if not optimizer_path.is_file():
        raise RuntimeError(
            "refusing PPO resume: discovered model checkpoint has no optimizer "
            f"sidecar: {checkpoint_path} -> {optimizer_path}"
        )
    try:
        payload = torch.load(
            str(optimizer_path), map_location=map_location, weights_only=False
        )
    except TypeError:  # torch<2.1 compatibility
        payload = torch.load(str(optimizer_path), map_location=map_location)
    except Exception as error:
        raise RuntimeError(
            f"refusing PPO resume: cannot load optimizer sidecar {optimizer_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("refusing PPO resume: optimizer sidecar is not a mapping")
    if payload.get("schema") != LEARNER_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            "refusing PPO resume: optimizer sidecar schema is missing or unsupported"
        )
    if payload.get("step") != int(step):
        raise RuntimeError(
            "refusing PPO resume: optimizer sidecar step does not match model checkpoint"
        )
    if payload.get("checkpoint_sha256") != dist.checkpoint_sha256(checkpoint_path):
        raise RuntimeError(
            "refusing PPO resume: optimizer sidecar binds different model bytes"
        )
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise RuntimeError(
            "refusing PPO resume: optimizer state is missing or malformed"
        )
    _validate_consumed_frontier(payload.get("consumed_shards"))
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        raise RuntimeError("refusing PPO resume: RNG state is missing or malformed")
    try:
        optimizer.load_state_dict(optimizer_state)
    except Exception as error:
        raise RuntimeError(
            f"refusing PPO resume: optimizer state is incompatible: {error}"
        ) from error
    _assert_finite_update(
        policy=policy,
        optimizer=optimizer,
        update_stats={},
    )
    return payload


def _checkpoint_schedule(
    config: LearnerConfig, *, completed_step: int
) -> tuple[bool, bool]:
    """Return periodic-evaluation and bounded-terminal reasons for a recovered step."""
    step = int(completed_step)
    if step <= 0:
        raise ValueError("completed PPO step must be positive")
    periodic = (
        int(config.checkpoint_every) > 0 and step % int(config.checkpoint_every) == 0
    )
    terminal = int(config.max_steps) > 0 and step >= int(config.max_steps)
    return periodic, terminal


def find_resume_checkpoint(root: str | os.PathLike) -> tuple[int, Path] | None:
    """Highest ``step_{N}.pt`` to resume the TRAINABLE policy from (FIX C1). None if none exist."""
    ckpts = _list_checkpoints(root)
    return ckpts[-1] if ckpts else None


def prune_checkpoints(
    root: str | os.PathLike, league: League, config: LearnerConfig
) -> list[str]:
    """FIX C4: bound the checkpoints dir so a multi-day run doesn't fill the volume with TBs.

    KEEP: (a) the last ``keep_last_checkpoints``, (b) every league-referenced checkpoint
    (``LeagueAgent.checkpoint_path``), (c) every ``checkpoint_milestone_every``-th step. Delete the
    rest plus their ``.opt.pt`` sidecars. A league-referenced checkpoint is NEVER deleted.
    """
    ckpts = _list_checkpoints(root)
    if not ckpts:
        return []

    # Per-update recovery requires at least the newest generation even when an
    # invalid/over-aggressive config requests zero retained checkpoints.
    keep_last = max(1, int(getattr(config, "keep_last_checkpoints", 5)))
    milestone = int(getattr(config, "checkpoint_milestone_every", 500))

    keep_steps: set[int] = set()
    # (a) last N by step
    for step, _ in ckpts[-keep_last:] if keep_last > 0 else []:
        keep_steps.add(step)
    # (c) milestones
    if milestone > 0:
        for step, _ in ckpts:
            if step % milestone == 0:
                keep_steps.add(step)

    # (b) league-referenced paths (resolved, so equality is robust to relative/abs differences).
    referenced: set[str] = set()
    for agent in list(getattr(league, "_agents", {}).values()):
        cp = getattr(agent, "checkpoint_path", None)
        if cp:
            try:
                referenced.add(str(Path(cp).resolve()))
            except OSError:
                referenced.add(str(cp))

    deleted: list[str] = []
    for step, path in ckpts:
        if step in keep_steps:
            continue
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in referenced:
            continue  # NEVER delete a league-referenced checkpoint
        try:
            path.unlink()
            deleted.append(str(path))
        except OSError:
            continue
        opt = _opt_path_for(path)
        try:
            if opt.exists():
                opt.unlink()
        except OSError:
            pass
    if deleted:
        print(
            {
                "event": "checkpoint_pruned",
                "deleted": len(deleted),
                "kept_steps": sorted(keep_steps),
            },
            flush=True,
        )
    return deleted


def _assert_finite_nested(value: Any, *, path: str) -> None:
    """Reject a non-finite numeric leaf in nested durable learner state."""

    import torch

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise FloatingPointError(f"refusing PPO recovery commit: non-finite {path}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise FloatingPointError(f"refusing PPO recovery commit: non-finite {path}")
        return
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value)):
            raise FloatingPointError(f"refusing PPO recovery commit: non-finite {path}")
        return
    if isinstance(value, numbers.Complex) and not isinstance(value, bool):
        if not (math.isfinite(float(value.real)) and math.isfinite(float(value.imag))):
            raise FloatingPointError(f"refusing PPO recovery commit: non-finite {path}")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite_nested(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_finite_nested(nested, path=f"{path}[{index}]")


def _assert_finite_update(
    *,
    policy: Any,
    optimizer: Any,
    update_stats: Mapping[str, Any],
) -> None:
    """Refuse to durably commit a poisoned model/optimizer/metric generation."""

    import torch
    from torch import nn

    policy_attributes = getattr(policy, "__dict__", {})
    checked_modules: set[int] = set()
    for attribute, value in policy_attributes.items():
        if not isinstance(value, nn.Module) or id(value) in checked_modules:
            continue
        checked_modules.add(id(value))
        _assert_finite_nested(
            value.state_dict(),
            path=f"policy.{attribute}",
        )
    for attribute, value in policy_attributes.items():
        if isinstance(value, torch.Tensor):
            _assert_finite_nested(value, path=f"policy.{attribute}")

    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group.get("params", ())):
            _assert_finite_nested(
                parameter,
                path=(
                    f"optimizer.param_groups[{group_index}].params[{parameter_index}]"
                ),
            )
            if parameter.grad is not None:
                _assert_finite_nested(
                    parameter.grad,
                    path=(
                        f"optimizer.param_groups[{group_index}]"
                        f".params[{parameter_index}].grad"
                    ),
                )

    _assert_finite_nested(optimizer.state_dict(), path="optimizer")
    _assert_finite_nested(update_stats, path="update_stats")


def _maybe_commit(volume_commit_fn: Any | None) -> None:
    """FIX H2: commit the Modal volume so actors on OTHER containers see new weights. No-op locally."""
    if volume_commit_fn is None:
        return
    try:
        volume_commit_fn()
    except Exception:  # commit is best-effort; never crash the learner on it
        pass


def _commit_or_raise(volume_commit_fn: Any | None, *, operation: str) -> None:
    """Make a learner transaction visible or stop before its dependent mutation."""
    if volume_commit_fn is None:
        return
    try:
        volume_commit_fn()
    except Exception as error:
        raise RuntimeError(f"PPO volume commit failed after {operation}") from error


def _commit_recovery_update(
    *,
    policy: Any,
    optimizer: Any,
    root: str | os.PathLike,
    completed_step: int,
    shard_paths: Iterable[str | os.PathLike],
    update_stats: Mapping[str, Any],
    volume_commit_fn: Any | None,
) -> tuple[Any, Path, Path]:
    """Durably checkpoint, publish, then finalize one applied PPO update."""
    _assert_finite_update(
        policy=policy,
        optimizer=optimizer,
        update_stats=update_stats,
    )
    shards = list(shard_paths)
    frontier = _relative_shard_frontier(root, shards)
    checkpoint_path, optimizer_path = _save_checkpoint_set(
        policy=policy,
        optimizer=optimizer,
        root=root,
        step=completed_step,
        consumed_shards=shards,
    )
    _commit_or_raise(volume_commit_fn, operation="recovery checkpoint")
    published = dist.publish_weights(root, policy.save, step=completed_step)
    _commit_or_raise(volume_commit_fn, operation="weight publication")
    _finalize_consumed_frontier(root, frontier)
    _commit_or_raise(volume_commit_fn, operation="shard frontier finalization")
    return published, checkpoint_path, optimizer_path


def _bind_configured_run_identity(
    root: str | os.PathLike, config: LearnerConfig
) -> dict[str, Any]:
    """Bind either the exact v2 manifest or the untouched historical v1 contract."""
    if config.run_manifest_path is not None:
        try:
            manifest = load_manifest(config.run_manifest_path)
        except (OSError, ManifestError) as error:
            raise RuntimeError(
                f"cannot reload bound PPO run manifest: {error}"
            ) from error
        if (
            manifest.status != "bound"
            or manifest.sha256() != config.run_manifest_sha256
        ):
            raise RuntimeError(
                "bound PPO run manifest changed after configuration resolution"
            )
        return dist.bind_run_manifest(root, manifest)
    return dist.bind_run_contract(
        root,
        init_checkpoint=config.init_checkpoint,
        architecture=config.architecture,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        behavior_temperature=config.behavior_temperature,
    )


# --------------------------------------------------------------------------- train loop
def train(
    config: LearnerConfig,
    *,
    volume_reload_fn: Any | None = None,
    volume_commit_fn: Any | None = None,
) -> None:
    """Run the GPU learner loop.

    FIX 6: ``volume_reload_fn`` (optional) is called before each shard scan so a learner running
    as a Modal function can ``volume.reload()`` to see actor writes to the shared volume. It is a
    no-op when ``None`` (plain-process / local runs), so behavior is unchanged by default. The
    Modal GPU wrapper passes ``volume.reload`` here.

    FIX H2: ``volume_commit_fn`` (optional) is called immediately after every ``publish_weights``
    and after each checkpoint/league save so actors on OTHER Modal containers actually see the new
    ``current.pt``. Without a commit the writes stay container-local and the fleet trains on stale
    weights forever. No-op when ``None`` (local runs); the Modal GPU wrapper passes ``volume.commit``.

    FIX C1 (resume): on restart we DON'T blindly reload the frozen BC and reset step=0 (that would
    discard all training progress on every ~24h Modal container recycle). Instead we scan for the
    freshest ``checkpoints/step_{N}.pt``, load THAT as the trainable policy, restore the optimizer
    from its ``.opt.pt`` sidecar, set ``step=N`` and ``current_version`` from ``read_version``. The
    BC anchor still loads from ``config.init_checkpoint`` (the anchor is ALWAYS the BC).
    """
    import torch

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        {"event": "learner_start", "device": device, "config": config.__dict__},
        flush=True,
    )

    root = dist.run_root(config.run_base, config.run_name)
    dist.ensure_run_dirs(root)

    _validate_w7_config(config)
    run_contract = _bind_configured_run_identity(root, config)
    print({"event": "run_contract", **run_contract}, flush=True)
    if config.legacy_value_trunk_grad_scale_compat:
        print(
            {
                "event": "legacy_value_trunk_grad_scale_compat",
                "value_trunk_grad_scale": config.value_trunk_grad_scale,
                "reason": "pre-field immutable canonical_entity_ppo_run_v2 manifest",
            },
            flush=True,
        )

    # 1. TRAINABLE policy + separate FROZEN anchor.  A cold start loads both
    # copies through one equality-checked factory call; a resume intentionally
    # restores the trainable checkpoint while keeping the exact initializer as
    # its immutable anchor.
    resume_ckpt = (
        find_resume_checkpoint(root) if getattr(config, "resume", True) else None
    )
    if resume_ckpt is None:
        random.seed(int(config.seed))
        np.random.seed(int(config.seed))
        torch.manual_seed(int(config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed))
    if resume_ckpt is not None:
        bc_anchor = load_frozen_bc_anchor(
            config.init_checkpoint, architecture=config.architecture, device=device
        )
        resume_step, resume_path = resume_ckpt
        policy = load_ppo_policy(
            str(resume_path), architecture=config.architecture, device=device
        )
        optimizer = make_ppo_optimizer(
            policy,
            learning_rate=config.lr,
            trunk_lr_mult=config.trunk_lr_mult,
        )
        resume_payload = _restore_optimizer_checkpoint(
            policy=policy,
            optimizer=optimizer,
            checkpoint_path=resume_path,
            step=resume_step,
            map_location=device,
        )
        opt_restored = True
        # current_version comes from version.json (do NOT reset to 0/1 — actors track it).
        published_meta = dist.read_version(root)
        current_version = (
            published_meta.version if published_meta is not None else resume_step
        )
        step = int(resume_step)
        print(
            {
                "event": "resume",
                "step": step,
                "checkpoint": str(resume_path),
                "optimizer_restored": opt_restored,
                "current_version": current_version,
            },
            flush=True,
        )
    else:
        # Cold start: exact corrected parent, separately loaded frozen anchor,
        # fresh optimizer, step 0.
        policy, bc_anchor = load_exact_parent_and_frozen_anchor(
            config.init_checkpoint,
            architecture=config.architecture,
            device=device,
        )
        optimizer = make_ppo_optimizer(
            policy,
            learning_rate=config.lr,
            trunk_lr_mult=config.trunk_lr_mult,
        )
        step = 0
        current_version = 0
        resume_payload = None

    # 2. League: register the live policy as `main`. On a fresh run publish step-0 weights so actors
    # can start; on resume the version line already exists and we keep ``current_version``.
    league_path = dist.league_dir(root)
    try:
        league = League.load(str(league_path))
        main_agents = [a for a in league._agents.values() if a.role == "main"]
        main_id = (
            main_agents[0].id
            if main_agents
            else league.add_main(str(dist.current_weights_path(root)), step=step).id
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        league = League(
            snapshot_interval=config.league_snapshot_interval,
            exploiter_promote_winrate=config.league_promote_winrate,
        )
        main_id = league.add_main(str(dist.current_weights_path(root)), step=step).id
    league.save(str(league_path))
    _maybe_commit(volume_commit_fn)

    if resume_ckpt is None:
        # Actor bootstrap may already have published this exact initializer and
        # generated version-1 trajectories. Reuse that publication atomically;
        # republishing the same model as v2 would discard the entire first dose
        # when exact-version/no-V-trace learning is selected.
        published, newly_published = dist.publish_initial_weights(root, policy.save)
        current_version = published.version
        if newly_published:
            _maybe_commit(volume_commit_fn)
        print(
            {
                "event": (
                    "published" if newly_published else "reused_actor_bootstrap"
                ),
                "version": current_version,
                "step": step,
            },
            flush=True,
        )
    else:
        # The recovery checkpoint commits this input frontier before publishing or
        # deleting it. Finish that deletion idempotently, then republish the exact
        # recovered model. Restore RNG last so setup I/O cannot perturb the next update.
        _finalize_consumed_frontier(root, resume_payload["consumed_shards"])
        _commit_or_raise(volume_commit_fn, operation="resume frontier finalization")
        published = dist.publish_weights(root, policy.save, step=step)
        current_version = published.version
        _commit_or_raise(volume_commit_fn, operation="resume weight publication")
        _restore_rng_state(resume_payload["rng_state"])
        print(
            {
                "event": "republished_on_resume",
                "version": current_version,
                "step": step,
            },
            flush=True,
        )

    baseline_ids: dict[str, str] = {}
    while config.max_steps <= 0 or step < config.max_steps:
        # 3a. pull bounded-staleness shards. FIX 5: with_envelope reuses the deserialized
        # envelope so read_shards does not deserialize each shard a second time. FIX 6:
        # volume_reload_fn lets a Modal-hosted learner see volume writes before scanning.
        min_version = (
            current_version
            if not config.use_vtrace
            else max(0, current_version - config.max_staleness)
        )
        max_version = current_version

        # FIX C2 (drop the WHOLE stale backlog, not just the scanned window): once per step sweep
        # every unconsumed shard below min_version to the consumed marker so a backlog of stale
        # shards cannot pile up unboundedly. The backbone fn may not exist yet (parallel edit), so
        # call it defensively and just fall back to the per-scan staleness drop if absent.
        sweep_fn = getattr(dist, "sweep_drop_outside_policy_window", None)
        if callable(sweep_fn):
            try:
                sweep_fn(
                    root,
                    min_policy_version=min_version,
                    max_policy_version=max_version,
                    expected_run_manifest_sha256=config.run_manifest_sha256,
                )
            except Exception as exc:
                print(
                    {"event": "sweep_drop_policy_window_error", "error": repr(exc)},
                    flush=True,
                )

        # FIX C2 (train on the FRESHEST data): ask for newest-first so the scanned window is the
        # freshest shards, not the stalest. ``newest_first`` may not exist yet (parallel edit) —
        # fall back to the default oldest-first iteration if the kwarg is rejected.
        iter_kwargs = dict(
            max_shards=config.shards_per_step,
            min_policy_version=min_version,
            max_policy_version=max_version,
            stable_secs=config.stable_secs,
            with_envelope=True,
            volume_reload_fn=volume_reload_fn,
            expected_run_manifest_sha256=config.run_manifest_sha256,
        )
        try:
            shard_items = list(
                dist.iter_unconsumed_shards(root, newest_first=True, **iter_kwargs)  # type: ignore[call-arg]
            )
        except TypeError:
            if config.run_manifest_sha256 is not None:
                raise
            shard_items = list(dist.iter_unconsumed_shards(root, **iter_kwargs))
        if not shard_items:
            time.sleep(config.poll_secs)
            continue
        shard_paths = [item[0] for item in shard_items]

        # 3b. read -> trajectories (reusing the already-deserialized envelopes). FIX M4: if a
        # shard's envelope failed to deserialize during the scan (envelope is None) it gets
        # re-read here; read_shards quarantines any that still raise. We additionally mark any
        # envelope-less item consumed so a permanently-corrupt shard is not retried every poll.
        trajectories = read_shards(root, shard_items)
        for shard, envelope in shard_items:
            if envelope is None:
                # The scan couldn't decode this shard; read_shards re-tried and logged. Mark it
                # consumed so the learner doesn't re-encounter the same corrupt shard forever.
                dist.mark_consumed(root, shard)
        if not trajectories or not any(t.samples for t in trajectories):
            for shard in shard_paths:
                dist.mark_consumed(root, shard)
            continue

        # 3c. V-trace off-policy correction (overwrites returns/advantages), else keep GAE.
        # FIX 4: when too many trajectories are bad, apply_vtrace_in_place flags vtrace_skipped;
        # we consume the shards and skip the PPO update entirely rather than train on a
        # contaminated batch.
        vtrace_stats: dict[str, float] = {}
        if config.use_vtrace:
            vtrace_stats = apply_vtrace_in_place(policy, trajectories, config)
            if vtrace_stats.get("vtrace_skipped", 0.0) >= 1.0:
                print(
                    {
                        "event": "step_skipped_bad_vtrace",
                        "step": step,
                        "n_shards": len(shard_paths),
                        **vtrace_stats,
                    },
                    flush=True,
                )
                for shard in shard_paths:
                    dist.mark_consumed(root, shard)
                continue

        # 3d. KL-to-BC β anneal
        beta = kl_to_bc_beta(
            step,
            init=config.kl_to_bc_init,
            final=config.kl_to_bc_final,
            anneal_steps=config.kl_to_bc_anneal_steps,
        )

        # 3e. PPO update with the frozen BC anchor as the KL magnet (via ema_policy args).
        stats = ppo_update(
            policy,
            trajectories,
            learning_rate=config.lr,
            clip_ratio=config.clip_ratio,
            value_coef=config.value_coef,
            value_clip_range=config.value_clip_range,
            entropy_coef=config.entropy_coef,
            epochs=config.ppo_epochs,
            minibatch_size=config.minibatch_size,
            optimizer=optimizer,
            ema_policy=bc_anchor,
            ema_policy_kl_coef=beta,
            target_kl=config.target_kl,
            top_advantage_fraction=config.top_advantage_fraction,
            min_advantage_samples=config.min_advantage_samples,
            behavior_temperature=config.behavior_temperature,
            advantage_normalization=config.advantage_normalization,
            advantage_group_weights=config.advantage_group_weights,
            value_trunk_grad_scale=config.value_trunk_grad_scale,
        )

        # 3f. Every applied update is a recovery point. Commit model + optimizer
        # + RNG + exact input frontier before actors can observe the update or
        # those inputs can be deleted.
        completed_step = step + 1
        periodic_evaluation, terminal_step = _checkpoint_schedule(
            config, completed_step=completed_step
        )
        published, checkpoint_path, optimizer_path = _commit_recovery_update(
            policy=policy,
            optimizer=optimizer,
            root=root,
            completed_step=completed_step,
            shard_paths=shard_paths,
            update_stats=stats,
            volume_commit_fn=volume_commit_fn,
        )
        print(
            {
                "event": "checkpoint",
                "step": completed_step,
                "path": str(checkpoint_path),
                "optimizer_path": str(optimizer_path),
                "terminal": terminal_step,
            },
            flush=True,
        )

        current_version = published.version

        # 3g. log
        mean_return = _mean_return(trajectories)
        log = {
            "event": "step",
            "step": step,
            "version": current_version,
            "n_shards": len(shard_paths),
            "n_trajectories": len(trajectories),
            "samples": stats.get("samples", 0.0),
            "policy_loss": stats.get("policy_loss", 0.0),
            "value_loss": stats.get("value_loss", 0.0),
            "entropy": stats.get("entropy", 0.0),
            "approx_kl": stats.get("approx_kl", 0.0),
            "ema_policy_kl": stats.get("ema_policy_kl", 0.0),
            "beta": beta,
            "mean_return": mean_return,
            "use_vtrace": config.use_vtrace,
            "value_trunk_grad_scale": config.value_trunk_grad_scale,
            **vtrace_stats,
        }
        print(log, flush=True)

        # 3h. Evaluation is periodic and strictly after the recovery transaction.
        if periodic_evaluation:
            _checkpoint_eval_league(
                policy=policy,
                optimizer=optimizer,
                league=league,
                main_id=main_id,
                baseline_ids=baseline_ids,
                root=root,
                step=completed_step,
                config=config,
                checkpoint_path=checkpoint_path,
            )
        prune_checkpoints(root, league, config)
        _maybe_commit(volume_commit_fn)

        step += 1

    print({"event": "learner_done", "steps": step}, flush=True)


def _checkpoint_eval_league(
    *,
    policy: Any,
    optimizer: Any,
    league: League,
    main_id: str,
    baseline_ids: dict[str, str],
    root: Path,
    step: int,
    config: LearnerConfig,
    checkpoint_path: Path | None = None,
    consumed_shards: Iterable[str | os.PathLike] = (),
    run_evaluation: bool = True,
) -> None:
    if checkpoint_path is None:
        ckpt_path, optimizer_path = _save_checkpoint_set(
            policy=policy,
            optimizer=optimizer,
            root=root,
            step=step,
            consumed_shards=consumed_shards,
        )
        print(
            {
                "event": "checkpoint",
                "step": step,
                "path": str(ckpt_path),
                "optimizer_path": str(optimizer_path),
            },
            flush=True,
        )
    else:
        ckpt_path = checkpoint_path

    if not run_evaluation:
        return

    out_path = dist.eval_dir(root) / f"scoreboard_step_{step}.json"
    report = run_scoreboard_eval(str(ckpt_path), str(out_path), config)
    if report is not None:
        record_eval_into_league(
            league, main_id, report, baseline_ids=baseline_ids, step=step
        )
        league.save(str(dist.league_dir(root)))
        ids, matrix = league.payoff_matrix()
        cyc = cycling_check(league, baseline_ids, main_id)
        print(
            {
                "event": "eval",
                "step": step,
                "payoff_ids": ids,
                "payoff_matrix": matrix.tolist(),
                "cycling": cyc,
            },
            flush=True,
        )

    # League hooks: log the recommendations. Full main-snapshot / exploiter-reset cells (with
    # actual head-to-head play) come later; for now we surface the decision the league would make.
    if league.should_snapshot_main(main_id, step):
        snap = league.snapshot(main_id, str(ckpt_path), step=step)
        league.save(str(dist.league_dir(root)))
        print(
            {"event": "league_snapshot_main", "step": step, "frozen_id": snap.id},
            flush=True,
        )
    for agent in list(league._agents.values()):
        if agent.role in (
            "main_exploiter",
            "league_exploiter",
        ) and league.should_reset_exploiter(agent.id):
            print(
                {
                    "event": "league_exploiter_reset_recommended",
                    "step": step,
                    "agent": agent.id,
                },
                flush=True,
            )


def _mean_return(trajectories: list[Any]) -> float:
    total = 0.0
    count = 0
    for trajectory in trajectories:
        for ret in trajectory.returns:
            value = float(ret)
            if math.isfinite(value):
                total += value
                count += 1
    return total / count if count else 0.0


# --------------------------------------------------------------------------- entrypoint
def main(argv: list[str] | None = None) -> None:
    config, _args = resolve_config(argv)
    train(config)


if __name__ == "__main__":
    main()
