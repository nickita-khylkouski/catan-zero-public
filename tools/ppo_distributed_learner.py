"""GPU LEARNER for distributed PPO (the actor-learner split's learner half).

The Modal actor fleet (``tools/modal_ppo_factory.py``) plays games with the latest
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
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Make the sibling ``tools/`` modules importable (factory_common, evaluate_scoreboard) whether
# the learner is launched as ``python tools/ppo_distributed_learner.py`` or ``-m``.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

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
from catan_zero.rl.torch_ppo import make_ppo_optimizer, ppo_update  # noqa: E402
from catan_zero.rl.vtrace import vtrace_from_log_probs  # noqa: E402


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
    vtrace_use_current_values: bool = True  # recompute critic values vs reuse traj.old_values
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GPU learner for distributed PPO (pull shards, KL-to-BC + V-trace, publish).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="JSON or YAML config (e.g. configs/selfplay/ppo_2p_v1.yaml). "
                        "Keys overlay the defaults below; explicit CLI flags still win.")

    g = p.add_argument_group("run wiring")
    g.add_argument("--run-base", default="runs/distributed",
                   help="Base dir holding the shared run directory (a Modal volume or NFS path).")
    g.add_argument("--run-name", default="ppo_distributed_v1",
                   help="Run name; the shared dir is {run-base}/{run-name}.")
    g.add_argument("--init-checkpoint", default=None,
                   help="BC warm-start checkpoint; also the frozen KL-to-BC anchor.")
    g.add_argument(
        "--architecture",
        choices=(CANONICAL_PPO_ARCHITECTURE,),
        default=CANONICAL_PPO_ARCHITECTURE,
        help="Canonical W7 policy architecture (legacy architectures fail closed).",
    )
    g.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N for the learner.")

    g = p.add_argument_group("pull / staleness")
    g.add_argument("--shards-per-step", type=int, default=16)
    g.add_argument("--max-staleness", type=int, default=4,
                   help="Drop shards whose policy_version < current_version - max_staleness.")
    g.add_argument("--poll-secs", type=float, default=5.0,
                   help="Sleep when no shards are available.")
    g.add_argument("--stable-secs", type=float, default=0.0,
                   help="Skip shards modified within the last N seconds (mid-write guard).")
    g.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = forever).")
    g.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Resume from the freshest checkpoints/step_{N}.pt + optimizer state "
                        "(default). Essential on Modal: every ~24h container restart would "
                        "otherwise wipe the run back to the BC warm-start.")
    g.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Always cold-start from --init-checkpoint (step 0); ignore prior progress.")

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
    g.add_argument("--target-kl", type=float, default=0.0075,
                   help="Early-stop PPO epochs once approx_kl exceeds this.")
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
    g.add_argument("--no-vtrace", dest="use_vtrace", action="store_false",
                   help="Disable V-trace; keep actor-side GAE returns/advantages (bounded-staleness PPO).")
    g.add_argument("--vtrace-clip-rho", type=float, default=1.0)
    g.add_argument("--vtrace-clip-pg-rho", type=float, default=1.0)
    g.add_argument("--vtrace-forward-chunk", type=int, default=8192,
                   help="Rows per sub-batch for the V-trace recompute forward (caps peak memory).")
    g.add_argument("--gamma", type=float, default=1.0)
    g.add_argument("--gae-lambda", type=float, default=0.95)
    g.add_argument("--vtrace-reuse-old-values", dest="vtrace_use_current_values",
                   action="store_false", default=True,
                   help="Use the stored traj.old_values as the V-trace baseline instead of "
                        "recomputing the current critic's values.")

    g = p.add_argument_group("checkpoint / eval / league")
    g.add_argument("--checkpoint-every", type=int, default=50)
    g.add_argument("--keep-last-checkpoints", type=int, default=5,
                   help="Rotation: keep the last N step_{N}.pt checkpoints (plus league-referenced "
                        "and milestone ones); delete older ones and their .opt.pt.")
    g.add_argument("--checkpoint-milestone-every", type=int, default=500,
                   help="Rotation: always keep every Nth-step checkpoint as a permanent milestone.")
    g.add_argument("--eval-games", type=int, default=200)
    g.add_argument("--eval-tracks", default="2p_no_trade")
    g.add_argument("--eval-opponents",
                   default="random,heuristic,jsettlers_lite,catanatron_ab3,catanatron_ab4")
    g.add_argument("--eval-workers", type=int, default=8,
                   help="Scoreboard eval workers; eval runs inside the learner container so a "
                        "higher value finishes faster.")
    g.add_argument("--eval-max-decisions", type=int, default=1000)
    g.add_argument("--eval-timeout-secs", type=float, default=1200.0,
                   help="Hard cap on the scoreboard subprocess; on timeout the process group is "
                        "killed and eval returns None (best-effort, never stalls training).")
    g.add_argument("--eval-device", default="cpu",
                   help="Device for the held-out scoreboard (cpu by default so eval cannot "
                        "steal the training GPU).")
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

    take(raw.get("track"), "eval_tracks")  # selfplay configs name the track at top level
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
        raise SystemExit(f"config {path} must be a mapping at top level, got {type(raw).__name__}")
    return _flatten_config(raw)


def resolve_config(argv: list[str] | None = None) -> tuple[LearnerConfig, argparse.Namespace]:
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

    if args.config:
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
        parser.error("--init-checkpoint is required (BC warm-start + frozen KL-to-BC anchor)")
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
        raise ValueError(f"canonical PPO requires clip_ratio=0.1, got {config.clip_ratio}")
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

    FIX A1 (V-trace temperature mismatch): the actor's stored ``old_log_probs`` (the V-trace
    behavior distribution) are computed from logits scaled by ``behavior_temperature`` — see
    ``ppo_update``'s ``behavior_logits = logits / behavior_temperature`` (both the entity-graph
    and flat/candidate paths). If this recompute used raw (T=1) logits instead, every V-trace
    importance ratio would be systematically wrong whenever ``behavior_temperature != 1.0``,
    independent of real policy drift. We therefore apply the IDENTICAL scale-and-clamp here
    before building the ``Categorical`` used for the target log-prob.
    """
    import torch

    # Reuse the exact batching helpers ppo_update uses so the recomputed numbers line up with
    # the actor's stored log-probs (same normalization, same masking, same context handling).
    from catan_zero.rl.torch_ppo import (
        _entity_action_column,
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

    entity_mode = (
        callable(getattr(policy, "forward_legal_np", None))
        and all(getattr(sample, "entity_features", None) is not None for sample in samples)
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

        obs_all = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
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
            if behavior_temperature != 1.0:
                logits = torch.clamp(
                    logits / behavior_temperature,
                    min=-50.0,
                    max=50.0,
                )
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            # Pull the cheap [chunk]-shaped outputs to host; the big ``logits``/``masked`` tensors
            # for this chunk are released as the loop iterates (never all held at once).
            logp_parts.append(log_probs.detach().to(dtype=torch.float64).cpu().numpy().reshape(-1))
            value_parts.append(values.detach().to(dtype=torch.float64).cpu().numpy().reshape(-1))

    target_logp = np.concatenate(logp_parts) if logp_parts else np.zeros(0, dtype=np.float64)
    current_values = np.concatenate(value_parts) if value_parts else np.zeros(0, dtype=np.float64)
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
    shaped = np.asarray(getattr(trajectory, "shaped_rewards", []) or [], dtype=np.float64)

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
    discounts[-1] = float(gamma) if bool(getattr(trajectory, "truncated", False)) else 0.0
    return discounts


# A V-trace step is SKIPPED if more than this fraction of trajectories fail the shape/finite
# checks (FIX 4): silently keeping stale actor-GAE returns for a large minority of bad
# trajectories contaminates the batch, so we bail on the whole step instead.
VTRACE_BAD_TRAJECTORY_SKIP_FRACTION = 0.10


def apply_vtrace_in_place(policy: Any, trajectories: list[Any], config: LearnerConfig) -> dict[str, float]:
    """Off-policy correct each trajectory with V-trace under the CURRENT policy, OVERWRITING
    ``traj.returns`` (<- vs) and ``traj.advantages`` (<- pg_advantages).

    Behavior log-probs are the actor's stored ``old_log_probs`` (the stale μ); target log-probs
    and (optionally) values are recomputed under the live π. NaN/inf guards protect the
    recursion from bad shards.

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
        }

    # FIX 4: trajectories are read here — assert the actor-side alignment invariant up front.
    for traj in non_empty:
        assert len(traj.old_log_probs) == len(traj.samples), (
            "PPOTrajectory.old_log_probs must align with samples "
            f"({len(traj.old_log_probs)} != {len(traj.samples)})"
        )

    # FIX 3 + H4: chunked batched forward over every step of every trajectory, then split by len.
    batched_target_logp, batched_current_values = _recompute_target_logp_and_values_batched(
        policy,
        non_empty,
        forward_chunk=getattr(config, "vtrace_forward_chunk", 8192),
        behavior_temperature=getattr(config, "behavior_temperature", 1.0),
    )

    n_steps = 0
    n_bad = 0
    offset = 0
    pending: list[tuple[Any, list[float], list[float]]] = []
    for trajectory in non_empty:
        n = len(trajectory.samples)
        target_logp = batched_target_logp[offset : offset + n]
        current_values = batched_current_values[offset : offset + n]
        offset += n

        behavior_logp = np.asarray(trajectory.old_log_probs, dtype=np.float64)
        if behavior_logp.shape[0] != n or target_logp.shape[0] != n:
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
            float(getattr(trajectory, "bootstrap_value", 0.0) or 0.0)
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
        if vs.shape[0] != n or pg.shape[0] != n or not np.isfinite(vs).all() or not np.isfinite(pg).all():
            n_bad += 1
            continue
        # Stage the result; only commit once we know the whole step is healthy (FIX 4).
        pending.append((trajectory, [float(x) for x in vs], [float(x) for x in pg]))
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
        }

    for trajectory, vs_list, pg_list in pending:
        trajectory.returns = vs_list
        trajectory.advantages = pg_list

    return {
        "vtrace_steps": float(n_steps),
        "vtrace_bad_trajectories": float(n_bad),
        "vtrace_total_trajectories": float(total),
        "vtrace_skipped": 0.0,
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
            except Exception as exc:  # corrupt / mid-write shard: skip, do not crash the learner
                # FIX M4 (quarantine): mark the bad shard consumed so it isn't re-read every poll
                # forever. Mid-write shards are rare here (atomic rename), so a persistent failure
                # means a genuinely corrupt file — drop it. Log once.
                print({"event": "shard_read_error", "shard": str(shard), "error": repr(exc)}, flush=True)
                try:
                    dist.mark_consumed(root, shard)
                except Exception:
                    pass
                continue
        shard_trajectories = envelope.get("trajectories") or []
        trajectories.extend(shard_trajectories)
    return trajectories


# --------------------------------------------------------------------------- eval + league
def run_scoreboard_eval(checkpoint_path: str, out_path: str, config: LearnerConfig) -> dict[str, Any] | None:
    """Run the held-out scoreboard (``tools/evaluate_scoreboard.py``) on ``checkpoint_path`` and
    return the parsed JSON it writes to ``out_path``. Shells out so the heavyweight eval runs in
    its own process (and on ``eval_device``, cpu by default, so it can't steal the training GPU).
    """
    import subprocess

    cmd = [
        sys.executable,
        str(_TOOLS_DIR / "evaluate_scoreboard.py"),
        "--candidate", str(checkpoint_path),
        "--candidate-kind", "checkpoint",
        "--games", str(int(config.eval_games)),
        "--tracks", str(config.eval_tracks),
        "--opponents", str(config.eval_opponents),
        "--workers", str(int(config.eval_workers)),
        "--device", str(config.eval_device),
        "--max-decisions", str(int(config.eval_max_decisions)),
        "--out", str(out_path),
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
            print({"event": "eval_error", "returncode": proc.returncode, "cmd": cmd}, flush=True)
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
        print({"event": "eval_timeout", "timeout_secs": timeout, "cmd": cmd}, flush=True)
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
        print({"event": "eval_parse_error", "error": repr(exc), "out": str(out_path)}, flush=True)
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


def cycling_check(league: League, baseline_ids: dict[str, str], main_id: str) -> dict[str, Any]:
    """Cheap non-transitivity / regression check on the payoff matrix.

    Reports:
      * ``dominant``: True if some row beats every column it has played (a dominant agent — the
        opposite of a rock-paper-scissors cycle).
      * ``min_baseline_winrate`` / ``worst_baseline``: the main's weakest matchup, to catch a
        win-rate regression against the structured bots (jsettlers/value/AB-k).
    """
    ids, matrix = league.payoff_matrix()
    info: dict[str, Any] = {"dominant": False, "min_baseline_winrate": None, "worst_baseline": None}
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
    if not name.startswith("step_") or not name.endswith(".pt") or name.endswith(".opt.pt"):
        return None
    try:
        return int(name[len("step_"):-len(".pt")])
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
    return ckpt_path.with_name(ckpt_path.name[:-len(".pt")] + ".opt.pt")


def find_resume_checkpoint(root: str | os.PathLike) -> tuple[int, Path] | None:
    """Highest ``step_{N}.pt`` to resume the TRAINABLE policy from (FIX C1). None if none exist."""
    ckpts = _list_checkpoints(root)
    return ckpts[-1] if ckpts else None


def prune_checkpoints(root: str | os.PathLike, league: League, config: LearnerConfig) -> list[str]:
    """FIX C4: bound the checkpoints dir so a multi-day run doesn't fill the volume with TBs.

    KEEP: (a) the last ``keep_last_checkpoints``, (b) every league-referenced checkpoint
    (``LeagueAgent.checkpoint_path``), (c) every ``checkpoint_milestone_every``-th step. Delete the
    rest plus their ``.opt.pt`` sidecars. A league-referenced checkpoint is NEVER deleted.
    """
    ckpts = _list_checkpoints(root)
    if not ckpts:
        return []

    keep_last = max(0, int(getattr(config, "keep_last_checkpoints", 5)))
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
        print({"event": "checkpoint_pruned", "deleted": len(deleted), "kept_steps": sorted(keep_steps)},
              flush=True)
    return deleted


def _policy_is_finite(policy: Any) -> bool:
    """FIX H6: True iff every trainable parameter is finite (no NaN/inf after the update)."""
    import torch

    model = getattr(policy, "model", None)
    params = model.parameters() if model is not None else []
    for p in params:
        if not torch.isfinite(p).all():
            return False
    return True


def _maybe_commit(volume_commit_fn: Any | None) -> None:
    """FIX H2: commit the Modal volume so actors on OTHER containers see new weights. No-op locally."""
    if volume_commit_fn is None:
        return
    try:
        volume_commit_fn()
    except Exception:  # commit is best-effort; never crash the learner on it
        pass


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
    print({"event": "learner_start", "device": device, "config": config.__dict__}, flush=True)

    root = dist.run_root(config.run_base, config.run_name)
    dist.ensure_run_dirs(root)

    _validate_w7_config(config)
    run_contract = dist.bind_run_contract(
        root,
        init_checkpoint=config.init_checkpoint,
        architecture=config.architecture,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        behavior_temperature=config.behavior_temperature,
    )
    print({"event": "run_contract", **run_contract}, flush=True)

    # 1. TRAINABLE policy + separate FROZEN anchor.  A cold start loads both
    # copies through one equality-checked factory call; a resume intentionally
    # restores the trainable checkpoint while keeping the exact initializer as
    # its immutable anchor.
    resume_ckpt = find_resume_checkpoint(root) if getattr(config, "resume", True) else None
    if resume_ckpt is not None:
        bc_anchor = load_frozen_bc_anchor(
            config.init_checkpoint, architecture=config.architecture, device=device
        )
        resume_step, resume_path = resume_ckpt
        policy = load_ppo_policy(str(resume_path), architecture=config.architecture, device=device)
        optimizer = make_ppo_optimizer(
            policy,
            learning_rate=config.lr,
            trunk_lr_mult=config.trunk_lr_mult,
        )
        opt_path = _opt_path_for(resume_path)
        if opt_path.exists():
            try:
                optimizer.load_state_dict(torch.load(str(opt_path), map_location=device))
                opt_restored = True
            except Exception as exc:  # corrupt sidecar: keep the fresh optimizer, log it
                print({"event": "resume_opt_load_error", "path": str(opt_path), "error": repr(exc)},
                      flush=True)
                opt_restored = False
        else:
            opt_restored = False
        # current_version comes from version.json (do NOT reset to 0/1 — actors track it).
        published_meta = dist.read_version(root)
        current_version = published_meta.version if published_meta is not None else resume_step
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

    # 2. League: register the live policy as `main`. On a fresh run publish step-0 weights so actors
    # can start; on resume the version line already exists and we keep ``current_version``.
    league_path = dist.league_dir(root)
    try:
        league = League.load(str(league_path))
        main_agents = [a for a in league._agents.values() if a.role == "main"]
        main_id = main_agents[0].id if main_agents else league.add_main(
            str(dist.current_weights_path(root)), step=step
        ).id
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        league = League(
            snapshot_interval=config.league_snapshot_interval,
            exploiter_promote_winrate=config.league_promote_winrate,
        )
        main_id = league.add_main(str(dist.current_weights_path(root)), step=step).id
    league.save(str(league_path))
    _maybe_commit(volume_commit_fn)

    if resume_ckpt is None:
        # Fresh run: publish the warm-started weights as the first version. FIX H2: commit so
        # actors see current.pt immediately.
        published = dist.publish_weights(root, policy.save, step=step)
        current_version = published.version
        _maybe_commit(volume_commit_fn)
        print({"event": "published", "version": current_version, "step": step}, flush=True)
    else:
        # On resume, re-publish the resumed weights so actors definitely have the trainable policy
        # (current.pt may be stale BC if the prior container died right after a checkpoint). FIX H2.
        published = dist.publish_weights(root, policy.save, step=step)
        current_version = published.version
        _maybe_commit(volume_commit_fn)
        print({"event": "republished_on_resume", "version": current_version, "step": step}, flush=True)

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
                )
            except Exception as exc:
                print({"event": "sweep_drop_policy_window_error", "error": repr(exc)}, flush=True)

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
        )
        try:
            shard_items = list(
                dist.iter_unconsumed_shards(root, newest_first=True, **iter_kwargs)  # type: ignore[call-arg]
            )
        except TypeError:
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
        )

        # 3f. publish new weights, mark shards consumed
        published = dist.publish_weights(root, policy.save, step=step + 1)
        current_version = published.version
        for shard in shard_paths:
            dist.mark_consumed(root, shard)

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
            **vtrace_stats,
        }
        print(log, flush=True)

        # 3h. periodic checkpoint + eval + league update
        if config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
            _checkpoint_eval_league(
                policy=policy,
                league=league,
                main_id=main_id,
                baseline_ids=baseline_ids,
                root=root,
                step=step + 1,
                config=config,
            )

        step += 1

    print({"event": "learner_done", "steps": step}, flush=True)


def _checkpoint_eval_league(
    *,
    policy: Any,
    league: League,
    main_id: str,
    baseline_ids: dict[str, str],
    root: Path,
    step: int,
    config: LearnerConfig,
) -> None:
    ckpt_path = dist.checkpoints_dir(root) / f"step_{step}.pt"
    policy.save(str(ckpt_path))
    print({"event": "checkpoint", "step": step, "path": str(ckpt_path)}, flush=True)

    out_path = dist.eval_dir(root) / f"scoreboard_step_{step}.json"
    report = run_scoreboard_eval(str(ckpt_path), str(out_path), config)
    if report is not None:
        record_eval_into_league(league, main_id, report, baseline_ids=baseline_ids, step=step)
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
        print({"event": "league_snapshot_main", "step": step, "frozen_id": snap.id}, flush=True)
    for agent in list(league._agents.values()):
        if agent.role in ("main_exploiter", "league_exploiter") and league.should_reset_exploiter(agent.id):
            print(
                {"event": "league_exploiter_reset_recommended", "step": step, "agent": agent.id},
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
