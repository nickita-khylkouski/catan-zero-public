#!/usr/bin/env python3
"""Run the narrow, single-node KLENT-direct arm of R&D experiment E5.

This runner intentionally does not pretend that the other E5 arms are wired.
Search distillation still uses the repository's existing generator and BC
learner, but there is not yet one tested transaction joining those programs to
this campaign format.  The only accepted arm here is therefore
``klent-direct``.

The runner consumes an explicit seed manifest, requires a checkpoint trained
for public observations, resets the otherwise-untrained action-Q output layer
to zero for a fresh KLENT run, and atomically publishes a checkpoint,
optimizer sidecar, and report after every iteration.  Resumption is only from
one of those complete iteration transactions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

from catan_zero.rl.entity_token_policy import EntityGraphPolicy  # noqa: E402
from catan_zero.rl.klent import KLENTConfig  # noqa: E402
from catan_zero.rl.klent_train import (  # noqa: E402
    collect_trajectory,
    update_entity_policy,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig  # noqa: E402
from catan_zero.rl.optim_state import (  # noqa: E402
    load_optimizer_state,
    optimizer_sidecar_path,
    save_optimizer_state,
)
from tools.factory_common import write_json  # noqa: E402


SCHEMA_VERSION = "catan-zero-e5-klent-run/v1"
SEED_SCHEMA_VERSION = "catan-zero-e5-seeds/v1"
SUPPORTED_ARM = "klent-direct"


@dataclass(frozen=True, slots=True)
class E5KLENTConfig:
    arm: str
    init_checkpoint: str
    init_checkpoint_sha256: str
    seed_manifest: str
    seed_manifest_sha256: str
    seeds: tuple[int, ...]
    run_dir: str
    device: str
    iterations: int
    games_per_iteration: int
    max_decisions: int
    learning_rate: float
    weight_decay: float
    epochs: int
    minibatch_size: int
    gradient_clip_norm: float
    value_loss_weight: float
    entropy_coefficient: float
    reverse_kl_coefficient: float
    trace_horizon: float
    q_loss_weight: float
    q_init: str
    max_truncation_fraction: float

    def validate(self) -> None:
        if self.arm != SUPPORTED_ARM:
            raise ValueError(
                f"only {SUPPORTED_ARM!r} is runnable; search-distillation and "
                "distill-then-klent remain blocked on a tested public-safe transaction"
            )
        if self.iterations < 1 or self.games_per_iteration < 1:
            raise ValueError("iterations and games_per_iteration must be positive")
        required = self.iterations * self.games_per_iteration
        if len(self.seeds) != required:
            raise ValueError(
                f"seed manifest must contain exactly {required} seeds; got {len(self.seeds)}"
            )
        if self.max_decisions < 1 or self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError(
                "max_decisions, epochs, and minibatch_size must be positive"
            )
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("gradient_clip_norm", self.gradient_clip_norm),
            ("value_loss_weight", self.value_loss_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        if self.q_init != "zero-output":
            raise ValueError("fresh KLENT runs currently require q_init='zero-output'")
        if (
            not math.isfinite(self.max_truncation_fraction)
            or not 0.0 <= self.max_truncation_fraction <= 1.0
        ):
            raise ValueError("max_truncation_fraction must be finite and in [0, 1]")
        KLENTConfig(
            entropy_coefficient=self.entropy_coefficient,
            reverse_kl_coefficient=self.reverse_kl_coefficient,
            trace_horizon=self.trace_horizon,
            q_loss_weight=self.q_loss_weight,
        ).validate()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_seed_manifest(path: str | Path) -> tuple[int, ...]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read seed manifest {source}: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SEED_SCHEMA_VERSION
    ):
        raise ValueError(
            f"seed manifest schema_version must be {SEED_SCHEMA_VERSION!r}"
        )
    raw = payload.get("seeds")
    if not isinstance(raw, list) or not raw:
        raise ValueError("seed manifest seeds must be a non-empty array")
    seeds: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"seed manifest seeds[{index}] must be an integer >= 0")
        seeds.append(int(value))
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed manifest seeds must be unique")
    return tuple(seeds)


def _reset_q_output_head(policy: Any) -> None:
    """Make initial Q exactly zero without deadening the upstream Q features."""

    q_head = getattr(policy.model, "q_head", None)
    if q_head is None or len(q_head) < 1:
        raise RuntimeError("KLENT requires an EntityGraph action-Q head")
    output = q_head[-1]
    if not isinstance(output, torch.nn.Linear) or output.out_features != 1:
        raise RuntimeError("unrecognized EntityGraph action-Q output layer")
    with torch.no_grad():
        output.weight.zero_()
        if output.bias is not None:
            output.bias.zero_()


def _atomic_save_policy(policy: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        policy.save(
            temporary,
            mask_hidden_info=True,
            soft_target_source="klent-direct",
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(
                f"policy checkpoint temp file was not written: {temporary}"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _iteration_rng_seed(iteration: int, seeds: Sequence[int]) -> int:
    """Derive one stable RNG seed from the iteration's durable seed slice."""

    encoded = json.dumps(
        {"iteration": int(iteration), "seeds": [int(seed) for seed in seeds]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _seed_iteration_rng(iteration: int, seeds: Sequence[int]) -> int:
    """Make an iteration reproduce identically after a checkpoint resume."""

    seed = _iteration_rng_seed(iteration, seeds)
    random.seed(seed)
    # Some downstream code still uses NumPy's legacy process-global RNG.
    import numpy as np

    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _assert_finite_training_state(policy: Any, optimizer: Any) -> None:
    """Refuse to publish a checkpoint containing non-finite model/Adam state."""

    for name, tensor in policy.model.state_dict().items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise RuntimeError(f"KLENT model state is non-finite after update: {name}")
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and (
                value.is_floating_point() or value.is_complex()
            ):
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(
                        "KLENT optimizer state is non-finite after update: "
                        f"parameter_id={id(parameter)} field={name}"
                    )


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain=v1"], cwd=_REPO_ROOT, text=True
            )
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _hardware(device: str) -> dict[str, Any]:
    resolved = torch.device(device)
    payload: dict[str, Any] = {"device": str(resolved), "cuda": resolved.type == "cuda"}
    if resolved.type == "cuda":
        index = (
            resolved.index
            if resolved.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "accelerator_model": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    return payload


def _resolved_config_payload(config: E5KLENTConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["seeds"] = list(config.seeds)
    return {"schema_version": SCHEMA_VERSION, "config": payload}


def _load_resume_state(
    run_dir: Path, config_payload: dict[str, Any]
) -> tuple[int, dict]:
    resolved_path = run_dir / "config.resolved.json"
    report_path = run_dir / "report.json"
    if not resolved_path.is_file() or not report_path.is_file():
        raise RuntimeError("resume requires config.resolved.json and report.json")
    if json.loads(resolved_path.read_text(encoding="utf-8")) != config_payload:
        raise RuntimeError(
            "resume configuration does not exactly match the recorded run"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    iterations = report.get("iterations")
    if not isinstance(iterations, list):
        raise RuntimeError("resume report has no iteration list")
    completed = len(iterations)
    configured_iterations = int(config_payload["config"]["iterations"])
    if completed > configured_iterations:
        raise RuntimeError("resume report contains more iterations than the run config")
    if report.get("status") == "complete" and completed != configured_iterations:
        raise RuntimeError(
            "complete resume report does not contain every configured iteration"
        )
    if [row.get("iteration") for row in iterations] != list(range(1, completed + 1)):
        raise RuntimeError("resume report iteration sequence is invalid")
    if completed:
        checkpoint = run_dir / "checkpoints" / f"iter_{completed:04d}.pt"
        sidecar = optimizer_sidecar_path(checkpoint)
        if not checkpoint.is_file() or not sidecar.is_file():
            raise RuntimeError("resume iteration transaction is incomplete")
        recorded = iterations[-1].get("checkpoint", {})
        if recorded.get("sha256") != sha256_file(checkpoint) or recorded.get(
            "optimizer_sha256"
        ) != sha256_file(sidecar):
            raise RuntimeError(
                "resume iteration checkpoint hash does not match its report"
            )
    return completed, report


def _validate_final_transaction(run_dir: Path, report: dict[str, Any]) -> None:
    final = run_dir / "final.pt"
    sidecar = optimizer_sidecar_path(final)
    recorded = report.get("final_checkpoint")
    if not isinstance(recorded, dict) or not final.is_file() or not sidecar.is_file():
        raise RuntimeError("complete run is missing its final checkpoint transaction")
    if recorded.get("sha256") != sha256_file(final) or recorded.get(
        "optimizer_sha256"
    ) != sha256_file(sidecar):
        raise RuntimeError("final checkpoint hash does not match the complete report")


def _finalize_run(
    run_dir: Path,
    config: E5KLENTConfig,
    report: dict[str, Any],
    *,
    started: float,
) -> dict[str, Any]:
    last = run_dir / "checkpoints" / f"iter_{config.iterations:04d}.pt"
    final = run_dir / "final.pt"
    _copy_atomic(last, final)
    _copy_atomic(optimizer_sidecar_path(last), optimizer_sidecar_path(final))
    report.update(
        {
            "status": "complete",
            "completed_iterations": config.iterations,
            "total_games": sum(row["games"] for row in report["iterations"]),
            "total_decisions": sum(row["decisions"] for row in report["iterations"]),
            "final_checkpoint": {
                "path": str(final),
                "sha256": sha256_file(final),
                "optimizer_path": str(optimizer_sidecar_path(final)),
                "optimizer_sha256": sha256_file(optimizer_sidecar_path(final)),
            },
            "invocation_wall_time_sec": time.monotonic() - started,
        }
    )
    write_json(run_dir / "report.json", report)
    return report


def train_klent_direct(
    config: E5KLENTConfig, *, resume: bool = False
) -> dict[str, Any]:
    started = time.monotonic()
    config.validate()
    if sha256_file(config.init_checkpoint) != config.init_checkpoint_sha256:
        raise RuntimeError("initial checkpoint hash does not match the resolved config")
    if sha256_file(config.seed_manifest) != config.seed_manifest_sha256:
        raise RuntimeError("seed manifest hash does not match the resolved config")
    if load_seed_manifest(config.seed_manifest) != config.seeds:
        raise RuntimeError("seed manifest contents do not match the resolved config")
    run_dir = Path(config.run_dir).resolve()
    config_payload = _resolved_config_payload(config)
    if resume:
        completed, report = _load_resume_state(run_dir, config_payload)
        if completed >= config.iterations:
            if report.get("status") == "complete":
                _validate_final_transaction(run_dir, report)
                return report
            return _finalize_run(run_dir, config, report, started=started)
        load_path = (
            Path(config.init_checkpoint)
            if completed == 0
            else run_dir / "checkpoints" / f"iter_{completed:04d}.pt"
        )
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"fresh run directory is not empty: {run_dir}")
        completed = 0
        load_path = Path(config.init_checkpoint)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "arm": SUPPORTED_ARM,
            "initial_checkpoint": {
                "path": str(Path(config.init_checkpoint).resolve()),
                "sha256": config.init_checkpoint_sha256,
            },
            "seed_manifest": {
                "path": str(Path(config.seed_manifest).resolve()),
                "sha256": config.seed_manifest_sha256,
                "schema_version": SEED_SCHEMA_VERSION,
                "seeds": list(config.seeds),
            },
            "q_initialization": "zero-output",
            "information_regime": "public_observation_policy",
            "git": _git_provenance(),
            "hardware": _hardware(config.device),
            "iterations": [],
        }

    policy = EntityGraphPolicy.load(load_path, device=config.device)
    if not bool(getattr(policy, "trained_with_masked_hidden_info", False)):
        raise RuntimeError(
            "KLENT requires an initialization checkpoint trained with masked hidden information"
        )
    starting_from_init = not resume or completed == 0
    if starting_from_init:
        _reset_q_output_head(policy)
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.resolved.json", config_payload)
        write_json(run_dir / "report.json", report)

    optimizer = torch.optim.AdamW(
        policy.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if (
        resume
        and completed > 0
        and not load_optimizer_state(load_path, policy.model, optimizer, None)
    ):
        raise RuntimeError(
            "resume refused because the KLENT optimizer sidecar could not load"
        )

    klent_config = KLENTConfig(
        entropy_coefficient=config.entropy_coefficient,
        reverse_kl_coefficient=config.reverse_kl_coefficient,
        trace_horizon=config.trace_horizon,
        q_loss_weight=config.q_loss_weight,
    )
    env_config = ColonistMultiAgentConfig(
        players=2,
        vps_to_win=10,
        max_player_trade_offers_per_turn=0,
        enable_table_chat=False,
        allow_free_text_chat=False,
    )
    for iteration_index in range(completed, config.iterations):
        offset = iteration_index * config.games_per_iteration
        seeds = config.seeds[offset : offset + config.games_per_iteration]
        iteration_rng_seed = _seed_iteration_rng(iteration_index + 1, seeds)
        iteration_started = time.monotonic()
        trajectories = [
            collect_trajectory(
                policy,
                seed=seed,
                env_config=env_config,
                config=klent_config,
                max_decisions=config.max_decisions,
            )
            for seed in seeds
        ]
        truncated_games = sum(trajectory.truncated for trajectory in trajectories)
        truncation_fraction = truncated_games / len(trajectories)
        if truncation_fraction > config.max_truncation_fraction:
            refusal = {
                "iteration": iteration_index + 1,
                "seeds": list(seeds),
                "iteration_rng_seed": iteration_rng_seed,
                "games": len(trajectories),
                "terminal_games": len(trajectories) - truncated_games,
                "truncated_games": truncated_games,
                "truncation_fraction": truncation_fraction,
                "max_truncation_fraction": config.max_truncation_fraction,
                "decisions": sum(len(trajectory.steps) for trajectory in trajectories),
                "reason": "insufficient_terminal_outcome_signal",
                "wall_time_sec": time.monotonic() - iteration_started,
            }
            report["status"] = "refused"
            report["refusal"] = refusal
            write_json(run_dir / "report.json", report)
            raise RuntimeError(
                "KLENT iteration truncation fraction exceeds configured maximum: "
                f"{truncation_fraction:.6f} > {config.max_truncation_fraction:.6f}"
            )
        update = update_entity_policy(
            policy,
            trajectories,
            optimizer,
            config=klent_config,
            epochs=config.epochs,
            minibatch_size=config.minibatch_size,
            value_loss_weight=config.value_loss_weight,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=seeds[0],
        )
        numeric_metrics = (
            float(update[key])
            for key in ("loss", "policy_loss", "q_loss", "value_loss")
        )
        if int(update.get("updates", 0)) < 1 or not all(
            math.isfinite(x) for x in numeric_metrics
        ):
            raise RuntimeError(
                "KLENT iteration produced no update or non-finite metrics"
            )
        _assert_finite_training_state(policy, optimizer)

        checkpoint = run_dir / "checkpoints" / f"iter_{iteration_index + 1:04d}.pt"
        _atomic_save_policy(policy, checkpoint)
        sidecar = save_optimizer_state(checkpoint, policy.model, optimizer, None)
        if sidecar is None or not sidecar.is_file():
            raise RuntimeError(
                "optimizer sidecar save failed; iteration was not published"
            )
        iteration_report = {
            "iteration": iteration_index + 1,
            "seeds": list(seeds),
            "iteration_rng_seed": iteration_rng_seed,
            "games": len(trajectories),
            "terminal_games": sum(
                not trajectory.truncated for trajectory in trajectories
            ),
            "truncated_games": truncated_games,
            "truncation_fraction": truncation_fraction,
            "decisions": sum(len(trajectory.steps) for trajectory in trajectories),
            "update": update,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "optimizer_path": str(sidecar),
                "optimizer_sha256": sha256_file(sidecar),
            },
            "wall_time_sec": time.monotonic() - iteration_started,
        }
        report["iterations"].append(iteration_report)
        report["status"] = "running"
        write_json(
            run_dir / "iterations" / f"iter_{iteration_index + 1:04d}.json",
            iteration_report,
        )
        write_json(run_dir / "report.json", report)

    return _finalize_run(run_dir, config, report, started=started)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=SUPPORTED_ARM)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iteration", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--entropy-coefficient", type=float, default=0.03)
    parser.add_argument("--reverse-kl-coefficient", type=float, default=0.1)
    parser.add_argument("--trace-horizon", type=float, default=8.0)
    parser.add_argument("--q-loss-weight", type=float, default=1.0)
    parser.add_argument("--q-init", choices=("zero-output",), default="zero-output")
    parser.add_argument("--max-truncation-fraction", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> E5KLENTConfig:
    checkpoint = Path(args.init_checkpoint).resolve()
    manifest = Path(args.seed_manifest).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"initial checkpoint is not a file: {checkpoint}")
    if not manifest.is_file():
        raise ValueError(f"seed manifest is not a file: {manifest}")
    config = E5KLENTConfig(
        arm=str(args.arm),
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=sha256_file(checkpoint),
        seed_manifest=str(manifest),
        seed_manifest_sha256=sha256_file(manifest),
        seeds=load_seed_manifest(manifest),
        run_dir=str(Path(args.run_dir).resolve()),
        device=str(args.device),
        iterations=int(args.iterations),
        games_per_iteration=int(args.games_per_iteration),
        max_decisions=int(args.max_decisions),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        minibatch_size=int(args.minibatch_size),
        gradient_clip_norm=float(args.gradient_clip_norm),
        value_loss_weight=float(args.value_loss_weight),
        entropy_coefficient=float(args.entropy_coefficient),
        reverse_kl_coefficient=float(args.reverse_kl_coefficient),
        trace_horizon=float(args.trace_horizon),
        q_loss_weight=float(args.q_loss_weight),
        q_init=str(args.q_init),
        max_truncation_fraction=float(args.max_truncation_fraction),
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = train_klent_direct(config_from_args(args), resume=bool(args.resume))
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(Path(args.run_dir) / "report.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
