#!/usr/bin/env python3
"""Seal and optionally launch the two-stage settlement-pointer commissioning run.

Stage 1 is deliberately not a candidate learner.  It starts from the exact f7
bytes after the forward-identical auxiliary-pointer upgrade, freezes every
inherited tensor, and fits only the five new auxiliary readouts.  Stage 2 is
emitted as a template but is not launchable by this tool: its coefficient must
first be filled from measured primary-vs-aux shared-trunk gradient geometry and
its initializer must be the admitted Stage-1 model bytes (never its optimizer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from catan_zero.rl.entity_token_policy import EntityGraphPolicy  # noqa: E402
from tools import a1_topology_gather_arm as temp_bridge  # noqa: E402
from tools import train_bc  # noqa: E402


SCHEMA = "a1-aux-settlement-pointer-commission-v1"
F7_SHA256 = "f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4"
WORLD_SIZE = 8
LOCAL_BATCH = 512
STAGE1_STEPS = 256
STAGE2_STEPS = 128
FREEZE_GROUPS = (
    "trunk,action_encoder,policy_head,value_heads,target_gather,edge_policy,"
    "action_cross,static_action_residual"
)
TRAINABLE_PREFIXES = (
    "aux_longest_road_head,aux_largest_army_head,aux_vp_in_n_head,"
    "aux_next_settlement_pointer_head,aux_robber_target_head"
)

# Exact empirical constant-predictor losses from the authenticated TEMP_ALL
# corpus.  Stage 1 is admitted only if every held-out conditional readout beats
# its corresponding no-feature baseline; this is the predeclared stabilization
# bound, not a post-hoc visually chosen stopping rule.
HELDOUT_STABILIZATION_MAX = {
    "aux_longest_road": 0.49779910,
    "aux_largest_army": 0.42494022,
    "aux_vp_in_n": 0.32847001,
    "aux_next_settlement": 3.81336001,
    "aux_robber_target": 2.65834001,
    "aux_subgoal_loss": 7.72290935,
}


class CommissionError(RuntimeError):
    """The requested run differs from the bounded commissioning experiment."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _set_option(command: list[str], flag: str, value: str) -> None:
    positions = [i for i, item in enumerate(command) if item == flag]
    if len(positions) > 1:
        raise CommissionError(f"duplicate option in source command: {flag}")
    if positions:
        command[positions[0] + 1] = value
    else:
        command.extend((flag, value))


def _set_switch(
    command: list[str], positive: str, negative: str, enabled: bool
) -> None:
    command[:] = [item for item in command if item not in {positive, negative}]
    command.append(positive if enabled else negative)


def _derive(
    source: Sequence[str],
    *,
    trainer: Path,
    descriptor: Path,
    sentinel: Path,
    initializer: Path,
    checkpoint: Path,
    report: Path,
    stage: int,
) -> list[str]:
    command = list(source)
    trainers = [
        i for i, value in enumerate(command) if Path(value).name == "train_bc.py"
    ]
    if len(trainers) != 1:
        raise CommissionError(
            "source command does not identify exactly one train_bc.py"
        )
    command[trainers[0]] = str(trainer)
    for flag, value in (
        ("--data", str(descriptor)),
        ("--validation-game-sentinel-manifest", str(sentinel)),
        ("--init-checkpoint", str(initializer)),
        ("--max-steps", str(STAGE1_STEPS if stage == 1 else STAGE2_STEPS)),
        ("--batch-size", str(LOCAL_BATCH)),
        ("--grad-accum-steps", "1"),
        ("--checkpoint", str(checkpoint)),
        ("--report", str(report)),
        ("--progress-every-batches", "8"),
        ("--train-diagnostics-every-batches", "16"),
        ("--objective-gradient-interference-every-batches", "0"),
    ):
        _set_option(command, flag, value)
    _set_switch(command, "--resume-optimizer", "--no-resume-optimizer", False)
    _set_switch(command, "--aux-subgoal-heads", "--no-aux-subgoal-heads", True)
    _set_switch(
        command,
        "--aux-settlement-pointer-head",
        "--no-aux-settlement-pointer-head",
        True,
    )
    if stage == 1:
        for flag, value in (
            ("--lr", "3e-4"),
            ("--lr-warmup-steps", "16"),
            ("--policy-loss-weight", "0"),
            ("--value-loss-weight", "0"),
            ("--value-lr-mult", "1"),
            ("--aux-subgoal-loss-weight", "1"),
            ("--freeze-modules", FREEZE_GROUPS),
            ("--require-only-trainable-prefixes", TRAINABLE_PREFIXES),
        ):
            _set_option(command, flag, value)
    else:
        for flag, value in (
            ("--lr", "3e-5"),
            ("--lr-warmup-steps", "100"),
            ("--policy-loss-weight", "1"),
            ("--value-loss-weight", "0.25"),
            ("--aux-subgoal-loss-weight", "__MEASURED_AUX_COEFFICIENT__"),
        ):
            _set_option(command, flag, value)
        for flag in ("--freeze-modules", "--require-only-trainable-prefixes"):
            if flag in command:
                index = command.index(flag)
                del command[index : index + 2]
    return command


def _stage1_data_contract(
    source_descriptor: Path, source_sentinel: Path, output: Path
) -> tuple[Path, Path, dict[str, Any]]:
    """Authenticate head-only overrides without changing data selection/order."""

    descriptor = json.loads(source_descriptor.read_text(encoding="utf-8"))
    if not isinstance(descriptor, dict) or not (
        descriptor.get("schema_version") == "memmap_composite_v2"
        and descriptor.get("diagnostic_only") is True
        and descriptor.get("promotion_eligible") is False
    ):
        raise CommissionError("Stage-1 source must be diagnostic memmap_composite_v2")
    source_components = descriptor.get("components")
    overrides = dict(descriptor.get("learner_recipe_overrides", {}))
    overrides.update({"lr": 3e-4, "value_loss_weight": 0.0})
    descriptor["learner_recipe_overrides"] = overrides
    descriptor["learner_recipe_overrides_sha256"] = _digest(overrides)
    derived_descriptor = output / "stage1-head-only" / "memmap_composite.json"
    _atomic_json(derived_descriptor, descriptor)

    sentinel = json.loads(source_sentinel.read_text(encoding="utf-8"))
    if not isinstance(sentinel, dict) or sentinel.get("schema_version") != (
        "train-validation-game-sentinel-v1"
    ):
        raise CommissionError("source validation sentinel schema drift")
    sentinel["source_composite_descriptor_file_sha256"] = (
        f"sha256:{_sha256(derived_descriptor)}"
    )
    sentinel["source_composite_descriptor_fingerprint"] = (
        train_bc._training_data_fingerprint(derived_descriptor, "memmap")
    )
    derived_sentinel = output / "stage1-head-only" / "validation.sentinel.json"
    _atomic_json(derived_sentinel, sentinel)

    checked = json.loads(derived_descriptor.read_text(encoding="utf-8"))
    if checked.get("components") != source_components:
        raise CommissionError("derived Stage-1 descriptor changed data identity/order")
    return (
        derived_descriptor,
        derived_sentinel,
        {
            "source_descriptor_sha256": f"sha256:{_sha256(source_descriptor)}",
            "derived_descriptor_sha256": f"sha256:{_sha256(derived_descriptor)}",
            "source_sentinel_sha256": f"sha256:{_sha256(source_sentinel)}",
            "derived_sentinel_sha256": f"sha256:{_sha256(derived_sentinel)}",
            "only_recipe_override_delta": {"lr": 3e-4, "value_loss_weight": 0.0},
            "components_and_sampling_ratios_identical": True,
            "selected_game_seed_set_sha256": sentinel.get(
                "selected_game_seed_set_sha256"
            ),
        },
    )


def _trainable_surface(checkpoint: Path) -> dict[str, Any]:
    policy = EntityGraphPolicy.load(checkpoint, device="cpu")
    model = policy.model
    train_bc._freeze_inactive_training_heads(
        model,
        final_vp_loss_weight=0.0,
        value_uncertainty_loss_weight=0.0,
        value_categorical_loss_weight=0.0,
        aux_subgoal_loss_weight=1.0,
        belief_resource_loss_weight=0.0,
    )
    train_bc._set_xdim_q_branch_trainable(model, False)
    train_bc._set_entity_graph_modules_trainable(
        model, set(FREEZE_GROUPS.split(",")), trainable=False
    )
    guarded = train_bc._require_only_trainable_prefixes(
        model, tuple(TRAINABLE_PREFIXES.split(","))
    )
    names = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    return {**guarded, "parameter_names": names}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--pointer-initializer", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--go", action="store_true")
    args = p.parse_args()

    source, _ = temp_bridge._load_temperature_source(args.source_manifest.resolve())
    initializer = args.pointer_initializer.resolve(strict=True)
    raw = __import__("torch").load(initializer, map_location="cpu", weights_only=False)
    provenance = raw.get("upgrade_provenance") if isinstance(raw, dict) else None
    if not isinstance(provenance, dict) or not (
        provenance.get("source_checkpoint_sha256") == F7_SHA256
        and provenance.get("flags")
        == {"aux_subgoal_heads": True, "aux_settlement_pointer_head": True}
        and provenance.get("forward_identical_at_init") is True
        and float(provenance.get("forward_max_diff", -1.0)) == 0.0
    ):
        raise CommissionError(
            "pointer initializer is not the exact forward-identical f7 upgrade"
        )
    if source["initialization"]["sha256"] != f"sha256:{F7_SHA256}":
        raise CommissionError("source TEMP manifest is not rooted at exact f7")

    output = args.output_root.resolve()
    stage1_dir = output / "stage1-head-only"
    stage2_dir = output / "stage2-joint-524288"
    stage1_descriptor, stage1_sentinel, stage1_data_contract = _stage1_data_contract(
        Path(source["descriptor"]["path"]),
        Path(source["validation_sentinel"]["path"]),
        output,
    )
    stage1 = _derive(
        source["command"],
        trainer=REPO / "tools/train_bc.py",
        descriptor=stage1_descriptor,
        sentinel=stage1_sentinel,
        initializer=initializer,
        checkpoint=stage1_dir / "candidate.pt",
        report=stage1_dir / "train.report.json",
        stage=1,
    )
    stage2 = _derive(
        source["command"],
        trainer=REPO / "tools/train_bc.py",
        descriptor=Path(source["descriptor"]["path"]),
        sentinel=Path(source["validation_sentinel"]["path"]),
        initializer=stage1_dir / "candidate.pt",
        checkpoint=stage2_dir / "candidate.pt",
        report=stage2_dir / "train.report.json",
        stage=2,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "checkout": {
            "path": str(REPO),
            "git_head": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
            ).strip(),
        },
        "source_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": _sha256(args.source_manifest.resolve()),
        },
        "data": source["descriptor"],
        "validation_sentinel": source["validation_sentinel"],
        "f7_parent": source["initialization"],
        "pointer_initializer": {
            "path": str(initializer),
            "sha256": f"sha256:{_sha256(initializer)}",
            "upgrade_provenance": provenance,
        },
        "stage1": {
            "purpose": "new-readout-only commissioning",
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH,
            "global_batch_size": WORLD_SIZE * LOCAL_BATCH,
            "optimizer_steps": STAGE1_STEPS,
            "row_dose": WORLD_SIZE * LOCAL_BATCH * STAGE1_STEPS,
            "fresh_adam": True,
            "discard_optimizer_after_admission": True,
            "data_contract": stage1_data_contract,
            "freeze_groups": FREEZE_GROUPS.split(","),
            "require_only_trainable_prefixes": TRAINABLE_PREFIXES.split(","),
            "trainable_surface": _trainable_surface(initializer),
            "heldout_stabilization_max": HELDOUT_STABILIZATION_MAX,
            "command": stage1,
            "command_sha256": _digest(stage1),
        },
        "stage2": {
            "launch_authorized": False,
            "authorization_requirements": [
                "stage1 held-out stabilization bound passes every head",
                "stage1 optimizer sidecars are not reused",
                "primary-vs-aux shared-trunk gradient norm ratio and cosine are recorded",
                "__MEASURED_AUX_COEFFICIENT__ is replaced and command resealed",
            ],
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH,
            "global_batch_size": WORLD_SIZE * LOCAL_BATCH,
            "optimizer_steps": STAGE2_STEPS,
            "row_dose": WORLD_SIZE * LOCAL_BATCH * STAGE2_STEPS,
            "fresh_adam": True,
            "command_template": stage2,
            "command_template_sha256": _digest(stage2),
        },
    }
    payload["plan_sha256"] = _digest(payload)
    plan = output / "plan.json"
    _atomic_json(plan, payload)
    print(
        json.dumps(
            {
                "plan": str(plan),
                "plan_sha256": payload["plan_sha256"],
                "stage1_command_sha256": payload["stage1"]["command_sha256"],
            },
            sort_keys=True,
        )
    )
    if not args.go:
        return
    stage1_dir.mkdir(parents=True, exist_ok=True)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = 65_536
    if hard < required:
        raise CommissionError(
            f"hard RLIMIT_NOFILE={hard} cannot satisfy required {required}"
        )
    if soft < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    log = (stage1_dir / "train.log").open("ab", buffering=0)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, range(WORLD_SIZE))))
    proc = subprocess.Popen(
        stage1,
        cwd=REPO,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (stage1_dir / "launcher.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    print(
        json.dumps(
            {"launched": True, "pid": proc.pid, "log": str(stage1_dir / "train.log")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
