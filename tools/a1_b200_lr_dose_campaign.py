#!/usr/bin/env python3
"""Run the independent-parent 8xB200 A1 LR/dose campaign.

The four 128-step arms differ only in LR and warmup. Every arm replays the
sealed one-dose transaction from the same explicitly hash-bound parent with a
fresh optimizer. After playing-strength evaluation, an operator records one
winner and may replay *that recipe* to 256 steps from the original parent; a
candidate checkpoint is never used as another candidate's initializer.

This tool deliberately does not choose a production champion. It is a
diagnostic campaign runner around :mod:`tools.a1_one_dose_train`; production
lineage remains owned by the sealed post-promotion handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "a1-b200-lr-dose-campaign-v1"
SELECTION_SCHEMA = "a1-b200-lr-dose-selection-v1"
WORLD_SIZE = 8
GLOBAL_BATCH_SIZE = 4096
SHORT_STEPS = 128
LONG_STEPS = 256
ARMS: dict[str, dict[str, int | float]] = {
    "A": {"lr": 3e-5, "lr_warmup_steps": 100},
    "B": {"lr": 3e-5, "lr_warmup_steps": 16},
    "C": {"lr": 6e-5, "lr_warmup_steps": 16},
    "D": {"lr": 1.2e-4, "lr_warmup_steps": 16},
}


class CampaignError(RuntimeError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _value_sha256(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _regular_file(path: Path, *, where: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CampaignError(f"{where} must be a regular file: {resolved}")
    return resolved


def _python_executable(path: Path) -> Path:
    """Preserve a virtualenv's lexical executable while authenticating target."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"cannot resolve Python executable {lexical}: {error}") from error
    if not target.is_file() or not os.access(lexical, os.X_OK):
        raise CampaignError(f"Python executable is not runnable: {lexical}")
    return lexical


def _normalize_sha256(value: str, *, where: str) -> str:
    text = str(value).strip().lower()
    if not text.startswith("sha256:"):
        text = f"sha256:{text}"
    if len(text) != 71 or any(ch not in "0123456789abcdef" for ch in text[7:]):
        raise CampaignError(f"{where} must be one SHA-256 digest")
    return text


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load_bound_json(path: Path, *, schema: str) -> dict[str, Any]:
    resolved = _regular_file(path, where=schema)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot load {schema}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        raise CampaignError(f"{resolved} is not {schema}")
    stated = payload.get("campaign_sha256" if schema == SCHEMA else "selection_sha256")
    unsigned = dict(payload)
    unsigned.pop("campaign_sha256" if schema == SCHEMA else "selection_sha256", None)
    if stated != _value_sha256(unsigned):
        raise CampaignError(f"{schema} digest drift")
    return payload


def _arm_overrides(
    arm: str,
    *,
    max_steps: int,
    policy_aux_active_batch_size: int,
) -> dict[str, object]:
    if arm not in ARMS:
        raise CampaignError(f"unknown campaign arm {arm!r}")
    recipe: dict[str, object] = {
        "epochs": 1,
        "max_steps": int(max_steps),
        "lr": float(ARMS[arm]["lr"]),
        "lr_warmup_steps": int(ARMS[arm]["lr_warmup_steps"]),
        "lr_schedule": "flat",
    }
    if policy_aux_active_batch_size > 0:
        recipe["policy_aux_active_batch_size"] = int(
            policy_aux_active_batch_size
        )
    return recipe


def _one_dose_invocation(
    campaign: Mapping[str, Any],
    *,
    arm: str,
    max_steps: int,
    suffix: str,
) -> list[str]:
    shared = campaign["inputs"]
    output = Path(campaign["output_root"]) / suffix
    overrides = _arm_overrides(
        arm,
        max_steps=max_steps,
        policy_aux_active_batch_size=int(
            campaign["policy_active_dose"]["policy_aux_active_batch_size"]
        ),
    )
    ablation_id = f"lr-dose-{arm.lower()}-steps{max_steps}"
    return [
        str(shared["python"]),
        str(shared["one_dose_trainer"]),
        "--lock",
        str(shared["lock"]),
        "--data",
        str(shared["data"]),
        "--composite-build-receipt",
        str(shared["composite_build_receipt"]),
        "--architecture-upgrade-receipt",
        str(shared["architecture_upgrade_receipt"]),
        "--checkpoint",
        str(output / "candidate.pt"),
        "--report",
        str(output / "train.report.json"),
        "--receipt",
        str(output / "one-dose.receipt.json"),
        "--python",
        str(shared["python"]),
        "--gpu",
        "0",
        "--topology",
        "b200-8gpu-ddp",
        "--ddp-canary-receipt",
        str(shared["ddp_canary_receipt"]),
        "--ablation-id",
        ablation_id,
        "--recipe-overrides-json",
        _canonical_bytes(overrides).decode("ascii"),
        "--ablation-code-tree-sha256",
        str(shared["reviewed_code_tree_sha256"]),
        "--reviewed-lock-file-sha256",
        str(shared["reviewed_lock_file_sha256"]),
        "--diagnostic-dose-curve",
    ]


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    trainer = _regular_file(
        repo / "tools" / "a1_one_dose_train.py", where="one-dose trainer"
    )
    python = _python_executable(args.python)
    lock = _regular_file(args.lock, where="sealed learner lock")
    data = _regular_file(args.data, where="post-wave composite descriptor")
    composite = _regular_file(
        args.composite_build_receipt, where="composite build receipt"
    )
    upgrade = _regular_file(
        args.architecture_upgrade_receipt, where="architecture upgrade receipt"
    )
    canary = _regular_file(args.ddp_canary_receipt, where="8xB200 DDP canary")
    expected_parent = _normalize_sha256(
        args.expected_parent_sha256, where="expected diagnostic parent"
    )
    reviewed_code = _normalize_sha256(
        args.reviewed_code_tree_sha256, where="reviewed code tree"
    )
    lock_sha = _file_sha256(lock)
    if args.reviewed_lock_file_sha256:
        expected_lock = _normalize_sha256(
            args.reviewed_lock_file_sha256, where="reviewed lock file"
        )
        if lock_sha != expected_lock:
            raise CampaignError(
                f"reviewed lock digest mismatch: expected={expected_lock} actual={lock_sha}"
            )

    target_active = int(args.target_policy_active_rows)
    observed_fraction = float(args.observed_base_policy_active_fraction)
    if target_active < 0 or not math.isfinite(observed_fraction) or not (
        0.0 <= observed_fraction <= 1.0
    ):
        raise CampaignError("policy-active dose inputs are invalid")
    base_draws = GLOBAL_BATCH_SIZE * SHORT_STEPS
    estimated_base_active = int(round(base_draws * observed_fraction))
    if target_active > 0:
        if observed_fraction <= 0.0:
            raise CampaignError(
                "--target-policy-active-rows requires an observed positive base fraction"
            )
        aux_local = max(
            0,
            math.ceil(
                (target_active - estimated_base_active)
                / (WORLD_SIZE * SHORT_STEPS)
            ),
        )
    else:
        aux_local = int(args.policy_aux_active_batch_size)
    if aux_local < 0 or aux_local > 512:
        raise CampaignError(
            "derived --policy-aux-active-batch-size must be in [0,512]"
        )
    estimated_total_active = estimated_base_active + (
        aux_local * WORLD_SIZE * SHORT_STEPS
    )

    output_root = args.output_root.expanduser().resolve(strict=False)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "purpose": "diagnostic_lr_warmup_and_same_parent_dose_curve",
        "production_promotion_eligible": False,
        "lineage_contract": {
            "expected_parent_sha256": expected_parent,
            "parent_is_operator_supplied_not_a_production_default": True,
            "every_arm_restarts_from_expected_parent": True,
            "fresh_optimizer_every_arm": True,
            "candidate_chaining_forbidden": True,
            "production_parent_owned_by_sealed_post_promotion_handoff": True,
        },
        "topology": {
            "name": "b200-8gpu-ddp",
            "world_size": WORLD_SIZE,
            "local_batch_size": 512,
            "global_batch_size": GLOBAL_BATCH_SIZE,
        },
        "short_dose": {
            "optimizer_steps": SHORT_STEPS,
            "base_row_draws": base_draws,
            "intermediate_checkpoint_steps": [64],
            "terminal_checkpoint_step": 128,
        },
        "winner_dose": {
            "optimizer_steps": LONG_STEPS,
            "replay_from_original_parent": True,
            "intermediate_checkpoint_steps": [64, 128],
            "terminal_checkpoint_step": 256,
        },
        "policy_active_dose": {
            "observed_base_policy_active_fraction": observed_fraction,
            "target_policy_active_rows": target_active or None,
            "policy_aux_active_batch_size": aux_local,
            "estimated_base_policy_active_rows": estimated_base_active,
            "estimated_total_policy_active_rows": estimated_total_active,
            "selection_uses_realized_report_not_estimate": True,
        },
        "arms": {
            arm: {
                **values,
                "max_steps": SHORT_STEPS,
                "recipe_overrides": _arm_overrides(
                    arm,
                    max_steps=SHORT_STEPS,
                    policy_aux_active_batch_size=aux_local,
                ),
                "output_subdir": f"arms/{arm}",
            }
            for arm, values in ARMS.items()
        },
        "selection_contract": {
            "primary": "paired_playing_strength",
            "validation_loss_may_select_checkpoint_within_arm_only": True,
            "all_four_one_dose_receipts_required": True,
            "all_four_evaluation_receipts_required": True,
            "only_selected_recipe_may_run_256_steps": True,
        },
        "reporting_contract": {
            "training_strata_dose_schema": "training-strata-dose-v1",
            "required_dimensions": [
                "draw_stream",
                "full_vs_fast",
                "simulation_budget",
                "decision_class",
                "legal_width",
                "phase",
                "fresh_vs_replay",
            ],
            "module_observability_schema": "module-optimizer-observability-v1",
            "module_diagnostics_cadence_batches": 16,
        },
        "inputs": {
            "python": str(python),
            "one_dose_trainer": str(trainer),
            "lock": str(lock),
            "lock_file_sha256": lock_sha,
            "data": str(data),
            "composite_build_receipt": str(composite),
            "composite_build_receipt_sha256": _file_sha256(composite),
            "architecture_upgrade_receipt": str(upgrade),
            "architecture_upgrade_receipt_sha256": _file_sha256(upgrade),
            "ddp_canary_receipt": str(canary),
            "ddp_canary_receipt_sha256": _file_sha256(canary),
            "reviewed_code_tree_sha256": reviewed_code,
            "reviewed_lock_file_sha256": lock_sha,
        },
        "output_root": str(output_root),
    }
    payload["commands"] = {
        arm: _one_dose_invocation(
            payload, arm=arm, max_steps=SHORT_STEPS, suffix=f"arms/{arm}"
        )
        for arm in ARMS
    }
    payload["campaign_sha256"] = _value_sha256(payload)
    return payload


def _verify_input_bytes(campaign: Mapping[str, Any]) -> None:
    inputs = campaign["inputs"]
    for path_key, digest_key in (
        ("lock", "lock_file_sha256"),
        ("composite_build_receipt", "composite_build_receipt_sha256"),
        ("architecture_upgrade_receipt", "architecture_upgrade_receipt_sha256"),
        ("ddp_canary_receipt", "ddp_canary_receipt_sha256"),
    ):
        actual = _file_sha256(_regular_file(Path(inputs[path_key]), where=path_key))
        if actual != inputs[digest_key]:
            raise CampaignError(
                f"campaign input changed: {path_key} expected={inputs[digest_key]} actual={actual}"
            )


def _option(command: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise CampaignError(f"rendered one-dose command lost {flag}")
    return str(command[positions[0] + 1])


def _one_dose_dry_run(invocation: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        invocation,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise CampaignError(
            "one-dose dry run refused: " + (result.stderr.strip() or result.stdout.strip())
        )
    # The sealed verifier emits JSON progress records before the final, pretty-
    # printed one-dose plan.  Decode the JSON stream and bind the last document
    # that is actually a command-bearing plan instead of requiring silent
    # verifier stdout.
    decoder = json.JSONDecoder()
    cursor = 0
    plans: list[dict[str, Any]] = []
    while cursor < len(result.stdout):
        while cursor < len(result.stdout) and result.stdout[cursor].isspace():
            cursor += 1
        if cursor >= len(result.stdout):
            break
        try:
            document, cursor = decoder.raw_decode(result.stdout, cursor)
        except json.JSONDecodeError as error:
            raise CampaignError(
                "one-dose dry run emitted a malformed JSON stream"
            ) from error
        if isinstance(document, dict) and isinstance(document.get("command"), list):
            plans.append(document)
    if len(plans) != 1:
        raise CampaignError(
            f"one-dose dry run emitted {len(plans)} command-bearing JSON plans"
        )
    return plans[0]


def _verify_rendered_arm(
    campaign: Mapping[str, Any], *, arm: str, max_steps: int, plan: Mapping[str, Any]
) -> None:
    command = [str(value) for value in plan["command"]]
    initializer = _regular_file(
        Path(_option(command, "--init-checkpoint")), where="rendered parent checkpoint"
    )
    actual_initializer = _file_sha256(initializer)
    expected_parent = campaign["lineage_contract"]["expected_parent_sha256"]
    upgrade = plan.get("function_preserving_upgrade")
    if isinstance(upgrade, dict):
        source = upgrade.get("source")
        upgraded = upgrade.get("upgraded_initializer")
        if not isinstance(source, dict) or not isinstance(upgraded, dict):
            raise CampaignError("one-dose plan has malformed upgrade lineage")
        actual_parent = source.get("sha256")
        expected_initializer = upgraded.get("sha256")
    else:
        actual_parent = actual_initializer
        expected_initializer = expected_parent
    if actual_parent != expected_parent:
        raise CampaignError(
            f"diagnostic source parent mismatch: expected={expected_parent} actual={actual_parent}"
        )
    if actual_initializer != expected_initializer:
        raise CampaignError(
            "rendered initializer bytes differ from the authenticated "
            f"function-preserving upgrade: expected={expected_initializer} "
            f"actual={actual_initializer} path={initializer}"
        )
    if (
        "--no-resume-optimizer" not in command
        or command.count("torch.distributed.run") != 1
    ):
        raise CampaignError("rendered arm lost fresh-Adam 8xDDP semantics")
    # The nproc flag uses --flag=value syntax, unlike ordinary train_bc
    # options, so check it without the separate-value parser.
    if not any(token == "--nproc_per_node=8" for token in command):
        raise CampaignError("rendered arm is not an 8-rank DDP command")
    if int(_option(command, "--max-steps")) != int(max_steps):
        raise CampaignError("rendered arm max_steps drift")
    if float(_option(command, "--lr")) != float(ARMS[arm]["lr"]):
        raise CampaignError("rendered arm LR drift")
    if int(_option(command, "--lr-warmup-steps")) != int(
        ARMS[arm]["lr_warmup_steps"]
    ):
        raise CampaignError("rendered arm warmup drift")
    expected_checkpoints = "64" if max_steps == SHORT_STEPS else "64,128"
    if _option(command, "--checkpoint-steps") != expected_checkpoints:
        raise CampaignError("rendered within-trajectory checkpoint curve drift")
    if _option(command, "--train-diagnostics-every-batches") != "16":
        raise CampaignError("rendered module-attribution cadence drift")


def _verify_training_report(
    campaign: Mapping[str, Any],
    *,
    arm: str,
    max_steps: int,
    one_dose_plan: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = _regular_file(
        Path(str(one_dose_plan["report"])), where="completed training report"
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read completed training report: {error}") from error
    if not isinstance(report, dict) or int(report.get("steps_completed", -1)) != max_steps:
        raise CampaignError("completed training report lost its exact optimizer dose")
    if report.get("optimizer_restored") is not False:
        raise CampaignError("completed campaign arm did not use fresh Adam")
    strata = report.get("training_strata_dose")
    required_dimensions = set(
        campaign["reporting_contract"]["required_dimensions"]
    )
    if (
        not isinstance(strata, dict)
        or strata.get("schema_version") != "training-strata-dose-v1"
        or not isinstance(strata.get("dimensions"), dict)
        or set(strata["dimensions"]) != required_dimensions
    ):
        raise CampaignError("training report lost required realized-dose strata")
    expected_base = GLOBAL_BATCH_SIZE * max_steps
    policy_active = int(strata.get("policy_active_row_draws", -1))
    if (
        int(strata.get("base_row_draws", -1)) != expected_base
        or int(strata.get("policy_aux_row_draws", -1))
        != int(report.get("policy_aux_active_rows", -2))
        or policy_active != int(report.get("policy_total_active_rows", -2))
        or policy_active <= 0
    ):
        raise CampaignError("training report realized policy-active dose arithmetic drift")
    module_report = report.get("module_optimizer_observability")
    expected_observations = max_steps // 16
    if (
        not isinstance(module_report, dict)
        or module_report.get("schema_version")
        != "module-optimizer-observability-v1"
        or int(module_report.get("observed_steps", -1)) != expected_observations
        or not isinstance(module_report.get("modules"), dict)
        or not module_report["modules"]
    ):
        raise CampaignError("training report lost module gradient/update attribution")
    return {
        "arm": arm,
        "max_steps": max_steps,
        "report": str(report_path),
        "report_file_sha256": _file_sha256(report_path),
        "base_row_draws": expected_base,
        "policy_active_row_draws": policy_active,
        "policy_active_fraction": float(strata["policy_active_fraction"]),
        "policy_aux_row_draws": int(strata["policy_aux_row_draws"]),
        "module_observed_steps": expected_observations,
    }


def _execute_invocation(
    campaign: Mapping[str, Any],
    *,
    arm: str,
    max_steps: int,
    invocation: list[str],
    go: bool,
) -> dict[str, Any]:
    _verify_input_bytes(campaign)
    plan = _one_dose_dry_run(invocation)
    _verify_rendered_arm(campaign, arm=arm, max_steps=max_steps, plan=plan)
    if not go:
        return {"mode": "dry-run", "one_dose_plan": plan}
    result = subprocess.run([*invocation, "--go"], check=False)
    if result.returncode != 0:
        raise CampaignError(f"one-dose arm {arm} exited {result.returncode}")
    report_summary = _verify_training_report(
        campaign,
        arm=arm,
        max_steps=max_steps,
        one_dose_plan=plan,
    )
    return {"mode": "go", "returncode": 0, **report_summary}


def _parse_bindings(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        arm, separator, path = raw.partition("=")
        if separator != "=" or arm not in ARMS or arm in result:
            raise CampaignError(f"{label} must contain each unique ARM=PATH binding")
        result[arm] = _regular_file(Path(path), where=f"{label} {arm}")
    if set(result) != set(ARMS):
        raise CampaignError(f"{label} requires A, B, C, and D")
    return result


def _select(args: argparse.Namespace) -> dict[str, Any]:
    campaign_path = _regular_file(args.campaign, where="campaign plan")
    campaign = _load_bound_json(campaign_path, schema=SCHEMA)
    if args.winner not in ARMS:
        raise CampaignError("winner must be A, B, C, or D")
    receipts = _parse_bindings(args.arm_receipt, label="arm receipt")
    evaluations = _parse_bindings(args.evaluation, label="evaluation")
    receipt_records: dict[str, Any] = {}
    for arm, path in receipts.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("returncode") != 0:
            raise CampaignError(f"arm {arm} has no successful one-dose receipt")
        expected = str(Path(campaign["output_root"]) / "arms" / arm / "one-dose.receipt.json")
        if str(path) != expected:
            raise CampaignError(f"arm {arm} receipt is outside its planned namespace")
        receipt_records[arm] = {
            "path": str(path),
            "file_sha256": _file_sha256(path),
            "checkpoint_sha256": (payload.get("outputs") or {}).get(
                "checkpoint_sha256"
            ),
        }
    selection: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "campaign": str(campaign_path),
        "campaign_file_sha256": _file_sha256(campaign_path),
        "campaign_sha256": campaign["campaign_sha256"],
        "winner": args.winner,
        "selection_basis": "operator_declared_after_paired_playing_strength",
        "winner_recipe": _arm_overrides(
            args.winner,
            max_steps=LONG_STEPS,
            policy_aux_active_batch_size=int(
                campaign["policy_active_dose"]["policy_aux_active_batch_size"]
            ),
        ),
        "winner_replays_from_original_parent": True,
        "candidate_chaining": False,
        "arm_receipts": receipt_records,
        "evaluation_receipts": {
            arm: {"path": str(path), "file_sha256": _file_sha256(path)}
            for arm, path in evaluations.items()
        },
        "long_dose_output_subdir": f"winner/{args.winner}-steps256",
    }
    selection["selection_sha256"] = _value_sha256(selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="seal the four-arm campaign")
    plan.add_argument("--lock", required=True, type=Path)
    plan.add_argument("--data", required=True, type=Path)
    plan.add_argument("--composite-build-receipt", required=True, type=Path)
    plan.add_argument("--architecture-upgrade-receipt", required=True, type=Path)
    plan.add_argument("--ddp-canary-receipt", required=True, type=Path)
    plan.add_argument("--expected-parent-sha256", required=True)
    plan.add_argument("--reviewed-code-tree-sha256", required=True)
    plan.add_argument("--reviewed-lock-file-sha256", default="")
    plan.add_argument("--python", required=True, type=Path)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)
    plan.add_argument("--observed-base-policy-active-fraction", type=float, default=0.0)
    plan.add_argument("--target-policy-active-rows", type=int, default=0)
    plan.add_argument("--policy-aux-active-batch-size", type=int, default=0)

    run = sub.add_parser("run-arm", help="dry-run or execute one 128-step arm")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--arm", required=True, choices=tuple(ARMS))
    run.add_argument("--go", action="store_true")

    select = sub.add_parser("select", help="bind evaluated winner after all arms")
    select.add_argument("--campaign", required=True, type=Path)
    select.add_argument("--winner", required=True, choices=tuple(ARMS))
    select.add_argument("--arm-receipt", action="append", default=[], metavar="ARM=PATH")
    select.add_argument("--evaluation", action="append", default=[], metavar="ARM=PATH")
    select.add_argument("--write", required=True, type=Path)

    winner = sub.add_parser(
        "run-winner-256", help="replay only the selected recipe to 256 from parent"
    )
    winner.add_argument("--campaign", required=True, type=Path)
    winner.add_argument("--selection", required=True, type=Path)
    winner.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            campaign = _plan(args)
            _write_json(args.write, campaign)
            print(json.dumps(campaign, indent=2, sort_keys=True))
            return 0
        if args.command == "run-arm":
            campaign = _load_bound_json(args.campaign, schema=SCHEMA)
            invocation = _one_dose_invocation(
                campaign,
                arm=args.arm,
                max_steps=SHORT_STEPS,
                suffix=f"arms/{args.arm}",
            )
            result = _execute_invocation(
                campaign,
                arm=args.arm,
                max_steps=SHORT_STEPS,
                invocation=invocation,
                go=bool(args.go),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "select":
            selection = _select(args)
            _write_json(args.write, selection)
            print(json.dumps(selection, indent=2, sort_keys=True))
            return 0
        if args.command == "run-winner-256":
            campaign = _load_bound_json(args.campaign, schema=SCHEMA)
            selection = _load_bound_json(args.selection, schema=SELECTION_SCHEMA)
            if (
                selection.get("campaign_sha256") != campaign["campaign_sha256"]
                or selection.get("campaign_file_sha256")
                != _file_sha256(_regular_file(args.campaign, where="campaign plan"))
            ):
                raise CampaignError("winner selection binds a different campaign")
            arm = str(selection["winner"])
            suffix = str(selection["long_dose_output_subdir"])
            invocation = _one_dose_invocation(
                campaign, arm=arm, max_steps=LONG_STEPS, suffix=suffix
            )
            result = _execute_invocation(
                campaign,
                arm=arm,
                max_steps=LONG_STEPS,
                invocation=invocation,
                go=bool(args.go),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        raise CampaignError(f"unknown command {args.command}")
    except (CampaignError, OSError, json.JSONDecodeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
