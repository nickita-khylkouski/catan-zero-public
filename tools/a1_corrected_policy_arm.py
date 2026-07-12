#!/usr/bin/env python3
"""Prepare, but never launch, the corrected one-dose A1 policy arm.

The builder derives a command from an authenticated prior launch receipt so
that obscure production flags cannot disappear during experiment iteration.
It then applies pure search targets, an auxiliary active-policy stream, and a
light forward-KL replay anchor. Authenticated policy and value scopes exclude
replay from supervised targets, making it genuinely anchor-only. The emitted
manifest is preparation only; this module deliberately has no launch mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import train_bc  # noqa: E402


SCHEMA = "a1-corrected-policy-arm-manifest-v1"
LINEAGE_ROLES = (
    "parent_failed_claim",
    "parent_failed_receipt",
    "retry_contract",
    "retry_receipt",
)
LINEAGE_DIGEST_FIELDS = {
    "parent_failed_claim": "state_sha256",
    "parent_failed_receipt": "receipt_sha256",
    "retry_contract": "retry_contract_sha256",
    "retry_receipt": "receipt_sha256",
}
SOURCE_FILES = (
    "tools/a1_corrected_policy_arm.py",
    "tools/a1_corrected_policy_arm_execute.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
)


class ArmError(RuntimeError):
    """The requested arm is not the exact corrected one-dose experiment."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_ref(path: Path) -> dict[str, str]:
    lexical = path.expanduser()
    if lexical.is_symlink():
        raise ArmError(f"bound artifact must be a regular non-symlink file: {lexical}")
    path = lexical.resolve(strict=True)
    if not path.is_file():
        raise ArmError(f"bound artifact must be a regular non-symlink file: {path}")
    return {"path": str(path), "sha256": _file_sha(path)}


def _load_json(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    ref = _file_ref(path)
    try:
        value = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArmError(f"cannot parse bound JSON {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("schema_version"), str):
        raise ArmError(f"bound JSON has no schema identity: {path}")
    return value, ref


def _option(command: Sequence[str], flag: str) -> str:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ArmError(f"source command must contain exactly one valued {flag}")
    value = str(command[positions[0] + 1])
    if value.startswith("--"):
        raise ArmError(f"source command has valueless {flag}")
    return value


def _set_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) > 1:
        raise ArmError(f"source command repeats {flag}")
    if positions:
        index = positions[0]
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ArmError(f"source command has valueless {flag}")
        command[index + 1] = value
    else:
        command.extend((flag, value))


def _load_source_receipt(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload, ref = _load_json(path)
    stated = payload.get("receipt_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if stated != _digest(unhashed):
        raise ArmError("source launch receipt schema or semantic digest is invalid")
    if payload.get("diagnostic_only") is not True or payload.get("promotion_eligible") is not False:
        raise ArmError("source launch receipt must be diagnostic-only/non-promotable")
    command = payload.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ArmError("source launch receipt has no replayable command")
    if payload.get("command_sha256") != _digest(command):
        raise ArmError("source launch command digest drift")
    return payload, ref


def _preflight_descriptor(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = path.expanduser().resolve(strict=True)
    try:
        verified = train_bc._preflight_memmap_composite_descriptor(path)  # noqa: SLF001
    except SystemExit as error:
        raise ArmError(f"corrected descriptor preflight failed: {error}") from error
    return verified, _file_ref(path)


def _build_corrected_descriptor(
    source_path: Path, output_path: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    source, source_ref = _preflight_descriptor(source_path)
    if source.get("schema_version") != "memmap_composite_v2" or source.get(
        "component_ids"
    ) != ["n128_current", "n256_current", "gen3_replay"]:
        raise ArmError("source descriptor must bind n128, n256, and gen3 replay in order")
    overrides = dict(source["learner_recipe_overrides"])
    overrides["policy_kl_anchor_weight"] = 0.006
    overrides["policy_kl_anchor_direction"] = "forward"
    overrides["loser_sample_weight"] = 1.0
    components = []
    ratios = (4.0 / 7.0, 8.0 / 35.0, 1.0 / 5.0)
    for source_component, ratio in zip(source["components"], ratios):
        components.append(
            {
                key: source_component[key]
                for key in (
                    "corpus_dir", "corpus_meta_sha256", "payload_inventory_sha256",
                    "validation_manifest", "validation_manifest_sha256", "component_id",
                )
            }
            | {"game_sampling_ratio": ratio}
        )
    payload = {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "components": components,
        "learner_recipe_overrides": overrides,
        "learner_recipe_overrides_sha256": _digest(overrides),
        "policy_kl_anchor_component_ids": ["gen3_replay"],
        "policy_distillation_component_ids": ["n128_current", "n256_current"],
        "value_training_component_ids": ["n128_current", "n256_current"],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != encoded:
            raise ArmError(f"prepared descriptor drift: {output_path}")
    else:
        temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, output_path)
    verified, ref = _preflight_descriptor(output_path)
    if (
        verified.get("schema_version") != "memmap_composite_v2"
        or verified.get("policy_distillation_scope_explicit") is not True
        or verified.get("policy_distillation_component_ids")
        != ["n128_current", "n256_current"]
        or verified.get("component_ids")
        != ["n128_current", "n256_current", "gen3_replay"]
        or verified.get("component_game_sampling_ratios") != list(ratios)
        or verified.get("policy_kl_anchor_component_ids") != ["gen3_replay"]
        or verified.get("value_training_component_ids")
        != ["n128_current", "n256_current"]
    ):
        raise ArmError(
            "derived descriptor must preserve exact 57.14/22.86/20 component ratios"
        )
    verified_overrides = verified.get("learner_recipe_overrides")
    if not isinstance(verified_overrides, dict) or (
        float(verified_overrides.get("policy_kl_anchor_weight", -1.0)) != 0.006
        or verified_overrides.get("policy_kl_anchor_direction") != "forward"
        or float(verified_overrides.get("loser_sample_weight", -1.0)) != 1.0
    ):
        raise ArmError("derived descriptor must bind loser=1 and exact light forward anchor")
    return verified, ref, source_ref


def _build_corrected_sentinel(
    *, source_receipt: dict[str, Any], source_descriptor: dict[str, Any],
    descriptor: Path, descriptor_meta: dict[str, Any], output_path: Path,
    python: str, repo: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    raw_source = source_receipt.get("sentinel")
    expected_source_sha = source_receipt.get("sentinel_sha256")
    if not isinstance(raw_source, str) or not isinstance(expected_source_sha, str):
        raise ArmError("source receipt does not bind a validation sentinel")
    source_payload, source_ref = _load_json(Path(raw_source))
    if source_ref["sha256"] != expected_source_sha:
        raise ArmError("source receipt validation sentinel bytes drifted")
    if (
        source_payload.get("schema_version") != "train-validation-game-sentinel-v1"
        or source_payload.get("source_composite_descriptor_file_sha256")
        != source_descriptor["descriptor_file_sha256"]
        or source_payload.get("source_composite_descriptor_fingerprint")
        != source_descriptor["descriptor_fingerprint"]
    ):
        raise ArmError("source validation sentinel is not bound to source descriptor")
    if not output_path.exists():
        command = [
            python, str(repo / "tools/derive_validation_game_sentinel.py"),
            "--composite", str(descriptor), "--out", str(output_path),
            "--target-rows", str(source_payload["target_row_count"]),
            "--selection-seed", str(source_payload["selection_seed"]),
            "--validation-fraction", "0.05", "--validation-seed", "17",
        ]
        try:
            subprocess.run(command, cwd=repo, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise ArmError(f"cannot derive corrected validation sentinel: {error}") from error
    corrected, corrected_ref = _load_json(output_path)
    if (
        corrected.get("schema_version") != "train-validation-game-sentinel-v1"
        or corrected.get("source_composite_descriptor_file_sha256")
        != descriptor_meta["descriptor_file_sha256"]
        or corrected.get("source_composite_descriptor_fingerprint")
        != descriptor_meta["descriptor_fingerprint"]
        or corrected.get("selection_seed") != source_payload.get("selection_seed")
        or corrected.get("target_row_count") != source_payload.get("target_row_count")
        or corrected.get("selected_game_seed_set_sha256")
        != source_payload.get("selected_game_seed_set_sha256")
    ):
        raise ArmError("corrected validation sentinel identity differs from source selection")
    return corrected, corrected_ref, source_ref


def _lineage(entries: Sequence[str]) -> dict[str, Any]:
    parsed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        role, separator, raw_path = entry.partition("=")
        if not separator or role not in LINEAGE_ROLES or role in parsed:
            raise ArmError("lineage entries must uniquely bind ROLE=PATH for all required roles")
        payload, ref = _load_json(Path(raw_path))
        digest_field = LINEAGE_DIGEST_FIELDS[role]
        stated = payload.get(digest_field)
        unhashed = {key: value for key, value in payload.items() if key != digest_field}
        if stated != _digest(unhashed):
            raise ArmError(f"{role} semantic digest is invalid")
        parsed[role] = {"file": ref, "schema_version": payload["schema_version"]}
    if tuple(role for role in LINEAGE_ROLES if role not in parsed):
        missing = [role for role in LINEAGE_ROLES if role not in parsed]
        raise ArmError(f"failed-retry lineage is incomplete: {missing}")
    ordered = [{"role": role, **parsed[role]} for role in LINEAGE_ROLES]
    return {"artifacts": ordered, "lineage_sha256": _digest(ordered)}


def _source_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES),
            cwd=repo,
            check=True,
        )
        for relative in SOURCE_FILES:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmError("corrected arm code must be clean, tracked canonical Git bytes") from error
    files = {relative: _file_ref(repo / relative) for relative in SOURCE_FILES}
    return {"repository_root": str(repo), "git_commit": commit, "files": files,
            "files_sha256": _digest(files)}


def _rebind_a1_metadata(command: list[str], repo: Path) -> dict[str, Any]:
    """Rebind effective recipe and the prior reviewed runtime closure."""
    required = (
        "--a1-learner-ablation-id",
        "--a1-effective-learner-recipe-json",
        "--a1-effective-learner-recipe-sha256",
        "--a1-ablation-code-binding-json",
        "--a1-ablation-code-tree-sha256",
        "--a1-reviewed-lock-file-sha256",
    )
    present = [flag in command for flag in required]
    if not any(present):
        return {
            "mode": "plain-authenticated-composite-diagnostic",
            "effective_recipe": None,
            "code_binding": _source_binding(repo),
        }
    if not all(present):
        raise ArmError("source command lacks sealed A1 effective-recipe/code metadata")
    try:
        effective = json.loads(_option(command, "--a1-effective-learner-recipe-json"))
        prior_binding = json.loads(_option(command, "--a1-ablation-code-binding-json"))
    except json.JSONDecodeError as error:
        raise ArmError(f"source A1 metadata is invalid JSON: {error}") from error
    if not isinstance(effective, dict) or not isinstance(prior_binding, dict):
        raise ArmError("source A1 metadata is not object-valued")
    recipe_updates: dict[str, Any] = {
        "batch_size": 512, "grad_accum_steps": 1, "global_batch_size": 4096,
        "world_size": 8, "max_steps": 1024, "epochs": 1,
        "loser_sample_weight": 1.0, "winner_sample_weight": 1.0,
        "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
        "policy_loss_weight": 1.0, "soft_target_source": "policy",
        "soft_target_weight": 1.0, "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5,
        "policy_aux_active_batch_size": 128,
        "policy_kl_anchor_weight": 0.006,
        "value_loss_weight": 0.25, "value_lr_mult": 0.3,
        "value_target_lambda": 1.0, "lr": 3e-5,
        "lr_warmup_steps": 100, "lr_schedule": "flat",
    }
    for key in set(recipe_updates) - {"policy_aux_active_batch_size"}:
        if key not in effective:
            raise ArmError(f"source effective recipe omits required field {key}")
    effective.update(recipe_updates)
    records = prior_binding.get("records")
    if not isinstance(records, list) or not records:
        raise ArmError("source A1 code binding has no reviewed runtime closure")
    rebound = []
    relative_paths = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("relative_path"), str):
            raise ArmError("source A1 code-binding record is malformed")
        relative = record["relative_path"]
        if relative == "tools/a1_shared_trunk_gradient_probe.py":
            raise ArmError("training runtime must not include the untracked gradient probe")
        path = (repo / relative).resolve(strict=True)
        relative_paths.append(relative)
        rebound.append(
            {"kind": str(record.get("kind", "runtime_code")),
             "relative_path": relative, "path": str(path), "sha256": _file_sha(path)}
        )
    try:
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", *relative_paths),
            cwd=repo, check=True,
        )
        for relative in relative_paths:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative), cwd=repo,
                check=True, stdout=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArmError("reviewed runtime closure is not clean and tracked") from error
    binding = {
        "schema_version": "a1-learner-ablation-code-binding-v1",
        "repository_root": str(repo),
        "records": rebound,
    }
    code_sha = _digest(binding)
    binding["code_tree_sha256"] = code_sha
    _set_option(command, "--a1-learner-ablation-id", "corrected-anchor-K3")
    _set_option(command, "--a1-effective-learner-recipe-json", _canonical(effective).decode())
    _set_option(command, "--a1-effective-learner-recipe-sha256", _digest(effective))
    _set_option(command, "--a1-ablation-code-binding-json", _canonical(binding).decode())
    _set_option(command, "--a1-ablation-code-tree-sha256", code_sha)
    return {"effective_recipe": effective, "code_binding": binding}


def _derive_command(
    source: Sequence[str], *, repo: Path, descriptor: Path, sentinel: Path,
    f7: Path, output_root: Path,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    command = list(source)
    if "torch.distributed.run" not in command or not any(
        item in {"--nproc-per-node=8", "--nproc_per_node=8"} for item in command
    ):
        raise ArmError("source command must use exactly eight torchrun ranks")
    trainer_positions = [
        index for index, item in enumerate(command) if Path(item).name == "train_bc.py"
    ]
    if len(trainer_positions) != 1:
        raise ArmError("source command must name exactly one train_bc.py")
    command[trainer_positions[0]] = str(repo / "tools/train_bc.py")
    required_flags = ("--no-resume-optimizer", "--mask-hidden-info")
    if any(flag not in command for flag in required_flags):
        raise ArmError(f"source command is missing a required safety flag: {required_flags}")
    updates = {
        "--data": str(descriptor),
        "--validation-game-sentinel-manifest": str(sentinel),
        "--init-checkpoint": str(f7),
        "--checkpoint": str(output_root / "candidate.pt"),
        "--report": str(output_root / "train.report.json"),
        "--batch-size": "512",
        "--grad-accum-steps": "1",
        "--max-steps": "1024",
        "--epochs": "1",
        "--loser-sample-weight": "1.0",
        "--winner-sample-weight": "1.0",
        "--forced-action-weight": "0.0",
        "--forced-row-value-weight": "1.0",
        "--policy-loss-weight": "1.0",
        "--soft-target-source": "policy",
        "--soft-target-weight": "1.0",
        "--soft-target-temperature": "0.7",
        "--soft-target-min-legal-coverage": "0.5",
        "--policy-aux-active-batch-size": "128",
        "--policy-kl-anchor-direction": "forward",
        "--policy-kl-anchor-weight": "0.006",
        "--value-loss-weight": "0.25",
        "--value-lr-mult": "0.3",
        "--value-target-lambda": "1.0",
        "--lr": "3e-5",
        "--lr-warmup-steps": "100",
        "--lr-schedule": "flat",
    }
    before = {flag: _option(command, flag) if flag in command else "<absent>" for flag in updates}
    for flag, value in updates.items():
        _set_option(command, flag, value)
    if "--validation-game-seed-manifest" in command:
        raise ArmError("source command mixes seed-manifest and sentinel validation controls")
    changes = {
        flag: {"source": before[flag], "corrected": value}
        for flag, value in updates.items()
        if before[flag] != value
    }
    return command, changes


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    repo = args.repo.expanduser().resolve(strict=True)
    source, source_ref = _load_source_receipt(args.source_receipt)
    source_descriptor = args.source_descriptor.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    descriptor = output_root / "corrected-anchor-memmap-composite.json"
    descriptor_meta, descriptor_ref, source_descriptor_ref = _build_corrected_descriptor(
        source_descriptor, descriptor
    )
    f7_ref = _file_ref(args.f7_checkpoint)
    if f7_ref["sha256"] != args.expected_f7_sha256:
        raise ArmError("initialization checkpoint is not the explicitly expected f7 bytes")
    source_identities = {
        "parent_checkpoint_sha256": f7_ref["sha256"],
        "descriptor_sha256": source_descriptor_ref["sha256"],
    }
    for field, expected in source_identities.items():
        if source.get(field) != expected:
            raise ArmError(f"source receipt does not reuse exact {field} identity")
    lineage = _lineage(args.failed_lineage_artifact)
    source_binding = _source_binding(repo)
    source_descriptor_meta = _preflight_descriptor(source_descriptor)[0]
    sentinel_payload, sentinel_ref, source_sentinel_ref = _build_corrected_sentinel(
        source_receipt=source,
        source_descriptor=source_descriptor_meta,
        descriptor=descriptor,
        descriptor_meta=descriptor_meta,
        output_path=output_root / "validation.sentinel.json",
        python=str(source["command"][0]),
        repo=repo,
    )
    for name in ("candidate.pt", "candidate.pt.optimizer.pt", "train.report.json"):
        if (output_root / name).exists():
            raise ArmError(f"refusing existing corrected-arm output: {output_root / name}")
    command, changes = _derive_command(
        source["command"], repo=repo, descriptor=descriptor,
        sentinel=Path(sentinel_ref["path"]), f7=Path(f7_ref["path"]),
        output_root=output_root,
    )
    a1_metadata = _rebind_a1_metadata(command, repo)
    recipe = {
        "world_size": 8, "local_batch_size": 512, "global_batch_size": 4096,
        "steps": 1024, "base_value_row_dose": 4_194_304,
        "policy_aux_active_batch_size_per_rank": 128,
        "policy_aux_active_row_dose": 1_048_576,
        "policy_distillation_component_ids": ["n128_current", "n256_current"],
        "component_game_sampling_ratios": [4.0 / 7.0, 8.0 / 35.0, 1.0 / 5.0],
        "current_supervised_base_row_dose": 3_355_443.2,
        "replay_anchor_base_row_dose": 838_860.8,
        "replay_supervised_policy": False,
        "replay_supervised_value": False,
        "replay_forward_kl_weight": 0.006,
        "soft_target_weight": 1.0, "fresh_optimizer": True,
        "independent_f7_initialization": True,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "diagnostic_execution_authorized": True,
        "launch_interface_present": "tools/a1_corrected_policy_arm_execute.py --go",
        "source_receipt": source_ref,
        "failed_retry_lineage": lineage,
        "source_descriptor": source_descriptor_ref,
        "descriptor": descriptor_ref,
        "descriptor_fingerprint": descriptor_meta["descriptor_fingerprint"],
        "source_validation_sentinel": source_sentinel_ref,
        "validation_sentinel": sentinel_ref,
        "validation_sentinel_selection_sha256": sentinel_payload[
            "selected_game_seed_set_sha256"
        ],
        "initialization": f7_ref,
        "source_binding": source_binding,
        "a1_runtime_metadata": a1_metadata,
        "recipe": recipe,
        "recipe_sha256": _digest(recipe),
        "causal_interpretation": {
            "bundled_optimization_not_f7_replication": True,
            "reason": (
                "f7 used loser=.3 and about 2.78M samples; this loser=1, 4.19M, "
                "aux128, pure-target arm is an optimization bundle, not a one-axis "
                "replication of f7"
            ),
        },
        "allowlisted_command_changes": changes,
        "command": command,
        "command_sha256": _digest(command),
    }
    manifest["manifest_sha256"] = _digest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "corrected-policy-arm.manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ArmError(f"prepared manifest drift: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    return manifest, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-descriptor", required=True, type=Path)
    parser.add_argument("--f7-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-f7-sha256", required=True)
    parser.add_argument(
        "--failed-lineage-artifact", action="append", default=[],
        help="Repeat ROLE=PATH for parent claim/receipt and retry contract/receipt.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo", default=REPO_ROOT, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    manifest, path = prepare(build_parser().parse_args(argv))
    print(json.dumps({"prepared": str(path), "launched": False,
                      "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
