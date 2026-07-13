#!/usr/bin/env python3
"""Seal and run the one-axis pure-search-target TEMP learner diagnostic.

The source is a verified production TEMP replication manifest.  This arm keeps
its exact f7 initializer, composite descriptor, per-component temperatures,
4.19M-row dose, fresh optimizer, RNG, masking, validation, and every auxiliary
objective.  The sole optimization delta is ``soft_target_weight: 0.9 -> 1.0``;
therefore the played-action hard CE contributes zero policy mass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_production_temperature_replication as temperature  # noqa: E402


SCHEMA = "a1-pure-soft-calibrated-arm-v1"
CLAIM_SCHEMA = "a1-pure-soft-arm-execution-claim-v1"
SUBMISSION_SCHEMA = "a1-pure-soft-arm-execution-receipt-v1"
COMPLETION_SCHEMA = "a1-pure-soft-arm-completion-v1"
SOURCE_WEIGHT = "0.9"
TREATMENT_WEIGHT = "1.0"
SOURCE_FILES = (
    "tools/a1_pure_soft_arm.py",
    "tools/a1_production_temperature_replication.py",
    "tools/a1_production_l1_rerun.py",
    "tools/train_bc.py",
    "tools/mixed_memmap_corpus.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
)
OUTPUT_NAMES = (
    "candidate.pt",
    "candidate.pt.optimizer.pt",
    "candidate.pt.training-progress.json",
    "train.report.json",
    "diagnostic-execution.claim.json",
    "diagnostic-execution.receipt.json",
    "diagnostic-completion.receipt.json",
    "stdout.log",
    "stderr.log",
)
MANIFEST_FIELDS = {
    "schema_version",
    "diagnostic_only",
    "promotion_eligible",
    "launch_authorized",
    "source_temperature_manifest",
    "source_temperature_manifest_sha256",
    "f7_parent",
    "source_descriptor",
    "validation_sentinel",
    "component_bindings",
    "stored_policy_component_temperatures",
    "event_history_training_contract",
    "selected_dose",
    "runtime_python",
    "execution_preconditions",
    "repo_binding",
    "only_declared_optimization_delta",
    "matched_contract",
    "command",
    "command_sha256",
    "output_root",
    "manifest_sha256",
}


class PureSoftArmError(RuntimeError):
    """The request is not the exact one-axis pure-soft diagnostic."""


def _fail(message: str) -> None:
    raise PureSoftArmError(message)


def _is_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _set_unique(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(command) if item == flag]
    equals = [
        index for index, item in enumerate(command) if item.startswith(flag + "=")
    ]
    if len(positions) + len(equals) != 1:
        _fail(f"source TEMP command must contain exactly one {flag}")
    if equals:
        command[equals[0]] = f"{flag}={value}"
        return
    index = positions[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        _fail(f"source TEMP command has no value for {flag}")
    command[index + 1] = value


def _trainer_index(command: Sequence[str]) -> int:
    positions = [
        index for index, item in enumerate(command) if Path(item).name == "train_bc.py"
    ]
    if len(positions) != 1:
        _fail("source TEMP command must name exactly one train_bc.py")
    return positions[0]


def _derive_command(
    source: Sequence[str], *, trainer: Path, output_root: Path
) -> list[str]:
    command = list(source)
    command[_trainer_index(command)] = str(trainer.resolve(strict=True))
    try:
        source_weight = temperature.base._option(  # noqa: SLF001
            command, "--soft-target-weight"
        )
    except temperature.base.L1Error as error:
        raise PureSoftArmError(str(error)) from error
    if source_weight != SOURCE_WEIGHT:
        _fail("source TEMP command is not the winning 0.9-soft recipe")
    _set_unique(command, "--soft-target-weight", TREATMENT_WEIGHT)
    _set_unique(command, "--checkpoint", str(output_root / "candidate.pt"))
    _set_unique(command, "--report", str(output_root / "train.report.json"))
    return command


def _verify_exact_derivation(
    source: Sequence[str],
    treatment: Sequence[str],
    *,
    trainer: Path,
    output_root: Path,
) -> None:
    expected = _derive_command(source, trainer=trainer, output_root=output_root)
    if list(treatment) != expected:
        _fail("pure-soft command is not the exact one-axis TEMP derivation")


def _repo_binding(repo: Path) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    try:
        commit = temperature.base._assert_bound_checkout(repo)  # noqa: SLF001
        for relative in SOURCE_FILES:
            subprocess.run(
                ("git", "ls-files", "--error-unmatch", relative),
                cwd=repo,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError, temperature.base.L1Error) as error:
        raise PureSoftArmError(
            "pure-soft sources must be clean tracked public-main bytes"
        ) from error
    return {
        "repository_root": str(repo),
        "public_main_commit": commit,
        "files": {
            relative: temperature.base._ref(repo / relative)  # noqa: SLF001
            for relative in SOURCE_FILES
        },
    }


def _fresh_root(output_root: Path, source_root: Path) -> Path:
    root = output_root.expanduser().resolve(strict=False)
    source = source_root.expanduser().resolve(strict=False)
    if _is_nested(root, source):
        _fail("pure-soft output root must be independent of TEMP")
    existing = [str(root / name) for name in OUTPUT_NAMES if (root / name).exists()]
    if existing:
        _fail(f"pure-soft output identity is not fresh: {existing}")
    return root


def prepare(
    *,
    source_temperature_manifest: Path,
    repo: Path,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    try:
        source = temperature.verify(source_temperature_manifest)
    except temperature.TemperatureReplicationError as error:
        raise PureSoftArmError(str(error)) from error
    source_manifest = source["manifest"]
    temperature._validate_recipe(  # noqa: SLF001
        source["command"],
        descriptor=source_manifest["source_descriptor"]["path"],
        sentinel=source_manifest["validation_sentinel"]["path"],
        f7=source_manifest["f7_parent"]["path"],
    )
    binding = _repo_binding(repo)
    root = _fresh_root(output_root, source["output_root"])
    manifest_path = manifest_path.expanduser().resolve(strict=False)
    if manifest_path.exists():
        _fail(f"refusing existing pure-soft manifest: {manifest_path}")
    command = _derive_command(
        source["command"],
        trainer=Path(binding["files"]["tools/train_bc.py"]["path"]),
        output_root=root,
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": True,
        "source_temperature_manifest": source["manifest_ref"],
        "source_temperature_manifest_sha256": source_manifest["manifest_sha256"],
        "f7_parent": source_manifest["f7_parent"],
        "source_descriptor": source_manifest["source_descriptor"],
        "validation_sentinel": source_manifest["validation_sentinel"],
        "component_bindings": source_manifest["component_bindings"],
        "stored_policy_component_temperatures": source_manifest[
            "stored_policy_component_temperatures"
        ],
        "event_history_training_contract": source_manifest[
            "event_history_training_contract"
        ],
        "selected_dose": source_manifest["selected_dose"],
        "runtime_python": source_manifest["runtime_python"],
        "execution_preconditions": source_manifest["execution_preconditions"],
        "repo_binding": binding,
        "only_declared_optimization_delta": {
            "soft_target_weight": {"source": 0.9, "treatment": 1.0},
            "played_action_hard_ce_weight": {"source": 0.1, "treatment": 0.0},
        },
        "matched_contract": {
            "exact_temperature_data_and_component_temperatures": True,
            "exact_f7_initializer": True,
            "fresh_adam": True,
            "candidate_chaining": False,
            "global_samples": 4_194_304,
            "world_size": 8,
            "per_rank_batch_size": 512,
            "training_rng_rank_offset": True,
            "mask_hidden_info": True,
            "all_other_objectives_exact": True,
        },
        "command": command,
        "command_sha256": temperature.base._digest(command),  # noqa: SLF001
        "output_root": str(root),
    }
    manifest["manifest_sha256"] = temperature.base._digest(manifest)  # noqa: SLF001
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temperature.base._write_exclusive(manifest_path, manifest, mode=0o444)  # noqa: SLF001
    except temperature.base.L1Error as error:
        raise PureSoftArmError(str(error)) from error
    return manifest


def verify(manifest_path: Path, *, require_fresh_output: bool = True) -> dict[str, Any]:
    try:
        manifest_ref = temperature.base._ref(manifest_path)  # noqa: SLF001
        manifest = temperature.base._load(Path(manifest_ref["path"]))  # noqa: SLF001
    except temperature.base.L1Error as error:
        raise PureSoftArmError(str(error)) from error
    unhashed = dict(manifest)
    stated = unhashed.pop("manifest_sha256", None)
    if stated != temperature.base._digest(unhashed):  # noqa: SLF001
        _fail("pure-soft manifest semantic digest drift")
    if set(manifest) != MANIFEST_FIELDS:
        _fail("pure-soft manifest fields differ from schema")
    if not (
        manifest.get("schema_version") == SCHEMA
        and manifest.get("diagnostic_only") is True
        and manifest.get("promotion_eligible") is False
        and manifest.get("launch_authorized") is True
    ):
        _fail("pure-soft authorization boundary drift")
    source_ref = manifest.get("source_temperature_manifest")
    temperature.base._verify_ref(source_ref, "source_temperature_manifest")  # noqa: SLF001
    source = temperature.verify(Path(source_ref["path"]))
    source_manifest = source["manifest"]
    if (
        manifest.get("source_temperature_manifest_sha256")
        != source_manifest["manifest_sha256"]
    ):
        _fail("source TEMP semantic identity drift")
    binding = manifest.get("repo_binding")
    if not isinstance(binding, Mapping) or set(binding) != {
        "repository_root",
        "public_main_commit",
        "files",
    }:
        _fail("pure-soft repository binding is malformed")
    repo = Path(str(binding["repository_root"])).resolve(strict=True)
    try:
        temperature.base._assert_bound_checkout(  # noqa: SLF001
            repo, str(binding["public_main_commit"])
        )
    except temperature.base.L1Error as error:
        raise PureSoftArmError(str(error)) from error
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILES):
        _fail("pure-soft source binding is incomplete")
    for relative, ref in files.items():
        path = temperature.base._verify_ref(ref, f"source.{relative}")  # noqa: SLF001
        if path != (repo / relative).resolve(strict=True):
            _fail(f"pure-soft source path escaped checkout: {relative}")
    exact_copies = (
        "f7_parent",
        "source_descriptor",
        "validation_sentinel",
        "component_bindings",
        "stored_policy_component_temperatures",
        "event_history_training_contract",
        "selected_dose",
        "runtime_python",
        "execution_preconditions",
    )
    drift = [
        key for key in exact_copies if manifest.get(key) != source_manifest.get(key)
    ]
    if drift:
        _fail(f"pure-soft arm changed TEMP causal contract fields: {drift}")
    if manifest.get("only_declared_optimization_delta") != {
        "soft_target_weight": {"source": 0.9, "treatment": 1.0},
        "played_action_hard_ce_weight": {"source": 0.1, "treatment": 0.0},
    }:
        _fail("pure-soft optimization-delta declaration drift")
    if manifest.get("matched_contract") != {
        "exact_temperature_data_and_component_temperatures": True,
        "exact_f7_initializer": True,
        "fresh_adam": True,
        "candidate_chaining": False,
        "global_samples": 4_194_304,
        "world_size": 8,
        "per_rank_batch_size": 512,
        "training_rng_rank_offset": True,
        "mask_hidden_info": True,
        "all_other_objectives_exact": True,
    }:
        _fail("pure-soft matched-contract declaration drift")
    root = Path(str(manifest.get("output_root", ""))).resolve(strict=False)
    if _is_nested(root, source["output_root"]):
        _fail("pure-soft output root is not independent of TEMP")
    command = manifest.get("command")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        _fail("pure-soft command is malformed")
    if manifest.get("command_sha256") != temperature.base._digest(command):  # noqa: SLF001
        _fail("pure-soft command digest drift")
    _verify_exact_derivation(
        source["command"],
        command,
        trainer=Path(files["tools/train_bc.py"]["path"]),
        output_root=root,
    )
    if (
        temperature.base._option(command, "--init-checkpoint")
        != manifest["f7_parent"]["path"]
    ):  # noqa: SLF001
        _fail("pure-soft command is not independently initialized from exact f7")
    if require_fresh_output:
        _fresh_root(root, source["output_root"])
    return {
        "manifest": manifest,
        "manifest_ref": manifest_ref,
        "source": source,
        "repo": repo,
        "command": command,
        "output_root": root,
    }


def execute(
    manifest_path: Path,
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    idle_probe: Callable[[], list[str]] = temperature.base._idle_b200s,  # noqa: SLF001
) -> dict[str, Any]:
    if temperature.base.SAFE_UNIT.fullmatch(unit) is None:
        _fail("systemd unit name is invalid")
    verified = verify(manifest_path)
    conflicts = idle_probe()
    if conflicts:
        _fail(f"B200 compute is not idle: {conflicts}")
    root = verified["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema_version": CLAIM_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "unit": unit,
    }
    claim["claim_sha256"] = temperature.base._digest(claim)  # noqa: SLF001
    claim_path = root / "diagnostic-execution.claim.json"
    temperature.base._write_exclusive(claim_path, claim)  # noqa: SLF001
    systemd_command = temperature._systemd_command(  # noqa: SLF001
        unit=unit,
        repo=verified["repo"],
        root=root,
        command=verified["command"],
    )
    try:
        result = runner(systemd_command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PureSoftArmError(
            f"systemd submission failed after one-shot claim: {error}"
        ) from error
    receipt = {
        "schema_version": SUBMISSION_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "claim": {
            "path": str(claim_path),
            "sha256": temperature.base._file_sha(claim_path),  # noqa: SLF001
        },
        "unit": unit,
        "command_sha256": verified["manifest"]["command_sha256"],
        "systemd_command_sha256": temperature.base._digest(systemd_command),  # noqa: SLF001
        "systemd_stdout": result.stdout.strip(),
    }
    receipt["receipt_sha256"] = temperature.base._digest(receipt)  # noqa: SLF001
    temperature.base._write_exclusive(  # noqa: SLF001
        root / "diagnostic-execution.receipt.json", receipt
    )
    return receipt


def _pure_soft_report_drift(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = dict(temperature.SEALED_REPORT_RECIPE)
    expected["soft_target_weight"] = 1.0
    return {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }


def finalize(manifest_path: Path, *, unit: str) -> dict[str, Any]:
    if temperature.base.SAFE_UNIT.fullmatch(unit) is None:
        _fail("systemd unit name is invalid")
    verified = verify(manifest_path, require_fresh_output=False)
    root = verified["output_root"]
    receipt_path = root / "diagnostic-execution.receipt.json"
    receipt = temperature.base._load(receipt_path)  # noqa: SLF001
    unhashed = dict(receipt)
    digest = unhashed.pop("receipt_sha256", None)
    if not (
        set(receipt)
        == {
            "schema_version",
            "diagnostic_only",
            "promotion_eligible",
            "created_at_unix_ns",
            "manifest",
            "claim",
            "unit",
            "command_sha256",
            "systemd_command_sha256",
            "systemd_stdout",
            "receipt_sha256",
        }
        and receipt.get("schema_version") == SUBMISSION_SCHEMA
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and receipt.get("manifest") == verified["manifest_ref"]
        and receipt.get("unit") == unit
        and receipt.get("command_sha256") == verified["manifest"]["command_sha256"]
        and digest == temperature.base._digest(unhashed)  # noqa: SLF001
    ):
        _fail("pure-soft submission receipt drift")
    try:
        claim_path = temperature.base._verify_ref(  # noqa: SLF001
            receipt.get("claim"), "pure-soft claim"
        )
        claim = temperature.base._load(claim_path)  # noqa: SLF001
    except temperature.base.L1Error as error:
        raise PureSoftArmError(str(error)) from error
    claim_unhashed = dict(claim)
    claim_digest = claim_unhashed.pop("claim_sha256", None)
    if not (
        claim_path == root / "diagnostic-execution.claim.json"
        and set(claim)
        == {
            "schema_version",
            "diagnostic_only",
            "promotion_eligible",
            "created_at_unix_ns",
            "manifest",
            "unit",
            "claim_sha256",
        }
        and claim.get("schema_version") == CLAIM_SCHEMA
        and claim.get("diagnostic_only") is True
        and claim.get("promotion_eligible") is False
        and claim.get("manifest") == verified["manifest_ref"]
        and claim.get("unit") == unit
        and claim_digest == temperature.base._digest(claim_unhashed)  # noqa: SLF001
    ):
        _fail("pure-soft one-shot claim drift")
    expected_systemd = temperature._systemd_command(  # noqa: SLF001
        unit=unit,
        repo=verified["repo"],
        root=root,
        command=verified["command"],
    )
    if receipt.get("systemd_command_sha256") != temperature.base._digest(
        expected_systemd
    ):  # noqa: SLF001
        _fail("pure-soft submission did not bind the sealed systemd command")
    try:
        state = subprocess.check_output(
            ("systemctl", "show", unit, "--property=ActiveState,Result,ExecMainStatus"),
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PureSoftArmError(f"cannot read systemd state: {error}") from error
    fields = dict(row.split("=", 1) for row in state.splitlines() if "=" in row)
    if fields != {
        "ActiveState": "inactive",
        "Result": "success",
        "ExecMainStatus": "0",
    }:
        _fail(f"pure-soft learner is not complete: {fields}")
    checkpoint = temperature.base._ref(root / "candidate.pt")  # noqa: SLF001
    if checkpoint["sha256"] == verified["manifest"]["f7_parent"]["sha256"]:
        _fail("pure-soft output reused the f7 initializer")
    source_candidate = verified["source"]["output_root"] / "candidate.pt"
    if source_candidate.is_file() and checkpoint[
        "sha256"
    ] == temperature.base._file_sha(  # noqa: SLF001
        source_candidate
    ):
        _fail("pure-soft output reused the TEMP control checkpoint")
    report_ref = temperature.base._ref(root / "train.report.json")  # noqa: SLF001
    report = temperature.base._load(Path(report_ref["path"]))  # noqa: SLF001
    drift = _pure_soft_report_drift(report)
    if drift:
        _fail(f"completed report differs from one-axis pure-soft recipe: {drift}")
    if not temperature._authenticated_objective_validation(report):  # noqa: SLF001
        _fail("completed report lacks authenticated objective-matched validation")
    # Reuse the production TEMP verifier for every path, descriptor, component,
    # event-history, and runtime binding.  Substitute only the declared axis in
    # an in-memory control view so the parent verifier remains the single source
    # of truth for all matched fields.
    parent_view = dict(report)
    parent_view["soft_target_weight"] = 0.9
    try:
        temperature._verify_completed_report(  # noqa: SLF001
            parent_view, verified=verified, checkpoint=checkpoint
        )
    except temperature.TemperatureReplicationError as error:
        raise PureSoftArmError(str(error)) from error
    source_manifest = verified["manifest"]
    try:
        init_path = Path(str(report["init_checkpoint"])).resolve(strict=True)
        data_path = Path(str(report["data"])).resolve(strict=True)
        checkpoint_path = Path(str(report["checkpoint"])).resolve(strict=True)
        sentinel_path = Path(
            str(report["input_validation_game_sentinel_manifest"])
        ).resolve(strict=True)
    except OSError as error:
        raise PureSoftArmError(
            f"completed report path cannot be resolved: {error}"
        ) from error
    if not (
        init_path == Path(source_manifest["f7_parent"]["path"])
        and data_path == Path(source_manifest["source_descriptor"]["path"])
        and sentinel_path == Path(source_manifest["validation_sentinel"]["path"])
        and checkpoint_path == root / "candidate.pt"
        and checkpoint["path"] == str(checkpoint_path)
    ):
        _fail("completed report input/output paths differ from sealed manifest")
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "submission": {
            "path": str(receipt_path),
            "sha256": temperature.base._file_sha(receipt_path),  # noqa: SLF001
        },
        "checkpoint": checkpoint,
        "report": report_ref,
        "unit_state": fields,
        "only_declared_optimization_delta": source_manifest[
            "only_declared_optimization_delta"
        ],
    }
    completion["receipt_sha256"] = temperature.base._digest(completion)  # noqa: SLF001
    temperature.base._write_exclusive(  # noqa: SLF001
        root / "diagnostic-completion.receipt.json", completion
    )
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-temperature-manifest", required=True, type=Path)
    prep.add_argument("--repo", default=REPO_ROOT, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--manifest", required=True, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--unit", default="a1-pure-soft-calibrated")
    run.add_argument("--go", action="store_true")
    done = sub.add_parser("finalize")
    done.add_argument("--manifest", required=True, type=Path)
    done.add_argument("--unit", default="a1-pure-soft-calibrated")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "prepare":
            payload = prepare(
                source_temperature_manifest=args.source_temperature_manifest,
                repo=args.repo,
                output_root=args.output_root,
                manifest_path=args.manifest,
            )
            result = {
                "prepared": True,
                "launched": False,
                "manifest_sha256": payload["manifest_sha256"],
            }
        elif args.action == "execute" and not args.go:
            payload = verify(args.manifest)
            result = {
                "verified": True,
                "launched": False,
                "manifest": payload["manifest_ref"],
            }
        elif args.action == "execute":
            payload = execute(args.manifest, unit=args.unit)
            result = {"submitted": True, "receipt_sha256": payload["receipt_sha256"]}
        else:
            payload = finalize(args.manifest, unit=args.unit)
            result = {"finalized": True, "receipt_sha256": payload["receipt_sha256"]}
    except (PureSoftArmError, temperature.TemperatureReplicationError) as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
