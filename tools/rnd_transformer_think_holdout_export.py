#!/usr/bin/env python3
"""Export authenticated holdout evidence for the Transformer fixed-K screen.

This is a deliberately thin, versioned specialization of the already exercised
E3 exporter engine.  It freezes the Transformer arm identities while reusing
the byte-authentication, public masking, inference, and atomic-publication code.
The specialization is process-local; it never mutates the E3 source or files.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import rnd_e3_holdout_export as _engine  # noqa: E402


EVIDENCE_SCHEMA = "catan-zero-transformer-think-holdout-evidence/v1"
EXPORT_CONTRACT_SCHEMA = "catan-zero-transformer-think-evidence-export/v1"
RUN_PROVENANCE_SCHEMA = "catan-zero-transformer-think-run-provenance/v1"
ARMS = {
    "transformer-k0": (0, 35_041_353, "smaller_k0"),
    "think-transformer-k1": (1, 40_793_673, "shared_think_40793673"),
    "think-transformer-k2": (2, 40_793_673, "shared_think_40793673"),
    "think-transformer-k4": (4, 40_793_673, "shared_think_40793673"),
}
SEEDS = (101, 103, 107)
EXPECTED_RUNS = 12
EXPECTED_HOLDOUT_GAMES = 596
EXPECTED_HOLDOUT_ROWS = 146_517
FROZEN_INCUMBENT_CHECKPOINT_SHA256 = (
    "89aa133d629e747021bc725f2ad63e0563f3b76e71f0dd563f056c6de8f77ebb"
)


ExportError = _engine.ExportError


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_registered_experiment(experiment: Mapping[str, Any], *, registered: bool) -> None:
    if (
        experiment.get("schema_version")
        != "catan-zero-transformer-think-a1-screen/v1"
    ):
        raise ValueError("unsupported Transformer-think experiment schema")
    if not registered or experiment.get("status") != "registered_ready":
        raise ValueError("Transformer-think experiment is not registered_ready")
    semantic = dict(experiment)
    declared = semantic.pop("config_sha256", None)
    if declared != _canonical_sha(semantic):
        raise ValueError("Transformer-think experiment self-hash is invalid")
    arms = experiment.get("arms")
    if not isinstance(arms, list) or len(arms) != len(ARMS):
        raise ValueError("Transformer-think experiment must contain exactly four arms")
    by_id = {item.get("arm_id"): item for item in arms if isinstance(item, Mapping)}
    if set(by_id) != set(ARMS):
        raise ValueError("Transformer-think arm identities drifted")
    for arm_id, (steps, parameters, capacity) in ARMS.items():
        arm = by_id[arm_id]
        if (
            arm.get("latent_deliberation_steps") != steps
            or arm.get("expected_parameters") != parameters
            or arm.get("capacity_class") != capacity
        ):
            raise ValueError(f"Transformer-think architecture drift for {arm_id}")
    common = experiment.get("common")
    frozen_common = {
        "hidden_size": 640,
        "state_layers": 6,
        "attention_heads": 8,
        "state_trunk": "transformer",
        "latent_deliberation_slots": 8,
        "identity_initialization_required": True,
        "frozen_incumbent_checkpoint_sha256": FROZEN_INCUMBENT_CHECKPOINT_SHA256,
    }
    if not isinstance(common, Mapping) or any(
        common.get(field) != value for field, value in frozen_common.items()
    ):
        raise ValueError("Transformer-think h640/L6 architecture drifted")
    matrix = experiment.get("run_matrix")
    if (
        not isinstance(matrix, Mapping)
        or matrix.get("seeds") != list(SEEDS)
        or matrix.get("required_run_count") != EXPECTED_RUNS
    ):
        raise ValueError("Transformer-think seeds must be exactly 101/103/107")
    if len(ARMS) * len(SEEDS) != EXPECTED_RUNS:
        raise AssertionError("frozen Transformer-think run count drifted")


def _validate_export_contract(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    experiment: Mapping[str, Any],
    experiment_path: Path,
) -> dict[str, str]:
    if contract.get("schema_version") != EXPORT_CONTRACT_SCHEMA:
        raise ExportError("unsupported Transformer-think evidence contract")
    semantic = dict(contract)
    declared = semantic.pop("config_sha256", None)
    if declared != _canonical_sha(semantic):
        raise ExportError("Transformer-think evidence contract self-hash is invalid")
    required = {
        "experiment_file_sha256": _sha256_file(experiment_path),
        "experiment_semantic_sha256": experiment.get("config_sha256"),
        "evidence_schema": EVIDENCE_SCHEMA,
        "information_regime": experiment["common"]["information_regime"],
        "public_masking_required": True,
        "holdout_games": EXPECTED_HOLDOUT_GAMES,
        "holdout_rows": EXPECTED_HOLDOUT_ROWS,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "runs": EXPECTED_RUNS,
    }
    for field, expected in required.items():
        if contract.get(field) != expected:
            raise ExportError(f"Transformer-think evidence contract {field} drifted")
    root = Path(__file__).resolve().parents[1]
    sources = {
        "exporter_source_sha256": Path(__file__).resolve(),
        "exporter_engine_source_sha256": root / "tools/rnd_e3_holdout_export.py",
        "exporter_helper_source_sha256": root / "tools/rnd_topology_holdout_export.py",
    }
    result = {
        "evidence_export_contract_sha256": _sha256_file(contract_path),
        "evidence_export_contract_semantic_sha256": declared,
    }
    for field, path in sources.items():
        expected = _engine._required_sha(contract.get(field), field=f"contract.{field}")
        if _sha256_file(path) != expected:
            raise ExportError(f"evidence contract source drift for {path.name}")
        result[field] = expected
    return result


def _validate_sources(report: Mapping[str, Any], experiment: Mapping[str, Any]) -> None:
    root = Path(__file__).resolve().parents[1]
    registered = experiment["registration"].get("executing_learner_source_sha256")
    reported = report.get("rnd_executing_learner_source_sha256")
    if not isinstance(registered, Mapping) or not isinstance(reported, Mapping):
        raise ExportError("executing learner source bindings are missing")
    if set(reported) != set(_engine._REPORT_SOURCE_FILES):
        raise ExportError("training report executing-source set is incomplete")
    for relative, digest in registered.items():
        expected = _engine._required_sha(digest, field=f"source {relative}")
        path = root / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ExportError(f"live executing source differs for {relative}")
    for relative in _engine._REPORT_SOURCE_FILES:
        live = _sha256_file(root / relative)
        if reported[relative] != registered.get(relative, live):
            raise ExportError(f"training report source differs for {relative}")


def _engine_experiment(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Supply irrelevant Transformer defaults expected by the older E3 engine."""

    normalized = dict(experiment)
    common = dict(experiment["common"])
    common.update(
        {
            "relational_block_pattern": "",
            "relational_ff_size": 0,
            "relational_bases": 4,
            "relational_action_cross_layers": 1,
        }
    )
    normalized["common"] = common
    return normalized


@contextmanager
def _specialized_engine() -> Iterator[None]:
    """Install frozen Transformer identities only for this exporter invocation."""

    replacements = {
        "EVIDENCE_SCHEMA": EVIDENCE_SCHEMA,
        "EXPORT_CONTRACT_SCHEMA": EXPORT_CONTRACT_SCHEMA,
        "ARMS": ARMS,
        "ADMISSION_SCHEMA": "catan-zero-transformer-think-a1-admission/v1",
        "_validate_contract": _validate_registered_experiment,
        "_validate_export_contract": _validate_export_contract,
        "_validate_sources": _validate_sources,
    }
    original = {name: getattr(_engine, name) for name in replacements}
    original_report_validator = _engine._validate_report
    original_policy_validator = _engine._validate_loaded_policy

    def validate_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["experiment"] = _engine_experiment(kwargs["experiment"])
        result = original_report_validator(*args, **kwargs)
        result["schema_version"] = RUN_PROVENANCE_SCHEMA
        return result

    def validate_policy(policy: Any, *, experiment: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
        original_policy_validator(
            policy, experiment=_engine_experiment(experiment), arm=arm
        )

    try:
        for name, value in replacements.items():
            setattr(_engine, name, value)
        _engine._validate_report = validate_report
        _engine._validate_loaded_policy = validate_policy
        yield
    finally:
        _engine._validate_report = original_report_validator
        _engine._validate_loaded_policy = original_policy_validator
        for name, value in original.items():
            setattr(_engine, name, value)


def export_holdout_evidence(**kwargs: Any) -> int:
    """Authenticate and export one of the twelve frozen Transformer runs."""

    with _specialized_engine():
        return _engine.export_holdout_evidence(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--evidence-contract", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = export_holdout_evidence(
            experiment_config=args.experiment_config,
            evidence_contract=args.evidence_contract,
            admission_manifest=args.admission_manifest,
            corpus_dir=args.corpus,
            training_manifest=args.training_manifest,
            validation_manifest=args.validation_manifest,
            checkpoint=args.checkpoint,
            training_report=args.training_report,
            output=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (ExportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Transformer-think holdout export failed: {exc}") from exc
    print(json.dumps({"output": str(args.output), "rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
