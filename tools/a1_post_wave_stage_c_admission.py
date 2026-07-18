#!/usr/bin/env python3
"""Admit an audited production post-wave corpus for diagnostic Stage-C reanalysis.

The original Stage-C admission command is intentionally bound to one historical
8,192-game R&D campaign.  A later production wave already has stronger
evidence: a signed post-wave audit, an exact selected-game manifest, a
whole-game validation split, and a memmap descriptor that binds all three.
This command converts those existing artifacts into a narrow, non-promotable
reanalysis admission.  It does not copy rows, regenerate games, or authorize
stored policy targets under a different search operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


ADMISSION_SCHEMA = "a1-post-wave-stage-c-corpus-admission-v1"
SOURCE_BINDING_SCHEMA = "a1-post-wave-source-operator-binding-v1"
COHERENT_REGIME = "public_belief_single_tree_v1"
SEARCH_EVIDENCE_SCHEMA = "gumbel_root_search_evidence_v2_fp32_prior"
REQUIRED_COLUMNS = frozenset(
    {
        "game_seed",
        "decision_index",
        "phase",
        "legal_action_ids",
        "target_policy",
        "policy_weight_multiplier",
        "value_weight_multiplier",
        "event_tokens",
        "event_mask",
        "search_evidence_offsets",
        "search_completed_q_flat",
    }
)
OPERATOR_FIELDS = frozenset(
    {
        "n_full",
        "n_fast",
        "p_full",
        "n_full_wide",
        "n_full_wide_threshold",
        "wide_roots_always_full",
        "c_visit",
        "c_scale",
        "sigma_eval",
        "max_depth",
        "prior_temperature",
        "exact_budget_sh",
        "exact_budget_sh_min_n",
        "coherent_public_belief_search",
        "boundary_value_particles",
        "information_set_search",
        "determinization_particles",
        "determinization_min_simulations",
        "belief_chance_spectra",
        "information_set_target_aggregation",
        "correct_rust_chance_spectra",
        "lazy_interior_chance",
        "symmetry_averaged_eval",
        "symmetry_averaged_eval_threshold",
        "public_observation",
        "forced_root_target_mode",
        "record_automatic_transitions",
        "meaningful_public_history",
        "event_history_limit",
        "public_card_count_feature_schema",
        "temperature_clock",
        "preserve_search_evidence",
    }
)


class AdmissionError(RuntimeError):
    """The post-wave evidence is incomplete or has drifted."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise AdmissionError(f"cannot resolve {where}: {error}") from error
    if path.is_symlink() or not resolved.is_file():
        raise AdmissionError(f"{where} must be a regular non-symlink file")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdmissionError(f"cannot decode {where}: {error}") from error
    if not isinstance(payload, dict):
        raise AdmissionError(f"{where} must contain one JSON object")
    return resolved, payload


def _ref(path: Path, *, where: str) -> dict[str, Any]:
    resolved, _payload = _load(path, where=where)
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    data = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    if target.exists() or target.is_symlink():
        if target.is_file() and not target.is_symlink() and target.read_bytes() == data:
            return
        raise AdmissionError(f"immutable output already exists with drift: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_source_uniformity(
    audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = audit.get("source_provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise AdmissionError("post-wave audit has no source provenance")
    fields = (
        "producer_checkpoint_sha256",
        "search_operator_sha256",
        "effective_search_config_sha256",
        "evaluator_sha256",
        "entity_feature_adapter_version",
        "event_history_semantic",
    )
    if any(
        not isinstance(record, Mapping)
        or any(not isinstance(record.get(field), str) or not record.get(field) for field in fields)
        for record in provenance.values()
    ):
        raise AdmissionError("post-wave source provenance is incomplete")
    identities = {
        field: {str(record.get(field)) for record in provenance.values()}
        for field in fields
    }
    if any(len(values) != 1 for values in identities.values()):
        raise AdmissionError("post-wave source categories do not share one operator")
    identity = {field: next(iter(values)) for field, values in identities.items()}
    return dict(provenance), identity


def build(
    *,
    corpus_meta: Path,
    representative_manifest: Path,
    source_binding_write: Path,
    admission_write: Path,
) -> dict[str, Any]:
    meta_path, meta = _load(corpus_meta, where="post-wave corpus metadata")
    audit_ref = meta.get("a1_post_wave_audit")
    selected_ref = meta.get("selected_game_seed_manifest")
    if not isinstance(audit_ref, Mapping) or not isinstance(selected_ref, Mapping):
        raise AdmissionError("corpus metadata lacks post-wave audit bindings")

    audit_path, audit = _load(
        Path(str(audit_ref.get("path", ""))), where="post-wave audit"
    )
    selected_path, selected = _load(
        Path(str(selected_ref.get("path", ""))), where="selected-game manifest"
    )
    validation_ref = audit_ref.get("validation_holdout")
    if not isinstance(validation_ref, Mapping):
        raise AdmissionError("corpus metadata lacks validation holdout binding")
    validation_path, validation = _load(
        Path(str(validation_ref.get("path", ""))), where="validation manifest"
    )
    contract_path, contract = _load(
        Path(str(audit.get("contract_path", ""))), where="generation contract"
    )
    manifest_path, manifest = _load(
        representative_manifest, where="representative generation manifest"
    )

    audit_unsigned = dict(audit)
    audit_digest = audit_unsigned.pop("audit_sha256", None)
    selected_digest = _value_sha256(selected)
    validation_digest = _value_sha256(validation)
    selected_audit = audit.get("selected_training_games")
    validation_audit = audit.get("validation_holdout")
    selected_meta_count = selected_ref.get("selected_game_count")
    selected_count = selected.get("selected_game_count")
    validation_count = validation.get("validation_game_seed_count")
    records = selected.get("records")
    validation_seeds = validation.get("game_seeds")
    columns = meta.get("columns")
    search_evidence = meta.get("search_evidence")
    audited_regime = audit.get("target_information_regime")
    audited_regime_ok = (
        audited_regime == COHERENT_REGIME
        or (
            isinstance(audited_regime, Mapping)
            and audited_regime.get("required") == COHERENT_REGIME
            and set(audited_regime.get("counts", {})) == {COHERENT_REGIME}
        )
    )
    if (
        audit.get("schema_version") != "a1-post-wave-audit-v3"
        or audit.get("passed") is not True
        or audit.get("errors") != []
        or audit_digest != _value_sha256(audit_unsigned)
        or not isinstance(selected_audit, Mapping)
        or not isinstance(validation_audit, Mapping)
        or selected_audit.get("manifest_file_sha256") != _file_sha256(selected_path)
        or selected_audit.get("manifest_sha256") != selected_digest
        or validation_audit.get("manifest_file_sha256")
        != _file_sha256(validation_path)
        or validation_audit.get("manifest_sha256") != validation_digest
        or audit_ref.get("file_sha256") != _file_sha256(audit_path)
        or audit_ref.get("audit_sha256") != audit_digest
        or selected_ref.get("file_sha256") != _file_sha256(selected_path)
        or selected_meta_count != selected_count
        or selected_count != audit.get("total_unique_games")
        or not isinstance(records, list)
        or len(records) != selected_count
        or not isinstance(validation_seeds, list)
        or len(validation_seeds) != validation_count
        or validation_ref.get("validation_game_seed_count") != validation_count
        or validation_ref.get("validation_game_seed_set_sha256")
        != validation.get("validation_game_seed_set_sha256")
        or audit.get("rows") != meta.get("row_count")
        or not audited_regime_ok
        or not isinstance(columns, Mapping)
        or not REQUIRED_COLUMNS <= set(columns)
        or columns.get("target_information_regime", {}).get("categories")
        != [COHERENT_REGIME]
        or not isinstance(search_evidence, Mapping)
        or search_evidence.get("schema") != SEARCH_EVIDENCE_SCHEMA
        or int(search_evidence.get("active_row_count", 0)) <= 0
        or audit.get("target_activation", {}).get("passed") is not True
        or contract.get("contract_sha256") != audit.get("contract_sha256")
        or contract.get("contract_sha256") != selected.get("a1_contract_sha256")
        or contract.get("contract_sha256") != validation.get("a1_contract_sha256")
    ):
        raise AdmissionError("post-wave audit/corpus/selection binding drifted")

    selected_seeds = [int(record["game_seed"]) for record in records]
    if (
        len(set(selected_seeds)) != selected_count
        or not set(map(int, validation_seeds)) <= set(selected_seeds)
    ):
        raise AdmissionError("selected/validation game identities are inconsistent")

    provenance, source_identity = _verify_source_uniformity(audit)
    cli = manifest.get("cli_args")
    if not isinstance(cli, Mapping):
        raise AdmissionError("representative manifest has no effective CLI fields")
    if (
        manifest.get("producer_checkpoint_sha256")
        != source_identity["producer_checkpoint_sha256"]
        or manifest.get("target_information_regime") != COHERENT_REGIME
        or manifest.get("search_evidence_schema") != SEARCH_EVIDENCE_SCHEMA
        or manifest.get("full_config_hash") is None
        or cli.get("n_full") != 128
        or cli.get("coherent_public_belief_search") is not True
        or cli.get("public_observation") is not True
        or cli.get("correct_rust_chance_spectra") is not True
        or cli.get("meaningful_public_history") is not True
        or cli.get("record_automatic_transitions") is not True
        or cli.get("preserve_search_evidence") is not True
        or cli.get("learner_entity_feature_adapter_version")
        != source_identity["entity_feature_adapter_version"]
    ):
        raise AdmissionError("representative manifest is not the audited source operator")
    checkpoint_path = Path(str(manifest.get("checkpoint", ""))).resolve(strict=True)
    if _file_sha256(checkpoint_path) != source_identity["producer_checkpoint_sha256"]:
        raise AdmissionError("source producer checkpoint bytes drifted")

    fields = dict(cli)
    operator = {name: fields.get(name) for name in sorted(OPERATOR_FIELDS)}
    source_binding: dict[str, Any] = {
        "schema_version": SOURCE_BINDING_SCHEMA,
        "status": "bound_historical_post_wave_source_operator",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "producer_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": source_identity["producer_checkpoint_sha256"],
        },
        "target_information_regime": COHERENT_REGIME,
        "operator": operator,
        "typed_generation_config": {
            "schema_version": 13,
            "pipeline": "generate",
            "fields": fields,
            "source_manifest": _ref(
                manifest_path, where="representative generation manifest"
            ),
        },
        "acceptance": {
            "require_search_evidence_schema": SEARCH_EVIDENCE_SCHEMA,
        },
        "evidence": {
            "post_wave_audit": _ref(audit_path, where="post-wave audit"),
            "generation_contract": _ref(
                contract_path, where="generation contract"
            ),
            "source_provenance": provenance,
            "uniform_identity": source_identity,
        },
    }
    source_binding["binding_sha256"] = _value_sha256(source_binding)
    _write_immutable(source_binding_write, source_binding)
    binding_path, written_binding = _load(
        source_binding_write, where="written source operator binding"
    )
    if written_binding != source_binding:
        raise AdmissionError("written source operator binding drifted")

    payload_inventory = meta.get("payload_inventory_sha256")
    source_shards = meta.get("source_shard_inventory")
    if (
        not isinstance(payload_inventory, str)
        or not isinstance(source_shards, list)
        or not source_shards
    ):
        raise AdmissionError("corpus metadata has no immutable payload inventory")
    admission: dict[str, Any] = {
        "schema_version": ADMISSION_SCHEMA,
        "status": "admitted_for_diagnostic_stage_c_reanalysis",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "contract": {
            "path": str(binding_path),
            "file_sha256": _file_sha256(binding_path),
            "contract_sha256": source_binding["binding_sha256"],
        },
        "post_wave_evidence": {
            "audit": _ref(audit_path, where="post-wave audit"),
            "audit_sha256": audit_digest,
            "selected_games": _ref(
                selected_path, where="selected-game manifest"
            ),
            "selected_manifest_sha256": selected_digest,
            "validation_manifest_sha256": validation_digest,
        },
        "corpus": {
            "data_path": str(meta_path.parent),
            "corpus_meta_path": str(meta_path),
            "corpus_meta_file_sha256": _file_sha256(meta_path),
            "payload_inventory_sha256": payload_inventory,
            "validation_manifest": {
                "path": str(validation_path),
                "file_sha256": _file_sha256(validation_path),
            },
            "producer_checkpoint_sha256": source_identity[
                "producer_checkpoint_sha256"
            ],
            "target_information_regime": COHERENT_REGIME,
            "search_evidence_schema": SEARCH_EVIDENCE_SCHEMA,
            "selected_games": int(selected_count),
            "selected_game_seed_set_sha256": selected[
                "selected_game_seed_set_sha256"
            ],
            "complete_two_seat_trace_games": int(selected_count),
            "stored_policy_target_distillation_eligible": False,
            "state_reanalysis_eligible": True,
            "search_evidence_storage": "training_memmap",
            "incompatible_policy_active_rows": 0,
        },
        "policy_target_policy": {
            "stored_targets_are_historical_operator_only": True,
            "current_teacher_requires_reanalysis": True,
            "legacy_pimc_rows_allowed": False,
        },
    }
    admission["admission_sha256"] = _value_sha256(admission)
    _write_immutable(admission_write, admission)
    return admission


def verify_admission(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved, admission = _load(path, where="post-wave Stage-C admission")
    unsigned = dict(admission)
    stated = unsigned.pop("admission_sha256", None)
    corpus = admission.get("corpus")
    contract = admission.get("contract")
    evidence = admission.get("post_wave_evidence")
    if (
        admission.get("schema_version") != ADMISSION_SCHEMA
        or stated != _value_sha256(unsigned)
        or admission.get("status") != "admitted_for_diagnostic_stage_c_reanalysis"
        or admission.get("diagnostic_only") is not True
        or admission.get("promotion_eligible") is not False
        or not isinstance(corpus, Mapping)
        or not isinstance(contract, Mapping)
        or not isinstance(evidence, Mapping)
        or corpus.get("state_reanalysis_eligible") is not True
        or corpus.get("stored_policy_target_distillation_eligible") is not False
        or corpus.get("target_information_regime") != COHERENT_REGIME
        or corpus.get("search_evidence_schema") != SEARCH_EVIDENCE_SCHEMA
    ):
        raise AdmissionError("post-wave Stage-C admission semantics drifted")
    for ref, where in (
        (contract, "source operator binding"),
        (evidence.get("audit"), "post-wave audit"),
        (evidence.get("selected_games"), "selected-game manifest"),
        (corpus.get("validation_manifest"), "validation manifest"),
    ):
        if not isinstance(ref, Mapping):
            raise AdmissionError(f"post-wave admission lost {where}")
        artifact, _payload = _load(Path(str(ref.get("path", ""))), where=where)
        if _file_sha256(artifact) != ref.get("file_sha256"):
            raise AdmissionError(f"post-wave admission {where} bytes drifted")
    meta, _payload = _load(
        Path(str(corpus.get("corpus_meta_path", ""))), where="corpus metadata"
    )
    if (
        meta.parent != Path(str(corpus.get("data_path", ""))).resolve(strict=True)
        or _file_sha256(meta) != corpus.get("corpus_meta_file_sha256")
    ):
        raise AdmissionError("post-wave admission corpus metadata drifted")
    return resolved, admission


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    admit = sub.add_parser("admit")
    admit.add_argument("--corpus-meta", required=True, type=Path)
    admit.add_argument("--representative-manifest", required=True, type=Path)
    admit.add_argument("--source-binding-write", required=True, type=Path)
    admit.add_argument("--write", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--admission", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "admit":
            result = build(
                corpus_meta=args.corpus_meta,
                representative_manifest=args.representative_manifest,
                source_binding_write=args.source_binding_write,
                admission_write=args.write,
            )
        else:
            _path, result = verify_admission(args.admission)
    except (AdmissionError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"post-wave Stage-C admission refused: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
