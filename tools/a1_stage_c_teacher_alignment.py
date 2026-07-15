#!/usr/bin/env python3
"""Seal policy-target quarantine and coherent-n128 reanalysis evidence.

Policy targets are valid only for the exact checkpoint/search operator that
produced them.  Public states and realised outcomes have a longer lifetime.
This tool makes that distinction executable without mutating the source
corpus:

* authenticate the source corpus admission and source/target operator identity;
* write immutable row sidecars for stored-policy eligibility, quarantine,
  value retention, and reanalysis candidacy;
* inventory duplicate-search reliability without treating neutral unaudited
  sentinels as measured confidence;
* choose a deterministic, phase/width/surprise/reliability-stratified subset;
* seal a current coherent-n128 reanalysis plan and its exact blockers.

The current repaired corpus intentionally omitted automatic UI transitions,
so its admission may be valid for stored target distillation while still being
unready for exact state reconstruction.  A plan remains useful in that state:
it seals the target identity and selected roots, but advertises
``execution_ready=false`` rather than silently invoking the legacy PIMC
reanalyzer or conditioning on hidden truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile
import io
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from tools import a1_b200_active_policy_campaign as active_campaign  # noqa: E402
from tools import a1_target_eligibility_inventory as target_inventory  # noqa: E402
from tools import train_bc  # noqa: E402
from catan_zero.rl.target_reliability import (  # noqa: E402
    TARGET_RELIABILITY_COLUMNS,
    TARGET_RELIABILITY_CONFIDENCE_FORMULA,
    TARGET_RELIABILITY_SCHEMA,
    TARGET_RELIABILITY_VERSION,
    target_reliability_confidence,
)
from catan_zero.search.rng_streams import SEARCH_RNG_STREAM_SCHEMA  # noqa: E402


PLAN_SCHEMA = "a1-stage-c-teacher-alignment-plan-v1"
OVERLAY_SCHEMA = "a1-stage-c-target-eligibility-overlay-v1"
SUBSET_SCHEMA = "a1-stage-c-reanalysis-subset-v1"
COHERENT_REGIME = "public_belief_single_tree_v1"
POLICY_STATUS = {
    "inactive_no_stored_policy": 0,
    "eligible_exact_operator": 1,
    "quarantined_stale_operator": 2,
}
RELIABILITY_CLASS = {
    "not_collected": 0,
    "unaudited_neutral_sentinel": 1,
    "duplicate_search_audited": 2,
}
SEARCH_FIELDS = (
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
)
BELIEF_FIELDS = (
    "coherent_public_belief_search",
    "information_set_search",
    "determinization_particles",
    "determinization_min_simulations",
    "belief_chance_spectra",
    "information_set_target_aggregation",
)
CHANCE_FIELDS = (
    "correct_rust_chance_spectra",
    "lazy_interior_chance",
)
SYMMETRY_FIELDS = (
    "symmetry_averaged_eval",
    "symmetry_averaged_eval_threshold",
)
TARGET_SEMANTIC_FIELDS = (
    "public_observation",
    "forced_root_target_mode",
    "record_automatic_transitions",
    "meaningful_public_history",
    "event_history_limit",
    "public_card_count_feature_schema",
    "temperature_clock",
    "preserve_search_evidence",
)


class AlignmentError(RuntimeError):
    """A target identity, corpus, reliability field, or plan is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular_file(path: Path, *, where: str) -> Path:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise AlignmentError(f"{where} must be a regular file: {lexical}")
    try:
        return lexical.resolve(strict=True)
    except OSError as error:
        raise AlignmentError(f"cannot resolve {where}: {error}") from error


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, where=where)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AlignmentError(f"cannot load {where}: {error}") from error
    if not isinstance(payload, dict):
        raise AlignmentError(f"{where} must contain one JSON object")
    return resolved, payload


def _write_immutable(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise AlignmentError(f"immutable output is not a regular file: {destination}")
        if destination.read_bytes() != payload:
            raise AlignmentError(f"immutable output already exists with drift: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    _write_immutable(path, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")


def _write_array_immutable(path: Path, array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    _write_immutable(path, value.tobytes(order="C"))


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            if "/" in key or "\\" in key:
                raise AlignmentError(f"unsafe subset column name: {key!r}")
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[key]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def _resolve_artifact(contract_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        # Operational contracts live three directories below the repository.
        path = contract_path.parents[3] / path
    return _regular_file(path, where="operator artifact")


def _field_bundle(fields: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    missing = [name for name in names if name not in fields]
    if missing:
        raise AlignmentError(f"typed operator config is missing {missing}")
    return {name: fields[name] for name in names}


def _operator_identity(
    contract_path: Path,
    checkpoint: Path,
    *,
    require_current_target: bool = False,
) -> dict[str, Any]:
    """Build the scientific identity of policy targets, excluding fleet layout."""

    if require_current_target:
        try:
            target_inventory.inspect_rd_contract(contract_path)
        except (target_inventory.InventoryError, OSError, ValueError) as error:
            raise AlignmentError(
                f"current coherent operator contract refused: {error}"
            ) from error
    contract_path, contract = _load_json(
        contract_path, where="coherent target operator contract"
    )
    config_ref = contract.get("artifacts", {}).get("typed_generation_config")
    if not isinstance(config_ref, Mapping):
        raise AlignmentError("operator contract has no typed generation config")
    config_path = _resolve_artifact(contract_path, str(config_ref.get("path", "")))
    if _file_sha256(config_path) != config_ref.get("sha256"):
        raise AlignmentError("typed generation config bytes drifted")
    _config_path, config = _load_json(config_path, where="typed generation config")
    fields = config.get("fields")
    operator = contract.get("operator")
    producer = contract.get("producer_checkpoint")
    checkpoint = _regular_file(checkpoint, where="operator checkpoint")
    contract_unsigned = dict(contract)
    contract_digest = contract_unsigned.pop("contract_sha256", None)
    if (
        contract.get("schema_version")
        != target_inventory.RD_CONTRACT_SCHEMA
        or contract_digest != _value_sha256(contract_unsigned)
        or config.get("schema_version") != 13
        or config.get("pipeline") != "generate"
        or not isinstance(fields, Mapping)
        or not isinstance(operator, Mapping)
        or not isinstance(producer, Mapping)
        or producer.get("sha256") != _file_sha256(checkpoint)
    ):
        raise AlignmentError("operator config/checkpoint/regime identity drifted")
    overlap_drift = {
        key: {"contract": value, "typed_config": fields.get(key)}
        for key, value in operator.items()
        if key in fields and fields.get(key) != value
    }
    if overlap_drift:
        raise AlignmentError(
            "contract and typed operator config disagree: "
            + json.dumps(overlap_drift, sort_keys=True)
        )
    search = _field_bundle(fields, SEARCH_FIELDS)
    belief = _field_bundle(fields, BELIEF_FIELDS)
    chance = _field_bundle(fields, CHANCE_FIELDS)
    symmetry = _field_bundle(fields, SYMMETRY_FIELDS)
    semantics = _field_bundle(fields, TARGET_SEMANTIC_FIELDS)
    regime = contract.get("target_information_regime")
    if not isinstance(regime, str) or not regime:
        raise AlignmentError("operator target information regime is missing")
    if require_current_target and (
        regime != COHERENT_REGIME
        or search["n_full"] != 128
        or search["c_scale"] is None
        or belief["coherent_public_belief_search"] is not True
        or belief["information_set_search"] is not False
        or belief["determinization_particles"] != 1
        or chance["correct_rust_chance_spectra"] is not True
        or chance["lazy_interior_chance"] is not True
        or symmetry["symmetry_averaged_eval"] is not True
        or semantics["public_observation"] is not True
    ):
        raise AlignmentError("target operator is not current coherent-public n128")
    value: dict[str, Any] = {
        "schema_version": "a1-operator-bound-policy-target-identity-v1",
        "producer_checkpoint": {
            "path": str(checkpoint),
            "sha256": _file_sha256(checkpoint),
        },
        "target_information_regime": regime,
        "operator_contract_semantics": dict(operator),
        "search": search,
        "belief": {
            **belief,
            "semantic_identity": regime,
        },
        "chance": {
            **chance,
            "rng_stream_schema": SEARCH_RNG_STREAM_SCHEMA,
            "separate_rng_domains": ["gumbel", "chance", "belief"],
        },
        "symmetry": symmetry,
        "target_semantics": {
            **semantics,
            "search_evidence_schema": contract.get("acceptance", {}).get(
                "require_search_evidence_schema"
            ),
            "typed_generation_config_schema": config["schema_version"],
        },
        "authority": {
            "contract": {
                "path": str(contract_path),
                "file_sha256": _file_sha256(contract_path),
                "contract_sha256": contract["contract_sha256"],
            },
            "typed_generation_config": {
                "path": str(config_path),
                "file_sha256": _file_sha256(config_path),
            },
            "operator_semantic_sha256": _value_sha256(operator),
        },
    }
    # The scientific identity deliberately excludes contract path, seed lanes,
    # and fleet placement. Those authenticate provenance but do not change a
    # root target. It includes the producer network and every search semantic.
    scientific = {
        key: value[key]
        for key in (
            "producer_checkpoint",
            "target_information_regime",
            "operator_contract_semantics",
            "search",
            "belief",
            "chance",
            "symmetry",
            "target_semantics",
        )
    }
    scientific["producer_checkpoint"] = {
        "sha256": value["producer_checkpoint"]["sha256"]
    }
    value["identity_sha256"] = _value_sha256(scientific)
    return value


def _classify_policy_rows(
    policy_active: np.ndarray,
    *,
    source_identity_sha256: str,
    target_identity_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(policy_active, dtype=np.bool_)
    exact = source_identity_sha256 == target_identity_sha256
    eligible = active.copy() if exact else np.zeros(active.shape, dtype=np.bool_)
    status = np.full(active.shape, POLICY_STATUS["inactive_no_stored_policy"], np.uint8)
    status[active & eligible] = POLICY_STATUS["eligible_exact_operator"]
    status[active & ~eligible] = POLICY_STATUS["quarantined_stale_operator"]
    return eligible, status


def _reliability_inventory(
    data: Mapping[str, Any], *, row_count: int
) -> tuple[np.ndarray, dict[str, Any]]:
    present = set(TARGET_RELIABILITY_COLUMNS) & set(data.keys())
    if not present:
        classes = np.full(
            row_count, RELIABILITY_CLASS["not_collected"], dtype=np.uint8
        )
        value = {
            "schema_version": "a1-stage-c-reliability-inventory-v1",
            "storage": "not_collected",
            "rows": row_count,
            "audited_rows": 0,
            "unaudited_rows": 0,
            "not_collected_rows": row_count,
            "confidence_weighting_authorized": False,
            "reason": "duplicate-search reliability columns are absent",
        }
        value["inventory_sha256"] = _value_sha256(value)
        return classes, value
    if present != set(TARGET_RELIABILITY_COLUMNS):
        raise AlignmentError(
            "partial target reliability schema; missing "
            f"{sorted(set(TARGET_RELIABILITY_COLUMNS) - present)}"
        )
    columns = {
        name: np.asarray(data[name]).reshape(-1) for name in TARGET_RELIABILITY_COLUMNS
    }
    if any(values.shape != (row_count,) for values in columns.values()):
        raise AlignmentError("target reliability columns are not row-aligned")
    version = columns["target_reliability_version"].astype(np.int64)
    audited = columns["target_reliability_audited"].astype(np.bool_)
    js = columns["target_reliability_js_divergence"].astype(np.float64)
    policy_agree = columns[
        "target_reliability_policy_top1_agreement"
    ].astype(np.bool_)
    q_agree = columns["target_reliability_q_top1_agreement"].astype(np.bool_)
    margin_primary = columns[
        "target_reliability_q_margin_primary"
    ].astype(np.float64)
    margin_duplicate = columns[
        "target_reliability_q_margin_duplicate"
    ].astype(np.float64)
    confidence = columns["target_reliability_confidence"].astype(np.float64)
    unaudited = ~audited
    if np.any(version != TARGET_RELIABILITY_VERSION):
        raise AlignmentError("target reliability version drifted")
    if np.any(
        unaudited
        & (
            ~np.isnan(js)
            | policy_agree
            | q_agree
            | ~np.isnan(margin_primary)
            | ~np.isnan(margin_duplicate)
            | (confidence != 1.0)
        )
    ):
        raise AlignmentError(
            "unaudited reliability rows do not carry the neutral typed sentinel"
        )
    if np.any(
        audited
        & (
            ~np.isfinite(js)
            | (js < 0.0)
            | (js > math.log(2.0) + 1.0e-6)
            | ~np.isfinite(margin_primary)
            | (margin_primary < 0.0)
            | ~np.isfinite(margin_duplicate)
            | (margin_duplicate < 0.0)
            | ~np.isfinite(confidence)
            | (confidence < 0.0)
            | (confidence > 1.0)
        )
    ):
        raise AlignmentError("audited reliability values are malformed")
    audited_indices = np.flatnonzero(audited)
    for index in audited_indices.tolist():
        expected = target_reliability_confidence(
            float(js[index]), policy_top1_agreement=bool(policy_agree[index])
        )
        if not math.isclose(float(confidence[index]), expected, abs_tol=2.0e-6):
            raise AlignmentError("audited reliability confidence formula drifted")
    classes = np.full(
        row_count,
        RELIABILITY_CLASS["unaudited_neutral_sentinel"],
        dtype=np.uint8,
    )
    classes[audited] = RELIABILITY_CLASS["duplicate_search_audited"]
    value = {
        "schema_version": "a1-stage-c-reliability-inventory-v1",
        "storage": "typed_duplicate_search_fields",
        "reliability_schema": TARGET_RELIABILITY_SCHEMA,
        "reliability_version": TARGET_RELIABILITY_VERSION,
        "rows": row_count,
        "audited_rows": int(np.count_nonzero(audited)),
        "unaudited_rows": int(np.count_nonzero(unaudited)),
        "not_collected_rows": 0,
        "confidence_formula": TARGET_RELIABILITY_CONFIDENCE_FORMULA,
        "confidence_weighting_authorized": bool(np.all(audited)),
        "unaudited_confidence_semantics": (
            "neutral sentinel for learner compatibility; never audited evidence"
        ),
    }
    value["inventory_sha256"] = _value_sha256(value)
    return classes, value


def _policy_surprise(
    data: Mapping[str, Any], rows: np.ndarray, *, chunk_rows: int = 8192
) -> np.ndarray:
    result = np.zeros(rows.shape, dtype=np.float32)
    target_column = data["target_policy"]
    prior_column = data["prior_policy"]
    legal_column = data["legal_action_ids"]
    for start in range(0, len(rows), chunk_rows):
        stop = min(start + chunk_rows, len(rows))
        index = rows[start:stop]
        target = np.asarray(target_column[index], dtype=np.float64)
        prior = np.asarray(prior_column[index], dtype=np.float64)
        legal = np.asarray(legal_column[index]) >= 0
        target = np.where(legal, np.maximum(target, 0.0), 0.0)
        prior = np.where(legal, np.maximum(prior, 0.0), 0.0)
        target /= np.maximum(target.sum(axis=1, keepdims=True), 1.0e-12)
        prior /= np.maximum(prior.sum(axis=1, keepdims=True), 1.0e-12)
        terms = np.where(
            target > 0.0,
            target * np.log((target + 1.0e-12) / (prior + 1.0e-12)),
            0.0,
        )
        values = terms.sum(axis=1)
        if np.any(~np.isfinite(values)) or np.any(values < -1.0e-7):
            raise AlignmentError("stored policy surprise is malformed")
        result[start:stop] = np.maximum(values, 0.0).astype(np.float32)
    return result


def _width_bucket(width: int) -> str:
    if width <= 3:
        return "w02_03"
    if width <= 7:
        return "w04_07"
    if width <= 15:
        return "w08_15"
    if width <= 31:
        return "w16_31"
    return "w32_plus"


def _stable_u64(seed: int, *values: int) -> int:
    payload = ":".join(("a1-stage-c-subset-v1", str(seed), *(str(v) for v in values)))
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def _select_stratified(
    *,
    rows: np.ndarray,
    game_seeds: np.ndarray,
    decision_indices: np.ndarray,
    phases: np.ndarray,
    legal_widths: np.ndarray,
    surprise: np.ndarray,
    reliability_class: np.ndarray,
    policy_status: np.ndarray,
    limit: int,
    selection_seed: int,
    max_rows_per_game: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if limit <= 0 or max_rows_per_game <= 0:
        raise AlignmentError("subset limit and max rows per game must be positive")
    arrays = (
        game_seeds,
        decision_indices,
        phases,
        legal_widths,
        surprise,
        reliability_class,
        policy_status,
    )
    if any(np.asarray(value).shape != rows.shape for value in arrays):
        raise AlignmentError("stratified subset inputs are not row-aligned")
    if not len(rows):
        raise AlignmentError("no multi-action policy roots are available for reanalysis")
    quantiles = np.quantile(surprise.astype(np.float64), [0.25, 0.5, 0.75])
    surprise_bins = np.searchsorted(quantiles, surprise, side="right")
    candidates: list[tuple[int, int, str]] = []
    for position, row in enumerate(rows.tolist()):
        stratum = "|".join(
            (
                str(phases[position]),
                _width_bucket(int(legal_widths[position])),
                f"surprise_q{int(surprise_bins[position])}",
                f"reliability_{int(reliability_class[position])}",
                f"policy_status_{int(policy_status[position])}",
            )
        )
        candidates.append(
            (
                _stable_u64(
                    selection_seed,
                    int(game_seeds[position]),
                    int(decision_indices[position]),
                    int(row),
                ),
                position,
                stratum,
            )
        )
    # Enforce game diversity before balancing strata.
    by_hash = sorted(candidates)
    per_game: dict[int, int] = {}
    admitted: list[tuple[int, int, str]] = []
    for item in by_hash:
        game = int(game_seeds[item[1]])
        if per_game.get(game, 0) >= max_rows_per_game:
            continue
        per_game[game] = per_game.get(game, 0) + 1
        admitted.append(item)
    groups: dict[str, list[tuple[int, int, str]]] = {}
    for item in admitted:
        groups.setdefault(item[2], []).append(item)
    for values in groups.values():
        values.sort()
    selected_positions: list[int] = []
    round_index = 0
    sorted_strata = sorted(groups)
    target = min(limit, len(admitted))
    while len(selected_positions) < target:
        progressed = False
        for stratum in sorted_strata:
            values = groups[stratum]
            if round_index < len(values):
                selected_positions.append(values[round_index][1])
                progressed = True
                if len(selected_positions) == target:
                    break
        if not progressed:
            break
        round_index += 1
    selected_positions_array = np.asarray(selected_positions, dtype=np.int64)
    selected_strata = np.asarray(
        [candidates[position][2] for position in selected_positions], dtype=str
    )
    counts = {
        stratum: int(np.count_nonzero(selected_strata == stratum))
        for stratum in np.unique(selected_strata)
    }
    return selected_positions_array, selected_strata, counts


def _artifact_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    try:
        admission_path, admission = active_campaign._load_admission(  # noqa: SLF001
            args.coherent_corpus_admission
        )
    except active_campaign.CampaignError as error:
        raise AlignmentError(f"coherent corpus admission refused: {error}") from error
    corpus_record = admission["corpus"]
    corpus_root = Path(str(corpus_record["data_path"])).resolve(strict=True)
    source_contract_path = Path(str(admission["contract"]["path"]))
    source_checkpoint = Path(
        str(_load_json(source_contract_path, where="source operator contract")[1][
            "producer_checkpoint"
        ]["path"])
    )
    source_identity = _operator_identity(source_contract_path, source_checkpoint)
    target_identity = _operator_identity(
        args.target_operator_contract,
        args.target_checkpoint,
        require_current_target=True,
    )
    if (
        source_identity["producer_checkpoint"]["sha256"]
        != corpus_record["producer_checkpoint_sha256"]
        or source_identity["target_information_regime"]
        != corpus_record["target_information_regime"]
    ):
        raise AlignmentError(
            "source operator identity is not bound to the admitted corpus"
        )

    data = train_bc.MemmapCorpus(corpus_root)
    rows = len(data)
    policy_weight = np.asarray(data["policy_weight_multiplier"], dtype=np.float32)
    policy_active = policy_weight > 0.0
    legal_widths_all = np.asarray(
        data["legal_action_ids"].row_counts(), dtype=np.int64
    )
    if policy_weight.shape != (rows,) or legal_widths_all.shape != (rows,):
        raise AlignmentError("corpus policy/legal columns are not row-aligned")
    stored_policy_eligible, policy_status = _classify_policy_rows(
        policy_active,
        source_identity_sha256=source_identity["identity_sha256"],
        target_identity_sha256=target_identity["identity_sha256"],
    )
    value_eligible = (
        np.asarray(data["value_weight_multiplier"], dtype=np.float32) > 0.0
        if "value_weight_multiplier" in data
        else np.ones(rows, dtype=np.bool_)
    )
    # Reanalysis creates a fresh operator-bound target.  It is therefore useful
    # for stale active rows *and* for multi-action roots whose original fast
    # search did not store policy supervision.  Policy inactivity must not be
    # confused with state ineligibility.
    reanalysis_candidate = legal_widths_all > 1
    reliability_classes, reliability = _reliability_inventory(data, row_count=rows)

    candidate_rows = np.flatnonzero(reanalysis_candidate).astype(np.int64)
    game_seeds = np.asarray(data["game_seed"][candidate_rows], dtype=np.int64)
    decision_indices = np.asarray(
        data["decision_index"][candidate_rows], dtype=np.int64
    )
    phases = np.asarray(data["phase"][candidate_rows]).astype(str)
    legal_widths = legal_widths_all[candidate_rows]
    surprise = _policy_surprise(data, candidate_rows)
    candidate_reliability = reliability_classes[candidate_rows]
    candidate_status = policy_status[candidate_rows]
    selected_positions, selected_strata, stratum_counts = _select_stratified(
        rows=candidate_rows,
        game_seeds=game_seeds,
        decision_indices=decision_indices,
        phases=phases,
        legal_widths=legal_widths,
        surprise=surprise,
        reliability_class=candidate_reliability,
        policy_status=candidate_status,
        limit=int(args.subset_rows),
        selection_seed=int(args.selection_seed),
        max_rows_per_game=int(args.max_rows_per_game),
    )
    selected_rows = candidate_rows[selected_positions]
    chunks = int(args.chunks)
    if chunks <= 0:
        raise AlignmentError("chunks must be positive")
    chunk_index = np.arange(len(selected_rows), dtype=np.int32) % chunks
    identity_sha = np.asarray(
        [
            _value_sha256(
                {
                    "corpus_meta_file_sha256": corpus_record[
                        "corpus_meta_file_sha256"
                    ],
                    "row_index": int(row),
                    "game_seed": int(game_seeds[position]),
                    "decision_index": int(decision_indices[position]),
                }
            )
            for row, position in zip(
                selected_rows.tolist(), selected_positions.tolist(), strict=True
            )
        ],
        dtype="<U71",
    )

    output_root = args.output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    policy_status_path = output_root / "policy_status.u8.dat"
    policy_eligible_path = output_root / "stored_policy_eligible.bool.dat"
    value_eligible_path = output_root / "value_eligible.bool.dat"
    candidate_path = output_root / "reanalysis_candidate.bool.dat"
    subset_path = output_root / "selected_reanalysis_rows.npz"
    _write_array_immutable(policy_status_path, policy_status)
    _write_array_immutable(policy_eligible_path, stored_policy_eligible)
    _write_array_immutable(value_eligible_path, value_eligible)
    _write_array_immutable(candidate_path, reanalysis_candidate)
    subset_arrays = {
        "row_index": selected_rows.astype(np.int64),
        "game_seed": game_seeds[selected_positions].astype(np.int64),
        "decision_index": decision_indices[selected_positions].astype(np.int64),
        "phase": phases[selected_positions].astype(str),
        "legal_width": legal_widths[selected_positions].astype(np.int16),
        "policy_surprise_kl": surprise[selected_positions].astype(np.float32),
        "reliability_class": candidate_reliability[selected_positions].astype(np.uint8),
        "source_policy_status": candidate_status[selected_positions].astype(np.uint8),
        "stratum": selected_strata.astype(str),
        "chunk_index": chunk_index,
        "identity_sha256": identity_sha,
    }
    _write_immutable(subset_path, _npz_bytes(subset_arrays))

    overlay: dict[str, Any] = {
        "schema_version": OVERLAY_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "corpus": {
            "path": str(corpus_root),
            "corpus_meta_file_sha256": corpus_record["corpus_meta_file_sha256"],
            "payload_inventory_sha256": corpus_record["payload_inventory_sha256"],
            "row_count": rows,
        },
        "source_policy_target_identity_sha256": source_identity["identity_sha256"],
        "required_policy_target_identity_sha256": target_identity["identity_sha256"],
        "status_codes": POLICY_STATUS,
        "artifacts": {
            "policy_status": _artifact_ref(policy_status_path),
            "stored_policy_eligible": _artifact_ref(policy_eligible_path),
            "value_eligible": _artifact_ref(value_eligible_path),
            "reanalysis_candidate": _artifact_ref(candidate_path),
        },
        "counts": {
            "rows": rows,
            "stored_policy_active_rows": int(np.count_nonzero(policy_active)),
            "stored_policy_eligible_rows": int(
                np.count_nonzero(stored_policy_eligible)
            ),
            "stored_policy_quarantined_rows": int(
                np.count_nonzero(
                    policy_status == POLICY_STATUS["quarantined_stale_operator"]
                )
            ),
            "value_retained_rows": int(np.count_nonzero(value_eligible)),
            "reanalysis_candidate_rows": int(np.count_nonzero(reanalysis_candidate)),
        },
        "policy_quarantine_changes_value_eligibility": False,
        "policy_quarantine_changes_state_reanalysis_eligibility": False,
    }
    overlay["overlay_sha256"] = _value_sha256(overlay)
    overlay_path = output_root / "target_eligibility.overlay.json"
    _write_json_immutable(overlay_path, overlay)

    state_ready = bool(corpus_record.get("state_reanalysis_eligible"))
    blockers: list[str] = []
    if not state_ready:
        blockers.extend(
            [
                "per_row_sparse_reconstruction_qualification_required",
            ]
        )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "purpose": "current_coherent_n128_operator_aligned_reanalysis_subset",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "source_corpus_admission": {
            "path": str(admission_path),
            "file_sha256": _file_sha256(admission_path),
            "admission_sha256": admission["admission_sha256"],
        },
        "evidence_lifetimes": {
            "state_evidence": "reusable_if_information_surface_and_reconstruction_hold",
            "terminal_value_evidence": "reusable_independent_of_search_operator",
            "stored_policy_targets": "eligible_only_for_exact_policy_target_identity_sha256",
        },
        "source_policy_target_identity": source_identity,
        "target_policy_target_identity": target_identity,
        "target_identity_matches_stored_policy": (
            source_identity["identity_sha256"] == target_identity["identity_sha256"]
        ),
        "eligibility_overlay": {
            "path": str(overlay_path),
            "file_sha256": _file_sha256(overlay_path),
            "overlay_sha256": overlay["overlay_sha256"],
        },
        "source_reliability_evidence": reliability,
        "output_reliability_contract": {
            "mode": "primary_search_only_unaudited_v1",
            "duplicate_search_audit_scheduled": False,
            "unaudited_rows_must_use_typed_neutral_sentinel": True,
            "unaudited_confidence_is_measured_evidence": False,
            "future_duplicate_search_input_schema": TARGET_RELIABILITY_SCHEMA,
            "future_duplicate_search_rng_stream_schema": SEARCH_RNG_STREAM_SCHEMA,
        },
        "subset": {
            "schema_version": SUBSET_SCHEMA,
            "artifact": _artifact_ref(subset_path),
            "selected_rows": len(selected_rows),
            "candidate_rows": len(candidate_rows),
            "requested_rows": int(args.subset_rows),
            "selection_seed": int(args.selection_seed),
            "max_rows_per_game": int(args.max_rows_per_game),
            "stratification": [
                "phase",
                "legal_width_bucket",
                "stored_policy_surprise_quartile",
                "reliability_evidence_class",
                "stored_policy_eligibility_status",
            ],
            "stratum_counts": stratum_counts,
            "chunks": chunks,
            "assignment": "selected_ordinal_mod_chunks",
        },
        "execution": {
            "executor_semantics": "coherent_public_belief_n128_reanalysis_v1",
            "reconstruction_qualifier": (
                "tools/a1_stage_c_reanalysis_executor.py qualify"
            ),
            "sparse_gap_rule": (
                "an omitted decision is executable iff replay proves exactly "
                "one legal action at that index"
            ),
            "readiness_scope": "per_selected_row_not_whole_corpus",
            "legacy_public_conservation_pimc_executor_allowed": False,
            "authoritative_hidden_state_search_allowed": False,
            "execution_ready": state_ready,
            "blockers": blockers,
        },
    }
    plan["plan_sha256"] = _value_sha256(plan)
    return plan


def _verify_plan(path: Path) -> dict[str, Any]:
    plan_path, plan = _load_json(path, where="Stage-C alignment plan")
    unsigned = dict(plan)
    stated = unsigned.pop("plan_sha256", None)
    if plan.get("schema_version") != PLAN_SCHEMA or stated != _value_sha256(unsigned):
        raise AlignmentError("Stage-C plan schema or semantic digest drifted")
    for identity_key in (
        "source_policy_target_identity",
        "target_policy_target_identity",
    ):
        identity = plan.get(identity_key)
        if not isinstance(identity, Mapping):
            raise AlignmentError(f"Stage-C plan lacks {identity_key}")
        authority = identity.get("authority", {}).get("contract")
        checkpoint = identity.get("producer_checkpoint")
        if not isinstance(authority, Mapping) or not isinstance(checkpoint, Mapping):
            raise AlignmentError(f"Stage-C {identity_key} authority is malformed")
        replayed = _operator_identity(
            Path(str(authority["path"])),
            Path(str(checkpoint["path"])),
            require_current_target=(identity_key == "target_policy_target_identity"),
        )
        if replayed != identity:
            raise AlignmentError(f"Stage-C {identity_key} no longer replays")
    overlay_ref = plan.get("eligibility_overlay")
    if not isinstance(overlay_ref, Mapping):
        raise AlignmentError("Stage-C plan lacks eligibility overlay")
    overlay_path, overlay = _load_json(
        Path(str(overlay_ref["path"])), where="Stage-C eligibility overlay"
    )
    overlay_unsigned = dict(overlay)
    overlay_stated = overlay_unsigned.pop("overlay_sha256", None)
    if (
        overlay.get("schema_version") != OVERLAY_SCHEMA
        or overlay_stated != _value_sha256(overlay_unsigned)
        or overlay_ref.get("file_sha256") != _file_sha256(overlay_path)
        or overlay_ref.get("overlay_sha256") != overlay_stated
    ):
        raise AlignmentError("Stage-C eligibility overlay drifted")
    for artifact in overlay["artifacts"].values():
        artifact_path = _regular_file(
            Path(str(artifact["path"])), where="Stage-C overlay artifact"
        )
        if (
            artifact.get("file_sha256") != _file_sha256(artifact_path)
            or artifact.get("size_bytes") != artifact_path.stat().st_size
        ):
            raise AlignmentError("Stage-C overlay artifact bytes drifted")
    subset = plan.get("subset", {}).get("artifact")
    if not isinstance(subset, Mapping):
        raise AlignmentError("Stage-C plan lacks selected subset")
    subset_path = _regular_file(Path(str(subset["path"])), where="Stage-C subset")
    if (
        subset.get("file_sha256") != _file_sha256(subset_path)
        or subset.get("size_bytes") != subset_path.stat().st_size
    ):
        raise AlignmentError("Stage-C selected subset bytes drifted")
    with np.load(subset_path, allow_pickle=False) as arrays:
        count = len(arrays["row_index"])
        if (
            count != int(plan["subset"]["selected_rows"])
            or any(len(arrays[name]) != count for name in arrays.files)
            or np.unique(arrays["identity_sha256"]).size != count
        ):
            raise AlignmentError("Stage-C selected subset row identity drifted")
    return {"path": str(plan_path), "file_sha256": _file_sha256(plan_path), **plan}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write quarantine overlay and sealed subset")
    plan.add_argument("--coherent-corpus-admission", required=True, type=Path)
    plan.add_argument("--target-operator-contract", required=True, type=Path)
    plan.add_argument("--target-checkpoint", required=True, type=Path)
    plan.add_argument("--subset-rows", type=int, default=8192)
    plan.add_argument("--selection-seed", type=int, default=20260715)
    plan.add_argument("--max-rows-per-game", type=int, default=4)
    plan.add_argument("--chunks", type=int, default=64)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)
    verify = sub.add_parser("verify", help="replay a sealed Stage-C plan")
    verify.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = _build_plan(args)
            _write_json_immutable(args.write, result)
        else:
            result = _verify_plan(args.plan)
    except (
        AlignmentError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Stage-C teacher alignment refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
