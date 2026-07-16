#!/usr/bin/env python3
"""Seal empirical evidence that coherent policy targets beat the raw evaluator.

The admission unit is a terminal, policy-active, full-search decision.  Both
candidate estimators are compared with the same terminal outcome and the same
pre-search ``root_prior_value`` baseline. Row deltas are averaged within each
component/game pair before a deterministic game-cluster bootstrap, so long
games and coincident seed namespaces cannot dominate the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import train_bc  # noqa: E402


RECEIPT_SCHEMA = "a1-policy-target-quality-admission-v1"
IDENTITY_SCHEMA = "a1-policy-target-quality-identity-v1"
METRIC_SCHEMA = "a1-policy-target-terminal-q-bootstrap-v1"
SEARCH_EVIDENCE_SCHEMA = "gumbel_root_search_evidence_v2_fp32_prior"
BOOTSTRAP_SEED = 20_260_716
BOOTSTRAP_REPLICATES = 10_000
ONE_SIDED_CONFIDENCE = 0.95
MINIMUM_GAME_CLUSTERS = 256
BOOTSTRAP_MAX_INDEX_ENTRIES = 1_000_000
REQUIRED_COLUMNS = frozenset(
    {
        "game_seed",
        "player",
        "winner",
        "terminated",
        "truncated",
        "policy_weight_multiplier",
        "used_full_search",
        "root_prior_value",
        "root_prior_value_mask",
        "action_taken",
        "legal_action_ids",
        "target_policy",
        "target_policy_mask",
        "search_evidence_version",
        "search_evidence_mask",
        "search_completed_q_flat",
    }
)


class AdmissionError(RuntimeError):
    """The empirical policy-target quality claim is missing or inconclusive."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _regular_file(path: Path, *, where: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise AdmissionError(f"{where} may not be a symlink: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise AdmissionError(f"{where} must be a regular file: {resolved}")
    return resolved


def _json_file(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, where=where)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdmissionError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise AdmissionError(f"{where} must contain a JSON object")
    return resolved, value


def _ref(path: Path, *, where: str) -> dict[str, str]:
    resolved = _regular_file(path, where=where)
    return {"path": str(resolved), "file_sha256": _file_sha256(resolved)}


def metric_contract() -> dict[str, Any]:
    return {
        "schema_version": METRIC_SCHEMA,
        "unit": "terminal_policy_active_full_search_row",
        "outcome": "terminal_z_from_acting_player_perspective",
        "baseline": "root_prior_value_before_search",
        "primary_estimator": "target_policy_weighted_completed_q",
        "secondary_estimator": "selected_action_completed_q",
        "paired_delta": "candidate_squared_error_minus_baseline_squared_error",
        "row_to_game_reduction": "uniform_mean_within_component_id_and_game_seed",
        "game_reduction": "uniform_mean_across_component_id_and_game_seed_clusters",
        "bootstrap": (
            "percentile_resample_component_id_and_game_seed_clusters_with_replacement"
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "one_sided_confidence": ONE_SIDED_CONFIDENCE,
        "required_upper_confidence_bound": 0.0,
        "minimum_game_clusters": MINIMUM_GAME_CLUSTERS,
        "both_primary_and_secondary_must_pass": True,
    }


def _metric_code_identity() -> dict[str, Any]:
    path = Path(__file__).resolve(strict=True)
    return {
        "relative_path": "tools/a1_policy_target_quality_admission.py",
        **_ref(path, where="policy-target metric code"),
        "metric_contract_sha256": _value_sha256(metric_contract()),
    }


def _component_records(
    composite_meta: Mapping[str, Any],
) -> list[dict[str, Any]]:
    components = composite_meta.get("components")
    selected_ids = composite_meta.get("policy_distillation_component_ids")
    if not isinstance(components, list) or not isinstance(selected_ids, list):
        raise AdmissionError("composite has no authenticated policy component scope")
    by_id = {
        str(component.get("component_id")): component
        for component in components
        if isinstance(component, Mapping)
    }
    if not selected_ids or len(set(map(str, selected_ids))) != len(selected_ids):
        raise AdmissionError("policy component scope is empty or duplicated")
    records: list[dict[str, Any]] = []
    for raw_id in selected_ids:
        component_id = str(raw_id)
        component = by_id.get(component_id)
        if not isinstance(component, Mapping):
            raise AdmissionError(f"policy component {component_id!r} is missing")
        corpus_meta = component.get("corpus_meta")
        if not isinstance(corpus_meta, Mapping):
            raise AdmissionError(f"policy component {component_id!r} has no metadata")
        evidence = corpus_meta.get("search_evidence")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("schema") != SEARCH_EVIDENCE_SCHEMA
        ):
            raise AdmissionError(
                f"policy component {component_id!r} lacks v2 search evidence"
            )
        provenance_ref = corpus_meta.get("flywheel_component_provenance")
        if not isinstance(provenance_ref, Mapping):
            raise AdmissionError(
                f"policy component {component_id!r} lacks source provenance"
            )
        provenance_path, provenance = _json_file(
            Path(str(provenance_ref.get("path", ""))),
            where=f"{component_id} source provenance",
        )
        if provenance_ref.get("file_sha256") != _file_sha256(provenance_path):
            raise AdmissionError(f"{component_id} source provenance bytes drifted")
        checkpoints = provenance.get("producer_checkpoints")
        shard_inventory = provenance.get("shard_inventory_sha256")
        target_identity = provenance.get("policy_target_identity_sha256")
        if (
            not isinstance(checkpoints, list)
            or not checkpoints
            or not isinstance(shard_inventory, str)
            or not train_bc._is_sha256(shard_inventory)  # noqa: SLF001
            or not isinstance(target_identity, str)
            or not train_bc._is_sha256(target_identity)  # noqa: SLF001
        ):
            raise AdmissionError(f"{component_id} source identity is incomplete")
        corpus_dir = Path(str(component.get("corpus_dir", ""))).resolve(strict=True)
        records.append(
            {
                "component_id": component_id,
                "corpus_dir": str(corpus_dir),
                "corpus_meta_sha256": component.get("corpus_meta_sha256"),
                "payload_inventory_sha256": component.get("payload_inventory_sha256"),
                "source_provenance": {
                    "path": str(provenance_path),
                    "file_sha256": _file_sha256(provenance_path),
                    "shard_inventory_sha256": shard_inventory,
                },
                "producer_checkpoints": checkpoints,
                "policy_target_identity_sha256": target_identity,
            }
        )
    return records


def expected_identity(
    *, verified: Mapping[str, Any], composite_meta: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the exact external identity a valid receipt must carry."""

    lock_path, lock = _json_file(Path(str(verified["lock_path"])), where="A1 lock")
    contract_sha256 = lock.get("contract_sha256")
    if not isinstance(contract_sha256, str) or not train_bc._is_sha256(  # noqa: SLF001
        contract_sha256
    ):
        raise AdmissionError("A1 lock has no sealed contract identity")
    accepted_target = verified.get("accepted_policy_target_identity_sha256")
    if not isinstance(accepted_target, str) or not train_bc._is_sha256(  # noqa: SLF001
        accepted_target
    ):
        raise AdmissionError("scratch inputs have no accepted policy-target identity")
    source_authority = verified.get("source_authority")
    selected_games = (
        source_authority.get("selected_game_manifest")
        if isinstance(source_authority, Mapping)
        else None
    )
    if not isinstance(selected_games, Mapping):
        raise AdmissionError("scratch source authority has no selected-game identity")
    try:
        from tools import build_memmap_corpus as memmap_builder

        selected = memmap_builder._load_a1_selected_game_manifest(  # noqa: SLF001
            Path(str(selected_games.get("path", "")))
        )
    except (OSError, SystemExit, ValueError) as error:
        raise AdmissionError(
            f"scratch selected-game identity cannot be verified: {error}"
        ) from error
    for field in (
        "file_sha256",
        "manifest_sha256",
        "records_sha256",
        "selected_game_seed_set_sha256",
    ):
        if selected_games.get(field) != selected.get(field):
            raise AdmissionError(f"scratch selected-game identity drifted at {field!r}")
    selected_game_identity = {
        "path": str(selected["path"]),
        "file_sha256": selected["file_sha256"],
        "manifest_sha256": selected["manifest_sha256"],
        "records_sha256": selected["records_sha256"],
        "selected_game_count": int(selected["selected_game_count"]),
        "selected_game_seed_set_sha256": selected["selected_game_seed_set_sha256"],
        "training_game_count": int(selected["training_game_count"]),
        "training_game_seed_set_sha256": selected["training_game_seed_set_sha256"],
        "validation_game_count": int(selected["validation_game_count"]),
        "validation_game_seed_set_sha256": selected["validation_game_seed_set_sha256"],
    }
    components = _component_records(composite_meta)
    component_targets = {
        str(record["policy_target_identity_sha256"]) for record in components
    }
    if component_targets != {accepted_target}:
        raise AdmissionError("component policy-target identity differs from scratch")
    science = current_science.load()
    search = current_science.search()
    if not isinstance(search, Mapping):
        raise AdmissionError("current science search operator is malformed")
    descriptor = _ref(Path(str(verified["data_path"])), where="composite descriptor")
    descriptor.update(
        {
            "fingerprint": verified.get("descriptor_fingerprint"),
            "payload_inventory_sha256": verified.get("payload_inventory_sha256"),
            "source_authority_semantic_sha256": verified.get(
                "source_authority_semantic_sha256"
            ),
        }
    )
    return {
        "schema_version": IDENTITY_SCHEMA,
        "science_contract": {
            **_ref(current_science.CONTRACT_PATH, where="current science contract"),
            "semantic_sha256": _value_sha256(science),
        },
        "sealed_a1_contract": {
            "path": str(lock_path),
            "file_sha256": _file_sha256(lock_path),
            "contract_sha256": contract_sha256,
        },
        "operator": {
            "search_operator_sha256": _value_sha256(search),
            "policy_target_identity_sha256": accepted_target,
        },
        "composite": descriptor,
        "selected_game_set": selected_game_identity,
        "policy_components": components,
        "metric_code": _metric_code_identity(),
    }


def _game_cluster_deltas(
    component_records: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray, int]:
    by_game: dict[tuple[str, int], list[tuple[float, float]]] = {}
    row_count = 0
    for record in component_records:
        component_id = str(record["component_id"])
        corpus_dir = Path(str(record["corpus_dir"]))
        corpus = train_bc.MemmapCorpus(corpus_dir)
        missing = REQUIRED_COLUMNS - set(corpus.keys())
        if missing:
            raise AdmissionError(
                f"{component_id} lacks policy quality columns: {sorted(missing)}"
            )
        authenticated = train_bc._validate_memmap_payload_inventory(  # noqa: SLF001
            corpus_dir, corpus.meta
        )
        if authenticated != record.get("payload_inventory_sha256"):
            raise AdmissionError(f"{component_id} payload inventory drifted")
        policy_weight = np.asarray(corpus["policy_weight_multiplier"], dtype=np.float64)
        if (
            policy_weight.ndim != 1
            or not np.isfinite(policy_weight).all()
            or bool(np.any(policy_weight < 0.0))
        ):
            raise AdmissionError(f"{component_id} has invalid policy weights")
        full_search = np.asarray(corpus["used_full_search"], dtype=np.bool_)
        search_mask = np.asarray(corpus["search_evidence_mask"], dtype=np.bool_)
        prior_mask = np.asarray(corpus["root_prior_value_mask"], dtype=np.bool_)
        if not (
            full_search.shape
            == search_mask.shape
            == prior_mask.shape
            == policy_weight.shape
        ):
            raise AdmissionError(f"{component_id} quality masks are not row-aligned")
        active = (policy_weight > 0.0) & full_search
        if bool(np.any(active & (~search_mask | ~prior_mask))):
            raise AdmissionError(
                f"{component_id} policy rows lack search evidence or root prior"
            )
        rows = np.flatnonzero(active)
        if rows.size == 0:
            raise AdmissionError(f"{component_id} has no eligible policy rows")
        block_rows = 65_536
        for start in range(0, int(rows.size), block_rows):
            batch = rows[start : start + block_rows]
            terminated = np.asarray(corpus["terminated"][batch], dtype=np.bool_)
            truncated = np.asarray(corpus["truncated"][batch], dtype=np.bool_)
            if not bool(np.all(terminated & ~truncated)):
                raise AdmissionError(
                    f"{component_id} policy evidence includes non-terminal games"
                )
            seeds = np.asarray(corpus["game_seed"][batch], dtype=np.int64)
            players = np.asarray(corpus["player"][batch]).astype(str)
            winners = np.asarray(corpus["winner"][batch]).astype(str)
            if bool(np.any(players == "")) or bool(np.any(winners == "")):
                raise AdmissionError(f"{component_id} has missing terminal outcomes")
            terminal_z = np.where(players == winners, 1.0, -1.0)
            root = np.asarray(corpus["root_prior_value"][batch], dtype=np.float64)
            legal = np.asarray(corpus["legal_action_ids"][batch], dtype=np.int64)
            actions = np.asarray(corpus["action_taken"][batch], dtype=np.int64)
            target = np.asarray(corpus["target_policy"][batch], dtype=np.float64)
            target_mask = np.asarray(
                corpus["target_policy_mask"][batch], dtype=np.bool_
            )
            completed_q = np.asarray(
                corpus["search_completed_q_flat"][batch], dtype=np.float64
            )
            evidence_version = np.asarray(
                corpus["search_evidence_version"][batch], dtype=np.uint8
            )
            legal_support = legal >= 0
            support = legal_support & target_mask
            invalid = (
                np.any(evidence_version != 2)
                or not np.isfinite(root).all()
                or bool(np.any((root < -1.0) | (root > 1.0)))
                or not bool(np.array_equal(target_mask, legal_support))
                or not np.isfinite(target[support]).all()
                or bool(np.any(target[support] < 0.0))
                or not np.isfinite(completed_q[legal_support]).all()
                or bool(
                    np.any(
                        (completed_q[legal_support] < -1.0)
                        | (completed_q[legal_support] > 1.0)
                    )
                )
            )
            if invalid:
                raise AdmissionError(f"{component_id} has malformed policy evidence")
            mass = np.where(support, target, 0.0).sum(axis=1)
            if bool(np.any(~np.isfinite(mass))) or not bool(
                np.allclose(mass, 1.0, rtol=1.0e-5, atol=1.0e-6)
            ):
                raise AdmissionError(f"{component_id} target policy is not normalized")
            weighted_q = np.where(support, target * completed_q, 0.0).sum(axis=1) / mass
            selected = legal == actions[:, None]
            if bool(np.any(selected.sum(axis=1) != 1)):
                raise AdmissionError(
                    f"{component_id} selected action is not legal-unique"
                )
            selected_q = completed_q[selected]
            if not np.isfinite(selected_q).all():
                raise AdmissionError(f"{component_id} selected-Q is non-finite")
            baseline_error = np.square(root - terminal_z)
            primary_delta = np.square(weighted_q - terminal_z) - baseline_error
            secondary_delta = np.square(selected_q - terminal_z) - baseline_error
            if (
                not np.isfinite(primary_delta).all()
                or not np.isfinite(secondary_delta).all()
            ):
                raise AdmissionError(f"{component_id} metric delta is non-finite")
            for seed, primary, secondary in zip(
                seeds, primary_delta, secondary_delta, strict=True
            ):
                by_game.setdefault((component_id, int(seed)), []).append(
                    (float(primary), float(secondary))
                )
            row_count += int(batch.size)
    if len(by_game) < MINIMUM_GAME_CLUSTERS:
        raise AdmissionError("policy quality evidence has too few game clusters")
    ordered = sorted(by_game)
    primary_games = np.asarray(
        [np.mean([row[0] for row in by_game[key]]) for key in ordered],
        dtype=np.float64,
    )
    secondary_games = np.asarray(
        [np.mean([row[1] for row in by_game[key]]) for key in ordered],
        dtype=np.float64,
    )
    return ordered, primary_games, secondary_games, row_count


def _bootstrap_metric(values: np.ndarray, samples: np.ndarray) -> dict[str, Any]:
    # Spell out NumPy's linear quantile so the sealed metric is identical on
    # the production H100 image's NumPy 1.21 and newer developer environments
    # (the keyword was renamed from ``interpolation`` to ``method`` in 1.22).
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    position = ONE_SIDED_CONFIDENCE * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper_index = int(math.ceil(position))
    fraction = position - float(lower)
    upper = float(ordered[lower] * (1.0 - fraction) + ordered[upper_index] * fraction)
    estimate = float(values.mean())
    if not math.isfinite(estimate) or not math.isfinite(upper):
        raise AdmissionError("bootstrap result is non-finite")
    return {
        "paired_delta_mean": estimate,
        "bootstrap_one_sided_95_ucb": upper,
        "passes": upper <= 0.0,
    }


def evaluate(component_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    game_keys, primary, secondary, row_count = _game_cluster_deltas(component_records)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary_samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    secondary_samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    # Keep bootstrap scratch bounded when the selected set contains thousands
    # of games. Both metrics use the same draws, preserving their paired audit.
    block = max(1, BOOTSTRAP_MAX_INDEX_ENTRIES // len(game_keys))
    for start in range(0, BOOTSTRAP_REPLICATES, block):
        stop = min(start + block, BOOTSTRAP_REPLICATES)
        draws = rng.integers(
            0,
            len(game_keys),
            size=(stop - start, len(game_keys)),
            dtype=np.int64,
        )
        primary_samples[start:stop] = primary[draws].mean(axis=1)
        secondary_samples[start:stop] = secondary[draws].mean(axis=1)
    primary_result = _bootstrap_metric(primary, primary_samples)
    secondary_result = _bootstrap_metric(secondary, secondary_samples)
    admitted = bool(primary_result["passes"] and secondary_result["passes"])
    return {
        "eligible_row_count": row_count,
        "game_cluster_count": int(len(game_keys)),
        "component_game_set_sha256": _value_sha256(
            [
                {"component_id": component_id, "game_seed": int(seed)}
                for component_id, seed in game_keys
            ]
        ),
        "primary": primary_result,
        "secondary_selected_q": secondary_result,
        "admitted": admitted,
    }


def generate_receipt(
    *,
    verified: Mapping[str, Any],
    composite_meta: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise AdmissionError(f"refusing non-fresh receipt path: {output_path}")
    identity = expected_identity(verified=verified, composite_meta=composite_meta)
    metrics = evaluate(identity["policy_components"])
    if not metrics["admitted"]:
        raise AdmissionError(
            "policy-target quality is inconclusive: both bootstrap UCBs must be <= 0"
        )
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "admitted",
        "identity": identity,
        "metric_contract": metric_contract(),
        "metrics": metrics,
    }
    payload["receipt_sha256"] = _value_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def verify_receipt(
    path: Path,
    *,
    verified: Mapping[str, Any],
    composite_meta: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path, payload = _json_file(path, where="policy-target quality receipt")
    unsigned = dict(payload)
    stated = unsigned.pop("receipt_sha256", None)
    expected = expected_identity(verified=verified, composite_meta=composite_meta)
    metrics = unsigned.get("metrics")
    if (
        stated != _value_sha256(unsigned)
        or set(unsigned)
        != {"schema_version", "status", "identity", "metric_contract", "metrics"}
        or unsigned.get("schema_version") != RECEIPT_SCHEMA
        or unsigned.get("status") != "admitted"
        or unsigned.get("identity") != expected
        or unsigned.get("metric_contract") != metric_contract()
        or not isinstance(metrics, Mapping)
        or metrics.get("admitted") is not True
        or int(metrics.get("eligible_row_count", 0)) <= 0
        or int(metrics.get("game_cluster_count", 0)) < MINIMUM_GAME_CLUSTERS
    ):
        raise AdmissionError("policy-target quality receipt identity or status drifted")
    for name in ("primary", "secondary_selected_q"):
        result = metrics.get(name)
        if (
            not isinstance(result, Mapping)
            or result.get("passes") is not True
            or not isinstance(result.get("bootstrap_one_sided_95_ucb"), (int, float))
            or not math.isfinite(float(result["bootstrap_one_sided_95_ucb"]))
            or float(result["bootstrap_one_sided_95_ucb"]) > 0.0
        ):
            raise AdmissionError(f"policy-target quality {name} gate did not pass")
    return {
        "path": str(receipt_path),
        "file_sha256": _file_sha256(receipt_path),
        "receipt_sha256": str(stated),
        "metrics": dict(metrics),
        "identity_sha256": _value_sha256(expected),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--composite-build-receipt", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from tools import a1_scratch_train as scratch

        verified, meta = scratch._verify_base_inputs(  # noqa: SLF001
            lock_path=args.lock,
            data_path=args.data,
            composite_build_receipt=args.composite_build_receipt,
        )
        payload = generate_receipt(
            verified=verified, composite_meta=meta, output_path=args.receipt
        )
    except (AdmissionError, OSError, SystemExit, ValueError) as error:
        print(f"a1_policy_target_quality_admission: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
