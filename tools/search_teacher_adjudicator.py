#!/usr/bin/env python3
"""Fail-closed S1/S2/S3 search-teacher adjudication.

This tool does not run matches or search probes.  It consumes immutable JSON
artifacts produced by the bounded B200 experiments, verifies their declared
input bytes and operator parity, and emits the typed stage decisions consumed
by the A1 pre-wave handoff.

The adjudication order is strict:

* S1: completed 85-pair post-D6 c-scale/D1 grid; select an H1 winner or the
  locked c_scale=.03/D1-off fallback.  The checkpoint-specific opening RMSE
  artifact must bind sigma_eval=.98 even when D1 is not selected.
* S2: n128 may replace n64 only after a 200-pair +15 Elo H1 and attributable
  fixed-root cost below 1.6x.  A cost below 1.8x is allowed only when the H1
  sample mean also clears +30 Elo (the predeclared ``clear margin`` rule).
* S3: adaptive n256 is legal only at >=40-action roots with always-full
  production semantics.  It needs either a confirmed H1 or >=15% lower
  repeated-root cross-seed JS with non-worse selected-action top-1 agreement,
  and <=20% whole-game search overhead.  Global n256 is always rejected.

Every output is created read-only with O_EXCL.  No decision can be made from a
mutable path, a partial screen, combined H2H wall time, an unspecified
``stability`` scalar, or a config that differs outside the tested dose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catan_zero.rl.pipeline_configs import CONFIG_SCHEMA_VERSION  # noqa: E402


MANIFEST_SCHEMA = "rl-rnd-search-stage-adjudication-v1"
DECISION_SCHEMA = "rl-rnd-stage-decision-v1"
FIXED_ROOT_SCHEMA = "fixed-root-search-stability-v2"
CALIBRATION_SCHEMA = "phase-sliced-value-calibration-v2"
# Schema 4 evidence was sealed before GenerateConfig gained checkpoint-byte
# provenance. Its typed payload is still self-hashed and semantically checked,
# so adjudication remains able to replay that immutable historical evidence.
SUPPORTED_PIPELINE_CONFIG_SCHEMAS = {4, CONFIG_SCHEMA_VERSION}

S1_ARMS = {
    "D1": (0.03, True),
    "cv50_cs0.1": (0.1, False),
    "cv50_cs0.1+D1": (0.1, True),
    "cv50_cs0.3": (0.3, False),
    "cv50_cs0.3+D1": (0.3, True),
}
S1_KEYS = {
    "c_scale",
    "symmetry_averaged_eval",
    "symmetry_averaged_eval_threshold",
    "rescale_noise_floor_c",
    "sigma_eval",
}
S2_KEYS = {"n_full", "n_fast", "p_full"}
S3_KEYS = {"n_full_wide", "n_full_wide_threshold", "wide_roots_always_full"}

SEARCH_OPERATOR_KEYS = {
    "max_depth",
    "c_visit",
    "c_scale",
    "prior_temperature",
    "n_full",
    "n_fast",
    "p_full",
    "n_full_wide",
    "n_full_wide_threshold",
    "wide_roots_always_full",
    "raw_policy_above_width",
    "symmetry_averaged_eval",
    "symmetry_averaged_eval_threshold",
    "wide_candidates_threshold",
    "correct_rust_chance_spectra",
    "lazy_interior_chance",
    "exact_budget_sh",
    "exact_budget_sh_min_n",
    "belief_chance_spectra",
    "rescale_noise_floor_c",
    "sigma_eval",
}
EVALUATOR_KEYS = {
    "value_scale",
    "prior_temperature",
    "context_fill",
    "cache_size",
    "value_squash",
    "value_readout",
    "public_observation",
    "rust_featurize",
    "emit_uncertainty",
}

POLICY = {
    "schema_version": "rl-rnd-search-adjudication-policy-v1",
    "d6_min_legal_actions": 20,
    "adaptive_n256_min_legal_actions": 40,
    "s1_pairs_per_arm": 85,
    "s1_sigma_eval": 0.98,
    "h1_elo1": 15.0,
    "h1_elo0": -10.0,
    "confirmation_pairs": 200,
    "screen_pairs": 50,
    "s2_cost_ratio_exclusive": 1.6,
    "s2_extended_cost_ratio_exclusive": 1.8,
    "s2_clear_margin_elo": 30.0,
    "s3_min_relative_js_reduction": 0.15,
    "s3_min_top1_agreement_delta": 0.0,
    "s3_max_whole_game_overhead_ratio": 1.20,
    "fixed_root_min_roots": 40,
    "fixed_root_min_repeats": 4,
}


class AdjudicationError(ValueError):
    """Raised whenever an input cannot bind a production decision."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AdjudicationError(f"value is not canonical JSON: {error}") from error


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise AdjudicationError(f"cannot hash {path}: {error}") from error
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdjudicationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AdjudicationError(f"{path} must contain a JSON object")
    return payload


def _absolute_path(raw: Any, *, base: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise AdjudicationError("artifact path must be a non-empty string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.absolute()


def _require_exact_keys(
    value: Any, expected: set[str], *, where: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdjudicationError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise AdjudicationError(
            f"{where} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdjudicationError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AdjudicationError(f"{where} must be a finite number")
    return result


def _integer(value: Any, *, where: str) -> int:
    number = _number(value, where=where)
    if not number.is_integer():
        raise AdjudicationError(f"{where} must be an integer")
    return int(number)


def _close(left: Any, right: Any, *, where: str, tolerance: float = 1.0e-9) -> None:
    if not math.isclose(
        _number(left, where=where),
        _number(right, where=where),
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise AdjudicationError(f"{where} mismatch: {left!r} != {right!r}")


def _validate_ref(raw: Any, *, base: Path, where: str) -> tuple[Path, dict[str, str]]:
    ref = _require_exact_keys(raw, {"path", "sha256"}, where=where)
    path = _absolute_path(ref["path"], base=base)
    declared = str(ref["sha256"])
    if not declared.startswith("sha256:") or len(declared) != 71:
        raise AdjudicationError(f"{where}.sha256 must be a full sha256: digest")
    actual = _sha256(path) if path.is_file() else "<missing>"
    if declared != actual:
        raise AdjudicationError(
            f"{where} artifact drift at {path}: declared {declared}, actual {actual}"
        )
    return path, {"path": str(path), "sha256": actual}


def _dedupe_records(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    by_path: dict[str, dict[str, str]] = {}
    for record in records:
        prior = by_path.get(record["path"])
        if prior is not None and prior != record:
            raise AdjudicationError(f"conflicting hashes declared for {record['path']}")
        by_path[record["path"]] = record
    return [by_path[path] for path in sorted(by_path)]


def _validate_search_operator(raw: Any, *, where: str) -> dict[str, Any]:
    operator = dict(_require_exact_keys(raw, SEARCH_OPERATOR_KEYS, where=where))
    for key in (
        "max_depth",
        "n_full",
        "n_fast",
        "symmetry_averaged_eval_threshold",
        "wide_candidates_threshold",
        "exact_budget_sh_min_n",
    ):
        operator[key] = _integer(operator[key], where=f"{where}.{key}")
    for key in (
        "c_visit",
        "c_scale",
        "prior_temperature",
        "p_full",
        "rescale_noise_floor_c",
        "sigma_eval",
    ):
        operator[key] = _number(operator[key], where=f"{where}.{key}")
    for key in (
        "symmetry_averaged_eval",
        "wide_roots_always_full",
        "correct_rust_chance_spectra",
        "lazy_interior_chance",
        "exact_budget_sh",
        "belief_chance_spectra",
    ):
        if not isinstance(operator[key], bool):
            raise AdjudicationError(f"{where}.{key} must be boolean")
    for key in ("n_full_wide", "n_full_wide_threshold", "raw_policy_above_width"):
        if operator[key] is not None:
            operator[key] = _integer(operator[key], where=f"{where}.{key}")

    if operator["max_depth"] <= 0 or operator["n_fast"] <= 0:
        raise AdjudicationError(f"{where} search depths/budgets must be positive")
    if operator["n_full"] not in {64, 128}:
        raise AdjudicationError(
            f"{where}.n_full must be 64 or 128; global n256 is forbidden"
        )
    if not 0.0 < operator["p_full"] <= 1.0:
        raise AdjudicationError(f"{where}.p_full must be in (0,1]")
    if operator["c_scale"] <= 0.0 or operator["sigma_eval"] <= 0.0:
        raise AdjudicationError(f"{where} c_scale and sigma_eval must be positive")
    if operator["symmetry_averaged_eval"] is not True:
        raise AdjudicationError(f"{where} must enable D6 symmetry averaging")
    if operator["symmetry_averaged_eval_threshold"] != POLICY["d6_min_legal_actions"]:
        raise AdjudicationError(
            f"{where} D6 threshold must be the independent inclusive >=20 gate"
        )
    if operator["n_full_wide"] is None:
        if (
            operator["n_full_wide_threshold"] is not None
            or operator["wide_roots_always_full"]
        ):
            raise AdjudicationError(
                f"{where} disabled adaptive budget requires null threshold and always_full=false"
            )
    else:
        if operator["n_full_wide"] != 256:
            raise AdjudicationError(f"{where} only permits adaptive n_full_wide=256")
        if (
            operator["n_full_wide_threshold"] is None
            or operator["n_full_wide_threshold"]
            < POLICY["adaptive_n256_min_legal_actions"]
        ):
            raise AdjudicationError(f"{where} adaptive n256 requires threshold >=40")
        if operator["wide_roots_always_full"] is not True:
            raise AdjudicationError(
                f"{where} adaptive n256 requires always-full wide roots"
            )
    return operator


def _validate_evaluator(raw: Any, *, where: str) -> dict[str, Any]:
    evaluator = dict(_require_exact_keys(raw, EVALUATOR_KEYS, where=where))
    if evaluator["public_observation"] is not True:
        raise AdjudicationError(f"{where}.public_observation must be true")
    if evaluator["value_readout"] not in {"scalar", "categorical"}:
        raise AdjudicationError(f"{where}.value_readout must be scalar or categorical")
    if _integer(evaluator["cache_size"], where=f"{where}.cache_size") != 0:
        raise AdjudicationError(f"{where}.cache_size must be 0 for attributable probes")
    return evaluator


def _operator_selected(operator: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: operator[key] for key in sorted(keys)}


def _apply_selected(
    operator: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    result = dict(operator)
    result.update(selected)
    return result


def _resolve_checkpoint_path(raw: Any, *, artifact_path: Path) -> Path:
    if isinstance(raw, dict):
        raw = raw.get("path")
    return _absolute_path(raw, base=artifact_path.parent)


def _validate_checkpoint_reference(
    artifact_checkpoint: Any, *, artifact_path: Path, checkpoint_path: Path, where: str
) -> None:
    actual = _resolve_checkpoint_path(artifact_checkpoint, artifact_path=artifact_path)
    if actual != checkpoint_path:
        raise AdjudicationError(
            f"{where} checkpoint mismatch: {actual} != locked {checkpoint_path}"
        )


def _verify_envelope_sources(envelope: dict[str, Any], *, path: Path) -> None:
    raw = envelope.get("source_artifacts")
    if not isinstance(raw, list) or not raw:
        raise AdjudicationError(f"predecessor {path} binds no source artifacts")
    for index, item in enumerate(raw):
        _validate_ref(
            item, base=path.parent, where=f"predecessor.source_artifacts[{index}]"
        )


def _load_predecessors(
    raw: Any, *, manifest_path: Path, stage: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(raw, list):
        raise AdjudicationError("predecessors must be a list")
    expected = {"s1": set(), "s2": {"s1"}, "s3": {"s1", "s2"}}[stage]
    loaded: dict[str, dict[str, Any]] = {}
    records: list[dict[str, str]] = []
    for index, ref in enumerate(raw):
        path, record = _validate_ref(
            ref, base=manifest_path.parent, where=f"predecessors[{index}]"
        )
        payload = _load_json(path)
        if payload.get("schema_version") != DECISION_SCHEMA:
            raise AdjudicationError(f"predecessor {path} has the wrong decision schema")
        predecessor_stage = str(payload.get("stage"))
        if predecessor_stage in loaded:
            raise AdjudicationError(f"duplicate {predecessor_stage} predecessor")
        if payload.get("passed") is not True or payload.get("decision") not in {
            "adopt",
            "hold",
        }:
            raise AdjudicationError(f"predecessor {path} is not a completed decision")
        selected = payload.get("selected_fields")
        expected_keys = {"s1": S1_KEYS, "s2": S2_KEYS}.get(predecessor_stage)
        if (
            expected_keys is None
            or not isinstance(selected, dict)
            or set(selected) != expected_keys
        ):
            raise AdjudicationError(
                f"predecessor {path} selected_fields shape is invalid"
            )
        if payload.get("selected_fields_sha256") != _digest_value(selected):
            raise AdjudicationError(f"predecessor {path} selected_fields hash mismatch")
        _verify_envelope_sources(payload, path=path)
        loaded[predecessor_stage] = payload
        records.append(record)
    if set(loaded) != expected:
        raise AdjudicationError(
            f"{stage.upper()} predecessors must be exactly {sorted(expected)}, got {sorted(loaded)}"
        )
    return loaded, records


def _validate_lineage(
    stage: str, base: dict[str, Any], predecessors: dict[str, dict[str, Any]]
) -> None:
    keys = (S1_KEYS,) if stage == "s2" else (S1_KEYS, S2_KEYS) if stage == "s3" else ()
    names = ("s1",) if stage == "s2" else ("s1", "s2") if stage == "s3" else ()
    for predecessor_stage, selected_keys in zip(names, keys):
        expected = _operator_selected(base, selected_keys)
        actual = predecessors[predecessor_stage]["selected_fields"]
        if actual != expected:
            raise AdjudicationError(
                f"{stage.upper()} base operator does not inherit {predecessor_stage.upper()} "
                f"selection: expected {expected}, got {actual}"
            )


def _validate_pentanomial(
    report: dict[str, Any], *, where: str, exact_pairs: int | None = None
) -> dict[str, Any]:
    pent = report.get("pentanomial_sprt")
    if not isinstance(pent, dict):
        raise AdjudicationError(f"{where}.pentanomial_sprt is required")
    if pent.get("model") != "pentanomial":
        raise AdjudicationError(f"{where} must use the pentanomial GSPRT")
    _close(pent.get("elo0"), POLICY["h1_elo0"], where=f"{where}.elo0")
    _close(pent.get("elo1"), POLICY["h1_elo1"], where=f"{where}.elo1")
    pairs = _integer(pent.get("pairs"), where=f"{where}.pairs")
    complete = _integer(report.get("complete_pairs"), where=f"{where}.complete_pairs")
    if pairs != complete:
        raise AdjudicationError(f"{where} pentanomial pair count != complete_pairs")
    if exact_pairs is not None and pairs != exact_pairs:
        raise AdjudicationError(
            f"{where} must contain exactly {exact_pairs} complete pairs"
        )
    if pent.get("decision") not in {"H0", "H1", "continue"}:
        raise AdjudicationError(f"{where} has invalid pentanomial decision")
    for field in ("mean_pair_score", "llr", "lower_bound", "upper_bound"):
        _number(pent.get(field), where=f"{where}.{field}")
    counts = [
        _integer(pent.get("ll_pairs"), where=f"{where}.ll_pairs"),
        _integer(pent.get("split_pairs"), where=f"{where}.split_pairs"),
        _integer(pent.get("ww_pairs"), where=f"{where}.ww_pairs"),
    ]
    if any(count < 0 for count in counts) or sum(counts) != pairs:
        raise AdjudicationError(f"{where} has inconsistent pentanomial counts")
    observed_mean = (0.5 * counts[1] + counts[2]) / pairs if pairs else None
    _close(
        pent.get("mean_pair_score"),
        observed_mean,
        where=f"{where}.mean_pair_score",
    )
    elo0 = float(pent["elo0"])
    elo1 = float(pent["elo1"])
    s0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    s1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
    regularized = [counts[0] + 1.0, counts[1], counts[2] + 1.0]
    total = sum(regularized)
    mean = (
        sum(count * value for count, value in zip(regularized, (0.0, 0.5, 1.0))) / total
    )
    variance = (
        sum(
            count * (value - mean) ** 2
            for count, value in zip(regularized, (0.0, 0.5, 1.0))
        )
        / total
    )
    expected_llr = total / (2.0 * variance) * (s1 - s0) * (2.0 * mean - s0 - s1)
    _close(pent.get("llr"), expected_llr, where=f"{where}.llr")
    alpha = _number(pent.get("alpha"), where=f"{where}.alpha")
    beta = _number(pent.get("beta"), where=f"{where}.beta")
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise AdjudicationError(f"{where} alpha/beta must be in (0,1)")
    expected_lower = math.log(beta / (1.0 - alpha))
    expected_upper = math.log((1.0 - beta) / alpha)
    _close(pent.get("lower_bound"), expected_lower, where=f"{where}.lower_bound")
    _close(pent.get("upper_bound"), expected_upper, where=f"{where}.upper_bound")
    expected_decision = (
        "H1"
        if expected_llr >= expected_upper
        else "H0"
        if expected_llr <= expected_lower
        else "continue"
    )
    if pent.get("decision") != expected_decision:
        raise AdjudicationError(
            f"{where} decision is inconsistent with its counts and bounds"
        )
    return pent


def _validate_complete_h2h(
    report: dict[str, Any],
    *,
    path: Path,
    where: str,
    checkpoint_path: Path,
    base: dict[str, Any],
    candidate: dict[str, Any],
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    _validate_checkpoint_reference(
        report.get("candidate_checkpoint"),
        artifact_path=path,
        checkpoint_path=checkpoint_path,
        where=f"{where}.candidate",
    )
    _validate_checkpoint_reference(
        report.get("baseline_checkpoint"),
        artifact_path=path,
        checkpoint_path=checkpoint_path,
        where=f"{where}.baseline",
    )
    if report.get("public_observation") is not True:
        raise AdjudicationError(f"{where} public_observation must be true")
    if report.get("symmetry_averaged_eval") is not True:
        raise AdjudicationError(f"{where} must enable D6 on both roles")
    if (
        _integer(
            report.get("symmetry_averaged_eval_threshold"),
            where=f"{where}.symmetry_averaged_eval_threshold",
        )
        != POLICY["d6_min_legal_actions"]
    ):
        raise AdjudicationError(f"{where} D6 threshold must be inclusive >=20")
    if report.get("errors") not in ([], None):
        raise AdjudicationError(f"{where} contains worker errors")
    if (
        _integer(report.get("games_truncated", 0), where=f"{where}.games_truncated")
        != 0
    ):
        raise AdjudicationError(f"{where} contains truncated games")
    requested = _integer(
        report.get("pairs_requested"), where=f"{where}.pairs_requested"
    )
    if requested not in {POLICY["screen_pairs"], POLICY["confirmation_pairs"]}:
        raise AdjudicationError(f"{where} must be a 50- or 200-pair protocol")
    pent = _validate_pentanomial(report, where=where)
    if int(pent["pairs"]) != requested:
        raise AdjudicationError(f"{where} is a partial H2H artifact")
    if (
        _integer(report.get("games_played"), where=f"{where}.games_played")
        != 2 * requested
    ):
        raise AdjudicationError(
            f"{where} does not contain both color-swapped games per pair"
        )
    typed = _require_exact_keys(
        report.get("typed_config"),
        {"pipeline", "schema_version", "fields"},
        where=f"{where}.typed_config",
    )
    if (
        typed["pipeline"] != "eval"
        or typed["schema_version"] not in SUPPORTED_PIPELINE_CONFIG_SCHEMAS
        or not isinstance(typed["fields"], dict)
    ):
        raise AdjudicationError(f"{where} typed config is not an eval config")
    full_hash = _digest_value(typed)
    if report.get("full_config_hash") != full_hash:
        raise AdjudicationError(f"{where} full_config_hash is not reproducible")
    if report.get("config_hash") != "sha256:" + full_hash.removeprefix("sha256:")[:16]:
        raise AdjudicationError(f"{where} short config_hash is not reproducible")
    fields = typed["fields"]
    base_seed = _integer(fields.get("base_seed"), where=f"{where}.base_seed")
    expected_fields = {
        "mode": "cross_net",
        "public_observation": True,
        "belief_chance_spectra": base["belief_chance_spectra"],
        "pairs": requested,
        "n_full": base["n_full"],
        "candidate_n_full": candidate["n_full"],
        "baseline_n_full": base["n_full"],
        "n_full_wide": base["n_full_wide"],
        "candidate_n_full_wide": candidate["n_full_wide"],
        "baseline_n_full_wide": base["n_full_wide"],
        "n_full_wide_threshold": base["n_full_wide_threshold"],
        "candidate_n_full_wide_threshold": candidate["n_full_wide_threshold"],
        "baseline_n_full_wide_threshold": base["n_full_wide_threshold"],
        "raw_policy_above_width": base["raw_policy_above_width"],
        "max_depth": base["max_depth"],
        "max_decisions": 600,
        "c_visit": base["c_visit"],
        "c_scale": base["c_scale"],
        "rescale_noise_floor_c": base["rescale_noise_floor_c"],
        "sigma_eval": base["sigma_eval"],
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
        "wide_candidates_threshold": base["wide_candidates_threshold"],
        "symmetry_averaged_eval": base["symmetry_averaged_eval"],
        "symmetry_averaged_eval_threshold": base[
            "symmetry_averaged_eval_threshold"
        ],
        "correct_rust_chance_spectra": base["correct_rust_chance_spectra"],
        "lazy_interior_chance": base["lazy_interior_chance"],
        "prior_temperature": evaluator["prior_temperature"],
        "value_scale": evaluator["value_scale"],
        "value_squash": evaluator["value_squash"],
        "value_readout": evaluator["value_readout"],
        "candidate_value_readout": evaluator["value_readout"],
        "baseline_value_readout": evaluator["value_readout"],
        "elo0": POLICY["h1_elo0"],
        "elo1": POLICY["h1_elo1"],
    }
    for key, expected in expected_fields.items():
        if fields.get(key) != expected:
            raise AdjudicationError(
                f"{where} typed config drift at {key}: {fields.get(key)!r} != {expected!r}"
            )
    for role_key in ("candidate", "baseline"):
        _validate_checkpoint_reference(
            fields.get(role_key),
            artifact_path=path,
            checkpoint_path=checkpoint_path,
            where=f"{where}.typed_config.{role_key}",
        )

    games = report.get("games")
    if not isinstance(games, list) or len(games) != 2 * requested:
        raise AdjudicationError(f"{where} raw games do not cover every paired game")
    by_pair: dict[int, list[dict[str, Any]]] = {}
    seen_game_seeds: set[int] = set()
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise AdjudicationError(f"{where}.games[{index}] must be an object")
        pair_id = _integer(game.get("pair_id"), where=f"{where}.games[{index}].pair_id")
        if type(game.get("candidate_won")) is not bool:
            raise AdjudicationError(f"{where}.games[{index}] lacks a decisive result")
        if game.get("terminated") is not True or game.get("truncated") is not False:
            raise AdjudicationError(f"{where}.games[{index}] is incomplete")
        by_pair.setdefault(pair_id, []).append(game)
    if set(by_pair) != set(range(requested)):
        raise AdjudicationError(f"{where} pair ids are not exactly 0..{requested - 1}")
    raw_counts = [0, 0, 0]
    for pair_id, pair_games in sorted(by_pair.items()):
        if len(pair_games) != 2:
            raise AdjudicationError(f"{where} pair {pair_id} is not color swapped")
        orientations = {str(game.get("orientation")) for game in pair_games}
        if orientations != {"candidate_red", "candidate_blue"}:
            raise AdjudicationError(f"{where} pair {pair_id} has invalid orientations")
        seeds = {
            _integer(game.get("game_seed"), where=f"{where}.pair[{pair_id}].game_seed")
            for game in pair_games
        }
        if len(seeds) != 1 or next(iter(seeds)) in seen_game_seeds:
            raise AdjudicationError(f"{where} pair {pair_id} has duplicate/drifted seed")
        if next(iter(seeds)) != base_seed + pair_id:
            raise AdjudicationError(f"{where} pair {pair_id} seed is outside the typed plan")
        seen_game_seeds.update(seeds)
        wins = sum(bool(game["candidate_won"]) for game in pair_games)
        raw_counts[wins] += 1
    pent_counts = [int(pent["ll_pairs"]), int(pent["split_pairs"]), int(pent["ww_pairs"])]
    if raw_counts != pent_counts:
        raise AdjudicationError(
            f"{where} pentanomial counts do not reconstruct from raw games"
        )
    return pent


def _effective_config_value(
    config: dict[str, Any], key: str, default: Any = None
) -> Any:
    return config[key] if key in config else default


def _validate_h2h_shared_operator(
    report: dict[str, Any], operator: dict[str, Any], *, where: str
) -> None:
    expected = {
        "c_visit": operator["c_visit"],
        "c_scale": operator["c_scale"],
        "rescale_noise_floor_c": operator["rescale_noise_floor_c"],
        "sigma_eval": operator["sigma_eval"],
    }
    for key, value in expected.items():
        _close(report.get(key), value, where=f"{where}.{key}")


def _validate_locked_fixed_inputs(report: dict[str, Any], *, where: str) -> None:
    locked = report.get("locked_input_file_hashes")
    if not isinstance(locked, dict) or not locked:
        raise AdjudicationError(f"{where} binds no fixed-root input files")
    for raw_path, declared in locked.items():
        path = Path(str(raw_path)).expanduser().absolute()
        actual = _sha256(path) if path.is_file() else "<missing>"
        if actual != declared:
            raise AdjudicationError(
                f"{where} fixed-root input drift at {path}: {declared} != {actual}"
            )


def _validate_fixed_root_report(
    report: dict[str, Any],
    *,
    path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    base: dict[str, Any],
    candidate: dict[str, Any],
    evaluator: dict[str, Any],
    baseline_role: str,
    candidate_role: str,
    stage: str,
) -> dict[str, Any]:
    where = f"{stage.upper()} fixed-root report"
    if report.get("schema_version") != FIXED_ROOT_SCHEMA:
        raise AdjudicationError(f"{where} schema must be {FIXED_ROOT_SCHEMA}")
    content = dict(report)
    declared_content_hash = content.pop("report_content_sha256", None)
    if declared_content_hash != _digest_value(content):
        raise AdjudicationError(f"{where} report_content_sha256 mismatch")
    checkpoint = _require_exact_keys(
        report.get("checkpoint"), {"path", "sha256"}, where=f"{where}.checkpoint"
    )
    _validate_checkpoint_reference(
        checkpoint, artifact_path=path, checkpoint_path=checkpoint_path, where=where
    )
    if checkpoint["sha256"] != checkpoint_sha256:
        raise AdjudicationError(f"{where} checkpoint byte hash mismatch")
    _validate_locked_fixed_inputs(report, where=where)
    protocol = report.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("force_full") is not True:
        raise AdjudicationError(f"{where} must force full search")
    if (
        _integer(protocol.get("repeats_per_root_per_role"), where=f"{where}.repeats")
        < POLICY["fixed_root_min_repeats"]
    ):
        raise AdjudicationError(f"{where} has too few independent repeats")
    if protocol.get("wide_slice") != "legal_width>=40":
        raise AdjudicationError(f"{where} wide slice is not the inclusive >=40 slice")
    root_panel = report.get("root_panel")
    if (
        not isinstance(root_panel, dict)
        or _integer(root_panel.get("root_count"), where=f"{where}.root_count")
        < POLICY["fixed_root_min_roots"]
    ):
        raise AdjudicationError(f"{where} has fewer than 40 roots")

    roles = report.get("roles")
    if not isinstance(roles, dict) or set(roles) != {baseline_role, candidate_role}:
        raise AdjudicationError(f"{where} role names do not match the manifest")
    base_config = roles[baseline_role].get("effective_search_config")
    candidate_config = roles[candidate_role].get("effective_search_config")
    if not isinstance(base_config, dict) or not isinstance(candidate_config, dict):
        raise AdjudicationError(f"{where} lacks effective search configs")
    for role, config, operator in (
        (baseline_role, base_config, base),
        (candidate_role, candidate_config, candidate),
    ):
        missing_operator_keys = SEARCH_OPERATOR_KEYS - set(config)
        if missing_operator_keys:
            raise AdjudicationError(
                f"{where} {role} omits operator fields {sorted(missing_operator_keys)}"
            )
        for key in SEARCH_OPERATOR_KEYS:
            if config[key] != operator[key]:
                raise AdjudicationError(
                    f"{where} {role} effective config drift at {key}"
                )
    common_keys = (
        "n_full",
        "n_fast",
        "c_visit",
        "c_scale",
        "rescale_noise_floor_c",
        "sigma_eval",
        "symmetry_averaged_eval",
        "symmetry_averaged_eval_threshold",
    )
    for key in common_keys:
        if (
            base_config.get(key) != base[key]
            or candidate_config.get(key) != candidate[key]
        ):
            raise AdjudicationError(f"{where} effective config mismatch at {key}")
    if stage == "s2":
        expected_differences = {"n_full"}
    else:
        expected_differences = set(S3_KEYS)
        for config, operator, role in (
            (base_config, base, baseline_role),
            (candidate_config, candidate, candidate_role),
        ):
            if (
                config.get("n_full_wide") != operator["n_full_wide"]
                or config.get("n_full_wide_threshold")
                != operator["n_full_wide_threshold"]
            ):
                raise AdjudicationError(f"{where} {role} adaptive budget mismatch")
    if set(report.get("search_config_differences", {})) != expected_differences:
        raise AdjudicationError(
            f"{where} comparison differs outside the exact {sorted(expected_differences)} dose"
        )
    if set(report.get("allowed_search_config_differences", [])) != expected_differences:
        raise AdjudicationError(f"{where} allowed-differences declaration is not exact")

    evaluator_spec = report.get("evaluator", {}).get("effective_evaluator_config")
    if not isinstance(evaluator_spec, dict):
        raise AdjudicationError(f"{where} lacks effective evaluator config")
    missing_evaluator_keys = EVALUATOR_KEYS - set(evaluator_spec)
    if missing_evaluator_keys:
        raise AdjudicationError(
            f"{where} evaluator omits fields {sorted(missing_evaluator_keys)}"
        )
    for key in EVALUATOR_KEYS:
        if evaluator_spec.get(key) != evaluator[key]:
            raise AdjudicationError(f"{where} evaluator mismatch at {key}")
    if evaluator_spec.get("cache_size") != 0:
        raise AdjudicationError(f"{where} evaluator cache must be zero")
    per_root = report.get("per_root")
    if not isinstance(per_root, list) or len(per_root) != int(root_panel["root_count"]):
        raise AdjudicationError(f"{where} per_root evidence is incomplete")
    seed_manifests = report.get("search_seed_manifests")
    if not isinstance(seed_manifests, dict) or set(seed_manifests) != {
        baseline_role,
        candidate_role,
    }:
        raise AdjudicationError(f"{where} search seed manifests are missing")
    role_seed_sets: dict[str, set[int]] = {}
    for role in (baseline_role, candidate_role):
        seed_manifest = seed_manifests[role]
        if not isinstance(seed_manifest, dict) or set(seed_manifest) != {
            "base_seed",
            "seeds_by_root",
            "seed_set_sha256",
        }:
            raise AdjudicationError(f"{where} {role} seed manifest shape drift")
        rows = seed_manifest["seeds_by_root"]
        if not isinstance(rows, list) or len(rows) != len(per_root):
            raise AdjudicationError(f"{where} {role} seed rows are incomplete")
        flat = [
            _integer(seed, where=f"{where}.{role}.search_seed")
            for row in rows
            if isinstance(row, list)
            for seed in row
        ]
        if any(not isinstance(row, list) or len(row) < POLICY["fixed_root_min_repeats"] for row in rows):
            raise AdjudicationError(f"{where} {role} seed repeats are incomplete")
        if len(flat) != len(set(flat)):
            raise AdjudicationError(f"{where} {role} repeats reuse search seeds")
        if seed_manifest["seed_set_sha256"] != _digest_value(sorted(flat)):
            raise AdjudicationError(f"{where} {role} seed-set digest mismatch")
        role_seed_sets[role] = set(flat)
    if role_seed_sets[baseline_role] & role_seed_sets[candidate_role]:
        raise AdjudicationError(f"{where} role search-seed sets overlap")
    try:
        from tools.fixed_root_search_stability import (
            aggregate_report_slices,
            summarize_cross_seed_runs,
        )

        for index, root in enumerate(per_root):
            if not isinstance(root, dict) or int(root.get("root_index", -1)) != index:
                raise AdjudicationError(f"{where} per_root ordering/identity drift")
            root_roles = root.get("roles")
            if not isinstance(root_roles, dict) or set(root_roles) != {
                baseline_role,
                candidate_role,
            }:
                raise AdjudicationError(f"{where} per-root role set drift")
            for role in (baseline_role, candidate_role):
                runs = root_roles[role].get("runs")
                if not isinstance(runs, list) or len(runs) < POLICY[
                    "fixed_root_min_repeats"
                ]:
                    raise AdjudicationError(f"{where} {role} repeats are incomplete")
                if [int(run.get("search_seed", -1)) for run in runs] != [
                    int(seed) for seed in seed_manifests[role]["seeds_by_root"][index]
                ]:
                    raise AdjudicationError(
                        f"{where} {role} raw runs do not match the seed manifest"
                    )
                recomputed_stability = summarize_cross_seed_runs(runs)
                if root_roles[role].get("stability") != recomputed_stability:
                    raise AdjudicationError(
                        f"{where} {role} stability does not reconstruct from runs"
                    )
        recomputed_slices = aggregate_report_slices(
            per_root, baseline_role, candidate_role
        )
    except AdjudicationError:
        raise
    except Exception as error:  # noqa: BLE001 - malformed raw evidence fails closed.
        raise AdjudicationError(
            f"{where} raw per-root evidence cannot be recomputed: {error}"
        ) from error
    if report.get("slices") != recomputed_slices:
        raise AdjudicationError(f"{where} slices do not reconstruct from per_root")
    return recomputed_slices


def _validate_s1_arm(
    report: dict[str, Any],
    *,
    path: Path,
    checkpoint_path: Path,
    arm_name: str,
    base: dict[str, Any],
) -> dict[str, Any]:
    where = f"S1 arm {arm_name}"
    if report.get("arm") != arm_name:
        raise AdjudicationError(f"{where} artifact arm name mismatch")
    _validate_checkpoint_reference(
        report.get("checkpoint"),
        artifact_path=path,
        checkpoint_path=checkpoint_path,
        where=where,
    )
    if report.get("masked") is not True:
        raise AdjudicationError(f"{where} must use public-observation masking")
    exact_report_values = {
        "max_depth": 80,
        "max_decisions": 600,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "prior_temperature": 1.0,
        "value_scale": 1.0,
        "value_squash": "tanh",
        "value_readout": "scalar",
        "d1_c": 1.0,
        "d1_sigma_eval": POLICY["s1_sigma_eval"],
    }
    for field, expected in exact_report_values.items():
        if report.get(field) != expected:
            raise AdjudicationError(
                f"{where}.{field} must be {expected!r}, got {report.get(field)!r}"
            )
    if report.get("symmetry_averaged_eval") is not True:
        raise AdjudicationError(f"{where} must enable D6 on both sides")
    baseline = report.get("baseline_search_config")
    candidate = report.get("candidate_search_config")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise AdjudicationError(f"{where} lacks effective baseline/candidate configs")
    for label, config in (("baseline", baseline), ("candidate", candidate)):
        if (
            config.get("symmetry_averaged_eval_threshold")
            != POLICY["d6_min_legal_actions"]
        ):
            raise AdjudicationError(
                f"{where} {label} does not attest the independent inclusive >=20 D6 gate"
            )
    expected_baseline = {
        "colors": ["RED", "BLUE"],
        "n_full": 64,
        "n_fast": 64,
        "p_full": 1.0,
        "max_depth": 80,
        "temperature": 0.0,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "c_visit": base["c_visit"],
        "c_scale": 0.03,
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": POLICY["d6_min_legal_actions"],
    }
    expected_candidate = dict(expected_baseline)
    expected_candidate["c_scale"] = S1_ARMS[arm_name][0]
    if S1_ARMS[arm_name][1]:
        expected_candidate.update(
            {
                "rescale_noise_floor_c": 1.0,
                "sigma_eval": POLICY["s1_sigma_eval"],
            }
        )
    if baseline != expected_baseline or candidate != expected_candidate:
        raise AdjudicationError(
            f"{where} effective configs differ outside the predeclared arm dose"
        )
    expected_overrides = {
        "c_visit": base["c_visit"],
        "c_scale": S1_ARMS[arm_name][0],
    }
    if S1_ARMS[arm_name][1]:
        expected_overrides.update(
            {
                "rescale_noise_floor_c": 1.0,
                "sigma_eval": POLICY["s1_sigma_eval"],
            }
        )
    if report.get("arm_config_overrides") != expected_overrides:
        raise AdjudicationError(f"{where} arm override declaration drift")
    if report.get("arm_mcts_cls_key") != "stock":
        raise AdjudicationError(f"{where} must use the stock search implementation")
    # The threshold must be explicit.  Missing means the legacy >24 fallback,
    # which is not the plan's independent inclusive >=20 D6 operator.
    for label, config in (("baseline", baseline), ("candidate", candidate)):
        if config.get("symmetry_averaged_eval") is not True:
            raise AdjudicationError(f"{where} {label} D6 is disabled")
        if (
            config.get("symmetry_averaged_eval_threshold")
            != POLICY["d6_min_legal_actions"]
        ):
            raise AdjudicationError(
                f"{where} {label} does not attest the independent inclusive >=20 D6 gate"
            )
        if _integer(config.get("n_full"), where=f"{where}.{label}.n_full") != 64:
            raise AdjudicationError(f"{where} must calibrate the n64 operator")
        _close(config.get("c_visit"), base["c_visit"], where=f"{where}.{label}.c_visit")
        for field, expected in (
            ("max_depth", 80),
            ("temperature", 0.0),
            ("correct_rust_chance_spectra", True),
            ("lazy_interior_chance", True),
            ("max_root_candidates", 16),
            ("max_root_candidates_wide", 54),
        ):
            if config.get(field) != expected:
                raise AdjudicationError(
                    f"{where}.{label}.{field} must be {expected!r}"
                )
    _close(baseline.get("c_scale"), 0.03, where=f"{where}.baseline.c_scale")
    if (
        _number(
            _effective_config_value(baseline, "rescale_noise_floor_c", 0.0),
            where=f"{where}.baseline.rescale_noise_floor_c",
        )
        != 0.0
    ):
        raise AdjudicationError(f"{where} baseline must have D1 off")
    expected_scale, d1_on = S1_ARMS[arm_name]
    _close(candidate.get("c_scale"), expected_scale, where=f"{where}.candidate.c_scale")
    d1_c = _number(
        _effective_config_value(candidate, "rescale_noise_floor_c", 0.0),
        where=f"{where}.candidate.rescale_noise_floor_c",
    )
    if d1_on:
        if d1_c <= 0.0:
            raise AdjudicationError(f"{where} is named D1 but D1 is disabled")
        _close(
            candidate.get("sigma_eval"),
            POLICY["s1_sigma_eval"],
            where=f"{where}.candidate.sigma_eval",
        )
    elif d1_c != 0.0:
        raise AdjudicationError(f"{where} is an off arm but D1 is enabled")
    if report.get("errors") not in ([], None):
        raise AdjudicationError(f"{where} contains worker errors")
    if (
        _integer(report.get("games_truncated", 0), where=f"{where}.games_truncated")
        != 0
    ):
        raise AdjudicationError(f"{where} contains truncated games")
    if (
        _integer(report.get("pairs_requested"), where=f"{where}.pairs_requested")
        != POLICY["s1_pairs_per_arm"]
    ):
        raise AdjudicationError(f"{where} must request exactly 85 pairs")
    if _integer(report.get("games_played"), where=f"{where}.games_played") != 170:
        raise AdjudicationError(f"{where} must contain exactly 170 games")
    pent = _validate_pentanomial(
        report, where=where, exact_pairs=POLICY["s1_pairs_per_arm"]
    )
    games = report.get("games")
    if not isinstance(games, list) or len(games) != 170:
        raise AdjudicationError(f"{where} lacks all 170 raw paired games")
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise AdjudicationError(f"{where}.games[{index}] must be an object")
        pair_id = _integer(game.get("pair_id"), where=f"{where}.games[{index}].pair_id")
        if type(game.get("candidate_won")) is not bool:
            raise AdjudicationError(f"{where}.games[{index}] lacks decisive outcome")
        if game.get("terminated") is not True or game.get("truncated") is not False:
            raise AdjudicationError(f"{where}.games[{index}] is incomplete")
        by_pair.setdefault(pair_id, []).append(game)
    if set(by_pair) != set(range(POLICY["s1_pairs_per_arm"])):
        raise AdjudicationError(f"{where} raw pair ids are incomplete")
    raw_counts = [0, 0, 0]
    seen_seeds: set[int] = set()
    seed_block_base = _integer(
        report.get("seed_block_base"), where=f"{where}.seed_block_base"
    )
    for pair_id, pair_games in sorted(by_pair.items()):
        if len(pair_games) != 2 or {
            str(game.get("orientation")) for game in pair_games
        } != {"candidate_red", "candidate_blue"}:
            raise AdjudicationError(f"{where} pair {pair_id} is not color swapped")
        seeds = {int(game.get("game_seed")) for game in pair_games}
        if len(seeds) != 1 or next(iter(seeds)) in seen_seeds:
            raise AdjudicationError(f"{where} pair {pair_id} seed drift/duplication")
        if next(iter(seeds)) != seed_block_base + pair_id:
            raise AdjudicationError(f"{where} pair {pair_id} seed is outside its block")
        seen_seeds.update(seeds)
        raw_counts[sum(bool(game["candidate_won"]) for game in pair_games)] += 1
    if raw_counts != [
        int(pent["ll_pairs"]),
        int(pent["split_pairs"]),
        int(pent["ww_pairs"]),
    ]:
        raise AdjudicationError(f"{where} pentanomial counts do not reconstruct")
    return pent


def _adjudicate_s1(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    base: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    if manifest.get("candidate_search_operator") is not None:
        raise AdjudicationError(
            "S1 candidate_search_operator must be null; candidates are the five arms"
        )
    if base["c_scale"] != 0.03 or base["rescale_noise_floor_c"] != 0.0:
        raise AdjudicationError("S1 base must be c_scale=.03 with D1 off")
    if base["n_full"] != 64 or base["n_full_wide"] is not None:
        raise AdjudicationError(
            "S1 base must be the global n64 operator with no adaptive n256"
        )
    _close(base["sigma_eval"], POLICY["s1_sigma_eval"], where="S1 base sigma_eval")
    evidence = _require_exact_keys(
        manifest.get("evidence"), {"arms", "sigma_eval"}, where="S1 evidence"
    )
    sigma_path, sigma_record = _validate_ref(
        evidence["sigma_eval"],
        base=manifest_path.parent,
        where="S1 sigma_eval evidence",
    )
    calibration = _load_json(sigma_path)
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise AdjudicationError(
            f"S1 sigma artifact schema must be {CALIBRATION_SCHEMA}"
        )
    if calibration.get("value_readout") != "scalar":
        raise AdjudicationError(
            "S1 sigma artifact must calibrate the scalar gen3 readout"
        )
    _validate_checkpoint_reference(
        calibration.get("checkpoint"),
        artifact_path=sigma_path,
        checkpoint_path=checkpoint_path,
        where="S1 sigma artifact",
    )
    opening = calibration.get("by_phase", {}).get("opening_placement", {})
    _close(
        opening.get("value_rmse"),
        POLICY["s1_sigma_eval"],
        where="S1 opening_placement.value_rmse",
        tolerance=5.0e-3,
    )
    arms = evidence["arms"]
    if not isinstance(arms, list) or len(arms) != len(S1_ARMS):
        raise AdjudicationError("S1 evidence must contain exactly five arm artifacts")
    results: dict[str, dict[str, Any]] = {}
    records = [sigma_record]
    for index, raw_ref in enumerate(arms):
        path, record = _validate_ref(
            raw_ref, base=manifest_path.parent, where=f"S1 arms[{index}]"
        )
        report = _load_json(path)
        arm_name = str(report.get("arm"))
        if arm_name not in S1_ARMS or arm_name in results:
            raise AdjudicationError(
                f"S1 arm set contains invalid/duplicate arm {arm_name!r}"
            )
        pent = _validate_s1_arm(
            report,
            path=path,
            checkpoint_path=checkpoint_path,
            arm_name=arm_name,
            base=base,
        )
        results[arm_name] = {"report": report, "pentanomial": pent}
        records.append(record)
    if set(results) != set(S1_ARMS):
        raise AdjudicationError(f"S1 arm set must be exactly {sorted(S1_ARMS)}")

    h1 = [
        name
        for name, result in results.items()
        if result["pentanomial"]["decision"] == "H1"
    ]
    # Deterministic, evidence-derived ranking.  Exact ties are ambiguous and
    # fall back rather than selecting by filename or iteration order.
    ranked = sorted(
        h1,
        key=lambda name: (
            float(results[name]["pentanomial"]["mean_pair_score"]),
            float(results[name]["pentanomial"]["llr"]),
            name,
        ),
        reverse=True,
    )
    winner: str | None = ranked[0] if ranked else None
    if len(ranked) > 1:
        best = results[ranked[0]]["pentanomial"]
        second = results[ranked[1]]["pentanomial"]
        if (
            best["mean_pair_score"] == second["mean_pair_score"]
            and best["llr"] == second["llr"]
        ):
            winner = None
    if winner is None:
        selected_operator = dict(base)
        decision = "hold"
        reason = "no_unique_h1_winner_use_cs003_d1_off_fallback"
    else:
        report = results[winner]["report"]
        candidate = report["candidate_search_config"]
        selected_operator = dict(base)
        selected_operator["c_scale"] = float(S1_ARMS[winner][0])
        if S1_ARMS[winner][1]:
            selected_operator["rescale_noise_floor_c"] = float(
                candidate["rescale_noise_floor_c"]
            )
        else:
            selected_operator["rescale_noise_floor_c"] = 0.0
        selected_operator["sigma_eval"] = POLICY["s1_sigma_eval"]
        decision = "adopt"
        reason = f"h1_winner:{winner}"
    selected = _operator_selected(selected_operator, S1_KEYS)
    metrics = {
        "reason": reason,
        "winner": winner,
        "arms": {
            name: {
                "decision": results[name]["pentanomial"]["decision"],
                "mean_pair_score": results[name]["pentanomial"]["mean_pair_score"],
                "llr": results[name]["pentanomial"]["llr"],
                "pairs": results[name]["pentanomial"]["pairs"],
            }
            for name in sorted(results)
        },
        "sigma_eval_artifact": {
            "opening_value_rmse": opening["value_rmse"],
            "bound_sigma_eval": POLICY["s1_sigma_eval"],
        },
    }
    return decision, selected, metrics, records


def _validate_s2_candidate(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    differences = {key for key in SEARCH_OPERATOR_KEYS if base[key] != candidate[key]}
    if differences != {"n_full"}:
        raise AdjudicationError(
            f"S2 candidate may differ only in n_full, got {sorted(differences)}"
        )
    if base["n_full"] != 64 or candidate["n_full"] != 128:
        raise AdjudicationError("S2 must compare global n128 against global n64")
    if base["n_full_wide"] is not None or candidate["n_full_wide"] is not None:
        raise AdjudicationError("S2 cannot include an adaptive wide-root budget")


def _s2_clear_margin(pent: dict[str, Any]) -> bool:
    threshold = 1.0 / (1.0 + 10.0 ** (-POLICY["s2_clear_margin_elo"] / 400.0))
    return pent["decision"] == "H1" and float(pent["mean_pair_score"]) >= threshold


def _adjudicate_s2(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    base: dict[str, Any],
    candidate: dict[str, Any] | None,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    evaluator: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    if candidate is None:
        raise AdjudicationError("S2 candidate_search_operator is required")
    _validate_s2_candidate(base, candidate)
    evidence = _require_exact_keys(
        manifest.get("evidence"),
        {"h2h", "fixed_root", "baseline_role", "candidate_role"},
        where="S2 evidence",
    )
    h2h_path, h2h_record = _validate_ref(
        evidence["h2h"], base=manifest_path.parent, where="S2 H2H evidence"
    )
    fixed_path, fixed_record = _validate_ref(
        evidence["fixed_root"],
        base=manifest_path.parent,
        where="S2 fixed-root evidence",
    )
    h2h = _load_json(h2h_path)
    pent = _validate_complete_h2h(
        h2h,
        path=h2h_path,
        where="S2 H2H",
        checkpoint_path=checkpoint_path,
        base=base,
        candidate=candidate,
        evaluator=evaluator,
    )
    _validate_h2h_shared_operator(h2h, base, where="S2 H2H")
    if h2h.get("candidate_n_full") != 128 or h2h.get("baseline_n_full") != 64:
        raise AdjudicationError(
            "S2 H2H role budgets must be candidate=128, baseline=64"
        )
    if (
        h2h.get("candidate_n_full_wide") is not None
        or h2h.get("baseline_n_full_wide") is not None
    ):
        raise AdjudicationError("S2 H2H must not include adaptive wide-root budgets")
    fixed = _load_json(fixed_path)
    slices = _validate_fixed_root_report(
        fixed,
        path=fixed_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        base=base,
        candidate=candidate,
        evaluator=evaluator,
        baseline_role=str(evidence["baseline_role"]),
        candidate_role=str(evidence["candidate_role"]),
        stage="s2",
    )
    cost = _number(
        slices.get("global", {})
        .get("comparison", {})
        .get("role_b_over_role_a_wall_ratio"),
        where="S2 fixed-root attributable wall ratio",
    )
    h1 = pent["decision"] == "H1"
    confirmed = int(pent["pairs"]) >= POLICY["confirmation_pairs"]
    clear_margin = _s2_clear_margin(pent)
    cost_pass = cost < POLICY["s2_cost_ratio_exclusive"] or (
        cost < POLICY["s2_extended_cost_ratio_exclusive"] and clear_margin
    )
    positive = h1 and cost_pass
    if positive and not confirmed:
        raise AdjudicationError(
            "S2 positive 50-pair screen requires 200-pair confirmation"
        )
    adopt = positive and confirmed
    selected_operator = candidate if adopt else base
    metrics = {
        "reason": (
            "confirmed_h1_and_cost_pass"
            if adopt
            else "retain_n64_no_binding_strength_cost_win"
        ),
        "pentanomial_decision": pent["decision"],
        "pairs": pent["pairs"],
        "mean_pair_score": pent["mean_pair_score"],
        "clear_margin": clear_margin,
        "attributable_fixed_root_wall_ratio": cost,
        "cost_pass": cost_pass,
    }
    return (
        "adopt" if adopt else "hold",
        _operator_selected(selected_operator, S2_KEYS),
        metrics,
        [h2h_record, fixed_record],
    )


def _validate_s3_candidate(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    differences = {key for key in SEARCH_OPERATOR_KEYS if base[key] != candidate[key]}
    if differences != S3_KEYS:
        raise AdjudicationError(
            f"S3 candidate may differ only in adaptive-wide fields, got {sorted(differences)}"
        )
    if base["n_full"] != 128 or candidate["n_full"] != 128:
        raise AdjudicationError("S3 requires n128 on both sides")
    if base["n_full_wide"] is not None or base["wide_roots_always_full"]:
        raise AdjudicationError("S3 baseline must have adaptive wide search disabled")
    if candidate["n_full_wide"] != 256 or candidate["n_full_wide_threshold"] < 40:
        raise AdjudicationError("S3 candidate must be adaptive n256 only at >=40 roots")
    if candidate["wide_roots_always_full"] is not True:
        raise AdjudicationError("S3 candidate must force every selected wide root full")


def _adjudicate_s3(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    base: dict[str, Any],
    candidate: dict[str, Any] | None,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    evaluator: dict[str, Any],
    predecessors: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    if predecessors["s2"]["decision"] == "hold":
        if candidate is not None or manifest.get("evidence") != {}:
            raise AdjudicationError(
                "S3 after an S2 hold must use candidate_search_operator=null and empty evidence"
            )
        if base["n_full"] != 64 or base["n_full_wide"] is not None:
            raise AdjudicationError(
                "ineligible S3 hold must retain the n64/no-adaptive base"
            )
        return (
            "hold",
            _operator_selected(base, S3_KEYS),
            {"reason": "s2_held_n64_s3_ineligible"},
            [],
            base,
        )
    if candidate is None:
        raise AdjudicationError(
            "S3 candidate_search_operator is required after S2 adopts n128"
        )
    _validate_s3_candidate(base, candidate)
    evidence = _require_exact_keys(
        manifest.get("evidence"),
        {"h2h", "fixed_root", "baseline_role", "candidate_role"},
        where="S3 evidence",
    )
    h2h_path, h2h_record = _validate_ref(
        evidence["h2h"], base=manifest_path.parent, where="S3 H2H evidence"
    )
    fixed_path, fixed_record = _validate_ref(
        evidence["fixed_root"],
        base=manifest_path.parent,
        where="S3 fixed-root evidence",
    )
    h2h = _load_json(h2h_path)
    pent = _validate_complete_h2h(
        h2h,
        path=h2h_path,
        where="S3 H2H",
        checkpoint_path=checkpoint_path,
        base=base,
        candidate=candidate,
        evaluator=evaluator,
    )
    _validate_h2h_shared_operator(h2h, base, where="S3 H2H")
    if h2h.get("candidate_n_full") != 128 or h2h.get("baseline_n_full") != 128:
        raise AdjudicationError("S3 H2H must keep normal n128 on both sides")
    if (
        h2h.get("candidate_n_full_wide") != 256
        or _integer(
            h2h.get("candidate_n_full_wide_threshold"),
            where="S3 candidate_n_full_wide_threshold",
        )
        < 40
    ):
        raise AdjudicationError("S3 H2H candidate must use adaptive n256 at >=40")
    if h2h.get("baseline_n_full_wide") is not None:
        raise AdjudicationError("S3 H2H baseline must not use adaptive n256")
    overhead = _number(
        h2h.get("search_telemetry", {}).get("candidate_over_baseline_elapsed_ratio"),
        where="S3 whole-game role-attributable overhead ratio",
    )
    by_role = h2h.get("search_telemetry", {}).get("by_role", {})
    if not isinstance(by_role, dict) or set(by_role) != {"candidate", "baseline"}:
        raise AdjudicationError("S3 H2H lacks role-attributable search telemetry")
    candidate_elapsed = _number(
        by_role["candidate"].get("search_elapsed_sec"),
        where="S3 candidate search elapsed",
    )
    baseline_elapsed = _number(
        by_role["baseline"].get("search_elapsed_sec"),
        where="S3 baseline search elapsed",
    )
    if candidate_elapsed < 0.0 or baseline_elapsed <= 0.0:
        raise AdjudicationError("S3 role-attributable elapsed values are invalid")
    _close(
        overhead,
        candidate_elapsed / baseline_elapsed,
        where="S3 recomputed whole-game overhead ratio",
    )
    fixed = _load_json(fixed_path)
    slices = _validate_fixed_root_report(
        fixed,
        path=fixed_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        base=base,
        candidate=candidate,
        evaluator=evaluator,
        baseline_role=str(evidence["baseline_role"]),
        candidate_role=str(evidence["candidate_role"]),
        stage="s3",
    )
    wide = slices.get("wide_ge_40", {})
    if _integer(wide.get("roots"), where="S3 fixed-root wide roots") <= 0:
        raise AdjudicationError("S3 fixed-root report contains no >=40-action roots")
    comparison = wide.get("comparison", {})
    js_reduction = _number(
        comparison.get("role_b_relative_js_reduction"),
        where="S3 wide-root relative JS reduction",
    )
    top1_delta = _number(
        comparison.get("role_b_minus_role_a_top1_agreement"),
        where="S3 wide-root top1 agreement delta",
    )
    strength_pass = pent["decision"] == "H1"
    stability_pass = (
        js_reduction >= POLICY["s3_min_relative_js_reduction"]
        and top1_delta >= POLICY["s3_min_top1_agreement_delta"]
    )
    overhead_pass = overhead <= POLICY["s3_max_whole_game_overhead_ratio"]
    positive = (strength_pass or stability_pass) and overhead_pass
    confirmed = int(pent["pairs"]) >= POLICY["confirmation_pairs"]
    if positive and not confirmed:
        raise AdjudicationError(
            "S3 positive 50-pair screen requires 200-pair confirmation"
        )
    adopt = positive and confirmed
    final_operator = candidate if adopt else base
    metrics = {
        "reason": (
            "confirmed_strength_or_stability_within_overhead"
            if adopt
            else "retain_base_no_binding_adaptive_n256_win"
        ),
        "pentanomial_decision": pent["decision"],
        "pairs": pent["pairs"],
        "strength_pass": strength_pass,
        "wide_root_relative_js_reduction": js_reduction,
        "wide_root_top1_agreement_delta": top1_delta,
        "stability_pass": stability_pass,
        "whole_game_search_overhead_ratio": overhead,
        "overhead_pass": overhead_pass,
    }
    return (
        "adopt" if adopt else "hold",
        _operator_selected(final_operator, S3_KEYS),
        metrics,
        [h2h_record, fixed_record],
        final_operator,
    )


def adjudicate(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().absolute()
    manifest = _load_json(manifest_path)
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "stage",
            "checkpoint",
            "base_search_operator",
            "candidate_search_operator",
            "teacher_evaluator",
            "predecessors",
            "evidence",
        },
        where="adjudication manifest",
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise AdjudicationError(f"manifest schema must be {MANIFEST_SCHEMA}")
    stage = str(manifest.get("stage"))
    if stage not in {"s1", "s2", "s3"}:
        raise AdjudicationError("manifest stage must be s1, s2, or s3")
    manifest_record = {"path": str(manifest_path), "sha256": _sha256(manifest_path)}
    checkpoint_path, checkpoint_record = _validate_ref(
        manifest["checkpoint"], base=manifest_path.parent, where="checkpoint"
    )
    base = _validate_search_operator(
        manifest["base_search_operator"], where="base_search_operator"
    )
    candidate_raw = manifest["candidate_search_operator"]
    candidate = (
        _validate_search_operator(candidate_raw, where="candidate_search_operator")
        if candidate_raw is not None
        else None
    )
    evaluator = _validate_evaluator(
        manifest["teacher_evaluator"], where="teacher_evaluator"
    )
    predecessors, predecessor_records = _load_predecessors(
        manifest["predecessors"], manifest_path=manifest_path, stage=stage
    )
    _validate_lineage(stage, base, predecessors)

    if stage == "s1":
        decision, selected, metrics, evidence_records = _adjudicate_s1(
            manifest,
            manifest_path=manifest_path,
            base=base,
            checkpoint_path=checkpoint_path,
        )
        final_operator = None
    elif stage == "s2":
        decision, selected, metrics, evidence_records = _adjudicate_s2(
            manifest,
            manifest_path=manifest_path,
            base=base,
            candidate=candidate,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_record["sha256"],
            evaluator=evaluator,
        )
        final_operator = None
    else:
        decision, selected, metrics, evidence_records, final_operator = _adjudicate_s3(
            manifest,
            manifest_path=manifest_path,
            base=base,
            candidate=candidate,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_record["sha256"],
            evaluator=evaluator,
            predecessors=predecessors,
        )

    sources = _dedupe_records(
        [
            manifest_record,
            checkpoint_record,
            {
                "path": str(Path(__file__).absolute()),
                "sha256": _sha256(Path(__file__).absolute()),
            },
            *predecessor_records,
            *evidence_records,
        ]
    )
    adjudicator_record = {
        "path": str(Path(__file__).absolute()),
        "sha256": _sha256(Path(__file__).absolute()),
    }
    config_hashes = {
        "base_search_operator_sha256": _digest_value(base),
        "candidate_search_operator_sha256": (
            _digest_value(candidate) if candidate is not None else None
        ),
        "teacher_evaluator_sha256": _digest_value(evaluator),
        "adjudication_policy_sha256": _digest_value(POLICY),
        "input_bundle_sha256": _digest_value(
            {
                "source_artifacts": sources,
                "base_search_operator": base,
                "candidate_search_operator": candidate,
                "teacher_evaluator": evaluator,
                "policy": POLICY,
            }
        ),
    }
    envelope: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "stage": stage,
        "passed": True,
        "decision": decision,
        "adjudicator": adjudicator_record,
        "source_artifacts": sources,
        "selected_fields": selected,
        "selected_fields_sha256": _digest_value(selected),
        "config_hashes": config_hashes,
        "thresholds": POLICY,
        "metrics": metrics,
    }
    if stage == "s3":
        assert final_operator is not None
        envelope.update(
            {
                "final_search_operator": final_operator,
                "final_search_operator_sha256": _digest_value(final_operator),
                "teacher_evaluator": evaluator,
                "teacher_evaluator_sha256": _digest_value(evaluator),
            }
        )
    return envelope


def _create_readonly(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise AdjudicationError(
            f"refusing to overwrite immutable decision {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adjudicate immutable S1/S2/S3 search-teacher evidence; never runs probes."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        payload = adjudicate(Path(args.manifest))
        _create_readonly(Path(args.out), payload)
    except AdjudicationError as error:
        parser.error(str(error))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
