#!/usr/bin/env python3
"""Seal the three panels that distinguish network gain from search compensation.

One searched candidate-vs-parent result is ambiguous: a worse raw network can
still win because search repairs it, or because its value/search interaction
changed.  This verifier requires the same paired cohort under three topologies:

* raw candidate vs raw parent;
* searched candidate vs searched parent with one shared operator;
* searched candidate vs the same candidate's raw policy.

All panels are emitted by ``gumbel_search_cross_net_h2h.py``.  Role-specific
``--*-raw-policy-above-width 0`` makes a role raw-prior argmax at every
multi-action root; forced roots are operator-invariant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
for _entry in (_SRC_DIR, _REPO_ROOT, _TOOLS_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from catan_zero.rl.pipeline_configs import config_from_payload  # noqa: E402
from factory_common import write_json  # noqa: E402

SCHEMA = "a1-neural-search-strength-decomposition-v2"
RAW_CONTRACT = "paired_same_seed_color_swap_raw_networks"
SEARCHED_CONTRACT = "paired_same_seed_color_swap_shared_search_operator"
UPLIFT_CONTRACT = "paired_same_seed_color_swap_candidate_search_vs_own_raw"


class DecompositionError(ValueError):
    """Raised when the panel set cannot identify neural and search effects."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest_value(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DecompositionError(f"{path} is not a JSON object")
    return value


def _checkpoint_ref(
    report: Mapping[str, Any], role: str, *, report_path: Path
) -> dict[str, str]:
    raw_path = report.get(f"{role}_checkpoint")
    raw_sha = report.get(f"{role}_checkpoint_sha256")
    if not isinstance(raw_path, str) or not isinstance(raw_sha, str):
        raise DecompositionError(f"{report_path} lacks {role} checkpoint identity")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = report_path.parent / path
    path = path.resolve(strict=True)
    observed = _sha256(path)
    if observed != raw_sha:
        raise DecompositionError(f"{report_path} {role} checkpoint bytes drifted")
    return {"path": str(path), "sha256": observed}


def _typed_config(report: Mapping[str, Any], *, report_path: Path) -> dict[str, Any]:
    payload = report.get("typed_config")
    if not isinstance(payload, dict):
        raise DecompositionError(f"{report_path} lacks typed_config")
    try:
        config = config_from_payload(payload)
    except (TypeError, ValueError) as error:
        raise DecompositionError(f"{report_path} typed_config is invalid") from error
    if (
        config.config_hash() != report.get("config_hash")
        or config.full_config_hash() != report.get("full_config_hash")
        or config.canonical_payload() != payload
    ):
        raise DecompositionError(f"{report_path} typed config/hash does not replay")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or fields.get("mode") != "cross_net":
        raise DecompositionError(f"{report_path} is not a typed cross-net panel")
    return dict(fields)


def _panel_config(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    candidate: Mapping[str, str],
    baseline: Mapping[str, str],
) -> dict[str, Any]:
    """Replay either one direct report or a canonical fleet-pooled report."""

    if isinstance(report.get("typed_config"), dict):
        return _typed_config(report, report_path=report_path)
    merge = report.get("fleet_merge")
    if (
        not isinstance(merge, dict)
        or merge.get("schema_version") != "a1-fleet-evaluation-pool-v1"
        or merge.get("kind") != "internal_h2h"
    ):
        raise DecompositionError(
            f"{report_path} has neither typed config nor canonical fleet provenance"
        )
    sources = merge.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DecompositionError(f"{report_path} pooled panel has no source reports")
    source_paths: list[Path] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise DecompositionError(
                f"{report_path}.fleet_merge.sources[{index}] is malformed"
            )
        source_path = Path(str(source["path"])).expanduser()
        if not source_path.is_absolute():
            source_path = report_path.parent / source_path
        source_path = source_path.resolve(strict=True)
        if _sha256(source_path) != source["sha256"]:
            raise DecompositionError(
                f"{report_path}.fleet_merge.sources[{index}] bytes drifted"
            )
        source_paths.append(source_path)
    try:
        from tools import a1_evaluation_pool as evaluation_pool

        replayed = evaluation_pool.pool_internal(
            source_paths,
            candidate=Path(candidate["path"]),
            champion=Path(baseline["path"]),
            allow_disjoint_cohorts=merge.get("disjoint_cohorts") is True,
        )
    except (evaluation_pool.PoolError, OSError, ValueError) as error:
        raise DecompositionError(f"{report_path} fleet pool does not replay") from error
    if replayed != report:
        raise DecompositionError(f"{report_path} differs from its replayed fleet pool")
    source_report = _load(source_paths[0])
    fields = _typed_config(source_report, report_path=source_paths[0])
    effective = dict(fields)
    for name in ("candidate", "baseline", "base_seed", "pairs"):
        effective.pop(name, None)
    if effective != report.get("effective_search_config"):
        raise DecompositionError(f"{report_path} pooled effective config drifted")
    return fields


def _game_keys(report: Mapping[str, Any], *, report_path: Path) -> list[tuple[int, str]]:
    games = report.get("games")
    if not isinstance(games, list) or not games or len(games) % 2:
        raise DecompositionError(f"{report_path} lacks complete paired games")
    if (
        report.get("errors") != []
        or report.get("games_truncated") != 0
        or report.get("games_played") != len(games)
        or report.get("games_with_winner") != len(games)
        or report.get("complete_pairs") != len(games) // 2
    ):
        raise DecompositionError(f"{report_path} is not a clean complete panel")
    keys: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    orientations: dict[int, set[str]] = {}
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise DecompositionError(f"{report_path}.games[{index}] is not an object")
        seed = game.get("game_seed")
        orientation = game.get("orientation")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or orientation not in {"candidate_red", "candidate_blue"}
            or type(game.get("candidate_won")) is not bool  # noqa: E721
            or game.get("terminated") is not True
            or game.get("truncated") is not False
        ):
            raise DecompositionError(f"{report_path}.games[{index}] is not clean")
        key = (seed, str(orientation))
        if key in seen:
            raise DecompositionError(f"{report_path} repeats game identity {key}")
        seen.add(key)
        orientations.setdefault(seed, set()).add(str(orientation))
        keys.append(key)
    if any(values != {"candidate_red", "candidate_blue"} for values in orientations.values()):
        raise DecompositionError(f"{report_path} omits a color-swapped orientation")
    return sorted(keys)


def _finite_rate(report: Mapping[str, Any], *, report_path: Path) -> float:
    value = report.get("candidate_win_rate")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise DecompositionError(f"{report_path} has no finite candidate win rate")
    return float(value)


def _role_operator(fields: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Return the fully resolved operator/evaluator semantics for one role."""

    shared_to_role = {
        "n_full": f"{role}_n_full",
        "n_full_wide": f"{role}_n_full_wide",
        "n_full_wide_threshold": f"{role}_n_full_wide_threshold",
        "wide_roots_always_full": f"{role}_wide_roots_always_full",
        "raw_policy_above_width": f"{role}_raw_policy_above_width",
        "c_scale": f"{role}_c_scale",
        "value_squash": f"{role}_value_squash",
        "value_readout": f"{role}_value_readout",
        "gameplay_policy_aggregation": f"{role}_gameplay_policy_aggregation",
        "rescale_noise_floor_c": f"{role}_rescale_noise_floor_c",
        "sigma_eval": f"{role}_sigma_eval",
        "sigma_reference_visits": f"{role}_sigma_reference_visits",
        "boundary_value_particles": f"{role}_boundary_value_particles",
    }
    global_keys = {
        "map_kind",
        "public_observation",
        "belief_chance_spectra",
        "information_set_search",
        "coherent_public_belief_search",
        "forced_root_target_mode",
        "native_mcts_hot_loop",
        "determinization_particles",
        "determinization_min_simulations",
        "max_depth",
        "max_decisions",
        "c_visit",
        "max_root_candidates",
        "max_root_candidates_wide",
        "wide_candidates_threshold",
        "symmetry_averaged_eval",
        "symmetry_averaged_eval_threshold",
        "correct_rust_chance_spectra",
        "lazy_interior_chance",
        "prior_temperature",
        "value_scale",
        "temperature",
        "force_full_every_decision",
        "exact_budget_sh",
        "root_wave_batching",
        "use_batch_api",
        "evaluator_rust_featurize",
        "evaluator_emit_uncertainty",
        "evaluator_context_fill",
    }
    result = {key: fields.get(key) for key in sorted(global_keys)}
    for shared, specific in shared_to_role.items():
        result[shared] = fields.get(specific, fields.get(shared))
    return result


def _raw_policy_signature(operator: Mapping[str, Any]) -> dict[str, Any]:
    """Semantics that can alter raw-policy games, excluding unused value/search."""

    keys = {
        "map_kind",
        "public_observation",
        "max_decisions",
        "correct_rust_chance_spectra",
        "prior_temperature",
        "evaluator_rust_featurize",
        "evaluator_context_fill",
    }
    return {key: operator.get(key) for key in sorted(keys)}


def _verify_panel(
    path: Path, *, expected_contract: str
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    report = _load(path)
    if report.get("comparison_contract") != expected_contract:
        raise DecompositionError(
            f"{path} has comparison_contract={report.get('comparison_contract')!r}; "
            f"expected {expected_contract!r}"
        )
    planned = report.get("planned_engine_identity")
    observed = report.get("engine_identity")
    if not isinstance(planned, dict) or planned != observed:
        raise DecompositionError(f"{path} engine identity is missing or drifted")
    candidate = _checkpoint_ref(report, "candidate", report_path=path)
    baseline = _checkpoint_ref(report, "baseline", report_path=path)
    fields = _panel_config(
        report,
        report_path=path,
        candidate=candidate,
        baseline=baseline,
    )
    keys = _game_keys(report, report_path=path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "candidate": candidate,
        "baseline": baseline,
        "fields": fields,
        "game_keys": keys,
        "candidate_win_rate": _finite_rate(report, report_path=path),
        "verdict": report.get("verdict"),
        "superiority_verdict": report.get("superiority_verdict"),
        "candidate_operator": _role_operator(fields, "candidate"),
        "baseline_operator": _role_operator(fields, "baseline"),
        "engine_identity": observed,
    }


def build_decomposition(
    *, raw_cross: Path, searched_cross: Path, candidate_search_vs_raw: Path
) -> dict[str, Any]:
    raw = _verify_panel(raw_cross, expected_contract=RAW_CONTRACT)
    searched = _verify_panel(searched_cross, expected_contract=SEARCHED_CONTRACT)
    uplift = _verify_panel(candidate_search_vs_raw, expected_contract=UPLIFT_CONTRACT)

    candidate = searched["candidate"]
    parent = searched["baseline"]
    if raw["candidate"] != candidate or raw["baseline"] != parent:
        raise DecompositionError("raw and searched panels bind different checkpoints")
    if uplift["candidate"] != candidate or uplift["baseline"] != candidate:
        raise DecompositionError(
            "search-uplift panel must compare the candidate checkpoint to itself"
        )
    if not (raw["game_keys"] == searched["game_keys"] == uplift["game_keys"]):
        raise DecompositionError(
            "all decomposition panels must use the exact same paired seed cohort"
        )
    if not (
        raw["engine_identity"]
        == searched["engine_identity"]
        == uplift["engine_identity"]
    ):
        raise DecompositionError("decomposition panels used different engine bytes")

    searched_candidate = dict(searched["candidate_operator"])
    searched_parent = dict(searched["baseline_operator"])
    if searched_candidate != searched_parent:
        raise DecompositionError(
            "searched candidate-vs-parent panel does not use one shared operator"
        )
    if uplift["candidate_operator"] != searched_candidate:
        raise DecompositionError(
            "candidate search-uplift panel changed the candidate search operator"
        )
    if raw["candidate_operator"]["raw_policy_above_width"] != 0 or raw[
        "baseline_operator"
    ]["raw_policy_above_width"] != 0:
        raise DecompositionError("raw panel is not raw-prior argmax on both roles")
    if uplift["baseline_operator"]["raw_policy_above_width"] != 0:
        raise DecompositionError("search-uplift baseline is not candidate raw policy")
    candidate_raw_signatures = {
        _digest_value(_raw_policy_signature(raw["candidate_operator"])),
        _digest_value(_raw_policy_signature(searched_candidate)),
        _digest_value(_raw_policy_signature(uplift["candidate_operator"])),
        _digest_value(_raw_policy_signature(uplift["baseline_operator"])),
    }
    if len(candidate_raw_signatures) != 1:
        raise DecompositionError(
            "candidate raw-policy observation/evaluator semantics changed across panels"
        )
    if _raw_policy_signature(raw["baseline_operator"]) != _raw_policy_signature(
        searched_parent
    ):
        raise DecompositionError(
            "parent raw-policy observation/evaluator semantics changed across panels"
        )

    searched_superiority = searched["superiority_verdict"]
    raw_resolved_nonregression = raw["verdict"] == "H1"
    uplift_resolved = uplift["verdict"] == "H1"
    raw_regressed = raw["verdict"] == "H0"
    search_uplift_failed = uplift["verdict"] == "H0"
    search_compensation_risk = (
        searched_superiority == "H1"
        and (
            raw["candidate_win_rate"] < 0.5
            or uplift["candidate_win_rate"] <= 0.5
        )
    )
    ready = (
        searched_superiority == "H1"
        and raw_resolved_nonregression
        and uplift_resolved
        and not search_compensation_risk
    )
    value = {
        "schema_version": SCHEMA,
        "candidate": candidate,
        "parent": parent,
        "cohort": {
            "complete_pairs": len(raw["game_keys"]) // 2,
            "ordered_game_identity_sha256": _digest_value(raw["game_keys"]),
        },
        "panels": {
            "raw_candidate_vs_raw_parent": {
                key: raw[key]
                for key in ("path", "sha256", "candidate_win_rate", "verdict")
            },
            "searched_candidate_vs_searched_parent": {
                key: searched[key]
                for key in (
                    "path",
                    "sha256",
                    "candidate_win_rate",
                    "verdict",
                    "superiority_verdict",
                )
            },
            "searched_candidate_vs_own_raw": {
                key: uplift[key]
                for key in ("path", "sha256", "candidate_win_rate", "verdict")
            },
        },
        "diagnosis": {
            "searched_checkpoint_superiority_proven": searched_superiority == "H1",
            "raw_network_nonregression_resolved": raw_resolved_nonregression,
            "candidate_search_uplift_resolved": uplift_resolved,
            "raw_network_material_regression_detected": raw_regressed,
            "candidate_search_uplift_failure_detected": search_uplift_failed,
            "search_compensation_risk": search_compensation_risk,
        },
        "ready_for_promotion_adjudication": ready,
        "contract_note": (
            "This receipt is a prerequisite/decomposition artifact, not promotion "
            "evidence by itself."
        ),
    }
    value["receipt_sha256"] = _digest_value(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cross", type=Path, required=True)
    parser.add_argument("--searched-cross", type=Path, required=True)
    parser.add_argument("--candidate-search-vs-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = build_decomposition(
            raw_cross=args.raw_cross,
            searched_cross=args.searched_cross,
            candidate_search_vs_raw=args.candidate_search_vs_raw,
        )
    except (DecompositionError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    write_json(args.out, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
