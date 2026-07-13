#!/usr/bin/env python3
"""Seal a replayable S3 HOLD from the same-checkpoint role-operator panel.

This artifact is deliberately diagnostic operator evidence, not model-promotion
evidence.  It re-pools every retained fleet shard, proves that the only role
difference is adaptive n256 at wide roots, and selects the baseline n128/no-
adaptive operator when strict superiority is unresolved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_evaluation_pool as evaluation_pool  # noqa: E402


SCHEMA = "a1-s3-role-operator-hold-v1"
ARTIFACT_KIND = "diagnostic_operator_decision_not_promotion_evidence"
STATEMENT = (
    "This same-checkpoint role-operator result selects the baseline search "
    "operator for the next wave; it is not checkpoint-promotion evidence."
)
SELECTED_FIELDS = {
    "n_full_wide": None,
    "n_full_wide_threshold": None,
    "wide_roots_always_full": False,
}
REASON = "adaptive_n256_did_not_establish_strict_superiority"


class HoldError(ValueError):
    """Raised when source evidence cannot form an honest S3 HOLD."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HoldError(f"value is not canonical JSON: {error}") from error


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise HoldError(f"cannot hash {path}: {error}") from error
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HoldError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise HoldError(f"{path} must contain a JSON object")
    return value


def _ref(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _timestamp(raw: str | None) -> str:
    if raw is None:
        value = dt.datetime.now(dt.timezone.utc)
    else:
        try:
            value = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as error:
            raise HoldError("decision time must be ISO-8601") from error
        if value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
            raise HoldError("decision time must carry an explicit UTC offset")
        value = value.astimezone(dt.timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldError(message)


def _validate_and_replay(pooled_path: Path) -> dict[str, Any]:
    pooled_path = pooled_path.expanduser().resolve(strict=True)
    pooled = _load(pooled_path)
    merge = pooled.get("fleet_merge")
    _require(
        isinstance(merge, dict)
        and merge.get("schema_version") == evaluation_pool.POOL_SCHEMA
        and merge.get("kind") == "internal_h2h",
        "S3 source is not a typed internal fleet pool",
    )
    source_refs = merge.get("sources")
    _require(isinstance(source_refs, list) and source_refs, "S3 pool has no shards")
    sources: list[Path] = []
    for index, raw in enumerate(source_refs):
        _require(
            isinstance(raw, dict) and set(raw) == {"path", "sha256"},
            f"S3 source reference {index} is malformed",
        )
        source = Path(str(raw["path"])).expanduser().resolve(strict=True)
        _require(_sha256(source) == raw["sha256"], f"S3 shard hash drift: {source}")
        sources.append(source)

    candidate = Path(str(pooled.get("candidate_checkpoint", ""))).expanduser().resolve(
        strict=True
    )
    baseline = Path(str(pooled.get("baseline_checkpoint", ""))).expanduser().resolve(
        strict=True
    )
    _require(candidate == baseline, "S3 must compare one checkpoint to itself")
    checkpoint_sha = _sha256(candidate)
    _require(
        pooled.get("candidate_checkpoint_sha256") == checkpoint_sha
        and pooled.get("baseline_checkpoint_sha256") == checkpoint_sha,
        "S3 checkpoint digest drift",
    )
    try:
        replayed = evaluation_pool.pool_internal(
            sources,
            candidate=candidate,
            champion=baseline,
            allow_disjoint_cohorts=bool(merge.get("disjoint_cohorts", False)),
        )
    except evaluation_pool.PoolError as error:
        raise HoldError(f"S3 pool replay failed: {error}") from error
    # The fleet controller appends these two authenticated provenance records
    # after generic pooling.  Replay every science/game/statistics field, then
    # validate the append-only records independently instead of pretending the
    # generic pooler emitted them.
    replay_projection = dict(pooled)
    evaluation_binding = replay_projection.pop("evaluation_binding", None)
    planned_engine = replay_projection.pop("planned_engine_identity", None)
    _require(
        _canonical(replayed) == _canonical(replay_projection),
        "S3 pooled result does not replay",
    )
    _require(
        isinstance(evaluation_binding, dict)
        and evaluation_binding.get("comparison_mode") == "historical_comparison"
        and evaluation_binding.get("promotion_eligible") is False
        and isinstance(evaluation_binding.get("historical_comparison_reason"), str)
        and bool(evaluation_binding["historical_comparison_reason"].strip()),
        "S3 source must explicitly bind diagnostic-only comparison semantics",
    )
    for role in ("candidate_parent", "baseline"):
        bound = evaluation_binding.get(role)
        _require(
            isinstance(bound, dict)
            and bound.get("path") == str(candidate)
            and bound.get("sha256") == checkpoint_sha,
            f"S3 evaluation binding {role} checkpoint drift",
        )
    incumbent = evaluation_binding.get("authoritative_incumbent")
    _require(
        isinstance(incumbent, dict)
        and incumbent.get("path") == str(candidate)
        and incumbent.get("sha256") == checkpoint_sha,
        "S3 evaluation binding incumbent drift",
    )
    registry = evaluation_binding.get("registry")
    _require(
        isinstance(registry, dict) and set(registry) == {"path", "sha256"},
        "S3 evaluation binding registry reference is malformed",
    )
    registry_path = Path(str(registry["path"])).expanduser().resolve(strict=True)
    _require(_sha256(registry_path) == registry["sha256"], "S3 registry hash drift")
    _require(
        isinstance(planned_engine, dict)
        and set(planned_engine)
        == {
            "schema_version",
            "repo_commit",
            "native_wheel_sha256",
            "python_referee_sha256",
        }
        and planned_engine.get("schema_version") == "a1-neutral-engine-identity-v1"
        and isinstance(planned_engine.get("repo_commit"), str)
        and len(planned_engine["repo_commit"]) == 40
        and all(character in "0123456789abcdef" for character in planned_engine["repo_commit"])
        and all(
            isinstance(planned_engine.get(key), str)
            and len(planned_engine[key]) == 71
            and planned_engine[key].startswith("sha256:")
            for key in ("native_wheel_sha256", "python_referee_sha256")
        ),
        "S3 planned engine identity is malformed",
    )

    _require(pooled.get("errors") == [], "S3 source contains errors")
    _require(pooled.get("games_truncated") == 0, "S3 source contains truncations")
    _require(
        pooled.get("complete_pairs") == 200 and pooled.get("games_played") == 400,
        "S3 source must contain exactly 200 complete paired games",
    )
    _require(
        pooled.get("public_observation") is True
        and pooled.get("information_set_search") is True
        and pooled.get("determinization_particles") == 4
        and pooled.get("determinization_min_simulations") == 32
        and pooled.get("native_mcts_hot_loop") is True
        and pooled.get("mcts_implementation") == "rust_native_hot_loop_v1",
        "S3 source used an unsupported information/runtime recipe",
    )

    budgets = pooled.get("search_budgets_by_role")
    _require(
        budgets
        == {
            "candidate": {
                "n_full": 128,
                "n_full_wide": 256,
                "n_full_wide_threshold": 40,
                "wide_roots_always_full": True,
            },
            "baseline": {
                "n_full": 128,
                "n_full_wide": None,
                "n_full_wide_threshold": None,
                "wide_roots_always_full": False,
            },
        },
        "S3 source does not bind adaptive-n256 versus global-n128 roles",
    )
    for suffix in (
        "c_scale",
        "gameplay_policy_aggregation",
        "rescale_noise_floor_c",
        "sigma_eval",
        "sigma_reference_visits",
        "value_readout",
        "value_squash",
    ):
        _require(
            pooled.get(f"candidate_{suffix}") == pooled.get(f"baseline_{suffix}"),
            f"S3 source changes non-budget role field {suffix}",
        )

    strict = pooled.get("superiority_pentanomial_sprt")
    _require(
        isinstance(strict, dict)
        and strict.get("elo0") == 0.0
        and strict.get("elo1") == 15.0
        and strict.get("decision") != "H1"
        and pooled.get("superiority_verdict") == strict.get("decision"),
        "adaptive n256 established strict superiority; HOLD is invalid",
    )
    telemetry = pooled.get("search_telemetry")
    _require(isinstance(telemetry, dict), "S3 source has no search telemetry")
    for key in (
        "candidate_over_baseline_simulations_ratio",
        "candidate_over_baseline_elapsed_ratio",
        "candidate_over_baseline_seconds_per_call_ratio",
    ):
        value = telemetry.get(key)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0,
            f"S3 telemetry {key} is invalid",
        )
    return pooled


def build_hold(
    pooled_path: Path,
    *,
    source_s1: Path,
    source_s2: Path,
    decision_time_utc: str | None = None,
    emitter_path: Path | None = None,
) -> dict[str, Any]:
    pooled_path = pooled_path.expanduser().resolve(strict=True)
    pooled = _validate_and_replay(pooled_path)
    telemetry = pooled["search_telemetry"]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "stage": "s3",
        "passed": True,
        "decision": "hold",
        "operator": "adaptive_n256_disabled",
        "reason": REASON,
        "statement": STATEMENT,
        "decision_time_utc": _timestamp(decision_time_utc),
        "selected_fields": dict(SELECTED_FIELDS),
        "selected_fields_sha256": _digest(SELECTED_FIELDS),
        "checkpoint": {
            "path": pooled["candidate_checkpoint"],
            "sha256": pooled["candidate_checkpoint_sha256"],
        },
        "source_pooled": _ref(pooled_path),
        "source_s1": _ref(source_s1),
        "source_s2": _ref(source_s2),
        "emitter": _ref(Path(__file__) if emitter_path is None else emitter_path),
        "observations": {
            "complete_pairs": pooled["complete_pairs"],
            "games_played": pooled["games_played"],
            "candidate_wins": pooled["candidate_wins"],
            "baseline_wins": pooled["baseline_wins"],
            "candidate_win_rate": pooled["candidate_win_rate"],
            "pair_diagnostics": pooled["pair_diagnostics"],
            "promotion_band": pooled["pentanomial_sprt"],
            "strict_superiority_band": pooled["superiority_pentanomial_sprt"],
            "candidate_over_baseline_simulations_ratio": telemetry[
                "candidate_over_baseline_simulations_ratio"
            ],
            "candidate_over_baseline_elapsed_ratio": telemetry[
                "candidate_over_baseline_elapsed_ratio"
            ],
            "candidate_over_baseline_seconds_per_call_ratio": telemetry[
                "candidate_over_baseline_seconds_per_call_ratio"
            ],
            "pooled_semantic_sha256": _digest(pooled),
        },
    }
    payload["artifact_content_sha256"] = _digest(payload)
    return payload


def write_hold(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled", required=True)
    parser.add_argument("--source-s1", required=True)
    parser.add_argument("--source-s2", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decision-time-utc", default=None)
    args = parser.parse_args()
    try:
        payload = build_hold(
            Path(args.pooled),
            source_s1=Path(args.source_s1),
            source_s2=Path(args.source_s2),
            decision_time_utc=args.decision_time_utc,
        )
        write_hold(Path(args.out), payload)
    except (HoldError, FileExistsError, OSError) as error:
        print(f"S3 HOLD ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
