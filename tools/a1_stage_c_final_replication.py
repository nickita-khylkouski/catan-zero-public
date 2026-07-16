#!/usr/bin/env python3
"""Seal the independent production replication selected by Stage-C.

Stage-C diagnostics are allowed to select a recipe, never to become production
evidence themselves.  This module closes that boundary in three explicit
transactions:

* ``adjudicate`` authenticates every paired external panel and applies one
  deterministic selection rule;
* ``seal-roots`` proves that the final coherent-n128 roots and game/search RNG
  domains do not overlap diagnostics or evaluation;
* ``admit-corpus`` and ``issue`` bind freshly produced target bytes, the exact
  authoritative current parent, fresh Adam, and the selected dose into a run that is
  eligible only for the full promotion gate.

The low-level Stage-C reanalysis/export machinery is reused, but a diagnostic
target patch or candidate checkpoint can never be relabelled as the final run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools import a1_stage_c_learner_overlay as overlay  # noqa: E402
from tools import a1_stage_c_reanalysis_executor as reanalysis  # noqa: E402
from tools import a1_stage_c_teacher_alignment as alignment  # noqa: E402


ADJUDICATION_SCHEMA = "a1-stage-c-recipe-adjudication-v2"
TIEBREAK_SCHEMA = "a1-stage-c-v5-dose-tiebreak-common-crn-v1"
ROOT_MANIFEST_SCHEMA = "a1-stage-c-independent-root-manifest-v1"
FINAL_CORPUS_ADMISSION_SCHEMA = "a1-stage-c-final-corpus-admission-v1"
FINAL_AUTHORITY_SCHEMA = "a1-stage-c-final-matched-replication-authority-v2"
FRESH_FINGERPRINT_SCHEMA = "a1-b200-stage-c-aligned-learner-fingerprint-v2"
EXPECTED_ARM = "STRATEGIC_BALANCED"
EXPECTED_ROOTS = 8_192
EXPECTED_PARTITIONS = 64
EXPECTED_PAIRS = 128
EXPECTED_GAMES = EXPECTED_PAIRS * 2
EXPECTED_COMPARISON = "paired_same_seed_color_swap_shared_search_operator"
EXPECTED_ENGINE_SCHEMA = "a1-internal-h2h-engine-identity-v1"
EXPECTED_POOL_SCHEMA = "a1-fleet-evaluation-pool-v1"
EXPECTED_OPERATOR = "public_belief_single_tree_v1"
EXPECTED_EXECUTOR = "coherent_public_belief_n128_reanalysis_v1"
PREP_INVENTORY_SCHEMA = "a1-stage-c-final-prep-independent-root-inventory-v1"
EXPECTED_SELECTION_RULE = (
    "select_step16_as_recipe_only_when_common_crn_v5_score_exceeds_step8_on_"
    "fresh256_and_combined384;retain_all_incumbent_h0_negative_evidence;"
    "final_strength_requires_fresh_current_parent_replication"
)
EXPECTED_TIEBREAK_SOURCE_KEYS = {
    "step16_plan",
    "step16_prior",
    "step16_replacement",
    "step16_replacement_plan",
    "step8_fresh",
    "step8_plan",
    "step8_prior",
    "step8_replacement",
    "step8_replacement_plan",
}
EXPECTED_COMBINED_PAIRS = 384
EXPECTED_TIEBREAK_PAIRS = 256
EXPECTED_COMBINED_STEP8_WINS = 351
EXPECTED_COMBINED_STEP16_WINS = 362
EXPECTED_COMBINED_LIFT = 11 / 768
EXPECTED_FRESH_LIFT = 12 / 512
FINAL_CONTROL_ARM = "terminal-value-control"
FINAL_VALUE_REPAIR_ARM = "terminal-value-repair"
FINAL_TRUNK_QUARTER_ARM = "trunk-protection-0.25"
FINAL_TRUNK_TENTH_ARM = "trunk-protection-0.10"
FINAL_ARM_NAMES = (
    FINAL_CONTROL_ARM,
    FINAL_VALUE_REPAIR_ARM,
    FINAL_TRUNK_QUARTER_ARM,
    FINAL_TRUNK_TENTH_ARM,
)
SHA_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class FinalReplicationError(RuntimeError):
    """The Stage-C final replication chain is not authentic or independent."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular_file(path: Path, *, where: str) -> Path:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise FinalReplicationError(f"{where} must be a regular file: {lexical}")
    return lexical.resolve(strict=True)


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, where=where)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalReplicationError(f"cannot read {where}: {error}") from error
    if not isinstance(payload, dict):
        raise FinalReplicationError(f"{where} must contain one JSON object")
    return resolved, payload


def _artifact(path: Path) -> dict[str, Any]:
    resolved = _regular_file(path, where="artifact")
    return {
        "path": str(resolved),
        "file_sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_artifact(reference: Mapping[str, Any], *, where: str) -> Path:
    if not {"path", "file_sha256", "size_bytes"} <= set(reference):
        raise FinalReplicationError(f"{where} artifact shape drifted")
    path = _regular_file(Path(str(reference["path"])), where=where)
    if (
        not SHA_PATTERN.fullmatch(str(reference["file_sha256"]))
        or file_sha256(path) != reference["file_sha256"]
        or path.stat().st_size != int(reference["size_bytes"])
    ):
        raise FinalReplicationError(f"{where} artifact bytes drifted")
    return path


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve(strict=False)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_text(encoding="utf-8") != rendered
        ):
            raise FinalReplicationError(
                f"immutable output already exists with drift: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sealed(path: Path, *, schema: str, digest_field: str, where: str) -> dict[str, Any]:
    _resolved, payload = _load_json(path, where=where)
    unsigned = dict(payload)
    stated = unsigned.pop(digest_field, None)
    if (
        payload.get("schema_version") != schema
        or not SHA_PATTERN.fullmatch(str(stated))
        or stated != value_sha256(unsigned)
    ):
        raise FinalReplicationError(f"{where} schema/digest drifted")
    return payload


def _fingerprint(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved, payload = _load_json(path, where="fresh-parent fingerprint")
    unsigned = dict(payload)
    stated = unsigned.pop("fingerprint_sha256", None)
    if (
        payload.get("schema_version") != FRESH_FINGERPRINT_SCHEMA
        or stated != value_sha256(unsigned)
        or payload.get("stored_generation_prior_used_as_selection_authority")
        is not False
        or payload.get("optimizer_batch_kl_used_as_trust_authority") is not False
        or payload.get("separate_exact_parent_evidence", {}).get(
            "selection_authority"
        )
        is not True
    ):
        raise FinalReplicationError("fresh-parent fingerprint semantics drifted")
    campaign_ref = payload.get("campaign")
    if not isinstance(campaign_ref, dict):
        raise FinalReplicationError("fresh-parent fingerprint lost campaign")
    campaign_path = _regular_file(
        Path(str(campaign_ref.get("path", ""))), where="Stage-C campaign"
    )
    if file_sha256(campaign_path) != campaign_ref.get("file_sha256"):
        raise FinalReplicationError("Stage-C campaign bytes drifted")
    _campaign_path, campaign = _load_json(campaign_path, where="Stage-C campaign")
    campaign_unsigned = dict(campaign)
    campaign_stated = campaign_unsigned.pop("campaign_sha256", None)
    if (
        campaign_stated != value_sha256(campaign_unsigned)
        or campaign_stated != campaign_ref.get("campaign_sha256")
        or campaign.get("arm") != EXPECTED_ARM
        or campaign.get("diagnostic_only") is not True
        or campaign.get("promotion_eligible") is not False
        or campaign.get("lineage", {}).get("fresh_adam") is not True
        or campaign.get("lineage", {}).get("candidate_chaining") is not False
        or campaign.get("topology")
        != {
            "name": "b200-8gpu-ddp",
            "world_size": 8,
            "local_batch_size": 512,
            "global_batch_size": 4096,
        }
    ):
        raise FinalReplicationError("Stage-C strategic campaign semantics drifted")
    payload["_verified_campaign"] = campaign
    payload["_verified_campaign_path"] = str(campaign_path)
    return resolved, payload


def _checkpoint_record(
    fingerprint: Mapping[str, Any],
    step: int,
    *,
    require_fingerprint_eligible: bool = True,
) -> dict[str, Any]:
    matches = [
        item
        for item in fingerprint.get("checkpoints", [])
        if isinstance(item, dict) and item.get("step") == step
    ]
    if len(matches) != 1 or (
        require_fingerprint_eligible and matches[0].get("eligible") is not True
    ):
        raise FinalReplicationError(
            f"selected step {step} is not one eligible fresh-parent checkpoint"
        )
    record = copy.deepcopy(matches[0])
    checkpoint = _regular_file(
        Path(str(record.get("checkpoint", ""))), where=f"step {step} checkpoint"
    )
    if file_sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise FinalReplicationError(f"step {step} checkpoint bytes drifted")
    return record


def _eval_seed_set(report: Mapping[str, Any]) -> list[int]:
    games = report.get("games")
    if not isinstance(games, list):
        raise FinalReplicationError("evaluation report has no games")
    seeds = [int(item["game_seed"]) for item in games if isinstance(item, dict)]
    if len(seeds) != EXPECTED_GAMES or len(set(seeds)) != EXPECTED_PAIRS:
        raise FinalReplicationError("evaluation does not contain exact paired CRN seeds")
    return sorted(set(seeds))


def _panel(
    path: Path,
    *,
    checkpoint_sha256: str,
    parent_sha256: str,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved, report = _load_json(path, where=f"{role} pooled evaluation")
    fleet = report.get("fleet_merge")
    engine = report.get("engine_identity")
    effective = report.get("effective_search_config")
    if (
        not isinstance(fleet, dict)
        or fleet.get("schema_version") != EXPECTED_POOL_SCHEMA
        or not isinstance(engine, dict)
        or engine.get("schema_version") != EXPECTED_ENGINE_SCHEMA
        or not isinstance(effective, dict)
        or report.get("candidate_checkpoint_sha256") != checkpoint_sha256
        or fleet.get("candidate", {}).get("sha256") != checkpoint_sha256
        or report.get("comparison_contract") != EXPECTED_COMPARISON
        or report.get("coherent_public_belief_search") is not True
        or report.get("public_observation") is not True
        or report.get("correct_rust_chance_spectra") is not True
        or report.get("native_mcts_hot_loop") is not True
        or report.get("forced_root_target_mode") != "trajectory_only"
        or report.get("candidate_n_full") != 128
        or report.get("baseline_n_full") != 128
        or report.get("candidate_gameplay_policy_aggregation")
        != "mean_improved_policy"
        or report.get("baseline_gameplay_policy_aggregation")
        != "mean_improved_policy"
        or report.get("errors") != []
        or report.get("games_truncated") != 0
        or report.get("games_played") != EXPECTED_GAMES
        or report.get("games_with_winner") != EXPECTED_GAMES
        or report.get("complete_pairs") != EXPECTED_PAIRS
        or report.get("pairs_requested") != EXPECTED_PAIRS
        or report.get("pairs_truncated_excluded") != 0
    ):
        raise FinalReplicationError(f"{role} panel is not the exact coherent CRN panel")
    baseline_sha = str(report.get("baseline_checkpoint_sha256"))
    if role == "f7" and baseline_sha != parent_sha256:
        raise FinalReplicationError("f7 panel does not use exact learner parent")
    seeds = _eval_seed_set(report)
    summary = {
        "artifact": _artifact(resolved),
        "role": role,
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_wins": int(report["candidate_wins"]),
        "baseline_wins": int(report["baseline_wins"]),
        "candidate_win_rate": float(report["candidate_win_rate"]),
        "complete_pairs": int(report["complete_pairs"]),
        "verdict": str(report["verdict"]),
        "pentanomial_verdict": str(report["pentanomial_sprt"]["decision"]),
        "engine_identity": copy.deepcopy(engine),
        "effective_search_config_sha256": fleet[
            "effective_search_config_sha256"
        ],
        "search_rng_contract": copy.deepcopy(report["search_rng_contract"]),
        "game_seed_set_sha256": value_sha256(seeds),
    }
    return summary, report


def _source_artifact(reference: Mapping[str, Any], *, where: str) -> Path:
    if set(reference) != {"path", "sha256"}:
        raise FinalReplicationError(f"{where} source-reference shape drifted")
    path = _regular_file(Path(str(reference["path"])), where=where)
    if (
        not SHA_PATTERN.fullmatch(str(reference["sha256"]))
        or file_sha256(path) != reference["sha256"]
    ):
        raise FinalReplicationError(f"{where} source bytes drifted")
    return path


def _pair_cohort(
    value: Mapping[str, Any], *, expected_pairs: int, where: str
) -> tuple[dict[int, float], dict[str, Any]]:
    outcomes = value.get("pair_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != expected_pairs:
        raise FinalReplicationError(f"{where} pair outcomes are incomplete")
    by_seed: dict[int, float] = {}
    candidate_wins = 0
    ll_pairs = split_pairs = ww_pairs = 0
    for item in outcomes:
        if not isinstance(item, dict) or set(item) != {
            "candidate_wins",
            "pair_score",
            "seed",
        }:
            raise FinalReplicationError(f"{where} pair-outcome shape drifted")
        wins = item["candidate_wins"]
        seed = item["seed"]
        if (
            isinstance(wins, bool)
            or not isinstance(wins, int)
            or wins not in (0, 1, 2)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed in by_seed
            or float(item["pair_score"]) != wins / 2.0
        ):
            raise FinalReplicationError(f"{where} pair outcome is invalid")
        by_seed[seed] = wins / 2.0
        candidate_wins += wins
        ll_pairs += wins == 0
        split_pairs += wins == 1
        ww_pairs += wins == 2
    baseline_wins = 2 * expected_pairs - candidate_wins
    win_rate = candidate_wins / (2 * expected_pairs)
    diagnostics = value.get("pair_diagnostics")
    if (
        value.get("pairs") != expected_pairs
        or value.get("games") != 2 * expected_pairs
        or value.get("candidate_wins") != candidate_wins
        or value.get("baseline_wins") != baseline_wins
        or abs(float(value.get("candidate_win_rate", -1.0)) - win_rate) > 1e-15
        or diagnostics
        != {
            "incomplete_pairs": 0,
            "ll_pairs": ll_pairs,
            "split_pairs": split_pairs,
            "ww_pairs": ww_pairs,
        }
    ):
        raise FinalReplicationError(f"{where} aggregate does not match pair outcomes")
    for name in ("sprt_minus10_plus15", "superiority_sprt_0_plus15"):
        sprt = value.get(name)
        if (
            not isinstance(sprt, dict)
            or sprt.get("pairs") != expected_pairs
            or abs(float(sprt.get("mean_pair_score", -1.0)) - win_rate) > 1e-15
            or sprt.get("ll_pairs") != ll_pairs
            or sprt.get("split_pairs") != split_pairs
            or sprt.get("ww_pairs") != ww_pairs
        ):
            raise FinalReplicationError(f"{where} {name} does not match outcomes")
    return by_seed, {
        "pairs": expected_pairs,
        "games": 2 * expected_pairs,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "candidate_win_rate": win_rate,
        "sprt_minus10_plus15": value["sprt_minus10_plus15"]["decision"],
        "superiority_sprt_0_plus15": value["superiority_sprt_0_plus15"][
            "decision"
        ],
        "seed_set_sha256": value_sha256(sorted(by_seed)),
    }


def _verify_tiebreak_sources(
    payload: Mapping[str, Any],
    *,
    checkpoint_shas: Mapping[int, str],
    incumbent_sha: str,
) -> dict[str, Any]:
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != EXPECTED_TIEBREAK_SOURCE_KEYS:
        raise FinalReplicationError("tie-break source set drifted")
    effective_hashes: set[str] = set()
    artifacts: dict[str, Any] = {}
    for name, reference in sorted(sources.items()):
        if not isinstance(reference, dict):
            raise FinalReplicationError(f"tie-break source {name} is malformed")
        path = _source_artifact(reference, where=f"tie-break source {name}")
        _source_path, source = _load_json(path, where=f"tie-break source {name}")
        step = 16 if name.startswith("step16") else 8
        if name.endswith("plan"):
            if (
                source.get("schema_version") != "a1-h100-eval-fleet-plan-v2"
                or source.get("operator_mode") != payload.get("operator_mode")
                or source.get("science_config_hash")
                != payload.get("science_config_hash")
                or source.get("repo_commit") != payload.get("repo_commit")
                or source.get("internal_engine_identity")
                != payload.get("internal_engine_identity")
                or source.get("role_search_config")
                != payload.get("role_search_config")
                or source.get("candidate", {}).get("sha256")
                != checkpoint_shas[step]
                or source.get("champion", {}).get("sha256") != incumbent_sha
            ):
                raise FinalReplicationError(f"tie-break plan {name} semantics drifted")
        else:
            fleet = source.get("fleet_merge")
            if (
                source.get("candidate_checkpoint_sha256") != checkpoint_shas[step]
                or source.get("baseline_checkpoint_sha256") != incumbent_sha
                or source.get("comparison_contract") != EXPECTED_COMPARISON
                or source.get("coherent_public_belief_search") is not True
                or source.get("public_observation") is not True
                or source.get("correct_rust_chance_spectra") is not True
                or source.get("native_mcts_hot_loop") is not True
                or source.get("candidate_n_full") != 128
                or source.get("baseline_n_full") != 128
                or source.get("candidate_gameplay_policy_aggregation")
                != "mean_improved_policy"
                or source.get("baseline_gameplay_policy_aggregation")
                != "mean_improved_policy"
                or source.get("errors") != []
                or source.get("games_truncated") != 0
                or source.get("pairs_truncated_excluded") != 0
                or source.get("games_played")
                != 2 * int(source.get("complete_pairs", -1))
                or source.get("engine_identity")
                != payload.get("internal_engine_identity")
                or not isinstance(fleet, dict)
                or not SHA_PATTERN.fullmatch(
                    str(fleet.get("effective_search_config_sha256"))
                )
            ):
                raise FinalReplicationError(f"tie-break report {name} semantics drifted")
            effective_hashes.add(str(fleet["effective_search_config_sha256"]))
        artifacts[name] = _artifact(path)
    if len(effective_hashes) != 1:
        raise FinalReplicationError("tie-break reports used different search operators")
    return {
        "artifacts": artifacts,
        "effective_search_config_sha256": next(iter(effective_hashes)),
    }


def _verify_tiebreak(
    path: Path, *, fingerprint: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved, payload = _load_json(path, where="common-CRN v5 dose tie-break")
    engine = payload.get("internal_engine_identity")
    role_config = payload.get("role_search_config")
    if (
        payload.get("schema_version") != TIEBREAK_SCHEMA
        or payload.get("diagnostic_non_promotable") is not True
        or payload.get("operator_mode") != "coherent_public"
        or not SHA_PATTERN.fullmatch(str(payload.get("science_config_hash")))
        or not isinstance(engine, dict)
        or engine.get("schema_version") != EXPECTED_ENGINE_SCHEMA
        or engine.get("repo_commit") != payload.get("repo_commit")
        or not isinstance(role_config, dict)
        or role_config.get("candidate") != role_config.get("champion")
        or role_config.get("candidate", {}).get("c_scale") != 0.1
        or role_config.get("candidate", {}).get("gameplay_policy_aggregation")
        != "mean_improved_policy"
        or payload.get("selection_result", {}).get("winner")
        != "strategic_step16"
    ):
        raise FinalReplicationError("common-CRN v5 tie-break semantics drifted")
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise FinalReplicationError("tie-break baseline is missing")
    incumbent_sha = str(baseline.get("sha256"))
    incumbent_path = _regular_file(
        Path(str(baseline.get("source", ""))), where="tie-break incumbent"
    )
    if (
        not SHA_PATTERN.fullmatch(incumbent_sha)
        or file_sha256(incumbent_path) != incumbent_sha
    ):
        raise FinalReplicationError("tie-break incumbent bytes drifted")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != {
        "strategic_step8",
        "strategic_step16",
    }:
        raise FinalReplicationError("tie-break finalist set drifted")
    records: dict[int, dict[str, Any]] = {}
    cohort_maps: dict[int, dict[str, dict[int, float]]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for step in (8, 16):
        record = _checkpoint_record(
            fingerprint, step, require_fingerprint_eligible=False
        )
        arm = arms[f"strategic_step{step}"]
        checkpoint = arm.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("sha256") != record.get("checkpoint_sha256")
            or Path(str(checkpoint.get("path", ""))).resolve(strict=True)
            != Path(str(record.get("checkpoint", ""))).resolve(strict=True)
        ):
            raise FinalReplicationError(f"step{step} tie-break checkpoint drifted")
        maps: dict[str, dict[int, float]] = {}
        arm_summaries: dict[str, Any] = {}
        for cohort_name, expected in (
            ("prior_128", EXPECTED_PAIRS),
            ("tie_break_256", EXPECTED_TIEBREAK_PAIRS),
            ("combined_384", EXPECTED_COMBINED_PAIRS),
        ):
            cohort, summary = _pair_cohort(
                arm.get(cohort_name, {}),
                expected_pairs=expected,
                where=f"step{step} {cohort_name}",
            )
            maps[cohort_name] = cohort
            arm_summaries[cohort_name] = summary
        if set(maps["combined_384"]) != (
            set(maps["prior_128"]) | set(maps["tie_break_256"])
        ) or set(maps["prior_128"]) & set(maps["tie_break_256"]):
            raise FinalReplicationError(f"step{step} cohort composition drifted")
        records[step] = record
        cohort_maps[step] = maps
        summaries[step] = arm_summaries
    if records[8].get("eligible") is not True or records[16].get("eligible") is not False:
        raise FinalReplicationError(
            "fingerprint eligibility changed; recipe-only override must remain explicit"
        )
    cohort = payload.get("cohort")
    expected_fresh_seeds = (
        set(range(6_198_726_000, 6_198_726_256))
        - {6_198_726_246}
    ) | {6_198_728_000}
    expected_prior_seeds = set(range(6_198_724_000, 6_198_724_128))
    if (
        not isinstance(cohort, dict)
        or cohort.get("combined_pairs") != EXPECTED_COMBINED_PAIRS
        or cohort.get("final_tiebreak_pairs") != EXPECTED_TIEBREAK_PAIRS
        or cohort.get("prior_interval") != [6_198_724_000, 6_198_724_128]
        or cohort.get("fresh_original_interval")
        != [6_198_726_000, 6_198_726_256]
        or cohort.get("deterministic_truncated_seed") != 6_198_726_246
        or cohort.get("selected_replacement_seed") != 6_198_728_000
        or cohort.get("shared_replacement_block")
        != [6_198_728_000, 6_198_728_004]
        or cohort.get("replacement_selection_rule")
        != "lowest seed in predeclared shared block; selected without inspecting outcomes"
    ):
        raise FinalReplicationError("tie-break cohort contract drifted")
    for step in (8, 16):
        if (
            set(cohort_maps[step]["prior_128"]) != expected_prior_seeds
            or set(cohort_maps[step]["tie_break_256"]) != expected_fresh_seeds
        ):
            raise FinalReplicationError(f"step{step} common-CRN seed cohort drifted")
    if any(
        set(cohort_maps[8][name]) != set(cohort_maps[16][name])
        for name in ("prior_128", "tie_break_256", "combined_384")
    ):
        raise FinalReplicationError("step8/step16 do not share exact CRN cohorts")
    matched = payload.get("matched_comparison")
    if not isinstance(matched, dict):
        raise FinalReplicationError("matched comparison is missing")
    for cohort_name, expected_pairs, expected_lift in (
        ("tie_break_256", EXPECTED_TIEBREAK_PAIRS, EXPECTED_FRESH_LIFT),
        ("combined_384", EXPECTED_COMBINED_PAIRS, EXPECTED_COMBINED_LIFT),
    ):
        actual = matched.get(cohort_name)
        if not isinstance(actual, dict):
            raise FinalReplicationError(f"{cohort_name} matched comparison is missing")
        deltas = [
            {
                "delta": cohort_maps[16][cohort_name][seed]
                - cohort_maps[8][cohort_name][seed],
                "seed": seed,
            }
            for seed in sorted(cohort_maps[8][cohort_name])
        ]
        better16 = sum(item["delta"] > 0 for item in deltas)
        better8 = sum(item["delta"] < 0 for item in deltas)
        same = expected_pairs - better16 - better8
        lift = sum(item["delta"] for item in deltas) / expected_pairs
        if (
            actual.get("pairs") != expected_pairs
            or actual.get("per_seed_delta") != deltas
            or actual.get("step16_better_pairs") != better16
            or actual.get("step8_better_pairs") != better8
            or actual.get("same_pairs") != same
            or abs(
                float(actual.get("step16_minus_step8_mean_pair_score", -1.0))
                - lift
            )
            > 1e-15
            or abs(lift - expected_lift) > 1e-15
        ):
            raise FinalReplicationError(
                f"{cohort_name} matched lift does not replay exactly"
            )
    if (
        summaries[8]["combined_384"]["candidate_wins"]
        != EXPECTED_COMBINED_STEP8_WINS
        or summaries[16]["combined_384"]["candidate_wins"]
        != EXPECTED_COMBINED_STEP16_WINS
        or any(
            summaries[step]["combined_384"]["sprt_minus10_plus15"] != "H0"
            or summaries[step]["combined_384"]["superiority_sprt_0_plus15"]
            != "H0"
            for step in (8, 16)
        )
    ):
        raise FinalReplicationError(
            "robust v5 negative evidence drifted or was incorrectly hidden"
        )
    source_summary = _verify_tiebreak_sources(
        payload,
        checkpoint_shas={
            step: str(records[step]["checkpoint_sha256"]) for step in (8, 16)
        },
        incumbent_sha=incumbent_sha,
    )
    lane_reports = payload.get("step16_fresh_lane_reports")
    if (
        not isinstance(lane_reports, list)
        or len(lane_reports) != 32
        or len(
            {
                (str(item.get("alias")), int(item.get("gpu", -1)))
                for item in lane_reports
                if isinstance(item, dict)
            }
        )
        != 32
        or sum(int(item.get("pairs_requested", -1)) for item in lane_reports)
        != 256
        or sum(int(item.get("complete_pairs", -1)) for item in lane_reports) != 255
        or sum(int(item.get("games_truncated", -1)) for item in lane_reports) != 1
        or any(
            not SHA_PATTERN.fullmatch(str(item.get("sha256")))
            or not str(item.get("path", ""))
            for item in lane_reports
        )
    ):
        raise FinalReplicationError("step16 fresh-lane evidence drifted")
    evidence = {
        "incumbent_checkpoint_sha256": incumbent_sha,
        "candidate_records": records,
        "candidate_summaries": summaries,
        "combined_common_seed_set_sha256": value_sha256(
            sorted(cohort_maps[8]["combined_384"])
        ),
        "fresh_step16_minus_step8": EXPECTED_FRESH_LIFT,
        "combined_step16_minus_step8": EXPECTED_COMBINED_LIFT,
        "source_summary": source_summary,
    }
    return resolved, payload, evidence


def build_adjudication(
    *,
    fingerprint_path: Path,
    tiebreak_adjudication_path: Path,
    selected_f7_report_path: Path,
) -> dict[str, Any]:
    fingerprint_resolved, fingerprint = _fingerprint(fingerprint_path)
    campaign = fingerprint.pop("_verified_campaign")
    campaign_path = fingerprint.pop("_verified_campaign_path")
    tiebreak_path, tiebreak, evidence = _verify_tiebreak(
        tiebreak_adjudication_path, fingerprint=fingerprint
    )
    selected_record = evidence["candidate_records"][16]
    parent_sha = str(campaign["lineage"]["learner_parent_sha256"])
    f7_panel, f7_report = _panel(
        selected_f7_report_path,
        checkpoint_sha256=str(selected_record["checkpoint_sha256"]),
        parent_sha256=parent_sha,
        role="f7",
    )
    if (
        f7_report.get("pentanomial_sprt", {}).get("decision") != "H1"
        or f7_report.get("verdict") != "H1"
    ):
        raise FinalReplicationError("selected dose lacks exact-f7 H1 recovery evidence")
    incumbent_sha = str(evidence["incumbent_checkpoint_sha256"])
    if incumbent_sha == parent_sha:
        raise FinalReplicationError("current-parent correction accidentally reloads f7")
    candidates = []
    for step in (8, 16):
        record = evidence["candidate_records"][step]
        candidates.append(
            {
                "step": step,
                "checkpoint": record["checkpoint"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "fresh_parent_fingerprint_eligible": record["eligible"],
                "combined_v5_panel": evidence["candidate_summaries"][step][
                    "combined_384"
                ],
                "diagnostic_strength_qualified": False,
            }
        )
    value: dict[str, Any] = {
        "schema_version": ADJUDICATION_SCHEMA,
        "selection_rule": EXPECTED_SELECTION_RULE,
        "selection_role": "dose_recipe_only",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "diagnostic_strength_qualification": False,
        "all_f7_start_finalists_failed_current_parent_h0": True,
        "current_parent_correction_required": True,
        "fingerprint": {
            **_artifact(fingerprint_resolved),
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
        },
        "campaign": {
            **_artifact(Path(campaign_path)),
            "campaign_sha256": campaign["campaign_sha256"],
            "arm": EXPECTED_ARM,
        },
        "canonical_common_crn_tiebreak": {
            **_artifact(tiebreak_path),
            "schema_version": tiebreak["schema_version"],
            "science_config_hash": tiebreak["science_config_hash"],
            "effective_search_config_sha256": evidence["source_summary"][
                "effective_search_config_sha256"
            ],
            "combined_common_seed_set_sha256": evidence[
                "combined_common_seed_set_sha256"
            ],
        },
        "exact_parent_checkpoint_sha256": parent_sha,
        "incumbent_checkpoint_sha256": incumbent_sha,
        "candidates": candidates,
        "selected_f7_recovery_panel": f7_panel,
        "selected": {
            "step": 16,
            "checkpoint": selected_record["checkpoint"],
            "checkpoint_sha256": selected_record["checkpoint_sha256"],
            "combined_v5_win_rate": evidence["candidate_summaries"][16][
                "combined_384"
            ]["candidate_win_rate"],
            "combined_matched_lift_over_step8": evidence[
                "combined_step16_minus_step8"
            ],
            "fresh_matched_lift_over_step8": evidence[
                "fresh_step16_minus_step8"
            ],
            "recipe_override_of_fingerprint_eligibility": True,
        },
        "selected_checkpoint_is_initializer": False,
        "selected_checkpoint_is_promotion_evidence": False,
        "final_replication_must_reload_exact_parent": False,
        "final_replication_must_reload_current_parent": True,
        "final_replication_strength_claim_pending": True,
    }
    value["adjudication_sha256"] = value_sha256(value)
    return value


def verify_adjudication(path: Path) -> dict[str, Any]:
    payload = _sealed(
        path,
        schema=ADJUDICATION_SCHEMA,
        digest_field="adjudication_sha256",
        where="Stage-C recipe adjudication",
    )
    if (
        payload.get("selection_rule") != EXPECTED_SELECTION_RULE
        or payload.get("selection_role") != "dose_recipe_only"
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("diagnostic_strength_qualification") is not False
        or payload.get("all_f7_start_finalists_failed_current_parent_h0") is not True
        or payload.get("current_parent_correction_required") is not True
        or payload.get("selected_checkpoint_is_initializer") is not False
        or payload.get("selected_checkpoint_is_promotion_evidence") is not False
        or payload.get("final_replication_must_reload_exact_parent") is not False
        or payload.get("final_replication_must_reload_current_parent") is not True
        or payload.get("final_replication_strength_claim_pending") is not True
    ):
        raise FinalReplicationError("recipe adjudication role drifted")
    fingerprint_path = _verify_artifact(payload["fingerprint"], where="fingerprint")
    tiebreak_path = _verify_artifact(
        payload["canonical_common_crn_tiebreak"], where="common-CRN tie-break"
    )
    rebuilt = build_adjudication(
        fingerprint_path=fingerprint_path,
        tiebreak_adjudication_path=tiebreak_path,
        selected_f7_report_path=Path(
            payload["selected_f7_recovery_panel"]["artifact"]["path"]
        ),
    )
    if rebuilt != payload:
        raise FinalReplicationError("recipe adjudication does not replay")
    return copy.deepcopy(payload)


def _subset_arrays(path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    resolved = _regular_file(path, where="root subset")
    try:
        with np.load(resolved, allow_pickle=False) as source:
            arrays = {name: np.asarray(source[name]) for name in source.files}
    except (OSError, ValueError) as error:
        raise FinalReplicationError(f"cannot read root subset: {error}") from error
    if not {"identity_sha256", "game_seed"} <= set(arrays):
        raise FinalReplicationError("root subset lacks identity_sha256/game_seed")
    count = len(arrays["identity_sha256"])
    if any(len(value) != count for value in arrays.values()):
        raise FinalReplicationError("root subset columns are misaligned")
    return resolved, arrays


def _set_sha(values: Sequence[object]) -> str:
    return value_sha256(sorted({str(value) for value in values}))


def build_root_manifest(
    *,
    production_plan_path: Path,
    prep_inventory_path: Path,
    forbidden_subset_paths: Sequence[Path],
    forbidden_eval_paths: Sequence[Path],
) -> dict[str, Any]:
    try:
        plan = alignment._verify_plan(production_plan_path)  # noqa: SLF001
    except alignment.AlignmentError as error:
        raise FinalReplicationError(f"final reanalysis plan refused: {error}") from error
    subset_ref = plan.get("subset", {}).get("artifact")
    if not isinstance(subset_ref, dict):
        raise FinalReplicationError("final reanalysis plan lost selected roots")
    subset_path, arrays = _subset_arrays(Path(str(subset_ref["path"])))
    identities = np.asarray(arrays["identity_sha256"]).astype(str)
    game_seeds = np.asarray(arrays["game_seed"], dtype=np.int64)
    prep_path, prep = _load_json(
        prep_inventory_path, where="independent-root preparation evidence"
    )
    prep_unsigned = dict(prep)
    prep_stated = prep_unsigned.pop("inventory_sha256", None)
    ready = prep.get("fully_reconstructable_ready_roots")
    proof = prep.get("proof")
    if (
        prep.get("schema_version") != PREP_INVENTORY_SCHEMA
        or prep_stated != value_sha256(prep_unsigned)
        or not isinstance(ready, list)
        or len(ready) < EXPECTED_ROOTS
        or not isinstance(proof, dict)
        or proof.get("satisfied") is not True
        or proof.get("independent_from_all_declared_eval_pair_seeds") is not True
        or proof.get("independent_from_diagnostic_selected_rows") is not True
        or proof.get("independent_from_learner_holdout_games") is not True
        or int(proof.get("required_fully_reconstructable_strategic_roots", -1))
        != EXPECTED_ROOTS
        or int(proof.get("observed_fully_reconstructable_strategic_roots", -1))
        < EXPECTED_ROOTS
        or proof.get("first_8192_ready_root_key_set_sha256")
        != value_sha256(ready[:EXPECTED_ROOTS])
        or prep.get("authority", {}).get("is_authority") is not False
        or prep.get("authority", {}).get("may_launch_search") is not False
    ):
        raise FinalReplicationError("independent-root preparation evidence drifted")
    required_ready_columns = {
        "row_index": np.asarray(arrays.get("row_index", []), dtype=np.int64),
        "game_seed": game_seeds,
        "decision_index": np.asarray(arrays.get("decision_index", []), dtype=np.int64),
        "identity_sha256": identities,
    }
    if any(value.shape != (EXPECTED_ROOTS,) for value in required_ready_columns.values()):
        raise FinalReplicationError("final subset lacks exact prepared root columns")
    for position, prepared in enumerate(ready[:EXPECTED_ROOTS]):
        if any(
            str(required_ready_columns[name][position]) != str(prepared[name])
            for name in required_ready_columns
        ):
            raise FinalReplicationError(
                "final plan does not consume the first 8,192 replay-qualified roots"
            )
    if (
        len(identities) != EXPECTED_ROOTS
        or np.unique(identities).size != EXPECTED_ROOTS
        or int(plan["subset"].get("selected_rows", -1)) != EXPECTED_ROOTS
        or int(plan["subset"].get("requested_rows", -1)) != EXPECTED_ROOTS
        or int(plan["subset"].get("chunks", -1)) != EXPECTED_PARTITIONS
        or plan.get("execution", {}).get("executor_semantics") != EXPECTED_EXECUTOR
        or plan.get("target_policy_target_identity", {}).get(
            "target_information_regime"
        )
        != EXPECTED_OPERATOR
        or int(
            plan.get("target_policy_target_identity", {})
            .get("search_operator", {})
            .get("n_full", -1)
        )
        != 128
    ):
        raise FinalReplicationError("final plan is not the exact 8,192-root/64-way n128 slice")
    forbidden_identity: set[str] = set()
    forbidden_games: set[int] = set()
    forbidden_subsets = []
    for path in forbidden_subset_paths:
        resolved, values = _subset_arrays(path)
        forbidden_identity.update(np.asarray(values["identity_sha256"]).astype(str))
        forbidden_games.update(np.asarray(values["game_seed"], dtype=np.int64).tolist())
        forbidden_subsets.append(_artifact(resolved))
    eval_seeds: set[int] = set()
    forbidden_evals = []
    for path in forbidden_eval_paths:
        resolved, report = _load_json(path, where="forbidden evaluation report")
        eval_seeds.update(_eval_seed_set(report))
        forbidden_evals.append(_artifact(resolved))
    root_overlap = len(set(identities.tolist()) & forbidden_identity)
    game_overlap = len(set(game_seeds.tolist()) & forbidden_games)
    eval_overlap = len(set(game_seeds.tolist()) & eval_seeds)
    if root_overlap or game_overlap or eval_overlap:
        raise FinalReplicationError(
            "final root slice overlaps diagnostics/evaluation: "
            f"roots={root_overlap} games={game_overlap} eval={eval_overlap}"
        )
    value: dict[str, Any] = {
        "schema_version": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_stage_c_production_replication_roots",
        "production_plan": {
            **_artifact(production_plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
        "preparation_evidence": {
            **_artifact(prep_path),
            "inventory_sha256": prep_stated,
            "first_8192_ready_root_key_set_sha256": proof[
                "first_8192_ready_root_key_set_sha256"
            ],
            "is_authority": False,
        },
        "subset": _artifact(subset_path),
        "root_count": EXPECTED_ROOTS,
        "partition_count": EXPECTED_PARTITIONS,
        "root_identity_set_sha256": _set_sha(identities.tolist()),
        "game_seed_set_sha256": _set_sha(game_seeds.tolist()),
        "unique_game_seeds": int(np.unique(game_seeds).size),
        "selection_seed": int(plan["subset"]["selection_seed"]),
        "rng_domain": "a1-stage-c-final-replication-v1",
        "forbidden_diagnostic_subsets": forbidden_subsets,
        "forbidden_evaluation_reports": forbidden_evals,
        "diagnostic_root_overlap_count": root_overlap,
        "diagnostic_game_seed_overlap_count": game_overlap,
        "evaluation_game_seed_overlap_count": eval_overlap,
        "diagnostic_target_bytes_reused": False,
        "diagnostic_checkpoint_used_as_initializer": False,
    }
    value["root_manifest_sha256"] = value_sha256(value)
    return value


def verify_root_manifest(path: Path) -> dict[str, Any]:
    payload = _sealed(
        path,
        schema=ROOT_MANIFEST_SCHEMA,
        digest_field="root_manifest_sha256",
        where="Stage-C independent root manifest",
    )
    rebuilt = build_root_manifest(
        production_plan_path=Path(payload["production_plan"]["path"]),
        prep_inventory_path=Path(payload["preparation_evidence"]["path"]),
        forbidden_subset_paths=[
            Path(item["path"]) for item in payload["forbidden_diagnostic_subsets"]
        ],
        forbidden_eval_paths=[
            Path(item["path"]) for item in payload["forbidden_evaluation_reports"]
        ],
    )
    if rebuilt != payload:
        raise FinalReplicationError("independent root manifest does not replay")
    return copy.deepcopy(payload)


def build_final_corpus_admission(
    *, root_manifest_path: Path, merge_receipt_path: Path, overlay_admission_path: Path
) -> dict[str, Any]:
    roots = verify_root_manifest(root_manifest_path)
    try:
        merge = reanalysis._verify_merge_receipt(merge_receipt_path)  # noqa: SLF001
        overlay_result = overlay.verify_overlay_admission(overlay_admission_path)
    except (reanalysis.ExecutorError, overlay.OverlayError) as error:
        raise FinalReplicationError(f"fresh final corpus refused: {error}") from error
    admission = overlay_result["admission"]
    receipt = overlay_result["receipt"]
    export_path = Path(str(receipt["export"]["path"]))
    export_path_resolved, export, _patch, subset = overlay._load_export(export_path)  # noqa: SLF001
    merge_ref = export["source_merge_receipt"]
    production_plan = roots["production_plan"]
    subset_identities = np.asarray(subset["identity_sha256"]).astype(str)
    required_value_columns = {
        "root_value",
        "root_value_mask",
        *reanalysis.TARGET_RELIABILITY_COLUMNS,
    }
    projected_value_columns = set(
        receipt.get("projection", {}).get("authoritative_search_fixed_columns", [])
    )
    patch_columns = set(merge.get("patch_columns", []))
    learner_projection = export.get("learner_projection", {})
    if (
        merge_ref.get("receipt_sha256") != merge.get("receipt_sha256")
        or merge.get("stage_c_plan", {}).get("plan_sha256")
        != production_plan.get("plan_sha256")
        or export.get("target_policy_target_identity_sha256")
        != merge.get("target_policy_target_identity_sha256")
        or export.get("target_reanalyzer_checkpoint", {}).get("sha256")
        != admission.get("corpus", {}).get("producer_checkpoint_sha256")
        or int(receipt.get("projection", {}).get("selected_rows", -1))
        != EXPECTED_ROOTS
        or len(subset_identities) != EXPECTED_ROOTS
        or _set_sha(subset_identities.tolist()) != roots["root_identity_set_sha256"]
        or admission.get("stage_c_policy_overlay", {}).get(
            "historical_policy_targets_active"
        )
        is not False
        or admission.get("policy_distillation_contract", {}).get(
            "stage_c_reanalysis_only"
        )
        is not True
        or not required_value_columns <= projected_value_columns
        or not {
            "root_value",
            "root_value_mask",
            "completed_q_values_flat",
            "completed_q_mask_flat",
            *reanalysis.TARGET_RELIABILITY_COLUMNS,
        }
        <= patch_columns
        or learner_projection.get("root_value_patch_consumed") is not True
        or learner_projection.get("target_reliability_patch_consumed") is not True
        or learner_projection.get("completed_q_evidence_sidecar_preserved") is not True
        or merge.get("reliability", {}).get("schema_version")
        != reanalysis.TARGET_RELIABILITY_SCHEMA
    ):
        raise FinalReplicationError(
            "final corpus does not derive exactly from fresh independent target bytes"
        )
    value: dict[str, Any] = {
        "schema_version": FINAL_CORPUS_ADMISSION_SCHEMA,
        "status": "admitted_for_independent_stage_c_final_replication",
        "diagnostic_only": False,
        "promotion_eligible": False,
        "promotion_eligible_after_full_gate": True,
        "full_gate_required": True,
        "auto_promotion": False,
        "root_manifest": {
            **_artifact(root_manifest_path),
            "root_manifest_sha256": roots["root_manifest_sha256"],
        },
        "merge_receipt": {
            **_artifact(merge_receipt_path),
            "receipt_sha256": merge["receipt_sha256"],
        },
        "low_level_overlay_admission": {
            **_artifact(overlay_admission_path),
            "admission_sha256": admission["admission_sha256"],
        },
        "corpus": copy.deepcopy(admission["corpus"]),
        "contract": copy.deepcopy(admission["contract"]),
        "policy_distillation_contract": copy.deepcopy(
            admission["policy_distillation_contract"]
        ),
        "target_policy_target_identity_sha256": merge[
            "target_policy_target_identity_sha256"
        ],
        "target_reanalyzer_checkpoint": copy.deepcopy(
            merge["target_reanalyzer_checkpoint"]
        ),
        "operator_contract": copy.deepcopy(merge["target_operator_contract"]),
        "search_value_evidence": {
            "root_value_columns_materialized": sorted(
                {"root_value", "root_value_mask"}
            ),
            "target_reliability_columns_materialized": sorted(
                reanalysis.TARGET_RELIABILITY_COLUMNS
            ),
            "completed_q_columns_preserved_in_immutable_patch_sidecar": [
                "completed_q_mask_flat",
                "completed_q_values_flat",
            ],
            "reliability": copy.deepcopy(merge["reliability"]),
            "naive_root_blend_authorized": False,
            "terminal_target_remains_authoritative": True,
        },
        "fresh_independent_target_bytes": True,
        "diagnostic_overlay_or_target_bytes_reused": False,
        "low_level_diagnostic_schema_used_only_as_transform": True,
        "export_manifest": _artifact(export_path_resolved),
    }
    value["admission_sha256"] = value_sha256(value)
    return value


def verify_final_corpus_admission(path: Path) -> dict[str, Any]:
    payload = _sealed(
        path,
        schema=FINAL_CORPUS_ADMISSION_SCHEMA,
        digest_field="admission_sha256",
        where="Stage-C final corpus admission",
    )
    if (
        payload.get("diagnostic_only") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_eligible_after_full_gate") is not True
        or payload.get("full_gate_required") is not True
        or payload.get("auto_promotion") is not False
        or payload.get("fresh_independent_target_bytes") is not True
        or payload.get("diagnostic_overlay_or_target_bytes_reused") is not False
    ):
        raise FinalReplicationError("Stage-C final corpus role drifted")
    rebuilt = build_final_corpus_admission(
        root_manifest_path=Path(payload["root_manifest"]["path"]),
        merge_receipt_path=Path(payload["merge_receipt"]["path"]),
        overlay_admission_path=Path(payload["low_level_overlay_admission"]["path"]),
    )
    if rebuilt != payload:
        raise FinalReplicationError("Stage-C final corpus admission does not replay")
    result = copy.deepcopy(payload)
    _overlay_path, coherent = _load_json(
        Path(payload["low_level_overlay_admission"]["path"]),
        where="low-level coherent admission",
    )
    result["_coherent_admission"] = coherent
    return result


def build_final_authority(
    *,
    adjudication_path: Path,
    corpus_admission_path: Path,
    architecture_upgrade_receipt_path: Path,
    reviewed_code_tree_sha256: str,
    reviewed_lock_path: Path,
    sampler_seed: int,
) -> dict[str, Any]:
    adjudication = verify_adjudication(adjudication_path)
    corpus = verify_final_corpus_admission(corpus_admission_path)
    corpus.pop("_coherent_admission", None)
    try:
        upgrade = architecture_upgrade.verify_receipt(architecture_upgrade_receipt_path)
    except architecture_upgrade.UpgradeError as error:
        raise FinalReplicationError(f"architecture initializer refused: {error}") from error
    if not SHA_PATTERN.fullmatch(reviewed_code_tree_sha256):
        raise FinalReplicationError("reviewed code-tree sha256 is malformed")
    lock_path = _regular_file(reviewed_lock_path, where="reviewed Stage-C lock")
    selected = adjudication["selected"]
    fingerprint_path = Path(adjudication["fingerprint"]["path"])
    _fingerprint_path, fingerprint = _fingerprint(fingerprint_path)
    campaign = fingerprint.pop("_verified_campaign")
    fingerprint.pop("_verified_campaign_path")
    record = _checkpoint_record(
        fingerprint,
        int(selected["step"]),
        require_fingerprint_eligible=False,
    )
    # The diagnostic campaign deliberately started from f7 so the two external
    # panels can measure recovery versus f7 and breadth versus the authoritative
    # incumbent.  FINAL is a new flywheel dose, not a replay of that diagnostic
    # initializer: both its coherent teacher and learner must reload the current
    # incumbent identified by every authenticated incumbent panel.
    current_parent_sha = str(adjudication["incumbent_checkpoint_sha256"])
    if (
        selected["checkpoint_sha256"] != record["checkpoint_sha256"]
        or upgrade["source"]["sha256"] != current_parent_sha
        or corpus["target_reanalyzer_checkpoint"]["sha256"] != current_parent_sha
        or corpus["corpus"]["producer_checkpoint_sha256"] != current_parent_sha
        or upgrade.get("forward_identical_at_init") is not True
        or upgrade.get("shared_parameters_bit_identical") is not True
        or float(upgrade.get("forward_max_diff", -1.0)) != 0.0
    ):
        raise FinalReplicationError(
            "final replication does not independently reload the exact current "
            "parent for teacher and learner"
        )
    if isinstance(sampler_seed, bool) or sampler_seed < 0:
        raise FinalReplicationError("final sampler seed must be non-negative")
    recipe = copy.deepcopy(campaign["recipe"])
    recipe["max_steps"] = 32
    recipe["sampler_seed"] = int(sampler_seed)
    control_recipe = copy.deepcopy(recipe)
    control_recipe["value_lr_mult"] = 0.3
    control_recipe["value_trunk_grad_scale"] = 0.1
    control_recipe["trunk_lr_mult"] = 1.0
    value_repair_recipe = copy.deepcopy(recipe)
    value_repair_recipe["value_lr_mult"] = 1.0
    value_repair_recipe["value_trunk_grad_scale"] = 1.0
    value_repair_recipe["trunk_lr_mult"] = 1.0
    trunk_quarter_recipe = copy.deepcopy(control_recipe)
    trunk_quarter_recipe["trunk_lr_mult"] = 0.25
    trunk_tenth_recipe = copy.deepcopy(control_recipe)
    trunk_tenth_recipe["trunk_lr_mult"] = 0.1
    matched_arms = {
        FINAL_CONTROL_ARM: {
            "role": "exact_v5_terminal_value_control",
            "recipe": control_recipe,
            "recipe_sha256": value_sha256(control_recipe),
            "value_target": "terminal_outcome_only",
        },
        FINAL_VALUE_REPAIR_ARM: {
            "role": "repair_value_learning_under_policy_driven_trunk_movement",
            "recipe": value_repair_recipe,
            "recipe_sha256": value_sha256(value_repair_recipe),
            "value_target": "terminal_outcome_only",
        },
        FINAL_TRUNK_QUARTER_ARM: {
            "role": "protect_mature_v5_trunk_at_quarter_learning_rate",
            "recipe": trunk_quarter_recipe,
            "recipe_sha256": value_sha256(trunk_quarter_recipe),
            "value_target": "terminal_outcome_only",
        },
        FINAL_TRUNK_TENTH_ARM: {
            "role": "protect_mature_v5_trunk_at_tenth_learning_rate",
            "recipe": trunk_tenth_recipe,
            "recipe_sha256": value_sha256(trunk_tenth_recipe),
            "value_target": "terminal_outcome_only",
        },
    }
    value: dict[str, Any] = {
        "schema_version": FINAL_AUTHORITY_SCHEMA,
        "purpose": "independent_stage_c_selected_dose_matched_value_repair",
        "external_adjudication": {
            **_artifact(adjudication_path),
            "adjudication_sha256": adjudication["adjudication_sha256"],
        },
        "final_corpus_admission": {
            **_artifact(corpus_admission_path),
            "admission_sha256": corpus["admission_sha256"],
        },
        "diagnostic_selection": {
            "arm": EXPECTED_ARM,
            "selected_step": int(selected["step"]),
            "selected_diagnostic_checkpoint_sha256": selected[
                "checkpoint_sha256"
            ],
            "selected_diagnostic_checkpoint_loaded": False,
            "selection_rule": adjudication["selection_rule"],
        },
        "initializer": {
            "exact_parent": copy.deepcopy(upgrade["source"]),
            "function_preserving_upgrade_receipt": _artifact(
                architecture_upgrade_receipt_path
            ),
            "upgrade_receipt_sha256": upgrade["receipt_sha256"],
            "upgraded_initializer": copy.deepcopy(upgrade["upgraded_initializer"]),
            "fresh_adam": True,
            "resume_optimizer": False,
            "candidate_chaining": False,
        },
        "training": {
            "matched_arms": matched_arms,
            "matched_arm_names": list(FINAL_ARM_NAMES),
            "matched_sample_order": True,
            "independent_fresh_adam_per_arm": True,
            "diagnostic_recipe_selected_step": int(selected["step"]),
            "max_optimizer_steps": 32,
            "checkpoint_steps": [8, 12, 16, 32],
            "topology": {
                "name": "b200-8gpu-ddp",
                "world_size": 8,
                "local_batch_size": 512,
                "global_batch_size": 4096,
            },
        },
        "value_treatment": {
            "root_value_blend_enabled": False,
            "value_target_lambda": 1.0,
            "reason": (
                "coherent root values currently have worse terminal MSE/calibration "
                "and cover too little learner mass; repair value gradient routing first"
            ),
            "control": {
                "value_lr_mult": 0.3,
                "value_trunk_grad_scale": 0.1,
            },
            "treatment": {
                "value_lr_mult": 1.0,
                "value_trunk_grad_scale": 1.0,
            },
            "trunk_protection": [
                {
                    "trunk_lr_mult": 0.25,
                    "value_lr_mult": 0.3,
                    "value_trunk_grad_scale": 0.1,
                },
                {
                    "trunk_lr_mult": 0.1,
                    "value_lr_mult": 0.3,
                    "value_trunk_grad_scale": 0.1,
                },
            ],
            "all_other_recipe_fields_matched": True,
            "selection_requires_parent_anchor_kl_value_calibration_and_external_v5": True,
        },
        "reviewed_code": {
            "code_tree_sha256": reviewed_code_tree_sha256,
            "lock": _artifact(lock_path),
        },
        "diagnostic_only": False,
        "promotion_eligible": False,
        "promotion_eligible_after_full_gate": True,
        "full_gate_required": True,
        "matched_external_v5_panel_required": True,
        "matched_value_calibration_required": True,
        "auto_promotion": False,
    }
    value["authority_sha256"] = value_sha256(value)
    return value


def verify_final_authority(path: Path) -> dict[str, Any]:
    payload = _sealed(
        path,
        schema=FINAL_AUTHORITY_SCHEMA,
        digest_field="authority_sha256",
        where="Stage-C final replication authority",
    )
    matched_arms = payload.get("training", {}).get("matched_arms")
    if not isinstance(matched_arms, dict):
        raise FinalReplicationError("Stage-C final matched arms are missing")
    if set(matched_arms) != set(FINAL_ARM_NAMES) or any(
        not isinstance(matched_arms.get(name), dict) for name in FINAL_ARM_NAMES
    ):
        raise FinalReplicationError("Stage-C final matched arm identity drifted")
    control = matched_arms[FINAL_CONTROL_ARM]
    treatment = matched_arms[FINAL_VALUE_REPAIR_ARM]
    quarter = matched_arms[FINAL_TRUNK_QUARTER_ARM]
    tenth = matched_arms[FINAL_TRUNK_TENTH_ARM]
    control_recipe = control.get("recipe")
    treatment_recipe = treatment.get("recipe")
    quarter_recipe = quarter.get("recipe")
    tenth_recipe = tenth.get("recipe")
    if any(
        not isinstance(recipe, dict)
        for recipe in (
            control_recipe,
            treatment_recipe,
            quarter_recipe,
            tenth_recipe,
        )
    ):
        raise FinalReplicationError("Stage-C final matched recipes are malformed")
    assert isinstance(control_recipe, dict)
    assert isinstance(treatment_recipe, dict)
    assert isinstance(quarter_recipe, dict)
    assert isinstance(tenth_recipe, dict)

    def differing(recipe: Mapping[str, Any]) -> set[str]:
        return {
            key
            for key in set(control_recipe) | set(recipe)
            if control_recipe.get(key) != recipe.get(key)
        }

    if (
        payload.get("diagnostic_only") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_eligible_after_full_gate") is not True
        or payload.get("full_gate_required") is not True
        or payload.get("auto_promotion") is not False
        or payload.get("matched_external_v5_panel_required") is not True
        or payload.get("matched_value_calibration_required") is not True
        or payload.get("training", {}).get("matched_arm_names")
        != list(FINAL_ARM_NAMES)
        or payload.get("training", {}).get("matched_sample_order") is not True
        or payload.get("training", {}).get("independent_fresh_adam_per_arm")
        is not True
        or payload.get("training", {}).get("diagnostic_recipe_selected_step") != 16
        or payload.get("training", {}).get("max_optimizer_steps") != 32
        or payload.get("training", {}).get("checkpoint_steps") != [8, 12, 16, 32]
        or differing(treatment_recipe)
        != {"value_lr_mult", "value_trunk_grad_scale"}
        or differing(quarter_recipe) != {"trunk_lr_mult"}
        or differing(tenth_recipe) != {"trunk_lr_mult"}
        or control_recipe.get("value_lr_mult") != 0.3
        or control_recipe.get("value_trunk_grad_scale") != 0.1
        or control_recipe.get("trunk_lr_mult") != 1.0
        or treatment_recipe.get("value_lr_mult") != 1.0
        or treatment_recipe.get("value_trunk_grad_scale") != 1.0
        or treatment_recipe.get("trunk_lr_mult") != 1.0
        or quarter_recipe.get("trunk_lr_mult") != 0.25
        or tenth_recipe.get("trunk_lr_mult") != 0.1
        or any(
            matched_arms[name].get("recipe_sha256")
            != value_sha256(matched_arms[name]["recipe"])
            or matched_arms[name].get("value_target")
            != "terminal_outcome_only"
            for name in FINAL_ARM_NAMES
        )
        or payload.get("value_treatment", {}).get("root_value_blend_enabled")
        is not False
        or payload.get("value_treatment", {}).get("value_target_lambda") != 1.0
        or payload.get("value_treatment", {}).get(
            "selection_requires_parent_anchor_kl_value_calibration_and_external_v5"
        )
        is not True
        or payload.get("diagnostic_selection", {}).get(
            "selected_diagnostic_checkpoint_loaded"
        )
        is not False
    ):
        raise FinalReplicationError("Stage-C final authority role drifted")
    rebuilt = build_final_authority(
        adjudication_path=Path(payload["external_adjudication"]["path"]),
        corpus_admission_path=Path(payload["final_corpus_admission"]["path"]),
        architecture_upgrade_receipt_path=Path(
            payload["initializer"]["function_preserving_upgrade_receipt"]["path"]
        ),
        reviewed_code_tree_sha256=payload["reviewed_code"]["code_tree_sha256"],
        reviewed_lock_path=Path(payload["reviewed_code"]["lock"]["path"]),
        sampler_seed=int(control_recipe["sampler_seed"]),
    )
    if rebuilt != payload:
        raise FinalReplicationError("Stage-C final authority does not replay")
    return copy.deepcopy(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    adjudicate = commands.add_parser("adjudicate")
    adjudicate.add_argument("--fingerprint", required=True, type=Path)
    adjudicate.add_argument("--tiebreak-adjudication", required=True, type=Path)
    adjudicate.add_argument("--selected-f7-report", required=True, type=Path)
    adjudicate.add_argument("--write", required=True, type=Path)

    seal_roots = commands.add_parser("seal-roots")
    seal_roots.add_argument("--production-plan", required=True, type=Path)
    seal_roots.add_argument("--prep-inventory", required=True, type=Path)
    seal_roots.add_argument(
        "--forbidden-subset", action="append", required=True, type=Path
    )
    seal_roots.add_argument(
        "--forbidden-eval-report", action="append", required=True, type=Path
    )
    seal_roots.add_argument("--write", required=True, type=Path)

    admit = commands.add_parser("admit-corpus")
    admit.add_argument("--root-manifest", required=True, type=Path)
    admit.add_argument("--merge-receipt", required=True, type=Path)
    admit.add_argument("--overlay-admission", required=True, type=Path)
    admit.add_argument("--write", required=True, type=Path)

    issue = commands.add_parser("issue")
    issue.add_argument("--adjudication", required=True, type=Path)
    issue.add_argument("--corpus-admission", required=True, type=Path)
    issue.add_argument("--architecture-upgrade-receipt", required=True, type=Path)
    issue.add_argument("--reviewed-code-tree-sha256", required=True)
    issue.add_argument("--reviewed-lock", required=True, type=Path)
    issue.add_argument("--sampler-seed", required=True, type=int)
    issue.add_argument("--write", required=True, type=Path)

    verify = commands.add_parser("verify")
    verify.add_argument(
        "--kind", required=True, choices=("adjudication", "roots", "corpus", "authority")
    )
    verify.add_argument("--path", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "adjudicate":
            result = build_adjudication(
                fingerprint_path=args.fingerprint,
                tiebreak_adjudication_path=args.tiebreak_adjudication,
                selected_f7_report_path=args.selected_f7_report,
            )
            _write_json_immutable(args.write, result)
        elif args.command == "seal-roots":
            result = build_root_manifest(
                production_plan_path=args.production_plan,
                prep_inventory_path=args.prep_inventory,
                forbidden_subset_paths=args.forbidden_subset,
                forbidden_eval_paths=args.forbidden_eval_report,
            )
            _write_json_immutable(args.write, result)
        elif args.command == "admit-corpus":
            result = build_final_corpus_admission(
                root_manifest_path=args.root_manifest,
                merge_receipt_path=args.merge_receipt,
                overlay_admission_path=args.overlay_admission,
            )
            _write_json_immutable(args.write, result)
        elif args.command == "issue":
            result = build_final_authority(
                adjudication_path=args.adjudication,
                corpus_admission_path=args.corpus_admission,
                architecture_upgrade_receipt_path=args.architecture_upgrade_receipt,
                reviewed_code_tree_sha256=args.reviewed_code_tree_sha256,
                reviewed_lock_path=args.reviewed_lock,
                sampler_seed=args.sampler_seed,
            )
            _write_json_immutable(args.write, result)
        else:
            verifier = {
                "adjudication": verify_adjudication,
                "roots": verify_root_manifest,
                "corpus": verify_final_corpus_admission,
                "authority": verify_final_authority,
            }[args.kind]
            result = verifier(args.path)
            result.pop("_coherent_admission", None)
    except (
        FinalReplicationError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Stage-C final replication refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
