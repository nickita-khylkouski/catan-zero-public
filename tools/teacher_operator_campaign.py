#!/usr/bin/env python3
"""Run and aggregate the coherent-public n128/adaptive-n256 teacher campaign.

The campaign is deliberately about one causal question, not checkpoint
strength: both seats use the *same* checkpoint and every operator field is
identical except the adaptive wide-root budget.  The fixed-root half measures
target stability and attributable cost on the same real roots.  The paired-game
half measures whether that target change actually improves play after exact
seed/color swaps.

Stages are independently runnable so the two paired-game comparisons can be
sent to separate GPU hosts.  ``all`` is a convenient single-host execution;
``render`` prints the exact commands without loading a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _REPO_ROOT / "tools"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from fixed_root_search_stability import (  # noqa: E402
    REPORT_SCHEMA as FIXED_ROOT_REPORT_SCHEMA,
    RootStratumQuota,
    _comparison_slice,
    content_sha256,
    enforce_root_stratum_quotas,
    load_evaluator_spec,
    load_search_spec,
    root_phase_width_summary,
    validate_search_comparison,
)
from factory_common import write_json  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402


CAMPAIGN_SCHEMA = "teacher-operator-causal-campaign-v1"
REPORT_SCHEMA = "teacher-operator-causal-report-v1"
STAGES = ("fixed-w20", "fixed-w40", "h2h-w20", "h2h-w40")
_ARM_BY_STAGE = {
    "fixed-w20": "adaptive_n256_w20_d6",
    "h2h-w20": "adaptive_n256_w20_d6",
    "fixed-w40": "adaptive_n256_w40_d6",
    "h2h-w40": "adaptive_n256_w40_d6",
}


class CampaignError(ValueError):
    """The teacher comparison is incomplete or semantically inconsistent."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{path} must contain a JSON object")
    return value


def _resolve(source: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise CampaignError("campaign file reference must be a non-empty string")
    path = Path(raw).expanduser()
    return (
        (source.parent / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normalize_sha256(raw: str) -> str:
    value = str(raw).lower()
    if not value.startswith("sha256:"):
        value = "sha256:" + value
    if len(value) != 71 or any(char not in "0123456789abcdef" for char in value[7:]):
        raise CampaignError("--checkpoint-sha256 must be a full SHA-256 digest")
    return value


def load_campaign(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    campaign = _load_object(source)
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise CampaignError(f"campaign schema must be {CAMPAIGN_SCHEMA!r}")
    if campaign.get("track") != "2p_no_trade":
        raise CampaignError("teacher campaign is restricted to 2p_no_trade")
    if campaign.get("checkpoint_contract") != "same_checkpoint_both_roles":
        raise CampaignError("teacher campaign must compare one checkpoint to itself")

    fixed_protocol = campaign.get("fixed_root_protocol")
    if not isinstance(fixed_protocol, dict):
        raise CampaignError("campaign must define fixed_root_protocol")
    raw_quotas = fixed_protocol.get("root_stratum_quotas")
    if not isinstance(raw_quotas, list):
        raise CampaignError("fixed-root protocol must preregister root_stratum_quotas")
    quota_contract: dict[tuple[str, int, int | None], int] = {}
    for raw in raw_quotas:
        if not isinstance(raw, dict):
            raise CampaignError("each root stratum quota must be an object")
        try:
            key = (
                str(raw["phase"]),
                int(raw["min_legal_width"]),
                (
                    None
                    if raw.get("max_legal_width") is None
                    else int(raw["max_legal_width"])
                ),
            )
            count = int(raw["count"])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError("invalid root stratum quota") from error
        if key in quota_contract:
            raise CampaignError(f"duplicate root stratum quota {key}")
        quota_contract[key] = count
    required_quota_contract = {
        ("play_turn", 2, 19): 24,
        ("play_turn", 20, 31): 16,
        ("play_turn", 32, 39): 8,
        ("opening_placement", 40, None): 8,
    }
    if quota_contract != required_quota_contract:
        raise CampaignError(
            "fixed-root campaign must reserve 24 play-turn roots at width 2-19, "
            "16 at width 20-31, 8 at width 32-39, and 8 opening-placement "
            "roots at width 40+"
        )
    if int(fixed_protocol.get("n_roots", -1)) != 64:
        raise CampaignError("stratified fixed-root campaign must use 64 roots")
    if int(fixed_protocol.get("max_root_games", -1)) <= 0:
        raise CampaignError("fixed-root campaign must bound max_root_games")
    if int(fixed_protocol.get("min_width_40_roots", -1)) < 8:
        raise CampaignError(
            "fixed-root campaign must include at least 8 width-40 roots"
        )

    base_raw = campaign.get("base_arm")
    arms_raw = campaign.get("adaptive_arms")
    if not isinstance(base_raw, dict) or not isinstance(arms_raw, list):
        raise CampaignError("campaign must define one base arm and adaptive arms")
    if base_raw.get("id") != "base_n128_d6":
        raise CampaignError("base arm must be base_n128_d6")
    base_path = _resolve(source, base_raw.get("config"))
    base = load_search_spec(base_path)

    evaluator_path = _resolve(source, campaign.get("evaluator_config"))
    evaluator = load_evaluator_spec(evaluator_path)
    evaluator_fields = evaluator["effective_evaluator_config"]
    if evaluator_fields.get("public_observation") is not True:
        raise CampaignError("teacher evaluator must use public observations")
    if evaluator_fields.get("cache_size") != 0:
        raise CampaignError("fixed-root evaluator cache must be zero")
    canonical_evaluator = current_science.evaluator()
    evaluator_drift = {
        key: {"expected": expected, "actual": evaluator_fields.get(key)}
        for key, expected in canonical_evaluator.items()
        if evaluator_fields.get(key) != expected
    }
    if evaluator_drift:
        raise CampaignError(
            f"teacher evaluator differs from current science contract: {evaluator_drift}"
        )

    expected_ids = {"adaptive_n256_w20_d6", "adaptive_n256_w40_d6"}
    arms: dict[str, dict[str, Any]] = {}
    for raw in arms_raw:
        if not isinstance(raw, dict) or raw.get("id") not in expected_ids:
            raise CampaignError("adaptive arms must be exactly width-20 and width-40")
        arm_id = str(raw["id"])
        if arm_id in arms:
            raise CampaignError(f"duplicate adaptive arm {arm_id}")
        threshold = int(raw.get("activation_legal_width", -1))
        expected_threshold = 20 if "w20" in arm_id else 40
        if threshold != expected_threshold:
            raise CampaignError(
                f"{arm_id} activation threshold must be {expected_threshold}"
            )
        spec_path = _resolve(source, raw.get("config"))
        spec = load_search_spec(spec_path)
        if spec["name"] != arm_id:
            raise CampaignError(f"{arm_id} config name does not match campaign id")
        validate_search_comparison(
            base,
            spec,
            allowed_differences={
                "n_full_wide",
                "n_full_wide_threshold",
                "wide_roots_always_full",
            },
        )
        arms[arm_id] = {**raw, "threshold": threshold, "spec": spec, "path": spec_path}
    if set(arms) != expected_ids:
        raise CampaignError(f"adaptive arm set must be {sorted(expected_ids)}")

    common = base["effective_search_config"]
    canonical_search = current_science.search()
    experimental_dose = {
        "n_full_wide",
        "n_full_wide_threshold",
        "wide_roots_always_full",
    }
    canonical_drift = {
        key: {"expected": expected, "actual": common.get(key)}
        for key, expected in canonical_search.items()
        if key not in experimental_dose and common.get(key) != expected
    }
    if canonical_drift:
        raise CampaignError(
            "base operator differs from current science contract outside the "
            f"adaptive dose: {canonical_drift}"
        )
    exact_common = {
        "n_full": 128,
        "n_fast": 16,
        "coherent_public_belief_search": True,
        "information_set_search": False,
        "belief_chance_spectra": False,
        "determinization_particles": 1,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "c_scale": 0.1,
        "c_visit": 50.0,
        "sigma_eval": 0.79,
        "temperature": 0.0,
    }
    drift = {
        key: {"expected": expected, "actual": common.get(key)}
        for key, expected in exact_common.items()
        if common.get(key) != expected
    }
    if drift:
        raise CampaignError(f"base operator drift: {drift}")
    if common.get("n_full") == 64 or common.get("n_fast") == 64:
        raise CampaignError("n64 is forbidden in this campaign")
    if common.get("n_full_wide") is not None:
        raise CampaignError("base n128 arm must not have an adaptive budget")
    for arm_id, arm in arms.items():
        fields = arm["spec"]["effective_search_config"]
        arm_common_drift = {
            key: {"expected": expected, "actual": fields.get(key)}
            for key, expected in canonical_search.items()
            if key not in experimental_dose and fields.get(key) != expected
        }
        if arm_common_drift:
            raise CampaignError(
                f"{arm_id} differs from current science contract outside the "
                f"adaptive dose: {arm_common_drift}"
            )
        if (
            fields.get("n_full") != 128
            or fields.get("n_full_wide") != 256
            or fields.get("n_full_wide_threshold") != arm["threshold"]
            or fields.get("wide_roots_always_full") is not True
        ):
            raise CampaignError(f"{arm_id} is not an n128/adaptive-n256 operator")

    return {
        "source": source,
        "payload": campaign,
        "base": {"id": str(base_raw["id"]), "spec": base, "path": base_path},
        "arms": arms,
        "evaluator": evaluator,
        "evaluator_path": evaluator_path,
        "science_contract": current_science.load(),
        "science_contract_path": current_science.CONTRACT_PATH,
        "science_contract_sha256": _sha256(current_science.CONTRACT_PATH),
    }


def _fixed_command(
    loaded: dict[str, Any],
    arm: dict[str, Any],
    *,
    checkpoint: Path,
    out_dir: Path,
    device: str,
    create_panel: bool,
) -> list[str]:
    protocol = loaded["payload"]["fixed_root_protocol"]
    arm_id = str(arm["spec"]["name"])
    command = [
        sys.executable,
        str(_TOOLS / "fixed_root_search_stability.py"),
        "--checkpoint",
        str(checkpoint),
        "--evaluator-config",
        str(loaded["evaluator_path"]),
        "--config-a",
        str(loaded["base"]["path"]),
        "--config-b",
        str(arm["path"]),
        "--allowed-search-config-differences",
        "n_full_wide,n_full_wide_threshold,wide_roots_always_full",
        "--root-panel",
        str(out_dir / "real-roots.json"),
        "--n-roots",
        str(int(protocol["n_roots"])),
        "--decisions-per-game",
        ",".join(str(int(value)) for value in protocol["decisions_per_game"]),
        "--root-base-seed",
        str(int(protocol["root_base_seed"])),
        "--min-legal-actions",
        str(int(protocol["min_legal_actions"])),
        "--min-wide-roots",
        str(int(protocol["min_width_40_roots"])),
        "--max-root-games",
        str(int(protocol["max_root_games"])),
        "--repeats",
        str(int(protocol["repeats"])),
        "--search-seed-base-a",
        str(int(protocol["base_search_seed"])),
        "--search-seed-base-b",
        str(int(arm["root_search_seed_base"])),
        "--device",
        device,
        "--max-batch-size",
        str(int(protocol["max_batch_size"])),
        "--max-wait-ms",
        str(float(protocol["max_wait_ms"])),
        "--out",
        str(out_dir / f"fixed.{arm_id}.json"),
    ]
    for quota in protocol["root_stratum_quotas"]:
        maximum = quota.get("max_legal_width")
        width = (
            f"{int(quota['min_legal_width'])}+"
            if maximum is None
            else f"{int(quota['min_legal_width'])}-{int(maximum)}"
        )
        command.extend(
            (
                "--root-stratum-quota",
                f"{quota['phase']}:{width}={int(quota['count'])}",
            )
        )
    if create_panel:
        command.append("--create-root-panel")
    return command


def _h2h_command(
    loaded: dict[str, Any],
    arm: dict[str, Any],
    *,
    checkpoint: Path,
    out_dir: Path,
    device: str,
    devices: str | None,
    workers: int,
    pairs: int,
) -> list[str]:
    protocol = loaded["payload"]["paired_game_protocol"]
    arm_id = str(arm["spec"]["name"])
    threshold = int(arm["threshold"])
    command = [
        sys.executable,
        str(_TOOLS / "gumbel_search_cross_net_h2h.py"),
        "--candidate",
        str(checkpoint),
        "--baseline",
        str(checkpoint),
        "--pairs",
        str(pairs),
        "--workers",
        str(workers),
        "--device",
        device,
        "--n-full",
        "128",
        "--candidate-n-full-wide",
        "256",
        "--candidate-n-full-wide-threshold",
        str(threshold),
        "--candidate-wide-roots-always-full",
        "--no-baseline-wide-roots-always-full",
        "--max-depth",
        "80",
        "--max-decisions",
        str(int(protocol["max_decisions"])),
        "--prior-temperature",
        "1.0",
        "--c-visit",
        "50.0",
        "--c-scale",
        "0.1",
        "--sigma-eval",
        "0.79",
        "--max-root-candidates",
        "16",
        "--max-root-candidates-wide",
        "54",
        "--wide-candidates-threshold",
        "24",
        "--public-observation",
        "--coherent-public-belief-search",
        "--no-information-set-search",
        "--no-belief-chance-spectra",
        "--determinization-particles",
        "1",
        "--determinization-min-simulations",
        "32",
        "--correct-rust-chance-spectra",
        "--lazy-interior-chance",
        "--forced-root-target-mode",
        "trajectory_only",
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold",
        "20",
        "--native-mcts-hot-loop",
        "--evaluator-rust-featurize",
        "--base-seed",
        str(int(arm["h2h_base_seed"])),
        "--gate-config",
        str(protocol["gate_config"]),
        "--out",
        str(out_dir / f"h2h.{arm_id}.json"),
    ]
    if devices:
        command.extend(("--devices", devices))
    return command


def build_stage_commands(
    loaded: dict[str, Any],
    *,
    checkpoint: Path,
    out_dir: Path,
    device: str,
    devices: str | None,
    workers: int | None = None,
    pairs: int | None = None,
) -> dict[str, list[str]]:
    paired = loaded["payload"]["paired_game_protocol"]
    effective_workers = int(workers if workers is not None else paired["workers"])
    effective_pairs = int(pairs if pairs is not None else paired["pairs"])
    w20 = loaded["arms"]["adaptive_n256_w20_d6"]
    w40 = loaded["arms"]["adaptive_n256_w40_d6"]
    panel_exists = (out_dir / "real-roots.json").is_file()
    return {
        "fixed-w20": _fixed_command(
            loaded,
            w20,
            checkpoint=checkpoint,
            out_dir=out_dir,
            device=device,
            create_panel=not panel_exists,
        ),
        "fixed-w40": _fixed_command(
            loaded,
            w40,
            checkpoint=checkpoint,
            out_dir=out_dir,
            device=device,
            create_panel=False,
        ),
        "h2h-w20": _h2h_command(
            loaded,
            w20,
            checkpoint=checkpoint,
            out_dir=out_dir,
            device=device,
            devices=devices,
            workers=effective_workers,
            pairs=effective_pairs,
        ),
        "h2h-w40": _h2h_command(
            loaded,
            w40,
            checkpoint=checkpoint,
            out_dir=out_dir,
            device=device,
            devices=devices,
            workers=effective_workers,
            pairs=effective_pairs,
        ),
    }


def _validate_h2h(
    report: dict[str, Any], *, threshold: int, pairs: int, checkpoint_sha: str
) -> None:
    expected_checkpoint_sha = _normalize_sha256(checkpoint_sha)
    for field in ("candidate_checkpoint_sha256", "baseline_checkpoint_sha256"):
        try:
            actual_checkpoint_sha = _normalize_sha256(str(report.get(field)))
        except CampaignError as error:
            raise CampaignError(f"paired-game report has invalid {field}") from error
        if actual_checkpoint_sha != expected_checkpoint_sha:
            raise CampaignError(
                f"paired-game report {field} drift: "
                f"{actual_checkpoint_sha} != {expected_checkpoint_sha}"
            )
    exact = {
        "candidate_n_full": 128,
        "baseline_n_full": 128,
        "candidate_n_full_wide": 256,
        "baseline_n_full_wide": None,
        "candidate_n_full_wide_threshold": threshold,
        "candidate_wide_roots_always_full": True,
        "baseline_wide_roots_always_full": False,
        "lazy_interior_chance": True,
        "value_squash": "tanh",
        "candidate_value_squash": "tanh",
        "baseline_value_squash": "tanh",
        "value_readout": "scalar",
        "candidate_value_readout": "scalar",
        "baseline_value_readout": "scalar",
        "c_scale": 0.1,
        "candidate_c_scale": 0.1,
        "baseline_c_scale": 0.1,
        "c_visit": 50.0,
        "rescale_noise_floor_c": 0.0,
        "candidate_rescale_noise_floor_c": 0.0,
        "baseline_rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.79,
        "candidate_sigma_eval": 0.79,
        "baseline_sigma_eval": 0.79,
        "max_root_candidates": 16,
        "max_root_candidates_wide": 54,
        "wide_candidates_threshold": 24,
        "correct_rust_chance_spectra": True,
        "public_observation": True,
        "coherent_public_belief_search": True,
        "information_set_search": False,
        "belief_chance_spectra": False,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "forced_root_target_mode": "trajectory_only",
        "native_mcts_hot_loop": True,
        "determinization_particles": 1,
        "determinization_min_simulations": 32,
        "raw_policy_above_width": None,
        "pairs_requested": pairs,
        "games_played": pairs * 2,
        "games_truncated": 0,
        "complete_pairs": pairs,
    }
    drift = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in exact.items()
        if report.get(key) != expected
    }
    if drift:
        raise CampaignError(f"paired-game report drift: {drift}")
    if report.get("errors"):
        raise CampaignError(f"paired-game report contains errors: {report['errors']}")
    typed_fields = report.get("typed_config", {}).get("fields", {})
    if typed_fields.get("evaluator_rust_featurize") is not True:
        raise CampaignError("paired-game report did not use the Rust featurizer")
    if int(typed_fields.get("max_depth", -1)) != 80:
        raise CampaignError("paired-game report did not use max_depth=80")


def _base_semantic_fingerprint(fixed: dict[str, Any], base_role: str) -> str:
    """Fingerprint repeated base searches while excluding non-semantic wall time."""

    roots: list[dict[str, Any]] = []
    for root in fixed["per_root"]:
        runs = []
        for raw in root["roles"][base_role]["runs"]:
            run = dict(raw)
            run.pop("wall_sec", None)
            runs.append(run)
        roots.append(
            {
                "root_sha256": root["root_sha256"],
                "runs": runs,
                "stability": root["roles"][base_role]["stability"],
            }
        )
    return content_sha256(roots)


def _activation_slices(
    fixed: dict[str, Any], *, base_role: str, arm_role: str, threshold: int
) -> dict[str, Any]:
    active = [
        root for root in fixed["per_root"] if int(root["legal_width"]) >= threshold
    ]
    if not active:
        raise CampaignError(f"fixed-root panel has no roots at width >= {threshold}")

    def grouped(field: str) -> dict[str, Any]:
        return {
            value: _comparison_slice(
                [root for root in active if str(root[field]) == value],
                base_role,
                arm_role,
            )
            for value in sorted({str(root[field]) for root in active})
        }

    return {
        "definition": f"legal_width>={threshold}",
        "global": _comparison_slice(active, base_role, arm_role),
        "by_phase": grouped("phase"),
        "by_raw_phase": grouped("phase_raw"),
        "by_legal_width_bucket": grouped("legal_width_bucket"),
    }


def aggregate_campaign(loaded: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    campaign = loaded["payload"]
    pairs = int(campaign["paired_game_protocol"]["pairs"])
    base_role = str(loaded["base"]["spec"]["name"])
    checkpoint_sha: str | None = None
    root_panel_sha: str | None = None
    base_semantic_sha: str | None = None
    root_distribution: dict[str, Any] | None = None
    results: dict[str, Any] = {}

    for arm_id, arm in loaded["arms"].items():
        fixed_path = out_dir / f"fixed.{arm_id}.json"
        h2h_path = out_dir / f"h2h.{arm_id}.json"
        fixed = _load_object(fixed_path)
        h2h = _load_object(h2h_path)
        if fixed.get("schema_version") != FIXED_ROOT_REPORT_SCHEMA:
            raise CampaignError(f"{fixed_path} has the wrong fixed-root schema")
        content = dict(fixed)
        recorded_content_sha = content.pop("report_content_sha256", None)
        if recorded_content_sha != content_sha256(content):
            raise CampaignError(f"{fixed_path} content digest mismatch")
        root_panel = fixed.get("root_panel", {})
        expected_quotas = campaign["fixed_root_protocol"]["root_stratum_quotas"]
        if (
            int(root_panel.get("root_count", -1))
            != int(campaign["fixed_root_protocol"]["n_roots"])
            or root_panel.get("root_stratum_quotas") != expected_quotas
        ):
            raise CampaignError(
                f"{fixed_path} did not use the preregistered phase/width panel"
            )
        quota_objects = tuple(
            RootStratumQuota(
                phase=str(raw["phase"]),
                min_legal_width=int(raw["min_legal_width"]),
                max_legal_width=(
                    None
                    if raw.get("max_legal_width") is None
                    else int(raw["max_legal_width"])
                ),
                count=int(raw["count"]),
            )
            for raw in expected_quotas
        )
        try:
            actual_stratum_counts = enforce_root_stratum_quotas(
                fixed.get("per_root", []), quota_objects
            )
        except ValueError as error:
            raise CampaignError(
                f"{fixed_path} violates root strata: {error}"
            ) from error
        if root_panel.get("root_stratum_counts") != actual_stratum_counts:
            raise CampaignError(f"{fixed_path} root stratum counts are inconsistent")
        observed_phase_widths = root_phase_width_summary(fixed.get("per_root", []))
        if root_panel.get("root_phase_width_summary") != observed_phase_widths:
            raise CampaignError(f"{fixed_path} phase/width summary is inconsistent")
        width_40_phases = sorted(
            {
                str(root["phase"])
                for root in fixed.get("per_root", [])
                if int(root["legal_width"]) >= 40
            }
        )
        this_root_distribution = {
            "phase_width_summary": observed_phase_widths,
            "width_40_activation_phases": width_40_phases,
            "width_40_classification": (
                "opening_only"
                if width_40_phases == ["opening_placement"]
                else "mixed_or_nonopening"
            ),
        }
        root_distribution = root_distribution or this_root_distribution
        if this_root_distribution != root_distribution:
            raise CampaignError("fixed reports disagree on root phase/width support")
        this_checkpoint_sha = str(fixed.get("checkpoint", {}).get("sha256"))
        this_root_panel_sha = str(fixed.get("root_panel", {}).get("content_sha256"))
        checkpoint_sha = checkpoint_sha or this_checkpoint_sha
        root_panel_sha = root_panel_sha or this_root_panel_sha
        if (
            this_checkpoint_sha != checkpoint_sha
            or this_root_panel_sha != root_panel_sha
        ):
            raise CampaignError(
                "comparisons did not use the same checkpoint/root panel"
            )
        arm_role = str(arm["spec"]["name"])
        if set(fixed.get("roles", {})) != {base_role, arm_role}:
            raise CampaignError(f"{fixed_path} role set is not the causal A/B pair")
        reported_base = fixed["roles"][base_role]
        reported_arm = fixed["roles"][arm_role]
        if (
            reported_base.get("effective_search_config_sha256")
            != loaded["base"]["spec"]["effective_search_config_sha256"]
            or reported_arm.get("effective_search_config_sha256")
            != arm["spec"]["effective_search_config_sha256"]
        ):
            raise CampaignError(
                f"{fixed_path} search config bytes drifted from campaign"
            )
        if set(fixed.get("search_config_differences", {})) != {
            "n_full_wide",
            "n_full_wide_threshold",
            "wide_roots_always_full",
        }:
            raise CampaignError(f"{fixed_path} changed fields outside adaptive search")
        this_base_semantic_sha = _base_semantic_fingerprint(fixed, base_role)
        base_semantic_sha = base_semantic_sha or this_base_semantic_sha
        if this_base_semantic_sha != base_semantic_sha:
            raise CampaignError(
                "the repeated base-n128 control changed between adaptive comparisons"
            )
        _validate_h2h(
            h2h,
            threshold=int(arm["threshold"]),
            pairs=pairs,
            checkpoint_sha=checkpoint_sha,
        )
        activation = _activation_slices(
            fixed,
            base_role=base_role,
            arm_role=arm_role,
            threshold=int(arm["threshold"]),
        )
        comparison = activation["global"]["comparison"]
        telemetry = h2h.get("search_telemetry", {})
        candidate_telemetry = telemetry.get("by_role", {}).get("candidate", {})
        baseline_telemetry = telemetry.get("by_role", {}).get("baseline", {})
        results[arm_id] = {
            "activation_legal_width": int(arm["threshold"]),
            "artifacts": {
                "fixed_root": {
                    "path": str(fixed_path.resolve()),
                    "sha256": _sha256(fixed_path),
                },
                "paired_games": {
                    "path": str(h2h_path.resolve()),
                    "sha256": _sha256(h2h_path),
                },
            },
            "fixed_root": {
                "all_roots": fixed["slices"],
                "activation_roots": activation,
                "cross_seed_js_relative_reduction": comparison[
                    "role_b_relative_js_reduction"
                ],
                "top1_agreement_delta": comparison[
                    "role_b_minus_role_a_top1_agreement"
                ],
                "target_prior_js_delta": comparison[
                    "role_b_minus_role_a_target_prior_js"
                ],
                "completed_q_top_margin_delta": comparison[
                    "role_b_minus_role_a_completed_q_top_margin"
                ],
                "wall_cost_ratio": comparison["role_b_over_role_a_wall_ratio"],
                "simulation_cost_ratio": comparison[
                    "role_b_over_role_a_simulations_ratio"
                ],
            },
            "paired_games": {
                "candidate_wins": int(h2h["candidate_wins"]),
                "baseline_wins": int(h2h["baseline_wins"]),
                "candidate_win_rate": float(h2h["candidate_win_rate"]),
                "pentanomial_verdict": h2h["verdict"],
                "positive_elo_verdict": h2h["superiority_verdict"],
                "pair_diagnostics": h2h["pair_diagnostics"],
                "whole_game_search_elapsed_ratio": telemetry.get(
                    "candidate_over_baseline_elapsed_ratio"
                ),
                "whole_game_simulation_ratio": telemetry.get(
                    "candidate_over_baseline_simulations_ratio"
                ),
                "candidate_wide_root_calls": candidate_telemetry.get("wide_root_calls"),
                "baseline_wide_root_calls": baseline_telemetry.get("wide_root_calls"),
                "candidate_wide_prior_disagreement_rate": candidate_telemetry.get(
                    "wide_selected_vs_prior_disagreement_rate"
                ),
                "baseline_wide_prior_disagreement_rate": baseline_telemetry.get(
                    "wide_selected_vs_prior_disagreement_rate"
                ),
            },
        }

    policy = campaign["selection_policy"]
    eligible: list[str] = []
    for arm_id, result in results.items():
        fixed = result["fixed_root"]
        games = result["paired_games"]
        elapsed_ratio = games["whole_game_search_elapsed_ratio"]
        cost_ok = elapsed_ratio is not None and float(elapsed_ratio) <= float(
            policy["max_whole_game_search_cost_ratio"]
        )
        h1 = games["positive_elo_verdict"] == "H1"
        js_reduction = fixed["cross_seed_js_relative_reduction"]
        top1_delta = fixed["top1_agreement_delta"]
        stability = (
            js_reduction is not None
            and top1_delta is not None
            and float(js_reduction)
            >= float(policy["min_relative_cross_seed_js_reduction"])
            and float(top1_delta) >= float(policy["min_top1_agreement_delta"])
        )
        result["selection_evidence"] = {
            "cost_ok": cost_ok,
            "positive_elo_h1": h1,
            "stability_proxy_ok": stability,
            "eligible": bool(cost_ok and (h1 or stability)),
        }
        if result["selection_evidence"]["eligible"]:
            eligible.append(arm_id)

    if eligible:
        selected = max(
            eligible,
            key=lambda arm_id: (
                results[arm_id]["paired_games"]["positive_elo_verdict"] == "H1",
                results[arm_id]["paired_games"]["candidate_win_rate"],
                results[arm_id]["fixed_root"]["cross_seed_js_relative_reduction"]
                or float("-inf"),
            ),
        )
        reason = "best cost-bounded arm with positive-Elo H1 or the preregistered stability proxy"
    else:
        selected = base_role
        reason = (
            "neither adaptive arm proved a cost-bounded target or playing-strength gain"
        )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "checkpoint_sha256": checkpoint_sha,
        "science_contract": {
            "path": str(loaded["science_contract_path"]),
            "sha256": loaded["science_contract_sha256"],
            "contract_id": loaded["science_contract"]["contract_id"],
            "target_information_regime": current_science.target_information_regime(),
            "experimental_dose_fields": [
                "n_full_wide",
                "n_full_wide_threshold",
                "wide_roots_always_full",
            ],
        },
        "root_panel_content_sha256": root_panel_sha,
        "root_distribution": root_distribution,
        "causal_contract": {
            "same_checkpoint_both_roles": True,
            "base_budget": 128,
            "adaptive_budget": 256,
            "d6_min_legal_width": 20,
            "information_regime": "coherent_public_belief_single_tree",
            "forbidden_budget": 64,
        },
        "results": results,
        "selection": {
            "selected_operator": selected,
            "eligible_adaptive_arms": sorted(eligible),
            "reason": reason,
            "policy": policy,
        },
    }
    report["report_content_sha256"] = content_sha256(report)
    return report


def _print_dispatch_commands(
    *,
    campaign: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    out_dir: Path,
    device: str,
    devices: str | None,
    workers: int | None,
    pairs: int | None,
) -> None:
    """Print portable wrapper commands that retain checkpoint verification."""

    try:
        campaign_arg = str(campaign.relative_to(_REPO_ROOT))
    except ValueError:
        campaign_arg = str(campaign)
    for stage in STAGES:
        command = [
            "python3",
            "tools/teacher_operator_campaign.py",
            "--campaign",
            campaign_arg,
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            checkpoint_sha256,
            "--out-dir",
            str(out_dir),
            "--stage",
            stage,
            "--device",
            device,
        ]
        if devices:
            command.extend(("--devices", devices))
        if workers is not None:
            command.extend(("--workers", str(workers)))
        if pairs is not None:
            command.extend(("--pairs", str(pairs)))
        print(f"# {stage}")
        print(shlex.join(command))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        default=str(
            _REPO_ROOT
            / "configs/experiments/teacher_operator_coherent_v1/campaign.json"
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="exact current-champion digest from the sealed handoff; no lineage default",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stage", choices=("render", "all", "aggregate", *STAGES), default="all"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated H2H devices distributed across workers",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--pairs", type=int, default=None)
    args = parser.parse_args(argv)

    loaded = load_campaign(args.campaign)
    # Preserve absolute paths for commands rendered for a Linux worker. On
    # macOS, ``resolve()`` rewrites a non-local /home/... through the host's
    # /home symlink and produces a path that cannot exist on the worker.
    checkpoint = Path(args.checkpoint).expanduser().absolute()
    expected_checkpoint_sha = _normalize_sha256(args.checkpoint_sha256)
    out_dir = Path(args.out_dir).expanduser().absolute()
    commands = build_stage_commands(
        loaded,
        checkpoint=checkpoint,
        out_dir=out_dir,
        device=str(args.device),
        devices=args.devices,
        workers=args.workers,
        pairs=args.pairs,
    )
    if args.stage == "render":
        _print_dispatch_commands(
            campaign=loaded["source"],
            checkpoint=checkpoint,
            checkpoint_sha256=expected_checkpoint_sha,
            out_dir=out_dir,
            device=str(args.device),
            devices=args.devices,
            workers=args.workers,
            pairs=args.pairs,
        )
        return 0
    if args.stage == "aggregate":
        report = aggregate_campaign(loaded, out_dir=out_dir)
        if report["checkpoint_sha256"] != expected_checkpoint_sha:
            raise CampaignError(
                "campaign artifacts used a different checkpoint than the sealed "
                f"handoff: {report['checkpoint_sha256']} != {expected_checkpoint_sha}"
            )
        write_json(out_dir / "teacher-operator-report.json", report)
        print(json.dumps(report["selection"], indent=2, sort_keys=True))
        return 0
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    actual_checkpoint_sha = _sha256(checkpoint)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        parser.error(
            "checkpoint digest differs from the sealed current-champion handoff: "
            f"expected {expected_checkpoint_sha}, actual {actual_checkpoint_sha}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "fixed-w40" and not (out_dir / "real-roots.json").is_file():
            raise CampaignError("fixed-w40 requires real-roots.json from fixed-w20")
        print(f"[{stage}] {shlex.join(commands[stage])}", flush=True)
        subprocess.run(commands[stage], cwd=_REPO_ROOT, check=True)
        if _sha256(checkpoint) != expected_checkpoint_sha:
            raise CampaignError(f"checkpoint bytes changed while {stage} was running")
    if args.stage == "all":
        report = aggregate_campaign(loaded, out_dir=out_dir)
        write_json(out_dir / "teacher-operator-report.json", report)
        print(json.dumps(report["selection"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
