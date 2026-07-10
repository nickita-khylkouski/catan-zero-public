from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.rnd_leaderboard import (
    SCHEMA_VERSION,
    ValidationError,
    main,
    render_markdown,
    validate_and_aggregate,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT = "c" * 40
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rnd"
SEED_MANIFEST_PATH = FIXTURE_DIR / "paired-seeds.json"
TRAINING_MANIFEST_PATH = FIXTURE_DIR / "training-manifest.json"
SEED_MANIFEST_SHA = hashlib.sha256(SEED_MANIFEST_PATH.read_bytes()).hexdigest()
TRAINING_MANIFEST_SHA = hashlib.sha256(TRAINING_MANIFEST_PATH.read_bytes()).hexdigest()


def _campaign(*, regime_metric: str = "logical_leaves") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "test-campaign",
        "required_information_regime": "public_only",
        "require_same_training_manifest": True,
        "required_arm_ids": ["arm-a", "arm-b"],
        "budget_contracts": {
            "equal_work": {
                "match_metrics": [regime_metric, "orientation_rows"],
                "absolute_tolerance": 0.0,
                "relative_tolerance": 0.0,
            },
            "equal_time": {
                "match_metrics": ["wall_time_sec"],
                "absolute_tolerance": 0.0,
                "relative_tolerance": 0.05,
            },
        },
        "arms": [
            {
                "arm_id": "arm-a",
                "architecture_id": "arch-a",
                "search_id": "search-a",
                "source_status": "implemented",
                "measurement_adapter_status": "implemented",
                "runnable": True,
            },
            {
                "arm_id": "arm-b",
                "architecture_id": "arch-b",
                "search_id": "search-b",
                "source_status": "implemented",
                "measurement_adapter_status": "implemented",
                "runnable": True,
            },
        ],
    }


def _game(*, arm_id: str, candidate_seat: int, score: float, leaves: int = 16, wall: float = 1.0) -> dict:
    candidate_color = ("RED", "BLUE")[candidate_seat]
    reference_color = ("BLUE", "RED")[candidate_seat]
    return {
        "game_id": f"{arm_id}-seat-{candidate_seat}",
        "pair_id": "pair-1",
        "seed": 123,
        "track": "2p_no_trade",
        "information_regime": "public_conservation_pimc",
        "reference_information_regime": "public_conservation_pimc",
        "seat_assignment": {"candidate": candidate_seat, "reference": 1 - candidate_seat},
        "completed": True,
        "winner": candidate_color if score == 1.0 else reference_color,
        "candidate_score": score,
        "counters": {
            "nominal_visits": 8,
            "scheduled_visits": 20,
            "logical_leaves": leaves,
            "orientation_rows": leaves * 2,
            "evaluator_calls": 4,
            "wall_time_sec": wall,
        },
    }


def _bundle(
    arm_id: str,
    architecture_id: str,
    search_id: str,
    *,
    regime: str = "equal_work",
    leaves: int = 16,
    wall: float = 1.0,
    scores: tuple[float, float] = (1.0, 0.0),
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "test-campaign",
        "run_id": f"run-{arm_id}-{regime}",
        "arm_id": arm_id,
        "budget_regime": regime,
        "track": "2p_no_trade",
        "required_information_regime": "public_only",
        "architecture": {
            "architecture_id": architecture_id,
            "parameter_count": 100 if arm_id == "arm-a" else 200,
            "config_sha256": SHA_A,
        },
        "search": {"search_id": search_id, "config_sha256": SHA_B},
        "checkpoint": {"path": f"/remote/{arm_id}.pt", "sha256": SHA_A},
        "reference": {
            "reference_id": "frozen-reference",
            "architecture": {
                "architecture_id": "reference-arch",
                "parameter_count": 300,
                "config_sha256": SHA_B,
            },
            "search": {"search_id": "reference-search", "config_sha256": SHA_A},
            "checkpoint": {"path": "/remote/reference.pt", "sha256": SHA_B},
        },
        "code": {"git_commit": COMMIT, "dirty": False},
        "seed_manifest": {
            "path": str(SEED_MANIFEST_PATH),
            "sha256": SEED_MANIFEST_SHA,
            "schema_version": "catan-zero-rnd-paired-seeds/v1",
            "track": "2p_no_trade",
            "seed_count": 1,
        },
        "training_manifest": {
            "path": str(TRAINING_MANIFEST_PATH),
            "sha256": TRAINING_MANIFEST_SHA,
            "schema_version": "catan-zero-rnd-training-manifest/v1",
        },
        "games": [
            _game(arm_id=arm_id, candidate_seat=0, score=scores[0], leaves=leaves, wall=wall),
            _game(arm_id=arm_id, candidate_seat=1, score=scores[1], leaves=leaves, wall=wall),
        ],
    }


def _bundles(**kwargs) -> list[dict]:
    return [
        _bundle("arm-a", "arch-a", "search-a", **kwargs),
        _bundle("arm-b", "arch-b", "search-b", scores=(1.0, 1.0), **kwargs),
    ]


def test_valid_equal_work_report_records_every_counter_and_provenance() -> None:
    report = validate_and_aggregate(_campaign(), _bundles())

    assert report["valid"] is True
    assert report["budget_validation"]["status"] == "pass"
    assert report["paired_seeds"] == [123]
    assert report["pairing_schedule"] == [
        {
            "pair_id": "pair-1",
            "seed": 123,
            "seat_assignments": [
                {"candidate": 0, "reference": 1},
                {"candidate": 1, "reference": 0},
            ],
        }
    ]
    assert [row["arm_id"] for row in report["leaderboard"]] == ["arm-b", "arm-a"]
    winner = report["leaderboard"][0]
    assert winner["checkpoint_sha256"] == SHA_A
    assert winner["parameter_count"] == 200
    assert winner["work_totals"] == {
        "nominal_visits": 16.0,
        "scheduled_visits": 40.0,
        "logical_leaves": 32.0,
        "orientation_rows": 64.0,
        "evaluator_calls": 8.0,
        "wall_time_sec": 2.0,
    }
    markdown = render_markdown(report)
    assert "Budget contract: **PASS**" in markdown
    assert "Nominal visits" in markdown
    assert SHA_A in markdown


def test_campaign_can_freeze_search_or_checkpoint_across_candidate_arms() -> None:
    campaign = _campaign()
    campaign["frozen_candidate_fields"] = ["search_config_sha256"]
    validate_and_aggregate(campaign, _bundles())

    bundles = _bundles()
    bundles[1]["search"]["config_sha256"] = SHA_A
    with pytest.raises(ValidationError, match="search_config_sha256 frozen"):
        validate_and_aggregate(campaign, bundles)


def test_missing_any_counter_fails_closed() -> None:
    bundles = _bundles()
    del bundles[0]["games"][0]["counters"]["evaluator_calls"]
    with pytest.raises(ValidationError, match="missing required measured counters: evaluator_calls"):
        validate_and_aggregate(_campaign(), bundles)


def test_non_swapped_pair_fails_closed() -> None:
    bundles = _bundles()
    bundles[0]["games"][1]["seat_assignment"] = {"candidate": 0, "reference": 1}
    bundles[0]["games"][1]["winner"] = "BLUE"
    with pytest.raises(ValidationError, match="not exact candidate/reference seat swaps"):
        validate_and_aggregate(_campaign(), bundles)


def test_pairing_rejects_impossible_seats_and_score_winner_mismatch() -> None:
    bundles = _bundles()
    bundles[0]["games"][0]["seat_assignment"] = {"candidate": 2, "reference": 3}
    with pytest.raises(ValidationError, match="seats must be exactly 0 and 1"):
        validate_and_aggregate(_campaign(), bundles)

    bundles = _bundles()
    bundles[0]["games"][0]["candidate_score"] = 0.0
    with pytest.raises(ValidationError, match="contradicts winner"):
        validate_and_aggregate(_campaign(), bundles)

    bundles = _bundles()
    del bundles[0]["games"][0]["winner"]
    with pytest.raises(ValidationError, match="winner must be a non-empty string"):
        validate_and_aggregate(_campaign(), bundles)


def test_equal_work_mismatch_fails_instead_of_ranking() -> None:
    bundles = _bundles()
    for game in bundles[1]["games"]:
        game["counters"]["logical_leaves"] += 1
    with pytest.raises(ValidationError, match="equal_work budget mismatch"):
        validate_and_aggregate(_campaign(), bundles)


def test_equal_time_accepts_declared_tolerance_but_rejects_overspend() -> None:
    bundles = _bundles(regime="equal_time", wall=1.0)
    for game in bundles[1]["games"]:
        game["counters"]["wall_time_sec"] = 1.04
    report = validate_and_aggregate(_campaign(), bundles)
    assert report["budget_regime"] == "equal_time"

    for game in bundles[1]["games"]:
        game["counters"]["wall_time_sec"] = 1.06
    with pytest.raises(ValidationError, match="equal_time budget mismatch"):
        validate_and_aggregate(_campaign(), bundles)


def test_non_compute_matched_control_is_reported_but_not_ranked() -> None:
    campaign = _campaign()
    campaign["arms"].append(
        {
            "arm_id": "raw-control",
            "architecture_id": "arch-control",
            "search_id": "raw",
            "source_status": "implemented",
            "measurement_adapter_status": "implemented",
            "comparison_role": "control",
            "runnable": True,
        }
    )
    campaign["required_arm_ids"].append("raw-control")
    bundles = _bundles()
    control = _bundle("raw-control", "arch-control", "raw", leaves=1, wall=0.1)
    control["architecture"]["parameter_count"] = 50
    bundles.append(control)

    report = validate_and_aggregate(campaign, bundles)

    assert [row["arm_id"] for row in report["leaderboard"]] == ["arm-b", "arm-a"]
    assert [row["arm_id"] for row in report["controls"]] == ["raw-control"]
    assert "Non-compute-matched controls" in render_markdown(report)


def test_unimplemented_arm_cannot_submit_results() -> None:
    campaign = _campaign()
    campaign["arms"][0].update(
        source_status="not_implemented",
        measurement_adapter_status="not_implemented",
        runnable=False,
    )
    with pytest.raises(ValidationError, match="is not runnable"):
        validate_and_aggregate(campaign, _bundles())


def test_registry_rejects_false_runnable_claim() -> None:
    campaign = _campaign()
    campaign["arms"][0]["source_status"] = "not_implemented"
    with pytest.raises(ValidationError, match="cannot be runnable"):
        validate_and_aggregate(campaign, _bundles())


def test_campaign_rejects_unknown_required_arm_and_missing_fairness_metric() -> None:
    campaign = _campaign()
    campaign["required_arm_ids"].append("phantom-arm")
    with pytest.raises(ValidationError, match="references unknown arms"):
        validate_and_aggregate(campaign, _bundles())

    campaign = _campaign()
    campaign["budget_contracts"]["equal_work"]["match_metrics"] = ["evaluator_calls"]
    with pytest.raises(ValidationError, match="must match on logical_leaves"):
        validate_and_aggregate(campaign, _bundles())


def test_reference_must_match_exactly_across_arms() -> None:
    bundles = _bundles()
    bundles[1]["reference"]["checkpoint"]["sha256"] = SHA_A
    with pytest.raises(ValidationError, match="reference does not exactly match"):
        validate_and_aggregate(_campaign(), bundles)


def test_public_campaign_rejects_authoritative_game_and_manifest_drift() -> None:
    bundles = _bundles()
    bundles[0]["games"][0]["information_regime"] = "authoritative_hidden_state"
    with pytest.raises(ValidationError, match="violates public_only campaign"):
        validate_and_aggregate(_campaign(), bundles)

    bundles = _bundles()
    bundles[1]["seed_manifest"]["sha256"] = SHA_A
    with pytest.raises(ValidationError, match="seed_manifest SHA mismatch"):
        validate_and_aggregate(_campaign(), bundles)

    bundles = _bundles()
    bundles[1]["training_manifest"]["sha256"] = SHA_B
    with pytest.raises(ValidationError, match="training_manifest SHA mismatch"):
        validate_and_aggregate(_campaign(), bundles)


def test_seed_manifest_file_and_game_schedule_are_verified(tmp_path: Path) -> None:
    bundles = _bundles()
    bundles[0]["games"][0]["seed"] = 999
    bundles[0]["games"][1]["seed"] = 999
    with pytest.raises(ValidationError, match="do not exactly match"):
        validate_and_aggregate(_campaign(), bundles)

    tampered = tmp_path / "paired-seeds.json"
    tampered.write_text(SEED_MANIFEST_PATH.read_text(encoding="utf-8") + " ", encoding="utf-8")
    bundles = _bundles()
    bundles[0]["seed_manifest"]["path"] = str(tampered)
    with pytest.raises(ValidationError, match="seed_manifest SHA mismatch"):
        validate_and_aggregate(_campaign(), bundles)


def test_local_checkpoint_hash_is_verified(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    bundles = _bundles()
    for bundle in bundles:
        bundle["checkpoint"] = {"path": str(checkpoint), "sha256": digest}
        bundle["reference"]["checkpoint"] = {"path": str(checkpoint), "sha256": digest}
    validate_and_aggregate(_campaign(), bundles, verify_local_checkpoints=True)

    bundles[0]["checkpoint"]["sha256"] = SHA_B
    with pytest.raises(ValidationError, match="checkpoint SHA mismatch"):
        validate_and_aggregate(_campaign(), bundles, verify_local_checkpoints=True)


def test_cli_writes_machine_json_and_markdown(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(_campaign()), encoding="utf-8")
    result_paths = []
    for index, bundle in enumerate(_bundles()):
        path = tmp_path / f"result-{index}.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        result_paths.append(path)
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    status = main(
        [
            "aggregate",
            "--campaign",
            str(campaign_path),
            "--result",
            str(result_paths[0]),
            "--result",
            str(result_paths[1]),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert status == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["valid"] is True
    assert out_md.read_text(encoding="utf-8").startswith("# R&D leaderboard")


def test_cli_failure_does_not_write_reports(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(_campaign()), encoding="utf-8")
    bundles = _bundles()
    del bundles[0]["games"][0]["counters"]["orientation_rows"]
    paths = []
    for index, bundle in enumerate(bundles):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        paths.append(path)
    out_json = tmp_path / "should-not-exist.json"
    out_md = tmp_path / "should-not-exist.md"

    status = main(
        [
            "aggregate",
            "--campaign",
            str(campaign_path),
            "--result",
            str(paths[0]),
            "--result",
            str(paths[1]),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert status == 2
    assert not out_json.exists()
    assert not out_md.exists()
