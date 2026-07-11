from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import numpy as np

from tools import a1_promotion_artifacts as artifacts
from tools import a1_promotion_transaction as promotion
from tools.champion_registry import ChampionRegistry


def _json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate.pt"
    champion = tmp_path / "champion.pt"
    candidate.write_bytes(b"candidate")
    champion.write_bytes(b"champion")
    return candidate, champion


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": promotion._sha256(path)}


def _high_regret_report(
    tmp_path: Path, candidate: Path, champion: Path, *, pairs: int = 200
) -> dict:
    source_manifest = tmp_path / "regret-source.npz"
    source_manifest.write_bytes(b"regret manifest fixture")
    suite = tmp_path / "held-out-suite.json"
    suite_payload = {
        "schema_version": artifacts.HIGH_REGRET_SUITE_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "source_manifest": _ref(source_manifest),
        "selection": {
            "algorithm": "stable-hash-holdout-stratified-regret-v1",
            "holdout_fraction": 0.1,
            "holdout_seed": 17,
            "eligible_unique_states": pairs,
            "selected_pairs": pairs,
            "stratum_min_pairs": 20,
            "selected_by_stratum": {
                "phase:opening": 20,
                "phase:robber_dev": 20,
                "phase:chance": 20,
                "phase:build_trade": 20,
                "41+": 20,
            },
        },
        "states": [
            {
                "pair_id": pair,
                "shard_path": str(tmp_path / "source.npz"),
                "shard_id": 0,
                "row_index": pair,
                "game_seed": 50_000 + pair,
                "decision_index": pair % 20,
                "phase": (
                    "BUILD_INITIAL_SETTLEMENT",
                    "MOVE_ROBBER",
                    "ROLL",
                    "BUILD_ROAD",
                )[pair % 4],
                "legal_count": 54 if pair < 20 else 12,
                "regret_score": 1.0,
            }
            for pair in range(pairs)
        ],
    }
    suite_payload["suite_sha256"] = promotion._digest_value(suite_payload)
    _json(suite, suite_payload)
    games = [
        {
            "pair_id": pair,
            "orientation": orientation,
            "candidate_won": True,
            "truncated": False,
            "archived_game_seed": 50_000 + pair,
            "archived_decision_index": pair % 20,
            "buckets": ["phase:BUILD", "close"],
        }
        for pair in range(pairs)
        for orientation in ("candidate_first", "candidate_second")
    ]
    normalized = [{**game, "search_won": game["candidate_won"]} for game in games]
    scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
    pentanomial = promotion.evaluate_pentanomial_sprt(
        scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    assert pentanomial["decision"] == "H1"
    return {
        "schema_version": artifacts.HIGH_REGRET_REPORT_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "suite_manifest": _ref(suite),
        "candidate": _ref(candidate),
        "champion": _ref(champion),
        "evaluation_config": {
            "c_scale": promotion.CANDIDATE_DEPLOYED_C_SCALE,
            "candidate_c_scale": promotion.CANDIDATE_DEPLOYED_C_SCALE,
            "baseline_c_scale": promotion.CHAMPION_DEPLOYED_C_SCALE,
            "candidate_n_full": 128,
            "baseline_n_full": 128,
            "p_full": 1.0,
            "force_full_every_decision": True,
        },
        "errors": [],
        "games": games,
        "pentanomial_sprt": pentanomial,
        "pair_diagnostics": diagnostics,
    }


def _truncate_high_regret_pair(value: dict, pair_id: int = 0) -> None:
    game = next(
        game
        for game in value["games"]
        if game["pair_id"] == pair_id
        and game["orientation"] in {"candidate_first", "candidate_red"}
    )
    game["candidate_won"] = None
    game["truncated"] = True
    normalized = [{**row, "search_won": row["candidate_won"]} for row in value["games"]]
    scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
    value["pair_diagnostics"] = diagnostics
    value["pentanomial_sprt"] = promotion.evaluate_pentanomial_sprt(
        scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )


def _use_color_orientations(value: dict) -> None:
    for game in value["games"]:
        if game["orientation"] == "candidate_first":
            game["orientation"] = "candidate_red"
            game["candidate_color"] = "RED"
            game["baseline_color"] = "BLUE"
        else:
            game["orientation"] = "candidate_blue"
            game["candidate_color"] = "BLUE"
            game["baseline_color"] = "RED"


def test_high_regret_builder_derives_source_from_passing_report(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    report = tmp_path / "high-regret.report.json"
    _json(report, _high_regret_report(tmp_path, candidate, champion))

    value = artifacts.build_high_regret_source(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["schema_version"] == promotion.HIGH_REGRET_SCHEMA
    assert value["verdict"] == "H1"
    assert value["complete_pairs"] == 200
    assert value["report"] == _ref(report)
    assert (
        value["suite_manifest"]
        == _high_regret_report(tmp_path, candidate, champion)["suite_manifest"]
    )
    assert value["pair_diagnostics"]["ww_pairs"] == 200


def test_high_regret_builder_excludes_one_legitimate_truncated_pair(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion, pairs=600)
    _use_color_orientations(raw)
    _truncate_high_regret_pair(raw)
    report = tmp_path / "high-regret-truncated.report.json"
    _json(report, raw)

    value = artifacts.build_high_regret_source(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["complete_pairs"] == 599
    assert value["pair_diagnostics"] == {
        "ww_pairs": 599,
        "split_pairs": 0,
        "ll_pairs": 0,
        "incomplete_pairs": 1,
    }
    assert value["pentanomial_sprt"] == raw["pentanomial_sprt"]


def test_held_out_suite_is_deterministic_and_derived_from_manifest(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "current_producer"
    shard_dir.mkdir()
    shard = shard_dir / "shard.npz"
    np.savez(
        shard,
        game_seed=np.arange(10_000, 10_200, dtype=np.int64),
        decision_index=np.zeros(200, dtype=np.int32),
        action_taken=np.arange(200, dtype=np.int32),
    )
    manifest = tmp_path / "regret.npz"
    np.savez(
        manifest,
        shard_id=np.zeros(200, dtype=np.int32),
        row_index=np.arange(200, dtype=np.int32),
        game_seed=np.arange(10_000, 10_200, dtype=np.int64),
        decision_index=np.zeros(200, dtype=np.int32),
        regret_score=np.linspace(0.0, 2.0, 200, dtype=np.float32),
        phase=np.asarray(
            [
                (
                    "BUILD_INITIAL_SETTLEMENT",
                    "MOVE_ROBBER",
                    "ROLL",
                    "BUILD_ROAD",
                )[index % 4]
                for index in range(200)
            ]
        ),
        legal_count=np.full(200, 54, dtype=np.int32),
        shard_paths=np.asarray([str(shard)]),
    )

    first = artifacts.build_held_out_high_regret_suite(
        manifest_path=manifest,
        holdout_fraction=0.75,
        holdout_seed=17,
        pairs=20,
    )
    second = artifacts.build_held_out_high_regret_suite(
        manifest_path=manifest,
        holdout_fraction=0.75,
        holdout_seed=17,
        pairs=20,
    )

    assert first == second
    assert len(first["states"]) == 20
    assert first["source_manifest"] == _ref(manifest)
    assert first["suite_sha256"] == promotion._digest_value(
        {key: value for key, value in first.items() if key != "suite_sha256"}
    )
    assert first["selection"]["selected_by_stratum"] == {
        "phase:opening": 4,
        "phase:robber_dev": 4,
        "phase:chance": 4,
        "phase:build_trade": 4,
        "41+": 4,
    }
    assert first["selection"]["replay_preflight"]["replay_complete_states"] > 20
    assert all(
        state["replay_source"]["scope"] == str(shard_dir)
        for state in first["states"]
    )


def test_held_out_suite_rejects_gaps_duplicates_and_partial_lanes(
    tmp_path: Path,
) -> None:
    valid_dir = tmp_path / "current_producer"
    gap_dir = tmp_path / "recent_history_gap"
    duplicate_dir = tmp_path / "hard_negative_duplicate"
    partial_dir = tmp_path / "recent_history_partial"
    negative_dir = tmp_path / "hard_negative_negative"
    for directory in (valid_dir, gap_dir, duplicate_dir, partial_dir, negative_dir):
        directory.mkdir()

    valid_seeds = np.arange(20_000, 20_024, dtype=np.int64)
    valid_rows = valid_dir / "rows.npz"
    # Twenty-four shallow games plus one valid deeper current-producer source.
    deep_seed = 29_999
    np.savez(
        valid_rows,
        game_seed=np.concatenate([valid_seeds, np.full(3, deep_seed)]),
        decision_index=np.concatenate(
            [np.zeros(len(valid_seeds), dtype=np.int32), np.arange(3, dtype=np.int32)]
        ),
        action_taken=np.arange(len(valid_seeds) + 3, dtype=np.int32),
    )
    gap_rows = gap_dir / "rows.npz"
    np.savez(
        gap_rows,
        game_seed=np.full(2, 30_001, dtype=np.int64),
        decision_index=np.asarray([0, 2], dtype=np.int32),
        action_taken=np.asarray([1, 2], dtype=np.int32),
    )
    duplicate_rows = duplicate_dir / "rows.npz"
    np.savez(
        duplicate_rows,
        game_seed=np.full(3, 30_002, dtype=np.int64),
        decision_index=np.asarray([0, 0, 1], dtype=np.int32),
        action_taken=np.asarray([1, 2, 3], dtype=np.int32),
    )
    partial_rows = partial_dir / "rows.npz"
    np.savez(
        partial_rows,
        game_seed=np.full(2, 30_003, dtype=np.int64),
        decision_index=np.asarray([4, 5], dtype=np.int32),
        action_taken=np.asarray([1, 2], dtype=np.int32),
    )
    negative_rows = negative_dir / "rows.npz"
    np.savez(
        negative_rows,
        game_seed=np.full(2, 30_004, dtype=np.int64),
        decision_index=np.asarray([-1, 0], dtype=np.int32),
        action_taken=np.asarray([1, 2], dtype=np.int32),
    )

    source_paths = [valid_rows, gap_rows, duplicate_rows, partial_rows, negative_rows]
    candidate_seeds = list(valid_seeds) + [
        deep_seed,
        30_001,
        30_002,
        30_003,
        30_004,
    ]
    candidate_decisions = [0] * len(valid_seeds) + [2, 2, 1, 5, 0]
    candidate_shards = [0] * (len(valid_seeds) + 1) + [1, 2, 3, 4]
    candidate_rows = list(range(len(valid_seeds))) + [
        len(valid_seeds) + 2,
        1,
        2,
        1,
        1,
    ]
    phases = [
        (
            "BUILD_INITIAL_SETTLEMENT",
            "MOVE_ROBBER",
            "ROLL",
            "BUILD_ROAD",
        )[index % 4]
        for index in range(len(candidate_seeds))
    ]
    manifest = tmp_path / "regret.npz"
    np.savez(
        manifest,
        shard_id=np.asarray(candidate_shards, dtype=np.int32),
        row_index=np.asarray(candidate_rows, dtype=np.int32),
        game_seed=np.asarray(candidate_seeds, dtype=np.int64),
        decision_index=np.asarray(candidate_decisions, dtype=np.int32),
        # Invalid partial sources rank first and must still be excluded.
        regret_score=np.asarray(
            [float(index) for index in range(len(candidate_seeds))], dtype=np.float32
        ),
        phase=np.asarray(phases),
        legal_count=np.full(len(candidate_seeds), 54, dtype=np.int32),
        shard_paths=np.asarray([str(path) for path in source_paths]),
    )

    suite = artifacts.build_held_out_high_regret_suite(
        manifest_path=manifest,
        holdout_fraction=0.999999999,
        holdout_seed=17,
        pairs=20,
    )

    identities = {
        (state["game_seed"], state["decision_index"]) for state in suite["states"]
    }
    assert (deep_seed, 2) in identities
    assert not identities.intersection(
        {(30_001, 2), (30_002, 1), (30_003, 5), (30_004, 0)}
    )
    preflight = suite["selection"]["replay_preflight"]
    assert preflight["rejected_noncontiguous"] == 4


def test_held_out_suite_fails_clearly_when_replay_complete_pool_is_too_small(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "recent_history" / "rows.npz"
    partial.parent.mkdir()
    np.savez(
        partial,
        game_seed=np.arange(40_000, 40_040, dtype=np.int64),
        decision_index=np.full(40, 3, dtype=np.int32),
        action_taken=np.arange(40, dtype=np.int32),
    )
    manifest = tmp_path / "regret.npz"
    np.savez(
        manifest,
        shard_id=np.zeros(40, dtype=np.int32),
        row_index=np.arange(40, dtype=np.int32),
        game_seed=np.arange(40_000, 40_040, dtype=np.int64),
        decision_index=np.full(40, 3, dtype=np.int32),
        regret_score=np.ones(40, dtype=np.float32),
        phase=np.asarray(["BUILD_ROAD"] * 40),
        legal_count=np.full(40, 54, dtype=np.int32),
        shard_paths=np.asarray([str(partial)]),
    )

    with pytest.raises(
        artifacts.ArtifactBuildError, match="after replay-completeness preflight"
    ):
        artifacts.build_held_out_high_regret_suite(
            manifest_path=manifest,
            holdout_fraction=0.999999999,
            holdout_seed=17,
            pairs=20,
        )


def test_bucket_report_is_extracted_from_retained_high_regret_games(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    report = tmp_path / "high-regret.report.json"
    _json(report, _high_regret_report(tmp_path, candidate, champion))

    value = artifacts.build_bucket_game_report(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["schema_version"] == artifacts.BUCKET_GAME_REPORT_SCHEMA
    assert len(value["games"]) == 400
    assert value["games"][0] == {
        "pair_id": 0,
        "orientation": "candidate_first",
        "candidate_won": True,
        "buckets": ["close", "phase:BUILD"],
    }


def test_bucket_report_excludes_both_orientations_of_truncated_pair(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion, pairs=600)
    _use_color_orientations(raw)
    _truncate_high_regret_pair(raw)
    report = tmp_path / "high-regret-truncated.report.json"
    _json(report, raw)

    value = artifacts.build_bucket_game_report(
        report_path=report, candidate=candidate, champion=champion
    )

    assert len(value["games"]) == 1_198
    assert {game["pair_id"] for game in value["games"]} == set(range(1, 600))
    assert value["games"][0]["orientation"] == "candidate_red"
    assert value["games"][0]["candidate_color"] == "RED"
    assert value["games"][0]["baseline_color"] == "BLUE"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["games"][0].update(candidate_color="BLUE"),
            "orientation does not bind",
        ),
        (
            lambda value: value["games"][0].update(
                orientation="candidate_first",
                candidate_color="RED",
                baseline_color="BLUE",
            ),
            "mixes orientation encodings",
        ),
    ],
)
def test_high_regret_builder_rejects_forged_or_mixed_color_orientations(
    tmp_path: Path, mutation, message: str
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    _use_color_orientations(raw)
    mutation(raw)
    report = tmp_path / "invalid-color-orientation.report.json"
    _json(report, raw)

    with pytest.raises(artifacts.ArtifactBuildError, match=message):
        artifacts.build_high_regret_source(
            report_path=report, candidate=candidate, champion=champion
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["games"][0].update(candidate_won=None),
            "nontruncated game",
        ),
        (
            lambda value: value["games"][0].update(truncated=True),
            "truncated game must have candidate_won=null",
        ),
        (
            lambda value: value["games"].pop(1),
            "must contain both orientations",
        ),
    ],
)
def test_high_regret_builder_rejects_mislabeled_or_half_pairs(
    tmp_path: Path, mutation, message: str
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    mutation(raw)
    report = tmp_path / "invalid-high-regret.report.json"
    _json(report, raw)

    with pytest.raises(artifacts.ArtifactBuildError, match=message):
        artifacts.build_high_regret_source(
            report_path=report, candidate=candidate, champion=champion
        )


def test_high_regret_builder_rejects_stale_truncation_statistics(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    raw["games"][0].update(candidate_won=None, truncated=True)
    report = tmp_path / "stale-high-regret.report.json"
    _json(report, raw)

    with pytest.raises(artifacts.ArtifactBuildError, match="paired statistics"):
        artifacts.build_high_regret_source(
            report_path=report, candidate=candidate, champion=champion
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("c_scale", 0.03),
        ("candidate_c_scale", 0.03),
        ("baseline_c_scale", 0.10),
    ],
)
def test_high_regret_builder_rejects_forged_role_scales(
    tmp_path: Path, key: str, value: float
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    raw["evaluation_config"][key] = value
    report = tmp_path / f"forged-{key}.report.json"
    _json(report, raw)

    with pytest.raises(artifacts.ArtifactBuildError, match=key):
        artifacts.build_high_regret_source(
            report_path=report, candidate=candidate, champion=champion
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(held_out=False), "not the held-out suite"),
        (lambda value: value.update(errors=["boom"]), "evaluation errors"),
        (
            lambda value: value["games"].__setitem__(
                0, {**value["games"][0], "candidate_won": False}
            ),
            "paired statistics do not replay",
        ),
        (
            lambda value: value["candidate"].update(sha256="sha256:" + "0" * 64),
            "does not bind the expected checkpoint bytes",
        ),
    ],
)
def test_high_regret_builder_refuses_non_evidence(
    tmp_path: Path, mutation, message: str
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    value = _high_regret_report(tmp_path, candidate, champion)
    mutation(value)
    report = tmp_path / "report.json"
    _json(report, value)
    with pytest.raises(artifacts.ArtifactBuildError, match=message):
        artifacts.build_high_regret_source(
            report_path=report, candidate=candidate, champion=champion
        )


def _bucket_report(candidate: Path, champion: Path) -> dict:
    return {
        "schema_version": artifacts.BUCKET_GAME_REPORT_SCHEMA,
        "candidate": _ref(candidate),
        "champion": _ref(champion),
        "errors": [],
        "games": [
            {
                "pair_id": index,
                "orientation": "candidate_first",
                "candidate_won": index < 6,
                "buckets": ["opening"],
            }
            for index in range(8)
        ]
        + [
            {
                "pair_id": 100 + index,
                "orientation": "candidate_second",
                "candidate_won": index < 9,
                "buckets": ["41+"],
            }
            for index in range(10)
        ],
    }


def test_bucket_builder_computes_pass_from_counts(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    report = tmp_path / "bucket-games.json"
    _json(report, _bucket_report(candidate, champion))

    value = artifacts.build_bucket_veto_source(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["veto"] is False
    assert value["veto_buckets"] == []
    assert value["per_bucket"]["opening"] == {
        "status": "pass",
        "n": 8,
        "winrate": 0.75,
    }


def test_bucket_builder_preserves_fail_and_insufficient_data(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _bucket_report(candidate, champion)
    raw["games"] = [
        {
            "pair_id": index,
            "orientation": "candidate_first",
            "candidate_won": index < 2,
            "buckets": ["opening"],
        }
        for index in range(8)
    ] + [
        {
            "pair_id": 9,
            "orientation": "candidate_first",
            "candidate_won": True,
            "buckets": ["rare"],
        }
    ]
    report = tmp_path / "bucket-games.json"
    _json(report, raw)

    value = artifacts.build_bucket_veto_source(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["veto"] is True
    assert value["veto_buckets"] == ["opening"]
    assert value["per_bucket"]["opening"]["status"] == "fail"
    assert value["per_bucket"]["rare"]["status"] == "insufficient_data"


def test_bucket_builder_uses_fixed_five_percent_regression_limit(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _bucket_report(candidate, champion)
    raw["games"] = [
        {
            "pair_id": index,
            "orientation": "candidate_first",
            "candidate_won": (index - offset) < wins,
            "buckets": [label],
        }
        for label, wins, offset in (("at_limit", 9, 0), ("over_limit", 8, 100))
        for index in range(offset, offset + 20)
    ]
    report = tmp_path / "bucket-games.json"
    _json(report, raw)

    value = artifacts.build_bucket_veto_source(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["per_bucket"]["at_limit"]["status"] == "pass"
    assert value["per_bucket"]["over_limit"]["status"] == "fail"
    assert value["veto_buckets"] == ["over_limit"]


def test_bucket_builder_refuses_duplicate_games(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _bucket_report(candidate, champion)
    raw["games"].append(dict(raw["games"][0]))
    report = tmp_path / "bucket-games.json"
    _json(report, raw)
    with pytest.raises(artifacts.ArtifactBuildError, match="duplicate games"):
        artifacts.build_bucket_veto_source(
            report_path=report, candidate=candidate, champion=champion
        )


def _legacy_incumbent_inputs(
    tmp_path: Path, *, historical_checkpoint: str
) -> tuple[Path, Path, Path, dict]:
    champion = tmp_path / "runs" / "bc" / "gen3_20260706" / "checkpoint.pt"
    champion.parent.mkdir(parents=True)
    champion.write_bytes(b"champion")
    calibration = tmp_path / "champion-calibration.json"
    _json(
        calibration,
        {
            "schema_version": "phase-sliced-value-calibration-v2",
            "checkpoint": str(champion.resolve()),
            "value_readout": "scalar",
            "readout_provenance": {
                "requested_readout": "scalar",
                "trained_value_readouts": ["scalar"],
                "optimizer_steps": None,
                "completed_epochs": None,
            },
        },
    )
    historical = tmp_path / "reports" / "archive" / "gen3-training-report.json"
    historical.parent.mkdir(parents=True)
    _json(
        historical,
        {
            "checkpoint": historical_checkpoint,
            "checkpoint_sha256": promotion._sha256(champion),
            "steps_completed": 912,
            "epochs": 1,
        },
    )
    contract = {
        "contract_sha256": "sha256:" + "a" * 64,
        "checkpoints": [
            {
                "role": "producer",
                "path": str(champion.resolve()),
                "sha256": promotion._sha256(champion),
            }
        ],
    }
    return champion, calibration, historical, contract


def test_legacy_incumbent_builder_resolves_historical_checkpoint_from_report_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    champion, calibration, historical, contract = _legacy_incumbent_inputs(
        tmp_path, historical_checkpoint="runs/bc/gen3_20260706/checkpoint.pt"
    )
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    value = artifacts.build_legacy_incumbent_calibration_source(
        calibration_path=calibration,
        historical_training_report=historical,
        contract=contract,
        champion=champion,
    )

    assert value["legacy_incumbent_provenance"] == {
        "schema_version": promotion.LEGACY_INCUMBENT_PROVENANCE_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": promotion._sha256(champion),
        "historical_training_report": _ref(historical),
    }


def test_legacy_incumbent_builder_accepts_exact_absolute_historical_checkpoint(
    tmp_path: Path,
) -> None:
    absolute_champion = (
        tmp_path / "runs" / "bc" / "gen3_20260706" / "checkpoint.pt"
    ).resolve()
    champion, calibration, historical, contract = _legacy_incumbent_inputs(
        tmp_path, historical_checkpoint=str(absolute_champion)
    )

    value = artifacts.build_legacy_incumbent_calibration_source(
        calibration_path=calibration,
        historical_training_report=historical,
        contract=contract,
        champion=champion,
    )

    assert value["legacy_incumbent_provenance"]["checkpoint_sha256"] == (
        promotion._sha256(champion)
    )


@pytest.mark.parametrize(
    "declared",
    ["../../runs/bc/gen3_20260706/checkpoint.pt", "checkpoint.pt"],
)
def test_legacy_incumbent_builder_refuses_traversal_and_unrelated_suffixes(
    tmp_path: Path, declared: str
) -> None:
    champion, calibration, historical, contract = _legacy_incumbent_inputs(
        tmp_path, historical_checkpoint=declared
    )

    with pytest.raises(artifacts.ArtifactBuildError, match="historical report checkpoint"):
        artifacts.build_legacy_incumbent_calibration_source(
            calibration_path=calibration,
            historical_training_report=historical,
            contract=contract,
            champion=champion,
        )


def test_legacy_incumbent_builder_refuses_adjacent_bare_checkpoint_name(
    tmp_path: Path,
) -> None:
    champion, calibration, historical, contract = _legacy_incumbent_inputs(
        tmp_path, historical_checkpoint="checkpoint.pt"
    )
    adjacent_report = champion.with_name("gen3-training-report.json")
    historical.replace(adjacent_report)

    with pytest.raises(artifacts.ArtifactBuildError, match="multi-component"):
        artifacts.build_legacy_incumbent_calibration_source(
            calibration_path=calibration,
            historical_training_report=adjacent_report,
            contract=contract,
            champion=champion,
        )


@pytest.mark.parametrize("link_kind", ["final", "ancestor", "broken_ancestor"])
def test_legacy_incumbent_builder_refuses_symlinked_relative_matches(
    tmp_path: Path, link_kind: str
) -> None:
    declared = "runs/bc/gen3_20260706/checkpoint.pt"
    champion, calibration, historical, contract = _legacy_incumbent_inputs(
        tmp_path, historical_checkpoint=declared
    )
    decoy_root = historical.parent
    if link_kind == "final":
        link = decoy_root / declared
        link.parent.mkdir(parents=True)
        link.symlink_to(champion)
    elif link_kind == "ancestor":
        link = decoy_root / "runs"
        link.symlink_to(tmp_path / "runs", target_is_directory=True)
    else:
        link = decoy_root / "runs"
        link.symlink_to(tmp_path / "missing-runs", target_is_directory=True)

    with pytest.raises(artifacts.ArtifactBuildError, match="must not contain symlinks"):
        artifacts.build_legacy_incumbent_calibration_source(
            calibration_path=calibration,
            historical_training_report=historical,
            contract=contract,
            champion=champion,
        )


def test_legacy_incumbent_builder_translates_absolute_symlink_loop(
    tmp_path: Path,
) -> None:
    loop = tmp_path / "checkpoint-loop.pt"
    loop.symlink_to(loop)
    champion, calibration, historical, contract = _legacy_incumbent_inputs(
        tmp_path, historical_checkpoint=str(loop)
    )

    with pytest.raises(artifacts.ArtifactBuildError, match="cannot resolve"):
        artifacts.build_legacy_incumbent_calibration_source(
            calibration_path=calibration,
            historical_training_report=historical,
            contract=contract,
            champion=champion,
        )


def test_legacy_incumbent_builder_refuses_ambiguous_ancestor_matches(
    tmp_path: Path,
) -> None:
    declared = "runs/bc/gen3_20260706/checkpoint.pt"
    champion, calibration, historical, contract = _legacy_incumbent_inputs(
        tmp_path, historical_checkpoint=declared
    )
    decoy = historical.parent / declared
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"unrelated checkpoint")

    with pytest.raises(artifacts.ArtifactBuildError, match="is ambiguous"):
        artifacts.build_legacy_incumbent_calibration_source(
            calibration_path=calibration,
            historical_training_report=historical,
            contract=contract,
            champion=champion,
        )


def test_legacy_incumbent_builder_refuses_nonproducer_checkpoint(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    calibration = tmp_path / "calibration.json"
    _json(
        calibration,
        {
            "schema_version": "phase-sliced-value-calibration-v2",
            "checkpoint": str(champion.resolve()),
            "value_readout": "scalar",
            "readout_provenance": {
                "optimizer_steps": None,
                "completed_epochs": None,
            },
        },
    )
    historical = tmp_path / "historical.json"
    _json(
        historical,
        {"checkpoint": str(champion.resolve()), "steps_completed": 9, "epochs": 1},
    )
    with pytest.raises(artifacts.ArtifactBuildError, match="contract-bound producer"):
        artifacts.build_legacy_incumbent_calibration_source(
            calibration_path=calibration,
            historical_training_report=historical,
            contract={
                "contract_sha256": "sha256:" + "a" * 64,
                "checkpoints": [
                    {
                        "role": "producer",
                        "path": str(candidate.resolve()),
                        "sha256": promotion._sha256(candidate),
                    }
                ],
            },
            champion=champion,
        )


def _contract() -> dict:
    return {
        "contract_sha256": "sha256:" + "a" * 64,
        "science": {"learner_value_objective": {"value_readout": "scalar"}},
    }


@pytest.mark.parametrize(
    ("kind", "verdict", "result"),
    [
        (
            "mechanism_calibration",
            "pass",
            {"value_readout": "scalar", "max_rmse_regression": 0.02},
        ),
        ("internal_h2h", "H1", {}),
        ("external_panel", "pass", {"max_win_rate_regression": 0.02}),
        ("high_regret", "pass", {}),
        ("bucket_veto", "pass", {}),
    ],
)
def test_evidence_builder_seals_fixed_policy(
    tmp_path: Path, kind: str, verdict: str, result: dict
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    source = tmp_path / "source.json"
    _json(source, {"real": "source"})

    value = artifacts.build_evidence_envelope(
        kind=kind,
        contract=_contract(),
        candidate=candidate,
        champion=champion,
        sources=[("raw", source)],
    )

    assert value["verdict"] == verdict
    assert value["result"] == result
    digest = value.pop("evidence_sha256")
    assert digest == promotion._digest_value(value)


def test_evidence_prewrite_replays_transaction_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    source = tmp_path / "source.json"
    _json(source, {"real": "source"})
    value = artifacts.build_evidence_envelope(
        kind="high_regret",
        contract=_contract(),
        candidate=candidate,
        champion=champion,
        sources=[("high_regret", source)],
    )
    called: list[Path] = []

    def verify(path: Path, **_kwargs):
        called.append(path)
        assert json.loads(path.read_text()) == value

    monkeypatch.setattr(promotion, "_verify_promotion_evidence", verify)
    monkeypatch.setattr(
        promotion, "_sealed_evaluation_semantics", lambda _contract: {"c_scale": 0.03}
    )
    artifacts._validate_envelope_before_write(
        tmp_path / "final.json",
        value=value,
        kind="high_regret",
        contract=_contract(),
        candidate=candidate,
        champion=champion,
    )
    assert len(called) == 1
    assert not called[0].exists()


def test_adjudication_builder_derives_every_third_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        promotion, "_sealed_evaluation_semantics", lambda _contract: {"c_scale": 0.03}
    )
    candidate, champion = _checkpoints(tmp_path)
    report = tmp_path / "training.json"
    _json(report, {"training": "report"})
    receipt = tmp_path / "receipt.json"
    _json(receipt, {"receipt": True})
    pointer = tmp_path / "CURRENT_CHAMPION"
    pointer.write_text(str(champion.resolve()) + "\n", encoding="utf-8")
    registry = ChampionRegistry(tmp_path / "registry.json")
    registry.record_promotion("generator_champion")
    registry.record_promotion("generator_champion")
    evidence = []
    for kind in sorted(promotion.REQUIRED_EVIDENCE_KINDS):
        path = tmp_path / f"{kind}.json"
        _json(path, {"kind": kind})
        evidence.append((kind, path))

    value = artifacts.build_adjudication(
        contract=_contract(),
        contract_lock=tmp_path / "contract.json",
        training_receipt=receipt,
        registry=registry,
        current_pointer=pointer,
        candidate=candidate,
        candidate_version=5,
        training_report=report,
        champion=champion,
        champion_version=4,
        evidence=evidence,
        nth_confirmation_passed=True,
    )

    assert value["nth_confirmation_required"] is True
    assert value["nth_confirmation_passed"] is True
    assert value["candidate"]["agent_identity"]["search_config"]["c_scale"] == 0.10
    assert value["champion"]["agent_identity"]["search_config"]["c_scale"] == 0.03
    digest = value.pop("adjudication_sha256")
    assert digest == promotion._digest_value(value)


def test_adjudication_builder_refuses_missing_evidence(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    with pytest.raises(
        artifacts.ArtifactBuildError, match="each promotion evidence kind"
    ):
        artifacts.build_adjudication(
            contract=_contract(),
            contract_lock=tmp_path / "contract.json",
            training_receipt=tmp_path / "receipt.json",
            registry=ChampionRegistry(tmp_path / "registry.json"),
            current_pointer=tmp_path / "CURRENT_CHAMPION",
            candidate=candidate,
            candidate_version=5,
            training_report=tmp_path / "report.json",
            champion=champion,
            champion_version=4,
            evidence=[],
            nth_confirmation_passed=False,
        )


def test_cli_writes_fresh_readonly_artifact_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    report = tmp_path / "report.json"
    _json(report, _high_regret_report(tmp_path, candidate, champion))
    out = tmp_path / "high-regret.source.json"
    argv = [
        "high-regret",
        "--report",
        str(report),
        "--candidate",
        str(candidate),
        "--champion",
        str(champion),
        "--out",
        str(out),
    ]

    assert artifacts.main(argv) == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o444
    before = out.read_bytes()
    assert artifacts.main(argv) == 2
    assert out.read_bytes() == before
