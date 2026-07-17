from __future__ import annotations

import copy
import json
import hashlib
import stat
from pathlib import Path

import pytest
import numpy as np

from tools import a1_promotion_artifacts as artifacts
from tools import a1_promotion_transaction as promotion
from tools.champion_registry import ChampionRegistry
from tools.regret_common import (
    H2H_SEARCH_RNG_CONTRACT,
    derive_promotion_bucket_labels,
    h2h_search_seed,
)


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


def _bucket_game(
    pair_id: int,
    orientation: str,
    *,
    candidate_won: bool,
    phase: str = "BUILD_ROAD",
    max_legal_count: int = 12,
    blowout: bool = False,
) -> dict:
    candidate_color, baseline_color = (
        ("RED", "BLUE") if orientation == "candidate_red" else ("BLUE", "RED")
    )
    game_seed = 50_000 + pair_id
    winner = candidate_color if candidate_won else baseline_color
    loser = baseline_color if candidate_won else candidate_color
    actual = {winner: 10, loser: 4 if blowout else 8}
    game = {
        "pair_id": pair_id,
        "game_seed": game_seed,
        "orientation": orientation,
        "search_seeds_by_role": {
            "candidate": h2h_search_seed(
                game_seed=game_seed, seat_color=candidate_color
            ),
            "baseline": h2h_search_seed(
                game_seed=game_seed, seat_color=baseline_color
            ),
        },
        "candidate_color": candidate_color,
        "baseline_color": baseline_color,
        "candidate_won": candidate_won,
        "winner": winner,
        "terminated": True,
        "truncated": False,
        "final_public_vps": dict(actual),
        "final_actual_vps": actual,
        "archived_phase": phase,
        "phases_seen": [phase],
        "max_legal_count": max_legal_count,
    }
    game["buckets"] = derive_promotion_bucket_labels(game)
    return game


def _set_bucket_game_outcome(game: dict, candidate_won: bool) -> None:
    candidate = game["candidate_color"]
    baseline = game["baseline_color"]
    winner = candidate if candidate_won else baseline
    loser = baseline if candidate_won else candidate
    margin = (
        4
        if "blowout" in game["buckets"]
        else 8
    )
    actual = {winner: 10, loser: margin}
    game.update(
        candidate_won=candidate_won,
        winner=winner,
        final_public_vps=dict(actual),
        final_actual_vps=actual,
    )
    game["buckets"] = derive_promotion_bucket_labels(game)


def test_full_game_is_multi_label_across_visited_promotion_phases() -> None:
    labels = derive_promotion_bucket_labels(
        {
            "candidate_color": "RED",
            "baseline_color": "BLUE",
            "final_actual_vps": {"RED": 10, "BLUE": 8},
            "archived_phase": "",
            "phases_seen": [
                "BUILD_INITIAL_SETTLEMENT",
                "BUILD_INITIAL_ROAD",
                "ROLL",
                "PLAY_TURN",
            ],
            "max_legal_count": 20,
        }
    )

    assert "phase:initial_settlement" in labels
    assert "phase:initial_road" in labels
    assert "phase:chance" in labels
    assert "phase:build_trade" in labels
    assert "opening" in labels


def test_cohort_exclusions_are_derived_from_candidate_bound_game_seeds(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    source = tmp_path / "arm-selection.json"
    _json(
        source,
        {
            "candidate_checkpoint_sha256": promotion._sha256(candidate),
            "games": [
                {"game_seed": 100},
                {"game_seed": 100},
                {"game_seed": 101},
                {"game_seed": 104},
            ],
        },
    )
    contract = {"contract_sha256": "sha256:" + "a" * 64}
    value = artifacts.build_cohort_exclusions(
        contract=contract,
        candidate=candidate,
        cohorts=[("p1-arm-selection", "internal_h2h", source)],
    )
    assert value["candidate_sha256"] == promotion._sha256(candidate)
    assert value["cohorts"][0]["seed_intervals"] == [
        {"base_seed": 100, "end_seed": 102},
        {"base_seed": 104, "end_seed": 105},
    ]
    unhashed = dict(value)
    assert unhashed.pop("manifest_sha256") == promotion._digest_value(unhashed)


def test_cohort_exclusions_accept_recipe_selection_ancestor(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    source = tmp_path / "other.json"
    _json(
        source,
        {
            "candidate_checkpoint_sha256": "sha256:" + "b" * 64,
            "games": [{"game_seed": 100}],
        },
    )
    value = artifacts.build_cohort_exclusions(
        contract={"contract_sha256": "sha256:" + "a" * 64},
        candidate=candidate,
        cohorts=[("ancestor", "internal_h2h", source)],
    )
    assert value["candidate_sha256"] == promotion._sha256(candidate)
    assert value["cohorts"][0]["seed_intervals"] == [
        {"base_seed": 100, "end_seed": 101}
    ]


def test_cohort_exclusions_reject_source_without_candidate_binding(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    source = tmp_path / "unbound.json"
    _json(source, {"games": [{"game_seed": 100}]})
    with pytest.raises(artifacts.ArtifactBuildError, match="no explicit candidate"):
        artifacts.build_cohort_exclusions(
            contract={"contract_sha256": "sha256:" + "a" * 64},
            candidate=candidate,
            cohorts=[("unbound", "internal_h2h", source)],
        )


def _held_out_npz(tmp_path: Path, seeds) -> dict:
    values = np.sort(np.unique(np.asarray(seeds, dtype=np.int64)))
    manifest = tmp_path / "validation-seeds.json"
    payload = {
        "schema_version": "train-validation-game-seeds-v1",
        "game_seeds": values.tolist(),
        "validation_game_seed_count": len(values),
        "validation_game_seed_set_sha256": "sha256:"
        + hashlib.sha256(values.astype("<i8").tobytes()).hexdigest(),
    }
    _json(manifest, payload)
    return {
        "held_out_only": np.asarray(True),
        "validation_seed_manifest_path": np.asarray(str(manifest.resolve())),
        "validation_seed_manifest_sha256": np.asarray(promotion._sha256(manifest)),
        "validation_seed_manifest_schema_version": np.asarray(payload["schema_version"]),
        "validation_game_seed_count": np.asarray(len(values), dtype=np.int64),
        "validation_game_seed_set_sha256": np.asarray(
            payload["validation_game_seed_set_sha256"]
        ),
    }


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
            "algorithm": "trainer-validation-stratified-regret-unique-game-v4",
            "selection_scope": "full_authenticated_training_validation_manifest",
            "holdout_fraction": 1.0,
            "holdout_seed": 17,
            "eligible_unique_states": pairs,
            "eligible_unique_games": pairs,
            "replay_complete_unique_games": pairs,
            "selected_unique_games": pairs,
            "selected_pairs": pairs,
            "stratum_min_pairs": 20,
            "selected_by_stratum": {
                "phase:initial_settlement": 20,
                "phase:initial_road": 20,
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
                    "BUILD_INITIAL_ROAD",
                    "MOVE_ROBBER",
                    "ROLL",
                    "BUILD_ROAD",
                )[pair % 5],
                "legal_count": 54 if pair < 20 else 12,
                "regret_score": 1.0,
            }
            for pair in range(pairs)
        ],
    }
    suite_payload["suite_sha256"] = promotion._digest_value(suite_payload)
    _json(suite, suite_payload)
    games = []
    phases = (
        "BUILD_INITIAL_SETTLEMENT",
        "BUILD_INITIAL_ROAD",
        "MOVE_ROBBER",
        "ROLL",
        "BUILD_ROAD",
    )
    for pair in range(pairs):
        for orientation in ("candidate_red", "candidate_blue"):
            game = _bucket_game(
                pair,
                orientation,
                candidate_won=True,
                phase=phases[pair % len(phases)],
                max_legal_count=54 if pair < 20 else 12,
                blowout=pair % 2 == 0,
            )
            game.update(
                archived_game_seed=50_000 + pair,
                archived_decision_index=pair % 20,
            )
            games.append(game)
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
            "c_scale": promotion.BOOTSTRAP_CANDIDATE_C_SCALE,
            "candidate_c_scale": promotion.BOOTSTRAP_CANDIDATE_C_SCALE,
            "baseline_c_scale": promotion.BOOTSTRAP_CHAMPION_C_SCALE,
            "candidate_n_full": 128,
            "baseline_n_full": 128,
            "p_full": 1.0,
            "force_full_every_decision": True,
        },
        "planned_engine_identity": {
            "schema_version": promotion.HIGH_REGRET_ENGINE_IDENTITY_SCHEMA,
            "repo_commit": "a" * 40,
            "native_wheel_sha256": "sha256:" + "b" * 64,
            "evaluator_sha256": "sha256:" + "c" * 64,
            "replay_sha256": "sha256:" + "d" * 64,
        },
        "engine_identity": {
            "schema_version": promotion.HIGH_REGRET_ENGINE_IDENTITY_SCHEMA,
            "repo_commit": "a" * 40,
            "native_wheel_sha256": "sha256:" + "b" * 64,
            "evaluator_sha256": "sha256:" + "c" * 64,
            "replay_sha256": "sha256:" + "d" * 64,
            "native_runtime_sha256": "sha256:" + "e" * 64,
        },
        "archived_state_reconstruction": {
            "schema_version": promotion.ARCHIVED_STATE_RECONSTRUCTION_SCHEMA,
            "constructor": "catanatron_rs.Game.simple",
            "map_kind": "BASE",
            "action_prefix": "[0,target_decision)",
            "chance_stream": "random.Random(game_seed ^ 0xA17E)",
            "replay_contract": artifacts.REPLAY_CONTRACT,
        },
        "search_rng_contract": H2H_SEARCH_RNG_CONTRACT,
        "errors": [],
        "games": games,
        "pentanomial_sprt": pentanomial,
        "pair_diagnostics": diagnostics,
    }


def _set_high_regret_pair_counts(
    value: dict, *, ll_pairs: int, split_pairs: int, ww_pairs: int
) -> None:
    pairs = sorted({int(game["pair_id"]) for game in value["games"]})
    assert len(pairs) == ll_pairs + split_pairs + ww_pairs
    outcomes_by_pair = {
        pair_id: (
            (False, False)
            if index < ll_pairs
            else (True, False)
            if index < ll_pairs + split_pairs
            else (True, True)
        )
        for index, pair_id in enumerate(pairs)
    }
    for game in value["games"]:
        first, second = outcomes_by_pair[int(game["pair_id"])]
        _set_bucket_game_outcome(
            game,
            first
            if game["orientation"] in {"candidate_first", "candidate_red"}
            else second,
        )
    normalized = [
        {**game, "search_won": game["candidate_won"]} for game in value["games"]
    ]
    scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
    value["pair_diagnostics"] = diagnostics
    value["pentanomial_sprt"] = promotion.evaluate_pentanomial_sprt(
        scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )


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
        if game["orientation"] in {"candidate_red", "candidate_blue"}:
            continue
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(search_rng_contract={}),
            "corrected per-game/seat",
        ),
        (
            lambda value: value["games"][0]["search_seeds_by_role"].__setitem__(
                "candidate",
                value["games"][0]["search_seeds_by_role"]["candidate"] + 1,
            ),
            "role/seat binding",
        ),
    ],
)
def test_high_regret_builder_requires_schedule_invariant_search_rng(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    mutation(raw)
    report = tmp_path / "invalid-search-rng.report.json"
    _json(report, raw)

    with pytest.raises(artifacts.ArtifactBuildError, match=message):
        artifacts.build_high_regret_source(
            report_path=report,
            candidate=candidate,
            champion=champion,
        )


def test_high_regret_and_bucket_builders_accept_continue_as_nonregression(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    _set_high_regret_pair_counts(
        raw, ll_pairs=50, split_pairs=100, ww_pairs=50
    )
    assert raw["pentanomial_sprt"]["decision"] == "continue"
    report = tmp_path / "high-regret-continue.report.json"
    _json(report, raw)

    source = artifacts.build_high_regret_source(
        report_path=report, candidate=candidate, champion=champion
    )
    bucket_report = artifacts.build_bucket_game_report(
        report_path=report, candidate=candidate, champion=champion
    )

    assert source["passed"] is True
    assert source["verdict"] == "continue"
    assert len(bucket_report["games"]) == 400


def test_high_regret_and_bucket_builders_reject_h0_veto(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    _set_high_regret_pair_counts(raw, ll_pairs=0, split_pairs=200, ww_pairs=0)
    assert raw["pentanomial_sprt"]["decision"] == "H0"
    report = tmp_path / "high-regret-h0.report.json"
    _json(report, raw)

    with pytest.raises(artifacts.ArtifactBuildError, match="veto reached H0"):
        artifacts.build_high_regret_source(
            report_path=report, candidate=candidate, champion=champion
        )
    with pytest.raises(artifacts.ArtifactBuildError, match="veto reached H0"):
        artifacts.build_bucket_game_report(
            report_path=report, candidate=candidate, champion=champion
        )


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
        **_held_out_npz(tmp_path, np.arange(10_000, 10_200)),
        shard_id=np.zeros(200, dtype=np.int32),
        row_index=np.arange(200, dtype=np.int32),
        game_seed=np.arange(10_000, 10_200, dtype=np.int64),
        decision_index=np.zeros(200, dtype=np.int32),
        regret_score=np.linspace(0.0, 2.0, 200, dtype=np.float32),
        phase=np.asarray(
            [
                (
                    "BUILD_INITIAL_SETTLEMENT",
                    "BUILD_INITIAL_ROAD",
                    "MOVE_ROBBER",
                    "ROLL",
                    "BUILD_ROAD",
                )[index % 5]
                for index in range(200)
            ]
        ),
        legal_count=np.full(200, 54, dtype=np.int32),
        shard_paths=np.asarray([str(shard)]),
    )

    first = artifacts.build_held_out_high_regret_suite(
        manifest_path=manifest,
        holdout_fraction=1.0,
        holdout_seed=17,
        pairs=24,
    )
    second = artifacts.build_held_out_high_regret_suite(
        manifest_path=manifest,
        holdout_fraction=1.0,
        holdout_seed=17,
        pairs=24,
    )

    assert first == second
    assert len(first["states"]) == 24
    assert len({state["game_seed"] for state in first["states"]}) == 24
    assert first["selection"]["selected_unique_games"] == 24
    assert first["source_manifest"] == _ref(manifest)
    assert first["suite_sha256"] == promotion._digest_value(
        {key: value for key, value in first.items() if key != "suite_sha256"}
    )
    assert first["selection"]["selected_by_stratum"] == {
        "phase:initial_settlement": 4,
        "phase:initial_road": 4,
        "phase:robber_dev": 4,
        "phase:chance": 4,
        "phase:build_trade": 4,
        "41+": 4,
    }
    assert first["selection"]["replay_preflight"]["replay_complete_states"] > 24
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
            "BUILD_INITIAL_ROAD",
            "MOVE_ROBBER",
            "ROLL",
            "BUILD_ROAD",
        )[index % 5]
        for index in range(len(candidate_seeds))
    ]
    manifest = tmp_path / "regret.npz"
    np.savez(
        manifest,
        **_held_out_npz(tmp_path, candidate_seeds),
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
        holdout_fraction=1.0,
        holdout_seed=17,
        pairs=24,
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
        **_held_out_npz(tmp_path, np.arange(40_000, 40_040)),
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
            holdout_fraction=1.0,
            holdout_seed=17,
            pairs=24,
        )


def test_held_out_suite_refuses_any_non_validation_source_row(tmp_path: Path) -> None:
    shard = tmp_path / "rows.npz"
    np.savez(
        shard,
        game_seed=np.asarray([1, 2]),
        decision_index=np.asarray([0, 0]),
        action_taken=np.asarray([1, 2]),
    )
    manifest = tmp_path / "regret.npz"
    np.savez(
        manifest,
        **_held_out_npz(tmp_path, [1]),
        shard_id=np.asarray([0, 0]),
        row_index=np.asarray([0, 1]),
        game_seed=np.asarray([1, 2]),
        decision_index=np.asarray([0, 0]),
        regret_score=np.asarray([1.0, 0.5]),
        phase=np.asarray(["ROLL", "ROLL"]),
        legal_count=np.asarray([54, 54]),
        shard_paths=np.asarray([str(shard)]),
    )
    with pytest.raises(artifacts.ArtifactBuildError, match="non-validation"):
        artifacts.build_held_out_high_regret_suite(
            manifest_path=manifest,
            holdout_fraction=1.0,
            holdout_seed=17,
            pairs=24,
        )


def test_held_out_suite_refuses_validation_manifest_drift(tmp_path: Path) -> None:
    shard = tmp_path / "rows.npz"
    np.savez(
        shard,
        game_seed=np.asarray([1]),
        decision_index=np.asarray([0]),
        action_taken=np.asarray([1]),
    )
    fields = _held_out_npz(tmp_path, [1])
    manifest = tmp_path / "regret.npz"
    np.savez(
        manifest,
        **fields,
        shard_id=np.asarray([0]),
        row_index=np.asarray([0]),
        game_seed=np.asarray([1]),
        decision_index=np.asarray([0]),
        regret_score=np.asarray([1.0]),
        phase=np.asarray(["ROLL"]),
        legal_count=np.asarray([54]),
        shard_paths=np.asarray([str(shard)]),
    )
    (tmp_path / "validation-seeds.json").write_text("{}", encoding="utf-8")
    with pytest.raises(artifacts.ArtifactBuildError, match="unsupported"):
        artifacts.build_held_out_high_regret_suite(
            manifest_path=manifest,
            holdout_fraction=1.0,
            holdout_seed=17,
            pairs=24,
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
    first = value["games"][0]
    assert first["pair_id"] == 0
    assert first["orientation"] == "candidate_red"
    assert first["candidate_color"] == "RED"
    assert first["baseline_color"] == "BLUE"
    assert first["final_actual_vps"] == {"RED": 10, "BLUE": 4}
    assert first["buckets"] == [
        "41+", "blowout", "opening", "phase:initial_settlement"
    ]


def test_bucket_report_rejects_forged_close_label_for_blowout(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    game = raw["games"][0]
    assert "blowout" in game["buckets"]
    game["buckets"] = sorted(
        "close" if label == "blowout" else label
        for label in game["buckets"]
    )
    report = tmp_path / "forged-margin-label.report.json"
    _json(report, raw)

    with pytest.raises(artifacts.ArtifactBuildError, match="labels do not replay"):
        artifacts.build_bucket_game_report(
            report_path=report,
            candidate=candidate,
            champion=champion,
        )


def test_bucket_report_preserves_hidden_public_vp_below_actual(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _high_regret_report(tmp_path, candidate, champion)
    game = raw["games"][0]
    game["final_public_vps"][game["candidate_color"]] = 8
    report = tmp_path / "hidden-vp.report.json"
    _json(report, raw)

    value = artifacts.build_bucket_game_report(
        report_path=report,
        candidate=candidate,
        champion=champion,
    )

    assert value["games"][0]["final_public_vps"]["RED"] == 8
    assert value["games"][0]["final_actual_vps"]["RED"] == 10


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
    games = [
        _bucket_game(
            index,
            "candidate_red",
            candidate_won=index < 6,
            phase="BUILD_INITIAL_SETTLEMENT",
        )
        for index in range(8)
    ] + [
        _bucket_game(
            100 + index,
            "candidate_blue",
            candidate_won=index < 9,
            max_legal_count=54,
        )
        for index in range(10)
    ]
    source_report = candidate.parent / "bucket-source-high-regret.json"
    _json(
        source_report,
        {
            "schema_version": artifacts.HIGH_REGRET_REPORT_SCHEMA,
            "suite": "held_out_high_regret",
            "held_out": True,
            "suite_manifest": {"path": "unused", "sha256": "sha256:" + "0" * 64},
            "candidate": _ref(candidate),
            "champion": _ref(champion),
            "evaluation_config": {},
            "errors": [],
            "games": copy.deepcopy(games),
            "pentanomial_sprt": {},
            "pair_diagnostics": {},
            "planned_engine_identity": {},
            "engine_identity": {},
            "archived_state_reconstruction": {},
            "search_rng_contract": H2H_SEARCH_RNG_CONTRACT,
        },
    )
    return {
        "schema_version": artifacts.BUCKET_GAME_REPORT_SCHEMA,
        "candidate": _ref(candidate),
        "champion": _ref(champion),
        "errors": [],
        "source_report": _ref(source_report),
        "search_rng_contract": H2H_SEARCH_RNG_CONTRACT,
        "games": games,
    }


def _sync_bucket_source(raw: dict) -> None:
    source_path = Path(raw["source_report"]["path"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["games"] = copy.deepcopy(raw["games"])
    _json(source_path, source)
    raw["source_report"] = _ref(source_path)


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


def test_prompt_regression_cannot_hide_in_pooled_opening_bucket(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _bucket_report(candidate, champion)
    raw["games"] = [
        _bucket_game(pair, orientation, candidate_won=False,
                     phase="BUILD_INITIAL_SETTLEMENT")
        for pair in range(4)
        for orientation in ("candidate_red", "candidate_blue")
    ] + [
        _bucket_game(100 + pair, orientation, candidate_won=True,
                     phase="BUILD_INITIAL_ROAD")
        for pair in range(4)
        for orientation in ("candidate_red", "candidate_blue")
    ]
    _sync_bucket_source(raw)
    report = tmp_path / "bucket-games.json"
    _json(report, raw)

    value = artifacts.build_bucket_veto_source(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["per_bucket"]["opening"] == {
        "status": "pass", "n": 16, "winrate": 0.5
    }
    assert value["per_bucket"]["phase:initial_settlement"] == {
        "status": "fail", "n": 8, "winrate": 0.0
    }
    assert value["per_bucket"]["phase:initial_road"]["status"] == "pass"
    assert "phase:initial_settlement" in value["veto_buckets"]


def test_bucket_builder_preserves_fail_and_insufficient_data(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _bucket_report(candidate, champion)
    raw["games"] = [
        _bucket_game(
            index,
            "candidate_red",
            candidate_won=index < 2,
            phase="BUILD_INITIAL_SETTLEMENT",
            blowout=True,
        )
        for index in range(8)
    ] + [
        _bucket_game(
            9,
            "candidate_red",
            candidate_won=True,
            phase="BUILD_ROAD",
        )
    ]
    _sync_bucket_source(raw)
    report = tmp_path / "bucket-games.json"
    _json(report, raw)

    value = artifacts.build_bucket_veto_source(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["veto"] is True
    assert value["per_bucket"]["opening"]["status"] == "fail"
    assert value["per_bucket"]["close"]["status"] == "insufficient_data"
    assert "opening" in value["veto_buckets"]


def test_bucket_builder_uses_fixed_five_percent_regression_limit(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _bucket_report(candidate, champion)
    raw["games"] = [
        _bucket_game(
            index,
            "candidate_red",
            candidate_won=(index - offset) < wins,
            phase=phase,
            blowout=True,
        )
        for phase, wins, offset in (
            ("BUILD_INITIAL_SETTLEMENT", 9, 0),
            ("BUILD_ROAD", 8, 100),
        )
        for index in range(offset, offset + 20)
    ]
    _sync_bucket_source(raw)
    report = tmp_path / "bucket-games.json"
    _json(report, raw)

    value = artifacts.build_bucket_veto_source(
        report_path=report, candidate=candidate, champion=champion
    )

    assert value["per_bucket"]["phase:initial_settlement"]["status"] == "pass"
    assert value["per_bucket"]["phase:build_trade"]["status"] == "fail"
    assert "phase:build_trade" in value["veto_buckets"]


def test_bucket_builder_refuses_duplicate_games(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _bucket_report(candidate, champion)
    raw["games"].append(dict(raw["games"][0]))
    _sync_bucket_source(raw)
    report = tmp_path / "bucket-games.json"
    _json(report, raw)
    with pytest.raises(artifacts.ArtifactBuildError, match="duplicate games"):
        artifacts.build_bucket_veto_source(
            report_path=report, candidate=candidate, champion=champion
        )


def test_bucket_builder_rejects_legacy_scoreless_report(tmp_path: Path) -> None:
    candidate, champion = _checkpoints(tmp_path)
    raw = _bucket_report(candidate, champion)
    raw["schema_version"] = "a1-bucket-game-report-v1"
    for game in raw["games"]:
        for field in (
            "candidate_color",
            "baseline_color",
            "winner",
            "terminated",
            "truncated",
            "final_public_vps",
            "final_actual_vps",
            "archived_phase",
            "phases_seen",
            "max_legal_count",
        ):
            game.pop(field)
    report = tmp_path / "legacy-bucket-games.json"
    _json(report, raw)

    with pytest.raises(artifacts.ArtifactBuildError, match="schema must be"):
        artifacts.build_bucket_veto_source(
            report_path=report,
            candidate=candidate,
            champion=champion,
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
            {
                "value_readout": "scalar",
                "max_rmse_regression": 0.02,
                "max_slice_rmse_regression": 0.02,
                "minimum_slice_rows": 30,
                "required_slices_if_present": list(
                    promotion.REQUIRED_CALIBRATION_SLICES_IF_PRESENT
                ),
            },
        ),
        (
            "internal_h2h",
            "H1",
            dict(promotion.INTERNAL_STRENGTH_RESULT),
        ),
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
        promotion, "_candidate_search_config", lambda _contract: {"c_scale": 0.10}
    )
    monkeypatch.setattr(
        promotion,
        "_incumbent_search_config",
        lambda _contract, **_kwargs: {"c_scale": 0.03},
    )
    registry = ChampionRegistry(tmp_path / "registry.json")
    artifacts._validate_envelope_before_write(
        tmp_path / "final.json",
        value=value,
        kind="high_regret",
        contract=_contract(),
        candidate=candidate,
        champion=champion,
        registry=registry,
    )
    assert len(called) == 1
    assert not called[0].exists()


def test_branch_evidence_builder_requires_two_strict_internal_cohorts(
    tmp_path: Path,
) -> None:
    candidate, champion = _checkpoints(tmp_path)
    cohort_1 = tmp_path / "cohort-1.json"
    cohort_2 = tmp_path / "cohort-2.json"
    _json(cohort_1, {"cohort": 1})
    _json(cohort_2, {"cohort": 2})

    value = artifacts.build_evidence_envelope(
        kind="internal_h2h",
        contract=_contract(),
        candidate=candidate,
        champion=champion,
        sources=[
            ("internal_h2h_cohort_1", cohort_1),
            ("internal_h2h_cohort_2", cohort_2),
        ],
        promotion_mode="branch_challenge",
    )

    assert value["verdict"] == "H1"
    assert value["result"] == {
        **promotion.INTERNAL_STRENGTH_RESULT,
        "required_fresh_cohorts": 2,
        "strict_superiority": True,
    }
    with pytest.raises(artifacts.ArtifactBuildError, match="two fresh"):
        artifacts.build_evidence_envelope(
            kind="internal_h2h",
            contract=_contract(),
            candidate=candidate,
            champion=champion,
            sources=[("internal_h2h", cohort_1)],
            promotion_mode="branch_challenge",
        )


def test_adjudication_builder_derives_every_third_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        promotion, "_candidate_search_config", lambda _contract: {"c_scale": 0.10}
    )
    monkeypatch.setattr(
        promotion,
        "_incumbent_search_config",
        lambda _contract, **_kwargs: {"c_scale": 0.03},
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
    nth_confirmation = tmp_path / "n64-confirmation.json"
    _json(nth_confirmation, {"verdict": "H1"})

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
        nth_confirmation=nth_confirmation,
    )

    assert value["nth_confirmation_required"] is True
    assert value["nth_confirmation"] == {
        "path": str(nth_confirmation.resolve()),
        "sha256": promotion._sha256(nth_confirmation),
    }
    assert value["candidate"]["agent_identity"]["search_config"]["c_scale"] == 0.10
    assert value["champion"]["agent_identity"]["search_config"]["c_scale"] == 0.03
    digest = value.pop("adjudication_sha256")
    assert digest == promotion._digest_value(value)


def test_branch_adjudication_builder_records_initializer_and_displaced_incumbent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        promotion, "_candidate_search_config", lambda _contract: {"c_scale": 0.10}
    )
    monkeypatch.setattr(
        promotion,
        "_incumbent_search_config",
        lambda _contract, **_kwargs: {"c_scale": 0.10},
    )
    candidate, champion = _checkpoints(tmp_path)
    parent = tmp_path / "f7-parent.pt"
    parent.write_bytes(b"older initializer")
    report = tmp_path / "training.json"
    _json(
        report,
        {
            "init_checkpoint": str(parent),
            "init_checkpoint_sha256": promotion._sha256(parent),
        },
    )
    receipt = tmp_path / "receipt.json"
    _json(receipt, {"receipt": True})
    pointer = tmp_path / "CURRENT_CHAMPION"
    pointer.write_text(str(champion.resolve()) + "\n", encoding="utf-8")
    evidence = []
    for kind in sorted(promotion.REQUIRED_EVIDENCE_KINDS):
        path = tmp_path / f"{kind}.json"
        _json(path, {"kind": kind})
        evidence.append((kind, path))

    value = artifacts.build_adjudication(
        contract=_contract(),
        contract_lock=tmp_path / "contract.json",
        training_receipt=receipt,
        registry=ChampionRegistry(tmp_path / "registry.json"),
        current_pointer=pointer,
        candidate=candidate,
        candidate_version=6,
        training_report=report,
        champion=champion,
        champion_version=5,
        evidence=evidence,
        nth_confirmation=None,
        promotion_mode="branch_challenge",
    )

    assert value["schema_version"] == promotion.BRANCH_CHALLENGE_ADJUDICATION_SCHEMA
    assert value["promotion_mode"] == "branch_challenge"
    assert value["candidate_lineage"]["initializer"] == artifacts._checkpoint_ref(
        parent
    )
    assert value["candidate_lineage"]["displaced_incumbent"]["sha256"] == (
        promotion._sha256(champion)
    )


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
            nth_confirmation=None,
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
