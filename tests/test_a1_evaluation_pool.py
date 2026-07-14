from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import a1_evaluation_pool as pool
from tools import a1_promotion_transaction as promotion
from tools.sprt_gate import evaluate_pentanomial_sprt, pair_scores_from_h2h_games


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _checkpoint(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(name.encode())
    return path


def _games(seed: int, *, orientations: tuple[str, str], won: bool = True) -> list[dict]:
    games = []
    for orientation in orientations:
        game = {
            "pair_id": 0,
            "game_seed": seed,
            "orientation": orientation,
            "candidate_won": won,
            "search_won": won,
            "winner": "RED",
            "terminated": True,
            "truncated": False,
            "error": None,
            "engine_divergence": False,
            "decisions": 100,
        }
        if orientation in {"candidate_red", "candidate_blue"}:
            candidate_color = "RED" if orientation == "candidate_red" else "BLUE"
            baseline_color = "BLUE" if candidate_color == "RED" else "RED"
            game["search_seeds_by_role"] = {
                "candidate": promotion._internal_h2h_search_seed(  # noqa: SLF001
                    game_seed=seed, seat_color=candidate_color
                ),
                "baseline": promotion._internal_h2h_search_seed(  # noqa: SLF001
                    game_seed=seed, seat_color=baseline_color
                ),
            }
        games.append(game)
    return games


def _gate(games: list[dict]) -> tuple[dict, dict]:
    scores, diagnostics = pair_scores_from_h2h_games(games)
    return (
        evaluate_pentanomial_sprt(scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05),
        diagnostics,
    )


def _internal_report(candidate: Path, champion: Path, seed: int) -> dict:
    games = _games(seed, orientations=("candidate_red", "candidate_blue"))
    pentanomial, diagnostics = _gate(games)
    typed = {
        "pipeline": "eval",
        "schema_version": 6,
        "fields": {
            "mode": "cross_net",
            "candidate": str(candidate.resolve()),
            "baseline": str(champion.resolve()),
            "base_seed": seed,
            "pairs": 1,
        },
    }
    digest = hashlib.sha256(
        json.dumps(typed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    planned_engine_identity = promotion._canonical_internal_h2h_engine_identity()  # noqa: SLF001
    return {
        "planned_engine_identity": planned_engine_identity,
        "engine_identity": dict(planned_engine_identity),
        "candidate_checkpoint": str(candidate.resolve()),
        "candidate_checkpoint_sha256": promotion._sha256(candidate),
        "baseline_checkpoint": str(champion.resolve()),
        "baseline_checkpoint_sha256": promotion._sha256(champion),
        "gate_config": "flywheel",
        "search_rng_contract": promotion.INTERNAL_H2H_SEARCH_RNG_CONTRACT,
        "n_full": 128,
        "config_hash": "sha256:" + digest[:16],
        "full_config_hash": "sha256:" + digest,
        "typed_config": typed,
        "pairs_requested": 1,
        "base_seed": seed,
        "games_played": 2,
        "games_with_winner": 2,
        "games_truncated": 0,
        "candidate_wins": 2,
        "baseline_wins": 0,
        "candidate_win_rate": 1.0,
        "sprt": {},
        "pair_sprt": {},
        "pentanomial_sprt": pentanomial,
        "verdict": pentanomial["decision"],
        "pair_diagnostics": diagnostics,
        "pairs_decisive": 1,
        "pairs_split_excluded": 0,
        "pairs_truncated_excluded": 0,
        "complete_pairs": 1,
        "split_rate": 0.0,
        "decisive_pair_yield": 1.0,
        "elapsed_sec": 10.0,
        "workers": 1,
        "threads_per_worker": 1,
        "search_telemetry": {
            "by_role": {
                role: {
                    "search_calls": 10,
                    "non_forced_search_calls": 8,
                    "search_elapsed_sec": 2.0,
                    "simulations_used": 1_280,
                    "wide_root_calls": 2,
                    "wide_root_simulations_used": 256,
                    "selected_vs_prior_disagreement_calls": 3,
                    "wide_selected_vs_prior_disagreement_calls": 1,
                    # Derived fields must be ignored and recomputed by pooling.
                    "search_seconds_per_call": 0.2,
                    "simulations_per_call": 128.0,
                    "wide_root_simulations_per_call": 128.0,
                    "selected_vs_prior_disagreement_rate": 0.375,
                    "wide_selected_vs_prior_disagreement_rate": 0.5,
                }
                for role in ("candidate", "baseline")
            }
        },
        "errors": [],
        "games": games,
    }


def _neutral_report(checkpoint: Path, seed: int) -> dict:
    games = _games(seed, orientations=("candidate_first", "candidate_second"))
    for game in games:
        game.update(
            illegal_policy_picks=0,
            search_decisions=80,
            simulations_used=10_240,
        )
    pentanomial, diagnostics = _gate(games)
    return {
        "stratum": "neutral-harness",
        "harness": "catanatron_native_engine",
        "referee_engine": "vendored_python_catanatron",
        "candidate_checkpoint": str(checkpoint.resolve()),
        "candidate_checkpoint_md5": promotion._md5(checkpoint),
        "candidate_checkpoint_sha256": promotion._sha256(checkpoint),
        "baseline_bot": "catanatron_value",
        "mode": "search",
        "map_kind": "TOURNAMENT",
        "n_full": 128,
        "search_config": {"n_full": 128, "public_observation": True},
        "gate_config": "flywheel",
        "pairs_requested": 1,
        "base_seed": seed,
        "complete_pairs": 1,
        "games_requested": 2,
        "games_played": 2,
        "games_with_winner": 2,
        "games_truncated": 0,
        "games_errored": 0,
        "games_engine_divergence": 0,
        "candidate_wins": 2,
        "baseline_wins": 0,
        "candidate_win_rate": 1.0,
        "candidate_win_rate_wilson_95ci": [0.0, 1.0],
        "total_illegal_policy_picks": 0,
        "total_search_decisions": 160,
        "total_simulations_used": 20_480,
        "sprt": {},
        "pentanomial_sprt": pentanomial,
        "verdict": pentanomial["decision"],
        "pair_diagnostics": diagnostics,
        "workers": 1,
        "threads_per_worker": 1,
        "run_fingerprint": f"run-{seed}",
        "artifact_dir": f"/artifacts/{seed}",
        "resume": {
            "enabled": False,
            "games_resumed": 0,
            "games_run_this_invocation": 2,
        },
        "elapsed_sec": 10.0,
        "worker_errors": [],
        "errors": [],
        "games": games,
    }


@pytest.mark.parametrize("kind", ["internal", "neutral"])
def test_complete_report_contract_rejects_zero_game_exit_zero_payload(
    tmp_path: Path, kind: str
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    report = (
        _internal_report(candidate, champion, 9001)
        if kind == "internal"
        else _neutral_report(candidate, 9001)
    )
    report.update(
        games=[],
        games_played=0,
        games_with_winner=0,
        complete_pairs=0,
        candidate_wins=0,
        baseline_wins=0,
    )
    if kind == "neutral":
        report["games_requested"] = 0

    with pytest.raises(pool.PoolError, match="raw games do not exactly cover"):
        pool.validate_complete_report(
            report, kind=kind, expected_pairs=1, where="exit-zero-report"
        )


@pytest.mark.parametrize("kind", ["internal", "neutral"])
@pytest.mark.parametrize("error_field", ["errors", "worker_errors", "pair_errors"])
def test_complete_report_contract_rejects_every_error_channel(
    tmp_path: Path, kind: str, error_field: str
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    report = (
        _internal_report(candidate, champion, 9001)
        if kind == "internal"
        else _neutral_report(candidate, 9001)
    )
    report[error_field] = [{"error": "worker failed before first game"}]

    with pytest.raises(pool.PoolError, match=f"{error_field} must be an empty list"):
        pool.validate_complete_report(
            report, kind=kind, expected_pairs=1, where="failed-report"
        )


def test_complete_report_contract_binds_lane_to_planned_pair_count(
    tmp_path: Path,
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    report = _internal_report(candidate, champion, 9001)

    with pytest.raises(pool.PoolError, match="differs from planned 2"):
        pool.validate_complete_report(
            report, kind="internal", expected_pairs=2, where="short-lane"
        )


def test_internal_pool_requires_corrected_search_rng_contract(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    report = _internal_report(candidate, champion, 9001)
    report.pop("search_rng_contract")

    with pytest.raises(pool.PoolError, match="corrected per-game/seat"):
        pool.validate_complete_report(report, kind="internal", where="old-rng")


def test_internal_pool_replays_role_to_seat_search_seed(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    report = _internal_report(candidate, champion, 9001)
    report["games"][0]["search_seeds_by_role"]["candidate"] += 1

    with pytest.raises(pool.PoolError, match="role/seat binding"):
        pool.validate_complete_report(report, kind="internal", where="drifted-rng")


def test_internal_pool_reindexes_local_pair_ids_and_recomputes_gate(
    tmp_path: Path,
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    paths = []
    for index, seed in enumerate((9002, 9001)):
        path = tmp_path / f"internal-{index}.json"
        _write(path, _internal_report(candidate, champion, seed))
        paths.append(path)

    result = pool.pool_internal(paths, candidate=candidate, champion=champion)

    assert result["complete_pairs"] == 2
    assert result["games_played"] == 4
    assert [(game["game_seed"], game["pair_id"]) for game in result["games"]] == [
        (9001, 0),
        (9001, 0),
        (9002, 1),
        (9002, 1),
    ]
    assert all("source_pair_id" in game for game in result["games"])
    scores, diagnostics = pair_scores_from_h2h_games(result["games"])
    assert result["pair_diagnostics"] == diagnostics
    assert result["pentanomial_sprt"] == evaluate_pentanomial_sprt(
        scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    assert result["superiority_pentanomial_sprt"] == evaluate_pentanomial_sprt(
        scores, elo0=0.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    assert result["superiority_verdict"] == result[
        "superiority_pentanomial_sprt"
    ]["decision"]
    assert result["gate_interpretation"] == {
        "schema_version": "a1-gate-interpretation-v1",
        "promotion_gate_semantics": "regression_protection",
        "promotion_elo0": -10.0,
        "promotion_elo1": 15.0,
        "h1_proves_positive_elo": False,
        "superiority_elo0": 0.0,
        "superiority_elo1": 15.0,
    }
    assert len(result["fleet_merge"]["sources"]) == 2
    assert "typed_config" not in result
    assert "config_hash" not in result
    assert result["effective_search_config"]["mode"] == "cross_net"
    assert result["search_telemetry"]["by_role"]["candidate"]["search_calls"] == 20
    assert (
        result["search_telemetry"]["by_role"]["candidate"][
            "selected_vs_prior_disagreement_rate"
        ]
        == 0.375
    )
    assert [row["base_seed"] for row in result["fleet_merge"]["seed_intervals"]] == [
        9001,
        9002,
    ]


@pytest.mark.parametrize("kind", ["internal", "neutral"])
def test_pool_refuses_candidate_search_outcome_alias_drift(
    tmp_path: Path, kind: str
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    report = (
        _internal_report(candidate, champion, 9001)
        if kind == "internal"
        else _neutral_report(candidate, 9001)
    )
    report["games"][0]["candidate_won"] = False
    path = tmp_path / f"{kind}.json"
    _write(path, report)

    with pytest.raises(pool.PoolError, match="candidate_won/search_won alias drift"):
        if kind == "internal":
            pool.pool_internal([path], candidate=candidate, champion=champion)
        else:
            pool.pool_neutral([path], checkpoint=candidate)


def test_internal_pool_refuses_duplicate_seed_across_hosts(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    paths = []
    for index in range(2):
        path = tmp_path / f"internal-{index}.json"
        _write(path, _internal_report(candidate, champion, 9001))
        paths.append(path)
    with pytest.raises(pool.PoolError, match="seed intervals have an overlap"):
        pool.pool_internal(paths, candidate=candidate, champion=champion)


def test_internal_pool_allows_explicit_disjoint_fresh_cohorts(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    paths = []
    for index, seed in enumerate((9001, 9101)):
        path = tmp_path / f"internal-disjoint-{index}.json"
        _write(path, _internal_report(candidate, champion, seed))
        paths.append(path)

    with pytest.raises(pool.PoolError, match="seed intervals have a gap"):
        pool.pool_internal(paths, candidate=candidate, champion=champion)

    result = pool.pool_internal(
        paths,
        candidate=candidate,
        champion=champion,
        allow_disjoint_cohorts=True,
    )
    assert result["complete_pairs"] == 2
    assert result["fleet_merge"]["disjoint_cohorts"] is True
    assert [
        (row["base_seed"], row["end_seed"])
        for row in result["fleet_merge"]["seed_intervals"]
    ] == [(9001, 9002), (9101, 9102)]


def test_internal_pool_disjoint_mode_still_refuses_overlap(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    paths = []
    for index in range(2):
        path = tmp_path / f"internal-overlap-{index}.json"
        _write(path, _internal_report(candidate, champion, 9001))
        paths.append(path)
    with pytest.raises(pool.PoolError, match="seed intervals have an overlap"):
        pool.pool_internal(
            paths,
            candidate=candidate,
            champion=champion,
            allow_disjoint_cohorts=True,
        )


def test_internal_pool_refuses_config_drift(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, _internal_report(candidate, champion, 9001))
    drifted = _internal_report(candidate, champion, 9002)
    drifted["typed_config"]["fields"]["n_full"] = 256
    digest = hashlib.sha256(
        json.dumps(
            drifted["typed_config"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    drifted["config_hash"] = "sha256:" + digest[:16]
    drifted["full_config_hash"] = "sha256:" + digest
    _write(second, drifted)
    with pytest.raises(pool.PoolError, match="science/config drift"):
        pool.pool_internal([first, second], candidate=candidate, champion=champion)


def test_internal_pool_refuses_native_runtime_identity_drift(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    first = tmp_path / "first-runtime.json"
    second = tmp_path / "second-runtime.json"
    _write(first, _internal_report(candidate, champion, 9001))
    drifted = _internal_report(candidate, champion, 9002)
    drifted["planned_engine_identity"]["native_runtime_sha256"] = "sha256:" + "0" * 64
    drifted["engine_identity"] = dict(drifted["planned_engine_identity"])
    _write(second, drifted)

    with pytest.raises(pool.PoolError, match="internal engine identity drift"):
        pool.pool_internal([first, second], candidate=candidate, champion=champion)


def test_internal_pool_refuses_role_value_squash_drift_with_valid_shard_hash(
    tmp_path: Path,
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, _internal_report(candidate, champion, 9001))
    drifted = _internal_report(candidate, champion, 9002)
    drifted["typed_config"]["fields"]["candidate_value_squash"] = "clip"
    digest = hashlib.sha256(
        json.dumps(
            drifted["typed_config"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    drifted["config_hash"] = "sha256:" + digest[:16]
    drifted["full_config_hash"] = "sha256:" + digest
    _write(second, drifted)

    with pytest.raises(pool.PoolError, match="science/config drift"):
        pool.pool_internal([first, second], candidate=candidate, champion=champion)


def test_internal_pool_refuses_forged_shard_statistics(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    path = tmp_path / "forged.json"
    report = _internal_report(candidate, champion, 9001)
    report["pentanomial_sprt"]["llr"] = 999.0
    _write(path, report)
    with pytest.raises(pool.PoolError, match="statistics do not replay"):
        pool.pool_internal([path], candidate=candidate, champion=champion)


def test_internal_pool_refuses_missing_or_invalid_search_telemetry(
    tmp_path: Path,
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    path = tmp_path / "missing-telemetry.json"
    report = _internal_report(candidate, champion, 9001)
    report.pop("search_telemetry")
    _write(path, report)
    with pytest.raises(pool.PoolError, match="search_telemetry.by_role"):
        pool.pool_internal([path], candidate=candidate, champion=champion)

    report = _internal_report(candidate, champion, 9001)
    report["search_telemetry"]["by_role"]["candidate"]["search_calls"] = -1
    _write(path, report)
    with pytest.raises(pool.PoolError, match="candidate.search_calls is invalid"):
        pool.pool_internal([path], candidate=candidate, champion=champion)


def test_neutral_pool_recomputes_stats_and_preserves_games(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "candidate.pt")
    paths = []
    for index, seed in enumerate((8102, 8101)):
        path = tmp_path / f"neutral-{index}.json"
        _write(path, _neutral_report(checkpoint, seed))
        paths.append(path)

    result = pool.pool_neutral(paths, checkpoint=checkpoint)

    assert result["complete_pairs"] == 2
    assert result["games_requested"] == result["games_played"] == 4
    assert result["candidate_win_rate"] == 1.0
    assert result["total_simulations_used"] == 40_960
    assert [game["game_seed"] for game in result["games"]] == [8101, 8101, 8102, 8102]
    assert result["fleet_merge"]["checkpoint"]["sha256"] == promotion._sha256(
        checkpoint
    )
    assert result["candidate_checkpoint"] == str(checkpoint.resolve())
    assert result["effective_search_config"] == result["search_config"]
    assert result["gate_interpretation"]["h1_proves_positive_elo"] is False
    scores, _ = pair_scores_from_h2h_games(result["games"])
    assert result["superiority_pentanomial_sprt"] == evaluate_pentanomial_sprt(
        scores, elo0=0.0, elo1=15.0, alpha=0.05, beta=0.05
    )


def test_neutral_pool_allows_explicit_disjoint_replication_cohorts(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, "candidate.pt")
    paths = []
    for index, seed in enumerate((8101, 9101)):
        path = tmp_path / f"neutral-disjoint-{index}.json"
        _write(path, _neutral_report(checkpoint, seed))
        paths.append(path)

    with pytest.raises(pool.PoolError, match="seed intervals have a gap"):
        pool.pool_neutral(paths, checkpoint=checkpoint)

    result = pool.pool_neutral(
        paths,
        checkpoint=checkpoint,
        allow_disjoint_cohorts=True,
    )
    assert result["complete_pairs"] == 2
    assert result["fleet_merge"]["disjoint_cohorts"] is True
    assert [
        (row["base_seed"], row["end_seed"])
        for row in result["fleet_merge"]["seed_intervals"]
    ] == [(8101, 8102), (9101, 9102)]


@pytest.mark.parametrize("kind", ["internal", "neutral"])
def test_pool_separates_wall_time_from_aggregate_lane_seconds(
    tmp_path: Path, kind: str
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    reports = []
    for index, (seed, elapsed) in enumerate(((8101, 10.0), (8102, 30.0))):
        report = (
            _internal_report(candidate, champion, seed)
            if kind == "internal"
            else _neutral_report(candidate, seed)
        )
        report["elapsed_sec"] = elapsed
        path = tmp_path / f"{kind}-{index}.json"
        _write(path, report)
        reports.append(path)

    result = (
        pool.pool_internal(reports, candidate=candidate, champion=champion)
        if kind == "internal"
        else pool.pool_neutral(reports, checkpoint=candidate)
    )

    assert result["elapsed_sec"] == 30.0
    assert result["aggregate_compute_sec"] == 40.0


def test_neutral_pool_refuses_checkpoint_hash_drift(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "candidate.pt")
    path = tmp_path / "neutral.json"
    report = _neutral_report(checkpoint, 8101)
    report["candidate_checkpoint_md5"] = "0" * 32
    _write(path, report)
    with pytest.raises(pool.PoolError, match="checkpoint MD5 drift"):
        pool.pool_neutral([path], checkpoint=checkpoint)


def test_internal_pool_rebases_remote_paths_only_after_hash_proof(
    tmp_path: Path,
) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    first = _internal_report(candidate, champion, 9001)
    second = _internal_report(candidate, champion, 9002)
    for report in (first, second):
        report["candidate_checkpoint"] = "/remote/h100/candidate.pt"
        report["baseline_checkpoint"] = "/remote/h100/champion.pt"
        report["typed_config"]["fields"]["candidate"] = "/remote/h100/candidate.pt"
        report["typed_config"]["fields"]["baseline"] = "/remote/h100/champion.pt"
        digest = hashlib.sha256(
            json.dumps(
                report["typed_config"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        report["config_hash"] = "sha256:" + digest[:16]
        report["full_config_hash"] = "sha256:" + digest
    paths = [tmp_path / "one.json", tmp_path / "two.json"]
    _write(paths[0], first)
    _write(paths[1], second)
    result = pool.pool_internal(paths, candidate=candidate, champion=champion)
    assert result["candidate_checkpoint"] == str(candidate.resolve())
    assert result["baseline_checkpoint"] == str(champion.resolve())


def test_internal_pool_refuses_seed_gap(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    paths = [tmp_path / "one.json", tmp_path / "two.json"]
    _write(paths[0], _internal_report(candidate, champion, 9001))
    _write(paths[1], _internal_report(candidate, champion, 9003))
    with pytest.raises(pool.PoolError, match="seed intervals have a gap"):
        pool.pool_internal(paths, candidate=candidate, champion=champion)


def test_internal_pool_refuses_checkpoint_sha256_drift(tmp_path: Path) -> None:
    candidate = _checkpoint(tmp_path, "candidate.pt")
    champion = _checkpoint(tmp_path, "champion.pt")
    path = tmp_path / "report.json"
    report = _internal_report(candidate, champion, 9001)
    report["candidate_checkpoint"] = "/remote/missing.pt"
    report["candidate_checkpoint_sha256"] = "sha256:" + "0" * 64
    _write(path, report)
    with pytest.raises(pool.PoolError, match="checkpoint SHA-256 drift"):
        pool.pool_internal([path], candidate=candidate, champion=champion)


def test_neutral_pool_refuses_incomplete_pair(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "candidate.pt")
    path = tmp_path / "neutral.json"
    report = _neutral_report(checkpoint, 8101)
    report["games"] = report["games"][:1]
    report["games_played"] = 1
    scores, diagnostics = pair_scores_from_h2h_games(report["games"])
    report["pentanomial_sprt"] = evaluate_pentanomial_sprt(
        scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    report["pair_diagnostics"] = diagnostics
    _write(path, report)
    with pytest.raises(pool.PoolError, match="raw games do not exactly cover"):
        pool.pool_neutral([path], checkpoint=checkpoint)
