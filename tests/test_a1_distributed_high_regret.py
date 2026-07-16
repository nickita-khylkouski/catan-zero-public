from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from tools import a1_distributed_high_regret as distributed
from tools import a1_promotion_artifacts as artifacts
from tools import a1_promotion_transaction as promotion
from tools.regret_common import (
    H2H_SEARCH_RNG_CONTRACT,
    h2h_search_seed,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fake_suite(tmp_path: Path, *, pairs: int = 40) -> Path:
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    states = [
        {
            "pair_id": pair,
            "shard_path": str(source),
            "shard_id": 0,
            "row_index": pair,
            "game_seed": 90_000 + pair,
            "decision_index": pair % 3,
            "phase": (
                "BUILD_INITIAL_SETTLEMENT",
                "MOVE_ROBBER",
                "ROLL",
                "BUILD_ROAD",
            )[pair % 4],
            "legal_count": 54,
            "regret_score": float(pair),
            "replay_source": {
                "contract": "fixture",
                "scope": str(tmp_path),
                "scope_inventory_sha256": "sha256:" + "a" * 64,
                "scope_shard_count": 1,
            },
        }
        for pair in range(pairs)
    ]
    value = {
        "schema_version": artifacts.HIGH_REGRET_SUITE_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "source_manifest": {
            "path": str(source),
            "sha256": promotion._sha256(source),  # noqa: SLF001
        },
        "validation_seed_manifest": {
            "path": str(source),
            "sha256": promotion._sha256(source),  # noqa: SLF001
            "schema_version": "train-validation-game-seeds-v1",
            "game_seed_count": pairs,
            "game_seed_set_sha256": "sha256:" + "b" * 64,
        },
        "selection": {
            "algorithm": "stable-hash-holdout-stratified-regret-v1",
            "holdout_fraction": 0.1,
            "holdout_seed": 17,
            "eligible_unique_states": pairs,
            "selected_pairs": pairs,
            "stratum_min_pairs": 4,
            "selected_by_stratum": {
                "phase:opening": 4,
                "phase:robber_dev": 4,
                "phase:chance": 4,
                "phase:build_trade": 4,
                "41+": 4,
            },
            "replay_preflight": {
                "candidate_states": pairs,
                "contract": "fixture",
                "rejected_bad_source": 0,
                "rejected_noncontiguous": 0,
                "replay_complete_states": pairs,
            },
        },
        "states": states,
    }
    value["suite_sha256"] = promotion._digest_value(value)  # noqa: SLF001
    path = tmp_path / "original.suite.json"
    _write(path, value)
    return path


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch):
    def load(path: Path):
        path = Path(path).resolve(strict=True)
        value = json.loads(path.read_text())
        unhashed = dict(value)
        digest = unhashed.pop("suite_sha256")
        if digest != promotion._digest_value(unhashed):  # noqa: SLF001
            raise ValueError("suite semantic digest mismatch")
        return path, value, [
            {
                "pair_id": state["pair_id"],
                "game_seed": state["game_seed"],
                "archived_state": state,
            }
            for state in value["states"]
        ]

    monkeypatch.setattr(distributed, "_load_held_out_high_regret_suite", load)
    return load


def _report(fragment: Path, candidate: Path, champion: Path) -> dict:
    suite = json.loads(fragment.read_text())
    games = []
    for state in suite["states"]:
        for orientation in ("candidate_red", "candidate_blue"):
            candidate_color, baseline_color = (
                ("RED", "BLUE")
                if orientation == "candidate_red"
                else ("BLUE", "RED")
            )
            game_seed = state["game_seed"]
            games.append(
                {
                    "pair_id": state["pair_id"],
                    "game_seed": game_seed,
                    "orientation": orientation,
                    "search_seeds_by_role": {
                        "candidate": h2h_search_seed(
                            game_seed=game_seed,
                            seat_color=candidate_color,
                        ),
                        "baseline": h2h_search_seed(
                            game_seed=game_seed,
                            seat_color=baseline_color,
                        ),
                    },
                    "candidate_color": candidate_color,
                    "baseline_color": baseline_color,
                    "candidate_won": True,
                    "truncated": False,
                    "archived_game_seed": state["game_seed"],
                    "archived_decision_index": state["decision_index"],
                    "buckets": ["phase:test"],
                }
            )
    normalized = [{**game, "search_won": True} for game in games]
    scores, diagnostics = promotion.pair_scores_from_h2h_games(normalized)
    pentanomial = promotion.evaluate_pentanomial_sprt(
        scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    return {
        "schema_version": artifacts.HIGH_REGRET_REPORT_SCHEMA,
        "suite": "held_out_high_regret",
        "held_out": True,
        "suite_manifest": {
            "path": str(fragment.resolve()),
            "sha256": promotion._sha256(fragment),  # noqa: SLF001
        },
        "candidate": {
            "path": str(candidate.resolve()),
            "sha256": promotion._sha256(candidate),  # noqa: SLF001
        },
        "champion": {
            "path": str(champion.resolve()),
            "sha256": promotion._sha256(champion),  # noqa: SLF001
        },
        "evaluation_config": {
            "pairs": len(suite["states"]),
            "c_scale": 0.1,
            "candidate_c_scale": 0.1,
            "baseline_c_scale": 0.03,
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
            "replay_contract": promotion.REPLAY_CONTRACT,
        },
        "errors": [],
        "search_rng_contract": H2H_SEARCH_RNG_CONTRACT,
        "games": games,
        "pentanomial_sprt": pentanomial,
        "pair_diagnostics": diagnostics,
    }


def _campaign(tmp_path: Path, fake_loader):
    suite = _fake_suite(tmp_path)
    partition = tmp_path / "partition"
    manifest = distributed.shard_suite(
        suite_path=suite, shards=2, out_dir=partition
    )
    candidate, champion = tmp_path / "candidate.pt", tmp_path / "champion.pt"
    candidate.write_bytes(b"candidate")
    champion.write_bytes(b"champion")
    reports = []
    for record in manifest["fragments"]:
        fragment = Path(record["suite"]["path"])
        path = tmp_path / f"report-{record['index']}.json"
        _write(path, _report(fragment, candidate, champion))
        reports.append(path)
    return suite, partition / "partition.manifest.json", candidate, champion, reports


def test_shard_is_deterministic_retryable_and_exact(tmp_path: Path, fake_loader) -> None:
    suite = _fake_suite(tmp_path)
    out = tmp_path / "partition"
    first = distributed.shard_suite(suite_path=suite, shards=2, out_dir=out)
    second = distributed.shard_suite(suite_path=suite, shards=2, out_dir=out)
    assert first == second
    pair_ids = [pair for row in first["fragments"] for pair in row["pair_ids"]]
    assert sorted(pair_ids) == list(range(40))
    assert len(pair_ids) == len(set(pair_ids))
    for record in first["fragments"]:
        fragment = json.loads(Path(record["suite"]["path"]).read_text())
        assert fragment["selection"]["stratum_min_pairs"] == 4
        assert all(
            value == 4
            for value in fragment["selection"]["selected_by_stratum"].values()
        )


def test_merge_recomputes_original_suite_report(tmp_path: Path, fake_loader) -> None:
    suite, manifest, candidate, champion, reports = _campaign(tmp_path, fake_loader)
    out = tmp_path / "merged.json"
    first = distributed.merge_reports(
        manifest_path=manifest,
        reports=reports,
        candidate=candidate,
        champion=champion,
        out=out,
    )
    second = distributed.merge_reports(
        manifest_path=manifest,
        reports=list(reversed(reports)),
        candidate=candidate,
        champion=champion,
        out=out,
    )
    assert first == second
    assert first["suite_manifest"]["path"] == str(suite.resolve())
    assert len(first["games"]) == 80
    assert first["pair_diagnostics"]["ww_pairs"] == 40
    assert first["pentanomial_sprt"]["decision"] == "H1"


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("checkpoint", "checkpoint drift"),
        ("config", "config drift"),
        ("orientation", "orientation"),
        ("error", "evaluation errors"),
        ("truncated", "truncated"),
        ("archived", "archived-state identity drift"),
        ("search_rng", "role/seat binding"),
        ("statistics", "statistics do not replay"),
    ],
)
def test_merge_rejects_report_tampering(
    tmp_path: Path, fake_loader, tamper: str, match: str
) -> None:
    _suite, manifest, candidate, champion, reports = _campaign(tmp_path, fake_loader)
    value = json.loads(reports[0].read_text())
    if tamper == "checkpoint":
        value["candidate"]["sha256"] = "sha256:" + "0" * 64
    elif tamper == "config":
        value["evaluation_config"]["candidate_c_scale"] = 0.2
    elif tamper == "orientation":
        value["games"][0]["orientation"] = "bogus"
    elif tamper == "error":
        value["errors"] = ["boom"]
    elif tamper == "truncated":
        value["games"][0]["truncated"] = True
        value["games"][0]["candidate_won"] = None
    elif tamper == "archived":
        value["games"][0]["archived_game_seed"] += 1
    elif tamper == "search_rng":
        value["games"][0]["search_seeds_by_role"]["candidate"] += 1
    else:
        value["pair_diagnostics"]["ww_pairs"] -= 1
    _write(reports[0], value)
    with pytest.raises(distributed.DistributedHighRegretError, match=match):
        distributed.merge_reports(
            manifest_path=manifest,
            reports=reports,
            candidate=candidate,
            champion=champion,
            out=tmp_path / "merged.json",
        )


def test_merge_rejects_missing_duplicate_and_fragment_drift(
    tmp_path: Path, fake_loader
) -> None:
    _suite, manifest, candidate, champion, reports = _campaign(tmp_path, fake_loader)
    with pytest.raises(distributed.DistributedHighRegretError, match="exactly one"):
        distributed.merge_reports(
            manifest_path=manifest,
            reports=reports[:1],
            candidate=candidate,
            champion=champion,
            out=tmp_path / "missing.json",
        )
    with pytest.raises(distributed.DistributedHighRegretError, match="same fragment"):
        distributed.merge_reports(
            manifest_path=manifest,
            reports=[reports[0], reports[0]],
            candidate=candidate,
            champion=champion,
            out=tmp_path / "duplicate.json",
        )
    partition = json.loads(manifest.read_text())
    fragment = Path(partition["fragments"][0]["suite"]["path"])
    fragment.chmod(0o644)
    value = json.loads(fragment.read_text())
    value["states"][0]["game_seed"] += 1
    _write(fragment, value)
    with pytest.raises(distributed.DistributedHighRegretError, match="file hash drift"):
        distributed.merge_reports(
            manifest_path=manifest,
            reports=reports,
            candidate=candidate,
            champion=champion,
            out=tmp_path / "drift.json",
        )


def test_atomic_output_refuses_different_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    distributed._publish_exact(path, {"a": 1})  # noqa: SLF001
    distributed._publish_exact(path, {"a": 1})  # noqa: SLF001
    with pytest.raises(distributed.DistributedHighRegretError, match="differs"):
        distributed._publish_exact(path, {"a": 2})  # noqa: SLF001


def test_real_evaluator_loader_accepts_every_fragment(tmp_path: Path) -> None:
    shard_dir = tmp_path / "producer"
    shard_dir.mkdir()
    shard = shard_dir / "rows.npz"
    seeds = np.arange(100_000, 101_600, dtype=np.int64)
    np.savez(
        shard,
        game_seed=seeds,
        decision_index=np.zeros(len(seeds), dtype=np.int32),
        action_taken=np.arange(len(seeds), dtype=np.int32),
    )
    validation = tmp_path / "validation-seeds.json"
    seed_digest = "sha256:" + hashlib.sha256(
        np.sort(seeds).astype("<i8", copy=False).tobytes()
    ).hexdigest()
    _write(
        validation,
        {
            "schema_version": "train-validation-game-seeds-v1",
            "game_seeds": seeds.tolist(),
            "validation_game_seed_count": len(seeds),
            "validation_game_seed_set_sha256": seed_digest,
        },
    )
    source = tmp_path / "regret.npz"
    np.savez(
        source,
        held_out_only=np.asarray(True),
        validation_seed_manifest_path=np.asarray(str(validation.resolve())),
        validation_seed_manifest_sha256=np.asarray(
            promotion._sha256(validation)  # noqa: SLF001
        ),
        validation_seed_manifest_schema_version=np.asarray(
            "train-validation-game-seeds-v1"
        ),
        validation_game_seed_count=np.asarray(len(seeds), dtype=np.int64),
        validation_game_seed_set_sha256=np.asarray(seed_digest),
        shard_id=np.zeros(len(seeds), dtype=np.int32),
        row_index=np.arange(len(seeds), dtype=np.int32),
        game_seed=seeds,
        decision_index=np.zeros(len(seeds), dtype=np.int32),
        regret_score=np.linspace(0.0, 1.0, len(seeds), dtype=np.float32),
        phase=np.asarray(
            [
                (
                    "BUILD_INITIAL_SETTLEMENT",
                    "MOVE_ROBBER",
                    "ROLL",
                    "BUILD_ROAD",
                )[index % 4]
                for index in range(len(seeds))
            ]
        ),
        legal_count=np.full(len(seeds), 54, dtype=np.int32),
        shard_paths=np.asarray([str(shard)]),
    )
    suite_value = artifacts.build_held_out_high_regret_suite(
        manifest_path=source,
        holdout_fraction=1.0,
        holdout_seed=17,
        pairs=80,
    )
    suite = tmp_path / "original.json"
    _write(suite, suite_value)

    manifest = distributed.shard_suite(
        suite_path=suite, shards=2, out_dir=tmp_path / "partition"
    )

    assert len(manifest["fragments"]) == 2
    for record in manifest["fragments"]:
        path, value, pairs = distributed._load_held_out_high_regret_suite(  # noqa: SLF001
            Path(record["suite"]["path"])
        )
        assert path.is_file()
        assert len(pairs) == len(value["states"])
