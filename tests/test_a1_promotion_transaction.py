from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import a1_promotion_transaction as promotion
from tools.champion_registry import ChampionRegistry


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _contract(
    *, n_full: int = 128, n_full_wide=None, producer: Path | None = None
) -> dict:
    recipe = {
        "world_size": 1,
        "optimizer": "adam",
        "mask_hidden_info": True,
        "epochs": 1,
        "max_steps": 0,
    }
    producer = producer or Path("/producer.pt")
    return {
        "contract_sha256": "sha256:" + "a" * 64,
        "science": {
            "search_operator": {
                "n_full": n_full,
                "n_full_wide": n_full_wide,
                "wide_roots_always_full": n_full_wide is not None,
            },
            "learner_training_recipe": recipe,
            "learner_training_recipe_sha256": promotion._digest_value(recipe),
            "learner_value_objective": {"value_readout": "scalar"},
        },
        "checkpoints": [
            {
                "role": "producer",
                "path": str(producer),
                "sha256": promotion._sha256(producer) if producer.is_file() else "sha256:" + "f" * 64,
            }
        ],
    }


def _checkpoint_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": promotion._sha256(path)}


def _write_evidence_envelope(
    path: Path,
    *,
    kind: str,
    contract: dict,
    candidate: Path,
    champion: Path,
    sources: list[tuple[str, Path]],
    verdict: str,
    result: dict,
) -> None:
    payload = {
        "schema_version": promotion.EVIDENCE_SCHEMA,
        "kind": kind,
        "passed": True,
        "verdict": verdict,
        "contract_sha256": contract["contract_sha256"],
        "candidate": _checkpoint_ref(candidate),
        "champion": _checkpoint_ref(champion),
        "sources": [
            {"role": role, "path": str(source), "sha256": promotion._sha256(source)}
            for role, source in sources
        ],
        "result": result,
    }
    payload["evidence_sha256"] = promotion._digest_value(payload)
    _write_json(path, payload)


def _fixture(tmp_path: Path, *, promotion_count: int = 0, n_full: int = 128) -> dict:
    champion = tmp_path / "champion.pt"
    candidate = tmp_path / "candidate.pt"
    champion.write_bytes(b"incumbent checkpoint")
    candidate.write_bytes(b"candidate checkpoint")
    registry_path = tmp_path / "registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role(
        "generator_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        reason="fixture",
    )
    registry.set_role(
        "public_champion",
        champion,
        expected_md5=promotion._md5(champion),
        version=4,
        reason="fixture",
    )
    for _ in range(promotion_count):
        registry.record_promotion()
    registry.save()
    pointer = tmp_path / "CURRENT_CHAMPION"
    pointer.write_text(str(champion.resolve()) + "\n", encoding="utf-8")
    contract_path = tmp_path / "contract.lock.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    contract = _contract(n_full=n_full, producer=champion)
    report_path = tmp_path / "report.json"
    _write_json(
        report_path,
        {
            "a1_contract_sha256": contract["contract_sha256"],
            "a1_learner_training_recipe_sha256": contract["science"][
                "learner_training_recipe_sha256"
            ],
            "a1_bound_learner_training_recipe": contract["science"][
                "learner_training_recipe"
            ],
            "arch": "entity_graph",
            "mask_hidden_info": True,
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "steps_completed": 7,
            "epochs": 1,
            "max_steps": 0,
            "checkpoint": str(candidate),
            "init_checkpoint_sha256": contract["checkpoints"][0]["sha256"],
        },
    )
    calibration_sources = []
    for role, checkpoint, rmse in (
        ("candidate_calibration", candidate, 0.20),
        ("champion_calibration", champion, 0.21),
    ):
        source = tmp_path / f"{role}.json"
        _write_json(
            source,
            {
                "schema_version": "phase-sliced-value-calibration-v2",
                "checkpoint": str(checkpoint),
                "shard_dir": str(tmp_path / "shared_validation_corpus"),
                "value_readout": "scalar",
                "readout_provenance": {
                    "requested_readout": "scalar",
                    "trained_value_readouts": ["scalar"],
                    "optimizer_steps": 7,
                    "completed_epochs": 1,
                },
                "row_selection": {
                    "mode": "validation_seed_manifest",
                    "held_out_filter_applied": True,
                    "validation_fraction": 0.05,
                    "validation_seed": 17,
                    "validation_game_seed_ranges": [],
                    "seed_manifest_sha256": "sha256:" + "9" * 64,
                    "configured_game_seed_count": 256,
                    "observed_game_seed_count": 256,
                    "observed_row_count": 4096,
                },
                "global": {"n": 4096, "value_rmse": rmse},
            },
        )
        calibration_sources.append((role, source))
    internal_games = [
        {"pair_id": pair, "search_won": True, "candidate_won": True}
        for pair in range(200)
        for _orientation in range(2)
    ]
    pair_scores, pair_diagnostics = promotion.pair_scores_from_h2h_games(
        internal_games
    )
    pentanomial = promotion.evaluate_pentanomial_sprt(
        pair_scores, elo0=-10.0, elo1=15.0, alpha=0.05, beta=0.05
    )
    assert pentanomial["decision"] == "H1"
    typed_config = {
        "pipeline": "eval",
        "schema_version": 5,
        "fields": {
            "mode": "cross_net",
            "candidate": str(candidate),
            "baseline": str(champion),
            "public_observation": True,
            "candidate_n_full": 128,
            "baseline_n_full": 128,
            "n_full_wide": None,
            "candidate_n_full_wide": None,
            "baseline_n_full_wide": None,
            "n_full_wide_threshold": None,
            "candidate_n_full_wide_threshold": None,
            "baseline_n_full_wide_threshold": None,
        },
    }
    config_digest = hashlib.sha256(promotion._canonical_bytes(typed_config)).hexdigest()
    internal_source = tmp_path / "internal_h2h.raw.json"
    _write_json(
        internal_source,
        {
            "candidate_checkpoint": str(candidate),
            "baseline_checkpoint": str(champion),
            "typed_config": typed_config,
            "config_hash": "sha256:" + config_digest[:16],
            "full_config_hash": "sha256:" + config_digest,
            "candidate_value_readout": "scalar",
            "baseline_value_readout": "scalar",
            "public_observation": True,
            "search_budgets_by_role": {
                role: {
                    "n_full": 128,
                    "n_full_wide": None,
                    "n_full_wide_threshold": None,
                }
                for role in ("candidate", "baseline")
            },
            "complete_pairs": 200,
            "games_played": 400,
            "games_with_winner": 400,
            "games_truncated": 0,
            "errors": [],
            "games": internal_games,
            "pair_diagnostics": pair_diagnostics,
            "pentanomial_sprt": pentanomial,
            "verdict": "H1",
        },
    )

    external_sources = []
    external_games = [
        {
            "pair_id": pair,
            "game_seed": 8_100_000 + pair,
            "orientation": orientation,
        }
        for pair in range(500)
        for orientation in ("candidate_first", "candidate_second")
    ]
    external_search_config = {
        "n_full": 128,
        "n_full_wide": None,
        "max_depth": 80,
        "c_scale": 0.03,
        "public_observation": True,
        "value_readout": "scalar",
    }
    for role, checkpoint, win_rate in (
        ("candidate_panel", candidate, 0.55),
        ("champion_panel", champion, 0.54),
    ):
        source = tmp_path / f"{role}.raw.json"
        _write_json(
            source,
            {
                "stratum": "neutral-harness",
                "harness": "catanatron_native_engine",
                "baseline_bot": "catanatron_value",
                "mode": "search",
                "public_observation": True,
                "candidate_value_readout": "scalar",
                "trained_value_readouts": ["scalar"],
                "n_full": 128,
                "n_full_wide": None,
                "map_kind": "TOURNAMENT",
                "search_config": external_search_config,
                "gate_config": "flywheel",
                "pairs_requested": 500,
                "games_requested": 1000,
                "games_played": 1000,
                "games": external_games,
                "candidate_checkpoint": str(checkpoint),
                "candidate_checkpoint_md5": promotion._md5(checkpoint),
                "complete_pairs": 500,
                "candidate_win_rate": win_rate,
                "pentanomial_sprt": {"decision": "continue"},
                "verdict": "continue",
                "errors": [],
                "worker_errors": [],
                "games_engine_divergence": 0,
            },
        )
        external_sources.append((role, source))

    high_regret_source = tmp_path / "high_regret.raw.json"
    _write_json(
        high_regret_source,
        {
            "schema_version": promotion.HIGH_REGRET_SCHEMA,
            "suite": "held_out_high_regret",
            "held_out": True,
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "passed": True,
            "verdict": "H1",
            "complete_pairs": 200,
            "errors": [],
        },
    )
    bucket_source = tmp_path / "bucket_veto.raw.json"
    _write_json(
        bucket_source,
        {
            "schema_version": promotion.BUCKET_VETO_SCHEMA,
            "candidate": _checkpoint_ref(candidate),
            "champion": _checkpoint_ref(champion),
            "veto": False,
            "veto_buckets": [],
            "per_bucket": {
                "opening": {"status": "pass", "n": 100, "winrate": 0.53},
                "41+": {"status": "pass", "n": 80, "winrate": 0.51},
            },
        },
    )
    evidence_specs = {
        "mechanism_calibration": (
            calibration_sources,
            "pass",
            {"value_readout": "scalar", "max_rmse_regression": 0.02},
        ),
        "internal_h2h": ([('internal_h2h', internal_source)], "H1", {}),
        "external_panel": (
            external_sources,
            "pass",
            {"max_win_rate_regression": 0.02},
        ),
        "high_regret": ([('high_regret', high_regret_source)], "pass", {}),
        "bucket_veto": ([('bucket_veto', bucket_source)], "pass", {}),
    }
    evidence = []
    for kind in sorted(promotion.REQUIRED_EVIDENCE_KINDS):
        sources, verdict, result = evidence_specs[kind]
        evidence_path = tmp_path / f"{kind}.json"
        _write_evidence_envelope(
            evidence_path,
            kind=kind,
            contract=contract,
            candidate=candidate,
            champion=champion,
            sources=sources,
            verdict=verdict,
            result=result,
        )
        evidence.append(
            {
                "kind": kind,
                "path": str(evidence_path),
                "sha256": promotion._sha256(evidence_path),
            }
        )
    next_count = promotion_count + 1
    nth_required = next_count % 3 == 0
    adjudication = {
        "schema_version": promotion.ADJUDICATION_SCHEMA,
        "passed": True,
        "decision": "promote",
        "contract_sha256": contract["contract_sha256"],
        "candidate": {
            "path": str(candidate),
            "sha256": promotion._sha256(candidate),
            "version": 5,
            "training_report": {
                "path": str(report_path),
                "sha256": promotion._sha256(report_path),
            },
        },
        "champion": {
            "path": str(champion),
            "sha256": promotion._sha256(champion),
            "version": 4,
        },
        "checks": {name: True for name in promotion.REQUIRED_CHECKS},
        "nth_confirmation_required": nth_required,
        "nth_confirmation_passed": True if nth_required else False,
        "evidence": evidence,
    }
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    adjudication_path = tmp_path / "adjudication.json"
    _write_json(adjudication_path, adjudication)
    return {
        "champion": champion,
        "candidate": candidate,
        "registry": registry_path,
        "pointer": pointer,
        "contract_path": contract_path,
        "contract": contract,
        "adjudication": adjudication_path,
        "report": report_path,
        "receipt": tmp_path / "promotion.receipt.json",
        "lock": registry_path.with_suffix(registry_path.suffix + ".a1.lock"),
    }


def _verify(fixture: dict):
    def verify(path: Path, *, require_all_job_claims: bool = False):
        assert path == fixture["contract_path"]
        assert require_all_job_claims is True
        return fixture["contract"]

    return verify


def _execute(fixture: dict, *, go: bool):
    return promotion.execute_promotion(
        registry_path=fixture["registry"],
        current_pointer=fixture["pointer"],
        contract_lock=fixture["contract_path"],
        adjudication_path=fixture["adjudication"],
        receipt_path=fixture["receipt"],
        reason="A1 typed promotion",
        lock_path=fixture["lock"],
        go=go,
        verify_lock_fn=_verify(fixture),
    )


def _mutate_evidence_source(
    fixture: dict, *, kind: str, role: str, mutate
) -> None:
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(item for item in adjudication["evidence"] if item["kind"] == kind)
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    source_ref = next(item for item in envelope["sources"] if item["role"] == role)
    source_path = Path(source_ref["path"])
    source = json.loads(source_path.read_text())
    mutate(source)
    _write_json(source_path, source)
    source_ref["sha256"] = promotion._sha256(source_path)
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)


def test_dry_run_is_read_only_and_attests_global_n128(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before_registry = fixture["registry"].read_bytes()
    before_pointer = fixture["pointer"].read_bytes()

    result = _execute(fixture, go=False)

    assert result["status"] == "dry_run"
    assert result["contract"]["n_full"] == 128
    assert result["contract"]["n_full_wide"] is None
    assert result["fleet_ckpt_updated"] is False
    assert fixture["registry"].read_bytes() == before_registry
    assert fixture["pointer"].read_bytes() == before_pointer
    assert not fixture["receipt"].exists()


def test_go_updates_generator_and_pointer_with_committed_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    public_before = ChampionRegistry.load(fixture["registry"]).get_role("public_champion")

    receipt = _execute(fixture, go=True)

    assert receipt["status"] == "committed"
    assert receipt["fleet_ckpt_updated"] is False
    registry = ChampionRegistry.load(fixture["registry"])
    generator = registry.get_role("generator_champion")
    assert generator is not None
    assert Path(generator.checkpoint_path).resolve() == fixture["candidate"].resolve()
    assert generator.version == 5
    assert registry.promotion_count() == 1
    assert any(
        Path(entry.checkpoint_path).resolve() == fixture["champion"].resolve()
        for entry in registry.opponent_pool()
    )
    assert registry.get_role("public_champion") == public_before
    assert fixture["pointer"].read_text().strip() == str(fixture["candidate"].resolve())
    saved = json.loads(fixture["receipt"].read_text())
    assert saved["status"] == "committed"
    assert Path(saved["rollback"]["registry_backup"]).is_file()
    assert Path(saved["rollback"]["current_backup"]).is_file()


def test_recovery_is_dry_run_then_restores_exact_before_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    _execute(fixture, go=True)

    dry = promotion.recover_transaction(
        receipt_path=fixture["receipt"], lock_path=fixture["lock"], go=False
    )
    assert dry["status"] == "recovery_dry_run"
    assert fixture["registry"].read_bytes() != registry_before

    recovered = promotion.recover_transaction(
        receipt_path=fixture["receipt"], lock_path=fixture["lock"], go=True
    )
    assert recovered["status"] == "recovered"
    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert json.loads(fixture["receipt"].read_text())["status"] == "recovered"


def test_global_n196_contract_is_rejected_before_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, n_full=196)
    before = fixture["registry"].read_bytes()

    with pytest.raises(promotion.PromotionError, match="n_full=128"):
        _execute(fixture, go=True)

    assert fixture["registry"].read_bytes() == before
    assert not fixture["receipt"].exists()


def test_candidate_hash_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["candidate"].write_bytes(b"mutated after adjudication")

    with pytest.raises(promotion.PromotionError, match="artifact drift"):
        _execute(fixture, go=False)


def test_training_report_must_name_exact_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = json.loads(fixture["report"].read_text())
    report["checkpoint"] = str(fixture["champion"])
    _write_json(fixture["report"], report)
    adjudication = json.loads(fixture["adjudication"].read_text())
    adjudication["candidate"]["training_report"]["sha256"] = promotion._sha256(
        fixture["report"]
    )
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="report checkpoint differs"):
        _execute(fixture, go=False)


def test_bucket_insufficient_data_is_a_binding_veto(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(
        item for item in adjudication["evidence"] if item["kind"] == "bucket_veto"
    )
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    source_path = Path(envelope["sources"][0]["path"])
    source = json.loads(source_path.read_text())
    source["per_bucket"]["41+"] = {
        "status": "insufficient_data",
        "n": 4,
        "winrate": 0.75,
    }
    _write_json(source_path, source)
    envelope["sources"][0]["sha256"] = promotion._sha256(source_path)
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="not a pass"):
        _execute(fixture, go=False)


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("mechanism_calibration", "max_rmse_regression"),
        ("external_panel", "max_win_rate_regression"),
    ],
)
def test_evidence_cannot_launder_regression_with_its_own_tolerance(
    tmp_path: Path, kind: str, field: str
) -> None:
    fixture = _fixture(tmp_path)
    adjudication = json.loads(fixture["adjudication"].read_text())
    evidence_ref = next(item for item in adjudication["evidence"] if item["kind"] == kind)
    evidence_path = Path(evidence_ref["path"])
    envelope = json.loads(evidence_path.read_text())
    envelope["result"][field] = 1.0
    envelope.pop("evidence_sha256")
    envelope["evidence_sha256"] = promotion._digest_value(envelope)
    _write_json(evidence_path, envelope)
    evidence_ref["sha256"] = promotion._sha256(evidence_path)
    adjudication.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = promotion._digest_value(adjudication)
    _write_json(fixture["adjudication"], adjudication)

    with pytest.raises(promotion.PromotionError, match="fixed policy"):
        _execute(fixture, go=False)


def test_calibration_comparison_rejects_different_validation_seed_cohorts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def change_seed_manifest(source: dict) -> None:
        source["row_selection"]["seed_manifest_sha256"] = "sha256:" + "8" * 64

    _mutate_evidence_source(
        fixture,
        kind="mechanism_calibration",
        role="candidate_calibration",
        mutate=change_seed_manifest,
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts"):
        _execute(fixture, go=False)


def test_external_comparison_rejects_different_pair_seed_cohorts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def change_pair_seed(source: dict) -> None:
        source["games"][0]["game_seed"] += 1_000_000

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=change_pair_seed,
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts/configs"):
        _execute(fixture, go=False)


def test_external_comparison_rejects_different_search_configs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def change_search_config(source: dict) -> None:
        source["search_config"]["c_scale"] = 0.3

    _mutate_evidence_source(
        fixture,
        kind="external_panel",
        role="candidate_panel",
        mutate=change_search_config,
    )

    with pytest.raises(promotion.PromotionError, match="different cohorts/configs"):
        _execute(fixture, go=False)


def test_every_third_confirmation_is_derived_from_registry(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, promotion_count=2)
    payload = json.loads(fixture["adjudication"].read_text())
    payload["nth_confirmation_required"] = False
    payload["nth_confirmation_passed"] = False
    payload.pop("adjudication_sha256")
    payload["adjudication_sha256"] = promotion._digest_value(payload)
    _write_json(fixture["adjudication"], payload)

    with pytest.raises(promotion.PromotionError, match="every-third"):
        _execute(fixture, go=False)


def test_exclusive_lock_refuses_a_second_writer(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    descriptor = os.open(fixture["lock"], os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(promotion.PromotionError, match="already held"):
            _execute(fixture, go=False)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_alternate_lock_path_is_forbidden(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(promotion.PromotionError, match="alternate promotion lock"):
        promotion.execute_promotion(
            registry_path=fixture["registry"],
            current_pointer=fixture["pointer"],
            contract_lock=fixture["contract_path"],
            adjudication_path=fixture["adjudication"],
            receipt_path=fixture["receipt"],
            reason="A1 typed promotion",
            lock_path=tmp_path / "bypass.lock",
            go=False,
            verify_lock_fn=_verify(fixture),
        )


def test_symlink_registry_is_rejected_before_lock_or_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = tmp_path / "registry.alias.json"
    alias.symlink_to(fixture["registry"])

    with pytest.raises(promotion.PromotionError, match="must not contain symlinks"):
        promotion.execute_promotion(
            registry_path=alias,
            current_pointer=fixture["pointer"],
            contract_lock=fixture["contract_path"],
            adjudication_path=fixture["adjudication"],
            receipt_path=fixture["receipt"],
            reason="A1 typed promotion",
            go=False,
            verify_lock_fn=_verify(fixture),
        )


def test_failed_second_replace_rolls_registry_and_pointer_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    registry_before = fixture["registry"].read_bytes()
    pointer_before = fixture["pointer"].read_bytes()
    real_write = promotion._atomic_write_bytes
    failed = False

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if path == fixture["pointer"] and not failed and data != pointer_before:
            failed = True
            raise OSError("synthetic pointer replace failure")
        real_write(path, data)

    monkeypatch.setattr(promotion, "_atomic_write_bytes", fail_once)
    with pytest.raises(promotion.PromotionError, match="original.*restored"):
        _execute(fixture, go=True)

    assert fixture["registry"].read_bytes() == registry_before
    assert fixture["pointer"].read_bytes() == pointer_before
    assert json.loads(fixture["receipt"].read_text())["status"] == "rolled_back"


def test_recovery_refuses_tampered_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    receipt = json.loads(fixture["receipt"].read_text())
    receipt["reason"] = "tampered"
    _write_json(fixture["receipt"], receipt)

    with pytest.raises(promotion.PromotionError, match="semantic digest mismatch"):
        promotion.recover_transaction(receipt_path=fixture["receipt"], go=True)


def test_failed_recovery_restores_pre_recovery_committed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _execute(fixture, go=True)
    committed_registry = fixture["registry"].read_bytes()
    committed_pointer = fixture["pointer"].read_bytes()
    before_pointer = Path(str(fixture["receipt"]) + ".current.before").read_bytes()
    real_write = promotion._atomic_write_bytes
    failed = False

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if path == fixture["pointer"] and data == before_pointer and not failed:
            failed = True
            raise OSError("synthetic recovery pointer failure")
        real_write(path, data)

    monkeypatch.setattr(promotion, "_atomic_write_bytes", fail_once)
    with pytest.raises(promotion.PromotionError, match="pre-recovery.*restored"):
        promotion.recover_transaction(receipt_path=fixture["receipt"], go=True)

    assert fixture["registry"].read_bytes() == committed_registry
    assert fixture["pointer"].read_bytes() == committed_pointer
    assert json.loads(fixture["receipt"].read_text())["status"] == "committed"
