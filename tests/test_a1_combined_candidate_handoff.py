from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_combined_candidate_handoff as handoff


def _file(tmp_path: Path, name: str, body: bytes = b"x") -> Path:
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    files = {
        name: _file(tmp_path, name)
        for name in (
            "champion.pt",
            "candidate.pt",
            "training.receipt.json",
            "internal.json",
            "candidate.external.json",
            "champion.external.json",
        )
    }
    manifest = {
        "schema_version": handoff.MANIFEST_SCHEMA,
        "champion": handoff._ref(files["champion.pt"], where="fixture"),  # noqa: SLF001
        "training_receipt": handoff._ref(  # noqa: SLF001
            files["training.receipt.json"], where="fixture"
        ),
        "internal_pool": handoff._ref(files["internal.json"], where="fixture"),  # noqa: SLF001
        "candidate_external_pool": handoff._ref(  # noqa: SLF001
            files["candidate.external.json"], where="fixture"
        ),
        "champion_external_pool": handoff._ref(  # noqa: SLF001
            files["champion.external.json"], where="fixture"
        ),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    files["manifest"] = path
    return path, files


def _install_replays(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, Path],
    *,
    internal_verdict: str = "accept_h1",
    candidate_rate: float = 0.41,
    champion_rate: float = 0.42,
    candidate_seeds: list[dict] | None = None,
) -> None:
    monkeypatch.setattr(
        handoff,
        "_verify_training_receipt",
        lambda _path: (
            {"receipt_sha256": "sha256:receipt"},
            files["candidate.pt"],
        ),
    )
    monkeypatch.setattr(
        handoff.dual_adjudicator,
        "_replay_internal",
        lambda _path, *, candidate, champion: {
            "baseline_checkpoint_sha256": handoff.promotion._sha256(champion),  # noqa: SLF001
            "verdict": internal_verdict,
            "pentanomial_sprt": {"llr": 3.0},
            "candidate_win_rate": 0.56,
            "complete_pairs": 560,
        },
    )

    def replay_external(path: Path, *, candidate: Path) -> dict:
        is_candidate = path.name.startswith("candidate.")
        return {
            "candidate_win_rate": candidate_rate if is_candidate else champion_rate,
            "effective_search_config": {"n_full": 128, "c_scale": 0.03},
            "fleet_merge": {
                "seed_intervals": (
                    candidate_seeds
                    if is_candidate and candidate_seeds is not None
                    else [{"base_seed": 10, "end_seed": 570}]
                )
            },
        }

    monkeypatch.setattr(
        handoff.dual_adjudicator, "_replay_neutral", replay_external
    )


def test_combined_handoff_requires_internal_h1_and_matched_external_nonregression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, files = _fixture(tmp_path)
    _install_replays(monkeypatch, files)
    value = handoff.adjudicate(manifest)
    assert value["passed"] is True
    assert value["decision"] == "promotion_evidence_may_proceed"
    assert value["gates"]["internal_h2h"]["passed"] is True
    assert value["gates"]["external_panel"]["passed"] is True
    assert value["promotion"]["ready"] is False

    _install_replays(monkeypatch, files, internal_verdict="continue")
    assert handoff.adjudicate(manifest)["passed"] is False

    _install_replays(monkeypatch, files, candidate_rate=0.39, champion_rate=0.42)
    rejected = handoff.adjudicate(manifest)
    assert rejected["passed"] is False
    assert rejected["gates"]["external_panel"]["passed"] is False


def test_combined_handoff_rejects_external_cohort_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, files = _fixture(tmp_path)
    _install_replays(
        monkeypatch,
        files,
        candidate_seeds=[{"base_seed": 11, "end_seed": 571}],
    )
    with pytest.raises(handoff.CombinedHandoffError, match="different cohorts"):
        handoff.adjudicate(manifest)


def test_handoff_result_replays_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, files = _fixture(tmp_path)
    _install_replays(monkeypatch, files)
    value = handoff.adjudicate(manifest)
    result = tmp_path / "handoff.json"
    handoff.write_new(result, value)
    assert handoff.verify_result(result) == value
    tampered = json.loads(result.read_text())
    tampered["passed"] = False
    result.chmod(0o644)
    result.write_text(json.dumps(tampered))
    with pytest.raises(handoff.CombinedHandoffError, match="digest"):
        handoff.verify_result(result)


def test_promotion_plan_requires_passing_handoff_and_full_transaction_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff_path = _file(tmp_path, "handoff.json")
    training_receipt = _file(tmp_path, "training.receipt.json")
    files = {
        name: _file(tmp_path, name)
        for name in (
            "registry.jsonl",
            "CURRENT_CHAMPION",
            "contract.lock.json",
            "promotion.adjudication.json",
            "candidate.pt",
        )
    }
    passing = {
        "passed": True,
        "training_receipt": handoff._ref(training_receipt, where="fixture"),  # noqa: SLF001
        "candidate": handoff._ref(files["candidate.pt"], where="fixture"),  # noqa: SLF001
    }
    monkeypatch.setattr(handoff, "verify_result", lambda _path: passing)
    manifest = {
        "schema_version": handoff.PROMOTION_MANIFEST_SCHEMA,
        "registry": handoff._ref(files["registry.jsonl"], where="fixture"),  # noqa: SLF001
        "current_pointer": handoff._ref(files["CURRENT_CHAMPION"], where="fixture"),  # noqa: SLF001
        "contract_lock": handoff._ref(files["contract.lock.json"], where="fixture"),  # noqa: SLF001
        "adjudication": handoff._ref(  # noqa: SLF001
            files["promotion.adjudication.json"], where="fixture"
        ),
        "training_receipt": handoff._ref(training_receipt, where="fixture"),  # noqa: SLF001
        "receipt": str(tmp_path / "promotion.receipt.json"),
        "reason": "combined candidate cleared every gate",
    }
    manifest_path = tmp_path / "promotion.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(
        handoff.promotion,
        "prepare_promotion",
        lambda **kwargs: calls.append(kwargs) or {"status": "dry_run"},
    )
    plan = handoff.build_promotion_plan(handoff_path, manifest_path)
    assert calls and plan["passed"] is True
    assert plan["command"][2] == "promote"
    assert "--go" not in plan["command"]

    monkeypatch.setattr(handoff, "verify_result", lambda _path: {"passed": False})
    with pytest.raises(handoff.CombinedHandoffError, match="both evaluation gates"):
        handoff.build_promotion_plan(handoff_path, manifest_path)


def test_promotion_report_accepts_only_exact_authenticated_curriculum_parent(
    tmp_path: Path,
) -> None:
    producer = _file(tmp_path, "producer.pt", b"producer")
    parent = _file(tmp_path, "n256.pt", b"n256")
    parent_receipt = _file(tmp_path, "n256.receipt.json", b"receipt")
    candidate = _file(tmp_path, "combined.pt", b"combined")
    producer_sha = handoff.promotion._sha256(producer)  # noqa: SLF001
    parent_binding = {
        "schema_version": "a1-curriculum-parent-binding-v1",
        "receipt_path": str(parent_receipt),
        "receipt_sha256": handoff.promotion._sha256(parent_receipt),  # noqa: SLF001
        "parent_arm_id": "n256",
        "parent_subset_id": "full-56k",
        "parent_checkpoint": {
            "path": str(parent),
            "sha256": handoff.promotion._sha256(parent),  # noqa: SLF001
        },
        "generation_producer_sha256": producer_sha,
    }
    recipe = {
        "epochs": 1,
        "max_steps": 0,
        "world_size": 8,
        "batch_size": 512,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
        "ddp_shard_data": False,
    }
    report = tmp_path / "report.json"
    payload = {
        "a1_dual_arm_execution_binding": {"schema_version": "binding"},
        "a1_contract_sha256": "sha256:contract",
        "a1_learner_training_recipe_sha256": handoff.promotion._digest_value(recipe),  # noqa: SLF001
        "a1_bound_learner_training_recipe": recipe,
        "arch": "entity_graph",
        "mask_hidden_info": True,
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "world_size": 8,
        "batch_size": 512,
        "checkpoint": str(candidate),
        "init_checkpoint_sha256": parent_binding["parent_checkpoint"]["sha256"],
        "a1_curriculum_parent": parent_binding,
        "steps_completed": 10,
        "epochs": 1,
        "max_steps": 0,
    }
    report.write_text(json.dumps(payload))
    verified = handoff.promotion._verify_training_report(  # noqa: SLF001
        report,
        contract={"checkpoints": [{"role": "producer", "sha256": producer_sha}]},
        contract_sha256="sha256:contract",
        candidate_path=candidate.resolve(),
        candidate_sha256=handoff.promotion._sha256(candidate),  # noqa: SLF001
    )
    assert verified["a1_curriculum_parent"] == parent_binding

    ordinary = dict(payload)
    ordinary.pop("a1_curriculum_parent")
    report.write_text(json.dumps(ordinary))
    with pytest.raises(
        handoff.promotion.PromotionError, match="init checkpoint differs from producer"
    ):
        handoff.promotion._verify_training_report(  # noqa: SLF001
            report,
            contract={"checkpoints": [{"role": "producer", "sha256": producer_sha}]},
            contract_sha256="sha256:contract",
            candidate_path=candidate.resolve(),
            candidate_sha256=handoff.promotion._sha256(candidate),  # noqa: SLF001
        )

    report.write_text(json.dumps(payload))
    parent.write_bytes(b"drift")
    with pytest.raises(handoff.promotion.PromotionError, match="bytes drifted"):
        handoff.promotion._verify_training_report(  # noqa: SLF001
            report,
            contract={"checkpoints": [{"role": "producer", "sha256": producer_sha}]},
            contract_sha256="sha256:contract",
            candidate_path=candidate.resolve(),
            candidate_sha256=handoff.promotion._sha256(candidate),  # noqa: SLF001
        )
