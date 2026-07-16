from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_combined_candidate_handoff as handoff
from tools import a1_lineage_dose as lineage


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
        rate = candidate_rate if is_candidate else champion_rate
        games = [
            {
                "game_seed": 10 + index // 2,
                "orientation": (
                    "candidate_first" if index % 2 == 0 else "candidate_second"
                ),
                "candidate_won": index < int(rate * 1_000),
            }
            for index in range(1_000)
        ]
        return {
            "candidate_win_rate": rate,
            "games": games,
            "effective_search_config": {"n_full": 128, "c_scale": 0.03},
            "fleet_merge": {
                "seed_intervals": (
                    candidate_seeds
                    if is_candidate and candidate_seeds is not None
                    else [
                        {
                            "base_seed": 10,
                            "end_seed": 570,
                            "path": "candidate.json" if is_candidate else "champion.json",
                        }
                    ]
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

    _install_replays(monkeypatch, files, internal_verdict="H1")
    assert handoff.adjudicate(manifest)["gates"]["internal_h2h"]["passed"] is True

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
            "cohort-exclusions.json",
            "candidate.pt",
        )
    }
    passing = {
        "passed": True,
        "training_receipt": handoff._ref(training_receipt, where="fixture"),  # noqa: SLF001
        "cohort_exclusions": handoff._ref(  # noqa: SLF001
            files["cohort-exclusions.json"], where="fixture"
        ),
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
        "cohort_exclusions": handoff._ref(  # noqa: SLF001
            files["cohort-exclusions.json"], where="fixture"
        ),
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


def test_promotion_report_rejects_curriculum_parent_chaining(
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
    curriculum_declaration = {
        "schema_version": "a1-curriculum-declaration-v1",
        "kind": "sequential_checkpoint_curriculum",
        "parent_receipt_path": str(parent_receipt),
        "parent_receipt_sha256": handoff.promotion._sha256(parent_receipt),  # noqa: SLF001
        "parent_arm_id": "n256",
        "parent_subset_id": "full-56k",
        "parent_checkpoint": parent_binding["parent_checkpoint"],
        "generation_producer_sha256": producer_sha,
        "parent_lineage_dose": {
            "schema_version": "a1-lineage-dose-v1",
            "mode": "direct_from_declared_producer",
            "declared_producer_sha256": producer_sha,
            "init_checkpoint_sha256": producer_sha,
            "parent_receipt_sha256": None,
            "optimizer_state_continuity": "fresh_optimizer_per_dose",
            "objective_exposure": {
                "measurement_status": "not_yet_bound_exactly",
                "policy_active_sampled_rows": None,
                "value_active_sampled_rows": None,
                "anchor_eligible_sampled_rows": None,
            },
            "prior_sampled_rows": 0,
            "prior_optimizer_steps": 0,
            "current_sampled_rows": 56_000,
            "current_optimizer_steps": 14,
            "cumulative_sampled_rows": 56_000,
            "cumulative_optimizer_steps": 14,
        },
        "parent_cumulative_sampled_rows": 56_000,
        "parent_cumulative_optimizer_steps": 14,
        "child_arm_id": "n128",
        "child_subset_id": "full-140k",
    }
    recipe = {
        "epochs": 1,
        "max_steps": 0,
        "world_size": 8,
        "batch_size": 512,
        "grad_accum_steps": 1,
        "global_batch_size": 4096,
        "ddp_shard_data": False,
        "symmetry_augment": False,
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
        "grad_accum_steps": 1,
        "a1_decisive_training_semantics": {
            "schema_version": "a1-decisive-training-semantics-v1",
            "grad_accum_steps": 1,
            "gradient_accumulation_contract": "single_microbatch_exact",
        },
        "checkpoint": str(candidate),
        "init_checkpoint_sha256": parent_binding["parent_checkpoint"]["sha256"],
        "a1_curriculum_parent": parent_binding,
        "a1_curriculum_declaration": curriculum_declaration,
        "a1_lineage_dose": lineage.curriculum_lineage_dose(
            declared_producer_sha256=producer_sha,
            init_checkpoint_sha256=parent_binding["parent_checkpoint"]["sha256"],
            parent_receipt_sha256=parent_binding["receipt_sha256"],
            parent_lineage_dose=curriculum_declaration["parent_lineage_dose"],
            current_sampled_rows=140_000,
            current_optimizer_steps=10,
        ),
        "steps_completed": 10,
        "epochs": 1,
        "max_steps": 0,
        "symmetry_augment": False,
    }
    report.write_text(json.dumps(payload))
    with pytest.raises(
        handoff.promotion.PromotionError,
        match="candidate chaining/curriculum lineage is not promotion-eligible",
    ):
        handoff.promotion._verify_training_report(  # noqa: SLF001
            report,
            contract={
                "science": {},
                "checkpoints": [{"role": "producer", "sha256": producer_sha}],
            },
            contract_sha256="sha256:contract",
            candidate_path=candidate.resolve(),
            candidate_sha256=handoff.promotion._sha256(candidate),  # noqa: SLF001
        )
