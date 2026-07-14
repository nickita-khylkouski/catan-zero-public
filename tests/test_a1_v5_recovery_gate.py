from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools import a1_promotion_transaction as promotion
from tools import a1_v5_disaster_recovery as recovery
from tools import a1_v5_recovery_gate as gate
from tools.champion_registry import ChampionRegistry, RolePointer
from tools.fleet import a1_h100_eval_fleet


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: bytes | dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    else:
        path.write_bytes(value)
    return path.resolve()


@pytest.fixture()
def gate_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    recovered = _write(tmp_path / "recovered-v5.pt", b"recovered")
    safety = _write(tmp_path / "f7.pt", b"safety")
    candidate = _write(tmp_path / "candidate.pt", b"candidate")
    registry_path = tmp_path / "champion_registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role(
        "generator_champion",
        recovered,
        version=5,
        provenance={"recovery_schema": recovery.RECOVERY_SCHEMA},
    )
    registry.save()
    registry_path = registry_path.resolve()
    pointer = _write(tmp_path / "CURRENT_CHAMPION", (str(recovered) + "\n").encode())
    receipt_path = _write(tmp_path / "recovery.json", {})
    contract_path = _write(tmp_path / "contract.json", {})
    adjudication_path = _write(tmp_path / "adjudication.json", {})
    training_path = _write(tmp_path / "training.json", {})
    cohort_exclusions_path = _write(tmp_path / "cohort-exclusions.json", {})
    games = [
        {
            "pair_id": pair,
            "game_seed": gate.F7_VETO_BASE_SEED + pair,
            "orientation": orientation,
        }
        for pair in range(gate.F7_VETO_COMPLETE_PAIRS)
        for orientation in ("candidate_first", "candidate_second")
    ]
    report_path = _write(
        tmp_path / "f7-report.json",
        {
            "complete_pairs": gate.F7_VETO_COMPLETE_PAIRS,
            "verdict": "continue",
            "games": games,
        },
    )
    search = {"c_scale": 0.1, "n_full": 128}
    recovery_authority = {
        "schema_version": "a1-v5-disaster-recovery-authority-v1",
        "recovered_generator": {
            "path": str(recovered),
            "sha256": _sha(recovered),
            "md5": hashlib.md5(recovered.read_bytes()).hexdigest(),  # noqa: S324
        },
        recovery.RECOVERY_RELATION: {
            "path": str(safety),
            "sha256": _sha(safety),
            "md5": hashlib.md5(safety.read_bytes()).hexdigest(),  # noqa: S324
            "relationship": recovery.RECOVERY_RELATION,
        },
        "producer_identity": {"search_config": search},
    }
    recovery_replay = {
        "authority": recovery_authority,
        "receipt": {
            "registry": {"path": str(registry_path)},
            "current_pointer": {"path": str(pointer)},
        },
    }
    contract = {"contract_sha256": "sha256:" + "a" * 64}
    candidate_identity = {"search_config": search, "agent_identity_sha256": "x"}
    verified = {
        "promotion_mode": "disaster_recovery_parent",
        "candidate": {
            "path": str(candidate),
            "sha256": _sha(candidate),
            "agent_identity": candidate_identity,
        },
        "champion": {
            "path": str(recovered),
            "sha256": _sha(recovered),
        },
        "adjudication_sha256": "sha256:" + "b" * 64,
        "final_cohort_intervals": [
            {"kind": "internal_h2h", "base_seed": 100, "end_seed": 400},
            {"kind": "external_panel", "base_seed": 500, "end_seed": 900},
        ],
    }
    observed: dict[str, Any] = {}

    def fake_contract(*_args: Any, **_kwargs: Any):
        return contract, None

    def fake_adjudication(*_args: Any, **kwargs: Any):
        observed["recovery_authority"] = kwargs.get("recovery_authority")
        return verified

    def fake_h2h(_payload: dict[str, Any], **kwargs: Any) -> None:
        observed["h2h"] = kwargs

    def fake_cohort_exclusions(
        path: Path,
        *,
        contract_sha256: str,
        candidate_sha256: str,
        final_intervals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        observed["cohort_exclusions"] = {
            "path": path,
            "contract_sha256": contract_sha256,
            "candidate_sha256": candidate_sha256,
            "final_intervals": final_intervals,
        }
        return {
            "manifest": {"path": str(path), "sha256": _sha(path)},
            "final_seed_intervals": final_intervals,
            "overlap_count": 0,
        }

    monkeypatch.setattr(promotion, "_verify_contract_with_snapshot", fake_contract)
    monkeypatch.setattr(promotion, "_verify_adjudication", fake_adjudication)
    monkeypatch.setattr(promotion, "_verify_internal_h2h_source", fake_h2h)
    monkeypatch.setattr(promotion, "_verify_internal_h2h_cohort", lambda *_a, **_k: None)
    monkeypatch.setattr(
        promotion,
        "_verify_cohort_exclusions",
        fake_cohort_exclusions,
    )
    monkeypatch.setattr(
        promotion,
        "_sealed_evaluation_semantics",
        lambda _contract: {"n_full": 128},
    )
    return {
        "paths": {
            "recovery_receipt_path": receipt_path,
            "contract_lock_path": contract_path,
            "standard_adjudication_path": adjudication_path,
            "training_receipt_path": training_path,
            "cohort_exclusions_path": cohort_exclusions_path,
            "registry_path": registry_path,
            "current_pointer_path": pointer,
            "f7_nonregression_report_path": report_path,
        },
        "replay": recovery_replay,
        "verified": verified,
        "observed": observed,
        "report": report_path,
    }


def _verify(inputs: dict[str, Any]) -> dict[str, Any]:
    return gate.verify_recovery_gate(
        **inputs["paths"],
        verify_lock_fn=lambda *_a, **_k: {},
        recovery_verifier_fn=lambda _path: inputs["replay"],
    )


def test_dual_gate_reuses_full_adjudication_and_fixed_f7_veto(
    gate_inputs: dict[str, Any], tmp_path: Path
) -> None:
    authority = _verify(gate_inputs)
    assert authority["policy"] == {
        "dual_baseline_conjunctive": True,
        "strict_h1_over_recovered_parent": True,
        "f7_h0_veto": True,
        "fresh_cohorts_required": True,
        "promotion_eligible": True,
        "auto_promotion": False,
    }
    assert authority["f7_non_regression_veto"]["complete_pairs"] == 300
    assert gate_inputs["observed"]["recovery_authority"] is gate_inputs["replay"][
        "authority"
    ]
    h2h = gate_inputs["observed"]["h2h"]
    assert h2h["comparison_mode"] == gate.F7_COMPARISON_MODE
    assert h2h["verdict_policy"] == "non_regression_veto"
    assert h2h["required_n_full"] == 128
    cohort = gate_inputs["observed"]["cohort_exclusions"]
    assert cohort["contract_sha256"] == "sha256:" + "a" * 64
    assert cohort["candidate_sha256"] == _sha(
        Path(gate_inputs["verified"]["candidate"]["path"])
    )
    assert cohort["final_intervals"] == [
        *gate_inputs["verified"]["final_cohort_intervals"],
        {
            "kind": "f7_non_regression_veto",
            "base_seed": gate.F7_VETO_BASE_SEED,
            "end_seed": gate.F7_VETO_BASE_SEED
            + gate.F7_VETO_COMPLETE_PAIRS,
        },
    ]

    output = tmp_path / "authority.json"
    written = gate.write_recovery_gate_authority(
        output,
        **gate_inputs["paths"],
        verify_lock_fn=lambda *_a, **_k: {},
        recovery_verifier_fn=lambda _path: gate_inputs["replay"],
    )
    replayed = gate.verify_recovery_gate_authority(
        output,
        verify_lock_fn=lambda *_a, **_k: {},
        recovery_verifier_fn=lambda _path: gate_inputs["replay"],
    )
    assert replayed == written == authority


def test_f7_h0_is_a_conjunctive_veto(gate_inputs: dict[str, Any]) -> None:
    report = json.loads(gate_inputs["report"].read_text())
    report["verdict"] = "H0"
    gate_inputs["report"].write_text(json.dumps(report) + "\n")
    with pytest.raises(gate.RecoveryGateError, match="reached H0"):
        _verify(gate_inputs)


def test_f7_cohort_must_be_exact_and_disjoint(gate_inputs: dict[str, Any]) -> None:
    report = json.loads(gate_inputs["report"].read_text())
    report["games"][0]["game_seed"] += 50_000
    gate_inputs["report"].write_text(json.dumps(report) + "\n")
    with pytest.raises(gate.RecoveryGateError, match="exact fixed fresh seed"):
        _verify(gate_inputs)

    report["games"][0]["game_seed"] = gate.F7_VETO_BASE_SEED
    gate_inputs["report"].write_text(json.dumps(report) + "\n")
    gate_inputs["verified"]["final_cohort_intervals"] = [
        {
            "kind": "internal_h2h",
            "base_seed": gate.F7_VETO_BASE_SEED,
            "end_seed": gate.F7_VETO_BASE_SEED + 1,
        }
    ]
    with pytest.raises(gate.RecoveryGateError, match="overlaps ordinary gate"):
        _verify(gate_inputs)


def test_recovery_gate_refuses_prior_cohort_reuse(
    gate_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_reuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise promotion.PromotionError(
            "final promotion cohort overlaps a prior diagnostic/adjudication cohort"
        )

    monkeypatch.setattr(promotion, "_verify_cohort_exclusions", refuse_reuse)
    with pytest.raises(gate.RecoveryGateError, match="overlaps a prior diagnostic"):
        _verify(gate_inputs)


def test_explicit_registry_and_pointer_must_be_receipt_owned(
    gate_inputs: dict[str, Any], tmp_path: Path
) -> None:
    gate_inputs["replay"]["receipt"]["registry"]["path"] = str(
        _write(tmp_path / "other-registry.json", {})
    )
    with pytest.raises(gate.RecoveryGateError, match="differ from recovery receipt"):
        _verify(gate_inputs)


def _recovery_incumbent_fixture(
    tmp_path: Path,
) -> tuple[Path, RolePointer, dict[str, Any]]:
    champion = _write(tmp_path / "recovered.pt", b"recovered-parent")
    lineage = "sha256:" + "1" * 64
    pointer = RolePointer(
        role="generator_champion",
        checkpoint_path=str(champion),
        md5=hashlib.md5(champion.read_bytes()).hexdigest(),  # noqa: S324
        version=5,
        updated_at=1.0,
        provenance={
            "recovery_schema": recovery.RECOVERY_SCHEMA,
            "recovery_lineage_id": lineage,
        },
    )
    authority: dict[str, Any] = {
        "schema_version": "a1-v5-disaster-recovery-authority-v1",
        "recovery_receipt": {
            "path": "/recovery.json",
            "sha256": "sha256:" + "2" * 64,
            "recovery_receipt_sha256": "sha256:" + "3" * 64,
        },
        "recovery_lineage_id": lineage,
        "recovered_generator": {
            "path": str(champion),
            "sha256": _sha(champion),
            "md5": pointer.md5,
        },
        recovery.RECOVERY_RELATION: {
            "path": "/f7.pt",
            "sha256": "sha256:" + "4" * 64,
        },
        "producer_identity": {"search_config": {"c_scale": 0.1}},
        "promotion_proof_recreated": False,
        "dual_baseline_fresh_gate_required": True,
        "promotion_eligible": False,
        "training_proof": False,
        "wave_lineage_mode": "recovery_reference",
    }
    authority["authority_sha256"] = promotion._digest_value(authority)  # noqa: SLF001
    return champion, pointer, authority


def test_ordinary_and_branch_modes_cannot_consume_recovery_incumbent(
    tmp_path: Path,
) -> None:
    champion, pointer, authority = _recovery_incumbent_fixture(tmp_path)
    with pytest.raises(promotion.PromotionError, match="ordinary promotion mode"):
        promotion._verify_recovery_incumbent_authority(  # noqa: SLF001
            incumbent=pointer,
            recovery_authority=None,
            champion_path=champion,
            champion_sha256=_sha(champion),
            branch_challenge=False,
        )
    with pytest.raises(promotion.PromotionError, match="branch-challenge fallback"):
        promotion._verify_recovery_incumbent_authority(  # noqa: SLF001
            incumbent=pointer,
            recovery_authority=authority,
            champion_path=champion,
            champion_sha256=_sha(champion),
            branch_challenge=True,
        )


def test_recovery_authority_must_match_registry_lineage_and_parent(
    tmp_path: Path,
) -> None:
    champion, pointer, authority = _recovery_incumbent_fixture(tmp_path)
    assert promotion._verify_recovery_incumbent_authority(  # noqa: SLF001
        incumbent=pointer,
        recovery_authority=authority,
        champion_path=champion,
        champion_sha256=_sha(champion),
        branch_challenge=False,
    )
    authority["recovery_lineage_id"] = "sha256:" + "9" * 64
    authority.pop("authority_sha256")
    authority["authority_sha256"] = promotion._digest_value(authority)  # noqa: SLF001
    with pytest.raises(promotion.PromotionError, match="differs from authoritative"):
        promotion._verify_recovery_incumbent_authority(  # noqa: SLF001
            incumbent=pointer,
            recovery_authority=authority,
            champion_path=champion,
            champion_sha256=_sha(champion),
            branch_challenge=False,
        )


def test_v3_recovery_safety_binding_tamper_is_rejected(tmp_path: Path) -> None:
    recovered = _write(tmp_path / "recovered.pt", b"recovered")
    safety = _write(tmp_path / "f7.pt", b"safety")
    search = {"c_scale": 0.1, "n_full": 128}
    checkpoint_ref = a1_h100_eval_fleet._checkpoint_ref(recovered)  # noqa: SLF001
    deployed_identity = a1_h100_eval_fleet._digest(  # noqa: SLF001
        {
            "schema_version": "a1-deployed-agent-search-config-v1",
            "checkpoint": checkpoint_ref,
            "search_config": search,
        }
    )
    registry = ChampionRegistry(tmp_path / "registry.json")
    registry.set_role(
        "generator_champion",
        recovered,
        version=5,
        provenance={
            "recovery_schema": recovery.RECOVERY_SCHEMA,
            "a1_candidate_search_config": search,
            "a1_candidate_agent_identity_sha256": deployed_identity,
        },
    )
    registry.save()
    binding = a1_h100_eval_fleet._evaluation_binding(  # noqa: SLF001
        candidate_parent=recovered,
        baseline=safety,
        registry=registry,
        comparison_mode=gate.F7_COMPARISON_MODE,
        historical_comparison_reason=gate.F7_COMPARISON_REASON,
        champion_c_scale=0.1,
    )
    promotion._verify_evaluation_baseline_binding(  # noqa: SLF001
        binding,
        champion_path=safety,
        champion_sha256=_sha(safety),
        champion_search_config=search,
        base=tmp_path,
        where="test recovery binding",
        comparison_mode=gate.F7_COMPARISON_MODE,
        candidate_parent_path=recovered,
        candidate_parent_sha256=_sha(recovered),
        candidate_parent_search_config=search,
    )
    tampered = json.loads(json.dumps(binding))
    tampered["historical_comparison_reason"] = "generic historical comparison"
    with pytest.raises(promotion.PromotionError, match="exact recovery safety"):
        promotion._verify_evaluation_baseline_binding(  # noqa: SLF001
            tampered,
            champion_path=safety,
            champion_sha256=_sha(safety),
            champion_search_config=search,
            base=tmp_path,
            where="test recovery binding",
            comparison_mode=gate.F7_COMPARISON_MODE,
            candidate_parent_path=recovered,
            candidate_parent_sha256=_sha(recovered),
            candidate_parent_search_config=search,
        )


def test_strict_parent_gate_cannot_be_relabelled_or_change_baseline(
    gate_inputs: dict[str, Any], tmp_path: Path
) -> None:
    gate_inputs["verified"]["promotion_mode"] = "branch_challenge"
    with pytest.raises(gate.RecoveryGateError, match="recovered parent"):
        _verify(gate_inputs)
    gate_inputs["verified"]["promotion_mode"] = "disaster_recovery_parent"
    other = _write(tmp_path / "other-parent.pt", b"other")
    gate_inputs["verified"]["champion"] = {
        "path": str(other),
        "sha256": _sha(other),
    }
    with pytest.raises(gate.RecoveryGateError, match="strict H1 baseline"):
        _verify(gate_inputs)
