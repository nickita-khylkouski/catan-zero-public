from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_dual_arm_adjudicator as adjudicator


IDENTITIES = [
    ("n256", "full-56k"),
    ("n128", "matched-56k"),
    ("n128", "compute-112k"),
    ("n128", "full-140k"),
]


def _file(tmp_path: Path, name: str, content: bytes = b"x") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    files: dict[str, Path] = {}
    champion = _file(tmp_path, "champion.pt", b"champion")
    files["champion"] = champion
    candidates = []
    for arm, subset in IDENTITIES:
        stem = f"{arm}-{subset}"
        receipt = _file(tmp_path, f"{stem}.receipt.json")
        internal = _file(tmp_path, f"{stem}.internal.json")
        neutral = _file(tmp_path, f"{stem}.neutral.json")
        checkpoint = _file(tmp_path, f"{stem}.pt", stem.encode())
        files[f"{stem}.checkpoint"] = checkpoint
        candidates.append({
            "arm_id": arm, "subset_id": subset,
            "training_receipt": adjudicator._ref(receipt, where="fixture"),  # noqa: SLF001
            "internal_pool": adjudicator._ref(internal, where="fixture"),  # noqa: SLF001
            "neutral_pool": adjudicator._ref(neutral, where="fixture"),  # noqa: SLF001
        })
    tournament_specs = {
        "causal": ("n256/full-56k", "n128/matched-56k"),
        "q0": ("n128/compute-112k", "n128/matched-56k"),
        "q1": ("n128/full-140k", "n128/matched-56k"),
        "q2": ("n128/full-140k", "n128/compute-112k"),
        "final": ("n128/full-140k", "n256/full-56k"),
    }
    matches = {}
    for name, (candidate, baseline) in tournament_specs.items():
        report = _file(tmp_path, f"tournament-{name}.json")
        matches[name] = {
            "candidate": candidate, "baseline": baseline,
            "pool": adjudicator._ref(report, where="fixture"),  # noqa: SLF001
        }
    value = {
        "schema_version": adjudicator.MANIFEST_SCHEMA,
        "champion": adjudicator._ref(champion, where="fixture"),  # noqa: SLF001
        "candidates": candidates,
        "tournament": {
            "causal_teacher": matches["causal"],
            "n128_quantity_curve": [matches["q0"], matches["q1"], matches["q2"]],
            "finalist": matches["final"],
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path, files


def test_adjudicator_selects_best_fixed_search_candidate_and_seals_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, files = _manifest(tmp_path)

    input_ref = adjudicator._ref(files["champion"], where="fixture")  # noqa: SLF001

    def fake_receipt(path: Path, **_kwargs) -> dict:
        stem = path.name.removesuffix(".receipt.json")
        arm, subset = next(
            identity for identity in IDENTITIES if stem == f"{identity[0]}-{identity[1]}"
        )
        checkpoint = files[f"{stem}.checkpoint"]
        return {
            "arm_id": arm, "subset_id": subset,
            "inputs": {
                "corpus_meta": input_ref,
                "learner_lock": input_ref,
                "validation": input_ref,
                "producer": input_ref,
            },
            "outputs": {"checkpoint": adjudicator._ref(checkpoint, where="fixture")},  # noqa: SLF001
        }

    monkeypatch.setattr(adjudicator.dual_train, "verify_receipt", fake_receipt)
    monkeypatch.setattr(adjudicator.dual_train, "verify_inputs", lambda **_kwargs: {})

    def fake_internal(path: Path, *, candidate: Path, champion: Path) -> dict:
        # Common-champion LLRs deliberately make n256 look largest. They are
        # eligibility evidence only and must not select the winner.
        rank = next(
            (i for i, identity in enumerate(IDENTITIES) if path.name.startswith(f"{identity[0]}-{identity[1]}")),
            0,
        )
        return {
            "verdict": "accept_h1",
            "pentanomial_sprt": {"llr": 999.0 if rank == 0 else float(rank + 1)},
            "effective_search_config": {"n_full": 128},
            "baseline_checkpoint_sha256": adjudicator.promotion._sha256(champion),  # noqa: SLF001
            "candidate_win_rate": 0.55,
            "complete_pairs": 100,
            "fleet_merge": {"seed_intervals": [{"base_seed": 1, "end_seed": 101}]},
        }

    monkeypatch.setattr(adjudicator, "_replay_internal", fake_internal)
    monkeypatch.setattr(
        adjudicator,
        "_replay_neutral",
        lambda path, *, candidate: {
            "verdict": "accept_h1", "pentanomial_sprt": {"llr": 1.0},
            "effective_search_config": {"n_full": 128},
            "fleet_merge": {"seed_intervals": [{"base_seed": 201, "end_seed": 301}]},
        },
    )
    result = adjudicator.adjudicate(manifest)
    assert result["winner"]["subset_id"] == "full-140k"
    assert result["decision"] == "winner_selected_full_evidence_required"
    assert result["promotion"]["ready"] is False
    assert "promotion_command" not in result

    def mismatched_neutral(path: Path, *, candidate: Path) -> dict:
        base = 202 if path.name.startswith("n128-matched-56k") else 201
        return {
            "verdict": "accept_h1", "pentanomial_sprt": {"llr": 1.0},
            "effective_search_config": {"n_full": 128},
            "fleet_merge": {"seed_intervals": [{"base_seed": base, "end_seed": 301}]},
        }

    monkeypatch.setattr(adjudicator, "_replay_neutral", mismatched_neutral)
    with pytest.raises(adjudicator.DualAdjudicationError, match="seed cohort drift"):
        adjudicator.adjudicate(manifest)
    monkeypatch.setattr(
        adjudicator,
        "_replay_neutral",
        lambda path, *, candidate: {
            "verdict": "accept_h1", "pentanomial_sprt": {"llr": 1.0},
            "effective_search_config": {"n_full": 128},
            "fleet_merge": {"seed_intervals": [{"base_seed": 201, "end_seed": 301}]},
        },
    )

    def inconclusive_final(path: Path, *, candidate: Path, champion: Path) -> dict:
        report = fake_internal(path, candidate=candidate, champion=champion)
        if path.name == "tournament-final.json":
            report["verdict"] = "continue"
            report["pentanomial_sprt"]["llr"] = 10_000.0
        return report

    monkeypatch.setattr(adjudicator, "_replay_internal", inconclusive_final)
    no_winner = adjudicator.adjudicate(manifest)
    assert no_winner["passed"] is False
    assert no_winner["winner"] is None
    assert no_winner["decision"] == "no_promotion_inconclusive_or_vetoed"

    out = tmp_path / "result.json"
    adjudicator.write_result(out, result)
    monkeypatch.setattr(adjudicator, "adjudicate", lambda _path: result)
    assert adjudicator.verify_result(out) == result


def test_result_verifier_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    value = {
        "schema_version": adjudicator.RESULT_SCHEMA,
        "passed": True,
        "manifest": {"path": str(tmp_path / "manifest.json"), "sha256": "x"},
    }
    value["adjudication_sha256"] = adjudicator._digest(value)  # noqa: SLF001
    value["passed"] = False
    path.write_text(json.dumps(value))
    monkeypatch.setattr(adjudicator, "adjudicate", lambda _path: value)
    with pytest.raises(adjudicator.DualAdjudicationError, match="digest/status"):
        adjudicator.verify_result(path)


def test_stage_b_requires_winner_full_evidence_before_rendering_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path = _file(tmp_path, "selection.json")
    winner_receipt = _file(tmp_path, "winner.receipt.json")
    winner = {
        "arm_id": "n128", "subset_id": "full-140k",
        "training_receipt": adjudicator._ref(winner_receipt, where="fixture"),  # noqa: SLF001
    }
    monkeypatch.setattr(
        adjudicator,
        "verify_result",
        lambda _path: {"passed": True, "winner": winner},
    )
    files = {
        name: _file(tmp_path, name)
        for name in (
            "registry.jsonl", "CURRENT_CHAMPION", "contract.lock.json",
            "promotion.adjudication.json",
        )
    }
    manifest = {
        "schema_version": "a1-dual-arm-winner-promotion-manifest-v1",
        "registry": adjudicator._ref(files["registry.jsonl"], where="fixture"),  # noqa: SLF001
        "current_pointer": adjudicator._ref(files["CURRENT_CHAMPION"], where="fixture"),  # noqa: SLF001
        "contract_lock": adjudicator._ref(files["contract.lock.json"], where="fixture"),  # noqa: SLF001
        "adjudication": adjudicator._ref(files["promotion.adjudication.json"], where="fixture"),  # noqa: SLF001
        "training_receipt": adjudicator._ref(winner_receipt, where="fixture"),  # noqa: SLF001
        "receipt": str(tmp_path / "promotion.receipt.json"),
        "reason": "winner cleared full evidence",
    }
    manifest_path = tmp_path / "promotion.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    called = []
    monkeypatch.setattr(
        adjudicator.promotion,
        "prepare_promotion",
        lambda **kwargs: called.append(kwargs) or {"status": "dry_run"},
    )
    plan = adjudicator.build_promotion_plan(selection_path, manifest_path)
    assert called and plan["passed"] is True
    assert plan["command"][2] == "promote"
    assert "--go" not in plan["command"]
