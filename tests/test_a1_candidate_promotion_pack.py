from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools import a1_candidate_promotion_pack as pack


def _file(path: Path, content: bytes = b"x") -> Path:
    path.write_bytes(content)
    return path


def test_prior_cohorts_are_required_and_typed(tmp_path: Path) -> None:
    report = _file(tmp_path / "screen.json")
    assert pack._parse_cohorts([f"dose:internal_h2h={report}"]) == [
        ("dose", "internal_h2h", report.resolve())
    ]
    with pytest.raises(pack.PackError, match="at least one"):
        pack._parse_cohorts([])
    with pytest.raises(pack.PackError, match="LABEL:KIND"):
        pack._parse_cohorts([f"dose={report}"])


def test_build_pack_publishes_one_candidate_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = (
        "lock",
        "training_receipt",
        "training_report",
        "registry",
        "current_pointer",
        "candidate",
        "champion",
        "candidate_calibration",
        "champion_calibration",
        "internal_h2h",
        "candidate_panel",
        "champion_panel",
        "high_regret_report",
        "prior",
    )
    paths = {name: _file(tmp_path / f"{name}.json") for name in names}
    contract = {"contract_sha256": "sha256:" + "1" * 64}
    monkeypatch.setattr(pack.promotion, "_verify_contract", lambda _path: contract)
    monkeypatch.setattr(pack.ChampionRegistry, "load", lambda _path: object())
    monkeypatch.setattr(
        pack.artifacts,
        "build_high_regret_source",
        lambda **_kwargs: {"kind": "high"},
    )
    monkeypatch.setattr(
        pack.artifacts,
        "build_bucket_game_report",
        lambda **_kwargs: {"kind": "bucket-report"},
    )
    monkeypatch.setattr(
        pack.artifacts,
        "build_bucket_veto_source",
        lambda **_kwargs: {"kind": "bucket-veto"},
    )
    monkeypatch.setattr(
        pack.artifacts,
        "build_evidence_envelope",
        lambda *, kind, **_kwargs: {"kind": kind},
    )
    monkeypatch.setattr(pack, "_validate_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pack.artifacts,
        "build_cohort_exclusions",
        lambda **_kwargs: {"kind": "exclusions"},
    )
    monkeypatch.setattr(
        pack.artifacts,
        "build_adjudication",
        lambda **_kwargs: {"kind": "adjudication"},
    )
    monkeypatch.setattr(
        pack.promotion,
        "_verify_adjudication",
        lambda *_args, **_kwargs: {
            "candidate": {"sha256": pack.promotion._sha256(paths["candidate"])},
            "final_cohort_intervals": [],
        },
    )
    monkeypatch.setattr(
        pack.promotion, "_verify_cohort_exclusions", lambda *_args, **_kwargs: None
    )
    args = argparse.Namespace(
        **{
            name: paths[name]
            for name in names
            if name not in {"lock", "prior"}
        },
        contract_lock=paths["lock"],
        candidate_version=8,
        champion_version=7,
        prior_cohort=[f"dose:internal_h2h={paths['prior']}"],
        nth_confirmation=None,
        out=tmp_path / "pack" / "adjudication.json",
        cohort_exclusions_out=tmp_path / "pack" / "exclusions.json",
        receipt=tmp_path / "pack" / "receipt.json",
    )

    result = pack.build_pack(args)

    assert set(result["evidence"]) == pack.promotion.REQUIRED_EVIDENCE_KINDS
    receipt_path = Path(result["receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    assert unsigned.pop("receipt_sha256") == pack.promotion._digest_value(unsigned)
    assert receipt["candidate"]["sha256"] == pack.promotion._sha256(
        paths["candidate"]
    )
    assert receipt_path.stat().st_mode & 0o222 == 0


def test_pack_failure_removes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _file(tmp_path / "source")
    args = argparse.Namespace(
        contract_lock=source,
        training_receipt=source,
        training_report=source,
        registry=source,
        current_pointer=source,
        candidate=source,
        candidate_version=1,
        champion=source,
        champion_version=0,
        candidate_calibration=source,
        champion_calibration=source,
        internal_h2h=source,
        candidate_panel=source,
        champion_panel=source,
        high_regret_report=source,
        prior_cohort=[f"dose:internal_h2h={source}"],
        nth_confirmation=None,
        out=tmp_path / "failed-pack" / "adjudication.json",
        cohort_exclusions_out=tmp_path / "failed-pack" / "exclusions.json",
        receipt=tmp_path / "failed-pack" / "receipt.json",
    )
    monkeypatch.setattr(
        pack.promotion,
        "_verify_contract",
        lambda _path: {"contract_sha256": "sha256:" + "1" * 64},
    )
    monkeypatch.setattr(pack.ChampionRegistry, "load", lambda _path: object())
    monkeypatch.setattr(
        pack.artifacts,
        "build_high_regret_source",
        lambda **_kwargs: (_ for _ in ()).throw(pack.PackError("bad report")),
    )

    with pytest.raises(pack.PackError, match="bad report"):
        pack.build_pack(args)
    assert not args.out.parent.exists()
