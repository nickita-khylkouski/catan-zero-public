from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from tools import search_operator_binding as binding


def _sha(path: Path) -> str:
    return binding._sha256(path)


def _s1_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict]:
    manifest = tmp_path / "s1.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": binding.search_adjudicator.MANIFEST_SCHEMA,
                "stage": "s1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selected = {
        "c_scale": 0.03,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
    }
    decision = {
        "schema_version": binding.search_adjudicator.DECISION_SCHEMA,
        "stage": "s1",
        "passed": True,
        "decision": "hold",
        "selected_fields": selected,
        "selected_fields_sha256": binding._digest_value(selected),
        "source_artifacts": [{"path": str(manifest), "sha256": _sha(manifest)}],
    }
    path = tmp_path / "s1.decision.json"
    path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        binding.search_adjudicator,
        "adjudicate",
        lambda candidate: decision if Path(candidate) == manifest else None,
    )
    return path, decision


def _assert_self_digest(payload: dict) -> None:
    unhashed = dict(payload)
    declared = unhashed.pop("artifact_content_sha256")
    assert declared == binding._digest_value(unhashed)


def test_write_bindings_are_read_only_self_digested_and_lineaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s1_path, s1 = _s1_fixture(tmp_path, monkeypatch)
    s2_path = tmp_path / "s2.binding.json"
    s3_path = tmp_path / "s3.binding.json"

    s2, s3 = binding.write_bindings(
        s1_path,
        s2_path,
        s3_path,
        binding_time_utc="2026-07-10T04:10:00Z",
    )

    assert json.loads(s2_path.read_text()) == s2
    assert json.loads(s3_path.read_text()) == s3
    assert stat.S_IMODE(s2_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(s3_path.stat().st_mode) == 0o444
    _assert_self_digest(s2)
    _assert_self_digest(s3)
    assert s2["artifact_kind"] == binding.ARTIFACT_KIND
    assert s2["statement"] == binding.STATEMENT
    assert s2["source_s1"] == {"path": str(s1_path), "sha256": _sha(s1_path)}
    assert s2["source_s1_selected_fields_sha256"] == s1[
        "selected_fields_sha256"
    ]
    assert s2["selected_fields"] == binding.S2_SELECTED
    assert s2["operator"] == binding.S2_OPERATOR
    assert s3["selected_fields"] == binding.S3_SELECTED
    assert s3["operator"] == binding.S3_OPERATOR
    assert s3["source_s2_binding"] == {
        "path": str(s2_path),
        "sha256": _sha(s2_path),
    }


def test_write_bindings_refuses_overwrite_and_cleans_partial_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s1_path, _ = _s1_fixture(tmp_path, monkeypatch)
    s2_path = tmp_path / "s2.binding.json"
    s3_path = tmp_path / "s3.binding.json"
    s3_path.write_text("occupied\n", encoding="utf-8")

    with pytest.raises(binding.BindingError, match="refusing to overwrite"):
        binding.write_bindings(s1_path, s2_path, s3_path)
    assert not s2_path.exists()
    assert s3_path.read_text() == "occupied\n"


def test_binding_rejects_unreplayable_or_drifted_s1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s1_path, decision = _s1_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        binding.search_adjudicator,
        "adjudicate",
        lambda _manifest: {**decision, "decision": "adopt"},
    )
    with pytest.raises(binding.BindingError, match="semantic replay"):
        binding.build_bindings(s1_path, s2_output_path=tmp_path / "s2.json")

    # Restore replay but mutate the manifest bytes after the S1 reference was
    # emitted. The bridge must reject that source before creating outputs.
    monkeypatch.setattr(binding.search_adjudicator, "adjudicate", lambda _: decision)
    manifest = tmp_path / "s1.manifest.json"
    manifest.write_text('{"drift":true}\n', encoding="utf-8")
    with pytest.raises(binding.BindingError, match="hash drift"):
        binding.build_bindings(s1_path, s2_output_path=tmp_path / "s2.json")


def test_binding_replays_with_the_historically_bound_adjudicator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s1_path, decision = _s1_fixture(tmp_path, monkeypatch)
    adjudicator = tmp_path / "historical_adjudicator.py"
    adjudicator.write_text("# immutable historical adjudicator\n", encoding="utf-8")
    decision["adjudicator"] = {"path": str(adjudicator), "sha256": _sha(adjudicator)}
    decision["source_artifacts"].append(
        {"path": str(adjudicator), "sha256": _sha(adjudicator)}
    )
    s1_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        binding.runpy,
        "run_path",
        lambda candidate: {
            "DECISION_SCHEMA": binding.search_adjudicator.DECISION_SCHEMA,
            "MANIFEST_SCHEMA": binding.search_adjudicator.MANIFEST_SCHEMA,
            "AdjudicationError": binding.search_adjudicator.AdjudicationError,
            "adjudicate": lambda _manifest: decision,
        }
        if Path(candidate) == adjudicator
        else {},
    )
    monkeypatch.setattr(
        binding.search_adjudicator,
        "adjudicate",
        lambda _manifest: (_ for _ in ()).throw(AssertionError("used current adjudicator")),
    )

    s2, _s3 = binding.build_bindings(
        s1_path, s2_output_path=tmp_path / "s2.json"
    )
    assert s2["selected_fields"] == binding.S2_SELECTED


@pytest.mark.parametrize(
    "timestamp",
    ["2026-07-10T04:10:00", "2026-07-10T04:10:00-07:00", "not-a-time"],
)
def test_binding_requires_explicit_utc_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timestamp: str
) -> None:
    s1_path, _ = _s1_fixture(tmp_path, monkeypatch)
    with pytest.raises(binding.BindingError, match="UTC|ISO-8601"):
        binding.build_bindings(
            s1_path,
            s2_output_path=tmp_path / "s2.json",
            binding_time_utc=timestamp,
        )
