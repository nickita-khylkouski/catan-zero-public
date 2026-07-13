from __future__ import annotations

import json
import shutil
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


def _handoff_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_s1: dict,
    *,
    search_overrides: dict | None = None,
) -> tuple[Path, dict]:
    checkpoint = tmp_path / "producer.pt"
    checkpoint.write_bytes(b"promoted-f7")
    search_config = dict(legacy_s1["selected_fields"])
    search_config["c_scale"] = 0.1
    search_config.update(search_overrides or {})
    identity = {
        "schema_version": "a1-deployed-agent-search-config-v1",
        "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
        "search_config": search_config,
    }
    identity["agent_identity_sha256"] = binding._digest_value(identity)
    payload = {
        "schema_version": binding.promotion_handoff.HANDOFF_SCHEMA,
        "promotion_receipt": {"path": str(tmp_path / "promotion.json")},
        "registry_after": {"checkpoint": dict(identity["checkpoint"])},
        "producer_identity": identity,
    }
    payload["handoff_sha256"] = binding._digest_value(payload)
    path = tmp_path / "post-promotion-handoff.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        binding.promotion_handoff,
        "build_handoff",
        lambda _receipt: payload,
    )
    return path, payload


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


def test_binding_replay_uses_payload_bound_emitter_path_even_when_bytes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s1_path, _decision = _s1_fixture(tmp_path, monkeypatch)
    historical_emitter = tmp_path / "frozen/tools/search_operator_binding.py"
    historical_emitter.parent.mkdir(parents=True)
    shutil.copy2(Path(binding.__file__), historical_emitter)
    assert _sha(historical_emitter) == _sha(Path(binding.__file__))

    bound, _ = binding.build_bindings(
        s1_path,
        s2_output_path=tmp_path / "s2.json",
        binding_time_utc="2026-07-10T04:10:00Z",
        emitter_path=historical_emitter,
    )
    replayed, _ = binding.build_bindings(
        s1_path,
        s2_output_path=tmp_path / "s2.json",
        binding_time_utc="2026-07-10T04:10:00Z",
        emitter_path=Path(bound["emitter"]["path"]),
    )
    current_path, _ = binding.build_bindings(
        s1_path,
        s2_output_path=tmp_path / "s2.json",
        binding_time_utc="2026-07-10T04:10:00Z",
    )

    assert replayed == bound
    assert current_path != bound
    assert current_path["emitter"]["sha256"] == bound["emitter"]["sha256"]


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


def test_post_promotion_s1_projects_only_deployed_c_scale_and_lines_s2_s3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_path, legacy = _s1_fixture(tmp_path, monkeypatch)
    handoff_path, handoff = _handoff_fixture(tmp_path, monkeypatch, legacy)
    s1_path = tmp_path / "s1.post-promotion.binding.json"
    s2_path = tmp_path / "s2.binding.json"
    s3_path = tmp_path / "s3.binding.json"
    archived_emitter = tmp_path / "frozen/tools/search_operator_binding.py"
    archived_emitter.parent.mkdir(parents=True)
    shutil.copy2(Path(binding.__file__), archived_emitter)

    s1, s2, s3 = binding.write_post_promotion_bindings(
        legacy_path,
        handoff_path,
        s1_path,
        s2_path,
        s3_path,
        binding_time_utc="2026-07-13T20:00:00Z",
        emitter_path=archived_emitter,
    )

    assert s1["schema_version"] == binding.POST_PROMOTION_S1_SCHEMA
    assert s1["artifact_kind"] == binding.ARTIFACT_KIND
    assert s1["statement"] == binding.POST_PROMOTION_S1_STATEMENT
    assert s1["continuity_override"] == binding.POST_PROMOTION_S1_OVERRIDE
    assert s1["selected_fields"] == {
        **legacy["selected_fields"],
        "c_scale": 0.1,
    }
    assert s1["source_legacy_s1"] == {
        "path": str(legacy_path),
        "sha256": _sha(legacy_path),
    }
    assert s1["source_post_promotion_handoff"] == {
        "path": str(handoff_path),
        "sha256": _sha(handoff_path),
        "handoff_sha256": handoff["handoff_sha256"],
    }
    assert s1["producer_checkpoint"] == handoff["producer_identity"]["checkpoint"]
    assert s1["producer_identity_sha256"] == handoff["producer_identity"][
        "agent_identity_sha256"
    ]
    assert s1["emitter"] == s2["emitter"] == s3["emitter"] == {
        "path": str(archived_emitter),
        "sha256": _sha(archived_emitter),
    }
    assert s2["source_s1"] == {"path": str(s1_path), "sha256": _sha(s1_path)}
    assert s3["source_s1"] == s2["source_s1"]
    assert s3["source_s2_binding"] == {
        "path": str(s2_path),
        "sha256": _sha(s2_path),
    }
    for path, payload in ((s1_path, s1), (s2_path, s2), (s3_path, s3)):
        assert json.loads(path.read_text()) == payload
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        _assert_self_digest(payload)
    assert binding._replay_post_promotion_s1(s1_path) == s1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"c_scale": 0.3}, "only deployed c_scale=.10"),
        ({"sigma_eval": 0.77}, "non-continuity S1 field sigma_eval"),
        (
            {"symmetry_averaged_eval_threshold": 24},
            "non-continuity S1 field symmetry_averaged_eval_threshold",
        ),
    ],
)
def test_post_promotion_s1_rejects_any_nonexact_handoff_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict,
    message: str,
) -> None:
    legacy_path, legacy = _s1_fixture(tmp_path, monkeypatch)
    handoff_path, _ = _handoff_fixture(
        tmp_path, monkeypatch, legacy, search_overrides=overrides
    )
    with pytest.raises(binding.BindingError, match=message):
        binding.build_post_promotion_s1_binding(legacy_path, handoff_path)


def test_post_promotion_s1_rejects_legacy_noncontrol_and_checkpoint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_path, legacy = _s1_fixture(tmp_path, monkeypatch)
    legacy["selected_fields"]["c_scale"] = 0.1
    legacy["selected_fields_sha256"] = binding._digest_value(legacy["selected_fields"])
    legacy_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    monkeypatch.setattr(binding.search_adjudicator, "adjudicate", lambda _: legacy)
    handoff_path, _ = _handoff_fixture(tmp_path, monkeypatch, legacy)
    with pytest.raises(binding.BindingError, match="legacy c_scale=.03"):
        binding.build_post_promotion_s1_binding(legacy_path, handoff_path)

    legacy["selected_fields"]["c_scale"] = 0.03
    legacy["selected_fields_sha256"] = binding._digest_value(legacy["selected_fields"])
    legacy_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    monkeypatch.setattr(binding.search_adjudicator, "adjudicate", lambda _: legacy)
    handoff_path, handoff = _handoff_fixture(tmp_path, monkeypatch, legacy)
    Path(handoff["producer_identity"]["checkpoint"]["path"]).write_bytes(b"drift")
    with pytest.raises(binding.BindingError, match="checkpoint hash drift"):
        binding.build_post_promotion_s1_binding(legacy_path, handoff_path)


def test_post_promotion_s1_rejects_tampered_binding_and_cleans_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_path, legacy = _s1_fixture(tmp_path, monkeypatch)
    handoff_path, _ = _handoff_fixture(tmp_path, monkeypatch, legacy)
    s1 = binding.build_post_promotion_s1_binding(
        legacy_path,
        handoff_path,
        binding_time_utc="2026-07-13T20:00:00Z",
    )
    s1["continuity_override"]["deployed_value"] = 0.3
    s1_path = tmp_path / "tampered-s1.json"
    s1_path.write_text(json.dumps(s1) + "\n", encoding="utf-8")
    with pytest.raises(binding.BindingError, match="does not equal semantic replay"):
        binding._replay_post_promotion_s1(s1_path)

    occupied = tmp_path / "occupied-s3.json"
    occupied.write_text("occupied\n", encoding="utf-8")
    fresh_s1 = tmp_path / "fresh-s1.json"
    fresh_s2 = tmp_path / "fresh-s2.json"
    with pytest.raises(binding.BindingError, match="refusing to overwrite"):
        binding.write_post_promotion_bindings(
            legacy_path,
            handoff_path,
            fresh_s1,
            fresh_s2,
            occupied,
        )
    assert not fresh_s1.exists()
    assert not fresh_s2.exists()
    assert occupied.read_text() == "occupied\n"
