from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import convert_teacher_to_entity_tokens as converter
from tools import curate_teacher_data as curator
from tools import report_teacher_data_quality as quality
from tools.factory_common import classical_teacher_hard_action_target_information


def _source_manifest(path: Path) -> Path:
    path.mkdir()
    manifest = path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "track": "2p_no_trade",
                "vps_to_win": 10,
                "mixed_seats": True,
                "mixed_seat_mode": "random",
                "graph_history_features": True,
                "hard_action_target_information": (
                    classical_teacher_hard_action_target_information()
                ),
                "tool_provenance": curator._tool_provenance(),
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _converted_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "curated"
    source_manifest = _source_manifest(source)
    converted = tmp_path / "entity"
    converted.mkdir()
    (converted / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": [str(source)],
                "input_manifests": converter._input_manifests([str(source)]),
                "hard_action_target_information": (
                    converter.propagated_hard_action_target_information(
                        converter._input_manifests([str(source)])
                    )
                ),
                "track": "2p_no_trade",
                "vps_to_win": 10,
                "graph_history_features": True,
                "tool_provenance": converter._tool_provenance(),
            }
        ),
        encoding="utf-8",
    )
    return converted, source_manifest

def _production_failures(data: Path) -> tuple[dict[str, object], list[str]]:
    metadata = quality._input_metadata(data)
    failures: list[str] = []
    quality._check_manifest_metadata(
        failures,
        SimpleNamespace(
            track="2p_no_trade",
            vps_to_win=10,
            production_35m_teacher=True,
        ),
        metadata,
    )
    return metadata, failures


def test_entity_conversion_preserves_authenticated_production_gate_lineage(
    tmp_path: Path,
) -> None:
    converted, source_manifest = _converted_manifest(tmp_path)
    converted_payload = json.loads((converted / "manifest.json").read_text())
    binding = converted_payload["input_manifests"][0]["source_manifest"]
    assert converted_payload["hard_action_target_information"] == (
        classical_teacher_hard_action_target_information()
    )

    assert binding == {
        "schema_version": curator.SOURCE_MANIFEST_BINDING_SCHEMA,
        "path": str(source_manifest.resolve()),
        "file_sha256": curator._sha256_bytes(source_manifest.read_bytes()),
    }
    assert binding["file_sha256"].startswith("sha256:")
    assert all(
        digest.startswith("sha256:")
        for digest in converter._tool_provenance()["file_sha256"].values()
    )

    metadata, failures = _production_failures(converted)
    assert metadata["mixed_seats"] == [True]
    assert metadata["mixed_seat_modes"] == ["random"]
    assert metadata["graph_history_features"] == [True]
    assert metadata["source_provenance_errors"] == []
    for path in quality.REQUIRED_SOURCE_FEATURE_FILES:
        assert metadata["source_provenance_hashes"][path] == [
            curator._tool_provenance()["file_sha256"][path]
        ]
    assert failures == []


def test_quality_gate_rejects_tampered_bound_source_manifest(tmp_path: Path) -> None:
    converted, source_manifest = _converted_manifest(tmp_path)
    source_manifest.write_text(
        source_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    metadata, failures = _production_failures(converted)

    assert any(
        "byte hash mismatch" in error
        for error in metadata["source_provenance_errors"]
    )
    assert any("source provenance authentication failed" in failure for failure in failures)


def test_quality_gate_rejects_missing_bound_source_manifest(tmp_path: Path) -> None:
    converted, source_manifest = _converted_manifest(tmp_path)
    source_manifest.unlink()

    metadata, failures = _production_failures(converted)

    assert any(
        "source manifest is unreadable" in error
        for error in metadata["source_provenance_errors"]
    )
    assert any("source provenance authentication failed" in failure for failure in failures)


def test_quality_gate_rejects_placeholder_or_recursively_hidden_hashes(
    tmp_path: Path,
) -> None:
    data = tmp_path / "teacher"
    data.mkdir()
    hidden = {
        path: f"digest-{index}"
        for index, path in enumerate(
            sorted(quality.REQUIRED_SOURCE_FEATURE_FILES),
            start=1,
        )
    }
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "track": "2p_no_trade",
                "vps_to_win": 10,
                "mixed_seats": True,
                "mixed_seat_mode": "random",
                "graph_history_features": True,
                "tool_provenance": {
                    "schema_version": curator.TOOL_PROVENANCE_SCHEMA,
                    "file_sha256": hidden,
                    "feature_semantics_files": sorted(hidden),
                },
                "untrusted_nested_object": {"file_sha256": hidden},
            }
        ),
        encoding="utf-8",
    )

    metadata, failures = _production_failures(data)

    assert metadata["source_provenance_hashes"] == {}
    assert any(
        "invalid sha256 entries" in error
        for error in metadata["source_provenance_errors"]
    )
    assert any("source provenance authentication failed" in failure for failure in failures)


def test_required_provenance_files_fail_closed_when_missing(tmp_path: Path) -> None:
    present = tmp_path / "present.py"
    present.write_text("pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="required provenance file"):
        curator._hash_required_files(
            tmp_path,
            ["present.py", "missing.py"],
        )


def test_input_manifest_binding_fails_closed_on_invalid_json(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(SystemExit, match="cannot authenticate source manifest"):
        converter._input_manifests([str(source)])
