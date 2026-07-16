from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools import convert_teacher_to_entity_tokens as converter
from tools import report_teacher_data_quality as quality


_REQUIRED_SOURCE_FILES = (
    "src/catan_zero/rl/self_play.py",
    "src/catan_zero/rl/action_features.py",
    "src/catan_zero/rl/xdim_lite_policy.py",
)


def test_entity_conversion_preserves_production_gate_lineage(tmp_path: Path) -> None:
    source = tmp_path / "curated"
    source.mkdir()
    converter_hashes = converter._tool_provenance()["file_sha256"]
    source_hashes = {
        path: converter_hashes.get(path, f"digest-{index}")
        for index, path in enumerate(_REQUIRED_SOURCE_FILES, start=1)
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "track": "2p_no_trade",
                "vps_to_win": 10,
                "mixed_seats": True,
                "mixed_seat_mode": "random",
                "graph_history_features": True,
                "tool_provenance": {"file_sha256": source_hashes},
            }
        ),
        encoding="utf-8",
    )

    converted = tmp_path / "entity"
    converted.mkdir()
    (converted / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": [str(source)],
                "input_manifests": converter._input_manifests([str(source)]),
                "track": "2p_no_trade",
                "vps_to_win": 10,
                "graph_history_features": True,
                "tool_provenance": converter._tool_provenance(),
            }
        ),
        encoding="utf-8",
    )

    metadata = quality._input_metadata(converted)
    assert metadata["mixed_seats"] == [True]
    assert metadata["mixed_seat_modes"] == ["random"]
    assert metadata["graph_history_features"] == [True]
    for path, digest in source_hashes.items():
        assert metadata["source_provenance_hashes"][path] == [digest]

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
    assert failures == []
