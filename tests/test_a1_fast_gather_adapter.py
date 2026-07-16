from __future__ import annotations

import json
from pathlib import Path

from tools import a1_fast_gather_adapter as adapter


def test_historical_gather_adapter_renders_explicit_legacy_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "command": [
                    "/python",
                    "/repo/tools/train_bc.py",
                    "--init-checkpoint",
                    "/old-parent.pt",
                    "--data",
                    "/old-data",
                    "--validation-game-sentinel-manifest",
                    "/old-sentinel.json",
                    "--checkpoint",
                    "/old-candidate.pt",
                    "--report",
                    "/old-report.json",
                    "--soft-target-weight",
                    "0.9",
                    "--max-steps",
                    "128",
                ]
            }
        ),
        encoding="utf-8",
    )

    command = adapter._derive_command(
        source_manifest=source,
        trainer=tmp_path / "tools/train_bc.py",
        gather_init=tmp_path / "gather-init.pt",
        data=tmp_path / "data",
        validation_sentinel=tmp_path / "sentinel.json",
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
        soft_target_weight=0.9,
        policy_target_blend_semantics="legacy_interpolate_v1",
        adapter="gather",
        policy_aux_active_batch_size=0,
    )

    assert adapter._option(
        command, "--policy-target-blend-semantics"
    ) == "legacy_interpolate_v1"
