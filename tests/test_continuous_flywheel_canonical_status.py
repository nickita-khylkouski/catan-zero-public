from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.champion_registry import ChampionRegistry


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import continuous_flywheel  # noqa: E402


def test_real_execution_fails_closed_as_noncanonical(tmp_path, capsys) -> None:
    loop_dir = tmp_path / "must-not-be-created"
    checkpoint = tmp_path / "unused.pt"

    result = continuous_flywheel.main(
        ["--loop-dir", str(loop_dir), "--seed-checkpoint", str(checkpoint)]
    )

    assert result == 2
    assert not loop_dir.exists()
    stderr = capsys.readouterr().err
    assert "permanently retired" in stderr
    assert "tools/loop.py" in stderr
    assert "a1_production_executor.py" in stderr
    assert "a1_one_dose_train.py" in stderr
    assert "a1_promotion_transaction.py" in stderr


def test_noncanonical_real_execution_escape_is_removed() -> None:
    with pytest.raises(SystemExit):
        continuous_flywheel.build_parser().parse_args(
            [
                "--loop-dir",
                "/tmp/experimental-loop",
                "--seed-checkpoint",
                "/tmp/seed.pt",
                "--allow-noncanonical-experimental-loop",
            ]
        )


def test_dry_run_is_plan_only_and_preserves_external_authority_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "seed.pt"
    checkpoint.write_bytes(b"dry-run fixture")
    incumbent = tmp_path / "incumbent.pt"
    incumbent.write_bytes(b"incumbent-v7")
    registry_path = tmp_path / "champion_registry.json"
    registry = ChampionRegistry(registry_path)
    registry.set_role("generator_champion", incumbent, version=7, reason="fixture")
    registry.save()
    current_pointer = tmp_path / "CURRENT_CHAMPION"
    current_pointer.write_text(str(incumbent), encoding="utf-8")
    registry_before = registry_path.read_bytes()
    current_before = current_pointer.read_bytes()
    loop_dir = tmp_path / "must-not-be-created"

    result = continuous_flywheel.main(
        [
            "--loop-dir",
            str(loop_dir),
            "--seed-checkpoint",
            str(checkpoint),
            "--champion-registry",
            str(registry_path),
            "--dry-run",
            "--max-rounds", "2",
        ]
    )

    assert result == 0
    assert not loop_dir.exists()
    assert registry_path.read_bytes() == registry_before
    assert current_pointer.read_bytes() == current_before
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema_version"] == continuous_flywheel.RETIRED_PLAN_SCHEMA
    assert plan["status"] == "plan_only"
    assert plan["side_effects_permitted"] is False
