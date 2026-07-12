from __future__ import annotations

import sys
from pathlib import Path


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
    assert "noncanonical experimental prototype" in stderr
    assert "a1_production_executor.py" in stderr
    assert "a1_one_dose_train.py" in stderr
    assert "a1_promotion_transaction.py" in stderr
    assert continuous_flywheel.NONCANONICAL_ACK_FLAG in stderr


def test_explicit_noncanonical_acknowledgement_is_parseable() -> None:
    args = continuous_flywheel.build_parser().parse_args(
        [
            "--loop-dir",
            "/tmp/experimental-loop",
            "--seed-checkpoint",
            "/tmp/seed.pt",
            continuous_flywheel.NONCANONICAL_ACK_FLAG,
        ]
    )

    assert args.allow_noncanonical_experimental_loop is True


def test_dry_run_does_not_require_noncanonical_acknowledgement(tmp_path) -> None:
    checkpoint = tmp_path / "seed.pt"
    checkpoint.write_bytes(b"dry-run fixture")

    result = continuous_flywheel.main(
        [
            "--loop-dir",
            str(tmp_path / "loop"),
            "--seed-checkpoint",
            str(checkpoint),
            "--dry-run",
            "--max-rounds",
            "0",
        ]
    )

    assert result == 0
