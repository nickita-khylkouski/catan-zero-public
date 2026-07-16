from __future__ import annotations

import json
from pathlib import Path

from tools import run_self_play_ladder as ladder


PASSING_SCORE = ladder.EvalScore(
    random_win_rate=0.75,
    heuristic_win_rate=0.5,
    value_win_rate=0.5,
)
WEAKER_SCORE = ladder.EvalScore(
    random_win_rate=0.5,
    heuristic_win_rate=0.3,
    value_win_rate=0.3,
)


def _fake_training_writes_candidate(command: list[str]) -> None:
    checkpoint = Path(command[command.index("--checkpoint") + 1])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"candidate checkpoint")


def test_noncanonical_defaults_fail_before_creating_run_directory(
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = tmp_path / "must-not-exist"

    result = ladder.main(["--run-dir", str(run_dir)])

    assert result == 2
    assert not run_dir.exists()
    stderr = capsys.readouterr().err
    assert "--vps-to-win=3" in stderr
    assert "--promotion-eval-games=24" in stderr
    assert "--promotion-value-games=8" in stderr
    assert ladder.NO_CHAMPION_WRITE_FLAG in stderr
    assert ladder.NONCANONICAL_OVERWRITE_ACK_FLAG in stderr


def test_noncanonical_diagnostic_run_never_writes_champion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "diagnostic"
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"existing champion")
    monkeypatch.setattr(ladder, "run_command", _fake_training_writes_candidate)
    monkeypatch.setattr(
        ladder,
        "evaluate_candidate",
        lambda checkpoint, **kwargs: (
            WEAKER_SCORE if Path(checkpoint) == champion else PASSING_SCORE
        ),
    )

    result = ladder.main(
        [
            "--run-dir",
            str(run_dir),
            "--champion",
            str(champion),
            ladder.NO_CHAMPION_WRITE_FLAG,
        ]
    )

    assert result == 0
    assert champion.read_bytes() == b"existing champion"
    report = json.loads((run_dir / "ladder_report.json").read_text(encoding="utf-8"))
    assert report["champion_write_policy"]["enabled"] is False
    assert report["champion_write_policy"]["noncanonical_settings"]
    assert report["cycles"][0]["promotion_recommended"] is True
    assert report["cycles"][0]["promoted"] is False
    assert "champion write disabled" in report["cycles"][0]["promotion_reason"]


def test_canonical_gate_can_write_champion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "canonical"
    champion = tmp_path / "champion.pt"
    monkeypatch.setattr(ladder, "run_command", _fake_training_writes_candidate)
    monkeypatch.setattr(ladder, "evaluate_candidate", lambda *args, **kwargs: PASSING_SCORE)

    result = ladder.main(
        [
            "--run-dir",
            str(run_dir),
            "--champion",
            str(champion),
            "--vps-to-win",
            str(ladder.CANONICAL_VPS_TO_WIN),
            "--promotion-eval-games",
            str(ladder.MIN_PROMOTION_GAMES_PER_OPPONENT),
            "--promotion-value-games",
            str(ladder.MIN_PROMOTION_GAMES_PER_OPPONENT),
        ]
    )

    assert result == 0
    assert champion.read_bytes() == b"candidate checkpoint"
    report = json.loads((run_dir / "ladder_report.json").read_text(encoding="utf-8"))
    assert report["champion_write_policy"]["enabled"] is True
    assert report["champion_write_policy"]["noncanonical_settings"] == []
    assert report["cycles"][0]["promoted"] is True


def test_explicit_acknowledgement_preserves_legacy_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "legacy"
    champion = tmp_path / "champion.pt"
    monkeypatch.setattr(ladder, "run_command", _fake_training_writes_candidate)
    monkeypatch.setattr(ladder, "evaluate_candidate", lambda *args, **kwargs: PASSING_SCORE)

    result = ladder.main(
        [
            "--run-dir",
            str(run_dir),
            "--champion",
            str(champion),
            ladder.NONCANONICAL_OVERWRITE_ACK_FLAG,
        ]
    )

    assert result == 0
    assert champion.read_bytes() == b"candidate checkpoint"
    report = json.loads((run_dir / "ladder_report.json").read_text(encoding="utf-8"))
    assert report["champion_write_policy"]["enabled"] is True
    assert report["champion_write_policy"]["noncanonical_settings"]
    assert report["champion_write_policy"]["explicit_noncanonical_overwrite"] is True
