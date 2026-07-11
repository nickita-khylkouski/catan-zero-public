"""Fail-closed production wiring for the opt-in native MCTS path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import generate_gumbel_selfplay_data as generator  # type: ignore  # noqa: E402
from catan_zero.rl.pipeline_configs import GenerateConfig  # noqa: E402


def test_native_generation_flags_are_explicit_and_default_off(tmp_path: Path) -> None:
    parser = generator.build_parser()
    defaults = parser.parse_args(["--out-dir", str(tmp_path / "default")])
    native = parser.parse_args(
        [
            "--out-dir",
            str(tmp_path / "native"),
            "--native-mcts-hot-loop",
            "--evaluator-rust-featurize",
        ]
    )

    assert defaults.native_mcts_hot_loop is False
    assert defaults.rust_featurize is False
    assert native.native_mcts_hot_loop is True
    # The canonical name and historical --rust-featurize alias intentionally
    # share one config/provenance destination.
    assert native.rust_featurize is True


def test_native_generation_flags_change_config_hash(tmp_path: Path) -> None:
    parser = generator.build_parser()
    reference = GenerateConfig.from_namespace(
        parser.parse_args(["--out-dir", str(tmp_path / "reference")])
    )
    native = GenerateConfig.from_namespace(
        parser.parse_args(
            [
                "--out-dir",
                str(tmp_path / "native"),
                "--native-mcts-hot-loop",
                "--evaluator-rust-featurize",
            ]
        )
    )

    assert native.native_mcts_hot_loop is True
    assert native.rust_featurize is True
    assert native.config_hash() != reference.config_hash()


def test_native_search_absence_refuses_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(generator, "native_hot_loop_available", lambda: False)
    output = tmp_path / "must-not-exist"

    with pytest.raises(SystemExit) as raised:
        generator.main(
            [
                "--skip-guards",
                "--out-dir",
                str(output),
                "--games",
                "0",
                "--native-mcts-hot-loop",
            ]
        )

    assert raised.value.code == 2
    assert "refusing silent Python fallback" in capsys.readouterr().err
    assert not output.exists()


def test_native_featurizer_absence_refuses_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable() -> None:
        raise RuntimeError("wheel omitted build_entity_features_flat")

    monkeypatch.setattr(generator, "require_rust_feature_path", unavailable)
    output = tmp_path / "must-not-exist"

    with pytest.raises(SystemExit) as raised:
        generator.main(
            [
                "--skip-guards",
                "--out-dir",
                str(output),
                "--games",
                "0",
                "--evaluator-rust-featurize",
            ]
        )

    assert raised.value.code == 2
    assert "omitted build_entity_features_flat" in capsys.readouterr().err
    assert not output.exists()


def test_generation_manifest_records_both_native_choices(tmp_path: Path) -> None:
    args = generator.build_parser().parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--native-mcts-hot-loop",
            "--evaluator-rust-featurize",
        ]
    )
    summary = generator._merge_worker_summaries(
        [], out_dir=tmp_path, elapsed_sec=1.0, args=args
    )

    assert summary["native_mcts_hot_loop"] is True
    assert summary["rust_featurize"] is True
