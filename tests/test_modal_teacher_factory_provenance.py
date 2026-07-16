from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.factory_common import classical_teacher_hard_action_target_information


_REPO = Path(__file__).resolve().parents[1]
_FACTORY_PATH = _REPO / "tools" / "modal_teacher_factory.py"


class _ModalImage:
    @classmethod
    def debian_slim(cls, **_kwargs):
        return cls()

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: self


class _ModalVolume:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()

    def commit(self) -> None:
        return None

    def reload(self) -> None:
        return None


class _ModalApp:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def function(self, *_args, **_kwargs):
        return lambda function: function

    def local_entrypoint(self, *_args, **_kwargs):
        return lambda function: function


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch):
    modal_stub = SimpleNamespace(
        App=_ModalApp,
        Image=_ModalImage,
        Volume=_ModalVolume,
    )
    monkeypatch.setitem(sys.modules, "modal", modal_stub)
    spec = importlib.util.spec_from_file_location(
        "_modal_teacher_factory_provenance_test",
        _FACTORY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _part_report(factory) -> dict:
    return factory._part_report(
        run_name="teacher-test",
        run_id="run-1",
        part_index=0,
        payload={
            "track": "2p_no_trade",
            "vps_to_win": 10,
            "cpu_workers": 1,
            "fmt": "npz_zst",
        },
        teachers=["test_teacher"],
        games=1,
        completed_games=1,
        wins=1,
        decisions=1,
        elapsed=1.0,
        teacher_counts=Counter({"test_teacher": 1}),
        phase_counts=Counter({"PLAY_TURN": 1}),
        score_source_counts=Counter({"none": 1}),
        forced_actions=0,
        invalid_labels=0,
        soft_policy_rows=0,
        soft_score_rows=0,
        final_public_vp_rows=1,
        final_actual_vp_rows=1,
        outcome_rows=1,
        clean_terminal_outcome_rows=1,
        truncated_rows=0,
        legal_counts=[2],
        shards=[Path("teacher_shard_00000.npz.zst")],
        complete=True,
    )


def test_part_report_matches_local_generator_target_provenance(factory) -> None:
    report = _part_report(factory)

    assert report["hard_action_target_information"] == (
        classical_teacher_hard_action_target_information()
    )


@pytest.mark.parametrize(
    "contract",
    [
        None,
        {},
        {
            **classical_teacher_hard_action_target_information(),
            "public_information_authenticated": True,
        },
    ],
)
def test_generated_part_reuse_rejects_missing_or_drifted_provenance(
    factory,
    contract: dict | None,
) -> None:
    manifest = {}
    if contract is not None:
        manifest["hard_action_target_information"] = contract

    with pytest.raises(RuntimeError, match="refusing to reuse or summarize"):
        factory._validated_generated_part_manifest(
            manifest,
            source=Path("manifest.json"),
        )


def test_generated_part_reuse_accepts_exact_authoritative_provenance(factory) -> None:
    manifest = {
        "hard_action_target_information": (
            classical_teacher_hard_action_target_information()
        )
    }

    assert (
        factory._validated_generated_part_manifest(
            manifest,
            source=Path("manifest.json"),
        )
        is manifest
    )


@pytest.mark.parametrize("authenticated", [False, True])
def test_worker_resume_fails_closed_on_legacy_manifest(
    factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authenticated: bool,
) -> None:
    monkeypatch.setattr(factory, "VOLUME_ROOT", tmp_path)
    part_dir = tmp_path / "teacher-test" / "parts" / "part_00000"
    part_dir.mkdir(parents=True)
    manifest = {"run_id": "run-1"}
    if authenticated:
        manifest["hard_action_target_information"] = (
            classical_teacher_hard_action_target_information()
        )
    (part_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    payload = {
        "run_name": "teacher-test",
        "run_id": "run-1",
        "part_index": 0,
        "games": 1,
        "seed": 1,
        "cpu_workers": 1,
        "teachers": "test_teacher",
        "resume": True,
    }

    if authenticated:
        assert factory._run_worker(payload) == manifest
    else:
        with pytest.raises(RuntimeError, match="refusing to reuse or summarize"):
            factory._run_worker(payload)


def test_summary_preserves_contract_and_rejects_legacy_partial(
    factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "VOLUME_ROOT", tmp_path)
    complete_dir = tmp_path / "teacher-test" / "parts" / "part_00000"
    complete_dir.mkdir(parents=True)
    report = _part_report(factory)
    (complete_dir / "manifest.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    summary = factory.summarize_run("teacher-test", "run-1")
    assert summary["hard_action_target_information"] == (
        classical_teacher_hard_action_target_information()
    )

    (complete_dir / "manifest.json").unlink()
    report.pop("hard_action_target_information")
    (complete_dir / "manifest.partial.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refusing to reuse or summarize"):
        factory.summarize_run("teacher-test", "run-1")
