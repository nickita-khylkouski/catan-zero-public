from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import train_bc  # type: ignore  # noqa: E402
from build_memmap_corpus import build_memmap_corpus  # type: ignore  # noqa: E402
from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V2  # noqa: E402
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    RUST_ENTITY_ADAPTER_VERSION,
)


def _teacher_arrays(adapter_versions: list[str] | None) -> dict[str, np.ndarray]:
    rows = 2 if adapter_versions is None else len(adapter_versions)
    arrays: dict[str, np.ndarray] = {
        "obs": np.zeros((rows, 4), dtype=np.float16),
        "legal_action_ids": np.tile(
            np.asarray([[0, 1]], dtype=np.int16), (rows, 1)
        ),
        "legal_action_context": np.zeros((rows, 2, 1), dtype=np.float16),
        "action_taken": np.zeros(rows, dtype=np.int16),
        "game_seed": np.full(rows, 17, dtype=np.int64),
        "decision_index": np.arange(rows, dtype=np.int32),
    }
    if adapter_versions is not None:
        arrays["adapter_version"] = np.asarray(adapter_versions)
    return arrays


def test_npz_normalization_and_loader_preserve_adapter_version(tmp_path: Path) -> None:
    versions = [RUST_ENTITY_ADAPTER_VERSION, RUST_ENTITY_ADAPTER_VERSION]
    arrays = _teacher_arrays(versions)
    normalized = train_bc._normalize_teacher_shard(arrays, tmp_path / "shard.npz")
    assert normalized["adapter_version"].tolist() == versions

    teacher = tmp_path / "teacher"
    teacher.mkdir()
    np.savez(teacher / "shard.npz", **arrays)
    loaded = train_bc.load_teacher_data(teacher)
    assert loaded["adapter_version"].tolist() == versions


def test_legacy_normalization_marks_adapter_version_unknown(tmp_path: Path) -> None:
    normalized = train_bc._normalize_teacher_shard(
        _teacher_arrays(None), tmp_path / "legacy.npz"
    )
    assert normalized["adapter_version"].tolist() == ["", ""]


def test_memmap_roundtrip_preserves_adapter_version(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    versions = [RUST_ENTITY_ADAPTER_VERSION, RUST_ENTITY_ADAPTER_VERSION]
    np.savez(teacher / "shard.npz", **_teacher_arrays(versions))

    corpus_dir = tmp_path / "corpus"
    build_memmap_corpus(teacher, corpus_dir, progress_every=0)
    corpus = train_bc.MemmapCorpus(corpus_dir)

    assert "adapter_version" in corpus
    assert np.asarray(corpus["adapter_version"]).tolist() == versions
    assert corpus["adapter_version"].present_values() == {
        RUST_ENTITY_ADAPTER_VERSION
    }


class _PolicyConfig:
    observation_size = 4
    action_size = 2
    context_action_feature_size = 1


class _Policy:
    config = _PolicyConfig()
    action_size = 2
    context_action_feature_size = 1
    policy_type = "dense"


class _EntityPolicy(_Policy):
    policy_type = "entity_graph"
    entity_feature_adapter_version = RUST_ENTITY_ADAPTER_VERSION


class _LegacyEntityPolicy(_EntityPolicy):
    entity_feature_adapter_version = RUST_ENTITY_ADAPTER_V2


@pytest.mark.parametrize(
    ("versions", "message"),
    [
        ([RUST_ENTITY_ADAPTER_VERSION, ""], "mixed known and unknown"),
        ([RUST_ENTITY_ADAPTER_VERSION, "obsolete-v1"], "mixed adapter_version"),
    ],
)
def test_schema_rejects_mixed_adapter_semantics(
    monkeypatch: pytest.MonkeyPatch,
    versions: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(train_bc, "_expected_action_mask_version", lambda _config: "")
    monkeypatch.setattr(
        train_bc, "_expected_static_action_features_sha256", lambda _config: ""
    )
    monkeypatch.setattr(
        train_bc, "_policy_static_action_features_sha256", lambda _policy: ""
    )
    data = _teacher_arrays(versions)
    data["action_mask_version"] = np.asarray(["", ""])
    with pytest.raises(SystemExit, match=message):
        train_bc.validate_teacher_data_schema(
            _Policy(), data, {"invalid_teacher_actions": 0}, object()
        )


def test_entity_schema_rejects_obsolete_known_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_bc, "_expected_action_mask_version", lambda _config: "")
    monkeypatch.setattr(
        train_bc, "_expected_static_action_features_sha256", lambda _config: ""
    )
    monkeypatch.setattr(
        train_bc, "_policy_static_action_features_sha256", lambda _policy: ""
    )
    data = _teacher_arrays(["obsolete-v1", "obsolete-v1"])
    data["action_mask_version"] = np.asarray(["", ""])
    with pytest.raises(SystemExit, match="does not match checkpoint entity adapter"):
        train_bc.validate_teacher_data_schema(
            _EntityPolicy(), data, {"invalid_teacher_actions": 0}, object()
        )


def test_current_entity_schema_rejects_missing_production_adapter_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_bc, "_expected_action_mask_version", lambda _config: "catalog-v1"
    )
    monkeypatch.setattr(
        train_bc, "_expected_static_action_features_sha256", lambda _config: ""
    )
    monkeypatch.setattr(
        train_bc, "_policy_static_action_features_sha256", lambda _policy: ""
    )
    seed = _teacher_arrays(None)
    data = {
        key: np.repeat(value, 500, axis=0)
        for key, value in seed.items()
    }
    data["action_mask_version"] = np.full(1000, "catalog-v1")

    with pytest.raises(SystemExit, match="missing adapter_version.*current entity adapter"):
        train_bc.validate_teacher_data_schema(
            _EntityPolicy(), data, {"invalid_teacher_actions": 0}, object()
        )


def test_legacy_entity_schema_does_not_reclassify_unknown_data_as_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_bc, "_expected_action_mask_version", lambda _config: "catalog-v1"
    )
    monkeypatch.setattr(
        train_bc, "_expected_static_action_features_sha256", lambda _config: ""
    )
    monkeypatch.setattr(
        train_bc, "_policy_static_action_features_sha256", lambda _policy: ""
    )
    seed = _teacher_arrays(None)
    data = {
        key: np.repeat(value, 500, axis=0)
        for key, value in seed.items()
    }
    data["action_mask_version"] = np.full(1000, "catalog-v1")

    with pytest.raises(SystemExit) as error:
        train_bc.validate_teacher_data_schema(
            _LegacyEntityPolicy(), data, {"invalid_teacher_actions": 0}, object()
        )
    assert "missing adapter_version" not in str(error.value)


def test_lazy_categorical_provenance_does_not_materialize() -> None:
    class LazyColumn:
        def value_counts(self):
            return {RUST_ENTITY_ADAPTER_VERSION: 9, "": 1}

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("categorical provenance was eagerly decoded")

    column = LazyColumn()
    assert train_bc._string_column_counts(column, rows=10) == {
        RUST_ENTITY_ADAPTER_VERSION: 9,
        "": 1,
    }
    assert train_bc._string_column_present_values(column, rows=10) == {
        RUST_ENTITY_ADAPTER_VERSION,
        "",
    }
