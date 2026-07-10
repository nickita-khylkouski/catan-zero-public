from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import rnd_build_topology_sensitive_mask as mask_builder  # noqa: E402


def _corpus_meta(validation_seeds: list[int] | tuple[int, ...] = (11, 13)) -> dict:
    array = np.asarray(validation_seeds, dtype="<i8")
    return {
        "payload_inventory_sha256": "sha256:" + "a" * 64,
        "selected_game_seed_manifest": {
            "validation_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(array.tobytes()).hexdigest(),
        },
    }


def _validation_manifest(path: Path, seeds: list[int]) -> Path:
    array = np.asarray(seeds, dtype="<i8")
    payload = {
        "schema_version": "train-validation-game-seeds-v1",
        "validation_game_seed_count": len(seeds),
        "validation_game_seed_set_sha256": "sha256:"
        + hashlib.sha256(array.tobytes()).hexdigest(),
        "game_seeds": seeds,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _tokens(action_types: list[str], width: int = 4) -> np.ndarray:
    result = np.zeros((width, 50), dtype=np.float16)
    for row, action_type in enumerate(action_types):
        result[row, 0] = 1
        result[row, 2 + mask_builder.ACTION_TYPES.index(action_type)] = 1
    return result


class _Corpus:
    forbidden = {
        "target_policy",
        "target_scores",
        "winner",
        "root_value",
        "prior_policy",
        "final_actual_vps",
    }

    def __init__(self, _: Path):
        self.meta = _corpus_meta()
        self.row_count = 5
        rows = [
            ["BUILD_SETTLEMENT", "BUILD_SETTLEMENT"],
            ["BUILD_ROAD", "BUILD_ROAD", "BUILD_ROAD"],
            ["MOVE_ROBBER", "MOVE_ROBBER"],
            ["BUILD_CITY", "BUILD_CITY"],
            ["BUILD_ROAD"],  # forced/single-target: excluded
        ]
        tokens = np.stack([_tokens(row) for row in rows])
        targets = np.full((5, 4, 4), -1, dtype=np.int16)
        targets[0, :2, 1] = [4, 8]
        targets[1, :3, 2] = [1, 2, 3]
        targets[2, :2, 0] = [5, 9]
        targets[3, :2, 1] = [7, 12]
        targets[4, 0, 2] = 6
        live = np.zeros((5, 4), dtype=np.bool_)
        for index, row in enumerate(rows):
            live[index, : len(row)] = True
        self.values = {
            "game_seed": np.asarray([11, 11, 13, 13, 13], dtype=np.int64),
            "decision_index": np.asarray([0, 1, 0, 1, 2], dtype=np.int32),
            "phase": np.asarray(
                ["initial_build", "INITIAL_BUILD", "robber", "main", "main"]
            ),
            "legal_action_tokens": tokens,
            "legal_action_target_ids": targets,
            "legal_action_mask": live,
        }

    def keys(self):
        return self.values.keys()

    def __getitem__(self, key: str):
        assert key not in self.forbidden, f"post-training/outcome field read: {key}"
        return self.values[key]


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_meta.json").write_text(
        json.dumps(_corpus_meta()), encoding="utf-8"
    )
    manifest = _validation_manifest(tmp_path / "validation.json", [11, 13])
    monkeypatch.setattr(mask_builder, "MemmapCorpus", _Corpus)
    monkeypatch.setattr(
        mask_builder,
        "_validate_memmap_payload_inventory",
        lambda *_: "sha256:" + "a" * 64,
    )
    return corpus, manifest


def test_build_mask_is_deterministic_metadata_only_and_auditable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, manifest = _prepare(tmp_path, monkeypatch)
    first = mask_builder.build_mask(
        corpus,
        manifest,
        min_games=2,
        min_decisions=4,
        min_category_decisions={"robber_hex_target": 1},
        batch_size=1,
    )
    second = mask_builder.build_mask(
        corpus,
        manifest,
        min_games=2,
        min_decisions=4,
        min_category_decisions={"robber_hex_target": 1},
        batch_size=99,
    )
    relocated_corpus = tmp_path / "relocated" / "corpus"
    relocated_corpus.mkdir(parents=True)
    (relocated_corpus / "corpus_meta.json").write_bytes(
        (corpus / "corpus_meta.json").read_bytes()
    )
    relocated_manifest = tmp_path / "relocated" / "validation.json"
    relocated_manifest.write_bytes(manifest.read_bytes())
    relocated = mask_builder.build_mask(
        relocated_corpus,
        relocated_manifest,
        min_games=2,
        min_decisions=4,
        min_category_decisions={"robber_hex_target": 1},
    )

    assert first == second == relocated
    assert first["summary"] == {
        "decision_count": 4,
        "game_count": 2,
        "category_counts": {
            "city_vertex_target": 1,
            "initial_road_edge_target": 1,
            "initial_settlement_vertex_target": 1,
            "robber_hex_target": 1,
        },
    }
    assert [row["decision_id"] for row in first["members"]] == [
        "seed:11:decision:0",
        "seed:11:decision:1",
        "seed:13:decision:0",
        "seed:13:decision:1",
    ]
    artifact = dict(first)
    declared_artifact_sha = artifact.pop("artifact_sha256")
    assert declared_artifact_sha == mask_builder._value_sha256(artifact)
    assert first["members_sha256"] == mask_builder._value_sha256(first["members"])
    assert first["config"]["port_category"]["included"] is False


@pytest.mark.parametrize(
    ("minimums", "match"),
    [
        ({"min_games": 3, "min_decisions": 1}, "selected games"),
        ({"min_games": 1, "min_decisions": 5}, "selected decisions"),
    ],
)
def test_build_mask_fails_closed_on_insufficient_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minimums: dict[str, int],
    match: str,
):
    corpus, manifest = _prepare(tmp_path, monkeypatch)
    with pytest.raises(mask_builder.MaskBuildError, match=match):
        mask_builder.build_mask(corpus, manifest, **minimums)


def test_build_mask_fails_closed_on_missing_validation_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_meta.json").write_text(
        json.dumps(_corpus_meta((11, 17))), encoding="utf-8"
    )
    manifest = _validation_manifest(tmp_path / "validation.json", [11, 17])

    class MissingGameCorpus(_Corpus):
        def __init__(self, path: Path):
            super().__init__(path)
            self.meta = _corpus_meta((11, 17))

    monkeypatch.setattr(mask_builder, "MemmapCorpus", MissingGameCorpus)
    monkeypatch.setattr(
        mask_builder,
        "_validate_memmap_payload_inventory",
        lambda *_: "sha256:" + "a" * 64,
    )
    with pytest.raises(mask_builder.MaskBuildError, match="validation games missing"):
        mask_builder.build_mask(corpus, manifest, min_games=1, min_decisions=1)


def test_build_mask_rejects_manifest_not_bound_to_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, manifest = _prepare(tmp_path, monkeypatch)
    wrong_meta = _corpus_meta()
    wrong_meta["selected_game_seed_manifest"] = {
        "validation_game_seed_set_sha256": "sha256:" + "b" * 64
    }
    (corpus / "corpus_meta.json").write_text(json.dumps(wrong_meta), encoding="utf-8")

    class BoundCorpus(_Corpus):
        def __init__(self, path: Path):
            super().__init__(path)
            self.meta = wrong_meta

    monkeypatch.setattr(mask_builder, "MemmapCorpus", BoundCorpus)
    with pytest.raises(mask_builder.MaskBuildError, match="holdout bound"):
        mask_builder.build_mask(corpus, manifest, min_games=1, min_decisions=1)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.pop("schema_version"), "schema_version"),
        (
            lambda payload: payload.__setitem__("schema_version", "wrong-schema"),
            "schema_version",
        ),
        (
            lambda payload: payload.pop("validation_game_seed_count"),
            "game count",
        ),
        (
            lambda payload: payload.__setitem__("validation_game_seed_count", 99),
            "game count",
        ),
        (
            lambda payload: payload.pop("validation_game_seed_set_sha256"),
            "game-seed digest",
        ),
        (
            lambda payload: payload.__setitem__(
                "validation_game_seed_set_sha256", "sha256:" + "f" * 64
            ),
            "game-seed digest",
        ),
    ],
)
def test_validation_manifest_contract_fails_closed(
    tmp_path: Path, mutation, match: str
):
    manifest = _validation_manifest(tmp_path / "validation.json", [11, 13])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(mask_builder.MaskBuildError, match=match):
        mask_builder._load_validation_seeds(manifest)


def test_build_mask_rejects_missing_corpus_holdout_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, manifest = _prepare(tmp_path, monkeypatch)
    unbound_meta = _corpus_meta()
    unbound_meta.pop("selected_game_seed_manifest")
    (corpus / "corpus_meta.json").write_text(
        json.dumps(unbound_meta), encoding="utf-8"
    )

    class UnboundCorpus(_Corpus):
        def __init__(self, path: Path):
            super().__init__(path)
            self.meta = unbound_meta

    monkeypatch.setattr(mask_builder, "MemmapCorpus", UnboundCorpus)
    with pytest.raises(mask_builder.MaskBuildError, match="must bind"):
        mask_builder.build_mask(corpus, manifest, min_games=1, min_decisions=1)


def test_inventory_is_authenticated_before_memmap_corpus_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, manifest = _prepare(tmp_path, monkeypatch)
    opened = False

    class MustNotOpen:
        def __init__(self, _path: Path):
            nonlocal opened
            opened = True

    def reject_inventory(_path: Path, _meta: dict):
        raise SystemExit("payload digest mismatch")

    monkeypatch.setattr(mask_builder, "MemmapCorpus", MustNotOpen)
    monkeypatch.setattr(
        mask_builder, "_validate_memmap_payload_inventory", reject_inventory
    )
    with pytest.raises(mask_builder.MaskBuildError, match="payload digest mismatch"):
        mask_builder.build_mask(corpus, manifest, min_games=1, min_decisions=1)
    assert opened is False


def test_loaded_corpus_metadata_must_match_authenticated_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, manifest = _prepare(tmp_path, monkeypatch)

    class DriftedCorpus(_Corpus):
        def __init__(self, path: Path):
            super().__init__(path)
            self.meta = {**self.meta, "row_count": 999}

    monkeypatch.setattr(mask_builder, "MemmapCorpus", DriftedCorpus)
    with pytest.raises(mask_builder.MaskBuildError, match="differs from authenticated"):
        mask_builder.build_mask(corpus, manifest, min_games=1, min_decisions=1)


def test_loaded_corpus_metadata_comparison_accepts_real_like_nan_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus, manifest = _prepare(tmp_path, monkeypatch)
    nan_meta = {**_corpus_meta(), "stats": {"optional_metric": float("nan")}}
    (corpus / "corpus_meta.json").write_text(
        json.dumps(nan_meta), encoding="utf-8"
    )

    class NanStatsCorpus(_Corpus):
        def __init__(self, path: Path):
            super().__init__(path)
            self.meta = json.loads(
                (path / "corpus_meta.json").read_text(encoding="utf-8")
            )

    monkeypatch.setattr(mask_builder, "MemmapCorpus", NanStatsCorpus)
    artifact = mask_builder.build_mask(
        corpus, manifest, min_games=1, min_decisions=1
    )
    assert artifact["summary"]["decision_count"] == 4


def test_live_action_without_exactly_one_type_fails_closed():
    tokens = _tokens(["BUILD_ROAD", "BUILD_ROAD"])
    tokens[1, 2 + mask_builder.ACTION_TYPES.index("BUILD_CITY")] = 1
    targets = np.full((4, 4), -1, dtype=np.int16)
    targets[:2, 2] = [1, 2]
    live = np.asarray([True, True, False, False])
    with pytest.raises(mask_builder.MaskBuildError, match="exactly one"):
        mask_builder._category_for_row(tokens, targets, live, "main")


def test_parse_category_minimums_rejects_unknown_or_duplicate():
    with pytest.raises(mask_builder.MaskBuildError, match="known_category"):
        mask_builder._parse_category_minimums(["port=1"])
    with pytest.raises(mask_builder.MaskBuildError, match="duplicate"):
        mask_builder._parse_category_minimums(
            ["robber_hex_target=1", "robber_hex_target=2"]
        )
