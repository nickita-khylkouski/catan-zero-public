from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "backfill_entity_adapter_v5", ROOT / "tools/backfill_entity_adapter_v5.py"
)
assert SPEC is not None and SPEC.loader is not None
backfill = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = backfill
SPEC.loader.exec_module(backfill)


def test_game_row_spans_allows_gaps_but_requires_ordered_trajectories():
    spans = backfill._game_row_spans(
        np.asarray([10, 10, 11, 11, 11]),
        np.asarray([0, 3, 0, 2, 7]),
        np.asarray([True, True, True, True, True]),
        np.asarray([False, False, False, False, False]),
    )
    assert spans == [slice(0, 2), slice(2, 5)]

    with pytest.raises(backfill.BackfillError, match="increase strictly"):
        backfill._game_row_spans(
            np.asarray([10, 10]),
            np.asarray([0, 0]),
            np.asarray([True, True]),
            np.asarray([False, False]),
        )

    with pytest.raises(backfill.BackfillError, match="reappears"):
        backfill._game_row_spans(
            np.asarray([10, 11, 10]),
            np.asarray([0, 0, 0]),
            np.asarray([True, True, True]),
            np.asarray([False, False, False]),
        )


def test_game_row_spans_rejects_incomplete_outcome_contract():
    with pytest.raises(backfill.BackfillError, match="exactly one"):
        backfill._game_row_spans(
            np.asarray([10]),
            np.asarray([0]),
            np.asarray([False]),
            np.asarray([False]),
        )
    with pytest.raises(backfill.BackfillError, match="sealed cutoff"):
        backfill._game_row_spans(
            np.asarray([10, 10]),
            np.asarray([0, 600]),
            np.asarray([False, False]),
            np.asarray([True, True]),
        )


def test_replay_identity_is_byte_exact_and_fails_closed():
    expected = np.asarray([[1.0, np.nan]], dtype=np.float16)
    backfill._require_exact("global_tokens", expected, expected.copy(), 7)
    changed = expected.copy()
    changed[0, 0] = 0.0
    with pytest.raises(backfill.BackfillError, match="global_tokens"):
        backfill._require_exact("global_tokens", expected, changed, 7)


def test_receipt_hash_binds_canonical_payload():
    left = {"schema": "x", "rows": 3}
    right = {"rows": 3, "schema": "x"}
    assert backfill._canonical_sha256(left) == backfill._canonical_sha256(right)
    assert backfill._canonical_sha256(left).startswith("sha256:")


def test_source_paths_are_recursive_and_keep_duplicate_basenames_distinct(tmp_path):
    first = tmp_path / "original" / "gpu0" / "worker_000" / "same.npz"
    second = tmp_path / "replacement" / "gpu1" / "worker_000" / "same.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    paths = backfill._source_paths(tmp_path)
    assert paths == [first, second]
    assert first.relative_to(tmp_path) != second.relative_to(tmp_path)


def test_missing_gap_replay_requires_exactly_one_legal_action(monkeypatch):
    class _Game:
        def __init__(self, legal):
            self.legal = legal

        def playable_action_indices(self, _colors, _kind):
            return self.legal

        def playable_actions_json(self):
            return '[["RED", "ROLL", null]]'

    monkeypatch.setattr(
        backfill,
        "_apply_selected_action",
        lambda game, *_args, **_kwargs: game,
    )
    game = _Game([17])
    assert (
        backfill._apply_missing_automatic_transition(
            game,
            chance_rng=__import__("random").Random(1),
            seed=9,
            decision_index=4,
        )
        is game
    )
    with pytest.raises(backfill.BackfillError, match="2 legal actions"):
        backfill._apply_missing_automatic_transition(
            _Game([17, 18]),
            chance_rng=__import__("random").Random(1),
            seed=9,
            decision_index=4,
        )


def test_output_manifest_binds_authoritative_public_awards_and_receipt(tmp_path):
    receipt = tmp_path / "adapter_v5_backfill_receipt.json"
    receipt.write_text('{"schema":"receipt"}\n')
    manifest_path = backfill._write_output_manifest(tmp_path, receipt)
    import json

    manifest = json.loads(manifest_path.read_text())
    assert "shards" not in manifest
    assert manifest["public_award_feature_provenance"] == {
        "schema_version": "public-award-feature-provenance-v1",
        "contract": "authoritative_v1",
        "feature_producer": "catanatron_rs_public_award_v1",
        "native_capability": "public_award_feature_parity",
    }
    binding = manifest["entity_adapter_backfill"]
    assert binding["receipt"] == receipt.name
    assert binding["receipt_file_sha256"] == backfill._sha256(receipt)
