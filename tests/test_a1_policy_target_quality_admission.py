from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_policy_target_quality_admission as admission


class _Corpus:
    def __init__(self, *, candidate_q: float, root_prior: float) -> None:
        rows = admission.MINIMUM_GAME_CLUSTERS
        self.meta = {}
        self._data = {
            "game_seed": np.arange(11, 11 + rows, dtype=np.int64),
            "player": np.full(rows, "RED"),
            "winner": np.full(rows, "RED"),
            "terminated": np.ones(rows, dtype=np.bool_),
            "truncated": np.zeros(rows, dtype=np.bool_),
            "policy_weight_multiplier": np.ones(rows, dtype=np.float32),
            "used_full_search": np.ones(rows, dtype=np.bool_),
            "root_prior_value": np.full(rows, root_prior, dtype=np.float32),
            "root_prior_value_mask": np.ones(rows, dtype=np.bool_),
            "action_taken": np.ones(rows, dtype=np.int16),
            "legal_action_ids": np.tile(
                np.asarray([[1, 2]], dtype=np.int16), (rows, 1)
            ),
            "target_policy": np.tile(
                np.asarray([[1.0, 0.0]], dtype=np.float32), (rows, 1)
            ),
            "target_policy_mask": np.ones((rows, 2), dtype=np.bool_),
            "search_evidence_version": np.full(rows, 2, dtype=np.uint8),
            "search_evidence_mask": np.ones(rows, dtype=np.bool_),
            "search_completed_q_flat": np.tile(
                np.asarray([[candidate_q, -0.5]], dtype=np.float32), (rows, 1)
            ),
        }

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key: str):
        return self._data[key]


def _record(tmp_path: Path) -> dict[str, object]:
    return {
        "component_id": "current_producer",
        "corpus_dir": str(tmp_path),
        "payload_inventory_sha256": "sha256:" + "1" * 64,
    }


def _patch_corpus(
    monkeypatch: pytest.MonkeyPatch, corpus: _Corpus, tmp_path: Path
) -> None:
    monkeypatch.setattr(admission.train_bc, "MemmapCorpus", lambda _path: corpus)
    monkeypatch.setattr(
        admission.train_bc,
        "_validate_memmap_payload_inventory",
        lambda _path, _meta: "sha256:" + "1" * 64,
    )


def test_terminal_game_balanced_policy_quality_passes_both_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_corpus(monkeypatch, _Corpus(candidate_q=1.0, root_prior=0.0), tmp_path)

    result = admission.evaluate([_record(tmp_path)])

    assert result["eligible_row_count"] == admission.MINIMUM_GAME_CLUSTERS
    assert result["game_cluster_count"] == admission.MINIMUM_GAME_CLUSTERS
    assert result["primary"]["paired_delta_mean"] == pytest.approx(-1.0)
    assert result["primary"]["bootstrap_one_sided_95_ucb"] <= 0.0
    assert result["secondary_selected_q"]["bootstrap_one_sided_95_ucb"] <= 0.0
    assert result["admitted"] is True


def test_component_seed_namespaces_are_distinct_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_corpus(monkeypatch, _Corpus(candidate_q=1.0, root_prior=0.0), tmp_path)
    first = _record(tmp_path)
    second = {**first, "component_id": "recent_history"}

    result = admission.evaluate([first, second])

    assert result["game_cluster_count"] == 2 * admission.MINIMUM_GAME_CLUSTERS


def test_policy_quality_fails_on_positive_ucb_or_missing_raw_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _Corpus(candidate_q=0.0, root_prior=1.0)
    _patch_corpus(monkeypatch, corpus, tmp_path)
    result = admission.evaluate([_record(tmp_path)])
    assert result["primary"]["bootstrap_one_sided_95_ucb"] > 0.0
    assert result["secondary_selected_q"]["bootstrap_one_sided_95_ucb"] > 0.0
    assert result["admitted"] is False

    del corpus._data["root_prior_value"]
    with pytest.raises(admission.AdmissionError, match="lacks policy quality columns"):
        admission.evaluate([_record(tmp_path)])


def test_policy_quality_rejects_nonterminal_policy_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _Corpus(candidate_q=1.0, root_prior=0.0)
    corpus._data["terminated"][0] = False
    _patch_corpus(monkeypatch, corpus, tmp_path)
    with pytest.raises(admission.AdmissionError, match="non-terminal games"):
        admission.evaluate([_record(tmp_path)])


def test_policy_quality_requires_every_active_row_and_normalized_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _Corpus(candidate_q=1.0, root_prior=0.0)
    corpus._data["search_evidence_mask"][0] = False
    _patch_corpus(monkeypatch, corpus, tmp_path)
    with pytest.raises(admission.AdmissionError, match="lack search evidence"):
        admission.evaluate([_record(tmp_path)])

    corpus._data["search_evidence_mask"][0] = True
    corpus._data["target_policy"][0] = [0.4, 0.4]
    with pytest.raises(admission.AdmissionError, match="not normalized"):
        admission.evaluate([_record(tmp_path)])

    corpus._data["target_policy"][0] = [1.0, 0.0]
    corpus._data["target_policy_mask"][0, 1] = False
    with pytest.raises(admission.AdmissionError, match="malformed policy evidence"):
        admission.evaluate([_record(tmp_path)])


def _signed_receipt(identity: dict) -> dict:
    payload = {
        "schema_version": admission.RECEIPT_SCHEMA,
        "status": "admitted",
        "identity": copy.deepcopy(identity),
        "metric_contract": admission.metric_contract(),
        "metrics": {
            "eligible_row_count": admission.MINIMUM_GAME_CLUSTERS,
            "game_cluster_count": admission.MINIMUM_GAME_CLUSTERS,
            "component_game_set_sha256": "sha256:" + "2" * 64,
            "primary": {
                "paired_delta_mean": -0.1,
                "bootstrap_one_sided_95_ucb": 0.0,
                "passes": True,
            },
            "secondary_selected_q": {
                "paired_delta_mean": -0.2,
                "bootstrap_one_sided_95_ucb": -0.01,
                "passes": True,
            },
            "admitted": True,
        },
    }
    payload["receipt_sha256"] = admission._value_sha256(payload)  # noqa: SLF001
    return payload


def test_receipt_verifier_fails_closed_on_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"schema_version": admission.IDENTITY_SCHEMA, "identity": "exact"}
    monkeypatch.setattr(admission, "expected_identity", lambda **_kwargs: expected)
    receipt = tmp_path / "quality.json"
    payload = _signed_receipt(expected)
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    verified = admission.verify_receipt(receipt, verified={}, composite_meta={})
    assert verified["metrics"]["admitted"] is True

    drifted = _signed_receipt({**expected, "identity": "drifted"})
    receipt.write_text(json.dumps(drifted) + "\n", encoding="utf-8")
    with pytest.raises(admission.AdmissionError, match="identity or status drifted"):
        admission.verify_receipt(receipt, verified={}, composite_meta={})
