from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.checkpoint_runtime_semantics import (  # noqa: E402
    ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
    assert_entity_graph_checkpoint_runtime_semantics,
    current_entity_graph_forward_semantics,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)
from tools import gumbel_search_cross_net_h2h as h2h  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
POLICY_SOURCE = REPO / "src/catan_zero/rl/entity_token_policy.py"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _identity(*, semantic_sha: str | None = None) -> dict[str, object]:
    identity = dict(current_entity_graph_forward_semantics(POLICY_SOURCE))
    identity.pop("binding_sha256")
    if semantic_sha is not None:
        identity["semantic_token_sha256"] = semantic_sha
    identity["binding_sha256"] = _canonical_sha256(identity)
    return identity


def _metadata_checkpoint(path: Path, identity: dict[str, object]) -> Path:
    torch.save({ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: identity}, path)
    return path


def test_new_entity_checkpoint_stamps_forward_semantics(tmp_path: Path) -> None:
    config = EntityGraphConfig(
        action_size=1,
        static_action_feature_size=1,
        hidden_size=8,
        state_layers=1,
        attention_heads=1,
        dropout=0.0,
    )
    policy = EntityGraphPolicy(
        config,
        np.zeros((1, 1), dtype=np.float32),
        device="cpu",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    policy.save(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] == _identity()


def test_h2h_preflight_accepts_same_forward_semantics(tmp_path: Path) -> None:
    candidate = _metadata_checkpoint(tmp_path / "candidate.pt", _identity())
    baseline = _metadata_checkpoint(tmp_path / "baseline.pt", _identity())

    result = h2h._preflight_checkpoint_runtime_semantics(candidate, baseline)

    assert result["candidate"]["compatible"] is True
    assert result["baseline"]["compatible"] is True
    assert result["candidate"]["provenance"] == "checkpoint_stamp"


def test_h2h_preflight_rejects_mismatch_before_evaluator_or_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _metadata_checkpoint(
        tmp_path / "candidate.pt",
        _identity(semantic_sha="sha256:" + "a" * 64),
    )
    baseline = _metadata_checkpoint(tmp_path / "baseline.pt", _identity())

    evaluator_constructed = False

    def _unexpected_evaluator(*_args, **_kwargs):
        nonlocal evaluator_constructed
        evaluator_constructed = True
        raise AssertionError("evaluator must not be constructed during preflight")

    monkeypatch.setattr(
        h2h.BatchedEntityGraphRustEvaluator,
        "from_checkpoint",
        _unexpected_evaluator,
    )
    with pytest.raises(RuntimeError, match="forward semantic mismatch"):
        h2h._preflight_checkpoint_runtime_semantics(candidate, baseline)

    assert evaluator_constructed is False


def test_reviewed_b4_training_binding_is_accepted_without_new_stamp(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pre-stamp.pt"
    checkpoint.write_bytes(b"only path identity is needed for stamped training source")
    payload = {
        "value_training": {
            "checkout_runtime_binding": {
                "modules": {
                    "catan_zero.rl.entity_token_policy": {
                        "sha256": (
                            "sha256:b4e2618bc36296470f13ce3dee228b34fd7d117c"
                            "0211380c46393450793ce975"
                        )
                    }
                }
            }
        }
    }

    result = assert_entity_graph_checkpoint_runtime_semantics(
        payload,
        checkpoint_path=checkpoint,
        policy_source=POLICY_SOURCE,
    )

    assert result["compatible"] is True
    assert result["provenance"] == "reviewed_training_source_binding"


def test_unreviewed_metadata_free_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unknown.pt"
    checkpoint.write_bytes(b"unknown legacy checkpoint")

    with pytest.raises(RuntimeError, match="no accepted.*forward semantic identity"):
        assert_entity_graph_checkpoint_runtime_semantics(
            {},
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
        )
