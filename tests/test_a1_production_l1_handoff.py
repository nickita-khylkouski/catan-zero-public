from __future__ import annotations

import json

import pytest

from tools import a1_production_l1_handoff as handoff


def test_pending_bundle_digest_and_authorization_fail_closed(tmp_path) -> None:
    payload = {
        "schema_version": handoff.SCHEMA,
        "promotion_ready": False,
        "pointer_mutation_authorized": False,
        "learner": {},
        "evidence": {},
        "authoritative_transaction_audit": {},
    }
    payload["bundle_sha256"] = handoff._digest(payload)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((KeyError, handoff.HandoffError)):
        handoff.verify(path)


def test_checkpoint_sha_refuses_missing_binding() -> None:
    with pytest.raises(handoff.HandoffError, match="candidate checkpoint SHA"):
        handoff._checkpoint_sha({}, "test")
