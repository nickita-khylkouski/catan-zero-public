from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import _validate_policy_target_identity_scope


IDENTITY = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


class _Corpus:
    def __init__(self, identity: str | None) -> None:
        self.meta = (
            {}
            if identity is None
            else {
                "stage_c_policy_overlay": {
                    "target_policy_target_identity_sha256": identity
                }
            }
        )


def test_exact_policy_target_identity_is_admitted() -> None:
    report = _validate_policy_target_identity_scope(
        _Corpus(IDENTITY),
        np.asarray([1.0, 0.0, 0.5], dtype=np.float32),
        accepted_identities=(IDENTITY,),
    )

    assert report["mode"] == "explicit_accepted_identity_set"
    assert report["realized_active_policy_target_identity_sha256"] == [IDENTITY]
    assert report["components"]["corpus"]["accepted"] is True


def test_missing_identity_cannot_supply_policy_ce_when_identity_is_required() -> None:
    with pytest.raises(SystemExit, match="lack exact target identity"):
        _validate_policy_target_identity_scope(
            _Corpus(None),
            np.asarray([1.0, 0.0], dtype=np.float32),
            accepted_identities=(IDENTITY,),
        )


def test_unaccepted_operator_is_rejected() -> None:
    with pytest.raises(SystemExit, match="unaccepted target operator"):
        _validate_policy_target_identity_scope(
            _Corpus(OTHER),
            np.asarray([1.0], dtype=np.float32),
            accepted_identities=(IDENTITY,),
        )


def test_legacy_unbound_corpus_is_labeled_diagnostic_without_explicit_gate() -> None:
    report = _validate_policy_target_identity_scope(
        _Corpus(None),
        np.asarray([1.0], dtype=np.float32),
        accepted_identities=(),
    )

    assert report["mode"] == "legacy_unbound_diagnostic"
    assert report["legacy_unbound_active_components"] == ["corpus"]
