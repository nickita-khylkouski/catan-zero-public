from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import a1_promotion_transaction as promotion
from tools import a1_v5_recovery_promotion_pack as pack


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(path: Path, candidate: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": promotion.CHECKPOINT_SELECTION_SCHEMA,
                "selected_checkpoint": {
                    "path": str(candidate.resolve()),
                    "sha256": _sha256(candidate),
                },
            }
        ),
        encoding="utf-8",
    )


def test_attach_checkpoint_selection_binds_file_and_refreshes_digest(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    selection = tmp_path / "selection.json"
    _selection(selection, candidate)
    original = {
        "schema_version": promotion.ADJUDICATION_SCHEMA,
        "candidate": {"path": str(candidate.resolve()), "sha256": _sha256(candidate)},
        "adjudication_sha256": "sha256:" + "0" * 64,
    }

    value = pack.attach_checkpoint_selection(
        original,
        checkpoint_selection=selection,
        candidate=candidate.resolve(),
    )

    assert value["candidate"]["training_checkpoint_selection"] == {
        "path": str(selection.resolve()),
        "sha256": _sha256(selection),
    }
    unsigned = dict(value)
    assert unsigned.pop("adjudication_sha256") == promotion._digest_value(unsigned)
    assert "training_checkpoint_selection" not in original["candidate"]


def test_attach_checkpoint_selection_rejects_different_candidate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    selection = tmp_path / "selection.json"
    _selection(selection, other)

    with pytest.raises(pack.PackError, match="requested candidate"):
        pack.attach_checkpoint_selection(
            {"candidate": {"path": str(candidate.resolve())}},
            checkpoint_selection=selection,
            candidate=candidate.resolve(),
        )


def test_cli_requires_recovery_and_frozen_authority_inputs() -> None:
    parser = pack._parser()
    required = {
        action.dest
        for action in parser._actions
        if getattr(action, "required", False)
    }
    assert {
        "contract_lock",
        "frozen_repo",
        "frozen_verifier_sha256",
        "recovery_receipt",
        "checkpoint_selection",
        "dose_screen",
    }.issubset(required)
