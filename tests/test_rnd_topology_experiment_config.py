from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT = (
    _ROOT / "configs" / "rnd" / "topology_real_train_20260710" / "experiment.json"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_config_sha256(payload: dict) -> str:
    semantic_payload = dict(payload)
    semantic_payload.pop("config_sha256")
    encoded = json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_real_train_experiment_has_exact_nonrecursive_self_hash() -> None:
    payload = json.loads(_EXPERIMENT.read_text(encoding="utf-8"))
    assert (
        payload["config_sha256_scope"]
        == "canonical_json_without_config_sha256"
    )
    declared = payload["config_sha256"]
    assert _SHA256_RE.fullmatch(declared)
    assert declared == _canonical_config_sha256(payload)
