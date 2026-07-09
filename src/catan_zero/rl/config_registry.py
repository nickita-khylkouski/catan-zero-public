"""Append-only config-hash registry (task CAT-66).

Maps a pipeline config's ``config_hash`` to its full, fully-resolved config
plus a timestamp and a free-text purpose. Every train/generate/gate/eval run
registers its config so a hash seen in an output artifact can always be
resolved back to the exact regime that produced it, without re-deriving it from
a CLI string.

FORMAT: newline-delimited JSON (JSONL), one record per line:
    {"config_hash": "sha256:...", "full_config_hash": "sha256:<64>",
     "pipeline": "train", "schema_version": 1, "timestamp": "<UTC ISO8601>",
     "purpose": "...", "config": {<canonical_payload>}}

DISCIPLINE: append-only and idempotent. Registering a hash already present
(same short hash) is a no-op -- the file never rewrites existing lines, so it
is safe to append to concurrently and safe to keep under version control as a
durable audit log. The registry path defaults to ``configs/config_registry.jsonl``
at the repo root and is overridable via ``$CATAN_ZERO_CONFIG_REGISTRY`` (tests
point it at a tmp file).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catan_zero.rl.pipeline_configs import PipelineConfig

REGISTRY_ENV_VAR = "CATAN_ZERO_CONFIG_REGISTRY"
_DEFAULT_RELATIVE = Path("configs") / "config_registry.jsonl"


def _repo_root() -> Path:
    # src/catan_zero/rl/config_registry.py -> repo root is three parents up
    # from the package dir (src/catan_zero -> src -> repo).
    return Path(__file__).resolve().parents[3]


def default_registry_path() -> Path:
    """The active registry path: ``$CATAN_ZERO_CONFIG_REGISTRY`` or the repo default."""
    override = os.environ.get(REGISTRY_ENV_VAR)
    if override:
        return Path(override)
    return _repo_root() / _DEFAULT_RELATIVE


def _iter_records(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A partially written trailing line (crash mid-append) must not
                # crash a reader; skip it. Full lines are always valid JSON.
                continue


def lookup(config_hash: str, *, path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Return the most recent registry record for ``config_hash``, or None."""
    registry = Path(path) if path is not None else default_registry_path()
    found: dict[str, Any] | None = None
    for record in _iter_records(registry):
        if record.get("config_hash") == config_hash:
            found = record
    return found


def is_registered(config_hash: str, *, path: str | os.PathLike[str] | None = None) -> bool:
    return lookup(config_hash, path=path) is not None


def register(
    config: PipelineConfig,
    *,
    purpose: str = "",
    path: str | os.PathLike[str] | None = None,
    timestamp: str | None = None,
) -> str:
    """Append ``config`` to the registry (idempotent) and return its short hash.

    A no-op when the short hash is already present, so callers can register
    unconditionally on every run. Creates the registry file and parent dir on
    first use.
    """
    registry = Path(path) if path is not None else default_registry_path()
    config_hash = config.config_hash()
    if is_registered(config_hash, path=registry):
        return config_hash
    record = {
        "config_hash": config_hash,
        "full_config_hash": config.full_config_hash(),
        "pipeline": config.PIPELINE,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "config": config.canonical_payload(),
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with registry.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return config_hash
