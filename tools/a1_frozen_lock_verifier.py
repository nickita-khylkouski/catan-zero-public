#!/usr/bin/env python3
"""Replay a path-bound A1 lock with its exact frozen verifier runtime.

Version-3 A1 locks intentionally bind absolute paths inside the checkout that
sealed them.  A newer checkout must not weaken that rule or reseal historical
bytes merely to consume the completed wave.  This module authenticates the
exact verifier named by the raw lock, executes it in a fresh interpreter rooted
at that frozen checkout, and emits a portable authority record for downstream
builder, learner, gate, and promotion receipts.

The implementation is deliberately stdlib-only.  Importing today's A1 modules
before selecting the frozen runtime would recreate the very ambient-checkout
dependency this bridge removes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


AUTHORITY_SCHEMA = "a1-frozen-lock-verifier-authority-v1"
VERIFIER_RELATIVE_PATH = Path("tools/a1_pre_wave_contract.py")


class FrozenVerifierError(RuntimeError):
    """The selected frozen verifier or its result is not authenticated."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FrozenVerifierError(f"cannot read frozen lock JSON: {error}") from error
    if not isinstance(value, dict):
        raise FrozenVerifierError("frozen lock must be a JSON object")
    return value


def verify_frozen_lock(
    lock_path: Path,
    *,
    frozen_repo: Path,
    expected_verifier_sha256: str,
    require_all_job_claims: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact verified lock and a hash-bound verifier authority.

    The caller supplies the verifier digest explicitly.  The raw lock must also
    name the same absolute verifier path and digest exactly once in
    ``provenance.runtime_code_tree``.  The requested job-claim mode is part of
    the returned authority, and the subprocess result must equal the raw JSON
    object exactly.  Current waves use ``True``; an incomplete, already-sealed
    historical wave may explicitly use ``False`` without weakening the current
    verifier path.
    """

    if not isinstance(require_all_job_claims, bool):
        raise FrozenVerifierError("require_all_job_claims must be an exact boolean")

    try:
        root = frozen_repo.expanduser().resolve(strict=True)
        requested_lock = lock_path.expanduser().resolve(strict=True)
        verifier_path = (root / VERIFIER_RELATIVE_PATH).resolve(strict=True)
    except OSError as error:
        raise FrozenVerifierError(
            f"cannot resolve frozen lock-verifier inputs: {error}"
        ) from error
    if not root.is_dir() or not verifier_path.is_file() or not requested_lock.is_file():
        raise FrozenVerifierError("frozen verifier inputs are not regular files/directories")

    raw_lock = _load_json_object(requested_lock)
    actual_verifier_sha256 = _file_sha256(verifier_path)
    if actual_verifier_sha256 != expected_verifier_sha256:
        raise FrozenVerifierError(
            "frozen lock verifier differs from its explicit SHA-256 binding"
        )
    provenance = raw_lock.get("provenance")
    if not isinstance(provenance, dict):
        raise FrozenVerifierError("frozen lock has no runtime-code provenance")
    runtime_records = provenance.get("runtime_code_tree")
    matches = [
        record
        for record in runtime_records or []
        if isinstance(record, dict)
        and record.get("path") == str(verifier_path)
        and record.get("sha256") == actual_verifier_sha256
    ]
    if len(matches) != 1:
        raise FrozenVerifierError(
            "frozen lock does not bind the explicitly selected verifier bytes"
        )

    script = r'''import json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(strict=True)
expected=(root/'tools/a1_pre_wave_contract.py').resolve(strict=True)
from tools import a1_pre_wave_contract as module
if pathlib.Path(module.__file__).resolve(strict=True)!=expected: raise SystemExit(f'frozen verifier import leaked to {module.__file__}')
mode={'true':True,'false':False}[sys.argv[3]]
lock=module.verify_lock(pathlib.Path(sys.argv[2]),require_all_job_claims=mode)
print(json.dumps(lock,sort_keys=True,separators=(',',':')))'''
    environment = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{root / 'src'}:{root}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            str(requested_lock),
            "true" if require_all_job_claims else "false",
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise FrozenVerifierError(f"frozen lock verifier refused: {detail}")
    try:
        verified_lock = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise FrozenVerifierError(
            "frozen lock verifier returned invalid JSON"
        ) from error
    if verified_lock != raw_lock:
        raise FrozenVerifierError(
            "frozen verifier result differs from the requested lock bytes"
        )

    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA,
        "lock": str(requested_lock),
        "lock_file_sha256": _file_sha256(requested_lock),
        "contract_sha256": raw_lock.get("contract_sha256"),
        "frozen_repo": str(root),
        "verifier": str(verifier_path),
        "verifier_sha256": actual_verifier_sha256,
        "require_all_job_claims": require_all_job_claims,
        "verified_lock_sha256": _canonical_sha256(verified_lock),
    }
    authority["authority_sha256"] = _canonical_sha256(authority)
    return copy.deepcopy(verified_lock), authority


def build_frozen_lock_verifier(
    *,
    frozen_repo: Path,
    expected_verifier_sha256: str,
    lock_path: Path,
    require_all_job_claims: bool = True,
) -> tuple[Callable[..., dict[str, Any]], dict[str, Any]]:
    """Build an exact-path ``verify_lock`` adapter plus its authority record."""

    verified_lock, authority = verify_frozen_lock(
        lock_path,
        frozen_repo=frozen_repo,
        expected_verifier_sha256=expected_verifier_sha256,
        require_all_job_claims=require_all_job_claims,
    )
    requested_lock = Path(authority["lock"])

    def replay_verified_lock(
        candidate: Path, *, require_all_job_claims: bool = False
    ) -> dict[str, Any]:
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except OSError as error:
            raise FrozenVerifierError(
                f"cannot replay frozen lock path: {error}"
            ) from error
        if (
            resolved != requested_lock
            or require_all_job_claims is not authority["require_all_job_claims"]
        ):
            raise FrozenVerifierError(
                "frozen lock replay requires the exact path and sealed job-claim mode"
            )
        return copy.deepcopy(verified_lock)

    return replay_verified_lock, copy.deepcopy(authority)
