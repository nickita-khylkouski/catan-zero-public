from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path, PurePath
import stat
import subprocess

import pytest

from tools import a1_one_dose_train as executor


def test_nested_json_frontier_reads_and_inspects_each_path_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text("# verifier\n", encoding="utf-8")
    artifacts = []
    for index in range(48):
        artifact = tmp_path / f"artifact-{index:03d}.bin"
        artifact.write_bytes(bytes([index]))
        artifacts.append(artifact)

    linked_json = [tmp_path / f"linked-{index:02d}.json" for index in range(8)]
    noise = [
        {"scalar": index, "nested": [index, {"again": index}]}
        for index in range(128)
    ]
    for index in reversed(range(len(linked_json))):
        payload: dict[str, object] = {
            "duplicate_artifact_path": artifacts[index % len(artifacts)].name,
            "noise": noise,
        }
        if index + 1 < len(linked_json):
            payload["next_path"] = linked_json[index + 1].name
        linked_json[index].write_text(json.dumps(payload), encoding="utf-8")

    raw_lock: dict[str, object] = {
        "entry_path": linked_json[0].name,
        "artifacts": [{"path": path.name} for path in artifacts],
        "noise": noise,
    }
    lock = tmp_path / "contract.lock.json"
    lock.write_text(json.dumps(raw_lock), encoding="utf-8")
    expected_json = {str(lock), *(str(path) for path in linked_json)}
    read_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    original_read_text = Path.read_text
    original_suffix = PurePath.suffix

    def counted_read_text(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        key = str(self)
        if key in expected_json:
            read_counts[key] += 1
        return original_read_text(self, *args, **kwargs)

    def counted_suffix(self: PurePath) -> str:
        suffix_counts[str(self)] += 1
        return original_suffix.__get__(self, type(self))

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    monkeypatch.setattr(PurePath, "suffix", property(counted_suffix))

    identities, eligible = executor._referenced_verification_paths(
        lock,
        raw_lock,
        verifier_path=verifier,
    )

    assert eligible is True
    assert read_counts == Counter({path: 1 for path in expected_json})
    assert suffix_counts and max(suffix_counts.values()) == 1
    identity_paths = [str(row["path"]) for row in identities]
    assert identity_paths == sorted(identity_paths)
    assert set(identity_paths) == {
        str(lock),
        str(verifier),
        str(Path(executor.__file__).resolve(strict=True)),
        *(str(path) for path in artifacts),
        *(str(path) for path in linked_json),
    }


def _reviewed_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, list[list[str]]]:
    sealed = tmp_path / "sealed"
    (sealed / "tools").mkdir(parents=True)
    (sealed / "src").mkdir()
    verifier = sealed / "tools" / "a1_pre_wave_contract.py"
    verifier.write_text("# sealed verifier\n", encoding="utf-8")
    dependency = sealed / "src" / "dependency.py"
    dependency.write_text("SAFE = True\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"accepted"}\n', encoding="utf-8")
    learner_code = [
        {"path": str(dependency), "sha256": executor._file_sha256(dependency)}
    ]
    runtime_code = [{"path": str(verifier), "sha256": executor._file_sha256(verifier)}]
    payload: dict[str, object] = {
        "schema_version": "test-reviewed-lock-v1",
        "provenance": {
            "learner_code": learner_code,
            "learner_code_sha256": executor._value_sha256(learner_code),
            "runtime_code_tree": runtime_code,
            "runtime_code_tree_sha256": executor._value_sha256(runtime_code),
        },
        "evidence": {"path": str(evidence)},
    }
    payload["contract_sha256"] = executor._value_sha256(payload)
    lock = tmp_path / "contract.lock.json"
    lock.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    lock_sha256 = executor._file_sha256(lock)
    monkeypatch.setattr(executor, "TRUSTED_A1_LOCK_PATH", lock)
    monkeypatch.setattr(executor, "TRUSTED_A1_LOCK_FILE_SHA256", lock_sha256)
    monkeypatch.setattr(executor, "TRUSTED_A1_VERIFIER_PATH", verifier)
    monkeypatch.setattr(
        executor, "TRUSTED_A1_VERIFIER_SHA256", executor._file_sha256(verifier)
    )
    monkeypatch.setenv(
        executor.LOCK_VERIFICATION_CACHE_ENV, str(tmp_path / "verification-cache")
    )
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload, sort_keys=True),
            stderr="",
        )

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    return lock, verifier, dependency, evidence, calls


def test_second_reviewed_verification_uses_owner_only_atomic_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, _verifier, _dependency, _evidence, calls = _reviewed_lock(
        tmp_path, monkeypatch
    )
    reviewed_sha256 = executor._file_sha256(lock)

    first = executor._verify_lock_with_sealed_runtime(
        lock, reviewed_lock_file_sha256=reviewed_sha256
    )
    second = executor._verify_lock_with_sealed_runtime(
        lock, reviewed_lock_file_sha256=reviewed_sha256
    )

    assert first == second
    assert len(calls) == 1
    cache_root = Path(os.environ[executor.LOCK_VERIFICATION_CACHE_ENV])
    entries = list(cache_root.glob("*.json"))
    assert len(entries) == 1
    assert stat.S_IMODE(cache_root.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(entries[0].stat().st_mode) == 0o600
    receipt = json.loads(entries[0].read_text(encoding="utf-8"))
    binding = receipt["binding"]
    assert binding["lock_file_sha256"] == reviewed_sha256
    assert binding["semantic_lock_sha256"] == first["contract_sha256"]
    assert binding["sealed_verifier_code_binding"]["verifier_sha256"].startswith(
        "sha256:"
    )
    assert binding["filesystem_identities"]


@pytest.mark.parametrize("changed_input", ["dependency", "evidence", "verifier"])
def test_filesystem_identity_drift_misses_and_replays_full_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
) -> None:
    lock, verifier, dependency, evidence, calls = _reviewed_lock(tmp_path, monkeypatch)
    reviewed_sha256 = executor._file_sha256(lock)
    executor._verify_lock_with_sealed_runtime(
        lock, reviewed_lock_file_sha256=reviewed_sha256
    )
    target = {"dependency": dependency, "evidence": evidence, "verifier": verifier}[
        changed_input
    ]
    before = target.stat()
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))

    executor._verify_lock_with_sealed_runtime(
        lock, reviewed_lock_file_sha256=reviewed_sha256
    )

    assert len(calls) == 2


def test_code_byte_drift_never_reuses_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, _verifier, dependency, _evidence, calls = _reviewed_lock(
        tmp_path, monkeypatch
    )
    reviewed_sha256 = executor._file_sha256(lock)
    executor._verify_lock_with_sealed_runtime(
        lock, reviewed_lock_file_sha256=reviewed_sha256
    )
    dependency.write_text("SAFE = False\n", encoding="utf-8")

    with pytest.raises(executor.ExecutorError, match="dependency drift before import"):
        executor._verify_lock_with_sealed_runtime(
            lock, reviewed_lock_file_sha256=reviewed_sha256
        )

    assert len(calls) == 1


def test_lock_byte_drift_never_reuses_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, _verifier, _dependency, _evidence, calls = _reviewed_lock(
        tmp_path, monkeypatch
    )
    reviewed_sha256 = executor._file_sha256(lock)
    executor._verify_lock_with_sealed_runtime(
        lock, reviewed_lock_file_sha256=reviewed_sha256
    )
    lock.write_text(lock.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(executor.ExecutorError, match="lock bytes"):
        executor._verify_lock_with_sealed_runtime(
            lock, reviewed_lock_file_sha256=reviewed_sha256
        )

    assert len(calls) == 1
