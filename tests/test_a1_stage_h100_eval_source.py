from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

import pytest

from tools.fleet import a1_h100_eval_fleet as eval_fleet
from tools.fleet import a1_stage_h100_eval_source as stage


GIT_URL = "https://github.com/nickita-khylkouski/catan-zero-public.git"
COMMIT = "a" * 40
DESTINATION = "/home/ubuntu/catan-zero-eval-aaaaaaaaaaaa"


def _manifest_path(tmp_path: Path) -> Path:
    hosts = [
        {"alias": alias, "address": address, "gpu_count": gpu_count}
        for alias, (address, gpu_count) in eval_fleet.EXPECTED_HOSTS.items()
    ]
    value = {
        "schema_version": eval_fleet.MANIFEST_SCHEMA,
        "ssh_user": "ubuntu",
        "ssh_key": str(tmp_path / "id_ed25519"),
        "strict_host_key_checking": "accept-new",
        "remote_repo": "/home/ubuntu/catan-zero-v1",
        "remote_python": "/home/ubuntu/catan-zero-v1/.venv/bin/python",
        "remote_root": "/home/ubuntu/a1-evaluation",
        "validation_ledger": str(tmp_path / "VAL_ONLY_EVAL_LEDGER.jsonl"),
        "hosts": hosts,
    }
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _receipt(*, created: bool) -> dict[str, object]:
    return {
        "schema_version": stage.HOST_RECEIPT_SCHEMA,
        "git_url": GIT_URL,
        "repo_commit": COMMIT,
        "remote_repo": DESTINATION,
        "clean": True,
        "detached_head": True,
        "read_only": True,
        "created": created,
    }


def test_payload_is_data_not_shell_and_round_trips_exactly(tmp_path: Path) -> None:
    manifest = eval_fleet.load_manifest(_manifest_path(tmp_path))
    host = manifest["hosts"][0]
    token = stage._payload_token(  # noqa: SLF001
        git_url=GIT_URL,
        repo_commit=COMMIT,
        destination=DESTINATION,
    )
    command = stage._ssh_command(manifest, host, token)  # noqa: SLF001

    assert re.fullmatch(r"[A-Za-z0-9_-]+", command[-1])
    assert command[-3:-1] == ["python3", "-"]
    assert GIT_URL not in " ".join(command)
    assert DESTINATION not in " ".join(command)
    assert stage._decode_payload_token(command[-1]) == {  # noqa: SLF001
        "destination": DESTINATION,
        "git_url": GIT_URL,
        "repo_commit": COMMIT,
    }


@pytest.mark.parametrize(
    "git_url",
    [
        "git@github.com:nickita-khylkouski/catan-zero-public.git",
        "https://user@github.com/org/repo.git",
        "https://github.com/org/repo",
        "https://github.com/org/repo.git?ref=main",
        "https://example.com/org/repo.git",
        "https://github.com/org/repo.git;touch-pwned",
    ],
)
def test_noncanonical_or_unsafe_git_urls_are_rejected(git_url: str) -> None:
    with pytest.raises(stage.SourceStageError, match="canonical public GitHub"):
        stage._validate_git_url(git_url)  # noqa: SLF001


@pytest.mark.parametrize(
    "destination",
    [
        "/home/ubuntu/catan-zero-v1",
        "/home/ubuntu/catan-zero-v1/eval",
        "relative/eval",
        "/home/ubuntu/eval/../sealed",
        "/home/ubuntu/eval;touch-pwned",
        "/",
    ],
)
def test_unsafe_or_sealed_destinations_are_rejected(destination: str) -> None:
    with pytest.raises(stage.SourceStageError):
        stage._validate_destination(destination)  # noqa: SLF001


def test_stage_fleet_is_parallel_deterministic_and_keeps_remote_python(
    tmp_path: Path,
) -> None:
    manifest = eval_fleet.load_manifest(_manifest_path(tmp_path))
    out = tmp_path / "derived.json"
    lock = threading.Lock()
    calls: list[list[str]] = []
    origin_checks = 0

    def check_origin() -> None:
        nonlocal origin_checks
        origin_checks += 1

    def runner(
        command: list[str], program: str
    ) -> subprocess.CompletedProcess[str]:
        assert program == stage._REMOTE_PROGRAM  # noqa: SLF001
        payload = stage._decode_payload_token(command[-1])  # noqa: SLF001
        assert payload == {
            "destination": DESTINATION,
            "git_url": GIT_URL,
            "repo_commit": COMMIT,
        }
        with lock:
            calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, json.dumps(_receipt(created=True)) + "\n", ""
        )

    result = stage.stage_fleet(
        manifest,
        git_url=GIT_URL,
        repo_commit=COMMIT,
        destination=DESTINATION,
        out=out,
        parallelism=8,
        runner=runner,
        origin_checker=check_origin,
    )

    assert origin_checks == 1
    assert len(calls) == len(eval_fleet.EXPECTED_HOSTS)
    assert result["hosts"] == len(eval_fleet.EXPECTED_HOSTS)
    assert result["created"] == len(eval_fleet.EXPECTED_HOSTS)
    derived = eval_fleet.load_manifest(out)
    assert derived["remote_repo"] == DESTINATION
    assert derived["remote_python"] == manifest["remote_python"]
    assert derived["evaluation_source"]["repo_commit"] == COMMIT
    assert derived["evaluation_source"]["git_url"] == GIT_URL
    assert [
        host["alias"] for host in derived["evaluation_source"]["hosts"]
    ] == sorted(eval_fleet.EXPECTED_HOSTS)

    first_bytes = out.read_bytes()

    def idempotent_runner(
        command: list[str], _program: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 0, json.dumps(_receipt(created=False)) + "\n", ""
        )

    second = stage.stage_fleet(
        manifest,
        git_url=GIT_URL,
        repo_commit=COMMIT,
        destination=DESTINATION,
        out=out,
        parallelism=8,
        runner=idempotent_runner,
    )
    assert second["created"] == 0
    assert out.read_bytes() == first_bytes


def test_any_host_failure_prevents_derived_manifest(tmp_path: Path) -> None:
    manifest = eval_fleet.load_manifest(_manifest_path(tmp_path))
    out = tmp_path / "must-not-exist.json"
    bad_address = eval_fleet.EXPECTED_HOSTS["c3"][0]

    def runner(
        command: list[str], _program: str
    ) -> subprocess.CompletedProcess[str]:
        if any(bad_address in part for part in command):
            raise subprocess.CalledProcessError(
                2, command, stderr="remote exact-source proof failed"
            )
        return subprocess.CompletedProcess(
            command, 0, json.dumps(_receipt(created=True)) + "\n", ""
        )

    with pytest.raises(stage.SourceStageError, match="no derived manifest"):
        stage.stage_fleet(
            manifest,
            git_url=GIT_URL,
            repo_commit=COMMIT,
            destination=DESTINATION,
            out=out,
            runner=runner,
        )
    assert not out.exists()


def _restore_write_bits(root: Path) -> None:
    paths = [root, *root.rglob("*")]
    for path in paths:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        os.chmod(path, mode | (0o700 if path.is_dir() else 0o600))


def test_remote_program_proves_existing_exact_checkout_and_is_idempotent(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "eval-checkout"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Fleet Test"],
        check=True,
    )
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", GIT_URL], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--detach", "-q", commit], check=True
    )
    token = stage._payload_token(  # noqa: SLF001
        git_url=GIT_URL,
        repo_commit=commit,
        destination=str(repo),
    )
    try:
        first = subprocess.run(
            [sys.executable, "-", token],
            input=stage._REMOTE_PROGRAM,  # noqa: SLF001
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert json.loads(first.stdout)["created"] is False
        assert not any(
            path.stat().st_mode & 0o222
            for path in [repo, *repo.rglob("*")]
            if not path.is_symlink()
        )
        # The evaluator's existing preflight uses ordinary ``git status``;
        # proving immutability must not make that proof path unusable.
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert status.stdout == ""
        second = subprocess.run(
            [sys.executable, "-", token],
            input=stage._REMOTE_PROGRAM,  # noqa: SLF001
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert json.loads(second.stdout)["created"] is False
    finally:
        _restore_write_bits(repo)
