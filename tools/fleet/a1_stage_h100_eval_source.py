#!/usr/bin/env python3
"""Stage one exact public source revision on every approved H100 eval host.

This is deliberately separate from the evaluator.  It only creates or proves
an immutable detached Git checkout and emits a derived fleet manifest that
points ``remote_repo`` at that checkout.  It never launches an evaluator and
never changes the sealed deployment at ``/home/ubuntu/catan-zero-v1``.
"""

from __future__ import annotations

import argparse
import base64
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fleet import a1_h100_eval_fleet as eval_fleet  # noqa: E402


SOURCE_AUTHORITY_SCHEMA = "a1-h100-eval-source-authority-v1"
HOST_RECEIPT_SCHEMA = "a1-h100-eval-source-host-receipt-v1"
SEALED_DEPLOYMENT = PurePosixPath("/home/ubuntu/catan-zero-v1")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_GITHUB_PATH_RE = re.compile(
    r"/[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/"
    r"[A-Za-z0-9_.-]+\.git\Z"
)
_DESTINATION_RE = re.compile(r"/[A-Za-z0-9_./-]+\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+\Z")


class SourceStageError(RuntimeError):
    """An exact source checkout could not be proved on the whole fleet."""


# The remote program receives one URL-safe base64 JSON token.  Dynamic values
# are never interpolated into a shell program, and every Git call uses argv.
_REMOTE_PROGRAM = r'''from __future__ import annotations
import base64
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import uuid
from urllib.parse import urlsplit

HOST_SCHEMA = "a1-h100-eval-source-host-receipt-v1"
SEALED = PurePosixPath("/home/ubuntu/catan-zero-v1")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
URL_PATH_RE = re.compile(
    r"/[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/"
    r"[A-Za-z0-9_.-]+\.git\Z"
)
DEST_RE = re.compile(r"/[A-Za-z0-9_./-]+\Z")


def fail(message):
    raise RuntimeError(message)


def decode_payload(token):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        fail("payload token is not URL-safe base64")
    token += "=" * (-len(token) % 4)
    try:
        value = json.loads(base64.urlsafe_b64decode(token).decode("utf-8"))
    except Exception as error:
        fail(f"payload is invalid: {error}")
    if not isinstance(value, dict) or set(value) != {
        "git_url", "repo_commit", "destination"
    }:
        fail("payload shape is invalid")
    return value


def validate_url(value):
    if not isinstance(value, str):
        fail("git_url must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or not URL_PATH_RE.fullmatch(parsed.path)
    ):
        fail("git_url must be canonical public GitHub HTTPS URL ending in .git")


def validate_destination(value):
    if not isinstance(value, str) or not DEST_RE.fullmatch(value):
        fail("destination must be a safe absolute POSIX path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {".", ".."} for part in path.parts):
        fail("destination must be lexically canonical")
    if path == PurePosixPath("/"):
        fail("destination cannot be filesystem root")
    if path == SEALED or SEALED in path.parents:
        fail("destination cannot be the sealed deployment or one of its children")


def run_git(*argv, cwd=None):
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", *argv],
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git failed"
        fail(detail)
    return result.stdout.strip()


def check_no_parent_symlinks(destination):
    cursor = Path("/")
    for part in destination.parts[1:-1]:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            fail(f"destination parent is a symlink: {cursor}")


def iter_non_symlinks(root):
    paths = [root, *root.rglob("*")]
    return [path for path in paths if not stat.S_ISLNK(path.lstat().st_mode)]


def make_read_only(root):
    paths = iter_non_symlinks(root)
    for path in reversed(paths):
        mode = stat.S_IMODE(path.lstat().st_mode)
        os.chmod(path, mode & ~0o222, follow_symlinks=False)


def verify_checkout(path, git_url, repo_commit, require_read_only):
    if path.is_symlink() or not path.is_dir() or not (path / ".git").is_dir():
        fail("destination exists but is not a non-symlink Git checkout")
    head = run_git("rev-parse", "--verify", "HEAD", cwd=path)
    if head != repo_commit:
        fail(f"destination HEAD mismatch: expected {repo_commit}, got {head}")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=path,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if symbolic.returncode == 0:
        fail("destination HEAD is attached to a mutable branch")
    if symbolic.returncode != 1:
        fail("cannot prove destination has detached HEAD")
    if run_git("status", "--porcelain=v1", "--untracked-files=all", cwd=path):
        fail("destination worktree is not clean")
    remotes = run_git("remote", cwd=path).splitlines()
    if remotes != ["origin"]:
        fail("destination must have exactly one Git remote named origin")
    urls = run_git("remote", "get-url", "--all", "origin", cwd=path).splitlines()
    if urls != [git_url]:
        fail("destination origin URL differs from the approved public URL")
    gitlinks = [
        line for line in run_git("ls-files", "--stage", cwd=path).splitlines()
        if line.startswith("160000 ")
    ]
    if gitlinks:
        fail("evaluation source checkouts may not contain Git submodules")
    if require_read_only:
        writable = [
            str(item) for item in iter_non_symlinks(path)
            if stat.S_IMODE(item.lstat().st_mode) & 0o222
        ]
        if writable:
            fail(f"destination is not immutable: writable path {writable[0]}")


def stage(payload):
    git_url = payload["git_url"]
    repo_commit = payload["repo_commit"]
    destination_text = payload["destination"]
    validate_url(git_url)
    if not isinstance(repo_commit, str) or not COMMIT_RE.fullmatch(repo_commit):
        fail("repo_commit must be exactly 40 lowercase hexadecimal characters")
    validate_destination(destination_text)
    destination = Path(destination_text)
    check_no_parent_symlinks(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_no_parent_symlinks(destination)
    if destination.is_symlink():
        fail("destination cannot be a symlink")

    lock_path = destination.parent / (destination.name + ".stage.lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    created = False
    try:
        with os.fdopen(lock_fd, "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if destination.exists():
                verify_checkout(destination, git_url, repo_commit, False)
            else:
                temporary = destination.parent / (
                    "." + destination.name + ".stage-" + uuid.uuid4().hex
                )
                try:
                    temporary.mkdir(mode=0o700)
                    run_git("init", "--quiet", str(temporary))
                    run_git("remote", "add", "origin", git_url, cwd=temporary)
                    run_git(
                        "fetch", "--quiet", "--depth=1", "--no-tags",
                        "origin", repo_commit, cwd=temporary
                    )
                    run_git("checkout", "--quiet", "--detach", repo_commit, cwd=temporary)
                    verify_checkout(temporary, git_url, repo_commit, False)
                    make_read_only(temporary)
                    verify_checkout(temporary, git_url, repo_commit, True)
                    if destination.exists():
                        fail("destination appeared concurrently during staging")
                    os.rename(temporary, destination)
                    created = True
                finally:
                    if temporary.exists():
                        for item in iter_non_symlinks(temporary):
                            mode = stat.S_IMODE(item.lstat().st_mode)
                            os.chmod(item, mode | 0o700, follow_symlinks=False)
                        shutil.rmtree(temporary)
            make_read_only(destination)
            verify_checkout(destination, git_url, repo_commit, True)
    finally:
        # Keep the lock file as stable coordination state for idempotent reruns.
        pass
    return {
        "schema_version": HOST_SCHEMA,
        "git_url": git_url,
        "repo_commit": repo_commit,
        "remote_repo": destination_text,
        "clean": True,
        "detached_head": True,
        "read_only": True,
        "created": created,
    }


try:
    if len(sys.argv) != 2:
        fail("expected one encoded staging payload")
    print(json.dumps(stage(decode_payload(sys.argv[1])), sort_keys=True))
except Exception as error:
    print(f"exact source staging failed: {error}", file=sys.stderr)
    raise SystemExit(2)
'''


RemoteRunner = Callable[
    [Sequence[str], str], subprocess.CompletedProcess[str]
]


def _validate_git_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or not _GITHUB_PATH_RE.fullmatch(parsed.path)
    ):
        raise SourceStageError(
            "git URL must be canonical public GitHub HTTPS URL ending in .git"
        )
    return value


def _validate_commit(value: str) -> str:
    if not _COMMIT_RE.fullmatch(value):
        raise SourceStageError(
            "commit must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def _validate_destination(value: str) -> str:
    if not _DESTINATION_RE.fullmatch(value):
        raise SourceStageError("destination must be a safe absolute POSIX path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {".", ".."} for part in path.parts):
        raise SourceStageError("destination must be lexically canonical")
    if path == PurePosixPath("/"):
        raise SourceStageError("destination cannot be filesystem root")
    if path == SEALED_DEPLOYMENT or SEALED_DEPLOYMENT in path.parents:
        raise SourceStageError(
            "destination cannot be the sealed deployment or one of its children"
        )
    return value


def _payload_token(*, git_url: str, repo_commit: str, destination: str) -> str:
    raw = json.dumps(
        {
            "destination": destination,
            "git_url": git_url,
            "repo_commit": repo_commit,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    if not _TOKEN_RE.fullmatch(token):
        raise AssertionError("URL-safe payload encoding produced an unsafe token")
    return token


def _decode_payload_token(token: str) -> dict[str, str]:
    """Decode a remote token for deterministic tests and dry-run inspection."""

    if not _TOKEN_RE.fullmatch(token):
        raise SourceStageError("staging payload token is not URL-safe base64")
    padded = token + "=" * (-len(token) % 4)
    try:
        value = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceStageError(f"invalid staging payload: {error}") from error
    if not isinstance(value, dict):
        raise SourceStageError("staging payload must be one JSON object")
    return {str(key): str(item) for key, item in value.items()}


def _ssh_command(
    manifest: Mapping[str, Any], host: Mapping[str, Any], token: str
) -> list[str]:
    if not _TOKEN_RE.fullmatch(token):
        raise SourceStageError("refusing unsafe remote payload token")
    return [
        "ssh",
        "-i",
        str(manifest["ssh_key"]),
        "-o",
        "BatchMode=yes",
        "-o",
        f"StrictHostKeyChecking={manifest['strict_host_key_checking']}",
        "-o",
        "ConnectTimeout=15",
        f"{manifest['ssh_user']}@{host['address']}",
        "python3",
        "-",
        token,
    ]


def _run_remote(
    command: Sequence[str], program: str
) -> subprocess.CompletedProcess[str]:
    last_error: subprocess.CalledProcessError | None = None
    for _attempt in range(3):
        try:
            return subprocess.run(
                list(command),
                input=program,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as error:
            last_error = error
            if error.returncode != 255:
                raise
    assert last_error is not None
    raise last_error


def _validate_host_receipt(
    value: Any,
    *,
    git_url: str,
    repo_commit: str,
    destination: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SourceStageError("remote staging receipt must be one JSON object")
    expected = {
        "schema_version": HOST_RECEIPT_SCHEMA,
        "git_url": git_url,
        "repo_commit": repo_commit,
        "remote_repo": destination,
        "clean": True,
        "detached_head": True,
        "read_only": True,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise SourceStageError(
                f"remote staging receipt has invalid {key}: {value.get(key)!r}"
            )
    if not isinstance(value.get("created"), bool):
        raise SourceStageError("remote staging receipt created flag is invalid")
    return dict(value)


def _stage_host(
    manifest: Mapping[str, Any],
    host: Mapping[str, Any],
    *,
    git_url: str,
    repo_commit: str,
    destination: str,
    runner: RemoteRunner = _run_remote,
) -> dict[str, Any]:
    token = _payload_token(
        git_url=git_url,
        repo_commit=repo_commit,
        destination=destination,
    )
    command = _ssh_command(manifest, host, token)
    try:
        result = runner(command, _REMOTE_PROGRAM)
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = str(getattr(error, "stderr", "") or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise SourceStageError(
            f"source staging failed on {host['alias']}{detail}"
        ) from error
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SourceStageError(
            f"source staging on {host['alias']} returned {len(lines)} records"
        )
    try:
        raw = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise SourceStageError(
            f"source staging on {host['alias']} returned invalid JSON"
        ) from error
    receipt = _validate_host_receipt(
        raw,
        git_url=git_url,
        repo_commit=repo_commit,
        destination=destination,
    )
    return {"alias": str(host["alias"]), **receipt}


def _source_authority(
    manifest: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    git_url: str,
    repo_commit: str,
    destination: str,
) -> dict[str, Any]:
    hosts = [
        {
            "alias": str(receipt["alias"]),
            "clean": True,
            "detached_head": True,
            "read_only": True,
            "remote_repo": destination,
            "repo_commit": repo_commit,
        }
        for receipt in sorted(receipts, key=lambda item: str(item["alias"]))
    ]
    authority: dict[str, Any] = {
        "schema_version": SOURCE_AUTHORITY_SCHEMA,
        "approved_manifest_hash": str(manifest["manifest_hash"]),
        "git_url": git_url,
        "repo_commit": repo_commit,
        "remote_repo": destination,
        "immutable": True,
        "hosts": hosts,
    }
    authority["authority_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(authority, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return authority


def _derived_manifest(
    manifest: Mapping[str, Any], authority: Mapping[str, Any]
) -> dict[str, Any]:
    derived = copy.deepcopy(dict(manifest))
    derived.pop("manifest_hash", None)
    remote_python = str(derived["remote_python"])
    derived["remote_repo"] = str(authority["remote_repo"])
    derived["remote_python"] = remote_python
    derived["evaluation_source"] = copy.deepcopy(dict(authority))
    return derived


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SourceStageError("derived manifest output cannot be a symlink")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def stage_fleet(
    manifest: Mapping[str, Any],
    *,
    git_url: str,
    repo_commit: str,
    destination: str,
    out: Path,
    parallelism: int = 12,
    runner: RemoteRunner = _run_remote,
    origin_checker: Callable[[], None] | None = None,
) -> dict[str, Any]:
    git_url = _validate_git_url(git_url)
    repo_commit = _validate_commit(repo_commit)
    destination = _validate_destination(destination)
    if parallelism < 1 or parallelism > 64:
        raise SourceStageError("parallelism must be in [1, 64]")
    if origin_checker is not None:
        origin_checker()

    hosts = list(manifest["hosts"])
    receipts: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(parallelism, len(hosts))) as pool:
        futures = {
            pool.submit(
                _stage_host,
                manifest,
                host,
                git_url=git_url,
                repo_commit=repo_commit,
                destination=destination,
                runner=runner,
            ): str(host["alias"])
            for host in hosts
        }
        for future in as_completed(futures):
            try:
                receipts.append(future.result())
            except SourceStageError as error:
                errors.append(f"{futures[future]}: {error}")
    if errors:
        raise SourceStageError(
            "fleet source staging failed; no derived manifest was written: "
            + "; ".join(sorted(errors))
        )
    if {receipt["alias"] for receipt in receipts} != {
        str(host["alias"]) for host in hosts
    }:
        raise SourceStageError("fleet source staging receipts do not cover every host")

    authority = _source_authority(
        manifest,
        receipts,
        git_url=git_url,
        repo_commit=repo_commit,
        destination=destination,
    )
    derived = _derived_manifest(manifest, authority)
    _atomic_write_json(out, derived)
    # Reuse the evaluator's approved-host parser on the final bytes.  It also
    # computes the exact manifest hash that future evaluation plans bind.
    loaded = eval_fleet.load_manifest(out)
    if loaded["remote_repo"] != destination:
        raise SourceStageError("derived manifest remote_repo did not round-trip")
    if loaded["remote_python"] != manifest["remote_python"]:
        raise SourceStageError("derived manifest changed remote_python")
    return {
        "schema_version": SOURCE_AUTHORITY_SCHEMA,
        "authority_sha256": authority["authority_sha256"],
        "derived_manifest": str(out.expanduser().resolve()),
        "derived_manifest_hash": loaded["manifest_hash"],
        "repo_commit": repo_commit,
        "remote_repo": destination,
        "hosts": len(receipts),
        "created": sum(bool(receipt["created"]) for receipt in receipts),
    }


def dry_run(
    manifest: Mapping[str, Any],
    *,
    git_url: str,
    repo_commit: str,
    destination: str,
    out: Path,
) -> dict[str, Any]:
    git_url = _validate_git_url(git_url)
    repo_commit = _validate_commit(repo_commit)
    destination = _validate_destination(destination)
    token = _payload_token(
        git_url=git_url,
        repo_commit=repo_commit,
        destination=destination,
    )
    return {
        "dry_run": True,
        "git_url": git_url,
        "repo_commit": repo_commit,
        "remote_repo": destination,
        "derived_manifest": str(out.expanduser()),
        "hosts": [
            {
                "alias": host["alias"],
                "target": f"{manifest['ssh_user']}@{host['address']}",
                "ssh_command": _ssh_command(manifest, host, token),
            }
            for host in manifest["hosts"]
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--git-url", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=12)
    parser.add_argument(
        "--go",
        action="store_true",
        help="stage source and write the derived manifest; otherwise print a dry-run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = eval_fleet.load_manifest(args.manifest)
        if args.manifest.expanduser().resolve() == args.out.expanduser().resolve():
            raise SourceStageError("derived manifest output must differ from its input")
        if not args.go:
            result = dry_run(
                manifest,
                git_url=args.git_url,
                repo_commit=args.commit,
                destination=args.destination,
                out=args.out,
            )
        else:
            result = stage_fleet(
                manifest,
                git_url=args.git_url,
                repo_commit=args.commit,
                destination=args.destination,
                out=args.out,
                parallelism=args.parallelism,
                origin_checker=eval_fleet._require_b200_origin,  # noqa: SLF001
            )
    except (SourceStageError, eval_fleet.FleetError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
