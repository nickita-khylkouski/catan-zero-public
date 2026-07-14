#!/usr/bin/env python3
"""Resolve sealed fleet aliases for the read-only A1 harvest transport.

``a1_harvest_transaction.py`` deliberately accepts one SSH executable.  A
sealed render stores stable host aliases, while cloud hosts are addressed by
the exact fleet manifest.  This tiny exec-only adapter resolves the alias from
that manifest and replaces itself with OpenSSH; it never interprets the remote
command or moves data through the operator machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys


_SAFE_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_ADDRESS = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]*\Z")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: a1_harvest_ssh_transport.py HOST_ALIAS REMOTE_COMMAND")
    manifest_raw = os.environ.get("A1_SSH_FLEET_MANIFEST", "")
    if not manifest_raw:
        raise SystemExit("A1_SSH_FLEET_MANIFEST is required")
    manifest_path = Path(manifest_raw).expanduser().resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "catan-gpu-fleet-v2":
        raise SystemExit("unsupported fleet manifest schema")
    alias = sys.argv[1]
    if _SAFE_ALIAS.fullmatch(alias) is None:
        raise SystemExit("unsafe fleet alias")
    matches = [host for host in payload.get("hosts", []) if host.get("alias") == alias]
    if len(matches) != 1:
        raise SystemExit(f"fleet alias must resolve exactly once: {alias}")
    address = str(matches[0].get("address", ""))
    user = str(payload.get("ssh_user", ""))
    key = Path(str(payload.get("ssh_key", ""))).expanduser()
    checking = str(payload.get("strict_host_key_checking", "accept-new"))
    if (
        _SAFE_ADDRESS.fullmatch(address) is None
        or _SAFE_ALIAS.fullmatch(user) is None
        or not key.is_absolute()
        or checking not in {"yes", "accept-new"}
    ):
        raise SystemExit("fleet SSH transport fields are invalid")
    os.execvp(
        "ssh",
        [
            "ssh",
            "-i",
            str(key),
            "-o",
            f"StrictHostKeyChecking={checking}",
            "-o",
            "ConnectTimeout=20",
            f"{user}@{address}",
            sys.argv[2],
        ],
    )


if __name__ == "__main__":
    main()
