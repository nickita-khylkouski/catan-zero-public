#!/usr/bin/env python3
"""Render Prometheus file-SD targets and localhost-only SSH tunnel units."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
BOX_KEYS = {"name", "host", "cluster", "role", "gpumodel"}
HUB_KEYS = BOX_KEYS - {"host"}


def _validate(config: dict[str, Any], ssh_key: Path) -> None:
    if set(config) != {"hub", "boxes"} or not isinstance(config["boxes"], list):
        raise ValueError("target config must contain exactly hub and boxes")
    hub = config["hub"]
    if not isinstance(hub, dict) or set(hub) != HUB_KEYS:
        raise ValueError(f"hub keys must be exactly {sorted(HUB_KEYS)}")
    names: set[str] = set()
    for index, raw in enumerate([hub, *config["boxes"]]):
        expected = HUB_KEYS if index == 0 else BOX_KEYS
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"target[{index}] keys must be exactly {sorted(expected)}")
        for key in expected:
            if not isinstance(raw[key], str) or not SAFE_LABEL.fullmatch(raw[key]):
                raise ValueError(f"target[{index}].{key} is not a safe label/host")
        if raw["name"] in names:
            raise ValueError(f"duplicate target name {raw['name']!r}")
        names.add(raw["name"])
    if not ssh_key.is_absolute() or any(character.isspace() for character in str(ssh_key)):
        raise ValueError("--ssh-key must be an absolute whitespace-free path")


def _target(box: dict[str, Any], endpoint: str) -> dict[str, Any]:
    # Never use target label `gpu`: DCGM already owns it for the device index.
    labels = {
        key: str(box[key])
        for key in ("name", "cluster", "role", "gpumodel")
    }
    labels["box"] = labels.pop("name")
    return {"targets": [endpoint], "labels": labels}


def render(config: dict[str, Any], *, out_dir: Path, ssh_key: Path) -> dict[str, int]:
    _validate(config, ssh_key)
    hub = dict(config["hub"])
    boxes = [dict(box) for box in config["boxes"]]
    target_dir = out_dir / "targets"
    tunnel_dir = out_dir / "tunnels"
    target_dir.mkdir(parents=True, exist_ok=True)
    tunnel_dir.mkdir(parents=True, exist_ok=True)
    # This directory is generated state. Prune files owned by this tool so a
    # removed box cannot remain deployable through a stale tunnel unit.
    for stale in (
        *target_dir.glob("*.json"),
        *tunnel_dir.glob("fleet-tunnel-*.service"),
        tunnel_dir / "units.list",
    ):
        stale.unlink(missing_ok=True)
    targets = {
        "node": [_target(hub, "localhost:9100")],
        "dcgm": [_target(hub, "localhost:9400")],
        # The hub is monitoring-only. Application exporter targets are the
        # generation boxes, so a deliberately absent hub exporter never raises
        # a false CatanExporterDown alert.
        "catan": [],
    }
    units: list[str] = []
    for index, box in enumerate(boxes, start=1):
        node_port = 19100 + index
        dcgm_port = 19400 + index
        catan_port = 19500 + index
        targets["node"].append(_target(box, f"localhost:{node_port}"))
        targets["dcgm"].append(_target(box, f"localhost:{dcgm_port}"))
        targets["catan"].append(_target(box, f"localhost:{catan_port}"))
        name = box["name"]
        unit_name = f"fleet-tunnel-{name}.service"
        unit = f"""[Unit]
Description=Catan-Zero metrics tunnel to {name}
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
ExecStart=/usr/bin/ssh -N -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -i {ssh_key} -L 127.0.0.1:{node_port}:127.0.0.1:9100 -L 127.0.0.1:{dcgm_port}:127.0.0.1:9400 -L 127.0.0.1:{catan_port}:127.0.0.1:9500 ubuntu@{box['host']}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
        (tunnel_dir / unit_name).write_text(unit, encoding="utf-8")
        units.append(unit_name)
    for job, records in targets.items():
        (target_dir / f"{job}.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (tunnel_dir / "units.list").write_text("\n".join(units) + "\n", encoding="utf-8")
    return {"boxes": len(boxes) + 1, "tunnels": len(units)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--ssh-key", type=Path, default=Path.home() / ".ssh/gpu_access_ed25519"
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = render(config, out_dir=args.out_dir, ssh_key=args.ssh_key)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
