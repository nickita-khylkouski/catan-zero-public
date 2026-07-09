from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig
from catan_zero.rl.policy_pool import PolicySpec, load_checkpoint_policy, make_policy


def parse_track(
    track: str,
    *,
    vps_to_win: int = 10,
    use_graph_history_features: bool = False,
) -> ColonistMultiAgentConfig:
    normalized = track.lower()
    players = 2 if normalized.startswith("2p") else 4
    config = ColonistMultiAgentConfig(
        players=players,
        vps_to_win=vps_to_win,
        use_graph_history_features=bool(use_graph_history_features),
    )
    if "no_trade" in normalized:
        config = replace(config, max_player_trade_offers_per_turn=0)
    return config


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomic write (CAT-runsix): temp file in the same dir, fsync, os.replace,
    so a reader never sees a half-written JSON. Byte-identical output."""
    import os
    import tempfile
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(output.parent), prefix=f".{output.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, output)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise SystemExit(
                f"{path} is not JSON and PyYAML is not installed; use JSON-compatible YAML"
            ) from error
        return yaml.safe_load(text)


def make_named_policy(name: str, checkpoint: str | None = None, *, device: str | None = None):
    if checkpoint:
        return load_checkpoint_policy(checkpoint, device=device)
    return make_policy(PolicySpec(kind=name), device=device)


def confidence_interval(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    p = wins / games
    half = 1.96 * ((p * (1.0 - p) / games) ** 0.5)
    return max(0.0, p - half), min(1.0, p + half)
