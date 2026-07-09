from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_RULE_KEYS = {
    "ruleset_id",
    "game",
    "players",
    "victory_points_to_win",
    "board",
    "turn",
    "trading",
    "robber",
    "hidden_information",
    "benchmark",
}


def load_rules(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    missing = REQUIRED_RULE_KEYS.difference(data)
    if missing:
        raise ValueError(f"rules file missing keys: {sorted(missing)}")
    if data["players"] != 4:
        raise ValueError("CatanBench-4P-Full-v1 requires exactly four players")
    if data["victory_points_to_win"] != 10:
        raise ValueError("CatanBench-4P-Full-v1 requires ten victory points")
    if not data["trading"].get("structured_offers_only"):
        raise ValueError("primary benchmark requires structured trade offers")
    if not data["hidden_information"].get("opponent_resources_hidden"):
        raise ValueError("opponent resources must be hidden")
    return data

