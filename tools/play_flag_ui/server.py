#!/usr/bin/env python3
"""Catan play-and-flag UI server.

Play against Catanatron bots (2p, no domestic trade, 10 VP — matches the
catan-zero production track). Every game is seeded and every executed action
recorded, so any moment converts to (seed, decision_index) for replay,
training, or exam curation. Flag hotkey captures labeled weakness moments.

Run:  .venv/bin/python server.py   →  http://localhost:8765
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CATAN_ZERO_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO / "vendor" / "catanatron" / "catanatron"))

from catanatron.game import Game  # noqa: E402
from catanatron.json import GameEncoder  # noqa: E402
from catanatron.models.enums import (  # noqa: E402
    CITY, ROAD, SETTLEMENT, Action, ActionType,
)
from catanatron.models.player import Color, Player  # noqa: E402
from catanatron.players.minimax import AlphaBetaPlayer  # noqa: E402
from catanatron.players.value import ValueFunctionPlayer  # noqa: E402
from catanatron.state_functions import get_actual_victory_points  # noqa: E402

DATA = ROOT / "data"
GAMES_DIR = DATA / "games"
FLAGS_PATH = DATA / "flags.jsonl"
GAMES_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "opening_placement",
    "second_placement",
    "robber_targeting",
    "knight_timing",
    "dev_timing",
    "maritime_port_use",
    "longest_road_race",
    "largest_army_race",
    "discard_choice",
    "endgame_race",
    "leader_blocking",
    "other",
]


class HumanPlayer(Player):
    """Never auto-decides; the server surfaces playable actions instead."""

    def decide(self, game, playable_actions):  # pragma: no cover
        raise RuntimeError("human player must not be auto-decided")


def make_bot(kind: str, color: Color):
    if kind == "value":
        return ValueFunctionPlayer(color)
    if kind.startswith("ab"):
        return AlphaBetaPlayer(color, depth=int(kind[2:]), prunning=True)
    raise ValueError(f"unknown opponent {kind!r}")


def action_to_json(action: Action):
    value = action.value
    if isinstance(value, Color):
        value = value.value
    elif isinstance(value, tuple):
        value = [
            v.value if isinstance(v, Color) else v for v in value
        ]
    return [action.color.value, action.action_type.value, value]


def describe_action(a: Action) -> str:
    t, v = a.action_type, a.value
    if t == ActionType.ROLL:
        return "Roll dice"
    if t == ActionType.END_TURN:
        return "End turn"
    if t == ActionType.BUILD_SETTLEMENT:
        return f"Build settlement @ node {v}"
    if t == ActionType.BUILD_CITY:
        return f"Build city @ node {v}"
    if t == ActionType.BUILD_ROAD:
        return f"Build road {tuple(v)}"
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        return "Buy development card"
    if t == ActionType.PLAY_KNIGHT_CARD:
        return "Play KNIGHT"
    if t == ActionType.PLAY_MONOPOLY:
        return f"Play MONOPOLY on {v}"
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        return f"Play YEAR OF PLENTY for {list(v)}"
    if t == ActionType.PLAY_ROAD_BUILDING:
        return "Play ROAD BUILDING"
    if t == ActionType.MOVE_ROBBER:
        coord, victim = v[0], v[1]
        who = f", steal from {victim.value if isinstance(victim, Color) else victim}" if victim else ""
        return f"Move robber to {tuple(coord)}{who}"
    if t == ActionType.MARITIME_TRADE:
        give = [x for x in v[:-1] if x is not None]
        return f"Maritime trade {len(give)} {give[0] if give else '?'} → {v[-1]}"
    if t == ActionType.DISCARD_RESOURCE:
        return f"Discard {v}"
    return f"{t.value} {v}"


# -- board geometry + view snapshot (shape shared with the Caratan viewer UI;
# -- math copied from catanatron's own renderer: cube_to_pixel + node deltas) --
import math  # noqa: E402

SQRT3 = math.sqrt(3)
_NODE_DELTA = {
    "NORTH": (0.0, -1.0),
    "NORTHEAST": (SQRT3 / 2, -0.5),
    "SOUTHEAST": (SQRT3 / 2, 0.5),
    "SOUTH": (0.0, 1.0),
    "SOUTHWEST": (-SQRT3 / 2, 0.5),
    "NORTHWEST": (-SQRT3 / 2, -0.5),
}
_DEV_KEYS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")


def tile_center(cube):
    x, _y, z = cube
    return (SQRT3 * x + SQRT3 / 2 * z, 1.5 * z)


def static_board(game) -> dict:
    """Unit-scaled geometry from the serialized board (static per seed)."""
    g = json.loads(json.dumps(game, cls=GameEncoder))
    tiles = []
    for t in g["tiles"]:
        coord = tuple(t["coordinate"])
        tile = t.get("tile", {}) or {}
        cx, cy = tile_center(coord)
        ttype = tile.get("type")
        tiles.append({"coord": list(coord), "x": cx, "y": cy, "type": ttype,
                      "resource": tile.get("resource"), "number": tile.get("number"),
                      "port": (tile.get("resource") or "3:1") if ttype == "PORT" else None,
                      "direction": tile.get("direction") if ttype == "PORT" else None})
    nodes = {}
    for nid, n in g["nodes"].items():
        cx, cy = tile_center(n["tile_coordinate"])
        dx, dy = _NODE_DELTA[n["direction"]]
        nodes[str(nid)] = {"x": cx + dx, "y": cy + dy}
    return {"tiles": tiles, "nodes": nodes, "edges": [list(e["id"]) for e in g["edges"]]}


def snapshot(state, hide_hand_of=None) -> dict:
    """Exact per-ply state from the engine. The bot's hand is redacted to
    public counts server-side (honest-information play)."""
    buildings, roads, hands, vp, awards = {}, {}, {}, {}, {}
    for color in state.colors:
        bc = state.buildings_by_color[color]
        for nid in bc.get(SETTLEMENT, []):
            buildings[str(nid)] = {"color": color.value, "type": "SETTLEMENT"}
        for nid in bc.get(CITY, []):
            buildings[str(nid)] = {"color": color.value, "type": "CITY"}
        for edge in bc.get(ROAD, []):
            a, b = sorted(edge)
            roads[f"{a}-{b}"] = color.value
        i = state.color_to_index[color]
        ps = state.player_state
        if color == hide_hand_of:
            hands[color.value] = {
                "hidden": True,
                "total": sum(ps[f"P{i}_{r}_IN_HAND"] for r in RESOURCES),
                "DEV": sum(ps.get(f"P{i}_{d}_IN_HAND", 0) for d in _DEV_KEYS),
            }
        else:
            hand = {r: ps[f"P{i}_{r}_IN_HAND"] for r in RESOURCES}
            dev = {d: ps.get(f"P{i}_{d}_IN_HAND", 0) for d in _DEV_KEYS}
            hand["DEV"] = sum(dev.values())
            hand["dev_cards"] = dev
            hands[color.value] = hand
        vp[color.value] = get_actual_victory_points(state, color)
        awards[color.value] = {
            "road_len": ps.get(f"P{i}_LONGEST_ROAD_LENGTH", 0),
            "knights": ps.get(f"P{i}_PLAYED_KNIGHT", 0),
            "has_road": bool(ps.get(f"P{i}_HAS_ROAD", False)),
            "has_army": bool(ps.get(f"P{i}_HAS_ARMY", False)),
        }
    robber = state.board.robber_coordinate
    return {"buildings": buildings, "roads": roads,
            "robber": list(robber) if robber else None,
            "hands": hands, "vp": vp, "awards": awards}


def all_rolls(game) -> list:
    """Every dice roll so far (newest last), from the recorded ROLL actions."""
    out = []
    for rec in getattr(game.state, "action_records", []):
        act = rec.action if hasattr(rec, "action") else rec[0]
        if act.action_type == ActionType.ROLL and act.value:
            d = list(act.value)
            out.append({"color": act.color.value, "dice": d, "total": sum(d)})
    return out


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


class Session:
    def __init__(self, seed: int, opponent: str, human_color_name: str):
        self.lock = threading.Lock()
        self.seed = seed
        self.opponent_kind = opponent
        self.human_color = Color[human_color_name]
        self.bot_color = Color.RED if self.human_color == Color.BLUE else Color.BLUE
        self.trace: list[dict] = []  # executed actions (post-execution records)
        self._board_cache = None
        self.started = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = GAMES_DIR / f"{self.started}_seed{seed}.jsonl"
        self._build_game(replay_to=None)
        self._log({"type": "header", "seed": seed, "opponent": opponent,
                   "human_color": self.human_color.value, "vps_to_win": 10,
                   "ts": time.time()})
        self._advance_bots()

    # -- engine ------------------------------------------------------------
    def _build_game(self, replay_to):
        players = sorted(
            [HumanPlayer(self.human_color), make_bot(self.opponent_kind, self.bot_color)],
            key=lambda p: p.color.value,
        )
        self.game = Game(players, seed=self.seed, vps_to_win=10)
        if replay_to is not None:
            for rec in replay_to:
                action = self._action_from_record(rec)
                self.game.execute(action)

    def _action_from_record(self, rec):
        from catanatron.json import action_from_json
        return action_from_json(rec["action"])

    def _record_last_execution(self):
        records = getattr(self.game.state, "action_records", None)
        if not records:
            return
        rec = records[-1]
        action, result = rec.action, rec.result
        try:
            result_json = json.loads(json.dumps(result, default=lambda o: getattr(o, "value", str(o))))
        except Exception:
            result_json = str(result)
        entry = {
            "i": len(self.trace),
            "action": action_to_json(action),
            "result": result_json,
            "desc": describe_action(action),
            "ts": time.time(),
        }
        self.trace.append(entry)
        self._log({"type": "action", **entry})

    def _execute(self, action: Action):
        self.game.execute(action)
        self._record_last_execution()

    def _advance_bots(self):
        guard = 0
        while (self.game.winning_color() is None
               and self.game.state.current_color() != self.human_color):
            guard += 1
            if guard > 2000:
                raise RuntimeError("bot loop did not terminate")
            bot = next(p for p in self.game.state.players
                       if p.color == self.game.state.current_color())
            action = bot.decide(self.game, self.game.playable_actions)
            self._execute(action)
        if self.game.winning_color() is not None:
            self._log({"type": "end", "winner": self.game.winning_color().value,
                       "decisions": len(self.trace), "ts": time.time()})

    # -- api ---------------------------------------------------------------
    def state_payload(self):
        if self._board_cache is None:
            self._board_cache = static_board(self.game)
        human_turn = (self.game.winning_color() is None
                      and self.game.state.current_color() == self.human_color)
        actions = []
        if human_turn:
            for i, a in enumerate(self.game.playable_actions):
                actions.append({"i": i, "type": a.action_type.value,
                                "value": action_to_json(a)[2],
                                "desc": describe_action(a)})
        view = snapshot(self.game.state, hide_hand_of=self.bot_color)
        view["board"] = self._board_cache
        view["rolls"] = all_rolls(self.game)
        return {
            "seed": self.seed,
            "opponent": self.opponent_kind,
            "human_color": self.human_color.value,
            "bot_color": self.bot_color.value,
            "decision_index": len(self.trace),
            "num_turns": self.game.state.num_turns,
            "human_turn": human_turn,
            "winner": (self.game.winning_color().value
                       if self.game.winning_color() else None),
            "actions": actions,
            "log": [{"i": e["i"],
                     "desc": ("Discard a card" if (e["action"][0] == self.bot_color.value
                              and e["action"][1] == "DISCARD_RESOURCE") else e["desc"]),
                     "color": e["action"][0]} for e in self.trace[-40:]],
            "categories": CATEGORIES,
            "view": view,
        }

    def act(self, index: int):
        with self.lock:
            if self.game.state.current_color() != self.human_color:
                raise ValueError("not your turn")
            actions = list(self.game.playable_actions)
            if not (0 <= index < len(actions)):
                raise ValueError("bad action index")
            self._execute(actions[index])
            self._advance_bots()

    def undo(self):
        """Rewind to just before the human's previous decision (replay from seed)."""
        with self.lock:
            human = self.human_color.value
            idx = None
            for j in range(len(self.trace) - 1, -1, -1):
                if self.trace[j]["action"][0] == human:
                    idx = j
                    break
            if idx is None:
                raise ValueError("nothing to undo")
            keep = self.trace[:idx]
            self.trace = []
            self._build_game(replay_to=None)
            for rec in keep:
                self._execute_replay(rec)
            self._log({"type": "undo", "rewound_to": idx, "ts": time.time()})
            self._advance_bots()

    def _execute_replay(self, rec):
        """Re-execute a recorded action.

        Recorded actions carry resolved values (dice results, drawn cards);
        the engine expects the unresolved playable action, with resolution
        reproduced by the seeded RNG stream. Match by type, then value.
        """
        color, type_name, value = rec["action"]
        if color == self.bot_color.value:
            # Bot decide() simulates candidates on game copies, consuming the
            # global RNG stream. Re-run the (deterministic) decision so replay
            # consumes RNG exactly like the original run did.
            bot = next(p for p in self.game.state.players if p.color == self.bot_color)
            action = bot.decide(self.game, self.game.playable_actions)
            if action.action_type.value != type_name:
                raise RuntimeError(
                    f"replay diverged: bot chose {action.action_type.value}, "
                    f"recorded {type_name}")
            self._execute(action)
            return
        candidates = [a for a in self.game.playable_actions
                      if a.color.value == color and a.action_type.value == type_name]
        if not candidates:
            raise RuntimeError(f"replay diverged: no playable {type_name} for {color}")
        chosen = None
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            for a in candidates:
                if action_to_json(a)[2] == value:
                    chosen = a
                    break
            if chosen is None and type_name == "MOVE_ROBBER":
                for a in candidates:
                    av = action_to_json(a)[2]
                    if av and value and av[0] == value[0] and av[1] == value[1]:
                        chosen = a
                        break
        if chosen is None:
            raise RuntimeError(f"replay diverged: ambiguous {type_name}")
        self._execute(chosen)

    def flag(self, category: str, reason: str):
        with self.lock:
            ps = self.game.state.player_state
            entry = {
                "ts": time.time(),
                "game_file": self.log_path.name,
                "seed": self.seed,
                "opponent": self.opponent_kind,
                "human_color": self.human_color.value,
                "decision_index": len(self.trace),
                "category": category if category in CATEGORIES else "other",
                "reason": reason[:400],
                "recent_actions": [e["desc"] for e in self.trace[-6:]],
                "vps": {c: ps.get(f"P{i}_ACTUAL_VICTORY_POINTS")
                        for i, c in enumerate(self.game.state.colors and
                                              [x.value for x in self.game.state.colors])},
            }
            with FLAGS_PATH.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            self._log({"type": "flag", **entry})
            return entry

    def _log(self, obj):
        with self.log_path.open("a") as f:
            f.write(json.dumps(obj) + "\n")


SESSION: Session | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "static" / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif self.path == "/api/state":
            if SESSION is None:
                self._send(200, {"no_game": True, "categories": CATEGORIES})
            else:
                self._send(200, SESSION.state_payload())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global SESSION
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/new":
                seed = int(body.get("seed") or random.randrange(1, 2**31))
                SESSION = Session(seed,
                                  body.get("opponent", "value"),
                                  body.get("human_color", "BLUE"))
                self._send(200, SESSION.state_payload())
            elif self.path == "/api/act":
                SESSION.act(int(body["i"]))
                self._send(200, SESSION.state_payload())
            elif self.path == "/api/undo":
                SESSION.undo()
                self._send(200, SESSION.state_payload())
            elif self.path == "/api/flag":
                entry = SESSION.flag(body.get("category", "other"),
                                     body.get("reason", ""))
                self._send(200, {"ok": True, "flag": entry})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # surface to UI
            self._send(400, {"error": str(e)})


if __name__ == "__main__":
    port = 8765
    print(f"Catan play-and-flag UI → http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
