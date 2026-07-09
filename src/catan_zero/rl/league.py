"""AlphaStar-style PFSP league for multi-agent self-play PPO.

This module implements a prioritized fictitious self-play (PFSP) league with a
payoff matrix, in the spirit of:

  * Vinyals et al. (2019), "Grandmaster level in StarCraft II using
    multi-agent reinforcement learning", Nature -- the League/PFSP section.
    Opponents are sampled in proportion to a function ``f`` of their win-rate
    against the training agent, focusing learning on opponents that beat us.
  * OpenAI Five (2019) -- an 80/20 mix of current-self vs. a pool of past
    frozen opponents to avoid strategy collapse / cycling.

The league is intentionally dependency-light: pure Python + numpy + JSON.  It
holds no torch models and imports no environment; checkpoints are referenced by
path only.  The learner/actor own model loading and game play and merely call
into this registry to choose opponents and record outcomes.

Roles
-----
``main``
    The agent(s) being optimized as the eventual product.  Periodically
    snapshotted (frozen) into the pool every ``snapshot_interval`` steps.
``main_exploiter``
    Trains against the current main(s) to find and fix their weaknesses.  Reset
    to its BC init once it beats its target above ``exploiter_promote_winrate``.
``league_exploiter``
    Trains against the whole frozen league to find global weaknesses.
``frozen``
    A frozen checkpoint snapshot used only as an opponent (never trained).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

EXPLOITER_ROLES = ("main_exploiter", "league_exploiter")
_DEFAULT_PRIOR_WINRATE = 0.5


@dataclass
class LeagueAgent:
    id: str
    role: str  # 'main' | 'main_exploiter' | 'league_exploiter' | 'frozen'
    checkpoint_path: str
    parent_id: str | None
    created_step: int


class League:
    def __init__(
        self,
        *,
        snapshot_interval: int = 200,
        exploiter_promote_winrate: float = 0.7,
        pfsp_p: float = 2.0,
    ) -> None:
        self.snapshot_interval = int(snapshot_interval)
        self.exploiter_promote_winrate = float(exploiter_promote_winrate)
        self.pfsp_p = float(pfsp_p)

        # registry: id -> LeagueAgent
        self._agents: dict[str, LeagueAgent] = {}
        # ordered (a, b) -> [wins_of_a, matches]; wins_of_a accumulates a_score
        self._payoffs: dict[tuple[str, str], list[float]] = {}
        # exploiter id -> target agent id it is trying to beat
        self._exploiter_targets: dict[str, str] = {}
        # monotonic counter for unique id generation
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    # Registry construction
    # ------------------------------------------------------------------ #
    def _new_id(self, role: str) -> str:
        self._counter += 1
        return f"{role}-{self._counter:04d}"

    def add_main(self, checkpoint_path: str, *, step: int = 0) -> LeagueAgent:
        agent = LeagueAgent(
            id=self._new_id("main"),
            role="main",
            checkpoint_path=str(checkpoint_path),
            parent_id=None,
            created_step=int(step),
        )
        self._agents[agent.id] = agent
        return agent

    def add_exploiter(
        self, role: str, init_checkpoint_path: str, *, step: int = 0
    ) -> LeagueAgent:
        if role not in EXPLOITER_ROLES:
            raise ValueError(
                f"exploiter role must be one of {EXPLOITER_ROLES}, got {role!r}"
            )
        agent = LeagueAgent(
            id=self._new_id(role),
            role=role,
            checkpoint_path=str(init_checkpoint_path),
            parent_id=None,
            created_step=int(step),
        )
        self._agents[agent.id] = agent
        return agent

    def snapshot(
        self, agent_id: str, checkpoint_path: str, *, step: int
    ) -> LeagueAgent:
        """Freeze a copy of ``agent_id`` into the opponent pool.

        The returned agent has role ``frozen`` and ``parent_id`` set to the
        source agent; the caller is responsible for having written
        ``checkpoint_path`` to disk before/after this call.
        """
        parent = self.get_agent(agent_id)
        agent = LeagueAgent(
            id=self._new_id("frozen"),
            role="frozen",
            checkpoint_path=str(checkpoint_path),
            parent_id=parent.id,
            created_step=int(step),
        )
        self._agents[agent.id] = agent
        return agent

    def set_exploiter_target(self, exploiter_id: str, target_id: str) -> None:
        """Record which agent an exploiter is being trained to beat."""
        exploiter = self.get_agent(exploiter_id)
        if exploiter.role not in EXPLOITER_ROLES:
            raise ValueError(f"{exploiter_id} is not an exploiter (role={exploiter.role})")
        # validate target exists
        self.get_agent(target_id)
        self._exploiter_targets[exploiter_id] = target_id

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def get_agent(self, agent_id: str) -> LeagueAgent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"unknown agent id: {agent_id!r}") from exc

    def _agents_list(self) -> list[LeagueAgent]:
        return list(self._agents.values())

    def _frozen_pool(self, *, exclude_id: str) -> list[LeagueAgent]:
        """Opponents that can be sampled: everything that is not the live agent.

        Live training agents (main/exploiter) are excluded because we play
        against the *frozen* pool; frozen snapshots are the canonical
        opponents, but other registered agents are allowed too (the live id is
        the only hard exclusion).
        """
        return [a for a in self._agents.values() if a.id != exclude_id]

    # ------------------------------------------------------------------ #
    # Match recording / payoffs
    # ------------------------------------------------------------------ #
    def record_match(self, agent_a_id: str, agent_b_id: str, a_score: float) -> None:
        """Record a game outcome. ``a_score`` in [0,1] (1 = a won, 0.5 = draw)."""
        a_score = float(a_score)
        if not (0.0 <= a_score <= 1.0):
            raise ValueError(f"a_score must be in [0,1], got {a_score}")
        # validate both agents exist
        self.get_agent(agent_a_id)
        self.get_agent(agent_b_id)

        ab = self._payoffs.setdefault((agent_a_id, agent_b_id), [0.0, 0.0])
        ab[0] += a_score
        ab[1] += 1.0
        # store complementary outcome for the reverse ordered pair
        ba = self._payoffs.setdefault((agent_b_id, agent_a_id), [0.0, 0.0])
        ba[0] += 1.0 - a_score
        ba[1] += 1.0

    def winrate(self, a_id: str, b_id: str) -> float | None:
        """Win-rate of ``a`` against ``b``; None if they never played."""
        entry = self._payoffs.get((a_id, b_id))
        if entry is None or entry[1] <= 0.0:
            return None
        return entry[0] / entry[1]

    def payoff_matrix(self) -> tuple[list[str], "np.ndarray"]:
        """Full win-rate grid ``M[i, j] = winrate(ids[i] beats ids[j])``.

        Unseen / self pairs are ``nan``.  Useful for detecting cycling /
        non-transitivity (rock-paper-scissors loops have no dominant row).
        """
        ids = [a.id for a in self._agents.values()]
        n = len(ids)
        matrix = np.full((n, n), np.nan, dtype=np.float64)
        index = {agent_id: i for i, agent_id in enumerate(ids)}
        for (a_id, b_id), (wins, matches) in self._payoffs.items():
            if matches <= 0.0:
                continue
            if a_id == b_id:
                continue
            i = index.get(a_id)
            j = index.get(b_id)
            if i is None or j is None:
                continue
            matrix[i, j] = wins / matches
        return ids, matrix

    # ------------------------------------------------------------------ #
    # Opponent sampling
    # ------------------------------------------------------------------ #
    def sample_opponent(
        self, for_agent_id: str, *, mode: str = "pfsp", rng=None
    ) -> LeagueAgent:
        agent = self.get_agent(for_agent_id)
        if mode == "self":
            return agent

        if rng is None:
            rng = np.random.default_rng()

        pool = self._frozen_pool(exclude_id=for_agent_id)
        if not pool:
            raise ValueError(
                f"no opponents available for {for_agent_id!r} in mode {mode!r}"
            )

        if mode == "uniform":
            idx = int(rng.integers(0, len(pool)))
            return pool[idx]

        if mode in ("pfsp", "pfsp_squared"):
            p = 2.0 if mode == "pfsp_squared" else self.pfsp_p
            weights = np.empty(len(pool), dtype=np.float64)
            for k, opp in enumerate(pool):
                w = self.winrate(for_agent_id, opp.id)
                if w is None:
                    w = _DEFAULT_PRIOR_WINRATE
                # f_hard(w) = (1 - w)^p : prioritize opponents that beat us.
                weights[k] = (1.0 - w) ** p
            total = float(weights.sum())
            if total <= 0.0:
                # all opponents are perfectly beaten -> fall back to uniform
                weights[:] = 1.0
                total = float(weights.sum())
            probs = weights / total
            idx = int(rng.choice(len(pool), p=probs))
            return pool[idx]

        raise ValueError(f"unknown sampling mode: {mode!r}")

    # ------------------------------------------------------------------ #
    # Promotion / snapshot decisions
    # ------------------------------------------------------------------ #
    def should_snapshot_main(self, agent_id: str, step: int) -> bool:
        agent = self.get_agent(agent_id)
        if agent.role != "main":
            return False
        if self.snapshot_interval <= 0:
            return False
        step = int(step)
        if step <= 0:
            return False
        return step % self.snapshot_interval == 0

    def should_reset_exploiter(self, agent_id: str) -> bool:
        """True once the exploiter beats its target above the promote threshold."""
        agent = self.get_agent(agent_id)
        if agent.role not in EXPLOITER_ROLES:
            return False
        target_id = self._exploiter_targets.get(agent_id)
        if target_id is None:
            return False
        w = self.winrate(agent_id, target_id)
        if w is None:
            return False
        return w >= self.exploiter_promote_winrate

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, dir: str) -> None:
        os.makedirs(dir, exist_ok=True)
        payoffs = [
            {"a": a_id, "b": b_id, "wins": wins, "matches": matches}
            for (a_id, b_id), (wins, matches) in self._payoffs.items()
        ]
        state = {
            "config": {
                "snapshot_interval": self.snapshot_interval,
                "exploiter_promote_winrate": self.exploiter_promote_winrate,
                "pfsp_p": self.pfsp_p,
            },
            "counter": self._counter,
            "agents": [asdict(a) for a in self._agents.values()],
            "payoffs": payoffs,
            "exploiter_targets": self._exploiter_targets,
        }
        path = os.path.join(dir, "league.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, dir: str) -> "League":
        path = os.path.join(dir, "league.json")
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        config = state.get("config", {})
        league = cls(
            snapshot_interval=int(config.get("snapshot_interval", 200)),
            exploiter_promote_winrate=float(config.get("exploiter_promote_winrate", 0.7)),
            pfsp_p=float(config.get("pfsp_p", 2.0)),
        )
        league._counter = int(state.get("counter", 0))
        for raw in state.get("agents", []):
            agent = LeagueAgent(
                id=str(raw["id"]),
                role=str(raw["role"]),
                checkpoint_path=str(raw["checkpoint_path"]),
                parent_id=raw.get("parent_id"),
                created_step=int(raw.get("created_step", 0)),
            )
            league._agents[agent.id] = agent
        for row in state.get("payoffs", []):
            league._payoffs[(str(row["a"]), str(row["b"]))] = [
                float(row["wins"]),
                float(row["matches"]),
            ]
        league._exploiter_targets = {
            str(k): str(v) for k, v in state.get("exploiter_targets", {}).items()
        }
        return league
