from __future__ import annotations

from dataclasses import dataclass, field
import contextlib
import io
import json
from pathlib import Path
import random
import time
from typing import Any, Protocol

import numpy as np

from catan_zero.rl._catanatron import import_catanatron_module
from catan_zero.rl.action_features import build_action_context_feature_table
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv


class Policy(Protocol):
    name: str

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        ...


@dataclass(slots=True)
class StepSample:
    observation: np.ndarray
    valid_actions: tuple[int, ...]
    action: int
    player: str
    action_context_features: np.ndarray | None = None
    entity_features: dict[str, np.ndarray] | None = None
    phase: str | None = None
    target_policy: dict[int, float] | None = None
    target_scores: dict[int, float] | None = None
    target_score_source: str | None = None
    sample_weight: float = 1.0
    decision_index: int | None = None
    teacher_name: str | None = None
    action_mask_version: str | None = None


@dataclass(slots=True)
class GameResult:
    seed: int
    winner: str | None
    rewards: dict[str, float]
    decisions: int
    terminated: bool
    truncated: bool
    invalid_actions: int
    final_public_vps: dict[str, int]
    final_actual_vps: dict[str, int]


@dataclass(slots=True)
class TrainingEpisode:
    result: GameResult
    samples_by_player: dict[str, list[StepSample]] = field(default_factory=dict)


class RandomPolicy:
    name = "random"

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        valid_actions = tuple(info["valid_actions"])
        return int(valid_actions[int(rng.integers(len(valid_actions)))])


class HeuristicPolicy:
    name = "heuristic"

    _priority = {
        "BUILD_CITY": 100,
        "BUILD_SETTLEMENT": 90,
        "BUILD_ROAD": 70,
        "BUY_DEVELOPMENT_CARD": 60,
        "PLAY_KNIGHT_CARD": 55,
        "PLAY_YEAR_OF_PLENTY": 54,
        "PLAY_MONOPOLY": 53,
        "PLAY_ROAD_BUILDING": 52,
        "offer_trade": 45,
        "MARITIME_TRADE": 42,
        "ROLL": 40,
        "MOVE_ROBBER": 35,
        "DISCARD_RESOURCE": 30,
        "accept_trade": 25,
        "reject_trade": 24,
        "confirm_trade": 23,
        "cancel_trade": 22,
        "END_TURN": 0,
    }

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        actions = tuple(info["structured_legal_actions"])
        scored = [
            (
                self._score(action, env, rng),
                int(action["index"]),
            )
            for action in actions
        ]
        return max(scored, key=lambda item: item[0])[1]

    def target_scores(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        no_noise = _NoNoiseRng()
        return {
            int(action["index"]): float(self._score(action, env, no_noise))
            for action in info["structured_legal_actions"]
        }

    def target_score_source(self) -> str:
        return "heuristic"

    def _score(
        self,
        action: dict[str, Any],
        env: ColonistMultiAgentEnv,
        rng: np.random.Generator,
    ) -> float:
        action_type = action["action_type"]
        score = self._priority.get(action_type, 0)
        args = action["args"]
        if action_type == "BUILD_SETTLEMENT":
            score += 10.0 * _node_production(env, int(args["node"]))
        elif action_type == "BUILD_ROAD":
            score += 2.0 * max(_node_production(env, node) for node in args["edge"])
        elif action_type == "BUILD_CITY":
            score += 10.0 * _node_production(env, int(args["node"]))
        elif action_type == "DISCARD_RESOURCE":
            score += {"wood": 0, "brick": 1, "sheep": 2, "wheat": 3, "ore": 4}.get(
                args["resource"],
                0,
            )
        elif action_type == "MOVE_ROBBER":
            score += 3.0 * _victim_public_vp(env, args["victim"])
        elif action_type == "offer_trade":
            # Avoid huge generosity in the basic baseline.
            score -= 5 * sum(args["give"].values())
            score += 3 * sum(args["want"].values())
        elif action_type == "accept_trade":
            score -= 10
        return score + float(rng.random()) * 1e-3


class JSettlersLitePolicy:
    name = "jsettlers_lite"

    _resources = ("wood", "brick", "sheep", "wheat", "ore")
    _costs = {
        "BUILD_CITY": {"ore": 3, "wheat": 2},
        "BUILD_SETTLEMENT": {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1},
        "BUILD_ROAD": {"wood": 1, "brick": 1},
        "BUY_DEVELOPMENT_CARD": {"sheep": 1, "wheat": 1, "ore": 1},
    }
    _fallback_resource_value = {
        "wood": 1.0,
        "brick": 1.0,
        "sheep": 0.85,
        "wheat": 1.15,
        "ore": 1.25,
    }

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        actions = tuple(info["structured_legal_actions"])
        actor = str(info["current_player"])
        payload = env.observation_payload(actor)["players"][actor]
        resources = _normal_resource_bundle(payload.get("resources", {}))
        plan = self._choose_plan(actions, payload, resources)
        scored = [
            (
                self._score(action, env, info, actor, resources, plan, rng),
                int(action["index"]),
            )
            for action in actions
        ]
        return max(scored, key=lambda item: item[0])[1]

    def _choose_plan(
        self,
        actions: tuple[dict[str, Any], ...],
        payload: dict[str, Any],
        resources: dict[str, int],
    ) -> dict[str, Any]:
        legal_types = {str(action["action_type"]) for action in actions}
        candidates = []
        if int(payload.get("cities_left", 0)) > 0 and "BUILD_CITY" in legal_types:
            candidates.append(("BUILD_CITY", 0))
        if int(payload.get("settlements_left", 0)) > 0:
            candidates.append(("BUILD_SETTLEMENT", 1))
        if int(payload.get("roads_left", 0)) > 0:
            candidates.append(("BUILD_ROAD", 2))
        candidates.append(("BUY_DEVELOPMENT_CARD", 3))
        best = min(
            candidates,
            key=lambda item: (
                _missing_resource_count(resources, self._costs[item[0]]),
                item[1],
            ),
        )[0]
        cost = self._costs[best]
        return {
            "target": best,
            "cost": cost,
            "deficits": _resource_deficits(resources, cost),
        }

    def _score(
        self,
        action: dict[str, Any],
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        actor: str,
        resources: dict[str, int],
        plan: dict[str, Any],
        rng: np.random.Generator,
    ) -> float:
        action_type = str(action["action_type"])
        args = action["args"]
        score = {
            "ROLL": 120.0,
            "BUILD_CITY": 110.0,
            "BUILD_SETTLEMENT": 102.0,
            "BUILD_ROAD": 82.0,
            "BUY_DEVELOPMENT_CARD": 72.0,
            "PLAY_YEAR_OF_PLENTY": 70.0,
            "PLAY_MONOPOLY": 68.0,
            "PLAY_KNIGHT_CARD": 67.0,
            "MARITIME_TRADE": 58.0,
            "offer_trade": 56.0,
            "MOVE_ROBBER": 54.0,
            "DISCARD_RESOURCE": 48.0,
            "confirm_trade": 42.0,
            "accept_trade": 36.0,
            "reject_trade": 34.0,
            "cancel_trade": 32.0,
            "END_TURN": 0.0,
        }.get(action_type, 0.0)

        if action_type == plan["target"]:
            score += 28.0
        if action_type == "BUILD_SETTLEMENT":
            score += 13.0 * _node_production(env, int(args["node"]))
        elif action_type == "BUILD_CITY":
            score += 15.0 * _node_production(env, int(args["node"]))
        elif action_type == "BUILD_ROAD":
            score += 5.0 * _edge_frontier_production(env, args["edge"])
        elif action_type in ("MARITIME_TRADE", "offer_trade"):
            score += self._trade_score(resources, args["give"], args["want"], plan)
        elif action_type == "accept_trade":
            score += self._accept_score(info, actor, resources, plan)
        elif action_type == "reject_trade":
            score -= max(0.0, self._accept_score(info, actor, resources, plan) - 4.0)
        elif action_type == "DISCARD_RESOURCE":
            score += self._discard_score(str(args["resource"]), resources, plan)
        elif action_type == "MOVE_ROBBER":
            score += self._robber_score(env, args.get("victim"))
        elif action_type == "PLAY_YEAR_OF_PLENTY":
            score += self._resource_gain_score(
                resources,
                _resource_tuple_bundle(args.get("resources", ())),
                plan,
            )
        elif action_type == "PLAY_MONOPOLY":
            resource = str(args.get("resource"))
            score += 8.0 * self._resource_need_value(resource, plan)
        elif action_type == "BUY_DEVELOPMENT_CARD":
            score += 4.0 * sum(plan["deficits"].values())
        return score + float(rng.random()) * 1e-3

    def target_scores(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        actions = tuple(info["structured_legal_actions"])
        actor = str(info["current_player"])
        payload = env.observation_payload(actor)["players"][actor]
        resources = _normal_resource_bundle(payload.get("resources", {}))
        plan = self._choose_plan(actions, payload, resources)
        no_noise = _NoNoiseRng()
        return {
            int(action["index"]): float(
                self._score(action, env, info, actor, resources, plan, no_noise)
            )
            for action in actions
        }

    def target_score_source(self) -> str:
        return "jsettlers_lite_heuristic"

    def _trade_score(
        self,
        resources: dict[str, int],
        give: dict[str, Any],
        want: dict[str, Any],
        plan: dict[str, Any],
    ) -> float:
        give_bundle = _normal_resource_bundle(give)
        want_bundle = _normal_resource_bundle(want)
        before = _missing_resource_count(resources, plan["cost"])
        after_resources = {
            resource: resources[resource] - give_bundle[resource] + want_bundle[resource]
            for resource in self._resources
        }
        after = _missing_resource_count(after_resources, plan["cost"])
        progress = before - after
        need_gain = sum(
            want_bundle[resource] * self._resource_need_value(resource, plan)
            for resource in self._resources
        )
        needed_loss = sum(
            give_bundle[resource] * self._resource_need_value(resource, plan)
            for resource in self._resources
        )
        volume = sum(give_bundle.values()) + sum(want_bundle.values())
        return 22.0 * progress + 5.0 * need_gain - 7.0 * needed_loss - 1.5 * volume

    def _accept_score(
        self,
        info: dict[str, Any],
        actor: str,
        resources: dict[str, int],
        plan: dict[str, Any],
    ) -> float:
        best = -12.0
        for offer in info.get("open_negotiation_offers", ()):
            if offer.get("creator") == actor or offer.get("actor") == actor:
                continue
            # From the responder's perspective: receive proposer give, pay proposer want.
            best = max(
                best,
                self._trade_score(
                    resources,
                    offer.get("want", {}),
                    offer.get("give", {}),
                    plan,
                ),
            )
        return best

    def _discard_score(
        self,
        resource: str,
        resources: dict[str, int],
        plan: dict[str, Any],
    ) -> float:
        resource = resource.lower()
        surplus = max(0, resources.get(resource, 0) - int(plan["cost"].get(resource, 0)))
        return 7.0 * surplus - 12.0 * self._resource_need_value(resource, plan)

    def _robber_score(self, env: ColonistMultiAgentEnv, victim: str | None) -> float:
        if victim is None:
            return -20.0
        public_vp = _victim_public_vp(env, victim)
        payload = env.observation_payload(env.current_player_name())["players"]
        resource_count = float(payload.get(victim, {}).get("resource_card_count", 0))
        return 8.0 * public_vp + 1.5 * resource_count

    def _resource_gain_score(
        self,
        resources: dict[str, int],
        gain: dict[str, int],
        plan: dict[str, Any],
    ) -> float:
        before = _missing_resource_count(resources, plan["cost"])
        after_resources = {
            resource: resources[resource] + gain[resource]
            for resource in self._resources
        }
        after = _missing_resource_count(after_resources, plan["cost"])
        return 20.0 * (before - after) + sum(
            gain[resource] * self._resource_need_value(resource, plan)
            for resource in self._resources
        )

    def _resource_need_value(self, resource: str, plan: dict[str, Any]) -> float:
        resource = resource.lower()
        if int(plan["deficits"].get(resource, 0)) > 0:
            return 2.0 + self._fallback_resource_value.get(resource, 1.0)
        if int(plan["cost"].get(resource, 0)) > 0:
            return 1.0 + 0.5 * self._fallback_resource_value.get(resource, 1.0)
        return self._fallback_resource_value.get(resource, 1.0)


class _NoNoiseRng:
    def random(self) -> float:
        return 0.0


class OnePlySearchPolicy:
    name = "one_ply_search"

    def __init__(self, *, candidate_limit: int = 48, rollout_decisions: int = 8) -> None:
        self.candidate_limit = candidate_limit
        self.rollout_decisions = rollout_decisions
        self._heuristic = HeuristicPolicy()

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        actor_color = env.current_player_color()
        candidates = self._candidate_actions(env, info, rng)
        if not candidates:
            return self._heuristic.select_action(env, observation, info, rng)

        root_random_state = random.getstate()
        try:
            scored = []
            for action_index in candidates:
                random.setstate(root_random_state)
                scored.append(
                    (
                        self._score_action(env, action_index, actor_color),
                        action_index,
                    )
                )
        finally:
            random.setstate(root_random_state)
        return max(scored, key=lambda item: item[0])[1]

    def _candidate_actions(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        structured = tuple(info["structured_legal_actions"])
        ordered = sorted(
            structured,
            key=lambda action: self._heuristic._score(action, env, rng),
            reverse=True,
        )
        return tuple(int(action["index"]) for action in ordered[: self.candidate_limit])

    def _score_action(
        self,
        env: ColonistMultiAgentEnv,
        action_index: int,
        actor_color: Any,
    ) -> float:
        action = env._decode_action(action_index)
        if action is None:
            return float("-inf")
        game_copy = env.game.copy()
        try:
            game_copy.execute(action)
        except Exception:
            return float("-inf")
        _rollout_game(game_copy, max_decisions=self.rollout_decisions)
        return _score_game_for_color(game_copy, actor_color)


# G2 roster style specialists (task #3): reweightings of the catanatron value
# function's params dict (consumed by base_fn in catanatron.players.value).
# public_vps/production/enemy_production keep their structural magnitudes so
# that actually WINNING always dominates any style term; only the smaller
# "how to get there" weights are reshaped to induce a recognizable style. Keys
# mirror value.py's DEFAULT_WEIGHTS/CONTENDER_WEIGHTS.
_CONTENDER_BASE_WEIGHTS = {
    "public_vps": 300000000000001.94,
    "production": 100000002.04188395,
    "enemy_production": -99999998.03389844,
    "num_tiles": 2.91440418,
    "reachable_production_0": 2.03820085,
    "reachable_production_1": 10002.018773150001,
    "buildable_nodes": 1001.86278466,
    "longest_road": 12.127388499999999,
    "hand_synergy": 102.40606877,
    "hand_resources": 2.43644327,
    "discard_penalty": -3.00141993,
    "hand_devs": 10.721669799999999,
    "army_size": 12.93844622,
}


def _specialist_weights(**overrides: float) -> dict[str, float]:
    weights = dict(_CONTENDER_BASE_WEIGHTS)
    weights.update(overrides)
    return weights


STYLE_SPECIALIST_WEIGHTS: dict[str, dict[str, float]] = {
    # Ore/city + dev-card engine: value the dev-card/army track heavily and
    # damp expansion so it upgrades settlements and buys development cards
    # rather than sprawling.
    "ore_city": _specialist_weights(
        hand_devs=40.0, army_size=45.0, reachable_production_1=3000.0,
        buildable_nodes=300.0, longest_road=3.0,
    ),
    # Road race: maximize longest road and expansion, ignore the dev/army track.
    "road_race": _specialist_weights(
        longest_road=120.0, buildable_nodes=4000.0, reachable_production_1=40000.0,
        num_tiles=12.0, hand_devs=2.0, army_size=2.0,
    ),
    # Robber-aggressive: weight suppressing enemy production and the knight/army
    # track (knights relocate the robber) far above baseline.
    "robber": _specialist_weights(
        enemy_production=-300000000.0, army_size=60.0, hand_devs=45.0,
    ),
}


class CatanatronWeightedRandomPolicy:
    """Weighted-random floor opponent (task #3): catanatron's WeightedRandomPlayer
    skew -- cities >> settlements >> dev cards >> everything else -- which is a
    touch above uniform random and serves as the G2 roster's floor anchor. We
    reuse the vendored WEIGHTS_BY_ACTION_TYPE table (so the skew stays in sync
    with catanatron) but sample through the harness rng rather than the global
    `random` module, so paired-seed games stay reproducible, and we weight over
    the ENCODABLE playable actions directly so the choice is always a legal
    action index for our env."""

    name = "catanatron_weighted_random"

    def __init__(self) -> None:
        module = import_catanatron_module("catanatron.players.weighted_random")
        self._weights_by_action_type = dict(module.WEIGHTS_BY_ACTION_TYPE)

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        valid = set(env.valid_actions())
        indices: list[int] = []
        weights: list[float] = []
        for action in env.game.playable_actions:
            index = env.action_catalog.try_encode(action)
            if index is None or int(index) not in valid:
                continue
            indices.append(int(index))
            weights.append(float(self._weights_by_action_type.get(action.action_type, 1)))
        if not indices:
            valid_actions = tuple(int(action) for action in info["valid_actions"])
            return int(valid_actions[int(rng.integers(len(valid_actions)))])
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        return int(rng.choice(indices, p=probabilities))


class CatanatronValuePolicy:
    name = "catanatron_value"

    def __init__(
        self,
        *,
        candidate_limit: int = 96,
        opponent_penalty: float = 0.05,
        distillation_temperature: float = 0.7,
        value_fn_builder_name: str = "contender_fn",
        params: dict[str, float] | None = None,
        name: str | None = None,
    ) -> None:
        self.candidate_limit = candidate_limit
        self.opponent_penalty = opponent_penalty
        self.distillation_temperature = distillation_temperature
        # Style-specialist support (G2 roster): the catanatron value function
        # is `base_fn(params=WEIGHTS)`, so injecting a reweighted params dict
        # (see STYLE_SPECIALIST_WEIGHTS) yields a distinct playing style
        # without any new bot code. `value_fn_builder_name` selects the
        # builder ("contender_fn" tuned weights, or "base_fn"); when `params`
        # is given it overrides the builder's default weights.
        self.value_fn_builder_name = value_fn_builder_name
        self.params = params
        if name is not None:
            self.name = name
        self._heuristic = HeuristicPolicy()
        self._value_fn = None

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        actor_color = env.current_player_color()
        candidates = self._candidate_actions(env, info, rng)
        if not candidates:
            return self._heuristic.select_action(env, observation, info, rng)
        return max(
            self._score_candidates(env, candidates, actor_color),
            key=lambda item: item[0],
        )[1]

    def target_policy(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        actor_color = env.current_player_color()
        candidates = self._candidate_actions(env, info, rng)
        if not candidates:
            return {}
        scored = self._score_candidates(env, candidates, actor_color)
        finite = [(score, action) for score, action in scored if np.isfinite(score)]
        if not finite:
            return {int(candidates[0]): 1.0}
        scores = np.asarray([score for score, _ in finite], dtype=np.float64)
        std = float(scores.std())
        if std <= 1e-9:
            logits = np.zeros_like(scores)
        else:
            logits = (scores - float(scores.mean())) / std
        temperature = max(float(self.distillation_temperature), 1e-6)
        logits = np.clip(logits / temperature, -30.0, 30.0)
        logits -= float(logits.max())
        weights = np.exp(logits)
        weights /= float(weights.sum())
        return {
            int(action): float(weight)
            for weight, (_, action) in zip(weights, finite)
            if weight > 0.0
        }

    def target_scores(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        actor_color = env.current_player_color()
        candidates = self._candidate_actions(env, info, rng)
        if not candidates:
            return {}
        scored = self._score_candidates(env, candidates, actor_color)
        return {
            int(action): float(score)
            for score, action in scored
            if np.isfinite(float(score))
        }

    def target_score_source(self) -> str:
        return "catanatron_value"

    def _candidate_actions(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        structured = tuple(info["structured_legal_actions"])
        no_noise = _NoNoiseRng()
        ordered = sorted(
            structured,
            key=lambda action: (
                self._heuristic._score(action, env, no_noise),
                -int(action["index"]),
            ),
            reverse=True,
        )
        return tuple(int(action["index"]) for action in ordered[: self.candidate_limit])

    def _score_candidates(
        self,
        env: ColonistMultiAgentEnv,
        candidates: tuple[int, ...],
        actor_color: Any,
    ) -> list[tuple[float, int]]:
        root_random_state = random.getstate()
        try:
            scored = []
            for action_index in candidates:
                random.setstate(root_random_state)
                scored.append(
                    (
                        self._score_action(env, action_index, actor_color),
                        action_index,
                    )
                )
            return scored
        finally:
            random.setstate(root_random_state)

    def _score_action(
        self,
        env: ColonistMultiAgentEnv,
        action_index: int,
        actor_color: Any,
    ) -> float:
        action = env._decode_action(action_index)
        if action is None:
            return float("-inf")
        game_copy = env.game.copy()
        try:
            game_copy.execute(action)
        except Exception:
            return float("-inf")
        return _catanatron_value_score(
            game_copy,
            actor_color,
            opponent_penalty=self.opponent_penalty,
            value_fn=self._get_value_fn(),
        )

    def _get_value_fn(self):
        if self._value_fn is None:
            from catanatron.players.value import get_value_fn

            self._value_fn = get_value_fn(self.value_fn_builder_name, self.params)
        return self._value_fn


class CatanatronAlphaBetaPolicy:
    name = "catanatron_ab3"

    def __init__(
        self,
        *,
        depth: int = 3,
        prunning: bool = True,
        value_fn_builder_name: str | None = None,
        candidate_limit: int = 96,
        ab_anchor_weight: float = 0.70,
    ) -> None:
        minimax_module = import_catanatron_module("catanatron.players.minimax")
        self._player_cls = minimax_module.AlphaBetaPlayer
        self._debug_state_node_cls = minimax_module.DebugStateNode
        self.depth = int(depth)
        self.name = f"catanatron_ab{self.depth}"
        self.prunning = bool(prunning)
        self.value_fn_builder_name = value_fn_builder_name
        self._players: dict[Any, Any] = {}
        self.candidate_limit = int(candidate_limit)
        self.ab_anchor_weight = min(max(float(ab_anchor_weight), 0.0), 1.0)
        self._fallback = CatanatronValuePolicy(candidate_limit=candidate_limit)
        self._last_action_key: tuple[Any, ...] | None = None
        self._last_action: int | None = None
        self._last_scores_key: tuple[Any, ...] | None = None
        self._last_scores: dict[int, float] | None = None

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        key = self._cache_key(env, info)
        scores: dict[int, float] = {}
        try:
            action_index, scores = self._root_search(env, info)
        except Exception:
            action_index = None
        if action_index is None:
            action_index = self._fallback.select_action(env, observation, info, rng, training=training)
            scores = {}
        action_index = int(action_index)
        self._last_action_key = key
        self._last_action = action_index
        self._last_scores_key = key
        self._last_scores = dict(scores)
        return action_index

    def target_policy(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        scores = self.target_scores(env, info, rng)
        chosen = self._cached_action(env, info)
        if chosen is None:
            return {}
        valid = tuple(int(action) for action in info["valid_actions"])
        if chosen not in valid:
            return {}
        if not scores:
            return {int(chosen): 1.0}

        score_policy = _score_dict_to_policy(scores)
        if not score_policy:
            return {int(chosen): 1.0}
        anchored = {
            int(action): (1.0 - self.ab_anchor_weight) * float(weight)
            for action, weight in score_policy.items()
        }
        anchored[int(chosen)] = anchored.get(int(chosen), 0.0) + self.ab_anchor_weight
        total = sum(anchored.values())
        if total <= 0.0:
            return {int(chosen): 1.0}
        return {
            int(action): float(weight / total)
            for action, weight in anchored.items()
            if weight > 0.0
        }

    def target_scores(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        del rng
        try:
            _, scores = self._root_search(env, info)
        except Exception:
            scores = {}
        return dict(scores)

    def target_score_source(self) -> str:
        return "ab_root"

    def _root_search(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
    ) -> tuple[int | None, dict[int, float]]:
        key = self._cache_key(env, info)
        if self._last_scores_key == key and self._last_scores is not None:
            return self._last_action, dict(self._last_scores)

        color = env.current_player_color()
        player = self._players.get(color)
        if player is None:
            player = self._player_cls(
                color,
                depth=self.depth,
                prunning=self.prunning,
                value_fn_builder_name=self.value_fn_builder_name,
                # FIX A4: this policy's whole purpose is distilling AB search into soft
                # policy/value TARGETS for every root child, so the root ply must not collapse
                # to the single "most impactful" MOVE_ROBBER action that list_prunned_actions
                # picks -- see AlphaBetaPlayer.full_width_root in minimax.py.
                full_width_root=True,
            )
            self._players[color] = player

        state_id = str(len(env.game.state.action_records))
        node = self._debug_state_node_cls(state_id, color)
        deadline = time.time() + 20.0
        root_random_state = random.getstate()
        try:
            chosen, _ = player.alphabeta(
                env.game.copy(),
                self.depth,
                float("-inf"),
                float("inf"),
                deadline,
                node,
            )
        finally:
            random.setstate(root_random_state)
        action_index = self._encode_native_action(env, chosen) if chosen is not None else None
        scores: dict[int, float] = {}
        for action_node in getattr(node, "children", ()):
            encoded = self._encode_native_action(env, getattr(action_node, "action", None))
            score = getattr(action_node, "expected_value", None)
            if encoded is not None and score is not None and np.isfinite(float(score)):
                scores[int(encoded)] = float(score)

        self._last_action_key = key
        self._last_action = None if action_index is None else int(action_index)
        self._last_scores_key = key
        self._last_scores = dict(scores)
        return self._last_action, dict(scores)

    def _encode_native_action(
        self,
        env: ColonistMultiAgentEnv,
        action: Any,
    ) -> int | None:
        valid = set(env.valid_actions())
        index = env.action_catalog.try_encode(action)
        if index is not None and index in valid:
            return int(index)
        for candidate in env._trade_response_indices_for(action):
            if int(candidate) in valid:
                return int(candidate)
        return None

    def _cache_key(self, env: ColonistMultiAgentEnv, info: dict[str, Any]) -> tuple[Any, ...]:
        return (
            id(env.game),
            len(getattr(env.game.state, "action_records", ())),
            str(info.get("current_player", "")),
            tuple(int(action) for action in info.get("valid_actions", ())),
        )

    def _cached_action(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
    ) -> int | None:
        if self._last_action_key == self._cache_key(env, info):
            return self._last_action
        return None


class CatanatronSameTurnAlphaBetaPolicy(CatanatronAlphaBetaPolicy):
    name = "catanatron_sab4"

    def __init__(
        self,
        *,
        depth: int = 4,
        prunning: bool = True,
        value_fn_builder_name: str | None = None,
        candidate_limit: int = 96,
        ab_anchor_weight: float = 0.70,
    ) -> None:
        super().__init__(
            depth=depth,
            prunning=prunning,
            value_fn_builder_name=value_fn_builder_name,
            candidate_limit=candidate_limit,
            ab_anchor_weight=ab_anchor_weight,
        )
        minimax_module = import_catanatron_module("catanatron.players.minimax")
        self._player_cls = minimax_module.SameTurnAlphaBetaPlayer
        self.name = f"catanatron_sab{self.depth}"


class _CatanatronNativeSearchPolicy:
    def __init__(self, *, name: str, module_name: str, class_name: str) -> None:
        self.name = name
        module = import_catanatron_module(module_name)
        self._player_cls = getattr(module, class_name)
        self._args: tuple[Any, ...] = ()
        self._players: dict[Any, Any] = {}
        self._last_action_key: tuple[Any, ...] | None = None
        self._last_action: int | None = None

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        del observation, rng, training
        key = self._cache_key(env, info)
        color = env.current_player_color()
        player = self._players.get(color)
        if player is None:
            player = self._player_cls(color, *self._args)
            self._players[color] = player

        root_random_state = random.getstate()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                native_action = player.decide(env.game.copy(), tuple(env.game.playable_actions))
        finally:
            random.setstate(root_random_state)
        encoded = self._encode_native_action(env, native_action)
        if encoded is None:
            valid_actions = tuple(int(action) for action in info["valid_actions"])
            encoded = valid_actions[0]
        self._last_action_key = key
        self._last_action = int(encoded)
        return int(encoded)

    def target_policy(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        del rng
        if self._last_action_key != self._cache_key(env, info) or self._last_action is None:
            return {}
        if int(self._last_action) not in set(int(action) for action in info["valid_actions"]):
            return {}
        return {int(self._last_action): 1.0}

    def target_score_source(self) -> str:
        return self.name

    def _encode_native_action(
        self,
        env: ColonistMultiAgentEnv,
        action: Any,
    ) -> int | None:
        valid = set(env.valid_actions())
        index = env.action_catalog.try_encode(action)
        if index is not None and int(index) in valid:
            return int(index)
        for candidate in env._trade_response_indices_for(action):
            if int(candidate) in valid:
                return int(candidate)
        return None

    def _cache_key(self, env: ColonistMultiAgentEnv, info: dict[str, Any]) -> tuple[Any, ...]:
        return (
            id(env.game),
            len(getattr(env.game.state, "action_records", ())),
            str(info.get("current_player", "")),
            tuple(int(action) for action in info.get("valid_actions", ())),
        )


class CatanatronMCTSPolicy(_CatanatronNativeSearchPolicy):
    def __init__(self, *, simulations: int = 100, prunning: bool = False) -> None:
        super().__init__(
            name=f"catanatron_mcts{int(simulations)}",
            module_name="catanatron.players.mcts",
            class_name="MCTSPlayer",
        )
        self._args = (int(simulations), bool(prunning))


class CatanatronGreedyPlayoutsPolicy(_CatanatronNativeSearchPolicy):
    def __init__(self, *, playouts: int = 25) -> None:
        playouts_module = import_catanatron_module("catanatron.players.playouts")
        # Avoid nested multiprocessing: the teacher generator already uses one
        # process per CPU worker inside each Modal container.
        playouts_module.USE_MULTIPROCESSING = False
        super().__init__(
            name=f"catanatron_greedy{int(playouts)}",
            module_name="catanatron.players.playouts",
            class_name="GreedyPlayoutsPlayer",
        )
        self._args = (int(playouts),)


class ValueRolloutSearchPolicy:
    name = "value_rollout_search"

    def __init__(
        self,
        *,
        candidate_limit: int = 24,
        presearch_candidate_limit: int | None = None,
        rollout_decisions: int = 6,
        rollout_samples: int = 1,
        root_value_weight: float = 0.0,
        opponent_penalty: float = 0.05,
        distillation_temperature: float = 0.7,
    ) -> None:
        self.candidate_limit = candidate_limit
        self.presearch_candidate_limit = max(
            candidate_limit,
            int(presearch_candidate_limit)
            if presearch_candidate_limit is not None
            else candidate_limit,
        )
        self.rollout_decisions = rollout_decisions
        self.rollout_samples = max(int(rollout_samples), 1)
        self.root_value_weight = min(max(float(root_value_weight), 0.0), 1.0)
        self.opponent_penalty = opponent_penalty
        self.distillation_temperature = distillation_temperature
        self._value_policy = CatanatronValuePolicy(
            candidate_limit=self.presearch_candidate_limit,
            opponent_penalty=opponent_penalty,
        )
        self._score_cache_key: tuple[Any, ...] | None = None
        self._score_cache: list[tuple[float, int]] | None = None

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        actor_color = env.current_player_color()
        candidates = self._candidate_actions(env, info, rng, actor_color)
        if not candidates:
            return self._value_policy.select_action(env, observation, info, rng)

        scored = self._score_candidates(
            env,
            info,
            candidates,
            actor_color,
        )
        return max(scored, key=lambda item: item[0])[1]

    def target_policy(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        actor_color = env.current_player_color()
        candidates = self._candidate_actions(env, info, rng, actor_color)
        if not candidates:
            return {}
        scored = self._score_candidates(
            env,
            info,
            candidates,
            actor_color,
        )
        finite = [(score, action) for score, action in scored if np.isfinite(score)]
        if not finite:
            return {int(candidates[0]): 1.0}
        scores = np.asarray([score for score, _ in finite], dtype=np.float64)
        std = float(scores.std())
        logits = np.zeros_like(scores) if std <= 1e-9 else (scores - float(scores.mean())) / std
        temperature = max(float(self.distillation_temperature), 1e-6)
        logits = np.clip(logits / temperature, -30.0, 30.0)
        logits -= float(logits.max())
        weights = np.exp(logits)
        weights /= float(weights.sum())
        return {
            int(action): float(weight)
            for weight, (_, action) in zip(weights, finite)
            if weight > 0.0
        }

    def target_scores(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[int, float]:
        actor_color = env.current_player_color()
        candidates = self._candidate_actions(env, info, rng, actor_color)
        if not candidates:
            return {}
        scored = self._score_candidates(
            env,
            info,
            candidates,
            actor_color,
        )
        return {
            int(action): float(score)
            for score, action in scored
            if np.isfinite(float(score))
        }

    def target_score_source(self) -> str:
        return "value_rollout_search"

    def _candidate_actions(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
        actor_color: Any,
    ) -> tuple[int, ...]:
        candidates = self._value_policy._candidate_actions(env, info, rng)
        if len(candidates) <= self.candidate_limit:
            return candidates
        if self._should_score_full_candidate_set(info, candidates):
            return tuple(int(action) for action in candidates)

        scored = self._value_policy._score_candidates(env, candidates, actor_color)
        finite = [(score, action) for score, action in scored if np.isfinite(score)]
        if not finite:
            return tuple(candidates[: self.candidate_limit])
        ordered = sorted(
            finite,
            key=lambda item: (item[0], -int(item[1])),
            reverse=True,
        )
        return tuple(int(action) for _, action in ordered[: self.candidate_limit])

    def _should_score_full_candidate_set(
        self,
        info: dict[str, Any],
        candidates: tuple[int, ...],
    ) -> bool:
        if len(candidates) > self.presearch_candidate_limit:
            return False
        prompt = str(info.get("current_prompt", ""))
        if "INITIAL" in prompt:
            return True
        legal_by_index = {
            int(action["index"]): str(action["action_type"])
            for action in info.get("structured_legal_actions", ())
            if isinstance(action, dict) and "index" in action
        }
        candidate_types = {
            legal_by_index[action]
            for action in candidates
            if action in legal_by_index
        }
        return bool(candidate_types) and candidate_types <= {
            "BUILD_SETTLEMENT",
            "BUILD_ROAD",
        }

    def _score_candidates(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        candidates: tuple[int, ...],
        actor_color: Any,
    ) -> list[tuple[float, int]]:
        cache_key = (
            id(env.game),
            id(info),
            actor_color,
            candidates,
            self.presearch_candidate_limit,
            self.rollout_decisions,
            self.rollout_samples,
            self.root_value_weight,
            self.opponent_penalty,
        )
        if self._score_cache_key == cache_key and self._score_cache is not None:
            return list(self._score_cache)

        value_fn = self._value_policy._get_value_fn()
        root_random_state = random.getstate()
        try:
            scored = []
            for action_index in candidates:
                root_score = self._score_root_action(
                    env,
                    action_index,
                    actor_color,
                    value_fn=value_fn,
                )
                action_scores = []
                for sample_index in range(self.rollout_samples):
                    random.setstate(root_random_state)
                    for _ in range(sample_index):
                        random.random()
                    action_scores.append(
                        self._score_action(
                            env,
                            action_index,
                            actor_color,
                            value_fn=value_fn,
                        )
                    )
                finite_scores = [
                    score for score in action_scores if np.isfinite(float(score))
                ]
                score = (
                    float(np.mean(finite_scores))
                    if finite_scores
                    else float("-inf")
                )
                if self.root_value_weight > 0.0 and np.isfinite(root_score):
                    score = (
                        self.root_value_weight * float(root_score)
                        + (1.0 - self.root_value_weight) * score
                    )
                scored.append(
                    (
                        score,
                        action_index,
                    )
                )
            return scored
        finally:
            random.setstate(root_random_state)
            self._score_cache_key = cache_key
            self._score_cache = list(scored)

    def _score_root_action(
        self,
        env: ColonistMultiAgentEnv,
        action_index: int,
        actor_color: Any,
        *,
        value_fn: Any,
    ) -> float:
        action = env._decode_action(action_index)
        if action is None:
            return float("-inf")
        game_copy = env.game.copy()
        try:
            game_copy.execute(action)
        except Exception:
            return float("-inf")
        return _catanatron_value_score(
            game_copy,
            actor_color,
            opponent_penalty=self.opponent_penalty,
            value_fn=value_fn,
        )

    def _score_action(
        self,
        env: ColonistMultiAgentEnv,
        action_index: int,
        actor_color: Any,
        *,
        value_fn: Any,
    ) -> float:
        action = env._decode_action(action_index)
        if action is None:
            return float("-inf")
        game_copy = env.game.copy()
        try:
            game_copy.execute(action)
        except Exception:
            return float("-inf")
        _rollout_game_with_value(
            game_copy,
            max_decisions=self.rollout_decisions,
            value_fn=value_fn,
            opponent_penalty=self.opponent_penalty,
        )
        return _catanatron_value_score(
            game_copy,
            actor_color,
            opponent_penalty=self.opponent_penalty,
            value_fn=value_fn,
        )


class LinearSoftmaxPolicy:
    name = "linear_softmax"

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
        entropy_bonus: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.entropy_bonus = entropy_bonus
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.001, size=(observation_size, action_size))
        self.bias = np.zeros(action_size, dtype=np.float64)
        self.reward_baseline = 0.0
        self.baseline_beta = 0.95

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        valid = tuple(info["valid_actions"])
        probs = self.action_probs(observation, valid)
        if training:
            return int(rng.choice(np.asarray(valid), p=probs))
        return int(valid[int(np.argmax(probs))])

    def action_probs(
        self,
        observation: np.ndarray,
        valid_actions: tuple[int, ...],
    ) -> np.ndarray:
        x = _normalize_observation(observation)
        logits = x @ self.weights[:, valid_actions] + self.bias[list(valid_actions)]
        logits = logits / max(self.temperature, 1e-6)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def update_episode(
        self,
        episode: TrainingEpisode,
        *,
        l2: float = 1e-5,
    ) -> None:
        average_reward = float(np.mean(tuple(episode.result.rewards.values())))
        self.reward_baseline = (
            self.baseline_beta * self.reward_baseline
            + (1.0 - self.baseline_beta) * average_reward
        )
        for player, samples in episode.samples_by_player.items():
            advantage = episode.result.rewards[player] - self.reward_baseline
            for sample in samples:
                self._update_sample(sample, advantage, l2=l2)

    def update_imitation(
        self,
        episode: TrainingEpisode,
        *,
        strength: float = 1.0,
        l2: float = 1e-5,
    ) -> None:
        for samples in episode.samples_by_player.values():
            for sample in samples:
                self._update_sample(sample, strength, l2=l2)

    def _update_sample(self, sample: StepSample, advantage: float, *, l2: float) -> None:
        if not sample.valid_actions:
            return
        x = _normalize_observation(sample.observation)
        valid = sample.valid_actions
        probs = self.action_probs(sample.observation, valid)
        grad = -probs
        chosen_index = valid.index(sample.action)
        grad[chosen_index] += 1.0
        scale = self.learning_rate * float(advantage)
        for local_idx, action in enumerate(valid):
            self.weights[:, action] += scale * grad[local_idx] * x
            self.bias[action] += scale * grad[local_idx]
        self.weights *= 1.0 - self.learning_rate * l2

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            observation_size=self.observation_size,
            action_size=self.action_size,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            entropy_bonus=self.entropy_bonus,
            reward_baseline=self.reward_baseline,
            weights=self.weights,
            bias=self.bias,
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearSoftmaxPolicy:
        data = np.load(Path(path), allow_pickle=False)
        policy = cls(
            int(data["observation_size"]),
            int(data["action_size"]),
            learning_rate=float(data["learning_rate"]),
            temperature=float(data["temperature"]),
            entropy_bonus=float(data["entropy_bonus"]),
        )
        policy.reward_baseline = float(data["reward_baseline"])
        policy.weights = data["weights"]
        policy.bias = data["bias"]
        return policy


class NumpyMLPPolicy:
    name = "numpy_mlp"

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(
            0.0,
            np.sqrt(2.0 / max(observation_size, 1)),
            size=(observation_size, hidden_size),
        )
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.w2 = rng.normal(
            0.0,
            np.sqrt(2.0 / max(hidden_size, 1)),
            size=(hidden_size, action_size),
        )
        self.b2 = np.zeros(action_size, dtype=np.float64)
        self.reward_baseline = 0.0
        self.baseline_beta = 0.95

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        valid = tuple(info["valid_actions"])
        probs = self.action_probs(observation, valid)
        if training:
            return int(rng.choice(np.asarray(valid), p=probs))
        return int(valid[int(np.argmax(probs))])

    def action_probs(
        self,
        observation: np.ndarray,
        valid_actions: tuple[int, ...],
    ) -> np.ndarray:
        _, logits = self._forward(observation, valid_actions)
        logits = logits / max(self.temperature, 1e-6)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def update_episode(
        self,
        episode: TrainingEpisode,
        *,
        l2: float = 1e-6,
    ) -> None:
        average_reward = float(np.mean(tuple(episode.result.rewards.values())))
        self.reward_baseline = (
            self.baseline_beta * self.reward_baseline
            + (1.0 - self.baseline_beta) * average_reward
        )
        for player, samples in episode.samples_by_player.items():
            advantage = episode.result.rewards[player] - self.reward_baseline
            for sample in samples:
                self._update_sample(sample, advantage, l2=l2)

    def update_imitation(
        self,
        episode: TrainingEpisode,
        *,
        strength: float = 1.0,
        l2: float = 1e-6,
    ) -> None:
        for samples in episode.samples_by_player.values():
            for sample in samples:
                self._update_sample(sample, strength, l2=l2)

    def _forward(
        self,
        observation: np.ndarray,
        valid_actions: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        x = _normalize_observation(observation)
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2[:, valid_actions] + self.b2[list(valid_actions)]
        return hidden, logits

    def _update_sample(self, sample: StepSample, advantage: float, *, l2: float) -> None:
        if not sample.valid_actions:
            return
        x = _normalize_observation(sample.observation)
        valid = sample.valid_actions
        hidden, _ = self._forward(sample.observation, valid)
        probs = self.action_probs(sample.observation, valid)
        grad_logits = -probs
        chosen_index = valid.index(sample.action)
        grad_logits[chosen_index] += 1.0

        w2_valid_before = self.w2[:, valid].copy()
        scale = self.learning_rate * float(advantage)
        for local_idx, action in enumerate(valid):
            self.w2[:, action] += scale * grad_logits[local_idx] * hidden
            self.b2[action] += scale * grad_logits[local_idx]

        grad_hidden = w2_valid_before @ grad_logits
        grad_z1 = grad_hidden * (1.0 - hidden * hidden)
        self.w1 += scale * np.outer(x, grad_z1)
        self.b1 += scale * grad_z1
        self.w1 *= 1.0 - self.learning_rate * l2
        self.w2 *= 1.0 - self.learning_rate * l2

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            observation_size=self.observation_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            reward_baseline=self.reward_baseline,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
        )

    @classmethod
    def load(cls, path: str | Path) -> NumpyMLPPolicy:
        data = np.load(Path(path), allow_pickle=False)
        policy = cls(
            int(data["observation_size"]),
            int(data["action_size"]),
            hidden_size=int(data["hidden_size"]),
            learning_rate=float(data["learning_rate"]),
            temperature=float(data["temperature"]),
        )
        policy.reward_baseline = float(data["reward_baseline"])
        policy.w1 = data["w1"]
        policy.b1 = data["b1"]
        policy.w2 = data["w2"]
        policy.b2 = data["b2"]
        return policy


def play_game(
    policies: dict[str, Policy],
    *,
    seed: int,
    config: ColonistMultiAgentConfig | None = None,
    max_decisions: int = 5000,
    rng: np.random.Generator | None = None,
    training_policy: LinearSoftmaxPolicy | NumpyMLPPolicy | None = None,
) -> TrainingEpisode:
    env = ColonistMultiAgentEnv(config or ColonistMultiAgentConfig())
    rng = rng or np.random.default_rng(seed)
    samples_by_player: dict[str, list[StepSample]] = {}
    try:
        observations, info = env.reset(seed=seed)
        rewards = {name: 0.0 for name in env.player_names}
        terminated = False
        truncated = False
        decisions = 0
        while not (terminated or truncated) and decisions < max_decisions:
            player = info["current_player"]
            policy = policies[player]
            observation = np.asarray(observations[player], dtype=np.float64)
            valid_actions = tuple(int(action) for action in info["valid_actions"])
            action_context_features = build_action_context_feature_table(env, info)
            action = policy.select_action(
                env,
                observation,
                info,
                rng,
                training=policy is training_policy,
            )
            if policy is training_policy:
                samples_by_player.setdefault(player, []).append(
                    StepSample(
                        observation=observation.copy(),
                        valid_actions=valid_actions,
                        action=action,
                        player=player,
                        action_context_features=action_context_features,
                    )
                )
            observations, rewards, terminated, truncated, info = env.step(action)
            decisions += 1
        if not terminated and decisions >= max_decisions:
            truncated = True
            rewards = _scoreboard_rewards(env)
        result = GameResult(
            seed=seed,
            winner=_winner_from_rewards(rewards),
            rewards=dict(rewards),
            decisions=decisions,
            terminated=terminated,
            truncated=truncated,
            invalid_actions=int(info["invalid_actions_count"]),
            final_public_vps=_public_vps(env),
            final_actual_vps=_actual_vps(env),
        )
        return TrainingEpisode(result=result, samples_by_player=samples_by_player)
    finally:
        env.close()


def collect_imitation_game(
    teacher: Policy | dict[str, Policy],
    *,
    seed: int,
    config: ColonistMultiAgentConfig | None = None,
    max_decisions: int = 5000,
    record_after_decisions: int = 0,
    record_until_decision: int | None = None,
    rng: np.random.Generator | None = None,
) -> TrainingEpisode:
    config = config or ColonistMultiAgentConfig()
    env = ColonistMultiAgentEnv(config)
    rng = rng or np.random.default_rng(seed)
    samples_by_player: dict[str, list[StepSample]] = {}
    try:
        observations, info = env.reset(seed=seed)
        rewards = {name: 0.0 for name in env.player_names}
        terminated = False
        truncated = False
        decisions = 0
        while not (terminated or truncated) and decisions < max_decisions:
            player = info["current_player"]
            observation = np.asarray(observations[player], dtype=np.float64)
            valid_actions = tuple(int(action) for action in info["valid_actions"])
            action_context_features = build_action_context_feature_table(env, info)
            acting_teacher = teacher[player] if isinstance(teacher, dict) else teacher
            action = acting_teacher.select_action(
                env,
                observation,
                info,
                rng,
                training=False,
            )
            target_policy_fn = getattr(acting_teacher, "target_policy", None)
            target_policy = (
                target_policy_fn(env, info, rng)
                if callable(target_policy_fn)
                else None
            )
            target_scores_fn = getattr(acting_teacher, "target_scores", None)
            target_scores = (
                target_scores_fn(env, info, rng)
                if callable(target_scores_fn)
                else None
            )
            target_score_source = _target_score_source(acting_teacher, target_scores)
            if should_record_imitation_sample(
                decisions,
                record_after_decisions=record_after_decisions,
                record_until_decision=record_until_decision,
            ):
                samples_by_player.setdefault(player, []).append(
                    StepSample(
                        observation=observation.copy(),
                        valid_actions=valid_actions,
                        action=action,
                        player=player,
                        action_context_features=action_context_features,
                        phase=_phase_from_info(info),
                        target_policy=target_policy,
                        target_scores=target_scores,
                        target_score_source=target_score_source,
                        decision_index=decisions,
                        teacher_name=getattr(acting_teacher, "name", type(acting_teacher).__name__),
                        action_mask_version=str(info.get("action_mask_version", "")),
                    )
                )
            observations, rewards, terminated, truncated, info = env.step(action)
            decisions += 1
        if not terminated and decisions >= max_decisions:
            truncated = True
            rewards = _scoreboard_rewards(env)
        result = GameResult(
            seed=seed,
            winner=_winner_from_rewards(rewards),
            rewards=dict(rewards),
            decisions=decisions,
            terminated=terminated,
            truncated=truncated,
            invalid_actions=int(info["invalid_actions_count"]),
            final_public_vps=_public_vps(env),
            final_actual_vps=_actual_vps(env),
        )
        return TrainingEpisode(result=result, samples_by_player=samples_by_player)
    finally:
        env.close()


def should_record_imitation_sample(
    decision_index: int,
    *,
    record_after_decisions: int = 0,
    record_until_decision: int | None = None,
) -> bool:
    if decision_index < max(0, int(record_after_decisions)):
        return False
    if record_until_decision is None:
        return True
    return decision_index < int(record_until_decision)


def _target_score_source(teacher: Policy, target_scores: dict[int, float] | None) -> str:
    if not target_scores:
        return ""
    source_fn = getattr(teacher, "target_score_source", None)
    if callable(source_fn):
        return str(source_fn())
    return type(teacher).__name__


def _phase_from_info(info: dict[str, Any]) -> str:
    timer = info.get("timer")
    if isinstance(timer, dict) and timer.get("phase"):
        return str(timer["phase"])
    return str(info.get("phase", ""))


def make_env_config(
    *,
    players: int = 4,
    vps_to_win: int = 10,
    use_graph_history_features: bool = False,
) -> ColonistMultiAgentConfig:
    return ColonistMultiAgentConfig(
        players=players,
        vps_to_win=vps_to_win,
        use_graph_history_features=use_graph_history_features,
    )


def create_linear_policy(
    *,
    config: ColonistMultiAgentConfig | None = None,
    seed: int = 0,
    learning_rate: float = 0.01,
    temperature: float = 1.0,
) -> LinearSoftmaxPolicy:
    env = ColonistMultiAgentEnv(config or ColonistMultiAgentConfig())
    try:
        observations, _ = env.reset(seed=seed)
        observation_size = len(next(iter(observations.values())))
        return LinearSoftmaxPolicy(
            observation_size,
            env.action_space.n,
            learning_rate=learning_rate,
            temperature=temperature,
            seed=seed,
        )
    finally:
        env.close()


def create_mlp_policy(
    *,
    config: ColonistMultiAgentConfig | None = None,
    seed: int = 0,
    learning_rate: float = 0.001,
    temperature: float = 1.0,
    hidden_size: int = 128,
) -> NumpyMLPPolicy:
    env = ColonistMultiAgentEnv(config or ColonistMultiAgentConfig())
    try:
        observations, _ = env.reset(seed=seed)
        observation_size = len(next(iter(observations.values())))
        return NumpyMLPPolicy(
            observation_size,
            env.action_space.n,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            temperature=temperature,
            seed=seed,
        )
    finally:
        env.close()


def evaluate_policy(
    candidate: Policy,
    opponent: Policy,
    *,
    games: int,
    seed: int,
    config: ColonistMultiAgentConfig | None = None,
    max_decisions: int = 5000,
    start_game_index: int = 0,
    progress_callback=None,
) -> dict[str, Any]:
    config = config or ColonistMultiAgentConfig()
    rng = np.random.default_rng(seed)
    wins = 0
    total_decisions = 0
    total_candidate_vp = 0.0
    total_best_opponent_vp = 0.0
    total_vp_margin = 0.0
    total_candidate_win_decisions = 0
    total_invalid_actions = 0
    stuck_games = 0
    seat_wins = {name: 0 for name in ("BLUE", "RED", "ORANGE", "WHITE")[: config.players]}
    for game_idx in range(games):
        env = ColonistMultiAgentEnv(config)
        try:
            env.reset(seed=seed + game_idx)
            candidate_seat = env.player_names[(start_game_index + game_idx) % config.players]
        finally:
            env.close()
        policies = {
            name: candidate if name == candidate_seat else opponent
            for name in seat_wins
        }
        episode = play_game(
            policies,
            seed=int(rng.integers(2**31)),
            config=config,
            max_decisions=max_decisions,
        )
        total_decisions += episode.result.decisions
        candidate_vp = int(episode.result.final_actual_vps.get(candidate_seat, 0))
        opponent_vps = [
            int(value)
            for seat, value in episode.result.final_actual_vps.items()
            if seat != candidate_seat
        ]
        best_opponent_vp = max(opponent_vps) if opponent_vps else 0
        total_candidate_vp += float(candidate_vp)
        total_best_opponent_vp += float(best_opponent_vp)
        total_vp_margin += float(candidate_vp - best_opponent_vp)
        total_invalid_actions += int(episode.result.invalid_actions)
        # FIX (adversarial review, truncation-as-loss bias): a truncated game
        # (hit max_decisions with no winner) is a missing data point, not a
        # loss -- callers pairing per-game outcomes (sprt_gate.py,
        # compare_scoreboards.py) need this flag to exclude it rather than
        # let `winner == candidate_seat` silently evaluate False for it.
        game_truncated = bool(episode.result.truncated or not episode.result.terminated)
        stuck_games += int(game_truncated)
        if episode.result.winner == candidate_seat:
            wins += 1
            total_candidate_win_decisions += int(episode.result.decisions)
            seat_wins[candidate_seat] += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "game": game_idx + 1,
                    "games": games,
                    "wins": wins,
                    "win_rate": wins / float(game_idx + 1),
                    "candidate_seat": candidate_seat,
                    "winner": episode.result.winner,
                    "truncated": game_truncated,
                    "decisions": episode.result.decisions,
                    "candidate_vp": candidate_vp,
                    "best_opponent_vp": best_opponent_vp,
                    "vp_margin": candidate_vp - best_opponent_vp,
                    "seat_wins": dict(seat_wins),
                }
            )
    win_rate = wins / games if games else 0.0
    return {
        "games": games,
        "candidate": candidate.name,
        "opponent": opponent.name,
        "wins": wins,
        "win_rate": win_rate,
        "elo_vs_opponent": elo_difference(win_rate),
        "seat_wins": seat_wins,
        "avg_decisions": total_decisions / games if games else 0.0,
        "moves_to_win": total_candidate_win_decisions / wins if wins else None,
        "avg_candidate_win_decisions": total_candidate_win_decisions / wins if wins else None,
        "avg_candidate_vp": total_candidate_vp / games if games else 0.0,
        "avg_best_opponent_vp": total_best_opponent_vp / games if games else 0.0,
        "avg_vp_margin": total_vp_margin / games if games else 0.0,
        "illegal_action_count": total_invalid_actions,
        "timeouts_or_stuck_games": stuck_games,
    }


def elo_difference(score: float) -> float:
    clipped = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * float(np.log10((1.0 / clipped) - 1.0))


def write_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_observation(observation: np.ndarray) -> np.ndarray:
    x = np.asarray(observation, dtype=np.float64)
    return np.nan_to_num(x / 20.0, nan=0.0, posinf=1.0, neginf=-1.0)


def _score_dict_to_policy(
    scores: dict[int, float],
    *,
    temperature: float = 0.7,
) -> dict[int, float]:
    finite = [
        (int(action), float(score))
        for action, score in scores.items()
        if np.isfinite(float(score))
    ]
    if not finite:
        return {}
    values = np.asarray([score for _, score in finite], dtype=np.float64)
    std = float(values.std())
    logits = np.zeros_like(values) if std <= 1e-9 else (values - float(values.mean())) / std
    logits = np.clip(logits / max(float(temperature), 1e-6), -30.0, 30.0)
    logits -= float(logits.max())
    weights = np.exp(logits)
    total = float(weights.sum())
    if total <= 0.0:
        return {}
    weights /= total
    return {
        int(action): float(weight)
        for weight, (action, _) in zip(weights, finite)
        if weight > 0.0
    }


def _winner_from_rewards(rewards: dict[str, float]) -> str | None:
    if not rewards:
        return None
    winner, reward = max(rewards.items(), key=lambda item: item[1])
    return winner if reward > 0 else None


def _public_vps(env: ColonistMultiAgentEnv) -> dict[str, int]:
    return {
        player: int(env.observation_payload(player)["players"][player]["public_victory_points"])
        for player in env.player_names
    }


def _actual_vps(env: ColonistMultiAgentEnv) -> dict[str, int]:
    state = env.game.state
    return {
        color.name: int(state.player_state[f"{env.player_key(state, color)}_ACTUAL_VICTORY_POINTS"])
        for color in env.player_colors
    }


def _scoreboard_rewards(env: ColonistMultiAgentEnv) -> dict[str, float]:
    values = _scoreboard_values(env)
    best = max(values.values())
    winners = [player for player, score in values.items() if score == best]
    if len(winners) == 1:
        return {
            player: 1.0 if player == winners[0] else -1.0 / (len(values) - 1)
            for player in values
        }
    return {player: 0.0 for player in values}


def _scoreboard_values(env: ColonistMultiAgentEnv) -> dict[str, float]:
    state = env.game.state
    values: dict[str, float] = {}
    for color in env.player_colors:
        name = color.name
        key = env.player_key(state, color)
        resources = env.player_num_resource_cards(state, color)
        devs = env.player_num_dev_cards(state, color)
        roads_used = 15 - state.player_state[f"{key}_ROADS_AVAILABLE"]
        settlements_used = 5 - state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"]
        cities_used = 4 - state.player_state[f"{key}_CITIES_AVAILABLE"]
        production = sum(
            _node_production(env, node)
            for node in state.buildings_by_color[color]["SETTLEMENT"]
        )
        production += 2 * sum(
            _node_production(env, node)
            for node in state.buildings_by_color[color]["CITY"]
        )
        values[name] = (
            100.0 * state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
            + 3.0 * production
            + 1.0 * resources
            + 2.0 * devs
            + 0.2 * roads_used
            + 0.5 * settlements_used
            + 0.7 * cities_used
        )
    return values


def _normal_resource_bundle(value: dict[str, Any]) -> dict[str, int]:
    if "resources" in value and isinstance(value["resources"], dict):
        value = value["resources"]
    return {
        resource: int(value.get(resource, value.get(resource.upper(), 0)) or 0)
        for resource in JSettlersLitePolicy._resources
    }


def _resource_tuple_bundle(values: tuple[Any, ...]) -> dict[str, int]:
    bundle = {resource: 0 for resource in JSettlersLitePolicy._resources}
    for value in values:
        resource = str(value).lower()
        if resource in bundle:
            bundle[resource] += 1
    return bundle


def _resource_deficits(
    resources: dict[str, int],
    cost: dict[str, int],
) -> dict[str, int]:
    return {
        resource: max(0, int(cost.get(resource, 0)) - int(resources.get(resource, 0)))
        for resource in JSettlersLitePolicy._resources
    }


def _missing_resource_count(resources: dict[str, int], cost: dict[str, int]) -> int:
    return sum(_resource_deficits(resources, cost).values())


def _score_game_for_color(game: Any, color: Any) -> float:
    winner = game.winning_color()
    if winner == color:
        return 10_000.0 + _raw_color_score(game, color)
    if winner is not None:
        return -10_000.0 + _raw_color_score(game, color)

    own = _raw_color_score(game, color)
    opponent_best = max(
        (
            _raw_color_score(game, opponent)
            for opponent in game.state.colors
            if opponent != color
        ),
        default=0.0,
    )
    return own - 0.15 * opponent_best


def _catanatron_value_score(
    game: Any,
    color: Any,
    *,
    opponent_penalty: float,
    value_fn: Any,
) -> float:
    winner = game.winning_color()
    if winner == color:
        return 1e18
    if winner is not None:
        return -1e18

    try:
        own = float(value_fn(game, color))
        opponent_best = max(
            (
                float(value_fn(game, opponent))
                for opponent in game.state.colors
                if opponent != color
            ),
            default=0.0,
        )
        return own - opponent_penalty * opponent_best
    except Exception:
        return _score_game_for_color(game, color)


def _rollout_game(game: Any, *, max_decisions: int) -> None:
    for _ in range(max_decisions):
        if game.winning_color() is not None:
            return
        playable = tuple(game.playable_actions)
        if not playable:
            return
        action = max(
            playable,
            key=lambda candidate: _score_catanatron_action(game, candidate),
        )
        try:
            game.execute(action)
        except Exception:
            return


def _rollout_game_with_value(
    game: Any,
    *,
    max_decisions: int,
    value_fn: Any,
    opponent_penalty: float,
) -> None:
    for _ in range(max_decisions):
        if game.winning_color() is not None:
            return
        playable = tuple(game.playable_actions)
        if not playable:
            return
        color = game.state.current_color()
        action = max(
            playable,
            key=lambda candidate: _score_catanatron_value_action(
                game,
                candidate,
                color,
                value_fn=value_fn,
                opponent_penalty=opponent_penalty,
            ),
        )
        try:
            game.execute(action)
        except Exception:
            return


def _score_catanatron_value_action(
    game: Any,
    action: Any,
    color: Any,
    *,
    value_fn: Any,
    opponent_penalty: float,
) -> float:
    game_copy = game.copy()
    try:
        game_copy.execute(action)
    except Exception:
        return float("-inf")
    return _catanatron_value_score(
        game_copy,
        color,
        opponent_penalty=opponent_penalty,
        value_fn=value_fn,
    )


def _score_catanatron_action(game: Any, action: Any) -> float:
    action_type = action.action_type.name
    value = action.value
    score = HeuristicPolicy._priority.get(action_type, 0)
    if action_type == "BUILD_SETTLEMENT" and isinstance(value, int):
        score += 10.0 * _node_production_from_game(game, value)
    elif action_type == "BUILD_CITY" and isinstance(value, int):
        score += 10.0 * _node_production_from_game(game, value)
    elif action_type == "BUILD_ROAD" and isinstance(value, (tuple, list)):
        score += 2.0 * max(_node_production_from_game(game, node) for node in value)
    elif action_type == "END_TURN":
        score -= 2.0
    return score + random.random() * 1e-3


def _raw_color_score(game: Any, color: Any) -> float:
    state = game.state
    key = f"P{state.color_to_index[color]}"
    resources = (
        state.player_state[f"{key}_WOOD_IN_HAND"]
        + state.player_state[f"{key}_BRICK_IN_HAND"]
        + state.player_state[f"{key}_SHEEP_IN_HAND"]
        + state.player_state[f"{key}_WHEAT_IN_HAND"]
        + state.player_state[f"{key}_ORE_IN_HAND"]
    )
    devs = (
        state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"]
        + state.player_state[f"{key}_MONOPOLY_IN_HAND"]
        + state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
        + state.player_state[f"{key}_KNIGHT_IN_HAND"]
        + state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"]
    )
    roads_used = 15 - state.player_state[f"{key}_ROADS_AVAILABLE"]
    settlements_used = 5 - state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"]
    cities_used = 4 - state.player_state[f"{key}_CITIES_AVAILABLE"]
    production = sum(
        _node_production_from_game(game, node)
        for node in state.buildings_by_color[color]["SETTLEMENT"]
    )
    production += 2 * sum(
        _node_production_from_game(game, node)
        for node in state.buildings_by_color[color]["CITY"]
    )
    return (
        100.0 * state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        + 3.0 * production
        + 1.0 * resources
        + 2.0 * devs
        + 0.2 * roads_used
        + 0.5 * settlements_used
        + 0.7 * cities_used
    )


def _node_production(env: ColonistMultiAgentEnv, node_id: int) -> float:
    board = env.game.state.board
    production = board.map.node_production.get(node_id)
    if production is None:
        return 0.0
    return float(sum(production.values()))


def _edge_frontier_production(env: ColonistMultiAgentEnv, edge: tuple[int, int]) -> float:
    return max((_node_production(env, int(node)) for node in edge), default=0.0)


def _node_production_from_game(game: Any, node_id: int) -> float:
    production = game.state.board.map.node_production.get(node_id)
    if production is None:
        return 0.0
    return float(sum(production.values()))


def _victim_public_vp(env: ColonistMultiAgentEnv, victim: str | None) -> float:
    if victim is None:
        return 0.0
    payload = env.observation_payload(env.current_player_name(), include_event_log=False)
    return float(payload["players"].get(victim, {}).get("public_victory_points", 0))
