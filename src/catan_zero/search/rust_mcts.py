from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Protocol


def _require_rust_module():
    try:
        import catanatron_rs  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised only without local binding.
        raise RuntimeError(
            "catanatron_rs is required for RustMCTS. Install the optimized Rust "
            "binding via tools/install_v1_freeze.sh or pip install the wheel "
            "from the release assets."
        ) from error

    required = (
        "copy",
        "playable_action_indices",
        "playable_actions_json",
        "execute_action_index",
        "spectrum_json",
        "apply_chance_outcome",
        "winning_color",
        "json_snapshot",
    )
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=0)
    missing = [name for name in required if not hasattr(game, name)]
    if missing:
        raise RuntimeError(
            "installed catanatron_rs is missing MCTS bindings: "
            + ", ".join(missing)
        )
    return catanatron_rs


@dataclass(frozen=True, slots=True)
class RustMCTSConfig:
    colors: tuple[str, ...] = ("RED", "BLUE")
    map_kind: str | None = None
    simulations: int = 64
    c_puct: float = 1.5
    max_depth: int = 80
    seed: int = 0
    temperature: float = 1.0
    # Opt-in root-consistent public-belief search. Simulations are a TOTAL
    # nominal budget divided across determinizations.
    information_set_search: bool = False
    determinization_particles: int = 1
    determinization_min_simulations: int = 32


@dataclass(frozen=True, slots=True)
class RustMCTSResult:
    action: int
    policy: dict[int, float]
    visits: dict[int, int]
    q_values: dict[int, float]
    priors: dict[int, float]
    root_value: float


class RustEvaluator(Protocol):
    def evaluate(
        self,
        game: Any,
        legal_actions: tuple[int, ...],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> tuple[dict[int, float], float]:
        """Return legal-action priors and value from root_color perspective."""


class HeuristicRustEvaluator:
    """Small deterministic evaluator for MCTS plumbing and smoke teachers.

    This is not intended to be the final evaluator. It exists so the search tree,
    chance handling, visit targets, and legality checks can be tested before the
    35M checkpoint is plugged in as a batched policy/value service.
    """

    def __init__(self, *, score_actions: bool = True) -> None:
        self.score_actions = bool(score_actions)

    def evaluate(
        self,
        game: Any,
        legal_actions: tuple[int, ...],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> tuple[dict[int, float], float]:
        acting_color = str(game.current_color())
        root_to_act = acting_color == str(root_color)
        value = _heuristic_value(game, root_color=root_color, colors=colors)
        if not legal_actions:
            return {}, value
        if not self.score_actions or len(legal_actions) == 1:
            prior = 1.0 / float(len(legal_actions))
            return {int(action): prior for action in legal_actions}, value

        action_json = _playable_action_json_by_index(game, legal_actions, colors, None)
        scores: dict[int, float] = {}
        for action_id in legal_actions:
            outcomes = _spectrum(game, action_json[action_id])
            expected = 0.0
            for outcome_index, probability in outcomes:
                outcome = game.apply_chance_outcome(
                    json.dumps(action_json[action_id]),
                    outcome_index,
                )
                expected += probability * _heuristic_value(
                    outcome,
                    root_color=root_color,
                    colors=colors,
                )
            # Priors should model the side to act.  The returned value remains
            # from root_color's perspective, but opponent actions should be
            # biased toward reducing that value rather than helping the root.
            scores[int(action_id)] = expected if root_to_act else -expected
        return _softmax_scores(scores, temperature=1.25), value

    def evaluate_many(
        self,
        requests: list[tuple[Any, tuple[int, ...]]],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> list[tuple[dict[int, float], float]]:
        """Evaluate several (game, legal_actions) pairs.

        Plain per-request loop -- this evaluator has no GPU batching to gain
        from a real batched call, but exposing the same `evaluate_many`
        interface as `EntityGraphRustEvaluator` lets callers (e.g.
        `gumbel_chance_mcts.py`'s ROLL-child expansion) treat both uniformly.
        """
        return [
            self.evaluate(game, legal_actions, root_color=root_color, colors=colors)
            for game, legal_actions in requests
        ]


@dataclass(slots=True)
class _ActionStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "_Node"] = field(default_factory=dict)

    @property
    def q(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / float(self.visits)


@dataclass(slots=True)
class _Node:
    game: Any
    root_color: str
    prior_value: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    actions: dict[int, _ActionStats] = field(default_factory=dict)
    action_json: dict[int, Any] = field(default_factory=dict)
    expanded: bool = False

    @property
    def value(self) -> float:
        if self.visits <= 0:
            return self.prior_value
        return self.value_sum / float(self.visits)


class RustMCTS:
    def __init__(
        self,
        config: RustMCTSConfig | None = None,
        evaluator: RustEvaluator | None = None,
    ) -> None:
        self.config = config or RustMCTSConfig()
        self.evaluator = evaluator or HeuristicRustEvaluator()
        self.rng = random.Random(self.config.seed)
        _require_rust_module()

    def search(self, game: Any) -> RustMCTSResult:
        if bool(self.config.information_set_search):
            return self._search_information_set(game)
        return self._search_single_world(game)

    def _search_single_world(self, game: Any) -> RustMCTSResult:
        root_color = str(game.current_color())
        legal_actions = _legal_action_indices(
            game,
            colors=self.config.colors,
            map_kind=self.config.map_kind,
        )
        if len(legal_actions) == 1:
            action = int(legal_actions[0])
            return RustMCTSResult(
                action=action,
                policy={action: 1.0},
                visits={action: 1},
                q_values={action: 0.0},
                priors={action: 1.0},
                root_value=_terminal_or_zero(game, root_color),
            )
        root = _Node(game=game.copy(), root_color=root_color)
        self._expand(root)
        for _ in range(max(int(self.config.simulations), 1)):
            self._simulate(root, depth=0)
        return self._result(root)

    def _spawn_information_set_particle(
        self, *, simulations: int, seed: int
    ) -> "RustMCTS":
        particle = RustMCTS(
            replace(
                self.config,
                simulations=int(simulations),
                seed=int(seed),
                information_set_search=False,
            ),
            self.evaluator,
        )
        particle._root_actor_turn_only = True
        return particle

    def _search_information_set(self, game: Any) -> RustMCTSResult:
        determinize = getattr(game, "determinize_for_player", None)
        if not callable(determinize):
            raise RuntimeError(
                "information_set_search requires Game.determinize_for_player"
            )
        evaluator_config = getattr(self.evaluator, "config", None)
        if evaluator_config is None or not bool(
            getattr(evaluator_config, "public_observation", False)
        ):
            raise RuntimeError(
                "information_set_search requires evaluator public_observation=True"
            )
        particles = int(self.config.determinization_particles)
        if particles < 1:
            raise ValueError("determinization_particles must be >= 1")
        min_per_particle = int(self.config.determinization_min_simulations)
        if min_per_particle < 1:
            raise ValueError("determinization_min_simulations must be >= 1")
        total_budget = max(int(self.config.simulations), 1)
        particles = min(particles, max(1, total_budget // min_per_particle))
        base, remainder = divmod(total_budget, particles)
        budgets = [base + int(index < remainder) for index in range(particles)]
        root_color = str(game.current_color())
        authoritative_legal = _legal_action_indices(
            game, colors=self.config.colors, map_kind=self.config.map_kind
        )
        if not authoritative_legal:
            raise RuntimeError("information-set root has no legal actions")
        seeds = [self.rng.getrandbits(64) for _ in range(particles)]
        results: list[RustMCTSResult] = []
        for budget, particle_seed in zip(budgets, seeds):
            sampled = determinize(root_color, int(particle_seed))
            sampled_legal = _legal_action_indices(
                sampled, colors=self.config.colors, map_kind=self.config.map_kind
            )
            if tuple(sampled_legal) != tuple(authoritative_legal):
                raise RuntimeError(
                    "public-belief determinization changed root legal actions"
                )
            particle = self._spawn_information_set_particle(
                simulations=budget, seed=particle_seed
            )
            particle._information_set_root_turn = int(sampled.num_turns())
            results.append(particle._search_single_world(sampled))
        return self._aggregate_information_set_results(
            results, legal_actions=authoritative_legal
        )

    def _aggregate_information_set_results(
        self,
        results: list[RustMCTSResult],
        *,
        legal_actions: tuple[int, ...],
    ) -> RustMCTSResult:
        if not results:
            raise RuntimeError("information-set search produced no particles")
        count = float(len(results))
        priors = _normalize_policy(
            {
                action: sum(result.priors.get(action, 0.0) for result in results)
                / count
                for action in legal_actions
            }
        )
        policy = _normalize_policy(
            {
                action: sum(result.policy.get(action, 0.0) for result in results)
                / count
                for action in legal_actions
            }
        )
        visits = {
            action: sum(result.visits.get(action, 0) for result in results)
            for action in legal_actions
        }
        q_values: dict[int, float] = {}
        for action in legal_actions:
            weighted = [
                (result.q_values[action], result.visits.get(action, 0))
                for result in results
                if action in result.q_values and result.visits.get(action, 0) > 0
            ]
            denominator = sum(weight for _value, weight in weighted)
            if denominator:
                q_values[action] = sum(
                    value * weight for value, weight in weighted
                ) / float(denominator)
        action = max(
            legal_actions,
            key=lambda candidate: (
                policy.get(candidate, 0.0),
                visits.get(candidate, 0),
                priors.get(candidate, 0.0),
                -int(candidate),
            ),
        )
        return RustMCTSResult(
            action=int(action),
            policy=policy,
            visits=visits,
            q_values=q_values,
            priors=priors,
            root_value=sum(result.root_value for result in results) / count,
        )

    def _simulate(self, node: _Node, *, depth: int) -> float:
        winner = node.game.winning_color()
        if winner is not None:
            return 1.0 if str(winner) == node.root_color else -1.0
        if depth >= int(self.config.max_depth):
            return _heuristic_value(
                node.game,
                root_color=node.root_color,
                colors=self.config.colors,
            )
        if self._is_information_set_turn_boundary(node, depth=depth):
            if not node.expanded:
                value = self._expand(node)
                node.visits += 1
                node.value_sum += value
                return value
            return node.value
        if not node.expanded:
            if self._expand_forced(node):
                return self._simulate(node, depth=depth)
            value = self._expand(node)
            node.visits += 1
            node.value_sum += value
            return value
        if not node.actions:
            value = node.prior_value
            node.visits += 1
            node.value_sum += value
            return value

        action_id, stats = self._select_action(node)
        outcomes = _spectrum(node.game, node.action_json[action_id])
        outcome_index = self._sample_outcome(outcomes)
        child = stats.children.get(outcome_index)
        if child is None:
            child_game = node.game.apply_chance_outcome(
                json.dumps(node.action_json[action_id]),
                outcome_index,
            )
            child = _Node(game=child_game, root_color=node.root_color)
            stats.children[outcome_index] = child

        value = self._simulate(child, depth=depth + 1)
        stats.visits += 1
        stats.value_sum += value
        node.visits += 1
        node.value_sum += value
        return value

    def _is_information_set_turn_boundary(self, node: _Node, *, depth: int) -> bool:
        if not bool(getattr(self, "_root_actor_turn_only", False)) or depth <= 0:
            return False
        if str(node.game.current_color()) != str(node.root_color):
            return True
        root_turn = getattr(self, "_information_set_root_turn", None)
        return root_turn is not None and int(node.game.num_turns()) != int(root_turn)

    def _expand_forced(self, node: _Node) -> bool:
        legal_actions = _legal_action_indices(
            node.game,
            colors=self.config.colors,
            map_kind=self.config.map_kind,
        )
        if len(legal_actions) != 1:
            return False
        action_id = int(legal_actions[0])
        node.action_json = _playable_action_json_by_index(
            node.game,
            legal_actions,
            self.config.colors,
            self.config.map_kind,
        )
        node.actions = {action_id: _ActionStats(prior=1.0)}
        node.prior_value = _terminal_or_zero(node.game, node.root_color)
        node.expanded = True
        return True

    def _expand(self, node: _Node) -> float:
        legal_actions = _legal_action_indices(
            node.game,
            colors=self.config.colors,
            map_kind=self.config.map_kind,
        )
        priors, value = self.evaluator.evaluate(
            node.game,
            legal_actions,
            root_color=node.root_color,
            colors=self.config.colors,
        )
        if legal_actions:
            missing = [action for action in legal_actions if action not in priors]
            if missing:
                floor = min((p for p in priors.values() if p > 0.0), default=1.0)
                for action in missing:
                    priors[int(action)] = floor * 0.01
            priors = _normalize_policy(
                {int(action): float(priors.get(int(action), 0.0)) for action in legal_actions}
            )
            node.action_json = _playable_action_json_by_index(
                node.game,
                legal_actions,
                self.config.colors,
                self.config.map_kind,
            )
            node.actions = {
                int(action): _ActionStats(prior=float(priors[int(action)]))
                for action in legal_actions
            }
        node.prior_value = float(max(min(value, 1.0), -1.0))
        node.expanded = True
        return node.prior_value

    def _select_action(self, node: _Node) -> tuple[int, _ActionStats]:
        total_visits = max(node.visits, 1)
        sqrt_total = math.sqrt(float(total_visits))
        best_score = float("-inf")
        best: tuple[int, _ActionStats] | None = None
        root_to_act = str(node.game.current_color()) == str(node.root_color)
        for action_id, stats in node.actions.items():
            exploration = (
                float(self.config.c_puct)
                * stats.prior
                * sqrt_total
                / float(1 + stats.visits)
            )
            # Zero-sum/paranoid backup for 2p and early multiplayer work:
            # the tree value is always root_color's value, so non-root turns
            # select actions that minimize that value.
            exploitation = stats.q if root_to_act else -stats.q
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best = (action_id, stats)
        if best is None:  # pragma: no cover - guarded by caller.
            raise RuntimeError("cannot select action from empty MCTS node")
        return best

    def _sample_outcome(self, outcomes: tuple[tuple[int, float], ...]) -> int:
        if len(outcomes) == 1:
            return outcomes[0][0]
        draw = self.rng.random()
        cumulative = 0.0
        for outcome_index, probability in outcomes:
            cumulative += probability
            if draw <= cumulative:
                return outcome_index
        return outcomes[-1][0]

    def _result(self, root: _Node) -> RustMCTSResult:
        visits = {action: stats.visits for action, stats in root.actions.items()}
        if not visits:
            raise RuntimeError("MCTS root has no legal actions")
        policy = _visit_policy(visits, temperature=float(self.config.temperature))
        action = max(policy, key=lambda key: (policy[key], visits[key], root.actions[key].prior))
        return RustMCTSResult(
            action=int(action),
            policy=policy,
            visits=visits,
            q_values={action: stats.q for action, stats in root.actions.items()},
            priors={action: stats.prior for action, stats in root.actions.items()},
            root_value=root.value,
        )


def _terminal_or_zero(game: Any, root_color: str) -> float:
    winner = game.winning_color()
    if winner is None:
        return 0.0
    return 1.0 if str(winner) == str(root_color) else -1.0


def _legal_action_indices(
    game: Any,
    *,
    colors: tuple[str, ...],
    map_kind: str | None,
) -> tuple[int, ...]:
    return tuple(
        int(action)
        for action in game.playable_action_indices(list(colors), map_kind)
    )


def _playable_action_json_by_index(
    game: Any,
    legal_actions: tuple[int, ...],
    colors: tuple[str, ...],
    map_kind: str | None,
) -> dict[int, Any]:
    ids = [
        int(action)
        for action in game.playable_action_indices(list(colors), map_kind)
    ]
    actions = json.loads(game.playable_actions_json())
    if len(ids) != len(actions):
        raise RuntimeError(
            f"playable action id/action mismatch: ids={len(ids)} actions={len(actions)}"
        )
    by_index = {action_id: action for action_id, action in zip(ids, actions)}
    missing = [action for action in legal_actions if action not in by_index]
    if missing:
        raise RuntimeError(f"legal actions missing JSON payloads: {missing[:8]}")
    return by_index


def _spectrum(game: Any, action_json: Any) -> tuple[tuple[int, float], ...]:
    raw = json.loads(game.spectrum_json(json.dumps(action_json)))
    out = tuple(
        (index, float(entry.get("probability", 0.0)))
        for index, entry in enumerate(raw)
    )
    total = sum(probability for _, probability in out)
    if total <= 0.0:
        return ((0, 1.0),)
    return tuple((index, probability / total) for index, probability in out)


def _heuristic_value(game: Any, *, root_color: str, colors: tuple[str, ...]) -> float:
    winner = game.winning_color()
    if winner is not None:
        return 1.0 if str(winner) == root_color else -1.0

    snapshot = json.loads(game.json_snapshot())
    color_order = tuple(str(color) for color in snapshot.get("colors", colors))
    states = snapshot.get("player_state", [])
    try:
        root_index = color_order.index(root_color)
    except ValueError:
        root_index = 0
    if root_index >= len(states):
        return 0.0

    def score(index: int, state: dict[str, Any]) -> float:
        resources = state.get("resources", {})
        dev_cards = state.get("dev_cards", {})
        resource_count = float(sum(float(value) for value in resources.values()))
        dev_count = float(sum(float(value) for value in dev_cards.values()))
        return (
            float(state.get("actual_victory_points", state.get("victory_points", 0))) * 1.35
            + float(state.get("victory_points", 0)) * 0.35
            + float(state.get("longest_road_length", 0)) * 0.03
            + resource_count * 0.025
            + dev_count * 0.07
            - abs(index - root_index) * 0.001
        )

    root_score = score(root_index, states[root_index])
    opponent_score = max(
        (score(index, state) for index, state in enumerate(states) if index != root_index),
        default=0.0,
    )
    return math.tanh((root_score - opponent_score) / 3.0)


def _softmax_scores(scores: dict[int, float], *, temperature: float) -> dict[int, float]:
    if not scores:
        return {}
    temp = max(float(temperature), 1.0e-6)
    max_score = max(scores.values())
    weights = {
        action: math.exp(max(min((score - max_score) / temp, 40.0), -40.0))
        for action, score in scores.items()
    }
    return _normalize_policy(weights)


def _normalize_policy(weights: dict[int, float]) -> dict[int, float]:
    clean = {
        int(action): max(float(weight), 0.0)
        for action, weight in weights.items()
        if math.isfinite(float(weight))
    }
    total = sum(clean.values())
    if total <= 0.0 and clean:
        uniform = 1.0 / float(len(clean))
        return {action: uniform for action in clean}
    if total <= 0.0:
        return {}
    return {action: weight / total for action, weight in clean.items()}


def _visit_policy(visits: dict[int, int], *, temperature: float) -> dict[int, float]:
    if temperature <= 1.0e-6:
        best = max(visits, key=lambda action: visits[action])
        return {int(action): 1.0 if action == best else 0.0 for action in visits}
    exponent = 1.0 / max(float(temperature), 1.0e-6)
    weights = {
        int(action): float(max(count, 0)) ** exponent
        for action, count in visits.items()
    }
    return _normalize_policy(weights)
