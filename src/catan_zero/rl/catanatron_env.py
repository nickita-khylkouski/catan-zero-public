from __future__ import annotations

import random
from typing import Any, Callable, Iterable

try:  # pragma: no cover - fallback exists for dependency-light imports.
    import numpy as np
except ImportError:  # pragma: no cover - exercised in clean envs without numpy.
    np = None

from catan_zero.rl._catanatron import import_catanatron_module
from catan_zero.rl.action_mask import ActionCatalog, MapType
from catan_zero.rl.spaces import make_box, make_discrete

try:  # pragma: no cover - exercised only when gymnasium is installed.
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback is covered in this repo.
    gym = None

BaseEnv = gym.Env if gym is not None else object
RewardFunction = Callable[[Any, Any, Any], float]


def terminal_win_loss_reward(action: Any, game: Any, actor_color: Any) -> float:
    winning_color = game.winning_color()
    if winning_color is None:
        return 0.0
    return 1.0 if winning_color == actor_color else -1.0


class CatanatronRLEnv(BaseEnv):
    """Gymnasium-style CatanZero RL env backed by Catanatron core modules."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if np is None:
            raise RuntimeError("numpy is required to instantiate CatanatronRLEnv")

        self.config = config or {}
        self.dtype = np.dtype(self.config.get("dtype", np.float32))
        self.map_type: MapType = self.config.get("map_type", "BASE")
        self.vps_to_win = int(self.config.get("vps_to_win", 10))
        self.invalid_action_reward = float(
            self.config.get("invalid_action_reward", -1.0)
        )
        self.max_invalid_actions = int(self.config.get("max_invalid_actions", 10))
        self.reward_function: RewardFunction = self.config.get(
            "reward_function", terminal_win_loss_reward
        )

        self._load_catanatron_symbols()
        self.p0_color = self.Color.BLUE
        self.players = self._build_players(self.config.get("enemies"))
        self.player_colors = tuple(player.color for player in self.players)

        self.features = tuple(
            self.features_module.get_feature_ordering(
                len(self.players),
                self.map_type,
            )
        )
        self.action_catalog = ActionCatalog(self.player_colors, self.map_type)
        self.action_space = make_discrete(self.action_catalog.size)
        self.observation_space = make_box(
            low=0,
            high=float(self.config.get("observation_high", 19 * 5)),
            shape=(len(self.features),),
            dtype=self.dtype,
        )

        self.game: Any | None = None
        self.invalid_actions_count = 0
        self.last_action: Any | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if gym is not None:
            super().reset(seed=seed)

        options = options or {}
        seed = options.get("seed", seed)
        reset_map_type = options.get("map_type", self.map_type)
        if reset_map_type != self.map_type:
            raise ValueError("map_type is fixed at construction to keep action_space stable")
        if seed is not None:
            random.seed(seed)

        catan_map = self.map_module.build_map(self.map_type)
        for player in self.players:
            player.reset_state()

        self.game = self.Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=int(options.get("vps_to_win", self.vps_to_win)),
        )
        self.invalid_actions_count = 0
        self.last_action = None
        self._advance_until_actor_decision()
        return self._get_observation(), self._info()

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._require_game()

        try:
            action_index = int(action)
            catan_action = self.action_catalog.decode(action_index, self.p0_color)
        except (TypeError, ValueError, IndexError):
            return self._invalid_action_step()

        if catan_action not in self.game.playable_actions:
            return self._invalid_action_step()

        self.last_action = catan_action
        self.game.execute(catan_action)
        self._advance_until_actor_decision()

        terminated = self.game.winning_color() is not None
        truncated = self.game.state.num_turns >= self.turns_limit
        reward = float(self.reward_function(catan_action, self.game, self.p0_color))
        return self._get_observation(), reward, terminated, truncated, self._info()

    def action_masks(self) -> Any:
        self._require_game()
        return self.action_catalog.mask(self.game.playable_actions)

    def get_valid_actions(self) -> tuple[int, ...]:
        self._require_game()
        return self.action_catalog.valid_actions(self.game.playable_actions)

    def sample_valid_action(self, rng: random.Random | None = None) -> int:
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            raise RuntimeError("no mapped valid actions are currently available")
        chooser = rng.choice if rng is not None else random.choice
        return int(chooser(valid_actions))

    def decode_action(self, action: int) -> dict[str, Any]:
        return self.action_catalog.describe(int(action))

    def close(self) -> None:
        self.game = None

    @property
    def unwrapped(self) -> "CatanatronRLEnv":
        return self

    def _load_catanatron_symbols(self) -> None:
        game_module = import_catanatron_module("catanatron.game")
        map_module = import_catanatron_module("catanatron.models.map")
        player_module = import_catanatron_module("catanatron.models.player")
        features_module = import_catanatron_module("catanatron.features")

        self.Game = game_module.Game
        self.turns_limit = int(game_module.TURNS_LIMIT)
        self.map_module = map_module
        self.Player = player_module.Player
        self.RandomPlayer = player_module.RandomPlayer
        self.Color = player_module.Color
        self.features_module = features_module

    def _build_players(self, enemies: Iterable[Any] | None) -> list[Any]:
        if enemies is not None:
            enemy_players = list(enemies)
        else:
            num_players = int(self.config.get("num_players", 4))
            if num_players < 2 or num_players > 4:
                raise ValueError("num_players must be between 2 and 4")
            enemy_colors = [self.Color.RED, self.Color.ORANGE, self.Color.WHITE]
            enemy_players = [
                self.RandomPlayer(color) for color in enemy_colors[: num_players - 1]
            ]

        if any(player.color == self.p0_color for player in enemy_players):
            raise ValueError("enemy players must not use Color.BLUE")
        return [self.Player(self.p0_color)] + enemy_players

    def _advance_until_actor_decision(self) -> None:
        self._require_game()
        while (
            self.game.winning_color() is None
            and self.game.state.num_turns < self.turns_limit
            and self.game.state.current_color() != self.p0_color
        ):
            self.game.play_tick()

    def _get_observation(self) -> Any:
        self._require_game()
        sample = self.features_module.create_sample(self.game, self.p0_color)
        return np.asarray([sample[name] for name in self.features], dtype=self.dtype)

    def _info(self) -> dict[str, Any]:
        self._require_game()
        mask = self.action_masks()
        valid_actions = tuple(
            int(index) for index, is_valid in enumerate(mask) if bool(is_valid)
        )
        return {
            "valid_actions": valid_actions,
            "action_mask": mask,
            "action_mask_version": self.action_catalog.version,
            "unmapped_valid_actions": self.action_catalog.unmapped_actions(
                self.game.playable_actions
            ),
            "invalid_actions_count": self.invalid_actions_count,
        }

    def _invalid_action_step(
        self,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._require_game()
        self.invalid_actions_count += 1
        terminated = self.game.winning_color() is not None
        truncated = (
            self.invalid_actions_count > self.max_invalid_actions
            or self.game.state.num_turns >= self.turns_limit
        )
        return (
            self._get_observation(),
            self.invalid_action_reward,
            terminated,
            truncated,
            self._info(),
        )

    def _require_game(self) -> None:
        if self.game is None:
            raise RuntimeError("environment has not been reset")
