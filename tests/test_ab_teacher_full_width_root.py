from __future__ import annotations

import random

import numpy as np

from catan_zero.rl._catanatron import import_catanatron_module


def _make_two_player_game(seed: int):
    game_module = import_catanatron_module("catanatron.game")
    player_module = import_catanatron_module("catanatron.models.player")
    Game = game_module.Game
    Color = player_module.Color
    RandomPlayer = player_module.RandomPlayer

    rng = random.Random(seed)
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players)
    while game.state.is_initial_build_phase:
        game.execute(rng.choice(game.playable_actions))
    return game, rng


def _find_multi_victim_robber_decision(seed: int, budget: int = 3000):
    enums_module = import_catanatron_module("catanatron.models.enums")
    ActionType = enums_module.ActionType

    game, rng = _make_two_player_game(seed)
    for _ in range(budget):
        move_robber_actions = [
            a for a in game.playable_actions if a.action_type == ActionType.MOVE_ROBBER
        ]
        if len(move_robber_actions) > 1:
            return game, move_robber_actions
        game.execute(rng.choice(game.playable_actions))
    raise AssertionError("could not find a multi-victim robber decision within budget")


def test_full_width_root_scores_every_root_child_at_robber_decisions() -> None:
    """FIX A4 (direct mechanism test): with full_width_root=True, the ROOT ply's action set
    is the full unpruned game.playable_actions, so every MOVE_ROBBER root child gets a
    DebugActionNode (and therefore a score) -- soft_score_legal_coverage == 1.0."""
    minimax_module = import_catanatron_module("catanatron.players.minimax")
    AlphaBetaPlayer = minimax_module.AlphaBetaPlayer
    DebugStateNode = minimax_module.DebugStateNode

    game, move_robber_actions = _find_multi_victim_robber_decision(seed=2)
    color = game.state.current_color()

    player = AlphaBetaPlayer(color, depth=1, prunning=True, full_width_root=True)
    node = DebugStateNode("root", color)
    player.alphabeta(game.copy(), player.depth, float("-inf"), float("inf"), float("inf"), node)

    assert len(node.children) == len(game.playable_actions) == len(move_robber_actions)


def test_pruned_root_collapses_robber_decision_to_one_child() -> None:
    """Documents the bug this fix targets: with full_width_root left at its default (False),
    prune_robber_actions collapses a multi-victim robber decision down to ONE root child,
    which is exactly the 1/18 soft_score_legal_coverage the audit measured."""
    minimax_module = import_catanatron_module("catanatron.players.minimax")
    AlphaBetaPlayer = minimax_module.AlphaBetaPlayer
    DebugStateNode = minimax_module.DebugStateNode

    game, move_robber_actions = _find_multi_victim_robber_decision(seed=2)
    assert len(move_robber_actions) > 1
    color = game.state.current_color()

    player = AlphaBetaPlayer(color, depth=1, prunning=True)  # full_width_root defaults False
    node = DebugStateNode("root", color)
    player.alphabeta(game.copy(), player.depth, float("-inf"), float("inf"), float("inf"), node)

    assert len(node.children) == 1


def test_ab_teacher_scores_full_coverage_at_multi_victim_robber_decision() -> None:
    """End-to-end: CatanatronAlphaBetaPolicy._root_search must score EVERY legal action at a
    real multi-victim robber decision reached through ColonistMultiAgentEnv."""
    import_catanatron_module("catanatron")
    enums_module = import_catanatron_module("catanatron.models.enums")
    ActionType = enums_module.ActionType

    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
    from catan_zero.rl.self_play import CatanatronAlphaBetaPolicy, make_env_config

    config = make_env_config(vps_to_win=3)
    env = ColonistMultiAgentEnv(config)
    rng = np.random.default_rng(0)
    try:
        _, info = env.reset(seed=2)
        found = False
        for _ in range(400):
            move_robber_actions = [
                a for a in env.game.playable_actions if a.action_type == ActionType.MOVE_ROBBER
            ]
            if len(move_robber_actions) > 1:
                found = True
                break
            valid = info["valid_actions"]
            action = int(rng.choice(np.asarray(valid, dtype=np.int64)))
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                _, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        assert found, "could not reach a multi-victim robber decision within step budget"

        policy = CatanatronAlphaBetaPolicy(depth=1)  # depth=1 keeps this test fast
        _, scores = policy._root_search(env, info)

        assert len(scores) == len(info["valid_actions"]) == len(move_robber_actions)
    finally:
        env.close()
