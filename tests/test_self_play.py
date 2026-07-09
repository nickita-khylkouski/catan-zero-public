import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tools.train_ppo as train_ppo_module

from catan_zero.rl.action_features import (
    CONTEXT_ACTION_FEATURE_SIZE,
    build_action_context_feature_table,
    _scaled_player_public_vp,
)
from catan_zero.rl import (
    CatanatronValuePolicy,
    HeuristicPolicy,
    JSettlersLitePolicy,
    LinearSoftmaxPolicy,
    NumpyMLPPolicy,
    OnePlySearchPolicy,
    RandomPolicy,
    ValueRolloutSearchPolicy,
    collect_imitation_game,
    create_linear_policy,
    create_mlp_policy,
    evaluate_policy,
    flatten_episode_for_reanalysis,
    load_reanalysis_jsonl,
    play_game,
    write_reanalysis_jsonl,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
from catan_zero.rl.self_play import (
    StepSample,
    make_env_config,
    should_record_imitation_sample,
)
from catan_zero.rl.reanalysis import (
    reanalysis_record_to_sample,
    sample_to_reanalysis_record,
)
from catan_zero.rl.torch_ppo import (
    PPOTrajectory,
    TorchPPOPolicy,
    _clipped_value_delta_reward,
    _discounted_terminal_returns,
    _expected_sarsa_q_targets,
    _gae_returns,
    _gae_terminal_returns,
    _normalize_observation,
    _old_q_policy_baselines,
    _ppo_value_loss,
    _resize_context_array,
    _scoreboard_rewards as _ppo_scoreboard_rewards,
    _score_margin_loss,
    _target_policy_tensor,
    _target_score_tensors,
    _weighted_mean,
    build_action_feature_table,
    collect_dagger_episode,
    create_ppo_policy,
    imitation_update,
    make_imitation_optimizer,
    ppo_update,
    update_ema_policy,
)
from catan_zero.rl.self_play import _scoreboard_values
from tools.train_ppo import (
    _append_policy_snapshot,
    _append_replay,
    _best_iteration_checkpoint,
    _best_training_checkpoint,
    _best_warmup_checkpoint,
    _effective_q_advantage_mix,
    _eval_win_lower_bound,
    _gated_q_advantage_mix,
    _make_opponent,
    _sample_checkpoint_opponent,
    _should_write_final_warmup_checkpoint,
    _uses_anchor_update,
    _weighted_index,
)
from tools.evaluate_self_play import _evaluate_policy_parallel, _progress_callback
from tools.evaluate_openings import evaluate_openings
from tools.run_self_play_ladder import EvalScore, build_train_command, should_promote


def test_play_game_collects_self_play_episode() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(vps_to_win=3)
    policy = create_linear_policy(config=config, seed=1)
    policies = {name: policy for name in ("BLUE", "RED", "ORANGE", "WHITE")}

    episode = play_game(
        policies,
        seed=2,
        config=config,
        max_decisions=80,
        rng=np.random.default_rng(3),
        training_policy=policy,
    )

    assert episode.result.decisions > 0
    assert set(episode.result.rewards) == {"BLUE", "RED", "ORANGE", "WHITE"}
    assert episode.samples_by_player
    assert sum(len(samples) for samples in episode.samples_by_player.values()) > 0


def test_opening_evaluator_records_initial_placement_states() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    candidate = HeuristicPolicy()
    teacher = HeuristicPolicy()
    report = evaluate_openings(
        candidate,
        {"heuristic": teacher},
        games=1,
        seed=1729,
        vps_to_win=3,
        max_opening_decisions=8,
        sample_records=8,
    )

    assert report["opening_states"] == 8
    assert report["prompt_counts"]["BUILD_INITIAL_SETTLEMENT"] == 4
    assert report["prompt_counts"]["BUILD_INITIAL_ROAD"] == 4
    assert report["invalid_candidate_actions"] == 0
    agreement_rate = report["teacher_metrics"]["heuristic"]["agreement_rate"]
    assert 0.0 <= agreement_rate <= 1.0
    by_prompt = report["teacher_metrics_by_prompt"]["heuristic"]
    assert by_prompt["BUILD_INITIAL_SETTLEMENT"]["states"] == 4
    assert by_prompt["BUILD_INITIAL_ROAD"]["states"] == 4
    assert report["sample_records"]
    assert all(record["candidate_label"] for record in report["sample_records"])


def test_linear_policy_updates_and_round_trips(tmp_path) -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(vps_to_win=3)
    policy = create_linear_policy(config=config, seed=4, learning_rate=0.01)
    policies = {name: policy for name in ("BLUE", "RED", "ORANGE", "WHITE")}
    episode = play_game(
        policies,
        seed=5,
        config=config,
        max_decisions=80,
        rng=np.random.default_rng(6),
        training_policy=policy,
    )
    before = policy.weights.copy()

    policy.update_episode(episode)
    path = tmp_path / "policy.npz"
    policy.save(path)
    loaded = LinearSoftmaxPolicy.load(path)

    assert not np.array_equal(before, policy.weights)
    assert loaded.weights.shape == policy.weights.shape
    assert loaded.bias.shape == policy.bias.shape


def test_mlp_policy_updates_and_round_trips(tmp_path) -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(vps_to_win=3)
    policy = create_mlp_policy(config=config, seed=14, learning_rate=0.001, hidden_size=16)
    policies = {name: policy for name in ("BLUE", "RED", "ORANGE", "WHITE")}
    episode = play_game(
        policies,
        seed=15,
        config=config,
        max_decisions=80,
        rng=np.random.default_rng(16),
        training_policy=policy,
    )
    before = policy.w1.copy()

    policy.update_episode(episode)
    path = tmp_path / "policy_mlp.npz"
    policy.save(path)
    loaded = NumpyMLPPolicy.load(path)

    assert not np.array_equal(before, policy.w1)
    assert loaded.w1.shape == policy.w1.shape
    assert loaded.w2.shape == policy.w2.shape


def test_search_reanalysis_targets_round_trip_and_train(tmp_path) -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(vps_to_win=3)
    teacher = ValueRolloutSearchPolicy(
        candidate_limit=2,
        presearch_candidate_limit=2,
        rollout_decisions=1,
        rollout_samples=1,
    )
    episode = collect_imitation_game(
        teacher,
        seed=71,
        config=config,
        max_decisions=32,
        rng=np.random.default_rng(72),
    )
    samples, returns = flatten_episode_for_reanalysis(episode, gamma=0.995)

    assert samples
    assert len(samples) == len(returns)
    assert any(sample.target_policy for sample in samples)
    assert any(sample.target_scores for sample in samples)

    path = tmp_path / "reanalysis.jsonl"
    written = write_reanalysis_jsonl(
        path,
        samples,
        returns,
        metadata={"teacher": teacher.name},
    )
    loaded_samples, loaded_returns = load_reanalysis_jsonl(path)

    assert written == len(samples)
    assert len(loaded_samples) == len(samples)
    assert loaded_returns == returns
    assert loaded_samples[0].valid_actions == samples[0].valid_actions
    assert loaded_samples[0].target_policy == samples[0].target_policy
    assert loaded_samples[0].target_scores == samples[0].target_scores
    assert loaded_samples[0].sample_weight == samples[0].sample_weight
    assert loaded_samples[0].decision_index == samples[0].decision_index
    assert loaded_samples[0].action_context_features is not None

    policy = create_ppo_policy(
        config=config,
        seed=73,
        hidden_size=16,
        architecture="candidate",
    )
    update = imitation_update(
        policy,
        loaded_samples,
        learning_rate=1e-3,
        epochs=1,
        minibatch_size=8,
        returns=loaded_returns,
        value_coef=0.25,
        score_coef=0.05,
    )

    assert update["samples"] == float(len(loaded_samples))
    assert update["policy_loss"] > 0.0


def test_reanalysis_records_preserve_decision_index() -> None:
    sample = StepSample(
        observation=np.asarray([1.0, 2.0], dtype=np.float64),
        valid_actions=(1, 3),
        action=3,
        player="BLUE",
        decision_index=17,
    )

    record = sample_to_reanalysis_record(sample, return_value=0.25)
    loaded_sample, loaded_return = reanalysis_record_to_sample(record)

    assert record["decision_index"] == 17
    assert loaded_sample.decision_index == 17
    assert loaded_return == 0.25


def test_should_record_imitation_sample_supports_midgame_windows() -> None:
    assert not should_record_imitation_sample(
        3,
        record_after_decisions=4,
        record_until_decision=8,
    )
    assert should_record_imitation_sample(
        4,
        record_after_decisions=4,
        record_until_decision=8,
    )
    assert should_record_imitation_sample(
        7,
        record_after_decisions=4,
        record_until_decision=8,
    )
    assert not should_record_imitation_sample(
        8,
        record_after_decisions=4,
        record_until_decision=8,
    )


def test_collect_imitation_game_can_record_later_decision_window() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    episode = collect_imitation_game(
        RandomPolicy(),
        seed=91,
        config=make_env_config(vps_to_win=3),
        max_decisions=12,
        record_after_decisions=4,
        record_until_decision=8,
        rng=np.random.default_rng(92),
    )
    samples = [
        sample
        for player_samples in episode.samples_by_player.values()
        for sample in player_samples
    ]

    assert samples
    assert all(sample.decision_index is not None for sample in samples)
    assert all(4 <= int(sample.decision_index) < 8 for sample in samples)


def test_train_ppo_writes_reanalysis_checkpoint(tmp_path) -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(vps_to_win=3)
    teacher = ValueRolloutSearchPolicy(
        candidate_limit=2,
        presearch_candidate_limit=2,
        rollout_decisions=1,
        rollout_samples=1,
    )
    episode = collect_imitation_game(
        teacher,
        seed=81,
        config=config,
        max_decisions=12,
        rng=np.random.default_rng(82),
    )
    samples, returns = flatten_episode_for_reanalysis(episode, gamma=0.995)
    reanalysis_path = tmp_path / "reanalysis.jsonl"
    checkpoint = tmp_path / "post_reanalysis.pt"
    report = tmp_path / "train.json"
    write_reanalysis_jsonl(reanalysis_path, samples, returns)

    subprocess.run(
        [
            sys.executable,
            "tools/train_ppo.py",
            "--seed",
            "83",
            "--vps-to-win",
            "3",
            "--max-decisions",
            "12",
            "--hidden-size",
            "8",
            "--architecture",
            "candidate",
            "--teacher",
            "value",
            "--warmup-games",
            "0",
            "--iterations",
            "0",
            "--eval-games",
            "0",
            "--eval-value-games",
            "0",
            "--reanalysis-input",
            str(reanalysis_path),
            "--reanalysis-epochs",
            "1",
            "--reanalysis-checkpoint",
            str(checkpoint),
            "--checkpoint",
            str(tmp_path / "final.pt"),
            "--report",
            str(report),
            "--minibatch-size",
            "8",
        ],
        check=True,
    )

    assert checkpoint.exists()
    assert report.exists()


def test_anchor_value_coef_is_wired_independently(monkeypatch, tmp_path) -> None:
    class FakePolicy:
        architecture = "flat"
        hidden_size = 8
        use_action_id_embedding = True

        def save(self, path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fake checkpoint")

    sample = StepSample(
        observation=np.zeros(3),
        valid_actions=(0, 1),
        action=1,
        player="BLUE",
    )
    episode = argparse.Namespace(
        samples_by_player={"BLUE": [sample]},
        result=argparse.Namespace(
            rewards={"BLUE": 1.0},
            winner="BLUE",
            decisions=1,
        ),
    )
    imitation_value_coefs = []
    imitation_train_critic = []

    def fake_imitation_update(*args, **kwargs):
        imitation_value_coefs.append(kwargs["value_coef"])
        return {
            "samples": float(len(args[1])),
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "score_loss": 0.0,
        }

    def fake_make_imitation_optimizer(*args, **kwargs):
        imitation_train_critic.append(kwargs["train_critic"])
        return object()

    monkeypatch.setattr(
        train_ppo_module,
        "create_ppo_policy",
        lambda **kwargs: FakePolicy(),
    )
    monkeypatch.setattr(
        train_ppo_module,
        "make_ppo_optimizer",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        train_ppo_module,
        "make_imitation_optimizer",
        fake_make_imitation_optimizer,
    )
    monkeypatch.setattr(train_ppo_module, "imitation_update", fake_imitation_update)
    monkeypatch.setattr(
        train_ppo_module,
        "collect_imitation_game",
        lambda *args, **kwargs: episode,
    )
    monkeypatch.setattr(
        train_ppo_module,
        "ppo_update",
        lambda *args, **kwargs: {
            "samples": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "q_value_loss": 0.0,
            "q_chosen_return_corr": 0.0,
            "q_legal_std": 0.0,
            "q_legal_spread_entropy": 0.0,
            "q_advantage_sign_agreement": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "old_policy_kl": 0.0,
            "clip_fraction": 0.0,
            "mean_shaped_reward": 0.0,
            "minibatches": 0.0,
            "early_stop": 0.0,
        },
    )
    monkeypatch.setattr(
        train_ppo_module,
        "evaluate_teacher_agreement",
        lambda *args, **kwargs: {
            "accuracy": 1.0,
            "mean_teacher_log_prob": 0.0,
        },
    )
    monkeypatch.setattr(
        train_ppo_module,
        "evaluate_policy",
        lambda *args, **kwargs: {"games": kwargs["games"], "win_rate": 0.0},
    )
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--seed",
            "91",
            "--hidden-size",
            "8",
            "--teacher",
            "heuristic",
            "--warmup-games",
            "0",
            "--warmup-value-coef",
            "0.75",
            "--iterations",
            "1",
            "--episodes-per-iteration",
            "0",
            "--anchor-games-per-iteration",
            "1",
            "--anchor-epochs",
            "1",
            "--eval-games",
            "0",
            "--eval-value-games",
            "0",
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--report",
            str(report),
        ],
    )

    train_ppo_module.main()

    saved_report = json.loads(report.read_text())
    assert imitation_train_critic == [True, False]
    assert imitation_value_coefs == [0.0]
    assert saved_report["config"]["warmup_value_coef"] == 0.75
    assert saved_report["config"]["anchor_value_coef"] == 0.0


def test_reanalysis_records_store_only_valid_action_context() -> None:
    context = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    sample = StepSample(
        observation=np.asarray([1.0, 2.0], dtype=np.float64),
        valid_actions=(1, 3),
        action=3,
        player="BLUE",
        action_context_features=context,
        target_policy={3: 1.0},
        sample_weight=2.5,
    )

    record = sample_to_reanalysis_record(sample, return_value=0.25)

    assert record["sample_weight"] == 2.5
    assert record["action_context_features"] is None
    assert record["action_context_storage"] == "valid_actions"
    assert record["action_context_action_size"] == 5
    assert record["action_context_feature_size"] == 3
    assert record["valid_action_context_features"] == [
        context[1].tolist(),
        context[3].tolist(),
    ]


def test_linear_policy_bootstraps_from_heuristic_trace() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(vps_to_win=3)
    policy = create_linear_policy(config=config, seed=8, learning_rate=0.01)
    before = policy.weights.copy()
    episode = collect_imitation_game(
        HeuristicPolicy(),
        seed=9,
        config=config,
        max_decisions=80,
        rng=np.random.default_rng(10),
    )

    policy.update_imitation(episode)

    assert episode.samples_by_player
    assert not np.array_equal(before, policy.weights)


def test_evaluate_policy_reports_win_rate_and_elo() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    report = evaluate_policy(
        HeuristicPolicy(),
        RandomPolicy(),
        games=2,
        seed=7,
        config=make_env_config(vps_to_win=3),
        max_decisions=80,
    )

    assert report["games"] == 2
    assert 0.0 <= report["win_rate"] <= 1.0
    assert "elo_vs_opponent" in report


def test_parallel_evaluator_matches_serial_report() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    serial = evaluate_policy(
        HeuristicPolicy(),
        RandomPolicy(),
        games=4,
        seed=71,
        config=make_env_config(vps_to_win=3),
        max_decisions=80,
    )
    parallel = _evaluate_policy_parallel(
        argparse.Namespace(
            candidate="heuristic",
            checkpoint=None,
            opponent="random",
            games=4,
            seed=71,
            vps_to_win=3,
            max_decisions=80,
            search_candidate_limit=48,
            search_presearch_candidate_limit=0,
            search_rollout_decisions=8,
            search_rollout_samples=1,
            search_root_value_weight=0.0,
            search_opponent_penalty=0.05,
            opponent_candidate_limit=48,
            opponent_rollout_decisions=8,
            opponent_value_penalty=0.05,
            progress_every=0,
            workers=2,
        )
    )

    for key in (
        "games",
        "candidate",
        "opponent",
        "wins",
        "win_rate",
        "seat_wins",
        "avg_decisions",
    ):
        assert parallel[key] == serial[key]
    assert parallel["workers"] == 2


def test_one_ply_search_returns_legal_action() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        observations, info = env.reset(seed=11)
        player = info["current_player"]
        policy = OnePlySearchPolicy(candidate_limit=8)

        action = policy.select_action(
            env,
            np.asarray(observations[player], dtype=np.float64),
            info,
            np.random.default_rng(12),
        )

        assert action in info["valid_actions"]
    finally:
        env.close()


def test_catanatron_value_policy_returns_legal_action() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        observations, info = env.reset(seed=17)
        player = info["current_player"]
        policy = CatanatronValuePolicy(candidate_limit=16)

        action = policy.select_action(
            env,
            np.asarray(observations[player], dtype=np.float64),
            info,
            np.random.default_rng(18),
        )

        assert action in info["valid_actions"]
    finally:
        env.close()


def test_action_context_features_cover_current_legal_actions() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=71)

        table = build_action_context_feature_table(env, info)

        assert table.shape == (env.action_space.n, CONTEXT_ACTION_FEATURE_SIZE)
        for action in info["valid_actions"]:
            assert table[int(action), 0] == 1.0
        assert table[:, 1].max() > 0.0
    finally:
        env.close()


def test_action_context_features_mark_opening_and_port_settlements() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=72)
        table = build_action_context_feature_table(env, info)
        port_nodes = {
            int(node)
            for port in env.observation_payload(include_event_log=False)["board"]["ports"]
            for node in port["nodes"]
        }
        settlement_actions = [
            action
            for action in info["structured_legal_actions"]
            if action["action_type"] == "BUILD_SETTLEMENT"
        ]
        assert settlement_actions
        assert all(table[int(action["index"]), 12] == 1.0 for action in settlement_actions)
        port_settlements = [
            action
            for action in settlement_actions
            if int(action["args"]["node"]) in port_nodes
        ]
        assert port_settlements
        assert any(table[int(action["index"]), 13] == 1.0 for action in port_settlements)
    finally:
        env.close()


def test_maritime_trade_context_features_include_want_bundle() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        for seed in range(200, 260):
            _, info = env.reset(seed=seed)
            for _ in range(120):
                table = build_action_context_feature_table(env, info)
                for action in info["valid_actions"]:
                    structured = env.structured_action(int(action))
                    if structured and structured["action_type"] == "MARITIME_TRADE":
                        assert structured["args"].get("want")
                        assert table[int(action), 8] > 0.0
                        assert table[int(action), 9] < 0.0
                        return
                action = int(np.random.default_rng(seed).choice(info["valid_actions"]))
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        pytest.fail("did not encounter a legal maritime trade in search window")
    finally:
        env.close()


def test_trade_response_context_features_include_current_offer() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=22)
        rng = np.random.default_rng(1)
        for _ in range(300):
            offer_actions = [
                action
                for action in info["structured_legal_actions"]
                if action["action_type"] == "offer_trade"
            ]
            if offer_actions:
                _, _, terminated, truncated, info = env.step(offer_actions[0]["index"])
                assert not (terminated or truncated)
                table = build_action_context_feature_table(env, info)
                response_actions = [
                    action
                    for action in info["structured_legal_actions"]
                    if action["action_type"] in ("accept_trade", "reject_trade")
                ]
                assert response_actions
                response = response_actions[0]
                assert table[int(response["index"]), 7] > 0.0
                assert table[int(response["index"]), 8] > 0.0
                return
            action = int(rng.choice(info["valid_actions"]))
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        pytest.fail("did not encounter a legal player-trade response")
    finally:
        env.close()


def test_robber_victim_context_feature_accepts_color_like_victim() -> None:
    class FakeColor:
        name = "RED"

        def __str__(self) -> str:
            return "Color.RED"

    payload = {
        "players": {
            "BLUE": {"public_victory_points": 2},
            "RED": {"public_victory_points": 5},
        }
    }

    assert _scaled_player_public_vp(payload, FakeColor()) == 0.5
    assert _scaled_player_public_vp(payload, "BLUE") == 0.2


def test_contextual_candidate_policy_updates_and_round_trips(tmp_path) -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(vps_to_win=3)
    policy = create_ppo_policy(
        config=config,
        seed=72,
        hidden_size=16,
        architecture="candidate",
    )
    assert policy.context_action_feature_size == CONTEXT_ACTION_FEATURE_SIZE
    episode = collect_imitation_game(
        HeuristicPolicy(),
        seed=73,
        config=config,
        max_decisions=40,
        rng=np.random.default_rng(74),
    )
    samples = [
        sample
        for player_samples in episode.samples_by_player.values()
        for sample in player_samples
    ]
    optimizer = make_imitation_optimizer(policy, learning_rate=0.001)

    update = imitation_update(
        policy,
        samples,
        learning_rate=0.001,
        epochs=1,
        minibatch_size=32,
        optimizer=optimizer,
    )
    path = tmp_path / "contextual.pt"
    policy.save(path)
    loaded = TorchPPOPolicy.load(path)

    assert update["samples"] == float(len(samples))
    assert loaded.context_action_feature_size == CONTEXT_ACTION_FEATURE_SIZE


def test_imitation_update_reports_sample_weights() -> None:
    pytest.importorskip("torch")
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=75)
    samples = [
        StepSample(
            observation=np.zeros(3),
            valid_actions=(0, 1),
            action=0,
            player="BLUE",
            sample_weight=3.0,
        ),
        StepSample(
            observation=np.zeros(3),
            valid_actions=(0, 1),
            action=1,
            player="RED",
            sample_weight=1.0,
        ),
    ]

    update = imitation_update(
        policy,
        samples,
        learning_rate=0.0,
        epochs=1,
        minibatch_size=2,
    )

    assert update["samples"] == 2.0
    assert update["mean_sample_weight"] == 2.0


def test_weighted_mean_prioritizes_high_weight_losses() -> None:
    torch = pytest.importorskip("torch")
    losses = torch.tensor([1.0, 3.0])
    weights = torch.tensor([3.0, 1.0])

    assert torch.allclose(_weighted_mean(losses, weights), torch.tensor(1.5))


def test_candidate_policy_resizes_newer_context_features_for_old_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    policy = TorchPPOPolicy(
        3,
        5,
        hidden_size=8,
        seed=78,
        architecture="candidate",
        action_features=np.zeros((5, 2), dtype=np.float32),
        context_action_feature_size=3,
    )
    newer_context = np.ones((1, 5, 6), dtype=np.float32)

    logits, values = policy.forward(
        torch.as_tensor(np.zeros((1, 3), dtype=np.float32)),
        newer_context,
    )

    assert tuple(logits.shape) == (1, 5)
    assert tuple(values.shape) == (1,)


def test_resize_context_array_pads_older_samples_for_current_policy() -> None:
    value = np.ones((5, 2), dtype=np.float32)

    resized = _resize_context_array(value, feature_size=4)

    assert resized.shape == (5, 4)
    assert np.all(resized[:, :2] == 1.0)
    assert np.all(resized[:, 2:] == 0.0)


def test_resize_context_array_truncates_newer_samples_for_old_policy() -> None:
    value = np.arange(20, dtype=np.float32).reshape(5, 4)

    resized = _resize_context_array(value, feature_size=2)

    assert resized.shape == (5, 2)
    assert np.all(resized == value[:, :2])


def test_value_rollout_search_returns_legal_action() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        observations, info = env.reset(seed=19)
        player = info["current_player"]
        policy = ValueRolloutSearchPolicy(candidate_limit=4, rollout_decisions=2)

        action = policy.select_action(
            env,
            np.asarray(observations[player], dtype=np.float64),
            info,
            np.random.default_rng(20),
        )

        assert action in info["valid_actions"]
    finally:
        env.close()


def test_jsettlers_lite_policy_returns_legal_action() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        observations, info = env.reset(seed=41)
        player = info["current_player"]
        policy = JSettlersLitePolicy()

        action = policy.select_action(
            env,
            np.asarray(observations[player], dtype=np.float64),
            info,
            np.random.default_rng(42),
        )

        assert action in info["valid_actions"]
    finally:
        env.close()


def test_jsettlers_lite_policy_exposes_resource_plan_scores() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=42)
        policy = JSettlersLitePolicy()
        scores = policy.target_scores(env, info, np.random.default_rng(43))
        valid_actions = set(info["valid_actions"])

        assert scores
        assert set(scores).issubset(valid_actions)
        assert all(np.isfinite(score) for score in scores.values())
        assert max(scores, key=scores.get) in valid_actions
    finally:
        env.close()


def test_baseline_mixed_teacher_exposes_blended_scores() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=43)
        teacher = train_ppo_module.BaselineMixedTeacherPolicy()
        scores = teacher.target_scores(env, info, np.random.default_rng(44))
        valid_actions = set(info["valid_actions"])

        assert scores
        assert set(scores).issubset(valid_actions)
        assert all(np.isfinite(score) for score in scores.values())
        if len(scores) > 1:
            assert len({round(score, 6) for score in scores.values()}) > 1
    finally:
        env.close()


def test_baseline_rollout_mixed_teacher_exposes_rollout_guard_scores() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=45)
        teacher = train_ppo_module.BaselineRolloutMixedTeacherPolicy(
            candidate_limit=4,
            presearch_candidate_limit=8,
            rollout_decisions=1,
            rollout_samples=1,
        )
        scores = teacher.target_scores(env, info, np.random.default_rng(46))
        valid_actions = set(info["valid_actions"])

        assert scores
        assert set(scores).issubset(valid_actions)
        assert all(np.isfinite(score) for score in scores.values())
        if len(scores) > 1:
            assert len({round(score, 6) for score in scores.values()}) > 1
    finally:
        env.close()


def test_tactical_rollout_score_map_preserves_baseline_action_class() -> None:
    info = {
        "structured_legal_actions": [
            {"index": 1, "action_type": "BUILD_CITY"},
            {"index": 2, "action_type": "BUILD_CITY"},
            {"index": 3, "action_type": "BUILD_ROAD"},
            {"index": 4, "action_type": "BUILD_ROAD"},
        ]
    }

    scores = train_ppo_module._tactical_rollout_score_map(
        info,
        baseline_scores={1: 10.0, 2: 9.0, 3: 1.0, 4: 0.0},
        rollout_scores={1: 0.0, 2: 1.0, 3: 100.0, 4: 101.0},
    )

    assert max(scores, key=scores.get) == 2
    assert scores[2] > scores[1]
    assert scores[2] > scores[4]


def test_parallel_evaluator_accepts_jsettlers_lite_opponent() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    args = argparse.Namespace(
        candidate="heuristic",
        checkpoint=None,
        opening_checkpoint=None,
        opening_prompts="",
        opponent="jsettlers_lite",
        games=1,
        seed=43,
        vps_to_win=3,
        max_decisions=80,
        search_candidate_limit=4,
        search_presearch_candidate_limit=0,
        search_rollout_decisions=1,
        search_rollout_samples=1,
        search_root_value_weight=0.0,
        search_opponent_penalty=0.05,
        opponent_candidate_limit=4,
        opponent_rollout_decisions=1,
        opponent_value_penalty=0.05,
        progress_every=0,
        workers=1,
    )

    report = _evaluate_policy_parallel(args)

    assert report["opponent"] == "jsettlers_lite"
    assert report["games"] == 1


def test_value_rollout_search_exposes_legal_teacher_targets() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=21)
        policy = ValueRolloutSearchPolicy(candidate_limit=4, rollout_decisions=1)
        rng = np.random.default_rng(22)

        target_policy = policy.target_policy(env, info, rng)
        target_scores = policy.target_scores(env, info, rng)
        valid_actions = set(info["valid_actions"])

        assert target_policy
        assert set(target_policy).issubset(valid_actions)
        assert np.isclose(sum(target_policy.values()), 1.0)
        assert target_scores
        assert set(target_scores).issubset(valid_actions)
        assert all(np.isfinite(score) for score in target_scores.values())
    finally:
        env.close()


def test_value_rollout_search_reuses_teacher_scores_for_same_state() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        observations, info = env.reset(seed=23)
        player = info["current_player"]
        policy = ValueRolloutSearchPolicy(candidate_limit=4, rollout_decisions=1)
        policy._value_policy._value_fn = lambda game, color: 0.0
        calls = 0

        def fake_score_action(env_arg, action_index, actor_color, *, value_fn):
            nonlocal calls
            calls += 1
            return float(action_index)

        policy._score_action = fake_score_action
        rng = np.random.default_rng(24)

        action = policy.select_action(
            env,
            np.asarray(observations[player], dtype=np.float64),
            info,
            rng,
        )
        target_policy = policy.target_policy(env, info, rng)
        target_scores = policy.target_scores(env, info, rng)

        assert action in info["valid_actions"]
        assert target_policy
        assert target_scores
        assert calls <= policy.candidate_limit
    finally:
        env.close()


def test_value_rollout_search_averages_multiple_root_samples() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        observations, info = env.reset(seed=25)
        player = info["current_player"]
        policy = ValueRolloutSearchPolicy(
            candidate_limit=4,
            rollout_decisions=1,
            rollout_samples=3,
        )
        policy._value_policy._value_fn = lambda game, color: 0.0
        calls = 0

        def fake_score_action(env_arg, action_index, actor_color, *, value_fn):
            nonlocal calls
            calls += 1
            return float(action_index + calls * 0.001)

        policy._score_action = fake_score_action
        rng = np.random.default_rng(26)

        action = policy.select_action(
            env,
            np.asarray(observations[player], dtype=np.float64),
            info,
            rng,
        )
        target_policy = policy.target_policy(env, info, rng)

        assert action in info["valid_actions"]
        assert target_policy
        assert calls == policy.candidate_limit * policy.rollout_samples
    finally:
        env.close()


def test_value_rollout_search_presearch_prunes_by_value_scores() -> None:
    policy = ValueRolloutSearchPolicy(
        candidate_limit=2,
        presearch_candidate_limit=5,
        rollout_decisions=1,
    )
    policy._value_policy._candidate_actions = lambda env, info, rng: (1, 2, 3, 4, 5)
    policy._value_policy._score_candidates = lambda env, candidates, actor: [
        (0.0, 1),
        (4.0, 2),
        (2.0, 3),
        (float("-inf"), 4),
        (3.0, 5),
    ]

    candidates = policy._candidate_actions(
        object(),
        {},
        np.random.default_rng(27),
        "BLUE",
    )

    assert candidates == (2, 5)


def test_value_rollout_search_keeps_full_opening_candidate_set() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=28)
        policy = ValueRolloutSearchPolicy(
            candidate_limit=4,
            presearch_candidate_limit=64,
            rollout_decisions=1,
        )

        candidates = policy._candidate_actions(
            env,
            info,
            np.random.default_rng(29),
            env.current_player_color(),
        )

        assert "INITIAL" in info["current_prompt"]
        assert len(candidates) > policy.candidate_limit
        assert set(candidates).issubset(set(info["valid_actions"]))
    finally:
        env.close()


def test_value_rollout_search_blends_root_and_rollout_scores() -> None:
    class FakeEnv:
        game = object()

    policy = ValueRolloutSearchPolicy(
        candidate_limit=2,
        rollout_decisions=1,
        root_value_weight=0.25,
    )
    policy._value_policy._get_value_fn = lambda: (lambda game, color: 0.0)
    policy._score_root_action = (
        lambda env, action_index, actor_color, *, value_fn: float(action_index * 10)
    )
    policy._score_action = (
        lambda env, action_index, actor_color, *, value_fn: float(action_index)
    )

    scored = dict(
        (action, score)
        for score, action in policy._score_candidates(
            FakeEnv(),
            {},
            (2, 4),
            "BLUE",
        )
    )

    assert scored[2] == pytest.approx(0.25 * 20.0 + 0.75 * 2.0)
    assert scored[4] == pytest.approx(0.25 * 40.0 + 0.75 * 4.0)


def test_one_ply_search_evaluates_without_invalid_actions() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    report = evaluate_policy(
        OnePlySearchPolicy(candidate_limit=8),
        RandomPolicy(),
        games=1,
        seed=13,
        config=make_env_config(vps_to_win=3),
        max_decisions=80,
    )

    assert report["games"] == 1
    assert 0.0 <= report["win_rate"] <= 1.0


def test_ppo_discounted_terminal_returns_are_per_player() -> None:
    returns = _discounted_terminal_returns(
        ["BLUE", "RED", "BLUE", "BLUE", "RED"],
        {"BLUE": 1.0, "RED": -1.0 / 3.0},
        gamma=0.5,
    )

    assert returns == [0.25, -1.0 / 6.0, 0.5, 1.0, -1.0 / 3.0]


def test_ppo_gae_matches_discounted_returns_with_zero_values_and_lambda_one() -> None:
    players = ["BLUE", "RED", "BLUE", "BLUE", "RED"]
    rewards = {"BLUE": 1.0, "RED": -1.0 / 3.0}

    returns, advantages = _gae_terminal_returns(
        players,
        rewards,
        [0.0] * len(players),
        gamma=0.5,
        gae_lambda=1.0,
    )

    expected = _discounted_terminal_returns(players, rewards, gamma=0.5)
    assert returns == expected
    assert advantages == expected


def test_ppo_gae_includes_shaped_rewards() -> None:
    players = ["BLUE", "BLUE"]
    rewards = {"BLUE": 1.0}

    returns, advantages = _gae_returns(
        players,
        rewards,
        [0.0, 0.0],
        [0.25, -0.5],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert returns == [0.75, 0.5]
    assert advantages == returns


def test_ppo_gae_bootstraps_time_limit_returns() -> None:
    returns, advantages = _gae_returns(
        ["BLUE"],
        {"BLUE": 0.0},
        [0.2],
        [0.0],
        gamma=0.5,
        gae_lambda=1.0,
        bootstrap_values={"BLUE": 0.8},
    )

    assert returns == pytest.approx([0.4])
    assert advantages == pytest.approx([0.2])


def test_value_delta_reward_is_scaled_and_clipped() -> None:
    assert _clipped_value_delta_reward(
        10.0,
        30.0,
        coef=0.5,
        scale=10.0,
    ) == 0.5
    assert _clipped_value_delta_reward(
        30.0,
        20.0,
        coef=0.5,
        scale=10.0,
    ) == -0.5
    assert _clipped_value_delta_reward(
        None,
        20.0,
        coef=0.5,
        scale=10.0,
    ) == 0.0


def test_ppo_truncation_rewards_use_rich_scoreboard() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        _, info = env.reset(seed=301)
        rng = np.random.default_rng(302)
        for _ in range(80):
            action = int(rng.choice(info["valid_actions"]))
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        values = _scoreboard_values(env)
        best = max(values.values())
        leaders = [player for player, value in values.items() if value == best]
        if len(leaders) != 1:
            pytest.skip("rich scoreboard tied in sampled state")

        rewards = _ppo_scoreboard_rewards(env)

        assert rewards[leaders[0]] == 1.0
        assert all(
            reward < 0.0 for player, reward in rewards.items() if player != leaders[0]
        )
    finally:
        env.close()


def test_ppo_value_loss_supports_clipped_critic_updates() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.2])
    returns = torch.tensor([1.0])
    old_values = torch.tensor([0.0])

    unclipped = _ppo_value_loss(values, returns, old_values, clip_range=0.0)
    clipped = _ppo_value_loss(values, returns, old_values, clip_range=0.1)

    assert torch.allclose(unclipped, torch.tensor(0.64))
    assert torch.allclose(clipped, torch.tensor(0.81))


def test_ppo_update_reports_old_policy_kl_penalty() -> None:
    pytest.importorskip("torch")
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=31)
    trajectory = PPOTrajectory(
        samples=[
            StepSample(
                observation=np.zeros(3),
                valid_actions=(0, 1, 2),
                action=0,
                player="BLUE",
            )
        ],
        returns=[1.0],
        advantages=[1.0],
        old_log_probs=[float(np.log(1.0 / 3.0))],
        old_values=[0.0],
        old_action_probs=[np.asarray([1.0 / 3.0] * 3, dtype=np.float32)],
        shaped_rewards=[0.0],
    )

    update = ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-4,
        clip_ratio=0.1,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=1,
        kl_coef=0.05,
    )

    assert update["samples"] == 1.0
    assert update["old_policy_kl"] >= 0.0
    assert update["minibatches"] == 1.0
    assert update["early_stop"] == 0.0


def test_ppo_update_can_filter_to_top_positive_advantages() -> None:
    pytest.importorskip("torch")
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=33)
    samples = [
        StepSample(
            observation=np.zeros(3),
            valid_actions=(0, 1, 2),
            action=0,
            player="BLUE",
        )
        for _ in range(4)
    ]
    trajectory = PPOTrajectory(
        samples=samples,
        returns=[-1.0, 0.2, 1.0, 2.0],
        advantages=[-1.0, 0.2, 1.0, 2.0],
        old_log_probs=[float(np.log(1.0 / 3.0))] * 4,
        old_values=[0.0] * 4,
        old_action_probs=[np.asarray([1.0 / 3.0] * 3, dtype=np.float32) for _ in range(4)],
        shaped_rewards=[0.0] * 4,
    )

    update = ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-4,
        clip_ratio=0.1,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=4,
        top_advantage_fraction=0.5,
        min_advantage_samples=1,
    )

    assert update["samples_before_filter"] == 4.0
    assert update["samples"] == 2.0
    assert update["advantage_filter_kept_fraction"] == 0.5
    assert update["advantage_filter_threshold"] == 1.0


def test_ppo_update_reports_ema_policy_kl_penalty() -> None:
    pytest.importorskip("torch")
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=31)
    ema_policy = policy.clone_frozen()
    trajectory = PPOTrajectory(
        samples=[
            StepSample(
                observation=np.zeros(3),
                valid_actions=(0, 1, 2),
                action=0,
                player="BLUE",
            )
        ],
        returns=[1.0],
        advantages=[1.0],
        old_log_probs=[float(np.log(1.0 / 3.0))],
        old_values=[0.0],
        old_action_probs=[np.asarray([1.0 / 3.0] * 3, dtype=np.float32)],
        shaped_rewards=[0.0],
    )

    update = ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-4,
        clip_ratio=0.1,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=1,
        ema_policy=ema_policy,
        ema_policy_kl_coef=0.05,
    )

    assert update["samples"] == 1.0
    assert update["ema_policy_kl"] >= 0.0


def test_update_ema_policy_moves_toward_source() -> None:
    pytest.importorskip("torch")
    import torch

    target = TorchPPOPolicy(3, 5, hidden_size=8, seed=41)
    source = TorchPPOPolicy(3, 5, hidden_size=8, seed=42)
    before = next(target.model.parameters()).detach().clone()
    source_param = next(source.model.parameters()).detach().clone()

    update_ema_policy(target, source, decay=0.25)

    after = next(target.model.parameters()).detach()
    assert torch.allclose(after, before * 0.25 + source_param * 0.75)


def test_ppo_update_trains_action_q_head() -> None:
    pytest.importorskip("torch")
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=33)
    trajectory = PPOTrajectory(
        samples=[
            StepSample(
                observation=np.zeros(3),
                valid_actions=(0, 1, 2),
                action=1,
                player="BLUE",
            )
        ],
        returns=[1.0],
        advantages=[1.0],
        old_log_probs=[float(np.log(1.0 / 3.0))],
        old_values=[0.0],
        old_action_probs=[np.asarray([1.0 / 3.0] * 3, dtype=np.float32)],
        shaped_rewards=[0.0],
        old_q_values=[0.0],
    )

    update = ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-4,
        clip_ratio=0.1,
        value_coef=0.0,
        q_value_coef=0.5,
        q_advantage_mix=0.25,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=1,
    )

    assert update["samples"] == 1.0
    assert update["q_value_loss"] > 0.0


def test_ppo_update_reports_finite_q_diagnostics() -> None:
    pytest.importorskip("torch")
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=34)
    trajectory = PPOTrajectory(
        samples=[
            StepSample(
                observation=np.zeros(3),
                valid_actions=(0, 1, 2),
                action=0,
                player="BLUE",
            ),
            StepSample(
                observation=np.ones(3),
                valid_actions=(0, 1, 2),
                action=2,
                player="BLUE",
            ),
        ],
        returns=[0.0, 1.0],
        advantages=[1.0, -1.0],
        old_log_probs=[float(np.log(1.0 / 3.0)), float(np.log(1.0 / 3.0))],
        old_values=[0.0, 0.0],
        old_action_probs=[np.asarray([1.0 / 3.0] * 3, dtype=np.float32)] * 2,
        shaped_rewards=[0.0, 0.0],
        old_q_values=[0.1, 0.9],
        old_action_q_values=[
            np.asarray([0.1, 0.0, -0.1], dtype=np.float32),
            np.asarray([0.2, 0.4, 0.9], dtype=np.float32),
        ],
    )

    update = ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-4,
        clip_ratio=0.1,
        value_coef=0.0,
        q_value_coef=0.5,
        q_advantage_mix=0.25,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=2,
    )

    for key in (
        "q_chosen_return_corr",
        "q_legal_std",
        "q_legal_spread_entropy",
        "q_advantage_sign_agreement",
    ):
        assert np.isfinite(update[key])
    assert update["q_chosen_return_corr"] > 0.99
    assert update["q_legal_std"] > 0.0
    assert 0.0 <= update["q_legal_spread_entropy"] <= 1.0
    assert update["q_advantage_sign_agreement"] == pytest.approx(0.5)


def test_q_advantage_baseline_uses_legal_policy_expectation() -> None:
    trajectory = PPOTrajectory(
        samples=[
            StepSample(
                observation=np.zeros(3),
                valid_actions=(0, 1, 2),
                action=1,
                player="BLUE",
            )
        ],
        returns=[1.0],
        advantages=[1.0],
        old_log_probs=[0.0],
        old_values=[0.2],
        old_action_probs=[np.asarray([0.25, 0.25, 0.50], dtype=np.float32)],
        shaped_rewards=[0.0],
        old_q_values=[0.7],
        old_action_q_values=[np.asarray([0.0, 1.0, 2.0], dtype=np.float32)],
    )

    baselines = _old_q_policy_baselines(
        [trajectory],
        fallback_values=np.asarray([0.2], dtype=np.float32),
    )

    assert baselines == pytest.approx([1.25])


def test_expected_sarsa_q_targets_bootstrap_same_player() -> None:
    trajectory = PPOTrajectory(
        samples=[
            StepSample(
                observation=np.zeros(3),
                valid_actions=(0, 1),
                action=0,
                player="BLUE",
            ),
            StepSample(
                observation=np.ones(3),
                valid_actions=(0, 1),
                action=1,
                player="RED",
            ),
            StepSample(
                observation=np.full(3, 2.0),
                valid_actions=(0, 1),
                action=1,
                player="BLUE",
            ),
        ],
        returns=[0.5, -0.25, 1.0],
        advantages=[0.5, -0.25, 1.0],
        old_log_probs=[0.0, 0.0, 0.0],
        old_values=[0.0, 0.0, 0.0],
        old_action_probs=[
            np.asarray([0.5, 0.5], dtype=np.float32),
            np.asarray([0.5, 0.5], dtype=np.float32),
            np.asarray([0.25, 0.75], dtype=np.float32),
        ],
        shaped_rewards=[0.1, 0.0, 0.0],
        old_q_values=[0.0, 0.0, 0.0],
        old_action_q_values=[
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0.2, 0.6], dtype=np.float32),
        ],
    )

    targets = _expected_sarsa_q_targets(
        [trajectory],
        returns=np.asarray(trajectory.returns, dtype=np.float32),
        gamma=0.9,
    )

    assert targets == pytest.approx([0.1 + 0.9 * 0.5, -0.25, 1.0])


def test_q_advantage_baseline_falls_back_without_legal_q_values() -> None:
    trajectory = PPOTrajectory(
        samples=[
            StepSample(
                observation=np.zeros(3),
                valid_actions=(0, 1, 2),
                action=1,
                player="BLUE",
            )
        ],
        returns=[1.0],
        advantages=[1.0],
        old_log_probs=[0.0],
        old_values=[0.2],
        old_action_probs=[np.asarray([1.0 / 3.0] * 3, dtype=np.float32)],
        shaped_rewards=[0.0],
        old_q_values=[0.7],
    )

    baselines = _old_q_policy_baselines(
        [trajectory],
        fallback_values=np.asarray([0.2], dtype=np.float32),
    )

    assert baselines == pytest.approx([0.2])


def test_q_advantage_mix_warms_up_then_ramps() -> None:
    assert _effective_q_advantage_mix(
        iteration_index=0,
        target_mix=0.12,
        warmup_iterations=2,
        ramp_iterations=3,
    ) == 0.0
    assert _effective_q_advantage_mix(
        iteration_index=2,
        target_mix=0.12,
        warmup_iterations=2,
        ramp_iterations=3,
    ) == pytest.approx(0.04)
    assert _effective_q_advantage_mix(
        iteration_index=3,
        target_mix=0.12,
        warmup_iterations=2,
        ramp_iterations=3,
    ) == pytest.approx(0.08)
    assert _effective_q_advantage_mix(
        iteration_index=5,
        target_mix=0.12,
        warmup_iterations=2,
        ramp_iterations=3,
    ) == pytest.approx(0.12)


def test_q_advantage_gate_waits_for_prior_diagnostics() -> None:
    mix, reason = _gated_q_advantage_mix(
        0.12,
        min_sign_agreement=0.55,
        previous_sign_agreement=None,
        min_return_corr=-1.0,
        previous_return_corr=None,
    )

    assert mix == 0.0
    assert reason == "waiting_for_q_sign_agreement"


def test_q_advantage_gate_requires_sign_agreement_and_return_corr() -> None:
    mix, reason = _gated_q_advantage_mix(
        0.12,
        min_sign_agreement=0.55,
        previous_sign_agreement=0.7,
        min_return_corr=0.05,
        previous_return_corr=0.01,
    )
    assert mix == 0.0
    assert reason == "q_return_corr_below_threshold"

    mix, reason = _gated_q_advantage_mix(
        0.12,
        min_sign_agreement=0.55,
        previous_sign_agreement=0.7,
        min_return_corr=0.05,
        previous_return_corr=0.08,
    )
    assert mix == pytest.approx(0.12)
    assert reason == "passed"


def test_ppo_update_stops_early_on_target_kl() -> None:
    pytest.importorskip("torch")
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=35)
    samples = [
        StepSample(
            observation=np.zeros(3),
            valid_actions=(0, 1, 2),
            action=0,
            player="BLUE",
        ),
        StepSample(
            observation=np.ones(3),
            valid_actions=(0, 1, 2),
            action=1,
            player="BLUE",
        ),
    ]
    trajectory = PPOTrajectory(
        samples=samples,
        returns=[1.0, -1.0],
        advantages=[1.0, -1.0],
        old_log_probs=[10.0, 10.0],
        old_values=[0.0, 0.0],
        old_action_probs=[np.asarray([1.0 / 3.0] * 3, dtype=np.float32)] * 2,
        shaped_rewards=[0.0, 0.0],
    )

    update = ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-4,
        clip_ratio=0.1,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=4,
        minibatch_size=1,
        target_kl=0.001,
    )

    assert update["early_stop"] == 1.0
    assert update["minibatches"] == 1.0


def test_dagger_episode_labels_policy_visited_states() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(vps_to_win=3)
    policy = create_ppo_policy(
        config=config,
        seed=32,
        hidden_size=16,
        architecture="candidate",
    )
    samples, returns = collect_dagger_episode(
        policy,
        HeuristicPolicy(),
        {name: RandomPolicy() for name in ("RED", "ORANGE", "WHITE")},
        seed=33,
        config=config,
        max_decisions=40,
        rng=np.random.default_rng(34),
        training_seats={"BLUE"},
        gamma=0.99,
    )

    assert len(samples) == len(returns)
    assert samples
    assert all(sample.action in sample.valid_actions for sample in samples)


def test_checkpoint_opponents_can_join_league() -> None:
    pytest.importorskip("torch")
    checkpoint_policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=23)
    rng = np.random.default_rng(0)

    direct = _make_opponent(
        "checkpoint",
        rng,
        checkpoint_opponents=(checkpoint_policy,),
    )
    league_members = {
        _make_opponent(
            "league",
            rng,
            checkpoint_opponents=(checkpoint_policy,),
        ).name
        for _ in range(20)
    }

    assert direct is checkpoint_policy
    assert "torch_ppo" in league_members
    assert {"random", "heuristic", "catanatron_value"} & league_members


def test_policy_snapshot_is_frozen_copy() -> None:
    torch = pytest.importorskip("torch")
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=25)
    snapshot = policy.clone_frozen()

    with torch.no_grad():
        next(policy.actor.parameters()).add_(1.0)

    live_param = next(policy.actor.parameters()).detach().clone()
    frozen_param = next(snapshot.actor.parameters()).detach().clone()
    assert not torch.allclose(live_param, frozen_param)


def test_dynamic_league_snapshots_keep_static_opponents() -> None:
    pytest.importorskip("torch")
    static_policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=26)
    live_policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=27)
    opponents = [static_policy]

    for _ in range(3):
        _append_policy_snapshot(
            opponents,
            live_policy,
            static_count=1,
            max_snapshots=2,
        )

    assert opponents[0] is static_policy
    assert len(opponents) == 3
    assert all(opponent is not live_policy for opponent in opponents[1:])


def test_evaluation_progress_callback_emits_interval_json(capsys) -> None:
    callback = _progress_callback(2)
    assert callback is not None

    callback({"game": 1, "games": 5, "wins": 0, "win_rate": 0.0})
    callback({"game": 2, "games": 5, "wins": 1, "win_rate": 0.5})
    callback({"game": 5, "games": 5, "wins": 2, "win_rate": 0.4})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert '"game": 1' not in captured.err
    assert '"game": 2' in captured.err
    assert '"game": 5' in captured.err


def test_strong_mixed_opponents_exclude_random() -> None:
    pytest.importorskip("torch")
    checkpoint_policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=24)
    rng = np.random.default_rng(1)

    names = {
        _make_opponent(
            "strong_mixed",
            rng,
            checkpoint_opponents=(checkpoint_policy,),
        ).name
        for _ in range(40)
    }

    assert "random" not in names
    assert "heuristic" in names
    assert "catanatron_value" in names
    assert "torch_ppo" in names


def test_adaptive_league_starts_without_random_or_checkpoints() -> None:
    rng = np.random.default_rng(41)

    names = {_make_opponent("adaptive_league", rng).name for _ in range(50)}

    assert "random" not in names
    assert names <= {"heuristic", "jsettlers_lite", "catanatron_value"}
    assert "heuristic" in names
    assert "catanatron_value" in names


def test_adaptive_league_adds_checkpoint_snapshots_without_random() -> None:
    pytest.importorskip("torch")
    checkpoint_policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=42)
    rng = np.random.default_rng(42)

    names = {
        _make_opponent(
            "adaptive_league",
            rng,
            checkpoint_opponents=(checkpoint_policy,),
        ).name
        for _ in range(80)
    }

    assert "random" not in names
    assert "heuristic" in names
    assert "catanatron_value" in names
    assert "torch_ppo" in names


def test_checkpoint_sampling_biases_newer_snapshots() -> None:
    pytest.importorskip("torch")
    policies = tuple(
        TorchPPOPolicy(3, 5, hidden_size=8, seed=50 + index)
        for index in range(3)
    )
    rng = np.random.default_rng(43)
    counts = {id(policy): 0 for policy in policies}

    for _ in range(700):
        selected = _sample_checkpoint_opponent(policies, rng)
        counts[id(selected)] += 1

    assert counts[id(policies[-1])] > counts[id(policies[0])]


def test_weighted_index_falls_back_when_weights_are_zero() -> None:
    rng = np.random.default_rng(44)

    selected = {_weighted_index(rng, (0.0, 0.0, 0.0)) for _ in range(20)}

    assert selected <= {0, 1, 2}
    assert selected


def test_training_value_opponent_strength_is_configurable() -> None:
    rng = np.random.default_rng(2)

    opponent = _make_opponent(
        "value",
        rng,
        value_candidate_limit=72,
        value_opponent_penalty=0.125,
    )

    assert isinstance(opponent, CatanatronValuePolicy)
    assert opponent.candidate_limit == 72
    assert opponent.opponent_penalty == 0.125


def test_warmup_replay_keeps_latest_samples() -> None:
    first = [
        StepSample(np.zeros(1), (0,), 0, "BLUE"),
        StepSample(np.zeros(1), (1,), 1, "RED"),
    ]
    second = [
        StepSample(np.zeros(1), (2,), 2, "ORANGE"),
        StepSample(np.zeros(1), (3,), 3, "WHITE"),
    ]

    samples, returns = _append_replay(
        first,
        [0.1, 0.2],
        second,
        [0.3, 0.4],
        max_samples=3,
    )

    assert [sample.action for sample in samples] == [1, 2, 3]
    assert returns == [0.2, 0.3, 0.4]


def test_dagger_only_iterations_still_use_anchor_update() -> None:
    assert _uses_anchor_update(
        anchor_games_per_iteration=0,
        dagger_games_per_iteration=4,
        anchor_epochs=2,
    )
    assert _uses_anchor_update(
        anchor_games_per_iteration=1,
        dagger_games_per_iteration=0,
        anchor_epochs=1,
    )
    assert not _uses_anchor_update(
        anchor_games_per_iteration=0,
        dagger_games_per_iteration=4,
        anchor_epochs=0,
    )


def test_best_warmup_checkpoint_prefers_value_gate() -> None:
    summaries = [
        {
            "episode": 8,
            "interim_checkpoint": {
                "path": "early.pt",
                "eval_vs_random": {"games": 12, "win_rate": 0.75, "wins": 9},
                "eval_vs_heuristic": {"games": 12, "win_rate": 0.10, "wins": 1},
                "eval_vs_value": {"games": 8, "win_rate": 0.25, "wins": 2},
            },
        },
        {
            "episode": 16,
            "interim_checkpoint": {
                "path": "late.pt",
                "eval_vs_random": {"games": 12, "win_rate": 0.90, "wins": 11},
                "eval_vs_heuristic": {"games": 12, "win_rate": 0.50, "wins": 6},
                "eval_vs_value": {"games": 8, "win_rate": 0.125, "wins": 1},
            },
        },
    ]

    selected = _best_warmup_checkpoint(summaries)

    assert selected is not None
    assert selected["episode"] == 8
    assert selected["path"] == "early.pt"


def test_eval_win_lower_bound_penalizes_tiny_samples() -> None:
    small = _eval_win_lower_bound({"games": 8, "wins": 2, "win_rate": 0.25})
    large = _eval_win_lower_bound({"games": 64, "wins": 16, "win_rate": 0.25})

    assert small is not None
    assert large is not None
    assert large > small
    assert large < 0.25


def test_best_warmup_checkpoint_uses_teacher_agreement_tiebreak() -> None:
    summaries = [
        {
            "episode": 8,
            "interim_checkpoint": {
                "path": "low_agreement.pt",
                "eval_vs_value": {"games": 16, "win_rate": 0.25, "wins": 4},
                "teacher_agreement": {
                    "samples": 100,
                    "accuracy": 0.35,
                    "mean_teacher_log_prob": -1.4,
                },
            },
        },
        {
            "episode": 16,
            "interim_checkpoint": {
                "path": "high_agreement.pt",
                "eval_vs_value": {"games": 16, "win_rate": 0.25, "wins": 4},
                "teacher_agreement": {
                    "samples": 100,
                    "accuracy": 0.50,
                    "mean_teacher_log_prob": -1.0,
                },
            },
        },
    ]

    selected = _best_warmup_checkpoint(summaries)

    assert selected is not None
    assert selected["episode"] == 16
    assert selected["path"] == "high_agreement.pt"
    assert selected["teacher_agreement"]["accuracy"] == 0.50


def test_best_warmup_checkpoint_can_use_agreement_without_game_eval() -> None:
    summaries = [
        {
            "episode": 8,
            "interim_checkpoint": {
                "path": "weak.pt",
                "teacher_agreement": {
                    "samples": 100,
                    "accuracy": 0.20,
                    "mean_teacher_log_prob": -2.0,
                },
            },
        },
        {
            "episode": 16,
            "interim_checkpoint": {
                "path": "strong.pt",
                "teacher_agreement": {
                    "samples": 100,
                    "accuracy": 0.45,
                    "mean_teacher_log_prob": -1.2,
                },
            },
        },
    ]

    selected = _best_warmup_checkpoint(summaries)

    assert selected is not None
    assert selected["path"] == "strong.pt"


def test_final_warmup_checkpoint_written_when_selection_needs_candidate() -> None:
    assert _should_write_final_warmup_checkpoint(
        [{"episode": 8}],
        select_best_checkpoint=True,
        select_best_warmup_checkpoint=False,
    )
    assert _should_write_final_warmup_checkpoint(
        [{"episode": 8}],
        select_best_checkpoint=False,
        select_best_warmup_checkpoint=True,
    )


def test_final_warmup_checkpoint_not_duplicated_or_written_without_selection() -> None:
    assert not _should_write_final_warmup_checkpoint(
        [{"episode": 8, "interim_checkpoint": {"path": "warmup.pt"}}],
        select_best_checkpoint=True,
        select_best_warmup_checkpoint=False,
    )
    assert not _should_write_final_warmup_checkpoint(
        [{"episode": 8}],
        select_best_checkpoint=False,
        select_best_warmup_checkpoint=False,
    )
    assert not _should_write_final_warmup_checkpoint(
        [],
        select_best_checkpoint=True,
        select_best_warmup_checkpoint=False,
    )


def test_best_iteration_checkpoint_prefers_value_gate() -> None:
    summaries = [
        {
            "iteration": 4,
            "interim_checkpoint": {
                "path": "iter4.pt",
                "eval_vs_value": {"games": 32, "win_rate": 0.25, "wins": 8},
            },
        },
        {
            "iteration": 8,
            "interim_checkpoint": {
                "path": "iter8.pt",
                "eval_vs_value": {"games": 32, "win_rate": 0.125, "wins": 4},
            },
        },
    ]

    selected = _best_iteration_checkpoint(summaries)

    assert selected is not None
    assert selected["stage"] == "iteration"
    assert selected["iteration"] == 4
    assert selected["path"] == "iter4.pt"


def test_best_training_checkpoint_compares_warmup_and_iterations() -> None:
    warmup = [
        {
            "episode": 16,
            "interim_checkpoint": {
                "path": "warmup.pt",
                "eval_vs_value": {"games": 32, "win_rate": 0.125, "wins": 4},
            },
        },
    ]
    iterations = [
        {
            "iteration": 4,
            "interim_checkpoint": {
                "path": "iter.pt",
                "eval_vs_value": {"games": 32, "win_rate": 0.3125, "wins": 10},
            },
        },
    ]

    selected = _best_training_checkpoint(warmup, iterations)

    assert selected is not None
    assert selected["stage"] == "iteration"
    assert selected["path"] == "iter.pt"


def test_best_training_checkpoint_can_require_value_floor() -> None:
    warmup = [
        {
            "episode": 16,
            "interim_checkpoint": {
                "path": "sub_baseline.pt",
                "eval_vs_value": {"games": 32, "win_rate": 0.1875, "wins": 6},
            },
        },
    ]
    iterations = [
        {
            "iteration": 4,
            "interim_checkpoint": {
                "path": "passes_floor.pt",
                "eval_vs_value": {"games": 32, "win_rate": 0.25, "wins": 8},
            },
        },
    ]

    selected = _best_training_checkpoint(
        warmup,
        iterations,
        min_value_win_rate=0.25,
    )

    assert selected is not None
    assert selected["path"] == "passes_floor.pt"


def test_best_training_checkpoint_keeps_strong_initial_checkpoint() -> None:
    initial = {
        "path": "init.pt",
        "eval_vs_value": {"games": 32, "win_rate": 0.25, "wins": 8},
        "eval_vs_heuristic": {"games": 32, "win_rate": 0.25, "wins": 8},
    }
    iterations = [
        {
            "iteration": 2,
            "interim_checkpoint": {
                "path": "degraded.pt",
                "eval_vs_value": {"games": 32, "win_rate": 0.0, "wins": 0},
                "eval_vs_heuristic": {"games": 32, "win_rate": 0.25, "wins": 8},
            },
        },
    ]

    selected = _best_training_checkpoint(
        [],
        iterations,
        initial_checkpoint=initial,
        min_value_win_rate=0.20,
    )

    assert selected is not None
    assert selected["stage"] == "initial"
    assert selected["path"] == "init.pt"


def test_best_training_checkpoint_returns_none_when_value_floor_fails() -> None:
    selected = _best_training_checkpoint(
        [],
        [
            {
                "iteration": 4,
                "interim_checkpoint": {
                    "path": "weak.pt",
                    "eval_vs_value": {"games": 32, "win_rate": 0.1875, "wins": 6},
                },
            },
        ],
        min_value_win_rate=0.25,
    )

    assert selected is None


def test_ladder_command_uses_champion_init_without_warmup(tmp_path) -> None:
    class Args:
        architecture = "flat"
        hidden_size = 32
        warmup_games = 8
        warmup_epochs = 1
        warmup_checkpoint_every = 4
        warmup_checkpoint_eval_games = 2
        warmup_checkpoint_eval_value_games = 2
        iterations = 2
        episodes_per_iteration = 1
        ppo_epochs = 1
        learning_rate = 0.001
        checkpoint_every = 1
        checkpoint_eval_games = 1
        checkpoint_eval_value_games = 1
        eval_games = 1
        eval_value_games = 1
        min_value_win_rate = 0.25
        vps_to_win = 3
        max_decisions = 80
        extra_train_arg = []

    command = build_train_command(
        Args(),
        seed=123,
        checkpoint=tmp_path / "candidate.pt",
        report=tmp_path / "report.json",
        init_checkpoint=tmp_path / "champion.pt",
    )

    assert "--init-checkpoint" in command
    assert "--opponent-checkpoints" in command
    assert str(tmp_path / "champion.pt") in command
    assert "--select-best-min-value-win-rate" in command
    assert command[command.index("--select-best-min-value-win-rate") + 1] == "0.25"
    assert "--warmup-checkpoint-every" in command
    assert command[command.index("--warmup-checkpoint-every") + 1] == "4"
    assert "--warmup-checkpoint-eval-value-games" in command
    assert command[command.index("--warmup-checkpoint-eval-value-games") + 1] == "2"
    assert command[command.index("--gamma") + 1] == "1.0"
    warmup_positions = [index for index, value in enumerate(command) if value == "--warmup-games"]
    assert command[warmup_positions[-1] + 1] == "0"


def test_ladder_promotion_requires_heuristic_gate() -> None:
    promoted, reason = should_promote(
        EvalScore(random_win_rate=1.0, heuristic_win_rate=0.1, value_win_rate=0.5),
        None,
        min_heuristic_win_rate=0.25,
        min_value_win_rate=0.0,
    )

    assert not promoted
    assert "heuristic" in reason


def test_ladder_promotion_compares_existing_champion() -> None:
    candidate = EvalScore(random_win_rate=0.9, heuristic_win_rate=0.4, value_win_rate=0.2)
    champion = EvalScore(random_win_rate=0.8, heuristic_win_rate=0.3, value_win_rate=0.1)

    promoted, reason = should_promote(
        candidate,
        champion,
        min_heuristic_win_rate=0.25,
        min_value_win_rate=0.0,
    )

    assert promoted
    assert "champion" in reason


def test_soft_teacher_targets_are_normalized_to_valid_actions() -> None:
    torch = pytest.importorskip("torch")
    samples = [
        StepSample(
            observation=np.zeros(3),
            valid_actions=(1, 3),
            action=1,
            player="BLUE",
            target_policy={1: 2.0, 2: 100.0, 3: 2.0},
        ),
        StepSample(
            observation=np.zeros(3),
            valid_actions=(0, 4),
            action=4,
            player="RED",
        ),
    ]

    targets = _target_policy_tensor(samples, action_size=5, device=torch.device("cpu"))

    assert targets is not None
    assert torch.allclose(targets[0], torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0]))
    assert torch.allclose(targets[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))


def test_soft_teacher_targets_can_blend_hard_teacher_action() -> None:
    torch = pytest.importorskip("torch")
    samples = [
        StepSample(
            observation=np.zeros(3),
            valid_actions=(1, 3),
            action=1,
            player="BLUE",
            target_policy={1: 2.0, 3: 2.0},
        )
    ]

    targets = _target_policy_tensor(
        samples,
        action_size=5,
        device=torch.device("cpu"),
        hard_target_weight=0.25,
    )

    assert targets is not None
    assert torch.allclose(targets[0], torch.tensor([0.0, 0.625, 0.0, 0.375, 0.0]))


def test_teacher_score_targets_are_normalized_to_valid_scored_actions() -> None:
    torch = pytest.importorskip("torch")
    samples = [
        StepSample(
            observation=np.zeros(3),
            valid_actions=(1, 3, 4),
            action=3,
            player="BLUE",
            target_scores={1: 10.0, 2: 999.0, 3: 20.0, 4: 30.0},
        ),
        StepSample(
            observation=np.zeros(3),
            valid_actions=(0, 4),
            action=4,
            player="RED",
            target_scores={4: 5.0},
        ),
    ]

    targets, mask = _target_score_tensors(
        samples,
        action_size=5,
        device=torch.device("cpu"),
    )

    assert targets is not None
    assert mask is not None
    assert mask.tolist() == [
        [False, True, False, True, True],
        [False, False, False, False, False],
    ]
    assert torch.allclose(targets[0, [1, 3, 4]].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(targets[0, [1, 3, 4]].std(unbiased=False), torch.tensor(1.0))


def test_score_margin_loss_prefers_teacher_ranking() -> None:
    torch = pytest.importorskip("torch")
    targets = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [False, True, False, True, True],
        ],
        dtype=torch.bool,
    )
    correctly_ordered = torch.tensor([[0.0, -2.0, 0.0, 0.0, 2.0]])
    reversed_order = torch.tensor([[0.0, 2.0, 0.0, 0.0, -2.0]])

    good_loss = _score_margin_loss(correctly_ordered, targets, mask)
    bad_loss = _score_margin_loss(reversed_order, targets, mask)

    assert good_loss < bad_loss


def test_action_feature_table_matches_action_space() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(make_env_config(vps_to_win=3))
    try:
        table = build_action_feature_table(env)

        assert table.shape[0] == env.action_space.n
        assert table.shape[1] > 30
        assert np.any(table[0])
    finally:
        env.close()


def test_observation_normalization_preserves_binary_features() -> None:
    normalized = _normalize_observation(
        np.asarray([0.0, 1.0, 2.0, 19.0, 25.0, -25.0, np.nan, np.inf])
    )

    assert np.allclose(
        normalized,
        np.asarray([0.0, 1.0, 0.08, 0.76, 1.0, -1.0, 0.0, 1.0], dtype=np.float32),
    )


def test_candidate_ppo_policy_round_trips(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    policy = create_ppo_policy(
        config=make_env_config(vps_to_win=3),
        seed=22,
        hidden_size=32,
        architecture="candidate",
    )
    obs = torch.zeros((2, policy.observation_size), dtype=torch.float32)
    logits, values = policy.forward(obs)
    q_values = policy.q_values(obs)
    path = tmp_path / "candidate_ppo.pt"
    policy.save(path)
    loaded = policy.load(path)
    loaded_logits, loaded_values = loaded.forward(obs)
    loaded_q_values = loaded.q_values(obs)

    assert logits.shape == (2, policy.action_size)
    assert values.shape == (2,)
    assert q_values.shape == (2, policy.action_size)
    assert policy.action_id_embedding is not None
    assert loaded.architecture == "candidate"
    assert loaded.action_id_embedding is not None
    assert loaded_logits.shape == logits.shape
    assert loaded_values.shape == values.shape
    assert loaded_q_values.shape == q_values.shape


def test_graph_history_candidate_ppo_policy_round_trips(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    policy = create_ppo_policy(
        config=make_env_config(vps_to_win=3),
        seed=24,
        hidden_size=32,
        architecture="graph_history_candidate",
    )
    base_policy = create_ppo_policy(
        config=make_env_config(vps_to_win=3),
        seed=24,
        hidden_size=32,
        architecture="candidate",
    )
    obs = torch.zeros((2, policy.observation_size), dtype=torch.float32)
    logits, values = policy.forward(obs)
    q_values = policy.q_values(obs)
    path = tmp_path / "graph_history_candidate_ppo.pt"
    policy.save(path)
    loaded = policy.load(path)
    loaded_logits, loaded_values = loaded.forward(obs)
    loaded_q_values = loaded.q_values(obs)

    assert policy.observation_size > base_policy.observation_size
    assert logits.shape == (2, policy.action_size)
    assert values.shape == (2,)
    assert q_values.shape == (2, policy.action_size)
    assert loaded.architecture == "graph_history_candidate"
    assert loaded.observation_size == policy.observation_size
    assert loaded.action_id_embedding is not None
    assert loaded_logits.shape == logits.shape
    assert loaded_values.shape == values.shape
    assert loaded_q_values.shape == q_values.shape


def test_candidate_ppo_policy_can_disable_action_id_embedding(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    policy = create_ppo_policy(
        config=make_env_config(vps_to_win=3),
        seed=23,
        hidden_size=32,
        architecture="candidate",
        use_action_id_embedding=False,
    )
    obs = torch.zeros((2, policy.observation_size), dtype=torch.float32)
    logits, values = policy.forward(obs)
    q_values = policy.q_values(obs)
    path = tmp_path / "candidate_ppo_no_id.pt"
    policy.save(path)
    loaded = policy.load(path)
    loaded_logits, loaded_values = loaded.forward(obs)
    loaded_q_values = loaded.q_values(obs)

    assert logits.shape == (2, policy.action_size)
    assert values.shape == (2,)
    assert q_values.shape == (2, policy.action_size)
    assert policy.action_id_embedding is None
    assert loaded.architecture == "candidate"
    assert not loaded.use_action_id_embedding
    assert loaded.action_id_embedding is None
    assert loaded_logits.shape == logits.shape
    assert loaded_values.shape == values.shape
    assert loaded_q_values.shape == q_values.shape
