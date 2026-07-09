from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from catan_zero.rl import (
    CatanatronValuePolicy,
    HeuristicPolicy,
    JSettlersLitePolicy,
    RandomPolicy,
    ValueRolloutSearchPolicy,
    collect_imitation_game,
    evaluate_policy,
    load_reanalysis_jsonl,
)
from catan_zero.rl.self_play import make_env_config, write_report
from catan_zero.rl.torch_ppo import (
    TorchPPOPolicy,
    collect_dagger_episode,
    collect_ppo_episode,
    create_ppo_policy,
    evaluate_teacher_agreement,
    imitation_update,
    make_imitation_optimizer,
    make_ppo_optimizer,
    ppo_update,
    update_ema_policy,
)


class BaselineMixedTeacherPolicy:
    name = "baseline_mixed"

    def __init__(self) -> None:
        self._heuristic = HeuristicPolicy()
        self._jsettlers = JSettlersLitePolicy()

    def select_action(self, env, observation, info, rng, *, training: bool = False) -> int:
        teacher = self._heuristic if float(rng.random()) < 0.5 else self._jsettlers
        return teacher.select_action(
            env,
            observation,
            info,
            rng,
            training=training,
        )

    def target_scores(self, env, info, rng) -> dict[int, float]:
        return _blend_teacher_score_maps(
            self._heuristic.target_scores(env, info, rng),
            self._jsettlers.target_scores(env, info, rng),
        )


class BaselineRolloutMixedTeacherPolicy:
    name = "baseline_rollout_mixed"

    def __init__(
        self,
        *,
        candidate_limit: int = 24,
        presearch_candidate_limit: int | None = 48,
        rollout_decisions: int = 2,
        rollout_samples: int = 1,
        root_value_weight: float = 0.25,
        opponent_penalty: float = 0.05,
        distillation_temperature: float = 0.45,
    ) -> None:
        self._baseline = BaselineMixedTeacherPolicy()
        self._rollout = ValueRolloutSearchPolicy(
            candidate_limit=candidate_limit,
            presearch_candidate_limit=presearch_candidate_limit,
            rollout_decisions=rollout_decisions,
            rollout_samples=rollout_samples,
            root_value_weight=root_value_weight,
            opponent_penalty=opponent_penalty,
            distillation_temperature=distillation_temperature,
        )

    def select_action(self, env, observation, info, rng, *, training: bool = False) -> int:
        teacher = self._rollout if float(rng.random()) < 0.5 else self._baseline
        return teacher.select_action(
            env,
            observation,
            info,
            rng,
            training=training,
        )

    def target_scores(self, env, info, rng) -> dict[int, float]:
        return _blend_teacher_score_maps(
            self._baseline.target_scores(env, info, rng),
            self._rollout.target_scores(env, info, rng),
        )


class TacticalRolloutMixedTeacherPolicy:
    name = "tactical_rollout_mixed"

    def __init__(
        self,
        *,
        candidate_limit: int = 24,
        presearch_candidate_limit: int | None = 48,
        rollout_decisions: int = 2,
        rollout_samples: int = 1,
        root_value_weight: float = 0.25,
        opponent_penalty: float = 0.05,
        distillation_temperature: float = 0.45,
    ) -> None:
        self._baseline = BaselineMixedTeacherPolicy()
        self._rollout = ValueRolloutSearchPolicy(
            candidate_limit=candidate_limit,
            presearch_candidate_limit=presearch_candidate_limit,
            rollout_decisions=rollout_decisions,
            rollout_samples=rollout_samples,
            root_value_weight=root_value_weight,
            opponent_penalty=opponent_penalty,
            distillation_temperature=distillation_temperature,
        )

    def select_action(self, env, observation, info, rng, *, training: bool = False) -> int:
        scores = self.target_scores(env, info, rng)
        if scores:
            return max(scores, key=scores.get)
        return self._baseline.select_action(
            env,
            observation,
            info,
            rng,
            training=training,
        )

    def target_scores(self, env, info, rng) -> dict[int, float]:
        return _tactical_rollout_score_map(
            info,
            baseline_scores=self._baseline.target_scores(env, info, rng),
            rollout_scores=self._rollout.target_scores(env, info, rng),
        )


def _tactical_rollout_score_map(
    info,
    *,
    baseline_scores: dict[int, float],
    rollout_scores: dict[int, float],
    group_weight: float = 1.0,
    baseline_within_weight: float = 0.25,
    rollout_within_weight: float = 0.75,
) -> dict[int, float]:
    action_types = {
        int(action["index"]): str(action["action_type"])
        for action in info.get("structured_legal_actions", ())
    }
    baseline = _standardized_score_map(baseline_scores)
    rollout = _standardized_score_map(rollout_scores)
    actions = sorted(action for action in baseline if action in action_types)
    if not actions:
        return _blend_teacher_score_maps(baseline_scores, rollout_scores)

    groups: dict[str, list[int]] = {}
    for action in actions:
        groups.setdefault(action_types[action], []).append(action)
    group_scores = {
        action_type: max(float(baseline[action]) for action in group_actions)
        for action_type, group_actions in groups.items()
    }
    group_scores = _standardized_score_map(
        {idx: group_scores[action_type] for idx, action_type in enumerate(groups)}
    )
    group_lookup = {
        action_type: float(group_scores[idx])
        for idx, action_type in enumerate(groups)
    }
    rollout_by_group: dict[str, dict[int, float]] = {}
    for action_type, group_actions in groups.items():
        group_rollout = {
            action: float(rollout[action])
            for action in group_actions
            if action in rollout
        }
        rollout_by_group[action_type] = _standardized_score_map(group_rollout)

    scores = {}
    for action in actions:
        action_type = action_types[action]
        scores[action] = float(
            group_weight * group_lookup.get(action_type, 0.0)
            + baseline_within_weight * baseline.get(action, 0.0)
            + rollout_within_weight
            * rollout_by_group.get(action_type, {}).get(action, 0.0)
        )
    return scores


def _blend_teacher_score_maps(*score_maps: dict[int, float]) -> dict[int, float]:
    blended: dict[int, list[float]] = {}
    for scores in score_maps:
        for action, score in _standardized_score_map(scores).items():
            blended.setdefault(int(action), []).append(float(score))
    return {
        action: float(np.mean(values))
        for action, values in blended.items()
        if values
    }


def _standardized_score_map(scores: dict[int, float]) -> dict[int, float]:
    finite = {
        int(action): float(score)
        for action, score in scores.items()
        if np.isfinite(float(score))
    }
    if len(finite) < 2:
        return finite
    values = np.asarray(list(finite.values()), dtype=np.float64)
    std = float(values.std())
    if std <= 1e-9:
        return {action: 0.0 for action in finite}
    mean = float(values.mean())
    return {
        action: float((score - mean) / std)
        for action, score in finite.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a masked PPO Catan policy.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--architecture",
        choices=("flat", "candidate", "graph_history_candidate"),
        default="flat",
        help=(
            "flat global logits, structured legal-action candidate scorer, "
            "or graph/history-observation candidate scorer."
        ),
    )
    parser.add_argument(
        "--disable-action-id-embedding",
        action="store_true",
        help=(
            "For candidate architecture, remove the learned per-action ID "
            "embedding so the policy must score legal actions from structured "
            "action features. Useful as a low-data generalization ablation."
        ),
    )
    parser.add_argument(
        "--teacher",
        choices=(
            "heuristic",
            "jsettlers_lite",
            "baseline_mixed",
            "baseline_rollout_mixed",
            "tactical_rollout_mixed",
            "value",
            "value_rollout",
        ),
        default="value",
    )
    parser.add_argument(
        "--teacher-candidate-limit",
        type=int,
        default=48,
        help="Number of heuristic-pruned legal actions considered by the value teacher.",
    )
    parser.add_argument(
        "--teacher-presearch-candidate-limit",
        type=int,
        default=0,
        help=(
            "If >0 with --teacher value_rollout, one-ply value-rank this wider "
            "candidate pool before rollout-scoring --teacher-candidate-limit actions."
        ),
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=0.7,
        help="Soft distillation temperature for the value teacher target policy.",
    )
    parser.add_argument(
        "--teacher-rollout-decisions",
        type=int,
        default=4,
        help="Rollout depth when --teacher value_rollout is used.",
    )
    parser.add_argument(
        "--teacher-rollout-samples",
        type=int,
        default=1,
        help="Chance-rollout samples per root action when --teacher value_rollout is used.",
    )
    parser.add_argument(
        "--teacher-root-value-weight",
        type=float,
        default=0.0,
        help=(
            "For --teacher value_rollout, blend this much immediate root value "
            "into the rollout teacher score. 0 preserves pure rollout search."
        ),
    )
    parser.add_argument(
        "--teacher-opponent-penalty",
        type=float,
        default=0.05,
        help="Opponent value penalty used by the value teacher.",
    )
    parser.add_argument("--warmup-games", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-value-coef", type=float, default=0.5)
    parser.add_argument(
        "--anchor-value-coef",
        type=float,
        default=0.0,
        help="Value loss coefficient for teacher-anchor imitation updates.",
    )
    parser.add_argument(
        "--reanalysis-input",
        nargs="*",
        default=(),
        help=(
            "One or more JSONL files produced by tools/generate_reanalysis.py. "
            "These search targets are replayed before online self-play."
        ),
    )
    parser.add_argument(
        "--reanalysis-max-samples",
        type=int,
        default=0,
        help="Maximum total reanalysis samples to load; 0 loads all samples.",
    )
    parser.add_argument(
        "--reanalysis-epochs",
        type=int,
        default=0,
        help="Supervised epochs over loaded reanalysis targets before warmup/PPO.",
    )
    parser.add_argument(
        "--reanalysis-value-coef",
        type=float,
        default=0.5,
        help="Value loss coefficient for reanalysis returns.",
    )
    parser.add_argument(
        "--reanalysis-score-coef",
        type=float,
        default=0.0,
        help="Search-score margin loss coefficient for reanalysis targets.",
    )
    parser.add_argument(
        "--reanalysis-checkpoint",
        default="",
        help=(
            "Optional checkpoint path to save immediately after reanalysis "
            "pretraining and before online PPO."
        ),
    )
    parser.add_argument(
        "--imitation-score-coef",
        type=float,
        default=0.0,
        help=(
            "If >0, add normalized teacher action-score margin regression "
            "during warmup and teacher-anchor imitation."
        ),
    )
    parser.add_argument(
        "--imitation-hard-target-weight",
        type=float,
        default=0.0,
        help=(
            "Blend this much one-hot teacher action into soft teacher targets "
            "during warmup and anchor imitation."
        ),
    )
    parser.add_argument(
        "--warmup-replay-size",
        type=int,
        default=0,
        help=(
            "If >0, train warmup on a rolling buffer of this many latest "
            "teacher samples instead of only the newest teacher game."
        ),
    )
    parser.add_argument(
        "--warmup-checkpoint-every",
        type=int,
        default=0,
        help="Save an interim checkpoint every N supervised warmup games; 0 disables.",
    )
    parser.add_argument(
        "--warmup-checkpoint-eval-games",
        type=int,
        default=0,
        help="If >0, run quick random/heuristic eval for each warmup checkpoint.",
    )
    parser.add_argument(
        "--warmup-checkpoint-eval-value-games",
        type=int,
        default=0,
        help="If >0, also run quick value-baseline eval for warmup checkpoints.",
    )
    parser.add_argument(
        "--warmup-checkpoint-agreement-games",
        type=int,
        default=0,
        help=(
            "If >0, collect fresh held-out teacher games at each warmup "
            "checkpoint and report teacher-action agreement. This gives "
            "checkpoint selection a low-variance imitation-quality gate before "
            "expensive game evaluation."
        ),
    )
    parser.add_argument(
        "--select-best-warmup-checkpoint",
        action="store_true",
        help=(
            "After warmup, reload the best interim warmup checkpoint before final "
            "eval/save. Prefers eval_vs_value win rate, then heuristic, then random."
        ),
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument(
        "--opponents",
        choices=(
            "self",
            "heuristic",
            "jsettlers_lite",
            "value",
            "value_rollout",
            "random",
            "mixed",
            "strong_mixed",
            "strict_mixed",
            "anti_regression_mixed",
            "jsettlers_value_repair_mixed",
            "strict_gate_repair_mixed",
            "pfsp_mixed",
            "search_mixed",
            "adaptive_league",
            "checkpoint",
            "league",
        ),
        default="self",
        help=(
            "Opponent source during PPO. self trains every seat with the shared "
            "policy; checkpoint/league require --opponent-checkpoints. "
            "adaptive_league is a harder weighted league that can start without "
            "checkpoints and will use snapshots once available."
        ),
    )
    parser.add_argument(
        "--opponent-checkpoints",
        nargs="*",
        default=(),
        help="Frozen TorchPPOPolicy checkpoints for checkpoint/league opponents.",
    )
    parser.add_argument(
        "--league-snapshot-every",
        type=int,
        default=0,
        help=(
            "If >0, freeze the current policy every N PPO iterations and add "
            "it to checkpoint/league opponent sampling."
        ),
    )
    parser.add_argument(
        "--league-max-snapshots",
        type=int,
        default=4,
        help="Maximum number of dynamic historical snapshots to keep in memory.",
    )
    parser.add_argument(
        "--training-value-candidate-limit",
        type=int,
        default=48,
        help=(
            "Candidate limit for value opponents sampled during PPO training. "
            "Keep this aligned with held-out value-bot gates so training does "
            "not optimize against an easier value opponent."
        ),
    )
    parser.add_argument(
        "--training-value-opponent-penalty",
        type=float,
        default=0.05,
        help="Opponent penalty for value opponents sampled during PPO training.",
    )
    parser.add_argument(
        "--learner-seats",
        choices=("all", "one"),
        default="all",
        help="Train all seats in each episode or one learner seat against fixed opponents.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--ppo-top-advantage-fraction",
        type=float,
        default=1.0,
        help=(
            "If <1, keep only the top fraction of positive-advantage PPO "
            "samples before the update. This is an opt-in top-advantage "
            "filter for sparse win/loss self-play."
        ),
    )
    parser.add_argument(
        "--ppo-min-advantage-samples",
        type=int,
        default=1,
        help="Minimum samples to retain when --ppo-top-advantage-fraction is active.",
    )
    parser.add_argument(
        "--q-value-coef",
        type=float,
        default=0.0,
        help=(
            "If >0, train a chosen-action Q critic toward the same returns. "
            "This is the lightweight path toward VRPO-style Q-boosting."
        ),
    )
    parser.add_argument(
        "--q-advantage-mix",
        type=float,
        default=0.0,
        help=(
            "Blend this much old Q(s,a)-V(s) into PPO advantages. Keep small "
            "until the Q critic has proven stable."
        ),
    )
    parser.add_argument(
        "--q-expected-sarsa-mix",
        type=float,
        default=0.0,
        help=(
            "Blend Expected-SARSA-style bootstrapped targets into the Q-head "
            "loss. This keeps the default PPO path unchanged while enabling a "
            "VRPO-like Q critic ablation."
        ),
    )
    parser.add_argument(
        "--q-advantage-warmup-iterations",
        type=int,
        default=0,
        help=(
            "Keep Q-advantage mixing at zero for this many PPO iterations while "
            "the Q critic learns from returns."
        ),
    )
    parser.add_argument(
        "--q-advantage-ramp-iterations",
        type=int,
        default=1,
        help=(
            "Linearly ramp Q-advantage mixing to --q-advantage-mix over this "
            "many iterations after the warmup."
        ),
    )
    parser.add_argument(
        "--q-advantage-min-sign-agreement",
        type=float,
        default=0.0,
        help=(
            "If >0, keep Q-advantage actor mixing disabled until the previous "
            "PPO iteration's Q-vs-GAE sign agreement is at least this value."
        ),
    )
    parser.add_argument(
        "--q-advantage-min-return-corr",
        type=float,
        default=-1.0,
        help=(
            "If >-1, keep Q-advantage actor mixing disabled until the previous "
            "PPO iteration's chosen-Q/return correlation is at least this value."
        ),
    )
    parser.add_argument(
        "--value-clip-range",
        type=float,
        default=0.0,
        help="If >0, use PPO clipped value loss with this absolute value range.",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--old-policy-kl-coef",
        type=float,
        default=0.0,
        help=(
            "If >0, add a PPO minibatch KL penalty toward the rollout "
            "policy's legal-action distribution."
        ),
    )
    parser.add_argument(
        "--ema-policy-kl-coef",
        type=float,
        default=0.0,
        help=(
            "If >0, add a KL penalty toward an exponential moving average of "
            "the policy. This is an EMAg-style self-play stabilizer."
        ),
    )
    parser.add_argument(
        "--ema-policy-decay",
        type=float,
        default=0.95,
        help="EMA decay used with --ema-policy-kl-coef.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.0,
        help="If >0, stop each PPO update early once approximate KL exceeds this value.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help=(
            "Discount for complete-game returns. Catan rewards are terminal "
            "and learner-action spacing varies by seat, so the default keeps "
            "win/loss credit undiscounted."
        ),
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--value-shaping-coef",
        type=float,
        default=0.0,
        help=(
            "If >0, add clipped Catanatron value-score deltas to PPO rewards "
            "during rollout collection."
        ),
    )
    parser.add_argument(
        "--value-shaping-scale",
        type=float,
        default=100.0,
        help="Score-delta scale used before clipping value-shaped PPO rewards.",
    )
    parser.add_argument(
        "--value-shaping-opponent-penalty",
        type=float,
        default=0.05,
        help="Opponent penalty used for the Catanatron value-shaped reward.",
    )
    parser.add_argument("--anchor-games-per-iteration", type=int, default=1)
    parser.add_argument(
        "--anchor-sample-weight",
        type=float,
        default=1.0,
        help="Sample weight applied to fresh teacher self-play anchor states.",
    )
    parser.add_argument(
        "--dagger-games-per-iteration",
        type=int,
        default=0,
        help=(
            "If >0, add teacher-labeled states visited by the current policy "
            "to the imitation anchor each PPO iteration."
        ),
    )
    parser.add_argument(
        "--dagger-sample-weight",
        type=float,
        default=1.0,
        help=(
            "Sample weight applied to DAgger states visited by the learner. "
            "Use >1 to target anti-regression drift states harder."
        ),
    )
    parser.add_argument(
        "--dagger-low-return-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiply DAgger sample weights for learner-visited states whose "
            "terminal return is <= --dagger-low-return-threshold. This focuses "
            "teacher correction on bad matchup trajectories instead of giving "
            "winning and losing states equal imitation pressure."
        ),
    )
    parser.add_argument(
        "--dagger-low-return-threshold",
        type=float,
        default=0.0,
        help="Return threshold used by --dagger-low-return-multiplier.",
    )
    parser.add_argument(
        "--anchor-replay-size",
        type=int,
        default=0,
        help=(
            "If >0, train the PPO teacher anchor on a rolling buffer of this "
            "many latest teacher samples instead of only the newest anchor games."
        ),
    )
    parser.add_argument("--anchor-epochs", type=int, default=1)
    parser.add_argument("--anchor-learning-rate-multiplier", type=float, default=0.25)
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--eval-value-games", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save an interim checkpoint every N PPO iterations; 0 disables.",
    )
    parser.add_argument(
        "--checkpoint-eval-games",
        type=int,
        default=0,
        help="If >0, run quick random/heuristic eval for each interim checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-eval-value-games",
        type=int,
        default=0,
        help="If >0, also run quick value-baseline eval for PPO checkpoints.",
    )
    parser.add_argument(
        "--select-best-checkpoint",
        action="store_true",
        help=(
            "Before final eval/save, reload the best evaluated interim checkpoint "
            "from warmup or PPO iteration gates. This prevents a run from "
            "publishing a regressed final policy when an earlier checkpoint won "
            "the held-out value-bot gate."
        ),
    )
    parser.add_argument(
        "--select-best-min-value-win-rate",
        type=float,
        default=0.0,
        help=(
            "When --select-best-checkpoint is enabled, ignore interim "
            "checkpoints with eval_vs_value below this raw win-rate floor. "
            "Use 0.25 for serious four-seat promotion gates to avoid building "
            "on noisy sub-baseline checkpoints."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device for policy inference and updates. Use cuda or "
            "cuda:N on GPU hosts; cpu preserves the historical default."
        ),
    )
    parser.add_argument("--checkpoint", default="runs/self_play/ppo_policy.pt")
    parser.add_argument(
        "--init-checkpoint",
        help="Optional TorchPPOPolicy checkpoint to continue training from.",
    )
    parser.add_argument("--report", default="runs/self_play/ppo_report.json")
    args = parser.parse_args()
    if args.opponents != "self" and args.learner_seats == "all":
        raise SystemExit(
            "--opponents only applies when --learner-seats one. "
            "Use --learner-seats one for fixed/league opponent training, or "
            "use --opponents self when training every seat with the shared policy."
        )

    rng = np.random.default_rng(args.seed)
    config = make_env_config(
        vps_to_win=args.vps_to_win,
        use_graph_history_features=args.architecture == "graph_history_candidate",
    )
    if args.init_checkpoint:
        policy = TorchPPOPolicy.load(args.init_checkpoint, device=args.device)
        args.architecture = policy.architecture
        args.hidden_size = policy.hidden_size
        args.disable_action_id_embedding = not policy.use_action_id_embedding
        config = make_env_config(
            vps_to_win=args.vps_to_win,
            use_graph_history_features=(
                args.architecture == "graph_history_candidate"
            ),
        )
        _validate_policy_shape(policy, config=config, seed=args.seed)
    else:
        policy = create_ppo_policy(
            config=config,
            seed=args.seed,
            hidden_size=args.hidden_size,
            architecture=args.architecture,
            use_action_id_embedding=not args.disable_action_id_embedding,
            device=args.device,
        )
    checkpoint_opponents = [
        TorchPPOPolicy.load(path, device=args.device) for path in args.opponent_checkpoints
    ]
    static_checkpoint_count = len(checkpoint_opponents)
    if args.opponents in ("checkpoint", "league") and not checkpoint_opponents:
        raise SystemExit(f"--opponents {args.opponents} requires --opponent-checkpoints")
    teacher = _make_teacher(args)

    initial_checkpoint = None
    if args.init_checkpoint and args.select_best_checkpoint:
        initial_checkpoint = _write_interim_checkpoint(
            policy,
            label="init",
            checkpoint=args.checkpoint,
            config=config,
            max_decisions=args.max_decisions,
            eval_games=(
                args.checkpoint_eval_games
                if args.checkpoint_eval_games > 0
                else args.warmup_checkpoint_eval_games
            ),
            value_eval_games=(
                args.checkpoint_eval_value_games
                if args.checkpoint_eval_value_games > 0
                else args.warmup_checkpoint_eval_value_games
            ),
            teacher=None,
            agreement_games=0,
            seed=args.seed + 70_000,
        )
        print(
            json.dumps({"initial_checkpoint": initial_checkpoint}, sort_keys=True),
            flush=True,
        )

    reanalysis_summary = {"samples": 0.0, "loss": 0.0, "policy_loss": 0.0}
    if args.reanalysis_input and args.reanalysis_epochs > 0:
        reanalysis_samples, reanalysis_returns = load_reanalysis_jsonl(
            args.reanalysis_input,
            max_samples=args.reanalysis_max_samples,
        )
        reanalysis_optimizer = make_imitation_optimizer(
            policy,
            learning_rate=args.learning_rate,
            train_critic=args.reanalysis_value_coef > 0.0,
        )
        reanalysis_summary = imitation_update(
            policy,
            reanalysis_samples,
            learning_rate=args.learning_rate,
            epochs=args.reanalysis_epochs,
            minibatch_size=args.minibatch_size,
            optimizer=reanalysis_optimizer,
            returns=reanalysis_returns,
            value_coef=args.reanalysis_value_coef,
            hard_target_weight=args.imitation_hard_target_weight,
            score_coef=args.reanalysis_score_coef,
        )
        reanalysis_summary["input_files"] = list(args.reanalysis_input)
        if args.reanalysis_checkpoint:
            policy.save(args.reanalysis_checkpoint)
            reanalysis_summary["checkpoint"] = args.reanalysis_checkpoint
        print(json.dumps({"reanalysis": reanalysis_summary}, sort_keys=True), flush=True)

    warmup_summaries = []
    warmup_replay_samples = []
    warmup_replay_returns = []
    imitation_optimizer = make_imitation_optimizer(
        policy,
        learning_rate=args.learning_rate,
        train_critic=args.warmup_value_coef > 0.0,
    )
    for idx in range(args.warmup_games):
        episode = collect_imitation_game(
            teacher,
            seed=int(rng.integers(2**31)),
            config=config,
            max_decisions=args.max_decisions,
            rng=rng,
        )
        samples, sample_returns = _flatten_samples_and_returns(
            episode.samples_by_player,
            episode.result.rewards,
            gamma=args.gamma,
        )
        if args.warmup_replay_size > 0:
            warmup_replay_samples, warmup_replay_returns = _append_replay(
                warmup_replay_samples,
                warmup_replay_returns,
                samples,
                sample_returns,
                max_samples=args.warmup_replay_size,
            )
            train_samples = warmup_replay_samples
            train_returns = warmup_replay_returns
        else:
            train_samples = samples
            train_returns = sample_returns
        update = imitation_update(
            policy,
            train_samples,
            learning_rate=args.learning_rate,
            epochs=args.warmup_epochs,
            minibatch_size=args.minibatch_size,
            optimizer=imitation_optimizer,
            returns=train_returns,
            value_coef=args.warmup_value_coef,
            hard_target_weight=args.imitation_hard_target_weight,
            score_coef=args.imitation_score_coef,
        )
        warmup_summaries.append(
            {
                "episode": idx + 1,
                "teacher": teacher.name,
                "winner": episode.result.winner,
                "decisions": episode.result.decisions,
                "samples": update["samples"],
                "new_samples": len(samples),
                "replay_samples": len(train_samples),
                "loss": update["loss"],
                "policy_loss": update["policy_loss"],
                "value_loss": update["value_loss"],
                "score_loss": update["score_loss"],
            }
        )
        if (
            args.warmup_checkpoint_every > 0
            and (idx + 1) % args.warmup_checkpoint_every == 0
        ):
            warmup_summaries[-1]["interim_checkpoint"] = _write_interim_checkpoint(
                policy,
                label=f"warmup{idx + 1:04d}",
                checkpoint=args.checkpoint,
                config=config,
                max_decisions=args.max_decisions,
                eval_games=args.warmup_checkpoint_eval_games,
                value_eval_games=args.warmup_checkpoint_eval_value_games,
                teacher=teacher,
                agreement_games=args.warmup_checkpoint_agreement_games,
                seed=args.seed + 50_000 + idx,
            )
        print(json.dumps({"warmup": warmup_summaries[-1]}, sort_keys=True), flush=True)

    if _should_write_final_warmup_checkpoint(
        warmup_summaries,
        select_best_checkpoint=args.select_best_checkpoint,
        select_best_warmup_checkpoint=args.select_best_warmup_checkpoint,
    ):
        warmup_summaries[-1]["interim_checkpoint"] = _write_interim_checkpoint(
            policy,
            label=f"warmup{args.warmup_games:04d}",
            checkpoint=args.checkpoint,
            config=config,
            max_decisions=args.max_decisions,
            eval_games=(
                args.warmup_checkpoint_eval_games
                if args.warmup_checkpoint_eval_games > 0
                else args.checkpoint_eval_games
            ),
            value_eval_games=(
                args.warmup_checkpoint_eval_value_games
                if args.warmup_checkpoint_eval_value_games > 0
                else args.checkpoint_eval_value_games
            ),
            teacher=teacher,
            agreement_games=args.warmup_checkpoint_agreement_games,
            seed=args.seed + 60_000 + args.warmup_games,
        )
        print(
            json.dumps({"warmup": warmup_summaries[-1]}, sort_keys=True),
            flush=True,
        )

    player_names = ("BLUE", "RED", "ORANGE", "WHITE")[: config.players]
    optimizer = make_ppo_optimizer(policy, learning_rate=args.learning_rate)
    anchor_optimizer = make_imitation_optimizer(
        policy,
        learning_rate=args.learning_rate * args.anchor_learning_rate_multiplier,
        train_critic=args.anchor_value_coef > 0.0,
    )
    anchor_replay_samples = []
    anchor_replay_returns = []
    iteration_summaries = []
    ema_policy = policy.clone_frozen() if args.ema_policy_kl_coef > 0.0 else None
    previous_q_advantage_sign_agreement: float | None = None
    previous_q_chosen_return_corr: float | None = None
    training_start_time = time.perf_counter()
    for iteration in range(args.iterations):
        iteration_start_time = time.perf_counter()
        trajectories = []
        collect_seconds = 0.0
        for episode_idx in range(args.episodes_per_iteration):
            if args.learner_seats == "all" or args.opponents == "self":
                training_seats = set(player_names)
                opponents = {}
            else:
                training_seat = player_names[
                    (iteration * args.episodes_per_iteration + episode_idx)
                    % len(player_names)
                ]
                training_seats = {training_seat}
                opponents = {
                    name: _make_opponent(
                        args.opponents,
                        rng,
                        checkpoint_opponents=tuple(checkpoint_opponents),
                        value_candidate_limit=args.training_value_candidate_limit,
                        value_opponent_penalty=args.training_value_opponent_penalty,
                    )
                    for name in player_names
                    if name != training_seat
                }
            collect_start_time = time.perf_counter()
            trajectory = collect_ppo_episode(
                policy,
                opponents,
                seed=int(rng.integers(2**31)),
                config=config,
                max_decisions=args.max_decisions,
                rng=rng,
                training_seats=training_seats,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                value_shaping_coef=args.value_shaping_coef,
                value_shaping_scale=args.value_shaping_scale,
                value_shaping_opponent_penalty=args.value_shaping_opponent_penalty,
            )
            collect_seconds += time.perf_counter() - collect_start_time
            trajectories.append(trajectory)
        requested_q_advantage_mix = _effective_q_advantage_mix(
            iteration_index=iteration,
            target_mix=args.q_advantage_mix,
            warmup_iterations=args.q_advantage_warmup_iterations,
            ramp_iterations=args.q_advantage_ramp_iterations,
        )
        effective_q_advantage_mix, q_advantage_gate_reason = _gated_q_advantage_mix(
            requested_q_advantage_mix,
            min_sign_agreement=args.q_advantage_min_sign_agreement,
            previous_sign_agreement=previous_q_advantage_sign_agreement,
            min_return_corr=args.q_advantage_min_return_corr,
            previous_return_corr=previous_q_chosen_return_corr,
        )
        ppo_update_start_time = time.perf_counter()
        update = ppo_update(
            policy,
            trajectories,
            learning_rate=args.learning_rate,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            optimizer=optimizer,
            value_clip_range=args.value_clip_range,
            q_value_coef=args.q_value_coef,
            q_advantage_mix=effective_q_advantage_mix,
            q_expected_sarsa_mix=args.q_expected_sarsa_mix,
            q_expected_sarsa_gamma=args.gamma,
            kl_coef=args.old_policy_kl_coef,
            ema_policy=ema_policy,
            ema_policy_kl_coef=args.ema_policy_kl_coef,
            target_kl=args.target_kl,
            top_advantage_fraction=args.ppo_top_advantage_fraction,
            min_advantage_samples=args.ppo_min_advantage_samples,
        )
        ppo_update_seconds = time.perf_counter() - ppo_update_start_time
        if ema_policy is not None:
            update_ema_policy(
                ema_policy,
                policy,
                decay=args.ema_policy_decay,
            )
        anchor_summary = {"samples": 0.0, "loss": 0.0, "accuracy": 0.0}
        anchor_collect_seconds = 0.0
        anchor_update_seconds = 0.0
        if _uses_anchor_update(
            anchor_games_per_iteration=args.anchor_games_per_iteration,
            dagger_games_per_iteration=args.dagger_games_per_iteration,
            anchor_epochs=args.anchor_epochs,
        ):
            anchor_samples = []
            anchor_returns = []
            for _ in range(args.anchor_games_per_iteration):
                anchor_collect_start_time = time.perf_counter()
                anchor_episode = collect_imitation_game(
                    teacher,
                    seed=int(rng.integers(2**31)),
                    config=config,
                    max_decisions=args.max_decisions,
                    rng=rng,
                )
                anchor_collect_seconds += time.perf_counter() - anchor_collect_start_time
                samples, sample_returns = _flatten_samples_and_returns(
                    anchor_episode.samples_by_player,
                    anchor_episode.result.rewards,
                    gamma=args.gamma,
                )
                _set_sample_weights(samples, args.anchor_sample_weight)
                anchor_samples.extend(samples)
                anchor_returns.extend(sample_returns)
            for dagger_idx in range(args.dagger_games_per_iteration):
                if args.learner_seats == "all" or args.opponents == "self":
                    dagger_training_seats = set(player_names)
                    dagger_opponents = {}
                else:
                    dagger_training_seat = player_names[
                        (
                            iteration * max(args.dagger_games_per_iteration, 1)
                            + dagger_idx
                        )
                        % len(player_names)
                    ]
                    dagger_training_seats = {dagger_training_seat}
                    dagger_opponents = {
                        name: _make_opponent(
                            args.opponents,
                            rng,
                            checkpoint_opponents=tuple(checkpoint_opponents),
                            value_candidate_limit=args.training_value_candidate_limit,
                            value_opponent_penalty=args.training_value_opponent_penalty,
                        )
                        for name in player_names
                        if name != dagger_training_seat
                    }
                dagger_collect_start_time = time.perf_counter()
                dagger_samples, dagger_returns = collect_dagger_episode(
                    policy,
                    teacher,
                    dagger_opponents,
                    seed=int(rng.integers(2**31)),
                    config=config,
                    max_decisions=args.max_decisions,
                    rng=rng,
                    training_seats=dagger_training_seats,
                    gamma=args.gamma,
                )
                anchor_collect_seconds += time.perf_counter() - dagger_collect_start_time
                _set_return_weighted_sample_weights(
                    dagger_samples,
                    dagger_returns,
                    base_weight=args.dagger_sample_weight,
                    low_return_multiplier=args.dagger_low_return_multiplier,
                    low_return_threshold=args.dagger_low_return_threshold,
                )
                anchor_samples.extend(dagger_samples)
                anchor_returns.extend(dagger_returns)
            if args.anchor_replay_size > 0:
                anchor_replay_samples, anchor_replay_returns = _append_replay(
                    anchor_replay_samples,
                    anchor_replay_returns,
                    anchor_samples,
                    anchor_returns,
                    max_samples=args.anchor_replay_size,
                )
                train_anchor_samples = anchor_replay_samples
                train_anchor_returns = anchor_replay_returns
            else:
                train_anchor_samples = anchor_samples
                train_anchor_returns = anchor_returns
            agreement_before = evaluate_teacher_agreement(policy, train_anchor_samples)
            anchor_update_start_time = time.perf_counter()
            anchor_update = imitation_update(
                policy,
                train_anchor_samples,
                learning_rate=args.learning_rate * args.anchor_learning_rate_multiplier,
                epochs=args.anchor_epochs,
                minibatch_size=args.minibatch_size,
                optimizer=anchor_optimizer,
                returns=train_anchor_returns,
                value_coef=args.anchor_value_coef,
                hard_target_weight=args.imitation_hard_target_weight,
                score_coef=args.imitation_score_coef,
            )
            anchor_update_seconds = time.perf_counter() - anchor_update_start_time
            agreement_after = evaluate_teacher_agreement(policy, train_anchor_samples)
            anchor_summary = {
                "samples": anchor_update["samples"],
                "new_samples": len(anchor_samples),
                "dagger_games": args.dagger_games_per_iteration,
                "dagger_low_return_multiplier": args.dagger_low_return_multiplier,
                "dagger_low_return_threshold": args.dagger_low_return_threshold,
                "replay_samples": len(train_anchor_samples),
                "loss": anchor_update["loss"],
                "policy_loss": anchor_update["policy_loss"],
                "value_loss": anchor_update["value_loss"],
                "score_loss": anchor_update["score_loss"],
                    "mean_sample_weight": anchor_update.get("mean_sample_weight", 1.0),
                "accuracy_before": agreement_before["accuracy"],
                "accuracy_after": agreement_after["accuracy"],
                "mean_teacher_log_prob_before": agreement_before["mean_teacher_log_prob"],
                "mean_teacher_log_prob_after": agreement_after["mean_teacher_log_prob"],
            }
        iteration_summaries.append(
            {
                "iteration": iteration + 1,
                "samples": update["samples"],
                "samples_before_filter": update.get("samples_before_filter", update["samples"]),
                "advantage_filter_kept_fraction": update.get(
                    "advantage_filter_kept_fraction",
                    1.0,
                ),
                "advantage_filter_threshold": update.get(
                    "advantage_filter_threshold",
                    0.0,
                ),
                "policy_loss": update["policy_loss"],
                "value_loss": update["value_loss"],
                "q_value_loss": update["q_value_loss"],
                "q_advantage_mix": effective_q_advantage_mix,
                "q_advantage_requested_mix": requested_q_advantage_mix,
                "q_advantage_gate_reason": q_advantage_gate_reason,
                "q_advantage_min_sign_agreement": args.q_advantage_min_sign_agreement,
                "q_advantage_min_return_corr": args.q_advantage_min_return_corr,
                "q_expected_sarsa_mix": args.q_expected_sarsa_mix,
                "q_chosen_return_corr": update["q_chosen_return_corr"],
                "q_legal_std": update["q_legal_std"],
                "q_legal_spread_entropy": update["q_legal_spread_entropy"],
                "q_advantage_sign_agreement": update["q_advantage_sign_agreement"],
                "entropy": update["entropy"],
                "approx_kl": update["approx_kl"],
                "old_policy_kl": update["old_policy_kl"],
                    "ema_policy_kl": update.get("ema_policy_kl", 0.0),
                "ema_policy_kl_coef": args.ema_policy_kl_coef,
                "ema_policy_decay": args.ema_policy_decay,
                "clip_fraction": update["clip_fraction"],
                "ppo_minibatches": update["minibatches"],
                "ppo_early_stop": update["early_stop"],
                "mean_shaped_reward": update["mean_shaped_reward"],
                "anchor": anchor_summary,
                "league_static_checkpoints": static_checkpoint_count,
                "league_dynamic_snapshots": max(
                    len(checkpoint_opponents) - static_checkpoint_count,
                    0,
                ),
            }
        )
        previous_q_advantage_sign_agreement = float(update["q_advantage_sign_agreement"])
        previous_q_chosen_return_corr = float(update["q_chosen_return_corr"])
        if (
            args.league_snapshot_every > 0
            and args.league_max_snapshots > 0
            and (iteration + 1) % args.league_snapshot_every == 0
        ):
            _append_policy_snapshot(
                checkpoint_opponents,
                policy,
                static_count=static_checkpoint_count,
                max_snapshots=args.league_max_snapshots,
            )
            iteration_summaries[-1]["league_snapshot_added"] = True
            iteration_summaries[-1]["league_dynamic_snapshots"] = max(
                len(checkpoint_opponents) - static_checkpoint_count,
                0,
            )
        if args.checkpoint_every > 0 and (iteration + 1) % args.checkpoint_every == 0:
            checkpoint_start_time = time.perf_counter()
            interim = _write_interim_checkpoint(
                policy,
                label=f"iter{iteration + 1:04d}",
                checkpoint=args.checkpoint,
                config=config,
                max_decisions=args.max_decisions,
                eval_games=args.checkpoint_eval_games,
                value_eval_games=args.checkpoint_eval_value_games,
                teacher=None,
                agreement_games=0,
                seed=args.seed + 40_000 + iteration,
            )
            iteration_summaries[-1]["interim_checkpoint"] = interim
            checkpoint_seconds = time.perf_counter() - checkpoint_start_time
        else:
            checkpoint_seconds = 0.0
        iteration_summaries[-1]["timing"] = _iteration_timing_summary(
            iteration_seconds=time.perf_counter() - iteration_start_time,
            collect_seconds=collect_seconds,
            ppo_update_seconds=ppo_update_seconds,
            anchor_collect_seconds=anchor_collect_seconds,
            anchor_update_seconds=anchor_update_seconds,
            checkpoint_seconds=checkpoint_seconds,
            ppo_samples=float(update["samples"]),
            anchor_samples=float(anchor_summary.get("samples", 0.0)),
        )
        print(json.dumps({"ppo": iteration_summaries[-1]}, sort_keys=True), flush=True)

    selected_warmup_checkpoint = None
    selected_training_checkpoint = None
    if args.select_best_checkpoint:
        selected_training_checkpoint = _best_training_checkpoint(
            warmup_summaries,
            iteration_summaries,
            initial_checkpoint=initial_checkpoint,
            min_value_win_rate=args.select_best_min_value_win_rate,
        )
        if selected_training_checkpoint is not None:
            policy = TorchPPOPolicy.load(
                selected_training_checkpoint["path"],
                device=args.device,
            )
    elif args.select_best_warmup_checkpoint:
        selected_warmup_checkpoint = _best_warmup_checkpoint(warmup_summaries)
        if selected_warmup_checkpoint is not None:
            policy = TorchPPOPolicy.load(
                selected_warmup_checkpoint["path"],
                device=args.device,
            )

    checkpoint = Path(args.checkpoint)
    policy.save(checkpoint)
    random_eval = evaluate_policy(
        policy,
        RandomPolicy(),
        games=args.eval_games,
        seed=args.seed + 10_000,
        config=config,
        max_decisions=args.max_decisions,
    )
    heuristic_eval = evaluate_policy(
        policy,
        HeuristicPolicy(),
        games=args.eval_games,
        seed=args.seed + 20_000,
        config=config,
        max_decisions=args.max_decisions,
    )
    value_eval = None
    if args.eval_value_games > 0:
        value_eval = evaluate_policy(
            policy,
            CatanatronValuePolicy(candidate_limit=48),
            games=args.eval_value_games,
            seed=args.seed + 30_000,
            config=config,
            max_decisions=args.max_decisions,
        )
    report = {
        "checkpoint": str(checkpoint),
        "config": vars(args),
        "reanalysis": reanalysis_summary,
        "warmup": warmup_summaries,
        "initial_checkpoint": initial_checkpoint,
        "selected_warmup_checkpoint": selected_warmup_checkpoint,
        "selected_training_checkpoint": selected_training_checkpoint,
        "iterations": iteration_summaries,
        "training_wall_seconds": time.perf_counter() - training_start_time,
        "eval_vs_random": random_eval,
        "eval_vs_heuristic": heuristic_eval,
        "eval_vs_value": value_eval,
    }
    write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def _flatten_samples_and_returns(samples_by_player, rewards, *, gamma: float):
    samples = []
    returns = []
    for player, player_samples in samples_by_player.items():
        n = len(player_samples)
        for idx, sample in enumerate(player_samples):
            samples.append(sample)
            returns.append(float(rewards[player]) * (gamma ** (n - idx - 1)))
    return samples, returns


def _set_sample_weights(samples, weight: float) -> None:
    sample_weight = max(0.0, float(weight))
    for sample in samples:
        sample.sample_weight = sample_weight


def _set_return_weighted_sample_weights(
    samples,
    returns,
    *,
    base_weight: float,
    low_return_multiplier: float,
    low_return_threshold: float,
) -> None:
    base = max(0.0, float(base_weight))
    multiplier = max(1.0, float(low_return_multiplier))
    threshold = float(low_return_threshold)
    if len(samples) != len(returns):
        _set_sample_weights(samples, base)
        return
    for sample, sample_return in zip(samples, returns):
        weight = base
        if float(sample_return) <= threshold:
            weight *= multiplier
        sample.sample_weight = weight


def _iteration_timing_summary(
    *,
    iteration_seconds: float,
    collect_seconds: float,
    ppo_update_seconds: float,
    anchor_collect_seconds: float,
    anchor_update_seconds: float,
    checkpoint_seconds: float,
    ppo_samples: float,
    anchor_samples: float,
) -> dict[str, float]:
    total = max(float(iteration_seconds), 1e-9)
    measured = (
        float(collect_seconds)
        + float(ppo_update_seconds)
        + float(anchor_collect_seconds)
        + float(anchor_update_seconds)
        + float(checkpoint_seconds)
    )
    other_seconds = max(total - measured, 0.0)
    return {
        "iteration_seconds": float(iteration_seconds),
        "collect_seconds": float(collect_seconds),
        "ppo_update_seconds": float(ppo_update_seconds),
        "anchor_collect_seconds": float(anchor_collect_seconds),
        "anchor_update_seconds": float(anchor_update_seconds),
        "checkpoint_seconds": float(checkpoint_seconds),
        "other_seconds": float(other_seconds),
        "collect_fraction": float(collect_seconds) / total,
        "ppo_update_fraction": float(ppo_update_seconds) / total,
        "anchor_collect_fraction": float(anchor_collect_seconds) / total,
        "anchor_update_fraction": float(anchor_update_seconds) / total,
        "checkpoint_fraction": float(checkpoint_seconds) / total,
        "ppo_samples_per_second": float(ppo_samples) / max(
            float(collect_seconds) + float(ppo_update_seconds),
            1e-9,
        ),
        "anchor_samples_per_second": float(anchor_samples) / max(
            float(anchor_collect_seconds) + float(anchor_update_seconds),
            1e-9,
        ),
    }


def _uses_anchor_update(
    *,
    anchor_games_per_iteration: int,
    dagger_games_per_iteration: int,
    anchor_epochs: int,
) -> bool:
    return (
        anchor_epochs > 0
        and (
            anchor_games_per_iteration > 0
            or dagger_games_per_iteration > 0
        )
    )


def _make_teacher(args):
    if args.teacher == "value":
        return CatanatronValuePolicy(
            candidate_limit=args.teacher_candidate_limit,
            opponent_penalty=args.teacher_opponent_penalty,
            distillation_temperature=args.teacher_temperature,
        )
    if args.teacher == "value_rollout":
        return ValueRolloutSearchPolicy(
            candidate_limit=args.teacher_candidate_limit,
            presearch_candidate_limit=(
                args.teacher_presearch_candidate_limit
                if args.teacher_presearch_candidate_limit > 0
                else None
            ),
            rollout_decisions=args.teacher_rollout_decisions,
            rollout_samples=args.teacher_rollout_samples,
            root_value_weight=args.teacher_root_value_weight,
            opponent_penalty=args.teacher_opponent_penalty,
            distillation_temperature=args.teacher_temperature,
        )
    if args.teacher == "heuristic":
        return HeuristicPolicy()
    if args.teacher == "jsettlers_lite":
        return JSettlersLitePolicy()
    if args.teacher == "baseline_mixed":
        return BaselineMixedTeacherPolicy()
    if args.teacher == "baseline_rollout_mixed":
        return BaselineRolloutMixedTeacherPolicy(
            candidate_limit=args.teacher_candidate_limit,
            presearch_candidate_limit=(
                args.teacher_presearch_candidate_limit
                if args.teacher_presearch_candidate_limit > 0
                else None
            ),
            rollout_decisions=args.teacher_rollout_decisions,
            rollout_samples=args.teacher_rollout_samples,
            root_value_weight=args.teacher_root_value_weight,
            opponent_penalty=args.teacher_opponent_penalty,
            distillation_temperature=args.teacher_temperature,
        )
    if args.teacher == "tactical_rollout_mixed":
        return TacticalRolloutMixedTeacherPolicy(
            candidate_limit=args.teacher_candidate_limit,
            presearch_candidate_limit=(
                args.teacher_presearch_candidate_limit
                if args.teacher_presearch_candidate_limit > 0
                else None
            ),
            rollout_decisions=args.teacher_rollout_decisions,
            rollout_samples=args.teacher_rollout_samples,
            root_value_weight=args.teacher_root_value_weight,
            opponent_penalty=args.teacher_opponent_penalty,
            distillation_temperature=args.teacher_temperature,
        )
    raise ValueError(f"unknown teacher {args.teacher!r}")


def _effective_q_advantage_mix(
    *,
    iteration_index: int,
    target_mix: float,
    warmup_iterations: int,
    ramp_iterations: int,
) -> float:
    target_mix = max(float(target_mix), 0.0)
    if target_mix <= 0.0:
        return 0.0
    warmup_iterations = max(int(warmup_iterations), 0)
    if int(iteration_index) < warmup_iterations:
        return 0.0
    ramp_iterations = max(int(ramp_iterations), 1)
    ramp_step = int(iteration_index) - warmup_iterations + 1
    return target_mix * min(float(ramp_step) / float(ramp_iterations), 1.0)


def _gated_q_advantage_mix(
    requested_mix: float,
    *,
    min_sign_agreement: float,
    previous_sign_agreement: float | None,
    min_return_corr: float,
    previous_return_corr: float | None,
) -> tuple[float, str]:
    requested_mix = max(float(requested_mix), 0.0)
    if requested_mix <= 0.0:
        return 0.0, "off"

    min_sign_agreement = float(min_sign_agreement)
    if min_sign_agreement > 0.0:
        if previous_sign_agreement is None:
            return 0.0, "waiting_for_q_sign_agreement"
        if float(previous_sign_agreement) < min_sign_agreement:
            return 0.0, "q_sign_agreement_below_threshold"

    min_return_corr = float(min_return_corr)
    if min_return_corr > -1.0:
        if previous_return_corr is None:
            return 0.0, "waiting_for_q_return_corr"
        if float(previous_return_corr) < min_return_corr:
            return 0.0, "q_return_corr_below_threshold"

    return requested_mix, "passed"


def _append_replay(
    replay_samples,
    replay_returns,
    new_samples,
    new_returns,
    *,
    max_samples: int,
):
    if max_samples <= 0:
        return list(new_samples), list(new_returns)
    samples = list(replay_samples) + list(new_samples)
    returns = list(replay_returns) + list(new_returns)
    if len(samples) > max_samples:
        samples = samples[-max_samples:]
        returns = returns[-max_samples:]
    return samples, returns


def _append_policy_snapshot(
    checkpoint_opponents: list[TorchPPOPolicy],
    policy: TorchPPOPolicy,
    *,
    static_count: int,
    max_snapshots: int,
) -> None:
    checkpoint_opponents.append(policy.clone_frozen())
    dynamic_count = len(checkpoint_opponents) - static_count
    if dynamic_count > max_snapshots:
        del checkpoint_opponents[static_count]


def _write_interim_checkpoint(
    policy,
    *,
    label: str,
    checkpoint: str,
    config,
    max_decisions: int,
    eval_games: int,
    value_eval_games: int = 0,
    teacher=None,
    agreement_games: int = 0,
    seed: int,
) -> dict:
    checkpoint_path = Path(checkpoint)
    interim_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.{label}{checkpoint_path.suffix}"
    )
    policy.save(interim_path)
    payload = {"path": str(interim_path)}
    if eval_games > 0:
        payload["eval_vs_random"] = evaluate_policy(
            policy,
            RandomPolicy(),
            games=eval_games,
            seed=seed,
            config=config,
            max_decisions=max_decisions,
        )
        payload["eval_vs_heuristic"] = evaluate_policy(
            policy,
            HeuristicPolicy(),
            games=eval_games,
            seed=seed + 1_000,
            config=config,
            max_decisions=max_decisions,
        )
    if value_eval_games > 0:
        payload["eval_vs_value"] = evaluate_policy(
            policy,
            CatanatronValuePolicy(candidate_limit=48),
            games=value_eval_games,
            seed=seed + 2_000,
            config=config,
            max_decisions=max_decisions,
        )
    if teacher is not None and agreement_games > 0:
        agreement_samples = _collect_teacher_samples(
            teacher,
            games=agreement_games,
            seed=seed + 3_000,
            config=config,
            max_decisions=max_decisions,
        )
        payload["teacher_agreement"] = evaluate_teacher_agreement(
            policy,
            agreement_samples,
        )
    return payload


def _collect_teacher_samples(
    teacher,
    *,
    games: int,
    seed: int,
    config,
    max_decisions: int,
) -> list:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(games):
        episode = collect_imitation_game(
            teacher,
            seed=int(rng.integers(2**31)),
            config=config,
            max_decisions=max_decisions,
            rng=rng,
        )
        for player_samples in episode.samples_by_player.values():
            samples.extend(player_samples)
    return samples


def _best_warmup_checkpoint(warmup_summaries: list[dict]) -> dict | None:
    return _best_interim_checkpoint(
        warmup_summaries,
        step_key="episode",
        stage="warmup",
    )


def _should_write_final_warmup_checkpoint(
    warmup_summaries: list[dict],
    *,
    select_best_checkpoint: bool,
    select_best_warmup_checkpoint: bool,
) -> bool:
    if not (select_best_checkpoint or select_best_warmup_checkpoint):
        return False
    if not warmup_summaries:
        return False
    return not isinstance(warmup_summaries[-1].get("interim_checkpoint"), dict)


def _best_iteration_checkpoint(iteration_summaries: list[dict]) -> dict | None:
    return _best_interim_checkpoint(
        iteration_summaries,
        step_key="iteration",
        stage="iteration",
    )


def _best_training_checkpoint(
    warmup_summaries: list[dict],
    iteration_summaries: list[dict],
    *,
    initial_checkpoint: dict | None = None,
    min_value_win_rate: float = 0.0,
) -> dict | None:
    candidates = []
    if isinstance(initial_checkpoint, dict) and "path" in initial_checkpoint:
        value_rate = _eval_win_rate(initial_checkpoint.get("eval_vs_value"))
        if value_rate is None or value_rate >= min_value_win_rate:
            score = _warmup_checkpoint_score(initial_checkpoint)
            if score is not None:
                candidates.append(
                    {
                        "stage": "initial",
                        "path": initial_checkpoint["path"],
                        "score": list((*score, 0)),
                        "eval_vs_value": initial_checkpoint.get("eval_vs_value"),
                        "eval_vs_heuristic": initial_checkpoint.get(
                            "eval_vs_heuristic"
                        ),
                        "eval_vs_random": initial_checkpoint.get("eval_vs_random"),
                        "teacher_agreement": initial_checkpoint.get(
                            "teacher_agreement"
                        ),
                    }
                )
    for stage, step_key, summaries in (
        ("warmup", "episode", warmup_summaries),
        ("iteration", "iteration", iteration_summaries),
    ):
        selected = _best_interim_checkpoint(
            summaries,
            step_key=step_key,
            stage=stage,
            min_value_win_rate=min_value_win_rate,
        )
        if selected is not None:
            candidates.append(selected)
    if not candidates:
        return None
    return max(candidates, key=lambda selected: tuple(selected["score"]))


def _best_interim_checkpoint(
    summaries: list[dict],
    *,
    step_key: str,
    stage: str,
    min_value_win_rate: float = 0.0,
) -> dict | None:
    best_summary = None
    best_score = None
    for summary in summaries:
        checkpoint = summary.get("interim_checkpoint")
        if not isinstance(checkpoint, dict) or "path" not in checkpoint:
            continue
        value_rate = _eval_win_rate(checkpoint.get("eval_vs_value"))
        if value_rate is not None and value_rate < min_value_win_rate:
            continue
        score = _warmup_checkpoint_score(checkpoint)
        if score is None:
            continue
        score = (*score, -int(summary.get(step_key, 0)))
        if best_score is None or score > best_score:
            best_score = score
            best_summary = summary
    if best_summary is None or best_score is None:
        return None
    checkpoint = best_summary["interim_checkpoint"]
    return {
        "stage": stage,
        step_key: int(best_summary.get(step_key, 0)),
        "path": checkpoint["path"],
        "score": list(best_score),
        "eval_vs_value": checkpoint.get("eval_vs_value"),
        "eval_vs_heuristic": checkpoint.get("eval_vs_heuristic"),
        "eval_vs_random": checkpoint.get("eval_vs_random"),
        "teacher_agreement": checkpoint.get("teacher_agreement"),
    }


def _warmup_checkpoint_score(checkpoint: dict) -> tuple[float, ...] | None:
    random_rate = _eval_win_rate(checkpoint.get("eval_vs_random"))
    heuristic_rate = _eval_win_rate(checkpoint.get("eval_vs_heuristic"))
    value_rate = _eval_win_rate(checkpoint.get("eval_vs_value"))
    random_bound = _eval_win_lower_bound(checkpoint.get("eval_vs_random"))
    heuristic_bound = _eval_win_lower_bound(checkpoint.get("eval_vs_heuristic"))
    value_bound = _eval_win_lower_bound(checkpoint.get("eval_vs_value"))
    agreement = checkpoint.get("teacher_agreement")
    agreement_accuracy = _teacher_agreement_accuracy(agreement)
    agreement_log_prob = _teacher_agreement_log_prob(agreement)
    agreement_score = agreement_accuracy or 0.0
    log_prob_score = agreement_log_prob if agreement_log_prob is not None else -1e9
    if value_rate is not None:
        return (
            2.0,
            value_bound if value_bound is not None else value_rate,
            agreement_score,
            log_prob_score,
            heuristic_bound if heuristic_bound is not None else (heuristic_rate or 0.0),
        )
    if heuristic_rate is not None:
        return (
            1.0,
            heuristic_bound if heuristic_bound is not None else heuristic_rate,
            agreement_score,
            log_prob_score,
            random_bound if random_bound is not None else (random_rate or 0.0),
        )
    if agreement_accuracy is not None:
        return (
            0.5,
            agreement_accuracy,
            log_prob_score,
        )
    if random_rate is not None:
        return (
            0.0,
            random_bound if random_bound is not None else random_rate,
            0.0,
        )
    return None


def _teacher_agreement_accuracy(report) -> float | None:
    if not isinstance(report, dict) or "accuracy" not in report:
        return None
    return float(report["accuracy"])


def _teacher_agreement_log_prob(report) -> float | None:
    if not isinstance(report, dict) or "mean_teacher_log_prob" not in report:
        return None
    return float(report["mean_teacher_log_prob"])


def _eval_win_rate(report) -> float | None:
    if not isinstance(report, dict) or "win_rate" not in report:
        return None
    return float(report["win_rate"])


def _eval_win_lower_bound(report, *, z: float = 1.0) -> float | None:
    if not isinstance(report, dict):
        return None
    wins = report.get("wins")
    games = report.get("games")
    if wins is None or games is None:
        return None
    games = int(games)
    if games <= 0:
        return None
    wins = max(min(int(wins), games), 0)
    p_hat = wins / games
    z2 = z * z
    denom = 1.0 + z2 / games
    center = p_hat + z2 / (2.0 * games)
    margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * games)) / games) ** 0.5
    return max((center - margin) / denom, 0.0)


def _validate_policy_shape(policy: TorchPPOPolicy, *, config, seed: int) -> None:
    probe = create_ppo_policy(
        config=config,
        seed=seed,
        hidden_size=policy.hidden_size,
        architecture=policy.architecture,
    )
    if policy.observation_size != probe.observation_size:
        raise SystemExit(
            "--init-checkpoint observation size does not match the requested env"
        )
    if policy.action_size != probe.action_size:
        raise SystemExit(
            "--init-checkpoint action size does not match the requested env"
        )


def _make_opponent(
    kind: str,
    rng: np.random.Generator,
    *,
    checkpoint_opponents: tuple[TorchPPOPolicy, ...] = (),
    value_candidate_limit: int = 48,
    value_opponent_penalty: float = 0.05,
):
    if kind == "random":
        return RandomPolicy()
    if kind == "heuristic":
        return HeuristicPolicy()
    if kind == "jsettlers_lite":
        return JSettlersLitePolicy()
    if kind == "value":
        return CatanatronValuePolicy(
            candidate_limit=value_candidate_limit,
            opponent_penalty=value_opponent_penalty,
        )
    if kind == "value_rollout":
        return ValueRolloutSearchPolicy(
            candidate_limit=min(value_candidate_limit, 24),
            presearch_candidate_limit=max(value_candidate_limit, 24),
            rollout_decisions=1,
            rollout_samples=1,
            root_value_weight=0.5,
            opponent_penalty=value_opponent_penalty,
        )
    if kind == "checkpoint":
        if not checkpoint_opponents:
            raise ValueError("checkpoint opponents require at least one checkpoint")
        return _sample_checkpoint_opponent(checkpoint_opponents, rng)
    if kind == "league":
        choices = 3 + len(checkpoint_opponents)
        choice = int(rng.integers(choices))
        if choice == 0:
            return RandomPolicy()
        if choice == 1:
            return HeuristicPolicy()
        if choice == 2:
            return CatanatronValuePolicy(
                candidate_limit=value_candidate_limit,
                opponent_penalty=value_opponent_penalty,
            )
        return checkpoint_opponents[choice - 3]
    if kind == "adaptive_league":
        if checkpoint_opponents:
            choice = _weighted_index(rng, (0.15, 0.20, 0.30, 0.35))
            if choice == 0:
                return HeuristicPolicy()
            if choice == 1:
                return JSettlersLitePolicy()
            if choice == 2:
                return CatanatronValuePolicy(
                    candidate_limit=value_candidate_limit,
                    opponent_penalty=value_opponent_penalty,
                )
            return _sample_checkpoint_opponent(checkpoint_opponents, rng)
        choice = _weighted_index(rng, (0.20, 0.30, 0.50))
        if choice == 0:
            return HeuristicPolicy()
        if choice == 1:
            return JSettlersLitePolicy()
        return CatanatronValuePolicy(
            candidate_limit=value_candidate_limit,
            opponent_penalty=value_opponent_penalty,
        )
    if kind == "strong_mixed":
        choices = 2 + len(checkpoint_opponents)
        choice = int(rng.integers(choices))
        if choice == 0:
            return HeuristicPolicy()
        if choice == 1:
            return CatanatronValuePolicy(
                candidate_limit=value_candidate_limit,
                opponent_penalty=value_opponent_penalty,
            )
        return checkpoint_opponents[choice - 2]
    if kind == "strict_mixed":
        choices = 4 + len(checkpoint_opponents)
        choice = int(rng.integers(choices))
        if choice == 0:
            return HeuristicPolicy()
        if choice == 1:
            return JSettlersLitePolicy()
        if choice == 2:
            return CatanatronValuePolicy(
                candidate_limit=value_candidate_limit,
                opponent_penalty=value_opponent_penalty,
            )
        if choice == 3:
            return ValueRolloutSearchPolicy(
                candidate_limit=min(value_candidate_limit, 24),
                presearch_candidate_limit=max(value_candidate_limit, 24),
                rollout_decisions=1,
                rollout_samples=1,
                root_value_weight=0.5,
                opponent_penalty=value_opponent_penalty,
            )
        return checkpoint_opponents[choice - 4]
    if kind == "anti_regression_mixed":
        if checkpoint_opponents:
            choice = _weighted_index(rng, (0.35, 0.35, 0.20, 0.10))
            if choice == 0:
                return HeuristicPolicy()
            if choice == 1:
                return JSettlersLitePolicy()
            if choice == 2:
                return CatanatronValuePolicy(
                    candidate_limit=value_candidate_limit,
                    opponent_penalty=value_opponent_penalty,
                )
            return _sample_checkpoint_opponent(checkpoint_opponents, rng)
        choice = _weighted_index(rng, (0.40, 0.40, 0.20))
        if choice == 0:
            return HeuristicPolicy()
        if choice == 1:
            return JSettlersLitePolicy()
        return CatanatronValuePolicy(
            candidate_limit=value_candidate_limit,
            opponent_penalty=value_opponent_penalty,
        )
    if kind == "jsettlers_value_repair_mixed":
        if checkpoint_opponents:
            # Current population gates mostly fail by regressing against
            # JSettlers-lite, with value-rollout weakness as the next hard leg.
            choice = _weighted_index(rng, (0.45, 0.25, 0.15, 0.05, 0.10))
            if choice == 0:
                return JSettlersLitePolicy()
            if choice == 1:
                return ValueRolloutSearchPolicy(
                    candidate_limit=min(value_candidate_limit, 24),
                    presearch_candidate_limit=max(value_candidate_limit, 24),
                    rollout_decisions=1,
                    rollout_samples=1,
                    root_value_weight=0.5,
                    opponent_penalty=value_opponent_penalty,
                )
            if choice == 2:
                return CatanatronValuePolicy(
                    candidate_limit=value_candidate_limit,
                    opponent_penalty=value_opponent_penalty,
                )
            if choice == 3:
                return HeuristicPolicy()
            return _sample_checkpoint_opponent(checkpoint_opponents, rng)
        choice = _weighted_index(rng, (0.50, 0.28, 0.16, 0.06))
        if choice == 0:
            return JSettlersLitePolicy()
        if choice == 1:
            return ValueRolloutSearchPolicy(
                candidate_limit=min(value_candidate_limit, 24),
                presearch_candidate_limit=max(value_candidate_limit, 24),
                rollout_decisions=1,
                rollout_samples=1,
                root_value_weight=0.5,
                opponent_penalty=value_opponent_penalty,
            )
        if choice == 2:
            return CatanatronValuePolicy(
                candidate_limit=value_candidate_limit,
                opponent_penalty=value_opponent_penalty,
            )
        return HeuristicPolicy()
    if kind == "strict_gate_repair_mixed":
        if checkpoint_opponents:
            # Match the strict promotion gate: every training batch must keep
            # JSettlers-lite, rollout search, and heuristic robustness alive.
            choice = _weighted_index(rng, (0.34, 0.30, 0.22, 0.08, 0.06))
            if choice == 0:
                return JSettlersLitePolicy()
            if choice == 1:
                return ValueRolloutSearchPolicy(
                    candidate_limit=min(value_candidate_limit, 24),
                    presearch_candidate_limit=max(value_candidate_limit, 24),
                    rollout_decisions=1,
                    rollout_samples=1,
                    root_value_weight=0.5,
                    opponent_penalty=value_opponent_penalty,
                )
            if choice == 2:
                return HeuristicPolicy()
            if choice == 3:
                return CatanatronValuePolicy(
                    candidate_limit=value_candidate_limit,
                    opponent_penalty=value_opponent_penalty,
                )
            return _sample_checkpoint_opponent(checkpoint_opponents, rng)
        choice = _weighted_index(rng, (0.36, 0.32, 0.24, 0.08))
        if choice == 0:
            return JSettlersLitePolicy()
        if choice == 1:
            return ValueRolloutSearchPolicy(
                candidate_limit=min(value_candidate_limit, 24),
                presearch_candidate_limit=max(value_candidate_limit, 24),
                rollout_decisions=1,
                rollout_samples=1,
                root_value_weight=0.5,
                opponent_penalty=value_opponent_penalty,
            )
        if choice == 2:
            return HeuristicPolicy()
        return CatanatronValuePolicy(
            candidate_limit=value_candidate_limit,
            opponent_penalty=value_opponent_penalty,
        )
    if kind == "pfsp_mixed":
        if checkpoint_opponents:
            # Ordered by current strict-gate weakness: value, jsettlers,
            # rollout-search, then heuristic, with a smaller frozen-policy lane.
            choice = _weighted_index(rng, (0.08, 0.30, 0.34, 0.18, 0.10))
            if choice == 0:
                return HeuristicPolicy()
            if choice == 1:
                return JSettlersLitePolicy()
            if choice == 2:
                return CatanatronValuePolicy(
                    candidate_limit=value_candidate_limit,
                    opponent_penalty=value_opponent_penalty,
                )
            if choice == 3:
                return ValueRolloutSearchPolicy(
                    candidate_limit=min(value_candidate_limit, 24),
                    presearch_candidate_limit=max(value_candidate_limit, 24),
                    rollout_decisions=1,
                    rollout_samples=1,
                    root_value_weight=0.5,
                    opponent_penalty=value_opponent_penalty,
                )
            return _sample_checkpoint_opponent(checkpoint_opponents, rng)
        choice = _weighted_index(rng, (0.09, 0.32, 0.36, 0.23))
        if choice == 0:
            return HeuristicPolicy()
        if choice == 1:
            return JSettlersLitePolicy()
        if choice == 2:
            return CatanatronValuePolicy(
                candidate_limit=value_candidate_limit,
                opponent_penalty=value_opponent_penalty,
            )
        return ValueRolloutSearchPolicy(
            candidate_limit=min(value_candidate_limit, 24),
            presearch_candidate_limit=max(value_candidate_limit, 24),
            rollout_decisions=1,
            rollout_samples=1,
            root_value_weight=0.5,
            opponent_penalty=value_opponent_penalty,
        )
    if kind == "search_mixed":
        choices = 3 + len(checkpoint_opponents)
        choice = int(rng.integers(choices))
        if choice == 0:
            return HeuristicPolicy()
        if choice == 1:
            return CatanatronValuePolicy(
                candidate_limit=value_candidate_limit,
                opponent_penalty=value_opponent_penalty,
            )
        if choice == 2:
            return ValueRolloutSearchPolicy(
                candidate_limit=min(value_candidate_limit, 24),
                presearch_candidate_limit=max(value_candidate_limit, 24),
                rollout_decisions=1,
                rollout_samples=1,
                root_value_weight=0.5,
                opponent_penalty=value_opponent_penalty,
            )
        return checkpoint_opponents[choice - 3]
    if kind == "mixed":
        choice = int(rng.integers(3))
        if choice == 0:
            return RandomPolicy()
        if choice == 1:
            return HeuristicPolicy()
        return CatanatronValuePolicy(
            candidate_limit=value_candidate_limit,
            opponent_penalty=value_opponent_penalty,
        )
    if kind == "self":
        raise ValueError("self opponents are handled by training every seat")
    raise ValueError(kind)


def _sample_checkpoint_opponent(
    checkpoint_opponents: tuple[TorchPPOPolicy, ...],
    rng: np.random.Generator,
):
    if not checkpoint_opponents:
        raise ValueError("checkpoint opponents require at least one checkpoint")
    weights = np.arange(1, len(checkpoint_opponents) + 1, dtype=np.float64)
    weights *= weights
    return checkpoint_opponents[_weighted_index(rng, weights)]


def _weighted_index(
    rng: np.random.Generator,
    weights,
) -> int:
    weights_array = np.asarray(tuple(weights), dtype=np.float64)
    if weights_array.ndim != 1 or len(weights_array) == 0:
        raise ValueError("weights must be a non-empty 1D sequence")
    weights_array = np.maximum(weights_array, 0.0)
    total = float(weights_array.sum())
    if total <= 0.0:
        return int(rng.integers(len(weights_array)))
    threshold = float(rng.random()) * total
    cumulative = 0.0
    for index, weight in enumerate(weights_array):
        cumulative += float(weight)
        if threshold < cumulative:
            return index
    return len(weights_array) - 1


if __name__ == "__main__":
    main()
