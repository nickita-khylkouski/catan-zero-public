from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catan_zero.rl import ColonistMultiAgentEnv, make_env_config
from catan_zero.rl.action_features import build_action_context_feature_table
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.policy_pool import load_checkpoint_policy
from catan_zero.rl.self_play import Policy
from catan_zero.rl.torch_ppo import TorchPPOPolicy, _masked_logits, _normalize_observation
from tools.evaluate_self_play import _make_policy


OPENING_PROMPTS = {"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CatanZero opening settlement and road choices.",
    )
    parser.add_argument(
        "--candidate",
        choices=(
            "random",
            "heuristic",
            "search",
            "value_rollout",
            "value",
            "linear",
            "mlp",
            "ppo",
            "checkpoint",
        ),
        default="ppo",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--driver",
        choices=("candidate", "heuristic", "value", "value_rollout"),
        default="candidate",
        help=(
            "Policy used to advance the opening sequence after recording each "
            "state. The candidate driver tests the positions the bot creates."
        ),
    )
    parser.add_argument(
        "--teachers",
        nargs="+",
        choices=("heuristic", "value", "value_rollout"),
        default=("value",),
        help="Opening teachers used for agreement and rank diagnostics.",
    )
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-opening-decisions", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=128)
    parser.add_argument("--presearch-candidate-limit", type=int, default=128)
    parser.add_argument("--rollout-decisions", type=int, default=2)
    parser.add_argument("--rollout-samples", type=int, default=1)
    parser.add_argument("--root-value-weight", type=float, default=0.35)
    parser.add_argument("--opponent-penalty", type=float, default=0.05)
    parser.add_argument("--sample-records", type=int, default=24)
    parser.add_argument("--output")
    args = parser.parse_args()

    candidate = (
        load_checkpoint_policy(args.checkpoint)
        if args.candidate == "checkpoint" and args.checkpoint
        else _make_policy(
            args.candidate,
            args.checkpoint,
            candidate_limit=args.candidate_limit,
            presearch_candidate_limit=args.presearch_candidate_limit,
            rollout_decisions=args.rollout_decisions,
            rollout_samples=args.rollout_samples,
            root_value_weight=args.root_value_weight,
            opponent_penalty=args.opponent_penalty,
        )
    )
    teachers = {
        name: _make_policy(
            name,
            None,
            candidate_limit=args.candidate_limit,
            presearch_candidate_limit=args.presearch_candidate_limit,
            rollout_decisions=args.rollout_decisions,
            rollout_samples=args.rollout_samples,
            root_value_weight=args.root_value_weight,
            opponent_penalty=args.opponent_penalty,
        )
        for name in args.teachers
    }
    driver = (
        candidate
        if args.driver == "candidate"
        else _make_policy(
            args.driver,
            None,
            candidate_limit=args.candidate_limit,
            presearch_candidate_limit=args.presearch_candidate_limit,
            rollout_decisions=args.rollout_decisions,
            rollout_samples=args.rollout_samples,
            root_value_weight=args.root_value_weight,
            opponent_penalty=args.opponent_penalty,
        )
    )

    report = evaluate_openings(
        candidate,
        teachers,
        driver=driver,
        games=args.games,
        players=args.players,
        seed=args.seed,
        vps_to_win=args.vps_to_win,
        max_opening_decisions=args.max_opening_decisions,
        sample_records=args.sample_records,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def evaluate_openings(
    candidate: Policy,
    teachers: dict[str, Policy],
    *,
    driver: Policy | None = None,
    games: int,
    seed: int,
    players: int = 2,
    vps_to_win: int = 10,
    max_opening_decisions: int = 16,
    sample_records: int = 24,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    game_seeds = [int(rng.integers(2**31)) for _ in range(games)]
    driver = driver or candidate
    records: list[dict[str, Any]] = []
    teacher_metrics = {
        name: _new_teacher_metrics() for name in sorted(teachers)
    }
    teacher_prompt_metrics = {
        name: {
            prompt: _new_teacher_metrics()
            for prompt in sorted(OPENING_PROMPTS)
        }
        for name in sorted(teachers)
    }
    prompt_counts = {prompt: 0 for prompt in sorted(OPENING_PROMPTS)}
    candidate_entropy_values: list[float] = []
    candidate_teacher_best_probs: list[float] = []
    invalid_actions = 0
    env_config = make_env_config(
        players=players,
        vps_to_win=vps_to_win,
        use_graph_history_features=(
            _needs_graph_history(candidate)
            or _needs_graph_history(driver)
            or any(_needs_graph_history(teacher) for teacher in teachers.values())
        ),
    )

    for game_index, game_seed in enumerate(game_seeds):
        env = ColonistMultiAgentEnv(env_config)
        try:
            observations, info = env.reset(seed=game_seed)
            terminated = False
            truncated = False
            opening_index = 0
            while (
                not (terminated or truncated)
                and opening_index < max_opening_decisions
                and str(info.get("current_prompt")) in OPENING_PROMPTS
            ):
                actor = str(info["current_player"])
                prompt = str(info["current_prompt"])
                observation = np.asarray(observations[actor], dtype=np.float64)
                valid_actions = tuple(int(action) for action in info["valid_actions"])
                candidate_action = int(
                    candidate.select_action(
                        env,
                        observation,
                        info,
                        rng,
                        training=False,
                    )
                )
                if candidate_action not in set(valid_actions):
                    invalid_actions += 1
                action_probs = _policy_action_probs(candidate, env, observation, info)
                if action_probs is not None:
                    entropy = _entropy(action_probs)
                    candidate_entropy_values.append(entropy)
                else:
                    entropy = None

                teacher_payloads: dict[str, Any] = {}
                for teacher_name, teacher in sorted(teachers.items()):
                    teacher_payload = _score_teacher(
                        teacher,
                        env,
                        observation,
                        info,
                        rng,
                        candidate_action=candidate_action,
                        action_probs=action_probs,
                    )
                    teacher_payloads[teacher_name] = teacher_payload
                    _accumulate_teacher_metrics(
                        teacher_metrics[teacher_name],
                        teacher_payload,
                    )
                    _accumulate_teacher_metrics(
                        teacher_prompt_metrics[teacher_name][prompt],
                        teacher_payload,
                    )
                    best_action = teacher_payload.get("best_action")
                    if (
                        action_probs is not None
                        and isinstance(best_action, int)
                        and best_action in action_probs
                    ):
                        candidate_teacher_best_probs.append(action_probs[best_action])

                prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
                if len(records) < sample_records:
                    records.append(
                        {
                            "game_index": game_index,
                            "seed": game_seed,
                            "opening_index": opening_index,
                            "actor": actor,
                            "prompt": prompt,
                            "valid_actions": len(valid_actions),
                            "candidate_action": candidate_action,
                            "candidate_label": _action_label(env, candidate_action),
                            "candidate_entropy": entropy,
                            "teachers": teacher_payloads,
                        }
                    )

                driver_action = int(
                    driver.select_action(
                        env,
                        observation,
                        info,
                        rng,
                        training=False,
                    )
                )
                observations, _, terminated, truncated, info = env.step(driver_action)
                opening_index += 1
        finally:
            env.close()

    opening_states = sum(prompt_counts.values())
    return {
        "games": games,
        "players": players,
        "seed": seed,
        "candidate": getattr(candidate, "name", type(candidate).__name__),
        "driver": getattr(driver, "name", type(driver).__name__),
        "teachers": sorted(teachers),
        "opening_states": opening_states,
        "prompt_counts": prompt_counts,
        "invalid_candidate_actions": invalid_actions,
        "candidate_mean_entropy": _mean(candidate_entropy_values),
        "candidate_mean_probability_on_teacher_best": _mean(
            candidate_teacher_best_probs,
        ),
        "teacher_metrics": {
            name: _finalize_teacher_metrics(metrics)
            for name, metrics in teacher_metrics.items()
        },
        "teacher_metrics_by_prompt": {
            name: {
                prompt: _finalize_teacher_metrics(metrics)
                for prompt, metrics in prompt_metrics.items()
            }
            for name, prompt_metrics in teacher_prompt_metrics.items()
        },
        "sample_records": records,
    }


def _score_teacher(
    teacher: Policy,
    env: ColonistMultiAgentEnv,
    observation: np.ndarray,
    info: dict[str, Any],
    rng: np.random.Generator,
    *,
    candidate_action: int,
    action_probs: dict[int, float] | None,
) -> dict[str, Any]:
    teacher_action = int(
        teacher.select_action(env, observation, info, rng, training=False)
    )
    target_scores = (
        teacher.target_scores(env, info, rng)  # type: ignore[attr-defined]
        if hasattr(teacher, "target_scores")
        else {}
    )
    target_policy = (
        teacher.target_policy(env, info, rng)  # type: ignore[attr-defined]
        if hasattr(teacher, "target_policy")
        else {}
    )
    payload: dict[str, Any] = {
        "teacher_action": teacher_action,
        "teacher_label": _action_label(env, teacher_action),
        "agreement": candidate_action == teacher_action,
        "scored_actions": len(target_scores),
        "teacher_probability_on_candidate": (
            float(target_policy.get(candidate_action, 0.0))
            if target_policy
            else None
        ),
        "candidate_probability_on_teacher": (
            float(action_probs.get(teacher_action, 0.0))
            if action_probs is not None
            else None
        ),
    }
    if target_scores:
        ranked = sorted(
            ((float(score), int(action)) for action, score in target_scores.items()),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        best_score, best_action = ranked[0]
        score_by_action = {action: score for score, action in ranked}
        candidate_score = score_by_action.get(candidate_action)
        rank = (
            next(
                index
                for index, (_, action) in enumerate(ranked, start=1)
                if action == candidate_action
            )
            if candidate_action in score_by_action
            else None
        )
        payload.update(
            {
                "best_action": best_action,
                "best_label": _action_label(env, best_action),
                "best_score": best_score,
                "candidate_score": candidate_score,
                "candidate_rank": rank,
                "candidate_rank_percentile": _rank_percentile(rank, len(ranked)),
                "score_gap_to_best": (
                    best_score - candidate_score
                    if candidate_score is not None
                    else None
                ),
            }
        )
    else:
        payload.update(
            {
                "best_action": teacher_action,
                "best_label": _action_label(env, teacher_action),
                "candidate_rank": 1 if candidate_action == teacher_action else None,
                "candidate_rank_percentile": (
                    1.0 if candidate_action == teacher_action else None
                ),
                "score_gap_to_best": None,
            }
        )
    return payload


def _policy_action_probs(
    policy: Policy,
    env: ColonistMultiAgentEnv,
    observation: np.ndarray,
    info: dict[str, Any],
) -> dict[int, float] | None:
    valid_actions = tuple(int(action) for action in info["valid_actions"])
    if not valid_actions:
        return None
    if isinstance(policy, EntityGraphPolicy):
        probs = policy.action_probs(env, info, valid_actions)
        return {
            int(action): float(probability)
            for action, probability in zip(valid_actions, probs)
        }
    if hasattr(policy, "action_probs"):
        probs = policy.action_probs(observation, valid_actions)  # type: ignore[attr-defined]
        return {
            int(action): float(probability)
            for action, probability in zip(valid_actions, probs)
        }
    if isinstance(policy, TorchPPOPolicy):
        import torch

        with torch.no_grad():
            obs = torch.as_tensor(
                _normalize_observation(observation),
                dtype=torch.float32,
                device=policy.device,
            ).unsqueeze(0)
            context = None
            if policy.context_action_feature_size > 0:
                context = torch.as_tensor(
                    build_action_context_feature_table(env, info),
                    dtype=torch.float32,
                    device=policy.device,
                ).unsqueeze(0)
            logits, _ = policy.forward(obs, context)
            masked = _masked_logits(logits, [valid_actions], policy.action_size)
            probs = torch.softmax(masked.squeeze(0), dim=-1)
            return {
                int(action): float(probs[int(action)].item())
                for action in valid_actions
            }
    return None


def _new_teacher_metrics() -> dict[str, Any]:
    return {
        "states": 0,
        "agreements": 0,
        "ranked_states": 0,
        "rank_sum": 0.0,
        "rank_percentiles": [],
        "score_gaps": [],
        "teacher_probability_on_candidate": [],
        "candidate_probability_on_teacher": [],
        "scored_actions": [],
    }


def _accumulate_teacher_metrics(metrics: dict[str, Any], payload: dict[str, Any]) -> None:
    metrics["states"] += 1
    metrics["agreements"] += int(bool(payload.get("agreement")))
    rank = payload.get("candidate_rank")
    if isinstance(rank, int):
        metrics["ranked_states"] += 1
        metrics["rank_sum"] += float(rank)
    for key, output_key in (
        ("candidate_rank_percentile", "rank_percentiles"),
        ("score_gap_to_best", "score_gaps"),
        ("teacher_probability_on_candidate", "teacher_probability_on_candidate"),
        ("candidate_probability_on_teacher", "candidate_probability_on_teacher"),
    ):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            metrics[output_key].append(float(value))
    metrics["scored_actions"].append(float(payload.get("scored_actions", 0)))


def _finalize_teacher_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    states = int(metrics["states"])
    ranked_states = int(metrics["ranked_states"])
    return {
        "states": states,
        "agreement_rate": metrics["agreements"] / states if states else 0.0,
        "ranked_states": ranked_states,
        "mean_candidate_rank": (
            metrics["rank_sum"] / ranked_states if ranked_states else None
        ),
        "mean_candidate_rank_percentile": _mean(metrics["rank_percentiles"]),
        "mean_score_gap_to_best": _mean(metrics["score_gaps"]),
        "mean_teacher_probability_on_candidate": _mean(
            metrics["teacher_probability_on_candidate"],
        ),
        "mean_candidate_probability_on_teacher": _mean(
            metrics["candidate_probability_on_teacher"],
        ),
        "mean_scored_actions": _mean(metrics["scored_actions"]),
    }


def _action_label(env: ColonistMultiAgentEnv, action: int | None) -> str | None:
    structured = env.structured_action(action)
    if structured is None:
        return None
    return str(structured.get("label") or structured.get("action_type"))


def _rank_percentile(rank: int | None, count: int) -> float | None:
    if rank is None or count <= 0:
        return None
    if count == 1:
        return 1.0
    return 1.0 - ((rank - 1) / float(count - 1))


def _entropy(probs_by_action: dict[int, float]) -> float:
    values = np.asarray(list(probs_by_action.values()), dtype=np.float64)
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    return float(-(values * np.log(values)).sum())


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _needs_graph_history(policy: Policy) -> bool:
    return getattr(policy, "architecture", "") == "graph_history_candidate"


if __name__ == "__main__":
    main()
