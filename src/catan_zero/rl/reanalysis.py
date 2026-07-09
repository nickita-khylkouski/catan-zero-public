from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np

from catan_zero.rl.self_play import StepSample, TrainingEpisode


REANALYSIS_JSONL_VERSION = 1


def flatten_episode_for_reanalysis(
    episode: TrainingEpisode,
    *,
    gamma: float,
) -> tuple[list[StepSample], list[float]]:
    samples: list[StepSample] = []
    returns: list[float] = []
    for player, player_samples in episode.samples_by_player.items():
        n = len(player_samples)
        for idx, sample in enumerate(player_samples):
            samples.append(sample)
            returns.append(float(episode.result.rewards[player]) * (gamma ** (n - idx - 1)))
    return samples, returns


def write_reanalysis_jsonl(
    path: str | Path,
    samples: list[StepSample],
    returns: list[float] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    append: bool = False,
) -> int:
    if returns is not None and len(returns) != len(samples):
        raise ValueError("returns must have the same length as samples")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for idx, sample in enumerate(samples):
            record = sample_to_reanalysis_record(
                sample,
                return_value=None if returns is None else returns[idx],
                metadata=metadata,
            )
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return len(samples)


def load_reanalysis_jsonl(
    paths: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    max_samples: int = 0,
) -> tuple[list[StepSample], list[float]]:
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = list(paths)
    samples: list[StepSample] = []
    returns: list[float] = []
    for path in path_list:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                sample, return_value = reanalysis_record_to_sample(json.loads(stripped))
                samples.append(sample)
                returns.append(float(return_value))
                if max_samples > 0 and len(samples) >= max_samples:
                    return samples, returns
    return samples, returns


def sample_to_reanalysis_record(
    sample: StepSample,
    *,
    return_value: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context_record = _context_to_record(sample)
    return {
        "version": REANALYSIS_JSONL_VERSION,
        "observation": _array_to_list(sample.observation),
        "valid_actions": [int(action) for action in sample.valid_actions],
        "action": int(sample.action),
        "player": sample.player,
        "sample_weight": float(sample.sample_weight),
        **context_record,
        "target_policy": _int_key_float_dict(sample.target_policy),
        "target_scores": _int_key_float_dict(sample.target_scores),
        "return": 0.0 if return_value is None else float(return_value),
        "decision_index": sample.decision_index,
        "metadata": metadata or {},
    }


def reanalysis_record_to_sample(record: dict[str, Any]) -> tuple[StepSample, float]:
    version = int(record.get("version", 0))
    if version != REANALYSIS_JSONL_VERSION:
        raise ValueError(f"unsupported reanalysis version: {version}")
    sample = StepSample(
        observation=np.asarray(record["observation"], dtype=np.float64),
        valid_actions=tuple(int(action) for action in record["valid_actions"]),
        action=int(record["action"]),
        player=str(record["player"]),
        action_context_features=_context_from_record(record),
        target_policy=_float_dict_to_int_keys(record.get("target_policy")),
        target_scores=_float_dict_to_int_keys(record.get("target_scores")),
        sample_weight=max(0.0, float(record.get("sample_weight", 1.0))),
        decision_index=(
            None
            if record.get("decision_index") is None
            else int(record["decision_index"])
        ),
    )
    return sample, float(record.get("return", 0.0))


def _context_to_record(sample: StepSample) -> dict[str, Any]:
    value = sample.action_context_features
    if value is None:
        return {"action_context_features": None}
    context = np.asarray(value, dtype=np.float32)
    if context.ndim != 2:
        raise ValueError("action_context_features must be a 2D array")
    valid_actions = [int(action) for action in sample.valid_actions]
    return {
        "action_context_features": None,
        "action_context_storage": "valid_actions",
        "action_context_action_size": int(context.shape[0]),
        "action_context_feature_size": int(context.shape[1]),
        "valid_action_context_features": [
            _array_to_list(context[action]) for action in valid_actions
        ],
    }


def _context_from_record(record: dict[str, Any]) -> np.ndarray | None:
    dense = record.get("action_context_features")
    if dense is not None:
        return np.asarray(dense, dtype=np.float32)
    if record.get("action_context_storage") != "valid_actions":
        return None
    valid_actions = tuple(int(action) for action in record["valid_actions"])
    rows = record.get("valid_action_context_features")
    if not isinstance(rows, list) or len(rows) != len(valid_actions):
        raise ValueError("valid action context features must align with valid_actions")
    action_size = int(record["action_context_action_size"])
    feature_size = int(record["action_context_feature_size"])
    context = np.zeros((action_size, feature_size), dtype=np.float32)
    for action, row in zip(valid_actions, rows):
        context[action] = np.asarray(row, dtype=np.float32)
    return context


def _array_to_list(value: np.ndarray) -> list[Any]:
    return np.asarray(value).tolist()


def _int_key_float_dict(value: dict[int, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {str(int(key)): float(score) for key, score in sorted(value.items())}


def _float_dict_to_int_keys(value: Any) -> dict[int, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("target policy/scores must be a JSON object")
    return {int(key): float(score) for key, score in value.items()}
