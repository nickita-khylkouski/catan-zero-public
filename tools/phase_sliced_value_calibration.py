#!/usr/bin/env python3
# ruff: noqa: E402 -- executable adds the sibling tools directory before imports.
"""Phase-sliced value-head calibration (f70 D4).

Takes any self-play shard directory (the `*.npz` shards written by
`gumbel_self_play` / the raw-selfplay generators) plus a checkpoint, and
reports value-head calibration -- corr(q, z), value RMSE, Brier, and binary
reliability/ECE -- GLOBALLY and SLICED by game phase and by legal-action-count
bucket. ``--value-readout scalar`` is the backward-compatible default;
``categorical`` selects the trained HL-Gauss expectation and additionally
reports terminal-outcome categorical cross-entropy/NLL. Categorical selection
fails closed unless the checkpoint carries positive value-training provenance,
so a config-only random head cannot be mistaken for a trained readout. The Gate-A
post-mortem showed global corr(q, z) hides the failure that actually matters
for search: the value head can be well-calibrated on average yet rank
candidates by noise at wide placement roots. Per-phase / per-legal-count
calibration is the diagnostic that exposes that.

`q` is the selected value-head expectation from a direct forward pass over the entity
features stored per row (no search, no game replay), exactly as
`tools/value_repair_calibration_probe.py` does; `z` is the true terminal
outcome (+1 win / -1 loss) for the row's acting player. Only rows whose game
TERMINATED naturally (not truncated) are used -- truncated games have no
clean +-1 label.

PHASE SLICING NOTE: the stored `phase` column is coarse -- ROLL, dev-card
plays, builds, trades and END_TURN all live under a single "PLAY_TURN"
value, so a true dev-vs-build split is NOT recoverable from the shard alone
(it would require decoding `action_taken` back to an action type by replaying
from `game_seed`, out of scope for a lightweight analysis tool). We therefore
slice by the real stored `phase` vocab (opening placement / robber / discard /
play-turn), plus a cross-cutting `forced` slice (the stored `is_forced`
flag), plus legal-action-count buckets. If a finer dev/build split is needed
later, decode `action_taken` per row.

For experiment go/no-go artifacts, pass ``--require-held-out`` together with
the trainer's explicit game-seed ranges, a saved validation-seed manifest, or
the exact ``--validation-fraction``/``--validation-seed`` pair. Fraction mode
reproduces ``train_bc``'s game-level selection from all source rows and can
persist the selected games for both arms; it never falls back to a row split.
"""

from __future__ import annotations

import json
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, TypedDict, cast

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluatorConfig,
    _assert_value_readout_available,
)
from factory_common import write_json

# Same entity feature keys the model consumes, matching
# `tools/value_repair_calibration_probe.py`.
ENTITY_KEYS = (
    "hex_tokens",
    "hex_vertex_ids",
    "hex_edge_ids",
    "vertex_tokens",
    "edge_tokens",
    "edge_vertex_ids",
    "player_tokens",
    "global_tokens",
    "legal_action_tokens",
    "legal_action_target_ids",
    "event_tokens",
    "event_target_ids",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "legal_action_mask",
    "event_mask",
)

# Map the coarse stored `phase` values to friendly slice labels.
_PHASE_LABELS = {
    "BUILD_INITIAL_SETTLEMENT": "opening_placement",
    "BUILD_INITIAL_ROAD": "opening_placement",
    "MOVE_ROBBER": "robber",
    "DISCARD": "discard",
    "PLAY_TURN": "play_turn",
}

# Architecture-probe legal-width contract (upper bound inclusive). Keep 41+
# pooled: this is the wide-root stratum the n64/n128/D6 decision acts on.
_LEGAL_BUCKETS = (
    (1, "1"),
    (4, "2-4"),
    (10, "5-10"),
    (20, "11-20"),
    (40, "21-40"),
)

ValueReadout = Literal["scalar", "categorical"]
RowSelectionMode = Literal[
    "all_natural_terminal_rows",
    "validation_fraction",
    "validation_seed_manifest",
    "validation_game_seed_ranges",
]


class ReadoutProvenance(TypedDict):
    """JSON-safe proof of which checkpoint tensor was calibrated.

    A config-only ``catbins:N`` upgrade has categorical parameters but no
    evidence that an optimizer ever trained them.  Keeping these fields in the
    artifact makes that distinction durable instead of relying on the command
    line or the checkpoint filename.
    """

    requested_readout: ValueReadout
    model_output_key: str
    categorical_training_verified: bool
    trained_value_readouts: list[str]
    value_training_schema_version: str | None
    categorical_bins: int
    categorical_truncation_class: bool
    categorical_objective_weight: float | None
    categorical_training_weight_sum: float | None
    hlgauss_sigma_ratio: float | None
    optimizer_steps: int | None
    completed_epochs: int | None


class ReliabilityBin(TypedDict):
    lower: float
    upper: float
    n: int
    mean_predicted_win_probability: float | None
    empirical_win_rate: float | None
    absolute_calibration_gap: float | None


class RowSelectionProvenance(TypedDict):
    mode: RowSelectionMode
    held_out_filter_applied: bool
    validation_fraction: float | None
    validation_seed: int | None
    validation_game_seed_ranges: list[list[int]]
    seed_manifest_path: str | None
    seed_manifest_sha256: str | None
    configured_game_seed_count: int | None
    configured_game_seed_set_sha256: str | None
    observed_game_seed_count: int
    observed_game_seed_set_sha256: str
    observed_row_count: int


class ValueScaleSplit(TypedDict):
    game_count: int
    row_count: int
    game_seed_set_sha256: str


class ValueScaleFit(TypedDict):
    """Held-out, game-disjoint diagnostic for the scalar search transform.

    This is deliberately not an operator mutation.  It estimates a positive
    multiplier for ``tanh(q_raw * value_scale)`` on one subset of held-out
    games and scores it on another subset.  A later search H2H is still needed
    before changing the sealed evaluator configuration.
    """

    schema_version: str
    diagnostic_only: bool
    changes_operator_default: bool
    selection_objective: str
    current_value_scale: float
    selected_value_scale: float
    scale_grid_min: float
    scale_grid_max: float
    scale_grid_count: int
    selected_at_grid_boundary: bool
    split_seed: int
    calibration_fraction: float
    calibration: ValueScaleSplit
    evaluation: ValueScaleSplit
    calibration_current: dict[str, Any]
    calibration_selected: dict[str, Any]
    evaluation_current: dict[str, Any]
    evaluation_selected: dict[str, Any]
    evaluation_row_weighted_mse_reduction: float
    evaluation_game_balanced_mse_reduction: float
    promotion_blocked_without_search_h2h: bool


@dataclass(frozen=True)
class ReadoutPredictions:
    """Per-row values needed by every slice, without retaining full logits."""

    q: np.ndarray
    categorical_hlgauss_ce: np.ndarray | None
    categorical_terminal_nll: np.ndarray | None
    categorical_truncation_probability: np.ndarray | None
    provenance: ReadoutProvenance


@dataclass(frozen=True)
class ValidationSeedSelection:
    game_seeds: np.ndarray
    validation_fraction: float
    validation_seed: int
    source_row_count: int
    source_game_count: int


def _legal_bucket(count: int) -> str:
    for upper, label in _LEGAL_BUCKETS:
        if count <= upper:
            return label
    return "41+"


def _normalized_shard_dirs(shard_dirs: str | Sequence[str]) -> list[str]:
    raw = [shard_dirs] if isinstance(shard_dirs, str) else list(shard_dirs)
    if not raw:
        raise SystemExit("at least one --shard-dir is required")
    resolved = [str(Path(value).expanduser().resolve(strict=True)) for value in raw]
    if len(set(resolved)) != len(resolved):
        raise SystemExit("--shard-dir roots must be unique")
    return resolved


def _iter_shards(shard_dirs: str | Sequence[str]) -> list[str]:
    roots = _normalized_shard_dirs(shard_dirs)
    shards = sorted(
        str(path.resolve(strict=True))
        for root in roots
        for path in Path(root).rglob("*.npz")
    )
    if len(set(shards)) != len(shards):
        raise SystemExit("--shard-dir roots overlap on the same .npz payload")
    if not shards:
        raise SystemExit(f"no .npz shards found under {roots}")
    return shards


def parse_validation_game_seed_ranges(raw: str) -> tuple[tuple[int, int], ...]:
    """Parse the trainer's inclusive ``start:end,start:end`` range syntax."""

    ranges: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"invalid validation game-seed range {chunk!r}; expected start:end"
            )
        start_text, end_text = chunk.split(":", 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError(
                f"invalid validation game-seed range {chunk!r}; end < start"
            )
        ranges.append((start, end))
    return tuple(ranges)


def derive_validation_game_seeds(
    shard_dir: str | Sequence[str], *, validation_fraction: float, validation_seed: int
) -> ValidationSeedSelection:
    """Reproduce ``train_bc.split_train_validation_indices`` at game level.

    Counts come from *all* source rows, not merely naturally terminated rows,
    because the trainer chooses validation games until their total row count
    reaches ``round(N * validation_fraction)``.  This produces the same game
    set even though calibration later excludes truncated/no-outcome rows.
    """

    fraction = float(np.clip(validation_fraction, 0.0, 0.9))
    if fraction <= 0.0:
        raise ValueError("validation_fraction must be > 0 for held-out calibration")
    seed_counts: dict[int, int] = {}
    total_rows = 0
    for shard_path in _iter_shards(shard_dir):
        with np.load(shard_path) as data:
            if "game_seed" not in data.files:
                raise ValueError(
                    f"{shard_path} has no game_seed; refusing a row-level split"
                )
            seeds = np.asarray(data["game_seed"], dtype=np.int64).reshape(-1)
        unique, counts = np.unique(seeds, return_counts=True)
        for game_seed, count in zip(unique.tolist(), counts.tolist(), strict=True):
            seed_counts[int(game_seed)] = seed_counts.get(int(game_seed), 0) + int(
                count
            )
        total_rows += int(len(seeds))
    if total_rows == 0 or len(seed_counts) <= 1:
        raise ValueError(
            "validation_fraction requires a non-empty corpus with multiple game seeds"
        )
    unique_seeds = np.asarray(sorted(seed_counts), dtype=np.int64)
    shuffled = np.random.default_rng(validation_seed).permutation(unique_seeds)
    target_rows = max(1, int(round(total_rows * fraction)))
    selected: list[int] = []
    selected_rows = 0
    for game_seed in shuffled:
        value = int(game_seed)
        selected.append(value)
        selected_rows += seed_counts[value]
        if selected_rows >= target_rows:
            break
    return ValidationSeedSelection(
        game_seeds=np.asarray(sorted(selected), dtype=np.int64),
        validation_fraction=fraction,
        validation_seed=int(validation_seed),
        source_row_count=total_rows,
        source_game_count=len(seed_counts),
    )


def load_validation_seed_manifest(path: str | Path) -> tuple[np.ndarray, str]:
    """Load a typed immutable list of validation games and return its file SHA-256.

    Accept both this tool's standalone manifest and the trainer's
    ``<report>.validation_seeds.json``.  Direct trainer-manifest support avoids a
    translation step that could silently calibrate a different game set from the
    one used for epoch validation.
    """

    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} is not a typed validation-seed manifest")
    schema = payload.get("schema_version")
    if schema == "value-calibration-validation-seeds-v1":
        seeds = payload.get("validation_game_seeds")
    elif schema == "train-validation-game-seeds-v1":
        seeds = payload.get("game_seeds")
    else:
        raise ValueError(
            f"{manifest_path} has unsupported validation-seed schema {schema!r}"
        )
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{manifest_path} has no validation game-seed list")
    try:
        values = np.asarray([int(seed) for seed in seeds], dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{manifest_path} validation_game_seeds must be integers"
        ) from error
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{manifest_path} contains duplicate validation game seeds")
    values = np.sort(values)
    if schema == "train-validation-game-seeds-v1":
        declared_count = payload.get("validation_game_seed_count")
        if declared_count is not None and int(declared_count) != len(values):
            raise ValueError(
                f"{manifest_path} validation_game_seed_count={declared_count} "
                f"does not match {len(values)} listed seeds"
            )
        declared_digest = payload.get("validation_game_seed_set_sha256")
        actual_digest = (
            "sha256:"
            + hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()
        )
        if declared_digest is not None and str(declared_digest) != actual_digest:
            raise ValueError(
                f"{manifest_path} validation game-seed digest mismatch: "
                f"declared={declared_digest!r} actual={actual_digest!r}"
            )
    return values, hashlib.sha256(raw).hexdigest()


def write_validation_seed_manifest(
    path: str | Path, selection: ValidationSeedSelection, *, shard_dir: str
) -> str:
    payload = {
        "schema_version": "value-calibration-validation-seeds-v1",
        "source_shard_dir": str(Path(shard_dir).resolve()),
        "source_row_count": int(selection.source_row_count),
        "source_game_count": int(selection.source_game_count),
        "validation_fraction": float(selection.validation_fraction),
        "validation_seed": int(selection.validation_seed),
        "validation_game_seed_count": int(len(selection.game_seeds)),
        "validation_game_seeds": [int(seed) for seed in selection.game_seeds],
    }
    write_json(path, payload)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_rows(
    shard_dir: str | Sequence[str],
    *,
    max_rows: int | None = None,
    max_rows_per_shard: int | None = None,
    validation_game_seeds: np.ndarray | None = None,
    validation_game_seed_ranges: tuple[tuple[int, int], ...] = (),
) -> list[dict[str, np.ndarray]]:
    """One group per shard (each shard has a self-consistent legal-action
    padding width -- see the calibration probe's grouping rationale), holding
    the entity features + legal-action arrays needed for the forward pass and
    the per-row slice keys (phase label, forced flag, legal count) and z."""
    if validation_game_seeds is not None and validation_game_seed_ranges:
        raise ValueError(
            "choose validation_game_seeds or validation_game_seed_ranges, not both"
        )
    if max_rows_per_shard is not None and int(max_rows_per_shard) < 1:
        raise ValueError("max_rows_per_shard must be positive")
    groups: list[dict[str, np.ndarray]] = []
    total = 0
    for shard_path in _iter_shards(shard_dir):
        data = np.load(shard_path)
        if "terminated" not in data.files:
            continue
        terminated = data["terminated"] & ~data["truncated"]
        if validation_game_seeds is not None or validation_game_seed_ranges:
            if "game_seed" not in data.files:
                raise ValueError(
                    f"{shard_path} has no game_seed; held-out selection cannot be verified"
                )
            shard_seeds = np.asarray(data["game_seed"], dtype=np.int64)
            if validation_game_seeds is not None:
                held_out = np.isin(shard_seeds, validation_game_seeds)
            else:
                held_out = np.zeros(len(shard_seeds), dtype=bool)
                for start, end in validation_game_seed_ranges:
                    held_out |= (shard_seeds >= start) & (shard_seeds <= end)
            terminated &= held_out
        if not np.any(terminated):
            continue
        idx = np.where(terminated)[0]
        if max_rows_per_shard is not None:
            idx = idx[: int(max_rows_per_shard)]
        if max_rows is not None:
            remaining = max(0, int(max_rows) - total)
            idx = idx[:remaining]
            if not len(idx):
                break
        winner = data["winner"][idx]
        player = data["player"][idx]
        z = np.where(winner == player, 1.0, -1.0).astype(np.float32)

        phases = data["phase"][idx]
        phase_labels = np.array([_PHASE_LABELS.get(str(p), str(p)) for p in phases])
        forced = (
            data["is_forced"][idx].astype(bool)
            if "is_forced" in data.files
            else np.zeros(len(idx), dtype=bool)
        )
        legal_count = data["legal_action_mask"][idx].sum(axis=1).astype(int)

        group: dict[str, np.ndarray] = {key: data[key][idx] for key in ENTITY_KEYS}
        group["legal_action_ids"] = data["legal_action_ids"][idx]
        group["legal_action_context"] = data["legal_action_context"][idx]
        if "game_seed" in data.files:
            group["game_seed"] = np.asarray(data["game_seed"])[idx]
        group["z"] = z
        group["phase_label"] = phase_labels
        group["forced"] = forced
        group["legal_count"] = legal_count
        groups.append(group)
        total += len(idx)
        if max_rows is not None and total >= max_rows:
            break
    if not groups:
        raise SystemExit("no naturally-terminated rows found in shard dir")
    return groups


def resolve_readout_provenance(
    policy: EntityGraphPolicy, value_readout: str
) -> ReadoutProvenance:
    """Validate the requested readout before any rows are scored.

    Scalar is the historical default and remains available to legacy
    checkpoints.  Categorical deliberately fails closed unless the loader has
    validated a positive ``value-training-v1`` attestation *and* the checkpoint
    contains the trained head weights.  Merely declaring ``catbins:N`` in the
    model config is not evidence that the random head was optimized.
    """

    if value_readout not in {"scalar", "categorical"}:
        raise ValueError(
            f"unknown value_readout {value_readout!r}; expected 'scalar' or 'categorical'"
        )
    readout = cast(ValueReadout, value_readout)
    _assert_value_readout_available(
        policy, EntityGraphRustEvaluatorConfig(value_readout=readout)
    )
    model = getattr(policy, "model", None)
    bins = int(getattr(model, "value_categorical_bins", 0) or 0)
    has_truncation_class = bool(
        getattr(model, "value_categorical_truncation_class", False)
    )
    trained_readouts = [
        str(item)
        for item in getattr(policy, "trained_value_readouts", ("scalar",))
        if str(item) in {"scalar", "categorical"}
    ]
    raw_value_training = getattr(policy, "value_training", None)
    value_training = (
        dict(raw_value_training) if isinstance(raw_value_training, dict) else None
    )

    def _number(key: str, kind):
        if value_training is None or key not in value_training:
            return None
        try:
            return kind(value_training[key])
        except (TypeError, ValueError):
            return None

    schema = (
        str(value_training.get("schema_version", "") or "") if value_training else None
    )
    categorical_weight = _number("resolved_categorical_ce_weight", float)
    categorical_mass = _number("categorical_training_weight_sum", float)
    sigma_ratio = _number("hlgauss_sigma_ratio", float)
    optimizer_steps = _number("optimizer_steps", int)
    completed_epochs = _number("completed_epochs", int)
    categorical_verified = "categorical" in trained_readouts
    if readout == "categorical" and (sigma_ratio is None or sigma_ratio <= 0.0):
        raise ValueError(
            "categorical calibration requires a positive hlgauss_sigma_ratio in "
            "value-training-v1 checkpoint provenance"
        )

    return ReadoutProvenance(
        requested_readout=readout,
        model_output_key="value" if readout == "scalar" else "value_categorical",
        categorical_training_verified=categorical_verified,
        trained_value_readouts=trained_readouts,
        value_training_schema_version=schema,
        categorical_bins=bins,
        categorical_truncation_class=has_truncation_class,
        categorical_objective_weight=categorical_weight,
        categorical_training_weight_sum=categorical_mass,
        hlgauss_sigma_ratio=sigma_ratio,
        optimizer_steps=optimizer_steps,
        completed_epochs=completed_epochs,
    )


def _hlgauss_targets(targets, bins: int, *, sigma_ratio: float):
    """Trainer-matched integrated-Gaussian targets for clean terminal rows."""

    import torch

    centers = torch.linspace(-1.0, 1.0, bins, device=targets.device)
    bin_width = 2.0 / float(bins - 1)
    sigma = max(float(sigma_ratio), 1.0e-6) * bin_width
    lower = (centers - bin_width / 2.0).clone()
    upper = (centers + bin_width / 2.0).clone()
    lower[0] = float("-inf")
    upper[-1] = float("inf")
    y = targets.detach().float().clamp(-1.0, 1.0).unsqueeze(-1)
    inv = 1.0 / (sigma * math.sqrt(2.0))
    cdf_hi = 0.5 * (1.0 + torch.erf((upper.unsqueeze(0) - y) * inv))
    cdf_lo = 0.5 * (1.0 + torch.erf((lower.unsqueeze(0) - y) * inv))
    probs = (cdf_hi - cdf_lo).clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def compute_readout(
    policy: EntityGraphPolicy,
    groups: list[dict[str, np.ndarray]],
    *,
    value_readout: str = "scalar",
) -> ReadoutPredictions:
    import torch

    provenance = resolve_readout_provenance(policy, value_readout)
    output_key = provenance["model_output_key"]
    q_chunks: list[np.ndarray] = []
    hlgauss_ce_chunks: list[np.ndarray] = []
    terminal_nll_chunks: list[np.ndarray] = []
    truncation_prob_chunks: list[np.ndarray] = []
    for group in groups:
        entity_batch = {key: group[key] for key in ENTITY_KEYS}
        with torch.no_grad():
            outputs = policy.forward_legal_np(
                entity_batch, group["legal_action_ids"], group["legal_action_context"]
            )
        if output_key not in outputs:
            raise RuntimeError(
                f"value_readout={value_readout!r} requested model output "
                f"{output_key!r}, but forward emitted {sorted(outputs)}"
            )
        q_chunks.append(outputs[output_key].detach().float().cpu().numpy().reshape(-1))
        if value_readout == "categorical":
            logits = outputs.get("value_categorical_logits")
            if logits is None:
                raise RuntimeError(
                    "categorical calibration requires value_categorical_logits; "
                    f"forward emitted {sorted(outputs)}"
                )
            logits = logits.detach().float()
            bins = int(provenance["categorical_bins"])
            expected_classes = bins + int(provenance["categorical_truncation_class"])
            if logits.ndim != 2 or int(logits.shape[-1]) != expected_classes:
                raise RuntimeError(
                    "categorical logit width does not match checkpoint metadata: "
                    f"shape={tuple(logits.shape)}, expected last dim {expected_classes}"
                )
            z = torch.as_tensor(group["z"], device=logits.device)
            log_probs = torch.log_softmax(logits, dim=-1)
            hlgauss_target = _hlgauss_targets(
                z,
                bins,
                sigma_ratio=float(provenance["hlgauss_sigma_ratio"]),
            )
            if provenance["categorical_truncation_class"]:
                hlgauss_target = torch.nn.functional.pad(
                    hlgauss_target, (0, 1), value=0.0
                )
            hlgauss_ce_chunks.append(
                (-(hlgauss_target * log_probs).sum(dim=-1)).cpu().numpy()
            )
            # Naturally terminated labels are +/-1, hence the endpoint bins.
            # The full softmax denominator includes the optional truncation
            # class: leaking mass there is correctly penalized by this proper
            # terminal-outcome cross-entropy / NLL.
            target_index = torch.where(
                z > 0,
                torch.full_like(z, bins - 1, dtype=torch.long),
                torch.zeros_like(z, dtype=torch.long),
            )
            row_nll = -log_probs.gather(1, target_index.reshape(-1, 1)).squeeze(1)
            terminal_nll_chunks.append(row_nll.cpu().numpy())
            truncation_prob = outputs.get("value_categorical_truncation_prob")
            if provenance["categorical_truncation_class"]:
                if truncation_prob is None:
                    raise RuntimeError(
                        "checkpoint declares a categorical truncation class, but "
                        "forward omitted value_categorical_truncation_prob"
                    )
                truncation_prob_chunks.append(
                    truncation_prob.detach().float().cpu().numpy().reshape(-1)
                )

    return ReadoutPredictions(
        q=np.concatenate(q_chunks, axis=0),
        categorical_hlgauss_ce=(
            np.concatenate(hlgauss_ce_chunks, axis=0) if hlgauss_ce_chunks else None
        ),
        categorical_terminal_nll=(
            np.concatenate(terminal_nll_chunks, axis=0) if terminal_nll_chunks else None
        ),
        categorical_truncation_probability=(
            np.concatenate(truncation_prob_chunks, axis=0)
            if truncation_prob_chunks
            else None
        ),
        provenance=provenance,
    )


def compute_q(
    policy: EntityGraphPolicy, groups: list[dict[str, np.ndarray]]
) -> np.ndarray:
    """Historical scalar helper retained byte-for-byte at the call boundary."""

    return compute_readout(policy, groups, value_readout="scalar").q


def _reliability_stats(
    q: np.ndarray, z: np.ndarray, *, bin_count: int
) -> tuple[float | None, list[ReliabilityBin]]:
    if bin_count <= 0:
        raise ValueError(f"reliability bin_count must be positive, got {bin_count}")
    n = int(len(z))
    if not n:
        return None, []
    p = np.clip((q.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)
    outcome = (z > 0).astype(np.float64)
    bin_index = np.minimum((p * bin_count).astype(np.int64), bin_count - 1)
    rows: list[ReliabilityBin] = []
    weighted_gap = 0.0
    for index in range(bin_count):
        mask = bin_index == index
        count = int(mask.sum())
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if count:
            mean_p = float(p[mask].mean())
            empirical = float(outcome[mask].mean())
            gap = abs(mean_p - empirical)
            weighted_gap += count * gap
        else:
            mean_p = empirical = gap = None
        rows.append(
            ReliabilityBin(
                lower=float(lower),
                upper=float(upper),
                n=count,
                mean_predicted_win_probability=mean_p,
                empirical_win_rate=empirical,
                absolute_calibration_gap=gap,
            )
        )
    return float(weighted_gap / n), rows


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ranks with equal values assigned their average rank."""

    array = np.asarray(values).reshape(-1)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """Spearman correlation without an optional scipy runtime dependency."""

    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = _average_tie_ranks(left)
    right_rank = _average_tie_ranks(right)
    if float(np.std(left_rank)) == 0.0 or float(np.std(right_rank)) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _calibration_stats(
    q: np.ndarray,
    z: np.ndarray,
    *,
    min_rows: int,
    categorical_hlgauss_ce: np.ndarray | None = None,
    categorical_terminal_nll: np.ndarray | None = None,
    categorical_truncation_probability: np.ndarray | None = None,
    reliability_bin_count: int = 10,
) -> dict[str, Any]:
    n = int(len(z))
    if len(q) != n:
        raise ValueError(f"q/z row mismatch: {len(q)} != {n}")
    if categorical_hlgauss_ce is not None and len(categorical_hlgauss_ce) != n:
        raise ValueError(
            "categorical_hlgauss_ce/z row mismatch: "
            f"{len(categorical_hlgauss_ce)} != {n}"
        )
    if categorical_terminal_nll is not None and len(categorical_terminal_nll) != n:
        raise ValueError(
            "categorical_terminal_nll/z row mismatch: "
            f"{len(categorical_terminal_nll)} != {n}"
        )
    if (
        categorical_truncation_probability is not None
        and len(categorical_truncation_probability) != n
    ):
        raise ValueError(
            "categorical_truncation_probability/z row mismatch: "
            f"{len(categorical_truncation_probability)} != {n}"
        )
    win_mask = z > 0
    n_win = int(win_mask.sum())
    n_loss = int((~win_mask).sum())
    stats: dict[str, Any] = {
        "n": n,
        "n_win": n_win,
        "n_loss": n_loss,
        "win_rate": (n_win / n) if n else None,
        "q_mean": float(q.mean()) if n else None,
        "q_std": float(q.std()) if n else None,
        "bias": float(np.mean(q - z)) if n else None,
    }
    if n < min_rows or n_win == 0 or n_loss == 0 or float(np.std(q)) == 0.0:
        # corr is undefined without both classes / enough rows.
        stats["corr_q_z"] = None
    else:
        stats["corr_q_z"] = float(np.corrcoef(q, z)[0, 1])
    stats["spearman_q_z"] = (
        _spearman_correlation(q, z)
        if n >= min_rows and n_win > 0 and n_loss > 0
        else None
    )
    stats["e_q_given_win"] = float(q[win_mask].mean()) if n_win else None
    stats["e_q_given_loss"] = float(q[~win_mask].mean()) if n_loss else None
    # Brier: outcome in {0,1}, predicted prob p = (q+1)/2 clipped to [0,1].
    if n:
        outcome = (z + 1.0) / 2.0
        p = np.clip((q + 1.0) / 2.0, 0.0, 1.0)
        stats["brier"] = float(np.mean((p - outcome) ** 2))
        # Value-space residual RMSE (q vs the +-1 outcome). This is the
        # recommended per-checkpoint / per-phase estimate for the search's
        # `sigma_eval` noise-floor knob (D1): the opening_placement slice's
        # value_rmse is the relevant sigma for the noise floor at wide
        # placement roots. It is an UPPER bound on the pure estimator noise
        # (it also absorbs the irreducible outcome variance given a state),
        # but it is the standard, directly-usable practical proxy.
        stats["value_rmse"] = float(np.sqrt(np.mean((q - z) ** 2)))
    else:
        stats["brier"] = None
        stats["value_rmse"] = None
    ece, reliability = _reliability_stats(q, z, bin_count=reliability_bin_count)
    stats["win_probability_ece"] = ece
    stats["reliability_bins"] = reliability
    stats["categorical_hlgauss_ce"] = (
        float(np.mean(categorical_hlgauss_ce))
        if categorical_hlgauss_ce is not None and n
        else None
    )
    stats["categorical_terminal_nll"] = (
        float(np.mean(categorical_terminal_nll))
        if categorical_terminal_nll is not None and n
        else None
    )
    stats["categorical_score_n"] = (
        n
        if categorical_hlgauss_ce is not None and categorical_terminal_nll is not None
        else 0
    )
    stats["categorical_truncation_probability_mean"] = (
        float(np.mean(categorical_truncation_probability))
        if categorical_truncation_probability is not None and n
        else None
    )
    return stats


def _game_seed_set_sha256(game_seeds: np.ndarray) -> str:
    canonical = np.unique(np.asarray(game_seeds, dtype=np.int64)).astype(
        "<i8", copy=False
    )
    return "sha256:" + hashlib.sha256(canonical.tobytes()).hexdigest()


def _tanh_scale_grid_mse(
    scales: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    *,
    row_chunk_size: int = 65_536,
    scale_chunk_size: int = 16,
) -> np.ndarray:
    """Score a scale grid with bounded memory even for multi-million-row probes."""

    squared_error_sum = np.zeros(len(scales), dtype=np.float64)
    for row_start in range(0, len(q), row_chunk_size):
        row_stop = min(row_start + row_chunk_size, len(q))
        q_chunk = q[row_start:row_stop]
        z_chunk = z[row_start:row_stop]
        for scale_start in range(0, len(scales), scale_chunk_size):
            scale_stop = min(scale_start + scale_chunk_size, len(scales))
            prediction = np.tanh(
                scales[scale_start:scale_stop, None] * q_chunk[None, :]
            )
            squared_error_sum[scale_start:scale_stop] += np.sum(
                (prediction - z_chunk[None, :]) ** 2, axis=1
            )
    return squared_error_sum / len(q)


def fit_scalar_tanh_value_scale(
    predictions: ReadoutPredictions,
    groups: list[dict[str, np.ndarray]],
    *,
    current_value_scale: float,
    calibration_fraction: float = 0.5,
    split_seed: int = 20260713,
    scale_min: float = 0.125,
    scale_max: float = 8.0,
    scale_count: int = 129,
    reliability_bin_count: int = 10,
) -> ValueScaleFit:
    """Fit the deployed scalar tanh multiplier without game leakage.

    ``train_bc`` regresses the raw scalar head against terminal outcomes while
    search consumes ``tanh(q_raw * value_scale)``.  Ranking checkpoints by raw
    MSE can therefore disagree with the value function MCTS actually sees.
    This helper isolates that mismatch: it splits an already-held-out corpus
    by whole game, chooses one positive scale on the calibration games, and
    reports both the current and fitted transforms on disjoint evaluation
    games.  It intentionally cannot alter the evaluator or promotion receipt.
    """

    if predictions.provenance["requested_readout"] != "scalar":
        raise ValueError("value-scale fitting is only defined for the scalar readout")
    numeric = {
        "current_value_scale": current_value_scale,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "calibration_fraction": calibration_fraction,
    }
    if any(not math.isfinite(float(value)) for value in numeric.values()):
        raise ValueError(f"value-scale fit inputs must be finite: {numeric}")
    if current_value_scale <= 0.0:
        raise ValueError("current_value_scale must be > 0")
    if scale_min <= 0.0 or scale_max <= scale_min:
        raise ValueError("value-scale grid requires 0 < scale_min < scale_max")
    if scale_count < 3:
        raise ValueError("value-scale grid requires at least 3 points")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")

    q = np.asarray(predictions.q, dtype=np.float64).reshape(-1)
    z_parts: list[np.ndarray] = []
    seed_parts: list[np.ndarray] = []
    for index, group in enumerate(groups):
        if "game_seed" not in group:
            raise ValueError(
                f"value-scale fitting requires game_seed in every group; group {index} lacks it"
            )
        z_part = np.asarray(group["z"], dtype=np.float64).reshape(-1)
        seed_part = np.asarray(group["game_seed"], dtype=np.int64).reshape(-1)
        if len(z_part) != len(seed_part):
            raise ValueError(
                f"group {index} z/game_seed row mismatch: {len(z_part)} != {len(seed_part)}"
            )
        z_parts.append(z_part)
        seed_parts.append(seed_part)
    z = np.concatenate(z_parts) if z_parts else np.empty(0, dtype=np.float64)
    game_seed = (
        np.concatenate(seed_parts) if seed_parts else np.empty(0, dtype=np.int64)
    )
    if len(q) != len(z):
        raise ValueError(f"prediction/label row mismatch: {len(q)} != {len(z)}")
    if not len(q) or not np.isfinite(q).all() or not np.isfinite(z).all():
        raise ValueError("value-scale fitting requires non-empty finite q/z rows")

    unique_games = np.unique(game_seed)
    if len(unique_games) < 2:
        raise ValueError("value-scale fitting requires at least two held-out games")
    shuffled = np.random.default_rng(split_seed).permutation(unique_games)
    calibration_game_count = int(round(len(shuffled) * calibration_fraction))
    calibration_game_count = min(max(calibration_game_count, 1), len(shuffled) - 1)
    calibration_games = np.sort(shuffled[:calibration_game_count])
    evaluation_games = np.sort(shuffled[calibration_game_count:])
    calibration_mask = np.isin(game_seed, calibration_games)
    evaluation_mask = ~calibration_mask
    if not calibration_mask.any() or not evaluation_mask.any():
        raise ValueError("game-level value-scale split produced an empty row partition")

    # Always score the configured scale even when it is not an exact point on
    # the geometric grid.  Sorting plus argmin's first-tie behavior selects the
    # smaller scale on exact ties, a conservative choice near saturation.
    scales = np.unique(
        np.concatenate(
            (
                np.geomspace(scale_min, scale_max, num=scale_count),
                np.asarray([current_value_scale], dtype=np.float64),
            )
        )
    )
    calibration_q = q[calibration_mask]
    calibration_z = z[calibration_mask]
    calibration_mse = _tanh_scale_grid_mse(
        scales, calibration_q, calibration_z
    )
    selected_index = int(np.argmin(calibration_mse))
    selected_scale = float(scales[selected_index])

    current_q = np.tanh(q * current_value_scale)
    selected_q = np.tanh(q * selected_scale)
    current_eval_sq = (current_q[evaluation_mask] - z[evaluation_mask]) ** 2
    selected_eval_sq = (selected_q[evaluation_mask] - z[evaluation_mask]) ** 2
    per_game_reductions = []
    for seed in evaluation_games:
        mask = evaluation_mask & (game_seed == seed)
        per_game_reductions.append(
            float(
                np.mean(
                    (current_q[mask] - z[mask]) ** 2
                    - (selected_q[mask] - z[mask]) ** 2
                )
            )
        )

    def _split(game_values: np.ndarray, mask: np.ndarray) -> ValueScaleSplit:
        return ValueScaleSplit(
            game_count=int(len(game_values)),
            row_count=int(mask.sum()),
            game_seed_set_sha256=_game_seed_set_sha256(game_values),
        )

    stats_kwargs = {
        "min_rows": 1,
        "reliability_bin_count": reliability_bin_count,
    }
    return ValueScaleFit(
        schema_version="scalar-tanh-value-scale-fit-v1",
        diagnostic_only=True,
        changes_operator_default=False,
        selection_objective="row_weighted_terminal_outcome_mse_on_calibration_games",
        current_value_scale=float(current_value_scale),
        selected_value_scale=selected_scale,
        scale_grid_min=float(scale_min),
        scale_grid_max=float(scale_max),
        scale_grid_count=int(len(scales)),
        selected_at_grid_boundary=bool(
            selected_scale == float(scales[0]) or selected_scale == float(scales[-1])
        ),
        split_seed=int(split_seed),
        calibration_fraction=float(calibration_fraction),
        calibration=_split(calibration_games, calibration_mask),
        evaluation=_split(evaluation_games, evaluation_mask),
        calibration_current=_calibration_stats(
            current_q[calibration_mask], calibration_z, **stats_kwargs
        ),
        calibration_selected=_calibration_stats(
            selected_q[calibration_mask], calibration_z, **stats_kwargs
        ),
        evaluation_current=_calibration_stats(
            current_q[evaluation_mask], z[evaluation_mask], **stats_kwargs
        ),
        evaluation_selected=_calibration_stats(
            selected_q[evaluation_mask], z[evaluation_mask], **stats_kwargs
        ),
        evaluation_row_weighted_mse_reduction=float(
            current_eval_sq.mean() - selected_eval_sq.mean()
        ),
        evaluation_game_balanced_mse_reduction=float(np.mean(per_game_reductions)),
        promotion_blocked_without_search_h2h=True,
    )


def _slice_by(
    q: np.ndarray,
    z: np.ndarray,
    keys: np.ndarray,
    *,
    min_rows: int,
    categorical_hlgauss_ce: np.ndarray | None = None,
    categorical_terminal_nll: np.ndarray | None = None,
    categorical_truncation_probability: np.ndarray | None = None,
    reliability_bin_count: int = 10,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        out[str(key)] = _calibration_stats(
            q[mask],
            z[mask],
            min_rows=min_rows,
            categorical_hlgauss_ce=(
                categorical_hlgauss_ce[mask]
                if categorical_hlgauss_ce is not None
                else None
            ),
            categorical_terminal_nll=(
                categorical_terminal_nll[mask]
                if categorical_terminal_nll is not None
                else None
            ),
            categorical_truncation_probability=(
                categorical_truncation_probability[mask]
                if categorical_truncation_probability is not None
                else None
            ),
            reliability_bin_count=reliability_bin_count,
        )
    return out


def build_calibration_summary(
    predictions: ReadoutPredictions,
    groups: list[dict[str, np.ndarray]],
    *,
    min_slice_rows: int,
    reliability_bin_count: int,
    deployed_value_scale: float = 1.0,
    deployed_value_squash: str = "tanh",
) -> dict[str, Any]:
    """Build the identical global/phase/forced/legal-width view for either head."""

    q = predictions.q
    z = np.concatenate([g["z"] for g in groups], axis=0)
    phase = np.concatenate([g["phase_label"] for g in groups], axis=0)
    forced = np.concatenate([g["forced"] for g in groups], axis=0)
    legal_count = np.concatenate([g["legal_count"] for g in groups], axis=0)
    if len(q) != len(z):
        raise ValueError(f"prediction/label row mismatch: {len(q)} != {len(z)}")
    legal_bucket = np.array([_legal_bucket(int(c)) for c in legal_count])
    forced_label = np.where(forced, "forced", "unforced")

    common = {
        "min_rows": min_slice_rows,
        "categorical_hlgauss_ce": predictions.categorical_hlgauss_ce,
        "categorical_terminal_nll": predictions.categorical_terminal_nll,
        "categorical_truncation_probability": (
            predictions.categorical_truncation_probability
        ),
        "reliability_bin_count": reliability_bin_count,
    }
    if not math.isfinite(float(deployed_value_scale)) or float(deployed_value_scale) <= 0:
        raise ValueError("deployed_value_scale must be finite and > 0")
    if deployed_value_squash not in {"tanh", "clip"}:
        raise ValueError("deployed_value_squash must be tanh or clip")

    def _view(view_q: np.ndarray) -> dict[str, Any]:
        # Transform comparisons deliberately score only the scalar expectation.
        # Proper categorical CE/NLL is invariant to a post-readout scalar
        # transform and remains reported once in the historical top-level view.
        view_common = {
            "min_rows": min_slice_rows,
            "reliability_bin_count": reliability_bin_count,
        }
        return {
            "global": _calibration_stats(view_q, z, **view_common),
            "by_phase": _slice_by(view_q, z, phase, **view_common),
            "by_forced": _slice_by(view_q, z, forced_label, **view_common),
            "by_legal_count_bucket": _slice_by(
                view_q, z, legal_bucket, **view_common
            ),
        }

    scaled = q.astype(np.float64) * float(deployed_value_scale)
    transformed_views = {
        "raw_training_readout": {
            "formula": "q_raw",
            **_view(q),
        },
        "scalar_tanh": {
            "formula": "tanh(q_raw * value_scale)",
            **_view(np.tanh(scaled)),
        },
        "scalar_clip": {
            "formula": "clip(q_raw * value_scale, -1, 1)",
            **_view(np.clip(scaled, -1.0, 1.0)),
        },
    }
    configured_transform = (
        f"scalar_{deployed_value_squash}"
        if predictions.provenance["requested_readout"] == "scalar"
        else "scalar_clip"
    )

    return {
        "schema_version": "phase-sliced-value-calibration-v2",
        "value_readout": predictions.provenance["requested_readout"],
        "readout_provenance": predictions.provenance,
        "metric_semantics": {
            "brier": "binary win Brier from clipped (selected_expectation + 1) / 2",
            "value_rmse": "RMSE of selected expectation against terminal +/-1 outcome",
            "win_probability_ece": (
                "weighted absolute binary calibration gap over fixed-width bins"
            ),
            "categorical_hlgauss_ce": (
                "cross-entropy against the checkpoint's trainer-matched integrated "
                "Gaussian target; null for scalar"
            ),
            "categorical_terminal_nll": (
                "hard terminal-endpoint NLL from the full categorical softmax, "
                "including optional truncation-class mass; null for scalar"
            ),
        },
        "deployed_readout_diagnostics": {
            "diagnostic_only": True,
            "changes_operator_default": False,
            "value_scale": float(deployed_value_scale),
            "configured_value_squash": deployed_value_squash,
            "configured_effective_transform": configured_transform,
            "categorical_bypasses_scalar_tanh": (
                predictions.provenance["requested_readout"] == "categorical"
            ),
            "views": transformed_views,
        },
        "global": _calibration_stats(q, z, **common),
        "by_phase": _slice_by(q, z, phase, **common),
        "by_forced": _slice_by(q, z, forced_label, **common),
        "by_legal_count_bucket": _slice_by(q, z, legal_bucket, **common),
    }


def build_row_selection_provenance(
    groups: list[dict[str, np.ndarray]],
    *,
    mode: RowSelectionMode,
    validation_fraction: float | None = None,
    validation_seed: int | None = None,
    validation_game_seed_ranges: tuple[tuple[int, int], ...] = (),
    seed_manifest_path: str | None = None,
    seed_manifest_sha256: str | None = None,
    configured_game_seed_count: int | None = None,
    configured_game_seeds: np.ndarray | None = None,
) -> RowSelectionProvenance:
    observed_rows = sum(len(group["z"]) for group in groups)
    observed_seed_arrays = [
        np.asarray(group["game_seed"], dtype=np.int64).reshape(-1)
        for group in groups
        if "game_seed" in group
    ]
    observed_seeds = (
        np.unique(np.concatenate(observed_seed_arrays))
        if observed_seed_arrays
        else np.asarray([], dtype=np.int64)
    )
    configured_seeds = (
        None
        if configured_game_seeds is None
        else np.unique(np.asarray(configured_game_seeds, dtype=np.int64))
    )
    seed_set_sha256 = lambda values: (  # noqa: E731
        "sha256:"
        + hashlib.sha256(
            np.sort(values).astype("<i8", copy=False).tobytes()
        ).hexdigest()
    )
    return RowSelectionProvenance(
        mode=mode,
        held_out_filter_applied=mode != "all_natural_terminal_rows",
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        validation_game_seed_ranges=[
            [int(start), int(end)] for start, end in validation_game_seed_ranges
        ],
        seed_manifest_path=seed_manifest_path,
        seed_manifest_sha256=seed_manifest_sha256,
        configured_game_seed_count=configured_game_seed_count,
        configured_game_seed_set_sha256=(
            None if configured_seeds is None else seed_set_sha256(configured_seeds)
        ),
        observed_game_seed_count=int(len(observed_seeds)),
        observed_game_seed_set_sha256=seed_set_sha256(observed_seeds),
        observed_row_count=int(observed_rows),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        action="append",
        required=True,
        help="repeatable directory searched recursively for *.npz",
    )
    parser.add_argument(
        "--deployed-value-scale",
        type=float,
        default=1.0,
        help="search evaluator value_scale to apply in diagnostic transform views",
    )
    parser.add_argument(
        "--deployed-value-squash",
        choices=("tanh", "clip"),
        default="tanh",
        help="current search transform to mark as configured (does not change it)",
    )
    parser.add_argument(
        "--fit-scalar-tanh-value-scale",
        action="store_true",
        help=(
            "diagnostically fit value_scale on one held-out game subset and score "
            "it on a disjoint held-out subset; never changes the evaluator"
        ),
    )
    parser.add_argument(
        "--value-scale-fit-fraction",
        type=float,
        default=0.5,
        help="fraction of held-out games used to select the diagnostic scale",
    )
    parser.add_argument(
        "--value-scale-fit-seed",
        type=int,
        default=20260713,
        help="deterministic whole-game calibration/evaluation split seed",
    )
    parser.add_argument("--value-scale-fit-min", type=float, default=0.125)
    parser.add_argument("--value-scale-fit-max", type=float, default=8.0)
    parser.add_argument("--value-scale-fit-count", type=int, default=129)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--value-readout",
        choices=("scalar", "categorical"),
        default="scalar",
        help=(
            "Value expectation to calibrate. Scalar is the backward-compatible "
            "default. Categorical requires positive value-training-v1 provenance "
            "and never accepts a config-only cat-head upgrade."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--max-rows-per-shard",
        type=int,
        default=None,
        help="cap each shard before the global cap to spread diagnostics across games",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help=(
            "derive the exact game-level validation seed set using train_bc's "
            "fraction/seed algorithm over all source rows"
        ),
    )
    selection.add_argument(
        "--validation-seed-manifest",
        default=None,
        help="typed value-calibration-validation-seeds-v1 manifest",
    )
    selection.add_argument(
        "--validation-game-seed-ranges",
        default="",
        help="inclusive held-out ranges in train_bc's start:end,start:end syntax",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=17,
        help="RNG seed paired with --validation-fraction (train_bc default: 17)",
    )
    parser.add_argument(
        "--write-validation-seed-manifest",
        default=None,
        help=(
            "persist the seed set derived by --validation-fraction so both "
            "training arms and later calibration can consume identical games"
        ),
    )
    parser.add_argument(
        "--require-held-out",
        action="store_true",
        help="refuse all-rows calibration; use this for go/no-go experiment artifacts",
    )
    parser.add_argument(
        "--min-slice-rows",
        type=int,
        default=30,
        help="minimum rows in a slice before corr(q,z) is reported (else null)",
    )
    parser.add_argument(
        "--reliability-bins",
        type=int,
        default=10,
        help="number of fixed-width win-probability bins used for ECE/reliability",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ranges = parse_validation_game_seed_ranges(args.validation_game_seed_ranges)
    if args.write_validation_seed_manifest and args.validation_fraction is None:
        parser.error("--write-validation-seed-manifest requires --validation-fraction")
    selected_seeds: np.ndarray | None = None
    seed_manifest_sha256: str | None = None
    selection_mode: RowSelectionMode = "all_natural_terminal_rows"
    configured_seed_count: int | None = None
    selected_fraction: float | None = None
    selected_seed: int | None = None
    if args.validation_fraction is not None:
        derived = derive_validation_game_seeds(
            args.shard_dir,
            validation_fraction=args.validation_fraction,
            validation_seed=args.validation_seed,
        )
        selected_seeds = derived.game_seeds
        configured_seed_count = int(len(selected_seeds))
        selected_fraction = float(derived.validation_fraction)
        selected_seed = int(derived.validation_seed)
        selection_mode = "validation_fraction"
        if args.write_validation_seed_manifest:
            seed_manifest_sha256 = write_validation_seed_manifest(
                args.write_validation_seed_manifest,
                derived,
                shard_dir=args.shard_dir,
            )
    elif args.validation_seed_manifest:
        selected_seeds, seed_manifest_sha256 = load_validation_seed_manifest(
            args.validation_seed_manifest
        )
        configured_seed_count = int(len(selected_seeds))
        selection_mode = "validation_seed_manifest"
    elif ranges:
        selection_mode = "validation_game_seed_ranges"
    if args.require_held_out and selection_mode == "all_natural_terminal_rows":
        parser.error(
            "--require-held-out needs --validation-fraction, "
            "--validation-seed-manifest, or --validation-game-seed-ranges"
        )
    if args.fit_scalar_tanh_value_scale:
        if args.value_readout != "scalar":
            parser.error("--fit-scalar-tanh-value-scale requires --value-readout scalar")
        if selection_mode == "all_natural_terminal_rows":
            parser.error(
                "--fit-scalar-tanh-value-scale requires an explicit held-out "
                "validation game selection"
            )

    groups = collect_rows(
        args.shard_dir,
        max_rows=args.max_rows,
        max_rows_per_shard=args.max_rows_per_shard,
        validation_game_seeds=selected_seeds,
        validation_game_seed_ranges=ranges,
    )
    policy = EntityGraphPolicy.load(args.checkpoint, device=args.device)

    predictions = compute_readout(policy, groups, value_readout=args.value_readout)
    summary = build_calibration_summary(
        predictions,
        groups,
        min_slice_rows=args.min_slice_rows,
        reliability_bin_count=args.reliability_bins,
        deployed_value_scale=args.deployed_value_scale,
        deployed_value_squash=args.deployed_value_squash,
    )
    if args.fit_scalar_tanh_value_scale:
        summary["scalar_tanh_value_scale_fit"] = fit_scalar_tanh_value_scale(
            predictions,
            groups,
            current_value_scale=args.deployed_value_scale,
            calibration_fraction=args.value_scale_fit_fraction,
            split_seed=args.value_scale_fit_seed,
            scale_min=args.value_scale_fit_min,
            scale_max=args.value_scale_fit_max,
            scale_count=args.value_scale_fit_count,
            reliability_bin_count=args.reliability_bins,
        )
    summary["checkpoint"] = args.checkpoint
    resolved_shard_dirs = _normalized_shard_dirs(args.shard_dir)
    summary["shard_dir"] = resolved_shard_dirs[0]
    summary["shard_dirs"] = resolved_shard_dirs
    summary["row_selection"] = build_row_selection_provenance(
        groups,
        mode=selection_mode,
        validation_fraction=selected_fraction,
        validation_seed=selected_seed,
        validation_game_seed_ranges=ranges,
        seed_manifest_path=(
            args.validation_seed_manifest or args.write_validation_seed_manifest
        ),
        seed_manifest_sha256=seed_manifest_sha256,
        configured_game_seed_count=configured_seed_count,
        configured_game_seeds=selected_seeds,
    )
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
