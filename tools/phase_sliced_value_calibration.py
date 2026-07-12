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
from typing import Any, Literal, TypedDict, cast

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
    observed_game_seed_count: int
    observed_row_count: int


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


def _iter_shards(shard_dir: str) -> list[str]:
    root = Path(shard_dir)
    shards = sorted(str(p) for p in root.rglob("*.npz"))
    if not shards:
        raise SystemExit(f"no .npz shards found under {shard_dir}")
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
    shard_dir: str, *, validation_fraction: float, validation_seed: int
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
    shard_dir: str,
    *,
    max_rows: int | None = None,
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
    }
    if n < min_rows or n_win == 0 or n_loss == 0 or float(np.std(q)) == 0.0:
        # corr is undefined without both classes / enough rows.
        stats["corr_q_z"] = None
    else:
        stats["corr_q_z"] = float(np.corrcoef(q, z)[0, 1])
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
) -> RowSelectionProvenance:
    observed_rows = sum(len(group["z"]) for group in groups)
    observed_seed_arrays = [
        np.asarray(group["game_seed"], dtype=np.int64).reshape(-1)
        for group in groups
        if "game_seed" in group
    ]
    observed_seed_count = (
        int(len(np.unique(np.concatenate(observed_seed_arrays))))
        if observed_seed_arrays
        else 0
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
        observed_game_seed_count=observed_seed_count,
        observed_row_count=int(observed_rows),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir", required=True, help="dir searched recursively for *.npz"
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

    groups = collect_rows(
        args.shard_dir,
        max_rows=args.max_rows,
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
    summary["checkpoint"] = args.checkpoint
    summary["shard_dir"] = args.shard_dir
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
    )
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
