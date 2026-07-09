"""Shared helpers for the high-regret restart system (task #64, Go-Exploit/RGSC).

Streaming shard iteration + regret scoring used by `tools/extract_regret_states.py`
and `tools/reconstruct_state.py`. Kept dependency-light (numpy + optional torch/
zstandard) so it runs anywhere a shard lives.

The regret score identifies archived decision states where the agent's own
evaluation diverged from what actually happened (value_surprise), or where the
state is intrinsically decision-rich / historically failure-prone (phase_bonus,
legal_count_bonus), or where search overruled the network prior
(search_prior_disagreement). See `docs/regret_restart_mixing_recipe.md` for how
the resulting states feed restart self-play, and the DAGS (arXiv 2605.14379)
hidden-info caveat handling.

Value-scale convention (must match `tools/train_bc.py`'s `_value_targets` and
`neural_rust_mcts.EntityGraphRustEvaluator`):
  * outcome z = +1 if winner == acting player, -1 if winner != acting player,
    for CLEAN TERMINAL rows only (winner != "" and not truncated).
  * searched shards store per-legal-action Q (`target_scores`) already on that
    [-1, 1] outcome scale (`target_score_source == "gumbel_mcts_visit_q"`).
  * raw shards carry NO Q (`target_scores_mask` all-False); v(s) comes from a
    fresh value-head pass over the STORED entity tokens (no Rust re-featurize),
    squashed with the same tanh/clip the searched Q went through so the two
    corpora's value_surprise live on one scale.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Rows whose acting player is BLUE/RED index into PLAYER_NAMES order for the
# final-VP arrays; mirrored from gumbel_self_play.PLAYER_NAMES.
PLAYER_NAMES = ("BLUE", "RED", "ORANGE", "WHITE")

# Entity-token keys the value-head forward needs (subset of gumbel_self_play
# ENTITY_KEYS); everything the model's `forward` reads.
_VALUE_PASS_ENTITY_KEYS = (
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

# Default phase-bonus table. Opening placement is scored highest per the pilot
# finding that placement blowouts were 74.6% of search losses; robber/dev
# (chance-heavy) states next; ordinary build/roll turns baseline. Keys are
# matched by case-insensitive substring against the row's `phase`
# (current_prompt), longest-key-first so "BUILD_INITIAL_SETTLEMENT" wins over
# a hypothetical "BUILD".
DEFAULT_PHASE_BONUS: dict[str, float] = {
    "BUILD_INITIAL_SETTLEMENT": 1.0,
    "BUILD_INITIAL_ROAD": 0.7,
    "MOVE_ROBBER": 0.5,
    "DISCARD": 0.4,
    "PLAY_KNIGHT_CARD": 0.35,
    "BUY_DEVELOPMENT_CARD": 0.3,
    "ROLL": 0.15,
}


@dataclass(frozen=True, slots=True)
class RegretConfig:
    """Weights for the additive regret score. All components are normalised to
    roughly [0, 1] before weighting so the weights are directly comparable."""

    value_surprise_weight: float = 1.0
    phase_bonus_weight: float = 0.4
    legal_count_weight: float = 0.2
    kl_disagreement_weight: float = 0.5
    argmax_mismatch_lost_weight: float = 0.4
    # legal_count_bonus normaliser: legal counts are divided by this and clipped
    # to [0, 1]. 54 is the max legal width (placement) in the 2p schema.
    legal_count_norm: float = 54.0
    # KL is divided by this and clipped to [0, 1] for the additive score; a KL
    # of ~kl_norm nats counts as a "full" disagreement unit.
    kl_norm: float = 2.0
    phase_bonus: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PHASE_BONUS))
    # Exclude forced (<=1 legal action) rows from candidacy: they carry no
    # decision to restart from and no search/policy signal.
    include_forced: bool = False


# --------------------------------------------------------------------------- #
# Shard discovery + streaming load
# --------------------------------------------------------------------------- #
def discover_shards(roots: list[Path]) -> list[Path]:
    """All shard files under the given roots, sorted for reproducibility.

    Accepts a shard file, a worker/run directory, or a corpus root. Matches
    both plain `.npz` and zstd-compressed `.npz.zst` shards.
    """
    shards: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            if root.name.endswith((".npz", ".npz.zst")):
                shards.append(root)
            continue
        for pattern in ("*.npz", "*.npz.zst"):
            shards.extend(root.rglob(pattern))
    # rglob("*.npz.zst") never matches "*.npz", so no dedup needed, but sort +
    # unique defensively (a root that is also a file, symlinks, etc.).
    return sorted(set(shards))


def load_shard(path: Path) -> dict[str, np.ndarray]:
    """Load one shard fully into memory (one shard at a time -- never the corpus).

    Transparently decompresses `.npz.zst`. `allow_pickle=True` is required
    because the writer stores 0-d object/str scalars in a couple of columns.
    """
    path = Path(path)
    if path.name.endswith(".npz.zst"):
        import io

        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with path.open("rb") as handle:
            raw = dctx.stream_reader(handle).read()
        with np.load(io.BytesIO(raw), allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def iter_shards(roots: list[Path], *, sample_frac: float = 1.0, seed: int = 0) -> Iterator[tuple[Path, dict[str, np.ndarray]]]:
    """Yield (path, shard_dict) one shard at a time (memory-streaming).

    `sample_frac < 1.0` keeps a deterministic hash-based subset of shards (whole
    shards, so per-game row contiguity within a kept shard is preserved).
    """
    shards = discover_shards(roots)
    for path in shards:
        if sample_frac < 1.0:
            # Deterministic per-path selection independent of iteration order.
            h = (hash((str(path), int(seed))) & 0xFFFFFFFF) / 0xFFFFFFFF
            if h >= sample_frac:
                continue
        yield path, load_shard(path)


# --------------------------------------------------------------------------- #
# Per-shard field helpers (all vectorised over the shard's rows)
# --------------------------------------------------------------------------- #
def _as_str_array(value: Any, n: int) -> np.ndarray:
    return np.asarray(value).astype(str).reshape(-1)[:n]


def legal_counts(shard: dict[str, np.ndarray]) -> np.ndarray:
    """Number of legal actions per row (padded entries are -1)."""
    if "legal_action_mask" in shard:
        return np.asarray(shard["legal_action_mask"]).sum(axis=1).astype(np.int32)
    lids = np.asarray(shard["legal_action_ids"])
    return (lids >= 0).sum(axis=1).astype(np.int32)


def outcome_z(shard: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(z, has_clean_outcome) per row on the [-1, +1] scale.

    z = +1 if winner == acting player, -1 otherwise, defined only for clean
    terminal rows (winner != "" and not truncated) -- exactly
    train_bc._value_targets's `has_outcome_np`. Rows without a clean outcome get
    z = 0 and has_clean_outcome = False.
    """
    n = int(np.asarray(shard["action_taken"]).shape[0])
    winners = _as_str_array(shard.get("winner", np.full(n, "")), n)
    players = _as_str_array(shard.get("player", np.full(n, "")), n)
    truncated = np.asarray(
        shard.get("truncated", np.zeros(n, dtype=bool)), dtype=bool
    ).reshape(-1)[:n]
    has_clean = (winners != "") & (~truncated)
    z = np.zeros(n, dtype=np.float32)
    z[has_clean & (winners == players)] = 1.0
    z[has_clean & (winners != players)] = -1.0
    return z, has_clean


def taken_column(shard: dict[str, np.ndarray]) -> np.ndarray:
    """Column index of the taken action within each row's legal list, or -1.

    `action_taken` is a policy-catalog id; `legal_action_ids` holds the policy
    ids of every legal action in the same column order the per-action arrays
    (`target_scores`, `target_policy`, ...) use. Returns the first matching
    column (the mapping is injective within a single state's legal set).
    """
    lids = np.asarray(shard["legal_action_ids"])
    taken = np.asarray(shard["action_taken"]).reshape(-1, 1)
    match = lids == taken
    has = match.any(axis=1)
    col = np.argmax(match, axis=1)  # first True, or 0 if none
    col = np.where(has, col, -1)
    return col.astype(np.int64)


def q_of_taken(shard: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(q_taken, has_q) per row from searched shards' `target_scores`.

    Returns has_q = False for raw shards (all-False target_scores_mask) or any
    row where the taken action's Q is masked/NaN.
    """
    n = int(np.asarray(shard["action_taken"]).shape[0])
    if "target_scores" not in shard:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)
    scores = np.asarray(shard["target_scores"], dtype=np.float32)
    mask = np.asarray(shard.get("target_scores_mask", np.isfinite(scores)), dtype=bool)
    col = taken_column(shard)
    rows = np.arange(n)
    valid_col = col >= 0
    safe_col = np.where(valid_col, col, 0)
    q = scores[rows, safe_col]
    m = mask[rows, safe_col] if mask.shape == scores.shape else np.isfinite(q)
    has_q = valid_col & m & np.isfinite(q)
    q = np.where(has_q, q, 0.0).astype(np.float32)
    return q, has_q


def kl_target_prior(shard: dict[str, np.ndarray], *, eps: float = 1e-6) -> np.ndarray:
    """KL(target_policy || prior_policy) per row, over the target's support.

    Zero for raw shards (target == prior by construction). Prior is clamped to
    `eps` to keep KL finite where fp16 flushed a small prior to 0 under a
    positive target. Padded columns (target == 0) contribute nothing.
    """
    n = int(np.asarray(shard["action_taken"]).shape[0])
    if "target_policy" not in shard or "prior_policy" not in shard:
        return np.zeros(n, dtype=np.float32)
    target = np.asarray(shard["target_policy"], dtype=np.float64)
    prior = np.asarray(shard["prior_policy"], dtype=np.float64)
    support = target > 0.0
    prior_c = np.clip(prior, eps, None)
    ratio = np.where(support, target / prior_c, 1.0)
    terms = np.where(support, target * np.log(ratio), 0.0)
    kl = terms.sum(axis=1)
    return np.clip(kl, 0.0, None).astype(np.float32)


def argmax_mismatch(shard: dict[str, np.ndarray]) -> np.ndarray:
    """Per row: True if argmax(target_policy) != argmax(prior_policy).

    Search picked a different action than the raw network prior would have.
    Always False for raw shards (target == prior).
    """
    n = int(np.asarray(shard["action_taken"]).shape[0])
    if "target_policy" not in shard or "prior_policy" not in shard:
        return np.zeros(n, dtype=bool)
    target = np.asarray(shard["target_policy"], dtype=np.float32)
    prior = np.asarray(shard["prior_policy"], dtype=np.float32)
    return np.argmax(target, axis=1) != np.argmax(prior, axis=1)


def phase_bonus_values(shard: dict[str, np.ndarray], table: dict[str, float]) -> np.ndarray:
    """Per-row phase bonus via longest-substring match against `phase`."""
    n = int(np.asarray(shard["action_taken"]).shape[0])
    phases = _as_str_array(shard.get("phase", np.full(n, "")), n)
    keys = sorted(table.keys(), key=len, reverse=True)
    out = np.zeros(n, dtype=np.float32)
    uniq, inverse = np.unique(phases, return_inverse=True)
    lut = np.zeros(len(uniq), dtype=np.float32)
    for idx, phase in enumerate(uniq):
        up = phase.upper()
        for key in keys:
            if key.upper() in up:
                lut[idx] = float(table[key])
                break
    return lut[inverse].astype(np.float32)


# --------------------------------------------------------------------------- #
# Value-head pass over stored features (raw shards)
# --------------------------------------------------------------------------- #
class StoredFeatureValuer:
    """Runs a checkpoint's value head over STORED entity tokens, in batches.

    Bypasses the Rust engine entirely: raw shards already persist the exact
    entity-token features the network consumes, so v(s) is a pure GPU forward
    over `shard[...]` slices. Applies the same value_scale + squash + clip the
    searched Q went through (default tanh, scale 1.0 -- the config the raw
    corpus was generated with) so searched-Q and raw-v value_surprise share one
    scale.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cuda",
        value_scale: float = 1.0,
        value_squash: str = "tanh",
        batch_size: int = 4096,
    ) -> None:
        import torch

        from catan_zero.rl.entity_token_policy import EntityGraphPolicy

        self._torch = torch
        self.policy = EntityGraphPolicy.load(checkpoint, device=device)
        self.policy.model.eval()
        self.device = device
        self.value_scale = float(value_scale)
        self.value_squash = str(value_squash)
        self.batch_size = int(batch_size)

    def _squash(self, raw: np.ndarray) -> np.ndarray:
        scaled = raw.astype(np.float32) * self.value_scale
        if self.value_squash == "tanh":
            squashed = np.tanh(scaled)
        elif self.value_squash == "clip":
            squashed = scaled
        else:
            raise ValueError(f"unknown value_squash: {self.value_squash!r}")
        return np.clip(squashed, -1.0, 1.0).astype(np.float32)

    def values(self, shard: dict[str, np.ndarray]) -> np.ndarray:
        """v(s) per row, actor-relative, on the [-1, 1] scale (see class doc)."""
        torch = self._torch
        n = int(np.asarray(shard["action_taken"]).shape[0])
        legal_ids = np.asarray(shard["legal_action_ids"])
        context = np.asarray(shard["legal_action_context"])
        out = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                entity = {
                    key: shard[key][start:stop]
                    for key in _VALUE_PASS_ENTITY_KEYS
                    if key in shard
                }
                outputs = self.policy.forward_legal_np(
                    entity,
                    legal_ids[start:stop],
                    context[start:stop],
                    return_q=False,
                )
                raw = outputs["value"].detach().float().cpu().numpy().reshape(-1)
                out[start:stop] = raw
        return self._squash(out)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_shard(
    shard: dict[str, np.ndarray],
    config: RegretConfig,
    *,
    values: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Score every row of one shard. Returns per-row arrays.

    `values` (optional) is a precomputed v(s) array for raw shards from
    `StoredFeatureValuer`; when provided it supplies value_surprise for rows
    that have no searched Q. Searched Q takes precedence when both exist.

    Returned dict keys:
      regret_score, value_surprise, phase_bonus, legal_count_bonus,
      kl_disagreement, argmax_mismatch_lost, is_forced, is_candidate,
      legal_count, z, has_value_surprise
    """
    n = int(np.asarray(shard["action_taken"]).shape[0])
    z, has_clean = outcome_z(shard)
    q_taken, has_q = q_of_taken(shard)

    # value estimate = searched Q where available, else raw v(s).
    value_est = np.where(has_q, q_taken, 0.0).astype(np.float32)
    has_value = has_q.copy()
    if values is not None:
        use_v = (~has_q) & np.isfinite(values)
        value_est = np.where(use_v, values, value_est).astype(np.float32)
        has_value = has_value | use_v

    has_value_surprise = has_value & has_clean
    value_surprise = np.where(
        has_value_surprise, np.abs(value_est - z), 0.0
    ).astype(np.float32)

    lc = legal_counts(shard)
    legal_count_bonus = np.clip(
        lc.astype(np.float32) / max(config.legal_count_norm, 1.0), 0.0, 1.0
    )
    phase_bonus = phase_bonus_values(shard, config.phase_bonus)
    kl = kl_target_prior(shard)
    kl_disagreement = np.clip(kl / max(config.kl_norm, 1e-9), 0.0, 1.0)
    lost = has_clean & (z < 0.0)
    argmax_mismatch_lost = (argmax_mismatch(shard) & lost).astype(np.float32)

    if "is_forced" in shard:
        is_forced = np.asarray(shard["is_forced"], dtype=bool).reshape(-1)[:n]
    else:
        is_forced = lc <= 1

    regret = (
        config.value_surprise_weight * value_surprise
        + config.phase_bonus_weight * phase_bonus
        + config.legal_count_weight * legal_count_bonus
        + config.kl_disagreement_weight * kl_disagreement
        + config.argmax_mismatch_lost_weight * argmax_mismatch_lost
    ).astype(np.float32)

    is_candidate = np.ones(n, dtype=bool)
    if not config.include_forced:
        is_candidate &= ~is_forced

    return {
        "regret_score": regret,
        "value_surprise": value_surprise,
        "phase_bonus": phase_bonus,
        "legal_count_bonus": legal_count_bonus,
        "kl_disagreement": kl_disagreement,
        "argmax_mismatch_lost": argmax_mismatch_lost,
        "is_forced": is_forced,
        "is_candidate": is_candidate,
        "legal_count": lc,
        "z": z,
        "has_value_surprise": has_value_surprise,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)
