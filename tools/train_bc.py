# ruff: noqa: E402
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from datetime import timedelta
import hashlib
import io
import json
import math
import operator
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

# A script path does not bind Python imports to the same checkout.  In particular,
# ``python /new/checkout/tools/train_bc.py`` can otherwise import ``catan_zero``
# from an older editable install or an ambient PYTHONPATH.  Put this checkout's
# source tree first *before the first project import*.  The assertion below also
# refuses execution when a caller has already imported a different checkout into
# this interpreter (inserting a path cannot repair an existing sys.modules entry).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = (_REPO_ROOT / "src").resolve(strict=True)
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _REPO_SRC]
sys.path.insert(0, str(_REPO_SRC))

import numpy as np

from catan_zero.rl.config_cli import add_config_flags, apply_config_file, resolve_config
from catan_zero.rl.aux_subgoal_targets import AUX_TARGET_KEYS
from catan_zero.rl.pipeline_configs import TrainConfig
from catan_zero.rl.torch_ppo import build_action_feature_table, create_ppo_policy
from catan_zero.rl.xdim_lite_policy import (
    XDimGraphPolicy,
    XDimLitePolicy,
    _array_sha256,
    normalize_observations,
)
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl import optim_state as _checkout_optim_state


CHECKOUT_RUNTIME_BINDING_SCHEMA = "train-bc-checkout-runtime-v1"


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_checkout_runtime_binding() -> dict[str, object]:
    """Refuse a trainer whose script and imported package come from different trees."""

    if _checkout_optim_state is not sys.modules.get("catan_zero.rl.optim_state"):
        raise RuntimeError("checkout optimizer module identity changed after import")

    modules: dict[str, dict[str, str]] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "catan_zero" and not name.startswith("catan_zero."):
            continue
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        module_path = Path(raw_path).resolve(strict=True)
        if not _path_within(module_path, _REPO_SRC):
            raise RuntimeError(
                "checkout/runtime package skew: "
                f"{name} resolved to {module_path}, but trainer {Path(__file__).resolve()} "
                f"requires every catan_zero module under {_REPO_SRC}"
            )
        modules[name] = {
            "path": str(module_path),
            "sha256": _sha256_existing_file(module_path),
        }
    required = {
        "catan_zero",
        "catan_zero.rl.optim_state",
        "catan_zero.rl.entity_token_policy",
    }
    missing = required - modules.keys()
    if missing:
        raise RuntimeError(
            f"checkout runtime binding cannot identify required modules: {sorted(missing)}"
        )
    trainer = Path(__file__).resolve(strict=True)
    binding: dict[str, object] = {
        "schema_version": CHECKOUT_RUNTIME_BINDING_SCHEMA,
        "repo_root": str(_REPO_ROOT),
        "source_root": str(_REPO_SRC),
        "trainer": str(trainer),
        "trainer_sha256": _sha256_existing_file(trainer),
        "modules": modules,
    }
    binding["binding_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return binding

# Make the sibling ``tools/`` modules importable (factory_common) whether this module is run
# as a script (``python tools/train_bc.py``) or imported as a package submodule
# (``from tools.train_bc import ...``, e.g. from tests) -- mirrors the same bootstrap already
# used by ``tools/ppo_distributed_learner.py``.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from factory_common import parse_track, write_json  # noqa: E402
import launcher_guards  # noqa: E402
from mixed_memmap_corpus import ConcatMemmapCorpus  # noqa: E402


# Public-observation masking of hidden player info (task #72). Canonical
# implementation lives in catan_zero.rl.entity_token_features (branch
# f72-public-observation, now merged); import it as the single source of truth.
from catan_zero.rl.entity_token_features import (  # noqa: E402
    mask_player_tokens_public as _mask_player_tokens_public,
)


# Set once from --mask-hidden-info in main(); read by _entity_batch so both the
# train and eval decode paths mask identically. The corpus stays UNMASKED on
# disk -- one corpus serves both masked and unmasked training regimes.
_MASK_HIDDEN_INFO_PLAYER_TOKENS = False

TARGET_INFORMATION_REGIME_PUBLIC = "public_conservation_pimc_v1"
TARGET_INFORMATION_REGIME_UNKNOWN = "unknown"

MEMMAP_PAYLOAD_INVENTORY_SCHEMA = "memmap-payload-inventory-v1"
MEMMAP_PAYLOAD_AUTH_CACHE_SCHEMA = "memmap-payload-auth-cache-v1"
# Bump whenever the inventory validation semantics or cache identity changes.
MEMMAP_PAYLOAD_AUTH_VALIDATOR_VERSION = 1
A1_REQUIRED_LEARNER_CODE_SUFFIXES = {
    "configs/guards/train_bc.json",
    "tools/factory_common.py",
    "tools/launcher_guards.py",
    "tools/prelaunch_guard.py",
    "tools/train_bc.py",
    "tools/build_memmap_corpus.py",
    "src/catan_zero/rl/config_cli.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
    "src/catan_zero/rl/pipeline_configs.py",
    "src/catan_zero/rl/aux_subgoal_targets.py",
    "src/catan_zero/rl/config_serialization.py",
    "src/catan_zero/rl/hex_symmetry.py",
    "src/catan_zero/rl/multiagent_env.py",
    "src/catan_zero/rl/optim_state.py",
    "src/catan_zero/rl/torch_ppo.py",
    "src/catan_zero/rl/xdim_lite_policy.py",
}
A1_REQUIRED_RUNTIME_CODE_SUFFIXES = (
    A1_REQUIRED_LEARNER_CODE_SUFFIXES
    | {
        "configs/guards/generate_gumbel_selfplay_data.json",
        "tools/generate_gumbel_selfplay_data.py",
        "tools/opponent_mix_registry.py",
        "src/catan_zero/rl/gumbel_self_play.py",
        "src/catan_zero/rl/flywheel/opponent_mix.py",
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/action_mask.py",
        "src/catan_zero/search/gumbel_chance_mcts.py",
        "src/catan_zero/search/neural_rust_mcts.py",
        "src/catan_zero/search/rust_mcts.py",
    }
)

DUAL_ARM_SELECTED_GAMES_SCHEMA = "a1-dual-arm-selected-training-games-v1"
DUAL_ARM_AUDIT_SCHEMAS = {
    "a1-dual-arm-post-wave-audit-v1",
    "a1-dual-arm-derived-post-wave-audit-v1",
}
DUAL_ARM_SUBSET_COUNTS = {
    ("n256", "full-56k"): 56_000,
    ("n128", "matched-56k"): 56_000,
    ("n128", "compute-112k"): 112_000,
    ("n128", "full-140k"): 140_000,
}
DUAL_ARM_SUBSET_CATEGORY_COUNTS = {
    ("n256", "full-56k"): {
        "current_producer": 44_800,
        "recent_history": 8_400,
        "hard_negative": 2_800,
    },
    ("n128", "matched-56k"): {
        "current_producer": 44_800,
        "recent_history": 8_400,
        "hard_negative": 2_800,
    },
    ("n128", "compute-112k"): {
        "current_producer": 89_600,
        "recent_history": 16_800,
        "hard_negative": 5_600,
    },
    ("n128", "full-140k"): {
        "current_producer": 112_000,
        "recent_history": 21_000,
        "hard_negative": 7_000,
    },
}


ENTITY_BATCH_KEYS = (
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

ENTITY_FIELD_DTYPES = {
    "hex_tokens": np.float16,
    "hex_vertex_ids": np.int16,
    "hex_edge_ids": np.int16,
    "vertex_tokens": np.float16,
    "edge_tokens": np.float16,
    "edge_vertex_ids": np.int16,
    "player_tokens": np.float16,
    "global_tokens": np.float16,
    "legal_action_tokens": np.float16,
    "legal_action_target_ids": np.int16,
    "event_tokens": np.float16,
    "event_target_ids": np.int16,
    "hex_mask": np.bool_,
    "vertex_mask": np.bool_,
    "edge_mask": np.bool_,
    "player_mask": np.bool_,
    "legal_action_mask": np.bool_,
    "event_mask": np.bool_,
}

AUX_SUBGOAL_FIELD_DTYPES = {
    "aux_longest_road": np.float32,
    "aux_largest_army": np.float32,
    "aux_vp_in_n": np.float32,
    "aux_next_settlement": np.int16,
    "aux_robber_target": np.int16,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train behavior cloning from teacher shards.")
    parser.add_argument(
        "--arch",
        choices=("candidate", "xdim_lite", "xdim_graph", "entity_graph"),
        default="candidate",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--data-format",
        choices=("npz", "memmap"),
        default="npz",
        help=(
            "npz (default): load and pad the whole corpus in host RAM via "
            "load_teacher_data. memmap: stream a flat corpus built by "
            "tools/build_memmap_corpus.py, materialising only per-batch rows "
            "for the large token/obs columns (removes the half-host-RAM ceiling "
            "on very large corpora). --data must point at the converted corpus "
            "directory (with corpus_meta.json)."
        ),
    )
    parser.add_argument(
        "--data-loader-workers",
        type=int,
        default=0,
        help=(
            "Background prefetch threads for --data-format memmap. 0 (default) "
            "reconstructs each batch synchronously in the train and validation "
            "loops. >0 overlaps per-batch reconstruction with GPU compute to "
            "recover the streaming throughput cost; ignored for --data-format npz "
            "(batches are cheap views)."
        ),
    )
    parser.add_argument(
        "--data-loader-prefetch",
        type=int,
        default=2,
        help="Batches to prefetch ahead when --data-loader-workers>0.",
    )
    parser.add_argument(
        "--mask-hidden-info",
        action="store_true",
        help=(
            "Public-observation training (hidden-info leak fix, f72): zero every "
            "OPPONENT player-token's hidden slots (resource-hand composition, "
            "unplayed dev-card identities, actual VP) at load time via "
            "catan_zero.rl.entity_token_features.mask_player_tokens_public, keeping "
            "public counts/VP and the actor's own hand. Makes the banked (omniscient) "
            "corpus trainable on public-only inputs WITHOUT regeneration; pair with "
            "EntityGraphRustEvaluatorConfig.public_observation=True at inference."
        ),
    )
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument(
        "--graph-history-features",
        action="store_true",
        help=(
            "Build the training environment with the graph/history observation "
            "suffix. If omitted, train_bc.py auto-detects this when loaded shard "
            "observation width matches the graph-history schema."
        ),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help=(
            "Hard cap on optimizer steps, counted globally across epochs. With "
            "--grad-accum-steps 1 (default) one optimizer step is one batch; with "
            "--grad-accum-steps N one optimizer step is N micro-batches. 0 (default) "
            "disables the cap and --epochs alone governs training length. When the "
            "cap is hit mid-epoch, the batch loop stops but the normal end-of-epoch "
            "bookkeeping (validation, per-epoch checkpoint) still runs for the "
            "partial epoch, then training stops -- the final checkpoint save and "
            "report.json are written exactly as they would be at natural completion."
        ),
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "C1 multi-GPU big-net path: accumulate gradients over N consecutive "
            "micro-batches, then take one optimizer step (effective batch = "
            "--batch-size * N * world_size). Each micro-batch's loss is divided by "
            "N before backward and grads accumulate, so N micro-batches of B rows "
            "approximate one batch of N*B rows (exact when per-micro-batch weight "
            "sums are equal, the standard grad-accum semantics). Under DDP the "
            "non-stepping micro-batches run in model.no_sync() so gradients are "
            "all-reduced once per optimizer step, not once per micro-batch. "
            "--max-steps and the LR schedule count OPTIMIZER steps, not micro-batches. "
            "N=1 (default) is byte-identical to the pre-C1 single-step-per-batch path. "
            "Only supported for --arch entity_graph/xdim_graph (the _train_xdim_batch "
            "trainer); other archs must use N=1."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.05,
        help="Held-out fraction split by game_seed for validation diagnostics.",
    )
    parser.add_argument(
        "--validation-max-samples",
        type=int,
        default=200_000,
        help="Maximum held-out samples evaluated per epoch; 0 disables the cap.",
    )
    parser.add_argument("--validation-seed", type=int, default=17)
    parser.add_argument(
        "--validation-game-seed-ranges",
        default="",
        help="Comma-separated start:end (inclusive) game_seed ranges forming an EXPLICIT, "
        "deterministic held-out validation set -- overrides --validation-fraction's random "
        "game_seed permutation entirely when non-empty (task #65 value-head-repair-v2 "
        "protocol: matches a holdout.json's documented ranges exactly, e.g. "
        "'5006335:5006667,6406335:6406667'). These games are excluded from training "
        "gradients and reported as validation telemetry each epoch -- same mechanism as "
        "--validation-fraction, just with a precomputed selection instead of a random one. "
        "--validation-max-samples still applies (subsamples the explicit set if it exceeds "
        "the cap).",
    )
    parser.add_argument(
        "--validation-game-seed-manifest",
        default="",
        help=(
            "Immutable train-validation-game-seeds-v1 manifest emitted by the A1 "
            "post-wave audit. When set, these exact game seeds form the validation "
            "split before the first optimizer step; the trainer rejects manifest "
            "digest/config drift, missing seeds, --validation-game-seed-ranges, a "
            "nonzero --validation-max-samples cap, or a memmap corpus bound to a "
            "different A1 contract."
        ),
    )
    parser.add_argument(
        "--validation-game-sentinel-manifest",
        default="",
        help=(
            "Optional immutable train-validation-game-sentinel-v1 receipt for an "
            "authenticated memmap composite. The selected whole games are evaluated, "
            "while the complete component holdouts remain excluded from training. "
            "Requires --validation-max-samples 0; derive with "
            "tools/derive_validation_game_sentinel.py."
        ),
    )
    parser.add_argument(
        "--allow-missing-game-seed-validation-split",
        action="store_true",
        help="CAT-52: split_train_validation_indices refuses, by default, to build a "
        "validation split from a corpus with no 'game_seed' column, because the old "
        "silent default (treating each row as its own 'game') reproduces the round-11 "
        "val-leak mechanism (a de facto ROW-LEVEL split). Pass this flag to explicitly "
        "opt into that row-level behavior (e.g. for a synthetic/legacy corpus with no "
        "game grouping at all) -- a loud warning is still printed when it fires.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help=(
            "Model width. Defaults to 768 for xdim_graph (~33.5M params with "
            "4 graph layers) and 512 for other architectures."
        ),
    )
    parser.add_argument(
        "--graph-tokens",
        type=int,
        default=32,
        help="Observation token count for --arch xdim_graph.",
    )
    parser.add_argument(
        "--graph-layers",
        type=int,
        default=4,
        help="Token message-passing layers for --arch xdim_graph/entity_graph.",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=8,
        help="Attention heads for --arch xdim_graph/entity_graph.",
    )
    parser.add_argument(
        "--graph-dropout",
        type=float,
        default=0.05,
        help="Dropout probability for --arch xdim_graph/entity_graph blocks.",
    )
    parser.add_argument(
        "--entity-state-trunk",
        choices=("transformer", "rrt", "resrgcn"),
        default="transformer",
        help=(
            "EntityGraph state trunk. transformer preserves the incumbent; rrt "
            "and resrgcn are explicit topology-aware R&D arms."
        ),
    )
    parser.add_argument(
        "--relational-block-pattern",
        default="",
        help="Explicit R/T pattern for --entity-state-trunk rrt (for example RRTRRTRRT).",
    )
    parser.add_argument(
        "--relational-ff-size",
        type=int,
        default=0,
        help="Relational trunk FF width; 0 selects the width-scaled architecture default.",
    )
    parser.add_argument(
        "--relational-bases",
        type=int,
        default=4,
        help="Number of basis transforms in each ResRGCN block.",
    )
    parser.add_argument(
        "--relational-action-cross-layers",
        type=int,
        default=1,
        help="From-scratch action-to-board cross-attention blocks for relational arms.",
    )
    parser.add_argument(
        "--relational-edge-policy-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether topology-aware trunks also enable CAT-97's direct target-token "
            "policy logit. Default true preserves existing relational models; disable "
            "for a causal relational/gather/cross-attention-only probe."
        ),
    )
    parser.add_argument(
        "--latent-deliberation-steps",
        type=int,
        default=0,
        help="Fixed shared-weight latent reasoning steps; 0 disables E3.",
    )
    parser.add_argument(
        "--latent-deliberation-slots",
        type=int,
        default=8,
        help="Learned plan-token count for fixed-K latent deliberation.",
    )
    parser.add_argument(
        "--moe-routed-experts",
        type=int,
        default=0,
        help="Routed experts in global RRT FFNs; 0 disables sparse MoE.",
    )
    parser.add_argument(
        "--moe-top-k",
        type=int,
        default=2,
        help="Experts dispatched per live token when sparse MoE is enabled.",
    )
    parser.add_argument(
        "--moe-expert-ff-size",
        type=int,
        default=0,
        help="SwiGLU inner width per shared/routed expert; 0 selects the scaled default.",
    )
    parser.add_argument(
        "--moe-balance-loss-weight",
        type=float,
        default=0.01,
        help=(
            "Weight on the sparse-router load-balance objective. Inert when "
            "--moe-routed-experts=0."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help=(
            "Linearly ramp the learning rate from 0 to --lr over this many batches at the "
            "start of training, then hold at --lr. Checkpoints do not persist optimizer "
            "state, so every resume restarts Adam's moment estimates from zero; a short "
            "ramp (e.g. 500-1000) protects a repair run from that fresh-Adam transient."
        ),
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("flat", "cosine", "linear"),
        default="flat",
        help=(
            "AUDIT FIX (LR decay): schedule applied AFTER --lr-warmup-steps completes. "
            "'flat' (default) is a strict no-op matching pre-fix behavior -- LR holds at "
            "--lr for the rest of training, exactly as before this flag existed. 'cosine' "
            "and 'linear' decay smoothly from --lr down to 0 over the remaining steps, "
            "reaching 0 at --max-steps if set, else at epochs x batches-per-epoch."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adam",
        help="Optimizer for BC. Use adamw for production 35M transformer-style runs.",
    )
    parser.add_argument(
        "--value-lr-mult",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the LEARNING RATE of the value_head/final_vp_head/"
            "value_uncertainty_head submodules only (whichever are present on the "
            "model), via a second optimizer param group at --lr * this multiplier; "
            "every other parameter keeps training at the base --lr. 1.0 (default) is "
            "a pure no-op -- a single param group at --lr, bit-identical to every "
            "training run before this flag existed. CAT-12/roadmap R6 (MuZero-"
            "Reanalyse-style value-head LR decoupling, value-head LR ~=0.3x torso): "
            "pass 0.3 to reproduce it. --lr-warmup-steps/--lr-schedule still apply, "
            "scaled per group. Requires --arch entity_graph/xdim_lite/xdim_graph (the "
            "model must expose at least one of value_head/final_vp_head/"
            "value_uncertainty_head as a named submodule) -- SystemExit otherwise."
        ),
    )
    parser.add_argument(
        "--action-module-lr-mult",
        type=float,
        default=1.0,
        help=(
            "LR multiplier for the opt-in action-target gather/cross-attention "
            "modules only. 1.0 keeps the historical flat/base LR; a smaller "
            "value (for example 0.3) de-risks newly initialized action-local "
            "modules during a warm-start A/B. Fails closed when the checkpoint "
            "does not contain those modules."
        ),
    )
    parser.add_argument(
        "--trunk-lr-mult",
        type=float,
        default=1.0,
        help=(
            "LR multiplier for the canonical EntityGraphNet trunk modules. 1.0 "
            "preserves the historical single optimizer group exactly. A non-unit "
            "value is supported only with --arch entity_graph and fails closed if "
            "the named trunk is absent or has no trainable parameters. Use "
            "--freeze-modules trunk for a true freeze rather than LR 0."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for --optimizer adamw.",
    )
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PyTorch's fused CUDA optimizer implementation when available.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--symmetry-augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "f74: augment each entity_graph training batch with a random D6 hex "
            "symmetry (12-fold) per row. Relabels board tokens + target ids while "
            "keeping legal-action row order, so policy/value targets are unchanged. "
            "Requires --arch entity_graph."
        ),
    )
    parser.add_argument(
        "--symmetry-augment-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --symmetry-augment is on, also relabel the event-log action-id "
            "scalar (event dim 35) through the action permutation. Minor history "
            "feature; disable to leave the event stream untouched."
        ),
    )
    parser.add_argument(
        "--soft-target-temperature",
        type=float,
        default=0.7,
        help="Temperature used to convert target_scores into distillation targets.",
    )
    parser.add_argument(
        "--soft-target-weight",
        type=float,
        default=0.7,
        help="Blend weight for soft distillation where soft labels exist; 1.0 means pure soft loss.",
    )
    parser.add_argument(
        "--soft-target-source",
        choices=("prefer_policy", "prefer_scores", "policy", "scores"),
        default="policy",
        help="Which stored soft labels to train against. prefer_scores allows temperature retuning.",
    )
    parser.add_argument(
        "--soft-target-min-legal-coverage",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of legal actions covered by a soft target before "
            "using KL/distillation for that row. Low-coverage rows fall back to "
            "hard CE so partial search labels do not silently mark unscored legal "
            "actions as bad."
        ),
    )
    parser.add_argument(
        "--truncated-vp-margin-value-weight",
        type=float,
        default=0.25,
        help=(
            "FIX F3: for TRUNCATED games (max_decisions cap, no clean winner), derive a "
            "soft value-head label from the VP margin at the point of truncation "
            "(final_actual_vps/final_public_vps + seat are populated even for truncated "
            "rows -- only the has_final_*_vps flag is gated on the game having actually "
            "ended) instead of excluding these rows from value supervision entirely. "
            "Applied at this weight (relative to a clean win/loss row's weight of 1.0). "
            "AUDIT FIX (default was 0.0, silently starving the value head of signal "
            "from the majority of rows on corpora with high truncation fractions --"
            "e.g. gen-1's 87.5%%): default is now 0.25, matching the value-repair-v2/v3 "
            "recipe's already-validated weight. Pass 0.0 to restore the pre-fix, "
            "truncated-rows-excluded-from-value-loss behavior. Scoped to the value "
            "head only -- never affects policy advantage weighting."
        ),
    )
    parser.add_argument(
        "--policy-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Scalar weight on the policy (action) loss term, alongside the existing "
            "value/final-vp/q loss weights. Previously implicit at a fixed 1.0 -- set to 0 "
            "for a pure value-head-repair pass (combine with --train-value-only / "
            "--freeze-modules to also stop the policy backbone from moving)."
        ),
    )
    parser.add_argument(
        "--policy-aux-active-batch-size",
        type=int,
        default=0,
        help=(
            "Opt-in local-rank microbatch of additional policy-active rows per "
            "optimizer step. Draws use the authenticated composite base measure "
            "conditioned on policy_weight_multiplier>0 and contribute policy loss "
            "only. 0 (default) is a strict no-op."
        ),
    )
    # EXP3 (value-reuse-discipline ablation, task #40): at a fixed policy recipe,
    # value-loss-weight 0.10 was strictly better than 0.25 under multi-epoch reuse
    # (same policy loss, lower value overfit). RUN-6 adopts 0.10 as the default.
    parser.add_argument("--value-loss-weight", type=float, default=0.10)
    parser.add_argument("--final-vp-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--q-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary Q-head regression weight from finite target_scores rows.",
    )
    parser.add_argument(
        "--policy-kl-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on a policy-KL anchor loss, pulling "
            "the trained policy toward the frozen seed checkpoint's recorded per-state "
            "prior distribution (the `prior_policy` shard column). 0.0 (default) disables "
            "it -- a pure no-op, bit-identical to prior runs. Its purpose is the "
            "unfreeze-with-KL value-repair recipe: train the FULL trunk on true outcomes "
            "(value loss) while this anchor keeps the policy from drifting off the seed, "
            "so a linear value head is no longer the only free parameter. Scoped to rows "
            "with a recorded prior (raw/gumbel self-play rows -- teacher rows contribute "
            "nothing, same has_prior filter as the KL telemetry). Anchors to the SAME "
            "stored priors. Forced single-action rows are excluded from its denominator. "
            "Use --policy-kl-anchor-direction to select the recovery-default forward "
            "distillation KL or the historical reverse-KL ablation."
        ),
    )
    parser.add_argument(
        "--policy-kl-anchor-direction",
        choices=("forward", "reverse"),
        default="forward",
        help=(
            "Direction of the behavior-preservation KL. 'forward' (default) is "
            "KL(champion_prior || trained_policy), the standard old-policy "
            "distillation objective. 'reverse' preserves the historical "
            "KL(trained_policy || champion_prior) only as an explicit legacy ablation."
        ),
    )
    parser.add_argument(
        "--value-uncertainty-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the optional value-uncertainty auxiliary head's regression loss "
            "(predicting the value head's own squared error (z - v)^2 against the true "
            "outcome, KataGo short-term-error style with a stop-gradient on the value "
            "prediction feeding the target). 0.0 (default) disables the loss; requires an "
            "--arch entity_graph model built with the head present (see "
            "EntityGraphNetConfig.value_uncertainty_head). Pure no-op when 0.0 or when the "
            "head is absent from outputs. Trains the head only -- search-side consumption "
            "of the prediction is a separate, not-yet-wired design (see docs)."
        ),
    )
    parser.add_argument(
        "--value-uncertainty-head",
        action="store_true",
        help=(
            "Build a fresh --arch entity_graph model (no --init-checkpoint) with the "
            "optional value-uncertainty auxiliary head present (EntityGraphNetConfig."
            "value_uncertainty_head=True), so --value-uncertainty-loss-weight has a "
            "head to train against. False (default) is a pure no-op -- matches every "
            "checkpoint built before this flag existed. Has no effect with "
            "--init-checkpoint: an existing checkpoint's own saved config already "
            "decides whether the head is present."
        ),
    )
    parser.add_argument(
        "--value-head-type",
        choices=("mse", "hlgauss"),
        default="mse",
        help=(
            "Select the PRIMARY value objective (CAT-39). 'mse' (default) trains only the "
            "historical scalar-MSE value head at --value-loss-weight. 'hlgauss' trains the "
            "distributional categorical head at --value-categorical-loss-weight, falling "
            "back to --value-loss-weight when that override is 0. The two modes therefore "
            "carry the same primary value-loss budget in a matched tournament. HL-Gauss "
            "requires an entity_graph model with value_categorical_bins >= 2 (use "
            "--value-categorical-bins for fresh/grown models, or "
            "f69_upgrade_checkpoint_config catbins:N for an existing checkpoint). To "
            "deliberately retain scalar MSE as "
            "an auxiliary in the HL-Gauss arm, set --hlgauss-scalar-aux-loss-weight; it is "
            "OFF by default so the primary-head comparison is not silently confounded."
        ),
    )
    parser.add_argument(
        "--value-categorical-bins",
        type=int,
        default=None,
        help=(
            "Number of HL-Gauss win/loss support bins to build on a fresh "
            "--arch entity_graph model (the optional truncation class is added "
            "separately by EntityGraphConfig). Values must be 0 (disabled) or >=2. "
            "With --init-checkpoint the default inherits the checkpoint value and an "
            "explicit value must match it. With --grow-from-checkpoint the default "
            "inherits the source value so a same-width/deeper model copies the trained "
            "categorical head; an explicit different value deliberately starts a new "
            "head. This makes fresh/grown HL-Gauss probes runnable without a mechanical "
            "config-upgrade checkpoint."
        ),
    )
    parser.add_argument(
        "--value-categorical-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Override for the PRIMARY HL-Gauss categorical cross-entropy weight (CAT-39). "
            "In --value-head-type hlgauss, 0.0 falls back to --value-loss-weight so scalar "
            "and categorical tournament arms use the same primary-loss budget. A nonzero "
            "value with --value-head-type mse is rejected as a contradictory configuration."
        ),
    )
    parser.add_argument(
        "--hlgauss-scalar-aux-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional scalar-MSE AUXILIARY weight in --value-head-type hlgauss. Default 0.0 "
            "makes the categorical-vs-scalar tournament a matched primary-objective test. "
            "Set explicitly only after the categorical primary wins and a hybrid objective "
            "is being tested as its own ablation; nonzero in mse mode is rejected."
        ),
    )
    parser.add_argument(
        "--value-hlgauss-sigma-ratio",
        type=float,
        default=0.75,
        help=(
            "HL-Gauss smoothing sigma as a multiple of the win-loss bin width (CAT-39). The "
            "scalar value target is projected onto the categorical bins as a Gaussian with "
            "this sigma (Farebrother et al. 2024 report ~0.75 x bin-width optimal; the CAT-39 "
            "spec says sigma ~ bin width, so 0.75 sits inside that band). Larger = smoother "
            "targets, smaller -> two-hot in the limit."
        ),
    )
    parser.add_argument(
        "--value-target-lambda",
        type=float,
        default=1.0,
        help=(
            "MuZero/ReZero-style value-target blend (CAT-39, arXiv:2404.16364): value target "
            "= lambda*z + (1-lambda)*V_search on rows carrying a stored search root value "
            "(the `root_value` / `root_value_mask` shard columns). 1.0 (default) is a pure "
            "no-op (pure realised-outcome z). Non-unit values require an explicit phase "
            "scope (or the named global-compatibility mode) and fail closed when no eligible "
            "root rows exist. The "
            "blend is applied in DISTRIBUTION space for the HL-Gauss head (project z and "
            "V_search each to a categorical distribution, then mix) and in scalar space for "
            "the MSE control arm -- the two are consistent because HL-Gauss preserves the "
            "expectation and blending is linear."
        ),
    )
    parser.add_argument(
        "--value-root-blend-phases",
        default="",
        help=(
            "Comma-separated authoritative game phases eligible for the search-root "
            "value blend. Example: DISCARD,MOVE_ROBBER,PLAY_TURN. When lambda < 1, "
            "either this option or --value-root-blend-global-compat is required. "
            "Opening placement/road rows remain pure outcome targets unless named."
        ),
    )
    parser.add_argument(
        "--value-root-blend-global-compat",
        action="store_true",
        help=(
            "Explicit compatibility mode for the historical behavior that blends every "
            "valid root-value row. Mutually exclusive with --value-root-blend-phases."
        ),
    )
    parser.add_argument(
        "--aux-subgoal-loss-weight",
        type=float,
        default=0.0,
        help=(
            "CAT-100: weight on the combined Catan-native auxiliary-subgoal loss "
            "(longest-road / largest-army / VP-in-N / next-settlement / robber-target "
            "heads; UNREAL-style, arXiv 1611.05397). A small value (0.02-0.1) is "
            "typical. 0.0 (default) disables it. Pure no-op unless the --arch "
            "entity_graph model was built with --aux-subgoal-heads (so the outputs are "
            "present) AND the corpus carries the matching aux target fields "
            "(aux_longest_road/aux_largest_army/aux_vp_in_n/aux_next_settlement/"
            "aux_robber_target); rows lacking a target are ignored per head. Targets "
            "are engine-free labels -- see catan_zero.rl.aux_subgoal_targets."
        ),
    )
    parser.add_argument(
        "--edge-policy-head",
        action="store_true",
        help=(
            "CAT-97: build the --arch entity_graph model with the GATEAU-style "
            "edge/node-feature policy head (EntityGraphConfig.edge_policy_head): a "
            "direct per-action logit read from each move's target entity token, added "
            "to the CLIP logits (zero-init, so identical at init). For warm-starting "
            "an existing checkpoint instead, use tools/f69_upgrade_checkpoint_config.py "
            "--flags edge. Ignored for non-entity architectures."
        ),
    )
    parser.add_argument(
        "--aux-subgoal-heads",
        action="store_true",
        help=(
            "CAT-100: build the --arch entity_graph model with the auxiliary subgoal "
            "heads (EntityGraphConfig.aux_subgoal_heads). Emits aux predictions only; "
            "value/policy outputs are unchanged. Pair with --aux-subgoal-loss-weight "
            "to train them. For warm-starting an existing checkpoint, use "
            "tools/f69_upgrade_checkpoint_config.py --flags aux."
        ),
    )
    parser.add_argument(
        "--freeze-modules",
        default="",
        help=(
            "Comma-separated --arch entity_graph module groups to freeze "
            "(requires_grad=False): trunk,action_encoder,policy_head,value_heads. "
            "The value_heads group includes scalar/categorical/final-VP/uncertainty "
            "readouts and optional value-attention-pool parameters. See "
            "--train-value-only for a shortcut covering the first three groups."
        ),
    )
    parser.add_argument(
        "--train-value-only",
        action="store_true",
        help=(
            "Shortcut for --freeze-modules trunk,action_encoder,policy_head (--arch "
            "entity_graph only): freezes everything except value_head/final_vp_head. "
            "Combine with --policy-loss-weight 0 so the frozen policy backbone also stops "
            "contributing gradient via the policy loss."
        ),
    )
    parser.add_argument(
        "--allow-teacher-score-q-loss",
        action="store_true",
        help=(
            "Allow target_scores to train the q_values head. Off by default because "
            "target_scores are teacher preference scores, while PPO expects q_values "
            "to be return-scale action values."
        ),
    )
    parser.add_argument(
        "--q-skip-teacher-prefixes",
        default="catanatron_ab",
        help=(
            "Comma-separated teacher-name prefixes excluded from target_scores Q loss. "
            "Rows marked target_score_source=ab_root are not skipped; older/fallback "
            "AB rows without that provenance stay excluded."
        ),
    )
    parser.add_argument(
        "--allow-legacy-action-mask-upgrade",
        action="store_true",
        help=(
            "Allow an old XDim checkpoint with missing action_mask_version to be "
            "stamped with the current environment version. Keep this off for "
            "production unless an action-ID replay smoke has passed."
        ),
    )
    parser.add_argument("--winner-sample-weight", type=float, default=1.0)
    parser.add_argument("--loser-sample-weight", type=float, default=0.3)
    parser.add_argument(
        "--vp-margin-weight",
        type=float,
        default=0.0,
        help="Optional sample multiplier by final VP margin / vps_to_win.",
    )
    parser.add_argument(
        "--advantage-policy-weighting",
        choices=("none", "outcome_value"),
        default="none",
        help=(
            "Optional AWR-lite policy reweighting. outcome_value multiplies the "
            "policy loss by exp((final_outcome - V(s)) / temperature), clamped "
            "and normalized per batch. Value/final-VP losses remain unbiased."
        ),
    )
    parser.add_argument(
        "--advantage-temperature",
        type=float,
        default=1.0,
        help="Temperature for --advantage-policy-weighting outcome_value.",
    )
    parser.add_argument(
        "--advantage-weight-cap",
        type=float,
        default=5.0,
        help="Maximum per-sample AWR multiplier before per-batch normalization.",
    )
    parser.add_argument(
        "--advantage-weight-floor",
        type=float,
        default=0.05,
        help="Minimum per-sample AWR multiplier before per-batch normalization.",
    )
    parser.add_argument(
        "--teacher-weights",
        default="",
        help="Comma-separated teacher weights, e.g. value_rollout_search=1.5,catanatron_value=1.2",
    )
    parser.add_argument(
        "--phase-weights",
        default="",
        help="Comma-separated phase weights, e.g. robber=3.0,initial_build=2.0",
    )
    parser.add_argument(
        "--value-phase-weights",
        default="",
        help=(
            "Comma-separated phase weights for the VALUE head, e.g. "
            "robber=8.0,initial_build=5.0 (FIX A5). Falls back to --phase-weights when empty, "
            "so the value head is phase-repaired the same way as the policy by default."
        ),
    )
    parser.add_argument(
        "--forced-action-weight",
        type=float,
        default=0.1,
        help="Multiplier for samples with exactly one legal action.",
    )
    parser.add_argument(
        "--per-game-policy-weight",
        action="store_true",
        help=(
            "Normalize positive POLICY-loss weights within each game_seed so long "
            "games do not contribute more policy mass merely because they contain "
            "more recorded rows. Zero-policy rows remain zero. Default OFF preserves "
            "historical row-level policy weighting exactly."
        ),
    )
    parser.add_argument(
        "--per-game-policy-weight-mode",
        choices=("equal", "sqrt"),
        default="equal",
        help=(
            "How --per-game-policy-weight scales positive per-game policy mass: "
            "'equal' gives every game equal total mass; 'sqrt' retains total mass "
            "proportional to sqrt(the game's original positive policy mass)."
        ),
    )
    parser.add_argument(
        "--policy-surprise-weight",
        type=float,
        default=0.0,
        help=(
            "CAT-45 diversity-strangulation countermeasure (roadmap R8): oversample "
            "rows where search disagreed most with the prior policy. Reweights the "
            "per-EPOCH SAMPLING order (not the loss) to draw rows with replacement, "
            "probability proportional to 1.0 + weight_scale * min(KL(target_policy || "
            "prior_policy), --policy-surprise-cap). 0.0 (default) disables this and "
            "keeps today's uniform rng.permutation epoch order exactly as-is. Rows "
            "with no recorded prior_policy (non-Gumbel-self-play teacher rows, or "
            "shards predating this column) always get the uniform baseline weight."
        ),
    )
    parser.add_argument(
        "--policy-surprise-cap",
        type=float,
        default=4.0,
        help="KL cap for --policy-surprise-weight; prevents a handful of extreme-"
        "surprise rows (e.g. wide/contested roots) from dominating the sampler.",
    )
    parser.add_argument(
        "--forced-row-value-weight",
        type=float,
        default=1.0,
        help=(
            "CAT-60: multiplier for VALUE-loss weight on rows with exactly one legal "
            "action. Default 1.0 is a no-op (byte-identical to pre-CAT-60 behavior). "
            "Distinct from --forced-action-weight, which only affects the POLICY loss."
        ),
    )
    parser.add_argument(
        "--per-game-value-weight",
        action="store_true",
        help=(
            "CAT-60: normalize VALUE-loss weights so every game (grouped by game_seed) "
            "contributes equal total loss mass regardless of its row count, addressing "
            "'16k games = 16k independent outcomes, not 3.6M labels'. Applied after "
            "phase weights, the CAT-45 value_weight_multiplier field, and "
            "--forced-row-value-weight -- see build_value_sample_weights' docstring for "
            "the exact combination rule. Default OFF: byte-identical to prior behavior."
        ),
    )
    parser.add_argument(
        "--per-game-value-weight-mode",
        choices=("equal", "sqrt"),
        default="equal",
        help=(
            "EXP3: how --per-game-value-weight distributes per-game mass. 'equal' (default, "
            "CAT-60 behavior): every game contributes EQUAL total value-loss mass (row weight "
            "divided by the game's summed weight -> long games heavily downweighted per row). "
            "'sqrt': row weight divided by SQRT of the game's summed weight, so a game of n "
            "value rows contributes ~sqrt(n) mass -- the effective-sample-size middle ground "
            "between row-level (mass n) and equal (mass 1), treating n correlated in-game rows "
            "as ~sqrt(n) independent value samples. No-op unless --per-game-value-weight is set."
        ),
    )
    # Internal A1 executor binding. These flags do not enable an ablation on
    # their own: the audited memmap/lock path below independently reconstructs
    # the effective recipe and rejects any mismatch or empty/no-op ablation.
    parser.add_argument("--a1-learner-ablation-id", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--a1-effective-learner-recipe-json", default="", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--a1-effective-learner-recipe-sha256", default="", help=argparse.SUPPRESS
    )
    parser.add_argument("--a1-ablation-code-binding-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--a1-ablation-code-tree-sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--a1-reviewed-lock-file-sha256", default="", help=argparse.SUPPRESS)
    parser.add_argument("--a1-dual-learner-lock", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--a1-dual-reviewed-lock-file-sha256", default="", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--a1-curriculum-parent-receipt", default="", help=argparse.SUPPRESS
    )
    parser.add_argument("--a1-batch-probe-plan", default="", help=argparse.SUPPRESS)
    parser.add_argument("--a1-batch-probe-run-id", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional checkpoint to continue XDim-lite BC training from.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restore a committed <init-checkpoint>.optimizer.pt + training-progress "
            "checkpoint set when its model hashes and recipe/schedule identity match. "
            "A partial, legacy, or mismatched set fails closed instead of silently "
            "restarting the LR/dose at step zero. Disable with --no-resume-optimizer "
            "for an explicit fresh-Adam warm start (including matched experiments). "
            "The resolved choice and restored step are recorded in the report."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=50,
        help="Rank-0 JSON heartbeat interval during each epoch; 0 disables.",
    )
    parser.add_argument(
        "--train-diagnostics-every-batches",
        type=int,
        default=0,
        help=(
            "Collect expensive per-phase/per-teacher train-batch diagnostics every N "
            "batches. 0 disables train diagnostics; validation diagnostics still run."
        ),
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable DDP unused-parameter discovery. The 35M xdim_graph BC path "
            "turns this on automatically when optional auxiliary heads are not "
            "part of the current loss."
        ),
    )
    parser.add_argument(
        "--amp",
        choices=("none", "bf16"),
        default="none",
        help=(
            "Mixed precision mode for XDim BC forward/loss. Use bf16 on B200/A100 "
            "to reduce activation memory and use tensor cores; reductions and "
            "reported metrics stay in float32."
        ),
    )
    parser.add_argument(
        "--allow-concurrent-bc",
        action="store_true",
        help=(
            "Allow multiple behavior-cloning jobs on the same host. Keep unset "
            "on the B200 training box so accidental launchers cannot stack two "
            "2-GPU DDP jobs on the same devices."
        ),
    )
    parser.add_argument(
        "--host-lock-file",
        default="/tmp/catan_zero_train_bc.lock",
        help="Host-local lock used to prevent accidental concurrent BC jobs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--require-strict-35m-teacher",
        action="store_true",
        help=(
            "Run tools/report_teacher_data_quality.py --strict-35m-teacher before "
            "loading data. This is a fail-closed preflight for serious 35M BC runs."
        ),
    )
    parser.add_argument(
        "--require-production-35m-teacher",
        action="store_true",
        help=(
            "Run the full production 35M teacher-data gate before training. Use this "
            "for B200 production BC runs; it implies --require-strict-35m-teacher."
        ),
    )
    parser.add_argument(
        "--require-35m-model",
        action="store_true",
        help=(
            "Fail unless --arch xdim_graph builds a model in the expected 35M "
            "parameter range. Implied by --require-production-35m-teacher."
        ),
    )
    parser.add_argument(
        "--skip-teacher-quality-gate",
        action="store_true",
        help=(
            "Skip the external teacher-data quality preflight. Use only for a "
            "curated dataset that was already gated in a previous run; in-process "
            "schema and quality checks still run after loading."
        ),
    )
    parser.add_argument(
        "--trust-curated-data-quality",
        action="store_true",
        help=(
            "Skip the expensive in-process teacher_data_quality diagnostic table and "
            "use a minimal trusted record with invalid_teacher_actions=0. Use only "
            "for a curated dataset that already passed the external quality gate."
        ),
    )
    parser.add_argument(
        "--ddp-shard-data",
        action="store_true",
        help=(
            "In DDP, load only the manifest shard files assigned to each rank. "
            "Use this for very large curated teacher sets so host RAM scales by "
            "rank instead of duplicating the full dataset on every GPU process."
        ),
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help=(
            "C1: shard model params/grads/optimizer state across ranks with "
            "FullyShardedDataParallel instead of replicating them with DDP. Use "
            "only when a net is too big to fit a DDP replica + optimizer state on "
            "one GPU; DDP is the workhorse for the 70-150M entity_graph configs on "
            "80GB H100s (they fit a replica with room to spare). Transformer blocks "
            "are auto-wrapped by module type; the final checkpoint is gathered to a "
            "full (unsharded) state_dict on rank 0 so it loads exactly like a "
            "DDP/single-GPU checkpoint. Requires torchrun (WORLD_SIZE>1) and "
            "--arch entity_graph/xdim_graph."
        ),
    )
    parser.add_argument(
        "--grow-from-checkpoint",
        default="",
        help=(
            "C1 warm-start-GROW: build a FRESH model at the requested "
            "--hidden-size/--graph-layers/--attention-heads (typically bigger than "
            "the checkpoint's) and copy every parameter/buffer whose NAME and SHAPE "
            "match from this checkpoint, leaving the rest at fresh init. Logs the "
            "fraction of the new model's parameters that were warm-started. Unlike "
            "--init-checkpoint (which rebuilds the model at the CHECKPOINT's config "
            "and enforces an exact architecture match), this deliberately allows a "
            "shape change: same-width/deeper configs warm-start the shared trunk "
            "blocks + encoders + heads cleanly; a width change matches little and "
            "falls back toward from-scratch. Mutually exclusive with "
            "--init-checkpoint."
        ),
    )
    parser.add_argument("--min-35m-params", type=int, default=30_000_000)
    parser.add_argument("--max-35m-params", type=int, default=40_000_000)
    parser.add_argument(
        "--skip-guards",
        action="store_true",
        help=(
            "Skip tools/prelaunch_guard.py's pre-launch checks (CLI-default-override "
            "trap, VAL-ONLY seed range, masked-regime mismatch on --init-checkpoint, "
            "fd-limit; CAT-69/CAT-75). Logs a loud WARNING and proceeds anyway -- use "
            "only for a known false positive or an intentional smoke test."
        ),
    )
    add_config_flags(parser, default_purpose="train_bc")
    return parser


def _build_guard_specs(
    args: argparse.Namespace, argv: Sequence[str], parser: argparse.ArgumentParser
) -> list[dict]:
    static_specs = launcher_guards.load_static_guard_specs("train_bc")
    specs = launcher_guards.merge_dynamic_args(
        static_specs, {"cli_flag_lint": {"argv": list(argv), "parser": parser}}
    )
    # VAL-ONLY-never-trains (b): best-effort discovery of the seed range(s) this
    # corpus was generated from, from any reachable generation manifest.json --
    # refuses a training launch whose corpus overlaps the reserved VAL-ONLY band.
    for seed_range in launcher_guards.discover_generation_seed_ranges(args.data):
        specs.append(
            {
                "name": "val_only_never_trains",
                "args": {"seed_range": seed_range, "purpose": "train"},
            }
        )
    # masked-regime (c): only meaningful when continuing from a checkpoint --
    # a fresh --arch entity_graph run has no prior regime to contradict.
    if args.init_checkpoint:
        specs.append(
            {
                "name": "masked_regime",
                "args": {
                    "checkpoint_path": args.init_checkpoint,
                    "expected_masked": bool(args.mask_hidden_info),
                },
            }
        )
    return specs


def _resolve_value_objective_weights(args) -> tuple[float, float]:
    """Return ``(scalar_mse_weight, categorical_ce_weight)``.

    ``--value-loss-weight`` is the primary-value budget shared by the matched
    scalar and HL-Gauss arms.  Historically the HL-Gauss path applied that
    budget to scalar MSE *and* again to categorical CE, so the purported
    one-variable tournament doubled its value gradient and validation only
    reported the scalar half.  Keep the default MSE path byte-for-byte, but
    make the opt-in HL-Gauss mode categorical-primary and require any hybrid
    scalar auxiliary to be explicit.
    """

    mode = str(getattr(args, "value_head_type", "mse"))
    # ``scalar`` was used by pre-parser unit fixtures and early config JSONs;
    # accept it as the exact historical alias for the CLI spelling ``mse``.
    if mode == "scalar":
        mode = "mse"
    primary = float(getattr(args, "value_loss_weight", 0.10))
    categorical_override = float(
        getattr(args, "value_categorical_loss_weight", 0.0)
    )
    scalar_aux = float(getattr(args, "hlgauss_scalar_aux_loss_weight", 0.0))
    weights = {
        "value_loss_weight": primary,
        "value_categorical_loss_weight": categorical_override,
        "hlgauss_scalar_aux_loss_weight": scalar_aux,
    }
    if any(value < 0.0 for value in weights.values()):
        raise SystemExit(
            "value objective weights must be non-negative: "
            + ", ".join(f"{key}={value}" for key, value in weights.items())
        )
    value_target_lambda = float(getattr(args, "value_target_lambda", 1.0))
    if not 0.0 <= value_target_lambda <= 1.0:
        raise SystemExit(
            "--value-target-lambda must be in [0, 1]; values outside the convex "
            "range create invalid categorical probability mass and extrapolated "
            f"scalar targets (got {value_target_lambda})"
        )
    sigma_ratio = float(getattr(args, "value_hlgauss_sigma_ratio", 0.75))
    if sigma_ratio <= 0.0:
        raise SystemExit(
            "--value-hlgauss-sigma-ratio must be > 0 so the HL-Gauss target is "
            f"well-defined (got {sigma_ratio})"
        )
    if mode == "mse":
        if categorical_override != 0.0:
            raise SystemExit(
                "--value-categorical-loss-weight is nonzero while "
                "--value-head-type=mse; select --value-head-type=hlgauss instead"
            )
        if scalar_aux != 0.0:
            raise SystemExit(
                "--hlgauss-scalar-aux-loss-weight is only valid with "
                "--value-head-type=hlgauss"
            )
        return primary, 0.0
    if mode == "hlgauss":
        categorical = categorical_override if categorical_override != 0.0 else primary
        if categorical <= 0.0:
            raise SystemExit(
                "--value-head-type=hlgauss requires a positive categorical primary "
                "weight via --value-loss-weight or --value-categorical-loss-weight"
            )
        return scalar_aux, categorical
    raise SystemExit(f"unknown --value-head-type={mode!r}")


def _value_training_metadata(
    args,
    *,
    scalar_weight: float,
    categorical_weight: float,
    categorical_bins: int,
    optimizer_steps: int,
    completed_epochs: int,
    scalar_training_weight_sum: float,
    categorical_training_weight_sum: float,
) -> dict[str, object]:
    """Build durable checkpoint provenance for value readouts trained here.

    Merely adding a categorical module to a checkpoint does not train it.  The
    explicit readout list lets search fail closed on config-only upgrades while
    remaining backwards compatible with legacy scalar checkpoints.
    """

    mode = str(getattr(args, "value_head_type", "mse"))
    if mode == "scalar":
        mode = "mse"
    primary_readout = "categorical" if mode == "hlgauss" else "scalar"
    trained_readouts: list[str] = []
    has_updates = int(optimizer_steps) > 0 and int(completed_epochs) > 0
    if (
        has_updates
        and float(scalar_weight) > 0.0
        and float(scalar_training_weight_sum) > 0.0
    ):
        trained_readouts.append("scalar")
    if (
        has_updates
        and float(categorical_weight) > 0.0
        and float(categorical_training_weight_sum) > 0.0
    ):
        trained_readouts.append("categorical")
    metadata = {
        "schema_version": "value-training-v1",
        "primary_readout": primary_readout,
        "trained_value_readouts": trained_readouts,
        "resolved_scalar_mse_weight": float(scalar_weight),
        "resolved_categorical_ce_weight": float(categorical_weight),
        "hlgauss_scalar_aux_loss_weight": float(
            getattr(args, "hlgauss_scalar_aux_loss_weight", 0.0)
        ),
        "hlgauss_bins": int(categorical_bins),
        "hlgauss_sigma_ratio": float(
            getattr(args, "value_hlgauss_sigma_ratio", 0.75)
        ),
        "value_target_lambda": float(getattr(args, "value_target_lambda", 1.0)),
        "value_root_blend_regime": _resolve_value_root_blend_regime(args),
        "value_root_blend_audit": getattr(args, "value_root_blend_audit", None),
        "truncated_vp_margin_value_weight": float(
            getattr(args, "truncated_vp_margin_value_weight", 0.0)
        ),
        "optimizer_steps": int(optimizer_steps),
        "completed_epochs": int(completed_epochs),
        "scalar_training_weight_sum": float(scalar_training_weight_sum),
        "categorical_training_weight_sum": float(
            categorical_training_weight_sum
        ),
        "checkout_runtime_binding": getattr(
            args, "checkout_runtime_binding", None
        ),
        **(
            {
                "a1_contract_sha256": args.a1_contract_sha256,
                "a1_selected_game_seed_set_sha256": (
                    getattr(args, "a1_selected_game_seed_set_sha256", None)
                ),
                "a1_training_game_seed_set_sha256": (
                    getattr(args, "a1_training_game_seed_set_sha256", None)
                ),
                "a1_learner_training_recipe_sha256": (
                    getattr(args, "a1_learner_training_recipe_sha256", None)
                ),
                "a1_memmap_payload_inventory_sha256": (
                    getattr(args, "a1_memmap_payload_inventory_sha256", None)
                ),
                "a1_learner_code_sha256": getattr(
                    args, "a1_learner_code_sha256", None
                ),
                "a1_runtime_code_tree_sha256": getattr(
                    args, "a1_runtime_code_tree_sha256", None
                ),
            }
            if getattr(args, "a1_contract_sha256", None)
            else {}
        ),
    }
    learner_ablation = getattr(args, "a1_learner_ablation", None)
    if learner_ablation is not None:
        metadata["learner_ablation"] = dict(learner_ablation)
    return metadata


def _assert_value_heads_present_for_losses(model, args) -> None:
    """Fail LOUD (SystemExit) when a value-head objective was requested on the
    CLI but the constructed/loaded model lacks the head that objective needs --
    instead of silently training with that loss term stuck at 0.0 for the whole
    (multi-hundred-GPU-hour) run. Mirrors the --value-lr-mult SystemExit guard:
    a requested-but-inert head is a misconfiguration, not a no-op.

    (1) --value-head-type hlgauss needs a categorical value head
        (value_categorical_bins >= 2). CAT-39's weight resolution makes the CE
        weight nonzero in hlgauss mode, but the loss term is still gated on
        "value_categorical_logits" in the model outputs; a scalar model never
        emits them, so the HL-Gauss objective is silently inert.
    (2) --value-uncertainty-loss-weight != 0 needs the value_uncertainty_head
        submodule; without it "value_uncertainty" is never in outputs and the
        loss stays 0.0.
    """
    _resolve_value_objective_weights(args)
    if str(getattr(args, "value_head_type", "mse")) == "hlgauss":
        bins = int(getattr(model, "value_categorical_bins", 0) or 0)
        if bins < 2:
            raise SystemExit(
                "--value-head-type hlgauss requires a categorical value head "
                "(EntityGraphNetConfig.value_categorical_bins >= 2; see "
                "f69_upgrade_checkpoint_config catbins:N), but the constructed/"
                f"loaded model has value_categorical_bins={bins}. Without it the "
                "HL-Gauss objective is silently inert and the run trains scalar-MSE "
                "only. Build/upgrade a catbins head (for a fresh model pass "
                "--value-categorical-bins >=2) or pass "
                "--value-head-type scalar."
            )
    if float(getattr(args, "value_uncertainty_loss_weight", 0.0) or 0.0) != 0.0:
        if getattr(model, "value_uncertainty_head", None) is None:
            raise SystemExit(
                "--value-uncertainty-loss-weight != 0 requires the model to carry a "
                "value_uncertainty_head (build a fresh model with "
                "--value-uncertainty-head, or resume a checkpoint that already has "
                "it), but the constructed/loaded model has none. Without it the "
                "uncertainty loss is silently inert; pass "
                "--value-uncertainty-loss-weight 0 or add the head."
            )


def _checkpoint_value_categorical_bins(checkpoint_path: str) -> int:
    """Read the categorical support width from a policy checkpoint config."""

    import torch

    from catan_zero.rl.config_serialization import config_attr_view

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise SystemExit(
            f"{checkpoint_path} is not a policy checkpoint with a config; cannot "
            "resolve --value-categorical-bins"
        )
    config = config_attr_view(checkpoint["config"])
    bins = int(getattr(config, "value_categorical_bins", 0) or 0)
    if bins == 1 or bins < 0:
        raise SystemExit(
            f"{checkpoint_path} records invalid value_categorical_bins={bins}; "
            "expected 0 (disabled) or >=2"
        )
    return bins


def _sha256_existing_file(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        return ""
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _training_data_fingerprint(path: str, data_format: str) -> str:
    """Return a durable corpus identity when the format exposes a manifest.

    A memmap metadata file describes the payload layout but, by itself, does
    not identify the flat-file bytes.  New corpora bind the metadata digest and
    the content-addressed payload inventory into one training fingerprint.
    Legacy non-A1 corpora without an inventory retain their historical metadata
    fingerprint for compatibility.
    """

    root = Path(path)
    if str(data_format) == "memmap" and root.is_file():
        try:
            descriptor = json.loads(root.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return _sha256_existing_file(root)
        if (
            isinstance(descriptor, dict)
            and descriptor.get("schema_version") in {
                "memmap_composite_v1", "memmap_composite_v2"
            }
        ):
            return _canonical_json_sha256(descriptor)
    candidates = (
        (root / "corpus_meta.json", root / "manifest.json")
        if str(data_format) == "memmap"
        else (root / "manifest.json", root / "corpus_meta.json")
    )
    for candidate in candidates:
        digest = _sha256_existing_file(candidate)
        if not digest:
            continue
        if str(data_format) == "memmap" and candidate.name == "corpus_meta.json":
            try:
                meta = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return digest
            inventory_sha = (
                meta.get("payload_inventory_sha256")
                if isinstance(meta, dict)
                else None
            )
            if _is_sha256(inventory_sha):
                return _canonical_json_sha256(
                    {
                        "corpus_meta_file_sha256": digest,
                        "payload_inventory_sha256": inventory_sha,
                    }
                )
        return digest
    return ""


def _preflight_memmap_composite_descriptor(path: str | Path) -> dict[str, object]:
    """Authenticate an ordered diagnostic-only no-copy descriptor.

    V1 remains the exact historical two-component format. V2 admits two or
    more components and binds a game-uniform sampling ratio for each one, plus
    the components whose recorded priors are allowed to enter the behavioral
    anchor. This prevents a current but regressed producer from silently
    becoming the policy that a recovery run preserves.
    """
    descriptor_path = Path(path).expanduser().resolve(strict=True)
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load memmap composite descriptor {path}: {error}") from error
    if not isinstance(descriptor, dict):
        raise SystemExit("memmap composite descriptor is not an object")
    schema_version = descriptor.get("schema_version")
    common_fields = {
        "schema_version", "diagnostic_only", "promotion_eligible", "components",
        "learner_recipe_overrides", "learner_recipe_overrides_sha256",
    }
    if schema_version == "memmap_composite_v1":
        valid_field_sets = (common_fields,)
    else:
        v2_fields = common_fields | {"policy_kl_anchor_component_ids"}
        # Optional for backward compatibility: absence preserves the historical
        # all-components policy-CE scope. Presence is authenticated by the
        # descriptor fingerprint and enables value-only replay components.
        valid_field_sets = (
            v2_fields,
            v2_fields | {"policy_distillation_component_ids"},
        )
    if schema_version not in {"memmap_composite_v1", "memmap_composite_v2"} or set(
        descriptor
    ) not in valid_field_sets:
        raise SystemExit("memmap composite descriptor fields differ from its schema")
    if (
        descriptor.get("diagnostic_only") is not True
        or descriptor.get("promotion_eligible") is not False
    ):
        raise SystemExit(
            "memmap composite v1 is diagnostic-only and must declare "
            "diagnostic_only=true, promotion_eligible=false"
        )
    raw_components = descriptor.get("components")
    if not isinstance(raw_components, list) or (
        len(raw_components) != 2
        if schema_version == "memmap_composite_v1"
        else len(raw_components) < 2
    ):
        requirement = "exactly two" if schema_version == "memmap_composite_v1" else "at least two"
        raise SystemExit(f"{schema_version} requires {requirement} ordered components")
    overrides = descriptor.get("learner_recipe_overrides")
    required_override_fields = {
        "per_game_policy_weight", "per_game_policy_weight_mode",
    }
    allowed_override_fields = {
        *required_override_fields,
        "forced_row_value_weight", "hlgauss_scalar_aux_loss_weight", "loser_sample_weight",
        "lr", "per_game_value_weight", "per_game_value_weight_mode",
        "policy_kl_anchor_direction", "policy_kl_anchor_weight",
        "value_categorical_bins", "value_categorical_loss_weight",
        "value_head_type", "value_hlgauss_sigma_ratio", "value_loss_weight",
    }
    if (
        not isinstance(overrides, dict)
        or not required_override_fields.issubset(overrides)
        or not set(overrides).issubset(allowed_override_fields)
    ):
        raise SystemExit(
            "memmap composite learner_recipe_overrides must bind per-game policy "
            "weighting and may contain only supported diagnostic recipe fields"
        )
    if not isinstance(overrides["per_game_policy_weight"], bool) or overrides[
        "per_game_policy_weight_mode"
    ] not in {"equal", "sqrt"}:
        raise SystemExit("memmap composite per-game policy recipe override is invalid")
    if descriptor.get("learner_recipe_overrides_sha256") != _canonical_json_sha256(overrides):
        raise SystemExit("memmap composite learner recipe override digest mismatch")
    components: list[dict[str, object]] = []
    inventory_bindings: list[dict[str, object]] = []
    seen_dirs: set[str] = set()
    expected_v1 = {
        "corpus_dir", "corpus_meta_sha256", "payload_inventory_sha256",
        "validation_manifest", "validation_manifest_sha256",
    }
    expected = (
        expected_v1
        if schema_version == "memmap_composite_v1"
        else expected_v1 | {"component_id", "game_sampling_ratio"}
    )
    component_ids: list[str] = []
    component_ratios: list[float] = []
    for index, raw in enumerate(raw_components):
        if not isinstance(raw, dict) or set(raw) != expected:
            raise SystemExit(f"memmap composite component {index} fields differ from schema")
        if schema_version == "memmap_composite_v2":
            component_id = raw["component_id"]
            ratio = raw["game_sampling_ratio"]
            if (
                not isinstance(component_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", component_id) is None
                or component_id in component_ids
            ):
                raise SystemExit(f"memmap composite component {index} has invalid/duplicate id")
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(
                float(ratio)
            ) or float(ratio) <= 0.0:
                raise SystemExit(f"memmap composite component {index} has invalid sampling ratio")
            component_ids.append(component_id)
            component_ratios.append(float(ratio))
        try:
            corpus_dir = Path(str(raw["corpus_dir"])).expanduser().resolve(strict=True)
            validation_path = Path(str(raw["validation_manifest"])).expanduser().resolve(strict=True)
        except OSError as error:
            raise SystemExit(f"cannot resolve memmap composite component {index}: {error}") from error
        if not corpus_dir.is_dir() or str(corpus_dir) != str(raw["corpus_dir"]):
            raise SystemExit(f"memmap composite component {index} corpus_dir is not canonical")
        if str(validation_path) != str(raw["validation_manifest"]):
            raise SystemExit(f"memmap composite component {index} validation_manifest is not canonical")
        if str(corpus_dir) in seen_dirs:
            raise SystemExit("memmap composite component corpus directories must be unique")
        seen_dirs.add(str(corpus_dir))
        meta_path = corpus_dir / "corpus_meta.json"
        actual_meta_sha = _sha256_existing_file(meta_path)
        if not _is_sha256(raw["corpus_meta_sha256"]) or raw["corpus_meta_sha256"] != actual_meta_sha:
            raise SystemExit(f"memmap composite component {index} corpus metadata hash mismatch")
        try:
            corpus_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot load component {index} corpus metadata {meta_path}: {error}") from error
        if not isinstance(corpus_meta, dict):
            raise SystemExit(f"memmap composite component {index} metadata is not an object")
        if not isinstance(corpus_meta.get("selected_game_seed_manifest"), dict) or not isinstance(
            corpus_meta.get("a1_post_wave_audit"), dict
        ):
            raise SystemExit(f"memmap composite component {index} is not an authenticated A1 corpus")
        actual_inventory_sha = _validate_memmap_payload_inventory(corpus_dir, corpus_meta)
        if not _is_sha256(raw["payload_inventory_sha256"]) or raw[
            "payload_inventory_sha256"
        ] != actual_inventory_sha:
            raise SystemExit(f"memmap composite component {index} payload inventory hash mismatch")
        actual_validation_sha = _sha256_existing_file(validation_path)
        if not _is_sha256(raw["validation_manifest_sha256"]) or raw[
            "validation_manifest_sha256"
        ] != actual_validation_sha:
            raise SystemExit(f"memmap composite component {index} validation manifest hash mismatch")
        components.append({
            **raw, "corpus_dir": str(corpus_dir),
            "validation_manifest": str(validation_path), "corpus_meta": corpus_meta,
        })
        inventory_bindings.append({
            "corpus_meta_sha256": actual_meta_sha,
            "payload_inventory_sha256": actual_inventory_sha,
        })
    anchor_component_ids: list[str] = []
    distillation_component_ids: list[str] = []
    distillation_scope_explicit = False
    if schema_version == "memmap_composite_v2":
        if not math.isclose(sum(component_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise SystemExit("memmap composite v2 game sampling ratios must sum to 1")
        raw_anchor_ids = descriptor["policy_kl_anchor_component_ids"]
        if (
            not isinstance(raw_anchor_ids, list)
            or any(not isinstance(value, str) for value in raw_anchor_ids)
            or len(set(raw_anchor_ids)) != len(raw_anchor_ids)
            or not set(raw_anchor_ids).issubset(component_ids)
        ):
            raise SystemExit("memmap composite v2 anchor component ids are invalid")
        anchor_component_ids = list(raw_anchor_ids)
        raw_distillation_ids = descriptor.get(
            "policy_distillation_component_ids", component_ids
        )
        if (
            not isinstance(raw_distillation_ids, list)
            or not raw_distillation_ids
            or any(not isinstance(value, str) for value in raw_distillation_ids)
            or len(set(raw_distillation_ids)) != len(raw_distillation_ids)
            or not set(raw_distillation_ids).issubset(component_ids)
        ):
            raise SystemExit(
                "memmap composite v2 policy distillation component ids are invalid"
            )
        # Canonical component order prevents a semantically-identical scope
        # from acquiring multiple authenticated identities.
        if raw_distillation_ids != [
            value for value in component_ids if value in set(raw_distillation_ids)
        ]:
            raise SystemExit(
                "memmap composite v2 policy distillation component ids must follow "
                "component order"
            )
        distillation_component_ids = list(raw_distillation_ids)
        distillation_scope_explicit = "policy_distillation_component_ids" in descriptor
    return {
        "schema_version": schema_version, "diagnostic_only": True,
        "promotion_eligible": False, "descriptor_path": str(descriptor_path),
        "descriptor_file_sha256": _sha256_existing_file(descriptor_path),
        "descriptor_fingerprint": _canonical_json_sha256(descriptor),
        "payload_inventory_sha256": _canonical_json_sha256(inventory_bindings),
        "learner_recipe_overrides": overrides,
        "learner_recipe_overrides_sha256": descriptor[
            "learner_recipe_overrides_sha256"
        ],
        "components": components,
        "component_ids": component_ids,
        "component_game_sampling_ratios": component_ratios,
        "policy_kl_anchor_component_ids": anchor_component_ids,
        "policy_distillation_component_ids": distillation_component_ids,
        "policy_distillation_scope_explicit": distillation_scope_explicit,
    }


def _validate_composite_learner_recipe_authorization(
    args: argparse.Namespace, composite_meta: dict[str, object]
) -> None:
    expected = composite_meta.get("learner_recipe_overrides")
    if not isinstance(expected, dict):
        raise SystemExit("memmap composite has no authenticated learner recipe override")
    converters = {
        "forced_row_value_weight": float,
        "hlgauss_scalar_aux_loss_weight": float,
        "loser_sample_weight": float,
        "lr": float,
        "per_game_policy_weight": bool,
        "per_game_policy_weight_mode": str,
        "per_game_value_weight": bool,
        "per_game_value_weight_mode": str,
        "policy_kl_anchor_direction": str,
        "policy_kl_anchor_weight": float,
        "value_categorical_bins": int,
        "value_categorical_loss_weight": float,
        "value_head_type": str,
        "value_hlgauss_sigma_ratio": float,
        "value_loss_weight": float,
    }
    actual = {
        key: converters[key](getattr(args, key))
        for key in expected
    }
    if expected != actual:
        raise SystemExit(
            "memmap composite command differs from its authenticated diagnostic "
            f"learner recipe override: descriptor={expected!r} command={actual!r}"
        )


def _expected_memmap_payload_filenames(
    corpus_meta: dict[str, object],
) -> set[str]:
    columns = corpus_meta.get("columns")
    if not isinstance(columns, dict) or not columns:
        raise SystemExit("A1 memmap metadata has no column schema")
    expected = {"row_offsets.dat"}
    for name, raw_schema in columns.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise SystemExit("A1 memmap column name is not a safe filename component")
        if not isinstance(raw_schema, dict):
            raise SystemExit(f"A1 memmap column {name!r} has malformed schema")
        kind = raw_schema.get("kind")
        if kind not in {
            "fixed",
            "ragged2d",
            "ragged3d",
            "string",
            "implicit_constant",
        }:
            raise SystemExit(
                f"A1 memmap column {name!r} has unsupported kind {kind!r}"
            )
        if kind == "implicit_constant":
            continue
        expected.add(
            f"{name}.codes.dat" if kind == "string" else f"{name}.dat"
        )
    return expected


def _payload_auth_cache_root() -> Path:
    override = os.environ.get("TRAIN_BC_PAYLOAD_AUTH_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "catan-zero" / "payload-auth"


def _payload_filesystem_identity(path: Path) -> dict[str, int | str]:
    """Return a fail-closed identity for one immutable payload path."""

    try:
        before = path.lstat()
    except OSError as error:
        raise SystemExit(f"cannot stat A1 memmap payload {path}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"A1 memmap payload must be a non-symlink regular file: {path}")
    return _payload_stat_identity(path.name, before)


def _payload_stat_identity(
    filename: str, value: os.stat_result
) -> dict[str, int | str]:
    return {
        "filename": filename,
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "mode": int(stat.S_IMODE(value.st_mode)),
    }


def _sha256_stable_payload(
    path: Path, expected_identity: dict[str, int | str]
) -> str:
    """Hash one exact inode and reject mutation or pathname replacement."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise SystemExit(f"cannot open A1 memmap payload {path}: {error}") from error
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "rb") as handle:
            identity_before = _payload_stat_identity(path.name, os.fstat(handle.fileno()))
            if identity_before != expected_identity:
                raise SystemExit(
                    f"A1 memmap payload {path.name} identity changed before hashing"
                )
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            identity_after = _payload_stat_identity(path.name, os.fstat(handle.fileno()))
    except OSError as error:
        raise SystemExit(f"cannot hash A1 memmap payload {path}: {error}") from error
    if identity_after != expected_identity or _payload_filesystem_identity(path) != expected_identity:
        raise SystemExit(f"A1 memmap payload {path.name} changed while being authenticated")
    return f"sha256:{digest.hexdigest()}"


def _payload_auth_cache_binding(
    root: Path,
    corpus_meta: dict[str, object],
    inventory_sha: str,
    identities: list[dict[str, int | str]],
) -> dict[str, object]:
    meta_path = root / "corpus_meta.json"
    return {
        "schema_version": MEMMAP_PAYLOAD_AUTH_CACHE_SCHEMA,
        "validator_version": MEMMAP_PAYLOAD_AUTH_VALIDATOR_VERSION,
        "corpus_dir": str(root.resolve(strict=True)),
        "corpus_meta_file_sha256": _sha256_existing_file(meta_path),
        "corpus_descriptor_sha256": _canonical_json_sha256(corpus_meta),
        "payload_inventory_schema": corpus_meta.get("payload_inventory_schema"),
        "payload_inventory_sha256": inventory_sha,
        "payloads": identities,
    }


def _payload_auth_cache_path(binding: dict[str, object]) -> Path:
    digest = _canonical_json_sha256(binding).split(":", 1)[1]
    return _payload_auth_cache_root() / f"{digest}.json"


def _load_payload_auth_cache(
    binding: dict[str, object], *, total_bytes: int
) -> bool:
    path = _payload_auth_cache_path(binding)
    try:
        directory = path.parent
        directory_stat = directory.lstat()
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(directory_stat.st_mode) & 0o077
        ):
            return False
        cache_stat = path.lstat()
        if (
            not stat.S_ISREG(cache_stat.st_mode)
            or stat.S_ISLNK(cache_stat.st_mode)
            or cache_stat.st_uid != os.geteuid()
            or stat.S_IMODE(cache_stat.st_mode) & 0o077
        ):
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or set(payload) != {
        "binding", "binding_sha256", "authenticated_bytes", "entry_sha256"
    }:
        return False
    unsigned = dict(payload)
    entry_sha = unsigned.pop("entry_sha256", None)
    return bool(
        payload.get("binding") == binding
        and payload.get("binding_sha256") == _canonical_json_sha256(binding)
        and payload.get("authenticated_bytes") == total_bytes
        and entry_sha == _canonical_json_sha256(unsigned)
    )


def _publish_payload_auth_cache(
    binding: dict[str, object], *, total_bytes: int
) -> None:
    """Best-effort publish; inability to cache never weakens verification."""

    path = _payload_auth_cache_path(binding)
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_stat = path.parent.lstat()
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(directory_stat.st_mode) != 0o700
        ):
            return
        unsigned = {
            "binding": binding,
            "binding_sha256": _canonical_json_sha256(binding),
            "authenticated_bytes": total_bytes,
        }
        payload = {**unsigned, "entry_sha256": _canonical_json_sha256(unsigned)}
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    except OSError:
        return


def _validate_memmap_payload_inventory(
    data_path: str | Path, corpus_meta: dict[str, object]
) -> str:
    """Verify every A1 flat payload file against its immutable inventory."""

    if (
        corpus_meta.get("payload_inventory_schema")
        != MEMMAP_PAYLOAD_INVENTORY_SCHEMA
    ):
        raise SystemExit(
            "A1 memmap payload inventory schema is missing or unsupported"
        )
    inventory = corpus_meta.get("payload_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise SystemExit("A1 memmap payload inventory is missing or empty")
    declared_inventory_sha = corpus_meta.get("payload_inventory_sha256")
    actual_inventory_sha = _canonical_json_sha256(inventory)
    if (
        not _is_sha256(declared_inventory_sha)
        or declared_inventory_sha != actual_inventory_sha
    ):
        raise SystemExit(
            "A1 memmap payload inventory semantic digest mismatch: "
            f"declared={declared_inventory_sha!r} actual={actual_inventory_sha!r}"
        )

    expected_names = _expected_memmap_payload_filenames(corpus_meta)
    records_by_name: dict[str, dict[str, object]] = {}
    prior_name: str | None = None
    for index, record in enumerate(inventory):
        if not isinstance(record, dict) or set(record) != {
            "filename",
            "size_bytes",
            "sha256",
        }:
            raise SystemExit(
                f"A1 memmap payload inventory record {index} has malformed fields"
            )
        filename = record["filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise SystemExit(
                f"A1 memmap payload inventory record {index} has unsafe filename"
            )
        if prior_name is not None and filename <= prior_name:
            raise SystemExit(
                "A1 memmap payload inventory filenames must be strictly sorted"
            )
        prior_name = filename
        size_bytes = record["size_bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not _is_sha256(record["sha256"])
        ):
            raise SystemExit(
                f"A1 memmap payload inventory record {index} has invalid size/hash"
            )
        records_by_name[filename] = record
    if set(records_by_name) != expected_names:
        raise SystemExit(
            "A1 memmap payload inventory filenames differ from the column schema: "
            f"missing={sorted(expected_names - set(records_by_name))} "
            f"unexpected={sorted(set(records_by_name) - expected_names)}"
        )

    root = Path(data_path).expanduser()
    actual_names = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and (path.name.endswith(".dat") or path.name.endswith(".codes.dat"))
    }
    if actual_names != expected_names:
        raise SystemExit(
            "A1 memmap on-disk payload filenames differ from the authenticated "
            f"inventory: missing={sorted(expected_names - actual_names)} "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    identities = [
        _payload_filesystem_identity(root / filename)
        for filename in sorted(expected_names)
    ]
    total_bytes = sum(int(identity["size_bytes"]) for identity in identities)
    binding = _payload_auth_cache_binding(
        root, corpus_meta, actual_inventory_sha, identities
    )
    cache_eligible = all(
        int(identity["mode"]) & 0o222 == 0 for identity in identities
    )
    if cache_eligible and _load_payload_auth_cache(binding, total_bytes=total_bytes):
        print(json.dumps({
            "progress": "a1_payload_auth_cache",
            "status": "hit",
            "payload_count": len(identities),
            "authenticated_bytes": total_bytes,
            "bytes_avoided": total_bytes,
            "binding_sha256": _canonical_json_sha256(binding),
        }, sort_keys=True), flush=True)
        return actual_inventory_sha

    for filename in sorted(expected_names):
        record = records_by_name[filename]
        payload_path = root / filename
        identity_before = _payload_filesystem_identity(payload_path)
        actual_size = int(identity_before["size_bytes"])
        if actual_size != record["size_bytes"]:
            raise SystemExit(
                f"A1 memmap payload {filename} size mismatch: "
                f"declared={record['size_bytes']} actual={actual_size}"
            )
        actual_sha = _sha256_stable_payload(payload_path, identity_before)
        identity_after = _payload_filesystem_identity(payload_path)
        if identity_after != identity_before:
            raise SystemExit(
                f"A1 memmap payload {filename} changed while being authenticated"
            )
        if actual_sha != record["sha256"]:
            raise SystemExit(
                f"A1 memmap payload {filename} sha256 mismatch: "
                f"declared={record['sha256']!r} actual={actual_sha!r}"
            )
    # Recompute the complete binding after hashing. This also catches a sibling
    # payload changing while a different file was being read.
    final_identities = [
        _payload_filesystem_identity(root / filename)
        for filename in sorted(expected_names)
    ]
    if final_identities != identities:
        raise SystemExit("A1 memmap payload identity changed during authentication")
    if cache_eligible:
        _publish_payload_auth_cache(binding, total_bytes=total_bytes)
    print(json.dumps({
        "progress": "a1_payload_auth_cache",
        "status": "miss",
        "miss_reason": (
            "cache_absent_or_invalid"
            if cache_eligible
            else "payload_not_read_only"
        ),
        "payload_count": len(identities),
        "authenticated_bytes": total_bytes,
        "bytes_avoided": 0,
        "cache_eligible": cache_eligible,
        "binding_sha256": _canonical_json_sha256(binding),
    }, sort_keys=True), flush=True)
    return actual_inventory_sha


def _preflight_a1_memmap_metadata(
    data_path: str | Path, *, validation_manifest_path: str | Path | None
) -> dict[str, object] | None:
    """Auto-detect an A1 corpus and forbid bypassing its exact holdout path."""

    expanded = Path(data_path).expanduser()
    if expanded.is_file():
        if validation_manifest_path:
            raise SystemExit(
                "memmap composite descriptor binds each component validation manifest; "
                "do not also pass --validation-game-seed-manifest"
            )
        return _preflight_memmap_composite_descriptor(expanded)
    meta_path = expanded / "corpus_meta.json"
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load memmap corpus metadata {meta_path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{meta_path} must contain a JSON object")
    selected = payload.get("selected_game_seed_manifest")
    audit = payload.get("a1_post_wave_audit")
    is_a1 = selected is not None or audit is not None
    if not is_a1:
        return None
    if not isinstance(selected, dict) or not isinstance(audit, dict):
        raise SystemExit(
            "A1 memmap metadata must bind both selected_game_seed_manifest and "
            "a1_post_wave_audit"
        )
    if not validation_manifest_path:
        raise SystemExit(
            "A1 memmap corpus detected: --validation-game-seed-manifest is "
            "mandatory and cannot be replaced by a recomputed fraction/range split"
        )
    _validate_memmap_payload_inventory(data_path, payload)
    return payload


def _single_node_ddp_preflight_enabled(ddp: dict[str, int | bool]) -> bool:
    """Return whether all ranks are workers on this one shared-filesystem node.

    ``LOCAL_WORLD_SIZE`` is deliberately required.  If a non-torchrun launcher
    does not provide enough topology information, every rank retains the
    historical independent verification instead of assuming shared storage.
    Multi-node launches likewise keep per-rank verification because equal path
    strings do not prove that the underlying files are equal across nodes.
    """

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0)
    return (
        bool(ddp.get("enabled", False))
        and local_world_size > 1
        and int(ddp.get("world_size", 1)) == local_world_size
    )


def _a1_preflight_store(
    ddp: dict[str, int | bool],
    *,
    data_path: str | Path,
    validation_manifest_path: str | Path | None,
) -> Any:
    """Connect to torchrun's CPU rendezvous store without initializing CUDA."""

    import torch.distributed as dist

    try:
        timeout_seconds = int(
            os.environ.get("TRAIN_BC_A1_PREFLIGHT_TIMEOUT_SECONDS", "21600")
        )
    except ValueError as error:
        raise SystemExit(
            "TRAIN_BC_A1_PREFLIGHT_TIMEOUT_SECONDS must be an integer"
        ) from error
    if timeout_seconds < 1:
        raise SystemExit("TRAIN_BC_A1_PREFLIGHT_TIMEOUT_SECONDS must be >= 1")
    master_addr = os.environ.get("MASTER_ADDR", "")
    master_port = os.environ.get("MASTER_PORT", "")
    if not master_addr or not master_port:
        raise SystemExit(
            "single-node DDP A1 preflight requires torchrun MASTER_ADDR/MASTER_PORT"
        )
    try:
        store = dist.TCPStore(
            master_addr,
            int(master_port),
            world_size=None,
            is_master=False,
            timeout=timedelta(seconds=timeout_seconds),
        )
    except Exception as error:
        raise SystemExit(
            "cannot connect to torchrun store for single-node A1 preflight: "
            f"{error}"
        ) from error
    identity = {
        "run_id": os.environ.get("TORCHELASTIC_RUN_ID", ""),
        "restart_count": os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
        "world_size": int(ddp["world_size"]),
        "data": str(Path(data_path).expanduser().absolute()),
        "validation_manifest": (
            ""
            if validation_manifest_path is None
            else str(Path(validation_manifest_path).expanduser().absolute())
        ),
    }
    prefix = "train_bc/a1_preflight/" + _canonical_json_sha256(identity)[7:] + "/"
    return dist.PrefixStore(prefix, store)


def _coordinated_a1_memmap_preflight(
    data_path: str | Path,
    *,
    validation_manifest_path: str | Path | None,
    ddp: dict[str, int | bool],
    _store: Any | None = None,
) -> dict[str, object] | None:
    """Verify an A1 corpus once for single-node DDP and fail closed on all ranks.

    Rank 0 publishes a small authenticated-metadata result through torchrun's
    CPU rendezvous store.  Peers never touch the large payload files during
    this preflight.  A rank-0 verification error, malformed response, store
    failure, or timeout stops the peer before CUDA/process-group setup.
    """

    if not _single_node_ddp_preflight_enabled(ddp):
        return _preflight_a1_memmap_metadata(
            data_path, validation_manifest_path=validation_manifest_path
        )

    store = (
        _store
        if _store is not None
        else _a1_preflight_store(
            ddp,
            data_path=data_path,
            validation_manifest_path=validation_manifest_path,
        )
    )
    result_key = "result"
    if int(ddp.get("rank", 0)) == 0:
        try:
            metadata = _preflight_a1_memmap_metadata(
                data_path, validation_manifest_path=validation_manifest_path
            )
            packet = {"schema_version": 1, "ok": True, "metadata": metadata}
        except BaseException as error:
            packet = {
                "schema_version": 1,
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            try:
                store.set(result_key, json.dumps(packet, sort_keys=True))
            except Exception as publish_error:
                raise SystemExit(
                    "rank 0 A1 preflight failed and its failure could not be "
                    f"published to peers: {publish_error}; original error: {error}"
                ) from error
            raise
        try:
            store.set(result_key, json.dumps(packet, sort_keys=True))
        except Exception as error:
            raise SystemExit(
                f"rank 0 could not publish successful A1 preflight: {error}"
            ) from error
        return metadata

    try:
        raw = store.get(result_key)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        packet = json.loads(raw)
    except Exception as error:
        raise SystemExit(
            f"rank {ddp.get('rank')} could not receive rank-0 A1 preflight: {error}"
        ) from error
    if not isinstance(packet, dict) or packet.get("schema_version") != 1:
        raise SystemExit("rank-0 A1 preflight returned a malformed response")
    if packet.get("ok") is not True:
        raise SystemExit(
            "rank-0 A1 preflight failed: "
            f"{packet.get('error_type', 'error')}: {packet.get('error', '')}"
        )
    metadata = packet.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise SystemExit("rank-0 A1 preflight returned malformed metadata")
    return metadata


def _game_seed_set_sha256(seeds: np.ndarray) -> str:
    canonical = np.sort(np.unique(np.asarray(seeds, dtype=np.int64))).astype(
        "<i8", copy=False
    )
    return f"sha256:{hashlib.sha256(canonical.tobytes()).hexdigest()}"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _load_validation_game_seed_manifest_for_training(
    path: str | Path,
    *,
    validation_fraction: float,
    validation_seed: int,
    validation_max_samples: int,
    validation_game_seed_ranges: list[tuple[int, int]],
) -> dict[str, object]:
    """Validate the exact A1 holdout before any optimizer step can run.

    This input is deliberately stricter than the trainer's output manifest.
    It is the immutable sidecar emitted by ``a1_pre_wave_contract.py audit``
    and binds the exact validation games, their byte-level seed digest, and
    the A1 contract that selected the corpus.  Recomputing a nominally equal
    5% split and comparing only after training would detect drift too late.
    """

    try:
        manifest_path = Path(path).expanduser().resolve(strict=True)
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise SystemExit(
            f"cannot read validation game-seed manifest {path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"cannot parse validation game-seed manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SystemExit("validation game-seed manifest must be a JSON object")
    expected_fields = {
        "schema_version",
        "a1_contract_sha256",
        "validation_fraction",
        "validation_seed",
        "validation_max_samples",
        "validation_game_seed_ranges",
        "validation_game_seed_count",
        "validation_row_count",
        "validation_game_seed_set_sha256",
        "game_seeds",
    }
    if set(payload) != expected_fields:
        raise SystemExit(
            "validation game-seed manifest fields differ from the exact "
            "train-validation-game-seeds-v1 schema; "
            f"missing={sorted(expected_fields - set(payload))} "
            f"extra={sorted(set(payload) - expected_fields)}"
        )
    if payload["schema_version"] != "train-validation-game-seeds-v1":
        raise SystemExit(
            "validation game-seed manifest schema must be "
            "'train-validation-game-seeds-v1'"
        )
    if not _is_sha256(payload["a1_contract_sha256"]):
        raise SystemExit("validation game-seed manifest has invalid a1_contract_sha256")

    manifest_fraction = payload["validation_fraction"]
    if isinstance(manifest_fraction, bool) or not isinstance(
        manifest_fraction, (int, float)
    ):
        raise SystemExit("validation game-seed manifest validation_fraction is invalid")
    if not math.isclose(
        float(manifest_fraction),
        float(validation_fraction),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit(
            "validation game-seed manifest fraction differs from CLI: "
            f"manifest={manifest_fraction} cli={validation_fraction}"
        )
    for field, cli_value in (
        ("validation_seed", validation_seed),
        ("validation_max_samples", validation_max_samples),
    ):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"validation game-seed manifest {field} is invalid")
        if int(value) != int(cli_value):
            raise SystemExit(
                f"validation game-seed manifest {field} differs from CLI: "
                f"manifest={value} cli={cli_value}"
            )
    if int(validation_max_samples) != 0:
        raise SystemExit(
            "--validation-game-seed-manifest requires --validation-max-samples 0; "
            "row-capping an exact game holdout would change its semantics"
        )
    if validation_game_seed_ranges:
        raise SystemExit(
            "--validation-game-seed-manifest and --validation-game-seed-ranges "
            "are mutually exclusive"
        )
    if payload["validation_game_seed_ranges"] != []:
        raise SystemExit(
            "A1 validation game-seed manifest must declare no range override"
        )

    raw_seeds = payload["game_seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise SystemExit("validation game-seed manifest has no game_seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise SystemExit("validation game-seed manifest game_seeds must be integers")
    try:
        seeds = np.asarray(raw_seeds, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise SystemExit(
            "validation game-seed manifest contains a seed outside int64"
        ) from error
    if not np.all(seeds[1:] > seeds[:-1]):
        raise SystemExit(
            "validation game-seed manifest game_seeds must be strictly sorted and unique"
        )
    count = payload["validation_game_seed_count"]
    if isinstance(count, bool) or not isinstance(count, int) or int(count) != len(seeds):
        raise SystemExit(
            "validation game-seed manifest validation_game_seed_count does not "
            f"match its seed list ({count!r} != {len(seeds)})"
        )
    row_count = payload["validation_row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise SystemExit("validation game-seed manifest validation_row_count is invalid")
    actual_digest = _game_seed_set_sha256(seeds)
    if payload["validation_game_seed_set_sha256"] != actual_digest:
        raise SystemExit(
            "validation game-seed manifest seed digest mismatch: "
            f"declared={payload['validation_game_seed_set_sha256']!r} "
            f"actual={actual_digest!r}"
        )
    return {
        "path": manifest_path,
        "file_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "manifest_sha256": _canonical_json_sha256(payload),
        "a1_contract_sha256": payload["a1_contract_sha256"],
        "validation_row_count": int(row_count),
        "validation_game_seed_set_sha256": actual_digest,
        "game_seeds": seeds,
    }


def _load_composite_validation_contract(
    composite_meta: dict[str, object],
    *,
    validation_fraction: float,
    validation_seed: int,
    validation_max_samples: int,
    validation_game_seed_ranges: list[tuple[int, int]],
) -> dict[str, object]:
    """Validate both bound holdouts and return their game-disjoint union."""
    raw_components = composite_meta.get("components")
    if not isinstance(raw_components, list) or len(raw_components) < 2:
        raise SystemExit("authenticated memmap composite metadata has invalid components")
    contracts: list[dict[str, object]] = []
    seen: set[int] = set()
    union: list[int] = []
    total_rows = 0
    bindings: list[dict[str, object]] = []
    for index, component in enumerate(raw_components):
        if not isinstance(component, dict):
            raise SystemExit(f"memmap composite component {index} metadata is malformed")
        contract = _load_validation_game_seed_manifest_for_training(
            component["validation_manifest"],
            validation_fraction=validation_fraction,
            validation_seed=validation_seed,
            validation_max_samples=validation_max_samples,
            validation_game_seed_ranges=validation_game_seed_ranges,
        )
        if contract["file_sha256"] != component["validation_manifest_sha256"]:
            raise SystemExit(f"memmap composite component {index} validation manifest binding drift")
        _validate_a1_validation_manifest_corpus_binding(component["corpus_meta"], contract)
        seeds = [int(seed) for seed in np.asarray(contract["game_seeds"], dtype=np.int64)]
        overlap = seen.intersection(seeds)
        if overlap:
            raise SystemExit(
                "memmap composite validation game seeds are not disjoint: "
                f"component={index} overlap_count={len(overlap)}"
            )
        seen.update(seeds)
        union.extend(seeds)
        total_rows += int(contract["validation_row_count"])
        contracts.append(contract)
        bindings.append({
            "a1_contract_sha256": contract["a1_contract_sha256"],
            "manifest_sha256": contract["manifest_sha256"],
            "validation_game_seed_set_sha256": contract["validation_game_seed_set_sha256"],
        })
    union_seeds = np.sort(np.asarray(union, dtype=np.int64))
    return {
        "path": Path(str(composite_meta["descriptor_path"])),
        "file_sha256": composite_meta["descriptor_file_sha256"],
        "manifest_sha256": _canonical_json_sha256(bindings),
        "a1_contract_sha256": _canonical_json_sha256(
            [binding["a1_contract_sha256"] for binding in bindings]
        ),
        "validation_row_count": total_rows,
        "validation_game_seed_set_sha256": _game_seed_set_sha256(union_seeds),
        "game_seeds": union_seeds,
        "component_contracts": contracts,
        "diagnostic_only": True,
        "promotion_eligible": False,
    }


def _load_composite_validation_sentinel_manifest(
    path: str | Path,
    *,
    composite_meta: dict[str, object],
    full_contract: dict[str, object],
) -> dict[str, object]:
    """Authenticate a whole-game subset of a composite's full holdout.

    The returned contract deliberately retains ``excluded_game_seeds`` as the
    complete authenticated holdout. Only ``game_seeds`` is narrowed for metric
    evaluation, so unused sentinel games can never leak back into training.
    """
    if int(full_contract.get("validation_row_count", 0)) <= 0:
        raise SystemExit("cannot derive a sentinel from an empty validation contract")
    try:
        manifest_path = Path(path).expanduser().resolve(strict=True)
        raw = manifest_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load validation sentinel manifest {path}: {error}") from error
    expected_fields = {
        "schema_version", "source_composite_descriptor_file_sha256",
        "source_composite_descriptor_fingerprint", "source_validation_bindings",
        "selection_seed", "target_row_count", "selected_row_count",
        "selected_game_seed_count", "selected_game_seed_set_sha256",
        "excluded_game_seed_count", "excluded_game_seed_set_sha256", "game_seeds",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise SystemExit("validation sentinel manifest fields differ from its schema")
    if payload["schema_version"] != "train-validation-game-sentinel-v1":
        raise SystemExit("validation sentinel manifest schema is unsupported")
    for field in (
        "selection_seed", "target_row_count", "selected_row_count",
        "selected_game_seed_count", "excluded_game_seed_count",
    ):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"validation sentinel manifest {field} is invalid")
    if payload["target_row_count"] <= 0 or payload["selected_row_count"] <= 0:
        raise SystemExit("validation sentinel row counts must be positive")
    if payload["source_composite_descriptor_file_sha256"] != composite_meta.get(
        "descriptor_file_sha256"
    ) or payload["source_composite_descriptor_fingerprint"] != composite_meta.get(
        "descriptor_fingerprint"
    ):
        raise SystemExit("validation sentinel source composite binding drift")

    contracts = full_contract.get("component_contracts")
    if not isinstance(contracts, list):
        raise SystemExit("validation sentinel source has no component contracts")
    expected_bindings = [
        {
            "component_index": index,
            "validation_manifest_file_sha256": contract["file_sha256"],
            "validation_manifest_sha256": contract["manifest_sha256"],
            "validation_game_seed_set_sha256": contract[
                "validation_game_seed_set_sha256"
            ],
        }
        for index, contract in enumerate(contracts)
    ]
    if payload["source_validation_bindings"] != expected_bindings:
        raise SystemExit("validation sentinel source manifest binding drift")

    raw_seeds = payload["game_seeds"]
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
    ):
        raise SystemExit("validation sentinel game_seeds must be non-empty integers")
    try:
        selected = np.asarray(raw_seeds, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise SystemExit("validation sentinel contains a seed outside int64") from error
    if not np.all(selected[1:] > selected[:-1]):
        raise SystemExit("validation sentinel game_seeds must be strictly sorted and unique")
    full = np.asarray(full_contract["game_seeds"], dtype=np.int64)
    if not set(map(int, selected)).issubset(set(map(int, full))):
        raise SystemExit("validation sentinel contains games outside the authenticated holdout")
    if payload["selected_game_seed_count"] != int(selected.size):
        raise SystemExit("validation sentinel selected game count mismatch")
    selected_digest = _game_seed_set_sha256(selected)
    if payload["selected_game_seed_set_sha256"] != selected_digest:
        raise SystemExit("validation sentinel selected seed digest mismatch")
    if (
        payload["excluded_game_seed_count"] != int(full.size)
        or payload["excluded_game_seed_set_sha256"]
        != full_contract["validation_game_seed_set_sha256"]
    ):
        raise SystemExit("validation sentinel full-holdout exclusion binding drift")

    return {
        **full_contract,
        "path": manifest_path,
        "file_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "manifest_sha256": _canonical_json_sha256(payload),
        "validation_row_count": int(payload["selected_row_count"]),
        "validation_game_seed_set_sha256": selected_digest,
        "game_seeds": selected,
        "excluded_game_seeds": full,
        "full_validation_row_count": int(full_contract["validation_row_count"]),
        "sentinel_target_row_count": int(payload["target_row_count"]),
        "sentinel_selection_seed": int(payload["selection_seed"]),
    }


def _validate_a1_validation_manifest_corpus_binding(
    corpus_meta: object, validation_seed_contract: dict[str, object]
) -> None:
    """Require the exact audited corpus selected by the holdout's A1 contract."""

    if not isinstance(corpus_meta, dict):
        raise SystemExit(
            "--validation-game-seed-manifest requires memmap corpus metadata"
        )
    selected_meta = corpus_meta.get("selected_game_seed_manifest")
    if not isinstance(selected_meta, dict):
        raise SystemExit(
            "the memmap corpus does not bind an audited A1 selected-game manifest"
        )
    expected_contract = str(validation_seed_contract["a1_contract_sha256"])
    if selected_meta.get("a1_contract_sha256") != expected_contract:
        raise SystemExit(
            "validation holdout and memmap corpus bind different A1 contracts: "
            f"holdout={expected_contract!r} "
            f"corpus={selected_meta.get('a1_contract_sha256')!r}"
        )
    audit_meta = corpus_meta.get("a1_post_wave_audit")
    if not isinstance(audit_meta, dict):
        raise SystemExit(
            "the memmap corpus does not bind a passing A1 post-wave audit"
        )
    if audit_meta.get("contract_sha256") != expected_contract:
        raise SystemExit(
            "A1 post-wave audit and validation holdout bind different contracts"
        )
    if not _is_sha256(audit_meta.get("file_sha256")) or not _is_sha256(
        audit_meta.get("audit_sha256")
    ):
        raise SystemExit("memmap A1 post-wave audit provenance is malformed")
    validation_meta = audit_meta.get("validation_holdout")
    if not isinstance(validation_meta, dict):
        raise SystemExit(
            "memmap A1 post-wave audit does not bind the validation sidecar"
        )
    expected_validation_binding = {
        "path": str(validation_seed_contract["path"]),
        "file_sha256": validation_seed_contract["file_sha256"],
        "manifest_sha256": validation_seed_contract["manifest_sha256"],
        "a1_contract_sha256": validation_seed_contract["a1_contract_sha256"],
        "validation_game_seed_count": int(
            np.asarray(validation_seed_contract["game_seeds"], dtype=np.int64).size
        ),
        "validation_row_count": int(validation_seed_contract["validation_row_count"]),
        "validation_game_seed_set_sha256": validation_seed_contract[
            "validation_game_seed_set_sha256"
        ],
    }
    if validation_meta != expected_validation_binding:
        raise SystemExit(
            "trainer validation sidecar differs from the exact file bound by the "
            "A1 post-wave audit"
        )
    dual_identity = (selected_meta.get("arm_id"), selected_meta.get("subset_id"))
    expected_selected_count = DUAL_ARM_SUBSET_COUNTS.get(dual_identity)
    if expected_selected_count is None:
        expected_selected_count = 12_000
        if any(value is not None for value in dual_identity):
            raise SystemExit("A1 memmap corpus has an invalid dual-arm identity")
    if selected_meta.get("selected_game_count") != expected_selected_count:
        raise SystemExit(
            "A1 memmap corpus selected-game count differs from its exact "
            f"contract ({selected_meta.get('selected_game_count')!r} != "
            f"{expected_selected_count})"
        )
    expected_validation_count = int(
        np.asarray(validation_seed_contract["game_seeds"], dtype=np.int64).size
    )
    if selected_meta.get("validation_game_count") != expected_validation_count:
        raise SystemExit(
            "validation holdout game count differs from the selected-game manifest "
            "bound into corpus_meta.json"
        )
    selected_validation_digest = selected_meta.get(
        "validation_game_seed_set_sha256"
    )
    if selected_validation_digest != validation_seed_contract[
        "validation_game_seed_set_sha256"
    ]:
        raise SystemExit(
            "validation holdout seed digest differs from the selected-game "
            "manifest bound into corpus_meta.json"
        )


def _read_sha256_bound_json(
    path_value: object, expected_file_sha256: object, *, label: str
) -> tuple[Path, dict[str, object], str]:
    try:
        path = Path(str(path_value)).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read bound {label} {path_value}: {error}") from error
    actual_file_sha256 = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    if actual_file_sha256 != expected_file_sha256:
        raise SystemExit(
            f"bound {label} file SHA-256 drift: "
            f"expected={expected_file_sha256!r} actual={actual_file_sha256!r}"
        )
    if not isinstance(payload, dict):
        raise SystemExit(f"bound {label} must be a JSON object")
    return path, payload, actual_file_sha256


def _validate_dual_arm_corpus_artifacts_and_seeds(
    corpus_meta: dict[str, object],
    validation_seed_contract: dict[str, object],
    game_seed_column: np.ndarray,
) -> dict[str, object]:
    """Replay the dual-arm selection/audit chain before optimizer creation.

    This is deliberately separate from the historical 12k validator below:
    accepting a typed dual-arm corpus must not relax one byte of the frozen
    A1 path.
    """

    selected_meta = corpus_meta["selected_game_seed_manifest"]
    audit_meta = corpus_meta["a1_post_wave_audit"]
    assert isinstance(selected_meta, dict) and isinstance(audit_meta, dict)
    identity = (selected_meta.get("arm_id"), selected_meta.get("subset_id"))
    expected_count = DUAL_ARM_SUBSET_COUNTS.get(identity)
    if expected_count is None:
        raise SystemExit("dual-arm corpus has an unauthorized arm/subset identity")

    selected_path, selected_payload, _ = _read_sha256_bound_json(
        selected_meta.get("path"),
        selected_meta.get("file_sha256"),
        label="dual-arm selected-game manifest",
    )
    if selected_payload.get("schema_version") != DUAL_ARM_SELECTED_GAMES_SCHEMA:
        raise SystemExit("dual-arm selected-game manifest schema drift")
    expected_selected_fields = {
        "schema_version",
        "arm_id",
        "subset_id",
        "a1_contract_sha256",
        "selection_rule",
        "selected_game_count",
        "selected_game_seed_set_sha256",
        "category_game_counts",
        "training_game_count",
        "training_game_seed_set_sha256",
        "validation_game_count",
        "validation_game_seed_set_sha256",
        "records_sha256",
        "records",
        "parent_manifest_sha256",
    }
    if set(selected_payload) != expected_selected_fields:
        raise SystemExit("dual-arm selected-game manifest fields drift")
    if (selected_payload.get("arm_id"), selected_payload.get("subset_id")) != identity:
        raise SystemExit("dual-arm selected-game manifest identity drift")
    expected_categories = DUAL_ARM_SUBSET_CATEGORY_COUNTS.get(identity)
    if selected_payload.get("category_game_counts") != expected_categories:
        raise SystemExit("dual-arm selected-game category quotas drift")
    parent_manifest_sha = selected_payload.get("parent_manifest_sha256")
    is_full = identity[1] in {"full-56k", "full-140k"}
    if (is_full and parent_manifest_sha is not None) or (
        not is_full and not _is_sha256(parent_manifest_sha)
    ):
        raise SystemExit("dual-arm selected-game parent provenance drift")
    records = selected_payload.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise SystemExit(
            f"dual-arm selected-game manifest must contain exactly {expected_count} records"
        )
    if selected_payload.get("records_sha256") != _canonical_json_sha256(records):
        raise SystemExit("dual-arm selected-game records_sha256 mismatch")

    all_seeds: list[int] = []
    train_seeds: list[int] = []
    validation_seeds: list[int] = []
    producer_shas: set[str] = set()
    category_counts = {category: 0 for category in expected_categories or {}}
    previous: tuple[int, str] | None = None
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("arm_id") != identity[0]:
            raise SystemExit(f"dual-arm selected-game record {index} identity drift")
        seed, job_id, split = (
            record.get("game_seed"),
            record.get("job_id"),
            record.get("split"),
        )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(job_id, str)
            or not job_id
            or split not in {"train", "validation"}
        ):
            raise SystemExit(f"dual-arm selected-game record {index} is malformed")
        key = (int(seed), job_id)
        if previous is not None and key <= previous:
            raise SystemExit("dual-arm selected-game records are not strictly sorted")
        previous = key
        producer_sha = record.get("producer_checkpoint_sha256")
        if not _is_sha256(producer_sha):
            raise SystemExit(f"dual-arm selected-game record {index} producer drift")
        producer_shas.add(str(producer_sha))
        category = record.get("category")
        if category not in category_counts:
            raise SystemExit(f"dual-arm selected-game record {index} category drift")
        category_counts[str(category)] += 1
        all_seeds.append(int(seed))
        (train_seeds if split == "train" else validation_seeds).append(int(seed))
    if len(set(all_seeds)) != len(all_seeds) or len(producer_shas) != 1:
        raise SystemExit("dual-arm selected-game seed or producer identity drift")
    if category_counts != expected_categories:
        raise SystemExit("dual-arm selected-game record category quotas drift")

    digest_fields = {
        "selected_game_count": len(all_seeds),
        "selected_game_seed_set_sha256": _game_seed_set_sha256(
            np.asarray(all_seeds, dtype=np.int64)
        ),
        "training_game_count": len(train_seeds),
        "training_game_seed_set_sha256": _game_seed_set_sha256(
            np.asarray(train_seeds, dtype=np.int64)
        ),
        "validation_game_count": len(validation_seeds),
        "validation_game_seed_set_sha256": _game_seed_set_sha256(
            np.asarray(validation_seeds, dtype=np.int64)
        ),
        "records_sha256": _canonical_json_sha256(records),
    }
    for field, actual in digest_fields.items():
        if selected_payload.get(field) != actual or selected_meta.get(field) != actual:
            raise SystemExit(f"dual-arm selected-game {field} drift")
    if not np.array_equal(
        np.asarray(validation_seeds, dtype=np.int64),
        np.asarray(validation_seed_contract["game_seeds"], dtype=np.int64),
    ):
        raise SystemExit("dual-arm selected validation split differs from holdout")

    observed = np.asarray(game_seed_column, dtype=np.int64).reshape(-1)
    if observed.size == 0 or not np.array_equal(
        np.sort(np.unique(observed)), np.asarray(all_seeds, dtype=np.int64)
    ):
        raise SystemExit("dual-arm memmap game-seed set differs from its selection")
    run_starts = np.concatenate(
        (np.asarray([0]), np.flatnonzero(observed[1:] != observed[:-1]) + 1)
    )
    if np.unique(observed[run_starts]).size != run_starts.size:
        raise SystemExit("dual-arm memmap repeats a non-contiguous game seed run")

    audit_path, audit_payload, _ = _read_sha256_bound_json(
        audit_meta.get("path"), audit_meta.get("file_sha256"), label="dual-arm audit"
    )
    audit_schema = audit_payload.get("schema_version")
    if (
        audit_schema not in DUAL_ARM_AUDIT_SCHEMAS
        or audit_payload.get("passed") is not True
        or audit_payload.get("errors") != []
        or (audit_payload.get("arm_id"), audit_payload.get("subset_id")) != identity
    ):
        raise SystemExit("dual-arm audit is not a clean matching authorization")
    actual_audit_sha = _canonical_json_sha256(
        {key: value for key, value in audit_payload.items() if key != "audit_sha256"}
    )
    if (
        audit_payload.get("audit_sha256") != actual_audit_sha
        or audit_meta.get("audit_sha256") != actual_audit_sha
    ):
        raise SystemExit("dual-arm audit semantic digest drift")
    if audit_schema == "a1-dual-arm-derived-post-wave-audit-v1":
        parent_binding = audit_payload.get("parent_audit")
        if not isinstance(parent_binding, dict) or set(parent_binding) != {
            "path",
            "file_sha256",
            "audit_sha256",
            "selected_manifest_file_sha256",
            "shard_inventory_sha256",
        }:
            raise SystemExit("derived dual-arm audit parent binding drift")
        parent_path, parent_payload, _ = _read_sha256_bound_json(
            parent_binding.get("path"),
            parent_binding.get("file_sha256"),
            label="parent dual-arm audit",
        )
        parent_audit_sha = _canonical_json_sha256(
            {key: value for key, value in parent_payload.items() if key != "audit_sha256"}
        )
        parent_selected = parent_payload.get("selected_training_games")
        if (
            parent_payload.get("schema_version") != "a1-dual-arm-post-wave-audit-v1"
            or parent_payload.get("passed") is not True
            or parent_payload.get("errors") != []
            or parent_payload.get("audit_sha256") != parent_audit_sha
            or parent_binding.get("audit_sha256") != parent_audit_sha
            or parent_payload.get("arm_id") != identity[0]
            or parent_payload.get("contract_sha256")
            != audit_payload.get("contract_sha256")
            or parent_payload.get("shard_inventory_sha256")
            != parent_binding.get("shard_inventory_sha256")
            or audit_payload.get("shard_inventory_sha256")
            != parent_binding.get("shard_inventory_sha256")
            or not isinstance(parent_selected, dict)
            or parent_selected.get("manifest_file_sha256")
            != parent_binding.get("selected_manifest_file_sha256")
            or parent_manifest_sha
            != parent_binding.get("selected_manifest_file_sha256")
            or not parent_path.is_file()
        ):
            raise SystemExit("derived dual-arm audit parent authorization drift")
    contract_sha = validation_seed_contract["a1_contract_sha256"]
    if (
        audit_payload.get("contract_sha256") != contract_sha
        or audit_meta.get("contract_sha256") != contract_sha
        or selected_payload.get("a1_contract_sha256") != contract_sha
    ):
        raise SystemExit("dual-arm audit/selection/holdout contract mismatch")
    validation_row_count = int(validation_seed_contract["validation_row_count"])
    if (
        audit_payload.get("rows") != int(observed.size)
        or corpus_meta.get("row_count") != int(observed.size)
        or audit_meta.get("selected_row_count") != int(observed.size)
        or audit_meta.get("training_row_count")
        != int(observed.size) - validation_row_count
    ):
        raise SystemExit("dual-arm memmap row exposure differs from its audit")
    expected_selected_binding = {
        "manifest": str(selected_path),
        "manifest_sha256": _canonical_json_sha256(selected_payload),
        "manifest_file_sha256": selected_meta["file_sha256"],
        "selected_game_count": expected_count,
        "selected_game_seed_set_sha256": digest_fields[
            "selected_game_seed_set_sha256"
        ],
        "records_sha256": digest_fields["records_sha256"],
    }
    if audit_payload.get("selected_training_games") != expected_selected_binding:
        raise SystemExit("dual-arm audit selected-game binding drift")
    expected_validation_binding = {
        "manifest": str(validation_seed_contract["path"]),
        "manifest_sha256": validation_seed_contract["manifest_sha256"],
        "manifest_file_sha256": validation_seed_contract["file_sha256"],
        "validation_game_seed_count": len(validation_seeds),
        "validation_game_seed_set_sha256": digest_fields[
            "validation_game_seed_set_sha256"
        ],
    }
    if audit_payload.get("validation_holdout") != expected_validation_binding:
        raise SystemExit("dual-arm audit validation binding drift")

    # The dual learner keeps the proven A1 objective/optimizer recipe while
    # changing only topology: 8 ranks x 512 local = the same global batch 4096.
    # ``train_bc.py`` is launched by torchrun as a file path, so sys.path[0]
    # is ``tools/`` rather than the repository root.  Import the sibling
    # through the tools-directory bootstrap above; a ``tools.*`` import only
    # works accidentally when callers inject the repository into PYTHONPATH.
    from a1_pre_wave_contract import EXPECTED_LEARNER_TRAINING_RECIPE

    recipe = dict(EXPECTED_LEARNER_TRAINING_RECIPE)
    recipe.update({"batch_size": 512, "world_size": 8, "global_batch_size": 4096})
    repo = Path(__file__).resolve().parents[1]
    learner_snapshot = [
        {"path": suffix, "sha256": _sha256_existing_file(repo / suffix)}
        for suffix in sorted(A1_REQUIRED_LEARNER_CODE_SUFFIXES)
    ]
    runtime_snapshot = [
        {"path": suffix, "sha256": _sha256_existing_file(repo / suffix)}
        for suffix in sorted(A1_REQUIRED_RUNTIME_CODE_SUFFIXES)
    ]
    return {
        "dual_arm": True,
        "arm_id": identity[0],
        "subset_id": identity[1],
        "learner_value_objective": {"objective": "mse", "value_readout": "scalar"},
        "learner_training_recipe": recipe,
        "learner_training_recipe_sha256": _canonical_json_sha256(recipe),
        "learner_code_sha256": _canonical_json_sha256(learner_snapshot),
        "runtime_code_tree_sha256": _canonical_json_sha256(runtime_snapshot),
        "producer_checkpoint_sha256": next(iter(producer_shas)),
        "selected_game_seed_set_sha256": digest_fields[
            "selected_game_seed_set_sha256"
        ],
        "training_game_seed_set_sha256": digest_fields[
            "training_game_seed_set_sha256"
        ],
        "audit_file_sha256": _sha256_existing_file(audit_path),
    }


def _validate_a1_corpus_artifacts_and_seeds(
    corpus_meta: dict[str, object],
    validation_seed_contract: dict[str, object],
    game_seed_column: np.ndarray,
) -> dict[str, object]:
    """Replay the selected/audit/lock chain against the actual memmap seeds."""

    selected_meta = corpus_meta["selected_game_seed_manifest"]
    audit_meta = corpus_meta["a1_post_wave_audit"]
    assert isinstance(selected_meta, dict)
    assert isinstance(audit_meta, dict)

    if selected_meta.get("arm_id") is not None:
        return _validate_dual_arm_corpus_artifacts_and_seeds(
            corpus_meta, validation_seed_contract, game_seed_column
        )

    selected_path, selected_payload, _ = _read_sha256_bound_json(
        selected_meta.get("path"),
        selected_meta.get("file_sha256"),
        label="A1 selected-game manifest",
    )
    if str(selected_path) != str(selected_meta.get("path")):
        raise SystemExit("A1 selected-game manifest path is not canonical")
    if selected_payload.get("schema_version") != "a1-selected-training-games-v1":
        raise SystemExit("A1 selected-game manifest schema drift")
    if (
        selected_payload.get("a1_contract_sha256")
        != validation_seed_contract["a1_contract_sha256"]
    ):
        raise SystemExit("A1 selected-game manifest contract hash drift")
    records = selected_payload.get("records")
    if not isinstance(records, list) or len(records) != 12_000:
        raise SystemExit("A1 selected-game manifest must contain exactly 12,000 records")
    if selected_payload.get("records_sha256") != _canonical_json_sha256(records):
        raise SystemExit("A1 selected-game manifest records_sha256 mismatch")

    all_seeds: list[int] = []
    train_seeds: list[int] = []
    validation_seeds: list[int] = []
    previous_seed: int | None = None
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"A1 selected-game record {index} is not an object")
        seed = record.get("game_seed")
        split = record.get("split")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SystemExit(f"A1 selected-game record {index} has invalid game_seed")
        if previous_seed is not None and int(seed) <= previous_seed:
            raise SystemExit("A1 selected-game records are not strictly seed-sorted")
        previous_seed = int(seed)
        if split not in {"train", "validation"}:
            raise SystemExit(f"A1 selected-game record {index} has invalid split")
        all_seeds.append(int(seed))
        (train_seeds if split == "train" else validation_seeds).append(int(seed))

    digest_fields = {
        "selected_game_count": len(all_seeds),
        "selected_game_seed_set_sha256": _game_seed_set_sha256(
            np.asarray(all_seeds, dtype=np.int64)
        ),
        "training_game_count": len(train_seeds),
        "training_game_seed_set_sha256": _game_seed_set_sha256(
            np.asarray(train_seeds, dtype=np.int64)
        ),
        "validation_game_count": len(validation_seeds),
        "validation_game_seed_set_sha256": _game_seed_set_sha256(
            np.asarray(validation_seeds, dtype=np.int64)
        ),
        "records_sha256": _canonical_json_sha256(records),
    }
    for field, actual in digest_fields.items():
        if selected_payload.get(field) != actual or selected_meta.get(field) != actual:
            raise SystemExit(f"A1 selected-game {field} drift")
    if not np.array_equal(
        np.asarray(validation_seeds, dtype=np.int64),
        np.asarray(validation_seed_contract["game_seeds"], dtype=np.int64),
    ):
        raise SystemExit(
            "A1 selected-game validation split differs from the exact holdout sidecar"
        )

    observed = np.asarray(game_seed_column, dtype=np.int64).reshape(-1)
    if observed.size == 0:
        raise SystemExit("A1 memmap corpus has no game_seed rows")
    run_starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(observed[1:] != observed[:-1]) + 1,
        )
    )
    run_values = observed[run_starts]
    if np.unique(run_values).size != run_values.size:
        raise SystemExit(
            "A1 memmap game_seed starts more than one non-contiguous run"
        )
    observed_unique = np.sort(np.unique(observed))
    if not np.array_equal(observed_unique, np.asarray(all_seeds, dtype=np.int64)):
        expected = set(all_seeds)
        actual = set(map(int, observed_unique.tolist()))
        raise SystemExit(
            "A1 memmap actual game-seed set differs from its selected manifest: "
            f"missing={len(expected - actual)} unexpected={len(actual - expected)}"
        )

    audit_path, audit_payload, _ = _read_sha256_bound_json(
        audit_meta.get("path"),
        audit_meta.get("file_sha256"),
        label="A1 post-wave audit",
    )
    if str(audit_path) != str(audit_meta.get("path")):
        raise SystemExit("A1 post-wave audit path is not canonical")
    if (
        audit_payload.get("schema_version") != "a1-post-wave-audit-v2"
        or audit_payload.get("passed") is not True
        or audit_payload.get("errors") != []
    ):
        raise SystemExit("A1 post-wave audit is not a clean passing artifact")
    actual_audit_sha = _canonical_json_sha256(
        {key: value for key, value in audit_payload.items() if key != "audit_sha256"}
    )
    if (
        audit_payload.get("audit_sha256") != actual_audit_sha
        or audit_meta.get("audit_sha256") != actual_audit_sha
    ):
        raise SystemExit("A1 post-wave audit semantic digest drift")
    contract_sha = validation_seed_contract["a1_contract_sha256"]
    if (
        audit_payload.get("contract_sha256") != contract_sha
        or audit_meta.get("contract_sha256") != contract_sha
    ):
        raise SystemExit("A1 post-wave audit contract hash drift")
    selected_row_count = audit_payload.get("rows")
    validation_row_count = int(validation_seed_contract["validation_row_count"])
    if (
        isinstance(selected_row_count, bool)
        or not isinstance(selected_row_count, int)
        or selected_row_count != int(observed.size)
        or corpus_meta.get("row_count") != selected_row_count
        or audit_meta.get("selected_row_count") != selected_row_count
        or audit_meta.get("training_row_count")
        != selected_row_count - validation_row_count
    ):
        raise SystemExit(
            "A1 memmap row exposure differs from the passing audit: "
            f"observed={observed.size} audit={selected_row_count!r} "
            f"meta_selected={audit_meta.get('selected_row_count')!r} "
            f"meta_training={audit_meta.get('training_row_count')!r}"
        )
    selected_binding = audit_payload.get("selected_training_games")
    if not isinstance(selected_binding, dict) or selected_binding != {
        "manifest": str(selected_path),
        "manifest_sha256": _canonical_json_sha256(selected_payload),
        "manifest_file_sha256": selected_meta["file_sha256"],
        "selected_game_count": 12_000,
        "selected_game_seed_set_sha256": digest_fields[
            "selected_game_seed_set_sha256"
        ],
        "records_sha256": digest_fields["records_sha256"],
    }:
        raise SystemExit("A1 post-wave audit selected-game binding drift")
    validation_binding = audit_payload.get("validation_holdout")
    expected_validation_binding = {
        "manifest": str(validation_seed_contract["path"]),
        "manifest_sha256": validation_seed_contract["manifest_sha256"],
        "manifest_file_sha256": validation_seed_contract["file_sha256"],
        "validation_game_seed_count": len(validation_seeds),
        "validation_game_seed_set_sha256": digest_fields[
            "validation_game_seed_set_sha256"
        ],
    }
    if validation_binding != expected_validation_binding:
        raise SystemExit("A1 post-wave audit validation binding drift")

    contract_path_value = audit_payload.get("contract_path")
    lock_path, lock_payload, _ = _read_sha256_bound_json(
        contract_path_value,
        _sha256_existing_file(str(contract_path_value)),
        label="A1 contract lock",
    )
    if str(lock_path) != str(contract_path_value):
        raise SystemExit("A1 contract lock path is not canonical")
    lock_schema = lock_payload.get("schema_version")
    if lock_schema not in {
        "a1-pre-wave-contract-lock-v2",
        "a1-pre-wave-contract-lock-v3",
    }:
        raise SystemExit("A1 contract lock schema drift")
    promotion_handoff = lock_payload.get("promotion_handoff")
    if lock_schema == "a1-pre-wave-contract-lock-v3":
        if (
            not isinstance(promotion_handoff, dict)
            or promotion_handoff.get("mode") != "post_promotion"
        ):
            raise SystemExit("A1 v3 contract lacks its post-promotion producer handoff")
    elif promotion_handoff is not None:
        # v2 locks created after the boundary carry an explicit historical
        # marker.  Markerless v2 remains readable only here for already-audited
        # pre-boundary corpora; a1_pre_wave_contract can no longer seal one.
        if (
            not isinstance(promotion_handoff, dict)
            or set(promotion_handoff) != {"mode", "reason"}
            or promotion_handoff.get("mode") != "historical_pre_promotion"
            or not isinstance(promotion_handoff.get("reason"), str)
            or not promotion_handoff["reason"].strip()
        ):
            raise SystemExit("A1 v2 historical promotion marker drift")
    lock_digest = _canonical_json_sha256(
        {key: value for key, value in lock_payload.items() if key != "contract_sha256"}
    )
    if lock_payload.get("contract_sha256") != contract_sha or lock_digest != contract_sha:
        raise SystemExit("A1 contract lock semantic digest drift")
    provenance = lock_payload.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit("A1 contract lock has no provenance section")
    learner_code = provenance.get("learner_code")
    if not isinstance(learner_code, list) or not learner_code:
        raise SystemExit("A1 contract lock does not bind learner implementation files")
    learner_code_sha256 = _canonical_json_sha256(learner_code)
    if provenance.get("learner_code_sha256") != learner_code_sha256:
        raise SystemExit("A1 learner-code provenance digest drift")
    learner_paths: list[str] = []
    for index, record in enumerate(learner_code):
        if not isinstance(record, dict) or set(record) != {"kind", "path", "sha256"}:
            raise SystemExit(
                f"A1 learner-code record {index} has malformed fields"
            )
        if record.get("kind") != "learner_code" or not _is_sha256(
            record.get("sha256")
        ):
            raise SystemExit(f"A1 learner-code record {index} is malformed")
        try:
            code_path = Path(str(record["path"])).expanduser().resolve(strict=True)
        except OSError as error:
            raise SystemExit(
                f"cannot resolve A1 learner-code file {record.get('path')}: {error}"
            ) from error
        if str(code_path) != str(record["path"]):
            raise SystemExit(
                f"A1 learner-code path is not canonical: {record['path']}"
            )
        actual_code_sha = _sha256_existing_file(code_path)
        if actual_code_sha != record["sha256"]:
            raise SystemExit(
                "A1 learner implementation drift before optimizer construction: "
                f"{code_path} declared={record['sha256']!r} actual={actual_code_sha!r}"
            )
        learner_paths.append(code_path.as_posix())
    missing_learner_code = {
        suffix
        for suffix in A1_REQUIRED_LEARNER_CODE_SUFFIXES
        if not any(path.endswith(suffix) for path in learner_paths)
    }
    if missing_learner_code:
        raise SystemExit(
            "A1 contract omits required learner implementation files: "
            f"{sorted(missing_learner_code)}"
        )
    runtime_code_tree = provenance.get("runtime_code_tree")
    if not isinstance(runtime_code_tree, list) or not runtime_code_tree:
        raise SystemExit("A1 contract lock does not bind the transitive runtime tree")
    runtime_code_tree_sha256 = _canonical_json_sha256(runtime_code_tree)
    if provenance.get("runtime_code_tree_sha256") != runtime_code_tree_sha256:
        raise SystemExit("A1 runtime-code-tree provenance digest drift")
    runtime_paths: list[str] = []
    for index, record in enumerate(runtime_code_tree):
        if not isinstance(record, dict) or set(record) != {"kind", "path", "sha256"}:
            raise SystemExit(
                f"A1 runtime-code-tree record {index} has malformed fields"
            )
        if record.get("kind") != "runtime_code" or not _is_sha256(
            record.get("sha256")
        ):
            raise SystemExit(f"A1 runtime-code-tree record {index} is malformed")
        try:
            runtime_path = Path(str(record["path"])).expanduser().resolve(strict=True)
        except OSError as error:
            raise SystemExit(
                f"cannot resolve A1 runtime file {record.get('path')}: {error}"
            ) from error
        if str(runtime_path) != str(record["path"]):
            raise SystemExit(
                f"A1 runtime-code-tree path is not canonical: {record['path']}"
            )
        actual_runtime_sha = _sha256_existing_file(runtime_path)
        if actual_runtime_sha != record["sha256"]:
            raise SystemExit(
                "A1 transitive runtime drift before optimizer construction: "
                f"{runtime_path} declared={record['sha256']!r} "
                f"actual={actual_runtime_sha!r}"
            )
        runtime_paths.append(runtime_path.as_posix())
    missing_runtime_code = {
        suffix
        for suffix in A1_REQUIRED_RUNTIME_CODE_SUFFIXES
        if not any(path.endswith(suffix) for path in runtime_paths)
    }
    if missing_runtime_code:
        raise SystemExit(
            "A1 contract omits required transitive runtime files: "
            f"{sorted(missing_runtime_code)}"
        )
    science = lock_payload.get("science")
    if not isinstance(science, dict):
        raise SystemExit("A1 contract lock has no science section")
    learner_objective = science.get("learner_value_objective")
    if not isinstance(learner_objective, dict) or science.get(
        "learner_value_objective_sha256"
    ) != _canonical_json_sha256(learner_objective):
        raise SystemExit("A1 learner objective binding drift")
    learner_training_recipe = science.get("learner_training_recipe")
    if not isinstance(learner_training_recipe, dict) or science.get(
        "learner_training_recipe_sha256"
    ) != _canonical_json_sha256(learner_training_recipe):
        raise SystemExit("A1 learner training recipe binding drift")
    producer_sha = next(
        (
            record.get("sha256")
            for record in lock_payload.get("checkpoints", [])
            if isinstance(record, dict) and record.get("role") == "producer"
        ),
        None,
    )
    if not _is_sha256(producer_sha):
        raise SystemExit("A1 contract lock has no valid producer checkpoint")
    return {
        "learner_value_objective": learner_objective,
        "learner_training_recipe": learner_training_recipe,
        "learner_training_recipe_sha256": _canonical_json_sha256(
            learner_training_recipe
        ),
        "learner_code_sha256": learner_code_sha256,
        "runtime_code_tree_sha256": runtime_code_tree_sha256,
        "producer_checkpoint_sha256": producer_sha,
        "selected_game_seed_set_sha256": digest_fields[
            "selected_game_seed_set_sha256"
        ],
        "training_game_seed_set_sha256": digest_fields[
            "training_game_seed_set_sha256"
        ],
    }


def _validate_a1_learner_objective(
    args: argparse.Namespace, bound: dict[str, object]
) -> None:
    objective = bound["learner_value_objective"]
    assert isinstance(objective, dict)
    mode = str(getattr(args, "value_head_type", "mse"))
    if mode == "scalar":
        mode = "mse"
    if mode != objective.get("objective"):
        raise SystemExit(
            "A1 learner objective differs from the immutable contract: "
            f"contract={objective.get('objective')!r} cli={mode!r}"
        )
    expected_readout = "scalar" if mode == "mse" else "categorical"
    if objective.get("value_readout") != expected_readout:
        raise SystemExit("A1 learner readout differs from the immutable contract")
    if mode == "mse":
        if int(getattr(args, "value_categorical_bins", 0)) != 0:
            raise SystemExit("A1 scalar learner cannot construct categorical bins")
        scalar_weight, categorical_weight = _resolve_value_objective_weights(args)
        if scalar_weight <= 0.0 or categorical_weight != 0.0:
            raise SystemExit("A1 scalar learner objective weights are contradictory")
    else:
        expected_bins = objective.get("value_categorical_bins")
        if int(getattr(args, "value_categorical_bins", 0)) != int(expected_bins):
            raise SystemExit("A1 categorical bin count differs from the contract")
        if not math.isclose(
            float(getattr(args, "value_hlgauss_sigma_ratio", 0.0)),
            float(objective.get("hlgauss_sigma_ratio")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit("A1 HL-Gauss sigma differs from the contract")
    curriculum_parent = _validate_a1_curriculum_parent(args, bound)
    if (
        getattr(args, "init_checkpoint_sha256", None)
        != bound["producer_checkpoint_sha256"]
        and curriculum_parent is None
    ):
        raise SystemExit(
            "A1 warm-start checkpoint differs from the producer bound by the contract"
        )
    args.a1_curriculum_parent = curriculum_parent


def _validate_a1_curriculum_parent(
    args: argparse.Namespace, bound: dict[str, object]
) -> dict[str, object] | None:
    """Authenticate the completed first dose used by a one-off two-arm curriculum.

    Generation remains bound to its original producer.  This narrow path only
    permits the learner warm start to advance to the exact output of a completed
    sealed n256 dose, so n128 can be the second half of an all-games curriculum.
    Ordinary A1 and production runs retain the historical producer-equality rule.
    """

    raw = str(getattr(args, "a1_curriculum_parent_receipt", "") or "")
    if not raw:
        return None
    path = Path(raw).expanduser().resolve(strict=True)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load A1 curriculum parent receipt: {error}") from error
    if not isinstance(receipt, dict):
        raise SystemExit("A1 curriculum parent receipt must be a JSON object")
    unhashed = dict(receipt)
    stated = unhashed.pop("receipt_sha256", None)
    if (
        receipt.get("schema_version") != "a1-dual-arm-training-receipt-v1"
        or receipt.get("status") != "complete"
        or (receipt.get("arm_id"), receipt.get("subset_id"))
        != ("n256", "full-56k")
        or (bound.get("arm_id"), bound.get("subset_id"))
        != ("n128", "full-140k")
        or stated != _canonical_json_sha256(unhashed)
    ):
        raise SystemExit("A1 curriculum parent receipt schema/status/digest drift")
    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    lineage_dose = receipt.get("lineage_dose")
    producer = inputs.get("producer") if isinstance(inputs, dict) else None
    checkpoint = outputs.get("checkpoint") if isinstance(outputs, dict) else None
    if (
        not isinstance(lineage_dose, dict)
        or lineage_dose.get("schema_version") != "a1-lineage-dose-v1"
        or lineage_dose.get("mode") != "direct_from_declared_producer"
        or lineage_dose.get("optimizer_state_continuity")
        != "fresh_optimizer_per_dose"
        or lineage_dose.get("declared_producer_sha256")
        != bound.get("producer_checkpoint_sha256")
        or lineage_dose.get("init_checkpoint_sha256")
        != bound.get("producer_checkpoint_sha256")
        or isinstance(lineage_dose.get("cumulative_sampled_rows"), bool)
        or not isinstance(lineage_dose.get("cumulative_sampled_rows"), int)
        or lineage_dose["cumulative_sampled_rows"] <= 0
        or isinstance(lineage_dose.get("cumulative_optimizer_steps"), bool)
        or not isinstance(lineage_dose.get("cumulative_optimizer_steps"), int)
        or lineage_dose["cumulative_optimizer_steps"] <= 0
    ):
        raise SystemExit(
            "A1 curriculum parent lacks a typed direct-producer cumulative dose"
        )
    if (
        not isinstance(producer, dict)
        or producer.get("sha256") != bound.get("producer_checkpoint_sha256")
        or not isinstance(checkpoint, dict)
        or set(checkpoint) != {"path", "sha256"}
        or checkpoint.get("sha256") != getattr(args, "init_checkpoint_sha256", None)
        or str(Path(str(checkpoint.get("path"))).expanduser().resolve(strict=True))
        != str(Path(str(args.init_checkpoint)).expanduser().resolve(strict=True))
        or _sha256_existing_file(str(checkpoint.get("path"))) != checkpoint.get("sha256")
    ):
        raise SystemExit("A1 curriculum parent does not bind producer/init checkpoint")
    return {
        "schema_version": "a1-curriculum-parent-binding-v1",
        "receipt_path": str(path),
        "receipt_sha256": _sha256_existing_file(path),
        "parent_arm_id": "n256",
        "parent_subset_id": "full-56k",
        "parent_checkpoint": checkpoint,
        "generation_producer_sha256": producer["sha256"],
    }


def _effective_a1_learner_training_recipe(
    args: argparse.Namespace, ddp: dict[str, int | bool]
) -> dict[str, object]:
    bound_fields = (
        "track",
        "vps_to_win",
        "graph_history_features",
        "seed",
        "epochs",
        "max_steps",
        "batch_size",
        "grad_accum_steps",
        "optimizer",
        "resume_optimizer",
        "lr",
        "lr_warmup_steps",
        "lr_schedule",
        "weight_decay",
        "fused_optimizer",
        "value_lr_mult",
        "action_module_lr_mult",
        "trunk_lr_mult",
        "policy_loss_weight",
        "soft_target_source",
        "soft_target_weight",
        "soft_target_temperature",
        "soft_target_min_legal_coverage",
        "value_loss_weight",
        "value_target_lambda",
        "value_categorical_loss_weight",
        "hlgauss_scalar_aux_loss_weight",
        "final_vp_loss_weight",
        "q_loss_weight",
        "policy_kl_anchor_weight",
        "value_uncertainty_loss_weight",
        "aux_subgoal_loss_weight",
        "train_value_only",
        "freeze_modules",
        "policy_surprise_weight",
        "advantage_policy_weighting",
        "per_game_value_weight",
        "vp_margin_weight",
        "truncated_vp_margin_value_weight",
        "amp",
        "mask_hidden_info",
        "symmetry_augment",
        "forced_action_weight",
        "forced_row_value_weight",
        "winner_sample_weight",
        "loser_sample_weight",
        "teacher_weights",
        "phase_weights",
        "value_phase_weights",
        "ddp_shard_data",
    )
    effective = {field: getattr(args, field) for field in bound_fields}
    world_size = int(ddp["world_size"])
    effective["world_size"] = world_size
    effective["global_batch_size"] = (
        int(args.batch_size) * int(args.grad_accum_steps) * world_size
    )
    if int(getattr(args, "policy_aux_active_batch_size", 0)) > 0:
        effective["policy_aux_active_batch_size"] = int(args.policy_aux_active_batch_size)
    return effective


def _validate_a1_batch_probe_authorization(
    args: argparse.Namespace,
    effective: dict[str, object],
) -> dict[str, object] | None:
    """Authenticate the narrow, non-promotable B200 batch-probe recipe."""
    plan_value = str(getattr(args, "a1_batch_probe_plan", "") or "")
    run_id = str(getattr(args, "a1_batch_probe_run_id", "") or "")
    if not plan_value and not run_id:
        return None
    if not plan_value or not run_id:
        raise SystemExit("A1 batch probe requires both plan and run id")
    plan_path = Path(plan_value).expanduser().resolve(strict=True)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load A1 batch-probe plan: {error}") from error
    if not isinstance(plan, dict):
        raise SystemExit("A1 batch-probe plan is not a JSON object")
    stated = plan.get("plan_sha256")
    replay = _canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if (
        plan.get("schema_version") != "a1-b200-batch-probe-plan-v1"
        or plan.get("diagnostic_only") is not True
        or plan.get("promotion_eligible") is not False
        or stated != replay
    ):
        raise SystemExit("A1 batch-probe plan schema/digest/authority drift")
    matches = [row for row in plan.get("runs", []) if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise SystemExit("A1 batch-probe run id is absent or duplicated")
    run = matches[0]
    command = run.get("command")
    if not isinstance(command, list):
        raise SystemExit("A1 batch-probe command is malformed")
    trainer_indices = [
        index for index, value in enumerate(command) if Path(str(value)).name == "train_bc.py"
    ]
    actual_argv = [str(Path(__file__).resolve()), *sys.argv[1:]]
    if len(trainer_indices) != 1 or command[trainer_indices[0] :] != actual_argv:
        raise SystemExit("A1 batch-probe plan does not bind the executing argv")
    runtime = plan.get("runtime", {})
    if (
        runtime.get("trainer") != str(Path(__file__).resolve())
        or runtime.get("trainer_sha256") != _sha256_existing_file(__file__)
    ):
        raise SystemExit("A1 batch-probe trainer runtime drift")
    receipt_ref = plan.get("midpoint_receipt", {})
    receipt_path = Path(str(receipt_ref.get("path", ""))).expanduser().resolve(strict=True)
    if receipt_ref.get("sha256") != _sha256_existing_file(receipt_path):
        raise SystemExit("A1 batch-probe midpoint receipt file drift")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load A1 batch-probe receipt: {error}") from error
    receipt_unhashed = dict(receipt)
    receipt_stated = receipt_unhashed.pop("receipt_sha256", None)
    if receipt.get("status") != "complete" or receipt_stated != _canonical_json_sha256(
        receipt_unhashed
    ):
        raise SystemExit("A1 batch-probe midpoint receipt digest/status drift")
    authorization = plan.get("batch_probe_authorization", {})
    baseline = receipt.get("inputs", {}).get("learner_ablation", {}).get(
        "effective_recipe"
    )
    allowed = {"batch_size", "global_batch_size", "max_steps"}
    if (
        not isinstance(baseline, dict)
        or authorization.get("baseline_effective_recipe") != baseline
        or authorization.get("baseline_effective_recipe_sha256")
        != _canonical_json_sha256(baseline)
        or set(authorization.get("allowed_recipe_drift", [])) != allowed
    ):
        raise SystemExit("A1 batch-probe baseline recipe authorization drift")
    actual = dict(effective)
    actual["per_game_value_weight_mode"] = str(args.per_game_value_weight_mode)
    if set(actual) != set(baseline):
        raise SystemExit("A1 batch-probe effective recipe shape drift")
    drift = {
        key: {"baseline": baseline[key], "effective": actual[key]}
        for key in baseline
        if baseline[key] != actual[key]
    }
    if not drift or set(drift) - allowed:
        raise SystemExit(
            "A1 batch-probe recipe exceeds its allowed drift: "
            f"drift={sorted(drift)} allowed={sorted(allowed)}"
        )
    return {
        "schema_version": "a1-b200-batch-probe-authorization-v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "plan": str(plan_path),
        "plan_sha256": stated,
        "run_id": run_id,
        "baseline_effective_recipe_sha256": _canonical_json_sha256(baseline),
        "effective_recipe": actual,
        "effective_recipe_sha256": _canonical_json_sha256(actual),
        "recipe_drift": drift,
    }


def _validate_a1_learner_training_recipe(
    args: argparse.Namespace,
    ddp: dict[str, int | bool],
    bound: dict[str, object],
) -> dict[str, object]:
    if bool(getattr(args, "per_game_policy_weight", False)) or str(
        getattr(args, "per_game_policy_weight_mode", "equal")
    ) != "equal":
        raise SystemExit(
            "A1 single-contract learner does not bind per-game policy weighting; "
            "use an explicitly authorized diagnostic recipe"
        )
    expected = bound.get("learner_training_recipe")
    if not isinstance(expected, dict):
        raise SystemExit("A1 contract has no typed learner training recipe")
    immutable_expected = expected
    effective = _effective_a1_learner_training_recipe(args, ddp)
    batch_probe = _validate_a1_batch_probe_authorization(args, effective)
    if batch_probe is not None:
        bound["learner_ablation"] = batch_probe
        bound["batch_probe_authorization"] = batch_probe
        return effective
    missing = set(expected) - set(effective)
    extra = set(effective) - set(expected)
    drift = {
        key: {"contract": expected.get(key), "effective": effective.get(key)}
        for key in sorted(set(expected) & set(effective))
        if expected[key] != effective[key]
    }
    ablation_id = str(getattr(args, "a1_learner_ablation_id", "") or "")
    declared_json = str(
        getattr(args, "a1_effective_learner_recipe_json", "") or ""
    )
    declared_sha = str(
        getattr(args, "a1_effective_learner_recipe_sha256", "") or ""
    )
    code_binding_json = str(
        getattr(args, "a1_ablation_code_binding_json", "") or ""
    )
    declared_code_sha = str(
        getattr(args, "a1_ablation_code_tree_sha256", "") or ""
    )
    reviewed_lock_sha = str(
        getattr(args, "a1_reviewed_lock_file_sha256", "") or ""
    )
    dual_lock_path = str(getattr(args, "a1_dual_learner_lock", "") or "")
    dual_reviewed_sha = str(
        getattr(args, "a1_dual_reviewed_lock_file_sha256", "") or ""
    )
    dual_topology_authorization: dict[str, object] | None = None
    dual_runtime_paths: set[str] | None = None
    if dual_lock_path or dual_reviewed_sha:
        if not dual_lock_path or not _is_sha256(dual_reviewed_sha):
            raise SystemExit(
                "dual learner topology requires lock path and reviewed raw lock digest"
            )
        if bound.get("dual_arm") is not True:
            raise SystemExit("dual learner topology authorization requires a dual-arm corpus")
        from tools import a1_dual_learner_contract as dual_contract

        # The generation-lock replay reaches the canonical promotion handoff,
        # whose registry snapshot is intentionally protected by a nonblocking
        # exclusive lock.  Replaying it independently on every DDP rank races
        # that lock and makes a valid world-size >1 learner nondeterministically
        # refuse before its first optimizer step.  Rank 0 performs the complete
        # byte/lineage replay once; all ranks consume that exact verified value.
        distributed = bool(ddp.get("enabled", False)) and int(
            ddp.get("world_size", 1)
        ) > 1
        rank = int(ddp.get("rank", 0))
        authority_payload: list[dict[str, object] | None] = [None]
        if not distributed or rank == 0:
            try:
                authority_payload[0] = {
                    "authority": dual_contract.verify_lock(
                        Path(dual_lock_path),
                        reviewed_file_sha256=dual_reviewed_sha,
                    )
                }
            # The generation-lock replay can leak PromotionError from the
            # canonical handoff transaction.  Broadcast every ordinary replay
            # exception so rank 0 cannot exit while peers block here.
            except Exception as error:
                # Broadcast the refusal as data so nonzero ranks cannot hang at
                # the collective while rank 0 exits early.
                authority_payload[0] = {"error": str(error)}
        if distributed:
            import torch.distributed as dist

            dist.broadcast_object_list(authority_payload, src=0)
        payload = authority_payload[0]
        if not isinstance(payload, dict):
            raise SystemExit("dual learner topology lock replay returned no authority")
        if "error" in payload:
            raise SystemExit(
                f"dual learner topology lock refused: {payload['error']}"
            )
        authority = payload.get("authority")
        if not isinstance(authority, dict):
            raise SystemExit("dual learner topology lock replay returned malformed authority")
        topology = authority.get("topology")
        if (
            authority.get("arm_id") != bound.get("arm_id")
            or authority.get("subset_id") != bound.get("subset_id")
            or authority.get("recipe") != expected
            or authority.get("objective") != bound.get("learner_value_objective")
            or topology not in dual_contract.TOPOLOGIES.values()
        ):
            raise SystemExit("dual learner topology lock differs from audited corpus")
        runtime = authority.get("runtime")
        if not isinstance(runtime, list) or any(
            not isinstance(record, dict) or not isinstance(record.get("path"), str)
            for record in runtime
        ):
            raise SystemExit("dual learner topology lock has malformed runtime closure")
        dual_runtime_paths = {str(record["path"]) for record in runtime}
        authorized = dict(expected)
        authorized.update(
            {
                "world_size": topology["world_size"],
                "batch_size": topology["local_batch_size"],
                "grad_accum_steps": topology["grad_accum_steps"],
                "global_batch_size": topology["global_batch_size"],
                "ddp_shard_data": topology["ddp_shard_data"],
            }
        )
        dual_topology_authorization = {
            "schema_version": "a1-dual-learner-topology-authorization-v1",
            "learner_lock": str(Path(dual_lock_path).expanduser().resolve(strict=True)),
            "learner_lock_file_sha256": dual_reviewed_sha,
            "topology": topology,
            "effective_recipe": effective,
            "effective_recipe_sha256": _canonical_json_sha256(effective),
        }
        ablation_metadata = (
            declared_json,
            declared_sha,
            code_binding_json,
            declared_code_sha,
            reviewed_lock_sha,
        )
        if not ablation_id:
            if any(ablation_metadata):
                raise SystemExit(
                    "A1 effective-recipe metadata requires a nonempty "
                    "--a1-learner-ablation-id"
                )
            if effective != authorized:
                raise SystemExit("command differs from reviewed dual learner topology")
            bound["learner_topology_authorization"] = dual_topology_authorization
            return effective

        # A diagnostic ablation may alter only the existing learner-ablation
        # allowlist.  The independently reviewed DDP topology remains exact.
        for key in (
            "world_size",
            "batch_size",
            "grad_accum_steps",
            "global_batch_size",
            "ddp_shard_data",
        ):
            if effective.get(key) != authorized.get(key):
                raise SystemExit(
                    f"dual learner ablation cannot alter topology field {key!r}"
                )
        expected = authorized
        missing = set(expected) - set(effective)
        extra = set(effective) - set(expected)
        drift = {
            key: {"contract": expected.get(key), "effective": effective.get(key)}
            for key in sorted(set(expected) & set(effective))
            if expected[key] != effective[key]
        }
    if not ablation_id:
        if (
            declared_json
            or declared_sha
            or code_binding_json
            or declared_code_sha
            or reviewed_lock_sha
        ):
            raise SystemExit(
                "A1 effective-recipe metadata requires a nonempty "
                "--a1-learner-ablation-id"
            )
        if missing or extra or drift:
            raise SystemExit(
                "A1 learner training recipe differs from the immutable one-dose "
                "contract: "
                f"missing={sorted(missing)} extra={sorted(extra)} drift={drift}"
            )
        if bound.get("learner_training_recipe_sha256") != _canonical_json_sha256(
            effective
        ):
            raise SystemExit("A1 effective learner training recipe digest drift")
        bound["learner_ablation"] = None
        return effective

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", ablation_id):
        raise SystemExit(
            "--a1-learner-ablation-id must be 1-80 safe identifier characters"
        )
    effective["per_game_value_weight_mode"] = str(args.per_game_value_weight_mode)
    authorized_extra_fields = {"per_game_value_weight_mode"}
    if int(getattr(args, "policy_aux_active_batch_size", 0)) > 0:
        authorized_extra_fields.add("policy_aux_active_batch_size")
    missing = set(expected) - set(effective)
    extra = set(effective) - (set(expected) | authorized_extra_fields)
    drift = {
        key: {"contract": expected.get(key), "effective": effective.get(key)}
        for key in sorted(set(expected) & set(effective))
        if expected[key] != effective[key]
    }
    if effective["per_game_value_weight_mode"] != "equal":
        drift["per_game_value_weight_mode"] = {
            "contract": "equal (implicit train_bc default; weighting locked off)",
            "effective": effective["per_game_value_weight_mode"],
        }
    if "policy_aux_active_batch_size" in effective:
        drift["policy_aux_active_batch_size"] = {
            "contract": 0,
            "effective": int(effective["policy_aux_active_batch_size"]),
        }
    if not declared_json or not _is_sha256(declared_sha):
        raise SystemExit(
            "A1 learner ablation requires canonical effective recipe JSON and sha256"
        )
    try:
        declared = json.loads(declared_json)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid A1 effective learner recipe JSON: {error}") from error
    if not isinstance(declared, dict):
        raise SystemExit("A1 effective learner recipe must be a JSON object")
    if _canonical_json_sha256(declared) != declared_sha:
        raise SystemExit("A1 declared effective learner recipe digest drift")
    expected_effective_keys = set(expected) | authorized_extra_fields
    if set(declared) != expected_effective_keys:
        raise SystemExit(
            "A1 ablation effective recipe key set differs from bound recipe: "
            f"missing={sorted(expected_effective_keys - set(declared))} "
            f"extra={sorted(set(declared) - expected_effective_keys)}"
        )
    if declared != effective:
        raise SystemExit("A1 command does not match its declared effective ablation recipe")
    if missing or extra:
        raise SystemExit(
            f"A1 effective recipe shape drift: missing={sorted(missing)} extra={sorted(extra)}"
        )
    if not drift:
        raise SystemExit("A1 learner ablation must change at least one recipe field")
    if dual_topology_authorization is not None:
        allowed_dual_drift = {"epochs", "lr", "loser_sample_weight"}
        forbidden_dual_drift = set(drift) - allowed_dual_drift
        if forbidden_dual_drift:
            raise SystemExit(
                "dual corrective ablation only permits epochs, lr, and loser_sample_weight; "
                f"got forbidden fields {sorted(forbidden_dual_drift)}"
            )
        if reviewed_lock_sha != dual_reviewed_sha:
            raise SystemExit(
                "dual corrective ablation must bind the reviewed dual learner lock"
            )
    if bound.get("learner_training_recipe_sha256") != _canonical_json_sha256(
        immutable_expected
    ):
        raise SystemExit("A1 immutable bound learner recipe digest drift")
    try:
        code_binding = json.loads(code_binding_json)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid A1 ablation code binding JSON: {error}") from error
    if not isinstance(code_binding, dict) or not _is_sha256(declared_code_sha):
        raise SystemExit("A1 ablation requires a reviewed code-tree binding/digest")
    if not _is_sha256(reviewed_lock_sha):
        raise SystemExit("A1 ablation requires a reviewed raw lock-file digest")
    unhashed_binding = dict(code_binding)
    embedded_code_sha = unhashed_binding.pop("code_tree_sha256", None)
    if (
        embedded_code_sha != declared_code_sha
        or _canonical_json_sha256(unhashed_binding) != declared_code_sha
    ):
        raise SystemExit("A1 ablation code-tree binding digest drift")
    records = code_binding.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit("A1 ablation code-tree binding has no records")
    seen_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "kind",
            "relative_path",
            "path",
            "sha256",
        }:
            raise SystemExit("A1 ablation code-tree record is malformed")
        path = Path(str(record["path"])).expanduser().resolve(strict=True)
        if str(path) != record["path"] or str(path) in seen_paths:
            raise SystemExit("A1 ablation code-tree path is noncanonical/duplicate")
        seen_paths.add(str(path))
        if _sha256_existing_file(str(path)) != record["sha256"]:
            raise SystemExit(f"A1 ablation code file drift: {path}")
    if dual_runtime_paths is not None and {
        str(record["relative_path"]) for record in records
    } != dual_runtime_paths:
        raise SystemExit(
            "dual corrective ablation code binding differs from reviewed runtime closure"
        )
    bound["learner_ablation"] = {
        "schema_version": "a1-learner-ablation-v1",
        "ablation_id": ablation_id,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "promotion_block_reason": "requires_normal_evidence_packaging_after_ablation",
        "bound_recipe": dict(expected),
        "bound_recipe_sha256": _canonical_json_sha256(expected),
        "effective_recipe": dict(effective),
        "effective_recipe_sha256": declared_sha,
        "recipe_drift": drift,
        "recipe_drift_sha256": _canonical_json_sha256(drift),
        "code_binding": code_binding,
        "code_tree_sha256": declared_code_sha,
        "reviewed_lock_file_sha256": reviewed_lock_sha,
    }
    if dual_topology_authorization is not None:
        dual_topology_authorization = dict(dual_topology_authorization)
        dual_topology_authorization["effective_recipe"] = dict(effective)
        dual_topology_authorization["effective_recipe_sha256"] = (
            _canonical_json_sha256(effective)
        )
        bound["learner_topology_authorization"] = dual_topology_authorization
    return effective


def _resolve_effective_value_categorical_bins(args: argparse.Namespace) -> int:
    """Resolve the fresh/resume/grow categorical-head construction contract.

    ``None`` means inherit for checkpoint-backed runs and disabled for a truly
    fresh run.  Resolving before ``TrainConfig`` construction ensures its hash
    records the effective architecture rather than the CLI sentinel.
    """

    requested_raw = getattr(args, "value_categorical_bins", None)
    requested = None if requested_raw is None else int(requested_raw)
    if requested is not None and (requested < 0 or requested == 1):
        raise SystemExit(
            "--value-categorical-bins must be 0 (disabled) or >=2, got "
            f"{requested}"
        )
    if str(args.arch) != "entity_graph":
        if requested not in (None, 0):
            raise SystemExit(
                "--value-categorical-bins is only supported for --arch entity_graph"
            )
        return 0

    init_checkpoint = str(getattr(args, "init_checkpoint", "") or "")
    grow_checkpoint = str(getattr(args, "grow_from_checkpoint", "") or "")
    if init_checkpoint:
        inherited = _checkpoint_value_categorical_bins(init_checkpoint)
        if requested is not None and requested != inherited:
            raise SystemExit(
                "--value-categorical-bins does not match --init-checkpoint: "
                f"checkpoint={inherited} cli={requested}. Resume uses the checkpoint's "
                "exact architecture; omit the flag to inherit it."
            )
        return inherited
    if grow_checkpoint:
        inherited = _checkpoint_value_categorical_bins(grow_checkpoint)
        return inherited if requested is None else requested
    return 0 if requested is None else requested


def main(argv: Sequence[str] | None = None) -> None:
    checkout_runtime_binding = _assert_checkout_runtime_binding()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.checkout_runtime_binding = checkout_runtime_binding
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    # Resolve file-supplied architecture values before derived defaults and
    # checkpoint inheritance. ``resolve_config`` calls this again later, which
    # is idempotent; doing it here prevents a config's categorical-bin/width
    # value from being mistaken for an omitted flag after we resolve sentinels.
    apply_config_file(
        args,
        parser,
        argv=raw_argv,
        expected_pipeline=TrainConfig.PIPELINE,
    )
    ddp = _distributed_state()
    # Authenticate diagnostic batch/topology drift before the expensive A1
    # memmap preflight.  The same authorization is replayed again when the
    # audited corpus recipe is bound below.
    if args.a1_batch_probe_plan or args.a1_batch_probe_run_id:
        _validate_a1_batch_probe_authorization(
            args,
            _effective_a1_learner_training_recipe(args, ddp),
        )
    # Fail cheap launch/host checks before authenticating a potentially
    # hundreds-of-gigabytes memmap payload. A missing explicit CLI value or a
    # low file-descriptor limit must not be reported only after the byte scan.
    launcher_guards.run_or_refuse(
        _build_guard_specs(args, raw_argv, parser),
        launcher="train_bc",
        skip=bool(args.skip_guards),
    )
    # Check checkpoint/topology compatibility before authenticating the data
    # payload.  A parser default that disagrees with a warm-start checkpoint is
    # a command-construction error and must fail in seconds, not after hashing
    # hundreds of gigabytes.
    if args.hidden_size is None:
        args.hidden_size = (
            640
            if args.arch == "entity_graph"
            else 768
            if args.arch == "xdim_graph"
            else 512
        )
    if args.grow_from_checkpoint and args.init_checkpoint:
        raise SystemExit(
            "--grow-from-checkpoint and --init-checkpoint are mutually exclusive: "
            "--init-checkpoint resumes at the checkpoint's exact architecture, "
            "--grow-from-checkpoint warm-starts a fresh (bigger) architecture"
        )
    if args.grow_from_checkpoint and args.arch != "entity_graph":
        raise SystemExit("--grow-from-checkpoint currently supports --arch entity_graph only")
    args.value_categorical_bins = _resolve_effective_value_categorical_bins(args)
    _preflight_init_checkpoint_architecture(args, ddp)
    a1_preflight_meta: dict[str, object] | None = None
    if args.data_format == "memmap":
        a1_preflight_meta = _coordinated_a1_memmap_preflight(
            args.data,
            validation_manifest_path=args.validation_game_seed_manifest or None,
            ddp=ddp,
        )

    validation_game_seed_ranges = _parse_game_seed_ranges(
        args.validation_game_seed_ranges
    )
    validation_seed_contract: dict[str, object] | None = None
    is_memmap_composite = bool(
        isinstance(a1_preflight_meta, dict)
        and a1_preflight_meta.get("schema_version") in {
            "memmap_composite_v1", "memmap_composite_v2"
        }
    )
    if is_memmap_composite:
        _validate_composite_learner_recipe_authorization(args, a1_preflight_meta)
        if int(args.validation_max_samples) != 0:
            raise SystemExit(
                "authenticated composite validation requires --validation-max-samples 0; "
                "use a whole-game --validation-game-sentinel-manifest instead"
            )
        validation_seed_contract = _load_composite_validation_contract(
            a1_preflight_meta,
            validation_fraction=float(args.validation_fraction),
            validation_seed=int(args.validation_seed),
            validation_max_samples=0,
            validation_game_seed_ranges=validation_game_seed_ranges,
        )
        if args.validation_game_sentinel_manifest:
            validation_seed_contract = _load_composite_validation_sentinel_manifest(
                args.validation_game_sentinel_manifest,
                composite_meta=a1_preflight_meta,
                full_contract=validation_seed_contract,
            )
    elif args.validation_game_seed_manifest:
        if args.validation_game_sentinel_manifest:
            raise SystemExit(
                "--validation-game-sentinel-manifest is supported only for an "
                "authenticated memmap composite"
            )
        if args.data_format != "memmap":
            raise SystemExit(
                "--validation-game-seed-manifest is the audited A1 memmap path; "
                "build the selected corpus first and pass --data-format memmap"
            )
        validation_seed_contract = _load_validation_game_seed_manifest_for_training(
            args.validation_game_seed_manifest,
            validation_fraction=float(args.validation_fraction),
            validation_seed=int(args.validation_seed),
            validation_max_samples=int(args.validation_max_samples),
            validation_game_seed_ranges=validation_game_seed_ranges,
        )
    elif args.validation_game_sentinel_manifest:
        raise SystemExit(
            "--validation-game-sentinel-manifest requires an authenticated memmap composite"
        )

    if args.hidden_size is None:
        args.hidden_size = 640 if args.arch == "entity_graph" else 768 if args.arch == "xdim_graph" else 512
    if args.require_production_35m_teacher:
        args.require_35m_model = True
    if args.require_35m_model and not args.skip_teacher_quality_gate:
        args.require_strict_35m_teacher = True
    if args.skip_teacher_quality_gate and (
        args.require_strict_35m_teacher or args.require_production_35m_teacher
    ):
        raise SystemExit(
            "--skip-teacher-quality-gate cannot be combined with strict/production "
            "teacher quality gates"
        )
    # CAT-128 item 5: the strict/production teacher-quality gate runs
    # report_teacher_data_quality.py, which discovers NPZ teacher shards -- it cannot
    # read a --data-format memmap corpus (it globs *.npz, finds none -> "no teacher
    # shards found" and aborts deep in a subprocess; this was a CAT-109 failure mode).
    # For a curated memmap corpus the correct guard is --skip-teacher-quality-gate
    # --trust-curated-data-quality (the corpus was gated when it was built). Fail fast
    # with that guidance rather than let the subprocess abort confusingly.
    if args.data_format == "memmap" and (
        args.require_strict_35m_teacher or args.require_production_35m_teacher
    ):
        raise SystemExit(
            "--require-strict-35m-teacher/--require-production-35m-teacher are "
            "incompatible with --data-format memmap: the external teacher-quality "
            "report globs NPZ shards, which a memmap corpus does not have. For a "
            "curated memmap corpus use --skip-teacher-quality-gate "
            "--trust-curated-data-quality instead."
        )
    # C1 grad-accum / grow / FSDP validation (kept next to the other early
    # argument-consistency checks so a misconfigured launch fails before any
    # data load or GPU allocation).
    if int(args.grad_accum_steps) < 1:
        raise SystemExit("--grad-accum-steps must be >= 1")
    if int(args.epochs) < 1:
        raise SystemExit(
            "--epochs must be >= 1; a zero-epoch run cannot produce a trained "
            "checkpoint or value-readout provenance"
        )
    if int(args.max_steps) < 0:
        raise SystemExit("--max-steps must be >= 0")
    if int(args.policy_aux_active_batch_size) < 0:
        raise SystemExit("--policy-aux-active-batch-size must be >= 0")
    if int(args.policy_aux_active_batch_size) > 0:
        if args.arch not in {"xdim_graph", "entity_graph"}:
            raise SystemExit(
                "--policy-aux-active-batch-size requires --arch entity_graph/xdim_graph"
            )
        if int(args.grad_accum_steps) != 1:
            raise SystemExit(
                "--policy-aux-active-batch-size requires --grad-accum-steps 1"
            )
        if not is_memmap_composite:
            raise SystemExit(
                "--policy-aux-active-batch-size requires an authenticated "
                "memmap_composite_v2 descriptor"
            )
        if float(args.policy_loss_weight) <= 0.0:
            raise SystemExit(
                "--policy-aux-active-batch-size requires positive --policy-loss-weight"
            )
    if int(args.grad_accum_steps) > 1 and args.arch not in {"xdim_graph", "entity_graph"}:
        raise SystemExit(
            "--grad-accum-steps > 1 is only supported for --arch entity_graph/"
            "xdim_graph (the _train_xdim_batch trainer); other archs must use "
            "--grad-accum-steps 1"
        )
    if args.grow_from_checkpoint and args.init_checkpoint:
        raise SystemExit(
            "--grow-from-checkpoint and --init-checkpoint are mutually exclusive: "
            "--init-checkpoint resumes at the checkpoint's exact architecture, "
            "--grow-from-checkpoint warm-starts a fresh (bigger) architecture"
        )
    if args.grow_from_checkpoint and args.arch != "entity_graph":
        raise SystemExit("--grow-from-checkpoint currently supports --arch entity_graph only")
    args.value_categorical_bins = _resolve_effective_value_categorical_bins(args)
    args.data_fingerprint = _training_data_fingerprint(args.data, args.data_format)
    args.a1_memmap_payload_inventory_sha256 = (
        None
        if a1_preflight_meta is None
        else a1_preflight_meta["payload_inventory_sha256"]
    )
    args.init_checkpoint_sha256 = _sha256_existing_file(args.init_checkpoint)
    args.grow_from_checkpoint_sha256 = _sha256_existing_file(
        args.grow_from_checkpoint
    )
    # CAT-66 typed config + config-hash. Built after --hidden-size resolution so
    # the recorded width is the effective one; registered only on rank 0 to avoid
    # duplicate JSONL writes under DDP. A pure no-op to the run when no --config*
    # flag is passed (see catan_zero.rl.config_cli).
    train_config = resolve_config(
        args,
        TrainConfig.from_namespace,
        parser=parser,
        argv=raw_argv,
        register=(ddp["rank"] == 0),
    )
    train_config_hash = train_config.config_hash()
    resume_recipe_identity = _training_resume_recipe_identity(train_config, args, ddp)
    _train_lock = None
    if not args.allow_concurrent_bc:
        # Retain the lock object for the entire training lifetime.
        _train_lock = _acquire_host_train_lock(args.host_lock_file, ddp)

    import torch

    if ddp["enabled"] and args.arch not in {"xdim_lite", "xdim_graph", "entity_graph"}:
        raise SystemExit("DDP behavior cloning currently supports XDim/entity architectures")
    if bool(args.fsdp):
        if not ddp["enabled"]:
            raise SystemExit(
                "--fsdp requires a multi-rank launch (torchrun --nproc_per_node>1); "
                "WORLD_SIZE==1 has nothing to shard"
            )
        if args.arch not in {"xdim_graph", "entity_graph"}:
            raise SystemExit("--fsdp currently supports --arch entity_graph/xdim_graph only")
    if ddp["enabled"]:
        import torch.distributed as dist

        torch.cuda.set_device(ddp["local_rank"])
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{ddp['local_rank']}"
    if args.amp == "bf16":
        torch.set_float32_matmul_precision("high")
        if str(args.device).startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise SystemExit("--amp bf16 requested but CUDA device lacks BF16 support")

    if args.skip_teacher_quality_gate:
        _rank0_print(
            json.dumps(
                {
                    "progress": "teacher_quality_gate",
                    "skipped": True,
                    "reason": "caller asserted this curated dataset was already gated",
                },
                sort_keys=True,
            ),
            ddp,
        )
    else:
        _run_teacher_quality_gate(
            Path(args.data),
            track=args.track,
            vps_to_win=args.vps_to_win,
            strict=bool(args.require_strict_35m_teacher),
            production=bool(args.require_production_35m_teacher),
            soft_target_min_legal_coverage=float(args.soft_target_min_legal_coverage),
            out_path=Path(args.report).with_suffix(".teacher_quality.json"),
            ddp=ddp,
        )

    global _MASK_HIDDEN_INFO_PLAYER_TOKENS
    if bool(args.mask_hidden_info) and args.arch != "entity_graph":
        raise SystemExit("--mask-hidden-info requires --arch entity_graph (masks player_tokens)")
    _MASK_HIDDEN_INFO_PLAYER_TOKENS = bool(args.mask_hidden_info)

    rng = np.random.default_rng(args.seed)
    if args.data_format == "memmap":
        if bool(args.ddp_shard_data):
            raise SystemExit(
                "--ddp-shard-data is not supported with --data-format memmap; the "
                "memmap corpus is streamed per batch, so per-rank RAM is already "
                "bounded and DDP batch sharding is handled at the index level."
            )
        data = load_teacher_data_memmap(
            Path(args.data),
            composite_meta=(a1_preflight_meta if is_memmap_composite else None),
        )
    else:
        data = load_teacher_data(Path(args.data), ddp=ddp, shard_data=bool(args.ddp_shard_data), mask_hidden_info=bool(getattr(args, "mask_hidden_info", False)))
    target_information_admission = _validate_target_information_admission(
        data,
        mask_hidden_info=bool(args.mask_hidden_info),
        soft_target_weight=float(args.soft_target_weight),
        policy_loss_weight=float(args.policy_loss_weight),
        q_loss_weight=float(args.q_loss_weight),
        value_target_lambda=float(args.value_target_lambda),
        policy_kl_anchor_weight=float(args.policy_kl_anchor_weight),
        policy_surprise_weight=float(args.policy_surprise_weight),
    )
    _rank0_print(
        json.dumps(
            {"progress": "target_information_admission", **target_information_admission},
            sort_keys=True,
        ),
        ddp,
    )
    a1_training_binding: dict[str, object] | None = None
    if is_memmap_composite:
        # Existing promotion receipts bind one A1 learner contract. Both
        # component payloads/holdouts are authenticated, but this mixed run is
        # diagnostic-only until a contract binds the ordered component set.
        args.a1_contract_sha256 = validation_seed_contract["a1_contract_sha256"]
        component_seed_sets: list[set[int]] = []
        for index, (corpus, contract) in enumerate(
            zip(data.corpora, validation_seed_contract["component_contracts"])
        ):
            component_seeds = set(
                map(int, np.unique(np.asarray(corpus["game_seed"], dtype=np.int64)))
            )
            for prior in component_seed_sets:
                overlap = prior & component_seeds
                if overlap:
                    raise SystemExit(
                        "memmap composite corpus game seeds overlap across components: "
                        f"component={index} overlap_count={len(overlap)}"
                    )
            component_seed_sets.append(component_seeds)
            heldout_rows = int(np.isin(
                np.asarray(corpus["game_seed"], dtype=np.int64),
                np.asarray(contract["game_seeds"], dtype=np.int64),
            ).sum())
            if heldout_rows != int(contract["validation_row_count"]):
                raise SystemExit(
                    "memmap composite component holdout row count drift: "
                    f"component={index} corpus={heldout_rows} "
                    f"manifest={contract['validation_row_count']}"
                )
    elif validation_seed_contract is not None:
        _validate_a1_validation_manifest_corpus_binding(
            getattr(data, "meta", None), validation_seed_contract
        )
        a1_training_binding = _validate_a1_corpus_artifacts_and_seeds(
            getattr(data, "meta"),
            validation_seed_contract,
            np.asarray(data["game_seed"], dtype=np.int64),
        )
        _validate_a1_learner_objective(args, a1_training_binding)
        a1_training_binding["effective_learner_training_recipe"] = (
            _validate_a1_learner_training_recipe(args, ddp, a1_training_binding)
        )
        args.a1_contract_sha256 = validation_seed_contract["a1_contract_sha256"]
        args.a1_selected_game_seed_set_sha256 = a1_training_binding[
            "selected_game_seed_set_sha256"
        ]
        args.a1_training_game_seed_set_sha256 = a1_training_binding[
            "training_game_seed_set_sha256"
        ]
        args.a1_learner_training_recipe_sha256 = a1_training_binding[
            "learner_training_recipe_sha256"
        ]
        args.a1_learner_code_sha256 = a1_training_binding[
            "learner_code_sha256"
        ]
        args.a1_runtime_code_tree_sha256 = a1_training_binding[
            "runtime_code_tree_sha256"
        ]
        args.a1_learner_ablation = a1_training_binding.get("learner_ablation")
    env_config = _env_config_for_teacher_data(args, data, ddp)
    if args.trust_curated_data_quality:
        data_quality = {
            "samples": int(len(data["action_taken"])),
            "invalid_teacher_actions": 0,
            "trusted_curated_data_quality": True,
            "quality_report_skipped": True,
            "reason": "caller asserted corpus already passed external quality gate",
        }
    else:
        data_quality = teacher_data_quality(
            data,
            q_skip_teacher_prefixes=_parse_prefixes(args.q_skip_teacher_prefixes),
            soft_target_temperature=args.soft_target_temperature,
            soft_target_source=args.soft_target_source,
            soft_target_min_legal_coverage=args.soft_target_min_legal_coverage,
        )
    if float(args.q_loss_weight) != 0.0 and not args.allow_teacher_score_q_loss:
        raise SystemExit(
            "--q-loss-weight trains q_values on normalized teacher preference scores, "
            "but PPO treats q_values as return-scale action values. Use "
            "--q-loss-weight 0 for the PPO warm-start checkpoint, or pass "
            "--allow-teacher-score-q-loss only for an explicitly non-PPO teacher-score "
            "experiment."
        )
    if ddp["enabled"]:
        from torch.nn.parallel import DistributedDataParallel
    _rank0_print(
        json.dumps({"progress": "bc_data_quality", **data_quality}, sort_keys=True),
        ddp,
    )
    # AUDIT FIX (truncated-value default): surface truncation fraction + the
    # effective F3 handling in train.log up front, since the CLI default now
    # feeds truncated rows into the value loss (see --truncated-vp-margin-value-
    # weight help) and this is easy to miss without an explicit startup line.
    truncated_vp_margin_value_weight = float(args.truncated_vp_margin_value_weight)
    _rank0_print(
        json.dumps(
            {
                "progress": "bc_truncated_value_handling",
                "truncated_fraction": data_quality.get("truncated_fraction"),
                "truncated_vp_margin_value_weight": truncated_vp_margin_value_weight,
                "truncated_rows_contribute_value_signal": truncated_vp_margin_value_weight > 0.0,
            },
            sort_keys=True,
        ),
        ddp,
    )
    (
        resolved_scalar_value_weight,
        resolved_categorical_value_weight,
    ) = _resolve_value_objective_weights(args)
    value_root_blend_regime = _resolve_value_root_blend_regime(args)
    _rank0_print(
        json.dumps(
            {
                "progress": "value_objective",
                "value_head_type": str(args.value_head_type),
                "primary_value_loss_weight": float(args.value_loss_weight),
                "resolved_scalar_mse_weight": resolved_scalar_value_weight,
                "resolved_categorical_ce_weight": resolved_categorical_value_weight,
                "hlgauss_scalar_aux_loss_weight": float(
                    args.hlgauss_scalar_aux_loss_weight
                ),
            },
            sort_keys=True,
        ),
        ddp,
    )
    if args.arch == "candidate":
        policy = create_ppo_policy(
            config=env_config,
            seed=args.seed,
            hidden_size=args.hidden_size,
            architecture="candidate",
            device=args.device,
        )
        if float(args.value_lr_mult) != 1.0:
            raise SystemExit(
                "--value-lr-mult is only supported for --arch entity_graph/xdim_lite/"
                "xdim_graph (it needs a named value_head/final_vp_head/"
                "value_uncertainty_head submodule to split into its own param group), "
                "not --arch candidate"
            )
        if float(args.trunk_lr_mult) != 1.0:
            raise SystemExit(
                "--trunk-lr-mult is supported only for --arch entity_graph, not "
                "--arch candidate"
            )
        params = []
        for name in ("model", "actor", "action_encoder", "action_id_embedding", "action_bias"):
            module = getattr(policy, name, None)
            if module is not None:
                params.extend(module.parameters())
        optimizer = _make_optimizer(params, args, getattr(policy, "device", args.device))
        train_fn = _train_candidate_batch
    else:
        if args.init_checkpoint:
            if args.arch == "entity_graph":
                policy = EntityGraphPolicy.load(
                    args.init_checkpoint,
                    device=args.device,
                    strict_metadata=not bool(args.allow_legacy_action_mask_upgrade),
                )
            else:
                policy = XDimLitePolicy.load(
                    args.init_checkpoint,
                    device=args.device,
                    strict_metadata=not bool(args.allow_legacy_action_mask_upgrade),
                )
            if getattr(policy, "policy_type", None) != args.arch:
                raise SystemExit(
                    f"--arch {args.arch} does not match init checkpoint policy_type "
                    f"{getattr(policy, 'policy_type', None)}"
                )
            _assert_init_config_matches(policy, args)
        else:
            if args.arch == "entity_graph":
                policy = EntityGraphPolicy.create(
                    env_config=env_config,
                    hidden_size=args.hidden_size,
                    state_layers=args.graph_layers,
                    attention_heads=args.attention_heads,
                    dropout=args.graph_dropout,
                    seed=args.seed,
                    device=args.device,
                    value_uncertainty_head=bool(args.value_uncertainty_head),
                    value_categorical_bins=int(args.value_categorical_bins),
                    edge_policy_head=bool(getattr(args, "edge_policy_head", False)),
                    aux_subgoal_heads=bool(getattr(args, "aux_subgoal_heads", False)),
                    state_trunk=str(args.entity_state_trunk),
                    relational_block_pattern=str(args.relational_block_pattern),
                    relational_ff_size=int(args.relational_ff_size),
                    relational_bases=int(args.relational_bases),
                    relational_action_cross_layers=int(
                        args.relational_action_cross_layers
                    ),
                    relational_edge_policy_head=bool(
                        args.relational_edge_policy_head
                    ),
                    latent_deliberation_steps=int(args.latent_deliberation_steps),
                    latent_deliberation_slots=int(args.latent_deliberation_slots),
                    moe_routed_experts=int(args.moe_routed_experts),
                    moe_top_k=int(args.moe_top_k),
                    moe_expert_ff_size=int(args.moe_expert_ff_size),
                )
            elif args.arch == "xdim_graph":
                policy = XDimGraphPolicy.create(
                    env_config=env_config,
                    hidden_size=args.hidden_size,
                    seed=args.seed,
                    device=args.device,
                    token_count=args.graph_tokens,
                    board_layers=args.graph_layers,
                    attention_heads=args.attention_heads,
                    dropout=args.graph_dropout,
                )
            else:
                policy = XDimLitePolicy.create(
                    env_config=env_config,
                    hidden_size=args.hidden_size,
                    seed=args.seed,
                    device=args.device,
                )
        _ensure_policy_action_mask_version(
            policy,
            env_config,
            allow_legacy_upgrade=bool(args.allow_legacy_action_mask_upgrade),
            checkpoint_path=args.init_checkpoint or "",
        )
        if args.grow_from_checkpoint:
            grow_report = _warm_start_grow(
                policy, args.grow_from_checkpoint, device=args.device
            )
            _rank0_print(
                json.dumps({"progress": "warm_start_grow", **grow_report}, sort_keys=True),
                ddp,
            )
        _enforce_35m_model_size(policy, args)
        _assert_value_heads_present_for_losses(policy.model, args)
        if (
            resolved_categorical_value_weight > 0.0
            and resolved_scalar_value_weight == 0.0
        ):
            _set_scalar_value_head_trainable(policy.model, False)
            _rank0_print(
                json.dumps(
                    {
                        "progress": "scalar_value_head",
                        "trainable": False,
                        "reason": "categorical-primary with scalar auxiliary weight 0",
                    },
                    sort_keys=True,
                ),
                ddp,
            )
        if float(args.q_loss_weight) == 0.0:
            _set_xdim_q_branch_trainable(policy.model, False)
            _rank0_print(
                json.dumps(
                    {
                        "progress": "q_branch",
                        "trainable": False,
                        "reason": "q_loss_weight=0",
                    },
                    sort_keys=True,
                ),
                ddp,
            )
        freeze_module_groups = set(_parse_prefixes(args.freeze_modules))
        if bool(args.train_value_only):
            freeze_module_groups |= {"trunk", "action_encoder", "policy_head"}
        if freeze_module_groups:
            if args.arch != "entity_graph":
                raise SystemExit(
                    "--freeze-modules/--train-value-only currently only supports "
                    "--arch entity_graph"
                )
            touched = _set_entity_graph_modules_trainable(
                policy.model, freeze_module_groups, trainable=False
            )
            _rank0_print(
                json.dumps(
                    {
                        "progress": "freeze_modules",
                        "frozen_groups": sorted(freeze_module_groups),
                        "frozen_submodules": touched,
                    },
                    sort_keys=True,
                ),
                ddp,
            )
        if bool(args.fsdp):
            # C1 minimal FSDP: FULL_SHARD across ranks, transformer blocks
            # auto-wrapped by module type. use_orig_params=True so
            # named_parameters()/named_modules() keep their original names -- the
            # value/final-vp/uncertainty param-group split in
            # _build_optimizer_param_groups and the module-freeze helpers keep
            # working unchanged. Mixed precision is left to the existing
            # _amp_context autocast (same as the DDP path): FSDP shards params in
            # fp32 and compute autocasts to bf16, so we do NOT also pass an FSDP
            # MixedPrecision policy (that would double-cast). Grad clipping under
            # FSDP must go through FSDP.clip_grad_norm_ (a collective) -- handled
            # in _train_xdim_batch -- and the checkpoint is gathered to a full
            # rank-0 state_dict in _save_policy.
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
            )
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy

            block_types: set = set()
            if hasattr(policy.model, "blocks"):
                for _block in policy.model.blocks:
                    block_types.add(type(_block))
            if getattr(policy.model, "action_cross_attention_layers", 0) and hasattr(
                policy.model, "action_cross_blocks"
            ):
                for _block in policy.model.action_cross_blocks:
                    block_types.add(type(_block))
            wrap_policy = ModuleWrapPolicy(block_types) if block_types else None
            # FSDP cannot flatten 0-dim (scalar) parameters (e.g. the entity_graph
            # net's logit_scale). Keep any scalar params OUT of FSDP via
            # ignored_states -- they stay replicated (unsharded) on every rank, a
            # negligible cost. FSDP will not all-reduce their gradients, so
            # _clip_grad_norm averages them across ranks before the step to keep
            # ranks in lockstep (see policy._fsdp_ignored_params).
            ignored_scalar_params = [
                p for p in policy.model.parameters() if p.dim() == 0
            ]
            _rank0_print(
                json.dumps(
                    {
                        "progress": "fsdp_wrap",
                        "sharding": "FULL_SHARD",
                        "wrapped_block_types": sorted(t.__name__ for t in block_types),
                        "ignored_scalar_params": int(len(ignored_scalar_params)),
                        "amp": str(args.amp),
                    },
                    sort_keys=True,
                ),
                ddp,
            )
            policy.model = FSDP(
                policy.model,
                auto_wrap_policy=wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=ddp["local_rank"],
                use_orig_params=True,
                ignored_states=ignored_scalar_params or None,
            )
            # Same Parameter objects survive the wrap (ignored -> unsharded), so
            # keep references for the manual gradient all-reduce at step time.
            policy._fsdp_ignored_params = ignored_scalar_params
        elif ddp["enabled"]:
            _rank0_print(
                json.dumps(
                    {
                        "progress": "ddp_wrap",
                        "find_unused_parameters": bool(args.ddp_find_unused_parameters),
                    },
                    sort_keys=True,
                ),
                ddp,
            )
            policy.model = DistributedDataParallel(
                policy.model,
                device_ids=[ddp["local_rank"]],
                output_device=ddp["local_rank"],
                find_unused_parameters=bool(args.ddp_find_unused_parameters),
            )
        policy.model.train()
        optimizer_params = _build_optimizer_param_groups(
            policy.model,
            base_lr=float(args.lr),
            value_lr_mult=float(args.value_lr_mult),
            action_module_lr_mult=float(args.action_module_lr_mult),
            trunk_lr_mult=float(args.trunk_lr_mult),
            architecture=str(args.arch),
        )
        _rank0_print(
            json.dumps(
                {
                    "progress": "optimizer_param_groups",
                    "groups": _optimizer_param_group_report(
                        optimizer_params, base_lr=float(args.lr)
                    ),
                },
                sort_keys=True,
            ),
            ddp,
        )
        optimizer = _make_optimizer(
            optimizer_params,
            args,
            getattr(policy, "device", args.device),
        )
        train_fn = _train_xdim_batch

    validate_teacher_data_schema(policy, data, data_quality, env_config)

    # CAT-128 patch #8: resume optimizer (Adam) moment state from the --init-checkpoint's
    # sidecar so a stop/crash continues with warm moments + correct LR position rather
    # than restarting Adam from zero (the fresh-Adam catastrophic-forgetting risk called
    # out by the --lr-warmup-steps guard). Called on ALL ranks (FSDP restore is a
    # collective) BEFORE the training loop, after the optimizer is built. Fail-safe: the
    # champion and any --grow-from-checkpoint arm have no matching sidecar, so this
    # returns False and training proceeds with a fresh optimizer -- the expected first
    # fine-tune behaviour; the win is on resuming a run this code started.
    optimizer_restored = False
    resume_progress = None
    if args.init_checkpoint and bool(args.resume_optimizer):
        from catan_zero.rl.optim_state import (
            TrainingProgressError,
            load_optimizer_state,
            load_training_progress,
            optimizer_sidecar_path,
            training_progress_sidecar_path,
        )

        optimizer_path = optimizer_sidecar_path(args.init_checkpoint)
        progress_path = training_progress_sidecar_path(args.init_checkpoint)
        if optimizer_path.exists() or progress_path.exists():
            if not optimizer_path.exists() or not progress_path.exists():
                raise SystemExit(
                    "incomplete resumable checkpoint set: model, optimizer, and "
                    "training-progress sidecars must all belong to one atomic commit; "
                    "use --no-resume-optimizer for an explicit fresh-optimizer start"
                )
            try:
                resume_progress = load_training_progress(
                    args.init_checkpoint,
                    expected_recipe_identity=resume_recipe_identity,
                )
            except TrainingProgressError as error:
                raise SystemExit(
                    f"refusing incompatible optimizer resume: {error}; use "
                    "--no-resume-optimizer for an explicit fresh-optimizer start"
                ) from error
            optimizer_restored = bool(
                load_optimizer_state(args.init_checkpoint, policy.model, optimizer, ddp)
            )
            if not optimizer_restored:
                raise SystemExit(
                    "training progress validated but optimizer restore failed; refusing "
                    "to restart Adam/LR state silently (use --no-resume-optimizer for "
                    "an explicit fresh-optimizer start)"
                )
        _rank0_print(
            json.dumps(
                {
                    "progress": "optimizer_resume",
                    "restored": optimizer_restored,
                    "optimizer_step": (
                        None
                        if resume_progress is None
                        else int(resume_progress["optimizer_step"])
                    ),
                },
                sort_keys=True,
            ),
            ddp,
        )
    elif args.init_checkpoint:
        _rank0_print(
            json.dumps(
                {
                    "progress": "optimizer_resume",
                    "restored": False,
                    "disabled": True,
                },
                sort_keys=True,
            ),
            ddp,
        )
    if (
        args.init_checkpoint
        and not optimizer_restored
        and int(args.lr_warmup_steps) == 0
    ):
        _rank0_print(
            "WARNING: warm-starting with fresh optimizer moments and "
            "--lr-warmup-steps 0 risks catastrophic forgetting; consider a "
            "500-1000 step warmup.",
            ddp,
        )

    symmetry = None
    symmetry_rng = None
    if getattr(args, "symmetry_augment", False):
        if args.arch != "entity_graph":
            raise SystemExit("--symmetry-augment requires --arch entity_graph")
        from catan_zero.rl.hex_symmetry import build_hex_symmetry

        symmetry = build_hex_symmetry()
        symmetry_rng = np.random.default_rng(int(args.seed) + 20260705)
        _rank0_print(
            json.dumps(
                {
                    "progress": "symmetry_augment",
                    "n_symmetries": int(symmetry.fwd_hex.shape[0]),
                    "relabel_events": bool(args.symmetry_augment_events),
                },
                sort_keys=True,
            ),
            ddp,
        )

    start = time.perf_counter()
    metrics = []
    n = len(data["action_taken"])
    teacher_weight_map = _parse_weight_map(args.teacher_weights)
    policy_phase_weight_map = _parse_weight_map(args.phase_weights)
    value_phase_weights_raw = args.value_phase_weights or args.phase_weights
    value_phase_weight_map = _parse_weight_map(value_phase_weights_raw)

    def _build_derived_training_arrays() -> dict[str, np.ndarray]:
        """Build deterministic O(rows) learner arrays once per host."""

        policy_weights = build_sample_weights(
            data,
            teacher_weights=teacher_weight_map,
            phase_weights=policy_phase_weight_map,
            forced_action_weight=args.forced_action_weight,
            winner_sample_weight=args.winner_sample_weight,
            loser_sample_weight=args.loser_sample_weight,
            vp_margin_weight=args.vp_margin_weight,
            vps_to_win=args.vps_to_win,
            per_game_policy_weight=bool(args.per_game_policy_weight),
            per_game_policy_weight_mode=str(args.per_game_policy_weight_mode),
        )
        policy_weights = _apply_authenticated_policy_distillation_scope(
            data, policy_weights
        )
        value_weights = build_value_sample_weights(
            data,
            phase_weights=value_phase_weight_map,
            forced_row_value_weight=args.forced_row_value_weight,
            per_game_value_weight=args.per_game_value_weight,
            per_game_value_weight_mode=args.per_game_value_weight_mode,
        )
        # Flag-off must remain exact rng.permutation, so these all-ones are only
        # retained for reporting and are not passed to _epoch_order.
        if float(args.policy_surprise_weight) == 0.0:
            surprise_weights = np.ones(n, dtype=np.float32)
        else:
            surprise_kl, surprise_has_prior = compute_policy_surprise_kl(data)
            surprise_weights = policy_surprise_sampling_weights(
                surprise_kl,
                surprise_has_prior,
                weight_scale=args.policy_surprise_weight,
                cap=args.policy_surprise_cap,
            )
        split_result = split_train_validation_indices(
            data,
            validation_fraction=args.validation_fraction,
            validation_seed=args.validation_seed,
            validation_max_samples=args.validation_max_samples,
            validation_game_seed_ranges=validation_game_seed_ranges,
            validation_game_seeds=(
                None
                if validation_seed_contract is None
                else np.asarray(
                    validation_seed_contract["game_seeds"], dtype=np.int64
                )
            ),
            training_excluded_game_seeds=(
                None
                if validation_seed_contract is None
                else np.asarray(
                    validation_seed_contract.get(
                        "excluded_game_seeds", validation_seed_contract["game_seeds"]
                    ),
                    dtype=np.int64,
                )
            ),
            allow_missing_game_seed=bool(
                args.allow_missing_game_seed_validation_split
            ),
        )
        result = {
            "policy_sample_weights": policy_weights,
            "value_sample_weights": value_weights,
            "policy_surprise_weights_full": surprise_weights,
            "train_indices": np.asarray(split_result["train"], dtype=np.int64),
            "validation_indices": np.asarray(
                split_result["validation"], dtype=np.int64
            ),
        }
        if "game_seed" in data:
            result["held_out_game_seeds"] = np.sort(
                np.unique(
                    np.asarray(
                        data["game_seed"][result["validation_indices"]],
                        dtype=np.int64,
                    )
                )
            ).astype(np.int64, copy=False)
        component_weights = _composite_game_sampling_weights(
            data, result["train_indices"]
        )
        if component_weights is not None:
            result["component_game_sampling"] = component_weights
        return result

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", ddp["world_size"]))
    cache_enabled = (
        bool(ddp["enabled"])
        and args.data_format == "memmap"
        and int(ddp["world_size"]) == local_world_size
    )
    if cache_enabled:
        cache_payload = {
            "data_fingerprint": str(args.data_fingerprint),
            "row_count": int(n),
            "teacher_weights": teacher_weight_map,
            "policy_phase_weights": policy_phase_weight_map,
            "value_phase_weights": value_phase_weight_map,
            "forced_action_weight": float(args.forced_action_weight),
            "winner_sample_weight": float(args.winner_sample_weight),
            "loser_sample_weight": float(args.loser_sample_weight),
            "vp_margin_weight": float(args.vp_margin_weight),
            "vps_to_win": int(args.vps_to_win),
            "per_game_policy_weight": bool(args.per_game_policy_weight),
            "per_game_policy_weight_mode": str(args.per_game_policy_weight_mode),
            "forced_row_value_weight": float(args.forced_row_value_weight),
            "per_game_value_weight": bool(args.per_game_value_weight),
            "per_game_value_weight_mode": str(args.per_game_value_weight_mode),
            "policy_surprise_weight": float(args.policy_surprise_weight),
            "policy_surprise_cap": float(args.policy_surprise_cap),
            "validation_fraction": float(args.validation_fraction),
            "validation_seed": int(args.validation_seed),
            "validation_max_samples": int(args.validation_max_samples),
            "validation_game_seed_ranges": validation_game_seed_ranges or [],
            "allow_missing_game_seed_validation_split": bool(
                args.allow_missing_game_seed_validation_split
            ),
            "validation_contract_file_sha256": (
                ""
                if validation_seed_contract is None
                else str(validation_seed_contract.get("file_sha256", ""))
            ),
            "validation_game_seed_set_sha256": (
                ""
                if validation_seed_contract is None
                else str(
                    validation_seed_contract.get(
                        "validation_game_seed_set_sha256", ""
                    )
                )
            ),
            "component_game_sampling_ratios": [
                float(value)
                for value in getattr(
                    data, "component_game_sampling_ratios", tuple()
                )
            ],
            "policy_distillation_component_indices": [
                int(value)
                for value in getattr(
                    data, "policy_distillation_component_indices", tuple()
                )
            ],
        }
        _cache_key, cache_identity = _derived_array_cache_key(cache_payload)
        cache_root = Path(
            os.environ.get(
                "CATAN_TRAIN_PRECOMPUTE_CACHE_DIR",
                "/dev/shm/catan-zero-train-precompute",
            )
        )
        if int(ddp["rank"]) == 0:
            cache_directory = cache_root / _cache_key
            if not cache_directory.exists():
                _write_derived_array_cache(
                    cache_root, cache_identity, _build_derived_training_arrays()
                )
            # One local rank authenticates every byte before releasing peers.
            # Peers subsequently map the exact same read-only files; rehashing
            # ~1 GiB eight times would recreate the startup bottleneck.
            derived = _load_derived_array_cache(cache_root, cache_identity)
        import torch.distributed as dist

        dist.barrier()
        if int(ddp["rank"]) != 0:
            derived = _load_derived_array_cache(
                cache_root, cache_identity, verify_digests=False
            )
        _rank0_print(
            json.dumps(
                {
                    "progress": "training_precompute_cache",
                    "cache_key": _cache_key,
                    "cache_root": str(cache_root),
                    "array_count": len(derived),
                    "shared_across_local_ddp_ranks": True,
                },
                sort_keys=True,
            ),
            ddp,
        )
    else:
        derived = _build_derived_training_arrays()

    policy_sample_weights = derived["policy_sample_weights"]
    policy_distillation_scope_report = _policy_distillation_scope_report(
        data, policy_sample_weights
    )
    value_sample_weights = derived["value_sample_weights"]
    value_root_blend_audit = _audit_value_root_blend_corpus(
        data,
        value_sample_weights,
        regime=value_root_blend_regime,
        indices=np.asarray(derived["train_indices"], dtype=np.int64),
    )
    # `_value_training_metadata` is rebuilt for every epoch checkpoint and the
    # final checkpoint; attach the immutable pre-optimizer realization once so
    # every saved artifact carries the same audited target operator.
    args.value_root_blend_audit = value_root_blend_audit
    _rank0_print(
        json.dumps(
            {"progress": "value_root_blend_audit", **value_root_blend_audit},
            sort_keys=True,
        ),
        ddp,
    )
    policy_surprise_weights_full = derived["policy_surprise_weights_full"]
    train_indices = derived["train_indices"]
    validation_indices = derived["validation_indices"]
    if len(train_indices) == 0:
        raise SystemExit(
            "training split is empty; refusing to save a checkpoint that could "
            "falsely attest a value readout without any optimizer update"
        )
    validation_seed_manifest_path: Path | None = None
    validation_seed_set_sha256 = ""
    validation_game_seed_count = 0
    if "game_seed" in data:
        held_out_game_seeds = derived["held_out_game_seeds"]
        validation_game_seed_count = int(len(held_out_game_seeds))
        validation_seed_set_sha256 = _game_seed_set_sha256(held_out_game_seeds)
        if validation_seed_contract is not None:
            if (
                validation_seed_set_sha256
                != validation_seed_contract["validation_game_seed_set_sha256"]
            ):
                raise SystemExit(
                    "trainer validation seed set differs from the immutable A1 "
                    "validation manifest"
                )
            if int(len(validation_indices)) != int(
                validation_seed_contract["validation_row_count"]
            ):
                raise SystemExit(
                    "trainer validation row count differs from the immutable A1 "
                    "validation manifest: "
                    f"trainer={len(validation_indices)} "
                    f"manifest={validation_seed_contract['validation_row_count']}"
                )
        validation_seed_manifest_path = Path(args.report).with_suffix(
            ".validation_seeds.json"
        )
        if int(ddp["rank"]) == 0:
            write_json(
                validation_seed_manifest_path,
                {
                    "schema_version": "train-validation-game-seeds-v1",
                    "data": str(args.data),
                    "data_fingerprint": str(args.data_fingerprint),
                    "validation_fraction": float(args.validation_fraction),
                    "validation_seed": int(args.validation_seed),
                    "validation_max_samples": int(args.validation_max_samples),
                    "validation_game_seed_ranges": (
                        validation_game_seed_ranges or []
                    ),
                    "validation_game_seed_count": validation_game_seed_count,
                    "validation_game_seed_set_sha256": validation_seed_set_sha256,
                    "training_excluded_game_seed_count": (
                        int(
                            np.asarray(
                                validation_seed_contract.get(
                                    "excluded_game_seeds",
                                    validation_seed_contract["game_seeds"],
                                ),
                                dtype=np.int64,
                            ).size
                        )
                        if validation_seed_contract is not None
                        else validation_game_seed_count
                    ),
                    "training_excluded_game_seed_set_sha256": (
                        _game_seed_set_sha256(
                            np.asarray(
                                validation_seed_contract.get(
                                    "excluded_game_seeds",
                                    validation_seed_contract["game_seeds"],
                                ),
                                dtype=np.int64,
                            )
                        )
                        if validation_seed_contract is not None
                        else validation_seed_set_sha256
                    ),
                    **(
                        {
                            "a1_contract_sha256": validation_seed_contract[
                                "a1_contract_sha256"
                            ],
                            "input_validation_game_seed_manifest": str(
                                validation_seed_contract["path"]
                            ),
                            "input_validation_game_seed_manifest_sha256": (
                                validation_seed_contract["file_sha256"]
                            ),
                        }
                        if validation_seed_contract is not None
                        else {}
                    ),
                    "game_seeds": [int(seed) for seed in held_out_game_seeds],
                },
            )
    epoch_sample_weights = (
        policy_surprise_weights_full[train_indices]
        if float(args.policy_surprise_weight) > 0.0
        else None
    )
    component_game_sampling = derived.get("component_game_sampling")
    if component_game_sampling is not None:
        if epoch_sample_weights is not None:
            raise SystemExit(
                "authenticated component game sampling cannot be combined with "
                "--policy-surprise-weight yet; their joint probability semantics "
                "must be explicitly specified"
            )
        epoch_sample_weights = component_game_sampling
    policy_aux_sampling_weights = None
    if int(args.policy_aux_active_batch_size) > 0:
        if component_game_sampling is None:
            raise SystemExit(
                "policy auxiliary sampling requires authenticated composite base weights"
            )
        policy_aux_sampling_weights = _conditioned_policy_aux_sampling_weights(
            component_game_sampling,
            np.asarray(policy_sample_weights)[train_indices],
        )
    if int(ddp["rank"]) == 0:
        policy_sample_weight_report = sample_weight_quality(data, policy_sample_weights)
        value_sample_weight_report = sample_weight_quality(data, value_sample_weights)
        policy_surprise_weight_report = sample_weight_quality(
            data, policy_surprise_weights_full
        )
        value_sample_weight_report["by_game"] = per_game_weight_quality(
            data, value_sample_weights
        )
    else:
        # Reports are only printed/written by rank 0. Avoid repeating their
        # O(rows) grouping work and transient arrays on every local DDP rank.
        policy_sample_weight_report = {}
        value_sample_weight_report = {}
        policy_surprise_weight_report = {}
    _rank0_print(
        json.dumps(
            {
                "progress": "bc_split",
                "train_samples": int(len(train_indices)),
                "validation_samples": int(len(validation_indices)),
                "validation_fraction": float(args.validation_fraction),
                "validation_seed": int(args.validation_seed),
                "validation_game_seed_ranges": validation_game_seed_ranges,
                "sample_weights": policy_sample_weight_report,
                "policy_sample_weights": policy_sample_weight_report,
                "value_sample_weights": value_sample_weight_report,
                "policy_surprise_weight": float(args.policy_surprise_weight),
                "policy_surprise_cap": float(args.policy_surprise_cap),
                "policy_surprise_weight_quality": policy_surprise_weight_report,
            },
            sort_keys=True,
        ),
        ddp,
    )
    first_batch_profile = None
    (
        global_step,
        start_epoch,
        cumulative_scalar_training_weight,
        cumulative_categorical_training_weight,
    ) = _restore_training_progress_state(
        resume_progress,
        epochs=int(args.epochs),
        rng=rng,
        symmetry_rng=symmetry_rng,
        ddp=ddp,
    )
    optimizer_observed_steps = 0
    optimizer_clipped_steps = 0
    optimizer_pre_clip_grad_norm_sum = 0.0
    optimizer_pre_clip_grad_norm_max = 0.0
    total_training_steps = int(args.max_steps) if int(args.max_steps) > 0 else 0
    # C1 gradient accumulation. accum==1 (default) preserves the pre-C1 path
    # exactly: every micro-batch zero-grads, backwards the undivided loss, clips,
    # and steps, and global_step (== optimizer steps) advances once per batch.
    accum = max(1, int(args.grad_accum_steps))
    for epoch in range(start_epoch, args.epochs):
        remaining_order_samples = None
        if int(args.max_steps) > 0 and epoch_sample_weights is not None:
            # Weighted sampling is with replacement, so a bounded draw is the
            # exact prefix of the historical full-epoch draw. Do not allocate
            # tens of millions of positions when a max-step probe consumes only
            # a few million. Uniform permutation remains uncapped because there
            # is no cheaper way to preserve np.random.permutation's exact prefix.
            remaining_optimizer_steps = max(0, int(args.max_steps) - global_step)
            remaining_order_samples = (
                remaining_optimizer_steps
                * accum
                * int(args.batch_size)
                * (int(ddp["world_size"]) if ddp["enabled"] else 1)
            )
        order = _epoch_order(
            rng,
            len(train_indices),
            args.batch_size,
            ddp,
            data_sharded=bool(args.ddp_shard_data),
            sample_weights=epoch_sample_weights,
            max_samples=remaining_order_samples,
        )
        aux_order = None
        if policy_aux_sampling_weights is not None:
            # Stateless per-epoch seed preserves exact resume behavior without
            # perturbing or extending the historical persisted base RNG state.
            policy_aux_rng = np.random.default_rng(
                np.random.SeedSequence([int(args.seed), 0xA17C1E, int(epoch)])
            )
            local_aux_draws = (
                int(np.ceil(len(order) / max(1, int(args.batch_size))))
                * int(args.policy_aux_active_batch_size)
            )
            aux_order = _policy_aux_epoch_order(
                policy_aux_rng,
                len(train_indices),
                policy_aux_sampling_weights,
                local_draws=local_aux_draws,
                ddp=ddp,
            )
        epoch_policy_component_dose: dict[str, float] = {}
        if bool(getattr(data, "policy_distillation_scope_authenticated", False)):
            component_ids = tuple(data.component_ids)
            base_rows = train_indices[np.asarray(order, dtype=np.int64)]
            base_rows = base_rows[
                np.asarray(policy_sample_weights[base_rows], dtype=np.float32) > 0.0
            ]
            base_components = data.component_indices_for_rows(base_rows)
            aux_components = (
                np.empty(0, dtype=np.int64)
                if aux_order is None
                else data.component_indices_for_rows(
                    train_indices[np.asarray(aux_order, dtype=np.int64)]
                )
            )
            for component, component_id in enumerate(component_ids):
                epoch_policy_component_dose[f"{component_id}.base"] = float(
                    np.count_nonzero(base_components == component)
                )
                epoch_policy_component_dose[f"{component_id}.aux"] = float(
                    np.count_nonzero(aux_components == component)
                )
        epoch_losses = []
        epoch_extra_sums: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "final_vp_loss": 0.0,
            "q_loss": 0.0,
            "policy_kl_anchor_loss": 0.0,
            "value_uncertainty_loss": 0.0,
            "aux_subgoal_loss": 0.0,
            "moe_balance_loss": 0.0,
            "value_categorical_loss": 0.0,
            "value_categorical_clean_loss": 0.0,
            "value_categorical_truncated_loss": 0.0,
            "q_score_rows_ge2": 0.0,
            "soft_distillation_rows": 0.0,
            "advantage_weight_rows": 0.0,
            "advantage_mean_sum": 0.0,
            "advantage_weight_mean_sum": 0.0,
        }
        epoch_extra_denominators: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "final_vp_loss": 0.0,
            "q_loss": 0.0,
            "policy_kl_anchor_loss": 0.0,
            "value_uncertainty_loss": 0.0,
            "aux_subgoal_loss": 0.0,
            "moe_balance_loss": 0.0,
            "value_categorical_loss": 0.0,
            "value_categorical_clean_loss": 0.0,
            "value_categorical_truncated_loss": 0.0,
        }
        epoch_acc = []
        epoch_top3 = []
        epoch_count = 0
        epoch_active_count = 0.0
        epoch_aux_active_count = 0.0
        phase_stats = _empty_phase_stats()
        phase_stats_unforced = _empty_phase_stats()
        teacher_stats = _empty_phase_stats()
        total_batches = int(np.ceil(len(order) / max(1, args.batch_size)))
        # AUDIT FIX (LR schedule): the post-warmup decay curve needs an end point.
        # --max-steps (if set) is the true hard stop; otherwise fall back to the
        # epochs x batches-per-epoch estimate (recomputed identically every epoch
        # since train_indices/order length is stable, so this is a cheap no-op for
        # every epoch after the first).
        # Optimizer steps per epoch = ceil(batches / accum) (the trailing partial
        # accumulation group is flushed as one step at epoch end), so the LR-decay
        # endpoint is counted in OPTIMIZER steps. At accum==1 this is exactly
        # total_batches, i.e. the pre-C1 value.
        optimizer_steps_per_epoch = int(np.ceil(total_batches / accum))
        total_training_steps = (
            int(args.max_steps)
            if int(args.max_steps) > 0
            else int(args.epochs) * optimizer_steps_per_epoch
        )
        # Micro-batch position within the current accumulation group; reset after
        # every optimizer step and flushed at epoch end so gradients never carry
        # across the epoch boundary.
        micro_in_group = 0
        accumulation_group_size = accum
        # The policy-KL anchor and value-uncertainty auxiliary loss are entity_graph
        # (_train_xdim_batch) features -- the legacy dense _train_candidate_batch path
        # does not accept them, so only forward them when the xdim trainer is active.
        train_fn_extra_kwargs: dict[str, object] = {}
        if train_fn is _train_xdim_batch:
            train_fn_extra_kwargs = {
                "policy_kl_anchor_weight": float(args.policy_kl_anchor_weight),
                "policy_kl_anchor_direction": str(args.policy_kl_anchor_direction),
                "value_uncertainty_loss_weight": float(args.value_uncertainty_loss_weight),
                "aux_subgoal_loss_weight": float(args.aux_subgoal_loss_weight),
                "value_categorical_loss_weight": resolved_categorical_value_weight,
                "value_hlgauss_sigma_ratio": float(args.value_hlgauss_sigma_ratio),
                "value_target_lambda": float(args.value_target_lambda),
                "value_root_blend_phases": tuple(
                    value_root_blend_regime["phases"]
                ),
                "value_root_blend_global_compat": (
                    value_root_blend_regime["mode"] == "global_compat"
                ),
                "moe_balance_loss_weight": float(args.moe_balance_loss_weight),
            }
        batch_iterator = _iterate_training_batches(
            data,
            order,
            train_indices,
            int(args.batch_size),
            policy_sample_weights,
            value_sample_weights,
            num_workers=int(args.data_loader_workers),
            prefetch=int(args.data_loader_prefetch),
        )
        for batch_number, (_batch_tuple, _is_last_batch) in enumerate(
            _iter_with_last(batch_iterator), start=1
        ):
            batch_data, batch, batch_policy_weights, batch_value_weights = _batch_tuple
            if len(batch) == 0:
                continue
            aux_batch = None
            if aux_order is not None:
                aux_start = (batch_number - 1) * int(args.policy_aux_active_batch_size)
                aux_stop = aux_start + int(args.policy_aux_active_batch_size)
                aux_batch = train_indices[aux_order[aux_start:aux_stop]]
                if len(aux_batch) != int(args.policy_aux_active_batch_size):
                    raise RuntimeError("policy auxiliary draw dose drift")
            # C1 grad-accum bookkeeping. At accum==1 do_zero_grad and do_step are
            # both True every batch (identical to pre-C1). The LR schedule uses
            # global_step (optimizer steps), so within an accumulation group every
            # micro-batch sees the same LR. The final batch of the epoch always
            # closes its group with a synced optimizer step -- this both flushes
            # the trailing partial group and, crucially, guarantees that last
            # backward runs WITH DDP/FSDP gradient sync (non-stepping micro-batches
            # run under no_sync, so a step on unsynced grads would diverge ranks).
            # The last epoch batch is never empty (the final order slice has >=1
            # row), so no pending gradient can survive the loop.
            micro_in_group += 1
            accum_do_zero_grad = micro_in_group == 1
            if accum_do_zero_grad:
                # The final accumulation group can contain fewer than ``accum``
                # micro-batches.  Dividing those losses by the configured size
                # would shrink the final optimizer update by k/accum and break
                # equivalence with the same effective batch trained directly.
                accumulation_group_size = _accumulation_group_size(
                    configured_size=accum,
                    batch_number=batch_number,
                    total_batches=total_batches,
                )
            accum_do_step = (micro_in_group >= accum) or bool(_is_last_batch)
            accum_kwargs: dict = {}
            if train_fn is _train_xdim_batch:
                accum_kwargs = {
                    "grad_accum_steps": accumulation_group_size,
                    "accum_do_zero_grad": accum_do_zero_grad,
                    "accum_do_step": accum_do_step,
                }
            _apply_lr_schedule(
                optimizer,
                base_lr=float(args.lr),
                step=global_step,
                warmup_steps=int(args.lr_warmup_steps),
                total_steps=total_training_steps,
                schedule=str(args.lr_schedule),
            )
            # DDP documents that no_sync() must wrap the forward pass as well as
            # backward().  Applying it only inside _train_xdim_batch after the
            # forward leaves reducer hooks armed and silently synchronizes every
            # micro-batch, defeating gradient accumulation's communication win.
            with _gradient_sync_context(policy.model, accum_do_step=accum_do_step):
                batch_metrics = train_fn(
                    policy,
                    optimizer,
                    batch_data,
                    batch,
                    batch_policy_weights,
                    batch_value_weights,
                    args.soft_target_temperature,
                    args.soft_target_weight,
                    args.soft_target_source,
                    args.soft_target_min_legal_coverage,
                    args.policy_loss_weight,
                    resolved_scalar_value_weight,
                    args.final_vp_loss_weight,
                    args.q_loss_weight,
                    _parse_prefixes(args.q_skip_teacher_prefixes),
                    args.vps_to_win,
                    args.advantage_policy_weighting,
                    args.advantage_temperature,
                    args.advantage_weight_cap,
                    args.advantage_weight_floor,
                    args.amp,
                    diagnostics=(
                        int(args.train_diagnostics_every_batches) > 0
                        and batch_number % int(args.train_diagnostics_every_batches) == 0
                    ),
                    truncated_vp_margin_value_weight=args.truncated_vp_margin_value_weight,
                    symmetry=symmetry,
                    symmetry_rng=symmetry_rng,
                    symmetry_relabel_events=bool(
                        getattr(args, "symmetry_augment_events", True)
                    ),
                    **train_fn_extra_kwargs,
                    **accum_kwargs,
                    **(
                        {
                            "policy_aux_data": data,
                            "policy_aux_batch": aux_batch,
                            "policy_aux_sample_weights": policy_sample_weights,
                        }
                        if aux_batch is not None
                        else {}
                    ),
                )
            loss = float(batch_metrics["loss"])
            accuracy = float(batch_metrics["accuracy"])
            optimizer_observability = batch_metrics.get("optimizer_observability")
            if optimizer_observability is not None:
                optimizer_observed_steps += 1
                optimizer_clipped_steps += int(optimizer_observability["clipped"])
                pre_clip_norm = float(
                    optimizer_observability["pre_clip_total_grad_norm"]
                )
                optimizer_pre_clip_grad_norm_sum += pre_clip_norm
                optimizer_pre_clip_grad_norm_max = max(
                    optimizer_pre_clip_grad_norm_max, pre_clip_norm
                )
                _rank0_print(
                    json.dumps(
                        {
                            "progress": "bc_optimizer_observability",
                            "arch": args.arch,
                            "epoch": epoch + 1,
                            "batch": batch_number,
                            "optimizer_step": global_step + 1,
                            **optimizer_observability,
                            "observed_steps": optimizer_observed_steps,
                            "clipped_steps": optimizer_clipped_steps,
                            "clipped_fraction": (
                                optimizer_clipped_steps / optimizer_observed_steps
                            ),
                        },
                        sort_keys=True,
                    ),
                    ddp,
                )
            if first_batch_profile is None:
                first_batch_profile = _batch_profile(policy, batch_data, batch)
                _rank0_print(
                    json.dumps(
                        {
                            "progress": "bc_batch_profile",
                            "arch": args.arch,
                            "world_size": ddp["world_size"],
                            "rank": ddp["rank"],
                            **first_batch_profile,
                        },
                        sort_keys=True,
                    ),
                    ddp,
            )
            epoch_losses.append(loss * len(batch))
            active_count = float(batch_metrics.get("active_count", len(batch)))
            epoch_acc.append(accuracy * active_count)
            epoch_top3.append(float(batch_metrics["top3_accuracy"]) * active_count)
            epoch_active_count += active_count
            epoch_aux_active_count += float(
                batch_metrics.get("policy_aux_active_count", 0)
            )
            for key in (
                "policy_loss",
                "value_loss",
                "final_vp_loss",
                "q_loss",
                "policy_kl_anchor_loss",
                "value_uncertainty_loss",
                "aux_subgoal_loss",
                "moe_balance_loss",
                "value_categorical_loss",
                "value_categorical_clean_loss",
                "value_categorical_truncated_loss",
            ):
                weighted_sum_key = f"{key}_weighted_sum"
                weight_sum_key = f"{key}_weight_sum"
                if weighted_sum_key in batch_metrics and weight_sum_key in batch_metrics:
                    epoch_extra_sums[key] += float(batch_metrics[weighted_sum_key])
                    epoch_extra_denominators[key] += float(batch_metrics[weight_sum_key])
                else:
                    epoch_extra_sums[key] += float(batch_metrics.get(key, 0.0)) * len(batch)
                    epoch_extra_denominators[key] += float(len(batch))
            epoch_extra_sums["q_score_rows_ge2"] += float(batch_metrics.get("q_score_rows_ge2", 0.0))
            epoch_extra_sums["soft_distillation_rows"] += float(
                batch_metrics.get("soft_distillation_rows", 0.0)
            )
            advantage_rows = float(batch_metrics.get("advantage_weight_rows", 0.0))
            epoch_extra_sums["advantage_weight_rows"] += advantage_rows
            epoch_extra_sums["advantage_mean_sum"] += (
                float(batch_metrics.get("advantage_mean", 0.0)) * advantage_rows
            )
            epoch_extra_sums["advantage_weight_mean_sum"] += (
                float(batch_metrics.get("advantage_weight_mean", 1.0)) * advantage_rows
            )
            _merge_phase_stats(phase_stats, batch_metrics["phase_stats"])
            _merge_phase_stats(
                phase_stats_unforced,
                batch_metrics.get("phase_stats_unforced", {}),
            )
            _merge_phase_stats(teacher_stats, batch_metrics["teacher_stats"])
            epoch_count += len(batch)
            # An optimizer step (and its LR-schedule tick) happens once per
            # accumulation group; at accum==1 that is every batch. train_fn did
            # the clip+step when accum_do_step was True.
            if accum_do_step:
                micro_in_group = 0
                global_step += 1
            if args.progress_every_batches and batch_number % int(args.progress_every_batches) == 0:
                _rank0_print(
                    json.dumps(
                        {
                            "progress": "bc_batch",
                            "arch": args.arch,
                            "epoch": epoch + 1,
                            "batch": batch_number,
                            "batches": total_batches,
                            "samples": int(epoch_count),
                            "loss": loss,
                            "accuracy": accuracy,
                            "cuda": _cuda_memory(policy),
                        },
                        sort_keys=True,
                    ),
                    ddp,
                )
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        phase_stats = _reduce_nested_count_stats(phase_stats, ddp)
        phase_stats_unforced = _reduce_nested_count_stats(phase_stats_unforced, ddp)
        teacher_stats = _reduce_nested_count_stats(teacher_stats, ddp)
        loss_sum = float(np.sum(epoch_losses)) if epoch_losses else 0.0
        acc_sum = float(np.sum(epoch_acc)) if epoch_acc else 0.0
        top3_sum = float(np.sum(epoch_top3)) if epoch_top3 else 0.0
        epoch_extra_sums = _reduce_named_sums(epoch_extra_sums, ddp)
        epoch_extra_denominators = _reduce_named_sums(epoch_extra_denominators, ddp)
        cumulative_scalar_training_weight += float(
            epoch_extra_denominators["value_loss"]
        )
        cumulative_categorical_training_weight += float(
            epoch_extra_denominators["value_categorical_loss"]
        )
        loss_sum, acc_sum, top3_sum, total_count = _reduce_epoch_metrics(
            loss_sum,
            acc_sum,
            top3_sum,
            float(epoch_count),
            ddp,
        )
        active_count_total = _reduce_scalar_sum(float(epoch_active_count), ddp)
        aux_active_count_total = _reduce_scalar_sum(float(epoch_aux_active_count), ddp)
        epoch_policy_component_dose = _reduce_named_sums(
            epoch_policy_component_dose, ddp
        )
        policy_component_active_dose = {
            component_id: {
                "base_active_rows": int(
                    round(epoch_policy_component_dose[f"{component_id}.base"])
                ),
                "aux_active_rows": int(
                    round(epoch_policy_component_dose[f"{component_id}.aux"])
                ),
                "total_active_rows": int(
                    round(
                        epoch_policy_component_dose[f"{component_id}.base"]
                        + epoch_policy_component_dose[f"{component_id}.aux"]
                    )
                ),
            }
            for component_id in getattr(data, "component_ids", tuple())
        }
        policy_loss_epoch = _metric_from_sum_denominator(
            epoch_extra_sums["policy_loss"], epoch_extra_denominators["policy_loss"]
        )
        value_loss_epoch = _metric_from_sum_denominator(
            epoch_extra_sums["value_loss"], epoch_extra_denominators["value_loss"]
        )
        final_vp_loss_epoch = _metric_from_sum_denominator(
            epoch_extra_sums["final_vp_loss"], epoch_extra_denominators["final_vp_loss"]
        )
        q_loss_epoch = _metric_from_sum_denominator(
            epoch_extra_sums["q_loss"], epoch_extra_denominators["q_loss"]
        )
        auxiliary_loss_epochs = {
            key: _metric_from_sum_denominator(
                epoch_extra_sums[key], epoch_extra_denominators[key]
            )
            for key in (
                "policy_kl_anchor_loss",
                "value_uncertainty_loss",
                "aux_subgoal_loss",
                "moe_balance_loss",
                "value_categorical_loss",
            )
        }
        categorical_breakdown_epochs = {
            key: _metric_from_sum_denominator(
                epoch_extra_sums[key], epoch_extra_denominators[key]
            )
            for key in (
                "value_categorical_clean_loss",
                "value_categorical_truncated_loss",
            )
        }
        # This is the objective that actually went through backward(), including
        # optional categorical/auxiliary terms and non-unit policy weights.  The
        # old component reconstruction omitted all of those terms and therefore
        # made HL-Gauss/aux runs look artificially cheap in their reports.
        loss_epoch = loss_sum / max(total_count, 1.0)
        component_reconstructed_loss = (
            float(args.policy_loss_weight) * policy_loss_epoch
            + resolved_scalar_value_weight * value_loss_epoch
            + float(args.final_vp_loss_weight) * final_vp_loss_epoch
            + float(args.q_loss_weight) * q_loss_epoch
            + float(args.policy_kl_anchor_weight)
            * auxiliary_loss_epochs["policy_kl_anchor_loss"]
            + float(args.value_uncertainty_loss_weight)
            * auxiliary_loss_epochs["value_uncertainty_loss"]
            + float(args.aux_subgoal_loss_weight)
            * auxiliary_loss_epochs["aux_subgoal_loss"]
            + float(args.moe_balance_loss_weight)
            * auxiliary_loss_epochs["moe_balance_loss"]
            + resolved_categorical_value_weight
            * auxiliary_loss_epochs["value_categorical_loss"]
        )
        metrics.append(
            {
                "epoch": epoch + 1,
                "loss": loss_epoch,
                "raw_batch_mean_loss": loss_epoch,
                "component_reconstructed_loss": component_reconstructed_loss,
                "policy_loss": policy_loss_epoch,
                "policy_base_active_rows": int(active_count_total),
                "policy_aux_active_rows": int(aux_active_count_total),
                "policy_total_active_rows": int(
                    active_count_total + aux_active_count_total
                ),
                "policy_component_active_dose": policy_component_active_dose,
                "value_loss": value_loss_epoch,
                "scalar_value_mse_diagnostic": value_loss_epoch,
                "final_vp_loss": final_vp_loss_epoch,
                "q_loss": q_loss_epoch,
                **auxiliary_loss_epochs,
                **categorical_breakdown_epochs,
                "primary_value_loss": (
                    auxiliary_loss_epochs["value_categorical_loss"]
                    if resolved_categorical_value_weight > 0.0
                    else value_loss_epoch
                ),
                "primary_value_loss_kind": (
                    "hlgauss_ce"
                    if resolved_categorical_value_weight > 0.0
                    else "scalar_mse"
                ),
                "loss_denominators": dict(epoch_extra_denominators),
                "q_score_rows_ge2": int(round(epoch_extra_sums["q_score_rows_ge2"])),
                "q_score_rows_ge2_fraction": epoch_extra_sums["q_score_rows_ge2"] / max(total_count, 1.0),
                "soft_distillation_rows": int(round(epoch_extra_sums["soft_distillation_rows"])),
                "soft_distillation_fraction": epoch_extra_sums["soft_distillation_rows"] / max(total_count, 1.0),
                "advantage_weight_rows": int(round(epoch_extra_sums["advantage_weight_rows"])),
                "advantage_mean": (
                    epoch_extra_sums["advantage_mean_sum"]
                    / max(epoch_extra_sums["advantage_weight_rows"], 1.0)
                ),
                "advantage_weight_mean": (
                    epoch_extra_sums["advantage_weight_mean_sum"]
                    / max(epoch_extra_sums["advantage_weight_rows"], 1.0)
                ),
                "accuracy_active_count": int(round(active_count_total)),
                "accuracy": acc_sum / max(active_count_total, 1.0),
                "top3_accuracy": top3_sum / max(active_count_total, 1.0),
                **(
                    {
                        "optimizer_observability": {
                            "observed_steps": optimizer_observed_steps,
                            "clipped_steps": optimizer_clipped_steps,
                            "clipped_fraction": (
                                optimizer_clipped_steps / optimizer_observed_steps
                            ),
                            "mean_pre_clip_total_grad_norm": (
                                optimizer_pre_clip_grad_norm_sum
                                / optimizer_observed_steps
                            ),
                            "max_pre_clip_total_grad_norm": (
                                optimizer_pre_clip_grad_norm_max
                            ),
                        }
                    }
                    if optimizer_observed_steps
                    else {}
                ),
                "phase_accuracy": _finalize_phase_stats(phase_stats),
                "phase_accuracy_excluding_forced": _finalize_phase_stats(phase_stats_unforced),
                "teacher_accuracy": _finalize_phase_stats(teacher_stats),
            }
        )
        def _evaluate_validation_indices(eval_indices: np.ndarray) -> dict:
            return evaluate_bc_batches(
                policy,
                data,
                eval_indices,
                policy_sample_weights,
                value_sample_weights,
                args.batch_size,
                args.soft_target_temperature,
                args.soft_target_weight,
                args.soft_target_source,
                args.soft_target_min_legal_coverage,
                args.policy_loss_weight,
                resolved_scalar_value_weight,
                args.final_vp_loss_weight,
                args.q_loss_weight,
                _parse_prefixes(args.q_skip_teacher_prefixes),
                args.vps_to_win,
                args.advantage_policy_weighting,
                args.advantage_temperature,
                args.advantage_weight_cap,
                args.advantage_weight_floor,
                ddp,
                args.amp,
                data_sharded=bool(args.ddp_shard_data),
                truncated_vp_margin_value_weight=args.truncated_vp_margin_value_weight,
                policy_kl_anchor_weight=float(args.policy_kl_anchor_weight),
                policy_kl_anchor_direction=str(args.policy_kl_anchor_direction),
                value_uncertainty_loss_weight=float(args.value_uncertainty_loss_weight),
                aux_subgoal_loss_weight=float(args.aux_subgoal_loss_weight),
                moe_balance_loss_weight=float(args.moe_balance_loss_weight),
                value_categorical_loss_weight=resolved_categorical_value_weight,
                value_hlgauss_sigma_ratio=float(args.value_hlgauss_sigma_ratio),
                value_target_lambda=float(args.value_target_lambda),
                value_root_blend_phases=tuple(
                    value_root_blend_regime["phases"]
                ),
                value_root_blend_global_compat=(
                    value_root_blend_regime["mode"] == "global_compat"
                ),
                data_loader_workers=int(args.data_loader_workers),
                data_loader_prefetch=int(args.data_loader_prefetch),
            )

        validation_metrics = _evaluate_validation_indices(validation_indices)
        validation_metrics["measure"] = "raw_row_concat"
        validation_metrics["objective_matched"] = False
        validation_metrics["warning"] = (
            "compatibility metric: raw held-out rows do not follow the "
            "authenticated component->game->row training measure"
        )
        metrics[-1]["validation"] = validation_metrics
        if is_memmap_composite and tuple(
            getattr(data, "component_game_sampling_ratios", tuple())
        ):
            metrics[-1]["validation_objective_matched"] = (
                evaluate_composite_validation_measure(
                    data,
                    validation_indices,
                    _evaluate_validation_indices,
                )
            )
        _rank0_print(
            json.dumps(
                {
                    "progress": "bc",
                    "arch": args.arch,
                    "world_size": ddp["world_size"],
                    **metrics[-1],
                    "cuda": _cuda_memory(policy),
                },
                sort_keys=True,
            ),
            ddp,
        )
        if args.save_each_epoch:
            # Called on every rank: _save_policy writes on rank 0 for DDP/single
            # and runs the collective full-state-dict gather (all ranks) for FSDP.
            epoch_path = _epoch_checkpoint_path(args.checkpoint, epoch + 1)
            checkpoint_model = getattr(policy.model, "module", policy.model)
            # Re-audit after training so lazily imported modules are covered too.
            checkout_runtime_binding = _assert_checkout_runtime_binding()
            args.checkout_runtime_binding = checkout_runtime_binding
            value_training = _value_training_metadata(
                args,
                scalar_weight=resolved_scalar_value_weight,
                categorical_weight=resolved_categorical_value_weight,
                categorical_bins=int(
                    getattr(checkpoint_model, "value_categorical_bins", 0) or 0
                ),
                optimizer_steps=global_step,
                completed_epochs=epoch + 1,
                scalar_training_weight_sum=cumulative_scalar_training_weight,
                categorical_training_weight_sum=(
                    cumulative_categorical_training_weight
                ),
            )
            _save_policy(
                policy,
                str(epoch_path),
                ddp,
                mask_hidden_info=bool(args.mask_hidden_info),
                soft_target_source=args.soft_target_source,
                value_training=value_training,
            )
            # CAT-128 patch #8: persist optimizer (Adam) state as <ckpt>.optimizer.pt.
            # Called on ALL ranks (the FSDP gather is a collective); rank-0 writes.
            optimizer_saved = _save_optimizer_sidecar(
                str(epoch_path), policy, optimizer, ddp
            )
            _save_training_progress_sidecar(
                str(epoch_path),
                optimizer_saved=optimizer_saved,
                optimizer_step=global_step,
                completed_epochs=epoch + 1,
                recipe_identity=resume_recipe_identity,
                rng=rng,
                symmetry_rng=symmetry_rng,
                scalar_training_weight_sum=cumulative_scalar_training_weight,
                categorical_training_weight_sum=(
                    cumulative_categorical_training_weight
                ),
                ddp=ddp,
            )
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    # Called on every rank (see _save_policy): rank-0 write for DDP/single, and a
    # collective full-state-dict gather for FSDP (C1). MUST stay unconditional --
    # the FSDP gather is collective, so wrapping in `if rank==0` would deadlock/skip
    # it. OPT-8 soft_target_source provenance kwarg threaded in (recorded in metadata).
    checkpoint_model = getattr(policy.model, "module", policy.model)
    # The durable identity is the complete runtime actually loaded by the end of
    # training, not merely the smaller module set present during argument parsing.
    checkout_runtime_binding = _assert_checkout_runtime_binding()
    args.checkout_runtime_binding = checkout_runtime_binding
    value_training = _value_training_metadata(
        args,
        scalar_weight=resolved_scalar_value_weight,
        categorical_weight=resolved_categorical_value_weight,
        categorical_bins=int(
            getattr(checkpoint_model, "value_categorical_bins", 0) or 0
        ),
        optimizer_steps=global_step,
        completed_epochs=start_epoch + len(metrics),
        scalar_training_weight_sum=cumulative_scalar_training_weight,
        categorical_training_weight_sum=(
            cumulative_categorical_training_weight
        ),
    )
    trained_value_readouts = set(value_training["trained_value_readouts"])
    if (
        resolved_categorical_value_weight > 0.0
        and "categorical" not in trained_value_readouts
    ):
        raise RuntimeError(
            "HL-Gauss objective completed without any effective categorical "
            "training mass and optimizer update; refusing to save a checkpoint "
            "that could be mistaken for a trained categorical readout"
        )
    if resolved_scalar_value_weight > 0.0 and "scalar" not in trained_value_readouts:
        raise RuntimeError(
            "scalar-MSE objective completed without any effective value training "
            "mass and optimizer update; refusing to save false provenance"
        )
    _save_policy(
        policy,
        args.checkpoint,
        ddp,
        mask_hidden_info=bool(args.mask_hidden_info),
        soft_target_source=args.soft_target_source,
        value_training=value_training,
    )
    # CAT-128 patch #8: persist final optimizer (Adam) state alongside the checkpoint.
    optimizer_saved = _save_optimizer_sidecar(
        args.checkpoint, policy, optimizer, ddp
    )
    _save_training_progress_sidecar(
        args.checkpoint,
        optimizer_saved=optimizer_saved,
        optimizer_step=global_step,
        completed_epochs=start_epoch + len(metrics),
        recipe_identity=resume_recipe_identity,
        rng=rng,
        symmetry_rng=symmetry_rng,
        scalar_training_weight_sum=cumulative_scalar_training_weight,
        categorical_training_weight_sum=cumulative_categorical_training_weight,
        ddp=ddp,
    )
    policy_component_active_dose = {
        component_id: {
            key: int(
                sum(
                    int(
                        metric.get("policy_component_active_dose", {})
                        .get(component_id, {})
                        .get(key, 0)
                    )
                    for metric in metrics
                )
            )
            for key in ("base_active_rows", "aux_active_rows", "total_active_rows")
        }
        for component_id in getattr(data, "component_ids", tuple())
    }
    report = {
        "checkout_runtime_binding": checkout_runtime_binding,
        "arch": args.arch,
        "config_hash": train_config_hash,
        "samples": int(n),
        "global_samples": int(_reduce_scalar_sum(float(n), ddp))
        if bool(args.ddp_shard_data)
        else int(n),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "epochs": args.epochs,
        "max_steps": int(args.max_steps),
        "steps_completed": int(global_step),
        "batch_size": args.batch_size,
        "amp": args.amp,
        "optimizer": args.optimizer,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "fused_optimizer": bool(args.fused_optimizer),
        "hidden_size": int(args.hidden_size),
        "mask_hidden_info": bool(args.mask_hidden_info),
        "seed": int(args.seed),
        "symmetry_augment": bool(args.symmetry_augment),
        "data": str(args.data),
        "data_fingerprint": str(args.data_fingerprint),
        "data_format": args.data_format,
        "track": args.track,
        "vps_to_win": int(args.vps_to_win),
        "validation_game_seed_ranges": validation_game_seed_ranges or None,
        "validation_game_seed_manifest": (
            str(validation_seed_manifest_path)
            if validation_seed_manifest_path is not None
            else None
        ),
        "input_validation_game_seed_manifest": (
            None
            if validation_seed_contract is None
            else str(validation_seed_contract["path"])
        ),
        "input_validation_game_seed_manifest_sha256": (
            None
            if validation_seed_contract is None
            else validation_seed_contract["file_sha256"]
        ),
        "input_validation_game_sentinel_manifest": (
            str(args.validation_game_sentinel_manifest) or None
        ),
        "training_excluded_game_seed_count": (
            0
            if validation_seed_contract is None
            else int(
                np.asarray(
                    validation_seed_contract.get(
                        "excluded_game_seeds", validation_seed_contract["game_seeds"]
                    ),
                    dtype=np.int64,
                ).size
            )
        ),
        "training_excluded_game_seed_set_sha256": (
            None
            if validation_seed_contract is None
            else _game_seed_set_sha256(
                np.asarray(
                    validation_seed_contract.get(
                        "excluded_game_seeds", validation_seed_contract["game_seeds"]
                    ),
                    dtype=np.int64,
                )
            )
        ),
        "a1_contract_sha256": (
            None
            if validation_seed_contract is None
            else validation_seed_contract["a1_contract_sha256"]
        ),
        "a1_selected_game_seed_set_sha256": (
            None
            if a1_training_binding is None
            else a1_training_binding["selected_game_seed_set_sha256"]
        ),
        "a1_training_game_seed_set_sha256": (
            None
            if a1_training_binding is None
            else a1_training_binding["training_game_seed_set_sha256"]
        ),
        "a1_bound_learner_value_objective": (
            None
            if a1_training_binding is None
            else a1_training_binding["learner_value_objective"]
        ),
        "a1_bound_learner_training_recipe": (
            None
            if a1_training_binding is None
            else a1_training_binding["learner_training_recipe"]
        ),
        "a1_learner_training_recipe_sha256": (
            None
            if a1_training_binding is None
            else a1_training_binding["learner_training_recipe_sha256"]
        ),
        "a1_memmap_payload_inventory_sha256": (
            None
            if a1_training_binding is None
            else args.a1_memmap_payload_inventory_sha256
        ),
        "a1_learner_code_sha256": (
            None
            if a1_training_binding is None
            else a1_training_binding["learner_code_sha256"]
        ),
        "a1_runtime_code_tree_sha256": (
            None
            if a1_training_binding is None
            else a1_training_binding["runtime_code_tree_sha256"]
        ),
        "validation_game_seed_count": validation_game_seed_count,
        "validation_game_seed_set_sha256": validation_seed_set_sha256 or None,
        "allow_missing_game_seed_validation_split": bool(
            args.allow_missing_game_seed_validation_split
        ),
        "graph_history_features": bool(
            getattr(env_config, "use_graph_history_features", False)
        ),
        "checkpoint": args.checkpoint,
        "init_checkpoint": args.init_checkpoint or None,
        "init_checkpoint_sha256": str(args.init_checkpoint_sha256) or None,
        "a1_curriculum_parent": getattr(args, "a1_curriculum_parent", None),
        "grow_from_checkpoint_sha256": (
            str(args.grow_from_checkpoint_sha256) or None
        ),
        "resume_optimizer": bool(args.resume_optimizer),
        "optimizer_restored": bool(optimizer_restored),
        "resumed_optimizer_step": (
            None
            if resume_progress is None
            else int(resume_progress["optimizer_step"])
        ),
        "resumed_completed_epochs": (
            None
            if resume_progress is None
            else int(resume_progress["completed_epochs"])
        ),
        "grow_from_checkpoint": args.grow_from_checkpoint or None,
        "metrics": metrics,
        "data_quality": data_quality,
        "policy_distillation_scope": policy_distillation_scope_report,
        "policy_component_active_dose": policy_component_active_dose,
        "sample_weight_quality": policy_sample_weight_report,
        "policy_sample_weight_quality": policy_sample_weight_report,
        "value_sample_weight_quality": value_sample_weight_report,
        "policy_surprise_weight": float(args.policy_surprise_weight),
        "policy_surprise_cap": float(args.policy_surprise_cap),
        "policy_surprise_weight_quality": policy_surprise_weight_report,
        "first_batch_profile": first_batch_profile,
        "parameter_count": int(_parameter_count(policy)),
        "world_size": ddp["world_size"],
        "ddp_shard_data": bool(args.ddp_shard_data),
        "teacher_weights": _parse_weight_map(args.teacher_weights),
        "phase_weights": _parse_weight_map(args.phase_weights),
        "value_phase_weights": _parse_weight_map(value_phase_weights_raw),
        "forced_action_weight": args.forced_action_weight,
        "per_game_policy_weight": bool(args.per_game_policy_weight),
        "per_game_policy_weight_mode": str(args.per_game_policy_weight_mode),
        "forced_row_value_weight": args.forced_row_value_weight,
        "per_game_value_weight": bool(args.per_game_value_weight),
        "per_game_value_weight_mode": str(args.per_game_value_weight_mode),
        "winner_sample_weight": args.winner_sample_weight,
        "loser_sample_weight": args.loser_sample_weight,
        "vp_margin_weight": args.vp_margin_weight,
        "advantage_policy_weighting": args.advantage_policy_weighting,
        "advantage_temperature": args.advantage_temperature,
        "advantage_weight_cap": args.advantage_weight_cap,
        "advantage_weight_floor": args.advantage_weight_floor,
        "soft_target_temperature": args.soft_target_temperature,
        "soft_target_weight": args.soft_target_weight,
        "soft_target_source": args.soft_target_source,
        "soft_target_min_legal_coverage": args.soft_target_min_legal_coverage,
        "policy_loss_weight": args.policy_loss_weight,
        "policy_aux_active_batch_size": int(args.policy_aux_active_batch_size),
        "policy_aux_active_rows": int(
            sum(int(metric.get("policy_aux_active_rows", 0)) for metric in metrics)
        ),
        "policy_base_active_rows": int(
            sum(int(metric.get("policy_base_active_rows", 0)) for metric in metrics)
        ),
        "value_loss_weight": args.value_loss_weight,
        "action_module_lr_mult": args.action_module_lr_mult,
        "trunk_lr_mult": args.trunk_lr_mult,
        "resolved_scalar_value_loss_weight": resolved_scalar_value_weight,
        "truncated_vp_margin_value_weight": float(args.truncated_vp_margin_value_weight),
        "final_vp_loss_weight": args.final_vp_loss_weight,
        "q_loss_weight": args.q_loss_weight,
        "policy_kl_anchor_weight": args.policy_kl_anchor_weight,
        "policy_kl_anchor_direction": args.policy_kl_anchor_direction,
        "policy_kl_anchor_normalization": (
            "conditional_authenticated_multi_action_prior_rows"
        ),
        "policy_kl_anchor_weight_semantics": (
            "coefficient_multiplies_the_conditional_eligible_row_mean"
        ),
        "value_uncertainty_loss_weight": args.value_uncertainty_loss_weight,
        "aux_subgoal_loss_weight": args.aux_subgoal_loss_weight,
        "moe_balance_loss_weight": args.moe_balance_loss_weight,
        "value_head_type": args.value_head_type,
        "value_categorical_bins": int(args.value_categorical_bins),
        "value_categorical_loss_weight": args.value_categorical_loss_weight,
        "resolved_categorical_value_loss_weight": resolved_categorical_value_weight,
        "hlgauss_scalar_aux_loss_weight": args.hlgauss_scalar_aux_loss_weight,
        "value_training": value_training,
        "value_hlgauss_sigma_ratio": args.value_hlgauss_sigma_ratio,
        "value_target_lambda": args.value_target_lambda,
        "value_root_blend_regime": value_root_blend_regime,
        "value_root_blend_audit": value_root_blend_audit,
        "edge_policy_head": bool(getattr(args, "edge_policy_head", False)),
        "aux_subgoal_heads": bool(getattr(args, "aux_subgoal_heads", False)),
        "freeze_modules": args.freeze_modules,
        "train_value_only": bool(args.train_value_only),
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_schedule": args.lr_schedule,
        "total_training_steps": int(total_training_steps),
        "allow_teacher_score_q_loss": bool(args.allow_teacher_score_q_loss),
        "allow_legacy_action_mask_upgrade": bool(args.allow_legacy_action_mask_upgrade),
        "trust_curated_data_quality": bool(args.trust_curated_data_quality),
        "require_strict_35m_teacher": bool(args.require_strict_35m_teacher),
        "require_production_35m_teacher": bool(args.require_production_35m_teacher),
        "require_35m_model": bool(args.require_35m_model),
        "min_35m_params": int(args.min_35m_params),
        "max_35m_params": int(args.max_35m_params),
        "teacher_quality_gate_report": str(Path(args.report).with_suffix(".teacher_quality.json"))
        if (args.require_strict_35m_teacher or args.require_production_35m_teacher)
        else None,
        "q_skip_teacher_prefixes": _parse_prefixes(args.q_skip_teacher_prefixes),
        "validation_fraction": args.validation_fraction,
        "validation_max_samples": args.validation_max_samples,
        "validation_seed": args.validation_seed,
        "graph_tokens": args.graph_tokens if args.arch == "xdim_graph" else None,
        "graph_layers": args.graph_layers if args.arch in ("xdim_graph", "entity_graph") else None,
        "attention_heads": args.attention_heads if args.arch in ("xdim_graph", "entity_graph") else None,
        "graph_dropout": args.graph_dropout if args.arch in ("xdim_graph", "entity_graph") else None,
        "progress_every_batches": args.progress_every_batches,
        "train_diagnostics_every_batches": args.train_diagnostics_every_batches,
        "elapsed_sec": time.perf_counter() - start,
    }
    if is_memmap_composite:
        report.update({
            "memmap_composite": {
                "schema_version": a1_preflight_meta["schema_version"],
                "descriptor_path": a1_preflight_meta["descriptor_path"],
                "descriptor_file_sha256": a1_preflight_meta["descriptor_file_sha256"],
                "descriptor_fingerprint": a1_preflight_meta["descriptor_fingerprint"],
                "payload_inventory_sha256": a1_preflight_meta["payload_inventory_sha256"],
                "learner_recipe_overrides": a1_preflight_meta[
                    "learner_recipe_overrides"
                ],
                "learner_recipe_overrides_sha256": a1_preflight_meta[
                    "learner_recipe_overrides_sha256"
                ],
                "component_count": len(a1_preflight_meta["components"]),
                "component_ids": a1_preflight_meta.get("component_ids", []),
                "component_game_sampling_ratios": a1_preflight_meta.get(
                    "component_game_sampling_ratios", []
                ),
                "policy_kl_anchor_component_ids": a1_preflight_meta.get(
                    "policy_kl_anchor_component_ids", []
                ),
                "policy_distillation_component_ids": a1_preflight_meta.get(
                    "policy_distillation_component_ids", []
                ),
                "policy_distillation_scope_explicit": bool(
                    a1_preflight_meta.get("policy_distillation_scope_explicit", False)
                ),
                "component_contract_sha256s": [
                    contract["a1_contract_sha256"]
                    for contract in validation_seed_contract["component_contracts"]
                ],
            },
            "diagnostic_only": True,
            "promotion_eligible": False,
        })
    learner_ablation = (
        None
        if a1_training_binding is None
        else a1_training_binding.get("learner_ablation")
    )
    if learner_ablation is not None:
        report.update(
            {
                "a1_effective_learner_training_recipe": a1_training_binding[
                    "effective_learner_training_recipe"
                ],
                "a1_effective_learner_training_recipe_sha256": (
                    _canonical_json_sha256(
                        a1_training_binding["effective_learner_training_recipe"]
                    )
                ),
                "a1_learner_ablation": learner_ablation,
                "diagnostic_only": True,
                "promotion_eligible": False,
            }
        )
    topology_authorization = (
        None
        if a1_training_binding is None
        else a1_training_binding.get("learner_topology_authorization")
    )
    if topology_authorization is not None:
        report.update(
            {
                "a1_effective_learner_training_recipe": a1_training_binding[
                    "effective_learner_training_recipe"
                ],
                "a1_effective_learner_training_recipe_sha256": (
                    _canonical_json_sha256(
                        a1_training_binding["effective_learner_training_recipe"]
                    )
                ),
                "a1_learner_topology_authorization": topology_authorization,
            }
        )
    if ddp["rank"] == 0:
        write_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    if ddp["enabled"]:
        import torch.distributed as dist

        dist.destroy_process_group()


def _train_candidate_batch(
    policy,
    optimizer,
    data: dict,
    batch: np.ndarray,
    policy_sample_weights: np.ndarray,
    value_sample_weights: np.ndarray,
    soft_target_temperature: float,
    soft_target_weight: float,
    soft_target_source: str,
    soft_target_min_legal_coverage: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    final_vp_loss_weight: float,
    q_loss_weight: float,
    q_skip_teacher_prefixes: tuple[str, ...],
    vps_to_win: int,
    advantage_policy_weighting: str,
    advantage_temperature: float,
    advantage_weight_cap: float,
    advantage_weight_floor: float,
    amp: str = "none",
    *,
    diagnostics: bool = True,
    truncated_vp_margin_value_weight: float = 0.0,
) -> dict:
    del q_skip_teacher_prefixes, amp
    del advantage_policy_weighting, advantage_temperature, advantage_weight_cap, advantage_weight_floor
    import torch
    from torch import nn

    obs = torch.as_tensor(
        normalize_observations(data["obs"][batch]),
        dtype=torch.float32,
        device=policy.device,
    )
    context = _dense_context(data, batch, policy.action_size, policy.context_action_feature_size)
    context_t = torch.as_tensor(context, dtype=torch.float32, device=policy.device)
    actions = torch.as_tensor(data["action_taken"][batch].astype(np.int64), device=policy.device)
    valid = _valid_lists(data["legal_action_ids"][batch])
    logits, values = policy.forward(obs, context_t)
    masked = _torch_ppo_masked_logits(logits, valid, policy.action_size)
    policy_weights = torch.as_tensor(
        policy_sample_weights[batch],
        dtype=torch.float32,
        device=policy.device,
    )
    value_weights = torch.as_tensor(
        value_sample_weights[batch],
        dtype=torch.float32,
        device=policy.device,
    )
    hard_loss = nn.functional.cross_entropy(masked, actions, reduction="none")
    soft_targets, has_soft, soft_support = _soft_targets_full(
        data,
        batch,
        policy.action_size,
        policy.device,
        soft_target_temperature,
        soft_target_source,
        soft_target_min_legal_coverage,
    )
    if soft_targets is not None:
        log_probs = _support_log_softmax(masked, soft_support)
        soft_loss = -(soft_targets * log_probs).sum(dim=-1)
        alpha = float(np.clip(soft_target_weight, 0.0, 1.0))
        per_sample_loss = torch.where(
            has_soft,
            alpha * soft_loss + (1.0 - alpha) * hard_loss,
            hard_loss,
        )
    else:
        per_sample_loss = hard_loss
    policy_loss = _weighted_mean_loss(per_sample_loss, policy_weights)
    policy_loss_sum, policy_loss_denominator = _weighted_loss_parts(per_sample_loss, policy_weights)
    _, _, _, _, outcome_targets, has_outcome, outcome_confidence = _value_targets(
        data,
        batch,
        policy.device,
        vps_to_win,
        truncated_vp_margin_value_weight=truncated_vp_margin_value_weight,
    )
    value_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
    if outcome_targets is not None:
        value_error = nn.functional.mse_loss(values, outcome_targets, reduction="none")
        value_loss = _weighted_mean_loss(
            value_error,
            value_weights * outcome_confidence,
            mask=has_outcome,
        )
        value_loss_sum, value_loss_denominator = _weighted_loss_parts(
            value_error,
            value_weights * outcome_confidence,
            mask=has_outcome,
        )
    else:
        value_loss_sum, value_loss_denominator = _zero_loss_parts(policy.device)
    loss = float(policy_loss_weight) * policy_loss + float(value_loss_weight) * value_loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite BC loss: {float(loss.detach().cpu())}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(_params(policy)), 1.0)
    optimizer.step()
    predictions = torch.argmax(masked, dim=-1)
    active = policy_weights > 0.0
    active_count = int(active.sum().item())
    accuracy = _masked_metric_mean((predictions == actions).float(), active)
    top3_accuracy = _topk_full_accuracy(masked, actions, k=3, mask=active)
    active_np = active.detach().cpu().numpy().astype(bool)
    if diagnostics:
        predictions_np = predictions.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        logits_np = masked.detach().cpu().numpy()
        phase_stats = _field_stats(
            data,
            batch[active_np],
            predictions_np[active_np],
            actions_np[active_np],
            logits_np[active_np],
            field="phase",
        )
        teacher_stats = _field_stats(
            data,
            batch[active_np],
            predictions_np[active_np],
            actions_np[active_np],
            logits_np[active_np],
            field="teacher_name",
        )
        phase_stats_unforced = _field_stats_unforced(
            data,
            batch[active_np],
            predictions_np[active_np],
            actions_np[active_np],
            logits_np[active_np],
            field="phase",
        )
    else:
        phase_stats = {}
        teacher_stats = {}
        phase_stats_unforced = {}
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "final_vp_loss": 0.0,
        "q_loss": 0.0,
        "policy_loss_weighted_sum": float(policy_loss_sum.item()),
        "policy_loss_weight_sum": float(policy_loss_denominator.item()),
        "value_loss_weighted_sum": float(value_loss_sum.item()),
        "value_loss_weight_sum": float(value_loss_denominator.item()),
        "final_vp_loss_weighted_sum": 0.0,
        "final_vp_loss_weight_sum": 0.0,
        "q_loss_weighted_sum": 0.0,
        "q_loss_weight_sum": 0.0,
        "q_score_rows_ge2": 0,
        "soft_distillation_rows": int(has_soft.sum().item()) if soft_targets is not None else 0,
        "soft_distillation_active_rows": (
            int((has_soft & active).sum().item()) if soft_targets is not None else 0
        ),
        "active_count": active_count,
        "accuracy": float(accuracy.item()),
        "top3_accuracy": float(top3_accuracy.item()),
        "phase_stats": phase_stats,
        "teacher_stats": teacher_stats,
        "phase_stats_unforced": phase_stats_unforced,
    }


def _entity_batch(data: dict, batch: np.ndarray) -> dict[str, np.ndarray]:
    missing = [key for key in ENTITY_BATCH_KEYS if key not in data]
    if missing:
        raise SystemExit(
            "entity_graph training requires converted entity-token fields; "
            f"missing {missing[:8]}{'...' if len(missing) > 8 else ''}. "
            "Run tools/convert_teacher_to_entity_tokens.py first."
        )
    result = {key: data[key][batch] for key in ENTITY_BATCH_KEYS}
    if _MASK_HIDDEN_INFO_PLAYER_TOKENS:
        # Load-time public-observation masking: zero non-actor hidden slots so a
        # model trained here matches the public-observation evaluator (task #72).
        result["player_tokens"] = _mask_player_tokens_public(result["player_tokens"])
    return result


def _forward_legal_np_for_batch(
    policy,
    data: dict,
    batch: np.ndarray,
    legal_action_ids: np.ndarray,
    *,
    return_q: bool,
    symmetry=None,
    symmetry_rng=None,
    symmetry_relabel_events: bool = True,
) -> dict:
    if getattr(policy, "policy_type", "") == "entity_graph":
        entity = _entity_batch(data, batch)
        symmetry_ids = None
        if symmetry is not None:
            # f74: draw one D6 orientation per row and relabel the board tokens.
            # Legal-action rows keep their order (only their target ids move), so
            # the per-legal-row policy/soft targets computed by the caller stay
            # aligned and the value target is invariant -- nothing else to change.
            n_sym = int(symmetry.fwd_hex.shape[0])
            b = int(np.asarray(entity["hex_tokens"]).shape[0])
            gen = symmetry_rng if symmetry_rng is not None else np.random.default_rng()
            g = gen.integers(n_sym, size=b)
            symmetry_ids = g
            entity = symmetry.permute_entity_batch(
                entity,
                g,
                relabel_events=symmetry_relabel_events,
                legal_action_ids=legal_action_ids,
                action_size=int(policy.action_size),
            )
        outputs = policy.forward_legal_np(
            entity,
            legal_action_ids,
            data["legal_action_context"][batch],
            return_q=return_q,
        )
        # Keep the exact sampled orientation with the forward result. Spatial
        # auxiliary labels live outside ``entity`` and must be relabelled by the
        # same per-row D6 element before their loss is computed. Without this
        # metadata, combining --symmetry-augment with CAT-100 silently trains
        # next-settlement/robber heads against the unrotated coordinates.
        if symmetry_ids is not None:
            outputs["_symmetry_ids"] = symmetry_ids
        return outputs
    return policy.forward_legal_np(
        data["obs"][batch],
        legal_action_ids,
        data["legal_action_context"][batch],
        return_q=return_q,
    )


def _advantage_reweighted_policy_weights(
    policy_weights,
    outputs: dict,
    outcome_targets,
    has_outcome,
    mode: str,
    temperature: float,
    weight_cap: float,
    weight_floor: float,
):
    import torch

    stats = {
        "advantage_weight_rows": 0,
        "advantage_mean": 0.0,
        "advantage_weight_mean": 1.0,
        "advantage_weight_min": 1.0,
        "advantage_weight_max": 1.0,
    }
    if mode == "none" or outcome_targets is None or "value" not in outputs:
        return policy_weights, stats
    if mode != "outcome_value":
        raise ValueError(f"unknown advantage policy weighting mode: {mode}")
    temperature = max(float(temperature), 1.0e-6)
    cap = max(float(weight_cap), 1.0e-6)
    floor = max(float(weight_floor), 0.0)
    if floor > cap:
        floor = cap
    with torch.no_grad():
        values = outputs["value"].detach().float()
        targets = outcome_targets.detach().float()
        has = has_outcome.bool() if has_outcome is not None else torch.ones_like(targets, dtype=torch.bool)
        active = (policy_weights > 0.0) & has & torch.isfinite(targets) & torch.isfinite(values)
        if not bool(active.any().item()):
            return policy_weights, stats
        advantages = targets - values
        raw = torch.exp(torch.clamp(advantages / temperature, min=-20.0, max=20.0))
        raw = raw.clamp(min=floor, max=cap)
        active_weights = policy_weights[active]
        raw_active = raw[active]
        mean_multiplier = (active_weights * raw_active).sum() / active_weights.sum().clamp_min(1.0e-6)
        multiplier = torch.ones_like(policy_weights)
        multiplier[active] = raw[active] / mean_multiplier.clamp_min(1.0e-6)
        updated = policy_weights * multiplier
        stats = {
            "advantage_weight_rows": int(active.sum().item()),
            "advantage_mean": float(advantages[active].mean().item()),
            "advantage_weight_mean": float(multiplier[active].mean().item()),
            "advantage_weight_min": float(multiplier[active].min().item()),
            "advantage_weight_max": float(multiplier[active].max().item()),
        }
        return updated, stats


# CAT-100 auxiliary subgoal heads: (data field == model output key, loss kind).
# binary -> BCE-with-logits, scalar -> MSE, categorical -> cross-entropy with a
# -1 ignore. Unlabeled rows are masked (non-finite for binary/scalar, <0 for
# categorical), so a partially-labeled corpus never poisons a head.
_AUX_SUBGOAL_SPECS = (
    ("aux_longest_road", "binary"),
    ("aux_largest_army", "binary"),
    ("aux_vp_in_n", "scalar"),
    ("aux_next_settlement", "categorical"),
    ("aux_robber_target", "categorical"),
)


def _aux_subgoal_loss(
    outputs: dict,
    data: dict,
    batch: np.ndarray,
    device,
    *,
    symmetry=None,
    symmetry_ids: np.ndarray | None = None,
) -> tuple:
    """Combined CAT-100 auxiliary-subgoal loss over head/target pairs present in
    BOTH the model outputs and the corpus. Returns (loss_tensor, active_heads).

    A no-op (returns 0.0, 0) only when no aux head is built (a heads-off model is
    unaffected). Each head contributes its own masked mean; the shared
    --aux-subgoal-loss-weight scales their sum (UNREAL uniform small-weight
    scheme, arXiv 1611.05397).

    CAT-105 (loud fail on requested-but-inert aux): this function is only called
    when --aux-subgoal-loss-weight > 0. If the model WAS built with the aux heads
    (their outputs are present) but the corpus carries NONE of the aux target
    columns, the objective would silently train as a no-op (active_heads stays 0,
    loss stays 0.0) -- the exact silent-inert class 1b99c56 fixed for the value
    head. Raise instead, so a mis-provisioned aux run fails loud at the first
    batch rather than wasting a whole run. (A corpus that HAS the columns but a
    given batch happens to hold no valid targets is fine -- that is transient and
    does not raise.)"""
    import torch
    from torch import nn

    heads_present = any(field in outputs for field, _ in _AUX_SUBGOAL_SPECS)
    labels_present = any(field in data for field, _ in _AUX_SUBGOAL_SPECS)
    if heads_present and not labels_present:
        wanted = ", ".join(field for field, _ in _AUX_SUBGOAL_SPECS)
        raise ValueError(
            "CAT-105: --aux-subgoal-loss-weight > 0 with the aux-subgoal heads built "
            "(--aux-subgoal-heads), but the training corpus carries NONE of the aux "
            f"target columns ({wanted}). The aux objective would train as a silent "
            "no-op. Build/select a corpus that includes the aux target fields (see "
            "catan_zero.rl.aux_subgoal_targets), or drop --aux-subgoal-heads / set "
            "--aux-subgoal-loss-weight 0."
        )

    total = torch.zeros((), dtype=torch.float32, device=device)
    active_heads = 0
    for field, kind in _AUX_SUBGOAL_SPECS:
        if field not in outputs or field not in data:
            continue
        raw = np.asarray(data[field][batch])
        if symmetry_ids is not None and field in {
            "aux_next_settlement",
            "aux_robber_target",
        }:
            if symmetry is None:
                raise ValueError("spatial aux relabel requires the D6 symmetry tables")
            g = np.asarray(symmetry_ids, dtype=np.int64)
            if g.shape != (len(batch),):
                raise ValueError(
                    "spatial aux symmetry ids must have one entry per batch row"
                )
            table = (
                symmetry.fwd_vertex[g]
                if field == "aux_next_settlement"
                else symmetry.fwd_hex[g]
            )
            raw = symmetry._remap_values(np.asarray(raw), table)
        pred = outputs[field]
        if kind == "categorical":
            target = torch.as_tensor(raw.astype(np.int64), device=device)
            valid = target >= 0
            if not bool(valid.any().item()):
                continue
            per = nn.functional.cross_entropy(
                pred, target.clamp_min(0), reduction="none"
            )
        elif kind == "binary":
            target = torch.as_tensor(raw.astype(np.float32), device=device)
            valid = torch.isfinite(target)
            if not bool(valid.any().item()):
                continue
            per = nn.functional.binary_cross_entropy_with_logits(
                pred, torch.nan_to_num(target), reduction="none"
            )
        else:  # scalar (vp_in_n): plain regression.
            target = torch.as_tensor(raw.astype(np.float32), device=device)
            valid = torch.isfinite(target)
            if not bool(valid.any().item()):
                continue
            per = nn.functional.mse_loss(
                pred, torch.nan_to_num(target), reduction="none"
            )
        weight = valid.to(per.dtype)
        total = total + (per * weight).sum() / weight.sum().clamp_min(1.0)
        active_heads += 1
    return total, active_heads


def _train_xdim_batch(
    policy,
    optimizer,
    data: dict,
    batch: np.ndarray,
    policy_sample_weights: np.ndarray,
    value_sample_weights: np.ndarray,
    soft_target_temperature: float,
    soft_target_weight: float,
    soft_target_source: str,
    soft_target_min_legal_coverage: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    final_vp_loss_weight: float,
    q_loss_weight: float,
    q_skip_teacher_prefixes: tuple[str, ...],
    vps_to_win: int,
    advantage_policy_weighting: str,
    advantage_temperature: float,
    advantage_weight_cap: float,
    advantage_weight_floor: float,
    amp: str = "none",
    *,
    diagnostics: bool = True,
    truncated_vp_margin_value_weight: float = 0.0,
    policy_kl_anchor_weight: float = 0.0,
    policy_kl_anchor_direction: str = "forward",
    value_uncertainty_loss_weight: float = 0.0,
    aux_subgoal_loss_weight: float = 0.0,
    moe_balance_loss_weight: float = 0.0,
    value_categorical_loss_weight: float = 0.0,
    value_hlgauss_sigma_ratio: float = 0.75,
    value_target_lambda: float = 1.0,
    value_root_blend_phases: tuple[str, ...] = (),
    value_root_blend_global_compat: bool = False,
    symmetry=None,
    symmetry_rng=None,
    symmetry_relabel_events: bool = True,
    grad_accum_steps: int = 1,
    accum_do_zero_grad: bool = True,
    accum_do_step: bool = True,
    policy_aux_data=None,
    policy_aux_batch: np.ndarray | None = None,
    policy_aux_sample_weights: np.ndarray | None = None,
) -> dict:
    import torch
    from torch import nn

    legal_action_ids = data["legal_action_ids"][batch]
    actions_np = data["action_taken"][batch].astype(np.int64)
    target_columns = _target_columns(legal_action_ids, actions_np)
    target = torch.as_tensor(target_columns, dtype=torch.long, device=policy.device)
    policy_weights = torch.as_tensor(
        policy_sample_weights[batch],
        dtype=torch.float32,
        device=policy.device,
    )
    value_weights = torch.as_tensor(
        value_sample_weights[batch],
        dtype=torch.float32,
        device=policy.device,
    )
    with _amp_context(policy.device, amp):
        outputs = _forward_legal_np_for_batch(
            policy,
            data,
            batch,
            legal_action_ids,
            return_q=float(q_loss_weight) != 0.0,
            symmetry=symmetry,
            symmetry_rng=symmetry_rng,
            symmetry_relabel_events=symmetry_relabel_events,
        )
        hard_loss = nn.functional.cross_entropy(outputs["logits"], target, reduction="none")
        soft_targets, has_soft, soft_support = _soft_targets_legal(
            data,
            batch,
            policy.device,
            soft_target_temperature,
            soft_target_source,
            soft_target_min_legal_coverage,
        )
        if soft_targets is not None:
            log_probs = _support_log_softmax(outputs["logits"], soft_support)
            soft_loss = -(soft_targets * log_probs).sum(dim=-1)
            alpha = float(np.clip(soft_target_weight, 0.0, 1.0))
            per_sample_loss = torch.where(
                has_soft,
                alpha * soft_loss + (1.0 - alpha) * hard_loss,
                hard_loss,
            )
        else:
            per_sample_loss = hard_loss
        (
            outcome_targets,
            vp_targets,
            has_outcome,
            has_vp_target,
            value_outcome_targets,
            value_has_outcome,
            outcome_confidence,
        ) = _value_targets(
            data,
            batch,
            policy.device,
            vps_to_win,
            truncated_vp_margin_value_weight=truncated_vp_margin_value_weight,
        )
        # CAT-39: truncated mask + stored search root value for the value-target
        # lambda blend. Both are read best-effort so shards without the columns
        # (every current shard) leave the targets untouched -- root_value blending
        # is a gen-1-onward lever, lambda=1.0 (default) is a pure no-op.
        truncated_mask = torch.as_tensor(
            np.asarray(
                _batch_array_or_fill(
                    data,
                    "truncated",
                    batch,
                    False,
                    dtype=np.bool_,
                ),
                dtype=np.bool_,
            ),
            dtype=torch.bool,
            device=policy.device,
        )
        root_value, root_value_mask = _root_value_targets(data, batch, policy.device)
        # Scalar-space blend for the MSE control arm: value target = lambda*z +
        # (1-lambda)*V_search on rows carrying a stored root value. Distribution-
        # space blend for the HL-Gauss head happens below; the two are consistent
        # because HL-Gauss preserves the expectation and blending is linear.
        if (
            float(value_target_lambda) != 1.0
            and value_outcome_targets is not None
            and root_value is not None
        ):
            lam = float(value_target_lambda)
            blend_rows = _value_root_blend_mask(
                data,
                batch,
                policy.device,
                root_value_mask,
                value_has_outcome,
                truncated_mask,
                phases=value_root_blend_phases,
                global_compat=value_root_blend_global_compat,
            )
            value_outcome_targets = torch.where(
                blend_rows,
                lam * value_outcome_targets + (1.0 - lam) * root_value,
                value_outcome_targets,
            )
        # Advantage reweighting stays on the policy-safe (unfilled) outcome/has_outcome --
        # FIX F3 is scoped to the value head only, must not leak into POLICY weighting.
        policy_weights, advantage_stats = _advantage_reweighted_policy_weights(
            policy_weights,
            outputs,
            outcome_targets,
            has_outcome,
            advantage_policy_weighting,
            advantage_temperature,
            advantage_weight_cap,
            advantage_weight_floor,
        )
        policy_loss_sum, policy_loss_denominator = _weighted_loss_parts(
            per_sample_loss, policy_weights
        )
        policy_aux_active_count = 0
        if policy_aux_batch is not None:
            if policy_aux_data is None or policy_aux_sample_weights is None:
                raise ValueError("incomplete policy auxiliary batch inputs")
            aux_legal_ids = policy_aux_data["legal_action_ids"][policy_aux_batch]
            aux_actions = torch.as_tensor(
                _target_columns(
                    aux_legal_ids,
                    policy_aux_data["action_taken"][policy_aux_batch].astype(np.int64),
                ),
                dtype=torch.long,
                device=policy.device,
            )
            aux_weights = torch.as_tensor(
                policy_aux_sample_weights[policy_aux_batch],
                dtype=torch.float32,
                device=policy.device,
            )
            if not bool((aux_weights > 0.0).all().item()):
                raise ValueError("policy auxiliary sampler admitted an inactive row")
            aux_outputs = _forward_legal_np_for_batch(
                policy,
                policy_aux_data,
                policy_aux_batch,
                aux_legal_ids,
                return_q=False,
                symmetry=symmetry,
                symmetry_rng=symmetry_rng,
                symmetry_relabel_events=symmetry_relabel_events,
            )
            aux_hard = nn.functional.cross_entropy(
                aux_outputs["logits"], aux_actions, reduction="none"
            )
            aux_soft, aux_has_soft, aux_support = _soft_targets_legal(
                policy_aux_data,
                policy_aux_batch,
                policy.device,
                soft_target_temperature,
                soft_target_source,
                soft_target_min_legal_coverage,
            )
            if aux_soft is not None:
                aux_log_probs = _support_log_softmax(aux_outputs["logits"], aux_support)
                aux_soft_loss = -(aux_soft * aux_log_probs).sum(dim=-1)
                alpha = float(np.clip(soft_target_weight, 0.0, 1.0))
                aux_per_sample = torch.where(
                    aux_has_soft,
                    alpha * aux_soft_loss + (1.0 - alpha) * aux_hard,
                    aux_hard,
                )
            else:
                aux_per_sample = aux_hard
            aux_sum, aux_denominator = _weighted_loss_parts(aux_per_sample, aux_weights)
            policy_loss_sum = policy_loss_sum + aux_sum
            policy_loss_denominator = policy_loss_denominator + aux_denominator
            policy_aux_active_count = int(len(policy_aux_batch))
        policy_loss = _weighted_mean_from_parts(
            policy_loss_sum, policy_loss_denominator
        )
        value_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        final_vp_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        q_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        if value_outcome_targets is not None and "value" in outputs:
            value_error = nn.functional.mse_loss(
                outputs["value"], value_outcome_targets, reduction="none"
            )
            value_loss = _weighted_mean_loss(
                value_error,
                value_weights * outcome_confidence,
                mask=value_has_outcome,
            )
            value_loss_sum, value_loss_denominator = _weighted_loss_parts(
                value_error,
                value_weights * outcome_confidence,
                mask=value_has_outcome,
            )
        else:
            value_loss_sum, value_loss_denominator = _zero_loss_parts(policy.device)
        if vp_targets is not None and "final_vp" in outputs:
            vp_error = nn.functional.mse_loss(outputs["final_vp"], vp_targets, reduction="none")
            final_vp_loss = _weighted_mean_loss(
                vp_error,
                value_weights,
                mask=has_vp_target,
            )
            final_vp_loss_sum, final_vp_loss_denominator = _weighted_loss_parts(
                vp_error,
                value_weights,
                mask=has_vp_target,
            )
        else:
            final_vp_loss_sum, final_vp_loss_denominator = _zero_loss_parts(policy.device)
        q_loss_sum, q_loss_denominator = _zero_loss_parts(policy.device)
        if float(q_loss_weight) != 0.0 and "q_values" in outputs:
            q_loss, q_loss_sum, q_loss_denominator = _q_score_loss_parts(
                outputs["q_values"],
                data,
                batch,
                policy_weights,
                policy.device,
                q_skip_teacher_prefixes=q_skip_teacher_prefixes,
            )
        # Policy-KL anchor (unfreeze-with-KL value-repair recipe): pull the trained
        # policy toward the seed's recorded prior_policy. Only computed when enabled,
        # so a 0-weight run is bit-identical to pre-anchor behavior.
        kl_anchor_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        kl_anchor_loss_sum, kl_anchor_loss_denominator = _zero_loss_parts(policy.device)
        if float(policy_kl_anchor_weight) != 0.0:
            _anchor = _policy_kl_anchor_loss_parts(
                data,
                batch,
                outputs["logits"],
                policy.device,
                direction=policy_kl_anchor_direction,
            )
            if _anchor is not None:
                (
                    kl_anchor_loss,
                    kl_anchor_loss_sum,
                    kl_anchor_loss_denominator,
                ) = _anchor
        # Value-uncertainty auxiliary head: regress the value head's own squared
        # error (z - v)^2 with a stop-gradient on v (KataGo short-term-error style;
        # Huber loss because the target is already a squared quantity). No-op unless
        # the head is present in outputs and the weight is nonzero.
        value_uncertainty_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        if (
            float(value_uncertainty_loss_weight) != 0.0
            and "value_uncertainty" in outputs
            and value_outcome_targets is not None
            and "value" in outputs
        ):
            uncertainty_target = (value_outcome_targets - outputs["value"].detach()) ** 2
            uncertainty_error = nn.functional.smooth_l1_loss(
                outputs["value_uncertainty"], uncertainty_target, reduction="none"
            )
            value_uncertainty_loss = _weighted_mean_loss(
                uncertainty_error,
                value_weights * outcome_confidence,
                mask=value_has_outcome,
            )
        # HL-Gauss categorical value head (CAT-39): cross-entropy against a
        # Gaussian-smeared win-loss target, with truncated rows routed to the
        # dedicated truncation class (R9 support = win/loss + truncation ONLY;
        # VP-margin lives on the separate aux head). In HL-Gauss mode the scalar
        # head above is diagnostic-only unless the explicit scalar-aux weight is
        # nonzero; the matched MSE control is a separate run.
        value_categorical_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        value_categorical_loss_sum, value_categorical_loss_denominator = (
            _zero_loss_parts(policy.device)
        )
        value_categorical_clean_loss_sum, value_categorical_clean_loss_denominator = (
            _zero_loss_parts(policy.device)
        )
        (
            value_categorical_truncated_loss_sum,
            value_categorical_truncated_loss_denominator,
        ) = _zero_loss_parts(policy.device)
        if (
            float(value_categorical_loss_weight) != 0.0
            and "value_categorical_logits" in outputs
            and outcome_targets is not None
        ):
            cat_logits = outputs["value_categorical_logits"].float()
            n_out = int(cat_logits.shape[-1])
            has_trunc_class = n_out > int(policy.model.value_categorical_bins)
            bins = int(policy.model.value_categorical_bins)
            # Real win/loss target on the continuous axis; truncated rows one-hot
            # on the truncation class (uses the policy-safe raw outcome, NOT the
            # F3 VP-margin-filled value_outcome_targets, per R9).
            cat_targets = _hl_gauss_value_targets(
                outcome_targets,
                bins,
                sigma_ratio=value_hlgauss_sigma_ratio,
                truncated=truncated_mask if has_trunc_class else None,
                add_truncation_class=has_trunc_class,
            )
            # Distribution-space value-target lambda blend: mix the realised-
            # outcome distribution with the fresh search-root-value distribution
            # BEFORE collapsing to a scalar (never blend scalars then discretize).
            if (
                float(value_target_lambda) != 1.0
                and root_value is not None
            ):
                lam = float(value_target_lambda)
                rv_targets = _hl_gauss_value_targets(
                    root_value,
                    bins,
                    sigma_ratio=value_hlgauss_sigma_ratio,
                    truncated=None,
                    add_truncation_class=has_trunc_class,
                )
                blend_rows = _value_root_blend_mask(
                    data,
                    batch,
                    policy.device,
                    root_value_mask,
                    has_outcome,
                    truncated_mask,
                    phases=value_root_blend_phases,
                    global_compat=value_root_blend_global_compat,
                ).unsqueeze(-1)
                cat_targets = torch.where(
                    blend_rows,
                    lam * cat_targets + (1.0 - lam) * rv_targets,
                    cat_targets,
                )
            cat_log_probs = torch.nn.functional.log_softmax(cat_logits, dim=-1)
            cat_ce = -(cat_targets * cat_log_probs).sum(dim=-1)
            # Trainable rows: a real outcome OR a truncation label (both are
            # explicit labels). Rows with neither are masked out. Match the
            # historical MSE arm's truncation mass so the tournament does not
            # silently give categorical truncations 4x the default weight.
            cat_mask = has_outcome | truncated_mask if has_trunc_class else has_outcome
            categorical_value_weights = value_weights * torch.where(
                truncated_mask,
                torch.full_like(
                    value_weights, float(truncated_vp_margin_value_weight)
                ),
                torch.ones_like(value_weights),
            )
            (
                value_categorical_loss_sum,
                value_categorical_loss_denominator,
            ) = _weighted_loss_parts(
                cat_ce,
                categorical_value_weights,
                mask=cat_mask,
            )
            # Training must use the same DDP-global denominator semantics as
            # scalar MSE/policy CE.  The local parts above are telemetry only;
            # averaging local rank means would bias gradients whenever masks or
            # per-game weights differ across ranks.
            value_categorical_loss = _weighted_mean_loss(
                cat_ce,
                categorical_value_weights,
                mask=cat_mask,
            )
            (
                value_categorical_clean_loss_sum,
                value_categorical_clean_loss_denominator,
            ) = _weighted_loss_parts(
                cat_ce,
                categorical_value_weights,
                mask=has_outcome & ~truncated_mask,
            )
            if has_trunc_class:
                (
                    value_categorical_truncated_loss_sum,
                    value_categorical_truncated_loss_denominator,
                ) = _weighted_loss_parts(
                    cat_ce,
                    categorical_value_weights,
                    mask=truncated_mask,
                )
        # CAT-100 auxiliary subgoal heads (longest-road/largest-army/VP-in-N/
        # next-settlement/robber-target). No-op unless the weight is nonzero AND
        # the model has the heads AND the corpus has the target fields.
        aux_subgoal_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        aux_subgoal_active_heads = 0
        if float(aux_subgoal_loss_weight) != 0.0:
            aux_subgoal_loss, aux_subgoal_active_heads = _aux_subgoal_loss(
                outputs,
                data,
                batch,
                policy.device,
                symmetry=symmetry,
                symmetry_ids=outputs.get("_symmetry_ids"),
            )
        moe_balance_loss = outputs.get("moe_balance_metric")
        if moe_balance_loss is None:
            moe_balance_loss = torch.tensor(
                0.0, dtype=torch.float32, device=policy.device
            )
            if (
                float(moe_balance_loss_weight) != 0.0
                and int(getattr(policy.config, "moe_routed_experts", 0)) > 0
            ):
                raise ValueError(
                    "MoE balance loss requested but the model emitted no routing metric"
                )
        loss = (
            float(policy_loss_weight) * policy_loss
            + float(value_loss_weight) * value_loss
            + float(final_vp_loss_weight) * final_vp_loss
            + float(q_loss_weight) * q_loss
            + float(policy_kl_anchor_weight) * kl_anchor_loss
            + float(value_uncertainty_loss_weight) * value_uncertainty_loss
            + float(value_categorical_loss_weight) * value_categorical_loss
            + float(aux_subgoal_loss_weight) * aux_subgoal_loss
            + float(moe_balance_loss_weight) * moe_balance_loss
        )
        objective_gradient_interference = None
        if diagnostics and accum_do_step:
            # Exact configured task objectives. Value-head LR groups cannot scale
            # either objective's gradient through the shared transformer.
            policy_objective = (
                float(policy_loss_weight) * policy_loss
                + float(q_loss_weight) * q_loss
                + float(policy_kl_anchor_weight) * kl_anchor_loss
            )
            value_objective = (
                float(value_loss_weight) * value_loss
                + float(final_vp_loss_weight) * final_vp_loss
                + float(value_uncertainty_loss_weight) * value_uncertainty_loss
                + float(value_categorical_loss_weight) * value_categorical_loss
                + float(aux_subgoal_loss_weight) * aux_subgoal_loss
            )
            objective_gradient_interference = _objective_gradient_interference(
                policy,
                policy_objective=policy_objective,
                value_objective=value_objective,
            )
    # C1 gradient accumulation. At grad_accum_steps==1 (accum_do_zero_grad and
    # accum_do_step both True) this is byte-identical to the pre-C1 path:
    # zero_grad, backward on the undivided loss, clip, step. For N>1 the loss is
    # divided by N and grads accumulate across N micro-batches; only the stepping
    # micro-batch zero-grads (at group start), all-reduces (the rest run under
    # no_sync), clips, and steps.
    if accum_do_zero_grad:
        optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite BC loss: {float(loss.detach().cpu())}")
    backward_loss = loss / float(grad_accum_steps) if int(grad_accum_steps) > 1 else loss
    backward_loss.backward()
    optimizer_observability = None
    if accum_do_step:
        observability_state = (
            _capture_optimizer_observability(policy) if diagnostics else None
        )
        pre_clip_total_grad_norm = _clip_grad_norm(policy, 1.0)
        optimizer.step()
        if observability_state is not None:
            optimizer_observability = _finish_optimizer_observability(
                policy,
                observability_state,
                pre_clip_total_grad_norm=pre_clip_total_grad_norm,
                max_grad_norm=1.0,
            )
            optimizer_observability["objective_gradient_interference"] = (
                objective_gradient_interference
            )
    predictions = torch.argmax(outputs["logits"], dim=-1)
    active = policy_weights > 0.0
    active_count = int(active.sum().item())
    accuracy = _masked_metric_mean((predictions == target).float(), active)
    top3_accuracy = _topk_legal_accuracy(outputs["logits"], target, k=3, mask=active)
    active_np = active.detach().cpu().numpy().astype(bool)
    if diagnostics:
        predictions_np = predictions.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        logits_np = outputs["logits"].float().detach().cpu().numpy()
        phase_stats = _field_stats(
            data,
            batch[active_np],
            predictions_np[active_np],
            target_np[active_np],
            logits_np[active_np],
            field="phase",
        )
        teacher_stats = _field_stats(
            data,
            batch[active_np],
            predictions_np[active_np],
            target_np[active_np],
            logits_np[active_np],
            field="teacher_name",
        )
        phase_stats_unforced = _field_stats_unforced(
            data,
            batch[active_np],
            predictions_np[active_np],
            target_np[active_np],
            logits_np[active_np],
            field="phase",
        )
    else:
        phase_stats = {}
        teacher_stats = {}
        phase_stats_unforced = {}
    q_score_rows_ge2 = (
        _q_score_rows_ge2(
            data,
            batch,
            q_skip_teacher_prefixes=q_skip_teacher_prefixes,
        )
        if float(q_loss_weight) != 0.0 or diagnostics
        else 0
    )
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "final_vp_loss": float(final_vp_loss.item()),
        "q_loss": float(q_loss.item()),
        "policy_kl_anchor_loss": float(kl_anchor_loss.item()),
        "policy_kl_anchor_loss_weighted_sum": float(kl_anchor_loss_sum.item()),
        "policy_kl_anchor_loss_weight_sum": float(
            kl_anchor_loss_denominator.item()
        ),
        "policy_kl_anchor_eligible_rows": int(kl_anchor_loss_denominator.item()),
        "value_uncertainty_loss": float(value_uncertainty_loss.item()),
        "aux_subgoal_loss": float(aux_subgoal_loss.item()),
        "moe_balance_loss": float(moe_balance_loss.item()),
        "aux_subgoal_active_heads": int(aux_subgoal_active_heads),
        "value_categorical_loss": float(value_categorical_loss.item()),
        "primary_value_loss": float(
            value_categorical_loss.item()
            if float(value_categorical_loss_weight) > 0.0
            else value_loss.item()
        ),
        "primary_value_loss_kind": (
            "hlgauss_ce"
            if float(value_categorical_loss_weight) > 0.0
            else "scalar_mse"
        ),
        "scalar_value_mse_diagnostic": float(value_loss.item()),
        "policy_loss_weighted_sum": float(policy_loss_sum.item()),
        "policy_loss_weight_sum": float(policy_loss_denominator.item()),
        "value_loss_weighted_sum": float(value_loss_sum.item()),
        "value_loss_weight_sum": float(value_loss_denominator.item()),
        "final_vp_loss_weighted_sum": float(final_vp_loss_sum.item()),
        "final_vp_loss_weight_sum": float(final_vp_loss_denominator.item()),
        "q_loss_weighted_sum": float(q_loss_sum.item()),
        "q_loss_weight_sum": float(q_loss_denominator.item()),
        "value_categorical_loss_weighted_sum": float(
            value_categorical_loss_sum.item()
        ),
        "value_categorical_loss_weight_sum": float(
            value_categorical_loss_denominator.item()
        ),
        "value_categorical_clean_loss_weighted_sum": float(
            value_categorical_clean_loss_sum.item()
        ),
        "value_categorical_clean_loss_weight_sum": float(
            value_categorical_clean_loss_denominator.item()
        ),
        "value_categorical_truncated_loss_weighted_sum": float(
            value_categorical_truncated_loss_sum.item()
        ),
        "value_categorical_truncated_loss_weight_sum": float(
            value_categorical_truncated_loss_denominator.item()
        ),
        "q_score_rows_ge2": q_score_rows_ge2,
        **advantage_stats,
        "soft_distillation_rows": int(has_soft.sum().item()) if soft_targets is not None else 0,
        "active_count": active_count,
        "policy_aux_active_count": int(policy_aux_active_count),
        "accuracy": float(accuracy.item()),
        "top3_accuracy": float(top3_accuracy.item()),
        "phase_stats": phase_stats,
        "teacher_stats": teacher_stats,
        "phase_stats_unforced": phase_stats_unforced,
        **(
            {"optimizer_observability": optimizer_observability}
            if optimizer_observability is not None
            else {}
        ),
    }


_OBJECTIVE_MATCHED_VALIDATION_MEANS = (
    "loss",
    "raw_batch_mean_loss",
    "component_reconstructed_loss",
    "policy_loss",
    "value_loss",
    "scalar_value_mse_diagnostic",
    "final_vp_loss",
    "q_loss",
    "policy_kl_anchor_loss",
    "value_uncertainty_loss",
    "aux_subgoal_loss",
    "moe_balance_loss",
    "value_categorical_loss",
    "value_categorical_clean_loss",
    "value_categorical_truncated_loss",
    "primary_value_loss",
    "accuracy",
    "top3_accuracy",
    "soft_distillation_fraction",
    "soft_distillation_active_fraction",
    "advantage_mean",
    "advantage_weight_mean",
    "prior_kl_model_prior_mean",
    "prior_kl_prior_model_mean",
    "prior_kl_target_prior_mean",
    "prior_kl_ratio",
    "active_policy_kl_target_model_mean",
    "active_policy_kl_target_prior_mean",
    "active_policy_teacher_gap_closure",
)


def objective_matched_validation_metrics(
    epoch_metrics: dict, *, require_matched: bool = False
) -> dict:
    """Return promotion-facing validation means, with historical fallback.

    Composite reports wrap their authenticated aggregate with measure metadata;
    ordinary and historical reports only have the raw ``validation`` object.
    Centralizing this choice prevents downstream sweep/adjudication code from
    accidentally continuing to rank new composite candidates by raw row mix.
    """
    matched = epoch_metrics.get("validation_objective_matched")
    if isinstance(matched, dict):
        metrics = matched.get("metrics")
        if matched.get("objective_matched") is True and isinstance(metrics, dict):
            return metrics
    if require_matched:
        raise ValueError(
            "authenticated composite adjudication requires objective-matched "
            "validation; raw concatenated-row fallback is not admissible"
        )
    legacy = epoch_metrics.get("validation")
    return legacy if isinstance(legacy, dict) else {}


def objective_matched_validation_component_metrics(
    epoch_metrics: dict, *, require_matched: bool = False
) -> dict[str, dict]:
    """Return per-component metrics from an authenticated matched wrapper."""

    matched = epoch_metrics.get("validation_objective_matched")
    if isinstance(matched, dict) and matched.get("objective_matched") is True:
        components = matched.get("components")
        if isinstance(components, dict) and components:
            result = {}
            for component_id, report in components.items():
                metrics = report.get("metrics") if isinstance(report, dict) else None
                if not isinstance(component_id, str) or not isinstance(metrics, dict):
                    raise ValueError("objective-matched component metrics are malformed")
                result[component_id] = metrics
            return result
    if require_matched:
        raise ValueError(
            "authenticated composite adjudication requires per-component "
            "objective-matched validation metrics"
        )
    return {}


def _weighted_validation_means(
    reports: list[dict], weights: np.ndarray
) -> dict[str, float]:
    """Combine mean-like validation fields under an explicit probability measure."""
    if not reports:
        raise SystemExit("cannot aggregate an empty validation report set")
    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.shape != (len(reports),) or not np.isfinite(normalized).all():
        raise SystemExit("validation aggregation weights are malformed")
    total = float(normalized.sum())
    if total <= 0.0:
        raise SystemExit("validation aggregation weights have no mass")
    normalized = normalized / total
    result: dict[str, float] = {}
    for key in _OBJECTIVE_MATCHED_VALIDATION_MEANS:
        if all(key in report for report in reports):
            result[key] = float(
                np.dot(
                    normalized,
                    np.asarray([float(report[key]) for report in reports]),
                )
            )
    return result


_TRAINING_OBJECTIVE_METRIC_KEYS = (
    "policy_loss",
    "value_loss",
    "final_vp_loss",
    "q_loss",
    "policy_kl_anchor_loss",
    "value_uncertainty_loss",
    "aux_subgoal_loss",
    "moe_balance_loss",
    "value_categorical_loss",
)


def _objective_measure_validation_aggregate(
    reports: list[dict], weights: np.ndarray
) -> tuple[dict[str, float], dict[str, dict[str, float]] | None]:
    """Aggregate validation under a row-sampling measure before normalization.

    A training loss is ``E_p[w * loss] / E_p[w]``. Averaging one normalized
    loss per game instead computes ``E_game[E_row[w*loss]/E_row[w]]`` and is
    different whenever active-policy or loser/value mass varies by game. The
    latter was incorrectly called objective-matched validation. Convert each
    game's sums to per-row densities, average those densities under the exact
    game/component draw probabilities, and divide only once.

    Historical/generic callbacks without sufficient statistics retain the old
    mean aggregation rather than fabricating exactness.
    """
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / float(normalized.sum())
    metrics = _weighted_validation_means(reports, normalized)
    coefficient_rows = [report.get("objective_coefficients") for report in reports]
    has_objective_coefficients = bool(coefficient_rows) and all(
        isinstance(row, dict) and row == coefficient_rows[0]
        for row in coefficient_rows
    )
    coefficients = coefficient_rows[0] if has_objective_coefficients else {}
    sufficient: dict[str, dict[str, float]] = {}
    for key in _TRAINING_OBJECTIVE_METRIC_KEYS:
        if not all(
            key in report
            and isinstance(report.get("loss_denominators"), dict)
            and key in report["loss_denominators"]
            and int(report.get("samples", 0)) > 0
            for report in reports
        ):
            continue
        numerator_density = 0.0
        denominator_density = 0.0
        for probability, report in zip(normalized, reports, strict=True):
            samples = float(report["samples"])
            denominator = float(report["loss_denominators"][key])
            numerator_density += (
                float(probability) * float(report[key]) * denominator / samples
            )
            denominator_density += float(probability) * denominator / samples
        metrics[key] = (
            numerator_density / denominator_density
            if denominator_density > 0.0
            else 0.0
        )
        sufficient[key] = {
            "weighted_numerator_per_sample": numerator_density,
            "weight_per_sample": denominator_density,
        }
    # These diagnostics are also conditional means.  In particular, averaging
    # per-game teacher-gap closure can Simpson-reverse an arm when games differ
    # in active-row density or target/prior gap. Reconstruct the KL densities
    # first, then form the ratio exactly once under the training measure.
    conditional_metrics = {
        "prior_kl_model_prior_mean": "prior_kl_rows",
        "prior_kl_prior_model_mean": "prior_kl_rows",
        "prior_kl_target_prior_mean": "prior_kl_rows",
        "active_policy_kl_target_model_mean": "active_policy_teacher_gap_rows",
        "active_policy_kl_target_prior_mean": "active_policy_teacher_gap_rows",
    }
    for key, denominator_key in conditional_metrics.items():
        if not all(
            key in report
            and denominator_key in report
            and int(report.get("samples", 0)) > 0
            for report in reports
        ):
            continue
        numerator_density = 0.0
        denominator_density = 0.0
        for probability, report in zip(normalized, reports, strict=True):
            samples = float(report["samples"])
            denominator = float(report[denominator_key])
            numerator_density += (
                float(probability) * float(report[key]) * denominator / samples
            )
            denominator_density += float(probability) * denominator / samples
        metrics[key] = (
            numerator_density / denominator_density
            if denominator_density > 0.0
            else 0.0
        )
        sufficient[key] = {
            "weighted_numerator_per_sample": numerator_density,
            "weight_per_sample": denominator_density,
        }
    target_model = metrics.get("active_policy_kl_target_model_mean")
    target_prior = metrics.get("active_policy_kl_target_prior_mean")
    if target_model is not None and target_prior is not None:
        metrics["active_policy_teacher_gap_closure"] = (
            1.0 - target_model / target_prior if target_prior > 1.0e-8 else 0.0
        )
    model_prior = metrics.get("prior_kl_model_prior_mean")
    target_prior_legacy = metrics.get("prior_kl_target_prior_mean")
    if model_prior is not None and target_prior_legacy is not None:
        metrics["prior_kl_ratio"] = (
            model_prior / target_prior_legacy
            if target_prior_legacy > 1.0e-8
            else 0.0
        )
    if coefficients and all(key in metrics for key in coefficients):
        exact_loss = sum(
            float(coefficients[key]) * float(metrics[key]) for key in coefficients
        )
        # Promotion-facing loss is now the exact configured population objective.
        # Preserve the historical batch/game-normalized statistic under its honest
        # name for diagnostics and backwards comparisons.
        metrics["raw_batch_mean_loss"] = metrics.get("loss", exact_loss)
        metrics["component_reconstructed_loss"] = exact_loss
        metrics["loss"] = exact_loss
    return metrics, sufficient or None


def evaluate_composite_validation_measure(
    data,
    validation_indices: np.ndarray,
    evaluate_indices,
) -> dict[str, object]:
    """Evaluate a composite under its authenticated training distribution.

    The v2 learner samples ``component -> game -> row``. Concatenating holdout
    rows instead samples components and games in proportion to their row counts;
    on A1 that diluted the replay component by about 3.15x. Evaluate every game
    independently so rows are uniform *within* a game, average games uniformly
    inside each component, then apply the descriptor's authenticated component
    ratios. ``evaluate_indices`` performs all DDP reductions, and every rank
    traverses this deterministic component/game order, so collectives remain
    aligned even when a game has fewer rows than ranks.
    """
    ratios = np.asarray(
        getattr(data, "component_game_sampling_ratios", tuple()), dtype=np.float64
    )
    component_ids = tuple(getattr(data, "component_ids", tuple()))
    corpora = tuple(getattr(data, "corpora", tuple()))
    if (
        ratios.shape != (len(corpora),)
        or len(component_ids) != len(corpora)
        or len(corpora) < 2
        or not np.isfinite(ratios).all()
        or np.any(ratios <= 0.0)
        or not math.isclose(float(ratios.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise SystemExit(
            "objective-matched validation requires authenticated composite ids/ratios"
        )
    indices = np.asarray(validation_indices, dtype=np.int64)
    distillation_scope_authenticated = bool(
        getattr(data, "policy_distillation_scope_authenticated", False)
    )
    distillation_indices = set(
        int(value)
        for value in getattr(data, "policy_distillation_component_indices", tuple())
    )
    if distillation_scope_authenticated and (
        not distillation_indices
        or any(value < 0 or value >= len(component_ids) for value in distillation_indices)
    ):
        raise SystemExit(
            "objective-matched validation received invalid policy distillation scope"
        )
    components = np.asarray(data.component_indices_for_rows(indices), dtype=np.int64)
    seeds = np.asarray(data["game_seed"][indices], dtype=np.int64)
    component_reports: dict[str, dict[str, object]] = {}
    all_game_reports: list[dict] = []
    all_game_weights: list[float] = []
    for component, component_id in enumerate(component_ids):
        positions = np.flatnonzero(components == component)
        if positions.size == 0:
            raise SystemExit(
                f"authenticated validation component {component_id!r} has no rows"
            )
        component_indices = indices[positions]
        component_seeds = seeds[positions]
        games = np.unique(component_seeds)
        game_reports: list[dict] = []
        rows_per_game: list[int] = []
        for game_seed in games:
            game_indices = component_indices[component_seeds == game_seed]
            game_reports.append(evaluate_indices(game_indices))
            rows_per_game.append(int(game_indices.size))
        game_uniform, sufficient = _objective_measure_validation_aggregate(
            game_reports, np.ones(len(game_reports), dtype=np.float64)
        )
        all_game_reports.extend(game_reports)
        all_game_weights.extend(
            [float(ratios[component]) / len(game_reports)] * len(game_reports)
        )
        component_reports[str(component_id)] = {
            "component_index": int(component),
            "policy_distillation_enabled": bool(
                not distillation_scope_authenticated
                or component in distillation_indices
            ),
            "authenticated_sampling_ratio": float(ratios[component]),
            "games": int(len(games)),
            "rows": int(positions.size),
            "min_rows_per_game": int(min(rows_per_game)),
            "max_rows_per_game": int(max(rows_per_game)),
            "metrics": game_uniform,
            **(
                {"objective_measure_sufficient_statistics": sufficient}
                if sufficient is not None
                else {}
            ),
        }
    aggregate, aggregate_sufficient = _objective_measure_validation_aggregate(
        all_game_reports, np.asarray(all_game_weights, dtype=np.float64)
    )
    return {
        "schema_version": "composite-validation-measure-v2",
        "measure": (
            "authenticated_component_then_uniform_game_then_uniform_row_"
            "with_objective_weight_density"
        ),
        "objective_matched": True,
        "samples": int(indices.size),
        "games": int(
            sum(int(report["games"]) for report in component_reports.values())
        ),
        "component_sampling_ratios": {
            str(component_id): float(ratio)
            for component_id, ratio in zip(component_ids, ratios, strict=True)
        },
        "policy_distillation_component_ids": (
            [component_ids[index] for index in sorted(distillation_indices)]
            if distillation_scope_authenticated
            else list(component_ids)
        ),
        "metrics": aggregate,
        "components": component_reports,
        **(
            {"objective_measure_sufficient_statistics": aggregate_sufficient}
            if aggregate_sufficient is not None
            else {}
        ),
    }


def evaluate_bc_batches(
    policy,
    data: dict,
    indices: np.ndarray,
    policy_sample_weights: np.ndarray,
    value_sample_weights: np.ndarray,
    batch_size: int,
    soft_target_temperature: float,
    soft_target_weight: float,
    soft_target_source: str,
    soft_target_min_legal_coverage: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    final_vp_loss_weight: float,
    q_loss_weight: float,
    q_skip_teacher_prefixes: tuple[str, ...],
    vps_to_win: int,
    advantage_policy_weighting: str,
    advantage_temperature: float,
    advantage_weight_cap: float,
    advantage_weight_floor: float,
    ddp: dict[str, int | bool],
    amp: str = "none",
    *,
    data_sharded: bool = False,
    truncated_vp_margin_value_weight: float = 0.0,
    policy_kl_anchor_weight: float = 0.0,
    policy_kl_anchor_direction: str = "forward",
    value_uncertainty_loss_weight: float = 0.0,
    aux_subgoal_loss_weight: float = 0.0,
    moe_balance_loss_weight: float = 0.0,
    value_categorical_loss_weight: float = 0.0,
    value_hlgauss_sigma_ratio: float = 0.75,
    value_target_lambda: float = 1.0,
    value_root_blend_phases: tuple[str, ...] = (),
    value_root_blend_global_compat: bool = False,
    data_loader_workers: int = 0,
    data_loader_prefetch: int = 2,
) -> dict:
    if len(indices) == 0:
        return {}
    eval_indices = np.asarray(indices, dtype=np.int64)
    if not data_sharded:
        eval_indices = _distributed_index_slice(eval_indices, ddp)
    previous_modes = _set_policy_training(policy, False)
    try:
        loss_sum = 0.0
        extra_sums: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "final_vp_loss": 0.0,
            "q_loss": 0.0,
            "policy_kl_anchor_loss": 0.0,
            "value_uncertainty_loss": 0.0,
            "aux_subgoal_loss": 0.0,
            "moe_balance_loss": 0.0,
            "value_categorical_loss": 0.0,
            "value_categorical_clean_loss": 0.0,
            "value_categorical_truncated_loss": 0.0,
            "q_score_rows_ge2": 0.0,
            "soft_distillation_rows": 0.0,
            "soft_distillation_active_rows": 0.0,
            "advantage_weight_rows": 0.0,
            "advantage_mean_sum": 0.0,
            "advantage_weight_mean_sum": 0.0,
            "prior_kl_rows": 0.0,
            "prior_kl_model_prior_sum": 0.0,
            "prior_kl_prior_model_sum": 0.0,
            "prior_kl_target_prior_sum": 0.0,
            "active_policy_teacher_gap_rows": 0.0,
            "active_policy_kl_target_model_sum": 0.0,
            "active_policy_kl_target_prior_sum": 0.0,
        }
        extra_denominators: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "final_vp_loss": 0.0,
            "q_loss": 0.0,
            "policy_kl_anchor_loss": 0.0,
            "value_uncertainty_loss": 0.0,
            "aux_subgoal_loss": 0.0,
            "moe_balance_loss": 0.0,
            "value_categorical_loss": 0.0,
            "value_categorical_clean_loss": 0.0,
            "value_categorical_truncated_loss": 0.0,
        }
        acc_sum = 0.0
        top3_sum = 0.0
        count = 0.0
        active_count = 0.0
        phase_stats = _empty_phase_stats()
        phase_stats_unforced = _empty_phase_stats()
        teacher_stats = _empty_phase_stats()
        eval_fn = _eval_xdim_batch if hasattr(policy, "forward_legal_np") else _eval_candidate_batch
        eval_fn_extra_kwargs: dict[str, object] = {}
        if eval_fn is _eval_xdim_batch:
            eval_fn_extra_kwargs = {
                "policy_kl_anchor_weight": float(policy_kl_anchor_weight),
                "policy_kl_anchor_direction": str(policy_kl_anchor_direction),
                "value_uncertainty_loss_weight": float(value_uncertainty_loss_weight),
                "aux_subgoal_loss_weight": float(aux_subgoal_loss_weight),
                "moe_balance_loss_weight": float(moe_balance_loss_weight),
                "value_categorical_loss_weight": float(value_categorical_loss_weight),
                "value_hlgauss_sigma_ratio": float(value_hlgauss_sigma_ratio),
                "value_target_lambda": float(value_target_lambda),
                "value_root_blend_phases": tuple(value_root_blend_phases),
                "value_root_blend_global_compat": bool(
                    value_root_blend_global_compat
                ),
            }
        # Validation used to bypass the streaming loader and synchronously
        # reconstruct every ragged memmap batch on the rank's main thread. On a
        # large validation sentinel that leaves the GPU idle while one CPU is
        # materialising the next batch. Reuse the exact training iterator: its
        # synchronous path preserves historical global indices, while its
        # threaded path materialises the same rows in deterministic order and
        # overlaps future CPU work with the current GPU forward pass.
        eval_order = np.arange(len(eval_indices), dtype=np.int64)
        for batch_data, batch, batch_policy_weights, batch_value_weights in (
            _iterate_training_batches(
                data,
                eval_order,
                eval_indices,
                batch_size,
                policy_sample_weights,
                value_sample_weights,
                num_workers=int(data_loader_workers),
                prefetch=int(data_loader_prefetch),
            )
        ):
            batch_metrics = eval_fn(
                policy,
                batch_data,
                batch,
                batch_policy_weights,
                batch_value_weights,
                soft_target_temperature,
                soft_target_weight,
                soft_target_source,
                soft_target_min_legal_coverage,
                policy_loss_weight,
                value_loss_weight,
                final_vp_loss_weight,
                q_loss_weight,
                q_skip_teacher_prefixes,
                vps_to_win,
                advantage_policy_weighting,
                advantage_temperature,
                advantage_weight_cap,
                advantage_weight_floor,
                amp,
                truncated_vp_margin_value_weight=truncated_vp_margin_value_weight,
                **eval_fn_extra_kwargs,
            )
            loss_sum += float(batch_metrics["loss"]) * len(batch)
            for key in (
                "policy_loss",
                "value_loss",
                "final_vp_loss",
                "q_loss",
                "policy_kl_anchor_loss",
                "value_uncertainty_loss",
                "aux_subgoal_loss",
                "moe_balance_loss",
                "value_categorical_loss",
                "value_categorical_clean_loss",
                "value_categorical_truncated_loss",
            ):
                weighted_sum_key = f"{key}_weighted_sum"
                weight_sum_key = f"{key}_weight_sum"
                if weighted_sum_key in batch_metrics and weight_sum_key in batch_metrics:
                    extra_sums[key] += float(batch_metrics[weighted_sum_key])
                    extra_denominators[key] += float(batch_metrics[weight_sum_key])
                else:
                    extra_sums[key] += float(batch_metrics.get(key, 0.0)) * len(batch)
                    extra_denominators[key] += float(len(batch))
            extra_sums["q_score_rows_ge2"] += float(batch_metrics.get("q_score_rows_ge2", 0.0))
            extra_sums["soft_distillation_rows"] += float(
                batch_metrics.get("soft_distillation_rows", 0.0)
            )
            extra_sums["soft_distillation_active_rows"] += float(
                batch_metrics.get("soft_distillation_active_rows", 0.0)
            )
            advantage_rows = float(batch_metrics.get("advantage_weight_rows", 0.0))
            extra_sums["advantage_weight_rows"] += advantage_rows
            extra_sums["advantage_mean_sum"] += (
                float(batch_metrics.get("advantage_mean", 0.0)) * advantage_rows
            )
            extra_sums["advantage_weight_mean_sum"] += (
                float(batch_metrics.get("advantage_weight_mean", 1.0)) * advantage_rows
            )
            extra_sums["prior_kl_rows"] += float(batch_metrics.get("prior_kl_rows", 0.0))
            extra_sums["prior_kl_model_prior_sum"] += float(
                batch_metrics.get("prior_kl_model_prior_sum", 0.0)
            )
            extra_sums["prior_kl_prior_model_sum"] += float(
                batch_metrics.get("prior_kl_prior_model_sum", 0.0)
            )
            extra_sums["prior_kl_target_prior_sum"] += float(
                batch_metrics.get("prior_kl_target_prior_sum", 0.0)
            )
            extra_sums["active_policy_teacher_gap_rows"] += float(
                batch_metrics.get("active_policy_teacher_gap_rows", 0.0)
            )
            extra_sums["active_policy_kl_target_model_sum"] += float(
                batch_metrics.get("active_policy_kl_target_model_sum", 0.0)
            )
            extra_sums["active_policy_kl_target_prior_sum"] += float(
                batch_metrics.get("active_policy_kl_target_prior_sum", 0.0)
            )
            batch_active_count = float(batch_metrics.get("active_count", len(batch)))
            acc_sum += float(batch_metrics["accuracy"]) * batch_active_count
            top3_sum += float(batch_metrics["top3_accuracy"]) * batch_active_count
            count += float(len(batch))
            active_count += batch_active_count
            _merge_phase_stats(phase_stats, batch_metrics["phase_stats"])
            _merge_phase_stats(
                phase_stats_unforced,
                batch_metrics.get("phase_stats_unforced", {}),
            )
            _merge_phase_stats(teacher_stats, batch_metrics["teacher_stats"])
        extra_sums = _reduce_named_sums(extra_sums, ddp)
        extra_denominators = _reduce_named_sums(extra_denominators, ddp)
        loss_sum, acc_sum, top3_sum, total_count = _reduce_epoch_metrics(
            loss_sum,
            acc_sum,
            top3_sum,
            count,
            ddp,
        )
        active_count_total = _reduce_scalar_sum(active_count, ddp)
        phase_stats = _reduce_nested_count_stats(phase_stats, ddp)
        phase_stats_unforced = _reduce_nested_count_stats(phase_stats_unforced, ddp)
        teacher_stats = _reduce_nested_count_stats(teacher_stats, ddp)
        policy_loss_eval = _metric_from_sum_denominator(
            extra_sums["policy_loss"], extra_denominators["policy_loss"]
        )
        value_loss_eval = _metric_from_sum_denominator(
            extra_sums["value_loss"], extra_denominators["value_loss"]
        )
        final_vp_loss_eval = _metric_from_sum_denominator(
            extra_sums["final_vp_loss"], extra_denominators["final_vp_loss"]
        )
        q_loss_eval = _metric_from_sum_denominator(
            extra_sums["q_loss"], extra_denominators["q_loss"]
        )
        auxiliary_loss_eval = {
            key: _metric_from_sum_denominator(
                extra_sums[key], extra_denominators[key]
            )
            for key in (
                "policy_kl_anchor_loss",
                "value_uncertainty_loss",
                "aux_subgoal_loss",
                "moe_balance_loss",
                "value_categorical_loss",
            )
        }
        categorical_breakdown_eval = {
            key: _metric_from_sum_denominator(
                extra_sums[key], extra_denominators[key]
            )
            for key in (
                "value_categorical_clean_loss",
                "value_categorical_truncated_loss",
            )
        }
        loss_eval = loss_sum / max(total_count, 1.0)
        component_reconstructed_loss = (
            float(policy_loss_weight) * policy_loss_eval
            + float(value_loss_weight) * value_loss_eval
            + float(final_vp_loss_weight) * final_vp_loss_eval
            + float(q_loss_weight) * q_loss_eval
            + float(policy_kl_anchor_weight)
            * auxiliary_loss_eval["policy_kl_anchor_loss"]
            + float(value_uncertainty_loss_weight)
            * auxiliary_loss_eval["value_uncertainty_loss"]
            + float(aux_subgoal_loss_weight) * auxiliary_loss_eval["aux_subgoal_loss"]
            + float(moe_balance_loss_weight)
            * auxiliary_loss_eval["moe_balance_loss"]
            + float(value_categorical_loss_weight)
            * auxiliary_loss_eval["value_categorical_loss"]
        )
        return {
            "samples": int(total_count),
            "loss": loss_eval,
            "raw_batch_mean_loss": loss_eval,
            "component_reconstructed_loss": component_reconstructed_loss,
            "policy_loss": policy_loss_eval,
            "value_loss": value_loss_eval,
            "scalar_value_mse_diagnostic": value_loss_eval,
            "final_vp_loss": final_vp_loss_eval,
            "q_loss": q_loss_eval,
            **auxiliary_loss_eval,
            **categorical_breakdown_eval,
            "primary_value_loss": (
                auxiliary_loss_eval["value_categorical_loss"]
                if float(value_categorical_loss_weight) > 0.0
                else value_loss_eval
            ),
            "primary_value_loss_kind": (
                "hlgauss_ce"
                if float(value_categorical_loss_weight) > 0.0
                else "scalar_mse"
            ),
            "loss_denominators": dict(extra_denominators),
            "objective_coefficients": {
                "policy_loss": float(policy_loss_weight),
                "value_loss": float(value_loss_weight),
                "final_vp_loss": float(final_vp_loss_weight),
                "q_loss": float(q_loss_weight),
                "policy_kl_anchor_loss": float(policy_kl_anchor_weight),
                "value_uncertainty_loss": float(value_uncertainty_loss_weight),
                "aux_subgoal_loss": float(aux_subgoal_loss_weight),
                "moe_balance_loss": float(moe_balance_loss_weight),
                "value_categorical_loss": float(value_categorical_loss_weight),
            },
            "q_score_rows_ge2": int(round(extra_sums["q_score_rows_ge2"])),
            "q_score_rows_ge2_fraction": extra_sums["q_score_rows_ge2"] / max(total_count, 1.0),
            "soft_distillation_rows": int(round(extra_sums["soft_distillation_rows"])),
            "soft_distillation_fraction": extra_sums["soft_distillation_rows"] / max(total_count, 1.0),
            "soft_distillation_active_rows": int(
                round(extra_sums["soft_distillation_active_rows"])
            ),
            "soft_distillation_active_fraction": (
                extra_sums["soft_distillation_active_rows"] / max(active_count_total, 1.0)
            ),
            "advantage_weight_rows": int(round(extra_sums["advantage_weight_rows"])),
            "advantage_mean": (
                extra_sums["advantage_mean_sum"]
                / max(extra_sums["advantage_weight_rows"], 1.0)
            ),
            "advantage_weight_mean": (
                extra_sums["advantage_weight_mean_sum"]
                / max(extra_sums["advantage_weight_rows"], 1.0)
            ),
            "accuracy_active_count": int(round(active_count_total)),
            "accuracy": acc_sum / max(active_count_total, 1.0),
            "top3_accuracy": top3_sum / max(active_count_total, 1.0),
            "phase_accuracy": _finalize_phase_stats(phase_stats),
            "phase_accuracy_excluding_forced": _finalize_phase_stats(phase_stats_unforced),
            "teacher_accuracy": _finalize_phase_stats(teacher_stats),
            # Legacy drift telemetry. This reverse-KL ratio is retained for report
            # compatibility and anchor diagnostics, but is not a calibrated teacher
            # uptake fraction and must not be used as an LR or launch gate.
            "prior_kl_rows": int(round(extra_sums["prior_kl_rows"])),
            "prior_kl_model_prior_mean": (
                extra_sums["prior_kl_model_prior_sum"] / max(extra_sums["prior_kl_rows"], 1.0)
            ),
            "prior_kl_prior_model_mean": (
                extra_sums["prior_kl_prior_model_sum"] / max(extra_sums["prior_kl_rows"], 1.0)
            ),
            "prior_kl_target_prior_mean": (
                extra_sums["prior_kl_target_prior_sum"] / max(extra_sums["prior_kl_rows"], 1.0)
            ),
            "prior_kl_ratio": (
                extra_sums["prior_kl_model_prior_sum"]
                / max(extra_sums["prior_kl_target_prior_sum"], 1.0e-6)
                if extra_sums["prior_kl_rows"] > 0
                else 0.0
            ),
            # Objective-aligned teacher uptake. Unlike the legacy prior_kl_* fields,
            # these rows exactly match positive policy weight + a usable
            # multi-action soft target + a recorded prior.
            **_active_policy_teacher_gap_report(
                rows=extra_sums["active_policy_teacher_gap_rows"],
                kl_target_model_sum=extra_sums[
                    "active_policy_kl_target_model_sum"
                ],
                kl_target_prior_sum=extra_sums[
                    "active_policy_kl_target_prior_sum"
                ],
            ),
        }
    finally:
        _restore_policy_training(policy, previous_modes)


def _run_teacher_quality_gate(
    data_path: Path,
    *,
    track: str,
    vps_to_win: int,
    strict: bool,
    production: bool,
    soft_target_min_legal_coverage: float,
    out_path: Path,
    ddp: dict[str, int | bool],
) -> None:
    if production:
        strict = True
    if not strict:
        return

    command = [
        sys.executable,
        "tools/report_teacher_data_quality.py",
        "--data",
        str(data_path),
        "--track",
        str(track),
        "--vps-to-win",
        str(int(vps_to_win)),
        "--out",
        str(out_path),
        "--soft-target-min-legal-coverage",
        str(float(soft_target_min_legal_coverage)),
    ]
    if production:
        command.append("--production-35m-teacher")
    else:
        command.append("--strict-35m-teacher")

    status = 0
    if int(ddp.get("rank", 0)) == 0:
        print(
            json.dumps(
                {
                    "progress": "teacher_quality_gate",
                    "command": command,
                    "production": bool(production),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        status = int(result.returncode)

    if ddp.get("enabled", False):
        import torch
        import torch.distributed as dist

        device = f"cuda:{int(ddp.get('local_rank', 0))}"
        status_tensor = torch.tensor([status], dtype=torch.int32, device=device)
        dist.broadcast(status_tensor, src=0)
        status = int(status_tensor.item())

    if status != 0:
        raise SystemExit(
            "teacher data quality gate failed; refusing to start BC training "
            f"on {data_path}"
        )


def _eval_candidate_batch(
    policy,
    data: dict,
    batch: np.ndarray,
    policy_sample_weights: np.ndarray,
    value_sample_weights: np.ndarray,
    soft_target_temperature: float,
    soft_target_weight: float,
    soft_target_source: str,
    soft_target_min_legal_coverage: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    final_vp_loss_weight: float,
    q_loss_weight: float,
    q_skip_teacher_prefixes: tuple[str, ...],
    vps_to_win: int,
    advantage_policy_weighting: str,
    advantage_temperature: float,
    advantage_weight_cap: float,
    advantage_weight_floor: float,
    amp: str = "none",
    *,
    truncated_vp_margin_value_weight: float = 0.0,
) -> dict:
    del final_vp_loss_weight, q_loss_weight, amp
    del q_skip_teacher_prefixes
    del advantage_policy_weighting, advantage_temperature, advantage_weight_cap, advantage_weight_floor
    import torch
    from torch import nn

    with torch.no_grad():
        obs = torch.as_tensor(
            normalize_observations(data["obs"][batch]),
            dtype=torch.float32,
            device=policy.device,
        )
        context = _dense_context(data, batch, policy.action_size, policy.context_action_feature_size)
        context_t = torch.as_tensor(context, dtype=torch.float32, device=policy.device)
        actions = torch.as_tensor(data["action_taken"][batch].astype(np.int64), device=policy.device)
        valid = _valid_lists(data["legal_action_ids"][batch])
        logits, values = policy.forward(obs, context_t)
        masked = _torch_ppo_masked_logits(logits, valid, policy.action_size)
        policy_weights = torch.as_tensor(
            policy_sample_weights[batch],
            dtype=torch.float32,
            device=policy.device,
        )
        value_weights = torch.as_tensor(
            value_sample_weights[batch],
            dtype=torch.float32,
            device=policy.device,
        )
        hard_loss = nn.functional.cross_entropy(masked, actions, reduction="none")
        soft_targets, has_soft, soft_support = _soft_targets_full(
            data,
            batch,
            policy.action_size,
            policy.device,
            soft_target_temperature,
            soft_target_source,
            soft_target_min_legal_coverage,
        )
        if soft_targets is not None:
            log_probs = _support_log_softmax(masked, soft_support)
            soft_loss = -(soft_targets * log_probs).sum(dim=-1)
            alpha = float(np.clip(soft_target_weight, 0.0, 1.0))
            per_sample_loss = torch.where(
                has_soft,
                alpha * soft_loss + (1.0 - alpha) * hard_loss,
                hard_loss,
            )
        else:
            per_sample_loss = hard_loss
        policy_loss_sum, policy_loss_denominator = _weighted_loss_parts(per_sample_loss, policy_weights)
        policy_loss = policy_loss_sum / torch.clamp(policy_loss_denominator, min=1e-6)
        _, _, _, _, outcome_targets, has_outcome, outcome_confidence = _value_targets(
            data,
            batch,
            policy.device,
            vps_to_win,
            truncated_vp_margin_value_weight=truncated_vp_margin_value_weight,
        )
        value_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        if outcome_targets is not None:
            value_error = nn.functional.mse_loss(values, outcome_targets, reduction="none")
            value_loss_sum, value_loss_denominator = _weighted_loss_parts(
                value_error,
                value_weights * outcome_confidence,
                mask=has_outcome,
            )
            value_loss = value_loss_sum / torch.clamp(value_loss_denominator, min=1e-6)
        else:
            value_loss_sum, value_loss_denominator = _zero_loss_parts(policy.device)
        loss = float(policy_loss_weight) * policy_loss + float(value_loss_weight) * value_loss
        predictions = torch.argmax(masked, dim=-1)
        active = policy_weights > 0.0
        active_count = int(active.sum().item())
        accuracy = _masked_metric_mean((predictions == actions).float(), active)
        top3_accuracy = _topk_full_accuracy(masked, actions, k=3, mask=active)
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = actions.detach().cpu().numpy()
        logits_np = masked.detach().cpu().numpy()
        active_np = active.detach().cpu().numpy().astype(bool)
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "final_vp_loss": 0.0,
            "q_loss": 0.0,
            "policy_loss_weighted_sum": float(policy_loss_sum.item()),
            "policy_loss_weight_sum": float(policy_loss_denominator.item()),
            "value_loss_weighted_sum": float(value_loss_sum.item()),
            "value_loss_weight_sum": float(value_loss_denominator.item()),
            "final_vp_loss_weighted_sum": 0.0,
            "final_vp_loss_weight_sum": 0.0,
            "q_loss_weighted_sum": 0.0,
            "q_loss_weight_sum": 0.0,
            "q_score_rows_ge2": 0,
            "soft_distillation_rows": int(has_soft.sum().item()) if soft_targets is not None else 0,
            "soft_distillation_active_rows": (
                int((has_soft & active).sum().item()) if soft_targets is not None else 0
            ),
            "active_count": active_count,
            "accuracy": float(accuracy.item()),
            "top3_accuracy": float(top3_accuracy.item()),
            "phase_stats": _field_stats(
                data,
                batch[active_np],
                predictions_np[active_np],
                targets_np[active_np],
                logits_np[active_np],
                field="phase",
            ),
            "phase_stats_unforced": _field_stats_unforced(
                data,
                batch[active_np],
                predictions_np[active_np],
                targets_np[active_np],
                logits_np[active_np],
                field="phase",
            ),
            "teacher_stats": _field_stats(
                data,
                batch[active_np],
                predictions_np[active_np],
                targets_np[active_np],
                logits_np[active_np],
                field="teacher_name",
            ),
        }


def _eval_xdim_batch(
    policy,
    data: dict,
    batch: np.ndarray,
    policy_sample_weights: np.ndarray,
    value_sample_weights: np.ndarray,
    soft_target_temperature: float,
    soft_target_weight: float,
    soft_target_source: str,
    soft_target_min_legal_coverage: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    final_vp_loss_weight: float,
    q_loss_weight: float,
    q_skip_teacher_prefixes: tuple[str, ...],
    vps_to_win: int,
    advantage_policy_weighting: str,
    advantage_temperature: float,
    advantage_weight_cap: float,
    advantage_weight_floor: float,
    amp: str = "none",
    *,
    truncated_vp_margin_value_weight: float = 0.0,
    policy_kl_anchor_weight: float = 0.0,
    policy_kl_anchor_direction: str = "forward",
    value_uncertainty_loss_weight: float = 0.0,
    aux_subgoal_loss_weight: float = 0.0,
    moe_balance_loss_weight: float = 0.0,
    value_categorical_loss_weight: float = 0.0,
    value_hlgauss_sigma_ratio: float = 0.75,
    value_target_lambda: float = 1.0,
    value_root_blend_phases: tuple[str, ...] = (),
    value_root_blend_global_compat: bool = False,
) -> dict:
    import torch
    from torch import nn

    with torch.no_grad():
        legal_action_ids = data["legal_action_ids"][batch]
        actions_np = data["action_taken"][batch].astype(np.int64)
        target_columns = _target_columns(legal_action_ids, actions_np)
        target = torch.as_tensor(target_columns, dtype=torch.long, device=policy.device)
        policy_weights = torch.as_tensor(
            policy_sample_weights[batch],
            dtype=torch.float32,
            device=policy.device,
        )
        value_weights = torch.as_tensor(
            value_sample_weights[batch],
            dtype=torch.float32,
            device=policy.device,
        )
        with _amp_context(policy.device, amp):
            outputs = _forward_legal_np_for_batch(
                policy,
                data,
                batch,
                legal_action_ids,
                return_q=float(q_loss_weight) != 0.0,
            )
            hard_loss = nn.functional.cross_entropy(outputs["logits"], target, reduction="none")
            soft_targets, has_soft, soft_support = _soft_targets_legal(
                data,
                batch,
                policy.device,
                soft_target_temperature,
                soft_target_source,
                soft_target_min_legal_coverage,
            )
            if soft_targets is not None:
                log_probs = _support_log_softmax(outputs["logits"], soft_support)
                soft_loss = -(soft_targets * log_probs).sum(dim=-1)
                alpha = float(np.clip(soft_target_weight, 0.0, 1.0))
                per_sample_loss = torch.where(
                    has_soft,
                    alpha * soft_loss + (1.0 - alpha) * hard_loss,
                    hard_loss,
                )
            else:
                per_sample_loss = hard_loss
            # Teacher-gap telemetry must follow the rows the POLICY objective can
            # actually update.  In PCR corpora, fast-search and forced rows still
            # carry stored target/prior distributions but have zero policy weight;
            # including them made the historical prior_kl_ratio compare against a
            # much larger, untrained target population.
            policy_active_for_teacher_gap = policy_weights > 0.0
            (
                outcome_targets,
                vp_targets,
                has_outcome,
                has_vp_target,
                value_outcome_targets,
                value_has_outcome,
                outcome_confidence,
            ) = _value_targets(
                data,
                batch,
                policy.device,
                vps_to_win,
                truncated_vp_margin_value_weight=truncated_vp_margin_value_weight,
            )
            truncated_mask = torch.as_tensor(
                np.asarray(
                    _batch_array_or_fill(
                        data,
                        "truncated",
                        batch,
                        False,
                        dtype=np.bool_,
                    ),
                    dtype=np.bool_,
                ),
                dtype=torch.bool,
                device=policy.device,
            )
            root_value, root_value_mask = _root_value_targets(
                data, batch, policy.device
            )
            if (
                float(value_target_lambda) != 1.0
                and value_outcome_targets is not None
                and root_value is not None
            ):
                lam = float(value_target_lambda)
                blend_rows = _value_root_blend_mask(
                    data,
                    batch,
                    policy.device,
                    root_value_mask,
                    value_has_outcome,
                    truncated_mask,
                    phases=value_root_blend_phases,
                    global_compat=value_root_blend_global_compat,
                )
                value_outcome_targets = torch.where(
                    blend_rows,
                    lam * value_outcome_targets + (1.0 - lam) * root_value,
                    value_outcome_targets,
                )
            # Advantage reweighting stays on the policy-safe (unfilled) outcome/has_outcome --
            # FIX F3 is scoped to the value head only, must not leak into POLICY weighting.
            policy_weights, advantage_stats = _advantage_reweighted_policy_weights(
                policy_weights,
                outputs,
                outcome_targets,
                has_outcome,
                advantage_policy_weighting,
                advantage_temperature,
                advantage_weight_cap,
                advantage_weight_floor,
            )
            policy_loss_sum, policy_loss_denominator = _weighted_loss_parts(per_sample_loss, policy_weights)
            policy_loss = policy_loss_sum / torch.clamp(policy_loss_denominator, min=1e-6)
            value_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            final_vp_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            q_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            if value_outcome_targets is not None and "value" in outputs:
                value_error = nn.functional.mse_loss(
                    outputs["value"], value_outcome_targets, reduction="none"
                )
                value_loss_sum, value_loss_denominator = _weighted_loss_parts(
                    value_error,
                    value_weights * outcome_confidence,
                    mask=value_has_outcome,
                )
                value_loss = value_loss_sum / torch.clamp(value_loss_denominator, min=1e-6)
            else:
                value_loss_sum, value_loss_denominator = _zero_loss_parts(policy.device)
            if vp_targets is not None and "final_vp" in outputs:
                vp_error = nn.functional.mse_loss(outputs["final_vp"], vp_targets, reduction="none")
                final_vp_loss_sum, final_vp_loss_denominator = _weighted_loss_parts(
                    vp_error,
                    value_weights,
                    mask=has_vp_target,
                )
                final_vp_loss = final_vp_loss_sum / torch.clamp(final_vp_loss_denominator, min=1e-6)
            else:
                final_vp_loss_sum, final_vp_loss_denominator = _zero_loss_parts(policy.device)
            q_loss_sum, q_loss_denominator = _zero_loss_parts(policy.device)
            if float(q_loss_weight) != 0.0 and "q_values" in outputs:
                q_loss, q_loss_sum, q_loss_denominator = _q_score_loss_parts(
                    outputs["q_values"],
                    data,
                    batch,
                    policy_weights,
                    policy.device,
                    q_skip_teacher_prefixes=q_skip_teacher_prefixes,
                )
            kl_anchor_loss = torch.tensor(
                0.0, dtype=torch.float32, device=policy.device
            )
            kl_anchor_loss_sum, kl_anchor_loss_denominator = _zero_loss_parts(
                policy.device
            )
            # Always measure the exact configured anchor direction and
            # authenticated replay scope during validation, including K0. A
            # zero coefficient still contributes nothing to the objective.
            anchor = _policy_kl_anchor_loss_parts(
                data,
                batch,
                outputs["logits"],
                policy.device,
                direction=policy_kl_anchor_direction,
            )
            if anchor is not None:
                (
                    kl_anchor_loss,
                    kl_anchor_loss_sum,
                    kl_anchor_loss_denominator,
                ) = anchor
            value_uncertainty_loss = torch.tensor(
                0.0, dtype=torch.float32, device=policy.device
            )
            if (
                float(value_uncertainty_loss_weight) != 0.0
                and "value_uncertainty" in outputs
                and value_outcome_targets is not None
                and "value" in outputs
            ):
                uncertainty_target = (
                    value_outcome_targets - outputs["value"].detach()
                ) ** 2
                uncertainty_error = nn.functional.smooth_l1_loss(
                    outputs["value_uncertainty"],
                    uncertainty_target,
                    reduction="none",
                )
                value_uncertainty_loss = _weighted_mean_loss(
                    uncertainty_error,
                    value_weights * outcome_confidence,
                    mask=value_has_outcome,
                )
            value_categorical_loss = torch.tensor(
                0.0, dtype=torch.float32, device=policy.device
            )
            value_categorical_loss_sum, value_categorical_loss_denominator = (
                _zero_loss_parts(policy.device)
            )
            (
                value_categorical_clean_loss_sum,
                value_categorical_clean_loss_denominator,
            ) = _zero_loss_parts(policy.device)
            (
                value_categorical_truncated_loss_sum,
                value_categorical_truncated_loss_denominator,
            ) = _zero_loss_parts(policy.device)
            if (
                float(value_categorical_loss_weight) != 0.0
                and "value_categorical_logits" in outputs
                and outcome_targets is not None
            ):
                cat_logits = outputs["value_categorical_logits"].float()
                bins = int(policy.model.value_categorical_bins)
                has_trunc_class = int(cat_logits.shape[-1]) > bins
                cat_targets = _hl_gauss_value_targets(
                    outcome_targets,
                    bins,
                    sigma_ratio=value_hlgauss_sigma_ratio,
                    truncated=truncated_mask if has_trunc_class else None,
                    add_truncation_class=has_trunc_class,
                )
                if float(value_target_lambda) != 1.0 and root_value is not None:
                    lam = float(value_target_lambda)
                    root_targets = _hl_gauss_value_targets(
                        root_value,
                        bins,
                        sigma_ratio=value_hlgauss_sigma_ratio,
                        truncated=None,
                        add_truncation_class=has_trunc_class,
                    )
                    blend_rows = _value_root_blend_mask(
                        data,
                        batch,
                        policy.device,
                        root_value_mask,
                        has_outcome,
                        truncated_mask,
                        phases=value_root_blend_phases,
                        global_compat=value_root_blend_global_compat,
                    ).unsqueeze(-1)
                    cat_targets = torch.where(
                        blend_rows,
                        lam * cat_targets + (1.0 - lam) * root_targets,
                        cat_targets,
                    )
                cat_log_probs = torch.nn.functional.log_softmax(
                    cat_logits, dim=-1
                )
                cat_error = -(cat_targets * cat_log_probs).sum(dim=-1)
                cat_mask = (
                    has_outcome | truncated_mask
                    if has_trunc_class
                    else has_outcome
                )
                categorical_value_weights = value_weights * torch.where(
                    truncated_mask,
                    torch.full_like(
                        value_weights, float(truncated_vp_margin_value_weight)
                    ),
                    torch.ones_like(value_weights),
                )
                (
                    value_categorical_loss_sum,
                    value_categorical_loss_denominator,
                ) = _weighted_loss_parts(
                    cat_error, categorical_value_weights, mask=cat_mask
                )
                value_categorical_loss = value_categorical_loss_sum / torch.clamp(
                    value_categorical_loss_denominator, min=1e-6
                )
                (
                    value_categorical_clean_loss_sum,
                    value_categorical_clean_loss_denominator,
                ) = _weighted_loss_parts(
                    cat_error,
                    categorical_value_weights,
                    mask=has_outcome & ~truncated_mask,
                )
                if has_trunc_class:
                    (
                        value_categorical_truncated_loss_sum,
                        value_categorical_truncated_loss_denominator,
                    ) = _weighted_loss_parts(
                        cat_error,
                        categorical_value_weights,
                        mask=truncated_mask,
                    )
            aux_subgoal_loss = torch.tensor(
                0.0, dtype=torch.float32, device=policy.device
            )
            aux_subgoal_active_heads = 0
            if float(aux_subgoal_loss_weight) != 0.0:
                aux_subgoal_loss, aux_subgoal_active_heads = _aux_subgoal_loss(
                    outputs, data, batch, policy.device
                )
            moe_balance_loss = outputs.get("moe_balance_metric")
            if moe_balance_loss is None:
                moe_balance_loss = torch.tensor(
                    0.0, dtype=torch.float32, device=policy.device
                )
                if (
                    float(moe_balance_loss_weight) != 0.0
                    and int(getattr(policy.config, "moe_routed_experts", 0)) > 0
                ):
                    raise ValueError(
                        "MoE balance loss requested but the model emitted no routing metric"
                    )
            loss = (
                float(policy_loss_weight) * policy_loss
                + float(value_loss_weight) * value_loss
                + float(final_vp_loss_weight) * final_vp_loss
                + float(q_loss_weight) * q_loss
                + float(policy_kl_anchor_weight) * kl_anchor_loss
                + float(value_uncertainty_loss_weight)
                * value_uncertainty_loss
                + float(aux_subgoal_loss_weight) * aux_subgoal_loss
                + float(moe_balance_loss_weight) * moe_balance_loss
                + float(value_categorical_loss_weight)
                * value_categorical_loss
            )
        predictions = torch.argmax(outputs["logits"], dim=-1)
        active = policy_weights > 0.0
        active_count = int(active.sum().item())
        accuracy = _masked_metric_mean((predictions == target).float(), active)
        top3_accuracy = _topk_legal_accuracy(outputs["logits"], target, k=3, mask=active)
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = target.detach().cpu().numpy()
        logits_np = outputs["logits"].float().detach().cpu().numpy()
        q_score_rows_ge2 = _q_score_rows_ge2(
            data,
            batch,
            q_skip_teacher_prefixes=q_skip_teacher_prefixes,
        )
        active_np = active.detach().cpu().numpy().astype(bool)
        # Success telemetry (gen-1 recipe): KL(model||prior_policy) vs the
        # reference KL(target_policy||prior_policy) on a held-out gen slice.
        prior_kl = _prior_kl_telemetry(data, batch, outputs["logits"], policy.device)
        if prior_kl is not None:
            has_prior = prior_kl["has_prior"]
            prior_kl_rows = int(has_prior.sum().item())
            kl_model_prior_sum = float(prior_kl["kl_model_prior"][has_prior].sum().item())
            kl_prior_model_sum = float(prior_kl["kl_prior_model"][has_prior].sum().item())
            kl_target_prior_sum = float(prior_kl["kl_target_prior"][has_prior].sum().item())
        else:
            prior_kl_rows = 0
            kl_model_prior_sum = 0.0
            kl_prior_model_sum = 0.0
            kl_target_prior_sum = 0.0
        teacher_gap = _active_policy_teacher_gap_telemetry(
            data,
            batch,
            outputs["logits"],
            policy.device,
            soft_targets=soft_targets,
            has_soft=has_soft,
            policy_active=policy_active_for_teacher_gap,
        )
        if teacher_gap is not None:
            teacher_gap_rows = int(teacher_gap["eligible"].sum().item())
            teacher_kl_target_model_sum = float(
                teacher_gap["kl_target_model"][teacher_gap["eligible"]].sum().item()
            )
            teacher_kl_target_prior_sum = float(
                teacher_gap["kl_target_prior"][teacher_gap["eligible"]].sum().item()
            )
        else:
            teacher_gap_rows = 0
            teacher_kl_target_model_sum = 0.0
            teacher_kl_target_prior_sum = 0.0
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "final_vp_loss": float(final_vp_loss.item()),
            "q_loss": float(q_loss.item()),
            "policy_kl_anchor_loss": float(kl_anchor_loss.item()),
            "policy_kl_anchor_loss_weighted_sum": float(
                kl_anchor_loss_sum.item()
            ),
            "policy_kl_anchor_loss_weight_sum": float(
                kl_anchor_loss_denominator.item()
            ),
            "policy_kl_anchor_eligible_rows": int(
                kl_anchor_loss_denominator.item()
            ),
            "value_uncertainty_loss": float(value_uncertainty_loss.item()),
            "aux_subgoal_loss": float(aux_subgoal_loss.item()),
            "moe_balance_loss": float(moe_balance_loss.item()),
            "aux_subgoal_active_heads": int(aux_subgoal_active_heads),
            "value_categorical_loss": float(value_categorical_loss.item()),
            "primary_value_loss": float(
                value_categorical_loss.item()
                if float(value_categorical_loss_weight) > 0.0
                else value_loss.item()
            ),
            "primary_value_loss_kind": (
                "hlgauss_ce"
                if float(value_categorical_loss_weight) > 0.0
                else "scalar_mse"
            ),
            "scalar_value_mse_diagnostic": float(value_loss.item()),
            "policy_loss_weighted_sum": float(policy_loss_sum.item()),
            "policy_loss_weight_sum": float(policy_loss_denominator.item()),
            "value_loss_weighted_sum": float(value_loss_sum.item()),
            "value_loss_weight_sum": float(value_loss_denominator.item()),
            "final_vp_loss_weighted_sum": float(final_vp_loss_sum.item()),
            "final_vp_loss_weight_sum": float(final_vp_loss_denominator.item()),
            "q_loss_weighted_sum": float(q_loss_sum.item()),
            "q_loss_weight_sum": float(q_loss_denominator.item()),
            "value_categorical_loss_weighted_sum": float(
                value_categorical_loss_sum.item()
            ),
            "value_categorical_loss_weight_sum": float(
                value_categorical_loss_denominator.item()
            ),
            "value_categorical_clean_loss_weighted_sum": float(
                value_categorical_clean_loss_sum.item()
            ),
            "value_categorical_clean_loss_weight_sum": float(
                value_categorical_clean_loss_denominator.item()
            ),
            "value_categorical_truncated_loss_weighted_sum": float(
                value_categorical_truncated_loss_sum.item()
            ),
            "value_categorical_truncated_loss_weight_sum": float(
                value_categorical_truncated_loss_denominator.item()
            ),
            "q_score_rows_ge2": q_score_rows_ge2,
            **advantage_stats,
            "soft_distillation_rows": int(has_soft.sum().item()) if soft_targets is not None else 0,
            "soft_distillation_active_rows": (
                int((has_soft & active).sum().item()) if soft_targets is not None else 0
            ),
            "active_count": active_count,
            "accuracy": float(accuracy.item()),
            "top3_accuracy": float(top3_accuracy.item()),
            "phase_stats": _field_stats(
                data,
                batch[active_np],
                predictions_np[active_np],
                targets_np[active_np],
                logits_np[active_np],
                field="phase",
            ),
            "phase_stats_unforced": _field_stats_unforced(
                data,
                batch[active_np],
                predictions_np[active_np],
                targets_np[active_np],
                logits_np[active_np],
                field="phase",
            ),
            "teacher_stats": _field_stats(
                data,
                batch[active_np],
                predictions_np[active_np],
                targets_np[active_np],
                logits_np[active_np],
                field="teacher_name",
            ),
            "prior_kl_rows": prior_kl_rows,
            "prior_kl_model_prior_sum": kl_model_prior_sum,
            "prior_kl_prior_model_sum": kl_prior_model_sum,
            "prior_kl_target_prior_sum": kl_target_prior_sum,
            "active_policy_teacher_gap_rows": teacher_gap_rows,
            "active_policy_kl_target_model_sum": teacher_kl_target_model_sum,
            "active_policy_kl_target_prior_sum": teacher_kl_target_prior_sum,
        }


# Large per-decision columns streamed per batch by MemmapCorpus rather than held
# resident. Everything else (scalars, strings, VP arrays, and the comparatively
# small legal_action_ids/target_policy/target_scores/masks needed by the
# full-corpus weight/split/quality passes) is materialised eagerly at load time.
MEMMAP_LAZY_COLUMNS = frozenset(
    {
        "obs",
        "legal_action_ids",
        "legal_action_context",
        "legal_action_tokens",
        "legal_action_target_ids",
        "legal_action_mask",
        "hex_tokens",
        "hex_vertex_ids",
        "hex_edge_ids",
        "hex_mask",
        "vertex_tokens",
        "vertex_mask",
        "edge_tokens",
        "edge_vertex_ids",
        "edge_mask",
        "player_tokens",
        "player_mask",
        "global_tokens",
        "event_tokens",
        "event_target_ids",
        "event_mask",
        "prior_policy",
        "target_policy",
        "target_policy_mask",
        "target_scores",
        "target_scores_mask",
    }
)


def _normalize_index(idx, n: int) -> np.ndarray:
    """Coerce a __getitem__ key (int array, slice, or list) into an int64 index array."""
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n), dtype=np.int64)
    arr = np.asarray(idx)
    if arr.dtype == np.bool_:
        return np.flatnonzero(arr).astype(np.int64, copy=False)
    return arr.astype(np.int64, copy=False)


class _MemmapFixedColumn:
    """Fixed-width column backed by a flat memmap; materialises only indexed rows."""

    def __init__(self, mm: np.memmap, n: int):
        self._mm = mm
        self.shape = tuple(mm.shape)
        self.ndim = mm.ndim
        self.dtype = mm.dtype
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        return np.asarray(self._mm[idx])

    def __array__(self, dtype=None):
        arr = np.asarray(self._mm)
        return arr.astype(dtype) if dtype is not None else arr


class _MemmapCategoricalColumn:
    """Dictionary-encoded string column decoded only for requested rows.

    Production corpora repeat a handful of labels across millions of rows.
    Eager ``categories[codes]`` creates a large fixed-width Unicode array in
    every DDP rank.  Keeping the int32 codes mapped preserves identical NumPy
    indexing/array semantics without retaining the decoded corpus eight times.
    """

    def __init__(self, codes: np.memmap, categories: np.ndarray):
        self._codes = codes
        self.categories = categories
        self.shape = tuple(codes.shape)
        self.ndim = codes.ndim
        self.dtype = categories.dtype

    def __len__(self) -> int:
        return int(self._codes.shape[0])

    def __getitem__(self, idx):
        return self.categories[np.asarray(self._codes[idx])]

    def __array__(self, dtype=None):
        values = self.categories[np.asarray(self._codes)]
        return values.astype(dtype) if dtype is not None else values

    def grouped_weights(
        self, weights: np.ndarray, *, limit: int
    ) -> dict[str, dict[str, float | int]]:
        codes = np.asarray(self._codes, dtype=np.int64)
        counts = np.bincount(codes, minlength=len(self.categories))
        totals = np.bincount(
            codes, weights=np.asarray(weights, dtype=np.float64), minlength=len(self.categories)
        )
        order = np.argsort(-counts)
        result: dict[str, dict[str, float | int]] = {}
        for index in order[:limit]:
            raw = int(counts[index])
            if raw == 0:
                continue
            total = float(totals[index])
            result[str(self.categories[index])] = {
                "raw_samples": raw,
                "weight_sum": total,
                "mean_weight": total / raw,
            }
        return result

    def present_values(self) -> set[str]:
        counts = np.bincount(
            np.asarray(self._codes, dtype=np.int64), minlength=len(self.categories)
        )
        return {
            str(self.categories[index])
            for index in np.flatnonzero(counts)
        }

    def value_counts(self, index=None) -> dict[str, int]:
        codes = np.asarray(
            self._codes if index is None else self._codes[index], dtype=np.int64
        )
        counts = np.bincount(codes, minlength=len(self.categories))
        return {
            str(self.categories[position]): int(counts[position])
            for position in np.flatnonzero(counts)
        }


class _ImplicitConstantColumn:
    """File-free fixed-width column materialised only for requested rows."""

    def __init__(self, n: int, inner_shape: tuple[int, ...], dtype, fill):
        self._n = int(n)
        self._inner_shape = tuple(int(d) for d in inner_shape)
        self.dtype = np.dtype(dtype)
        self._fill = fill
        self.shape = (self._n, *self._inner_shape)
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        output_prefix = self._indexed_prefix_shape(idx)
        if not output_prefix:
            return np.full(self._inner_shape, self._fill, dtype=self.dtype)
        return np.full(
            (*output_prefix, *self._inner_shape), self._fill, dtype=self.dtype
        )

    def _indexed_prefix_shape(self, idx) -> tuple[int, ...]:
        """Validate a row index and return NumPy's output prefix shape.

        The column is constant, so reading the actual index values would be
        wasted work, but accepting invalid indices would hide sampler bugs.
        Validate bounds/boolean-mask length and preserve advanced-index shapes
        without allocating an ``arange(row_count)`` for a large corpus.
        """
        if isinstance(idx, slice):
            return (len(range(*idx.indices(self._n))),)

        array = np.asarray(idx)
        if array.ndim == 0:
            if array.dtype.kind == "b":
                # NumPy treats a scalar bool as an advanced index that inserts
                # a leading 0/1 dimension while retaining the whole row axis.
                return (int(bool(array)), self._n)
            try:
                value = operator.index(idx)
            except TypeError:
                try:
                    value = operator.index(array.item())
                except (TypeError, ValueError) as error:
                    raise IndexError(
                        "implicit column indices must be integers, slices, or "
                        "integer/boolean arrays"
                    ) from error
            if not -self._n <= value < self._n:
                raise IndexError(
                    f"index {value} is out of bounds for axis 0 with size {self._n}"
                )
            return ()

        if array.dtype.kind == "b":
            if array.ndim != 1 or array.shape[0] != self._n:
                raise IndexError(
                    "boolean index did not match implicit column row axis; "
                    f"axis has size {self._n} but mask shape is {array.shape}"
                )
            return (int(np.count_nonzero(array)),)

        # NumPy accepts a literal empty list as an integer advanced index even
        # though np.asarray([]) defaults to float64. Explicit floating arrays,
        # including empty ones, remain invalid.
        literal_empty_list = isinstance(idx, list) and array.size == 0
        if array.dtype.kind not in {"i", "u"} and not literal_empty_list:
            raise IndexError("arrays used as indices must be of integer or boolean type")
        if array.size and bool(np.any((array < -self._n) | (array >= self._n))):
            bad = int(array[(array < -self._n) | (array >= self._n)].flat[0])
            raise IndexError(
                f"index {bad} is out of bounds for axis 0 with size {self._n}"
            )
        return tuple(int(d) for d in array.shape)

    def __array__(self, dtype=None):
        arr = np.full(self.shape, self._fill, dtype=self.dtype)
        return arr.astype(dtype) if dtype is not None else arr


class _MemmapRaggedColumn:
    """Legal-action-ragged column stored trimmed on disk.

    Reconstructs an ``(len(batch), legal_width[, feat])`` array padded with the
    loader's fill value, byte-identical to load_teacher_data's padded column for
    the same rows.
    """

    def __init__(self, flat: np.memmap, offsets: np.ndarray, legal_width: int, fill, dtype, feat):
        self._flat = flat
        self._offsets = offsets
        self._width = int(legal_width)
        self._fill = fill
        self.dtype = np.dtype(dtype)
        self._feat = feat
        self._n = int(offsets.shape[0] - 1)
        self.ndim = 3 if feat is not None else 2
        self.shape = (self._n, self._width, feat) if feat is not None else (self._n, self._width)

    def __len__(self) -> int:
        return self._n

    def row_counts(self) -> np.ndarray:
        """Return legal-width prefix lengths without reconstructing padding."""
        return (self._offsets[1:] - self._offsets[:-1]).astype(
            np.int64, copy=False
        )

    def _reconstruct(self, indices: np.ndarray | None) -> np.ndarray:
        width = self._width
        if indices is None:
            # Whole corpus: the flat file is already the row-major prefix concat,
            # so scatter it straight into the padded output without gathering.
            counts = (self._offsets[1:] - self._offsets[:-1]).astype(np.int64)
            m = self._n
            prefix = np.arange(width)[None, :] < counts[:, None]
            out = self._new_full(m)
            out[prefix] = np.asarray(self._flat)
            return out
        starts = self._offsets[indices]
        counts = (self._offsets[indices + 1] - starts).astype(np.int64)
        m = int(indices.shape[0])
        out = self._new_full(m)
        total = int(counts.sum())
        if total:
            prefix = np.arange(width)[None, :] < counts[:, None]
            within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
            src = np.repeat(starts, counts) + within
            out[prefix] = np.asarray(self._flat[src])
        return out

    def _new_full(self, m: int) -> np.ndarray:
        if self._feat is not None:
            return np.full((m, self._width, self._feat), self._fill, dtype=self.dtype)
        return np.full((m, self._width), self._fill, dtype=self.dtype)

    def __getitem__(self, idx):
        return self._reconstruct(_normalize_index(idx, self._n))

    def __array__(self, dtype=None):
        arr = self._reconstruct(None)
        return arr.astype(dtype) if dtype is not None else arr


class MemmapCorpus:
    """Dict-of-arrays view over a corpus built by tools/build_memmap_corpus.py.

    Exposes the same ``data[key][batch]`` interface load_teacher_data returns.
    Small/full-corpus columns are materialised eagerly; large per-decision
    columns are lazy (streamed per batch), so host RAM stays bounded regardless
    of corpus size.
    """

    def __init__(self, corpus_dir: Path):
        corpus_dir = Path(corpus_dir)
        meta_path = corpus_dir / "corpus_meta.json"
        if not meta_path.exists():
            raise SystemExit(
                f"{corpus_dir} is not a memmap corpus (no corpus_meta.json). "
                "Build it with tools/build_memmap_corpus.py or use --data-format npz."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("schema") not in {"memmap_corpus_v1", "memmap_corpus_v2"}:
            raise SystemExit(f"{meta_path}: unsupported corpus schema {self.meta.get('schema')!r}")
        self.row_count = int(self.meta["row_count"])
        self.legal_width = int(self.meta["legal_width"])
        self.stats = self.meta.get("stats", {})
        self._columns = self.meta["columns"]
        implicit_columns = {
            name for name, schema in self._columns.items()
            if schema.get("kind") == "implicit_constant"
        }
        declared_implicit_raw = self.meta.get("implicit_zero_columns", ())
        try:
            declared_implicit_columns = set(declared_implicit_raw)
        except TypeError as error:
            raise SystemExit(
                f"{meta_path}: implicit_zero_columns must be a sequence of names"
            ) from error
        if implicit_columns != declared_implicit_columns:
            raise SystemExit(
                f"{meta_path}: implicit column metadata mismatch: "
                f"columns={sorted(implicit_columns)} "
                f"declared={sorted(declared_implicit_columns)}"
            )
        unsupported_implicit = implicit_columns - {"event_tokens", "event_mask"}
        if unsupported_implicit:
            raise SystemExit(
                f"{meta_path}: unsupported implicit columns "
                f"{sorted(unsupported_implicit)}"
            )
        if self.meta.get("schema") == "memmap_corpus_v2":
            required_implicit = {"event_tokens", "event_mask"}
            if (
                implicit_columns != required_implicit
                or len(declared_implicit_raw) != len(required_implicit)
            ):
                raise SystemExit(
                    f"{meta_path}: memmap_corpus_v2 requires exactly implicit-zero "
                    f"columns {sorted(required_implicit)}; got "
                    f"{sorted(implicit_columns)}"
                )
            nonzero_fill = [
                name
                for name in sorted(required_implicit)
                if self._columns[name].get("fill") != 0
            ]
            if nonzero_fill:
                raise SystemExit(
                    f"{meta_path}: implicit-zero columns must declare fill=0; "
                    f"nonzero/missing fill for {nonzero_fill}"
                )
        self._offsets = np.fromfile(corpus_dir / "row_offsets.dat", dtype=np.int64)
        if self._offsets.shape[0] != self.row_count + 1:
            raise SystemExit(
                f"{corpus_dir}: row_offsets length {self._offsets.shape[0]} != row_count+1 "
                f"{self.row_count + 1}"
            )
        self._eager: dict[str, np.ndarray] = {}
        self._lazy: dict[str, object] = {}
        for name, schema in self._columns.items():
            kind = schema["kind"]
            if kind == "string":
                codes = np.memmap(
                    corpus_dir / f"{name}.codes.dat",
                    dtype=np.int32,
                    mode="r",
                    shape=(self.row_count,),
                )
                categories = np.asarray(schema["categories"], dtype=str)
                if categories.size == 0:
                    categories = np.asarray([""], dtype=str)
                self._lazy[name] = _MemmapCategoricalColumn(codes, categories)
                continue
            if kind == "fixed":
                inner = tuple(int(d) for d in schema["inner_shape"])
                mm = np.memmap(
                    corpus_dir / f"{name}.dat",
                    dtype=np.dtype(schema["dtype"]),
                    mode="r",
                    shape=(self.row_count, *inner),
                )
                if name in MEMMAP_LAZY_COLUMNS:
                    self._lazy[name] = _MemmapFixedColumn(mm, self.row_count)
                else:
                    self._eager[name] = np.asarray(mm)
                continue
            if kind == "implicit_constant":
                if self.meta.get("schema") != "memmap_corpus_v2":
                    raise SystemExit(
                        f"{meta_path}: implicit_constant column {name!r} requires "
                        "memmap_corpus_v2"
                    )
                inner = tuple(int(d) for d in schema["inner_shape"])
                self._lazy[name] = _ImplicitConstantColumn(
                    self.row_count, inner, schema["dtype"], schema["fill"]
                )
                continue
            if kind not in {"ragged2d", "ragged3d"}:
                raise SystemExit(f"{meta_path}: unsupported storage kind {kind!r} for {name!r}")
            # ragged2d / ragged3d
            feat = int(schema["feat"]) if kind == "ragged3d" else None
            flat_shape = (int(self.meta["flat_count"]), feat) if feat is not None else (int(self.meta["flat_count"]),)
            flat = np.memmap(
                corpus_dir / f"{name}.dat",
                dtype=np.dtype(schema["dtype"]),
                mode="r",
                shape=flat_shape,
            )
            column = _MemmapRaggedColumn(
                flat, self._offsets, self.legal_width, schema["fill"], schema["dtype"], feat
            )
            if name in MEMMAP_LAZY_COLUMNS:
                self._lazy[name] = column
            else:
                self._eager[name] = column._reconstruct(None)

    def __contains__(self, key: str) -> bool:
        return key in self._eager or key in self._lazy

    def __getitem__(self, key: str):
        if key in self._eager:
            return self._eager[key]
        if key in self._lazy:
            return self._lazy[key]
        raise KeyError(key)

    def get(self, key: str, default=None):
        if key in self:
            return self[key]
        return default

    def keys(self):
        return list(self._eager.keys()) + list(self._lazy.keys())

    def __len__(self) -> int:
        return self.row_count


def load_teacher_data_memmap(
    path: Path, *, composite_meta: dict[str, object] | None = None
) -> MemmapCorpus | ConcatMemmapCorpus:
    if path.is_file():
        authenticated = (
            composite_meta
            if composite_meta is not None
            else _preflight_memmap_composite_descriptor(path)
        )
        if authenticated.get("schema_version") not in {
            "memmap_composite_v1", "memmap_composite_v2"
        }:
            raise SystemExit("memmap composite loader received invalid authenticated metadata")
        components = authenticated["components"]
        assert isinstance(components, list)
        dirs = [Path(str(component["corpus_dir"])) for component in components]
        corpus = ConcatMemmapCorpus(
            [MemmapCorpus(component_dir) for component_dir in dirs], dirs=dirs
        )
        if authenticated["schema_version"] == "memmap_composite_v2":
            corpus.component_ids = tuple(authenticated["component_ids"])
            corpus.component_game_sampling_ratios = tuple(
                float(value)
                for value in authenticated["component_game_sampling_ratios"]
            )
            anchor_ids = set(authenticated["policy_kl_anchor_component_ids"])
            corpus.policy_kl_anchor_component_indices = tuple(
                index
                for index, component_id in enumerate(corpus.component_ids)
                if component_id in anchor_ids
            )
            corpus.policy_kl_anchor_scope_authenticated = True
            distillation_ids = set(
                authenticated["policy_distillation_component_ids"]
            )
            corpus.policy_distillation_component_indices = tuple(
                index
                for index, component_id in enumerate(corpus.component_ids)
                if component_id in distillation_ids
            )
            corpus.policy_distillation_scope_authenticated = True
        corpus.meta.update({
            "schema": authenticated["schema_version"],
            "descriptor_path": authenticated["descriptor_path"],
            "descriptor_fingerprint": authenticated["descriptor_fingerprint"],
            "payload_inventory_sha256": authenticated["payload_inventory_sha256"],
            "diagnostic_only": True,
            "promotion_eligible": False,
        })
    else:
        corpus = MemmapCorpus(path)
    load_event = {
        "progress": "bc_memmap_load",
        "corpus_dir": str(path),
        "rows": corpus.row_count,
        "legal_width": corpus.legal_width,
        "shard_count": int(corpus.meta.get("shard_count", 0)),
    }
    if isinstance(corpus, ConcatMemmapCorpus):
        load_event.update({
            "progress": "bc_memmap_composite_load",
            "component_count": len(corpus.corpora),
        })
    print(json.dumps(load_event, sort_keys=True), flush=True)
    return corpus


def _iterate_training_batches(
    data,
    order: np.ndarray,
    train_indices: np.ndarray,
    batch_size: int,
    policy_sample_weights: np.ndarray,
    value_sample_weights: np.ndarray,
    *,
    num_workers: int,
    prefetch: int,
):
    """Yield ``(data, batch, policy_weights, value_weights)`` tuples for one epoch.

    Default path (npz dict, or num_workers<=0): yields the corpus and the GLOBAL
    batch indices unchanged, so train_fn indexes exactly as before -- zero
    behaviour change and identical batch order.

    Prefetch path (MemmapCorpus + num_workers>0): background threads materialise
    each batch's columns into a plain dict while the GPU trains the previous
    batch, overlapping the per-batch ragged reconstruction. train_fn then sees a
    materialised dict indexed by a local ``arange``, which is element-for-element
    identical to indexing the corpus with the global batch. Threads (not
    processes) share the read-only memmaps safely and avoid the fork-duplication
    pitfall of DataLoader workers.
    """
    batches = [
        train_indices[order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]
    batches = [b for b in batches if len(b) > 0]

    if num_workers <= 0 or not isinstance(data, (MemmapCorpus, ConcatMemmapCorpus)):
        for batch in batches:
            yield data, batch, policy_sample_weights, value_sample_weights
        return

    keys = list(data.keys())

    def _materialize(batch: np.ndarray):
        materialized = {key: data[key][batch] for key in keys}
        local = np.arange(len(batch), dtype=np.int64)
        return materialized, local, policy_sample_weights[batch], value_sample_weights[batch]

    prefetch = max(1, int(prefetch))
    with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
        pending: deque = deque()
        next_index = 0
        while next_index < len(batches) and len(pending) < prefetch:
            pending.append(executor.submit(_materialize, batches[next_index]))
            next_index += 1
        while pending:
            future = pending.popleft()
            if next_index < len(batches):
                pending.append(executor.submit(_materialize, batches[next_index]))
                next_index += 1
            yield future.result()


def load_teacher_data(
    path: Path,
    *,
    ddp: dict[str, int | bool] | None = None,
    shard_data: bool = False,
    mask_hidden_info: bool = False,
) -> dict:
    files = _teacher_shard_files(path)
    if not files:
        raise SystemExit(f"no teacher shards found in {path}")
    total_files = len(files)
    if shard_data:
        if ddp is None or not bool(ddp.get("enabled", False)):
            raise SystemExit("--ddp-shard-data requires torchrun/DDP")
        world_size = int(ddp["world_size"])
        rank = int(ddp["rank"])
        files = files[rank::world_size]
        if not files:
            raise SystemExit(
                f"rank {rank} received no teacher shards from {path}; "
                f"total_shards={total_files}, world_size={world_size}"
            )
        print(
            json.dumps(
                {
                    "progress": "bc_data_shard_load",
                    "rank": rank,
                    "world_size": world_size,
                    "loaded_shards": len(files),
                    "total_shards": total_files,
                    "first_shard": str(files[0]),
                    "last_shard": str(files[-1]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    arrays: dict[str, list[np.ndarray]] = {}
    aux_targets_enabled = False
    prior_row_counts: list[int] = []
    keys = (
        "obs",
        "legal_action_ids",
        "legal_action_context",
        "action_taken",
        "target_policy",
        "prior_policy",
        "target_scores",
        "target_policy_mask",
        "target_scores_mask",
        "target_score_source",
        "target_information_regime",
        "root_value",
        "root_value_mask",
        "afterstate_target",
        "afterstate_target_mask",
        "simulations_used",
        "is_forced",
        "used_full_search",
        "game_seed",
        "teacher_name",
        "player",
        "seat",
        "phase",
        "decision_index",
        "action_mask_version",
        "winner",
        "terminated",
        "truncated",
        "final_public_vps",
        "has_final_public_vps",
        "final_actual_vps",
        "has_final_actual_vps",
        "policy_weight_multiplier",
        "value_weight_multiplier",
        *AUX_TARGET_KEYS,
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
    if mask_hidden_info:
        from catan_zero.rl.entity_token_features import mask_player_tokens_public
    for file in files:
        raw = _load_npz(file)
        raw_has_aux_targets = any(key in raw for key in AUX_TARGET_KEYS)
        if raw_has_aux_targets and not aux_targets_enabled:
            # A replay window may intentionally mix legacy shards with newly
            # labeled CAT-100 shards. Backfill already-seen legacy rows with
            # ignore values when the first labeled shard appears so every
            # column remains aligned to action_taken.
            aux_targets_enabled = True
            for key in AUX_TARGET_KEYS:
                arrays[key] = [
                    _aux_subgoal_default_array(key, rows) for rows in prior_row_counts
                ]
        shard = _normalize_teacher_shard(
            raw,
            file,
            include_aux_defaults=aux_targets_enabled,
        )
        prior_row_counts.append(int(len(shard["action_taken"])))
        if mask_hidden_info and "player_tokens" in shard:
            # f72 public-observation training: strip opponent hidden slots from the
            # banked (omniscient) tokens so this corpus trains a public-only model.
            shard["player_tokens"] = mask_player_tokens_public(shard["player_tokens"])
        for key in keys:
            if key in shard:
                arrays.setdefault(key, []).append(shard[key])
    return {key: _concat_padded(key, values) for key, values in arrays.items()}


def _validate_target_information_admission(
    data,
    *,
    mask_hidden_info: bool,
    soft_target_weight: float,
    policy_loss_weight: float,
    q_loss_weight: float,
    value_target_lambda: float,
    policy_kl_anchor_weight: float,
    policy_surprise_weight: float,
) -> dict[str, object]:
    """Fail closed before public-information training consumes search targets.

    Observation masking and search-state safety are independent contracts.  A
    masked student may use realised outcomes and hard recorded actions from a
    legacy corpus only when every search-derived objective is disabled for the
    run.  The current trainer configures those objectives corpus-wide, so a
    mixed corpus is admitted to them only when every row explicitly carries
    the public-conservation PIMC regime. Missing provenance is ``unknown`` and
    is intentionally unsafe.
    """

    n = int(len(data["action_taken"]))
    regimes = np.asarray(
        data.get(
            "target_information_regime",
            np.full(n, TARGET_INFORMATION_REGIME_UNKNOWN),
        )
    ).astype(str)
    if regimes.shape != (n,):
        raise SystemExit(
            "target_information_regime must be a one-dimensional per-row column: "
            f"expected {(n,)}, got {regimes.shape}"
        )
    counts = {
        str(value): int(count)
        for value, count in zip(*np.unique(regimes, return_counts=True))
    }
    unsafe_count = int(np.sum(regimes != TARGET_INFORMATION_REGIME_PUBLIC))
    objectives: list[str] = []
    if (
        float(policy_loss_weight) != 0.0
        and float(soft_target_weight) > 0.0
        and ("target_policy" in data or "target_scores" in data)
    ):
        objectives.append("soft_policy")
    if float(q_loss_weight) != 0.0 and "target_scores" in data:
        objectives.append("q_target")
    if float(value_target_lambda) != 1.0 and "root_value" in data:
        objectives.append("root_value")
    if float(policy_kl_anchor_weight) > 0.0 and "prior_policy" in data:
        objectives.append("policy_kl_anchor")
    if float(policy_surprise_weight) > 0.0 and (
        "target_policy" in data or "prior_policy" in data
    ):
        objectives.append("policy_surprise_sampling")

    report: dict[str, object] = {
        "mask_hidden_info": bool(mask_hidden_info),
        "target_information_regime_counts": counts,
        "unsafe_or_unknown_rows": unsafe_count,
        "search_target_objectives": objectives,
    }
    if bool(mask_hidden_info) and objectives and unsafe_count:
        raise SystemExit(
            "public-observation training refused unsafe/unknown search targets: "
            f"objectives={objectives}, unsafe_or_unknown_rows={unsafe_count}/{n}, "
            f"target_information_regimes={counts}. Only "
            f"{TARGET_INFORMATION_REGIME_PUBLIC!r} may supply soft policy, Q, or "
            "search-root value targets to --mask-hidden-info training. Re-generate "
            "with public-conservation PIMC search, or disable every listed search-target "
            "objective and train only on hard actions/realised outcomes."
        )
    return report


def _env_config_for_teacher_data(args, data: dict, ddp: dict[str, int | bool]):
    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv

    observed_width = int(data["obs"].shape[1])
    base_config = parse_track(
        args.track,
        vps_to_win=args.vps_to_win,
        use_graph_history_features=False,
    )
    graph_config = parse_track(
        args.track,
        vps_to_win=args.vps_to_win,
        use_graph_history_features=True,
    )

    def _obs_width(config) -> int:
        env = ColonistMultiAgentEnv(config)
        try:
            return int(env.observation_space.shape[0])
        finally:
            env.close()

    base_width = _obs_width(base_config)
    graph_width = _obs_width(graph_config)
    if bool(args.graph_history_features):
        selected = graph_config
        reason = "explicit"
    elif observed_width == graph_width and observed_width != base_width:
        selected = graph_config
        reason = "auto_detected_from_obs_width"
    elif observed_width == base_width:
        selected = base_config
        reason = "base_obs_width"
    else:
        raise SystemExit(
            "teacher observation width does not match a known train schema: "
            f"observed={observed_width}, base={base_width}, graph_history={graph_width}. "
            "Regenerate/curate the shards with the current env or pass the correct "
            "--graph-history-features setting if the schema changed intentionally."
        )
    _rank0_print(
        json.dumps(
            {
                "progress": "bc_env_schema",
                "track": args.track,
                "vps_to_win": int(args.vps_to_win),
                "observed_obs_width": observed_width,
                "base_obs_width": base_width,
                "graph_history_obs_width": graph_width,
                "graph_history_features": bool(
                    getattr(selected, "use_graph_history_features", False)
                ),
                "reason": reason,
            },
            sort_keys=True,
        ),
        ddp,
    )
    return selected


def _teacher_shard_files(path: Path) -> list[Path]:
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        files = _manifest_shard_files(manifest_path)
        if files:
            return files

    if (path / "manifest.partial.json").exists() or (path / "parts").exists():
        raise SystemExit(
            f"{path} looks like a partial Modal/raw teacher root without a completed "
            "top-level manifest.json; refusing recursive shard glob. Curate or "
            "summarize the run first, or pass a completed leaf teacher_data directory."
        )

    files = sorted(path.glob("*.npz")) + sorted(path.glob("*.npz.zst"))
    if files:
        return files

    child_manifests = sorted(
        candidate
        for candidate in path.glob("**/manifest.json")
        if candidate.parent != path
    )
    if len(child_manifests) == 1:
        files = _manifest_shard_files(child_manifests[0])
        if files:
            return files
    if len(child_manifests) > 1:
        files = _modal_part_manifest_shards(path, child_manifests)
        if files:
            return files
        files = _entity_partition_manifest_shards(child_manifests)
        if files:
            return files
        previews = ", ".join(str(candidate.parent) for candidate in child_manifests[:5])
        raise SystemExit(
            f"{path} contains multiple nested teacher manifests; pass a single leaf "
            f"curated data directory instead of recursively mixing runs. Examples: {previews}"
        )

    return sorted(path.glob("**/*.npz")) + sorted(path.glob("**/*.npz.zst"))


def _modal_part_manifest_shards(path: Path, manifests: list[Path]) -> list[Path]:
    files: list[Path] = []
    for manifest in manifests:
        try:
            relative = manifest.relative_to(path)
        except ValueError:
            return []
        parts = relative.parts
        if len(parts) != 3 or parts[0] != "parts" or not parts[1].startswith("part_"):
            return []
        files.extend(_manifest_shard_files(manifest))
    return sorted(files)


def _entity_partition_manifest_shards(manifests: list[Path]) -> list[Path]:
    files: list[Path] = []
    signatures: set[tuple[str, int, bool]] = set()
    partition_pairs: set[tuple[int, int]] = set()
    partition_counts: set[int] = set()
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if manifest.get("schema") != "entity_tokens_v1":
            return []
        signatures.add(
            (
                str(manifest.get("track", "")),
                int(manifest.get("vps_to_win", -1)),
                bool(manifest.get("graph_history_features", False)),
            )
        )
        partition_count = int(manifest.get("partition_count", 1))
        partition_index = int(manifest.get("partition_index", 0))
        partition_counts.add(partition_count)
        partition_pairs.add((partition_count, partition_index))
        if manifest.get("mismatches"):
            raise SystemExit(f"{manifest_path} contains entity conversion mismatches")
        files.extend(_manifest_shard_files(manifest_path))
    if len(signatures) != 1:
        return []
    if len(partition_pairs) != len(manifests):
        raise SystemExit("duplicate entity conversion partition manifests detected")
    if len(partition_counts) != 1:
        raise SystemExit("inconsistent entity conversion partition counts detected")
    partition_count = next(iter(partition_counts))
    expected = {(partition_count, index) for index in range(partition_count)}
    missing = sorted(index for _, index in (expected - partition_pairs))
    extra = sorted(index for _, index in (partition_pairs - expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing partition indices {missing[:10]}")
        if extra:
            details.append(f"unexpected partition indices {extra[:10]}")
        raise SystemExit("incomplete entity conversion partition manifests: " + "; ".join(details))
    return sorted(files)


def _manifest_shard_files(manifest_path: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    missing = []
    for value in manifest.get("shards", ()):
        raw = Path(value)
        candidates = [raw] if raw.is_absolute() else [raw, manifest_path.parent / raw]
        if raw.is_absolute():
            candidates.append(manifest_path.parent / raw.name)
        chosen = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        files.append(chosen)
        if not chosen.exists():
            missing.append(str(chosen))
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            f"{manifest_path} points to missing teacher shards. "
            f"First missing paths: {preview}"
        )
    return files


def _normalize_teacher_shard(
    shard,
    path: Path,
    *,
    include_aux_defaults: bool = False,
) -> dict[str, np.ndarray]:
    required = ("obs", "legal_action_ids", "legal_action_context", "action_taken")
    missing = [key for key in required if key not in shard]
    if missing:
        raise SystemExit(f"{path} is missing required teacher fields: {missing}")

    action_taken = np.asarray(shard["action_taken"], dtype=np.int16)
    n = int(len(action_taken))
    legal = np.asarray(shard["legal_action_ids"], dtype=np.int16)
    context = np.asarray(shard["legal_action_context"], dtype=np.float16)
    obs = np.asarray(shard["obs"], dtype=np.float16)
    if obs.shape[0] != n or legal.shape[0] != n or context.shape[0] != n:
        raise SystemExit(
            f"{path} has inconsistent row counts: "
            f"obs={obs.shape[0]} legal={legal.shape[0]} "
            f"context={context.shape[0]} action_taken={n}"
        )
    if legal.ndim != 2 or context.ndim != 3 or context.shape[1] != legal.shape[1]:
        raise SystemExit(
            f"{path} has invalid legal/context shapes: "
            f"legal={legal.shape} context={context.shape}"
        )

    legal_width = int(legal.shape[1])
    shard_has_final_public_vps = False
    shard_has_final_actual_vps = False
    target_policy = _field_or_default(
        shard,
        "target_policy",
        np.zeros((n, legal_width), dtype=np.float32),
        path,
        leading=n,
        width=legal_width,
    ).astype(np.float32, copy=False)  # F4: fp32, not fp16 (see gumbel_self_play.py._build_decision_row)
    # Success telemetry: root prior (pre-search network policy), same legal_rust
    # ordering as target_policy -- only populated by Gumbel self-play rows
    # (gumbel_self_play.py._build_decision_row), defaults to all-zero for any
    # other teacher source. That default is exactly what scopes
    # _prior_kl_telemetry to a held-out GEN slice with no extra bookkeeping.
    prior_policy = _field_or_default(
        shard,
        "prior_policy",
        np.zeros((n, legal_width), dtype=np.float32),
        path,
        leading=n,
        width=legal_width,
    ).astype(np.float32, copy=False)
    target_scores = _field_or_default(
        shard,
        "target_scores",
        np.full((n, legal_width), np.nan, dtype=np.float32),
        path,
        leading=n,
        width=legal_width,
    ).astype(np.float32, copy=False)
    target_policy_mask = _field_or_default(
        shard,
        "target_policy_mask",
        (np.asarray(target_policy, dtype=np.float32) > 0.0),
        path,
        leading=n,
        width=legal_width,
    ).astype(np.bool_, copy=False)
    target_scores_mask = _field_or_default(
        shard,
        "target_scores_mask",
        np.isfinite(target_scores),
        path,
        leading=n,
        width=legal_width,
    ).astype(np.bool_, copy=False)

    result: dict[str, np.ndarray] = {
        "obs": obs,
        "legal_action_ids": legal,
        "legal_action_context": context,
        "action_taken": action_taken,
        "target_policy": target_policy,
        "prior_policy": prior_policy,
        "target_scores": target_scores,
        "target_policy_mask": target_policy_mask,
        "target_scores_mask": target_scores_mask,
        "target_score_source": _field_or_default(
            shard,
            "target_score_source",
            np.full(n, "", dtype="<U1"),
            path,
            leading=n,
        ).astype(str),
        "target_information_regime": _field_or_default(
            shard,
            "target_information_regime",
            np.full(n, TARGET_INFORMATION_REGIME_UNKNOWN),
            path,
            leading=n,
        ).astype(str),
        "game_seed": _field_or_default(
            shard,
            "game_seed",
            np.zeros(n, dtype=np.int64),
            path,
            leading=n,
        ).astype(np.int64, copy=False),
        "teacher_name": _field_or_default(
            shard,
            "teacher_name",
            np.full(n, "", dtype="<U1"),
            path,
            leading=n,
        ).astype(str),
        "player": _field_or_default(
            shard,
            "player",
            np.full(n, "", dtype="<U1"),
            path,
            leading=n,
        ).astype(str),
        "seat": _field_or_default(
            shard,
            "seat",
            np.full(n, -1, dtype=np.int8),
            path,
            leading=n,
        ).astype(np.int8, copy=False),
        "phase": _field_or_default(
            shard,
            "phase",
            np.full(n, "", dtype="<U1"),
            path,
            leading=n,
        ).astype(str),
        "decision_index": _field_or_default(
            shard,
            "decision_index",
            np.full(n, -1, dtype=np.int32),
            path,
            leading=n,
        ).astype(np.int32, copy=False),
        "action_mask_version": _field_or_default(
            shard,
            "action_mask_version",
            np.full(n, "", dtype="<U1"),
            path,
            leading=n,
        ).astype(str),
        "winner": _field_or_default(
            shard,
            "winner",
            np.full(n, "", dtype="<U1"),
            path,
            leading=n,
        ).astype(str),
        "terminated": _field_or_default(
            shard,
            "terminated",
            np.full(n, True, dtype=np.bool_),
            path,
            leading=n,
        ).astype(np.bool_, copy=False),
        "truncated": _field_or_default(
            shard,
            "truncated",
            np.full(n, False, dtype=np.bool_),
            path,
            leading=n,
        ).astype(np.bool_, copy=False),
        "final_public_vps": _field_or_default(
            shard,
            "final_public_vps",
            np.zeros((n, 4), dtype=np.int16),
            path,
            leading=n,
        ).astype(np.int16, copy=False),
        "has_final_public_vps": _field_or_default(
            shard,
            "has_final_public_vps",
            np.full(n, shard_has_final_public_vps, dtype=np.bool_),
            path,
            leading=n,
        ).astype(np.bool_, copy=False),
        "final_actual_vps": _field_or_default(
            shard,
            "final_actual_vps",
            np.zeros((n, 4), dtype=np.int16),
            path,
            leading=n,
        ).astype(np.int16, copy=False),
        "has_final_actual_vps": _field_or_default(
            shard,
            "has_final_actual_vps",
            np.full(n, shard_has_final_actual_vps, dtype=np.bool_),
            path,
            leading=n,
        ).astype(np.bool_, copy=False),
        "policy_weight_multiplier": _field_or_default(
            shard,
            "policy_weight_multiplier",
            np.ones(n, dtype=np.float32),
            path,
            leading=n,
        ).astype(np.float32, copy=False),
        "value_weight_multiplier": _field_or_default(
            shard,
            "value_weight_multiplier",
            np.ones(n, dtype=np.float32),
            path,
            leading=n,
        ).astype(np.float32, copy=False),
    }
    if "root_value" in shard:
        result["root_value"] = _field_or_default(
            shard,
            "root_value",
            np.full(n, np.nan, dtype=np.float32),
            path,
            leading=n,
        ).astype(np.float32, copy=False)
        result["root_value_mask"] = _field_or_default(
            shard,
            "root_value_mask",
            np.isfinite(result["root_value"]),
            path,
            leading=n,
        ).astype(np.bool_, copy=False)
    if "afterstate_target" in shard:
        result["afterstate_target"] = _field_or_default(
            shard,
            "afterstate_target",
            np.full((n, legal_width), np.nan, dtype=np.float32),
            path,
            leading=n,
            width=legal_width,
        ).astype(np.float32, copy=False)
        result["afterstate_target_mask"] = _field_or_default(
            shard,
            "afterstate_target_mask",
            np.isfinite(result["afterstate_target"]),
            path,
            leading=n,
            width=legal_width,
        ).astype(np.bool_, copy=False)
    if "simulations_used" in shard:
        result["simulations_used"] = _field_or_default(
            shard,
            "simulations_used",
            np.zeros(n, dtype=np.int32),
            path,
            leading=n,
        ).astype(np.int32, copy=False)
    # Preserve optional self-play search provenance when present.  Besides
    # supporting shard audits, these columns make the root-value admission
    # mask independently checkable after normalization.
    for key in ("is_forced", "used_full_search"):
        if key in shard:
            result[key] = _field_or_default(
                shard,
                key,
                np.zeros(n, dtype=np.bool_),
                path,
                leading=n,
            ).astype(np.bool_, copy=False)
    # CAT-100 columns are optional for legacy/non-self-play corpora, but when
    # present they must survive normalization into the training dict. Numeric
    # heads use NaN as their per-row ignore mask; categorical heads use -1.
    for key, dtype in AUX_SUBGOAL_FIELD_DTYPES.items():
        if key not in shard:
            if not include_aux_defaults:
                continue
            value = _aux_subgoal_default_array(key, n)
        else:
            value = np.asarray(shard[key])
        if value.shape != (n,):
            raise SystemExit(
                f"{path} field {key} has shape {value.shape}, expected ({n},)"
            )
        result[key] = value.astype(dtype, copy=False)
    for key, dtype in ENTITY_FIELD_DTYPES.items():
        if key not in shard:
            continue
        value = np.asarray(shard[key])
        if value.shape[0] != n:
            raise SystemExit(
                f"{path} field {key} has {value.shape[0]} rows, expected {n}"
            )
        result[key] = value.astype(dtype, copy=False)
    return result


def _aux_subgoal_default_array(key: str, rows: int) -> np.ndarray:
    """Per-head ignore fill used when mixing legacy and CAT-100 shards."""

    dtype = AUX_SUBGOAL_FIELD_DTYPES[key]
    fill = -1 if key in {"aux_next_settlement", "aux_robber_target"} else np.nan
    return np.full(int(rows), fill, dtype=dtype)


def _field_or_default(
    shard,
    key: str,
    default: np.ndarray,
    path: Path,
    *,
    leading: int,
    width: int | None = None,
) -> np.ndarray:
    if key not in shard:
        return default
    value = np.asarray(shard[key])
    if value.shape[0] != leading:
        raise SystemExit(
            f"{path} field {key} has {value.shape[0]} rows, expected {leading}"
        )
    if width is not None and (value.ndim < 2 or value.shape[1] != width):
        raise SystemExit(
            f"{path} field {key} has shape {value.shape}, expected width {width}"
        )
    return value


def teacher_data_quality(
    data: dict,
    *,
    q_skip_teacher_prefixes: tuple[str, ...] = ("catanatron_ab",),
    soft_target_temperature: float = 0.7,
    soft_target_source: str = "policy",
    soft_target_min_legal_coverage: float = 0.5,
) -> dict:
    n = int(len(data["action_taken"]))
    legal_counts = np.sum(data["legal_action_ids"] >= 0, axis=1)
    target_policy = np.asarray(data.get("target_policy", np.zeros((n, 0))), dtype=np.float32)
    target_scores = np.asarray(data.get("target_scores", np.full((n, 0), np.nan)), dtype=np.float32)
    legal_mask = data["legal_action_ids"] >= 0
    target_policy_mask = np.asarray(
        data.get("target_policy_mask", np.where(np.isfinite(target_policy), target_policy, 0.0) > 0.0),
        dtype=np.bool_,
    )
    target_scores_mask = np.asarray(
        data.get("target_scores_mask", np.isfinite(target_scores)),
        dtype=np.bool_,
    )
    finite_scores = legal_mask & target_scores_mask & np.isfinite(target_scores)
    positive_policy = legal_mask & target_policy_mask & (
        np.where(np.isfinite(target_policy), target_policy, 0.0) > 0.0
    )
    policy_rows = np.sum(positive_policy.any(axis=1))
    score_rows = np.sum(finite_scores.any(axis=1))
    legal_denominator = np.maximum(legal_counts, 1)
    score_coverage = np.sum(finite_scores, axis=1) / legal_denominator
    policy_coverage = np.sum(positive_policy, axis=1) / legal_denominator
    q_score_rows_ge2 = _q_score_rows_ge2(data, np.arange(n, dtype=np.int64))
    usable_q_score_rows_ge2 = _q_score_rows_ge2(
        data,
        np.arange(n, dtype=np.int64),
        q_skip_teacher_prefixes=q_skip_teacher_prefixes,
    )
    winners = np.asarray(data.get("winner", np.full(n, ""))).astype(str)
    truncated = np.asarray(data.get("truncated", np.zeros(n, dtype=np.bool_)), dtype=np.bool_)
    has_final_vps = np.asarray(
        data.get("has_final_public_vps", np.zeros(n, dtype=np.bool_)),
        dtype=np.bool_,
    )
    has_actual_vps = np.asarray(
        data.get("has_final_actual_vps", np.zeros(n, dtype=np.bool_)),
        dtype=np.bool_,
    )
    teachers = np.asarray(data.get("teacher_name", np.full(n, ""))).astype(str)
    phases = np.asarray(data.get("phase", np.full(n, ""))).astype(str)
    score_sources = np.asarray(data.get("target_score_source", np.full(n, ""))).astype(str)
    action_mask_versions = np.asarray(data.get("action_mask_version", np.full(n, ""))).astype(str)
    policy_multiplier = np.asarray(
        data.get("policy_weight_multiplier", np.ones(n, dtype=np.float32)),
        dtype=np.float32,
    )
    value_multiplier = np.asarray(
        data.get("value_weight_multiplier", np.ones(n, dtype=np.float32)),
        dtype=np.float32,
    )
    policy_active = policy_multiplier > 0.0
    value_active = value_multiplier > 0.0
    soft_payload = _soft_target_array(
        data,
        np.arange(n, dtype=np.int64),
        soft_target_temperature,
        soft_target_source,
    )
    if soft_payload is None:
        effective_soft_distillation = np.zeros(n, dtype=bool)
    else:
        effective_soft_distillation = _has_distillation_distribution(
            soft_payload[0],
            soft_payload[1],
            legal_action_ids=data["legal_action_ids"],
            min_legal_coverage=soft_target_min_legal_coverage,
        )
    unflagged_actual_vp_rows = _unflagged_vp_rows(data, "final_actual_vps", "has_final_actual_vps")
    unflagged_public_vp_rows = _unflagged_vp_rows(data, "final_public_vps", "has_final_public_vps")
    matches = data["legal_action_ids"] == data["action_taken"][:, None]
    selected_action_has_score = np.any(matches & finite_scores, axis=1)
    soft_scores = np.any(finite_scores, axis=1)
    soft_policy = np.any(positive_policy, axis=1)
    ab_root_scores = (score_sources == "ab_root") & soft_scores
    invalid = int(np.sum(~np.any(matches, axis=1)))
    policy_active_count = int(np.sum(policy_active))
    value_active_count = int(np.sum(value_active))
    q_rows_ge2 = np.sum(finite_scores, axis=1) >= 2
    usable_finite_scores = finite_scores.copy()
    skip_rows = _q_skip_rows_for_arrays(
        teachers,
        score_sources,
        q_skip_teacher_prefixes,
    )
    if np.any(skip_rows):
        usable_finite_scores[skip_rows] = False
    usable_q_rows_ge2 = np.sum(usable_finite_scores, axis=1) >= 2
    q_score_rows_ge2 = int(np.sum(q_rows_ge2))
    usable_q_score_rows_ge2 = int(np.sum(usable_q_rows_ge2))
    return {
        "samples": n,
        "soft_policy_rows": int(policy_rows),
        "soft_policy_fraction": float(policy_rows / max(n, 1)),
        "soft_score_rows": int(score_rows),
        "soft_score_fraction": float(score_rows / max(n, 1)),
        "policy_active_soft_policy_fraction": (
            float(np.sum(policy_active & soft_policy) / max(policy_active_count, 1))
        ),
        "policy_active_soft_score_fraction": (
            float(np.sum(policy_active & soft_scores) / max(policy_active_count, 1))
        ),
        "soft_score_legal_coverage_mean": float(np.mean(score_coverage)) if n else 0.0,
        "soft_score_legal_coverage_p50": float(np.percentile(score_coverage, 50)) if n else 0.0,
        "soft_policy_legal_coverage_mean": float(np.mean(policy_coverage)) if n else 0.0,
        "soft_policy_legal_coverage_p50": float(np.percentile(policy_coverage, 50)) if n else 0.0,
        "effective_soft_distillation_rows": int(np.sum(effective_soft_distillation)),
        "effective_soft_distillation_fraction": (
            float(np.mean(effective_soft_distillation)) if n else 0.0
        ),
        "policy_active_effective_soft_distillation_rows": int(
            np.sum(policy_active & effective_soft_distillation)
        ),
        "policy_active_effective_soft_distillation_fraction": (
            float(np.sum(policy_active & effective_soft_distillation) / max(policy_active_count, 1))
        ),
        "soft_target_min_legal_coverage": float(soft_target_min_legal_coverage),
        "selected_action_score_fraction": (
            float(np.mean(selected_action_has_score)) if n else 0.0
        ),
        "q_score_rows_ge2": int(q_score_rows_ge2),
        "q_score_rows_ge2_fraction": float(q_score_rows_ge2 / max(n, 1)),
        "usable_q_score_rows_ge2": int(usable_q_score_rows_ge2),
        "usable_q_score_rows_ge2_fraction": float(usable_q_score_rows_ge2 / max(n, 1)),
        "q_score_rows_ge2_policy_active_fraction": (
            float(np.sum(policy_active & q_rows_ge2) / max(policy_active_count, 1))
        ),
        "usable_q_score_rows_ge2_policy_active_fraction": (
            float(np.sum(policy_active & usable_q_rows_ge2) / max(policy_active_count, 1))
        ),
        "q_skip_teacher_prefixes": list(q_skip_teacher_prefixes),
        "outcome_rows": int(np.sum(winners != "")),
        "outcome_fraction": float(np.mean(winners != "")) if n else 0.0,
        "clean_terminal_outcome_rows": int(np.sum((winners != "") & ~truncated)),
        "clean_terminal_outcome_fraction": (
            float(np.mean((winners != "") & ~truncated)) if n else 0.0
        ),
        "final_public_vp_rows": int(np.sum(has_final_vps)),
        "final_public_vp_fraction": float(np.mean(has_final_vps)) if n else 0.0,
        "final_actual_vp_rows": int(np.sum(has_actual_vps)),
        "final_actual_vp_fraction": float(np.mean(has_actual_vps)) if n else 0.0,
        "unflagged_final_actual_vp_rows": int(np.sum(unflagged_actual_vp_rows)),
        "unflagged_final_public_vp_rows": int(np.sum(unflagged_public_vp_rows)),
        "truncated_rows": int(np.sum(truncated)),
        "truncated_fraction": float(np.mean(truncated)) if n else 0.0,
        "forced_action_rows": int(np.sum(legal_counts == 1)),
        "forced_action_fraction": float(np.mean(legal_counts == 1)) if n else 0.0,
        "policy_weight_zero_rows": int(np.sum(policy_multiplier <= 0.0)),
        "policy_weight_zero_fraction": float(np.mean(policy_multiplier <= 0.0)) if n else 0.0,
        "value_weight_zero_rows": int(np.sum(value_multiplier <= 0.0)),
        "value_weight_zero_fraction": float(np.mean(value_multiplier <= 0.0)) if n else 0.0,
        "policy_active_rows": policy_active_count,
        "policy_active_fraction": float(policy_active_count / max(n, 1)),
        "value_active_rows": value_active_count,
        "value_active_fraction": float(value_active_count / max(n, 1)),
        "policy_effective_forced_action_fraction": (
            float(np.sum(policy_active & (legal_counts == 1))) / float(max(policy_active_count, 1))
        ),
        "policy_effective_roll_fraction": (
            float(np.sum(policy_active & (phases == "roll"))) / float(max(policy_active_count, 1))
        ),
        "invalid_teacher_actions": invalid,
        "legal_actions_mean": float(np.mean(legal_counts)) if n else 0.0,
        "legal_actions_p90": int(np.percentile(legal_counts, 90)) if n else 0,
        "legal_actions_max": int(np.max(legal_counts)) if n else 0,
        "teacher_counts": _string_counts(teachers),
        "phase_counts": _string_counts(phases),
        "target_score_source_counts": _string_counts(
            np.where(score_sources == "", "none", score_sources)
        ),
        "ab_root_score_rows": int(np.sum((score_sources == "ab_root") & np.isfinite(target_scores).any(axis=1))),
        "ab_root_score_fraction": float(
            np.mean((score_sources == "ab_root") & np.isfinite(target_scores).any(axis=1))
        )
        if n
        else 0.0,
        "policy_active_ab_root_score_fraction": (
            float(np.sum(policy_active & ab_root_scores) / max(policy_active_count, 1))
        ),
        "action_mask_version_counts": _string_counts(action_mask_versions),
        "by_teacher": _quality_by_field(
            data,
            "teacher_name",
            q_skip_teacher_prefixes=q_skip_teacher_prefixes,
            soft_target_temperature=soft_target_temperature,
            soft_target_source=soft_target_source,
            soft_target_min_legal_coverage=soft_target_min_legal_coverage,
            effective_soft_distillation=effective_soft_distillation,
        ),
        "by_phase": _quality_by_field(
            data,
            "phase",
            q_skip_teacher_prefixes=q_skip_teacher_prefixes,
            soft_target_temperature=soft_target_temperature,
            soft_target_source=soft_target_source,
            soft_target_min_legal_coverage=soft_target_min_legal_coverage,
            effective_soft_distillation=effective_soft_distillation,
        ),
        "by_teacher_phase": _quality_by_fields(
            data,
            ("teacher_name", "phase"),
            q_skip_teacher_prefixes=q_skip_teacher_prefixes,
            soft_target_temperature=soft_target_temperature,
            soft_target_source=soft_target_source,
            soft_target_min_legal_coverage=soft_target_min_legal_coverage,
            effective_soft_distillation=effective_soft_distillation,
        ),
    }


def sample_weight_quality(data: dict, weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "effective_sample_size": 0.0,
            "effective_sample_fraction": 0.0,
            "positive_sample_count": 0,
            "positive_sample_fraction": 0.0,
            "positive_effective_sample_size": 0.0,
            "positive_effective_sample_fraction": 0.0,
            "by_teacher": {},
            "by_phase": {},
        }
    denom = float(np.sum(weights * weights))
    effective = float(np.sum(weights) ** 2 / denom) if denom > 0.0 else 0.0
    positive = weights[weights > 0.0]
    positive_denom = float(np.sum(positive * positive))
    positive_effective = (
        float(np.sum(positive) ** 2 / positive_denom) if positive_denom > 0.0 else 0.0
    )
    return {
        "mean": float(np.mean(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "effective_sample_size": effective,
        "effective_sample_fraction": effective / float(len(weights)),
        # Sparse self-play policy supervision deliberately assigns zero weight to
        # forced and fast-search rows.  The all-row ESS fraction therefore mostly
        # measures target coverage, not variance among examples that actually
        # enter the loss.  Report both factors so an n128/n256 run cannot mistake
        # 12% policy coverage for a 12%-efficient importance-weighting scheme.
        "positive_sample_count": int(positive.size),
        "positive_sample_fraction": float(positive.size / len(weights)),
        "positive_effective_sample_size": positive_effective,
        "positive_effective_sample_fraction": (
            positive_effective / float(positive.size) if positive.size else 0.0
        ),
        "by_teacher": _weight_by_field(data, weights, "teacher_name"),
        "by_phase": _weight_by_field(data, weights, "phase"),
    }


def per_game_weight_quality(data: dict, weights: np.ndarray) -> dict:
    """CAT-60 verification: log the actual per-game total weight mass directly (not just
    trust the per-game normalization formula) so a smoke train can confirm games of very
    different lengths end up with roughly equal total value-loss mass when
    --per-game-value-weight is on -- and, for comparison, how unequal it is when off."""

    n = len(weights)
    if n == 0:
        return {"n_games": 0, "rows_per_game": {}, "total_weight_per_game": {}}
    seeds = np.asarray(data.get("game_seed", np.arange(n, dtype=np.int64)), dtype=np.int64)
    weight_array = np.asarray(weights)
    # Production memmaps are written game-by-game, so every game's rows form
    # one contiguous run.  Exploit that layout before falling back to the fully
    # general factorisation.  ``np.unique(..., return_inverse=True)`` allocates
    # an int64 inverse for every row (about 363 MiB at 47.6M rows) and sorts all
    # of those rows merely to produce four scalar diagnostics.  The run path
    # allocates one boundary per game and remains exact for arbitrary seed
    # ordering.  A repeated non-contiguous seed is detected and takes the old
    # path, so this is not an assumption hidden in the report.
    boundaries = np.flatnonzero(seeds[1:] != seeds[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries))
    run_seeds = seeds[starts]
    if len(np.unique(run_seeds)) == len(run_seeds):
        counts = np.diff(
            np.concatenate((starts, np.asarray([n], dtype=np.int64)))
        )
        # Accumulate as float64 without first materialising a float64 copy of
        # the whole row vector (another ~363 MiB at production scale).
        totals = np.add.reduceat(weight_array, starts, dtype=np.float64)
        unique_seed_count = len(run_seeds)
    else:
        unique_seeds, inverse, counts = np.unique(
            seeds, return_inverse=True, return_counts=True
        )
        totals = np.zeros(len(unique_seeds), dtype=np.float64)
        np.add.at(totals, inverse, np.asarray(weight_array, dtype=np.float64))
        unique_seed_count = len(unique_seeds)
    return {
        "n_games": int(unique_seed_count),
        "rows_per_game": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
        },
        "total_weight_per_game": {
            "min": float(totals.min()),
            "max": float(totals.max()),
            "mean": float(totals.mean()),
            "std": float(totals.std()),
        },
    }


def validate_teacher_data_schema(policy, data: dict, data_quality: dict, env_config) -> None:
    config = getattr(policy, "config", None)
    action_size = int(getattr(policy, "action_size", getattr(config, "action_size", 0)))
    observation_size = int(getattr(config, "observation_size", 0))
    context_size = int(
        getattr(
            policy,
            "context_action_feature_size",
            getattr(config, "context_action_feature_size", 0),
        )
    )
    # obs / context / entity-token columns are checked for shape only, so read
    # them without np.asarray -- a MemmapCorpus streams these lazily and
    # materialising the full column here would defeat the streaming loader.
    obs = data["obs"]
    legal = np.asarray(data["legal_action_ids"])
    context = data["legal_action_context"]
    actions = np.asarray(data["action_taken"])
    problems: list[str] = []
    if observation_size and obs.ndim == 2 and int(obs.shape[1]) != observation_size:
        problems.append(
            f"obs width {obs.shape[1]} != checkpoint observation_size {observation_size}"
        )
    if legal.ndim != 2:
        problems.append(f"legal_action_ids must be rank 2, got {legal.shape}")
    else:
        duplicate_rows = _duplicate_legal_action_rows(legal)
        if duplicate_rows.size:
            preview = ", ".join(str(int(row)) for row in duplicate_rows[:5])
            problems.append(
                "duplicate legal action ids in rows "
                f"{preview}; each legal_action_ids row must be unique"
            )
    if context.ndim != 3:
        problems.append(f"legal_action_context must be rank 3, got {context.shape}")
    elif context_size and int(context.shape[2]) != context_size:
        problems.append(
            f"legal_action_context width {context.shape[2]} != checkpoint context size {context_size}"
        )
    if getattr(policy, "policy_type", "") == "entity_graph":
        missing_entity = [key for key in ENTITY_BATCH_KEYS if key not in data]
        if missing_entity:
            problems.append(
                "entity_graph requires converted entity-token fields; "
                f"missing {missing_entity[:8]}{'...' if len(missing_entity) > 8 else ''}"
            )
        else:
            entity_shapes = {
                "hex_tokens": (3, 19, HEX_FEATURE_SIZE),
                "vertex_tokens": (3, 54, VERTEX_FEATURE_SIZE),
                "edge_tokens": (3, 72, EDGE_FEATURE_SIZE),
                "player_tokens": (3, None, PLAYER_FEATURE_SIZE),
                "global_tokens": (3, 1, GLOBAL_FEATURE_SIZE),
                "legal_action_tokens": (3, legal.shape[1] if legal.ndim == 2 else None, None),
                "event_tokens": (3, None, EVENT_FEATURE_SIZE),
            }
            feature_sizes = {
                "legal_action_tokens": int(getattr(config, "legal_action_feature_size", 0) or 0),
            }
            for key, expected in entity_shapes.items():
                value = data[key]  # shape-only checks; keep lazy columns unmaterialised
                if value.ndim != expected[0]:
                    problems.append(f"{key} must be rank {expected[0]}, got {value.shape}")
                    continue
                if expected[1] is not None and int(value.shape[1]) != int(expected[1]):
                    problems.append(
                        f"{key} dim1 {value.shape[1]} != expected {expected[1]}"
                    )
                expected_width = feature_sizes.get(key)
                if expected_width is None:
                    expected_width = expected[2]
                if expected_width and int(value.shape[2]) != int(expected_width):
                    problems.append(
                        f"{key} width {value.shape[2]} != checkpoint width {expected_width}"
                    )
            for key in (
                "hex_mask",
                "vertex_mask",
                "edge_mask",
                "player_mask",
                "legal_action_mask",
                "event_mask",
            ):
                value = data[key]  # shape-only checks; keep lazy columns unmaterialised
                if value.ndim != 2 or value.shape[0] != len(actions):
                    problems.append(f"{key} must be rank 2 with {len(actions)} rows, got {value.shape}")
            if (
                legal.ndim == 2
                and "legal_action_tokens" in data
                and "legal_action_mask" in data
                and data["legal_action_tokens"].shape[1] != legal.shape[1]
            ):
                problems.append(
                    "legal_action_tokens candidate width must match legal_action_ids width"
                )
    if action_size:
        valid_legal = legal[legal >= 0]
        if valid_legal.size and int(np.max(valid_legal)) >= action_size:
            problems.append(
                f"legal action id max {int(np.max(valid_legal))} >= checkpoint action_size {action_size}"
            )
        if actions.size and (int(np.min(actions)) < 0 or int(np.max(actions)) >= action_size):
            problems.append(
                f"action_taken range [{int(np.min(actions))}, {int(np.max(actions))}] "
                f"is outside checkpoint action_size {action_size}"
            )
    if int(data_quality.get("invalid_teacher_actions", 0)) != 0:
        problems.append(
            f"invalid_teacher_actions={int(data_quality['invalid_teacher_actions'])}; "
            "teacher action must be present in legal_action_ids"
        )
    version_column = data.get("action_mask_version")
    if isinstance(version_column, _MemmapCategoricalColumn):
        present_versions = version_column.present_values()
        nonempty_versions = sorted(version for version in present_versions if version)
        has_empty_version = "" in present_versions
    else:
        versions = np.asarray(
            np.full(len(actions), "") if version_column is None else version_column
        ).astype(str)
        nonempty_versions = sorted({version for version in versions if version})
        has_empty_version = bool(np.any(versions == ""))
    if nonempty_versions and has_empty_version:
        problems.append(
            "mixed known and unknown action_mask_version values; regenerate or curate "
            "old shards instead of mixing empty-version rows with versioned rows"
        )
    if not nonempty_versions and len(actions) >= 1000:
        problems.append(
            "missing action_mask_version for a production-sized teacher dataset; "
            "regenerate or curate shards with current action catalog provenance"
        )
    if len(nonempty_versions) > 1:
        problems.append(f"mixed action_mask_version values: {nonempty_versions}")
    expected_version = _expected_action_mask_version(env_config)
    checkpoint_version = str(getattr(config, "action_mask_version", "") or "")
    if expected_version and checkpoint_version and checkpoint_version != expected_version:
        problems.append(
            f"checkpoint action_mask_version {checkpoint_version!r} does not match "
            f"current env action_mask_version {expected_version!r}"
        )
    if expected_version and nonempty_versions and nonempty_versions != [expected_version]:
        problems.append(
            f"teacher action_mask_version {nonempty_versions} does not match current "
            f"env action_mask_version {expected_version!r}"
        )
    expected_static_hash = _expected_static_action_features_sha256(env_config)
    checkpoint_static_hash = _policy_static_action_features_sha256(policy)
    if (
        expected_static_hash
        and checkpoint_static_hash
        and checkpoint_static_hash != expected_static_hash
    ):
        problems.append(
            "checkpoint static_action_features_sha256 does not match current "
            f"env action features: checkpoint={checkpoint_static_hash} "
            f"current={expected_static_hash}"
        )
    if problems:
        raise SystemExit("teacher data schema validation failed: " + "; ".join(problems))


def _expected_action_mask_version(env_config) -> str:
    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv

    env = ColonistMultiAgentEnv(env_config)
    try:
        _, info = env.reset(seed=0)
        return str(info.get("action_mask_version", ""))
    finally:
        env.close()


def _expected_static_action_features_sha256(env_config) -> str:
    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv

    env = ColonistMultiAgentEnv(env_config)
    try:
        env.reset(seed=0)
        return _array_sha256(build_action_feature_table(env))
    finally:
        env.close()


def _policy_static_action_features_sha256(policy) -> str:
    static = getattr(policy, "static_action_features", None)
    if static is None:
        static = getattr(policy, "action_features", None)
    if static is None:
        return ""
    if hasattr(static, "detach"):
        static = static.detach().cpu().numpy()
    return _array_sha256(np.asarray(static, dtype=np.float32))


def _quality_by_field(
    data: dict,
    field: str,
    *,
    limit: int = 40,
    q_skip_teacher_prefixes: tuple[str, ...] = (),
    soft_target_temperature: float = 0.7,
    soft_target_source: str = "policy",
    soft_target_min_legal_coverage: float = 0.5,
    effective_soft_distillation: np.ndarray | None = None,
) -> dict[str, dict]:
    n = int(len(data["action_taken"]))
    values = np.asarray(data.get(field, np.full(n, ""))).astype(str)
    result = {}
    for key in _top_group_keys(values, limit=limit):
        mask = values == key
        result[str(key)] = _quality_for_mask(
            data,
            mask,
            q_skip_teacher_prefixes=q_skip_teacher_prefixes,
            soft_target_temperature=soft_target_temperature,
            soft_target_source=soft_target_source,
            soft_target_min_legal_coverage=soft_target_min_legal_coverage,
            effective_soft_distillation=effective_soft_distillation,
        )
    return result


def _quality_by_fields(
    data: dict,
    fields: tuple[str, ...],
    *,
    limit: int = 80,
    q_skip_teacher_prefixes: tuple[str, ...] = (),
    soft_target_temperature: float = 0.7,
    soft_target_source: str = "policy",
    soft_target_min_legal_coverage: float = 0.5,
    effective_soft_distillation: np.ndarray | None = None,
) -> dict[str, dict]:
    n = int(len(data["action_taken"]))
    columns = [
        np.asarray(data.get(field, np.full(n, ""))).astype(str)
        for field in fields
    ]
    keys = np.asarray(["|".join(parts) for parts in zip(*columns, strict=False)])
    result = {}
    for key in _top_group_keys(keys, limit=limit):
        mask = keys == key
        result[str(key)] = _quality_for_mask(
            data,
            mask,
            q_skip_teacher_prefixes=q_skip_teacher_prefixes,
            soft_target_temperature=soft_target_temperature,
            soft_target_source=soft_target_source,
            soft_target_min_legal_coverage=soft_target_min_legal_coverage,
            effective_soft_distillation=effective_soft_distillation,
        )
    return result


def _quality_for_mask(
    data: dict,
    mask: np.ndarray,
    *,
    q_skip_teacher_prefixes: tuple[str, ...] = (),
    soft_target_temperature: float = 0.7,
    soft_target_source: str = "policy",
    soft_target_min_legal_coverage: float = 0.5,
    effective_soft_distillation: np.ndarray | None = None,
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    n = int(np.sum(mask))
    if n == 0:
        return {}
    legal = data["legal_action_ids"][mask]
    actions = data["action_taken"][mask]
    legal_counts = np.sum(legal >= 0, axis=1)
    target_policy = np.asarray(data.get("target_policy", np.zeros((len(data["action_taken"]), 0)))[mask], dtype=np.float32)
    target_scores = np.asarray(data.get("target_scores", np.full((len(data["action_taken"]), 0), np.nan))[mask], dtype=np.float32)
    teachers = np.asarray(data.get("teacher_name", np.full(len(data["action_taken"]), "")))[mask].astype(str)
    score_sources = np.asarray(
        data.get("target_score_source", np.full(len(data["action_taken"]), ""))
    )[mask].astype(str)
    winners = np.asarray(data.get("winner", np.full(len(data["action_taken"]), "")))[mask].astype(str)
    players = np.asarray(data.get("player", np.full(len(data["action_taken"]), "")))[mask].astype(str)
    truncated = np.asarray(data.get("truncated", np.zeros(len(data["action_taken"]), dtype=np.bool_)))[mask].astype(bool)
    phases = np.asarray(data.get("phase", np.full(len(data["action_taken"]), "")))[mask].astype(str)
    policy_multiplier = np.asarray(
        data.get("policy_weight_multiplier", np.ones(len(data["action_taken"]), dtype=np.float32))
    )[mask].astype(np.float32)
    value_multiplier = np.asarray(
        data.get("value_weight_multiplier", np.ones(len(data["action_taken"]), dtype=np.float32))
    )[mask].astype(np.float32)
    policy_active = policy_multiplier > 0.0
    value_active = value_multiplier > 0.0
    if effective_soft_distillation is not None:
        effective_soft_distillation_masked = np.asarray(
            effective_soft_distillation,
            dtype=bool,
        )[mask]
    else:
        batch = np.flatnonzero(mask).astype(np.int64)
        soft_payload = _soft_target_array(
            data,
            batch,
            soft_target_temperature,
            soft_target_source,
        )
        if soft_payload is None:
            effective_soft_distillation_masked = np.zeros(n, dtype=bool)
        else:
            effective_soft_distillation_masked = _has_distillation_distribution(
                soft_payload[0],
                soft_payload[1],
                legal_action_ids=legal,
                min_legal_coverage=soft_target_min_legal_coverage,
            )
    has_final_vps = np.asarray(
        data.get("has_final_public_vps", np.zeros(len(data["action_taken"]), dtype=np.bool_))
    )[mask].astype(bool)
    has_actual_vps = np.asarray(
        data.get("has_final_actual_vps", np.zeros(len(data["action_taken"]), dtype=np.bool_))
    )[mask].astype(bool)
    matches = legal == actions[:, None]
    target_policy_mask_all = np.asarray(
        data.get(
            "target_policy_mask",
            np.where(
                np.isfinite(data.get("target_policy", np.zeros((len(data["action_taken"]), 0)))),
                data.get("target_policy", np.zeros((len(data["action_taken"]), 0))),
                0.0,
            )
            > 0.0,
        ),
        dtype=np.bool_,
    )
    target_scores_all = data.get(
        "target_scores",
        np.full((len(data["action_taken"]), 0), np.nan),
    )
    target_scores_mask_all = np.asarray(
        data.get("target_scores_mask", np.isfinite(target_scores_all)),
        dtype=np.bool_,
    )
    target_policy_mask = np.asarray(
        target_policy_mask_all,
        dtype=np.bool_,
    )[mask]
    target_scores_mask = np.asarray(
        target_scores_mask_all,
        dtype=np.bool_,
    )[mask]
    finite_scores = (legal >= 0) & target_scores_mask & np.isfinite(target_scores)
    positive_policy = (legal >= 0) & target_policy_mask & (
        np.where(np.isfinite(target_policy), target_policy, 0.0) > 0.0
    )
    soft_policy = np.any(positive_policy, axis=1)
    soft_scores = np.any(finite_scores, axis=1)
    ab_root_scores = (score_sources == "ab_root") & soft_scores
    selected_action_has_score = np.any(matches & finite_scores, axis=1)
    legal_denominator = np.maximum(legal_counts, 1)
    score_coverage = np.sum(finite_scores, axis=1) / legal_denominator
    policy_coverage = np.sum(positive_policy, axis=1) / legal_denominator
    q_rows_ge2 = np.sum(finite_scores, axis=1) >= 2
    usable_finite_scores = finite_scores.copy()
    skip_rows = _q_skip_rows_for_arrays(
        teachers,
        score_sources,
        q_skip_teacher_prefixes,
    )
    if np.any(skip_rows):
        usable_finite_scores[skip_rows] = False
    usable_q_rows_ge2 = np.sum(usable_finite_scores, axis=1) >= 2
    clean_outcome = (winners != "") & ~truncated
    winner_rows = clean_outcome & (winners == players)
    policy_active_count = int(np.sum(policy_active))
    value_active_count = int(np.sum(value_active))
    return {
        "samples": n,
        "target_score_source_counts": _string_counts(
            np.where(score_sources == "", "none", score_sources)
        ),
        "ab_root_score_fraction": float(
            np.mean((score_sources == "ab_root") & soft_scores)
        ),
        "soft_policy_fraction": float(np.mean(soft_policy)),
        "soft_score_fraction": float(np.mean(soft_scores)),
        "policy_active_soft_policy_fraction": (
            float(np.sum(policy_active & soft_policy) / max(policy_active_count, 1))
        ),
        "policy_active_soft_score_fraction": (
            float(np.sum(policy_active & soft_scores) / max(policy_active_count, 1))
        ),
        "policy_active_ab_root_score_fraction": (
            float(np.sum(policy_active & ab_root_scores) / max(policy_active_count, 1))
        ),
        "soft_score_legal_coverage_mean": float(np.mean(score_coverage)),
        "soft_score_legal_coverage_p50": float(np.percentile(score_coverage, 50)),
        "soft_policy_legal_coverage_mean": float(np.mean(policy_coverage)),
        "soft_policy_legal_coverage_p50": float(np.percentile(policy_coverage, 50)),
        "effective_soft_distillation_fraction": float(np.mean(effective_soft_distillation_masked)),
        "policy_active_effective_soft_distillation_fraction": (
            float(np.sum(policy_active & effective_soft_distillation_masked) / max(policy_active_count, 1))
        ),
        "effective_soft_distillation_rows": int(np.sum(effective_soft_distillation_masked)),
        "policy_active_effective_soft_distillation_rows": int(
            np.sum(policy_active & effective_soft_distillation_masked)
        ),
        "selected_action_score_fraction": float(np.mean(selected_action_has_score)),
        "q_score_rows_ge2": int(np.sum(q_rows_ge2)),
        "q_score_rows_ge2_fraction": float(np.mean(q_rows_ge2)),
        "q_score_rows_ge2_policy_active_fraction": (
            float(np.sum(policy_active & q_rows_ge2) / max(policy_active_count, 1))
        ),
        "usable_q_score_rows_ge2": int(np.sum(usable_q_rows_ge2)),
        "usable_q_score_rows_ge2_fraction": float(np.mean(usable_q_rows_ge2)),
        "usable_q_score_rows_ge2_policy_active_fraction": (
            float(np.sum(policy_active & usable_q_rows_ge2) / max(policy_active_count, 1))
        ),
        "forced_action_fraction": float(np.mean(legal_counts == 1)),
        "policy_weight_zero_fraction": float(np.mean(policy_multiplier <= 0.0)),
        "value_weight_zero_fraction": float(np.mean(value_multiplier <= 0.0)),
        "policy_active_rows": policy_active_count,
        "policy_active_fraction": float(policy_active_count / max(n, 1)),
        "value_active_rows": value_active_count,
        "value_active_fraction": float(value_active_count / max(n, 1)),
        "policy_effective_forced_action_fraction": (
            float(np.sum(policy_active & (legal_counts == 1))) / float(max(policy_active_count, 1))
        ),
        "policy_effective_roll_fraction": (
            float(np.sum(policy_active & (phases == "roll"))) / float(max(policy_active_count, 1))
        ),
        "invalid_teacher_actions": int(np.sum(~np.any(matches, axis=1))),
        "truncated_fraction": float(np.mean(truncated)),
        "outcome_fraction": float(np.mean(winners != "")),
        "clean_terminal_outcome_fraction": float(np.mean(clean_outcome)),
        "final_public_vp_fraction": float(np.mean(has_final_vps)),
        "final_actual_vp_fraction": float(np.mean(has_actual_vps)),
        "winner_row_fraction": float(np.mean(winner_rows)) if np.any(clean_outcome) else 0.0,
        "legal_actions_mean": float(np.mean(legal_counts)),
        "legal_actions_p90": int(np.percentile(legal_counts, 90)),
    }


def _weight_by_field(
    data: dict,
    weights: np.ndarray,
    field: str,
    *,
    limit: int = 40,
) -> dict[str, dict]:
    n = int(len(data["action_taken"]))
    column = data.get(field)
    if isinstance(column, _MemmapCategoricalColumn) or bool(
        getattr(column, "supports_grouped_weights", False)
    ):
        return column.grouped_weights(weights, limit=limit)
    values = np.asarray(np.full(n, "") if column is None else column).astype(str)
    result = {}
    for key in _top_group_keys(values, limit=limit):
        mask = values == key
        raw_count = int(np.sum(mask))
        total = float(np.sum(weights[mask]))
        result[str(key)] = {
            "raw_samples": raw_count,
            "weight_sum": total,
            "mean_weight": total / max(raw_count, 1),
        }
    return result


def _top_group_keys(values: np.ndarray, *, limit: int) -> list[str]:
    if values.size == 0:
        return []
    unique, counts = np.unique(values.astype(str), return_counts=True)
    order = np.argsort(-counts)
    return [str(unique[index]) for index in order[:limit]]


def _string_counts(values: np.ndarray, *, limit: int = 20) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values.astype(str), return_counts=True)
    order = np.argsort(-counts)
    return {
        str(unique[index]): int(counts[index])
        for index in order[:limit]
    }


def _load_npz(path: Path):
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as error:
            raise SystemExit("zstandard is required to read .npz.zst shards") from error
        data = zstd.ZstdDecompressor().decompress(path.read_bytes())
        return np.load(io.BytesIO(data), allow_pickle=False)
    return np.load(path, allow_pickle=False)


def _dense_context(data: dict, batch: np.ndarray, action_size: int, context_size: int) -> np.ndarray:
    dense = np.zeros((len(batch), action_size, context_size), dtype=np.float32)
    valid = data["legal_action_ids"][batch]
    context = data["legal_action_context"][batch]
    for row in range(len(batch)):
        actions = valid[row]
        keep = actions >= 0
        dense[row, actions[keep].astype(np.int64), :] = context[row, keep, :context_size]
    return dense


def _valid_lists(values: np.ndarray) -> list[tuple[int, ...]]:
    # OPT-4: numpy boolean mask + tolist avoids the per-element double int()
    # cast and the Python filter loop. Output is identical -- non-negative
    # action ids in original order as a tuple of Python ints, per row.
    return [tuple(row[row >= 0].tolist()) for row in np.asarray(values)]


def _target_columns(legal_action_ids: np.ndarray, actions: np.ndarray) -> np.ndarray:
    duplicate_rows = _duplicate_legal_action_rows(legal_action_ids)
    if duplicate_rows.size:
        first = int(duplicate_rows[0])
        valid = legal_action_ids[first][legal_action_ids[first] >= 0]
        raise ValueError(
            "duplicate legal action ids in teacher row "
            f"{first}: {valid.astype(np.int64).tolist()}"
        )
    matches = legal_action_ids == actions[:, None]
    missing = ~np.any(matches, axis=1)
    if np.any(missing):
        first = int(np.flatnonzero(missing)[0])
        raise ValueError(
            f"teacher action {int(actions[first])} is not present in legal candidates"
        )
    return np.argmax(matches, axis=1).astype(np.int64, copy=False)


def _duplicate_legal_action_rows(legal_action_ids: np.ndarray) -> np.ndarray:
    """Vectorized (task #76 side-finding): the original per-row Python loop
    with a np.unique() call inside was O(n) Python-level overhead that got
    dramatically slower under real memory pressure on very large corpora
    (discovered as a multi-hour pre-training stall on a 14.4M-row corpus,
    task #65). Sort each row (padding sentinel -1 sorts first), then a
    duplicate is any adjacent-equal pair among the non-negative (real action
    id) entries -- two adjacent -1 padding slots are explicitly excluded, not
    real duplicates.
    """
    legal = np.asarray(legal_action_ids)
    if legal.ndim != 2 or legal.shape[1] < 2:
        return np.asarray([], dtype=np.int64)
    sorted_rows = np.sort(legal, axis=1)
    adjacent_equal = np.diff(sorted_rows, axis=1) == 0
    later_is_valid = sorted_rows[:, 1:] >= 0
    has_duplicate = np.any(adjacent_equal & later_is_valid, axis=1)
    return np.nonzero(has_duplicate)[0].astype(np.int64)


def _concat_padded(key: str, values: list[np.ndarray]) -> np.ndarray:
    if len(values) == 1:
        return values[0]
    if key in {
        "legal_action_ids",
        "target_policy",
        "prior_policy",
        "target_scores",
        "target_policy_mask",
        "target_scores_mask",
        "afterstate_target",
        "afterstate_target_mask",
        "legal_action_mask",
    }:
        if key == "legal_action_ids":
            fill = -1
        elif key in {"target_scores", "afterstate_target"}:
            fill = np.nan
        elif key in {
            "target_policy_mask",
            "target_scores_mask",
            "afterstate_target_mask",
            "legal_action_mask",
        }:
            fill = False
        else:
            fill = 0.0
        width = max(value.shape[1] for value in values)
        padded = []
        for value in values:
            if value.shape[1] == width:
                padded.append(value)
                continue
            out = np.full((value.shape[0], width), fill, dtype=value.dtype)
            out[:, : value.shape[1]] = value
            padded.append(out)
        return np.concatenate(padded, axis=0)
    if key == "legal_action_context":
        width = max(value.shape[1] for value in values)
        feature_size = max(value.shape[2] for value in values)
        padded = []
        for value in values:
            if value.shape[1] == width and value.shape[2] == feature_size:
                padded.append(value)
                continue
            out = np.zeros((value.shape[0], width, feature_size), dtype=value.dtype)
            out[:, : value.shape[1], : value.shape[2]] = value
            padded.append(out)
        return np.concatenate(padded, axis=0)
    if key in {"legal_action_tokens", "legal_action_target_ids"}:
        width = max(value.shape[1] for value in values)
        feature_size = max(value.shape[2] for value in values)
        fill = -1 if key == "legal_action_target_ids" else 0.0
        padded = []
        for value in values:
            if value.shape[1] == width and value.shape[2] == feature_size:
                padded.append(value)
                continue
            out = np.full((value.shape[0], width, feature_size), fill, dtype=value.dtype)
            out[:, : value.shape[1], : value.shape[2]] = value
            padded.append(out)
        return np.concatenate(padded, axis=0)
    return np.concatenate(values, axis=0)


def _soft_targets_legal(
    data: dict,
    batch: np.ndarray,
    device,
    temperature: float,
    source: str,
    min_legal_coverage: float = 0.0,
):
    import torch

    payload = _soft_target_array(data, batch, temperature, source)
    if payload is None:
        return None, None, None
    target, support = payload
    soft = torch.as_tensor(target, dtype=torch.float32, device=device)
    support_t = torch.as_tensor(support, dtype=torch.bool, device=device)
    has_soft = torch.as_tensor(
        _has_distillation_distribution(
            target,
            support,
            legal_action_ids=data["legal_action_ids"][batch],
            min_legal_coverage=min_legal_coverage,
        ),
        dtype=torch.bool,
        device=device,
    )
    return soft, has_soft, support_t


def _soft_targets_full(
    data: dict,
    batch: np.ndarray,
    action_size: int,
    device,
    temperature: float,
    source: str,
    min_legal_coverage: float = 0.0,
):
    import torch

    payload = _soft_target_array(data, batch, temperature, source)
    if payload is None:
        return None, None, None
    target, support = payload
    legal = data["legal_action_ids"][batch]
    has_soft_np = _has_distillation_distribution(
        target,
        support,
        legal_action_ids=legal,
        min_legal_coverage=min_legal_coverage,
    )
    dense = np.zeros((len(batch), action_size), dtype=np.float32)
    dense_support = np.zeros((len(batch), action_size), dtype=np.bool_)
    for row in range(len(batch)):
        keep = legal[row] >= 0
        dense[row, legal[row, keep].astype(np.int64)] = target[row, keep]
        dense_support[row, legal[row, keep].astype(np.int64)] = support[row, keep]
    soft = torch.as_tensor(dense, dtype=torch.float32, device=device)
    support_t = torch.as_tensor(dense_support, dtype=torch.bool, device=device)
    has_soft = torch.as_tensor(
        has_soft_np,
        dtype=torch.bool,
        device=device,
    )
    return soft, has_soft, support_t


def _batch_array_or_fill(
    data: dict,
    field: str,
    batch: np.ndarray,
    fill_value,
    *,
    dtype=None,
) -> np.ndarray:
    """Read one optional column without constructing a full-corpus default."""

    if field in data:
        return np.asarray(data[field][batch], dtype=dtype)
    return np.full(len(batch), fill_value, dtype=dtype)


def _soft_target_array(
    data: dict,
    batch: np.ndarray,
    temperature: float,
    source: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    policy = np.zeros_like(data["legal_action_ids"][batch], dtype=np.float32)
    policy_support = np.zeros_like(policy, dtype=np.bool_)
    has_policy = np.zeros(policy.shape[0], dtype=bool)
    if "target_policy" in data and source in {"prefer_policy", "policy", "prefer_scores"}:
        policy = np.asarray(data["target_policy"][batch], dtype=np.float32)
        if "target_policy_mask" in data:
            policy_support = np.asarray(data["target_policy_mask"][batch], dtype=np.bool_)
        else:
            policy_support = np.where(np.isfinite(policy), policy, 0.0) > 0.0
        policy = np.where(np.isfinite(policy), np.maximum(policy, 0.0), 0.0)
        policy = np.where(policy_support, policy, 0.0)
        sums = np.sum(policy, axis=1, keepdims=True)
        has_policy = sums[:, 0] > 0.0
        if np.any(has_policy):
            policy[has_policy] = policy[has_policy] / sums[has_policy]
    score_target = np.zeros_like(policy, dtype=np.float32)
    score_support = np.zeros_like(policy, dtype=np.bool_)
    has_scores = np.zeros(policy.shape[0], dtype=bool)
    if "target_scores" in data and source in {"prefer_scores", "scores", "prefer_policy"}:
        scores = np.asarray(data["target_scores"][batch], dtype=np.float32)
        if "target_scores_mask" in data:
            score_support = np.asarray(data["target_scores_mask"][batch], dtype=np.bool_)
        else:
            score_support = np.isfinite(scores)
        score_target = _scores_to_policy(
            scores,
            data["legal_action_ids"][batch],
            temperature,
            score_support=score_support,
        )
        has_scores = np.sum(score_target, axis=1) > 0.0

    teacher_names = _batch_array_or_fill(data, "teacher_name", batch, "").astype(str)
    score_sources = _batch_array_or_fill(
        data,
        "target_score_source",
        batch,
        "",
    ).astype(str)
    prefer_policy_rows = _prefer_policy_over_scores_for_teachers(
        teacher_names,
        score_sources,
    )
    ab_root_blend_rows = (
        _ab_root_score_rows(teacher_names, score_sources) & has_policy & has_scores
    )

    if source == "policy":
        target = policy
        support = policy_support
    elif source == "scores":
        target = score_target
        support = score_support
    elif source == "prefer_scores":
        target = policy.copy()
        support = policy_support.copy()
        target[has_scores] = score_target[has_scores]
        support[has_scores] = score_support[has_scores]
        target[prefer_policy_rows & has_policy] = policy[prefer_policy_rows & has_policy]
        support[prefer_policy_rows & has_policy] = policy_support[prefer_policy_rows & has_policy]
        if np.any(ab_root_blend_rows):
            blended = _blend_soft_target_rows(
                policy,
                score_target,
                ab_root_blend_rows,
                policy_weight=0.30,
            )
            target[ab_root_blend_rows] = blended
            support[ab_root_blend_rows] = blended > 0.0
    else:
        target = policy.copy()
        support = policy_support.copy()
        target[~has_policy & has_scores] = score_target[~has_policy & has_scores]
        support[~has_policy & has_scores] = score_support[~has_policy & has_scores]
    if not np.any(np.sum(target, axis=1) > 0.0):
        return None
    support &= target > 0.0
    return target.astype(np.float32, copy=False), support.astype(np.bool_, copy=False)


def _has_distillation_distribution(
    target: np.ndarray,
    support: np.ndarray,
    *,
    legal_action_ids: np.ndarray | None = None,
    min_legal_coverage: float = 0.0,
) -> np.ndarray:
    """Rows with only one supported target should train as hard labels.

    A one-hot "soft" target makes KL over the supported set equal zero because
    the supported softmax contains only the chosen action. Treating those rows
    as hard labels preserves the full cross-entropy gradient.
    """

    target = np.asarray(target, dtype=np.float32)
    support = np.asarray(support, dtype=np.bool_)
    positive = support & (target > 0.0)
    has_distribution = np.sum(positive, axis=1) > 1
    min_coverage = float(np.clip(min_legal_coverage, 0.0, 1.0))
    if min_coverage <= 0.0 or legal_action_ids is None:
        return has_distribution
    legal = np.asarray(legal_action_ids)
    legal_counts = np.maximum(np.sum(legal >= 0, axis=1), 1)
    coverage = np.sum(positive, axis=1) / legal_counts
    return has_distribution & (coverage >= min_coverage)


def _scores_to_policy(
    scores: np.ndarray,
    legal_action_ids: np.ndarray,
    temperature: float,
    *,
    score_support: np.ndarray | None = None,
) -> np.ndarray:
    target = np.zeros(scores.shape, dtype=np.float32)
    temp = max(float(temperature), 1.0e-6)
    if score_support is None:
        score_support = np.isfinite(scores)
    valid = (legal_action_ids >= 0) & np.asarray(score_support, dtype=np.bool_) & np.isfinite(scores)
    for row in range(scores.shape[0]):
        keep = valid[row]
        if not np.any(keep):
            continue
        logits = scores[row, keep].astype(np.float32)
        logits = logits / temp
        logits = logits - float(np.max(logits))
        logits = np.clip(logits, -60.0, 0.0)
        probs = np.exp(logits)
        total = float(np.sum(probs))
        if total > 0.0:
            target[row, keep] = probs / total
    return target


def _support_log_softmax(logits, support):
    import torch

    if support is None:
        return torch.nn.functional.log_softmax(logits, dim=-1)
    support = support.to(device=logits.device, dtype=torch.bool)
    if support.shape != logits.shape:
        raise ValueError(
            f"soft target support shape {tuple(support.shape)} does not match "
            f"logits shape {tuple(logits.shape)}"
        )
    # The support mask is only a shape/provenance guard here. Candidate logits are
    # already masked to legal actions upstream, so the soft-label denominator must
    # include every legal candidate. Otherwise a high logit on an unscored legal
    # action gets no distillation penalty and can silently dominate evaluation.
    return torch.nn.functional.log_softmax(logits, dim=-1)


def _prior_kl_telemetry(
    data: dict,
    batch: np.ndarray,
    logits,
    device,
):
    """SUCCESS TELEMETRY (gen-1 recipe): KL(model_policy || prior_policy) on a
    held-out GEN slice, alongside the reference KL(target_policy || prior_policy)
    -- training "worked" once the former reaches 60-80% of the latter; if it's
    still under ~30% of it by the end of training, the run underfit (a retrain
    at a higher LR is cheaper than spending gate games finding that out).

    Scoped to rows with a recorded prior, with no extra bookkeeping: prior_policy
    is populated by the self-play data generators -- gumbel_self_play.py's
    _build_decision_row and raw_selfplay.py's _build_raw_decision_row (the latter
    stores the seed model's raw prior distribution, since there is no search to
    improve it) -- while any other teacher source defaults it to all-zero (see
    _normalize_teacher_shard), which this function's `has_prior` filter excludes
    automatically.

    The returned per-row `kl_model_prior`/`kl_target_prior` tensors are
    differentiable through `logits` (the caller detaches at `.item()` for
    telemetry); _policy_kl_anchor_loss reuses this same computation, un-detached,
    as the training-time policy-KL anchor so the anchor and the telemetry measure
    the identical quantity.

    `logits` must already be in the SAME per-row legal-action ordering as
    data["target_policy"]/data["prior_policy"] (true for _eval_xdim_batch's
    outputs["logits"], which _soft_targets_legal already relies on for the
    same reason -- see _support_log_softmax above).
    """
    import torch

    if "prior_policy" not in data or "target_policy" not in data:
        return None
    prior_np = np.asarray(data["prior_policy"][batch], dtype=np.float32)
    target_np = np.asarray(data["target_policy"][batch], dtype=np.float32)
    legal_ids = np.asarray(data["legal_action_ids"][batch])
    valid_np = legal_ids >= 0
    has_prior_np = (prior_np * valid_np).sum(axis=1) > 1.0e-6
    if not has_prior_np.any():
        return None

    valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
    has_prior = torch.as_tensor(has_prior_np, dtype=torch.bool, device=device)
    prior = torch.as_tensor(prior_np, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    zeros = torch.zeros_like(prior)

    eps = 1.0e-8
    prior = torch.where(valid, prior, zeros)
    prior_norm = prior / torch.clamp(prior.sum(dim=-1, keepdim=True), min=eps)
    target = torch.where(valid, target, zeros)
    target_norm = target / torch.clamp(target.sum(dim=-1, keepdim=True), min=eps)

    model_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    model_probs = torch.where(valid, model_log_probs.exp(), zeros)

    log_prior = torch.log(torch.clamp(prior_norm, min=eps))
    log_target = torch.log(torch.clamp(target_norm, min=eps))

    kl_model_prior = torch.where(valid, model_probs * (model_log_probs - log_prior), zeros).sum(
        dim=-1
    )
    kl_prior_model = torch.where(
        valid, prior_norm * (log_prior - model_log_probs), zeros
    ).sum(dim=-1)
    kl_target_prior = torch.where(valid, target_norm * (log_target - log_prior), zeros).sum(
        dim=-1
    )
    return {
        "has_prior": has_prior,
        "kl_model_prior": kl_model_prior,
        "kl_prior_model": kl_prior_model,
        "kl_target_prior": kl_target_prior,
    }


def _active_policy_teacher_gap_telemetry(
    data: dict,
    batch: np.ndarray,
    logits,
    device,
    *,
    soft_targets,
    has_soft,
    policy_active,
):
    """Measure how much of the active soft-policy teacher gap was closed.

    This intentionally differs from the historical ``_prior_kl_telemetry``:

    * only rows with positive policy loss weight are eligible;
    * only usable multi-action soft targets are eligible; and
    * the primary distance is ``KL(target || model)``, the direction aligned
      with the soft cross-entropy objective.

    PCR shards retain improved policies for fast-search and forced rows even
    when their ``policy_weight_multiplier`` is zero.  Those rows are useful for
    provenance but are not optimization targets, so including them in a
    teacher-uptake denominator makes the resulting ratio structurally
    unattainable.  Rows without a recorded prior are also excluded because the
    baseline gap ``KL(target || prior)`` is undefined for them.
    """

    import torch

    if soft_targets is None or has_soft is None or "prior_policy" not in data:
        return None

    legal_np = np.asarray(data["legal_action_ids"][batch]) >= 0
    prior_np = np.asarray(data["prior_policy"][batch], dtype=np.float32)
    if prior_np.shape != legal_np.shape:
        return None

    valid = torch.as_tensor(legal_np, dtype=torch.bool, device=device)
    target = soft_targets.to(device=device, dtype=torch.float32)
    if target.shape != valid.shape or logits.shape != valid.shape:
        return None
    zeros = torch.zeros_like(target)
    eps = 1.0e-8

    target = torch.where(valid, torch.clamp(target, min=0.0), zeros)
    target = target / torch.clamp(target.sum(dim=-1, keepdim=True), min=eps)
    prior = torch.as_tensor(prior_np, dtype=torch.float32, device=device)
    prior = torch.where(valid, torch.clamp(prior, min=0.0), zeros)
    prior_mass = prior.sum(dim=-1)
    prior = prior / torch.clamp(prior_mass.unsqueeze(-1), min=eps)

    has_multi_action_target = ((target > 0.0) & valid).sum(dim=-1) > 1
    eligible = (
        policy_active.to(device=device, dtype=torch.bool)
        & has_soft.to(device=device, dtype=torch.bool)
        & has_multi_action_target
        & (prior_mass > eps)
    )

    log_target = torch.log(torch.clamp(target, min=eps))
    log_prior = torch.log(torch.clamp(prior, min=eps))
    model_log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    kl_target_model = torch.where(
        valid,
        target * (log_target - model_log_probs),
        zeros,
    ).sum(dim=-1).clamp_min(0.0)
    kl_target_prior = torch.where(
        valid,
        target * (log_target - log_prior),
        zeros,
    ).sum(dim=-1).clamp_min(0.0)
    return {
        "eligible": eligible,
        "kl_target_model": kl_target_model,
        "kl_target_prior": kl_target_prior,
    }


def _active_policy_teacher_gap_report(
    *, rows: float, kl_target_model_sum: float, kl_target_prior_sum: float
) -> dict[str, float | int]:
    """Reduce additive teacher-gap statistics into report-level metrics."""

    count = max(float(rows), 0.0)
    target_model = float(kl_target_model_sum)
    target_prior = float(kl_target_prior_sum)
    return {
        "active_policy_teacher_gap_rows": int(round(count)),
        "active_policy_kl_target_model_mean": target_model / max(count, 1.0),
        "active_policy_kl_target_prior_mean": target_prior / max(count, 1.0),
        "active_policy_teacher_gap_closure": (
            1.0 - target_model / target_prior
            if count > 0.0 and target_prior > 1.0e-8
            else 0.0
        ),
    }


def compute_policy_surprise_kl(
    data: dict, batch: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row "policy surprise" KL(target_policy || prior_policy) -- a pure,
    data-derived (no model/torch) NumPy port of ``_prior_kl_telemetry``'s
    ``kl_target_prior`` term, for use as a CAT-45 sample-frequency weighting
    signal rather than a training-time loss/telemetry term. Kept as a separate
    function (not a refactor of ``_prior_kl_telemetry``) so that function's
    existing tests/behaviour are untouched; the masking/eps/normalization
    conventions are intentionally identical so the two never silently drift
    apart.

    Returns ``(kl_target_prior, has_prior)`` over ``batch`` (or the whole
    corpus when ``batch`` is None), matching ``_prior_kl_telemetry``'s
    ``has_prior`` scoping: rows with no recorded ``prior_policy`` (any non-
    Gumbel-self-play teacher row, or any shard predating this column) get
    ``has_prior=False`` and ``kl_target_prior=0.0`` -- exactly the "old shards
    default to uniform weight" behaviour CAT-45 requires.
    """
    n = len(data["action_taken"]) if batch is None else len(batch)
    if "prior_policy" not in data or "target_policy" not in data:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.bool_)

    index = slice(None) if batch is None else batch
    prior = np.asarray(data["prior_policy"][index], dtype=np.float32)
    target = np.asarray(data["target_policy"][index], dtype=np.float32)
    legal_ids = np.asarray(data["legal_action_ids"][index])
    valid = legal_ids >= 0

    has_prior = (prior * valid).sum(axis=1) > 1.0e-6
    if not has_prior.any():
        return np.zeros(n, dtype=np.float32), has_prior

    eps = 1.0e-8
    prior = np.where(valid, prior, 0.0)
    prior_norm = prior / np.clip(prior.sum(axis=-1, keepdims=True), eps, None)
    target = np.where(valid, target, 0.0)
    target_norm = target / np.clip(target.sum(axis=-1, keepdims=True), eps, None)

    log_prior = np.log(np.clip(prior_norm, eps, None))
    log_target = np.log(np.clip(target_norm, eps, None))
    kl_target_prior = np.where(valid, target_norm * (log_target - log_prior), 0.0).sum(axis=-1)
    # Rows without a recorded prior have an undefined KL -- zero them explicitly
    # rather than trust the eps-clamped math above to land near zero.
    kl_target_prior = np.where(has_prior, kl_target_prior, 0.0)
    return kl_target_prior.astype(np.float32, copy=False), has_prior


def policy_surprise_sampling_weights(
    kl_target_prior: np.ndarray,
    has_prior: np.ndarray,
    *,
    weight_scale: float,
    cap: float,
) -> np.ndarray:
    """CAT-45 sample-frequency weight from per-row policy surprise.

    ``weight = 1.0 + weight_scale * min(kl_target_prior, cap)`` for rows with a
    recorded prior, ``weight = 1.0`` (uniform baseline) for everything else --
    rows with no recorded prior (pre-CAT-45 shards, non-Gumbel teacher rows) and
    ANY row whenever ``weight_scale <= 0.0`` (the default-off case). This is a
    strict *upweighting* of surprising states relative to a uniform floor of
    1.0, never a downweighting of agreeing ones -- consistent with "oversample
    high-KL states" (R8 / master plan Sec 4.5), not "starve low-KL states".

    This is deliberately NOT a literal reproduction of KataGo's own (unpublished
    here) sample-frequency constants -- it is "roughly KataGo-style" per the
    ticket's own framing ("half of KataGo's sample-frequency weighting scheme"),
    with an explicit, capped, documented formula of our own so the behaviour is
    auditable and testable independent of that precedent.
    """
    weight_scale = float(weight_scale)
    cap = float(cap)
    weights = np.ones_like(kl_target_prior, dtype=np.float64)
    if weight_scale > 0.0:
        weights[has_prior] = 1.0 + weight_scale * np.minimum(kl_target_prior[has_prior], cap)
    return weights.astype(np.float32, copy=False)


def _policy_kl_anchor_loss(
    data: dict,
    batch: np.ndarray,
    logits,
    device,
    *,
    direction: str = "forward",
):
    """Differentiable behavior anchor over multi-action rows with a prior.

    ``forward`` is KL(prior_policy || pi_theta), the recovery/default old-policy
    distillation objective. ``reverse`` retains historical KL(pi_theta ||
    prior_policy) only as an explicit compatibility ablation.

    Reuses _prior_kl_telemetry's exact per-row computation (un-detached here).
    Its optimization mean then excludes forced rows, whose one-element policy
    simplex makes KL identically zero; raw prior-KL telemetry may still report
    those rows as coverage. The masked mean goes through _weighted_mean_loss so
    it inherits the same DDP-correct global-denominator reduction the
    value/policy losses use.

    Returns None when the batch has no prior rows (caller then adds nothing to
    the loss). MUST be called with grad enabled -- i.e. from the training path,
    not the no_grad eval path where the telemetry variant lives.
    """
    parts = _policy_kl_anchor_loss_parts(
        data, batch, logits, device, direction=direction
    )
    return None if parts is None else parts[0]


def _policy_kl_anchor_loss_parts(
    data: dict,
    batch: np.ndarray,
    logits,
    device,
    *,
    direction: str = "forward",
):
    """Return conditional anchor mean plus its eligible-row sum/denominator.

    The scalar objective is normalized over authenticated, multi-action rows
    that carry a prior. Returning the raw parts lets epoch/eval telemetry use
    that same denominator instead of silently diluting the reported metric by
    the full batch size.
    """
    import torch

    terms = _prior_kl_telemetry(data, batch, logits, device)
    if terms is None:
        return None
    # A forced row has a one-element legal simplex, so its KL is identically
    # zero for every model.  Counting those rows in the denominator diluted the
    # configured anchor by roughly the corpus forced fraction (~51.5%) even
    # though they can never contribute an anchor gradient.
    legal = np.asarray(data["legal_action_ids"][batch])
    non_forced = torch.as_tensor(
        np.sum(legal >= 0, axis=1) > 1,
        dtype=torch.bool,
        device=device,
    )
    weights = (terms["has_prior"] & non_forced).to(torch.float32)
    eligible_components = getattr(data, "policy_kl_anchor_component_indices", tuple())
    if bool(getattr(data, "policy_kl_anchor_scope_authenticated", False)):
        component_indices = data.component_indices_for_rows(batch)
        eligible = torch.as_tensor(
            np.isin(component_indices, np.asarray(eligible_components, dtype=np.int64)),
            dtype=torch.bool,
            device=device,
        )
        weights = weights * eligible.to(torch.float32)
    if float(weights.sum().item()) <= 0.0:
        return None
    if direction == "forward":
        values = terms["kl_prior_model"]
    elif direction == "reverse":
        values = terms["kl_model_prior"]
    else:
        raise ValueError(f"unknown policy KL anchor direction {direction!r}")
    weighted_sum, denominator = _weighted_loss_parts(values, weights)
    return _weighted_mean_loss(values, weights), weighted_sum, denominator


def _weighted_mean_loss(values, weights, *, mask=None):
    numerator, denominator = _weighted_loss_parts(values, weights, mask=mask)
    return _weighted_mean_from_parts(numerator, denominator)


def _weighted_mean_from_parts(numerator, denominator):
    """Form one DDP-correct mean after independently accumulated loss parts."""
    import torch

    if torch.is_grad_enabled():
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            # Never fall back to the local denominator after a collective error.
            # That would silently make each rank optimize a different ratio while
            # DDP still averages their gradients -- exactly the biased mean-of-rank-
            # means estimator this collective exists to prevent.
            global_denominator = denominator.detach().clone()
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
            return (
                numerator
                * float(dist.get_world_size())
                / torch.clamp(global_denominator, min=1.0e-6)
            )
    return numerator / torch.clamp(denominator, min=1.0e-6)


def _weighted_loss_parts(values, weights, *, mask=None):
    effective_weights = weights
    if mask is not None:
        effective_weights = effective_weights * mask.to(
            device=weights.device,
            dtype=weights.dtype,
        )
    return (values * effective_weights).sum(), effective_weights.sum()


def _zero_loss_parts(device):
    import torch

    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    return zero, zero


def _metric_from_sum_denominator(weighted_sum: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(weighted_sum) / float(denominator)


def _q_score_loss_parts(
    q_values,
    data: dict,
    batch: np.ndarray,
    weights,
    device,
    *,
    q_skip_teacher_prefixes: tuple[str, ...],
):
    import torch

    scores = np.asarray(data.get("target_scores", np.empty((len(data["action_taken"]), 0)))[batch], dtype=np.float32)
    legal = np.asarray(data["legal_action_ids"][batch], dtype=np.int64)
    if scores.size == 0:
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        return zero, zero, zero
    if "target_scores_mask" in data:
        score_mask = np.asarray(data["target_scores_mask"][batch], dtype=np.bool_)
    else:
        score_mask = np.isfinite(scores)
    finite_np = (legal >= 0) & score_mask & np.isfinite(scores)
    skip_rows = _q_skip_rows(data, batch, q_skip_teacher_prefixes)
    if np.any(skip_rows):
        finite_np[skip_rows] = False
    row_has_scores_np = np.sum(finite_np, axis=1) >= 2
    if not np.any(row_has_scores_np):
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        return zero, zero, zero

    mask = torch.as_tensor(finite_np, dtype=torch.bool, device=device)
    row_has_scores = torch.as_tensor(row_has_scores_np, dtype=torch.float32, device=device)
    targets = torch.as_tensor(
        np.where(finite_np, scores, 0.0),
        dtype=torch.float32,
        device=device,
    )
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    target_mean = (targets * mask.float()).sum(dim=1, keepdim=True) / counts
    centered_targets = torch.where(mask, targets - target_mean, torch.zeros_like(targets))
    target_var = (centered_targets.pow(2) * mask.float()).sum(dim=1, keepdim=True) / counts
    target_std = target_var.sqrt().clamp_min(1.0e-4)
    normalized_targets = centered_targets / target_std

    q_safe = torch.where(mask, q_values, torch.zeros_like(q_values))
    per_row = ((q_safe - normalized_targets).pow(2) * mask.float()).sum(dim=1) / counts.squeeze(1)
    effective_weights = weights * row_has_scores
    loss = _weighted_mean_loss(per_row, effective_weights)
    weighted_sum, denominator = _weighted_loss_parts(per_row, effective_weights)
    return loss, weighted_sum, denominator


def _unflagged_vp_rows(data: dict, values_key: str, flags_key: str) -> np.ndarray:
    values = np.asarray(data.get(values_key, np.zeros((len(data["action_taken"]), 0))), dtype=np.float32)
    flags = np.asarray(
        data.get(flags_key, np.zeros(len(data["action_taken"]), dtype=np.bool_)),
        dtype=np.bool_,
    )
    if values.ndim != 2 or values.shape[0] != len(flags):
        return np.zeros(len(data["action_taken"]), dtype=bool)
    return (np.sum(np.abs(values), axis=1) > 0.0) & ~flags


def _q_score_rows_ge2(
    data: dict,
    batch: np.ndarray,
    *,
    q_skip_teacher_prefixes: tuple[str, ...] = (),
) -> int:
    scores_all = data.get("target_scores")
    if scores_all is None:
        return 0
    scores = np.asarray(scores_all[batch], dtype=np.float32)
    legal = np.asarray(data["legal_action_ids"][batch], dtype=np.int64)
    if scores.shape != legal.shape:
        return 0
    if "target_scores_mask" in data:
        score_mask = np.asarray(data["target_scores_mask"][batch], dtype=np.bool_)
    else:
        score_mask = np.isfinite(scores)
    finite = (legal >= 0) & score_mask & np.isfinite(scores)
    skip_rows = _q_skip_rows(data, batch, q_skip_teacher_prefixes)
    if np.any(skip_rows):
        finite[skip_rows] = False
    return int(np.sum(np.sum(finite, axis=1) >= 2))


def _truncated_vp_margin_outcome(
    data: dict,
    batch: np.ndarray,
    truncated: np.ndarray,
    vps_to_win: int,
    *,
    public_information_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """FIX F3: derive a soft value label for TRUNCATED rows from the VP margin at the
    point of truncation. final_actual_vps/final_public_vps + seat are populated in every
    row regardless of whether the game actually ended -- only the has_final_*_vps flag is
    gated on termination (see gumbel_self_play.py's _game_outcome_fields, which always
    snapshots the live state). Without this, the value head only ever learns from the
    9-19% of games that finish naturally, starving it of signal from the majority.

    margin = clip((my_vp - opponent_vp) / vps_to_win, -1, 1). Since this is always a
    2-seated game, opponent_vp = sum(all 4 PLAYER_NAMES slots) - my_vp (the other two
    slots are unseated and always 0), avoiding any need to know which color is "the
    opponent" explicitly.
    """
    n = len(batch)
    mask = np.zeros(n, dtype=bool)
    outcome = np.zeros(n, dtype=np.float32)
    if "seat" not in data:
        return mask, outcome
    seats = np.asarray(data["seat"][batch], dtype=np.int64)
    rows = np.arange(n)
    # Opponents' actual VP includes hidden victory-point development cards.
    # A public-observation learner must never derive even a low-weight target
    # from that hidden truth.  Its own public score and every opponent's public
    # score are sufficient for the truncation proxy.
    vp_keys = (
        ("final_public_vps",)
        if public_information_only
        else ("final_actual_vps", "final_public_vps")
    )
    for vp_key in vp_keys:
        if vp_key not in data:
            continue
        remaining = truncated & ~mask & (seats >= 0)
        if not remaining.any():
            continue
        vps = np.asarray(data[vp_key][batch], dtype=np.float32)
        valid = remaining & (seats < vps.shape[1])
        if not valid.any():
            continue
        my_vp = vps[rows[valid], seats[valid]]
        opponent_vp = vps[valid].sum(axis=1) - my_vp
        margin = (my_vp - opponent_vp) / max(float(vps_to_win), 1.0)
        outcome[valid] = np.clip(margin, -1.0, 1.0)
        mask[valid] = True
    return mask, outcome


_KNOWN_VALUE_ROOT_BLEND_PHASES = frozenset(
    {
        "BUILD_INITIAL_SETTLEMENT",
        "BUILD_INITIAL_ROAD",
        "DISCARD",
        "MOVE_ROBBER",
        "DECIDE_TRADE",
        "DECIDE_ACCEPTEES",
        "PLAY_TURN",
    }
)


def _parse_value_root_blend_phases(value: str | tuple[str, ...]) -> tuple[str, ...]:
    raw = value if isinstance(value, tuple) else tuple(str(value).split(","))
    phases = tuple(dict.fromkeys(part.strip().upper() for part in raw if part.strip()))
    unknown = sorted(set(phases) - _KNOWN_VALUE_ROOT_BLEND_PHASES)
    if unknown:
        raise SystemExit(
            "unknown --value-root-blend-phases value(s): "
            + ", ".join(unknown)
            + "; known phases are "
            + ", ".join(sorted(_KNOWN_VALUE_ROOT_BLEND_PHASES))
        )
    return phases


def _resolve_value_root_blend_regime(args) -> dict[str, object]:
    """Resolve the root-value target operator, requiring explicit scope.

    ``lambda=1`` stays an exact no-op. A non-unit lambda may either target named
    phases or request the historical global operator explicitly; it can never
    silently fall back to global blending.
    """

    lam = float(getattr(args, "value_target_lambda", 1.0))
    phases = _parse_value_root_blend_phases(
        getattr(args, "value_root_blend_phases", "")
    )
    global_compat = bool(getattr(args, "value_root_blend_global_compat", False))
    if phases and global_compat:
        raise SystemExit(
            "--value-root-blend-phases and --value-root-blend-global-compat are "
            "mutually exclusive"
        )
    if lam != 1.0 and not phases and not global_compat:
        raise SystemExit(
            "--value-target-lambda < 1 requires an explicit target-information "
            "scope: use --value-root-blend-phases or the historical "
            "--value-root-blend-global-compat"
        )
    return {
        "schema_version": "value-root-blend-regime-v1",
        "mode": "phase_gated" if phases else ("global_compat" if global_compat else "disabled"),
        "lambda": lam,
        "phases": list(phases),
    }


def _value_root_phase_mask(
    data: dict,
    batch: np.ndarray,
    phases: tuple[str, ...],
    *,
    global_compat: bool,
) -> np.ndarray:
    if global_compat:
        return np.ones(len(batch), dtype=np.bool_)
    if not phases or "phase" not in data:
        return np.zeros(len(batch), dtype=np.bool_)
    observed = np.char.upper(np.asarray(data["phase"][batch]).astype(str))
    return np.isin(observed, np.asarray(phases, dtype=str))


def _value_root_blend_mask(
    data: dict,
    batch: np.ndarray,
    device,
    root_value_mask,
    value_has_outcome,
    truncated_mask,
    *,
    phases: tuple[str, ...] = (),
    global_compat: bool = False,
):
    import torch

    phase_mask = torch.as_tensor(
        _value_root_phase_mask(data, batch, phases, global_compat=global_compat),
        dtype=torch.bool,
        device=device,
    )
    # Truncated games and missing/invalid search roots retain pure z. This also
    # keeps scalar and categorical objectives on exactly the same row operator.
    return phase_mask & root_value_mask & value_has_outcome & ~truncated_mask


def _audit_value_root_blend_corpus(
    data: dict,
    value_sample_weights: np.ndarray,
    *,
    regime: dict[str, object],
    indices: np.ndarray | None = None,
) -> dict[str, object]:
    """Fail-closed corpus realization and durable target-operator telemetry."""

    total_rows = len(data["action_taken"])
    rows = (
        np.arange(total_rows, dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    n = len(rows)
    lam = float(regime["lambda"])
    phases = tuple(str(value) for value in regime["phases"])
    mode = str(regime["mode"])
    report: dict[str, object] = {
        **regime,
        "rows": n,
        "eligible_rows": 0,
        "blended_rows": 0,
        "eligible_weighted_mass": 0.0,
        "blended_weighted_mass": 0.0,
        "per_phase": {},
        "mean_abs_root_minus_z": None,
        "root_value_finite": True,
        "root_value_in_range": True,
        "target_information_regime_counts": {},
        "public_target_information_only": None,
    }
    if "target_information_regime" in data:
        information_column = data["target_information_regime"]
        if bool(getattr(information_column, "supports_value_counts", False)) or isinstance(
            information_column, _MemmapCategoricalColumn
        ):
            report["target_information_regime_counts"] = (
                information_column.value_counts(rows)
            )
        else:
            values, counts = np.unique(
                np.asarray(information_column[rows]).astype(str),
                return_counts=True,
            )
            report["target_information_regime_counts"] = {
                str(value): int(count)
                for value, count in zip(values, counts, strict=True)
            }
        report["public_target_information_only"] = set(
            report["target_information_regime_counts"]
        ) == {"public_conservation_pimc_v1"}
    if lam == 1.0:
        return report
    if "root_value" not in data:
        raise SystemExit("requested root-value blend but corpus has no root_value column")
    root = np.asarray(data["root_value"][rows], dtype=np.float32).reshape(n)
    mask = np.asarray(
        (
            data["root_value_mask"][rows]
            if "root_value_mask" in data
            else np.isfinite(root)
        ),
        dtype=np.bool_,
    ).reshape(n)
    masked_finite = np.isfinite(root[mask])
    masked_range = np.abs(root[mask]) <= 1.0
    report["root_value_finite"] = bool(np.all(masked_finite))
    report["root_value_in_range"] = bool(np.all(masked_range))
    if not report["root_value_finite"] or not report["root_value_in_range"]:
        raise SystemExit(
            "requested root-value blend contains masked non-finite or out-of-range "
            "root_value values; expected finite [-1, 1]"
        )
    phase_mask = _value_root_phase_mask(
        data, rows, phases, global_compat=(mode == "global_compat")
    )
    truncated = np.asarray(
        data["truncated"][rows] if "truncated" in data else np.zeros(n),
        dtype=np.bool_,
    )
    # Outcome existence is deliberately conservative and public: winner labels
    # are the realized z source. Rows without one retain pure z/no blend.
    winner = np.asarray(
        data["winner"][rows] if "winner" in data else np.full(n, "")
    ).astype(str)
    has_outcome = winner != ""
    eligible = phase_mask & mask & has_outcome & ~truncated
    weights = np.asarray(value_sample_weights[rows], dtype=np.float64)
    report["eligible_rows"] = int(np.sum(eligible))
    report["blended_rows"] = int(np.sum(eligible))
    report["eligible_weighted_mass"] = float(np.sum(weights[eligible]))
    report["blended_weighted_mass"] = float(np.sum(weights[eligible]))
    observed_phases = np.asarray(
        data["phase"][rows] if "phase" in data else np.full(n, "")
    ).astype(str)
    report["per_phase"] = {
        phase: {
            "eligible_rows": int(np.sum(eligible & (np.char.upper(observed_phases) == phase))),
            "weighted_mass": float(
                np.sum(weights[eligible & (np.char.upper(observed_phases) == phase)])
            ),
        }
        for phase in (phases or tuple(sorted(set(np.char.upper(observed_phases[eligible])))))
    }
    if not np.any(eligible) or float(report["blended_weighted_mass"]) <= 0.0:
        raise SystemExit(
            "requested root-value blend realized zero eligible rows or weighted mass "
            "after phase, outcome, truncation, root-value, and learner-weight masks"
        )
    if "player" in data:
        player = np.asarray(data["player"][rows]).astype(str)
        z = np.where(player == winner, 1.0, -1.0)
        report["mean_abs_root_minus_z"] = float(
            np.average(np.abs(root[eligible] - z[eligible]), weights=weights[eligible])
        )
    return report


def _root_value_targets(data: dict, batch: np.ndarray, device):
    """Read the stored search root value (`root_value` + `root_value_mask`) for
    the CAT-39 value-target lambda blend (MuZero/ReZero, arXiv:2404.16364).

    Returns (root_value[B], root_value_mask[B]) as torch tensors, or (None, None)
    when the shard has no root_value column (every current shard) so the blend is
    a pure no-op and inert -- a gen-1-onward lever. root_value is expected in the
    same [-1, 1] value scale as the outcome target; rows without a stored value
    are masked (NaNs treated as unset).
    """
    import torch

    if "root_value" not in data:
        return None, None
    n = len(batch)
    raw = np.asarray(data["root_value"][batch], dtype=np.float32).reshape(n)
    if "root_value_mask" in data:
        mask = np.asarray(data["root_value_mask"][batch], dtype=np.bool_).reshape(n)
    else:
        mask = np.isfinite(raw)
    mask = mask & np.isfinite(raw)
    values = np.where(mask, raw, 0.0).astype(np.float32)
    return (
        torch.as_tensor(values, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


def _hl_gauss_value_targets(
    targets,
    bins: int,
    *,
    sigma_ratio: float = 0.75,
    truncated=None,
    add_truncation_class: bool = True,
):
    """HL-Gauss projection (Farebrother et al. 2024, arXiv:2403.03950 "Stop
    Regressing") of scalar win-loss targets in [-1, 1] onto a uniform categorical
    support of `bins` atoms.

    Each scalar target y is smeared as a Gaussian of std ``sigma = sigma_ratio *
    bin_width`` and integrated over atom-centred cells via the Gaussian CDF
    (erf); the outer cells extend to +/-inf so tail mass beyond the support is
    captured rather than clipped. This is the regression-as-classification target
    that beats both plain two-hot (which underperforms MSE) and scalar MSE,
    especially under stochastic dynamics.

    Returns a row-stochastic ``[B, bins (+1 when add_truncation_class)]`` tensor.
    When ``truncated`` is given, those rows are routed ENTIRELY to the extra
    truncation class (one-hot) with zero mass on the win-loss bins -- the CAT-39
    R9 support (win/loss + truncation ONLY; VP-margin routes to a separate aux
    head), never a margin bump on the continuous axis. The support expectation
    over the win-loss atoms recovers the (clamped) scalar target to within the
    bin resolution, which is what the scalar readout consumes downstream.
    """
    import torch

    n = int(targets.shape[0])
    device = targets.device
    n_out = int(bins) + (1 if add_truncation_class else 0)
    out = torch.zeros((n, n_out), dtype=torch.float32, device=device)
    centers = torch.linspace(-1.0, 1.0, int(bins), device=device)
    bin_width = 2.0 / float(int(bins) - 1)
    sigma = max(float(sigma_ratio), 1.0e-6) * bin_width
    lower = centers - bin_width / 2.0
    upper = centers + bin_width / 2.0
    lower = lower.clone()
    upper = upper.clone()
    lower[0] = float("-inf")
    upper[-1] = float("inf")
    y = targets.detach().float().clamp(-1.0, 1.0).unsqueeze(-1)
    inv = 1.0 / (sigma * math.sqrt(2.0))
    cdf_hi = 0.5 * (1.0 + torch.erf((upper.unsqueeze(0) - y) * inv))
    cdf_lo = 0.5 * (1.0 + torch.erf((lower.unsqueeze(0) - y) * inv))
    probs = (cdf_hi - cdf_lo).clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    out[:, : int(bins)] = probs
    if truncated is not None and add_truncation_class:
        tmask = truncated.to(device=device, dtype=torch.bool)
        if bool(tmask.any().item()):
            out[tmask] = 0.0
            out[tmask, int(bins)] = 1.0
    return out


def _value_targets(
    data: dict,
    batch: np.ndarray,
    device,
    vps_to_win: int,
    *,
    truncated_vp_margin_value_weight: float = 0.0,
    public_information_only: bool | None = None,
):
    import torch

    if "winner" not in data or "player" not in data:
        return None, None, None, None, None, None, None
    winners = np.asarray(data["winner"][batch]).astype(str)
    players = np.asarray(data["player"][batch]).astype(str)
    truncated = _batch_array_or_fill(
        data,
        "truncated",
        batch,
        False,
        dtype=np.bool_,
    )
    has_outcome_np = (winners != "") & ~truncated
    outcome = np.zeros(len(batch), dtype=np.float32)
    outcome[has_outcome_np & (winners == players)] = 1.0
    outcome[has_outcome_np & (winners != players)] = -1.0
    # FIX F3: value_outcome/value_has_outcome_np/outcome_confidence are a SEPARATE,
    # value-loss-only view -- outcome/has_outcome_np above stay exactly as they were
    # (some callers, e.g. _train_xdim_batch, also feed them into
    # _advantage_reweighted_policy_weights for POLICY advantage weighting, which this
    # fix must not touch; team-lead scoped F3 to the value head only).
    value_outcome = outcome.copy()
    value_has_outcome_np = has_outcome_np.copy()
    outcome_confidence = np.where(has_outcome_np, 1.0, 0.0).astype(np.float32)
    if float(truncated_vp_margin_value_weight) > 0.0 and truncated.any():
        if public_information_only is None:
            public_information_only = _MASK_HIDDEN_INFO_PLAYER_TOKENS
        soft_mask, soft_outcome = _truncated_vp_margin_outcome(
            data,
            batch,
            truncated,
            vps_to_win,
            public_information_only=bool(public_information_only),
        )
        fill = soft_mask & ~has_outcome_np
        value_outcome[fill] = soft_outcome[fill]
        value_has_outcome_np = value_has_outcome_np | fill
        outcome_confidence[fill] = float(truncated_vp_margin_value_weight)
    vp_target = np.zeros(len(batch), dtype=np.float32)
    has_vp_np = np.zeros(len(batch), dtype=bool)
    if "seat" in data and "final_actual_vps" in data:
        seats = np.asarray(data["seat"][batch], dtype=np.int64)
        rows = np.arange(len(batch))
        actual_vps = np.asarray(data["final_actual_vps"][batch], dtype=np.float32)
        has_actual = _batch_array_or_fill(
            data,
            "has_final_actual_vps",
            batch,
            False,
            dtype=np.bool_,
        )
        valid_actual = (
            (seats >= 0)
            & (seats < actual_vps.shape[1])
            & ~truncated
            & has_actual
        )
        vp_target[valid_actual] = actual_vps[rows[valid_actual], seats[valid_actual]]
        has_vp_np[valid_actual] = True
    if "seat" in data and "final_public_vps" in data:
        seats = np.asarray(data["seat"][batch], dtype=np.int64)
        rows = np.arange(len(batch))
        public_vps = np.asarray(data["final_public_vps"][batch], dtype=np.float32)
        has_public = _batch_array_or_fill(
            data,
            "has_final_public_vps",
            batch,
            False,
            dtype=np.bool_,
        )
        valid_public = (
            ~has_vp_np
            & (seats >= 0)
            & (seats < public_vps.shape[1])
            & ~truncated
            & has_public
        )
        vp_target[valid_public] = public_vps[rows[valid_public], seats[valid_public]]
        has_vp_np[valid_public] = True
    vp_target[has_vp_np] /= max(float(vps_to_win), 1.0)
    return (
        torch.as_tensor(outcome, dtype=torch.float32, device=device),
        torch.as_tensor(vp_target, dtype=torch.float32, device=device),
        torch.as_tensor(has_outcome_np, dtype=torch.bool, device=device),
        torch.as_tensor(has_vp_np, dtype=torch.bool, device=device),
        torch.as_tensor(value_outcome, dtype=torch.float32, device=device),
        torch.as_tensor(value_has_outcome_np, dtype=torch.bool, device=device),
        torch.as_tensor(outcome_confidence, dtype=torch.float32, device=device),
    )


def _masked_metric_mean(values, mask):
    import torch

    mask = mask.to(device=values.device, dtype=torch.bool)
    if not torch.any(mask):
        return torch.tensor(0.0, dtype=values.dtype, device=values.device)
    return values[mask].mean()


def _topk_legal_accuracy(logits, target, *, k: int, mask=None):
    import torch

    k = min(int(k), int(logits.shape[-1]))
    topk = torch.topk(logits, k=k, dim=-1).indices
    hits = (topk == target.unsqueeze(-1)).any(dim=-1).float()
    if mask is not None:
        return _masked_metric_mean(hits, mask)
    return hits.mean()


def _topk_full_accuracy(masked_logits, actions, *, k: int, mask=None):
    import torch

    k = min(int(k), int(masked_logits.shape[-1]))
    topk = torch.topk(masked_logits, k=k, dim=-1).indices
    hits = (topk == actions.unsqueeze(-1)).any(dim=-1).float()
    if mask is not None:
        return _masked_metric_mean(hits, mask)
    return hits.mean()


def _field_stats(
    data: dict,
    batch: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    logits: np.ndarray,
    *,
    field: str,
) -> dict[str, dict[str, int]]:
    if field in data:
        values = np.asarray(data[field])[batch].astype(str)
    else:
        # The old ``dict.get`` default constructed a full-corpus Python string
        # list eagerly even when ``field`` existed.  Validation calls this for
        # every batch, turning a 12M-row corpus into minutes of CPU allocation
        # per batch.  Missing fields need only one constant per requested row.
        values = np.full(len(batch), "unknown", dtype="<U7")
    top3 = _numpy_topk_contains(logits, targets, k=3)
    stats = _empty_phase_stats()
    for value, pred, target, in_top3 in zip(values, predictions, targets, top3, strict=False):
        key = str(value or "unknown")
        row = stats.setdefault(key, {"count": 0, "top1": 0, "top3": 0})
        row["count"] += 1
        row["top1"] += int(int(pred) == int(target))
        row["top3"] += int(bool(in_top3))
    return stats


def _field_stats_unforced(
    data: dict,
    batch: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    logits: np.ndarray,
    *,
    field: str,
) -> dict[str, dict[str, int]]:
    legal_counts = np.sum(data["legal_action_ids"][batch] >= 0, axis=1)
    keep = legal_counts > 1
    if not np.any(keep):
        return {}
    return _field_stats(
        data,
        batch[keep],
        predictions[keep],
        targets[keep],
        logits[keep],
        field=field,
    )


def _numpy_topk_contains(logits: np.ndarray, targets: np.ndarray, *, k: int) -> np.ndarray:
    if logits.shape[1] == 0:
        return np.zeros(len(targets), dtype=bool)
    k = min(int(k), int(logits.shape[1]))
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return np.any(topk == targets[:, None], axis=1)


def _empty_phase_stats() -> dict[str, dict[str, int]]:
    return {}


def _merge_phase_stats(
    total: dict[str, dict[str, int]],
    update: dict[str, dict[str, int]],
) -> None:
    for phase, row in update.items():
        target = total.setdefault(str(phase), {"count": 0, "top1": 0, "top3": 0})
        target["count"] += int(row.get("count", 0))
        target["top1"] += int(row.get("top1", 0))
        target["top3"] += int(row.get("top3", 0))


def _reduce_nested_count_stats(
    stats: dict[str, dict[str, int]],
    ddp: dict[str, int | bool],
) -> dict[str, dict[str, int]]:
    if not ddp["enabled"]:
        return stats
    import torch.distributed as dist

    gathered: list[dict[str, dict[str, int]] | None] = [None] * int(ddp["world_size"])
    dist.all_gather_object(gathered, stats)
    merged: dict[str, dict[str, int]] = {}
    for item in gathered:
        if item:
            _merge_phase_stats(merged, item)
    return merged


def _finalize_phase_stats(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    result = {}
    for phase, row in sorted(stats.items()):
        count = int(row.get("count", 0))
        result[phase] = {
            "count": count,
            "top1_accuracy": float(row.get("top1", 0)) / max(count, 1),
            "top3_accuracy": float(row.get("top3", 0)) / max(count, 1),
        }
    return result


def _params(policy):
    for name in ("model", "actor", "action_encoder", "action_id_embedding", "action_bias"):
        module = getattr(policy, name, None)
        if module is not None:
            yield from module.parameters()


def _optimizer_observability_module_name(parameter_name: str) -> str:
    """Return a stable, compact top-level module label for optimizer telemetry."""
    name = str(parameter_name)
    for prefix in ("module.", "_fsdp_wrapped_module."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    return name.split(".", 1)[0] or "<root>"


def _objective_gradient_module_name(parameter_name: str) -> str:
    """Keep transformer block indices in objective-interference telemetry."""
    name = str(parameter_name)
    for prefix in ("module.", "_fsdp_wrapped_module."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
        return f"blocks.{parts[1]}"
    return parts[0] or "<root>"


def _shared_trunk_named_parameters(policy) -> list[tuple[str, object]]:
    """Select logical EntityGraph trunk parameters without either task head."""
    model = policy.model
    module = getattr(model, "module", model)
    trunk_names = ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["trunk"]
    trunk_prefixes = tuple(f"{name}." for name in trunk_names)
    selected = []
    for name, parameter in module.named_parameters():
        normalized = str(name)
        while normalized.startswith("_fsdp_wrapped_module."):
            normalized = normalized[len("_fsdp_wrapped_module.") :]
        if parameter.requires_grad and (
            normalized in trunk_names or normalized.startswith(trunk_prefixes)
        ):
            selected.append((normalized, parameter))
    return selected


def _objective_gradient_interference(
    policy,
    *,
    policy_objective,
    value_objective,
) -> dict[str, object]:
    """Measure weighted policy/value gradient interaction in the shared trunk.

    ``autograd.grad`` leaves ``Parameter.grad`` and the optimizer trajectory
    untouched. Objectives already include configured loss coefficients. This
    therefore measures what reaches shared layers, unlike ``--value-lr-mult``,
    which only changes named value-head parameter groups.

    FSDP flattens logical parameters, so it cannot provide a faithful block-level
    decomposition here. Report that limitation instead of plausible false data;
    run this high-information diagnostic on one GPU or DDP. DDP results are
    explicitly rank-local because ranks see different microbatches.
    """
    import torch

    model = policy.model
    is_fsdp = "FullyShardedDataParallel" in type(model).__name__ or (
        callable(getattr(model, "clip_grad_norm_", None))
        and not isinstance(model, torch.nn.parallel.DistributedDataParallel)
    )
    if is_fsdp:
        return {
            "available": False,
            "reason": "fsdp_logical_parameters_are_flattened",
            "recommended_probe": "single_gpu_or_ddp",
        }
    named = _shared_trunk_named_parameters(policy)
    if not named:
        return {"available": False, "reason": "no_trainable_shared_trunk_parameters"}
    if any(
        not getattr(objective, "requires_grad", False)
        for objective in (policy_objective, value_objective)
    ):
        return {"available": False, "reason": "inactive_policy_or_value_objective"}
    parameters = [parameter for _, parameter in named]
    try:
        policy_grads = torch.autograd.grad(
            policy_objective, parameters, retain_graph=True, allow_unused=True
        )
        value_grads = torch.autograd.grad(
            value_objective, parameters, retain_graph=True, allow_unused=True
        )
    except RuntimeError as exc:
        return {
            "available": False,
            "reason": "autograd_probe_failed",
            "detail": str(exc)[:240],
        }

    zero = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    policy_sq = zero.clone()
    value_sq = zero.clone()
    dot = zero.clone()
    conflict_coordinates = zero.clone()
    joint_coordinates = zero.clone()
    by_module: dict[str, dict[str, object]] = {}
    for (name, _), policy_grad, value_grad in zip(named, policy_grads, value_grads):
        if policy_grad is None and value_grad is None:
            continue
        if policy_grad is None:
            policy_grad = torch.zeros_like(value_grad)
        if value_grad is None:
            value_grad = torch.zeros_like(policy_grad)
        pg = policy_grad.detach().double()
        vg = value_grad.detach().double()
        p_sq = pg.square().sum()
        v_sq = vg.square().sum()
        pv = (pg * vg).sum()
        jointly_nonzero = (pg != 0) & (vg != 0)
        policy_sq += p_sq
        value_sq += v_sq
        dot += pv
        joint_coordinates += jointly_nonzero.sum(dtype=torch.float64)
        conflict_coordinates += (
            jointly_nonzero & ((pg * vg) < 0)
        ).sum(dtype=torch.float64)
        group = _objective_gradient_module_name(name)
        row = by_module.setdefault(
            group,
            {"policy_sq": zero.clone(), "value_sq": zero.clone(), "dot": zero.clone()},
        )
        row["policy_sq"] += p_sq
        row["value_sq"] += v_sq
        row["dot"] += pv

    epsilon = torch.finfo(torch.float64).eps
    policy_norm = torch.sqrt(policy_sq)
    value_norm = torch.sqrt(value_sq)
    denominator = policy_norm * value_norm

    def _float(value) -> float:
        return float(value.detach().cpu().item())

    modules = {}
    for group, row in sorted(by_module.items()):
        p_norm = torch.sqrt(row["policy_sq"])
        v_norm = torch.sqrt(row["value_sq"])
        denom = p_norm * v_norm
        modules[group] = {
            "policy_grad_norm": _float(p_norm),
            "value_grad_norm": _float(v_norm),
            "cosine": _float(row["dot"] / denom.clamp_min(epsilon))
            if _float(denom) > 0.0
            else None,
        }
    combined_sq = policy_sq + value_sq + 2.0 * dot
    return {
        "available": True,
        "scope": (
            "rank_local_microbatch"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else "single_process_microbatch"
        ),
        "value_lr_mult_scales_shared_trunk": False,
        "policy_objective": _float(policy_objective.detach()),
        "value_objective": _float(value_objective.detach()),
        "policy_trunk_grad_norm": _float(policy_norm),
        "value_trunk_grad_norm": _float(value_norm),
        "value_to_policy_grad_norm_ratio": (
            _float(value_norm / policy_norm.clamp_min(epsilon))
            if _float(policy_norm) > 0.0
            else None
        ),
        "trunk_gradient_cosine": (
            _float(dot / denominator.clamp_min(epsilon))
            if _float(denominator) > 0.0
            else None
        ),
        "opposing_coordinate_fraction": (
            _float(conflict_coordinates / joint_coordinates)
            if _float(joint_coordinates) > 0.0
            else None
        ),
        "combined_trunk_grad_norm": _float(torch.sqrt(combined_sq.clamp_min(0.0))),
        "modules": modules,
    }


def _norms_from_squared_sums(squared_sums: dict[str, object]) -> dict[str, float]:
    import torch

    return {
        name: float(torch.sqrt(value).detach().cpu().item())
        for name, value in sorted(squared_sums.items())
    }


def _capture_optimizer_observability(policy) -> dict:
    """Capture pre-clip module gradient norms and pre-step parameter bytes.

    This helper is called only at the existing opt-in train-diagnostics cadence.
    The normal/default path therefore performs no parameter clones, device
    synchronizations, or extra reductions. Under DDP the gradients and parameters
    are already replicated and synchronized. Under FSDP these per-module values are
    explicitly rank-local shard diagnostics; the total norm returned by
    ``_clip_grad_norm`` remains FSDP's authoritative global norm.
    """
    model = policy.model
    module = getattr(model, "module", model)
    named_parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    snapshots = [
        (
            _optimizer_observability_module_name(name),
            parameter,
            parameter.detach().clone(),
        )
        for name, parameter in named_parameters
    ]
    grad_squared_sums: dict[str, object] = {}
    for name, parameter in named_parameters:
        group = _optimizer_observability_module_name(name)
        squared = parameter.grad.detach().float().square().sum()
        grad_squared_sums[group] = grad_squared_sums.get(group, 0.0) + squared
    import torch

    is_fsdp = callable(getattr(model, "clip_grad_norm_", None)) and not isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    )
    return {
        "snapshots": snapshots,
        "module_pre_clip_grad_norms": _norms_from_squared_sums(
            grad_squared_sums
        ),
        "module_norm_scope": "rank_local_shard" if is_fsdp else "global_replicated",
    }


def _finish_optimizer_observability(
    policy,
    state: dict,
    *,
    pre_clip_total_grad_norm,
    max_grad_norm: float,
) -> dict:
    """Measure the actual optimizer update after ``optimizer.step()``."""
    del policy  # The captured Parameter objects remain live across optimizer.step().
    delta_squared_sums: dict[str, object] = {}
    for group, parameter, before in state["snapshots"]:
        squared = (parameter.detach().float() - before.float()).square().sum()
        delta_squared_sums[group] = delta_squared_sums.get(group, 0.0) + squared
    if hasattr(pre_clip_total_grad_norm, "detach"):
        total_norm = float(pre_clip_total_grad_norm.detach().cpu().item())
    else:
        total_norm = float(pre_clip_total_grad_norm)
    return {
        "pre_clip_total_grad_norm": total_norm,
        "max_grad_norm": float(max_grad_norm),
        "clipped": bool(not math.isfinite(total_norm) or total_norm > max_grad_norm),
        "module_pre_clip_grad_norms": state["module_pre_clip_grad_norms"],
        "module_parameter_delta_norms": _norms_from_squared_sums(
            delta_squared_sums
        ),
        "module_norm_scope": state["module_norm_scope"],
    }


def _clip_grad_norm(policy, max_norm: float = 1.0):
    """Clip the gradient norm of ``policy.model``, correct under DDP, FSDP, and
    single-GPU. FSDP shards parameters, so ``torch.nn.utils.clip_grad_norm_`` over
    local shards would compute the wrong global norm -- FSDP exposes its own
    collective ``clip_grad_norm_`` for this. Plain modules and DDP-wrapped modules
    have no such method, so they take the standard path (byte-identical to the
    pre-C1 ``torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)``)."""
    import torch

    model = policy.model
    fsdp_clip = getattr(model, "clip_grad_norm_", None)
    if callable(fsdp_clip) and not isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        # FSDP ignores 0-dim params (e.g. logit_scale); FSDP's collective clip
        # never sees them and never all-reduced their grads. Average those grads
        # across ranks here so the replicated scalar stays identical everywhere.
        import torch.distributed as dist

        for param in getattr(policy, "_fsdp_ignored_params", []) or []:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        return fsdp_clip(max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _torch_ppo_masked_logits(logits, valid_actions, action_size):
    from catan_zero.rl.torch_ppo import _masked_logits

    return _masked_logits(logits, valid_actions, action_size)


def build_sample_weights(
    data: dict,
    *,
    teacher_weights: dict[str, float],
    phase_weights: dict[str, float],
    forced_action_weight: float,
    winner_sample_weight: float,
    loser_sample_weight: float,
    vp_margin_weight: float,
    vps_to_win: int,
    per_game_policy_weight: bool = False,
    per_game_policy_weight_mode: str = "equal",
) -> np.ndarray:
    n = len(data["action_taken"])
    weights = np.ones(n, dtype=np.float32)
    if teacher_weights and "teacher_name" in data:
        teachers = np.asarray(data["teacher_name"]).astype(str)
        for teacher, weight in teacher_weights.items():
            weights[teachers == teacher] *= float(weight)
    if phase_weights and "phase" in data:
        phases = np.asarray(data["phase"]).astype(str)
        for phase, weight in phase_weights.items():
            weights[phases == phase] *= float(weight)
    if forced_action_weight != 1.0:
        legal_column = data["legal_action_ids"]
        legal_counts = (
            legal_column.row_counts()
            if isinstance(legal_column, _MemmapRaggedColumn)
            else np.sum(np.asarray(legal_column) >= 0, axis=1)
        )
        weights[legal_counts == 1] *= float(forced_action_weight)
    if (
        (float(winner_sample_weight) != 1.0 or float(loser_sample_weight) != 1.0)
        and "winner" in data
        and "player" in data
    ):
        winners = np.asarray(data["winner"]).astype(str)
        players = np.asarray(data["player"]).astype(str)
        truncated = np.asarray(
            data.get("truncated", np.zeros(n, dtype=np.bool_)),
            dtype=np.bool_,
        )
        has_winner = (winners != "") & ~truncated
        weights[has_winner & (winners == players)] *= float(winner_sample_weight)
        weights[has_winner & (winners != players)] *= float(loser_sample_weight)
    if float(vp_margin_weight) != 0.0 and "seat" in data and "final_actual_vps" in data:
        rows = np.arange(n)
        seats = np.asarray(data["seat"], dtype=np.int64)
        truncated = np.asarray(
            data.get("truncated", np.zeros(n, dtype=np.bool_)),
            dtype=np.bool_,
        )
        vps = np.zeros((n, 4), dtype=np.float32)
        has_final_vps = np.zeros(n, dtype=np.bool_)
        actual = np.asarray(data["final_actual_vps"], dtype=np.float32)
        if actual.ndim == 2 and actual.shape[0] == n:
            vps[:, : actual.shape[1]] = actual[:, : vps.shape[1]]
            has_actual = np.asarray(
                data.get("has_final_actual_vps", np.zeros(n, dtype=np.bool_)),
                dtype=np.bool_,
            )
            has_final_vps |= has_actual
        valid = (seats >= 0) & (seats < vps.shape[1]) & ~truncated & has_final_vps
        own = np.zeros(n, dtype=np.float32)
        own[valid] = vps[rows[valid], seats[valid]]
        masked = vps.copy()
        for idx, seat in enumerate(seats):
            if 0 <= int(seat) < masked.shape[1]:
                masked[idx, int(seat)] = -1.0
        best_opp = np.max(masked, axis=1)
        margin = (own - best_opp) / max(float(vps_to_win), 1.0)
        multiplier = np.ones(n, dtype=np.float32)
        multiplier[valid] = np.maximum(
            0.1,
            1.0 + float(vp_margin_weight) * margin[valid],
        )
        weights *= multiplier
    mean = float(np.mean(weights)) if len(weights) else 1.0
    if "policy_weight_multiplier" in data:
        weights *= np.asarray(data["policy_weight_multiplier"], dtype=np.float32)
        mean = float(np.mean(weights)) if len(weights) else 1.0
    if per_game_policy_weight and len(weights):
        # Only rows that already carry policy loss are redistributed. In
        # particular, fast/filtered rows with policy_weight_multiplier=0 stay
        # exactly zero and zero-active games acquire no synthetic policy mass.
        if per_game_policy_weight_mode not in {"equal", "sqrt"}:
            raise ValueError(
                f"unknown per_game_policy_weight_mode {per_game_policy_weight_mode!r}"
            )
        positive = np.where(weights > 0.0, weights, 0.0).astype(np.float32, copy=False)
        weights = _normalize_weights_per_game(
            data, positive, mode=per_game_policy_weight_mode
        )
        mean = float(np.mean(weights)) if len(weights) else 1.0
    if mean > 0:
        weights = weights / mean
    return weights.astype(np.float32, copy=False)


def _apply_authenticated_policy_distillation_scope(
    data, weights: np.ndarray
) -> np.ndarray:
    """Zero policy CE outside an authenticated composite component scope.

    Value weights are intentionally untouched. Stored-prior KL has its own
    independently authenticated component scope, so a replay component can
    rehearse outcomes and optionally anchor incumbent behavior without
    distilling an obsolete search teacher.
    """

    if not bool(getattr(data, "policy_distillation_scope_authenticated", False)):
        return weights
    component_ids = tuple(getattr(data, "component_ids", tuple()))
    eligible = tuple(
        int(value)
        for value in getattr(data, "policy_distillation_component_indices", tuple())
    )
    if (
        not component_ids
        or not eligible
        or any(value < 0 or value >= len(component_ids) for value in eligible)
        or len(set(eligible)) != len(eligible)
    ):
        raise SystemExit("authenticated policy distillation component scope is invalid")
    rows = np.arange(len(weights), dtype=np.int64)
    component_indices = data.component_indices_for_rows(rows)
    keep = np.isin(component_indices, np.asarray(eligible, dtype=np.int64))
    scoped = np.asarray(weights, dtype=np.float32).copy()
    scoped[~keep] = 0.0
    if not np.any(scoped > 0.0):
        raise SystemExit("authenticated policy distillation scope has no positive policy rows")
    return scoped


def _policy_distillation_scope_report(data, weights: np.ndarray) -> dict[str, object] | None:
    if not bool(getattr(data, "policy_distillation_scope_authenticated", False)):
        return None
    component_ids = tuple(data.component_ids)
    eligible = set(data.policy_distillation_component_indices)
    offsets = np.asarray(data.component_offsets, dtype=np.int64)
    components = {}
    for index, component_id in enumerate(component_ids):
        part = np.asarray(weights[offsets[index] : offsets[index + 1]], dtype=np.float64)
        components[str(component_id)] = {
            "component_index": int(index),
            "policy_distillation_enabled": bool(index in eligible),
            "rows": int(part.size),
            "positive_policy_rows": int(np.count_nonzero(part > 0.0)),
            "policy_weight_sum": float(part.sum()),
        }
    return {
        "schema_version": "component-policy-distillation-scope-v1",
        "component_ids": [component_ids[index] for index in sorted(eligible)],
        "components": components,
    }


def build_value_sample_weights(
    data: dict,
    *,
    phase_weights: dict[str, float] | None = None,
    forced_row_value_weight: float = 1.0,
    per_game_value_weight: bool = False,
    per_game_value_weight_mode: str = "equal",
) -> np.ndarray:
    """Keep value targets unbiased by filtered-BC policy weighting.

    Winner/loser filtering is useful for action imitation because we prefer to
    copy decisions from stronger trajectories. The value head needs both sides
    of a completed game at full weight, otherwise PPO starts from an optimistic
    baseline and advantages become high-variance.

    FIX A5: ``phase_weights`` (e.g. ``robber=8.0,initial_build=5.0``) previously only reached
    the POLICY loss via ``build_sample_weights`` -- the value head had no way to be
    phase-weighted at all, so any "phase-repair" pass could never touch the value head by
    construction. This mirrors ``build_sample_weights``'s phase-weight application (per-phase
    multiplier, then renormalize to mean 1) so the value head can be repaired the same way.

    CAT-60: ``forced_row_value_weight`` and ``per_game_value_weight`` address "16k games = 16k
    independent outcomes, not 3.6M labels" -- naive per-row value MSE overweights games with
    many recorded decisions relative to short games, and wastes weight on near-zero-information
    forced-decision rows (states with exactly one legal action). Combination rule, applied in
    this exact order, since every factor here is multiplicative and the final mean-renormalize
    makes the order of scalar factors irrelevant except for ``per_game_value_weight``, which
    must go LAST:

      1. ``phase_weights`` multiplier (this function, existing).
      2. ``value_weight_multiplier`` (CAT-45's per-row sampling-weight field, already stored on
         the corpus -- existing).
      3. ``forced_row_value_weight`` multiplier on rows with exactly one legal action (new).
      4. ``per_game_value_weight`` normalization (new): divides every row's weight (as
         accumulated by steps 1-3) by the total weight its game already accumulated, so every
         game contributes EXACTLY the same total value-loss mass regardless of (a) its row
         count and (b) how steps 1-3 happen to be distributed within it. This equalizes GAMES
         against each other; it does not undo forced-row downweighting *within* a game -- a
         game that is mostly forced moves still has its low-information rows suppressed by
         step 3, it just doesn't lose overall game-level mass for being long.

    Finally the whole array is renormalized to global mean 1, as with every other weight
    builder in this module (a uniform scalar rescale, so it preserves the equal-per-game-mass
    property from step 4).
    """

    weights = np.ones(len(data["action_taken"]), dtype=np.float32)
    if phase_weights and "phase" in data:
        phases = np.asarray(data["phase"]).astype(str)
        for phase, weight in phase_weights.items():
            weights[phases == phase] *= float(weight)
    if "value_weight_multiplier" in data:
        weights *= np.asarray(data["value_weight_multiplier"], dtype=np.float32)
    if float(forced_row_value_weight) != 1.0 and "legal_action_ids" in data:
        legal_column = data["legal_action_ids"]
        legal_counts = (
            legal_column.row_counts()
            if isinstance(legal_column, _MemmapRaggedColumn)
            else np.sum(np.asarray(legal_column) >= 0, axis=1)
        )
        weights[legal_counts == 1] *= float(forced_row_value_weight)
    if per_game_value_weight and len(weights):
        weights = _normalize_weights_per_game(data, weights, mode=per_game_value_weight_mode)
    mean = float(np.mean(weights)) if len(weights) else 1.0
    if mean > 0.0:
        weights = weights / mean
    return weights.astype(np.float32, copy=False)


def _normalize_weights_per_game(data: dict, weights: np.ndarray, *, mode: str = "equal") -> np.ndarray:
    """Rescale ``weights`` so every game (grouped by ``game_seed``) contributes an equal total.

    Unlike ``split_train_validation_indices`` (which falls back to ``np.arange(n)`` -- one row
    per "game" -- when ``game_seed`` is absent, which is safe there because it only affects
    the train/validation split), a one-row-per-game fallback here would silently collapse
    every row's weight to exactly 1.0 (a one-row group trivially holds 100% of its own mass),
    erasing phase weights, --forced-row-value-weight, and the CAT-45 value_weight_multiplier
    in the process. So a missing ``game_seed`` column is instead a hard no-op: this ticket's
    step 1 requires spot-checking that the column is populated before enabling the flag.
    """

    if "game_seed" not in data:
        return weights
    seeds = np.asarray(data["game_seed"], dtype=np.int64)
    weights64 = weights.astype(np.float64, copy=False)
    unique_seeds, inverse = np.unique(seeds, return_inverse=True)
    game_totals = np.zeros(len(unique_seeds), dtype=np.float64)
    np.add.at(game_totals, inverse, weights64)
    safe_totals = np.where(game_totals > 0.0, game_totals, 1.0)
    if mode == "sqrt":
        # EXP3: divide by sqrt(game_total) -> a game of summed-weight W contributes
        # sqrt(W) total mass (with unit base weights, W == n_value_rows, so ~sqrt(n)),
        # the effective-sample-size correction for n correlated in-game value labels.
        denom = np.sqrt(safe_totals)
    elif mode == "equal":
        denom = safe_totals
    else:
        raise ValueError(f"unknown per_game_value_weight_mode {mode!r}")
    normalized = weights64 / denom[inverse]
    return normalized.astype(np.float32, copy=False)


def _parse_game_seed_ranges(raw: str) -> list[tuple[int, int]]:
    """Parse "start1:end1,start2:end2" (inclusive bounds) for
    --validation-game-seed-ranges. Matches a holdout.json's documented ranges
    (task #65 value-head-repair-v2 protocol)."""
    ranges: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(f"invalid --validation-game-seed-ranges entry {chunk!r}: expected start:end")
        start_str, end_str = chunk.split(":", 1)
        start, end = int(start_str), int(end_str)
        if end < start:
            raise SystemExit(f"invalid --validation-game-seed-ranges entry {chunk!r}: end < start")
        ranges.append((start, end))
    return ranges


def split_train_validation_indices(
    data: dict,
    *,
    validation_fraction: float,
    validation_seed: int,
    validation_max_samples: int,
    validation_game_seed_ranges: list[tuple[int, int]] | None = None,
    validation_game_seeds: np.ndarray | None = None,
    training_excluded_game_seeds: np.ndarray | None = None,
    allow_missing_game_seed: bool = False,
) -> dict[str, np.ndarray]:
    """Split into train/validation indices.

    CAT-52 AUDIT (2026-07-08, see AUDIT.md): every path below is REQUIRED to be
    game-level (grouping/filtering by ``data["game_seed"]``, one value per GAME,
    repeated across every decision row of that game), never row-level -- a
    row-level split is the exact mechanism that caused the round-11 val-leak
    incident (per-round random re-splits leaked ~95% of validation rows,
    because the "split" secretly operated per-ROW instead of per-GAME).

    Both branches below used to silently default a MISSING ``"game_seed"``
    column to ``np.arange(n)`` -- i.e. treat every row as its own distinct
    "game". That default defeats game-level grouping by construction (each
    row becomes a singleton group) and reproduces the round-11 mechanism
    exactly, with no warning. Refuse it by default; ``allow_missing_game_seed``
    is an explicit, loud opt-out for callers that genuinely have no game_seed
    column (documented as a row-level split when used).
    """
    n = int(len(data["action_taken"]))
    all_indices = np.arange(n, dtype=np.int64)
    if "game_seed" not in data and not allow_missing_game_seed:
        raise SystemExit(
            "split_train_validation_indices: data has no 'game_seed' column. "
            "Defaulting to a row-index-as-seed split would silently degrade to a "
            "ROW-LEVEL split (the exact round-11 val-leak mechanism -- see "
            "AUDIT.md), so this is refused by default. Pass "
            "allow_missing_game_seed=True (train_bc.py: "
            "--allow-missing-game-seed-validation-split) only if you have "
            "verified this corpus has no meaningful game grouping and a "
            "row-level split is acceptable for your use case."
        )
    if validation_game_seeds is not None:
        if validation_game_seed_ranges:
            raise SystemExit(
                "choose validation_game_seeds or validation_game_seed_ranges, not both"
            )
        if int(validation_max_samples) != 0:
            raise SystemExit(
                "an exact validation game-seed manifest requires "
                "validation_max_samples=0"
            )
        if "game_seed" not in data:
            raise SystemExit(
                "an exact validation game-seed manifest requires a game_seed column"
            )
        requested = np.asarray(validation_game_seeds, dtype=np.int64).reshape(-1)
        if requested.size == 0 or len(np.unique(requested)) != len(requested):
            raise SystemExit(
                "exact validation game seeds must be non-empty and unique"
            )
        seeds = np.asarray(data["game_seed"], dtype=np.int64)
        present = set(map(int, np.unique(seeds).tolist()))
        missing = set(map(int, requested.tolist())) - present
        if missing:
            raise SystemExit(
                "exact validation game-seed manifest references games absent from "
                f"the corpus: missing={len(missing)}"
            )
        excluded = (
            requested
            if training_excluded_game_seeds is None
            else np.asarray(training_excluded_game_seeds, dtype=np.int64).reshape(-1)
        )
        if excluded.size == 0 or len(np.unique(excluded)) != len(excluded):
            raise SystemExit("training-excluded game seeds must be non-empty and unique")
        excluded_set = set(map(int, excluded.tolist()))
        if not set(map(int, requested.tolist())).issubset(excluded_set):
            raise SystemExit("validation game seeds must be a subset of training exclusions")
        excluded_missing = excluded_set - present
        if excluded_missing:
            raise SystemExit(
                "training-excluded game seeds reference games absent from the corpus: "
                f"missing={len(excluded_missing)}"
            )
        validation_mask = np.isin(seeds, requested)
        validation = all_indices[validation_mask]
        train = all_indices[~np.isin(seeds, excluded)]
        if len(train) == 0:
            raise SystemExit(
                "exact validation game-seed manifest selects the entire corpus"
            )
        return {
            "train": train.astype(np.int64, copy=False),
            "validation": validation.astype(np.int64, copy=False),
        }
    if validation_game_seed_ranges:
        # Explicit, deterministic held-out set (task #65): overrides the random
        # game_seed permutation below entirely -- these EXACT games are what a
        # holdout.json documents for coordination with the separate calibration
        # probe (docs/catan_postrepair_revalidation_protocol_20260704.md Step 1).
        seeds = np.asarray(data.get("game_seed", np.arange(n, dtype=np.int64)), dtype=np.int64)
        validation_mask = np.zeros(n, dtype=bool)
        for start, end in validation_game_seed_ranges:
            validation_mask |= (seeds >= start) & (seeds <= end)
        validation = all_indices[validation_mask]
        train = all_indices[~validation_mask]
        if validation_max_samples > 0 and len(validation) > validation_max_samples:
            rng = np.random.default_rng(validation_seed + 1)
            validation = np.sort(rng.choice(validation, size=validation_max_samples, replace=False))
        return {
            "train": train.astype(np.int64, copy=False),
            "validation": validation.astype(np.int64, copy=False),
        }
    fraction = float(np.clip(validation_fraction, 0.0, 0.9))
    if n == 0 or fraction <= 0.0:
        return {
            "train": all_indices,
            "validation": np.asarray([], dtype=np.int64),
        }
    seeds = np.asarray(data.get("game_seed", np.arange(n, dtype=np.int64)), dtype=np.int64)
    unique_seeds = np.unique(seeds)
    if unique_seeds.size <= 1:
        if n >= 1000:
            raise SystemExit(
                "validation_fraction requires non-degenerate game_seed values for "
                "large teacher datasets; refusing row-level validation split because "
                "it can leak decisions from the same game into train and validation."
            )
        print(
            "WARNING: split_train_validation_indices is falling back to a ROW-LEVEL "
            f"random split ({n} rows, {unique_seeds.size} unique game_seed value(s)) -- "
            "this is the round-11 val-leak mechanism (see AUDIT.md). Only acceptable "
            "for tiny synthetic/smoke corpora where within-game correlation across "
            "train/validation is a non-issue; never for a real training corpus.",
            file=sys.stderr,
        )
        rng = np.random.default_rng(validation_seed)
        shuffled = rng.permutation(all_indices)
        validation_count = max(1, int(round(n * fraction)))
        validation = np.sort(shuffled[:validation_count])
        train = np.sort(shuffled[validation_count:])
    else:
        rng = np.random.default_rng(validation_seed)
        shuffled_seeds = rng.permutation(unique_seeds)
        target_rows = max(1, int(round(n * fraction)))
        selected: list[int] = []
        selected_rows = 0
        seed_counts = {
            int(seed): int(np.sum(seeds == seed))
            for seed in shuffled_seeds
        }
        for seed in shuffled_seeds:
            selected.append(int(seed))
            selected_rows += seed_counts[int(seed)]
            if selected_rows >= target_rows:
                break
        validation_mask = np.isin(seeds, np.asarray(selected, dtype=np.int64))
        validation = all_indices[validation_mask]
        train = all_indices[~validation_mask]
    if validation_max_samples > 0 and len(validation) > validation_max_samples:
        rng = np.random.default_rng(validation_seed + 1)
        validation = np.sort(rng.choice(validation, size=validation_max_samples, replace=False))
    if len(train) == 0:
        train = np.setdiff1d(all_indices, validation[: max(0, len(validation) - 1)], assume_unique=False)
        validation = np.setdiff1d(all_indices, train, assume_unique=False)
    return {
        "train": train.astype(np.int64, copy=False),
        "validation": validation.astype(np.int64, copy=False),
    }


def _distributed_index_slice(indices: np.ndarray, ddp: dict[str, int | bool]) -> np.ndarray:
    if not ddp["enabled"]:
        return indices
    world_size = int(ddp["world_size"])
    rank = int(ddp["rank"])
    return np.asarray(indices, dtype=np.int64)[rank::world_size]


def _set_policy_training(policy, training: bool) -> list[tuple[object, bool]]:
    seen: set[int] = set()
    modes: list[tuple[object, bool]] = []
    for name in ("model", "actor", "action_encoder", "action_id_embedding", "action_bias"):
        module = getattr(policy, name, None)
        if module is None or not hasattr(module, "train") or id(module) in seen:
            continue
        seen.add(id(module))
        modes.append((module, bool(getattr(module, "training", False))))
        module.train(bool(training))
    return modes


def _restore_policy_training(policy, modes: list[tuple[object, bool]]) -> None:
    del policy
    for module, was_training in modes:
        if hasattr(module, "train"):
            module.train(was_training)


def _parse_weight_map(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid --teacher-weights entry: {item}")
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


def _parse_prefixes(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _teacher_name_has_prefix(values: np.ndarray, prefixes: tuple[str, ...]) -> np.ndarray:
    if not prefixes:
        return np.zeros(values.shape, dtype=bool)
    result = np.zeros(values.shape, dtype=bool)
    for prefix in prefixes:
        result |= np.char.startswith(values.astype(str), prefix)
    return result


def _q_skip_rows(data: dict, batch: np.ndarray, prefixes: tuple[str, ...]) -> np.ndarray:
    teacher_names = _batch_array_or_fill(data, "teacher_name", batch, "").astype(str)
    score_sources = _batch_array_or_fill(
        data,
        "target_score_source",
        batch,
        "",
    ).astype(str)
    return _q_skip_rows_for_arrays(teacher_names, score_sources, prefixes)


def _q_skip_rows_for_arrays(
    teacher_names: np.ndarray,
    score_sources: np.ndarray,
    prefixes: tuple[str, ...],
) -> np.ndarray:
    if not prefixes:
        return np.zeros(teacher_names.shape, dtype=bool)
    prefix_match = _teacher_name_has_prefix(teacher_names.astype(str), prefixes)
    # AB rows generated by the fixed teacher expose true root alpha-beta scores.
    # Older AB rows have no source or fallback value scores and stay excluded.
    return prefix_match & (score_sources.astype(str) != "ab_root")


def _prefer_policy_over_scores_for_teachers(
    teacher_names: np.ndarray,
    score_sources: np.ndarray | None = None,
) -> np.ndarray:
    """Keep AB rows anchored to the action actually chosen by alpha-beta.

    New AB shards expose root search scores, but the anchored policy target is
    still safer for older/fallback rows where scores may be missing or generated
    by a fallback teacher. Rows marked target_score_source=ab_root are handled
    by the prefer_scores path with an explicit score/policy blend.
    """

    ab_rows = _teacher_name_has_prefix(teacher_names, ("catanatron_ab",))
    if score_sources is None:
        return ab_rows
    return ab_rows & (np.asarray(score_sources).astype(str) != "ab_root")


def _ab_root_score_rows(teacher_names: np.ndarray, score_sources: np.ndarray) -> np.ndarray:
    return _teacher_name_has_prefix(teacher_names, ("catanatron_ab",)) & (
        np.asarray(score_sources).astype(str) == "ab_root"
    )


def _blend_soft_target_rows(
    policy: np.ndarray,
    score_target: np.ndarray,
    rows: np.ndarray,
    *,
    policy_weight: float,
) -> np.ndarray:
    alpha = float(np.clip(policy_weight, 0.0, 1.0))
    mixed = alpha * policy[rows] + (1.0 - alpha) * score_target[rows]
    sums = np.sum(mixed, axis=1, keepdims=True)
    valid = sums[:, 0] > 0.0
    if np.any(valid):
        mixed[valid] = mixed[valid] / sums[valid]
    return mixed.astype(np.float32, copy=False)


def _checkpoint_config_mismatches(
    *,
    policy_type: str | None,
    config,
    args: argparse.Namespace,
) -> list[str]:
    mismatches: list[str] = []
    if policy_type is not None and str(policy_type) != str(args.arch):
        mismatches.append(f"policy_type checkpoint={policy_type} cli={args.arch}")
    if config is None:
        return mismatches
    # Task #74: checkpoints may store the config as a name-keyed dict; the
    # getattr probes below need an attribute view over that form.
    from catan_zero.rl.config_serialization import config_attr_view

    config = config_attr_view(config)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None and int(hidden_size) != int(args.hidden_size):
        mismatches.append(f"hidden_size checkpoint={hidden_size} cli={args.hidden_size}")
    if args.arch == "xdim_graph":
        token_count = getattr(config, "token_count", None)
        if token_count is not None and int(token_count) != int(args.graph_tokens):
            mismatches.append(f"graph_tokens checkpoint={token_count} cli={args.graph_tokens}")
        board_layers = getattr(config, "board_layers", None)
        if board_layers is not None and int(board_layers) != int(args.graph_layers):
            mismatches.append(f"graph_layers checkpoint={board_layers} cli={args.graph_layers}")
        attention_heads = getattr(config, "attention_heads", None)
        if attention_heads is not None and int(attention_heads) != int(args.attention_heads):
            mismatches.append(
                f"attention_heads checkpoint={attention_heads} cli={args.attention_heads}"
            )
        dropout = getattr(config, "dropout", None)
        if dropout is not None and abs(float(dropout) - float(args.graph_dropout)) > 1.0e-9:
            mismatches.append(f"graph_dropout checkpoint={dropout} cli={args.graph_dropout}")
    if args.arch == "entity_graph":
        state_trunk = str(getattr(config, "state_trunk", "transformer"))
        requested_state_trunk = str(
            getattr(args, "entity_state_trunk", "transformer")
        )
        if state_trunk != requested_state_trunk:
            mismatches.append(
                "entity_state_trunk "
                f"checkpoint={state_trunk} cli={requested_state_trunk}"
            )
        state_layers = getattr(config, "state_layers", None)
        if state_layers is not None and int(state_layers) != int(args.graph_layers):
            mismatches.append(f"graph_layers checkpoint={state_layers} cli={args.graph_layers}")
        attention_heads = getattr(config, "attention_heads", None)
        if attention_heads is not None and int(attention_heads) != int(args.attention_heads):
            mismatches.append(
                f"attention_heads checkpoint={attention_heads} cli={args.attention_heads}"
            )
        dropout = getattr(config, "dropout", None)
        if dropout is not None and abs(float(dropout) - float(args.graph_dropout)) > 1.0e-9:
            mismatches.append(f"graph_dropout checkpoint={dropout} cli={args.graph_dropout}")
        for config_name, cli_name, default in (
            ("relational_block_pattern", "relational_block_pattern", ""),
            ("relational_ff_size", "relational_ff_size", 0),
            ("relational_bases", "relational_bases", 4),
            (
                "relational_action_cross_layers",
                "relational_action_cross_layers",
                1,
            ),
            (
                "relational_edge_policy_head",
                "relational_edge_policy_head",
                True,
            ),
            ("latent_deliberation_steps", "latent_deliberation_steps", 0),
            ("latent_deliberation_slots", "latent_deliberation_slots", 8),
            ("moe_routed_experts", "moe_routed_experts", 0),
            ("moe_top_k", "moe_top_k", 2),
            ("moe_expert_ff_size", "moe_expert_ff_size", 0),
        ):
            checkpoint_value = getattr(config, config_name, default)
            cli_value = getattr(args, cli_name, default)
            if checkpoint_value != cli_value:
                mismatches.append(
                    f"{cli_name} checkpoint={checkpoint_value} cli={cli_value}"
                )
        categorical_bins = int(getattr(config, "value_categorical_bins", 0) or 0)
        requested_bins = int(getattr(args, "value_categorical_bins", 0) or 0)
        if categorical_bins != requested_bins:
            mismatches.append(
                "value_categorical_bins "
                f"checkpoint={categorical_bins} cli={requested_bins}"
            )
    return mismatches


def _preflight_init_checkpoint_architecture(args: argparse.Namespace, ddp: dict) -> None:
    if not args.init_checkpoint or args.arch not in {"xdim_graph", "entity_graph"}:
        return
    import torch

    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return
    mismatches = _checkpoint_config_mismatches(
        policy_type=checkpoint.get("policy_type"),
        config=checkpoint.get("config"),
        args=args,
    )
    if mismatches:
        raise SystemExit(
            "init checkpoint architecture does not match requested run: "
            + "; ".join(mismatches)
        )
    _rank0_print(
        json.dumps(
            {
                "progress": "init_checkpoint_architecture_preflight",
                "checkpoint": args.init_checkpoint,
                "policy_type": checkpoint.get("policy_type"),
                "ok": True,
            },
            sort_keys=True,
        ),
        ddp,
    )


def _assert_init_config_matches(policy, args: argparse.Namespace) -> None:
    mismatches = _checkpoint_config_mismatches(
        policy_type=getattr(policy, "policy_type", None),
        config=getattr(policy, "config", None),
        args=args,
    )
    if mismatches:
        raise SystemExit(
            "init checkpoint architecture does not match requested run: "
            + "; ".join(mismatches)
        )


def _enforce_35m_model_size(policy, args: argparse.Namespace) -> None:
    if not bool(getattr(args, "require_35m_model", False)):
        return
    if str(args.arch) not in {"xdim_graph", "entity_graph"}:
        raise SystemExit("--require-35m-model requires --arch xdim_graph or --arch entity_graph")
    count = int(_parameter_count(policy))
    lower = int(args.min_35m_params)
    upper = int(args.max_35m_params)
    if count < lower or count > upper:
        raise SystemExit(
            f"{args.arch} parameter count is outside the required 35M range: "
            f"count={count} expected=[{lower}, {upper}]. Check --hidden-size, "
            "--graph-tokens, and --graph-layers before launching a production BC run."
        )


def _set_xdim_q_branch_trainable(model, trainable: bool) -> None:
    module = getattr(model, "module", model)
    for name in ("q_state", "q_action", "q_bias", "q_head"):
        layer = getattr(module, name, None)
        if layer is None:
            continue
        for param in layer.parameters():
            param.requires_grad = bool(trainable)


def _set_scalar_value_head_trainable(model, trainable: bool) -> None:
    """Freeze the legacy scalar readout in categorical-primary runs.

    A zero coefficient alone is insufficient under AdamW: a zero gradient can
    still receive decoupled weight decay.  Freezing makes the reported scalar
    metric a genuinely fixed diagnostic unless the explicit HL scalar-aux
    weight is enabled.
    """

    module = getattr(model, "module", model)
    value_head = getattr(module, "value_head", None)
    if value_head is None:
        raise SystemExit(
            "categorical-primary training expected a named scalar value_head "
            "to freeze for diagnostic integrity"
        )
    for parameter in value_head.parameters():
        parameter.requires_grad = bool(trainable)


# Named groups of EntityGraphNet submodules that --freeze-modules / --train-value-only can
# freeze. ``value_heads`` is opt-in: value-repair runs leave it trainable, while the
# later action-local policy warmup can freeze every value-specific parameter cleanly
# instead of relying on a zero loss coefficient (which is not an AdamW freeze).
ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    "trunk": (
        "hex_encoder",
        "vertex_encoder",
        "edge_encoder",
        "player_encoder",
        "global_encoder",
        "event_encoder",
        "type_embedding",
        "cls_token",
        "blocks",
        "state_norm",
        "deliberation_slots",
        "deliberation_block",
        "deliberation_fusion_norm",
        "deliberation_fusion",
        "deliberation_halt_head",
    ),
    "action_encoder": ("action_encoder",),
    "policy_head": ("action_bias", "logit_scale"),
    "value_heads": (
        "value_head",
        "value_categorical_head",
        "final_vp_head",
        "value_uncertainty_head",
        "value_probe",
        "value_probe_norm_q",
        "value_probe_norm_kv",
        "value_probe_attn",
        "value_pool_head",
    ),
}


def _set_entity_graph_modules_trainable(
    model, group_names, *, trainable: bool
) -> list[str]:
    """Freeze/unfreeze named EntityGraphNet module groups (mirrors
    ``_set_xdim_q_branch_trainable``'s pattern for the XDim architecture).

    Recognized group names are the keys of ``ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS``.
    Raises SystemExit on an unrecognized name. Returns the list of attribute names
    actually touched, for logging.
    """
    module = getattr(model, "module", model)
    group_names = list(group_names)
    unknown = sorted(set(group_names) - set(ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS))
    if unknown:
        raise SystemExit(
            "unknown --freeze-modules group(s): "
            f"{unknown}; valid groups are {sorted(ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS)}"
        )
    touched: list[str] = []
    for group_name in group_names:
        for attr_name in ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS[group_name]:
            attr = getattr(module, attr_name, None)
            if attr is None:
                continue
            touched.append(attr_name)
            parameters = attr.parameters() if hasattr(attr, "parameters") else (attr,)
            for param in parameters:
                param.requires_grad = bool(trainable)
    return touched


def _lr_warmup_multiplier(step: int, warmup_steps: int) -> float:
    """Linear ramp from 0 at step 0 to 1.0 at (and past) ``warmup_steps``. ``step`` is
    0-indexed (the step about to run); a ``warmup_steps <= 0`` disables the ramp (always 1.0).
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def _apply_lr_warmup(optimizer, *, base_lr: float, step: int, warmup_steps: int) -> float:
    """Set every optimizer param group's lr to ``group_base_lr * warmup multiplier`` for
    this step, where ``group_base_lr`` is the group's own ``"base_lr"`` key when present
    (see ``_build_optimizer_param_groups``'s --value-lr-mult split) and ``base_lr``
    otherwise -- so a single-group optimizer (no "base_lr" key, e.g. every call site
    before --value-lr-mult existed) is unaffected. Returns the multiplier actually
    applied, for logging."""
    multiplier = _lr_warmup_multiplier(step, warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = float(group.get("base_lr", base_lr)) * multiplier
    return multiplier


def _lr_schedule_multiplier(
    step: int, *, warmup_steps: int, total_steps: int, schedule: str
) -> float:
    """AUDIT FIX (LR decay): combine the existing linear warmup ramp with an optional
    post-warmup decay curve. ``step`` is 0-indexed (the step about to run).

    ``schedule="flat"`` returns exactly ``_lr_warmup_multiplier(step, warmup_steps)`` --
    i.e. warmup-then-hold, bit-identical to every training recipe run before this flag
    existed. ``"cosine"``/``"linear"`` only change behavior once the caller opts in via
    --lr-schedule.
    """
    warmup_multiplier = _lr_warmup_multiplier(step, warmup_steps)
    if schedule == "flat":
        return warmup_multiplier
    if step < warmup_steps:
        return warmup_multiplier
    decay_span = max(1, int(total_steps) - int(warmup_steps))
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_span)))
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        return max(0.0, 1.0 - progress)
    raise SystemExit(f"unknown --lr-schedule {schedule!r}; expected flat, cosine, or linear")


def _apply_lr_schedule(
    optimizer,
    *,
    base_lr: float,
    step: int,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
) -> float:
    """Set every optimizer param group's lr to ``group_base_lr * schedule multiplier``
    for this step, where ``group_base_lr`` is the group's own ``"base_lr"`` key when
    present (see ``_build_optimizer_param_groups``'s --value-lr-mult split) and
    ``base_lr`` otherwise -- so a single-group optimizer (no "base_lr" key, e.g. every
    call site before --value-lr-mult existed) is unaffected. Returns the multiplier
    actually applied, for logging."""
    multiplier = _lr_schedule_multiplier(
        step, warmup_steps=warmup_steps, total_steps=total_steps, schedule=schedule
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group.get("base_lr", base_lr)) * multiplier
    return multiplier


# Submodule attribute names treated as "value head" parameters for --value-lr-mult.
# This must cover the complete value-only readout path, including the optional
# attention pool. Otherwise ``--value-lr-mult`` misleadingly slows the final
# linear head while its fresh value probe/attention layers train at trunk LR.
# Missing attributes are harmless for XDimLite/XDimGraph.
VALUE_HEAD_MODULE_ATTRS: tuple[str, ...] = (
    "value_head",
    "value_categorical_head",
    "final_vp_head",
    "value_uncertainty_head",
    "value_probe",
    "value_probe_norm_q",
    "value_probe_norm_kv",
    "value_probe_attn",
    "value_pool_head",
)

# Opt-in action-local modules introduced by the gather/cross-attention upgrade.
# They are freshly initialized when upgrading a scalar/global-policy checkpoint,
# so the later architecture A/B needs an independent LR from both the mature
# trunk and the value heads.
ACTION_LOCAL_MODULE_ATTRS: tuple[str, ...] = (
    "target_gather_proj",
    "action_cross_blocks",
)


def _optimizer_param_group_report(params, *, base_lr: float) -> list[dict[str, object]]:
    """Describe optimizer groups without mutating the objects passed to torch.

    ``_group_name`` is private construction metadata and is stripped by
    ``_make_optimizer`` before optimizer/checkpoint state is created.
    """
    param_groups = list(params)
    if not param_groups or not isinstance(param_groups[0], dict):
        tensors = [p for p in param_groups if p.requires_grad]
        return [
            {
                "group": "historical_flat",
                "lr": float(base_lr),
                "parameter_tensors": len(tensors),
                "parameters": sum(int(p.numel()) for p in tensors),
            }
        ]
    return [
        {
            "group": str(group.get("_group_name", "unnamed")),
            "lr": float(group["lr"]),
            "parameter_tensors": len(group["params"]),
            "parameters": sum(int(p.numel()) for p in group["params"]),
        }
        for group in param_groups
    ]


def _build_optimizer_param_groups(
    model,
    *,
    base_lr: float,
    value_lr_mult: float,
    action_module_lr_mult: float = 1.0,
    trunk_lr_mult: float = 1.0,
    architecture: str | None = None,
):
    """Return the optimizer's ``params`` argument for ``model``'s trainable parameters.

    All multipliers at 1.0 return a FLAT list of parameters -- exactly the
    historical single-implicit-group optimizer. Non-unit multipliers split the
    corresponding named modules into independent groups; every other trainable
    parameter stays in a base group at ``base_lr``. Each group dict carries its
    own ``"base_lr"`` key so
    ``_apply_lr_schedule``/``_apply_lr_warmup`` scale the right rate per group instead
    of overwriting every group with the same absolute rate.
    """
    module = getattr(model, "module", model)
    trainable = [p for p in module.parameters() if p.requires_grad]
    multipliers = {
        "value-lr-mult": float(value_lr_mult),
        "action-module-lr-mult": float(action_module_lr_mult),
        "trunk-lr-mult": float(trunk_lr_mult),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in multipliers.values()):
        raise SystemExit(
            "--value-lr-mult, --action-module-lr-mult, and --trunk-lr-mult "
            "must all be > 0; "
            "freeze modules explicitly instead of encoding a freeze as LR 0"
        )
    if all(value == 1.0 for value in multipliers.values()):
        return trainable
    if float(trunk_lr_mult) != 1.0 and architecture != "entity_graph":
        raise SystemExit(
            "--trunk-lr-mult != 1.0 is supported only for --arch entity_graph; "
            f"received architecture={architecture!r}"
        )

    def _params_under(attrs: tuple[str, ...]) -> list:
        params: list = []
        for attr_name in attrs:
            submodule = getattr(module, attr_name, None)
            if submodule is None:
                continue
            parameters = (
                submodule.parameters()
                if hasattr(submodule, "parameters")
                else (submodule,)
            )
            params.extend(p for p in parameters if p.requires_grad)
        return params

    value_params = (
        _params_under(VALUE_HEAD_MODULE_ATTRS)
        if float(value_lr_mult) != 1.0
        else []
    )
    action_params = (
        _params_under(ACTION_LOCAL_MODULE_ATTRS)
        if float(action_module_lr_mult) != 1.0
        else []
    )
    trunk_params = (
        _params_under(ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["trunk"])
        if float(trunk_lr_mult) != 1.0
        else []
    )
    if float(value_lr_mult) != 1.0 and not value_params:
        raise SystemExit(
            "--value-lr-mult != 1.0 but the model has no trainable parameters under "
            f"any of {VALUE_HEAD_MODULE_ATTRS} -- nothing to apply the multiplier to."
        )
    if float(action_module_lr_mult) != 1.0 and not action_params:
        raise SystemExit(
            "--action-module-lr-mult != 1.0 but the model has no trainable "
            f"parameters under any of {ACTION_LOCAL_MODULE_ATTRS}"
        )
    if float(trunk_lr_mult) != 1.0 and not trunk_params:
        raise SystemExit(
            "--trunk-lr-mult != 1.0 but the entity-graph model has no trainable "
            "parameters under the canonical trunk module group"
        )
    value_param_ids = {id(p) for p in value_params}
    action_param_ids = {id(p) for p in action_params}
    trunk_param_ids = {id(p) for p in trunk_params}
    pairwise_overlaps = {
        "value/action": value_param_ids & action_param_ids,
        "value/trunk": value_param_ids & trunk_param_ids,
        "action/trunk": action_param_ids & trunk_param_ids,
    }
    overlapping = {name: ids for name, ids in pairwise_overlaps.items() if ids}
    if overlapping:
        raise RuntimeError(
            "optimizer parameter groups overlap: "
            + ", ".join(f"{name}={len(ids)}" for name, ids in overlapping.items())
        )
    grouped_ids = value_param_ids | action_param_ids | trunk_param_ids
    base_params = [p for p in trainable if id(p) not in grouped_ids]
    groups = [
        {
            "params": base_params,
            "lr": float(base_lr),
            "base_lr": float(base_lr),
            "_group_name": "base",
        }
    ]
    if value_params:
        value_lr = float(base_lr) * float(value_lr_mult)
        groups.append(
            {
                "params": value_params,
                "lr": value_lr,
                "base_lr": value_lr,
                "_group_name": "value",
            }
        )
    if action_params:
        action_lr = float(base_lr) * float(action_module_lr_mult)
        groups.append(
            {
                "params": action_params,
                "lr": action_lr,
                "base_lr": action_lr,
                "_group_name": "action_local",
            }
        )
    if trunk_params:
        trunk_lr = float(base_lr) * float(trunk_lr_mult)
        groups.append(
            {
                "params": trunk_params,
                "lr": trunk_lr,
                "base_lr": trunk_lr,
                "_group_name": "trunk",
            }
        )
    assigned = [id(p) for group in groups for p in group["params"]]
    trainable_ids = [id(p) for p in trainable]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(trainable_ids):
        raise RuntimeError(
            "optimizer parameter grouping did not assign every trainable parameter "
            "exactly once"
        )
    return groups


def _make_optimizer(params, args, device):
    import torch

    param_groups = list(params)
    if param_groups and isinstance(param_groups[0], dict):
        # --value-lr-mult path: `params` is already a list of param-group dicts (see
        # `_build_optimizer_param_groups`), each carrying its own "lr". Drop any
        # non-trainable stragglers per group (mirrors the flat-list branch below).
        trainable_params = [
            {
                **{key: value for key, value in group.items() if key != "_group_name"},
                "params": [p for p in group["params"] if p.requires_grad],
            }
            for group in param_groups
        ]
        if not any(group["params"] for group in trainable_params):
            raise SystemExit("no trainable parameters found for BC optimizer")
    else:
        trainable_params = [parameter for parameter in param_groups if parameter.requires_grad]
        if not trainable_params:
            raise SystemExit("no trainable parameters found for BC optimizer")
    optimizer_name = str(args.optimizer).lower()
    weight_decay = float(args.weight_decay)
    # AUDIT FIX (weight-decay silent no-op): plain torch.optim.Adam's weight_decay
    # is L2-regularization added to the gradient BEFORE the Adam moment estimates,
    # not AdamW's decoupled decay -- they are not interchangeable, and this
    # function used to only ever forward --weight-decay when --optimizer adamw was
    # also passed, silently dropping it under the default "adam" optimizer. Fail
    # loud instead: a nonzero --weight-decay with --optimizer adam is always a
    # config mistake here (this repo's only two choices are "adam"/"adamw"), so
    # refuse rather than guess which regularization the caller meant.
    if weight_decay != 0.0 and optimizer_name != "adamw":
        raise SystemExit(
            f"--weight-decay {weight_decay} has no effect with --optimizer "
            f"{optimizer_name!r}: this trainer only forwards weight_decay to "
            "torch.optim.AdamW's decoupled decay. Pass --optimizer adamw to "
            "apply it, or --weight-decay 0.0 to train without decay."
        )
    cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    kwargs = {"lr": float(args.lr)}
    if optimizer_name == "adamw":
        kwargs["weight_decay"] = weight_decay
    if bool(args.fused_optimizer) and str(device).startswith("cuda"):
        kwargs["fused"] = True
    try:
        return cls(trainable_params, **kwargs)
    except TypeError:
        if "fused" not in kwargs:
            raise
        kwargs.pop("fused", None)
        return cls(trainable_params, **kwargs)


def _amp_context(device, amp: str):
    from contextlib import nullcontext

    if str(amp).lower() != "bf16":
        return nullcontext()
    if not str(device).startswith("cuda"):
        return nullcontext()
    import torch

    if not torch.cuda.is_available():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _ensure_policy_action_mask_version(
    policy,
    env_config,
    *,
    allow_legacy_upgrade: bool,
    checkpoint_path: str,
) -> None:
    config = getattr(policy, "config", None)
    if config is None:
        return
    current = str(getattr(config, "action_mask_version", "") or "")
    if current:
        return
    expected = _expected_action_mask_version(env_config)
    if not expected:
        return
    if not allow_legacy_upgrade:
        source = f" {checkpoint_path}" if checkpoint_path else ""
        raise SystemExit(
            f"init checkpoint{source} is missing action_mask_version; refusing "
            "to stamp it with the current action catalog automatically. Regenerate "
            "from a versioned checkpoint or pass --allow-legacy-action-mask-upgrade "
            "only after an action-ID replay smoke test passes."
        )
    object.__setattr__(config, "action_mask_version", expected)


def _batch_profile(policy, data: dict, batch: np.ndarray) -> dict:
    config = getattr(policy, "config", None)
    action_size = int(getattr(policy, "action_size", getattr(config, "action_size", 0)))
    context_size = int(
        getattr(
            policy,
            "context_action_feature_size",
            getattr(config, "context_action_feature_size", 0),
        )
    )
    valid = data["legal_action_ids"][batch]
    legal_counts = np.sum(valid >= 0, axis=1)
    profile = {
        "batch_size": int(len(batch)),
        "obs_shape": list(data["obs"][batch].shape),
        "legal_action_ids_shape": list(valid.shape),
        "legal_action_context_shape": list(data["legal_action_context"][batch].shape),
        "dense_action_context_shape": [int(len(batch)), action_size, context_size],
        "legal_actions_mean": float(np.mean(legal_counts)) if len(legal_counts) else 0.0,
        "legal_actions_p90": int(np.percentile(legal_counts, 90)) if len(legal_counts) else 0,
        "legal_actions_max": int(np.max(legal_counts)) if len(legal_counts) else 0,
        "parameter_count": int(_parameter_count(policy)),
        "cuda": _cuda_memory(policy),
    }
    if getattr(policy, "policy_type", "") == "entity_graph":
        for key in (
            "hex_tokens",
            "vertex_tokens",
            "edge_tokens",
            "player_tokens",
            "global_tokens",
            "legal_action_tokens",
            "event_tokens",
        ):
            if key in data:
                profile[f"{key}_shape"] = list(data[key][batch].shape)
    return profile


def _parameter_count(policy) -> int:
    return int(sum(parameter.numel() for parameter in _params(policy)))


def _cuda_memory(policy) -> dict:
    try:
        import torch
    except ImportError:
        return {}
    device = getattr(policy, "device", None)
    if not torch.cuda.is_available() or device is None or str(device).startswith("cpu"):
        return {}
    return {
        "allocated_mib": float(torch.cuda.memory_allocated(device) / 1024 / 1024),
        "reserved_mib": float(torch.cuda.memory_reserved(device) / 1024 / 1024),
        "max_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024 / 1024),
        "max_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024 / 1024),
    }


def _epoch_checkpoint_path(checkpoint: str, epoch: int) -> Path:
    path = Path(checkpoint)
    suffix = "".join(path.suffixes) or ".pt"
    stem = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
    return path.with_name(f"{stem}_epoch{epoch:04d}{suffix}")


def _acquire_host_train_lock(lock_file: str, ddp: dict[str, int | bool]):
    """Prevent accidental same-host BC job stacking.

    Only local rank 0 owns the lock for DDP jobs. If a second DDP launcher starts,
    its rank 0 exits quickly and torchrun tears down the sibling ranks.
    """

    if bool(ddp["enabled"]) and int(ddp["local_rank"]) != 0:
        return None
    path = Path(lock_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit(
            f"another train_bc.py job is already running on this host; "
            f"lock_file={path}. Pass --allow-concurrent-bc only if the GPUs "
            "are intentionally partitioned."
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "rank": int(ddp["rank"]),
                "local_rank": int(ddp["local_rank"]),
                "started_unix": time.time(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def _iter_with_last(iterable):
    """Yield ``(item, is_last)`` for every item, with ``is_last=True`` only on the
    final item. Uses one-step lookahead so it works on single-pass generators
    (the streaming memmap/prefetch batch iterator) without materialising them.
    An empty iterable yields nothing.
    """
    iterator = iter(iterable)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    for current in iterator:
        yield previous, False
        previous = current
    yield previous, True


def _gradient_sync_context(model, *, accum_do_step: bool):
    """Suppress DDP/FSDP gradient sync around a complete micro-batch.

    PyTorch requires ``no_sync`` to cover the forward pass that creates reducer
    hooks, not merely ``backward``.  The caller therefore enters this context
    before invoking the batch trainer.  Plain modules retain a strict no-op.
    """
    import contextlib

    if not bool(accum_do_step) and hasattr(model, "no_sync"):
        return model.no_sync()
    return contextlib.nullcontext()


def _accumulation_group_size(
    *, configured_size: int, batch_number: int, total_batches: int
) -> int:
    """Return the actual divisor for the group beginning at ``batch_number``."""
    configured = int(configured_size)
    batch = int(batch_number)
    total = int(total_batches)
    if configured < 1 or batch < 1 or total < batch:
        raise ValueError("invalid gradient accumulation group bounds")
    return min(configured, total - batch + 1)


def _warm_start_grow(policy, checkpoint_path: str, *, device: str) -> dict:
    """C1 warm-start-GROW: copy every parameter/buffer whose NAME and SHAPE
    match from ``checkpoint_path`` into the already-constructed (fresh, typically
    bigger) ``policy.model``; leave every other tensor at its fresh init.

    This is the "load matching keys, init the rest, log the fraction loaded"
    contract from the C1 spec. It deliberately does NOT require an architecture
    match (that is what --init-checkpoint is for): a same-width/deeper config
    warm-starts the shared trunk blocks + token encoders + heads cleanly (only
    the extra blocks.<i> stay fresh), while a width change matches little and the
    run is effectively from-scratch -- either way the fraction loaded is logged
    so the caller sees exactly how much signal was transferred.

    Returns a JSON-safe report: parameter/tensor counts, the loaded fraction (by
    parameter element count -- the number that matters for "how warm is this
    start"), and the sizes of the mismatched/missing/unexpected buckets.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit(
            f"--grow-from-checkpoint {checkpoint_path} is not a policy checkpoint "
            "(no 'model' state_dict)"
        )
    source = ckpt["model"]
    target_model = policy.model
    target_sd = target_model.state_dict()

    to_copy: dict = {}
    shape_mismatch: list[str] = []
    unexpected: list[str] = []
    for name, src_tensor in source.items():
        if name not in target_sd:
            unexpected.append(name)
            continue
        if tuple(src_tensor.shape) == tuple(target_sd[name].shape):
            to_copy[name] = src_tensor
        else:
            shape_mismatch.append(name)
    missing = [name for name in target_sd if name not in source]

    # strict=False so the fresh-init tensors for missing/mismatched keys are
    # left untouched; assign=False keeps the target's dtype/device.
    target_model.load_state_dict(to_copy, strict=False)

    def _param_elems(names) -> int:
        return int(sum(int(target_sd[n].numel()) for n in names if n in target_sd))

    total_target_params = int(sum(p.numel() for p in target_model.parameters()))
    # Restrict the "loaded" accounting to trainable parameters (buffers such as
    # the non-persistent value_categorical_support never appear in state_dict
    # persistently, but any registered buffers that do should not inflate the
    # "fraction of the MODEL warm-started" number the operator reads).
    param_names = {name for name, _ in target_model.named_parameters()}
    loaded_param_elems = _param_elems(n for n in to_copy if n in param_names)
    loaded_fraction = (
        float(loaded_param_elems) / float(total_target_params)
        if total_target_params > 0
        else 0.0
    )
    return {
        "checkpoint": checkpoint_path,
        "source_tensors": int(len(source)),
        "target_tensors": int(len(target_sd)),
        "copied_tensors": int(len(to_copy)),
        "shape_mismatch_tensors": int(len(shape_mismatch)),
        "missing_in_source_tensors": int(len(missing)),
        "unexpected_in_source_tensors": int(len(unexpected)),
        "target_total_params": total_target_params,
        "loaded_params": loaded_param_elems,
        "loaded_fraction": round(loaded_fraction, 6),
        "shape_mismatch_examples": sorted(shape_mismatch)[:8],
        "missing_examples": sorted(missing)[:8],
    }


def _distributed_state() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "enabled": world_size > 1,
        "world_size": world_size,
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }


_TRAINING_PRECOMPUTE_CACHE_SCHEMA = "train-bc-derived-arrays-v1"


def _array_content_sha256(array: np.ndarray) -> str:
    """Hash an array's exact C-order bytes without materialising a second copy."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    view = memoryview(contiguous).cast("B")
    block = 8 * 1024 * 1024
    for offset in range(0, len(view), block):
        digest.update(view[offset : offset + block])
    return f"sha256:{digest.hexdigest()}"


def _derived_array_cache_key(payload: dict[str, object]) -> tuple[str, dict[str, object]]:
    """Return a content key and the canonical, versioned cache identity.

    The caller must bind every corpus and recipe input that can affect an array.
    The implementation file and NumPy version are bound here centrally, so a
    code or numerical-runtime change cannot silently reuse an older result.
    """

    identity = {
        "schema_version": _TRAINING_PRECOMPUTE_CACHE_SCHEMA,
        "implementation_sha256": _sha256_existing_file(__file__),
        "numpy_version": str(np.__version__),
        "inputs": payload,
    }
    digest = _canonical_json_sha256(identity).removeprefix("sha256:")
    return digest, identity


def _write_derived_array_cache(
    cache_root: Path,
    identity: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> Path:
    """Atomically publish immutable, checksummed ``.npy`` derived arrays."""

    key = _canonical_json_sha256(identity).removeprefix("sha256:")
    destination = cache_root / key
    cache_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        # Never overwrite an existing content-addressed entry. Validation is
        # deliberately delegated to the loader, which fails closed on damage.
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
    try:
        inventory: dict[str, dict[str, object]] = {}
        for name, value in sorted(arrays.items()):
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"unsafe derived-array cache name {name!r}")
            array = np.ascontiguousarray(value)
            path = temporary / f"{name}.npy"
            with open(path, "wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o444)
            inventory[name] = {
                "file": path.name,
                "dtype": array.dtype.str,
                "shape": [int(item) for item in array.shape],
                "content_sha256": _array_content_sha256(array),
                "file_sha256": _sha256_existing_file(path),
            }
        manifest = {
            "schema_version": _TRAINING_PRECOMPUTE_CACHE_SCHEMA,
            "identity": identity,
            "identity_sha256": f"sha256:{key}",
            "arrays": inventory,
        }
        manifest_path = temporary / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_path.chmod(0o444)
        try:
            os.replace(temporary, destination)
        except FileExistsError:
            # A concurrent run with the same authenticated identity won the
            # publish race. Its atomic rename guarantees a complete entry;
            # the caller still authenticates every byte before use.
            import shutil

            shutil.rmtree(temporary)
        directory_fd = os.open(cache_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise
    return destination


def _load_derived_array_cache(
    cache_root: Path,
    identity: dict[str, object],
    *,
    verify_digests: bool = True,
) -> dict[str, np.ndarray]:
    """Authenticate a cache entry before exposing read-only memory maps.

    Both the ``.npy`` file bytes and decoded C-order array bytes are checked.
    This is intentionally fail-closed: an existing but partial/tampered entry
    aborts training instead of being silently regenerated or reused.
    """

    key = _canonical_json_sha256(identity).removeprefix("sha256:")
    directory = cache_root / key
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"derived-array cache manifest is unreadable: {manifest_path}") from error
    if manifest.get("schema_version") != _TRAINING_PRECOMPUTE_CACHE_SCHEMA:
        raise SystemExit("derived-array cache schema drift")
    if manifest.get("identity") != identity or manifest.get("identity_sha256") != f"sha256:{key}":
        raise SystemExit("derived-array cache identity drift")
    inventory = manifest.get("arrays")
    if not isinstance(inventory, dict) or not inventory:
        raise SystemExit("derived-array cache has no array inventory")
    loaded: dict[str, np.ndarray] = {}
    for name, record in sorted(inventory.items()):
        if not isinstance(record, dict) or record.get("file") != f"{name}.npy":
            raise SystemExit(f"derived-array cache inventory is invalid for {name!r}")
        path = directory / f"{name}.npy"
        if verify_digests and _sha256_existing_file(path) != record.get("file_sha256"):
            raise SystemExit(f"derived-array cache file digest mismatch: {name}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise SystemExit(f"derived-array cache cannot decode {name}") from error
        if array.dtype.str != record.get("dtype") or list(array.shape) != record.get("shape"):
            raise SystemExit(f"derived-array cache shape/dtype drift: {name}")
        if verify_digests and _array_content_sha256(array) != record.get("content_sha256"):
            raise SystemExit(f"derived-array cache content digest mismatch: {name}")
        loaded[name] = array
    return loaded


def _composite_game_sampling_weights(
    data, train_indices: np.ndarray
) -> np.ndarray | None:
    """Return authenticated component->game->row sampling probabilities.

    For a v2 composite, a draw first selects a component by its bound ratio,
    then a training game uniformly within that component, then a row uniformly
    within that game.  Returning weights aligned to ``train_indices`` lets the
    existing seeded weighted epoch sampler perform that hierarchy without
    copying any corpus payload. V1 and ordinary corpora return ``None`` and
    therefore retain the historical permutation path exactly.
    """
    ratios = tuple(getattr(data, "component_game_sampling_ratios", tuple()))
    if not ratios:
        return None
    if len(ratios) != len(getattr(data, "corpora", tuple())):
        raise SystemExit("authenticated component sampling ratio count drift")
    indices = np.asarray(train_indices, dtype=np.int64)
    components = np.asarray(data.component_indices_for_rows(indices), dtype=np.int64)
    seeds = np.asarray(data["game_seed"][indices], dtype=np.int64)
    weights = np.zeros(indices.shape[0], dtype=np.float64)
    for component, ratio in enumerate(ratios):
        positions = np.flatnonzero(components == component)
        if positions.size == 0:
            raise SystemExit(
                f"authenticated component {component} has no training rows"
            )
        component_seeds = seeds[positions]
        _games, inverse, counts = np.unique(
            component_seeds, return_inverse=True, return_counts=True
        )
        if counts.size == 0:
            raise SystemExit(
                f"authenticated component {component} has no training games"
            )
        weights[positions] = float(ratio) / (
            float(counts.size) * counts[inverse].astype(np.float64)
        )
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise SystemExit("authenticated component game sampling produced invalid weights")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit("authenticated component game sampling mass drift")
    return weights


def _conditioned_policy_aux_sampling_weights(
    base_sampling_weights: np.ndarray,
    policy_weight_multiplier: np.ndarray,
) -> np.ndarray:
    """Condition an authenticated base row measure on policy-active rows."""
    base = np.asarray(base_sampling_weights, dtype=np.float64)
    multiplier = np.asarray(policy_weight_multiplier)
    if base.ndim != 1 or multiplier.ndim != 1 or base.shape != multiplier.shape:
        raise ValueError("policy auxiliary base/multiplier shape drift")
    if not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError("policy auxiliary base measure is invalid")
    # Admission only: phase/winner/loser weights remain loss weights and must
    # not silently become a second sampling-frequency correction.
    conditioned = np.where(multiplier > 0.0, base, 0.0)
    total = float(conditioned.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("policy auxiliary measure has no active mass")
    conditioned /= total
    return conditioned


def _policy_aux_epoch_order(
    rng: np.random.Generator,
    n: int,
    sample_weights: np.ndarray,
    *,
    local_draws: int,
    ddp: dict[str, int | bool],
) -> np.ndarray:
    """Draw an exact per-rank dose from one deterministic global stream."""
    if int(local_draws) < 0:
        raise ValueError("policy auxiliary local_draws must be non-negative")
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.shape != (int(n),):
        raise ValueError("policy auxiliary sampling weight length drift")
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("policy auxiliary sampling weights have no mass")
    world = int(ddp["world_size"]) if ddp.get("enabled", False) else 1
    rank = int(ddp["rank"]) if ddp.get("enabled", False) else 0
    global_order = rng.choice(
        int(n), size=int(local_draws) * world, replace=True, p=weights / total
    )
    return np.asarray(global_order[rank::world], dtype=np.int64)


def _epoch_order(
    rng: np.random.Generator,
    n: int,
    batch_size: int,
    ddp: dict[str, int | bool],
    *,
    data_sharded: bool = False,
    sample_weights: np.ndarray | None = None,
    max_samples: int | None = None,
) -> np.ndarray:
    """Per-epoch traversal order over ``train_indices`` positions ``0..n-1``.

    Default (``sample_weights=None``): a uniform ``rng.permutation`` -- every
    row visited exactly once per epoch, byte-identical to pre-CAT-45 behaviour.
    This is NOT merely "uniform weights passed through the weighted path" --
    ``rng.choice`` and ``rng.permutation`` consume a NumPy bit generator's
    stream differently even when the distribution is uniform, so taking that
    shortcut would silently change existing seeded runs' reproducibility.
    Callers MUST pass ``sample_weights=None`` (not an all-ones array) to get
    today's exact behaviour.

    CAT-45 weighted path (``sample_weights`` given, length ``n``, aligned to
    ``train_indices`` positions): draws ``n`` positions WITH replacement,
    probability proportional to ``sample_weights`` -- a KataGo-style weighted
    sampler (oversampling), matching torch's ``WeightedRandomSampler``
    convention. This changes SAMPLING FREQUENCY (how often a row is drawn per
    epoch), which is orthogonal to -- and composes multiplicatively in effect
    with -- ``policy_sample_weights``/``value_sample_weights`` (which scale the
    LOSS magnitude of whichever row was drawn); see the CAT-45 call site for the
    explicit non-combination-rule with those. If a future ticket also wants to
    change epoch sampling frequency (not loss weight), it must combine its
    weights with these explicitly (e.g. multiply then renormalize) rather than
    overwriting this argument.

    ``max_samples`` only bounds the weighted-with-replacement path. Its result
    is byte-identical to the same-length prefix of the historical ``size=n``
    draw for the same RNG state. ``None`` and values >= ``n`` retain the exact
    historical call. The permutation path intentionally ignores the bound.
    """
    if sample_weights is None:
        order = rng.permutation(n)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape[0] != n:
            raise ValueError(
                f"sample_weights length {weights.shape[0]} != train_indices length {n}"
            )
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("sample_weights must sum to a positive value")
        draw_count = n
        if max_samples is not None:
            if int(max_samples) < 0:
                raise ValueError("max_samples must be non-negative")
            draw_count = min(n, int(max_samples))
        order = rng.choice(n, size=draw_count, replace=True, p=weights / total)
    if not ddp["enabled"]:
        return order
    if data_sharded:
        local_order_size = len(order) if max_samples is not None else n
        total_size = int(
            np.ceil(
                _distributed_scalar_max(float(local_order_size), ddp)
                / max(1, int(batch_size))
            )
            * max(1, int(batch_size))
        )
        if total_size > len(order):
            pad = np.resize(order, total_size - len(order))
            order = np.concatenate((order, pad), axis=0)
        return order
    world_size = int(ddp["world_size"])
    rank = int(ddp["rank"])
    global_batch = max(1, int(batch_size)) * world_size
    total_size = int(np.ceil(len(order) / global_batch) * global_batch)
    if total_size > len(order):
        pad = np.resize(order, total_size - len(order))
        order = np.concatenate((order, pad), axis=0)
    return order[rank:total_size:world_size]


def _reduce_epoch_metrics(
    loss_sum: float,
    acc_sum: float,
    top3_sum: float,
    count: float,
    ddp: dict[str, int | bool],
) -> tuple[float, float, float, float]:
    if not ddp["enabled"]:
        return loss_sum, acc_sum, top3_sum, count
    import torch
    import torch.distributed as dist

    values = torch.tensor(
        [loss_sum, acc_sum, top3_sum, count],
        dtype=torch.float64,
        device=f"cuda:{int(ddp['local_rank'])}",
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return (
        float(values[0].item()),
        float(values[1].item()),
        float(values[2].item()),
        float(values[3].item()),
    )


def _reduce_named_sums(
    values_by_name: dict[str, float],
    ddp: dict[str, int | bool],
) -> dict[str, float]:
    if not ddp["enabled"] or not values_by_name:
        return dict(values_by_name)
    import torch
    import torch.distributed as dist

    names = sorted(values_by_name)
    values = torch.tensor(
        [float(values_by_name[name]) for name in names],
        dtype=torch.float64,
        device=f"cuda:{int(ddp['local_rank'])}",
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {name: float(values[index].item()) for index, name in enumerate(names)}


def _reduce_scalar_sum(value: float, ddp: dict[str, int | bool]) -> float:
    if not ddp["enabled"]:
        return float(value)
    import torch
    import torch.distributed as dist

    values = torch.tensor(
        [float(value)],
        dtype=torch.float64,
        device=f"cuda:{int(ddp['local_rank'])}",
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item())


def _distributed_scalar_max(value: float, ddp: dict[str, int | bool]) -> float:
    if not ddp["enabled"]:
        return float(value)
    import torch
    import torch.distributed as dist

    values = torch.tensor(
        [float(value)],
        dtype=torch.float64,
        device=f"cuda:{int(ddp['local_rank'])}",
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return float(values[0].item())


def _rank0_print(message: str, ddp: dict[str, int | bool]) -> None:
    if int(ddp["rank"]) == 0:
        print(message, flush=True)


def _is_fsdp(model) -> bool:
    # Single FSDP-detection path: delegate to the shared util so model-weight gather
    # (_save_policy) and optimizer-state gather (optim_state.save_optimizer_state) can
    # never disagree on what "is FSDP" means (CAT-128).
    from catan_zero.rl.optim_state import is_fsdp

    return is_fsdp(model)


def _training_resume_recipe_identity(
    train_config: TrainConfig,
    args: argparse.Namespace,
    ddp: dict[str, int | bool],
) -> dict[str, object]:
    """Stable identity of the trajectory whose Adam/schedule state may continue.

    ``init_checkpoint`` necessarily changes from the original parent to the checkpoint
    being resumed, so it is normalized out. Everything else in the typed science
    config remains bound, augmented with optimizer-step topology fields not yet in
    TrainConfig. This intentionally rejects changing max_steps, warmup/decay, loss
    recipe, corpus, batch geometry, or world size while reusing moments.
    """
    normalized = dataclasses.replace(
        train_config,
        init_checkpoint="",
        init_checkpoint_sha256="",
        resume_optimizer=True,
    )
    return {
        "schema_version": "train-bc-resume-recipe-v1",
        "normalized_train_config_sha256": normalized.full_config_hash(),
        "grad_accum_steps": int(args.grad_accum_steps),
        "world_size": int(ddp["world_size"]),
        "ddp_shard_data": bool(args.ddp_shard_data),
        "fsdp": bool(args.fsdp),
        "policy_aux_active_batch_size": int(args.policy_aux_active_batch_size),
    }


def _restore_training_progress_state(
    progress: dict[str, object] | None,
    *,
    epochs: int,
    rng,
    symmetry_rng,
    ddp: dict[str, int | bool],
) -> tuple[int, int, float, float]:
    """Restore schedule/dose counters and sampler RNG from validated progress."""
    if progress is None:
        return 0, 0, 0.0, 0.0
    global_step = int(progress["optimizer_step"])
    completed_epochs = int(progress["completed_epochs"])
    if completed_epochs > int(epochs):
        raise SystemExit(
            f"resumed completed_epochs={completed_epochs} exceeds --epochs={epochs}"
        )
    rng.bit_generator.state = progress["rng_state"]
    saved_symmetry_state = progress.get("symmetry_rng_state")
    if symmetry_rng is not None:
        if not isinstance(saved_symmetry_state, dict):
            raise SystemExit(
                "resumed symmetry-augmentation recipe lacks symmetry RNG state"
            )
        symmetry_rng.bit_generator.state = saved_symmetry_state
    elif saved_symmetry_state is not None:
        raise SystemExit(
            "training progress has symmetry RNG state but current recipe disables it"
        )
    rank_rng_states = progress.get("rank_torch_rng_states")
    if not isinstance(rank_rng_states, list) or len(rank_rng_states) != int(
        ddp["world_size"]
    ):
        raise SystemExit("resumed checkpoint lacks matching per-rank torch RNG state")
    rank_rng = rank_rng_states[int(ddp["rank"])]
    import torch

    torch.set_rng_state(torch.tensor(rank_rng["cpu"], dtype=torch.uint8))
    if rank_rng.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise SystemExit("resumed checkpoint requires CUDA RNG state on a CPU run")
        torch.cuda.set_rng_state(
            torch.tensor(rank_rng["cuda"], dtype=torch.uint8),
            device=int(ddp["local_rank"]),
        )
    return (
        global_step,
        completed_epochs,
        float(progress["scalar_training_weight_sum"]),
        float(progress["categorical_training_weight_sum"]),
    )


def _save_optimizer_sidecar(checkpoint_path: str, policy, optimizer, ddp: dict):
    """CAT-128 patch #8 wrapper: persist optimizer (Adam) state as
    ``<checkpoint_path>.optimizer.pt`` via the shared FSDP-safe util. MUST be called on
    every rank (the FSDP gather is collective); the util rank-guards the write and is
    fail-soft (a save error logs and does not crash the run)."""
    from catan_zero.rl.optim_state import save_optimizer_state

    return save_optimizer_state(checkpoint_path, policy.model, optimizer, ddp)


def _save_training_progress_sidecar(
    checkpoint_path: str,
    *,
    optimizer_saved,
    optimizer_step: int,
    completed_epochs: int,
    recipe_identity: dict[str, object],
    rng,
    symmetry_rng,
    scalar_training_weight_sum: float,
    categorical_training_weight_sum: float,
    ddp: dict,
) -> None:
    """Write the checkpoint-set commit marker on rank 0 after optimizer save."""
    from catan_zero.rl.optim_state import (
        TrainingProgressError,
        save_training_progress,
    )

    import torch

    local_rng = {
        "rank": int(ddp["rank"]),
        "cpu": torch.get_rng_state().tolist(),
        "cuda": (
            torch.cuda.get_rng_state(int(ddp["local_rank"])).tolist()
            if torch.cuda.is_available()
            else None
        ),
    }
    if bool(ddp["enabled"]):
        import torch.distributed as dist

        rank_rng_states: list[dict[str, object] | None] = [
            None for _ in range(int(ddp["world_size"]))
        ]
        dist.all_gather_object(rank_rng_states, local_rng)
        gathered_rng_states = [row for row in rank_rng_states if row is not None]
    else:
        gathered_rng_states = [local_rng]

    # Non-zero DDP/FSDP ranks participate in optimizer/RNG collectives but never write.
    if int(ddp["rank"]) != 0:
        return
    if optimizer_saved is None:
        raise RuntimeError(
            "model checkpoint saved but optimizer sidecar failed; refusing to emit "
            "a falsely resumable training checkpoint"
        )
    try:
        save_training_progress(
            checkpoint_path,
            optimizer_step=int(optimizer_step),
            completed_epochs=int(completed_epochs),
            recipe_identity=recipe_identity,
            rng_state=rng.bit_generator.state,
            symmetry_rng_state=(
                None if symmetry_rng is None else symmetry_rng.bit_generator.state
            ),
            rank_torch_rng_states=gathered_rng_states,
            scalar_training_weight_sum=float(scalar_training_weight_sum),
            categorical_training_weight_sum=float(
                categorical_training_weight_sum
            ),
            ddp=ddp,
        )
    except TrainingProgressError as error:
        raise RuntimeError(
            f"could not commit resumable checkpoint set: {error}"
        ) from error


def _save_policy(
    policy,
    path: str,
    ddp: dict[str, int | bool],
    *,
    mask_hidden_info: bool = False,
    soft_target_source: str | None = None,
    value_training: dict[str, object] | None = None,
) -> None:
    """Write a policy checkpoint. Safe to call from EVERY rank: single-GPU and
    DDP write on rank 0 only, while FSDP gathers a full (unsharded) state_dict --
    a collective every rank must enter -- and writes it on rank 0. The on-disk
    format is identical across all three paths so a checkpoint loads the same way
    regardless of how it was trained."""
    is_rank0 = int(ddp["rank"]) == 0

    # FSDP: gather the full state_dict on all ranks (offloaded to CPU, populated
    # only on rank 0), then write on rank 0. Keys come back with their original
    # (unwrapped) names because the model was wrapped with use_orig_params=True,
    # so the checkpoint is interchangeable with the DDP/single-GPU form.
    if _is_fsdp(policy.model):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        gather_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(policy.model, StateDictType.FULL_STATE_DICT, gather_cfg):
            model_state = policy.model.state_dict()
        if not is_rank0:
            return
        _write_entity_checkpoint(
            policy,
            path,
            model_state,
            mask_hidden_info,
            soft_target_source=soft_target_source,
            value_training=value_training,
        )
        return

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if not ddp["enabled"]:
        try:
            if getattr(policy, "policy_type", "") == "entity_graph":
                policy.save(
                    tmp,
                    mask_hidden_info=bool(mask_hidden_info),
                    soft_target_source=soft_target_source,
                    value_training=value_training,
                )
            else:
                policy.save(tmp)
            if not tmp.exists() or tmp.stat().st_size <= 0:
                raise RuntimeError(f"checkpoint temp file was not written: {tmp}")
            os.replace(tmp, output)
        finally:
            if tmp.exists():
                tmp.unlink()
        return

    # DDP: only rank 0 writes; other ranks return (state_dict() here is a local
    # non-collective call, so this guard is safe even when every rank calls in).
    if not is_rank0:
        return
    _write_entity_checkpoint(
        policy,
        path,
        policy.model.module.state_dict(),
        mask_hidden_info,
        soft_target_source=soft_target_source,
        value_training=value_training,
    )


def _write_entity_checkpoint(
    policy,
    path: str,
    model_state: dict,
    mask_hidden_info: bool,
    *,
    soft_target_source: str | None = None,
    value_training: dict[str, object] | None = None,
) -> None:
    """Atomically write the durable name-keyed checkpoint dict shared by the DDP
    and FSDP save paths (mirrors EntityGraphPolicy.save's fields)."""
    import torch

    from catan_zero.rl.config_serialization import config_to_dict

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        payload = {
                "policy_type": getattr(policy, "policy_type", "xdim_lite"),
                "config": config_to_dict(policy.config),
                "action_mask_version": str(getattr(policy.config, "action_mask_version", "")),
                "mask_hidden_info": bool(mask_hidden_info),
                # OPT-8 provenance (mirrors EntityGraphPolicy.save).
                "soft_target_source": str(soft_target_source) if soft_target_source is not None else "",
                "static_action_features_sha256": _array_sha256(
                    policy.static_action_features.detach().cpu().numpy()
                ),
                "static_action_features": policy.static_action_features.detach().cpu(),
                "model": model_state,
            }
        if value_training is not None:
            durable_value_training = dict(value_training)
            payload["value_training"] = durable_value_training
            payload["trained_value_readouts"] = [
                str(readout)
                for readout in durable_value_training.get(
                    "trained_value_readouts", ()
                )
                if str(readout) in {"scalar", "categorical"}
            ]
        torch.save(payload, tmp)
        if not tmp.exists() or tmp.stat().st_size <= 0:
            raise RuntimeError(f"checkpoint temp file was not written: {tmp}")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()


if __name__ == "__main__":
    main()
