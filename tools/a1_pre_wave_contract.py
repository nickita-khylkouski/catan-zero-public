#!/usr/bin/env python3
"""Freeze, verify, render, and postflight-audit sealed A1 data handoffs.

This tool is deliberately *not* a launcher.  It turns the winning bounded-R&D
artifacts into an immutable contract and renders argv/environment records for
the data-production lane.  There is no subprocess/exec path: every legacy or
dual-arm wave remains an explicit operator boundary.

The contract uses category-specific jobs rather than a probabilistic opponent
mix. Historical v2 locks retain their original 40-worker 240/45/15 layout.
Current v3 locks bind the canonical 64-GPU fleet.  The original profile keeps
the exact 9,600/1,800/600 science totals; the scale profile binds exactly
800/150/50 selected games on every GPU (64,000 total) without rewriting issued
locks.  Both use deterministic, sealed quotas.
Every job receives a bounded category-specific reserve, and postflight selects
the lowest-seed complete games before row expansion. The audit rejects an
insufficient complete quota, duplicate or VAL-ONLY seeds, invalid selected
actions, config drift, and missing shard provenance.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TOOLS = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig  # noqa: E402
from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig  # noqa: E402
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    ENTITY_FEATURE_ADAPTER_SPECS,
    require_known_entity_feature_adapter,
)
from catan_zero.rl.gumbel_self_play import (  # noqa: E402
    GumbelSelfPlayConfig,
    PLAYER_NAMES,
    TARGET_INFORMATION_REGIME_PUBLIC,
    _pool_champion_plays_first_seat,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
)
from catan_zero.rl.pipeline_configs import GenerateConfig  # noqa: E402
from tools import a1_frozen_lock_verifier as frozen_lock_verifier  # noqa: E402
from tools import audit_entity_graph_information_surface as information_surface  # noqa: E402
from tools import generate_gumbel_selfplay_data as generation_cli  # noqa: E402
from tools import legacy_scalar_readout_attestation as legacy_scalar  # noqa: E402
from tools import a0_binding_verdict as a0_binding  # noqa: E402
from tools import search_teacher_adjudicator as search_adjudicator  # noqa: E402
from tools import search_operator_binding as operator_binding  # noqa: E402
from tools import a1_post_promotion_handoff as promotion_handoff  # noqa: E402
from tools.prelaunch_guard import VAL_ONLY_SEED_RANGE, parse_seed_ledger  # noqa: E402
from tools.seed_fleet_planner import assert_disjoint_seed_blocks  # noqa: E402
from tools.sprt_gate import evaluate_pentanomial_sprt  # noqa: E402

DRAFT_SCHEMA = "a1-pre-wave-contract-draft-v3"
LEGACY_DRAFT_SCHEMA = "a1-pre-wave-contract-draft-v2"
LOCK_SCHEMA = "a1-pre-wave-contract-lock-v3"
LEGACY_LOCK_SCHEMA = "a1-pre-wave-contract-lock-v2"
RENDER_SCHEMA = "a1-pre-wave-render-v2"
MPS_PIPE_DIRECTORY = "/tmp/mps_pipe_host"
MPS_LOG_DIRECTORY = "/tmp/mps_log_host"
CONFIG_REGISTRY_ENVIRONMENT_VARIABLE = "CATAN_ZERO_CONFIG_REGISTRY"
CONFIG_REGISTRY_FILENAME = "config_registry.jsonl"
RUNTIME_REPO_TOKEN = "__A1_RUNTIME_REPO__"
SEALED_RUNTIME_ENVIRONMENT = {
    "HOME": "/home/ubuntu",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": f"{RUNTIME_REPO_TOKEN}/src:{RUNTIME_REPO_TOKEN}",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
}
AUDIT_SCHEMA = "a1-post-wave-audit-v2"
RELOCATED_AUDIT_SCHEMA = "a1-post-wave-audit-v3"
DUAL_ARM_AUDIT_SCHEMA = "a1-dual-arm-post-wave-audit-v1"
DUAL_ARM_SELECTED_GAMES_SCHEMA = "a1-dual-arm-selected-training-games-v1"
HARVEST_RELOCATION_SCHEMA = "a1-fleet-harvest-relocation-v1"
GUARD_SYNC_SCHEMA = "a1-pre-wave-generation-guard-sync-v1"
CLAIM_RECEIPT_SCHEMA = "a1-seed-claim-transaction-v1"
GUARD_SYNC_KEY = "a1_pre_wave_guard_sync"
GUARD_SYNC_TOOL = "tools/a1_pre_wave_contract.py"
DEFAULT_GENERATION_C_SCALE = 0.03
UNRESOLVED = "__UNRESOLVED__"
EXPECTED_GAMES = {
    "current_producer": 9_600,
    "recent_history": 1_800,
    "hard_negative": 600,
}
# Historical v2 constants. They are intentionally immutable so an issued 40-GPU
# draft/lock can still reconstruct byte-for-byte.
EXPECTED_WORKER_COUNT = 40
EXPECTED_PER_WORKER = {
    "current_producer": 240,
    "recent_history": 45,
    "hard_negative": 15,
}
# A bounded reserve makes the selected zero-truncation quota achievable when
# otherwise healthy production has rare max-decision truncations.  Postflight
# deterministically selects the lowest-seed complete games from each job.
EXPECTED_ATTEMPTS_PER_WORKER = {
    "current_producer": 245,
    "recent_history": 47,
    "hard_negative": 16,
}
EXPECTED_ATTEMPTS = {
    category: attempts * EXPECTED_WORKER_COUNT
    for category, attempts in EXPECTED_ATTEMPTS_PER_WORKER.items()
}

# New v3 waves use the checked-in authoritative topology rather than copying a
# mutable worker list into an operator draft.  Quota remainders go to the first
# workers in the manifest's canonical host/GPU order; hashing the full manifest
# makes even an order-only edit a contract change.
CURRENT_FLEET_MANIFEST = REPO_ROOT / "configs" / "gpu_fleet_64.json"
CURRENT_FLEET_SCHEMA = "catan-gpu-fleet-v2"
CURRENT_FLEET_AUTHORITY = "catan-h100-exact64-v1"
CURRENT_WORKER_COUNT = 64
BALANCED_PREFIX_QUOTA_POLICY = "balanced_prefix_v1"
CURRENT_GAME_CONTRACT_PROFILE = "pre_wave_generation_v3"
# The scale wave is a new quota/profile identity, not a reinterpretation of the
# already-issued 12K v3 locks.  Keeping the same draft/lock envelope is safe
# because ``quota_policy`` and ``game_contract.profile`` are both hashed and
# verification reconstructs the exact profile selected by the source draft.
BALANCED_PER_LANE_64K_QUOTA_POLICY = "balanced_per_lane_64k_v1"
SCALE_64K_GAME_CONTRACT_PROFILE = "pre_wave_generation_v3_64k"
SCALE_64K_PER_WORKER_GAMES = {
    "current_producer": 800,
    "recent_history": 150,
    "hard_negative": 50,
}
SCALE_64K_GAMES = {
    category: games * CURRENT_WORKER_COUNT
    for category, games in SCALE_64K_PER_WORKER_GAMES.items()
}
AUTHORIZED_CURRENT_QUOTA_POLICIES = {
    BALANCED_PREFIX_QUOTA_POLICY,
    BALANCED_PER_LANE_64K_QUOTA_POLICY,
}
DUAL_ARM_GAME_CONTRACT_PROFILE = "dual_arm_generation_v1"
ATTEMPT_RESERVE_PER_JOB = {
    "current_producer": 5,
    "recent_history": 2,
    "hard_negative": 1,
}

# One deliberately short, single-B200 learner dose.  A search/data contract is
# not scientifically reproducible if the optimizer or loss mixture can drift
# after the wave, so these are effective values rather than a partial CLI
# overlay.  Keep the keys aligned with train_bc's resolved Namespace; the two
# derived topology fields make the batch semantics explicit.
EXPECTED_LEARNER_TRAINING_RECIPE: dict[str, Any] = {
    "track": "2p_no_trade",
    "vps_to_win": 10,
    "graph_history_features": True,
    "seed": 1,
    "epochs": 1,
    "max_steps": 0,
    "batch_size": 4096,
    "grad_accum_steps": 1,
    "world_size": 1,
    "global_batch_size": 4096,
    "optimizer": "adam",
    "resume_optimizer": False,
    "lr": 3e-5,
    "lr_warmup_steps": 100,
    "lr_schedule": "flat",
    "weight_decay": 0.0,
    "fused_optimizer": False,
    "value_lr_mult": 0.3,
    "action_module_lr_mult": 1.0,
    "trunk_lr_mult": 1.0,
    "policy_loss_weight": 1.0,
    "soft_target_source": "policy",
    "soft_target_weight": 0.9,
    "soft_target_temperature": 0.7,
    "soft_target_min_legal_coverage": 0.5,
    "value_loss_weight": 0.25,
    "value_target_lambda": 1.0,
    "value_categorical_loss_weight": 0.0,
    "hlgauss_scalar_aux_loss_weight": 0.0,
    "final_vp_loss_weight": 0.0,
    "q_loss_weight": 0.0,
    "policy_kl_anchor_weight": 0.0,
    "value_uncertainty_loss_weight": 0.0,
    "aux_subgoal_loss_weight": 0.0,
    "train_value_only": False,
    "freeze_modules": "",
    "policy_surprise_weight": 0.0,
    "advantage_policy_weighting": "none",
    "per_game_value_weight": False,
    "vp_margin_weight": 0.0,
    "truncated_vp_margin_value_weight": 0.25,
    "amp": "bf16",
    "mask_hidden_info": True,
    "symmetry_augment": False,
    "forced_action_weight": 0.1,
    "forced_row_value_weight": 1.0,
    "winner_sample_weight": 1.0,
    "loser_sample_weight": 1.0,
    "teacher_weights": "",
    "phase_weights": "",
    "value_phase_weights": "",
    "ddp_shard_data": False,
}
# The sole issued markerless v2 lock predates ``trunk_lr_mult`` and used the
# earlier loser-row weight. Its raw lock hash authenticates this exact recipe;
# no newly sealed v2/v3 lock may inherit these archival values.
HISTORICAL_MARKERLESS_LEARNER_TRAINING_RECIPE: dict[str, Any] = {
    "track": "2p_no_trade",
    "vps_to_win": 10,
    "graph_history_features": True,
    "seed": 1,
    "epochs": 1,
    "max_steps": 0,
    "batch_size": 4096,
    "grad_accum_steps": 1,
    "world_size": 1,
    "global_batch_size": 4096,
    "optimizer": "adam",
    "resume_optimizer": False,
    "lr": 3e-5,
    "lr_warmup_steps": 100,
    "lr_schedule": "flat",
    "weight_decay": 0.0,
    "fused_optimizer": False,
    "value_lr_mult": 0.3,
    "action_module_lr_mult": 1.0,
    "policy_loss_weight": 1.0,
    "soft_target_source": "policy",
    "soft_target_weight": 0.9,
    "soft_target_temperature": 0.7,
    "soft_target_min_legal_coverage": 0.5,
    "value_loss_weight": 0.25,
    "value_target_lambda": 1.0,
    "value_categorical_loss_weight": 0.0,
    "hlgauss_scalar_aux_loss_weight": 0.0,
    "final_vp_loss_weight": 0.0,
    "q_loss_weight": 0.0,
    "policy_kl_anchor_weight": 0.0,
    "value_uncertainty_loss_weight": 0.0,
    "aux_subgoal_loss_weight": 0.0,
    "train_value_only": False,
    "freeze_modules": "",
    "policy_surprise_weight": 0.0,
    "advantage_policy_weighting": "none",
    "per_game_value_weight": False,
    "vp_margin_weight": 0.0,
    "truncated_vp_margin_value_weight": 0.25,
    "amp": "bf16",
    "mask_hidden_info": True,
    "symmetry_augment": False,
    "forced_action_weight": 0.1,
    "forced_row_value_weight": 1.0,
    "winner_sample_weight": 1.0,
    "loser_sample_weight": 0.3,
    "teacher_weights": "",
    "phase_weights": "",
    "value_phase_weights": "",
    "ddp_shard_data": False,
}
# Current post-promotion v3 learner semantics. Keep the historical constant
# above untouched so issued v2 locks reconstruct exactly.
CURRENT_LEARNER_TRAINING_RECIPE: dict[str, Any] = {
    **EXPECTED_LEARNER_TRAINING_RECIPE,
    # The matched B200 dose adjudication selected 128 global-batch updates;
    # issued v2 locks retain their historical uncapped one-epoch recipe above.
    "max_steps": 128,
    "amp": "none",
    "forced_action_weight": 0.0,
    "forced_row_value_weight": 1.0,
    "per_game_policy_weight": True,
    "per_game_policy_weight_mode": "equal",
    # DDP ranks must share one NumPy epoch order but use independent PyTorch
    # dropout streams after identical model construction/loading.
    "training_rng_rank_offset": True,
}
REQUIRED_EVIDENCE = {"a0", "s1", "s2", "s3"}
HISTORICAL_HANDOFF_MODE = "historical_pre_promotion"
POST_PROMOTION_HANDOFF_MODE = "post_promotion"
DISASTER_RECOVERY_HANDOFF_MODE = "disaster_recovery"
RECOVERY_REFERENCE_SEMANTIC = "recovery_reference"
RECOVERY_REFERENCE_RELATION = "safety_reference_unproven_predecessor"
HISTORICAL_V5_HANDOFF_DEFAULTS = {
    "gameplay_policy_aggregation": "mean_improved_policy",
    "sigma_reference_visits": None,
}
HISTORICAL_V5_HANDOFF_FINGERPRINT = {
    "checkpoint_sha256": "sha256:6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c",
    "handoff_file_sha256": "sha256:314d86a4860497a90d665a0d05e66a458f91f3a1e54d4315bb61e9014556b52a",
    "handoff_sha256": "sha256:0e815149264a024ed69aecfcadccea6647d24522c207ee945a4cc2d314ca5b9a",
    "producer_identity_sha256": "sha256:9ac746cb249706bb0682e8d677e44d93df954b4bbb5510d6f315432baa3b6316",
    "promotion_receipt_file_sha256": "sha256:ca592a265dea3e045d41d8e63f423b2e413e757d6900aa3ff5a9ea7364ae5083",
    "promotion_receipt_sha256": "sha256:7c0a3335ac40fd326364beba5f696887cac43f8d6b36ed20d239d310e75b3fca",
    "registry_version": 5,
}
HISTORICAL_V5_HANDOFF_COMPATIBILITY_SCHEMA = (
    "a1-v5-deployed-search-default-projection-v1"
)
HISTORICAL_V5_RUST_FEATURIZER_COMPATIBILITY_SCHEMA = (
    "a1-v5-rust-featurizer-parity-compatibility-v1"
)
RUST_FEATURIZER_PARITY_EVIDENCE_SCHEMA = "eval-rust-feature-b200-evidence-v1"
RUST_FEATURIZER_PARITY_EVIDENCE_PATH = (
    REPO_ROOT / "docs/evidence/EVAL_RUST_FEATURE_B200_20260711.json"
)
RUST_FEATURIZER_PARITY_EVIDENCE_SHA256 = (
    "sha256:a5619e4601acc90793ba708e6ab7a1316dbdad7b6de72b1114bb1dde947d7ff3"
)
RUST_FEATURIZER_PARITY_TESTED_COMMIT = "6cb878e"
GENERATION_CAMPAIGN_SCHEMA = "a1-dual-arm-generation-contract-v1"
GENERATION_CAMPAIGN_REVISION_SCHEMA = "a1-dual-arm-generation-contract-v2"
POST_PROMOTION_CAMPAIGN_SCHEMA = "a1-post-promotion-generation-campaign-v1"
POST_PROMOTION_CAMPAIGN_STATUS = "ready_post_promotion_pending_placement"
GENERATION_CAMPAIGN_CONTRACT_ID = "a1-dual-arm-n256-n128-56gpu-20260710-r1"
GENERATION_CAMPAIGN_CONTRACT_SHA256 = (
    "sha256:029d4370c031d967994055b74578306fe99e34ec653221db39e277e6d22c1f74"
)
GENERATION_CAMPAIGN_CONTRACT_PATH = (
    REPO_ROOT / "configs/operations/a1-dual-arm-56gpu-20260710/contract.json"
)
# Immutable v1 campaign schemas predate an in-payload implementation_commit.
# These commits are therefore part of the validator's canonical identity for
# the two exact historical byte streams below.  Never replace them with HEAD:
# replay must read the files the campaign actually bound, not today's tools.
GENERATION_CAMPAIGN_R1_IMPLEMENTATION_COMMIT = (
    "fb8088bd6e161eeebdd24da2b47875b02f191ffa"
)
HISTORICAL_DB1_IMPLEMENTATION_COMMIT = (
    "db1c8b158fb89fd2421d85fcaf1f44f398eaa364"
)
# The issued r1 campaign predates the native-hot-loop guard revision.  Its
# contract cannot be rewritten, and its guard records deliberately continue to
# name the production paths that were live when it was issued.  Keep exact
# copies of those two historical guard blobs next to the contract so semantic
# replay never reads newer bytes through the old path.  This allowlist is keyed
# by both original path and sealed digest; it is not a general fallback for
# missing or drifting files.
GENERATION_CAMPAIGN_R1_GUARD_SNAPSHOTS = {
    (
        "configs/guards/a1_generation_n128.json",
        "sha256:81020e447a3bc55fbc17b6cdcdc1c56187e3f7266cfb92526746aa067661e1b3",
    ): GENERATION_CAMPAIGN_CONTRACT_PATH.parent
    / "snapshots/guards/a1_generation_n128.json",
    (
        "configs/guards/a1_generation_n256.json",
        "sha256:9fa693ba1bd87a422010cd992ae2f1ec0b4c20863b1dd7ef29aa59082306931a",
    ): GENERATION_CAMPAIGN_CONTRACT_PATH.parent
    / "snapshots/guards/a1_generation_n256.json",
}
GENERATION_CAMPAIGN_R2_CONTRACT_ID = "a1-dual-arm-n256-n128-56gpu-20260711-r2"
GENERATION_CAMPAIGN_R2_CONTRACT_SHA256 = (
    "sha256:a9b89b3885041b9d6f61c211d86d3e22cfb17426fdf4b221958a56d0def12e51"
)
GENERATION_CAMPAIGN_R2_CONTRACT_PATH = (
    REPO_ROOT / "configs/operations/a1-dual-arm-56gpu-20260711-r2/contract.json"
)
GENERATION_PLACEMENT_SCHEMA = "a1-dual-arm-generation-placement-v1"
GENERATION_ARM_LOCK_SCHEMA = "a1-generation-arm-lock-v1"
GENERATION_JOB_ATTESTATION_SCHEMA = "a1-generation-job-attestation-v3"
LEGACY_GENERATION_JOB_ATTESTATION_SCHEMA = "a1-generation-job-attestation-v2"
GENERATION_CAMPAIGN_PENDING = "blocked_pending_post_promotion_handoff"
GENERATION_CAMPAIGN_CHECKPOINT_SHA256 = (
    "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4"
)
GENERATION_CAMPAIGN_SEED_FLOOR = 300_000_160_000
GENERATION_CAMPAIGN_R1_NEXT_SEED_FLOOR = 300_000_626_944
GENERATION_CAMPAIGN_R1_TRANSACTION_ID = "f2e99d219c8f4825a80e1aaab70cb254"
GENERATION_CAMPAIGN_R1_HANDOFF_SHA256 = (
    "sha256:40a1eece93a9346030da2d569fccbfeac5e567ca66618cec936c78f3a2a16e51"
)
GENERATION_CAMPAIGN_REVISION_IMPLEMENTATION_COMMIT = (
    "c179fe7f3ea314f675af9207275c78ee012a245b"
)
GENERATION_CAMPAIGN_R1_LOCKS = {
    "n128": {
        "contract_id": f"{GENERATION_CAMPAIGN_CONTRACT_ID}-n128",
        "contract_sha256": "sha256:ab874b87aeff6817568349aa661f2e875cb0da402a7fe1f63574e9e1fa81019d",
        "file_sha256": "sha256:dfee05c8dea9bc8ba0a0a82d1af9b785ba488f76b24bca49276faaf5779305b5",
    },
    "n256": {
        "contract_id": f"{GENERATION_CAMPAIGN_CONTRACT_ID}-n256",
        "contract_sha256": "sha256:0fa4af39b62e09151e49d9e06e147b6d929b9ab5e07db4e57ba81d5029c70333",
        "file_sha256": "sha256:88f56891af3c6fa0b3dcd6fee7027e422004f6ca116c8e00e6faa7d02a7c5990",
    },
}
HISTORICAL_DB1_CAMPAIGN_SHA256 = (
    "sha256:ceecfa414c006dbe37c34b7b8d1e2f27d4028779d3c49626b6b9562ebeb99153"
)
HISTORICAL_DB1_CAMPAIGN_FILE_SHA256 = (
    "sha256:27c893690c65c02a6cfb38560b92b402f0cb893474f5eafa8d540ad1f4cd1a82"
)
HISTORICAL_DB1_EXECUTOR_SHA256 = (
    "sha256:fb92619ce98b2381267ba83a4d32c77236c54444c17d5d3258f3bac6b6a27db3"
)
# Sole issued pre-promotion lock created before promotion_handoff was added.
# Compatibility is raw-file and semantic-identity bound; no other markerless
# v2 lock can acquire generation/training authority by copying its shape.
HISTORICAL_MARKERLESS_A1_LOCK = {
    "contract_id": "a1-infoset-n128-p4-12000games-20260710-r1",
    "contract_sha256": "sha256:c88cec355237f4526159650befb209ea3a8c2d095a32dd645fe04bd01d1c59c4",
    "lock_file_sha256": "sha256:8301c7547e1745812c69ca04934424755c7116eb5e221688abc58c1bcb7a3122",
    "source_draft_sha256": "sha256:ae4af7ba7df732137bca201198bdbef73a2500bebe42bc8cda118cfb082d10fe",
}
A0_EVIDENCE_SCHEMA = "a0-binding-verdict-v1"
SEARCH_STAGE_EVIDENCE_SCHEMA = "rl-rnd-stage-decision-v1"
REQUIRED_REPORTS = {
    "truncation",
    "forced_fraction",
    "phase_mix",
    "decision_index_mix",
    "legal_width",
    "target_entropy",
    "full_search_policy_mass",
}
REQUIRED_SELECTED_TELEMETRY_COLUMNS = {
    "is_forced",
    "used_full_search",
    "phase",
    "decision_index",
    "target_policy",
    "target_policy_mask",
}
REQUIRED_GENERATOR_CODE_SUFFIXES = {
    "tools/generate_gumbel_selfplay_data.py",
    "tools/fleet/systemd/nvidia-mps.service",
    "tools/opponent_mix_registry.py",
    "src/catan_zero/rl/gumbel_self_play.py",
    "src/catan_zero/rl/flywheel/opponent_mix.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/entity_token_features.py",
    "src/catan_zero/rl/hex_symmetry.py",
    "src/catan_zero/search/gumbel_chance_mcts.py",
    "src/catan_zero/search/neural_rust_mcts.py",
    "src/catan_zero/rl/pipeline_configs.py",
    "native/catanatron-rs/Cargo.toml",
    "native/catanatron-rs/WHEEL_SHA256SUMS",
    "native/catanatron-rs/python/Cargo.toml",
    "native/catanatron-rs/src/lib.rs",
    "native/catanatron-rs/python/src/lib.rs",
    "native/gumbel_mcts_rs/Cargo.toml",
    "native/gumbel_mcts_rs/Cargo.lock",
    "native/gumbel_mcts_rs/src/lib.rs",
    "native/gumbel_mcts_rs/src/python_binding.rs",
}
REQUIRED_LEARNER_CODE_SUFFIXES = {
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
# The issued 56-GPU campaign schema is immutable and predates the two native
# wheel inputs added to future draft/runtime closures. Keep its exact file-set
# contract separate instead of retroactively rewriting already-running data.
ISSUED_CAMPAIGN_GENERATOR_CODE_SUFFIXES = REQUIRED_GENERATOR_CODE_SUFFIXES - {
    "native/gumbel_mcts_rs/Cargo.lock",
    "native/gumbel_mcts_rs/src/python_binding.rs",
}
REQUIRED_RUNTIME_CODE_SUFFIXES = (
    REQUIRED_GENERATOR_CODE_SUFFIXES
    | REQUIRED_LEARNER_CODE_SUFFIXES
    | {
        "configs/runtime/a1_production_runtime.json",
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/action_mask.py",
        "src/catan_zero/search/rust_mcts.py",
        "vendor/catanatron/catanatron/catanatron/__init__.py",
        "vendor/catanatron/catanatron/catanatron/models/board.py",
        "vendor/catanatron/catanatron/catanatron/models/enums.py",
        "vendor/catanatron/catanatron/catanatron/models/map.py",
        "vendor/catanatron/catanatron/catanatron/models/player.py",
    }
)

_SEARCH_INPUT_KEYS = {
    "max_depth",
    "c_visit",
    "c_scale",
    "prior_temperature",
    "n_full",
    "n_fast",
    "p_full",
    "n_full_wide",
    "n_full_wide_threshold",
    "wide_roots_always_full",
    "raw_policy_above_width",
    "symmetry_averaged_eval",
    "symmetry_averaged_eval_threshold",
    "wide_candidates_threshold",
    "correct_rust_chance_spectra",
    "lazy_interior_chance",
    "exact_budget_sh",
    "exact_budget_sh_min_n",
    "belief_chance_spectra",
    "information_set_search",
    "determinization_particles",
    "determinization_min_simulations",
    "rescale_noise_floor_c",
    "sigma_eval",
}
# Frozen explicit-operator shape in the sole issued markerless v2 lock. Keep
# this independent of the expanding current dataclass/input schema.
HISTORICAL_MARKERLESS_SEARCH_INPUT_KEYS = frozenset(
    {
        "max_depth",
        "c_visit",
        "c_scale",
        "prior_temperature",
        "n_full",
        "n_fast",
        "p_full",
        "n_full_wide",
        "n_full_wide_threshold",
        "wide_roots_always_full",
        "raw_policy_above_width",
        "symmetry_averaged_eval",
        "symmetry_averaged_eval_threshold",
        "wide_candidates_threshold",
        "correct_rust_chance_spectra",
        "lazy_interior_chance",
        "exact_budget_sh",
        "exact_budget_sh_min_n",
        "belief_chance_spectra",
        "information_set_search",
        "determinization_particles",
        "determinization_min_simulations",
        "rescale_noise_floor_c",
        "sigma_eval",
    }
)
_EVALUATOR_INPUT_KEYS = {
    "value_scale",
    "prior_temperature",
    "context_fill",
    "cache_size",
    "value_squash",
    "value_readout",
    "public_observation",
    "rust_featurize",
    "emit_uncertainty",
}
_GENERATION_KEYS = {
    "track",
    "vps_to_win",
    "obs_width",
    "max_decisions",
    "temperature_decisions",
    "temperature_high",
    "temperature_low",
    "late_temperature_decisions",
    "late_temperature",
    "workers_per_gpu",
    "shard_size",
    "format",
    "device",
    "eval_server",
    "native_mcts_hot_loop",
}


class ContractError(ValueError):
    """A fail-closed contract validation error."""


def _sealed_game_contract_shape(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Return topology only when the lock schema binds it unambiguously."""

    schema = lock.get("schema_version")
    game = lock.get("game_contract")
    fleet = lock.get("fleet")
    if not isinstance(game, dict) or not isinstance(fleet, dict):
        raise ContractError(
            "sealed lock has no typed game/fleet contract; historical locks require "
            "the explicit promotion handoff attestation path"
        )
    if schema == GENERATION_ARM_LOCK_SCHEMA:
        if (
            game.get("profile") != DUAL_ARM_GAME_CONTRACT_PROFILE
            or game.get("arm_id") not in {"n128", "n256"}
            or game.get("worker_count") != 28
            or game.get("job_count") != 84
        ):
            raise ContractError("dual-arm sealed topology drift")
        profile = DUAL_ARM_GAME_CONTRACT_PROFILE
        arm_id: str | None = str(game["arm_id"])
        worker_count = 28
        job_count = 84
    elif schema == LOCK_SCHEMA:
        if (
            game.get("profile")
            not in {CURRENT_GAME_CONTRACT_PROFILE, SCALE_64K_GAME_CONTRACT_PROFILE}
            or game.get("worker_count") != CURRENT_WORKER_COUNT
            or game.get("job_count")
            != CURRENT_WORKER_COUNT * len(EXPECTED_GAMES)
            or "arm_id" in game
        ):
            raise ContractError("current v3 sealed topology/profile drift")
        profile = str(game["profile"])
        arm_id = None
        worker_count = CURRENT_WORKER_COUNT
        job_count = CURRENT_WORKER_COUNT * len(EXPECTED_GAMES)
    elif schema == LEGACY_LOCK_SCHEMA:
        if any(key in game for key in ("profile", "arm_id", "worker_count", "job_count")):
            raise ContractError("historical v2 topology fields drift")
        profile = "historical_pre_wave_v2"
        arm_id = None
        worker_count = EXPECTED_WORKER_COUNT
        job_count = EXPECTED_WORKER_COUNT * len(EXPECTED_GAMES)
    else:
        raise ContractError("lock schema has no authorized game topology")

    jobs = fleet.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != job_count:
        raise ContractError("sealed job count differs from game contract")
    worker_ids: set[str] = set()
    placements: set[tuple[str, int]] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ContractError("sealed job topology entry is not an object")
        worker_id = job.get("worker_id")
        host_alias = job.get("host_alias")
        gpu = job.get("gpu")
        if (
            not isinstance(worker_id, str)
            or not worker_id
            or not isinstance(host_alias, str)
            or not host_alias
            or isinstance(gpu, bool)
            or not isinstance(gpu, int)
            or gpu < 0
        ):
            raise ContractError("sealed job placement fields are malformed")
        worker_ids.add(worker_id)
        placements.add((host_alias, gpu))
    if (
        len(worker_ids) != worker_count
        or len(placements) != worker_count
    ):
        raise ContractError("sealed physical lane count differs from game contract")
    return {
        "profile": profile,
        "arm_id": arm_id,
        "worker_count": worker_count,
        "job_count": job_count,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _is_historical_markerless_a1_lock(path: Path, lock: Mapping[str, Any]) -> bool:
    source = lock.get("source_draft")
    return bool(
        lock.get("schema_version") == LEGACY_LOCK_SCHEMA
        and "promotion_handoff" not in lock
        and lock.get("contract_id")
        == HISTORICAL_MARKERLESS_A1_LOCK["contract_id"]
        and lock.get("contract_sha256")
        == HISTORICAL_MARKERLESS_A1_LOCK["contract_sha256"]
        and path.is_file()
        and _sha256(path) == HISTORICAL_MARKERLESS_A1_LOCK["lock_file_sha256"]
        and isinstance(source, dict)
        and source.get("sha256")
        == HISTORICAL_MARKERLESS_A1_LOCK["source_draft_sha256"]
    )


def _issued_r1_guard_snapshot(relative_path: str, digest: str) -> Path | None:
    """Return an exact checked-in r1 guard blob, or fail closed.

    The returned file must still hash to the digest in the issued contract.
    A modified/missing snapshot is therefore no more trusted than a modified
    live guard, and unknown path/digest pairs never receive archival treatment.
    """

    snapshot = GENERATION_CAMPAIGN_R1_GUARD_SNAPSHOTS.get(
        (relative_path, digest)
    )
    if snapshot is None or not snapshot.is_file() or _sha256(snapshot) != digest:
        return None
    return snapshot


def _git_blob(commit: str, relative_path: str) -> bytes:
    relative = PurePath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ContractError(f"invalid repository path: {relative_path}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError("generation revision implementation commit is malformed")
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative_path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ContractError(
            f"cannot recover {relative_path} from implementation commit {commit}"
        ) from error


def _historical_campaign_provenance_bytes(
    commit: str, relative_path: str, digest: str
) -> bytes:
    """Return one historical campaign blob from its sole sealed authority.

    The two revised r1 guards have checked-in, digest-keyed snapshots because
    their semantics must remain parseable even in a source archive.  Every
    other record must exist at the campaign's exact implementation commit.
    Searching arbitrary history is intentionally forbidden: an unrelated old
    blob with the same path is not authority for this campaign.
    """

    snapshot_key = (relative_path, digest)
    if snapshot_key in GENERATION_CAMPAIGN_R1_GUARD_SNAPSHOTS:
        snapshot = _issued_r1_guard_snapshot(relative_path, digest)
        if snapshot is None:
            raise ContractError(
                "generation campaign immutable snapshot drift: "
                f"{relative_path}"
            )
        payload = snapshot.read_bytes()
    else:
        payload = _git_blob(commit, relative_path)
    if _sha256_bytes(payload) != digest:
        raise ContractError(
            "generation campaign implementation blob drift: "
            f"{commit}:{relative_path}"
        )
    return payload


def _campaign_historical_implementation_commit(
    path: Path,
    value: Mapping[str, Any],
    *,
    historical_lock_source: bool = False,
) -> str | None:
    """Resolve the exact source authority for an immutable campaign."""

    schema = value.get("schema_version")
    if schema == GENERATION_CAMPAIGN_REVISION_SCHEMA:
        commit = str(value.get("implementation_commit", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ContractError("generation revision implementation commit is malformed")
        return commit
    if schema != GENERATION_CAMPAIGN_SCHEMA:
        return None
    if historical_lock_source:
        return HISTORICAL_DB1_IMPLEMENTATION_COMMIT
    if (
        path.resolve() == GENERATION_CAMPAIGN_CONTRACT_PATH.resolve()
        and value.get("contract_id") == GENERATION_CAMPAIGN_CONTRACT_ID
        and value.get("contract_sha256") == GENERATION_CAMPAIGN_CONTRACT_SHA256
    ):
        return GENERATION_CAMPAIGN_R1_IMPLEMENTATION_COMMIT
    return None


def _generation_revision_provenance(commit: str) -> dict[str, Any]:
    guards = [
        "configs/guards/a1_generation_n256.json",
        "configs/guards/a1_generation_n128.json",
        "configs/guards/a1_generation_n256_legacy.json",
        "configs/guards/a1_generation_n128_legacy.json",
    ]
    generator = sorted(
        REQUIRED_GENERATOR_CODE_SUFFIXES
        | {
            "tools/launcher_guards.py",
            "tools/prelaunch_guard.py",
            "tools/fleet/a1_lane_supervisor.py",
        }
    )

    def record(path: str) -> dict[str, str]:
        return {"path": path, "sha256": _sha256_bytes(_git_blob(commit, path))}

    return {
        "arm_guards": [record(path) for path in guards],
        "generator_code": [record(path) for path in generator],
        "executor": record("tools/fleet/a1_production_executor.py"),
        "harvest": record("tools/fleet/a1_harvest_transaction.py"),
        "fleet_manifest": record("configs/gpu_fleet_56.json"),
    }


def _strict_config_registry_record(
    payload: bytes, *, expected: Mapping[str, Any], where: str
) -> dict[str, Any]:
    """Validate one job-private registry record against exact typed provenance."""

    try:
        text = payload.decode("utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{where}: invalid config registry: {error}") from error
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ContractError(f"{where}: config registry must contain exactly one object")
    record = records[0]
    required = {
        "config_hash",
        "full_config_hash",
        "pipeline",
        "timestamp",
        "purpose",
        "config",
    }
    if set(record) != required:
        raise ContractError(f"{where}: config registry record fields drift")
    if not isinstance(record.get("timestamp"), str) or not isinstance(
        record.get("purpose"), str
    ):
        raise ContractError(f"{where}: config registry metadata types drift")
    config = record.get("config")
    if not isinstance(config, dict):
        raise ContractError(f"{where}: config registry canonical payload is absent")
    full_hash = _digest_value(config)
    short_hash = "sha256:" + full_hash.removeprefix("sha256:")[:16]
    if (
        record.get("pipeline") != "generate"
        or config.get("pipeline") != "generate"
        or record.get("full_config_hash") != full_hash
        or record.get("config_hash") != short_hash
    ):
        raise ContractError(f"{where}: config registry record is not self-consistent")
    for key in ("pipeline", "config_hash", "full_config_hash", "config"):
        if record.get(key) != expected.get(key):
            raise ContractError(f"{where}: config registry {key} differs from sealed config")
    return record


def _read_sealed_regular(path: Path, *, where: str) -> bytes:
    """Read one immutable regular file once, rejecting symlinks and write bits."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"{where}: cannot open sealed file {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o222:
            raise ContractError(f"{where}: file is not a read-only regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        payload = b"".join(chunks)
        os.lseek(descriptor, 0, os.SEEK_SET)
        verification_chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            verification_chunks.append(chunk)
        if b"".join(verification_chunks) != payload:
            raise ContractError(f"{where}: sealed file mutated during read: {path}")
        after = os.fstat(descriptor)
        before_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(payload) != after.st_size:
            raise ContractError(f"{where}: sealed file mutated during read: {path}")
        current = os.stat(path, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ContractError(f"{where}: sealed file changed during inspection: {path}")
        return payload
    finally:
        os.close(descriptor)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - compatibility identity, SHA-256 is authoritative.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _absolute_ref(raw: str, *, base: Path) -> Path:
    path = Path(raw).expanduser()
    candidate = base / path if not path.is_absolute() else path
    # ``absolute()`` preserves ``..`` components and makes a logically correct
    # checked-in relative path fail later canonical-path provenance checks.
    # resolve(strict=False) normalizes traversal/symlinks while downstream
    # file-specific validators still produce the useful missing-file error.
    return candidate.resolve(strict=False)


def _require_exact_keys(
    value: dict[str, Any], allowed: set[str], *, where: str
) -> None:
    extra = set(value) - allowed
    missing = allowed - set(value)
    if extra or missing:
        raise ContractError(
            f"{where} fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _authenticated_rust_featurizer_parity_evidence() -> dict[str, Any]:
    """Replay the sole checked-in Python/Rust feature-equivalence receipt.

    Rust featurization is execution machinery, but the historical v5 agent
    identity recorded the old Python implementation as a boolean field.  A
    new v3 wave may cross that implementation boundary only through the exact
    B200 parity receipt checked into this repository.  Pinning its bytes and
    replaying its proof projection prevents a similarly named, hand-authored,
    or partially passing JSON file from authorizing the transition.
    """

    path = RUST_FEATURIZER_PARITY_EVIDENCE_PATH.expanduser().absolute()
    try:
        metadata = path.lstat()
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise ContractError(
            f"Rust-featurizer parity evidence is unavailable: {error}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or canonical != path:
        raise ContractError(
            "Rust-featurizer parity evidence must be a canonical regular file"
        )
    record = _file_record(path, kind="rust_featurizer_parity_evidence")
    if record["sha256"] != RUST_FEATURIZER_PARITY_EVIDENCE_SHA256:
        raise ContractError("Rust-featurizer parity evidence fingerprint mismatch")
    payload = _load_json(path)
    parity = payload.get("parity")
    output = payload.get("real_checkpoint_output_parity")
    if (
        payload.get("schema") != RUST_FEATURIZER_PARITY_EVIDENCE_SCHEMA
        or payload.get("commit") != RUST_FEATURIZER_PARITY_TESTED_COMMIT
        or not isinstance(parity, dict)
        or parity.get("passed") != 26
        or parity.get("failed") != 0
        or parity.get("skipped") != 0
        or not isinstance(parity.get("coverage"), str)
        or "public-observation and omniscient modes" not in parity["coverage"]
        or not isinstance(output, dict)
        or output.get("states") != 128
        or output.get("exact_output_states") != 128
        or output.get("max_abs_prior_diff") != 0.0
        or output.get("max_abs_value_diff") != 0.0
    ):
        raise ContractError("Rust-featurizer parity evidence semantic replay failed")
    record.update(
        {
            "document_schema": payload["schema"],
            "document_digest": _digest_value(payload),
            "tested_commit": payload["commit"],
            "tensor_parity_tests": {
                "passed": parity["passed"],
                "failed": parity["failed"],
                "skipped": parity["skipped"],
            },
            "real_checkpoint_output_parity": {
                "states": output["states"],
                "exact_output_states": output["exact_output_states"],
                "max_abs_prior_diff": output["max_abs_prior_diff"],
                "max_abs_value_diff": output["max_abs_value_diff"],
            },
        }
    )
    return record


def _find_unresolved(value: Any, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if value == UNRESOLVED:
        found.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_find_unresolved(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_unresolved(child, path=f"{path}[{index}]"))
    return found


def _assert_no_unresolved(value: Any) -> None:
    unresolved = _find_unresolved(value)
    if unresolved:
        preview = ", ".join(unresolved[:12])
        raise ContractError(
            f"contract still has unresolved science fields ({preview}); finish A0/S1-S3 "
            "and replace every __UNRESOLVED__ value before seal/render"
        )


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _file_record(
    path: Path, *, kind: str, artifact_id: str | None = None
) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"{kind} is missing or not a file: {path}")
    record = {"kind": kind, "path": str(path), "sha256": _sha256(path)}
    if artifact_id is not None:
        record["id"] = artifact_id
    return record


def _historical_v5_handoff_identity_compatibility(
    *,
    path: Path,
    payload: Mapping[str, Any],
    deployed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and normalize the sole issued 48-key v5 identity.

    The committed v5 receipt predates two fields that later became explicit in
    the promotion identity projection. Both values were already the runtime
    defaults. The one parity-certified Python-to-Rust featurizer transition is
    implementation-only and is authenticated separately below. This is not a
    subset comparison: only the exact issued artifact, exact two-key omission,
    exact defaults, and that one evidence-bound implementation change qualify.
    """

    missing = set(expected) - set(deployed)
    extra = set(deployed) - set(expected)
    shared_drift = {
        key: (deployed[key], expected[key])
        for key in set(deployed) & set(expected)
        if deployed[key] != expected[key]
    }
    rust_featurizer_transition = bool(
        set(shared_drift) == {"evaluator_rust_featurize"}
        and deployed.get("evaluator_rust_featurize") is False
        and expected.get("evaluator_rust_featurize") is True
    )
    required_missing = set(HISTORICAL_V5_HANDOFF_DEFAULTS)
    if (
        missing != required_missing
        or extra
        or (shared_drift and not rust_featurizer_transition)
    ):
        raise ContractError(
            "promoted deployed search identity compatibility shape drift: "
            f"missing={sorted(missing)}, extra={sorted(extra)}, "
            f"shared_drift={shared_drift}"
        )

    receipt = payload.get("promotion_receipt")
    identity = payload.get("producer_identity")
    registry = payload.get("registry_after")
    if not all(isinstance(value, Mapping) for value in (receipt, identity, registry)):
        raise ContractError("historical v5 handoff compatibility lineage is malformed")
    checkpoint = identity.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ContractError("historical v5 handoff checkpoint lineage is malformed")
    actual_fingerprint = {
        "checkpoint_sha256": checkpoint.get("sha256"),
        "handoff_file_sha256": _sha256(path),
        "handoff_sha256": payload.get("handoff_sha256"),
        "producer_identity_sha256": identity.get("agent_identity_sha256"),
        "promotion_receipt_file_sha256": receipt.get("sha256"),
        "promotion_receipt_sha256": receipt.get("receipt_sha256"),
        "registry_version": registry.get("version"),
    }
    if actual_fingerprint != HISTORICAL_V5_HANDOFF_FINGERPRINT:
        raise ContractError(
            "historical v5 handoff compatibility fingerprint mismatch"
        )

    runtime_defaults = {
        "gameplay_policy_aggregation": GumbelChanceMCTSConfig().gameplay_policy_aggregation,
        "sigma_reference_visits": GumbelChanceMCTSConfig().sigma_reference_visits,
    }
    if runtime_defaults != HISTORICAL_V5_HANDOFF_DEFAULTS:
        raise ContractError("historical v5 runtime search defaults drifted")
    projected_defaults = {key: expected[key] for key in required_missing}
    if projected_defaults != HISTORICAL_V5_HANDOFF_DEFAULTS:
        raise ContractError(
            "historical v5 handoff omitted fields do not resolve to exact defaults"
        )
    normalized = {**deployed, **HISTORICAL_V5_HANDOFF_DEFAULTS}
    parity_evidence: dict[str, Any] | None = None
    if rust_featurizer_transition:
        parity_evidence = _authenticated_rust_featurizer_parity_evidence()
        normalized["evaluator_rust_featurize"] = True
    if normalized != dict(expected):
        raise ContractError(
            "historical v5 handoff compatibility normalization is not exact"
        )
    compatibility = {
        "schema_version": (
            HISTORICAL_V5_RUST_FEATURIZER_COMPATIBILITY_SCHEMA
            if parity_evidence is not None
            else HISTORICAL_V5_HANDOFF_COMPATIBILITY_SCHEMA
        ),
        "authenticated_fingerprint": actual_fingerprint,
        "omitted_historical_defaults": dict(HISTORICAL_V5_HANDOFF_DEFAULTS),
        "raw_deployed_search_config_sha256": _digest_value(deployed),
        "normalized_search_config_sha256": _digest_value(normalized),
    }
    if parity_evidence is not None:
        compatibility["rust_featurizer_implementation_transition"] = {
            "from": "python_entity_features_v1",
            "to": "rust_entity_features_parity_v1",
            "semantic_identity_changed": False,
            "parity_evidence": parity_evidence,
        }
    return compatibility


def _promotion_handoff_record(
    raw: Any,
    *,
    base: Path,
    producer: dict[str, Any],
    effective_search: dict[str, Any] | None = None,
    evaluator: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize and replay the producer lineage selected for this wave."""

    if not isinstance(raw, dict):
        raise ContractError("promotion_handoff must be an object")
    mode = raw.get("mode")
    if mode == HISTORICAL_HANDOFF_MODE:
        _require_exact_keys(raw, {"mode", "reason"}, where="promotion_handoff")
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ContractError(
                "historical pre-promotion contracts require a non-empty reason"
            )
        return {"mode": HISTORICAL_HANDOFF_MODE, "reason": reason}
    if mode == DISASTER_RECOVERY_HANDOFF_MODE:
        _require_exact_keys(raw, {"mode", "path"}, where="promotion_handoff")
        path = _absolute_ref(str(raw.get("path", "")), base=base)
        from tools import a1_v5_disaster_recovery as recovery

        try:
            verified = recovery.verify_committed_receipt(path)
        except recovery.RecoveryError as error:
            raise ContractError(f"disaster-recovery receipt replay failed: {error}") from error
        authority = verified["authority"]
        receipt = verified["receipt"]
        recovered = authority.get("recovered_generator")
        identity = authority.get("producer_identity")
        safety = authority.get(RECOVERY_REFERENCE_RELATION)
        if not all(isinstance(item, dict) for item in (recovered, identity, safety)):
            raise ContractError("disaster-recovery authority lineage is malformed")
        if (
            recovered.get("path") != producer.get("path")
            or recovered.get("sha256") != producer.get("sha256")
            or identity.get("checkpoint")
            != {"path": producer.get("path"), "sha256": producer.get("sha256")}
        ):
            raise ContractError("draft producer is not the exact recovered generator")
        if (
            authority.get("promotion_proof_recreated") is not False
            or authority.get("wave_lineage_mode") != RECOVERY_REFERENCE_SEMANTIC
            or safety.get("relationship") != RECOVERY_REFERENCE_RELATION
            or safety.get("causal_parent_proven") is not False
        ):
            raise ContractError("disaster-recovery authority weakens evidence-loss policy")
        if effective_search is None or evaluator is None or generation is None:
            raise ContractError("disaster recovery requires complete generation science")
        identity_search = identity.get("search_config")
        if not isinstance(identity_search, dict):
            raise ContractError("recovered producer has no typed search config")
        try:
            from tools import a1_promotion_transaction as promotion

            expected_identity_search = promotion._sealed_evaluation_semantics(  # noqa: SLF001
                {
                    "science": {
                        "effective_search_config": effective_search,
                        "evaluator": evaluator,
                    },
                    "generation": generation,
                }
            )
        except promotion.PromotionError as error:
            raise ContractError(
                f"cannot project recovered producer identity: {error}"
            ) from error
        if effective_search.get("c_scale") != identity_search.get("c_scale"):
            raise ContractError(
                "generation c_scale differs from recovered deployed producer identity"
            )
        identity_compatibility: dict[str, Any] | None = None
        if identity_search != expected_identity_search:
            surviving_path = Path(
                str(receipt.get("surviving_handoff", {}).get("path", ""))
            ).resolve(strict=True)
            identity_compatibility = _historical_v5_handoff_identity_compatibility(
                path=surviving_path,
                payload=_load_json(surviving_path),
                deployed=identity_search,
                expected=expected_identity_search,
            )
        record = _file_record(path, kind="disaster_recovery_receipt")
        record.update(
            {
                "mode": DISASTER_RECOVERY_HANDOFF_MODE,
                "document_schema": recovery.RECOVERY_SCHEMA,
                "recovery_receipt_sha256": receipt["recovery_receipt_sha256"],
                "recovery_lineage_id": authority["recovery_lineage_id"],
                "registry_role": recovery.RECOVERED_GENERATOR_ROLE,
                "registry_version": int(
                    recovered["historical_generation_version_claim"]
                ),
                "producer_checkpoint": {
                    "path": recovered["path"],
                    "sha256": recovered["sha256"],
                },
                "producer_identity_sha256": identity["agent_identity_sha256"],
                "producer_search_config": dict(identity_search),
                "producer_search_config_sha256": _digest_value(identity_search),
                "safety_reference": dict(safety),
                "wave_lineage_mode": RECOVERY_REFERENCE_SEMANTIC,
                "promotion_proof_recreated": False,
            }
        )
        if identity_compatibility is not None:
            record["producer_search_identity_compatibility"] = identity_compatibility
        return record
    if mode != POST_PROMOTION_HANDOFF_MODE:
        raise ContractError(
            "promotion_handoff must explicitly select historical_pre_promotion "
            "post_promotion, or disaster_recovery"
        )
    _require_exact_keys(raw, {"mode", "path"}, where="promotion_handoff")
    path = _absolute_ref(str(raw.get("path", "")), base=base)
    payload = _load_json(path)
    if payload.get("schema_version") != promotion_handoff.HANDOFF_SCHEMA:
        raise ContractError("post-promotion handoff schema is not supported")
    declared = payload.get("handoff_sha256")
    unhashed = dict(payload)
    unhashed.pop("handoff_sha256", None)
    if declared != _digest_value(unhashed):
        raise ContractError("post-promotion handoff semantic digest mismatch")
    try:
        replayed = promotion_handoff.build_handoff(
            Path(str(payload.get("promotion_receipt", {}).get("path", "")))
        )
    except promotion_handoff.HandoffError as error:
        raise ContractError(f"post-promotion handoff replay failed: {error}") from error
    if payload != replayed:
        raise ContractError("post-promotion handoff differs from committed live lineage")
    registry_after = payload["registry_after"]
    if registry_after.get("role") != promotion_handoff.GENERATOR_ROLE:
        raise ContractError("post-promotion handoff role must be generator_champion")
    bound_checkpoint = registry_after.get("checkpoint")
    if not isinstance(bound_checkpoint, dict) or (
        bound_checkpoint.get("path") != producer.get("path")
        or bound_checkpoint.get("sha256") != producer.get("sha256")
    ):
        raise ContractError(
            "draft producer is not exactly the committed promoted producer"
        )
    identity_checkpoint = payload.get("producer_identity", {}).get("checkpoint")
    if identity_checkpoint != bound_checkpoint:
        raise ContractError("post-promotion producer identity lineage drift")
    if effective_search is None or evaluator is None or generation is None:
        raise ContractError("post-promotion handoff requires complete generation science")
    identity_search = payload.get("producer_identity", {}).get("search_config")
    if not isinstance(identity_search, dict):
        raise ContractError("post-promotion producer identity has no typed search config")
    # Promotion evaluates the deployed candidate with exactly four operational
    # overrides: every decision is full n128 (n_fast=n_full, p_full=1,
    # force_full_every_decision=true) and play temperature is zero.  Evaluator
    # fields and max_decisions are projected into the typed identity; every
    # other search semantic is inherited exactly.  In particular c_scale is
    # the deployed agent's identity and is *not* an allowed generation drift.
    try:
        from tools import a1_promotion_transaction as promotion

        sealed = promotion._sealed_evaluation_semantics(  # noqa: SLF001
            {
                "science": {
                    "effective_search_config": effective_search,
                    "evaluator": evaluator,
                },
                "generation": generation,
            }
        )
        # This is already a committed post-promotion identity. Its c_scale is
        # the producer's durable operator identity and must match the next
        # contract's generation science verbatim; never reclassify it through a
        # hardcoded candidate/champion role default.
        expected_identity_search = sealed
    except promotion.PromotionError as error:
        raise ContractError(f"cannot project promoted producer identity: {error}") from error
    if effective_search.get("c_scale") != identity_search.get("c_scale"):
        raise ContractError(
            "generation c_scale differs from promoted deployed producer identity"
        )
    identity_compatibility: dict[str, Any] | None = None
    if identity_search != expected_identity_search:
        identity_compatibility = _historical_v5_handoff_identity_compatibility(
            path=path,
            payload=payload,
            deployed=identity_search,
            expected=expected_identity_search,
        )
    record = _file_record(path, kind="post_promotion_handoff")
    record.update(
        {
            "mode": POST_PROMOTION_HANDOFF_MODE,
            "document_schema": payload["schema_version"],
            "handoff_sha256": payload["handoff_sha256"],
            "transaction_id": payload["promotion_receipt"]["transaction_id"],
            "registry_role": registry_after["role"],
            "registry_version": registry_after["version"],
            "producer_checkpoint": dict(bound_checkpoint),
            "producer_identity_sha256": payload["producer_identity"][
                "agent_identity_sha256"
            ],
            "producer_search_config": dict(identity_search),
            "producer_search_config_sha256": _digest_value(identity_search),
        }
    )
    if identity_compatibility is not None:
        record["producer_search_identity_compatibility"] = identity_compatibility
    return record


def _promoted_producer_job_identity(
    lock: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Bind one promoted producer to the identity-stable search semantics it runs.

    Search *budget* is a data-generation operator choice, so n128/n256 and PCR
    may legitimately differ from the promotion panel.  ``c_scale`` is not: it
    changes how the deployed agent combines its policy and value estimates and
    is part of the promoted ``agent_identity``.  Historically the dual-arm job
    table overrode that field to .03 for opponent jobs while the promoted f7
    producer was bound at .10.  The lock still named the base operator digest,
    so checkpoint and executed operator were never joined into one identity.

    Historical pre-promotion locks remain replayable.  Every post-promotion job
    must return this exact checkpoint/operator pair or fail before execution.
    """

    handoff = lock.get("promotion_handoff")
    if not isinstance(handoff, Mapping) or handoff.get("mode") not in {
        POST_PROMOTION_HANDOFF_MODE,
        DISASTER_RECOVERY_HANDOFF_MODE,
    }:
        return None
    if handoff.get("mode") == POST_PROMOTION_HANDOFF_MODE:
        if handoff.get("document_schema") != promotion_handoff.HANDOFF_SCHEMA:
            raise ContractError("promoted producer job handoff schema drift")
    else:
        from tools import a1_v5_disaster_recovery as recovery

        if handoff.get("document_schema") != recovery.RECOVERY_SCHEMA:
            raise ContractError("recovered producer job receipt schema drift")
    producer = _producer(dict(lock))
    bound_checkpoint = handoff.get("producer_checkpoint")
    if bound_checkpoint != {
        "path": producer.get("path"),
        "sha256": producer.get("sha256"),
    }:
        raise ContractError("promoted producer job checkpoint identity drift")
    deployed = handoff.get("producer_search_config")
    if not isinstance(deployed, Mapping) or not deployed:
        raise ContractError("promoted producer job lacks deployed search identity")
    search = lock.get("science", {}).get("search_operator")
    if not isinstance(search, Mapping):
        raise ContractError("promoted producer job lacks a typed search operator")
    executed = dict(search)
    if "c_scale" in job:
        executed["c_scale"] = float(job["c_scale"])
    if float(executed.get("c_scale", float("nan"))) != float(
        deployed.get("c_scale", float("nan"))
    ):
        raise ContractError(
            "promoted producer search identity mismatch: "
            f"job {job.get('job_id', '<unknown>')} executes c_scale="
            f"{executed.get('c_scale')!r}, promoted checkpoint is deployed at "
            f"c_scale={deployed.get('c_scale')!r}"
        )
    identity = {
        "checkpoint": {
            "path": producer["path"],
            "sha256": producer["sha256"],
        },
        "producer_identity_sha256": handoff.get("producer_identity_sha256"),
        "deployed_search_config_sha256": handoff.get(
            "producer_search_config_sha256"
        ),
        "executed_search_operator": executed,
        "executed_search_operator_sha256": _digest_value(executed),
    }
    identity["checkpoint_search_identity_sha256"] = _digest_value(identity)
    return identity


def _tracked_vendor_catanatron_runtime_paths(
    repo_root: Path = REPO_ROOT,
) -> set[Path]:
    """Return exactly the committed pure-Python engine used by generation.

    The sealed runtime imports Catanatron for the policy action catalog and
    canonical BASE-map topology even when game transitions and MCTS are native.
    Globbing is insufficient here: an untracked local module must never acquire
    launch authority merely because it exists in the operator checkout.
    """

    relative_root = Path("vendor/catanatron/catanatron/catanatron")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z", "--", str(relative_root)],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError("cannot enumerate tracked vendored Catanatron runtime") from error
    paths: set[Path] = set()
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        try:
            relative = Path(encoded.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ContractError("vendored Catanatron path is not UTF-8") from error
        if relative.suffix != ".py" or not relative.is_relative_to(relative_root):
            continue
        candidate = repo_root / relative
        try:
            metadata = candidate.lstat()
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ContractError(f"tracked vendored Catanatron file is missing: {relative}") from error
        if not stat.S_ISREG(metadata.st_mode) or resolved != candidate.absolute():
            raise ContractError(f"tracked vendored Catanatron file is not canonical: {relative}")
        paths.add(resolved)
    if not paths:
        raise ContractError("tracked vendored Catanatron runtime is empty")
    return paths


def _runtime_code_tree_records() -> list[dict[str, Any]]:
    """Content-address the complete local Python runtime used by gen/A1.

    Explicit role lists document the expected load-bearing entry points, but
    cannot safely approximate a transitive import closure.  Hashing every
    Python module under ``src/catan_zero`` and ``tools`` closes that gap; the
    two static guard configs are included because they alter launch semantics.
    """

    paths = {
        path.resolve(strict=True)
        for pattern in ("src/catan_zero/**/*.py", "tools/**/*.py")
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    }
    paths.update(_tracked_vendor_catanatron_runtime_paths())
    for relative in (
        "configs/runtime/a1_production_runtime.json",
        "native/catanatron-rs/Cargo.toml",
        "native/catanatron-rs/Cargo.lock",
        "native/catanatron-rs/pyproject.toml",
        "native/catanatron-rs/WHEEL_SHA256SUMS",
        "native/catanatron-rs/python/Cargo.toml",
        "native/catanatron-rs/python/Cargo.lock",
        "native/catanatron-rs/src/lib.rs",
        "native/catanatron-rs/python/src/lib.rs",
        "native/gumbel_mcts_rs/Cargo.toml",
        "native/gumbel_mcts_rs/Cargo.lock",
        "native/gumbel_mcts_rs/src/lib.rs",
        "native/gumbel_mcts_rs/src/python_binding.rs",
        "tools/fleet/systemd/nvidia-mps.service",
    ):
        paths.add((REPO_ROOT / relative).resolve(strict=True))
    paths.update(
        {
            (REPO_ROOT / "configs/guards/generate_gumbel_selfplay_data.json").resolve(
                strict=True
            ),
            (REPO_ROOT / "configs/guards/train_bc.json").resolve(strict=True),
        }
    )
    records = [_file_record(path, kind="runtime_code") for path in sorted(paths)]
    record_paths = {Path(record["path"]).as_posix() for record in records}
    missing = {
        suffix
        for suffix in REQUIRED_RUNTIME_CODE_SUFFIXES
        if not any(path.endswith(suffix) for path in record_paths)
    }
    if missing:
        raise ContractError(
            f"runtime code tree omits required transitive files: {sorted(missing)}"
        )
    return records


def _checkpoint_metadata(
    path: Path,
    *,
    checkpoint_sha256: str,
    value_readout: str,
    require_trained_readout: bool,
    legacy_scalar_attestation: Path | None,
) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - a checkpoint must be inspectable to seal.
        raise ContractError(f"cannot inspect checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"checkpoint {path} is not a mapping")
    if payload.get("mask_hidden_info") is not True:
        raise ContractError(
            f"checkpoint {path} does not attest mask_hidden_info=true; public-observation "
            "generation may not use it"
        )
    value_training = payload.get("value_training")
    positive_provenance = (
        isinstance(value_training, dict)
        and value_training.get("schema_version") == "value-training-v1"
        and value_readout
        in set(map(str, value_training.get("trained_value_readouts", [])))
    )
    if value_readout == "categorical":
        if legacy_scalar_attestation is not None:
            raise ContractError(
                "a legacy scalar-readout attestation cannot authorize categorical readout"
            )
        if not positive_provenance:
            raise ContractError(
                f"checkpoint {path} does not prove teacher readout 'categorical' was trained"
            )
        if (
            not isinstance(value_training, dict)
            or value_training.get("schema_version") != "value-training-v1"
        ):
            raise ContractError(
                f"checkpoint {path} lacks positive value-training-v1 provenance required "
                "for categorical readout"
            )
        if "categorical" not in set(
            map(str, value_training.get("trained_value_readouts", []))
        ):
            raise ContractError(
                f"checkpoint {path} categorical readout was requested but categorical training "
                "provenance is not positive"
            )
        if float(value_training.get("resolved_categorical_ce_weight", 0.0)) <= 0.0:
            raise ContractError(
                f"checkpoint {path} categorical CE weight is not positive"
            )
        if int(value_training.get("hlgauss_bins", 0)) != 33:
            raise ContractError(
                f"checkpoint {path} must attest the selected 33-bin HL-Gauss head"
            )
    elif positive_provenance:
        if float(value_training.get("resolved_scalar_mse_weight", 0.0)) <= 0.0:
            raise ContractError(f"checkpoint {path} scalar MSE weight is not positive")
    elif require_trained_readout and legacy_scalar_attestation is None:
        raise ContractError(
            f"checkpoint {path} does not prove teacher readout {value_readout!r} was trained; "
            "supply a typed legacy scalar-readout attestation"
        )

    attestation_record: dict[str, Any] | None = None
    if legacy_scalar_attestation is not None:
        if value_readout != "scalar":
            raise ContractError("legacy scalar-readout attestation is scalar-only")
        try:
            attestation = legacy_scalar.verify_attestation(
                legacy_scalar_attestation,
                expected_checkpoint_path=path,
                expected_checkpoint_sha256=checkpoint_sha256,
            )
        except legacy_scalar.AttestationError as error:
            raise ContractError(
                f"invalid legacy scalar-readout attestation: {error}"
            ) from error
        attestation_record = {
            "kind": "legacy_scalar_readout_attestation",
            "path": str(legacy_scalar_attestation),
            "sha256": _sha256(legacy_scalar_attestation),
            "schema_version": legacy_scalar.SCHEMA_VERSION,
            "attestation_sha256": attestation["attestation_sha256"],
            "checkpoint": dict(attestation["checkpoint"]),
            "report": dict(attestation["report"]),
        }
    record = {
        "mask_hidden_info": True,
        "value_training_schema": (
            "value-training-v1"
            if isinstance(value_training, dict)
            and value_training.get("schema_version") == "value-training-v1"
            else (
                legacy_scalar.SCHEMA_VERSION
                if attestation_record is not None
                else "legacy-scalar-unattested-opponent"
            )
        ),
    }
    if isinstance(value_training, dict):
        record["value_training_sha256"] = _digest_value(value_training)
    if attestation_record is not None:
        record["legacy_scalar_readout_attestation"] = attestation_record
    return record


def _validate_search_operator_fields(
    raw: dict[str, Any], *, expected_keys: set[str] | frozenset[str] = _SEARCH_INPUT_KEYS
) -> None:
    """Validate explicit operator fields without expanding runtime defaults."""

    _require_exact_keys(raw, set(expected_keys), where="science.search")
    bool_keys = {
        "wide_roots_always_full",
        "symmetry_averaged_eval",
        "correct_rust_chance_spectra",
        "lazy_interior_chance",
        "exact_budget_sh",
        "belief_chance_spectra",
        "information_set_search",
    }
    int_keys = {
        "max_depth",
        "n_full",
        "n_fast",
        "wide_candidates_threshold",
        "exact_budget_sh_min_n",
        "determinization_particles",
        "determinization_min_simulations",
    }
    optional_int_keys = {
        "n_full_wide",
        "n_full_wide_threshold",
        "raw_policy_above_width",
        "symmetry_averaged_eval_threshold",
    }
    numeric_keys = {
        "c_visit",
        "c_scale",
        "prior_temperature",
        "p_full",
        "rescale_noise_floor_c",
        "sigma_eval",
    }
    for key in bool_keys:
        if type(raw[key]) is not bool:
            raise ContractError(f"science.search.{key} must be a JSON boolean")
    for key in int_keys:
        if type(raw[key]) is not int:
            raise ContractError(f"science.search.{key} must be an integer")
    for key in optional_int_keys:
        if raw[key] is not None and type(raw[key]) is not int:
            raise ContractError(f"science.search.{key} must be an integer or null")
    for key in numeric_keys:
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
            raise ContractError(f"science.search.{key} must be numeric")


def _effective_search(raw: dict[str, Any]) -> dict[str, Any]:
    _validate_search_operator_fields(raw)
    config = GumbelChanceMCTSConfig(colors=("RED", "BLUE"), seed=0, **raw)
    effective = dataclasses.asdict(config)
    effective.pop("seed")
    return effective


def _realized_search_identity(
    search_operator: Mapping[str, Any], *, c_scale: Any
) -> dict[str, Any]:
    """Bind the search identity actually realized by a category/job."""

    if isinstance(c_scale, bool) or not isinstance(c_scale, (int, float)):
        raise ContractError("realized search c_scale must be numeric")
    realized_c_scale = float(c_scale)
    if not math.isfinite(realized_c_scale) or realized_c_scale <= 0.0:
        raise ContractError("realized search c_scale must be finite and positive")
    realized_operator = dict(search_operator)
    realized_operator["c_scale"] = realized_c_scale
    realized_operator = _search_operator(realized_operator)
    effective = json.loads(json.dumps(_effective_search(realized_operator)))
    return {
        "search_operator": realized_operator,
        "search_operator_sha256": _digest_value(realized_operator),
        "effective_search_config": effective,
        "effective_search_config_sha256": _digest_value(effective),
    }


def _job_search_identity(
    lock: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    # This call is intentionally a validation boundary, not dead descriptive
    # metadata.  A post-promotion opponent job still records the promoted
    # producer's seat, and therefore must execute the c_scale that is part of
    # that producer's deployed agent identity.
    _promoted_producer_job_identity(lock, job)
    search = dict(lock["science"]["search_operator"])
    return _realized_search_identity(
        search, c_scale=job.get("c_scale", search["c_scale"])
    )


def _category_search_identities(
    search_operator: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for job in jobs:
        category = str(job["category"])
        identity = _realized_search_identity(
            search_operator,
            c_scale=job.get("c_scale", search_operator["c_scale"]),
        )
        previous = identities.setdefault(category, identity)
        if previous != identity:
            raise ContractError(
                f"category {category!r} has multiple realized search operators"
            )
    return {category: identities[category] for category in sorted(identities)}


def _search_operator(raw: dict[str, Any]) -> dict[str, Any]:
    """The explicit, adjudicated operator (separate from code-default fields)."""
    effective = _effective_search(raw)
    return {key: effective[key] for key in sorted(_SEARCH_INPUT_KEYS)}


def _effective_evaluator(raw: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(raw, _EVALUATOR_INPUT_KEYS, where="science.evaluator")
    for key in ("public_observation", "rust_featurize", "emit_uncertainty"):
        if type(raw[key]) is not bool:
            raise ContractError(f"science.evaluator.{key} must be a JSON boolean")
    if type(raw["cache_size"]) is not int:
        raise ContractError("science.evaluator.cache_size must be an integer")
    for key in ("value_scale", "prior_temperature", "context_fill"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
            raise ContractError(f"science.evaluator.{key} must be numeric")
    for key in ("value_squash", "value_readout"):
        if not isinstance(raw[key], str):
            raise ContractError(f"science.evaluator.{key} must be a string")
    config = EntityGraphRustEvaluatorConfig(**raw)
    return dataclasses.asdict(config)


def _guard_cli_flag_lint(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    matches = [
        spec
        for spec in payload.get("guards", [])
        if isinstance(spec, dict) and spec.get("name") == "cli_flag_lint"
    ]
    if len(matches) != 1:
        raise ContractError(
            f"{path} must have exactly one cli_flag_lint guard, found {len(matches)}"
        )
    args = matches[0].get("args")
    if not isinstance(args, dict):
        raise ContractError(f"{path} cli_flag_lint args must be an object")
    return args


def _guard_expected_values(path: Path) -> tuple[dict[str, Any], set[str]]:
    args = _guard_cli_flag_lint(_load_json(path), path=path)
    return dict(args.get("expected_values", {})), set(args.get("critical_flags", []))


def _validate_guard_sync_provenance(
    payload: dict[str, Any],
    *,
    path: Path,
    selected_c_scale: float,
    s1_evidence: dict[str, Any] | None,
    allow_stale_synchronizer: bool = False,
) -> bool:
    """Require a self-contained receipt when S1 moves the static guard.

    The default ``.03`` selection intentionally leaves the checked-in guard
    byte-for-byte untouched.  A non-default S1 selection is different: the
    guard itself must carry the exact S1 artifact and synchronizer identities,
    so the later guard/runtime hashes preserve why that mutable config changed.
    """

    receipt = payload.get(GUARD_SYNC_KEY)
    if selected_c_scale == DEFAULT_GENERATION_C_SCALE:
        if receipt is not None:
            raise ContractError(
                f"guard {path} carries stale {GUARD_SYNC_KEY} metadata for the "
                "default c_scale=0.03 selection"
            )
        return False
    if not isinstance(receipt, dict):
        raise ContractError(
            f"guard {path} selects non-default c_scale={selected_c_scale!r} without "
            f"a {GUARD_SYNC_SCHEMA} receipt; run sync-generation-guard before seal"
        )
    required = {
        "schema_version",
        "selected_c_scale",
        "source_s1_evidence",
        "previous_guard_sha256",
        "synchronizer",
    }
    _require_exact_keys(receipt, required, where=f"guard {GUARD_SYNC_KEY}")
    if receipt["schema_version"] != GUARD_SYNC_SCHEMA:
        raise ContractError(f"guard {path} has an unsupported guard-sync receipt")
    if receipt["selected_c_scale"] != selected_c_scale:
        raise ContractError(f"guard {path} guard-sync selected_c_scale drift")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt["previous_guard_sha256"])):
        raise ContractError(f"guard {path} guard-sync previous SHA-256 is malformed")
    if not isinstance(s1_evidence, dict):
        raise ContractError(
            f"guard {path} cannot validate non-default c_scale without typed S1 evidence"
        )
    expected_s1 = {
        "path": str(Path(str(s1_evidence["path"])).resolve(strict=True)),
        "sha256": str(s1_evidence["sha256"]),
    }
    if receipt["source_s1_evidence"] != expected_s1:
        raise ContractError(f"guard {path} guard-sync S1 provenance drift")
    synchronizer = receipt["synchronizer"]
    if not isinstance(synchronizer, dict) or set(synchronizer) != {"path", "sha256"}:
        raise ContractError(f"guard {path} has malformed synchronizer provenance")
    if synchronizer["path"] != GUARD_SYNC_TOOL:
        raise ContractError(f"guard {path} was synchronized by an unexpected tool")
    synchronizer_path = (REPO_ROOT / GUARD_SYNC_TOOL).resolve(strict=True)
    synchronizer_stale = synchronizer["sha256"] != _sha256(synchronizer_path)
    if synchronizer_stale and not allow_stale_synchronizer:
        raise ContractError(f"guard {path} synchronizer implementation drift")
    return synchronizer_stale


def _validate_guard(
    path: Path,
    *,
    search: dict[str, Any],
    evaluator: dict[str, Any],
    generation: dict[str, Any],
    s1_evidence: dict[str, Any] | None = None,
    archived_markerless: bool = False,
) -> None:
    _validate_guard_payload(
        _load_json(path),
        path=path,
        search=search,
        evaluator=evaluator,
        generation=generation,
        s1_evidence=s1_evidence,
        archived_markerless=archived_markerless,
    )


def _validate_guard_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    search: dict[str, Any],
    evaluator: dict[str, Any],
    generation: dict[str, Any],
    s1_evidence: dict[str, Any] | None = None,
    archived_markerless: bool = False,
) -> None:
    """Validate prospective or on-disk guard bytes with identical semantics."""

    args = _guard_cli_flag_lint(payload, path=path)
    expected = dict(args.get("expected_values", {}))
    critical = set(args.get("critical_flags", []))
    native_flag = "--native-mcts-hot-loop"
    rust_featurize_flag = "--rust-featurize"
    required_critical = {
        "--c-scale",
        "--c-visit",
        "--n-full",
        "--n-fast",
        "--p-full",
        "--base-seed",
        "--games",
        "--max-depth",
        "--symmetry-averaged-eval",
        "--symmetry-averaged-eval-threshold",
        "--belief-chance-spectra",
        "--information-set-search",
        "--determinization-particles",
        "--determinization-min-simulations",
    }
    if archived_markerless:
        if (
            "native_mcts_hot_loop" in generation
            or native_flag in critical
            or native_flag in expected
            or rust_featurize_flag in critical
            or rust_featurize_flag in expected
        ):
            raise ContractError(
                "archived markerless guard must preserve the omitted native-MCTS "
                "and Rust-featurizer flags and their legacy false runtime defaults"
            )
    else:
        required_critical.add(native_flag)
        if evaluator["rust_featurize"] is True:
            required_critical.add(rust_featurize_flag)
        elif rust_featurize_flag in critical or rust_featurize_flag in expected:
            raise ContractError(
                "guard advertises the Rust featurizer for a Python-feature contract"
            )
    if not required_critical.issubset(critical):
        raise ContractError(
            f"guard {path} is missing critical flags {sorted(required_critical - critical)}"
        )
    comparisons = {
        "--c-scale": search["c_scale"],
        "--c-visit": search["c_visit"],
        "--n-full": search["n_full"],
        "--n-fast": search["n_fast"],
        "--p-full": search["p_full"],
        "--max-depth": search["max_depth"],
        "--temperature-decisions": generation["temperature_decisions"],
        "--public-observation": evaluator["public_observation"],
        "--lazy-interior-chance": search["lazy_interior_chance"],
        "--symmetry-averaged-eval": search["symmetry_averaged_eval"],
        "--symmetry-averaged-eval-threshold": search[
            "symmetry_averaged_eval_threshold"
        ],
        "--belief-chance-spectra": search["belief_chance_spectra"],
        "--information-set-search": search["information_set_search"],
        "--determinization-particles": search["determinization_particles"],
        "--determinization-min-simulations": search[
            "determinization_min_simulations"
        ],
    }
    if not archived_markerless:
        comparisons[native_flag] = generation["native_mcts_hot_loop"]
        if evaluator["rust_featurize"] is True:
            comparisons[rust_featurize_flag] = True
    for flag, contract_value in comparisons.items():
        if flag not in expected or expected[flag] != contract_value:
            raise ContractError(
                f"guard drift: {path} expected_values[{flag!r}]={expected.get(flag)!r}, "
                f"winning contract requires {contract_value!r}"
            )
    _validate_guard_sync_provenance(
        payload,
        path=path,
        selected_c_scale=float(search["c_scale"]),
        s1_evidence=s1_evidence,
    )


def _atomic_replace_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a mutable config without exposing a partial JSON file."""

    path = path.resolve(strict=True)
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.sync-{os.getpid()}.tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sync_generation_guard(draft_path: Path) -> dict[str, Any]:
    """Synchronize the static generation guard from replayable typed S1.

    This is intentionally a narrow pre-seal mutation.  It cannot infer a
    choice from defaults or informal output, and it never edits the draft or
    launches work.  The common c_scale=.03 decision is a byte-for-byte no-op.
    Non-default decisions embed their S1/tool receipt inside the guard itself,
    which the subsequent guard record and runtime-tree hashes then seal.
    """

    draft_path = draft_path.resolve(strict=True)
    draft = _load_json(draft_path)
    if draft.get("schema_version") not in {DRAFT_SCHEMA, LEGACY_DRAFT_SCHEMA}:
        raise ContractError(
            f"draft schema must be {DRAFT_SCHEMA!r} or historical {LEGACY_DRAFT_SCHEMA!r}"
        )
    science = draft.get("science")
    if not isinstance(science, dict) or not isinstance(science.get("search"), dict):
        raise ContractError("draft has no science.search object")
    raw_search = dict(science["search"])
    s1_keys = {
        "c_scale",
        "symmetry_averaged_eval",
        "symmetry_averaged_eval_threshold",
        "rescale_noise_floor_c",
        "sigma_eval",
    }
    unresolved_s1 = sorted(
        key for key in s1_keys if raw_search.get(key, UNRESOLVED) == UNRESOLVED
    )
    if unresolved_s1:
        raise ContractError(
            f"cannot synchronize generation guard before typed S1 resolves {unresolved_s1}"
        )
    final_s1 = {key: raw_search[key] for key in s1_keys}
    selected_c_scale = final_s1["c_scale"]
    if isinstance(selected_c_scale, bool) or not isinstance(
        selected_c_scale, (int, float)
    ):
        raise ContractError("science.search.c_scale must be numeric")
    selected_c_scale = float(selected_c_scale)
    if selected_c_scale not in {0.03, 0.1, 0.3}:
        raise ContractError(
            "typed S1 c_scale must be one of the predeclared {.03,.1,.3} arms"
        )

    evidence_items = science.get("evidence")
    if not isinstance(evidence_items, list):
        raise ContractError("science.evidence must be a list")
    s1_items = [
        item
        for item in evidence_items
        if isinstance(item, dict) and item.get("kind") == "s1"
    ]
    if len(s1_items) != 1 or set(s1_items[0]) != {"kind", "path"}:
        raise ContractError("science.evidence must contain exactly one typed S1 path")
    if s1_items[0]["path"] == UNRESOLVED:
        raise ContractError("cannot synchronize generation guard before typed S1 exists")
    s1_path = _absolute_ref(str(s1_items[0]["path"]), base=draft_path.parent)
    s1_payload = _load_json(s1_path)
    post_promotion_s1_path: Path | None = None
    allow_post_promotion_s1 = False
    recovery_s1_path: Path | None = None
    allow_recovery_s1 = False
    if (
        draft.get("schema_version") == DRAFT_SCHEMA
        and isinstance(draft.get("promotion_handoff"), dict)
        and draft["promotion_handoff"].get("mode") == POST_PROMOTION_HANDOFF_MODE
    ):
        post_promotion_s1_path = _absolute_ref(
            str(draft["promotion_handoff"].get("path", "")),
            base=draft_path.parent,
        )
        allow_post_promotion_s1 = True
    elif (
        draft.get("schema_version") == DRAFT_SCHEMA
        and isinstance(draft.get("promotion_handoff"), dict)
        and draft["promotion_handoff"].get("mode")
        == DISASTER_RECOVERY_HANDOFF_MODE
    ):
        recovery_s1_path = _absolute_ref(
            str(draft["promotion_handoff"].get("path", "")),
            base=draft_path.parent,
        )
        allow_recovery_s1 = True
    _validate_search_stage_evidence(
        s1_payload,
        path=s1_path,
        expected_stage="s1",
        final_search=final_s1,
        final_evaluator={},
        post_promotion_handoff_path=post_promotion_s1_path,
        allow_post_promotion_s1=allow_post_promotion_s1,
        recovery_receipt_path=recovery_s1_path,
        allow_recovery_s1=allow_recovery_s1,
    )
    s1_record = _file_record(s1_path, kind="s1")

    provenance = draft.get("provenance")
    if not isinstance(provenance, dict) or "guard_config" not in provenance:
        raise ContractError("draft provenance has no guard_config")
    guard_path = _absolute_ref(
        str(provenance["guard_config"]), base=draft_path.parent
    ).resolve(strict=True)
    guard_payload = _load_json(guard_path)
    args = _guard_cli_flag_lint(guard_payload, path=guard_path)
    expected = args.get("expected_values")
    if not isinstance(expected, dict) or "--c-scale" not in expected:
        raise ContractError(f"{guard_path} has no expected --c-scale value")
    current_c_scale = expected["--c-scale"]

    # The generation guard binds the complete winning search recipe, not only
    # the S1-selected c_scale.  Preserve every resolved search field from the
    # draft while substituting the typed S1 winner.  Passing the former
    # two-field projection made exact guard validation crash as soon as the
    # guard began binding n_full, information-set search, and the remaining
    # production search parameters.
    guard_search = dict(raw_search)
    guard_search["c_scale"] = selected_c_scale
    evaluator = science.get("evaluator")
    generation = draft.get("generation")
    if not isinstance(evaluator, dict) or not isinstance(generation, dict):
        raise ContractError("draft evaluator/generation objects are missing")
    guard_evaluator = {
        "public_observation": evaluator.get("public_observation"),
        "rust_featurize": evaluator.get("rust_featurize"),
    }
    guard_generation = {
        "temperature_decisions": generation.get("temperature_decisions"),
        "native_mcts_hot_loop": generation.get("native_mcts_hot_loop"),
    }

    before_sha256 = _sha256(guard_path)
    if selected_c_scale == DEFAULT_GENERATION_C_SCALE:
        if current_c_scale != DEFAULT_GENERATION_C_SCALE:
            raise ContractError(
                "typed S1 retained c_scale=0.03 but the guard is not pristine; "
                "refusing to hide manual guard drift"
            )
        _validate_guard(
            guard_path,
            search=guard_search,
            evaluator=guard_evaluator,
            generation=guard_generation,
            s1_evidence=s1_record,
        )
        return {
            "status": "already_synchronized",
            "changed": False,
            "selected_c_scale": selected_c_scale,
            "guard": str(guard_path),
            "before_sha256": before_sha256,
            "after_sha256": before_sha256,
            "s1_evidence": {"path": str(s1_path), "sha256": s1_record["sha256"]},
        }

    if current_c_scale == selected_c_scale:
        expected_s1_reference = {
            "path": str(s1_path.resolve(strict=True)),
            "sha256": s1_record["sha256"],
        }
        existing_receipt = guard_payload.get(GUARD_SYNC_KEY)
        # Disaster recovery replaces the lost post-promotion authority with a
        # new, replayable S1 binding while deliberately retaining the deployed
        # c_scale=.10 operator.  The old synchronizer treated that legitimate
        # authority change as manual drift and made recovery impossible.  Only
        # this typed recovery mode may rebind provenance in place; the complete
        # guard recipe is validated below before the atomic replacement.
        if (
            allow_recovery_s1
            and isinstance(existing_receipt, dict)
            and existing_receipt.get("source_s1_evidence")
            != expected_s1_reference
        ):
            rebound_payload = json.loads(json.dumps(guard_payload))
            rebound_payload[GUARD_SYNC_KEY] = {
                "schema_version": GUARD_SYNC_SCHEMA,
                "selected_c_scale": selected_c_scale,
                "source_s1_evidence": expected_s1_reference,
                "previous_guard_sha256": before_sha256,
                "synchronizer": {
                    "path": GUARD_SYNC_TOOL,
                    "sha256": _sha256(
                        (REPO_ROOT / GUARD_SYNC_TOOL).resolve(strict=True)
                    ),
                },
            }
            _validate_guard_payload(
                rebound_payload,
                path=guard_path,
                search=guard_search,
                evaluator=guard_evaluator,
                generation=guard_generation,
                s1_evidence=s1_record,
            )
            if _sha256(guard_path) != before_sha256:
                raise ContractError(
                    f"guard {guard_path} changed concurrently during recovery rebind"
                )
            _atomic_replace_json(guard_path, rebound_payload)
            after_sha256 = _sha256(guard_path)
            return {
                "status": "recovery_provenance_rebound",
                "changed": True,
                "selected_c_scale": selected_c_scale,
                "guard": str(guard_path),
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "s1_evidence": expected_s1_reference,
            }
        synchronizer_stale = _validate_guard_sync_provenance(
            guard_payload,
            path=guard_path,
            selected_c_scale=selected_c_scale,
            s1_evidence=s1_record,
            allow_stale_synchronizer=True,
        )
        if synchronizer_stale:
            refreshed_payload = json.loads(json.dumps(guard_payload))
            refreshed_payload[GUARD_SYNC_KEY]["synchronizer"]["sha256"] = _sha256(
                (REPO_ROOT / GUARD_SYNC_TOOL).resolve(strict=True)
            )
            _validate_guard_payload(
                refreshed_payload,
                path=guard_path,
                search=guard_search,
                evaluator=guard_evaluator,
                generation=guard_generation,
                s1_evidence=s1_record,
            )
            if _sha256(guard_path) != before_sha256:
                raise ContractError(
                    f"guard {guard_path} changed concurrently during synchronization"
                )
            _atomic_replace_json(guard_path, refreshed_payload)
            after_sha256 = _sha256(guard_path)
            return {
                "status": "provenance_refreshed",
                "changed": True,
                "selected_c_scale": selected_c_scale,
                "guard": str(guard_path),
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "s1_evidence": {
                    "path": str(s1_path),
                    "sha256": s1_record["sha256"],
                },
            }
        _validate_guard(
            guard_path,
            search=guard_search,
            evaluator=guard_evaluator,
            generation=guard_generation,
            s1_evidence=s1_record,
        )
        return {
            "status": "already_synchronized",
            "changed": False,
            "selected_c_scale": selected_c_scale,
            "guard": str(guard_path),
            "before_sha256": before_sha256,
            "after_sha256": before_sha256,
            "s1_evidence": {"path": str(s1_path), "sha256": s1_record["sha256"]},
        }
    if current_c_scale != DEFAULT_GENERATION_C_SCALE or GUARD_SYNC_KEY in guard_payload:
        raise ContractError(
            f"refusing to overwrite unexplained guard c_scale={current_c_scale!r}; "
            "expected the pristine c_scale=0.03 guard"
        )

    expected["--c-scale"] = selected_c_scale
    guard_payload[GUARD_SYNC_KEY] = {
        "schema_version": GUARD_SYNC_SCHEMA,
        "selected_c_scale": selected_c_scale,
        "source_s1_evidence": {
            "path": str(s1_path.resolve(strict=True)),
            "sha256": s1_record["sha256"],
        },
        "previous_guard_sha256": before_sha256,
        "synchronizer": {
            "path": GUARD_SYNC_TOOL,
            "sha256": _sha256((REPO_ROOT / GUARD_SYNC_TOOL).resolve(strict=True)),
        },
    }
    # Validate the exact prospective payload before the atomic replacement.
    # A semantic error therefore leaves the original guard bytes untouched.
    _validate_guard_payload(
        guard_payload,
        path=guard_path,
        search=guard_search,
        evaluator=guard_evaluator,
        generation=guard_generation,
        s1_evidence=s1_record,
    )
    if _sha256(guard_path) != before_sha256:
        raise ContractError(
            f"guard {guard_path} changed concurrently during synchronization"
        )
    _atomic_replace_json(guard_path, guard_payload)
    _validate_guard(
        guard_path,
        search=guard_search,
        evaluator=guard_evaluator,
        generation=guard_generation,
        s1_evidence=s1_record,
    )
    return {
        "status": "synchronized",
        "changed": True,
        "selected_c_scale": selected_c_scale,
        "guard": str(guard_path),
        "before_sha256": before_sha256,
        "after_sha256": _sha256(guard_path),
        "s1_evidence": {"path": str(s1_path), "sha256": s1_record["sha256"]},
    }


def _validate_generation(generation: dict[str, Any]) -> None:
    _require_exact_keys(generation, _GENERATION_KEYS, where="generation")
    for key in ("eval_server", "native_mcts_hot_loop"):
        if type(generation[key]) is not bool:
            raise ContractError(f"generation.{key} must be a JSON boolean")
    for key in (
        "vps_to_win",
        "obs_width",
        "max_decisions",
        "temperature_decisions",
        "workers_per_gpu",
        "shard_size",
    ):
        if type(generation[key]) is not int:
            raise ContractError(f"generation.{key} must be an integer")
    if (
        generation["late_temperature_decisions"] is not None
        and type(generation["late_temperature_decisions"]) is not int
    ):
        raise ContractError(
            "generation.late_temperature_decisions must be an integer or null"
        )
    for key in ("temperature_high", "temperature_low", "late_temperature"):
        if isinstance(generation[key], bool) or not isinstance(
            generation[key], (int, float)
        ):
            raise ContractError(f"generation.{key} must be numeric")
    if generation["track"] != "2p_no_trade" or int(generation["vps_to_win"]) != 10:
        raise ContractError("A1 supports only the locked 2p_no_trade, 10-VP regime")
    if int(generation["max_decisions"]) <= 0:
        raise ContractError("max_decisions must be positive")
    opening = int(generation["temperature_decisions"])
    late = generation["late_temperature_decisions"]
    if opening < 0 or opening > int(generation["max_decisions"]):
        raise ContractError("temperature_decisions is outside the game cap")
    if late is not None and not (
        opening <= int(late) <= int(generation["max_decisions"])
    ):
        raise ContractError(
            "late_temperature_decisions must be between opening and max_decisions"
        )
    if late is None and float(generation["late_temperature"]) != 0.0:
        raise ContractError(
            "disabled late-temperature window must use late_temperature=0"
        )
    if late is not None and float(generation["late_temperature"]) <= 0.0:
        raise ContractError(
            "enabled late-temperature window requires positive late_temperature"
        )
    if generation["format"] != "npz":
        raise ContractError("the A1 postflight scanner currently requires format=npz")
    if generation["device"] != "cuda" or bool(generation["eval_server"]):
        raise ContractError(
            "A1 renders per-GPU CUDA jobs with eval_server=false (MPS is host-managed)"
        )
    if int(generation["workers_per_gpu"]) <= 0 or int(generation["shard_size"]) <= 0:
        raise ContractError("workers_per_gpu and shard_size must be positive")


def _validate_current_runtime_execution(
    draft_schema: str,
    *,
    evaluator: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> None:
    """Keep historical locks replayable while new waves use the fast sealed path."""

    if draft_schema != DRAFT_SCHEMA:
        return
    if evaluator.get("rust_featurize") is not True:
        raise ContractError(
            "current v3 waves require the parity-certified Rust featurizer"
        )
    if generation.get("native_mcts_hot_loop") is not True:
        raise ContractError(
            "current v3 waves require the capability-sealed native MCTS hot loop"
        )


def _validate_post_wave(value: dict[str, Any]) -> None:
    required = {
        "require_complete_games",
        "selected_truncations_max",
        "invalid_teacher_actions_max",
        "require_public_observation",
        "require_unique_game_seeds",
        "require_no_val_only_overlap",
        "selection_before_row_expansion",
        "required_reports",
        "require_shard_sha256",
        "require_contract_attestation",
        "require_target_information_regime",
        "validation_holdout",
    }
    _require_exact_keys(value, required, where="post_wave_acceptance")
    if not all(
        bool(value[key])
        for key in (
            "require_complete_games",
            "require_public_observation",
            "require_unique_game_seeds",
            "require_no_val_only_overlap",
            "selection_before_row_expansion",
            "require_shard_sha256",
            "require_contract_attestation",
        )
    ):
        raise ContractError("all fail-closed A1 post-wave requirements must be true")
    if value["require_target_information_regime"] != TARGET_INFORMATION_REGIME_PUBLIC:
        raise ContractError(
            "require_target_information_regime must be exactly "
            f"{TARGET_INFORMATION_REGIME_PUBLIC!r}"
        )
    if (
        int(value["selected_truncations_max"]) != 0
        or int(value["invalid_teacher_actions_max"]) != 0
    ):
        raise ContractError(
            "selected truncations and invalid teacher actions must both be zero"
        )
    reports = set(map(str, value["required_reports"]))
    if reports != REQUIRED_REPORTS:
        raise ContractError(
            f"required_reports must be exactly {sorted(REQUIRED_REPORTS)}, got {sorted(reports)}"
        )
    validation = dict(value["validation_holdout"])
    if validation != {
        "split_unit": "game_seed",
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
    }:
        raise ContractError(
            "validation_holdout must be the shared game-level 5%, seed=17, max_samples=0 contract"
        )


def _canonical_workers_from_fleet_manifest(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the exact current topology and derive its ordered GPU lanes.

    The worker id is a pure projection of ``alias`` and the physical GPU index.
    A draft may echo this list for operator readability, but it cannot select a
    different order, omit a GPU, or invent a placement.
    """

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    payload = _load_json(manifest_path)
    if payload.get("schema_version") != CURRENT_FLEET_SCHEMA:
        raise ContractError(
            f"fleet manifest schema must be {CURRENT_FLEET_SCHEMA!r}"
        )
    if payload.get("fleet_authority") != CURRENT_FLEET_AUTHORITY:
        raise ContractError(
            f"fleet manifest authority must be {CURRENT_FLEET_AUTHORITY!r}"
        )
    hosts = payload.get("hosts")
    if not isinstance(hosts, list) or not hosts:
        raise ContractError("fleet manifest must contain an ordered non-empty hosts list")
    workers: list[dict[str, Any]] = []
    aliases: set[str] = set()
    addresses: set[str] = set()
    for index, host in enumerate(hosts):
        if not isinstance(host, dict):
            raise ContractError(f"fleet manifest host {index} is not an object")
        alias = host.get("alias")
        address = host.get("address")
        gpu_count = host.get("gpu_count")
        if (
            not isinstance(alias, str)
            or not alias
            or not isinstance(address, str)
            or not address
            or isinstance(gpu_count, bool)
            or not isinstance(gpu_count, int)
            or gpu_count < 1
        ):
            raise ContractError(f"fleet manifest host {index} identity is invalid")
        if alias in aliases or address in addresses:
            raise ContractError("fleet manifest contains duplicate host alias/address")
        aliases.add(alias)
        addresses.add(address)
        workers.extend(
            {
                "id": f"{alias}_gpu{gpu}",
                "host_alias": alias,
                "gpu": gpu,
            }
            for gpu in range(gpu_count)
        )
    if len(workers) != CURRENT_WORKER_COUNT:
        raise ContractError(
            "current fleet manifest must project exactly "
            f"{CURRENT_WORKER_COUNT} GPU workers, got {len(workers)}"
        )
    return workers, _file_record(manifest_path, kind="fleet_manifest")


def _balanced_worker_quotas(
    workers: Sequence[Mapping[str, Any]],
    *,
    quota_policy: str = BALANCED_PREFIX_QUOTA_POLICY,
) -> dict[str, dict[str, int]]:
    """Return exact category quotas in canonical balanced-prefix order."""

    if len(workers) != CURRENT_WORKER_COUNT:
        raise ContractError(
            f"balanced v3 quota policy requires {CURRENT_WORKER_COUNT} workers"
        )
    worker_ids = [str(worker["id"]) for worker in workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise ContractError("balanced v3 quota policy received duplicate worker ids")
    expected_games = _games_for_quota_policy(quota_policy)
    quotas = {worker_id: {} for worker_id in worker_ids}
    for category, total in expected_games.items():
        base, extra = divmod(int(total), len(worker_ids))
        for index, worker_id in enumerate(worker_ids):
            quotas[worker_id][category] = base + int(index < extra)
    if {
        category: sum(worker[category] for worker in quotas.values())
        for category in expected_games
    } != expected_games:
        raise AssertionError("balanced quota construction lost exact science totals")
    return quotas


def _games_for_quota_policy(quota_policy: str) -> dict[str, int]:
    """Return one immutable selected-game profile for a current 64-GPU wave."""

    if quota_policy == BALANCED_PREFIX_QUOTA_POLICY:
        return dict(EXPECTED_GAMES)
    if quota_policy == BALANCED_PER_LANE_64K_QUOTA_POLICY:
        return dict(SCALE_64K_GAMES)
    raise ContractError(
        "current fleet quota_policy must be one of "
        f"{sorted(AUTHORIZED_CURRENT_QUOTA_POLICIES)}, got {quota_policy!r}"
    )


def _profile_for_quota_policy(quota_policy: str) -> str:
    _games_for_quota_policy(quota_policy)
    return (
        SCALE_64K_GAME_CONTRACT_PROFILE
        if quota_policy == BALANCED_PER_LANE_64K_QUOTA_POLICY
        else CURRENT_GAME_CONTRACT_PROFILE
    )


def _build_balanced_jobs(
    workers: list[dict[str, Any]],
    *,
    seed_base: int,
    block_size: int,
    output_root: str,
    contract_id: str,
    quota_policy: str = BALANCED_PREFIX_QUOTA_POLICY,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    expected_games = _games_for_quota_policy(quota_policy)
    quotas = _balanced_worker_quotas(workers, quota_policy=quota_policy)
    max_attempts_per_worker = max(
        sum(
            int(quotas[worker_id][category])
            + int(ATTEMPT_RESERVE_PER_JOB[category])
            for category in expected_games
        )
        for worker_id in quotas
    )
    if block_size < max_attempts_per_worker:
        raise ContractError(
            f"seed block_size={block_size} is smaller than "
            f"the v3 maximum {max_attempts_per_worker} attempts/worker"
        )
    jobs: list[dict[str, Any]] = []
    for worker_index, worker in enumerate(workers):
        cursor = seed_base + worker_index * block_size
        worker_id = str(worker["id"])
        for category in expected_games:
            games = int(quotas[worker_id][category])
            attempts = games + int(ATTEMPT_RESERVE_PER_JOB[category])
            job_id = f"{worker_id}__{category}"
            jobs.append(
                {
                    "job_id": job_id,
                    "worker_id": worker_id,
                    "host_alias": str(worker["host_alias"]),
                    "gpu": int(worker["gpu"]),
                    "category": category,
                    "base_seed": cursor,
                    "games": games,
                    "attempts": attempts,
                    "seed_end": cursor + attempts,
                    "output_dir": str(Path(output_root) / contract_id / job_id),
                    "claim_label": f"{contract_id}:{job_id}",
                }
            )
            cursor += attempts
    assert_disjoint_seed_blocks(
        [
            (job["job_id"], int(job["base_seed"]), int(job["attempts"]))
            for job in jobs
        ]
    )
    for job in jobs:
        interval = (int(job["base_seed"]), int(job["seed_end"]))
        if _ranges_overlap(interval, VAL_ONLY_SEED_RANGE):
            raise ContractError(
                f"job {job['job_id']} seed range {interval} overlaps "
                f"VAL-ONLY {VAL_ONLY_SEED_RANGE}"
            )
    return jobs, quotas


def _build_jobs(
    workers: list[dict[str, Any]],
    *,
    seed_base: int,
    block_size: int,
    per_worker: dict[str, int],
    output_root: str,
    contract_id: str,
) -> list[dict[str, Any]]:
    if len(workers) != EXPECTED_WORKER_COUNT:
        raise ContractError(
            "the pre-wave handoff requires exactly "
            f"{EXPECTED_WORKER_COUNT} workers, got {len(workers)}"
        )
    if per_worker != EXPECTED_PER_WORKER:
        raise ContractError(
            f"per_worker_games must be exactly {EXPECTED_PER_WORKER}, got {per_worker}"
        )
    worker_ids = [str(worker["id"]) for worker in workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise ContractError("fleet.workers contains duplicate ids")
    for worker in workers:
        if set(worker) != {"id", "host_alias", "gpu"}:
            raise ContractError(
                "each fleet worker must have exactly id, host_alias, gpu"
            )
        if int(worker["gpu"]) < 0:
            raise ContractError(f"worker {worker['id']} has a negative GPU index")
    placements = [(str(worker["host_alias"]), int(worker["gpu"])) for worker in workers]
    if len(set(placements)) != len(placements):
        raise ContractError(
            "fleet.workers assigns the same host/GPU placement more than once"
        )
    attempts_per_worker = sum(EXPECTED_ATTEMPTS_PER_WORKER.values())
    if block_size < attempts_per_worker:
        raise ContractError(
            f"seed block_size={block_size} is smaller than "
            f"{attempts_per_worker} attempts/worker"
        )
    jobs: list[dict[str, Any]] = []
    category_order = tuple(EXPECTED_GAMES)
    for worker_index, worker in enumerate(workers):
        cursor = seed_base + worker_index * block_size
        for category in category_order:
            games = int(per_worker[category])
            attempts = int(EXPECTED_ATTEMPTS_PER_WORKER[category])
            job_id = f"{worker['id']}__{category}"
            jobs.append(
                {
                    "job_id": job_id,
                    "worker_id": str(worker["id"]),
                    "host_alias": str(worker["host_alias"]),
                    "gpu": int(worker["gpu"]),
                    "category": category,
                    "base_seed": cursor,
                    "games": games,
                    "attempts": attempts,
                    "seed_end": cursor + attempts,
                    "output_dir": str(Path(output_root) / contract_id / job_id),
                    "claim_label": f"{contract_id}:{job_id}",
                }
            )
            cursor += attempts
    assert_disjoint_seed_blocks(
        [
            (job["job_id"], int(job["base_seed"]), int(job["attempts"]))
            for job in jobs
        ]
    )
    totals = Counter()
    for job in jobs:
        totals[job["category"]] += int(job["games"])
        interval = (int(job["base_seed"]), int(job["seed_end"]))
        if _ranges_overlap(interval, VAL_ONLY_SEED_RANGE):
            raise ContractError(
                f"job {job['job_id']} seed range {interval} overlaps VAL-ONLY {VAL_ONLY_SEED_RANGE}"
            )
    if dict(totals) != EXPECTED_GAMES:
        raise ContractError(
            f"job category totals are {dict(totals)}, expected {EXPECTED_GAMES}"
        )
    return jobs


def _read_strict_ledger(ledger: Path) -> tuple[str, list[tuple[int, int, str]]]:
    try:
        ledger_text = ledger.read_text(encoding="utf-8")
        ledger_lines = ledger_text.splitlines()
        claims = parse_seed_ledger(ledger)
    except Exception as error:  # noqa: BLE001 - malformed/unreadable ledger blocks sealing.
        raise ContractError(f"cannot parse seed ledger {ledger}: {error}") from error
    # The shared parser intentionally skips malformed rows so routine generator
    # guards do not crash every launch.  A one-time immutable handoff must be
    # stricter: any line that looks like a range row but was not parsed could
    # conceal a claim, so refuse instead of treating it as free space.
    range_like = [
        line
        for line in ledger_lines
        if re.match(r"^\s*\|?\s*\[.*\)\s*\|", line) is not None
    ]
    if len(range_like) != len(claims):
        raise ContractError(
            f"seed ledger {ledger} has {len(range_like)} range-like row(s) but only "
            f"{len(claims)} parsed claim(s); repair malformed ledger rows before sealing"
        )
    return ledger_text, claims


def _seed_ledger_snapshot(ledger: Path) -> dict[str, Any]:
    ledger_text, claims = _read_strict_ledger(ledger)
    if ledger_text and not ledger_text.endswith("\n"):
        raise ContractError(
            f"seed ledger {ledger} must end with a newline so later claims append safely"
        )
    record = _file_record(ledger, kind="seed_ledger_snapshot")
    record.update(
        {
            "snapshot_text": ledger_text,
            "snapshot_size_bytes": len(ledger_text.encode("utf-8")),
            "claims": [
                {"start": int(start), "end": int(end), "label": str(label)}
                for start, end, label in claims
            ],
            "claims_sha256": _digest_value(
                [
                    {"start": int(start), "end": int(end), "label": str(label)}
                    for start, end, label in claims
                ]
            ),
        }
    )
    return record


def _validate_against_ledger(jobs: list[dict[str, Any]], ledger: Path) -> None:
    _ledger_text, claims = _read_strict_ledger(ledger)
    for job in jobs:
        requested = (int(job["base_seed"]), int(job["seed_end"]))
        for start, end, label in claims:
            if _ranges_overlap(requested, (int(start), int(end))):
                raise ContractError(
                    f"job {job['job_id']} range {requested} overlaps ledger claim "
                    f"[{start}, {end}) {label!r}"
                )


def _ledger_claim_label(contract_sha256: str, job: dict[str, Any]) -> str:
    return (
        f"claim={job['claim_label']} contract={contract_sha256} "
        f"job={job['job_id']} |"
    )


def _verify_live_seed_ledger(
    snapshot: dict[str, Any],
    jobs: list[dict[str, Any]],
    *,
    contract_sha256: str,
    require_all_job_claims: bool,
) -> None:
    """Verify an immutable pre-claim prefix plus append-only live claims.

    The shared seed ledger is intentionally mutable after sealing.  Its sealed
    bytes must remain an exact prefix, while appended rows may contain unrelated
    disjoint work and this contract's own exact job claims.  Any overlapping peer
    claim, mislabeled range, or missing required post-wave claim fails closed.
    """

    expected_fields = {
        "kind",
        "path",
        "sha256",
        "snapshot_text",
        "snapshot_size_bytes",
        "claims",
        "claims_sha256",
    }
    if set(snapshot) != expected_fields:
        raise ContractError(
            "seed-ledger snapshot fields mismatch; "
            f"missing={sorted(expected_fields - set(snapshot))} "
            f"extra={sorted(set(snapshot) - expected_fields)}"
        )
    if snapshot.get("kind") != "seed_ledger_snapshot":
        raise ContractError("seed-ledger record is not an immutable snapshot")
    snapshot_text = snapshot.get("snapshot_text")
    if not isinstance(snapshot_text, str):
        raise ContractError("seed-ledger snapshot_text is not a string")
    snapshot_bytes = snapshot_text.encode("utf-8")
    if int(snapshot.get("snapshot_size_bytes", -1)) != len(snapshot_bytes):
        raise ContractError("seed-ledger snapshot byte count drift")
    if "sha256:" + hashlib.sha256(snapshot_bytes).hexdigest() != snapshot.get(
        "sha256"
    ):
        raise ContractError("seed-ledger snapshot hash drift inside contract")
    locked_claims = snapshot.get("claims")
    if not isinstance(locked_claims, list) or snapshot.get(
        "claims_sha256"
    ) != _digest_value(locked_claims):
        raise ContractError("seed-ledger snapshot claim digest drift")

    ledger = Path(str(snapshot["path"]))
    try:
        live_bytes = ledger.read_bytes()
    except OSError as error:
        raise ContractError(f"cannot read live seed ledger {ledger}: {error}") from error
    if not live_bytes.startswith(snapshot_bytes):
        raise ContractError(
            f"live seed ledger {ledger} is not an append-only extension of the sealed snapshot"
        )
    try:
        live_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"live seed ledger {ledger} is not UTF-8: {error}") from error
    _live_text, live_claims = _read_strict_ledger(ledger)
    normalized_locked = [
        (int(item["start"]), int(item["end"]), str(item["label"]))
        for item in locked_claims
        if isinstance(item, dict)
        and set(item) == {"start", "end", "label"}
    ]
    if len(normalized_locked) != len(locked_claims) or live_claims[
        : len(normalized_locked)
    ] != normalized_locked:
        raise ContractError("seed-ledger sealed claim prefix drift")

    own_matches: dict[str, int] = {str(job["job_id"]): 0 for job in jobs}
    for start, end, label in live_claims:
        claim = (int(start), int(end))
        for job in jobs:
            requested = (int(job["base_seed"]), int(job["seed_end"]))
            claim_token = f"claim={job['claim_label']}"
            expected_label = _ledger_claim_label(contract_sha256, job)
            names_us = claim_token in str(label)
            exact_own_claim = claim == requested and str(label) == expected_label
            if names_us and not exact_own_claim:
                raise ContractError(
                    "ledger claim does not exactly match its rendered contract/job row: "
                    f"label={label!r} range={claim}, expected_label={expected_label!r} "
                    f"expected_range={requested}"
                )
            if not _ranges_overlap(requested, claim):
                continue
            if not exact_own_claim:
                raise ContractError(
                    f"job {job['job_id']} range {requested} overlaps live ledger claim "
                    f"[{start}, {end}) {label!r}"
                )
            own_matches[str(job["job_id"])] += 1
    duplicates = sorted(job_id for job_id, count in own_matches.items() if count > 1)
    if duplicates:
        raise ContractError(
            "live ledger repeats exact own claim(s); resume without appending a second row: "
            f"{duplicates[:8]}"
        )
    if require_all_job_claims:
        missing = sorted(job_id for job_id, count in own_matches.items() if count == 0)
        if missing:
            raise ContractError(
                "post-wave ledger is missing exact own claim(s) for "
                f"{len(missing)} job(s): {missing[:8]}"
            )


LEGACY_HARD_NEGATIVE_SELECTION_FIELDS = {
    "schema_version",
    "checkpoint",
    "selection_reason",
    "evaluation_evidence",
    "selection_sha256",
}
RICH_HARD_NEGATIVE_SELECTION_FIELDS = {
    "authoritative_incumbent",
    "candidate",
    "candidate_bundle",
    "evidence",
    "limitations",
    "opponent_pool_mutation_performed",
    "promotion_decision",
    "promotion_eligible",
    "registry_mutation_authorized",
    "schema_version",
    "selection_basis",
    "selection_role",
    "selection_sha256",
    "status",
}


def _verified_sized_artifact_ref(
    raw: Any,
    *,
    where: str,
    kind: str,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    """Resolve one explicit path/hash/size binding without filesystem search."""

    if not isinstance(raw, dict):
        raise ContractError(f"{where} must be an object")
    _require_exact_keys(raw, {"path", "sha256", "bytes"}, where=where)
    raw_path = raw["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractError(f"{where}.path must be a non-empty string")
    unresolved = Path(raw_path).expanduser()
    if relative_to is None:
        if not unresolved.is_absolute():
            raise ContractError(f"{where}.path must be absolute")
        candidate = unresolved
    else:
        if unresolved.is_absolute() or ".." in PurePath(raw_path).parts:
            raise ContractError(
                f"{where}.path must be a bundle-relative path without traversal"
            )
        candidate = relative_to / unresolved
    try:
        path = candidate.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve {where} at {candidate}: {error}") from error
    if relative_to is not None:
        try:
            path.relative_to(relative_to.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise ContractError(f"{where}.path escapes its artifact bundle") from error
    if not path.is_file():
        raise ContractError(f"{where} is missing or not a file: {path}")
    declared_bytes = raw["bytes"]
    if (
        isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes < 0
    ):
        raise ContractError(f"{where}.bytes must be a non-negative integer")
    actual_bytes = path.stat().st_size
    if declared_bytes != actual_bytes:
        raise ContractError(
            f"{where} byte count drift: declared {declared_bytes}, actual {actual_bytes}"
        )
    actual_sha256 = _sha256(path)
    if raw["sha256"] != actual_sha256:
        raise ContractError(
            f"{where} hash drift: declared {raw['sha256']!r}, actual {actual_sha256}"
        )
    return {
        "kind": kind,
        "path": str(path),
        "sha256": actual_sha256,
        "bytes": actual_bytes,
    }


def _verified_artifact_ref(raw: Any, *, where: str, kind: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError(f"{where} must be an object")
    _require_exact_keys(raw, {"path", "sha256"}, where=where)
    raw_path = raw["path"]
    if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
        raise ContractError(f"{where}.path must be an absolute non-empty string")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve {where} at {raw_path}: {error}") from error
    if not path.is_file():
        raise ContractError(f"{where} is missing or not a file: {path}")
    actual_sha256 = _sha256(path)
    if raw["sha256"] != actual_sha256:
        raise ContractError(
            f"{where} hash drift: declared {raw['sha256']!r}, actual {actual_sha256}"
        )
    return {"kind": kind, "path": str(path), "sha256": actual_sha256}


def _artifact_identity(record: Mapping[str, Any]) -> dict[str, str]:
    return {"path": str(record["path"]), "sha256": str(record["sha256"])}


def _require_artifact_identity(
    raw: Any,
    expected: Mapping[str, Any],
    *,
    where: str,
    allow_extra: bool = False,
) -> None:
    if not isinstance(raw, dict):
        raise ContractError(f"{where} must be an object")
    if allow_extra:
        if not {"path", "sha256"}.issubset(raw):
            raise ContractError(f"{where} must bind path and sha256")
    else:
        _require_exact_keys(raw, {"path", "sha256"}, where=where)
    try:
        actual_path = Path(str(raw["path"])).expanduser().resolve(strict=True)
        expected_path = Path(str(expected["path"])).expanduser().resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve {where}: {error}") from error
    if actual_path != expected_path or raw["sha256"] != expected["sha256"]:
        raise ContractError(f"{where} binds different checkpoint bytes")


def _require_checkpoint_path(
    raw: Any, expected: Mapping[str, Any], *, where: str
) -> None:
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{where} must be a non-empty absolute path")
    try:
        actual = Path(raw).expanduser().resolve(strict=True)
        wanted = Path(str(expected["path"])).expanduser().resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve {where}: {error}") from error
    if actual != wanted:
        raise ContractError(f"{where} binds a different checkpoint")


def _legacy_hard_negative_selection_record(
    path: Path,
    payload: dict[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        LEGACY_HARD_NEGATIVE_SELECTION_FIELDS,
        where="hard-negative selection evidence",
    )
    if payload["schema_version"] != "a1-hard-negative-selection-v1":
        raise ContractError("hard-negative selection evidence schema is unsupported")
    unhashed = dict(payload)
    declared = unhashed.pop("selection_sha256")
    if declared != _digest_value(unhashed):
        raise ContractError("hard-negative selection evidence digest mismatch")
    if payload["checkpoint"] != {
        "path": str(checkpoint),
        "sha256": checkpoint_sha256,
    }:
        raise ContractError("hard-negative selection evidence binds different bytes")
    reason = payload["selection_reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ContractError("hard-negative selection reason must be non-empty")
    evidence = payload["evaluation_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
        raise ContractError("hard-negative evaluation evidence binding is malformed")
    evidence_path = Path(str(evidence["path"])).expanduser().resolve(strict=True)
    if _sha256(evidence_path) != evidence["sha256"]:
        raise ContractError("hard-negative evaluation evidence hash drift")
    record = _file_record(path, kind="hard_negative_selection")
    record.update(
        {
            "selection_sha256": declared,
            "evaluation_evidence": {
                "path": str(evidence_path),
                "sha256": evidence["sha256"],
            },
        }
    )
    return record


def _validate_hard_negative_bundle(
    bundle_record: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    bundle_path = Path(str(bundle_record["path"]))
    payload = _load_json(bundle_path)
    _require_exact_keys(
        payload,
        {
            "artifacts",
            "candidate_role",
            "code_commits",
            "evaluation",
            "files",
            "registry_mutation_authorized",
            "schema_version",
        },
        where="hard-negative candidate bundle",
    )
    if payload["schema_version"] != "aux-pointer-candidate-bundle-v1":
        raise ContractError("hard-negative candidate bundle schema is unsupported")
    if payload["candidate_role"] != "experimental_nonpromotable":
        raise ContractError("hard-negative candidate bundle role is promotion-capable")
    if payload["registry_mutation_authorized"] is not False:
        raise ContractError("hard-negative candidate bundle authorizes registry mutation")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict):
        raise ContractError("hard-negative candidate bundle artifacts must be an object")
    _require_exact_keys(
        artifacts,
        {"authoritative_v5", "candidate", "exact_parent"},
        where="hard-negative candidate bundle artifacts",
    )
    artifact_records = {
        name: _verified_sized_artifact_ref(
            artifacts[name],
            where=f"hard-negative candidate bundle artifacts.{name}",
            kind=f"hard_negative_bundle_{name}",
        )
        for name in sorted(artifacts)
    }
    if _artifact_identity(artifact_records["candidate"]) != _artifact_identity(
        candidate
    ):
        raise ContractError("hard-negative candidate bundle binds a different candidate")
    if _artifact_identity(
        artifact_records["authoritative_v5"]
    ) != _artifact_identity(incumbent):
        raise ContractError("hard-negative candidate bundle binds a different incumbent")
    raw_files = payload["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ContractError("hard-negative candidate bundle files must be non-empty")
    file_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(raw_files):
        record = _verified_sized_artifact_ref(
            raw,
            where=f"hard-negative candidate bundle files[{index}]",
            kind="hard_negative_bundle_file",
            relative_to=bundle_path.parent,
        )
        if record["path"] in seen_paths:
            raise ContractError("hard-negative candidate bundle repeats a file path")
        seen_paths.add(str(record["path"]))
        file_records.append(record)
    return artifact_records, file_records


def _validate_held_out_hard_negative_evidence(
    raw: Any,
    *,
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("held-out hard-negative evidence must be an object")
    required = {
        "artifact",
        "complete_candidate_win_rate",
        "complete_candidate_wins",
        "complete_incumbent_wins",
        "complete_pairs",
        "engine_identity",
        "evaluation_config",
        "incomplete_pairs",
        "pair_diagnostics",
        "pentanomial_sprt",
        "suite_manifest",
        "suite_pairs",
    }
    _require_exact_keys(raw, required, where="held-out hard-negative evidence")
    artifact = _verified_sized_artifact_ref(
        raw["artifact"],
        where="held-out hard-negative evidence artifact",
        kind="hard_negative_held_out_report",
    )
    suite_record = _verified_artifact_ref(
        raw["suite_manifest"],
        where="held-out hard-negative suite manifest",
        kind="hard_negative_held_out_suite",
    )
    report = _load_json(Path(str(artifact["path"])))
    if report.get("schema_version") != "a1-held-out-high-regret-report-v1":
        raise ContractError("held-out hard-negative report schema is unsupported")
    if report.get("held_out") is not True or report.get("suite") != "held_out_high_regret":
        raise ContractError("held-out hard-negative report is not an isolated suite")
    if report.get("errors") != []:
        raise ContractError("held-out hard-negative report contains evaluation errors")
    _require_artifact_identity(report.get("candidate"), candidate, where="held-out candidate")
    _require_artifact_identity(report.get("champion"), incumbent, where="held-out incumbent")
    _require_artifact_identity(
        report.get("suite_manifest"), suite_record, where="held-out suite manifest"
    )
    planned_engine = report.get("planned_engine_identity")
    if (
        report.get("engine_identity") != raw["engine_identity"]
        or not isinstance(planned_engine, dict)
        or any(raw["engine_identity"].get(key) != value for key, value in planned_engine.items())
    ):
        raise ContractError("held-out hard-negative engine identity drift")
    if report.get("evaluation_config") != raw["evaluation_config"]:
        raise ContractError("held-out hard-negative evaluation config drift")
    config = raw["evaluation_config"]
    if not isinstance(config, dict):
        raise ContractError("held-out hard-negative evaluation config is malformed")
    _require_checkpoint_path(
        config.get("candidate"), candidate, where="held-out config candidate"
    )
    _require_checkpoint_path(
        config.get("baseline"), incumbent, where="held-out config incumbent"
    )
    if report.get("pair_diagnostics") != raw["pair_diagnostics"]:
        raise ContractError("held-out hard-negative pair diagnostics drift")
    if report.get("pentanomial_sprt") != raw["pentanomial_sprt"]:
        raise ContractError("held-out hard-negative SPRT drift")
    if not isinstance(raw["pair_diagnostics"], dict) or not isinstance(
        raw["pentanomial_sprt"], dict
    ):
        raise ContractError("held-out hard-negative result summaries are malformed")
    complete_pairs = raw["complete_pairs"]
    incomplete_pairs = raw["incomplete_pairs"]
    suite_pairs = raw["suite_pairs"]
    declared_candidate_wins = raw["complete_candidate_wins"]
    declared_incumbent_wins = raw["complete_incumbent_wins"]
    if (
        isinstance(complete_pairs, bool)
        or not isinstance(complete_pairs, int)
        or complete_pairs <= 0
        or isinstance(incomplete_pairs, bool)
        or not isinstance(incomplete_pairs, int)
        or incomplete_pairs < 0
        or isinstance(suite_pairs, bool)
        or not isinstance(suite_pairs, int)
        or suite_pairs != complete_pairs + incomplete_pairs
        or isinstance(declared_candidate_wins, bool)
        or not isinstance(declared_candidate_wins, int)
        or isinstance(declared_incumbent_wins, bool)
        or not isinstance(declared_incumbent_wins, int)
    ):
        raise ContractError("held-out hard-negative result counts are inconsistent")
    suite = _load_json(Path(str(suite_record["path"])))
    _require_exact_keys(
        suite,
        {
            "held_out",
            "schema_version",
            "selection",
            "source_manifest",
            "states",
            "suite",
            "suite_sha256",
            "validation_seed_manifest",
        },
        where="held-out hard-negative suite",
    )
    suite_unhashed = dict(suite)
    suite_digest = suite_unhashed.pop("suite_sha256")
    if (
        suite["schema_version"] != "a1-held-out-high-regret-suite-v4"
        or suite["held_out"] is not True
        or suite["suite"] != "held_out_high_regret"
        or suite_digest != _digest_value(suite_unhashed)
        or not isinstance(suite["states"], list)
        or len(suite["states"]) != suite_pairs
        or not isinstance(suite["selection"], dict)
        or suite["selection"].get("selected_pairs") != suite_pairs
    ):
        raise ContractError("held-out hard-negative suite manifest is inconsistent")
    states_by_pair: dict[int, tuple[int, int]] = {}
    for index, state in enumerate(suite["states"]):
        if not isinstance(state, dict):
            raise ContractError(f"held-out suite state[{index}] is not an object")
        pair_id = state.get("pair_id")
        game_seed = state.get("game_seed")
        decision_index = state.get("decision_index")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or isinstance(game_seed, bool)
            or not isinstance(game_seed, int)
            or isinstance(decision_index, bool)
            or not isinstance(decision_index, int)
            or pair_id in states_by_pair
        ):
            raise ContractError("held-out suite state identity is malformed or repeated")
        states_by_pair[pair_id] = (game_seed, decision_index)
    games = report.get("games")
    if not isinstance(games, list):
        raise ContractError("held-out hard-negative report games are malformed")
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise ContractError(f"held-out hard-negative game[{index}] is not an object")
        pair_id = game.get("pair_id")
        if isinstance(pair_id, bool) or not isinstance(pair_id, int):
            raise ContractError("held-out hard-negative game has no integer pair identity")
        by_pair.setdefault(pair_id, []).append(game)
    if set(by_pair) != set(states_by_pair):
        raise ContractError("held-out report games do not bind the exact suite pair set")
    complete: list[dict[str, Any]] = []
    recomputed_diagnostics = {
        "incomplete_pairs": 0,
        "ll_pairs": 0,
        "split_pairs": 0,
        "ww_pairs": 0,
    }
    for pair_id, pair in by_pair.items():
        archived_identity = states_by_pair[pair_id]
        if (
            len(pair) != 2
            or {game.get("orientation") for game in pair}
            != {"candidate_red", "candidate_blue"}
        ):
            raise ContractError(
                "held-out hard-negative pair lacks exact candidate orientations"
            )
        for game in pair:
            if (
                (game.get("archived_game_seed"), game.get("archived_decision_index"))
                != archived_identity
                or game.get("game_seed") != archived_identity[0]
            ):
                raise ContractError(
                    "held-out report game does not bind its suite state identity"
                )
        pair_complete = all(
            game.get("terminated") is True
            and game.get("truncated") is False
            and isinstance(game.get("candidate_won"), bool)
            for game in pair
        )
        if not pair_complete:
            if any(
                not (
                    (
                        game.get("terminated") is True
                        and game.get("truncated") is False
                        and isinstance(game.get("candidate_won"), bool)
                    )
                    or (
                        game.get("terminated") is False
                        and game.get("truncated") is True
                        and game.get("candidate_won") is None
                    )
                )
                for game in pair
            ):
                raise ContractError("held-out hard-negative incomplete pair is malformed")
            recomputed_diagnostics["incomplete_pairs"] += 1
            continue
        complete.extend(pair)
        pair_wins = sum(game["candidate_won"] is True for game in pair)
        bucket = "ww_pairs" if pair_wins == 2 else "ll_pairs" if pair_wins == 0 else "split_pairs"
        recomputed_diagnostics[bucket] += 1
    candidate_wins = sum(game["candidate_won"] is True for game in complete)
    incumbent_wins = sum(game["candidate_won"] is False for game in complete)
    if (
        len(complete) != 2 * complete_pairs
        or recomputed_diagnostics != raw["pair_diagnostics"]
        or candidate_wins != declared_candidate_wins
        or incumbent_wins != declared_incumbent_wins
        or candidate_wins + incumbent_wins != 2 * complete_pairs
        or raw["pentanomial_sprt"].get("pairs") != complete_pairs
        or any(
            raw["pentanomial_sprt"].get(key) != value
            for key, value in recomputed_diagnostics.items()
            if key != "incomplete_pairs"
        )
    ):
        raise ContractError("held-out hard-negative result counts are inconsistent")
    win_rate = candidate_wins / (candidate_wins + incumbent_wins)
    declared_win_rate = raw["complete_candidate_win_rate"]
    if (
        isinstance(declared_win_rate, bool)
        or not isinstance(declared_win_rate, (int, float))
        or not math.isfinite(float(declared_win_rate))
        or not math.isclose(
            float(declared_win_rate),
            win_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ContractError("held-out hard-negative win rate is inconsistent")
    elo_values: dict[str, float] = {}
    for key in ("elo0", "elo1"):
        value = config.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ContractError(
                f"held-out hard-negative evaluation config has invalid {key}"
            )
        elo_values[key] = float(value)
    replayed_pentanomial = evaluate_pentanomial_sprt(
        counts=(
            recomputed_diagnostics["ll_pairs"],
            recomputed_diagnostics["split_pairs"],
            recomputed_diagnostics["ww_pairs"],
        ),
        elo0=elo_values["elo0"],
        elo1=elo_values["elo1"],
        alpha=0.05,
        beta=0.05,
    )
    if (
        raw["pentanomial_sprt"] != replayed_pentanomial
        or report.get("pentanomial_sprt") != replayed_pentanomial
    ):
        raise ContractError(
            "held-out hard-negative pentanomial SPRT does not equal replay"
        )
    return {"artifact": artifact, "suite_manifest": suite_record}


def _validate_internal_hard_negative_evidence(
    raw: Any,
    *,
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("internal hard-negative evidence must be an object")
    required = {
        "artifact",
        "candidate_win_rate",
        "candidate_wins",
        "complete_pairs",
        "games",
        "incumbent_wins",
        "pentanomial_decision",
        "strict_superiority_decision",
    }
    _require_exact_keys(raw, required, where="internal hard-negative evidence")
    artifact = _verified_sized_artifact_ref(
        raw["artifact"],
        where="internal hard-negative evidence artifact",
        kind="hard_negative_internal_report",
    )
    report = _load_json(Path(str(artifact["path"])))
    if report.get("schema_version") != "a1-complete-internal-cohort-receipt-v1":
        raise ContractError("internal hard-negative report schema is unsupported")
    if report.get("registry_mutation_authorized") is not False:
        raise ContractError("internal hard-negative report authorizes registry mutation")
    if report.get("errors") != [] or report.get("truncations") != 0:
        raise ContractError("internal hard-negative report is incomplete or errored")
    receipt_unhashed = dict(report)
    receipt_digest = receipt_unhashed.pop("receipt_sha256", None)
    if receipt_digest != _digest_value(receipt_unhashed):
        raise ContractError("internal hard-negative receipt digest mismatch")
    _require_artifact_identity(report.get("candidate"), candidate, where="internal candidate")
    _require_artifact_identity(report.get("baseline"), incumbent, where="internal incumbent")
    comparisons = {
        "candidate_win_rate": report.get("candidate_win_rate"),
        "candidate_wins": report.get("candidate_wins"),
        "complete_pairs": report.get("complete_pairs"),
        "games": report.get("games"),
        "incumbent_wins": report.get("baseline_wins"),
        "pentanomial_decision": (
            report.get("pentanomial_sprt", {}).get("decision")
            if isinstance(report.get("pentanomial_sprt"), dict)
            else None
        ),
        "strict_superiority_decision": (
            report.get("superiority_pentanomial_sprt", {}).get("decision")
            if isinstance(report.get("superiority_pentanomial_sprt"), dict)
            else None
        ),
    }
    if any(raw[key] != value for key, value in comparisons.items()):
        raise ContractError("internal hard-negative summary drifts from its receipt")
    integer_fields = ("candidate_wins", "complete_pairs", "games", "incumbent_wins")
    if (
        any(
            isinstance(raw[field], bool) or not isinstance(raw[field], int)
            for field in integer_fields
        )
        or raw["complete_pairs"] <= 0
        or raw["candidate_wins"] < 0
        or raw["incumbent_wins"] < 0
        or raw["games"] != 2 * raw["complete_pairs"]
        or raw["candidate_wins"] + raw["incumbent_wins"] != raw["games"]
    ):
        raise ContractError("internal hard-negative result counts are inconsistent")
    diagnostics = report.get("pair_diagnostics")
    required_diagnostics = {
        "incomplete_pairs",
        "ll_pairs",
        "split_pairs",
        "ww_pairs",
    }
    if not isinstance(diagnostics, dict) or set(diagnostics) != required_diagnostics:
        raise ContractError("internal hard-negative pair diagnostics are malformed")
    if any(
        isinstance(diagnostics[key], bool)
        or not isinstance(diagnostics[key], int)
        or diagnostics[key] < 0
        for key in required_diagnostics
    ):
        raise ContractError("internal hard-negative pair diagnostics are malformed")
    ll_pairs = diagnostics["ll_pairs"]
    split_pairs = diagnostics["split_pairs"]
    ww_pairs = diagnostics["ww_pairs"]
    if (
        diagnostics["incomplete_pairs"] != 0
        or ll_pairs + split_pairs + ww_pairs != raw["complete_pairs"]
        or 2 * ww_pairs + split_pairs != raw["candidate_wins"]
        or 2 * ll_pairs + split_pairs != raw["incumbent_wins"]
    ):
        raise ContractError("internal hard-negative pair diagnostics are inconsistent")
    counts = (ll_pairs, split_pairs, ww_pairs)
    replayed_promotion = evaluate_pentanomial_sprt(
        counts=counts,
        elo0=-10.0,
        elo1=15.0,
        alpha=0.05,
        beta=0.05,
    )
    replayed_superiority = evaluate_pentanomial_sprt(
        counts=counts,
        elo0=0.0,
        elo1=15.0,
        alpha=0.05,
        beta=0.05,
    )
    if report.get("pentanomial_sprt") != replayed_promotion:
        raise ContractError(
            "internal hard-negative promotion SPRT does not equal replay"
        )
    if report.get("superiority_pentanomial_sprt") != replayed_superiority:
        raise ContractError(
            "internal hard-negative superiority SPRT does not equal replay"
        )
    candidate_win_rate = raw["candidate_win_rate"]
    if (
        isinstance(candidate_win_rate, bool)
        or not isinstance(candidate_win_rate, (int, float))
        or not math.isclose(
            float(candidate_win_rate),
            raw["candidate_wins"] / raw["games"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ContractError("internal hard-negative win rate is inconsistent")
    return {"artifact": artifact}


def _validate_external_report_identity(
    report: dict[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    where: str,
) -> dict[str, Any]:
    _require_checkpoint_path(
        report.get("candidate_checkpoint"), checkpoint, where=f"{where} checkpoint"
    )
    if report.get("candidate_checkpoint_sha256") != checkpoint["sha256"]:
        raise ContractError(f"{where} checkpoint hash drift")
    if report.get("baseline_bot") != "catanatron_value":
        raise ContractError(f"{where} is not the matched external panel")
    if (
        report.get("errors") != []
        or report.get("worker_errors") != []
        or report.get("games_errored") != 0
        or report.get("games_truncated") != 0
    ):
        raise ContractError(f"{where} is incomplete or errored")
    binding = report.get("evaluation_binding")
    if not isinstance(binding, dict) or binding.get("schema_version") != (
        "a1-evaluation-baseline-binding-v2"
    ):
        raise ContractError(f"{where} evaluation binding is malformed")
    _require_artifact_identity(
        binding.get("authoritative_incumbent"),
        incumbent,
        where=f"{where} authoritative incumbent",
        allow_extra=True,
    )
    _require_artifact_identity(
        binding.get("baseline"), incumbent, where=f"{where} baseline"
    )
    fleet_merge = report.get("fleet_merge")
    if not isinstance(fleet_merge, dict) or fleet_merge.get("schema_version") != (
        "a1-fleet-evaluation-pool-v1"
    ):
        raise ContractError(f"{where} fleet merge is malformed")
    _require_artifact_identity(
        fleet_merge.get("checkpoint"), checkpoint, where=f"{where} pooled checkpoint"
    )
    effective_search = report.get("effective_search_config")
    if not isinstance(effective_search, dict) or not effective_search:
        raise ContractError(f"{where} has no effective search config")
    search_sha256 = _digest_value(effective_search)
    if (
        report.get("search_config") != effective_search
        or fleet_merge.get("effective_search_config_sha256") != search_sha256
    ):
        raise ContractError(f"{where} effective search identity drift")
    games = report.get("games")
    games_played = report.get("games_played")
    complete_pairs = report.get("complete_pairs")
    if (
        not isinstance(games, list)
        or isinstance(games_played, bool)
        or not isinstance(games_played, int)
        or isinstance(complete_pairs, bool)
        or not isinstance(complete_pairs, int)
        or games_played != len(games)
        or games_played != 2 * complete_pairs
        or complete_pairs <= 0
    ):
        raise ContractError(f"{where} game cohort is malformed")
    outcomes: dict[tuple[int, str, int], bool] = {}
    pair_orientations: dict[tuple[int, int], set[str]] = {}
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise ContractError(f"{where} game[{index}] is not an object")
        seed = game.get("game_seed")
        orientation = game.get("orientation")
        pair_id = game.get("source_pair_id")
        won = game.get("candidate_won")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or orientation not in {"candidate_first", "candidate_second"}
            or isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or not isinstance(won, bool)
            or game.get("error") is not None
            or game.get("terminated") is not True
            or game.get("truncated") is not False
            or game.get("engine_divergence") is not False
        ):
            raise ContractError(f"{where} game[{index}] is incomplete or malformed")
        key = (seed, orientation, pair_id)
        if key in outcomes:
            raise ContractError(f"{where} repeats matched game identity {key}")
        outcomes[key] = won
        pair_orientations.setdefault((seed, pair_id), set()).add(orientation)
    if (
        len(pair_orientations) != complete_pairs
        or any(
            orientations != {"candidate_first", "candidate_second"}
            for orientations in pair_orientations.values()
        )
    ):
        raise ContractError(f"{where} does not contain exact paired orientations")
    wins = sum(outcomes.values())
    losses = games_played - wins
    reported_rate = report.get("candidate_win_rate")
    if (
        report.get("candidate_wins") != wins
        or report.get("baseline_wins") != losses
        or isinstance(reported_rate, bool)
        or not isinstance(reported_rate, (int, float))
        or not math.isclose(
            float(reported_rate), wins / games_played, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise ContractError(f"{where} win summary is inconsistent with its games")
    return {
        "outcomes": outcomes,
        "search_sha256": search_sha256,
    }


def _validate_external_hard_negative_evidence(
    raw: Any,
    *,
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("external hard-negative evidence must be an object")
    required = {
        "candidate_artifact",
        "candidate_win_rate",
        "delta",
        "differential_artifact",
        "incumbent_artifact",
        "incumbent_win_rate",
        "matched_games",
        "matched_pairs",
        "mcnemar_exact_two_sided_p",
    }
    _require_exact_keys(raw, required, where="external hard-negative evidence")
    candidate_artifact = _verified_sized_artifact_ref(
        raw["candidate_artifact"],
        where="external hard-negative candidate artifact",
        kind="hard_negative_external_candidate_report",
    )
    incumbent_artifact = _verified_sized_artifact_ref(
        raw["incumbent_artifact"],
        where="external hard-negative incumbent artifact",
        kind="hard_negative_external_incumbent_report",
    )
    differential_artifact = _verified_sized_artifact_ref(
        raw["differential_artifact"],
        where="external hard-negative differential artifact",
        kind="hard_negative_external_differential",
    )
    candidate_report = _load_json(Path(str(candidate_artifact["path"])))
    incumbent_report = _load_json(Path(str(incumbent_artifact["path"])))
    differential = _load_json(Path(str(differential_artifact["path"])))
    candidate_cohort = _validate_external_report_identity(
        candidate_report,
        checkpoint=candidate,
        incumbent=incumbent,
        where="external hard-negative candidate report",
    )
    incumbent_cohort = _validate_external_report_identity(
        incumbent_report,
        checkpoint=incumbent,
        incumbent=incumbent,
        where="external hard-negative incumbent report",
    )
    _require_exact_keys(
        differential,
        {
            "both_lose",
            "both_win",
            "candidate_only_wins",
            "candidate_win_rate",
            "champion_only_wins",
            "champion_win_rate",
            "delta",
            "matched_games",
            "matched_pairs",
            "mcnemar_exact_two_sided_p",
            "paired_seed_delta_bootstrap_95ci",
            "schema_version",
        },
        where="external hard-negative differential",
    )
    if differential["schema_version"] != "a1-matched-external-differential-v1":
        raise ContractError("external hard-negative differential schema is unsupported")
    if candidate_cohort["search_sha256"] != incumbent_cohort["search_sha256"]:
        raise ContractError("external hard-negative panels used different search recipes")
    candidate_outcomes = candidate_cohort["outcomes"]
    incumbent_outcomes = incumbent_cohort["outcomes"]
    if set(candidate_outcomes) != set(incumbent_outcomes):
        raise ContractError("external hard-negative panels are not the same matched cohort")
    both_win = sum(
        candidate_outcomes[key] and incumbent_outcomes[key]
        for key in candidate_outcomes
    )
    both_lose = sum(
        not candidate_outcomes[key] and not incumbent_outcomes[key]
        for key in candidate_outcomes
    )
    candidate_only = sum(
        candidate_outcomes[key] and not incumbent_outcomes[key]
        for key in candidate_outcomes
    )
    incumbent_only = sum(
        not candidate_outcomes[key] and incumbent_outcomes[key]
        for key in candidate_outcomes
    )
    discordant = candidate_only + incumbent_only
    mcnemar_p = (
        1.0
        if discordant == 0
        else min(
            1.0,
            2.0
            * sum(
                math.comb(discordant, k)
                for k in range(min(candidate_only, incumbent_only) + 1)
            )
            / (2**discordant),
        )
    )
    recomputed_differential = {
        "both_lose": both_lose,
        "both_win": both_win,
        "candidate_only_wins": candidate_only,
        "champion_only_wins": incumbent_only,
        "matched_games": len(candidate_outcomes),
        "matched_pairs": len(candidate_outcomes) // 2,
    }
    if any(
        differential[key] != value
        for key, value in recomputed_differential.items()
    ) or not math.isclose(
        float(differential["delta"]),
        float(differential["candidate_win_rate"])
        - float(differential["champion_win_rate"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        float(differential["mcnemar_exact_two_sided_p"]),
        mcnemar_p,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ContractError(
            "external hard-negative differential does not match paired game outcomes"
        )
    comparisons = {
        "candidate_win_rate": differential["candidate_win_rate"],
        "delta": differential["delta"],
        "incumbent_win_rate": differential["champion_win_rate"],
        "matched_games": differential["matched_games"],
        "matched_pairs": differential["matched_pairs"],
        "mcnemar_exact_two_sided_p": differential["mcnemar_exact_two_sided_p"],
    }
    if any(raw[key] != value for key, value in comparisons.items()):
        raise ContractError("external hard-negative summary drifts from its differential")
    if (
        isinstance(raw["matched_games"], bool)
        or not isinstance(raw["matched_games"], int)
        or isinstance(raw["matched_pairs"], bool)
        or not isinstance(raw["matched_pairs"], int)
        or raw["matched_pairs"] <= 0
    ):
        raise ContractError("external hard-negative cohort counts are malformed")
    if (
        candidate_report.get("candidate_win_rate") != raw["candidate_win_rate"]
        or incumbent_report.get("candidate_win_rate") != raw["incumbent_win_rate"]
        or candidate_report.get("complete_pairs") != raw["matched_pairs"]
        or incumbent_report.get("complete_pairs") != raw["matched_pairs"]
        or candidate_report.get("games_played") != raw["matched_games"]
        or incumbent_report.get("games_played") != raw["matched_games"]
        or raw["matched_games"] != 2 * raw["matched_pairs"]
    ):
        raise ContractError("external hard-negative cohort counts are inconsistent")
    return {
        "candidate_artifact": candidate_artifact,
        "incumbent_artifact": incumbent_artifact,
        "differential_artifact": differential_artifact,
    }


def _rich_hard_negative_selection_record(
    path: Path,
    payload: dict[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        RICH_HARD_NEGATIVE_SELECTION_FIELDS,
        where="rich hard-negative selection evidence",
    )
    if payload["schema_version"] != "a1-hard-negative-selection-v1":
        raise ContractError("hard-negative selection evidence schema is unsupported")
    unhashed = dict(payload)
    declared = unhashed.pop("selection_sha256")
    if declared != _digest_value(unhashed):
        raise ContractError("hard-negative selection evidence digest mismatch")
    if (
        payload["status"] != "selected"
        or payload["selection_role"] != "hard_negative"
        or payload["promotion_eligible"] is not False
        or payload["promotion_decision"] != "not_authorized_nonparent_diagnostic"
        or payload["registry_mutation_authorized"] is not False
        or payload["opponent_pool_mutation_performed"] is not False
    ):
        raise ContractError("rich hard-negative selection is promotion-capable or inactive")
    basis = payload["selection_basis"]
    if not isinstance(basis, dict):
        raise ContractError("rich hard-negative selection basis must be an object")
    _require_exact_keys(
        basis,
        {"kind", "not_a_strength_promotion", "rationale"},
        where="rich hard-negative selection basis",
    )
    if (
        basis["kind"] != "architecture_diverse_near_peer"
        or basis["not_a_strength_promotion"] is not True
        or not isinstance(basis["rationale"], list)
        or not basis["rationale"]
        or any(not isinstance(item, str) or not item.strip() for item in basis["rationale"])
    ):
        raise ContractError("rich hard-negative selection basis is invalid")
    limitations = payload["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise ContractError("rich hard-negative limitations must be non-empty strings")
    candidate = _verified_sized_artifact_ref(
        payload["candidate"],
        where="rich hard-negative candidate",
        kind="hard_negative_candidate",
    )
    expected_checkpoint = checkpoint.expanduser().resolve(strict=True)
    if (
        Path(str(candidate["path"])) != expected_checkpoint
        or candidate["sha256"] != checkpoint_sha256
    ):
        raise ContractError("hard-negative selection evidence binds different bytes")
    incumbent = _verified_sized_artifact_ref(
        payload["authoritative_incumbent"],
        where="rich hard-negative authoritative incumbent",
        kind="hard_negative_authoritative_incumbent",
    )
    bundle = _verified_sized_artifact_ref(
        payload["candidate_bundle"],
        where="rich hard-negative candidate bundle",
        kind="hard_negative_candidate_bundle",
    )
    bundle_artifacts, bundle_files = _validate_hard_negative_bundle(
        bundle, candidate=candidate, incumbent=incumbent
    )
    evidence = payload["evidence"]
    if not isinstance(evidence, dict):
        raise ContractError("rich hard-negative evidence must be an object")
    _require_exact_keys(
        evidence,
        {"held_out_high_regret_v5", "matched_external_v5", "matched_internal_v5"},
        where="rich hard-negative evidence",
    )
    held_out = _validate_held_out_hard_negative_evidence(
        evidence["held_out_high_regret_v5"],
        candidate=candidate,
        incumbent=incumbent,
    )
    matched_external = _validate_external_hard_negative_evidence(
        evidence["matched_external_v5"],
        candidate=candidate,
        incumbent=incumbent,
    )
    matched_internal = _validate_internal_hard_negative_evidence(
        evidence["matched_internal_v5"],
        candidate=candidate,
        incumbent=incumbent,
    )
    record = _file_record(path, kind="hard_negative_selection")
    record.update(
        {
            "selection_format": "rich-v1",
            "selection_sha256": declared,
            "candidate": candidate,
            "authoritative_incumbent": incumbent,
            "candidate_bundle": bundle,
            "candidate_bundle_artifacts": bundle_artifacts,
            "candidate_bundle_files": bundle_files,
            "evidence": {
                "held_out_high_regret_v5": held_out,
                "matched_external_v5": matched_external,
                "matched_internal_v5": matched_internal,
            },
        }
    )
    return record


def _hard_negative_selection_record(
    path: Path, *, checkpoint: Path, checkpoint_sha256: str
) -> dict[str, Any]:
    payload = _load_json(path)
    rich_only = RICH_HARD_NEGATIVE_SELECTION_FIELDS - (
        LEGACY_HARD_NEGATIVE_SELECTION_FIELDS | {"schema_version", "selection_sha256"}
    )
    if set(payload) & rich_only:
        return _rich_hard_negative_selection_record(
            path,
            payload,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        )
    return _legacy_hard_negative_selection_record(
        path,
        payload,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )


def _hard_negative_selection_artifacts(
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = [dict(selection)]
    if selection.get("selection_format") != "rich-v1":
        evidence = selection.get("evaluation_evidence")
        if not isinstance(evidence, dict):
            raise ContractError("legacy hard-negative evaluation evidence is missing")
        records.append(evidence)
        return records
    for key in ("candidate", "authoritative_incumbent", "candidate_bundle"):
        record = selection.get(key)
        if not isinstance(record, dict):
            raise ContractError(f"rich hard-negative {key} binding is missing")
        records.append(record)
    artifacts = selection.get("candidate_bundle_artifacts")
    files = selection.get("candidate_bundle_files")
    evidence = selection.get("evidence")
    if not isinstance(artifacts, dict) or not isinstance(files, list) or not isinstance(
        evidence, dict
    ):
        raise ContractError("rich hard-negative sealed artifact graph is malformed")
    records.extend(artifacts.values())
    records.extend(files)
    for section in evidence.values():
        if not isinstance(section, dict):
            raise ContractError("rich hard-negative sealed evidence section is malformed")
        records.extend(section.values())
    if any(not isinstance(record, dict) for record in records):
        raise ContractError("rich hard-negative sealed artifact record is malformed")
    return records


def _bind_hard_negative_incumbent_to_producer(
    records: Sequence[Mapping[str, Any]],
) -> None:
    producers = [record for record in records if record.get("role") == "producer"]
    hard_negatives = [
        record for record in records if record.get("role") == "hard_negative"
    ]
    if len(producers) != 1 or len(hard_negatives) != 1:
        raise ContractError(
            "rich hard-negative lineage requires one producer and one hard negative"
        )
    producer = producers[0]
    hard = hard_negatives[0]
    selection = hard.get("selection_evidence")
    if not isinstance(selection, dict) or selection.get("selection_format") != "rich-v1":
        return
    incumbent = selection.get("authoritative_incumbent")
    if not isinstance(incumbent, dict) or _artifact_identity(
        incumbent
    ) != _artifact_identity(producer):
        raise ContractError(
            "rich hard-negative authoritative incumbent is not the current producer"
        )


def _checkpoint_records(
    raw: list[dict[str, Any]],
    *,
    base: Path,
    value_readout: str,
    draft_schema: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    sha_to_id: dict[str, str] = {}
    for entry in raw:
        role = str(entry.get("role", ""))
        expected_fields = {"id", "role", "path"}
        if role == "producer":
            expected_fields.add("legacy_scalar_readout_attestation")
        elif draft_schema == DRAFT_SCHEMA and role == "hard_negative":
            expected_fields.update({"version", "selection_evidence"})
        if set(entry) != expected_fields:
            raise ContractError(
                f"checkpoint role={role!r} fields mismatch; expected {sorted(expected_fields)}"
            )
        artifact_id = str(entry["id"])
        if artifact_id in ids:
            raise ContractError(f"duplicate checkpoint id {artifact_id!r}")
        ids.add(artifact_id)
        path = _absolute_ref(str(entry["path"]), base=base)
        record = _file_record(path, kind="checkpoint", artifact_id=artifact_id)
        record["role"] = role
        record["md5"] = _md5(path)
        if draft_schema == DRAFT_SCHEMA and role == "hard_negative":
            version = entry["version"]
            if isinstance(version, bool) or not isinstance(version, int) or version < 0:
                raise ContractError("hard-negative checkpoint version must be >= 0")
            record["version"] = version
            selection_path = _absolute_ref(str(entry["selection_evidence"]), base=base)
            record["selection_evidence"] = _hard_negative_selection_record(
                selection_path,
                checkpoint=path,
                checkpoint_sha256=str(record["sha256"]),
            )
        raw_attestation = entry.get("legacy_scalar_readout_attestation")
        attestation_path = (
            _absolute_ref(str(raw_attestation), base=base)
            if raw_attestation is not None
            else None
        )
        # Every neural opponent is constructed with the same evaluator config
        # and readout as the producer.  A categorical producer therefore makes
        # positive HL-Gauss provenance mandatory for history/hard-negative
        # checkpoints too; otherwise those jobs would fail only after launch.
        record["metadata"] = _checkpoint_metadata(
            path,
            checkpoint_sha256=str(record["sha256"]),
            value_readout=value_readout,
            require_trained_readout=(
                role == "producer" or value_readout == "categorical"
            ),
            legacy_scalar_attestation=attestation_path,
        )
        previous = sha_to_id.get(record["sha256"])
        if previous is not None:
            raise ContractError(
                f"checkpoint ids {previous!r} and {artifact_id!r} contain identical bytes"
            )
        sha_to_id[record["sha256"]] = artifact_id
        records.append(record)
    producers = [record for record in records if record["role"] == "producer"]
    if len(producers) != 1:
        raise ContractError("checkpoints must contain exactly one role=producer")
    return records


def _bind_v3_checkpoint_lineage(
    records: list[dict[str, Any]], handoff_record: dict[str, Any]
) -> None:
    """Bind producer/history versions to the committed promotion transaction."""

    producer = next(record for record in records if record["role"] == "producer")
    history = [record for record in records if record["role"] == "history"]
    hard = [record for record in records if record["role"] == "hard_negative"]
    if len(history) != 1 or len(hard) != 1:
        raise ContractError(
            "v3 source mix requires exactly one recent-history and one hard-negative checkpoint"
        )
    producer["version"] = int(handoff_record["registry_version"])
    history_record = history[0]
    if handoff_record.get("mode") == DISASTER_RECOVERY_HANDOFF_MODE:
        safety = handoff_record.get("safety_reference")
        if not isinstance(safety, dict):
            raise ContractError("recovery handoff has no safety reference")
        try:
            safety_path = Path(str(safety["path"])).resolve(strict=True)
            safety_version = safety["version"]
        except (KeyError, OSError) as error:
            raise ContractError(f"recovery safety reference is malformed: {error}") from error
        if (
            safety_path != Path(str(history_record["path"])).resolve(strict=True)
            or safety.get("sha256") != history_record["sha256"]
            or safety.get("relationship") != RECOVERY_REFERENCE_RELATION
            or safety.get("causal_parent_proven") is not False
            or isinstance(safety_version, bool)
            or not isinstance(safety_version, int)
            or safety_version < 1
        ):
            raise ContractError(
                "recent_history scheduler lane must be exact authenticated f7 "
                "recovery reference"
            )
        history_record["version"] = int(safety_version)
        history_record["lineage"] = {
            "relation": RECOVERY_REFERENCE_RELATION,
            "semantic": RECOVERY_REFERENCE_SEMANTIC,
            "causal_parent_proven": False,
            "promotion_proof_recreated": False,
            "recovery_receipt": {
                "path": str(handoff_record["path"]),
                "sha256": handoff_record["sha256"],
                "recovery_receipt_sha256": handoff_record[
                    "recovery_receipt_sha256"
                ],
                "recovery_lineage_id": handoff_record["recovery_lineage_id"],
            },
        }
        _bind_hard_negative_incumbent_to_producer(records)
        return
    handoff_payload = _load_json(Path(str(handoff_record["path"])))
    receipt_ref = handoff_payload.get("promotion_receipt")
    if not isinstance(receipt_ref, dict):
        raise ContractError("post-promotion handoff has no promotion receipt binding")
    receipt_path = Path(str(receipt_ref.get("path", ""))).expanduser().resolve(
        strict=True
    )
    if _sha256(receipt_path) != receipt_ref.get("sha256"):
        raise ContractError("promotion receipt bytes drifted while binding recent history")
    receipt = _load_json(receipt_path)
    displaced = receipt.get("champion")
    if not isinstance(displaced, dict):
        raise ContractError("promotion receipt has no displaced incumbent checkpoint")
    try:
        displaced_path = Path(str(displaced["path"])).expanduser().resolve(strict=True)
        displaced_version = displaced["version"]
    except (KeyError, OSError) as error:
        raise ContractError(f"promotion receipt incumbent is malformed: {error}") from error
    if (
        displaced_path != Path(str(history_record["path"])).resolve(strict=True)
        or displaced.get("sha256") != history_record["sha256"]
        or isinstance(displaced_version, bool)
        or not isinstance(displaced_version, int)
        or displaced_version < 1
    ):
        raise ContractError(
            "recent_history must be the exact incumbent displaced by the producer promotion"
        )
    history_record["version"] = int(displaced_version)
    history_record["lineage"] = {
        "relation": "immediate_displaced_incumbent",
        "promotion_receipt": {
            "path": str(receipt_path),
            "sha256": receipt_ref["sha256"],
            "transaction_id": receipt_ref["transaction_id"],
        },
    }
    _bind_hard_negative_incumbent_to_producer(records)


def _category_semantics(
    records: list[dict[str, Any]], handoff_record: Mapping[str, Any]
) -> dict[str, Any]:
    """Separate stable scheduler IDs from their authenticated lineage meaning."""

    by_role = {record["role"]: record for record in records}
    history = by_role["history"]
    hard = by_role["hard_negative"]
    producer = by_role["producer"]
    history_lineage = history.get("lineage")
    if not isinstance(history_lineage, dict):
        raise ContractError("recent-history scheduler lane has no sealed semantics")
    recovery_mode = handoff_record.get("mode") == DISASTER_RECOVERY_HANDOFF_MODE
    if recovery_mode:
        recent = {
            "scheduler_category": "recent_history",
            "semantic": RECOVERY_REFERENCE_SEMANTIC,
            "relation": RECOVERY_REFERENCE_RELATION,
            "causal_parent_proven": False,
            "promotion_proof_recreated": False,
            "checkpoint": {
                key: history[key] for key in ("id", "path", "sha256", "version")
            },
            "recovery_lineage_id": handoff_record["recovery_lineage_id"],
        }
    else:
        recent = {
            "scheduler_category": "recent_history",
            "semantic": "recent_history",
            "relation": "immediate_displaced_incumbent",
            "causal_parent_proven": True,
            "checkpoint": {
                key: history[key] for key in ("id", "path", "sha256", "version")
            },
        }
    return {
        "current_producer": {
            "scheduler_category": "current_producer",
            "semantic": "current_producer",
            "relation": "self_play",
            "checkpoint": {
                key: producer[key] for key in ("id", "path", "sha256", "version")
            },
        },
        "recent_history": recent,
        "hard_negative": {
            "scheduler_category": "hard_negative",
            "semantic": "hard_negative",
            "relation": "sealed_hard_negative_selection",
            "checkpoint": {
                key: hard[key] for key in ("id", "path", "sha256", "version")
            },
        },
    }


def _verify_declared_source_artifacts(
    raw: Any, *, envelope_path: Path
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ContractError(
            f"evidence {envelope_path} must bind at least one source artifact"
        )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ContractError(
                f"evidence {envelope_path} source_artifacts[{index}] must have path and sha256"
            )
        path = _absolute_ref(str(item["path"]), base=envelope_path.parent)
        actual = _sha256(path) if path.is_file() else "<missing>"
        if actual != item["sha256"]:
            raise ContractError(
                f"evidence {envelope_path} source artifact drift at {path}: "
                f"declared {item['sha256']}, actual {actual}"
            )
        records.append({"path": str(path), "sha256": actual})
    return records


def _validate_learner_objective(value: dict[str, Any]) -> None:
    required = {
        "objective",
        "value_readout",
        "value_categorical_bins",
        "hlgauss_sigma_ratio",
    }
    _require_exact_keys(value, required, where="science.learner_value_objective")
    objective = str(value["objective"])
    readout = str(value["value_readout"])
    if objective == "hlgauss":
        if readout != "categorical" or value["value_categorical_bins"] != 33:
            raise ContractError(
                "HL-Gauss learner objective requires categorical readout and 33 bins"
            )
        if float(value["hlgauss_sigma_ratio"]) != 0.75:
            raise ContractError(
                "HL-Gauss learner objective requires the locked sigma ratio 0.75"
            )
    elif objective == "mse":
        if readout != "scalar" or value["value_categorical_bins"] is not None:
            raise ContractError(
                "MSE learner objective requires scalar readout and null categorical bins"
            )
        if value["hlgauss_sigma_ratio"] is not None:
            raise ContractError(
                "MSE learner objective requires null HL-Gauss sigma ratio"
            )
    else:
        raise ContractError("learner objective must be 'hlgauss' or 'mse'")


def _validate_learner_training_recipe(
    value: dict[str, Any],
    *,
    expected_recipe: Mapping[str, Any] = EXPECTED_LEARNER_TRAINING_RECIPE,
) -> None:
    """Require the exact effective one-dose recipe, including JSON types."""

    _require_exact_keys(
        value,
        set(expected_recipe),
        where="science.learner_training_recipe",
    )
    for key, expected in expected_recipe.items():
        actual = value[key]
        if type(actual) is not type(expected):
            raise ContractError(
                f"science.learner_training_recipe.{key} must have JSON type "
                f"{type(expected).__name__}, got {type(actual).__name__}"
            )
        if actual != expected:
            raise ContractError(
                f"science.learner_training_recipe.{key} must equal the locked "
                f"pre-wave value {expected!r}, got {actual!r}"
            )
    expected_global_batch = (
        int(value["batch_size"])
        * int(value["grad_accum_steps"])
        * int(value["world_size"])
    )
    if int(value["global_batch_size"]) != expected_global_batch:
        raise ContractError(
            "science.learner_training_recipe.global_batch_size does not equal "
            "batch_size * grad_accum_steps * world_size"
        )


def _validate_a0_evidence(
    payload: dict[str, Any], *, path: Path, learner_objective: dict[str, Any]
) -> dict[str, Any]:
    # A typed-looking JSON object is not evidence.  Replay the exact A0
    # adjudicator in-process from its sealed lock/result (and, on the adoption
    # path, its calibration/policy inputs) and require byte-semantic equality.
    sealed_for_replay = dict(payload.get("sealed_inputs", {}))
    lock_for_replay = _absolute_ref(
        str(sealed_for_replay.get("lock", "")), base=path.parent
    )
    result_for_replay = _absolute_ref(
        str(sealed_for_replay.get("training_result", "")), base=path.parent
    )
    try:
        a0_lock = _load_json(lock_for_replay)
        repo_root = Path(
            str(
                a0_lock.get("repo_root_at_seal")
                or a0_lock.get("artifact_root_at_seal")
                or REPO_ROOT
            )
        ).expanduser().absolute()
        calibration = payload.get("calibration_artifacts")
        scalar_calibration: Path | None = None
        hl_calibration: Path | None = None
        if isinstance(calibration, dict):
            scalar_raw = dict(calibration.get("scalar") or {}).get("calibration")
            hl_raw = dict(calibration.get("hlgauss33") or {}).get("calibration")
            if scalar_raw is not None:
                scalar_calibration = _absolute_ref(
                    str(scalar_raw), base=path.parent
                )
            if hl_raw is not None:
                hl_calibration = _absolute_ref(str(hl_raw), base=path.parent)
        policy = payload.get("policy_drift")
        policy_path = (
            _absolute_ref(str(policy["artifact"]), base=path.parent)
            if isinstance(policy, dict) and policy.get("artifact")
            else None
        )
        replayed = a0_binding.build_binding_verdict(
            lock_path=lock_for_replay,
            result_path=result_for_replay,
            scalar_calibration_path=scalar_calibration,
            hl_calibration_path=hl_calibration,
            policy_drift_path=policy_path,
            repo_root=repo_root,
        )
    except Exception as error:  # noqa: BLE001 - any replay failure blocks sealing.
        raise ContractError(f"A0 evidence {path} failed semantic replay: {error}") from error
    if payload != replayed:
        raise ContractError(
            f"A0 evidence {path} does not equal the replayed binding verdict"
        )

    if payload.get("schema_version") != A0_EVIDENCE_SCHEMA:
        raise ContractError(
            f"A0 evidence {path} schema must be {A0_EVIDENCE_SCHEMA!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("a0_interpretable") is not True:
        raise ContractError(f"A0 evidence {path} has a0_interpretable != true")
    if payload.get("a0_stage_complete") is not True:
        raise ContractError(f"A0 evidence {path} has a0_stage_complete != true")
    if payload.get("a0_binding_pass") is not True:
        raise ContractError(f"A0 evidence {path} has a0_binding_pass != true")
    if type(payload.get("hlgauss_adoption_pass")) is not bool:
        raise ContractError(
            f"A0 evidence {path} must carry boolean hlgauss_adoption_pass"
        )
    gates = dict(payload.get("gates", {}))
    required_gates = {
        "scalar_reproduction",
        "hl_training_stability",
        "exact_validation_seeds",
        "categorical_readout_provenance",
        "calibration",
        "policy_drift",
    }
    if set(gates) != required_gates or any(
        value is not None and type(value) is not bool for value in gates.values()
    ):
        raise ContractError(f"A0 evidence {path} has malformed binding gates")
    if gates["scalar_reproduction"] is not True:
        raise ContractError(f"A0 evidence {path} failed scalar reproduction")
    # A failed HL calibration/policy/readout gate is a legitimate, informative
    # retain-scalar decision.  The binding-verdict producer owns the stronger
    # cross-artifact checks; this consumer requires its explicit stage-complete
    # and interpretable decision plus immutable source bytes below.
    decision = dict(payload.get("decision", {}))
    required_decision = {
        "status",
        "learner_objective",
        "learner_value_readout",
        "mechanism_checkpoint_sha256",
        "mechanism_checkpoint_is_production_candidate",
    }
    _require_exact_keys(
        decision, required_decision, where=f"A0 evidence {path} decision"
    )
    expected_statuses = (
        {"adopt_hlgauss_for_a1", "adopt_hlgauss"}
        if learner_objective["objective"] == "hlgauss"
        else {"retain_scalar_for_a1", "retain_scalar"}
    )
    if decision["status"] not in expected_statuses:
        raise ContractError(
            f"A0 evidence selected {decision['status']!r}, contract learner objective requires "
            f"one of {sorted(expected_statuses)!r}"
        )
    if (
        decision["learner_objective"] != learner_objective["objective"]
        or decision["learner_value_readout"] != learner_objective["value_readout"]
    ):
        raise ContractError("A0 learner objective/readout does not match the contract")
    if decision["mechanism_checkpoint_is_production_candidate"] is not False:
        raise ContractError(
            "A0 mechanism checkpoint must remain evidence-only, not production"
        )
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(decision["mechanism_checkpoint_sha256"])
    ):
        raise ContractError("A0 mechanism checkpoint SHA-256 is malformed")
    should_adopt_hl = learner_objective["objective"] == "hlgauss"
    if payload["hlgauss_adoption_pass"] is not should_adopt_hl:
        raise ContractError(
            "A0 HL-adoption verdict does not match the selected learner objective"
        )
    records: list[dict[str, Any]] = []
    sealed = dict(payload.get("sealed_inputs", {}))
    for path_key, sha_key in (
        ("lock", "lock_sha256"),
        ("training_result", "training_result_sha256"),
    ):
        source = _absolute_ref(str(sealed.get(path_key, "")), base=path.parent)
        actual = _sha256(source) if source.is_file() else "<missing>"
        declared = str(sealed.get(sha_key, ""))
        if actual.removeprefix("sha256:") != declared.removeprefix("sha256:"):
            raise ContractError(f"A0 evidence {path} {path_key} artifact hash drift")
        records.append({"path": str(source), "sha256": actual})
    for arm, raw in dict(payload.get("calibration_artifacts") or {}).items():
        artifact = dict(raw)
        for path_key, sha_key in (
            ("calibration", "calibration_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("report", "report_sha256"),
            ("manifest", "manifest_sha256"),
        ):
            if path_key not in artifact or sha_key not in artifact:
                continue
            source = _absolute_ref(str(artifact[path_key]), base=path.parent)
            actual = _sha256(source) if source.is_file() else "<missing>"
            if actual.removeprefix("sha256:") != str(artifact[sha_key]).removeprefix(
                "sha256:"
            ):
                raise ContractError(
                    f"A0 evidence {path} {arm}.{path_key} artifact hash drift"
                )
            records.append({"path": str(source), "sha256": actual})
    policy = dict(payload.get("policy_drift") or {})
    if "artifact" in policy and "artifact_sha256" in policy:
        source = _absolute_ref(str(policy["artifact"]), base=path.parent)
        actual = _sha256(source) if source.is_file() else "<missing>"
        if actual.removeprefix("sha256:") != str(
            policy["artifact_sha256"]
        ).removeprefix("sha256:"):
            raise ContractError(f"A0 evidence {path} policy-drift artifact hash drift")
        records.append({"path": str(source), "sha256": actual})
    if not records:
        raise ContractError(f"A0 evidence {path} binds no source artifacts")
    # Bind the adjudicator implementation as well as its inputs.  The A0
    # verdict schema predates the generic stage envelope's explicit
    # ``adjudicator`` field, so the consumer supplies this known path.
    records.append(
        _file_record(
            REPO_ROOT / "tools" / "a0_binding_verdict.py",
            kind="a0_adjudicator",
        )
    )
    return {"decision": decision, "source_artifacts": records}


def _validate_operator_binding_reference(
    raw: Any, *, owner_path: Path, where: str
) -> tuple[Path, dict[str, str]]:
    _require_exact_keys(raw, {"path", "sha256"}, where=where)
    path = _absolute_ref(str(raw["path"]), base=owner_path.parent)
    actual = _sha256(path) if path.is_file() else "<missing>"
    if actual != raw["sha256"]:
        raise ContractError(
            f"{where} hash drift at {path}: declared {raw['sha256']}, actual {actual}"
        )
    return path, {"path": str(path), "sha256": actual}


def _validate_operator_binding_evidence(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_stage: str,
    final_search: dict[str, Any],
    final_evaluator: dict[str, Any],
    post_promotion_handoff_path: Path | None = None,
    allow_post_promotion_s1: bool = False,
    recovery_receipt_path: Path | None = None,
    allow_recovery_s1: bool = False,
) -> dict[str, Any]:
    """Validate the narrow no-S2/no-S3 operator-choice bridge.

    This path cannot accept experimental verdicts and the ordinary
    search-adjudication path cannot accept these bindings.  The exact constants
    below make the exception specific to the current n128/no-adaptive operator
    directive rather than a general mechanism for bypassing evidence.
    """

    if expected_stage not in {"s2", "s3"}:
        raise ContractError(
            f"{expected_stage.upper()} cannot use the operator-binding schema"
        )
    common_keys = {
        "schema_version",
        "artifact_kind",
        "stage",
        "operator",
        "passed",
        "decision",
        "reason",
        "binding_time_utc",
        "statement",
        "selected_fields",
        "selected_fields_sha256",
        "source_s1",
        "source_s1_selected_fields_sha256",
        "emitter",
        "artifact_content_sha256",
    }
    expected_keys = set(common_keys)
    if expected_stage == "s3":
        expected_keys.add("source_s2_binding")
    _require_exact_keys(
        payload,
        expected_keys,
        where=f"{expected_stage.upper()} operator binding {path}",
    )
    if payload["schema_version"] != operator_binding.SCHEMA:
        raise ContractError(
            f"{expected_stage.upper()} operator-binding schema mismatch"
        )
    if payload["artifact_kind"] != operator_binding.ARTIFACT_KIND:
        raise ContractError(
            f"{expected_stage.upper()} artifact_kind must explicitly deny strength evidence"
        )
    if payload["stage"] != expected_stage or payload["passed"] is not True:
        raise ContractError(
            f"{expected_stage.upper()} operator binding has wrong stage/passed state"
        )
    expected_decision = "operator_bind" if expected_stage == "s2" else "operator_hold"
    expected_operator = (
        operator_binding.S2_OPERATOR
        if expected_stage == "s2"
        else operator_binding.S3_OPERATOR
    )
    expected_reason = (
        operator_binding.S2_REASON
        if expected_stage == "s2"
        else operator_binding.S3_REASON
    )
    expected_selected = (
        operator_binding.S2_SELECTED
        if expected_stage == "s2"
        else operator_binding.S3_SELECTED
    )
    if payload["decision"] != expected_decision:
        raise ContractError(
            f"{expected_stage.upper()} operator-binding decision mismatch"
        )
    if payload["operator"] != expected_operator:
        raise ContractError(
            f"{expected_stage.upper()} operator-binding operator mismatch"
        )
    if payload["reason"] != expected_reason:
        raise ContractError(f"{expected_stage.upper()} operator-binding reason mismatch")
    if payload["statement"] != operator_binding.STATEMENT:
        raise ContractError(
            f"{expected_stage.upper()} must state that the binding is not strength evidence"
        )
    timestamp = payload["binding_time_utc"]
    if not isinstance(timestamp, str):
        raise ContractError(f"{expected_stage.upper()} binding_time_utc must be a string")
    try:
        parsed_time = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(
            f"{expected_stage.upper()} binding_time_utc is not ISO-8601"
        ) from error
    if parsed_time.tzinfo is None or parsed_time.utcoffset() != dt.timedelta(0):
        raise ContractError(
            f"{expected_stage.upper()} binding_time_utc must carry an explicit UTC offset"
        )
    selected = payload["selected_fields"]
    if selected != expected_selected:
        raise ContractError(
            f"{expected_stage.upper()} operator binding must select exactly {expected_selected}"
        )
    if payload["selected_fields_sha256"] != _digest_value(selected):
        raise ContractError(
            f"{expected_stage.upper()} operator-binding selected_fields_sha256 mismatch"
        )
    mismatches = {
        key: (selected[key], final_search[key])
        for key in selected
        if selected[key] != final_search[key]
    }
    if mismatches:
        raise ContractError(
            f"{expected_stage.upper()} selected search fields mismatch final contract: {mismatches}"
        )
    unhashed = dict(payload)
    declared_content_digest = unhashed.pop("artifact_content_sha256")
    if declared_content_digest != _digest_value(unhashed):
        raise ContractError(
            f"{expected_stage.upper()} operator-binding self digest mismatch"
        )

    emitter_path, emitter_record = _validate_operator_binding_reference(
        payload["emitter"],
        owner_path=path,
        where=f"{expected_stage.upper()} operator-binding emitter",
    )
    if not emitter_path.as_posix().endswith("tools/search_operator_binding.py"):
        raise ContractError(
            f"{expected_stage.upper()} operator binding has an untrusted emitter"
        )
    s1_path, s1_record = _validate_operator_binding_reference(
        payload["source_s1"],
        owner_path=path,
        where=f"{expected_stage.upper()} source S1",
    )
    s1_payload = _load_json(s1_path)
    s1_semantic = _validate_search_stage_evidence(
        s1_payload,
        path=s1_path,
        expected_stage="s1",
        final_search=final_search,
        final_evaluator=final_evaluator,
        post_promotion_handoff_path=post_promotion_handoff_path,
        allow_post_promotion_s1=allow_post_promotion_s1,
        recovery_receipt_path=recovery_receipt_path,
        allow_recovery_s1=allow_recovery_s1,
    )
    if (
        payload["source_s1_selected_fields_sha256"]
        != s1_semantic["selected_fields_sha256"]
    ):
        raise ContractError(
            f"{expected_stage.upper()} source S1 selected-fields digest mismatch"
        )

    if expected_stage == "s2":
        try:
            replayed_s2, _ = operator_binding.build_bindings(
                s1_path,
                s2_output_path=path,
                binding_time_utc=timestamp,
                emitter_path=emitter_path,
            )
        except operator_binding.BindingError as error:
            raise ContractError(f"S2 operator-binding replay failed: {error}") from error
        if payload != replayed_s2:
            raise ContractError("S2 operator binding does not equal semantic replay")

    records = [s1_record, emitter_record]
    semantic: dict[str, Any] = {
        "decision": payload["decision"],
        "evidence_class": operator_binding.ARTIFACT_KIND,
        "selected_fields": dict(selected),
        "selected_fields_sha256": payload["selected_fields_sha256"],
        "checkpoint": dict(s1_semantic["checkpoint"]),
        "source_artifacts": records,
        "artifact_content_sha256": declared_content_digest,
        "binding_time_utc": timestamp,
    }
    if expected_stage == "s3":
        s2_path, s2_record = _validate_operator_binding_reference(
            payload["source_s2_binding"],
            owner_path=path,
            where="S3 source S2 operator binding",
        )
        s2_payload = _load_json(s2_path)
        s2_semantic = _validate_operator_binding_evidence(
            s2_payload,
            path=s2_path,
            expected_stage="s2",
            final_search=final_search,
            final_evaluator=final_evaluator,
            post_promotion_handoff_path=post_promotion_handoff_path,
            allow_post_promotion_s1=allow_post_promotion_s1,
            recovery_receipt_path=recovery_receipt_path,
            allow_recovery_s1=allow_recovery_s1,
        )
        if s2_payload["source_s1"] != payload["source_s1"]:
            raise ContractError("S3 and S2 operator bindings do not share exact S1 lineage")
        if s2_payload["binding_time_utc"] != timestamp:
            raise ContractError("S3 and S2 operator bindings must share one binding time")
        if s2_semantic["evidence_class"] != operator_binding.ARTIFACT_KIND:
            raise ContractError("S3 predecessor is not an operator binding")
        try:
            replayed_s2, replayed_s3 = operator_binding.build_bindings(
                s1_path,
                s2_output_path=s2_path,
                binding_time_utc=timestamp,
                emitter_path=emitter_path,
            )
        except operator_binding.BindingError as error:
            raise ContractError(f"S3 operator-binding replay failed: {error}") from error
        if s2_payload != replayed_s2 or payload != replayed_s3:
            raise ContractError("S3 operator binding does not equal semantic replay")
        records.append(s2_record)
    return semantic


def _validate_post_promotion_s1_operator_binding(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_stage: str,
    final_search: dict[str, Any],
    final_evaluator: dict[str, Any],
    post_promotion_handoff_path: Path | None,
    allow_post_promotion_s1: bool,
) -> dict[str, Any]:
    """Accept the single c_scale continuity bridge only in a v3 handoff wave."""

    if expected_stage != "s1":
        raise ContractError("post-promotion S1 operator binding is S1-only")
    if not allow_post_promotion_s1 or post_promotion_handoff_path is None:
        raise ContractError(
            "post-promotion S1 operator binding is accepted only by a v3 "
            "post-promotion contract"
        )
    try:
        replayed = operator_binding._replay_post_promotion_s1(path)  # noqa: SLF001
    except operator_binding.BindingError as error:
        raise ContractError(
            f"post-promotion S1 operator-binding replay failed: {error}"
        ) from error
    if replayed != payload:
        raise ContractError(
            "post-promotion S1 operator binding does not equal semantic replay"
        )
    selected = payload.get("selected_fields")
    if not isinstance(selected, dict) or set(selected) != operator_binding.S1_SELECTED_KEYS:
        raise ContractError("post-promotion S1 selected_fields shape mismatch")
    mismatches = {
        key: (selected[key], final_search.get(key))
        for key in selected
        if selected[key] != final_search.get(key)
    }
    if mismatches:
        raise ContractError(
            "post-promotion S1 fields mismatch final contract: " f"{mismatches}"
        )
    if payload.get("selected_fields_sha256") != _digest_value(selected):
        raise ContractError(
            "post-promotion S1 selected_fields_sha256 mismatch"
        )

    handoff_ref = payload.get("source_post_promotion_handoff")
    if not isinstance(handoff_ref, dict) or set(handoff_ref) != {
        "path",
        "sha256",
        "handoff_sha256",
    }:
        raise ContractError("post-promotion S1 handoff reference is malformed")
    bound_handoff_path, handoff_record = _validate_operator_binding_reference(
        {"path": handoff_ref["path"], "sha256": handoff_ref["sha256"]},
        owner_path=path,
        where="post-promotion S1 source handoff",
    )
    if bound_handoff_path != post_promotion_handoff_path.resolve(strict=True):
        raise ContractError(
            "post-promotion S1 binds a different handoff than the v3 contract"
        )
    handoff_payload = _load_json(bound_handoff_path)
    if handoff_ref["handoff_sha256"] != handoff_payload.get("handoff_sha256"):
        raise ContractError("post-promotion S1 handoff semantic digest mismatch")

    legacy_path, legacy_record = _validate_operator_binding_reference(
        payload.get("source_legacy_s1"),
        owner_path=path,
        where="post-promotion S1 legacy source",
    )
    legacy_search = dict(final_search)
    legacy_search["c_scale"] = operator_binding.POST_PROMOTION_S1_OVERRIDE[
        "legacy_value"
    ]
    legacy_semantic = _validate_search_stage_evidence(
        _load_json(legacy_path),
        path=legacy_path,
        expected_stage="s1",
        final_search=legacy_search,
        final_evaluator=final_evaluator,
    )
    if (
        payload.get("source_legacy_s1_selected_fields_sha256")
        != legacy_semantic["selected_fields_sha256"]
    ):
        raise ContractError(
            "post-promotion S1 legacy selected-fields digest mismatch"
        )
    emitter_path, emitter_record = _validate_operator_binding_reference(
        payload.get("emitter"),
        owner_path=path,
        where="post-promotion S1 emitter",
    )
    if not emitter_path.as_posix().endswith("tools/search_operator_binding.py"):
        raise ContractError("post-promotion S1 binding has an untrusted emitter")
    checkpoint = payload.get("producer_checkpoint")
    _require_exact_keys(
        checkpoint,
        {"path", "sha256"},
        where="post-promotion S1 producer checkpoint",
    )
    checkpoint_path = _absolute_ref(str(checkpoint["path"]), base=path.parent)
    checkpoint_record = {
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
    }
    if checkpoint_record != checkpoint:
        raise ContractError("post-promotion S1 producer checkpoint hash drift")
    identity = handoff_payload.get("producer_identity", {})
    if (
        checkpoint != identity.get("checkpoint")
        or payload.get("producer_identity_sha256")
        != identity.get("agent_identity_sha256")
        or payload.get("producer_search_config_sha256")
        != _digest_value(identity.get("search_config"))
    ):
        raise ContractError("post-promotion S1 producer identity drift")

    records = [legacy_record, *legacy_semantic["source_artifacts"]]
    records.extend([handoff_record, emitter_record, checkpoint_record])
    return {
        "decision": payload["decision"],
        "evidence_class": operator_binding.ARTIFACT_KIND,
        "selected_fields": dict(selected),
        "selected_fields_sha256": payload["selected_fields_sha256"],
        "checkpoint": dict(checkpoint),
        "source_artifacts": records,
        "artifact_content_sha256": payload["artifact_content_sha256"],
        "binding_time_utc": payload["binding_time_utc"],
    }


def _validate_recovery_s1_operator_binding(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_stage: str,
    final_search: dict[str, Any],
    final_evaluator: dict[str, Any],
    recovery_receipt_path: Path | None,
    allow_recovery_s1: bool,
) -> dict[str, Any]:
    """Replay the recovery-only c_scale bridge without promotion laundering."""

    if expected_stage != "s1" or not allow_recovery_s1 or recovery_receipt_path is None:
        raise ContractError(
            "recovery S1 is accepted only by a v3 disaster-recovery contract"
        )
    try:
        replayed = operator_binding._replay_recovery_s1(path)  # noqa: SLF001
    except operator_binding.BindingError as error:
        raise ContractError(f"recovery S1 replay failed: {error}") from error
    if replayed != payload or payload.get("promotion_proof_recreated") is not False:
        raise ContractError("recovery S1 semantic replay/evidence-loss policy drift")
    selected = payload.get("selected_fields")
    if not isinstance(selected, dict) or set(selected) != operator_binding.S1_SELECTED_KEYS:
        raise ContractError("recovery S1 selected_fields shape mismatch")
    mismatches = {
        key: (selected[key], final_search.get(key))
        for key in selected
        if selected[key] != final_search.get(key)
    }
    if mismatches or payload.get("selected_fields_sha256") != _digest_value(selected):
        raise ContractError(f"recovery S1 fields mismatch final contract: {mismatches}")
    receipt_ref = payload.get("source_recovery_receipt")
    if not isinstance(receipt_ref, dict) or set(receipt_ref) != {
        "path",
        "sha256",
        "recovery_receipt_sha256",
        "recovery_lineage_id",
    }:
        raise ContractError("recovery S1 receipt reference is malformed")
    bound_receipt, receipt_record = _validate_operator_binding_reference(
        {"path": receipt_ref["path"], "sha256": receipt_ref["sha256"]},
        owner_path=path,
        where="recovery S1 source receipt",
    )
    if bound_receipt != recovery_receipt_path.resolve(strict=True):
        raise ContractError("recovery S1 binds a different recovery receipt")
    from tools import a1_v5_disaster_recovery as recovery

    try:
        verified = recovery.verify_committed_receipt(bound_receipt)
    except recovery.RecoveryError as error:
        raise ContractError(f"recovery S1 receipt replay failed: {error}") from error
    authority = verified["authority"]
    receipt = verified["receipt"]
    if (
        receipt_ref["recovery_receipt_sha256"]
        != receipt["recovery_receipt_sha256"]
        or receipt_ref["recovery_lineage_id"] != authority["recovery_lineage_id"]
    ):
        raise ContractError("recovery S1 receipt semantic identity drift")
    legacy_path, legacy_record = _validate_operator_binding_reference(
        payload.get("source_legacy_s1"),
        owner_path=path,
        where="recovery S1 legacy source",
    )
    legacy_search = dict(final_search)
    legacy_search["c_scale"] = operator_binding.RECOVERY_S1_OVERRIDE[
        "legacy_value"
    ]
    legacy_semantic = _validate_search_stage_evidence(
        _load_json(legacy_path),
        path=legacy_path,
        expected_stage="s1",
        final_search=legacy_search,
        final_evaluator=final_evaluator,
    )
    if (
        payload.get("source_legacy_s1_selected_fields_sha256")
        != legacy_semantic["selected_fields_sha256"]
    ):
        raise ContractError("recovery S1 legacy selected-fields digest mismatch")
    emitter_path, emitter_record = _validate_operator_binding_reference(
        payload.get("emitter"), owner_path=path, where="recovery S1 emitter"
    )
    if not emitter_path.as_posix().endswith("tools/search_operator_binding.py"):
        raise ContractError("recovery S1 binding has an untrusted emitter")
    checkpoint = payload.get("producer_checkpoint")
    _require_exact_keys(
        checkpoint, {"path", "sha256"}, where="recovery S1 producer checkpoint"
    )
    checkpoint_path = _absolute_ref(str(checkpoint["path"]), base=path.parent)
    checkpoint_record = {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)}
    identity = authority["producer_identity"]
    if (
        checkpoint_record != checkpoint
        or checkpoint != identity.get("checkpoint")
        or payload.get("producer_identity_sha256")
        != identity.get("agent_identity_sha256")
        or payload.get("producer_search_config_sha256")
        != _digest_value(identity.get("search_config"))
    ):
        raise ContractError("recovery S1 producer identity drift")
    records = [legacy_record, *legacy_semantic["source_artifacts"]]
    records.extend([receipt_record, emitter_record, checkpoint_record])
    return {
        "decision": payload["decision"],
        "evidence_class": operator_binding.ARTIFACT_KIND,
        "selected_fields": dict(selected),
        "selected_fields_sha256": payload["selected_fields_sha256"],
        "checkpoint": dict(checkpoint),
        "source_artifacts": records,
        "artifact_content_sha256": payload["artifact_content_sha256"],
        "binding_time_utc": payload["binding_time_utc"],
    }


def _validate_search_stage_evidence(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_stage: str,
    final_search: dict[str, Any],
    final_evaluator: dict[str, Any],
    post_promotion_handoff_path: Path | None = None,
    allow_post_promotion_s1: bool = False,
    recovery_receipt_path: Path | None = None,
    allow_recovery_s1: bool = False,
) -> dict[str, Any]:
    if payload.get("schema_version") == operator_binding.POST_PROMOTION_S1_SCHEMA:
        return _validate_post_promotion_s1_operator_binding(
            payload,
            path=path,
            expected_stage=expected_stage,
            final_search=final_search,
            final_evaluator=final_evaluator,
            post_promotion_handoff_path=post_promotion_handoff_path,
            allow_post_promotion_s1=allow_post_promotion_s1,
        )
    if payload.get("schema_version") == operator_binding.RECOVERY_S1_SCHEMA:
        return _validate_recovery_s1_operator_binding(
            payload,
            path=path,
            expected_stage=expected_stage,
            final_search=final_search,
            final_evaluator=final_evaluator,
            recovery_receipt_path=recovery_receipt_path,
            allow_recovery_s1=allow_recovery_s1,
        )
    if payload.get("schema_version") == "a1-s3-role-operator-hold-v1":
        # Lazy import avoids the pre-wave -> pool -> promotion -> pre-wave
        # cycle during module initialization.
        from tools import s3_role_operator_hold as s3_operator_hold

        if expected_stage != "s3":
            raise ContractError("same-checkpoint role-operator HOLD is S3-only")
        refs: dict[str, tuple[Path, dict[str, str]]] = {}
        for key in ("source_pooled", "source_s1", "source_s2", "emitter"):
            refs[key] = _validate_operator_binding_reference(
                payload.get(key),
                owner_path=path,
                where=f"S3 role-operator HOLD {key}",
            )
        emitter_path = refs["emitter"][0]
        if not emitter_path.as_posix().endswith("tools/s3_role_operator_hold.py"):
            raise ContractError("S3 role-operator HOLD has an untrusted emitter")
        try:
            replayed = s3_operator_hold.build_hold(
                refs["source_pooled"][0],
                source_s1=refs["source_s1"][0],
                source_s2=refs["source_s2"][0],
                decision_time_utc=payload.get("decision_time_utc"),
                emitter_path=emitter_path,
            )
        except s3_operator_hold.HoldError as error:
            raise ContractError(f"S3 role-operator HOLD replay failed: {error}") from error
        if payload != replayed:
            raise ContractError("S3 role-operator HOLD does not equal semantic replay")
        selected = dict(payload.get("selected_fields", {}))
        mismatches = {
            key: (value, final_search.get(key))
            for key, value in selected.items()
            if value != final_search.get(key)
        }
        if selected != s3_operator_hold.SELECTED_FIELDS or mismatches:
            raise ContractError(
                "S3 role-operator HOLD differs from final no-adaptive search: "
                f"{mismatches}"
            )
        source_records = [
            refs[key][1]
            for key in ("source_pooled", "emitter", "source_s1", "source_s2")
        ]
        return {
            "decision": "hold",
            "evidence_class": s3_operator_hold.ARTIFACT_KIND,
            "selected_fields": selected,
            "selected_fields_sha256": payload["selected_fields_sha256"],
            "checkpoint": dict(payload["checkpoint"]),
            "source_artifacts": source_records,
            "artifact_content_sha256": payload["artifact_content_sha256"],
        }
    if payload.get("schema_version") == operator_binding.SCHEMA:
        return _validate_operator_binding_evidence(
            payload,
            path=path,
            expected_stage=expected_stage,
            final_search=final_search,
            final_evaluator=final_evaluator,
            post_promotion_handoff_path=post_promotion_handoff_path,
            allow_post_promotion_s1=allow_post_promotion_s1,
            recovery_receipt_path=recovery_receipt_path,
            allow_recovery_s1=allow_recovery_s1,
        )
    if payload.get("schema_version") != SEARCH_STAGE_EVIDENCE_SCHEMA:
        raise ContractError(
            f"{expected_stage.upper()} evidence {path} schema must be "
            f"{SEARCH_STAGE_EVIDENCE_SCHEMA!r}"
        )
    if payload.get("stage") != expected_stage:
        raise ContractError(
            f"evidence {path} stage={payload.get('stage')!r}, expected {expected_stage!r}"
        )
    if payload.get("passed") is not True:
        raise ContractError(f"{expected_stage.upper()} evidence {path} passed != true")
    if payload.get("decision") not in {"adopt", "hold"}:
        raise ContractError(
            f"{expected_stage.upper()} evidence must decide adopt or hold"
        )
    source_records = _verify_declared_source_artifacts(
        payload.get("source_artifacts", []), envelope_path=path
    )
    manifest_candidates: list[Path] = []
    for record in source_records:
        candidate = Path(record["path"])
        if candidate.suffix.lower() != ".json":
            continue
        try:
            candidate_payload = _load_json(candidate)
        except ContractError:
            continue
        if (
            candidate_payload.get("schema_version")
            == search_adjudicator.MANIFEST_SCHEMA
            and candidate_payload.get("stage") == expected_stage
        ):
            manifest_candidates.append(candidate)
    if len(manifest_candidates) != 1:
        raise ContractError(
            f"{expected_stage.upper()} evidence must bind exactly one replayable "
            f"adjudication manifest, found {len(manifest_candidates)}"
        )
    try:
        replayed = (
            operator_binding._replay_s1(path)
            if expected_stage == "s1"
            else search_adjudicator.adjudicate(manifest_candidates[0])
        )
    except operator_binding.BindingError as error:
        raise ContractError(
            f"{expected_stage.upper()} evidence does not equal the replayed adjudication: "
            f"{error}"
        ) from error
    except search_adjudicator.AdjudicationError as error:
        raise ContractError(
            f"{expected_stage.upper()} evidence failed semantic replay: {error}"
        ) from error
    if payload != replayed:
        raise ContractError(
            f"{expected_stage.upper()} evidence does not equal the replayed adjudication"
        )
    selected = dict(payload.get("selected_fields", {}))
    expected_keys = {
        "s1": {
            "c_scale",
            "symmetry_averaged_eval",
            "symmetry_averaged_eval_threshold",
            "rescale_noise_floor_c",
            "sigma_eval",
        },
        "s2": {"n_full", "n_fast", "p_full"},
        "s3": {"n_full_wide", "n_full_wide_threshold", "wide_roots_always_full"},
    }[expected_stage]
    if set(selected) != expected_keys:
        raise ContractError(
            f"{expected_stage.upper()} selected_fields must be exactly {sorted(expected_keys)}"
        )
    mismatches = {
        key: (selected[key], final_search[key])
        for key in expected_keys
        if selected[key] != final_search[key]
    }
    if mismatches:
        raise ContractError(
            f"{expected_stage.upper()} selected search fields mismatch final contract: {mismatches}"
        )
    declared_selected_hash = payload.get("selected_fields_sha256")
    if declared_selected_hash != _digest_value(selected):
        raise ContractError(f"{expected_stage.upper()} selected_fields_sha256 mismatch")
    records = source_records
    adjudicator = payload.get("adjudicator")
    adjudicator_records = _verify_declared_source_artifacts(
        [adjudicator] if isinstance(adjudicator, dict) else [], envelope_path=path
    )
    if (
        not Path(adjudicator_records[0]["path"])
        .as_posix()
        .endswith("tools/search_teacher_adjudicator.py")
    ):
        raise ContractError(
            f"{expected_stage.upper()} evidence was not emitted by search_teacher_adjudicator.py"
        )
    records.extend(adjudicator_records)
    manifest_payload = _load_json(manifest_candidates[0])
    manifest_checkpoint = dict(manifest_payload.get("checkpoint", {}))
    _require_exact_keys(
        manifest_checkpoint,
        {"path", "sha256"},
        where=f"{expected_stage.upper()} manifest checkpoint",
    )
    manifest_checkpoint_path = _absolute_ref(
        str(manifest_checkpoint["path"]), base=manifest_candidates[0].parent
    )
    actual_checkpoint_sha = (
        _sha256(manifest_checkpoint_path)
        if manifest_checkpoint_path.is_file()
        else "<missing>"
    )
    if actual_checkpoint_sha != manifest_checkpoint["sha256"]:
        raise ContractError(
            f"{expected_stage.upper()} manifest checkpoint hash drift"
        )
    semantic: dict[str, Any] = {
        "decision": payload["decision"],
        "selected_fields": selected,
        "selected_fields_sha256": declared_selected_hash,
        "manifest": next(
            record
            for record in source_records
            if Path(record["path"]) == manifest_candidates[0]
        ),
        "checkpoint": {
            "path": str(manifest_checkpoint_path),
            "sha256": actual_checkpoint_sha,
        },
        "source_artifacts": records,
    }
    if expected_stage == "s3":
        if payload.get("final_search_operator_sha256") != _digest_value(final_search):
            raise ContractError(
                "S3 final search-operator hash does not match the contract"
            )
        if payload.get("teacher_evaluator_sha256") != _digest_value(final_evaluator):
            raise ContractError("S3 teacher-evaluator hash does not match the contract")
        if _digest_value(payload.get("final_search_operator")) != _digest_value(
            final_search
        ):
            raise ContractError(
                "S3 final search-operator fields do not match the contract"
            )
        if _digest_value(payload.get("teacher_evaluator")) != _digest_value(
            final_evaluator
        ):
            raise ContractError("S3 teacher-evaluator fields do not match the contract")
        semantic["final_search_operator_sha256"] = payload[
            "final_search_operator_sha256"
        ]
        semantic["teacher_evaluator_sha256"] = payload["teacher_evaluator_sha256"]
    return semantic


def _validate_categories(
    categories: list[dict[str, Any]], checkpoint_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in checkpoint_records}
    if {str(category.get("name")) for category in categories} != set(EXPECTED_GAMES):
        raise ContractError(
            f"source categories must be exactly {sorted(EXPECTED_GAMES)}"
        )
    normalized: list[dict[str, Any]] = []
    for category in categories:
        if set(category) != {"name", "mode", "checkpoint_ids"}:
            raise ContractError(
                "each source category must have exactly name, mode, checkpoint_ids"
            )
        name = str(category["name"])
        mode = str(category["mode"])
        ids = list(map(str, category["checkpoint_ids"]))
        if name == "current_producer":
            if mode != "self" or ids:
                raise ContractError(
                    "current_producer must use mode=self and no opponent checkpoint"
                )
        else:
            if mode != "checkpoint_list" or not ids:
                raise ContractError(f"{name} must use a non-empty checkpoint_list")
            for artifact_id in ids:
                if artifact_id not in by_id:
                    raise ContractError(
                        f"{name} references unknown checkpoint id {artifact_id!r}"
                    )
                if by_id[artifact_id]["role"] == "producer":
                    raise ContractError(
                        f"{name} may not disguise the producer as an opponent"
                    )
                required_role = (
                    "history" if name == "recent_history" else "hard_negative"
                )
                if by_id[artifact_id]["role"] != required_role:
                    raise ContractError(
                        f"{name} checkpoint {artifact_id!r} has role={by_id[artifact_id]['role']!r}; "
                        f"expected {required_role!r}"
                    )
        normalized.append({"name": name, "mode": mode, "checkpoint_ids": ids})
    return sorted(normalized, key=lambda item: list(EXPECTED_GAMES).index(item["name"]))


def _verify_artifact_records(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        path = Path(record["path"])
        actual = _sha256(path) if path.is_file() else "<missing>"
        if actual != record["sha256"]:
            raise ContractError(
                f"artifact drift for {record.get('id', record.get('kind'))}: "
                f"expected {record['sha256']}, got {actual} at {path}"
            )


def _verify_archived_code_provenance_records(
    records: Iterable[dict[str, Any]],
) -> None:
    """Validate descriptors whose bytes are authenticated by one issued lock.

    The sole markerless v2 lock is identified by its raw file hash before this
    helper is reachable. Its generator, learner, and transitive-runtime records
    name a retired checkout, so replay cannot require those mutable paths to
    exist today. The raw lock bytes still authenticate every path and digest;
    this helper merely type-checks the sealed descriptors without consulting
    the current filesystem. Guard, evidence, checkpoint, seed-ledger, job, and
    science records continue through their ordinary live validators.
    """

    for record in records:
        if not isinstance(record, dict):
            raise ContractError("archived code provenance record is not an object")
        path = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise ContractError("archived code provenance descriptor is malformed")


def _issued_arm_lock_fingerprint(path: Path) -> dict[str, str]:
    value = _load_json(path)
    expected_digest = str(value.get("contract_sha256", ""))
    unhashed = dict(value)
    unhashed.pop("contract_sha256", None)
    if expected_digest != _digest_value(unhashed):
        raise ContractError(f"superseded arm lock semantic digest drift: {path}")
    arm_id = str(value.get("game_contract", {}).get("arm_id", ""))
    expected = GENERATION_CAMPAIGN_R1_LOCKS.get(arm_id)
    actual = {
        "contract_id": str(value.get("contract_id", "")),
        "contract_sha256": expected_digest,
        "file_sha256": _sha256(path),
    }
    if expected is None or actual != expected:
        raise ContractError(f"superseded {arm_id or 'unknown'} lock is not issued r1 bytes")
    return {"arm_id": arm_id, **actual}


def _generation_revision_payload(
    source: dict[str, Any],
    *,
    contract_id: str,
    output_root: Path,
    superseded_locks: list[dict[str, str]],
) -> dict[str, Any]:
    if not re.fullmatch(
        r"a1-dual-arm-n256-n128-56gpu-[0-9]{8}-r(?:[2-9]|[1-9][0-9]+)",
        contract_id,
    ):
        raise ContractError("generation revision contract id must be a fresh r2+ id")
    output_root = output_root.expanduser()
    if not output_root.is_absolute():
        raise ContractError("generation revision output root must be absolute")
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    expected_locks = [
        {"arm_id": arm_id, **GENERATION_CAMPAIGN_R1_LOCKS[arm_id]}
        for arm_id in ("n128", "n256")
    ]
    if (
        not isinstance(superseded_locks, list)
        or len(superseded_locks) != 2
        or any(
            not isinstance(row, dict)
            or set(row)
            != {"arm_id", "contract_id", "contract_sha256", "file_sha256"}
            for row in superseded_locks
        )
        or sorted(superseded_locks, key=lambda row: str(row["arm_id"]))
        != expected_locks
    ):
        raise ContractError("generation revision does not bind both issued r1 arm locks")

    value = json.loads(json.dumps(source))
    value["schema_version"] = GENERATION_CAMPAIGN_REVISION_SCHEMA
    value["contract_id"] = contract_id
    value["implementation_commit"] = GENERATION_CAMPAIGN_REVISION_IMPLEMENTATION_COMMIT
    value["supersedes"] = {
        "campaign_contract_id": GENERATION_CAMPAIGN_CONTRACT_ID,
        "campaign_contract_sha256": GENERATION_CAMPAIGN_CONTRACT_SHA256,
        "reason": "original post-promotion lineage artifacts are unavailable; old locks remain immutable",
        "arm_locks": expected_locks,
    }
    recipe = dict(value["common_recipe"])
    recipe["native_mcts_hot_loop"] = True
    recipe["rust_featurize"] = True
    value["common_recipe"] = recipe
    arms = {str(arm["id"]): arm for arm in value["arms"]}
    cursor = GENERATION_CAMPAIGN_R1_NEXT_SEED_FLOOR
    for arm_id in ("n256", "n128"):
        arm = arms[arm_id]
        arm["seed_start"] = cursor
        cursor += int(arm["gpu_count"]) * int(arm["seed_block_size"])
        arm["seed_end"] = cursor
        arm["output_root"] = str(output_root / arm_id)
    value["fleet"]["next_campaign_seed_floor"] = cursor
    value["provenance"] = _generation_revision_provenance(
        GENERATION_CAMPAIGN_REVISION_IMPLEMENTATION_COMMIT
    )
    value.pop("contract_sha256", None)
    value["contract_sha256"] = _digest_value(value)
    return value


def build_generation_campaign_revision(
    source_path: Path,
    *,
    superseded_lock_paths: Sequence[Path],
    contract_id: str,
    output_root: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Create a fresh pending blueprint; never restore, claim, render, or launch."""

    source = validate_generation_campaign(source_path)
    if source.get("schema_version") != GENERATION_CAMPAIGN_SCHEMA:
        raise ContractError("generation revision source must be the canonical issued r1 campaign")
    fingerprints = [_issued_arm_lock_fingerprint(path) for path in superseded_lock_paths]
    value = _generation_revision_payload(
        source,
        contract_id=contract_id,
        output_root=output_root,
        superseded_locks=fingerprints,
    )
    _create_readonly(out_path.absolute(), value)
    return value


def _validate_generation_campaign_revision(
    path: Path, value: dict[str, Any], *, require_ready: bool
) -> dict[str, Any]:
    _require_exact_keys(
        value,
        {
            "schema_version", "contract_id", "status", "promotion_handoff",
            "checkpoints", "source_categories", "common_recipe", "arms", "fleet",
            "provenance", "execution_policy", "implementation_commit", "supersedes",
            "contract_sha256",
        },
        where="generation campaign revision",
    )
    declared = str(value.get("contract_sha256", ""))
    unhashed = dict(value)
    unhashed.pop("contract_sha256", None)
    if declared != _digest_value(unhashed):
        raise ContractError("generation campaign revision semantic digest mismatch")
    if (
        path.resolve() == GENERATION_CAMPAIGN_R2_CONTRACT_PATH.resolve()
        and (
            value.get("contract_id") != GENERATION_CAMPAIGN_R2_CONTRACT_ID
            or declared != GENERATION_CAMPAIGN_R2_CONTRACT_SHA256
        )
    ):
        raise ContractError("canonical r2 generation campaign identity drift")
    if value.get("implementation_commit") != GENERATION_CAMPAIGN_REVISION_IMPLEMENTATION_COMMIT:
        raise ContractError("generation campaign revision implementation commit drift")
    arms = {str(arm.get("id")): dict(arm) for arm in value.get("arms", [])}
    if set(arms) != {"n128", "n256"}:
        raise ContractError("generation campaign revision arm set drift")
    roots = {arm_id: Path(str(arm["output_root"])) for arm_id, arm in arms.items()}
    if any(root.name != arm_id for arm_id, root in roots.items()):
        raise ContractError("generation campaign revision output roots must end in arm ids")
    if roots["n128"].parent != roots["n256"].parent:
        raise ContractError("generation campaign revision output roots have different parents")
    superseded = dict(value.get("supersedes", {}))
    source = validate_generation_campaign(GENERATION_CAMPAIGN_CONTRACT_PATH)
    expected = _generation_revision_payload(
        source,
        contract_id=str(value.get("contract_id", "")),
        output_root=roots["n128"].parent,
        superseded_locks=list(superseded.get("arm_locks", [])),
    )
    if value != expected:
        raise ContractError("generation campaign revision differs from deterministic rebuild")
    if require_ready:
        raise ContractError(
            "generation campaign revision is not launchable: a new committed promotion "
            "handoff and newly sealed placement are required"
        )
    return value


def _require_fresh_revision_handoff(record: Mapping[str, Any]) -> None:
    if (
        record.get("transaction_id") == GENERATION_CAMPAIGN_R1_TRANSACTION_ID
        or record.get("handoff_sha256") == GENERATION_CAMPAIGN_R1_HANDOFF_SHA256
    ):
        raise ContractError(
            "generation revision requires a newly revalidated promotion handoff; "
            "the issued r1 lineage cannot authorize r2"
        )


def _current_repo_commit() -> str:
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError("cannot resolve campaign-rebase Git identity") from error
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError("campaign-rebase Git identity is malformed")
    return commit


def _refresh_campaign_provenance(source: Mapping[str, Any]) -> dict[str, Any]:
    refreshed: dict[str, Any] = {}
    for group in ("arm_guards", "generator_code"):
        rows = []
        for raw in source[group]:
            item = dict(raw)
            path = REPO_ROOT / str(item["path"])
            item["sha256"] = _sha256(path)
            rows.append(item)
        refreshed[group] = rows
    for group in ("executor", "harvest", "fleet_manifest"):
        item = dict(source[group])
        item["sha256"] = _sha256(REPO_ROOT / str(item["path"]))
        refreshed[group] = item
    return refreshed


def _post_promotion_campaign_payload(
    source_path: Path,
    source: dict[str, Any],
    handoff_path: Path,
    handoff: dict[str, Any],
    *,
    contract_id: str,
    output_root: Path,
) -> dict[str, Any]:
    if source.get("schema_version") != GENERATION_CAMPAIGN_REVISION_SCHEMA:
        raise ContractError("post-promotion rebase requires the latest sealed revision")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", contract_id):
        raise ContractError("post-promotion campaign id is malformed")
    output_root = Path(os.path.abspath(os.fspath(output_root.expanduser())))
    if not output_root.is_absolute():
        raise ContractError("post-promotion output root must be absolute")
    producer = handoff.get("producer_identity", {}).get("checkpoint")
    if not isinstance(producer, dict) or set(producer) != {"path", "sha256"}:
        raise ContractError("post-promotion handoff has no producer checkpoint")
    producer_path = Path(str(producer["path"]))
    if not producer_path.is_file() or _sha256(producer_path) != producer["sha256"]:
        raise ContractError("post-promotion producer bytes drifted")
    old_producers = [row for row in source["checkpoints"] if row["role"] == "producer"]
    if len(old_producers) != 1 or old_producers[0]["sha256"] == producer["sha256"]:
        raise ContractError("post-promotion rebase requires a distinct prior producer")
    prior = {**old_producers[0], "id": "prior_generator", "role": "history"}
    checkpoints = [
        {"id": "a1_producer", "role": "producer", **producer},
        prior,
        *[row for row in source["checkpoints"] if row["role"] != "producer"],
    ]
    categories = json.loads(json.dumps(source["source_categories"]))
    recent = next(row for row in categories if row["name"] == "recent_history")
    recent["checkpoint_ids"] = [
        "prior_generator",
        *[item for item in recent["checkpoint_ids"] if item != "prior_generator"],
    ]
    value = json.loads(json.dumps(source))
    producer_search = handoff.get("producer_identity", {}).get("search_config")
    if not isinstance(producer_search, dict) or not isinstance(
        producer_search.get("c_scale"), (int, float)
    ):
        raise ContractError("post-promotion producer has no deployed c_scale identity")
    common_recipe = dict(value["common_recipe"])
    common_recipe["c_scale"] = float(producer_search["c_scale"])
    category_c_scale = {
        category: float(producer_search["c_scale"])
        for category in common_recipe["category_c_scale"]
    }
    common_recipe["category_c_scale"] = category_c_scale
    value.update(
        {
            "schema_version": POST_PROMOTION_CAMPAIGN_SCHEMA,
            "contract_id": contract_id,
            "status": POST_PROMOTION_CAMPAIGN_STATUS,
            "promotion_handoff": {
                "mode": "post_promotion",
                "path": str(handoff_path),
                "sha256": _sha256(handoff_path),
                "handoff_sha256": handoff["handoff_sha256"],
                "transaction_id": handoff["promotion_receipt"]["transaction_id"],
                "expected_schema": promotion_handoff.HANDOFF_SCHEMA,
                "expected_checkpoint_sha256": producer["sha256"],
            },
            "checkpoints": checkpoints,
            "source_categories": categories,
            "common_recipe": common_recipe,
            "implementation_commit": _current_repo_commit(),
            "provenance": _refresh_campaign_provenance(source["provenance"]),
        }
    )
    cursor = int(source["fleet"]["next_campaign_seed_floor"])
    for arm in sorted(value["arms"], key=lambda row: 0 if row["id"] == "n256" else 1):
        arm["seed_start"] = cursor
        cursor += int(arm["gpu_count"]) * int(arm["seed_block_size"])
        arm["seed_end"] = cursor
        arm["output_root"] = str(output_root / str(arm["id"]))
    value["fleet"]["next_campaign_seed_floor"] = cursor
    value["rebase"] = {
        "source_campaign": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "contract_sha256": source["contract_sha256"],
        },
        "previous_producer": old_producers[0],
        "promoted_producer": producer,
        "builder": {
            # Bind source identity, not the ephemeral checkout that happened
            # to build the campaign.  An absolute ``__file__`` made otherwise
            # identical clean worktrees rebuild different payloads.
            "path": "tools/a1_pre_wave_contract.py",
            "sha256": _sha256(REPO_ROOT / "tools/a1_pre_wave_contract.py"),
        },
    }
    value.pop("supersedes", None)
    value.pop("contract_sha256", None)
    value["contract_sha256"] = _digest_value(value)
    return value


def build_post_promotion_generation_campaign(
    source_path: Path,
    *,
    handoff_path: Path,
    contract_id: str,
    output_root: Path,
    out_path: Path,
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve(strict=True)
    source = validate_generation_campaign(source_path)
    handoff_path = handoff_path.expanduser().resolve(strict=True)
    handoff = _load_json(handoff_path)
    try:
        replayed = promotion_handoff.build_handoff(
            Path(str(handoff.get("promotion_receipt", {}).get("path", "")))
        )
    except promotion_handoff.HandoffError as error:
        raise ContractError(f"post-promotion handoff replay failed: {error}") from error
    if handoff != replayed:
        raise ContractError("post-promotion handoff differs from committed live lineage")
    value = _post_promotion_campaign_payload(
        source_path, source, handoff_path, handoff,
        contract_id=contract_id, output_root=output_root,
    )
    _create_readonly(out_path.absolute(), value)
    return value


def _validate_post_promotion_campaign(
    path: Path, value: dict[str, Any], *, require_ready: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version", "contract_id", "status", "promotion_handoff",
        "checkpoints", "source_categories", "common_recipe", "arms", "fleet",
        "provenance", "execution_policy", "implementation_commit", "rebase",
        "contract_sha256",
    }
    _require_exact_keys(value, expected_keys, where="post-promotion generation campaign")
    unhashed = dict(value)
    declared = unhashed.pop("contract_sha256")
    if declared != _digest_value(unhashed):
        raise ContractError("post-promotion campaign semantic digest mismatch")
    if value["status"] != POST_PROMOTION_CAMPAIGN_STATUS:
        raise ContractError("post-promotion campaign status drift")
    source_ref = value["rebase"]["source_campaign"]
    source_path = Path(str(source_ref["path"]))
    if _sha256(source_path) != source_ref["sha256"]:
        raise ContractError("post-promotion source campaign bytes drifted")
    source = validate_generation_campaign(source_path)
    if source["contract_sha256"] != source_ref["contract_sha256"]:
        raise ContractError("post-promotion source campaign identity drifted")
    handoff_path = Path(str(value["promotion_handoff"]["path"]))
    if _sha256(handoff_path) != value["promotion_handoff"]["sha256"]:
        raise ContractError("post-promotion handoff bytes drifted")
    handoff = _load_json(handoff_path)
    try:
        replayed = promotion_handoff.build_handoff(
            Path(str(handoff.get("promotion_receipt", {}).get("path", "")))
        )
    except promotion_handoff.HandoffError as error:
        raise ContractError(f"post-promotion handoff replay failed: {error}") from error
    if handoff != replayed:
        raise ContractError("post-promotion handoff live lineage drifted")
    roots = {row["id"]: Path(row["output_root"]).parent for row in value["arms"]}
    if len(set(roots.values())) != 1:
        raise ContractError("post-promotion campaign output roots diverge")
    expected = _post_promotion_campaign_payload(
        source_path, source, handoff_path, handoff,
        contract_id=value["contract_id"], output_root=next(iter(roots.values())),
    )
    if expected != value:
        raise ContractError("post-promotion campaign differs from deterministic rebuild")
    if require_ready:
        raise ContractError(
            "post-promotion campaign is not launchable until exact placement and arm "
            "locks are sealed"
        )
    return value


def validate_generation_campaign(
    path: Path,
    *,
    require_ready: bool = False,
    _allow_historical_lock_source: bool = False,
) -> dict[str, Any]:
    """Validate the canonical post-A1 dual-arm generation blueprint.

    This blueprint deliberately cannot be rendered or executed.  It freezes the
    science, seeds, output roots, and implementation bytes while the committed
    promotion handoff and exact 56-GPU host placement remain unavailable.
    """

    value = _load_json(path)
    if value.get("schema_version") == POST_PROMOTION_CAMPAIGN_SCHEMA:
        return _validate_post_promotion_campaign(
            path, value, require_ready=require_ready
        )
    if value.get("schema_version") == GENERATION_CAMPAIGN_REVISION_SCHEMA:
        return _validate_generation_campaign_revision(
            path, value, require_ready=require_ready
        )
    historical_lock_source = bool(
        _allow_historical_lock_source
        and value.get("contract_sha256") == HISTORICAL_DB1_CAMPAIGN_SHA256
        and path.is_file()
        and _sha256(path) == HISTORICAL_DB1_CAMPAIGN_FILE_SHA256
    )
    _require_exact_keys(
        value,
        {
            "schema_version",
            "contract_id",
            "status",
            "promotion_handoff",
            "checkpoints",
            "source_categories",
            "common_recipe",
            "arms",
            "fleet",
            "provenance",
            "execution_policy",
            "contract_sha256",
        },
        where="generation campaign",
    )
    if value["schema_version"] != GENERATION_CAMPAIGN_SCHEMA:
        raise ContractError("generation campaign schema drift")
    unhashed = dict(value)
    declared = unhashed.pop("contract_sha256")
    if declared != _digest_value(unhashed):
        raise ContractError("generation campaign semantic digest mismatch")
    # The already-issued campaign and the one explicit historical DB1 lock
    # source verify from their one canonical implementation commit (plus the
    # two digest-keyed guard snapshots). No copied/recomputed/new contract gets
    # that privilege: it must bind the current source bytes.
    canonical_archived_provenance = bool(
        path.resolve() == GENERATION_CAMPAIGN_CONTRACT_PATH.resolve()
        and value.get("contract_id") == GENERATION_CAMPAIGN_CONTRACT_ID
        and declared == GENERATION_CAMPAIGN_CONTRACT_SHA256
    )
    allow_archived_provenance = (
        historical_lock_source or canonical_archived_provenance
    )
    historical_implementation_commit = _campaign_historical_implementation_commit(
        path,
        value,
        historical_lock_source=historical_lock_source,
    )
    if allow_archived_provenance != (historical_implementation_commit is not None):
        raise ContractError("generation campaign historical source authority drift")
    if value["status"] != GENERATION_CAMPAIGN_PENDING:
        raise ContractError("generation campaign must remain explicitly pending")
    handoff = dict(value["promotion_handoff"])
    if handoff != {
        "mode": "required_post_promotion",
        "path": None,
        "expected_schema": promotion_handoff.HANDOFF_SCHEMA,
        "expected_checkpoint_sha256": GENERATION_CAMPAIGN_CHECKPOINT_SHA256,
    }:
        raise ContractError("generation campaign promotion handoff gate drift")
    checkpoints = list(value["checkpoints"])
    expected_checkpoints = [
        {
            "id": "a1_producer",
            "role": "producer",
            "path": "/home/ubuntu/catan-zero-production/runs/learner/"
            "a1-infoset-n128-20260710-r2/candidate.pt",
            "sha256": GENERATION_CAMPAIGN_CHECKPOINT_SHA256,
        },
        {
            "id": "gen3_history",
            "role": "history",
            "path": "/home/ubuntu/catan-zero/runs/bc/gen3_20260706/checkpoint.pt",
            "sha256": "sha256:89aa133d629e747021bc725f2ad63e0563f3b76e71f0dd563f056c6de8f77ebb",
        },
        {
            "id": "gen4_hard_negative",
            "role": "hard_negative",
            "path": "/home/ubuntu/catan-zero/runs/bc/gen4_20260708/checkpoint.pt",
            "sha256": "sha256:b0f939464c138d6d0dca5586585d7e71aacb7ed86183cccbc2131d95750fe1c5",
        },
    ]
    if checkpoints != expected_checkpoints:
        raise ContractError("generation campaign checkpoint identities drift")
    if list(value["source_categories"]) != [
        {"name": "current_producer", "mode": "self", "checkpoint_ids": []},
        {
            "name": "recent_history",
            "mode": "checkpoint_list",
            "checkpoint_ids": ["gen3_history"],
        },
        {
            "name": "hard_negative",
            "mode": "checkpoint_list",
            "checkpoint_ids": ["gen4_hard_negative"],
        },
    ]:
        raise ContractError("generation campaign source-category bindings drift")

    recipe = dict(value["common_recipe"])
    expected_recipe = {
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "public_observation": True,
        "information_set_search": True,
        "belief_chance_spectra": False,
        "determinization_particles": 4,
        "determinization_min_simulations": 32,
        "c_visit": 50.0,
        "c_scale": 0.1,
        "category_c_scale": {
            "current_producer": 0.1,
            "recent_history": 0.03,
            "hard_negative": 0.03,
        },
        "prior_temperature": 1.0,
        "n_fast": 16,
        "p_full": 0.25,
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "wide_roots_always_full": False,
        "raw_policy_above_width": None,
        "wide_candidates_threshold": 24,
        "max_depth": 80,
        "max_decisions": 600,
        "correct_rust_chance_spectra": True,
        "lazy_interior_chance": True,
        "exact_budget_sh": False,
        "exact_budget_sh_min_n": 0,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
        "value_scale": 1.0,
        "value_squash": "tanh",
        "value_readout": "scalar",
        "context_fill": 0.0,
        "cache_size": 0,
        "rust_featurize": False,
        "emit_uncertainty": False,
        "obs_width": 806,
        "temperature_decisions": 90,
        "temperature_high": 1.0,
        "temperature_low": 0.0,
        "late_temperature_decisions": None,
        "late_temperature": 0.0,
        "shard_size": 512,
        "format": "npz",
        "device": "cuda",
        "eval_server": False,
        "workers_per_gpu": 16,
        "mps_client_environment": {
            "CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE_DIRECTORY,
            "CUDA_MPS_LOG_DIRECTORY": MPS_LOG_DIRECTORY,
        },
    }
    if recipe != expected_recipe:
        raise ContractError("generation campaign common recipe drift")

    arms = list(value["arms"])
    if len(arms) != 2 or {arm.get("id") for arm in arms} != {"n256", "n128"}:
        raise ContractError("generation campaign requires exact n256/n128 arms")
    expected_arms = {"n256": (256, 2_000), "n128": (128, 5_000)}
    intervals: list[tuple[int, int]] = []
    roots: list[PurePath] = []
    lane_ids: set[str] = set()
    for arm in arms:
        _require_exact_keys(
            arm,
            {
                "id",
                "n_full",
                "gpu_count",
                "games_per_gpu",
                "selected_per_gpu",
                "max_attempts_per_gpu",
                "total_games",
                "seed_start",
                "seed_end",
                "seed_block_size",
                "output_root",
                "logical_lanes",
            },
            where="generation campaign arm",
        )
        arm_id = str(arm["id"])
        expected_n, expected_games = expected_arms[arm_id]
        expected_selected = (
            {"current_producer": 1_600, "recent_history": 300, "hard_negative": 100}
            if arm_id == "n256"
            else {"current_producer": 4_000, "recent_history": 750, "hard_negative": 250}
        )
        expected_attempts = (
            {"current_producer": 1_640, "recent_history": 310, "hard_negative": 104}
            if arm_id == "n256"
            else {"current_producer": 4_080, "recent_history": 765, "hard_negative": 255}
        )
        if (
            int(arm["n_full"]) != expected_n
            or int(arm["gpu_count"]) != 28
            or int(arm["games_per_gpu"]) != expected_games
            or dict(arm["selected_per_gpu"]) != expected_selected
            or dict(arm["max_attempts_per_gpu"]) != expected_attempts
            or sum(expected_attempts.values()) > int(arm["seed_block_size"])
            or int(arm["seed_block_size"]) != 8_192
            or int(arm["total_games"]) != 28 * expected_games
        ):
            raise ContractError(f"generation campaign {arm_id} budget drift")
        start, end = int(arm["seed_start"]), int(arm["seed_end"])
        if start <= GENERATION_CAMPAIGN_SEED_FLOOR or end != start + 28 * int(
            arm["seed_block_size"]
        ):
            raise ContractError(f"generation campaign {arm_id} seed range drift")
        intervals.append((start, end))
        root = PurePath(str(arm["output_root"]))
        if not root.is_absolute():
            raise ContractError(f"generation campaign {arm_id} output root is relative")
        roots.append(root)
        lanes = list(map(str, arm["logical_lanes"]))
        if len(lanes) != 28 or len(set(lanes)) != 28 or lane_ids.intersection(lanes):
            raise ContractError("generation campaign logical lanes are not disjoint 28-lane sets")
        lane_ids.update(lanes)
    try:
        assert_disjoint_seed_blocks(
            [
                (str(arms[index]["id"]), start, end - start)
                for index, (start, end) in enumerate(intervals)
            ]
        )
    except ValueError as error:
        raise ContractError(f"generation campaign seed ranges overlap: {error}") from error
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        raise ContractError("generation campaign output roots overlap")

    fleet = dict(value["fleet"])
    expected_fleet = {
        "total_gpus": 56,
        "placement_mode": "canonical_assignments_pending_seal",
        "seed_ledger": "/home/ubuntu/catan-zero-production/SEED_LEDGER.md",
        "next_campaign_seed_floor": 300_000_626_944,
        "placement_assignments": {
            "path": "configs/operations/a1-dual-arm-56gpu-20260710/placement.assignments.json",
            "sha256": "sha256:b977391d837888d5aea879bfaa0548feec19f70641253c6ba5bbefd22ee5c523",
        },
    }
    if fleet != expected_fleet:
        raise ContractError("generation campaign fleet gate drift")
    assignment_path = REPO_ROOT / fleet["placement_assignments"]["path"]
    if _sha256(assignment_path) != fleet["placement_assignments"]["sha256"]:
        raise ContractError("generation campaign placement assignments drift")
    assignment_payload = _load_json(assignment_path)
    _validate_campaign_assignments(assignment_payload.get("assignments"), value)
    policy = dict(value["execution_policy"])
    if policy != {
        "launch_authorized": False,
        "seal_authorized": False,
        "required_executor": "tools/fleet/a1_production_executor.py",
        "required_harvest": "tools/fleet/a1_harvest_transaction.py",
        "materialization": "seal_with_a1_pre_wave_contract_after_handoff_and_placement",
    }:
        raise ContractError("generation campaign execution policy drift")

    provenance = dict(value["provenance"])
    _require_exact_keys(
        provenance,
        {"arm_guards", "generator_code", "executor", "harvest", "fleet_manifest"},
        where="generation campaign provenance",
    )
    records = [
        *map(dict, provenance["arm_guards"]),
        *map(dict, provenance["generator_code"]),
        dict(provenance["executor"]),
        dict(provenance["harvest"]),
        dict(provenance["fleet_manifest"]),
    ]
    expected_suffixes = ISSUED_CAMPAIGN_GENERATOR_CODE_SUFFIXES | {
        "configs/guards/a1_generation_n256.json",
        "configs/guards/a1_generation_n128.json",
        "configs/guards/a1_generation_n256_legacy.json",
        "configs/guards/a1_generation_n128_legacy.json",
        "tools/launcher_guards.py",
        "tools/prelaunch_guard.py",
        "tools/fleet/a1_production_executor.py",
        "tools/fleet/a1_lane_supervisor.py",
        "tools/fleet/a1_harvest_transaction.py",
        "configs/gpu_fleet_56.json",
    }
    if historical_lock_source:
        expected_suffixes.remove("tools/fleet/a1_lane_supervisor.py")
    if {str(record.get("path")) for record in records} != expected_suffixes:
        raise ContractError("generation campaign provenance file set drift")
    guards = {record["path"] for record in provenance["arm_guards"]}
    if guards != {
        "configs/guards/a1_generation_n256.json",
        "configs/guards/a1_generation_n128.json",
        "configs/guards/a1_generation_n256_legacy.json",
        "configs/guards/a1_generation_n128_legacy.json",
    }:
        raise ContractError("generation campaign arm guard bindings drift")
    historical_provenance: dict[str, bytes] = {}
    for record in records:
        _require_exact_keys(record, {"path", "sha256"}, where="generation campaign file")
        relative_path = str(record["path"])
        if historical_implementation_commit is not None:
            historical_provenance[relative_path] = (
                _historical_campaign_provenance_bytes(
                    historical_implementation_commit,
                    relative_path,
                    str(record["sha256"]),
                )
            )
            continue
        source = REPO_ROOT / relative_path
        if not source.is_file() or _sha256(source) != record["sha256"]:
            raise ContractError(f"generation campaign immutable file drift: {source}")
    for record in provenance["arm_guards"]:
        name = Path(str(record["path"])).stem
        match = re.fullmatch(r"a1_generation_(n128|n256)(_legacy)?", name)
        if match is None:
            raise ContractError(f"generation campaign has unknown arm guard {name}")
        relative_path = str(record["path"])
        live_guard = REPO_ROOT / relative_path
        if historical_implementation_commit is not None:
            try:
                payload = json.loads(historical_provenance[relative_path])
            except (KeyError, UnicodeError, json.JSONDecodeError) as error:
                raise ContractError(
                    "generation campaign immutable guard bytes are invalid: "
                    f"{relative_path}"
                ) from error
        else:
            payload = _load_json(live_guard)
        guards = list(payload.get("guards", []))
        lint = next(
            (item for item in guards if item.get("name") == "cli_flag_lint"), None
        )
        expected = dict((lint or {}).get("args", {}).get("expected_values", {}))
        expected_n = int(match.group(1).removeprefix("n"))
        expected_c = 0.03 if match.group(2) else 0.1
        if (
            expected.get("--n-full") != expected_n
            or expected.get("--c-scale") != expected_c
            or expected.get("--n-fast") != 16
            or expected.get("--p-full") != 0.25
        ):
            raise ContractError(f"generation campaign arm guard science drift: {name}")
    if require_ready:
        raise ContractError(
            "generation campaign is not launchable: committed post-promotion handoff "
            "and a sealed copy of the canonical 56-GPU placement are required"
        )
    return value


def _campaign_science(campaign: dict[str, Any], *, n_full: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    recipe = dict(campaign["common_recipe"])
    search = {
        key: recipe[key]
        for key in _SEARCH_INPUT_KEYS
        if key != "n_full"
    }
    search["n_full"] = n_full
    evaluator = {
        key: recipe[key]
        for key in _EVALUATOR_INPUT_KEYS
    }
    generation = {
        key: recipe[key]
        for key in {
            "track",
            "vps_to_win",
            "obs_width",
            "max_decisions",
            "temperature_decisions",
            "temperature_high",
            "temperature_low",
            "late_temperature_decisions",
            "late_temperature",
            "workers_per_gpu",
            "shard_size",
            "format",
            "device",
            "eval_server",
        }
    }
    # The issued r1 generation-arm locks predate this implementation field and
    # therefore bind a generation object with no such key.  Replaying an
    # immutable r1 lock must reconstruct those historical semantics exactly;
    # only revision campaigns that explicitly carry the field may add it.
    if "native_mcts_hot_loop" in recipe:
        generation["native_mcts_hot_loop"] = bool(recipe["native_mcts_hot_loop"])
    return _search_operator(search), _effective_evaluator(evaluator), generation


def _campaign_post_wave_acceptance() -> dict[str, Any]:
    return {
        "require_complete_games": True,
        "selected_truncations_max": 0,
        "invalid_teacher_actions_max": 0,
        "require_public_observation": True,
        "require_unique_game_seeds": True,
        "require_no_val_only_overlap": True,
        "selection_before_row_expansion": True,
        "required_reports": sorted(REQUIRED_REPORTS),
        "require_shard_sha256": True,
        "require_contract_attestation": True,
        "require_target_information_regime": TARGET_INFORMATION_REGIME_PUBLIC,
        "validation_holdout": {
            "split_unit": "game_seed",
            "validation_fraction": 0.05,
            "validation_seed": 17,
            "validation_max_samples": 0,
        },
    }


def _campaign_placements(path: Path, campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = _load_json(path)
    _require_exact_keys(
        value,
        {"schema_version", "campaign_sha256", "assignments", "placement_sha256"},
        where="generation campaign placement",
    )
    unhashed = dict(value)
    declared = unhashed.pop("placement_sha256")
    if value["schema_version"] != GENERATION_PLACEMENT_SCHEMA or declared != _digest_value(unhashed):
        raise ContractError("generation campaign placement digest/schema drift")
    if value["campaign_sha256"] != campaign["contract_sha256"]:
        raise ContractError("generation placement binds a different campaign")
    return _validate_campaign_assignments(value["assignments"], campaign)


def _validate_campaign_assignments(
    raw_assignments: Any, campaign: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_assignments, list):
        raise ContractError("generation placement assignments must be a list")
    expected_lanes = {
        str(lane) for arm in campaign["arms"] for lane in arm["logical_lanes"]
    }
    assignments: dict[str, dict[str, Any]] = {}
    placements: set[tuple[str, int]] = set()
    for raw in raw_assignments:
        item = dict(raw)
        _require_exact_keys(
            item, {"logical_lane", "host_alias", "gpu"}, where="generation placement"
        )
        lane = str(item["logical_lane"])
        placement = (str(item["host_alias"]), int(item["gpu"]))
        if lane in assignments or placement in placements or int(item["gpu"]) < 0:
            raise ContractError("generation placement duplicates a lane or host/GPU")
        assignments[lane] = item
        placements.add(placement)
    if set(assignments) != expected_lanes or len(assignments) != 56:
        raise ContractError("generation placement must bind all 56 logical lanes exactly")
    fleet = _load_json(REPO_ROOT / "configs/gpu_fleet_56.json")
    authorized = {
        (str(host["alias"]), gpu)
        for host in fleet.get("hosts", [])
        for gpu in range(int(host["gpu_count"]))
    }
    if placements != authorized or len(authorized) != 56:
        raise ContractError("generation placement differs from the authorized 56-GPU fleet")
    arms_by_host: dict[str, set[str]] = {}
    for lane, item in assignments.items():
        arms_by_host.setdefault(str(item["host_alias"]), set()).add(
            lane.split("_", 1)[0]
        )
    if any(len(arms) != 1 for arms in arms_by_host.values()):
        raise ContractError("generation placement may not split one host across arms")
    return assignments


def seal_generation_placement(
    campaign_path: Path, assignments_path: Path, out_path: Path
) -> dict[str, Any]:
    """Seal an operator-supplied 56-row placement without launching anything."""

    campaign = validate_generation_campaign(campaign_path)
    raw = json.loads(assignments_path.read_text(encoding="utf-8"))
    assignments = raw.get("assignments") if isinstance(raw, dict) else raw
    if not isinstance(assignments, list):
        raise ContractError("generation placement assignments input must be a list")
    payload: dict[str, Any] = {
        "schema_version": GENERATION_PLACEMENT_SCHEMA,
        "campaign_sha256": campaign["contract_sha256"],
        "assignments": assignments,
    }
    payload["placement_sha256"] = _digest_value(payload)
    _validate_campaign_assignments(assignments, campaign)
    _create_readonly(out_path.absolute(), payload)
    return payload


def materialize_generation_campaign(
    campaign_path: Path,
    *,
    promotion_handoff_path: Path,
    placement_path: Path,
    out_dir: Path,
) -> list[Path]:
    """Create two immutable sealed arm locks; never render, claim, or launch."""

    campaign = validate_generation_campaign(campaign_path)
    if campaign.get("schema_version") in {
        GENERATION_CAMPAIGN_REVISION_SCHEMA,
        POST_PROMOTION_CAMPAIGN_SCHEMA,
    }:
        provenance = dict(campaign["provenance"])
        records = [
            *map(dict, provenance["arm_guards"]),
            *map(dict, provenance["generator_code"]),
            dict(provenance["executor"]),
            dict(provenance["harvest"]),
            dict(provenance["fleet_manifest"]),
        ]
        for record in records:
            source = REPO_ROOT / str(record["path"])
            if not source.is_file() or _sha256(source) != record["sha256"]:
                raise ContractError(
                    "generation materialization requires exact bound runtime bytes: "
                    f"{record['path']}"
                )
    if campaign.get("schema_version") == POST_PROMOTION_CAMPAIGN_SCHEMA:
        bound_handoff = campaign["promotion_handoff"]
        supplied = promotion_handoff_path.expanduser().resolve(strict=True)
        if (
            supplied != Path(str(bound_handoff["path"])).resolve(strict=True)
            or _sha256(supplied) != bound_handoff["sha256"]
        ):
            raise ContractError("materialization handoff differs from rebased campaign")
    placements = _campaign_placements(placement_path, campaign)
    checkpoints: list[dict[str, Any]] = []
    for raw in campaign["checkpoints"]:
        item = dict(raw)
        source = Path(str(item["path"]))
        if not source.is_file() or _sha256(source) != item["sha256"]:
            raise ContractError(f"generation campaign checkpoint drift: {source}")
        checkpoints.append(
            {
                **item,
                "md5": _md5(source),
            }
        )
    producer = next(item for item in checkpoints if item["role"] == "producer")
    n128_search, evaluator, generation = _campaign_science(campaign, n_full=128)
    handoff_record = _promotion_handoff_record(
        {"mode": POST_PROMOTION_HANDOFF_MODE, "path": str(promotion_handoff_path)},
        base=Path.cwd(),
        producer=producer,
        effective_search=_effective_search(n128_search),
        evaluator=evaluator,
        generation=generation,
    )
    if campaign.get("schema_version") == GENERATION_CAMPAIGN_REVISION_SCHEMA:
        _require_fresh_revision_handoff(handoff_record)
    ledger_path = Path(str(campaign["fleet"]["seed_ledger"]))
    ledger_record = _seed_ledger_snapshot(ledger_path)
    provenance = dict(campaign["provenance"])
    code_records = [
        {"kind": "generator_code", **dict(item)}
        for item in provenance["generator_code"]
    ]
    runtime_code_tree = _runtime_code_tree_records()
    guard_by_name = {
        Path(str(item["path"])).stem: dict(item)
        for item in provenance["arm_guards"]
    }
    output_paths: list[Path] = []
    out_dir = out_dir.absolute()
    for arm in campaign["arms"]:
        arm_id = str(arm["id"])
        search, arm_evaluator, arm_generation = _campaign_science(
            campaign, n_full=int(arm["n_full"])
        )
        jobs: list[dict[str, Any]] = []
        categories = tuple(arm["selected_per_gpu"])
        for lane_index, lane in enumerate(arm["logical_lanes"]):
            placement = placements[str(lane)]
            cursor = int(arm["seed_start"]) + lane_index * int(arm["seed_block_size"])
            for category in categories:
                attempts = int(arm["max_attempts_per_gpu"][category])
                job_id = f"{lane}__{category}"
                jobs.append(
                    {
                        "arm_id": arm_id,
                        "job_id": job_id,
                        "worker_id": str(lane),
                        "host_alias": str(placement["host_alias"]),
                        "gpu": int(placement["gpu"]),
                        "category": category,
                        "c_scale": float(campaign["common_recipe"]["category_c_scale"][category]),
                        "base_seed": cursor,
                        "games": int(arm["selected_per_gpu"][category]),
                        "attempts": attempts,
                        "seed_end": cursor + attempts,
                        "output_dir": str(Path(str(arm["output_root"])) / job_id),
                        "claim_label": f"{campaign['contract_id']}:{arm_id}:{job_id}",
                    }
                )
                cursor += attempts
        assert_disjoint_seed_blocks(
            [(job["job_id"], int(job["base_seed"]), int(job["attempts"])) for job in jobs]
        )
        category_games = {
            category: 28 * int(arm["selected_per_gpu"][category])
            for category in categories
        }
        category_attempts = {
            category: 28 * int(arm["max_attempts_per_gpu"][category])
            for category in categories
        }
        _validate_against_ledger(jobs, ledger_path)
        guard_configs = {
            category: dict(
                guard_by_name[
                    f"a1_generation_{arm_id}"
                    + ("" if category == "current_producer" else "_legacy")
                ]
            )
            for category in categories
        }
        lock: dict[str, Any] = {
            "schema_version": GENERATION_ARM_LOCK_SCHEMA,
            "contract_id": f"{campaign['contract_id']}-{arm_id}",
            "promotion_handoff": handoff_record,
            "source_campaign": {"path": str(campaign_path.absolute()), "sha256": _sha256(campaign_path)},
            "source_placement": {"path": str(placement_path.absolute()), "sha256": _sha256(placement_path)},
            "science": {
                "search_operator": search,
                "search_operator_sha256": _digest_value(search),
                "effective_search_config": _effective_search(search),
                "effective_search_config_sha256": _digest_value(_effective_search(search)),
                "category_search_identities": _category_search_identities(search, jobs),
                "evaluator": arm_evaluator,
                "evaluator_sha256": _digest_value(arm_evaluator),
                "value_readout": arm_evaluator["value_readout"],
            },
            "generation": arm_generation,
            "checkpoints": checkpoints,
            "source_categories": list(campaign["source_categories"]),
            "game_contract": {
                "profile": DUAL_ARM_GAME_CONTRACT_PROFILE,
                "arm_id": arm_id,
                "worker_count": 28,
                "job_count": 84,
                "total_complete_games": sum(category_games.values()),
                "category_games": category_games,
                "total_attempts": sum(category_attempts.values()),
                "category_attempts": category_attempts,
                "selection_rule": "lowest_seed_complete_per_job",
                "selection_before_row_expansion": True,
            },
            "fleet": {
                "workers": [placements[str(lane)] | {"id": str(lane)} for lane in arm["logical_lanes"]],
                "per_worker_games": dict(arm["selected_per_gpu"]),
                "max_attempts_per_worker": dict(arm["max_attempts_per_gpu"]),
                "seed_base": int(arm["seed_start"]),
                "seed_block_size": int(arm["seed_block_size"]),
                "seed_ledger": ledger_record,
                "val_only_range": list(VAL_ONLY_SEED_RANGE),
                "output_root": str(arm["output_root"]),
                "jobs": jobs,
                "seed_plan_sha256": _digest_value(jobs),
            },
            "provenance": {
                "guard_configs": {
                    category: {"kind": "guard_config", **record}
                    for category, record in guard_configs.items()
                },
                "generator_code": code_records,
                "executor": {"kind": "executor", **dict(provenance["executor"])},
                "harvest": {"kind": "harvest", **dict(provenance["harvest"])},
                "runtime_code_tree": runtime_code_tree,
                "runtime_code_tree_sha256": _digest_value(runtime_code_tree),
            },
            "post_wave_acceptance": _campaign_post_wave_acceptance(),
        }
        # Fail before sealing if any category would search the promoted
        # producer's retained seat with a different deployed c_scale.
        for job in jobs:
            _job_search_identity(lock, job)
        lock["contract_sha256"] = _digest_value(lock)
        target = out_dir / f"{arm_id}.lock.json"
        _create_readonly(target, lock)
        output_paths.append(target)
    return output_paths


def _verify_checkpoint_provenance_records(
    records: Iterable[dict[str, Any]], *, value_readout: str
) -> None:
    """Reconstruct checkpoint provenance, including legacy sidecar sources."""

    for record in records:
        metadata = dict(record.get("metadata", {}))
        attestation_record = metadata.get("legacy_scalar_readout_attestation")
        attestation_path: Path | None = None
        if attestation_record is not None:
            if not isinstance(attestation_record, dict):
                raise ContractError(
                    "legacy scalar-readout attestation lock record is malformed"
                )
            _verify_artifact_records([attestation_record])
            attestation_path = Path(str(attestation_record["path"]))
        reconstructed = _checkpoint_metadata(
            Path(str(record["path"])),
            checkpoint_sha256=str(record["sha256"]),
            value_readout=value_readout,
            require_trained_readout=(
                str(record.get("role")) == "producer" or value_readout == "categorical"
            ),
            legacy_scalar_attestation=attestation_path,
        )
        if metadata != reconstructed:
            raise ContractError(
                f"checkpoint provenance drift for {record.get('id', record['path'])}"
            )


def build_lock(
    draft_path: Path,
    *,
    seed_ledger_snapshot: dict[str, Any] | None = None,
    seed_ledger_contract_sha256: str | None = None,
) -> dict[str, Any]:
    draft_path = draft_path.absolute()
    draft = _load_json(draft_path)
    draft_schema = draft.get("schema_version")
    if draft_schema not in {DRAFT_SCHEMA, LEGACY_DRAFT_SCHEMA}:
        raise ContractError(
            f"draft schema must be {DRAFT_SCHEMA!r} or historical {LEGACY_DRAFT_SCHEMA!r}"
        )
    required_top = {
        "schema_version",
        "contract_id",
        "promotion_handoff",
        "science",
        "generation",
        "checkpoints",
        "source_categories",
        "fleet",
        "provenance",
        "post_wave_acceptance",
    }
    _require_exact_keys(draft, required_top, where="draft")
    handoff_mode = draft["promotion_handoff"].get("mode") if isinstance(
        draft["promotion_handoff"], dict
    ) else None
    if draft_schema == LEGACY_DRAFT_SCHEMA and handoff_mode != HISTORICAL_HANDOFF_MODE:
        raise ContractError(
            "legacy v2 drafts are accepted only as explicitly historical pre-promotion contracts"
        )
    if draft_schema == DRAFT_SCHEMA and handoff_mode not in {
        POST_PROMOTION_HANDOFF_MODE,
        DISASTER_RECOVERY_HANDOFF_MODE,
    }:
        raise ContractError(
            "new v3 waves require a committed promotion handoff or the one "
            "authenticated disaster-recovery receipt"
        )
    _assert_no_unresolved(draft)
    contract_id = str(draft["contract_id"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", contract_id):
        raise ContractError("contract_id must be a stable lowercase identifier")

    base = draft_path.parent
    science = dict(draft["science"])
    if set(science) != {
        "search",
        "evaluator",
        "learner_value_objective",
        "learner_training_recipe",
        "evidence",
    }:
        raise ContractError(
            "science must have exactly search, evaluator, learner_value_objective, "
            "learner_training_recipe, evidence"
        )
    raw_search = dict(science["search"])
    search = _search_operator(raw_search)
    effective_search = _effective_search(raw_search)
    evaluator = _effective_evaluator(dict(science["evaluator"]))
    learner_objective = dict(science["learner_value_objective"])
    _validate_learner_objective(learner_objective)
    learner_training_recipe = dict(science["learner_training_recipe"])
    _validate_learner_training_recipe(
        learner_training_recipe,
        expected_recipe=(
            EXPECTED_LEARNER_TRAINING_RECIPE
            if draft_schema == LEGACY_DRAFT_SCHEMA
            else CURRENT_LEARNER_TRAINING_RECIPE
        ),
    )
    if evaluator["public_observation"] is not True:
        raise ContractError("science.evaluator.public_observation must be true")
    if search["information_set_search"] is not True:
        raise ContractError(
            "A1 public-observation policy targets require information_set_search=true"
        )
    if int(search["determinization_particles"]) < 1:
        raise ContractError("determinization_particles must be >= 1")
    if int(search["determinization_particles"]) > int(search["n_fast"]):
        raise ContractError(
            "determinization_particles cannot exceed the n_fast total budget"
        )
    if int(search["determinization_min_simulations"]) < 1:
        raise ContractError("determinization_min_simulations must be >= 1")
    if bool(search["belief_chance_spectra"]):
        raise ContractError(
            "belief_chance_spectra must be false under full-world determinization"
        )
    if evaluator["value_readout"] not in {"scalar", "categorical"}:
        raise ContractError("value_readout must be scalar or categorical")
    if evaluator["value_squash"] != "tanh":
        raise ContractError(
            "A1 generator currently implements only value_squash=tanh"
        )
    if float(evaluator["context_fill"]) != 0.0:
        raise ContractError(
            "A1 generator does not expose context_fill; it must remain 0"
        )
    if int(evaluator["cache_size"]) < 0:
        raise ContractError("evaluator cache_size must be non-negative")
    if evaluator["prior_temperature"] != search["prior_temperature"]:
        raise ContractError("search/evaluator prior_temperature values must match")
    if bool(evaluator["emit_uncertainty"]):
        raise ContractError(
            "A1 isolation does not enable the unselected uncertainty path"
        )
    if search["n_full"] < search["n_fast"] or not 0.0 <= float(search["p_full"]) <= 1.0:
        raise ContractError("invalid n_fast/n_full/p_full search budget")
    if int(search["n_fast"]) != 16 or int(search["n_full"]) not in {64, 128}:
        raise ContractError(
            "A1 permits n_fast=16 and global n_full in {64,128}; never global n256"
        )
    if search["wide_roots_always_full"] and search["n_full_wide"] is None:
        raise ContractError("wide_roots_always_full requires n_full_wide")
    if search["n_full_wide"] is None:
        if (
            search["n_full_wide_threshold"] is not None
            or search["wide_roots_always_full"]
        ):
            raise ContractError(
                "disabled adaptive n256 must use null threshold and always_full=false"
            )
    elif (
        search["n_full_wide_threshold"] is None or not search["wide_roots_always_full"]
    ):
        raise ContractError(
            "selected adaptive n256 requires an explicit threshold and always_full=true"
        )
    elif (
        int(search["n_full_wide"]) != 256 or int(search["n_full_wide_threshold"]) != 40
    ):
        raise ContractError(
            "the only permitted n256 arm is adaptive n_full_wide=256 at >=40"
        )
    if (
        search["symmetry_averaged_eval"]
        and search["symmetry_averaged_eval_threshold"] is None
    ):
        raise ContractError("selected D6 requires an explicit independent threshold")
    if (
        not search["symmetry_averaged_eval"]
        and search["symmetry_averaged_eval_threshold"] is not None
    ):
        raise ContractError("disabled D6 must use a null D6 threshold")
    if (
        search["symmetry_averaged_eval"]
        and int(search["symmetry_averaged_eval_threshold"]) < 20
    ):
        raise ContractError("selected D6 threshold must remain independent and >=20")
    if not search["exact_budget_sh"] and int(search["exact_budget_sh_min_n"]) != 0:
        raise ContractError("disabled exact-budget SH must use exact_budget_sh_min_n=0")
    if (
        float(search["rescale_noise_floor_c"]) > 0.0
        and float(search["sigma_eval"]) <= 0.0
    ):
        raise ContractError("selected D1 requires positive sigma_eval")

    evidence_raw = list(science["evidence"])
    evidence_kinds = {str(item.get("kind")) for item in evidence_raw}
    if evidence_kinds != REQUIRED_EVIDENCE or len(evidence_raw) != len(
        REQUIRED_EVIDENCE
    ):
        raise ContractError(
            f"science.evidence must contain exactly {sorted(REQUIRED_EVIDENCE)}"
        )
    evidence = []
    post_promotion_s1_path: Path | None = None
    allow_post_promotion_s1 = False
    recovery_s1_path: Path | None = None
    allow_recovery_s1 = False
    if draft_schema == DRAFT_SCHEMA and handoff_mode == POST_PROMOTION_HANDOFF_MODE:
        post_promotion_s1_path = _absolute_ref(
            str(draft["promotion_handoff"].get("path", "")), base=base
        )
        allow_post_promotion_s1 = True
    elif (
        draft_schema == DRAFT_SCHEMA
        and handoff_mode == DISASTER_RECOVERY_HANDOFF_MODE
    ):
        recovery_s1_path = _absolute_ref(
            str(draft["promotion_handoff"].get("path", "")), base=base
        )
        allow_recovery_s1 = True
    for item in sorted(evidence_raw, key=lambda item: str(item["kind"])):
        if set(item) != {"kind", "path"}:
            continue
        evidence_path = _absolute_ref(str(item["path"]), base=base)
        evidence_payload = _load_json(evidence_path)
        record = _file_record(evidence_path, kind=str(item["kind"]))
        record["document_schema"] = evidence_payload.get("schema_version")
        record["document_digest"] = _digest_value(evidence_payload)
        if item["kind"] == "a0":
            record["semantic_decision"] = _validate_a0_evidence(
                evidence_payload,
                path=evidence_path,
                learner_objective=learner_objective,
            )
        else:
            record["semantic_decision"] = _validate_search_stage_evidence(
                evidence_payload,
                path=evidence_path,
                expected_stage=str(item["kind"]),
                final_search=search,
                final_evaluator=evaluator,
                post_promotion_handoff_path=post_promotion_s1_path,
                allow_post_promotion_s1=allow_post_promotion_s1,
                recovery_receipt_path=recovery_s1_path,
                allow_recovery_s1=allow_recovery_s1,
            )
        evidence.append(record)
    if len(evidence) != len(evidence_raw):
        raise ContractError("each evidence entry must have exactly kind and path")

    generation = dict(draft["generation"])
    _validate_generation(generation)
    _validate_current_runtime_execution(
        str(draft_schema), evaluator=evaluator, generation=generation
    )
    if (
        learner_training_recipe["track"] != generation["track"]
        or learner_training_recipe["vps_to_win"] != generation["vps_to_win"]
    ):
        raise ContractError(
            "learner track/vps_to_win must match the locked generation regime"
        )
    if int(generation["obs_width"]) != 806 or not bool(
        learner_training_recipe["graph_history_features"]
    ):
        raise ContractError(
            "A1 fixes obs_width=806 and requires graph_history_features=true "
            "to match the gen3 learner regime"
        )
    checkpoint_records = _checkpoint_records(
        list(draft["checkpoints"]),
        base=base,
        value_readout=str(evaluator["value_readout"]),
        draft_schema=str(draft_schema),
    )
    evidence_by_kind = {str(record["kind"]): record for record in evidence}
    for stage, predecessors in {"s2": ("s1",), "s3": ("s1", "s2")}.items():
        stage_sources = {
            (str(record["path"]), str(record["sha256"]))
            for record in evidence_by_kind[stage]["semantic_decision"][
                "source_artifacts"
            ]
        }
        for predecessor in predecessors:
            expected = evidence_by_kind[predecessor]
            identity = (str(expected["path"]), str(expected["sha256"]))
            if identity not in stage_sources:
                raise ContractError(
                    f"{stage.upper()} does not inherit the exact sealed "
                    f"{predecessor.upper()} decision artifact"
                )
    producer_record = next(
        record for record in checkpoint_records if record["role"] == "producer"
    )
    handoff_record = _promotion_handoff_record(
        draft["promotion_handoff"],
        base=base,
        producer=producer_record,
        effective_search=effective_search,
        evaluator=evaluator,
        generation=generation,
    )
    if draft_schema == DRAFT_SCHEMA:
        _bind_v3_checkpoint_lineage(checkpoint_records, handoff_record)
    for stage in ("s1", "s2", "s3"):
        stage_checkpoint = evidence_by_kind[stage]["semantic_decision"]["checkpoint"]
        if (
            stage_checkpoint["sha256"] != producer_record["sha256"]
            or Path(stage_checkpoint["path"]).resolve(strict=True)
            != Path(producer_record["path"]).resolve(strict=True)
        ):
            raise ContractError(
                f"{stage.upper()} adjudicated a different teacher checkpoint than "
                "the production contract"
            )
    categories = _validate_categories(
        list(draft["source_categories"]), checkpoint_records
    )

    fleet = dict(draft["fleet"])
    fleet_manifest_record: dict[str, Any] | None = None
    worker_quotas: dict[str, dict[str, int]] | None = None
    if draft_schema == LEGACY_DRAFT_SCHEMA:
        expected_fleet_fields = {
            "workers",
            "per_worker_games",
            "seed_base",
            "seed_block_size",
            "seed_ledger",
            "output_root",
        }
    else:
        expected_fleet_fields = {
            "fleet_manifest",
            "quota_policy",
            "seed_base",
            "seed_block_size",
            "seed_ledger",
            "output_root",
        }
    if set(fleet) != expected_fleet_fields:
        raise ContractError(
            "fleet fields are not the exact "
            f"{'historical v2' if draft_schema == LEGACY_DRAFT_SCHEMA else 'current v3'} "
            "pre-wave schema"
        )
    ledger_path = _absolute_ref(str(fleet["seed_ledger"]), base=base)
    output_root = Path(str(fleet["output_root"])).expanduser()
    if not output_root.is_absolute():
        raise ContractError(
            "fleet.output_root must be an absolute path shared by the data lane"
        )
    if draft_schema == LEGACY_DRAFT_SCHEMA:
        selected_games = dict(EXPECTED_GAMES)
        quota_policy = BALANCED_PREFIX_QUOTA_POLICY
        workers = list(fleet["workers"])
        jobs = _build_jobs(
            workers,
            seed_base=int(fleet["seed_base"]),
            block_size=int(fleet["seed_block_size"]),
            per_worker={
                str(k): int(v)
                for k, v in dict(fleet["per_worker_games"]).items()
            },
            output_root=str(output_root),
            contract_id=contract_id,
        )
    else:
        quota_policy = str(fleet["quota_policy"])
        selected_games = _games_for_quota_policy(quota_policy)
        fleet_manifest_path = _absolute_ref(
            str(fleet["fleet_manifest"]), base=base
        )
        if fleet_manifest_path != CURRENT_FLEET_MANIFEST.resolve(strict=True):
            raise ContractError(
                "v3 fleet_manifest must bind the checked-in authoritative "
                f"{CURRENT_FLEET_MANIFEST.relative_to(REPO_ROOT)}"
            )
        workers, fleet_manifest_record = _canonical_workers_from_fleet_manifest(
            fleet_manifest_path
        )
        jobs, worker_quotas = _build_balanced_jobs(
            workers,
            seed_base=int(fleet["seed_base"]),
            block_size=int(fleet["seed_block_size"]),
            output_root=str(output_root),
            contract_id=contract_id,
            quota_policy=quota_policy,
        )
    if seed_ledger_snapshot is None:
        _validate_against_ledger(jobs, ledger_path)
        ledger_record = _seed_ledger_snapshot(ledger_path)
    else:
        if seed_ledger_contract_sha256 is None:
            raise ContractError(
                "rebuilding from a seed-ledger snapshot requires the sealed contract SHA-256"
            )
        ledger_record = dict(seed_ledger_snapshot)
        try:
            locked_ledger_path = Path(str(ledger_record["path"])).resolve(strict=True)
        except (KeyError, OSError) as error:
            raise ContractError(f"invalid locked seed-ledger snapshot: {error}") from error
        if locked_ledger_path != ledger_path.resolve(strict=True):
            raise ContractError(
                "source draft seed-ledger path differs from its immutable snapshot"
            )
        _verify_live_seed_ledger(
            ledger_record,
            jobs,
            contract_sha256=seed_ledger_contract_sha256,
            require_all_job_claims=False,
        )

    provenance = dict(draft["provenance"])
    if set(provenance) != {
        "guard_config",
        "generator_code_files",
        "learner_code_files",
    }:
        raise ContractError(
            "provenance must have exactly guard_config, generator_code_files, "
            "and learner_code_files"
        )
    guard_path = _absolute_ref(str(provenance["guard_config"]), base=base)
    _validate_guard(
        guard_path,
        search=search,
        evaluator=evaluator,
        generation=generation,
        s1_evidence=evidence_by_kind["s1"],
    )
    guard_record = _file_record(guard_path, kind="guard_config")
    code_records = [
        _file_record(_absolute_ref(str(raw), base=base), kind="generator_code")
        for raw in provenance["generator_code_files"]
    ]
    code_paths = {Path(record["path"]).as_posix() for record in code_records}
    missing_code = {
        suffix
        for suffix in REQUIRED_GENERATOR_CODE_SUFFIXES
        if not any(path.endswith(suffix) for path in code_paths)
    }
    if missing_code:
        raise ContractError(
            f"generator_code_files omits required semantics files: {sorted(missing_code)}"
        )
    learner_code_records = [
        _file_record(_absolute_ref(str(raw), base=base), kind="learner_code")
        for raw in provenance["learner_code_files"]
    ]
    learner_code_paths = {
        Path(record["path"]).as_posix() for record in learner_code_records
    }
    missing_learner_code = {
        suffix
        for suffix in REQUIRED_LEARNER_CODE_SUFFIXES
        if not any(path.endswith(suffix) for path in learner_code_paths)
    }
    if missing_learner_code:
        raise ContractError(
            "learner_code_files omits required learner semantics files: "
            f"{sorted(missing_learner_code)}"
        )
    runtime_code_tree = _runtime_code_tree_records()
    _validate_post_wave(dict(draft["post_wave_acceptance"]))

    category_attempts = {
        category: sum(
            int(job["attempts"])
            for job in jobs
            if job["category"] == category
        )
        for category in selected_games
    }
    lock_fleet: dict[str, Any] = {
        "workers": workers,
        "seed_base": int(fleet["seed_base"]),
        "seed_block_size": int(fleet["seed_block_size"]),
        "seed_ledger": ledger_record,
        "val_only_range": list(VAL_ONLY_SEED_RANGE),
        "output_root": str(output_root),
        "jobs": jobs,
        "seed_plan_sha256": _digest_value(jobs),
    }
    if draft_schema == LEGACY_DRAFT_SCHEMA:
        lock_fleet["per_worker_games"] = dict(EXPECTED_PER_WORKER)
    else:
        assert fleet_manifest_record is not None and worker_quotas is not None
        lock_fleet.update(
            {
                "fleet_manifest": fleet_manifest_record,
                "quota_policy": quota_policy,
                "worker_quotas": worker_quotas,
                "worker_quotas_sha256": _digest_value(worker_quotas),
            }
        )

    lock: dict[str, Any] = {
        "schema_version": (
            LEGACY_LOCK_SCHEMA if draft_schema == LEGACY_DRAFT_SCHEMA else LOCK_SCHEMA
        ),
        "contract_id": contract_id,
        "promotion_handoff": handoff_record,
        "source_draft": {"path": str(draft_path), "sha256": _sha256(draft_path)},
        "science": {
            "search_operator": search,
            "search_operator_sha256": _digest_value(search),
            "effective_search_config": effective_search,
            "effective_search_config_sha256": _digest_value(effective_search),
            "evaluator": evaluator,
            "evaluator_sha256": _digest_value(evaluator),
            "value_readout": evaluator["value_readout"],
            "learner_value_objective": learner_objective,
            "learner_value_objective_sha256": _digest_value(learner_objective),
            "learner_training_recipe": learner_training_recipe,
            "learner_training_recipe_sha256": _digest_value(
                learner_training_recipe
            ),
            "evidence": evidence,
        },
        "generation": generation,
        "checkpoints": checkpoint_records,
        "source_categories": categories,
        **(
            {}
            if draft_schema == LEGACY_DRAFT_SCHEMA
            else {
                "category_semantics": _category_semantics(
                    checkpoint_records, handoff_record
                )
            }
        ),
        "game_contract": {
            **(
                {}
                if draft_schema == LEGACY_DRAFT_SCHEMA
                else {
                    "profile": _profile_for_quota_policy(quota_policy),
                    "worker_count": CURRENT_WORKER_COUNT,
                    "job_count": CURRENT_WORKER_COUNT * len(selected_games),
                }
            ),
            "total_complete_games": sum(selected_games.values()),
            "category_games": dict(selected_games),
            "total_attempts": sum(category_attempts.values()),
            "category_attempts": category_attempts,
            "selection_rule": "lowest_seed_complete_per_job",
            "selection_before_row_expansion": True,
        },
        "fleet": lock_fleet,
        "provenance": {
            "guard_config": guard_record,
            "generator_code": code_records,
            "learner_code": learner_code_records,
            "learner_code_sha256": _digest_value(learner_code_records),
            "runtime_code_tree": runtime_code_tree,
            "runtime_code_tree_sha256": _digest_value(runtime_code_tree),
        },
        "post_wave_acceptance": dict(draft["post_wave_acceptance"]),
    }
    # Make the promoted checkpoint/operator join a seal-time invariant.  It is
    # not sufficient for render/audit to discover the mismatch after a lock
    # has already been issued.
    for job in jobs:
        _job_search_identity(lock, job)
    lock["contract_sha256"] = _digest_value(lock)
    return lock


def verify_lock(
    lock_path: Path, *, require_all_job_claims: bool = False
) -> dict[str, Any]:
    lock = _load_json(lock_path)
    lock_schema = lock.get("schema_version")
    if lock_schema == GENERATION_ARM_LOCK_SCHEMA:
        return _verify_generation_arm_lock(
            lock, require_all_job_claims=require_all_job_claims
        )
    if lock_schema not in {LOCK_SCHEMA, LEGACY_LOCK_SCHEMA}:
        raise ContractError(
            f"lock schema must be {LOCK_SCHEMA!r} or historical {LEGACY_LOCK_SCHEMA!r}"
        )
    _assert_no_unresolved(lock)
    expected_digest = str(lock.get("contract_sha256", ""))
    unhashed = dict(lock)
    unhashed.pop("contract_sha256", None)
    actual_digest = _digest_value(unhashed)
    if expected_digest != actual_digest:
        raise ContractError(
            f"contract digest mismatch: expected {expected_digest or '<missing>'}, got {actual_digest}"
        )
    _sealed_game_contract_shape(lock)
    historical_markerless = _is_historical_markerless_a1_lock(lock_path, lock)
    _verify_artifact_records([lock["source_draft"]])
    handoff_record = lock.get("promotion_handoff")
    if handoff_record is None and historical_markerless:
        pass
    elif not isinstance(handoff_record, dict):
        raise ContractError("lock does not bind an explicit promotion handoff mode")
    elif handoff_record.get("mode") == POST_PROMOTION_HANDOFF_MODE:
        if lock_schema != LOCK_SCHEMA:
            raise ContractError("post-promotion handoffs require the v3 lock schema")
        _verify_artifact_records([handoff_record])
    elif handoff_record.get("mode") == DISASTER_RECOVERY_HANDOFF_MODE:
        if lock_schema != LOCK_SCHEMA:
            raise ContractError("disaster recovery requires the v3 lock schema")
        _verify_artifact_records([handoff_record])
        if (
            handoff_record.get("wave_lineage_mode")
            != RECOVERY_REFERENCE_SEMANTIC
            or handoff_record.get("promotion_proof_recreated") is not False
        ):
            raise ContractError("disaster-recovery handoff policy drift")
    elif handoff_record.get("mode") != HISTORICAL_HANDOFF_MODE:
        raise ContractError("lock promotion handoff mode is invalid")
    elif lock_schema != LEGACY_LOCK_SCHEMA:
        raise ContractError("historical handoff mode is accepted only in legacy v2 locks")
    else:
        _require_exact_keys(
            handoff_record, {"mode", "reason"}, where="historical promotion_handoff"
        )
        reason = handoff_record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ContractError("historical promotion_handoff reason must be non-empty")
    _verify_artifact_records(lock["science"]["evidence"])
    for evidence in lock["science"]["evidence"]:
        _verify_artifact_records(evidence["semantic_decision"]["source_artifacts"])
    _verify_artifact_records(lock["checkpoints"])
    if lock_schema == LOCK_SCHEMA:
        category_semantics = lock.get("category_semantics")
        if category_semantics is None:
            if handoff_record.get("mode") == DISASTER_RECOVERY_HANDOFF_MODE:
                raise ContractError(
                    "recovery lock cannot omit category semantics"
                )
        elif category_semantics != _category_semantics(
            lock["checkpoints"], handoff_record
        ):
            raise ContractError("sealed source-category semantics drift")
        for record in lock["checkpoints"]:
            version = record.get("version")
            if isinstance(version, bool) or not isinstance(version, int) or version < 0:
                raise ContractError(
                    f"v3 checkpoint {record.get('id')} has no authenticated version"
                )
            if record.get("role") == "history":
                lineage = record.get("lineage")
                if handoff_record.get("mode") == DISASTER_RECOVERY_HANDOFF_MODE:
                    if (
                        not isinstance(lineage, dict)
                        or lineage.get("relation") != RECOVERY_REFERENCE_RELATION
                        or lineage.get("semantic") != RECOVERY_REFERENCE_SEMANTIC
                        or lineage.get("causal_parent_proven") is not False
                        or lineage.get("promotion_proof_recreated") is not False
                        or not isinstance(lineage.get("recovery_receipt"), dict)
                        or "promotion_receipt" in lineage
                    ):
                        raise ContractError(
                            "v3 recovery-reference lineage binding drift"
                        )
                    _verify_artifact_records([lineage["recovery_receipt"]])
                else:
                    if (
                        not isinstance(lineage, dict)
                        or lineage.get("relation")
                        != "immediate_displaced_incumbent"
                        or not isinstance(lineage.get("promotion_receipt"), dict)
                    ):
                        raise ContractError("v3 recent-history lineage binding drift")
                    _verify_artifact_records([lineage["promotion_receipt"]])
            elif record.get("role") == "hard_negative":
                selection = record.get("selection_evidence")
                if not isinstance(selection, dict):
                    raise ContractError("v3 hard-negative selection evidence is missing")
                rebuilt = _hard_negative_selection_record(
                    Path(str(selection["path"])),
                    checkpoint=Path(str(record["path"])),
                    checkpoint_sha256=str(record["sha256"]),
                )
                if rebuilt != selection:
                    raise ContractError("v3 hard-negative sealed evidence record drift")
                _verify_artifact_records(
                    _hard_negative_selection_artifacts(selection)
                )
        _bind_hard_negative_incumbent_to_producer(lock["checkpoints"])
    _verify_artifact_records([lock["provenance"]["guard_config"]])
    verify_code_records = (
        _verify_archived_code_provenance_records
        if historical_markerless
        else _verify_artifact_records
    )
    verify_code_records(lock["provenance"]["generator_code"])
    learner_code_records = lock["provenance"].get("learner_code")
    if not isinstance(learner_code_records, list) or not learner_code_records:
        raise ContractError("lock does not bind the learner implementation")
    if lock["provenance"].get("learner_code_sha256") != _digest_value(
        learner_code_records
    ):
        raise ContractError("learner-code provenance digest drift")
    verify_code_records(learner_code_records)
    runtime_code_tree = lock["provenance"].get("runtime_code_tree")
    if not isinstance(runtime_code_tree, list) or not runtime_code_tree:
        raise ContractError("lock does not bind the transitive runtime code tree")
    if lock["provenance"].get("runtime_code_tree_sha256") != _digest_value(
        runtime_code_tree
    ):
        raise ContractError("runtime-code-tree provenance digest drift")
    verify_code_records(runtime_code_tree)
    search = dict(lock["science"]["search_operator"])
    effective_search = dict(lock["science"]["effective_search_config"])
    evaluator = dict(lock["science"]["evaluator"])
    if _digest_value(search) != lock["science"]["search_operator_sha256"]:
        raise ContractError("search operator digest mismatch")
    if (
        _digest_value(effective_search)
        != lock["science"]["effective_search_config_sha256"]
    ):
        raise ContractError("effective search-config digest mismatch")
    if historical_markerless:
        # The sealed effective config predates several dataclass defaults and
        # serialized colors as a list. Validate every explicit operator field
        # above, but never reinterpret its authenticated historical effective
        # bytes through today's expanding runtime dataclass.
        _validate_search_operator_fields(
            search, expected_keys=HISTORICAL_MARKERLESS_SEARCH_INPUT_KEYS
        )
    else:
        reconstructed_effective = _effective_search(search)
        if _digest_value(reconstructed_effective) != _digest_value(effective_search):
            raise ContractError(
                "effective search config does not reconstruct from selected operator"
            )
    if _digest_value(evaluator) != lock["science"]["evaluator_sha256"]:
        raise ContractError("evaluator digest mismatch")
    if evaluator["value_readout"] != lock["science"]["value_readout"]:
        raise ContractError("value readout attestation drift")
    _verify_checkpoint_provenance_records(
        lock["checkpoints"], value_readout=str(evaluator["value_readout"])
    )
    learner_objective = dict(lock["science"]["learner_value_objective"])
    _validate_learner_objective(learner_objective)
    if (
        _digest_value(learner_objective)
        != lock["science"]["learner_value_objective_sha256"]
    ):
        raise ContractError("learner value-objective digest mismatch")
    learner_training_recipe = dict(lock["science"]["learner_training_recipe"])
    _validate_learner_training_recipe(
        learner_training_recipe,
        expected_recipe=(
            HISTORICAL_MARKERLESS_LEARNER_TRAINING_RECIPE
            if historical_markerless
            else (
                EXPECTED_LEARNER_TRAINING_RECIPE
                if lock_schema == LEGACY_LOCK_SCHEMA
                else CURRENT_LEARNER_TRAINING_RECIPE
            )
        ),
    )
    if (
        _digest_value(learner_training_recipe)
        != lock["science"]["learner_training_recipe_sha256"]
    ):
        raise ContractError("learner training-recipe digest mismatch")
    if lock_schema == LOCK_SCHEMA:
        fleet_manifest = lock["fleet"].get("fleet_manifest")
        if not isinstance(fleet_manifest, dict):
            raise ContractError("v3 lock does not bind its authoritative fleet manifest")
        _verify_artifact_records([fleet_manifest])
        manifest_path = Path(str(fleet_manifest.get("path", ""))).resolve(strict=True)
        if manifest_path != CURRENT_FLEET_MANIFEST.resolve(strict=True):
            raise ContractError("v3 lock fleet manifest path is not authoritative")
        canonical_workers, canonical_manifest_record = (
            _canonical_workers_from_fleet_manifest(manifest_path)
        )
        if canonical_manifest_record != fleet_manifest:
            raise ContractError("v3 lock fleet manifest record drift")
        if lock["fleet"].get("workers") != canonical_workers:
            raise ContractError("v3 lock worker topology/order drift")
        quota_policy = str(lock["fleet"].get("quota_policy", ""))
        selected_games = _games_for_quota_policy(quota_policy)
        expected_profile = _profile_for_quota_policy(quota_policy)
        if lock["game_contract"].get("profile") != expected_profile:
            raise ContractError("v3 lock game/quota profile drift")
        expected_quotas = _balanced_worker_quotas(
            canonical_workers, quota_policy=quota_policy
        )
        if (
            lock["fleet"].get("worker_quotas") != expected_quotas
            or lock["fleet"].get("worker_quotas_sha256")
            != _digest_value(expected_quotas)
        ):
            raise ContractError("v3 lock balanced worker quotas drift")
    else:
        selected_games = dict(EXPECTED_GAMES)
    jobs = list(lock["fleet"]["jobs"])
    if _digest_value(jobs) != lock["fleet"]["seed_plan_sha256"]:
        raise ContractError("seed plan digest mismatch")
    for job in jobs:
        _job_search_identity(lock, job)
    assert_disjoint_seed_blocks(
        [
            (job["job_id"], int(job["base_seed"]), int(job["attempts"]))
            for job in jobs
        ]
    )
    if Counter({category: 0 for category in selected_games}) + Counter(
        {
            category: sum(int(j["games"]) for j in jobs if j["category"] == category)
            for category in selected_games
        }
    ) != Counter(selected_games):
        raise ContractError(
            f"job category totals drifted from sealed profile {selected_games}"
        )
    if (
        lock["game_contract"].get("category_games") != selected_games
        or lock["game_contract"].get("total_complete_games")
        != sum(selected_games.values())
    ):
        raise ContractError("game-contract selected-game totals drifted")
    observed_attempts = {
        category: sum(
            int(j["attempts"]) for j in jobs if j["category"] == category
        )
        for category in selected_games
    }
    expected_attempts = (
        EXPECTED_ATTEMPTS
        if lock_schema == LEGACY_LOCK_SCHEMA
        else {
            category: int(selected_games[category])
            + CURRENT_WORKER_COUNT * int(ATTEMPT_RESERVE_PER_JOB[category])
            for category in selected_games
        }
    )
    if Counter(observed_attempts) != Counter(expected_attempts):
        raise ContractError("job attempt totals drifted from the bounded reserve")
    if (
        lock["game_contract"].get("category_attempts") != expected_attempts
        or lock["game_contract"].get("total_attempts")
        != sum(expected_attempts.values())
    ):
        raise ContractError("game-contract attempt totals drifted")
    for job in jobs:
        if _ranges_overlap(
            (int(job["base_seed"]), int(job["seed_end"])),
            tuple(lock["fleet"]["val_only_range"]),
        ):
            raise ContractError(f"job {job['job_id']} overlaps VAL-ONLY")
    _verify_live_seed_ledger(
        dict(lock["fleet"]["seed_ledger"]),
        jobs,
        contract_sha256=expected_digest,
        require_all_job_claims=require_all_job_claims,
    )
    _validate_guard(
        Path(lock["provenance"]["guard_config"]["path"]),
        search=search,
        evaluator=evaluator,
        generation=dict(lock["generation"]),
        s1_evidence=next(
            record for record in lock["science"]["evidence"] if record["kind"] == "s1"
        ),
        archived_markerless=historical_markerless,
    )
    _validate_post_wave(dict(lock["post_wave_acceptance"]))
    if not historical_markerless:
        rebuilt = build_lock(
            Path(str(lock["source_draft"]["path"])),
            seed_ledger_snapshot=dict(lock["fleet"]["seed_ledger"]),
            seed_ledger_contract_sha256=expected_digest,
        )
        if _digest_value(rebuilt) != _digest_value(lock):
            raise ContractError(
                "lock does not exactly reconstruct from its immutable source draft"
            )
    return lock


def _verify_generation_arm_lock(
    lock: dict[str, Any], *, require_all_job_claims: bool
) -> dict[str, Any]:
    expected_digest = str(lock.get("contract_sha256", ""))
    unhashed = dict(lock)
    unhashed.pop("contract_sha256", None)
    if expected_digest != _digest_value(unhashed):
        raise ContractError("generation arm lock semantic digest mismatch")
    _sealed_game_contract_shape(lock)
    game = dict(lock.get("game_contract", {}))
    _require_exact_keys(
        game,
        {
            "profile",
            "arm_id",
            "worker_count",
            "job_count",
            "total_complete_games",
            "category_games",
            "total_attempts",
            "category_attempts",
            "selection_rule",
            "selection_before_row_expansion",
        },
        where="generation arm game contract",
    )
    arm_id = str(game.get("arm_id"))
    if (
        game.get("profile") != DUAL_ARM_GAME_CONTRACT_PROFILE
        or arm_id not in {"n256", "n128"}
        or int(game.get("worker_count", -1)) != 28
        or int(game.get("job_count", -1)) != 84
    ):
        raise ContractError("generation arm profile drift")
    if (
        game["selection_rule"] != "lowest_seed_complete_per_job"
        or game["selection_before_row_expansion"] is not True
    ):
        raise ContractError("generation arm selection policy drift")
    _verify_artifact_records(
        [lock["source_campaign"], lock["source_placement"], lock["promotion_handoff"]]
    )
    campaign_path = Path(str(lock["source_campaign"]["path"]))
    campaign = validate_generation_campaign(
        campaign_path,
        _allow_historical_lock_source=True,
    )
    historical_lock_source = (
        campaign["contract_sha256"] == HISTORICAL_DB1_CAMPAIGN_SHA256
    )
    historical_implementation_commit = _campaign_historical_implementation_commit(
        campaign_path,
        campaign,
        historical_lock_source=historical_lock_source,
    )
    placements = _campaign_placements(
        Path(str(lock["source_placement"]["path"])), campaign
    )
    arm = next(item for item in campaign["arms"] if item["id"] == arm_id)
    search, evaluator, generation = _campaign_science(
        campaign, n_full=int(arm["n_full"])
    )
    science = dict(lock["science"])
    if (
        _digest_value(science["search_operator"]) != _digest_value(search)
        or science["search_operator_sha256"] != _digest_value(search)
        or _digest_value(science["effective_search_config"])
        != _digest_value(_effective_search(search))
        or science["effective_search_config_sha256"]
        != _digest_value(_effective_search(search))
        or _digest_value(science["evaluator"]) != _digest_value(evaluator)
        or science["evaluator_sha256"] != _digest_value(evaluator)
        # Historical A1 locks were sealed before the explicit generation
        # ``native_mcts_hot_loop`` field was added.  Its omitted value was the
        # production default (False); normalize only that one additive field
        # so old sealed locks remain replayable without accepting science drift.
        or _digest_value(
            {
                **lock["generation"],
                **(
                    {
                        "native_mcts_hot_loop": generation[
                            "native_mcts_hot_loop"
                        ]
                    }
                    if "native_mcts_hot_loop" in generation
                    and "native_mcts_hot_loop" not in lock["generation"]
                    else {}
                ),
            }
        )
        != _digest_value(generation)
    ):
        raise ContractError("generation arm science drift")
    for expected, actual in zip(campaign["checkpoints"], lock["checkpoints"]):
        source = Path(str(actual["path"]))
        if (
            {key: actual[key] for key in expected} != expected
            or not source.is_file()
            or _sha256(source) != actual["sha256"]
            or _md5(source) != actual["md5"]
        ):
            raise ContractError("generation arm checkpoint drift")
    producer = next(item for item in lock["checkpoints"] if item["role"] == "producer")
    n128_search, n128_evaluator, n128_generation = _campaign_science(
        campaign, n_full=128
    )
    rebuilt_handoff = _promotion_handoff_record(
        {
            "mode": POST_PROMOTION_HANDOFF_MODE,
            "path": str(lock["promotion_handoff"]["path"]),
        },
        base=Path.cwd(),
        producer=producer,
        effective_search=_effective_search(n128_search),
        evaluator=n128_evaluator,
        generation=n128_generation,
    )
    if rebuilt_handoff != lock["promotion_handoff"]:
        raise ContractError("generation arm promotion handoff replay drift")
    if lock["source_categories"] != campaign["source_categories"]:
        raise ContractError("generation arm source categories drift")
    if lock.get("post_wave_acceptance") != _campaign_post_wave_acceptance():
        raise ContractError("generation arm post-wave acceptance drift")
    jobs = list(lock["fleet"]["jobs"])
    if len(jobs) != 84 or _digest_value(jobs) != lock["fleet"]["seed_plan_sha256"]:
        raise ContractError("generation arm job plan drift")
    for job in jobs:
        _job_search_identity(lock, job)
    if any(
        job.get("arm_id") != arm_id
        or float(job.get("c_scale", -1.0))
        != float(campaign["common_recipe"]["category_c_scale"][job["category"]])
        for job in jobs
    ):
        raise ContractError("generation arm job identity drift")
    expected_category_identities = _category_search_identities(search, jobs)
    sealed_category_identities = science.get("category_search_identities")
    if (
        sealed_category_identities is not None
        and _digest_value(sealed_category_identities)
        != _digest_value(expected_category_identities)
    ):
        raise ContractError("generation arm realized search identity drift")
    expected_workers = [
        placements[str(lane)] | {"id": str(lane)} for lane in arm["logical_lanes"]
    ]
    if lock["fleet"]["workers"] != expected_workers:
        raise ContractError("generation arm physical placement drift")
    expected_jobs: list[dict[str, Any]] = []
    for lane_index, lane in enumerate(arm["logical_lanes"]):
        placement = placements[str(lane)]
        cursor = int(arm["seed_start"]) + lane_index * int(arm["seed_block_size"])
        for category in arm["selected_per_gpu"]:
            attempts = int(arm["max_attempts_per_gpu"][category])
            job_id = f"{lane}__{category}"
            expected_jobs.append(
                {
                    "arm_id": arm_id,
                    "job_id": job_id,
                    "worker_id": str(lane),
                    "host_alias": str(placement["host_alias"]),
                    "gpu": int(placement["gpu"]),
                    "category": category,
                    "c_scale": float(
                        campaign["common_recipe"]["category_c_scale"][category]
                    ),
                    "base_seed": cursor,
                    "games": int(arm["selected_per_gpu"][category]),
                    "attempts": attempts,
                    "seed_end": cursor + attempts,
                    "output_dir": str(Path(str(arm["output_root"])) / job_id),
                    "claim_label": f"{campaign['contract_id']}:{arm_id}:{job_id}",
                }
            )
            cursor += attempts
    if jobs != expected_jobs:
        raise ContractError("generation arm jobs do not reconstruct from campaign/placement")
    assert_disjoint_seed_blocks(
        [(str(job["job_id"]), int(job["base_seed"]), int(job["attempts"])) for job in jobs]
    )
    category_games = {
        category: sum(int(job["games"]) for job in jobs if job["category"] == category)
        for category in arm["selected_per_gpu"]
    }
    category_attempts = {
        category: sum(int(job["attempts"]) for job in jobs if job["category"] == category)
        for category in arm["max_attempts_per_gpu"]
    }
    if (
        category_games != game["category_games"]
        or category_attempts != game["category_attempts"]
        or sum(category_games.values()) != int(game["total_complete_games"])
        or sum(category_attempts.values()) != int(game["total_attempts"])
    ):
        raise ContractError("generation arm category totals drift")
    records = [
        *lock["provenance"]["guard_configs"].values(),
        *lock["provenance"]["generator_code"],
        lock["provenance"]["executor"],
        lock["provenance"]["harvest"],
        *lock["provenance"]["runtime_code_tree"],
    ]
    campaign_provenance = dict(campaign["provenance"])
    campaign_bound_records = [
        *map(dict, campaign_provenance["arm_guards"]),
        *map(dict, campaign_provenance["generator_code"]),
        dict(campaign_provenance["executor"]),
        dict(campaign_provenance["harvest"]),
        dict(campaign_provenance["fleet_manifest"]),
    ]
    campaign_bound_identities = {
        (str(record["path"]), str(record["sha256"]))
        for record in campaign_bound_records
    }
    historical_repo_root: Path | None = None
    if historical_lock_source:
        campaign_source = Path(str(lock["source_campaign"]["path"]))
        if len(campaign_source.parents) >= 4:
            historical_repo_root = campaign_source.parents[3]
    for record in records:
        relative_path = str(record["path"])
        digest = str(record["sha256"])
        if (
            historical_implementation_commit is not None
            and (relative_path, digest) in campaign_bound_identities
        ):
            _historical_campaign_provenance_bytes(
                historical_implementation_commit,
                relative_path,
                digest,
            )
            continue
        raw_path = Path(relative_path)
        source = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
        historical_source = (
            historical_repo_root / raw_path
            if historical_repo_root is not None and not raw_path.is_absolute()
            else None
        )
        current_matches = source.is_file() and _sha256(source) == digest
        historical_matches = bool(
            historical_source is not None
            and historical_source.is_file()
            and _sha256(historical_source) == digest
        )
        if not current_matches and not historical_matches:
            raise ContractError(f"generation arm provenance drift: {source}")
    if _digest_value(lock["provenance"]["runtime_code_tree"]) != lock["provenance"][
        "runtime_code_tree_sha256"
    ]:
        raise ContractError("generation arm runtime tree digest drift")
    _verify_live_seed_ledger(
        dict(lock["fleet"]["seed_ledger"]),
        jobs,
        contract_sha256=expected_digest,
        require_all_job_claims=require_all_job_claims,
    )
    return lock


def _bool_flag(name: str, value: bool) -> str:
    return name if value else "--no-" + name.removeprefix("--")


def _producer(lock: dict[str, Any]) -> dict[str, Any]:
    return next(
        record for record in lock["checkpoints"] if record["role"] == "producer"
    )


def _category_by_name(lock: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        category for category in lock["source_categories"] if category["name"] == name
    )


def _category_opponent_sha256(lock: dict[str, Any], name: str) -> list[str]:
    """Exact opponent bytes; self-play names the producer on the second seat."""

    if name == "current_producer":
        return [_producer(lock)["sha256"]]
    category = _category_by_name(lock, name)
    checkpoints = {record["id"]: record for record in lock["checkpoints"]}
    return sorted(
        checkpoints[checkpoint_id]["sha256"]
        for checkpoint_id in category["checkpoint_ids"]
    )


def _sealed_category_semantic(
    lock: Mapping[str, Any], category: str
) -> dict[str, Any] | None:
    """Return the optional v3 scheduler-lane meaning without inventing one.

    Issued pre-recovery locks did not carry this field, so their byte-level
    attestations remain valid.  New locks seal it and every downstream artifact
    must copy it exactly; in particular, the stable ``recent_history`` scheduler
    identifier cannot launder a recovery safety reference into a causal-parent
    claim.
    """

    semantics = lock.get("category_semantics")
    if semantics is None:
        return None
    if not isinstance(semantics, Mapping):
        raise ContractError("source-category semantics are malformed")
    value = semantics.get(category)
    if not isinstance(value, Mapping):
        raise ContractError(f"source-category semantic is missing for {category}")
    return dict(value)


def _render_mix_manifest(lock: dict[str, Any], category: str) -> dict[str, Any]:
    spec = _category_by_name(lock, category)
    by_id = {record["id"]: record for record in lock["checkpoints"]}
    category_semantic = _sealed_category_semantic(lock, category)
    return {
        "_a1_contract": {
            "contract_sha256": lock["contract_sha256"],
            "category": category,
            **(
                {}
                if category_semantic is None
                else {"category_semantic": category_semantic}
            ),
        },
        "categories": [
            {
                "name": category,
                "weight": 1.0,
                "source": "checkpoint_list",
                "pending": False,
                "engine": None,
                "checkpoints": [
                    {
                        "path": by_id[checkpoint_id]["path"],
                        "version": int(by_id[checkpoint_id].get("version", -1)),
                        "md5": by_id[checkpoint_id]["md5"],
                        # The runtime parser intentionally consumes the legacy
                        # path/version/md5 triple. Keep SHA-256 alongside it in
                        # the sealed manifest so render/attestation verification
                        # can authenticate the stronger identity without
                        # weakening older readers.
                        **(
                            {"sha256": by_id[checkpoint_id]["sha256"]}
                            if lock.get("schema_version") == LOCK_SCHEMA
                            else {}
                        ),
                    }
                    for checkpoint_id in spec["checkpoint_ids"]
                ],
            }
        ],
    }


def _generator_argv(
    lock: dict[str, Any], job: dict[str, Any], *, mix_paths: dict[str, Path]
) -> list[str]:
    search = lock["science"]["search_operator"]
    evaluator = lock["science"]["evaluator"]
    generation = lock["generation"]
    producer = _producer(lock)
    argv = [
        "tools/generate_gumbel_selfplay_data.py",
        "--out-dir",
        job["output_dir"],
        "--games",
        str(job["attempts"]),
        "--workers",
        str(generation["workers_per_gpu"]),
        "--checkpoint",
        producer["path"],
        "--device",
        generation["device"],
        "--n-full",
        str(search["n_full"]),
        "--n-fast",
        str(search["n_fast"]),
        "--p-full",
        str(search["p_full"]),
        "--c-visit",
        str(search["c_visit"]),
        "--c-scale",
        str(job.get("c_scale", search["c_scale"])),
        "--rescale-noise-floor-c",
        str(search["rescale_noise_floor_c"]),
        "--sigma-eval",
        str(search["sigma_eval"]),
        "--wide-candidates-threshold",
        str(search["wide_candidates_threshold"]),
        "--max-depth",
        str(search["max_depth"]),
        "--max-decisions",
        str(generation["max_decisions"]),
        "--temperature-decisions",
        str(generation["temperature_decisions"]),
        "--temperature-high",
        str(generation["temperature_high"]),
        "--temperature-low",
        str(generation["temperature_low"]),
        "--late-temperature",
        str(generation["late_temperature"]),
        "--prior-temperature",
        str(evaluator["prior_temperature"]),
        "--value-scale",
        str(evaluator["value_scale"]),
        "--value-readout",
        str(evaluator["value_readout"]),
        "--eval-cache-size",
        str(evaluator["cache_size"]),
        "--track",
        generation["track"],
        "--vps-to-win",
        str(generation["vps_to_win"]),
        "--obs-width",
        str(generation["obs_width"]),
        "--base-seed",
        str(job["base_seed"]),
        "--shard-size",
        str(generation["shard_size"]),
        "--format",
        generation["format"],
        "--ledger-claim-label",
        job["claim_label"],
        _bool_flag("--symmetry-averaged-eval", bool(search["symmetry_averaged_eval"])),
        _bool_flag("--wide-roots-always-full", bool(search["wide_roots_always_full"])),
        _bool_flag(
            "--correct-rust-chance-spectra", bool(search["correct_rust_chance_spectra"])
        ),
        _bool_flag("--lazy-interior-chance", bool(search["lazy_interior_chance"])),
        _bool_flag("--exact-budget-sh", bool(search["exact_budget_sh"])),
        "--exact-budget-sh-min-n",
        str(search["exact_budget_sh_min_n"]),
        _bool_flag("--belief-chance-spectra", bool(search["belief_chance_spectra"])),
        _bool_flag("--information-set-search", bool(search["information_set_search"])),
        _bool_flag(
            # Issued pre-native arm locks and their resumable executor receipts
            # predate this implementation field. Future draft validation
            # requires the key explicitly; only legacy/render-only shapes may
            # reach this compatibility default.
            "--native-mcts-hot-loop",
            bool(generation.get("native_mcts_hot_loop", False)),
        ),
        "--determinization-particles",
        str(search["determinization_particles"]),
        "--determinization-min-simulations",
        str(search["determinization_min_simulations"]),
        _bool_flag("--public-observation", bool(evaluator["public_observation"])),
        _bool_flag("--rust-featurize", bool(evaluator["rust_featurize"])),
        _bool_flag("--eval-server", bool(generation["eval_server"])),
        "--seed-claim",
        "--resume",
    ]
    if lock.get("schema_version") == GENERATION_ARM_LOCK_SCHEMA:
        argv.extend(
            (
                "--prelaunch-guard-config",
                str(lock["provenance"]["guard_configs"][job["category"]]["path"]),
                "--generation-arm-id",
                str(job["arm_id"]),
            )
        )
    optional = (
        ("--n-full-wide", search["n_full_wide"]),
        ("--n-full-wide-threshold", search["n_full_wide_threshold"]),
        ("--raw-policy-above-width", search["raw_policy_above_width"]),
        (
            "--symmetry-averaged-eval-threshold",
            search["symmetry_averaged_eval_threshold"],
        ),
        ("--late-temperature-decisions", generation["late_temperature_decisions"]),
    )
    for flag, value in optional:
        if value is not None:
            argv.extend((flag, str(value)))
    if job["category"] != "current_producer":
        argv.extend(("--opponent-mix-manifest", str(mix_paths[job["category"]])))
    return argv


def _job_environment(lock: dict[str, Any], job: dict[str, Any]) -> dict[str, str]:
    """Return the complete sealed environment for one writable job sandbox."""

    output_dir = Path(str(job["output_dir"]))
    return {
        **SEALED_RUNTIME_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
        "CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE_DIRECTORY,
        "CUDA_MPS_LOG_DIRECTORY": MPS_LOG_DIRECTORY,
        "CATAN_SEED_LEDGER": str(lock["fleet"]["seed_ledger"]["path"]),
        "CATAN_A1_CONTRACT_SHA256": str(lock["contract_sha256"]),
        CONFIG_REGISTRY_ENVIRONMENT_VARIABLE: str(
            output_dir / CONFIG_REGISTRY_FILENAME
        ),
    }


def _job_attestation(lock: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    search_identity = _job_search_identity(lock, job)
    promoted_identity = _promoted_producer_job_identity(lock, job)
    category_semantic = _sealed_category_semantic(lock, str(job["category"]))
    return {
        "schema_version": GENERATION_JOB_ATTESTATION_SCHEMA,
        **({} if "arm_id" not in job else {"arm_id": job["arm_id"]}),
        **({} if "c_scale" not in job else {"c_scale": job["c_scale"]}),
        "contract_sha256": lock["contract_sha256"],
        "seed_plan_sha256": lock["fleet"]["seed_plan_sha256"],
        "job_id": job["job_id"],
        "worker_id": job["worker_id"],
        "category": job["category"],
        **(
            {}
            if category_semantic is None
            else {"category_semantic": category_semantic}
        ),
        "base_seed": job["base_seed"],
        "games": job["games"],
        "attempts": job["attempts"],
        "seed_end": job["seed_end"],
        "producer_checkpoint_sha256": _producer(lock)["sha256"],
        **(
            {}
            if promoted_identity is None
            else {
                "producer_checkpoint_search_identity_sha256": promoted_identity[
                    "checkpoint_search_identity_sha256"
                ]
            }
        ),
        "opponent_checkpoint_sha256": _category_opponent_sha256(
            lock, job["category"]
        ),
        "search_operator_sha256": search_identity["search_operator_sha256"],
        "effective_search_config_sha256": search_identity[
            "effective_search_config_sha256"
        ],
        "evaluator_sha256": lock["science"]["evaluator_sha256"],
        "runtime_code_tree_sha256": lock["provenance"][
            "runtime_code_tree_sha256"
        ],
        "teacher_value_readout": lock["science"]["value_readout"],
    }


def _legacy_job_attestation(lock: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct immutable v2 sidecars issued before realized-operator binding."""

    value = _job_attestation(lock, job)
    value["schema_version"] = LEGACY_GENERATION_JOB_ATTESTATION_SCHEMA
    value["search_operator_sha256"] = lock["science"]["search_operator_sha256"]
    value["effective_search_config_sha256"] = lock["science"][
        "effective_search_config_sha256"
    ]
    return value


def _ledger_claim_row(lock: dict[str, Any], job: dict[str, Any]) -> str:
    return (
        f"[{int(job['base_seed'])} – {int(job['seed_end'])}) | "
        f"{_ledger_claim_label(str(lock['contract_sha256']), job)}"
    )


def _validate_claim_render(
    lock: dict[str, Any], render_path: Path
) -> tuple[dict[str, Any], list[str]]:
    """Reconstruct every rendered claim from the sealed lock."""

    rendered = _load_json(render_path)
    if rendered.get("schema_version") != RENDER_SCHEMA:
        raise ContractError(f"render schema must be {RENDER_SCHEMA!r}")
    unhashed = dict(rendered)
    declared_digest = unhashed.pop("render_sha256", None)
    if declared_digest != _digest_value(unhashed):
        raise ContractError("render semantic digest mismatch")
    if rendered.get("contract_sha256") != lock["contract_sha256"]:
        raise ContractError("render binds a different sealed contract")
    commands = rendered.get("commands")
    jobs = {str(job["job_id"]): job for job in lock["fleet"]["jobs"]}
    expected_jobs = int(_sealed_game_contract_shape(lock)["job_count"])
    if (
        not isinstance(commands, list)
        or len(commands) != len(jobs)
        or len(jobs) != expected_jobs
    ):
        raise ContractError(
            f"claim transaction requires exactly {expected_jobs} rendered jobs"
        )
    expected_path = str(lock["fleet"]["seed_ledger"]["path"])
    rows_by_job: dict[str, str] = {}
    for command in commands:
        if not isinstance(command, dict):
            raise ContractError("rendered claim command is not an object")
        job_id = str(command.get("job_id", ""))
        if job_id not in jobs or job_id in rows_by_job:
            raise ContractError(f"unknown or duplicate rendered claim job {job_id!r}")
        expected_row = _ledger_claim_row(lock, jobs[job_id])
        claim = command.get("ledger_claim")
        if not isinstance(claim, dict) or set(claim) != {
            "path",
            "row",
            "row_sha256",
        }:
            raise ContractError(f"rendered ledger claim fields drift for {job_id}")
        if (
            claim["path"] != expected_path
            or claim["row"] != expected_row
            or claim["row_sha256"] != _digest_value(expected_row)
        ):
            raise ContractError(f"rendered ledger claim differs from lock for {job_id}")
        rows_by_job[job_id] = expected_row
    if set(rows_by_job) != set(jobs):
        raise ContractError("rendered claims do not cover the exact sealed job set")
    rows = [rows_by_job[str(job["job_id"])] for job in lock["fleet"]["jobs"]]
    if len(set(rows)) != len(rows):
        raise ContractError("sealed jobs render duplicate ledger rows")
    return rendered, rows


def _durable_replace(path: Path, data: bytes) -> None:
    """O_EXCL temp + fsync + replace + parent fsync."""

    path = path.resolve(strict=True)
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.claim-{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _create_durable_readonly(path: Path, payload: dict[str, Any]) -> None:
    """Create one immutable transaction receipt and durably link its name."""

    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
    except FileExistsError as error:
        raise ContractError(f"refusing to overwrite claim receipt {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _verify_claim_receipt(
    receipt_path: Path,
    *,
    lock: dict[str, Any],
    rendered: dict[str, Any],
    rows: list[str],
    ledger_bytes: bytes,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path)
    expected_fields = {
        "schema_version",
        "status",
        "contract_sha256",
        "render_sha256",
        "ledger_path",
        "ledger_before_sha256",
        "ledger_after_sha256",
        "ledger_after_size_bytes",
        "claim_count",
        "claims_sha256",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise ContractError("claim receipt fields drift")
    unhashed = dict(receipt)
    declared = unhashed.pop("receipt_sha256", None)
    if declared != _digest_value(unhashed):
        raise ContractError("claim receipt semantic digest mismatch")
    if (
        receipt["schema_version"] != CLAIM_RECEIPT_SCHEMA
        or receipt["status"] not in {"claimed", "already_claimed"}
        or receipt["contract_sha256"] != lock["contract_sha256"]
        or receipt["render_sha256"] != rendered["render_sha256"]
        or receipt["ledger_path"] != str(lock["fleet"]["seed_ledger"]["path"])
        or receipt["claim_count"] != len(rows)
        or receipt["claims_sha256"] != _digest_value(rows)
    ):
        raise ContractError("claim receipt binds different contract/render/rows")
    size = int(receipt["ledger_after_size_bytes"])
    if size < 0 or len(ledger_bytes) < size:
        raise ContractError("live ledger is shorter than claim receipt prefix")
    if "sha256:" + hashlib.sha256(ledger_bytes[:size]).hexdigest() != receipt[
        "ledger_after_sha256"
    ]:
        raise ContractError("live ledger claim-receipt prefix drift")
    return receipt


def claim_seed_ledger(
    lock_path: Path, render_path: Path, receipt_path: Path
) -> dict[str, Any]:
    """Atomically install every sealed job claim, or validate the prior transaction."""

    lock_path = lock_path.absolute()
    render_path = render_path.absolute()
    receipt_path = receipt_path.absolute()
    # The first verification rejects spoofed, duplicated, overlapping, or
    # non-append-only live rows before any mutation. It is repeated under the
    # advisory lock so cooperating operators cannot race the transaction.
    lock = verify_lock(lock_path)
    rendered, rows = _validate_claim_render(lock, render_path)
    ledger = Path(str(lock["fleet"]["seed_ledger"]["path"])).absolute()
    sidecar = ledger.with_name(ledger.name + ".a1-claim.lock")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(sidecar, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        lock = verify_lock(lock_path)
        rendered, rows = _validate_claim_render(lock, render_path)
        before = ledger.read_bytes()
        if receipt_path.exists():
            receipt = _verify_claim_receipt(
                receipt_path,
                lock=lock,
                rendered=rendered,
                rows=rows,
                ledger_bytes=before,
            )
            verify_lock(lock_path, require_all_job_claims=True)
            return receipt

        _text, live_claims = _read_strict_ledger(ledger)
        counts: list[int] = []
        for job in lock["fleet"]["jobs"]:
            expected = (
                int(job["base_seed"]),
                int(job["seed_end"]),
                _ledger_claim_label(str(lock["contract_sha256"]), job),
            )
            counts.append(sum(claim == expected for claim in live_claims))
        if any(count not in (0, 1) for count in counts):
            raise ContractError("live ledger repeats an exact own claim")
        present = sum(counts)
        if present not in (0, len(rows)):
            raise ContractError(
                f"refusing partial own claim set: found {present}/{len(rows)} exact rows"
            )
        status = "already_claimed" if present == len(rows) else "claimed"
        after = before
        if present == 0:
            if not before.endswith(b"\n"):
                raise ContractError("live seed ledger must end with a newline")
            after = before + b"".join(row.encode("utf-8") + b"\n" for row in rows)
            _durable_replace(ledger, after)
        verify_lock(lock_path, require_all_job_claims=True)
        receipt = {
            "schema_version": CLAIM_RECEIPT_SCHEMA,
            "status": status,
            "contract_sha256": lock["contract_sha256"],
            "render_sha256": rendered["render_sha256"],
            "ledger_path": str(ledger),
            "ledger_before_sha256": "sha256:" + hashlib.sha256(before).hexdigest(),
            "ledger_after_sha256": "sha256:" + hashlib.sha256(after).hexdigest(),
            "ledger_after_size_bytes": len(after),
            "claim_count": len(rows),
            "claims_sha256": _digest_value(rows),
        }
        receipt["receipt_sha256"] = _digest_value(receipt)
        _create_durable_readonly(receipt_path, receipt)
        return receipt
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _create_readonly(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
    except FileExistsError as error:
        raise ContractError(
            f"refusing to overwrite immutable artifact {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _create_or_verify_readonly(path: Path, payload: dict[str, Any]) -> None:
    """Durably create deterministic output, or verify an exact prior write."""

    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        try:
            _create_readonly(path, payload)
            return
        except ContractError:
            # A cooperating retry may have won O_EXCL after our absence check.
            try:
                descriptor = os.open(path, flags)
            except OSError as error:
                raise ContractError(
                    f"cannot replay immutable artifact {path}: {error}"
                ) from error
    except OSError as error:
        raise ContractError(f"cannot replay immutable artifact {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"immutable artifact is not regular: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or b"".join(chunks) != serialized:
            raise ContractError(f"existing immutable artifact differs: {path}")
    finally:
        os.close(descriptor)


def render(lock_path: Path, out_dir: Path) -> dict[str, Any]:
    lock = verify_lock(lock_path)
    out_dir = out_dir.absolute()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ContractError(f"render output must be absent or empty: {out_dir}")
    mix_paths: dict[str, Path] = {}
    for category in ("recent_history", "hard_negative"):
        path = out_dir / "opponent_mix" / f"{category}.json"
        _create_readonly(path, _render_mix_manifest(lock, category))
        mix_paths[category] = path
    commands = []
    for job in lock["fleet"]["jobs"]:
        argv = _generator_argv(lock, job, mix_paths=mix_paths)
        environment = _job_environment(lock, job)
        config_provenance = _expected_generate_config_provenance(
            lock,
            job,
            opponent_mix_manifest=(
                None
                if job["category"] == "current_producer"
                else str(mix_paths[job["category"]])
            ),
        )
        attestation = _job_attestation(lock, job)
        attestation_source = out_dir / "job_attestations" / f"{job['job_id']}.json"
        _create_readonly(attestation_source, attestation)
        commands.append(
            {
                **({} if "arm_id" not in job else {"arm_id": job["arm_id"]}),
                "job_id": job["job_id"],
                "worker_id": job["worker_id"],
                "host_alias": job["host_alias"],
                "gpu": job["gpu"],
                "category": job["category"],
                "environment": environment,
                "environment_sha256": _digest_value(environment),
                "config_provenance": config_provenance,
                "python": "python",
                "argv": argv,
                "argv_sha256": _digest_value(argv),
                "ledger_claim": {
                    "path": lock["fleet"]["seed_ledger"]["path"],
                    "row": _ledger_claim_row(lock, job),
                    "row_sha256": _digest_value(_ledger_claim_row(lock, job)),
                },
                "output_attestation": {
                    "source": str(attestation_source),
                    "source_file_sha256": _sha256(attestation_source),
                    "destination": str(Path(job["output_dir"]) / "a1_contract.json"),
                    "payload_sha256": _digest_value(attestation),
                },
                "must_run_after": (
                    []
                    if job["category"] == "current_producer"
                    else [
                        f"{job['worker_id']}__{list(EXPECTED_GAMES)[list(EXPECTED_GAMES).index(job['category']) - 1]}"
                    ]
                ),
            }
        )
    payload = {
        "schema_version": RENDER_SCHEMA,
        "contract_path": str(lock_path.absolute()),
        "contract_sha256": lock["contract_sha256"],
        "required_artifacts": {
            "checkpoints": [
                {key: record[key] for key in ("id", "path", "sha256", "md5")}
                for record in lock["checkpoints"]
            ],
            "seed_ledger": lock["fleet"]["seed_ledger"],
            **(
                {"guard_config": lock["provenance"]["guard_config"]}
                if "guard_config" in lock["provenance"]
                else {"guard_configs": list(lock["provenance"]["guard_configs"].values())}
            ),
            "generator_code": lock["provenance"]["generator_code"],
            "learner_code": lock["provenance"].get("learner_code", []),
            "learner_code_sha256": lock["provenance"].get(
                "learner_code_sha256", _digest_value([])
            ),
            "runtime_code_tree": lock["provenance"]["runtime_code_tree"],
            "runtime_code_tree_sha256": lock["provenance"][
                "runtime_code_tree_sha256"
            ],
            "rendered_opponent_mix": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in mix_paths.values()
            ],
        },
        "execution_policy": {
            "execute": False,
            "category_jobs_are_sequential_per_gpu": True,
            "operator_must_claim_ledger_before_each_job": True,
            "operator_must_copy_output_attestation_before_each_job": True,
            "operator_must_run_post_wave_audit_before_ingest": True,
        },
        "commands": commands,
    }
    payload["render_sha256"] = _digest_value(payload)
    _create_readonly(out_dir / "commands.json", payload)
    return payload


@dataclasses.dataclass(frozen=True)
class _HarvestRelocation:
    path: Path
    payload: dict[str, Any]
    by_source: dict[str, Path]
    output_roots: tuple[PurePath, ...]

    def resolve(self, raw: str | Path, *, owner_source: str | Path | None = None) -> Path:
        source = PurePath(str(raw))
        if not source.is_absolute():
            if owner_source is None:
                raise ContractError(f"relative relocated path has no owner: {raw}")
            source = PurePath(str(owner_source)).parent / source
        if ".." in source.parts:
            raise ContractError(f"relocated path contains traversal: {raw}")
        key = str(source)
        relocated = self.by_source.get(key)
        if relocated is not None:
            return relocated
        if any(source == root or root in source.parents for root in self.output_roots):
            raise ContractError(f"harvest relocation is missing sealed output {key}")
        original = Path(key)
        if not original.is_file():
            raise ContractError(f"unrelocated external artifact is missing: {key}")
        return original.absolute()


def _load_harvest_relocation(
    path: Path, *, lock: dict[str, Any], lock_path: Path | None = None
) -> _HarvestRelocation:
    try:
        relocation_path = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve harvest relocation map {path}: {error}") from error
    payload = _load_json(relocation_path)
    unhashed = dict(payload)
    declared = unhashed.pop("relocation_sha256", None)
    if payload.get("schema_version") != HARVEST_RELOCATION_SCHEMA:
        raise ContractError("harvest relocation schema drift")
    if declared != _digest_value(unhashed):
        raise ContractError("harvest relocation semantic digest mismatch")
    if payload.get("contract_sha256") != lock["contract_sha256"]:
        raise ContractError("harvest relocation binds a different A1 contract")
    arm_id = lock.get("game_contract", {}).get("arm_id")
    if arm_id is not None and payload.get("arm_id") != arm_id:
        raise ContractError("harvest relocation arm identity drift")
    if lock_path is not None and (
        payload.get("contract_path") != str(lock_path)
        or payload.get("contract_file_sha256") != _sha256(lock_path)
    ):
        raise ContractError("harvest relocation immutable lock-file identity drift")
    jobs = list(lock["fleet"]["jobs"])
    identity_keys = (
        "job_id",
        "worker_id",
        "host_alias",
        "gpu",
        "category",
        "output_dir",
    ) + (() if arm_id is None else ("arm_id",))
    expected_identities = [
        {key: job[key] for key in identity_keys}
        for job in jobs
    ]
    if (
        payload.get("job_count") != len(jobs)
        or payload.get("host_count") != len({job["host_alias"] for job in jobs})
        or payload.get("job_identities") != expected_identities
        or payload.get("job_identities_sha256") != _digest_value(expected_identities)
    ):
        raise ContractError("harvest relocation job/host identity drift")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("harvest relocation has no file inventory")
    if payload.get("file_inventory_sha256") != _digest_value(files):
        raise ContractError("harvest relocation file inventory digest mismatch")
    root = relocation_path.parent
    by_source: dict[str, Path] = {}
    seen_local: set[Path] = set()
    expected_fields = {
        "source_path",
        "relative_path",
        "size_bytes",
        "sha256",
        "job_id",
        "host_alias",
    }
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    for index, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ContractError(f"harvest relocation file record {index} fields drift")
        source = PurePath(str(record["source_path"]))
        relative = PurePath(str(record["relative_path"]))
        job = jobs_by_id.get(str(record["job_id"]))
        if (
            job is None
            or record["host_alias"] != job["host_alias"]
            or not source.is_absolute()
            or ".." in source.parts
            or not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2] != ("jobs", str(job["job_id"]))
            or not (
                source == PurePath(str(job["output_dir"]))
                or PurePath(str(job["output_dir"])) in source.parents
            )
        ):
            raise ContractError(f"harvest relocation file record {index} path/identity drift")
        local = root.joinpath(*relative.parts)
        try:
            canonical = local.resolve(strict=True)
        except OSError as error:
            raise ContractError(f"harvested file {index} is missing: {error}") from error
        if canonical != local.absolute() or not canonical.is_file():
            raise ContractError(f"harvested file {index} uses a symlink or is not regular")
        if source.as_posix() in by_source or canonical in seen_local:
            raise ContractError("harvest relocation repeats a source or local file")
        if (
            isinstance(record["size_bytes"], bool)
            or int(record["size_bytes"]) != canonical.stat().st_size
            or record["sha256"] != _sha256(canonical)
        ):
            raise ContractError(f"harvested file {index} byte digest/size mismatch")
        by_source[source.as_posix()] = canonical
        seen_local.add(canonical)
    local_items = list((root / "jobs").rglob("*"))
    if any(item.is_symlink() for item in local_items):
        raise ContractError("harvested jobs tree contains an unbound symlink")
    if any(not item.is_dir() and not item.is_file() for item in local_items):
        raise ContractError("harvested jobs tree contains an unbound special file")
    actual_local = {
        item.resolve(strict=True) for item in local_items if item.is_file()
    }
    if actual_local != seen_local:
        raise ContractError("harvest relocation does not bijectively cover local job files")
    return _HarvestRelocation(
        path=relocation_path,
        payload=payload,
        by_source=by_source,
        output_roots=tuple(PurePath(str(job["output_dir"])) for job in jobs),
    )


def _resolve_shard(
    manifest: Path,
    raw: str,
    *,
    relocation: _HarvestRelocation | None = None,
    manifest_source: Path | None = None,
) -> Path:
    path = Path(raw)
    # A frozen handoff may not guess by basename or process cwd.  Such fallback
    # made a stale absolute path silently bind unrelated bytes with the same
    # filename.  Relative shard paths have exactly one owner: the manifest.
    if relocation is not None:
        resolved = relocation.resolve(
            raw, owner_source=manifest_source if manifest_source is not None else manifest
        )
    else:
        resolved = path if path.is_absolute() else manifest.parent / path
    if not resolved.is_file():
        raise ContractError(f"manifest {manifest} points to missing shard {raw}")
    return resolved.absolute()


def _row_legal(action: int, legal: np.ndarray, mask: np.ndarray | None) -> bool:
    if mask is not None:
        legal = legal[np.asarray(mask, dtype=bool)]
    return bool(np.any(np.asarray(legal, dtype=np.int64) == int(action)))


def _selected_telemetry_arrays(
    payload: Any,
    *,
    game_seeds: np.ndarray,
    selected_mask: np.ndarray,
    max_decisions: int,
    where: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load mandatory report inputs without inventing fallback telemetry."""

    missing = sorted(
        column
        for column in REQUIRED_SELECTED_TELEMETRY_COLUMNS
        if column not in payload
    )
    if missing:
        raise ContractError(f"{where}: missing selected telemetry columns {missing}")

    row_aligned: dict[str, np.ndarray] = {}
    for column in ("is_forced", "used_full_search", "phase", "decision_index"):
        raw = np.asarray(payload[column])
        if raw.shape != game_seeds.shape:
            raise ContractError(
                f"{where}: telemetry column {column} is not row-aligned"
            )
        row_aligned[column] = raw[selected_mask]

    phase = row_aligned["phase"].astype(str)
    if np.any(np.char.str_len(phase) == 0) or np.any(phase == "<missing>"):
        raise ContractError(f"{where}: selected phase telemetry is empty")
    decision = np.asarray(row_aligned["decision_index"], dtype=np.int64)
    if np.any(decision < 0) or np.any(decision >= int(max_decisions)):
        raise ContractError(
            f"{where}: selected decision_index is outside [0,{max_decisions})"
        )

    raw_policy = np.asarray(payload["target_policy"], dtype=np.float64)
    raw_policy_mask = np.asarray(payload["target_policy_mask"], dtype=bool)
    if (
        raw_policy.ndim != 2
        or raw_policy.shape[0] != game_seeds.size
        or raw_policy.shape[1] == 0
        or raw_policy_mask.shape != raw_policy.shape
    ):
        raise ContractError(
            f"{where}: target_policy/target_policy_mask are empty or not row-aligned"
        )
    policy = raw_policy[selected_mask]
    policy_mask = raw_policy_mask[selected_mask]
    if np.any(~np.any(policy_mask, axis=1)):
        raise ContractError(f"{where}: selected target policy has no active entries")
    active_values = policy[policy_mask]
    if np.any(~np.isfinite(active_values)) or np.any(active_values < 0.0):
        raise ContractError(
            f"{where}: selected target policy has invalid active probabilities"
        )
    mass = np.where(policy_mask, policy, 0.0).sum(axis=1)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ContractError(f"{where}: selected target policy has non-positive mass")

    return (
        np.asarray(row_aligned["is_forced"], dtype=bool),
        np.asarray(row_aligned["used_full_search"], dtype=bool),
        phase,
        decision,
        policy,
        policy_mask,
    )


def _advance_game_seed_runs(
    game_seeds: np.ndarray,
    *,
    active_seed: int | None,
    closed_seeds: set[int],
    where: str,
) -> int | None:
    """Track one contiguous raw row run per game across ordered shards.

    Keeping ``active_seed`` across calls permits a game to span adjacent shard
    files.  Once another seed begins, the old run is closed forever; seeing it
    again would duplicate a whole/partial game under the same selected seed.
    """

    flat = np.asarray(game_seeds, dtype=np.int64).reshape(-1)
    current = active_seed
    for raw_seed in flat:
        seed = int(raw_seed)
        if current is None:
            if seed in closed_seeds:
                raise ContractError(
                    f"{where}: game_seed {seed} starts a second non-contiguous raw run"
                )
            current = seed
            continue
        if seed == current:
            continue
        closed_seeds.add(current)
        if seed in closed_seeds:
            raise ContractError(
                f"{where}: game_seed {seed} starts a second non-contiguous raw run"
            )
        current = seed
    return current


def _advance_game_decision_run(
    game_seeds: np.ndarray,
    decision_indices: np.ndarray,
    *,
    active_seed: int | None,
    active_decision_index: int | None,
    where: str,
) -> int | None:
    """Reject an adjacent second copy of a game that seed runs cannot see.

    Current generation emits at most one selected row per game decision, in
    chronological order.  Thus ``decision_index`` increases strictly within a
    seed even when only one player's rows are retained.  A non-increase while
    the seed remains equal is a new/copy game boundary, not a continuation.
    Negative indices are legacy/unknown provenance and are not used for this
    stronger check.
    """

    seeds = np.asarray(game_seeds, dtype=np.int64).reshape(-1)
    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    if decisions.shape != seeds.shape:
        raise ContractError(
            f"{where}: decision_index is not row-aligned with game_seed"
        )
    if not seeds.size:
        return active_decision_index
    if (
        active_seed == int(seeds[0])
        and active_decision_index is not None
        and int(decisions[0]) >= 0
        and int(decisions[0]) <= active_decision_index
    ):
        raise ContractError(
            f"{where}: game_seed {int(seeds[0])} decision_index resets across "
            "a shard boundary; adjacent duplicate game"
        )
    if seeds.size > 1:
        same_seed = seeds[1:] == seeds[:-1]
        known = (decisions[1:] >= 0) & (decisions[:-1] >= 0)
        reset_offsets = np.flatnonzero(
            same_seed & known & (decisions[1:] <= decisions[:-1])
        )
        if reset_offsets.size:
            offset = int(reset_offsets[0] + 1)
            raise ContractError(
                f"{where}: game_seed {int(seeds[offset])} decision_index "
                f"non-increase {int(decisions[offset - 1])}->{int(decisions[offset])}; "
                "adjacent duplicate game"
            )
    final = int(decisions[-1])
    return final if final >= 0 else None


def _expected_cli_fields(lock: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    search = lock["science"]["search_operator"]
    evaluator = lock["science"]["evaluator"]
    generation = lock["generation"]
    late_decisions = generation["late_temperature_decisions"]
    return {
        "out_dir": job["output_dir"],
        "games": int(job["attempts"]),
        "workers": int(generation["workers_per_gpu"]),
        "checkpoint": _producer(lock)["path"],
        "device": generation["device"],
        "n_full": search["n_full"],
        "n_fast": search["n_fast"],
        "p_full": search["p_full"],
        "c_visit": search["c_visit"],
        "c_scale": job.get("c_scale", search["c_scale"]),
        "rescale_noise_floor_c": search["rescale_noise_floor_c"],
        "sigma_eval": search["sigma_eval"],
        "n_full_wide": search["n_full_wide"],
        "n_full_wide_threshold": search["n_full_wide_threshold"],
        "wide_roots_always_full": search["wide_roots_always_full"],
        "raw_policy_above_width": search["raw_policy_above_width"],
        "symmetry_averaged_eval": search["symmetry_averaged_eval"],
        "symmetry_averaged_eval_threshold": search["symmetry_averaged_eval_threshold"],
        "wide_candidates_threshold": search["wide_candidates_threshold"],
        "correct_rust_chance_spectra": search["correct_rust_chance_spectra"],
        "lazy_interior_chance": search["lazy_interior_chance"],
        "exact_budget_sh": search["exact_budget_sh"],
        "exact_budget_sh_min_n": search["exact_budget_sh_min_n"],
        "belief_chance_spectra": search["belief_chance_spectra"],
        "information_set_search": search["information_set_search"],
        "native_mcts_hot_loop": bool(
            generation.get("native_mcts_hot_loop", False)
        ),
        "determinization_particles": search["determinization_particles"],
        "determinization_min_simulations": search[
            "determinization_min_simulations"
        ],
        "max_depth": search["max_depth"],
        "prior_temperature": evaluator["prior_temperature"],
        "value_scale": evaluator["value_scale"],
        "value_readout": evaluator["value_readout"],
        "public_observation": evaluator["public_observation"],
        "rust_featurize": evaluator["rust_featurize"],
        "eval_cache_size": evaluator["cache_size"],
        "track": generation["track"],
        "vps_to_win": generation["vps_to_win"],
        "obs_width": generation["obs_width"],
        "max_decisions": generation["max_decisions"],
        "temperature_decisions": generation["temperature_decisions"],
        "temperature_decisions_effective": generation["temperature_decisions"],
        "temperature_move_fraction": float(generation["temperature_decisions"])
        / float(generation["max_decisions"]),
        "temperature_high": generation["temperature_high"],
        "temperature_low": generation["temperature_low"],
        "late_temperature_decisions": generation["late_temperature_decisions"],
        "late_temperature_move_fraction": (
            None
            if late_decisions is None
            else float(late_decisions) / float(generation["max_decisions"])
        ),
        "late_temperature": generation["late_temperature"],
        "base_seed": int(job["base_seed"]),
        "shard_size": generation["shard_size"],
        "format": generation["format"],
        "eval_server": generation["eval_server"],
        "opponent_pool_manifest": None,
        "exploiter_fraction": None,
        "seed_claim": True,
        "ledger_claim_label": job["claim_label"],
        "skip_guards": False,
    }


def _expected_generate_config_provenance(
    lock: dict[str, Any],
    job: dict[str, Any],
    *,
    opponent_mix_manifest: str | None,
) -> dict[str, Any]:
    """Reconstruct the exact typed GenerateConfig the sealed argv must produce."""

    values = _expected_cli_fields(lock, job)
    values["opponent_mix_manifest"] = opponent_mix_manifest
    values["producer_checkpoint_sha256"] = _producer(lock)["sha256"]
    config = GenerateConfig.from_namespace(argparse.Namespace(**values))
    provenance = {
        "pipeline": "generate",
        "config_hash": config.config_hash(),
        "full_config_hash": config.full_config_hash(),
        "config": config.canonical_payload(),
    }
    provenance["provenance_sha256"] = _digest_value(provenance)
    return provenance


def _expected_selfplay_config(lock: dict[str, Any]) -> dict[str, Any]:
    generation = lock["generation"]
    search = lock["science"]["search_operator"]
    opening_fraction = float(generation["temperature_decisions"]) / float(
        generation["max_decisions"]
    )
    late = generation["late_temperature_decisions"]
    late_fraction = (
        None if late is None else float(late) / float(generation["max_decisions"])
    )
    effective = dataclasses.asdict(
        GumbelSelfPlayConfig(
            colors=("RED", "BLUE"),
            track=str(generation["track"]),
            vps_to_win=int(generation["vps_to_win"]),
            obs_width=int(generation["obs_width"]),
            max_decisions=int(generation["max_decisions"]),
            temperature_move_fraction=opening_fraction,
            temperature_high=float(generation["temperature_high"]),
            temperature_low=float(generation["temperature_low"]),
            late_temperature_move_fraction=late_fraction,
            late_temperature=float(generation["late_temperature"]),
            correct_rust_chance_spectra=bool(search["correct_rust_chance_spectra"]),
        )
    )
    # Match JSON-loaded worker manifests (tuples become lists).
    return json.loads(json.dumps(effective))


def _validate_selected_opponent_rows(
    payload: Any,
    *,
    selected_mask: np.ndarray,
    game_seeds: np.ndarray,
    job: Mapping[str, Any],
    allowed_versions: set[int],
    colors: Sequence[str],
) -> None:
    """Prove an opponent shard contains only current-producer decisions.

    Opponent games deliberately advance both seats through one shared game,
    but only the promoted producer's deterministic seat is a policy teacher.
    A category tag/checkpoint hash alone cannot prove that filtering happened:
    an unfiltered shard would carry the same game-level opponent identity.  Bind
    the selected rows to the runtime's deterministic seat assignment and to the
    exact sealed opponent version before they are admitted to training.
    """

    required = {"is_pool_game", "opponent_version", "player", "seat"}
    missing = sorted(required.difference(payload.files))
    if missing:
        raise ContractError(
            "selected opponent rows lack producer-seat provenance columns: "
            f"{missing}"
        )
    if len(colors) != 2 or any(color not in PLAYER_NAMES for color in colors):
        raise ContractError("sealed opponent rows have invalid two-player color order")
    pool_raw = np.asarray(payload["is_pool_game"])
    version_raw = np.asarray(payload["opponent_version"])
    player_raw = np.asarray(payload["player"])
    seat_raw = np.asarray(payload["seat"])
    if not (
        pool_raw.shape
        == version_raw.shape
        == player_raw.shape
        == seat_raw.shape
        == game_seeds.shape
    ):
        raise ContractError("opponent producer-seat provenance arrays are not row-aligned")
    if pool_raw.dtype.kind != "b" or not np.all(pool_raw[selected_mask]):
        raise ContractError("selected opponent rows are not all authenticated pool games")
    if version_raw.dtype.kind not in {"i", "u"} or not allowed_versions:
        raise ContractError("selected opponent version provenance is invalid")
    selected_versions = set(
        map(int, np.asarray(version_raw[selected_mask], dtype=np.int64).tolist())
    )
    if not selected_versions.issubset(allowed_versions):
        raise ContractError(
            "selected opponent rows bind an unsealed opponent version: "
            f"observed={sorted(selected_versions)} allowed={sorted(allowed_versions)}"
        )
    selected_seeds = np.asarray(game_seeds[selected_mask], dtype=np.int64)
    expected_players = np.asarray(
        [
            colors[0]
            if _pool_champion_plays_first_seat(
                int(seed) - int(job["base_seed"])
            )
            else colors[1]
            for seed in selected_seeds
        ]
    )
    selected_players = np.asarray(player_raw[selected_mask]).astype(str)
    if not np.array_equal(selected_players, expected_players):
        raise ContractError(
            "selected opponent rows include non-producer-seat policy targets"
        )
    if seat_raw.dtype.kind not in {"i", "u"}:
        raise ContractError("selected opponent seat provenance is not integral")
    expected_seats = np.asarray(
        [PLAYER_NAMES.index(str(player)) for player in expected_players],
        dtype=np.int64,
    )
    if not np.array_equal(
        np.asarray(seat_raw[selected_mask], dtype=np.int64), expected_seats
    ):
        raise ContractError("selected opponent player/seat provenance disagrees")


def _expected_public_award_feature_provenance(
    *, rust_featurize: bool
) -> dict[str, Any]:
    """Return the one producer record authorized by the sealed feature path.

    The generator owns the provenance schema/contract constants.  The model
    owns the checkpoint-side contract constant.  Requiring those two existing
    authorities to agree prevents this audit from silently inventing a third
    interpretation of player-token slot 12.
    """

    if (
        generation_cli.PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
        != PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    ):
        raise ContractError(
            "generator/model public-award authoritative contracts disagree"
        )
    return {
        "schema_version": generation_cli.PUBLIC_AWARD_FEATURE_PROVENANCE_SCHEMA,
        "contract": PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
        "feature_producer": (
            "catanatron_rs_public_award_v1"
            if rust_featurize
            else "python_snapshot_public_award_v1"
        ),
        "native_capability": (
            "public_award_feature_parity" if rust_featurize else None
        ),
    }


def _require_public_award_feature_provenance(
    raw: object,
    *,
    rust_featurize: bool,
    where: str,
) -> dict[str, Any]:
    expected = _expected_public_award_feature_provenance(
        rust_featurize=rust_featurize
    )
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        raise ContractError(
            f"{where}: public_award_feature_provenance drift; "
            f"observed={raw!r} expected={expected!r}"
        )
    return expected


def _authenticated_empty_event_authority() -> dict[str, Any]:
    """Resolve the checked-in adapter/native empty-history authorities.

    This intentionally fails when either existing registry advances to a
    public-event implementation.  Such a rollout must issue a new generation
    contract instead of letting the old post-wave acceptance path reinterpret
    new event tensors as the authenticated-empty corpus it was written for.
    """

    adapter_version = require_known_entity_feature_adapter(
        CURRENT_RUST_ENTITY_ADAPTER_VERSION
    )
    adapter_spec = ENTITY_FEATURE_ADAPTER_SPECS[adapter_version]
    native = information_surface.native_inference_event_history_capability()
    if adapter_spec.event_history != "empty" or native.get("available") is not False:
        raise ContractError(
            "post-wave authenticated-empty event audit is incompatible with the "
            "current adapter/native event-history authority"
        )
    return {
        "entity_feature_adapter_version": adapter_version,
        "adapter_event_history_semantic": adapter_spec.event_history,
        "native_inference": native,
    }


def _require_shard_feature_semantics(
    payload: Any,
    *,
    rows: int,
    where: str,
) -> dict[str, Any]:
    """Authenticate adapter identity and the current constant-empty events.

    All attempts are checked, including reserve/truncated games.  That matters
    because the audit publishes every shard as accepted source material even
    though only a deterministic game subset is selected for this learner.
    """

    authority = _authenticated_empty_event_authority()
    required = {"adapter_version", "event_tokens", "event_mask", "event_target_ids"}
    missing = sorted(required.difference(payload.files))
    if missing:
        raise ContractError(
            f"{where}: missing feature-semantic columns {missing}"
        )

    adapters = np.asarray(payload["adapter_version"])
    if adapters.shape != (rows,):
        raise ContractError(f"{where}: adapter_version is not row-aligned")
    observed_adapters = set(adapters.astype(str).tolist())
    expected_adapter = str(authority["entity_feature_adapter_version"])
    if observed_adapters != {expected_adapter}:
        raise ContractError(
            f"{where}: adapter_version drift; observed={sorted(observed_adapters)} "
            f"expected={[expected_adapter]}"
        )

    event_mask = np.asarray(payload["event_mask"])
    if (
        event_mask.ndim != 2
        or event_mask.shape[0] != rows
        or event_mask.shape[1] <= 0
    ):
        raise ContractError(
            f"{where}: event_mask violates the entity schema: {event_mask.shape}"
        )
    if event_mask.dtype.kind != "b":
        raise ContractError(f"{where}: event_mask is not boolean")
    if np.any(event_mask):
        raise ContractError(f"{where}: authenticated-empty event_mask has live entries")
    empty_scan = {
        "schema": "training-empty-event-mask-scan-v1",
        "row_count": rows,
        "padded_event_width": int(event_mask.shape[1]),
        "nonzero_event_mask_count": 0,
    }
    empty_scan["scan_sha256"] = _digest_value(empty_scan)
    return {
        "row_count": rows,
        "entity_feature_adapter_version": expected_adapter,
        "event_history": {
            "authenticated_empty": True,
            "empty_event_mask_scan": empty_scan,
        },
    }


def audit_outputs(
    lock_path: Path,
    out_path: Path,
    *,
    harvest_relocation: Path | None = None,
    frozen_repo: Path | None = None,
    frozen_verifier_sha256: str | None = None,
) -> dict[str, Any]:
    """Deep post-wave audit.  This is callable only after generation; it never launches it."""
    if (frozen_repo is None) != (frozen_verifier_sha256 is None):
        raise ContractError(
            "post-wave audit requires --frozen-repo and "
            "--frozen-verifier-sha256 together"
        )
    try:
        lock_path = lock_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve contract lock {lock_path}: {error}") from error
    lock_verifier_authority: dict[str, Any] | None = None
    if frozen_repo is None:
        lock = verify_lock(lock_path, require_all_job_claims=True)
    else:
        assert frozen_verifier_sha256 is not None
        try:
            lock, lock_verifier_authority = frozen_lock_verifier.verify_frozen_lock(
                lock_path,
                frozen_repo=frozen_repo,
                expected_verifier_sha256=frozen_verifier_sha256,
                require_all_job_claims=True,
            )
        except frozen_lock_verifier.FrozenVerifierError as error:
            raise ContractError(
                f"frozen post-wave lock verification failed: {error}"
            ) from error
    game_contract = dict(lock["game_contract"])
    expected_games = {
        str(key): int(value) for key, value in game_contract["category_games"].items()
    }
    expected_attempts = {
        str(key): int(value)
        for key, value in game_contract["category_attempts"].items()
    }
    arm_id = game_contract.get("arm_id")
    relocation = (
        None
        if harvest_relocation is None
        else _load_harvest_relocation(
            harvest_relocation, lock=lock, lock_path=lock_path
        )
    )
    all_seeds: set[int] = set()
    category_seeds: dict[str, set[int]] = {name: set() for name in expected_games}
    rows_by_seed: Counter[int] = Counter()
    shard_records: list[dict[str, Any]] = []
    invalid_actions = 0
    rows = 0
    forced = 0
    full_active = 0
    target_entropy_sum = 0.0
    target_entropy_count = 0
    phases: Counter[str] = Counter()
    decision_bins: Counter[str] = Counter()
    legal_widths: Counter[str] = Counter()
    errors: list[str] = []
    seen_shards: set[Path] = set()
    job_selections: list[dict[str, Any]] = []
    selected_game_records: list[dict[str, Any]] = []
    target_information_regimes: Counter[str] = Counter()
    feature_semantic_rows = 0
    adapter_version_counts: Counter[str] = Counter()
    event_history_width_counts: Counter[int] = Counter()
    public_award_generation_manifests = 0
    public_award_worker_manifests = 0
    required_target_information_regime = str(
        lock["post_wave_acceptance"]["require_target_information_regime"]
    )
    producer = _producer(lock)
    checkpoint_by_id = {record["id"]: record for record in lock["checkpoints"]}
    category_specs = {item["name"]: item for item in lock["source_categories"]}
    selfplay_colors = tuple(_expected_selfplay_config(lock)["colors"])
    rust_featurize = bool(lock["science"]["evaluator"]["rust_featurize"])
    expected_public_award_provenance = _expected_public_award_feature_provenance(
        rust_featurize=rust_featurize
    )
    empty_event_authority = _authenticated_empty_event_authority()
    for job in lock["fleet"]["jobs"]:
        attestation_source = Path(job["output_dir"]) / "a1_contract.json"
        attestation_path = (
            attestation_source
            if relocation is None
            else relocation.resolve(attestation_source)
        )
        expected_attestation = _job_attestation(lock, job)
        if not attestation_path.is_file():
            errors.append(f"missing contract attestation: {attestation_path}")
        else:
            try:
                actual_attestation = _load_json(attestation_path)
                legacy_attestation = _legacy_job_attestation(lock, job)
                if actual_attestation not in (expected_attestation, legacy_attestation):
                    errors.append(f"{job['job_id']}: output contract attestation drift")
                shard_records.append(
                    {
                        "kind": "contract_attestation",
                        "path": str(attestation_path),
                        "sha256": _sha256(attestation_path),
                        "job_id": job["job_id"],
                        "category": job["category"],
                    }
                )
            except ContractError as error:
                errors.append(f"{job['job_id']}: {error}")
        manifest_source = Path(job["output_dir"]) / "manifest.json"
        manifest_path = (
            manifest_source
            if relocation is None
            else relocation.resolve(manifest_source)
        )
        if not manifest_path.is_file():
            errors.append(f"missing manifest: {manifest_path}")
            continue
        manifest = _load_json(manifest_path)
        try:
            _require_public_award_feature_provenance(
                manifest.get("public_award_feature_provenance"),
                rust_featurize=rust_featurize,
                where=str(job["job_id"]),
            )
        except ContractError as error:
            errors.append(str(error))
        else:
            public_award_generation_manifests += 1
        shard_records.append(
            {
                "kind": "generation_manifest",
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "job_id": job["job_id"],
                "arm_id": job.get("arm_id"),
                "category": job["category"],
                "producer_checkpoint_sha256": producer["sha256"],
                "opponent_checkpoint_sha256": _category_opponent_sha256(
                    lock, job["category"]
                ),
                "public_award_feature_provenance": manifest.get(
                    "public_award_feature_provenance"
                ),
            }
        )
        if int(manifest.get("games_requested", -1)) != int(job["attempts"]):
            errors.append(f"{job['job_id']}: games_requested drift")
        if int(manifest.get("games_completed", -1)) != int(job["attempts"]):
            errors.append(f"{job['job_id']}: incomplete attempts")
        if int(manifest.get("games_failed", -1)) != 0 or manifest.get("errors"):
            errors.append(f"{job['job_id']}: failures/errors present")
        manifest_truncated = int(manifest.get("games_truncated", -1))
        if manifest_truncated < 0:
            errors.append(f"{job['job_id']}: invalid games_truncated")
        if int(manifest.get("base_seed", -1)) != int(job["base_seed"]):
            errors.append(f"{job['job_id']}: base_seed drift")
        if str(manifest.get("checkpoint")) != producer["path"]:
            errors.append(f"{job['job_id']}: producer checkpoint path drift")
        if manifest.get("target_information_regime") != required_target_information_regime:
            errors.append(
                f"{job['job_id']}: generation manifest target_information_regime="
                f"{manifest.get('target_information_regime')!r}, expected "
                f"{required_target_information_regime!r}"
            )
        cli = dict(manifest.get("cli_args", {}))
        for key, expected in _expected_cli_fields(lock, job).items():
            if cli.get(key) != expected:
                errors.append(
                    f"{job['job_id']}: cli_args.{key}={cli.get(key)!r}, expected {expected!r}"
                )
        actual_config_provenance: dict[str, Any] | None = None
        try:
            actual_config = GenerateConfig.from_namespace(argparse.Namespace(**cli))
            actual_config_hash = actual_config.config_hash()
            actual_config_provenance = {
                "pipeline": "generate",
                "config_hash": actual_config_hash,
                "full_config_hash": actual_config.full_config_hash(),
                "config": actual_config.canonical_payload(),
            }
        except Exception as error:  # noqa: BLE001 - malformed config provenance blocks ingest.
            errors.append(f"{job['job_id']}: cannot reconstruct config_hash: {error}")
        else:
            if manifest.get("config_hash") != actual_config_hash:
                errors.append(
                    f"{job['job_id']}: config_hash={manifest.get('config_hash')!r}, "
                    f"reconstructed={actual_config_hash!r}"
                )
            expected_config_provenance = _expected_generate_config_provenance(
                lock,
                job,
                opponent_mix_manifest=cli.get("opponent_mix_manifest"),
            )
            expected_config_provenance.pop("provenance_sha256")
            if actual_config_provenance != expected_config_provenance:
                errors.append(f"{job['job_id']}: full typed GenerateConfig drift")
        registry_source = Path(job["output_dir"]) / CONFIG_REGISTRY_FILENAME
        registry_path = (
            registry_source
            if relocation is None
            else relocation.resolve(registry_source)
        )
        try:
            registry_payload = _read_sealed_regular(
                registry_path, where=str(job["job_id"])
            )
            if actual_config_provenance is None:
                raise ContractError(
                    f"{job['job_id']}: cannot validate registry without typed config"
                )
            _strict_config_registry_record(
                registry_payload,
                expected=actual_config_provenance,
                where=str(job["job_id"]),
            )
        except ContractError as error:
            errors.append(str(error))
        else:
            shard_records.append(
                {
                    "kind": "config_registry",
                    "path": str(registry_path),
                    "sha256": _sha256_bytes(registry_payload),
                    "job_id": job["job_id"],
                    "category": job["category"],
                    "config_hash": manifest.get("config_hash"),
                    "full_config_hash": actual_config_provenance[
                        "full_config_hash"
                    ],
                }
            )
        worker_summaries = list(manifest.get("worker_summaries", []))
        expected_workers = min(
            int(lock["generation"]["workers_per_gpu"]), int(job["attempts"])
        )
        if len(worker_summaries) != expected_workers:
            errors.append(
                f"{job['job_id']}: worker_summaries={len(worker_summaries)}, "
                f"expected {expected_workers}"
            )
        for raw_worker_summary in worker_summaries:
            worker_manifest_path = (
                Path(raw_worker_summary)
                if relocation is None
                else relocation.resolve(
                    str(raw_worker_summary), owner_source=manifest_source
                )
            )
            if not worker_manifest_path.is_file():
                errors.append(
                    f"{job['job_id']}: missing worker manifest {worker_manifest_path}"
                )
                continue
            worker_manifest = _load_json(worker_manifest_path)
            try:
                _require_public_award_feature_provenance(
                    worker_manifest.get("public_award_feature_provenance"),
                    rust_featurize=rust_featurize,
                    where=str(worker_manifest_path),
                )
            except ContractError as error:
                errors.append(f"{job['job_id']}: {error}")
            else:
                public_award_worker_manifests += 1
            expected_adapter_version = empty_event_authority[
                "entity_feature_adapter_version"
            ]
            if worker_manifest.get("adapter_version") != expected_adapter_version:
                errors.append(
                    f"{job['job_id']}: worker adapter_version="
                    f"{worker_manifest.get('adapter_version')!r}, expected "
                    f"{expected_adapter_version!r}"
                )
            shard_records.append(
                {
                    "kind": "worker_generation_manifest",
                    "path": str(worker_manifest_path),
                    "sha256": _sha256(worker_manifest_path),
                    "job_id": job["job_id"],
                    "category": job["category"],
                    "entity_feature_adapter_version": worker_manifest.get(
                        "adapter_version"
                    ),
                    "public_award_feature_provenance": worker_manifest.get(
                        "public_award_feature_provenance"
                    ),
                }
            )
            if (
                worker_manifest.get("target_information_regime")
                != required_target_information_regime
            ):
                errors.append(
                    f"{job['job_id']}: worker manifest target_information_regime="
                    f"{worker_manifest.get('target_information_regime')!r}, expected "
                    f"{required_target_information_regime!r}"
                )
            actual_search = dict(worker_manifest.get("search_config", {}))
            actual_search.pop("seed", None)
            if actual_search != _job_search_identity(lock, job)[
                "effective_search_config"
            ]:
                errors.append(f"{job['job_id']}: worker effective search config drift")
            if worker_manifest.get("selfplay_config") != _expected_selfplay_config(
                lock
            ):
                errors.append(
                    f"{job['job_id']}: worker effective self-play config drift"
                )
        expected_category = category_specs[job["category"]]
        allowed_md5 = {
            checkpoint_by_id[checkpoint_id]["md5"]
            for checkpoint_id in expected_category["checkpoint_ids"]
        }
        mix_manifest_raw = cli.get("opponent_mix_manifest")
        if job["category"] == "current_producer":
            if mix_manifest_raw not in (None, ""):
                errors.append(
                    f"{job['job_id']}: current-producer job unexpectedly used a mix"
                )
        elif not mix_manifest_raw:
            errors.append(f"{job['job_id']}: missing category-specific opponent mix")
        else:
            try:
                mix_manifest_path = (
                    Path(str(mix_manifest_raw))
                    if relocation is None
                    else relocation.resolve(str(mix_manifest_raw))
                )
                mix_manifest = _load_json(mix_manifest_path)
                attestation = dict(mix_manifest.get("_a1_contract", {}))
                category_semantic = _sealed_category_semantic(
                    lock, str(job["category"])
                )
                expected_mix_attestation = {
                    "contract_sha256": lock["contract_sha256"],
                    "category": job["category"],
                    **(
                        {}
                        if category_semantic is None
                        else {"category_semantic": category_semantic}
                    ),
                }
                if attestation != expected_mix_attestation:
                    errors.append(
                        f"{job['job_id']}: opponent-mix contract attestation drift"
                    )
            except ContractError as error:
                errors.append(f"{job['job_id']}: {error}")
        # Pass 1 inventories every attempt and hashes every shard.  Selection is
        # game-level and deterministic; no reserve/truncated row contributes to
        # metrics, the holdout, or the accepted shard-row inventory below.
        job_shards: list[Path] = []
        seed_status: dict[int, tuple[bool, bool]] = {}
        active_seed_run: int | None = None
        active_decision_run: int | None = None
        closed_seed_runs: set[int] = set()
        for raw_shard in manifest.get("shards", []):
            try:
                shard = _resolve_shard(
                    manifest_path,
                    str(raw_shard),
                    relocation=relocation,
                    manifest_source=manifest_source,
                )
                canonical_shard = shard.resolve(strict=True)
                if canonical_shard in seen_shards:
                    raise ContractError(f"duplicate shard reference {canonical_shard}")
                seen_shards.add(canonical_shard)
                job_shards.append(canonical_shard)
                data_shard_record = {
                        "kind": "data_shard",
                        "path": str(canonical_shard),
                        "sha256": _sha256(canonical_shard),
                        "job_id": job["job_id"],
                        "category": job["category"],
                        "producer_checkpoint_sha256": producer["sha256"],
                        "opponent_checkpoint_sha256": _category_opponent_sha256(
                            lock, job["category"]
                        ),
                        "search_operator_sha256": _job_search_identity(lock, job)[
                            "search_operator_sha256"
                        ],
                        "effective_search_config_sha256": _job_search_identity(
                            lock, job
                        )["effective_search_config_sha256"],
                        "evaluator_sha256": lock["science"]["evaluator_sha256"],
                        "entity_feature_adapter_version": empty_event_authority[
                            "entity_feature_adapter_version"
                        ],
                        "public_award_feature_contract": (
                            PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
                        ),
                        "event_history_semantic": empty_event_authority[
                            "adapter_event_history_semantic"
                        ],
                    }
                shard_records.append(data_shard_record)
                with np.load(canonical_shard, allow_pickle=False) as payload:
                    game_seeds = np.asarray(payload["game_seed"], dtype=np.int64)
                    decision_indices = np.asarray(
                        payload["decision_index"], dtype=np.int64
                    )
                    regimes = np.asarray(payload["target_information_regime"]).astype(str)
                    terminated = np.asarray(payload["terminated"], dtype=bool)
                    truncated = np.asarray(payload["truncated"], dtype=bool)
                    if game_seeds.ndim != 1 or not (
                        regimes.shape
                        == terminated.shape
                        == truncated.shape
                        == decision_indices.shape
                        == game_seeds.shape
                    ):
                        raise ContractError(
                            "game status/target-information/decision arrays are not "
                            "row-aligned"
                        )
                    feature_semantics = _require_shard_feature_semantics(
                        payload,
                        rows=int(game_seeds.size),
                        where=f"{job['job_id']}:{canonical_shard.name}",
                    )
                    feature_semantic_rows += int(feature_semantics["row_count"])
                    adapter_version_counts.update(
                        [str(feature_semantics["entity_feature_adapter_version"])]
                        * int(feature_semantics["row_count"])
                    )
                    event_history_width_counts[
                        int(
                            feature_semantics["event_history"][
                                "empty_event_mask_scan"
                            ]["padded_event_width"]
                        )
                    ] += int(feature_semantics["row_count"])
                    data_shard_record["feature_semantics_sha256"] = _digest_value(
                        feature_semantics
                    )
                    target_information_regimes.update(regimes.tolist())
                    if np.any(regimes != required_target_information_regime):
                        actual = sorted(set(regimes.tolist()))
                        errors.append(
                            f"{job['job_id']}: shard target_information_regime values "
                            f"{actual}, expected only {required_target_information_regime!r}"
                        )
                    active_decision_run = _advance_game_decision_run(
                        game_seeds,
                        decision_indices,
                        active_seed=active_seed_run,
                        active_decision_index=active_decision_run,
                        where=job["job_id"],
                    )
                    active_seed_run = _advance_game_seed_runs(
                        game_seeds,
                        active_seed=active_seed_run,
                        closed_seeds=closed_seed_runs,
                        where=job["job_id"],
                    )
                    for seed in np.unique(game_seeds):
                        seed_int = int(seed)
                        if not int(job["base_seed"]) <= seed_int < int(job["seed_end"]):
                            errors.append(
                                f"{job['job_id']}: out-of-range seed {seed_int}"
                            )
                        mask = game_seeds == seed
                        statuses = set(
                            zip(
                                map(bool, terminated[mask].tolist()),
                                map(bool, truncated[mask].tolist()),
                            )
                        )
                        if len(statuses) != 1:
                            errors.append(
                                f"{job['job_id']}: inconsistent row status for seed {seed_int}"
                            )
                            continue
                        status = next(iter(statuses))
                        prior = seed_status.get(seed_int)
                        if prior is not None and prior != status:
                            errors.append(
                                f"{job['job_id']}: cross-shard status drift for seed {seed_int}"
                            )
                        seed_status[seed_int] = status
            except (ContractError, KeyError, OSError, ValueError) as error:
                errors.append(f"{job['job_id']}: {error}")

        expected_attempt_seeds = set(
            range(int(job["base_seed"]), int(job["seed_end"]))
        )
        observed_attempt_seeds = set(seed_status)
        if observed_attempt_seeds != expected_attempt_seeds:
            errors.append(
                f"{job['job_id']}: attempted seed set drift; "
                f"missing={len(expected_attempt_seeds - observed_attempt_seeds)}, "
                f"extra={len(observed_attempt_seeds - expected_attempt_seeds)}"
            )
        observed_truncated = sum(
            1 for terminated, truncated in seed_status.values() if truncated or not terminated
        )
        if manifest_truncated != observed_truncated:
            errors.append(
                f"{job['job_id']}: games_truncated={manifest_truncated}, "
                f"row evidence={observed_truncated}"
            )
        complete = sorted(
            seed
            for seed, (terminated, truncated) in seed_status.items()
            if terminated and not truncated
        )
        selected = set(complete[: int(job["games"])])
        if len(selected) != int(job["games"]):
            errors.append(
                f"{job['job_id']}: only {len(complete)} complete attempts for "
                f"selected quota {job['games']}"
            )
        category_seeds[job["category"]].update(selected)
        category_semantic = _sealed_category_semantic(lock, str(job["category"]))
        selected_game_records.extend(
            {
                "game_seed": int(seed),
                "job_id": job["job_id"],
                "worker_id": job["worker_id"],
                "category": job["category"],
                **(
                    {}
                    if category_semantic is None
                    else {"category_semantic": category_semantic}
                ),
                "producer_checkpoint_sha256": producer["sha256"],
                "opponent_checkpoint_sha256": _category_opponent_sha256(
                    lock, job["category"]
                ),
            }
            for seed in sorted(selected)
        )
        job_selections.append(
            {
                "job_id": job["job_id"],
                "category": job["category"],
                **(
                    {}
                    if category_semantic is None
                    else {"category_semantic": category_semantic}
                ),
                "attempts": int(job["attempts"]),
                "complete_attempts": len(complete),
                "truncated_attempts": observed_truncated,
                "selected_games": len(selected),
                "selected_seed_sha256": _digest_value(sorted(selected)),
            }
        )

        # Pass 2 computes all acceptance metrics from selected complete games
        # only.  Reserve rows remain hashed in shard_records but are excluded.
        for shard in job_shards:
            try:
                with np.load(shard, allow_pickle=False) as payload:
                    game_seeds = np.asarray(payload["game_seed"], dtype=np.int64)
                    selected_mask = np.isin(
                        game_seeds, np.asarray(sorted(selected), dtype=np.int64)
                    )
                    selected_indices = np.flatnonzero(selected_mask)
                    n = int(selected_indices.size)
                    if n == 0:
                        continue
                    rows += n
                    selected_seeds = game_seeds[selected_mask]
                    rows_by_seed.update(map(int, selected_seeds.tolist()))
                    actions = np.asarray(payload["action_taken"])[selected_mask]
                    legal_ids = np.asarray(payload["legal_action_ids"])[selected_mask]
                    raw_legal_mask = payload.get("legal_action_mask")
                    legal_mask = (
                        None
                        if raw_legal_mask is None
                        else np.asarray(raw_legal_mask)[selected_mask]
                    )
                    for index in range(n):
                        mask = None if legal_mask is None else legal_mask[index]
                        invalid_actions += int(
                            not _row_legal(int(actions[index]), legal_ids[index], mask)
                        )
                    if np.any(np.asarray(payload["truncated"], dtype=bool)[selected_mask]):
                        errors.append(f"{job['job_id']}: selected truncation leaked")
                    if not np.all(
                        np.asarray(payload["terminated"], dtype=bool)[selected_mask]
                    ):
                        errors.append(f"{job['job_id']}: selected incomplete game leaked")
                    (
                        is_forced,
                        used_full,
                        phase,
                        decision,
                        policy,
                        policy_mask,
                    ) = _selected_telemetry_arrays(
                        payload,
                        game_seeds=game_seeds,
                        selected_mask=selected_mask,
                        max_decisions=int(lock["generation"]["max_decisions"]),
                        where=job["job_id"],
                    )
                    forced += int(is_forced.sum())
                    full_active += int(np.sum(used_full & ~is_forced))
                    phases.update(phase.tolist())
                    for value in decision:
                        key = (
                            f"{(int(value) // 25) * 25:03d}-"
                            f"{(int(value) // 25) * 25 + 24:03d}"
                        )
                        decision_bins[key] += 1
                    for index in range(n):
                        probs = policy[index][policy_mask[index]]
                        probs = probs[np.isfinite(probs) & (probs > 0)]
                        legal_widths[str(int(probs.size))] += 1
                        total = float(probs.sum())
                        if total > 0:
                            normalized = probs / total
                            target_entropy_sum += float(
                                -np.sum(normalized * np.log(normalized))
                            )
                            target_entropy_count += 1
                    if job["category"] != "current_producer":
                        allowed_versions = {
                            int(checkpoint_by_id[checkpoint_id].get("version", -1))
                            for checkpoint_id in category_specs[job["category"]][
                                "checkpoint_ids"
                            ]
                        }
                        _validate_selected_opponent_rows(
                            payload,
                            selected_mask=selected_mask,
                            game_seeds=game_seeds,
                            job=job,
                            allowed_versions=allowed_versions,
                            colors=selfplay_colors,
                        )
                        tags = np.asarray(
                            payload.get(
                                "opponent_tag", np.full(game_seeds.size, "")
                            )
                        ).astype(str)[selected_mask]
                        md5s = np.asarray(
                            payload.get(
                                "opponent_checkpoint_md5",
                                np.full(game_seeds.size, ""),
                            )
                        ).astype(str)[selected_mask]
                        if np.any(tags != job["category"]):
                            errors.append(
                                f"{job['job_id']}: opponent source label drift"
                            )
                        if any(md5 not in allowed_md5 for md5 in md5s.tolist()):
                            errors.append(
                                f"{job['job_id']}: opponent checkpoint identity drift"
                            )
                    elif "opponent_tag" in payload:
                        tags = np.asarray(payload["opponent_tag"]).astype(str)[
                            selected_mask
                        ]
                        if np.any(tags != ""):
                            errors.append(
                                f"{job['job_id']}: current-producer source label drift"
                            )
            except (ContractError, KeyError, OSError, ValueError, IndexError) as error:
                errors.append(f"{job['job_id']}: {error}")
    for category, seeds in category_seeds.items():
        if len(seeds) != expected_games[category]:
            errors.append(
                f"{category}: unique complete games={len(seeds)}, expected {expected_games[category]}"
            )
        duplicate = all_seeds.intersection(seeds)
        if duplicate:
            errors.append(
                f"{category}: {len(duplicate)} game seeds overlap another category"
            )
        all_seeds.update(seeds)
    if invalid_actions:
        errors.append(f"invalid_teacher_actions={invalid_actions}, expected 0")
    if any(
        _ranges_overlap((seed, seed + 1), VAL_ONLY_SEED_RANGE) for seed in all_seeds
    ):
        errors.append("selected corpus overlaps VAL-ONLY seeds")
    record_seeds = [int(record["game_seed"]) for record in selected_game_records]
    if len(record_seeds) != len(all_seeds) or set(record_seeds) != all_seeds:
        errors.append("selected game/source records do not bijectively cover selected seeds")
    expected_worker_manifest_count = sum(
        min(int(lock["generation"]["workers_per_gpu"]), int(job["attempts"]))
        for job in lock["fleet"]["jobs"]
    )
    if public_award_generation_manifests != len(lock["fleet"]["jobs"]):
        errors.append(
            "public-award provenance did not authenticate every generation manifest"
        )
    if public_award_worker_manifests != expected_worker_manifest_count:
        errors.append(
            "public-award provenance did not authenticate every worker manifest"
        )
    if len(event_history_width_counts) > 1:
        errors.append(
            "authenticated-empty event history uses mixed padded widths: "
            f"{dict(event_history_width_counts)}"
        )
    validation_contract = lock["post_wave_acceptance"]["validation_holdout"]
    validation_seed_manifest_path = out_path.with_suffix(".validation_seeds.json")
    selected_game_manifest_path = out_path.with_suffix(".selected_games.json")
    validation_manifest: dict[str, Any] | None = None
    selected_game_manifest: dict[str, Any] | None = None
    if not errors:
        unique_seeds = np.asarray(sorted(all_seeds), dtype=np.int64)
        rng = np.random.default_rng(int(validation_contract["validation_seed"]))
        shuffled = rng.permutation(unique_seeds)
        target_rows = max(
            1, int(round(rows * float(validation_contract["validation_fraction"])))
        )
        selected: list[int] = []
        selected_rows = 0
        for seed in shuffled:
            selected.append(int(seed))
            selected_rows += int(rows_by_seed[int(seed)])
            if selected_rows >= target_rows:
                break
        held_out = np.asarray(sorted(selected), dtype="<i8")
        seed_set_digest = "sha256:" + hashlib.sha256(held_out.tobytes()).hexdigest()
        validation_manifest = {
            "schema_version": "train-validation-game-seeds-v1",
            "a1_contract_sha256": lock["contract_sha256"],
            "validation_fraction": float(validation_contract["validation_fraction"]),
            "validation_seed": int(validation_contract["validation_seed"]),
            "validation_max_samples": int(
                validation_contract["validation_max_samples"]
            ),
            "validation_game_seed_ranges": [],
            "validation_game_seed_count": int(held_out.size),
            "validation_row_count": int(selected_rows),
            "validation_game_seed_set_sha256": seed_set_digest,
            "game_seeds": held_out.tolist(),
        }
        held_out_set = set(map(int, held_out.tolist()))
        selected_records = [
            {**record, "split": "validation" if record["game_seed"] in held_out_set else "train"}
            for record in sorted(
                selected_game_records,
                key=lambda item: (int(item["game_seed"]), str(item["job_id"])),
            )
        ]
        selected_seed_array = np.asarray(
            [record["game_seed"] for record in selected_records], dtype="<i8"
        )
        training_seed_array = np.asarray(
            [
                record["game_seed"]
                for record in selected_records
                if record["split"] == "train"
            ],
            dtype="<i8",
        )
        dual_subset_id = (
            None
            if arm_id is None
            else ("full-140k" if arm_id == "n128" else "full-56k")
        )
        selected_game_manifest = {
            "schema_version": (
                "a1-selected-training-games-v1"
                if arm_id is None
                else DUAL_ARM_SELECTED_GAMES_SCHEMA
            ),
            "a1_contract_sha256": lock["contract_sha256"],
            "selection_rule": "lowest_seed_complete_per_job",
            "selected_game_count": len(selected_records),
            "selected_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(selected_seed_array.tobytes()).hexdigest(),
            "category_game_counts": dict(expected_games),
            **(
                {}
                if arm_id is None
                else {
                    "arm_id": arm_id,
                    "subset_id": dual_subset_id,
                    # A full-arm audit is the root selection artifact.  Derived
                    # comparison subsets bind this file's SHA-256 here instead.
                    "parent_manifest_sha256": None,
                }
            ),
            "training_game_count": int(training_seed_array.size),
            "training_game_seed_set_sha256": "sha256:"
            + hashlib.sha256(training_seed_array.tobytes()).hexdigest(),
            "validation_game_count": int(held_out.size),
            "validation_game_seed_set_sha256": seed_set_digest,
            "records": (
                selected_records
                if arm_id is None
                else [{**record, "arm_id": arm_id} for record in selected_records]
            ),
            "records_sha256": _digest_value(
                selected_records
                if arm_id is None
                else [{**record, "arm_id": arm_id} for record in selected_records]
            ),
        }
    adapter_version = str(empty_event_authority["entity_feature_adapter_version"])
    padded_event_width = (
        next(iter(event_history_width_counts))
        if len(event_history_width_counts) == 1
        else None
    )
    aggregate_empty_event_scan: dict[str, Any] = {
        "schema": "training-empty-event-mask-scan-v1",
        "row_count": feature_semantic_rows,
        "padded_event_width": padded_event_width,
        "nonzero_event_mask_count": 0,
    }
    aggregate_empty_event_scan["scan_sha256"] = _digest_value(
        aggregate_empty_event_scan
    )
    feature_semantics_report: dict[str, Any] = {
        "public_award_feature_provenance": {
            "expected": expected_public_award_provenance,
            "generation_manifests_authenticated": (
                public_award_generation_manifests
            ),
            "worker_manifests_authenticated": public_award_worker_manifests,
        },
        "entity_feature_adapter": {
            "version": adapter_version,
            "base_adapter_spec": dataclasses.asdict(
                ENTITY_FEATURE_ADAPTER_SPECS[adapter_version]
            ),
            "row_counts": dict(adapter_version_counts),
        },
        "event_history": {
            **empty_event_authority,
            "authenticated_empty": True,
            "row_count": feature_semantic_rows,
            "history_width_row_counts": {
                str(width): count
                for width, count in sorted(event_history_width_counts.items())
            },
            "empty_event_mask_scan": aggregate_empty_event_scan,
        },
    }
    feature_semantics_report["feature_semantics_sha256"] = _digest_value(
        feature_semantics_report
    )
    report: dict[str, Any] = {
        "schema_version": (
            DUAL_ARM_AUDIT_SCHEMA
            if arm_id is not None
            else (AUDIT_SCHEMA if relocation is None else RELOCATED_AUDIT_SCHEMA)
        ),
        **(
            {}
            if arm_id is None
            else {
                "arm_id": arm_id,
                "subset_id": "full-140k" if arm_id == "n128" else "full-56k",
            }
        ),
        "contract_path": str(lock_path.absolute()),
        "contract_sha256": lock["contract_sha256"],
        **(
            {}
            if lock_verifier_authority is None
            else {"lock_verifier_authority": lock_verifier_authority}
        ),
        "passed": not errors,
        "errors": errors,
        "games": {category: len(seeds) for category, seeds in category_seeds.items()},
        "attempts": dict(expected_attempts),
        **({} if arm_id is None else {"category_game_counts": dict(expected_games)}),
        "total_unique_games": len(all_seeds),
        "selection_rule": "lowest_seed_complete_per_job",
        "job_selections": job_selections,
        "job_selection_sha256": _digest_value(job_selections),
        "rows": rows,
        "invalid_teacher_actions": invalid_actions,
        "target_information_regime": {
            "required": required_target_information_regime,
            "counts": dict(target_information_regimes),
        },
        "feature_semantics": feature_semantics_report,
        "reports": {
            "truncation": {
                "selected_truncated_games": 0,
                "reserve_truncated_or_incomplete_attempts": sum(
                    int(item["truncated_attempts"]) for item in job_selections
                ),
            },
            "forced_fraction": forced / rows if rows else None,
            "phase_mix": dict(phases),
            "decision_index_mix": dict(decision_bins),
            "legal_width": dict(legal_widths),
            "target_entropy": target_entropy_sum / target_entropy_count
            if target_entropy_count
            else None,
            "full_search_policy_mass": full_active / rows if rows else None,
        },
        "shards": shard_records,
        "shard_inventory_sha256": _digest_value(shard_records),
        "source_provenance": {
            category: {
                **(
                    {}
                    if _sealed_category_semantic(lock, category) is None
                    else {
                        "category_semantic": _sealed_category_semantic(
                            lock, category
                        )
                    }
                ),
                "producer_checkpoint_sha256": producer["sha256"],
                "opponent_checkpoint_sha256": _category_opponent_sha256(
                    lock, category
                ),
                "search_operator_sha256": _category_search_identities(
                    lock["science"]["search_operator"], lock["fleet"]["jobs"]
                )[category]["search_operator_sha256"],
                "effective_search_config_sha256": _category_search_identities(
                    lock["science"]["search_operator"], lock["fleet"]["jobs"]
                )[category]["effective_search_config_sha256"],
                "evaluator_sha256": lock["science"]["evaluator_sha256"],
                "public_award_feature_provenance": (
                    expected_public_award_provenance
                ),
                "entity_feature_adapter_version": adapter_version,
                "event_history_semantic": empty_event_authority[
                    "adapter_event_history_semantic"
                ],
            }
            for category in expected_games
        },
        **(
            {}
            if lock.get("category_semantics") is None
            else {"category_semantics": lock["category_semantics"]}
        ),
        **(
            {}
            if relocation is None
            else {
                "harvest_relocation": {
                    **({} if arm_id is None else {"arm_id": arm_id}),
                    "path": str(relocation.path),
                    "file_sha256": _sha256(relocation.path),
                    "relocation_sha256": relocation.payload["relocation_sha256"],
                    "render_sha256": relocation.payload["render_sha256"],
                    "job_identities_sha256": relocation.payload[
                        "job_identities_sha256"
                    ],
                    "file_inventory_sha256": relocation.payload[
                        "file_inventory_sha256"
                    ],
                }
            }
        ),
        "selected_training_games": (
            {
                "manifest": str(selected_game_manifest_path),
                "manifest_sha256": _digest_value(selected_game_manifest),
                "selected_game_count": selected_game_manifest[
                    "selected_game_count"
                ],
                "selected_game_seed_set_sha256": selected_game_manifest[
                    "selected_game_seed_set_sha256"
                ],
                "records_sha256": selected_game_manifest["records_sha256"],
            }
            if selected_game_manifest is not None
            else None
        ),
        "validation_holdout": (
            {
                "manifest": str(validation_seed_manifest_path),
                "manifest_sha256": _digest_value(validation_manifest),
                "validation_game_seed_count": validation_manifest[
                    "validation_game_seed_count"
                ],
                "validation_game_seed_set_sha256": validation_manifest[
                    "validation_game_seed_set_sha256"
                ],
            }
            if validation_manifest is not None
            else None
        ),
    }
    report["audit_sha256"] = _digest_value(report)
    if validation_manifest is not None:
        _create_or_verify_readonly(validation_seed_manifest_path, validation_manifest)
        assert selected_game_manifest is not None
        _create_or_verify_readonly(selected_game_manifest_path, selected_game_manifest)
        report["validation_holdout"]["manifest_file_sha256"] = _sha256(
            validation_seed_manifest_path
        )
        report["selected_training_games"]["manifest_file_sha256"] = _sha256(
            selected_game_manifest_path
        )
        # The on-disk file digest is added after writing; refresh the report
        # digest so the immutable report binds the exact sidecar bytes too.
        report["audit_sha256"] = _digest_value(
            {key: value for key, value in report.items() if key != "audit_sha256"}
        )
    _create_or_verify_readonly(out_path.absolute(), report)
    if errors:
        raise ContractError(
            f"post-wave audit failed with {len(errors)} error(s); see {out_path}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser(
        "inspect-template", help="report unresolved fields; never seals"
    )
    inspect_parser.add_argument("--draft", required=True)
    campaign_parser = sub.add_parser(
        "verify-generation-campaign",
        help="verify the immutable dual-arm blueprint; never seals or launches",
    )
    campaign_parser.add_argument("--contract", required=True)
    campaign_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless promotion handoff and physical placement have been materialized",
    )
    revision_parser = sub.add_parser(
        "revise-generation-campaign",
        help="derive a fresh pending campaign from the issued r1 locks; never launches",
    )
    revision_parser.add_argument("--source", required=True)
    revision_parser.add_argument(
        "--superseded-lock", required=True, action="append"
    )
    revision_parser.add_argument("--contract-id", required=True)
    revision_parser.add_argument("--output-root", required=True)
    revision_parser.add_argument("--out", required=True)
    rebase_parser = sub.add_parser(
        "rebase-post-promotion-campaign",
        help="bind a fresh campaign to the committed promoted producer; never launches",
    )
    rebase_parser.add_argument("--source", required=True)
    rebase_parser.add_argument("--promotion-handoff", required=True)
    rebase_parser.add_argument("--contract-id", required=True)
    rebase_parser.add_argument("--output-root", required=True)
    rebase_parser.add_argument("--out", required=True)
    materialize_parser = sub.add_parser(
        "materialize-generation-campaign",
        help="seal both arm locks after handoff and exact placement; never launches",
    )
    materialize_parser.add_argument("--contract", required=True)
    materialize_parser.add_argument("--promotion-handoff", required=True)
    materialize_parser.add_argument("--placement", required=True)
    materialize_parser.add_argument("--out-dir", required=True)
    placement_parser = sub.add_parser(
        "seal-generation-placement",
        help="seal the exact 56 logical-lane to host/GPU assignments; never launches",
    )
    placement_parser.add_argument("--contract", required=True)
    placement_parser.add_argument("--assignments", required=True)
    placement_parser.add_argument("--out", required=True)
    sync_guard_parser = sub.add_parser(
        "sync-generation-guard",
        help=(
            "replay typed S1 and synchronize only the provenance-declared static "
            "generation guard; never seals or launches"
        ),
    )
    sync_guard_parser.add_argument("--draft", required=True)
    seal_parser = sub.add_parser(
        "seal", help="freeze a fully resolved draft exactly once"
    )
    seal_parser.add_argument("--draft", required=True)
    seal_parser.add_argument("--out", required=True)
    verify_parser = sub.add_parser(
        "verify", help="rehash and semantically verify a frozen lock"
    )
    verify_parser.add_argument("--lock", required=True)
    render_parser = sub.add_parser(
        "render", help="render immutable commands; never execute"
    )
    render_parser.add_argument("--lock", required=True)
    render_parser.add_argument("--out-dir", required=True)
    claim_parser = sub.add_parser(
        "claim", help="atomically append every exact rendered seed claim"
    )
    claim_parser.add_argument("--lock", required=True)
    claim_parser.add_argument("--render", required=True)
    claim_parser.add_argument("--receipt", required=True)
    audit_parser = sub.add_parser(
        "audit", help="deep post-wave audit before corpus ingest"
    )
    audit_parser.add_argument("--lock", required=True)
    audit_parser.add_argument("--out", required=True)
    audit_parser.add_argument(
        "--frozen-repo",
        help=(
            "exact historical checkout whose path-bound verifier authenticates "
            "the sealed lock"
        ),
    )
    audit_parser.add_argument(
        "--frozen-verifier-sha256",
        help="explicit SHA-256 of the frozen checkout's lock verifier",
    )
    audit_parser.add_argument(
        "--harvest-relocation",
        help=(
            "typed a1-fleet-harvest-relocation-v1 map for atomically "
            "consolidated fleet outputs"
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-template":
            draft = _load_json(Path(args.draft))
            unresolved = _find_unresolved(draft)
            print(
                json.dumps(
                    {
                        "schema_version": draft.get("schema_version"),
                        "unresolved": unresolved,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "verify-generation-campaign":
            campaign = validate_generation_campaign(
                Path(args.contract), require_ready=args.require_ready
            )
            print(
                json.dumps(
                    {
                        "status": campaign["status"],
                        "contract_sha256": campaign["contract_sha256"],
                        "launch_authorized": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "revise-generation-campaign":
            campaign = build_generation_campaign_revision(
                Path(args.source),
                superseded_lock_paths=[Path(path) for path in args.superseded_lock],
                contract_id=args.contract_id,
                output_root=Path(args.output_root),
                out_path=Path(args.out),
            )
            print(
                json.dumps(
                    {
                        "status": campaign["status"],
                        "contract_id": campaign["contract_id"],
                        "contract_sha256": campaign["contract_sha256"],
                        "launch_authorized": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "rebase-post-promotion-campaign":
            campaign = build_post_promotion_generation_campaign(
                Path(args.source),
                handoff_path=Path(args.promotion_handoff),
                contract_id=args.contract_id,
                output_root=Path(args.output_root),
                out_path=Path(args.out),
            )
            print(json.dumps({
                "status": campaign["status"],
                "contract_id": campaign["contract_id"],
                "contract_sha256": campaign["contract_sha256"],
                "producer": next(
                    row for row in campaign["checkpoints"] if row["role"] == "producer"
                ),
                "launch_authorized": False,
            }, sort_keys=True))
            return 0
        if args.command == "materialize-generation-campaign":
            locks = materialize_generation_campaign(
                Path(args.contract),
                promotion_handoff_path=Path(args.promotion_handoff),
                placement_path=Path(args.placement),
                out_dir=Path(args.out_dir),
            )
            print(json.dumps({"locks": list(map(str, locks))}, sort_keys=True))
            return 0
        if args.command == "seal-generation-placement":
            placement = seal_generation_placement(
                Path(args.contract), Path(args.assignments), Path(args.out)
            )
            print(
                json.dumps(
                    {"placement_sha256": placement["placement_sha256"]},
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "sync-generation-guard":
            print(
                json.dumps(
                    sync_generation_guard(Path(args.draft)), indent=2, sort_keys=True
                )
            )
            return 0
        if args.command == "seal":
            payload = build_lock(Path(args.draft))
            _create_readonly(Path(args.out).absolute(), payload)
            print(
                json.dumps(
                    {
                        "out": str(Path(args.out).absolute()),
                        "contract_sha256": payload["contract_sha256"],
                    }
                )
            )
            return 0
        if args.command == "verify":
            payload = verify_lock(Path(args.lock).absolute())
            print(
                json.dumps(
                    {"status": "PASS", "contract_sha256": payload["contract_sha256"]}
                )
            )
            return 0
        if args.command == "render":
            payload = render(Path(args.lock).absolute(), Path(args.out_dir))
            print(
                json.dumps(
                    {
                        "out": str(Path(args.out_dir).absolute()),
                        "jobs": len(payload["commands"]),
                    }
                )
            )
            return 0
        if args.command == "claim":
            payload = claim_seed_ledger(
                Path(args.lock), Path(args.render), Path(args.receipt)
            )
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "receipt": str(Path(args.receipt).absolute()),
                        "receipt_sha256": payload["receipt_sha256"],
                        "claim_count": payload["claim_count"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        payload = audit_outputs(
            Path(args.lock).absolute(),
            Path(args.out).absolute(),
            harvest_relocation=(
                None
                if args.harvest_relocation is None
                else Path(args.harvest_relocation).absolute()
            ),
            frozen_repo=(
                None if args.frozen_repo is None else Path(args.frozen_repo).absolute()
            ),
            frozen_verifier_sha256=args.frozen_verifier_sha256,
        )
        print(json.dumps({"status": "PASS", "audit_sha256": payload["audit_sha256"]}))
        return 0
    except ContractError as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
