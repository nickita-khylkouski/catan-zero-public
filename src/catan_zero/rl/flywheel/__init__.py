"""Reusable opponent-pool, checkpoint, and evaluation contracts.

Modules (each has a stdlib-only ``__main__`` self-test):
  - ``checkpoint_registry`` — candidate publish / gated champion promotion / opponent archive
  - ``opponent_pool``       — deterministic archived-opponent sampling (anti-forgetting)
  - ``opponent_mix``        — generalized N-way categorical opponent-mix sampling (CAT-54)
  - ``exploit_probe``       — adversary-vs-FROZEN-champion exploitability probe (R8/Wang, scaffolding)

The old all-in-one continuous-loop prototype has been removed. Production
orchestration lives in ``tools/loop.py``; the modules here remain because the
generator and corpus/training contracts import them directly.
"""
from __future__ import annotations

from .checkpoint_registry import (
    CandidateRef, ChampionRef, publish_candidate, read_candidate, read_champion,
    seed_champion, promote, list_archive, ensure_dirs,
)
from .opponent_pool import OpponentPolicy, OpponentChoice, choose_opponent, realized_pool_fraction
from .opponent_mix import (
    MixCategory, MixCheckpointRef, MixChoice, OpponentMixConfig,
    choose_mix_category, choose_checkpoint_in_category, choose_mix_opponent,
    realized_mix_fractions, read_opponent_mix_manifest, config_to_dict as opponent_mix_config_to_dict,
)
from .exploit_probe import (
    AdversaryNetSpec, SMALL_ADVERSARY, EXPLOIT_WIN_RATE_THRESHOLD, ExploitReport,
    frozen_opponent_manifest, write_frozen_opponent_manifest,
    adversary_only_rows, is_adversary_row,
    build_exploit_report, exploit_rate_from_pool_stats, exploit_rate_from_outcomes, exploit_verdict,
    file_sha256, param_fingerprint, assert_params_unchanged, adversary_optimizer_params,
    build_adversary_policy, save_fresh_adversary, compute_fraction,
)

__all__ = [
    "CandidateRef", "ChampionRef", "publish_candidate", "read_candidate", "read_champion",
    "seed_champion", "promote", "list_archive", "ensure_dirs",
    "OpponentPolicy", "OpponentChoice", "choose_opponent", "realized_pool_fraction",
    "MixCategory", "MixCheckpointRef", "MixChoice", "OpponentMixConfig",
    "choose_mix_category", "choose_checkpoint_in_category", "choose_mix_opponent",
    "realized_mix_fractions", "read_opponent_mix_manifest", "opponent_mix_config_to_dict",
    "AdversaryNetSpec", "SMALL_ADVERSARY", "EXPLOIT_WIN_RATE_THRESHOLD", "ExploitReport",
    "frozen_opponent_manifest", "write_frozen_opponent_manifest",
    "adversary_only_rows", "is_adversary_row",
    "build_exploit_report", "exploit_rate_from_pool_stats", "exploit_rate_from_outcomes",
    "exploit_verdict", "file_sha256", "param_fingerprint", "assert_params_unchanged",
    "adversary_optimizer_params", "build_adversary_policy", "save_fresh_adversary",
    "compute_fraction",
]
