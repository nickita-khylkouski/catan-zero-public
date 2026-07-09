"""KataGo-style hybrid continuous-flywheel for catan-zero self-play.

The architecture the discrete-vs-continuous research (memory ``catan-discrete-vs-continuous-verdict``)
landed on: a *continuous* training loop over a *growing windowed* replay buffer, with a *cheap
discrete gate* on the self-play-generating checkpoint only, plus a *15-25% opponent pool* of
archived champions. NOT pure discrete (idles GPUs on a barrier), NOT pure AlphaZero (a bad net
poisons a small buffer).

Modules (each has a stdlib-only ``__main__`` self-test):
  - ``replay_window``       — KataGo power-law window + shard registry (the sampling distribution)
  - ``checkpoint_registry`` — candidate publish / gated champion promotion / opponent archive
  - ``opponent_pool``       — deterministic archived-opponent sampling (anti-forgetting)
  - ``opponent_mix``        — generalized N-way categorical opponent-mix sampling (CAT-54)
  - ``exploit_probe``       — adversary-vs-FROZEN-champion exploitability probe (R8/Wang, scaffolding)
  - ``config``              — durable name-keyed FlywheelConfig (regime switch for the ablation)

Orchestration lives in ``tools/continuous_flywheel.py`` (references the existing generation/train/
gate scripts, mirroring ``tools/selfplay_loop.py``'s shape).

Build status: modules + self-tests written; NOT yet wired into a running fleet (intentional — the
current discrete gen-1 must clear its G1 gate first to validate the loop before we flip continuous).
"""
from __future__ import annotations

from .config import FlywheelConfig, SCHEMA_VERSION
from .replay_window import WindowedReplay, katago_window_rows, ShardMeta, WindowSelection
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
    "FlywheelConfig", "SCHEMA_VERSION",
    "WindowedReplay", "katago_window_rows", "ShardMeta", "WindowSelection",
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
