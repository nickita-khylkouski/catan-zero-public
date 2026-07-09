"""OPT-1: --eval-cache-size lets self-play disable the per-leaf eval cache.

The eval cache keys every leaf by blake2b(json_snapshot). Self-play states are
unique (Catan transpositions over full state are measure-zero), so the cache
never hits and the key work is pure overhead. This flag threads a cache_size
into every EntityGraphRustEvaluatorConfig; 0 skips the key/store entirely.
Default 100000 preserves prior behavior (verified here so deploying the changed
gen script is a no-op until a launcher opts in with --eval-cache-size 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as cli  # type: ignore  # noqa: E402


def _min_argv(extra):
    return ["--out-dir", "/tmp/x", "--games", "1", "--checkpoint", "/ckpt.pt", *extra]


def test_eval_cache_size_default_preserves_prior_behavior():
    args = cli.build_parser().parse_args(_min_argv([]))
    assert args.eval_cache_size == 100_000


def test_eval_cache_size_zero_parses():
    args = cli.build_parser().parse_args(_min_argv(["--eval-cache-size", "0"]))
    assert args.eval_cache_size == 0


def test_cache_size_zero_disables_store_and_key_work():
    """Correctness: cache_size <= 0 makes the evaluator memoize nothing, so its
    output is a pure function of state (identical to the cached path). Assert the
    config-level switch the flag drives (cache_enabled = cache_size > 0)."""
    from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig

    assert int(EntityGraphRustEvaluatorConfig(cache_size=0).cache_size) == 0
    assert int(EntityGraphRustEvaluatorConfig().cache_size) == 100_000
    # The evaluate() paths gate ALL key/hash/store work on `cache_size > 0`
    # (neural_rust_mcts.py: cache_enabled = int(self.config.cache_size) > 0),
    # so 0 cannot change eval outputs -- it only removes the never-hit lookup.
