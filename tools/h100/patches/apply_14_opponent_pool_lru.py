#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #33: Opponent pool evaluator cache never evicts.

pool_evaluator_cache and mix_evaluator_cache in gumbel_self_play.py grow
without limit. Each loaded evaluator holds ~1.1GB GPU memory. Over a long
run, a worker can load many distinct opponent checkpoints, slowly filling
GPU memory until OOM.

This patch adds LRU eviction with a configurable max resident count.

Usage: python3 apply_14_opponent_pool_lru.py /path/to/gumbel_self_play.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_14_opponent_pool_lru.py <path>")
with open(path) as f:
    src = f.read()

if "MAX_POOL_EVALUATORS" in src:
    print("[SKIP] opponent pool LRU already applied")
    sys.exit(0)

# Add OrderedDict import
OLD_IMPORT = "import json"
NEW_IMPORT = """import json
from collections import OrderedDict"""
if OLD_IMPORT in src and "OrderedDict" not in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("[OK] Added OrderedDict import")

# Replace pool_evaluator_cache dict with OrderedDict + LRU eviction
OLD_POOL_CACHE = "    pool_evaluator_cache: dict[str, RustEvaluator] = {}"
NEW_POOL_CACHE = """    # SYSTEM_DESIGN_FINDINGS #33: LRU eviction for opponent evaluator cache.
    # Each loaded evaluator holds ~1.1GB GPU memory. Without eviction, a
    # worker loading many distinct opponent checkpoints will OOM.
    pool_evaluator_cache: OrderedDict[str, RustEvaluator] = OrderedDict()
    MAX_POOL_EVALUATORS = 3  # Max resident opponent models per worker"""

if OLD_POOL_CACHE in src:
    src = src.replace(OLD_POOL_CACHE, NEW_POOL_CACHE, 1)
    print("[OK] Replaced pool_evaluator_cache with LRU OrderedDict")
else:
    print("[WARN] could not find pool_evaluator_cache")
    sys.exit(1)

# Replace mix_evaluator_cache dict with OrderedDict + LRU eviction
OLD_MIX_CACHE = "    mix_evaluator_cache: dict[str, RustEvaluator] = {}"
NEW_MIX_CACHE = """    mix_evaluator_cache: OrderedDict[str, RustEvaluator] = OrderedDict()
    MAX_MIX_EVALUATORS = 3  # Same LRU limit for mix opponents"""

if OLD_MIX_CACHE in src:
    src = src.replace(OLD_MIX_CACHE, NEW_MIX_CACHE, 1)
    print("[OK] Replaced mix_evaluator_cache with LRU OrderedDict")
else:
    print("[WARN] could not find mix_evaluator_cache")

# Add LRU move_to_end on cache hit + eviction on insert for pool cache
OLD_POOL_HIT = """                    opponent_evaluator = pool_evaluator_cache.get(choice.path)
                    if opponent_evaluator is None:
                        opponent_evaluator = opponent_pool.evaluator_factory(choice.path)
                        pool_evaluator_cache[choice.path] = opponent_evaluator"""

NEW_POOL_HIT = """                    opponent_evaluator = pool_evaluator_cache.get(choice.path)
                    if opponent_evaluator is not None:
                        pool_evaluator_cache.move_to_end(choice.path)
                    else:
                        opponent_evaluator = opponent_pool.evaluator_factory(choice.path)
                        pool_evaluator_cache[choice.path] = opponent_evaluator
                        if len(pool_evaluator_cache) > MAX_POOL_EVALUATORS:
                            _evicted = pool_evaluator_cache.popitem(last=False)
                            del _evicted[1]"""

if OLD_POOL_HIT in src:
    src = src.replace(OLD_POOL_HIT, NEW_POOL_HIT, 1)
    print("[OK] Added LRU hit/eviction for pool_evaluator_cache")
else:
    print("[WARN] could not find pool cache hit pattern")

# Add LRU move_to_end on cache hit + eviction on insert for mix cache
OLD_MIX_HIT = """                    opponent_evaluator = mix_evaluator_cache.get(mix_choice.path)
                    if opponent_evaluator is None:
                        opponent_evaluator = opponent_mix.evaluator_factory(mix_choice.path)
                        mix_evaluator_cache[mix_choice.path] = opponent_evaluator"""

NEW_MIX_HIT = """                    opponent_evaluator = mix_evaluator_cache.get(mix_choice.path)
                    if opponent_evaluator is not None:
                        mix_evaluator_cache.move_to_end(mix_choice.path)
                    else:
                        opponent_evaluator = opponent_mix.evaluator_factory(mix_choice.path)
                        mix_evaluator_cache[mix_choice.path] = opponent_evaluator
                        if len(mix_evaluator_cache) > MAX_MIX_EVALUATORS:
                            _evicted = mix_evaluator_cache.popitem(last=False)
                            del _evicted[1]"""

if OLD_MIX_HIT in src:
    src = src.replace(OLD_MIX_HIT, NEW_MIX_HIT, 1)
    print("[OK] Added LRU hit/eviction for mix_evaluator_cache")
else:
    print("[WARN] could not find mix cache hit pattern")

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
