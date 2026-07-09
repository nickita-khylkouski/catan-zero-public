#!/usr/bin/env python3
"""
Apply LRU cache eviction to neural_rust_mcts.py.
SYSTEM_DESIGN_FINDINGS #15: FIFO → LRU via OrderedDict + move_to_end.

Fixes all 3 cache locations:
  1. EntityGraphRustEvaluator.evaluate() — single-leaf path
  2. EntityGraphRustEvaluator.evaluate_many() — batch path
  3. BatchedEntityGraphRustEvaluator — threaded batch path (with lock)

Usage: python3 apply_02_lru_cache.py /path/to/neural_rust_mcts.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_02_lru_cache.py <path>")
with open(path) as f:
    src = f.read()

if "OrderedDict" in src and "move_to_end" in src:
    print("[SKIP] LRU cache already applied")
    sys.exit(0)

# --- Add OrderedDict import ---
OLD_IMPORT = "import hashlib"
NEW_IMPORT = "from collections import OrderedDict\nimport hashlib"
if OLD_IMPORT in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("[OK] Added OrderedDict import")
else:
    print("[WARN] could not find import anchor")

# --- Change dict to OrderedDict in __init__ ---
OLD_CACHE_INIT = "self._cache: dict[tuple[str, str, tuple[str, ...], tuple[int, ...]], tuple[dict[int, float], float]] = {}"
NEW_CACHE_INIT = "self._cache: OrderedDict[tuple[str, str, tuple[str, ...], tuple[int, ...]], tuple[dict[int, float], float]] = OrderedDict()"
if OLD_CACHE_INIT in src:
    src = src.replace(OLD_CACHE_INIT, NEW_CACHE_INIT, 1)
    print("[OK] Changed cache dict to OrderedDict")
else:
    print("[WARN] could not find cache init line")

# --- Add move_to_end on cache hits (single-leaf path) ---
# Pattern: "cached = self._cache.get(cache_key)\n            if cached is not None:"
# We need to add move_to_end after the "if cached is not None:" line
# But there are multiple such patterns. Let's be more specific.

# Location 1: EntityGraphRustEvaluator.evaluate()
OLD_HIT_1 = """            cached = self._cache.get(cache_key)
            if cached is not None:
                # CAT-61: cache entries are (priors, value) or (priors, value,
                # uncertainty); tolerate both so a mixed-format cache is safe.
                uncertainty = cached[2] if len(cached) > 2 else 0.0
                return self._eval_result(dict(cached[0]), float(cached[1]), uncertainty)"""
NEW_HIT_1 = """            cached = self._cache.get(cache_key)
            if cached is not None:
                # CAT-61: cache entries are (priors, value) or (priors, value,
                # uncertainty); tolerate both so a mixed-format cache is safe.
                # SYSTEM_DESIGN_FINDINGS #15: LRU — move hit to end.
                self._cache.move_to_end(cache_key)
                uncertainty = cached[2] if len(cached) > 2 else 0.0
                return self._eval_result(dict(cached[0]), float(cached[1]), uncertainty)"""
if OLD_HIT_1 in src:
    src = src.replace(OLD_HIT_1, NEW_HIT_1, 1)
    print("[OK] Added move_to_end to evaluate() cache hit")
else:
    print("[WARN] could not find evaluate() cache hit pattern")

# Location 2: evaluate_many() batch path
OLD_HIT_2 = """                cached = self._cache.get(cache_key)
                if cached is not None:
                    # CAT-61: tolerate (priors, value) and (priors, value, unc).
                    uncertainty = cached[2] if len(cached) > 2 else 0.0
                    results[request_index] = self._eval_result(
                        dict(cached[0]), float(cached[1]), uncertainty
                    )
                    continue"""
NEW_HIT_2 = """                cached = self._cache.get(cache_key)
                if cached is not None:
                    # CAT-61: tolerate (priors, value) and (priors, value, unc).
                    # SYSTEM_DESIGN_FINDINGS #15: LRU — move hit to end.
                    self._cache.move_to_end(cache_key)
                    uncertainty = cached[2] if len(cached) > 2 else 0.0
                    results[request_index] = self._eval_result(
                        dict(cached[0]), float(cached[1]), uncertainty
                    )
                    continue"""
if OLD_HIT_2 in src:
    src = src.replace(OLD_HIT_2, NEW_HIT_2, 1)
    print("[OK] Added move_to_end to evaluate_many() cache hit")
else:
    print("[WARN] could not find evaluate_many() cache hit pattern")

# Location 3: BatchedEntityGraphRustEvaluator.evaluate() (threaded, with lock)
OLD_HIT_3 = """            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                # CAT-61: tolerate (priors, value) and (priors, value, unc).
                uncertainty = cached[2] if len(cached) > 2 else 0.0
                return self._eval_result(dict(cached[0]), float(cached[1]), uncertainty)"""
NEW_HIT_3 = """            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    # SYSTEM_DESIGN_FINDINGS #15: LRU move_to_end (under lock).
                    self._cache.move_to_end(cache_key)
            if cached is not None:
                # CAT-61: tolerate (priors, value) and (priors, value, unc).
                uncertainty = cached[2] if len(cached) > 2 else 0.0
                return self._eval_result(dict(cached[0]), float(cached[1]), uncertainty)"""
if OLD_HIT_3 in src:
    src = src.replace(OLD_HIT_3, NEW_HIT_3, 1)
    print("[OK] Added move_to_end to BatchedEntityGraphRustEvaluator cache hit")
else:
    print("[WARN] could not find BatchedEntityGraphRustEvaluator cache hit pattern")

# --- Change all FIFO evictions to LRU ---
# All 3 locations use: self._cache.pop(next(iter(self._cache)))
# Change to: self._cache.popitem(last=False)
COUNT = src.count("self._cache.pop(next(iter(self._cache)))")
if COUNT > 0:
    src = src.replace("self._cache.pop(next(iter(self._cache)))", "self._cache.popitem(last=False)")
    print(f"[OK] Changed {COUNT} FIFO evictions to LRU popitem(last=False)")
else:
    print("[SKIP] no FIFO eviction patterns found")

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
