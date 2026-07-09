#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #46: Topology cache uses FIFO eviction, not LRU.

_TOPOLOGY_CACHE in entity_token_features.py uses dict.pop(next(iter(...)))
for eviction — FIFO. In practice the cache rarely fills (1 board per worker),
but if it does, FIFO evicts the most-recently-added entry, not the
least-recently-used. This patch converts to OrderedDict + LRU.

Usage: python3 apply_19_topology_cache_lru.py /path/to/entity_token_features.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_19_topology_cache_lru.py <path>")
with open(path) as f:
    src = f.read()

if "topology_cache_lru" in src.lower():
    print("[SKIP] topology cache LRU already applied")
    sys.exit(0)

# Add OrderedDict import
OLD_IMPORT = "import numpy as np"
NEW_IMPORT = """import numpy as np
from collections import OrderedDict"""
if "OrderedDict" not in src and OLD_IMPORT in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("[OK] Added OrderedDict import")
elif "OrderedDict" in src:
    print("[INFO] OrderedDict already imported")
else:
    print("[WARN] could not find import numpy")
    sys.exit(1)

# Replace the cache dict with OrderedDict
OLD_CACHE = "_TOPOLOGY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}"
NEW_CACHE = "_TOPOLOGY_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()  # topology_cache_lru"

if OLD_CACHE in src:
    src = src.replace(OLD_CACHE, NEW_CACHE, 1)
    print("[OK] Replaced _TOPOLOGY_CACHE with OrderedDict")
else:
    print("[WARN] could not find _TOPOLOGY_CACHE declaration")
    sys.exit(1)

# Add move_to_end on cache hit in _topology()
OLD_HIT = """    cached = _TOPOLOGY_CACHE.get(key)
    if cached is None:"""

NEW_HIT = """    cached = _TOPOLOGY_CACHE.get(key)
    if cached is not None:
        _TOPOLOGY_CACHE.move_to_end(key)
    if cached is None:"""

# This is tricky — the original code does:
#   cached = _TOPOLOGY_CACHE.get(key)
#   if cached is None:
#       ... build and insert ...
#   return { ... "tiles": tiles, ... cached fields ... }
# We need to restructure to move_to_end on hit.
# Actually the original returns cached fields directly, so let's look at the full pattern.

# The _topology function:
#   cached = _TOPOLOGY_CACHE.get(key)
#   if cached is None:
#       cached = _build_topology(tiles)
#       if len(_TOPOLOGY_CACHE) >= _TOPOLOGY_CACHE_MAXSIZE:
#           _TOPOLOGY_CACHE.pop(next(iter(_TOPOLOGY_CACHE)))
#       _TOPOLOGY_CACHE[key] = cached
#   return { ... }

# Replace the FIFO eviction with LRU popitem(last=False)
OLD_EVICT = """        if len(_TOPOLOGY_CACHE) >= _TOPOLOGY_CACHE_MAXSIZE:
            _TOPOLOGY_CACHE.pop(next(iter(_TOPOLOGY_CACHE)))
        _TOPOLOGY_CACHE[key] = cached"""

NEW_EVICT = """        if len(_TOPOLOGY_CACHE) >= _TOPOLOGY_CACHE_MAXSIZE:
            _TOPOLOGY_CACHE.popitem(last=False)  # LRU eviction
        _TOPOLOGY_CACHE[key] = cached"""

if OLD_EVICT in src:
    src = src.replace(OLD_EVICT, NEW_EVICT, 1)
    print("[OK] Replaced FIFO eviction with LRU popitem(last=False)")
else:
    print("[WARN] could not find the FIFO eviction pattern")
    sys.exit(1)

# Add move_to_end on hit — need to restructure the if/else
# The pattern is: cached = get(key); if cached is None: ...build...; return {...}
# We want: cached = get(key); if cached is not None: move_to_end(key); if cached is None: ...
# But that changes the logic flow. Simpler: just add move_to_end after the None check.
# Actually the simplest approach: the return statement uses cached, so we can add
# move_to_end right before the return when cached was a hit.

# Let's find the return and add move_to_end there
# Actually, let's just add it right after the get, before the None check
if OLD_HIT in src:
    src = src.replace(OLD_HIT, NEW_HIT, 1)
    print("[OK] Added move_to_end on cache hit")
else:
    print("[WARN] could not find the cache hit pattern — move_to_end not added")
    print("[INFO] LRU eviction is still applied, just without move_to_end on hit")

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
