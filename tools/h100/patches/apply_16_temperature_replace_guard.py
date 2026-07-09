#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #37: dataclasses.replace(mcts.config, ...) every decision.

mcts.config = dataclasses.replace(mcts.config, temperature=temperature) is
called every decision (~100-200 times per game), creating a new config object
each time even when the temperature hasn't changed. This patch adds a guard
to only replace when the temperature actually differs.

Usage: python3 apply_16_temperature_replace_guard.py /path/to/gumbel_self_play.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_16_temperature_replace_guard.py <path>")
with open(path) as f:
    src = f.read()

if "mcts.config.temperature != temperature" in src:
    print("[SKIP] temperature replace guard already applied")
    sys.exit(0)

OLD_REPLACE = """        mcts.config = dataclasses.replace(mcts.config, temperature=temperature)"""

NEW_REPLACE = """        # SYSTEM_DESIGN_FINDINGS #37: only replace config when temperature
        # actually changes (avoids ~150 unnecessary allocations per game).
        if mcts.config.temperature != temperature:
            mcts.config = dataclasses.replace(mcts.config, temperature=temperature)"""

if OLD_REPLACE in src:
    src = src.replace(OLD_REPLACE, NEW_REPLACE, 1)
    print("[OK] Added temperature change guard")
else:
    print("[WARN] could not find the dataclasses.replace line")
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
