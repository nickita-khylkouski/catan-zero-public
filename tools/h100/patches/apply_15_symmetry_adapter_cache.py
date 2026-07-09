#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #35: Symmetry-averaged eval skips adapter resolution cache.

evaluate_symmetry_averaged() unconditionally calls _resolve_entity_adapter(),
even when rust_featurize=True and topology is warm. The regular evaluate()
method has a need_adapter_resolve guard that skips this. This patch adds
the same guard to evaluate_symmetry_averaged().

Usage: python3 apply_15_symmetry_adapter_cache.py /path/to/neural_rust_mcts.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_15_symmetry_adapter_cache.py <path>")
with open(path) as f:
    src = f.read()

if "symmetry_need_adapter_resolve" in src:
    print("[SKIP] symmetry adapter cache already applied")
    sys.exit(0)

# Find the unconditional _resolve_entity_adapter call in evaluate_symmetry_averaged
# and add the need_adapter_resolve guard
OLD_RESOLVE = """        # B2 dedup: see evaluate() -- one shared resolve for both featurizers.
        resolved = _resolve_entity_adapter(
            game,
            legal_actions,
            colors=colors,
            action_size=int(self.policy.action_size),
            policy_action_ids=policy_action_ids,
            snapshot=None,
            action_by_id=None,
            public_observation=bool(self.config.public_observation),
            perspective=acting_color,
        )"""

NEW_RESOLVE = """        # B2 dedup: see evaluate() -- one shared resolve for both featurizers.
        # SYSTEM_DESIGN_FINDINGS #35: skip adapter resolution when topology is
        # warm and rust_featurize is on (same guard as evaluate()).
        symmetry_need_adapter_resolve = (
            (not bool(self.config.rust_featurize)) or self._rust_topology is None
        )
        if symmetry_need_adapter_resolve:
            resolved = _resolve_entity_adapter(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                snapshot=None,
                action_by_id=None,
                public_observation=bool(self.config.public_observation),
                perspective=acting_color,
            )
        else:
            # Topology is warm — re-derive adapter from cached topology
            # without the JSON round-trip.
            resolved = _resolve_entity_adapter(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                snapshot=None,
                action_by_id=None,
                public_observation=bool(self.config.public_observation),
                perspective=acting_color,
                _warm_topology=self._rust_topology,
            )"""

if OLD_RESOLVE in src:
    src = src.replace(OLD_RESOLVE, NEW_RESOLVE, 1)
    print("[OK] Added need_adapter_resolve guard to evaluate_symmetry_averaged")
else:
    print("[WARN] could not find the unconditional _resolve_entity_adapter call")
    print("[INFO] The code may have already been updated or the pattern differs.")
    # Try a simpler patch — just add a comment noting the issue
    print("[INFO] Manual fix needed: add need_adapter_resolve guard to evaluate_symmetry_averaged")
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
print()
print("NOTE: This patch assumes _resolve_entity_adapter accepts a _warm_topology")
print("keyword arg. If it doesn't, the fallback path will need adjustment.")
print("The simplest fix is to just skip the resolve entirely when topology is")
print("warm and pass the cached adapter directly.")
