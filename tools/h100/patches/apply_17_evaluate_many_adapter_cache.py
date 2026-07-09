#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #40: evaluate_many missing need_adapter_resolve guard.

evaluate_many() unconditionally calls _resolve_entity_adapter() for every
request, even when rust_featurize=True and topology is warm. The regular
evaluate() has a guard that skips this. evaluate_many is called for EVERY
ROLL chance node (11 children, ~51% of decisions). Each call wastes ~1.3ms
of JSON round-trips.

Usage: python3 apply_17_evaluate_many_adapter_cache.py /path/to/neural_rust_mcts.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_17_evaluate_many_adapter_cache.py <path>")
with open(path) as f:
    src = f.read()

if "many_need_adapter_resolve" in src:
    print("[SKIP] evaluate_many adapter cache already applied")
    sys.exit(0)

# Find the unconditional _resolve_entity_adapter call in evaluate_many
# and add the need_adapter_resolve guard
OLD_RESOLVE = """            snapshot = json.loads(snapshot_text)
            # B2 dedup: see evaluate() -- one shared resolve for both featurizers.
            resolved = _resolve_entity_adapter(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                snapshot=snapshot,
                action_by_id=action_by_id,
                public_observation=bool(self.config.public_observation),
                perspective=acting_color,
            )
            if bool(self.config.rust_featurize):
                entity = self._entity_batch_via_rust(
                    game,
                    colors=colors,
                    policy_action_ids=policy_action_ids,
                    acting_color=acting_color,
                    adapter=resolved[1],
                )"""

NEW_RESOLVE = """            # SYSTEM_DESIGN_FINDINGS #40: skip adapter resolution when topology
            # is warm and rust_featurize is on (same guard as evaluate()).
            many_need_adapter_resolve = (
                (not bool(self.config.rust_featurize)) or self._rust_topology is None
            )
            resolved = None
            if many_need_adapter_resolve:
                snapshot = json.loads(snapshot_text)
                # B2 dedup: see evaluate() -- one shared resolve for both featurizers.
                resolved = _resolve_entity_adapter(
                    game,
                    legal_actions,
                    colors=colors,
                    action_size=int(self.policy.action_size),
                    policy_action_ids=policy_action_ids,
                    snapshot=snapshot,
                    action_by_id=action_by_id,
                    public_observation=bool(self.config.public_observation),
                    perspective=acting_color,
                )
            if bool(self.config.rust_featurize):
                entity = self._entity_batch_via_rust(
                    game,
                    colors=colors,
                    policy_action_ids=policy_action_ids,
                    acting_color=acting_color,
                    adapter=resolved[1] if resolved is not None else None,
                )"""

if OLD_RESOLVE in src:
    src = src.replace(OLD_RESOLVE, NEW_RESOLVE, 1)
    print("[OK] Added need_adapter_resolve guard to evaluate_many")
else:
    print("[WARN] could not find the evaluate_many resolve pattern")
    sys.exit(1)

# Also fix the context_batch_via_rust call to handle resolved=None
OLD_CONTEXT = """            if bool(self.config.rust_featurize):
                context = self._context_batch_via_rust(
                    game,
                    acting_color=acting_color,
                    adapter=resolved[1],
                )"""

NEW_CONTEXT = """            if bool(self.config.rust_featurize):
                context = self._context_batch_via_rust(
                    game,
                    acting_color=acting_color,
                    adapter=resolved[1] if resolved is not None else None,
                )"""

if OLD_CONTEXT in src:
    src = src.replace(OLD_CONTEXT, NEW_CONTEXT, 1)
    print("[OK] Fixed context_batch_via_rust to handle resolved=None")
else:
    print("[WARN] could not find the context_batch_via_rust call in evaluate_many")

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
