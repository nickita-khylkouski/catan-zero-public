#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #47: prior_policy stored as fp16 — precision loss.

prior_policy is stored as fp16 in decision rows, but it's used for KL
divergence computation during training. fp16's ~1e-3 relative precision
flushes small priors (1e-4 to 1e-3 range at 54-action placement roots) to
zero, biasing KL diagnostics. target_policy is already fp32 — prior_policy
should match.

Usage: python3 apply_18_prior_policy_fp32.py /path/to/gumbel_self_play.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_18_prior_policy_fp32.py <path>")
with open(path) as f:
    src = f.read()

if "prior_policy_fp32" in src:
    print("[SKIP] prior_policy fp32 already applied")
    sys.exit(0)

OLD = """    prior_policy = np.asarray(
        [float(result.priors.get(int(action), 0.0)) for action in legal_rust],
        dtype=np.float16,
    )"""

NEW = """    # SYSTEM_DESIGN_FINDINGS #47: fp32, not fp16. prior_policy feeds KL
    # divergence diagnostics; fp16's ~1e-3 precision flushes small priors
    # (1e-4 range at 54-action placement) to zero, biasing KL. +4 bytes/row.
    prior_policy = np.asarray(
        [float(result.priors.get(int(action), 0.0)) for action in legal_rust],
        dtype=np.float32,
    )"""

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    print("[OK] Changed prior_policy from fp16 to fp32")
else:
    print("[WARN] could not find the prior_policy fp16 line")
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
