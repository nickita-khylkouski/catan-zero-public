#!/usr/bin/env python3
"""CAT-129 champion no-op BIT-IDENTICAL gate (whole-model forward identity).

Loads champion_v0 with the new heads' flags OFF and runs a DETERMINISTIC forward
over a committed 64-row entity-token fixture; asserts logits+value reproduce the
banked reference EXACTLY (max_diff 0.0 over all 64 rows; the 16-row subset is the
first 16). This is the whole-champion complement to the per-head no-op unit tests
in the suite (test_entity_token_edge_policy_head::test_bit_identical_at_init,
test_subgraph_ensemble) — it catches ANY change to the end-to-end forward path
across a merge, which is exactly what the re-gate + CAT-130 post-deploy need.

The fixture + reference are committed under tests/fixtures/, and the forward
runs on CPU, so it reproduces on any freshly-deployed box without external
corpus paths or a GPU. BUT byte-exact (--atol 0.0, the default) is only
box-independent WITHIN THE SAME CPU VENDOR: float32 reduction order/FMA usage
differs across vendors (e.g. Intel vs AMD), so an unchanged champion+code can
produce a benign ~1e-6 delta on a different vendor and fail a strict
byte-exact compare. For cross-vendor fleet-box acceptance, pass --atol 1e-4
(PASS iff max_diff <= atol for both logits and value); the strict default
(--atol 0.0) remains the same-CPU-vendor regression guard.

Modes:
  --extract-fixture <shard.npz>   one-time: bank the 64-row input fixture
  --bank        --champion <ckpt> one-time (on the certified tag): bank the ref
  (default)     --champion <ckpt> [--atol A] gate: forward + assert max_diff
                <= A (A=0.0 default = bit-identical) vs the banked ref
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/noop_input_64.npz"
REF = ROOT / "tests/fixtures/noop_ref.npz"
# Match the canonical fleet/install contract. The old absolute
# /home/ubuntu/catan-zero/... path points at a retired checkout and makes a
# fresh v1 deployment fail before it can inspect the actual bundled champion.
DEFAULT_CKPT = str(Path.home() / "bundle" / "champion_v0.pt")  # md5 8fadfb36
N_ROWS = 64

ENTITY_KEYS = (
    "hex_tokens", "hex_vertex_ids", "hex_edge_ids", "vertex_tokens", "edge_tokens",
    "edge_vertex_ids", "player_tokens", "global_tokens", "legal_action_tokens",
    "legal_action_target_ids", "event_tokens", "event_target_ids", "hex_mask",
    "vertex_mask", "edge_mask", "player_mask", "legal_action_mask", "event_mask",
)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:8]


def extract_fixture(shard: Path) -> None:
    d = np.load(shard, allow_pickle=True)
    n = min(N_ROWS, int(np.asarray(d["legal_action_ids"]).shape[0]))
    cols = {k: np.asarray(d[k])[:n] for k in ENTITY_KEYS if k in d.files}
    cols["legal_action_ids"] = np.asarray(d["legal_action_ids"])[:n]
    cols["legal_action_context"] = np.asarray(d["legal_action_context"])[:n]
    FIX.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIX, **cols)
    print(f"banked fixture: {FIX} ({n} rows, {len(cols)} arrays)")


def _forward(ckpt: str):
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    pol = EntityGraphPolicy.load(ckpt, device="cpu")
    pol.model.eval()
    d = np.load(FIX, allow_pickle=True)
    entity = {k: d[k] for k in ENTITY_KEYS if k in d.files}
    out = pol.forward_legal_np(entity, d["legal_action_ids"], d["legal_action_context"], return_q=False)
    import torch

    logits = out["logits"].detach().cpu().float().numpy()
    value = out["value"].detach().cpu().float().numpy().reshape(-1)
    return logits, value


def bank(ckpt: str) -> None:
    logits, value = _forward(ckpt)
    np.savez(REF, logits=logits, value=value)
    print(f"banked reference: {REF}  value_sha1={_sha(value)} logits_sha1={_sha(logits)}  rows={value.shape[0]}")


def gate(ckpt: str, atol: float = 0.0) -> int:
    if not FIX.exists() or not REF.exists():
        print(f"FAIL: fixture/reference missing ({FIX}, {REF}) — run --extract-fixture + --bank on the certified tag first")
        return 2
    logits, value = _forward(ckpt)
    ref = np.load(REF)
    ok = True
    for name, cur, banked in (("logits", logits, ref["logits"]), ("value", value, ref["value"])):
        if cur.shape != banked.shape:
            print(f"FAIL {name}: shape {cur.shape} != banked {banked.shape}"); ok = False; continue
        md_all = float(np.abs(cur - banked).max()) if cur.size else 0.0
        md_16 = float(np.abs(cur[:16] - banked[:16]).max()) if cur.shape[0] >= 16 else md_all
        identical = np.array_equal(cur, banked)
        within_atol = md_all <= atol
        print(f"{name}: max_diff(16-row)={md_16} max_diff(64-row)={md_all} sha1={_sha(cur)} banked_sha1={_sha(banked)} bit_identical={identical}")
        ok = ok and within_atol
    label = "BIT-IDENTICAL" if atol == 0.0 else f"WITHIN --atol {atol}"
    print(f"champion no-op {label}: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", default=DEFAULT_CKPT)
    ap.add_argument("--extract-fixture", default=None, help="source shard .npz (one-time)")
    ap.add_argument("--bank", action="store_true", help="bank the reference (one-time, on the certified tag)")
    ap.add_argument("--atol", type=float, default=0.0,
                     help="max_diff tolerance (default 0.0 = strict byte-exact, same-CPU-vendor only; "
                          "use 1e-4 for cross-vendor fleet-box acceptance)")
    args = ap.parse_args()
    if args.extract_fixture:
        extract_fixture(Path(args.extract_fixture)); return
    if args.bank:
        bank(args.champion); return
    sys.exit(gate(args.champion, args.atol))


if __name__ == "__main__":
    main()
