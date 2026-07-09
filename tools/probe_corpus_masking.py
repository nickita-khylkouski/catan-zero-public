#!/usr/bin/env python3
"""Empirically detect whether a training corpus stores player_tokens that are
already public-masked (opponent hidden slots zeroed AT GENERATION) or omniscient.

Per-STACK discriminator (NOT the generation flag): the OLD gen-3 memmap ran
--public-observation at generation yet banked OMNISCIENT player_tokens, whereas
the H100/runsix stack masks BEFORE featurization. The only trustworthy signal is
the rows themselves. Mirrors train_bc's mask target exactly:
catan_zero.rl.entity_token_features.mask_player_tokens_public zeroes
PUBLIC_MASK_PLAYER_SLOTS on every NON-actor player row.

Verdict tokens (stdout, last line): PROVENANCE=masked | omniscient | ambiguous
Exit: 0 decisive (masked or omniscient), 3 ambiguous/unreadable (FAIL LOUD).
"""
import json, sys, glob, os
import numpy as np

ACTOR_SLOT = 1
SLOTS = (4, 5, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)  # PUBLIC_MASK_PLAYER_SLOTS
NZ = 1e-3            # |value| above this = "carries hidden info"
MASKED_MAX = 1e-4    # nonzero-fraction below this = decisively masked
OMNI_MIN   = 2e-2    # nonzero-fraction above this = decisively omniscient
SAMPLE = 8192

def _load_memmap(path):
    meta = json.load(open(os.path.join(path, "corpus_meta.json")))
    n = int(meta["row_count"])
    pt = np.memmap(os.path.join(path, "player_tokens.dat"), dtype="<f2", mode="r").reshape(n, 4, 31)
    pm = np.memmap(os.path.join(path, "player_mask.dat"), dtype="|b1", mode="r").reshape(n, 4)
    idx = np.linspace(0, n - 1, min(n, SAMPLE)).astype(np.int64)
    return np.asarray(pt[idx]).astype(np.float32), np.asarray(pm[idx]), n

def _load_shards(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True)) or \
            sorted(glob.glob(os.path.join(path, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no memmap corpus_meta.json and no .npz shards under {path}")
    pts, pms = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "player_tokens" not in d or "player_mask" not in d:
            continue
        pts.append(np.asarray(d["player_tokens"]).astype(np.float32).reshape(-1, 4, 31))
        pms.append(np.asarray(d["player_mask"]).reshape(-1, 4))
        if sum(x.shape[0] for x in pts) >= SAMPLE:
            break
    if not pts:
        raise ValueError(f"shards under {path} carry no player_tokens/player_mask")
    return np.concatenate(pts)[:SAMPLE], np.concatenate(pms)[:SAMPLE], sum(x.shape[0] for x in pts)

def main():
    path = sys.argv[1]
    if os.path.exists(os.path.join(path, "corpus_meta.json")):
        pt, pm, n = _load_memmap(path); kind = "memmap"
    else:
        pt, pm, n = _load_shards(path); kind = "npz-shards"
    S = pt.shape[0]
    actor = pt[:, :, ACTOR_SLOT] > 0.5                    # (S,4)
    active = pm.astype(bool)                              # (S,4)
    nonactor_active = active & (~actor)                  # (S,4) the opponents we can see publicly
    # layout sanity: every sampled row must have exactly one actor among active players
    actors_per_row = (actor & active).sum(axis=1)
    bad_actor_rows = int((actors_per_row != 1).sum())
    n_opp = int(nonactor_active.sum())
    hidden = np.abs(pt[:, :, SLOTS])                     # (S,4,len(SLOTS))
    mask3d = nonactor_active[:, :, None]
    opp_hidden = hidden[np.broadcast_to(mask3d, hidden.shape)]
    nz_frac = float((opp_hidden > NZ).mean()) if opp_hidden.size else float("nan")
    mean_abs = float(opp_hidden.mean()) if opp_hidden.size else float("nan")
    print(json.dumps({"path": path, "kind": kind, "row_count": n, "sampled": S,
                      "opp_slot_entries": int(opp_hidden.size), "bad_actor_rows": bad_actor_rows,
                      "opp_hidden_nonzero_frac": round(nz_frac, 6), "opp_hidden_mean_abs": round(mean_abs, 6)}))
    # decision
    if opp_hidden.size == 0 or bad_actor_rows > 0.01 * S or not np.isfinite(nz_frac):
        print("PROVENANCE=ambiguous"); sys.exit(3)
    if nz_frac <= MASKED_MAX:
        print("PROVENANCE=masked"); sys.exit(0)
    if nz_frac >= OMNI_MIN:
        print("PROVENANCE=omniscient"); sys.exit(0)
    print("PROVENANCE=ambiguous"); sys.exit(3)

if __name__ == "__main__":
    main()
