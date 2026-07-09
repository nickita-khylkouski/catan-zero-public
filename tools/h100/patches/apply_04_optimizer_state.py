#!/usr/bin/env python3
"""
Apply optimizer state persistence to train_bc.py.
SYSTEM_DESIGN_FINDINGS #8: Save/load optimizer.state_dict() as sidecar files.

Saves to <checkpoint>.optimizer.pt after each epoch and at final checkpoint.
Loads from <init_checkpoint>.optimizer.pt on resume if the file exists.

Usage: python3 apply_04_optimizer_state.py /path/to/train_bc.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_04_optimizer_state.py <path>")
with open(path) as f:
    src = f.read()

if "optimizer.pt" in src:
    print("[SKIP] optimizer state persistence already applied")
    sys.exit(0)

# --- Add optimizer state load after optimizer creation ---
OLD_OPT_CREATE = "        optimizer = _make_optimizer(params, args, getattr(policy, \"device\", args.device))\n        train_fn = _train_candidate_batch"
NEW_OPT_CREATE = """        optimizer = _make_optimizer(params, args, getattr(policy, "device", args.device))
        # SYSTEM_DESIGN_FINDINGS #8: Load optimizer state sidecar on resume.
        if args.init_checkpoint:
            _opt_sidecar = Path(str(args.init_checkpoint) + ".optimizer.pt")
            if _opt_sidecar.exists():
                import torch as _torch
                try:
                    optimizer.load_state_dict(_torch.load(_opt_sidecar, map_location=str(getattr(policy, "device", args.device))))
                    _rank0_print(json.dumps({"progress": "optimizer_state_loaded", "sidecar": str(_opt_sidecar)}), ddp)
                except Exception as _e:
                    _rank0_print(json.dumps({"progress": "optimizer_state_load_failed", "error": str(_e)[:200]}), ddp)
        train_fn = _train_candidate_batch"""

if OLD_OPT_CREATE in src:
    src = src.replace(OLD_OPT_CREATE, NEW_OPT_CREATE, 1)
    print("[OK] Added optimizer state load on resume")
else:
    print("[WARN] could not find optimizer creation anchor")

# --- Add optimizer state save after per-epoch save ---
OLD_EPOCH_SAVE = """            _save_policy(
                policy, str(epoch_path), ddp, mask_hidden_info=bool(args.mask_hidden_info)
            )
        if args.max_steps > 0 and global_step >= args.max_steps:"""
NEW_EPOCH_SAVE = """            _save_policy(
                policy, str(epoch_path), ddp, mask_hidden_info=bool(args.mask_hidden_info)
            )
            # SYSTEM_DESIGN_FINDINGS #8: Save optimizer state sidecar per epoch.
            if int(ddp["rank"]) == 0:
                import torch as _torch
                _torch.save(optimizer.state_dict(), str(epoch_path) + ".optimizer.pt")
        if args.max_steps > 0 and global_step >= args.max_steps:"""

if OLD_EPOCH_SAVE in src:
    src = src.replace(OLD_EPOCH_SAVE, NEW_EPOCH_SAVE, 1)
    print("[OK] Added optimizer state save per epoch")
else:
    print("[WARN] could not find per-epoch save anchor")

# --- Add optimizer state save after final save ---
OLD_FINAL_SAVE = '    _save_policy(policy, args.checkpoint, ddp, mask_hidden_info=bool(args.mask_hidden_info))\n    report = {'
NEW_FINAL_SAVE = """    _save_policy(policy, args.checkpoint, ddp, mask_hidden_info=bool(args.mask_hidden_info))
    # SYSTEM_DESIGN_FINDINGS #8: Save optimizer state sidecar at final checkpoint.
    if int(ddp["rank"]) == 0:
        import torch as _torch
        _torch.save(optimizer.state_dict(), str(args.checkpoint) + ".optimizer.pt")
    report = {"""

if OLD_FINAL_SAVE in src:
    src = src.replace(OLD_FINAL_SAVE, NEW_FINAL_SAVE, 1)
    print("[OK] Added optimizer state save at final checkpoint")
else:
    print("[WARN] could not find final save anchor")

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
