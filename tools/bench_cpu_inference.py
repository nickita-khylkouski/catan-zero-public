"""CPU inference benchmark for the 35M EntityGraphPolicy checkpoint.

Standalone measurement script (task #45, Modal feasibility). Does NOT touch
src/. Loads the real checkpoint on device=cpu, featurizes real Rust game
states, and measures:
  - featurization cost per state on this CPU
  - forward latency/throughput at batch 1/8/16/32 x torch threads 1/2/4/8
  - dynamic int8 quantization: speed + max logit/value drift on a probe batch
  - optional torch.compile (guarded, --compile)
  - RSS after model load and after eval

Usage:
  .venv/bin/python /tmp/bench_cpu_inference.py \
      --checkpoint runs/bc/.../checkpoint.pt [--compile]
Pin to 8 cores to simulate a Modal container:
  taskset -c 0-7 .venv/bin/python /tmp/bench_cpu_inference.py ...
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # linux: KB


def collect_states(n_states: int, seed: int = 123):
    """Play random actions; keep decision states with >=6 legal actions."""
    import random

    import catanatron_rs

    colors = ["RED", "BLUE"]
    rng = random.Random(seed)
    states = []
    game = catanatron_rs.Game.simple(colors, seed=seed)
    steps = 0
    while len(states) < n_states and steps < 600:
        steps += 1
        try:
            if game.winner() is not None:
                game = catanatron_rs.Game.simple(colors, seed=seed + steps)
                continue
        except Exception:
            pass
        ids = [int(a) for a in game.playable_action_indices(colors, None)]
        if not ids:
            game = catanatron_rs.Game.simple(colors, seed=seed + steps)
            continue
        if len(ids) >= 6:
            states.append(game.copy())
        game.execute_action_index(rng.choice(ids), colors, None)
    if not states:
        raise RuntimeError("no multi-action states collected")
    return states, tuple(colors)


def featurize(game, colors, action_size):
    from catan_zero.search.neural_rust_mcts import (
        rust_action_context_batch,
        rust_game_to_entity_batch,
        rust_policy_action_ids,
    )

    legal = tuple(int(a) for a in game.playable_action_indices(list(colors), None))
    actor = str(game.current_color())
    pol_ids = rust_policy_action_ids(game, legal, colors=colors, action_size=action_size)
    entity = rust_game_to_entity_batch(
        game, legal, actor=actor, colors=colors, action_size=action_size,
        policy_action_ids=pol_ids,
    )
    ctx = rust_action_context_batch(
        game, legal, actor=actor, colors=colors, action_size=action_size,
        policy_action_ids=pol_ids,
    )
    ids = np.asarray(pol_ids, dtype=np.int64)[None, :]
    return entity, ids, ctx


def make_batch(entity, ids, ctx, batch_size):
    eb = {k: np.repeat(v, batch_size, axis=0) for k, v in entity.items()}
    return eb, np.repeat(ids, batch_size, axis=0), np.repeat(ctx, batch_size, axis=0)


def bench_forward(policy, batches, iters, warmup=3):
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            policy.forward_legal_np(*batches[0])
        t0 = time.perf_counter()
        for i in range(iters):
            policy.forward_legal_np(*batches[i % len(batches)])
        dt = time.perf_counter() - t0
    return dt / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--batches", default="1,8,16,32")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    import torch

    torch.set_num_interop_threads(1)
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    print(f"rss_before_load_gb={rss_gb():.2f}")
    t0 = time.perf_counter()
    policy = EntityGraphPolicy.load(args.checkpoint, device="cpu")
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in policy.model.parameters())
    print(f"model_load_s={load_s:.2f} params={n_params/1e6:.1f}M rss_gb={rss_gb():.2f}")

    states, colors = collect_states(16)
    action_size = policy.action_size

    # featurization timing on real, distinct states
    t0 = time.perf_counter()
    feats = [featurize(g, colors, action_size) for g in states]
    feat_s = (time.perf_counter() - t0) / len(states)
    n_legal = [f[1].shape[1] for f in feats]
    print(f"featurize_ms_per_state={feat_s*1000:.1f} n_states={len(states)} "
          f"legal_actions min/med/max={min(n_legal)}/{sorted(n_legal)[len(n_legal)//2]}/{max(n_legal)}")
    print(f"rss_after_featurize_gb={rss_gb():.2f}")

    thread_counts = [int(x) for x in args.threads.split(",")]
    batch_sizes = [int(x) for x in args.batches.split(",")]
    probe = feats[0]

    results = []
    for nt in thread_counts:
        torch.set_num_threads(nt)
        for bs in batch_sizes:
            batches = [make_batch(*f, bs) for f in feats[:4]]
            iters = max(10, args.iters if bs <= 8 else args.iters // 2)
            per_call = bench_forward(policy, batches, iters)
            results.append(dict(mode="fp32", threads=nt, batch=bs,
                                ms_per_call=per_call * 1000,
                                ms_per_sample=per_call * 1000 / bs))
            print(f"fp32 threads={nt} batch={bs}: {per_call*1000:.1f} ms/call "
                  f"{per_call*1000/bs:.2f} ms/sample")

    # reference outputs for drift check
    with torch.no_grad():
        ref = policy.forward_legal_np(*probe)
        ref_logits = ref["logits"].numpy().copy()
        ref_value = float(ref["value"].numpy()[0])

    # dynamic int8 quantization of Linear layers
    try:
        qmodel = torch.ao.quantization.quantize_dynamic(
            policy.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        orig_model = policy.model
        policy.model = qmodel
        with torch.no_grad():
            qout = policy.forward_legal_np(*probe)
            q_logits = qout["logits"].numpy()
            q_value = float(qout["value"].numpy()[0])
        finite = ref_logits > -1e8
        max_logit_delta = float(np.abs(q_logits[finite] - ref_logits[finite]).max())
        ref_p = np.exp(ref_logits[finite]) / np.exp(ref_logits[finite]).sum()
        q_p = np.exp(q_logits[finite]) / np.exp(q_logits[finite]).sum()
        argmax_same = bool(np.argmax(q_logits[finite]) == np.argmax(ref_logits[finite]))
        print(f"int8 drift: max_logit_delta={max_logit_delta:.4f} "
              f"max_prob_delta={float(np.abs(q_p-ref_p).max()):.4f} "
              f"value_delta={abs(q_value-ref_value):.4f} argmax_same={argmax_same}")
        for nt in thread_counts:
            torch.set_num_threads(nt)
            for bs in batch_sizes:
                batches = [make_batch(*f, bs) for f in feats[:4]]
                per_call = bench_forward(policy, batches, max(10, args.iters // 2))
                results.append(dict(mode="int8", threads=nt, batch=bs,
                                    ms_per_call=per_call * 1000,
                                    ms_per_sample=per_call * 1000 / bs))
                print(f"int8 threads={nt} batch={bs}: {per_call*1000:.1f} ms/call "
                      f"{per_call*1000/bs:.2f} ms/sample")
        policy.model = orig_model
    except Exception as exc:  # noqa: BLE001
        print(f"quantization failed: {exc!r}")

    if args.compile:
        try:
            torch.set_num_threads(4)
            policy.model = torch.compile(policy.model, dynamic=True)
            t0 = time.perf_counter()
            with torch.no_grad():
                policy.forward_legal_np(*probe)
            print(f"compile_first_call_s={time.perf_counter()-t0:.1f}")
            for bs in (1, 16):
                batches = [make_batch(*f, bs) for f in feats[:4]]
                per_call = bench_forward(policy, batches, 15)
                print(f"compiled threads=4 batch={bs}: {per_call*1000:.1f} ms/call "
                      f"{per_call*1000/bs:.2f} ms/sample")
        except Exception as exc:  # noqa: BLE001
            print(f"torch.compile failed: {exc!r}")

    print(f"rss_final_gb={rss_gb():.2f}")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
