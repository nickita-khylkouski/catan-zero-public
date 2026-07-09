"""Export the 35M EntityGraphNet to ONNX (dynamic batch) and bench onnxruntime
CPU int8 vs torch eager.

The batch axis is now exportable as a dynamic dimension: `_state_tokens` keeps
the symbolic `tokens.shape[0]` instead of `int(...)`, so a single exported graph
serves any batch size. This matters for the gen-2 CPU evaluator, whose chance
fan-outs (up to 11 dice outcomes) and ragged action counts produce a variable
batch per call. This script proves batch-independence (identical per-sample
outputs at several batch sizes) before benchmarking.

Run (CPU-only, pin to a container's worth of cores to mirror Modal):
  taskset -c 0-7 .venv/bin/python tools/bench_onnx_cpu.py \
      --checkpoint runs/bc/<...>/checkpoint.pt
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

INPUT_NAMES = [
    "hex_tokens", "hex_mask", "vertex_tokens", "vertex_mask",
    "edge_tokens", "edge_mask", "player_tokens", "player_mask",
    "global_tokens", "event_tokens", "event_mask",
    "legal_action_tokens", "legal_action_context",
]

# Batch sizes to prove the dynamic axis on: 1, a couple of small ragged sizes,
# 8 (the container's worker count), and 11 (the max ROLL chance fan-out).
VERIFY_BATCHES = (1, 2, 3, 8, 11)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--out-dir", default=None,
                        help="Directory for the exported .onnx files (default: a temp dir).")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Max allowed |delta| between batched and single-sample outputs.")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from bench_cpu_inference import collect_states, featurize, make_batch
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="entity35m_onnx_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = EntityGraphPolicy.load(args.checkpoint, device="cpu")
    states, colors = collect_states(8)
    feats = [featurize(g, colors, policy.action_size) for g in states]
    entity, ids, ctx = feats[0]
    print("entity keys:", sorted(entity.keys()))

    class Wrapper(torch.nn.Module):
        def __init__(self, model) -> None:  # noqa: ANN001
            super().__init__()
            self.model = model

        def forward(self, *tensors):  # noqa: ANN002
            batch = dict(zip(INPUT_NAMES, tensors))
            out = self.model(batch, return_q=False)
            return out["logits"], out["value"]

    wrapper = Wrapper(policy.model).eval()

    def to_inputs(entity_b, ctx_b):  # noqa: ANN001
        vals = []
        for name in INPUT_NAMES:
            if name == "legal_action_context":
                vals.append(torch.as_tensor(ctx_b, dtype=torch.float32))
            else:
                vals.append(torch.as_tensor(entity_b[name]))
        return tuple(vals)

    example = to_inputs(entity, ctx)
    with torch.no_grad():
        ref_logits, ref_value = wrapper(*example)

    # Batch (axis 0) is now a dynamic dimension for EVERY input and output --
    # `_state_tokens` no longer bakes int(batch) into the mask zeros(). event/
    # action lengths (axis 1) stay dynamic as before.
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES}
    for name in ("event_tokens", "event_mask"):
        dynamic_axes[name][1] = "events"
    for name in ("legal_action_tokens", "legal_action_context"):
        dynamic_axes[name][1] = "actions"
    dynamic_axes["logits"] = {0: "batch", 1: "actions"}
    dynamic_axes["value"] = {0: "batch"}

    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_path = out_dir / "entity35m.onnx"
    int8_path = out_dir / "entity35m_int8.onnx"
    t0 = time.perf_counter()
    # Export a single graph from a batch>1 example so the tracer never sees a
    # size-1 axis it might fold to a constant.
    eb, _ib, cb = make_batch(entity, ids, ctx, 4)
    torch.onnx.export(
        wrapper, to_inputs(eb, cb), str(fp32_path),
        input_names=INPUT_NAMES, output_names=["logits", "value"],
        dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"export+quantize (single dynamic-batch graph): {time.perf_counter()-t0:.1f}s")

    import onnxruntime as ort

    def np_inputs(entity_b, ctx_b):  # noqa: ANN001
        out = {}
        for name in INPUT_NAMES:
            if name == "legal_action_context":
                out[name] = np.asarray(ctx_b, dtype=np.float32)
            else:
                out[name] = np.asarray(entity_b[name])
        return out

    # --- Correctness: one exported graph must serve every batch size, and each
    # row of a repeated batch must equal the single-sample torch reference.
    ref_logits_np = ref_logits.numpy()
    ref_value_np = ref_value.numpy()
    print("dynamic-batch verification (int8 graph):")
    worst = 0.0
    for bs in VERIFY_BATCHES:
        sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        eb, _ib, cb = make_batch(entity, ids, ctx, bs)
        out_logits, out_value = sess.run(None, np_inputs(eb, cb))
        assert out_logits.shape[0] == bs, f"batch={bs}: got logits shape {out_logits.shape}"
        assert out_value.shape[0] == bs, f"batch={bs}: got value shape {out_value.shape}"
        # Every repeated row must match row 0 (batch-independence) ...
        row_delta = float(np.abs(out_logits - out_logits[0:1]).max()) if bs > 1 else 0.0
        # ... and row 0 must match the torch reference (up to int8 drift).
        d_logit = float(np.abs(out_logits[0] - ref_logits_np[0]).max())
        d_value = float(np.abs(out_value[0] - ref_value_np[0]).max())
        worst = max(worst, row_delta)
        status = "OK" if row_delta <= args.tol else "FAIL"
        print(f"  batch={bs:2d}: logits{out_logits.shape} value{out_value.shape} "
              f"row_spread={row_delta:.2e} vs_torch_logit={d_logit:.4f} "
              f"value={d_value:.4f} [{status}]")
    assert worst <= args.tol, (
        f"dynamic batch produced batch-dependent outputs (max row spread {worst:.2e} "
        f"> tol {args.tol}); the export is not truly batch-independent"
    )
    print(f"dynamic-batch verification PASSED (max row spread {worst:.2e} <= {args.tol})")

    # --- Benchmark: int8, single-eval latency (the gen-2 evaluator's hot path).
    print("int8 latency benchmark:")
    for quant_tag, model_path in (("_int8", int8_path), ("_fp32", fp32_path)):
        for bs in (1, 8, 11):
            for intra in (1, 8):
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = intra
                opts.inter_op_num_threads = 1
                sess = ort.InferenceSession(str(model_path), opts,
                                            providers=["CPUExecutionProvider"])
                batches = [make_batch(*f, bs) for f in feats[:4]]
                inps = [np_inputs(eb, cb) for eb, _ib, cb in batches]
                for i in inps[:2]:
                    sess.run(None, i)
                iters = args.iters if bs == 1 else max(10, args.iters // 2)
                t0 = time.perf_counter()
                for i in range(iters):
                    sess.run(None, inps[i % len(inps)])
                per = (time.perf_counter() - t0) / iters
                print(f"  ort{quant_tag} intra={intra} batch={bs:2d}: "
                      f"{per*1000:6.1f} ms/call {per*1000/bs:6.2f} ms/sample")


if __name__ == "__main__":
    main()
