#!/usr/bin/env python3
"""CAT-71: standing performance measurement snapshot tool.

One invocation = one (or a few, for gpu-util's multi-GPU case) JSONL row
appended to the perf ledger (default runs/perf/perf_ledger.jsonl), plus the
same row(s) printed as JSON to stdout. tools/perf_report.py reads the ledger
back and renders the regression-catching report.

Subcommands
-----------
leaf
    Per-leaf profiler: snapshot-fetch / FFI-resolve / featurize / FFI-context /
    NN-forward / postprocess time split for `EntityGraphRustEvaluator.evaluate()`,
    single-threaded, on whatever --device you point it at. Reuses
    tools/bench_leaf_eval_batching.py's deterministic leaf-state collector
    and tiny "fast policy" builder (see --checkpoint to use a real
    checkpoint instead -- required for a real baseline-reproduction claim,
    see the docstring on `_baseline_check`).

    Known baseline to validate against (docs/plans/
    CATAN_ZERO_RESEARCH_CHRONICLE.md section 10.1 / 10.2): GPU per-leaf
    ~3.4ms with NN at only ~4% of that (featurize+FFI = ~96%); CPU int8
    ~38ms/eval, forward-pass dominated.

    Example (dev smoke test, tiny policy, fast):
        tools/perf_snapshot.py leaf --device cpu --num-evals 32

    Example (real baseline reproduction, host-only, needs a real checkpoint):
        tools/perf_snapshot.py leaf --device cuda --checkpoint runs/self_play/champions/current_best.pt --num-evals 200

gen-log
    Parse a tools/generate_gumbel_selfplay_data.py manifest.json for
    rows/hr (and games/hr) attributed to a host.

    Example:
        tools/perf_snapshot.py gen-log --manifest runs/self_play/wave42/manifest.json --hostname gpu0

gate
    Gate cost tracking: wall-clock + game-count + extension-tier info per
    gate run. Either point --summary at an already-produced gate/H2H summary
    JSON, or (host-only) wrap the gate command itself with --cmd so the
    wall-clock is actually measured rather than trusted from the summary.

    Example (parse existing summary):
        tools/perf_snapshot.py gate --gate-name gen2_vs_gen1 --summary runs/gates/h2h_summary.json

    Example (host-only, measure wall-clock around the gate command itself --
    the gate command must itself support --out <path>):
        tools/perf_snapshot.py gate --gate-name gen2_vs_gen1 \\
            --cmd "python tools/gumbel_search_vs_bot_h2h.py --candidate ckpt.pt --champion champ.pt --games 1000 --out /tmp/h2h.json" \\
            --out-capture /tmp/h2h.json

gpu-util
    MPS/context-thrash monitoring: parse `nvidia-smi dmon -s pucm` output
    (one row per GPU) and flag the documented context-thrash anomaly
    signature (~90% SM utilization, ~0% memory-bandwidth utilization).

    Example (host-only, requires nvidia-smi + a live GPU):
        tools/perf_snapshot.py gpu-util --live

    Example (parse a previously captured sample, works anywhere):
        nvidia-smi dmon -s pucm -c 1 > /tmp/dmon.txt   # run this on the GPU host
        tools/perf_snapshot.py gpu-util --input /tmp/dmon.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import perf_common


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _policy_id_from_path(path: str) -> str:
    return Path(path).name.removesuffix(".pt")


# --- leaf mode ---------------------------------------------------------------


def _baseline_check(stage_pct: dict[str, float], *, device: str, has_real_checkpoint: bool) -> dict[str, Any]:
    """Compare a measured stage-percentage split against the documented
    baseline (GPU: NN forward ~4%, featurize+FFI ~96%; CPU: forward-pass
    dominated). Only meaningful with a real (not tiny-fast-policy)
    checkpoint -- bench_leaf_eval_batching.py's own docstring notes the fast
    policy's forward pass (~1ms) is deliberately too small to reproduce the
    CPU-forward-dominated regime, so `matches_baseline` is left `None`
    (not evaluated, not silently "passed") unless `has_real_checkpoint`.

    `featurize_ffi_pct` sums every stage that is Rust-FFI/snapshot-resolution
    work: `snapshot_fetch` + `ffi_resolve` (see `profile_leaf_eval` -- these
    two are the ones `EntityGraphRustEvaluator.evaluate()` actually pays on
    its real `resolved=` fast path) plus the legacy `featurize`/`ffi_context`
    buckets (now near-no-op on that path, but still real numpy work).
    """
    nn_pct = stage_pct.get("nn_forward", 0.0)
    featurize_ffi_pct = (
        stage_pct.get("snapshot_fetch", 0.0)
        + stage_pct.get("ffi_resolve", 0.0)
        + stage_pct.get("featurize", 0.0)
        + stage_pct.get("ffi_context", 0.0)
    )
    is_gpu = device.startswith("cuda") or device == "gpu"
    if is_gpu:
        expected_regime = "gpu"
        notes = (
            "expect featurize+FFI to dominate (~96%), NN forward small (~4%) "
            "per CATAN_ZERO_RESEARCH_CHRONICLE.md section 10.1/10.2"
        )
        matches = nn_pct <= 15.0 and featurize_ffi_pct >= 80.0
    else:
        expected_regime = "cpu"
        notes = (
            "expect NN forward to dominate on CPU (historical int8 baseline "
            "~38ms/eval, forward-pass dominated)"
        )
        matches = nn_pct >= 50.0
    return {
        "expected_regime": expected_regime,
        "nn_forward_pct": nn_pct,
        "featurize_ffi_pct": featurize_ffi_pct,
        "matches_baseline": bool(matches) if has_real_checkpoint else None,
        "notes": notes if has_real_checkpoint else notes + " -- NOT evaluated: pass --checkpoint for a real baseline check (tiny fast-policy forward pass is too small to reproduce the CPU/GPU split, see bench_leaf_eval_batching.py docstring)",
    }


# Measured per-leaf stages (excludes the residual "postprocess" bucket and
# "total"). CAT-71 named split; opt_sweep.py (CAT-72) iterates this to rank the
# post-Rust-featurizer bottleneck, so it must stay the set of keys present in
# both `stages` and `stage_pct_of_total`.
_NAMED_STAGES: tuple[str, ...] = (
    "snapshot_fetch",
    "ffi_resolve",
    "featurize",
    "ffi_context",
    "nn_forward",
)


def profile_leaf_eval(
    *,
    num_evals: int,
    seed: int,
    device: str,
    checkpoint: str | None,
    public_observation: bool,
    hostname: str | None = None,
    rust_featurize: bool = False,
) -> dict[str, Any]:
    """Run `num_evals` single-threaded `EntityGraphRustEvaluator.evaluate()`
    calls with stage-level timing (snapshot-fetch / FFI-resolve / featurize /
    FFI-context / NN-forward / postprocess), via monkeypatching the
    module-level featurizer functions, the policy's forward method, and the
    two FFI-resolution helpers `evaluate()` itself calls, for the duration
    of the run.

    CAT-71 review finding 1: `evaluate()` resolves the entity adapter ITSELF
    (`_fetch_leaf_decision_inputs` + `_resolve_entity_adapter`) and passes
    the result as `resolved=`/`snapshot=`/`action_by_id=` into
    `rust_game_to_entity_batch`/`rust_action_context_batch`, so on the real
    `evaluate()` path those two functions take their short-circuit branch and
    never repeat the resolution work -- monkeypatching only them (the old
    bug) leaves the real Rust FFI cost (`game.json_snapshot`,
    `game.playable_action_indices`, `game.playable_actions_json`,
    `game.player_state_json` per color) completely untimed, so it silently
    falls into the residual "postprocess" bucket and the featurize_ffi_pct>=80
    GPU baseline check false-fails. Timing `_fetch_leaf_decision_inputs`
    ("snapshot_fetch") and `_resolve_entity_adapter` ("ffi_resolve")
    themselves closes that gap regardless of the `resolved=` fast path.

    CAT-72: `rust_featurize` mirrors
    `EntityGraphRustEvaluatorConfig.rust_featurize` (task #81 phase 2). When
    True, the "featurize"/"ffi_context" stages instead time the native
    `build_entity_features_rust`/`build_action_context_rust` calls, and the
    one-time-per-evaluator-lifetime `compute_rust_topology` bootstrap is timed
    separately as `topology_bootstrap_ms` (excluded from the steady-state
    stage split -- it is a fixed one-off, not a per-leaf cost). This lets
    tools/opt_sweep.py A/B the legacy vs native featurize path on the same
    deterministic leaf sequence and name the new top cost from data.

    Reuses tools/bench_leaf_eval_batching.py's `_collect_leaf_states` (real
    self-play-shaped states) and `_fast_policy` (tiny-but-structurally-real
    policy for a quick dev smoke test when --checkpoint isn't given).
    """
    import torch

    torch.set_num_threads(1)

    from bench_leaf_eval_batching import _collect_leaf_states, _fast_policy
    from catan_zero.rl.gumbel_self_play import COLORS
    from catan_zero.search import neural_rust_mcts as nrm

    # cache_size=0 disables the evaluate()-level result cache so every
    # collected leaf state actually reaches the featurizer/forward calls --
    # required for the per-call stage lists below to stay aligned 1:1 with
    # each "total" measurement (a cache hit would skip straight to return,
    # silently desynchronizing the lists).
    config = nrm.EntityGraphRustEvaluatorConfig(
        public_observation=bool(public_observation),
        cache_size=0,
        rust_featurize=bool(rust_featurize),
    )
    if checkpoint:
        evaluator = nrm.EntityGraphRustEvaluator.from_checkpoint(checkpoint, device=device, config=config)
    else:
        policy = _fast_policy(seed=seed, device=device)
        evaluator = nrm.EntityGraphRustEvaluator(policy, config=config)

    states = _collect_leaf_states(num_states=num_evals, seed=seed)

    stage_ms: dict[str, list[float]] = {name: [] for name in _NAMED_STAGES}
    stage_ms["total"] = []
    topology_bootstrap_ms: list[float] = []

    real_forward = evaluator.policy.forward_legal_np
    real_fetch_inputs = nrm._fetch_leaf_decision_inputs
    real_resolve_adapter = nrm._resolve_entity_adapter

    # Restore list: (target, attr, original) so the finally block is uniform
    # regardless of which featurizer path was patched.
    patched: list[tuple[Any, str, Any]] = []

    def _patch(target: Any, attr: str, stage: str) -> None:
        real = getattr(target, attr)

        def _timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = real(*args, **kwargs)
            stage_ms[stage].append((time.perf_counter() - start) * 1000.0)
            return result

        setattr(target, attr, _timed)
        patched.append((target, attr, real))

    def _timed_forward(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = real_forward(*args, **kwargs)
        stage_ms["nn_forward"].append((time.perf_counter() - start) * 1000.0)
        return result

    def _timed_fetch_inputs(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = real_fetch_inputs(*args, **kwargs)
        stage_ms["snapshot_fetch"].append((time.perf_counter() - start) * 1000.0)
        return result

    def _timed_resolve_adapter(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = real_resolve_adapter(*args, **kwargs)
        stage_ms["ffi_resolve"].append((time.perf_counter() - start) * 1000.0)
        return result

    if rust_featurize:
        # Native path: evaluate() routes featurization through the Rust
        # builders (CAT-65/task #81), so time those instead of the Python-path
        # FFI wrappers. `compute_rust_topology` is the one-time bootstrap.
        from catan_zero.rl import action_context_features_rust as acfr
        from catan_zero.rl import entity_token_features_rust as etfr

        _patch(etfr, "build_entity_features_rust", "featurize")
        _patch(acfr, "build_action_context_rust", "ffi_context")
        real_topology = etfr.compute_rust_topology

        def _timed_topology(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = real_topology(*args, **kwargs)
            topology_bootstrap_ms.append((time.perf_counter() - start) * 1000.0)
            return result

        etfr.compute_rust_topology = _timed_topology
        patched.append((etfr, "compute_rust_topology", real_topology))
    else:
        _patch(nrm, "rust_game_to_entity_batch", "featurize")
        _patch(nrm, "rust_action_context_batch", "ffi_context")

    evaluator.policy.forward_legal_np = _timed_forward
    # These two are called directly by `evaluate()` (bare module-level
    # names, resolved from the module globals at call time) BEFORE it builds
    # `resolved=` and hands it to the two functions patched above -- patching
    # them here is what actually captures the real FFI cost regardless of
    # the `resolved=` short-circuit (CAT-71 review finding 1).
    nrm._fetch_leaf_decision_inputs = _timed_fetch_inputs
    nrm._resolve_entity_adapter = _timed_resolve_adapter
    try:
        for game, legal_actions, root_color in states:
            start = time.perf_counter()
            evaluator.evaluate(game, legal_actions, root_color=root_color, colors=COLORS)
            stage_ms["total"].append((time.perf_counter() - start) * 1000.0)
    finally:
        for target, attr, real in patched:
            setattr(target, attr, real)
        evaluator.policy.forward_legal_np = real_forward
        nrm._fetch_leaf_decision_inputs = real_fetch_inputs
        nrm._resolve_entity_adapter = real_resolve_adapter

    stages = {name: perf_common.summarize_latencies(values) for name, values in stage_ms.items()}
    total_sum = stages["total"]["total_ms"] or 1.0e-9
    stage_pct = {
        name: (stages[name]["total_ms"] / total_sum) * 100.0
        for name in _NAMED_STAGES
    }
    stage_pct["postprocess"] = max(
        0.0, 100.0 - sum(stage_pct[name] for name in _NAMED_STAGES)
    )

    checkpoint_label = _policy_id_from_path(checkpoint) if checkpoint else "fast_policy"
    row = {
        "kind": "leaf",
        "timestamp": _now_iso(),
        "hostname": hostname,
        "device": device,
        "checkpoint": checkpoint,
        "checkpoint_label": checkpoint_label,
        "num_evals": len(states),
        "seed": seed,
        "public_observation": bool(public_observation),
        "rust_featurize": bool(rust_featurize),
        "stages": stages,
        "stage_pct_of_total": stage_pct,
        "topology_bootstrap_ms": (
            perf_common.summarize_latencies(topology_bootstrap_ms) if rust_featurize else None
        ),
        "baseline_check": _baseline_check(stage_pct, device=device, has_real_checkpoint=bool(checkpoint)),
        "key": perf_common.stable_key(
            "leaf", device, checkpoint_label, bool(rust_featurize), _now_iso(), os.getpid()
        ),
    }
    return row


# --- gen-log mode --------------------------------------------------------------


def _run_genlog_mode(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    return perf_common.parse_generation_manifest(payload, hostname=args.hostname)


# --- gate mode -----------------------------------------------------------------


def _run_gate_mode(args: argparse.Namespace) -> dict[str, Any]:
    wall_clock: float | None = None
    if args.cmd:
        if not args.out_capture:
            raise SystemExit("--cmd requires --out-capture <path> (the gate command's own --out file)")
        started = time.perf_counter()
        subprocess.run(args.cmd, shell=True, check=True)
        wall_clock = time.perf_counter() - started
        summary_path = Path(args.out_capture)
    elif args.summary:
        summary_path = Path(args.summary)
    else:
        raise SystemExit("gate mode requires either --cmd (+ --out-capture) or --summary")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return perf_common.parse_gate_summary(
        payload,
        gate_name=args.gate_name,
        summary_path=str(summary_path),
        wall_clock_sec=wall_clock,
        hostname=args.hostname,
    )


# --- gpu-util mode ---------------------------------------------------------------


def _run_gpu_util_mode(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.live:
        result = subprocess.run(
            ["nvidia-smi", "dmon", "-s", "pucm", "-c", "1"],
            capture_output=True,
            text=True,
            check=True,
        )
        text = result.stdout
    elif args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        raise SystemExit(
            "gpu-util mode requires --live (host-only, needs nvidia-smi + a live GPU) "
            "or --input <captured `nvidia-smi dmon -s pucm` output>"
        )
    parsed = perf_common.parse_dmon_pucm(text)
    timestamp = _now_iso()
    rows: list[dict[str, Any]] = []
    for entry in parsed:
        sm = float(entry.get("sm") or 0.0)
        mem = float(entry.get("mem") or 0.0)
        gpu_index = entry.get("gpu")
        rows.append(
            {
                "kind": "gpu_util",
                "timestamp": timestamp,
                "hostname": args.hostname,
                "gpu_index": gpu_index,
                "sm_util_pct": sm,
                "mem_util_pct": mem,
                "context_thrash_flagged": perf_common.check_gpu_context_thrash(sm, mem),
                "raw": entry,
                "key": perf_common.stable_key("gpu_util", gpu_index, timestamp, os.getpid()),
            }
        )
    return rows


# --- CLI -----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", default=str(perf_common.DEFAULT_LEDGER_PATH))
    parser.add_argument("--no-ledger", action="store_true", help="print the row(s) but skip the ledger append")
    sub = parser.add_subparsers(dest="mode", required=True)

    leaf = sub.add_parser("leaf", help="per-leaf featurize/FFI/NN-forward/postprocess profiler")
    leaf.add_argument("--num-evals", type=int, default=64)
    leaf.add_argument("--seed", type=int, default=7)
    leaf.add_argument("--device", default="cpu")
    leaf.add_argument("--checkpoint", default=None, help="real EntityGraphPolicy checkpoint (required for a real baseline check)")
    leaf.add_argument("--public-observation", action="store_true")
    leaf.add_argument("--hostname", default=None, help="fleet host this snapshot was taken on (keeps regression grouping per-host, e.g. B200 vs A100A/A100B)")
    leaf.add_argument(
        "--rust-featurize",
        action="store_true",
        help="use the native Rust featurizer path (config.rust_featurize, task #81 phase 2) instead of the Python path",
    )

    genlog = sub.add_parser("gen-log", help="rows/hr per host from a generate_gumbel_selfplay_data.py manifest.json")
    genlog.add_argument("--manifest", required=True)
    genlog.add_argument("--hostname", default=None)

    gate = sub.add_parser("gate", help="gate wall-clock + game-count + extension-tier tracking")
    gate.add_argument("--gate-name", required=True)
    gate.add_argument("--summary", default=None, help="an already-produced gate/H2H summary JSON")
    gate.add_argument("--cmd", default=None, help="(host-only) gate command to run and time; must itself write --out-capture")
    gate.add_argument("--out-capture", default=None, help="path the wrapped --cmd writes its summary JSON to")
    gate.add_argument("--hostname", default=None, help="fleet host this gate ran on (keeps regression grouping per-host, e.g. B200 vs A100A/A100B)")

    gpu = sub.add_parser("gpu-util", help="MPS/context-thrash monitoring via nvidia-smi dmon -s pucm")
    gpu.add_argument("--live", action="store_true", help="(host-only) invoke nvidia-smi dmon directly")
    gpu.add_argument("--input", default=None, help="path to captured `nvidia-smi dmon -s pucm -c 1` output")
    gpu.add_argument("--hostname", default=None, help="fleet host this sample was taken on (keeps regression grouping per-host, e.g. B200 vs A100A/A100B)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "leaf":
        rows = [
            profile_leaf_eval(
                num_evals=args.num_evals,
                seed=args.seed,
                device=args.device,
                checkpoint=args.checkpoint,
                public_observation=args.public_observation,
                hostname=args.hostname,
                rust_featurize=args.rust_featurize,
            )
        ]
    elif args.mode == "gen-log":
        rows = [_run_genlog_mode(args)]
    elif args.mode == "gate":
        rows = [_run_gate_mode(args)]
    elif args.mode == "gpu-util":
        rows = _run_gpu_util_mode(args)
    else:  # pragma: no cover - argparse `required=True` prevents this.
        raise SystemExit(f"unknown mode: {args.mode}")

    written = 0
    if not args.no_ledger:
        written = perf_common.append_ledger_rows(Path(args.ledger), rows)

    print(json.dumps({"mode": args.mode, "rows": rows, "written": written}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
