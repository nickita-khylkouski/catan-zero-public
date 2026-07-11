#!/usr/bin/env python3
"""Summarize py-spy raw output from the Catan evaluation harnesses.

The report separates the main game/search thread from the asynchronous neural
evaluator thread.  This matters for ``py-spy --idle --threads`` captures:
otherwise the evaluator's intentionally idle queue wait is counted as useful
wall time and makes all percentages misleading.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _category(thread: str, stack: str) -> str:
    if "rust-mcts-batched-evaluator" in thread and (
        "queue.py" in stack or "threading.py" in stack
    ):
        return "evaluator_queue_idle"
    if "neural_rust_mcts.py:1057" in stack and "threading.py:629" in stack:
        return "neural_evaluator_wait"
    if any(
        token in stack
        for token in (
            "forward_legal_np",
            "entity_token_policy.py:669",
            "entity_token_policy.py:821",
            "site-packages/torch/",
            "libtorch",
            "wrapper_CUDA",
        )
    ):
        return "neural_forward"
    if any(
        token in stack
        for token in (
            "entity_token_features.py",
            "action_context_features.py",
            "rust_game_to_entity_batch",
            "rust_action_context_batch",
            "_fetch_leaf_decision_inputs",
            "_resolve_entity_adapter",
        )
    ):
        return "feature_encoding_and_leaf_ffi"
    if any(
        token in stack
        for token in (
            "sync_from_native",
            "apply_native_action_record_to_rust",
            "audit_current_game",
        )
    ):
        return "python_rust_referee_sync"
    if any(
        token in stack
        for token in (
            "/catanatron/players/value.py",
            "/catanatron/players/minimax.py",
        )
    ):
        return "external_bot_policy"
    # Once the native hot loop is enabled, ``_search_information_set`` remains
    # as the Python PIMC particle orchestrator.  It is not tree traversal.  A
    # sample that has crossed into the extension belongs to native traversal /
    # allocation; samples between particle calls are Python orchestration.
    if "catan_zero/search/native_gumbel_mcts.py" in stack:
        if "catanatron_rs" in stack or "gumbel_mcts::" in stack:
            return "native_mcts_traversal_and_allocator"
        return "python_pimc_orchestration"
    if (
        "catan_zero/search/gumbel_chance_mcts.py" in stack
        or "catan_zero/search/rust_mcts.py" in stack
    ):
        return "python_mcts_traversal"
    if "/vendor/catanatron/" in stack:
        return "python_catanatron_referee"
    if any(
        token in stack
        for token in ("multiprocessing/", "threading.py", "queue.py", "waitpid")
    ):
        return "process_thread_wait_or_ipc"
    if any(token in stack for token in ("importlib", "dlopen", "EntityGraphPolicy.load")):
        return "startup_import_checkpoint"
    if "catanatron_rs" in stack:
        return "other_rust_ffi"
    return "other_runtime"


def analyze(path: Path) -> dict[str, Any]:
    threads: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    inclusive_main: Counter[str] = Counter()
    total = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line or raw_line.startswith("Warning:"):
            continue
        try:
            stack, count_text = raw_line.rsplit(" ", 1)
            count = int(count_text)
        except (ValueError, IndexError):
            continue
        fields = stack.split(";")
        thread = fields[1] if len(fields) > 1 else "unknown"
        frames = fields[2:]
        joined = ";".join(frames)
        total += count
        threads[thread] += count
        categories[_category(thread, joined)] += count
        if "MainThread" in thread:
            for frame in set(frames):
                inclusive_main[frame] += count

    main_total = sum(value for key, value in threads.items() if "MainThread" in key)
    category_rows = []
    for name, samples in categories.most_common():
        denominator = (
            sum(
                value
                for category, value in categories.items()
                if category != "evaluator_queue_idle"
            )
            if name != "evaluator_queue_idle"
            else total
        )
        category_rows.append(
            {
                "category": name,
                "samples": samples,
                "pct": 100.0 * samples / denominator if denominator else 0.0,
            }
        )
    return {
        "source": str(path),
        "samples_total_all_threads": total,
        "samples_main_thread": main_total,
        "threads": [
            {"thread": key, "samples": value, "pct_all_threads": 100.0 * value / total}
            for key, value in threads.most_common()
        ],
        "exclusive_categories": category_rows,
        "inclusive_hotspots": [
            {
                "frame": key,
                "samples": value,
                "pct_main_thread": 100.0 * value / main_total if main_total else 0.0,
            }
            for key, value in inclusive_main.most_common(40)
            if "rust-mcts-batched-evaluator" not in key
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = analyze(args.raw)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
