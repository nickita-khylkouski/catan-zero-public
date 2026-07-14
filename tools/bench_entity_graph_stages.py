#!/usr/bin/env python3
"""Deep strict-FP32 CUDA profile of one EntityGraph inference window.

The benchmark has two deliberately separate measurements:

* ``exact`` calls ``EntityGraphPolicy.forward_legal_np`` and performs the same
  output copies as EvalServer.  This is the authoritative window latency.
* ``attributed`` reconstructs that forward from the model's existing modules,
  placing CUDA events around projections, sequence assembly, every transformer
  block, action/value/Q heads, masking, and D2H.  Its outputs are checked
  against the exact call before any timings are reported.

Fine-grained numbers are attribution, not an additive replacement for the
exact number: CUDA-event insertion and isolated synchronization perturb very
small kernels.  The synthetic tensors match production dtypes/shapes but are
not a self-play throughput or strength benchmark.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np


_NON_MODEL_ENTITY_KEYS = frozenset(
    {
        "hex_vertex_ids",
        "hex_edge_ids",
        "edge_vertex_ids",
        "event_target_ids",
        "legal_action_mask",
    }
)


def _synthetic_batch(
    *,
    batch_size: int,
    legal_width: int,
    valid_legal_fraction: float,
    event_width: int,
    valid_players: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build EvalServer-shaped host arrays (FP16 entities, FP32 context)."""
    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import (
        EDGE_FEATURE_SIZE,
        EVENT_FEATURE_SIZE,
        GLOBAL_FEATURE_SIZE,
        HEX_FEATURE_SIZE,
        LEGAL_ACTION_FEATURE_SIZE,
        PLAYER_FEATURE_SIZE,
        VERTEX_FEATURE_SIZE,
    )

    if batch_size <= 0 or legal_width <= 0 or event_width < 0:
        raise ValueError("batch/legal widths must be positive and event width non-negative")
    if not 0.0 < valid_legal_fraction <= 1.0:
        raise ValueError("valid_legal_fraction must be within (0, 1]")
    if not 1 <= valid_players <= 4:
        raise ValueError("valid_players must be within [1, 4]")

    rng = np.random.default_rng(seed)
    entity: dict[str, np.ndarray] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", event_width, EVENT_FEATURE_SIZE),
    ):
        entity[f"{name}_tokens"] = rng.normal(
            size=(batch_size, count, width)
        ).astype(np.float16)
        if name != "global":
            entity[f"{name}_mask"] = np.ones((batch_size, count), dtype=np.bool_)
    entity["player_mask"][:, valid_players:] = False
    # event0 is the production frontier.  A nonzero width is supported for a
    # control arm, but all synthetic event positions remain masked.
    entity["event_mask"][:] = False
    entity["legal_action_tokens"] = rng.normal(
        size=(batch_size, legal_width, LEGAL_ACTION_FEATURE_SIZE)
    ).astype(np.float16)
    entity["legal_action_target_ids"] = np.full(
        (batch_size, legal_width, 4), -1, dtype=np.int16
    )
    context = rng.normal(
        size=(batch_size, legal_width, CONTEXT_ACTION_FEATURE_SIZE)
    ).astype(np.float32)

    # Vary the live prefix slightly by row while preserving the requested mean.
    mean_live = max(1, min(legal_width, round(legal_width * valid_legal_fraction)))
    spread = max(1, min(mean_live - 1, legal_width - mean_live, legal_width // 8))
    live_counts = mean_live + rng.integers(-spread, spread + 1, size=batch_size)
    live_counts = np.clip(live_counts, 1, legal_width)
    legal_ids = np.full((batch_size, legal_width), -1, dtype=np.int64)
    for row, count in enumerate(live_counts):
        legal_ids[row, : int(count)] = np.arange(int(count), dtype=np.int64)
    entity["legal_action_mask"] = legal_ids >= 0
    return entity, legal_ids, context


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in samples)
    if not ordered:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "min": math.nan}
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "min": float(ordered[0]),
    }


class _StageRecorder:
    """Record one pipeline iteration without synchronizing between stages."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.events: list[tuple[str, Any, Any]] = []
        self.host_ms: dict[str, float] = defaultdict(float)

    def run(self, name: str, call: Callable[[], Any]) -> Any:
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        host_start = time.perf_counter()
        result = call()
        self.host_ms[name] += (time.perf_counter() - host_start) * 1000.0
        end.record()
        self.events.append((name, start, end))
        return result

    def collect(self) -> tuple[dict[str, float], dict[str, float]]:
        if self.events:
            self.events[-1][2].synchronize()
        device_ms: dict[str, float] = defaultdict(float)
        for name, start, end in self.events:
            device_ms[name] += float(start.elapsed_time(end))
        return dict(device_ms), dict(self.host_ms)


def _host_to_device(policy: Any, entity: dict[str, np.ndarray], legal_ids: np.ndarray, context: np.ndarray, torch: Any):
    needs_targets = bool(
        getattr(policy.config, "action_target_gather", False)
        or getattr(policy.config, "edge_policy_head", False)
    )
    batch = {
        key: torch.as_tensor(value, device=policy.device)
        for key, value in entity.items()
        if key not in _NON_MODEL_ENTITY_KEYS
        and (key != "legal_action_target_ids" or needs_targets)
    }
    batch["legal_action_context"] = torch.as_tensor(
        context, dtype=torch.float32, device=policy.device
    )
    action_ids = torch.as_tensor(legal_ids, dtype=torch.long, device=policy.device)
    return batch, action_ids


def _attributed_forward(
    policy: Any,
    entity: dict[str, np.ndarray],
    legal_ids: np.ndarray,
    context: np.ndarray,
    *,
    return_q: bool,
    return_aux_subgoals: bool,
    recorder: _StageRecorder,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Semantics-equivalent decomposition of the current eager forward."""
    from catan_zero.rl.entity_token_policy import _apply_public_award_feature_contract

    torch = recorder.torch
    model = policy.model
    # Exact inference applies the checkpoint-owned public-award compatibility
    # bridge before any tensor transfer.  Keep attribution on the same inputs;
    # otherwise legacy checkpoints receive a random, never-trained player
    # column here and the profiler silently measures a different function.
    entity = _apply_public_award_feature_contract(
        entity, policy.public_award_feature_contract
    )
    batch, action_ids = recorder.run(
        "numpy_to_torch_h2d",
        lambda: _host_to_device(policy, entity, legal_ids, context, torch),
    )

    batch_size = batch["hex_tokens"].shape[0]
    pieces = []
    cls_piece = recorder.run(
        "expand_cls", lambda: model.cls_token.expand(batch_size, -1, -1)
    )
    pieces.append(
        recorder.run(
            "add_cls_type", lambda: cls_piece + model.type_embedding[0].view(1, 1, -1)
        )
    )
    for index, name in enumerate(("hex", "vertex", "edge", "player", "global", "event"), start=1):
        encoder = getattr(model, f"{name}_encoder")
        if name == "event" and batch["event_tokens"].shape[1] == 0:
            pieces.append(
                recorder.run(
                    "event0_empty_short_circuit",
                    lambda: model.type_embedding.new_empty(
                        (batch_size, 0, model.type_embedding.shape[1])
                    ),
                )
            )
            continue
        projected = recorder.run(
            f"project_{name}",
            lambda name=name, encoder=encoder: encoder(batch[f"{name}_tokens"].float()),
        )
        pieces.append(
            recorder.run(
                f"add_{name}_type",
                lambda projected=projected, index=index: projected
                + model.type_embedding[index].view(1, 1, -1),
            )
        )

    def assemble_state():
        tokens = torch.cat(pieces, dim=1)
        masks = [
            torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device),
            ~batch["hex_mask"].bool(),
            ~batch["vertex_mask"].bool(),
            ~batch["edge_mask"].bool(),
            ~batch["player_mask"].bool(),
            torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device),
            ~batch["event_mask"].bool(),
        ]
        return tokens, torch.cat(masks, dim=1)

    tokens, padding_mask = recorder.run("sequence_and_mask_assembly", assemble_state)
    # Expand _Block.forward exactly, rather than treating each transformer as
    # an opaque 2-ms box.  The final exact-vs-attributed bit-parity assertion
    # protects this internal attribution against implementation drift.
    for block_index, block in enumerate(model.blocks):
        prefix = f"transformer_{block_index}"
        attn_in = recorder.run(
            f"{prefix}_norm_attn", lambda: block.norm_attn(tokens)
        )
        attn_out = recorder.run(
            f"{prefix}_self_attention",
            lambda: block.attn(
                attn_in,
                attn_in,
                attn_in,
                key_padding_mask=padding_mask,
                need_weights=False,
            )[0],
        )
        tokens = recorder.run(
            f"{prefix}_attention_residual", lambda: tokens + attn_out
        )
        ff_in = recorder.run(
            f"{prefix}_norm_ff", lambda: block.norm_ff(tokens)
        )
        ff_out = recorder.run(f"{prefix}_feed_forward", lambda: block.ff(ff_in))
        tokens = recorder.run(
            f"{prefix}_ff_residual", lambda: tokens + ff_out
        )
    state = recorder.run("state_norm_extract", lambda: model.state_norm(tokens[:, 0]))

    action_features = recorder.run(
        "action_feature_concat",
        lambda: torch.cat(
            (batch["legal_action_tokens"].float(), batch["legal_action_context"].float()),
            dim=-1,
        ),
    )
    encoded_actions = recorder.run(
        "action_encoder", lambda: model.action_encoder(action_features)
    )

    pooled_targets = None
    if model.action_target_gather or model.edge_policy_head:
        pooled_targets = recorder.run(
            "action_target_gather", lambda: model._gather_target_tokens(tokens, batch)
        )
    if model.action_target_gather:
        encoded_actions = recorder.run(
            "action_target_projection",
            lambda: encoded_actions + model.target_gather_proj(pooled_targets),
        )
    if model.action_cross_attention_layers > 0:
        for index, cross_block in enumerate(model.action_cross_blocks):
            encoded_actions = recorder.run(
                f"action_cross_attention_{index}",
                lambda encoded_actions=encoded_actions, cross_block=cross_block: cross_block(
                    encoded_actions, tokens, key_padding_mask=padding_mask
                ),
            )

    def policy_score():
        policy_state = torch.nn.functional.normalize(state, dim=-1)
        policy_actions = torch.nn.functional.normalize(encoded_actions, dim=-1)
        scale = torch.clamp(model.logit_scale.exp(), max=50.0)
        result = scale * (policy_state.unsqueeze(1) * policy_actions).sum(dim=-1)
        result = result + model.action_bias(action_features).squeeze(-1)
        if model.edge_policy_head:
            result = result + model.edge_policy_mlp(pooled_targets).squeeze(-1)
        return result

    logits = recorder.run("policy_normalize_dot_bias", policy_score)
    value = recorder.run("value_head", lambda: model.value_head(state).squeeze(-1))
    if model.value_attention_pool:
        value = recorder.run(
            "value_attention_pool", lambda: value + model._value_pool(state, tokens, padding_mask)
        )
    final_vp = recorder.run(
        "final_vp_head", lambda: model.final_vp_head(state).squeeze(-1)
    )
    outputs: dict[str, Any] = {"logits": logits, "value": value, "final_vp": final_vp}

    if model.value_uncertainty_head is not None:
        outputs["value_uncertainty"] = recorder.run(
            "value_uncertainty_head",
            lambda: torch.nn.functional.softplus(
                model.value_uncertainty_head(state.detach()).squeeze(-1)
            ),
        )
    if model.value_categorical_head is not None:
        def categorical_heads():
            cat_logits = model.value_categorical_head(state)
            probabilities = torch.softmax(cat_logits.float(), dim=-1)
            win_probs = probabilities[..., : model.value_categorical_bins]
            win_mass = win_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            result = {
                "value_categorical_logits": cat_logits,
                "value_categorical": (
                    (win_probs / win_mass) * model.value_categorical_support
                ).sum(dim=-1),
            }
            if model.value_categorical_truncation_class:
                result["value_categorical_truncation_prob"] = probabilities[..., -1]
            return result
        outputs.update(recorder.run("value_categorical_head", categorical_heads))
    if model.aux_subgoal_heads and return_aux_subgoals:
        def aux_heads():
            result = {
                "aux_longest_road": model.aux_longest_road_head(state).squeeze(-1),
                "aux_largest_army": model.aux_largest_army_head(state).squeeze(-1),
                "aux_vp_in_n": model.aux_vp_in_n_head(state).squeeze(-1),
                "aux_robber_target": model.aux_robber_target_head(state),
            }
            if bool(getattr(model, "aux_settlement_pointer_head_enabled", False)):
                vertex_count = int(batch["vertex_tokens"].shape[1])
                vertex_start = 1 + int(batch["hex_tokens"].shape[1])
                result["aux_next_settlement"] = (
                    model.aux_next_settlement_pointer_head(
                        tokens[:, vertex_start : vertex_start + vertex_count]
                    ).squeeze(-1)
                )
            else:
                result["aux_next_settlement"] = model.aux_next_settlement_head(state)
            return result
        outputs.update(recorder.run("aux_subgoal_heads", aux_heads))
    if return_q:
        q_features = recorder.run(
            "q_feature_assembly",
            lambda: torch.cat(
                (
                    state.unsqueeze(1).expand_as(encoded_actions),
                    encoded_actions,
                    state.unsqueeze(1).expand_as(encoded_actions) * encoded_actions,
                ),
                dim=-1,
            ),
        )
        outputs["q_values"] = recorder.run(
            "q_head", lambda: model.q_head(q_features).squeeze(-1)
        )

    valid = recorder.run("legal_mask_prepare", lambda: action_ids >= 0)
    outputs["logits"] = recorder.run(
        "legal_logit_mask", lambda: outputs["logits"].masked_fill(~valid, -1.0e9)
    )

    def device_to_host():
        keys = ("logits", "value", "value_uncertainty", "q_values")
        return {
            key: outputs[key].detach().float().cpu().numpy()
            for key in keys
            if key in outputs
        }

    host_outputs = recorder.run("device_to_host_sync", device_to_host)
    return outputs, host_outputs


def _exact_forward(
    policy: Any,
    entity: dict[str, np.ndarray],
    legal_ids: np.ndarray,
    context: np.ndarray,
    *,
    return_q: bool,
    return_aux_subgoals: bool,
):
    outputs = policy.forward_legal_np(
        entity,
        legal_ids,
        context,
        return_q=return_q,
        return_aux_subgoals=return_aux_subgoals,
    )
    keys = ("logits", "value", "value_uncertainty", "q_values")
    return outputs, {
        key: outputs[key].detach().float().cpu().numpy()
        for key in keys
        if key in outputs
    }


def _compare_outputs(reference: dict[str, Any], candidate: dict[str, Any], torch: Any) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for key in sorted(reference):
        if key not in candidate:
            raise RuntimeError(f"attributed forward omitted output {key}")
        left = reference[key].detach().float()
        right = candidate[key].detach().float()
        if left.shape != right.shape:
            raise RuntimeError(f"output shape mismatch for {key}: {left.shape} != {right.shape}")
        delta = (left - right).abs()
        comparison[key] = {
            "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        }
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    if reference.keys() != candidate.keys():
        raise RuntimeError(
            f"output key mismatch: exact={sorted(reference)} attributed={sorted(candidate)}"
        )
    return comparison


def _benchmark_exact(call: Callable[[], Any], *, warmup: int, iterations: int, torch: Any) -> dict[str, Any]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        call()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
        gpu_samples.append(float(start.elapsed_time(end)))
    return {"cuda_ms": _summary(gpu_samples), "wall_ms": _summary(wall_samples)}


def _benchmark_attributed(call: Callable[[_StageRecorder], Any], *, warmup: int, iterations: int, torch: Any) -> dict[str, Any]:
    for _ in range(warmup):
        recorder = _StageRecorder(torch)
        call(recorder)
        recorder.collect()
    device_samples: dict[str, list[float]] = defaultdict(list)
    host_samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        recorder = _StageRecorder(torch)
        call(recorder)
        device, host = recorder.collect()
        for key, value in device.items():
            device_samples[key].append(value)
        for key, value in host.items():
            host_samples[key].append(value)
    return {
        name: {
            "cuda_ms": _summary(samples),
            "host_call_ms": _summary(host_samples.get(name, [])),
        }
        for name, samples in device_samples.items()
    }


def _transformer_block_totals(stages: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for name, stage in stages.items():
        if not name.startswith("transformer_"):
            continue
        parts = name.split("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            totals[f"transformer_block_{parts[1]}"] += stage["cuda_ms"]["mean"]
    return dict(sorted(totals.items()))


def _stage_family_totals(stages: dict[str, Any]) -> dict[str, float]:
    families: dict[str, float] = defaultdict(float)
    for name, stage in stages.items():
        value = stage["cuda_ms"]["mean"]
        if name.startswith("transformer_"):
            family = "transformer_blocks"
        elif name.startswith("project_") or name.startswith("add_") or name == "expand_cls":
            family = "entity_projection_and_type"
        elif name.startswith("action_") or name.startswith("policy_"):
            family = "action_policy_head"
        elif name.startswith("q_"):
            family = "q_head_and_preparation"
        elif "value" in name or name in {"state_norm_extract", "final_vp_head"}:
            family = "state_and_value_heads"
        elif name.startswith("legal_"):
            family = "legal_mask"
        else:
            family = name
        families[family] += value
    return dict(sorted(families.items(), key=lambda item: item[1], reverse=True))


def _preloaded_forward(model: Any, batch: dict[str, Any], action_ids: Any, *, return_q: bool, crop_players: int | None, no_padding_mask: bool):
    if crop_players is not None:
        batch = dict(batch)
        batch["player_tokens"] = batch["player_tokens"][:, :crop_players]
        batch["player_mask"] = batch["player_mask"][:, :crop_players]
    tokens, padding_mask = model._state_tokens(batch)
    if no_padding_mask:
        # The host-side TTFF/event0 invariant is validated once by
        # _ab_player_crop.  Do not call padding_mask.any().item() here: that
        # would introduce a device synchronization into every timed forward.
        padding_mask = None
    for block in model.blocks:
        tokens = block(tokens, key_padding_mask=padding_mask)
    state = model.state_norm(tokens[:, 0])
    # Target-aware heads need fixed player offsets and are intentionally kept
    # out of this structural A/B until separately proven safe.
    outputs = model.score_actions((tokens, padding_mask, state), batch, return_q=return_q)
    outputs["logits"] = outputs["logits"].masked_fill(~(action_ids >= 0), -1.0e9)
    return outputs


def _ab_player_crop(policy: Any, entity: dict[str, np.ndarray], legal_ids: np.ndarray, context: np.ndarray, *, return_q: bool, warmup: int, iterations: int, torch: Any) -> dict[str, Any]:
    if bool(getattr(policy.config, "action_target_gather", False) or getattr(policy.config, "edge_policy_head", False)):
        return {"skipped": "target-aware action heads require a separate offset audit"}
    batch, action_ids = _host_to_device(policy, entity, legal_ids, context, torch)
    if entity["event_tokens"].shape[1] != 0:
        return {"skipped": "A/B requires event_width=0"}
    if not np.all(entity["player_mask"][:, :2]) or np.any(entity["player_mask"][:, 2:]):
        return {"skipped": "A/B requires the exact TTFF player mask invariant"}

    variants = {
        "baseline_player4_ttff_mask": (None, False),
        "player2_all_false_mask": (2, False),
        "player2_no_padding_mask": (2, True),
    }
    outputs = {
        name: _preloaded_forward(
            policy.model, batch, action_ids, return_q=return_q,
            crop_players=crop, no_padding_mask=no_mask,
        )
        for name, (crop, no_mask) in variants.items()
    }
    timings = {
        name: _benchmark_exact(
            lambda crop=crop, no_mask=no_mask: _preloaded_forward(
                policy.model, batch, action_ids, return_q=return_q,
                crop_players=crop, no_padding_mask=no_mask,
            ),
            warmup=warmup, iterations=iterations, torch=torch,
        )
        for name, (crop, no_mask) in variants.items()
    }
    baseline = outputs["baseline_player4_ttff_mask"]
    drift = {}
    for name, candidate in outputs.items():
        if name == "baseline_player4_ttff_mask":
            continue
        metrics = _compare_outputs_tolerant(baseline, candidate, torch)
        drift[name] = metrics
    baseline_ms = timings["baseline_player4_ttff_mask"]["cuda_ms"]["mean"]
    for name in timings:
        candidate_ms = timings[name]["cuda_ms"]["mean"]
        timings[name]["speedup_vs_baseline"] = baseline_ms / candidate_ms
    return {"timings": timings, "strict_fp32_output_drift": drift}


def _compare_outputs_tolerant(reference: dict[str, Any], candidate: dict[str, Any], torch: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    common = sorted(reference.keys() & candidate.keys())
    for key in common:
        left = reference[key].detach().float()
        right = candidate[key].detach().float()
        delta = (left - right).abs()
        entry: dict[str, Any] = {
            "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        }
        if key == "logits":
            valid = left > -1.0e8
            left_argmax = left.argmax(dim=-1)
            right_argmax = right.argmax(dim=-1)
            entry["argmax_agreement"] = float((left_argmax == right_argmax).float().mean().item())
            if bool(valid.any().item()):
                left_probs = torch.softmax(left.masked_fill(~valid, -1.0e9), dim=-1)
                right_probs = torch.softmax(right.masked_fill(~valid, -1.0e9), dim=-1)
                entry["mean_policy_l1"] = float((left_probs - right_probs).abs().sum(dim=-1).mean().item())
        metrics[key] = entry
    return metrics


def _print_flame(result: dict[str, Any]) -> None:
    stages = result["attributed_stages"]
    total = sum(stage["cuda_ms"]["mean"] for stage in stages.values())
    print(f"Exact forward + EvalServer D2H: {result['exact_window']['cuda_ms']['mean']:.3f} ms/window")
    print(f"Attributed event sum:          {total:.3f} ms/window")
    print()
    print("Stage families:")
    for name, value in result["attributed_stage_families_cuda_ms"].items():
        percent = 100.0 * value / total if total else 0.0
        bar = "█" * max(1, round(percent / 2.5))
        print(f"{name:31s} {value:8.3f} ms {percent:6.2f}%  {bar}")
    print("\nTransformer blocks:")
    for name, value in result["transformer_block_totals_cuda_ms"].items():
        percent = 100.0 * value / total if total else 0.0
        bar = "█" * max(1, round(percent / 2.5))
        print(f"{name:31s} {value:8.3f} ms {percent:6.2f}%  {bar}")
    print("\nDeep stages:")
    for name, stage in sorted(stages.items(), key=lambda item: item[1]["cuda_ms"]["mean"], reverse=True):
        value = stage["cuda_ms"]["mean"]
        percent = 100.0 * value / total if total else 0.0
        bar = "█" * max(1, round(percent / 2.5))
        print(f"{name:31s} {value:8.3f} ms {percent:6.2f}%  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--legal-width", type=int, default=54)
    parser.add_argument("--valid-legal-fraction", type=float, default=0.392)
    parser.add_argument("--event-width", type=int, default=0)
    parser.add_argument("--valid-players", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--return-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--return-aux-subgoals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include learner-only CAT-100 outputs in the timed inference window.",
    )
    parser.add_argument("--player-crop-ab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    import torch
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this benchmark requires an available CUDA device")
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--warmup and --iterations must be positive")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = EntityGraphPolicy.load(args.checkpoint, device=str(device))
    policy.model.eval().float()
    entity, legal_ids, context = _synthetic_batch(
        batch_size=args.batch_size,
        legal_width=args.legal_width,
        valid_legal_fraction=args.valid_legal_fraction,
        event_width=args.event_width,
        valid_players=args.valid_players,
        seed=args.seed,
    )

    with torch.inference_mode():
        exact_outputs, _ = _exact_forward(
            policy,
            entity,
            legal_ids,
            context,
            return_q=args.return_q,
            return_aux_subgoals=args.return_aux_subgoals,
        )
        parity_recorder = _StageRecorder(torch)
        attributed_outputs, _ = _attributed_forward(
            policy, entity, legal_ids, context,
            return_q=args.return_q,
            return_aux_subgoals=args.return_aux_subgoals,
            recorder=parity_recorder,
        )
        parity_recorder.collect()
        parity = _compare_outputs(exact_outputs, attributed_outputs, torch)

        exact = _benchmark_exact(
            lambda: _exact_forward(
                policy,
                entity,
                legal_ids,
                context,
                return_q=args.return_q,
                return_aux_subgoals=args.return_aux_subgoals,
            ),
            warmup=args.warmup, iterations=args.iterations, torch=torch,
        )
        attributed = _benchmark_attributed(
            lambda recorder: _attributed_forward(
                policy, entity, legal_ids, context,
                return_q=args.return_q,
                return_aux_subgoals=args.return_aux_subgoals,
                recorder=recorder,
            ),
            warmup=args.warmup, iterations=args.iterations, torch=torch,
        )
        player_ab = (
            _ab_player_crop(
                policy, entity, legal_ids, context, return_q=args.return_q,
                warmup=args.warmup, iterations=args.iterations, torch=torch,
            )
            if args.player_crop_ab
            else {"skipped": "disabled by CLI"}
        )

    properties = torch.cuda.get_device_properties(device)
    result = {
        "device": properties.name,
        "checkpoint": str(args.checkpoint),
        "strict_fp32": {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "autocast": False,
        },
        "shape": {
            "batch_size": args.batch_size,
            "legal_width": args.legal_width,
            "real_legal_cells": int((legal_ids >= 0).sum()),
            "padded_legal_cells": int(legal_ids.size),
            "event_width": args.event_width,
            "valid_players": args.valid_players,
            "state_tokens": 1 + 19 + 54 + 72 + 4 + 1 + args.event_width,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "return_q": args.return_q,
        "return_aux_subgoals": args.return_aux_subgoals,
        "exact_window": exact,
        "attributed_stages": attributed,
        "attributed_stage_families_cuda_ms": _stage_family_totals(attributed),
        "transformer_block_totals_cuda_ms": _transformer_block_totals(attributed),
        "exact_vs_attributed_output_parity": parity,
        "player2_no_mask_ab": player_ab,
        "limitations": [
            "exact_window is the authoritative forward_legal_np plus EvalServer-style D2H boundary",
            "attributed stages reconstruct the same operations and are bit-exact parity checked, but CUDA events perturb tiny kernels",
            "host_call_ms is Python enqueue/blocking time; D2H host_call_ms includes waiting for queued upstream compute, while its cuda_ms isolates the copy on the device stream",
            "synthetic values match production shapes/dtypes but do not include merge, IPC, featurization, MCTS, or disk",
            "a sum of attributed stages should not replace the separately measured exact window latency",
        ],
    }
    _print_flame(result)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(f"\nJSON: {args.output_json}")
    else:
        print("\n--- JSON ---")
        print(payload)


if __name__ == "__main__":
    main()
