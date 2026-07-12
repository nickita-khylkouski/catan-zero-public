#!/usr/bin/env python3
"""Zero-cost structural probes for the incumbent entity-graph policy.

The probe uses synthetic features because the claims are architectural: the
incumbent dense Transformer does not consume action target ids or board
incidence, and is permutation invariant within each entity-token type.  A
target-gather control activates that branch's zero-initialized output layer to
show that the same target-id perturbation becomes observable when wired in.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def probe_config():
    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_policy import EntityGraphConfig

    return EntityGraphConfig(
        action_size=567,
        static_action_feature_size=45,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=64,
        state_layers=2,
        attention_heads=4,
        dropout=0.0,
    )


def synthetic_batch(config: Any, *, seed: int = 20260712) -> dict[str, Any]:
    import torch

    from catan_zero.rl.entity_token_features import (
        EDGE_FEATURE_SIZE,
        EVENT_FEATURE_SIZE,
        GLOBAL_FEATURE_SIZE,
        HEX_FEATURE_SIZE,
        LEGAL_ACTION_FEATURE_SIZE,
        PLAYER_FEATURE_SIZE,
        VERTEX_FEATURE_SIZE,
    )

    generator = torch.Generator().manual_seed(seed)
    batch: dict[str, Any] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", 8, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(2, count, width, generator=generator)
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(2, count, dtype=torch.bool)
    batch["legal_action_tokens"] = torch.randn(
        2, 7, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        2, 7, int(config.context_action_feature_size), generator=generator
    )
    targets = torch.full((2, 7, 4), -1, dtype=torch.long)
    targets[:, :, 1] = torch.arange(7)
    batch["legal_action_target_ids"] = targets
    return batch


def changed_targets(batch: dict[str, Any]) -> dict[str, Any]:
    import torch

    changed = {key: value.clone() for key, value in batch.items()}
    changed["legal_action_target_ids"][:, :, 1] = torch.roll(
        changed["legal_action_target_ids"][:, :, 1], shifts=1, dims=1
    )
    return changed


def permuted_board_tokens(batch: dict[str, Any]) -> dict[str, Any]:
    """Permute complete token rows without changing the feature multiset."""
    import torch

    changed = {key: value.clone() for key, value in batch.items()}
    changed["vertex_tokens"] = changed["vertex_tokens"][:, torch.arange(53, -1, -1)]
    changed["vertex_mask"] = changed["vertex_mask"][:, torch.arange(53, -1, -1)]
    changed["edge_tokens"] = changed["edge_tokens"][:, torch.arange(71, -1, -1)]
    changed["edge_mask"] = changed["edge_mask"][:, torch.arange(71, -1, -1)]
    return changed


def _max_output_diff(
    model: Any, left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    import torch

    model.eval()
    with torch.inference_mode():
        lhs = model(left, return_q=True)
        rhs = model(right, return_q=True)
    return {
        key: float((lhs[key] - rhs[key]).abs().max().item())
        for key in ("logits", "value", "final_vp", "q_values")
    }


def run_probe() -> dict[str, Any]:
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphNet

    torch.manual_seed(20260712)
    config = probe_config()
    incumbent = EntityGraphNet(config)
    batch = synthetic_batch(config)
    target_changed = changed_targets(batch)
    token_permuted = permuted_board_tokens(batch)

    gather = EntityGraphNet(dataclasses.replace(config, action_target_gather=True))
    gather.load_state_dict(incumbent.state_dict(), strict=False)
    # The warm-start branch is deliberately a no-op at initialization. Activate
    # only its output projection so this control measures target-id wiring.
    with torch.no_grad():
        projection = gather.target_gather_proj[1]
        projection.weight.copy_(torch.eye(config.hidden_size))
        projection.bias.zero_()

    return {
        "schema_version": "entity-graph-structural-invariance-probe/v1",
        "incumbent_target_id_diff": _max_output_diff(incumbent, batch, target_changed),
        "incumbent_token_permutation_diff": _max_output_diff(
            incumbent, batch, token_permuted
        ),
        "enabled_gather_target_id_diff": _max_output_diff(
            gather, batch, target_changed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_probe()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
