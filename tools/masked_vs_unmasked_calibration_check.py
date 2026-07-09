#!/usr/bin/env python3
"""Empirical resolution of the v3a/v3b masked-training-status question (task
#76 re-save precondition, 2026-07-05): since no artifact (checkpoint metadata
predating 96b2819, report.json schema, train.log) records whether
--mask-hidden-info was actually used, run each checkpoint's value head over
the SAME held-out states in BOTH regimes -- player_tokens as-stored
(unmasked/omniscient) and with mask_player_tokens_public applied (masked) --
and compare corr(q,z). A masked-trained net fed unmasked (in-distribution
mismatch) inputs should calibrate WORSE than in its native regime, and vice
versa for an omniscient-trained net. Whichever regime wins is the actual
training regime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from value_repair_calibration_probe import (  # type: ignore  # noqa: E402
    DEFAULT_HOLDOUT_BLOCKS,
    collect_holdout_rows,
    compute_q,
)
from catan_zero.rl.entity_token_features import mask_player_tokens_public
from catan_zero.rl.entity_token_policy import EntityGraphPolicy


def _corr(q: np.ndarray, z: np.ndarray) -> float:
    return float(np.corrcoef(q, z)[0, 1])


def evaluate_both_regimes(checkpoint: str, device: str, max_rows: int | None) -> dict:
    groups = collect_holdout_rows(DEFAULT_HOLDOUT_BLOCKS, max_rows=max_rows)
    policy = EntityGraphPolicy.load(checkpoint, device=device)

    # Unmasked (as-stored) regime.
    q_unmasked, z = compute_q(policy, groups)
    corr_unmasked = _corr(q_unmasked, z)

    # Masked regime: same groups, player_tokens replaced with the masked copy.
    masked_groups = []
    for group in groups:
        masked_group = dict(group)
        masked_group["player_tokens"] = mask_player_tokens_public(group["player_tokens"])
        masked_groups.append(masked_group)
    q_masked, z_masked = compute_q(policy, masked_groups)
    corr_masked = _corr(q_masked, z_masked)

    assert np.array_equal(z, z_masked), "z must be identical between the two regimes"

    verdict = "masked-trained" if corr_masked > corr_unmasked else "omniscient-trained"
    return {
        "checkpoint": checkpoint,
        "n_rows": int(len(z)),
        "corr_q_z_unmasked": corr_unmasked,
        "corr_q_z_masked": corr_masked,
        "delta_masked_minus_unmasked": corr_masked - corr_unmasked,
        "verdict": verdict,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = evaluate_both_regimes(args.checkpoint, args.device, args.max_rows)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
