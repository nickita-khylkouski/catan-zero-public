#!/usr/bin/env python3
"""Causal commissioning probe for legal-action target binding.

The probe deliberately does *not* measure playing strength.  It answers the
smaller P0-A admission question before an expensive learner arm is allowed:

1. Does the historical policy remain invariant when action target ids change?
2. Does an ``action_target_gather`` warm start remain function-identical?
3. Does supervised policy loss put a non-zero gradient into the new branch?
4. After a few branch-only steps, do logits and loss respond to target identity
   and to the entity token stored at that target?

Real native Catan roots are used for BUILD_SETTLEMENT, BUILD_ROAD and
MOVE_ROBBER.  The short optimization updates only ``target_gather_proj``; it is
an architecture wiring probe, not a candidate checkpoint or strength result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    policy_entity_feature_adapter_version,
)
from catan_zero.rl.entity_token_policy import EntityGraphPolicy  # noqa: E402
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    _policy_history_options,
    rust_action_context_batch,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)
from catan_zero.search.rust_mcts import _require_rust_module  # noqa: E402
from factory_common import write_json  # noqa: E402

COLORS: tuple[str, ...] = ("RED", "BLUE")
TARGET_ACTION_TYPES: tuple[str, ...] = (
    "BUILD_SETTLEMENT",
    "BUILD_ROAD",
    "MOVE_ROBBER",
)
_TARGET_COLUMN_BY_ACTION_TYPE = {
    "BUILD_SETTLEMENT": 1,
    "BUILD_CITY": 1,
    "BUILD_ROAD": 2,
    "MOVE_ROBBER": 0,
}
_TOKEN_KEY_BY_TARGET_COLUMN = {
    0: "hex_tokens",
    1: "vertex_tokens",
    2: "edge_tokens",
    3: "player_tokens",
}
_MASK_KEY_BY_TARGET_COLUMN = {
    0: "hex_mask",
    1: "vertex_mask",
    2: "edge_mask",
    3: "player_mask",
}


class ProbeError(RuntimeError):
    """The causal probe could not establish its preconditions."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone_arrays(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in batch.items()}


def permute_action_targets(
    entity: Mapping[str, np.ndarray],
    action_types: Sequence[str],
) -> tuple[dict[str, np.ndarray], int]:
    """Rotate same-type target ids while leaving action rows unchanged.

    Keeping action features and legal ids fixed makes this a direct intervention
    on action-to-entity identity.  Per-type rotation avoids assigning a road an
    invalid vertex id or a settlement an invalid edge id.
    """

    result = _clone_arrays(entity)
    targets = np.asarray(result["legal_action_target_ids"])
    if targets.ndim != 3 or targets.shape[0] != 1:
        raise ProbeError(
            "target permutation expects one unpadded root [1, actions, 4]"
        )
    if len(action_types) != int(targets.shape[1]):
        raise ProbeError("action type count does not match legal action rows")

    changed = 0
    for action_type in sorted(set(str(item) for item in action_types)):
        column = _TARGET_COLUMN_BY_ACTION_TYPE.get(action_type)
        if column is None:
            continue
        rows = [
            row
            for row, row_type in enumerate(action_types)
            if str(row_type) == action_type and int(targets[0, row, column]) >= 0
        ]
        if len(rows) < 2:
            continue
        values = targets[0, rows, column].copy()
        rotated = np.roll(values, 1)
        if np.array_equal(values, rotated):
            continue
        targets[0, rows, column] = rotated
        changed += int(np.count_nonzero(values != rotated))
    return result, changed


def permute_target_tokens(
    entity: Mapping[str, np.ndarray],
    action_types: Sequence[str],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Reverse spatial token rows addressed by the root's target-bearing actions.

    Target ids stay fixed.  A target-aware action therefore receives a different
    entity representation, while the legacy permutation-invariant CLS path does
    not gain any positional signal.
    """

    result = _clone_arrays(entity)
    columns = {
        _TARGET_COLUMN_BY_ACTION_TYPE[action_type]
        for action_type in action_types
        if action_type in _TARGET_COLUMN_BY_ACTION_TYPE
    }
    changed_keys: list[str] = []
    for column in sorted(columns):
        # Player-row reversal changes actor/opponent semantics rather than board
        # topology. MOVE_ROBBER still exercises its hex target here; the victim
        # player target remains untouched.
        if column == 3:
            continue
        token_key = _TOKEN_KEY_BY_TARGET_COLUMN[column]
        tokens = np.asarray(result[token_key])
        result[token_key] = tokens[:, ::-1, :].copy()
        mask_key = _MASK_KEY_BY_TARGET_COLUMN[column]
        if mask_key in result:
            result[mask_key] = np.asarray(result[mask_key])[:, ::-1].copy()
        changed_keys.append(token_key)
    return result, tuple(changed_keys)


def _action_types(game: Any) -> tuple[str, ...]:
    raw_actions = json.loads(game.playable_actions_json())
    return tuple(str(action[1]) for action in raw_actions)


def collect_target_roots(
    *,
    roots_per_type: int,
    base_seed: int,
    max_ticks_per_seed: int,
) -> list[tuple[str, int, int, Any]]:
    """Collect bounded native roots containing at least two same-type targets."""

    if roots_per_type <= 0:
        raise ProbeError("roots_per_type must be positive")
    rust = _require_rust_module()
    found: dict[str, list[tuple[str, int, int, Any]]] = {
        action_type: [] for action_type in TARGET_ACTION_TYPES
    }
    seed = int(base_seed)
    # MOVE_ROBBER is stochastic and less frequent.  A generous but finite seed
    # ceiling keeps the probe bounded and fails visibly if engine behavior drifts.
    seed_limit = seed + max(32, roots_per_type * 16)
    while seed < seed_limit and any(
        len(found[action_type]) < roots_per_type
        for action_type in TARGET_ACTION_TYPES
    ):
        game = rust.Game.simple(list(COLORS), seed=seed)
        for tick in range(int(max_ticks_per_seed)):
            if game.winning_color() is not None:
                break
            action_types = _action_types(game)
            for action_type in TARGET_ACTION_TYPES:
                if len(found[action_type]) >= roots_per_type:
                    continue
                if sum(item == action_type for item in action_types) < 2:
                    continue
                found[action_type].append((action_type, seed, tick, game.copy()))
            if all(
                len(found[action_type]) >= roots_per_type
                for action_type in TARGET_ACTION_TYPES
            ):
                break
            game.play_tick()
        seed += 1

    missing = {
        action_type: roots_per_type - len(roots)
        for action_type, roots in found.items()
        if len(roots) < roots_per_type
    }
    if missing:
        raise ProbeError(f"could not collect requested target roots: {missing}")
    return [
        root
        for action_type in TARGET_ACTION_TYPES
        for root in found[action_type][:roots_per_type]
    ]


def _root_inputs(
    policy: EntityGraphPolicy,
    game: Any,
    *,
    context_fill: float = 0.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, tuple[str, ...]]:
    history_enabled, history_limit, history_schema = _policy_history_options(policy)
    adapter_version = policy_entity_feature_adapter_version(policy)
    public_observation = bool(
        getattr(policy, "trained_with_masked_hidden_info", False)
    )
    actor = str(game.current_color())
    legal_native_ids = tuple(
        int(action)
        for action in game.playable_action_indices(list(COLORS), None)
    )
    raw_action_types = _action_types(game)
    if len(raw_action_types) != len(legal_native_ids):
        raise ProbeError("native legal ids and action JSON are not aligned")
    policy_action_ids = rust_policy_action_ids(
        game,
        legal_native_ids,
        colors=COLORS,
        action_size=int(policy.action_size),
    )
    entity = rust_game_to_entity_batch(
        game,
        legal_native_ids,
        actor=actor,
        colors=COLORS,
        action_size=int(policy.action_size),
        policy_action_ids=policy_action_ids,
        public_observation=public_observation,
        meaningful_public_history=history_enabled,
        history_limit=history_limit,
        meaningful_public_history_schema=history_schema,
        entity_feature_adapter_version=adapter_version,
    )
    context = rust_action_context_batch(
        game,
        legal_native_ids,
        actor=actor,
        colors=COLORS,
        action_size=int(policy.action_size),
        policy_action_ids=policy_action_ids,
        fill=float(context_fill),
        public_observation=public_observation,
        entity_feature_adapter_version=adapter_version,
    )
    legal_ids = np.asarray(policy_action_ids, dtype=np.int64)[None, :]
    return entity, legal_ids, context, raw_action_types


def _warmstart_gather(base: EntityGraphPolicy, *, seed: int) -> EntityGraphPolicy:
    if bool(getattr(base.config, "action_target_gather", False)):
        raise ProbeError(
            "source checkpoint already enables action_target_gather; use an "
            "uncommissioned baseline so B0/G1 attribution remains causal"
        )
    config = dataclasses.replace(base.config, action_target_gather=True)
    static = base.static_action_features.detach().cpu().numpy()
    treatment = EntityGraphPolicy(
        config,
        static,
        seed=int(seed),
        device=str(base.device),
        entity_feature_adapter_version=policy_entity_feature_adapter_version(base),
    )
    treatment.trained_with_masked_hidden_info = bool(
        getattr(base, "trained_with_masked_hidden_info", False)
    )
    treatment.public_award_feature_contract = str(
        base.public_award_feature_contract
    )
    treatment.entity_feature_adapter_binding_source = str(
        getattr(base, "entity_feature_adapter_binding_source", "legacy_policy")
    )
    missing, unexpected = treatment.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    disallowed = [
        name for name in missing if not name.startswith("target_gather_proj.")
    ]
    if unexpected or disallowed:
        raise ProbeError(
            f"warm-start mismatch: unexpected={unexpected[:8]} "
            f"disallowed_missing={disallowed[:8]}"
        )
    treatment.model.eval()
    return treatment


def _cross_entropy(logits: Any, target: int) -> Any:
    import torch.nn.functional as functional

    return functional.cross_entropy(
        logits,
        logits.new_tensor([int(target)], dtype=None).long(),
    )


def _forward(
    policy: EntityGraphPolicy,
    entity: Mapping[str, np.ndarray],
    legal_ids: np.ndarray,
    context: np.ndarray,
) -> Any:
    return policy.forward_legal_np(
        dict(entity),
        legal_ids,
        context,
        return_q=False,
        return_final_vp=False,
        return_aux_subgoals=False,
    )["logits"]


def _preferred_action_index(base_logits: Any, action_types: Sequence[str]) -> int:
    """Use the baseline's best target-bearing action as a realistic hard label."""

    import torch

    candidates = [
        index
        for index, action_type in enumerate(action_types)
        if action_type in _TARGET_COLUMN_BY_ACTION_TYPE
    ]
    if not candidates:
        raise ProbeError("root has no target-bearing action")
    candidate_tensor = torch.as_tensor(candidates, device=base_logits.device)
    local = torch.argmax(base_logits[0].index_select(0, candidate_tensor))
    return int(candidate_tensor[int(local)].item())


def _l2_norm(values: Iterable[Any]) -> float:
    total = 0.0
    for value in values:
        if value is None:
            continue
        total += float(value.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def _max_abs_delta(left: Any, right: Any) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _mean_abs_delta(left: Any, right: Any) -> float:
    return float((left.detach() - right.detach()).abs().mean().item())


def run_probe(
    *,
    checkpoint: Path,
    device: str,
    roots: Sequence[tuple[str, int, int, Any]],
    learning_rate: float,
    steps: int,
    initialization_seed: int,
    context_fill: float = 0.0,
) -> dict[str, Any]:
    """Run the B0/G1 causal sequence and return a JSON-serializable report."""

    import torch

    if steps <= 0:
        raise ProbeError("steps must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ProbeError("learning_rate must be positive and finite")

    torch.manual_seed(int(initialization_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(initialization_seed))
    base = EntityGraphPolicy.load(str(checkpoint), device=device)
    base.model.eval()
    treatment = _warmstart_gather(base, seed=initialization_seed)

    prepared = []
    for action_type, game_seed, tick, game in roots:
        entity, legal_ids, context, action_types = _root_inputs(
            base, game, context_fill=float(context_fill)
        )
        target_permuted, changed_target_count = permute_action_targets(
            entity, action_types
        )
        topology_permuted, changed_token_keys = permute_target_tokens(
            entity, action_types
        )
        if changed_target_count <= 0:
            raise ProbeError(
                f"{action_type} root seed={game_seed} has no permutable targets"
            )
        if not changed_token_keys:
            raise ProbeError(
                f"{action_type} root seed={game_seed} has no spatial target tokens"
            )
        prepared.append(
            {
                "action_type": action_type,
                "game_seed": int(game_seed),
                "tick": int(tick),
                "entity": entity,
                "target_permuted": target_permuted,
                "topology_permuted": topology_permuted,
                "changed_target_count": changed_target_count,
                "changed_token_keys": changed_token_keys,
                "legal_ids": legal_ids,
                "context": context,
                "action_types": action_types,
            }
        )

    initial_rows = []
    for row in prepared:
        with torch.no_grad():
            base_logits = _forward(
                base, row["entity"], row["legal_ids"], row["context"]
            )
            base_target_permuted = _forward(
                base,
                row["target_permuted"],
                row["legal_ids"],
                row["context"],
            )
            treatment_logits = _forward(
                treatment, row["entity"], row["legal_ids"], row["context"]
            )
            treatment_target_permuted = _forward(
                treatment,
                row["target_permuted"],
                row["legal_ids"],
                row["context"],
            )
        target = _preferred_action_index(base_logits, row["action_types"])
        row["target"] = target
        initial_rows.append(
            {
                "action_type": row["action_type"],
                "game_seed": row["game_seed"],
                "tick": row["tick"],
                "legal_width": int(row["legal_ids"].shape[1]),
                "changed_target_count": int(row["changed_target_count"]),
                "changed_token_keys": list(row["changed_token_keys"]),
                "supervised_action_row": int(target),
                "baseline_target_permutation_max_abs_logit_delta": (
                    _max_abs_delta(base_logits, base_target_permuted)
                ),
                "warmstart_base_max_abs_logit_delta": _max_abs_delta(
                    base_logits, treatment_logits
                ),
                "warmstart_target_permutation_max_abs_logit_delta": (
                    _max_abs_delta(
                        treatment_logits, treatment_target_permuted
                    )
                ),
            }
        )

    parameters = list(treatment.model.target_gather_proj.parameters())
    optimizer = torch.optim.SGD(parameters, lr=float(learning_rate))
    optimizer.zero_grad(set_to_none=True)
    admission_loss = None
    for row in prepared:
        logits = _forward(
            treatment, row["entity"], row["legal_ids"], row["context"]
        )
        root_loss = _cross_entropy(logits, row["target"])
        admission_loss = (
            root_loss if admission_loss is None else admission_loss + root_loss
        )
    assert admission_loss is not None
    admission_loss = admission_loss / len(prepared)
    admission_loss.backward()
    admission_gradient_l2 = _l2_norm(parameter.grad for parameter in parameters)
    if not math.isfinite(admission_gradient_l2) or admission_gradient_l2 <= 0.0:
        raise ProbeError(
            "action_target_gather admitted no finite supervised gradient"
        )

    initial_parameter_values = [parameter.detach().clone() for parameter in parameters]
    losses = [float(admission_loss.detach().item())]
    optimizer.step()
    for _ in range(1, int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = None
        for row in prepared:
            logits = _forward(
                treatment, row["entity"], row["legal_ids"], row["context"]
            )
            root_loss = _cross_entropy(logits, row["target"])
            loss = root_loss if loss is None else loss + root_loss
        assert loss is not None
        loss = loss / len(prepared)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    parameter_delta_l2 = _l2_norm(
        parameter.detach() - initial
        for parameter, initial in zip(parameters, initial_parameter_values)
    )
    final_rows = []
    for row in prepared:
        with torch.no_grad():
            logits = _forward(
                treatment, row["entity"], row["legal_ids"], row["context"]
            )
            target_permuted = _forward(
                treatment,
                row["target_permuted"],
                row["legal_ids"],
                row["context"],
            )
            topology_permuted = _forward(
                treatment,
                row["topology_permuted"],
                row["legal_ids"],
                row["context"],
            )
            original_loss = _cross_entropy(logits, row["target"])
            target_permuted_loss = _cross_entropy(
                target_permuted, row["target"]
            )
            topology_permuted_loss = _cross_entropy(
                topology_permuted, row["target"]
            )
        final_rows.append(
            {
                "action_type": row["action_type"],
                "game_seed": row["game_seed"],
                "tick": row["tick"],
                "target_permutation_max_abs_logit_delta": _max_abs_delta(
                    logits, target_permuted
                ),
                "target_permutation_mean_abs_logit_delta": _mean_abs_delta(
                    logits, target_permuted
                ),
                "target_permutation_loss_delta": float(
                    target_permuted_loss.item() - original_loss.item()
                ),
                "topology_permutation_max_abs_logit_delta": _max_abs_delta(
                    logits, topology_permuted
                ),
                "topology_permutation_mean_abs_logit_delta": _mean_abs_delta(
                    logits, topology_permuted
                ),
                "topology_permutation_loss_delta": float(
                    topology_permuted_loss.item() - original_loss.item()
                ),
                "selected_action_changed_by_target_permutation": bool(
                    int(torch.argmax(logits, dim=-1).item())
                    != int(torch.argmax(target_permuted, dim=-1).item())
                ),
                "selected_action_changed_by_topology_permutation": bool(
                    int(torch.argmax(logits, dim=-1).item())
                    != int(torch.argmax(topology_permuted, dim=-1).item())
                ),
            }
        )

    baseline_invariant = all(
        row["baseline_target_permutation_max_abs_logit_delta"] == 0.0
        for row in initial_rows
    )
    warmstart_identical = all(
        row["warmstart_base_max_abs_logit_delta"] == 0.0
        and row["warmstart_target_permutation_max_abs_logit_delta"] == 0.0
        for row in initial_rows
    )
    target_sensitive = all(
        row["target_permutation_max_abs_logit_delta"] > 0.0
        for row in final_rows
    )
    topology_sensitive = all(
        row["topology_permutation_max_abs_logit_delta"] > 0.0
        for row in final_rows
    )
    passed = (
        baseline_invariant
        and warmstart_identical
        and admission_gradient_l2 > 0.0
        and parameter_delta_l2 > 0.0
        and target_sensitive
        and topology_sensitive
    )
    return {
        "schema_version": "action-target-binding-probe/v1",
        "scope": "bounded_architecture_wiring_not_strength_evidence",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
            "source_action_target_gather": bool(
                getattr(base.config, "action_target_gather", False)
            ),
            "entity_feature_adapter_version": (
                policy_entity_feature_adapter_version(base)
            ),
            "public_observation": bool(
                getattr(base, "trained_with_masked_hidden_info", False)
            ),
            "meaningful_public_history": bool(_policy_history_options(base)[0]),
            "meaningful_public_history_schema": str(
                _policy_history_options(base)[2]
            ),
            "event_history_limit": int(_policy_history_options(base)[1]),
            "action_context_fill": float(context_fill),
            "public_award_feature_contract": str(
                base.public_award_feature_contract
            ),
        },
        "device": str(device),
        "commissioning": {
            "initialization_seed": int(initialization_seed),
            "optimizer": "SGD",
            "learning_rate": float(learning_rate),
            "steps": int(steps),
            "updated_parameters": "target_gather_proj_only",
            "target_identity_encoding": "typed-local-id-sinusoid-v1",
            "d6_identity_contract": (
                "encoding consumes post-symmetry legal_action_target_ids; "
                "no pre-symmetry/cached target identity is used"
            ),
            "loss": "hard_cross_entropy_to_baseline_best_target_bearing_action",
        },
        "measurements": {
            "initial_loss": float(losses[0]),
            "last_pre_step_loss": float(losses[-1]),
            "admission_gradient_l2": float(admission_gradient_l2),
            "target_gather_parameter_delta_l2": float(parameter_delta_l2),
            "initial_roots": initial_rows,
            "commissioned_roots": final_rows,
        },
        "criteria": {
            "baseline_target_invariant": baseline_invariant,
            "warmstart_function_identical": warmstart_identical,
            "target_gather_gradient_admitted": admission_gradient_l2 > 0.0,
            "target_gather_parameters_updated": parameter_delta_l2 > 0.0,
            "all_roots_target_sensitive": target_sensitive,
            "all_roots_topology_sensitive": topology_sensitive,
        },
        "passed": passed,
    }


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    measurements = report["measurements"]
    roots = measurements["commissioned_roots"]
    return {
        "passed": bool(report["passed"]),
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "device": report["device"],
        "roots": len(roots),
        "action_types": sorted({row["action_type"] for row in roots}),
        "admission_gradient_l2": measurements["admission_gradient_l2"],
        "target_gather_parameter_delta_l2": measurements[
            "target_gather_parameter_delta_l2"
        ],
        "min_target_permutation_max_abs_logit_delta": min(
            row["target_permutation_max_abs_logit_delta"] for row in roots
        ),
        "min_topology_permutation_max_abs_logit_delta": min(
            row["topology_permutation_max_abs_logit_delta"] for row in roots
        ),
        "criteria": report["criteria"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roots-per-type", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=610001)
    parser.add_argument("--max-ticks-per-seed", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--initialization-seed", type=int, default=20260716)
    parser.add_argument("--context-fill", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    roots = collect_target_roots(
        roots_per_type=int(args.roots_per_type),
        base_seed=int(args.base_seed),
        max_ticks_per_seed=int(args.max_ticks_per_seed),
    )
    report = run_probe(
        checkpoint=args.checkpoint,
        device=str(args.device),
        roots=roots,
        learning_rate=float(args.learning_rate),
        steps=int(args.steps),
        initialization_seed=int(args.initialization_seed),
        context_fill=float(args.context_fill),
    )
    write_json(args.out, report)
    print(json.dumps(_summary(report), indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
