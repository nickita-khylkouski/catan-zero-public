#!/usr/bin/env python3
"""Construct reviewed entity-graph checkpoint topology transitions.

Historical modes in this tool are function-preserving: they warm-start shared
weights and construct only zero-output adapters or an exact policy-suffix clone
for the late value tower.  The explicitly named adapter-v6 *information
migration* is different: it changes the observation surface, measures that
change on deterministic anchors, and writes non-promotable migration
provenance.  It must never be described or consumed as function-preserving.

This is the *mechanical enabler* for the v3b finetune: `tools/train_bc.py` has
no CLI argument for the new EntityGraphConfig flags, and its `--init-checkpoint`
path rebuilds the module from the checkpoint's own pickled config. So instead of
touching train_bc, we produce an upgraded-config checkpoint here and point the
IDENTICAL v3a command at it via `--init-checkpoint`. train_bc's
`EntityGraphPolicy.load` then reads the upgraded config (flags ON), builds the
upgraded module, and loads these weights strictly (the new zero-init params are
already present in this checkpoint, so nothing is missing).

Historical upgrade outputs are behaviorally identical at initialization.  The
adapter-v6 migration asserts the opposite and records its measured policy and
value drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_SRC = (_TOOLS_DIR.parent / "src").resolve(strict=True)
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _REPO_SRC]
sys.path.insert(0, str(_REPO_SRC))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V4,
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
)
from catan_zero.rl.entity_token_features import (  # noqa: E402
    PUBLIC_RULE_STATE_FEATURE_SCHEMA_VERSION,
)
from catan_zero.rl.meaningful_history import (  # noqa: E402
    MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
    MEANINGFUL_PUBLIC_HISTORY_V2_LIMIT,
)
from catan_zero.rl.ordered_history import ORDERED_ATTENTION_V2  # noqa: E402

_ENTITY_POLICY_MODULE = sys.modules[EntityGraphPolicy.__module__]
_ENTITY_POLICY_PATH = Path(str(_ENTITY_POLICY_MODULE.__file__)).resolve(strict=True)
if _REPO_SRC not in _ENTITY_POLICY_PATH.parents:
    raise RuntimeError(
        "checkpoint upgrader imported catan_zero outside its checkout: "
        f"{_ENTITY_POLICY_PATH}"
    )

# The exact param prefixes introduced by the three upgrades (see
# entity_token_policy.EntityGraphNet). Must equal the load() allow-list.
NEW_PARAM_PREFIXES = (
    "target_gather_proj.",
    "action_cross_blocks.",
    "value_probe",
    "value_pool_head.",
    # CAT-97 edge-feature policy head + CAT-100 aux subgoal heads.
    "edge_policy_mlp.",
    "aux_longest_road_head.",
    "aux_largest_army_head.",
    "aux_vp_in_n_head.",
    "aux_next_settlement_head.",
    "aux_robber_target_head.",
    "aux_next_settlement_pointer_head.",
    "belief_resource_head.",
    "value_categorical_head.",
    "topology_residual_adapter.",
    "static_action_residual_proj.",
    "legal_action_value_residual_proj.",
    "legal_action_value_static_proj.",
    "legal_action_value_max_proj.",
    "legal_action_value_count_proj.",
    "legal_action_value_static_max_proj.",
    "public_card_count_residual.",
    "meaningful_history_residual_gate",
    "meaningful_history_ordered_gate",
    "meaningful_history_sequence.",
    "meaningful_history_target_proj.",
    "value_blocks.",
    "value_state_norm.",
    "public_rule_state_residual.",
)


def _build_upgraded_config(
    base_config, overrides: dict[str, object]
) -> EntityGraphConfig:
    """Reconstruct an EntityGraphConfig from a possibly-STALE base config.

    `dataclasses.replace(base_config, **overrides)` reads EVERY current field
    off `base_config`, so a config pickled before a field existed makes it
    raise AttributeError. This is exactly what happens to a seed checkpoint
    whose config predates both the f69 flags AND other later fields (e.g.
    f67's value_uncertainty_head): replace tries to read a field the stale
    object never had. Instead: copy the fields that DO exist, let the dataclass
    fill any the stale pickle lacks from its current defaults, then apply the
    flag overrides. Correct for arbitrary past-or-future config drift.
    """
    base_dict = {
        f.name: getattr(base_config, f.name)
        for f in fields(EntityGraphConfig)
        if hasattr(base_config, f.name)
    }
    base_dict.update(overrides)
    return EntityGraphConfig(**base_dict)


def _parse_flags(raw: str) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for entry in (piece.strip() for piece in raw.split(",") if piece.strip()):
        if entry in ("current_v5_split1", "current-v5-split1"):
            # Canonical direct legacy-incumbent -> commissioned parent
            # topology. Keep this one token so operators cannot accidentally
            # omit a zero-output module or request the historical default
            # value split of two layers.
            overrides.update(
                {
                    "action_target_gather": True,
                    "static_action_residual": True,
                    "legal_action_value_residual": True,
                    "legal_action_value_set_statistics": True,
                    "public_card_count_features": True,
                    "public_card_count_residual_bias": False,
                    "meaningful_public_history": True,
                    "meaningful_public_history_schema": (
                        MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
                    ),
                    "event_history_limit": MEANINGFUL_PUBLIC_HISTORY_V2_LIMIT,
                    "meaningful_public_history_pooling": ORDERED_ATTENTION_V2,
                    "meaningful_public_history_target_gather": True,
                    "public_rule_state_features": True,
                    "public_rule_state_feature_schema": (
                        PUBLIC_RULE_STATE_FEATURE_SCHEMA_VERSION
                    ),
                    "value_tower_split_layers": 1,
                }
            )
        elif entry in (
            "current_v5_topology_split1",
            "current-v5-topology-split1",
        ):
            # Reviewed topology-aware successor to current_v5_split1. This is
            # still only an initializer construction; science admission remains
            # fail-closed until the checked-in recipe's independent gates pass.
            overrides.update(
                {
                    **_parse_flags("current_v5_split1"),
                    "topology_residual_adapter": True,
                }
            )
        elif entry in (
            "current_v6_information_migration_topology_split1",
            "current-v6-information-migration-topology-split1",
        ):
            # This is deliberately NOT a function-preserving upgrade. Adapter
            # v6 changes real feature values (exact actor resources and the
            # corrected initial-road two-hop context). The construction shares
            # the reviewed topology with v5, but is issued only through the
            # explicit information-contract migration receipt.
            overrides.update(
                {
                    **_parse_flags("current_v5_topology_split1"),
                }
            )
        elif entry in (
            "current_v6_topology_split1",
            "current-v6-topology-split1",
        ):
            raise SystemExit(
                "adapter-v6 is not function preserving; use "
                "current_v6_information_migration_topology_split1"
            )
        elif entry in ("gather", "action_target_gather"):
            overrides["action_target_gather"] = True
        elif entry in ("value", "value_attention_pool"):
            overrides["value_attention_pool"] = True
        elif entry.startswith("cross"):
            n = entry.split(":", 1)[1] if ":" in entry else "2"
            overrides["action_cross_attention_layers"] = int(n)
        elif entry in ("edge", "edge_policy_head"):
            overrides["edge_policy_head"] = True
        elif entry in ("aux", "aux_subgoal_heads"):
            overrides["aux_subgoal_heads"] = True
        elif entry in (
            "aux_settlement_pointer",
            "aux_settlement_pointer_head",
        ):
            overrides["aux_subgoal_heads"] = True
            overrides["aux_settlement_pointer_head"] = True
        elif entry in ("topology", "topology_residual_adapter"):
            overrides["topology_residual_adapter"] = True
        elif entry in ("belief", "belief_resource_head"):
            overrides["belief_resource_head"] = True
        elif entry in ("static", "static_action_residual"):
            overrides["static_action_residual"] = True
        elif entry in (
            "structured_action_value",
            "legal_action_value_residual",
        ):
            overrides["static_action_residual"] = True
            overrides["legal_action_value_residual"] = True
        elif entry in ("card_count", "public_card_count_features"):
            overrides["public_card_count_features"] = True
        elif entry in (
            "card_count_v2",
            "public_card_count_features_v2",
            "bias_free_card_count",
        ):
            overrides["public_card_count_features"] = True
            overrides["public_card_count_residual_bias"] = False
        elif entry in (
            "history",
            "meaningful_history",
            "meaningful_public_history",
        ):
            overrides["meaningful_public_history"] = True
            overrides["meaningful_public_history_schema"] = (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            )
            overrides["event_history_limit"] = MEANINGFUL_PUBLIC_HISTORY_LIMIT
        elif entry in (
            "ordered_history",
            "meaningful_history_ordered",
            "meaningful_public_history_ordered",
        ):
            overrides["meaningful_public_history"] = True
            overrides["meaningful_public_history_schema"] = (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
            )
            overrides["event_history_limit"] = MEANINGFUL_PUBLIC_HISTORY_LIMIT
            overrides["meaningful_public_history_pooling"] = ORDERED_ATTENTION_V2
        elif entry in (
            "history_v2",
            "meaningful_history_v2",
            "meaningful_public_history_v2",
        ):
            overrides["meaningful_public_history"] = True
            overrides["meaningful_public_history_schema"] = (
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
            )
            overrides["event_history_limit"] = MEANINGFUL_PUBLIC_HISTORY_V2_LIMIT
            overrides["meaningful_public_history_pooling"] = ORDERED_ATTENTION_V2
        elif entry in (
            "history_target_gather",
            "meaningful_history_target_gather",
        ):
            overrides["meaningful_public_history_target_gather"] = True
        elif entry in (
            "legal_action_value_set_statistics",
            "value_set_statistics",
            "legal_set_statistics",
        ):
            overrides["legal_action_value_residual"] = True
            overrides["legal_action_value_set_statistics"] = True
        elif entry in (
            "public_rule_state",
            "public_rule_state_features",
            "actor_public_rule_state",
        ):
            overrides["public_rule_state_features"] = True
            overrides["public_rule_state_feature_schema"] = (
                PUBLIC_RULE_STATE_FEATURE_SCHEMA_VERSION
            )
        elif entry.startswith("catbins"):
            # CAT-39: build the HL-Gauss categorical value head with N win-loss
            # bins (plus the truncation class, which the config enables by
            # default). Zero-initialised? No -- but purely ADDITIVE: the scalar
            # value/final_vp/q outputs stay bit-identical (the new head only adds
            # value_categorical* outputs), so the forward-identity assertion below
            # still holds on those keys.
            n = entry.split(":", 1)[1] if ":" in entry else "33"
            overrides["value_categorical_bins"] = int(n)
        elif entry == "value_tower_split1":
            overrides["value_tower_split_layers"] = 1
        elif entry.startswith(("value_split", "value_tower")):
            n = entry.split(":", 1)[1] if ":" in entry else "2"
            overrides["value_tower_split_layers"] = int(n)
        else:
            raise SystemExit(f"unknown upgrade flag: {entry!r}")
    return overrides


def _verify_forward_identical(
    base: EntityGraphPolicy, upgraded: EntityGraphPolicy, device: str
) -> float:
    """Max abs diff of logits/q over one real 54-wide placement root."""
    import torch

    from catan_zero.search.neural_rust_mcts import (
        rust_action_context_batch,
        rust_game_to_entity_batch,
        rust_policy_action_ids,
    )
    from catan_zero.search.rust_mcts import _require_rust_module
    from sigma_trace_placement_root import COLORS, find_placement_roots

    catanatron_rs = _require_rust_module()
    game = find_placement_roots(catanatron_rs, n_states=1, base_seed=500001)[0]
    acting_color = str(game.current_color())
    legal_actions = tuple(
        int(a) for a in game.playable_action_indices(list(COLORS), None)
    )
    pids = rust_policy_action_ids(
        game, legal_actions, colors=COLORS, action_size=int(base.action_size)
    )
    base_entity = rust_game_to_entity_batch(
        game,
        legal_actions,
        actor=acting_color,
        colors=COLORS,
        action_size=int(base.action_size),
        policy_action_ids=pids,
    )
    upgraded_entity = rust_game_to_entity_batch(
        game,
        legal_actions,
        actor=acting_color,
        colors=COLORS,
        action_size=int(base.action_size),
        policy_action_ids=pids,
        meaningful_public_history=bool(
            getattr(upgraded.config, "meaningful_public_history", False)
        ),
        history_limit=int(getattr(upgraded.config, "event_history_limit", 64)),
        entity_feature_adapter_version=upgraded.entity_feature_adapter_version,
    )
    context = rust_action_context_batch(
        game,
        legal_actions,
        actor=acting_color,
        colors=COLORS,
        action_size=int(base.action_size),
        policy_action_ids=pids,
    )
    legal_ids = np.asarray(pids, dtype=np.int64)[None, :]
    max_diff = 0.0
    with torch.no_grad():
        ob = base.forward_legal_np(base_entity, legal_ids, context, return_q=True)
        ou = upgraded.forward_legal_np(
            upgraded_entity, legal_ids, context, return_q=True
        )
        for key in ("logits", "value", "final_vp", "q_values"):
            max_diff = max(max_diff, float((ob[key] - ou[key]).abs().max().item()))
    return max_diff


def _apply_deterministic_first_action(game, *, colors: tuple[str, ...]):
    """Advance one real game edge with deterministic chance resolution."""

    ids = [int(value) for value in game.playable_action_indices(list(colors), None)]
    actions = json.loads(game.playable_actions_json())
    if not ids or len(ids) != len(actions):
        raise RuntimeError("migration anchor root has malformed legal actions")
    action_id, action = ids[0], actions[0]
    spectrum = json.loads(game.spectrum_json(json.dumps(action)))
    if spectrum:
        outcome = max(
            range(len(spectrum)),
            key=lambda index: (
                float(spectrum[index].get("probability", 0.0)),
                -index,
            ),
        )
        return game.apply_chance_outcome(json.dumps(action), outcome)
    game.execute_action_index(action_id, list(colors), None)
    return game


def _migration_anchor_roots(catanatron_rs) -> list[tuple[str, object]]:
    """Build deterministic roots that exercise both v6 feature corrections."""

    colors = ("RED", "BLUE")
    game = catanatron_rs.Game.simple(list(colors), seed=123)
    roots: list[tuple[str, object]] = []
    saw_initial_road = False
    saw_resource_initial_road = False
    saw_resource_play_turn = False
    for _step in range(96):
        snapshot = json.loads(game.json_snapshot())
        phase = str(snapshot.get("current_prompt", ""))
        actor = str(game.current_color())
        player = json.loads(game.player_state_json(actor))
        resources = player.get("resources", [])
        resource_total = sum(int(value) for value in resources)
        legal = tuple(
            int(value)
            for value in game.playable_action_indices(list(colors), None)
        )
        if not roots:
            roots.append(("opening_settlement", game.copy()))
        if phase == "BUILD_INITIAL_ROAD" and not saw_initial_road:
            roots.append(("initial_road", game.copy()))
            saw_initial_road = True
        if (
            phase == "BUILD_INITIAL_ROAD"
            and resource_total > 0
            and not saw_resource_initial_road
        ):
            roots.append(("resource_initial_road", game.copy()))
            saw_resource_initial_road = True
        if (
            phase == "PLAY_TURN"
            and resource_total > 0
            and len(legal) > 1
            and not saw_resource_play_turn
        ):
            roots.append(("resource_play_turn", game.copy()))
            saw_resource_play_turn = True
        if saw_initial_road and saw_resource_initial_road and saw_resource_play_turn:
            break
        game = _apply_deterministic_first_action(game, colors=colors)
        if game.winning_color() is not None:
            break
    if not (saw_initial_road and saw_resource_initial_road and saw_resource_play_turn):
        raise RuntimeError(
            "deterministic migration anchors did not cover initial-road and "
            "resource-bearing states"
        )
    return roots


def _migration_anchor_evidence(
    base: EntityGraphPolicy, upgraded: EntityGraphPolicy, device: str
) -> dict[str, object]:
    """Measure the deliberate v2->v6 surface change without calling it parity.

    Every policy receives features and action contexts built using its own
    adapter. Parameter-topology construction is proven independently by the
    migration receipt's deterministic parameter replay; constructing a V2
    "shadow" with the V6 history topology would itself violate the adapter
    contract and is intentionally forbidden.
    """

    import torch

    from catan_zero.search.neural_rust_mcts import (
        rust_action_context_batch,
        rust_game_to_entity_batch,
        rust_policy_action_ids,
    )
    from catan_zero.search.rust_mcts import _require_rust_module
    from sigma_trace_placement_root import COLORS

    source_adapter = str(base.entity_feature_adapter_version)
    target_adapter = str(upgraded.entity_feature_adapter_version)
    if (
        source_adapter != RUST_ENTITY_ADAPTER_V2
        or target_adapter != RUST_ENTITY_ADAPTER_V6
    ):
        raise RuntimeError(
            "information migration requires the exact incumbent v2 adapter and v6 target"
        )
    rows: list[dict[str, object]] = []
    migration_max_diff = 0.0
    feature_max_diff = 0.0
    feature_changed_values = 0
    forward_kls: list[float] = []
    reverse_kls: list[float] = []
    value_errors: list[float] = []
    top1_flips = 0
    catanatron_rs = _require_rust_module()
    for label, game in _migration_anchor_roots(catanatron_rs):
        actor = str(game.current_color())
        legal_actions = tuple(
            int(value)
            for value in game.playable_action_indices(list(COLORS), None)
        )
        pids = rust_policy_action_ids(
            game,
            legal_actions,
            colors=COLORS,
            action_size=int(base.action_size),
        )

        def _surface(policy, adapter: str):
            entity = rust_game_to_entity_batch(
                game,
                legal_actions,
                actor=actor,
                colors=COLORS,
                action_size=int(policy.action_size),
                policy_action_ids=pids,
                public_observation=True,
                meaningful_public_history=bool(
                    getattr(policy.config, "meaningful_public_history", False)
                ),
                history_limit=int(getattr(policy.config, "event_history_limit", 64)),
                meaningful_public_history_schema=str(
                    getattr(
                        policy.config,
                        "meaningful_public_history_schema",
                        MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
                    )
                ),
                entity_feature_adapter_version=adapter,
            )
            context = rust_action_context_batch(
                game,
                legal_actions,
                actor=actor,
                colors=COLORS,
                action_size=int(policy.action_size),
                policy_action_ids=pids,
                public_observation=True,
                entity_feature_adapter_version=adapter,
            )
            legal_ids = np.asarray(pids, dtype=np.int64)[None, :]
            output = policy.forward_legal_np(
                entity, legal_ids, context, return_q=True
            )
            return entity, context, output

        base_entity, base_context, base_output = _surface(base, source_adapter)
        migrated_entity, migrated_context, migrated_output = _surface(
            upgraded, target_adapter
        )
        per_migration = 0.0
        output_diffs: dict[str, float] = {}
        for key in ("logits", "value", "final_vp", "q_values"):
            migration_diff = float(
                (base_output[key] - migrated_output[key]).abs().max().item()
            )
            if not np.isfinite(migration_diff):
                raise RuntimeError("non-finite migration anchor output")
            per_migration = max(per_migration, migration_diff)
            output_diffs[key] = migration_diff
        migration_max_diff = max(migration_max_diff, per_migration)

        base_policy = torch.softmax(base_output["logits"].float(), dim=-1)
        migrated_policy = torch.softmax(migrated_output["logits"].float(), dim=-1)
        epsilon = torch.finfo(torch.float32).tiny
        forward_kl = float(
            (
                base_policy
                * (
                    torch.log(base_policy.clamp_min(epsilon))
                    - torch.log(migrated_policy.clamp_min(epsilon))
                )
            )
            .sum()
            .item()
        )
        reverse_kl = float(
            (
                migrated_policy
                * (
                    torch.log(migrated_policy.clamp_min(epsilon))
                    - torch.log(base_policy.clamp_min(epsilon))
                )
            )
            .sum()
            .item()
        )
        value_error = float(
            (base_output["value"] - migrated_output["value"]).abs().max().item()
        )
        top1_flip = bool(
            int(torch.argmax(base_output["logits"], dim=-1).item())
            != int(torch.argmax(migrated_output["logits"], dim=-1).item())
        )
        if not all(np.isfinite(value) for value in (forward_kl, reverse_kl, value_error)):
            raise RuntimeError("non-finite migration policy/value metric")
        forward_kls.append(max(0.0, forward_kl))
        reverse_kls.append(max(0.0, reverse_kl))
        value_errors.append(value_error)
        top1_flips += int(top1_flip)

        per_feature_max = 0.0
        per_feature_changed = 0
        if set(base_entity) != set(migrated_entity):
            raise RuntimeError("adapter migration changed entity tensor keys")
        for key in sorted(base_entity):
            left = np.asarray(base_entity[key])
            right = np.asarray(migrated_entity[key])
            if left.shape != right.shape:
                raise RuntimeError(f"adapter migration changed {key} shape")
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            per_feature_max = max(per_feature_max, float(delta.max(initial=0.0)))
            per_feature_changed += int(np.count_nonzero(delta))
        context_delta = np.abs(
            base_context.astype(np.float64) - migrated_context.astype(np.float64)
        )
        per_feature_max = max(
            per_feature_max, float(context_delta.max(initial=0.0))
        )
        per_feature_changed += int(np.count_nonzero(context_delta))
        feature_max_diff = max(feature_max_diff, per_feature_max)
        feature_changed_values += per_feature_changed
        snapshot = json.loads(game.json_snapshot())
        anchor_identity = {
            "label": label,
            "actor": actor,
            # The Rust JSON snapshot contains map iteration order that is not a
            # cross-process identity contract. Bind the deterministic game and
            # root coordinates plus the exact legal/action-catalog surface.
            "game_seed": int(snapshot["seed"]),
            "state_index": int(snapshot["state_index"]),
            "current_turn_index": int(snapshot["current_turn_index"]),
            "phase": str(snapshot["current_prompt"]),
            "legal_action_ids": list(legal_actions),
            "policy_action_ids": list(pids),
        }
        anchor_sha256 = "sha256:" + hashlib.sha256(
            json.dumps(
                anchor_identity, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        resources = json.loads(game.player_state_json(actor)).get("resources", [])
        rows.append(
            {
                "label": label,
                "phase": str(snapshot.get("current_prompt", "")),
                "actor": actor,
                "actor_resource_total": sum(int(value) for value in resources),
                "legal_width": len(legal_actions),
                "migration_output_max_abs_diff": per_migration,
                "migration_output_max_abs_diff_by_key": output_diffs,
                "legal_policy_forward_kl": max(0.0, forward_kl),
                "legal_policy_reverse_kl": max(0.0, reverse_kl),
                "legal_policy_top1_flip": top1_flip,
                "scalar_value_abs_error": value_error,
                "feature_max_abs_diff": per_feature_max,
                "feature_changed_value_count": per_feature_changed,
                "anchor_identity_sha256": anchor_sha256,
            }
        )
    if feature_changed_values <= 0 or feature_max_diff <= 0.0:
        raise RuntimeError("v6 migration anchors did not observe a feature change")
    if migration_max_diff <= 0.0:
        raise RuntimeError("v6 migration anchors did not observe output drift")
    anchor_count = len(rows)
    anchor_set_sha256 = "sha256:" + hashlib.sha256(
        json.dumps(
            [row["anchor_identity_sha256"] for row in rows],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "adapter-v6-step0-anchor-evidence-v1",
        "device": "cpu",
        "source_adapter": source_adapter,
        "target_adapter": target_adapter,
        "public_observation": True,
        "separate_adapter_specific_entity_features": True,
        "separate_adapter_specific_action_contexts": True,
        "forward_identical": False,
        "promotion_eligible": False,
        "topology_construction_proof": "deterministic_parameter_replay_in_receipt",
        "migration_output_max_abs_diff": migration_max_diff,
        "legal_policy_forward_kl_mean": sum(forward_kls) / anchor_count,
        "legal_policy_forward_kl_max": max(forward_kls),
        "legal_policy_reverse_kl_mean": sum(reverse_kls) / anchor_count,
        "legal_policy_reverse_kl_max": max(reverse_kls),
        "legal_policy_top1_flip_count": top1_flips,
        "legal_policy_top1_flip_rate": top1_flips / anchor_count,
        "scalar_value_rmse": float(
            np.sqrt(np.mean(np.square(np.asarray(value_errors, dtype=np.float64))))
        ),
        "scalar_value_max_abs_error": max(value_errors),
        "feature_max_abs_diff": feature_max_diff,
        "feature_changed_value_count": feature_changed_values,
        "anchor_count": anchor_count,
        "anchor_set_sha256": anchor_set_sha256,
        "anchors": rows,
    }


def _preserve_source_top_level_keys(
    in_checkpoint: str,
    out_checkpoint: str,
    *,
    mutated_keys: tuple[str, ...] = ("model", "config"),
) -> list[str]:
    """CAT-80: ``EntityGraphPolicy.save()`` rebuilds the checkpoint from the
    freshly-constructed ``upgraded`` policy and does NOT carry over the source
    checkpoint's top-level provenance keys (``mask_hidden_info``,
    ``action_mask_version``, ``static_action_features*``, ``policy_type`` ...).
    In particular ``mask_hidden_info`` silently reset True->False, mislabeling a
    masked net as omniscient -- the exact #71 hidden-info-leak class the #76
    masked-regime guard exists to catch.

    Re-open both checkpoints and restore every top-level key from the SOURCE
    except the ones this upgrade intentionally mutates (``model`` weights and the
    ``config`` flags). Returns the sorted list of preserved source keys.
    """
    import torch

    in_raw = torch.load(in_checkpoint, map_location="cpu", weights_only=False)
    out_raw = torch.load(out_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(in_raw, dict) or not isinstance(out_raw, dict):
        return []
    merged = dict(in_raw)
    for key in mutated_keys:
        if key in out_raw:
            merged[key] = out_raw[key]
    torch.save(merged, out_checkpoint)
    return sorted(k for k in in_raw if k not in mutated_keys)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forward_tolerance(flags: dict[str, object]) -> float:
    """Return the reviewed numerical tolerance for an upgrade bundle.

    Zero-output adapters must remain bit-exact.  Cloning the late policy
    suffix into a separate value tower changes only the order of otherwise
    identical FP32 operations, so its forward check may differ by at most one
    float32 epsilon.  This exception is deliberately derived from the typed
    split flag rather than exposed as caller-controlled configuration.
    """

    if int(flags.get("value_tower_split_layers", 0) or 0) > 0:
        return float(np.finfo(np.float32).eps)
    return 0.0


def _record_upgrade_provenance(
    out_checkpoint: str,
    *,
    in_checkpoint: str,
    flags: dict[str, object],
    seed: int,
    forward_max_diff: float | None,
    forward_tolerance: float,
) -> None:
    """Atomically attest how freshly initialized upgrade modules were built."""

    import torch

    output = Path(out_checkpoint)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": _sha256_file(in_checkpoint),
        "flags": dict(flags),
        "initialization_seed": int(seed),
        "trained_value_readouts_added": [],
        "forward_max_diff": forward_max_diff,
        "forward_tolerance": float(forward_tolerance),
        "forward_identical_at_init": (
            forward_max_diff <= forward_tolerance
            if forward_max_diff is not None
            else False
        ),
        "value_tower_initialization": (
            {
                "method": "exact_policy_suffix_clone",
                "split_layers": int(flags["value_tower_split_layers"]),
            }
            if int(flags.get("value_tower_split_layers", 0) or 0) > 0
            else None
        ),
    }
    tmp = output.with_name(f".{output.name}.upgrade.tmp.{os.getpid()}")
    try:
        torch.save(raw, tmp)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()


def _record_information_migration_provenance(
    out_checkpoint: str,
    *,
    in_checkpoint: str,
    flags: dict[str, object],
    seed: int,
    anchor_evidence: dict[str, object],
) -> None:
    """Atomically record an honest non-function-preserving v6 transition."""

    import torch

    output = Path(out_checkpoint)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["information_contract_migration_provenance"] = {
        "schema_version": "entity-graph-information-contract-migration-v1",
        "migration": "current_v2_to_v6_topology_split1",
        "source_checkpoint_sha256": _sha256_file(in_checkpoint),
        "flags": dict(flags),
        "initialization_seed": int(seed),
        "source_adapter": anchor_evidence["source_adapter"],
        "target_adapter": anchor_evidence["target_adapter"],
        "forward_identical": False,
        "promotion_eligible": False,
        "commissioning_status": "non_promotable_architecture_treatment",
        "step0_anchor_evidence": anchor_evidence,
        "value_tower_initialization": (
            {
                "method": "exact_policy_suffix_clone",
                "split_layers": int(flags["value_tower_split_layers"]),
            }
            if int(flags.get("value_tower_split_layers", 0) or 0) > 0
            else None
        ),
    }
    # A migration must never inherit or manufacture the historical upgrade
    # key consumed by promotion-eligible function-preserving receipt replay.
    raw.pop("upgrade_provenance", None)
    tmp = output.with_name(f".{output.name}.migration.tmp.{os.getpid()}")
    try:
        torch.save(raw, tmp)
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-checkpoint", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    parser.add_argument("--flags", default="gather,cross:2,value")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deterministic initialization seed for every newly added module.",
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    overrides = _parse_flags(args.flags)
    requested_v6_migration = any(
        piece.strip()
        in {
            "current_v6_information_migration_topology_split1",
            "current-v6-information-migration-topology-split1",
        }
        for piece in args.flags.split(",")
    )
    if requested_v6_migration and str(args.device) != "cpu":
        raise SystemExit(
            "information-contract migration evidence is replayed on CPU; "
            "issue it with --device cpu"
        )
    import torch

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    base = EntityGraphPolicy.load(args.in_checkpoint, device=args.device)
    base.model.eval()

    upgraded_config = _build_upgraded_config(base.config, overrides)
    static = base.static_action_features.detach().cpu().numpy()
    # EntityGraphPolicy owns model initialization and resets Torch's RNG from
    # its ``seed`` argument.  Passing no seed here silently reset every upgrade
    # to seed 0 even though the CLI/provenance recorded ``--seed``.
    upgraded = EntityGraphPolicy(
        upgraded_config,
        static,
        seed=int(args.seed),
        device=args.device,
        entity_feature_adapter_version=(
            RUST_ENTITY_ADAPTER_V6
            if requested_v6_migration
            else RUST_ENTITY_ADAPTER_V5
            if overrides.get("meaningful_public_history_schema")
            == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
            else RUST_ENTITY_ADAPTER_V4
            if bool(overrides.get("public_rule_state_features", False))
            else base.entity_feature_adapter_version
        ),
    )
    missing, unexpected = upgraded.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    source_value_split_layers = int(
        getattr(base.config, "value_tower_split_layers", 0) or 0
    )
    upgraded_value_split_layers = int(
        getattr(upgraded_config, "value_tower_split_layers", 0) or 0
    )
    if upgraded_value_split_layers > source_value_split_layers:
        upgraded.model.initialize_value_tower_from_policy()
    disallowed = [
        k for k in missing if not k.startswith(NEW_PARAM_PREFIXES + ("q_head.",))
    ]
    if disallowed or unexpected:
        raise SystemExit(
            f"warm-start mismatch: missing={disallowed[:8]} unexpected={unexpected[:8]}"
        )
    upgraded.model.eval()

    max_diff = None
    migration_anchor_evidence = None
    forward_tolerance = _forward_tolerance(overrides)
    if not args.no_verify:
        if requested_v6_migration:
            migration_anchor_evidence = _migration_anchor_evidence(
                base, upgraded, args.device
            )
        else:
            max_diff = _verify_forward_identical(base, upgraded, args.device)
            if max_diff > forward_tolerance:
                raise SystemExit(
                    "forward changed beyond reviewed initialization tolerance: "
                    f"max_diff={max_diff} tolerance={forward_tolerance}"
                )
    elif requested_v6_migration:
        raise SystemExit("information-contract migration cannot skip anchor evidence")

    upgraded.save(args.out_checkpoint)
    # CAT-80: restore top-level provenance keys the fresh-policy save() drops
    # (mask_hidden_info et al.); only model weights + config flags are mutated.
    preserved_source_keys = _preserve_source_top_level_keys(
        args.in_checkpoint,
        args.out_checkpoint,
        mutated_keys=(
            ("model", "config", "entity_feature_adapter")
            if bool(overrides.get("public_rule_state_features", False))
            or overrides.get("meaningful_public_history_schema")
            == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
            else ("model", "config")
        ),
    )
    if requested_v6_migration:
        assert migration_anchor_evidence is not None
        _record_information_migration_provenance(
            args.out_checkpoint,
            in_checkpoint=args.in_checkpoint,
            flags=overrides,
            seed=int(args.seed),
            anchor_evidence=migration_anchor_evidence,
        )
    else:
        _record_upgrade_provenance(
            args.out_checkpoint,
            in_checkpoint=args.in_checkpoint,
            flags=overrides,
            seed=int(args.seed),
            forward_max_diff=max_diff,
            forward_tolerance=forward_tolerance,
        )
    print(
        json.dumps(
            {
                "in_checkpoint": args.in_checkpoint,
                "out_checkpoint": args.out_checkpoint,
                "flags": overrides,
                "new_params_added": sorted(
                    set(k for k in missing if k.startswith(NEW_PARAM_PREFIXES))
                ),
                "forward_max_diff": max_diff,
                "forward_tolerance": forward_tolerance,
                "forward_identical_at_init": (
                    False
                    if requested_v6_migration
                    else (max_diff <= forward_tolerance)
                    if max_diff is not None
                    else "skipped"
                ),
                "information_contract_migration": migration_anchor_evidence,
                "promotion_eligible": False if requested_v6_migration else None,
                "preserved_source_keys": preserved_source_keys,
                "initialization_seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
