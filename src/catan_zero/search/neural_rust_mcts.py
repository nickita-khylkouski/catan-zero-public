from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE, _context_vector
from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    policy_entity_feature_adapter_version,
    require_known_entity_feature_adapter,
)
from catan_zero.rl.entity_token_features import build_entity_token_features
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    public_events_from_native_action_records,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv


# Backward-compatible public alias used by teacher-data writers.  The canonical
# definition lives in the dependency-free contract module so policy checkpoint
# code and search cannot drift or form a circular import.
RUST_ENTITY_ADAPTER_VERSION = CURRENT_RUST_ENTITY_ADAPTER_VERSION


def _policy_history_options(policy: EntityGraphPolicy) -> tuple[bool, int]:
    config = getattr(policy, "config", None)
    enabled = bool(getattr(config, "meaningful_public_history", False))
    limit = int(getattr(config, "event_history_limit", 64) or 0)
    if enabled:
        limit = min(limit, MEANINGFUL_PUBLIC_HISTORY_LIMIT)
    return enabled, limit


@dataclass(frozen=True, slots=True)
class EntityGraphRustEvaluatorConfig:
    """Adapter options for using an EntityGraphPolicy inside RustMCTS.

    The first supported target is the current 2p/no-trade track.  For two-player
    zero-sum search we evaluate priors/value from the side-to-act perspective and
    flip the scalar value on opponent turns so RustMCTS still backs up root value.
    """

    value_scale: float = 1.0
    prior_temperature: float = 1.0
    context_fill: float = 0.0
    cache_size: int = 100_000
    entity_feature_adapter_version: str = RUST_ENTITY_ADAPTER_VERSION
    # #60 A/B knob. The entity_graph value head is a raw Linear trained with
    # MSE on z in {-1,+1} (no tanh in the model or the loss), so applying
    # tanh at inference double-squashes: monotonic (root argmax and rescaled
    # sigma nearly unaffected) but interior chance-node expectation backups
    # take E[tanh(v)] != tanh(E[v]) (Jensen bias) and v_mix magnitudes
    # compress. "tanh" = historical behavior, stays the DEFAULT;
    # "clip" = identity here, leaving the existing np.clip(-1, 1) at the
    # call sites as the only squash. Gate: post-Gate-A A/B (strength-based),
    # never blind-ship.
    value_squash: str = "tanh"
    # Select which trained value head search backs up. ``scalar`` preserves the
    # historical MSE-head path exactly. ``categorical`` opts into the calibrated
    # support expectation emitted as ``outputs["value_categorical"]`` by an
    # HL-Gauss checkpoint. That expectation is already bounded/calibrated, so it
    # keeps value_scale but bypasses the scalar head's inference-time tanh; both
    # modes retain the same perspective flip and final clipping.
    # Categorical is deliberately fail-closed: evaluator construction rejects a
    # checkpoint without a real categorical head instead of silently falling
    # back to the scalar head (or using freshly initialised upgrade weights).
    value_readout: str = "scalar"
    # Hidden-information leak fix (f72). When True, the model input is masked to
    # PUBLIC information from the acting player's perspective: every OPPONENT's
    # resource-hand composition, unplayed dev-card identities, and actual VP
    # (which reveals hidden VICTORY_POINT cards) are dropped before featurization
    # -- only their public counts (resource_card_count, development_card_count),
    # public VP, and played dev cards remain. The actor's own hand is untouched.
    # Default OFF: the shipped 35M checkpoint was trained on UNMASKED (omniscient)
    # inputs, so enabling this at inference against that checkpoint is off-
    # distribution; it becomes correct only paired with a checkpoint retrained on
    # masked shards (train_bc.py --mask-hidden-info). The env/world transitions
    # are unaffected -- this is a model-INPUT boundary only. See
    # `catan_zero.rl.entity_token_features.mask_player_tokens_public`.
    public_observation: bool = False
    # Task #81 phase 2: build the ENTITY-TOKEN arrays via the Rust featurizer
    # (`catanatron_rs.build_entity_features_flat`, bit-exact parity-gated --
    # see entity_token_features_rust.py + tests/test_rust_featurize_parity.py)
    # instead of `build_entity_token_features`'s Python per-token loops. Board
    # topology is hoisted to ONE construction per evaluator lifetime (lazy, on
    # the first leaf, via the existing Python `_topology` path -- sound because
    # topology is a pure function of the BASE-map tile structure, identical for
    # every game this track plays; the same BASE-layout-only constraint
    # entity_token_features.py already documents). The per-leaf CONTEXT
    # features (`rust_action_context_batch`) still use the JSON-derived
    # payload -- that port is a separate phase. Default OFF = exact current
    # behavior; ON fails loudly (no silent fallback) if the installed wheel
    # lacks `build_entity_features_flat`.
    rust_featurize: bool = False
    # CAT-61: surface the policy's value-uncertainty head to the searcher. When
    # True, evaluate()/evaluate_many() return a 3-tuple (priors, value,
    # uncertainty) instead of the default 2-tuple, where `uncertainty` is the
    # (non-negative, perspective-invariant) value-error prediction -- 0.0 when
    # the loaded checkpoint has no value_uncertainty_head. Default False keeps
    # the return shape, the cache contents, and every existing caller
    # bit-identical; the scalar is only consumed by GumbelChanceMCTS when its
    # own `uncertainty_backup_weighting` flag is also on. Wired on all evaluate
    # paths: the sync `evaluate`/`evaluate_many` and the async
    # `BatchedEntityGraphRustEvaluator` (queue + cache). The symmetry-averaged
    # path emits 0.0 (no uncertainty is defined over board orientations).
    emit_uncertainty: bool = False


def _assert_value_readout_available(
    policy: "EntityGraphPolicy", config: "EntityGraphRustEvaluatorConfig"
) -> None:
    """Validate the science-critical value-head selection at load time."""
    readout = str(config.value_readout)
    if readout not in {"scalar", "categorical"}:
        raise ValueError(
            f"unknown value_readout mode: {readout!r} "
            "(expected 'scalar' or 'categorical')"
        )
    if readout == "scalar":
        return

    model = getattr(policy, "model", None)
    bins = int(getattr(model, "value_categorical_bins", 0) or 0)
    head = getattr(model, "value_categorical_head", None)
    missing_state_keys = tuple(getattr(policy, "_checkpoint_missing_state_keys", ()))
    missing_head_weights = any(
        str(key).startswith("value_categorical_head.") for key in missing_state_keys
    )
    trained_readouts = tuple(
        str(readout)
        for readout in getattr(policy, "trained_value_readouts", ("scalar",))
    )
    categorical_provenance = "categorical" in trained_readouts
    if bins < 2 or head is None or missing_head_weights or not categorical_provenance:
        detail = ""
        if missing_head_weights:
            detail = " (the checkpoint config declares the head but its trained weights are absent)"
        elif not categorical_provenance:
            provenance_errors = tuple(
                str(error)
                for error in getattr(policy, "_value_training_provenance_errors", ())
            )
            error_detail = (
                f"; validation errors: {', '.join(provenance_errors)}"
                if provenance_errors
                else ""
            )
            detail = (
                " (the checkpoint has no positive value-training-v1 provenance "
                f"that the categorical readout was optimized{error_detail})"
            )
        raise ValueError(
            "value_readout='categorical' requires a checkpoint with a trained "
            f"HL-Gauss categorical value head (value_categorical_bins >= 2){detail}; "
            f"loaded model reports value_categorical_bins={bins}. Use "
            "value_readout='scalar' or load a categorical-value checkpoint."
        )


def _assert_uncertainty_readout_available(
    policy: "EntityGraphPolicy", config: "EntityGraphRustEvaluatorConfig"
) -> None:
    """Fail closed before search can consume an untrained uncertainty head.

    ``EntityGraphPolicy.load`` deliberately permits optional-head tensors to be
    absent so an older checkpoint can be warm-started into a newer architecture.
    In that case the module exists in memory but its parameters are freshly
    initialized.  That is safe while uncertainty emission is disabled, but it
    is not a valid search signal: backup weighting would otherwise consume
    random predictions merely because the checkpoint config names the head.

    This is intentionally narrower than a training-provenance requirement.  It
    verifies only that an opted-in consumer has a real module whose complete
    tensor set came from the checkpoint.
    """

    if not bool(config.emit_uncertainty):
        return

    model = getattr(policy, "model", None)
    head = getattr(model, "value_uncertainty_head", None)
    missing_state_keys = tuple(
        str(key) for key in getattr(policy, "_checkpoint_missing_state_keys", ())
    )
    missing_head_weights = any(
        key.startswith("value_uncertainty_head.") for key in missing_state_keys
    )
    if head is None or missing_head_weights:
        detail = (
            " (the checkpoint config declares the head but its trained weights "
            "are absent)"
            if missing_head_weights
            else ""
        )
        raise ValueError(
            "emit_uncertainty=True requires a checkpoint with a complete "
            f"value_uncertainty_head{detail}; disable uncertainty emission or "
            "load a checkpoint containing the head tensors."
        )


def _uncertainty_from_outputs(outputs: dict[str, Any], row: int) -> float:
    """Extract the value-uncertainty head's scalar for batch `row` (CAT-61).

    Returns 0.0 when the loaded policy has no value_uncertainty_head (the key is
    absent from `outputs`), so a non-uncertainty checkpoint flows through with a
    zero signal. The value is a magnitude (predicted value error), so it takes
    NO opponent sign flip and NO value squash -- it is emitted as the head
    produces it (softplus, already non-negative)."""
    tensor = outputs.get("value_uncertainty")
    if tensor is None:
        return 0.0
    return float(tensor.detach().float().cpu().numpy()[row])


def _forward_search_policy(
    policy: Any,
    entity: dict[str, np.ndarray],
    legal_ids: np.ndarray,
    context: np.ndarray,
    *,
    return_q: bool,
) -> dict[str, Any]:
    """Request only tensors consumed by search when the policy supports it."""
    kwargs: dict[str, bool] = {"return_q": bool(return_q)}
    if bool(getattr(policy, "supports_final_vp_selection", False)):
        kwargs["return_final_vp"] = False
    return policy.forward_legal_np(entity, legal_ids, context, **kwargs)


def _fetch_leaf_decision_inputs(
    game: Any,
    colors: tuple[str, ...],
    *,
    include_snapshot: bool = True,
) -> tuple[str | None, dict[int, Any]]:
    """ONE Rust round-trip for everything the leaf-eval preamble needs.

    Returns (snapshot_text, action_by_id). ``snapshot_text`` is ``None`` when
    ``include_snapshot`` is false; the action map is always fetched. Previously
    each leaf evaluation re-fetched `playable_action_indices`/
    `playable_actions_json` up to 3x
    (rust_policy_action_ids + both featurizers via `_resolve_entity_adapter`)
    and `json_snapshot` up to 3x (_state_key + both featurizers) on the same,
    unchanged game state -- the measured top cost at wide placement roots
    (playable_action_indices scales with branching: ~0.2ms at 1 legal action,
    ~12ms at 54). The featurizers already accepted `snapshot=`/`action_by_id=`
    for exactly this reuse; this helper is the caller-side plumbing that was
    never wired up.
    """
    snapshot_text = game.json_snapshot() if include_snapshot else None
    action_ids = [
        int(action) for action in game.playable_action_indices(list(colors), None)
    ]
    raw_actions = json.loads(game.playable_actions_json())
    return snapshot_text, dict(zip(action_ids, raw_actions))


def _assert_public_observation_matches_checkpoint_training(
    policy: "EntityGraphPolicy", config: "EntityGraphRustEvaluatorConfig"
) -> None:
    """Task #76 safety net (f72 hidden-info leak, CLI-default-override trap class):
    a checkpoint's own recorded `trained_with_masked_hidden_info` metadata (see
    EntityGraphPolicy.save/load) must agree with this evaluator's
    `public_observation` request, or fail closed rather than silently running
    mismatched -- either direction is a real misconfiguration: requesting
    public_observation=True against an omniscient-trained (or legacy, pre-#76)
    checkpoint would silently regenerate the exact leaked-hidden-info corpus #71
    fixed; requesting False against a masked-trained checkpoint would silently
    feed it omniscient inputs it never learned to use.
    """
    requested = bool(config.public_observation)
    trained_masked = bool(getattr(policy, "trained_with_masked_hidden_info", False))
    if requested:
        print(
            json.dumps(
                {"progress": "public_observation_enabled", "public_observation": True}
            ),
            flush=True,
        )
    if requested != trained_masked:
        raise ValueError(
            "public_observation/checkpoint-training mismatch (task #76 safety net): "
            f"requested public_observation={requested} but checkpoint's recorded "
            f"trained_with_masked_hidden_info={trained_masked}. Generating or "
            "evaluating with this combination would silently run on hidden-info "
            "input the checkpoint was never trained on (or regenerate the f72 leak "
            "this check exists to prevent) -- pass a checkpoint trained with "
            "train_bc.py --mask-hidden-info to use --public-observation, or drop "
            "--public-observation to match an omniscient-trained checkpoint."
        )


def _assert_feature_adapter_matches_checkpoint(
    policy: "EntityGraphPolicy", config: "EntityGraphRustEvaluatorConfig"
) -> None:
    requested = require_known_entity_feature_adapter(
        config.entity_feature_adapter_version
    )
    if requested != RUST_ENTITY_ADAPTER_VERSION:
        raise ValueError(
            "requested entity feature adapter is known but not implemented by "
            f"this runtime: requested={requested!r} "
            f"implemented={RUST_ENTITY_ADAPTER_VERSION!r}"
        )
    checkpoint_version = policy_entity_feature_adapter_version(policy)
    if requested != checkpoint_version:
        source = str(
            getattr(policy, "entity_feature_adapter_binding_source", "legacy_policy")
        )
        raise ValueError(
            "entity feature adapter/checkpoint mismatch: "
            f"runtime={requested!r} checkpoint={checkpoint_version!r} "
            f"checkpoint_binding_source={source!r}. Input tensor shapes can match "
            "while slot meanings differ; use a checkpoint trained with this exact "
            "adapter version or explicitly run its versioned legacy adapter."
        )


class EntityGraphRustEvaluator:
    def __init__(
        self,
        policy: EntityGraphPolicy,
        *,
        config: EntityGraphRustEvaluatorConfig | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or EntityGraphRustEvaluatorConfig()
        _assert_feature_adapter_matches_checkpoint(policy, self.config)
        _assert_public_observation_matches_checkpoint_training(policy, self.config)
        _assert_value_readout_available(policy, self.config)
        _assert_uncertainty_readout_available(policy, self.config)
        # CAT-126 #15: OrderedDict gives LRU eviction (move_to_end on hit,
        # popitem(last=False) on evict) instead of FIFO. Bit-identical outputs
        # (a hit returns the same deterministic value); only WHICH entry is
        # evicted changes, improving hit rate for revisited MCTS positions.
        self._cache: OrderedDict[
            tuple[str, str, tuple[str, ...], tuple[int, ...]],
            tuple[dict[int, float], float] | tuple[dict[int, float], float, float],
        ] = OrderedDict()
        # ``BatchedEntityGraphRustEvaluator`` shares this cache between caller
        # threads running inherited ``evaluate_many`` and its background batch
        # worker.  One lock in the base class lets every path make LRU
        # get+touch and evict+store atomic.  The lock is deliberately never held
        # across featurization or a model forward.
        self._cache_lock = threading.RLock()
        # Task #81 phase 2 (config.rust_featurize): board topology, computed
        # lazily ONCE per evaluator lifetime on the first Rust-featurized leaf
        # and reused for every subsequent one (see the config field's comment
        # for why once-per-lifetime is sound on the BASE map).
        self._rust_topology: Any = None
        # BatchedEntityGraphRustEvaluator prepares features in producer
        # threads.  Two cold producers can therefore observe ``None`` at the
        # same time.  Keep initialization once-per-evaluator as promised while
        # leaving the hot path as one unlocked attribute read.
        self._rust_topology_lock = threading.Lock()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        device: str = "cpu",
        config: EntityGraphRustEvaluatorConfig | None = None,
    ) -> "EntityGraphRustEvaluator":
        return cls(EntityGraphPolicy.load(checkpoint, device=device), config=config)

    def _entity_batch_via_rust(
        self,
        game: Any,
        *,
        colors: tuple[str, ...],
        policy_action_ids: tuple[int, ...],
        acting_color: str,
        adapter: Any,
    ) -> dict[str, np.ndarray]:
        """Rust-featurized replacement for the `rust_game_to_entity_batch`
        call (task #81 phase 2, gated by `config.rust_featurize`). Same output
        contract: `build_entity_token_features`'s array dict (sans "schema"),
        each array wrapped with a leading batch dim. `policy_action_ids` must
        be 1:1 aligned with `game.playable_actions`' native order (every
        evaluator call site satisfies this -- verified across all of them for
        the batch-API sign-off). `adapter` is only consumed on the FIRST call
        ever, to bootstrap the topology via the existing Python `_topology`
        path -- CAT-72: callers skip building it (pass `adapter=None`) once
        `self._rust_topology` is already warm, since this branch never runs
        again after that."""
        from catan_zero.rl.entity_token_features_rust import (
            build_entity_features_rust,
        )

        topology = self._get_or_init_rust_topology(adapter, acting_color=acting_color)
        entity = build_entity_features_rust(
            game,
            colors=tuple(str(color) for color in colors),
            policy_action_ids=tuple(int(a) for a in policy_action_ids),
            action_size=int(self.policy.action_size),
            topology=topology,
            public_observation=bool(self.config.public_observation),
            public_card_count_features=bool(
                getattr(
                    getattr(self.policy, "config", None),
                    "public_card_count_features",
                    getattr(self.policy, "public_card_count_features", False),
                )
            ),
            meaningful_public_history=_policy_history_options(self.policy)[0],
            history_limit=_policy_history_options(self.policy)[1],
        )
        return {key: np.asarray(value)[None, ...] for key, value in entity.items()}

    def _get_or_init_rust_topology(self, adapter: Any, *, acting_color: str) -> Any:
        """Return the immutable native topology, initializing it exactly once.

        The async evaluator's callers featurize before enqueueing, so topology
        bootstrap can be reached concurrently.  A double-checked lock keeps
        the common warm path lock-free and prevents duplicate cold builds.
        """
        topology = self._rust_topology
        if topology is not None:
            return topology

        from catan_zero.rl.entity_token_features_rust import compute_rust_topology

        with self._rust_topology_lock:
            topology = self._rust_topology
            if topology is None:
                if adapter is None:
                    raise RuntimeError(
                        "Rust topology is cold but no entity adapter was provided"
                    )
                topology = compute_rust_topology(adapter, str(acting_color))
                self._rust_topology = topology
        return topology

    def _context_batch_via_rust(
        self,
        game: Any,
        *,
        acting_color: str,
        adapter: Any,
    ) -> np.ndarray:
        """Rust-featurized replacement for the `rust_action_context_batch`
        call (task #81 context wiring, gated by the SAME `config.rust_featurize`
        flag as `_entity_batch_via_rust`). Reuses the identical lazily-
        bootstrapped `self._rust_topology` the entity path builds/reuses --
        context features need the same fixed hex/edge topology (node
        adjacency, port lookups), so whichever of entity/context runs FIRST
        in a given call bootstraps it for both. Same output contract as
        `rust_action_context_batch`: `(1, n_legal, CONTEXT_ACTION_FEATURE_SIZE)`
        float32. CAT-72: `adapter` may be `None` once `self._rust_topology`
        is already warm (see `_entity_batch_via_rust`'s docstring)."""
        from catan_zero.rl.action_context_features_rust import build_action_context_rust

        topology = self._get_or_init_rust_topology(adapter, acting_color=acting_color)
        context = build_action_context_rust(game, topology=topology)
        return context[None, ...]

    def _apply_value_squash(self, raw_value: float) -> float:
        """Scale the raw value-head output and apply `config.value_squash`.

        "tanh" reproduces the historical `tanh(raw * value_scale)`
        bit-for-bit; "clip" returns the scaled value unchanged so the
        existing `np.clip(-1, 1)` at the call sites (applied AFTER the
        opponent sign flip, order unchanged in both modes) is the only
        squash. A categorical readout is already a calibrated expectation on
        [-1, 1], so it always follows the latter path (scale + final clip) and
        is never double-squashed. See the config field comments for rationale.
        """
        scaled = float(raw_value) * float(self.config.value_scale)
        if str(self.config.value_readout) == "categorical":
            return scaled
        squash = str(self.config.value_squash)
        if squash == "tanh":
            return float(np.tanh(scaled))
        if squash == "clip":
            return scaled
        raise ValueError(
            f"unknown value_squash mode: {squash!r} (expected 'tanh' or 'clip')"
        )

    def _value_output(self, outputs: dict[str, Any]) -> Any:
        """Return the configured value tensor, never silently falling back."""
        readout = str(self.config.value_readout)
        key = "value" if readout == "scalar" else "value_categorical"
        if key not in outputs:
            raise RuntimeError(
                f"value_readout={readout!r} requested model output {key!r}, but the "
                f"forward pass emitted keys={sorted(outputs)}"
            )
        return outputs[key]

    def _eval_result(
        self, priors: dict[int, float], value: float, uncertainty: float
    ) -> tuple[dict[int, float], float] | tuple[dict[int, float], float, float]:
        """Pack an evaluation into the configured return shape (CAT-61). When
        `config.emit_uncertainty` is False (default) returns the historical
        2-tuple `(priors, value)` -- byte-for-byte the previous behavior, so
        every existing caller and the cache contents are unchanged. When True,
        appends the non-negative `uncertainty`."""
        if bool(self.config.emit_uncertainty):
            return priors, value, max(0.0, float(uncertainty))
        return priors, value

    def _cache_entry(
        self, priors: dict[int, float], value: float, uncertainty: float
    ) -> tuple[dict[int, float], float] | tuple[dict[int, float], float, float]:
        """Cache entry matching the configured return shape (CAT-61). Stores a
        COPY of `priors` (as the pre-CAT-61 code did) and appends `uncertainty`
        only when emitting, so the default-path cache stays `(dict, value)` --
        byte-identical to before."""
        if bool(self.config.emit_uncertainty):
            return dict(priors), value, max(0.0, float(uncertainty))
        return dict(priors), value

    def _cache_get(
        self,
        cache_key: tuple[str, str, tuple[str, ...], tuple[int, ...]] | None,
    ) -> tuple[dict[int, float], float] | tuple[dict[int, float], float, float] | None:
        """Return and LRU-touch one entry atomically across evaluator paths."""
        if cache_key is None or int(self.config.cache_size) <= 0:
            return None
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
            return cached

    def _cache_store(
        self,
        cache_key: tuple[str, str, tuple[str, ...], tuple[int, ...]] | None,
        priors: dict[int, float],
        value: float,
        uncertainty: float,
    ) -> None:
        """Atomically replace/touch or evict+insert one completed evaluation."""
        capacity = int(self.config.cache_size)
        if cache_key is None or capacity <= 0:
            return
        entry = self._cache_entry(priors, value, uncertainty)
        with self._cache_lock:
            if cache_key in self._cache:
                self._cache[cache_key] = entry
                self._cache.move_to_end(cache_key)
                return
            if len(self._cache) >= capacity:
                self._cache.popitem(last=False)
            self._cache[cache_key] = entry

    def evaluate(
        self,
        game: Any,
        legal_actions: tuple[int, ...],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> tuple[dict[int, float], float] | tuple[dict[int, float], float, float]:
        if not legal_actions:
            return self._eval_result({}, _terminal_or_zero(game, root_color), 0.0)

        acting_color = str(game.current_color())
        cache_enabled = int(self.config.cache_size) > 0
        need_adapter_resolve = (
            not bool(self.config.rust_featurize)
        ) or self._rust_topology is None
        # B1 dedup: fetch once, share across the policy-id translation, the
        # cache key, and (on a miss) both featurizer calls below. A warm native
        # evaluator with caching disabled needs only the action map: avoid
        # serializing a full JSON snapshot that no downstream consumer reads.
        snapshot_text, action_by_id = _fetch_leaf_decision_inputs(
            game,
            colors,
            include_snapshot=cache_enabled or need_adapter_resolve,
        )
        policy_action_ids = rust_policy_action_ids(
            game,
            legal_actions,
            colors=colors,
            action_size=int(self.policy.action_size),
            action_by_id=action_by_id,
        )
        # cache_size <= 0 disables the eval cache ENTIRELY, including the
        # per-leaf blake2b(_state_key) over the full snapshot text and the
        # key-tuple build -- previously only the STORE was gated, so a
        # cache_size=0 evaluator still hashed every leaf for a cache it
        # never wrote (speed-czar cache audit, 2026-07-06).
        cache_key = None
        if cache_enabled:
            assert snapshot_text is not None
            cache_key = (
                _state_key(game, snapshot_text=snapshot_text),
                str(root_color),
                tuple(str(color) for color in colors),
                tuple(int(action) for action in policy_action_ids),
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                # CAT-61: cache entries are (priors, value) or (priors, value,
                # uncertainty); tolerate both so a mixed-format cache is safe.
                uncertainty = cached[2] if len(cached) > 2 else 0.0
                return self._eval_result(dict(cached[0]), float(cached[1]), uncertainty)

        # CAT-72: `resolved` (the (payload, adapter, structured) tuple from
        # `_resolve_entity_adapter`, plus the `json.loads(snapshot_text)` that
        # feeds it) is ONLY consumed by the rust_featurize path to bootstrap
        # `self._rust_topology` on the FIRST leaf of this evaluator's
        # lifetime -- every leaf after that, `_entity_batch_via_rust`/
        # `_context_batch_via_rust` never touch `adapter` at all (see their
        # `if self._rust_topology is None:` guards). Building it unconditionally
        # was measured (tools/perf_snapshot.py leaf --rust-featurize, CAT-72
        # re-profile) to cost ~0.1ms/leaf regardless of `rust_featurize` --
        # dead weight on every non-bootstrap Rust-path leaf. Skip it once
        # topology is warm; the legacy (non-rust_featurize) path still needs
        # `resolved` every leaf, unchanged.
        resolved: (
            tuple[dict[str, Any], "_RustEntityFeatureEnv", list[dict[str, Any]]] | None
        ) = None
        if need_adapter_resolve:
            assert snapshot_text is not None
            snapshot = json.loads(snapshot_text)
            # B2 dedup: resolve the (payload, adapter, structured) tuple ONCE and
            # share it with both featurizers below -- see `_resolve_entity_adapter`
            # (previously each featurizer resolved it independently, doubling the
            # snapshot-parse/players-payload/masking-gate cost per leaf).
            resolved = _resolve_entity_adapter(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                snapshot=snapshot,
                action_by_id=action_by_id,
                public_observation=bool(self.config.public_observation),
                perspective=acting_color,
                meaningful_public_history=_policy_history_options(self.policy)[0],
            )
        if bool(self.config.rust_featurize):
            entity = self._entity_batch_via_rust(
                game,
                colors=colors,
                policy_action_ids=policy_action_ids,
                acting_color=acting_color,
                adapter=resolved[1] if resolved is not None else None,
            )
        else:
            entity = rust_game_to_entity_batch(
                game,
                legal_actions,
                actor=acting_color,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                public_observation=bool(self.config.public_observation),
                meaningful_public_history=_policy_history_options(self.policy)[0],
                history_limit=_policy_history_options(self.policy)[1],
                resolved=resolved,
            )
        legal_ids = np.asarray(policy_action_ids, dtype=np.int64)[None, :]
        if bool(self.config.rust_featurize):
            context = self._context_batch_via_rust(
                game,
                acting_color=acting_color,
                adapter=resolved[1] if resolved is not None else None,
            )
        else:
            context = rust_action_context_batch(
                game,
                legal_actions,
                actor=acting_color,
                colors=colors,
                action_size=int(self.policy.action_size),
                fill=float(self.config.context_fill),
                policy_action_ids=policy_action_ids,
                public_observation=bool(self.config.public_observation),
                resolved=resolved,
            )

        outputs = _forward_search_policy(
            self.policy,
            entity,
            legal_ids,
            context,
            return_q=False,
        )
        logits = outputs["logits"].detach().float().cpu().numpy()[0]
        temperature = max(float(self.config.prior_temperature), 1.0e-6)
        priors_arr = _softmax(logits / temperature)
        priors = {
            int(action): float(probability)
            for action, probability in zip(legal_actions, priors_arr)
        }

        raw_value = float(self._value_output(outputs).detach().float().cpu().numpy()[0])
        value = self._apply_value_squash(raw_value)
        if acting_color != str(root_color) and len(tuple(colors)) == 2:
            value = -value
        value = float(np.clip(value, -1.0, 1.0))
        uncertainty = _uncertainty_from_outputs(outputs, 0)
        if cache_enabled:
            self._cache_store(cache_key, priors, value, uncertainty)
        return self._eval_result(priors, value, uncertainty)

    def evaluate_symmetry_averaged(
        self,
        game: Any,
        legal_actions: tuple[int, ...],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> tuple[dict[int, float], float] | tuple[dict[int, float], float, float]:
        """f74b wide-root denoiser: average the net over all 12 D6 board
        orientations, then apply the SAME prior-softmax and value squash/sign
        as `evaluate()`. Legal-action rows are order-preserved by every
        symmetry, so per-candidate averaging is a direct column mean (no inverse
        action permutation). Intentionally uncached -- it is called only at the
        few wide placement roots per game and must not pollute the plain-eval
        cache keyed by the same state."""
        if not legal_actions:
            return self._eval_result({}, _terminal_or_zero(game, root_color), 0.0)

        import torch

        from catan_zero.rl.hex_symmetry import build_hex_symmetry

        acting_color = str(game.current_color())
        # The native feature builders consume the Python adapter only to
        # bootstrap fixed BASE-map topology.  Once warm, symmetry averaging is
        # just another native entity/context call and must not rebuild or parse
        # that adapter.  On the cold/native and Python paths, fetch the snapshot
        # and action mapping once and share them with both translation and
        # resolution, matching evaluate().
        need_adapter_resolve = (
            not bool(self.config.rust_featurize)
        ) or self._rust_topology is None
        resolved: (
            tuple[dict[str, Any], "_RustEntityFeatureEnv", list[dict[str, Any]]] | None
        ) = None
        if need_adapter_resolve:
            snapshot_text, action_by_id = _fetch_leaf_decision_inputs(
                game,
                colors,
                include_snapshot=True,
            )
            policy_action_ids = rust_policy_action_ids(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
                action_by_id=action_by_id,
            )
            resolved = _resolve_entity_adapter(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                snapshot=json.loads(snapshot_text),
                action_by_id=action_by_id,
                public_observation=bool(self.config.public_observation),
                perspective=acting_color,
                meaningful_public_history=_policy_history_options(self.policy)[0],
            )
        else:
            policy_action_ids = rust_policy_action_ids(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
            )
        # Task #81 gap #1 fix: this method previously never checked
        # `rust_featurize` at all, so a wide root with BOTH
        # `symmetry_averaged_eval=True` and `rust_featurize=True` set would
        # silently fall back to the slow Python path exactly where the
        # native featurizer wins most. Same gating as `evaluate()`/
        # `evaluate_many()` -- `sym.average_forward` below consumes whatever
        # entity dict/context array it's given by shape/value alone, so this
        # is a drop-in swap (bit-exact parity already proven per-array).
        if bool(self.config.rust_featurize):
            entity = self._entity_batch_via_rust(
                game,
                colors=colors,
                policy_action_ids=policy_action_ids,
                acting_color=acting_color,
                adapter=resolved[1] if resolved is not None else None,
            )
            context = self._context_batch_via_rust(
                game,
                acting_color=acting_color,
                adapter=resolved[1] if resolved is not None else None,
            )
        else:
            entity = rust_game_to_entity_batch(
                game,
                legal_actions,
                actor=acting_color,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                public_observation=bool(self.config.public_observation),
                meaningful_public_history=_policy_history_options(self.policy)[0],
                history_limit=_policy_history_options(self.policy)[1],
                resolved=resolved,
            )
            context = rust_action_context_batch(
                game,
                legal_actions,
                actor=acting_color,
                colors=colors,
                action_size=int(self.policy.action_size),
                fill=float(self.config.context_fill),
                policy_action_ids=policy_action_ids,
                public_observation=bool(self.config.public_observation),
                resolved=resolved,
            )
        legal_ids = np.asarray(policy_action_ids, dtype=np.int64)[None, :]

        def forward_fn(entity_n, legal_n, ctx_n, return_q):
            with torch.no_grad():
                out = _forward_search_policy(
                    self.policy,
                    entity_n,
                    legal_n,
                    ctx_n,
                    return_q=return_q,
                )
            return {
                "logits": out["logits"].detach().float().cpu().numpy(),
                "value": self._value_output(out)
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1),
            }

        sym = build_hex_symmetry()
        avg = sym.average_forward(
            entity,
            legal_ids,
            context,
            forward_fn,
            return_q=False,
            action_size=int(self.policy.action_size),
        )

        logits = np.asarray(avg["logits"], dtype=np.float64)
        temperature = max(float(self.config.prior_temperature), 1.0e-6)
        priors_arr = _softmax(logits / temperature)
        priors = {
            int(action): float(probability)
            for action, probability in zip(legal_actions, priors_arr)
        }

        raw_value = float(avg["value"])
        value = self._apply_value_squash(raw_value)
        if acting_color != str(root_color) and len(tuple(colors)) == 2:
            value = -value
        value = float(np.clip(value, -1.0, 1.0))
        # The symmetry-averaged forward extracts only logits+value (no
        # uncertainty averaging is defined over orientations), so this path
        # emits a 0.0 uncertainty when emit_uncertainty is on (CAT-61).
        return self._eval_result(priors, value, 0.0)

    def evaluate_many(
        self,
        requests: list[tuple[Any, tuple[int, ...]]],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> list[tuple[dict[int, float], float] | tuple[dict[int, float], float, float]]:
        """Evaluate several (game, legal_actions) pairs in ONE batched forward pass.

        Used to batch the up-to-11 ROLL-child leaf evaluations in
        `gumbel_chance_mcts.py`'s true chance-node expansion, which previously
        trickled through `evaluate()` one request at a time. Terminal
        (no-legal-actions) requests and cache hits are resolved without
        touching the model; only the remaining uncached, non-terminal
        requests are padded and stacked into a single forward call, reusing
        the same `_merge_batched_eval_requests` padding logic
        `BatchedEntityGraphRustEvaluator`'s async queue already uses.
        """
        if not requests:
            return []

        results: list[tuple[dict[int, float], float] | None] = [None] * len(requests)
        pending_indices: list[int] = []
        pending_batch_requests: list[_BatchedEvalRequest] = []

        for request_index, (game, legal_actions) in enumerate(requests):
            if not legal_actions:
                results[request_index] = self._eval_result(
                    {}, _terminal_or_zero(game, root_color), 0.0
                )
                continue

            acting_color = str(game.current_color())
            cache_enabled = int(self.config.cache_size) > 0
            need_adapter_resolve = (
                not bool(self.config.rust_featurize)
            ) or self._rust_topology is None
            # B1 dedup: see evaluate() -- one fetch shared by everything below.
            snapshot_text, action_by_id = _fetch_leaf_decision_inputs(
                game,
                colors,
                include_snapshot=cache_enabled or need_adapter_resolve,
            )
            policy_action_ids = rust_policy_action_ids(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
                action_by_id=action_by_id,
            )
            # See evaluate(): cache_size <= 0 skips key/hash work entirely.
            cache_key = None
            if cache_enabled:
                assert snapshot_text is not None
                cache_key = (
                    _state_key(game, snapshot_text=snapshot_text),
                    str(root_color),
                    tuple(str(color) for color in colors),
                    tuple(int(action) for action in policy_action_ids),
                )
                cached = self._cache_get(cache_key)
                if cached is not None:
                    # CAT-61: tolerate (priors, value) and (priors, value, unc).
                    uncertainty = cached[2] if len(cached) > 2 else 0.0
                    results[request_index] = self._eval_result(
                        dict(cached[0]), float(cached[1]), uncertainty
                    )
                    continue

            # Mirror evaluate()'s warm-topology fast path. Native entity/context
            # featurization consumes the Python adapter only on the first leaf,
            # where it bootstraps the immutable BASE-map topology. Rebuilding the
            # adapter after that point performs json.loads + per-player state JSON
            # + payload construction for data neither native call reads.
            resolved: (
                tuple[dict[str, Any], "_RustEntityFeatureEnv", list[dict[str, Any]]]
                | None
            ) = None
            if need_adapter_resolve:
                assert snapshot_text is not None
                snapshot = json.loads(snapshot_text)
                # B2 dedup: see evaluate() -- one shared resolve for both featurizers.
                resolved = _resolve_entity_adapter(
                    game,
                    legal_actions,
                    colors=colors,
                    action_size=int(self.policy.action_size),
                    policy_action_ids=policy_action_ids,
                    snapshot=snapshot,
                    action_by_id=action_by_id,
                    public_observation=bool(self.config.public_observation),
                    perspective=acting_color,
                    meaningful_public_history=_policy_history_options(self.policy)[0],
                )
            if bool(self.config.rust_featurize):
                entity = self._entity_batch_via_rust(
                    game,
                    colors=colors,
                    policy_action_ids=policy_action_ids,
                    acting_color=acting_color,
                    adapter=resolved[1] if resolved is not None else None,
                )
            else:
                entity = rust_game_to_entity_batch(
                    game,
                    legal_actions,
                    actor=acting_color,
                    colors=colors,
                    action_size=int(self.policy.action_size),
                    policy_action_ids=policy_action_ids,
                    public_observation=bool(self.config.public_observation),
                    meaningful_public_history=_policy_history_options(self.policy)[0],
                    history_limit=_policy_history_options(self.policy)[1],
                    resolved=resolved,
                )
            if bool(self.config.rust_featurize):
                context = self._context_batch_via_rust(
                    game,
                    acting_color=acting_color,
                    adapter=resolved[1] if resolved is not None else None,
                )
            else:
                context = rust_action_context_batch(
                    game,
                    legal_actions,
                    actor=acting_color,
                    colors=colors,
                    action_size=int(self.policy.action_size),
                    fill=float(self.config.context_fill),
                    policy_action_ids=policy_action_ids,
                    public_observation=bool(self.config.public_observation),
                    resolved=resolved,
                )
            pending_indices.append(request_index)
            pending_batch_requests.append(
                _BatchedEvalRequest(
                    entity=entity,
                    legal_action_ids=np.asarray(policy_action_ids, dtype=np.int64)[
                        None, :
                    ],
                    legal_action_context=context,
                    legal_actions=tuple(int(action) for action in legal_actions),
                    acting_color=acting_color,
                    root_color=str(root_color),
                    colors=tuple(str(color) for color in colors),
                    cache_key=cache_key,
                )
            )

        if pending_batch_requests:
            entity_batch, legal_ids, context = _merge_batched_eval_requests(
                pending_batch_requests
            )
            import torch

            with torch.no_grad():
                outputs = _forward_search_policy(
                    self.policy,
                    entity_batch,
                    legal_ids,
                    context,
                    return_q=False,
                )
            logits_batch = outputs["logits"].detach().float().cpu().numpy()
            values = self._value_output(outputs).detach().float().cpu().numpy()
            temperature = max(float(self.config.prior_temperature), 1.0e-6)
            for batch_row, (request_index, batch_request) in enumerate(
                zip(pending_indices, pending_batch_requests)
            ):
                width = len(batch_request.legal_actions)
                logits = logits_batch[batch_row, :width]
                priors_arr = _softmax(logits / temperature)
                priors = {
                    int(action): float(probability)
                    for action, probability in zip(
                        batch_request.legal_actions, priors_arr
                    )
                }
                value = self._apply_value_squash(float(values[batch_row]))
                if (
                    batch_request.acting_color != batch_request.root_color
                    and len(batch_request.colors) == 2
                ):
                    value = -value
                value = float(np.clip(value, -1.0, 1.0))
                uncertainty = _uncertainty_from_outputs(outputs, batch_row)
                if int(self.config.cache_size) > 0:
                    self._cache_store(
                        batch_request.cache_key, priors, value, uncertainty
                    )
                results[request_index] = self._eval_result(priors, value, uncertainty)

        return [
            result if result is not None else self._eval_result({}, 0.0, 0.0)
            for result in results
        ]


class BatchedEntityGraphRustEvaluator(EntityGraphRustEvaluator):
    """Thread-safe batching wrapper for neural Rust MCTS evaluation.

    RustMCTS is intentionally synchronous, but multiple search/game threads can
    share this evaluator.  Each thread prepares its entity/action tensors, then a
    single background worker pads and batches requests into one policy forward.
    That avoids loading one 35M model per actor and turns many tiny GPU kernels
    into larger, steadier batches.
    """

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        device: str = "cpu",
        config: EntityGraphRustEvaluatorConfig | None = None,
        max_batch_size: int = 64,
        max_wait_ms: float = 3.0,
    ) -> "BatchedEntityGraphRustEvaluator":
        return cls(
            EntityGraphPolicy.load(checkpoint, device=device),
            config=config,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )

    def __init__(
        self,
        policy: EntityGraphPolicy,
        *,
        config: EntityGraphRustEvaluatorConfig | None = None,
        max_batch_size: int = 64,
        max_wait_ms: float = 3.0,
    ) -> None:
        super().__init__(policy, config=config)
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0.0, float(max_wait_ms))
        self._requests: queue.Queue[_BatchedEvalRequest | None] = queue.Queue()
        self._closed = threading.Event()
        self._worker = threading.Thread(
            target=self._batch_loop,
            name="rust-mcts-batched-evaluator",
            daemon=True,
        )
        self._worker.start()
        self._observed_concurrency = False

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._requests.put(None)
        self._worker.join(timeout=5.0)

    def evaluate(
        self,
        game: Any,
        legal_actions: tuple[int, ...],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> tuple[dict[int, float], float] | tuple[dict[int, float], float, float]:
        if not legal_actions:
            return self._eval_result({}, _terminal_or_zero(game, root_color), 0.0)
        if self.max_batch_size <= 1:
            return super().evaluate(
                game, legal_actions, root_color=root_color, colors=colors
            )

        acting_color = str(game.current_color())
        cache_enabled = int(self.config.cache_size) > 0
        need_adapter_resolve = (
            not bool(self.config.rust_featurize)
        ) or self._rust_topology is None
        # B1 dedup: fetch once, share across the policy-id translation, the
        # cache key, and (on a miss) both featurizer calls below.
        snapshot_text, action_by_id = _fetch_leaf_decision_inputs(
            game,
            colors,
            include_snapshot=cache_enabled or need_adapter_resolve,
        )
        policy_action_ids = rust_policy_action_ids(
            game,
            legal_actions,
            colors=colors,
            action_size=int(self.policy.action_size),
            action_by_id=action_by_id,
        )
        # See EntityGraphRustEvaluator.evaluate(): cache_size <= 0 skips the
        # per-leaf blake2b/key-tuple work entirely, not just the store.
        cache_key = None
        if cache_enabled:
            assert snapshot_text is not None
            cache_key = (
                _state_key(game, snapshot_text=snapshot_text),
                str(root_color),
                tuple(str(color) for color in colors),
                tuple(int(action) for action in policy_action_ids),
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                # CAT-61: tolerate (priors, value) and (priors, value, unc).
                uncertainty = cached[2] if len(cached) > 2 else 0.0
                return self._eval_result(dict(cached[0]), float(cached[1]), uncertainty)

        # Same warm-topology fast path as EntityGraphRustEvaluator.evaluate()
        # and evaluate_many(): after the first native leaf, adapter construction
        # is dead work. This preamble runs in every caller thread before the
        # request reaches the batching queue, so skipping it also improves the
        # cross-game EvalServer client path inherited from this evaluator.
        resolved: (
            tuple[dict[str, Any], "_RustEntityFeatureEnv", list[dict[str, Any]]] | None
        ) = None
        if need_adapter_resolve:
            assert snapshot_text is not None
            snapshot = json.loads(snapshot_text)
            # B2 dedup: see EntityGraphRustEvaluator.evaluate() -- one shared
            # resolve for both featurizers.
            resolved = _resolve_entity_adapter(
                game,
                legal_actions,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                snapshot=snapshot,
                action_by_id=action_by_id,
                public_observation=bool(self.config.public_observation),
                perspective=acting_color,
                meaningful_public_history=_policy_history_options(self.policy)[0],
            )
        if bool(self.config.rust_featurize):
            entity = self._entity_batch_via_rust(
                game,
                colors=colors,
                policy_action_ids=policy_action_ids,
                acting_color=acting_color,
                adapter=resolved[1] if resolved is not None else None,
            )
        else:
            entity = rust_game_to_entity_batch(
                game,
                legal_actions,
                actor=acting_color,
                colors=colors,
                action_size=int(self.policy.action_size),
                policy_action_ids=policy_action_ids,
                public_observation=bool(self.config.public_observation),
                meaningful_public_history=_policy_history_options(self.policy)[0],
                history_limit=_policy_history_options(self.policy)[1],
                resolved=resolved,
            )
        if bool(self.config.rust_featurize):
            context = self._context_batch_via_rust(
                game,
                acting_color=acting_color,
                adapter=resolved[1] if resolved is not None else None,
            )
        else:
            context = rust_action_context_batch(
                game,
                legal_actions,
                actor=acting_color,
                colors=colors,
                action_size=int(self.policy.action_size),
                fill=float(self.config.context_fill),
                policy_action_ids=policy_action_ids,
                public_observation=bool(self.config.public_observation),
                resolved=resolved,
            )
        request = _BatchedEvalRequest(
            entity=entity,
            legal_action_ids=np.asarray(policy_action_ids, dtype=np.int64)[None, :],
            legal_action_context=context,
            legal_actions=tuple(int(action) for action in legal_actions),
            acting_color=acting_color,
            root_color=str(root_color),
            colors=tuple(str(color) for color in colors),
            cache_key=cache_key,
        )
        self._requests.put(request)
        request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:  # pragma: no cover - defensive.
            raise RuntimeError("batched evaluator returned no result")
        return request.result

    def _batch_loop(self) -> None:
        while not self._closed.is_set():
            first = self._get_next_request()
            if first is None:
                continue
            batch = [first]
            # Non-blocking drain: grab whatever is ALREADY queued. Genuine
            # concurrent producers (multiple threads each mid-`evaluate()`)
            # will already have their requests sitting here; a lone
            # single-threaded caller never has anything more to drain.
            while len(batch) < self.max_batch_size:
                try:
                    item = self._requests.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._closed.set()
                    break
                batch.append(item)
            # Only pay the max_wait_ms straggler timer once concurrency has
            # actually been observed (a previous batch in this evaluator's
            # lifetime already contained >1 request). A single-threaded
            # caller can never flip this flag (it cannot produce a second
            # request until this one resolves), so it never waits; a
            # genuinely concurrent caller (e.g.
            # generate_rust_mcts_reanalysis_threaded.py) still gets
            # cross-thread batching as soon as the first real burst shows up.
            if (
                len(batch) == 1
                and self._observed_concurrency
                and self.max_wait_ms > 0.0
                and not self._closed.is_set()
            ):
                deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)
                while len(batch) < self.max_batch_size:
                    timeout = max(0.0, deadline - time.perf_counter())
                    if timeout <= 0.0:
                        break
                    try:
                        item = self._requests.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if item is None:
                        self._closed.set()
                        break
                    batch.append(item)
            if len(batch) > 1:
                self._observed_concurrency = True
            self._run_batch(batch)

    def _get_next_request(self) -> "_BatchedEvalRequest | None":
        try:
            item = self._requests.get(timeout=0.05)
        except queue.Empty:
            return None
        if item is None:
            self._closed.set()
            return None
        return item

    def _run_batch(self, requests: list["_BatchedEvalRequest"]) -> None:
        try:
            entity_batch, legal_ids, context = _merge_batched_eval_requests(requests)
            import torch

            with torch.no_grad():
                outputs = _forward_search_policy(
                    self.policy,
                    entity_batch,
                    legal_ids,
                    context,
                    return_q=False,
                )
            logits_batch = outputs["logits"].detach().float().cpu().numpy()
            values = self._value_output(outputs).detach().float().cpu().numpy()
            temperature = max(float(self.config.prior_temperature), 1.0e-6)
            for index, request in enumerate(requests):
                width = len(request.legal_actions)
                logits = logits_batch[index, :width]
                priors_arr = _softmax(logits / temperature)
                priors = {
                    int(action): float(probability)
                    for action, probability in zip(request.legal_actions, priors_arr)
                }
                value = self._apply_value_squash(float(values[index]))
                if (
                    request.acting_color != request.root_color
                    and len(request.colors) == 2
                ):
                    value = -value
                value = float(np.clip(value, -1.0, 1.0))
                uncertainty = _uncertainty_from_outputs(outputs, index)
                if int(self.config.cache_size) > 0:
                    self._cache_store(request.cache_key, priors, value, uncertainty)
                request.result = self._eval_result(priors, value, uncertainty)
                request.done.set()
        except (
            BaseException
        ) as error:  # pragma: no cover - exercised in remote runtime.
            for request in requests:
                request.error = error
                request.done.set()


@dataclass(slots=True)
class _BatchedEvalRequest:
    entity: dict[str, np.ndarray]
    legal_action_ids: np.ndarray
    legal_action_context: np.ndarray
    legal_actions: tuple[int, ...]
    acting_color: str
    root_color: str
    colors: tuple[str, ...]
    cache_key: tuple[str, str, tuple[str, ...], tuple[int, ...]] | None
    done: threading.Event = field(init=False)
    result: (
        tuple[dict[int, float], float] | tuple[dict[int, float], float, float] | None
    ) = None
    error: BaseException | None = None

    def __post_init__(self) -> None:
        self.done = threading.Event()


def _merge_batched_eval_requests(
    requests: list[_BatchedEvalRequest],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    max_legal = max(int(request.legal_action_ids.shape[1]) for request in requests)
    legal_ids = np.stack(
        [
            _pad_1d_np(request.legal_action_ids[0], max_legal, fill=-1)
            for request in requests
        ],
        axis=0,
    )
    context_width = int(requests[0].legal_action_context.shape[2])
    context = np.stack(
        [
            _pad_2d_np(
                request.legal_action_context[0], max_legal, context_width, fill=0.0
            )
            for request in requests
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    entity_batch: dict[str, np.ndarray] = {}
    for key in requests[0].entity:
        values = [request.entity[key] for request in requests]
        if key == "legal_action_tokens":
            feature_size = int(values[0].shape[2])
            entity_batch[key] = np.stack(
                [
                    _pad_2d_np(value[0], max_legal, feature_size, fill=0.0)
                    for value in values
                ],
                axis=0,
            ).astype(values[0].dtype, copy=False)
        elif key == "legal_action_target_ids":
            feature_size = int(values[0].shape[2])
            entity_batch[key] = np.stack(
                [
                    _pad_2d_np(value[0], max_legal, feature_size, fill=-1)
                    for value in values
                ],
                axis=0,
            ).astype(values[0].dtype, copy=False)
        elif key == "legal_action_mask":
            entity_batch[key] = np.stack(
                [_pad_1d_np(value[0], max_legal, fill=False) for value in values],
                axis=0,
            ).astype(np.bool_, copy=False)
        else:
            entity_batch[key] = np.concatenate(values, axis=0)
    return entity_batch, legal_ids.astype(np.int64, copy=False), context


def _pad_1d_np(value: np.ndarray, width: int, *, fill: Any) -> np.ndarray:
    value = np.asarray(value)
    out = np.full((int(width),), fill, dtype=value.dtype)
    count = min(int(width), int(value.shape[0]))
    out[:count] = value[:count]
    return out


def _pad_2d_np(
    value: np.ndarray, width: int, feature_size: int, *, fill: Any
) -> np.ndarray:
    value = np.asarray(value)
    out = np.full((int(width), int(feature_size)), fill, dtype=value.dtype)
    rows = min(int(width), int(value.shape[0]))
    cols = min(int(feature_size), int(value.shape[1]))
    out[:rows, :cols] = value[:rows, :cols]
    return out


def _resolve_entity_adapter(
    game: Any,
    legal_actions: tuple[int, ...],
    *,
    colors: tuple[str, ...],
    action_size: int,
    policy_action_ids: tuple[int, ...] | None,
    snapshot: dict[str, Any] | None,
    action_by_id: dict[int, Any] | None,
    public_observation: bool = False,
    perspective: str | None = None,
    meaningful_public_history: bool = False,
) -> tuple[dict[str, Any], "_RustEntityFeatureEnv", list[dict[str, Any]]]:
    """Shared preamble for `rust_game_to_entity_batch`/`rust_action_context_batch`:
    resolve the game snapshot and the rust-action-id -> raw-json mapping (accepting
    already-fetched values from the caller to skip a second `json_snapshot`/
    `playable_action_indices`/`playable_actions_json` round trip when the caller
    already has them -- e.g. from the same node's earlier `decision_context_json`
    call or a prior call to the sibling function on the same game state), then
    build the structured legal-action list and `_RustEntityFeatureEnv` adapter
    both callers need.
    """
    if snapshot is None:
        snapshot = json.loads(game.json_snapshot())
    states_by_color = {
        str(color): json.loads(game.player_state_json(str(color))) for color in colors
    }
    if action_by_id is None:
        action_ids = [
            int(action) for action in game.playable_action_indices(list(colors), None)
        ]
        raw_actions = json.loads(game.playable_actions_json())
        action_by_id = {
            action_id: raw for action_id, raw in zip(action_ids, raw_actions)
        }
    translated = policy_action_ids or rust_policy_action_ids(
        game,
        legal_actions,
        colors=colors,
        action_size=action_size,
    )
    structured = [
        _structured_action(int(policy_action), action_by_id[int(rust_action)])
        for rust_action, policy_action in zip(legal_actions, translated)
    ]
    payload = _entity_payload_from_rust_snapshot(
        snapshot,
        states_by_color=states_by_color,
        structured_legal_actions=structured,
        legal_action_ids=legal_actions,
        public_observation=public_observation,
        perspective=perspective,
        meaningful_public_history=meaningful_public_history,
    )
    adapter = _RustEntityFeatureEnv(payload, action_size=action_size)
    return payload, adapter, structured


def rust_game_to_entity_batch(
    game: Any,
    legal_actions: tuple[int, ...],
    *,
    actor: str,
    colors: tuple[str, ...],
    action_size: int,
    policy_action_ids: tuple[int, ...] | None = None,
    snapshot: dict[str, Any] | None = None,
    action_by_id: dict[int, Any] | None = None,
    public_observation: bool = False,
    meaningful_public_history: bool = False,
    history_limit: int = 64,
    resolved: tuple[dict[str, Any], "_RustEntityFeatureEnv", list[dict[str, Any]]]
    | None = None,
) -> dict[str, np.ndarray]:
    # B2 dedup: `resolved` lets a caller that already built the shared
    # (payload, adapter, structured) tuple -- e.g. because it also needs it
    # for `rust_action_context_batch` on the same leaf -- skip a second,
    # redundant `_resolve_entity_adapter` call (snapshot re-fetch + players
    # payload rebuild + masking gate rerun). Independent callers (most
    # tools/, tests/) keep resolving on their own by leaving this None.
    if resolved is not None:
        _payload, adapter, _structured = resolved
    else:
        # The perspective player (whose own hand stays visible) IS the actor:
        # the model always evaluates from the side-to-act's point of view.
        _payload, adapter, _structured = _resolve_entity_adapter(
            game,
            legal_actions,
            colors=colors,
            action_size=action_size,
            policy_action_ids=policy_action_ids,
            snapshot=snapshot,
            action_by_id=action_by_id,
            public_observation=public_observation,
            perspective=str(actor),
            meaningful_public_history=meaningful_public_history,
        )
    entity = build_entity_token_features(
        adapter,
        actor=actor,
        include_event_log=True,
        history_limit=(
            min(int(history_limit), MEANINGFUL_PUBLIC_HISTORY_LIMIT)
            if meaningful_public_history
            else int(history_limit)
        ),
        meaningful_public_history=meaningful_public_history,
    )
    return {
        key: np.asarray(value)[None, ...]
        for key, value in entity.items()
        if key != "schema"
    }


def rust_action_context_batch(
    game: Any,
    legal_actions: tuple[int, ...],
    *,
    actor: str,
    colors: tuple[str, ...],
    action_size: int,
    fill: float = 0.0,
    policy_action_ids: tuple[int, ...] | None = None,
    snapshot: dict[str, Any] | None = None,
    action_by_id: dict[int, Any] | None = None,
    public_observation: bool = False,
    resolved: tuple[dict[str, Any], "_RustEntityFeatureEnv", list[dict[str, Any]]]
    | None = None,
) -> np.ndarray:
    # f72-class leak, found by audit: this preamble used to call
    # `_resolve_entity_adapter` with neither `public_observation` nor
    # `perspective`, so the `players` payload built here was ALWAYS unmasked
    # regardless of the evaluator's config -- the same leak `_mask_players_to_public`
    # exists to close, just on the context-feature path rather than the
    # entity-token path. Currently a no-op in practice (`_context_vector` only
    # reads public fields off `payload["players"]`: public_victory_points,
    # board/production, trade_panel), but the raw per-opponent dict it's handed
    # still carried unmasked `resources`/`development_cards`/`actual_victory_points`
    # -- one new context feature away from a real leak. Gate identically to
    # `rust_game_to_entity_batch`.
    #
    # B2 dedup: same `resolved` short-circuit as `rust_game_to_entity_batch` --
    # when the caller already built the (payload, adapter, structured) tuple
    # (already gated through the same masking above), reuse it instead of
    # re-resolving from scratch.
    if resolved is not None:
        payload, adapter, structured = resolved
    else:
        payload, adapter, structured = _resolve_entity_adapter(
            game,
            legal_actions,
            colors=colors,
            action_size=action_size,
            policy_action_ids=policy_action_ids,
            snapshot=snapshot,
            action_by_id=action_by_id,
            public_observation=public_observation,
            perspective=str(actor),
        )
    actor_public_vp = float(
        payload.get("players", {}).get(actor, {}).get("public_victory_points", 0)
    )
    prompt = str(payload.get("current_prompt", ""))
    rows = np.full(
        (len(legal_actions), CONTEXT_ACTION_FEATURE_SIZE),
        float(fill),
        dtype=np.float32,
    )
    for row, action in enumerate(structured):
        rows[row] = _context_vector(
            adapter,
            action,
            valid=True,
            actor_public_vp=actor_public_vp,
            payload=payload,
            prompt=prompt,
        )
    return rows[None, ...]


def rust_policy_action_ids(
    game: Any,
    legal_actions: tuple[int, ...],
    *,
    colors: tuple[str, ...],
    action_size: int,
    action_by_id: dict[int, Any] | None = None,
) -> tuple[int, ...]:
    if action_by_id is None:
        action_ids = [
            int(action) for action in game.playable_action_indices(list(colors), None)
        ]
        raw_actions = json.loads(game.playable_actions_json())
        raw_by_id = {action_id: raw for action_id, raw in zip(action_ids, raw_actions)}
    else:
        raw_by_id = action_by_id
    catalog = _policy_action_index_by_key(tuple(str(color) for color in colors))
    translated: list[int] = []
    for action in legal_actions:
        raw = raw_by_id[int(action)]
        key = _raw_action_key(raw)
        mapped = catalog.get(key)
        if mapped is None:
            raise ValueError(
                f"could not map Rust action to policy action id: {raw!r} key={key!r}"
            )
        if not 0 <= int(mapped) < int(action_size):
            raise ValueError(
                f"mapped action id out of range: rust={action} mapped={mapped}"
            )
        translated.append(int(mapped))
    return tuple(translated)


@lru_cache(maxsize=8)
def _policy_action_index_by_key(colors: tuple[str, ...]) -> dict[tuple[str, Any], int]:
    catalog = ActionCatalog(colors)
    return {
        (str(descriptor["action_type"]), _canonical_value(descriptor["value"])): int(
            index
        )
        for index in range(catalog.size)
        for descriptor in (catalog.describe(index),)
    }


def _raw_action_key(raw: Any) -> tuple[str, Any]:
    parts = list(raw) if isinstance(raw, (list, tuple)) else []
    action_type = str(parts[1] if len(parts) > 1 else "")
    value = parts[2] if len(parts) > 2 else None
    if action_type == "MOVE_ROBBER":
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            coordinate, victim = value[0], value[1]
            return action_type, (_canonical_value(coordinate), _canonical_value(victim))
        victim = parts[3] if len(parts) > 3 else None
        return action_type, (_canonical_value(value), _canonical_value(victim))
    if action_type in {
        "ROLL",
        "END_TURN",
        "BUY_DEVELOPMENT_CARD",
        "PLAY_KNIGHT_CARD",
        "PLAY_ROAD_BUILDING",
    }:
        return action_type, None
    if action_type == "BUILD_ROAD":
        edge = (
            tuple(sorted(int(node) for node in value))
            if isinstance(value, (list, tuple))
            else value
        )
        return action_type, _canonical_value(edge)
    return action_type, _canonical_value(value)


def _canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        upper = value.upper()
        if upper in {
            "WOOD",
            "BRICK",
            "SHEEP",
            "WHEAT",
            "ORE",
            "BLUE",
            "RED",
            "ORANGE",
            "WHITE",
        }:
            return upper
        return value
    if isinstance(value, (int, float, bool)):
        return (
            int(value)
            if isinstance(value, bool) is False and float(value).is_integer()
            else value
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _canonical_value(raw)) for key, raw in value.items())
        )
    name = getattr(value, "name", None)
    if name is not None:
        return _canonical_value(str(name))
    return value


class _RustEntityFeatureEnv:
    def __init__(self, payload: dict[str, Any], *, action_size: int) -> None:
        self._payload = payload
        self.action_space = SimpleNamespace(n=int(action_size))
        self.game = SimpleNamespace(
            state=SimpleNamespace(
                board=SimpleNamespace(
                    map=SimpleNamespace(
                        node_production=payload.get("_node_production", {})
                    )
                )
            )
        )

    def observation_payload(
        self,
        actor_name: str,
        *,
        include_event_log: bool = True,
    ) -> dict[str, Any]:
        del actor_name, include_event_log
        return self._payload

    def current_player_name(self) -> str:
        return str(self._payload.get("current_player", ""))


def _entity_payload_from_rust_snapshot(
    snapshot: dict[str, Any],
    *,
    states_by_color: dict[str, dict[str, Any]],
    structured_legal_actions: list[dict[str, Any]],
    legal_action_ids: tuple[int, ...],
    public_observation: bool = False,
    perspective: str | None = None,
    meaningful_public_history: bool = False,
) -> dict[str, Any]:
    colors = tuple(str(color) for color in snapshot.get("colors", ()))
    robber = tuple(snapshot.get("robber_coordinate") or ())
    tiles = []
    port_tiles = []
    for raw in snapshot.get("tiles", ()):
        tile = raw.get("tile") if isinstance(raw, dict) else {}
        if not isinstance(tile, dict):
            continue
        coordinate = raw.get("coordinate")
        coordinate_key = _coordinate(coordinate)
        tile_type = str(tile.get("type", ""))
        if tile_type == "PORT":
            port_tiles.append(raw)
            continue
        if tile_type not in {"RESOURCE_TILE", "DESERT"}:
            continue
        topology_tile = _base_tile_topology().get(coordinate_key or ())
        tile_id = _safe_int(tile.get("id"), default=len(tiles))
        if tile_id is None or not 0 <= int(tile_id) < 19:
            tile_id = len(tiles)
        nodes = dict((topology_tile or {}).get("nodes", {}))
        edges = dict((topology_tile or {}).get("edges", {}))
        tiles.append(
            {
                "tile_id": int(tile_id),
                "coordinate": coordinate,
                "resource": _resource_name(tile.get("resource")),
                "number": tile.get("number", 0),
                "has_robber": tuple(coordinate or ()) == robber,
                "nodes": nodes,
                "edges": edges,
            }
        )

    buildings = []
    for node in snapshot.get("nodes", {}).values():
        if not isinstance(node, dict) or node.get("building") is None:
            continue
        buildings.append(
            {
                "node": int(node.get("id", -1)),
                "player": str(node.get("color", "")),
                "building_type": str(node.get("building", "")),
            }
        )

    roads = []
    for edge in snapshot.get("edges", ()):
        if not isinstance(edge, dict) or edge.get("color") is None:
            continue
        roads.append({"edge": edge.get("id"), "player": str(edge.get("color", ""))})

    ports = _ports_from_rust_tiles(port_tiles, snapshot)
    if len(ports) < 9:
        ports = _base_ports()
    node_production = _node_production(tiles)
    players = _players_from_rust_snapshot(snapshot, colors, states_by_color)
    if public_observation:
        players = _mask_players_to_public(players, perspective)
    return {
        "board": {
            "tiles": tiles,
            "buildings": buildings,
            "roads": roads,
            "ports": ports,
        },
        "players": players,
        "current_player": str(snapshot.get("current_color", "")),
        "current_prompt": str(snapshot.get("current_prompt", "")),
        "structured_legal_actions": structured_legal_actions,
        "legal_actions": list(legal_action_ids),
        "event_log": (
            public_events_from_native_action_records(
                snapshot.get("action_records", ()),
                snapshot.get("action_public_legal_counts", ()),
            )
            if meaningful_public_history
            else []
        ),
        "replay_frame_count": int(snapshot.get("state_index", 0) or 0),
        "bank": {
            "resources": {
                _resource_name(key): int(value)
                for key, value in dict(snapshot.get("resource_bank", {})).items()
            },
            "development_cards_remaining": int(
                snapshot.get("development_deck_count", 0) or 0
            ),
        },
        "trade_panel": {
            "offers_remaining": 0,
            "current_offer": None,
            "is_resolving": bool(snapshot.get("is_resolving_trade", False)),
        },
        "_node_production": node_production,
    }


def _players_from_rust_snapshot(
    snapshot: dict[str, Any],
    colors: tuple[str, ...],
    states_by_color: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    players: dict[str, dict[str, Any]] = {}
    longest = dict(snapshot.get("longest_roads_by_player", {}))
    for color in colors:
        state = states_by_color.get(str(color), {})
        resources = _resource_counts(state.get("resources"))
        dev_cards = _dev_card_counts(state.get("dev_cards"))
        played = _dev_card_counts(state.get("played_dev_cards"))
        players[color] = {
            "public_victory_points": int(state.get("victory_points", 0) or 0),
            "actual_victory_points": int(state.get("actual_victory_points", 0) or 0),
            "resource_card_count": sum(resources.values()),
            "development_card_count": sum(dev_cards.values()),
            "resources": resources,
            "development_cards": dev_cards,
            "played_development_cards": played,
            "roads_left": int(state.get("roads_available", 0) or 0),
            "settlements_left": int(state.get("settlements_available", 0) or 0),
            "cities_left": int(state.get("cities_available", 0) or 0),
            "has_largest_army": bool(state.get("has_army", False)),
            # Longest-road ownership is public information and is already
            # maintained authoritatively by the Rust engine.  Preserve it in
            # the Python adapter just like largest-army ownership; hardcoding
            # this to False made native inference disagree with the training
            # feature surface after the award changed hands.
            "has_longest_road": bool(state.get("has_road", False)),
            "has_rolled": bool(state.get("has_rolled", False)),
            "longest_road_length": int(
                state.get("longest_road_length", longest.get(color, 0)) or 0
            ),
        }
    return players


def _mask_players_to_public(
    players: dict[str, dict[str, Any]], perspective: str | None
) -> dict[str, dict[str, Any]]:
    """Return a copy of `players` with every OPPONENT's hidden fields dropped,
    keeping the actor (`perspective`) fully visible.

    Drops `resources` and `development_cards` composition (so `_player_tokens`
    emits their has-flags as 0 and the composition slots as 0) and removes
    `actual_victory_points` (so `has_actual` is 0 and public VP alone remains).
    The public counts `resource_card_count`/`development_card_count`,
    `public_victory_points`, and `played_development_cards` are preserved --
    they are visible to all players. This is the online twin of the load-time
    `entity_token_features.mask_player_tokens_public`; the two MUST agree
    slot-for-slot (asserted by tests/test_public_observation_masking.py).

    If `perspective` is None (no side-to-act identified) every player is masked,
    the conservative choice -- but in practice the evaluator always passes the
    acting color, so exactly one player (the actor) stays visible.
    """
    masked: dict[str, dict[str, Any]] = {}
    for color, player in players.items():
        if perspective is not None and str(color) == str(perspective):
            masked[color] = player
            continue
        public = dict(player)
        public["resources"] = None
        public["development_cards"] = None
        public.pop("actual_victory_points", None)
        masked[color] = public
    return masked


def _structured_action(action_id: int, raw: Any) -> dict[str, Any]:
    parts = list(raw) if isinstance(raw, (list, tuple)) else []
    action_type = str(parts[1] if len(parts) > 1 else "")
    value = parts[2] if len(parts) > 2 else None
    args: dict[str, Any] = {}
    if action_type in {"BUILD_SETTLEMENT", "BUILD_CITY"}:
        args["node"] = value
    elif action_type == "BUILD_ROAD":
        args["edge"] = value
    elif action_type == "MOVE_ROBBER":
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (list, tuple))
        ):
            args["tile_coordinate"] = list(value[0][:3])
            args["victim"] = value[1]
        elif isinstance(value, (list, tuple)) and len(value) >= 3:
            args["tile_coordinate"] = list(value[:3])
        elif len(parts) > 3:
            args["victim"] = parts[3]
    elif action_type in {"DISCARD_RESOURCE", "PLAY_MONOPOLY"}:
        args["resource"] = _resource_name(value)
    elif action_type == "MARITIME_TRADE":
        if isinstance(value, dict):
            args.update(value)
        elif isinstance(value, (list, tuple)):
            give = [_resource_name(item) for item in value[:-1] if item is not None]
            want = _resource_name(value[-1]) if value else None
            args["give"] = [item for item in give if item is not None]
            args["want"] = [want] if want is not None else []
    return {
        "index": int(action_id),
        "action_type": action_type,
        "category": _action_category(action_type),
        "args": args,
    }


def _ports_from_rust_tiles(
    port_tiles: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    del snapshot
    base_by_id = _base_ports_by_id()
    ports = []
    for raw in port_tiles:
        tile = raw.get("tile") if isinstance(raw, dict) else {}
        port_id = _safe_int(tile.get("id"))
        base = base_by_id.get(int(port_id)) if port_id is not None else None
        if base is None:
            continue
        ports.append(
            {
                "port_id": int(port_id),
                "nodes": tuple(int(node) for node in base.get("nodes", ())),
                "resource": _resource_name(tile.get("resource")),
            }
        )
    return ports


@lru_cache(maxsize=1)
def _base_tile_topology() -> dict[tuple[int, int, int], dict[str, Any]]:
    env = ColonistMultiAgentEnv(
        ColonistMultiAgentConfig(
            players=2,
            vps_to_win=10,
            max_player_trade_offers_per_turn=0,
            enable_table_chat=False,
            enable_timers=False,
        )
    )
    env.reset(seed=0)
    board = env.observation_payload("BLUE", include_event_log=False)["board"]
    return {
        _coordinate(tile.get("coordinate")) or (): {
            "nodes": dict(tile.get("nodes", {})),
            "edges": dict(tile.get("edges", {})),
        }
        for tile in board.get("tiles", ())
        if isinstance(tile, dict)
    }


@lru_cache(maxsize=1)
def _base_ports() -> list[dict[str, Any]]:
    env = ColonistMultiAgentEnv(
        ColonistMultiAgentConfig(
            players=2,
            vps_to_win=10,
            max_player_trade_offers_per_turn=0,
            enable_table_chat=False,
            enable_timers=False,
        )
    )
    env.reset(seed=0)
    board = env.observation_payload("BLUE", include_event_log=False)["board"]
    return [
        {
            "port_id": int(port.get("port_id", index)),
            "nodes": tuple(int(node) for node in port.get("nodes", ())),
            "resource": _resource_name(port.get("resource")),
        }
        for index, port in enumerate(board.get("ports", ()))
        if isinstance(port, dict)
    ]


@lru_cache(maxsize=1)
def _base_ports_by_id() -> dict[int, dict[str, Any]]:
    return {
        int(port.get("port_id", index)): port
        for index, port in enumerate(_base_ports())
    }


def _node_production(tiles: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for tile in tiles:
        resource = tile.get("resource")
        number = _safe_int(tile.get("number"), default=0) or 0
        pips = max(0, 6 - abs(int(number) - 7)) if int(number) != 7 else 0
        if not resource or not pips:
            continue
        for node in dict(tile.get("nodes", {})).values():
            node_id = int(node)
            out.setdefault(node_id, {})
            out[node_id][str(resource).lower()] = (
                out[node_id].get(str(resource).lower(), 0) + pips
            )
    return out


def _terminal_or_zero(game: Any, root_color: str) -> float:
    winner = game.winning_color()
    if winner is None:
        return 0.0
    return 1.0 if str(winner) == str(root_color) else -1.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    total = float(np.sum(exp))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=np.float64)
    return exp / total


def _state_key(game: Any, *, snapshot_text: str | None = None) -> str:
    text = snapshot_text if snapshot_text is not None else game.json_snapshot()
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _action_category(action_type: str) -> str:
    if action_type in {"BUILD_SETTLEMENT", "BUILD_ROAD", "BUILD_CITY"}:
        return "build"
    if "TRADE" in action_type:
        return "trade"
    if "ROBBER" in action_type or action_type == "PLAY_KNIGHT_CARD":
        return "robber"
    if "DEVELOPMENT" in action_type or action_type.startswith("PLAY_"):
        return "development"
    return "turn"


def _resource_name(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).lower()
    return {
        "wood": "wood",
        "brick": "brick",
        "sheep": "sheep",
        "wheat": "wheat",
        "ore": "ore",
    }.get(raw)


def _resource_counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {
            str(name): int(count)
            for name, count in (
                (_resource_name(key), raw) for key, raw in value.items()
            )
            if name is not None
        }
    if isinstance(value, (list, tuple)):
        # Rust serializes player hands in Resource enum order.
        names = ("wood", "brick", "sheep", "wheat", "ore")
        return {
            name: int(value[index] or 0)
            for index, name in enumerate(names[: len(value)])
        }
    return {}


def _dev_card_counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        names = (
            "KNIGHT",
            "YEAR_OF_PLENTY",
            "MONOPOLY",
            "ROAD_BUILDING",
            "VICTORY_POINT",
        )
        return {
            name: int(value[index] or 0)
            for index, name in enumerate(names[: len(value)])
        }
    return {}


def _safe_int(value: Any, *, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coordinate(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return int(value[0]), int(value[1]), int(value[2])
    except (TypeError, ValueError):
        return None
