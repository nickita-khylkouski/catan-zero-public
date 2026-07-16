"""Opt-in native hot loop for the production Gumbel search.

Only tree traversal/bookkeeping moves to Rust.  The established Python layer
still owns information-set particle construction/aggregation and the neural
evaluator.  This keeps P4/min32, public-observation checks, and the Python
fallback as one source of truth while removing the per-simulation Python loop.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    SearchResult,
    _UNATTESTED_ROOT_PHASE,
    _UNSET_ROOT_EVALUATION,
    _matches_explicit_or_legacy_width_gate,
)


def native_hot_loop_available() -> bool:
    try:
        import catanatron_rs  # type: ignore
    except ImportError:
        return False
    return callable(getattr(catanatron_rs, "gumbel_search", None))


class NativeGumbelChanceMCTS(GumbelChanceMCTS):
    """Reference-compatible search with an explicitly enabled Rust hot loop."""

    def __init__(
        self,
        config: GumbelChanceMCTSConfig | None = None,
        evaluator: Any | None = None,
        *,
        allow_python_fallback: bool = False,
    ) -> None:
        super().__init__(config, evaluator)
        self._leaf_evaluation_observer: (
            Callable[[Any, tuple[int, ...], str], None] | None
        ) = None
        self.using_native_hot_loop = native_hot_loop_available()
        self._validate_native_semantics()
        if not self.using_native_hot_loop and not allow_python_fallback:
            raise RuntimeError(
                "native Gumbel hot loop requested but catanatron_rs.gumbel_search "
                "is unavailable; install the matching native wheel or explicitly "
                "set allow_python_fallback=True"
            )

    def set_leaf_evaluation_observer(
        self,
        observer: Callable[[Any, tuple[int, ...], str], None] | None,
    ) -> None:
        """Observe native leaf queries without confusing them with the root.

        The PyO3 bridge already hands Python an independent clone of every
        evaluated game. A bounded frontier recorder may retain selected clones
        safely; the default ``None`` path adds no featurization or persistence.
        """

        if observer is not None and not callable(observer):
            raise TypeError("leaf evaluation observer must be callable or None")
        self._leaf_evaluation_observer = observer

    def _observe_leaf_evaluation(
        self,
        native_game: Any,
        legal: tuple[int, ...],
        root_color: str,
    ) -> None:
        observer = getattr(self, "_leaf_evaluation_observer", None)
        if observer is not None:
            observer(native_game, legal, str(root_color))

    def _validate_native_semantics(self) -> None:
        unsupported: list[str] = []
        if not bool(self.config.correct_rust_chance_spectra):
            unsupported.append("correct_rust_chance_spectra=False")
        if bool(self.config.belief_chance_spectra):
            unsupported.append("belief_chance_spectra=True")
        if bool(self.config.root_wave_batching):
            unsupported.append("root_wave_batching=True")
        if not bool(self.config.use_batch_api):
            unsupported.append("use_batch_api=False")
        if bool(self.config.uncertainty_backup_weighting):
            unsupported.append("uncertainty_backup_weighting=True")
        if self.using_native_hot_loop and bool(
            self.config.rescale_noise_floor_initial_road_only
        ):
            import catanatron_rs  # type: ignore

            capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
            capabilities = set(capability_fn()) if callable(capability_fn) else set()
            if "initial_road_d1_scope" not in capabilities:
                unsupported.append(
                    "rescale_noise_floor_initial_road_only requires a native wheel "
                    "advertising initial_road_d1_scope"
                )
        if (
            self.using_native_hot_loop
            and self.config.sigma_reference_visits is not None
        ):
            import catanatron_rs  # type: ignore

            capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
            capabilities = set(capability_fn()) if callable(capability_fn) else set()
            if "sigma_reference_visits" not in capabilities:
                unsupported.append(
                    "sigma_reference_visits requires a native wheel advertising "
                    "the matching calibration capability"
                )
        if (
            self.using_native_hot_loop
            and float(self.config.temperature) > 0.0
            and float(self.config.temperature) != 1.0
        ):
            import catanatron_rs  # type: ignore

            capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
            capabilities = set(capability_fn()) if callable(capability_fn) else set()
            if "policy_temperature_semantics" not in capabilities:
                unsupported.append(
                    "non-unit gameplay temperature requires a native wheel "
                    "advertising policy_temperature_semantics"
                )
        if self.using_native_hot_loop and (
            self.config.information_set_target_aggregation == "aggregate_q_then_improve"
            or self.config.gameplay_policy_aggregation == "aggregate_q_then_improve"
        ):
            import catanatron_rs  # type: ignore

            capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
            capabilities = set(capability_fn()) if callable(capability_fn) else set()
            if "belief_target_evidence" not in capabilities:
                unsupported.append(
                    "aggregate_q_then_improve belief aggregation requires a native wheel advertising "
                    "belief_target_evidence"
                )
        if self.using_native_hot_loop and bool(
            self.config.coherent_public_belief_search
        ):
            import catanatron_rs  # type: ignore

            capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
            capabilities = set(capability_fn()) if callable(capability_fn) else set()
            if "coherent_public_belief_search" not in capabilities:
                unsupported.append(
                    "coherent_public_belief_search requires a native wheel "
                    "advertising coherent_public_belief_search"
                )
            if (
                int(self.config.boundary_value_particles) > 1
                and "boundary_value_particles" not in capabilities
            ):
                unsupported.append(
                    "boundary_value_particles requires a native wheel "
                    "advertising boundary_value_particles"
                )
        if (
            self.using_native_hot_loop
            and self.config.forced_root_target_mode == "trajectory_only"
        ):
            import catanatron_rs  # type: ignore

            capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
            capabilities = set(capability_fn()) if callable(capability_fn) else set()
            if "forced_root_trajectory_only" not in capabilities:
                unsupported.append(
                    "forced_root_target_mode=trajectory_only requires a native "
                    "wheel advertising forced_root_trajectory_only"
                )
        if unsupported:
            raise ValueError(
                "native MCTS hot loop does not implement the requested reference "
                "semantics: "
                + ", ".join(unsupported)
                + "; refusing silent operator drift"
            )

    def _native_config(
        self,
        *,
        n_simulations_override: int | None = None,
        attested_root_phase: str | None = None,
    ) -> dict[str, Any]:
        config = self.config
        values = {
            name: getattr(config, name)
            for name in (
                "max_depth",
                "seed",
                "c_visit",
                "c_scale",
                "sigma_reference_visits",
                "temperature",
                "play_sh_winner",
                "prior_temperature",
                "n_full",
                "n_fast",
                "p_full",
                "forced_root_target_mode",
                "lazy_interior_chance",
                "max_root_candidates",
                "max_root_candidates_wide",
                "wide_candidates_threshold",
                "exact_budget_sh",
                "exact_budget_sh_min_n",
                "rescale_noise_floor_c",
                "sigma_eval",
                "variance_aware_q",
                "variance_aware_k",
                "variance_aware_closed_form_js",
                "uncertainty_backup_weighting",
                "uncertainty_backup_a",
                "uncertainty_backup_exp",
                "uncertainty_backup_cap",
                "policy_target_min_visits",
                "wide_roots_always_full",
            )
        }
        values.update(
            colors=list(config.colors),
            map_kind=config.map_kind or "BASE",
            stop_at_root_turn_boundary=bool(
                config.information_set_search or config.coherent_public_belief_search
            ),
            coherent_public_belief_search=bool(config.coherent_public_belief_search),
            # A binding call constructs a fresh Rust engine. Seed it from the
            # reference search object's ADVANCING RNG rather than resetting to
            # config.seed on every move/particle. This is deterministic across
            # identically seeded search objects but distinct within one run.
            seed=self.rng.getrandbits(64),
        )
        values["rescale_noise_floor_initial_road_only"] = bool(
            config.rescale_noise_floor_initial_road_only
        )
        if bool(config.rescale_noise_floor_initial_road_only):
            if not isinstance(attested_root_phase, str) or not attested_root_phase:
                raise RuntimeError(
                    "native initial-road-only D1 requires an authoritative root-phase "
                    "attestation"
                )
            values["attested_root_phase"] = attested_root_phase
        for optional in (
            "n_full_wide",
            "n_full_wide_threshold",
            "raw_policy_above_width",
            "root_candidate_cap",
        ):
            value = getattr(config, optional)
            if value is not None:
                values[optional] = value
        if n_simulations_override is not None:
            # A PIMC particle receives an exact fraction of the locked TOTAL
            # budget; do not let legacy SH phase rounding multiply it.
            budget = max(int(n_simulations_override), 1)
            values.update(
                n_full=budget,
                n_fast=budget,
                p_full=1.0,
                exact_budget_sh=True,
                exact_budget_sh_min_n=0,
            )
            values.pop("n_full_wide", None)
        particle_seeds = tuple(
            int(seed)
            for seed in getattr(self, "_boundary_value_particle_seeds", ())
        )
        if len(particle_seeds) > 1:
            values["boundary_value_particle_seeds"] = list(particle_seeds)
        return values

    def _search_single_world(
        self,
        game: Any,
        *,
        force_full: bool | None = None,
        n_simulations_override: int | None = None,
        attested_root_phase: str | None | object = _UNATTESTED_ROOT_PHASE,
        precomputed_root_evaluation: Any = _UNSET_ROOT_EVALUATION,
    ) -> SearchResult:
        if not self.using_native_hot_loop:
            return super()._search_single_world(
                game,
                force_full=force_full,
                n_simulations_override=n_simulations_override,
                attested_root_phase=attested_root_phase,
                precomputed_root_evaluation=precomputed_root_evaluation,
            )

        import catanatron_rs  # type: ignore

        colors = tuple(self.config.colors)

        def evaluate(native_game: Any, legal: list[int], root_color: str):
            legal_tuple = tuple(int(action) for action in legal)
            self._observe_leaf_evaluation(native_game, legal_tuple, root_color)
            return self.evaluator.evaluate(
                native_game, legal_tuple, root_color=root_color, colors=colors
            )

        evaluate_many = None
        if callable(getattr(self.evaluator, "evaluate_many", None)):

            def evaluate_many(requests: list[tuple[Any, list[int], str]]):
                if not requests:
                    return []
                root_colors = {str(request[2]) for request in requests}
                if len(root_colors) != 1:
                    raise RuntimeError(
                        "native evaluation batch mixed root perspectives"
                    )
                normalized = [
                    (request[0], tuple(int(action) for action in request[1]))
                    for request in requests
                ]
                for native_game, legal in normalized:
                    self._observe_leaf_evaluation(
                        native_game, legal, str(requests[0][2])
                    )
                return self.evaluator.evaluate_many(
                    normalized,
                    root_color=next(iter(root_colors)),
                    colors=colors,
                )

        # Always provide a distinct root callback.  The native bridge otherwise
        # falls back to ``evaluate`` and a frontier recorder would silently
        # capture trajectory roots as counterfactual leaves.
        def root_evaluator(native_game: Any, legal: list[int], root_color: str):
            return self.evaluator.evaluate(
                native_game,
                tuple(int(action) for action in legal),
                root_color=root_color,
                colors=colors,
            )

        native_legal = tuple(
            int(action)
            for action in game.playable_action_indices(
                list(colors), self.config.map_kind
            )
        )
        if (
            len(native_legal) == 1
            and self.config.forced_root_target_mode == "trajectory_only"
        ):
            return self._forced_trajectory_only_result(native_legal[0])
        legal_width = len(native_legal)
        if precomputed_root_evaluation is not _UNSET_ROOT_EVALUATION:

            def root_evaluator(_native_game: Any, _legal: list[int], _root_color: str):
                result = precomputed_root_evaluation
                # Match the reference expansion boundary: each particle gets
                # its own mutable prior mapping while values/uncertainty retain
                # their exact evaluator representation.
                return (dict(result[0]), *result[1:])

        elif (
            bool(self.config.symmetry_averaged_eval)
            and _matches_explicit_or_legacy_width_gate(
                legal_width,
                min_legal_actions=self.config.symmetry_averaged_eval_threshold,
                legacy_exclusive_threshold=self.config.wide_candidates_threshold,
            )
            and hasattr(self.evaluator, "evaluate_symmetry_averaged")
        ):

            def root_evaluator(native_game: Any, legal: list[int], root_color: str):
                return self.evaluator.evaluate_symmetry_averaged(
                    native_game, tuple(legal), root_color=root_color, colors=colors
                )

        boundary_evaluator = None
        if int(self.config.boundary_value_particles) > 1:

            def boundary_evaluator(
                native_game: Any,
                root_color: str,
                particle_seeds: list[int],
            ) -> float:
                if len(particle_seeds) < 2:
                    raise RuntimeError(
                        "native boundary evaluator requires at least two seeds"
                    )
                determinize = getattr(
                    native_game, "determinize_from_observer_information", None
                )
                if not callable(determinize):
                    raise RuntimeError(
                        "native boundary evaluator requires "
                        "determinize_from_observer_information"
                    )
                requests: list[tuple[Any, tuple[int, ...]]] = []
                for seed in particle_seeds:
                    sampled = determinize(str(root_color), int(seed))
                    legal = tuple(
                        int(action)
                        for action in sampled.playable_action_indices(
                            list(colors), self.config.map_kind
                        )
                    )
                    if not legal:
                        raise RuntimeError(
                            "boundary determinization produced no legal actions"
                        )
                    requests.append((sampled, legal))
                evaluations = list(
                    self.evaluator.evaluate_many(
                        requests,
                        root_color=str(root_color),
                        colors=colors,
                    )
                )
                if len(evaluations) != len(requests):
                    raise RuntimeError(
                        "boundary evaluator batch cardinality mismatch"
                    )
                values = [float(result[1]) for result in evaluations]
                if not values or not all(math.isfinite(value) for value in values):
                    raise RuntimeError(
                        "boundary evaluator produced non-finite values"
                    )
                return float(sum(values) / len(values))

        root_phase = self._resolve_d1_root_phase(game, attested_root_phase)
        native_kwargs = {
            "evaluator_many": evaluate_many,
            "root_evaluator": root_evaluator,
            "force_full": force_full,
        }
        # Preserve compatibility with pre-boundary-particle wheels on the K=1
        # path: do not pass the newly added keyword unless it is load-bearing.
        if boundary_evaluator is not None:
            native_kwargs["boundary_evaluator"] = boundary_evaluator
        raw = catanatron_rs.gumbel_search(
            game,
            evaluate,
            self._native_config(
                n_simulations_override=n_simulations_override,
                attested_root_phase=root_phase,
            ),
            **native_kwargs,
        )
        return SearchResult(
            selected_action=int(raw["selected_action"]),
            improved_policy={
                int(key): float(value) for key, value in raw["improved_policy"].items()
            },
            visit_counts={
                int(key): int(value) for key, value in raw["visit_counts"].items()
            },
            q_values={int(key): float(value) for key, value in raw["q_values"].items()},
            priors={int(key): float(value) for key, value in raw["priors"].items()},
            root_value=float(raw["root_value"]),
            used_full_search=bool(raw["used_full_search"]),
            simulations_used=int(raw["simulations_used"]),
            afterstate_values={
                int(key): float(value)
                for key, value in raw["afterstate_values"].items()
            },
            completed_q_values={
                int(key): float(value)
                for key, value in raw.get("completed_q_values", {}).items()
            },
            q_values_root_perspective=bool(raw.get("q_values_root_perspective", False)),
        )


def create_gumbel_search(
    config: GumbelChanceMCTSConfig,
    evaluator: Any,
    *,
    native_hot_loop: bool = False,
    allow_python_fallback: bool = False,
) -> GumbelChanceMCTS:
    """Build search without changing the historical default implementation."""
    if not native_hot_loop:
        return GumbelChanceMCTS(config, evaluator)
    return NativeGumbelChanceMCTS(
        config, evaluator, allow_python_fallback=allow_python_fallback
    )
