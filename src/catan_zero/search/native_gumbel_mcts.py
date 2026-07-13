"""Opt-in native hot loop for the production Gumbel search.

Only tree traversal/bookkeeping moves to Rust.  The established Python layer
still owns information-set particle construction/aggregation and the neural
evaluator.  This keeps P4/min32, public-observation checks, and the Python
fallback as one source of truth while removing the per-simulation Python loop.
"""

from __future__ import annotations

from typing import Any

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    SearchResult,
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
        self.using_native_hot_loop = native_hot_loop_available()
        self._validate_native_semantics()
        if not self.using_native_hot_loop and not allow_python_fallback:
            raise RuntimeError(
                "native Gumbel hot loop requested but catanatron_rs.gumbel_search "
                "is unavailable; install the matching native wheel or explicitly "
                "set allow_python_fallback=True"
            )

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
        if self.using_native_hot_loop and self.config.sigma_reference_visits is not None:
            import catanatron_rs  # type: ignore

            capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
            capabilities = set(capability_fn()) if callable(capability_fn) else set()
            if "sigma_reference_visits" not in capabilities:
                unsupported.append(
                    "sigma_reference_visits requires a native wheel advertising "
                    "the matching calibration capability"
                )
        if unsupported:
            raise ValueError(
                "native MCTS hot loop does not implement the requested reference "
                "semantics: "
                + ", ".join(unsupported)
                + "; refusing silent operator drift"
            )

    def _native_config(
        self, *, n_simulations_override: int | None = None
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
            stop_at_root_turn_boundary=bool(config.information_set_search),
            # A binding call constructs a fresh Rust engine. Seed it from the
            # reference search object's ADVANCING RNG rather than resetting to
            # config.seed on every move/particle. This is deterministic across
            # identically seeded search objects but distinct within one run.
            seed=self.rng.getrandbits(64),
        )
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
        return values

    def _search_single_world(
        self,
        game: Any,
        *,
        force_full: bool | None = None,
        n_simulations_override: int | None = None,
    ) -> SearchResult:
        if not self.using_native_hot_loop:
            return super()._search_single_world(
                game,
                force_full=force_full,
                n_simulations_override=n_simulations_override,
            )

        import catanatron_rs  # type: ignore

        colors = tuple(self.config.colors)

        def evaluate(native_game: Any, legal: list[int], root_color: str):
            return self.evaluator.evaluate(
                native_game, tuple(legal), root_color=root_color, colors=colors
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
                return self.evaluator.evaluate_many(
                    [(request[0], tuple(request[1])) for request in requests],
                    root_color=next(iter(root_colors)),
                    colors=colors,
                )

        root_evaluator = None
        legal_width = len(
            game.playable_action_indices(list(colors), self.config.map_kind)
        )
        if (
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

        raw = catanatron_rs.gumbel_search(
            game,
            evaluate,
            self._native_config(n_simulations_override=n_simulations_override),
            evaluator_many=evaluate_many,
            root_evaluator=root_evaluator,
            force_full=force_full,
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
