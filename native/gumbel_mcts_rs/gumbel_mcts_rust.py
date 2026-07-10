"""Python wrapper for the native Rust Gumbel MCTS (v2 — fully native).

Drop-in replacement for catan_zero.search.gumbel_chance_mcts.GumbelChanceMCTS.

The Rust crate extracts the Game from PyGame ONCE, then holds it natively.
All game operations (legal actions, chance outcomes, winner check, game copy)
happen in Rust. Only the neural network forward pass crosses to Python.
"""

from __future__ import annotations

from typing import Any, Optional

import catanatron_rs
from catanatron_rs import Game as PyGame

# gumbel_search is compiled into the catanatron_rs extension module
# (catanatron-rs/python/src/lib.rs). The standalone gumbel_mcts extension
# is the older build without root_color support.
_gumbel_search = catanatron_rs.gumbel_search


class GumbelChanceMCTSRust:
    """Drop-in replacement for GumbelChanceMCTS using the native Rust engine."""

    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.evaluator_many = getattr(evaluator, 'evaluate_many', None)

    def _build_config_dict(self) -> dict:
        c = self.config
        d = {
            "max_depth": c.max_depth,
            "seed": c.seed,
            "c_visit": c.c_visit,
            "c_scale": c.c_scale,
            "temperature": c.temperature,
            "play_sh_winner": c.play_sh_winner,
            "prior_temperature": c.prior_temperature,
            "n_full": c.n_full,
            "n_fast": c.n_fast,
            "p_full": c.p_full,
            "lazy_interior_chance": c.lazy_interior_chance,
            "max_root_candidates": c.max_root_candidates,
            "max_root_candidates_wide": c.max_root_candidates_wide,
            "wide_candidates_threshold": c.wide_candidates_threshold,
            "exact_budget_sh": c.exact_budget_sh,
            "exact_budget_sh_min_n": c.exact_budget_sh_min_n,
            "rescale_noise_floor_c": c.rescale_noise_floor_c,
            "sigma_eval": c.sigma_eval,
            "variance_aware_q": c.variance_aware_q,
            "variance_aware_k": c.variance_aware_k,
            "variance_aware_closed_form_js": c.variance_aware_closed_form_js,
            "uncertainty_backup_weighting": c.uncertainty_backup_weighting,
            "uncertainty_backup_a": c.uncertainty_backup_a,
            "uncertainty_backup_exp": c.uncertainty_backup_exp,
            "uncertainty_backup_cap": c.uncertainty_backup_cap,
            "policy_target_min_visits": c.policy_target_min_visits,
            "colors": list(c.colors),
        }
        if hasattr(c, 'n_full_wide') and c.n_full_wide is not None:
            d["n_full_wide"] = c.n_full_wide
        if hasattr(c, 'raw_policy_above_width') and c.raw_policy_above_width is not None:
            d["raw_policy_above_width"] = c.raw_policy_above_width
        if hasattr(c, 'root_candidate_cap') and c.root_candidate_cap is not None:
            d["root_candidate_cap"] = c.root_candidate_cap
        return d

    def search(self, game: PyGame, force_full: bool = None) -> Any:
        """Run MCTS search. Drop-in for GumbelChanceMCTS.search().

        The Rust crate extracts the Game from the PyGame ONCE (one clone),
        then runs the entire search in native Rust. Only the evaluator
        callback (GPU forward pass) crosses to Python.
        """
        config_dict = self._build_config_dict()

        # The evaluator callback: called from Rust with (PyGame, legal_actions, root_color)
        # root_color is the color of the player at the root of the search — the
        # evaluator uses it to flip the value sign for opponent-turn positions.
        def eval_callback(wrapper_obj, legal_actions, root_color):
            return self.evaluator.evaluate(
                wrapper_obj,
                tuple(legal_actions),
                root_color=root_color,
                colors=tuple(self.config.colors),
            )

        def eval_many_callback(requests_list):
            if self.evaluator_many is None:
                raise RuntimeError("evaluator_many not available")
            games = [r[0] for r in requests_list]
            legal = [tuple(r[1]) for r in requests_list]
            root_colors = [r[2] for r in requests_list]
            return self.evaluator_many.evaluate_many(
                games, legal,
                root_color=root_colors[0] if root_colors else self.config.colors[0],
                colors=tuple(self.config.colors),
            )

        result = _gumbel_search(
            game,  # Pass PyGame directly — Rust extracts the Game
            eval_callback,
            config_dict,
            evaluator_many=eval_many_callback,
            force_full=force_full,
        )

        return RustSearchResult(result, self.config)


class RustSearchResult:
    """Drop-in for the Python SearchResult dataclass."""

    def __init__(self, result_dict: dict, config):
        self.selected_action = result_dict["selected_action"]
        self.improved_policy = result_dict["improved_policy"]
        self.visit_counts = result_dict["visit_counts"]
        self.q_values = result_dict["q_values"]
        self.priors = result_dict["priors"]
        self.root_value = result_dict["root_value"]
        self.used_full_search = result_dict["used_full_search"]
        self.simulations_used = result_dict["simulations_used"]
        self.afterstate_values = result_dict["afterstate_values"]
        self.config = config

    def __repr__(self):
        return (
            f"RustSearchResult(selected={self.selected_action}, "
            f"sims={self.simulations_used}, value={self.root_value:.4f}, "
            f"full={self.used_full_search})"
        )
