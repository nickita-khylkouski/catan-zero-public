"""Shared row-level provenance for archived-state restart trajectories."""

from __future__ import annotations


RESTART_PROVENANCE_KEYS: tuple[str, ...] = (
    "restart_provenance_present",
    "start_mode",
    "start_bucket",
    "archived_game_seed",
    "archived_decision_index",
    "restart_select_seed",
)

RESTART_BOOL_KEYS: tuple[str, ...] = ("restart_provenance_present",)
RESTART_STRING_KEYS: tuple[str, ...] = ("start_mode", "start_bucket")
RESTART_INT64_KEYS: tuple[str, ...] = (
    "archived_game_seed",
    "archived_decision_index",
    "restart_select_seed",
)
