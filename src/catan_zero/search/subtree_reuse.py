"""MCGS cross-move subtree reuse — INTERFACE STUB (CAT-68, Phase D).

DESIGN-COMPLETE, BUILD-DEFERRED. Non-functional interface sketch so downstream
code can reference the subtree-reuse surface. Every method raises
``NotImplementedError``. Do NOT implement until the BUILD TRIGGER in
``docs/designs/CAT68_mcgs_subtree_reuse.md`` holds: CAT-67 resolved, tree-ops
>= ~15% of per-leaf cost (CAT-71), and a reusable-subtree fraction >= ~20% given
the current ``lazy_interior_chance`` setting.

Scope is Stage 1 — GUARDED TREE RE-ROOTING (not the full transposition graph).
The node/edge dataclasses (``_GNode`` / ``_GAction`` in ``gumbel_chance_mcts.py``)
are reused UNCHANGED; the only new component is this controller. See design doc §4.

The hard part is Catan's chance + hidden-info structure (design doc §3):

- The successor sits behind a chance node keyed by ``outcome_index``; the realized
  real-game outcome may not map to any materialized child -> fall back to fresh
  search (the COMMON case, not an error).
- ``lazy_interior_chance`` means interior ROLL subtrees mostly don't persist, so
  reuse is realistically limited to the root's own enumerated chance children.
- Belief-reweighted chance nodes (MOVE_ROBBER-steal, dev-card-draw,
  ``belief_*_spectra``) accumulate stats under an information set that CHANGES when
  the outcome resolves; carrying those stats over uncritically is the corruption
  risk. Stage 1 does NOT reuse across belief-changing transitions.

Reference: docs/designs/CAT68_mcgs_subtree_reuse.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from catan_zero.search.gumbel_chance_mcts import _GNode

_DESIGN_DOC = "docs/designs/CAT68_mcgs_subtree_reuse.md"
_DEFERRED = (
    "CAT-68 subtree reuse is design-complete but build-deferred. Do not implement "
    f"until the BUILD TRIGGER in {_DESIGN_DOC} holds: CAT-67 resolved, tree-ops "
    ">= ~15% of per-leaf cost (CAT-71 profiler), and reusable-subtree fraction "
    ">= ~20% under the current lazy_interior_chance setting."
)


@dataclass(frozen=True, slots=True)
class SubtreeReuseConfig:
    """Reuse controller knobs. Gated by ``GumbelChanceMCTSConfig.subtree_reuse``
    (default False -> exact current behavior, no-op when off).

    Attributes:
        reuse_across_belief_transitions: Stage 1 keeps this False — never reuse a
            subtree across a MOVE_ROBBER-steal / dev-card-draw / belief-reweighted
            transition (design doc §3.3/§3.4). Reserved for a future Stage-2 study.
        validate_state_snapshot: Require the adopted node's ``game`` snapshot to
            equal the real successor state before adopting; never trust the
            ``outcome_index`` mapping alone (design doc §3.4).
    """

    reuse_across_belief_transitions: bool = False
    validate_state_snapshot: bool = True


class SubtreeReuseController:
    """Holds the previous decision's root ``_GNode`` and, on the next decision,
    returns the guard-validated child subtree to re-root at (or None -> caller
    falls back to a fresh ``search()``). Sits beside ``GumbelChanceMCTS``, not in
    the hot path. See design doc §4.
    """

    def __init__(self, config: SubtreeReuseConfig | None = None) -> None:
        raise NotImplementedError(_DEFERRED)

    def retain(self, root: "_GNode") -> None:
        """Remember the just-searched root so its subtree can be reused next move."""
        raise NotImplementedError(_DEFERRED)

    def find_reusable_root(
        self,
        played_action: int,
        realized_outcomes: tuple[int, ...],
        next_state: Any,
    ) -> "_GNode | None":
        """Walk ``prev_root.actions[played_action].children`` for the child whose
        ``game`` snapshot matches ``next_state`` under the §3.4 guards. Return it
        (stats intact) to warm-start the next search, or None to fall back to a
        fresh root. Returning None is the expected common case under
        ``lazy_interior_chance`` + hidden info."""
        raise NotImplementedError(_DEFERRED)

    def probe_reusable_fraction(self) -> float:
        """Instrumentation-only: fraction of decisions so far whose realized
        successor mapped to a valid reusable child. Feeds BUILD-TRIGGER condition
        3 (reusable fraction >= ~20%). Cheap enough to run without full reuse."""
        raise NotImplementedError(_DEFERRED)
