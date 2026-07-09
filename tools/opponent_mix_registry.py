#!/usr/bin/env python3
"""CAT-9 <-> CAT-54 bridge: resolve an opponent-mix manifest whose categories may
REFERENCE the CAT-9 champion registry (``tools/champion_registry.py``'s roles and
append-only ``opponent_pool[]``) into a plain, registry-free
``catan_zero.rl.flywheel.opponent_mix.OpponentMixConfig`` that
``run_worker_games``/``choose_mix_opponent`` can sample from directly.

This module is deliberately the ONLY place that imports both
``tools.champion_registry`` and ``catan_zero.rl.flywheel.opponent_mix`` --
``opponent_mix.py`` itself stays registry-free (pure stdlib, like
``opponent_pool.py``), matching the existing "``src/catan_zero/rl`` does not
depend on ``tools/``" boundary documented in ``gumbel_self_play.py``.

Manifest schema (superset of ``opponent_mix.read_opponent_mix_manifest``'s):

    {
      "registry": "<path to a CAT-9 ChampionRegistry JSON>",   // required iff
                                                                // any category
                                                                // below uses a
                                                                // registry_* source
      "categories": [
        {"name": "producer_self_play", "weight": 75, "source": "self"},
        {"name": "previous_public_champion", "weight": 10,
         "source": "registry_role", "role": "public_champion"},
        {"name": "older_champion", "weight": 5,
         "source": "registry_pool", "filter": {"status": "active"}},
        {"name": "hard_experimental", "weight": 5,
         "source": "registry_pool", "filter": {"tag": "hard_negative"}},
        {"name": "catanatron_value", "weight": 5, "source": "external_engine",
         "engine": "catanatron_value", "pending": true}
      ]
    }

"registry_role" resolves ONE checkpoint from ``registry.get_role(role)``.
"registry_pool" resolves ANY NUMBER of checkpoints from
``registry.opponent_pool()``, filtered by an optional ``filter`` dict whose
keys are either ``"status"`` (matched against ``PoolEntry.status``) or any
other key (matched against ``PoolEntry.provenance[key]``, e.g. the ``"tag"``
a caller stamped when appending a hard-negative/exploiter checkpoint to the
pool -- CAT-9 doesn't mandate a provenance vocabulary, so this reads whatever
key the append-pool caller chose). An empty ``filter`` (or omitted) matches
every pool entry.

A ``registry_role``/``registry_pool`` category that resolves to ZERO
checkpoints is an error UNLESS the manifest also marks it ``"pending": true``
(same convention as ``opponent_mix``'s own "external_engine" categories) --
early runs (small/empty opponent_pool, no public_champion set yet) should say
so loudly rather than silently vanish from the mix. "self"/"checkpoint_list"/
"external_engine" categories pass through unchanged (no registry needed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from catan_zero.rl.flywheel.opponent_mix import MixCategory, MixCheckpointRef, OpponentMixConfig

if TYPE_CHECKING:
    from tools.champion_registry import ChampionRegistry, PoolEntry, RolePointer

REGISTRY_SOURCES: tuple[str, ...] = ("registry_role", "registry_pool")


def _import_champion_registry() -> Any:
    """Import ``champion_registry`` LAZILY (only called when a manifest
    actually uses a "registry_role"/"registry_pool" category) -- NOT at this
    module's top level. `champion_registry.py` itself does a package-qualified
    `from tools.sprt_gate import score_to_elo`, which needs the repo root on
    `sys.path`; `tools/generate_gumbel_selfplay_data.py` (this bridge's only
    caller today) instead assumes only `tools/` itself is on `sys.path`. An
    eager top-level import here would therefore make EVERY invocation of the
    generator -- even ones that never pass --opponent-mix-manifest at all --
    depend on the repo root being importable, silently breaking the "no
    flag = today's exact behavior" default path. Deferring the import to only
    the manifests that need it keeps that regression impossible.
    """
    try:
        # Package-qualified: works when the repo root is on sys.path (`python
        # -m tools.opponent_mix_registry`, `python -m pytest` from the repo
        # root, or any caller that already imported `tools.champion_registry`
        # this way).
        from tools import champion_registry
    except ImportError:
        # Bare sibling import: works when only `tools/` itself is on
        # sys.path -- but ALSO requires the repo root for champion_registry's
        # own `from tools.sprt_gate import score_to_elo`, so this fallback
        # only helps when the caller has separately arranged for `tools/` to
        # resolve as a top-level package elsewhere on the path.
        import champion_registry  # type: ignore[no-redef]
    return champion_registry


def _pool_entry_matches(entry: "PoolEntry", *, filter_spec: dict[str, Any]) -> bool:
    for key, value in filter_spec.items():
        if key == "status":
            if entry.status != value:
                return False
        elif entry.provenance.get(key) != value:
            return False
    return True


def _resolve_registry_role(
    registry: "ChampionRegistry", *, role: str, category_name: str, pending: bool
) -> tuple[MixCheckpointRef, ...]:
    pointer: "RolePointer | None" = registry.get_role(role)
    if pointer is None:
        if pending:
            return ()
        raise ValueError(
            f"mix category {category_name!r} references registry role {role!r}, but that role "
            "has no pointer set yet in the registry -- mark this category \"pending\": true until "
            "it does, or set the role first (tools/champion_registry.py set-role)"
        )
    return (MixCheckpointRef(path=pointer.checkpoint_path, version=pointer.version or -1, md5=pointer.md5),)


def _resolve_registry_pool(
    registry: "ChampionRegistry", *, filter_spec: dict[str, Any], category_name: str, pending: bool
) -> tuple[MixCheckpointRef, ...]:
    matches = tuple(
        MixCheckpointRef(path=entry.checkpoint_path, version=entry.version or -1, md5=entry.md5)
        for entry in registry.opponent_pool()
        if _pool_entry_matches(entry, filter_spec=filter_spec)
    )
    if not matches and not pending:
        raise ValueError(
            f"mix category {category_name!r} (source=registry_pool, filter={filter_spec!r}) "
            "matched zero opponent_pool entries -- mark this category \"pending\": true until the "
            "registry has matching entries, or broaden/fix the filter"
        )
    return matches


def _resolve_category(entry: dict[str, Any], *, registry: "ChampionRegistry | None") -> MixCategory:
    source = str(entry["source"])
    name = str(entry["name"])
    weight = float(entry["weight"])
    pending = bool(entry.get("pending", False))

    if source not in REGISTRY_SOURCES:
        # Plain source ("self"/"checkpoint_list"/"external_engine"): identical
        # shape to opponent_mix.read_opponent_mix_manifest's own parsing, kept
        # in sync deliberately (not delegated to it, to avoid reaching into
        # that module's private `_category_from_dict`).
        checkpoints = tuple(
            MixCheckpointRef(
                path=str(ck["path"]),
                version=int(ck.get("version", -1)) if ck.get("version") is not None else -1,
                md5=str(ck.get("md5", "")),
            )
            for ck in entry.get("checkpoints", [])
        )
        return MixCategory(
            name=name,
            weight=weight,
            source=source,
            checkpoints=checkpoints,
            engine=entry.get("engine"),
            pending=pending,
        )

    if registry is None:
        raise ValueError(
            f"mix category {name!r} has source={source!r} but the manifest has no top-level "
            '"registry" path -- add one (a CAT-9 ChampionRegistry JSON) to resolve registry-backed categories'
        )
    if source == "registry_role":
        checkpoints = _resolve_registry_role(
            registry, role=str(entry["role"]), category_name=name, pending=pending
        )
    else:  # "registry_pool"
        checkpoints = _resolve_registry_pool(
            registry, filter_spec=dict(entry.get("filter", {})), category_name=name, pending=pending
        )
    # An empty-but-pending registry category has no checkpoints to sample --
    # downgrade its effective source to "checkpoint_list" (MixCategory already
    # tolerates an empty checkpoints list when pending=True) rather than
    # inventing a fourth source value the pure opponent_mix module would need
    # to understand.
    return MixCategory(
        name=name,
        weight=weight,
        source="checkpoint_list",
        checkpoints=checkpoints,
        pending=pending or not checkpoints,
    )


def resolve_opponent_mix_manifest(path: str | Path) -> OpponentMixConfig:
    """Parse an opponent-mix manifest that may reference the CAT-9 registry,
    resolving every "registry_role"/"registry_pool" category into a concrete
    "checkpoint_list" category, and return a plain, registry-free
    ``OpponentMixConfig`` ready for ``choose_mix_opponent``/``MixRuntime``.

    Safe to call once in the main process before workers spawn (same
    fail-fast-early pattern as ``read_opponent_pool_manifest``/
    ``opponent_mix.read_opponent_mix_manifest``) -- ``ChampionRegistry.load``
    is pure JSON, no torch/device involved.
    """
    data = json.loads(Path(path).read_text())
    raw_categories = list(data.get("categories", []))
    if not raw_categories:
        raise ValueError(f"opponent-mix manifest {path} has no 'categories' entries")

    registry: "ChampionRegistry | None" = None
    registry_path = data.get("registry")
    needs_registry = any(str(entry.get("source")) in REGISTRY_SOURCES for entry in raw_categories)
    if needs_registry:
        if not registry_path:
            raise ValueError(
                f"opponent-mix manifest {path} has a registry_role/registry_pool category but no "
                'top-level "registry" path'
            )
        registry = _import_champion_registry().ChampionRegistry.load(registry_path)

    categories = tuple(_resolve_category(entry, registry=registry) for entry in raw_categories)
    return OpponentMixConfig(categories=categories)


def _cmd_show(args: Any) -> None:
    from catan_zero.rl.flywheel.opponent_mix import config_to_dict

    config = resolve_opponent_mix_manifest(args.manifest)
    print(json.dumps(config_to_dict(config), indent=2, sort_keys=True))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Resolve a (possibly registry-referencing) opponent-mix manifest and print it."
    )
    parser.add_argument("--manifest", required=True, help="Path to the opponent-mix manifest JSON.")
    args = parser.parse_args()
    _cmd_show(args)


if __name__ == "__main__":
    main()
