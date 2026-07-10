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
         "source": "registry_pool", "filter": {"tag": "older_champion"}},
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

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from catan_zero.rl.flywheel.opponent_mix import (
    MixCategory,
    MixCheckpointRef,
    OpponentMixConfig,
    config_to_dict,
    scale_external_engine_fraction,
    validate_external_engine_fraction,
)

if TYPE_CHECKING:
    from tools.champion_registry import ChampionRegistry, PoolEntry, RolePointer

REGISTRY_SOURCES: tuple[str, ...] = ("registry_role", "registry_pool")
FROZEN_SCHEMA_VERSION = 1


def _file_md5(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _config_sha256(config: OpponentMixConfig) -> str:
    return hashlib.sha256(_canonical_json_bytes(config_to_dict(config))).hexdigest()


def _verify_checkpoint_file(
    checkpoint: MixCheckpointRef, *, category_name: str
) -> MixCheckpointRef:
    path = Path(checkpoint.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"mix category {category_name!r} checkpoint missing or not a regular file: {path}"
        )
    actual_md5 = _file_md5(path)
    expected_md5 = checkpoint.md5.strip().lower()
    if expected_md5 and expected_md5 != actual_md5:
        raise ValueError(
            f"mix category {category_name!r} checkpoint md5 mismatch for {path}: "
            f"manifest/registry says {expected_md5}, actual bytes are {actual_md5}"
        )
    return MixCheckpointRef(path=str(path), version=checkpoint.version, md5=actual_md5)


def validate_resolved_opponent_mix(
    config: OpponentMixConfig,
    *,
    producer_checkpoint: str | os.PathLike[str] | None = None,
) -> OpponentMixConfig:
    """Verify and byte-bind a fully resolved mix before generation starts.

    Every checkpoint-list entry is required to exist and is re-hashed even when
    it came from ``ChampionRegistry``. Missing md5 values in explicit lists are
    filled with the computed digest; supplied/registry digests must match. The
    same bytes cannot occupy two opponent slots, and the producer's bytes cannot
    reappear as an opponent under an alias path.

    The returned config contains absolute paths and computed md5 values, so it
    can be pickled into workers or written as an immutable run manifest without
    consulting the mutable registry again.
    """
    effective_self = [
        category
        for category in config.effective_categories
        if category.source == "self"
    ]
    if len(effective_self) > 1:
        names = ", ".join(repr(category.name) for category in effective_self)
        raise ValueError(
            f"opponent mix has multiple effective self categories ({names}); "
            "use one producer self-play category so its probability is unambiguous"
        )

    producer_md5: str | None = None
    producer_path: Path | None = None
    if producer_checkpoint is not None:
        producer_path = Path(producer_checkpoint).expanduser().resolve()
        if not producer_path.is_file():
            raise FileNotFoundError(
                f"producer checkpoint missing or not a regular file: {producer_path}"
            )
        producer_md5 = _file_md5(producer_path)

    seen_by_md5: dict[str, tuple[str, str]] = {}
    verified_categories: list[MixCategory] = []
    for category in config.categories:
        if category.source == "self" and (category.checkpoints or category.engine):
            raise ValueError(
                f"self category {category.name!r} must not carry checkpoints or an engine"
            )
        if category.source == "external_engine" and category.checkpoints:
            raise ValueError(
                f"external-engine category {category.name!r} must not carry checkpoint entries"
            )
        if category.source != "checkpoint_list":
            verified_categories.append(category)
            continue
        if category.engine:
            raise ValueError(
                f"checkpoint-list category {category.name!r} must not carry an engine name"
            )

        verified_checkpoints: list[MixCheckpointRef] = []
        for checkpoint in category.checkpoints:
            verified = _verify_checkpoint_file(checkpoint, category_name=category.name)
            if producer_md5 is not None and verified.md5 == producer_md5:
                raise ValueError(
                    f"mix category {category.name!r} checkpoint {verified.path} has the same "
                    f"bytes (md5={verified.md5}) as producer checkpoint {producer_path}; "
                    "this is duplicate self-play disguised as an opponent reference"
                )
            previous = seen_by_md5.get(verified.md5)
            if previous is not None:
                previous_category, previous_path = previous
                raise ValueError(
                    "duplicate checkpoint bytes in opponent mix: "
                    f"category {previous_category!r} path {previous_path} and category "
                    f"{category.name!r} path {verified.path} both have md5={verified.md5}; "
                    "each checkpoint must occupy exactly one mix slot"
                )
            seen_by_md5[verified.md5] = (category.name, verified.path)
            verified_checkpoints.append(verified)
        verified_categories.append(
            dataclasses.replace(category, checkpoints=tuple(verified_checkpoints))
        )

    return OpponentMixConfig(categories=tuple(verified_categories))


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


def resolve_opponent_mix_manifest(
    path: str | Path,
    *,
    producer_checkpoint: str | os.PathLike[str] | None = None,
    registry_path_override: str | os.PathLike[str] | None = None,
) -> OpponentMixConfig:
    """Parse an opponent-mix manifest that may reference the CAT-9 registry,
    resolving every "registry_role"/"registry_pool" category into a concrete
    "checkpoint_list" category, and return a plain, registry-free
    ``OpponentMixConfig`` ready for ``choose_mix_opponent``/``MixRuntime``.

    Safe to call once in the main process before workers spawn (same
    fail-fast-early pattern as ``read_opponent_pool_manifest``/
    ``opponent_mix.read_opponent_mix_manifest``) -- ``ChampionRegistry.load``
    is pure JSON, no torch/device involved.
    """
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"opponent-mix manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_categories = list(data.get("categories", []))
    if not raw_categories:
        raise ValueError(f"opponent-mix manifest {manifest_path} has no 'categories' entries")

    registry: "ChampionRegistry | None" = None
    registry_path = registry_path_override or data.get("registry")
    needs_registry = any(str(entry.get("source")) in REGISTRY_SOURCES for entry in raw_categories)
    if needs_registry:
        if not registry_path:
            raise ValueError(
                f"opponent-mix manifest {manifest_path} has a registry_role/registry_pool category but no "
                'top-level "registry" path'
            )
        resolved_registry_path = Path(registry_path).expanduser().resolve()
        if not resolved_registry_path.is_file():
            raise FileNotFoundError(
                f"opponent-mix registry not found: {resolved_registry_path}"
            )
        registry = _import_champion_registry().ChampionRegistry.load(resolved_registry_path)

    categories = tuple(_resolve_category(entry, registry=registry) for entry in raw_categories)
    config = validate_resolved_opponent_mix(
        OpponentMixConfig(categories=categories),
        producer_checkpoint=producer_checkpoint,
    )

    frozen = data.get("_frozen")
    if frozen is not None:
        if int(frozen.get("schema_version", -1)) != FROZEN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported frozen opponent-mix schema version "
                f"{frozen.get('schema_version')!r}; expected {FROZEN_SCHEMA_VERSION}"
            )
        expected_digest = str(frozen.get("resolved_config_sha256", ""))
        actual_digest = _config_sha256(config)
        if not expected_digest or expected_digest != actual_digest:
            raise ValueError(
                "frozen opponent-mix digest mismatch: resolved config bytes do not match "
                f"the frozen digest (expected {expected_digest or '<missing>'}, actual {actual_digest})"
            )
        frozen_producer = dict(frozen.get("producer_checkpoint", {}))
        if producer_checkpoint is not None:
            actual_producer_md5 = _file_md5(Path(producer_checkpoint).expanduser().resolve())
            frozen_producer_md5 = str(frozen_producer.get("md5", ""))
            if not frozen_producer_md5 or frozen_producer_md5 != actual_producer_md5:
                raise ValueError(
                    "frozen opponent mix was bound to a different producer checkpoint: "
                    f"frozen md5={frozen_producer_md5 or '<missing>'}, "
                    f"current producer md5={actual_producer_md5}"
                )
    return config


def freeze_opponent_mix_manifest(
    source_manifest: str | Path,
    output_path: str | Path,
    *,
    producer_checkpoint: str | os.PathLike[str],
    registry_path_override: str | os.PathLike[str] | None = None,
    external_fraction: float | None = None,
) -> Path:
    """Resolve, validate, and write a content-bound run manifest exactly once.

    Registry roles/pools are expanded to concrete absolute checkpoint paths;
    every checkpoint and the producer are re-hashed; duplicate/self aliases are
    rejected. ``external_fraction`` optionally applies the existing exploiter
    rescaling algorithm before freezing. The output is created with O_EXCL and
    chmod 0444, so this command never overwrites or silently mutates a run's
    provenance file.
    """
    source_path = Path(source_manifest).expanduser().resolve()
    producer_path = Path(producer_checkpoint).expanduser().resolve()
    config = resolve_opponent_mix_manifest(
        source_path,
        producer_checkpoint=producer_path,
        registry_path_override=registry_path_override,
    )
    if external_fraction is not None:
        config = scale_external_engine_fraction(config, float(external_fraction))
    validate_external_engine_fraction(config)

    config_dict = config_to_dict(config)
    payload = {
        "_frozen": {
            "schema_version": FROZEN_SCHEMA_VERSION,
            "source_manifest": str(source_path),
            "source_manifest_sha256": _file_sha256(source_path),
            "producer_checkpoint": {
                "path": str(producer_path),
                "md5": _file_md5(producer_path),
            },
            "resolved_config_sha256": _config_sha256(config),
        },
        **config_dict,
    }

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite frozen opponent-mix manifest: {output}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(output, 0o444)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


def _cmd_show(args: Any) -> None:
    config = resolve_opponent_mix_manifest(
        args.manifest,
        producer_checkpoint=args.producer_checkpoint,
        registry_path_override=args.registry,
    )
    if args.external_fraction is not None:
        config = scale_external_engine_fraction(config, args.external_fraction)
    validate_external_engine_fraction(config)
    print(json.dumps(config_to_dict(config), indent=2, sort_keys=True))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Resolve and byte-validate a registry-backed opponent mix. By default print the "
            "resolved JSON; --freeze-output writes a read-only, content-bound run manifest."
        )
    )
    parser.add_argument("--manifest", required=True, help="Path to the opponent-mix manifest JSON.")
    parser.add_argument(
        "--registry",
        help="Override the manifest's top-level registry path (useful with the checked-in R9 template).",
    )
    parser.add_argument(
        "--producer-checkpoint",
        help=(
            "Current producer checkpoint. Required for --freeze-output and recommended for validation; "
            "rejects opponent entries whose bytes duplicate the producer."
        ),
    )
    parser.add_argument(
        "--external-fraction",
        type=float,
        help=(
            "Apply the existing external-engine rescaling before printing/freezing, e.g. 0.03 for "
            "the production 3%% catanatron lane. The existing 5%% safety cap still applies."
        ),
    )
    parser.add_argument(
        "--freeze-output",
        help=(
            "Create this resolved run manifest exactly once (absolute paths, verified md5s, config "
            "digest, mode 0444). Existing files are never overwritten."
        ),
    )
    args = parser.parse_args()
    if args.freeze_output:
        if not args.producer_checkpoint:
            parser.error("--freeze-output requires --producer-checkpoint")
        output = freeze_opponent_mix_manifest(
            args.manifest,
            args.freeze_output,
            producer_checkpoint=args.producer_checkpoint,
            registry_path_override=args.registry,
            external_fraction=args.external_fraction,
        )
        print(output)
    else:
        _cmd_show(args)


if __name__ == "__main__":
    main()
