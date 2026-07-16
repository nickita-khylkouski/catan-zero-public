"""Fail-closed catalog for supported production pipeline recipes.

The compact launchers deliberately do not expose science knobs.  Approved
recipes live in :mod:`configs/production_recipes.json`, so commissioning a new
recipe is a data review rather than a launcher-code change.  Historical and
R&D configs remain importable by their specialized executors but cannot enter
the production launchers merely because they happen to share a schema.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

CATALOG_SCHEMA_VERSION = 2
CATALOG_RELATIVE_PATH = Path("configs/production_recipes.json")
SUPPORTED_ENTRYPOINTS = frozenset({"generate", "evaluate", "train"})


class ProductionRecipeError(ValueError):
    """The requested recipe is not an authenticated production recipe."""


def canonical_json_sha256(payload: object) -> str:
    """Hash JSON semantics rather than whitespace or key order."""

    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise ProductionRecipeError(
            f"recipe payload is not canonical JSON: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_catalog(root: Path) -> Mapping[str, Any]:
    path = root / CATALOG_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionRecipeError(
            f"cannot load production recipe catalog {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ProductionRecipeError("production recipe catalog must be a JSON object")
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ProductionRecipeError(
            "production recipe catalog schema mismatch: "
            f"expected={CATALOG_SCHEMA_VERSION} "
            f"actual={payload.get('schema_version')!r}"
        )
    recipes = payload.get("recipes")
    if not isinstance(recipes, dict):
        raise ProductionRecipeError("production recipe catalog requires recipes object")
    unknown = sorted(set(recipes) - SUPPORTED_ENTRYPOINTS)
    missing = sorted(SUPPORTED_ENTRYPOINTS - set(recipes))
    if unknown or missing:
        raise ProductionRecipeError(
            f"production recipe catalog entrypoints drifted: missing={missing} "
            f"unknown={unknown}"
        )
    return recipes


def _catalog_entry(
    *, root: Path, entrypoint: str, requested_path: Path
) -> Mapping[str, Any]:
    recipes = _load_catalog(root)
    entries = recipes[entrypoint]
    if not isinstance(entries, list) or not entries:
        raise ProductionRecipeError(
            f"production recipe catalog has no approved {entrypoint} recipes"
        )

    source = requested_path.expanduser()
    try:
        if source.is_symlink():
            raise ProductionRecipeError(
                f"{entrypoint} recipe must not be a symlink: {requested_path}"
            )
        requested = source.resolve(strict=True)
        relative = requested.relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise ProductionRecipeError(
            f"{entrypoint} recipe must be a checked-in regular file: {requested_path}"
        ) from error
    if not requested.is_file() or requested.is_symlink():
        raise ProductionRecipeError(
            f"{entrypoint} recipe must be a checked-in regular file: {requested_path}"
        )

    matches: list[Mapping[str, Any]] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, value in enumerate(entries):
        if not isinstance(value, dict):
            raise ProductionRecipeError(
                f"production recipe catalog {entrypoint}[{index}] must be an object"
            )
        expected_keys = {"name", "path", "canonical_sha256"}
        if entrypoint == "generate":
            expected_keys.add("guard")
        if set(value) != expected_keys:
            raise ProductionRecipeError(
                f"production recipe catalog {entrypoint}[{index}] fields drifted"
            )
        name = value.get("name")
        path = value.get("path")
        digest = value.get("canonical_sha256")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ProductionRecipeError(
                f"production recipe catalog has invalid/duplicate {entrypoint} name"
            )
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in seen_paths
        ):
            raise ProductionRecipeError(
                f"production recipe catalog has invalid/duplicate {entrypoint} path"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ProductionRecipeError(
                f"production recipe catalog has invalid {entrypoint} SHA-256"
            )
        seen_names.add(name)
        seen_paths.add(path)
        if entrypoint == "generate":
            guard = value.get("guard")
            if not isinstance(guard, dict) or set(guard) != {
                "path",
                "canonical_sha256",
            }:
                raise ProductionRecipeError(
                    "production generation recipe requires an exact guard identity"
                )
            guard_path = guard.get("path")
            guard_digest = guard.get("canonical_sha256")
            if (
                not isinstance(guard_path, str)
                or not guard_path
                or Path(guard_path).is_absolute()
                or ".." in Path(guard_path).parts
                or not isinstance(guard_digest, str)
                or len(guard_digest) != 64
                or any(
                    character not in "0123456789abcdef" for character in guard_digest
                )
            ):
                raise ProductionRecipeError(
                    "production generation recipe has an invalid guard identity"
                )
        if path == relative:
            matches.append(value)
    if len(matches) != 1:
        approved = sorted(seen_paths)
        raise ProductionRecipeError(
            f"{relative!r} is not an approved production {entrypoint} recipe; "
            f"approved={approved}"
        )
    return matches[0]


def require_production_recipe(
    *, entrypoint: str, path: str | Path, payload: object
) -> str:
    """Authenticate an approved recipe and return its stable catalog name."""

    if entrypoint not in SUPPORTED_ENTRYPOINTS:
        raise ProductionRecipeError(f"unsupported production entrypoint {entrypoint!r}")
    root = _repository_root()
    entry = _catalog_entry(root=root, entrypoint=entrypoint, requested_path=Path(path))
    expected = str(entry["canonical_sha256"])
    actual = canonical_json_sha256(payload)
    if actual != expected:
        raise ProductionRecipeError(
            f"approved production {entrypoint} recipe bytes drifted: "
            f"name={entry['name']!r} expected_sha256={expected} "
            f"actual_sha256={actual}"
        )
    return str(entry["name"])


def production_recipes(entrypoint: str) -> tuple[dict[str, str], ...]:
    """Return every authenticated recipe for one compact entrypoint.

    Operator layers use this instead of copying catalog paths or digests into a
    second registry. Every returned entry has already replayed the same
    checked-in path and semantic-hash checks as the compact launcher.
    """

    if entrypoint not in SUPPORTED_ENTRYPOINTS:
        raise ProductionRecipeError(f"unsupported production entrypoint {entrypoint!r}")
    root = _repository_root()
    entries = _load_catalog(root)[entrypoint]
    if not isinstance(entries, list):
        raise ProductionRecipeError(
            f"production recipe catalog {entrypoint!r} must be a list"
        )
    authenticated: list[dict[str, str]] = []
    for value in entries:
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ProductionRecipeError(
                f"production recipe catalog has malformed {entrypoint} entry"
            )
        path = root / value["path"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProductionRecipeError(
                f"cannot load cataloged {entrypoint} recipe {path}: {error}"
            ) from error
        name = require_production_recipe(
            entrypoint=entrypoint, path=path, payload=payload
        )
        authenticated_entry = {
            "name": name,
            "path": str(path.resolve()),
            "canonical_sha256": canonical_json_sha256(payload),
        }
        if entrypoint == "generate":
            guard = value["guard"]
            assert isinstance(guard, dict)
            guard_path = root / str(guard["path"])
            try:
                if guard_path.is_symlink():
                    raise ProductionRecipeError(
                        f"production generation guard must not be a symlink: {guard_path}"
                    )
                resolved_guard = guard_path.resolve(strict=True)
                resolved_guard.relative_to(root)
                guard_payload = json.loads(resolved_guard.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                raise ProductionRecipeError(
                    f"cannot authenticate production generation guard {guard_path}: {error}"
                ) from error
            if not resolved_guard.is_file() or resolved_guard.is_symlink():
                raise ProductionRecipeError(
                    f"production generation guard must be a regular file: {resolved_guard}"
                )
            actual_guard_sha256 = canonical_json_sha256(guard_payload)
            expected_guard_sha256 = str(guard["canonical_sha256"])
            if actual_guard_sha256 != expected_guard_sha256:
                raise ProductionRecipeError(
                    "approved production generation guard bytes drifted: "
                    f"name={name!r} expected_sha256={expected_guard_sha256} "
                    f"actual_sha256={actual_guard_sha256}"
                )
            authenticated_entry.update(
                guard=str(resolved_guard),
                guard_sha256=actual_guard_sha256,
            )
        authenticated.append(authenticated_entry)
    return tuple(authenticated)
