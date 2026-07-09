"""Durable, order-independent serialization for config dataclasses (task #74).

WHY THIS EXISTS: every persisted config in this repo is a frozen+slots
dataclass, and pickle serializes those POSITIONALLY (`_dataclass_getstate`
returns a bare list of values in field order; `_dataclass_setstate` zips it
against the CURRENT field list). That has two failure modes, both observed or
demonstrated on 2026-07-05:

  1. CRASH (fixed by a413df8): an old, shorter pickle leaves the newest slots
     UNSET; the next re-pickle raises AttributeError (v3a lost its first epoch
     to exactly this at checkpoint-save time).
  2. SILENT SHIFT (the remaining class this module kills): a checkpoint
     pickled under a field order that differs MID-LIST from the loader's
     (e.g. a feature branch that inserted a field where master later put a
     different one) zips every later value into the wrong slot. `hasattr`
     succeeds on all of them, so no reconstruction based on present-ness can
     detect it -- the architecture silently changes.

THE FIX: persist configs as a NAME-KEYED dict plus a schema marker, never as
the dataclass object. Reconstruction is by field NAME: order becomes
irrelevant, missing fields take the current dataclass defaults, and unknown
fields are dropped with a warning (forward compatibility). Loaders accept
BOTH forms -- the legacy pickled dataclass (all existing checkpoints) and the
new dict -- through `config_from_dict`.

Append-only field discipline remains good hygiene for LEGACY checkpoints
(they stay positional until re-saved), but new saves are immune.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from types import SimpleNamespace
from typing import Any, Callable, TypeVar

CONFIG_CLASS_KEY = "__config_dataclass__"
CONFIG_SCHEMA_KEY = "__config_schema__"
CONFIG_FIELDS_KEY = "fields"
CONFIG_DICT_SCHEMA_VERSION = 1

_T = TypeVar("_T")


def _default_warn(message: str) -> None:
    print(json.dumps({"progress": "config_serialization_warning", "message": message},
                     sort_keys=True), file=sys.stderr, flush=True)


def is_config_dict(payload: Any) -> bool:
    """True when `payload` is the name-keyed dict form written by config_to_dict."""
    return (
        isinstance(payload, dict)
        and CONFIG_CLASS_KEY in payload
        and isinstance(payload.get(CONFIG_FIELDS_KEY), dict)
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize a config dataclass instance to the durable name-keyed form.

    Field VALUES are stored as-is (they ride inside torch.save's pickle and may
    be numpy scalars etc.); only the container stops being a dataclass. Slots
    left unset by a stale positional unpickle are simply omitted -- they read
    back as the current defaults, which is the same semantics load-time
    reconstruction (a413df8) already gives them.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"config_to_dict expects a dataclass instance, got {type(config).__name__}")
    fields = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(config)
        if hasattr(config, f.name)
    }
    return {
        CONFIG_CLASS_KEY: type(config).__name__,
        CONFIG_SCHEMA_KEY: CONFIG_DICT_SCHEMA_VERSION,
        CONFIG_FIELDS_KEY: fields,
    }


def config_from_dict(
    cls: type[_T],
    payload: Any,
    *,
    warn: Callable[[str], None] | None = None,
) -> _T:
    """Reconstruct `cls` from either serialized form, by field NAME.

    - `payload` already an instance of `cls` (legacy pickled dataclass, possibly
      stale with unset slots): rebuild field-by-name so unset slots become
      current defaults and any later re-pickle cannot crash. Positional SHIFT
      in a legacy pickle is NOT detectable here (values land in plausible
      slots) -- that is exactly why new saves use the dict form.
    - `payload` in the name-keyed dict form: reconstruct by name. Missing
      fields take the current dataclass defaults; unknown fields are dropped
      with a warning; a class-name mismatch warns but proceeds (renames stay
      loadable).
    """
    emit = warn or _default_warn
    if isinstance(payload, cls):
        present = {
            f.name: getattr(payload, f.name)
            for f in dataclasses.fields(cls)
            if hasattr(payload, f.name)
        }
        return cls(**present)
    if is_config_dict(payload):
        recorded_class = str(payload.get(CONFIG_CLASS_KEY, ""))
        if recorded_class and recorded_class != cls.__name__:
            emit(f"config class name mismatch: checkpoint={recorded_class!r} expected={cls.__name__!r}; reconstructing by field name")
        schema = payload.get(CONFIG_SCHEMA_KEY)
        if schema is not None and int(schema) > CONFIG_DICT_SCHEMA_VERSION:
            emit(f"config dict schema {schema} is newer than supported {CONFIG_DICT_SCHEMA_VERSION}; reconstructing best-effort by field name")
        stored = dict(payload[CONFIG_FIELDS_KEY])
        known = {f.name for f in dataclasses.fields(cls)}
        extra = sorted(set(stored) - known)
        if extra:
            emit(f"dropping unknown {cls.__name__} fields from checkpoint: {extra}")
        kept = {name: value for name, value in stored.items() if name in known}
        return cls(**kept)
    raise TypeError(
        f"cannot reconstruct {cls.__name__} from {type(payload).__name__}: "
        "expected the dataclass instance (legacy checkpoints) or the "
        "name-keyed config dict (new checkpoints)"
    )


def config_attr_view(payload: Any) -> Any:
    """Attribute-access adapter for code that probes configs via getattr.

    Returns `payload` unchanged unless it is the name-keyed dict form, in which
    case a SimpleNamespace over its fields is returned so existing
    `getattr(config, name, default)` call sites work on both formats without
    modification.
    """
    if is_config_dict(payload):
        return SimpleNamespace(**payload[CONFIG_FIELDS_KEY])
    return payload
