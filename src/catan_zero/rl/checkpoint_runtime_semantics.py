"""Fail-closed binding between entity checkpoints and serving semantics.

Checkpoint tensor shapes and config schemas are not enough to identify a model
function.  A serving checkout can accept every tensor while implementing a
different forward pass.  This module binds the executable subset of
``entity_token_policy.py`` without coupling checkpoints to comments, saving
code, or other non-forward edits in that large module.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import tokenize
from typing import Mapping


ENTITY_GRAPH_FORWARD_SEMANTICS_KEY = "entity_graph_forward_semantics"
ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA = "entity-graph-forward-semantics-v1"
ENTITY_GRAPH_POLICY_SCHEMA = "entity_graph_policy_v1"

# These are the source definitions that construct or execute the entity-graph
# neural function.  In particular, save/load/provenance code is excluded so a
# metadata-only release does not invalidate otherwise identical checkpoints.
_TOP_LEVEL_FORWARD_FUNCTIONS = frozenset(
    {
        "_validate_public_award_feature_contract",
        "_apply_public_award_feature_contract",
        "_entity_token_start_offsets",
        "_token_encoder",
        "_assert_entity_batch_shapes",
    }
)
_POLICY_FORWARD_METHODS = frozenset({"__init__", "forward_legal_np"})
_SELECTED_SYMBOLS = (
    "_validate_public_award_feature_contract",
    "_apply_public_award_feature_contract",
    "_entity_token_start_offsets",
    "EntityGraphNet",
    "_token_encoder",
    "EntityGraphPolicy.__init__",
    "EntityGraphPolicy.forward_legal_np",
    "_assert_entity_batch_shapes",
)

# Before this binding was added, the selected canonical learner saved its
# checkout binding inside value_training.  This exact whole-file SHA is the
# reviewed b4e261 runtime that produced those checkpoints.  Map it to the
# semantic AST fingerprint below; do not accept arbitrary legacy source SHAs.
_REVIEWED_TRAINING_SOURCE_SEMANTICS = {
    "sha256:b4e2618bc36296470f13ce3dee228b34fd7d117c0211380c46393450793ce975": (
        "sha256:460f78322abb3af4ba5255593b9ef3a4db93c53a114fdef15a9a8f1dae828f7e"
    )
}

# The canonical f7 incumbent predates checkout-runtime metadata.  Its exact
# bytes are explicitly reviewed against the compatibility bridges in the
# current loader.  No other metadata-free checkpoint inherits this exception.
_REVIEWED_LEGACY_CHECKPOINTS = {
    "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4": (
        "legacy-f7-current-compat-v1"
    )
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _semantic_nodes(source: str) -> list[ast.AST]:
    tree = ast.parse(source)
    selected: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _TOP_LEVEL_FORWARD_FUNCTIONS:
                selected.append(node)
            continue
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == "EntityGraphNet":
            selected.append(node)
            continue
        if node.name == "EntityGraphPolicy":
            selected.extend(
                method
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name in _POLICY_FORWARD_METHODS
            )
    if len(selected) != len(_SELECTED_SYMBOLS):
        raise RuntimeError(
            "cannot identify the complete entity-graph forward semantic surface"
        )
    return selected


@lru_cache(maxsize=4)
def current_entity_graph_forward_semantics(
    policy_source: str | Path,
) -> dict[str, object]:
    """Return the canonical, comment-insensitive forward semantic identity."""

    source_path = Path(policy_source).resolve(strict=True)
    source = source_path.read_text(encoding="utf-8")
    nodes = _semantic_nodes(source)
    ignored_tokens = {
        "ENCODING",
        "COMMENT",
        "NL",
        "NEWLINE",
        "INDENT",
        "DEDENT",
        "ENDMARKER",
    }
    # Token text is stable across supported Python parsers, unlike ast.dump's
    # field list (for example Python 3.12 adds type_params).  Ignore comments
    # and layout while retaining every executable token and literal.
    semantic_tokens = []
    for node in nodes:
        segment = ast.get_source_segment(source, node)
        if segment is None:
            raise RuntimeError("cannot recover entity-graph forward source segment")
        semantic_tokens.append(
            [
                (tokenize.tok_name[token.type], token.string)
                for token in tokenize.generate_tokens(io.StringIO(segment).readline)
                if tokenize.tok_name[token.type] not in ignored_tokens
            ]
        )
    identity: dict[str, object] = {
        "schema_version": ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA,
        "policy_schema_version": ENTITY_GRAPH_POLICY_SCHEMA,
        "semantic_token_sha256": _canonical_sha256(semantic_tokens),
        "selected_symbols": list(_SELECTED_SYMBOLS),
    }
    identity["binding_sha256"] = _canonical_sha256(identity)
    return identity


def _validate_explicit_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("checkpoint forward semantic identity must be an object")
    identity = dict(value)
    if identity.get("schema_version") != ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA:
        raise RuntimeError(
            "checkpoint forward semantic identity has unsupported schema: "
            f"{identity.get('schema_version')!r}"
        )
    if identity.get("policy_schema_version") != ENTITY_GRAPH_POLICY_SCHEMA:
        raise RuntimeError(
            "checkpoint forward semantic identity has wrong policy schema: "
            f"{identity.get('policy_schema_version')!r}"
        )
    if identity.get("selected_symbols") != list(_SELECTED_SYMBOLS):
        raise RuntimeError(
            "checkpoint forward semantic identity names a different executable surface"
        )
    semantic_sha = str(identity.get("semantic_token_sha256", "") or "")
    if not semantic_sha.startswith("sha256:") or len(semantic_sha) != 71:
        raise RuntimeError(
            "checkpoint forward semantic identity has an invalid semantic hash"
        )
    binding_sha = str(identity.pop("binding_sha256", "") or "")
    if binding_sha != _canonical_sha256(identity):
        raise RuntimeError("checkpoint forward semantic identity hash is invalid")
    identity["binding_sha256"] = binding_sha
    return identity


def _legacy_training_source_sha(checkpoint: Mapping[str, object]) -> str:
    value_training = checkpoint.get("value_training")
    if not isinstance(value_training, Mapping):
        return ""
    runtime = value_training.get("checkout_runtime_binding")
    if not isinstance(runtime, Mapping):
        return ""
    modules = runtime.get("modules")
    if not isinstance(modules, Mapping):
        return ""
    entity_policy = modules.get("catan_zero.rl.entity_token_policy")
    if not isinstance(entity_policy, Mapping):
        return ""
    return str(entity_policy.get("sha256", "") or "")


def assert_entity_graph_checkpoint_runtime_semantics(
    checkpoint: Mapping[str, object],
    *,
    checkpoint_path: str | Path,
    policy_source: str | Path,
) -> dict[str, object]:
    """Refuse checkpoints whose trained forward differs from this checkout."""

    current = current_entity_graph_forward_semantics(policy_source)
    explicit = checkpoint.get(ENTITY_GRAPH_FORWARD_SEMANTICS_KEY)
    provenance: str
    if explicit is not None:
        stamped = _validate_explicit_identity(explicit)
        checkpoint_semantic_sha = str(stamped["semantic_token_sha256"])
        provenance = "checkpoint_stamp"
    else:
        source_sha = _legacy_training_source_sha(checkpoint)
        checkpoint_semantic_sha = _REVIEWED_TRAINING_SOURCE_SEMANTICS.get(
            source_sha, ""
        )
        provenance = "reviewed_training_source_binding"
        if not checkpoint_semantic_sha:
            path = Path(checkpoint_path).resolve(strict=True)
            checkpoint_sha = _file_sha256(path)
            legacy_review = _REVIEWED_LEGACY_CHECKPOINTS.get(checkpoint_sha)
            if legacy_review:
                checkpoint_semantic_sha = str(current["semantic_token_sha256"])
                provenance = legacy_review
            else:
                raise RuntimeError(
                    f"{path} has no accepted entity-graph forward semantic identity; "
                    "evaluation is fail-closed. Re-save it through a reviewed "
                    "function-preserving upgrade or add one exact checkpoint review."
                )
    current_semantic_sha = str(current["semantic_token_sha256"])
    if checkpoint_semantic_sha != current_semantic_sha:
        raise RuntimeError(
            "entity-graph checkpoint/runtime forward semantic mismatch: "
            f"checkpoint={checkpoint_semantic_sha} runtime={current_semantic_sha} "
            f"path={Path(checkpoint_path)}"
        )
    return {
        "schema_version": ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA,
        "checkpoint_semantic_token_sha256": checkpoint_semantic_sha,
        "runtime_semantic_token_sha256": current_semantic_sha,
        "provenance": provenance,
        "compatible": True,
    }
