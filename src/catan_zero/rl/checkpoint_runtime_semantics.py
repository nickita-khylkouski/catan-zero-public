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
ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA = "entity-graph-forward-semantics-v2"
_LEGACY_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA = "entity-graph-forward-semantics-v1"
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
_POLICY_SELECTED_SYMBOLS = (
    "_validate_public_award_feature_contract",
    "_apply_public_award_feature_contract",
    "_entity_token_start_offsets",
    "EntityGraphNet",
    "_token_encoder",
    "EntityGraphPolicy.__init__",
    "EntityGraphPolicy.forward_legal_np",
    "_assert_entity_batch_shapes",
)
_RELATIONAL_SELECTED_SYMBOLS = (
    "RELATION_COUNT",
    "DISTANCE_BUCKETS",
    "REL_NONE",
    "REL_SELF",
    "REL_HEX_TO_VERTEX",
    "REL_VERTEX_TO_HEX",
    "REL_HEX_TO_EDGE",
    "REL_EDGE_TO_HEX",
    "REL_EDGE_TO_VERTEX",
    "REL_VERTEX_TO_EDGE",
    "REL_HUB_READS",
    "REL_READ_GLOBAL",
    "REL_EVENT_TO_TARGET",
    "REL_TARGET_TO_EVENT",
    "TopologyResidualAdapter",
    "build_relation_ids",
    "RelationalAttention",
    "RelationalTransformerBlock",
    "VectorizedRelGraphBlock",
    "SparseTopKMoE",
    "SparseMoERelationalTransformerBlock",
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

# Reviewed compatibility is deliberately directional and restricted to the
# topology-disabled historical Transformer. The newer policy fingerprint adds
# the zero-output topology construction branch; with the config gate off, that
# branch and every relational_trunks symbol are unreachable, so the executable
# legacy function remains 460f. _uses_relational_forward_semantics is checked
# before this translation can authorize either the adapter or rrt/resrgcn.
_REVIEWED_LEGACY_POLICY_SEMANTIC_TRANSLATIONS = {
    "sha256:460f78322abb3af4ba5255593b9ef3a4db93c53a114fdef15a9a8f1dae828f7e": {
        "sha256:334f85e44e1c0482a66ccc607d7091e6b123a9f49f658687b74328eec7dbb84e"
    }
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


def _policy_semantic_nodes(source: str) -> list[ast.AST]:
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
    if len(selected) != len(_POLICY_SELECTED_SYMBOLS):
        raise RuntimeError(
            "cannot identify the complete entity-graph forward semantic surface"
        )
    return selected


def _relational_semantic_nodes(source: str) -> list[ast.AST]:
    tree = ast.parse(source)
    selected: list[ast.AST] = []
    selected_names: list[str] = []
    for node in tree.body:
        name: str | None = None
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            name = target.id if isinstance(target, ast.Name) else None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name in _RELATIONAL_SELECTED_SYMBOLS:
            selected.append(node)
            selected_names.append(name)
    if selected_names != list(_RELATIONAL_SELECTED_SYMBOLS):
        raise RuntimeError(
            "cannot identify the complete relational forward semantic surface"
        )
    return selected


def _semantic_tokens(source: str, nodes: list[ast.AST]) -> list[object]:
    ignored_tokens = {
        "ENCODING",
        "COMMENT",
        "NL",
        "NEWLINE",
        "INDENT",
        "DEDENT",
        "ENDMARKER",
    }
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
    return semantic_tokens


def _component_identity(
    source_path: Path,
    *,
    selected_symbols: tuple[str, ...],
    nodes: list[ast.AST],
) -> dict[str, object]:
    source = source_path.read_text(encoding="utf-8")
    return {
        "selected_symbols": list(selected_symbols),
        "semantic_token_sha256": _canonical_sha256(_semantic_tokens(source, nodes)),
    }


@lru_cache(maxsize=4)
def current_entity_graph_forward_semantics(
    policy_source: str | Path,
    relational_source: str | Path | None = None,
) -> dict[str, object]:
    """Return the canonical, comment-insensitive multi-file neural identity."""

    source_path = Path(policy_source).resolve(strict=True)
    relational_path = (
        Path(relational_source).resolve(strict=True)
        if relational_source is not None
        else source_path.with_name("relational_trunks.py").resolve(strict=True)
    )
    policy_text = source_path.read_text(encoding="utf-8")
    relational_text = relational_path.read_text(encoding="utf-8")
    components = {
        "entity_token_policy": _component_identity(
            source_path,
            selected_symbols=_POLICY_SELECTED_SYMBOLS,
            nodes=_policy_semantic_nodes(policy_text),
        ),
        "relational_trunks": _component_identity(
            relational_path,
            selected_symbols=_RELATIONAL_SELECTED_SYMBOLS,
            nodes=_relational_semantic_nodes(relational_text),
        ),
    }
    identity: dict[str, object] = {
        "schema_version": ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA,
        "policy_schema_version": ENTITY_GRAPH_POLICY_SCHEMA,
        "components": components,
        "semantic_token_sha256": _canonical_sha256(components),
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
    components = identity.get("components")
    if not isinstance(components, Mapping) or set(components) != {
        "entity_token_policy",
        "relational_trunks",
    }:
        raise RuntimeError(
            "checkpoint forward semantic identity has an incomplete source surface"
        )
    expected_symbols = {
        "entity_token_policy": list(_POLICY_SELECTED_SYMBOLS),
        "relational_trunks": list(_RELATIONAL_SELECTED_SYMBOLS),
    }
    for name, symbols in expected_symbols.items():
        component = components.get(name)
        if (
            not isinstance(component, Mapping)
            or component.get("selected_symbols") != symbols
            or not str(component.get("semantic_token_sha256", "")).startswith(
                "sha256:"
            )
            or len(str(component.get("semantic_token_sha256", ""))) != 71
        ):
            raise RuntimeError(
                "checkpoint forward semantic identity names a different executable "
                f"surface: {name}"
            )
    semantic_sha = str(identity.get("semantic_token_sha256", "") or "")
    if (
        semantic_sha != _canonical_sha256(dict(components))
        or not semantic_sha.startswith("sha256:")
        or len(semantic_sha) != 71
    ):
        raise RuntimeError(
            "checkpoint forward semantic identity has an invalid semantic hash"
        )
    binding_sha = str(identity.pop("binding_sha256", "") or "")
    if binding_sha != _canonical_sha256(identity):
        raise RuntimeError("checkpoint forward semantic identity hash is invalid")
    identity["binding_sha256"] = binding_sha
    return identity


def _validate_legacy_explicit_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("checkpoint forward semantic identity must be an object")
    identity = dict(value)
    if (
        identity.get("schema_version")
        != _LEGACY_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA
        or identity.get("policy_schema_version") != ENTITY_GRAPH_POLICY_SCHEMA
        or identity.get("selected_symbols") != list(_POLICY_SELECTED_SYMBOLS)
    ):
        raise RuntimeError("checkpoint legacy forward semantic identity is malformed")
    semantic_sha = str(identity.get("semantic_token_sha256", "") or "")
    binding_sha = str(identity.pop("binding_sha256", "") or "")
    if (
        not semantic_sha.startswith("sha256:")
        or len(semantic_sha) != 71
        or binding_sha != _canonical_sha256(identity)
    ):
        raise RuntimeError("checkpoint legacy forward semantic identity hash is invalid")
    identity["binding_sha256"] = binding_sha
    return identity


def _uses_relational_forward_semantics(checkpoint: Mapping[str, object]) -> bool:
    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        fields = config.get("fields", config)
        if not isinstance(fields, Mapping):
            raise RuntimeError("checkpoint config fields are malformed")
        state_trunk = str(fields.get("state_trunk", "transformer"))
        topology = bool(fields.get("topology_residual_adapter", False))
    elif config is not None:
        state_trunk = str(getattr(config, "state_trunk", "transformer"))
        topology = bool(getattr(config, "topology_residual_adapter", False))
    else:
        raise RuntimeError(
            "legacy checkpoint has no config proving topology-disabled transformer semantics"
        )
    return state_trunk != "transformer" or topology


def _reviewed_policy_semantics_match(checkpoint_sha: str, runtime_sha: str) -> bool:
    return checkpoint_sha == runtime_sha or runtime_sha in (
        _REVIEWED_LEGACY_POLICY_SEMANTIC_TRANSLATIONS.get(checkpoint_sha, set())
    )


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
    relational_source: str | Path | None = None,
) -> dict[str, object]:
    """Refuse checkpoints whose trained forward differs from this checkout."""

    current = current_entity_graph_forward_semantics(policy_source, relational_source)
    explicit = checkpoint.get(ENTITY_GRAPH_FORWARD_SEMANTICS_KEY)
    provenance: str
    if explicit is not None:
        explicit_schema = (
            explicit.get("schema_version") if isinstance(explicit, Mapping) else None
        )
        if explicit_schema == _LEGACY_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA:
            if _uses_relational_forward_semantics(checkpoint):
                raise RuntimeError(
                    "legacy checkpoint stamp does not bind relational topology semantics"
                )
            stamped = _validate_legacy_explicit_identity(explicit)
            checkpoint_semantic_sha = str(stamped["semantic_token_sha256"])
            checkpoint_current_sha = str(
                current["components"]["entity_token_policy"]["semantic_token_sha256"]
            )
            provenance = "checkpoint_stamp_v1_topology_disabled_compat"
        else:
            stamped = _validate_explicit_identity(explicit)
            checkpoint_semantic_sha = str(stamped["semantic_token_sha256"])
            checkpoint_current_sha = str(current["semantic_token_sha256"])
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
                checkpoint_semantic_sha = str(
                    current["components"]["entity_token_policy"][
                        "semantic_token_sha256"
                    ]
                )
                provenance = legacy_review
            else:
                raise RuntimeError(
                    f"{path} has no accepted entity-graph forward semantic identity; "
                    "evaluation is fail-closed. Re-save it through a reviewed "
                    "function-preserving upgrade or add one exact checkpoint review."
                )
        if _uses_relational_forward_semantics(checkpoint):
            raise RuntimeError(
                "unstamped checkpoint does not bind relational topology semantics"
            )
        checkpoint_current_sha = str(
            current["components"]["entity_token_policy"]["semantic_token_sha256"]
        )
    if not _reviewed_policy_semantics_match(
        checkpoint_semantic_sha, checkpoint_current_sha
    ):
        raise RuntimeError(
            "entity-graph checkpoint/runtime forward semantic mismatch: "
            f"checkpoint={checkpoint_semantic_sha} runtime={checkpoint_current_sha} "
            f"path={Path(checkpoint_path)}"
        )
    return {
        "schema_version": ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA,
        "checkpoint_semantic_token_sha256": checkpoint_semantic_sha,
        "runtime_semantic_token_sha256": checkpoint_current_sha,
        "provenance": provenance,
        "compatible": True,
    }
