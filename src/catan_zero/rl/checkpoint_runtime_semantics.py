"""Fail-closed binding between entity checkpoints and serving semantics.

Checkpoint tensor shapes and config schemas are not enough to identify a model
function.  A serving checkout can accept every tensor while implementing a
different forward pass.  This module binds the dependency-closed executable
surface from environment features through the entity policy without coupling
checkpoints to comments, saving code, or other non-forward edits.
"""

from __future__ import annotations

import ast
import copy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize
from typing import Iterable, Mapping


ENTITY_GRAPH_FORWARD_SEMANTICS_KEY = "entity_graph_forward_semantics"
ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA = "entity-graph-forward-semantics-v3"
_V2_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA = "entity-graph-forward-semantics-v2"
_LEGACY_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA = "entity-graph-forward-semantics-v1"
ENTITY_GRAPH_POLICY_SCHEMA = "entity_graph_policy_v1"

# These are the source definitions that construct or execute the entity-graph
# neural function.  In particular, save/load/provenance code is excluded so a
# metadata-only release does not invalidate otherwise identical checkpoints.
_POLICY_FORWARD_METHODS = frozenset(
    {"__init__", "forward_legal_np", "_legal_outputs_from_env"}
)
_POLICY_SELECTED_SYMBOLS = (
    "_LEGACY_EVENT_HISTORY_WIDTH",
    "PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO",
    "PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE",
    "PUBLIC_AWARD_FEATURE_CONTRACTS",
    "PLAYER_LONGEST_ROAD_SLOT",
    "_validate_public_award_feature_contract",
    "_apply_public_award_feature_contract",
    "AUX_NUM_INTERSECTIONS",
    "AUX_NUM_HEXES",
    "_entity_token_start_offsets",
    "_NON_MODEL_ENTITY_KEYS",
    "_RELATIONAL_TOPOLOGY_KEYS",
    "STATIC_ACTION_RESIDUAL_SLICE",
    "STATIC_ACTION_RESIDUAL_FEATURE_SIZE",
    "EntityGraphConfig",
    "EntityGraphNet",
    "_token_encoder",
    "EntityGraphPolicy.__init__",
    "EntityGraphPolicy.forward_legal_np",
    "EntityGraphPolicy._legal_outputs_from_env",
    "_assert_entity_batch_shapes",
)
_V2_POLICY_SELECTED_SYMBOLS = (
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

_COMPONENT_MODULES = {
    "entity_token_policy": "catan_zero.rl.entity_token_policy",
    "relational_trunks": "catan_zero.rl.relational_trunks",
    "action_features": "catan_zero.rl.action_features",
    "entity_token_features": "catan_zero.rl.entity_token_features",
    "meaningful_history": "catan_zero.rl.meaningful_history",
    "ordered_history": "catan_zero.rl.ordered_history",
    "deduction_tracker": "catan_zero.deduction_tracker",
    "entity_feature_adapter": "catan_zero.rl.entity_feature_adapter",
}
_V2_RUNTIME_EVIDENCE_COMPONENTS = frozenset(
    {
        "entity_token_policy",
        "action_features",
        "entity_token_features",
        "meaningful_history",
        "ordered_history",
        "deduction_tracker",
        "entity_feature_adapter",
    }
)
_TRAINING_RUNTIME_BINDING_SCHEMA = "train-bc-checkout-runtime-v1"
_LEGACY_ENTITY_FEATURE_ADAPTER_SCHEMA = "entity-feature-adapter-v1"
_LEGACY_ENTITY_FEATURE_ADAPTER_VERSION = (
    "rust_entity_adapter_v2_land_topology_ports_maritime"
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
        "sha256:334f85e44e1c0482a66ccc607d7091e6b123a9f49f658687b74328eec7dbb84e",
        # Adapter v6 is opt-in. This policy revision only expands constructor
        # allowlists for explicitly versioned v6 checkpoints; the legacy
        # missing-metadata v2 adapter path remains numerically unchanged.
        "sha256:99e56f7bd3916b1c18a425f3c0f39dff6b5e274bf3fdc81944205d32fd53fff1",
        # V7 adds only the config-gated, zero-output exact-resource residual
        # path. Legacy checkpoints resolve the flag false and therefore retain
        # the reviewed topology-disabled Transformer function exactly.
        "sha256:60f49b48fa550f36ad709cbaaf041d632370ad8bf6a9f9e3c5ce0f6f37de7758",
        # The complete V7 input migration also adds the config-gated,
        # zero-output initial-road residual. Both compatibility paths are off
        # for legacy checkpoints, so the reviewed topology-disabled function
        # remains identical after the second adapter landed.
        "sha256:ebfaac6c97ac6e9d0f3960814e40d648a43e51c5e1e2e68be42ffb096d2c7e84",
        # V7's final bit-exact inherited-input reconstruction and action
        # decoder remain behind the same absent/false legacy config flags.
        # The default topology-disabled path is therefore still the reviewed
        # legacy function even though its guarded AST surface changed again.
        "sha256:c2f9dd663839489263447263a0b01deb247aa3067cb1958e35465ffd73d6b473",
        # Constructing the zero-init action decoder without consuming RNG is
        # likewise gated to V7 checkpoints. Legacy topology-disabled models
        # still execute the same reviewed forward function.
        "sha256:911589137df1296d927feba059eaebb78704d8d5d41ffbf1dabb6e52016a9df9",
        # Public-history exposure to the V7 action decoder is also gated off
        # for legacy checkpoints, leaving their reviewed forward unchanged.
        "sha256:166693b514e1feb5e73a3c7a23c1a5e0bc39c97884d959a29e9b1b7afb8935e9",
        # Rebasing the V7 decoder onto the final bit-exact input migration
        # changed the selected policy AST fingerprint once more. Every added
        # path remains behind the absent/false V7 compatibility flags for a
        # legacy checkpoint, so the topology-disabled forward is unchanged.
        "sha256:3824e6f68fb23240cfa3193160af3625fa649195a8db31f4c55677d1ad1c992a",
        # The legacy v2 tokenizer produces a different identity for the same
        # source across supported Python f-string tokenization modes. Schema v3
        # uses a stable AST surface, but both reviewed v2 spellings remain.
        "sha256:3e4ca62d77dd5d9f8ddb7cf625a4347f085d6ae08080aaafc9417d59a01180cb",
    }
}

# The canonical f7 incumbent predates checkout-runtime metadata.  Its exact
# bytes are explicitly reviewed against the compatibility bridges in the
# current loader.  No other metadata-free checkpoint inherits this exception.
_REVIEWED_LEGACY_CHECKPOINTS = {
    "sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4": (
        "legacy-f7-current-compat-v1"
    ),
    # The exact recovered 6817 checkpoint records entity_token_policy source
    # SHA a2e3514e9d1e1378ff59e5affbd3aa16d3c74dd44af26f45eb8f35c50e367921
    # in its authenticated checkout binding. Those are exactly the full-file
    # bytes at commit 9e5bc913bca9b1e68949c0dd97e90f949aba0a92. The checkpoint
    # predates adapter metadata, so the loader resolves it only through the
    # explicit missing-metadata mapping to legacy adapter v2; this review does
    # not relabel it as a newer adapter or stamp it with today's source identity.
    "sha256:6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c": (
        "legacy-6817-current-compat-v1"
    ),
    # The sealed gen4 hard-negative opponent is the same topology-disabled,
    # adapter-v2 EntityGraphPolicy family.  It predates all runtime stamps and
    # carries no optional relational or information-surface flags.  Production
    # opponent mixing binds these exact bytes; the exception must not extend to
    # any other metadata-free checkpoint.
    "sha256:b0f939464c138d6d0dca5586585d7e71aacb7ed86183cccbc2131d95750fe1c5": (
        "legacy-gen4-hard-negative-current-compat-v1"
    ),
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


def _assigned_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Assign):
        return tuple(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return (node.target.id,)
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return (node.name,)
    return ()


def _policy_semantic_nodes(
    source: str,
    *,
    selected_symbols: tuple[str, ...] = _POLICY_SELECTED_SYMBOLS,
    selected_methods: frozenset[str] = _POLICY_FORWARD_METHODS,
) -> list[ast.AST]:
    tree = ast.parse(source)
    selected: list[ast.AST] = []
    selected_names: list[str] = []
    top_level_symbols = {
        symbol for symbol in selected_symbols if not symbol.startswith("EntityGraphPolicy.")
    }
    for node in tree.body:
        assigned_names = _assigned_names(node)
        matching_names = [name for name in assigned_names if name in top_level_symbols]
        if matching_names:
            selected.append(node)
            selected_names.extend(matching_names)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == "EntityGraphPolicy":
            for method in node.body:
                if (
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name in selected_methods
                ):
                    selected.append(method)
                    selected_names.append(f"EntityGraphPolicy.{method.name}")
    if selected_names != list(selected_symbols):
        raise RuntimeError(
            "cannot identify the complete entity-graph forward semantic surface"
        )
    return selected


def _selected_top_level_nodes(
    source: str,
    selected_symbols: tuple[str, ...],
    *,
    component_name: str,
) -> list[ast.AST]:
    tree = ast.parse(source)
    selected: list[ast.AST] = []
    selected_names: list[str] = []
    for node in tree.body:
        names = _assigned_names(node)
        matching_names = [name for name in names if name in selected_symbols]
        if matching_names:
            selected.append(node)
            selected_names.extend(matching_names)
    if selected_names != list(selected_symbols):
        raise RuntimeError(
            f"cannot identify the complete {component_name} semantic surface"
        )
    return selected


def _relational_semantic_nodes(source: str) -> list[ast.AST]:
    return _selected_top_level_nodes(
        source,
        _RELATIONAL_SELECTED_SYMBOLS,
        component_name="relational forward",
    )


def _imported_symbols(source: str, module_name: str) -> set[str]:
    imported: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            imported.update(alias.asname or alias.name for alias in node.names)
    return imported


def _loaded_names(nodes: Iterable[ast.AST]) -> set[str]:
    return {
        child.id
        for node in nodes
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _loaded_imports(
    source: str,
    module_name: str,
    loaded_names: set[str],
) -> set[str]:
    return _imported_symbols(source, module_name) & loaded_names


def _semantic_dependency_symbols(
    source: str,
    roots: Iterable[str],
    *,
    component_name: str,
) -> tuple[str, ...]:
    """Return the source-ordered transitive top-level closure of ``roots``."""

    tree = ast.parse(source)
    nodes_by_name: dict[str, ast.AST] = {}
    source_order: list[str] = []
    for node in tree.body:
        for name in _assigned_names(node):
            nodes_by_name[name] = node
            source_order.append(name)
    missing = set(roots) - set(nodes_by_name)
    if missing:
        raise RuntimeError(
            f"cannot identify {component_name} roots: {sorted(missing)}"
        )
    reachable = set(roots)
    pending = list(roots)
    while pending:
        name = pending.pop()
        node = nodes_by_name[name]
        dependencies = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id in nodes_by_name
        }
        for dependency in dependencies - reachable:
            reachable.add(dependency)
            pending.append(dependency)
    return tuple(name for name in source_order if name in reachable)


def _legacy_semantic_tokens(source: str, nodes: list[ast.AST]) -> list[object]:
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


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
        return node

    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def _canonical_ast_value(value: object) -> object:
    if isinstance(value, ast.AST):
        fields = []
        for name, field_value in ast.iter_fields(value):
            if field_value is None or field_value == []:
                continue
            fields.append([name, _canonical_ast_value(field_value)])
        return [type(value).__name__, fields]
    if isinstance(value, list):
        return [_canonical_ast_value(item) for item in value]
    if isinstance(value, tuple):
        return ["tuple", [_canonical_ast_value(item) for item in value]]
    if value is Ellipsis:
        return ["literal", "Ellipsis"]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, complex):
        return ["complex", repr(value)]
    return value


def _semantic_ast(nodes: list[ast.AST]) -> list[object]:
    """Canonicalize executable AST independent of tokenizer/Python f-string modes."""

    stripper = _DocstringStripper()
    return [
        _canonical_ast_value(stripper.visit(copy.deepcopy(node))) for node in nodes
    ]


def _component_identity(
    source: str,
    *,
    selected_symbols: tuple[str, ...],
    nodes: list[ast.AST],
    legacy_tokens: bool = False,
) -> dict[str, object]:
    semantic_surface = (
        _legacy_semantic_tokens(source, nodes)
        if legacy_tokens
        else _semantic_ast(nodes)
    )
    return {
        "selected_symbols": list(selected_symbols),
        "semantic_token_sha256": _canonical_sha256(semantic_surface),
    }


def _resolve_component_sources(
    policy_source: str | Path,
    relational_source: str | Path | None,
    action_features_source: str | Path | None,
    entity_token_features_source: str | Path | None,
    meaningful_history_source: str | Path | None,
    ordered_history_source: str | Path | None,
    deduction_tracker_source: str | Path | None,
    entity_feature_adapter_source: str | Path | None,
) -> dict[str, Path]:
    policy_path = Path(policy_source).resolve(strict=True)
    rl_dir = policy_path.parent

    def _resolve(value: str | Path | None, default: Path) -> Path:
        return (Path(value) if value is not None else default).resolve(strict=True)

    return {
        "entity_token_policy": policy_path,
        "relational_trunks": _resolve(
            relational_source, rl_dir / "relational_trunks.py"
        ),
        "action_features": _resolve(
            action_features_source, rl_dir / "action_features.py"
        ),
        "entity_token_features": _resolve(
            entity_token_features_source, rl_dir / "entity_token_features.py"
        ),
        "meaningful_history": _resolve(
            meaningful_history_source, rl_dir / "meaningful_history.py"
        ),
        "ordered_history": _resolve(
            ordered_history_source, rl_dir / "ordered_history.py"
        ),
        "deduction_tracker": _resolve(
            deduction_tracker_source, rl_dir.parent / "deduction_tracker.py"
        ),
        "entity_feature_adapter": _resolve(
            entity_feature_adapter_source, rl_dir / "entity_feature_adapter.py"
        ),
    }


@lru_cache(maxsize=16)
def current_entity_graph_forward_semantics(
    policy_source: str | Path,
    relational_source: str | Path | None = None,
    *,
    action_features_source: str | Path | None = None,
    entity_token_features_source: str | Path | None = None,
    meaningful_history_source: str | Path | None = None,
    ordered_history_source: str | Path | None = None,
    deduction_tracker_source: str | Path | None = None,
    entity_feature_adapter_source: str | Path | None = None,
) -> dict[str, object]:
    """Return the canonical, comment-insensitive multi-file neural identity."""

    source_paths = _resolve_component_sources(
        policy_source,
        relational_source,
        action_features_source,
        entity_token_features_source,
        meaningful_history_source,
        ordered_history_source,
        deduction_tracker_source,
        entity_feature_adapter_source,
    )
    source_text = {
        name: path.read_text(encoding="utf-8") for name, path in source_paths.items()
    }
    policy_nodes = _policy_semantic_nodes(source_text["entity_token_policy"])
    policy_loaded = _loaded_names(policy_nodes)
    components = {
        "entity_token_policy": _component_identity(
            source_text["entity_token_policy"],
            selected_symbols=_POLICY_SELECTED_SYMBOLS,
            nodes=policy_nodes,
        ),
        "relational_trunks": _component_identity(
            source_text["relational_trunks"],
            selected_symbols=_RELATIONAL_SELECTED_SYMBOLS,
            nodes=_relational_semantic_nodes(source_text["relational_trunks"]),
        ),
    }

    selected_loaded: dict[str, set[str]] = {
        "entity_token_policy": policy_loaded,
    }

    def _add_dependency_component(name: str, roots: set[str]) -> None:
        selected_symbols = _semantic_dependency_symbols(
            source_text[name],
            roots,
            component_name=name,
        )
        nodes = _selected_top_level_nodes(
            source_text[name],
            selected_symbols,
            component_name=name,
        )
        selected_loaded[name] = _loaded_names(nodes)
        components[name] = _component_identity(
            source_text[name],
            selected_symbols=selected_symbols,
            nodes=nodes,
        )

    _add_dependency_component(
        "action_features",
        _loaded_imports(
            source_text["entity_token_policy"],
            _COMPONENT_MODULES["action_features"],
            policy_loaded,
        ),
    )
    _add_dependency_component(
        "entity_token_features",
        _loaded_imports(
            source_text["entity_token_policy"],
            _COMPONENT_MODULES["entity_token_features"],
            policy_loaded,
        )
        | _loaded_imports(
            source_text["action_features"],
            _COMPONENT_MODULES["entity_token_features"],
            selected_loaded["action_features"],
        ),
    )
    consumers = (
        "entity_token_policy",
        "action_features",
        "entity_token_features",
    )
    for name in (
        "meaningful_history",
        "deduction_tracker",
        "entity_feature_adapter",
    ):
        roots = set().union(
            *(
                _loaded_imports(
                    source_text[consumer],
                    _COMPONENT_MODULES[name],
                    selected_loaded[consumer],
                )
                for consumer in consumers
            )
        )
        _add_dependency_component(name, roots)
    _add_dependency_component(
        "ordered_history",
        _loaded_imports(
            source_text["entity_token_policy"],
            _COMPONENT_MODULES["ordered_history"],
            policy_loaded,
        ),
    )
    identity: dict[str, object] = {
        "schema_version": ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA,
        "policy_schema_version": ENTITY_GRAPH_POLICY_SCHEMA,
        "components": components,
        "semantic_token_sha256": _canonical_sha256(components),
    }
    identity["binding_sha256"] = _canonical_sha256(identity)
    return identity


@lru_cache(maxsize=16)
def _current_v2_entity_graph_forward_semantics(
    policy_source: str | Path,
    relational_source: str | Path | None = None,
) -> dict[str, object]:
    policy_path = Path(policy_source).resolve(strict=True)
    relational_path = (
        Path(relational_source).resolve(strict=True)
        if relational_source is not None
        else policy_path.with_name("relational_trunks.py").resolve(strict=True)
    )
    policy_text = policy_path.read_text(encoding="utf-8")
    relational_text = relational_path.read_text(encoding="utf-8")
    components = {
        "entity_token_policy": _component_identity(
            policy_text,
            selected_symbols=_V2_POLICY_SELECTED_SYMBOLS,
            nodes=_policy_semantic_nodes(
                policy_text,
                selected_symbols=_V2_POLICY_SELECTED_SYMBOLS,
                selected_methods=frozenset({"__init__", "forward_legal_np"}),
            ),
            legacy_tokens=True,
        ),
        "relational_trunks": _component_identity(
            relational_text,
            selected_symbols=_RELATIONAL_SELECTED_SYMBOLS,
            nodes=_relational_semantic_nodes(relational_text),
            legacy_tokens=True,
        ),
    }
    identity: dict[str, object] = {
        "schema_version": _V2_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA,
        "policy_schema_version": ENTITY_GRAPH_POLICY_SCHEMA,
        "components": components,
        "semantic_token_sha256": _canonical_sha256(components),
    }
    identity["binding_sha256"] = _canonical_sha256(identity)
    return identity


def _validate_explicit_identity(
    value: object,
    *,
    expected_identity: Mapping[str, object],
) -> dict[str, object]:
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
    expected_components = expected_identity.get("components")
    if (
        not isinstance(components, Mapping)
        or not isinstance(expected_components, Mapping)
        or set(components) != set(expected_components)
    ):
        raise RuntimeError(
            "checkpoint forward semantic identity has an incomplete source surface"
        )
    for name, expected_component in expected_components.items():
        if not isinstance(expected_component, Mapping):
            raise RuntimeError(f"runtime semantic component is malformed: {name}")
        symbols = expected_component.get("selected_symbols")
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


def _validate_v2_explicit_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("checkpoint forward semantic identity must be an object")
    identity = dict(value)
    if (
        identity.get("schema_version")
        != _V2_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA
        or identity.get("policy_schema_version") != ENTITY_GRAPH_POLICY_SCHEMA
    ):
        raise RuntimeError("checkpoint v2 forward semantic identity is malformed")
    components = identity.get("components")
    expected_symbols = {
        "entity_token_policy": list(_V2_POLICY_SELECTED_SYMBOLS),
        "relational_trunks": list(_RELATIONAL_SELECTED_SYMBOLS),
    }
    if not isinstance(components, Mapping) or set(components) != set(expected_symbols):
        raise RuntimeError(
            "checkpoint v2 forward semantic identity has an incomplete source surface"
        )
    for name, symbols in expected_symbols.items():
        component = components.get(name)
        semantic_sha = (
            str(component.get("semantic_token_sha256", ""))
            if isinstance(component, Mapping)
            else ""
        )
        if (
            not isinstance(component, Mapping)
            or component.get("selected_symbols") != symbols
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", semantic_sha)
        ):
            raise RuntimeError(
                "checkpoint v2 forward semantic identity names a different "
                f"executable surface: {name}"
            )
    semantic_sha = str(identity.get("semantic_token_sha256", "") or "")
    if semantic_sha != _canonical_sha256(dict(components)):
        raise RuntimeError("checkpoint v2 forward semantic identity has invalid hash")
    binding_sha = str(identity.pop("binding_sha256", "") or "")
    if binding_sha != _canonical_sha256(identity):
        raise RuntimeError("checkpoint v2 forward semantic identity hash is invalid")
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
        or identity.get("selected_symbols") != list(_V2_POLICY_SELECTED_SYMBOLS)
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


def _uses_unbound_feature_adapter_semantics(
    checkpoint: Mapping[str, object],
) -> bool:
    """Identify adapter semantics a v1/unstamped checkpoint cannot authorize."""

    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        fields = config.get("fields", config)
        if isinstance(fields, Mapping) and "entity_feature_adapter_version" in fields:
            if (
                str(fields.get("entity_feature_adapter_version") or "")
                != _LEGACY_ENTITY_FEATURE_ADAPTER_VERSION
            ):
                return True
    if "entity_feature_adapter" not in checkpoint:
        return False
    raw = checkpoint.get("entity_feature_adapter")
    return not (
        isinstance(raw, Mapping)
        and raw.get("schema_version") == _LEGACY_ENTITY_FEATURE_ADAPTER_SCHEMA
        and raw.get("version") == _LEGACY_ENTITY_FEATURE_ADAPTER_VERSION
    )


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


def _assert_v2_runtime_module_evidence(
    checkpoint: Mapping[str, object],
    source_paths: Mapping[str, Path],
) -> None:
    value_training = checkpoint.get("value_training")
    runtime = (
        value_training.get("checkout_runtime_binding")
        if isinstance(value_training, Mapping)
        else None
    )
    if not isinstance(runtime, Mapping):
        raise RuntimeError(
            "v2 checkpoint stamp does not authenticate the newly bound runtime modules"
        )
    binding = dict(runtime)
    binding_sha = str(binding.pop("binding_sha256", "") or "")
    if (
        binding.get("schema_version") != _TRAINING_RUNTIME_BINDING_SCHEMA
        or binding_sha != _canonical_sha256(binding)
    ):
        raise RuntimeError(
            "v2 checkpoint checkout runtime module evidence is unauthenticated"
        )
    modules = binding.get("modules")
    if not isinstance(modules, Mapping):
        raise RuntimeError("v2 checkpoint runtime module evidence is malformed")
    for component in sorted(_V2_RUNTIME_EVIDENCE_COMPONENTS):
        module_name = _COMPONENT_MODULES[component]
        record = modules.get(module_name)
        recorded_sha = (
            str(record.get("sha256", "")) if isinstance(record, Mapping) else ""
        )
        current_sha = _file_sha256(source_paths[component])
        if recorded_sha != current_sha:
            raise RuntimeError(
                "v2 checkpoint runtime evidence does not cover current newly bound "
                f"component {component}: checkpoint={recorded_sha or '<missing>'} "
                f"runtime={current_sha}"
            )


def assert_entity_graph_checkpoint_runtime_semantics(
    checkpoint: Mapping[str, object],
    *,
    checkpoint_path: str | Path,
    policy_source: str | Path,
    relational_source: str | Path | None = None,
    action_features_source: str | Path | None = None,
    entity_token_features_source: str | Path | None = None,
    meaningful_history_source: str | Path | None = None,
    ordered_history_source: str | Path | None = None,
    deduction_tracker_source: str | Path | None = None,
    entity_feature_adapter_source: str | Path | None = None,
) -> dict[str, object]:
    """Refuse checkpoints whose trained forward differs from this checkout."""

    source_paths = _resolve_component_sources(
        policy_source,
        relational_source,
        action_features_source,
        entity_token_features_source,
        meaningful_history_source,
        ordered_history_source,
        deduction_tracker_source,
        entity_feature_adapter_source,
    )
    current = current_entity_graph_forward_semantics(
        policy_source,
        relational_source,
        action_features_source=action_features_source,
        entity_token_features_source=entity_token_features_source,
        meaningful_history_source=meaningful_history_source,
        ordered_history_source=ordered_history_source,
        deduction_tracker_source=deduction_tracker_source,
        entity_feature_adapter_source=entity_feature_adapter_source,
    )
    current_v2 = _current_v2_entity_graph_forward_semantics(
        policy_source, relational_source
    )
    explicit = checkpoint.get(ENTITY_GRAPH_FORWARD_SEMANTICS_KEY)
    provenance: str
    if explicit is not None:
        explicit_schema = (
            explicit.get("schema_version") if isinstance(explicit, Mapping) else None
        )
        if explicit_schema == _LEGACY_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA:
            if _uses_unbound_feature_adapter_semantics(checkpoint):
                raise RuntimeError(
                    "legacy checkpoint stamp does not bind requested feature adapter "
                    "semantics"
                )
            if _uses_relational_forward_semantics(checkpoint):
                raise RuntimeError(
                    "legacy checkpoint stamp does not bind relational topology semantics"
                )
            stamped = _validate_legacy_explicit_identity(explicit)
            checkpoint_semantic_sha = str(stamped["semantic_token_sha256"])
            checkpoint_current_sha = str(
                current_v2["components"]["entity_token_policy"][
                    "semantic_token_sha256"
                ]
            )
            provenance = "checkpoint_stamp_v1_topology_disabled_compat"
        elif explicit_schema == _V2_ENTITY_GRAPH_FORWARD_SEMANTICS_SCHEMA:
            stamped = _validate_v2_explicit_identity(explicit)
            checkpoint_semantic_sha = str(stamped["semantic_token_sha256"])
            checkpoint_current_sha = str(current_v2["semantic_token_sha256"])
            if checkpoint_semantic_sha != checkpoint_current_sha:
                raise RuntimeError(
                    "entity-graph v2 checkpoint/runtime forward semantic mismatch: "
                    f"checkpoint={checkpoint_semantic_sha} "
                    f"runtime={checkpoint_current_sha} path={Path(checkpoint_path)}"
                )
            _assert_v2_runtime_module_evidence(checkpoint, source_paths)
            provenance = "checkpoint_stamp_v2_authenticated_runtime_compat"
        else:
            stamped = _validate_explicit_identity(
                explicit,
                expected_identity=current,
            )
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
                    current_v2["components"]["entity_token_policy"][
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
        if _uses_unbound_feature_adapter_semantics(checkpoint):
            raise RuntimeError(
                "unstamped checkpoint does not bind requested feature adapter semantics"
            )
        if _uses_relational_forward_semantics(checkpoint):
            raise RuntimeError(
                "unstamped checkpoint does not bind relational topology semantics"
            )
        checkpoint_current_sha = str(
            current_v2["components"]["entity_token_policy"]["semantic_token_sha256"]
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
        "runtime_semantic_token_sha256": str(current["semantic_token_sha256"]),
        "provenance": provenance,
        "compatible": True,
    }
