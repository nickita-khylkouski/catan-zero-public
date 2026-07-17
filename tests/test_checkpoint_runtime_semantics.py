from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.checkpoint_runtime_semantics import (  # noqa: E402
    ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
    _current_v2_entity_graph_forward_semantics,
    assert_entity_graph_checkpoint_runtime_semantics,
    current_entity_graph_forward_semantics,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)
from catan_zero.rl.entity_feature_adapter import (  # noqa: E402
    RUST_ENTITY_ADAPTER_V6,
    checkpoint_entity_feature_adapter_metadata,
)
from tools import gumbel_search_cross_net_h2h as h2h  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
POLICY_SOURCE = REPO / "src/catan_zero/rl/entity_token_policy.py"
RL_SOURCE = POLICY_SOURCE.parent
DEPENDENCY_SOURCES = {
    "relational_source": RL_SOURCE / "relational_trunks.py",
    "action_features_source": RL_SOURCE / "action_features.py",
    "entity_token_features_source": RL_SOURCE / "entity_token_features.py",
    "meaningful_history_source": RL_SOURCE / "meaningful_history.py",
    "ordered_history_source": RL_SOURCE / "ordered_history.py",
    "deduction_tracker_source": RL_SOURCE.parent / "deduction_tracker.py",
    "entity_feature_adapter_source": RL_SOURCE / "entity_feature_adapter.py",
}
RUNTIME_MODULE_SOURCES = {
    "catan_zero.rl.entity_token_policy": POLICY_SOURCE,
    "catan_zero.rl.action_features": DEPENDENCY_SOURCES["action_features_source"],
    "catan_zero.rl.entity_token_features": DEPENDENCY_SOURCES[
        "entity_token_features_source"
    ],
    "catan_zero.rl.meaningful_history": DEPENDENCY_SOURCES[
        "meaningful_history_source"
    ],
    "catan_zero.rl.ordered_history": DEPENDENCY_SOURCES[
        "ordered_history_source"
    ],
    "catan_zero.deduction_tracker": DEPENDENCY_SOURCES[
        "deduction_tracker_source"
    ],
    "catan_zero.rl.entity_feature_adapter": DEPENDENCY_SOURCES[
        "entity_feature_adapter_source"
    ],
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _identity(*, semantic_sha: str | None = None) -> dict[str, object]:
    identity = copy.deepcopy(current_entity_graph_forward_semantics(POLICY_SOURCE))
    identity.pop("binding_sha256")
    if semantic_sha is not None:
        identity["components"]["entity_token_policy"][
            "semantic_token_sha256"
        ] = semantic_sha
        identity["semantic_token_sha256"] = _canonical_sha256(
            identity["components"]
        )
    identity["binding_sha256"] = _canonical_sha256(identity)
    return identity


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _authenticated_runtime_binding(
    *,
    overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    overrides = overrides or {}
    binding: dict[str, object] = {
        "schema_version": "train-bc-checkout-runtime-v1",
        "repo_root": str(REPO),
        "source_root": str(REPO / "src"),
        "trainer": str(REPO / "tools/train_bc.py"),
        "trainer_sha256": "sha256:" + "0" * 64,
        "modules": {
            module: {
                "path": str(path),
                "sha256": overrides.get(module, _file_sha256(path)),
            }
            for module, path in RUNTIME_MODULE_SOURCES.items()
        },
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding


def _v2_checkpoint(*, runtime_binding: object | None) -> dict[str, object]:
    payload: dict[str, object] = {
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: copy.deepcopy(
            _current_v2_entity_graph_forward_semantics(POLICY_SOURCE)
        )
    }
    if runtime_binding is not None:
        payload["value_training"] = {
            "checkout_runtime_binding": runtime_binding,
        }
    return payload


def _metadata_checkpoint(path: Path, identity: dict[str, object]) -> Path:
    torch.save({ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: identity}, path)
    return path


def test_new_entity_checkpoint_stamps_forward_semantics(tmp_path: Path) -> None:
    config = EntityGraphConfig(
        action_size=1,
        static_action_feature_size=1,
        hidden_size=8,
        state_layers=1,
        attention_heads=1,
        dropout=0.0,
    )
    policy = EntityGraphPolicy(
        config,
        np.zeros((1, 1), dtype=np.float32),
        device="cpu",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    policy.save(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] == _identity()


def test_h2h_preflight_accepts_same_forward_semantics(tmp_path: Path) -> None:
    candidate = _metadata_checkpoint(tmp_path / "candidate.pt", _identity())
    baseline = _metadata_checkpoint(tmp_path / "baseline.pt", _identity())

    result = h2h._preflight_checkpoint_runtime_semantics(candidate, baseline)

    assert result["candidate"]["compatible"] is True
    assert result["baseline"]["compatible"] is True
    assert result["candidate"]["provenance"] == "checkpoint_stamp"
    assert (
        result["candidate"]["entity_feature_adapter_provenance"]
        == "legacy_missing_metadata_explicit_v2_mapping"
    )


def test_h2h_native_preflight_rejects_stale_context_runtime_before_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: _identity(),
        "entity_feature_adapter": checkpoint_entity_feature_adapter_metadata(
            RUST_ENTITY_ADAPTER_V6
        ),
    }
    candidate = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pt"
    torch.save(payload, candidate)
    torch.save(payload, baseline)
    evidence = h2h._preflight_checkpoint_runtime_semantics(candidate, baseline)
    stale_native = type("StaleNative", (), {})()
    monkeypatch.setattr(
        h2h.importlib,
        "import_module",
        lambda name: stale_native if name == "catanatron_rs" else None,
    )

    with pytest.raises(
        RuntimeError,
        match="candidate checkpoint cannot use the loaded native context runtime",
    ):
        h2h._preflight_native_context_adapters(evidence)


def test_h2h_native_preflight_accepts_advertised_v6_context_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: _identity(),
        "entity_feature_adapter": checkpoint_entity_feature_adapter_metadata(
            RUST_ENTITY_ADAPTER_V6
        ),
    }
    candidate = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pt"
    torch.save(payload, candidate)
    torch.save(payload, baseline)
    evidence = h2h._preflight_checkpoint_runtime_semantics(candidate, baseline)
    native = type(
        "Native",
        (),
        {
            "supported_action_context_adapter_versions": staticmethod(
                lambda: [RUST_ENTITY_ADAPTER_V6]
            )
        },
    )()
    monkeypatch.setattr(
        h2h.importlib,
        "import_module",
        lambda name: native if name == "catanatron_rs" else None,
    )

    h2h._preflight_native_context_adapters(evidence)


def test_h2h_preflight_rejects_mismatch_before_evaluator_or_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _metadata_checkpoint(
        tmp_path / "candidate.pt",
        _identity(semantic_sha="sha256:" + "a" * 64),
    )
    baseline = _metadata_checkpoint(tmp_path / "baseline.pt", _identity())

    evaluator_constructed = False

    def _unexpected_evaluator(*_args, **_kwargs):
        nonlocal evaluator_constructed
        evaluator_constructed = True
        raise AssertionError("evaluator must not be constructed during preflight")

    monkeypatch.setattr(
        h2h.BatchedEntityGraphRustEvaluator,
        "from_checkpoint",
        _unexpected_evaluator,
    )
    with pytest.raises(RuntimeError, match="forward semantic mismatch"):
        h2h._preflight_checkpoint_runtime_semantics(candidate, baseline)

    assert evaluator_constructed is False


def test_reviewed_b4_training_binding_is_accepted_without_new_stamp(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pre-stamp.pt"
    checkpoint.write_bytes(b"only path identity is needed for stamped training source")
    payload = {
        "config": {
            "fields": {
                "state_trunk": "transformer",
                "topology_residual_adapter": False,
            }
        },
        "value_training": {
            "checkout_runtime_binding": {
                "modules": {
                    "catan_zero.rl.entity_token_policy": {
                        "sha256": (
                            "sha256:b4e2618bc36296470f13ce3dee228b34fd7d117c"
                            "0211380c46393450793ce975"
                        )
                    }
                }
            }
        }
    }

    result = assert_entity_graph_checkpoint_runtime_semantics(
        payload,
        checkpoint_path=checkpoint,
        policy_source=POLICY_SOURCE,
    )

    assert result["compatible"] is True
    assert result["provenance"] == "reviewed_training_source_binding"

    payload["config"]["fields"]["topology_residual_adapter"] = True
    with pytest.raises(RuntimeError, match="does not bind relational topology"):
        assert_entity_graph_checkpoint_runtime_semantics(
            payload,
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
        )


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "adjacency.sum(dim=-1, keepdim=True)",
            "adjacency.sum(dim=-1, keepdim=True) + 1.0",
        ),
        (
            "qkv = self.qkv(x).reshape(",
            "qkv = (self.qkv(x) * 0.5).reshape(",
        ),
    ),
)
def test_relational_semantic_mutation_changes_binding_and_is_rejected(
    tmp_path: Path, old: str, new: str
) -> None:
    relational_source = REPO / "src/catan_zero/rl/relational_trunks.py"
    mutated_source = tmp_path / "relational_trunks.py"
    original = relational_source.read_text(encoding="utf-8")
    mutated = original.replace(old, new, 1)
    assert mutated != original
    mutated_source.write_text(mutated, encoding="utf-8")

    stamped = current_entity_graph_forward_semantics(POLICY_SOURCE)
    changed = current_entity_graph_forward_semantics(POLICY_SOURCE, mutated_source)
    assert stamped["semantic_token_sha256"] != changed["semantic_token_sha256"]
    assert (
        stamped["components"]["relational_trunks"]["semantic_token_sha256"]
        != changed["components"]["relational_trunks"]["semantic_token_sha256"]
    )

    checkpoint = tmp_path / "topology.pt"
    checkpoint.write_bytes(b"semantic preflight does not load checkpoint tensors")
    payload = {
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: stamped,
        "config": {
            "fields": {
                "state_trunk": "transformer",
                "topology_residual_adapter": True,
            }
        },
    }
    with pytest.raises(RuntimeError, match="forward semantic mismatch"):
        assert_entity_graph_checkpoint_runtime_semantics(
            payload,
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
            relational_source=mutated_source,
        )


def test_v3_binds_dependency_closed_forward_surface() -> None:
    identity = current_entity_graph_forward_semantics(POLICY_SOURCE)

    assert identity["schema_version"] == "entity-graph-forward-semantics-v3"
    assert set(identity["components"]) == {
        "entity_token_policy",
        "relational_trunks",
        "action_features",
        "entity_token_features",
        "meaningful_history",
        "ordered_history",
        "deduction_tracker",
        "entity_feature_adapter",
    }
    assert "EntityGraphPolicy._legal_outputs_from_env" in identity["components"][
        "entity_token_policy"
    ]["selected_symbols"]
    assert "build_action_context_feature_table" in identity["components"][
        "action_features"
    ]["selected_symbols"]
    assert "mask_player_tokens_public" in identity["components"][
        "entity_token_features"
    ]["selected_symbols"]


@pytest.mark.parametrize(
    ("source_key", "old", "new", "component"),
    (
        (
            "action_features_source",
            '"ROLL": 40,',
            '"ROLL": 0,',
            "action_features",
        ),
        (
            "entity_token_features_source",
            "PLAYER_ACTOR_FLAG_SLOT = 1",
            "PLAYER_ACTOR_FLAG_SLOT = 2",
            "entity_token_features",
        ),
    ),
)
def test_feature_semantic_mutation_changes_v3_binding(
    tmp_path: Path,
    source_key: str,
    old: str,
    new: str,
    component: str,
) -> None:
    original_path = DEPENDENCY_SOURCES[source_key]
    original = original_path.read_text(encoding="utf-8")
    mutated = original.replace(old, new, 1)
    assert mutated != original
    mutated_path = tmp_path / original_path.name
    mutated_path.write_text(mutated, encoding="utf-8")

    stamped = current_entity_graph_forward_semantics(POLICY_SOURCE)
    changed = current_entity_graph_forward_semantics(
        POLICY_SOURCE,
        **{source_key: mutated_path},
    )

    assert stamped["semantic_token_sha256"] != changed["semantic_token_sha256"]
    assert (
        stamped["components"][component]["semantic_token_sha256"]
        != changed["components"][component]["semantic_token_sha256"]
    )


def test_policy_environment_bridge_mutation_changes_v3_binding(
    tmp_path: Path,
) -> None:
    original = POLICY_SOURCE.read_text(encoding="utf-8")
    old = """        context_table = build_action_context_feature_table(
            env,
            info,
            entity_feature_adapter_version=self.entity_feature_adapter_version,
        )"""
    new = """        context_table = build_action_context_feature_table(
            env,
            info,
            entity_feature_adapter_version=self.entity_feature_adapter_version,
        ) * 0.0"""
    mutated = original.replace(old, new, 1)
    assert mutated != original
    mutated_policy = tmp_path / POLICY_SOURCE.name
    mutated_policy.write_text(mutated, encoding="utf-8")

    stamped = current_entity_graph_forward_semantics(POLICY_SOURCE)
    changed = current_entity_graph_forward_semantics(
        mutated_policy,
        **DEPENDENCY_SOURCES,
    )

    assert (
        stamped["components"]["entity_token_policy"]["semantic_token_sha256"]
        != changed["components"]["entity_token_policy"]["semantic_token_sha256"]
    )


def test_v2_stamp_requires_authenticated_current_runtime_module_evidence(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "v2.pt"
    checkpoint.write_bytes(b"v2 semantic preflight")
    payload = _v2_checkpoint(runtime_binding=_authenticated_runtime_binding())

    result = assert_entity_graph_checkpoint_runtime_semantics(
        payload,
        checkpoint_path=checkpoint,
        policy_source=POLICY_SOURCE,
    )

    assert result["compatible"] is True
    assert result["provenance"] == "checkpoint_stamp_v2_authenticated_runtime_compat"


def test_v2_stamp_without_runtime_module_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "v2-no-runtime.pt"
    checkpoint.write_bytes(b"v2 semantic preflight")

    with pytest.raises(RuntimeError, match="does not authenticate"):
        assert_entity_graph_checkpoint_runtime_semantics(
            _v2_checkpoint(runtime_binding=None),
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
        )


def test_v2_authenticated_but_stale_feature_module_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "v2-stale-runtime.pt"
    checkpoint.write_bytes(b"v2 semantic preflight")
    payload = _v2_checkpoint(
        runtime_binding=_authenticated_runtime_binding(
            overrides={"catan_zero.rl.action_features": "sha256:" + "a" * 64}
        )
    )

    with pytest.raises(RuntimeError, match="action_features"):
        assert_entity_graph_checkpoint_runtime_semantics(
            payload,
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
        )


def test_v2_runtime_evidence_rejects_tampered_binding_digest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "v2-tampered-runtime.pt"
    checkpoint.write_bytes(b"v2 semantic preflight")
    binding = _authenticated_runtime_binding()
    binding["binding_sha256"] = "sha256:" + "f" * 64

    with pytest.raises(RuntimeError, match="unauthenticated"):
        assert_entity_graph_checkpoint_runtime_semantics(
            _v2_checkpoint(runtime_binding=binding),
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
        )


def test_legacy_single_file_stamp_is_refused_for_topology_checkpoint(
    tmp_path: Path,
) -> None:
    legacy = {
        "schema_version": "entity-graph-forward-semantics-v1",
        "policy_schema_version": "entity_graph_policy_v1",
        "semantic_token_sha256": (
            "sha256:460f78322abb3af4ba5255593b9ef3a4db93c53a114fdef15a9a8f1dae828f7e"
        ),
        "selected_symbols": [
            "_validate_public_award_feature_contract",
            "_apply_public_award_feature_contract",
            "_entity_token_start_offsets",
            "EntityGraphNet",
            "_token_encoder",
            "EntityGraphPolicy.__init__",
            "EntityGraphPolicy.forward_legal_np",
            "_assert_entity_batch_shapes",
        ],
    }
    legacy["binding_sha256"] = _canonical_sha256(legacy)
    checkpoint = tmp_path / "legacy-topology.pt"
    checkpoint.write_bytes(b"legacy topology stamp")

    compatible_payload = {
        ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: legacy,
        "config": {
            "fields": {
                "state_trunk": "transformer",
                "topology_residual_adapter": False,
            }
        },
    }
    accepted = assert_entity_graph_checkpoint_runtime_semantics(
        compatible_payload,
        checkpoint_path=checkpoint,
        policy_source=POLICY_SOURCE,
    )
    assert accepted["provenance"] == "checkpoint_stamp_v1_topology_disabled_compat"

    adapter_v6_payload = copy.deepcopy(compatible_payload)
    adapter_v6_payload["entity_feature_adapter"] = {
        "schema_version": "entity-feature-adapter-v1",
        "version": RUST_ENTITY_ADAPTER_V6,
    }
    with pytest.raises(RuntimeError, match="does not bind.*feature adapter"):
        assert_entity_graph_checkpoint_runtime_semantics(
            adapter_v6_payload,
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
        )

    with pytest.raises(RuntimeError, match="does not bind relational topology"):
        assert_entity_graph_checkpoint_runtime_semantics(
            {
                ENTITY_GRAPH_FORWARD_SEMANTICS_KEY: legacy,
                "config": {
                    "fields": {
                        "state_trunk": "transformer",
                        "topology_residual_adapter": True,
                    }
                },
            },
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
        )


def test_unreviewed_metadata_free_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unknown.pt"
    checkpoint.write_bytes(b"unknown legacy checkpoint")

    with pytest.raises(RuntimeError, match="no accepted.*forward semantic identity"):
        assert_entity_graph_checkpoint_runtime_semantics(
            {},
            checkpoint_path=checkpoint,
            policy_source=POLICY_SOURCE,
        )
