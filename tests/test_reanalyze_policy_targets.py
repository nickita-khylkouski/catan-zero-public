from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from catan_zero.rl.entity_token_policy import EntityGraphConfig
from tools import reanalyze_policy_targets as target
from tools import train_bc


class _MaliciousCheckpointPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


def _write_entity_checkpoint(
    path: Path,
    *,
    action_size: int = 567,
    action_mask_version: str = "colonist-multiagent-v1",
    top_level_action_mask_version: str | None = None,
    meaningful_public_history: bool = True,
    history_schema: str = "meaningful_public_history_2p_no_trade_v2",
    event_history_limit: int = 64,
    adapter_version: str = (
        "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
    ),
) -> None:
    torch.save(
        {
            "policy_type": "entity_graph",
            "action_mask_version": (
                action_mask_version
                if top_level_action_mask_version is None
                else top_level_action_mask_version
            ),
            "config": {
                "__config_dataclass__": "EntityGraphConfig",
                "__config_schema__": 1,
                "fields": {
                    "action_size": action_size,
                    "action_mask_version": action_mask_version,
                    "meaningful_public_history": meaningful_public_history,
                    "meaningful_public_history_schema": history_schema,
                    "event_history_limit": event_history_limit,
                },
            },
            "entity_feature_adapter": {
                "schema_version": "entity-feature-adapter-v1",
                "version": adapter_version,
            },
        },
        path,
    )


def _write_source(
    tmp_path: Path, *, regime: str = target.TARGET_INFORMATION_REGIME_PUBLIC
):
    producer = tmp_path / "producer.pt"
    reanalyzer = tmp_path / "reanalyzer.pt"
    _write_entity_checkpoint(producer)
    _write_entity_checkpoint(reanalyzer)
    shard = tmp_path / "source.npz"
    arrays = {
        "obs": np.zeros((2, 1), dtype=np.float16),
        "action_taken": np.asarray([10, 11], dtype=np.int16),
        "game_seed": np.asarray([7, 7], dtype=np.int64),
        "decision_index": np.asarray([0, 1], dtype=np.int32),
        "phase": np.asarray(["A", "B"]),
        "player": np.asarray(["RED", "BLUE"]),
        "terminated": np.asarray([True, True]),
        "truncated": np.asarray([False, False]),
        "winner": np.asarray(["RED", "RED"]),
        "policy_weight_multiplier": np.asarray([1.0, 1.0], dtype=np.float32),
        "used_full_search": np.asarray([True, True]),
        "simulations_used": np.asarray([16, 16], dtype=np.int32),
        "is_forced": np.asarray([False, False]),
        "target_information_regime": np.asarray([regime, regime]),
        "legal_action_ids": np.asarray([[10, 12, -1], [11, 13, -1]], dtype=np.int16),
        "legal_action_context": np.zeros((2, 3, 1), dtype=np.float16),
        "target_policy": np.asarray(
            [[0.6, 0.4, 0.0], [0.7, 0.3, 0.0]], dtype=np.float32
        ),
        "target_policy_mask": np.asarray([[True, True, False], [True, True, False]]),
        "target_scores": np.asarray(
            [[0.2, 0.1, np.nan], [0.3, -0.1, np.nan]], dtype=np.float32
        ),
        "target_scores_mask": np.asarray([[True, True, False], [True, True, False]]),
        "root_value": np.asarray([0.2, 0.3], dtype=np.float32),
        "root_value_mask": np.asarray([True, True]),
        "root_prior_value": np.asarray([0.1, 0.15], dtype=np.float32),
        "root_prior_value_mask": np.asarray([True, True]),
        "search_evidence_version": np.asarray(2, dtype=np.uint8),
        "search_evidence_offsets": np.asarray([0, 2, 4], dtype=np.uint32),
        "search_visit_counts_flat": np.asarray([8, 8, 8, 8], dtype=np.uint16),
        "search_completed_q_flat": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "search_prior_policy_flat": np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "target_reliability_version": np.asarray([1, 1], dtype=np.uint8),
        "target_reliability_audited": np.asarray([True, True]),
        "target_reliability_js_divergence": np.asarray([0.1, 0.1], dtype=np.float32),
        "target_reliability_policy_top1_agreement": np.asarray([True, True]),
        "target_reliability_q_top1_agreement": np.asarray([True, True]),
        "target_reliability_q_margin_primary": np.asarray([0.2, 0.2], dtype=np.float32),
        "target_reliability_q_margin_duplicate": np.asarray(
            [0.2, 0.2], dtype=np.float32
        ),
        "target_reliability_confidence": np.asarray([0.5, 0.5], dtype=np.float32),
        "prior_policy": np.asarray(
            [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=np.float16
        ),
        "aux_vp_in_n": np.asarray([1.0, -1.0], dtype=np.float32),
        "is_pool_game": np.asarray([False, False]),
        "opponent_version": np.asarray([-1, -1], dtype=np.int32),
        "opponent_tag": np.asarray(["producer_self_play", "producer_self_play"]),
        "opponent_checkpoint_md5": np.asarray(["", ""]),
        "opponent_type": np.asarray(["", ""]),
        "teacher_name": np.asarray(["gumbel_self_play", "gumbel_self_play"]),
    }
    np.savez(shard, **arrays)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "producer_checkpoint_sha256": target._sha256(producer),
                "shards": [str(shard)],
            }
        )
    )
    auth_key = tmp_path / "claim.key"
    auth_key.write_bytes(b"k" * 32)
    return producer, reanalyzer, shard, manifest, arrays, auth_key


def _plan(
    tmp_path: Path,
    *,
    chunks: int = 1,
    regime: str = target.TARGET_INFORMATION_REGIME_PUBLIC,
):
    producer, reanalyzer, shard, manifest, arrays, auth_key = _write_source(
        tmp_path, regime=regime
    )
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    plan = target.build_plan(
        source_manifest=manifest,
        trajectory_producer_checkpoint=producer,
        target_checkpoint=reanalyzer,
        chunks=chunks,
        search_config=target.default_search_config(),
        claim_auth_key=auth_key,
        runtime_attestation=runtime,
    )
    return plan, producer, reanalyzer, shard, arrays, auth_key, runtime


def _patch(_search, _game, _feature, **_kwargs):
    return {
        "target_policy": [0.25, 0.75],
        "target_policy_mask": [True, True],
        "target_scores": [-0.2, 0.8],
        "target_scores_mask": [True, True],
        "root_value": 0.55,
        "root_value_mask": True,
        "root_prior_value": 0.35,
        "root_prior_value_mask": True,
        "prior_policy": [0.4, 0.6],
        "simulations_used": 32,
        "used_full_search": True,
    }


def _bypass_reconstruction(**_kwargs):
    return object(), {"legal_policy_ids": (10, 12)}


def _input_abi(*, action_size: int = 567, role: str = "target") -> dict:
    return target._seal_producer_input_abi(
        {
            "action_size": action_size,
            "action_mask_version": "colonist-multiagent-v1",
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                "meaningful_public_history_2p_no_trade_v2"
            ),
            "event_history_limit": 64,
            "entity_feature_adapter_version": (
                "rust_entity_adapter_v6_exact_actor_resources_"
                "initial_road_two_hop"
            ),
        },
        checkpoint_sha256="sha256:" + "a" * 64,
        binding_source=f"{role}_checkpoint",
    )


def _current_manifest_input_abi(checkpoint_sha256: str) -> dict:
    return target._seal_producer_input_abi(
        {
            "action_size": 567,
            "action_mask_version": "colonist-multiagent-v1",
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                "meaningful_public_history_2p_no_trade_v2"
            ),
            "event_history_limit": 64,
            "entity_feature_adapter_version": (
                "rust_entity_adapter_v6_exact_actor_resources_"
                "initial_road_two_hop"
            ),
        },
        checkpoint_sha256=checkpoint_sha256,
        binding_source="source_manifest_explicit_legacy_contract",
    )


def test_plan_seals_current_producer_input_abi(tmp_path: Path) -> None:
    plan, *_rest = _plan(tmp_path)
    abi = plan["trajectory_producer"]["input_abi"]
    assert abi["schema_version"] == target.PRODUCER_INPUT_ABI_SCHEMA
    assert abi["binding_source"] == "trajectory_producer_checkpoint"
    assert abi["action_size"] == 567
    assert abi["action_mask_version"] == "colonist-multiagent-v1"
    assert abi["meaningful_public_history"] is True
    assert (
        abi["meaningful_public_history_schema"]
        == "meaningful_public_history_2p_no_trade_v2"
    )
    assert abi["event_history_limit"] == 64
    assert abi["entity_feature_adapter_version"].startswith(
        "rust_entity_adapter_v6_"
    )
    assert abi["input_abi_sha256"] == target._value_sha256(
        {key: value for key, value in abi.items() if key != "input_abi_sha256"}
    )
    target_abi = plan["target_reanalyzer"]["input_abi"]
    assert target_abi["binding_source"] == "target_reanalyzer_checkpoint"
    assert target_abi["action_size"] == 567
    assert target_abi["action_mask_version"] == "colonist-multiagent-v1"


def test_plan_rejects_mismatched_producer_and_target_policy_catalogs(
    tmp_path: Path,
) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    _write_entity_checkpoint(reanalyzer, action_size=332)
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="policy catalogs are incompatible"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )


def test_plan_rejects_mismatched_action_mask_versions(tmp_path: Path) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    _write_entity_checkpoint(
        reanalyzer, action_mask_version="catanatron-flat-v1"
    )
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="policy catalogs are incompatible"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )


def test_plan_rejects_same_unknown_action_mask_version(tmp_path: Path) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    unknown_version = "matching-but-unknown-policy-catalog-v99"
    _write_entity_checkpoint(producer, action_mask_version=unknown_version)
    _write_entity_checkpoint(reanalyzer, action_mask_version=unknown_version)
    payload = json.loads(manifest.read_text())
    payload["producer_checkpoint_sha256"] = target._sha256(producer)
    manifest.write_text(json.dumps(payload))
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="unsupported action_mask_version"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )


def test_enabled_history_limit_above_schema_cap_is_rejected(tmp_path: Path) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    _write_entity_checkpoint(producer, event_history_limit=999)
    payload = json.loads(manifest.read_text())
    payload["producer_checkpoint_sha256"] = target._sha256(producer)
    manifest.write_text(json.dumps(payload))
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="exceeds its enabled schema cap"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )


@pytest.mark.parametrize(
    ("history_schema", "adapter_version"),
    [
        (
            "meaningful_public_history_2p_no_trade_v2",
            "rust_entity_adapter_v4_actor_public_rule_state",
        ),
        (
            "meaningful_public_history_2p_no_trade_v1",
            (
                "rust_entity_adapter_v6_exact_actor_resources_"
                "initial_road_two_hop"
            ),
        ),
    ],
)
def test_enabled_history_schema_must_match_adapter_generation(
    history_schema: str, adapter_version: str
) -> None:
    with pytest.raises(target.ReanalysisError, match="schema/adapter mismatch"):
        target._seal_producer_input_abi(
            {
                "action_size": 567,
                "action_mask_version": "colonist-multiagent-v1",
                "meaningful_public_history": True,
                "meaningful_public_history_schema": history_schema,
                "event_history_limit": (
                    64 if history_schema.endswith("_v2") else 32
                ),
                "entity_feature_adapter_version": adapter_version,
            },
            checkpoint_sha256="sha256:" + "a" * 64,
            binding_source="trajectory_producer_checkpoint",
        )


def test_modern_checkpoint_conflict_cannot_use_manifest_fallback(
    tmp_path: Path,
) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    _write_entity_checkpoint(
        producer,
        top_level_action_mask_version="conflicting-top-level-mask-v2",
    )
    producer_sha = target._sha256(producer)
    payload = json.loads(manifest.read_text())
    payload["producer_checkpoint_sha256"] = producer_sha
    payload["producer_input_abi"] = _current_manifest_input_abi(producer_sha)
    manifest.write_text(json.dumps(payload))
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="conflicts"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )


def test_modern_invalid_adapter_cannot_use_manifest_fallback(tmp_path: Path) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    _write_entity_checkpoint(producer, adapter_version="invalid-adapter")
    producer_sha = target._sha256(producer)
    payload = json.loads(manifest.read_text())
    payload["producer_checkpoint_sha256"] = producer_sha
    payload["producer_input_abi"] = _current_manifest_input_abi(producer_sha)
    manifest.write_text(json.dumps(payload))
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="invalid entity feature adapter"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )


def test_unsupported_pickle_uses_manifest_without_executing_payload(
    tmp_path: Path,
) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    marker = tmp_path / "unsafe-load-executed"
    torch.save({"payload": _MaliciousCheckpointPayload(marker)}, producer)
    producer_sha = target._sha256(producer)
    payload = json.loads(manifest.read_text())
    payload["producer_checkpoint_sha256"] = producer_sha
    manifest.write_text(json.dumps(payload))
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="cannot authenticate"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )
    assert not marker.exists()
    payload["producer_input_abi"] = _current_manifest_input_abi(producer_sha)
    manifest.write_text(json.dumps(payload))
    plan = target.build_plan(
        source_manifest=manifest,
        trajectory_producer_checkpoint=producer,
        target_checkpoint=reanalyzer,
        chunks=1,
        search_config=target.default_search_config(),
        claim_auth_key=auth_key,
        runtime_attestation=runtime,
    )
    assert not marker.exists()
    assert (
        plan["trajectory_producer"]["input_abi"]["binding_source"]
        == "source_manifest_explicit_legacy_contract"
    )


def test_reconstruction_uses_sealed_producer_input_abi(monkeypatch) -> None:
    shard = {
        key: np.zeros((1,), dtype=np.float32)
        for key in target.RECONSTRUCTION_COLUMNS
    }
    shard["decision_index"] = np.asarray([0], dtype=np.int32)
    shard["legal_action_ids"] = np.asarray([[10, -1]], dtype=np.int16)
    sequence = target.GameActionSequence(
        7, target.COLORS, [10], [0], ["A"], ["RED"]
    )
    abi = target._seal_producer_input_abi(
        {
            "action_size": 567,
            "action_mask_version": "colonist-multiagent-v1",
            "meaningful_public_history": True,
            "meaningful_public_history_schema": (
                "meaningful_public_history_2p_no_trade_v2"
            ),
            "event_history_limit": 64,
            "entity_feature_adapter_version": (
                "rust_entity_adapter_v6_exact_actor_resources_"
                "initial_road_two_hop"
            ),
        },
        checkpoint_sha256="sha256:" + "a" * 64,
        binding_source="trajectory_producer_checkpoint",
    )
    captured: dict[str, dict] = {}

    def fake_round_trip(*_args, **kwargs):
        captured["round_trip"] = kwargs
        return SimpleNamespace(
            ok=True,
            legal_ids_match=True,
            worst_key="",
            max_abs_diff=0.0,
        )

    def fake_reconstruct(*_args, **kwargs):
        captured["reconstruct"] = kwargs
        return object()

    def fake_featurize(_game, **kwargs):
        captured["featurize"] = kwargs
        return {"legal_policy_ids": (10,)}

    monkeypatch.setattr(target, "round_trip_row", fake_round_trip)
    monkeypatch.setattr(target, "reconstruct_state", fake_reconstruct)
    monkeypatch.setattr(target, "featurize_state", fake_featurize)
    _game, feature = target._verify_reconstruction(
        shard=shard,
        row=0,
        sequence=sequence,
        producer_input_abi=abi,
    )
    assert feature == {"legal_policy_ids": (10,)}
    for call in ("round_trip", "featurize"):
        assert captured[call]["action_size"] == 567
        assert captured[call]["meaningful_public_history"] is True
        assert (
            captured[call]["meaningful_public_history_schema"]
            == "meaningful_public_history_2p_no_trade_v2"
        )
        assert captured[call]["history_limit"] == 64
        assert captured[call]["entity_feature_adapter_version"].startswith(
            "rust_entity_adapter_v6_"
        )
    assert captured["reconstruct"]["action_size"] == 567
    assert captured["reconstruct"]["decision_indices"] == [0]


def test_legacy_checkpoint_without_explicit_manifest_abi_fails_closed(
    tmp_path: Path,
) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    producer.write_bytes(b"legacy-checkpoint-without-input-contract")
    payload = json.loads(manifest.read_text())
    payload["producer_checkpoint_sha256"] = target._sha256(producer)
    manifest.write_text(json.dumps(payload))
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="cannot authenticate"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )


def test_legacy_manifest_cannot_override_explicit_checkpoint_action_size(
    tmp_path: Path,
) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    torch.save(
        {
            "policy_type": "entity_graph",
            "action_mask_version": "colonist-multiagent-v1",
            "config": EntityGraphConfig(
                action_size=567,
                static_action_feature_size=45,
                action_mask_version="colonist-multiagent-v1",
                meaningful_public_history=False,
                meaningful_public_history_schema=(
                    "meaningful_public_history_2p_no_trade_v1"
                ),
                event_history_limit=64,
            ),
        },
        producer,
    )
    _write_entity_checkpoint(reanalyzer, action_size=332)
    producer_sha = target._sha256(producer)
    payload = json.loads(manifest.read_text())
    payload["producer_checkpoint_sha256"] = producer_sha
    payload["producer_input_abi"] = target._seal_producer_input_abi(
        {
            "action_size": 332,
            "action_mask_version": "colonist-multiagent-v1",
            "meaningful_public_history": False,
            "meaningful_public_history_schema": (
                "meaningful_public_history_2p_no_trade_v1"
            ),
            "event_history_limit": 64,
            "entity_feature_adapter_version": (
                "rust_entity_adapter_v2_land_topology_ports_maritime"
            ),
        },
        checkpoint_sha256=producer_sha,
        binding_source="source_manifest_explicit_legacy_contract",
    )
    manifest.write_text(json.dumps(payload))
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    with pytest.raises(target.ReanalysisError, match="conflicts.*action_size"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation=runtime,
        )


def test_legacy_checkpoint_requires_checkpoint_bound_hashed_manifest_abi(
    tmp_path: Path,
) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    producer.write_bytes(b"legacy-checkpoint")
    _write_entity_checkpoint(reanalyzer, action_size=332)
    producer_sha = target._sha256(producer)
    payload = json.loads(manifest.read_text())
    payload["producer_checkpoint_sha256"] = producer_sha
    payload["producer_input_abi"] = target._seal_producer_input_abi(
        {
            "action_size": 332,
            "action_mask_version": "colonist-multiagent-v1",
            "meaningful_public_history": False,
            "meaningful_public_history_schema": (
                "meaningful_public_history_2p_no_trade_v1"
            ),
            "event_history_limit": 64,
            "entity_feature_adapter_version": (
                "rust_entity_adapter_v2_land_topology_ports_maritime"
            ),
        },
        checkpoint_sha256=producer_sha,
        binding_source="source_manifest_explicit_legacy_contract",
    )
    manifest.write_text(json.dumps(payload))
    runtime = {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
    runtime["runtime_sha256"] = target._value_sha256(runtime)
    plan = target.build_plan(
        source_manifest=manifest,
        trajectory_producer_checkpoint=producer,
        target_checkpoint=reanalyzer,
        chunks=1,
        search_config=target.default_search_config(),
        claim_auth_key=auth_key,
        runtime_attestation=runtime,
    )
    assert plan["trajectory_producer"]["input_abi"]["action_size"] == 332
    assert (
        plan["trajectory_producer"]["input_abi"]["binding_source"]
        == "source_manifest_explicit_legacy_contract"
    )


def test_search_patch_preserves_zero_mass_coverage_and_pairs_root_prior(
    monkeypatch,
) -> None:
    game = SimpleNamespace(
        playable_action_indices=lambda _colors, _map_kind: [101, 102]
    )
    result = SimpleNamespace(
        improved_policy={101: 1.0, 102: 0.0},
        q_values={101: 0.4, 102: 0.1},
        priors={101: 0.6, 102: 0.4},
        used_full_search=True,
        root_value=0.3,
        root_prior_value=0.2,
        simulations_used=16,
    )
    captured = {}

    def mapped(*_args, **kwargs):
        captured.update(kwargs)
        return (10, 12)

    monkeypatch.setattr(target, "rust_policy_action_ids", mapped)
    patch = target._search_patch(  # noqa: SLF001
        SimpleNamespace(search=lambda _game, force_full: result),
        game,
        {"legal_policy_ids": (10, 12)},
        target_input_abi=_input_abi(action_size=567),
    )
    assert captured["action_size"] == 567
    assert patch["target_policy_mask"] == [True, True]
    assert patch["root_value"] == pytest.approx(0.3)
    assert patch["root_prior_value"] == pytest.approx(0.2)


@pytest.mark.parametrize("root_prior_value", [np.nan, np.inf, -1.01, 1.01])
def test_search_patch_rejects_invalid_root_prior(
    monkeypatch, root_prior_value: float
) -> None:
    game = SimpleNamespace(playable_action_indices=lambda _colors, _map_kind: [101])
    result = SimpleNamespace(
        improved_policy={101: 1.0},
        q_values={101: 0.4},
        priors={101: 1.0},
        used_full_search=True,
        root_value=0.3,
        root_prior_value=root_prior_value,
        simulations_used=16,
    )
    monkeypatch.setattr(
        target,
        "rust_policy_action_ids",
        lambda *_args, **_kwargs: (10,),
    )
    with pytest.raises(target.ReanalysisError, match="root search/prior"):
        target._search_patch(  # noqa: SLF001
            SimpleNamespace(search=lambda _game, force_full: result),
            game,
            {"legal_policy_ids": (10,)},
            target_input_abi=_input_abi(action_size=567),
        )


def test_reconstruction_mismatch_stops_before_search(tmp_path: Path) -> None:
    plan_value, _producer, _reanalyzer, shard_path, _arrays, _key, _runtime = _plan(
        tmp_path
    )
    shard = target.load_shard(shard_path)
    sequence = target.GameActionSequence(
        7, target.COLORS, [10, 11], [0, 1], ["A", "B"], ["RED", "BLUE"]
    )
    with pytest.raises(
        target.ReanalysisError, match="complete public reconstruction surface"
    ):
        target._verify_reconstruction(
            shard=shard,
            row=0,
            sequence=sequence,
            producer_input_abi=plan_value["trajectory_producer"]["input_abi"],
        )


def test_hidden_information_targets_are_not_admitted(tmp_path: Path) -> None:
    producer, reanalyzer, _shard, manifest, _arrays, auth_key = _write_source(
        tmp_path, regime="authoritative_hidden_state_search_v1"
    )
    with pytest.raises(target.ReanalysisError, match="no authenticated policy-active"):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation={
                "repo_commit": "test",
                "source_files": [],
                "catanatron_rs": {},
                "runtime_sha256": target._value_sha256(
                    {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
                ),
            },
        )


def test_missing_mirror_provenance_is_not_admitted(tmp_path: Path) -> None:
    producer, reanalyzer, shard, manifest, _arrays, auth_key = _write_source(tmp_path)
    arrays = target.load_shard(shard)
    del arrays["opponent_tag"]
    np.savez(shard, **arrays)
    with pytest.raises(
        target.ReanalysisError, match="explicit producer-mirror provenance"
    ):
        target.build_plan(
            source_manifest=manifest,
            trajectory_producer_checkpoint=producer,
            target_checkpoint=reanalyzer,
            chunks=1,
            search_config=target.default_search_config(),
            claim_auth_key=auth_key,
            runtime_attestation={
                "repo_commit": "test",
                "source_files": [],
                "catanatron_rs": {},
                "runtime_sha256": target._value_sha256(
                    {"repo_commit": "test", "source_files": [], "catanatron_rs": {}}
                ),
            },
        )


def test_merge_changes_only_search_target_columns(monkeypatch, tmp_path: Path) -> None:
    plan, _producer, _reanalyzer, shard_path, _arrays, auth_key, runtime = _plan(
        tmp_path
    )
    monkeypatch.setattr(target, "_runtime_attestation", lambda: runtime)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    claim_path = tmp_path / "claim.json"
    target.run_chunk(
        plan=plan,
        chunk_index=0,
        output=claim_path,
        claim_auth_key=auth_key,
        search_factory=lambda _seed: object(),
    )
    output = tmp_path / "merged"
    manifest = target.merge_claims(
        plan=plan, claim_paths=[claim_path], output=output, claim_auth_key=auth_key
    )
    original = target.load_shard(shard_path)
    rebuilt = target.load_shard(output / manifest["shards"][0])
    assert manifest["rewritten_columns"] == sorted(target.REWRITTEN_COLUMNS)
    for key in original:
        if key not in target.REWRITTEN_COLUMNS:
            assert target._array_equal(original[key], rebuilt[key]), key
    assert np.allclose(rebuilt["target_policy"][:, :2], [[0.25, 0.75], [0.25, 0.75]])
    assert np.allclose(rebuilt["root_value"], 0.55)
    assert np.all(rebuilt["root_value_mask"])
    assert np.allclose(rebuilt["root_prior_value"], 0.35)
    assert np.all(rebuilt["root_prior_value_mask"])
    assert not any(
        key.startswith("search_") and key != "search_seed" for key in rebuilt
    )
    assert manifest["search_evidence_invalidated"] is True
    assert not np.any(rebuilt["target_reliability_audited"])
    assert np.allclose(rebuilt["target_reliability_confidence"], 1.0)
    assert set(rebuilt["teacher_name"].astype(str)) == {"policy_target_reanalysis"}
    assert manifest["payload_inventory_sha256"] == target._value_sha256(
        manifest["payload_inventory"]
    )


def test_chunk_rerun_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    plan, _producer, _reanalyzer, _shard, _arrays, auth_key, runtime = _plan(tmp_path)
    monkeypatch.setattr(target, "_runtime_attestation", lambda: runtime)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    first = target.run_chunk(
        plan=plan,
        chunk_index=0,
        output=tmp_path / "first.json",
        claim_auth_key=auth_key,
        search_factory=lambda _seed: object(),
    )
    second = target.run_chunk(
        plan=plan,
        chunk_index=0,
        output=tmp_path / "second.json",
        claim_auth_key=auth_key,
        search_factory=lambda _seed: object(),
    )
    assert first == second
    assert (tmp_path / "first.json").read_bytes() == (
        tmp_path / "second.json"
    ).read_bytes()

    first_out = tmp_path / "merge-first"
    second_out = tmp_path / "merge-second"
    first_merge = target.merge_claims(
        plan=plan,
        claim_paths=[tmp_path / "first.json"],
        output=first_out,
        claim_auth_key=auth_key,
    )
    second_merge = target.merge_claims(
        plan=plan,
        claim_paths=[tmp_path / "second.json"],
        output=second_out,
        claim_auth_key=auth_key,
    )
    assert first_merge["payload_inventory"] == second_merge["payload_inventory"]
    assert (first_out / first_merge["shards"][0]).read_bytes() == (
        second_out / second_merge["shards"][0]
    ).read_bytes()


def test_checkpoint_swap_invalidates_plan(monkeypatch, tmp_path: Path) -> None:
    plan, _producer, reanalyzer, _shard, _arrays, _key, runtime = _plan(tmp_path)
    monkeypatch.setattr(target, "_runtime_attestation", lambda: runtime)
    reanalyzer.write_bytes(b"swapped")
    with pytest.raises(
        target.ReanalysisError, match="target_reanalyzer checkpoint hash drift"
    ):
        target._verify_plan(plan)


def test_merge_refuses_incomplete_chunk_set(monkeypatch, tmp_path: Path) -> None:
    plan, _producer, _reanalyzer, _shard, _arrays, auth_key, runtime = _plan(
        tmp_path, chunks=2
    )
    monkeypatch.setattr(target, "_runtime_attestation", lambda: runtime)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    first = tmp_path / "claim0.json"
    target.run_chunk(
        plan=plan,
        chunk_index=0,
        output=first,
        claim_auth_key=auth_key,
        search_factory=lambda _seed: object(),
    )
    with pytest.raises(target.ReanalysisError, match="incomplete claims"):
        target.merge_claims(
            plan=plan,
            claim_paths=[first],
            output=tmp_path / "merged",
            claim_auth_key=auth_key,
        )


def test_fabricated_recomputed_patch_fails_hmac(monkeypatch, tmp_path: Path) -> None:
    plan, _p, _r, _s, _a, auth_key, runtime = _plan(tmp_path)
    monkeypatch.setattr(target, "_runtime_attestation", lambda: runtime)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    claim_path = tmp_path / "claim.json"
    target.run_chunk(
        plan=plan,
        chunk_index=0,
        output=claim_path,
        claim_auth_key=auth_key,
        search_factory=lambda _seed: object(),
    )
    claim = json.loads(claim_path.read_text())
    claim["patches"][0]["values"]["root_value"] = -0.99
    claim["patches_sha256"] = target._value_sha256(claim["patches"])
    claim["claim_sha256"] = target._value_sha256(
        {
            key: value
            for key, value in claim.items()
            if key not in {"claim_sha256", "claim_hmac_sha256"}
        }
    )
    claim_path.write_text(json.dumps(claim))
    with pytest.raises(target.ReanalysisError, match="authentication failed"):
        target.merge_claims(
            plan=plan,
            claim_paths=[claim_path],
            output=tmp_path / "forged",
            claim_auth_key=auth_key,
        )


def test_action_tamper_under_stale_plan_is_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    plan, _p, _r, shard, _a, auth_key, runtime = _plan(tmp_path)
    monkeypatch.setattr(target, "_runtime_attestation", lambda: runtime)
    arrays = target.load_shard(shard)
    arrays["action_taken"][0] = 99
    np.savez(shard, **arrays)
    with pytest.raises(target.ReanalysisError, match="source shard hash drift"):
        target.run_chunk(
            plan=plan,
            chunk_index=0,
            output=tmp_path / "claim.json",
            claim_auth_key=auth_key,
            search_factory=lambda _seed: object(),
        )


def test_row_search_seed_is_chunk_count_invariant(monkeypatch, tmp_path: Path) -> None:
    one, producer, reanalyzer, _shard, _arrays, auth_key, runtime = _plan(
        tmp_path, chunks=1
    )
    two = target.build_plan(
        source_manifest=Path(one["source_manifest"]["path"]),
        trajectory_producer_checkpoint=producer,
        target_checkpoint=reanalyzer,
        chunks=2,
        search_config=target.default_search_config(),
        claim_auth_key=auth_key,
        runtime_attestation=runtime,
    )
    monkeypatch.setattr(target, "_runtime_attestation", lambda: runtime)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    target.run_chunk(
        plan=one,
        chunk_index=0,
        output=tmp_path / "one.json",
        claim_auth_key=auth_key,
        search_factory=lambda _seed: object(),
    )
    for chunk in range(2):
        target.run_chunk(
            plan=two,
            chunk_index=chunk,
            output=tmp_path / f"two-{chunk}.json",
            claim_auth_key=auth_key,
            search_factory=lambda _seed: object(),
        )
    one_claim = json.loads((tmp_path / "one.json").read_text())
    two_patches = [
        patch
        for chunk in range(2)
        for patch in json.loads((tmp_path / f"two-{chunk}.json").read_text())["patches"]
    ]
    assert {
        patch["identity_sha256"]: patch["search_seed"] for patch in one_claim["patches"]
    } == {patch["identity_sha256"]: patch["search_seed"] for patch in two_patches}


def test_train_loader_authenticates_provenance_and_masks(
    monkeypatch, tmp_path: Path
) -> None:
    plan, _p, _r, _s, _a, auth_key, runtime = _plan(tmp_path)
    monkeypatch.setattr(target, "_runtime_attestation", lambda: runtime)
    monkeypatch.setattr(target, "_verify_reconstruction", _bypass_reconstruction)
    monkeypatch.setattr(target, "_search_patch", _patch)
    claim = tmp_path / "claim.json"
    target.run_chunk(
        plan=plan,
        chunk_index=0,
        output=claim,
        claim_auth_key=auth_key,
        search_factory=lambda _seed: object(),
    )
    output = tmp_path / "merged"
    manifest = target.merge_claims(
        plan=plan,
        claim_paths=[claim],
        output=output,
        claim_auth_key=auth_key,
    )
    verified = train_bc._validate_policy_target_reanalysis_manifest(
        output / "manifest.json"
    )
    assert verified is not None
    assert verified["trajectory_producer"] == manifest["trajectory_producer"]
    assert verified["target_reanalyzer"] == manifest["target_reanalyzer"]
    assert verified["search_config_sha256"] == manifest["search_config_sha256"]
    assert train_bc._manifest_shard_files(output / "manifest.json") == [
        output / manifest["shards"][0]
    ]
    loaded = train_bc.load_teacher_data(output)
    assert np.all(np.asarray(loaded["root_value_mask"], dtype=bool))
    assert np.all(np.asarray(loaded["target_policy_mask"])[:, :2])

    shard_path = output / manifest["shards"][0]
    arrays = target.load_shard(shard_path)
    arrays["root_value_mask"][0] = False
    np.savez(shard_path, **arrays)
    with pytest.raises(SystemExit, match="output shard hash/size mismatch"):
        train_bc._validate_policy_target_reanalysis_manifest(output / "manifest.json")
