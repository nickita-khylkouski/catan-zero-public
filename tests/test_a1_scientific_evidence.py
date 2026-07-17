from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import a1_scientific_evidence as evidence
from tools import a1_aux_pair_coordinator as coordinator


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _category_semantics() -> dict:
    checkpoint = lambda name, digest, version: {  # noqa: E731
        "id": name,
        "path": f"/srv/checkpoints/{name}.pt",
        "sha256": digest,
        "version": version,
    }
    return {
        "current_producer": {
            "scheduler_category": "current_producer",
            "semantic": "current_producer",
            "relation": "self_play",
            "checkpoint": checkpoint("current", _sha("a"), 5),
        },
        "recent_history": {
            "scheduler_category": "recent_history",
            "semantic": "recovery_reference",
            "relation": "safety_reference_unproven_predecessor",
            "causal_parent_proven": False,
            "promotion_proof_recreated": False,
            "checkpoint": checkpoint("safety", _sha("b"), 3),
            "recovery_lineage_id": "recovery-v5-test",
        },
        "hard_negative": {
            "scheduler_category": "hard_negative",
            "semantic": "hard_negative",
            "relation": "sealed_hard_negative_selection",
            "checkpoint": checkpoint("hard", _sha("c"), 2),
        },
    }


class _Component:
    def __init__(self, rows: int, *, award: bool) -> None:
        self.row_count = rows
        self._values = np.zeros((rows, 4, 13), dtype=np.float32)
        if award:
            self._values[::3, 0, 12] = 1.0

    def __getitem__(self, key: str):
        assert key == "player_tokens"
        return self._values


class _Composite:
    def __init__(self, rows_per_component: int = 20) -> None:
        self.component_ids = evidence.COMPONENT_IDS
        self.corpora = tuple(
            _Component(
                rows_per_component,
                award=component_id != "historical_replay",
            )
            for component_id in self.component_ids
        )
        self.component_offsets = np.arange(5, dtype=np.int64) * rows_per_component
        self.row_count = rows_per_component * 4
        width = 3
        self._legal = np.tile(np.arange(width, dtype=np.int64), (self.row_count, 1))
        self._prior = np.full((self.row_count, width), 1.0 / width, dtype=np.float32)
        self._games = np.arange(self.row_count, dtype=np.int64) // 2
        self._actions = np.zeros(self.row_count, dtype=np.int64)

    def component_indices_for_rows(self, rows):
        return np.searchsorted(self.component_offsets[1:], rows, side="right")

    def __getitem__(self, key: str):
        return {
            "legal_action_ids": self._legal,
            "prior_policy": self._prior,
            "game_seed": self._games,
            "action_taken": self._actions,
        }[key]

    def __contains__(self, key: str) -> bool:
        return key in {"legal_action_ids", "prior_policy", "game_seed", "action_taken"}


def _authenticated(data: _Composite) -> dict:
    semantics = _category_semantics()
    return {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": False,
        "promotion_eligible": True,
        "descriptor_file_sha256": _sha("1"),
        "payload_inventory_sha256": _sha("2"),
        "category_semantics": semantics,
        "category_semantics_sha256": evidence._digest(semantics),
        "source_authority_ref": {
            "path": "/srv/composite/source_authority.json",
            "file_sha256": _sha("e"),
            "authority_sha256": _sha("f"),
        },
        "components": [
            {
                "component_id": component_id,
                "payload_inventory_sha256": _sha(str(index + 3)),
            }
            for index, component_id in enumerate(data.component_ids)
        ],
    }


def test_recovery_semantics_refuse_scheduler_lane_laundering() -> None:
    semantics = _category_semantics()
    assert evidence.verify_recovery_component_semantics(semantics) == semantics

    stripped = {key: dict(value) for key, value in semantics.items()}
    stripped["recent_history"].pop("causal_parent_proven")
    with pytest.raises(evidence.EvidenceError, match="relation drift"):
        evidence.verify_recovery_component_semantics(stripped)

    swapped = {key: dict(value) for key, value in semantics.items()}
    swapped["recent_history"] = dict(semantics["current_producer"])
    with pytest.raises(evidence.EvidenceError, match="checkpoint drift"):
        evidence.verify_recovery_component_semantics(swapped)


def test_runtime_receipt_binds_committed_producer_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "host_id": evidence.B200_LEARNER_HOST_ID,
        "tool_sha256": evidence.origin_tool_sha256(),
    }
    monkeypatch.setattr(evidence, "_local_runtime_report", lambda _root: report)
    receipt = evidence.build_runtime_admission_receipt()
    assert receipt["origin_tool_sha256"] == evidence.origin_tool_sha256()
    assert receipt["hosts"][evidence.B200_LEARNER_HOST_ID] == report

    monkeypatch.setattr(
        evidence,
        "_local_runtime_report",
        lambda _root: {**report, "tool_sha256": _sha("f")},
    )
    with pytest.raises(evidence.EvidenceError, match="does not bind"):
        evidence.build_runtime_admission_receipt()


def test_routing_receipt_measures_component_rows_and_slot12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _Composite()
    authenticated = _authenticated(data)
    monkeypatch.setattr(
        evidence,
        "_load_composite",
        lambda _descriptor: (tmp_path / "descriptor.json", authenticated, data),
    )
    monkeypatch.setattr(evidence, "_assert_composite_stable", lambda *_args: None)

    receipt = evidence.build_mixed_routing_receipt(tmp_path / "descriptor.json")
    assert receipt["component_row_counts"] == {
        component_id: 20 for component_id in data.component_ids
    }
    assert receipt["legacy_slot12_nonzero_count"] == 0
    assert receipt["legacy_slot12_all_zero"] is True
    assert receipt["origin_tool_sha256"] == evidence.origin_tool_sha256()
    routing_path = tmp_path / "routing.json"
    evidence._atomic_write(routing_path, receipt)
    assert evidence.verify_mixed_routing_receipt(
        routing_path,
        descriptor=tmp_path / "descriptor.json",
        expected_origin_tool_sha256=evidence.origin_tool_sha256(),
    ) == receipt

    data.corpora[-1]._values[0, 0, 12] = 1.0
    with pytest.raises(evidence.EvidenceError, match="legacy replay"):
        evidence.build_mixed_routing_receipt(tmp_path / "descriptor.json")


def test_sample_evidence_replays_order_and_measures_kl_and_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _Composite()
    authenticated = _authenticated(data)
    monkeypatch.setattr(evidence, "SHORT_SAMPLE_DOSE", 64)
    monkeypatch.setattr(coordinator, "SHORT_SAMPLE_DOSE", 64)
    monkeypatch.setattr(
        evidence,
        "_load_composite",
        lambda _descriptor: (tmp_path / "descriptor.json", authenticated, data),
    )
    monkeypatch.setattr(evidence, "_assert_composite_stable", lambda *_args: None)
    monkeypatch.setattr(
        evidence,
        "_training_indices",
        lambda _data: np.arange(data.row_count, dtype=np.int64),
    )
    monkeypatch.setattr(
        evidence.train_bc,
        "_composite_game_sampling_weights",
        lambda _data, indices: np.full(len(indices), 1.0 / len(indices)),
    )
    first_rows = tmp_path / "first.rows.jsonl"
    first = evidence.build_sample_evidence(
        tmp_path / "descriptor.json",
        sampler_seed=424242,
        sample_dose=64,
        rows_path=first_rows,
    )
    assert first["sample_dose"] == 64
    assert 0 < first["kl_eligible_rows"] < 64
    assert first["rows_file_sha256"] == evidence._file_sha256(first_rows)
    assert first["sample_order_sha256"].startswith("sha256:")

    second = evidence.build_sample_evidence(
        tmp_path / "descriptor.json",
        sampler_seed=424243,
        sample_dose=64,
        rows_path=tmp_path / "second.rows.jsonl",
        prior_rows_path=first_rows,
    )
    assert second["sampler_identity_sha256"] != first["sampler_identity_sha256"]
    assert second["sample_order_sha256"] != first["sample_order_sha256"]
    assert second["row_set_sha256"] != first["row_set_sha256"]
    assert second["prior_unique_row_count"] == first["unique_row_count"]
    assert second["prior_rows_file_sha256"] == evidence._file_sha256(first_rows)
    assert second["prior_row_set_sha256"] == first["row_set_sha256"]
    assert second["observed_unique_overlap_count"] > 0
    assert second["overlap_within_independent_bound"] is True
    second_path = tmp_path / "second.json"
    evidence._atomic_write(second_path, second)
    assert evidence.verify_sample_evidence(
        second_path,
        descriptor=tmp_path / "descriptor.json",
        rows_path=tmp_path / "second.rows.jsonl",
        prior_rows_path=first_rows,
        expected_origin_tool_sha256=evidence.origin_tool_sha256(),
    ) == second

    rows = [json.loads(line) for line in first_rows.read_text().splitlines()]
    composite = {
        "schema_version": "a1-typed-64-12-4-20-composite-v1",
        "component_ids": list(coordinator.COMPONENT_IDS),
        "component_sampling_ratios": list(coordinator.COMPONENT_RATIOS),
        "descriptor_sha256": authenticated["descriptor_file_sha256"],
        "data_fingerprint": _sha("8"),
            "payload_inventory_sha256": authenticated["payload_inventory_sha256"],
            "category_semantics": authenticated["category_semantics"],
            "category_semantics_sha256": authenticated[
                "category_semantics_sha256"
            ],
        "source_authority": authenticated["source_authority_ref"],
        "learner_recipe_overrides_sha256": _sha("e"),
        "aux_subgoal_target_contract_sha256": _sha("f"),
        "public_award_feature_transition_contract_sha256": _sha("0"),
        "source_authority_semantic_sha256": _sha("1"),
        "production_sampling_receipt_sha256": _sha("9"),
        "validation_split_receipt_sha256": _sha("a"),
        "sampler_identity_sha256": first["sampler_identity_sha256"],
        "sample_order_sha256": first["sample_order_sha256"],
        "training_game_seed_set_sha256": _sha("b"),
        "validation_game_seed_set_sha256": _sha("c"),
        "truncation_surface_sha256": _sha("d"),
        "truncated_rows": 0,
        "complete_game_inputs": True,
    }
    replay = coordinator.build_p1_kl_eligibility_authority(
        composite=composite,
        sampled_row_evidence=rows,
    )
    assert replay["eligible_rows"] == first["kl_eligible_rows"]
    assert replay["ordered_evidence_sha256"] == first[
        "kl_ordered_evidence_sha256"
    ]


def test_slot12_receipts_bind_pre_optimizer_zero_and_post_optimizer_delta(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "initializer.pt"
    torch.save(
        {
            "public_award_feature_contract": "authoritative_v1",
            "model": {"model.player_encoder.0.weight": torch.zeros((8, 13))},
        },
        checkpoint,
    )
    receipt = evidence.build_initializer_slot12_zero_receipt(checkpoint)
    assert receipt["initializer_slot12_max_abs_decimal"] == "0"
    assert receipt["model_slot12_parameter_count"] == 8
    assert receipt["initializer_checkpoint_sha256"] == evidence._file_sha256(
        checkpoint
    )
    receipt_path = tmp_path / "slot12.json"
    evidence._atomic_write(receipt_path, receipt)
    assert evidence.verify_initializer_slot12_zero_receipt(
        receipt_path,
        checkpoint=checkpoint,
        expected_origin_tool_sha256=evidence.origin_tool_sha256(),
    ) == receipt

    candidate = tmp_path / "candidate.pt"
    trained = torch.zeros((8, 13))
    trained[:, 12] = torch.arange(1, 9, dtype=torch.float32)
    torch.save(
        {
            "public_award_feature_contract": "authoritative_v1",
            "model": {"model.player_encoder.0.weight": trained},
        },
        candidate,
    )
    delta = evidence.build_trained_slot12_delta_receipt(checkpoint, candidate)
    assert delta["candidate_slot12_nonzero_count"] == 8
    assert delta["learned_signal_observed"] is True
    assert delta["candidate_slot12_finite"] is True
    delta_path = tmp_path / "delta.json"
    evidence._atomic_write(delta_path, delta)
    assert evidence.verify_trained_slot12_delta_receipt(
        delta_path,
        initializer_checkpoint=checkpoint,
        candidate_checkpoint=candidate,
        expected_origin_tool_sha256=evidence.origin_tool_sha256(),
    ) == delta

    torch.save(
        {
            "public_award_feature_contract": "authoritative_v1",
            "model": {"model.player_encoder.0.weight": torch.ones((8, 13))},
        },
        checkpoint,
    )
    with pytest.raises(evidence.EvidenceError, match="not exactly zero"):
        evidence.build_initializer_slot12_zero_receipt(checkpoint)


def test_public_award_initializer_transition_is_exact_and_replayable(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "legacy-parent.pt"
    transitioned = tmp_path / "authoritative-parent.pt"
    receipt_path = tmp_path / "transition.json"
    encoder = torch.arange(104, dtype=torch.float32).reshape(8, 13)
    trunk = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(
        {
            "public_award_feature_contract": "legacy_zero_v0",
            "model": {
                "model.player_encoder.0.weight": encoder.clone(),
                "model.trunk.weight": trunk.clone(),
            },
            "mask_hidden_info": True,
            "epoch": 7,
        },
        source,
    )
    receipt = evidence.build_public_award_transition_initializer(
        source, transitioned
    )
    evidence._atomic_write(receipt_path, receipt)
    replay = evidence.verify_public_award_transition_receipt(
        receipt_path,
        source_checkpoint=source,
        transitioned_checkpoint=transitioned,
        expected_origin_tool_sha256=evidence.origin_tool_sha256(),
        expected_source_checkpoint_sha256=receipt[
            "source_checkpoint_sha256"
        ],
        expected_transitioned_checkpoint_sha256=receipt[
            "transitioned_checkpoint_sha256"
        ],
    )
    assert replay["source_checkpoint_sha256"] == evidence._file_sha256(source)
    assert replay["transitioned_checkpoint_sha256"] == evidence._file_sha256(
        transitioned
    )
    assert replay["legacy_zero_input_function_preserving"] is True
    assert replay["unchanged_parameters_bit_identical"] is True
    raw = torch.load(transitioned, map_location="cpu", weights_only=False)
    assert raw["public_award_feature_contract"] == "authoritative_v1"
    assert torch.count_nonzero(
        raw["model"]["model.player_encoder.0.weight"][:, 12]
    ).item() == 0
    assert torch.equal(raw["model"]["model.trunk.weight"], trunk)


def test_authenticated_checkpoint_load_uses_hashed_descriptor_after_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint.pt"
    displaced = tmp_path / "checkpoint.authenticated.pt"
    torch.save({"identity": "authenticated"}, checkpoint)
    expected_sha256 = evidence._file_sha256(checkpoint)
    real_load = torch.load

    def replace_path_then_load(handle, *args, **kwargs):
        checkpoint.rename(displaced)
        torch.save({"identity": "replacement"}, checkpoint)
        return real_load(handle, *args, **kwargs)

    monkeypatch.setattr(torch, "load", replace_path_then_load)
    actual_sha256, payload = evidence._load_checkpoint_after_digest(
        checkpoint,
        expected_sha256=expected_sha256,
        where="test checkpoint",
    )

    assert actual_sha256 == expected_sha256
    assert payload == {"identity": "authenticated"}
    assert real_load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    ) == {"identity": "replacement"}


def test_public_award_initializer_transition_rejects_tampered_non_slot12_tensor(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "legacy-parent.pt"
    transitioned = tmp_path / "authoritative-parent.pt"
    torch.save(
        {
            "public_award_feature_contract": "legacy_zero_v0",
            "model": {
                "model.player_encoder.0.weight": torch.ones((8, 13)),
                "model.trunk.weight": torch.ones((3, 4)),
            },
        },
        source,
    )
    evidence.build_public_award_transition_initializer(source, transitioned)
    transitioned.chmod(0o644)
    raw = torch.load(transitioned, map_location="cpu", weights_only=False)
    raw["model"]["model.trunk.weight"][0, 0] += 1
    torch.save(raw, transitioned)
    with pytest.raises(
        evidence.EvidenceError, match="changed inherited parameter"
    ):
        evidence._public_award_transition_evidence(source, transitioned)
