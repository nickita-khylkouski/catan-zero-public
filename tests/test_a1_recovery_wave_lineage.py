from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
)

from tools import a1_build_post_wave_composite as composite
from tools import a1_pre_wave_contract as contract
from tools import search_operator_binding as binding
from tools import train_bc


def _sha(path: Path) -> str:
    return binding._sha256(path)  # noqa: SLF001


def _recovery_semantics(tmp_path: Path) -> tuple[dict, dict, Path, Path, Path]:
    producer = tmp_path / "recovered-v5.pt"
    history = tmp_path / "f7.pt"
    hard = tmp_path / "hard.pt"
    producer.write_bytes(b"recovered")
    history.write_bytes(b"f7")
    hard.write_bytes(b"hard")
    receipt = tmp_path / "recovery.json"
    receipt.write_text("{}\n", encoding="utf-8")
    records = [
        {
            "id": "recovered-v5",
            "role": "producer",
            "path": str(producer),
            "sha256": _sha(producer),
            "version": 5,
            "md5": "producer-md5",
        },
        {
            "id": "f7",
            "role": "history",
            "path": str(history),
            "sha256": _sha(history),
            "version": 4,
            "md5": "history-md5",
            "lineage": {
                "relation": contract.RECOVERY_REFERENCE_RELATION,
                "semantic": contract.RECOVERY_REFERENCE_SEMANTIC,
                "causal_parent_proven": False,
                "promotion_proof_recreated": False,
            },
        },
        {
            "id": "hard",
            "role": "hard_negative",
            "path": str(hard),
            "sha256": _sha(hard),
            "version": 5,
            "md5": "hard-md5",
        },
    ]
    handoff = {
        "mode": contract.DISASTER_RECOVERY_HANDOFF_MODE,
        "path": str(receipt),
        "sha256": _sha(receipt),
        "recovery_receipt_sha256": "sha256:" + "a" * 64,
        "recovery_lineage_id": "recovery-v5-test",
    }
    semantics = contract._category_semantics(records, handoff)  # noqa: SLF001
    return semantics, handoff, producer, history, hard


def _minimal_lock(tmp_path: Path) -> dict:
    semantics, _handoff, producer, history, hard = _recovery_semantics(tmp_path)
    return {
        "contract_sha256": "sha256:" + "c" * 64,
        "category_semantics": semantics,
        "checkpoints": [
            {
                "id": "recovered-v5",
                "role": "producer",
                "path": str(producer),
                "sha256": _sha(producer),
                "version": 5,
                "md5": "producer-md5",
            },
            {
                "id": "f7",
                "role": "history",
                "path": str(history),
                "sha256": _sha(history),
                "version": 4,
                "md5": "history-md5",
            },
            {
                "id": "hard",
                "role": "hard_negative",
                "path": str(hard),
                "sha256": _sha(hard),
                "version": 5,
                "md5": "hard-md5",
            },
        ],
        "source_categories": [
            {"name": "current_producer", "checkpoint_ids": ["recovered-v5"]},
            {"name": "recent_history", "checkpoint_ids": ["f7"]},
            {"name": "hard_negative", "checkpoint_ids": ["hard"]},
        ],
        "fleet": {
            "jobs": [
                {
                    "job_id": "recent-job",
                    "worker_id": "worker-0",
                    "category": "recent_history",
                    "base_seed": 100,
                    "seed_end": 102,
                }
            ]
        },
    }


def test_recovery_scheduler_lane_cannot_claim_displaced_incumbent(
    tmp_path: Path,
) -> None:
    semantics, handoff, _producer, history, _hard = _recovery_semantics(tmp_path)
    recent = semantics["recent_history"]
    assert recent == {
        "scheduler_category": "recent_history",
        "semantic": "recovery_reference",
        "relation": "safety_reference_unproven_predecessor",
        "causal_parent_proven": False,
        "promotion_proof_recreated": False,
        "checkpoint": {
            "id": "f7",
            "path": str(history),
            "sha256": _sha(history),
            "version": 4,
        },
        "recovery_lineage_id": handoff["recovery_lineage_id"],
    }
    assert "promotion_receipt" not in recent


def test_mix_and_selected_game_require_exact_recovery_semantic(
    tmp_path: Path,
) -> None:
    lock = _minimal_lock(tmp_path)
    expected = lock["category_semantics"]["recent_history"]
    rendered = contract._render_mix_manifest(lock, "recent_history")  # noqa: SLF001
    assert rendered["_a1_contract"]["category_semantic"] == expected

    base_record = {
        "game_seed": 100,
        "job_id": "recent-job",
        "worker_id": "worker-0",
        "category": "recent_history",
        "producer_checkpoint_sha256": contract._producer(lock)["sha256"],  # noqa: SLF001
        "opponent_checkpoint_sha256": contract._category_opponent_sha256(  # noqa: SLF001
            lock, "recent_history"
        ),
    }
    with pytest.raises(composite.CompositeBuildError, match="sealed job/category"):
        composite._selection_by_job(  # noqa: SLF001
            lock, {"records": [base_record]}, expected_games={"recent_history": 1}
        )
    laundered = {
        **base_record,
        "category_semantic": {
            **expected,
            "semantic": "recent_history",
            "relation": "immediate_displaced_incumbent",
            "causal_parent_proven": True,
        },
    }
    with pytest.raises(composite.CompositeBuildError, match="sealed job/category"):
        composite._selection_by_job(  # noqa: SLF001
            lock, {"records": [laundered]}, expected_games={"recent_history": 1}
        )
    accepted = {**base_record, "category_semantic": expected}
    selected, owners, normalized = composite._selection_by_job(  # noqa: SLF001
        lock, {"records": [accepted]}, expected_games={"recent_history": 1}
    )
    assert selected == {"recent-job": {100}}
    assert owners == {100: ("recent-job", "recent_history")}
    assert normalized == [accepted]


def test_trainer_derives_selected_semantics_from_authenticated_records(
    tmp_path: Path,
) -> None:
    lock = _minimal_lock(tmp_path)
    semantics = lock["category_semantics"]
    selected = {
        "records": [
            {
                "category": category,
                "category_semantic": semantic,
            }
            for category, semantic in semantics.items()
        ]
    }
    audit = {
        "category_semantics": semantics,
        "source_provenance": {
            category: {"category_semantic": semantic}
            for category, semantic in semantics.items()
        },
    }

    assert train_bc._require_exact_flywheel_category_semantics(  # noqa: SLF001
        authority={"category_semantics": semantics},
        contract={"category_semantics": semantics},
        selected=selected,
        audit=audit,
    ) == semantics

    selected["records"][1]["category_semantic"] = {
        **selected["records"][1]["category_semantic"],
        "semantic": "recent_history",
    }
    with pytest.raises(SystemExit, match="selected-game manifest recent-history"):
        train_bc._require_exact_flywheel_category_semantics(  # noqa: SLF001
            authority={"category_semantics": semantics},
            contract={"category_semantics": semantics},
            selected=selected,
            audit=audit,
        )


def test_fresh_source_binding_rejects_semantic_stripping_or_laundering(
    tmp_path: Path,
) -> None:
    lock = _minimal_lock(tmp_path)
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    base = {
        "contract_sha256": lock["contract_sha256"],
        "audit_file_sha256": "sha256:" + "1" * 64,
        "audit_sha256": "sha256:" + "2" * 64,
        "selected_manifest_file_sha256": "sha256:" + "3" * 64,
        "selected_records_sha256": "sha256:" + "4" * 64,
        "job_id": "recent-job",
        "category": "recent_history",
        "source_path": str(source),
        "source_sha256": composite._file_sha256(source),  # noqa: SLF001
        "generation_manifest_path": str(manifest),
        "generation_manifest_sha256": composite._file_sha256(manifest),  # noqa: SLF001
    }

    def validate(value: dict) -> list[dict]:
        return composite._validate_source_bindings(  # noqa: SLF001
            [{"source_id": composite._binding_source_id(value), **value}],  # noqa: SLF001
            lock=lock,
            selected_file_sha256=base["selected_manifest_file_sha256"],
            selected_records_sha256=base["selected_records_sha256"],
            audit_file_sha256=base["audit_file_sha256"],
            audit_sha256=base["audit_sha256"],
        )

    with pytest.raises(composite.CompositeBuildError, match="fields differ"):
        validate(base)
    laundered = {
        **base,
        "category_semantic": {
            **lock["category_semantics"]["recent_history"],
            "semantic": "recent_history",
        },
    }
    with pytest.raises(composite.CompositeBuildError, match="identity/digest drift"):
        validate(laundered)
    exact = {
        **base,
        "category_semantic": lock["category_semantics"]["recent_history"],
    }
    assert validate(exact)[0]["category_semantic"] == exact["category_semantic"]


def test_composite_descriptor_carries_exact_recovery_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantics, _handoff, producer, _history, _hard = _recovery_semantics(tmp_path)
    components: list[dict] = []
    for component_id, ratio in composite.EFFECTIVE_COMPONENT_RATIOS.items():
        provenance = tmp_path / f"{component_id}.provenance.json"
        provenance.write_text(
            json.dumps({"checkpoint_versions": [4, 5]}) + "\n",
            encoding="utf-8",
        )
        component = {
            "component_id": component_id,
            "game_sampling_ratio": ratio,
            "provenance_manifest": str(provenance),
            "provenance_manifest_sha256": composite._file_sha256(provenance),  # noqa: SLF001
        }
        if component_id in composite.FRESH_SOURCE_GAME_RATIOS:
            corpus = tmp_path / f"{component_id}.corpus"
            corpus.mkdir()
            (corpus / "corpus_meta.json").write_text(
                json.dumps(
                    {
                        "row_count": 10,
                        "columns": {
                            "adapter_version": {
                                "kind": "string",
                                "categories": [CURRENT_RUST_ENTITY_ADAPTER_VERSION],
                            }
                        },
                        "aux_subgoal_target_contract": {
                            "version_key": composite.AUX_SUBGOAL_TARGET_VERSION_KEY,
                            "supported_version": composite.AUX_SUBGOAL_TARGET_VERSION,
                            "semantic": composite.AUX_SUBGOAL_TARGET_SEMANTIC,
                            "realized_version_counts": {
                                str(composite.AUX_SUBGOAL_TARGET_VERSION): 10
                            },
                            "all_rows_semantically_eligible": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            component["corpus_dir"] = str(corpus)
        components.append(component)
    monkeypatch.setattr(
        composite, "build_sampling_receipt", lambda _components: {"sealed": True}
    )
    descriptor = composite._build_descriptor(  # noqa: SLF001
        components=components,
        producer_path=producer,
        producer_sha256=_sha(producer),
        current_version=5,
        source_authority={
            "path": str(tmp_path / "source-authority.json"),
            "file_sha256": "sha256:" + "7" * 64,
            "authority_sha256": "sha256:" + "8" * 64,
        },
        category_semantics=semantics,
    )
    assert descriptor["category_semantics"] == semantics
    assert (
        descriptor["category_semantics"]["recent_history"]["semantic"]
        == contract.RECOVERY_REFERENCE_SEMANTIC
    )


def test_recovery_s1_binding_replays_without_promotion_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = {
        "c_scale": 0.03,
        "rescale_noise_floor_c": 0.0,
        "sigma_eval": 0.98,
        "symmetry_averaged_eval": True,
        "symmetry_averaged_eval_threshold": 20,
    }
    legacy = tmp_path / "legacy-s1.json"
    legacy.write_text("{}\n", encoding="utf-8")
    legacy_payload = {
        "selected_fields": selected,
        "selected_fields_sha256": binding._digest_value(selected),  # noqa: SLF001
    }
    checkpoint = tmp_path / "recovered.pt"
    checkpoint.write_bytes(b"recovered")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    search = {**selected, "c_scale": 0.1}
    identity = {
        "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
        "search_config": search,
        "agent_identity_sha256": "sha256:" + "5" * 64,
    }
    replay = {
        "authority": {
            "recovered_generator": dict(identity["checkpoint"]),
            "producer_identity": identity,
            "recovery_lineage_id": "recovery-v5-test",
        },
        "receipt": {"recovery_receipt_sha256": "sha256:" + "6" * 64},
    }
    monkeypatch.setattr(binding, "_replay_s1", lambda _path: legacy_payload)
    monkeypatch.setattr(binding, "_replay_recovery_receipt", lambda _path: replay)
    payload = binding.build_recovery_s1_binding(
        legacy,
        receipt,
        binding_time_utc="2026-07-13T00:00:00Z",
    )
    assert payload["selected_fields"]["c_scale"] == 0.1
    assert payload["promotion_proof_recreated"] is False
    assert "promotion" not in payload["source_recovery_receipt"]
    output = tmp_path / "recovery-s1.json"
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    assert binding._replay_recovery_s1(output) == payload  # noqa: SLF001

    tampered = dict(payload)
    tampered["promotion_proof_recreated"] = True
    output.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(binding.BindingError, match="semantic replay"):
        binding._replay_recovery_s1(output)  # noqa: SLF001
