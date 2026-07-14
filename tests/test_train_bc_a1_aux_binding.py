from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools import a1_aux_pair_coordinator as coordinator
from tools import train_bc


SELECTED_DECIMAL = "0.013"
SELECTED = 0.013


def _sha(value: str) -> str:
    return "sha256:" + value * 64


def _binding(
    tmp_path: Path, *, arm: str = "AUX0"
) -> tuple[argparse.Namespace, dict[str, object]]:
    upgrade_initializer = (tmp_path / "pointer-upgrade.pt").resolve()
    upgrade_initializer.write_bytes(b"function-preserving pointer initializer")
    upgrade_initializer_sha = train_bc._sha256_existing_file(  # noqa: SLF001
        upgrade_initializer
    )
    warmed_initializer = (tmp_path / "shared-warmed-pointer.pt").resolve()
    warmed_initializer.write_bytes(b"one shared commissioned pointer initializer")
    warmed_initializer_sha = train_bc._sha256_existing_file(  # noqa: SLF001
        warmed_initializer
    )
    receipt_path = (tmp_path / "pointer-upgrade.receipt.json").resolve()
    unsigned_receipt = {
        "schema_version": "a1-function-preserving-architecture-upgrade-v1",
        "module": "entity_graph.aux_subgoal_pointer_heads.v1",
        "source": {
            "path": str((tmp_path / "current-promoted-parent.pt").resolve()),
            "sha256": _sha("1"),
        },
        "upgraded_initializer": {
            "path": str(upgrade_initializer),
            "sha256": upgrade_initializer_sha,
        },
    }
    receipt_digest = train_bc._canonical_json_sha256(unsigned_receipt)  # noqa: SLF001
    receipt_path.write_text(
        json.dumps({**unsigned_receipt, "receipt_sha256": receipt_digest}),
        encoding="utf-8",
    )
    receipt_file_sha = train_bc._sha256_existing_file(receipt_path)  # noqa: SLF001
    shared = {
        "schema_version": "a1-aux-pointer-shared-identity-v1",
        "pair_id": "aux-pointer-pair",
        "upgrade_module": "entity_graph.aux_subgoal_pointer_heads.v1",
        "upgrade_receipt_file_sha256": receipt_file_sha,
        "upgrade_receipt_digest": receipt_digest,
        "initializer_sha256": warmed_initializer_sha,
        "pair_contract_state_sha256": _sha("2"),
        "p1_selection_authority_sha256": _sha("3"),
        "warmup_terminal_sha256": _sha("4"),
        "gradient_geometry_terminal_sha256": _sha("5"),
        "selector_rule_sha256": _sha("6"),
        "selected_aux_coefficient_decimal": SELECTED_DECIMAL,
    }
    weight = 0.0 if arm == "AUX0" else SELECTED
    binding: dict[str, object] = {
        "schema_version": "a1-matched-aux-pointer-arm-v1",
        "arm_id": arm,
        "aux_subgoal_loss_weight": weight,
        "selected_aux_coefficient_decimal": SELECTED_DECIMAL,
        "upgrade_module": "entity_graph.aux_subgoal_pointer_heads.v1",
        "upgrade_receipt": str(receipt_path),
        "upgrade_receipt_file_sha256": receipt_file_sha,
        "upgrade_receipt_digest": receipt_digest,
        "upgrade_initializer": str(upgrade_initializer),
        "upgrade_initializer_sha256": upgrade_initializer_sha,
        "initializer": str(warmed_initializer),
        "initializer_sha256": warmed_initializer_sha,
        "aux_pair_authority_sha256": _sha("7"),
        "shared_identity": shared,
        "shared_identity_sha256": train_bc._canonical_json_sha256(shared),  # noqa: SLF001
    }
    args = argparse.Namespace(
        a1_aux_regularization_binding_json=json.dumps(binding),
        aux_subgoal_loss_weight=weight,
        aux_subgoal_heads=True,
        aux_settlement_pointer_head=True,
        init_checkpoint=str(warmed_initializer),
        init_checkpoint_sha256=warmed_initializer_sha,
    )
    return args, binding


def test_aux0_authenticates_shared_pointer_commissioning_bytes(tmp_path: Path) -> None:
    args, binding = _binding(tmp_path, arm="AUX0")

    assert train_bc._validate_a1_aux_regularization_binding(  # noqa: SLF001
        args, recipe_drift={}
    ) == binding


def test_auxt_accepts_only_geometry_selected_delta(tmp_path: Path) -> None:
    args, binding = _binding(tmp_path, arm="AUXT")
    drift = {
        "aux_subgoal_loss_weight": {"contract": 0.0, "effective": SELECTED}
    }
    assert train_bc._validate_a1_aux_regularization_binding(  # noqa: SLF001
        args, recipe_drift=drift
    ) == binding

    with pytest.raises(SystemExit, match="differ only"):
        train_bc._validate_a1_aux_regularization_binding(  # noqa: SLF001
            args,
            recipe_drift={
                **drift,
                "lr": {"contract": 3e-5, "effective": 1e-4},
            },
        )


@pytest.mark.parametrize(
    "field",
    ("upgrade_receipt", "upgrade_initializer", "initializer"),
)
def test_aux_binding_rejects_substituted_bytes(
    tmp_path: Path, field: str
) -> None:
    args, binding = _binding(tmp_path, arm="AUXT")
    Path(str(binding[field])).write_bytes(b"substituted bytes")

    with pytest.raises(SystemExit, match="byte drift"):
        train_bc._validate_a1_aux_regularization_binding(  # noqa: SLF001
            args,
            recipe_drift={
                "aux_subgoal_loss_weight": {
                    "contract": 0.0,
                    "effective": SELECTED,
                }
            },
        )


def test_aux_binding_rejects_cls_settlement_architecture(tmp_path: Path) -> None:
    args, _ = _binding(tmp_path, arm="AUXT")
    args.aux_settlement_pointer_head = False

    with pytest.raises(SystemExit, match="architecture drift"):
        train_bc._validate_a1_aux_regularization_binding(  # noqa: SLF001
            args,
            recipe_drift={
                "aux_subgoal_loss_weight": {
                    "contract": 0.0,
                    "effective": SELECTED,
                }
            },
        )


@pytest.mark.parametrize(
    ("arm", "weight"),
    (("AUX2", 0.02), ("AUXT", 0.02), ("AUX0", SELECTED)),
)
def test_aux_binding_rejects_operator_chosen_label_or_weight(
    tmp_path: Path, arm: str, weight: float
) -> None:
    args, binding = _binding(tmp_path, arm="AUXT")
    binding["arm_id"] = arm
    binding["aux_subgoal_loss_weight"] = weight
    args.aux_subgoal_loss_weight = weight
    args.a1_aux_regularization_binding_json = json.dumps(binding)

    with pytest.raises(SystemExit, match="arm/weight/architecture drift"):
        train_bc._validate_a1_aux_regularization_binding(  # noqa: SLF001
            args, recipe_drift=None
        )


def test_aux_binding_rejects_differing_shared_identity(tmp_path: Path) -> None:
    args, binding = _binding(tmp_path, arm="AUX0")
    shared = dict(binding["shared_identity"])
    shared["initializer_sha256"] = _sha("9")
    binding["shared_identity"] = shared
    binding["shared_identity_sha256"] = train_bc._canonical_json_sha256(  # noqa: SLF001
        shared
    )
    args.a1_aux_regularization_binding_json = json.dumps(binding)

    with pytest.raises(SystemExit, match="shared-pair identity drift"):
        train_bc._validate_a1_aux_regularization_binding(  # noqa: SLF001
            args, recipe_drift={}
        )


def test_aux_weight_drift_without_central_binding_is_rejected(tmp_path: Path) -> None:
    args, _ = _binding(tmp_path, arm="AUXT")
    args.a1_aux_regularization_binding_json = ""

    with pytest.raises(SystemExit, match="requires the matched aux"):
        train_bc._validate_a1_aux_regularization_binding(  # noqa: SLF001
            args,
            recipe_drift={
                "aux_subgoal_loss_weight": {
                    "contract": 0.0,
                    "effective": SELECTED,
                }
            },
        )


def _final_published_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_aux: str,
    broken_warmup: bool = False,
) -> tuple[argparse.Namespace, dict[str, object], dict[str, object]]:
    raw_parent_sha = _sha("1")
    transitioned_sha = _sha("2")
    pointer_sha = _sha("3")
    warmed_sha = _sha("4")
    raw_parent = {
        "checkpoint_path": str((tmp_path / "raw-parent.pt").resolve()),
        "checkpoint_sha256": raw_parent_sha,
    }
    transition = {
        "transitioned_checkpoint": {
            "path": str((tmp_path / "transitioned.pt").resolve()),
            "sha256": transitioned_sha,
        }
    }
    pointer = {"upgraded_initializer_sha256": pointer_sha}
    warmup = {
        "result": {
            "input_initializer_sha256": (
                _sha("f") if broken_warmup else pointer_sha
            ),
            "warmed_checkpoint_sha256": warmed_sha,
            "optimizer_sidecar_discarded_for_joint": True,
        }
    }
    initializer = {
        "exact_current_parent_authority": raw_parent,
        "public_award_transition_authority": transition,
        "pointer_upgrade_authority": pointer if selected_aux == "AUXT" else None,
        "reference_warmup_terminal": warmup if selected_aux == "AUXT" else None,
    }
    sample = {
        "state_sha256": _sha("5"),
        "descriptor_sha256": _sha("6"),
        "payload_inventory_sha256": _sha("7"),
        "category_semantics": {"current_producer": "fresh"},
        "category_semantics_sha256": _sha("8"),
        "source_authority": {
            "path": "/immutable/source.json",
            "file_sha256": _sha("9"),
            "authority_sha256": _sha("a"),
        },
        "sampler_identity_sha256": _sha("b"),
        "sample_order_sha256": _sha("c"),
        "row_set_sha256": _sha("d"),
        "unique_row_count": 10,
        "rows_file_sha256": _sha("e"),
        "sample_dose": 524_288,
        "sampler_seed": 424243,
        "prior_rows_file_sha256": _sha("f"),
        "prior_row_set_sha256": _sha("0"),
        "kl_eligible_rows": 1,
        "kl_eligible_mass_decimal": "0.1",
        "kl_ordered_evidence_sha256": _sha("1"),
        "kl_eligible_evidence_sha256": _sha("2"),
    }
    recipe = {"lr": 3e-5, "aux_subgoal_loss_weight": 0.0}
    final = {
        "effective_recipe": recipe,
        "sampling_receipt": sample,
        "selected_aux_decision": selected_aux,
        "initializer_authority": initializer,
    }
    authority = {
        "schema_version": coordinator.FINAL_EXECUTOR_AUTHORITY_SCHEMA,
        "authority_sha256": _sha("3"),
        "state_sha256": _sha("4"),
        "final_replication_authority": final,
    }
    authority_path = (tmp_path / f"final-{selected_aux}.json").resolve()
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    authority_file_sha = train_bc._sha256_existing_file(authority_path)  # noqa: SLF001
    central = {
        "stage": "FINAL",
        "central_authority_schema": coordinator.FINAL_EXECUTOR_AUTHORITY_SCHEMA,
        "central_authority_sha256": authority["authority_sha256"],
        "executor_authority_path": str(authority_path),
        "executor_authority_file_sha256": authority_file_sha,
        "executor_authority_state_sha256": authority["state_sha256"],
        "selected_aux_decision": selected_aux,
        "effective_recipe": recipe,
        "sample_binding": train_bc._a1_sample_binding_projection(sample),  # noqa: SLF001
        "initializer_sha256": (
            transitioned_sha if selected_aux == "AUX0" else warmed_sha
        ),
    }
    args = argparse.Namespace(
        a1_central_executor_authority=str(authority_path),
        a1_central_executor_authority_sha256=authority_file_sha,
    )

    def verify_published(path: Path):
        assert path == authority_path
        return {
            "path": str(authority_path),
            "file_sha256": authority_file_sha,
            "authority": authority,
        }

    def verify_transition(value, *, expected_parent):
        assert value == transition
        assert expected_parent == raw_parent
        return transition

    def verify_pointer(value, *, expected_parent_sha256):
        assert value == pointer
        assert expected_parent_sha256 == transitioned_sha
        return pointer

    monkeypatch.setattr(
        coordinator, "verify_published_executor_authority", verify_published
    )
    monkeypatch.setattr(
        coordinator, "verify_public_award_transition_authority", verify_transition
    )
    monkeypatch.setattr(
        coordinator, "verify_pointer_upgrade_authority", verify_pointer
    )
    return args, central, authority


def test_final_aux0_projection_uses_transitioned_not_raw_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, central, _authority = _final_published_authority(
        tmp_path, monkeypatch, selected_aux="AUX0"
    )
    published = train_bc._validate_a1_published_executor_authority(  # noqa: SLF001
        args, central
    )
    assert published["authority"]["final_replication_authority"][
        "initializer_authority"
    ]["public_award_transition_authority"]["transitioned_checkpoint"][
        "sha256"
    ] == central["initializer_sha256"]

    central["initializer_sha256"] = _sha("1")
    with pytest.raises(SystemExit, match="inline projection"):
        train_bc._validate_a1_published_executor_authority(args, central)  # noqa: SLF001


def test_final_auxt_projection_replays_transition_pointer_and_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, central, _authority = _final_published_authority(
        tmp_path, monkeypatch, selected_aux="AUXT"
    )
    train_bc._validate_a1_published_executor_authority(args, central)  # noqa: SLF001

    bad_args, bad_central, _bad_authority = _final_published_authority(
        tmp_path,
        monkeypatch,
        selected_aux="AUXT",
        broken_warmup=True,
    )
    with pytest.raises(SystemExit, match="warmup does not continue"):
        train_bc._validate_a1_published_executor_authority(  # noqa: SLF001
            bad_args, bad_central
        )
