from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import a1_function_preserving_upgrade as upgrade
from tools import a1_lineage_dose as lineage
from tools import a1_one_dose_train as one_dose
from tools import a1_promotion_transaction as promotion


def _checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "champion.pt"
    upgraded = tmp_path / "champion-gather.pt"
    base_config = {
        "state_trunk": "transformer",
        "action_size": 567,
        "static_action_feature_size": 1,
    }
    base_model = {
        "encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "policy.weight": torch.ones(2, 2),
    }
    torch.save(
        {"config": {"fields": base_config}, "model": base_model, "epoch": 7},
        source,
    )
    model = dict(base_model)
    model.update(
        {
            "target_gather_proj.0.bias": torch.zeros(3),
            "target_gather_proj.0.weight": torch.ones(3),
            "target_gather_proj.1.bias": torch.zeros(3),
            "target_gather_proj.1.weight": torch.zeros(3, 3),
        }
    )
    torch.save(
        {
            "config": {"fields": {**base_config, "action_target_gather": True}},
            "model": model,
            "epoch": 7,
            "upgrade_provenance": {
                "schema_version": "entity-graph-upgrade-v1",
                "source_checkpoint_sha256": upgrade._sha(source).removeprefix(  # noqa: SLF001
                    "sha256:"
                ),
                "flags": {"action_target_gather": True},
                "initialization_seed": 1,
                "trained_value_readouts_added": [],
                "forward_max_diff": 0.0,
                "forward_identical_at_init": True,
            },
        },
        upgraded,
    )
    return source, upgraded


def _issued(tmp_path: Path) -> tuple[Path, dict]:
    source, initializer = _checkpoints(tmp_path)
    receipt = tmp_path / "upgrade.receipt.json"
    payload = upgrade.issue_receipt(source, initializer, receipt)
    return receipt, payload


def _topology_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    source, _gather = _checkpoints(tmp_path)
    upgraded = tmp_path / "champion-topology-gather.pt"
    raw = torch.load(source, map_location="cpu", weights_only=False)
    model = dict(raw["model"])
    width = 3
    spec = upgrade.ALLOWLIST[upgrade.MODULE_TOPOLOGY_TARGET_GATHER]
    for name, kind in spec["new_parameter_initialization"].items():
        if name.endswith(".weight") and (
            "norm." not in name and "target_gather_proj.0" not in name
        ):
            shape = (width, width)
        else:
            shape = (width,)
        if kind == "ones":
            tensor = torch.ones(shape)
        elif kind == "zeros":
            tensor = torch.zeros(shape)
        elif kind == "identity":
            tensor = torch.eye(width)
        else:  # pragma: no cover - the allowlist itself is closed above
            raise AssertionError(kind)
        model[name] = tensor
    flags = {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    raw["model"] = model
    raw["config"] = {"fields": {**raw["config"]["fields"], **flags}}
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": flags,
        "initialization_seed": 1,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, upgraded)
    return source, upgraded


def _tamper_and_rehash(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = upgrade._digest(value)  # noqa: SLF001
    path.write_text(json.dumps(value), encoding="utf-8")


def test_receipt_replays_exact_allowlisted_zero_diff_upgrade(tmp_path: Path) -> None:
    receipt, payload = _issued(tmp_path)
    verified = upgrade.verify_receipt(receipt)
    assert verified["receipt_sha256"] == payload["receipt_sha256"]
    assert verified["module"] == upgrade.MODULE_TARGET_GATHER
    assert verified["forward_max_diff"] == 0.0
    assert verified["new_parameters"] == sorted(
        upgrade.ALLOWLIST[upgrade.MODULE_TARGET_GATHER][
            "new_parameter_initialization"
        ]
    )
    with pytest.raises(upgrade.UpgradeError, match="overwrite"):
        upgrade.issue_receipt(
            Path(payload["source"]["path"]),
            Path(payload["upgraded_initializer"]["path"]),
            receipt,
        )


def test_receipt_replays_combined_topology_target_gather_upgrade(tmp_path: Path) -> None:
    source, initializer = _topology_checkpoints(tmp_path)
    evidence = upgrade.inspect_upgrade(
        source,
        initializer,
        module=upgrade.MODULE_TOPOLOGY_TARGET_GATHER,
    )
    assert evidence["flags"] == {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    assert evidence["new_parameter_initialization"][
        "topology_residual_adapter.source_projection.weight"
    ] == "identity"


def test_receipt_digest_normalizes_numpy_config_scalars(tmp_path: Path) -> None:
    source, initializer = _checkpoints(tmp_path)
    for path in (source, initializer):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        raw["config"]["fields"]["action_size"] = np.int64(567)
        torch.save(raw, path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"]["source_checkpoint_sha256"] = upgrade._sha(  # noqa: SLF001
        source
    ).removeprefix("sha256:")
    torch.save(raw, initializer)

    receipt = tmp_path / "numpy-config.receipt.json"
    payload = upgrade.issue_receipt(source, initializer, receipt)
    assert upgrade.verify_receipt(receipt)["receipt_sha256"] == payload["receipt_sha256"]
    assert upgrade._digest({"value": np.int64(7)}) == upgrade._digest(  # noqa: SLF001
        {"value": 7}
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source"].__setitem__("sha256", "sha256:" + "1" * 64),
        lambda value: value["upgraded_initializer"].__setitem__(
            "sha256", "sha256:" + "2" * 64
        ),
        lambda value: value.__setitem__("flags", {"action_target_gather": False}),
        lambda value: value.__setitem__("forward_max_diff", 1e-12),
        lambda value: value["new_parameters"].append("attacker.weight"),
    ],
    ids=("source", "initializer", "flags", "nonzero-diff", "new-key"),
)
def test_semantically_rehashed_receipt_tampering_is_rejected(
    tmp_path: Path, mutate
) -> None:
    receipt, _ = _issued(tmp_path)
    _tamper_and_rehash(receipt, mutate)
    with pytest.raises(upgrade.UpgradeError, match="does not replay exactly"):
        upgrade.verify_receipt(receipt)


def test_checkpoint_parameter_or_metadata_drift_is_rejected(tmp_path: Path) -> None:
    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["target_gather_proj.1.weight"][0, 0] = 0.01
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="deterministic zeros"):
        upgrade.inspect_upgrade(source, initializer)

    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["epoch"] = 8
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="metadata/provenance changed"):
        upgrade.inspect_upgrade(source, initializer)


def test_one_dose_binds_upgraded_init_and_typed_lineage(tmp_path: Path) -> None:
    receipt, payload = _issued(tmp_path)
    verified = {
        "producer": payload["source"],
        "contract_sha256": "sha256:" + "c" * 64,
        "recipe": {
            "resume_optimizer": False,
            "batch_size": 512,
            "grad_accum_steps": 1,
            "max_steps": 1024,
        },
        "training_row_count": 4_194_304,
    }
    bound = one_dose.bind_function_preserving_upgrade(verified, receipt)
    dose = one_dose._direct_lineage_dose(bound)  # noqa: SLF001
    assert dose["declared_producer_sha256"] == payload["source"]["sha256"]
    assert dose["init_checkpoint_sha256"] == payload["upgraded_initializer"]["sha256"]
    assert dose["function_preserving_upgrade"]["receipt_sha256"] == upgrade._sha(  # noqa: SLF001
        receipt
    )
    assert lineage.validate_lineage_dose(dose) == dose


def test_exact_parent_remains_default_and_untyped_delta_is_refused() -> None:
    producer = "sha256:" + "a" * 64
    other = "sha256:" + "b" * 64
    exact = lineage.direct_lineage_dose(
        declared_producer_sha256=producer,
        init_checkpoint_sha256=producer,
        current_sampled_rows=10,
        current_optimizer_steps=1,
    )
    assert exact["function_preserving_upgrade"] is None
    with pytest.raises(lineage.LineageDoseError, match="untyped checkpoint chaining"):
        lineage.direct_lineage_dose(
            declared_producer_sha256=producer,
            init_checkpoint_sha256=other,
            current_sampled_rows=10,
            current_optimizer_steps=1,
        )


def test_promotion_report_accepts_only_replayed_upgrade_lineage(tmp_path: Path) -> None:
    receipt, payload = _issued(tmp_path)
    verified = {
        "producer": payload["source"],
        "contract_sha256": "sha256:" + "c" * 64,
        "recipe": {
            "resume_optimizer": False,
            "batch_size": 512,
            "grad_accum_steps": 1,
            "max_steps": 1024,
        },
        "training_row_count": 4_194_304,
    }
    bound = one_dose.bind_function_preserving_upgrade(verified, receipt)
    dose = one_dose._direct_lineage_dose(bound)  # noqa: SLF001
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"trained candidate")
    recipe = {"epochs": 1, "max_steps": 1024, "symmetry_augment": False}
    contract = {
        "contract_sha256": verified["contract_sha256"],
        "science": {
            "learner_training_recipe": recipe,
            "learner_training_recipe_sha256": promotion._digest_value(recipe),  # noqa: SLF001
        },
        "checkpoints": [{"role": "producer", "sha256": payload["source"]["sha256"]}],
    }
    report_value = {
        "a1_contract_sha256": verified["contract_sha256"],
        "a1_learner_training_recipe_sha256": promotion._digest_value(recipe),  # noqa: SLF001
        "a1_bound_learner_training_recipe": recipe,
        "arch": "entity_graph",
        "mask_hidden_info": True,
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "symmetry_augment": False,
        "checkpoint": str(candidate),
        "init_checkpoint": payload["upgraded_initializer"]["path"],
        "init_checkpoint_sha256": payload["upgraded_initializer"]["sha256"],
        "a1_lineage_dose": dose,
        "steps_completed": 1024,
        "epochs": 1,
        "max_steps": 1024,
    }
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_value), encoding="utf-8")
    assert promotion._verify_training_report(  # noqa: SLF001
        report,
        contract=contract,
        contract_sha256=verified["contract_sha256"],
        candidate_path=candidate,
        candidate_sha256=promotion._sha256(candidate),  # noqa: SLF001
    ) == report_value

    report_value["a1_lineage_dose"]["function_preserving_upgrade"][
        "receipt_sha256"
    ] = "sha256:" + "9" * 64
    report.write_text(json.dumps(report_value), encoding="utf-8")
    with pytest.raises(promotion.PromotionError, match="does not bind producer/init"):
        promotion._verify_training_report(  # noqa: SLF001
            report,
            contract=contract,
            contract_sha256=verified["contract_sha256"],
            candidate_path=candidate,
            candidate_sha256=promotion._sha256(candidate),  # noqa: SLF001
        )
