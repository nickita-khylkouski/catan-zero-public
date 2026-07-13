from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from tools import a1_function_preserving_upgrade as upgrade
from tools import f69_upgrade_checkpoint_config as upgrade_tool
from tools import a1_lineage_dose as lineage
from tools import a1_one_dose_train as one_dose
from tools import a1_promotion_transaction as promotion


def test_upgrade_tools_bind_project_imports_to_their_checkout() -> None:
    module = sys.modules[upgrade_tool.EntityGraphPolicy.__module__]
    module_path = Path(str(module.__file__)).resolve(strict=True)

    assert upgrade.REPO_SRC in module_path.parents
    assert upgrade_tool._REPO_SRC in module_path.parents  # noqa: SLF001


def _checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "champion.pt"
    upgraded = tmp_path / "champion-gather.pt"
    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=7,
        device="cpu",
    )
    base.save(source, mask_hidden_info=True)
    gather = EntityGraphPolicy(
        dataclasses.replace(base.config, action_target_gather=True),
        base.static_action_features.detach().cpu().numpy(),
        seed=1,
        device="cpu",
    )
    missing, unexpected = gather.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == set(
        upgrade.ALLOWLIST[upgrade.MODULE_TARGET_GATHER][
            "new_parameter_initialization"
        ]
    )
    gather.save(upgraded, mask_hidden_info=True)
    raw = torch.load(upgraded, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix(  # noqa: SLF001
            "sha256:"
        ),
        "flags": {"action_target_gather": True},
        "initialization_seed": 1,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, upgraded)
    return source, upgraded


def _issued(tmp_path: Path) -> tuple[Path, dict]:
    source, initializer = _checkpoints(tmp_path)
    receipt = tmp_path / "upgrade.receipt.json"
    payload = upgrade.issue_receipt(source, initializer, receipt)
    return receipt, payload


def _topology_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    source, _gather = _checkpoints(tmp_path)
    upgraded = tmp_path / "champion-topology-gather.pt"
    base = EntityGraphPolicy.load(source, device="cpu")
    flags = {
        "action_target_gather": True,
        "topology_residual_adapter": True,
    }
    treatment = EntityGraphPolicy(
        dataclasses.replace(base.config, **flags),
        base.static_action_features.detach().cpu().numpy(),
        seed=1,
        device="cpu",
    )
    missing, unexpected = treatment.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == set(
        upgrade.ALLOWLIST[upgrade.MODULE_TOPOLOGY_TARGET_GATHER][
            "new_parameter_initialization"
        ]
    )
    treatment.save(upgraded, mask_hidden_info=True)
    raw = torch.load(upgraded, map_location="cpu", weights_only=False)
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


def _topology_on_trained_gather_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    """Append topology while preserving an already-nonzero gather exactly."""

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "trained-gather.pt"
    upgraded = tmp_path / "trained-gather-plus-topology.pt"
    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=7,
        device="cpu",
    )
    gather = EntityGraphPolicy(
        dataclasses.replace(base.config, action_target_gather=True),
        base.static_action_features.detach().cpu().numpy(),
        seed=11,
        device="cpu",
    )
    gather.model.load_state_dict(base.model.state_dict(), strict=False)
    with torch.no_grad():
        gather.model.target_gather_proj[1].weight.copy_(
            torch.eye(gather.config.hidden_size)
        )
        gather.model.target_gather_proj[1].bias.fill_(0.25)
    gather.save(source, mask_hidden_info=True)

    treatment = EntityGraphPolicy(
        dataclasses.replace(gather.config, topology_residual_adapter=True),
        gather.static_action_features.detach().cpu().numpy(),
        seed=1,
        device="cpu",
    )
    missing, unexpected = treatment.model.load_state_dict(
        gather.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == set(
        upgrade.ALLOWLIST[upgrade.MODULE_TOPOLOGY_RESIDUAL][
            "new_parameter_initialization"
        ]
    )
    treatment.save(upgraded, mask_hidden_info=True)
    raw = torch.load(upgraded, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix(  # noqa: SLF001
            "sha256:"
        ),
        "flags": {"topology_residual_adapter": True},
        "initialization_seed": 1,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, upgraded)
    return source, upgraded


def _belief_checkpoints(tmp_path: Path, *, seed: int = 73) -> tuple[Path, Path]:
    """Build the real additive belief-head upgrade used by the learner."""
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    source = tmp_path / "champion-real.pt"
    output = tmp_path / "champion-belief.pt"
    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=11,
        device="cpu",
    )
    base.save(source, mask_hidden_info=True)
    values = {
        field.name: getattr(base.config, field.name)
        for field in dataclasses.fields(EntityGraphConfig)
        if hasattr(base.config, field.name)
    }
    values["belief_resource_head"] = True
    belief = EntityGraphPolicy(
        EntityGraphConfig(**values),
        base.static_action_features.detach().cpu().numpy(),
        seed=seed,
        device="cpu",
    )
    missing, unexpected = belief.model.load_state_dict(
        base.model.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == set(
        upgrade.ALLOWLIST[upgrade.MODULE_BELIEF_RESOURCE_HEAD][
            "new_parameter_initialization"
        ]
    )
    belief.save(output, mask_hidden_info=True)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"] = {
        "schema_version": "entity-graph-upgrade-v1",
        "source_checkpoint_sha256": upgrade._sha(source).removeprefix("sha256:"),  # noqa: SLF001
        "flags": {"belief_resource_head": True},
        "initialization_seed": seed,
        "trained_value_readouts_added": [],
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
    }
    torch.save(raw, output)
    return source, output


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


def test_receipt_appends_topology_to_trained_nonzero_gather_without_drift(
    tmp_path: Path,
) -> None:
    source, initializer = _topology_on_trained_gather_checkpoints(tmp_path)
    receipt = tmp_path / "topology-only-upgrade.receipt.json"

    payload = upgrade.issue_receipt(
        source,
        initializer,
        receipt,
        module=upgrade.MODULE_TOPOLOGY_RESIDUAL,
    )
    verified = upgrade.verify_receipt(receipt)
    source_raw = torch.load(source, map_location="cpu", weights_only=False)
    upgraded_raw = torch.load(initializer, map_location="cpu", weights_only=False)

    assert payload["module"] == upgrade.MODULE_TOPOLOGY_RESIDUAL
    assert payload["flags"] == {"topology_residual_adapter": True}
    assert len(payload["new_parameters"]) == 8
    assert verified["shared_parameters_bit_identical"] is True
    assert torch.equal(
        source_raw["model"]["target_gather_proj.1.weight"],
        upgraded_raw["model"]["target_gather_proj.1.weight"],
    )
    assert torch.count_nonzero(
        upgraded_raw["model"]["target_gather_proj.1.weight"]
    ).item() > 0


def test_topology_only_receipt_rejects_drift_in_trained_gather(tmp_path: Path) -> None:
    source, initializer = _topology_on_trained_gather_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["target_gather_proj.1.weight"][0, 0] += 1.0
    torch.save(raw, initializer)

    with pytest.raises(
        upgrade.UpgradeError, match="shared checkpoint parameters changed"
    ):
        upgrade.inspect_upgrade(
            source,
            initializer,
            module=upgrade.MODULE_TOPOLOGY_RESIDUAL,
        )


def test_topology_only_receipt_rejects_zero_tensor_with_wrong_model_shape(
    tmp_path: Path,
) -> None:
    source, initializer = _topology_on_trained_gather_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["topology_residual_adapter.output_projection.weight"] = (
        torch.zeros(3, 3)
    )
    torch.save(raw, initializer)

    with pytest.raises(upgrade.UpgradeError, match="not deterministic zeros"):
        upgrade.inspect_upgrade(
            source,
            initializer,
            module=upgrade.MODULE_TOPOLOGY_RESIDUAL,
        )


@pytest.mark.parametrize(
    ("variant", "module"),
    (
        ("target_gather", upgrade.MODULE_TARGET_GATHER),
        ("combined", upgrade.MODULE_TOPOLOGY_TARGET_GATHER),
    ),
)
def test_legacy_additive_receipts_reject_correct_value_with_wrong_shape(
    tmp_path: Path,
    variant: str,
    module: str,
) -> None:
    if variant == "target_gather":
        source, initializer = _checkpoints(tmp_path)
    else:
        source, initializer = _topology_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["target_gather_proj.1.weight"] = torch.zeros(3, 3)
    torch.save(raw, initializer)

    with pytest.raises(upgrade.UpgradeError, match="not deterministic zeros"):
        upgrade.inspect_upgrade(source, initializer, module=module)


def test_receipt_replays_seeded_belief_head_upgrade(tmp_path: Path) -> None:
    source, initializer = _belief_checkpoints(tmp_path)
    receipt = tmp_path / "belief-upgrade.receipt.json"
    payload = upgrade.issue_receipt(
        source,
        initializer,
        receipt,
        module=upgrade.MODULE_BELIEF_RESOURCE_HEAD,
    )
    verified = upgrade.verify_receipt(receipt)
    expected_seeded = {
        name
        for name, kind in upgrade.ALLOWLIST[upgrade.MODULE_BELIEF_RESOURCE_HEAD][
            "new_parameter_initialization"
        ].items()
        if kind == "seeded_torch_default"
    }
    assert payload["initialization_seed"] == 73
    assert set(verified["seeded_parameter_sha256"]) == expected_seeded
    assert verified["shared_parameters_bit_identical"] is True


def test_receipt_accepts_belief_head_from_real_seeded_upgrader(
    tmp_path: Path, monkeypatch
) -> None:
    source, _ = _belief_checkpoints(tmp_path)
    initializer = tmp_path / "belief-real-upgrader.pt"
    monkeypatch.setattr(upgrade_tool, "_verify_forward_identical", lambda *_: 0.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f69_upgrade_checkpoint_config.py",
            "--in-checkpoint",
            str(source),
            "--out-checkpoint",
            str(initializer),
            "--flags",
            "belief",
            "--seed",
            "73",
            "--device",
            "cpu",
        ],
    )
    upgrade_tool.main()
    evidence = upgrade.inspect_upgrade(
        source,
        initializer,
        module=upgrade.MODULE_BELIEF_RESOURCE_HEAD,
    )
    assert evidence["initialization_seed"] == 73


def test_belief_receipt_rejects_wrong_seed_or_tampered_random_head(tmp_path: Path) -> None:
    source, initializer = _belief_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["upgrade_provenance"]["initialization_seed"] = 72
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="deterministic seeded_torch_default"):
        upgrade.inspect_upgrade(
            source,
            initializer,
            module=upgrade.MODULE_BELIEF_RESOURCE_HEAD,
        )

    source, initializer = _belief_checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["belief_resource_head.1.weight"][0, 0] += 0.01
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="deterministic seeded_torch_default"):
        upgrade.inspect_upgrade(
            source,
            initializer,
            module=upgrade.MODULE_BELIEF_RESOURCE_HEAD,
        )


def test_receipt_digest_normalizes_numpy_config_scalars(tmp_path: Path) -> None:
    source, initializer = _checkpoints(tmp_path)
    for path in (source, initializer):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        raw["config"]["fields"]["hidden_size"] = np.int64(16)
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

    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["model"]["hex_encoder.0.weight"] = raw["model"][
        "hex_encoder.0.weight"
    ].double()
    torch.save(raw, initializer)
    with pytest.raises(upgrade.UpgradeError, match="shared checkpoint parameters changed"):
        upgrade.inspect_upgrade(source, initializer)

    source, initializer = _checkpoints(tmp_path)
    raw = torch.load(initializer, map_location="cpu", weights_only=False)
    raw["mask_hidden_info"] = False
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
