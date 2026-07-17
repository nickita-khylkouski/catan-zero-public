from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from train_bc import (  # type: ignore  # noqa: E402
    A1_REQUIRED_LEARNER_CODE_SUFFIXES,
    A1_REQUIRED_RUNTIME_CODE_SUFFIXES,
    _canonical_json_sha256,
    _game_seed_set_sha256,
    _load_validation_game_seed_manifest_for_training,
    _preflight_a1_memmap_metadata,
    _training_data_fingerprint,
    _validation_contract_config_identity,
    _value_training_metadata,
    _a1_report_eligibility_from_training_semantics,
    _validate_a1_corpus_artifacts_and_seeds,
    _validate_a1_decisive_training_semantics,
    _validate_a1_learner_objective,
    _validate_a1_learner_training_recipe,
    _validate_a1_validation_manifest_corpus_binding,
    split_train_validation_indices,
)
from a1_pre_wave_contract import (  # type: ignore  # noqa: E402
    EXPECTED_LEARNER_TRAINING_RECIPE,
)


_CONTRACT_SHA = "sha256:" + "a" * 64


def test_validation_contract_identity_binds_metric_and_excluded_game_sets() -> None:
    selected = np.asarray([11, 13], dtype=np.int64)
    excluded = np.asarray([11, 13, 17], dtype=np.int64)
    identity = _validation_contract_config_identity(
        {
            "file_sha256": "sha256:sentinel",
            "game_seeds": selected,
            "excluded_game_seeds": excluded,
        }
    )

    assert identity == {
        "validation_contract_file_sha256": "sha256:sentinel",
        "validation_game_seed_set_sha256": _game_seed_set_sha256(selected),
        "training_excluded_game_seed_set_sha256": _game_seed_set_sha256(excluded),
    }


def test_missing_validation_contract_has_empty_config_identity() -> None:
    assert _validation_contract_config_identity(None) == {
        "validation_contract_file_sha256": "",
        "validation_game_seed_set_sha256": "",
        "training_excluded_game_seed_set_sha256": "",
    }


def test_composite_contract_value_metadata_allows_no_single_corpus_binding() -> None:
    """A composite binds component holdouts, not one selected-game manifest.

    The trainer used to complete every optimizer step and validation batch,
    then crash while saving provenance because these single-corpus attributes
    were read unconditionally whenever an A1 contract was present.
    """

    args = argparse.Namespace(
        a1_contract_sha256=_CONTRACT_SHA,
        value_head_type="mse",
    )
    metadata = _value_training_metadata(
        args,
        scalar_weight=0.25,
        categorical_weight=0.0,
        categorical_bins=0,
        optimizer_steps=1024,
        completed_epochs=1,
        scalar_training_weight_sum=1.0,
        categorical_training_weight_sum=0.0,
    )
    assert metadata["a1_contract_sha256"] == _CONTRACT_SHA
    assert metadata["a1_selected_game_seed_set_sha256"] is None
    assert metadata["a1_training_game_seed_set_sha256"] is None
    assert metadata["a1_learner_training_recipe_sha256"] is None


def _write_manifest(path: Path, *, seeds: list[int] | None = None) -> dict:
    seeds = [11, 13] if seeds is None else seeds
    seed_array = np.asarray(seeds, dtype="<i8")
    payload = {
        "schema_version": "train-validation-game-seeds-v1",
        "a1_contract_sha256": _CONTRACT_SHA,
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "validation_game_seed_ranges": [],
        "validation_game_seed_count": len(seeds),
        "validation_row_count": 4,
        "validation_game_seed_set_sha256": (
            "sha256:" + hashlib.sha256(seed_array.tobytes()).hexdigest()
        ),
        "game_seeds": seeds,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _load(path: Path) -> dict[str, object]:
    return _load_validation_game_seed_manifest_for_training(
        path,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seed_ranges=[],
    )


def _write_a1_chain(tmp_path: Path) -> tuple[dict, dict[str, object], np.ndarray]:
    producer_sha = "sha256:" + "e" * 64
    # This fixture deliberately constructs a historical v2 lock.  Do not read
    # the live v3 template here: current waves bind per-game policy weighting
    # and rank-offset RNG semantics that the immutable v2 learner never had.
    learner_training_recipe = dict(EXPECTED_LEARNER_TRAINING_RECIPE)
    learner_objective = {
        "objective": "mse",
        "value_readout": "scalar",
        "value_categorical_bins": None,
        "hlgauss_sigma_ratio": None,
    }
    learner_code: list[dict[str, str]] = []
    for suffix in sorted(A1_REQUIRED_LEARNER_CODE_SUFFIXES):
        source = _ROOT / suffix
        shadow = tmp_path / "learner" / suffix
        shadow.parent.mkdir(parents=True, exist_ok=True)
        shadow.write_bytes(source.read_bytes())
        learner_code.append(
            {
                "kind": "learner_code",
                "path": str(shadow.resolve()),
                "sha256": "sha256:" + hashlib.sha256(shadow.read_bytes()).hexdigest(),
            }
        )
    runtime_code_tree: list[dict[str, str]] = []
    for suffix in sorted(A1_REQUIRED_RUNTIME_CODE_SUFFIXES):
        source = _ROOT / suffix
        shadow = tmp_path / "runtime" / suffix
        shadow.parent.mkdir(parents=True, exist_ok=True)
        shadow.write_bytes(source.read_bytes())
        runtime_code_tree.append(
            {
                "kind": "runtime_code",
                "path": str(shadow.resolve()),
                "sha256": "sha256:" + hashlib.sha256(shadow.read_bytes()).hexdigest(),
            }
        )
    lock = {
        "schema_version": "a1-pre-wave-contract-lock-v2",
        "science": {
            "learner_value_objective": learner_objective,
            "learner_value_objective_sha256": _canonical_json_sha256(
                learner_objective
            ),
            "learner_training_recipe": learner_training_recipe,
            "learner_training_recipe_sha256": _canonical_json_sha256(
                learner_training_recipe
            ),
        },
        "checkpoints": [{"role": "producer", "sha256": producer_sha}],
        "provenance": {
            "learner_code": learner_code,
            "learner_code_sha256": _canonical_json_sha256(learner_code),
            "runtime_code_tree": runtime_code_tree,
            "runtime_code_tree_sha256": _canonical_json_sha256(
                runtime_code_tree
            ),
        },
    }
    contract_sha = _canonical_json_sha256(lock)
    lock["contract_sha256"] = contract_sha
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")

    seeds = np.arange(100_000, 112_000, dtype=np.int64)
    validation_seeds = seeds[-2:]
    records = [
        {
            "game_seed": int(seed),
            "job_id": f"job-{index % 72}",
            "worker_id": f"gpu{index % 24:02d}",
            "category": "current_producer",
            "producer_checkpoint_sha256": producer_sha,
            "opponent_checkpoint_sha256": [producer_sha],
            "split": "validation" if seed in validation_seeds else "train",
        }
        for index, seed in enumerate(seeds)
    ]
    train_seeds = seeds[:-2]
    selected = {
        "schema_version": "a1-selected-training-games-v1",
        "a1_contract_sha256": contract_sha,
        "selection_rule": "lowest_seed_complete_per_job",
        "selected_game_count": 12_000,
        "category_game_counts": {
            "current_producer": 9_600,
            "recent_history": 1_800,
            "hard_negative": 600,
        },
        "selected_game_seed_set_sha256": _game_seed_set_sha256(seeds),
        "training_game_count": len(train_seeds),
        "training_game_seed_set_sha256": _game_seed_set_sha256(train_seeds),
        "validation_game_count": len(validation_seeds),
        "validation_game_seed_set_sha256": _game_seed_set_sha256(
            validation_seeds
        ),
        "records_sha256": _canonical_json_sha256(records),
        "records": records,
    }
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8"
    )

    validation = {
        "schema_version": "train-validation-game-seeds-v1",
        "a1_contract_sha256": contract_sha,
        "validation_fraction": 0.05,
        "validation_seed": 17,
        "validation_max_samples": 0,
        "validation_game_seed_ranges": [],
        "validation_game_seed_count": 2,
        "validation_row_count": 2,
        "validation_game_seed_set_sha256": _game_seed_set_sha256(
            validation_seeds
        ),
        "game_seeds": validation_seeds.tolist(),
    }
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    validation_contract = _load_validation_game_seed_manifest_for_training(
        validation_path,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seed_ranges=[],
    )

    selected_file_sha = "sha256:" + hashlib.sha256(
        selected_path.read_bytes()
    ).hexdigest()
    validation_file_sha = validation_contract["file_sha256"]
    audit = {
        "schema_version": "a1-post-wave-audit-v2",
        "contract_path": str(lock_path.resolve()),
        "contract_sha256": contract_sha,
        "passed": True,
        "errors": [],
        "rows": 12_000,
        "selected_training_games": {
            "manifest": str(selected_path.resolve()),
            "manifest_sha256": _canonical_json_sha256(selected),
            "manifest_file_sha256": selected_file_sha,
            "selected_game_count": 12_000,
            "selected_game_seed_set_sha256": selected[
                "selected_game_seed_set_sha256"
            ],
            "records_sha256": selected["records_sha256"],
        },
        "validation_holdout": {
            "manifest": str(validation_path.resolve()),
            "manifest_sha256": validation_contract["manifest_sha256"],
            "manifest_file_sha256": validation_file_sha,
            "validation_game_seed_count": 2,
            "validation_game_seed_set_sha256": validation[
                "validation_game_seed_set_sha256"
            ],
        },
    }
    audit["audit_sha256"] = _canonical_json_sha256(audit)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    audit_file_sha = "sha256:" + hashlib.sha256(audit_path.read_bytes()).hexdigest()

    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": 12_000,
        "selected_game_seed_manifest": {
            "path": str(selected_path.resolve()),
            "file_sha256": selected_file_sha,
            "a1_contract_sha256": contract_sha,
            "selected_game_count": 12_000,
            "selected_game_seed_set_sha256": selected[
                "selected_game_seed_set_sha256"
            ],
            "training_game_count": len(train_seeds),
            "training_game_seed_set_sha256": selected[
                "training_game_seed_set_sha256"
            ],
            "validation_game_count": 2,
            "validation_game_seed_set_sha256": selected[
                "validation_game_seed_set_sha256"
            ],
            "records_sha256": selected["records_sha256"],
        },
        "a1_post_wave_audit": {
            "path": str(audit_path.resolve()),
            "file_sha256": audit_file_sha,
            "audit_sha256": audit["audit_sha256"],
            "contract_sha256": contract_sha,
            "shard_inventory_sha256": "sha256:" + "f" * 64,
            "source_provenance": {"current_producer": {}},
            "selected_row_count": 12_000,
            "training_row_count": 11_998,
            "validation_holdout": {
                "path": str(validation_path.resolve()),
                "file_sha256": validation_file_sha,
                "manifest_sha256": validation_contract["manifest_sha256"],
                "a1_contract_sha256": contract_sha,
                "validation_game_seed_count": 2,
                "validation_row_count": 2,
                "validation_game_seed_set_sha256": validation[
                    "validation_game_seed_set_sha256"
                ],
            },
        },
    }
    return meta, validation_contract, seeds


def test_loads_exact_a1_validation_manifest_and_binds_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    payload = _write_manifest(path)

    loaded = _load(path)

    np.testing.assert_array_equal(loaded["game_seeds"], np.asarray([11, 13]))
    assert loaded["validation_game_seed_count"] == 2
    assert loaded["a1_contract_sha256"] == _CONTRACT_SHA
    assert loaded["validation_game_seed_set_sha256"] == payload[
        "validation_game_seed_set_sha256"
    ]
    assert loaded["file_sha256"] == (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_validation_manifest_rejects_digest_and_cli_drift(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    payload = _write_manifest(path)
    payload["validation_game_seed_set_sha256"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="seed digest mismatch"):
        _load(path)

    _write_manifest(path)
    with pytest.raises(SystemExit, match="fraction differs from CLI"):
        _load_validation_game_seed_manifest_for_training(
            path,
            validation_fraction=0.10,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )
    with pytest.raises(SystemExit, match="validation_max_samples differs from CLI"):
        _load_validation_game_seed_manifest_for_training(
            path,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=1,
            validation_game_seed_ranges=[],
        )


def test_exact_manifest_split_is_game_level_and_rejects_missing_seed() -> None:
    data = {
        "action_taken": np.zeros(8, dtype=np.int16),
        "game_seed": np.asarray([10, 10, 11, 11, 12, 12, 13, 13]),
    }
    split = split_train_validation_indices(
        data,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seeds=np.asarray([11, 13]),
    )
    np.testing.assert_array_equal(split["validation"], np.asarray([2, 3, 6, 7]))
    np.testing.assert_array_equal(split["train"], np.asarray([0, 1, 4, 5]))
    assert _game_seed_set_sha256(data["game_seed"][split["validation"]]) == (
        _game_seed_set_sha256(np.asarray([11, 13]))
    )

    with pytest.raises(SystemExit, match="absent from the corpus: missing=1"):
        split_train_validation_indices(
            data,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seeds=np.asarray([11, 99]),
        )


def test_validation_manifest_must_match_selected_corpus_contract_and_split(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "validation.json"
    _write_manifest(manifest_path)
    contract = _load(manifest_path)
    validation_binding = {
        "path": str(contract["path"]),
        "file_sha256": contract["file_sha256"],
        "manifest_sha256": contract["manifest_sha256"],
        "a1_contract_sha256": contract["a1_contract_sha256"],
        "validation_game_seed_count": 2,
        "validation_row_count": contract["validation_row_count"],
        "validation_game_seed_set_sha256": contract[
            "validation_game_seed_set_sha256"
        ],
    }
    meta = {
        "selected_game_seed_manifest": {
            "a1_contract_sha256": _CONTRACT_SHA,
            "selected_game_count": 12_000,
            "validation_game_count": 2,
            "validation_game_seed_set_sha256": contract[
                "validation_game_seed_set_sha256"
            ],
        },
        "a1_post_wave_audit": {
            "contract_sha256": _CONTRACT_SHA,
            "file_sha256": "sha256:" + "c" * 64,
            "audit_sha256": "sha256:" + "d" * 64,
            "validation_holdout": validation_binding,
        },
    }
    _validate_a1_validation_manifest_corpus_binding(meta, contract)

    meta["selected_game_seed_manifest"]["a1_contract_sha256"] = (
        "sha256:" + "b" * 64
    )
    with pytest.raises(SystemExit, match="different A1 contracts"):
        _validate_a1_validation_manifest_corpus_binding(meta, contract)


def test_a1_memmap_auto_detection_forbids_manifest_bypass(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "row_offsets.dat").write_bytes(b"offsets")
    (corpus / "game_seed.dat").write_bytes(b"seeds")
    inventory = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(corpus.glob("*.dat"))
    ]
    meta = {
        "selected_game_seed_manifest": {},
        "a1_post_wave_audit": {},
        "columns": {"game_seed": {"kind": "fixed"}},
        "payload_inventory_schema": "memmap-payload-inventory-v1",
        "payload_inventory": inventory,
        "payload_inventory_sha256": _canonical_json_sha256(inventory),
    }
    (corpus / "corpus_meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="validation-game-seed-manifest is mandatory"):
        _preflight_a1_memmap_metadata(corpus, validation_manifest_path=None)
    assert (
        _preflight_a1_memmap_metadata(
            corpus, validation_manifest_path=tmp_path / "validation.json"
        )
        == meta
    )


def test_a1_memmap_preflight_rejects_post_build_payload_mutation(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    offsets = corpus / "row_offsets.dat"
    seeds = corpus / "game_seed.dat"
    offsets.write_bytes(b"offsets")
    seeds.write_bytes(b"seed0")
    inventory = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(corpus.glob("*.dat"))
    ]
    meta = {
        "selected_game_seed_manifest": {},
        "a1_post_wave_audit": {},
        "columns": {"game_seed": {"kind": "fixed"}},
        "payload_inventory_schema": "memmap-payload-inventory-v1",
        "payload_inventory": inventory,
        "payload_inventory_sha256": _canonical_json_sha256(inventory),
    }
    (corpus / "corpus_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    fingerprint = _training_data_fingerprint(str(corpus), "memmap")
    assert fingerprint == _canonical_json_sha256(
        {
            "corpus_meta_file_sha256": "sha256:"
            + hashlib.sha256((corpus / "corpus_meta.json").read_bytes()).hexdigest(),
            "payload_inventory_sha256": meta["payload_inventory_sha256"],
        }
    )
    _preflight_a1_memmap_metadata(
        corpus, validation_manifest_path=tmp_path / "validation.json"
    )

    seeds.write_bytes(b"seed1")  # same size: only content addressing can catch this.
    with pytest.raises(SystemExit, match=r"game_seed\.dat sha256 mismatch"):
        _preflight_a1_memmap_metadata(
            corpus, validation_manifest_path=tmp_path / "validation.json"
        )


def test_a1_artifact_chain_replays_actual_seed_set_and_learner_objective(
    tmp_path: Path,
) -> None:
    meta, validation_contract, seeds = _write_a1_chain(tmp_path)
    _validate_a1_validation_manifest_corpus_binding(meta, validation_contract)

    bound = _validate_a1_corpus_artifacts_and_seeds(
        meta, validation_contract, seeds
    )
    assert bound["learner_code_sha256"].startswith("sha256:")
    assert bound["runtime_code_tree_sha256"].startswith("sha256:")
    recipe = dict(bound["learner_training_recipe"])
    args = argparse.Namespace(
        **{
            key: value
            for key, value in recipe.items()
            if key not in {"world_size", "global_batch_size"}
        },
        value_head_type="mse",
        value_categorical_bins=0,
        value_hlgauss_sigma_ratio=0.75,
        init_checkpoint_sha256=bound["producer_checkpoint_sha256"],
    )
    _validate_a1_learner_objective(args, bound)
    ddp = {"world_size": 1, "rank": 0, "local_rank": 0, "enabled": False}
    assert _validate_a1_learner_training_recipe(args, ddp, bound) == recipe
    assert bound["decisive_training_semantics"] == {
        "schema_version": "a1-decisive-training-semantics-v1",
        "decisive": True,
        "diagnostic_authority_present": False,
        "world_size": 1,
        "grad_accum_steps": 1,
        "gradient_accumulation_contract": "single_microbatch_exact",
        "symmetry_augmentation": False,
        "distributed_symmetry_contract": "not_applicable",
        "advantage_policy_weighting": "none",
        "distributed_advantage_contract": "not_applicable",
    }

    with pytest.raises(SystemExit, match="unexpected=1"):
        tampered_seeds = seeds.copy()
        tampered_seeds[0] = 999_999
        _validate_a1_corpus_artifacts_and_seeds(
            meta,
            validation_contract,
            tampered_seeds,
        )


    args.value_head_type = "hlgauss"
    with pytest.raises(SystemExit, match="learner objective differs"):
        _validate_a1_learner_objective(args, bound)
    args.value_head_type = "mse"
    args.epochs = 2
    with pytest.raises(SystemExit, match=r"learner training recipe differs.*epochs"):
        _validate_a1_learner_training_recipe(args, ddp, bound)

    args.epochs = 1
    args.a1_contract_sha256 = validation_contract["a1_contract_sha256"]
    args.a1_selected_game_seed_set_sha256 = bound[
        "selected_game_seed_set_sha256"
    ]
    args.a1_training_game_seed_set_sha256 = bound[
        "training_game_seed_set_sha256"
    ]
    args.a1_learner_training_recipe_sha256 = bound[
        "learner_training_recipe_sha256"
    ]
    args.a1_memmap_payload_inventory_sha256 = "sha256:" + "9" * 64
    args.a1_learner_code_sha256 = bound["learner_code_sha256"]
    args.a1_runtime_code_tree_sha256 = bound["runtime_code_tree_sha256"]
    provenance = _value_training_metadata(
        args,
        scalar_weight=0.25,
        categorical_weight=0.0,
        categorical_bins=0,
        optimizer_steps=3,
        completed_epochs=1,
        scalar_training_weight_sum=1.0,
        categorical_training_weight_sum=0.0,
    )
    assert provenance["a1_contract_sha256"] == validation_contract[
        "a1_contract_sha256"
    ]
    assert provenance["a1_learner_training_recipe_sha256"] == bound[
        "learner_training_recipe_sha256"
    ]
    assert provenance["a1_memmap_payload_inventory_sha256"] == (
        args.a1_memmap_payload_inventory_sha256
    )
    assert provenance["a1_learner_code_sha256"] == bound[
        "learner_code_sha256"
    ]
    assert provenance["a1_runtime_code_tree_sha256"] == bound[
        "runtime_code_tree_sha256"
    ]


def test_a1_artifact_chain_accepts_authenticated_relocated_v3_audit(
    tmp_path: Path,
) -> None:
    meta, validation_contract, seeds = _write_a1_chain(tmp_path)
    audit_meta = meta["a1_post_wave_audit"]
    audit_path = Path(audit_meta["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["schema_version"] = "a1-post-wave-audit-v3"
    relocation = {
        "path": str((tmp_path / "relocation.json").resolve()),
        "file_sha256": "sha256:" + "a" * 64,
        "relocation_sha256": "sha256:" + "a" * 64,
        "render_sha256": "sha256:" + "b" * 64,
        "job_identities_sha256": "sha256:" + "c" * 64,
        "file_inventory_sha256": "sha256:" + "d" * 64,
    }
    audit["harvest_relocation"] = relocation
    audit.pop("audit_sha256")
    audit["audit_sha256"] = _canonical_json_sha256(audit)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    audit_meta["file_sha256"] = (
        "sha256:" + hashlib.sha256(audit_path.read_bytes()).hexdigest()
    )
    audit_meta["audit_sha256"] = audit["audit_sha256"]
    audit_meta["harvest_relocation"] = dict(relocation)

    bound = _validate_a1_corpus_artifacts_and_seeds(
        meta, validation_contract, seeds
    )

    assert bound["producer_checkpoint_sha256"].startswith("sha256:")

    audit_meta["harvest_relocation"]["render_sha256"] = "sha256:" + "e" * 64
    with pytest.raises(SystemExit, match="relocated post-wave audit binding"):
        _validate_a1_corpus_artifacts_and_seeds(
            meta, validation_contract, seeds
        )


def test_a1_artifact_chain_rejects_learner_code_drift_before_training(
    tmp_path: Path,
) -> None:
    meta, validation_contract, seeds = _write_a1_chain(tmp_path)
    train_code = tmp_path / "learner" / "tools" / "train_bc.py"
    original = train_code.read_bytes()
    train_code.write_bytes(b"!" + original[1:])
    with pytest.raises(SystemExit, match="learner implementation drift"):
        _validate_a1_corpus_artifacts_and_seeds(
            meta, validation_contract, seeds
        )


def _decisive_semantics_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "grad_accum_steps": 1,
        "symmetry_augment": False,
        "advantage_policy_weighting": "none",
        "a1_batch_probe_plan": "",
        "a1_batch_probe_run_id": "",
        "a1_learner_ablation_id": "",
        "a1_dual_learner_lock": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grad_accum_steps": 2}, "union-weighted gradient accumulation"),
        (
            {"advantage_policy_weighting": "outcome_value"},
            "global DDP normalization",
        ),
    ],
)
def test_decisive_distributed_a1_rejects_unsealed_semantics(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        _validate_a1_decisive_training_semantics(
            _decisive_semantics_args(**overrides),
            {"world_size": 8, "enabled": True},
            {},
        )


def test_decisive_distributed_a1_binds_global_rank_strided_symmetry_stream() -> None:
    contract = _validate_a1_decisive_training_semantics(
        _decisive_semantics_args(symmetry_augment=True),
        {"world_size": 8, "enabled": True},
        {},
    )
    assert contract["decisive"] is True
    assert (
        contract["distributed_symmetry_contract"]
        == "global_rank_strided_stream_exact_v2"
    )


def test_explicit_a1_diagnostic_authority_records_but_does_not_promote_unsafe_knobs(
) -> None:
    contract = _validate_a1_decisive_training_semantics(
        _decisive_semantics_args(
            grad_accum_steps=2,
            symmetry_augment=True,
            advantage_policy_weighting="outcome_value",
            a1_learner_ablation_id="diagnostic-only",
        ),
        {"world_size": 8, "enabled": True},
        {},
    )
    assert contract["decisive"] is False
    assert contract["diagnostic_authority_present"] is True
    assert contract["gradient_accumulation_contract"] == (
        "diagnostic_approximate_microbatch_means"
    )
    assert (
        contract["distributed_symmetry_contract"]
        == "global_rank_strided_stream_exact_v2"
    )
    assert contract["distributed_advantage_contract"] == (
        "global_normalization_unsealed_for_a1"
    )
    assert _a1_report_eligibility_from_training_semantics(
        contract,
        diagnostic_only=False,
        promotion_eligible=True,
    ) == (True, False)


def test_exact_single_microbatch_semantics_preserve_source_eligibility() -> None:
    contract = _validate_a1_decisive_training_semantics(
        _decisive_semantics_args(),
        {"world_size": 8, "enabled": True},
        {},
    )
    assert _a1_report_eligibility_from_training_semantics(
        contract,
        diagnostic_only=False,
        promotion_eligible=True,
    ) == (False, True)


def test_four_way_conditional_mean_accumulation_can_attenuate_signal_75_percent() -> None:
    # One sparse labeled microbatch followed by three empty ones. The current
    # approximate operator divides every microbatch mean by grad_accum_steps:
    # (g + 0 + 0 + 0) / 4. The union-weighted conditional mean is g / 1.
    microbatch_gradients = np.asarray([1.0, 0.0, 0.0, 0.0])
    approximate = float(microbatch_gradients.mean())
    union_weighted = 1.0
    assert approximate == pytest.approx(0.25)
    assert 1.0 - approximate / union_weighted == pytest.approx(0.75)


@pytest.mark.parametrize(
    "suffix",
    [
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/action_mask.py",
        "src/catan_zero/search/rust_mcts.py",
        "tools/launcher_guards.py",
    ],
)
def test_a1_artifact_chain_rejects_transitive_runtime_drift(
    tmp_path: Path, suffix: str
) -> None:
    meta, validation_contract, seeds = _write_a1_chain(tmp_path)
    runtime_path = tmp_path / "runtime" / suffix
    original = runtime_path.read_bytes()
    replacement = b"!" if original[:1] != b"!" else b"#"
    runtime_path.write_bytes(replacement + original[1:])
    with pytest.raises(SystemExit, match="transitive runtime drift"):
        _validate_a1_corpus_artifacts_and_seeds(
            meta, validation_contract, seeds
        )
