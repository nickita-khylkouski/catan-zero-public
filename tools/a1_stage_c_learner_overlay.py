#!/usr/bin/env python3
"""Export and materialize a Stage-C coherent-policy learner overlay.

Stage C emits a sparse, authenticated patch keyed by absolute corpus row.  The
learner consumes a normal ``MemmapCorpus``.  This tool joins those two ABIs
without copying the large observation/entity payloads and, critically, without
letting historical policy targets remain active:

* every base row remains available to the terminal-outcome/value objective;
* policy weight and policy tensors are zero for every non-reanalysed row;
* qualified rows receive the coherent-n128 target, prior and score evidence;
* all other memmap payloads are hard-linked byte-for-byte from the base corpus.

``export`` runs beside the completed Stage-C merge, where the full receipt DAG
is still replayable.  It creates a portable content-addressed bundle.  After
that bundle is copied host-to-host, ``materialize`` binds it to the exact base
corpus and emits a normal authenticated memmap plus a derived diagnostic
admission that the existing one-dose learner can consume unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools import a1_b200_active_policy_campaign as active_campaign  # noqa: E402
from tools import a1_stage_c_reanalysis_executor as stage_c  # noqa: E402
from tools import a1_stage_c_teacher_alignment as alignment  # noqa: E402
from tools import train_bc  # noqa: E402


EXPORT_SCHEMA = "a1-stage-c-learner-overlay-export-v1"
MATERIALIZATION_SCHEMA = "a1-stage-c-policy-overlay-materialization-v1"
ADMISSION_OVERLAY_SCHEMA = "a1-stage-c-policy-overlay-admission-binding-v1"
POLICY_TEACHER = "stage_c_coherent_n128_reanalysis"
REWRITTEN_COLUMNS = frozenset(
    {
        "policy_weight_multiplier",
        "prior_policy",
        "target_policy",
        "target_policy_mask",
        "target_scores",
        "target_scores_mask",
        "teacher_name",
    }
)
POLICY_RAGGED_COLUMNS = {
    "prior_policy": ("prior_policy_flat", 0.0),
    "target_policy": ("target_policy_flat", 0.0),
    "target_policy_mask": ("target_policy_mask_flat", False),
    "target_scores": ("target_scores_flat", np.nan),
    "target_scores_mask": ("target_scores_mask_flat", False),
}


class OverlayError(RuntimeError):
    """A Stage-C export, base corpus, or derived overlay is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise OverlayError(f"{where} must be a regular file: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OverlayError(f"cannot read {where}: {error}") from error
    if not isinstance(payload, dict):
        raise OverlayError(f"{where} must contain one JSON object")
    return resolved, payload


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise OverlayError(f"immutable output is not a file: {destination}")
        if destination.read_text(encoding="utf-8") != rendered:
            raise OverlayError(f"immutable output already exists with drift: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _copy_immutable(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise OverlayError(f"immutable bundle path is not a file: {destination}")
        if _file_sha256(destination) != _file_sha256(source):
            raise OverlayError(f"immutable bundle path already differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _export(args: argparse.Namespace) -> dict[str, Any]:
    try:
        merge = stage_c._verify_merge_receipt(args.merge_receipt)  # noqa: SLF001
        plan = alignment._verify_plan(  # noqa: SLF001
            Path(str(merge["stage_c_plan"]["path"]))
        )
        _overlay_path, eligibility = alignment._load_json(  # noqa: SLF001
            Path(str(plan["eligibility_overlay"]["path"])),
            where="Stage-C eligibility overlay",
        )
    except (stage_c.ExecutorError, alignment.AlignmentError, OSError) as error:
        raise OverlayError(f"Stage-C merge export refused: {error}") from error

    base_root = Path(str(eligibility["corpus"]["path"])).resolve(strict=True)
    meta_path, meta = _load_json(base_root / "corpus_meta.json", where="base corpus metadata")
    if (
        _file_sha256(meta_path)
        != eligibility["corpus"]["corpus_meta_file_sha256"]
        or meta.get("payload_inventory_sha256")
        != eligibility["corpus"]["payload_inventory_sha256"]
    ):
        raise OverlayError("Stage-C source corpus metadata drifted before export")

    output = args.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    patch_source = Path(str(merge["artifact"]["path"])).resolve(strict=True)
    merge_source = args.merge_receipt.expanduser().resolve(strict=True)
    patch_path = output / "stage_c_target_patch.npz"
    merge_path = output / "source_merge_receipt.json"
    _copy_immutable(patch_source, patch_path)
    _copy_immutable(merge_source, merge_path)

    with np.load(patch_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    stage_c._verify_patch_arrays(arrays, receipt=merge)  # noqa: SLF001
    identities = [
        {
            "row_index": int(row),
            "game_seed": int(seed),
            "decision_index": int(decision),
            "identity_sha256": str(identity),
        }
        for row, seed, decision, identity in zip(
            arrays["row_index"],
            arrays["game_seed"],
            arrays["decision_index"],
            arrays["identity_sha256"],
            strict=True,
        )
    ]
    manifest: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "source_merge_receipt": {
            **_artifact(merge_path),
            "receipt_sha256": merge["receipt_sha256"],
            "schema_version": merge["schema_version"],
        },
        "source_stage_c_plan": copy.deepcopy(merge["stage_c_plan"]),
        "source_corpus": {
            "corpus_meta_file_sha256": _file_sha256(meta_path),
            "payload_inventory_sha256": meta["payload_inventory_sha256"],
            "row_count": int(meta["row_count"]),
            "flat_count": int(meta["flat_count"]),
            "legal_width": int(meta["legal_width"]),
        },
        "target_policy_target_identity_sha256": merge[
            "target_policy_target_identity_sha256"
        ],
        "target_reanalyzer_checkpoint": copy.deepcopy(
            merge["target_reanalyzer_checkpoint"]
        ),
        "target_operator_contract": copy.deepcopy(merge["target_operator_contract"]),
        "patch": _artifact(patch_path),
        "counts": copy.deepcopy(merge["counts"]),
        "row_identity_sha256": _value_sha256(identities),
        "learner_projection": {
            "policy_rows": "exact_stage_c_reanalysed_rows_only",
            "nonselected_policy_weight": 0.0,
            "selected_policy_weight": 1.0,
            "base_value_rows_retained": True,
            "root_value_patch_consumed": False,
            "completed_q_patch_consumed": False,
            "rewritten_columns": sorted(REWRITTEN_COLUMNS),
        },
    }
    manifest["export_sha256"] = _value_sha256(manifest)
    _write_json_immutable(output / "manifest.json", manifest)
    return manifest


def _load_export(path: Path) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    manifest_path, manifest = _load_json(path, where="Stage-C learner export")
    unsigned = dict(manifest)
    stated = unsigned.pop("export_sha256", None)
    if (
        manifest.get("schema_version") != EXPORT_SCHEMA
        or manifest.get("diagnostic_only") is not True
        or manifest.get("promotion_eligible") is not False
        or stated != _value_sha256(unsigned)
        or manifest.get("learner_projection", {}).get("rewritten_columns")
        != sorted(REWRITTEN_COLUMNS)
    ):
        raise OverlayError("Stage-C learner export schema/digest/semantics drifted")
    patch_ref = manifest.get("patch")
    merge_ref = manifest.get("source_merge_receipt")
    if not isinstance(patch_ref, dict) or not isinstance(merge_ref, dict):
        raise OverlayError("Stage-C learner export artifact bindings are malformed")
    patch = manifest_path.parent / Path(str(patch_ref.get("path", ""))).name
    merge_path = manifest_path.parent / Path(str(merge_ref.get("path", ""))).name
    for artifact_path, reference, where in (
        (patch, patch_ref, "target patch"),
        (merge_path, merge_ref, "merge receipt"),
    ):
        if (
            artifact_path.is_symlink()
            or not artifact_path.is_file()
            or _file_sha256(artifact_path) != reference.get("file_sha256")
            or artifact_path.stat().st_size != int(reference.get("size_bytes", -1))
        ):
            raise OverlayError(f"Stage-C exported {where} bytes drifted")
    _merge_path, merge = _load_json(merge_path, where="exported Stage-C merge receipt")
    if (
        merge.get("schema_version") != stage_c.MERGE_RECEIPT_SCHEMA
        or merge.get("receipt_sha256") != merge_ref.get("receipt_sha256")
        or merge.get("artifact", {}).get("file_sha256")
        != patch_ref.get("file_sha256")
    ):
        raise OverlayError("exported Stage-C merge binding drifted")
    with np.load(patch, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    stage_c._verify_patch_arrays(arrays, receipt=merge)  # noqa: SLF001
    return manifest_path, manifest, arrays


def _column_payload_filename(name: str, schema: Mapping[str, Any]) -> str | None:
    kind = schema.get("kind")
    if kind == "implicit_constant":
        return None
    return f"{name}.codes.dat" if kind == "string" else f"{name}.dat"


def _sha_record(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _hardlink_payloads(
    base_root: Path,
    output_root: Path,
    columns: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    linked: set[str] = {"row_offsets.dat"}
    for name, schema in columns.items():
        filename = _column_payload_filename(name, schema)
        if filename is not None and name not in REWRITTEN_COLUMNS:
            linked.add(filename)
    for filename in sorted(linked):
        source = base_root / filename
        destination = output_root / filename
        try:
            os.link(source, destination)
        except OSError as error:
            raise OverlayError(
                f"cannot hard-link immutable base payload {source} -> {destination}: {error}"
            ) from error
    return linked


def _fixed_memmap(
    root: Path, name: str, schema: Mapping[str, Any], rows: int, *, mode: str
) -> np.memmap:
    if schema.get("kind") != "fixed":
        raise OverlayError(f"required overlay column {name!r} is not fixed")
    inner = tuple(int(value) for value in schema.get("inner_shape", ()))
    return np.memmap(
        root / f"{name}.dat",
        dtype=np.dtype(str(schema["dtype"])),
        mode=mode,
        shape=(rows, *inner),
    )


def _ragged_flat_memmap(
    root: Path, name: str, schema: Mapping[str, Any], flat_count: int, *, mode: str
) -> np.memmap:
    if schema.get("kind") != "ragged2d":
        raise OverlayError(f"required overlay column {name!r} is not ragged2d")
    return np.memmap(
        root / f"{name}.dat",
        dtype=np.dtype(str(schema["dtype"])),
        mode=mode,
        shape=(flat_count,),
    )


def _project_policy_patch(
    *,
    base_root: Path,
    output_root: Path,
    meta: dict[str, Any],
    patch: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Write the seven policy-only payloads and return projection evidence."""

    columns = meta.get("columns")
    if not isinstance(columns, dict) or not REWRITTEN_COLUMNS <= set(columns):
        raise OverlayError(
            "base corpus lacks required Stage-C policy projection columns: "
            f"{sorted(REWRITTEN_COLUMNS - set(columns or {}))}"
        )
    rows = int(meta["row_count"])
    flat_count = int(meta["flat_count"])
    offsets = np.memmap(
        base_root / "row_offsets.dat",
        dtype=np.int64,
        mode="r",
        shape=(rows + 1,),
    )
    if int(offsets[0]) != 0 or int(offsets[-1]) != flat_count:
        raise OverlayError("base corpus row offsets drifted")
    selected_rows = np.asarray(patch["row_index"], dtype=np.int64)
    if (
        selected_rows.ndim != 1
        or selected_rows.size == 0
        or np.unique(selected_rows).size != selected_rows.size
        or np.any(selected_rows < 0)
        or np.any(selected_rows >= rows)
    ):
        raise OverlayError("Stage-C patch row indices are invalid for base corpus")

    base_seed = _fixed_memmap(base_root, "game_seed", columns["game_seed"], rows, mode="r")
    base_decision = _fixed_memmap(
        base_root, "decision_index", columns["decision_index"], rows, mode="r"
    )
    if not np.array_equal(
        np.asarray(base_seed[selected_rows], dtype=np.int64).reshape(-1),
        np.asarray(patch["game_seed"], dtype=np.int64),
    ) or not np.array_equal(
        np.asarray(base_decision[selected_rows], dtype=np.int64).reshape(-1),
        np.asarray(patch["decision_index"], dtype=np.int64),
    ):
        raise OverlayError("Stage-C patch row seed/decision identity differs from base")

    legal_schema = columns.get("legal_action_ids")
    if not isinstance(legal_schema, dict) or legal_schema.get("kind") != "ragged2d":
        raise OverlayError("base legal_action_ids is not a ragged2d column")
    legal_flat = np.memmap(
        base_root / "legal_action_ids.dat",
        dtype=np.dtype(str(legal_schema["dtype"])),
        mode="r",
        shape=(flat_count,),
    )

    policy_weight = _fixed_memmap(
        output_root,
        "policy_weight_multiplier",
        columns["policy_weight_multiplier"],
        rows,
        mode="w+",
    )
    policy_weight[...] = 0
    policy_weight[selected_rows] = 1
    policy_weight.flush()

    outputs: dict[str, np.memmap] = {}
    for name, (_patch_name, fill) in POLICY_RAGGED_COLUMNS.items():
        output = _ragged_flat_memmap(
            output_root, name, columns[name], flat_count, mode="w+"
        )
        output[...] = fill
        outputs[name] = output

    patch_offsets = np.asarray(patch["legal_action_offsets"], dtype=np.int64)
    patch_legal = np.asarray(patch["legal_action_ids_flat"], dtype=np.int64)
    for ordinal, row in enumerate(selected_rows.tolist()):
        base_start, base_stop = int(offsets[row]), int(offsets[row + 1])
        patch_start = int(patch_offsets[ordinal])
        patch_stop = int(patch_offsets[ordinal + 1])
        base_ids = np.asarray(legal_flat[base_start:base_stop], dtype=np.int64)
        patch_ids = patch_legal[patch_start:patch_stop]
        if (
            base_ids.size != patch_ids.size
            or np.unique(base_ids).size != base_ids.size
            or set(base_ids.tolist()) != set(patch_ids.tolist())
        ):
            raise OverlayError(
                f"Stage-C legal action set differs at corpus row {row}"
            )
        patch_position = {int(action): index for index, action in enumerate(patch_ids)}
        gather = np.asarray([patch_position[int(action)] for action in base_ids], dtype=np.int64)
        for name, (patch_name, _fill) in POLICY_RAGGED_COLUMNS.items():
            source = np.asarray(patch[patch_name])[patch_start:patch_stop]
            outputs[name][base_start:base_stop] = source[gather]
    for output in outputs.values():
        output.flush()

    teacher_schema = columns["teacher_name"]
    if teacher_schema.get("kind") != "string":
        raise OverlayError("base teacher_name is not dictionary encoded")
    categories = [str(value) for value in teacher_schema.get("categories", ())]
    if POLICY_TEACHER not in categories:
        categories.append(POLICY_TEACHER)
    teacher_code = categories.index(POLICY_TEACHER)
    source_codes = np.memmap(
        base_root / "teacher_name.codes.dat",
        dtype=np.int32,
        mode="r",
        shape=(rows,),
    )
    target_codes = np.memmap(
        output_root / "teacher_name.codes.dat",
        dtype=np.int32,
        mode="w+",
        shape=(rows,),
    )
    target_codes[...] = source_codes
    target_codes[selected_rows] = teacher_code
    target_codes.flush()
    teacher_schema["categories"] = categories

    target_policy = outputs["target_policy"]
    target_mask = outputs["target_policy_mask"]
    selected_mass = np.asarray(
        [
            float(np.asarray(target_policy[int(offsets[row]) : int(offsets[row + 1])]).sum())
            for row in selected_rows
        ]
    )
    if not np.allclose(selected_mass, 1.0, rtol=0.0, atol=1.0e-5):
        raise OverlayError("materialized Stage-C target policies are not normalized")
    if int(np.count_nonzero(np.asarray(target_mask))) != int(
        np.count_nonzero(np.asarray(patch["target_policy_mask_flat"]))
    ):
        raise OverlayError("materialized Stage-C target mask lost support")

    return {
        "selected_rows": int(selected_rows.size),
        "nonselected_policy_disabled_rows": rows - int(selected_rows.size),
        "selected_row_index_sha256": _value_sha256(selected_rows.tolist()),
        "selected_policy_mass_min": float(selected_mass.min()),
        "selected_policy_mass_max": float(selected_mass.max()),
        "base_value_rows_retained": rows,
    }


def _updated_inventory(
    *,
    base_meta: Mapping[str, Any],
    output_root: Path,
    rewritten_filenames: set[str],
) -> list[dict[str, Any]]:
    base_records = {
        str(record["filename"]): dict(record)
        for record in base_meta.get("payload_inventory", ())
    }
    expected = train_bc._expected_memmap_payload_filenames(base_meta)  # noqa: SLF001
    if set(base_records) != expected:
        raise OverlayError("base payload inventory differs from its column schema")
    result = []
    for filename in sorted(expected):
        path = output_root / filename
        if filename in rewritten_filenames:
            result.append(_sha_record(path))
        else:
            record = base_records[filename]
            if path.stat().st_size != int(record["size_bytes"]):
                raise OverlayError(f"hard-linked payload size drifted: {filename}")
            result.append(record)
    return result


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    export_path, export, patch = _load_export(args.export_manifest)
    try:
        base_admission_path, base_admission = active_campaign._load_admission(  # noqa: SLF001
            args.base_admission
        )
    except active_campaign.CampaignError as error:
        raise OverlayError(f"base coherent admission refused: {error}") from error
    base_root = args.base_corpus.expanduser().resolve(strict=True)
    base_meta_path, base_meta = _load_json(
        base_root / "corpus_meta.json", where="base corpus metadata"
    )
    source_binding = export["source_corpus"]
    if (
        Path(str(base_admission["corpus"]["data_path"])).resolve(strict=True)
        != base_root
        or _file_sha256(base_meta_path) != source_binding["corpus_meta_file_sha256"]
        or base_meta.get("payload_inventory_sha256")
        != source_binding["payload_inventory_sha256"]
        or int(base_meta.get("row_count", -1)) != int(source_binding["row_count"])
        or int(base_meta.get("flat_count", -1)) != int(source_binding["flat_count"])
    ):
        raise OverlayError("portable Stage-C export binds a different base corpus")

    output = args.output_root.expanduser().resolve(strict=False)
    if output.exists():
        raise OverlayError(f"overlay output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        meta = copy.deepcopy(base_meta)
        columns = meta.get("columns")
        if not isinstance(columns, dict):
            raise OverlayError("base corpus column schema is malformed")
        _hardlink_payloads(base_root, temporary, columns)
        projection = _project_policy_patch(
            base_root=base_root,
            output_root=temporary,
            meta=meta,
            patch=patch,
        )
        validation_ref = base_admission["corpus"]["validation_manifest"]
        try:
            validation = train_bc._load_validation_game_seed_manifest_for_training(  # noqa: SLF001
                Path(str(validation_ref["path"])),
                validation_fraction=0.05,
                validation_seed=17,
                validation_max_samples=0,
                validation_game_seed_ranges=[],
            )
        except SystemExit as error:
            raise OverlayError(
                f"base coherent validation manifest refused: {error}"
            ) from error
        selected_validation = np.isin(
            np.asarray(patch["game_seed"], dtype=np.int64),
            np.asarray(validation["game_seeds"], dtype=np.int64),
        )
        projection["selected_validation_policy_rows"] = int(
            np.count_nonzero(selected_validation)
        )
        projection["selected_training_policy_rows"] = int(
            len(selected_validation) - np.count_nonzero(selected_validation)
        )
        if projection["selected_training_policy_rows"] <= 0:
            raise OverlayError("Stage-C overlay has no policy roots in the training split")
        rewritten_filenames = {
            _column_payload_filename(name, columns[name])
            for name in REWRITTEN_COLUMNS
        }
        if None in rewritten_filenames:
            raise OverlayError("Stage-C rewritten column unexpectedly has no payload")
        inventory = _updated_inventory(
            base_meta=base_meta,
            output_root=temporary,
            rewritten_filenames={str(value) for value in rewritten_filenames},
        )
        inventory_sha = _value_sha256(inventory)
        meta["payload_inventory"] = inventory
        meta["payload_inventory_sha256"] = inventory_sha
        stats = meta.setdefault("stats", {})
        if isinstance(stats, dict):
            stats["policy_weight_zero_rows"] = int(meta["row_count"]) - int(
                projection["selected_rows"]
            )
            stats["stage_c_reanalysed_policy_rows"] = int(projection["selected_rows"])
        scan = meta.get("event_history_payload_scan")
        if isinstance(scan, dict):
            scan["payload_inventory_sha256"] = inventory_sha
            scan.pop("scan_sha256", None)
            scan["scan_sha256"] = _value_sha256(scan)
        meta["stage_c_policy_overlay"] = {
            "schema_version": ADMISSION_OVERLAY_SCHEMA,
            "export_sha256": export["export_sha256"],
            "target_policy_target_identity_sha256": export[
                "target_policy_target_identity_sha256"
            ],
            "selected_policy_rows": int(projection["selected_rows"]),
            "nonselected_policy_weight": 0.0,
            "selected_policy_weight": 1.0,
            "base_value_rows_retained": True,
            "rewritten_columns": sorted(REWRITTEN_COLUMNS),
        }
        meta_path = temporary / "corpus_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        final_meta = output / "corpus_meta.json"
        final_receipt = output / "stage_c_policy_overlay.receipt.json"
        receipt: dict[str, Any] = {
            "schema_version": MATERIALIZATION_SCHEMA,
            "diagnostic_only": True,
            "promotion_eligible": False,
            "export": {
                "path": str(export_path),
                "file_sha256": _file_sha256(export_path),
                "export_sha256": export["export_sha256"],
            },
            "base_admission": {
                "path": str(base_admission_path),
                "file_sha256": _file_sha256(base_admission_path),
                "admission_sha256": base_admission["admission_sha256"],
            },
            "base_corpus": {
                "path": str(base_root),
                "corpus_meta_file_sha256": _file_sha256(base_meta_path),
                "payload_inventory_sha256": base_meta["payload_inventory_sha256"],
            },
            "overlay_corpus": {
                "path": str(output),
                "corpus_meta_path": str(final_meta),
                "corpus_meta_file_sha256": _file_sha256(meta_path),
                "payload_inventory_sha256": inventory_sha,
                "row_count": int(meta["row_count"]),
                "flat_count": int(meta["flat_count"]),
            },
            "target_policy_target_identity_sha256": export[
                "target_policy_target_identity_sha256"
            ],
            "target_reanalyzer_checkpoint": copy.deepcopy(
                export["target_reanalyzer_checkpoint"]
            ),
            "target_operator_contract": copy.deepcopy(export["target_operator_contract"]),
            "projection": projection,
            "rewritten_columns": sorted(REWRITTEN_COLUMNS),
            "preserved_columns": sorted(set(columns) - REWRITTEN_COLUMNS),
            "non_target_source_columns_mutated": False,
            "base_value_and_outcome_columns_retained": True,
        }
        receipt["receipt_sha256"] = _value_sha256(receipt)
        _write_json_immutable(temporary / final_receipt.name, receipt)

        admission = copy.deepcopy(base_admission)
        admission.pop("admission_sha256", None)
        corpus = admission["corpus"]
        corpus.update(
            {
                "data_path": str(output),
                "corpus_meta_path": str(final_meta),
                "corpus_meta_file_sha256": _file_sha256(meta_path),
                "payload_inventory_sha256": inventory_sha,
                "stored_policy_target_distillation_eligible": True,
                "incompatible_policy_active_rows": 0,
            }
        )
        admission["policy_distillation_contract"].update(
            {
                "policy_active_rows": int(projection["selected_rows"]),
                "stage_c_reanalysis_only": True,
                "target_policy_target_identity_sha256": export[
                    "target_policy_target_identity_sha256"
                ],
            }
        )
        admission["stage_c_policy_overlay"] = {
            "schema_version": ADMISSION_OVERLAY_SCHEMA,
            "materialization_receipt": {
                "path": str(final_receipt),
                "file_sha256": _file_sha256(temporary / final_receipt.name),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            "export": receipt["export"],
            "target_policy_target_identity_sha256": export[
                "target_policy_target_identity_sha256"
            ],
            "selected_policy_rows": int(projection["selected_rows"]),
            "selected_training_policy_rows": int(
                projection["selected_training_policy_rows"]
            ),
            "selected_validation_policy_rows": int(
                projection["selected_validation_policy_rows"]
            ),
            "base_value_rows_retained": True,
            "historical_policy_targets_active": False,
        }
        admission["admission_sha256"] = _value_sha256(admission)
        _write_json_immutable(temporary / "overlay.admission.json", admission)
        os.replace(temporary, output)
        return {
            "receipt": str(final_receipt),
            "receipt_sha256": receipt["receipt_sha256"],
            "admission": str(output / "overlay.admission.json"),
            "admission_sha256": admission["admission_sha256"],
            "corpus": str(output),
            "selected_policy_rows": int(projection["selected_rows"]),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_overlay_admission(path: Path) -> dict[str, Any]:
    """Verify the portable Stage-C binding on a derived coherent admission."""

    admission_path, admission = _load_json(path, where="Stage-C overlay admission")
    unsigned = dict(admission)
    stated = unsigned.pop("admission_sha256", None)
    overlay = admission.get("stage_c_policy_overlay")
    if (
        admission.get("schema_version") != active_campaign.ADMISSION_SCHEMA
        or stated != _value_sha256(unsigned)
        or not isinstance(overlay, dict)
        or overlay.get("schema_version") != ADMISSION_OVERLAY_SCHEMA
        or overlay.get("historical_policy_targets_active") is not False
        or overlay.get("base_value_rows_retained") is not True
        or int(overlay.get("selected_policy_rows", 0)) <= 0
        or int(overlay.get("selected_training_policy_rows", 0)) <= 0
        or int(overlay.get("selected_validation_policy_rows", -1)) < 0
        or int(overlay.get("selected_training_policy_rows", 0))
        + int(overlay.get("selected_validation_policy_rows", 0))
        != int(overlay.get("selected_policy_rows", 0))
        or admission.get("policy_distillation_contract", {}).get(
            "stage_c_reanalysis_only"
        )
        is not True
    ):
        raise OverlayError("Stage-C overlay admission digest/semantics drifted")
    receipt_ref = overlay.get("materialization_receipt")
    if not isinstance(receipt_ref, dict):
        raise OverlayError("Stage-C overlay admission lost materialization receipt")
    receipt_path = Path(str(receipt_ref.get("path", ""))).resolve(strict=True)
    _receipt_path, receipt = _load_json(
        receipt_path, where="Stage-C overlay materialization receipt"
    )
    receipt_unsigned = dict(receipt)
    receipt_stated = receipt_unsigned.pop("receipt_sha256", None)
    if (
        receipt.get("schema_version") != MATERIALIZATION_SCHEMA
        or receipt_stated != _value_sha256(receipt_unsigned)
        or _file_sha256(receipt_path) != receipt_ref.get("file_sha256")
        or receipt_stated != receipt_ref.get("receipt_sha256")
        or receipt.get("target_policy_target_identity_sha256")
        != overlay.get("target_policy_target_identity_sha256")
    ):
        raise OverlayError("Stage-C overlay materialization binding drifted")
    corpus_root = Path(str(admission["corpus"]["data_path"])).resolve(strict=True)
    meta_path = corpus_root / "corpus_meta.json"
    if (
        _file_sha256(meta_path)
        != admission["corpus"]["corpus_meta_file_sha256"]
        or corpus_root != receipt_path.parent
        or receipt["overlay_corpus"]["payload_inventory_sha256"]
        != admission["corpus"]["payload_inventory_sha256"]
    ):
        raise OverlayError("Stage-C overlay admission differs from corpus bytes")
    return {
        "path": str(admission_path),
        "admission": admission,
        "receipt": receipt,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--merge-receipt", required=True, type=Path)
    export.add_argument("--output-root", required=True, type=Path)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--export-manifest", required=True, type=Path)
    materialize.add_argument("--base-corpus", required=True, type=Path)
    materialize.add_argument("--base-admission", required=True, type=Path)
    materialize.add_argument("--output-root", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--admission", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export":
            result = _export(args)
        elif args.command == "materialize":
            result = _materialize(args)
        else:
            result = verify_overlay_admission(args.admission)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OverlayError, stage_c.ExecutorError, OSError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
