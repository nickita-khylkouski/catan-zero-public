from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import train_bc
from tools.mixed_memmap_corpus import ConcatMemmapCorpus


def _sha(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _canonical(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(raw.encode()).hexdigest()}"


def _component(root: Path, name: str) -> dict:
    corpus = root / name
    corpus.mkdir()
    (corpus / "row_offsets.dat").write_bytes(
        np.asarray([0, 0, 0], dtype="<i8").tobytes()
    )
    (corpus / "game_seed.dat").write_bytes(np.asarray([1, 2], dtype="<i8").tobytes())
    inventory = []
    for filename in ("game_seed.dat", "row_offsets.dat"):
        path = corpus / filename
        inventory.append(
            {
                "filename": filename,
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": 2,
        "legal_width": 1,
        "flat_count": 0,
        "columns": {"game_seed": {"kind": "fixed", "dtype": "<i8", "inner_shape": []}},
        "payload_inventory_schema": "memmap-payload-inventory-v1",
        "payload_inventory": inventory,
        "payload_inventory_sha256": _canonical(inventory),
        "selected_game_seed_manifest": {"a1_contract_sha256": "sha256:" + "1" * 64},
        "a1_post_wave_audit": {"contract_sha256": "sha256:" + "1" * 64},
    }
    meta_path = corpus / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    validation = root / f"{name}.validation.json"
    validation.write_text("{}\n", encoding="utf-8")
    return {
        "corpus_dir": str(corpus.resolve()),
        "corpus_meta_sha256": _sha(meta_path),
        "payload_inventory_sha256": meta["payload_inventory_sha256"],
        "validation_manifest": str(validation.resolve()),
        "validation_manifest_sha256": _sha(validation),
    }


def _descriptor(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "memmap_composite_v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "components": [_component(tmp_path, "a"), _component(tmp_path, "b")],
    }
    path = tmp_path / "composite.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_descriptor_authenticates_ordered_component_bytes(tmp_path):
    path = _descriptor(tmp_path)
    verified = train_bc._preflight_memmap_composite_descriptor(path)
    assert verified["schema_version"] == "memmap_composite_v1"
    assert verified["diagnostic_only"] is True
    assert verified["promotion_eligible"] is False
    assert [Path(item["corpus_dir"]).name for item in verified["components"]] == [
        "a",
        "b",
    ]
    assert verified["descriptor_fingerprint"] == train_bc._training_data_fingerprint(
        str(path), "memmap"
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("corpus_meta_sha256", "metadata hash mismatch"),
        ("payload_inventory_sha256", "payload inventory hash mismatch"),
        ("validation_manifest_sha256", "validation manifest hash mismatch"),
    ],
)
def test_descriptor_refuses_component_binding_drift(tmp_path, field, message):
    path = _descriptor(tmp_path)
    payload = json.loads(path.read_text())
    payload["components"][1][field] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match=message):
        train_bc._preflight_memmap_composite_descriptor(path)


def test_descriptor_cannot_claim_promotion_eligibility(tmp_path):
    path = _descriptor(tmp_path)
    payload = json.loads(path.read_text())
    payload["promotion_eligible"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="diagnostic-only"):
        train_bc._preflight_memmap_composite_descriptor(path)


def test_validation_contract_unions_disjoint_component_seeds(monkeypatch):
    meta = {
        "descriptor_path": "/tmp/composite.json",
        "descriptor_file_sha256": "sha256:" + "a" * 64,
        "components": [
            {
                "validation_manifest": "/tmp/a.json",
                "validation_manifest_sha256": "sha256:" + "b" * 64,
                "corpus_meta": {},
            },
            {
                "validation_manifest": "/tmp/b.json",
                "validation_manifest_sha256": "sha256:" + "c" * 64,
                "corpus_meta": {},
            },
        ],
    }

    def load(path, **_kwargs):
        second = str(path).endswith("b.json")
        seeds = np.asarray([20, 21] if second else [10, 11], dtype=np.int64)
        return {
            "path": Path(path),
            "file_sha256": "sha256:" + ("c" if second else "b") * 64,
            "manifest_sha256": "sha256:" + ("e" if second else "d") * 64,
            "a1_contract_sha256": "sha256:" + ("2" if second else "1") * 64,
            "validation_row_count": 4 if second else 3,
            "validation_game_seed_set_sha256": train_bc._game_seed_set_sha256(seeds),
            "game_seeds": seeds,
        }

    monkeypatch.setattr(
        train_bc, "_load_validation_game_seed_manifest_for_training", load
    )
    monkeypatch.setattr(
        train_bc, "_validate_a1_validation_manifest_corpus_binding", lambda *_: None
    )
    contract = train_bc._load_composite_validation_contract(
        meta,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seed_ranges=[],
    )
    assert np.array_equal(contract["game_seeds"], [10, 11, 20, 21])
    assert contract["validation_row_count"] == 7
    assert contract["diagnostic_only"] is True


def test_validation_contract_refuses_overlapping_games(monkeypatch):
    meta = {
        "descriptor_path": "/tmp/composite.json",
        "descriptor_file_sha256": "sha256:" + "a" * 64,
        "components": [
            {
                "validation_manifest": f"/tmp/{name}.json",
                "validation_manifest_sha256": "sha256:" + digit * 64,
                "corpus_meta": {},
            }
            for name, digit in (("a", "b"), ("b", "c"))
        ],
    }

    def load(path, **_kwargs):
        second = str(path).endswith("b.json")
        seeds = np.asarray([11, 12] if second else [10, 11], dtype=np.int64)
        return {
            "path": Path(path),
            "file_sha256": "sha256:" + ("c" if second else "b") * 64,
            "manifest_sha256": "sha256:" + "d" * 64,
            "a1_contract_sha256": "sha256:" + "1" * 64,
            "validation_row_count": 2,
            "validation_game_seed_set_sha256": train_bc._game_seed_set_sha256(seeds),
            "game_seeds": seeds,
        }

    monkeypatch.setattr(
        train_bc, "_load_validation_game_seed_manifest_for_training", load
    )
    monkeypatch.setattr(
        train_bc, "_validate_a1_validation_manifest_corpus_binding", lambda *_: None
    )
    with pytest.raises(SystemExit, match="not disjoint"):
        train_bc._load_composite_validation_contract(
            meta,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=0,
            validation_game_seed_ranges=[],
        )


class _TinyCorpus:
    def __init__(self, values):
        array = np.asarray(values, dtype=np.int64)
        self.row_count = len(array)
        self.legal_width = 1
        self._columns = {"row": {"kind": "fixed", "dtype": "<i8", "inner_shape": []}}
        self._eager = {"row": array}
        self._lazy = {}
        self.meta = {}
        self.stats = {}

    def keys(self):
        return ["row"]

    def __getitem__(self, key):
        return self._eager[key]


def test_global_epoch_shuffle_interleaves_component_rows_with_prefetch():
    data = ConcatMemmapCorpus([_TinyCorpus([0, 1, 2]), _TinyCorpus([10, 11, 12])])
    # Positions deliberately alternate across the component boundary. The
    # iterator must preserve this one global order, not emit each corpus in turn.
    order = np.asarray([0, 3, 1, 4, 2, 5], dtype=np.int64)
    batches = list(
        train_bc._iterate_training_batches(
            data,
            order,
            np.arange(6, dtype=np.int64),
            2,
            np.ones(6, dtype=np.float32),
            np.ones(6, dtype=np.float32),
            num_workers=1,
            prefetch=2,
        )
    )
    observed = [data_part["row"][batch].tolist() for data_part, batch, _, _ in batches]
    assert observed == [[0, 10], [1, 11], [2, 12]]
