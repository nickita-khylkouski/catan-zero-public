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
    overrides = {
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "equal",
    }
    payload = {
        "schema_version": "memmap_composite_v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "learner_recipe_overrides": overrides,
        "learner_recipe_overrides_sha256": _canonical(overrides),
        "components": [_component(tmp_path, "a"), _component(tmp_path, "b")],
    }
    path = tmp_path / "composite.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _descriptor_v2(
    tmp_path: Path,
    *,
    policy_distillation_component_ids: list[str] | None = None,
    policy_aux_phase_sampling_weights: dict[str, float] | None = None,
    value_training_component_ids: list[str] | None = None,
) -> Path:
    overrides = {
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "sqrt",
        "policy_kl_anchor_direction": "forward",
        "policy_kl_anchor_weight": 0.03,
    }
    components = []
    for name, ratio in (("n128", 0.57), ("n256", 0.23), ("gen3", 0.20)):
        components.append(
            {
                **_component(tmp_path, name),
                "component_id": name,
                "game_sampling_ratio": ratio,
            }
        )
    payload = {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "learner_recipe_overrides": overrides,
        "learner_recipe_overrides_sha256": _canonical(overrides),
        "policy_kl_anchor_component_ids": ["gen3"],
        "components": components,
    }
    if policy_distillation_component_ids is not None:
        payload["policy_distillation_component_ids"] = policy_distillation_component_ids
    if policy_aux_phase_sampling_weights is not None:
        payload["policy_aux_phase_sampling_weights"] = (
            policy_aux_phase_sampling_weights
        )
    if value_training_component_ids is not None:
        payload["value_training_component_ids"] = value_training_component_ids
    path = tmp_path / "composite-v2.json"
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


def test_v2_authenticates_three_component_ratios_and_anchor_scope(tmp_path):
    path = _descriptor_v2(tmp_path)
    verified = train_bc._preflight_memmap_composite_descriptor(path)
    assert verified["schema_version"] == "memmap_composite_v2"
    assert verified["component_ids"] == ["n128", "n256", "gen3"]
    assert verified["component_game_sampling_ratios"] == [0.57, 0.23, 0.20]
    assert verified["policy_kl_anchor_component_ids"] == ["gen3"]
    assert verified["policy_distillation_component_ids"] == ["n128", "n256", "gen3"]
    assert verified["policy_distillation_scope_explicit"] is False
    assert verified["value_training_component_ids"] == ["n128", "n256", "gen3"]
    assert verified["value_training_scope_explicit"] is False
    assert verified["descriptor_fingerprint"] == train_bc._training_data_fingerprint(
        str(path), "memmap"
    )


def test_v2_refuses_ratio_and_anchor_scope_drift(tmp_path):
    path = _descriptor_v2(tmp_path)
    payload = json.loads(path.read_text())
    payload["components"][0]["game_sampling_ratio"] = 0.50
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="ratios must sum to 1"):
        train_bc._preflight_memmap_composite_descriptor(path)


def test_v2_authenticates_policy_distillation_scope_and_refuses_drift(tmp_path):
    path = _descriptor_v2(
        tmp_path, policy_distillation_component_ids=["n128", "n256"]
    )
    verified = train_bc._preflight_memmap_composite_descriptor(path)
    assert verified["policy_distillation_component_ids"] == ["n128", "n256"]
    assert verified["policy_distillation_scope_explicit"] is True

    payload = json.loads(path.read_text())
    payload["policy_distillation_component_ids"] = ["n256", "n128"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="must follow component order"):
        train_bc._preflight_memmap_composite_descriptor(path)


def test_v2_authenticates_policy_aux_phase_allocation(tmp_path):
    allocation = {"PLAY_TURN": 0.60, "MOVE_ROBBER": 0.25, "DISCARD": 0.15}
    path = _descriptor_v2(
        tmp_path,
        policy_distillation_component_ids=["n128", "n256"],
        policy_aux_phase_sampling_weights=allocation,
    )
    verified = train_bc._preflight_memmap_composite_descriptor(path)
    assert verified["policy_aux_phase_sampling_weights"] == allocation
    assert verified["descriptor_fingerprint"] == train_bc._training_data_fingerprint(
        str(path), "memmap"
    )


@pytest.mark.parametrize(
    "allocation",
    [
        {},
        {"PLAY_TURN": 0.9},
        {"PLAY_TURN": 1.0, "DISCARD": 0.0},
        {"PLAY_TURN": True},
    ],
)
def test_v2_refuses_invalid_policy_aux_phase_allocation(tmp_path, allocation):
    path = _descriptor_v2(
        tmp_path, policy_distillation_component_ids=["n128", "n256"]
    )
    payload = json.loads(path.read_text())
    payload["policy_aux_phase_sampling_weights"] = allocation
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="policy auxiliary phase"):
        train_bc._preflight_memmap_composite_descriptor(path)


def test_v2_refuses_phase_allocation_without_explicit_policy_scope(tmp_path):
    path = _descriptor_v2(tmp_path)
    payload = json.loads(path.read_text())
    payload["policy_aux_phase_sampling_weights"] = {"PLAY_TURN": 1.0}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="explicit policy distillation"):
        train_bc._preflight_memmap_composite_descriptor(path)


def test_v2_authenticates_value_training_scope_and_refuses_drift(tmp_path):
    path = _descriptor_v2(
        tmp_path, value_training_component_ids=["n128", "n256"]
    )
    verified = train_bc._preflight_memmap_composite_descriptor(path)
    assert verified["value_training_component_ids"] == ["n128", "n256"]
    assert verified["value_training_scope_explicit"] is True

    payload = json.loads(path.read_text())
    payload["value_training_component_ids"] = ["n256", "n128"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="value training.*must follow component order"):
        train_bc._preflight_memmap_composite_descriptor(path)

    second = tmp_path / "second"
    second.mkdir()
    path = _descriptor_v2(second)
    payload = json.loads(path.read_text())
    payload["policy_kl_anchor_component_ids"] = ["f7-current"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="anchor component ids"):
        train_bc._preflight_memmap_composite_descriptor(path)


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


def test_descriptor_authorizes_exact_diagnostic_policy_weighting_and_refuses_drift(
    tmp_path,
):
    verified = train_bc._preflight_memmap_composite_descriptor(_descriptor(tmp_path))
    matching = type(
        "Args",
        (),
        {"per_game_policy_weight": True, "per_game_policy_weight_mode": "equal"},
    )()
    train_bc._validate_composite_learner_recipe_authorization(matching, verified)
    drifted = type(
        "Args",
        (),
        {"per_game_policy_weight": False, "per_game_policy_weight_mode": "equal"},
    )()
    with pytest.raises(SystemExit, match="authenticated diagnostic learner recipe"):
        train_bc._validate_composite_learner_recipe_authorization(drifted, verified)


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


class _TinyGameCorpus:
    def __init__(self, seeds):
        array = np.asarray(seeds, dtype=np.int64)
        self.row_count = len(array)
        self.legal_width = 1
        self._columns = {
            "game_seed": {"kind": "fixed", "dtype": "<i8", "inner_shape": []}
        }
        self._eager = {"game_seed": array}
        self._lazy = {}
        self.meta = {}
        self.stats = {}

    def keys(self):
        return ["game_seed"]

    def __getitem__(self, key):
        return self._eager[key]


def test_component_sampling_is_component_then_game_then_row_uniform():
    data = ConcatMemmapCorpus(
        [
            _TinyGameCorpus([1, 1, 1, 2]),
            _TinyGameCorpus([10, 11, 11]),
            _TinyGameCorpus([20, 20]),
        ]
    )
    data.component_game_sampling_ratios = (0.5, 0.3, 0.2)
    weights = train_bc._composite_game_sampling_weights(
        data, np.arange(len(data), dtype=np.int64)
    )
    assert weights is not None
    assert weights.sum() == pytest.approx(1.0)
    offsets = data.component_offsets
    assert [
        weights[offsets[i] : offsets[i + 1]].sum() for i in range(3)
    ] == pytest.approx([0.5, 0.3, 0.2])
    # Component 0 gives each game 0.25 mass despite their 3:1 row counts.
    assert weights[:3].sum() == pytest.approx(0.25)
    assert weights[3] == pytest.approx(0.25)


def test_policy_aux_conditioning_preserves_authenticated_base_measure() -> None:
    base = np.asarray([0.30, 0.20, 0.10, 0.40], dtype=np.float64)
    multiplier = np.asarray([0.0, 1.0, 2.0, 1.0], dtype=np.float32)
    conditioned = train_bc._conditioned_policy_aux_sampling_weights(base, multiplier)
    # Conditioning changes admission only. A multiplier of 2 does not become a
    # sampling-frequency weight; phase/winner/etc. remain loss weights.
    assert conditioned == pytest.approx([0.0, 2.0 / 7.0, 1.0 / 7.0, 4.0 / 7.0])


def test_policy_aux_phase_allocation_sets_exact_phase_shares() -> None:
    active = np.asarray([0.10, 0.20, 0.30, 0.40, 0.0], dtype=np.float64)
    phases = np.asarray(["PLAY", "PLAY", "ROBBER", "DISCARD", "PLAY"])
    allocated = train_bc._apply_authenticated_policy_aux_phase_allocation(
        active,
        phases,
        {"PLAY": 0.50, "ROBBER": 0.30, "DISCARD": 0.20},
    )
    assert allocated.sum() == pytest.approx(1.0)
    assert allocated[phases == "PLAY"].sum() == pytest.approx(0.50)
    assert allocated[phases == "ROBBER"].sum() == pytest.approx(0.30)
    assert allocated[phases == "DISCARD"].sum() == pytest.approx(0.20)
    # The authenticated base measure remains proportional inside each phase.
    assert allocated[0] / allocated[1] == pytest.approx(0.5)


def test_policy_aux_phase_allocation_fails_closed_for_empty_active_stratum() -> None:
    with pytest.raises(ValueError, match="has no policy-active mass"):
        train_bc._apply_authenticated_policy_aux_phase_allocation(
            np.asarray([0.5, 0.5], dtype=np.float64),
            np.asarray(["PLAY", "PLAY"]),
            {"PLAY": 0.5, "ROBBER": 0.5},
        )


def test_chunked_policy_aux_phase_allocation_uses_corpus_rows() -> None:
    data = ConcatMemmapCorpus(
        [
            _TinyGameCorpus([1, 1, 1]),
            _TinyGameCorpus([10, 10, 10]),
        ]
    )
    phases = np.asarray(["PLAY", "ROBBER", "PLAY", "PLAY", "ROBBER", "DISCARD"])
    data._eager["phase"] = phases  # type: ignore[assignment]
    weights = train_bc._authenticated_policy_aux_phase_sampling_weights(
        data,
        np.arange(6, dtype=np.int64),
        np.full(6, 1.0 / 6.0, dtype=np.float64),
        {"PLAY": 0.5, "ROBBER": 0.3, "DISCARD": 0.2},
        chunk_rows=2,
    )
    assert weights[phases == "PLAY"].sum() == pytest.approx(0.5)
    assert weights[phases == "ROBBER"].sum() == pytest.approx(0.3)
    assert weights[phases == "DISCARD"].sum() == pytest.approx(0.2)


def test_policy_component_phase_dose_reports_exact_realized_draws() -> None:
    data = ConcatMemmapCorpus(
        [_TinyGameCorpus([1, 1, 1]), _TinyGameCorpus([10, 10, 10])]
    )
    data.component_ids = ("n128", "n256")
    data._eager["phase"] = np.asarray(  # type: ignore[assignment]
        ["PLAY", "ROBBER", "PLAY", "PLAY", "ROBBER", "DISCARD"]
    )
    dose = train_bc._policy_component_phase_dose(
        data,
        np.asarray([0, 1, 1, 3, 4, 5, 5], dtype=np.int64),
        ("PLAY", "ROBBER", "DISCARD"),
        suffix="aux",
        chunk_rows=2,
    )
    assert dose["n128\0PLAY\0aux"] == 1
    assert dose["n128\0ROBBER\0aux"] == 2
    assert dose["n256\0PLAY\0aux"] == 1
    assert dose["n256\0ROBBER\0aux"] == 1
    assert dose["n256\0DISCARD\0aux"] == 2


def test_authenticated_policy_scope_excludes_replay_ce_and_aux_sampling() -> None:
    data = ConcatMemmapCorpus(
        [
            _TinyGameCorpus([1, 1]),
            _TinyGameCorpus([10, 10]),
            _TinyGameCorpus([20, 20]),
        ]
    )
    data.component_ids = ("n128", "n256", "gen3")
    data.policy_distillation_component_indices = (0, 1)
    data.policy_distillation_scope_authenticated = True
    scoped = train_bc._apply_authenticated_policy_distillation_scope(
        data, np.ones(6, dtype=np.float32)
    )
    assert scoped.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    base = np.full(6, 1.0 / 6.0, dtype=np.float64)
    aux = train_bc._conditioned_policy_aux_sampling_weights(base, scoped)
    assert aux.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])
    report = train_bc._policy_distillation_scope_report(data, scoped)
    assert report is not None
    assert report["component_ids"] == ["n128", "n256"]
    assert report["components"]["gen3"]["policy_weight_sum"] == 0.0
    assert report["components"]["gen3"]["positive_policy_rows"] == 0


def test_authenticated_policy_scope_fails_closed_when_empty() -> None:
    data = ConcatMemmapCorpus(
        [_TinyGameCorpus([1]), _TinyGameCorpus([10])]
    )
    data.component_ids = ("current", "replay")
    data.policy_distillation_component_indices = ()
    data.policy_distillation_scope_authenticated = True
    with pytest.raises(SystemExit, match="scope is invalid"):
        train_bc._apply_authenticated_policy_distillation_scope(
            data, np.ones(2, dtype=np.float32)
        )


def test_policy_aux_order_is_exact_and_ddp_rank_sliced() -> None:
    weights = np.asarray([0.0, 0.25, 0.0, 0.75], dtype=np.float64)
    orders = []
    for rank in range(3):
        orders.append(
            train_bc._policy_aux_epoch_order(
                np.random.default_rng(81),
                4,
                weights,
                local_draws=17,
                ddp={"enabled": True, "world_size": 3, "rank": rank},
            )
        )
    assert all(len(order) == 17 for order in orders)
    assert all(set(order.tolist()) <= {1, 3} for order in orders)
    global_draw = np.random.default_rng(81).choice(4, size=51, replace=True, p=weights)
    assert np.array_equal(orders[0], global_draw[0::3])
    assert np.array_equal(orders[1], global_draw[1::3])
    assert np.array_equal(orders[2], global_draw[2::3])


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
