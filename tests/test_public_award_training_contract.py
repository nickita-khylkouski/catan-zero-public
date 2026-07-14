from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.entity_token_policy import (
    PLAYER_LONGEST_ROAD_SLOT,
    PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
    PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
    EntityGraphConfig,
    EntityGraphPolicy,
)
from tools.build_memmap_corpus import (
    PUBLIC_AWARD_FEATURE_CONTRACT_MIXED,
    _load_public_award_source_provenance,
)
from tools.train_bc import (
    ENTITY_BATCH_KEYS,
    _canonical_json_sha256,
    _configure_public_award_feature_training,
    _entity_batch,
    _write_entity_checkpoint,
)
import tools.train_bc as train_bc


def _policy() -> EntityGraphPolicy:
    pytest.importorskip("torch")
    return EntityGraphPolicy(
        EntityGraphConfig(
            action_size=8,
            static_action_feature_size=4,
            hidden_size=16,
            state_layers=1,
            attention_heads=2,
            dropout=0.0,
        ),
        np.zeros((8, 4), dtype=np.float32),
        device="cpu",
    )


def _producer_record() -> dict[str, object]:
    return {
        "schema_version": "public-award-feature-provenance-v1",
        "contract": PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
        "feature_producer": "catanatron_rs_public_award_v1",
        "native_capability": "public_award_feature_parity",
    }


def _data(contract: str) -> SimpleNamespace:
    if contract == PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO:
        binding = {
            "source": "/legacy",
            "manifest": None,
            "manifest_file_sha256": None,
            "contract": contract,
            "producer_provenance": None,
        }
        bindings = [binding]
    elif contract == PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE:
        binding = {
            "source": "/corrected",
            "manifest": "/corrected/manifest.json",
            "manifest_file_sha256": "sha256:" + "a" * 64,
            "contract": contract,
            "producer_provenance": _producer_record(),
        }
        bindings = [binding]
    else:
        bindings = [
            {
                "source": "/legacy",
                "manifest": None,
                "manifest_file_sha256": None,
                "contract": PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
                "producer_provenance": None,
            },
            {
                "source": "/corrected",
                "manifest": "/corrected/manifest.json",
                "manifest_file_sha256": "sha256:" + "b" * 64,
                "contract": PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
                "producer_provenance": _producer_record(),
            },
        ]
    return SimpleNamespace(
        meta={
            "public_award_feature_provenance": {
                "schema_version": "public-award-corpus-provenance-v1",
                "contract": contract,
                "source_manifest_bindings": bindings,
                "source_manifest_bindings_sha256": _canonical_json_sha256(bindings),
            }
        }
    )


def _args(contract: str, *, allow_mixed: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        public_award_feature_contract=contract,
        allow_mixed_public_award_feature_contracts=allow_mixed,
    )


def test_authoritative_transition_zero_initializes_legacy_column_and_attests() -> None:
    torch = pytest.importorskip("torch")
    policy = _policy()
    with torch.no_grad():
        policy.model.player_encoder[0].weight[:, PLAYER_LONGEST_ROAD_SLOT].fill_(3.0)

    report = _configure_public_award_feature_training(
        policy,
        _data(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE),
        _args(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE),
    )

    assert torch.count_nonzero(
        policy.model.player_encoder[0].weight[:, PLAYER_LONGEST_ROAD_SLOT]
    ).item() == 0
    assert policy.public_award_feature_contract == PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    assert report["legacy_column_zero_initialized"] is True
    assert report["corpus_provenance"]["contract"] == PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE


def test_authoritative_resume_preserves_trained_column() -> None:
    torch = pytest.importorskip("torch")
    policy = _policy()
    policy.public_award_feature_contract = PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    with torch.no_grad():
        policy.model.player_encoder[0].weight[:, PLAYER_LONGEST_ROAD_SLOT].fill_(2.0)

    report = _configure_public_award_feature_training(
        policy,
        _data(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE),
        _args(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE),
    )

    assert torch.all(
        policy.model.player_encoder[0].weight[:, PLAYER_LONGEST_ROAD_SLOT] == 2.0
    )
    assert report["legacy_column_zero_initialized"] is False


def test_authoritative_training_rejects_legacy_or_mixed_corpus() -> None:
    policy = _policy()
    with pytest.raises(SystemExit, match="entirely corrected"):
        _configure_public_award_feature_training(
            policy,
            _data(PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO),
            _args(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE),
        )
    with pytest.raises(SystemExit, match="exact 64/12/4/20"):
        _configure_public_award_feature_training(
            policy,
            _data(PUBLIC_AWARD_FEATURE_CONTRACT_MIXED),
            _args(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE, allow_mixed=True),
        )


def test_corrected_corpus_requires_explicit_authoritative_request() -> None:
    with pytest.raises(SystemExit, match="requires explicit"):
        _configure_public_award_feature_training(
            _policy(),
            _data(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE),
            _args(PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO),
        )


def test_mixed_corpus_requires_explicit_legacy_acknowledgement() -> None:
    policy = _policy()
    with pytest.raises(SystemExit, match="--allow-mixed"):
        _configure_public_award_feature_training(
            policy,
            _data(PUBLIC_AWARD_FEATURE_CONTRACT_MIXED),
            _args(PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO),
        )
    report = _configure_public_award_feature_training(
        policy,
        _data(PUBLIC_AWARD_FEATURE_CONTRACT_MIXED),
        _args(PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO, allow_mixed=True),
    )
    assert report["effective_contract"] == PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO
    assert report["diagnostic_only"] is True
    assert report["promotion_eligible"] is False


class _AwardComponent:
    def __init__(self, contract: str, *, rows: int = 4, positive: bool = False):
        self.meta = _data(contract).meta
        self.row_count = rows
        self._player_tokens = np.zeros((rows, 4, 31), dtype=np.float32)
        if positive:
            self._player_tokens[0, 0, PLAYER_LONGEST_ROAD_SLOT] = 1.0

    def __len__(self):
        return self.row_count

    def __contains__(self, key):
        return key == "player_tokens"

    def __getitem__(self, key):
        if key != "player_tokens":
            raise KeyError(key)
        return self._player_tokens


class _MixedAwardComposite:
    component_ids = (
        "current_producer",
        "recent_history",
        "hard_negative",
        "historical_replay",
    )
    component_game_sampling_ratios = (0.64, 0.12, 0.04, 0.20)

    def __init__(self):
        self.corpora = (
            _AwardComponent(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE, positive=True),
            _AwardComponent(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE, positive=True),
            _AwardComponent(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE, positive=True),
            _AwardComponent(PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO),
        )


class _RoutedBatch(dict):
    public_award_component_contracts = (
        PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
        PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
        PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
        PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
    )

    @staticmethod
    def component_indices_for_rows(rows):
        return np.asarray(rows, dtype=np.int64)


def test_exact_mixed_transition_routes_legacy_rows_only_and_is_promotable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    policy = _policy()
    with torch.no_grad():
        policy.model.player_encoder[0].weight[:, PLAYER_LONGEST_ROAD_SLOT].fill_(3.0)
    data = _MixedAwardComposite()

    report = _configure_public_award_feature_training(
        policy,
        data,
        _args(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE, allow_mixed=True),
    )

    transition = report["mixed_authoritative_transition"]
    assert transition["schema_version"] == "mixed-authoritative-transition-v1"
    assert transition["corrected_sampler_mass"] == pytest.approx(0.80)
    assert transition["legacy_sampler_mass"] == pytest.approx(0.20)
    assert transition["legacy_rows_zero_slot12"] is True
    assert report["diagnostic_only"] is False
    assert report["promotion_eligible"] is True
    assert report["legacy_column_zero_initialized"] is True
    assert policy.public_award_feature_contract == PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE

    monkeypatch.setattr(
        train_bc,
        "_PUBLIC_AWARD_FEATURE_CONTRACT",
        PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
    )
    batch_data = _RoutedBatch(
        {key: np.zeros((4, 1), dtype=np.float32) for key in ENTITY_BATCH_KEYS}
    )
    batch_data["player_tokens"] = np.zeros((4, 4, 31), dtype=np.float32)
    batch_data["player_tokens"][..., PLAYER_LONGEST_ROAD_SLOT] = 1.0
    routed = _entity_batch(batch_data, np.arange(4, dtype=np.int64))
    slot = routed["player_tokens"][..., PLAYER_LONGEST_ROAD_SLOT]
    assert np.all(slot[:3] == 1.0)
    assert np.all(slot[3] == 0.0)


def test_mixed_transition_refuses_nonzero_legacy_or_empty_corrected_support() -> None:
    corrupted = _MixedAwardComposite()
    corrupted.corpora[-1]._player_tokens[0, 0, PLAYER_LONGEST_ROAD_SLOT] = 1.0
    with pytest.raises(SystemExit, match="legacy.*nonzero"):
        _configure_public_award_feature_training(
            _policy(),
            corrupted,
            _args(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE, allow_mixed=True),
        )

    empty = _MixedAwardComposite()
    empty.corpora[1]._player_tokens[..., PLAYER_LONGEST_ROAD_SLOT] = 0.0
    with pytest.raises(SystemExit, match="no positive"):
        _configure_public_award_feature_training(
            _policy(),
            empty,
            _args(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE, allow_mixed=True),
        )


def test_legacy_training_batch_bridge_zeroes_corrected_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_bc,
        "_PUBLIC_AWARD_FEATURE_CONTRACT",
        PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
    )
    data = {
        key: np.zeros((1, 1), dtype=np.float32) for key in ENTITY_BATCH_KEYS
    }
    data["player_tokens"] = np.zeros((1, 4, 31), dtype=np.float32)
    data["player_tokens"][..., PLAYER_LONGEST_ROAD_SLOT] = 1.0

    batch = _entity_batch(data, np.asarray([0], dtype=np.int64))

    assert np.count_nonzero(batch["player_tokens"][..., PLAYER_LONGEST_ROAD_SLOT]) == 0


def test_no_flag_legacy_path_is_weight_and_contract_noop() -> None:
    torch = pytest.importorskip("torch")
    policy = _policy()
    before = policy.model.player_encoder[0].weight.detach().clone()
    report = _configure_public_award_feature_training(
        policy,
        _data(PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO),
        argparse.Namespace(),
    )
    torch.testing.assert_close(policy.model.player_encoder[0].weight, before, rtol=0, atol=0)
    assert policy.public_award_feature_contract == PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO
    assert report["legacy_column_zero_initialized"] is False


def test_ddp_writer_and_loader_preserve_authoritative_contract(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    policy = _policy()
    _configure_public_award_feature_training(
        policy,
        _data(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE),
        _args(PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE),
    )
    output = tmp_path / "ddp.pt"
    _write_entity_checkpoint(
        policy,
        str(output),
        policy.model.state_dict(),
        True,
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["public_award_feature_contract"] == PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    loaded = EntityGraphPolicy.load(output, device="cpu")
    assert loaded.public_award_feature_contract == PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE


def test_memmap_builder_binds_manifest_hash_and_labels_mixed(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    corrected = tmp_path / "corrected"
    legacy.mkdir()
    corrected.mkdir()
    (legacy / "manifest.json").write_text(json.dumps({"shards": []}), encoding="utf-8")
    (corrected / "manifest.json").write_text(
        json.dumps({"shards": [], "public_award_feature_provenance": _producer_record()}),
        encoding="utf-8",
    )

    provenance = _load_public_award_source_provenance([legacy, corrected])

    assert provenance["contract"] == PUBLIC_AWARD_FEATURE_CONTRACT_MIXED
    assert len(provenance["source_manifest_bindings"]) == 2
    assert provenance["source_manifest_bindings_sha256"] == _canonical_json_sha256(
        provenance["source_manifest_bindings"]
    )


def test_malformed_corrected_manifest_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    malformed = _producer_record()
    malformed["contract"] = "guess_from_values"
    (source / "manifest.json").write_text(
        json.dumps({"shards": [], "public_award_feature_provenance": malformed}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="unsupported public-award provenance"):
        _load_public_award_source_provenance([source])
