from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import train_bc  # noqa: E402
from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.belief_aux_targets import resource_belief_targets  # noqa: E402
from catan_zero.rl.entity_token_features import (  # noqa: E402
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
    mask_player_tokens_public,
)
from catan_zero.rl.entity_feature_adapter import RUST_ENTITY_ADAPTER_V6  # noqa: E402
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphNet,
)


def _privileged_players() -> np.ndarray:
    tokens = np.zeros((2, 4, PLAYER_FEATURE_SIZE), dtype=np.float32)
    tokens[:, :2, 0] = 1.0
    tokens[:, 0, 1] = 1.0  # actor
    # Actor has three cards; it must never be a belief target.
    tokens[:, 0, 6] = 3 / 20
    tokens[:, 0, 15] = 1.0
    tokens[:, 0, 16:21] = np.array([1, 1, 1, 0, 0]) / 10
    # Opponent has public total four and private composition 2/0/1/0/1.
    tokens[:, 1, 6] = 4 / 20
    tokens[:, 1, 15] = 1.0
    tokens[:, 1, 16:21] = np.array([2, 0, 1, 0, 1]) / 10
    return tokens


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    counts = {
        "hex": (19, HEX_FEATURE_SIZE),
        "vertex": (54, VERTEX_FEATURE_SIZE),
        "edge": (72, EDGE_FEATURE_SIZE),
        "player": (4, PLAYER_FEATURE_SIZE),
        "global": (1, GLOBAL_FEATURE_SIZE),
        "event": (0, EVENT_FEATURE_SIZE),
    }
    batch: dict[str, torch.Tensor] = {}
    for name, (count, width) in counts.items():
        batch[f"{name}_tokens"] = torch.randn(batch_size, count, width)
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(
                batch_size, count, dtype=torch.bool
            )
    batch["legal_action_tokens"] = torch.randn(
        batch_size, 3, LEGAL_ACTION_FEATURE_SIZE
    )
    batch["legal_action_context"] = torch.randn(
        batch_size, 3, CONTEXT_ACTION_FEATURE_SIZE
    )
    return batch


def _config(enabled: bool) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        belief_resource_head=enabled,
    )


def test_privileged_target_survives_only_outside_public_input() -> None:
    raw = _privileged_players()
    composition, totals, valid = resource_belief_targets(raw)
    public = mask_player_tokens_public(raw)
    assert np.array_equal(composition[:, 1], [[2, 0, 1, 0, 1]] * 2)
    assert np.array_equal(totals[:, 1], [4, 4])
    assert np.array_equal(valid[:, 0], [False, False])
    assert np.array_equal(valid[:, 1], [True, True])
    assert np.count_nonzero(public[:, 1, 16:21]) == 0
    assert np.allclose(raw[:, 1, 16:21], np.array([[.2, 0, .1, 0, .1]] * 2))


def test_inconsistent_private_composition_is_rejected() -> None:
    raw = _privileged_players()
    raw[0, 1, 16] = 0.3  # private sum five, public total four
    _, _, valid = resource_belief_targets(raw)
    assert not valid[0, 1]
    assert valid[1, 1]


def test_v6_exact_resource_scales_produce_privileged_labels() -> None:
    """V6's /19 and /95 surface must not be decoded as legacy V2 scales."""

    raw = np.zeros((1, 4, PLAYER_FEATURE_SIZE), dtype=np.float32)
    raw[:, :2, 0] = 1.0
    raw[:, 0, 1] = 1.0
    raw[:, 0, 6] = 4 / 95
    raw[:, 0, 15] = 1.0
    raw[:, 0, 16:21] = np.array([1, 1, 1, 1, 0]) / 19
    expected = np.array([5, 2, 3, 1, 1], dtype=np.float32)
    raw[:, 1, 6] = expected.sum() / 95
    raw[:, 1, 15] = 1.0
    raw[:, 1, 16:21] = expected / 19

    composition, totals, valid = resource_belief_targets(
        raw, entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6
    )

    assert valid[0, 1]
    assert totals[0, 1] == 12
    assert np.array_equal(composition[0, 1], expected)


def test_v6_physical_deck_boundary_is_not_treated_as_legacy_clipping() -> None:
    raw = np.zeros((1, 4, PLAYER_FEATURE_SIZE), dtype=np.float32)
    raw[:, :2, 0] = 1.0
    raw[:, 0, 1] = 1.0
    exact = np.array([19, 0, 0, 0, 0], dtype=np.float32)
    raw[:, 1, 6] = exact.sum() / 95
    raw[:, 1, 15] = 1.0
    raw[:, 1, 16:21] = exact / 19

    composition, totals, valid = resource_belief_targets(
        raw, entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V6
    )

    assert valid[0, 1]
    assert totals[0, 1] == 19
    assert np.array_equal(composition[0, 1], exact)


def test_saturation_ambiguous_hidden_hand_is_rejected() -> None:
    raw = _privileged_players()
    # The featurizer clips an actual [11, 3, 2, 2, 3] / total=21 hand to this
    # apparently self-consistent [10, 3, 2, 2, 3] / total=20 representation.
    # It is not recoverable privileged truth and must not become supervision.
    raw[0, 1, 6] = 1.0
    raw[0, 1, 16:21] = np.array([10, 3, 2, 2, 3]) / 10
    _, _, valid = resource_belief_targets(raw)
    assert not valid[0, 1]
    assert valid[1, 1]


def test_fractional_hidden_card_encoding_is_rejected_before_rounding() -> None:
    raw = _privileged_players()
    raw[0, 1, 16:21] = np.array([1.49, 0.0, 1.0, 0.0, 1.0]) / 10
    raw[0, 1, 6] = 3 / 20
    _, _, valid = resource_belief_targets(raw)
    assert not valid[0, 1]
    assert valid[1, 1]


def test_nonfinite_private_label_is_rejected_without_nan_loss(monkeypatch) -> None:
    raw = _privileged_players()
    raw[0, 1, 16] = np.nan
    monkeypatch.setattr(train_bc, "_MASK_HIDDEN_INFO_PLAYER_TOKENS", True)
    loss, active, _, denominator = train_bc._belief_resource_loss(
        {"belief_resource_logits": torch.zeros((2, 4, 5))},
        {"player_tokens": raw},
        np.arange(2),
        torch.device("cpu"),
    )
    assert active == 1
    assert denominator.item() == 4
    assert torch.isfinite(loss)


def test_head_is_default_off_and_main_outputs_are_exact() -> None:
    assert not EntityGraphConfig(1, 1).belief_resource_head
    torch.manual_seed(7)
    off = EntityGraphNet(_config(False))
    torch.manual_seed(7)
    on = EntityGraphNet(_config(True))
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert unexpected == []
    assert missing and all(key.startswith("belief_resource_head.") for key in missing)
    batch = _batch()
    off.eval()
    on.eval()
    out_off = off(batch)
    out_on = on(batch)
    assert out_on["belief_resource_logits"].shape == (2, 4, 5)
    for key in ("logits", "value", "final_vp"):
        assert torch.equal(out_off[key], out_on[key])


def test_structured_loss_has_global_denominator_and_exact_total(monkeypatch) -> None:
    raw = _privileged_players()
    logits = torch.zeros((2, 4, 5), requires_grad=True)
    monkeypatch.setattr(train_bc, "_MASK_HIDDEN_INFO_PLAYER_TOKENS", True)
    loss, active, weighted_sum, denominator = train_bc._belief_resource_loss(
        {"belief_resource_logits": logits},
        {"player_tokens": raw},
        np.arange(2),
        torch.device("cpu"),
    )
    assert active == 2
    assert denominator.item() == 8
    assert torch.allclose(loss, torch.log(torch.tensor(5.0)))
    assert torch.allclose(weighted_sum / denominator, loss)
    expected_counts = 4 * torch.softmax(logits[:, 1], dim=-1)
    assert torch.equal(expected_counts.sum(dim=-1), torch.tensor([4.0, 4.0]))
    loss.backward()
    assert logits.grad is not None and logits.grad[:, 1].abs().sum() > 0


def test_loss_indexes_lazy_player_column_before_numpy_conversion(monkeypatch) -> None:
    class _LazyColumn:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            return self.values[index]

        def __array__(self, *args, **kwargs):
            raise AssertionError("full lazy column must never be materialized")

    monkeypatch.setattr(train_bc, "_MASK_HIDDEN_INFO_PLAYER_TOKENS", True)
    loss, active, _, denominator = train_bc._belief_resource_loss(
        {"belief_resource_logits": torch.zeros((1, 4, 5))},
        {"player_tokens": _LazyColumn(_privileged_players())},
        np.array([1]),
        torch.device("cpu"),
    )
    assert active == 1
    assert denominator.item() == 4
    assert torch.isfinite(loss)


def test_loss_refuses_omniscient_model_input(monkeypatch) -> None:
    monkeypatch.setattr(train_bc, "_MASK_HIDDEN_INFO_PLAYER_TOKENS", False)
    with pytest.raises(ValueError, match="requires --mask-hidden-info"):
        train_bc._belief_resource_loss(
            {"belief_resource_logits": torch.zeros((2, 4, 5))},
            {"player_tokens": _privileged_players()},
            np.arange(2),
            torch.device("cpu"),
        )


def test_loss_can_use_labels_preserved_before_npz_load_time_mask(monkeypatch) -> None:
    raw = _privileged_players()
    composition, total, valid = resource_belief_targets(raw)
    public = mask_player_tokens_public(raw)
    monkeypatch.setattr(train_bc, "_MASK_HIDDEN_INFO_PLAYER_TOKENS", True)
    loss, active, _, denominator = train_bc._belief_resource_loss(
        {"belief_resource_logits": torch.zeros((2, 4, 5))},
        {
            "player_tokens": public,
            "belief_resource_composition": composition,
            "belief_resource_total": total,
            "belief_resource_valid": valid,
        },
        np.arange(2),
        torch.device("cpu"),
    )
    assert active == 2
    assert denominator.item() == 8
    assert torch.allclose(loss, torch.log(torch.tensor(5.0)))


def test_npz_loader_preserves_labels_before_public_mask(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "fake.npz"
    fake.touch()
    raw = _privileged_players()
    monkeypatch.setattr(train_bc, "_teacher_shard_files", lambda _path: [fake])
    monkeypatch.setattr(train_bc, "_load_npz", lambda _path: {})
    monkeypatch.setattr(
        train_bc,
        "_normalize_teacher_shard",
        lambda *_args, **_kwargs: {
            "action_taken": np.array([0, 1], dtype=np.int64),
            "player_tokens": raw.copy(),
        },
    )
    loaded = train_bc.load_teacher_data(
        tmp_path,
        mask_hidden_info=True,
        preserve_belief_resource_targets=True,
    )
    assert np.count_nonzero(loaded["player_tokens"][:, 1, 16:21]) == 0
    assert np.array_equal(
        loaded["belief_resource_composition"][:, 1],
        np.array([[2, 0, 1, 0, 1]] * 2, dtype=np.float32),
    )
    assert np.array_equal(loaded["belief_resource_valid"][:, 1], [True, True])


def test_complete_corpus_coverage_reports_hidden_cards() -> None:
    raw = _privileged_players()
    report = train_bc._belief_resource_coverage(
        {
            "action_taken": np.array([0, 1]),
            "player_tokens": raw,
        },
        chunk_rows=1,
    )
    assert report["eligible_players"] == 2
    assert report["eligible_cards"] == 8
    assert report["components"]["corpus"]["eligible_row_fraction"] == 1.0


def test_complete_corpus_coverage_refuses_source_masked_component(
    monkeypatch,
) -> None:
    class _FakeComposite:
        component_ids = ("omniscient", "source-masked")

        def __init__(self):
            raw = _privileged_players()
            self.corpora = (
                {"action_taken": np.array([0, 1]), "player_tokens": raw},
                {
                    "action_taken": np.array([0, 1]),
                    "player_tokens": mask_player_tokens_public(raw),
                },
            )

    monkeypatch.setattr(train_bc, "ConcatMemmapCorpus", _FakeComposite)
    with pytest.raises(SystemExit, match="source-masked"):
        train_bc._belief_resource_coverage(_FakeComposite(), chunk_rows=1)


def test_a1_effective_recipe_omits_exact_off_belief_for_legacy_seals() -> None:
    args = train_bc.build_parser().parse_args(
        ["--data", "data", "--checkpoint", "candidate.pt", "--report", "report.json"]
    )
    ddp = {"world_size": 1, "rank": 0, "local_rank": 0, "enabled": False}
    effective = train_bc._effective_a1_learner_training_recipe(args, ddp)
    assert "belief_resource_loss_weight" not in effective
    args.belief_resource_loss_weight = 0.02
    effective = train_bc._effective_a1_learner_training_recipe(args, ddp)
    assert effective["belief_resource_loss_weight"] == 0.02
    del args.belief_resource_loss_weight
    effective = train_bc._effective_a1_learner_training_recipe(args, ddp)
    assert "belief_resource_loss_weight" not in effective
