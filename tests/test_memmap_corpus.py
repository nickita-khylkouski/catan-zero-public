"""Round-trip, batch-equality, split-equality and memory-ceiling tests for the
streaming memmap corpus (tools/build_memmap_corpus.py + train_bc.MemmapCorpus).

These run against a few real raw-selfplay shards so the equality guarantees are
exercised on the actual production schema, not a synthetic fixture.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from build_memmap_corpus import (  # type: ignore  # noqa: E402
    _stamp_selected_game_source_provenance,
    build_memmap_corpus,
)
import train_bc  # type: ignore  # noqa: E402
from train_bc import (  # type: ignore  # noqa: E402
    ENTITY_BATCH_KEYS,
    MemmapCorpus,
    _entity_batch,
    _iterate_training_batches,
    _mask_player_tokens_public,
    build_sample_weights,
    build_value_sample_weights,
    load_teacher_data,
    split_train_validation_indices,
    teacher_data_quality,
    teacher_provenance_quality,
)

# Shard data (runs/) is gitignored and only present in a full checkout, not in a
# lightweight worktree, so search a few candidate roots. Shard entries are stored
# relative to their manifest's repo root.
_CANDIDATE_ROOTS = [_REPO, _REPO.parent / "catan-zero", Path("/home/ubuntu/catan-zero")]
_MANIFEST_REL = Path("runs/raw_selfplay_gen1_subset/manifest.json")


def _real_shards(count: int) -> list[str]:
    manifest_path = next(
        (root / _MANIFEST_REL for root in _CANDIDATE_ROOTS if (root / _MANIFEST_REL).exists()),
        None,
    )
    if manifest_path is None:
        pytest.skip(f"real shard manifest {_MANIFEST_REL} not found under {_CANDIDATE_ROOTS}")
    repo_root = manifest_path.parents[2]  # runs/<name>/manifest.json -> root
    shards = json.loads(manifest_path.read_text())["shards"]
    resolved: list[str] = []
    for entry in shards:
        path = Path(entry)
        if not path.is_absolute():
            path = repo_root / path
        if path.exists():
            resolved.append(str(path))
        if len(resolved) >= count:
            break
    if len(resolved) < count:
        pytest.skip(f"only {len(resolved)} real shards available, need {count}")
    return resolved


def _make_teacher_dir(tmp_path: Path, shard_paths: list[str]) -> Path:
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (teacher_dir / "manifest.json").write_text(json.dumps({"shards": shard_paths}))
    return teacher_dir


def _minimal_teacher_shard(path: Path, *, seed: int, opponent: bool) -> None:
    rows = 2
    payload: dict[str, np.ndarray] = {
        "obs": np.zeros((rows, 8), dtype=np.float16),
        "legal_action_ids": np.asarray([[3, 9], [4, -1]], dtype=np.int16),
        "legal_action_context": np.zeros((rows, 2, 5), dtype=np.float16),
        "action_taken": np.asarray([3, 4], dtype=np.int16),
        "target_policy": np.asarray([[0.75, 0.25], [1.0, 0.0]], dtype=np.float32),
        "game_seed": np.full(rows, seed, dtype=np.int64),
        "decision_index": np.arange(rows, dtype=np.int32),
        "terminated": np.ones(rows, dtype=np.bool_),
        "truncated": np.zeros(rows, dtype=np.bool_),
    }
    if opponent:
        payload.update(
            {
                "is_pool_game": np.ones(rows, dtype=np.bool_),
                "opponent_version": np.full(rows, 7, dtype=np.int32),
                "opponent_tag": np.full(rows, "recent_history"),
                "opponent_checkpoint_md5": np.full(rows, "deadbeef"),
                "opponent_type": np.asarray(["", "catanatron_value"]),
            }
        )
    np.savez(path, **payload)


def test_opponent_provenance_survives_mixed_npz_and_memmap_paths(tmp_path):
    plain = tmp_path / "plain.npz"
    tagged = tmp_path / "tagged.npz"
    _minimal_teacher_shard(plain, seed=101, opponent=False)
    _minimal_teacher_shard(tagged, seed=202, opponent=True)
    teacher = _make_teacher_dir(tmp_path, [str(plain), str(tagged)])
    corpus_dir = tmp_path / "provenance.memmap"
    build_memmap_corpus(teacher, corpus_dir, progress_every=0)

    in_memory = load_teacher_data(teacher)
    memmap = MemmapCorpus(corpus_dir)
    fields = (
        "is_pool_game",
        "opponent_version",
        "opponent_tag",
        "opponent_checkpoint_md5",
        "opponent_type",
        "opponent_provenance_present",
        "training_source_category",
        "training_source_category_verified",
    )
    for field in fields:
        np.testing.assert_array_equal(np.asarray(memmap[field]), np.asarray(in_memory[field]))

    np.testing.assert_array_equal(memmap["opponent_provenance_present"][:], [False, False, True, True])
    np.testing.assert_array_equal(memmap["opponent_version"][:], [-1, -1, 7, 7])
    np.testing.assert_array_equal(
        memmap["opponent_tag"][:].astype(str),
        ["", "", "recent_history", "recent_history"],
    )
    quality = teacher_data_quality(in_memory)
    assert quality["opponent_provenance_rows"] == 2
    assert quality["opponent_tag_counts"] == {"recent_history": 2}
    assert teacher_data_quality(memmap) == quality
    assert teacher_provenance_quality(memmap, chunk_rows=1)["opponent_provenance_fraction"] == 0.5


def test_restart_provenance_survives_npz_and_memmap_paths(tmp_path):
    shard = tmp_path / "restart.npz"
    _minimal_teacher_shard(shard, seed=700_001, opponent=False)
    with np.load(shard, allow_pickle=True) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload.update(
        {
            "restart_provenance_present": np.ones(2, dtype=np.bool_),
            "start_mode": np.asarray(["archived_public_state"] * 2),
            "start_bucket": np.asarray(["opening"] * 2),
            "archived_game_seed": np.asarray([123, 123], dtype=np.int64),
            "archived_decision_index": np.asarray([4, 4], dtype=np.int64),
            "restart_select_seed": np.asarray([700_001, 700_001], dtype=np.int64),
        }
    )
    np.savez(shard, **payload)
    teacher = _make_teacher_dir(tmp_path, [str(shard)])
    corpus_dir = tmp_path / "restart.memmap"
    build_memmap_corpus(teacher, corpus_dir, progress_every=0)

    in_memory = load_teacher_data(teacher)
    memmap = MemmapCorpus(corpus_dir)
    expected = {
        "restart_provenance_present": [True, True],
        "start_mode": ["archived_public_state", "archived_public_state"],
        "start_bucket": ["opening", "opening"],
        "archived_game_seed": [123, 123],
        "archived_decision_index": [4, 4],
        "restart_select_seed": [700_001, 700_001],
    }
    for field, values in expected.items():
        assert field in in_memory
        assert field in memmap
        np.testing.assert_array_equal(
            np.asarray(in_memory[field]).astype(str)
            if np.asarray(in_memory[field]).dtype.kind in {"O", "U", "S"}
            else np.asarray(in_memory[field]),
            np.asarray(values).astype(str)
            if np.asarray(in_memory[field]).dtype.kind in {"O", "U", "S"}
            else np.asarray(values),
        )
        np.testing.assert_array_equal(
            np.asarray(memmap[field]).astype(str),
            np.asarray(values).astype(str),
        )


def test_mixed_npz_restart_provenance_is_row_aligned(tmp_path):
    legacy = tmp_path / "legacy.npz"
    restart = tmp_path / "restart.npz"
    _minimal_teacher_shard(legacy, seed=10, opponent=False)
    _minimal_teacher_shard(restart, seed=700_001, opponent=False)
    with np.load(restart, allow_pickle=True) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload.update(
        {
            "restart_provenance_present": np.ones(2, dtype=np.bool_),
            "start_mode": np.asarray(["archived_public_state"] * 2),
            "start_bucket": np.asarray(["opening"] * 2),
            "archived_game_seed": np.asarray([123, 123], dtype=np.int64),
            "archived_decision_index": np.asarray([4, 4], dtype=np.int64),
            "restart_select_seed": np.asarray([700_001, 700_001], dtype=np.int64),
        }
    )
    np.savez(restart, **payload)
    teacher = _make_teacher_dir(tmp_path, [str(legacy), str(restart)])

    loaded = load_teacher_data(teacher)

    assert len(loaded["action_taken"]) == len(loaded["start_mode"]) == 4
    assert loaded["restart_provenance_present"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert loaded["start_mode"].tolist() == [
        "legacy_unknown",
        "legacy_unknown",
        "archived_public_state",
        "archived_public_state",
    ]
    assert loaded["archived_game_seed"].tolist() == [-1, -1, 123, 123]


@pytest.mark.parametrize("malformation", ["partial", "source_seed_as_game_seed"])
def test_restart_provenance_loader_rejects_malformed_identity(
    tmp_path,
    malformation,
):
    shard = tmp_path / "restart-malformed.npz"
    _minimal_teacher_shard(shard, seed=700_001, opponent=False)
    with np.load(shard, allow_pickle=True) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    payload.update(
        {
            "restart_provenance_present": np.ones(2, dtype=np.bool_),
            "start_mode": np.asarray(["archived_public_state"] * 2),
            "start_bucket": np.asarray(["opening"] * 2),
            "archived_game_seed": np.asarray([123, 123], dtype=np.int64),
            "archived_decision_index": np.asarray([4, 4], dtype=np.int64),
            "restart_select_seed": np.asarray([700_001, 700_001], dtype=np.int64),
        }
    )
    if malformation == "partial":
        del payload["archived_decision_index"]
        message = "incomplete restart provenance"
    else:
        payload["game_seed"][:] = 123
        message = "game_seed differs from restart_select_seed"
    np.savez(shard, **payload)
    teacher = _make_teacher_dir(tmp_path, [str(shard)])

    with pytest.raises(SystemExit, match=message):
        load_teacher_data(teacher)


def test_authenticated_selected_game_categories_are_stamped_without_guessing():
    normalized = {
        "game_seed": np.asarray([11, 11, 22, 33], dtype=np.int64),
        "opponent_tag": np.asarray(["", "", "recent_history", "hard_negative"]),
        "opponent_provenance_present": np.asarray([False, False, True, True]),
    }
    _stamp_selected_game_source_provenance(
        normalized,
        category_by_seed={11: "current_producer", 22: "recent_history", 33: "hard_negative"},
        path=Path("sealed-shard.npz"),
    )
    np.testing.assert_array_equal(
        normalized["training_source_category"],
        ["current_producer", "current_producer", "recent_history", "hard_negative"],
    )
    assert normalized["training_source_category_verified"].all()


def test_authenticated_source_category_rejects_raw_tag_contradiction():
    normalized = {
        "game_seed": np.asarray([22], dtype=np.int64),
        "opponent_tag": np.asarray(["hard_negative"]),
        "opponent_provenance_present": np.asarray([True]),
    }
    with pytest.raises(SystemExit, match="contradicts authenticated"):
        _stamp_selected_game_source_provenance(
            normalized,
            category_by_seed={22: "recent_history"},
            path=Path("contradictory-shard.npz"),
        )


@pytest.fixture(scope="module")
def corpus_pair(tmp_path_factory):
    """Build (in-RAM dict, MemmapCorpus) from the same 4 real shards, once."""
    tmp_path = tmp_path_factory.mktemp("memmap")
    shard_paths = _real_shards(4)
    teacher_dir = _make_teacher_dir(tmp_path, shard_paths)
    corpus_dir = tmp_path / "corpus"
    build_memmap_corpus(teacher_dir, corpus_dir, progress_every=0)
    old = load_teacher_data(teacher_dir)
    new = MemmapCorpus(corpus_dir)
    return old, new


def test_conversion_round_trip_full_column_equality(corpus_pair):
    old, new = corpus_pair
    assert len(new) == len(old["action_taken"])
    assert set(new.keys()) == set(old.keys())
    for key in old:
        expected = np.asarray(old[key])
        actual = np.asarray(new[key])
        assert actual.shape == expected.shape, f"{key} shape {actual.shape} != {expected.shape}"
        if expected.dtype.kind == "U":
            np.testing.assert_array_equal(actual.astype(str), expected.astype(str), err_msg=key)
        else:
            # assert_array_equal treats NaN==NaN as equal (target_scores has NaN pads).
            np.testing.assert_array_equal(actual, expected, err_msg=key)


def test_teacher_data_quality_accepts_streaming_memmap(corpus_pair):
    in_memory, memmap = corpus_pair
    assert teacher_data_quality(memmap) == teacher_data_quality(in_memory)


def test_batch_content_equality_on_identical_indices(corpus_pair):
    old, new = corpus_pair
    n = len(new)
    rng = np.random.default_rng(1234)
    for _ in range(5):
        batch = rng.choice(n, size=min(97, n), replace=False).astype(np.int64)
        # Unsorted indices exercise the ragged gather path.
        rng.shuffle(batch)
        for key in old:
            expected = np.asarray(old[key])[batch]
            actual = np.asarray(new[key][batch])
            assert actual.shape == expected.shape, f"{key} batch shape mismatch"
            if expected.dtype.kind == "U":
                np.testing.assert_array_equal(actual.astype(str), expected.astype(str), err_msg=key)
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=key)


def test_validation_split_equality(corpus_pair):
    old, new = corpus_pair
    kwargs = dict(validation_fraction=0.1, validation_seed=17, validation_max_samples=0)
    old_split = split_train_validation_indices(old, **kwargs)
    new_split = split_train_validation_indices(new, **kwargs)
    np.testing.assert_array_equal(new_split["train"], old_split["train"])
    np.testing.assert_array_equal(new_split["validation"], old_split["validation"])

    # And the explicit game-seed-range holdout path (task #65 semantics).
    seeds = np.asarray(old["game_seed"])
    lo = int(np.percentile(seeds, 90))
    hi = int(seeds.max())
    ranges = [(lo, hi)]
    old_r = split_train_validation_indices(old, validation_game_seed_ranges=ranges, **kwargs)
    new_r = split_train_validation_indices(new, validation_game_seed_ranges=ranges, **kwargs)
    np.testing.assert_array_equal(new_r["train"], old_r["train"])
    np.testing.assert_array_equal(new_r["validation"], old_r["validation"])


def test_sample_weight_equality(corpus_pair):
    old, new = corpus_pair
    old_v = build_value_sample_weights(old, phase_weights={"robber": 8.0, "initial_build": 5.0})
    new_v = build_value_sample_weights(new, phase_weights={"robber": 8.0, "initial_build": 5.0})
    np.testing.assert_array_equal(new_v, old_v)

    weight_kwargs = dict(
        teacher_weights={},
        phase_weights={"robber": 2.0},
        forced_action_weight=0.5,
        winner_sample_weight=1.5,
        loser_sample_weight=0.8,
        vp_margin_weight=0.3,
        vps_to_win=10,
    )
    old_p = build_sample_weights(old, **weight_kwargs)
    new_p = build_sample_weights(new, **weight_kwargs)
    np.testing.assert_allclose(new_p, old_p, rtol=0, atol=0)


def test_memory_ceiling_streams_large_columns(corpus_pair):
    old, new = corpus_pair
    # The large per-decision columns must be lazy (not resident), not eager.
    for lazy_key in ("obs", "legal_action_context", "legal_action_tokens", "event_tokens", "vertex_tokens"):
        assert lazy_key in new._lazy, f"{lazy_key} should be streamed lazily"

    full_bytes = sum(np.asarray(v).nbytes for v in old.values())
    eager_bytes = sum(np.asarray(v).nbytes for v in new._eager.values())
    # Resident (owned) bytes are dominated by the streamed columns, so the eager
    # baseline is a small fraction of the full in-RAM dict.
    assert eager_bytes < 0.5 * full_bytes, (
        f"eager baseline {eager_bytes} not < 50% of full {full_bytes}"
    )

    # A batch of the largest column materialises O(batch) rows, not O(N).
    n = len(new)
    batch = np.arange(64, dtype=np.int64)
    tokens = new["legal_action_tokens"][batch]
    assert tokens.shape[0] == 64
    per_row = np.asarray(old["legal_action_tokens"][:1]).nbytes
    assert tokens.nbytes <= per_row * 64 * 1.01
    assert tokens.nbytes < np.asarray(old["legal_action_tokens"]).nbytes / (n / 128)


def test_memory_ceiling_subprocess_rss(tmp_path):
    """End-to-end RSS check: loading + iterating the memmap corpus peaks well
    below loading the full corpus into RAM."""
    shard_paths = _real_shards(8)
    teacher_dir = _make_teacher_dir(tmp_path, shard_paths)
    corpus_dir = tmp_path / "corpus"
    build_memmap_corpus(teacher_dir, corpus_dir, progress_every=0)

    script = textwrap.dedent(
        """
        import resource, sys
        from pathlib import Path
        sys.path.insert(0, {tools!r})
        from train_bc import load_teacher_data, MemmapCorpus
        import numpy as np
        mode = sys.argv[1]
        if mode == "npz":
            data = load_teacher_data(Path({teacher!r}))
            # touch everything the way the in-RAM path holds it
            total = sum(np.asarray(v).nbytes for v in data.values())
        else:
            c = MemmapCorpus(Path({corpus!r}))
            n = len(c)
            rng = np.random.default_rng(0)
            for _ in range(20):
                b = rng.choice(n, size=min(256, n), replace=False).astype("int64")
                for k in c.keys():
                    _ = c[k][b]
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        """
    ).format(tools=str(_TOOLS_DIR), teacher=str(teacher_dir), corpus=str(corpus_dir))

    def _rss(mode: str) -> int:
        out = subprocess.run(
            [sys.executable, "-c", script, mode],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip().splitlines()[-1])

    npz_rss = _rss("npz")
    memmap_rss = _rss("memmap")
    # ru_maxrss is KB on Linux. The memmap path should not exceed the full-load
    # path (it avoids the padded-corpus blowup and the concatenate transient).
    assert memmap_rss <= npz_rss, f"memmap rss {memmap_rss} > npz rss {npz_rss}"


# ---------------------------------------------------------------------------
# Prefetch loader (--data-loader-workers): must be order- and value-identical
# to the synchronous path.
# ---------------------------------------------------------------------------


def test_prefetch_loader_matches_synchronous(corpus_pair):
    _, new = corpus_pair
    n = len(new)
    rng = np.random.default_rng(7)
    order = rng.permutation(n).astype(np.int64)
    train_indices = np.arange(n, dtype=np.int64)
    psw = rng.random(n).astype(np.float32)
    vsw = rng.random(n).astype(np.float32)
    batch_size = 128

    sync = list(
        _iterate_training_batches(
            new, order, train_indices, batch_size, psw, vsw, num_workers=0, prefetch=2
        )
    )
    pref = list(
        _iterate_training_batches(
            new, order, train_indices, batch_size, psw, vsw, num_workers=2, prefetch=3
        )
    )
    assert len(sync) == len(pref)
    for (s_data, s_batch, s_psw, s_vsw), (p_data, p_batch, p_psw, p_vsw) in zip(sync, pref):
        # Synchronous yields the corpus + global indices; prefetch yields a
        # materialised dict + local arange. Both must index to the same rows.
        np.testing.assert_array_equal(p_batch, np.arange(len(s_batch)))
        np.testing.assert_array_equal(p_psw, s_psw[s_batch])
        np.testing.assert_array_equal(p_vsw, s_vsw[s_batch])
        for key in new.keys():
            expected = np.asarray(s_data[key][s_batch])
            actual = np.asarray(p_data[key][p_batch])
            if expected.dtype.kind == "U":
                np.testing.assert_array_equal(actual.astype(str), expected.astype(str), err_msg=key)
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=key)


def test_prefetch_disabled_yields_corpus_and_global_indices(corpus_pair):
    _, new = corpus_pair
    n = len(new)
    order = np.arange(n, dtype=np.int64)
    train_indices = np.arange(n, dtype=np.int64)
    w = np.ones(n, dtype=np.float32)
    out = list(_iterate_training_batches(new, order, train_indices, 256, w, w, num_workers=0, prefetch=2))
    # Passthrough: same corpus object, global batch indices.
    assert out[0][0] is new
    np.testing.assert_array_equal(out[0][1], order[:256])


# ---------------------------------------------------------------------------
# --mask-hidden-info: public-observation masking of player_tokens at decode.
# ---------------------------------------------------------------------------

_MASK_SLOTS = (4, 5, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
_ACTOR_SLOT = 1


def test_entity_batch_masks_hidden_player_info_when_enabled(corpus_pair):
    _, new = corpus_pair
    batch = np.arange(200, dtype=np.int64)
    previous = train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS
    try:
        train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS = False
        unmasked = _entity_batch(new, batch)
        train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS = True
        masked = _entity_batch(new, batch)
    finally:
        train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS = previous

    # Only player_tokens changes; every other entity column is untouched.
    for key in ENTITY_BATCH_KEYS:
        if key == "player_tokens":
            continue
        np.testing.assert_array_equal(np.asarray(masked[key]), np.asarray(unmasked[key]), err_msg=key)

    pt_unmasked = np.asarray(unmasked["player_tokens"])
    pt_masked = np.asarray(masked["player_tokens"])
    # Matches the canonical masking contract applied to the unmasked batch.
    np.testing.assert_array_equal(pt_masked, _mask_player_tokens_public(pt_unmasked))

    actor = pt_unmasked[:, :, _ACTOR_SLOT] > 0.5
    nonactor = ~actor
    for slot in _MASK_SLOTS:
        assert np.all(pt_masked[:, :, slot][nonactor] == 0), f"slot {slot} not zeroed for non-actor"
    # Actor rows are left intact.
    np.testing.assert_array_equal(pt_masked[actor], pt_unmasked[actor])


def test_mask_player_tokens_public_is_copy_and_shape_safe():
    rng = np.random.default_rng(0)
    tokens = rng.random((5, 4, 31)).astype(np.float32)
    tokens[:, 0, _ACTOR_SLOT] = 1.0  # player 0 is the actor
    tokens[:, 1:, _ACTOR_SLOT] = 0.0
    original = tokens.copy()
    masked = _mask_player_tokens_public(tokens)
    np.testing.assert_array_equal(tokens, original)  # input untouched
    for slot in _MASK_SLOTS:
        assert np.all(masked[:, 1:, slot] == 0)
    np.testing.assert_array_equal(masked[:, 0, :], tokens[:, 0, :])  # actor kept
    # Single (4, F) sample path.
    single = _mask_player_tokens_public(tokens[0])
    assert single.shape == (4, 31)


# ---------------------------------------------------------------------------
# Multi-source conversion: concatenating N sources == one source of all shards.
# ---------------------------------------------------------------------------


def test_multi_source_conversion_concatenates_in_order(tmp_path):
    shard_paths = _real_shards(4)
    single_dir = _make_teacher_dir(tmp_path, shard_paths)  # helper creates tmp_path/teacher

    # Two sources, 2 shards each, in order.
    src_a = tmp_path / "a"
    src_a.mkdir()
    (src_a / "manifest.json").write_text(json.dumps({"shards": shard_paths[:2]}))
    src_b = tmp_path / "b"
    src_b.mkdir()
    (src_b / "manifest.json").write_text(json.dumps({"shards": shard_paths[2:4]}))

    single_corpus_dir = tmp_path / "single_corpus"
    multi_corpus_dir = tmp_path / "multi_corpus"
    single_meta = build_memmap_corpus(single_dir, single_corpus_dir, progress_every=0)
    multi_meta = build_memmap_corpus([src_a, src_b], multi_corpus_dir, progress_every=0)

    assert multi_meta["row_count"] == single_meta["row_count"]
    assert multi_meta["sources"] == [str(src_a), str(src_b)]
    single = MemmapCorpus(single_corpus_dir)
    multi = MemmapCorpus(multi_corpus_dir)
    assert set(single.keys()) == set(multi.keys())
    for key in single.keys():
        expected = np.asarray(single[key])
        actual = np.asarray(multi[key])
        if expected.dtype.kind == "U":
            np.testing.assert_array_equal(actual.astype(str), expected.astype(str), err_msg=key)
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=key)
