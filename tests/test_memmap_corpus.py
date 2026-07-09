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

from build_memmap_corpus import build_memmap_corpus  # type: ignore  # noqa: E402
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
