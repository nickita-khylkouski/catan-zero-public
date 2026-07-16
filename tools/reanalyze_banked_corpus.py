#!/usr/bin/env python3
"""CAT-63: banked-corpus reanalyze FLEET JOB -- run reanalyze-lite's
provenance-qualified per-action Q refresh over a large banked corpus
(``runs/memmap_corpus_full``) in resumable, preemption-safe chunks sized for A100
between-wave idle slots.

WHAT THIS IS (and is NOT)
-------------------------
This is the fleet-scale GRADUATION of reanalyze-lite v1 (CAT-34). It does NOT
reimplement the forward-and-rewrite logic -- it imports ``reanalyze_lite`` and
reuses ``batch_forward`` (the exact featurize+forward path), the ragged per-action
column rewriter, and the checkpoint/EMA resolution unchanged. Root values are
intentionally unsupported because this direct forward is not the sealed search
operator. This module adds
ONLY fleet-job orchestration:

* CHUNKING -- partition the corpus into contiguous row ranges along the EXISTING
  ``row_offsets`` boundaries (no new sharding invented). Each chunk is small enough
  to complete inside a between-wave idle slot.
* RESUMABILITY / PREEMPTION SAFETY -- per-chunk atomic ``.done`` markers + content
  hashes. A chunk is "done" only after its rewritten piece is written, fsync'd, and
  hashed. Preemption mid-chunk leaves no done marker, so ``run`` redoes exactly that
  chunk (deterministic overwrite -- never a double-append). The "350-game parts
  never finish" lesson: never assume a chunk finishes; size conservatively and make
  the redo idempotent.
* PARALLEL CLAIMS -- optional O_EXCL claim files let multiple fleet workers (one per
  idle GPU) grab DISJOINT chunks without a coordinator; stale claims (preempted
  workers) are reclaimed.
* MERGE-BACK as a VERSIONED OVERLAY -- the source corpus is NEVER modified. The
  merged output is a new corpus dir whose unchanged columns are HARDLINKS (near-zero
  extra disk vs a 417GB copy) to the source and whose single rewritten column is a
  fresh file assembled from the chunk pieces in row order. Row counts before/after
  are compared to prove no row is lost or duplicated across chunk boundaries.
* MIX-FRACTION -- ``--mix-fraction`` records/computes how much of the reanalyzed
  banked corpus blends into a downstream training window (roadmap B1: "mix ~20% into
  the window").
* COST TRACKING -- ``status`` sums per-chunk wall-clock and rows/s and compares the
  running total against the "~1 fleet-day of forwards" estimate to catch scope
  creep early.

CONTINGENCY (ticket step 1)
---------------------------
CAT-63 is BLOCKED BY CAT-34 and is contingent on CAT-34's decision gate actually
triggering the graduation. This tool builds/verifies the orchestration; do NOT
launch the full 417GB run until that gate has fired. ``plan``/``run``/``merge`` all
work on any memmap corpus, so the mechanics are verified on a tiny corpus first
(see tests/test_reanalyze_banked_corpus.py and the SW-0 small-subset check).

LAYOUT (the job directory, --job-dir)
-------------------------------------
    <job-dir>/
        job_manifest.json           # immutable plan: corpus, checkpoint md5,
                                     #   v_component, chunk boundaries, mask, ...
        chunks/
            chunk_000000.dat        # rewritten piece for that chunk's rows
            chunk_000000.done.json  # atomic completion marker (+ piece sha256)
            chunk_000000.claim.json  # (transient) worker claim, O_EXCL
            ...

SUBCOMMANDS
-----------
    plan    partition the corpus and write job_manifest.json (idempotent)
    run     process pending chunks (all, or --max-chunks / --chunk-ids); resumable
    status  done/pending counts, per-chunk + total wall-clock vs the fleet-day budget
    merge   verify all chunks done, assemble the overlay corpus, write merge manifest
    mix     print the mix plan (rows to draw) for a window of a given size

HOST RUN (the real 417GB job -- see the completion note for the full resource plan)
    # 1. plan (cheap: reads only row_offsets + meta)
    python tools/reanalyze_banked_corpus.py plan \
        --corpus runs/memmap_corpus_full --job-dir runs/reanalyze_job_full \
        --checkpoint runs/champion.pt --v-component target_scores \
        --q-head-provenance q_head_provenance.json --chunk-rows 500000
    # 2. run -- one invocation per idle GPU slot; safe to run many in parallel and to
    #    re-run after preemption (each grabs unclaimed/stale chunks)
    CUDA_VISIBLE_DEVICES=0 python tools/reanalyze_banked_corpus.py run \
        --job-dir runs/reanalyze_job_full --device cuda --batch-size 8192 \
        --max-chunks 8 --progress-every 50
    # 3. status (repeat until done == total)
    python tools/reanalyze_banked_corpus.py status --job-dir runs/reanalyze_job_full
    # 4. merge -> versioned overlay corpus + merge manifest
    python tools/reanalyze_banked_corpus.py merge --job-dir runs/reanalyze_job_full \
        --out runs/memmap_corpus_full_reanalyzed --mix-fraction 0.2
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
_SRC_DIR = _TOOLS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import reanalyze_lite as rl  # noqa: E402

TOOL_NAME = "reanalyze_banked_corpus"
TOOL_VERSION = "1.2"

# Conservative default: 32.6M rows / 500k ~= 66 chunks, each small enough to finish
# in a between-wave idle slot even at modest rows/s (the preemption lesson).
DEFAULT_CHUNK_ROWS = 500_000
# The roadmap's B1 budget ("~1 fleet-day of forwards"); status warns past this.
FLEET_DAY_HOURS = 24.0
# A claim older than this (seconds) is assumed to belong to a preempted worker and
# may be re-grabbed. Generous vs a single chunk's runtime.
DEFAULT_CLAIM_STALE_SEC = 3600.0

_MANIFEST_NAME = "job_manifest.json"
_CHUNKS_DIRNAME = "chunks"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Atomic / durable file helpers
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON durably: temp file -> fsync -> os.replace (atomic rename)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


# --------------------------------------------------------------------------- #
# Chunk-view adapter -- lets reanalyze_lite's rewriters operate on ONE chunk
# --------------------------------------------------------------------------- #
class _ChunkView:
    """Read-only sub-corpus over a contiguous row range ``[row_start, row_end)`` of
    a parent ``MemmapCorpus``, exposing exactly the interface reanalyze_lite's
    ``rewrite_per_action_column`` / ``rewrite_per_state_column`` touch
    (``__getitem__``, ``__contains__``, ``keys``, ``.meta``, ``.legal_width``,
    ``__len__``, ``.row_count``).

    Only the small value / legal / mask columns are ever fetched (and only for this
    chunk's rows), so a chunk fits in RAM regardless of the 417GB corpus. The heavy
    token/obs columns are read by ``batch_forward`` directly off the parent, per
    batch -- this view never materialises them.
    """

    def __init__(self, parent, row_start: int, row_end: int):
        self._parent = parent
        self._rows = np.arange(int(row_start), int(row_end), dtype=np.int64)
        self.row_start = int(row_start)
        self.row_end = int(row_end)
        self.meta = parent.meta
        self.legal_width = parent.legal_width
        self.row_count = int(row_end - row_start)
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.row_count

    def __contains__(self, key: str) -> bool:
        return key in self._parent

    def keys(self):
        return self._parent.keys()

    def __getitem__(self, key: str):
        if key not in self._cache:
            self._cache[key] = np.asarray(self._parent[key][self._rows])
        return self._cache[key]

    def get(self, key: str, default=None):
        return self[key] if key in self else default


# --------------------------------------------------------------------------- #
# Chunk planning (reuses the corpus's OWN row_offsets boundaries)
# --------------------------------------------------------------------------- #
def load_meta(corpus_dir: Path) -> dict:
    meta_path = Path(corpus_dir) / "corpus_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{corpus_dir} is not a memmap corpus (no corpus_meta.json)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != "memmap_corpus_v1":
        raise SystemExit(f"{meta_path}: unsupported schema {meta.get('schema')!r}")
    return meta


def load_row_offsets(corpus_dir: Path, row_count: int) -> np.ndarray:
    """Read row_offsets.dat directly (cheap: (N+1) int64) -- NOT via MemmapCorpus,
    which would eagerly reconstruct the padded ragged columns and blow RAM."""
    offsets = np.fromfile(Path(corpus_dir) / "row_offsets.dat", dtype=np.int64)
    if offsets.shape[0] != row_count + 1:
        raise SystemExit(
            f"row_offsets length {offsets.shape[0]} != row_count+1 {row_count + 1}"
        )
    return offsets


def plan_chunks(row_offsets: np.ndarray, chunk_rows: int) -> list[dict]:
    """Partition ``[0, row_count)`` into contiguous, non-overlapping, gap-free chunks
    of at most ``chunk_rows`` rows each, snapped to existing row_offsets boundaries.

    Each chunk records its flat span so per-action pieces can be validated
    byte-exactly against the source's flat layout at merge time.
    """
    if chunk_rows < 1:
        raise SystemExit(f"--chunk-rows must be >= 1 (got {chunk_rows})")
    row_count = int(row_offsets.shape[0] - 1)
    chunks: list[dict] = []
    for cid, start in enumerate(range(0, row_count, chunk_rows)):
        end = min(start + chunk_rows, row_count)
        chunks.append(
            {
                "chunk_id": cid,
                "row_start": int(start),
                "row_end": int(end),
                "n_rows": int(end - start),
                "flat_start": int(row_offsets[start]),
                "flat_end": int(row_offsets[end]),
                "n_flat": int(row_offsets[end] - row_offsets[start]),
            }
        )
    return chunks


def validate_partition(chunks: list[dict], row_count: int, flat_count: int) -> None:
    """Assert the chunk plan covers every row exactly once (no loss, no overlap)."""
    if not chunks:
        raise SystemExit("empty chunk plan")
    if chunks[0]["row_start"] != 0:
        raise SystemExit("first chunk does not start at row 0")
    if chunks[-1]["row_end"] != row_count:
        raise SystemExit(
            f"last chunk ends at {chunks[-1]['row_end']} != row_count {row_count}"
        )
    rows = 0
    flat = 0
    for i, ch in enumerate(chunks):
        if ch["row_end"] <= ch["row_start"]:
            raise SystemExit(f"chunk {ch['chunk_id']} is empty/negative")
        if i > 0 and ch["row_start"] != chunks[i - 1]["row_end"]:
            raise SystemExit(
                f"gap/overlap between chunk {i - 1} and {i}: "
                f"{chunks[i - 1]['row_end']} != {ch['row_start']}"
            )
        rows += ch["n_rows"]
        flat += ch["n_flat"]
    if rows != row_count:
        raise SystemExit(f"chunk rows sum {rows} != row_count {row_count}")
    if flat != flat_count:
        raise SystemExit(f"chunk flat sum {flat} != flat_count {flat_count}")


# --------------------------------------------------------------------------- #
# Job manifest
# --------------------------------------------------------------------------- #
def _chunks_dir(job_dir: Path) -> Path:
    return Path(job_dir) / _CHUNKS_DIRNAME


def _piece_path(job_dir: Path, chunk_id: int) -> Path:
    return _chunks_dir(job_dir) / f"chunk_{chunk_id:06d}.dat"


def _done_path(job_dir: Path, chunk_id: int) -> Path:
    return _chunks_dir(job_dir) / f"chunk_{chunk_id:06d}.done.json"


def _claim_path(job_dir: Path, chunk_id: int) -> Path:
    return _chunks_dir(job_dir) / f"chunk_{chunk_id:06d}.claim.json"


def read_manifest(job_dir: Path) -> dict:
    path = Path(job_dir) / _MANIFEST_NAME
    if not path.exists():
        raise SystemExit(f"no {_MANIFEST_NAME} in {job_dir}; run `plan` first")
    return json.loads(path.read_text(encoding="utf-8"))


def do_plan(
    *,
    corpus_dir: Path,
    job_dir: Path,
    reanalyzer_meta: dict,
    v_component: str,
    chunk_rows: int,
    mask_hidden_info: bool,
    force: bool,
    q_head_provenance: Path | dict | None = None,
) -> dict:
    spec = rl.validate_v_component(v_component)
    meta = load_meta(corpus_dir)
    row_count = int(meta["row_count"])
    flat_count = int(meta["flat_count"])
    legal_width = int(meta["legal_width"])
    verified_q_provenance = rl.validate_q_head_provenance(
        q_head_provenance,
        reanalyzer_meta=reanalyzer_meta,
        v_component=v_component,
    )
    if spec["kind"] == "per_action" and v_component not in meta["columns"]:
        raise SystemExit(
            f"corpus has no {v_component!r} column; present: {sorted(meta['columns'])}"
        )

    offsets = load_row_offsets(corpus_dir, row_count)
    chunks = plan_chunks(offsets, chunk_rows)
    validate_partition(chunks, row_count, flat_count)

    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    _chunks_dir(job_dir).mkdir(parents=True, exist_ok=True)

    manifest = {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "created_utc": _utc_now(),
        "source_corpus": str(Path(corpus_dir).resolve()),
        "v_component": v_component,
        "column_kind": spec["kind"],
        "forward_output": (spec["forward_output"]),
        "q_head_provenance": verified_q_provenance,
        "column_dtype": meta["columns"][v_component]["dtype"],
        "row_count": row_count,
        "flat_count": flat_count,
        "legal_width": legal_width,
        "chunk_rows": int(chunk_rows),
        "n_chunks": len(chunks),
        "mask_hidden_info": bool(mask_hidden_info),
        "reanalyzer": reanalyzer_meta,
        "fleet_day_hours": FLEET_DAY_HOURS,
        "chunks": chunks,
    }

    manifest_path = job_dir / _MANIFEST_NAME
    shape_keys = (
        "source_corpus",
        "v_component",
        "q_head_provenance",
        "chunk_rows",
        "row_count",
        "flat_count",
        "n_chunks",
    )
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Idempotent: the plan may only be re-affirmed, never silently changed under
        # in-flight chunks (that would orphan pieces built against the old config).
        for key in shape_keys:
            if existing.get(key) != manifest.get(key):
                raise SystemExit(
                    f"{manifest_path} already exists with different {key} "
                    f"({existing.get(key)!r} != {manifest.get(key)!r}); pass --force to replace "
                    f"(this discards in-flight chunk state)"
                )
        if existing.get("reanalyzer", {}).get("md5") != reanalyzer_meta.get("md5"):
            raise SystemExit(
                f"{manifest_path} already planned for a different checkpoint "
                f"(md5 {existing.get('reanalyzer', {}).get('md5')} != {reanalyzer_meta.get('md5')}); "
                f"pass --force to replace"
            )
        return existing

    if manifest_path.exists() and force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != manifest.get(key) for key in shape_keys):
            # The chunk_id numbering means something different under the new plan
            # (e.g. a different --chunk-rows re-slices row ranges). Leftover
            # piece/done/claim files under recycled chunk_ids are never silently
            # trusted (chunk_is_done also checks row range), but purge them here so
            # `--force` actually "discards in-flight chunk state" as documented,
            # instead of leaving orphaned files from the old shape on disk forever.
            import shutil as _shutil

            _shutil.rmtree(_chunks_dir(job_dir), ignore_errors=True)
            _chunks_dir(job_dir).mkdir(parents=True, exist_ok=True)

    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "progress": "reanalyze_job_planned",
                "job_dir": str(job_dir),
                "n_chunks": len(chunks),
                "row_count": row_count,
                "chunk_rows": int(chunk_rows),
                "v_component": v_component,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


# --------------------------------------------------------------------------- #
# Chunk completion state
# --------------------------------------------------------------------------- #
def chunk_is_done(job_dir: Path, chunk: dict, reanalyzer_md5: str) -> bool:
    """A chunk is done iff its done marker exists AND its piece file still matches the
    marker's hash AND it was built by the manifest's checkpoint AND it covers the
    SAME row range this chunk dict describes. This survives:
      * preemption mid-write (no marker -> not done -> redo),
      * a truncated/corrupt piece (hash mismatch -> not done -> redo),
      * a piece built by a stale/different checkpoint (md5 mismatch -> redo),
      * a leftover piece from a discarded `--force` re-plan under a different
        --chunk-rows (this chunk_id now spans different rows -> shape mismatch ->
        redo, never silently trusted just because the id number recurred).
    """
    done_path = _done_path(job_dir, chunk["chunk_id"])
    piece_path = _piece_path(job_dir, chunk["chunk_id"])
    if not done_path.exists() or not piece_path.exists():
        return False
    try:
        marker = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("reanalyzer_md5") != reanalyzer_md5:
        return False
    if (
        marker.get("row_start") != chunk["row_start"]
        or marker.get("row_end") != chunk["row_end"]
        or marker.get("n_rows") != chunk["n_rows"]
        or marker.get("n_flat") != chunk["n_flat"]
    ):
        return False
    if marker.get("piece_bytes") != piece_path.stat().st_size:
        return False
    return marker.get("piece_sha256") == rl.sha256_file(piece_path)


def _try_claim(job_dir: Path, chunk_id: int, stale_sec: float) -> bool:
    """Atomically claim a chunk via O_EXCL. Returns True if this worker now owns it.
    A stale claim (older than ``stale_sec`` -- a preempted worker) is stolen."""
    claim_path = _claim_path(job_dir, chunk_id)
    payload = json.dumps(
        {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_utc": _utc_now(),
            "claimed_at": time.time(),
        },
        sort_keys=True,
    ).encode("utf-8")
    try:
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        # Someone holds it; steal only if stale.
        try:
            existing = json.loads(claim_path.read_text(encoding="utf-8"))
            age = time.time() - float(existing.get("claimed_at", 0.0))
        except (OSError, json.JSONDecodeError, ValueError):
            age = float("inf")
        if age < stale_sec:
            return False
        try:
            os.unlink(claim_path)
        except FileNotFoundError:
            pass
        try:
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError:
            return False  # lost the race to another worker
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _release_claim(job_dir: Path, chunk_id: int) -> None:
    try:
        os.unlink(_claim_path(job_dir, chunk_id))
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# Per-chunk processing (reuses reanalyze_lite forward + rewrite unchanged)
# --------------------------------------------------------------------------- #
def process_chunk(
    parent_corpus,
    policy,
    chunk: dict,
    *,
    job_dir: Path,
    v_component: str,
    spec: dict,
    legal_width: int,
    batch_size: int,
    reanalyzer_md5: str,
    device: str,
    progress_every: int,
) -> dict:
    """Forward + rewrite ONE chunk, write its piece + atomic done marker.

    Reuses ``reanalyze_lite.batch_forward`` (over the parent corpus with this chunk's
    GLOBAL row indices, so only per-batch token columns stream into RAM) and the
    ``reanalyze_lite`` column rewriters (over a ``_ChunkView`` so exactly the chunk's
    flat piece is written, byte-compatible with the source's trimmed-flat layout).
    """
    cid = chunk["chunk_id"]
    row_start, row_end = chunk["row_start"], chunk["row_end"]
    want_q = spec["forward_output"] == "q_values"

    started = time.perf_counter()
    global_idx = np.arange(row_start, row_end, dtype=np.int64)
    fwd = rl.batch_forward(
        policy,
        parent_corpus,
        global_idx,
        batch_size=batch_size,
        want_q=want_q,
        legal_width=legal_width,
        progress_every=progress_every,
        value_materialization=None,
    )

    view = _ChunkView(parent_corpus, row_start, row_end)
    piece_path = _piece_path(job_dir, cid)
    # reanalyze_lite's rewriters use ``name`` to BOTH read the source column and name
    # the output ``<name>.dat``. So call them with the real column name into a
    # per-chunk temp dir (isolated -> parallel workers never collide), then rename the
    # produced ``<v_component>.dat`` to this chunk's piece path.
    tmp_dir = _chunks_dir(job_dir) / f".tmp_chunk_{cid:06d}"
    if tmp_dir.exists():
        import shutil as _shutil

        _shutil.rmtree(tmp_dir)  # scrub a previous preempted attempt
    tmp_dir.mkdir(parents=True)
    rewrite = rl.rewrite_per_action_column(
        view, tmp_dir, v_component, fwd["q_values"], legal_width=legal_width
    )
    produced = tmp_dir / f"{v_component}.dat"

    # Validate the piece length against the source's own flat layout: this is the
    # per-chunk half of the "no row lost or duplicated across chunk boundaries" proof.
    itemsize = np.dtype(view.meta["columns"][v_component]["dtype"]).itemsize
    expected_entries = chunk["n_flat"]
    expected_bytes = expected_entries * itemsize
    got_bytes = produced.stat().st_size
    if got_bytes != expected_bytes:
        import shutil as _shutil

        _shutil.rmtree(tmp_dir, ignore_errors=True)
        raise SystemExit(
            f"chunk {cid}: piece is {got_bytes} bytes, expected {expected_bytes} "
            f"({expected_entries} entries * {itemsize}); refusing to mark done"
        )

    _fsync_file(produced)
    os.replace(produced, piece_path)
    _fsync_file(piece_path)
    import shutil as _shutil

    _shutil.rmtree(tmp_dir, ignore_errors=True)
    piece_sha = rl.sha256_file(piece_path)

    elapsed = time.perf_counter() - started
    marker = {
        "chunk_id": cid,
        "row_start": row_start,
        "row_end": row_end,
        "n_rows": chunk["n_rows"],
        "n_flat": chunk["n_flat"],
        "column_kind": spec["kind"],
        "entries_rewritten": int(rewrite["entries_rewritten"]),
        "piece_bytes": int(piece_path.stat().st_size),
        "piece_sha256": piece_sha,
        "reanalyzer_md5": reanalyzer_md5,
        "device": device,
        "host": socket.gethostname(),
        "elapsed_s": round(elapsed, 3),
        "rows_per_s": round(chunk["n_rows"] / max(elapsed, 1e-9), 1),
        "completed_utc": _utc_now(),
    }
    _atomic_write_json(_done_path(job_dir, cid), marker)
    print(
        json.dumps(
            {
                "progress": "reanalyze_chunk_done",
                **{
                    k: marker[k]
                    for k in (
                        "chunk_id",
                        "n_rows",
                        "entries_rewritten",
                        "elapsed_s",
                        "rows_per_s",
                    )
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return marker


def do_run(
    *,
    job_dir: Path,
    reanalyzer_path: Path,
    device: str,
    batch_size: int,
    max_chunks: int | None,
    chunk_ids: list[int] | None,
    use_claim: bool,
    claim_stale_sec: float,
    progress_every: int,
) -> dict:
    from train_bc import MemmapCorpus

    manifest = read_manifest(job_dir)
    reanalyzer_meta = manifest["reanalyzer"]
    reanalyzer_md5 = reanalyzer_meta["md5"]
    v_component = manifest["v_component"]
    spec = rl.validate_v_component(v_component)
    legal_width = int(manifest["legal_width"])
    # Re-validate at execution time as well as plan time. This makes pre-hardening
    # target_scores jobs (which have no provenance field) fail closed instead of
    # silently continuing to forward an untrained q branch.
    rl.validate_q_head_provenance(
        manifest.get("q_head_provenance"),
        reanalyzer_meta=reanalyzer_meta,
        v_component=v_component,
    )
    if manifest.get("root_value_materialization") is not None:
        raise SystemExit(
            "legacy banked value-materialization plans are semantically invalid; "
            "re-plan a true search reanalysis"
        )

    # The manifest pins the checkpoint md5 at `plan` time; verify the checkpoint
    # file this invocation actually resolved to still matches it. Without this, a
    # checkpoint swapped in place at the same path (or an EMA source checkpoint
    # edited after `plan`) would forward with different weights while every chunk
    # this run produces is still stamped with the OLD, now-wrong md5 -- silently
    # mixing two reanalyzer nets into one job with no detectable trace.
    actual_md5 = rl.md5_file(Path(reanalyzer_path))
    if actual_md5 != reanalyzer_md5:
        raise SystemExit(
            f"reanalyzer checkpoint at {reanalyzer_path} has md5 {actual_md5} but "
            f"{job_dir}/{_MANIFEST_NAME} pins {reanalyzer_md5} from `plan` time -- "
            f"the checkpoint was swapped after planning; re-plan (with --force) "
            f"before running, or restore the original checkpoint"
        )

    chunks = manifest["chunks"]
    if chunk_ids is not None:
        wanted = set(chunk_ids)
        chunks = [c for c in chunks if c["chunk_id"] in wanted]

    pending = [c for c in chunks if not chunk_is_done(job_dir, c, reanalyzer_md5)]
    if not pending:
        print(
            json.dumps(
                {"progress": "reanalyze_run_nothing_pending", "job_dir": str(job_dir)},
                sort_keys=True,
            ),
            flush=True,
        )
        return {"processed": [], "skipped_claimed": [], "pending_remaining": 0}

    # Load the corpus + policy ONCE per invocation (amortise eager-column
    # materialisation and checkpoint load across all chunks this worker does).
    import train_bc

    train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS = bool(manifest["mask_hidden_info"])
    policy_type = reanalyzer_meta.get("policy_type") or "entity_graph"
    parent = MemmapCorpus(manifest["source_corpus"])
    policy = rl.load_policy(
        Path(reanalyzer_path), device=device, policy_type=policy_type
    )

    processed: list[int] = []
    skipped: list[int] = []
    for chunk in pending:
        cid = chunk["chunk_id"]
        if use_claim and not _try_claim(job_dir, cid, claim_stale_sec):
            skipped.append(cid)
            continue
        # Re-check done AFTER claiming (another worker may have finished it between the
        # pending scan and our claim), so we never double-process.
        if chunk_is_done(job_dir, chunk, reanalyzer_md5):
            if use_claim:
                _release_claim(job_dir, cid)
            continue
        try:
            process_chunk(
                parent,
                policy,
                chunk,
                job_dir=Path(job_dir),
                v_component=v_component,
                spec=spec,
                legal_width=legal_width,
                batch_size=batch_size,
                reanalyzer_md5=reanalyzer_md5,
                device=device,
                progress_every=progress_every,
            )
            processed.append(cid)
        finally:
            if use_claim:
                _release_claim(job_dir, cid)
        if max_chunks is not None and len(processed) >= max_chunks:
            break

    remaining = sum(
        1 for c in manifest["chunks"] if not chunk_is_done(job_dir, c, reanalyzer_md5)
    )
    summary = {
        "progress": "reanalyze_run_done",
        "job_dir": str(job_dir),
        "processed": processed,
        "processed_count": len(processed),
        "skipped_claimed": skipped,
        "pending_remaining": remaining,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


# --------------------------------------------------------------------------- #
# Status / cost tracking
# --------------------------------------------------------------------------- #
def do_status(*, job_dir: Path) -> dict:
    manifest = read_manifest(job_dir)
    reanalyzer_md5 = manifest["reanalyzer"]["md5"]
    chunks = manifest["chunks"]
    done, pending = [], []
    total_elapsed = 0.0
    total_rows_done = 0
    total_entries = 0
    for c in chunks:
        if chunk_is_done(job_dir, c, reanalyzer_md5):
            done.append(c["chunk_id"])
            try:
                marker = json.loads(_done_path(job_dir, c["chunk_id"]).read_text())
                total_elapsed += float(marker.get("elapsed_s", 0.0))
                total_rows_done += int(marker.get("n_rows", 0))
                total_entries += int(marker.get("entries_rewritten", 0))
            except (OSError, json.JSONDecodeError):
                pass
        else:
            pending.append(c["chunk_id"])

    gpu_hours = total_elapsed / 3600.0
    frac_done = len(done) / max(len(chunks), 1)
    # Extrapolate remaining cost from the mean per-chunk time of finished chunks.
    projected_total_h = (gpu_hours / frac_done) if frac_done > 0 else None
    status = {
        "job_dir": str(job_dir),
        "source_corpus": manifest["source_corpus"],
        "v_component": manifest["v_component"],
        "n_chunks": len(chunks),
        "done": len(done),
        "pending": len(pending),
        "rows_total": manifest["row_count"],
        "rows_done": total_rows_done,
        "entries_rewritten": total_entries,
        "gpu_hours_consumed": round(gpu_hours, 3),
        "gpu_hours_projected_total": round(projected_total_h, 3)
        if projected_total_h
        else None,
        "fleet_day_budget_h": manifest.get("fleet_day_hours", FLEET_DAY_HOURS),
        "over_budget": bool(
            projected_total_h
            and projected_total_h > manifest.get("fleet_day_hours", FLEET_DAY_HOURS)
        ),
        "pending_chunk_ids": pending[:50],
        "complete": len(pending) == 0,
    }
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    return status


# --------------------------------------------------------------------------- #
# Merge -> versioned overlay corpus
# --------------------------------------------------------------------------- #
def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """Materialise ``dst`` from ``src`` for an overlay. hardlink (default) shares
    disk blocks with the never-modified source; symlink/copy are fallbacks."""
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass  # cross-device or unsupported -> fall through to symlink
    if mode in ("hardlink", "symlink"):
        try:
            os.symlink(os.path.abspath(src), dst)
            return "symlink"
        except OSError:
            pass
    import shutil

    shutil.copy2(src, dst)
    return "copy"


def do_merge(
    *,
    job_dir: Path,
    out_dir: Path,
    link_mode: str,
    mix_fraction: float | None,
) -> dict:
    from train_bc import MemmapCorpus

    manifest = read_manifest(job_dir)
    reanalyzer_md5 = manifest["reanalyzer"]["md5"]
    v_component = manifest["v_component"]
    spec = rl.validate_v_component(v_component)
    # Do not bless/merge pieces from a legacy unsafe q-values plan.
    rl.validate_q_head_provenance(
        manifest.get("q_head_provenance"),
        reanalyzer_meta=manifest["reanalyzer"],
        v_component=v_component,
    )
    if manifest.get("root_value_materialization") is not None:
        raise SystemExit(
            "legacy banked value-materialization plans are semantically invalid; "
            "re-plan a true search reanalysis"
        )
    source = Path(manifest["source_corpus"])
    chunks = manifest["chunks"]

    # 1. Every chunk must be done (verified: marker + hash + checkpoint match).
    not_done = [
        c["chunk_id"] for c in chunks if not chunk_is_done(job_dir, c, reanalyzer_md5)
    ]
    if not_done:
        raise SystemExit(
            f"cannot merge: {len(not_done)} chunk(s) not done: {not_done[:20]}"
            f"{' ...' if len(not_done) > 20 else ''}"
        )

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(
            f"output dir already exists (refusing to overwrite): {out_dir}"
        )

    dtype = np.dtype(manifest["column_dtype"])
    expected_entries = manifest["flat_count"]

    out_dir.mkdir(parents=True)
    rewritten_file = f"{v_component}.dat"

    # 2. Assemble the rewritten column by concatenating chunk pieces IN ROW ORDER
    #    (streamed, bounded memory). This is the second half of the no-loss/no-dup
    #    proof: the assembled length must equal the source's own flat/row count.
    total_entries = 0
    assembled_path = out_dir / rewritten_file
    with open(assembled_path, "wb") as out_f:
        for chunk in sorted(chunks, key=lambda c: c["chunk_id"]):
            cid = chunk["chunk_id"]
            piece = _piece_path(job_dir, cid)
            marker = json.loads(_done_path(job_dir, cid).read_text())
            piece_entries = marker["piece_bytes"] // dtype.itemsize
            expected_chunk = chunk["n_flat"]
            if piece_entries != expected_chunk:
                raise SystemExit(
                    f"chunk {cid}: piece has {piece_entries} entries, plan expects {expected_chunk}"
                )
            with open(piece, "rb") as pf:
                while True:
                    buf = pf.read(1 << 22)
                    if not buf:
                        break
                    out_f.write(buf)
            total_entries += piece_entries
        out_f.flush()
        os.fsync(out_f.fileno())
    if total_entries != expected_entries:
        raise SystemExit(
            f"assembled {v_component} has {total_entries} entries != expected {expected_entries}; "
            f"aborting (row loss/duplication across chunk boundaries)"
        )

    # 3. Overlay every OTHER source file (unchanged columns, offsets, string codes)
    #    as hardlinks -- near-zero extra disk, and the source is never touched.
    link_report: dict[str, str] = {}
    for src_file in sorted(source.iterdir()):
        if not src_file.is_file():
            continue
        name = src_file.name
        if name == rewritten_file:
            continue  # replaced by the assembled column
        if name in (
            "corpus_meta.json",
            "reanalyze_manifest.json",
            "reanalyze_merge_manifest.json",
        ):
            continue  # written fresh below
        link_report[name] = _link_or_copy(src_file, out_dir / name, link_mode)

    # 4. Fresh corpus_meta.json.
    out_meta = load_meta(source)
    meta_changed = False
    (out_dir / "corpus_meta.json").write_text(
        json.dumps(out_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 5. Row-count proof via a real MemmapCorpus open of source + overlay.
    src_row_count = int(load_meta(source)["row_count"])
    reloaded = MemmapCorpus(out_dir)
    if reloaded.row_count != src_row_count:
        raise SystemExit(
            f"overlay row_count {reloaded.row_count} != source {src_row_count}"
        )
    if v_component not in reloaded:
        raise SystemExit(f"overlay is missing the reanalyzed column {v_component!r}")

    # 6. Mix plan (roadmap B1: blend ~20% of the reanalyzed banked corpus into a window).
    mix_plan = None
    if mix_fraction is not None:
        mix_plan = {
            "mix_fraction": float(mix_fraction),
            "reanalyzed_rows": src_row_count,
            "note": (
                "At window-build time draw round(window_size * mix_fraction) rows from "
                "this reanalyzed corpus (capped at reanalyzed_rows); use `mix` subcommand "
                "for a concrete window size."
            ),
        }

    total_elapsed = 0.0
    total_entries_rewritten = 0
    for c in chunks:
        marker = json.loads(_done_path(job_dir, c["chunk_id"]).read_text())
        total_elapsed += float(marker.get("elapsed_s", 0.0))
        total_entries_rewritten += int(marker.get("entries_rewritten", 0))

    merge_manifest = {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "kind": "versioned_overlay",
        "merged_utc": _utc_now(),
        "source_corpus": str(source),
        "output_corpus": str(out_dir),
        "v_component": v_component,
        "column_kind": spec["kind"],
        "forward_output": manifest["forward_output"],
        "q_head_provenance": manifest.get("q_head_provenance"),
        "reanalyzer": manifest["reanalyzer"],
        "mask_hidden_info": manifest["mask_hidden_info"],
        "n_chunks": len(chunks),
        "row_count_before": src_row_count,
        "row_count_after": reloaded.row_count,
        "flat_count": manifest["flat_count"],
        "assembled_entries": total_entries,
        "expected_entries": expected_entries,
        "entries_rewritten": total_entries_rewritten,
        "no_loss_no_dup_verified": total_entries == expected_entries
        and reloaded.row_count == src_row_count,
        "meta_changed": meta_changed,
        "link_mode": link_mode,
        "overlay_links": link_report,
        "assembled_column_file": rewritten_file,
        "total_gpu_hours": round(total_elapsed / 3600.0, 3),
        "mix_plan": mix_plan,
        "fallback_note": (
            "The source corpus is untouched; delete this overlay dir to fall back. "
            "Unchanged columns are hardlinks/symlinks into the source, so the overlay "
            "adds ~one column's worth of disk, not a 417GB copy."
        ),
    }
    _atomic_write_json(out_dir / "reanalyze_merge_manifest.json", merge_manifest)
    print(
        json.dumps(
            {
                "progress": "reanalyze_merge_done",
                "output_corpus": str(out_dir),
                "row_count_after": reloaded.row_count,
                "no_loss_no_dup_verified": merge_manifest["no_loss_no_dup_verified"],
                "total_gpu_hours": merge_manifest["total_gpu_hours"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return merge_manifest


# --------------------------------------------------------------------------- #
# Mix plan
# --------------------------------------------------------------------------- #
def compute_mix_plan(
    reanalyzed_rows: int, window_size: int, mix_fraction: float
) -> dict:
    """How many rows of the reanalyzed banked corpus blend into a window of
    ``window_size`` at ``mix_fraction`` (roadmap B1 default ~0.2), plus the fresh
    remainder. Capped at what the reanalyzed corpus actually holds."""
    if not 0.0 <= mix_fraction <= 1.0:
        raise SystemExit(f"--mix-fraction must be in [0, 1] (got {mix_fraction})")
    if window_size < 0:
        raise SystemExit(f"--window-size must be >= 0 (got {window_size})")
    desired = int(round(window_size * mix_fraction))
    banked = min(desired, int(reanalyzed_rows))
    return {
        "mix_fraction": float(mix_fraction),
        "window_size": int(window_size),
        "reanalyzed_rows_available": int(reanalyzed_rows),
        "banked_rows_to_draw": banked,
        "fresh_rows": int(window_size) - banked,
        "capped": banked < desired,
    }


def do_mix(
    *,
    job_dir: Path | None,
    reanalyzed_rows: int | None,
    window_size: int,
    mix_fraction: float,
) -> dict:
    if reanalyzed_rows is None:
        if job_dir is None:
            raise SystemExit(
                "mix needs --window-size and either --job-dir or --reanalyzed-rows"
            )
        reanalyzed_rows = int(read_manifest(job_dir)["row_count"])
    plan = compute_mix_plan(reanalyzed_rows, window_size, mix_fraction)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    return plan


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # plan
    p = sub.add_parser("plan", help="partition the corpus + write job_manifest.json")
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--job-dir", required=True, type=Path)
    p.add_argument(
        "--v-component",
        required=True,
        choices=sorted(rl.V_COMPONENTS),
        help="q-value column to rewrite; requires --q-head-provenance. Root value "
        "columns require true search reanalysis and are unsupported here.",
    )
    p.add_argument(
        "--q-head-provenance",
        type=Path,
        default=None,
        help="required checkpoint-bound validation JSON for target_scores or "
        "afterstate_target",
    )
    p.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    p.add_argument(
        "--reanalyzer-net", default="checkpoint", choices=("checkpoint", "ema")
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--ema-checkpoints", nargs="+", type=Path, default=None)
    p.add_argument("--ema-decay", type=float, default=0.75)
    p.add_argument(
        "--mask-hidden-info",
        dest="mask_hidden_info",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="public-observation masking during forwards (default: inherit checkpoint flag)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="replace an existing plan (discards chunk state)",
    )

    # run
    p = sub.add_parser(
        "run", help="process pending chunks (resumable, preemption-safe)"
    )
    p.add_argument("--job-dir", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="stop after N chunks (one idle slot)",
    )
    p.add_argument(
        "--chunk-ids", type=int, nargs="+", default=None, help="only these chunk ids"
    )
    p.add_argument(
        "--no-claim",
        dest="use_claim",
        action="store_false",
        default=True,
        help="disable O_EXCL claim files (single-worker mode)",
    )
    p.add_argument("--claim-stale-sec", type=float, default=DEFAULT_CLAIM_STALE_SEC)
    p.add_argument("--progress-every", type=int, default=0)

    # status
    p = sub.add_parser(
        "status", help="done/pending + GPU-hours vs the fleet-day budget"
    )
    p.add_argument("--job-dir", required=True, type=Path)

    # merge
    p = sub.add_parser("merge", help="assemble the versioned overlay corpus")
    p.add_argument("--job-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--link-mode", default="hardlink", choices=("hardlink", "symlink", "copy")
    )
    p.add_argument(
        "--mix-fraction",
        type=float,
        default=None,
        help="record a mix plan (fraction of the reanalyzed corpus into a window)",
    )

    # mix
    p = sub.add_parser("mix", help="print the mix plan for a given window size")
    p.add_argument("--job-dir", type=Path, default=None)
    p.add_argument("--reanalyzed-rows", type=int, default=None)
    p.add_argument("--window-size", type=int, required=True)
    p.add_argument("--mix-fraction", type=float, default=0.2)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.command == "plan":
        reanalyzer_path, reanalyzer_meta = rl.resolve_reanalyzer_checkpoint(
            mode=args.reanalyzer_net,
            checkpoint=args.checkpoint,
            ema_checkpoints=args.ema_checkpoints,
            ema_decay=args.ema_decay,
            work_dir=Path(args.job_dir),
        )
        mask = rl._resolve_mask_hidden_info(
            args.mask_hidden_info, reanalyzer_meta["mask_hidden_info"]
        )
        do_plan(
            corpus_dir=args.corpus,
            job_dir=args.job_dir,
            reanalyzer_meta=reanalyzer_meta,
            v_component=args.v_component,
            chunk_rows=args.chunk_rows,
            mask_hidden_info=mask,
            force=args.force,
            q_head_provenance=args.q_head_provenance,
        )
        return

    if args.command == "run":
        manifest = read_manifest(args.job_dir)
        rmeta = manifest["reanalyzer"]
        # Re-resolve the reanalyzer checkpoint path (EMA is re-derived deterministically
        # into the job dir; a plain checkpoint uses its recorded path).
        if rmeta["mode"] == "ema":
            reanalyzer_path, _ = rl.resolve_reanalyzer_checkpoint(
                mode="ema",
                checkpoint=None,
                ema_checkpoints=[Path(p) for p in rmeta["ema_source_checkpoints"]],
                ema_decay=float(rmeta["ema_decay"]),
                work_dir=Path(args.job_dir),
            )
        else:
            reanalyzer_path = Path(rmeta["path"])
        do_run(
            job_dir=args.job_dir,
            reanalyzer_path=reanalyzer_path,
            device=args.device,
            batch_size=args.batch_size,
            max_chunks=args.max_chunks,
            chunk_ids=args.chunk_ids,
            use_claim=args.use_claim,
            claim_stale_sec=args.claim_stale_sec,
            progress_every=args.progress_every,
        )
        return

    if args.command == "status":
        do_status(job_dir=args.job_dir)
        return

    if args.command == "merge":
        do_merge(
            job_dir=args.job_dir,
            out_dir=args.out,
            link_mode=args.link_mode,
            mix_fraction=args.mix_fraction,
        )
        return

    if args.command == "mix":
        do_mix(
            job_dir=args.job_dir,
            reanalyzed_rows=args.reanalyzed_rows,
            window_size=args.window_size,
            mix_fraction=args.mix_fraction,
        )
        return


if __name__ == "__main__":
    main()
