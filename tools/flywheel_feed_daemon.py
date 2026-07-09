"""Flywheel feed daemon (phase-2 window feed, task #94 PR4).

Pulls COMPLETED self-play npz shards from the generation fleet into the flywheel's
loop dir as immutable, md5-verified memmap corpus batches. The orchestrator
(tools/continuous_flywheel.py::ingest_feed_batches) registers every batch that
carries a ``.ready`` marker into the KataGo replay window at round start; nothing
here touches ``window_state.json`` — the daemon is a pure producer, the flywheel
is the single registry writer (same non-invasive split as the report harvester).

Batch lifecycle (all under <loop-dir>/feed/):
  incoming/<batch_id>/    rsync staging (deleted after the corpus is built)
  corpus/<batch_id>/      immutable memmap corpus + feed_manifest.json
  corpus/<batch_id>/.ready        batch verified end-to-end -> flywheel may ingest
  corpus/<batch_id>/.quarantined  something failed a safety check -> NEVER ingested
  feed_state.json         dedup registry {host:relpath: md5} + ingested seed ranges
  feed_daemon.log         cycle log (also stdout)

CHECKPOINT ATTRIBUTION (manual-first champion push): each --config source maps to
ONE flywheel ckpt_version. The expected checkpoint md5 is NOT hand-entered — it is
computed from the flywheel's own registry file for that version (champion_vN.pt in
champion/ or archive/), and the fleet host's live checkpoint file must hash-match
it every cycle. A mismatch (fleet rotated, config not updated — or vice versa)
skips the source LOUDLY until a human reconciles, so foreign-checkpoint data can
never enter the window silently.

INTEGRITY: shard md5s are computed on the source host BEFORE transfer and
re-verified locally AFTER; a mismatch (in-flight write, network corruption) drops
the shard from the batch and it retries next cycle (min-age + stable-mtime make
this rare). Seed safety: each built batch records the min/max game_seed actually
in the corpus; a batch overlapping any previously ingested range is quarantined.

Config (JSON):
{
  "interval_seconds": 300,
  "ssh_key": "~/.ssh/catan_a100_sync_ed25519",
  "min_shard_age_seconds": 180,
  "min_batch_shards": 2,
  "max_batch_shards": 64,
  "sources": [
    {"name": "a100a", "host": "ubuntu@a100a",
     "repo": "/home/ubuntu/catan-zero",
     "shard_globs": ["runs/selfplay/gen3_mps_20260707_r9/*/worker_*/gumbel_self_play_shard_*.npz",
                      "runs/selfplay/gen3_auto/*/worker_*/gumbel_self_play_shard_*.npz"],
     "checkpoint_path": "runs/bc/gen3_20260706/checkpoint.pt",
     "ckpt_version": 0}
  ]
}

Run: PYTHONUNBUFFERED=1 nohup .venv/bin/python tools/flywheel_feed_daemon.py \
       --loop-dir runs/flywheel_20260707b --config runs/flywheel_20260707b/feed_config.json &
``--once`` runs a single cycle and exits (smoke tests / cron-style operation).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ small utils
def _log(feed_dir: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(f"[feed] {line}", flush=True)
    try:
        with open(feed_dir / "feed_daemon.log", "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _atomic_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _atomic_write_text(p: Path, text: str) -> None:
    """Atomic text write: temp + fsync + os.replace. Prevents truncation on crash."""
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ssh(host: str, key: str, remote_cmd: str, *, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote_cmd],
        capture_output=True, text=True, timeout=timeout)


def _py() -> str:
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


# ------------------------------------------------------------------ state
def load_state(feed_dir: Path) -> dict:
    p = feed_dir / "feed_state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"ingested": {}, "seed_ranges": [], "next_batch": 0}


def save_state(feed_dir: Path, state: dict) -> None:
    _atomic_json(feed_dir / "feed_state.json", state)


# ------------------------------------------------------------------ registry checkpoint md5
def registry_checkpoint_md5(loop_dir: Path, version: int, cache: dict) -> str | None:
    """md5 of the flywheel registry's checkpoint for ``version`` (champion/ first, then
    archive/) — the ground truth a source's fleet checkpoint must match. Cached by
    (path, size, mtime) so the 140MB file is hashed once, not every cycle."""
    for cand in (loop_dir / "champion" / f"champion_v{version}.pt",
                 loop_dir / "archive" / f"champion_v{version}.pt"):
        if cand.exists():
            st = cand.stat()
            key = f"{cand}:{st.st_size}:{st.st_mtime_ns}"
            if key not in cache:
                cache[key] = _md5_file(cand)
            return cache[key]
    return None


# ------------------------------------------------------------------ remote scan
def scan_source(src: dict, key: str, min_age: float) -> list[dict] | None:
    """List COMPLETED remote shards for one source: matching a shard glob, mtime older
    than ``min_age`` seconds. Returns [{path(relative to repo), size, mtime}] or None on
    ssh failure (skip the source this cycle)."""
    conds = " -o ".join(f"-path {shlex.quote(g)}" for g in src["shard_globs"])
    remote = (f"cd {shlex.quote(src['repo'])} && "
              f"find . \\( {conds} \\) -type f -printf '%T@ %s %P\\n' 2>/dev/null")
    try:
        r = _ssh(src["host"], key, remote)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    now = time.time()
    out = []
    for line in r.stdout.splitlines():
        try:
            mtime_s, size_s, rel = line.split(" ", 2)
            mtime, size = float(mtime_s), int(size_s)
        except ValueError:
            continue
        if now - mtime >= min_age and size > 0:
            out.append({"rel": rel, "size": size, "mtime": mtime})
    return out


def remote_md5s(src: dict, key: str, rels: list[str]) -> dict[str, str]:
    files = " ".join(shlex.quote(r) for r in rels)
    remote = f"cd {shlex.quote(src['repo'])} && md5sum {files}"
    r = _ssh(src["host"], key, remote, timeout=600)
    out: dict[str, str] = {}
    if r.returncode not in (0, 1):  # md5sum exits 1 if SOME files failed; keep the good lines
        return out
    for line in r.stdout.splitlines():
        try:
            digest, name = line.split(None, 1)
        except ValueError:
            continue
        out[name.strip().lstrip("./")] = digest
    return out


# ------------------------------------------------------------------ batch pipeline
def corpus_game_seed_range(corpus_dir: Path) -> tuple[int, int] | None:
    """min/max game_seed actually in a built corpus (authoritative seed-safety record)."""
    try:
        import numpy as np
        meta = json.loads((corpus_dir / "corpus_meta.json").read_text())
        schema = meta["columns"]["game_seed"]
        if schema["kind"] != "fixed":
            return None
        mm = np.memmap(corpus_dir / "game_seed.dat", dtype=np.dtype(schema["dtype"]),
                       mode="r", shape=(int(meta["row_count"]),
                                        *(int(d) for d in (schema.get("inner_shape") or ()))))
        return int(mm.min()), int(mm.max())
    except Exception:
        return None


def process_source(loop_dir: Path, feed_dir: Path, src: dict, cfg: dict, state: dict,
                   md5_cache: dict) -> None:
    key = os.path.expanduser(cfg.get("ssh_key", "~/.ssh/catan_a100_sync_ed25519"))
    name = src["name"]

    # 0. checkpoint contract: the fleet's live checkpoint must hash-match the flywheel
    #    registry's file for the version this source claims to generate.
    expect = registry_checkpoint_md5(loop_dir, int(src["ckpt_version"]), md5_cache)
    if expect is None:
        _log(feed_dir, f"{name}: no registry checkpoint for v{src['ckpt_version']} — source skipped")
        return
    r = _ssh(src["host"], key, f"md5sum {shlex.quote(src['repo'] + '/' + src['checkpoint_path'])}")
    remote_ckpt_md5 = r.stdout.split()[0] if r.returncode == 0 and r.stdout.split() else None
    if remote_ckpt_md5 != expect:
        _log(feed_dir, f"{name}: CHECKPOINT MISMATCH — remote {src['checkpoint_path']} md5 "
                       f"{remote_ckpt_md5} != registry v{src['ckpt_version']} md5 {expect}. "
                       f"Fleet rotated or config stale; source skipped until reconciled.")
        return

    # 1. list completed shards; drop already-ingested ones
    shards = scan_source(src, key, float(cfg.get("min_shard_age_seconds", 180)))
    if shards is None:
        _log(feed_dir, f"{name}: remote scan failed — skipped this cycle")
        return
    new = [s for s in shards if f"{name}:{s['rel']}" not in state["ingested"]]
    if len(new) < int(cfg.get("min_batch_shards", 2)):
        return
    # ONE WAVE ROOT PER BATCH (round-12-era fix): a batch spanning waves produces a nonsense
    # [min,max] seed envelope (observed: r8+r9 = 84.2M..6.2B) that swallows other hosts'
    # legitimate ranges and false-quarantines their batches. Take the OLDEST root's shards only;
    # other roots flush on subsequent cycles.
    new = sorted(new, key=lambda s: s["mtime"])
    first_root = "/".join(new[0]["rel"].split("/")[:3])
    new = [s for s in new if "/".join(s["rel"].split("/")[:3]) == first_root]
    new = new[: int(cfg.get("max_batch_shards", 64))]

    # 2. source-side md5, then transfer, then local re-verify
    digests = remote_md5s(src, key, [s["rel"] for s in new])
    new = [s for s in new if s["rel"] in digests]
    if not new:
        _log(feed_dir, f"{name}: no shards survived remote md5 — skipped this cycle")
        return
    batch_id = f"batch_{int(state['next_batch']):06d}_{name}"
    staging = feed_dir / "incoming" / batch_id
    staging.mkdir(parents=True, exist_ok=True)
    listfile = staging / ".files"
    _atomic_write_text(listfile, "".join(s["rel"] + "\n" for s in new))
    rsync = subprocess.run(
        ["rsync", "-a", "--files-from", str(listfile), "-e",
         f"ssh -i {key} -o BatchMode=yes -o ConnectTimeout=15",
         f"{src['host']}:{src['repo']}/", str(staging) + "/"],
        capture_output=True, text=True, timeout=1800)
    if rsync.returncode != 0:
        _log(feed_dir, f"{name}: rsync failed ({rsync.returncode}): {rsync.stderr[-300:]}")
        shutil.rmtree(staging, ignore_errors=True)
        return
    verified: list[dict] = []
    for s in new:
        local = staging / s["rel"]
        if local.exists() and _md5_file(local) == digests[s["rel"]]:
            verified.append({**s, "md5": digests[s["rel"]]})
        else:
            try:
                local.unlink(missing_ok=True)  # in-flight write or corruption: retry next cycle
            except OSError:
                pass
    if not verified:
        _log(feed_dir, f"{name}: 0/{len(new)} shards verified — batch abandoned")
        shutil.rmtree(staging, ignore_errors=True)
        return

    # 3. build the immutable corpus for the batch. Manifest order MUST be path-sorted:
    #    a game's rows may span adjacent shards of one worker, and the corpus builder's
    #    duplicate-seed detector (task #85) only tolerates same-seed rows in ADJACENT
    #    manifest entries — mtime order interleaves workers and false-positives the abort.
    verified = sorted(verified, key=lambda v: v["rel"])
    corpus_dir = feed_dir / "corpus" / batch_id
    _atomic_json(staging / "manifest.json",
                 {"shards": [str(staging / v["rel"]) for v in verified], "rows": None})
    build = subprocess.run(
        [_py(), str(REPO_ROOT / "tools" / "build_memmap_corpus.py"),
         "--source", str(staging), "--out", str(corpus_dir)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=3600)
    if build.returncode != 0 or not (corpus_dir / "corpus_meta.json").exists():
        _log(feed_dir, f"{name}: corpus build failed for {batch_id} "
                       f"({build.returncode}): {(build.stdout + build.stderr)[-300:]}")
        shutil.rmtree(corpus_dir, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)  # bad npz retries next cycle (not ingested)
        return
    row_count = int(json.loads((corpus_dir / "corpus_meta.json").read_text())["row_count"])

    # 4. seed safety: refuse a seed-range overlap with a DIFFERENT wave root (out-dir root,
    #    host-qualified). That is the true duplicate-game hazard (the #77 seed-collision class:
    #    two waves generating the same seeds replay IDENTICAL games). Same-wave batches routinely
    #    interleave their [min,max] envelopes (worker shards are pulled oldest-first with holes) —
    #    within a wave, the path+md5 dedup registry already guarantees every row is ingested at
    #    most once, so same-root overlap is expected and allowed.
    seed_range = corpus_game_seed_range(corpus_dir)
    wave_roots = sorted({f"{name}:{'/'.join(v['rel'].split('/')[:3])}" for v in verified})
    manifest = {
        "batch_id": batch_id, "source": name, "host": src["host"],
        "ckpt_version": int(src["ckpt_version"]), "checkpoint_md5": expect,
        "row_count": row_count, "shard_count": len(verified),
        "shards": [{"rel": v["rel"], "md5": v["md5"], "size": v["size"]} for v in verified],
        "game_seed_range": list(seed_range) if seed_range else None,
        "wave_roots": wave_roots,
        "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(corpus_dir / "feed_manifest.json", manifest)
    # RESERVED VAL-ONLY RANGES (ledgered): data in these game_seed ranges must NEVER enter the
    # training window from ANY path — mark .valonly (not .ready, not quarantined). The trainer's
    # --validation-game-seed-ranges is the second, independent enforcement layer.
    reserved_hit = None
    if seed_range is not None:
        for lo_hi in cfg.get("reserved_val_ranges", []):
            lo, hi = int(lo_hi[0]), int(lo_hi[1])
            if not (seed_range[1] < lo or seed_range[0] > hi):
                reserved_hit = [lo, hi]
                break
    overlap = None
    if seed_range is not None and reserved_hit is None:
        for prev in state["seed_ranges"]:
            if set(prev.get("roots", [])) & set(wave_roots):
                continue  # same wave: envelope overlap is normal (see above)
            lo, hi = int(prev["min"]), int(prev["max"])
            if not (seed_range[1] < lo or seed_range[0] > hi):
                overlap = prev
                break
    if reserved_hit is not None:
        _atomic_write_text(corpus_dir / ".valonly", json.dumps(
            {"reason": "reserved val-only game_seed range", "range": list(seed_range),
             "reserved": reserved_hit}))
        _log(feed_dir, f"{name}: {batch_id} VAL-ONLY — seed range {seed_range} intersects "
                       f"reserved validation range {reserved_hit}; never fed to training")
    elif overlap is not None:
        _atomic_write_text(corpus_dir / ".quarantined", json.dumps(
            {"reason": "game_seed overlap", "range": list(seed_range), "conflicts_with": overlap}))
        _log(feed_dir, f"{name}: {batch_id} QUARANTINED — game_seed range {seed_range} overlaps "
                       f"previously ingested {overlap} (duplicate training data)")
    else:
        _atomic_write_text(corpus_dir / ".ready", "")
        _log(feed_dir, f"{name}: {batch_id} READY — {row_count:,} rows from {len(verified)} shards "
                       f"(v{src['ckpt_version']}, seeds {seed_range})")
        if seed_range is not None:
            state["seed_ranges"].append({"batch": batch_id, "min": seed_range[0],
                                         "max": seed_range[1], "roots": wave_roots})

    # 5. commit: dedup registry + cleanup. Quarantined shards are marked ingested too —
    #    re-pulling identical duplicate data every cycle helps nobody; a human clears the
    #    quarantine (and, if desired, the state entries) after diagnosing.
    for v in verified:
        state["ingested"][f"{name}:{v['rel']}"] = v["md5"]
    state["next_batch"] = int(state["next_batch"]) + 1
    save_state(feed_dir, state)
    shutil.rmtree(staging, ignore_errors=True)


# ------------------------------------------------------------------ main
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loop-dir", required=True)
    p.add_argument("--config", required=True, help="feed config JSON (see module docstring)")
    p.add_argument("--once", action="store_true", help="single cycle, then exit")
    args = p.parse_args()

    loop_dir = Path(args.loop_dir)
    feed_dir = loop_dir / "feed"
    feed_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())

    lock_fh = open(feed_dir / "feed_daemon.lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[feed] ERROR: another feed daemon already holds the lock. Exiting.", flush=True)
        return 1
    lock_fh.write(f"pid={os.getpid()}\n")
    lock_fh.flush()

    md5_cache: dict = {}
    _log(feed_dir, f"daemon start: {len(cfg.get('sources', []))} sources, "
                   f"interval {cfg.get('interval_seconds', 300)}s")
    while True:
        state = load_state(feed_dir)
        for src in cfg.get("sources", []):
            try:
                process_source(loop_dir, feed_dir, src, cfg, state, md5_cache)
            except Exception as e:  # one source's failure must not kill the daemon
                _log(feed_dir, f"{src.get('name', '?')}: cycle error {e!r}")
        if args.once or (feed_dir / "STOP").exists():
            _log(feed_dir, "exiting (STOP or --once)")
            return 0
        time.sleep(float(cfg.get("interval_seconds", 300)))


if __name__ == "__main__":
    sys.exit(main())
