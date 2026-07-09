"""Checkpoint registry for the continuous flywheel: candidate publishing, gated champion
promotion, and a champion ARCHIVE for the opponent pool.

Three roles, three on-disk objects (all atomic; version and bytes can never disagree — the same
FIX-H1 discipline as ``ppo_distributed.publish_weights``):

    {root}/
      candidates/
        weights_v{N}.pt      # trainer emits a fresh candidate every checkpoint interval
        candidate.json       # {version, step, path, updated_at}
      champion/
        champion.json        # {version, weights, promoted_at, source_step, gate}  <- self-play reads THIS
        champion_v{N}.pt     # the promoted weights (self-play workers load these)
      archive/
        champion_v{N}.pt     # frozen snapshot of every past champion (opponent pool draws from here)
        archive.json         # [{version, weights, promoted_at, elo}]  newest last

Why the split (research basis, memory ``catan-discrete-vs-continuous-verdict``):
  - The TRAINER updates continuously and emits *candidates* — it never blocks.
  - Only the CHAMPION (the net that feeds self-play) advances through the cheap gate. This is
    KataGo's exact structure: continuous training, but a gated pointer for whichever checkpoint is
    trusted to GENERATE data, so a regressed net can't silently poison the replay buffer (we have
    already been bitten by corpus poisoning once — the seed-collision incident).
  - The ARCHIVE keeps past champions so self-play can play 15-25% of games against older nets
    (Tablut / OpenAI Five anti-forgetting result) — Catan is asymmetric (1st-player advantage +
    hidden dev cards) so latest-vs-latest self-play risks role-forgetting / non-convergent cycling.

Pure stdlib + a caller-supplied ``save_fn`` / file copy; no torch import at module load.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CANDIDATES_DIRNAME = "candidates"
CHAMPION_DIRNAME = "champion"
ARCHIVE_DIRNAME = "archive"
CANDIDATE_META = "candidate.json"
CHAMPION_META = "champion.json"
ARCHIVE_META = "archive.json"
# Keep the newest N candidate weight files (the trainer emits many between promotions).
KEEP_CANDIDATES = 3
# Keep the newest N live champion weight files in champion/ (self-play only ever loads the current
# one; a small tail covers an in-flight reader during a pointer swap).
KEEP_LIVE_CHAMPIONS = 3
# Keep N archived champions for the opponent pool: the OLDEST (a weak diversity anchor) plus the
# newest N-1. Longer tail than candidates — opponent diversity wants a spread of strengths.
KEEP_ARCHIVE = 12


# --------------------------------------------------------------------------- paths
def candidates_dir(root: str | os.PathLike) -> Path:
    return Path(root) / CANDIDATES_DIRNAME


def champion_dir(root: str | os.PathLike) -> Path:
    return Path(root) / CHAMPION_DIRNAME


def archive_dir(root: str | os.PathLike) -> Path:
    return Path(root) / ARCHIVE_DIRNAME


def ensure_dirs(root: str | os.PathLike) -> None:
    for d in (candidates_dir(root), champion_dir(root), archive_dir(root)):
        Path(d).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- helpers
def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _gc_versioned(dir_: Path, prefix: str, *, keep: int) -> int:
    """Delete all but the newest ``keep`` ``{prefix}v{N}.pt`` files by version N."""
    versioned: list[tuple[int, Path]] = []
    for p in dir_.glob(f"{prefix}v*.pt"):
        try:
            n = int(p.stem.split("_v", 1)[1]) if "_v" in p.stem else int(p.stem.split("v", 1)[1])
        except (IndexError, ValueError):
            continue
        versioned.append((n, p))
    versioned.sort(key=lambda t: t[0], reverse=True)
    removed = 0
    for _, p in versioned[keep:]:
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


# --------------------------------------------------------------------------- candidate
@dataclass(frozen=True)
class CandidateRef:
    version: int
    step: int
    updated_at: float
    path: str


def publish_candidate(root: str | os.PathLike, save_fn: Callable[[str], Any], *, step: int) -> CandidateRef:
    """Trainer side: atomically publish a new candidate (version = prev+1). ``save_fn(tmp)`` writes
    a loadable checkpoint. Never blocks the trainer; promotion is a separate, gated step."""
    cdir = candidates_dir(root)
    cdir.mkdir(parents=True, exist_ok=True)
    prev = read_candidate(root)
    version = (prev.version + 1) if prev else 1
    final = cdir / f"weights_v{version}.pt"
    tmp = final.with_suffix(final.suffix + ".tmp")
    save_fn(str(tmp))
    os.replace(tmp, final)
    _atomic_write_json(cdir / CANDIDATE_META,
                       {"version": version, "step": int(step), "updated_at": time.time(),
                        "weights": final.name})
    _gc_versioned(cdir, "weights_", keep=KEEP_CANDIDATES)
    return CandidateRef(version=version, step=int(step), updated_at=time.time(), path=str(final))


def read_candidate(root: str | os.PathLike) -> CandidateRef | None:
    meta_p = candidates_dir(root) / CANDIDATE_META
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    wp = candidates_dir(root) / str(meta.get("weights", ""))
    if not wp.exists():
        return None
    return CandidateRef(version=int(meta.get("version", 0)), step=int(meta.get("step", 0)),
                        updated_at=float(meta.get("updated_at", 0.0)), path=str(wp))


# --------------------------------------------------------------------------- champion
@dataclass(frozen=True)
class ChampionRef:
    version: int
    path: str
    promoted_at: float
    source_step: int = 0
    elo: float | None = None


def read_champion(root: str | os.PathLike) -> ChampionRef | None:
    """Self-play workers poll THIS between games; reload the evaluator when ``version`` changes."""
    meta_p = champion_dir(root) / CHAMPION_META
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    wp = champion_dir(root) / str(meta.get("weights", ""))
    if not wp.exists():
        return None
    return ChampionRef(version=int(meta.get("version", 0)), path=str(wp),
                       promoted_at=float(meta.get("promoted_at", 0.0)),
                       source_step=int(meta.get("source_step", 0)),
                       elo=meta.get("elo"))


def seed_champion(root: str | os.PathLike, seed_checkpoint: str | os.PathLike, *,
                  version: int = 0) -> ChampionRef:
    """Install the initial (gen-0) champion from a seed checkpoint (e.g. v3a) so self-play has a net
    to start from before the trainer has promoted anything. Version 0 by convention."""
    ensure_dirs(root)
    dst = champion_dir(root) / f"champion_v{version}.pt"
    _atomic_copy(Path(seed_checkpoint), dst)
    meta = {"version": int(version), "weights": dst.name, "promoted_at": time.time(),
            "source_step": 0, "elo": None, "gate": {"seed": True}}
    _atomic_write_json(champion_dir(root) / CHAMPION_META, meta)
    _archive_append(root, version=version, weights_src=dst, promoted_at=meta["promoted_at"], elo=None)
    return ChampionRef(version=version, path=str(dst), promoted_at=meta["promoted_at"])


def promote(root: str | os.PathLike, candidate: CandidateRef, *, gate: dict | None = None,
            elo: float | None = None) -> ChampionRef:
    """Advance the champion pointer to ``candidate`` (call ONLY after the cheap gate passes). Copies
    the candidate weights into champion/ (self-play source) AND archive/ (opponent pool), atomically
    swaps ``champion.json`` last so a reader never sees a version whose weights aren't present yet."""
    ensure_dirs(root)
    cdir = champion_dir(root)
    dst = cdir / f"champion_v{candidate.version}.pt"
    _atomic_copy(Path(candidate.path), dst)
    _archive_append(root, version=candidate.version, weights_src=dst,
                    promoted_at=time.time(), elo=elo)
    meta = {"version": int(candidate.version), "weights": dst.name, "promoted_at": time.time(),
            "source_step": int(candidate.step), "elo": elo, "gate": gate or {}}
    _atomic_write_json(cdir / CHAMPION_META, meta)  # swap pointer LAST
    _gc_versioned(cdir, "champion_", keep=KEEP_LIVE_CHAMPIONS)  # keep a couple live champion files
    return ChampionRef(version=candidate.version, path=str(dst), promoted_at=meta["promoted_at"],
                       source_step=int(candidate.step), elo=elo)


# --------------------------------------------------------------------------- archive
def _archive_append(root: str | os.PathLike, *, version: int, weights_src: Path,
                    promoted_at: float, elo: float | None) -> None:
    adir = archive_dir(root)
    adir.mkdir(parents=True, exist_ok=True)
    frozen = adir / f"champion_v{version}.pt"
    if not frozen.exists():
        _atomic_copy(weights_src, frozen)
    meta_p = adir / ARCHIVE_META
    entries: list[dict] = []
    if meta_p.exists():
        try:
            entries = json.loads(meta_p.read_text())
        except (json.JSONDecodeError, OSError):
            entries = []
    # de-dup by version (idempotent on re-promotion / resume)
    entries = [e for e in entries if int(e.get("version", -1)) != int(version)]
    entries.append({"version": int(version), "weights": frozen.name,
                    "promoted_at": float(promoted_at), "elo": elo})
    entries.sort(key=lambda e: int(e.get("version", 0)))
    _atomic_write_json(meta_p, entries)
    _gc_archive(root, keep=KEEP_ARCHIVE)


def _gc_archive(root: str | os.PathLike, *, keep: int) -> int:
    """Keep the newest ``keep`` archived champions; drop older files + trim archive.json.

    Always retains the OLDEST surviving entry's file plus the newest ``keep-1`` so the opponent
    pool keeps at least one weak anchor. Returns count removed."""
    adir = archive_dir(root)
    meta_p = adir / ARCHIVE_META
    if not meta_p.exists():
        return 0
    try:
        entries = json.loads(meta_p.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    if len(entries) <= keep:
        return 0
    entries.sort(key=lambda e: int(e.get("version", 0)))
    # CORRECTNESS FIX: keep the OLDEST (a weak diversity anchor — e.g. the gen-0 seed) plus the
    # newest keep-1, NOT a plain newest-keep slice. Prevents the anchor being aged out over a long
    # run, which is exactly what the opponent pool needs to resist role-forgetting drift.
    oldest = entries[0]
    tail = entries[-(keep - 1):] if keep > 1 else []
    survivors = [oldest] + [e for e in tail if int(e["version"]) != int(oldest["version"])]
    survivors.sort(key=lambda e: int(e.get("version", 0)))
    survivor_names = {e["weights"] for e in survivors}
    removed = 0
    for e in entries:
        name = e.get("weights")
        if name and name not in survivor_names:
            try:
                (adir / name).unlink()
                removed += 1
            except OSError:
                pass
    _atomic_write_json(meta_p, survivors)
    return removed


def list_archive(root: str | os.PathLike) -> list[ChampionRef]:
    """Return archived champions (oldest-first) that still have their weight files present."""
    meta_p = archive_dir(root) / ARCHIVE_META
    if not meta_p.exists():
        return []
    try:
        entries = json.loads(meta_p.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    out: list[ChampionRef] = []
    for e in entries:
        wp = archive_dir(root) / str(e.get("weights", ""))
        if wp.exists():
            out.append(ChampionRef(version=int(e.get("version", 0)), path=str(wp),
                                    promoted_at=float(e.get("promoted_at", 0.0)),
                                    elo=e.get("elo")))
    out.sort(key=lambda c: c.version)
    return out


if __name__ == "__main__":  # self-test (pure stdlib; "weights" are just text files)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "flywheel_run"
        ensure_dirs(root)
        assert read_candidate(root) is None and read_champion(root) is None

        # seed champion (gen-0)
        seed = Path(d) / "seed.pt"
        seed.write_text("v3a-seed")
        ch0 = seed_champion(root, seed, version=0)
        assert ch0.version == 0
        assert read_champion(root).version == 0
        assert Path(read_champion(root).path).read_text() == "v3a-seed"
        assert [c.version for c in list_archive(root)] == [0]

        # trainer publishes candidates
        c1 = publish_candidate(root, lambda p: Path(p).write_text("cand1"), step=1000)
        c2 = publish_candidate(root, lambda p: Path(p).write_text("cand2"), step=2000)
        assert c2.version == 2 and read_candidate(root).version == 2
        assert Path(read_candidate(root).path).read_text() == "cand2"

        # promote c2 through the (external) gate
        champ = promote(root, c2, gate={"games": 100, "winrate": 0.58}, elo=25.0)
        assert champ.version == 2
        rc = read_champion(root)
        assert rc.version == 2 and Path(rc.path).read_text() == "cand2" and rc.elo == 25.0
        # champion bytes always match the version the pointer names (H1 discipline)
        assert Path(rc.path).name == "champion_v2.pt"
        # archive now has {0, 2}
        assert [c.version for c in list_archive(root)] == [0, 2]

        # candidate GC keeps only newest KEEP_CANDIDATES
        for s in range(3, 3 + KEEP_CANDIDATES + 2):
            publish_candidate(root, lambda p, s=s: Path(p).write_text(f"c{s}"), step=s * 1000)
        live = sorted(candidates_dir(root).glob("weights_v*.pt"))
        assert len(live) == KEEP_CANDIDATES, live

        # archive GC keeps newest KEEP_ARCHIVE
        for v in range(3, 3 + KEEP_ARCHIVE + 5):
            cN = publish_candidate(root, lambda p, v=v: Path(p).write_text(f"champ{v}"), step=v)
            promote(root, cN)
        arch = list_archive(root)
        assert len(arch) == KEEP_ARCHIVE, len(arch)
        # newest archived champion is the latest promoted version
        assert arch[-1].version == read_champion(root).version
        # CORRECTNESS FIX: the OLDEST anchor (gen-0 seed, version 0) must survive GC, not be
        # aged out by the sliding window — the opponent pool needs a weak-diversity anchor.
        assert arch[0].version == 0, f"seed anchor evicted from archive: {[a.version for a in arch]}"

        # de-dup: re-promoting the same version does not duplicate the archive entry
        n_before = len(list_archive(root))
        again = read_candidate(root)
        promote(root, again)
        assert len(list_archive(root)) == n_before

    print("checkpoint_registry self-test OK")
