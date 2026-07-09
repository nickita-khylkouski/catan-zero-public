"""Continuous KataGo-hybrid flywheel orchestrator (Step 7, the continuous regime).

Continuous counterpart to ``tools/selfplay_loop.py`` (the DISCRETE generational baseline). Same
external scripts (generate / build_memmap / train / gate), different data-flow:

  DISCRETE (selfplay_loop.py):  gen full batch -> merge -> 1-epoch train -> gate -> promote -> repeat
                                (barrier every step; trains only on THIS gen's shards)

  CONTINUOUS (this file):       trainer consumes a GROWING WINDOW of all recent shards + emits
                                candidates; a CHEAP gate advances the champion pointer; self-play
                                workers hot-reload the champion and play 15-25% vs the archived
                                opponent pool. No barrier; GPUs never idle on a gate.

Research basis + knobs: ``catan_zero.rl.flywheel.config.FlywheelConfig`` (memory
``catan-discrete-vs-continuous-verdict``). ``--regime`` selects continuous|discrete so the SAME
driver runs BOTH arms of the (never-cleanly-published) discrete-vs-continuous ablation.

STATE (all under --loop-dir, resumable, atomic per round):
  window_state.json     WindowedReplay registry (+ monotonic total_rows_ever)
  {candidates,champion,archive}/   checkpoint_registry layout
  flywheel_state.json   loop journal: rounds, candidates, promotions, gate verdicts. Promotion is
                         committed in TWO phases (promotion_pending: true -> false) so a crash
                         mid-promote leaves a reconcilable trailing record instead of silent
                         desync between the journal and checkpoint_registry's actual champion;
                         reconcile_pending_promotion() repairs this on the next startup.
  flywheel_config.json  resolved FlywheelConfig (name-keyed)
  flywheel.lock          exclusive, non-blocking fcntl lock held for the process lifetime (two
                         orchestrators on the same --loop-dir would otherwise race window_state.json)
  gen/round_NNN/.round_done   per-round completion marker (atomicity: window+journal persisted only
                              at round end; a crashed round's out_dir is wiped + redone cleanly)
  corpus/round_NNN/      windowed memmap corpus for the round; PRIOR rounds' corpora are deleted
                         once a round's train step succeeds (only the current + about-to-be-
                         superseded round survive on disk -- see cleanup_old_corpora()).
  gates/round_NNN.json   h2h gate summary JSON (tools/gumbel_search_cross_net_h2h.py output)

THREE INTEGRATION FOLLOW-UPS this orchestrator depends on for the FULL design (flagged, not yet in
the pipeline — see the review findings):
  (H1) checkpoint hot-reload: gumbel_self_play.run_worker_games loads ONE frozen net per process.
       Continuous generation needs it to poll checkpoint_registry.read_champion() between games and
       set ``mcts.evaluator = <new>`` on a version bump (insertion point: the per-game loop).
  (H2) opponent-pool wiring: generate_gumbel_selfplay_data.py + play_one_game must accept a per-seat
       (per-color) evaluator so opponent_pool.choose_opponent(game_index) can set the non-champion
       seat. This is a play_one_game signature change, not just a call insertion.
  (T3/T4) train_bc needs a bounded-step mode (``--max-steps``; today only ``--epochs`` exists) and
       build_memmap needs incremental append/evict (today it rebuilds the whole window each round).

Until those land, ``--regime continuous`` runs in "relaunch" mode: each round relaunches a bounded
self-play batch against the *current* champion file (a fast discrete cadence with a windowed trainer
+ opponent pool at the batch boundary) — a correct-but-suboptimal stand-in this file makes explicit
rather than hiding. The real-path subprocess commands below are the VERIFIED-correct argparse forms
(a review pass caught several invented/missing flags); the two lines still needing a trainer change
are marked ``NEEDS T3``/``NEEDS T4`` inline.

Run ``--dry-run`` to walk the full control flow with stub gen/train/gate (no fleet, no torch).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from catan_zero.rl.flywheel import (  # noqa: E402
    FlywheelConfig, WindowedReplay, ensure_dirs, seed_champion, read_champion,
    read_candidate, publish_candidate, promote, list_archive,
)

import launcher_guards  # noqa: E402

# Round-seed stride: each round's game_seed block starts here*round, disjoint across rounds so
# game_seed = base + game_index never collides between rounds (the seed-collision bug class).
SEED_STRIDE = 10_000_019  # prime, matches selfplay_loop.py's stride convention

# Gate seeds live in a DISJOINT additive space from generation seeds (finding #1): generation seeds
# span [base_seed, base_seed + max_rounds*SEED_STRIDE + games_per_round), so a large fixed offset
# (comfortably beyond any plausible max_rounds*SEED_STRIDE footprint) plus a different per-round
# stride guarantees gate seeds never collide with training-data seeds, in either direction.
GATE_SEED_BASE_OFFSET = 100_000_000_000  # 1e11
GATE_SEED_STRIDE = 10_000_103            # distinct prime from SEED_STRIDE


# ------------------------------------------------------------------ journal
def load_journal(loop_dir: Path) -> dict:
    p = loop_dir / "flywheel_state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"rounds": [], "started_at": None, "total_rows_trained": 0}


def save_journal(loop_dir: Path, j: dict) -> None:
    _atomic_json(loop_dir / "flywheel_state.json", j)


def save_config(loop_dir: Path, cfg: FlywheelConfig) -> None:
    _atomic_json(loop_dir / "flywheel_config.json", cfg.to_dict())


def record_anchor_telemetry(
    loop_dir: Path, round_idx: int, candidate_version: int, results: dict, cfg: FlywheelConfig,
) -> dict:
    """Append this round's anchor-probe ``results`` to the longitudinal per-anchor
    trend file ``anchor_telemetry.json`` (one growing list per anchor name, never
    truncated/overwritten). PURELY OBSERVATIONAL: this function only appends+prints;
    it returns the updated telemetry dict but nothing in ``main()`` may branch a
    promotion decision on that return value (see ``Runner.anchor_probe``'s docstring).

    Also prints a WARNING (tripwire alert, never an exception, never a hold/promote
    override) if a value_mse regresses more than ``cfg.anchor_drift_alert_threshold``
    (relative) against that anchor's OWN first-recorded baseline round."""
    path = loop_dir / "anchor_telemetry.json"
    telemetry: dict = {}
    if path.exists():
        try:
            telemetry = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            telemetry = {}
    for name, metrics in results.items():
        history = telemetry.setdefault(name, [])
        record = {
            "round": round_idx, "candidate_version": candidate_version,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **metrics,
        }
        history.append(record)
        baseline = history[0]
        value_mse = metrics.get("value_mse")
        baseline_mse = baseline.get("value_mse")
        if (
            isinstance(value_mse, (int, float)) and isinstance(baseline_mse, (int, float))
            and baseline_mse > 0
        ):
            drift = (value_mse - baseline_mse) / baseline_mse
            if drift > cfg.anchor_drift_alert_threshold:
                print(f"[flywheel] ANCHOR DRIFT TRIPWIRE ({name}): value_mse {value_mse:.4f} is "
                      f"{drift:.1%} above baseline {baseline_mse:.4f} (round {baseline['round']}) -- "
                      "informational only, does NOT affect promotion (tripwire, never a gate).",
                      flush=True)
    _atomic_json(path, telemetry)
    return telemetry


def _atomic_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, p)


# ------------------------------------------------------------------ scan shards
def scan_new_shards(gen_out_dir: Path) -> list[tuple[str, int]]:
    """Discover this round's shards + row counts WITHOUT opening npz twice.

    Row-count source order (cheapest first):
      1. manifest.json entries that carry an explicit ``rows`` (dict-shaped);
      2. a ``<shard>.rows`` sidecar we write once, so an npz header is opened at most ONCE per shard
         over the whole run (the real generator currently emits ``shards: [path, ...]`` with no row
         counts — the proper long-term fix is to have it emit rows; this sidecar bounds the cost
         until then instead of re-opening every shard every round).
    Scoped to this round's dir only (never a full-history rglob)."""
    out: list[tuple[str, int]] = []
    if not gen_out_dir.exists():
        return out
    manifest_rows: dict[str, int] = {}
    manifest = gen_out_dir / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            for entry in m.get("shards", []):
                if isinstance(entry, dict) and "path" in entry and "rows" in entry:
                    manifest_rows[str(entry["path"])] = int(entry["rows"])
                    manifest_rows[Path(str(entry["path"])).name] = int(entry["rows"])
        except (json.JSONDecodeError, OSError):
            pass
    for npz in sorted(gen_out_dir.rglob("*.npz")):
        key = str(npz)
        # falsy-zero-safe lookup (a legit rows==0 must not fall through to the npz open).
        if key in manifest_rows:
            rows: int | None = manifest_rows[key]
        elif npz.name in manifest_rows:
            rows = manifest_rows[npz.name]
        else:
            rows = _npz_rows_cached(npz)
        if rows is not None:
            out.append((key, int(rows)))
    return out


def _npz_rows_cached(npz: Path) -> int | None:
    sidecar = npz.with_suffix(npz.suffix + ".rows")
    if sidecar.exists():
        try:
            return int(sidecar.read_text().strip())
        except (ValueError, OSError):
            pass
    rows = _npz_rows(npz)
    if rows is not None:
        try:
            sidecar.write_text(str(rows))
        except OSError:
            pass
    return rows


def _npz_rows(npz: Path) -> int | None:
    try:
        import numpy as np
        with np.load(npz, allow_pickle=True) as z:
            key = "game_seed" if "game_seed" in z else list(z.keys())[0]
            return int(z[key].shape[0])
    except Exception:
        return None


# ------------------------------------------------------------------ round steps (real / stub)
class Runner:
    """Wraps the external scripts. ``dry_run`` swaps every subprocess for a deterministic stub so the
    control flow is fully testable without a fleet or torch."""

    def __init__(self, cfg: FlywheelConfig, loop_dir: Path, *, dry_run: bool,
                 workers: int, device: str, base_seed: int):
        self.cfg = cfg
        self.loop_dir = loop_dir
        self.dry_run = dry_run
        self.workers = workers
        self.device = device
        self.base_seed = base_seed

    def generate(self, champion_path: str, round_idx: int, out_dir: Path, n_games: int) -> dict:
        """Bounded self-play batch vs ``champion_path`` (+ opponent pool). Round-derived base seed
        keeps game_seed disjoint across rounds. INTEGRATION (H1/H2): the real generator must
        hot-reload the champion and wire the opponent pool per-game — the pool manifest below is
        written for the generator to consume once H2 lands."""
        if self.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            shards = []
            for k in range(2):
                sp = out_dir / f"stub_shard_{round_idx:03d}_{k}.npz"
                sp.write_bytes(b"")
                shards.append({"path": str(sp), "rows": 1024})
            (out_dir / "manifest.json").write_text(json.dumps({"shards": shards, "rows": 2048}))
            return {"ok": True, "out_dir": str(out_dir), "note": "dry-run stub", "rows": 2048}
        archive = list_archive(self.loop_dir)
        pool_manifest_path = out_dir.parent / f"opponent_pool_r{round_idx:03d}.json"
        pool_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # H2 LANDED: this is the exact schema tools/generate_gumbel_selfplay_data.py's
        # --opponent-pool-manifest consumes ("opponents" with "checkpoint" keys, NOT the old
        # "archive"/"path" draft), via read_opponent_pool_manifest in gumbel_self_play.py.
        _atomic_json(pool_manifest_path, {
            "pool_fraction": self.cfg.opponent_pool_fraction, "champion": champion_path,
            "opponents": [{"checkpoint": a.path, "version": a.version} for a in archive]})
        round_seed = self.base_seed + round_idx * SEED_STRIDE  # disjoint per round (collision fix)
        cmd = [
            _py(), "tools/generate_gumbel_selfplay_data.py",
            "--out-dir", str(out_dir), "--games", str(n_games),
            "--checkpoint", champion_path, "--public-observation",
            "--workers", str(self.workers), "--device", self.device,
            "--base-seed", str(round_seed),
        ]
        # CAT-88: pin the generation search config EXPLICITLY from FlywheelConfig.
        # resolve_gen_search_argv() RAISES if any field is unset -- the flywheel never
        # silently inherits generate_gumbel_selfplay_data.py's tool defaults (c_scale 0.1
        # vs 0.03, temperature-decisions 45 vs 90, lazy-interior-chance OFF vs ON = the
        # "unvalidated preset incl D1" drift). gen config is run-dependent, so the operator
        # MUST set it (volume n64/p0.25 vs teacher n128/p1.0); no safe hardcoded default.
        cmd += self.cfg.resolve_gen_search_argv()
        if archive and self.cfg.opponent_pool_fraction > 0.0:
            # Pool games only make sense once at least one archived champion exists
            # (round 0 has no archive -> pure mirror self-play, flag omitted).
            cmd += ["--opponent-pool-manifest", str(pool_manifest_path)]
        code = _run(cmd, self.loop_dir / "generation.log")
        man = out_dir / "manifest.json"
        return {"ok": code == 0 and man.exists(), "out_dir": str(out_dir),
                "note": "relaunch-mode (H1/H2 pending)", "exit_code": code, "round_seed": round_seed}

    def train_window(self, window_paths: list[str], init_ckpt: str, round_idx: int,
                     new_rows_this_round: int) -> dict:
        """Build a windowed corpus over ``window_paths`` and train a bounded number of steps, then
        publish a candidate. Steps are sized so INCREMENTAL reuse on this round's NEW rows tracks
        target_reuse (not the whole window — the reuse-math bug the review caught)."""
        steps = self._planned_steps(new_rows_this_round)
        if self.dry_run:
            cand = publish_candidate(self.loop_dir, lambda p: Path(p).write_text(f"cand-{round_idx}"),
                                     step=steps)
            return {"ok": True, "candidate": cand.path, "version": cand.version, "steps": steps}
        corpus_dir = self.loop_dir / "corpus" / f"round_{round_idx:03d}"
        src_root = corpus_dir / "window_src"
        src_root.mkdir(parents=True, exist_ok=True)
        # build_memmap takes --source DIRECTORY ROOTS each with a manifest.json (NOT --source-list,
        # which doesn't exist). Write a synthetic manifest listing the in-window shard paths.
        _atomic_json(src_root / "manifest.json", {"shards": window_paths, "rows": None})
        build = _run([_py(), "tools/build_memmap_corpus.py",
                      "--source", str(src_root), "--out", str(corpus_dir)],
                     self.loop_dir / "corpus.log")  # NEEDS T4: incremental append/evict (full rebuild today)
        if build != 0:
            return {"ok": False, "note": "memmap build failed", "exit_code": build}
        ckpt = corpus_dir / "candidate.pt"
        report = corpus_dir / "report.json"
        train = _run([_py(), "tools/train_bc.py", "--arch", "entity_graph",
                      "--data-format", "memmap", "--data", str(corpus_dir),
                      "--init-checkpoint", init_ckpt, "--checkpoint", str(ckpt),
                      "--report", str(report),           # REQUIRED flag (was missing)
                      "--batch-size", str(self.cfg.train_batch_size),  # must match reuse math
                      "--mask-hidden-info", "--amp", "bf16",
                      "--max-steps", str(steps)],        # NEEDS T3: add --max-steps to train_bc (only --epochs today)
                     self.loop_dir / "train.log")
        if train != 0 or not ckpt.exists():
            return {"ok": False, "note": "train failed (or --max-steps unsupported: NEEDS T3)",
                    "exit_code": train, "steps": steps}
        cand = publish_candidate(self.loop_dir, lambda p: shutil.copyfile(ckpt, p), step=steps)
        return {"ok": True, "candidate": cand.path, "version": cand.version, "steps": steps}

    def _gate_seed(self, round_idx: int) -> int:
        # Disjoint additive space + a different stride from generation's round_seed (see the
        # GATE_SEED_* module constants) -- gate games can NEVER replay a training-data seed.
        return self.base_seed + GATE_SEED_BASE_OFFSET + round_idx * GATE_SEED_STRIDE

    def _h2h_gate_cmd(self, candidate_path: str, champion_path: str, round_idx: int, out: Path) -> list[str]:
        pairs = max(1, self.cfg.gate_games // 2)  # total games played = 2x pairs (color-swapped)
        cmd = [
            _py(), "tools/gumbel_search_cross_net_h2h.py",
            "--candidate", candidate_path, "--baseline", champion_path,
            "--pairs", str(pairs),
            "--n-full", str(self.cfg.gate_sims),   # reduced sim budget for a cheap, low-variance gate
            "--c-scale", str(self.cfg.gate_c_scale),  # NEVER rely on the tool's 0.1 default (drift trap)
            "--devices", self.device,
            "--workers", str(self.workers),
            "--base-seed", str(self._gate_seed(round_idx)),
            "--out", str(out),
        ]
        if self.cfg.gate_lazy_interior_chance:
            cmd.append("--lazy-interior-chance")
        if self.cfg.masked:
            # Symmetric hidden-info masking for BOTH nets' search (f72): required so a candidate
            # trained with train_bc --mask-hidden-info is gated on the SAME observation regime it
            # was trained under, instead of the omniscient-feature train/eval mismatch this was
            # replacing (finding #1). force_full=True is already hardcoded in play_one_h2h_game, so
            # gate_sims (n_full) alone controls the reduced gate budget -- no separate "force full"
            # switch is needed on this tool.
            cmd.append("--public-observation")
        return cmd

    def gate(self, candidate_path: str, champion_path: str, round_idx: int) -> dict:
        """Cheap KataGo-style gate. Returns {ok, pass, verdict}. ALLOWLIST: promote ONLY on an
        explicit accept-H1 verdict — SPRT reject (H0) or continue (inconclusive) are both a HOLD,
        never a promote (the review caught this: !=reject previously silently promoted inconclusive
        candidates). KataGo-style >=50%-winrate promotion (rather than requiring a significant SPRT
        win) is a config choice for later, not implemented on this path yet."""
        if not self.cfg.gate_enabled:
            return {"ok": True, "pass": True, "verdict": "disabled", "note": "gate disabled"}
        if self.cfg.gate_style == "scoreboard":
            print("[flywheel] WARNING: gate_style='scoreboard' (tools/promotion_gate_runner.py -> "
                  "evaluate_scoreboard.py) does NOT apply hidden-info masking anywhere in its policy "
                  "-loading chain. A masked-trained candidate gated on this path is evaluated with "
                  "omniscient features (train/eval mismatch). Use gate_style='h2h' (default) unless "
                  "you have a specific reason not to.", flush=True)
            return self._gate_scoreboard(candidate_path, champion_path, round_idx)
        return self._gate_h2h(candidate_path, champion_path, round_idx)

    def _gate_h2h(self, candidate_path: str, champion_path: str, round_idx: int) -> dict:
        out = self.loop_dir / "gates" / f"round_{round_idx:03d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._h2h_gate_cmd(candidate_path, champion_path, round_idx, out)
        if self.dry_run:
            # Surface the exact planned gate command (masking flag, gate_sims -> --n-full, disjoint
            # gate seed) so dry-run control-flow coverage includes the real gate path, without
            # actually running search games.
            print(f"[flywheel][dry-run] gate cmd (round {round_idx}): {shlex.join(cmd)}", flush=True)
            wr = 0.50 + max(0.0, 0.06 - 0.01 * round_idx)
            v = "promote" if wr >= self.cfg.gate_min_winrate else "reject"
            return {"ok": True, "pass": v == "promote", "verdict": v, "winrate": round(wr, 3),
                    "note": "dry-run stub (h2h gate path)", "cmd": cmd}
        code = _run(cmd, self.loop_dir / "gate.log")
        summary = None
        if out.exists():
            try:
                summary = json.loads(out.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        sprt = (summary or {}).get("pentanomial_sprt") or {}
        decision = sprt.get("decision")  # "H1" (candidate better) | "H0" (reject) | "continue"
        promoted = decision == "H1"
        return {"ok": code == 0 and summary is not None, "pass": promoted, "verdict": decision,
                "sprt": sprt, "winrate": (summary or {}).get("candidate_win_rate")}

    def _gate_scoreboard(self, candidate_path: str, champion_path: str, round_idx: int) -> dict:
        """DEPRECATED gate path (finding #1): kept only behind ``gate_style="scoreboard"`` as an
        explicit opt-out. Does NOT mask hidden info -- see the warning in :meth:`gate`."""
        if self.dry_run:
            wr = 0.50 + max(0.0, 0.06 - 0.01 * round_idx)
            v = "promote" if wr >= self.cfg.gate_min_winrate else "reject"
            return {"ok": True, "pass": v == "promote", "verdict": v, "winrate": round(wr, 3),
                    "note": "dry-run stub (DEPRECATED scoreboard gate path, unmasked)"}
        out = self.loop_dir / "gate" / f"round_{round_idx:03d}.json"
        legs = self.loop_dir / "gate" / f"round_{round_idx:03d}_legs"
        legs.mkdir(parents=True, exist_ok=True)
        code = _run([_py(), "tools/promotion_gate_runner.py",
                     "--candidate", candidate_path, "--baseline", champion_path,
                     "--games-per-leg", str(self.cfg.gate_games),
                     "--leg-dir", str(legs),          # REQUIRED flag (was missing)
                     "--out", str(out)], self.loop_dir / "gate.log")
        verdict = None
        if out.exists():
            try:
                verdict = json.loads(out.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        vstr = (verdict or {}).get("verdict")
        return {"ok": code == 0 and verdict is not None, "pass": vstr == "promote",
                "verdict": vstr, "sprt": (verdict or {}).get("h2h_sprt")}

    def anchor_probe(self, candidate_path: str, round_idx: int) -> dict:
        """Tripwire-ONLY anchor-corpus telemetry (CAT-30). Evaluates ``candidate_path``
        against each anchor in ``cfg.anchor_corpora`` (built by
        ``tools/build_anchor_corpus.py``) and returns ``{anchor_name: {...}}``.

        DECISION RULE (Roadmap Sec 1 / R8 gen-4 lesson): this is drift telemetry, NEVER
        a promotion signal. The caller (``main()``) MUST call this only for logging/
        storage, after the promotion decision (``g.get("pass")`` from ``gate()``) has
        already been computed -- this method's return value must never feed back into
        ``rec["decision"]``. Enforced by ``tests/test_continuous_flywheel_anchor_tripwire.py``,
        which asserts (via AST inspection of ``main``) that no ``if``/boolean expression
        computing ``rec["decision"]`` or ``g.get("pass")`` references ``anchor_probe`` or
        its result variable.

        Uses the same "lr(asymptotically)-0 probe" technique as the project's prior
        one-off anchor-holdout telemetry (a single optimizer step at a vanishing
        learning rate = a pure forward-pass evaluation through the real training code
        path, with --validation-fraction 0 so the WHOLE anchor corpus is scored, no
        split needed -- the anchor is never trained on, gradients or not, because the
        LR keeps any update imperceptible)."""
        if not self.cfg.anchor_corpora or self.cfg.anchor_eval_every_rounds <= 0:
            return {}
        if round_idx % self.cfg.anchor_eval_every_rounds != 0:
            return {}
        results: dict[str, dict] = {}
        anchors_root = self.loop_dir / "anchors"
        for name in self.cfg.anchor_corpora:
            corpus_dir = anchors_root / name
            if self.dry_run:
                # Stub, matching generate()/gate()'s dry-run convention: exercise the full
                # control flow (including record_anchor_telemetry below) without requiring
                # a real anchor corpus on disk or a torch/GPU forward pass.
                results[name] = {"ok": True, "note": "dry-run stub", "policy_ce": 1.0, "value_mse": 0.1}
                continue
            if not corpus_dir.exists():
                print(f"[flywheel] anchor tripwire: {name!r} not found at {corpus_dir} "
                      "(build it with tools/build_anchor_corpus.py) -- skipping this round.",
                      flush=True)
                continue
            report_path = self.loop_dir / "anchor_reports" / f"{name}_round_{round_idx:03d}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            scratch_ckpt = self.loop_dir / "anchor_reports" / f"{name}_scratch.pt"
            code = _run([
                _py(), "tools/train_bc.py", "--arch", "entity_graph",
                "--data-format", "memmap", "--data", str(corpus_dir),
                "--init-checkpoint", candidate_path, "--checkpoint", str(scratch_ckpt),
                "--report", str(report_path),
                "--epochs", "1", "--max-steps", "1", "--learning-rate", "1e-12",
                "--validation-fraction", "0.0", "--mask-hidden-info", "--amp", "bf16",
            ], self.loop_dir / "anchor_probe.log")
            summary = None
            if report_path.exists():
                try:
                    summary = json.loads(report_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            if code != 0 or summary is None:
                results[name] = {"ok": False, "note": "anchor probe run failed"}
                continue
            metrics = (summary.get("epochs") or [{}])[-1] if isinstance(summary.get("epochs"), list) else {}
            results[name] = {
                "ok": True,
                "policy_ce": metrics.get("train_policy_loss"),
                "value_mse": metrics.get("train_value_loss"),
            }
        return results

    def _planned_steps(self, new_rows_this_round: int) -> int:
        """steps = new_rows * target_reuse / batch_size — reuse tracks the INCREMENTAL new data, and
        uses the REAL trainer batch size (not a hardcoded 4096 that was 16x off)."""
        batch = max(1, self.cfg.train_batch_size)
        return max(1, int(new_rows_this_round * self.cfg.target_reuse / batch))


def _py() -> str:
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _run(cmd: list[str], log_path: Path) -> int:
    import subprocess
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} $ {shlex.join(cmd)}\n")
        log.flush()
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT)).returncode


# ------------------------------------------------------------------ corpus cleanup (finding #3)
def cleanup_old_corpora(loop_dir: Path, keep_round_idx: int) -> list[str]:
    """Delete ``corpus/round_NNN`` dirs from PRIOR rounds once THIS round's train step has
    succeeded. The current round's corpus is kept until the next round supersedes it (so a fresh
    failure is still debuggable on disk). Without this, up to ``--max-rounds`` full-window memmap
    corpora accumulate under ``--loop-dir`` and exhaust disk on a long run.

    Guards ``shutil.rmtree`` to paths that are actually under ``loop_dir/corpus`` and match the
    ``round_*`` naming convention, so a bad ``keep_round_idx`` or a symlink can never make this
    delete something outside the corpus directory."""
    corpus_root = (loop_dir / "corpus").resolve()
    removed: list[str] = []
    if not corpus_root.is_dir():
        return removed
    keep_name = f"round_{keep_round_idx:03d}"
    for d in sorted(corpus_root.glob("round_*")):
        if d.name == keep_name or not d.is_dir():
            continue
        resolved = d.resolve()
        if corpus_root not in resolved.parents:
            continue  # safety guard: never rmtree outside loop_dir/corpus/round_*
        shutil.rmtree(resolved, ignore_errors=True)
        removed.append(str(d))
    return removed


# ------------------------------------------------------------------ crash-atomicity (finding #4)
def reconcile_pending_promotion(loop_dir: Path, journal: dict, gen_root: Path) -> None:
    """Startup crash-recovery for the two-phase promotion journal: if the LAST round record was
    left with ``promotion_pending=True`` (the process died between committing that pending record
    and ``promote()`` finishing), reconcile the journal against ``checkpoint_registry``'s actual
    champion -- the registry is the durable source of truth, the journal is just an audit trail.
    Logs the reconciliation either way. A crash before the pending record was even committed leaves
    no trace here (by design: nothing durable happened yet for that round)."""
    rounds = journal.get("rounds") or []
    if not rounds:
        return
    last = rounds[-1]
    if not last.get("promotion_pending"):
        return
    champ = read_champion(loop_dir)
    champ_version = champ.version if champ else None
    print(f"[flywheel] RECOVERY: round {last.get('round')} was left with promotion_pending=True "
          f"(process died between the journal commit and promote() completing). "
          f"checkpoint_registry's actual champion is v{champ_version}. Reconciling the journal to "
          f"match the registry (registry state wins).", flush=True)
    last["promotion_pending"] = False
    last["promoted_version"] = champ_version
    last["decision"] = (f"promote(v{champ_version})_reconciled" if champ_version is not None
                        else "promote_reconcile_failed_no_champion")
    last["reconciled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_journal(loop_dir, journal)
    round_idx = last.get("round")
    if round_idx is not None:
        out_dir = gen_root / f"round_{int(round_idx):03d}"
        if out_dir.exists():
            try:
                (out_dir / ".round_done").write_text(last["decision"])
            except OSError:
                pass


# ------------------------------------------------------------------ interprocess lock (finding #6)
def acquire_loop_lock(loop_dir: Path):
    """Exclusive, non-blocking fcntl lock on a lockfile in ``loop_dir``, held for the process
    lifetime (the returned file handle must stay referenced; closing/GC'ing it releases the lock).
    Prevents two orchestrators racing on ``window_state.json`` read-modify-write. Returns None if
    the lock is already held elsewhere."""
    lock_path = loop_dir / "flywheel.lock"
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(f"pid={os.getpid()} at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    fh.flush()
    return fh


# ------------------------------------------------------------------ main loop
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loop-dir", required=True)
    p.add_argument("--seed-checkpoint", required=True, help="gen-0 champion (e.g. v3a masked)")
    p.add_argument("--regime", choices=["continuous", "discrete"], default="continuous")
    p.add_argument("--window-c-rows", type=int, default=300_000)
    p.add_argument("--opponent-pool-fraction", type=float, default=0.20)
    p.add_argument("--gate-games", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=65536, help="MUST match train_bc --batch-size")
    p.add_argument("--games-per-round", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--base-seed", type=int, default=20_260_705)
    p.add_argument("--max-rounds", type=int, default=100)
    p.add_argument("--gen-out-root", default=None, help="shard output root (default <loop-dir>/gen)")
    # CAT-88: generation search config -- REQUIRED, no defaults (run-dependent: volume
    # n64/p0.25 vs teacher n128/p1.0). Unset -> resolve_gen_search_argv() raises at the
    # first generation round, so the flywheel never silently inherits generate_gumbel's
    # tool defaults. Threaded verbatim onto every generate subprocess.
    p.add_argument("--gen-n-full", type=int, default=None, help="CAT-88 gen search: full-sim budget (e.g. volume 64, teacher 128)")
    p.add_argument("--gen-n-fast", type=int, default=None, help="CAT-88 gen search: fast-sim budget (e.g. 16)")
    p.add_argument("--gen-p-full", type=float, default=None, help="CAT-88 gen search: full-sim probability (e.g. volume 0.25, teacher 1.0)")
    p.add_argument("--gen-c-visit", type=float, default=None, help="CAT-88 gen search: c_visit (e.g. 50.0)")
    p.add_argument("--gen-c-scale", type=float, default=None, help="CAT-88 gen search: c_scale (canonical 0.03, NOT the tool default 0.1)")
    p.add_argument("--gen-max-decisions", type=int, default=None, help="CAT-88 gen search: max decisions/game (e.g. 600)")
    p.add_argument("--gen-max-depth", type=int, default=None, help="CAT-88 gen search: max search depth (e.g. 80)")
    p.add_argument("--gen-temperature-decisions", type=int, default=None, help="CAT-88 gen search: temperature decisions (canonical 90, NOT the tool default 45)")
    p.add_argument("--gen-lazy-interior-chance", action=argparse.BooleanOptionalAction, default=None, help="CAT-88 gen search: lazy-interior-chance (canonical ON, NOT the tool default OFF)")
    p.add_argument("--gen-correct-rust-chance-spectra", action=argparse.BooleanOptionalAction, default=None, help="CAT-88 gen search: correct-rust-chance-spectra (canonical ON)")
    p.add_argument("--evict-stale-shards", action="store_true",
                   help="physically delete out-of-window shards each round (recommended for long runs)")
    p.add_argument("--evict-grace-seconds", type=float, default=0.0,
                   help="defer deletion until a shard has been out-of-window this long (async safety)")
    p.add_argument("--dry-run", action="store_true", help="stub gen/train/gate; exercise control flow only")
    p.add_argument(
        "--anchor-corpus", dest="anchor_corpora", action="append", default=[],
        help="Name of an anchor corpus (built by tools/build_anchor_corpus.py, lives at "
        "<loop-dir>/anchors/<name>) to probe each round as DRIFT TELEMETRY ONLY -- never "
        "a promotion signal (Roadmap Sec 1 standing rule, R8/gen-4 lesson). Repeatable: "
        "pass once per anchor, e.g. --anchor-corpus anchor_r7 --anchor-corpus anchor_gen4 "
        "for the full longitudinal series.",
    )
    p.add_argument(
        "--anchor-holdout-ranges", default="",
        help="Informational provenance only: comma-separated start:end .valonly game_seed "
        "ranges the --anchor-corpus entries were built from. Recorded in "
        "flywheel_config.json; does not affect probe behavior.",
    )
    p.add_argument("--anchor-eval-every-rounds", type=int, default=1,
                    help="Probe cadence in rounds; 0 disables the anchor tripwire entirely.")
    p.add_argument("--anchor-drift-alert-threshold", type=float, default=0.10,
                    help="Relative value_mse increase vs an anchor's first-recorded baseline "
                    "that triggers a WARNING log line (alert only, never gates promotion).")
    p.add_argument(
        "--skip-guards",
        action="store_true",
        help=(
            "Skip tools/prelaunch_guard.py's pre-launch checks (CLI-default-override "
            "trap, masked-regime mismatch on --seed-checkpoint, resumed-config "
            "provenance drift, fd-limit, lock availability; CAT-69/CAT-75). Logs a "
            "loud WARNING and proceeds anyway -- use only for a known false positive "
            "or an intentional smoke test. No-op under --dry-run (guards never run "
            "there in the first place)."
        ),
    )
    return p


def _build_guard_specs(
    args: argparse.Namespace, argv: Sequence[str], parser: argparse.ArgumentParser, loop_dir: Path
) -> list[dict]:
    static_specs = launcher_guards.load_static_guard_specs("continuous_flywheel")
    dynamic = {
        "cli_flag_lint": {"argv": list(argv), "parser": parser},
        "masked_regime": {"checkpoint_path": args.seed_checkpoint},
        "lock_available": {"lock_path": str(loop_dir / "flywheel.lock")},
    }
    specs = launcher_guards.merge_dynamic_args(static_specs, dynamic)
    # Provenance (d): only meaningful when RESUMING a loop that already recorded a
    # flywheel_config.json -- a fresh loop has nothing to compare against yet, so
    # this guard is added conditionally rather than always running (guard_provenance
    # FAILs on a missing report, which would be the wrong verdict for a fresh start).
    existing_config_path = loop_dir / "flywheel_config.json"
    if existing_config_path.exists():
        specs.append(
            {
                "name": "provenance",
                "args": {
                    "report_path": existing_config_path,
                    "claims": {
                        "regime": args.regime,
                        "window_c_rows": args.window_c_rows,
                        "train_batch_size": args.batch_size,
                    },
                },
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    loop_dir = Path(args.loop_dir)
    loop_dir.mkdir(parents=True, exist_ok=True)
    gen_root = Path(args.gen_out_root) if args.gen_out_root else loop_dir / "gen"

    # Guards never run under --dry-run: dry-run exercises control flow only (stub
    # gen/train/gate, no fleet, no torch) and legitimately points --seed-checkpoint
    # at a fixture path a real masked_regime guard would correctly refuse.
    if not args.dry_run:
        launcher_guards.run_or_refuse(
            _build_guard_specs(
                args, argv if argv is not None else sys.argv[1:], parser, loop_dir
            ),
            launcher="continuous_flywheel",
            skip=bool(args.skip_guards),
        )

    # Interprocess lock (finding #6): two orchestrators pointed at the same --loop-dir would race on
    # window_state.json's read-modify-write. Held for the whole process lifetime (non-blocking; a
    # second process exits immediately with a clear message instead of silently corrupting state).
    _lock_fh = acquire_loop_lock(loop_dir)
    if _lock_fh is None:
        print(f"[flywheel] ERROR: another orchestrator already holds the lock on "
              f"{loop_dir / 'flywheel.lock'} (--loop-dir {loop_dir}). Exiting.", flush=True)
        return 1

    cfg = FlywheelConfig(
        regime=args.regime, window_c_rows=args.window_c_rows,
        opponent_pool_fraction=args.opponent_pool_fraction, gate_games=args.gate_games,
        train_batch_size=args.batch_size, evict_stale_shards=args.evict_stale_shards,
        anchor_corpora=list(args.anchor_corpora), anchor_holdout_ranges=args.anchor_holdout_ranges,
        anchor_eval_every_rounds=args.anchor_eval_every_rounds,
        anchor_drift_alert_threshold=args.anchor_drift_alert_threshold,
        # CAT-88: gen search config (run-dependent; None -> raises at first generation).
        gen_n_full=args.gen_n_full, gen_n_fast=args.gen_n_fast, gen_p_full=args.gen_p_full,
        gen_c_visit=args.gen_c_visit, gen_c_scale=args.gen_c_scale,
        gen_max_decisions=args.gen_max_decisions, gen_max_depth=args.gen_max_depth,
        gen_temperature_decisions=args.gen_temperature_decisions,
        gen_lazy_interior_chance=args.gen_lazy_interior_chance,
        gen_correct_rust_chance_spectra=args.gen_correct_rust_chance_spectra,
    ).validate()
    save_config(loop_dir, cfg)

    ensure_dirs(loop_dir)
    window = WindowedReplay(loop_dir / "window_state.json", c=cfg.window_c_rows,
                            alpha=cfg.window_alpha, beta=cfg.window_beta)
    if read_champion(loop_dir) is None:
        seed_champion(loop_dir, args.seed_checkpoint, version=0)
        print(f"[flywheel] seeded champion v0 from {args.seed_checkpoint}", flush=True)

    journal = load_journal(loop_dir)
    if journal["started_at"] is None:
        journal["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    reconcile_pending_promotion(loop_dir, journal, gen_root)  # finding #4: crash-recovery
    runner = Runner(cfg, loop_dir, dry_run=args.dry_run,
                    workers=args.workers, device=args.device, base_seed=args.base_seed)

    start_round = len(journal["rounds"])  # completed rounds are journaled; resume redoes only the tail
    for round_idx in range(start_round, args.max_rounds):
        if (loop_dir / "STOP").exists():
            print("[flywheel] STOP file present — exiting cleanly.", flush=True)
            break
        champ = read_champion(loop_dir)
        rec: dict = {"round": round_idx, "champion_version": champ.version, "regime": cfg.regime,
                     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

        # ATOMICITY: window + journal are persisted only at round end. A crashed round is not
        # journaled, so on resume we wipe its (partial) out_dir and redo it cleanly — no half-
        # registered shards, no ckpt_version mislabelling.
        out_dir = gen_root / f"round_{round_idx:03d}"
        if out_dir.exists() and not (out_dir / ".round_done").exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        gen = runner.generate(champ.path, round_idx, out_dir, args.games_per_round)
        rec["generate"] = gen
        if not gen.get("ok"):
            rec["decision"] = "abort_generation"
            journal["rounds"].append(rec); save_journal(loop_dir, journal)
            print(f"[round {round_idx}] generation failed — stopping.", flush=True)
            return 1

        new_shards = scan_new_shards(out_dir)
        new_rows = sum(r for _, r in new_shards)
        window.register_many(new_shards, ckpt_version=champ.version)
        sel = window.select()  # compute ONCE, thread everywhere (was 3x per round)
        if cfg.evict_stale_shards:
            window.evict(delete=True, grace_seconds=args.evict_grace_seconds, selection=sel)
        rec["window"] = {"total_rows_ever": window.total_rows_ever, "live_rows": window.total_rows,
                         "window_rows": sel.window_rows, "in_window_shards": len(sel.in_window),
                         "new_rows": new_rows}

        # Zero-row round detection (finding #5): a generation round that yields no new data must
        # not silently retrain ~1 step on stale data (or crash train_bc on an empty round-0 corpus).
        # Skip train+gate entirely, count consecutive occurrences, and abort after 3 in a row —
        # that many zero-row rounds back to back means the generator itself is broken upstream.
        if new_rows == 0:
            streak = int(journal.get("consecutive_zero_rows", 0)) + 1
            journal["consecutive_zero_rows"] = streak
            rec["decision"] = "skip_zero_rows"
            print(f"[round {round_idx}] WARNING: generation produced 0 new rows this round "
                  f"(consecutive zero-row rounds: {streak}/3). Skipping train+gate.", flush=True)
            window.save()
            journal["rounds"].append(rec)
            save_journal(loop_dir, journal)
            try:
                (out_dir / ".round_done").write_text(rec["decision"])
            except OSError:
                pass
            if streak >= 3:
                print(f"[flywheel] ABORT: {streak} consecutive zero-row generation rounds — "
                      f"the generator is producing no data; stopping.", flush=True)
                return 1
            continue
        journal["consecutive_zero_rows"] = 0

        tr = runner.train_window([s.path for s in sel.in_window], champ.path, round_idx, new_rows)
        rec["train"] = tr
        if not tr.get("ok"):
            rec["decision"] = "abort_train"
            window.save(); journal["rounds"].append(rec); save_journal(loop_dir, journal)
            return 1
        # actual rows consumed by the trainer this round (steps*batch), not the window size.
        journal["total_rows_trained"] = journal.get("total_rows_trained", 0) + int(tr["steps"]) * cfg.train_batch_size

        # Corpus cleanup (finding #3): the trainer succeeded and a candidate was published from
        # corpus/round_{round_idx:03d}, so every PRIOR round's memmap corpus can go — keep only the
        # current round's on disk (debuggable) until the next round's train step supersedes it.
        removed = cleanup_old_corpora(loop_dir, round_idx)
        if removed:
            rec["corpus_cleanup"] = removed
            print(f"[round {round_idx}] cleaned up stale corpus dirs: {removed}", flush=True)

        cand = read_candidate(loop_dir)
        g = runner.gate(cand.path, champ.path, round_idx)
        rec["gate"] = g

        # Two-phase promotion journal (finding #4): commit the round's verdict + a
        # "promotion_pending" marker BEFORE calling promote(), so a crash mid-promote leaves an
        # auditable trailing record (reconciled on the next startup by
        # reconcile_pending_promotion) instead of silently redoing the round against whatever
        # champion the registry ends up with, with no trace of what happened.
        round_committed = False
        if g.get("pass"):
            rec["decision"] = "promoting"
            rec["promotion_pending"] = True
            window.save()
            journal["rounds"].append(rec)
            save_journal(loop_dir, journal)
            round_committed = True

            new_champ = promote(loop_dir, cand, gate=g, elo=None)
            rec["decision"] = f"promote(v{new_champ.version})"
            rec["promotion_pending"] = False
            rec["promoted_version"] = new_champ.version
            journal["rounds"][-1] = rec
            save_journal(loop_dir, journal)
            print(f"[round {round_idx}] promoted candidate v{cand.version} (verdict={g.get('verdict')})", flush=True)
        else:
            rec["decision"] = f"hold(gate_{g.get('verdict')})"  # distinguishes continue vs reject
            print(f"[round {round_idx}] candidate v{cand.version} HELD (verdict={g.get('verdict')}); "
                  f"champion v{champ.version} unchanged.", flush=True)

        # Anchor tripwire telemetry (CAT-30): computed and recorded strictly AFTER
        # rec["decision"] is already final, on purpose -- this is drift telemetry only
        # (Roadmap Sec 1 standing rule, R8/gen-4 lesson) and must never be able to
        # influence the promotion decision above. Evaluated against the CANDIDATE
        # (not the possibly-just-promoted champion) so the signal is about this
        # round's produced model regardless of whether it was promoted or held.
        anchor_results = runner.anchor_probe(cand.path, round_idx)
        if anchor_results:
            rec["anchor_telemetry"] = anchor_results
            record_anchor_telemetry(loop_dir, round_idx, cand.version, anchor_results, cfg)

        # H2 (opponent-pool wiring) is NOT implemented: generate_gumbel_selfplay_data.py has no
        # per-seat evaluator wiring yet, so every self-play game this round was champion-vs-champion
        # mirror play regardless of cfg.opponent_pool_fraction. Reporting a computed-but-fictional
        # H2 LANDED: the generator now plays real pool games and reports the realized fraction in
        # its own manifest.json (opponent_pool_fraction_realized + per-version champion win rates).
        # Read the MEASURED value from there; 0.0-with-marker only when no pool ran this round
        # (empty archive / round 0 / disabled) -- never a plausible-looking fabricated number.
        rec["opponent_pool_realized"] = 0.0
        rec["opponent_pool_note"] = "no pool this round (empty archive, round 0, or disabled)"
        gen_manifest = Path(gen.get("out_dir", "")) / "manifest.json" if gen.get("out_dir") else None
        if gen_manifest is not None and gen_manifest.exists():
            try:
                gen_stats = json.loads(gen_manifest.read_text())
                if "opponent_pool_fraction_realized" in gen_stats:
                    rec["opponent_pool_realized"] = gen_stats["opponent_pool_fraction_realized"]
                    rec["opponent_pool_versions_used"] = gen_stats.get("opponent_pool_versions_used")
                    rec["opponent_pool_champion_winrates"] = gen_stats.get(
                        "opponent_pool_per_version_champion_winrate")
                    rec["opponent_pool_note"] = "measured from generation manifest"
            except (json.JSONDecodeError, OSError):
                rec["opponent_pool_note"] = "generation manifest unreadable; realized fraction unknown"

        # persist window + journal together, THEN mark the round done (atomic completion)
        window.save()
        if round_committed:
            journal["rounds"][-1] = rec
        else:
            journal["rounds"].append(rec)
        save_journal(loop_dir, journal)
        try:
            (out_dir / ".round_done").write_text(rec["decision"])
        except OSError:
            pass

    cur = read_champion(loop_dir)
    print(f"[flywheel] done: {len(journal['rounds'])} rounds, champion v{cur.version}, "
          f"{journal.get('total_rows_trained', 0):,} rows trained "
          f"(N_total={window.total_rows_ever:,}).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
