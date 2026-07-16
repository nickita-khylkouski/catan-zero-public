#!/usr/bin/env python3
"""Reanalyze-lite v1.3: refresh provenance-qualified per-action Q targets.

WHAT THIS FIXES (CAT-34)
------------------------
This tool batch-forwards the current checkpoint over stored states and may rewrite
``target_scores`` or ``afterstate_target`` only when the checkpoint carries
explicit, validation-bound Q-head provenance. It is not a search reanalyzer and
therefore refuses both root value columns.

WHAT IT REUSES
--------------
* ``train_bc._forward_legal_np_for_batch`` -- the exact featurize + forward path
  the ``--lr 1e-12 --max-steps 1`` probe uses (documented as "~80% of this build").
  Same entity/xdim batch construction, same public-observation masking hook.
* ``train_bc.MemmapCorpus`` / ``tools/build_memmap_corpus.py`` -- the stored-window
  I/O and the exact trimmed-flat on-disk layout. The rewritten column is written
  byte-compatibly so the output loads back through ``MemmapCorpus`` unchanged.
* ``tools/ema_average_checkpoints.py`` -- the ``--reanalyzer-net ema`` option
  (R8's lagged/EMA fallback if drift telemetry is ambiguous).

SAFETY
------
The source corpus is NEVER modified. The tool copies the whole corpus into an
``<corpus>_reanalyzed_<ckpt-tag>`` directory, rewrites exactly ONE column's
``.dat`` in the copy, and hashes every other ``.dat`` before/after to prove no
other column was corrupted. A provenance manifest (source hash, checkpoint md5,
reanalyzer config, before/after stats, timestamp) is written into the copy.

USAGE
-----
Spot-check first (no write; forward a sample, print before/after stats)::

    python tools/reanalyze_lite.py --corpus runs/memmap_corpus_window \
        --checkpoint runs/champion.pt --v-component target_scores \
        --q-head-provenance q_head_provenance.json \
        --sample 20000 --device cuda --batch-size 8192

Full rewrite on a host GPU (inference only, verification/smoke scale)::

    python tools/reanalyze_lite.py --corpus runs/memmap_corpus_window \
        --checkpoint runs/champion.pt --v-component target_scores \
        --q-head-provenance q_head_provenance.json \
        --device cuda --batch-size 8192 \
        --progress-every 50
    # -> runs/memmap_corpus_window_reanalyzed_<tag>/  (+ reanalyze_manifest.json)

Lagged/EMA reanalyzer (R8 fallback when drift telemetry is ambiguous)::

    python tools/reanalyze_lite.py --corpus runs/memmap_corpus_window \
        --reanalyzer-net ema --ema-checkpoints ckpt_gen1.pt ckpt_gen2a.pt \
        --ema-decay 0.75 --v-component target_scores \
        --q-head-provenance q_head_provenance.json --device cuda

Then retrain ONE dose champion-init on the rewritten corpus and gate it against a
same-data control trained on the untouched corpus (both separate steps).

SAFE DEFAULT
------------
There is deliberately no default component. A direct checkpoint forward is not
necessarily the search root evaluator: wide roots can use symmetry averaging and
information-set search can aggregate determinizations. Therefore this lite tool
must never rewrite either ``root_value`` or ``root_prior_value``. Those paired
columns may only be refreshed by a tool that actually reruns the sealed search
operator.

The per-action ``target_scores`` and ``afterstate_target`` modes remain available
only for a checkpoint whose q head has explicit, validated provenance supplied by
``--q-head-provenance``. The provenance is checkpoint-md5-bound and must attest
that the q head was trained for root-to-move search-action values in ``[-1, 1]``
and passed a named validation. See ``docs/REANALYZE_Q_HEAD_PROVENANCE.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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

TOOL_NAME = "reanalyze_lite"
TOOL_VERSION = "1.3"

Q_HEAD_PROVENANCE_SCHEMA = "catan_zero_q_head_provenance_v1"
Q_HEAD_TARGET_SEMANTICS = "root_to_move_search_action_value_v1"

# The two q-value columns this tool can refresh. Both are ragged
#   (N, legal_width), stored trimmed to each row's legal count; refreshed from the
#   q_values head, overwriting only the entries that were finite (masked) before.
V_COMPONENTS: dict[str, dict[str, str]] = {
    "target_scores": {"forward_output": "q_values", "kind": "per_action"},
    "afterstate_target": {"forward_output": "q_values", "kind": "per_action"},
}


def validate_v_component(v_component: str) -> dict[str, str]:
    if v_component in {"root_value", "root_prior_value"}:
        raise SystemExit(
            f"REFUSING --v-component {v_component}: a single stored-feature forward "
            "does not reproduce the sealed root search evaluator. Use "
            "reanalyze_policy_targets.py or the Stage-C search reanalyzer so "
            "root_value and root_prior_value are refreshed atomically."
        )
    spec = V_COMPONENTS.get(v_component)
    if spec is None:
        raise SystemExit(
            f"unknown --v-component {v_component!r}; choices: {sorted(V_COMPONENTS)}"
        )
    return spec


def _is_hex_digest(value: object, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_q_head_provenance(
    provenance: Path | dict | None,
    *,
    reanalyzer_meta: dict,
    v_component: str,
) -> dict | None:
    """Fail closed before any q-head output can become a corpus target.

    Supported components use ``q_values`` and therefore require an explicit record bound to
    the exact checkpoint. A path is normalized into a self-contained manifest
    record (including the provenance file hash); an already-normalized dict is
    accepted when re-validating a banked job manifest at ``run``/``merge`` time.
    """
    spec = validate_v_component(v_component)
    needs_q = spec["forward_output"] == "q_values"
    if not needs_q:
        if provenance is not None:
            raise SystemExit(
                "--q-head-provenance is only valid with a q_values component "
                "(target_scores or afterstate_target)"
            )
        return None

    if provenance is None:
        raise SystemExit(
            f"REFUSING --v-component {v_component}: it rewrites corpus targets from "
            "q_values, but normal train_bc runs freeze an untrained q branch when "
            "q_loss_weight=0. Supply --q-head-provenance for a q head trained and validated with the "
            "required search-action-value semantics."
        )

    source_path: str | None = None
    source_sha256: str | None = None
    if isinstance(provenance, (str, Path)):
        path = Path(provenance)
        if not path.exists():
            raise SystemExit(f"q-head provenance file not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid q-head provenance JSON {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SystemExit(f"q-head provenance {path} must contain a JSON object")
        source_path = str(path.resolve())
        source_sha256 = sha256_file(path)
    elif isinstance(provenance, dict):
        raw = dict(provenance)
        source_path = raw.pop("source_path", None)
        source_sha256 = raw.pop("source_sha256", None)
    else:
        raise SystemExit("q-head provenance must be a JSON path or manifest object")

    errors: list[str] = []
    if raw.get("schema") != Q_HEAD_PROVENANCE_SCHEMA:
        errors.append(
            f"schema must be {Q_HEAD_PROVENANCE_SCHEMA!r} (got {raw.get('schema')!r})"
        )

    expected_md5 = str(reanalyzer_meta.get("md5", "")).lower()
    claimed_md5 = str(raw.get("checkpoint_md5", "")).lower()
    if not _is_hex_digest(claimed_md5, 32):
        errors.append("checkpoint_md5 must be a 32-character hex md5")
    elif claimed_md5 != expected_md5:
        errors.append(
            f"checkpoint_md5 {claimed_md5} does not match reanalyzer {expected_md5}"
        )

    q_head = raw.get("q_head")
    if not isinstance(q_head, dict):
        errors.append("q_head must be an object")
    else:
        if q_head.get("trained") is not True:
            errors.append("q_head.trained must be true")
        if q_head.get("target_semantics") != Q_HEAD_TARGET_SEMANTICS:
            errors.append(
                f"q_head.target_semantics must be {Q_HEAD_TARGET_SEMANTICS!r}"
            )
        value_range = q_head.get("value_range")
        if value_range != [-1.0, 1.0] and value_range != [-1, 1]:
            errors.append("q_head.value_range must be [-1, 1]")

    validation = raw.get("validation")
    if not isinstance(validation, dict):
        errors.append("validation must be an object")
    else:
        if validation.get("passed") is not True:
            errors.append("validation.passed must be true")
        evidence = validation.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(
                "validation.evidence must be a non-empty run/report identifier"
            )

    if errors:
        raise SystemExit("invalid q-head provenance: " + "; ".join(errors))

    normalized = dict(raw)
    normalized["checkpoint_md5"] = claimed_md5
    if source_path is not None:
        normalized["source_path"] = source_path
    if source_sha256 is not None:
        if not _is_hex_digest(source_sha256, 64):
            raise SystemExit(
                "invalid q-head provenance: source_sha256 must be 64 hex chars"
            )
        normalized["source_sha256"] = source_sha256.lower()
    return normalized


# --------------------------------------------------------------------------- #
# Hashing / integrity helpers
# --------------------------------------------------------------------------- #
def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return _hash_file(path, "sha256")


def md5_file(path: Path) -> str:
    return _hash_file(path, "md5")


def hash_corpus_dats(corpus_dir: Path) -> dict[str, str]:
    """sha256 of every ``.dat`` file in a corpus dir, keyed by filename."""
    return {p.name: sha256_file(p) for p in sorted(Path(corpus_dir).glob("*.dat"))}


# --------------------------------------------------------------------------- #
# Reanalyzer-net resolution (checkpoint | ema)
# --------------------------------------------------------------------------- #
def read_checkpoint_metadata(path: Path) -> dict:
    """Return md5 + the metadata fields that determine how to forward the net."""
    import torch

    md5 = md5_file(path)
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a dict checkpoint")
    return {
        "path": str(path),
        "md5": md5,
        "policy_type": str(data.get("policy_type", "") or ""),
        "mask_hidden_info": bool(data.get("mask_hidden_info", False)),
        "action_mask_version": data.get("action_mask_version"),
    }


def resolve_reanalyzer_checkpoint(
    *,
    mode: str,
    checkpoint: Path | None,
    ema_checkpoints: list[Path] | None,
    ema_decay: float,
    work_dir: Path,
) -> tuple[Path, dict]:
    """Return (checkpoint_path, reanalyzer_meta).

    ``checkpoint`` mode uses the given path directly. ``ema`` mode averages the
    given checkpoints (chronological, oldest->newest) via ema_average_checkpoints
    into ``work_dir/reanalyzer_ema.pt`` and uses that -- the R8 lagged/EMA fallback
    that is safer when generation-vs-champion drift telemetry is ambiguous.
    """
    if mode == "checkpoint":
        if checkpoint is None:
            raise SystemExit("--reanalyzer-net checkpoint requires --checkpoint")
        path = Path(checkpoint)
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")
        return path, {"mode": "checkpoint", **read_checkpoint_metadata(path)}

    if mode == "ema":
        if not ema_checkpoints:
            raise SystemExit(
                "--reanalyzer-net ema requires --ema-checkpoints P1 P2 ..."
            )
        from ema_average_checkpoints import (  # noqa: E402
            compute_ema_weights,
            ema_average_checkpoints,
        )

        paths = [Path(p) for p in ema_checkpoints]
        for p in paths:
            if not p.exists():
                raise SystemExit(f"ema checkpoint not found: {p}")
        work_dir.mkdir(parents=True, exist_ok=True)
        averaged_path = work_dir / "reanalyzer_ema.pt"
        ema_average_checkpoints(
            checkpoints=paths, decay=ema_decay, output=averaged_path
        )
        meta = {
            "mode": "ema",
            "ema_decay": float(ema_decay),
            "ema_source_checkpoints": [str(p) for p in paths],
            "ema_weights": compute_ema_weights(len(paths), ema_decay),
            **read_checkpoint_metadata(averaged_path),
        }
        return averaged_path, meta

    raise SystemExit(f"unknown --reanalyzer-net mode {mode!r}")


# --------------------------------------------------------------------------- #
# Batch forward (reuses train_bc's featurize + forward path)
# --------------------------------------------------------------------------- #
def load_policy(checkpoint_path: Path, *, device: str, policy_type: str):
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.xdim_lite_policy import XDimLitePolicy

    if policy_type == "entity_graph":
        return EntityGraphPolicy.load(
            checkpoint_path, device=device, strict_metadata=False
        )
    return XDimLitePolicy.load(checkpoint_path, device=device, strict_metadata=False)


def batch_forward(
    policy,
    corpus,
    indices: np.ndarray,
    *,
    batch_size: int,
    want_q: bool,
    legal_width: int,
    progress_every: int = 0,
    value_materialization: dict | None = None,
) -> dict[str, np.ndarray]:
    """Batch-forward ``policy`` over ``corpus`` rows ``indices`` (inference only).

    Returns ``{"value": (M,)}`` and, when ``want_q``, ``{"q_values": (M, W)}``
    aligned to the corpus legal width. When ``value_materialization`` is supplied,
    ``value`` is the selected readout after the exact search scale/squash/final-
    clip contract, never the model's raw scalar. Reuses
    ``_forward_legal_np_for_batch`` so the entity/xdim batch construction and
    public-observation masking are byte-for-byte the lr-approx-0 probe's path.
    """
    import torch

    from train_bc import _forward_legal_np_for_batch  # noqa: E402

    indices = np.asarray(indices, dtype=np.int64)
    m = int(indices.shape[0])
    values = np.empty(m, dtype=np.float32)
    q_out = np.full((m, legal_width), np.nan, dtype=np.float32) if want_q else None

    started = time.perf_counter()
    policy.model.eval()
    with torch.no_grad():
        for start in range(0, m, batch_size):
            batch = indices[start : start + batch_size]
            legal_action_ids = np.asarray(corpus["legal_action_ids"][batch])
            outputs = _forward_legal_np_for_batch(
                policy,
                corpus,
                batch,
                legal_action_ids,
                return_q=want_q,
            )
            if value_materialization is not None:
                raise SystemExit(
                    "direct value materialization is disabled; rerun the sealed search "
                    "operator to refresh root values"
                )
            if "value" not in outputs:
                raise SystemExit(
                    f"forward pass emitted no 'value' output; keys={sorted(outputs)}"
                )
            value = outputs["value"].detach().float().reshape(-1).cpu().numpy()
            if not np.all(np.isfinite(value)):
                raise SystemExit("forward output 'value' contains non-finite value(s)")
            if value.shape[0] != len(batch):
                raise SystemExit(
                    f"value output has {value.shape[0]} entries for batch size {len(batch)}"
                )
            values[start : start + len(batch)] = value
            if want_q:
                q = outputs["q_values"].detach().float().cpu().numpy()
                # q is (b, W_batch); the corpus pads legal_action_ids to legal_width,
                # so W_batch == legal_width. Guard anyway against a narrower slice.
                w = min(q.shape[1], legal_width)
                q_out[start : start + len(batch), :w] = q[:, :w]
            if progress_every and (start // batch_size + 1) % progress_every == 0:
                elapsed = time.perf_counter() - started
                done = start + len(batch)
                print(
                    json.dumps(
                        {
                            "progress": "reanalyze_forward",
                            "rows_done": int(done),
                            "rows_total": m,
                            "elapsed_s": round(elapsed, 1),
                            "rows_per_s": round(done / max(elapsed, 1e-9), 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    result: dict[str, np.ndarray] = {"value": values}
    if want_q:
        result["q_values"] = q_out
    return result


# --------------------------------------------------------------------------- #
# Column rewrite (byte-compatible with build_memmap_corpus's trimmed-flat layout)
# --------------------------------------------------------------------------- #
def _legal_prefix_mask(corpus, legal_width: int) -> tuple[np.ndarray, np.ndarray]:
    legal_ids = np.asarray(corpus["legal_action_ids"])  # (N, W)
    counts = np.sum(legal_ids >= 0, axis=1).astype(np.int64)
    prefix = np.arange(legal_width)[None, :] < counts[:, None]
    return prefix, counts


def rewrite_per_action_column(
    corpus,
    dst_dir: Path,
    name: str,
    fresh_q: np.ndarray,
    *,
    legal_width: int,
) -> dict:
    """Overwrite the finite (masked) legal entries of a per-action ragged column
    with fresh q-head values, re-trim to the flat prefix layout, and write the
    ``<name>.dat`` in the copy. NaN pads and non-finite legal entries are kept.

    Returns before/after stat inputs (finite entries only) + counts.
    """
    old_padded = np.asarray(corpus[name], dtype=np.float32)  # (N, W), nan pads
    mask_name = f"{name}_mask"
    if mask_name in corpus:
        mask_padded = np.asarray(corpus[mask_name]).astype(bool)
    else:
        mask_padded = np.isfinite(old_padded)
    prefix, counts = _legal_prefix_mask(corpus, legal_width)
    change = mask_padded & prefix & np.isfinite(old_padded)

    new_padded = old_padded.copy()
    new_padded[change] = fresh_q[change].astype(np.float32)

    schema = corpus.meta["columns"][name]
    dtype = np.dtype(schema["dtype"])
    flat_new = np.ascontiguousarray(new_padded[prefix].astype(dtype))
    flat_new.tofile(dst_dir / f"{name}.dat")

    return {
        "changed_files": [f"{name}.dat"],
        "meta_changed": False,
        "rows_total": int(old_padded.shape[0]),
        "entries_rewritten": int(change.sum()),
        "before": old_padded[change].astype(np.float64),
        "after": new_padded[change].astype(np.float64),
        "row_index_per_entry": np.repeat(
            np.arange(old_padded.shape[0]), change.sum(axis=1)
        ),
    }


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
def compute_stats(rewrite: dict, phases: np.ndarray | None) -> dict:
    """Before/after v-component stats: mean shift, correlation, per-phase deltas."""
    before = np.asarray(rewrite["before"], dtype=np.float64)
    after = np.asarray(rewrite["after"], dtype=np.float64)
    has_before = before.shape[0] == after.shape[0] and before.shape[0] > 0

    stats: dict = {
        "entries": int(after.shape[0]),
        "after_mean": float(np.mean(after)) if after.size else None,
        "after_std": float(np.std(after)) if after.size else None,
        "after_min": float(np.min(after)) if after.size else None,
        "after_max": float(np.max(after)) if after.size else None,
    }
    if has_before:
        shift = after - before
        stats.update(
            {
                "before_mean": float(np.mean(before)),
                "before_std": float(np.std(before)),
                "mean_shift": float(np.mean(shift)),
                "mean_abs_shift": float(np.mean(np.abs(shift))),
                "shift_std": float(np.std(shift)),
                "correlation": _safe_corr(before, after),
            }
        )
    if phases is not None and after.size:
        rows = np.asarray(rewrite["row_index_per_entry"], dtype=np.int64)
        per_phase: dict[str, dict] = {}
        entry_phases = np.asarray(phases)[rows].astype(str)
        for phase in sorted(set(entry_phases.tolist())):
            sel = entry_phases == phase
            row = {
                "entries": int(sel.sum()),
                "after_mean": float(np.mean(after[sel])),
            }
            if has_before:
                row["before_mean"] = float(np.mean(before[sel]))
                row["mean_shift"] = float(np.mean(after[sel] - before[sel]))
            per_phase[phase or "unknown"] = row
        stats["per_phase"] = per_phase
    return stats


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape[0] < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _checkpoint_tag(reanalyzer_meta: dict) -> str:
    if reanalyzer_meta["mode"] == "ema":
        return "ema_" + reanalyzer_meta["md5"][:10]
    return Path(reanalyzer_meta["path"]).stem + "_" + reanalyzer_meta["md5"][:10]


def _resolve_mask_hidden_info(cli_value, checkpoint_mask: bool) -> bool:
    if cli_value is None:
        return bool(checkpoint_mask)
    return bool(cli_value)


def verify_reanalyzer_identity(reanalyzer_path: Path, reanalyzer_meta: dict) -> str:
    """Bind caller-supplied metadata to the checkpoint bytes actually forwarded.

    CLI callers normally obtain both values from ``resolve_reanalyzer_checkpoint``,
    but programmatic callers can otherwise pass stale metadata for checkpoint A
    while forwarding checkpoint B. That would also defeat q-provenance's exact-
    checkpoint boundary. Hash the real path at execution time for every mode.
    """
    path = Path(reanalyzer_path)
    if not path.is_file():
        raise SystemExit(f"reanalyzer checkpoint not found: {path}")
    actual_md5 = md5_file(path)
    expected_md5 = str(reanalyzer_meta.get("md5", "")).lower()
    if actual_md5 != expected_md5:
        raise SystemExit(
            f"reanalyzer checkpoint md5 mismatch: metadata pins {expected_md5!r}, "
            f"but {path} hashes to {actual_md5}"
        )
    return actual_md5


def run_reanalyze(
    *,
    corpus_dir: Path,
    out_dir: Path | None,
    reanalyzer_path: Path,
    reanalyzer_meta: dict,
    v_component: str,
    device: str,
    batch_size: int,
    mask_hidden_info: bool | None,
    sample: int | None,
    seed: int,
    progress_every: int,
    q_head_provenance: Path | dict | None = None,
) -> dict:
    import train_bc
    from train_bc import MemmapCorpus

    spec = validate_v_component(v_component)
    verify_reanalyzer_identity(reanalyzer_path, reanalyzer_meta)
    verified_q_provenance = validate_q_head_provenance(
        q_head_provenance,
        reanalyzer_meta=reanalyzer_meta,
        v_component=v_component,
    )
    corpus = MemmapCorpus(corpus_dir)
    legal_width = corpus.legal_width
    n = len(corpus)

    if v_component not in corpus:
        raise SystemExit(
            f"corpus {corpus_dir} has no {v_component!r} column; present columns: "
            f"{sorted(corpus.keys())}"
        )
    policy_type = reanalyzer_meta["policy_type"] or "entity_graph"
    effective_mask = _resolve_mask_hidden_info(
        mask_hidden_info, reanalyzer_meta["mask_hidden_info"]
    )
    # Load-time public-observation masking hook (train_bc global). Must match the
    # regime the reanalyzer net was trained under, or the forward is off-distribution.
    train_bc._MASK_HIDDEN_INFO_PLAYER_TOKENS = bool(effective_mask)

    policy = load_policy(reanalyzer_path, device=device, policy_type=policy_type)
    want_q = spec["forward_output"] == "q_values"

    phases = np.asarray(corpus["phase"]).astype(str) if "phase" in corpus else None

    # ---- Sample / spot-check mode: forward N random rows, print stats, no write.
    if sample is not None:
        rng = np.random.default_rng(seed)
        take = min(int(sample), n)
        idx = np.sort(rng.choice(n, size=take, replace=False).astype(np.int64))
        fwd = batch_forward(
            policy,
            corpus,
            idx,
            batch_size=batch_size,
            want_q=want_q,
            legal_width=legal_width,
            progress_every=progress_every,
            value_materialization=None,
        )
        report = _sample_report(
            corpus,
            idx,
            fwd,
            v_component,
            spec,
            legal_width,
            phases[idx] if phases is not None else None,
        )
        payload = {
            "mode": "sample",
            "corpus": str(corpus_dir),
            "sampled_rows": take,
            "row_count": n,
            "v_component": v_component,
            "forward_output": spec["forward_output"],
            "q_head_provenance": verified_q_provenance,
            "reanalyzer": reanalyzer_meta,
            "mask_hidden_info": bool(effective_mask),
            "device": device,
            "stats": report,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return payload

    # ---- Full mode: forward all rows, copy corpus, rewrite one column, manifest.
    if out_dir is None:
        out_dir = (
            corpus_dir.parent
            / f"{corpus_dir.name}_reanalyzed_{_checkpoint_tag(reanalyzer_meta)}"
        )
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(
            f"output dir already exists (refusing to overwrite): {out_dir}"
        )

    src_hashes = hash_corpus_dats(corpus_dir)
    src_meta_hash = sha256_file(corpus_dir / "corpus_meta.json")

    all_idx = np.arange(n, dtype=np.int64)
    fwd = batch_forward(
        policy,
        corpus,
        all_idx,
        batch_size=batch_size,
        want_q=want_q,
        legal_width=legal_width,
        progress_every=progress_every,
        value_materialization=None,
    )

    shutil.copytree(corpus_dir, out_dir)

    rewrite = rewrite_per_action_column(
        corpus, out_dir, v_component, fwd["q_values"], legal_width=legal_width
    )

    stats = compute_stats(rewrite, phases)

    # Integrity: every .dat except the rewritten one must be byte-identical.
    dst_hashes = hash_corpus_dats(out_dir)
    changed = set(rewrite["changed_files"])
    unexpected = []
    for name, src_hash in src_hashes.items():
        if name in changed:
            continue
        if dst_hashes.get(name) != src_hash:
            unexpected.append(name)
    new_files = sorted(set(dst_hashes) - set(src_hashes) - changed)
    integrity = {
        "unchanged_columns_verified": not unexpected,
        "unexpectedly_changed_files": sorted(unexpected),
        "expected_changed_files": sorted(changed),
        "new_files": new_files,
        "row_count_before": corpus.row_count,
        "row_count_after": MemmapCorpus(out_dir).row_count,
    }
    if unexpected:
        raise SystemExit(
            f"INTEGRITY FAILURE: columns changed unexpectedly: {unexpected}. "
            f"Output corpus at {out_dir} is suspect; investigate before training on it."
        )

    manifest = {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus": {
            "path": str(corpus_dir),
            "row_count": n,
            "legal_width": legal_width,
            "corpus_meta_sha256": src_meta_hash,
            "dat_sha256": src_hashes,
        },
        "output_corpus": str(out_dir),
        "reanalyzer": reanalyzer_meta,
        "v_component": v_component,
        "forward_output": spec["forward_output"],
        "q_head_provenance": verified_q_provenance,
        "mask_hidden_info": bool(effective_mask),
        "device": device,
        "batch_size": batch_size,
        "rows_total": rewrite["rows_total"],
        "entries_rewritten": rewrite["entries_rewritten"],
        "meta_changed": rewrite["meta_changed"],
        "integrity": integrity,
        "stats": stats,
        "perspective_note": (
            "Fresh Q comes from the same root-to-move policy class whose archived "
            "per-action estimate it replaces, so no sign flip is applied."
        ),
    }
    (out_dir / "reanalyze_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "progress": "reanalyze_lite_done",
                "output_corpus": str(out_dir),
                "v_component": v_component,
                "entries_rewritten": rewrite["entries_rewritten"],
                "mean_shift": stats.get("mean_shift"),
                "correlation": stats.get("correlation"),
                "unchanged_columns_verified": integrity["unchanged_columns_verified"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


def _sample_report(corpus, idx, fwd, v_component, spec, legal_width, sample_phases):
    """Before/after stats on a sampled row subset (no write)."""
    if spec["kind"] == "per_action":
        old_padded = np.asarray(corpus[v_component][idx], dtype=np.float32)
        mask_name = f"{v_component}_mask"
        if mask_name in corpus:
            mask_padded = np.asarray(corpus[mask_name][idx]).astype(bool)
        else:
            mask_padded = np.isfinite(old_padded)
        legal_ids = np.asarray(corpus["legal_action_ids"][idx])
        counts = np.sum(legal_ids >= 0, axis=1).astype(np.int64)
        prefix = np.arange(legal_width)[None, :] < counts[:, None]
        change = mask_padded & prefix & np.isfinite(old_padded)
        fresh = fwd["q_values"]
        rewrite = {
            "before": old_padded[change].astype(np.float64),
            "after": fresh[change].astype(np.float64),
            "row_index_per_entry": np.repeat(np.arange(len(idx)), change.sum(axis=1)),
        }
    else:
        existed = v_component in corpus
        old = (
            np.asarray(corpus[v_component][idx], dtype=np.float64).reshape(-1)
            if existed
            else np.empty(0, dtype=np.float64)
        )
        after = np.asarray(fwd["value"], dtype=np.float64).reshape(-1)
        rewrite = {
            "before": old if existed else np.empty(0, dtype=np.float64),
            "after": after,
            "row_index_per_entry": np.arange(len(idx)),
        }
    return compute_stats(rewrite, sample_phases)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="memmap corpus dir (build_memmap_corpus.py output)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output corpus dir (default: <corpus>_reanalyzed_<ckpt-tag>); must not exist",
    )
    parser.add_argument(
        "--v-component",
        required=True,
        choices=sorted(V_COMPONENTS),
        help="q-value component to rewrite; requires --q-head-provenance. Value "
        "columns require a true search reanalysis and are intentionally unsupported.",
    )
    parser.add_argument(
        "--q-head-provenance",
        type=Path,
        default=None,
        help="required JSON provenance for target_scores/afterstate_target q_values "
        "rewrites; must be bound to this checkpoint and attest trained, validated "
        "root-to-move search-action-value semantics",
    )
    parser.add_argument(
        "--reanalyzer-net",
        default="checkpoint",
        choices=("checkpoint", "ema"),
        help="reanalyzer net: a single checkpoint (--checkpoint), or an EMA/lagged "
        "average of several (--ema-checkpoints). EMA is the R8 fallback when "
        "generation-vs-champion drift telemetry is ambiguous.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint for --reanalyzer-net checkpoint",
    )
    parser.add_argument(
        "--ema-checkpoints",
        nargs="+",
        type=Path,
        default=None,
        help="checkpoints (CHRONOLOGICAL: oldest->newest) for --reanalyzer-net ema",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.75,
        help="EMA decay for --reanalyzer-net ema",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for the forward pass (cpu | cuda | cuda:0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="forward-pass batch size"
    )
    parser.add_argument(
        "--mask-hidden-info",
        dest="mask_hidden_info",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="public-observation player-token masking during the forward pass. "
        "Default: inherit the reanalyzer checkpoint's mask_hidden_info flag.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="spot-check mode: forward N random rows, print before/after v-component "
        "stats (mean shift, correlation, per-phase deltas), and DO NOT write.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for --sample row selection"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="log a progress line every N forward batches",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    work_dir = (
        args.out or (args.corpus.parent / f"{args.corpus.name}_reanalyzed")
    ).parent
    reanalyzer_path, reanalyzer_meta = resolve_reanalyzer_checkpoint(
        mode=args.reanalyzer_net,
        checkpoint=args.checkpoint,
        ema_checkpoints=args.ema_checkpoints,
        ema_decay=args.ema_decay,
        work_dir=work_dir,
    )
    run_reanalyze(
        corpus_dir=args.corpus,
        out_dir=args.out,
        reanalyzer_path=reanalyzer_path,
        reanalyzer_meta=reanalyzer_meta,
        v_component=args.v_component,
        device=args.device,
        batch_size=args.batch_size,
        mask_hidden_info=args.mask_hidden_info,
        sample=args.sample,
        seed=args.seed,
        progress_every=args.progress_every,
        q_head_provenance=args.q_head_provenance,
    )


if __name__ == "__main__":
    main()
