#!/usr/bin/env python3
"""Reanalyze-lite v1: batch-forward a checkpoint over a stored window and rewrite
the lambda-blend's value component into a COPY of the corpus.

WHAT THIS FIXES (CAT-34)
------------------------
The reanalyze value-target design (docs/REANALYZE_VALUE_TARGETS_DESIGN_20260704.md)
blends the Monte-Carlo outcome ``z`` with a search-derived value component into the
training value target::

    v_target = lambda * z + (1 - lambda) * V_component

The persisted ``V_component`` is the *generation-time* network's own estimate
(``target_scores`` = ``result.q_values`` per legal action, produced while that
same net was generating the game). Reusing a net's own archived value as its own
regression target is a self-distillation amplifier -- the effective-rank-collapse
failure mode Kumar et al. (2010.14498) describe, and the reason MuZero-Reanalyse
pins the value-loss weight at 0.25. It is plausibly co-causal in the reported
+69% drift incident.

"Reanalyze-lite" is the cheap remedy that extracts more from data already paid
for, with ZERO new games: batch-forward the CURRENT champion (or a lagged/EMA net)
over the stored states and overwrite the value component with those fresh forward
passes. This is a batch FORWARD pass, not a full re-search (that is the larger,
separate "full selective reanalyze" graduation step, CAT-63) -- and it is a
HYPOTHESIS TEST, not a guaranteed fix: a fresh search-consistent forward value may
or may not beat the stale search-completed value. The selected scalar/categorical
readout is materialized with search's exact scale, squash, and final clip; the
retrain + gate (a separate step, run on the output of this tool) is what decides.

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
        --checkpoint runs/champion.pt --sample 20000 --device cuda --batch-size 8192

Full rewrite on a host GPU (inference only, verification/smoke scale)::

    python tools/reanalyze_lite.py --corpus runs/memmap_corpus_window \
        --checkpoint runs/champion.pt --device cuda --batch-size 8192 \
        --progress-every 50
    # -> runs/memmap_corpus_window_reanalyzed_<tag>/  (+ reanalyze_manifest.json)

Lagged/EMA reanalyzer (R8 fallback when drift telemetry is ambiguous)::

    python tools/reanalyze_lite.py --corpus runs/memmap_corpus_window \
        --reanalyzer-net ema --ema-checkpoints ckpt_gen1.pt ckpt_gen2a.pt \
        --ema-decay 0.75 --device cuda

Then retrain ONE dose champion-init on the rewritten corpus and gate it against a
same-data control trained on the untouched corpus (both separate steps).

SAFE DEFAULT
------------
The default is ``--v-component root_value``. It materialises a fresh per-state
``V(s)`` column from the configured trained value readout, using the same
``value_scale`` / scalar ``value_squash`` / final ``[-1, 1]`` clip as search. The
full contract is persisted in the column schema and manifest. This is the column
consumed by ``train_bc --value-target-lambda``. Normal training uses
``q_loss_weight=0`` and freezes the q branch, so silently replacing
``target_scores`` from that branch would turn random/untrained outputs into
training targets.

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
TOOL_VERSION = "1.2"
DEFAULT_V_COMPONENT = "root_value"

Q_HEAD_PROVENANCE_SCHEMA = "catan_zero_q_head_provenance_v1"
Q_HEAD_TARGET_SEMANTICS = "root_to_move_search_action_value_v1"
ROOT_VALUE_MATERIALIZATION_SCHEMA = "catan_zero_root_value_materialization_v1"
ROOT_VALUE_TARGET_SEMANTICS = "root_to_move_search_backup_value_v1"
ROOT_VALUE_RANGE = [-1.0, 1.0]

# The three value-component columns the tool can refresh, and which forward-pass
# output feeds each. Only ``target_scores`` is present in a standard memmap corpus
# (build_memmap_corpus.LOADER_KEYS carries it but drops afterstate_target /
# root_value, which live in gumbel_self_play.EXTRA_KEYS); the other two are
# supported for corpora that carry them or when materialising a fresh per-state
# ``root_value`` column for a future per-state value-target-lambda.
#
# * per_action columns (target_scores, afterstate_target) are ragged
#   (N, legal_width), stored trimmed to each row's legal count; refreshed from the
#   q_values head, overwriting only the entries that were finite (masked) before.
# * per_state columns (root_value) are a scalar (N,) fixed column; refreshed from
#   the configured search value readout after its exact scale/squash/final clip.
#   If absent it is MATERIALISED (a new .dat + a provenance-bearing schema entry).
V_COMPONENTS: dict[str, dict[str, str]] = {
    "target_scores": {"forward_output": "q_values", "kind": "per_action"},
    "afterstate_target": {"forward_output": "q_values", "kind": "per_action"},
    "root_value": {"forward_output": "value", "kind": "per_state"},
}


def resolve_root_value_materialization(
    *,
    value_readout: str = "scalar",
    value_squash: str = "tanh",
    value_scale: float = 1.0,
) -> dict:
    """Return the canonical, search-equivalent root-value materialization spec.

    This mirrors :class:`EntityGraphRustEvaluator` exactly: scalar search reads
    ``outputs["value"]`` and applies ``tanh(raw * scale)`` (or the experimental
    clip-only path); categorical search reads the already-calibrated
    ``outputs["value_categorical"]``, bypasses scalar tanh, then both paths apply
    the evaluator's final ``[-1, 1]`` clip.  Persisting this complete spec keeps a
    corpus target tied to the readout actually used by search instead of merely
    saying that it came from an unspecified "value head".
    """
    readout = str(value_readout)
    squash = str(value_squash)
    try:
        scale = float(value_scale)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--value-scale must be a finite positive float (got {value_scale!r})") from exc
    if readout not in {"scalar", "categorical"}:
        raise SystemExit(
            f"unknown --value-readout {readout!r}; expected 'scalar' or 'categorical'"
        )
    if squash not in {"tanh", "clip"}:
        raise SystemExit(
            f"unknown --value-squash {squash!r}; expected 'tanh' or 'clip'"
        )
    if not np.isfinite(scale) or scale <= 0.0:
        raise SystemExit(f"--value-scale must be finite and > 0 (got {value_scale!r})")
    return {
        "schema": ROOT_VALUE_MATERIALIZATION_SCHEMA,
        "target_semantics": ROOT_VALUE_TARGET_SEMANTICS,
        "value_readout": readout,
        "forward_output": "value" if readout == "scalar" else "value_categorical",
        "value_scale": scale,
        "configured_value_squash": squash,
        "applied_value_squash": squash if readout == "scalar" else "none",
        "final_clip": True,
        "value_range": list(ROOT_VALUE_RANGE),
    }


def validate_root_value_materialization(
    provenance: dict | None,
    *,
    v_component: str,
) -> dict | None:
    """Validate a persisted materialization record, rejecting legacy ambiguity.

    Per-action Q targets do not use this record.  ``root_value`` jobs must carry
    the full canonical record at plan, run, and merge time; otherwise a pre-v1.2
    banked job containing raw, unsquashed scalar outputs could be mistaken for
    search-backup values.
    """
    if v_component != "root_value":
        if provenance is not None:
            raise SystemExit(
                "root-value materialization provenance is only valid with "
                "--v-component root_value"
            )
        return None
    if not isinstance(provenance, dict):
        raise SystemExit(
            "root_value requires search-consistent value materialization provenance; "
            "legacy/raw-forward jobs must be re-planned"
        )
    try:
        canonical = resolve_root_value_materialization(
            value_readout=provenance.get("value_readout"),
            value_squash=provenance.get("configured_value_squash"),
            value_scale=provenance.get("value_scale"),
        )
    except SystemExit as exc:
        raise SystemExit(f"invalid root-value materialization provenance: {exc}") from exc
    if provenance != canonical:
        mismatched = sorted(
            key
            for key in set(provenance) | set(canonical)
            if provenance.get(key) != canonical.get(key)
        )
        raise SystemExit(
            "invalid root-value materialization provenance: non-canonical/mismatched "
            f"field(s) {mismatched}"
        )
    return canonical


def materialize_search_root_values(outputs: dict, provenance: dict) -> np.ndarray:
    """Convert one forward batch into the exact bounded value search backs up."""
    spec = validate_root_value_materialization(provenance, v_component="root_value")
    assert spec is not None
    key = spec["forward_output"]
    if key not in outputs:
        raise SystemExit(
            f"value_readout={spec['value_readout']!r} requires forward output {key!r}; "
            f"model emitted {sorted(outputs)} (refusing fallback to another head)"
        )
    tensor = outputs[key]
    try:
        raw = tensor.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
    except AttributeError as exc:
        raise SystemExit(f"forward output {key!r} is not a tensor-like value") from exc
    if not np.all(np.isfinite(raw)):
        bad = int((~np.isfinite(raw)).sum())
        raise SystemExit(f"forward output {key!r} contains {bad} non-finite value(s)")

    scaled = raw * float(spec["value_scale"])
    if spec["applied_value_squash"] == "tanh":
        bounded = np.tanh(scaled)
    elif spec["applied_value_squash"] in {"clip", "none"}:
        bounded = scaled
    else:  # Defensive even after canonical validation.
        raise SystemExit(
            f"unsupported applied value squash {spec['applied_value_squash']!r}"
        )
    # This is the final clip performed at every search call site, after squash.
    bounded = np.clip(bounded, ROOT_VALUE_RANGE[0], ROOT_VALUE_RANGE[1])
    if not np.all(np.isfinite(bounded)):
        raise SystemExit("materialized root values contain non-finite value(s)")
    if np.any(bounded < ROOT_VALUE_RANGE[0]) or np.any(bounded > ROOT_VALUE_RANGE[1]):
        raise SystemExit("materialized root values escaped the declared [-1, 1] range")
    return bounded.astype(np.float32, copy=False)


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

    ``root_value`` uses the value head and needs no q provenance. Per-action
    components use ``q_values`` and therefore require an explicit record bound to
    the exact checkpoint. A path is normalized into a self-contained manifest
    record (including the provenance file hash); an already-normalized dict is
    accepted when re-validating a banked job manifest at ``run``/``merge`` time.
    """
    spec = V_COMPONENTS.get(v_component)
    if spec is None:
        raise SystemExit(
            f"unknown --v-component {v_component!r}; choices: {sorted(V_COMPONENTS)}"
        )
    needs_q = spec["forward_output"] == "q_values"
    if not needs_q:
        if provenance is not None:
            raise SystemExit(
                "--q-head-provenance is only valid with a q_values component "
                "(target_scores or afterstate_target); root_value uses the trained value head"
            )
        return None

    if provenance is None:
        raise SystemExit(
            f"REFUSING --v-component {v_component}: it rewrites corpus targets from "
            "q_values, but normal train_bc runs freeze an untrained q branch when "
            "q_loss_weight=0. Use --v-component root_value (the safe default), or "
            "supply --q-head-provenance for a q head trained and validated with the "
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
            errors.append("validation.evidence must be a non-empty run/report identifier")

    if errors:
        raise SystemExit("invalid q-head provenance: " + "; ".join(errors))

    normalized = dict(raw)
    normalized["checkpoint_md5"] = claimed_md5
    if source_path is not None:
        normalized["source_path"] = source_path
    if source_sha256 is not None:
        if not _is_hex_digest(source_sha256, 64):
            raise SystemExit("invalid q-head provenance: source_sha256 must be 64 hex chars")
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
            raise SystemExit("--reanalyzer-net ema requires --ema-checkpoints P1 P2 ...")
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
        ema_average_checkpoints(checkpoints=paths, decay=ema_decay, output=averaged_path)
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
        return EntityGraphPolicy.load(checkpoint_path, device=device, strict_metadata=False)
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
            if value_materialization is None:
                if "value" not in outputs:
                    raise SystemExit(
                        f"forward pass emitted no 'value' output; keys={sorted(outputs)}"
                    )
                value = outputs["value"].detach().float().reshape(-1).cpu().numpy()
                if not np.all(np.isfinite(value)):
                    raise SystemExit("forward output 'value' contains non-finite value(s)")
            else:
                value = materialize_search_root_values(outputs, value_materialization)
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
        "row_index_per_entry": np.repeat(np.arange(old_padded.shape[0]), change.sum(axis=1)),
    }


def rewrite_per_state_column(
    corpus,
    dst_dir: Path,
    name: str,
    fresh_values: np.ndarray,
    *,
    write_mask: np.ndarray | None = None,
    column_provenance: dict | None = None,
) -> dict:
    """Write/overwrite a per-state scalar column with fresh value-head V(s).

    Materialises the column if the corpus does not already carry it (updates the
    copy's corpus_meta.json with a scalar ``fixed`` schema entry). Rows outside
    ``write_mask`` are left NaN (mixable-rows-only discipline; a corpus with no
    full-search flag defaults to writing every row). ``column_provenance`` is
    embedded in the column schema and values are rejected unless they satisfy its
    declared finite ``[-1, 1]`` target contract. This catches callers that bypass
    the canonical batch-forward materializer.
    """
    n = int(len(corpus))
    fresh_values = np.asarray(fresh_values, dtype=np.float32).reshape(-1)
    if fresh_values.shape[0] != n:
        raise SystemExit(f"fresh_values length {fresh_values.shape[0]} != row_count {n}")
    if write_mask is None:
        write_mask = np.ones(n, dtype=bool)
    write_mask = np.asarray(write_mask, dtype=bool).reshape(-1)
    if write_mask.shape[0] != n:
        raise SystemExit(f"write_mask length {write_mask.shape[0]} != row_count {n}")

    canonical_provenance = None
    if column_provenance is not None:
        canonical_provenance = validate_root_value_materialization(
            column_provenance, v_component=name
        )
        selected = fresh_values[write_mask]
        if not np.all(np.isfinite(selected)):
            raise SystemExit("refusing to write non-finite materialized root_value target(s)")
        tolerance = 1.0e-6
        if np.any(selected < ROOT_VALUE_RANGE[0] - tolerance) or np.any(
            selected > ROOT_VALUE_RANGE[1] + tolerance
        ):
            lo = float(np.min(selected))
            hi = float(np.max(selected))
            raise SystemExit(
                "refusing out-of-range materialized root_value target(s): "
                f"observed [{lo}, {hi}], required [-1, 1]"
            )

    existed = name in corpus
    old = np.asarray(corpus[name], dtype=np.float64).reshape(-1) if existed else None

    col = np.full(n, np.nan, dtype=np.float32)
    col[write_mask] = fresh_values[write_mask]
    np.ascontiguousarray(col).tofile(dst_dir / f"{name}.dat")

    meta_changed = False
    if not existed or canonical_provenance is not None:
        meta_path = dst_dir / "corpus_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        schema = {"kind": "fixed", "dtype": "<f4", "inner_shape": []}
        if canonical_provenance is not None:
            schema["target_semantics"] = ROOT_VALUE_TARGET_SEMANTICS
            schema["materialization"] = canonical_provenance
        if meta["columns"].get(name) != schema:
            meta["columns"][name] = schema
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            meta_changed = True

    changed = write_mask
    return {
        "changed_files": [f"{name}.dat"],
        "meta_changed": meta_changed,
        "rows_total": n,
        "entries_rewritten": int(write_mask.sum()),
        "before": (old[changed] if existed else np.empty(0, dtype=np.float64)),
        "after": col[changed].astype(np.float64),
        "row_index_per_entry": np.flatnonzero(changed),
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
    value_readout: str = "scalar",
    value_squash: str = "tanh",
    value_scale: float = 1.0,
) -> dict:
    import train_bc
    from train_bc import MemmapCorpus

    if v_component not in V_COMPONENTS:
        raise SystemExit(f"unknown --v-component {v_component!r}; choices: {sorted(V_COMPONENTS)}")
    verify_reanalyzer_identity(reanalyzer_path, reanalyzer_meta)
    spec = V_COMPONENTS[v_component]
    verified_q_provenance = validate_q_head_provenance(
        q_head_provenance,
        reanalyzer_meta=reanalyzer_meta,
        v_component=v_component,
    )
    root_value_provenance = (
        resolve_root_value_materialization(
            value_readout=value_readout,
            value_squash=value_squash,
            value_scale=value_scale,
        )
        if v_component == "root_value"
        else None
    )

    corpus = MemmapCorpus(corpus_dir)
    legal_width = corpus.legal_width
    n = len(corpus)

    if spec["kind"] == "per_action" and v_component not in corpus:
        raise SystemExit(
            f"corpus {corpus_dir} has no {v_component!r} column; present columns: "
            f"{sorted(corpus.keys())}"
        )

    policy_type = reanalyzer_meta["policy_type"] or "entity_graph"
    effective_mask = _resolve_mask_hidden_info(mask_hidden_info, reanalyzer_meta["mask_hidden_info"])
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
            policy, corpus, idx, batch_size=batch_size, want_q=want_q,
            legal_width=legal_width, progress_every=progress_every,
            value_materialization=root_value_provenance,
        )
        report = _sample_report(
            corpus, idx, fwd, v_component, spec, legal_width,
            phases[idx] if phases is not None else None,
        )
        payload = {
            "mode": "sample",
            "corpus": str(corpus_dir),
            "sampled_rows": take,
            "row_count": n,
            "v_component": v_component,
            "forward_output": (
                root_value_provenance["forward_output"]
                if root_value_provenance is not None
                else spec["forward_output"]
            ),
            "q_head_provenance": verified_q_provenance,
            "root_value_materialization": root_value_provenance,
            "reanalyzer": reanalyzer_meta,
            "mask_hidden_info": bool(effective_mask),
            "device": device,
            "stats": report,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return payload

    # ---- Full mode: forward all rows, copy corpus, rewrite one column, manifest.
    if out_dir is None:
        out_dir = corpus_dir.parent / f"{corpus_dir.name}_reanalyzed_{_checkpoint_tag(reanalyzer_meta)}"
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"output dir already exists (refusing to overwrite): {out_dir}")

    src_hashes = hash_corpus_dats(corpus_dir)
    src_meta_hash = sha256_file(corpus_dir / "corpus_meta.json")

    all_idx = np.arange(n, dtype=np.int64)
    fwd = batch_forward(
        policy, corpus, all_idx, batch_size=batch_size, want_q=want_q,
        legal_width=legal_width, progress_every=progress_every,
        value_materialization=root_value_provenance,
    )

    shutil.copytree(corpus_dir, out_dir)

    if spec["kind"] == "per_action":
        rewrite = rewrite_per_action_column(
            corpus, out_dir, v_component, fwd["q_values"], legal_width=legal_width
        )
    else:
        rewrite = rewrite_per_state_column(
            corpus,
            out_dir,
            v_component,
            fwd["value"],
            column_provenance=root_value_provenance,
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
    # A newly materialised column (root_value.dat) is expected to be absent in src.
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
        "forward_output": (
            root_value_provenance["forward_output"]
            if root_value_provenance is not None
            else spec["forward_output"]
        ),
        "q_head_provenance": verified_q_provenance,
        "root_value_materialization": root_value_provenance,
        "mask_hidden_info": bool(effective_mask),
        "device": device,
        "batch_size": batch_size,
        "rows_total": rewrite["rows_total"],
        "entries_rewritten": rewrite["entries_rewritten"],
        "meta_changed": rewrite["meta_changed"],
        "integrity": integrity,
        "stats": stats,
        "perspective_note": (
            "Fresh value/q come from the same policy class whose archived estimate "
            "they replace (root-to-move perspective), so no sign flip is applied. "
            "target_scores <- q_values head; root_value follows the persisted "
            "search readout/scale/squash/final-clip materialization contract."
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
    parser.add_argument("--corpus", required=True, type=Path, help="memmap corpus dir (build_memmap_corpus.py output)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output corpus dir (default: <corpus>_reanalyzed_<ckpt-tag>); must not exist",
    )
    parser.add_argument(
        "--v-component",
        default=DEFAULT_V_COMPONENT,
        choices=sorted(V_COMPONENTS),
        help="which value-component column to rewrite (default: root_value, a fresh "
        "per-state V(s) column from the trained value head). target_scores and "
        "afterstate_target require --q-head-provenance.",
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
        "--value-readout",
        choices=("scalar", "categorical"),
        default="scalar",
        help="root_value head to materialize. Must match the readout used by search; "
        "categorical fails closed if the forward emits no value_categorical output.",
    )
    parser.add_argument(
        "--value-squash",
        choices=("tanh", "clip"),
        default="tanh",
        help="search value squash. Scalar applies it after --value-scale; categorical "
        "records but bypasses it exactly like search, then final-clips to [-1,1].",
    )
    parser.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="positive finite search value scale applied before squash/final clipping",
    )
    parser.add_argument(
        "--reanalyzer-net",
        default="checkpoint",
        choices=("checkpoint", "ema"),
        help="reanalyzer net: a single checkpoint (--checkpoint), or an EMA/lagged "
        "average of several (--ema-checkpoints). EMA is the R8 fallback when "
        "generation-vs-champion drift telemetry is ambiguous.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint for --reanalyzer-net checkpoint")
    parser.add_argument(
        "--ema-checkpoints",
        nargs="+",
        type=Path,
        default=None,
        help="checkpoints (CHRONOLOGICAL: oldest->newest) for --reanalyzer-net ema",
    )
    parser.add_argument("--ema-decay", type=float, default=0.75, help="EMA decay for --reanalyzer-net ema")
    parser.add_argument("--device", default="cpu", help="torch device for the forward pass (cpu | cuda | cuda:0)")
    parser.add_argument("--batch-size", type=int, default=4096, help="forward-pass batch size")
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
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --sample row selection")
    parser.add_argument("--progress-every", type=int, default=0, help="log a progress line every N forward batches")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    work_dir = (args.out or (args.corpus.parent / f"{args.corpus.name}_reanalyzed")).parent
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
        value_readout=args.value_readout,
        value_squash=args.value_squash,
        value_scale=args.value_scale,
    )


if __name__ == "__main__":
    main()
