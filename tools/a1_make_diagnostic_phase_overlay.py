#!/usr/bin/env python3
"""Bind a no-copy, diagnostic-only AUX phase allocation to one A1 corpus.

The descriptor is intentionally not a production composite.  It authenticates
the existing corpus bytes, its held-out game set, and an exact policy-AUX phase
measure without rewriting a row or changing the value distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

# Direct ``python tools/...py`` execution puts ``tools/`` rather than the
# repository root on sys.path.  Keep the CLI usable by the sealed executor.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import train_bc
from catan_zero.rl.memmap_corpus import MemmapCorpus


class OverlayError(RuntimeError):
    """The requested immutable diagnostic overlay is malformed or drifts."""


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _parse_phase_weights(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw in values:
        phase, separator, value = raw.partition("=")
        if not separator or not phase or phase in result:
            raise OverlayError(f"invalid or duplicate --phase-weight {raw!r}")
        try:
            weight = float(value)
        except ValueError as error:
            raise OverlayError(f"invalid phase weight {raw!r}") from error
        if not math.isfinite(weight) or weight <= 0.0:
            raise OverlayError(f"phase weight must be finite and positive: {raw!r}")
        result[phase] = weight
    if not result or not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise OverlayError("--phase-weight values must be nonempty and sum to 1")
    return result


def build_overlay(
    *,
    corpus_dir: Path,
    validation_manifest: Path,
    component_id: str,
    phase_weights: dict[str, float],
) -> dict[str, object]:
    corpus_dir = corpus_dir.resolve(strict=True)
    validation_manifest = validation_manifest.resolve(strict=True)
    meta_path = corpus_dir / "corpus_meta.json"
    if not meta_path.is_file() or not validation_manifest.is_file():
        raise OverlayError("corpus metadata or validation manifest is missing")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise OverlayError("corpus metadata is not an object")
    inventory_sha = meta.get("payload_inventory_sha256")
    if not isinstance(inventory_sha, str) or not inventory_sha.startswith("sha256:"):
        raise OverlayError("corpus metadata lacks an authenticated payload inventory")
    overrides = {
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "equal",
    }
    return {
        "schema_version": "memmap_composite_v2",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "components": [
            {
                "corpus_dir": str(corpus_dir),
                "corpus_meta_sha256": _sha256_file(meta_path),
                "payload_inventory_sha256": inventory_sha,
                "validation_manifest": str(validation_manifest),
                "validation_manifest_sha256": _sha256_file(validation_manifest),
                "component_id": component_id,
                "game_sampling_ratio": 1.0,
            }
        ],
        "learner_recipe_overrides": overrides,
        "learner_recipe_overrides_sha256": _canonical_sha256(overrides),
        "policy_kl_anchor_component_ids": [component_id],
        "policy_distillation_component_ids": [component_id],
        "value_training_component_ids": [component_id],
        "policy_aux_phase_sampling_weights": phase_weights,
    }


def _write_once(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise OverlayError(f"refusing descriptor drift at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)


def _canonical_validation_contract(
    *, corpus_dir: Path, source: Path, output: Path | None
) -> Path:
    """Accept an exact holdout or convert a trainer receipt into one once.

    Older Stage-C receipts contain the exact game seed set but include runtime
    fields and omit the held-out row count.  The descriptor contract requires a
    smaller immutable sidecar.  Derive it from the authenticated corpus rather
    than recomputing a nominal split.
    """

    source = source.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OverlayError("validation source is not an object")
    exact_fields = {
        "schema_version",
        "a1_contract_sha256",
        "validation_fraction",
        "validation_seed",
        "validation_max_samples",
        "validation_game_seed_ranges",
        "validation_game_seed_count",
        "validation_row_count",
        "validation_game_seed_set_sha256",
        "game_seeds",
    }
    if set(payload) == exact_fields:
        return source
    if output is None:
        raise OverlayError(
            "trainer validation receipt needs --validation-contract-output"
        )
    required = {
        "a1_contract_sha256",
        "validation_fraction",
        "validation_seed",
        "validation_max_samples",
        "game_seeds",
    }
    if not required.issubset(payload):
        raise OverlayError("validation source lacks an exact Stage-C seed receipt")
    raw_seeds = payload["game_seeds"]
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
    ):
        raise OverlayError("validation source game seeds are invalid")
    seeds = np.asarray(raw_seeds, dtype=np.int64)
    if not np.all(seeds[1:] > seeds[:-1]):
        raise OverlayError("validation source game seeds are not sorted and unique")
    corpus = MemmapCorpus(corpus_dir)
    game_seeds = np.asarray(corpus["game_seed"], dtype=np.int64)
    validation_rows = int(np.count_nonzero(np.isin(game_seeds, seeds)))
    if validation_rows <= 0:
        raise OverlayError("validation source selects no rows in this corpus")
    contract = {
        "schema_version": "train-validation-game-seeds-v1",
        "a1_contract_sha256": payload["a1_contract_sha256"],
        "validation_fraction": payload["validation_fraction"],
        "validation_seed": payload["validation_seed"],
        "validation_max_samples": payload["validation_max_samples"],
        "validation_game_seed_ranges": [],
        "validation_game_seed_count": int(seeds.size),
        "validation_row_count": validation_rows,
        "validation_game_seed_set_sha256": train_bc._game_seed_set_sha256(seeds),
        "game_seeds": seeds.tolist(),
    }
    _write_once(output.resolve(), contract)
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument(
        "--validation-contract-output",
        default="",
        help="required only when --validation-manifest is an older trainer receipt",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--component-id", default="stage_c")
    parser.add_argument("--phase-weight", action="append", default=[])
    args = parser.parse_args()
    phase_weights = _parse_phase_weights(list(args.phase_weight))
    validation_manifest = _canonical_validation_contract(
        corpus_dir=Path(args.corpus_dir),
        source=Path(args.validation_manifest),
        output=(
            None
            if not args.validation_contract_output
            else Path(args.validation_contract_output)
        ),
    )
    payload = build_overlay(
        corpus_dir=Path(args.corpus_dir),
        validation_manifest=validation_manifest,
        component_id=str(args.component_id),
        phase_weights=phase_weights,
    )
    output = Path(args.output).resolve()
    _write_once(output, payload)
    verified = train_bc._preflight_memmap_composite_descriptor(output)
    if verified["policy_aux_phase_sampling_weights"] != phase_weights:
        raise OverlayError("overlay phase-allocation authentication drift")
    print(json.dumps({
        "descriptor": str(output),
        "descriptor_fingerprint": verified["descriptor_fingerprint"],
        "component_ids": verified["component_ids"],
        "policy_aux_phase_sampling_weights": phase_weights,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
