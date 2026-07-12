from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import a1_policy_game_weight_probe as probe
from tools import train_bc


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus(root: Path, name: str) -> tuple[Path, Path]:
    corpus = root / name
    corpus.mkdir()
    for filename, payload in {
        "row_offsets.dat": np.asarray([0, 0], dtype="<i8").tobytes(),
        "game_seed.dat": np.asarray([1], dtype="<i8").tobytes(),
    }.items():
        (corpus / filename).write_bytes(payload)
    inventory = [{"filename": f, "size_bytes": (corpus/f).stat().st_size,
                  "sha256": _sha(corpus/f)} for f in ("game_seed.dat", "row_offsets.dat")]
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    meta = {"schema": "memmap_corpus_v1", "row_count": 1, "legal_width": 1,
            "flat_count": 0,
            "columns": {"game_seed": {"kind": "fixed", "dtype": "<i8", "inner_shape": []}},
            "payload_inventory_schema": "memmap-payload-inventory-v1",
            "payload_inventory": inventory,
            "payload_inventory_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
            "selected_game_seed_manifest": {"a1_contract_sha256": "sha256:" + "1"*64},
            "a1_post_wave_audit": {"contract_sha256": "sha256:" + "1"*64}}
    (corpus / "corpus_meta.json").write_text(json.dumps(meta))
    validation = root / f"{name}.validation.json"
    validation.write_text("{}\n")
    return corpus, validation


def _args(tmp_path: Path) -> list[str]:
    n256, v256 = _corpus(tmp_path, "n256")
    n128, v128 = _corpus(tmp_path, "n128")
    init = tmp_path / "init.pt"; init.write_bytes(b"init")
    return ["--lr", "1.2e-4", "--max-steps", "500",
            "--n256-corpus", str(n256), "--n256-validation", str(v256),
            "--n128-corpus", str(n128), "--n128-validation", str(v128),
            "--init-checkpoint", str(init), "--output-root", str(tmp_path/"out")]


def test_plan_is_matched_bounded_and_nonpromotable(tmp_path: Path) -> None:
    args = probe.build_parser().parse_args(_args(tmp_path))
    manifest, _ = probe.prepare(args)
    assert manifest["diagnostic_only"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["max_steps"] == 500
    equal = dict(manifest["arms"]["equal"]["recipe"])
    sqrt = dict(manifest["arms"]["sqrt"]["recipe"])
    assert equal.pop("per_game_policy_weight_mode") == "equal"
    assert sqrt.pop("per_game_policy_weight_mode") == "sqrt"
    assert equal == sqrt
    assert all("--nproc-per-node=8" in arm["command"] for arm in manifest["arms"].values())
    assert all("--max-steps" in arm["command"] for arm in manifest["arms"].values())


def test_descriptors_authorize_only_their_policy_mode(tmp_path: Path) -> None:
    manifest, _ = probe.prepare(probe.build_parser().parse_args(_args(tmp_path)))
    for mode, arm in manifest["arms"].items():
        verified = train_bc._preflight_memmap_composite_descriptor(arm["descriptor"])
        train_bc._validate_composite_learner_recipe_authorization(
            type("Args", (), arm["recipe"])(), verified
        )
        drift = dict(arm["recipe"])
        drift["per_game_policy_weight_mode"] = "sqrt" if mode == "equal" else "equal"
        with pytest.raises(SystemExit, match="authenticated diagnostic learner recipe"):
            train_bc._validate_composite_learner_recipe_authorization(
                type("Args", (), drift)(), verified
            )


def test_nonpositive_step_budget_is_refused(tmp_path: Path) -> None:
    argv = _args(tmp_path)
    argv[argv.index("--max-steps") + 1] = "0"
    with pytest.raises(SystemExit, match="must be positive"):
        probe.prepare(probe.build_parser().parse_args(argv))


def test_prepare_preserves_lexical_virtualenv_python(tmp_path: Path) -> None:
    lexical = tmp_path / "venv-python"
    lexical.symlink_to(Path(sys.executable))
    args = probe.build_parser().parse_args(
        [*_args(tmp_path), "--python", str(lexical)]
    )
    manifest, _ = probe.prepare(args)
    assert all(arm["command"][0] == str(lexical) for arm in manifest["arms"].values())
    assert str(lexical) != str(lexical.resolve())
