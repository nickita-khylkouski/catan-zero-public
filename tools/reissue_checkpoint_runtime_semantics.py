#!/usr/bin/env python3
"""Reissue an entity checkpoint by adding only its forward-semantics stamp.

This is for distributed checkpoints written before the DDP/FSDP writer learned
to mirror ``EntityGraphPolicy.save``'s runtime binding.  It never reconstructs
the model: every existing payload value is copied and recursively compared
after the atomic re-save, and the receipt binds the unchanged model tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from catan_zero.rl.checkpoint_runtime_semantics import (
    ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
    current_entity_graph_forward_semantics,
)
import catan_zero.rl.entity_token_policy as entity_token_policy_module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _tensor_inventory_sha256(model: dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(model):
        value = model[name]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"model entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _assert_equal(before: Any, after: Any, *, path: str) -> None:
    import torch

    if isinstance(before, torch.Tensor):
        if not isinstance(after, torch.Tensor) or not torch.equal(before, after):
            raise RuntimeError(f"checkpoint value changed at {path}")
        return
    if isinstance(before, dict):
        if not isinstance(after, dict) or list(before) != list(after):
            raise RuntimeError(f"checkpoint mapping changed at {path}")
        for key in before:
            _assert_equal(before[key], after[key], path=f"{path}.{key}")
        return
    if isinstance(before, (list, tuple)):
        if not isinstance(after, type(before)) or len(before) != len(after):
            raise RuntimeError(f"checkpoint sequence changed at {path}")
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            _assert_equal(left, right, path=f"{path}[{index}]")
        return
    if type(before) is not type(after) or before != after:
        raise RuntimeError(f"checkpoint scalar changed at {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-checkpoint", type=Path, required=True)
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    import torch

    source = args.in_checkpoint.expanduser().resolve(strict=True)
    output = args.out_checkpoint.expanduser().resolve()
    receipt = args.receipt.expanduser().resolve()
    if output.exists() or receipt.exists():
        raise SystemExit("output checkpoint and receipt must be fresh")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("policy_type") != "entity_graph":
        raise SystemExit("source is not an entity_graph policy checkpoint")
    if ENTITY_GRAPH_FORWARD_SEMANTICS_KEY in payload:
        raise SystemExit("source checkpoint already has a forward-semantics stamp")
    model = payload.get("model")
    if not isinstance(model, dict) or not model:
        raise SystemExit("source checkpoint has no model state dictionary")

    identity = current_entity_graph_forward_semantics(
        Path(entity_token_policy_module.__file__).resolve(strict=True)
    )
    source_sha = _file_sha256(source)
    tensor_sha = _tensor_inventory_sha256(model)
    reissued = dict(payload)
    reissued[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] = identity

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(reissued, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    loaded = torch.load(output, map_location="cpu", weights_only=False)
    if set(loaded) != set(payload) | {ENTITY_GRAPH_FORWARD_SEMANTICS_KEY}:
        raise RuntimeError("reissued checkpoint changed its top-level key set")
    for key, value in payload.items():
        _assert_equal(value, loaded[key], path=key)
    if loaded[ENTITY_GRAPH_FORWARD_SEMANTICS_KEY] != identity:
        raise RuntimeError("reissued checkpoint has the wrong semantic identity")
    if _tensor_inventory_sha256(loaded["model"]) != tensor_sha:
        raise RuntimeError("reissued checkpoint model tensor inventory changed")

    report = {
        "schema_version": "entity-checkpoint-runtime-semantics-reissue-v1",
        "source_checkpoint": str(source),
        "source_sha256": source_sha,
        "reissued_checkpoint": str(output),
        "reissued_sha256": _file_sha256(output),
        "model_tensor_inventory_sha256": tensor_sha,
        "only_added_top_level_key": ENTITY_GRAPH_FORWARD_SEMANTICS_KEY,
        "forward_semantics": identity,
        "all_preexisting_values_identical": True,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    try:
        with temporary_receipt.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_receipt, receipt)
    finally:
        if temporary_receipt.exists():
            temporary_receipt.unlink()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
