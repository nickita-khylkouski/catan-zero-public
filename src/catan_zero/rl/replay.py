from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

REPLAY_JSONL_VERSION = "colonist-replay-v1"


def dump_replay_jsonl(
    frames: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> int:
    """Write replay frames as JSONL and return the number of frames written."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            count += 1
            record = {
                "version": REPLAY_JSONL_VERSION,
                "metadata": _jsonable(metadata or {}),
                "frame": _jsonable(frame),
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return count


def load_replay_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read replay frames written by `dump_replay_jsonl`."""
    frames: list[dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            version = record.get("version")
            if version != REPLAY_JSONL_VERSION:
                raise ValueError(
                    f"unsupported replay version on line {line_number}: {version}"
                )
            frame = record.get("frame")
            if not isinstance(frame, dict):
                raise ValueError(f"missing replay frame on line {line_number}")
            frames.append(frame)
    return tuple(frames)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)
