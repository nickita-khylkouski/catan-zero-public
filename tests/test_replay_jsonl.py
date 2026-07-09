import json

import pytest

from catan_zero.rl import REPLAY_JSONL_VERSION, dump_replay_jsonl, load_replay_jsonl


def test_replay_jsonl_round_trip_and_normalizes_tuples(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    frames = (
        {
            "frame_id": 1,
            "event": {"turn_key": (0, 0), "payload": {"value": ("WOOD", "ORE")}},
            "observations": {"BLUE": {"legal_actions": (1, 2)}},
            "rewards": {"BLUE": 0.0},
            "terminated": False,
            "truncated": False,
        },
    )

    count = dump_replay_jsonl(frames, path, metadata={"seed": 123})
    loaded = load_replay_jsonl(path)

    assert count == 1
    assert loaded[0]["event"]["turn_key"] == [0, 0]
    assert loaded[0]["event"]["payload"]["value"] == ["WOOD", "ORE"]
    assert loaded[0]["observations"]["BLUE"]["legal_actions"] == [1, 2]


def test_replay_jsonl_rejects_unknown_version(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"version": "old", "frame": {}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported replay version"):
        load_replay_jsonl(path)


def test_replay_jsonl_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text(
        "\n"
        + json.dumps({"version": REPLAY_JSONL_VERSION, "frame": {"frame_id": 1}})
        + "\n\n",
        encoding="utf-8",
    )

    assert load_replay_jsonl(path) == ({"frame_id": 1},)
