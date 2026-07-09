import json
import pytest

from catan_zero.data.colonist import (
    anonymize_history_game,
    extract_replay_urls,
    history_game_to_replay_export_request,
    replay_slug_from_url_or_slug,
    replay_to_training_frames,
    stable_hash,
)
from tools.fetch_colonist_replays_from_manifest import (
    _reject_unsafe_url_template,
    _render_export_url,
)


def test_anonymize_history_game_removes_direct_identity() -> None:
    game = {
        "id": "123",
        "setting": {"privateGame": False, "victoryPointsToWin": 10},
        "finished": True,
        "turnCount": 72,
        "duration": "12345",
        "hasReplay": True,
        "players": [
            {
                "userId": "42",
                "username": "Alice",
                "rank": 1,
                "points": 10,
                "finished": True,
                "isHuman": True,
                "playerColor": 1,
                "playOrder": 2,
            }
        ],
    }

    record = anonymize_history_game(game, salt="test")

    serialized = json.dumps(record)
    assert "Alice" not in serialized
    assert '"42"' not in serialized
    assert record["players"][0]["player_hash"] == stable_hash("42", salt="test")
    assert record["game_id"] == "123"


def test_replay_export_to_training_frames_redacts_players_and_events() -> None:
    replay = {
        "databaseGameId": 999,
        "gameSettings": {"privateGame": True},
        "players": [{"userId": "7", "username": "Bob", "playerColor": 2}],
        "eventHistory": {
            "events": [
                {"type": "roll", "userId": "7", "username": "Bob", "roll": 8},
                {"type": "build", "payload": {"countryCode": "US"}},
            ]
        },
    }

    frames = replay_to_training_frames(replay, source_path="sample.json", salt="test")
    serialized = json.dumps(frames)

    assert len(frames) == 2
    assert "Bob" not in serialized
    assert '"7"' not in serialized
    assert "US" not in serialized
    assert frames[0]["game"]["database_game_id"] == "999"


def test_replay_slug_from_url_or_slug() -> None:
    assert replay_slug_from_url_or_slug("abc123") == "abc123"
    assert replay_slug_from_url_or_slug("https://colonist.io/replay/abc123?action=57") == "abc123"
    assert replay_slug_from_url_or_slug("https://colonist.io/replay?replayUrlSlug=def456") == "def456"


def test_extract_replay_urls() -> None:
    text = """
    Review https://colonist.io/replay/abc123?action=57.
    Another https://colonist.io/replay?gameId=211223932&playerColor=1
    """
    urls = extract_replay_urls(text)
    assert "https://colonist.io/replay/abc123?action=57" in urls
    assert "https://colonist.io/replay?gameId=211223932&playerColor=1" in urls


def test_history_game_to_replay_export_request() -> None:
    game = {
        "game_id": "123",
        "has_replay": True,
        "setting": {"privateGame": True, "victoryPointsToWin": 10},
        "players": [
            {"player_color": 2},
            {"player_color": 1},
            {"player_color": None},
        ],
    }

    request = history_game_to_replay_export_request(game)

    assert request == {
        "source": "colonist_profile_history_replay_manifest",
        "game_id": "123",
        "private_game": True,
        "has_replay": True,
        "player_colors": [1, 2],
        "player_count": 2,
        "setting": {"privateGame": True, "victoryPointsToWin": 10},
    }


def test_authorized_export_url_can_use_player_color() -> None:
    request = {"game_id": "123", "player_colors": [3, 1]}

    url = _render_export_url(
        "https://admin.example/replay-export/{game_id}?playerColor={player_color}",
        request,
    )

    assert url == "https://admin.example/replay-export/123?playerColor=1"


def test_authorized_export_rejects_shareable_link_endpoint() -> None:
    with pytest.raises(ValueError, match="shareable-link"):
        _reject_unsafe_url_template(
            "https://colonist.io/api/replay/shareable-link?gameId={game_id}&playerColor={player_color}"
        )
