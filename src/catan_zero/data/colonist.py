from __future__ import annotations

import hashlib
import html
import json
import re
import base64
from http.client import RemoteDisconnected
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen


COLONIST_BASE_URL = "https://colonist.io"
DEFAULT_USER_AGENT = "CatanZeroResearch/0.1"


@dataclass(frozen=True, slots=True)
class HistoryCrawlStats:
    profiles_requested: int
    profiles_saved: int
    games_seen: int
    games_written: int
    unique_players_seen: int
    queue_remaining: int

    def to_dict(self) -> dict[str, int]:
        return {
            "profiles_requested": self.profiles_requested,
            "profiles_saved": self.profiles_saved,
            "games_seen": self.games_seen,
            "games_written": self.games_written,
            "unique_players_seen": self.unique_players_seen,
            "queue_remaining": self.queue_remaining,
        }


def fetch_json(url: str, *, user_agent: str = DEFAULT_USER_AGENT, timeout_seconds: float = 20) -> Any:
    request = Request(url, headers={"accept": "application/json", "user-agent": user_agent})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str, *, user_agent: str = DEFAULT_USER_AGENT, timeout_seconds: float = 20) -> str:
    request = Request(url, headers={"user-agent": user_agent})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_live_usernames(*, base_url: str = COLONIST_BASE_URL) -> tuple[str, ...]:
    payload = fetch_json(f"{base_url}/api/game-list.json")
    usernames: list[str] = []
    for game in payload.get("games", []):
        for player in game.get("players", []):
            username = player.get("username")
            if username and not player.get("isBot"):
                usernames.append(str(username))
    return tuple(dict.fromkeys(usernames))


def crawl_public_profile_histories(
    *,
    output_dir: str | Path,
    seeds: Iterable[str],
    max_profiles: int,
    delay_seconds: float = 0.05,
    include_private_games: bool = False,
    base_url: str = COLONIST_BASE_URL,
    salt: str = "catan-zero",
    timeout_seconds: float = 12,
    progress_every: int = 25,
) -> HistoryCrawlStats:
    """Crawl public profile-history metadata and write raw plus anonymized data.

    This intentionally does not call replay share-link endpoints. Full replay
    JSON should come from an authorized internal export and then go through the
    replay-export importer.
    """

    root = Path(output_dir)
    raw_dir = root / "raw" / "profile_history"
    raw_dir.mkdir(parents=True, exist_ok=True)
    games_path = root / "public_games.jsonl"
    players_path = root / "players.jsonl"
    index_path = root / "username_index.json"
    stats_path = root / "crawl_stats.json"

    queue: deque[str] = deque(_clean_username(seed) for seed in seeds if _clean_username(seed))
    queued = set(queue)
    visited: set[str] = set()
    player_records: dict[str, dict[str, Any]] = {}
    game_ids_written: set[str] = set()
    games_seen = 0
    games_written = 0
    profiles_requested = 0
    profiles_saved = 0
    username_index: dict[str, str] = {}

    with games_path.open("w", encoding="utf-8") as games_handle:
        while queue and profiles_requested < max_profiles:
            username = queue.popleft()
            if username in visited:
                continue
            visited.add(username)
            profiles_requested += 1
            url = f"{base_url}/api/profile/{quote(username, safe='')}/history"
            try:
                history = fetch_json(url, timeout_seconds=timeout_seconds)
            except (
                HTTPError,
                URLError,
                TimeoutError,
                RemoteDisconnected,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                _append_jsonl(
                    root / "errors.jsonl",
                    {"username_hash": stable_hash(username, salt=salt), "url": url, "error": repr(exc)},
                )
                continue

            username_hash = stable_hash(username, salt=salt)
            username_index[username_hash] = username
            (raw_dir / f"{username_hash}.json").write_text(
                json.dumps(history, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            profiles_saved += 1

            for game in history.get("gameDatas", []):
                if not isinstance(game, dict):
                    continue
                games_seen += 1
                setting = dict(game.get("setting") or {})
                is_private = bool(setting.get("privateGame"))
                if is_private and not include_private_games:
                    continue
                game_id = str(game.get("id") or "")
                if not game_id or game_id in game_ids_written:
                    continue
                game_ids_written.add(game_id)
                record = anonymize_history_game(game, salt=salt)
                games_handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                games_written += 1

                for player in game.get("players", []):
                    if not isinstance(player, dict):
                        continue
                    player_username = _clean_username(player.get("username"))
                    if player_username and player_username.lower() != "bot":
                        player_hash = stable_hash(player_username, salt=salt)
                        player_records[player_hash] = anonymize_history_player(player, salt=salt)
                        if (
                            player_username not in visited
                            and player_username not in queued
                            and len(visited) + len(queue) < max_profiles * 4
                        ):
                            queue.append(player_username)
                            queued.add(player_username)

            if delay_seconds > 0:
                sleep(delay_seconds)
            if progress_every > 0 and profiles_requested % progress_every == 0:
                _write_history_progress(
                    root / "crawl_progress.json",
                    profiles_requested=profiles_requested,
                    profiles_saved=profiles_saved,
                    games_seen=games_seen,
                    games_written=games_written,
                    unique_players_seen=len(player_records),
                    queue_remaining=len(queue),
                )

    with players_path.open("w", encoding="utf-8") as handle:
        for record in sorted(player_records.values(), key=lambda item: item["player_hash"]):
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    index_path.write_text(json.dumps(username_index, sort_keys=True, indent=2), encoding="utf-8")
    stats = HistoryCrawlStats(
        profiles_requested=profiles_requested,
        profiles_saved=profiles_saved,
        games_seen=games_seen,
        games_written=games_written,
        unique_players_seen=len(player_records),
        queue_remaining=len(queue),
    )
    stats_path.write_text(json.dumps(stats.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
    _write_history_progress(
        root / "crawl_progress.json",
        profiles_requested=profiles_requested,
        profiles_saved=profiles_saved,
        games_seen=games_seen,
        games_written=games_written,
        unique_players_seen=len(player_records),
        queue_remaining=len(queue),
    )
    return stats


def _write_history_progress(path: Path, **stats: int) -> None:
    path.write_text(json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8")


def anonymize_history_game(game: dict[str, Any], *, salt: str) -> dict[str, Any]:
    setting = dict(game.get("setting") or {})
    return {
        "source": "colonist_profile_history",
        "game_id": str(game.get("id") or ""),
        "setting": setting,
        "finished": bool(game.get("finished")),
        "turn_count": _int_or_none(game.get("turnCount")),
        "start_time": str(game.get("startTime") or ""),
        "duration_ms": _int_or_none(game.get("duration")),
        "has_replay": bool(game.get("hasReplay")),
        "players": [
            anonymize_history_player(player, salt=salt)
            for player in game.get("players", [])
            if isinstance(player, dict)
        ],
    }


def anonymize_history_player(player: dict[str, Any], *, salt: str) -> dict[str, Any]:
    username = str(player.get("username") or "")
    raw_user_id = player.get("userId")
    identity = str(raw_user_id or username)
    is_human = bool(player.get("isHuman", username.lower() != "bot"))
    return {
        "player_hash": stable_hash(identity, salt=salt) if is_human else "bot",
        "username_hash": stable_hash(username, salt=salt) if username and is_human else "bot",
        "is_human": is_human,
        "rank": _int_or_none(player.get("rank")),
        "points": _int_or_none(player.get("points")),
        "finished": bool(player.get("finished")),
        "quit_with_penalty": bool(player.get("quitWithPenalty")),
        "player_color": _int_or_none(player.get("playerColor")),
        "play_order": _int_or_none(player.get("playOrder")),
        "device_type_id": _int_or_none(player.get("deviceTypeId")),
    }


def history_game_to_replay_export_request(game: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a crawled profile-history game row into a full-replay export request.

    The profile-history API is metadata only. This record is meant for a
    first-party/internal replay export job that can look up the full replay JSON
    by game id, optionally using player colors when the exporter is perspective
    aware.
    """

    game_id = str(game.get("game_id") or game.get("id") or "")
    if not game_id:
        return None
    if not bool(game.get("has_replay")):
        return None
    setting = dict(game.get("setting") or {})
    player_colors = sorted(
        {
            color
            for player in game.get("players", [])
            if isinstance(player, dict)
            for color in [_int_or_none(player.get("player_color") or player.get("playerColor"))]
            if color is not None
        }
    )
    return {
        "source": "colonist_profile_history_replay_manifest",
        "game_id": game_id,
        "private_game": bool(setting.get("privateGame")),
        "has_replay": True,
        "player_colors": player_colors,
        "player_count": len(player_colors),
        "setting": setting,
    }


def write_replay_export_manifest(
    *,
    history_games_jsonl: str | Path,
    output_path: str | Path,
    include_private_games: bool = True,
) -> int:
    seen: set[str] = set()
    count = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(history_games_jsonl).open("r", encoding="utf-8") as source, output.open(
        "w",
        encoding="utf-8",
    ) as sink:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            request = history_game_to_replay_export_request(json.loads(stripped))
            if request is None:
                continue
            if request["private_game"] and not include_private_games:
                continue
            game_id = str(request["game_id"])
            if game_id in seen:
                continue
            seen.add(game_id)
            sink.write(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def iter_replay_export_files(path: str | Path) -> tuple[Path, ...]:
    root = Path(path)
    if root.is_file():
        return (root,)
    return tuple(sorted(item for item in root.rglob("*.json") if item.is_file()))


def replay_slug_from_url_or_slug(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if "/" not in value and "?" not in value:
        return value
    parsed = urlparse(value)
    if parsed.path.startswith("/replay/"):
        slug = parsed.path.removeprefix("/replay/").strip("/")
        return slug or None
    return parse_qs(parsed.query).get("replayUrlSlug", [None])[0]


def extract_replay_urls(text: str) -> tuple[str, ...]:
    decoded = html.unescape(text)
    candidates = re.findall(
        r"https?://(?:www\.)?colonist\.io/replay(?:/[A-Za-z0-9_-]+(?:\?[^\s\"'<>)]*)?|\?[^\s\"'<>)]*)",
        decoded,
        flags=re.IGNORECASE,
    )
    return tuple(dict.fromkeys(candidate.rstrip(".,]") for candidate in candidates))


def extract_http_urls(text: str) -> tuple[str, ...]:
    decoded = html.unescape(text)
    urls = re.findall(r"https?://[^\s\"'<>)]{5,}", decoded)
    cleaned: list[str] = []
    for url in urls:
        url = _normalize_search_url(unquote(url).rstrip(".,]"))
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            cleaned.append(url)
    return tuple(dict.fromkeys(cleaned))


def search_result_urls(query: str, *, timeout_seconds: float = 8) -> tuple[str, ...]:
    encoded = quote(query)
    urls: list[str] = []
    for search_url in (
        f"https://www.bing.com/search?q={encoded}",
        f"https://duckduckgo.com/html/?q={encoded}",
    ):
        try:
            html_text = fetch_text(
                search_url,
                user_agent="Mozilla/5.0",
                timeout_seconds=timeout_seconds,
            )
        except (HTTPError, URLError, TimeoutError, RemoteDisconnected, OSError):
            continue
        urls.extend(extract_http_urls(html_text))
    filtered = [
        url
        for url in urls
        if "colonist.io/replay" in url
        or any(domain in urlparse(url).netloc for domain in ("reddit.com", "youtube.com", "catanbot.com"))
    ]
    return tuple(dict.fromkeys(filtered))


def _normalize_search_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if parsed.netloc.endswith("duckduckgo.com") and "uddg" in query:
        return query["uddg"][0]
    if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/") and "u" in query:
        decoded = _decode_bing_u(query["u"][0])
        if decoded:
            return decoded
    return url


def _decode_bing_u(value: str) -> str | None:
    if value.startswith("a1"):
        value = value[2:]
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode(value + padding).decode("utf-8")
    except Exception:
        return None
    parsed = urlparse(decoded)
    if parsed.scheme in {"http", "https"}:
        return decoded
    return None


def download_public_replay_slug(
    slug: str,
    *,
    output_dir: str | Path,
    base_url: str = COLONIST_BASE_URL,
) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_slug = safe_label(slug)
    output_path = output_root / f"{safe_slug}.json"
    payload = fetch_json(
        f"{base_url}/api/replay/data-from-slug?replayUrlSlug={quote(slug, safe='')}"
    )
    output_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return output_path


def load_json_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def replay_to_training_frames(
    replay: dict[str, Any],
    *,
    source_path: str,
    salt: str = "catan-zero",
) -> tuple[dict[str, Any], ...]:
    if isinstance(replay.get("data"), dict):
        replay = replay["data"]
    events = _find_event_history(replay)
    if not isinstance(events, list):
        raise ValueError(f"replay has no eventHistory list: {source_path}")
    metadata = replay_metadata(replay, source_path=source_path, salt=salt)
    frames: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            event = {"value": event}
        frames.append(
            {
                "source": "colonist_replay_export",
                "source_path": source_path,
                "frame_id": index,
                "game": metadata,
                "event": redact_replay_event(event, salt=salt),
            }
        )
    return tuple(frames)


def replay_metadata(replay: dict[str, Any], *, source_path: str, salt: str) -> dict[str, Any]:
    if isinstance(replay.get("data"), dict):
        replay = replay["data"]
    players = replay.get("players") or replay.get("playerUserStates") or replay.get("userStates") or []
    if isinstance(players, dict):
        players = list(players.values())
    return {
        "source_path": source_path,
        "database_game_id": str(replay.get("databaseGameId") or replay.get("gameId") or ""),
        "settings": replay.get("gameSettings") or replay.get("setting") or {},
        "players": [
            _redact_replay_player(player, salt=salt)
            for player in players
            if isinstance(player, dict)
        ],
    }


def redact_replay_event(event: dict[str, Any], *, salt: str) -> dict[str, Any]:
    return _redact_value(event, salt=salt)


def stable_hash(value: str, *, salt: str = "catan-zero") -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:20]


def _find_event_history(value: Any) -> Any:
    if isinstance(value, dict):
        event_history = value.get("eventHistory")
        if isinstance(event_history, dict) and isinstance(event_history.get("events"), list):
            return event_history["events"]
        for key in ("eventHistory", "events", "history"):
            found = value.get(key)
            if isinstance(found, list):
                return found
        for item in value.values():
            found = _find_event_history(item)
            if isinstance(found, list):
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_event_history(item)
            if isinstance(found, list):
                return found
    return None


def _redact_replay_player(player: dict[str, Any], *, salt: str) -> dict[str, Any]:
    identity = str(player.get("userId") or player.get("username") or player.get("id") or "")
    return {
        "player_hash": stable_hash(identity, salt=salt) if identity else "",
        "username_hash": stable_hash(str(player.get("username")), salt=salt)
        if player.get("username")
        else "",
        "player_color": _int_or_none(player.get("playerColor") or player.get("color")),
        "membership": _int_or_none(player.get("membership")),
    }


def _redact_value(value: Any, *, salt: str) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in {"username", "displayName", "countryCode"}:
                result[f"{normalized_key}Hash"] = stable_hash(str(item), salt=salt) if item else ""
            elif normalized_key in {"userId", "profileUserId"}:
                result[f"{normalized_key}Hash"] = stable_hash(str(item), salt=salt) if item else ""
            else:
                result[normalized_key] = _redact_value(item, salt=salt)
        return result
    if isinstance(value, list):
        return [_redact_value(item, salt=salt) for item in value]
    return value


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _clean_username(value: Any) -> str:
    username = str(value or "").strip()
    return username if username and username.lower() != "bot" else ""


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"
