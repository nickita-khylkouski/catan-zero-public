from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch full Colonist replay JSONs from a first-party/internal export "
            "endpoint using a replay-export manifest. This intentionally does "
            "not call /api/replay/shareable-link."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--url-template",
        required=True,
        help=(
            "Internal replay export URL with {game_id} and optional {player_color}. "
            "Example: https://admin.example/replays/{game_id}.json"
        ),
    )
    parser.add_argument(
        "--auth-env",
        default="COLONIST_REPLAY_EXPORT_AUTH",
        help="Environment variable containing an Authorization header value.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/render export URLs without fetching replay JSON.",
    )
    args = parser.parse_args()

    _reject_unsafe_url_template(args.url_template)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    auth_value = os.environ.get(args.auth_env)

    fetched = 0
    skipped_existing = 0
    failed: list[dict[str, str]] = []
    for index, request in enumerate(_iter_manifest(args.manifest), start=1):
        if args.limit is not None and fetched >= args.limit:
            break
        game_id = str(request.get("game_id") or "")
        if not game_id:
            continue
        output_path = output_dir / f"{_safe_label(game_id)}.json"
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        url = _render_export_url(args.url_template, request)
        if args.dry_run:
            print(
                json.dumps(
                    {"event": "planned", "game_id": game_id, "url": url},
                    sort_keys=True,
                ),
                flush=True,
            )
            fetched += 1
            continue
        try:
            payload = _fetch_json(url, auth_value=auth_value, timeout_seconds=args.timeout_seconds)
            _validate_replay_payload(payload, game_id=game_id)
            output_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
            fetched += 1
            print(json.dumps({"event": "fetched", "game_id": game_id, "count": fetched}, sort_keys=True), flush=True)
        except Exception as exc:
            failed.append({"game_id": game_id, "url": url, "error": repr(exc)})
            print(json.dumps({"event": "failed", "game_id": game_id, "error": repr(exc)}, sort_keys=True), flush=True)

        if args.delay_seconds > 0:
            sleep(args.delay_seconds)

    (output_dir / "fetch_report.json").write_text(
        json.dumps(
            {
                "fetched": fetched,
                "skipped_existing": skipped_existing,
                "failed": failed[-500:],
                "manifest": args.manifest,
                "url_template": args.url_template,
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fetched": fetched,
                "failed": len(failed),
                "skipped_existing": skipped_existing,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


def _iter_manifest(path: str) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            records.append(record)
    return tuple(records)


def _fetch_json(url: str, *, auth_value: str | None, timeout_seconds: float) -> Any:
    headers = {"accept": "application/json", "user-agent": "CatanZeroReplayExport/0.1"}
    if auth_value:
        headers["authorization"] = auth_value
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _reject_unsafe_url_template(url_template: str) -> None:
    if "/api/replay/shareable-link" in url_template:
        raise ValueError(
            "Refusing to automate /api/replay/shareable-link. Use a first-party "
            "admin/export endpoint that authorizes replay access directly."
        )


def _render_export_url(url_template: str, request: dict[str, Any]) -> str:
    game_id = str(request.get("game_id") or "")
    if not game_id:
        raise ValueError("manifest request missing game_id")
    player_colors = request.get("player_colors") or []
    if "{player_color}" in url_template:
        if not player_colors:
            raise ValueError(f"manifest request {game_id} has no player colors")
        player_color = str(sorted(int(color) for color in player_colors)[0])
    else:
        player_color = ""
    return url_template.format(
        game_id=quote(game_id, safe=""),
        player_color=quote(player_color, safe=""),
    )


def _validate_replay_payload(payload: Any, *, game_id: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError("replay payload is not an object")
    replay = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    replay_id = str(replay.get("databaseGameId") or replay.get("gameId") or "")
    if replay_id and replay_id != game_id:
        raise ValueError(f"replay id mismatch: expected {game_id}, got {replay_id}")
    event_history = replay.get("eventHistory")
    if isinstance(event_history, dict):
        events = event_history.get("events")
    else:
        events = event_history
    if not isinstance(events, list) or not events:
        raise ValueError("replay payload has no eventHistory events")


def _safe_label(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


if __name__ == "__main__":
    main()
