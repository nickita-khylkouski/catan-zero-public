# Executive Summary

# Unauthenticated Bulk Player and Game-History Data Exposure on Colonist.io

# Executive Summary

A publicly reachable set of Colonist.io API endpoints allows unauthenticated clients to enumerate player identities and retrieve large volumes of player game-history records. The exposed history records include internal numeric user IDs, usernames, ranks, points, finish/quit status, player colors, device type IDs, game settings, game duration, turn count, private-game flags, and replay availability flags. Separate public replay-share URLs also return full replay JSON for the shared replay perspective without authentication.

The strongest validated issue is High severity. The positive controls returned real application data, not only non-401 responses or application-level errors. In one bounded run, unauthenticated requests retrieved 491 profile histories, 43,075 game records, 40,119 unique game IDs, 35,228 human player references, and 29,724 replay-available game references before rate limiting. A separate bounded numeric user-ID test resolved 492 usernames from 507 completed unauthenticated requests in dense numeric ID ranges.

This report does not claim that arbitrary full replay JSON is retrievable by direct \`gameId\` without authentication. Unauthenticated direct replay-by-game-ID requests returned \`userNotFound\` or \`noUser\` errors during this run. This report also does not claim a payment or Premium-membership bypass; checkout and gift-membership probes did not validate a free entitlement grant.

# Finding 1 \- 1 Unauthenticated Bulk Access to Playe

# Finding 1: Unauthenticated Bulk Access to Player and Game-History Data

Severity: High

Affected host: \`https://colonist.io\`

Affected endpoints:

\- \`GET /api/profile/{username}/history\`  
\- \`GET /api/profile/{userId}/username\`  
\- \`GET /api/game-list.json\`  
\- \`GET /api/leaderboards/{mode}/\`  
\- \`GET /api/leaderboards-username-search/{mode}/?usernameSearchInput={query}\`  
\- \`GET /api/replay/data-from-slug?replayUrlSlug={slug}\` for already disclosed replay-share slugs

Required role/session: None. The confirmed requests were made without a Colonist login session or authorization header.

Confirmed affected count:

\- 491 profile histories returned HTTP 200 in the bounded bulk run.  
\- 43,075 game-history records were returned from those histories.  
\- 40,119 unique game IDs were observed.  
\- 35,228 human player references were observed in returned histories.  
\- 29,724 replay-available game references were observed.  
\- 4,569 returned game records were marked as private games.  
\- 492 numeric user IDs resolved to usernames in a bounded 507-request user-ID test.  
\- 4 public replay slugs returned full replay JSON samples.

Where the issue was found:

The public web client route map includes profile-history and replay endpoints. Profile history pages construct replay URLs from \`game.id\` and the profile player color. The same public APIs can be called directly without a browser login. Public game-list and leaderboard endpoints provide seed usernames, and profile-history responses disclose additional co-player usernames and internal numeric user IDs, enabling recursive enumeration.

Reproduction steps:

1\. Start without any Colonist.io login cookies or authorization header.  
2\. Request the public live-game list to collect seed usernames.  
3\. Request one seed user's history through \`/api/profile/{username}/history\`.  
4\. Observe that the response contains up to 100 completed games for that profile and includes co-player identities and internal numeric user IDs.  
5\. Feed returned co-player usernames into the same history endpoint to recurse.  
6\. Optionally resolve dense numeric user IDs through \`/api/profile/{userId}/username\` and use the returned usernames in the history endpoint.  
7\. For a publicly disclosed replay-share URL, request \`/api/replay/data-from-slug?replayUrlSlug={slug}\` and observe the full replay JSON envelope for the shared perspective.  
8\. As a negative control, request \`/api/replay/data-from-game-id?gameId={gameId}\&playerColor={color}\` without a login session and observe that direct game-ID replay access is rejected.

Example request: seed usernames from live games

GET /api/game-list.json HTTP/1.1  
Host: colonist.io  
Accept: application/json

Example response snippet: seed usernames

{  
  "rooms": \[  
    {  
      "gameId": "238542940",  
      "players": \[  
        { "username": "\[redacted-player-1\]" },  
        { "username": "\[redacted-player-2\]" }  
      \]  
    }  
  \]  
}

Example request: unauthenticated profile history

GET /api/profile/\[redacted-username\]/history HTTP/1.1  
Host: colonist.io  
Accept: application/json

Example response snippet: profile game-history record

{  
  "id": "228852378",  
  "setting": {  
    "id": "grain3051",  
    "gameType": 3,  
    "privateGame": true,  
    "modeSetting": 4,  
    "extensionSetting": 0,  
    "scenarioSetting": 3,  
    "mapSetting": 14,  
    "diceSetting": 1,  
    "victoryPointsToWin": 12,  
    "karmaActive": true,  
    "cardDiscardLimit": 7,  
    "maxPlayers": 4,  
    "gameSpeed": 4,  
    "hideBankCards": false,  
    "friendlyRobber": false  
  },  
  "finished": true,  
  "turnCount": 85,  
  "startTime": "1778553885149",  
  "duration": "5493357",  
  "players": \[  
    {  
      "userId": "103005879",  
      "username": "\[redacted-player-1\]",  
      "rank": 1,  
      "points": 12,  
      "finished": true,  
      "quitWithPenalty": false,  
      "isHuman": true,  
      "playerColor": 5,  
      "deviceTypeId": 1,  
      "playOrder": 1  
    },  
    {  
      "userId": "103032504",  
      "username": "\[redacted-player-2\]",  
      "rank": 3,  
      "points": 7,  
      "finished": true,  
      "quitWithPenalty": false,  
      "isHuman": true,  
      "playerColor": 1,  
      "deviceTypeId": 1,  
      "playOrder": 2  
    }  
  \],  
  "userStats": \[\],  
  "hasReplay": true  
}

Example request: unauthenticated numeric user-ID to username resolution

GET /api/profile/103005879/username HTTP/1.1  
Host: colonist.io  
Accept: application/json

Example response snippet: user-ID resolution

{  
  "username": "\[redacted-username\]"  
}

Example request: public replay-share JSON

GET /api/replay/data-from-slug?replayUrlSlug=\[redacted-public-slug\] HTTP/1.1  
Host: colonist.io  
Accept: application/json

Example response snippet: replay JSON envelope

{  
  "data": {  
    "databaseGameId": "227786050",  
    "playerPerspective": 8,  
    "playOrder": \[2, 8, 1, 5\],  
    "playerUserStates": \[  
      {  
        "userId": "\[redacted-user-id\]",  
        "username": "\[redacted-player\]",  
        "selectedColor": 8  
      }  
    \],  
    "gameSettings": {  
      "gameType": 5,  
      "privateGame": false,  
      "maxPlayers": 4,  
      "victoryPointsToWin": 10  
    },  
    "gameDetails": {  
      "isRanked": false,  
      "isDiscord": false  
    },  
    "eventHistory": {  
      "version": "\[redacted-version\]",  
      "startTime": "2026-05-07T13:07:34.091Z",  
      "initialState": { "...": "redacted" },  
      "events": \[  
        {  
          "input": { "...": "redacted" },  
          "stateChange": { "...": "redacted" }  
        }  
      \],  
      "eventsPerspectives": \[8\],  
      "endGameState": { "...": "redacted" },  
      "botUserNames": \[\]  
    }  
  }  
}

Example negative control: direct replay by game ID without login

GET /api/replay/data-from-game-id?gameId=227786050\&playerColor=8 HTTP/1.1  
Host: colonist.io  
Accept: application/json

Example negative-control response snippet

{  
  "error": {  
    "description": {  
      "key": "strings:popups.errors.general.userNotFound"  
    },  
    "httpStatusCode": 401  
  }  
}

Impact:

An unauthenticated client can collect player and game-history data at scale. The exposed data includes internal numeric user IDs, player usernames, player ranks and points within games, quit/finish status, player colors, device type IDs, game settings, timestamps, durations, private-game flags, and replay availability. Because each profile-history response contains multiple co-player identities and internal IDs, the dataset can be expanded recursively from public seed sources such as the live game list and leaderboards.

The numeric user-ID route further amplifies enumeration because dense numeric ranges resolved to usernames at a high rate in bounded testing. In the bounded numeric-ID test, 492 usernames resolved from 507 completed unauthenticated requests before rate limiting.

Public replay-share slugs return full replay JSON for the originally shared perspective, including the replay event stream, initial state, end-game state, game settings, player states, and database game ID. This was confirmed for four public slugs with event counts of 537, 472, 239, and 630\. The run did not validate unauthenticated arbitrary direct replay access by game ID.

Validation notes:

\- Required role/session: unauthenticated.  
\- Exact endpoint for the bulk history exposure: \`GET /api/profile/{username}/history\`.  
\- Exact endpoint for numeric ID resolution: \`GET /api/profile/{userId}/username\`.  
\- Exact endpoint for public replay-share JSON: \`GET /api/replay/data-from-slug?replayUrlSlug={slug}\`.  
\- Identifiers are disclosed and listable. Usernames were obtained from public game-list and leaderboard APIs, and additional usernames and internal numeric IDs were disclosed inside profile-history responses. Numeric user IDs were also successfully guessed in dense observed ranges.  
\- Confirmed affected count from bounded testing: 491 profile histories, 43,075 game-history records, 40,119 unique game IDs, 35,228 human player references, 29,724 replay-available game references, 4,569 private-game records, 492 resolved numeric user IDs, and 4 full public-share replay JSON payloads.  
\- Maximum affected population is unknown. The confirmed count is bounded by the test volume and observed rate limiting, not by an application-side authorization boundary.  
\- Pre-publish severity gate: Did the positive control return protected data or a confirmed mutation, or just a non-401 / 200-with-app-error response? The positive controls returned real player/game-history records, internal numeric user IDs, private-game metadata, and replay JSON envelopes. They were not merely non-401 responses or 200 responses with application-level errors. Direct arbitrary replay-by-game-ID was separately tested as a negative control and rejected unauthenticated, so that narrower claim is not included.

# Plain-English Summary

# Plain-English Summary

1\. Unauthenticated bulk player/game-history exposure \- Public APIs expose player histories \- 43,075 records exposed

# Public Company Context

# Public Company Context

\- Demi Yilmaz \- Co-Founder at Colonist.io  
\- Mohammed S. Yaseen \- Sr. Software Engineer at Colonist  
\- Gabriel Bernardi Fantin \- Operations at Colonist

Company website: https://colonist.io  
