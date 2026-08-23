# Misty engine protocol (v1)

The wire contract between a Fog-of-War chess engine and a game server. The
engine never sees the canonical game state — every message is built from
what its side could legally observe under fog. This document is vendored
into the engine repo so it is self-describing; the reference server-side
implementation lives in the Mistboard platform (TypeScript,
`packages/game/src/engine-protocol.ts`, with the redaction boundary
enforced and tested in its server's `engine-protocol/build.ts`).

The Python mirror is `src/fow_chess/engine_protocol.py`; its wire shape is
pinned by `tests/test_engine_protocol_roundtrip.py`.

## Transport

Line-delimited JSON over stdio (long-lived worker) or a single
request/response exchange (one-shot runner). `scripts/live_move_worker.py`
is the worker loop; the host wraps each turn request in an envelope with a
request id and per-move budget/deadline hints.

## The redaction guarantee

The engine receives ONLY what the side-to-move can observe:

- its own pieces and the squares its pieces see (visibility mask),
- opponent pieces standing on visible squares,
- its own capture square / the landing square of an opponent capture that
  removed one of its pieces,
- its OWN clock only,
- a derived `engineSeed` (never the room master seed).

It never receives the opponent's hidden pieces or moves, the opponent's
clock, raw server events, or canonical game state. Post-game analysis with
full information is a *separate* request type built only from finished
games — never a relaxation flag on the live request.

## Types (v1)

`protocolVersion` is the string `"1"` on every message.

### Move

```json
{ "from": "e2", "to": "e4", "promotion": "queen" }
```

Square NAMES (not indices); `promotion` is omitted unless the move
promotes (`"queen" | "rook" | "bishop" | "knight"`).

### EngineObservation

One per ply, from the engine's perspective:

| field | meaning |
|---|---|
| `ply` | ply index this observation belongs to |
| `kind` | `"initial"` \| `"own_move"` \| `"opp_move"` |
| `own_move` | the engine's own move (`kind == "own_move"`), else null |
| `visibility_mask` | `"0x…"` 64-bit hex of visible squares |
| `visible_pieces` | `[square, {type, color}]` pairs on visible squares |
| `own_capture_square` | square where the engine just captured, or null |
| `opp_capture_landing_square` | square where an opponent capture landed, or null |
| `game_over` | `{winner, reason}` when the game ended, else null |

### EngineTurnRequest (server → engine)

| field | meaning |
|---|---|
| `gameId`, `engineId`, `sessionId` | correlation ids (sessionId is opaque to the engine) |
| `gameSpecId` | variant id; `"dark-chess"` when omitted |
| `color` | the engine's seat |
| `ply` | current ply |
| `engineSeed` | derived per-game seed for deterministic play |
| `clock` | `{remaining_ms, increment_ms}` — the engine's OWN clock |
| `legalMoves` | the full legal move list (the engine must answer from it) |
| `observationTranscript` | full per-ply history (cold start) |
| `latestObservationDelta` | just the newest observation (warm session) |

Exactly one of `observationTranscript` / `latestObservationDelta` is set:
transcript on a cold start, delta once a session is warm.

### EngineTurnResponse (engine → server)

`{protocolVersion, gameId, sessionId, move, diagnostics?}` — one move,
which MUST come from `legalMoves` (the server validates; an illegal or
missing move triggers the server's fallback policy). `diagnostics` is an
opaque object the server may log/store; Misty uses it for search telemetry
(belief size, iterations, move ranking).

### EngineObservationPush / EngineObservationAck (optional)

Server → engine immediately after the engine's own move is applied, before
the opponent replies — the "observe the instant you move" step, enabling
belief advance + pondering on the opponent's clock. Opt-in and additive:
an engine that ignores it still plays correctly, because the same
`own_move` observation arrives in its next turn request; an engine
handling both must dedupe by `ply`. The push expects only an ack
(`{..., "received": true}`); no move is requested.

## Versioning

Breaking wire changes bump `protocolVersion`. Additive optional fields may
appear within a version; engines must ignore unknown fields.
