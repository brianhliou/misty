"""Python mirror of `packages/game/src/engine-protocol.ts`.

This is the wire-format the engine speaks to the Mistboard server. The
TypeScript file is the canonical contract; this Python module mirrors it
so the engine side can parse + emit the same JSON shapes.

When the public TS spec changes, update this file to match. The wire shape
is pinned by tests/test_engine_protocol_roundtrip.py.

Why dataclasses + TypedDict
  - TypedDicts give us JSON-shape types matching the TS exactly (the wire
    format)
  - Dataclasses give us ergonomic in-engine objects (`EngineObservation`
    instances with attribute access, `__eq__`, etc.)
  - Converters between the two (`from_dict` / `to_dict`) keep wire-format
    handling at the boundary

This file MUST stay free of imports from `fow_chess` engine internals —
it's a pure protocol module. The engine adapter (separate file) does the
mapping between protocol-level Observation and the engine's internal
`fow_chess.observation.Observation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NotRequired, Optional, TypedDict


# ---------------------------------------------------------------------------
# JSON wire types (TypedDict) — match the TS file 1:1
# ---------------------------------------------------------------------------

ProtocolVersion = Literal["1"]
Color = Literal["white", "black"]
# Chess: P N B R Q K. Dark Mini Xiangqi: G(eneral) H(orse) C(annon) R(chariot)
# S(oldier). Full Dark Xiangqi uses K/A/B/N/R/C/P for
# general/advisor/elephant/horse/chariot/cannon/soldier. R/B/N/P are
# disambiguated by the request's game.
PieceLetter = Literal["P", "N", "B", "R", "Q", "K", "G", "H", "C", "S", "A"]
PromotionRole = Literal["queen", "rook", "bishop", "knight"]


class VisiblePieceJson(TypedDict):
    type: PieceLetter
    color: Color


class GameOverJson(TypedDict):
    winner: Optional[Color]
    reason: str


class EngineObservationJson(TypedDict, total=False):
    ply: int
    kind: Literal["initial", "own_move", "opp_move"]
    own_move: Optional["MoveJson"]
    visibility_mask: str  # "0x..." 64-bit hex
    visible_pieces: list[tuple[int, VisiblePieceJson]]
    # Dark Mini Xiangqi only: squares revealed color-only (cannon screens / horse
    # legs). Absent/empty for chess. The belief uses it but the role stays hidden.
    shrouded: list[tuple[int, Color]]
    own_capture_square: Optional[int]
    opp_capture_landing_square: Optional[int]
    game_over: Optional[GameOverJson]


class EngineClockJson(TypedDict):
    remaining_ms: Optional[int]
    increment_ms: int


# Wire shape of the TS `Move`: string square NAMES ("e2"), under the literal
# key "from" — a Python keyword, hence the functional TypedDict form (the old
# class form declared `from_: int`, matching neither the TS type nor this
# module's own serializers). `move_from_json` still ACCEPTS a legacy "from_"
# key on input.
MoveJson = TypedDict(
    "MoveJson",
    {
        "from": str,
        "to": str,
        "promotion": NotRequired[Optional[PromotionRole]],
    },
)


class EngineTurnRequestJson(TypedDict, total=False):
    protocolVersion: ProtocolVersion
    gameId: str
    gameSpecId: str  # "dark-chess" (default) | "dark-mini-xiangqi" | "dark-xiangqi"
    engineId: str
    sessionId: str
    color: Color
    ply: int
    engineSeed: int
    clock: EngineClockJson
    legalMoves: list[MoveJson]
    observationTranscript: list[EngineObservationJson]
    latestObservationDelta: EngineObservationJson


class EngineTurnResponseJson(TypedDict, total=False):
    protocolVersion: ProtocolVersion
    gameId: str
    sessionId: str
    move: MoveJson
    diagnostics: dict


# ---------------------------------------------------------------------------
# Dataclass mirrors — what engine code actually holds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisiblePiece:
    type: PieceLetter
    color: Color


@dataclass(frozen=True)
class GameOver:
    winner: Optional[Color]
    reason: str


@dataclass(frozen=True)
class EngineObservation:
    ply: int
    kind: Literal["initial", "own_move", "opp_move"]
    visibility_mask: int  # held as int internally; serialized to "0x…" hex
    visible_pieces: tuple[tuple[int, VisiblePiece], ...] = field(default_factory=tuple)
    own_capture_square: Optional[int] = None
    opp_capture_landing_square: Optional[int] = None
    game_over: Optional[GameOver] = None
    # DMX only: color-only reveals (cannon screens / horse legs). Empty for chess.
    shrouded: tuple[tuple[int, Color], ...] = field(default_factory=tuple)
    # Present when kind == 'own_move' — the move the engine made this ply.
    # Required for the engine to deterministically advance its belief set
    # during cold-start transcript replay. Null for 'initial' and 'opp_move'.
    own_move: Optional["Move"] = None


@dataclass(frozen=True)
class EngineClock:
    remaining_ms: Optional[int]
    increment_ms: int


@dataclass(frozen=True)
class Move:
    from_square: int
    to_square: int
    promotion: Optional[PromotionRole] = None


@dataclass(frozen=True)
class EngineTurnRequest:
    protocol_version: ProtocolVersion
    game_id: str
    engine_id: str
    session_id: str
    color: Color
    ply: int
    engine_seed: int
    clock: EngineClock
    legal_moves: tuple[Move, ...]
    # Defaulted so every existing chess construction site works unchanged.
    game_spec_id: str = "dark-chess"
    observation_transcript: Optional[tuple[EngineObservation, ...]] = None
    latest_observation_delta: Optional[EngineObservation] = None


@dataclass(frozen=True)
class EngineTurnResponse:
    protocol_version: ProtocolVersion
    game_id: str
    session_id: str
    move: Move
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wire-format ⇄ dataclass conversion
# ---------------------------------------------------------------------------


def _hex_mask(n: int) -> str:
    if n.bit_length() <= 64:
        return f"0x{n & 0xFFFFFFFFFFFFFFFF:016x}"
    return f"0x{n:x}"


def _parse_mask(s: str) -> int:
    return int(s, 16) if s.startswith("0x") else int(s)


def observation_to_json(o: EngineObservation, size: int = 8) -> EngineObservationJson:
    out: EngineObservationJson = {
        "ply": o.ply,
        "kind": o.kind,
        "own_move": move_to_json(o.own_move, size) if o.own_move is not None else None,
        "visibility_mask": _hex_mask(o.visibility_mask),
        "visible_pieces": [
            (sq, {"type": p.type, "color": p.color}) for sq, p in o.visible_pieces
        ],
        "own_capture_square": o.own_capture_square,
        "opp_capture_landing_square": o.opp_capture_landing_square,
        "game_over": (
            {"winner": o.game_over.winner, "reason": o.game_over.reason}
            if o.game_over is not None
            else None
        ),
    }
    # Only emitted for DMX (chess always has empty shrouded → chess JSON unchanged).
    if o.shrouded:
        out["shrouded"] = [(sq, color) for sq, color in o.shrouded]
    return out


def observation_from_json(d: EngineObservationJson, size: int = 8) -> EngineObservation:
    go_json = d.get("game_over")
    go = None
    if go_json is not None:
        go = GameOver(winner=go_json.get("winner"), reason=go_json["reason"])
    own_move_json = d.get("own_move")
    own_move = move_from_json(own_move_json, size) if own_move_json is not None else None
    return EngineObservation(
        ply=d["ply"],
        kind=d["kind"],
        visibility_mask=_parse_mask(d["visibility_mask"]),
        visible_pieces=tuple(
            (sq, VisiblePiece(type=vp["type"], color=vp["color"]))
            for sq, vp in d.get("visible_pieces", [])
        ),
        shrouded=tuple((sq, color) for sq, color in d.get("shrouded", [])),
        own_capture_square=d.get("own_capture_square"),
        opp_capture_landing_square=d.get("opp_capture_landing_square"),
        game_over=go,
        own_move=own_move,
    )


_SQUARE_FILES = "abcdefghi"


def _board_size(game_spec_id: str) -> int:
    """Board width / row stride for the protocol geometry."""
    if game_spec_id == "dark-mini-xiangqi":
        return 7
    if game_spec_id == "dark-xiangqi":
        return 9
    return 8


def _board_height(size: int) -> int:
    return 10 if size == 9 else size


def _square_to_int(value: object, size: int = 8) -> int:
    """Accept either an int square index or a string square name and return the
    integer. ``size`` is the board width/stride (8 chess / 7 mini / 9 xiangqi):
    idx = rank0*size+file.
    The TS side emits string-square Move objects on the wire; this absorbs both."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and len(value) >= 2:
        file = ord(value[0]) - ord("a")
        try:
            rank = int(value[1:]) - 1
        except ValueError as exc:
            raise ValueError(f"invalid square: {value!r}") from exc
        if 0 <= file < size and 0 <= rank < _board_height(size):
            return rank * size + file
    raise ValueError(f"invalid square: {value!r}")


def _int_to_square(idx: int, size: int = 8) -> str:
    """Inverse of `_square_to_int` (size = board width/stride). For
    chess `% 8` / `// 8` equal the old `& 7` / `>> 3`, so chess is byte-identical."""
    return f"{_SQUARE_FILES[idx % size]}{idx // size + 1}"


_PROMOTION_ALIASES: dict[str, PromotionRole] = {
    "queen": "queen",
    "rook": "rook",
    "bishop": "bishop",
    "knight": "knight",
    "q": "queen",
    "r": "rook",
    "b": "bishop",
    "n": "knight",
    "Q": "queen",
    "R": "rook",
    "B": "bishop",
    "N": "knight",
}


def _promotion_from_json(value: object) -> Optional[PromotionRole]:
    if value is None:
        return None
    if isinstance(value, str) and value in _PROMOTION_ALIASES:
        return _PROMOTION_ALIASES[value]
    raise ValueError(f"invalid promotion: {value!r}")


def move_to_json(m: Move, size: int = 8) -> MoveJson:
    out: MoveJson = {
        "from": _int_to_square(m.from_square, size),
        "to": _int_to_square(m.to_square, size),
    }
    if m.promotion is not None:
        out["promotion"] = m.promotion
    return out


def move_from_json(d: MoveJson, size: int = 8) -> Move:
    # JSON-key "from" is reserved in Python; we accept both "from_" and "from".
    from_sq = d.get("from_", d.get("from"))  # type: ignore[arg-type]
    if from_sq is None:
        raise ValueError("MoveJson missing 'from' / 'from_'")
    return Move(
        from_square=_square_to_int(from_sq, size),
        to_square=_square_to_int(d["to"], size),
        promotion=_promotion_from_json(d.get("promotion")),
    )


def request_to_json(r: EngineTurnRequest) -> EngineTurnRequestJson:
    size = _board_size(r.game_spec_id)
    out: EngineTurnRequestJson = {
        "protocolVersion": r.protocol_version,
        "gameId": r.game_id,
        "engineId": r.engine_id,
        "sessionId": r.session_id,
        "color": r.color,
        "ply": r.ply,
        "engineSeed": r.engine_seed,
        "clock": {
            "remaining_ms": r.clock.remaining_ms,
            "increment_ms": r.clock.increment_ms,
        },
        "legalMoves": [move_to_json(m, size) for m in r.legal_moves],
    }
    # Omitted for dark-chess so existing chess request JSON stays byte-identical.
    if r.game_spec_id != "dark-chess":
        out["gameSpecId"] = r.game_spec_id
    if r.observation_transcript is not None:
        out["observationTranscript"] = [
            observation_to_json(o, size) for o in r.observation_transcript
        ]
    if r.latest_observation_delta is not None:
        out["latestObservationDelta"] = observation_to_json(r.latest_observation_delta, size)
    return out


def request_from_json(d: EngineTurnRequestJson) -> EngineTurnRequest:
    game_spec_id = d.get("gameSpecId", "dark-chess")
    size = _board_size(game_spec_id)
    transcript_json = d.get("observationTranscript")
    delta_json = d.get("latestObservationDelta")
    return EngineTurnRequest(
        protocol_version=d["protocolVersion"],
        game_id=d["gameId"],
        engine_id=d["engineId"],
        game_spec_id=game_spec_id,
        session_id=d["sessionId"],
        color=d["color"],
        ply=d["ply"],
        engine_seed=d["engineSeed"],
        clock=EngineClock(
            remaining_ms=d["clock"]["remaining_ms"],
            increment_ms=d["clock"]["increment_ms"],
        ),
        legal_moves=tuple(move_from_json(m, size) for m in d["legalMoves"]),
        observation_transcript=(
            tuple(observation_from_json(o, size) for o in transcript_json)
            if transcript_json is not None
            else None
        ),
        latest_observation_delta=(
            observation_from_json(delta_json, size) if delta_json is not None else None
        ),
    )


def response_to_json(r: EngineTurnResponse, size: int = 8) -> EngineTurnResponseJson:
    out: EngineTurnResponseJson = {
        "protocolVersion": r.protocol_version,
        "gameId": r.game_id,
        "sessionId": r.session_id,
        "move": move_to_json(r.move, size),
    }
    # TS declares `diagnostics?` — omit the key entirely when empty so a
    # no-diagnostics response is exactly the TS-shaped payload. (The live
    # worker always attaches telemetry, so the prod wire is unchanged.)
    if r.diagnostics:
        out["diagnostics"] = dict(r.diagnostics)
    return out


def response_from_json(d: EngineTurnResponseJson, size: int = 8) -> EngineTurnResponse:
    # The response carries no game tag; the caller knows the game and passes size.
    return EngineTurnResponse(
        protocol_version=d["protocolVersion"],
        game_id=d["gameId"],
        session_id=d["sessionId"],
        move=move_from_json(d["move"], size),
        diagnostics=dict(d.get("diagnostics", {})),
    )


# ---------------------------------------------------------------------------
# Post-move observation push (server → engine) + ack — TS
# `EngineObservationPush` / `EngineObservationAck`. Opt-in and additive: an
# engine that ignores the push still plays correctly (the same own_move
# observation arrives in its next EngineTurnRequest transcript); an engine
# handling BOTH must dedupe by ply.
# ---------------------------------------------------------------------------


class EngineObservationPushJson(TypedDict, total=False):
    protocolVersion: ProtocolVersion
    gameId: str
    engineId: str
    gameSpecId: str
    sessionId: str
    color: Color
    ply: int
    observation: EngineObservationJson


class EngineObservationAckJson(TypedDict):
    protocolVersion: ProtocolVersion
    gameId: str
    sessionId: str
    received: Literal[True]


@dataclass(frozen=True)
class EngineObservationPush:
    protocol_version: ProtocolVersion
    game_id: str
    engine_id: str
    session_id: str
    color: Color
    ply: int
    observation: EngineObservation
    game_spec_id: str = "dark-chess"


@dataclass(frozen=True)
class EngineObservationAck:
    protocol_version: ProtocolVersion
    game_id: str
    session_id: str
    received: Literal[True] = True


@dataclass(frozen=True)
class EngineSessionIdInputs:
    """How the server constructs sessionId (TS `EngineSessionIdInputs`).

    Engines treat sessionId as opaque; this mirror exists so logging and
    persistence on the engine side can format the same way."""

    game_id: str
    engine_id: str
    color: Color


def observation_push_to_json(p: EngineObservationPush) -> EngineObservationPushJson:
    size = _board_size(p.game_spec_id)
    out: EngineObservationPushJson = {
        "protocolVersion": p.protocol_version,
        "gameId": p.game_id,
        "engineId": p.engine_id,
        "sessionId": p.session_id,
        "color": p.color,
        "ply": p.ply,
        "observation": observation_to_json(p.observation, size),
    }
    if p.game_spec_id != "dark-chess":
        out["gameSpecId"] = p.game_spec_id
    return out


def observation_push_from_json(d: EngineObservationPushJson) -> EngineObservationPush:
    game_spec_id = d.get("gameSpecId", "dark-chess")
    size = _board_size(game_spec_id)
    return EngineObservationPush(
        protocol_version=d["protocolVersion"],
        game_id=d["gameId"],
        engine_id=d["engineId"],
        session_id=d["sessionId"],
        color=d["color"],
        ply=d["ply"],
        observation=observation_from_json(d["observation"], size),
        game_spec_id=game_spec_id,
    )


def observation_ack_to_json(a: EngineObservationAck) -> EngineObservationAckJson:
    return {
        "protocolVersion": a.protocol_version,
        "gameId": a.game_id,
        "sessionId": a.session_id,
        "received": True,
    }


def observation_ack_from_json(d: EngineObservationAckJson) -> EngineObservationAck:
    if d.get("received") is not True:
        raise ValueError("EngineObservationAck.received must be true")
    return EngineObservationAck(
        protocol_version=d["protocolVersion"],
        game_id=d["gameId"],
        session_id=d["sessionId"],
    )
