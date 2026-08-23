"""Variant-aware engine protocol — Dark Mini Xiangqi (7x7) path.

The protocol module is shared between dark chess (8x8) and DMX (7x7). These pin
the mini path: 7-wide square geometry, mini piece letters, and the DMX-only
`shrouded` reveal channel (cannon screens / horse legs) — and confirm chess
stays byte-identical (no gameSpecId / shrouded keys emitted for chess).
"""

from __future__ import annotations

from fow_chess import engine_protocol as proto


def test_mini_square_geometry_differs_from_chess():
    # "e4": file e=4, rank 4 -> rank0=3. mini idx = 3*7+4 = 25; chess = 3*8+4 = 28.
    assert proto._square_to_int("e4", 7) == 25
    assert proto._square_to_int("e4", 8) == 28
    assert proto._int_to_square(25, 7) == "e4"
    assert proto._square_to_int("a1", 7) == 0
    assert proto._square_to_int("g7", 7) == 48
    assert proto._int_to_square(48, 7) == "g7"
    assert proto._board_size("dark-mini-xiangqi") == 7
    assert proto._board_size("dark-chess") == 8


def _mini_request() -> proto.EngineTurnRequest:
    obs0 = proto.EngineObservation(
        ply=0,
        kind="initial",
        visibility_mask=0b101,
        visible_pieces=((3, proto.VisiblePiece(type="G", color="red")),),
        shrouded=((10, "black"),),
    )
    obs1 = proto.EngineObservation(
        ply=1,
        kind="own_move",
        visibility_mask=0b11,
        visible_pieces=(),
        own_move=proto.Move(
            from_square=proto._square_to_int("a1", 7),
            to_square=proto._square_to_int("a3", 7),
        ),
    )
    return proto.EngineTurnRequest(
        protocol_version="1",
        game_id="g",
        engine_id="v2",
        game_spec_id="dark-mini-xiangqi",
        session_id="s",
        color="red",
        ply=2,
        engine_seed=7,
        clock=proto.EngineClock(remaining_ms=None, increment_ms=0),
        legal_moves=(
            proto.Move(
                from_square=proto._square_to_int("d1", 7),
                to_square=proto._square_to_int("d2", 7),
            ),
        ),
        observation_transcript=(obs0, obs1),
    )


def test_mini_request_roundtrip():
    req = _mini_request()
    back = proto.request_from_json(proto.request_to_json(req))
    assert back.game_spec_id == "dark-mini-xiangqi"
    assert back.legal_moves == req.legal_moves
    assert back.observation_transcript is not None
    assert back.observation_transcript[0].shrouded == ((10, "black"),)
    assert back.observation_transcript[0].visible_pieces[0][1].type == "G"
    # own_move round-trips through 7-wide geometry (string a1/a3 ↔ mini ints).
    assert back.observation_transcript[1].own_move == req.observation_transcript[1].own_move


def test_mini_response_roundtrip_uses_mini_geometry():
    resp = proto.EngineTurnResponse(
        protocol_version="1",
        game_id="g",
        session_id="s",
        move=proto.Move(
            from_square=proto._square_to_int("d1", 7),
            to_square=proto._square_to_int("d7", 7),  # flying-general capture
        ),
    )
    j = proto.response_to_json(resp, size=7)
    assert j["move"]["from"] == "d1" and j["move"]["to"] == "d7"
    assert proto.response_from_json(j, size=7).move == resp.move


def test_chess_request_json_byte_identical_and_defaults():
    chess_req = proto.EngineTurnRequest(
        protocol_version="1",
        game_id="g",
        engine_id="v2",
        session_id="s",
        color="white",
        ply=0,
        engine_seed=1,
        clock=proto.EngineClock(remaining_ms=None, increment_ms=0),
        legal_moves=(
            proto.Move(
                from_square=proto._square_to_int("e2", 8),
                to_square=proto._square_to_int("e4", 8),
            ),
        ),
    )
    j = proto.request_to_json(chess_req)
    assert "gameSpecId" not in j  # chess request JSON unchanged
    assert proto.request_from_json(j).game_spec_id == "dark-chess"  # parse defaults
