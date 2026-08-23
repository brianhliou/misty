"""Move-encoding round-trip across the protocol boundary (risk A3).

The production-fidelity corpus sweep under-stresses the special moves —
castling, all four promotions, capture-promotions, en passant — because they
live late in games. This pins them deterministically: every special move must
survive the wire round-trip (chess.Move -> protocol -> JSON -> protocol ->
chess.Move) and encode correctly in the worker's response, so the engine never
emits a move the server rejects.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import pytest

import live_move_worker as worker
from fow_chess import engine_protocol as proto
from fow_chess.protocol_adapter import move_from_protocol, move_to_protocol

S = chess.parse_square

SPECIAL_MOVES = [
    pytest.param(chess.Move(S("e1"), S("g1")), id="castle-kingside"),
    pytest.param(chess.Move(S("e1"), S("c1")), id="castle-queenside"),
    pytest.param(chess.Move(S("e8"), S("g8")), id="castle-kingside-black"),
    pytest.param(chess.Move(S("a7"), S("a8"), promotion=chess.QUEEN), id="promo-Q"),
    pytest.param(chess.Move(S("a7"), S("a8"), promotion=chess.ROOK), id="promo-R"),
    pytest.param(chess.Move(S("a7"), S("a8"), promotion=chess.BISHOP), id="promo-B"),
    pytest.param(chess.Move(S("a7"), S("a8"), promotion=chess.KNIGHT), id="promo-N"),
    pytest.param(chess.Move(S("b7"), S("a8"), promotion=chess.QUEEN), id="capture-promo-Q"),
    pytest.param(chess.Move(S("e5"), S("d6")), id="en-passant"),
    pytest.param(chess.Move(S("e2"), S("e4")), id="double-push"),
]


def _wire_roundtrip(mv: chess.Move) -> chess.Move:
    """chess.Move -> protocol Move -> request JSON -> back -> chess.Move."""
    pm = move_to_protocol(mv)
    req = proto.EngineTurnRequest(
        protocol_version="1", game_id="g", engine_id="e", session_id="s",
        color="white", ply=1, engine_seed=0,
        clock=proto.EngineClock(remaining_ms=1000, increment_ms=0),
        legal_moves=(pm,),
        observation_transcript=(
            proto.EngineObservation(ply=1, kind="own_move", visibility_mask=0, own_move=pm),
        ),
    )
    req2 = proto.request_from_json(proto.request_to_json(req))
    assert move_from_protocol(req2.legal_moves[0]) == mv  # legal_moves survives
    assert move_from_protocol(req2.observation_transcript[0].own_move) == mv  # own_move survives
    return move_from_protocol(req2.legal_moves[0])


@pytest.mark.parametrize("mv", SPECIAL_MOVES)
def test_special_move_survives_wire_roundtrip(mv):
    assert _wire_roundtrip(mv) == mv


@pytest.mark.parametrize("promo,letter", [
    (chess.QUEEN, "queen"), (chess.ROOK, "rook"),
    (chess.BISHOP, "bishop"), (chess.KNIGHT, "knight"),
])
def test_response_encodes_promotion_letter(promo, letter):
    mv = chess.Move(S("a7"), S("a8"), promotion=promo)
    req = types.SimpleNamespace(game_id="room-1")
    resp = worker._move_response({"id": "python-v2-current"}, mv, req, "v2")
    assert resp["move"]["from"] == "a7"
    assert resp["move"]["to"] == "a8"
    assert resp["move"]["promotion"] == letter


def test_response_castling_is_king_destination():
    # The response uses the king's destination (e1->g1), which the server
    # accepts via its alias generation (see _move_response docstring).
    mv = chess.Move(S("e1"), S("g1"))
    req = types.SimpleNamespace(game_id="room-1")
    resp = worker._move_response({"id": "python-v2-current"}, mv, req, "v2")
    assert resp["move"]["from"] == "e1"
    assert resp["move"]["to"] == "g1"
    assert "promotion" not in resp["move"]
