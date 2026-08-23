from __future__ import annotations

import chess

from fow_chess.engine_protocol import Move, move_from_json, move_to_json
from fow_chess.protocol_adapter import move_from_protocol, move_to_protocol


def test_move_promotion_accepts_typescript_role_names() -> None:
    move = move_from_json({"from": "e7", "to": "e8", "promotion": "queen"})

    assert move == Move(from_square=chess.E7, to_square=chess.E8, promotion="queen")
    assert move_from_protocol(move) == chess.Move.from_uci("e7e8q")
    assert move_to_json(move) == {"from": "e7", "to": "e8", "promotion": "queen"}


def test_move_promotion_accepts_legacy_piece_letters() -> None:
    move = move_from_json({"from": "e7", "to": "e8", "promotion": "Q"})

    assert move == Move(from_square=chess.E7, to_square=chess.E8, promotion="queen")
    assert move_from_protocol(move) == chess.Move.from_uci("e7e8q")
    assert move_to_json(move) == {"from": "e7", "to": "e8", "promotion": "queen"}


def test_move_to_protocol_emits_typescript_role_name_promotion() -> None:
    move = move_to_protocol(chess.Move.from_uci("e7e8q"))

    assert move == Move(from_square=chess.E7, to_square=chess.E8, promotion="queen")
    assert move_to_json(move) == {"from": "e7", "to": "e8", "promotion": "queen"}
