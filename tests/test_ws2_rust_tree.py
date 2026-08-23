"""WS2 slice 1: the Rust tree (EqEngine) can hold a node's board and build a root
from a FEN — the foundation for moving expansion/selection into Rust. Verifies
the board round-trips and matches the Python `root_node`'s board exactly.
"""
import random

import chess
import pytest

import fow_rust
from fow_chess.cfr.gt_cfr import root_node

pytestmark = pytest.mark.skipif(
    not hasattr(fow_rust, "EqEngine") or not hasattr(fow_rust.EqEngine, "add_root_from_fen"),
    reason="fow_rust EqEngine.add_root_from_fen not built",
)


def _engine():
    st = random.Random(0).getstate()[1]
    return fow_rust.EqEngine(list(st[:624]), st[624])


def _random_positions(n):
    rng = random.Random(7)
    out = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randint(2, 40)):
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(rng.choice(legal))
            if not b.king(chess.WHITE) or not b.king(chess.BLACK):
                break
        out.append(b.fen())
    return out


def test_rust_root_board_roundtrips_and_matches_python():
    eng = _engine()
    for fen in _random_positions(30):
        nid = eng.add_root_from_fen(fen)
        rust_fen = eng.node_fen(nid)
        # round-trip: the Rust tree's packed board serializes back to the FEN
        assert rust_fen == fen, f"round-trip mismatch: {fen!r} -> {rust_fen!r}"
        # equivalence: matches the board the Python root_node would hold
        assert rust_fen == root_node(chess.Board(fen)).truth.fen()


def test_node_fen_none_for_boardless_node():
    # mirror-built nodes (add_node) carry no board → node_fen is None
    eng = _engine()
    nid = eng.add_node(True, False, 0.0, 0.0, 0)
    assert eng.node_fen(nid) is None
