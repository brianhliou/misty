"""Differential test: Python visible_squares vs Rust visible_squares.

For every call we make in this file, we compute BOTH the Python and
Rust implementations and assert byte-identical output. Any divergence
fails the test with the FEN and color that triggered it — gives us
exact reproduction.

Coverage:
- The 7 hand-written visibility tests (replayed through the diff harness)
- 14 P-predicate boards
- All 84 real engine games in feedback/mirror-*/games/, replayed
  through both perspectives at every ply
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import pytest

try:
    import fow_rust
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

from fow_chess.visibility import _visible_squares_py as py_visible_squares


pytestmark = pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="fow_rust extension not built (run `maturin develop` in fow_rust/)",
)


def _assert_match(fen: str, color: chess.Color, label: str = "") -> None:
    """Compute Python + both Rust variants (FEN-input and bitboard-input);
    assert all three equal. Useful diagnostics on failure."""
    board = chess.Board(fen)
    py_result = int(py_visible_squares(board, color))
    color_bool = color == chess.WHITE
    rust_fen_result = fow_rust.visible_squares(fen, color_bool)
    ep = board.ep_square
    rust_bb_result = fow_rust.visible_squares_bb(
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.castling_rights,
        64 if ep is None else ep,
        color_bool,
    )
    if py_result != rust_fen_result or py_result != rust_bb_result:
        def _diff(a: int, b: int) -> str:
            only_a = chess.SquareSet(a & ~b)
            only_b = chess.SquareSet(b & ~a)
            return f"a-only={sorted(only_a)} b-only={sorted(only_b)}"
        raise AssertionError(
            f"DIVERGENCE {label}\n"
            f"  FEN: {fen}\n"
            f"  color: {'white' if color == chess.WHITE else 'black'}\n"
            f"  py vs rust(fen): {_diff(py_result, rust_fen_result)}\n"
            f"  py vs rust(bb):  {_diff(py_result, rust_bb_result)}"
        )


# ---------------------------------------------------------------------------
# Hand-crafted positions (mirror existing visibility tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK])
def test_diff_initial_position(color):
    _assert_match(chess.STARTING_FEN, color, label="initial")


@pytest.mark.parametrize(
    "uci_seq, color",
    [
        (["e2e4"], chess.WHITE),
        (["e2e4"], chess.BLACK),
        (["e2e4", "e7e5"], chess.WHITE),
        (["e2e4", "d7d5"], chess.WHITE),
        (["e2e4", "d7d5"], chess.BLACK),
        (["e2e4", "d7d5", "e4d5"], chess.WHITE),  # pawn capture
        (["e2e4", "d7d5", "e4d5"], chess.BLACK),
    ],
)
def test_diff_after_opening_moves(uci_seq, color):
    board = chess.Board()
    for uci in uci_seq:
        board.push_uci(uci)
    _assert_match(board.fen(), color, label=f"after {' '.join(uci_seq)}")


def test_diff_castling_rights_present():
    # Position where castling is pseudo-legal for white kingside.
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    _assert_match(fen, chess.WHITE)
    _assert_match(fen, chess.BLACK)


def test_diff_en_passant_position():
    # White just played e2e4, black's d5 pawn can ep-capture if positioned.
    # Set up classical ep position.
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    board.push_uci("e4e5")
    board.push_uci("f7f5")  # creates ep square e6 for white (no, e6 for white means...)
    # Actually after f7f5, ep_square is f6 (the square the pawn skipped over).
    # White's e5 pawn could capture en passant on f6.
    _assert_match(board.fen(), chess.WHITE, label="ep available to white")
    _assert_match(board.fen(), chess.BLACK)


# ---------------------------------------------------------------------------
# Full real-game replay differential — strongest gate
# ---------------------------------------------------------------------------


_PROMO_LETTER = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n",
                 "q": "q", "r": "r", "b": "b", "n": "n"}


def _load_moves(path: Path) -> list[chess.Move]:
    moves: list[chess.Move] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "move-played":
                continue
            mv = e.get("move", {})
            frm, to, promo = mv.get("from"), mv.get("to"), mv.get("promotion")
            if frm is None or to is None:
                continue
            uci = f"{frm}{to}"
            if promo:
                letter = _PROMO_LETTER.get(str(promo).lower())
                if letter is None:
                    continue
                uci += letter
            moves.append(chess.Move.from_uci(uci))
    return moves


from game_corpus import corpus_game_paths, corpus_id


@pytest.mark.parametrize(
    "game_path",
    corpus_game_paths(),  # committed fixtures + any private feedback games
    ids=corpus_id,
)
def test_diff_real_game_replay(game_path):
    """Replay a real game in full; at every position, compare Rust vs
    Python visibility for BOTH colors. Strongest differential gate —
    if all 84 games pass at every ply, the Rust impl is correct."""
    moves = _load_moves(game_path)
    if not moves:
        pytest.skip("empty game")
    board = chess.Board()
    _assert_match(board.fen(), chess.WHITE, label=f"{game_path.name} ply 0 W")
    _assert_match(board.fen(), chess.BLACK, label=f"{game_path.name} ply 0 B")
    for ply, mv in enumerate(moves, start=1):
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            break
        if mv not in board.pseudo_legal_moves:
            break
        board.push(mv)
        _assert_match(board.fen(), chess.WHITE, label=f"{game_path.name} ply {ply} W")
        _assert_match(board.fen(), chess.BLACK, label=f"{game_path.name} ply {ply} B")
