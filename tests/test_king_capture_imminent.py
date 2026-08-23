"""Unit tests for the king-aware leaf-eval shim (Concern 4b in the parity audit).

The narrow rule: side-to-move can capture opponent's king on the next ply.
Covers en-prise (queen attacks king through cleared diagonal) + adjacent
kings. Must NOT fire on normal in-check (side-to-move is in check — valid
in standard chess, Stockfish handles it) or on a normal valid position.
"""
from __future__ import annotations

import chess
import pytest

from fow_chess.cfr.leaf_eval import (
    king_aware_leaf_enabled,
    king_capture_imminent,
    material_leaf_eval,
    set_king_aware_leaf,
)


@pytest.fixture
def king_aware_on():
    """Enable the flag for the duration of one test and restore after."""
    prior = king_aware_leaf_enabled()
    set_king_aware_leaf(True)
    yield
    set_king_aware_leaf(prior)


@pytest.fixture
def king_aware_off():
    """Force flag OFF and restore after — for verifying default behavior."""
    prior = king_aware_leaf_enabled()
    set_king_aware_leaf(False)
    yield
    set_king_aware_leaf(prior)


def _build(moves: list[str]) -> chess.Board:
    b = chess.Board()
    for uci in moves:
        b.push(chess.Move.from_uci(uci))
    return b


def test_qa5_diag_canonical_white_king_en_prise():
    """1.e4 c5 2.d4 Qa5 3.d5 → Black to move, Qa5 attacks Ke1 through
    the cleared a5-e1 diagonal. Side-to-move (Black) captures next ply."""
    board = _build(["e2e4", "c7c5", "d2d4", "d8a5", "d4d5"])
    assert not board.is_valid()
    assert board.turn == chess.BLACK
    assert king_capture_imminent(board, chess.BLACK) == 1.0
    assert king_capture_imminent(board, chess.WHITE) == -1.0


def test_adjacent_kings_side_to_move_wins():
    """Kings on adjacent squares — each attacks the other; side-to-move
    captures first."""
    board = chess.Board("8/8/8/3k4/3K4/8/8/8 w - - 0 1")
    assert not board.is_valid()
    assert board.turn == chess.WHITE
    assert king_capture_imminent(board, chess.WHITE) == 1.0
    assert king_capture_imminent(board, chess.BLACK) == -1.0
    # Black to move — sign flips.
    board.turn = chess.BLACK
    assert king_capture_imminent(board, chess.BLACK) == 1.0
    assert king_capture_imminent(board, chess.WHITE) == -1.0


def test_normal_in_check_must_not_fire():
    """Side-to-move is in check — valid in standard chess. The narrow rule
    must NOT fire (side-to-move CANNOT capture opp's king; they have to
    defend their own)."""
    # Black's queen checks White's king. Valid in standard chess.
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_valid()
    assert board.is_check()
    assert king_capture_imminent(board, chess.WHITE) is None
    assert king_capture_imminent(board, chess.BLACK) is None


def test_normal_valid_opening_does_not_fire():
    """Standard opening position — no king-capture available."""
    board = chess.Board()
    assert board.is_valid()
    assert king_capture_imminent(board, chess.WHITE) is None
    assert king_capture_imminent(board, chess.BLACK) is None


def test_missing_king_returns_none():
    """If the target king is already gone, the helper returns None
    (the terminal-value path owns king-captured positions)."""
    # Position with no Black king.
    board = chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1")
    assert board.king(chess.BLACK) is None
    assert king_capture_imminent(board, chess.WHITE) is None
    assert king_capture_imminent(board, chess.BLACK) is None


def test_flag_off_material_leaf_eval_unchanged(king_aware_off):
    """With the flag OFF, material_leaf_eval on the en-prise position
    returns the old buggy ~0.0 (both kings on board, material balanced).
    Sanity-checks that the patch is gated."""
    board = _build(["e2e4", "c7c5", "d2d4", "d8a5", "d4d5"])
    assert material_leaf_eval(board, chess.BLACK) == pytest.approx(0.0, abs=1e-9)
    assert material_leaf_eval(board, chess.WHITE) == pytest.approx(0.0, abs=1e-9)


def test_flag_on_material_leaf_eval_emits_king_signal(king_aware_on):
    """With the flag ON, material_leaf_eval emits ±1.0 on the en-prise
    position — the bug fix."""
    board = _build(["e2e4", "c7c5", "d2d4", "d8a5", "d4d5"])
    assert material_leaf_eval(board, chess.BLACK) == 1.0
    assert material_leaf_eval(board, chess.WHITE) == -1.0


def test_flag_on_does_not_affect_valid_positions(king_aware_on):
    """The helper only fires on king-capture-imminent positions; valid
    positions still get the material score."""
    board = chess.Board()
    assert material_leaf_eval(board, chess.WHITE) == pytest.approx(0.0, abs=1e-9)
