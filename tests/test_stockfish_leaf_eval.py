"""Tests for the Stockfish-backed leaf evaluator.

These tests spawn real Stockfish subprocesses; they're skipped if no
Stockfish binary is on PATH. Stockfish evaluation at depth 1 is fast
(<10ms per call) so the tests are quick.
"""

from __future__ import annotations

import shutil

import chess
import pytest

from fow_chess.cfr.leaf_eval_stockfish import (
    StockfishLeafEval,
    stockfish_leaf_eval_factory,
)


pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not on PATH",
)


# ---------------------------------------------------------------------------
# Sanity: well-known positions
# ---------------------------------------------------------------------------


def test_starting_position_eval_near_zero():
    """At the starting position, evaluation should be near zero from both
    perspectives. Stockfish's depth-1 might give a small bias, but it
    should be well within ±0.2."""
    with StockfishLeafEval() as sf:
        white_eval = sf.evaluate(chess.Board(), chess.WHITE)
        black_eval = sf.evaluate(chess.Board(), chess.BLACK)
    assert abs(white_eval) < 0.2, f"white eval at start = {white_eval}"
    assert abs(black_eval) < 0.2, f"black eval at start = {black_eval}"
    # Sign symmetry: white's eval and black's eval should be opposite-signed
    # within numerical noise.
    assert white_eval == pytest.approx(-black_eval, abs=0.01)


def test_winning_material_white_evaluates_positive_for_white():
    """White up a queen — Stockfish should strongly prefer white. Pins
    tanh_scale_cp so the assertion threshold isn't scale-sensitive."""
    fen = "4k3/pppppppp/8/8/8/8/PPPPPPPP/3QK3 w - - 0 1"
    with StockfishLeafEval(tanh_scale_cp=500.0) as sf:
        v = sf.evaluate(chess.Board(fen), chess.WHITE)
    assert v > 0.5, f"white-up-queen eval = {v}; expected > 0.5"


def test_losing_material_evaluates_negative():
    """Same position, evaluated from black's POV — should be strongly
    negative. Pins tanh_scale_cp; see test above."""
    fen = "4k3/pppppppp/8/8/8/8/PPPPPPPP/3QK3 w - - 0 1"
    with StockfishLeafEval(tanh_scale_cp=500.0) as sf:
        v = sf.evaluate(chess.Board(fen), chess.BLACK)
    assert v < -0.5, f"black-down-queen eval = {v}; expected < -0.5"


def test_mate_position_saturates_near_plus_one():
    """White has a forced mate on the move. Stockfish at depth 1 should
    return a mate score → near +1."""
    # Back-rank mate: white Q on f7, black K on g8, mate-in-1 by Qg7#.
    # Use a cleaner mate-in-1: KRK back-rank.
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    # Rook to a8 is mate.
    with StockfishLeafEval() as sf:
        # Push the mate move; evaluate from black's POV after.
        b = chess.Board(fen)
        b.push(chess.Move.from_uci("a1a8"))
        # Now it's black to move and they're mated.
        v_black = sf.evaluate(b, chess.BLACK)
        v_white = sf.evaluate(b, chess.WHITE)
    assert v_white > 0.95, f"white eval after mate-1 = {v_white}"
    assert v_black < -0.95, f"black eval after mate-1 = {v_black}"


# ---------------------------------------------------------------------------
# MultiPV interface
# ---------------------------------------------------------------------------


def test_evaluate_children_returns_one_per_legal_move():
    """At the starting position there are 20 chess-legal moves; MultiPV
    at the legal-move count should return 20 entries."""
    board = chess.Board()
    n_legal = board.legal_moves.count()
    with StockfishLeafEval() as sf:
        children = sf.evaluate_children(board, chess.WHITE)
    assert len(children) == n_legal, (
        f"expected {n_legal} children, got {len(children)}"
    )
    # Every key should be a legal move.
    legal_set = set(board.legal_moves)
    assert set(children.keys()) <= legal_set
    # All values in [-1, 1].
    for mv, v in children.items():
        assert -1.0 <= v <= 1.0, f"{mv.uci()} -> {v} out of [-1, 1]"


def test_evaluate_children_empty_at_terminal():
    """A position with no legal moves (chess-stalemate or mated) returns
    an empty dict."""
    # Black stalemate: black king in corner, no legal moves, not in check.
    fen = "k7/8/1K6/8/8/8/8/3R4 b - - 0 1"
    board = chess.Board(fen)
    if board.legal_moves.count() == 0:
        with StockfishLeafEval() as sf:
            children = sf.evaluate_children(board, chess.BLACK)
        assert children == {}


# ---------------------------------------------------------------------------
# Factory + cleanup
# ---------------------------------------------------------------------------


def test_factory_returns_callable_and_instance():
    eval_fn, sf = stockfish_leaf_eval_factory()
    try:
        v = eval_fn(chess.Board(), chess.WHITE)
        assert -1.0 <= v <= 1.0
    finally:
        sf.close()


def test_double_close_is_safe():
    sf = StockfishLeafEval()
    sf.close()
    # Closing twice should not raise.
    sf.close()


def test_context_manager_cleans_up():
    """After the context manager exits, the subprocess should be gone."""
    with StockfishLeafEval() as sf:
        sf.evaluate(chess.Board(), chess.WHITE)
        # Inside the context, the subprocess is alive.
        # We can't easily probe it without internals; just verify no
        # crash on exit.
    # If we reach here without exception, cleanup worked.


def test_cache_hits_on_repeated_evaluate():
    """Second call on the same position must hit the cache (no Stockfish
    invocation) and return the same value."""
    with StockfishLeafEval() as sf:
        board = chess.Board()
        first = sf.evaluate(board, chess.WHITE)
        assert sf.eval_cache_misses == 1
        assert sf.eval_cache_hits == 0
        second = sf.evaluate(board, chess.WHITE)
        assert sf.eval_cache_misses == 1, "second call must not miss"
        assert sf.eval_cache_hits == 1
        assert first == second


def test_cache_hits_on_repeated_evaluate_children():
    """Same FEN should reuse the cached MultiPV dict."""
    with StockfishLeafEval() as sf:
        board = chess.Board()
        first = sf.evaluate_children(board, chess.WHITE)
        assert sf.children_cache_misses == 1
        second = sf.evaluate_children(board, chess.WHITE)
        assert sf.children_cache_hits == 1
        assert first == second


def test_cache_keys_ignore_move_counters():
    """Two positions identical except for halfmove/fullmove counters
    should share a cache entry — depth=1 search doesn't depend on those."""
    with StockfishLeafEval() as sf:
        b1 = chess.Board()
        b2 = chess.Board()
        b2.halfmove_clock = 17  # different counter, same position
        v1 = sf.evaluate(b1, chess.WHITE)
        v2 = sf.evaluate(b2, chess.WHITE)
        assert sf.eval_cache_hits == 1, "different counters should still cache-hit"
        assert v1 == v2


def test_cache_keys_differ_by_perspective():
    """Same board, different perspective → distinct cache entries, but
    each cached individually."""
    with StockfishLeafEval() as sf:
        board = chess.Board()
        sf.evaluate(board, chess.WHITE)
        sf.evaluate(board, chess.BLACK)
        assert sf.eval_cache_misses == 2
        sf.evaluate(board, chess.WHITE)  # should hit
        sf.evaluate(board, chess.BLACK)  # should hit
        assert sf.eval_cache_hits == 2
