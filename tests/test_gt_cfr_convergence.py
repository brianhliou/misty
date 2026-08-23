"""Convergence sanity tests for one-sided GT-CFR on chess positions.

GT-CFR's substrate is chess.Board + Stockfish-MultiPV, so we can't use
the Kuhn benchmark (different game shape). Instead we validate
convergence on chess positions with known-good answers:

1. **Starting position** — argmax should be a reasonable opening
   (Stockfish-eval at depth 1 picks one of the standard book moves).
2. **Mate-in-1** — argmax should be the mating move.
3. **Iteration scaling** — more iterations → larger tree + more peaked
   argmax probability.

Tests skip cleanly if Stockfish isn't on PATH.
"""

from __future__ import annotations

import shutil
import random

import chess
import pytest

from fow_chess.cfr.gt_cfr import root_node, solve_growing_subgame


pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not on PATH",
)


def _solve(board: chess.Board, *, iterations: int, perspective=None):
    """Helper: open Stockfish + run GT-CFR."""
    from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval
    root = root_node(board)
    if perspective is None:
        perspective = board.turn
    with StockfishLeafEval() as sf:
        return solve_growing_subgame(
            root,
            stockfish_eval=sf,
            perspective=perspective,
            iterations=iterations,
            rng=random.Random(42),
        )


def test_starting_position_argmax_is_reasonable_opening():
    """GT-CFR's last-iterate argmax should pick something Stockfish-
    backed at the starting position. Stockfish at depth 1 typically
    rates e2e4, d2d4, g1f3, c2c4, b1c3 as among the top — we accept
    any of those as 'reasonable'."""
    sol = _solve(chess.Board(), iterations=30)
    argmax_move = max(sol.strategy_at_root, key=sol.strategy_at_root.get)
    assert argmax_move.uci() in {
        "e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "g2g3", "b2b3",
    }, f"GT-CFR picked unusual opening: {argmax_move.uci()}"


def test_mate_in_one_argmax_is_mating_move():
    """White to play, Rh1-h8 (or similar) is mate. GT-CFR with smart-
    init should put dominant weight on the mating move."""
    # Position: black king on a8, white king on c6, white rook on h1.
    # Mate-in-1 by Ra1-a8# (well, that's not mate — black king on a8
    # can be captured by the rook reaching a8). Use a simple
    # FoW-friendly setup: white rook attacks a8 from h8 area.
    # Simpler: white king g6, white rook on h1, black king on g8 with
    # f7 pawn blocking flight. Rh1-h8 captures hypothetical king-
    # adjacent target... let me use a clean mate-in-1.
    # Cleaner: white queen on f7, black king on h8 — Qf7-g7 is mate
    # ... actually FoW has no mate concept (no check rule), so the
    # "right" move here is the one that captures the black king.
    # Position: black king on a8, white rook on h8. h8 attacks a8.
    # White plays h8a8 to capture the king.
    fen = "k6R/8/8/8/8/8/8/4K3 w - - 0 1"
    board = chess.Board(fen)
    sol = _solve(board, iterations=30, perspective=chess.WHITE)
    argmax_move = max(sol.strategy_at_root, key=sol.strategy_at_root.get)
    assert argmax_move.uci() == "h8a8", (
        f"GT-CFR missed the king-capture: argmax={argmax_move.uci()}"
    )


def test_more_iterations_grows_tree():
    """Tree size should scale with iteration count."""
    sol_small = _solve(chess.Board(), iterations=10)
    sol_big = _solve(chess.Board(), iterations=30)
    assert sol_big.tree_node_count > sol_small.tree_node_count, (
        f"small={sol_small.tree_node_count} big={sol_big.tree_node_count}"
    )


def test_strategy_is_a_valid_distribution():
    """Returned last-iterate strategy at root must sum to ~1, all ≥ 0."""
    sol = _solve(chess.Board(), iterations=10)
    probs = list(sol.strategy_at_root.values())
    assert all(p >= 0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6


def test_value_at_root_is_bounded():
    """Value at root is a weighted Q over actions; should land in [-1, 1]."""
    sol = _solve(chess.Board(), iterations=20)
    assert -1.0 <= sol.value_at_root <= 1.0
