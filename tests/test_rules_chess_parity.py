"""Parity gate for the ``Rules`` seam's chess adapter (Phase 1, slice 1).

``ChessRules`` reimplements a handful of small pure helpers inline (to avoid an
import cycle with the modules that own them and will later import ``Rules``).
This test is the contract that those reimplementations are **byte-identical** to
their canonical incumbents:

  - ``ChessRules.action_key``           == ``cfr.gt_cfr._mk``
  - ``ChessRules.canonicalize_move``    == ``p_enum.enumerator._canonicalize_castling``
  - ``ChessRules.normalize_committed_move`` == ``engine_v2._upgrade_dominated_promotion``
  - ``ChessRules.is_terminal`` / ``terminal_value`` == ``GTCFRTreeNode`` (king capture)
  - ``apply`` / ``pseudo_legal_moves`` / ``board_fen`` / ``to_move`` == direct python-chess

That equality is what makes the later per-module rewiring (routing call-sites
through ``Rules``) a true no-op. If any incumbent changes, this test fails and
the adapter must be brought back into line (NEVER the reverse).
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fow_chess.cfr.gt_cfr import _mk, root_node
from fow_chess.engine_v2 import _upgrade_dominated_promotion
from fow_chess.p_enum.enumerator import _canonicalize_castling
from fow_chess.rules import ChessRules
from fow_chess.visibility import visible_squares


@pytest.fixture
def rules() -> ChessRules:
    return ChessRules()


# --- a battery of boards spanning castling, promotions, captures, FoW ---------
def _battery() -> list[chess.Board]:
    boards = [chess.Board()]
    # Italian-ish opening with castling rights live.
    b = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
        b.push(chess.Move.from_uci(uci))
    boards.append(b)
    # Position with both kings able to castle either side.
    boards.append(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
    # Promotion-rich position.
    boards.append(chess.Board("8/P6P/8/8/8/8/p6p/8 w - - 0 1"))
    # A king-captured (terminal) position: black has no king.
    boards.append(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1").mirror())  # both kings; baseline
    return boards


def test_identity_and_board_ops(rules: ChessRules):
    assert rules.name == "chess"
    assert rules.is_first_player(chess.WHITE) is True
    assert rules.is_first_player(chess.BLACK) is False
    start = rules.start_position()
    assert rules.board_fen(start) == chess.Board().board_fen()
    assert rules.to_move(start) == chess.WHITE
    # round-trip
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    rb = rules.board_from_fen(fen)
    assert rules.board_fen(rb) == chess.Board(fen).board_fen()


def test_color_helpers_match_python_chess(rules: ChessRules):
    assert rules.first_player == chess.WHITE
    assert rules.second_player == chess.BLACK
    assert rules.opponent(chess.WHITE) == (not chess.WHITE) == chess.BLACK
    assert rules.opponent(chess.BLACK) == (not chess.BLACK) == chess.WHITE
    # opponent is an involution; first/second are each other's opponent
    assert rules.opponent(rules.opponent(chess.WHITE)) == chess.WHITE
    assert rules.opponent(rules.first_player) == rules.second_player


def test_pseudo_legal_and_apply_match_python_chess(rules: ChessRules):
    for board in _battery():
        expected = sorted(m.uci() for m in board.pseudo_legal_moves)
        got = sorted(m.uci() for m in rules.pseudo_legal_moves(board))
        assert got == expected, f"pseudo-legal mismatch on {board.fen()}"
        # apply must not mutate input and must match copy+push board_fen.
        before = board.board_fen()
        for mv in list(board.pseudo_legal_moves)[:8]:
            ref = board.copy()
            ref.push(mv)
            out = rules.apply(board, mv)
            assert rules.board_fen(out) == ref.board_fen()
            assert board.board_fen() == before, "apply mutated its input"


def test_action_key_matches_mk(rules: ChessRules):
    # Cover from/to + every promotion piece on a wide square range.
    for from_sq in range(0, 64, 7):
        for to_sq in range(0, 64, 5):
            for promo in (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                mv = chess.Move(from_sq, to_sq, promotion=promo)
                assert rules.action_key(mv) == _mk(mv), f"action_key != _mk for {mv}"


def test_canonicalize_move_matches_incumbent(rules: ChessRules):
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    # King-takes-rook castling encodings + a few ordinary moves.
    samples = ["e1h1", "e1a1", "e1e2", "e1f1", "a1a4"]
    for uci in samples:
        mv = chess.Move.from_uci(uci)
        assert rules.canonicalize_move(mv, board) == _canonicalize_castling(mv, board)
    bb = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    for uci in ["e8h8", "e8a8", "e8e7"]:
        mv = chess.Move.from_uci(uci)
        assert rules.canonicalize_move(mv, bb) == _canonicalize_castling(mv, bb)


def test_normalize_committed_move_matches_incumbent(rules: ChessRules):
    for uci in ["a7a8q", "a7a8r", "a7a8b", "a7a8n", "e2e4"]:
        mv = chess.Move.from_uci(uci)
        assert rules.normalize_committed_move(mv) == _upgrade_dominated_promotion(mv)


def test_terminal_matches_tree_node(rules: ChessRules):
    cases = [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", False, 0.0),    # both kings
        ("8/8/8/8/8/8/8/4K3 w - - 0 1", True, 1.0),       # black king gone -> +1 for white
        ("4k3/8/8/8/8/8/8/8 w - - 0 1", True, -1.0),      # white king gone -> -1 for white
    ]
    for fen, exp_term, exp_val_white in cases:
        board = chess.Board(fen)
        node = root_node(board)
        assert rules.is_terminal(board) == node.is_terminal == exp_term
        assert rules.terminal_value(board, chess.WHITE) == exp_val_white
        # cross-check against the node for both perspectives
        assert rules.terminal_value(board, chess.WHITE) == node.terminal_value(chess.WHITE)
        assert rules.terminal_value(board, chess.BLACK) == node.terminal_value(chess.BLACK)


def test_visible_squares_forwards(rules: ChessRules):
    for board in _battery():
        for color in (chess.WHITE, chess.BLACK):
            assert rules.visible_squares(board, color) == visible_squares(board, color)


def test_node_methods_route_through_rules(rules: ChessRules):
    """GTCFRTreeNode (slice 4) delegates terminal/infoset to its `rules`; a node
    built with an explicit ChessRules must match one built with the default."""
    from fow_chess.cfr.gt_cfr import GTCFRTreeNode, root_node

    for fen, persp in [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", chess.WHITE),
        ("8/8/8/8/8/8/8/4K3 w - - 0 1", chess.WHITE),     # black king gone
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", chess.BLACK),
    ]:
        board = chess.Board(fen)
        explicit = root_node(board, rules=rules)
        default = root_node(board)
        assert explicit.is_terminal == default.is_terminal
        assert explicit.terminal_value(persp) == default.terminal_value(persp)
        assert explicit.info_set_id() == default.info_set_id()
    # default field is the shared ChessRules singleton → chess behavior out of the box
    n = GTCFRTreeNode(
        truth=chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1"),
        to_move=chess.WHITE, obs_history_white=(), obs_history_black=(), depth=0,
    )
    assert n.is_terminal is True
    assert n.terminal_value(chess.WHITE) == 1.0


def test_make_belief_matches_direct_penumerator(rules: ChessRules):
    import random

    from fow_chess.p_enum import PEnumerator

    for color in (chess.WHITE, chess.BLACK):
        belief = rules.make_belief(color, rng=random.Random(0))
        direct = PEnumerator(color, rng=random.Random(0))
        assert isinstance(belief, PEnumerator)
        assert belief.perspective == direct.perspective == color
        assert belief.size == direct.size
        # Initial P membership must match the direct construction exactly.
        assert sorted(belief.iter_positions()) == sorted(direct.iter_positions())
