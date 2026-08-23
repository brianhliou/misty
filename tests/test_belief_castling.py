"""Belief filter handles castling — both kingside and queenside.

Concrete shape of the bug fixed in this commit:
  - q0 annotation at ply 13: "white castled, but none of black's belief
    worlds is white castled."
  - Root cause: BeliefHardFacts.with_piece_fact_moved_by only dropped the
    fact at the king's from-square. If black had any hard fact about the
    rook's square (a1/h1/a8/h8), matches_hard_transition would then check
    "is the rook still at h1?" — find it gone — and reject the particle.
  - Fix: when opp_move is a castling move, also drop the rook's hard fact.
"""

from dataclasses import replace

import chess

from fow_chess.belief import BeliefHardFacts
from fow_chess.observation import Observation, observation_from_transition


def _facts_with(rook_sq: chess.Square, piece: chess.Piece) -> BeliefHardFacts:
    obs = Observation(
        visibility_mask=frozenset(),
        visible_pieces={},
        own_capture_square=None,
        opp_capture_landing_square=None,
        game_over=None,
    )
    return BeliefHardFacts(
        observation=obs,
        perspective=chess.BLACK,
        opp_remaining_counts={},
        opp_bishop_colors_remaining={True: 1, False: 1},
        hard_opp_piece_facts={rook_sq: piece},
    )


def test_kingside_castling_drops_rook_fact():
    """Black has a hard fact white rook on h1. White castles kingside."""
    prev = chess.Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPBPPP/RNBQK2R w KQkq - 0 1")
    move = chess.Move.from_uci("e1g1")
    assert prev.is_castling(move), "test setup: move must be castling"
    facts = _facts_with(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
    new_facts = facts.with_piece_fact_moved_by(prev, move)
    assert chess.H1 not in new_facts.hard_opp_piece_facts, (
        "rook fact on h1 must be dropped when white castles kingside"
    )


def test_queenside_castling_drops_rook_fact():
    """Same but queenside: rook fact on a1."""
    prev = chess.Board("rnbqkbnr/pppppppp/8/8/8/2N5/PPPQPPPP/R3KBNR w KQkq - 0 1")
    move = chess.Move.from_uci("e1c1")
    assert prev.is_castling(move), "test setup: move must be castling"
    facts = _facts_with(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    new_facts = facts.with_piece_fact_moved_by(prev, move)
    assert chess.A1 not in new_facts.hard_opp_piece_facts


def test_castling_with_no_rook_fact_is_noop():
    """If black has no fact about either rook, castling shouldn't disturb anything."""
    prev = chess.Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPBPPP/RNBQK2R w KQkq - 0 1")
    move = chess.Move.from_uci("e1g1")
    obs = Observation(
        visibility_mask=frozenset(),
        visible_pieces={},
        own_capture_square=None,
        opp_capture_landing_square=None,
        game_over=None,
    )
    facts = BeliefHardFacts(
        observation=obs,
        perspective=chess.BLACK,
        opp_remaining_counts={},
        opp_bishop_colors_remaining={True: 1, False: 1},
    )
    new_facts = facts.with_piece_fact_moved_by(prev, move)
    assert new_facts.hard_opp_piece_facts == {}


def test_castled_particle_survives_matches_hard_transition():
    """End-to-end: with hard rook fact on h1, castling now passes the check."""
    prev = chess.Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPBPPP/RNBQK2R w KQkq - 0 1")
    move = chess.Move.from_uci("e1g1")
    next_board = prev.copy()
    next_board.push(move)
    # Black would see nothing on the 1st rank (no observation visibility on
    # white's back rank from the starting black position), so build a no-op
    # observation that doesn't visualize the castled squares.
    obs = observation_from_transition(prev, next_board, chess.BLACK)
    facts = BeliefHardFacts(
        observation=obs,
        perspective=chess.BLACK,
        opp_remaining_counts={
            chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
            chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1,
        },
        opp_bishop_colors_remaining={True: 1, False: 1},
        hard_opp_piece_facts={chess.H1: chess.Piece(chess.ROOK, chess.WHITE)},
    )
    new_facts = facts.with_piece_fact_moved_by(prev, move)
    assert new_facts.matches_hard_transition(next_board, prev), (
        "after dropping the rook fact on castling, the castled particle "
        "should pass matches_hard_transition. Without the fix, this returns "
        "False because the check reads h1 and finds no rook there."
    )


def test_non_castle_king_move_only_drops_king_fact():
    """If the king moves WITHOUT castling, the rook fact must NOT be dropped."""
    prev = chess.Board("rnbqkbnr/pppppppp/8/8/8/4P3/PPPPKPPP/RNBQ1BNR w kq - 0 1")
    move = chess.Move.from_uci("e2e1")  # king step back, not castling
    assert not prev.is_castling(move)
    facts = BeliefHardFacts(
        observation=Observation(
            visibility_mask=frozenset(), visible_pieces={},
            own_capture_square=None, opp_capture_landing_square=None, game_over=None,
        ),
        perspective=chess.BLACK,
        opp_remaining_counts={},
        opp_bishop_colors_remaining={True: 1, False: 1},
        hard_opp_piece_facts={
            chess.H1: chess.Piece(chess.ROOK, chess.WHITE),
            chess.A1: chess.Piece(chess.ROOK, chess.WHITE),
        },
    )
    new_facts = facts.with_piece_fact_moved_by(prev, move)
    assert chess.H1 in new_facts.hard_opp_piece_facts
    assert chess.A1 in new_facts.hard_opp_piece_facts
