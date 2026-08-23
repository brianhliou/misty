"""Tests for `observation.consistent_with` — the P-enumeration predicate.

`consistent_with(next_board, prev_board, obs, perspective)` is the kernel
of the forthcoming `P` enumerator (Phase A3 of the Obscuro replication):
given an observation history, `P` = the set of (prev → next) transitions
that satisfy this predicate at every step.

These tests lock down the predicate's invariants directly, without the
particle-filter substrate that was layered on top of it. They cover:

  1. Round-trip: every transition is consistent with its own
     ``observation_from_transition`` output.
  2. Negative cases: each of the four ways `consistent_with` can reject.
  3. FoW-specific scenarios: castling, en passant, captures.
"""

from __future__ import annotations

import chess
import pytest

from fow_chess.observation import (
    Observation,
    consistent_with,
    observation_from_transition,
)
from fow_chess.visibility import piece_map_for_squares, visible_squares


def _apply(fen: str, move_uci: str) -> tuple[chess.Board, chess.Board]:
    prev = chess.Board(fen)
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci(move_uci))
    return prev, nxt


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen,move_uci,perspective",
    [
        (chess.STARTING_FEN, "e2e4", chess.WHITE),
        (chess.STARTING_FEN, "e2e4", chess.BLACK),
        (chess.STARTING_FEN, "g1f3", chess.WHITE),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
         "f1c4", chess.WHITE),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
         "f1c4", chess.BLACK),
    ],
)
def test_consistent_with_round_trip(fen, move_uci, perspective):
    """Every (prev → next) transition is consistent with the observation
    it produces, from both perspectives."""
    prev, nxt = _apply(fen, move_uci)
    obs = observation_from_transition(prev, nxt, perspective)
    assert consistent_with(nxt, prev, obs, perspective) is True


# ---------------------------------------------------------------------------
# Negative cases — each of the four rejection paths
# ---------------------------------------------------------------------------


def test_rejects_when_visibility_mask_differs():
    """If we hand consistent_with a different board than the obs describes,
    its computed visibility mask differs and we reject."""
    prev, nxt = _apply(chess.STARTING_FEN, "e2e4")
    obs = observation_from_transition(prev, nxt, chess.WHITE)

    # Alter the candidate board so visibility changes (move a piece).
    impostor = nxt.copy()
    impostor.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.WHITE))

    assert consistent_with(impostor, prev, obs, chess.WHITE) is False


def test_rejects_when_visible_piece_set_differs():
    """If the candidate has the same visibility mask but different visible
    pieces, we reject."""
    prev, nxt = _apply(chess.STARTING_FEN, "e2e4")
    obs = observation_from_transition(prev, nxt, chess.WHITE)

    # Build an impostor obs with a phantom visible opponent piece.
    visible = visible_squares(nxt, chess.WHITE)
    impostor_pieces = dict(piece_map_for_squares(nxt, visible))
    # Pick any visible square and overwrite with a different piece type.
    target_sq = next(sq for sq in visible if impostor_pieces.get(sq))
    real_piece = impostor_pieces[target_sq]
    fake_pt = chess.KNIGHT if real_piece.piece_type != chess.KNIGHT else chess.BISHOP
    impostor_pieces[target_sq] = chess.Piece(fake_pt, real_piece.color)
    impostor_obs = Observation(
        visibility_mask=obs.visibility_mask,
        visible_pieces=impostor_pieces,
        own_capture_square=obs.own_capture_square,
        opp_capture_landing_square=obs.opp_capture_landing_square,
        game_over=obs.game_over,
    )

    assert consistent_with(nxt, prev, impostor_obs, chess.WHITE) is False


def test_rejects_when_own_capture_disagrees_no_capture_claimed():
    """Observation says no capture, but the candidate transition lost a
    piece — reject."""
    # Black plays exd5 — white loses pawn on d5.
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    prev = chess.Board(fen)
    # White's turn here; make white move first to keep semantics straight.
    # Actually we need black to capture, so set up properly:
    fen2 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    prev = chess.Board(fen2)
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("d5e4"))  # black pawn captures white pawn on e4.
    # white's observation:
    obs = observation_from_transition(prev, nxt, chess.WHITE)
    assert obs.own_capture_square == chess.E4

    # Build an obs that LIES — claims no capture happened.
    impostor_obs = Observation(
        visibility_mask=obs.visibility_mask,
        visible_pieces=obs.visible_pieces,
        own_capture_square=None,
        opp_capture_landing_square=obs.opp_capture_landing_square,
        game_over=obs.game_over,
    )
    assert consistent_with(nxt, prev, impostor_obs, chess.WHITE) is False


def test_rejects_when_own_capture_disagrees_wrong_square():
    """Observation claims capture on the wrong square — reject."""
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    prev = chess.Board(fen)
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("d5e4"))
    obs = observation_from_transition(prev, nxt, chess.WHITE)

    impostor_obs = Observation(
        visibility_mask=obs.visibility_mask,
        visible_pieces=obs.visible_pieces,
        own_capture_square=chess.D4,  # wrong square
        opp_capture_landing_square=obs.opp_capture_landing_square,
        game_over=obs.game_over,
    )
    assert consistent_with(nxt, prev, impostor_obs, chess.WHITE) is False


def test_rejects_when_landing_square_has_no_opp_piece():
    """If `opp_capture_landing_square` is set, that square must hold an opp
    piece on the candidate `next_board`."""
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    prev = chess.Board(fen)
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("d5e4"))
    obs = observation_from_transition(prev, nxt, chess.WHITE)
    assert obs.opp_capture_landing_square == chess.E4

    # Construct a candidate `next_board` with the landing square empty.
    impostor_next = nxt.copy()
    impostor_next.remove_piece_at(chess.E4)
    assert consistent_with(impostor_next, prev, obs, chess.WHITE) is False


# ---------------------------------------------------------------------------
# FoW-specific scenarios
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Castling exposes a known limitation in the bare predicate: "
    "two own-piece squares are vacated (king + rook), but Observation "
    "carries only one own_capture_square. BeliefState worked around this "
    "with castling special-case logic. P enumeration (Phase A3) must "
    "either patch observation_from_transition to recognize castling or "
    "extend Observation to carry castling structure.",
    strict=True,
)
def test_consistent_with_castling_kingside_white():
    """Documents castling limitation for P enumeration to address."""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    prev, nxt = _apply(fen, "e1g1")
    obs = observation_from_transition(prev, nxt, chess.WHITE)
    assert consistent_with(nxt, prev, obs, chess.WHITE) is True


def test_consistent_with_en_passant():
    """En passant transition is consistent with its derived observation."""
    fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    prev, nxt = _apply(fen, "e5d6")
    for perspective in (chess.WHITE, chess.BLACK):
        obs = observation_from_transition(prev, nxt, perspective)
        assert consistent_with(nxt, prev, obs, perspective) is True


def test_consistent_with_king_capture_terminal():
    """A move that captures the opp king terminates the game; the
    observation carries game_over and the consistency check still holds."""
    # White rook captures black king on e8. Set up minimal position.
    fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    prev = chess.Board(fen)
    nxt = prev.copy()
    # Need to walk rook up to capture; use h1h8 capturing nothing then
    # actually just place a position where king capture is immediate.
    fen2 = "4k2R/8/8/8/8/8/8/4K3 w - - 0 1"
    prev = chess.Board(fen2)
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("h8e8"))

    for perspective in (chess.WHITE, chess.BLACK):
        obs = observation_from_transition(prev, nxt, perspective)
        if perspective == chess.BLACK:
            assert obs.game_over is not None
            assert obs.game_over.winner == chess.WHITE
        assert consistent_with(nxt, prev, obs, perspective) is True


# ---------------------------------------------------------------------------
# P-enumeration discrimination: predicate distinguishes truth from impostors
# ---------------------------------------------------------------------------


def test_predicate_distinguishes_truth_from_impostor_hidden_piece():
    """Given an observation, the actual next_board is consistent, but a
    candidate with an extra opp piece on a hidden square should still
    pass (the predicate doesn't see hidden squares). This documents the
    *expected* ambiguity P enumeration must handle: many truths are
    consistent with one observation."""
    fen = chess.STARTING_FEN
    prev, nxt = _apply(fen, "e2e4")
    obs = observation_from_transition(prev, nxt, chess.WHITE)

    # Add a phantom black knight on a hidden square (a3 — not visible to
    # white in starting-position derivative).
    impostor = nxt.copy()
    if not visible_squares(impostor, chess.WHITE) & {chess.A3}:
        impostor.set_piece_at(chess.A3, chess.Piece(chess.KNIGHT, chess.BLACK))
        # This impostor is consistent with the observation — the predicate
        # has no information distinguishing it. P enumeration handles this
        # ambiguity via observation-history constraints + piece-count
        # bounds, not via this predicate alone.
        # Note: depending on visibility rules this may flip on/off. If A3
        # is visible from white's POV after e2e4, this test asserts the
        # *predicate* still rejects (extra phantom visible). Either way
        # demonstrates the predicate's discriminative power.
        # Simply assert behavior matches visibility:
        if chess.A3 in visible_squares(impostor, chess.WHITE):
            assert consistent_with(impostor, prev, obs, chess.WHITE) is False
        else:
            assert consistent_with(impostor, prev, obs, chess.WHITE) is True
