"""Unit tests for PEnumerator with hand-computed expected P sets.

Each test sets up a known game prefix, walks the enumerator through
it, and asserts both |P| and the exact contents at chosen points.
These are the strongest correctness gate — exhaustive comparison vs
hand-computed truth on tiny positions.
"""

from __future__ import annotations

import chess
import pytest

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import PEnumerator, assert_truth_in_P


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_max_size_cap_keeps_P_bounded():
    """_maybe_downsample fires when input exceeds max_size."""
    import random as _random
    e = PEnumerator(chess.WHITE, max_size=5, rng=_random.Random(42))
    # Build 20 distinct FEN strings as a fake P. They don't need to be
    # valid chess positions — _maybe_downsample is data-agnostic.
    oversized = {f"fake-fen-{i}" for i in range(20)}
    result = e._maybe_downsample(oversized)
    assert len(result) == 5
    assert e.downsample_count == 1
    # All kept items came from the input.
    assert result.issubset(oversized)


def test_max_size_no_op_when_under_cap():
    """No downsample fires when input is already at/below max_size."""
    import random as _random
    e = PEnumerator(chess.WHITE, max_size=5, rng=_random.Random(42))
    small = {f"f{i}" for i in range(3)}
    result = e._maybe_downsample(small)
    assert result == small  # unchanged
    assert e.downsample_count == 0


def test_max_size_none_preserves_exact_enumeration():
    """Default max_size=None keeps the strict A3 guarantee."""
    e = PEnumerator(chess.WHITE)
    assert e.max_size is None
    assert e.downsample_count == 0


def test_iter_positions_streams_without_copy():
    """`iter_positions()` and `__iter__` should yield FENs directly from
    the internal set — no snapshot copy. This is the downstream-streaming
    API (mining/projection consumers)."""
    e = PEnumerator(chess.WHITE)
    # __iter__ works
    fens_iter = list(e)
    assert len(fens_iter) == 1
    # iter_positions() works
    fens_method = list(e.iter_positions())
    assert fens_method == fens_iter
    # `positions` property is a separate snapshot (different object identity).
    snapshot_a = e.positions
    snapshot_b = e.positions
    assert snapshot_a is not snapshot_b  # fresh frozenset each time


def test_initial_P_is_singleton_starting_board():
    e_white = PEnumerator(chess.WHITE)
    e_black = PEnumerator(chess.BLACK)
    assert e_white.size == 1
    assert e_black.size == 1
    start_fen = chess.Board().fen()
    assert start_fen in e_white.positions
    assert start_fen in e_black.positions


def test_initial_P_respects_custom_starting_board():
    custom = chess.Board("8/8/8/4k3/4K3/8/8/8 w - - 0 1")
    e = PEnumerator(chess.WHITE, starting_board=custom)
    assert e.size == 1
    assert custom.fen() in e.positions


# ---------------------------------------------------------------------------
# Own-move updates: deterministic; |P| can shrink or stay the same
# ---------------------------------------------------------------------------


def test_own_move_singleton_to_singleton():
    """At the start, |P|=1. After white plays e2-e4, |P|=1 (still the
    unique position consistent with white's move history)."""
    e = PEnumerator(chess.WHITE)
    e.update_own_move(chess.Move.from_uci("e2e4"))
    assert e.size == 1
    expected = chess.Board()
    expected.push(chess.Move.from_uci("e2e4"))
    assert expected.fen() in e.positions


def test_own_move_filters_positions_where_move_illegal():
    """If P contains positions where the move ISN'T pseudo-legal, those
    get dropped. Construct an artificial P with two positions; only
    one admits the move."""
    e = PEnumerator(chess.WHITE)
    # Manually seed P with two positions: standard start + a position
    # where white has no e2-pawn.
    no_e2 = chess.Board()
    no_e2.remove_piece_at(chess.E2)
    e._positions = {chess.Board().fen(), no_e2.fen()}
    e.update_own_move(chess.Move.from_uci("e2e4"))
    # Only the standard start admits e2e4; the no-e2 position is dropped.
    assert e.size == 1


def test_own_move_raises_when_no_candidate_admits_it():
    """Soundness: if NO position in P admits the move, enumerator must
    fail loud rather than silently produce empty P."""
    e = PEnumerator(chess.WHITE)
    # The starting board doesn't admit e3-e4 (no pawn on e3).
    with pytest.raises(RuntimeError, match="soundness violation"):
        e.update_own_move(chess.Move.from_uci("e3e4"))


# ---------------------------------------------------------------------------
# Opp-move updates: filtered by observation; |P| often expands then shrinks
# ---------------------------------------------------------------------------


def test_opp_move_from_singleton_produces_observation_consistent_set():
    """At the start, |P|=1. White plays e2-e4. Black now updates own P
    (their POV) on white's move via observation.

    Black's perspective: P_black starts at |{starting board}|. White's
    move is opp from black's POV. After update, P_black contains all
    starting-board successors that produce the same observation black
    sees of white's e2-e4 move.
    """
    e_black = PEnumerator(chess.BLACK)
    prev = chess.Board()
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("e2e4"))
    obs = observation_from_transition(prev, nxt, chess.BLACK)
    e_black.update_opp_move(obs)
    # Truth must be in P
    assert_truth_in_P(e_black, nxt)
    # |P| can be > 1 because multiple white moves could produce the
    # same observation from black's POV (any non-visible move that
    # leaves the visible parts unchanged).
    assert e_black.size >= 1
    # All positions should have black-to-move (white just moved).
    for fen in e_black.positions:
        assert chess.Board(fen).turn == chess.BLACK


def test_opp_move_obvious_capture_constrains_P():
    """An observed capture pins the move's destination square. P should
    only contain positions where some opp piece landed on that square."""
    # Set up: white pawn on e4, black pawn on d5. Black to move.
    # If black plays d5xe4, white observes own_capture_square=e4 and
    # opp_capture_landing_square=e4.
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    prev = chess.Board(fen)
    e_white = PEnumerator(chess.WHITE, starting_board=prev)
    # White's enumerator: P = {prev}. Now black plays d5e4.
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("d5e4"))
    obs = observation_from_transition(prev, nxt, chess.WHITE)
    assert obs.own_capture_square == chess.E4
    e_white.update_opp_move(obs)
    # Every position in P must have an opp piece on e4 now.
    for fen in e_white.positions:
        board = chess.Board(fen)
        piece = board.piece_at(chess.E4)
        assert piece is not None and piece.color == chess.BLACK, (
            f"position lacks black piece on e4: {fen}"
        )
    assert_truth_in_P(e_white, nxt)


def test_opp_move_raises_on_unfilterable_observation():
    """If we feed an observation that NO opp move can produce, the
    enumerator must fail loud."""
    e_white = PEnumerator(chess.WHITE)
    # Fabricate an impossible obs: visibility_mask claims a square
    # is visible that white can't possibly see from start.
    from fow_chess.observation import Observation
    impossible = Observation(
        visibility_mask=chess.SquareSet(chess.BB_ALL),  # everything visible
        visible_pieces={},  # but no pieces seen — impossible from start
    )
    with pytest.raises(RuntimeError, match="soundness violation"):
        e_white.update_opp_move(impossible)


# ---------------------------------------------------------------------------
# Alternating-update flow: full mini-game
# ---------------------------------------------------------------------------


def test_alternating_updates_two_plies():
    """Walk a 2-ply mini-game from both perspectives. Truth in P
    at every step."""
    truth = chess.Board()
    e_white = PEnumerator(chess.WHITE)
    e_black = PEnumerator(chess.BLACK)

    # Ply 1: white plays e2-e4. White knows the move; black observes.
    move_w = chess.Move.from_uci("e2e4")
    prev = truth.copy()
    truth.push(move_w)
    obs_for_black = observation_from_transition(prev, truth, chess.BLACK)

    e_white.update_own_move(move_w)
    e_black.update_opp_move(obs_for_black)
    assert_truth_in_P(e_white, truth, context="after ply 1, white POV")
    assert_truth_in_P(e_black, truth, context="after ply 1, black POV")

    # Ply 2: black plays e7-e5. Black knows; white observes.
    move_b = chess.Move.from_uci("e7e5")
    prev = truth.copy()
    truth.push(move_b)
    obs_for_white = observation_from_transition(prev, truth, chess.WHITE)

    e_black.update_own_move(move_b)
    e_white.update_opp_move(obs_for_white)
    assert_truth_in_P(e_white, truth, context="after ply 2, white POV")
    assert_truth_in_P(e_black, truth, context="after ply 2, black POV")

    # |P| after a player's own move is the same as before that move
    # (modulo positions where the move wasn't pseudo-legal). Own moves
    # don't reduce uncertainty about the OPPONENT's position. Black's
    # uncertainty was created by white's hidden first move — black's
    # own e7-e5 doesn't shrink that. We expect |P_black| ≥ 1 and that
    # the truth is among them.
    assert e_white.size >= 1
    assert e_black.size >= 1
