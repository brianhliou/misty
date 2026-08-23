"""Invariant assertions for the P enumerator.

Use these in tests and in replay-validation harnesses to gate the
enumerator's correctness. Each invariant is callable as a function
returning the offending evidence on failure, so tests can produce
useful diagnostics rather than bare AssertionErrors.
"""

from __future__ import annotations

import chess

from ..observation import Observation
from .enumerator import PEnumerator


def assert_truth_in_P(
    enumerator: PEnumerator,
    truth: chess.Board,
    *,
    context: str = "",
) -> None:
    """Soundness: the actual truth must be one of the candidates in P.

    If this ever fails, the enumerator has *lost* the truth — the
    engine reasoning over P has zero chance of finding the right
    strategy.

    Args:
        enumerator: the enumerator under test.
        truth: the actual board state at the current ply.
        context: optional caller-provided string (e.g., "ply 14, white POV").
    """
    truth_fen = truth.fen()
    if truth_fen in enumerator.positions:
        return
    raise AssertionError(
        f"truth ∉ P {context}\n"
        f"  truth FEN: {truth_fen}\n"
        f"  |P| = {enumerator.size}\n"
        f"  perspective: {'white' if enumerator.perspective == chess.WHITE else 'black'}"
    )


def assert_all_consistent_with_observation(
    enumerator: PEnumerator,
    observation: Observation,
    *,
    context: str = "",
) -> None:
    """Soundness: every position in P satisfies the latest observation.

    This is what ``update_opp_move`` filters on internally; the check
    here is defensive — should never fire if the enumerator is correct.

    Note: requires the predecessor board to actually re-check
    consistent_with(). Since the enumerator doesn't store predecessors,
    we re-check the *visibility/observation* parts of the predicate
    against the candidate next-board alone (which is what
    consistent_with does in practice — the prev_board only matters for
    the own_capture_square check, and we approximate that by trusting
    the enumerator's filter).
    """
    perspective = enumerator.perspective
    failing: list[str] = []
    from ..visibility import visible_squares, piece_map_for_squares
    for fen in enumerator.positions:
        board = chess.Board(fen)
        # Visibility mask + visible pieces must match.
        v = visible_squares(board, perspective)
        if v != observation.visibility_mask:
            failing.append(f"visibility mismatch on {fen}")
            continue
        if piece_map_for_squares(board, v) != observation.visible_pieces:
            failing.append(f"visible_pieces mismatch on {fen}")
            continue
        # opp_capture_landing_square: if set, the named square must
        # hold an opp piece on this candidate board.
        if observation.opp_capture_landing_square is not None:
            piece = board.piece_at(observation.opp_capture_landing_square)
            if piece is None or piece.color == perspective:
                failing.append(
                    f"opp_capture_landing_square has no opp piece on {fen}"
                )
                continue
    if failing:
        raise AssertionError(
            f"observation-consistency invariant violated {context}: "
            f"{len(failing)} positions failed; first: {failing[0]}"
        )


def assert_cardinality_bound(
    enumerator: PEnumerator,
    upper_bound: int,
    *,
    context: str = "",
) -> None:
    """Generous performance gate: |P| should stay under a bound.

    Obscuro reports avg |P| ≈ 17K, max ~10⁶ for FoW chess. Use
    upper_bound=10⁶ as a hard ceiling — exceeding it suggests a
    soundness leak (positions being added that shouldn't be) or a
    pathological history.
    """
    if enumerator.size <= upper_bound:
        return
    raise AssertionError(
        f"|P| = {enumerator.size} exceeds bound {upper_bound} {context}"
    )
