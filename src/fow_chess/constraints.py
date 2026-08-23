"""Piece-count and square-color constraints for opponent positions.

These helpers describe the structural constraints any opponent position must
satisfy: piece counts bounded by the standard starting set (minus any
observed captures), bishop pairs constrained by square color, etc.

They were lifted from the particle-filter `belief.py` in Phase A0 of the
Obscuro replication. Their next consumer is the `P` enumerator (Phase A3):
the predicate that decides "given observation history H, is position p
consistent?" prunes using these constraints to avoid enumerating positions
with impossible piece counts.

Promotion edge case: opp pawn -> opp queen (or other) increments their
non-pawn count while decrementing pawn count. Count constraints must be
promotion-aware: non-pawn excess is allowed only when compensated by
missing pawns.
"""

from __future__ import annotations

import chess


# Standard starting piece counts (per side). Used as a hard upper bound on
# every candidate opponent position's piece count after observed captures
# are subtracted.
STANDARD_OPP_COUNTS: dict[chess.PieceType, int] = {
    chess.PAWN: 8,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 2,
    chess.QUEEN: 1,
    chess.KING: 1,
}


def opp_piece_counts(
    board: chess.Board, perspective: chess.Color
) -> dict[chess.PieceType, int]:
    """Count opponent pieces on `board` by piece type."""
    counts: dict[chess.PieceType, int] = {}
    opp = not perspective
    for piece in board.piece_map().values():
        if piece.color == opp:
            counts[piece.piece_type] = counts.get(piece.piece_type, 0) + 1
    return counts


def is_light_square(square: chess.Square) -> bool:
    """True if `square` is a light square (a1 = dark; alternates standard)."""
    return (chess.square_file(square) + chess.square_rank(square)) % 2 == 1


def opp_bishop_color_counts(
    board: chess.Board, perspective: chess.Color
) -> dict[bool, int]:
    """Count opp bishops by square color (True=light, False=dark)."""
    counts = {True: 0, False: 0}
    opp = not perspective
    for sq, piece in board.piece_map().items():
        if piece.color == opp and piece.piece_type == chess.BISHOP:
            counts[is_light_square(sq)] += 1
    return counts
