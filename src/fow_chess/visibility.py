"""Visibility helpers for fog of war chess.

Visibility follows the mistboard canonical rule: a player sees their own occupied
squares plus the destination square of every pseudo-legal move they can make
under the current board state, regardless of whose turn it is. Castling moves
contribute the rook's original square (matching the mistboard fog castling
representation). En passant moves additionally contribute the captured pawn's
square.

Hot-path acceleration: when the optional ``fow_rust`` extension is built
(see fow_rust/), ``visible_squares`` delegates to a native Rust implementation
that's ~60x faster per call. Identical semantics — verified via
tests/test_rust_visible_squares_diff.py over 84 real games + every ply.
"""

from __future__ import annotations

import chess

try:
    import fow_rust as _fow_rust
    # Import success is NOT enough: when the compiled extension isn't built,
    # ``fow_rust`` resolves to the source dir as an empty namespace package
    # (import succeeds, zero symbols). Verify the symbol we actually call so a
    # missing/stale build falls back to pure Python instead of crashing
    # mid-call with AttributeError.
    _HAS_RUST = hasattr(_fow_rust, "visible_squares_bb")
except ImportError:
    _HAS_RUST = False


def visible_squares(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    """Return squares visible to `color` under fog of war."""
    if _HAS_RUST:
        ep = board.ep_square
        mask = _fow_rust.visible_squares_bb(
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
            color == chess.WHITE,
        )
        return chess.SquareSet(mask)
    return _visible_squares_py(board, color)


def _visible_squares_py(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    """Pure-Python implementation. Kept as a fallback when fow_rust isn't built,
    and as the differential-testing reference."""
    visible = chess.SquareSet(board.occupied_co[color])
    work = board if board.turn == color else _with_turn(board, color)

    for move in work.pseudo_legal_moves:
        if work.is_castling(move):
            rook_sq = _castling_rook_square(work, move)
            if rook_sq is not None:
                visible.add(rook_sq)
            continue
        visible.add(move.to_square)
        if work.is_en_passant(move):
            captured_sq = chess.square(
                chess.square_file(move.to_square),
                chess.square_rank(move.from_square),
            )
            visible.add(captured_sq)

    return visible


def visible_piece_map(
    board: chess.Board, color: chess.Color
) -> dict[chess.Square, chess.Piece]:
    """Return the pieces visible to `color`, keyed by square."""
    return piece_map_for_squares(board, visible_squares(board, color))


def piece_map_for_squares(
    board: chess.Board, squares: chess.SquareSet | set[chess.Square]
) -> dict[chess.Square, chess.Piece]:
    """Return pieces on a precomputed square set.

    Useful in hot paths that already computed FOW visibility and need the
    matching visible piece map without regenerating pseudo-legal moves.
    """
    return {
        square: piece
        for square, piece in board.piece_map().items()
        if square in squares
    }


def _with_turn(board: chess.Board, color: chess.Color) -> chess.Board:
    work = board.copy()
    work.turn = color
    return work


def _castling_rook_square(
    board: chess.Board, move: chess.Move
) -> chess.Square | None:
    king_from_file = chess.square_file(move.from_square)
    king_to_file = chess.square_file(move.to_square)
    rank = chess.square_rank(move.from_square)
    side_kingside = king_to_file > king_from_file

    for rook_sq in chess.SquareSet(board.castling_rights):
        if chess.square_rank(rook_sq) != rank:
            continue
        rook_file = chess.square_file(rook_sq)
        if side_kingside and rook_file > king_from_file:
            return rook_sq
        if not side_kingside and rook_file < king_from_file:
            return rook_sq
    return None
