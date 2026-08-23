"""Per-ply observation a perspective player makes in fog of war chess."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .visibility import piece_map_for_squares, visible_squares

try:
    import fow_rust as _fow_rust
    # See visibility.py: a namespace-package shadow (unbuilt extension) imports
    # cleanly but has no symbols. Probe the one we call.
    _HAS_RUST = hasattr(_fow_rust, "observation_from_transition_bb")
except ImportError:
    _HAS_RUST = False


@dataclass(frozen=True)
class GameOver:
    """Terminal signal observed when the game ends."""

    winner: chess.Color | None
    reason: str


@dataclass(frozen=True)
class Observation:
    """What the perspective player learns immediately after the opponent moves."""

    visibility_mask: chess.SquareSet
    visible_pieces: dict[chess.Square, chess.Piece] = field(default_factory=dict)
    own_capture_square: chess.Square | None = None
    # When an opponent normally captures one of our pieces, we may not know the
    # capturer's type, but we do know an opponent piece landed on the captured
    # square. En passant is the exception; then the captured square is not the
    # landing square, so this stays None unless the landing square is otherwise
    # inferable by visibility.
    opp_capture_landing_square: chess.Square | None = None
    game_over: GameOver | None = None


def consistent_with(
    next_board: chess.Board,
    prev_board: chess.Board,
    obs: Observation,
    perspective: chess.Color,
) -> bool:
    """True iff `next_board` could have produced `obs` for `perspective` from `prev_board`."""
    visible = visible_squares(next_board, perspective)
    if visible != obs.visibility_mask:
        return False
    if piece_map_for_squares(next_board, visible) != obs.visible_pieces:
        return False

    own_before = {
        sq for sq, p in prev_board.piece_map().items() if p.color == perspective
    }
    own_after = {
        sq for sq, p in next_board.piece_map().items() if p.color == perspective
    }
    captures = own_before - own_after

    if obs.own_capture_square is None:
        if captures:
            return False
    elif captures != {obs.own_capture_square}:
        return False

    if obs.opp_capture_landing_square is not None:
        landing_piece = next_board.piece_at(obs.opp_capture_landing_square)
        if landing_piece is None or landing_piece.color == perspective:
            return False

    return True


def _obs_from_rust_tuple(
    mask: int,
    rust_pieces,
    captured,
    opp_capture_landing_square,
    game_over_winner,
    game_over_reason: str,
) -> Observation:
    """Build an Observation from the 6-tuple returned by the Rust
    observation_from_transition[_both]_bb entry points. Shared so the single-
    and both-perspective paths construct identical Observation objects."""
    visible_pieces = {
        sq: chess.Piece(role_int, chess.WHITE if color_white else chess.BLACK)
        for sq, role_int, color_white in rust_pieces
    }
    game_over: GameOver | None = None
    if game_over_reason:
        winner: chess.Color | None
        if game_over_winner is None:
            winner = None
        else:
            winner = chess.WHITE if game_over_winner else chess.BLACK
        game_over = GameOver(winner=winner, reason=game_over_reason)
    return Observation(
        visibility_mask=chess.SquareSet(mask),
        visible_pieces=visible_pieces,
        own_capture_square=captured,
        opp_capture_landing_square=opp_capture_landing_square,
        game_over=game_over,
    )


def observation_from_transition_both(
    prev_board: chess.Board,
    next_board: chess.Board,
) -> tuple[Observation, Observation]:
    """Both perspectives' Observations from one transition, as
    ``(white_obs, black_obs)``. Equivalent to calling
    ``observation_from_transition`` for WHITE and BLACK, but extracts the board
    bitboards once and (in Rust) builds the next-board setup once — the hot path
    in ``expand_leaf``, which needs both perspectives per child."""
    if _HAS_RUST and hasattr(_fow_rust, "observation_from_transition_both_bb"):
        next_ep_idx = next_board.ep_square if next_board.ep_square is not None else 64
        white_t, black_t = _fow_rust.observation_from_transition_both_bb(
            prev_board.occupied_co[chess.WHITE],
            prev_board.occupied_co[chess.BLACK],
            prev_board.kings,
            next_board.pawns, next_board.knights, next_board.bishops,
            next_board.rooks, next_board.queens, next_board.kings,
            next_board.occupied_co[chess.WHITE],
            next_board.occupied_co[chess.BLACK],
            next_board.castling_rights,
            next_ep_idx,
        )
        return _obs_from_rust_tuple(*white_t), _obs_from_rust_tuple(*black_t)
    return (
        observation_from_transition(prev_board, next_board, chess.WHITE),
        observation_from_transition(prev_board, next_board, chess.BLACK),
    )


def observation_from_transition(
    prev_board: chess.Board,
    next_board: chess.Board,
    perspective: chess.Color,
) -> Observation:
    """Build the Observation `perspective` makes from the canonical transition `prev_board` -> `next_board`."""
    if _HAS_RUST:
        # RP10: native Rust hot-path (~10× per-call). 25% of pick_move
        # wall time per the 2026-05-25 profile; this single call is the
        # biggest single function-port win in the engine. The Python
        # fallback below is byte-identical (60-case diff test pinned).
        next_ep_idx = next_board.ep_square if next_board.ep_square is not None else 64
        (
            mask,
            rust_pieces,
            captured,
            opp_capture_landing_square,
            game_over_winner,
            game_over_reason,
        ) = _fow_rust.observation_from_transition_bb(
            prev_board.occupied_co[chess.WHITE],
            prev_board.occupied_co[chess.BLACK],
            prev_board.kings,
            next_board.pawns, next_board.knights, next_board.bishops,
            next_board.rooks, next_board.queens, next_board.kings,
            next_board.occupied_co[chess.WHITE],
            next_board.occupied_co[chess.BLACK],
            next_board.castling_rights,
            next_ep_idx,
            perspective == chess.WHITE,
        )
        return _obs_from_rust_tuple(
            mask, rust_pieces, captured, opp_capture_landing_square,
            game_over_winner, game_over_reason,
        )

    # Pure-Python fallback. Kept structurally identical to the original
    # so the Rust diff tests still pin behavior.
    own_before = {
        sq for sq, p in prev_board.piece_map().items() if p.color == perspective
    }
    own_after = {
        sq for sq, p in next_board.piece_map().items() if p.color == perspective
    }
    captures = own_before - own_after
    captured = next(iter(captures), None)
    opp_capture_landing_square: chess.Square | None = None
    if captured is not None:
        landing_piece = next_board.piece_at(captured)
        if landing_piece is not None and landing_piece.color != perspective:
            opp_capture_landing_square = captured

    game_over: GameOver | None = None
    if (
        prev_board.king(perspective) is not None
        and next_board.king(perspective) is None
    ):
        game_over = GameOver(winner=not perspective, reason="king-captured")

    visible = visible_squares(next_board, perspective)
    return Observation(
        visibility_mask=visible,
        visible_pieces=piece_map_for_squares(next_board, visible),
        own_capture_square=captured,
        opp_capture_landing_square=opp_capture_landing_square,
        game_over=game_over,
    )
