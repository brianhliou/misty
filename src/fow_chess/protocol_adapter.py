"""Adapter between the public engine protocol and fow_chess internals.

The protocol (`fow_chess.engine_protocol`) is a pure wire-format module
with no engine dependencies. This adapter bridges it to the engine's
internal types:

  - protocol `EngineObservation` ⇄ internal `Observation`
  - protocol `Move` ⇄ `chess.Move`
  - protocol `'white' | 'black'` ⇄ `chess.WHITE | chess.BLACK`
  - protocol `PieceLetter` ⇄ `chess.PieceType` / `chess.Piece`

Used by the Python live-move worker to consume `EngineTurnRequest`
from the server. Also used by any future per-game engine driver that
speaks the protocol directly.
"""

from __future__ import annotations

from typing import Optional

import chess

from . import engine_protocol as proto
from .observation import GameOver, Observation
from .selfplay import PerspectiveView


_PIECE_LETTER_TO_TYPE = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}
_PIECE_TYPE_TO_LETTER: dict[int, proto.PieceLetter] = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}
_PROMOTION_ROLE_TO_TYPE: dict[proto.PromotionRole, chess.PieceType] = {
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
}
_PROMOTION_TYPE_TO_ROLE: dict[int, proto.PromotionRole] = {
    chess.QUEEN: "queen",
    chess.ROOK: "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
}


def color_from_protocol(c: proto.Color) -> chess.Color:
    return chess.WHITE if c == "white" else chess.BLACK


def color_to_protocol(c: chess.Color) -> proto.Color:
    return "white" if c == chess.WHITE else "black"


def move_from_protocol(m: proto.Move) -> chess.Move:
    promo = _PROMOTION_ROLE_TO_TYPE[m.promotion] if m.promotion else None
    return chess.Move(m.from_square, m.to_square, promotion=promo)


def move_to_protocol(m: chess.Move) -> proto.Move:
    promo: Optional[proto.PromotionRole] = (
        _PROMOTION_TYPE_TO_ROLE[m.promotion] if m.promotion else None
    )
    return proto.Move(from_square=m.from_square, to_square=m.to_square, promotion=promo)


def observation_from_protocol(
    obs: proto.EngineObservation,
) -> Observation:
    """Convert protocol observation → internal Observation.

    The protocol's `own_move` is ignored here (used separately by the
    replay loop to know which move was made). `game_over` is also
    ignored at the per-observation level — the engine reads game-over
    from terminal state of the board it constructs from observations.
    """
    visibility_mask = chess.SquareSet(obs.visibility_mask)
    visible_pieces = {
        sq: chess.Piece(_PIECE_LETTER_TO_TYPE[vp.type], color_from_protocol(vp.color))
        for sq, vp in obs.visible_pieces
    }
    game_over = None
    if obs.game_over is not None:
        winner: Optional[chess.Color] = (
            color_from_protocol(obs.game_over.winner)
            if obs.game_over.winner is not None
            else None
        )
        game_over = GameOver(winner=winner, reason=obs.game_over.reason)
    return Observation(
        visibility_mask=visibility_mask,
        visible_pieces=visible_pieces,
        own_capture_square=obs.own_capture_square,
        opp_capture_landing_square=obs.opp_capture_landing_square,
        game_over=game_over,
    )


def build_perspective_view(req: proto.EngineTurnRequest) -> PerspectiveView:
    """Build the PerspectiveView strategy.pick_move(view) expects.

    Uses the LAST observation in the transcript (or the delta) for
    visibility + piece map, and the request's legalMoves for own
    legal moves. Clock fields come from the protocol's EngineClock.
    """
    perspective = color_from_protocol(req.color)
    last_obs: Optional[proto.EngineObservation] = None
    if req.observation_transcript:
        last_obs = req.observation_transcript[-1]
    elif req.latest_observation_delta is not None:
        last_obs = req.latest_observation_delta
    if last_obs is None:
        raise ValueError(
            "EngineTurnRequest missing both observation_transcript and "
            "latest_observation_delta"
        )

    visibility = chess.SquareSet(last_obs.visibility_mask)
    visible_piece_map = {
        sq: chess.Piece(_PIECE_LETTER_TO_TYPE[vp.type], color_from_protocol(vp.color))
        for sq, vp in last_obs.visible_pieces
    }
    own_legal_moves = [move_from_protocol(m) for m in req.legal_moves]

    return PerspectiveView(
        perspective=perspective,
        own_legal_moves=own_legal_moves,
        visible_squares=visibility,
        visible_piece_map=visible_piece_map,
        clock_remaining_ms=req.clock.remaining_ms,
        increment_ms=req.clock.increment_ms,
    )


def replay_transcript_into_strategy(
    strategy,
    req: proto.EngineTurnRequest,
) -> None:
    """Apply the protocol's observation_transcript to a strategy via its
    observe_own_move / observe_opp_move hooks. Resets the strategy first.

    Cold-start path — the caller has just initialized (or wants to fully
    re-seed) the strategy's belief from scratch. Steady-state path with
    `latest_observation_delta` is a different code path (the strategy
    keeps state across requests; only the new observation is applied).
    """
    perspective = color_from_protocol(req.color)
    # Pass game_id for per-game seeding (v2 EV-margin mixing variety) when the
    # strategy's reset() accepts it; tier1/legacy resets take only perspective.
    try:
        strategy.reset(perspective, game_id=req.game_id)
    except TypeError:
        strategy.reset(perspective)
    transcript = req.observation_transcript or ()
    for obs in transcript:
        if obs.kind == "initial":
            continue
        internal = observation_from_protocol(obs)
        if obs.kind == "own_move":
            if obs.own_move is None:
                raise ValueError(
                    f"protocol observation at ply {obs.ply} has kind='own_move' "
                    "but own_move field is missing"
                )
            strategy.observe_own_move(move_from_protocol(obs.own_move), internal)
        elif obs.kind == "opp_move":
            strategy.observe_opp_move(internal)
        else:
            raise ValueError(f"unknown observation kind: {obs.kind}")


def feed_transcript_tail(
    strategy,
    req: proto.EngineTurnRequest,
    start_idx: int,
) -> int:
    """Steady-state path: apply only the observations at index >= ``start_idx``
    WITHOUT resetting — the strategy keeps its belief across requests. This is
    the stateful-session counterpart to ``replay_transcript_into_strategy``:
    feeding [0:n] then [n:m] is equivalent to a full replay of [0:m] because the
    observe_* hooks are incremental. Returns the new processed length (len of
    the transcript). Caller is responsible for ensuring continuity (same game,
    append-only transcript); otherwise use the cold-start full replay.
    """
    transcript = req.observation_transcript or ()
    for obs in transcript[start_idx:]:
        if obs.kind == "initial":
            continue
        internal = observation_from_protocol(obs)
        if obs.kind == "own_move":
            if obs.own_move is None:
                raise ValueError(
                    f"protocol observation at ply {obs.ply} has kind='own_move' "
                    "but own_move field is missing"
                )
            strategy.observe_own_move(move_from_protocol(obs.own_move), internal)
        elif obs.kind == "opp_move":
            strategy.observe_opp_move(internal)
        else:
            raise ValueError(f"unknown observation kind: {obs.kind}")
    return len(transcript)


def board_from_request(req: proto.EngineTurnRequest) -> chess.Board:
    """Reconstruct a `chess.Board` from the last observation's visible
    pieces — used by the deadline-guard fallback in live workers.

    This is NOT a full canonical board (opp pieces on invisible squares
    are absent). It's only sufficient for the fallback move generator,
    which sorts the engine's own legal moves by a material-and-castle
    heuristic — both inputs the partial board does provide.
    """
    board = chess.Board.empty()
    last_obs: Optional[proto.EngineObservation] = None
    if req.observation_transcript:
        last_obs = req.observation_transcript[-1]
    elif req.latest_observation_delta is not None:
        last_obs = req.latest_observation_delta
    if last_obs is None:
        raise ValueError(
            "EngineTurnRequest missing both observation_transcript and "
            "latest_observation_delta"
        )
    for sq, vp in last_obs.visible_pieces:
        board.set_piece_at(
            sq,
            chess.Piece(_PIECE_LETTER_TO_TYPE[vp.type], color_from_protocol(vp.color)),
        )
    board.turn = color_from_protocol(req.color)
    return board


def apply_delta_to_strategy(
    strategy,
    delta: proto.EngineObservation,
) -> None:
    """Apply a single observation delta to a stateful strategy. Used
    on steady-state requests where the strategy retained state across
    the prior request and only needs the latest observation."""
    if delta.kind == "initial":
        # Shouldn't be sent as a delta in practice; no-op for safety.
        return
    internal = observation_from_protocol(delta)
    if delta.kind == "own_move":
        if delta.own_move is None:
            raise ValueError(
                f"protocol observation at ply {delta.ply} has kind='own_move' "
                "but own_move field is missing"
            )
        strategy.observe_own_move(move_from_protocol(delta.own_move), internal)
    elif delta.kind == "opp_move":
        strategy.observe_opp_move(internal)
    else:
        raise ValueError(f"unknown observation kind: {delta.kind}")
