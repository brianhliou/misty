"""Bridge from mistboard TypeScript GameEvent logs to python-chess and Observations.

Currently supports standard-start Fog of War games. Chess960 starts (via
`draft-start-resolved`) and Bid For White games are not yet supported and will
raise `NotImplementedError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import chess

from .observation import Observation, observation_from_transition

GameEvent = dict[str, Any]


@dataclass(frozen=True)
class PerspectiveStep:
    """One ply from `perspective`'s POV; exactly one of own_move or opp_observation is set."""

    ply: int
    canonical_before: chess.Board
    canonical_after: chess.Board
    own_move: chess.Move | None = None
    opp_observation: Observation | None = None


def replay_canonical(events: list[GameEvent]) -> Iterator[chess.Board]:
    """Yield canonical boards: ply 0 (initial), then ply 1 (after first move-played), and so on."""
    board = _initial_board(events)
    yield board.copy()
    for event in events:
        if event.get("type") != "move-played":
            continue
        move = _convert_move(event["move"], board)
        board.push(move)
        yield board.copy()


def iter_steps(
    events: list[GameEvent], perspective: chess.Color
) -> Iterator[PerspectiveStep]:
    """Walk events and yield a `PerspectiveStep` per move ply."""
    board = _initial_board(events)
    ply = 0
    for event in events:
        if event.get("type") != "move-played":
            continue
        ply += 1
        actor = chess.WHITE if event["color"] == "white" else chess.BLACK
        move = _convert_move(event["move"], board)
        prev = board.copy()
        board.push(move)
        after = board.copy()

        if actor == perspective:
            yield PerspectiveStep(
                ply=ply,
                canonical_before=prev,
                canonical_after=after,
                own_move=move,
            )
        else:
            yield PerspectiveStep(
                ply=ply,
                canonical_before=prev,
                canonical_after=after,
                opp_observation=observation_from_transition(prev, after, perspective),
            )


def observations_for(
    events: list[GameEvent], perspective: chess.Color
) -> list[Observation]:
    """Project events into the observations `perspective` would make (one per opp move)."""
    return [
        step.opp_observation
        for step in iter_steps(events, perspective)
        if step.opp_observation is not None
    ]


def own_moves_for(
    events: list[GameEvent], perspective: chess.Color
) -> list[chess.Move]:
    """Project events into `perspective`'s own moves in order."""
    return [
        step.own_move
        for step in iter_steps(events, perspective)
        if step.own_move is not None
    ]


def _initial_board(events: list[GameEvent]) -> chess.Board:
    for event in events:
        if event.get("type") == "draft-start-resolved":
            raise NotImplementedError(
                "Chess960 starting positions are not yet supported in event_log"
            )
        if event.get("type") == "bid-resolved":
            raise NotImplementedError(
                "Bid For White games are not yet supported in event_log"
            )
    return chess.Board()


def _convert_move(move_dict: dict[str, Any], board: chess.Board) -> chess.Move:
    from_sq = chess.parse_square(move_dict["from"])
    to_sq = chess.parse_square(move_dict["to"])

    piece = board.piece_at(from_sq)
    if piece is not None and piece.piece_type == chess.KING:
        target = board.piece_at(to_sq)
        if (
            target is not None
            and target.piece_type == chess.ROOK
            and target.color == piece.color
        ):
            kingside = chess.square_file(to_sq) > chess.square_file(from_sq)
            if piece.color == chess.WHITE:
                to_sq = chess.G1 if kingside else chess.C1
            else:
                to_sq = chess.G8 if kingside else chess.C8

    promotion: int | None = None
    if "promotion" in move_dict:
        promotion = _PROMOTION_MAP[move_dict["promotion"]]
    return chess.Move(from_square=from_sq, to_square=to_sq, promotion=promotion)


_PROMOTION_MAP: dict[str, int] = {
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
}
