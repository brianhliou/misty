"""Drive a fog-of-war game between two strategies, emitting mistboard event logs."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import chess

from .observation import Observation, observation_from_transition
from .visibility import visible_piece_map, visible_squares

GameEvent = dict[str, Any]


@dataclass(frozen=True)
class PerspectiveView:
    """The information a player has at decision time under fog of war.

    Computed from the canonical board by the harness and handed to strategies
    in place of the canonical board. Strategies see only their own pieces and
    squares their pieces can reach — they cannot peek at hidden opp pieces or
    squares outside their visibility set.

    `visible_piece_map` includes both own pieces (always visible to self) and
    visible opp pieces. Filter by `piece.color` to separate.

    `clock_remaining_ms` is this player's remaining clock at decision time
    (None when running in regime-1 / untimed mode). Strategies in regime-2
    (tournaments) MUST self-budget within this clock — exceeding it loses
    the game on time.
    """

    perspective: chess.Color
    own_legal_moves: list[chess.Move]
    visible_squares: chess.SquareSet
    visible_piece_map: dict[chess.Square, chess.Piece]
    clock_remaining_ms: int | None = None
    increment_ms: int = 0


@dataclass(frozen=True)
class TimeControlSpec:
    """Standard chess time control for a game.

    Both sides start with `initial_seconds` on their clock; after each of
    their own moves, `increment_seconds` is added. Clock <= 0 at end of
    pick_move = forfeit on time.
    """

    initial_seconds: float
    increment_seconds: float


@dataclass(frozen=True)
class OpeningPolicy:
    """How to seed the first plies of a game so engines don't see identical openings.

    `random_first_n_plies` plays N uniformly-random pseudo-legal moves before
    handing control to the strategies. Each side observes the random moves
    via the same Stage A/Stage B observation channels used for normal play,
    so beliefs are correctly initialized from the post-randomization state.

    Random moves don't consume clock time — they're not engine decisions.
    The opening is reproducible: same `seed` argument to `play_game`
    produces the same opening sequence.
    """

    kind: str  # "canonical" | "random_first_n_plies"
    n: int = 0

    @classmethod
    def canonical(cls) -> "OpeningPolicy":
        return cls(kind="canonical", n=0)

    @classmethod
    def random_first_n_plies(cls, n: int) -> "OpeningPolicy":
        if n < 0:
            raise ValueError("n must be >= 0")
        return cls(kind="random_first_n_plies", n=n)


class Strategy(Protocol):
    """Per-game player. Receives observations; returns moves.

    `pick_move` is called when it is this strategy's turn. The harness hands it
    a `PerspectiveView` — own legal moves, visibility data, and (in regime-2)
    the strategy's remaining clock. Visibility enables evaluators that score
    from observed truth (e.g. visibility-grounded threats) rather than
    particle-aggregated hypotheses. Clock enables self-budgeting.
    """

    def reset(self, perspective: chess.Color) -> None: ...
    def observe_own_move(self, move: chess.Move, observation: Observation) -> None: ...
    def observe_opp_move(self, observation: Observation) -> None: ...
    def pick_move(self, view: PerspectiveView) -> chess.Move: ...


@dataclass
class GameResult:
    events: list[GameEvent]
    plies: int
    winner: str | None  # 'white' | 'black' | None
    end_reason: str  # 'king-captured' | 'truncated' | 'draw' | 'no-legal-moves' | 'clock-expired'
    truncated: bool
    final_clocks_ms: tuple[int | None, int | None] = (None, None)  # (white, black)


MoveAnalyzer = Callable[[chess.Board, chess.Move, chess.Color], None]


def play_game(
    white: Strategy,
    black: Strategy,
    *,
    max_plies: int = 300,
    room_id: str = "engine-play",
    seed: int = 0,
    analyzer: MoveAnalyzer | None = None,
    time_control: TimeControlSpec | None = None,
    opening_policy: OpeningPolicy | None = None,
    events_sink: list | None = None,
) -> GameResult:
    """Run one FOW game from the standard start to a terminal state.

    Game-over: king-captured | draw | clock-expired | truncated (max_plies) |
    no-legal-moves. Draw is automatic on threefold repetition or the 50-move
    rule. Clock is enforced if `time_control` is set: each side
    starts at initial_seconds; pick_move wall time is debited from the
    moving side; `increment_seconds` added after the move. Side whose
    clock hits 0 forfeits.

    `opening_policy` controls the first plies of the game. Default
    (None or canonical) starts from the standard position with both
    strategies in control immediately. `random_first_n_plies` plays N
    uniformly-random pseudo-legal moves before handing control to the
    strategies. Random opening moves do not consume clock time.

    `analyzer`, if provided, is called after each move with the canonical
    board state BEFORE the move, the move played, and the mover's color.
    """
    board = chess.Board()
    white.reset(chess.WHITE)
    black.reset(chess.BLACK)

    # Caller can pass events_sink to retain the partial event list even when
    # play_game raises (e.g. PEnumerator soundness error mid-game). Lets
    # crash-postmortem code save what was played so far for viewer/replay.
    events: list[GameEvent] = events_sink if events_sink is not None else []
    events.append(
        {
            "type": "room-created",
            "at": 0,
            "roomId": room_id,
            "variant": "fog-of-war",
            "offer": [],
        }
    )

    if time_control is not None:
        white_clock_ms = int(time_control.initial_seconds * 1000)
        black_clock_ms = int(time_control.initial_seconds * 1000)
        increment_ms = int(time_control.increment_seconds * 1000)
    else:
        white_clock_ms = None
        black_clock_ms = None
        increment_ms = 0

    plies = 0
    end_reason = "truncated"

    # Random opening, if requested. Runs before the main loop so the strategies
    # see the random moves via their own Stage A / Stage B observation channels
    # and update their beliefs from the post-randomization state.
    if opening_policy is not None and opening_policy.kind == "random_first_n_plies":
        opening_rng = random.Random(seed)
        for _ in range(opening_policy.n):
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                end_reason = "king-captured"
                break
            color = board.turn
            legals = list(board.pseudo_legal_moves)
            if not legals:
                end_reason = "no-legal-moves"
                break
            move = opening_rng.choice(legals)
            prev = board.copy()
            board.push(move)
            plies += 1
            events.append(
                {
                    "type": "move-played",
                    "at": plies,
                    "roomId": room_id,
                    "color": "white" if color == chess.WHITE else "black",
                    "move": _move_to_event(move, prev),
                    "opening_random": True,
                    "thinkTimeMs": 0,
                }
            )
            active = white if color == chess.WHITE else black
            passive = black if color == chess.WHITE else white
            active.observe_own_move(
                move, observation_from_transition(prev, board, color)
            )
            opp = chess.BLACK if color == chess.WHITE else chess.WHITE
            passive.observe_opp_move(observation_from_transition(prev, board, opp))

    while plies < max_plies:
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            end_reason = "king-captured"
            break

        color = board.turn
        active = white if color == chess.WHITE else black
        passive = black if color == chess.WHITE else white

        own_legals = list(board.pseudo_legal_moves)
        if not own_legals:
            end_reason = "no-legal-moves"
            break

        active_clock_ms = (
            white_clock_ms if color == chess.WHITE else black_clock_ms
        )

        view = PerspectiveView(
            perspective=color,
            own_legal_moves=own_legals,
            visible_squares=visible_squares(board, color),
            visible_piece_map=visible_piece_map(board, color),
            clock_remaining_ms=active_clock_ms,
            increment_ms=increment_ms,
        )
        prev = board.copy()
        t0 = time.monotonic()
        move = active.pick_move(view)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Debit clock + check for time loss before applying the move.
        if active_clock_ms is not None:
            active_clock_ms -= elapsed_ms
            if active_clock_ms <= 0:
                end_reason = "clock-expired"
                if color == chess.WHITE:
                    white_clock_ms = active_clock_ms
                else:
                    black_clock_ms = active_clock_ms
                break
            active_clock_ms += increment_ms
            if color == chess.WHITE:
                white_clock_ms = active_clock_ms
            else:
                black_clock_ms = active_clock_ms

        if analyzer is not None:
            analyzer(prev, move, color)
        board.push(move)
        plies += 1

        events.append(
            {
                "type": "move-played",
                "at": plies,
                "roomId": room_id,
                "color": "white" if color == chess.WHITE else "black",
                "move": _move_to_event(move, prev),
                "compute_ms": elapsed_ms,
                "thinkTimeMs": elapsed_ms,
            }
        )

        # Two fog updates per move cycle:
        # Stage A — active sees what their own move revealed/hid (own piece moved).
        # Stage B — passive sees what active's move did from passive's perspective.
        active.observe_own_move(move, observation_from_transition(prev, board, color))
        opp = chess.BLACK if color == chess.WHITE else chess.WHITE
        passive.observe_opp_move(observation_from_transition(prev, board, opp))

        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            end_reason = "king-captured"
            break
        if board.halfmove_clock >= 100 or board.is_repetition(3):
            end_reason = "draw"
            break

    truncated = end_reason == "truncated"
    if end_reason == "clock-expired":
        # Side that ran out forfeits; opponent wins.
        winner = "black" if color == chess.WHITE else "white"
        events.append(
            {
                "type": "clock-expired",
                "at": plies,
                "roomId": room_id,
                "color": "white" if color == chess.WHITE else "black",
            }
        )
    elif board.king(chess.WHITE) is None:
        winner = "black"
    elif board.king(chess.BLACK) is None:
        winner = "white"
    else:
        winner = None

    return GameResult(
        events=events,
        plies=plies,
        winner=winner,
        end_reason=end_reason,
        truncated=truncated,
        final_clocks_ms=(white_clock_ms, black_clock_ms),
    )


def _move_to_event(move: chess.Move, prev: chess.Board) -> dict[str, Any]:
    out: dict[str, Any] = {
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
    }

    # Mistboard castling representation is "king-takes-friendly-rook" — convert
    # python-chess's standard king-2-square form when emitting.
    if prev.is_castling(move):
        is_kingside = chess.square_file(move.to_square) > chess.square_file(
            move.from_square
        )
        rank = chess.square_rank(move.from_square)
        rook_file = 7 if is_kingside else 0
        out["to"] = chess.square_name(chess.square(rook_file, rank))

    if move.promotion is not None:
        out["promotion"] = {
            chess.QUEEN: "queen",
            chess.ROOK: "rook",
            chess.BISHOP: "bishop",
            chess.KNIGHT: "knight",
        }[move.promotion]

    return out
