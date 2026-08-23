import random
from typing import Any

import chess

from fow_chess.belief import BeliefState
from fow_chess.event_log import (
    iter_steps,
    observations_for,
    own_moves_for,
    replay_canonical,
)
from fow_chess.move_priors import uniform_prior
from fow_chess.observation import consistent_with


def _move(uci: str) -> dict[str, Any]:
    move_dict: dict[str, Any] = {"from": uci[:2], "to": uci[2:4]}
    if len(uci) == 5:
        move_dict["promotion"] = {
            "q": "queen",
            "r": "rook",
            "b": "bishop",
            "n": "knight",
        }[uci[4]]
    return move_dict


def _events_for(uci_moves: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "room-created",
            "at": 0,
            "roomId": "test",
            "variant": "fog-of-war",
            "offer": [],
        },
        {
            "type": "seat-assigned",
            "at": 1,
            "roomId": "test",
            "clientId": "alice",
            "seat": "white",
        },
        {
            "type": "seat-assigned",
            "at": 2,
            "roomId": "test",
            "clientId": "bob",
            "seat": "black",
        },
    ]
    color = "white"
    for i, uci in enumerate(uci_moves):
        events.append(
            {
                "type": "move-played",
                "at": 100 + i,
                "roomId": "test",
                "color": color,
                "move": _move(uci),
            }
        )
        color = "black" if color == "white" else "white"
    return events


def test_replay_canonical_yields_initial_then_each_post_move_board() -> None:
    events = _events_for(["e2e4", "e7e5"])

    boards = list(replay_canonical(events))

    assert len(boards) == 3
    assert boards[0].fen() == chess.Board().fen()
    expected = chess.Board()
    expected.push_uci("e2e4")
    assert boards[1].fen() == expected.fen()
    expected.push_uci("e7e5")
    assert boards[2].fen() == expected.fen()


def test_own_moves_for_returns_only_own_moves_in_order() -> None:
    events = _events_for(["e2e4", "e7e5", "g1f3"])

    white_moves = own_moves_for(events, chess.WHITE)
    black_moves = own_moves_for(events, chess.BLACK)

    assert [m.uci() for m in white_moves] == ["e2e4", "g1f3"]
    assert [m.uci() for m in black_moves] == ["e7e5"]


def test_observations_for_returns_one_per_opp_move() -> None:
    events = _events_for(["e2e4", "e7e5", "g1f3"])

    white_obs = observations_for(events, chess.WHITE)
    black_obs = observations_for(events, chess.BLACK)

    assert len(white_obs) == 1
    assert len(black_obs) == 2


def test_truth_passes_consistency_at_every_opp_ply() -> None:
    moves = [
        "e2e4", "e7e5",
        "g1f3", "b8c6",
        "f1c4", "g8f6",
    ]
    events = _events_for(moves)

    for perspective in [chess.WHITE, chess.BLACK]:
        for step in iter_steps(events, perspective):
            if step.opp_observation is None:
                continue
            assert consistent_with(
                step.canonical_after,
                step.canonical_before,
                step.opp_observation,
                perspective,
            ), (
                f"truth rejected by consistency check at ply {step.ply} "
                f"for {'white' if perspective == chess.WHITE else 'black'}"
            )


def test_belief_stays_alive_through_short_game() -> None:
    moves = [
        "e2e4", "e7e5",
        "g1f3", "b8c6",
        "f1c4", "g8f6",
    ]
    events = _events_for(moves)

    for perspective in [chess.WHITE, chess.BLACK]:
        belief = BeliefState.initial(
            perspective=perspective,
            move_prior=uniform_prior,
            target_n=256,
            rng=random.Random(0),
        )
        for step in iter_steps(events, perspective):
            if step.own_move is not None:
                belief.update_after_own_move(step.own_move)
            else:
                assert step.opp_observation is not None
                belief.update_after_opp_move(step.opp_observation)
            assert not belief.collapsed(), (
                f"belief collapsed at ply {step.ply} for "
                f"{'white' if perspective == chess.WHITE else 'black'}"
            )


def test_promotion_move_converts() -> None:
    from fow_chess.event_log import _convert_move

    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))

    move = _convert_move(
        {"from": "e7", "to": "e8", "promotion": "queen"}, board
    )

    assert move.promotion == chess.QUEEN
    assert move.from_square == chess.E7
    assert move.to_square == chess.E8


def test_castling_move_translates_to_king_destination() -> None:
    from fow_chess.event_log import _convert_move

    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))

    move = _convert_move({"from": "e1", "to": "h1"}, board)

    assert move.from_square == chess.E1
    assert move.to_square == chess.G1


def test_selfplay_emits_canonical_rook_square_castling_event() -> None:
    from fow_chess.selfplay import _move_to_event

    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_H1

    move = chess.Move(chess.E1, chess.G1)

    assert board.is_castling(move)
    assert _move_to_event(move, board) == {"from": "e1", "to": "h1"}
