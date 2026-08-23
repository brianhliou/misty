"""Tier-1 engine plumbing tests (Stockfish-independent)."""

from __future__ import annotations

import chess

from fow_chess import engine as engine_mod
from fow_chess.belief import BeliefState
from fow_chess.engine import best_action
from fow_chess.evaluator import king_safety_evaluator
from fow_chess.move_priors import uniform_prior


def test_best_action_picks_move_preferred_by_evaluator() -> None:
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
    )
    e2e4 = chess.Move.from_uci("e2e4")
    a2a3 = chess.Move.from_uci("a2a3")

    def evaluator(board, move, perspective):
        return 100.0 if move == e2e4 else 0.0

    chosen = best_action(belief, evaluator, [a2a3, e2e4], max_particles=None)
    assert chosen == e2e4


def test_best_action_returns_first_move_when_belief_empty() -> None:
    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        particles=[],
        weights=[],
    )

    def evaluator(board, move, perspective):
        return 0.0

    chosen = best_action(
        belief, evaluator, [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    )
    assert chosen == chess.Move.from_uci("e2e4")


def test_best_action_skips_moves_no_particle_considers_legal() -> None:
    # All particles share the standard initial position, so e7e5 is illegal
    # for white to play (not a white piece on e7). best_action should ignore
    # it and pick e2e4.
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
    )

    def evaluator(board, move, perspective):
        return 1.0  # uniform — only legality differs

    e2e4 = chess.Move.from_uci("e2e4")
    e7e5_white_attempt = chess.Move.from_uci("e7e5")
    chosen = best_action(
        belief, evaluator, [e7e5_white_attempt, e2e4], max_particles=None
    )
    assert chosen == e2e4


def test_best_action_deadline_can_interrupt_first_particle_round() -> None:
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=1,
    )
    a2a3 = chess.Move.from_uci("a2a3")
    e2e4 = chess.Move.from_uci("e2e4")
    evaluated: list[chess.Move] = []

    def evaluator(board, move, perspective):
        evaluated.append(move)
        return 100.0 if move == e2e4 else 0.0

    original_monotonic = engine_mod.time.monotonic
    engine_mod.time.monotonic = lambda: 1.0
    try:
        chosen = best_action(
            belief,
            evaluator,
            [a2a3, e2e4],
            max_particles=1,
            deadline_monotonic=0.5,
        )
    finally:
        engine_mod.time.monotonic = original_monotonic

    assert evaluated == [a2a3]
    assert chosen == a2a3


def test_king_safety_penalizes_mate_score_when_own_king_remains_attacked() -> None:
    # Regression for q10 ply 53: a Stockfish-like base evaluator returned a
    # mate-sized score for Kg3 even though the believed black king on h3 could
    # capture our king immediately. FOW king safety must not pass that score
    # through unless our move is itself a king capture.
    board = chess.Board.empty()
    board.set_piece_at(chess.F4, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H3, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    move = chess.Move.from_uci("f4g3")

    def mate_happy_base(_board, _move, _perspective):
        return 100_000.0

    score = king_safety_evaluator(mate_happy_base)(board, move, chess.WHITE)

    assert score <= -50_000.0
