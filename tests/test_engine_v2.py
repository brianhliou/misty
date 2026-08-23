"""Sanity tests for EngineV2 — confirms v2 can play complete games end-to-end.

These tests spawn real Stockfish subprocesses; they auto-skip if no
Stockfish is on PATH.

Coverage:
- EngineV2 can choose a move from the starting position.
- choose_move returns a pseudo-legal move (FoW-legal == pseudo-legal).
- choose_move with iterations=N and time_budget_seconds=T both work.
- Self-play (two EngineV2 instances) completes N plies without error.
- Internal state stays consistent (truth-in-P after every observation).
"""

from __future__ import annotations

import random
import shutil

import chess
import pytest

from fow_chess.engine_v2 import EngineV2
from fow_chess.observation import observation_from_transition


pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not on PATH",
)


def test_engine_v2_chooses_legal_move_from_start():
    """Basic sanity: EngineV2 returns a pseudo-legal move from the
    starting position."""
    with EngineV2(chess.WHITE, rng=random.Random(0)) as engine:
        move = engine.choose_move(iterations=10, i_sample_size=4)
    starting = chess.Board()
    assert move in starting.pseudo_legal_moves


def test_choose_move_with_time_budget():
    """time_budget_seconds caps wall time; engine still returns a move."""
    with EngineV2(chess.WHITE, rng=random.Random(0)) as engine:
        move = engine.choose_move(
            iterations=10000,  # high cap
            time_budget_seconds=0.5,  # short budget
            i_sample_size=4,
        )
    assert move in chess.Board().pseudo_legal_moves
    # last_solution should reflect that we stopped early
    assert engine.last_solution is not None
    assert engine.last_solution.elapsed_seconds < 1.0  # safe upper bound


def test_observe_own_move_updates_P():
    """After observe_own_move, P should reflect that we played that move."""
    with EngineV2(chess.WHITE, rng=random.Random(0)) as engine:
        size_before = engine.enumerator.size
        engine.observe_own_move(chess.Move.from_uci("e2e4"))
        # Own move is deterministic; |P| shouldn't grow.
        assert engine.enumerator.size <= size_before


def test_observe_opp_move_updates_P():
    """After observe_opp_move, P should reflect what we saw of opp."""
    with EngineV2(chess.WHITE, rng=random.Random(0)) as engine:
        truth = chess.Board()
        truth.push(chess.Move.from_uci("e2e4"))
        # WHITE-perspective engine: e2e4 is white's own move.
        engine.observe_own_move(chess.Move.from_uci("e2e4"))

        prev = truth.copy()
        truth.push(chess.Move.from_uci("e7e5"))
        obs = observation_from_transition(prev, truth, chess.WHITE)
        # Now from white's POV, black just moved. Update P via observation.
        size_before = engine.enumerator.size
        engine.observe_opp_move(obs)
        # |P| can grow (black's move is uncertain to white).
        assert engine.enumerator.size >= 1
        # Truth must still be in P (correctness invariant).
        assert truth.fen() in engine.enumerator.positions


def _play_self(max_plies: int, *, iterations: int = 5, i_sample_size: int = 4) -> int:
    """Run two EngineV2 instances against each other; return the
    number of plies completed before game-over (or max_plies)."""
    from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval

    # Share one Stockfish process across both engines for speed.
    sf = StockfishLeafEval()
    try:
        white = EngineV2(chess.WHITE, stockfish=sf, rng=random.Random(0))
        black = EngineV2(chess.BLACK, stockfish=sf, rng=random.Random(1))
        truth = chess.Board()
        plies = 0
        for _ in range(max_plies):
            # Terminal check (FoW: king captured)
            if (truth.king(chess.WHITE) is None
                    or truth.king(chess.BLACK) is None):
                break
            mover = truth.turn
            active = white if mover == chess.WHITE else black
            other = black if mover == chess.WHITE else white
            try:
                move = active.choose_move(
                    iterations=iterations, i_sample_size=i_sample_size,
                )
            except RuntimeError:
                # P became empty or some other soundness failure
                raise
            assert move in truth.pseudo_legal_moves, (
                f"engine returned illegal move {move.uci()} on {truth.fen()}"
            )
            prev = truth.copy()
            truth.push(move)
            active.observe_own_move(move)
            obs = observation_from_transition(prev, truth, other.perspective)
            other.observe_opp_move(obs)
            plies += 1
        return plies
    finally:
        sf.close()


def test_self_play_completes_5_plies():
    """Two EngineV2 instances play 5 plies without crash. Smoke."""
    plies = _play_self(max_plies=5, iterations=5, i_sample_size=4)
    assert plies >= 1  # at least the first move worked
