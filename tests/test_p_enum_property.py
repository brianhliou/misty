"""Property-based tests for PEnumerator: random game histories.

Random-walk a short FoW game from the start and at every ply assert
the truth-in-P invariant for BOTH perspectives. Many seeds → high
coverage of the move-shape × visibility space.

Each test fixes a seed range so failures reproduce deterministically.
"""

from __future__ import annotations

import random

import chess
import pytest

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import (
    PEnumerator,
    assert_truth_in_P,
)


def _play_random_game(
    seed: int,
    max_plies: int = 10,
) -> list[tuple[chess.Board, chess.Move, chess.Board]]:
    """Return [(prev, move, nxt), ...] for a random FoW play."""
    rng = random.Random(seed)
    truth = chess.Board()
    history: list[tuple[chess.Board, chess.Move, chess.Board]] = []
    for _ in range(max_plies):
        # FoW legal = pseudo-legal. King-capture ends the game.
        if truth.king(chess.WHITE) is None or truth.king(chess.BLACK) is None:
            break
        moves = list(truth.pseudo_legal_moves)
        if not moves:
            break
        move = rng.choice(moves)
        prev = truth.copy()
        truth.push(move)
        history.append((prev, move, truth.copy()))
    return history


def _drive_enumerators(
    history: list[tuple[chess.Board, chess.Move, chess.Board]],
) -> tuple[PEnumerator, PEnumerator, list[chess.Board]]:
    """Replay ``history`` through two enumerators (W + B). Return both
    enumerators + the list of post-move truth boards at each ply."""
    e_white = PEnumerator(chess.WHITE)
    e_black = PEnumerator(chess.BLACK)
    truths: list[chess.Board] = []
    for prev, move, nxt in history:
        mover = prev.turn  # the color who just moved
        if mover == chess.WHITE:
            obs_for_black = observation_from_transition(prev, nxt, chess.BLACK)
            e_white.update_own_move(move)
            e_black.update_opp_move(obs_for_black)
        else:
            obs_for_white = observation_from_transition(prev, nxt, chess.WHITE)
            e_black.update_own_move(move)
            e_white.update_opp_move(obs_for_white)
        truths.append(nxt)
    return e_white, e_black, truths


@pytest.mark.parametrize("seed", range(15))
def test_truth_in_P_at_every_ply_short_game(seed):
    """15 random 6-ply games; truth in P at every ply for both POVs.

    Random play is pathological for |P|: no purposeful captures means
    fog never clears, and |P| can grow into the 10s of thousands by
    ply 10. We cap at 6 plies for the property gate. Real-game replays
    (test_p_enum_replay.py) provide longer-horizon coverage with
    purposeful play that keeps |P| reasonable.
    """
    history = _play_random_game(seed, max_plies=6)
    if not history:
        pytest.skip(f"seed {seed} produced empty history")
    e_white = PEnumerator(chess.WHITE)
    e_black = PEnumerator(chess.BLACK)
    for ply, (prev, move, nxt) in enumerate(history, start=1):
        mover = prev.turn
        if mover == chess.WHITE:
            obs_for_black = observation_from_transition(prev, nxt, chess.BLACK)
            e_white.update_own_move(move)
            e_black.update_opp_move(obs_for_black)
        else:
            obs_for_white = observation_from_transition(prev, nxt, chess.WHITE)
            e_black.update_own_move(move)
            e_white.update_opp_move(obs_for_white)

        assert_truth_in_P(e_white, nxt,
                          context=f"seed={seed} ply={ply} W-POV")
        assert_truth_in_P(e_black, nxt,
                          context=f"seed={seed} ply={ply} B-POV")


@pytest.mark.parametrize("seed", range(10))
def test_P_size_under_generous_bound_short_game(seed):
    """At 10 plies of random play, |P| should be well under 5×10⁵.

    Obscuro reports avg |P| ≈ 17K and max ~10⁶ for FoW chess in full
    games with purposeful play. Random play is pathological — no
    captures means fog never clears — so 10⁵+ at ply 10 is normal.
    Cap at 5×10⁵ to catch obvious explosions while allowing random-
    play pathology.
    """
    history = _play_random_game(seed, max_plies=10)
    if not history:
        pytest.skip(f"seed {seed} produced empty history")
    e_white, e_black, _ = _drive_enumerators(history)
    BOUND = 500_000
    assert e_white.size < BOUND, (
        f"seed={seed} |P_white|={e_white.size} exceeds {BOUND}"
    )
    assert e_black.size < BOUND, (
        f"seed={seed} |P_black|={e_black.size} exceeds {BOUND}"
    )


@pytest.mark.parametrize("seed", range(5))
def test_P_strictly_contains_truth_FEN(seed):
    """Sanity: P is a set of FEN strings; truth.fen() is in P (the
    FEN-string check ensures we're not just storing chess.Board objects
    that hash equal but differ in some other way)."""
    history = _play_random_game(seed, max_plies=10)
    if not history:
        pytest.skip(f"seed {seed} produced empty history")
    e_white, e_black, truths = _drive_enumerators(history)
    final_truth = truths[-1]
    assert final_truth.fen() in e_white.positions
    assert final_truth.fen() in e_black.positions
