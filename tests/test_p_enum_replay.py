"""Replay-validation tests for PEnumerator on real recorded games.

Loads JSONL game logs from feedback/mirror-mcts-*/games/, replays
each from the start through both enumerators (white + black POV),
and asserts truth-in-P at every ply.

This is the strongest correctness gate: real adversarial play with
purposeful captures, exposed pieces, fog-clearing tactics. If
truth-in-P holds across hundreds of plies in real games, the
enumerator is unlikely to silently drop the truth in production use.

We also collect |P| stats per ply for cardinality reporting (consumed
by lab/diag/p_enum_replay_stats.py in A3.5).
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import PEnumerator, assert_truth_in_P


from game_corpus import corpus_game_paths


_PROMO_LETTER = {
    "queen": "q", "rook": "r", "bishop": "b", "knight": "n",
    "q": "q", "r": "r", "b": "b", "n": "n",
}


def _load_moves(replay_path: Path) -> list[chess.Move]:
    """Pull (move) tuples from a game-log JSONL in order."""
    moves: list[chess.Move] = []
    with replay_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") != "move-played":
                continue
            mv = event.get("move", {})
            frm = mv.get("from")
            to = mv.get("to")
            promo = mv.get("promotion")
            if frm is None or to is None:
                continue
            uci = f"{frm}{to}"
            if promo:
                letter = _PROMO_LETTER.get(str(promo).lower())
                if letter is None:
                    continue
                uci += letter
            moves.append(chess.Move.from_uci(uci))
    return moves


def _replay_first_n_games(limit: int) -> list[Path]:
    return corpus_game_paths(family="mcts-200-depth8", limit=limit)


# If |P| exceeds this on either side, we still record truth-in-P up
# to that ply but stop the test early. The point of this gate is
# "truth-in-P never fails on real play"; a |P|-explosion is a perf
# concern (A3.5 territory) not a correctness one.
_MAX_P_SIZE = 200_000


@pytest.mark.parametrize("replay_path", _replay_first_n_games(20), ids=lambda p: p.stem)
def test_truth_in_P_throughout_real_game(replay_path: Path):
    """For each ply of a real recorded game, truth must be in P for
    both perspectives. Stops early on |P| > _MAX_P_SIZE (recorded as
    `last_ply_validated`); the earlier plies still validate."""
    moves = _load_moves(replay_path)
    if not moves:
        pytest.skip(f"empty replay: {replay_path.name}")
    truth = chess.Board()
    e_white = PEnumerator(chess.WHITE)
    e_black = PEnumerator(chess.BLACK)
    last_ply_validated = 0
    bail_reason: str | None = None

    for ply, move in enumerate(moves, start=1):
        prev = truth.copy()
        if truth.king(chess.WHITE) is None or truth.king(chess.BLACK) is None:
            break  # game already ended
        if move not in truth.pseudo_legal_moves:
            pytest.fail(
                f"{replay_path.name}: ply {ply} move {move.uci()} not "
                f"pseudo-legal in truth {truth.fen()}"
            )
        truth.push(move)
        mover = prev.turn
        if mover == chess.WHITE:
            obs_for_black = observation_from_transition(prev, truth, chess.BLACK)
            e_white.update_own_move(move)
            e_black.update_opp_move(obs_for_black)
        else:
            obs_for_white = observation_from_transition(prev, truth, chess.WHITE)
            e_black.update_own_move(move)
            e_white.update_opp_move(obs_for_white)

        assert_truth_in_P(
            e_white, truth,
            context=f"{replay_path.name} ply={ply} W-POV (|P_w|={e_white.size})",
        )
        assert_truth_in_P(
            e_black, truth,
            context=f"{replay_path.name} ply={ply} B-POV (|P_b|={e_black.size})",
        )
        last_ply_validated = ply

        if e_white.size > _MAX_P_SIZE or e_black.size > _MAX_P_SIZE:
            bail_reason = (
                f"|P_w|={e_white.size} |P_b|={e_black.size} > {_MAX_P_SIZE}"
            )
            break

    # Make sure we validated at least a few plies; an immediate bail
    # would indicate something else is wrong.
    assert last_ply_validated >= 3, (
        f"{replay_path.name}: only validated {last_ply_validated} plies; "
        f"bail={bail_reason}"
    )
