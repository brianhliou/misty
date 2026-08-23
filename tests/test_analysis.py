"""Post-game analyzer: belief context + Stockfish-on-truth grading."""

from __future__ import annotations

import json
import shutil

import chess
import pytest

from fow_chess.analysis import PlyRow, TruthGrader, analyze_game
from game_corpus import corpus_game_paths

_PROMO = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}


def _fixture_moves(limit_plies: int) -> list[chess.Move]:
    path = corpus_game_paths(family="fow-eval-seed1", limit=1)[0]
    moves: list[chess.Move] = []
    with path.open() as f:
        for line in f:
            event = json.loads(line)
            if event.get("type") != "move-played":
                continue
            mv = event["move"]
            uci = mv["from"] + mv["to"] + _PROMO.get(str(mv.get("promotion", "")).lower(), "")
            moves.append(chess.Move.from_uci(uci))
            if len(moves) >= limit_plies:
                break
    assert moves
    return moves


def test_analyze_without_grader_builds_belief_rows():
    rows = analyze_game(_fixture_moves(12), chess.WHITE)
    assert len(rows) == 12
    white_rows = [r for r in rows if r.color == "white"]
    black_rows = [r for r in rows if r.color == "black"]
    assert len(white_rows) == 6 and len(black_rows) == 6
    for r in white_rows:
        # Exact enumeration: the truth is ALWAYS in P (tripwire invariant).
        assert r.truth_in_p is True
        assert r.belief_size >= 1
        assert r.grade is None and r.verdict is None
    for r in black_rows:
        # Opponent plies carry no belief context for the analyzed color.
        assert r.truth_in_p is None and r.belief_size is None
    # Belief grows once the opponent has moved under fog.
    assert white_rows[0].belief_size == 1
    assert white_rows[-1].belief_size > 1


def test_analyze_black_perspective_swaps_roles():
    rows = analyze_game(_fixture_moves(8), chess.BLACK)
    assert all(
        (r.truth_in_p is True) == (r.color == "black") or r.color == "white"
        for r in rows
    )
    assert all(r.belief_size is None for r in rows if r.color == "white")


@pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="TruthGrader needs a Stockfish binary",
)
def test_grader_produces_grades_and_no_false_verdicts_on_sane_opening():
    with TruthGrader(depth=6) as grader:
        rows = analyze_game(_fixture_moves(8), chess.WHITE, grader=grader)
    graded = [r for r in rows if r.color == "white"]
    assert all(r.grade is not None for r in graded)
    for r in graded:
        assert isinstance(r.grade.sf_before_cp, int)
        assert isinstance(r.grade.sf_after_played_cp, int)
        assert r.grade.cp_loss >= 0
        assert chess.Move.from_uci(r.grade.sf_best_uci)
        # A 300cp-bar verdict, when present, must be the search/decision
        # class — belief_lost_truth would mean the exact enumerator broke.
        assert r.verdict in (None, "search_or_decision")


@pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="TruthGrader needs a Stockfish binary",
)
def test_grader_flags_a_hung_queen():
    # 1. e4 e5 2. Qh5?? then Black plays g6 and White plays Qxe5?? losing
    # material is overkill to construct; instead grade a known blunder:
    # 1. f3 e5 2. g4?? — the position before Qh4# is graded for White.
    moves = [chess.Move.from_uci(u) for u in ("f2f3", "e7e5", "g2g4")]
    with TruthGrader(depth=8) as grader:
        rows = analyze_game(moves, chess.WHITE, grader=grader)
    g4 = rows[-1]
    assert g4.uci == "g2g4" and g4.grade is not None
    # Mate-in-one against: the eval collapses by well over the mistake bar.
    assert g4.grade.cp_loss >= 300
    assert g4.verdict == "search_or_decision"


def test_rows_are_frozen_records():
    import dataclasses

    row = PlyRow(ply=1, color="white", uci="e2e4")
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.ply = 2  # type: ignore[misc]


@pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="analyze_game_deep runs the real engine (Stockfish leaf eval)",
)
def test_deep_analysis_populates_search_layer():
    from fow_chess.analysis import analyze_game_deep

    rows = analyze_game_deep(
        _fixture_moves(6),
        chess.WHITE,
        iterations=60,
        i_sample_size=4,
    )
    engine_rows = [r for r in rows if r.color == "white"]
    assert engine_rows
    for r in engine_rows:
        assert r.truth_in_p is True
        assert r.i_size >= 1
        assert isinstance(r.truth_in_i, bool)
        assert r.engine_top_uci is not None
        assert chess.Move.from_uci(r.engine_top_uci)
        assert r.engine_top_value is not None
        # The played move was legal, so the solve rated it.
        assert r.played_value is not None
    # With |P| tiny in the opening, the sample holds the whole belief.
    first = engine_rows[0]
    assert first.belief_size == 1 and first.truth_in_i is True


@pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="needs Stockfish for both the grader and the engine leaf eval",
)
def test_deep_analysis_resolves_full_verdict_on_blunder():
    from fow_chess.analysis import analyze_game_deep

    moves = [chess.Move.from_uci(u) for u in ("f2f3", "e7e5", "g2g4")]
    with TruthGrader(depth=8) as grader:
        rows = analyze_game_deep(
            moves, chess.WHITE, grader=grader, iterations=60, i_sample_size=4
        )
    g4 = rows[-1]
    assert g4.uci == "g2g4"
    # Search layer ran, so the unrefined class must not appear: the verdict
    # resolves to sample_error or decision_error (belief is exact + tiny
    # here, so in practice decision_error).
    assert g4.verdict in ("sample_error", "decision_error")
