"""Post-game analysis: ground-truth grading + belief context, per ply.

Implements the belief and ground-truth layers of the three-way error
taxonomy (belief / sample / decision) from the analyze-game extension
spec. The search layer — was the truth in the sampled ``I``, and what did
the engine's own solve rank — plugs in on top of these rows; until then a
graded mistake is classified ``search_or_decision``.

Two consumers by design: the benchmark analyzer scripts, and the server
analysis pathway (post-game review), which turns these rows into per-move
review UI. Analysis runs only on FINISHED games with the full move list —
it never participates in live play, so seeing the true board here is not
a redaction concern.

Belief membership is a bug tripwire more than an error class: enumeration
is exact, so the truth is ALWAYS in ``P`` unless the enumerator lost it.
What the taxonomy calls a belief error therefore shows up as findability
(``|P|`` huge, so any fixed search sample almost surely misses the truth),
which is why every engine-ply row records ``belief_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import chess
import chess.engine

from .observation import observation_from_transition
from .p_enum import PEnumerator

# Mate scores mapped into centipawns for cp_loss arithmetic (same order of
# magnitude as leaf_eval's _MATE_SCORE_CP; the exact value only needs to
# dominate any real eval).
MATE_CP = 10_000

# Spec decision (2026-06-03): ~300cp is the headline mistake bar — 100cp is
# blitz/fog noise.
DEFAULT_MISTAKE_CP = 300

Verdict = Literal[
    "belief_lost_truth",  # truth ∉ P: the exact enumerator broke (bug tripwire)
    "sample_error",       # truth ∈ P but ∉ the sampled I — search never saw it
    "decision_error",     # truth ∈ I and the engine still chose badly
    "search_or_decision", # graded mistake, search layer not run — unrefined
]


def _verdict(
    cp_loss: Optional[int],
    mistake_cp: int,
    truth_in_p: bool,
    truth_in_i: Optional[bool],
) -> Optional[Verdict]:
    if cp_loss is None or cp_loss < mistake_cp:
        return None
    if not truth_in_p:
        return "belief_lost_truth"
    if truth_in_i is None:
        return "search_or_decision"
    return "decision_error" if truth_in_i else "sample_error"


@dataclass(frozen=True)
class TruthGrade:
    """Stockfish-on-truth grade of one played move, from the mover's POV."""

    sf_before_cp: int
    sf_best_uci: str
    sf_after_played_cp: int

    @property
    def cp_loss(self) -> int:
        return max(0, self.sf_before_cp - self.sf_after_played_cp)


@dataclass(frozen=True)
class PlyRow:
    """One ply of a finished game, engine-perspective.

    ``belief_size`` / ``truth_in_p`` are populated only for the analyzed
    color's own plies (the belief is that color's); ``grade`` is None when
    no grader was supplied or Stockfish couldn't evaluate the position
    (FoW-reachable positions can be standard-chess-illegal)."""

    ply: int
    color: Literal["white", "black"]
    uci: str
    belief_size: Optional[int] = None
    truth_in_p: Optional[bool] = None
    grade: Optional[TruthGrade] = None
    verdict: Optional[Verdict] = None
    # Search layer (analyze_game_deep only): the sampled I and the engine's
    # own solve at this ply.
    i_size: Optional[int] = None
    truth_in_i: Optional[bool] = None
    engine_top_uci: Optional[str] = None
    engine_top_value: Optional[float] = None
    played_value: Optional[float] = None


class TruthGrader:
    """Stockfish at a fixed depth on the TRUE board.

    Deliberately separate from the engine's depth-1 leaf eval: fixed depth
    (not movetime) keeps grades machine-independent and reproducible, and
    this grader sees full information — it exists to judge finished games,
    never to play. Restarts its subprocess when Stockfish chokes on a
    FoW-reachable, standard-chess-illegal position and reports that ply as
    ungradeable (None).
    """

    def __init__(self, *, depth: int = 18, path: Optional[str] = None) -> None:
        from .cfr.leaf_eval_stockfish import _find_stockfish

        self.depth = depth
        self.path = path or _find_stockfish()
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self._closed = False

    def _restart(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)

    def _eval_cp(self, board: chess.Board, pov: chess.Color) -> tuple[int, Optional[str]]:
        info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].pov(pov).score(mate_score=MATE_CP)
        pv = info.get("pv")
        best = pv[0].uci() if pv else None
        return int(score), best

    def grade(self, board: chess.Board, played: chess.Move) -> Optional[TruthGrade]:
        """Grade ``played`` on the true ``board`` (not mutated). None when
        Stockfish rejects either position."""
        mover = board.turn
        try:
            before_cp, best = self._eval_cp(board, mover)
            after = board.copy(stack=False)
            after.push(played)
            if after.king(chess.WHITE) is None or after.king(chess.BLACK) is None:
                # King capture: terminal for the mover — a win, graded as mate.
                after_cp = MATE_CP
            else:
                after_cp, _ = self._eval_cp(after, mover)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError):
            self._restart()
            return None
        return TruthGrade(
            sf_before_cp=before_cp,
            sf_best_uci=best or played.uci(),
            sf_after_played_cp=after_cp,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._engine.quit()
        except (chess.engine.EngineError, OSError):
            pass

    def __enter__(self) -> "TruthGrader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def analyze_game(
    moves: Iterable[chess.Move],
    engine_color: chess.Color,
    *,
    grader: Optional[TruthGrader] = None,
    mistake_cp: int = DEFAULT_MISTAKE_CP,
) -> list[PlyRow]:
    """Replay a finished game and produce per-ply analysis rows for
    ``engine_color`` (the analyzed side — nothing engine-specific yet;
    it's whichever seat you want the belief context for).

    Belief is reconstructed exactly the way the engine builds it live
    (``PEnumerator`` + ``observation_from_transition``, byte-identical
    observation derivation). Rows for the analyzed color carry belief
    size, truth membership, and — with a grader — the Stockfish-on-truth
    grade plus a verdict for mistakes at or above ``mistake_cp``.
    """
    board = chess.Board()
    pen = PEnumerator(engine_color)
    rows: list[PlyRow] = []

    for ply, mv in enumerate(moves, start=1):
        prev = board.copy()
        if prev.king(chess.WHITE) is None or prev.king(chess.BLACK) is None:
            break
        if mv not in prev.pseudo_legal_moves:
            break
        mover = prev.turn
        color: Literal["white", "black"] = "white" if mover == chess.WHITE else "black"
        board.push(mv)

        if mover == engine_color:
            belief_size = pen.size
            truth_in_p = prev.fen() in pen
            grade = grader.grade(prev, mv) if grader is not None else None
            verdict = _verdict(
                grade.cp_loss if grade is not None else None,
                mistake_cp,
                truth_in_p,
                None,
            )
            rows.append(
                PlyRow(
                    ply=ply,
                    color=color,
                    uci=mv.uci(),
                    belief_size=belief_size,
                    truth_in_p=truth_in_p,
                    grade=grade,
                    verdict=verdict,
                )
            )
            pen.update_own_move(mv)
        else:
            rows.append(PlyRow(ply=ply, color=color, uci=mv.uci()))
            obs = observation_from_transition(prev, board, engine_color)
            pen.update_opp_move(obs)

    return rows


def analyze_game_deep(
    moves: Iterable[chess.Move],
    engine_color: chess.Color,
    *,
    grader: Optional[TruthGrader] = None,
    mistake_cp: int = DEFAULT_MISTAKE_CP,
    iterations: int = 200,
    i_sample_size: int = 8,
    time_budget_seconds: Optional[float] = None,
    seed: int = 7,
    engine_factory=None,
) -> list[PlyRow]:
    """The full taxonomy: :func:`analyze_game` plus the SEARCH layer.

    Replays the game through a real ``EngineV2`` (uncapped belief = the
    true ``P``) and, at every analyzed ply, runs the engine's own solve to
    record the sampled ``I`` (``truth_in_i``), the engine's top move, and
    the played move's solve value. Graded mistakes then resolve to the
    complete belief / sample / decision verdict.

    Note the re-sample caveat from the spec: ``truth_in_i`` is measured
    for THIS solve's root draw, not the live game's — a valid findability
    measure, not a replay of the live decision. Costs one solve per
    analyzed ply; budget accordingly (tests use tiny iteration counts).
    """
    import random as _random

    from .engine_v2 import EngineV2

    if engine_factory is None:
        def engine_factory():
            return EngineV2(
                engine_color, rng=_random.Random(seed), p_max_size=None
            )

    eng = engine_factory()
    board = chess.Board()
    rows: list[PlyRow] = []
    try:
        for ply, mv in enumerate(moves, start=1):
            prev = board.copy()
            if prev.king(chess.WHITE) is None or prev.king(chess.BLACK) is None:
                break
            if mv not in prev.pseudo_legal_moves:
                break
            mover = prev.turn
            color: Literal["white", "black"] = (
                "white" if mover == chess.WHITE else "black"
            )
            board.push(mv)
            obs = observation_from_transition(prev, board, engine_color)

            if mover != engine_color:
                rows.append(PlyRow(ply=ply, color=color, uci=mv.uci()))
                eng.observe_opp_move(obs)
                continue

            belief_size = eng.enumerator.size
            truth_fen = prev.fen()
            truth_in_p = truth_fen in eng.enumerator

            eng.choose_move(
                iterations=iterations,
                i_sample_size=i_sample_size,
                time_budget_seconds=time_budget_seconds,
            )
            av = {
                m.uci(): v
                for m, v in (eng.last_solution.action_values_at_root or {}).items()
            }
            root_fens = eng.last_root_fens or []
            truth_in_i = truth_fen in set(root_fens)
            top_uci, top_value = (
                max(av.items(), key=lambda kv: kv[1]) if av else (None, None)
            )
            grade = grader.grade(prev, mv) if grader is not None else None
            rows.append(
                PlyRow(
                    ply=ply,
                    color=color,
                    uci=mv.uci(),
                    belief_size=belief_size,
                    truth_in_p=truth_in_p,
                    grade=grade,
                    verdict=_verdict(
                        grade.cp_loss if grade is not None else None,
                        mistake_cp,
                        truth_in_p,
                        truth_in_i,
                    ),
                    i_size=len(root_fens),
                    truth_in_i=truth_in_i,
                    engine_top_uci=top_uci,
                    engine_top_value=top_value,
                    played_value=av.get(mv.uci()),
                )
            )
            eng.observe_own_move(mv, obs)
    finally:
        eng.close()
    return rows
