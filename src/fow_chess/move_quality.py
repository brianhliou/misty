"""P3.2 — Move-quality-vs-truth analyzer.

For each instrumented move during a self-play game, query Stockfish on the
canonical full-info board (the position the mover would see if FOW didn't
exist) and record whether the move played matches Stockfish's choice.

Separates two failure modes that look identical from win-rate alone:
- "Wrong belief" — Tier-1 picked the best move available given what it could
  see / believed. Stockfish-with-truth would pick a different move because it
  has more information. Disagreement here is unavoidable under FOW.
- "Wrong reasoning given belief" — Tier-1 picked a move that's worse even
  given full info. Disagreement here is engine-strength signal we can act on.

v1 records agreement only (binary). Eval-loss in centipawns can be added in
v1.5 by re-evaluating after the played move and negating the resulting score.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import chess

from .evaluator import _UCIEngine


@dataclass
class MoveQualityRecord:
    game_id: int
    ply: int
    mover_color: str  # 'white' or 'black'
    fen: str
    move_played: str  # UCI
    stockfish_best: str | None  # UCI; None on timeout / no bestmove
    eval_best_cp: float | None  # cp from mover POV
    agreement: bool  # False when stockfish_best is None


@dataclass
class MoveQualityAnalyzer:
    """Stockfish-on-truth analyzer. Reuses one Stockfish process across calls.

    Use as a context manager so the subprocess is cleaned up on exit.
    """

    depth: int = 8
    movetime_ms: int = 200
    stockfish_path: str = "stockfish"
    threads: int = 1
    records: list[MoveQualityRecord] = field(default_factory=list)
    _engine: _UCIEngine | None = None
    _current_game_id: int = 0
    _current_ply: int = 0

    def __enter__(self) -> "MoveQualityAnalyzer":
        self._engine = _UCIEngine(self.stockfish_path, threads=self.threads)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def begin_game(self, game_id: int) -> None:
        self._current_game_id = game_id
        self._current_ply = 0

    def record_move(
        self,
        canonical_board: chess.Board,
        move_played: chess.Move,
        mover_color: chess.Color,
    ) -> None:
        assert self._engine is not None
        self._current_ply += 1
        fen = canonical_board.fen()
        try:
            best, score = self._engine.analyze_fen_for_move(
                fen, depth=self.depth, movetime_ms=self.movetime_ms
            )
        except Exception:
            # Stockfish has been observed to crash mid-session on some FOW
            # positions (BrokenPipeError on the next send). _UCIEngine
            # self-heals on the next call; for this ply we record None.
            best, score = None, None
        played_uci = move_played.uci()
        self.records.append(
            MoveQualityRecord(
                game_id=self._current_game_id,
                ply=self._current_ply,
                mover_color="white" if mover_color == chess.WHITE else "black",
                fen=fen,
                move_played=played_uci,
                stockfish_best=best,
                eval_best_cp=score,
                agreement=(best is not None and best == played_uci),
            )
        )

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "game_id",
                    "ply",
                    "mover_color",
                    "fen",
                    "move_played",
                    "stockfish_best",
                    "eval_best_cp",
                    "agreement",
                ]
            )
            for r in self.records:
                writer.writerow(
                    [
                        r.game_id,
                        r.ply,
                        r.mover_color,
                        r.fen,
                        r.move_played,
                        r.stockfish_best if r.stockfish_best is not None else "",
                        f"{r.eval_best_cp:.1f}" if r.eval_best_cp is not None else "",
                        int(r.agreement),
                    ]
                )

    def summary(self) -> dict[str, float | int]:
        total = len(self.records)
        analyzed = sum(1 for r in self.records if r.stockfish_best is not None)
        agreed = sum(1 for r in self.records if r.agreement)
        return {
            "moves_recorded": total,
            "moves_analyzed": analyzed,
            "moves_agreed": agreed,
            "agreement_rate_over_recorded": (agreed / total) if total else 0.0,
            "agreement_rate_over_analyzed": (agreed / analyzed) if analyzed else 0.0,
            "analyze_success_rate": (analyzed / total) if total else 0.0,
        }
