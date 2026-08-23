"""Differential test: Rust ``apply_move`` must produce the same post-
move FEN as python-chess ``Board.push(move)``, byte-for-byte, for every
pseudo-legal move at every position in 84 real games.

This is the strongest correctness gate on the move-application path.
Catches:
  - Castling-rights edge cases (king-capture clears all rights for the
    captured color; rook capture clears that side's right)
  - En passant emit-only-if-legal (mirrors python-chess
    EnPassantMode.LEGAL FEN behavior)
  - EP-skewered check: capturer pawn vs captured pawn subtraction in
    legal-EP scan
  - Halfmove clock reset on pawn move or capture
  - Fullmove increment after black moves
  - Promotion piece placement
  - Any future change to apply_move semantics

Runtime: ~10-30s (178K push+apply+compare across 84 games). Slow test;
marked accordingly. Run via `pytest -m slow` or no-marker (still runs
by default — flip to opt-in if CI time becomes an issue).
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

try:
    import fow_rust
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="fow_rust extension not built (run `maturin develop` in fow_rust/)",
)


_PROMO_LETTER = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n",
                 "q": "q", "r": "r", "b": "b", "n": "n"}


def _load_moves(path: Path) -> list[chess.Move]:
    moves: list[chess.Move] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "move-played":
                continue
            mv = e.get("move", {})
            frm, to, promo = mv.get("from"), mv.get("to"), mv.get("promotion")
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


from game_corpus import corpus_game_paths


def _check_all_pseudo_legal(board: chess.Board, errors: list[str], counters: dict) -> None:
    fen = board.fen()
    for mv in board.pseudo_legal_moves:
        py_board = board.copy()
        py_board.push(mv)
        py_fen = py_board.fen()
        rust_fen = fow_rust.apply_move(
            fen, mv.from_square, mv.to_square, mv.promotion or 0
        )
        counters["total"] += 1
        if py_fen != rust_fen:
            counters["diffs"] += 1
            if len(errors) < 5:
                errors.append(
                    f"  POSITION: {fen}\n"
                    f"  MOVE:     {mv.uci()}\n"
                    f"  PY:       {py_fen}\n"
                    f"  RUST:     {rust_fen}"
                )


@pytest.mark.slow
def test_apply_move_real_game_replay():
    """Replay the corpus (committed fixtures + any private feedback games);
    at every position try EVERY pseudo-legal move and compare resulting FEN
    byte-for-byte against python-chess."""
    paths = corpus_game_paths()
    errors: list[str] = []
    counters = {"total": 0, "diffs": 0}
    for game_path in paths:
        moves = _load_moves(game_path)
        if not moves:
            continue
        board = chess.Board()
        _check_all_pseudo_legal(board, errors, counters)
        for mv in moves:
            if mv not in board.pseudo_legal_moves:
                break
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                break
            board.push(mv)
            _check_all_pseudo_legal(board, errors, counters)
    # Volume floor: catches a corpus that silently shrank. The committed
    # 16-game fixture layer alone produces ~35K tuples; the full private
    # feedback corpus far more.
    assert counters["total"] >= 30_000, (
        f"expected ≥30K (prev, move) tuples tested, got {counters['total']}"
    )
    if counters["diffs"]:
        raise AssertionError(
            f"apply_move diverged at {counters['diffs']}/{counters['total']} "
            f"tuples:\n" + "\n".join(errors)
        )
