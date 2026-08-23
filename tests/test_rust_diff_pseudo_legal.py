"""Differential test: Rust ``pseudo_legal_moves`` must produce the same
move set as python-chess ``Board.pseudo_legal_moves``, byte-for-byte.

Move set comparison is set-equality on (from_sq, to_sq, promotion)
tuples. Castling is encoded as king's from→to. En passant is encoded as
pawn's from→ep_target. Promotion uses python-chess piece-type ints
(2=knight … 5=queen; 0 = no promotion).

Regression guard for the FoW-tolerant move generator: any drift in
pseudo-legality semantics (e.g., a castling-rights check, a missed
promotion variant, an ep edge case) corrupts every downstream P
update — and silently, because PEnumerator soundness only requires
truth-in-P, not that P doesn't contain extra junk.

Coverage: starting position + every position in 84 real engine games,
both colors. ~5K positions, ~1-2s on every CI.
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


def _py_move_set(board: chess.Board) -> set[tuple[int, int, int]]:
    return {(m.from_square, m.to_square, m.promotion or 0)
            for m in board.pseudo_legal_moves}


def _rust_move_set(fen: str) -> set[tuple[int, int, int]]:
    return set(fow_rust.pseudo_legal_moves(fen))


def _compare(fen: str, label: str, errors: list[str]) -> None:
    board = chess.Board(fen)
    py = _py_move_set(board)
    rust = _rust_move_set(fen)
    if py != rust:
        only_py = sorted(py - rust)
        only_rust = sorted(rust - py)
        errors.append(
            f"{label}\n  FEN: {fen}\n"
            f"  py-only:   {only_py}\n"
            f"  rust-only: {only_rust}"
        )


def test_pseudo_legal_starting_position():
    errors: list[str] = []
    _compare(chess.STARTING_FEN, "starting", errors)
    assert not errors, "\n".join(errors)


def test_pseudo_legal_real_game_replay():
    """Replay the corpus (committed fixtures + any private feedback games);
    every position's pseudo-legal move set must match python-chess
    byte-for-byte."""
    paths = corpus_game_paths()
    errors: list[str] = []
    total = 0
    for game_path in paths:
        moves = _load_moves(game_path)
        if not moves:
            continue
        board = chess.Board()
        _compare(board.fen(), f"{game_path.name} ply 0", errors)
        total += 1
        for ply, mv in enumerate(moves, start=1):
            if mv not in board.pseudo_legal_moves:
                break
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                break
            board.push(mv)
            _compare(board.fen(), f"{game_path.name} ply {ply}", errors)
            total += 1
            if errors and len(errors) >= 5:
                # Cap at 5 examples for readable failure output
                break
        if errors and len(errors) >= 5:
            break
    assert total >= 1000, f"expected ≥1000 positions tested, got {total}"
    if errors:
        raise AssertionError(
            f"pseudo_legal_moves diverged at {len(errors)} sampled positions:\n"
            + "\n".join(errors[:5])
        )
