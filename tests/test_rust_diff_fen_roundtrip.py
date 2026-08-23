"""Differential test: shakmaty FEN parse + serialize must produce byte-
identical output to python-chess ``board.fen()``.

Regression guard for the PEnumerator's set-dedup layer: if shakmaty's
FEN serializer ever diverges from python-chess (or our `apply_move`
post-condition stops normalizing ep_square the same way python-chess
does), set-based dedup over FEN strings silently breaks — duplicate
positions are kept under different canonical strings.

Coverage: starting position + every position reached during play in the
first 5 real engine games (≥200 unique FENs). Fast (~1s) — runs on
every CI.
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


def _game_paths(limit: int = 5) -> list[Path]:
    return corpus_game_paths(limit=limit)


def test_fen_roundtrip_starting_position():
    fen = chess.STARTING_FEN
    assert fow_rust.fen_roundtrip(fen) == fen


def test_fen_roundtrip_across_real_games():
    """Replay 5 real games; every position's FEN must round-trip byte-
    identical through fow_rust.fen_roundtrip."""
    game_paths = _game_paths()
    total = 0
    diffs: list[tuple[str, str]] = []
    for game_path in game_paths:
        moves = _load_moves(game_path)
        board = chess.Board()
        py_fen = board.fen()
        rust_fen = fow_rust.fen_roundtrip(py_fen)
        total += 1
        if py_fen != rust_fen:
            diffs.append((py_fen, rust_fen))
        for mv in moves:
            if mv not in board.pseudo_legal_moves:
                break
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                break
            board.push(mv)
            py_fen = board.fen()
            rust_fen = fow_rust.fen_roundtrip(py_fen)
            total += 1
            if py_fen != rust_fen:
                diffs.append((py_fen, rust_fen))
    assert total >= 200, f"expected ≥200 positions, got {total}"
    if diffs:
        sample = "\n".join(f"  PY:   {p}\n  RUST: {r}" for p, r in diffs[:3])
        raise AssertionError(
            f"FEN roundtrip diverged for {len(diffs)}/{total} positions:\n{sample}"
        )
