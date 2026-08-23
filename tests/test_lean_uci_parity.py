"""Byte-parity gate for the lean UCI leaf eval (FOW_LEAN_UCI / use_lean).

The lean ``_FaithfulUCIClient`` exists purely for throughput (skips python-chess's
asyncio loop + full-PV parse, ~1.9x faster per eval). Its ONE contract is that it
returns the bit-identical float python-chess returns — so flipping it on is a
pure-throughput change with zero strength delta. This test is that contract.

A prior lean attempt broke parity by sending ``ucinewgame`` per eval (cleared
Stockfish's hash -> different depth-1 evals). Depth-1 evals are hash-history
dependent, so parity must be checked over the SAME ORDERED sequence fed to both
engines (not position-by-position in isolation) — which is what this does.

Standalone deep version (467-position random stress): lab/parity_lean_uci.py.
"""
from __future__ import annotations

import random
import shutil

import chess
import pytest

from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not on PATH",
)

W, B = chess.WHITE, chess.BLACK


def _midgame(moves: list[str]) -> str:
    b = chess.Board()
    for u in moves:
        b.push_uci(u)
    return b.fen()


# Curated edge cases: ep, mate (score mate), stalemate/checkmate (0 legal moves),
# FoW-invalid (-> material fallback, must match), high branching, both POVs.
_CORPUS: list[tuple[str, chess.Color]] = [
    (chess.STARTING_FEN, W),                                                  # startpos branch
    (chess.STARTING_FEN, B),                                                  # POV flip
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", W),      # ep square
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3", B),  # italian
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3", W),  # same, other POV
    (_midgame(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3",
               "g8f6", "d2d4", "e5d4", "c3d4", "c5b4", "b1c3", "f6e4"]), W),   # high branching
    ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", W),                                     # mate in 1
    ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", W),                             # back rank
    ("8/8/8/8/8/5k2/6q1/7K w - - 0 1", W),                                     # checkmated (0 legal)
    ("7k/8/6Q1/8/8/8/8/6K1 b - - 0 1", B),                                     # stalemate (0 legal)
    ("4k3/4K3/8/8/8/8/8/8 w - - 0 1", W),                                      # invalid: kings touching
    ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", W),      # invalid: not-to-move in check
]


def _random_sequence(n_games: int, plies: int, seed: int) -> list[tuple[str, chess.Color]]:
    rng = random.Random(seed)
    out: list[tuple[str, chess.Color]] = []
    seen: set[str] = set()
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(plies):
            if b.is_game_over():
                break
            epd = b.epd()
            if epd not in seen:
                seen.add(epd)
                out.append((b.fen(), b.turn))
            b.push(rng.choice(list(b.legal_moves)))
    return out


def test_lean_uci_byte_identical_to_python_chess():
    """evaluate() and evaluate_children() are bit-identical between the
    python-chess path and the lean path, over the SAME ordered sequence (so both
    Stockfish transposition tables evolve in lockstep under hash retention)."""
    sequence = _CORPUS + _random_sequence(n_games=4, plies=30, seed=20260529)
    ref = StockfishLeafEval(hash_mb=1, threads=1, tanh_scale_cp=500.0, use_lean=False)
    lean = StockfishLeafEval(hash_mb=1, threads=1, tanh_scale_cp=500.0, use_lean=True)
    assert lean.use_lean and not ref.use_lean
    try:
        for fen, persp in sequence:
            b = chess.Board(fen)
            rc = ref.evaluate_children(b, persp)
            lc = lean.evaluate_children(b, persp)
            assert rc == lc, f"children mismatch at {fen} (persp={persp}): {rc} != {lc}"
            rs = ref.evaluate(b, persp)
            ls = lean.evaluate(b, persp)
            assert rs == ls, f"single mismatch at {fen} (persp={persp}): {rs!r} != {ls!r}"
    finally:
        ref.close()
        lean.close()
    # The two FoW-invalid corpus entries must have hit the material fallback
    # identically on both paths (proves the is_valid guard fires the same way).
    assert ref.fallback_count == lean.fallback_count >= 2


def test_use_lean_defaults_to_env(monkeypatch):
    """FOW_LEAN_UCI toggles the lean path; DEFAULT is now ON (flipped 2026-05-29),
    opt OUT with FOW_LEAN_UCI=0."""
    assert StockfishLeafEval is not None  # import guard
    from fow_chess.cfr.leaf_eval_stockfish import _lean_uci_enabled
    monkeypatch.setenv("FOW_LEAN_UCI", "1")
    assert _lean_uci_enabled() is True
    monkeypatch.setenv("FOW_LEAN_UCI", "0")
    assert _lean_uci_enabled() is False
    monkeypatch.delenv("FOW_LEAN_UCI", raising=False)
    assert _lean_uci_enabled() is True  # default ON
