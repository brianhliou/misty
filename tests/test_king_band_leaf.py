"""King-capture band floor: the leaf-eval CONTRACT.

The king-aware shim returns a HARD ±1.0 when king-capture is imminent, which
erases material ordering AT THE LEAF — a queen-promo leaf and a rook-promo leaf
both clamp to +1.0. The band returns ``sign*floor + (1-floor)*tanh(mat/scale)``
so king-capture still dominates while material orders the tie within the band.

These tests pin the LEAF contract:
  - ``floor == 1.0`` (default) is BYTE-IDENTICAL to the old hard clamp, and
    reproduces the leaf collapse (Q-leaf == R-leaf).
  - ``floor < 1.0`` restores leaf ordering (Q-leaf > R-leaf) while both stay
    dominant (>= 2*floor - 1).

NOTE — the band does NOT fix the live 4e2292f6 underpromotion end-to-end (that
is a CFR-integration artifact, not a leaf tie; see lab/probe_king_band.py for
the measured negative result). These tests assert the leaf math only.
"""

from __future__ import annotations

import chess
import pytest

from fow_chess.cfr.leaf_eval import (
    king_aware_leaf_enabled,
    king_band_floor,
    king_capture_imminent,
    material_leaf_eval,
    set_king_aware_leaf,
    set_king_band_floor,
    set_tanh_scale_cp,
    tanh_scale_cp,
)
from fow_chess.evaluator import material_score

# Two king-capture-imminent boards: White to move, White's queen on a1 attacks
# the Black king on a8 (king-capture next ply). They differ ONLY in White's
# material edge — an extra queen (+900cp) vs an extra rook (+500cp) — the
# leaf-eval analog of a queen-promo vs rook-promo choice.
FEN_BIG_EDGE = "k3q3/8/8/8/8/8/8/Q3Q2K w - - 0 1"  # White +900cp (2Q vs Q)
FEN_SMALL_EDGE = "k3q3/8/8/8/8/8/8/Q3R2K w - - 0 1"  # White +500cp (Q+R vs Q)


@pytest.fixture
def leaf_globals():
    """Save/restore the process-global leaf-eval flags (shared pytest process)."""
    saved = (king_aware_leaf_enabled(), king_band_floor(), tanh_scale_cp())
    set_king_aware_leaf(True)
    set_tanh_scale_cp(500.0)
    try:
        yield
    finally:
        set_king_aware_leaf(saved[0])
        set_king_band_floor(saved[1])
        set_tanh_scale_cp(saved[2])


def test_boards_are_king_capture_imminent(leaf_globals):
    """Sanity: both boards trigger the shim, with the expected material gap."""
    set_king_band_floor(1.0)
    big = chess.Board(FEN_BIG_EDGE)
    small = chess.Board(FEN_SMALL_EDGE)
    assert king_capture_imminent(big, chess.WHITE) is not None
    assert king_capture_imminent(small, chess.WHITE) is not None
    # White is up more material on the big-edge board.
    assert material_score(big, chess.WHITE) > material_score(small, chess.WHITE) > 0


def test_floor_one_is_byte_identical_hard_clamp(leaf_globals):
    """floor=1.0 returns the exact old ±1.0 — the default stays byte-identical."""
    set_king_band_floor(1.0)
    big = chess.Board(FEN_BIG_EDGE)
    # Perspective == side-to-move (White can take Black's king) -> +1.0 exactly.
    assert king_capture_imminent(big, chess.WHITE) == 1.0
    # Perspective == the side whose king is en-prise -> -1.0 exactly.
    assert king_capture_imminent(big, chess.BLACK) == -1.0


def test_clamp_collapses_material_ordering(leaf_globals):
    """floor=1.0 reproduces the bug: Q-leaf and R-leaf are indistinguishable."""
    set_king_band_floor(1.0)
    big = material_leaf_eval(chess.Board(FEN_BIG_EDGE), chess.WHITE)
    small = material_leaf_eval(chess.Board(FEN_SMALL_EDGE), chess.WHITE)
    assert big == small == 1.0  # collapsed -> CFR can't tell them apart


def test_band_preserves_material_ordering(leaf_globals):
    """floor<1.0 restores ordering while keeping king-capture dominant."""
    set_king_band_floor(0.9)
    big = material_leaf_eval(chess.Board(FEN_BIG_EDGE), chess.WHITE)
    small = material_leaf_eval(chess.Board(FEN_SMALL_EDGE), chess.WHITE)
    # Ordering restored: more material -> strictly higher leaf value.
    assert big > small
    # Both still dominant (>= 2*floor - 1 == 0.8) and bounded by +1.
    assert 0.8 <= small < big <= 1.0


def test_band_dominates_a_non_king_capture_leaf(leaf_globals):
    """A king-capture leaf with a modest edge still outranks a non-king-capture
    leaf with a comparable edge — king safety is not traded away for material."""
    set_king_band_floor(0.9)
    # King-capture-imminent, White +500cp.
    kc = material_leaf_eval(chess.Board(FEN_SMALL_EDGE), chess.WHITE)
    # A quiet (valid, no king-capture) position where White is up ~a rook.
    quiet = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert king_capture_imminent(quiet, chess.WHITE) is None
    quiet_v = material_leaf_eval(quiet, chess.WHITE)
    assert kc > quiet_v
