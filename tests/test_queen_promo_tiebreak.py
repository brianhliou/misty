"""Queen-promotion tiebreak: never commit a dominated underpromotion.

In FoW (no stalemate) a queen strictly dominates a rook or bishop, so a
rook/bishop promotion is never better than a queen promotion. The engine is
only ever indifferent between them because the king-capture +1 is material-blind
(root-cause: lab/diag_king_capture_leaves.py). ``_upgrade_dominated_promotion``
breaks that tie toward the dominant piece, while leaving KNIGHT promotions alone
(a knight is NOT dominated — it reaches squares the queen can't).
"""

from __future__ import annotations

import shutil

import chess
import pytest

from fow_chess.engine_v2 import _upgrade_dominated_promotion

D2, D1 = chess.D2, chess.D1
PROMO_Q = chess.Move(D2, D1, promotion=chess.QUEEN)
PROMO_R = chess.Move(D2, D1, promotion=chess.ROOK)
PROMO_B = chess.Move(D2, D1, promotion=chess.BISHOP)
PROMO_N = chess.Move(D2, D1, promotion=chess.KNIGHT)
QUIET = chess.Move(chess.E2, chess.E4)


def test_rook_promo_upgraded_to_queen():
    assert _upgrade_dominated_promotion(PROMO_R) == PROMO_Q


def test_bishop_promo_upgraded_to_queen():
    assert _upgrade_dominated_promotion(PROMO_B) == PROMO_Q


def test_knight_promo_left_alone():
    # Knight is not dominated by the queen — never rewrite it.
    assert _upgrade_dominated_promotion(PROMO_N) == PROMO_N


def test_queen_promo_unchanged():
    assert _upgrade_dominated_promotion(PROMO_Q) == PROMO_Q


def test_non_promotion_unchanged():
    assert _upgrade_dominated_promotion(QUIET) == QUIET


def test_capture_promotion_preserves_target_square():
    # A capture-promotion (e.g. dxc1=R) upgrades to dxc1=Q — same squares.
    cap_r = chess.Move(chess.D2, chess.C1, promotion=chess.ROOK)
    cap_q = chess.Move(chess.D2, chess.C1, promotion=chess.QUEEN)
    assert _upgrade_dominated_promotion(cap_r) == cap_q


@pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="bare EngineV2 spawns the default Stockfish leaf eval at init",
)
def test_default_engine_leaves_tiebreak_off():
    # Bare EngineV2 default must preserve prior behavior (parity guards).
    import random

    from fow_chess.engine_v2 import EngineV2

    eng = EngineV2(chess.WHITE, rng=random.Random(0))
    assert eng.queen_promo_tiebreak is False
    eng.close()


def test_strongest_profile_enables_tiebreak():
    from fow_chess.engine_profile import STRONGEST

    assert STRONGEST.queen_promo_tiebreak is True
