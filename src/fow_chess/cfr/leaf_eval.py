"""Position-only leaf evaluators for tabular CFR.

The CFR algorithm itself is value-scale-agnostic — but ``terminal_value`` on
the node and ``leaf_eval`` at the depth bound must produce comparable scales
or regret estimates will be dominated by whichever has larger magnitude.

Convention used here: all values are squashed into roughly ``[-1, 1]`` from
the perspective player's POV. ``+1`` ≈ winning, ``-1`` ≈ losing.
"""

from __future__ import annotations

import math
import os

import chess

from ..evaluator import fog_discount_term, material_score


_KING_AWARE_LEAF: bool = os.environ.get("FOW_KING_AWARE_LEAF") == "1"

# Shared tanh normalization scale (centipawns). Mutable via set_tanh_scale_cp
# so A/B harnesses can sweep without subprocesses. material_leaf_eval +
# hybrid_fog_leaf_eval read this at call-time; StockfishLeafEval reads the
# default at construction via tanh_scale_cp().
_TANH_SCALE_CP: float = float(os.environ.get("FOW_TANH_SCALE_CP", "500.0"))


def tanh_scale_cp() -> float:
    return _TANH_SCALE_CP


def set_tanh_scale_cp(scale: float) -> None:
    """Set the global tanh normalization scale. Used by A/B harnesses to
    compare different scales in a single process."""
    global _TANH_SCALE_CP
    _TANH_SCALE_CP = scale


# King-capture-imminent band floor. The king-aware shim returns a HARD +/-1.0
# when king-capture is imminent, which erases material ordering AT THE LEAF (a
# queen-promo and a rook-promo leaf both clamp to +1.0). The band returns
#     sign*floor + (1 - floor) * tanh(material / scale)     (clamped to [-1, 1])
# so king-capture still dominates (the sign*floor term) while material orders
# WITHIN the band. floor=1.0 reproduces the exact prior hard clamp
# (byte-identical) and is the default.
#
# MEASURED — NEGATIVE RESULT (lab/probe_king_band.py, 2026-05-30): the band does
# NOT by itself fix the live 4e2292f6 underpromotion. That underpromotion is a
# CFR-INTEGRATION artifact (the rook line's king-capture leaves integrate to a
# higher ROOT value — not a raw leaf tie), so only floor <= 0.3 flips the move
# back to queen, and floor=0 is byte-identical to king-aware OFF. The knob is
# therefore a king-safety <-> material-precision DIAL / diagnostic, NOT a
# validated fix. STRONGEST stays at 1.0 until a bakeoff says otherwise.
_KING_BAND_FLOOR: float = float(os.environ.get("FOW_KING_BAND_FLOOR", "1.0"))


def king_band_floor() -> float:
    return _KING_BAND_FLOOR


def set_king_band_floor(floor: float) -> None:
    """Set the king-capture-imminent band floor. 1.0 = the prior hard +/-1
    clamp; a value in (0, 1) preserves material ordering within the band. Used
    by the profile and A/B harnesses to sweep in a single process."""
    global _KING_BAND_FLOOR
    _KING_BAND_FLOOR = floor


def king_aware_leaf_enabled() -> bool:
    return _KING_AWARE_LEAF


def set_king_aware_leaf(enabled: bool) -> None:
    """Toggle the king-aware leaf-eval emission flag at runtime. Used by A/B
    harnesses (diag, replay-deviation) to compare OFF (prior behavior) vs ON
    in a single process with identical RNG state."""
    global _KING_AWARE_LEAF
    _KING_AWARE_LEAF = enabled


def king_capture_imminent(
    board: chess.Board, perspective: chess.Color
) -> float | None:
    """If the side to move can capture the opponent's king on the next ply,
    return a king-dominant value from ``perspective``'s POV. Otherwise None.

    With ``_KING_BAND_FLOOR == 1.0`` (default) this is a hard ±1.0 — the
    original shim, byte-identical. With a floor < 1.0 the value is BANDED:
    ``sign*floor + (1 - floor) * tanh(material / scale)`` (clamped to [-1, 1]),
    so king-capture still dominates while material orders ties WITHIN the band.
    This is a king-safety↔material DIAL, not a validated underpromotion fix —
    see the ``_KING_BAND_FLOOR`` note above for the measured negative result.

    Two FoW-legitimate "king-capture-imminent" patterns that standard-chess
    invalidation otherwise hides from the leaf evaluator:

    1. King-en-prise (e.g., Qa5 attacks Ke1 with Black to move on the line
       ``1.e4 c5 2.d4 Qa5 3.d5``). In FoW the attacker may have been hidden;
       the defender did not see the check and did not move out. Side to move
       captures the king next ply.
    2. Adjacent kings — each king attacks the other; side to move captures.

    Detection is via ``is_attacked_by(turn, opponent_king_sq)`` — narrower
    than ``not board.is_valid()``, which also fires on pawns-on-1st-rank,
    multi-king, etc. None of those are real king-capture signals.
    """
    target_king_sq = board.king(not board.turn)
    if target_king_sq is None:
        return None
    if not board.is_attacked_by(board.turn, target_king_sq):
        return None
    sign = 1.0 if perspective == board.turn else -1.0
    if _KING_BAND_FLOOR >= 1.0:
        return sign  # prior behavior: hard ±1.0 (byte-identical default)
    mat = math.tanh(material_score(board, perspective) / _TANH_SCALE_CP)
    v = sign * _KING_BAND_FLOOR + (1.0 - _KING_BAND_FLOOR) * mat
    return max(-1.0, min(1.0, v))


def material_leaf_eval(board: chess.Board, perspective: chess.Color) -> float:
    """Tanh-normalized material balance from ``perspective``'s POV.

    Squashes centipawn material into ``[-1, 1]``. A rook advantage (+500cp)
    maps to ~0.76; a queen advantage (+900cp) maps to ~0.95.
    """
    if _KING_AWARE_LEAF:
        v = king_capture_imminent(board, perspective)
        if v is not None:
            return v
    cp = material_score(board, perspective)
    return math.tanh(cp / _TANH_SCALE_CP)


def hybrid_fog_leaf_eval(board: chess.Board, perspective: chess.Color) -> float:
    """Tanh-normalized (material - 0.2 * fog_discount) from ``perspective``'s POV.

    Adds the simplest FoW-specific signal to material balance. ``fog_discount``
    penalizes own non-king pieces deep in enemy territory without defensive
    support — captures the FoW-implicit risk of exposed pieces to hidden
    attackers. The 0.2 weight matches ``fow_evaluator``'s ``fog_risk_weight``
    default, keeping this consistent with how the production evaluator
    blends the two signals.

    Cheaper than running full ``fow_evaluator`` at every leaf (which would
    require evaluating all legal moves) while still carrying real FoW
    knowledge into the CFR search.
    """
    if _KING_AWARE_LEAF:
        v = king_capture_imminent(board, perspective)
        if v is not None:
            return v
    cp = material_score(board, perspective) - 0.2 * fog_discount_term(
        board, perspective
    )
    return math.tanh(cp / _TANH_SCALE_CP)
