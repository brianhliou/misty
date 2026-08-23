"""Factored marginals over opponent piece types — observability only.

These helpers project a board or belief state down to a `[64, 6]` tensor
giving P(opp_piece_type | square) marginal mass per square. They were
the input encoding for Deep CFR's function approximator; that
substrate has been removed. They survive here as a *view* on belief
distributions — useful for debugging an exact `P` enumerator's output
or comparing its marginal distribution to a lossy baseline.

Not in the engine's hot path. Lifted from `cfr/walker.py` during
Phase A0 of the Obscuro replication.
"""

from __future__ import annotations

import chess
import numpy as np


# Column order for factored marginal tensors. `marginals[sq, i]` =
# probability that opp piece type ``OPP_PIECE_TYPE_ORDER[i]`` sits on
# ``sq`` from this player's POV.
OPP_PIECE_TYPE_ORDER: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
_PIECE_TYPE_TO_INDEX: dict[chess.PieceType, int] = {
    pt: i for i, pt in enumerate(OPP_PIECE_TYPE_ORDER)
}


def factored_marginals_from_truth(
    board: chess.Board, perspective: chess.Color
) -> np.ndarray:
    """[64, 6] marginals deterministically derived from the truth board.

    The marginal at (sq, piece_type) is 1.0 if an opp piece of that type
    sits on sq, 0.0 otherwise.
    """
    out = np.zeros((64, len(OPP_PIECE_TYPE_ORDER)), dtype=np.float32)
    opp = not perspective
    for sq, piece in board.piece_map().items():
        if piece.color != opp:
            continue
        out[sq, _PIECE_TYPE_TO_INDEX[piece.piece_type]] = 1.0
    return out


def factored_marginals_from_belief(belief_state) -> np.ndarray:
    """[64, 6] marginals derived from a multi-particle BeliefState.

    Reads each square's marginal distribution via
    ``BeliefState.marginal_piece_at`` and projects onto opp piece types.

    Note: ``BeliefState`` will be removed when `P` enumeration ships
    (Phase A3). At that point this function should be retargeted to
    accept an iterable of consistent positions, or be removed.
    """
    out = np.zeros((64, len(OPP_PIECE_TYPE_ORDER)), dtype=np.float32)
    opp = not belief_state.perspective
    for sq in range(64):
        m = belief_state.marginal_piece_at(sq)
        for piece, prob in m.items():
            if piece is None or piece.color != opp:
                continue
            idx = _PIECE_TYPE_TO_INDEX.get(piece.piece_type)
            if idx is None:
                continue
            out[sq, idx] = float(prob)
    return out
