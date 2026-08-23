"""Determinism test for ``fow_rust.update_opp_move_rust``.

The Rust function should produce a byte-identical result set for
identical inputs, every time. This is true for the current single-
threaded implementation by construction. **It will be the canary for
RP7** (multi-core via rayon): any race condition that drops or
double-counts an entry breaks set equality across runs.

Determinism is checked at the SET level, not the Vec-of-strings level.
The Rust function returns a Vec (insertion order); Python ``set()``s
the result before storing in the PEnumerator. Multi-threaded execution
may produce different Vec orderings — that's fine — but the final
set membership must be identical.

Mid-game position chosen to exercise non-trivial belief update:
~100-1000 prev positions × ~30 opp moves each = enough work for a
race to manifest.
"""

from __future__ import annotations

import chess
import pytest

try:
    import fow_rust
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import PEnumerator


pytestmark = pytest.mark.skipif(
    not _RUST_AVAILABLE,
    reason="fow_rust extension not built (run `maturin develop` in fow_rust/)",
)


def _build_obs_args(obs, perspective):
    """Pre-extract observation into the 17 scalar args update_opp_move_rust takes."""
    obs_w = [0] * 6
    obs_b = [0] * 6
    for sq, piece in obs.visible_pieces.items():
        bb = 1 << sq
        if piece.color:
            obs_w[piece.piece_type - 1] |= bb
        else:
            obs_b[piece.piece_type - 1] |= bb
    return {
        "obs_visibility_mask": int(obs.visibility_mask),
        "obs_w": obs_w,
        "obs_b": obs_b,
        "obs_own_idx": -1 if obs.own_capture_square is None else int(obs.own_capture_square),
        "obs_opp_idx": (
            -1 if obs.opp_capture_landing_square is None
            else int(obs.opp_capture_landing_square)
        ),
    }


def test_update_opp_move_rust_deterministic():
    """Build a real mid-game opp-move update scenario, run it 10 times,
    assert all 10 result sets are byte-identical."""
    # Play a short opening to get to a real opp-move update with non-trivial P
    board = chess.Board()
    penum = PEnumerator(chess.WHITE)
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
        prev = board.copy()
        mv = chess.Move.from_uci(uci)
        board.push(mv)
        if prev.turn == chess.WHITE:
            penum.update_own_move(mv)
        else:
            obs = observation_from_transition(prev, board, chess.WHITE)
            penum.update_opp_move(obs)

    # Now stage one more white move + black response so we have a real
    # observation to update_opp_move with.
    prev = board.copy()
    mv = chess.Move.from_uci("e1g1")  # white castles kingside
    board.push(mv)
    penum.update_own_move(mv)

    # Black plays — we update P via observation
    prev2 = board.copy()
    black_mv = chess.Move.from_uci("d7d6")
    board.push(black_mv)
    obs = observation_from_transition(prev2, board, chess.WHITE)
    args = _build_obs_args(obs, chess.WHITE)
    prev_fens = list(penum.iter_positions())
    assert len(prev_fens) >= 1, "PEnumerator should have at least one position"

    def call_once() -> frozenset[str]:
        kept, _raw = fow_rust.update_opp_move_rust(
            prev_fens,
            False,  # opp_white: black just moved, so opp is black=False
            True,   # perspective_white
            args["obs_visibility_mask"],
            *args["obs_w"], *args["obs_b"],
            args["obs_own_idx"], args["obs_opp_idx"],
        )
        return frozenset(kept)

    reference = call_once()
    for i in range(1, 10):
        other = call_once()
        assert other == reference, (
            f"determinism break on iteration {i}: "
            f"|ref|={len(reference)} |other|={len(other)} "
            f"|ref-other|={len(reference - other)} "
            f"|other-ref|={len(other - reference)}"
        )
