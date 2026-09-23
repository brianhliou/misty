"""The one FEN spelling the belief set is keyed by.

``P`` is a set of FEN strings, and two serializers write into it: python-chess
``Board.fen()`` on the pure-Python paths, and shakmaty ``Fen::to_string`` in
``fow_rust``. Membership is bare string equality, so any disagreement between
them silently drops positions from the belief rather than raising.

The en-passant field is the whole problem. python-chess defaults to
``EnPassantMode.LEGAL``, which records the ep square only when an ep capture is
legal by STANDARD chess rules, king safety included. Fog of war chess has no
king-safety rule: you may leave your king attacked, and the king is captured
rather than mated, so an ep capture that "leaves you in check" is an ordinary
move here. Under LEGAL those positions lose their ep square, and because ep
availability is part of FoW visibility (a pawn that can capture ep sees the
landing square and the pawn it would take), the round-trip through a FEN
quietly changes what the side to move could see.

``XFEN`` is the fog rule exactly: record the ep square when an enemy pawn
stands beside the pushed pawn, regardless of what that capture would do to the
capturer's king. It also keeps the ep square OUT of positions where no pawn is
adjacent, so the belief does not split on a square nobody can use.

Use this for every FEN that enters ``P`` or is tested against it. The Rust
mirror is ``has_pseudo_legal_ep_capture`` in ``fow_rust/src/lib.rs``; the two
are checked against each other by ``tests/test_rust_diff_fen_roundtrip.py``.
"""

from __future__ import annotations

import chess

__all__ = ["belief_fen"]


def belief_fen(board: chess.Board) -> str:
    """The canonical belief-set spelling of ``board``."""
    return board.fen(en_passant="xfen")
