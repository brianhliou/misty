"""Belief keeps the truth when a double push happens with the king attacked.

Concrete shape of the bug, from prod game e751c6c4 (Misty as Black), ply 56:

  - White plays g2-g4 beside Black's pawn on f4, while White's queen on d6
    attacks Black's king on f8 down the d6-e7-f8 diagonal.
  - |P| goes 625 -> 41 and the TRUE board is not among the 41. Misty played
    its last four decisions of that game against a belief that could not
    contain reality.
  - Root cause: the en-passant square was recorded under a STANDARD-CHESS
    legality filter, in a variant that has no such rule. python-chess's
    `Board.fen()` emits the ep target only when a LEGAL ep capture exists, and
    legality there includes king safety; `apply_move_to_setup` in fow_rust
    deliberately mirrored that filter for byte-equal dedup. Fog of war chess
    has no king-safety rule: you may leave your king attacked, and the king is
    captured rather than mated, so an ep capture that would "leave you in
    check" is an ordinary legal move here.

    That matters because ep availability is part of FoW VISIBILITY. A pawn
    that can capture en passant sees the landing square and the pawn it would
    take. In the fixture below Black sees 14 squares with the ep square set and
    12 without it, losing g3 and g4.

    So the filter cost the belief the truth twice over:
      - The Rust path builds the successor's setup with ep already filtered
        away, computes 12 visible squares for it, compares that against the
        14 in the real observation, and rejects the true transition outright.
      - The pure-Python path accepts it (python-chess's `push` sets ep on the
        Board object; only `fen()` suppresses it) and then stores it as
        `... b KQ - 0 29`. Reconstructing that FEN next ply yields a board with
        the wrong visibility, so the truth falls out one ply later instead.

    Fix: record the ep square on the pseudo-legal rule (an enemy pawn stands
    beside the pushed pawn), which is python-chess's `EnPassantMode.XFEN` and
    exactly the fog-chess rule.

These tests assert the invariant rather than a particular ep field: a belief
seeded with the truth, handed the observation that the truth itself produced,
must still contain the truth. If it cannot reproduce reality from reality,
nothing downstream is sound.
"""

import chess
import pytest

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import PEnumerator, belief_fen

# The real predecessor from e751c6c4 after Black's 28...f4, White to move.
# White queen d6 attacks the black king on f8; g2-g4 is the double push.
PRED_KING_ATTACKED = "5k2/2Q4p/P2Qp3/4N3/5p2/B7/4PPPP/RN2KB1R w KQ - 0 29"
# Same position with the black king on h8, where nothing attacks it. This is
# the control: it exercises every identical code path and has always passed.
PRED_KING_SAFE = "6k1/2Q4p/P2Qp3/4N3/5p2/B7/4PPPP/RN2KB1R w KQ - 0 29"
DOUBLE_PUSH = "g2g4"


def _step(pred_fen: str, use_rust_state: bool):
    """Seed P with only the truth, then apply the real transition's observation."""
    pred = chess.Board(pred_fen)
    after = pred.copy()
    after.push(chess.Move.from_uci(DOUBLE_PUSH))
    obs = observation_from_transition(pred, after, chess.BLACK)
    pen = PEnumerator(
        chess.BLACK,
        starting_board=pred,
        max_size=None,
        use_rust_state=use_rust_state,
    )
    pen.update_opp_move(obs)
    return after, pen


@pytest.mark.parametrize("use_rust_state", [False, True], ids=["set", "rust_state"])
def test_truth_survives_double_push_while_king_attacked(use_rust_state):
    after, pen = _step(PRED_KING_ATTACKED, use_rust_state)
    assert after.is_check(), "test setup: the black king must be attacked"
    assert chess.Move.from_uci("f4g3") in after.pseudo_legal_moves, (
        "test setup: the en passant capture must be pseudo-legal, which is what "
        "fog rules allow and standard legality does not"
    )
    assert pen.size >= 1
    assert any(
        chess.Board(f).board_fen() == after.board_fen() for f in pen.iter_positions()
    ), "the true position left P after its own observation"


@pytest.mark.parametrize("use_rust_state", [False, True], ids=["set", "rust_state"])
def test_control_truth_survives_when_king_is_safe(use_rust_state):
    """The same double push, with nothing attacking the king. Always passed."""
    after, pen = _step(PRED_KING_SAFE, use_rust_state)
    assert not after.is_check(), "test setup: the black king must be safe here"
    assert any(
        chess.Board(f).board_fen() == after.board_fen() for f in pen.iter_positions()
    )


@pytest.mark.parametrize("use_rust_state", [False, True], ids=["set", "rust_state"])
def test_p_records_the_en_passant_square(use_rust_state):
    """P must carry the ep square, because FoW visibility depends on it."""
    after, pen = _step(PRED_KING_ATTACKED, use_rust_state)
    stored = [f for f in pen.iter_positions()
              if chess.Board(f).board_fen() == after.board_fen()]
    assert stored, "the true position left P after its own observation"
    assert stored[0].split()[3] == "g3", (
        f"P stored {stored[0]!r}. Dropping the ep square changes what Black can "
        f"see, so the next ply's consistency check runs against the wrong vision."
    )
    assert chess.Board(stored[0]).ep_square == chess.G3


@pytest.mark.parametrize("use_rust_state", [False, True], ids=["set", "rust_state"])
def test_membership_accepts_a_board_without_the_caller_choosing_a_spelling(use_rust_state):
    """`board in pen` must work for a caller holding a python-chess board.

    `board.fen()` uses python-chess's default en-passant mode and spells this
    position `... b KQ - 0 29` while P holds `... b KQ g3 0 29`, so passing the
    raw string is the trap this bug was made of. Passing the Board canonicalizes.
    """
    after, pen = _step(PRED_KING_ATTACKED, use_rust_state)
    assert after in pen
    assert belief_fen(after) in pen
    assert after.fen() != belief_fen(after), (
        "test setup: this fixture exists because the two spellings differ here"
    )
