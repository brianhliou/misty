"""Regression suite for the observation-keyed opening book.

Pins the live holes the book was built to close (Misty = the blunderer):

  g2 6a76912b  1.e4 c5 2.d4  Qa5 3.d5 Qxe1   White: d-push -> king captured
  g3 7509c295  1.e4 c5 2.Nf3 Qa5 3.d4 Qxe1   White: same trap, one move later
  g1 747112db  ...4.Nf3 Qh4 5.Nxh4           Black: Qh4 hangs to unseen Nf3
  g12 5a039274  ...5.exd6 Qh4 6.Nxh4         Black: Qh4 hangs to unseen Nf3
  g14 bd621af5  ...5.fxe5 Qh4 6.Nxh4         Black: Qh4 hangs to unseen Nf3
  g15 1f7e25ee  ...5.Nc3 Qh4 6.Nxh4          Black: Qh4 hangs to unseen Nf3

The Qxe1 king-trap FORCE entries (g2/g3 above) were DROPPED 2026-06-20: v1.2's
king guard now blocks the a5-e1 diagonal itself (lab/book_redundancy_probe.py,
king safe 8/8 without the book). The book now holds the still-live queen-hang
BLOCKs + a dxe4 FORCE (the move-2 commit-variance fix, deep-dive 9ed7d9a5).

The book fires only on exact fog-of-war view fingerprints. These tests use the
real ``PerspectiveView`` but stub ``_engine`` so the FORCE/BLOCK wiring is
exercised without spawning Stockfish — fast and deterministic.
"""

from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

from fow_chess.engine_v2 import EngineV2Strategy
from fow_chess.move_policy import MovePolicy, next_best_legal
from fow_chess.observation import Observation, observation_from_transition
from fow_chess.opening_book import (
    BookAction,
    OpeningBook,
    observation_event_fingerprint,
    observation_history_fingerprint,
    load,
    view_fingerprint,
)
from fow_chess.selfplay import PerspectiveView
from fow_chess.visibility import visible_piece_map, visible_squares


def _board_after(ucis: list[str]) -> chess.Board:
    b = chess.Board()
    for u in ucis:
        b.push(chess.Move.from_uci(u))
    return b


def _view(ucis: list[str], color: chess.Color) -> PerspectiveView:
    board = _board_after(ucis)
    work = board if board.turn == color else board.copy()
    work.turn = color
    return PerspectiveView(
        perspective=color,
        own_legal_moves=list(work.pseudo_legal_moves),
        visible_squares=visible_squares(board, color),
        visible_piece_map=visible_piece_map(board, color),
    )


def _history_fingerprint(ucis: list[str], color: chess.Color) -> str:
    board = chess.Board()
    events = []
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        prev = board.copy(stack=False)
        mover = board.turn
        board.push(move)
        obs = observation_from_transition(prev, board, color)
        if mover == color:
            events.append(observation_event_fingerprint("own", obs, move=move))
        else:
            events.append(observation_event_fingerprint("opp", obs))
    return observation_history_fingerprint(events)


def _history_view(ucis: list[str], color: chess.Color) -> SimpleNamespace:
    return SimpleNamespace(
        **vars(_view(ucis, color)),
        observation_history_fingerprint=_history_fingerprint(ucis, color),
    )


# Decision positions (side to move is the one that blundered).
F2 = (["e2e4", "c7c5"], chess.WHITE)                          # blind move-2 view (now unbooked)
F3 = (["e2e4", "c7c5", "b1c3", "d8a5"], chess.WHITE)          # -> force Nf3 (move 3)
DXE4 = (["g1f3", "d7d5", "e2e4"], chess.BLACK)               # -> force dxe4 (commit-variance fix)
G1 = (["f2f4", "e7e5", "f4e5", "d7d6", "e5d6", "f8d6", "g1f3"], chess.BLACK)  # -> block Qh4
G12 = ([
    "f2f4", "c7c5", "g1f3", "b8c6", "g2g4", "e7e5", "f4e5", "d7d6", "e5d6",
], chess.BLACK)  # -> block Qh4
G14 = ([
    "f2f4", "c7c5", "g1f3", "b8c6", "g2g4", "d7d5", "b1c3", "e7e5", "f4e5",
], chess.BLACK)  # -> block Qh4
G15 = ([
    "f2f4", "e7e5", "f4e5", "d7d6", "e5d6", "f8d6", "g1f3", "b8c6", "b1c3",
], chess.BLACK)  # -> block Qh4
QH4_CASES = (G1, G12, G14, G15)
QG5_CASES = QH4_CASES
MINED_OQE_CASES = (
    (["d2d4", "c7c5", "d1d2", "c5d4", "g1f3", "e7e5", "f3e5"], chess.BLACK, "d8a5"),
    (["e2e4", "c7c5", "g1f3", "b8c6", "b1c3", "d7d5", "e4d5"], chess.BLACK, "d8d5"),
    (
        ["e2e4", "b8c6", "d2d4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d8",
         "c1e3", "e7e6", "d4d5"],
        chess.BLACK,
        "d8d5",
    ),
    (
        ["e2e4", "c7c5", "g1f3", "g8f6", "b1c3", "e7e6", "d2d4", "d7d6",
         "d4c5", "b8c6", "c5d6"],
        chess.BLACK,
        "d8d6",
    ),
    (
        ["e2e4", "g8f6", "d2d4", "e7e5", "d4e5", "f6e4", "d1d5", "e4c5",
         "c1e3", "d8e7", "e3c5"],
        chess.BLACK,
        "e7c5",
    ),
)


# ---- fingerprint behaviour -------------------------------------------------


def test_fingerprint_is_deterministic():
    assert view_fingerprint(_view(*F2)) == view_fingerprint(_view(*F2))


def test_blind_replies_to_e4_collapse_to_one_fingerprint():
    # Every reply White cannot see produces the same White view -> one entry
    # covers them all.
    fps = {
        r: view_fingerprint(_view(["e2e4", r], chess.WHITE))
        for r in ("c7c5", "e7e6", "d7d6", "b8c6", "g7g6", "g8f6", "c7c6")
    }
    assert len(set(fps.values())) == 1, fps


def test_visible_reply_does_not_collapse():
    # 1...d5 is visible to White (e4 attacks d5) -> a distinct fingerprint, so
    # the book steps aside and the engine plays freely.
    blind = view_fingerprint(_view(["e2e4", "c7c5"], chess.WHITE))
    seen = view_fingerprint(_view(["e2e4", "d7d5"], chess.WHITE))
    assert blind != seen


# ---- book lookups close the three holes ------------------------------------


def test_book_loads():
    book = load()
    assert book is not None and len(book) == 15


def test_history_specific_book_entry_requires_matching_observation_history():
    view = _view(*G1)
    fp = view_fingerprint(view)
    h1 = observation_history_fingerprint(["own:e2e4|obs-a"])
    h2 = observation_history_fingerprint(["own:d2d4|obs-b"])
    qh4 = chess.Move.from_uci("d8h4")
    nf6 = chess.Move.from_uci("g8f6")
    book = OpeningBook(
        {
            fp: [
                BookAction(kind="block", move=qh4, history_fingerprint=h1),
                BookAction(kind="block", move=nf6, history_fingerprint=h2),
            ]
        }
    )

    assert book.lookup(view) is None
    assert book.lookup(SimpleNamespace(**vars(view), observation_history_fingerprint=h1)).move == qh4
    assert book.lookup(SimpleNamespace(**vars(view), observation_history_fingerprint=h2)).move == nf6


def test_view_only_entry_is_fallback_when_history_is_absent_or_different():
    view = _view(*G1)
    fp = view_fingerprint(view)
    qh4 = chess.Move.from_uci("d8h4")
    book = OpeningBook({fp: [BookAction(kind="block", move=qh4)]})
    with_history = SimpleNamespace(
        **vars(view),
        observation_history_fingerprint=observation_history_fingerprint(["opp|obs-c"]),
    )

    assert book.lookup(view).move == qh4
    assert book.lookup(with_history).move == qh4


@pytest.mark.parametrize("first", ["e2e4", "d2d4", "g1f3", "c2c4"])
def test_king_trap_nc3_force_dropped(first):
    # DROPPED 2026-06-20: v1.2's king guard now blocks the a5-e1 diagonal on its
    # own (lab/book_redundancy_probe.py: it plays c3 on move 3, king safe 8/8
    # without the book), so the prophylactic Nc3 force was removed and the blind
    # move-2 view is now unbooked — the engine plays its own opening freely.
    book = load()
    assert book.lookup(_view([first, "d7d6"], chess.WHITE)) is None
    assert book.pre_move(_view([first, "d7d6"], chess.WHITE)) is None


def test_move3_forces_nf3_and_supports_d4():
    # After 2.Nc3, the engine's free 3.d4 is unsound (cxd4 and the only recapture
    # Qxd4 hangs to an unseen Nc6). Forcing 3.Nf3 supports d4 (cxd4 Nxd4) and
    # develops; sound because Nc3 already defends e4.
    book = load()
    a = book.lookup(_view(*F3))
    assert a is not None and a.kind == "force"
    assert a.move == chess.Move.from_uci("g1f3")
    b = chess.Board()
    for u in ("e2e4", "c7c5", "b1c3", "d8a5", "g1f3"):
        b.push(chess.Move.from_uci(u))
    assert b.is_attacked_by(chess.WHITE, chess.D4)  # Nf3 now supports d4
    assert b.is_attacked_by(chess.WHITE, chess.E4)  # Nc3 still defends e4


def test_dxe4_force_fires():
    # ADDED 2026-06-20 (deep-dive 9ed7d9a5): after 1.Nf3 d5 2.e4 the engine values
    # dxe4 #1 (+0.14 over c6) but ~6% of time-bounded commits slip to the weaker
    # c6/Nf6. Force the value-best central pawn grab; view-only so it also covers
    # transpositions to the same Black view. Sound (even trade at worst).
    book = load()
    a = book.lookup(_view(*DXE4))
    assert a is not None and a.kind == "force"
    assert a.move == chess.Move.from_uci("d5e4")
    assert chess.Move.from_uci("d5e4") in _view(*DXE4).own_legal_moves


@pytest.mark.parametrize("case", QH4_CASES)
def test_qh4_cases_block_qh4(case):
    book = load()
    a = book.lookup(_view(*case))
    assert a is not None and a.kind == "block"
    assert a.move == chess.Move.from_uci("d8h4")


@pytest.mark.parametrize("case", QG5_CASES)
def test_qg5_cases_block_qg5_only_with_matching_observation_history(case):
    book = load()
    qg5 = chess.Move.from_uci("d8g5")
    nf6 = chess.Move.from_uci("g8f6")
    sol = SimpleNamespace(action_values_at_root={qg5: 0.9, nf6: 0.4})
    # Without the matching observation-history fingerprint, the old view-only
    # Qh4 block remains the only action for this current view.
    assert book.post_move(_view(*case), qg5, sol) is None
    assert book.post_move(_history_view(*case), qg5, sol) == nf6


@pytest.mark.parametrize("ucis,color,bad_move", MINED_OQE_CASES)
def test_mined_opening_queen_exposure_cases_are_history_scoped_blocks(ucis, color, bad_move):
    book = load()
    bad = chess.Move.from_uci(bad_move)
    view = _history_view(ucis, color)
    replacement = book.post_move(view, bad, SimpleNamespace(action_values_at_root={}))
    assert replacement is not None
    assert replacement != bad


@pytest.mark.parametrize("case", QH4_CASES)
def test_qh4_views_cannot_see_the_refuting_knight(case):
    # The whole bug: Black has no piece observing f3, so it never sees the knight
    # that takes the queen. The fingerprint must reflect that blindness.
    v = _view(*case)
    assert chess.F3 not in v.visible_piece_map


# ---- the book is a MovePolicy -----------------------------------------------


def test_book_satisfies_move_policy_protocol():
    book = load()
    assert isinstance(book, MovePolicy)
    assert book.name == "opening_book"


def test_policy_pre_move_forces_and_post_move_keeps():
    book = load()
    # FORCE entry -> pre_move returns the move; post_move leaves it alone.
    assert book.pre_move(_view(*DXE4)) == chess.Move.from_uci("d5e4")
    # BLOCK entry -> pre_move declines (None); it acts in post_move.
    for case in QH4_CASES:
        assert book.pre_move(_view(*case)) is None


@pytest.mark.parametrize("case,next_best", [
    (G1, "g8f6"),
    (G12, "f8d6"),
    (G14, "c6e5"),
    (G15, "g8f6"),
])
def test_policy_post_move_drops_blocked_top_pick(case, next_best):
    book = load()
    qh4 = chess.Move.from_uci("d8h4")
    replacement = chess.Move.from_uci(next_best)
    sol = SimpleNamespace(action_values_at_root={qh4: 0.9, replacement: 0.4})
    view = _view(*case)
    # top pick is the booked bad move -> replaced with next-best
    assert book.post_move(view, qh4, sol) == replacement
    # a different top pick -> no override
    assert book.post_move(view, replacement, sol) is None


def test_policy_post_move_skips_every_blocked_move_for_same_view():
    book = load()
    qh4 = chess.Move.from_uci("d8h4")
    qg5 = chess.Move.from_uci("d8g5")
    nf6 = chess.Move.from_uci("g8f6")
    sol = SimpleNamespace(action_values_at_root={qh4: 0.9, qg5: 0.8, nf6: 0.4})
    view = _history_view(*G1)

    assert book.post_move(view, qh4, sol) == nf6
    assert book.post_move(view, qg5, sol) == nf6


def test_next_best_legal_falls_back_when_ranking_empty():
    qh4 = chess.Move.from_uci("d8h4")
    view = _view(*G1)
    out = next_best_legal(SimpleNamespace(action_values_at_root={}), view, exclude=qh4)
    assert out != qh4 and out in view.own_legal_moves


# ---- FORCE / BLOCK end-to-end through pick_move (stub engine, no Stockfish) --


class _StubEngine:
    """Minimal EngineV2 stand-in: returns a fixed pick, no Stockfish/search."""

    def __init__(self, pick=None, solution=None):
        self._pick = pick
        self.last_solution = solution
        self.queen_promo_tiebreak = False

    def choose_move(self, **kwargs):
        return self._pick

    def observe_own_move(self, move, observation):
        self.last_observed = ("own", move, observation)

    def observe_opp_move(self, observation):
        self.last_observed = ("opp", observation)


class _CaptureHistoryPolicy:
    name = "capture_history"

    def __init__(self):
        self.seen = []

    def pre_move(self, view):
        self.seen.append(view.observation_history_fingerprint)
        return view.own_legal_moves[0]

    def post_move(self, view, chosen, solution):
        return None


def test_force_short_circuits_through_pick_move():
    strat = EngineV2Strategy(opening_book=True)
    strat._engine = _StubEngine(solution="stale")
    assert strat.pick_move(_view(*DXE4)) == chess.Move.from_uci("d5e4")
    assert strat.pick_move(_view(*F3)) == chess.Move.from_uci("g1f3")
    # search bypassed -> stale ranking cleared; policy recorded
    assert strat._engine.last_solution is None
    assert strat.last_policy_action == "opening_book:pre"


def test_block_replaces_through_pick_move():
    qh4 = chess.Move.from_uci("d8h4")
    nf6 = chess.Move.from_uci("g8f6")
    sol = SimpleNamespace(action_values_at_root={qh4: 0.9, nf6: 0.4})
    strat = EngineV2Strategy(opening_book=True)
    strat._engine = _StubEngine(pick=qh4, solution=sol)  # search would pick Qh4
    assert strat.pick_move(_view(*G1)) == nf6
    assert strat.last_policy_action == "opening_book:post"


def test_book_off_is_inert():
    strat = EngineV2Strategy(opening_book=False)
    assert strat._policies == []
    strat._engine = _StubEngine(pick=chess.Move.from_uci("d8h4"))
    # no policies -> the search pick is returned untouched, nothing recorded
    assert strat.pick_move(_view(*G1)) == chess.Move.from_uci("d8h4")
    assert strat.last_policy_action is None


def test_strategy_passes_observation_history_to_policies():
    strat = EngineV2Strategy(opening_book=False)
    strat._engine = _StubEngine()
    policy = _CaptureHistoryPolicy()
    strat._policies = [policy]
    view = _view(*G1)
    assert strat.pick_move(view) == view.own_legal_moves[0]
    assert policy.seen[-1] == observation_history_fingerprint([])

    obs = Observation(
        visibility_mask=view.visible_squares,
        visible_pieces=view.visible_piece_map,
    )
    move = chess.Move.from_uci("e2e4")
    strat.observe_own_move(move, obs)
    assert strat.pick_move(view) == view.own_legal_moves[0]
    assert policy.seen[-1] == observation_history_fingerprint(
        [observation_event_fingerprint("own", obs, move=move)]
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
