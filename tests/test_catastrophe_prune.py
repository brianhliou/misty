"""The catastrophe prune's NET-HANG floor (FOW_HV_PRUNE_NET_FLOOR).

Prod game 42b652b6 move 24: Qa4xe8 took a rook DEFENDED by the (unseen) black
queen — a net 400cp loss, under the prune's 500cp gross floor, so the prune
scored the game-losing move at 0.0cp risk. Worse, when the incoming commit was
the material guard's SAFE switch, the prune's STEP-1 redirect vetoed the safe
move (a real >=500cp tail in one belief world) and re-committed the highest
value "mat_safe" alternative — the blind-spot hang itself. Guard saves, prune
un-saves; the prune runs last by design, so its blind spot always wins.

These tests pin the fixed behavior (floor=300) AND the legacy blind spot
(floor unset), so the bug shape stays documented in the suite.
"""
import random

import chess

from fow_chess.engine_v2 import EngineV2


class _FakeEnum:
    def __init__(self, fens):
        self._fens = fens
        self.size = len(fens)

    def sample_root_fens(self, n, rng):
        return self._fens[:n]


def _engine_with_belief(fens):
    eng = object.__new__(EngineV2)
    eng.perspective = chess.WHITE
    eng.enumerator = _FakeEnum(fens)
    eng.rng = random.Random(0)
    eng._hv_prune_frac = None
    eng._hv_prune_adaptive = True
    eng._hv_prune_tau = 12.0
    eng._hv_prune_pref = 100.0
    eng._hv_prune_pmax = 8.0
    eng._hv_prune_king_floor = 0.02
    return eng


# Black: Ra8, Pa7, Re8, Qe6 (defends e8 down the open e-file), Kg8.
# White: Qa4 (sees e8 on the a4-e8 diagonal, a7 up the a-file), Kg1.
#   Qxe8 = queen for DEFENDED rook: gross 900, net 400 — the blind spot.
#   Qxa7 = queen for pawn into Ra8: gross 900, net 800 — legacy-flagged too.
#   Qa3  = quiet, nothing >=500cp is attackable after it.
_FEN = "r3r1k1/p7/4q3/8/Q7/8/8/6K1 w - - 0 1"

QXE8 = chess.Move.from_uci("a4e8")
QXA7 = chess.Move.from_uci("a4a7")
QA3 = chess.Move.from_uci("a4a3")


def test_net_floor_vetoes_queen_for_defended_rook(monkeypatch):
    monkeypatch.setenv("FOW_HV_PRUNE_NET_FLOOR", "300")
    eng = _engine_with_belief([_FEN, _FEN])
    avals = {QXE8: 0.50, QA3: 0.40}
    out = eng._catastrophe_prune(QXE8, avals, 0.0, {})
    assert out == QA3


def test_legacy_floor_is_blind_to_queen_for_rook(monkeypatch):
    monkeypatch.delenv("FOW_HV_PRUNE_NET_FLOOR", raising=False)
    eng = _engine_with_belief([_FEN, _FEN])
    avals = {QXE8: 0.50, QA3: 0.40}
    out = eng._catastrophe_prune(QXE8, avals, 0.0, {})
    assert out == QXE8  # documents the blind spot the floor closes


def test_redirect_must_not_land_in_the_blind_spot(monkeypatch):
    """The prod shape: the incoming move is the material guard's safe-ish
    switch, which the prune vetoes on its own >=500cp tail; the redirect must
    not re-commit the netted queen-for-rook hang."""
    monkeypatch.setenv("FOW_HV_PRUNE_NET_FLOOR", "300")
    eng = _engine_with_belief([_FEN, _FEN])
    avals = {QXA7: 0.50, QXE8: 0.45, QA3: 0.40}
    out = eng._catastrophe_prune(QXA7, avals, 0.0, {})
    assert out == QA3


def test_legacy_redirect_lands_in_the_blind_spot(monkeypatch):
    monkeypatch.delenv("FOW_HV_PRUNE_NET_FLOOR", raising=False)
    eng = _engine_with_belief([_FEN, _FEN])
    avals = {QXA7: 0.50, QXE8: 0.45, QA3: 0.40}
    out = eng._catastrophe_prune(QXA7, avals, 0.0, {})
    assert out == QXE8  # documents the guard-save -> prune-un-save inversion


def test_winning_capture_stays_unflagged_with_floor(monkeypatch):
    """RxQ that gets recaptured nets -400: the floor must not veto winning
    captures (the declined-capture regression class, games 56f30e52/57d77bdc)."""
    monkeypatch.setenv("FOW_HV_PRUNE_NET_FLOOR", "300")
    # White Rf1, Kh1; black Qf2 defended by Kg3. Rxf2 wins Q for R net +400.
    fen = "8/8/8/8/8/6k1/5q2/5R1K w - - 0 1"
    eng = _engine_with_belief([fen, fen])
    rxf2 = chess.Move.from_uci("f1f2")
    rb1 = chess.Move.from_uci("f1b1")
    avals = {rxf2: 0.60, rb1: 0.30}
    out = eng._catastrophe_prune(rxf2, avals, 0.0, {})
    assert out == rxf2
