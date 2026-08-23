"""Unit tests for the gadget Blueprint protocol + Stub/Carryover blueprints.

Keeps the gadget's slices honest: a flag/feature lands with a test, per
the discipline that the carryover/CFV backlog skipped (see
tests/test_carryover_cfv_parity.py).
"""
from __future__ import annotations

import random
import shutil

import chess
import pytest

from fow_chess.cfr.blueprint import Blueprint, CarryoverBlueprint, StubBlueprint
from fow_chess.rules import ChessRules


def test_stub_satisfies_blueprint_protocol() -> None:
    assert isinstance(StubBlueprint(), Blueprint)


def test_stub_opp_cfv_is_constant_across_infosets() -> None:
    bp = StubBlueprint(opp_cfv=-0.25)
    assert bp.opp_cfv(0) == -0.25
    assert bp.opp_cfv(123) == -0.25
    assert bp.opp_cfv(999_999) == -0.25


def test_stub_default_cfv() -> None:
    assert StubBlueprint().opp_cfv(0) == -0.1


def test_stub_opp_strategy_uniform_and_normalized() -> None:
    bp = StubBlueprint()
    moves = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1f3")]
    strat = bp.opp_strategy(0, moves)
    assert set(strat) == set(moves)
    assert abs(sum(strat.values()) - 1.0) < 1e-12
    for p in strat.values():
        assert abs(p - 1.0 / 3.0) < 1e-12


def test_stub_opp_strategy_empty_actions() -> None:
    assert StubBlueprint().opp_strategy(0, []) == {}


# ---------------------------------------------------------------------------
# CarryoverBlueprint — continual-resolve baseline (Obscuro-Parity Phase 2)
# ---------------------------------------------------------------------------


def test_carryover_satisfies_blueprint_protocol() -> None:
    assert isinstance(CarryoverBlueprint(ChessRules()), Blueprint)


def test_carryover_pos_key_is_epd() -> None:
    """Subtree-carryover match key = EPD (pieces+side+castling+ep), counters
    dropped. Guards the full-vs-piece FEN bug that made carryover a silent no-op."""
    from fow_chess.engine_v2 import _carryover_pos_key

    full = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 12"
    assert _carryover_pos_key(full) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    # same position, different (path-dependent) move counters -> same key (this is
    # WHY carryover must dedup: counter-twin belief worlds collide to one node).
    assert _carryover_pos_key("8/8/8/8/8/8/8/8 w - - 0 1") == _carryover_pos_key("8/8/8/8/8/8/8/8 w - - 9 40")
    # different castling rights -> DIFFERENT key (not conflated, unlike piece-only)
    assert _carryover_pos_key("X w KQkq - 0 1") != _carryover_pos_key("X w - - 0 1")


def test_carryover_opp_cfv_known_and_fallback() -> None:
    rules = ChessRules()
    bp = CarryoverBlueprint(rules, fallback=-0.1)
    board = chess.Board()
    # No carried value yet -> fallback.
    assert bp.opp_cfv(board) == -0.1
    # Set a value keyed by this world's board_fen -> returned exactly.
    bp.set_values({rules.board_fen(board): 0.42})
    assert bp.opp_cfv(board) == 0.42
    # A different (uncarried) world still falls back.
    other = chess.Board()
    other.push(chess.Move.from_uci("e2e4"))
    assert bp.opp_cfv(other) == -0.1


def test_carryover_default_fallback() -> None:
    assert CarryoverBlueprint(ChessRules()).opp_cfv(chess.Board()) == -0.1


def test_carryover_set_values_replaces_not_merges() -> None:
    """A stale value from two moves ago must never leak into the current solve:
    set_values REPLACES the map."""
    rules = ChessRules()
    bp = CarryoverBlueprint(rules)
    board = chess.Board()
    fen = rules.board_fen(board)
    bp.set_values({fen: 0.5})
    assert bp.opp_cfv(board) == 0.5
    bp.set_values({})  # next move carried nothing for this world
    assert bp.opp_cfv(board) == bp._fallback


def test_carryover_opp_strategy_uniform_and_normalized() -> None:
    bp = CarryoverBlueprint(ChessRules())
    moves = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1f3")]
    strat = bp.opp_strategy(0, moves)
    assert set(strat) == set(moves)
    assert abs(sum(strat.values()) - 1.0) < 1e-12
    assert bp.opp_strategy(0, []) == {}


# ---------------------------------------------------------------------------
# Slice 0 integration: the carried (x,y) actually crosses the move boundary.
# Needs Stockfish (leaf eval) + the carryover-subtree substrate.
# ---------------------------------------------------------------------------

_SF = shutil.which("stockfish") is None


@pytest.mark.skipif(_SF, reason="needs Stockfish on PATH")
def test_carryover_blueprint_populates_across_moves(monkeypatch) -> None:
    """With gadget + carryover blueprint + subtree carryover on, the first move
    has no prior tree (empty map) but move 2 reads a non-empty u(x,y|J) map off
    the carried tree, with every value finite and in the leaf value range
    [-1, 1]. This is the structural proof that Slice 0 persists the previous
    solve's (x,y) across the move boundary."""
    from fow_chess.engine_v2 import EngineV2
    from fow_chess.observation import observation_from_transition

    monkeypatch.setenv("FOW_RESOLVE_GADGET", "1")
    monkeypatch.setenv("FOW_RESOLVE_BLUEPRINT", "carryover")
    monkeypatch.setenv("FOW_CARRYOVER_SUBTREE", "1")

    persp = chess.WHITE
    eng = EngineV2(persp, rng=random.Random(42))
    assert isinstance(eng.gadget_blueprint, CarryoverBlueprint)
    rng = random.Random(42)
    sim = chess.Board()
    # Warmup: random plies to grow |P| > 1 (so sampling/carryover are exercised),
    # leaving White to move. The observer always uses White's POV.
    for ply in range(8):
        legal = list(sim.pseudo_legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        prev = sim.copy()
        sim.push(mv)
        obs = observation_from_transition(prev, sim, persp)
        if ply % 2 == 0:
            eng.observe_own_move(mv, obs)
        else:
            eng.observe_opp_move(obs)
    assert eng.enumerator.size > 1

    map_sizes = []
    try:
        for _ in range(2):
            m = eng.choose_move(iterations=300, i_sample_size=16, time_budget_seconds=None)
            vals = eng.gadget_blueprint._vals
            assert all(v == v and -1.0 <= v <= 1.0 for v in vals.values())
            map_sizes.append(len(vals))
            prev = sim.copy()
            sim.push(m)
            eng.observe_own_move(m, observation_from_transition(prev, sim, persp))
            legal = list(sim.pseudo_legal_moves)
            if not legal:
                break
            opp = rng.choice(legal)
            prev = sim.copy()
            sim.push(opp)
            eng.observe_opp_move(observation_from_transition(prev, sim, persp))
    finally:
        eng.close()

    assert map_sizes[0] == 0, "first move has no prior tree -> empty carried map"
    assert map_sizes[1] > 0, "move 2 must carry a non-empty (x,y) from move 1"


@pytest.mark.skipif(_SF, reason="needs Stockfish on PATH")
def test_carryover_opp_cfv_finds_carried_worlds(monkeypatch) -> None:
    """Regression guard for the FEN-key bug that gave 0% coverage (both 33%/29%
    bakeoffs were silently pure stub-gadget). The map was keyed by ``node_fen``
    (a FULL fen: side-to-move/castling/counters) while ``opp_cfv`` looks up
    ``rules.board_fen`` (piece placement only) — so a carried world was NEVER
    found. We assert a world whose board_fen IS a map key resolves to the carried
    value, not the fallback."""
    from fow_chess.engine_v2 import EngineV2
    from fow_chess.observation import observation_from_transition

    monkeypatch.setenv("FOW_RESOLVE_GADGET", "1")
    monkeypatch.setenv("FOW_RESOLVE_BLUEPRINT", "carryover")
    monkeypatch.setenv("FOW_CARRYOVER_SUBTREE", "1")

    persp = chess.WHITE
    eng = EngineV2(persp, rng=random.Random(1))
    rng = random.Random(1)
    sim = chess.Board()
    for ply in range(12):
        legal = list(sim.pseudo_legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        prev = sim.copy()
        sim.push(mv)
        obs = observation_from_transition(prev, sim, persp)
        if ply % 2 == 0:
            eng.observe_own_move(mv, obs)
        else:
            eng.observe_opp_move(obs)
    bp = eng.gadget_blueprint
    try:
        # Drive on-policy moves until the carried map is non-empty (population is
        # sensitive to whether the played move's subtree got expanded).
        vals = {}
        for _ in range(6):
            m = eng.choose_move(iterations=800, i_sample_size=32, time_budget_seconds=None)
            vals = dict(bp._vals)
            if vals:
                break
            prev = sim.copy()
            sim.push(m)
            eng.observe_own_move(m, observation_from_transition(prev, sim, persp))
            legal = list(sim.pseudo_legal_moves)
            if not legal:
                break
            opp = rng.choice(legal)
            prev = sim.copy()
            sim.push(opp)
            eng.observe_opp_move(observation_from_transition(prev, sim, persp))
        assert vals, "could not populate a carried map to exercise the lookup"
        # A map key is a piece-only board_fen; a board with that placement must be
        # FOUND by opp_cfv (== the carried value), not fall back. Under the bug the
        # keys were full fens, so this construction/lookup would miss.
        key = next(iter(vals))
        board = eng.rules.board_from_fen(key + " w - - 0 1")
        assert eng.rules.board_fen(board) == key, "map key is not a piece-only board_fen"
        assert bp.opp_cfv(board) == vals[key], "opp_cfv did not find the carried world (key mismatch)"
    finally:
        eng.close()


@pytest.mark.skipif(_SF, reason="needs Stockfish on PATH")
def test_carryover_opp_cfv_is_opponent_pov(monkeypatch) -> None:
    """Regression guard for the POV/sign bug (measured 19-39-2 / 33% H2H before the
    fix): ``opp_cfv(world)`` must be the OPPONENT-POV value, which is
    ``-eq_eval(world, OUR perspective)``. ``eq_eval`` flips terminal values by
    perspective but returns leaf (Stockfish) values UNFLIPPED, so reading the
    opponent perspective yields OUR-POV values for the (leaf-bottomed) carryover
    worlds — a sign-inverted baseline. We pin the negation relationship directly."""
    from fow_chess.engine_v2 import EngineV2
    from fow_chess.observation import observation_from_transition

    monkeypatch.setenv("FOW_RESOLVE_GADGET", "1")
    monkeypatch.setenv("FOW_RESOLVE_BLUEPRINT", "carryover")
    monkeypatch.setenv("FOW_CARRYOVER_SUBTREE", "1")

    persp = chess.WHITE
    eng = EngineV2(persp, rng=random.Random(42))
    rng = random.Random(42)
    sim = chess.Board()
    for ply in range(8):
        legal = list(sim.pseudo_legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        prev = sim.copy()
        sim.push(mv)
        obs = observation_from_transition(prev, sim, persp)
        if ply % 2 == 0:
            eng.observe_own_move(mv, obs)
        else:
            eng.observe_opp_move(obs)
    try:
        m = eng.choose_move(iterations=400, i_sample_size=16, time_budget_seconds=None)
        prev = sim.copy()
        sim.push(m)
        eng.observe_own_move(m, observation_from_transition(prev, sim, persp))
        opp = random.Random(7).choice(list(sim.pseudo_legal_moves))
        prev = sim.copy()
        sim.push(opp)
        eng.observe_opp_move(observation_from_transition(prev, sim, persp))

        vals = eng._carryover_blueprint_values()
        assert vals, "expected a non-empty carried map on-policy"

        # Independently rebuild the expected map = -(our-POV eq_eval), last-wins,
        # mirroring the helper's walk. The helper must match this exactly.
        e = eng._eq_engine
        persp_white = eng.rules.is_first_player(persp)
        expected: dict[str, float] = {}
        for r in eng._prev_root_ids:
            nc = e.node_children(r)
            if nc is None or eng._prev_played_action_key not in nc[0]:
                continue
            j_node = nc[1][nc[0].index(eng._prev_played_action_key)]
            ourpov = e.root_child_values([j_node], persp_white)[0]
            gnc = e.node_children(j_node)
            if not ourpov or gnc is None:
                continue
            for (_k, wv), gc in zip(ourpov, gnc[1]):
                nf = e.node_fen(gc)
                if nf is not None:
                    # same canonical key the helper uses (piece-only board_fen)
                    key = eng.rules.board_fen(eng.rules.board_from_fen(nf))
                    expected[key] = -wv  # opponent POV
        assert expected, "guard could not reconstruct any expected value"
        for fen, exp in expected.items():
            assert abs(vals[fen] - exp) < 1e-9, (
                f"opp_cfv sign/POV wrong: {vals[fen]} != -ourPOV({-exp}) for {fen}"
            )
    finally:
        eng.close()


@pytest.mark.skipif(_SF, reason="needs Stockfish on PATH")
def test_carryover_subtree_game_completes_no_duplicate_roots(monkeypatch) -> None:
    """Regression guard for the duplicate-root crash (56/60 games in the bakeoff).
    EPD keys are not injective on the belief (counter-twin worlds collide to one
    carried node) → duplicate root_ids → the double-counted-regrets invariant
    fires. With the dedup, a gadget + carryover + subtree-carryover game must play
    to completion through the real harness."""
    from fow_chess.engine_v2 import EngineV2Strategy
    from fow_chess.selfplay import play_game

    monkeypatch.setenv("FOW_RESOLVE_BLUEPRINT", "carryover")
    monkeypatch.setenv("FOW_CARRYOVER_SUBTREE", "1")
    for seed in (0, 2):
        white = EngineV2Strategy(seed=seed, iterations=300, i_sample_size=12,
                                 kluss_k=2, resolve_gadget=True, use_rust_state=True)
        black = EngineV2Strategy(seed=seed + 13, iterations=300, i_sample_size=12,
                                 kluss_k=2, resolve_gadget=False, use_rust_state=True)
        try:
            play_game(white, black, max_plies=40, seed=seed)  # must not raise
        finally:
            for s in (white, black):
                try:
                    s.close()
                except Exception:
                    pass


class _FakeSF:
    """Antisymmetric fake eval: returns +0.4 for white POV, -0.4 for black."""

    def evaluate(self, world, color):
        return 0.4 if color else -0.4


def test_carryover_c1_fallback_off_is_constant(monkeypatch):
    monkeypatch.delenv("FOW_GADGET_C1_FALLBACK", raising=False)
    bp = CarryoverBlueprint(ChessRules(), fallback=-0.1,
                            stockfish=_FakeSF(), opponent_color=chess.BLACK)
    assert bp.opp_cfv(chess.Board()) == -0.1  # flag off -> constant


def test_carryover_c1_fallback_uses_sf_and_prev_value(monkeypatch):
    monkeypatch.setenv("FOW_GADGET_C1_FALLBACK", "1")
    rules = ChessRules()
    bp = CarryoverBlueprint(rules, fallback=-0.1,
                            stockfish=_FakeSF(), opponent_color=chess.BLACK)
    board = chess.Board()
    # Uncarried, no v* yet: gift = sf eval in OPPONENT POV (black) = -0.4.
    assert bp.opp_cfv(board) == pytest.approx(-0.4)
    # v* = +0.2 (our POV) -> gift = max(sf_opp, -v*) = max(-0.4, -0.2) = -0.2.
    bp.set_prev_value(0.2)
    assert bp.opp_cfv(board) == pytest.approx(-0.2)
    # v* = -0.7 (we were losing) -> max(-0.4, 0.7) = 0.7 (opp can expect a lot).
    bp.set_prev_value(-0.7)
    assert bp.opp_cfv(board) == pytest.approx(0.7)
    # CARRIED worlds always use the solved value, never the C.1 fill.
    bp.set_values({rules.board_fen(board): 0.42})
    assert bp.opp_cfv(board) == pytest.approx(0.42)


def test_carryover_c1_needs_sf_and_color(monkeypatch):
    monkeypatch.setenv("FOW_GADGET_C1_FALLBACK", "1")
    bp = CarryoverBlueprint(ChessRules(), fallback=-0.1)  # no stockfish handed
    assert bp.opp_cfv(chess.Board()) == -0.1  # degrades to the constant
