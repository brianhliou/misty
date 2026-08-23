"""Tests for the read-only Resolve gadget integration (gt_cfr Slice 1).

Two layers:
  * Pure cap-logic tests (FakeEng, no Stockfish/Rust) — validate that the
    per-world blueprint cap flips an aggregation-dilution blunder to the safe
    move, the 74126ceb/a84dbaf9 fix in miniature.
  * End-to-end OFF-parity + ON-determinism (need Stockfish + WS2 EqEngine).
"""
from __future__ import annotations

import random
import shutil

import chess
import pytest

import fow_rust
from fow_chess.cfr import gt_cfr
from fow_chess.cfr.blueprint import StubBlueprint
from fow_chess.cfr.gt_cfr import solve_multiroot_rust_tree
from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval


class _FakeEng:
    """Minimal stand-in exposing root_child_values for the cap-logic tests.

    Three worlds, two actions. Action 1 wins big (+0.5) in the two 'safe'
    worlds but is punished (-0.5) in world 2 (the truth, where a hidden
    defender refutes it). Action 2 is a modest, uniformly-safe move (+0.05).
    The uniform belief-mean of action 1 = (0.5 + 0.5 - 0.5)/3 = +0.167 > action
    2 (+0.05) — so WITHOUT the gadget the dilution picks the blunder.
    """

    def root_child_values(self, root_ids, perspective_white):
        return [
            [(1, 0.5), (2, 0.05)],
            [(1, 0.5), (2, 0.05)],
            [(1, -0.5), (2, 0.05)],
        ]


def test_gadget_maxmargin_flips_dilution_blunder_to_safe_move(monkeypatch):
    monkeypatch.setattr(gt_cfr._DEFAULT_RULES, "decode_action_key", lambda k: k)  # identity key->move
    bp = StubBlueprint(opp_cfv=0.0)  # bp_value=0 -> margin M(a,j) == value(a,j)
    strat, av, val = gt_cfr._apply_resolve_gadget(_FakeEng(), [0, 1, 2], [1, 2], True, bp, 0.0)
    # action 1 has a -0.5 world (worst-case margin -0.5, exploitable); action 2
    # is safe everywhere (worst-case margin +0.05). A safe action exists -> the
    # Maxmargin regime picks the highest worst-case margin = the safe action.
    assert av[1] == pytest.approx(-0.5)  # worst-case (min) margin, NOT a mean
    assert av[2] == pytest.approx(0.05)
    assert strat == {2: 1.0}  # avoids the dilution blunder


def test_gadget_maxmargin_no_overcaution_takes_bigger_safe_win(monkeypatch):
    monkeypatch.setattr(gt_cfr._DEFAULT_RULES, "decode_action_key", lambda k: k)

    class _BothWin:
        def root_child_values(self, root_ids, perspective_white):
            return [[(1, 0.6), (2, 0.1)], [(1, 0.5), (2, 0.1)]]  # both safe; 1 wins more

    bp = StubBlueprint(opp_cfv=0.0)
    strat, av, val = gt_cfr._apply_resolve_gadget(_BothWin(), [0, 1], [1, 2], True, bp, 0.0)
    # both safe (worst-case margins +0.5 and +0.1) -> Maxmargin takes the bigger
    # one. The gadget is NOT blindly defensive: it claims a real advantage.
    assert strat == {1: 1.0}


def test_gadget_resolve_regime_picks_least_exploitable(monkeypatch):
    monkeypatch.setattr(gt_cfr._DEFAULT_RULES, "decode_action_key", lambda k: k)

    class _NoneSafe:
        def root_child_values(self, root_ids, perspective_white):
            return [[(1, -0.1), (2, -0.4)], [(1, -0.2), (2, -0.05)]]  # both exploitable

    bp = StubBlueprint(opp_cfv=0.0)
    # cvar_q=1.0 = the full-belief limit (CVaR over ALL worlds = the mean), which
    # is the original Resolve objective. (The default cvar_q=0.1 reduces to the
    # single worst world here since there are only 2 worlds — kq=1.)
    strat, av, val = gt_cfr._apply_resolve_gadget(
        _NoneSafe(), [0, 1], [1, 2], True, bp, 0.0, cvar_q=1.0)
    # no fully-safe action (full-belief margins -0.15 and -0.225) -> Resolve
    # regime, mean of negative-truncated margins:
    #   resolve(1) = (-0.1 + -0.2)/2 = -0.15 ; resolve(2) = (-0.4 + -0.05)/2 = -0.225
    assert av[1] == pytest.approx(-0.15)
    assert av[2] == pytest.approx(-0.225)
    assert strat == {1: 1.0}  # least exploitable


def test_gadget_skips_actions_illegal_in_all_worlds(monkeypatch):
    monkeypatch.setattr(gt_cfr._DEFAULT_RULES, "decode_action_key", lambda k: k)

    class _OneAction:
        def root_child_values(self, root_ids, perspective_white):
            return [[(1, 0.1)], [(1, 0.1)]]  # action 2 absent everywhere

    bp = StubBlueprint(opp_cfv=0.0)
    strat, av, val = gt_cfr._apply_resolve_gadget(_OneAction(), [0, 1], [1, 2], True, bp, 0.0)
    assert set(av) == {1}  # action 2 had no legal world -> dropped
    assert strat == {1: 1.0}


# --- CVaR (worst-q-fraction) robustness at scale --------------------------------
# These exercise the two matrix-NOISE failure modes the oracle de-risk found on
# 74126ceb (lab/debug_gadget_oracle_derisk.py): a single spurious-loss world kills
# pure Maxmargin (min), and a spread-out catastrophe gets washed out of the
# full-belief Resolve mean. CVaR over the worst q-fraction is robust to both.


class _Worlds:
    """Per-world value matrix fake (key 1 / key 2 per world)."""

    def __init__(self, matrix):
        self._m = matrix  # list[dict[int, float]]

    def root_child_values(self, root_ids, perspective_white):
        return [list(d.items()) for d in self._m]


def test_gadget_cvar_dilutes_single_spurious_outlier(monkeypatch):
    """One spurious -1.0 world should not condemn an otherwise-safe move.

    key 1 ('safe'): -0.02 in 19 worlds, -1.0 in ONE (a noise/under-searched
    world). key 2 ('worse'): -0.3 everywhere. Pure worst-case (cvar_q->0) is
    fooled by the outlier and picks the genuinely-worse move; CVaR over the worst
    20% averages the outlier away and picks the safe move."""
    monkeypatch.setattr(gt_cfr._DEFAULT_RULES, "decode_action_key", lambda k: k)
    matrix = [{1: -1.0, 2: -0.3}] + [{1: -0.02, 2: -0.3}] * 19
    bp = StubBlueprint(opp_cfv=0.0)
    rids = list(range(20))

    # cvar_q -> 0 (kq=1) = pure worst-case: the single -1.0 world makes key 1's
    # min worse than key 2 -> picks the genuinely-worse key 2 (the failure mode).
    strat0, _, _ = gt_cfr._apply_resolve_gadget(
        _Worlds(matrix), rids, [1, 2], True, bp, 0.0, cvar_q=1e-9)
    assert strat0 == {2: 1.0}

    # cvar_q = 0.2 (kq=4): key 1 worst-4 = (-1.0 - 0.02*3)/4 = -0.265 beats
    # key 2's -0.3 -> the outlier is diluted, picks the safe key 1.
    strat, av, _ = gt_cfr._apply_resolve_gadget(
        _Worlds(matrix), rids, [1, 2], True, bp, 0.0, cvar_q=0.2)
    assert av[1] == pytest.approx(-0.265)
    assert av[2] == pytest.approx(-0.3)
    assert strat == {1: 1.0}


def test_gadget_cvar_captures_spread_catastrophe(monkeypatch):
    """A move that wins in the majority but loses badly in a minority must not be
    washed out by the full-belief mean — the 74126ceb Qxc6 re-dilution in
    miniature.

    key 1 ('blunder'): +0.3 in 18 worlds, -0.36 in 2 (the catastrophe, even
    under-scored). key 2 ('safe'): -0.05 everywhere. The full-belief mean
    (cvar_q=1) has a POSITIVE worst-tail for the blunder (mean +0.234) -> it goes
    Maxmargin and picks the blunder. CVaR over the worst 10% sees only the
    catastrophe worlds and picks the safe move."""
    monkeypatch.setattr(gt_cfr._DEFAULT_RULES, "decode_action_key", lambda k: k)
    matrix = [{1: 0.3, 2: -0.05}] * 18 + [{1: -0.36, 2: -0.05}] * 2
    bp = StubBlueprint(opp_cfv=0.0)
    rids = list(range(20))

    # cvar_q = 1.0 (full-belief mean): the 18 winning worlds drown the 2
    # catastrophes -> re-dilution, picks the blunder key 1.
    strat1, _, _ = gt_cfr._apply_resolve_gadget(
        _Worlds(matrix), rids, [1, 2], True, bp, 0.0, cvar_q=1.0)
    assert strat1 == {1: 1.0}

    # cvar_q = 0.1 (kq=2): key 1 worst-2 = -0.36 < key 2's -0.05 -> Resolve
    # regime picks the safe key 2.
    strat, av, _ = gt_cfr._apply_resolve_gadget(
        _Worlds(matrix), rids, [1, 2], True, bp, 0.0, cvar_q=0.1)
    assert av[1] == pytest.approx(-0.36)
    assert av[2] == pytest.approx(-0.05)
    assert strat == {2: 1.0}


# --- end-to-end (need Stockfish + WS2) ---------------------------------------

pytestmark_e2e = pytest.mark.skipif(
    shutil.which("stockfish") is None or not hasattr(fow_rust.EqEngine, "root_child_values"),
    reason="needs stockfish on PATH + WS2 EqEngine.root_child_values",
)


def _boards():
    b1 = chess.Board()
    b2 = chess.Board()
    b2.push(chess.Move.from_uci("e2e4"))
    b2.push(chess.Move.from_uci("e7e5"))
    return [b1, b2]


@pytestmark_e2e
def test_gadget_off_matches_baseline():
    """resolve_gadget=False must be byte-identical to the no-gadget call."""
    with StockfishLeafEval() as sf:
        base = solve_multiroot_rust_tree(
            [b.copy() for b in _boards()], stockfish_eval=sf, perspective=chess.WHITE,
            iterations=12, expansion_budget=20, rng=random.Random(7),
        )
        off = solve_multiroot_rust_tree(
            [b.copy() for b in _boards()], stockfish_eval=sf, perspective=chess.WHITE,
            iterations=12, expansion_budget=20, rng=random.Random(7),
            resolve_gadget=False, gadget_blueprint=StubBlueprint(),
        )
    assert set(base.strategy_at_root) == set(off.strategy_at_root)
    for mv, p in base.strategy_at_root.items():
        assert off.strategy_at_root[mv] == pytest.approx(p, abs=1e-12)


@pytestmark_e2e
def test_gadget_on_returns_deterministic_strategy():
    """resolve_gadget=True (Resolve regime) purifies to a single top action."""
    with StockfishLeafEval() as sf:
        sol = solve_multiroot_rust_tree(
            [b.copy() for b in _boards()], stockfish_eval=sf, perspective=chess.WHITE,
            iterations=12, expansion_budget=20, rng=random.Random(7),
            full_cfv_backprop=True,
            resolve_gadget=True, gadget_blueprint=StubBlueprint(opp_cfv=0.0),
        )
    assert len(sol.strategy_at_root) == 1
    assert next(iter(sol.strategy_at_root.values())) == pytest.approx(1.0)


@pytestmark_e2e
def test_iterative_gadget_off_is_byte_identical_to_readonly():
    """gadget_iterative=False (or unset) must leave the read-only gadget path
    byte-identical — the proper gadget is strictly opt-in."""
    def _run(**extra):
        with StockfishLeafEval() as sf:
            return solve_multiroot_rust_tree(
                [b.copy() for b in _boards()], stockfish_eval=sf,
                perspective=chess.WHITE, iterations=12, expansion_budget=20,
                rng=random.Random(7), full_cfv_backprop=True, resolve_gadget=True,
                gadget_blueprint=StubBlueprint(opp_cfv=0.0), **extra,
            )
    base = _run()
    off = _run(gadget_iterative=False)
    assert set(base.strategy_at_root) == set(off.strategy_at_root)
    for mv, p in base.strategy_at_root.items():
        assert off.strategy_at_root[mv] == pytest.approx(p, abs=1e-12)


@pytestmark_e2e
def test_iterative_gadget_deterministic_distribution():
    """The PROPER (iterative) gadget bakes the safe strategy INTO the weighted
    regrets, so it reads out the raw root distribution (purification to top-1 is
    done later by choose_move) — a valid distribution, deterministic at fixed
    iters (the property the king-risk rig depends on)."""
    def _run():
        with StockfishLeafEval() as sf:
            return solve_multiroot_rust_tree(
                [b.copy() for b in _boards()], stockfish_eval=sf,
                perspective=chess.WHITE, iterations=20, expansion_budget=20,
                rng=random.Random(7), full_cfv_backprop=True, resolve_gadget=True,
                gadget_blueprint=StubBlueprint(opp_cfv=0.0), gadget_iterative=True,
            )
    a = _run()
    b = _run()
    assert a.strategy_at_root  # non-empty
    assert all(p >= 0.0 for p in a.strategy_at_root.values())
    assert sum(a.strategy_at_root.values()) == pytest.approx(1.0, abs=1e-9)
    # Deterministic at fixed iters.
    assert set(a.strategy_at_root) == set(b.strategy_at_root)
    for mv, p in a.strategy_at_root.items():
        assert b.strategy_at_root[mv] == pytest.approx(p, abs=1e-12)


# ---------------------------------------------------------------------------
# Non-uniform Resolve root distribution alpha(J) = ½(y/Σy + 1/m) (paper C.3)
# ---------------------------------------------------------------------------

class _ReachBlueprint:
    """Fake blueprint with a dict-backed carried reach keyed by board_fen."""

    def __init__(self, reach_by_fen, rules):
        self._r = reach_by_fen
        self._rules = rules

    def opp_cfv(self, world):
        return -0.1

    def reach(self, world):
        return self._r.get(self._rules.board_fen(world), 0.0)


def test_nonuniform_resolve_alpha_math():
    from fow_chess.rules import ChessRules
    rules = ChessRules()
    b1 = chess.Board()
    b2 = chess.Board()
    b2.push(chess.Move.from_uci("e2e4"))
    bp = _ReachBlueprint({rules.board_fen(b2): 0.8, rules.board_fen(b1): 0.2}, rules)
    alpha = gt_cfr._nonuniform_resolve_alpha(bp, [b1, b2])
    # alpha(J) = ½(y/Σy + 1/m): m=2, y=[0.2, 0.8] -> [½(0.2+0.5), ½(0.8+0.5)]
    assert alpha == pytest.approx([0.35, 0.65])
    assert sum(alpha) == pytest.approx(1.0)


def test_nonuniform_resolve_alpha_fallbacks_to_uniform_none():
    from fow_chess.rules import ChessRules
    rules = ChessRules()
    b = chess.Board()
    # No reach method (StubBlueprint) -> None.
    assert gt_cfr._nonuniform_resolve_alpha(StubBlueprint(), [b]) is None
    # Zero coverage (sample missed every carried world) -> None.
    assert gt_cfr._nonuniform_resolve_alpha(_ReachBlueprint({}, rules), [b]) is None
    # Empty worlds -> None.
    assert gt_cfr._nonuniform_resolve_alpha(_ReachBlueprint({}, rules), []) is None


def test_carryover_reach_accumulates_across_prior_roots():
    """y(J) is a reach probability: a world reachable from several prior roots
    must SUM their strategy mass (paper C.3 'distribution of infosets generated
    from the opponent strategy'), not keep one arbitrary path's value."""
    from fow_chess.engine_v2 import EngineV2
    from fow_chess.rules import ChessRules

    fen_a = chess.Board().fen()                     # same placement reachable twice
    b_b = chess.Board(); b_b.push(chess.Move.from_uci("e2e4"))
    fen_b = b_b.fen()

    class _FakeEqEng:
        # prior roots 10, 20; played action key 99; J nodes 11, 21.
        _children = {
            10: ([99], [11]), 20: ([99], [21]),
            11: ([1, 2], [12, 13]), 21: ([1], [22]),
        }
        _infosets = {11: 111, 21: 222}
        _strats = {111: [0.7, 0.3], 222: [1.0]}
        _fens = {12: fen_a, 13: fen_b, 22: fen_a}

        def node_children(self, nid):
            return self._children.get(nid)

        def node_infoset(self, nid):
            return self._infosets.get(nid)

        def current_strategy(self, infoset, keys):
            return self._strats[infoset]

        def node_fen(self, nid):
            return self._fens.get(nid)

    eng = object.__new__(EngineV2)
    eng._eq_engine = _FakeEqEng()
    eng._prev_root_ids = [10, 20]
    eng._prev_played_action_key = 99
    eng.rules = ChessRules()

    reach = eng._carryover_blueprint_reach()
    key_a = eng.rules.board_fen(chess.Board())
    key_b = eng.rules.board_fen(b_b)
    assert reach[key_a] == pytest.approx(1.7)  # 0.7 (root 10) + 1.0 (root 20)
    assert reach[key_b] == pytest.approx(0.3)


@pytestmark_e2e
def test_iterative_gadget_with_alpha_deterministic_distribution():
    """PROPER gadget + non-uniform alpha (the faithful combo): runs, yields a
    valid root distribution, deterministic at fixed iters."""
    from fow_chess.rules import ChessRules
    rules = ChessRules()
    boards = _boards()
    bp = _ReachBlueprint({rules.board_fen(boards[1]): 1.0}, rules)

    def _run():
        with StockfishLeafEval() as sf:
            return solve_multiroot_rust_tree(
                [b.copy() for b in _boards()], stockfish_eval=sf,
                perspective=chess.WHITE, iterations=20, expansion_budget=20,
                rng=random.Random(7), full_cfv_backprop=True, resolve_gadget=True,
                gadget_blueprint=bp, gadget_iterative=True, gadget_alpha=True,
            )
    a = _run()
    b = _run()
    assert a.strategy_at_root
    assert all(p >= 0.0 for p in a.strategy_at_root.values())
    assert sum(a.strategy_at_root.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(a.strategy_at_root) == set(b.strategy_at_root)
    for mv, p in a.strategy_at_root.items():
        assert b.strategy_at_root[mv] == pytest.approx(p, abs=1e-12)


@pytestmark_e2e
def test_iterative_gadget_fused_eval_runs_and_is_deterministic(monkeypatch):
    """FOW_GADGET_FUSED_EVAL: the fused weighted pass (values returned by the
    pass itself, gadget steps after, weights lag one iter) yields a valid,
    deterministic root distribution. Not byte-identical to unfused (the one-iter
    weight lag is a documented semantic difference, rig-validated)."""
    monkeypatch.setenv("FOW_GADGET_FUSED_EVAL", "1")

    def _run():
        with StockfishLeafEval() as sf:
            return solve_multiroot_rust_tree(
                [b.copy() for b in _boards()], stockfish_eval=sf,
                perspective=chess.WHITE, iterations=20, expansion_budget=20,
                rng=random.Random(7), full_cfv_backprop=True, resolve_gadget=True,
                gadget_blueprint=StubBlueprint(opp_cfv=0.0), gadget_iterative=True,
            )
    a = _run()
    b = _run()
    assert a.strategy_at_root
    assert sum(a.strategy_at_root.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(a.strategy_at_root) == set(b.strategy_at_root)
    for mv, p in a.strategy_at_root.items():
        assert b.strategy_at_root[mv] == pytest.approx(p, abs=1e-12)


@pytestmark_e2e
def test_iterative_gadget_fused_off_is_byte_identical(monkeypatch):
    """With FOW_GADGET_FUSED_EVAL unset, the iterative gadget must be
    byte-identical to its pre-fuse behavior (the unfused dispatch path)."""
    monkeypatch.delenv("FOW_GADGET_FUSED_EVAL", raising=False)

    def _run():
        with StockfishLeafEval() as sf:
            return solve_multiroot_rust_tree(
                [b.copy() for b in _boards()], stockfish_eval=sf,
                perspective=chess.WHITE, iterations=20, expansion_budget=20,
                rng=random.Random(7), full_cfv_backprop=True, resolve_gadget=True,
                gadget_blueprint=StubBlueprint(opp_cfv=0.0), gadget_iterative=True,
            )
    a = _run()
    b = _run()
    for mv, p in a.strategy_at_root.items():
        assert b.strategy_at_root[mv] == pytest.approx(p, abs=1e-12)


@pytestmark_e2e
def test_iterative_gadget_merged_runs_and_is_deterministic(monkeypatch):
    """FOW_GADGET_MERGED: gadget update on the clean pre-pass snapshot (no lag)
    + one merged weighted walk. Valid distribution, deterministic at fixed
    iters. Different CFR update scheme -> not byte-identical to the two-pass."""
    monkeypatch.setenv("FOW_GADGET_MERGED", "1")

    def _run():
        with StockfishLeafEval() as sf:
            return solve_multiroot_rust_tree(
                [b.copy() for b in _boards()], stockfish_eval=sf,
                perspective=chess.WHITE, iterations=20, expansion_budget=20,
                rng=random.Random(7), full_cfv_backprop=True, resolve_gadget=True,
                gadget_blueprint=StubBlueprint(opp_cfv=0.0), gadget_iterative=True,
            )
    a = _run()
    b = _run()
    assert a.strategy_at_root
    assert sum(a.strategy_at_root.values()) == pytest.approx(1.0, abs=1e-9)
    for mv, p in a.strategy_at_root.items():
        assert b.strategy_at_root[mv] == pytest.approx(p, abs=1e-12)


@pytestmark_e2e
def test_iterative_gadget_commit_cvar_filter(monkeypatch):
    """FOW_GADGET_COMMIT_CVAR: with the iterative gadget on, the commit readout
    passes through the read-only CVaR filter (deterministic top-1) instead of
    the raw root strategy. Off -> prior behavior (raw distribution)."""
    def _run(env_on):
        if env_on:
            monkeypatch.setenv("FOW_GADGET_COMMIT_CVAR", "1")
        else:
            monkeypatch.delenv("FOW_GADGET_COMMIT_CVAR", raising=False)
        with StockfishLeafEval() as sf:
            return solve_multiroot_rust_tree(
                [b.copy() for b in _boards()], stockfish_eval=sf,
                perspective=chess.WHITE, iterations=20, expansion_budget=20,
                rng=random.Random(7), full_cfv_backprop=True, resolve_gadget=True,
                gadget_blueprint=StubBlueprint(opp_cfv=0.0), gadget_iterative=True,
            )
    on = _run(True)
    assert len(on.strategy_at_root) == 1          # CVaR filter purifies top-1
    assert next(iter(on.strategy_at_root.values())) == pytest.approx(1.0)
    off = _run(False)
    assert sum(off.strategy_at_root.values()) == pytest.approx(1.0, abs=1e-9)
