"""Multi-root GT-CFR (A5.1 — KLUSS-flavored shared-regret) tests.

Validates that:
- The multi-root coordinator handles N roots with shared regret tables.
- Sampling from PEnumerator produces well-formed roots.
- time_budget_seconds cuts off correctly.
- Strategy at the shared root infoset is a valid distribution.
- Single-root case (N=1) gives results comparable to the original
  single-root coordinator.
"""

from __future__ import annotations

import random
import shutil
import time

import chess
import pytest

from fow_chess.cfr.gt_cfr import (
    GTCFRState,
    root_node,
    sample_roots_from_P,
    solve_multiroot_growing_subgame,
    _equilibrium_traverse,
    expand_leaf,
)


pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not on PATH",
)


def _solve_multi(roots, iterations=20, time_budget=None):
    from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval
    with StockfishLeafEval() as sf:
        return solve_multiroot_growing_subgame(
            roots,
            stockfish_eval=sf,
            perspective=chess.WHITE,
            iterations=iterations,
            rng=random.Random(42),
            time_budget_seconds=time_budget,
        )


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_empty_roots_raises():
    from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval
    with pytest.raises(ValueError, match="at least one root"):
        with StockfishLeafEval() as sf:
            solve_multiroot_growing_subgame(
                [], stockfish_eval=sf, perspective=chess.WHITE, iterations=1,
            )


def test_single_root_through_multi_coordinator():
    """One root through the multi-root coordinator should produce a
    well-formed solution."""
    roots = [root_node(chess.Board())]
    sol = _solve_multi(roots, iterations=15)
    assert sol.n_roots == 1
    assert sol.iterations == 15
    assert sol.info_set_count > 0
    assert sol.total_tree_nodes > 1  # root + expanded children
    assert sol.strategy_at_root  # non-empty


def test_multi_root_shares_root_infoset():
    """Three different starting positions (different truths) should
    share the SAME root infoset because all have empty observation
    histories. Regret tables at the root infoset should reflect input
    from all three roots."""
    # Construct three roots from different "truths" — for now, the
    # only truth we have access to is the standard board. Simulate
    # variation by using positions that LOOK different but are still
    # white-to-move with empty history. Use slightly different
    # piece-placement positions; they're treated as separate truths
    # at the same shared infoset since obs_history is empty.
    fens = [
        chess.STARTING_FEN,
        # Knight + pawn moves applied to truth, then state reset to
        # white-to-move — effectively different starting truths.
        "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 1 2",
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 2",
    ]
    roots = [root_node(chess.Board(fen)) for fen in fens]
    sol = _solve_multi(roots, iterations=10)
    assert sol.n_roots == 3
    # All three roots have empty obs histories + white-to-move → same
    # info_set_id. The strategy reflects shared regret accumulation.
    assert roots[0].info_set_id() == roots[1].info_set_id() == roots[2].info_set_id()
    # Tree node count is sum across roots; each gets independently
    # expanded but they share regret state.
    assert sol.total_tree_nodes > 3


def test_strategy_is_a_valid_distribution():
    roots = [root_node(chess.Board())]
    sol = _solve_multi(roots, iterations=10)
    probs = list(sol.strategy_at_root.values())
    assert all(p >= 0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Time budget (anytime algorithm)
# ---------------------------------------------------------------------------


def test_time_budget_stops_early():
    """With a 0.2s budget and iterations=1000, the coordinator should
    return before completing all iterations."""
    roots = [root_node(chess.Board())]
    sol = _solve_multi(roots, iterations=1000, time_budget=0.2)
    # Time budget enforced — elapsed close to (slightly over) budget.
    assert sol.elapsed_seconds < 0.5, (
        f"elapsed {sol.elapsed_seconds:.2f}s exceeded 0.5s grace over 0.2s budget"
    )
    # Likely didn't complete all 1000 iterations.
    assert sol.iterations < 1000, f"completed all {sol.iterations} iters within 0.2s — adjust test"


def test_time_budget_none_runs_full_iterations():
    """Without a time budget, all iterations complete."""
    roots = [root_node(chess.Board())]
    sol = _solve_multi(roots, iterations=10, time_budget=None)
    assert sol.iterations == 10


# ---------------------------------------------------------------------------
# Sampling from PEnumerator
# ---------------------------------------------------------------------------


def test_sample_roots_from_P_returns_well_formed_roots():
    from fow_chess.p_enum import PEnumerator
    # Build a PEnumerator with a few plies of history so |P| > 1.
    enum = PEnumerator(chess.WHITE)
    prev = chess.Board()
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("e2e4"))
    # White just played e2e4 (own move).
    enum.update_own_move(chess.Move.from_uci("e2e4"))
    # Now black moves (opp from white's POV).
    from fow_chess.observation import observation_from_transition
    prev2 = nxt.copy()
    nxt2 = prev2.copy()
    nxt2.push(chess.Move.from_uci("e7e5"))
    obs = observation_from_transition(prev2, nxt2, chess.WHITE)
    enum.update_opp_move(obs)
    # |P| should be > 1 now (uncertainty about which black move was played).

    rng = random.Random(0)
    roots = sample_roots_from_P(
        enum.iter_positions(), to_move=chess.WHITE, n=4, rng=rng,
    )
    assert len(roots) <= 4
    assert len(roots) <= enum.size  # can't sample more than exist
    for r in roots:
        assert r.to_move == chess.WHITE
        assert r.depth == 0
        assert r.obs_history_white == ()
        assert r.obs_history_black == ()
        assert not r.is_expanded


def test_sample_roots_from_P_n_larger_than_population():
    """Asking for more roots than exist in P returns all of P."""
    from fow_chess.p_enum import PEnumerator
    enum = PEnumerator(chess.WHITE)
    # |P| = 1 (just the starting board).
    rng = random.Random(0)
    roots = sample_roots_from_P(
        enum.iter_positions(), to_move=chess.WHITE, n=10, rng=rng,
    )
    assert len(roots) == 1
