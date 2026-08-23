"""End-to-end smoke test: tabular CFR on a real FoW subgame.

This complements the Kuhn poker validation in ``test_cfr_kuhn.py``. Kuhn proves
the algorithm converges to a known Nash value on a 12-info-set imperfect-info
game. This test proves the same algorithm runs end-to-end on the FoW
``SubgameNode`` substrate without crashing, produces sensible values, and
populates the regret tables.

This is a behavioral smoke test, not a correctness gate — there is no known
ground-truth equilibrium value for an arbitrary FoW position.
"""

from __future__ import annotations

import time

import chess

from fow_chess.cfr.leaf_eval import material_leaf_eval
from fow_chess.cfr.tabular import solve_subgame
from fow_chess.cfr.walker import SubgameNode


def _start_root() -> SubgameNode:
    return SubgameNode.root(chess.Board())


def test_cfr_runs_on_start_position_depth_2():
    """CFR should complete on a depth-2 subgame from the start position."""
    root = _start_root()
    t0 = time.monotonic()
    solution = solve_subgame(
        root,
        depth=2,
        leaf_eval=material_leaf_eval,
        iterations=50,  # small for smoke; convergence not asserted
        value_estimate_samples=50,
    )
    wall = time.monotonic() - t0

    # Sensible value range (we're using tanh-normalized material in [-1, 1]
    # plus possible terminal ±1).
    assert -1.0 <= solution.value_at_root <= 1.0, (
        f"value out of [-1,1]: {solution.value_at_root}"
    )

    # Strategy at root must distribute mass over the 20 starting moves.
    assert len(solution.strategy_at_root) == 20
    total_prob = sum(solution.strategy_at_root.values())
    assert abs(total_prob - 1.0) < 1e-6

    # Info sets visited should be at least the root + a handful of depth-1 nodes.
    # (Exact count depends on external sampling RNG; just assert non-trivial.)
    assert solution.info_set_count > 1

    # Wall time sanity bound — way under what production tolerances will need
    # but useful as a regression check.
    assert wall < 30.0, f"smoke too slow: {wall:.2f}s"


def test_cfr_value_at_start_is_near_zero():
    """At the start position, equilibrium value should be near 0 (no material asymmetry).

    Not a strict assertion — but a big positive or negative bias would suggest
    a sign error or a non-zero-sum bug.
    """
    root = _start_root()
    solution = solve_subgame(
        root,
        depth=2,
        leaf_eval=material_leaf_eval,
        iterations=100,
        value_estimate_samples=200,
    )
    # Starting position is symmetric; equilibrium under material eval should be
    # close to 0 — bounded above by what 2-ply material exchange can produce.
    assert abs(solution.value_at_root) < 0.5, (
        f"start position equilibrium value too biased: {solution.value_at_root}"
    )


def test_cfr_strategy_uses_average_not_current():
    """The returned strategy should be the average (Nash-convergent), not last-iteration current.

    Sanity check: at the start position, the average strategy should not be a
    delta function on a single move (which would happen if we accidentally
    returned the current strategy after enough iterations let regrets pile up).
    """
    root = _start_root()
    solution = solve_subgame(
        root,
        depth=2,
        leaf_eval=material_leaf_eval,
        iterations=100,
        value_estimate_samples=10,
    )
    max_prob = max(solution.strategy_at_root.values())
    # The average strategy should give SOME spread across actions early in
    # training. A delta of 1.0 on one move = we're definitely not using the
    # average correctly.
    assert max_prob < 0.99, f"strategy looks like delta: {solution.strategy_at_root}"
