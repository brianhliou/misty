"""PCFR+ correctness gate: Kuhn poker convergence.

PCFR+ (Farina, Kroer, Sandholm AAAI-21) is a predictive variant of
Regret Matching+. It has last-iterate convergence to Nash, unlike
vanilla CFR which only has average-iterate convergence. On Kuhn
poker (a 12-infoset benchmark with hand-computed equilibrium), both
solvers must reach player-0 value ≈ -1/18.

Empirical claim from the paper: PCFR+ converges meaningfully faster
than vanilla CFR (3-6 orders of magnitude on Battleship/Pursuit
after 500 iterations). Kuhn is too small to show a dramatic
difference, but we should at least see PCFR+ converging at iter=5000
to the same equilibrium value.
"""

from __future__ import annotations

import pytest

from fow_chess.cfr.tabular import solve_subgame

from test_cfr_kuhn import KuhnRoot, _leaf_eval_never_used


def test_kuhn_pcfr_plus_converges_to_neg_1_over_18():
    """PCFR+ on Kuhn poker reaches the same equilibrium value as vanilla CFR."""
    root = KuhnRoot()
    solution = solve_subgame(
        root,
        depth=100,
        leaf_eval=_leaf_eval_never_used,
        iterations=5000,
        value_estimate_samples=5000,
        players=(0, 1),
        predictive=True,
    )
    target = -1.0 / 18.0
    assert abs(solution.value_at_root - target) < 0.05, (
        f"PCFR+ value_at_root={solution.value_at_root:.4f} target={target:.4f} "
        f"info_sets={solution.info_set_count}"
    )


def test_kuhn_pcfr_plus_info_set_count_is_12():
    """PCFR+ visits the same 12 info sets as vanilla CFR."""
    root = KuhnRoot()
    solution = solve_subgame(
        root,
        depth=100,
        leaf_eval=_leaf_eval_never_used,
        iterations=200,
        value_estimate_samples=10,
        players=(0, 1),
        predictive=True,
    )
    assert solution.info_set_count == 12


def test_kuhn_pcfr_plus_root_strategy_valid_distribution():
    """Last-iterate strategy at the root should be a valid distribution
    (sums to ~1, all components ≥ 0). PCFR+ guarantees last-iterate
    convergence, so the final strategy is meaningful — not just the
    average."""
    root = KuhnRoot()
    solution = solve_subgame(
        root,
        depth=100,
        leaf_eval=_leaf_eval_never_used,
        iterations=1000,
        value_estimate_samples=100,
        players=(0, 1),
        predictive=True,
    )
    # Root is chance, so strategy_at_root is empty by convention. Take a
    # spot-check on a decision node instead by solving a subgame rooted
    # at a known KuhnDecisionNode.
    from test_cfr_kuhn import KuhnDecisionNode
    decision_root = KuhnDecisionNode(
        card_p0=2,  # King — strongest hand
        card_p1=0,  # Jack — opp's actual card; treated as known here
        history=(),
        depth=0,
    )
    sol = solve_subgame(
        decision_root,
        depth=100,
        leaf_eval=_leaf_eval_never_used,
        iterations=1000,
        value_estimate_samples=10,
        players=(0, 1),
        predictive=True,
    )
    probs = list(sol.strategy_at_root.values())
    assert all(p >= 0 for p in probs), f"negative prob in last iterate: {probs}"
    assert abs(sum(probs) - 1.0) < 1e-6, f"probs sum to {sum(probs)}"


@pytest.mark.parametrize("iters", [200, 1000, 5000])
def test_kuhn_pcfr_plus_value_improves_with_iterations(iters):
    """Sanity: PCFR+ value-at-root estimate gets closer to target as
    iterations increase. We don't compare iteration-to-iteration here
    (would need fresh state per run); instead we check the absolute
    error shrinks at higher iter counts."""
    root = KuhnRoot()
    sol = solve_subgame(
        root,
        depth=100,
        leaf_eval=_leaf_eval_never_used,
        iterations=iters,
        value_estimate_samples=2000,
        players=(0, 1),
        predictive=True,
    )
    target = -1.0 / 18.0
    err = abs(sol.value_at_root - target)
    # Generous bounds — Monte Carlo noise at low iter counts is real
    bounds = {200: 0.18, 1000: 0.10, 5000: 0.05}
    assert err < bounds[iters], (
        f"iters={iters} err={err:.4f} bound={bounds[iters]} "
        f"value={sol.value_at_root:.4f}"
    )
