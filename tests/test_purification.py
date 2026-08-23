"""Tests for A6.1 strategy purification (Obscuro Appendix C.8)."""

from __future__ import annotations

import pytest

from fow_chess.cfr.purification import (
    PurifiedStrategy,
    purify_strategy,
    select_regime,
)


# ---------------------------------------------------------------------------
# Empty / boundary cases
# ---------------------------------------------------------------------------


def test_empty_strategy_returns_empty():
    result = purify_strategy({}, [], 0)
    assert result.strategy == {}
    assert result.n_actions == 0


def test_max_actions_must_be_positive():
    with pytest.raises(ValueError):
        purify_strategy({"a": 1.0}, [{"a": 1.0}], 0, max_actions=0)


# ---------------------------------------------------------------------------
# max_actions=1 (Resolve regime — deterministic top action)
# ---------------------------------------------------------------------------


def test_resolve_regime_picks_top_action():
    """Top action by last-iterate probability is always returned with
    probability 1, regardless of support history."""
    last = {"e2e4": 0.6, "d2d4": 0.3, "g1f3": 0.1}
    history = [last] * 10
    result = purify_strategy(last, history, t_half=5, max_actions=1)
    assert result.strategy == {"e2e4": 1.0}
    assert result.n_actions == 1
    assert result.excluded_unstable == []


def test_resolve_regime_top_action_with_zero_other_actions():
    """Single-action strategy purifies to itself."""
    last = {"e2e4": 1.0}
    history = [last]
    result = purify_strategy(last, history, t_half=0, max_actions=1)
    assert result.strategy == {"e2e4": 1.0}


# ---------------------------------------------------------------------------
# max_actions=3 (Maxmargin regime — top-m with stable filter)
# ---------------------------------------------------------------------------


def test_maxmargin_includes_stable_top_actions():
    """If actions 2 and 3 were in support continuously after t_half,
    include them up to max_actions=3."""
    last = {"e2e4": 0.5, "d2d4": 0.3, "g1f3": 0.2}
    # 10 iterations; t_half=5. Iterations 6,7,8,9 must all have these
    # three actions with prob > 0.
    history = [last] * 10
    result = purify_strategy(last, history, t_half=5, max_actions=3)
    assert result.n_actions == 3
    assert set(result.strategy.keys()) == {"e2e4", "d2d4", "g1f3"}
    # Renormalized — should sum to 1.
    assert abs(sum(result.strategy.values()) - 1.0) < 1e-9
    # Proportions preserved from last-iterate.
    assert result.strategy["e2e4"] == pytest.approx(0.5)
    assert result.strategy["d2d4"] == pytest.approx(0.3)
    assert result.strategy["g1f3"] == pytest.approx(0.2)


def test_maxmargin_excludes_unstable_secondary_actions():
    """An action that drops out of support between t_half and the end
    should be excluded by the stable-actions filter."""
    last = {"e2e4": 0.5, "d2d4": 0.3, "g1f3": 0.2}
    # d2d4 drops to 0 at iter 7 (post-t_half=5).
    history = []
    for t in range(10):
        if t == 7:
            history.append({"e2e4": 0.6, "g1f3": 0.4})  # d2d4 gone
        else:
            history.append(last)
    result = purify_strategy(last, history, t_half=5, max_actions=3)
    assert "d2d4" not in result.strategy
    assert "d2d4" in result.excluded_unstable
    assert "e2e4" in result.strategy  # always (top)
    assert "g1f3" in result.strategy  # stable throughout
    assert result.n_actions == 2


def test_maxmargin_caps_at_max_actions():
    """Even if 5 actions are stable, only top max_actions are considered."""
    last = {f"m{i}": 0.2 for i in range(5)}
    history = [last] * 10
    result = purify_strategy(last, history, t_half=5, max_actions=3)
    assert result.n_actions == 3


def test_top_action_always_included_even_if_top_unstable():
    """The top action by last-iterate probability is always included,
    even if its own history shows fluctuations. CFR ranked it #1 at
    the last iterate; trust that."""
    last = {"e2e4": 0.6, "d2d4": 0.4}
    # e2e4 disappears at iter 7 (post-t_half=5); d2d4 also disappears
    # at iter 8. Both would fail the stable-actions filter if treated
    # the same way — but the top (e2e4) is always retained, while
    # secondary actions are subject to the filter.
    history = []
    for t in range(10):
        if t == 7:
            history.append({"d2d4": 1.0})
        elif t == 8:
            history.append({"e2e4": 1.0})
        else:
            history.append(last)
    result = purify_strategy(last, history, t_half=5, max_actions=3)
    assert "e2e4" in result.strategy  # top always included
    assert "d2d4" not in result.strategy  # excluded — failed filter at iter 8
    assert "d2d4" in result.excluded_unstable


# ---------------------------------------------------------------------------
# Probability handling
# ---------------------------------------------------------------------------


def test_renormalized_to_sum_one():
    """Selected actions are renormalized; the result is a valid distribution."""
    last = {"a": 0.5, "b": 0.3, "c": 0.2}
    history = [last] * 10
    result = purify_strategy(last, history, t_half=5, max_actions=2)
    total = sum(result.strategy.values())
    assert total == pytest.approx(1.0)


def test_zero_probability_top_action_falls_back_to_uniform():
    """If all selected actions have zero last-iterate probability,
    fall back to uniform over selection (edge case)."""
    last = {"a": 0.0, "b": 0.0}  # weird edge case
    history = [last] * 10
    result = purify_strategy(last, history, t_half=5, max_actions=1)
    # Top by rank is still "a" (first in sort); uniform over 1 = {a: 1.0}
    assert result.strategy == {"a": 1.0}


# ---------------------------------------------------------------------------
# Integration with GT-CFR solution (smoke)
# ---------------------------------------------------------------------------


def test_integration_with_multiroot_solution():
    """End-to-end: run multi-root GT-CFR and purify its output."""
    import shutil
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish not on PATH")
    import chess
    import random
    from fow_chess.cfr.gt_cfr import root_node, solve_multiroot_growing_subgame
    from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval

    roots = [root_node(chess.Board())]
    with StockfishLeafEval() as sf:
        sol = solve_multiroot_growing_subgame(
            roots, stockfish_eval=sf, perspective=chess.WHITE,
            iterations=20, rng=random.Random(42),
        )
    # Strategy history should have been recorded.
    assert len(sol.strategy_history_at_root) == sol.iterations
    assert sol.t_half == sol.iterations // 2

    # Purify the solution's strategy.
    purified = purify_strategy(
        sol.strategy_at_root,
        sol.strategy_history_at_root,
        sol.t_half,
        max_actions=1,
    )
    assert purified.n_actions == 1
    top_move = next(iter(purified.strategy.keys()))
    # Should match the argmax of the last-iterate strategy.
    expected_top = max(sol.strategy_at_root, key=sol.strategy_at_root.get)
    assert top_move == expected_top
    assert purified.strategy[top_move] == 1.0


# ---------------------------------------------------------------------------
# A6.2: select_regime
# ---------------------------------------------------------------------------


def test_select_regime_empty_returns_one():
    """No actions → no regime selection possible; default to deterministic."""
    assert select_regime({}) == 1


def test_select_regime_single_action_returns_one():
    """Single action → deterministic by default (no second to compare against)."""
    assert select_regime({"e2e4": 0.5}) == 1


def test_select_regime_positive_margin_returns_maxmargin():
    """Top EV > second EV by margin >= threshold → Maxmargin (top-3)."""
    values = {"a": 0.5, "b": 0.2, "c": 0.1, "d": -0.3}
    # Margin = 0.5 - 0.2 = 0.3 >= 0.0 default threshold
    assert select_regime(values) == 3


def test_select_regime_zero_margin_returns_maxmargin_at_default():
    """Two top actions tied → margin 0 >= default threshold 0.0 → Maxmargin."""
    values = {"a": 0.5, "b": 0.5, "c": 0.0}
    assert select_regime(values) == 3


def test_select_regime_zero_margin_returns_resolve_at_positive_threshold():
    """Tie → margin 0 < positive threshold → Resolve (top-1)."""
    values = {"a": 0.5, "b": 0.5}
    assert select_regime(values, margin_threshold=0.05) == 1


def test_select_regime_negative_top_still_maxmargin_if_margin_positive():
    """Even if all EVs are negative, what matters is the GAP between
    top and second, not their absolute values."""
    values = {"a": -0.1, "b": -0.4, "c": -0.6}
    # Margin = -0.1 - (-0.4) = 0.3 >= 0
    assert select_regime(values) == 3


def test_select_regime_respects_custom_top_m():
    """Caller can change Maxmargin support cap."""
    values = {"a": 0.5, "b": 0.2}
    assert select_regime(values, maxmargin_top_m=2) == 2
    assert select_regime(values, maxmargin_top_m=5) == 5
