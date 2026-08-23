"""FOW_KLUSS_SOFT: the keep-mask expansion deadlock and its fallback.

The bug (2026-06-12): with kluss_k set, the keep-restricted leaf walk halts
(returns None) when every child at some node filters out of the keep set.
Once the root strategy concentrates, every walk follows the same favored
line to the keep boundary and dies — the expansion budget goes unspent and
the tree stays at a handful of nodes (measured 4-10 at |P|=1 vs 1800+ with
kluss off; sf children-cache misses = expansion count). Soft mode retries
the walk unrestricted when the in-scope walk returns None, so KLUSS scope
is a preference, not a starvation cap.
"""
import random
import shutil

import chess
import pytest

from fow_chess.cfr.gt_cfr import solve_multiroot_rust_tree

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not on PATH",
)

EXPANSION_BUDGET = 200


def _expansions(monkeypatch, *, soft: bool, iterations: int = 400) -> int:
    if soft:
        monkeypatch.setenv("FOW_KLUSS_SOFT", "1")
    else:
        monkeypatch.delenv("FOW_KLUSS_SOFT", raising=False)
    from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval
    with StockfishLeafEval() as sf:
        solve_multiroot_rust_tree(
            [chess.Board()],
            stockfish_eval=sf,
            perspective=chess.WHITE,
            iterations=iterations,
            expansion_budget=EXPANSION_BUDGET,
            kluss_k=2,
            rng=random.Random(0),
        )
        return sf.children_cache_misses


def test_hard_kluss_starves_singleton_belief(monkeypatch):
    """Documents the deadlock at |P|=1: expansion stalls far below the
    budget. If this starts FAILING, the hard-KLUSS walk got fixed and
    soft mode may be retired."""
    assert _expansions(monkeypatch, soft=False) < EXPANSION_BUDGET // 4


def test_soft_kluss_spends_the_expansion_budget(monkeypatch):
    soft = _expansions(monkeypatch, soft=True)
    assert soft > EXPANSION_BUDGET * 3 // 4, soft


def test_flag_off_is_deterministic_prior_behavior(monkeypatch):
    a = _expansions(monkeypatch, soft=False)
    b = _expansions(monkeypatch, soft=False)
    assert a == b
