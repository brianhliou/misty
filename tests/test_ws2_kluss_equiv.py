"""WS2 KLUSS-on-rust byte-parity: the Rust-tree driver with kluss_k=2 must
produce the SAME last-iterate root strategy as the Python-tree driver
(solve_multiroot_growing_subgame, use_rust_eq=True, kluss_k=2), given the
same roots / seed / Stockfish.

Companion to test_ws2_full_loop_equiv.py (which proves equivalence without
KLUSS). This test adds the KLUSS-filter ingredient and asserts the
filtered selection paths agree.
"""
import random
import shutil

import chess
import pytest

import fow_rust
from fow_chess.cfr.gt_cfr import (
    root_node,
    solve_multiroot_growing_subgame,
    solve_multiroot_rust_tree,
)
from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None
    or not hasattr(fow_rust.EqEngine, "set_kluss_keep_from"),
    reason="needs stockfish on PATH + EqEngine.set_kluss_keep_from (KLUSS-on-rust)",
)


def _after(ucis):
    b = chess.Board()
    for u in ucis:
        b.push(chess.Move.from_uci(u))
    return b


@pytest.mark.parametrize("seed", [7, 123, 99999])
@pytest.mark.parametrize("kluss_k", [2])
def test_kluss_rust_matches_python_path(seed, kluss_k):
    perspective = chess.WHITE
    boards = [chess.Board(), _after(["e2e4", "e7e5"]), _after(["d2d4", "d7d5"])]
    iters, budget = 14, 30

    with StockfishLeafEval() as sf:
        py = solve_multiroot_growing_subgame(
            [root_node(b.copy()) for b in boards],
            stockfish_eval=sf,
            perspective=perspective,
            iterations=iters,
            expansion_budget=budget,
            rng=random.Random(seed),
            use_rust_eq=True,
            kluss_k=kluss_k,
        )
        ru = solve_multiroot_rust_tree(
            [b.copy() for b in boards],
            stockfish_eval=sf,
            perspective=perspective,
            iterations=iters,
            expansion_budget=budget,
            rng=random.Random(seed),
            kluss_k=kluss_k,
        )

    assert py.iterations == ru.iterations
    assert set(py.strategy_at_root) == set(ru.strategy_at_root), (
        f"root action sets differ (seed={seed}, k={kluss_k}): "
        f"py={set(m.uci() for m in py.strategy_at_root)} "
        f"rust={set(m.uci() for m in ru.strategy_at_root)}"
    )
    for mv, p in py.strategy_at_root.items():
        assert ru.strategy_at_root[mv] == pytest.approx(p, abs=1e-9), (
            f"strategy diverged at {mv.uci()}: py={p} rust={ru.strategy_at_root[mv]} "
            f"(seed={seed}, k={kluss_k})"
        )
    assert len(py.strategy_history_at_root) == len(ru.strategy_history_at_root)
