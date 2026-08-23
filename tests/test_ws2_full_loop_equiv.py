"""WS2 step 4: full-loop equivalence. The use_rust_tree driver
(solve_multiroot_rust_tree — authoritative Rust tree drives select/expand/eq/seed,
Stockfish at the FFI boundary) must produce the SAME last-iterate root strategy as
solve_multiroot_growing_subgame(use_rust_eq=True), given the same roots, seed, and
(deterministic depth-1) Stockfish. This is the equivalence check the move-order +
RNG byte-matching work was for — correctness without a bakeoff. Requires stockfish.
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
    shutil.which("stockfish") is None or not hasattr(fow_rust.EqEngine, "pick_best_root"),
    reason="needs stockfish on PATH + WS2 EqEngine.pick_best_root",
)


def _after(ucis):
    b = chess.Board()
    for u in ucis:
        b.push(chess.Move.from_uci(u))
    return b


@pytest.mark.parametrize("seed", [7, 123, 99999])
def test_full_loop_strategy_matches_python_path(seed):
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
        )
        ru = solve_multiroot_rust_tree(
            [b.copy() for b in boards],
            stockfish_eval=sf,
            perspective=perspective,
            iterations=iters,
            expansion_budget=budget,
            rng=random.Random(seed),
        )

    assert py.iterations == ru.iterations
    assert set(py.strategy_at_root) == set(ru.strategy_at_root), "root action sets differ"
    for mv, p in py.strategy_at_root.items():
        assert ru.strategy_at_root[mv] == pytest.approx(p, abs=1e-9), (
            f"strategy diverged at {mv.uci()}: py={p} rust={ru.strategy_at_root[mv]} (seed={seed})"
        )
    assert len(py.strategy_history_at_root) == len(ru.strategy_history_at_root)
