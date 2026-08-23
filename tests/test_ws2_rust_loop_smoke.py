"""WS2 step 3/4 (end-to-end smoke): the authoritative Rust tree drives a full
GT-CFR grow loop locally — select_leaf -> expand_node -> Stockfish eval at the
FFI batch boundary (option A) -> seed_expansion -> equilibrium_pass — over
multiple shared-infoset roots, and reads back a valid last-iterate root strategy.

This is a SMOKE (it runs + the result is a valid distribution), not yet the
byte-identical driver: best_root here is a plain size round-robin (no shared-RNG
tie-break) and there's no purification. Full-loop equivalence vs
solve_multiroot_growing_subgame is the next step. Requires stockfish on PATH.
"""
import random
import shutil

import chess
import pytest

import fow_rust
from fow_chess.cfr.leaf_eval import material_leaf_eval
from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None or not hasattr(fow_rust.EqEngine, "seed_expansion"),
    reason="needs stockfish on PATH + WS2 EqEngine.seed_expansion",
)


def _engine(seed=0):
    st = random.Random(seed).getstate()[1]
    eng = fow_rust.EqEngine(list(st[:624]), st[624])
    sel = random.Random(seed + 1).getstate()[1]
    eng.seed_select_rng(list(sel[:624]), sel[624])
    return eng


def _after(ucis):
    b = chess.Board()
    for u in ucis:
        b.push(chess.Move.from_uci(u))
    return b.fen()


def _expand_and_seed(eng, sf, node_id, perspective):
    """The Stockfish FFI batch boundary: Rust expands, Python evaluates the new
    leaves, Rust seeds them. Mirrors expand_leaf's eval logic."""
    children = eng.expand_node(node_id, perspective == chess.WHITE)
    if not children:
        return 0
    board = chess.Board(eng.node_fen(node_id))
    child_evals = sf.evaluate_children(board, perspective) if board.is_valid() else {}
    values = []
    for (frm, to, promo, cid, is_term, term_val, _wk, _bk) in children:
        if is_term:
            values.append(term_val)  # expand_node's exact terminal value
            continue
        mv = chess.Move(frm, to, promotion=promo or None)
        if mv in child_evals:
            values.append(child_evals[mv])
        else:  # FoW-legal-but-chess-illegal → material fallback (post-move board)
            values.append(material_leaf_eval(chess.Board(eng.node_fen(cid)), perspective))
    eng.seed_expansion(node_id, values, perspective == chess.WHITE)
    return len(children)


def test_rust_tree_grow_loop_runs_and_yields_valid_strategy():
    perspective = chess.WHITE
    # Three distinct white-to-move positions standing in for sampled belief truths.
    fens = [chess.Board().fen(), _after(["e2e4", "e7e5"]), _after(["d2d4", "d7d5"])]

    with StockfishLeafEval() as sf:
        eng = _engine()
        root_ids = [eng.add_root_from_fen(f) for f in fens]
        root_sizes = {r: 1 for r in root_ids}

        # bootstrap: expand + seed each root so the eq pass has a tree to walk
        expansions = 0
        for r in root_ids:
            if not eng.node_is_terminal(r):
                root_sizes[r] += _expand_and_seed(eng, sf, r, perspective)
                expansions += 1

        iters, budget = 25, 20
        for t in range(iters):
            eng.equilibrium_pass(root_ids, perspective == chess.WHITE)
            if expansions < budget:
                exploring_white = t % 2 == 0
                best = min(root_ids, key=lambda r: root_sizes[r])
                leaf = eng.select_leaf(best, exploring_white)
                if leaf is not None:
                    root_sizes[best] += _expand_and_seed(eng, sf, leaf, perspective)
                    expansions += 1

        # last-iterate strategy at the shared root infoset
        root_infoset = eng.node_infoset(root_ids[0])
        keys, _children = eng.node_children(root_ids[0])
        strat = eng.current_strategy(root_infoset, list(keys))

    assert len(strat) == len(keys) and len(strat) > 0
    assert all(p >= 0.0 for p in strat)
    assert abs(sum(strat) - 1.0) < 1e-9
    assert expansions > len(fens)  # the loop actually grew the tree
