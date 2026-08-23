"""WS2 slice 2: leaf selection in Rust (EqEngine.select_leaf) is byte-identical
to gt_cfr._select_leaf_for_expansion. Validated by EQUIVALENCE, not a bakeoff:
a Python reference walk reads the SAME Rust tree (topology + interned infosets +
eq state via accessors) and consumes a random.Random seeded to the SAME state as
the engine's transplanted selection RNG. If the Rust port's logic + RNG
replication (random()->next_f64, choice->getrandbits/_randbelow) are exact, both
pick the same leaf for every (exploring color, seed).
"""
import random

import chess
import pytest

import fow_rust
from fow_chess.cfr.gt_cfr import _sample  # the exact strategy index-sampler

pytestmark = pytest.mark.skipif(
    not hasattr(fow_rust, "EqEngine") or not hasattr(fow_rust.EqEngine, "select_leaf"),
    reason="fow_rust EqEngine.select_leaf not built",
)


def _engine(seed=0):
    st = random.Random(seed).getstate()[1]
    return fow_rust.EqEngine(list(st[:624]), st[624])


def _ref_select(eng, root_id, exploring_white, rng):
    """Pure-Python mirror of _select_leaf_for_expansion (non-KLUSS) reading the
    Rust tree through accessors — the equivalence oracle for select_leaf."""
    node = root_id
    while True:
        if eng.node_is_terminal(node):
            return None
        keys, kids = eng.node_children(node)
        if not kids:
            return node
        infoset = eng.node_infoset(node)
        strat = eng.current_strategy(infoset, list(keys))
        if eng.node_to_move_white(node) == exploring_white:
            if rng.random() < 0.5:
                support = [i for i, p in enumerate(strat) if p > 0.0] or list(range(len(keys)))
                idx = rng.choice(support)
            else:
                best_i, best = 0, float("-inf")
                for i, k in enumerate(keys):
                    s = eng.puct_get(infoset, k, 1.0)
                    if s > best:
                        best, best_i = s, i
                idx = best_i
        else:
            idx = _sample(strat, rng)
        node = kids[idx]


def _build_tree(eng, persp=True):
    """Root + expand_node a couple levels so the walk has real depth, then run
    equilibrium passes to populate visits/values (the PUCT inputs)."""
    root = eng.add_root_from_fen(chess.Board().fen())
    kids = eng.expand_node(root, persp)
    for rc in kids[:6]:
        gk = eng.expand_node(rc[3], persp)  # depth 2
        if gk:
            eng.expand_node(gk[0][3], persp)  # one depth-3 branch
    for _ in range(80):
        eng.equilibrium_pass([root], persp)  # uses the eq rng, not sel_rng
    return root


@pytest.mark.parametrize("exploring_white", [True, False])
@pytest.mark.parametrize("seed", [1, 7, 42, 12377, 99999])
def test_select_leaf_matches_python_reference(exploring_white, seed):
    eng = _engine()
    root = _build_tree(eng)

    sel_state = random.Random(seed).getstate()[1]
    eng.seed_select_rng(list(sel_state[:624]), sel_state[624])
    ref_rng = random.Random(seed)  # same initial state as the engine sel_rng

    rust_leaf = eng.select_leaf(root, exploring_white)
    ref_leaf = _ref_select(eng, root, exploring_white, ref_rng)
    assert rust_leaf == ref_leaf, (
        f"select_leaf diverged: rust={rust_leaf} ref={ref_leaf} "
        f"(exploring_white={exploring_white}, seed={seed})"
    )
    # the walk should actually descend below the root (real coverage)
    assert rust_leaf is not None and rust_leaf != root


def test_select_leaf_consumes_rng_in_lockstep_over_repeats():
    """Run many selections back-to-back on one rng stream; Rust and Python must
    stay in lockstep (validates RNG-consumption count per descent, not just the
    final leaf of a single call)."""
    eng = _engine()
    root = _build_tree(eng)

    sel_state = random.Random(2024).getstate()[1]
    eng.seed_select_rng(list(sel_state[:624]), sel_state[624])
    ref_rng = random.Random(2024)

    for i in range(60):
        exploring_white = i % 2 == 0
        rust_leaf = eng.select_leaf(root, exploring_white)
        ref_leaf = _ref_select(eng, root, exploring_white, ref_rng)
        assert rust_leaf == ref_leaf, f"diverged at repeat {i}"
