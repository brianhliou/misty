"""WS2 slice 3 (Rust side of the Stockfish FFI boundary): EqEngine.seed_expansion
sets each child's leaf_value and seeds the parent infoset's regret to
all-weight-on-the-best-child, mirroring expand_leaf's post-MultiPV step. Best =
argmax leaf_value if the parent's to_move == perspective else argmin (first on
ties). Validated in isolation with synthetic eval arrays (no Stockfish needed):
read back leaf_values, and check the current strategy concentrates on the best
child (PCFR+ on regret 1.0@best / 0 else → strategy 1.0@best).
"""
import random

import chess
import pytest

import fow_rust

pytestmark = pytest.mark.skipif(
    not hasattr(fow_rust, "EqEngine") or not hasattr(fow_rust.EqEngine, "seed_expansion"),
    reason="fow_rust EqEngine.seed_expansion not built",
)


def _engine():
    st = random.Random(0).getstate()[1]
    return fow_rust.EqEngine(list(st[:624]), st[624])


def _expanded_root():
    eng = _engine()
    root = eng.add_root_from_fen(chess.Board().fen())  # white to move
    kids = eng.expand_node(root, True)
    keys, child_ids = eng.node_children(root)  # global move-identity (_mk) keys
    return eng, root, list(child_ids), list(keys)


@pytest.mark.parametrize("perspective_white", [True, False])
def test_seed_expansion_sets_values_and_best_child(perspective_white):
    eng, root, child_ids, keys = _expanded_root()
    n = len(child_ids)
    # Distinct synthetic values with a unique max and min so best is unambiguous.
    values = [0.10 * i - 0.5 for i in range(n)]  # strictly increasing
    eng.seed_expansion(root, values, perspective_white)

    # leaf values set on each child
    for cid, v in zip(child_ids, values):
        assert eng.node_leaf_value(cid) == pytest.approx(v)

    # root is white-to-move; best = argmax if perspective is white else argmin
    best_idx = (n - 1) if perspective_white else 0
    strat = eng.current_strategy(eng.node_infoset(root), keys)
    assert strat[best_idx] == pytest.approx(1.0), f"strategy not concentrated on best: {strat}"
    assert sum(strat) == pytest.approx(1.0)
    for i, p in enumerate(strat):
        if i != best_idx:
            assert p == pytest.approx(0.0)


def test_seed_expansion_best_child_ties_take_first():
    eng, root, child_ids, keys = _expanded_root()
    n = len(child_ids)
    # All equal → max and min both pick index 0 (first on ties), either perspective.
    values = [0.25] * n
    eng.seed_expansion(root, values, True)
    strat = eng.current_strategy(eng.node_infoset(root), keys)
    assert strat[0] == pytest.approx(1.0)


def test_seed_expansion_length_mismatch_errors():
    eng, root, child_ids, keys = _expanded_root()
    with pytest.raises(ValueError):
        eng.seed_expansion(root, [0.1, 0.2], True)  # wrong length
