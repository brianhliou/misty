"""GT-CFR tree data structure unit tests.

Just the tree mechanics — no algorithm yet. Validates root construction,
expansion bookkeeping, leaf-finding, terminal detection.
"""

from __future__ import annotations

import chess

from fow_chess.cfr.gt_cfr import (
    GTCFRTreeNode,
    find_leaves,
    root_node,
)


def test_root_is_unexpanded_non_terminal_leaf():
    root = root_node(chess.Board())
    assert root.is_leaf
    assert not root.is_terminal
    assert not root.is_expanded
    assert root.depth == 0
    assert root.to_move == chess.WHITE
    assert root.children == {}


def test_find_leaves_on_root_only_tree():
    root = root_node(chess.Board())
    leaves = find_leaves(root)
    assert leaves == [root]


def test_find_leaves_after_expansion():
    """When a node has been expanded with children, find_leaves
    descends and returns the un-expanded grandchildren."""
    root = root_node(chess.Board())
    root.is_expanded = True
    e2e4 = chess.Move.from_uci("e2e4")
    d2d4 = chess.Move.from_uci("d2d4")
    for move in (e2e4, d2d4):
        nxt = root.truth.copy()
        nxt.push(move)
        child = GTCFRTreeNode(
            truth=nxt,
            to_move=not root.to_move,
            obs_history_white=(),
            obs_history_black=(),
            depth=root.depth + 1,
            leaf_value=0.0,
        )
        root.children[move] = child
    leaves = find_leaves(root)
    assert len(leaves) == 2
    assert all(not leaf.is_expanded for leaf in leaves)
    assert all(leaf in root.children.values() for leaf in leaves)


def test_terminal_node_not_a_leaf_for_expansion():
    """Terminal nodes are excluded from the expansion leaf list (their
    value is known via terminal_value, not via Stockfish)."""
    # White just captured black's king with Rxe8.
    fen = "4k2R/8/8/8/8/8/8/4K3 w - - 0 1"
    prev = chess.Board(fen)
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("h8e8"))
    # nxt should be terminal (black king gone).
    terminal = GTCFRTreeNode(
        truth=nxt,
        to_move=chess.BLACK,
        obs_history_white=(),
        obs_history_black=(),
        depth=1,
    )
    assert terminal.is_terminal
    assert find_leaves(terminal) == []


def test_terminal_value_from_perspective():
    """Black king captured → white wins → +1 from white POV, -1 from black."""
    fen = "4k2R/8/8/8/8/8/8/4K3 w - - 0 1"
    prev = chess.Board(fen)
    nxt = prev.copy()
    nxt.push(chess.Move.from_uci("h8e8"))
    n = GTCFRTreeNode(
        truth=nxt,
        to_move=chess.BLACK,
        obs_history_white=(),
        obs_history_black=(),
        depth=1,
    )
    assert n.terminal_value(chess.WHITE) == 1.0
    assert n.terminal_value(chess.BLACK) == -1.0


# ---------------------------------------------------------------------------
# PUCT score + exploring-player selection
# ---------------------------------------------------------------------------


def test_puct_zero_visits_treats_actions_symmetrically():
    """At zero visits, Q=0 for all actions, and the explore term is
    determined by the prior variance + n_infoset floor — same for every
    untouched action. All scores should be equal (symmetry)."""
    from fow_chess.cfr.gt_cfr import GTCFRState, puct_score
    state = GTCFRState()
    info_set = "I0"
    actions = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    scores = [puct_score(info_set, a, state) for a in actions]
    assert scores[0] == scores[1]
    assert scores[0] >= 0  # the prior exploration bonus is non-negative


def test_puct_higher_q_wins_when_all_actions_visited_equally():
    """With equal visit counts, higher empirical Q dominates."""
    from fow_chess.cfr.gt_cfr import GTCFRState, puct_score, _mk
    state = GTCFRState()
    info_set = "I0"
    a_good = chess.Move.from_uci("e2e4")
    a_bad = chess.Move.from_uci("d2d4")
    # State dicts are int-keyed (see _mk); populate with encoded keys.
    k_good, k_bad = _mk(a_good), _mk(a_bad)
    # Both visited 10 times; a_good has Q=+0.5, a_bad has Q=-0.5
    for _ in range(10):
        state.visit_counts[info_set][k_good] += 1
        state.value_sum[info_set][k_good] += 0.5
        state.value_sq_sum[info_set][k_good] += 0.25
        state.visit_counts[info_set][k_bad] += 1
        state.value_sum[info_set][k_bad] += -0.5
        state.value_sq_sum[info_set][k_bad] += 0.25
    state.visits[info_set] = 20
    s_good = puct_score(info_set, a_good, state)
    s_bad = puct_score(info_set, a_bad, state)
    assert s_good > s_bad, f"good={s_good} bad={s_bad}"


def test_puct_under_visited_action_wins_when_q_tied():
    """When Q values are tied, the less-visited action should win
    (exploration term dominates)."""
    import random
    from fow_chess.cfr.gt_cfr import GTCFRState, puct_score, _mk
    state = GTCFRState()
    info_set = "I0"
    a_seen = chess.Move.from_uci("e2e4")
    a_fresh = chess.Move.from_uci("d2d4")
    k_seen, k_fresh = _mk(a_seen), _mk(a_fresh)  # state dicts are int-keyed
    # a_seen visited 50 times; a_fresh visited once. Same mean Q=0.
    for _ in range(50):
        state.visit_counts[info_set][k_seen] += 1
        state.value_sum[info_set][k_seen] += 0.0
        state.value_sq_sum[info_set][k_seen] += 0.5  # nonzero variance
    state.visit_counts[info_set][k_fresh] += 1
    state.value_sum[info_set][k_fresh] += 0.0
    state.value_sq_sum[info_set][k_fresh] += 0.5
    state.visits[info_set] = 51
    s_seen = puct_score(info_set, a_seen, state)
    s_fresh = puct_score(info_set, a_fresh, state)
    assert s_fresh > s_seen, f"fresh={s_fresh} seen={s_seen}"


def test_exploring_selection_returns_legal_action():
    """The PUCT-mixture selector always returns one of the supplied
    legal actions."""
    import random
    from fow_chess.cfr.gt_cfr import GTCFRState, select_action_for_exploring_player
    state = GTCFRState()
    info_set = "I0"
    actions = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("g1f3"),
    ]
    strategy = {actions[0]: 0.5, actions[1]: 0.5, actions[2]: 0.0}
    rng = random.Random(42)
    seen = set()
    for _ in range(100):
        chosen = select_action_for_exploring_player(
            info_set, actions, state, strategy, rng=rng,
        )
        assert chosen in actions
        seen.add(chosen)
    # Across 100 samples we should see both support actions; argmax may
    # also surface non-support via PUCT branch.
    assert actions[0] in seen
    assert actions[1] in seen


# ---------------------------------------------------------------------------
# Leaf expansion (Stockfish-driven)
# ---------------------------------------------------------------------------


import shutil


def test_expand_root_via_stockfish():
    """Expand the root from the starting position. All 20 chess-legal
    children should be added, each with a Stockfish-derived leaf_value.
    Smart-init regret puts all weight on the best child."""
    if shutil.which("stockfish") is None:
        import pytest
        pytest.skip("Stockfish binary not on PATH")
    from fow_chess.cfr.gt_cfr import GTCFRState, expand_leaf
    from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval

    root = root_node(chess.Board())
    state = GTCFRState()
    with StockfishLeafEval() as sf:
        added = expand_leaf(root, state, stockfish_eval=sf, perspective=chess.WHITE)
    assert added == 20  # 20 starting-position legal moves
    assert root.is_expanded
    assert len(root.children) == 20
    # Every child has a leaf_value in [-1, 1].
    for child in root.children.values():
        assert child.leaf_value is not None
        assert -1.0 <= child.leaf_value <= 1.0
    # Smart-init regret put a single positive value on the best move.
    info_set = root.info_set_id()
    regrets = state.regrets[info_set]
    positives = [a for a, r in regrets.items() if r > 0]
    assert len(positives) == 1, f"expected 1 best move, got {len(positives)}: {positives}"


def test_expand_idempotent():
    """Calling expand_leaf twice on the same node does nothing the second time."""
    if shutil.which("stockfish") is None:
        import pytest
        pytest.skip("Stockfish binary not on PATH")
    from fow_chess.cfr.gt_cfr import GTCFRState, expand_leaf
    from fow_chess.cfr.leaf_eval_stockfish import StockfishLeafEval

    root = root_node(chess.Board())
    state = GTCFRState()
    with StockfishLeafEval() as sf:
        n1 = expand_leaf(root, state, stockfish_eval=sf, perspective=chess.WHITE)
        n2 = expand_leaf(root, state, stockfish_eval=sf, perspective=chess.WHITE)
    assert n1 == 20
    assert n2 == 0


def test_info_set_id_keyed_by_to_move_and_history():
    """Two nodes with the same observation history (from to_move's POV)
    must produce the same info_set_id."""
    root_a = GTCFRTreeNode(
        truth=chess.Board(),
        to_move=chess.WHITE,
        obs_history_white=(("h1",),),
        obs_history_black=(),
        depth=0,
    )
    root_b = GTCFRTreeNode(
        truth=chess.Board(),
        to_move=chess.WHITE,
        obs_history_white=(("h1",),),
        obs_history_black=(("h2",),),  # different black history; shouldn't matter
        depth=5,  # different depth; doesn't matter for info_set_id
    )
    assert root_a.info_set_id() == root_b.info_set_id()
