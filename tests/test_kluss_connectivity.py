"""Unit tests for KLUSS connectivity graph + knowledge-distance computation.

Tests are synthetic — small game trees constructed by hand where the
expected distances are obvious. Real GT-CFR trees have hundreds to
thousands of nodes; testing on those would not verify correctness.
"""

from __future__ import annotations

import chess

from fow_chess.cfr.gt_cfr import GTCFRTreeNode
from fow_chess.cfr.kluss import (
    enumerate_tree_nodes,
    knowledge_distances,
    kluss_keep_mask,
)


def _node(
    *,
    to_move: chess.Color,
    obs_w: tuple,
    obs_b: tuple,
    depth: int = 0,
) -> GTCFRTreeNode:
    """Construct a minimal GTCFRTreeNode for testing.
    truth doesn't matter for connectivity-graph semantics (the
    connectivity graph is computed purely from obs_history fields)."""
    return GTCFRTreeNode(
        truth=chess.Board(),
        to_move=to_move,
        obs_history_white=obs_w,
        obs_history_black=obs_b,
        depth=depth,
    )


def test_enumerate_walks_full_tree():
    """A 3-node chain: root → child → grandchild."""
    grandchild = _node(to_move=chess.WHITE, obs_w=(2,), obs_b=(2,), depth=2)
    child = _node(to_move=chess.BLACK, obs_w=(1,), obs_b=(1,), depth=1)
    child.children[chess.Move.from_uci("a2a3")] = grandchild
    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=(), depth=0)
    root.children[chess.Move.from_uci("a1a2")] = child

    nodes = enumerate_tree_nodes([root])
    assert len(nodes) == 3
    assert nodes[0] is root
    assert nodes[1] is child
    assert nodes[2] is grandchild


def test_distance_zero_for_source_infoset():
    """The source infoset members all have distance 0."""
    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=())
    dist = knowledge_distances([root], source_infoset_ids={root.info_set_id()})
    assert dist == {0: 0}


def test_distance_one_via_same_white_infoset():
    """Two roots with the same white obs history — distance 1 via
    white's infoset equivalence."""
    # Both roots represent positions where white has seen the same
    # observation history but black has seen different things. White
    # cannot distinguish them → they share white's infoset.
    root_a = _node(to_move=chess.WHITE, obs_w=("e4",), obs_b=("d5",))
    root_b = _node(to_move=chess.WHITE, obs_w=("e4",), obs_b=("d6",))
    # Source: root_a's infoset (which root_b is also in, since both have
    # to_move=white + obs_w=("e4",) → same info_set_id())
    dist = knowledge_distances([root_a, root_b],
                               source_infoset_ids={root_a.info_set_id()})
    # Both nodes have distance 0 because they share the source infoset
    assert dist[0] == 0
    assert dist[1] == 0


def test_distance_one_via_same_black_infoset():
    """Same obs_history_black but different white history — black's
    infoset equivalence connects them at distance 1 (since neither
    is the source infoset, but they're equivalent for black)."""
    root_a = _node(to_move=chess.WHITE, obs_w=("e4",), obs_b=("e5",))
    root_b = _node(to_move=chess.WHITE, obs_w=("d4",), obs_b=("e5",))
    # Source: just root_a's infoset
    dist = knowledge_distances([root_a, root_b],
                               source_infoset_ids={root_a.info_set_id()})
    assert dist[0] == 0
    # root_b is in a different white-infoset (different obs_w)
    # but same black-infoset (same obs_b). So distance from root_a is 1.
    assert dist[1] == 1


def test_distance_two_via_chain_of_infosets():
    """A → B via white-infoset; B → C via black-infoset; A → C is 2."""
    a = _node(to_move=chess.WHITE, obs_w=("x",), obs_b=("p",))
    b = _node(to_move=chess.WHITE, obs_w=("x",), obs_b=("q",))   # same w as A
    c = _node(to_move=chess.WHITE, obs_w=("y",), obs_b=("q",))   # same b as B
    dist = knowledge_distances([a, b, c], source_infoset_ids={a.info_set_id()})
    # A (idx 0) has distance 0 (it's in source)
    # B (idx 1) has distance 0 — shares white-infoset with A, also in source
    # C (idx 2) has distance 1 — shares black-infoset with B at distance 0
    assert dist[0] == 0
    assert dist[1] == 0  # same info_set_id as A
    assert dist[2] == 1


def test_unreachable_node_has_minus_one():
    """A node with no shared infoset with the source is unreachable."""
    a = _node(to_move=chess.WHITE, obs_w=("x",), obs_b=("p",))
    isolated = _node(to_move=chess.BLACK, obs_w=("y",), obs_b=("q",))
    dist = knowledge_distances([a, isolated],
                               source_infoset_ids={a.info_set_id()})
    assert dist[0] == 0
    assert dist[1] == -1


def test_kluss_keep_mask_k2_includes_distance_3():
    """k=2 → I^(k+1) = I^3 → keep nodes at distance ≤ 3."""
    # Build a 5-node chain by alternating w/b infoset equivalence:
    # n0 (src) - n1 via w - n2 via b - n3 via w - n4 via b
    n0 = _node(to_move=chess.WHITE, obs_w=(0,), obs_b=(0,))
    n1 = _node(to_move=chess.BLACK, obs_w=(0,), obs_b=(1,))  # same w as n0
    n2 = _node(to_move=chess.WHITE, obs_w=(2,), obs_b=(1,))  # same b as n1
    n3 = _node(to_move=chess.BLACK, obs_w=(2,), obs_b=(3,))  # same w as n2
    n4 = _node(to_move=chess.WHITE, obs_w=(4,), obs_b=(3,))  # same b as n3
    # n5 is past the k=2 boundary
    n5 = _node(to_move=chess.BLACK, obs_w=(4,), obs_b=(5,))  # same w as n4

    roots = [n0, n1, n2, n3, n4, n5]
    nodes, keep = kluss_keep_mask(roots, source_infoset_ids={n0.info_set_id()}, k=2)
    # Distances: n0=0, n1=1, n2=2, n3=3, n4=4, n5=5
    # k=2 → keep distance ≤ 3 → n0, n1, n2, n3
    assert nodes == [n0, n1, n2, n3, n4, n5]
    assert keep == {0, 1, 2, 3}


def test_multiple_source_infosets_take_min():
    """If both n0 and n4 are source infosets, n2 is reachable from
    either side; distance is the minimum."""
    n0 = _node(to_move=chess.WHITE, obs_w=(0,), obs_b=(0,))
    n1 = _node(to_move=chess.BLACK, obs_w=(0,), obs_b=(1,))
    n2 = _node(to_move=chess.WHITE, obs_w=(2,), obs_b=(1,))
    n3 = _node(to_move=chess.BLACK, obs_w=(2,), obs_b=(3,))
    n4 = _node(to_move=chess.WHITE, obs_w=(4,), obs_b=(3,))

    sources = {n0.info_set_id(), n4.info_set_id()}
    dist = knowledge_distances([n0, n1, n2, n3, n4], source_infoset_ids=sources)
    # From n0: n0=0, n1=1, n2=2, n3=3, n4=4
    # From n4: n4=0, n3=1, n2=2, n1=3, n0=4
    # min: n0=0, n1=1, n2=2, n3=1, n4=0
    assert dist == {0: 0, 1: 1, 2: 2, 3: 1, 4: 0}


def test_empty_input_returns_empty_distances():
    assert knowledge_distances([], source_infoset_ids=set()) == {}


def test_tree_edges_are_traversed():
    """Regression for 2026-05-25 KLUSS bug: BFS must traverse tree
    edges (parent ↔ child), not just infoset-equivalence edges.

    A root's children have a LONGER obs_history than the root, so
    infoset edges alone never cross to the next ply. Before the fix,
    knowledge_distances returned -1 for every descendant and KLUSS
    silently pruned the entire game tree past the roots.
    """
    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=(), depth=0)
    child = _node(to_move=chess.BLACK, obs_w=("e2e4",), obs_b=("e2e4",), depth=1)
    grandchild = _node(to_move=chess.WHITE, obs_w=("e2e4",), obs_b=("e2e4", "e7e5"), depth=2)
    root.children[chess.Move.from_uci("e2e4")] = child
    child.children[chess.Move.from_uci("e7e5")] = grandchild

    dist = knowledge_distances([root], source_infoset_ids={root.info_set_id()})
    # Without tree edges: child and grandchild are unreachable (dist == -1).
    # With tree edges: BFS hops root → child → grandchild.
    assert dist[0] == 0   # root in source infoset
    assert dist[1] == 1   # child via tree edge
    assert dist[2] == 2   # grandchild via tree edge


def test_tree_edges_bidirectional():
    """Parent reachable from child via the backward tree edge."""
    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=(), depth=0)
    child = _node(to_move=chess.BLACK, obs_w=("m",), obs_b=("m",), depth=1)
    root.children[chess.Move.from_uci("e2e4")] = child

    # Source is the CHILD's infoset (different from root). BFS must
    # walk UP the tree edge to find the root.
    dist = knowledge_distances([root], source_infoset_ids={child.info_set_id()})
    assert dist[1] == 0   # child in source
    assert dist[0] == 1   # root reached via backward tree edge


def test_keep_mask_in_real_game_tree_shape():
    """End-to-end on a 4-depth tree at k=2: keep_mask must include all
    nodes up to depth 3 (one beyond k), not just the root."""
    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=(), depth=0)
    d1 = _node(to_move=chess.BLACK, obs_w=("a",), obs_b=("a",), depth=1)
    d2 = _node(to_move=chess.WHITE, obs_w=("a", "b"), obs_b=("a", "b"), depth=2)
    d3 = _node(to_move=chess.BLACK, obs_w=("a", "b", "c"), obs_b=("a", "b", "c"), depth=3)
    d4 = _node(to_move=chess.WHITE, obs_w=("a", "b", "c", "d"), obs_b=("a", "b", "c", "d"), depth=4)
    root.children[chess.Move.from_uci("a2a3")] = d1
    d1.children[chess.Move.from_uci("a7a6")] = d2
    d2.children[chess.Move.from_uci("b2b3")] = d3
    d3.children[chess.Move.from_uci("b7b6")] = d4

    nodes, keep = kluss_keep_mask([root], source_infoset_ids={root.info_set_id()}, k=2)
    # k=2 → keep distance ≤ 3 → root + d1 + d2 + d3, NOT d4.
    assert keep == {0, 1, 2, 3}, f"expected first four nodes, got {keep}"


# ---------------------------------------------------------------------------
# Phase 2: keep_ids in _select_leaf_for_expansion restricts descent
# ---------------------------------------------------------------------------


def test_select_leaf_with_keep_ids_filters_actions():
    """When keep_ids is set, _select_leaf_for_expansion must not descend
    into children whose id() isn't in keep_ids. Build a tiny 3-node
    setup: root expanded with two children A and B. A is in keep_ids,
    B is not. Walker should always return A (the leaf) regardless of
    PUCT randomness."""
    import random
    from fow_chess.cfr.gt_cfr import GTCFRState, _select_leaf_for_expansion

    # Build root + two synthetic child leaves
    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=(), depth=0)
    child_a = _node(to_move=chess.BLACK, obs_w=("a",), obs_b=("a",), depth=1)
    child_b = _node(to_move=chess.BLACK, obs_w=("b",), obs_b=("b",), depth=1)
    root.children[chess.Move.from_uci("a1a2")] = child_a
    root.children[chess.Move.from_uci("a1a3")] = child_b
    root.is_expanded = True
    state = GTCFRState()

    keep_ids = {id(root), id(child_a)}  # A in, B out
    rng = random.Random(123)
    # Run multiple trials — without keep_ids, PUCT randomness could pick B;
    # with keep_ids, must always be A.
    seen_ids: set[int] = set()
    for _ in range(20):
        leaf = _select_leaf_for_expansion(
            root, state, exploring_player=chess.WHITE, rng=rng,
            keep_ids=keep_ids,
        )
        assert leaf is not None
        seen_ids.add(id(leaf))
    assert seen_ids == {id(child_a)}, (
        f"keep_ids filter leaked — saw {seen_ids}, expected only id(A)={id(child_a)}"
    )


def test_select_leaf_returns_none_when_all_children_filtered():
    """If keep_ids excludes all children of an expanded node, the walker
    can't make progress and must return None."""
    import random
    from fow_chess.cfr.gt_cfr import GTCFRState, _select_leaf_for_expansion

    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=(), depth=0)
    child_a = _node(to_move=chess.BLACK, obs_w=("a",), obs_b=("a",), depth=1)
    root.children[chess.Move.from_uci("a1a2")] = child_a
    root.is_expanded = True
    state = GTCFRState()

    keep_ids = {id(root)}  # root only, no children
    leaf = _select_leaf_for_expansion(
        root, state, exploring_player=chess.WHITE, rng=random.Random(0),
        keep_ids=keep_ids,
    )
    assert leaf is None


def test_select_leaf_at_unexpanded_root_in_keep_set_returns_root():
    """An unexpanded root in keep_ids should be returned (it IS the leaf)."""
    import random
    from fow_chess.cfr.gt_cfr import GTCFRState, _select_leaf_for_expansion

    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=(), depth=0)
    state = GTCFRState()
    keep_ids = {id(root)}
    leaf = _select_leaf_for_expansion(
        root, state, exploring_player=chess.WHITE, rng=random.Random(0),
        keep_ids=keep_ids,
    )
    assert leaf is root


def test_select_leaf_at_unexpanded_root_NOT_in_keep_set_returns_none():
    """An unexpanded root NOT in keep_ids should yield no leaf to expand."""
    import random
    from fow_chess.cfr.gt_cfr import GTCFRState, _select_leaf_for_expansion

    root = _node(to_move=chess.WHITE, obs_w=(), obs_b=(), depth=0)
    state = GTCFRState()
    keep_ids: set[int] = set()  # excludes root
    leaf = _select_leaf_for_expansion(
        root, state, exploring_player=chess.WHITE, rng=random.Random(0),
        keep_ids=keep_ids,
    )
    assert leaf is None
