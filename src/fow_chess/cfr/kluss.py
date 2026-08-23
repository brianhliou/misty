"""KLUSS connectivity graph + knowledge-distance computation (A5.2 Phase 1).

Implements the graph machinery from Obscuro paper §3.1 / Appendix C.6:

  - Connectivity graph G of a game tree Γ̃:
      vertices = nodes of Γ̃
      edges = pairs of nodes in the same infoset of EITHER player

  - Order-k knowledge set I^k:
      nodes at graph-distance ≤ k from any node in the current infoset I

  - 2-KLUSS subgame (Obscuro's choice):
      - Restrict the search to nodes within I^3
      - Nodes in I^2 \\ I^1 are UNFROZEN (optimized in subgame, not
        locked to blueprint) — the KLUSS innovation vs KLSS

This module is the STANDALONE compute layer. Phase 1: pure functions
operating on a sequence of GTCFRTreeNode roots. No behavioral change
to the solver — the solver wires this in during Phase 2 to restrict
leaf-selection candidates.

Phase 1 goals:
  - Correctness against synthetic tiny game trees (verifiable BFS distance)
  - Reasonable performance on real GT-CFR trees (1e3-1e5 nodes)
  - Zero coupling to GT-CFR state: takes the tree, returns distances
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Hashable, Iterable

from .gt_cfr import GTCFRTreeNode


def enumerate_tree_nodes(roots: Iterable[GTCFRTreeNode]) -> list[GTCFRTreeNode]:
    """Walk every node reachable from each root via parent → children.

    Returns nodes in BFS order. Across multiple roots, the same node
    object is included once per occurrence (multi-root trees do NOT
    share node objects; same-infoset nodes from different roots are
    distinct objects in memory, linked only by infoset-equivalence —
    which is exactly what the connectivity graph captures).
    """
    out: list[GTCFRTreeNode] = []
    queue: deque[GTCFRTreeNode] = deque(roots)
    while queue:
        node = queue.popleft()
        out.append(node)
        for child in node.children.values():
            queue.append(child)
    return out


def knowledge_distances(
    roots: Iterable[GTCFRTreeNode],
    source_infoset_ids: set[Hashable],
) -> dict[int, int]:
    """Compute graph-distance in the connectivity graph G from any
    node in ``source_infoset_ids`` to every other node.

    Edges in G are of two kinds, both bidirectional:

      1. **Infoset-equivalence edges.** Two nodes share an edge if their
         ``obs_history_white`` tuples are equal (same white-infoset),
         OR their ``obs_history_black`` tuples are equal (same
         black-infoset). These connect nodes across different game
         trajectories that one player can't distinguish.

      2. **Tree edges.** Parent ↔ child in the GT-CFR tree itself.
         Without these, BFS from a root infoset can't reach any
         descendant (children have a longer obs_history that doesn't
         match the parent's, so infoset-equivalence alone never
         crosses to the next game ply). Empirically (2026-05-25),
         omitting tree edges made KLUSS k=2 keep only the root nodes
         and silently disable tree growth, causing a measured -2σ Elo
         regression vs k=0 — see [[a5.2-kluss-k2-fix-2026-05-25]].

    Returns ``{node_index → distance}`` where ``node_index`` is the
    position of each node in ``enumerate_tree_nodes(roots)``. Distance 0
    means the node is itself in the source infoset(s). Unreachable
    nodes map to ``-1``.

    BFS implementation. O(V + E) where E counts infoset-equivalence
    pairs (bounded by sum_of_group_sizes²) plus tree edges (one per
    parent→child relation in Γ̃).
    """
    nodes = enumerate_tree_nodes(roots)
    n = len(nodes)
    if n == 0:
        return {}

    # Index by Python object id for fast tree-edge lookup. Building once
    # avoids re-scanning the BFS-order list during neighbor expansion.
    index_of: dict[int, int] = {id(node): i for i, node in enumerate(nodes)}

    # Group node indices by each player's observation history.
    # Two nodes with the same obs_history_X are 1 hop apart in G via player X.
    by_white: dict[Hashable, list[int]] = defaultdict(list)
    by_black: dict[Hashable, list[int]] = defaultdict(list)
    # Reverse-tree map: child_index → parent_index. Walked once during
    # enumeration so BFS at any node can hop "up" to its parent without
    # storing parent pointers on GTCFRTreeNode.
    parent_of: dict[int, int] = {}
    for i, node in enumerate(nodes):
        by_white[node.obs_history_white].append(i)
        by_black[node.obs_history_black].append(i)
        for child in node.children.values():
            child_idx = index_of.get(id(child))
            if child_idx is not None:
                parent_of[child_idx] = i

    dist: dict[int, int] = {i: -1 for i in range(n)}
    queue: deque[int] = deque()
    for i, node in enumerate(nodes):
        if node.info_set_id() in source_infoset_ids:
            dist[i] = 0
            queue.append(i)

    while queue:
        u = queue.popleft()
        node = nodes[u]
        # Neighbors via white-infoset equivalence
        for v in by_white[node.obs_history_white]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
        # Neighbors via black-infoset equivalence
        for v in by_black[node.obs_history_black]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
        # Tree edges: children (forward) + parent (backward).
        for child in node.children.values():
            v = index_of.get(id(child))
            if v is not None and dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
        parent_idx = parent_of.get(u)
        if parent_idx is not None and dist[parent_idx] == -1:
            dist[parent_idx] = dist[u] + 1
            queue.append(parent_idx)

    return dist


def kluss_keep_mask(
    roots: Iterable[GTCFRTreeNode],
    source_infoset_ids: set[Hashable],
    k: int = 2,
) -> tuple[list[GTCFRTreeNode], set[int]]:
    """For a given k (Obscuro uses k=2), return the set of node indices
    in ``enumerate_tree_nodes(roots)`` that are within the I^(k+1)
    boundary — i.e., the nodes KEPT by the k-KLUSS subgame.

    Per paper: k-KLUSS removes nodes outside I^(k+1). At k=2 this means
    keep nodes at graph-distance ≤ 3 from the current infoset I, and
    among those, nodes in I^2 \\ I^1 are unfrozen (handled by Phase 2).

    Returns (nodes, keep_indices). The nodes list is the same indexing
    used for keep_indices.
    """
    nodes = enumerate_tree_nodes(roots)
    distances = knowledge_distances(roots, source_infoset_ids)
    keep: set[int] = {
        i for i, d in distances.items() if d != -1 and d <= k + 1
    }
    return nodes, keep
