"""One-sided Growing-Tree CFR (Obscuro variant of Schmid et al. 2023).

GT-CFR alternates two operations on a *growing* game tree:

* **Equilibrium pass:** runs PCFR+ over the currently-expanded subtree.
* **Expansion pass:** selects one leaf via PUCT-mixture and expands all
  its children at once using Stockfish MultiPV.

The Obscuro variant differs from standard GT-CFR in two ways the
synthesis doc flags as essential:

1. Operates on the **game tree itself** (per-(truth, history) nodes),
   not the public tree. FoW chess has rare common knowledge, so the
   public tree is degenerate.
2. **One-sided**: each iteration alternates which player is "exploring".
   Non-exploring player plays the current solved strategy y^t.
   Exploring player plays a PUCT-mixture biased toward visiting
   high-value, under-visited leaves.

This module is written fresh rather than bolted onto tabular.py because
the data flow is meaningfully different: tabular CFR enumerates every
legal action at every visit; GT-CFR only sees actions that have been
*expanded*. We share PCFR+ regret-matching math conceptually but
re-implement it here to keep the traversal coherent.

Reference: docs/fog-of-war/library/research/papers/architecture-synthesis.md
(One-sided GT-CFR, Appendix C.4) and the PCFR+ section above it.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Hashable, Iterable

import chess

from ..rules import ChessRules, Rules
from .gadget import ResolveGadget

# Shared chess-rules singleton: the default game seam for tree nodes built
# without an explicit `rules` (all chess construction). Stateless, so sharing one
# instance is safe; a mini engine threads its own Rules via sample_roots_from_P /
# root_node. gt_cfr -> rules is cycle-free (rules.py never imports gt_cfr; it
# reimplements _mk's encoding inline, pinned byte-equal in test_rules_chess_parity).
_DEFAULT_RULES: Rules = ChessRules()


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------


@dataclass
class GTCFRTreeNode:
    """A node in the GT-CFR growing tree.

    Wraps a game-state descriptor (truth + observation histories) and
    tracks which actions out of this node have been expanded into child
    nodes. Leaves have no expanded children; their value comes from
    the Stockfish evaluation captured at expansion time.

    ``stockfish_child_evals`` is populated when this node is EXPANDED
    (children added). The dict maps each chess-legal action → tanh-
    normalized Stockfish value from the perspective player's POV.
    Children that aren't FoW-legal-but-not-chess-legal lack entries
    and rely on a material-eval fallback at use time.
    """

    truth: chess.Board
    to_move: chess.Color
    obs_history_white: tuple
    obs_history_black: tuple
    depth: int
    # Children that have been expanded out of this node.
    children: dict[chess.Move, "GTCFRTreeNode"] = field(default_factory=dict)
    # Whether this node has been expanded (children populated). Distinct
    # from "has children" because a terminal node has no children even
    # when "expanded" (no expansion needed; terminal_value is exact).
    is_expanded: bool = False
    # Leaf value from the search's perspective POV. Populated by the
    # parent's expansion (this node was a child created during that
    # expansion). None on the root.
    leaf_value: float | None = None
    # The game seam (Phase 1 slice 4): routes is_terminal / terminal_value /
    # info_set_id through the game's Rules so the node is game-parameterized.
    # Defaults to chess; roots + children inherit it. Results are unchanged for
    # chess (ChessRules delegates to the same king-capture / side logic, pinned
    # byte-equal by test_rules_chess_parity), and the cached_property caching is
    # preserved so the per-access hot path is untouched.
    rules: Rules = _DEFAULT_RULES

    @cached_property
    def is_terminal(self) -> bool:
        """King-capture is the only terminal condition in FoW. Cached
        because each node's truth is immutable for its lifetime and
        the property is accessed ~318K times per pick_move (per the
        2026-05-25 profile). Cached, so the per-access cost is unchanged by the
        Rules delegation — the method call is paid once, when the cache fills."""
        return self.rules.is_terminal(self.truth)

    @property
    def is_leaf(self) -> bool:
        """A leaf in the *current* tree: hasn't been expanded yet (or is terminal)."""
        return self.is_terminal or not self.is_expanded

    def info_set_id(self) -> Hashable:
        """Identifier of the infoset this node belongs to (the
        moving player's observation history + side to move). Used as
        a dict key into state.regrets, state.last_strategy, etc.
        Hot — called from 12 sites including the CFR walk's per-visit
        path. Cached because to_move + obs_history fields are
        immutable for the node's lifetime."""
        cached = getattr(self, "_info_set_id_cache", None)
        if cached is not None:
            return cached
        history = (
            self.obs_history_white
            if self.rules.is_first_player(self.to_move)
            else self.obs_history_black
        )
        result = (self.to_move, history)
        self._info_set_id_cache = result
        return result

    def terminal_value(self, perspective: chess.Color) -> float:
        return self.rules.terminal_value(self.truth, perspective)


def root_node(
    truth: chess.Board,
    to_move: chess.Color | None = None,
    rules: Rules = _DEFAULT_RULES,
) -> GTCFRTreeNode:
    """Construct the root tree node from a known truth board (no expansion yet)."""
    if to_move is None:
        to_move = rules.to_move(truth)
    return GTCFRTreeNode(
        truth=truth.copy(),
        to_move=to_move,
        obs_history_white=(),
        obs_history_black=(),
        depth=0,
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Per-search state (regrets + visit counts + value sums)
# ---------------------------------------------------------------------------


@dataclass
class GTCFRState:
    """Mutable state of a GT-CFR search.

    Per-infoset state:
    * ``regrets[I][a]``   — cumulative thresholded regret (PCFR+ z).
    * ``last_regret[I][a]`` — regret vector observed at most recent visit.
    * ``last_strategy[I][a]`` — most recent strategy (for last-iterate).
    * ``visits[I]`` — total visits to this infoset across iterations.

    Per-(infoset, action) state:
    * ``visit_counts[(I, a)]`` — N(I, a).
    * ``value_sum[(I, a)]`` — Σ u(x^t, y^t | I, a) over visits.
    * ``value_sq_sum[(I, a)]`` — Σ u² for empirical variance.
    """

    regrets: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    last_regret: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    last_strategy: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    visits: dict = field(default_factory=lambda: defaultdict(int))
    visit_counts: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    value_sum: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    value_sq_sum: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))


# ---------------------------------------------------------------------------
# Tree navigation helpers
# ---------------------------------------------------------------------------


def find_leaves(root: GTCFRTreeNode) -> list[GTCFRTreeNode]:
    """Return all non-terminal leaves currently in the tree.

    "Leaf" here = node not yet expanded AND not terminal. Terminal
    nodes are not expansion candidates; they have known terminal value.
    """
    leaves: list[GTCFRTreeNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal:
            continue
        if not node.is_expanded:
            leaves.append(node)
            continue
        stack.extend(node.children.values())
    return leaves


# ---------------------------------------------------------------------------
# PUCT-based leaf selection (Obscuro Appendix C.4)
# ---------------------------------------------------------------------------


# Empirical-variance prior: 2 fake samples of ±1 give an initial σ² estimate.
# Implementation detail per the paper.
_PUCT_C = 1.0
_PRIOR_VARIANCE = 1.0  # variance of two ±1 samples = 1.0


def _mk(move: chess.Move) -> int:
    """Encode a move as a collision-free int key for the per-(infoset, action)
    state dicts. chess.Move is a frozen dataclass whose Python-level __hash__
    builds+hashes a tuple on every dict access — ~6.6M times / ~1s of a 5s
    pick_move in the 2026-05-26 profile, the dominant CFR-walk cost. Hashing a
    plain int is a C builtin. Byte-equal: the encoding is a bijection over
    (from, to, promotion, drop) for the standard move space, so two moves share
    a key iff they're equal. Only the SHARED state dicts are re-keyed; node
    children / local strategy dicts / action lists stay Move-keyed."""
    return (
        move.from_square
        | (move.to_square << 6)
        | ((move.promotion or 0) << 12)
        | ((move.drop or 0) << 16)
    )


def _rule_action_key(rules: Rules, move) -> int:
    """Rule-owned action key for Python-tree state.

    Chess and mini keep the historical 6-bit packing through their adapters.
    Full Xiangqi needs wider square fields, so generic Python-tree state must
    ask the active Rules instead of calling _mk directly.
    """
    return rules.action_key(move)


def _q_value(
    info_set_id: Hashable,
    action: chess.Move,
    state: GTCFRState,
    rules: Rules = _DEFAULT_RULES,
) -> float:
    """Empirical mean value Q̄(I, a) over visits to (I, a). 0 if unvisited."""
    a = _rule_action_key(rules, action)
    n = state.visit_counts[info_set_id][a]
    if n == 0:
        return 0.0
    return state.value_sum[info_set_id][a] / n


def _empirical_variance(
    info_set_id: Hashable,
    action: chess.Move,
    state: GTCFRState,
    rules: Rules = _DEFAULT_RULES,
) -> float:
    """σ̂²(I, a) with two prior samples of ±1 mixed in (paper convention)."""
    a = _rule_action_key(rules, action)
    n_real = state.visit_counts[info_set_id][a]
    n_total = n_real + 2  # +2 priors
    sum_x = state.value_sum[info_set_id][a]
    sum_x2 = state.value_sq_sum[info_set_id][a] + _PRIOR_VARIANCE * 2  # +1² + (-1)² = 2
    if n_total <= 1:
        return _PRIOR_VARIANCE
    mean = sum_x / n_total
    var = max(0.0, sum_x2 / n_total - mean * mean)
    return var


def puct_score(
    info_set_id: Hashable,
    action: chess.Move,
    state: GTCFRState,
    *,
    c: float = _PUCT_C,
    rules: Rules = _DEFAULT_RULES,
) -> float:
    """Obscuro's PUCT formula:

        Q̄(I, a) + C · σ̂(I, a) · √N(I) / (1 + N(I, a))

    Where σ̂ is empirical-std-dev with two prior ±1 samples baked in.
    """
    q = _q_value(info_set_id, action, state, rules)
    sigma = math.sqrt(_empirical_variance(info_set_id, action, state, rules))
    n_infoset = max(1, state.visits[info_set_id])
    n_action = state.visit_counts[info_set_id][_rule_action_key(rules, action)]
    explore = c * sigma * math.sqrt(n_infoset) / (1 + n_action)
    return q + explore


def select_action_for_exploring_player(
    info_set_id: Hashable,
    legal_actions: list[chess.Move],
    state: GTCFRState,
    current_strategy: dict[chess.Move, float],
    *,
    rng: random.Random,
    c: float = _PUCT_C,
    eq_mirror: "_RustEqMirror | None" = None,
    rules: Rules = _DEFAULT_RULES,
) -> chess.Move:
    """Exploring-player action selection (Obscuro Appendix C.4).

    x̃ = (1/2) x̃_Max + (1/2) x̃_PUCT
    * x̃_Max: uniform over support of current strategy
    * x̃_PUCT: argmax of PUCT score over legal_actions

    Implementation: with prob 0.5, sample uniformly from current
    strategy's support; with prob 0.5, take argmax of PUCT.

    When ``eq_mirror`` is set, PUCT scores are read from the Rust EqEngine
    (which owns the CFR state on the Rust path) instead of ``state``.
    """
    if not legal_actions:
        raise ValueError("no legal actions to select from")
    if rng.random() < 0.5:
        support = [a for a in legal_actions if current_strategy.get(a, 0.0) > 0.0]
        if not support:
            support = list(legal_actions)
        return rng.choice(support)
    # PUCT branch
    if eq_mirror is not None:
        return max(legal_actions, key=lambda a: eq_mirror.puct_score(info_set_id, a, c=c))
    return max(
        legal_actions,
        key=lambda a: puct_score(info_set_id, a, state, c=c, rules=rules),
    )


# ---------------------------------------------------------------------------
# Leaf expansion (Obscuro Appendix C.5)
# ---------------------------------------------------------------------------


# Importing here to keep the module's top-level imports light; the
# Stockfish path is only exercised when callers actually use it.
def expand_leaf(
    leaf: GTCFRTreeNode,
    state: GTCFRState,
    *,
    stockfish_eval,  # StockfishLeafEval — duck-typed to avoid circular import
    perspective: chess.Color,
    eq_mirror: "_RustEqMirror | None" = None,
) -> int:
    """Expand all children of ``leaf`` at once using Stockfish MultiPV.

    Per Obscuro Appendix C.5:
    * Call Stockfish in MultiPV mode at depth 1 to evaluate every
      child position in a single call.
    * Add every chess-legal child to the tree.
    * Initialize the regret minimizer at the now-non-leaf infoset with
      all weight on the best Stockfish-eval'd child. This avoids the
      "max-to-average" instability when a fresh infoset transitions
      from leaf-eval to mixed-strategy evaluation.

    FoW-only-legal moves (chess-illegal but FoW-legal) are added with a
    material-eval fallback inside the larger search; here we ONLY
    populate the children that Stockfish returned values for.

    Returns: number of children added (0 if leaf is terminal or
    Stockfish returned no usable evals; caller should bail).
    """
    if leaf.is_terminal:
        return 0
    if leaf.is_expanded:
        return 0  # idempotent

    # MultiPV at depth 1: per-action Stockfish eval from `perspective` POV.
    child_evals = stockfish_eval.evaluate_children(leaf.truth, perspective)

    legal_moves = list(leaf.rules.pseudo_legal_moves(leaf.truth))
    added = 0
    for move in legal_moves:
        next_truth = leaf.rules.apply(leaf.truth, move)
        key_white, key_black = leaf.rules.observation_keys(leaf.truth, next_truth)
        # Each child carries its own leaf_value (from perspective POV).
        # Terminal check FIRST: a king-capturing move is terminal (-1/+1 from
        # perspective). Stockfish doesn't emit king-captures as candidates in
        # standard chess, so without this check the king-capture child falls
        # through to material_leaf_eval, which uses material_score with king=0
        # → the king-capture child is scored as just "I'm up some material"
        # (~0.5-0.9 tanh-saturated) instead of an exact +1.0 win. CFR seeded
        # from that wrong leaf_value can prefer a non-king-capture move that
        # Stockfish ranked higher (e.g., a visible rook capture at +500cp).
        # The terminal-first branch fixes this and also handles stalemate
        # correctly (own_king missing → -1.0). Routed through leaf.rules so the
        # terminal definition is single-sourced per game (king-capture for chess;
        # general-capture / Try for the xiangqi / fusion adapters).
        if leaf.rules.is_terminal(next_truth):
            child_leaf = leaf.rules.terminal_value(next_truth, perspective)
        elif move in child_evals:
            child_leaf = child_evals[move]
        else:
            # FoW-legal-but-chess-illegal: material fallback.
            child_leaf = leaf.rules.material_leaf_eval(next_truth, perspective)
        child = GTCFRTreeNode(
            truth=next_truth,
            to_move=leaf.rules.opponent(leaf.to_move),
            obs_history_white=(*leaf.obs_history_white, key_white),
            obs_history_black=(*leaf.obs_history_black, key_black),
            depth=leaf.depth + 1,
            leaf_value=child_leaf,
            rules=leaf.rules,
        )
        leaf.children[move] = child
        added += 1

    leaf.is_expanded = True

    # Smart regret init at the now-non-leaf infoset: bias the next
    # regret-matching pass toward the best-Stockfish-eval'd child.
    # leaf.children store their leaf_value from `perspective` POV;
    # leaf.to_move may differ from perspective, in which case best-for-
    # leaf-to-move = lowest-leaf_value (from perspective's POV).
    info_set_id = leaf.info_set_id()
    if leaf.children:
        if leaf.to_move == perspective:
            best_move = max(leaf.children.keys(),
                            key=lambda m: leaf.children[m].leaf_value or 0.0)
        else:
            best_move = min(leaf.children.keys(),
                            key=lambda m: leaf.children[m].leaf_value or 0.0)
        _SEED_REGRET = 1.0
        for move in leaf.children:
            state.regrets[info_set_id][_rule_action_key(leaf.rules, move)] = (
                _SEED_REGRET if move == best_move else 0.0
            )

    # Mirror the freshly-expanded node + its regret seeds into the Rust engine.
    if eq_mirror is not None and leaf.is_expanded:
        eq_mirror.sync_expansion(leaf, state)

    return added


# ---------------------------------------------------------------------------
# Equilibrium pass (PCFR+ over current expanded tree)
# ---------------------------------------------------------------------------


def _current_strategy(
    info_set_id: Hashable,
    actions: list[chess.Move],
    state: GTCFRState,
    keys: list[int] | None = None,
) -> list[float]:
    """PCFR+ strategy: x = [z + last_regret]^+ / ||·||_1.

    ``keys`` = precomputed _mk(action) ints; hot callers pass them to avoid
    recomputing the encoding (expansion/solution callers leave it None)."""
    z = state.regrets[info_set_id]
    prev = state.last_regret.get(info_set_id, {})
    if keys is None:
        keys = [_mk(a) for a in actions]
    positive = [max(0.0, z.get(k, 0.0) + prev.get(k, 0.0)) for k in keys]
    total = sum(positive)
    if total > 0.0:
        return [r / total for r in positive]
    n = len(actions)
    return [1.0 / n] * n


def _sample(probs: list[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1


def _equilibrium_traverse(
    node: GTCFRTreeNode,
    state: GTCFRState,
    traversing_player: chess.Color,
    perspective: chess.Color,
    rng: random.Random,
) -> float:
    """PCFR+ traversal over the currently-expanded tree.

    Returns the value of ``node`` from ``traversing_player`` POV.

    * Terminal: terminal_value(traversing_player).
    * Leaf (not expanded, not terminal): leaf_value, sign-flipped if
      ``traversing_player != perspective`` (leaves are stored in
      ``perspective`` POV).
    * Expanded internal node, traverser's turn: enumerate all expanded
      actions, recurse, accumulate regret. RM+ thresholded update.
    * Expanded internal node, opp's turn: sample one action from opp's
      current strategy, recurse.
    """
    if node.is_terminal:
        return node.terminal_value(traversing_player)
    if not node.is_expanded:
        v = node.leaf_value if node.leaf_value is not None else 0.0
        return v if traversing_player == perspective else -v
    actions = list(node.children.keys())
    if not actions:
        return 0.0  # shouldn't normally happen; safety
    # Encode action keys once per visit. int keys avoid chess.Move's slow
    # dataclass __hash__ on every per-action state-dict access — the dominant
    # CFR-walk cost (~1s/5s) in the 2026-05-26 profile. children stays Move-keyed.
    akeys = [_rule_action_key(node.rules, a) for a in actions]
    info_set_id = node.info_set_id()
    strategy = _current_strategy(info_set_id, actions, state, akeys)
    state.visits[info_set_id] += 1
    # Hoist subdict lookups once per call. Before: every per-action
    # write went through state.<field>[info_set_id][action], doing the
    # info_set_id hash + dict lookup N×6 times. After: one lookup
    # per state field per call. 6.5M-call hash storm dropped substantially
    # in the 2026-05-25 profile session.
    last_strategy_dict = state.last_strategy[info_set_id]
    if node.to_move == traversing_player:
        action_values = [0.0] * len(actions)
        for i, action in enumerate(actions):
            action_values[i] = _equilibrium_traverse(
                node.children[action], state, traversing_player, perspective, rng,
            )
        node_value = sum(s * v for s, v in zip(strategy, action_values, strict=False))
        # PCFR+ regret update: z := [z + r]^+ ; stash raw r for next-iter prediction.
        regrets_dict = state.regrets[info_set_id]
        last_regret_dict = state.last_regret[info_set_id]
        visit_counts_dict = state.visit_counts[info_set_id]
        value_sum_dict = state.value_sum[info_set_id]
        value_sq_sum_dict = state.value_sq_sum[info_set_id]
        for i in range(len(actions)):
            k = akeys[i]
            av = action_values[i]
            r = av - node_value
            cur = regrets_dict.get(k, 0.0)
            regrets_dict[k] = max(0.0, cur + r)
            last_regret_dict[k] = r
            last_strategy_dict[k] = strategy[i]
            # PUCT bookkeeping
            visit_counts_dict[k] += 1
            value_sum_dict[k] += av
            value_sq_sum_dict[k] += av * av
        return node_value
    # Opp's turn: external sampling
    chosen = _sample(strategy, rng)
    for i, k in enumerate(akeys):
        last_strategy_dict[k] = strategy[i]
    return _equilibrium_traverse(
        node.children[actions[chosen]], state, traversing_player, perspective, rng,
    )


# ---------------------------------------------------------------------------
# Expansion pass (PUCT-mixture walk to find a leaf)
# ---------------------------------------------------------------------------


def _select_leaf_for_expansion(
    root: GTCFRTreeNode,
    state: GTCFRState,
    exploring_player: chess.Color,
    rng: random.Random,
    *,
    keep_ids: set[int] | None = None,
    eq_mirror: "_RustEqMirror | None" = None,
) -> GTCFRTreeNode | None:
    """Walk from root using exploring-player = PUCT-mixture, non-exploring
    = current strategy. Return the first non-terminal leaf encountered,
    or None if the entire reachable subtree is terminal/exhausted.

    When ``keep_ids`` is provided (Python ``id()`` of allowed nodes —
    typically from KLUSS k=2 keep-mask), restrict descent to children
    in that set. If all children at any node are outside the keep set,
    the walk halts and returns None (no expansion this iter for this
    root). Returned leaf must itself be in the keep set; if the walk
    arrives at a leaf outside the keep set, also returns None."""
    node = root
    while True:
        if node.is_terminal:
            return None
        if not node.is_expanded:
            if keep_ids is not None and id(node) not in keep_ids:
                return None
            return node
        actions = list(node.children.keys())
        if keep_ids is not None:
            actions = [a for a in actions if id(node.children[a]) in keep_ids]
        if not actions:
            return None
        info_set_id = node.info_set_id()
        if eq_mirror is not None:
            strategy_list = eq_mirror.strategy(info_set_id, actions)
        else:
            strategy_list = _current_strategy(
                info_set_id,
                actions,
                state,
                [_rule_action_key(node.rules, a) for a in actions],
            )
        strategy = {a: p for a, p in zip(actions, strategy_list, strict=False)}
        if node.to_move == exploring_player:
            move = select_action_for_exploring_player(
                info_set_id,
                actions,
                state,
                strategy,
                rng=rng,
                eq_mirror=eq_mirror,
                rules=node.rules,
            )
        else:
            idx = _sample(strategy_list, rng)
            move = actions[idx]
        node = node.children[move]


# ---------------------------------------------------------------------------
# Rust equilibrium-pass bridge (Option B: Rust owns the tree mirror + state)
# ---------------------------------------------------------------------------


class _RustEqMirror:
    """Bridges the Python GT-CFR tree to the native ``fow_rust.EqEngine``.

    On the Rust equilibrium path, ALL per-(infoset, action) CFR state lives in
    Rust. Python keeps only the tree topology and the expansion pass (Stockfish
    MultiPV). This object owns the EqEngine handle and the two interning maps:
    info_set_id tuple -> u32, and id(node) -> rust node id.

    The equilibrium walk runs entirely in Rust (``equilibrium_pass``); the
    expansion walk stays in Python but reads strategy + PUCT from here so the
    tree grows byte-identically to the Python reference. Mirrors
    ``_current_strategy`` / ``puct_score`` / ``_q_value`` / ``_empirical_variance``
    exactly (same f64 arithmetic) — see those for the formulas.
    """

    def __init__(self, eq_rng: random.Random):
        import fow_rust  # local import; only needed on the Rust path
        st = eq_rng.getstate()[1]
        self.eng = fow_rust.EqEngine(list(st[:624]), st[624])
        self._iset: dict = {}
        self._node_ids: dict = {}

    def intern_iset(self, tup: Hashable) -> int:
        i = self._iset.get(tup)
        if i is None:
            i = len(self._iset)
            self._iset[tup] = i
        return i

    def register(self, node: "GTCFRTreeNode") -> int:
        """Register a node (idempotent); returns its rust id. No child linking."""
        nid = self._node_ids.get(id(node))
        if nid is not None:
            return nid
        nid = self.eng.add_node(
            node.rules.is_first_player(node.to_move),
            node.is_terminal,
            float(node.leaf_value) if node.leaf_value is not None else 0.0,
            float(node.terminal_value(node.rules.first_player)),
            self.intern_iset(node.info_set_id()),
        )
        self._node_ids[id(node)] = nid
        return nid

    def sync_expansion(self, parent: "GTCFRTreeNode", state: GTCFRState) -> None:
        """Mirror an ``expand_leaf`` into Rust: register children in
        ``parent.children`` order (= the order strategy/regret folds depend on),
        link them, and replicate exactly the regret seeds expand_leaf just wrote
        to ``state.regrets`` for THIS parent's children (not the whole infoset
        dict — a shared infoset's other keys must keep their Rust-evolved
        values)."""
        pid = self.register(parent)
        keys: list[int] = []
        child_ids: list[int] = []
        for move, child in parent.children.items():
            child_ids.append(self.register(child))
            keys.append(_mk(move))
        if keys:
            self.eng.link_children(pid, keys, child_ids)
        iset = self.intern_iset(parent.info_set_id())
        seeds = state.regrets.get(parent.info_set_id())
        if seeds is not None:
            for move in parent.children:
                k = _mk(move)
                self.eng.seed_regret(iset, k, float(seeds[k]))

    # --- read-side mirrors used by the expansion walk + snapshots ---

    def strategy(self, info_set_id: Hashable, actions: list[chess.Move]) -> list[float]:
        return self.eng.current_strategy(self.intern_iset(info_set_id),
                                         [_mk(a) for a in actions])

    def visits(self, info_set_id: Hashable) -> int:
        return self.eng.visits_get(self.intern_iset(info_set_id))

    def _q_value(self, iset: int, k: int) -> float:
        n = self.eng.visit_count_get(iset, k)
        if n == 0:
            return 0.0
        return self.eng.value_sum_get(iset, k) / n

    def _empirical_variance(self, iset: int, k: int) -> float:
        n_real = self.eng.visit_count_get(iset, k)
        n_total = n_real + 2
        sum_x = self.eng.value_sum_get(iset, k)
        sum_x2 = self.eng.value_sq_sum_get(iset, k) + _PRIOR_VARIANCE * 2
        if n_total <= 1:
            return _PRIOR_VARIANCE
        mean = sum_x / n_total
        return max(0.0, sum_x2 / n_total - mean * mean)

    def puct_score(self, info_set_id: Hashable, action: chess.Move,
                   *, c: float = _PUCT_C) -> float:
        iset = self.intern_iset(info_set_id)
        k = _mk(action)
        q = self._q_value(iset, k)
        sigma = math.sqrt(self._empirical_variance(iset, k))
        n_infoset = max(1, self.eng.visits_get(iset))
        n_action = self.eng.visit_count_get(iset, k)
        explore = c * sigma * math.sqrt(n_infoset) / (1 + n_action)
        return q + explore


# ---------------------------------------------------------------------------
# Top-level coordinator
# ---------------------------------------------------------------------------


@dataclass
class GTCFRSolution:
    """Output of solve_growing_subgame."""

    strategy_at_root: dict[chess.Move, float]
    """Last-iterate strategy at the root (PCFR+ convention)."""

    value_at_root: float
    """Empirical Q-value at the root, averaged over visits."""

    iterations: int
    info_set_count: int
    tree_node_count: int


@dataclass
class MultiRootGTCFRSolution:
    """Output of solve_multiroot_growing_subgame.

    Strategy is computed at the SHARED root infoset (all roots have the
    same to_move + empty observation history, so they share infoset
    keys; the multi-root architecture is what gives KLUSS its
    cross-truth reasoning).
    """

    strategy_at_root: dict[chess.Move, float]
    value_at_root: float
    iterations: int
    info_set_count: int
    total_tree_nodes: int
    n_roots: int
    elapsed_seconds: float
    """Wall-time consumed (matters when time_budget_seconds is set)."""

    strategy_history_at_root: list[dict] = field(default_factory=list)
    """Per-iteration snapshots of the root-infoset strategy. Used by
    A6 purification (stable-actions filter checks support continuously
    for t > T_{1/2})."""

    action_values_at_root: dict[chess.Move, float] = field(default_factory=dict)
    """Empirical mean EV per action at the root infoset, from the
    perspective player's POV. A6.2 (regime selection) reads this to
    decide whether to play deterministically (Resolve regime, top-1)
    or mix top-m ≤ 3 (Maxmargin regime). Computed as
    state.value_sum / state.visit_counts at the root infoset.
    Empty dict if no visits accumulated (shouldn't happen in practice)."""

    t_half: int = 0
    """Iteration index at which half the wall budget had elapsed (when
    time_budget_seconds is set) or iterations // 2 otherwise. The
    stable-actions filter checks support continuously for t > t_half."""

    component_ms: dict[str, float] = field(default_factory=dict)
    """Per-component wall-time breakdown across the solve, in ms. Keys:
    ``sf_eval``, ``sf_children`` (Stockfish leaf eval — measured at the
    StockfishLeafEval boundary, via the ``eval_wall_ns`` + ``children_wall_ns``
    counters); ``eq_pass``, ``select_leaf``, ``kluss``, ``expand_seed``
    (rust-tree-driver call sites). Empty when the solver doesn't populate it
    (Python-tree path)."""

    root_ids: list[int] = field(default_factory=list)
    """Rust-tree root node ids used by this solve. Surfaced so callers can
    chain Phase 2a carryover across choose_move calls (next pick walks these
    roots to discover depth-2 grandchildren under the just-played action).
    Empty on the Python-tree path."""


def _count_nodes(root: GTCFRTreeNode) -> int:
    n = 1
    for child in root.children.values():
        n += _count_nodes(child)
    return n


def solve_growing_subgame(
    root: GTCFRTreeNode,
    *,
    stockfish_eval,
    perspective: chess.Color,
    iterations: int,
    expansion_budget: int | None = None,
    rng: random.Random | None = None,
) -> GTCFRSolution:
    """One-sided GT-CFR over a growing game tree.

    Each iteration:
    1. Equilibrium pass — PCFR+ traversal over the currently-expanded
       tree. Alternates traversing player by iteration index.
    2. Expansion pass — walk from root using exploring-player =
       PUCT-mixture, non-exploring = current solved strategy. Expand
       the first unexpanded leaf encountered.

    Args:
        root: tree root (typically constructed via root_node(board)).
        stockfish_eval: a StockfishLeafEval instance (or duck-compatible).
        perspective: which color is the search's "from-POV" reference.
            Leaf evals are stored in this POV; the equilibrium pass
            assumes traversing_player == perspective for sign convention.
        iterations: total equilibrium passes to run.
        expansion_budget: max expansions performed. Defaults to
            ``iterations`` (one expansion per iter).
        rng: deterministic RNG; defaults to fresh random.Random(0).
    """
    if rng is None:
        rng = random.Random(0)
    # Equilibrium pass gets its OWN deterministic RNG stream, isolated from the
    # expansion pass (which keeps `rng`). The equilibrium walk's only randomness
    # is the per-opp-node external sampling in _equilibrium_traverse; isolating
    # those draws is the prerequisite for the Rust port to reproduce a single
    # MT19937 stream byte-for-byte. Seeded off `rng` so the search stays
    # deterministic given the caller's seed.
    eq_rng = random.Random(rng.getrandbits(64))
    if expansion_budget is None:
        expansion_budget = iterations
    state = GTCFRState()

    # Bootstrap: root must be expanded so the first equilibrium pass
    # has structure to walk. (If root is terminal we have nothing to do.)
    if not root.is_terminal and not root.is_expanded:
        expand_leaf(root, state, stockfish_eval=stockfish_eval, perspective=perspective)
    expansions_done = 1 if root.is_expanded else 0

    for t in range(iterations):
        # Equilibrium pass — alternate traversing player each iter.
        # Standard external-sampling CFR requires both players' regrets
        # to be updated; without alternating, the non-perspective player
        # never refines and the perspective player's strategy degenerates
        # to "best response to a fixed bad strategy". Leaf values are
        # stored in `perspective` POV; _equilibrium_traverse sign-flips
        # them when traversing_player != perspective.
        for traversing_player in (perspective, root.rules.opponent(perspective)):
            _equilibrium_traverse(root, state, traversing_player, perspective, eq_rng)

        # Expansion pass — alternate exploring player by iter index.
        if expansions_done < expansion_budget:
            exploring = root.rules.first_player if t % 2 == 0 else root.rules.second_player
            leaf = _select_leaf_for_expansion(root, state, exploring, rng)
            if leaf is not None:
                expand_leaf(
                    leaf, state,
                    stockfish_eval=stockfish_eval,
                    perspective=perspective,
                )
                expansions_done += 1

    # Last-iterate strategy at the root.
    root_info_set = root.info_set_id()
    actions_at_root = list(root.children.keys())
    if not actions_at_root:
        return GTCFRSolution(
            strategy_at_root={},
            value_at_root=0.0,
            iterations=iterations,
            info_set_count=0,
            tree_node_count=_count_nodes(root),
        )
    last = state.last_strategy.get(root_info_set, {})
    root_action_keys = [_rule_action_key(root.rules, a) for a in actions_at_root]
    if last:
        raw = [last.get(a, 0.0) for a in actions_at_root]
        total = sum(raw)
        if total > 0:
            strat = {a: r / total for a, r in zip(actions_at_root, raw, strict=False)}
        else:
            strat_list = _current_strategy(
                root_info_set,
                actions_at_root,
                state,
                root_action_keys,
            )
            strat = dict(zip(actions_at_root, strat_list, strict=False))
    else:
        strat_list = _current_strategy(
            root_info_set,
            actions_at_root,
            state,
            root_action_keys,
        )
        strat = dict(zip(actions_at_root, strat_list, strict=False))

    # Root value estimate: weight Q(I,a) by strategy.
    value = 0.0
    for a, k in zip(actions_at_root, root_action_keys, strict=False):
        n = state.visit_counts[root_info_set][k]
        if n > 0:
            value += strat[a] * (state.value_sum[root_info_set][k] / n)

    return GTCFRSolution(
        strategy_at_root=strat,
        value_at_root=value,
        iterations=iterations,
        info_set_count=len(state.regrets),
        tree_node_count=_count_nodes(root),
    )


# ---------------------------------------------------------------------------
# Multi-root KLUSS-flavored coordinator (Phase A5)
# ---------------------------------------------------------------------------


def sample_roots_from_P(
    iter_positions: Iterable[str],
    *,
    to_move: chess.Color,
    n: int,
    rng: random.Random,
    rules: Rules = _DEFAULT_RULES,
) -> list[GTCFRTreeNode]:
    """Reservoir-sample ``n`` board FENs from a streaming `P` iterator and
    build ``GTCFRTreeNode`` roots from them.

    Each root has empty observation history (fresh subgame); they share
    the root infoset key ``(to_move, (), ())`` so PCFR+ regret tables
    at the root are shared automatically across truths.

    Returns ``min(n, |P|)`` roots — if ``P`` is smaller than ``n``,
    sample without replacement gives every position.
    """
    reservoir: list[str] = []
    seen = 0
    for fen in iter_positions:
        seen += 1
        if len(reservoir) < n:
            reservoir.append(fen)
            continue
        i = rng.randint(0, seen - 1)
        if i < n:
            reservoir[i] = fen
    roots: list[GTCFRTreeNode] = []
    for fen in reservoir:
        board = rules.board_from_fen(fen)
        roots.append(root_node(board, to_move=to_move, rules=rules))
    return roots


def _multi_count_nodes(roots: list[GTCFRTreeNode]) -> int:
    return sum(_count_nodes(r) for r in roots)


def solve_multiroot_growing_subgame(
    roots: list[GTCFRTreeNode],
    *,
    stockfish_eval,
    perspective: chess.Color,
    iterations: int,
    expansion_budget: int | None = None,
    rng: random.Random | None = None,
    time_budget_seconds: float | None = None,
    kluss_k: int | None = None,
    use_rust_eq: bool = False,
) -> MultiRootGTCFRSolution:
    """Multi-root one-sided GT-CFR with shared regret tables — KLUSS-flavored.

    Each root in ``roots`` represents a sampled truth from the player's
    belief P. All roots share regret tables via the per-infoset
    ``GTCFRState``, so two roots that hit the same observation-history
    infoset at any depth contribute to the same regret table. This is
    the cross-truth reasoning that KLUSS provides — no per-truth
    aggregation step is needed.

    Each iteration:
    1. Equilibrium pass — walk EVERY root with the current alternating
       traverser. Regrets accumulate at shared infosets.
    2. Expansion pass — across all roots, find one non-terminal leaf
       via PUCT-mixture walk (alternating exploring player) and expand
       it via Stockfish MultiPV.

    Args:
        roots: list of fresh GTCFRTreeNode roots (typically from
            sample_roots_from_P).
        stockfish_eval: StockfishLeafEval instance (or compatible).
        perspective: the player's POV. Leaf evals stored in this POV;
            traversals from the other player sign-flip leaf reads.
        iterations: target number of equilibrium passes.
        expansion_budget: max expansions across all roots. Defaults to
            iterations × len(roots).
        rng: deterministic RNG. Defaults to random.Random(0).
        time_budget_seconds: if set, stops as soon as cumulative wall
            time exceeds this — anytime algorithm.

    Returns:
        MultiRootGTCFRSolution. ``strategy_at_root`` is the last-iterate
        strategy at the shared root infoset.
    """
    if not roots:
        raise ValueError("at least one root required")
    if rng is None:
        rng = random.Random(0)
    # Equilibrium pass gets its OWN deterministic RNG stream, isolated from the
    # expansion pass (which keeps `rng`). See solve_growing_subgame for the
    # rationale: this is the byte-equality prerequisite for the Rust port of
    # _equilibrium_traverse. Seeded off `rng` so the search stays deterministic.
    eq_rng = random.Random(rng.getrandbits(64))
    # Rust equilibrium path (Option B): the mirror owns the EqEngine + the CFR
    # state; the Python `state` keeps only the regret seeds expand_leaf writes
    # (sync_expansion replicates them into Rust). Default off — Python path is
    # byte-identical to before.
    eq_mirror = _RustEqMirror(eq_rng) if use_rust_eq else None
    if expansion_budget is None:
        expansion_budget = iterations * len(roots)
    state = GTCFRState()
    t_start = time.monotonic()
    # Game seam: all roots share one Rules (set by sample_roots_from_P). Drives
    # the per-iteration color alternation below.
    rules = roots[0].rules if roots else _DEFAULT_RULES

    # Bootstrap: expand each non-terminal root so the equilibrium pass
    # has something to walk.
    expansions_done = 0
    for r in roots:
        if not r.is_terminal and not r.is_expanded:
            expand_leaf(r, state, stockfish_eval=stockfish_eval,
                        perspective=perspective, eq_mirror=eq_mirror)
            expansions_done += 1
    # Register every root in the mirror — terminal roots aren't expanded above
    # but equilibrium_pass still walks them (returns their terminal value).
    root_rust_ids = (
        [eq_mirror.register(r) for r in roots] if eq_mirror is not None else []
    )
    # Incremental per-root subtree size for the round-robin best_root pick.
    # Recomputing _count_nodes(r) for every root every iteration is
    # O(roots × tree × iters) — ~30% of wall in the 2026-05-26 Rust-path
    # profile (4.99M _count_nodes calls). Seed once post-bootstrap and bump by
    # the children added at each expansion; every node is counted exactly once
    # when added, so this equals _count_nodes(r) and the best_root pick (hence
    # all results) is byte-identical.
    root_sizes = {id(r): _count_nodes(r) for r in roots}

    iters_completed = 0
    strategy_history: list[dict] = []
    half_budget_reached_at: int | None = None
    for t in range(iterations):
        # Time-budget check at iteration boundary.
        elapsed = time.monotonic() - t_start
        if time_budget_seconds is not None and elapsed >= time_budget_seconds:
            break

        # Record when we cross the half-time-budget mark (for A6 purification
        # stable-actions filter).
        if (time_budget_seconds is not None
                and half_budget_reached_at is None
                and elapsed >= time_budget_seconds / 2.0):
            half_budget_reached_at = t

        # Equilibrium pass: alternate traverser, walk all roots. On the Rust
        # path the whole 2×N walk runs natively (and owns the eq RNG stream).
        if eq_mirror is not None:
            eq_mirror.eng.equilibrium_pass(root_rust_ids, rules.is_first_player(perspective))
        else:
            for traversing_player in (perspective, rules.opponent(perspective)):
                for r in roots:
                    _equilibrium_traverse(r, state, traversing_player, perspective, eq_rng)

        # Snapshot root-infoset strategy for purification's stable-actions
        # filter. Cheap (one dict copy per iteration).
        if roots:
            root_info_set = roots[0].info_set_id()
            all_actions = set()
            for r in roots:
                all_actions.update(r.children.keys())
            if all_actions:
                aa = list(all_actions)
                if eq_mirror is not None:
                    strat_now = eq_mirror.strategy(root_info_set, aa)
                else:
                    strat_now = _current_strategy(
                        root_info_set,
                        aa,
                        state,
                        [_rule_action_key(rules, a) for a in aa],
                    )
                strategy_history.append(dict(zip(aa, strat_now, strict=False)))

        # Expansion pass: pick one root × leaf via PUCT-mixture walk.
        if expansions_done < expansion_budget:
            exploring = rules.first_player if t % 2 == 0 else rules.second_player
            # Pick the root with fewest expansions so far (round-robin-ish);
            # ties broken by random choice.
            best_root = min(
                roots,
                key=lambda r: (root_sizes[id(r)], rng.random()),
            )
            # KLUSS k=2: compute keep-set from the connectivity graph of
            # the current Γ̃ and restrict leaf-selection to nodes within
            # I^(k+1). Recomputed each iteration as the tree grows.
            # Disabled when kluss_k is None — preserves prior behavior.
            keep_ids: set[int] | None = None
            if kluss_k is not None:
                from .kluss import kluss_keep_mask
                source_infosets = {roots[0].info_set_id()}
                nodes, keep_indices = kluss_keep_mask(
                    roots, source_infosets, k=kluss_k,
                )
                keep_ids = {id(nodes[i]) for i in keep_indices}
            leaf = _select_leaf_for_expansion(
                best_root, state, exploring, rng, keep_ids=keep_ids,
                eq_mirror=eq_mirror,
            )
            # Soft KLUSS (FOW_KLUSS_SOFT, 2026-06-12): the keep-mask walk
            # DEADLOCKS once the strategy concentrates — every walk follows
            # the favored line to the keep boundary (children filter to
            # empty) and returns None, so the expansion budget goes unspent
            # forever. Measured: |P|=1 solves expand 4-10 nodes of an
            # eb=2000 budget (sf_ch_size telemetry), leaving the engine on
            # depth-1 values in thin-fog positions — where the 2026-06-11
            # H2H fog deaths cluster (fatal plies at |P|=1-500). At small
            # |P| the connectivity graph has no cross-world edges, so
            # "knowledge distance" degenerates to tree depth ≤ k+1 — the
            # paper's |P|=1 case should degenerate to a FULL perfect-info
            # subgame, not a 3-ply one. Soft mode keeps the KLUSS scope as
            # a PREFERENCE: when no in-scope leaf is expandable, fall back
            # to an unrestricted walk instead of skipping the iteration.
            if (leaf is None and keep_ids is not None
                    and os.environ.get("FOW_KLUSS_SOFT") == "1"):
                leaf = _select_leaf_for_expansion(
                    best_root, state, exploring, rng, keep_ids=None,
                    eq_mirror=eq_mirror,
                )
            if leaf is not None:
                added = expand_leaf(
                    leaf, state,
                    stockfish_eval=stockfish_eval,
                    perspective=perspective,
                    eq_mirror=eq_mirror,
                )
                expansions_done += 1
                # Leaf is within best_root's subtree → its new children grow
                # that root's count by exactly `added`.
                root_sizes[id(best_root)] += added
        iters_completed += 1

    # Resolve t_half for purification. If we had a time budget, use the
    # iteration index where wall time crossed budget/2. Otherwise use
    # the conventional iters_completed // 2.
    if half_budget_reached_at is not None:
        t_half = half_budget_reached_at
    else:
        t_half = iters_completed // 2

    # Last-iterate strategy at the SHARED root infoset.
    # All roots share info_set_id == (to_move, (), ()) since their
    # observation histories are empty.
    root_info_set = roots[0].info_set_id()
    # Union of all legal actions across roots (different truths admit
    # different move sets; the regret table is keyed by action and
    # only contains actions visited by SOME root).
    all_actions = set()
    for r in roots:
        all_actions.update(r.children.keys())
    actions_at_root = list(all_actions)
    if not actions_at_root:
        return MultiRootGTCFRSolution(
            strategy_at_root={},
            value_at_root=0.0,
            iterations=iters_completed,
            info_set_count=len(state.regrets),
            total_tree_nodes=_multi_count_nodes(roots),
            n_roots=len(roots),
            elapsed_seconds=time.monotonic() - t_start,
            strategy_history_at_root=strategy_history,
            t_half=t_half,
        )
    # On the Rust path `state.last_strategy` is empty (equilibrium ran in Rust),
    # so this falls through to the current-strategy computation — which matches
    # the Python path, whose int-keyed `last` dict yields 0.0 under the
    # Move-keyed lookup, also falling through. Route the strategy + value reads
    # through the mirror when on the Rust path.
    def _root_strategy(actions):
        if eq_mirror is not None:
            return eq_mirror.strategy(root_info_set, actions)
        return _current_strategy(
            root_info_set,
            actions,
            state,
            [_rule_action_key(rules, a) for a in actions],
        )

    last = state.last_strategy.get(root_info_set, {})
    if last:
        raw = [last.get(a, 0.0) for a in actions_at_root]
        total = sum(raw)
        if total > 0:
            strat = {a: r / total for a, r in zip(actions_at_root, raw, strict=False)}
        else:
            strat = dict(
                zip(actions_at_root, _root_strategy(actions_at_root), strict=False)
            )
    else:
        strat = dict(
            zip(actions_at_root, _root_strategy(actions_at_root), strict=False)
        )

    # Empirical mean EV per action at the root infoset. A6.2 (regime
    # selection) uses these to compute the margin = best - second_best;
    # margin >= threshold → Maxmargin regime (mix top-m); else Resolve
    # regime (deterministic top).
    action_values: dict[chess.Move, float] = {}
    value = 0.0
    for a in actions_at_root:
        if eq_mirror is not None:
            iset = eq_mirror.intern_iset(root_info_set)
            n = eq_mirror.eng.visit_count_get(iset, _mk(a))
            mean = eq_mirror.eng.value_sum_get(iset, _mk(a)) / n if n > 0 else None
        else:
            key = _rule_action_key(rules, a)
            n = state.visit_counts[root_info_set][key]
            mean = state.value_sum[root_info_set][key] / n if n > 0 else None
        if n > 0:
            action_values[a] = mean
            value += strat[a] * mean

    return MultiRootGTCFRSolution(
        strategy_at_root=strat,
        value_at_root=value,
        iterations=iters_completed,
        info_set_count=len(state.regrets),
        total_tree_nodes=_multi_count_nodes(roots),
        n_roots=len(roots),
        elapsed_seconds=time.monotonic() - t_start,
        strategy_history_at_root=strategy_history,
        action_values_at_root=action_values,
        t_half=t_half,
    )


# ---------------------------------------------------------------------------
# WS2: use_rust_tree path — authoritative Rust search tree drives the loop
# ---------------------------------------------------------------------------


def _decode_mk(key: int) -> chess.Move:
    """Inverse of _mk for the Rust tree's action keys: from | to<<6 | promo<<12."""
    return chess.Move(key & 0x3F, (key >> 6) & 0x3F, promotion=((key >> 12) & 0x7) or None)


def _rust_expand_and_seed(
    eng, node_id: int, stockfish_eval, perspective, rules, persp_white: bool
) -> int:
    """Stockfish FFI batch boundary (design option A): the Rust tree expands a
    node, Python evaluates the new leaves (mirroring expand_leaf's child-eval:
    terminal value from expand_node, leaf-eval MultiPV where the position is
    search-valid, material fallback otherwise), and Rust seeds leaf values +
    smart regret. Game-agnostic via ``rules``: chess uses full FENs + python-
    chess legality; mini uses self-describing FENs + an always-valid leaf eval.
    Returns the number of children added."""
    children = eng.expand_node(node_id, persp_white)
    if not children:
        return 0
    board = rules.board_from_fen(eng.node_fen(node_id))
    child_evals = (
        stockfish_eval.evaluate_children(board, perspective)
        if rules.is_search_valid(board)
        else {}
    )
    values = []
    for (frm, to, promo, cid, is_term, term_val, _wk, _bk) in children:
        if is_term:
            values.append(term_val)
            continue
        mv = rules.make_move(frm, to, promo)
        if mv in child_evals:
            values.append(child_evals[mv])
        else:
            values.append(
                rules.material_leaf_eval(rules.board_from_fen(eng.node_fen(cid)), perspective)
            )
    eng.seed_expansion(node_id, values, persp_white)
    return len(children)


def _union_root_action_keys(eng, root_ids: list[int]) -> list[int]:
    """Union of the roots' child action keys = actions_at_root. (Order is
    irrelevant: current_strategy normalizes over the same key SET, so per-key
    probabilities are order-independent.)"""
    seen: set[int] = set()
    out: list[int] = []
    for rid in root_ids:
        keys, _ = eng.node_children(rid)
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


# Static-exchange piece values (centipawns) for the gadget's defensive material
# floor. KING is huge so a king capture dominates any material swap.
_SEE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
}


def _see_gain(board, to_sq):
    """Static exchange evaluation on ``to_sq`` for ``board.turn`` — the net
    centipawns the side to move wins by starting the capture sequence with its
    least-valuable attacker, both sides recapturing optimally (stand-pat allowed).
    Board-copy recursion, so x-ray attackers are revealed; pins are ignored
    (standard SEE convention). 0 when there is no capturing attacker."""
    victim = board.piece_at(to_sq)
    if victim is None:
        return 0
    attackers = board.attackers(board.turn, to_sq)
    if not attackers:
        return 0
    lva_sq = min(attackers, key=lambda s: _SEE_VAL[board.piece_at(s).piece_type])
    nb = board.copy(stack=False)
    try:
        nb.push(chess.Move(lva_sq, to_sq))
    except (AssertionError, ValueError):
        return 0  # e.g. a capture-promotion missing its promo piece — skip
    return max(0, _SEE_VAL[victim.piece_type] - _see_gain(nb, to_sq))


def _max_opponent_material_gain(after, my_color, min_cp):
    """Largest SEE the opponent (``after.turn``) can win next ply by capturing one
    of ``my_color``'s non-pawn pieces. ``min_cp`` prunes cheap targets — a hung pawn
    is not the dilution failure this floors."""
    targets = set()
    for mv in after.generate_pseudo_legal_moves():
        if not after.is_capture(mv):
            continue
        victim = after.piece_at(mv.to_square)
        if (
            victim is not None
            and victim.color == my_color
            and victim.piece_type != chess.KING
            and _SEE_VAL[victim.piece_type] >= min_cp
        ):
            targets.add(mv.to_square)
    best = 0
    for sq in targets:
        g = _see_gain(after, sq)
        if g > best:
            best = g
    return best


def _apply_defensive_danger(
    per_world, action_keys, worlds, perspective_white, rules: Rules = _DEFAULT_RULES
):
    """DMX-style defensive danger floor on the gadget's per-world move values.

    For each (action ``a``, world ``j``): play ``a`` in world ``j``; if the result
    lets the opponent win on their next ply, floor ``value(a, j)`` to the post-loss
    value — so the gadget's worst-case sees the hang DIRECTLY instead of waiting on
    the i-limited solve to surface it in the (often rare) bad worlds. Two tiers:

      * **King** (terminal): if ``a`` leaves our king capturable, floor to the
        king-loss value (``king_capture_imminent``) — dominates everything.
      * **Material** (the generalization beyond DMX, whose general == king): floor to
        ``tanh((material_after_move - SEE_gain) / scale)`` when the opponent can win
        >= ``FOW_GADGET_DEFENSIVE_MIN_CP`` (default 300) by static exchange. This is
        what addresses chess's QUEEN/ROOK-hang dilution corpus (bg4, Nxf7, 74126ceb,
        a84dbaf9) that the king-only floor cannot. Set MIN_CP huge for king-only.

    Move-value level, not the king-aware LEAF (a terminal detector that only reaches
    ``value(a, j)`` if the solve expands the capture). FoW-legal hang moves are
    *pseudo-legal* (standard chess calls "moving into check" illegal) — exactly the
    moves to floor — so we gate on ``is_pseudo_legal``. Mutates ``per_world`` in
    place. See ``docs/engine/gadget-blueprint-STATUS.md``.
    """
    from ..evaluator import material_score
    from .leaf_eval import _TANH_SCALE_CP

    persp = rules.first_player if perspective_white else rules.second_player
    min_cp = float(os.environ.get("FOW_GADGET_DEFENSIVE_MIN_CP", "300"))
    for k in action_keys:
        mv = rules.decode_action_key(k)
        for j, cvals in enumerate(per_world):
            v = cvals.get(k)
            if v is None:
                continue  # action illegal in world j — nothing to floor
            board_j = worlds[j]
            world_move = rules.matching_pseudo_legal_move(board_j, mv)
            if world_move is None:
                continue
            after = rules.apply(board_j, world_move)
            # Royal hang dominates (terminal): king for chess, general for DMX.
            kv = rules.royal_capture_imminent(after, persp)
            if kv is not None:
                if kv < v:
                    cvals[k] = kv
                continue
            # Material SEE is chess-only. DMX's urgent catastrophe is the general,
            # and its default Rules hook above already handles it.
            if not isinstance(after, chess.Board):
                continue
            # Material hang: floor to the post-loss material value.
            gain = _max_opponent_material_gain(after, persp, min_cp)
            if gain >= min_cp:
                floor_val = math.tanh((material_score(after, persp) - gain) / _TANH_SCALE_CP)
                if floor_val < v:
                    cvals[k] = floor_val


def _nonuniform_resolve_alpha(blueprint, worlds):
    """Obscuro's non-uniform Resolve root distribution (Appendix C.3):

        alpha(J) = ½ (y(J)/Σ_J' y(J') + 1/m)

    over the m sampled worlds (opponent root infosets), where ``y(J)`` is the
    carried opponent reach (``blueprint.reach``) — the probability the opponent's
    blueprint strategy generates world J. The even mixture with uniform keeps
    every world's weight positive (Resolve stays correct under any fully-mixed
    root distribution); the y-term focuses the safety effort on the worlds the
    opponent actually plays into (the paper's −32%-when-off lever).

    Returns ``None`` (= uniform) when no reach is carried: blueprint without a
    ``reach`` method (stub), empty worlds, or zero coverage (Σy == 0 — first
    move / carryover off / the sample missed every carried world). This is the
    single source of truth for the alpha math — both the read-only gadget
    (``_apply_resolve_gadget``) and the PROPER iterative gadget use it.
    """
    if not hasattr(blueprint, "reach"):
        return None
    m = len(worlds)
    if m == 0:
        return None
    yv = [max(0.0, float(blueprint.reach(w))) for w in worlds]
    ysum = sum(yv)
    if ysum <= 0.0:
        return None
    return [0.5 * (y / ysum + 1.0 / m) for y in yv]


def _apply_resolve_gadget(
    eng, root_ids, action_keys, perspective_white, blueprint, margin, worlds=None,
    cvar_q=0.1, rules: Rules = _DEFAULT_RULES, gadget_faithful=None, gadget_alpha=None,
):
    """Read-only Resolve/Maxmargin gadget at the root (MVP).

    For each action ``a`` and sampled world (opponent infoset) ``j``, read
    ``value(a, j)`` — our value of playing ``a`` in world ``j`` (via
    ``eng.root_child_values``, no tree mutation) — and form the per-world margin
    relative to the blueprint baseline:

        M(a, j) = value(a, j) - bp_value(j),   bp_value(j) = -opp_cfv(j) + margin.

    ``M >= 0`` is safe (we do no worse than the blueprint, so the opponent does
    no better). We pick ``a`` by Obscuro's objective + automatic switch
    (Appendix B.2/C.1):

      * a fully-safe action exists (``max_a CVaR_q M >= 0``) -> **Maxmargin**:
        ``argmax_a CVaR_q M(a, j)`` (win the most in the worst case — no
        over-caution in genuinely-winning positions);
      * otherwise -> **Resolve**: ``argmax_a CVaR_q min(0, M(a, j))`` (least
        exploitable; only penalizes worlds where ``a`` does worse than blueprint
        — this is what kills aggregation-dilution, since a uniform mean of raw
        values cannot).

    ``CVaR_q X`` is the mean of ``X`` over the worst ``ceil(q * n_worlds)``
    worlds (the smallest margins). ``cvar_q -> 0`` recovers pure Maxmargin /
    full-belief Resolve worst-case (a single world); ``cvar_q = 1`` recovers the
    full-belief mean. The default (0.1) makes both regimes robust to per-world
    value NOISE — see the CVaR comment below and the oracle de-risk.

    IMPORTANT: the objective is the negative-truncated MARGIN, not a capped
    value — rewarding winning worlds (an earlier bug) re-introduces dilution.
    Returns ``(strategy, action_values, value)`` (Resolve top-1 purified), or
    ``None`` if no action has a legal world. ``alpha(J)`` is uniform by default;
    FOW_GADGET_ALPHA (faithful path only, Slice 2) weights the Resolve mean by
    the Obscuro root distribution ``alpha(J)=½(y(J)/Σy + 1/m)`` from the carried
    opponent reach (``blueprint.reach``). Maxmargin's ``min_J`` is not
    alpha-weighted (paper). Top-K Maxmargin mixing and the iterative coupling
    back into the solve remain deferred (gadget-mvp-build-notes Slices 3/4).
    """
    per_world = [dict(pairs) for pairs in eng.root_child_values(list(root_ids), perspective_white)]
    if worlds is None:
        worlds = list(root_ids)
    # Per-world blueprint baseline in OUR POV: bp_value(j) = -opp_cfv(j), shifted
    # up by `margin` (the Maxmargin safety bar / gift gap). One blueprint eval
    # per world (Stockfish evals are EPD-cached; avoids the N_actions x N_worlds
    # blowup).
    bp_value = [(-blueprint.opp_cfv(worlds[j])) + margin for j in range(len(per_world))]
    # DMX-style defensive king-danger floor (FOW_GADGET_DEFENSIVE_DANGER; default
    # OFF -> byte-identical, this never runs). Floors value(a,j) when action a hangs
    # OUR king next ply in world j, so the gadget worst-case sees the hang directly
    # instead of waiting on the i-limited solve to surface it. See helper docstring.
    if os.environ.get("FOW_GADGET_DEFENSIVE_DANGER") == "1":
        _apply_defensive_danger(per_world, action_keys, worlds, perspective_white, rules)
    # Per action: maxmargin = worst-case per-world margin M(a,j) = value(a,j) -
    # bp_value(j) (>= 0 means safe — we do no worse than the blueprint, so the
    # opponent does no better); resolve = mean of the negative-truncated margins
    # (penalize only exploitation, ignore upside).
    # FOW_GADGET_FAITHFUL: Obscuro-faithful aggregation — full-belief Maxmargin
    # (MIN margin) + Resolve (MEAN of negative-truncated margins), uniform alpha.
    # Replaces our CVaR worst-tail (the over-defense deviation). The calibrated
    # carryover blueprint makes margins relative to what's achievable per world,
    # so naturally-bad worlds don't read as failures. Default OFF -> byte-identical.
    # Per-arm override (gadget_faithful arg) wins over the process env so a
    # v2-vs-v2 bakeoff can put the faithful gadget on ONE arm; None = read env.
    _faithful = (os.environ.get("FOW_GADGET_FAITHFUL") == "1"
                 if gadget_faithful is None else bool(gadget_faithful))
    _alpha_on = (os.environ.get("FOW_GADGET_ALPHA") == "1"
                 if gadget_alpha is None else bool(gadget_alpha))
    # FOW_GADGET_ALPHA (faithful only, Slice 2): non-uniform Obscuro root
    # distribution alpha(J)=½(y(J)/Σy + 1/m), y = the carried opponent reach
    # (blueprint.reach). The Resolve objective Σ_J α(J)[M]^- up-weights worlds
    # the opponent actually plays into (real danger) while the ½·(1/m) floor
    # never zeroes a world. Maxmargin (min_J M) stays unweighted (paper). When
    # no reach is carried (Σy==0: first move / carryover off) alpha is None ->
    # uniform -> recovers faithful v0. Default OFF -> alpha None -> byte-identical.
    alpha = None
    if _faithful and _alpha_on:
        alpha = _nonuniform_resolve_alpha(blueprint, worlds)
    maxmargin: dict[int, float] = {}
    resolve: dict[int, float] = {}
    mean_margin: dict[int, float] = {}  # full-belief mean (tie-break, below)
    for k in action_keys:
        margins = []
        idxs = []  # world index per margin (for alpha-weighting; unused otherwise)
        for j, cvals in enumerate(per_world):
            v = cvals.get(k)
            if v is None:
                continue  # action illegal in world j (e.g. slider blocked by a hidden piece)
            margins.append(v - bp_value[j])
            idxs.append(j)
        if not margins:
            continue
        mean_margin[k] = sum(margins) / len(margins)  # mean over ALL legal worlds
        # CVaR over the worst ``cvar_q`` fraction of worlds, not a single world
        # (``min``) or the full-belief mean. The worst-tail average is robust to
        # the two matrix-NOISE failure modes the oracle de-risk found on
        # 74126ceb (lab/debug_gadget_oracle_derisk.py + seed2_matrix.py): a
        # single spurious-loss world is diluted across the tail (pure Maxmargin's
        # ``min`` is killed by it), while a REAL catastrophe spread over many
        # worlds is captured rather than washed out by the safe majority
        # (full-belief Resolve re-dilutes it). cvar_q -> 0 (kq=1) recovers pure
        # Maxmargin/worst-case; cvar_q = 1 recovers the full-belief mean.
        if _faithful:
            # Obscuro-faithful: full-belief MIN (Maxmargin) + MEAN of
            # negative-truncated margins (Resolve). No worst-tail slice — the
            # active move's good majority isn't discarded. Maxmargin's min_J is
            # unweighted (paper); only Resolve takes the non-uniform alpha.
            maxmargin[k] = min(margins)
            if alpha is None:
                resolve[k] = sum(min(0.0, mm) for mm in margins) / len(margins)
            else:
                # alpha-weighted Resolve Σ_J α(J)[M]^-, renormalized over the
                # worlds legal for THIS action (mirrors the uniform mean's
                # /len(margins); the literal global Σ_J without renorm would
                # reward an action being illegal in the danger worlds).
                _num = 0.0
                _den = 0.0
                for _jj, mm in zip(idxs, margins, strict=False):
                    _num += alpha[_jj] * min(0.0, mm)
                    _den += alpha[_jj]
                resolve[k] = _num / _den if _den > 0.0 else 0.0
        else:
            margins.sort()
            kq = max(1, round(cvar_q * len(margins)))
            worst = margins[:kq]
            maxmargin[k] = sum(worst) / kq                      # CVaR of the raw margin
            resolve[k] = sum(min(0.0, m) for m in worst) / kq   # CVaR of the neg-truncated margin
    if not maxmargin:
        return None
    # Obscuro's automatic switch (Appendix C.1): if a fully-safe action exists
    # (some action's worst-case margin >= 0), use Maxmargin — win the most in
    # the worst case, NO over-caution. Otherwise use Resolve — least exploitable.
    if max(maxmargin.values()) >= 0.0:
        objective = maxmargin
    else:
        objective = resolve
    # FOW_GADGET_MEAN_TIEBREAK: break worst-case ties by the full-belief MEAN
    # margin. The pure worst-case objective can't distinguish two actions whose
    # worst tail is equally bad — e.g. in a fog position where EVERY move has a
    # losing tail world, all actions floor to the same CVaR and `max` picks one
    # by arbitrary dict order. That declined Kxh1/Kg3 (~79% king-survival across
    # the belief) for a 0%-survival move (g08 ply-197). Among actions within
    # `eps` of the best objective, prefer the highest mean margin → the action
    # that's safest ON AVERAGE. Fires ONLY on ties, so worst-case-distinct
    # positions (the gadget's measured win) are byte-identical. Default OFF;
    # bakeoff-gate before flipping (could over-fire in flat-value FoW spots).
    if os.environ.get("FOW_GADGET_MEAN_TIEBREAK") == "1":
        best_obj = max(objective.values())
        eps = float(os.environ.get("FOW_GADGET_TIEBREAK_EPS", "0.05"))
        contenders = [k for k in objective if best_obj - objective[k] <= eps]
        best_k = max(contenders, key=lambda k: mean_margin.get(k, float("-inf")))
    else:
        best_k = max(objective, key=objective.__getitem__)
    strat = {rules.decode_action_key(best_k): 1.0}  # Resolve regime: deterministic top-1
    action_values = {rules.decode_action_key(k): objective[k] for k in objective}
    return strat, action_values, objective[best_k]


def solve_multiroot_rust_tree(
    root_boards: list[chess.Board],
    *,
    stockfish_eval,
    perspective: chess.Color,
    iterations: int,
    expansion_budget: int | None = None,
    rng: random.Random | None = None,
    time_budget_seconds: float | None = None,
    kluss_k: int | None = None,
    kluss_soft: bool | None = None,
    record_strategy_history: bool = True,
    eq_engine=None,
    carryover_infosets: bool = False,
    carryover_subtree: bool = False,
    root_carryover_ids: list[int | None] | None = None,
    full_cfv_backprop: bool = False,
    resolve_gadget: bool = False,
    gadget_blueprint=None,
    gadget_margin: float = 0.0,
    gadget_cvar_q: float = 0.1,
    gadget_faithful: bool | None = None,
    gadget_alpha: bool | None = None,
    gadget_iterative: bool | None = None,
    rules: Rules = _DEFAULT_RULES,
) -> MultiRootGTCFRSolution:
    """WS2 use_rust_tree path: the authoritative Rust ``EqEngine`` tree drives the
    whole multi-root GT-CFR grow loop — select / expand / equilibrium / seed all
    in Rust, with Stockfish leaf eval crossing back to Python at the batch
    boundary. The Python ``GTCFRTreeNode`` tree + ``_RustEqMirror`` are gone.

    Byte-identical target vs solve_multiroot_growing_subgame(use_rust_eq=True):
    same rng discipline (eq stream forked off one rng.getrandbits(64); selection
    rng = rng's state after that fork), same bootstrap / round-robin / expansion
    eval, so the last-iterate root strategy matches per move."""
    import fow_rust

    if not root_boards:
        raise ValueError("at least one root required")
    if rng is None:
        rng = random.Random(0)
    # Game-agnostic: persp_white = "is the perspective the first player" (chess
    # white / mini red), the bool the Rust CFR core keys on.
    persp_white = rules.is_first_player(perspective)

    # Lever 1 Phase 1: if caller passes an existing eq_engine, reset its tree
    # state in place. RNG consumption is identical to per-call construction so
    # byte-parity holds.
    #
    # Lever 1 Phase 2 (variant b, carryover_infosets=True): use
    # reset_tree_keep_infosets to preserve the infoset_intern + per-infoset
    # CFR state across the call. New tree's search will warm-start at any
    # infoset whose (to_move, hist) hash matches a prior search's. Intentional
    # divergence from per-call construction; requires caller opt-in.
    eq_rng = random.Random(rng.getrandbits(64))
    eqst = eq_rng.getstate()[1]
    if eq_engine is None:
        eng = fow_rust.EqEngine(list(eqst[:624]), eqst[624])
    else:
        eng = eq_engine
        if carryover_subtree:
            # Phase 2a: preserve nodes + infosets + intern; KLUSS caches reset
            # (rebuilt during the new search).
            eng.reset_for_carryover(list(eqst[:624]), eqst[624])
        elif carryover_infosets:
            eng.reset_tree_keep_infosets(list(eqst[:624]), eqst[624])
        else:
            eng.reset_tree(list(eqst[:624]), eqst[624])
    selst = rng.getstate()[1]  # rng AFTER the getrandbits(64) fork
    eng.seed_select_rng(list(selst[:624]), selst[624])
    # Tell the Rust engine which game it's solving (board-specific methods —
    # add_root_from_fen / expand_node / node_fen — branch on this). Chess is the
    # default; idempotent + persists across reset_tree.
    if rules.name == "mini-xiangqi":
        eng.set_mini(True)
    elif rules.name == "xiangqi":
        eng.set_xiangqi(True)
    else:
        eng.set_mini(False)

    if expansion_budget is None:
        expansion_budget = iterations * len(root_boards)
    t_start = time.monotonic()

    # Component timers — accumulated ns across the solve, reported in ms
    # on MultiRootGTCFRSolution.component_ms. Always-on; one monotonic_ns()
    # call per measured component per iter is sub-microsecond. The
    # StockfishLeafEval boundary timers (sf_eval, sf_children) are read
    # as a delta across the whole solve at the end.
    sf_eval_ns_start = stockfish_eval.eval_wall_ns
    sf_children_ns_start = stockfish_eval.children_wall_ns
    eq_ns = 0
    select_ns = 0
    kluss_ns = 0
    expand_ns = 0
    # EXPERIMENT (FOW_EQ_MERGED): merged single-walk full-CFV eq pass — one walk
    # updating BOTH players' regrets instead of two (~2x fewer node visits). A
    # valid but NON-byte-identical CFR scheme (different per-iterate values), so
    # default OFF + strength-validated (bakeoff), not byte-parity. Only applies to
    # the full_cfv_backprop (gadget) regime; external sampling is untouched.
    _merged_eq = full_cfv_backprop and os.environ.get(
        "FOW_EQ_MERGED", "").strip().lower() in ("1", "true", "yes", "on")

    # Phase 2a: root_carryover_ids, when provided, gives a per-root
    # carryover hint. None means "add fresh via add_root_from_fen"; int
    # means "use this existing node_id as the new root" (subtree reuse).
    # The hint list must match root_boards length so non-carryover roots
    # still go through add_root_from_fen for board placement.
    root_ids: list[int] = []
    if root_carryover_ids is not None:
        if len(root_carryover_ids) != len(root_boards):
            raise ValueError(
                "root_carryover_ids must match root_boards length "
                f"({len(root_carryover_ids)} vs {len(root_boards)})"
            )
        for hint, board in zip(root_carryover_ids, root_boards, strict=False):
            if hint is None:
                root_ids.append(eng.add_root_from_fen(rules.root_fen(board)))
            else:
                root_ids.append(int(hint))
        # Invariant: root_ids must be duplicate-free when carryover is on.
        # Duplicate node_ids would cause the equilibrium pass to traverse
        # the same node twice per iter, double-counting regrets.
        #
        # Why this cannot happen in practice:
        #   - Roots are sampled WITHOUT REPLACEMENT from P (a set) so their
        #     board FENs are unique.
        #   - fen_to_node (in engine_v2.choose_move) maps FEN→node_id; looking
        #     up distinct FENs yields distinct dict keys → distinct values.
        #   - add_root_from_fen always allocates a fresh node, so None-hint
        #     roots are also unique.
        # This assertion is a regression guard: if any future refactor breaks
        # the uniqueness chain, it fires here rather than silently corrupting
        # the regret tables.
        if len(root_ids) != len(set(root_ids)):
            raise AssertionError(
                "Phase 2a carryover produced duplicate root_ids — "
                f"root_ids={root_ids}; this would double-count regrets."
            )
    else:
        root_ids = [eng.add_root_from_fen(rules.root_fen(b)) for b in root_boards]
    expansions_done = 0
    root_sizes: list[int] = []
    for rid in root_ids:
        sz = 1
        if not eng.node_is_terminal(rid):
            _t0 = time.monotonic_ns()
            sz += _rust_expand_and_seed(eng, rid, stockfish_eval, perspective, rules, persp_white)
            expand_ns += time.monotonic_ns() - _t0
            expansions_done += 1
        root_sizes.append(sz)

    root_infoset = eng.node_infoset(root_ids[0])
    iters_completed = 0
    strategy_history: list[dict] = []
    half_budget_reached_at: int | None = None

    # PROPER (iterative) Resolve gadget — Step 1 (FOW_GADGET_ITERATIVE; per-arm arg
    # gadget_iterative overrides the env). Couples the follow/exit gadget INTO the
    # eq loop instead of the read-only post-hoc cap: each iter we read every world's
    # opponent-POV value under the current strategy, step the gadget's RM+, and
    # weight the next eq pass by alpha(J)·P(follow|J). Worlds the opponent would EXIT
    # stop pressuring our shared root strategy → the safe move is computed IN the
    # solve. Needs full-CFV (Obscuro) values; the gadget regime already forces it.
    # Default OFF → this block never builds the gadget → byte-identical.
    _iterative_gadget = (
        resolve_gadget
        and gadget_blueprint is not None
        and (
            bool(gadget_iterative) if gadget_iterative is not None
            else os.environ.get("FOW_GADGET_ITERATIVE") == "1"
        )
    )
    _gadget = None
    if _iterative_gadget:
        # gift(J) = blueprint.opp_cfv(J) (opponent POV); the safety margin is applied
        # inside ResolveGadget. Built once per choose_move, one world per sampled root
        # (aligned with root_ids / root_boards order). With gadget_alpha on
        # (FOW_GADGET_ALPHA / per-arm), the root distribution is Obscuro's
        # non-uniform alpha(J)=½(y/Σy+1/m) from the carried opponent reach
        # (paper Appendix C.3 — the −32%-when-off lever); the in-solve world
        # weights become alpha(J)·P(follow|J). Otherwise (or with no carried
        # reach) alpha is None -> uniform -> Step-1 behavior, byte-identical.
        _gifts = [gadget_blueprint.opp_cfv(b) for b in root_boards]
        _alpha_on_iter = (os.environ.get("FOW_GADGET_ALPHA") == "1"
                          if gadget_alpha is None else bool(gadget_alpha))
        _alpha_iter = (
            _nonuniform_resolve_alpha(gadget_blueprint, root_boards)
            if _alpha_on_iter else None
        )
        _gadget = ResolveGadget(_gifts, margin=gadget_margin, alpha=_alpha_iter)
        # B' (2026-06-15): severity weighting. The reach-weighted Resolve average
        # lets one catastrophic world (we lose a queen) get diluted by many small
        # passive-margin worlds. Multiply a FOLLOWED world's gadget weight by how
        # badly we're losing it (opp-POV value), so a real catastrophe dominates
        # the worst-case. 0 = off = faithful Resolve. Only fires on worlds we're
        # losing (opp_v > 0) -> safe positions untouched (no over-defense).
        _sev_boost = float(os.environ.get("FOW_GADGET_SEVERITY_BOOST", "0"))
        # How often to re-step the gadget (each step = one extra read-only full-tree
        # eval per root). 1 = every iter (most faithful); raise to amortize cost.
        _gadget_interval = max(1, int(os.environ.get("FOW_GADGET_ITER_INTERVAL", "1")))
        _gadget_weights = _gadget.world_weights()
        # FOW_GADGET_FUSED_EVAL: reuse the weighted pass's own per-root values for
        # the gadget RM+ update instead of a separate root_node_values traversal
        # (~1 of 3 full walks per iter saved; throughput probe 2026-06-09: eq_pass
        # = 77-81% of solve time, gadget config 2.3x slower per iter than OFF).
        # The gadget steps AFTER the pass -> weights lag the values by ONE
        # iteration. ★ MEASURED NEGATIVE for safety configs (2026-06-09): the lag
        # flips the c5a9eb83 knife-edge back to the d2d4 hang at 32K iters under
        # non-uniform alpha. Keep OFF where king-safety matters.
        _gadget_fused = os.environ.get("FOW_GADGET_FUSED_EVAL") == "1"
        # FOW_GADGET_MERGED: the LAG-FREE throughput lever — gadget update on the
        # clean pre-pass root_node_values snapshot (as the base path), then ONE
        # merged walk (both players' regrets, perspective regret carries the
        # gadget weight) instead of the two-pass weighted eq. 2 full walks/iter
        # vs 3. Merged is a different (valid) CFR update scheme — same
        # equilibrium in the limit, different per-iterate values (see
        # eq_traverse_merged) — so rig + strength validate, not byte-parity.
        # Takes precedence over FUSED if both are set. Default OFF.
        _gadget_merged = os.environ.get("FOW_GADGET_MERGED") == "1"
        # FOW_GADGET_WEXP: gadget-weighted expansion-root allocation (see the
        # expansion block in the loop). Default OFF. FOW_GADGET_WEXP_MIX =
        # uniform-floor dial (0.0 = pure weighted, rig-green; see loop comment).
        _gadget_wexp = os.environ.get("FOW_GADGET_WEXP") == "1"
        # FOW_GADGET_MM_SWITCH: the in-solve Maxmargin<->Resolve switch (paper
        # C.1; our Step-3 gap). When every world exits the Resolve gadget
        # (is_safe), the world-weights collapse to ~0 and the weighted pass
        # goes gradient-free (the 2026-06-11 H2H 1-ply-blunder cluster).
        # Switch the pass weight to the adversary's best world instead.
        _gadget_mm_switch = os.environ.get("FOW_GADGET_MM_SWITCH") == "1"
        _wexp_mix = min(1.0, max(0.0, float(os.environ.get("FOW_GADGET_WEXP_MIX", "0"))))
    # Convergence early-stop (FOW_V2_EARLY_STOP): once the root's top action has
    # been stable across several SPACED checks, more search won't change the move
    # (PCFR+ has last-iterate convergence), so stop and bank the time/compute —
    # the big win in the gadget-OFF regime where small beliefs converge well
    # before the 5s budget. Checks are PERIODIC (every _es_interval iters) so we
    # don't reintroduce the per-iter current_strategy PyO3 round-trip Lever 5
    # removed. Conservative defaults; default OFF → byte-identical until enabled,
    # then bakeoff-gated.
    # Defaults are CONSERVATIVE (stable across ~5000 iters): a 4-position local
    # check showed a short window (4 checks) changed the move in 1/4 positions —
    # early-stop is a speed/strength tradeoff, not move-preserving, unless the
    # stability window is long. At these defaults the same 4 were move-identical
    # (40-91% iters saved on fast-converging positions; the genuinely-unsettled
    # one correctly saved ~7%). STILL bakeoff-gate (on-vs-off, confirm strength-
    # neutral) before flipping on in production.
    _kluss_soft = (kluss_soft if kluss_soft is not None
                   else os.environ.get("FOW_KLUSS_SOFT") == "1")
    _es_on = os.environ.get("FOW_V2_EARLY_STOP") == "1" and time_budget_seconds is not None
    _es_interval = int(os.environ.get("FOW_V2_EARLY_STOP_INTERVAL", "500"))
    _es_min_iters = int(os.environ.get("FOW_V2_EARLY_STOP_MIN_ITERS", "3000"))
    _es_min_s = float(os.environ.get("FOW_V2_EARLY_STOP_MIN_S", "0.3"))
    _es_need = int(os.environ.get("FOW_V2_EARLY_STOP_STABLE_CHECKS", "10"))
    _es_prev_top = None
    _es_stable = 0
    _es_stopped = False
    for t in range(iterations):
        elapsed = time.monotonic() - t_start
        if time_budget_seconds is not None and elapsed >= time_budget_seconds:
            break
        if (time_budget_seconds is not None and half_budget_reached_at is None
                and elapsed >= time_budget_seconds / 2.0):
            half_budget_reached_at = t

        _t0 = time.monotonic_ns()
        if _iterative_gadget:
            if _gadget_merged:
                # Lag-free merged: clean pre-pass values for the gadget step,
                # then one merged weighted walk for both players' regrets.
                if t % _gadget_interval == 0:
                    _v_persp = eng.root_node_values(root_ids, persp_white)  # our POV
                    _v_opp = [-v for v in _v_persp]                         # opponent POV
                    _gadget.update(_v_opp)
                    _gadget_weights = _gadget.world_weights()
                    if _sev_boost > 0.0:
                        _gadget_weights = [
                            w * (1.0 + _sev_boost * max(0.0, vo))
                            for w, vo in zip(_gadget_weights, _v_opp, strict=False)
                        ]
                    # C.1 Maxmargin switch: never let the pass go dark.
                    if _gadget_mm_switch and _gadget.is_safe():
                        _gadget_weights = _gadget.maxmargin_weights(_v_opp)
                eng.equilibrium_pass_merged_weighted(
                    root_ids, persp_white, _gadget_weights
                )
            elif _gadget_fused:
                # Fused: the weighted pass returns its own per-root values (our
                # POV); step the gadget AFTER the pass (weights lag one iter).
                _v_persp = eng.equilibrium_pass_weighted_values(
                    root_ids, persp_white, _gadget_weights
                )
                if t % _gadget_interval == 0:
                    _gadget.update([-v for v in _v_persp])  # opponent POV
                    _gadget_weights = _gadget.world_weights()
            else:
                # Proper Resolve gadget: step the follow/exit RM+ on the current
                # per-world opponent values, then run the weighted full-CFV pass.
                if t % _gadget_interval == 0:
                    _v_persp = eng.root_node_values(root_ids, persp_white)  # our POV
                    _v_opp = [-v for v in _v_persp]                         # opponent POV
                    _gadget.update(_v_opp)
                    _gadget_weights = _gadget.world_weights()
                    if _sev_boost > 0.0:
                        _gadget_weights = [
                            w * (1.0 + _sev_boost * max(0.0, vo))
                            for w, vo in zip(_gadget_weights, _v_opp, strict=False)
                        ]
                    if _gadget_mm_switch and _gadget.is_safe():
                        _gadget_weights = _gadget.maxmargin_weights(_v_opp)
                eng.equilibrium_pass_weighted(root_ids, persp_white, _gadget_weights)
        elif _merged_eq:
            # Merged single-walk full-CFV (both players' regrets in one pass).
            eng.equilibrium_pass_merged(root_ids, persp_white)
        elif full_cfv_backprop:
            # PCFR+ Obscuro-faithful: opponent branch sums over all children.
            eng.equilibrium_pass_with(root_ids, persp_white, True)
        else:
            eng.equilibrium_pass(root_ids, persp_white)
        eq_ns += time.monotonic_ns() - _t0

        # Lever 5 (efficiency campaign): the strategy_history snapshot per iter
        # is only needed by purify_strategy's stable-actions filter, which only
        # runs when max_actions > 1 (Maxmargin regime). For Resolve regime
        # (max_actions=1, production default) we skip the per-iter PyO3
        # round-trip + dict comprehension entirely. At 200K+ iters this is the
        # difference between ~30% Python orchestration overhead and a flat loop.
        if record_strategy_history:
            aa = _union_root_action_keys(eng, root_ids)
            if aa:
                strat_now = eng.current_strategy(root_infoset, aa)
                strategy_history.append(
                    {
                        rules.decode_action_key(k): p
                        for k, p in zip(aa, strat_now, strict=False)
                    }
                )

        if expansions_done < expansion_budget:
            # FOW_GADGET_WEXP (faithful expansion allocation): sample the
            # expansion root by the gadget world-weights alpha(J)·P(follow|J)
            # instead of size-balancing. pick_best_root spreads expansions
            # UNIFORMLY across all sampled worlds (min-subtree-size), so depth
            # goes everywhere equally and the decisive lines in the worlds the
            # opponent actually plays into stay shallow — the reason small
            # expansion budgets fail to resolve near-ties. Obscuro expands
            # "according to the profile (x̃t, yt)" (C.4): likely-reached lines
            # get the budget. Falls back to size-balancing when weights are
            # degenerate. Default OFF -> byte-identical.
            #
            # FOW_GADGET_WEXP_MIX (uniform-floor dial, default 0.0 = PURE
            # weighted): sample by (1-mix)·(weights/Σ) + mix·(1/m). Measured
            # 2026-06-10 on the rig (merged+eb2000, 30K iters):
            #   mix=0.0  -> ALL FOUR cells 0.0% king-risk (rig-green)
            #   mix=0.5  -> 3/4 cells pick hangs (diluting depth un-resolves
            #               the near-ties; rig-RED)
            # At scale mix=0.0 OOM'd 4/6 games (|P| 16M by move 16) — but that
            # run was CONFOUNDED by king-aware-leaf OFF (the cloud probes never
            # set it; every rig run did). Re-test pure+king-aware before
            # turning this dial.
            if (_iterative_gadget and _gadget_wexp
                    and sum(_gadget_weights) > 1e-12):
                _wsum = sum(_gadget_weights)
                _m = len(_gadget_weights)
                _x = rng.random()
                _acc = 0.0
                best_i = _m - 1
                for _i, _w in enumerate(_gadget_weights):
                    _acc += (1.0 - _wexp_mix) * (_w / _wsum) + _wexp_mix / _m
                    if _x <= _acc:
                        best_i = _i
                        break
            else:
                best_i = eng.pick_best_root(root_sizes)
            # KLUSS: recompute the keep-set from the current Γ̃ before this iter's
            # leaf selection. Source = roots[0]'s infoset (all roots share an
            # infoset, mirroring gt_cfr.py:1064's Python-tree behavior). Order
            # matches Python (pick_best_root → keep → select_leaf) for byte-
            # identical rng consumption; set_kluss_keep_from is rng-free, so
            # order isn't strictly required, but mirroring helps audits. The
            # rust select_leaf reads engine.kluss_keep internally — see lib.rs.
            if kluss_k is not None:
                _t0 = time.monotonic_ns()
                eng.set_kluss_keep_from(root_ids, kluss_k)
                kluss_ns += time.monotonic_ns() - _t0
            _t0 = time.monotonic_ns()
            leaf = eng.select_leaf(root_ids[best_i], t % 2 == 0)
            # Soft KLUSS (FOW_KLUSS_SOFT, 2026-06-12): the keep-restricted walk
            # DEADLOCKS once the strategy concentrates — every walk follows the
            # favored line to the keep boundary (children filter empty → None)
            # and the expansion budget goes unspent forever. Measured: |P|=1
            # solves expand 4-10 nodes of an eb=2000 budget (sf_ch_size), so
            # the engine plays on ~depth-1 values exactly where fog is thin —
            # where the 2026-06-11 H2H fog deaths cluster (fatal plies at
            # |P|=1-500). At small |P| the connectivity graph has no cross-
            # world edges, so "knowledge distance" degenerates to tree depth
            # ≤ k+1; the paper's |P|=1 case should degenerate to a FULL
            # perfect-info subgame, not a 3-ply one. Soft mode keeps the KLUSS
            # scope as a PREFERENCE: when no in-scope leaf is expandable, drop
            # the filter for this pick and take an unrestricted walk. Extra
            # sel_rng draws only on the fallback path; flag-off is
            # byte-identical to prior behavior.
            if leaf is None and kluss_k is not None and _kluss_soft:
                eng.clear_kluss_keep()
                leaf = eng.select_leaf(root_ids[best_i], t % 2 == 0)
            select_ns += time.monotonic_ns() - _t0
            if leaf is not None:
                _t0 = time.monotonic_ns()
                added = _rust_expand_and_seed(eng, leaf, stockfish_eval, perspective, rules, persp_white)
                expand_ns += time.monotonic_ns() - _t0
                expansions_done += 1
                root_sizes[best_i] += added
        iters_completed += 1

        # Periodic convergence check (cheap: one current_strategy round-trip per
        # _es_interval iters). Stop once the top action key has held across
        # _es_need consecutive checks, past the min-iters/min-time guards.
        if _es_on and t >= _es_min_iters and elapsed >= _es_min_s and t % _es_interval == 0:
            aa_es = _union_root_action_keys(eng, root_ids)
            if aa_es:
                strat_es = eng.current_strategy(root_infoset, aa_es)
                top_es = aa_es[max(range(len(aa_es)), key=lambda i: strat_es[i])]
                if top_es == _es_prev_top:
                    _es_stable += 1
                else:
                    _es_stable = 0
                    _es_prev_top = top_es
                if _es_stable >= _es_need:
                    _es_stopped = True
                    break

    sf_eval_ns = stockfish_eval.eval_wall_ns - sf_eval_ns_start
    sf_children_ns = stockfish_eval.children_wall_ns - sf_children_ns_start
    component_ms = {
        "sf_eval": sf_eval_ns / 1e6,
        "sf_children": sf_children_ns / 1e6,
        "eq_pass": eq_ns / 1e6,
        "select_leaf": select_ns / 1e6,
        "kluss": kluss_ns / 1e6,
        "expand_seed": expand_ns / 1e6,
    }
    total_tree_nodes = int(eng.tree_kluss_sizes()[0])

    t_half = half_budget_reached_at if half_budget_reached_at is not None else iters_completed // 2

    aa = _union_root_action_keys(eng, root_ids)
    if not aa:
        return MultiRootGTCFRSolution(
            strategy_at_root={}, value_at_root=0.0, iterations=iters_completed,
            info_set_count=0, total_tree_nodes=total_tree_nodes, n_roots=len(root_ids),
            elapsed_seconds=time.monotonic() - t_start,
            strategy_history_at_root=strategy_history, t_half=t_half,
            component_ms=component_ms,
            root_ids=list(root_ids),
        )
    gadget_applied = None
    # FOW_GADGET_COMMIT_CVAR: the commit-layer catastrophe filter. The original
    # assumption — "when the PROPER (iterative) gadget is on, the safe strategy
    # is already baked into the weighted regrets, so read it out normally" — is
    # EMPIRICALLY FALSE (2026-06-11 H2H forensics): the in-solve weighting
    # bounds margins vs the BLUEPRINT baseline, and in bad endgames the
    # baseline is already bad, so a move that dies in 8-54%% of the engine's
    # OWN belief worlds doesn't violate the constraint — 13/15 losses ended
    # exactly that way (killers all in fog, believed risk measured up to 54%%).
    # The read-only CVaR gadget was the measured fix for this class (2026-06-05:
    # demoted the king-risky move #1 -> #22); restoring it ON TOP of the
    # iterative solve re-prices the worst-tail at the only place that matters —
    # the move that leaves the engine. Default OFF -> prior behavior.
    _commit_cvar = os.environ.get("FOW_GADGET_COMMIT_CVAR") == "1"
    if (resolve_gadget and gadget_blueprint is not None
            and (not _iterative_gadget or _commit_cvar)):
        # Read-only Resolve gadget: override the root readout with the
        # blueprint-capped, per-world-aggregated strategy. OFF path (default)
        # is byte-identical — this branch never runs.
        gadget_applied = _apply_resolve_gadget(
            eng, root_ids, aa, persp_white, gadget_blueprint, gadget_margin,
            worlds=root_boards, cvar_q=gadget_cvar_q, rules=rules,
            gadget_faithful=gadget_faithful, gadget_alpha=gadget_alpha,
        )
    if gadget_applied is not None:
        strat, action_values, value = gadget_applied
    else:
        strat_list = eng.current_strategy(root_infoset, aa)
        strat = {
            rules.decode_action_key(k): p
            for k, p in zip(aa, strat_list, strict=False)
        }
        action_values: dict = {}
        value = 0.0
        for k, p in zip(aa, strat_list, strict=False):
            n = eng.visit_count_get(root_infoset, k)
            if n > 0:
                mean = eng.value_sum_get(root_infoset, k) / n
                action_values[rules.decode_action_key(k)] = mean
                value += p * mean
    return MultiRootGTCFRSolution(
        strategy_at_root=strat, value_at_root=value, iterations=iters_completed,
        info_set_count=0, total_tree_nodes=total_tree_nodes, n_roots=len(root_ids),
        elapsed_seconds=time.monotonic() - t_start,
        strategy_history_at_root=strategy_history,
        action_values_at_root=action_values, t_half=t_half,
        component_ms=component_ms,
        root_ids=list(root_ids),
    )
