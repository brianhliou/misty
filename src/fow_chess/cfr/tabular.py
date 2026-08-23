"""Tabular CFR with external sampling on opponent nodes.

This module is the algorithmic core. It does not depend on Fog of War chess
directly — any "node" supporting the following duck-typed interface works:

    node.is_terminal -> bool
    node.terminal_value(perspective) -> float
    node.legal_moves() -> list[Action]
    node.info_set_id() -> Hashable
    node.apply(action) -> Node
    node.to_move -> Player
    node.depth -> int

This generality is intentional: the Kuhn poker correctness gate
(``tests/test_cfr_kuhn.py``) supplies a non-chess node type. The FoW
``SubgameNode`` satisfies the interface for production solves.

Two solver variants share this module:

* **Vanilla CFR** (``solve_subgame``): standard regret matching, returns
  the AVERAGE strategy at the root (Nash-convergent in the limit).
* **PCFR+** (``solve_subgame`` with ``predictive=True``): Predictive
  Regret Matching+ (Farina, Kroer, Sandholm AAAI-21). Adds a "predicted
  next regret" term to the strategy computation and returns the LAST
  iterate (which has last-iterate Nash convergence under PCFR+). This
  is the inner solver Obscuro uses inside KLUSS.

Per-iteration algorithm (both variants):

  For each iteration t in [0, iterations):
    For each player T in [WHITE, BLACK]:
      Walk the tree with T as the *traversing player*.
        At T-nodes: enumerate all actions; compute counterfactual value per
          action; accumulate regret = action_value - node_value.
        At opp nodes: sample one action from opp's current regret-matching
          strategy; recurse.
        At terminals or depth bound: return leaf value from T's POV.
      Update average-strategy accumulators with T's current strategy.
  Return:
    - Vanilla: average strategy at root.
    - PCFR+: last-iterate strategy at root.

PCFR+ deviation from vanilla: at each infoset visit, strategy =
[z + previous_regret]^+ / ||·||_1 where z is the thresholded cumulative
regret and previous_regret is the regret vector observed at the most
recent visit to this infoset. Predicts next loss equals previous loss
(simplest and Obscuro's choice).

Pragmatic relaxations (see ``engine-deep-cfr-feasibility.md``):

- Action set at each node = ``node.legal_moves()``. In strict FoW,
  different truths in the same info set could have different pseudo-legal-
  move sets. Regret tables keyed by ``(info_set_id, action)`` accumulate
  only for actions actually visited; this is an approximation that is
  acceptable for Phase 1 validation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Hashable


# Type aliases for clarity. The algorithm is generic — we don't import
# chess types into this module.
Player = object
Action = Hashable


@dataclass
class SubgameSolution:
    """CFR solve output."""

    strategy_at_root: dict[Action, float]
    """Average (Nash-convergent) strategy at the root info set."""

    value_at_root: float
    """Monte-Carlo estimate of root value under the average strategy,
    from the root-mover's POV."""

    iterations: int
    """Iterations actually run."""

    info_set_count: int
    """Distinct info sets visited during the solve."""


@dataclass
class _CFRState:
    depth_bound: int
    leaf_eval: Callable[[object, Player], float]
    rng: random.Random
    # info_set_id -> {action -> cumulative regret}
    regrets: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    # info_set_id -> {action -> cumulative strategy probability}
    strategy_sum: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    # info_set_id -> {action -> regret observed at last visit} (PCFR+ only)
    last_regret: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    # info_set_id -> {action -> probability} of the last strategy played
    # at this infoset (PCFR+ last-iterate output)
    last_strategy: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    # set during traversal
    traversing_player: Player | None = None
    # If True: use PCFR+ predictive update + return last iterate at root.
    predictive: bool = False


def _current_strategy(
    info_set_id: Hashable,
    actions: list[Action],
    state: _CFRState,
) -> list[float]:
    """Regret-matching: action probabilities ∝ positive regret.

    For PCFR+ (``state.predictive=True``), uses the predictive variant:
    ``θ = [z + previous_regret]^+`` where ``z`` is cumulative thresholded
    regret and ``previous_regret`` is the regret vector observed at the
    most recent visit to this infoset. Falls back to plain ``[z]^+`` on
    the first visit (no prediction available).

    Both variants fall back to uniform when no action has positive
    (predicted) cumulative regret.
    """
    regrets = state.regrets[info_set_id]
    if state.predictive:
        prev = state.last_regret.get(info_set_id, {})
        positive = [
            max(0.0, regrets.get(a, 0.0) + prev.get(a, 0.0)) for a in actions
        ]
    else:
        positive = [max(0.0, regrets.get(a, 0.0)) for a in actions]
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


def _cfr_traverse(node, state: _CFRState) -> float:
    """Return the value of ``node`` from ``state.traversing_player``'s POV.

    External sampling: at traverser's nodes, enumerate all actions; at opp
    nodes, sample one action from opp's current strategy. Chance nodes
    (where ``getattr(node, 'is_chance', False)`` is True) iterate over all
    outcomes weighted by their probabilities — standard CFR-with-chance.
    """
    if getattr(node, "is_chance", False):
        value = 0.0
        for action, prob in node.chance_outcomes():
            value += prob * _cfr_traverse(node.apply(action), state)
        return value
    if node.is_terminal:
        return node.terminal_value(state.traversing_player)
    if node.depth >= state.depth_bound:
        return state.leaf_eval(node.truth, state.traversing_player)

    actions = node.legal_moves()
    if not actions:
        # Stalemate-like: no legal moves but not king-captured. Treat as
        # neutral. (In FoW this is a draw under standard rules.)
        return 0.0

    info_set_id = node.info_set_id()
    strategy = _current_strategy(info_set_id, actions, state)

    if node.to_move == state.traversing_player:
        # Enumerate all actions
        action_values = [0.0] * len(actions)
        for i, action in enumerate(actions):
            child = node.apply(action)
            action_values[i] = _cfr_traverse(child, state)
        node_value = sum(s * v for s, v in zip(strategy, action_values, strict=False))
        # Accumulate regrets + strategy
        if state.predictive:
            # RM+ thresholded update: z := [z + r]^+. Also stash the
            # raw regret vector for the NEXT visit's prediction term.
            for i, action in enumerate(actions):
                r = action_values[i] - node_value
                cur = state.regrets[info_set_id].get(action, 0.0)
                state.regrets[info_set_id][action] = max(0.0, cur + r)
                state.last_regret[info_set_id][action] = r
                state.last_strategy[info_set_id][action] = strategy[i]
            # PCFR+ uses last-iterate, but we still accumulate
            # strategy_sum for diagnostics + parity with vanilla.
            for i, action in enumerate(actions):
                state.strategy_sum[info_set_id][action] += strategy[i]
        else:
            for i, action in enumerate(actions):
                state.regrets[info_set_id][action] += action_values[i] - node_value
                state.strategy_sum[info_set_id][action] += strategy[i]
        return node_value

    # Opp node: sample one action according to opp's current strategy
    chosen_idx = _sample(strategy, state.rng)
    return _cfr_traverse(node.apply(actions[chosen_idx]), state)


def _average_strategy(
    info_set_id: Hashable,
    actions: list[Action],
    state: _CFRState,
) -> list[float]:
    """Normalized accumulated strategy = Nash-convergent strategy."""
    strat_sum = state.strategy_sum[info_set_id]
    weights = [strat_sum.get(a, 0.0) for a in actions]
    total = sum(weights)
    if total > 0.0:
        return [w / total for w in weights]
    n = len(actions)
    return [1.0 / n] * n


def _rollout_with_avg_strategy(
    node,
    state: _CFRState,
    perspective: Player,
) -> float:
    """One rollout sampling from accumulated strategies.

    Vanilla CFR samples from the *average* strategy (Nash convergent in
    the limit). PCFR+ samples from the *last iterate* (which itself has
    last-iterate convergence under PCFR+).

    Chance nodes sample one outcome weighted by its probability.
    """
    if getattr(node, "is_chance", False):
        outcomes = node.chance_outcomes()
        probs = [p for _, p in outcomes]
        chosen_idx = _sample(probs, state.rng)
        action = outcomes[chosen_idx][0]
        return _rollout_with_avg_strategy(node.apply(action), state, perspective)
    if node.is_terminal:
        return node.terminal_value(perspective)
    if node.depth >= state.depth_bound:
        return state.leaf_eval(node.truth, perspective)
    actions = node.legal_moves()
    if not actions:
        return 0.0
    info_set_id = node.info_set_id()
    if state.predictive:
        last = state.last_strategy.get(info_set_id, {})
        if last:
            raw = [last.get(a, 0.0) for a in actions]
            total = sum(raw)
            probs = [r / total for r in raw] if total > 0 else _current_strategy(
                info_set_id, actions, state
            )
        else:
            probs = _current_strategy(info_set_id, actions, state)
    else:
        probs = _average_strategy(info_set_id, actions, state)
    chosen_idx = _sample(probs, state.rng)
    return _rollout_with_avg_strategy(node.apply(actions[chosen_idx]), state, perspective)


def solve_subgame(
    root,
    depth: int,
    leaf_eval: Callable[[object, Player], float],
    *,
    iterations: int = 500,
    value_estimate_samples: int = 200,
    players: tuple[Player, Player] | None = None,
    rng: random.Random | None = None,
    predictive: bool = False,
) -> SubgameSolution:
    """Solve a subgame rooted at ``root`` with tabular CFR + external sampling.

    Parameters
    ----------
    root : Node
        Subgame root (any node supporting the duck-typed interface above).
    depth : int
        Maximum plies from root before falling back to ``leaf_eval``.
    leaf_eval : (truth, perspective) -> float
        Position evaluator at the depth bound. Values should be on a scale
        compatible with the node's ``terminal_value``.
    iterations : int
        CFR iterations. 500 is a reasonable Phase-1 default; bump up if
        regrets haven't settled.
    value_estimate_samples : int
        Monte-Carlo rollouts used to estimate ``value_at_root`` under the
        accumulated average strategy (vanilla) or last iterate (PCFR+).
    players : (player_a, player_b)
        The two players in the game. Defaults to ``(root.to_move,
        not root.to_move)``; override for game types where ``not`` isn't
        the right inversion.
    rng : random.Random
        RNG for sampling. Defaults to a fresh ``random.Random(0)``.
    predictive : bool
        If True, use PCFR+ (predictive regret-matching+, last-iterate
        output). If False (default), use vanilla CFR (average iterate).
    """
    rng = rng or random.Random(0)
    root_is_chance = getattr(root, "is_chance", False)
    if players is None:
        if root_is_chance:
            raise ValueError(
                "players must be supplied explicitly when root is a chance node"
            )
        players = (root.to_move, not root.to_move)

    state = _CFRState(
        depth_bound=depth,
        leaf_eval=leaf_eval,
        rng=rng,
        predictive=predictive,
    )

    for _ in range(iterations):
        for traversing_player in players:
            state.traversing_player = traversing_player
            _cfr_traverse(root, state)

    # Extract strategy at root (only meaningful for decision-node roots).
    # PCFR+ uses last-iterate (Farina et al. 2021); vanilla uses average.
    if root_is_chance:
        strategy_at_root: dict = {}
    else:
        root_info_set = root.info_set_id()
        root_actions = root.legal_moves()
        if predictive:
            # Last iterate: use the most-recent strategy at the root
            # infoset. Fall back to current regret-matching read if
            # the root was somehow never visited.
            last = state.last_strategy.get(root_info_set, {})
            if last:
                root_strat = [last.get(a, 0.0) for a in root_actions]
                total = sum(root_strat)
                if total > 0:
                    root_strat = [s / total for s in root_strat]
                else:
                    root_strat = _current_strategy(root_info_set, root_actions, state)
            else:
                root_strat = _current_strategy(root_info_set, root_actions, state)
            strategy_at_root = dict(zip(root_actions, root_strat, strict=False))
        else:
            avg_strat = _average_strategy(root_info_set, root_actions, state)
            strategy_at_root = dict(zip(root_actions, avg_strat, strict=False))

    # Estimate root value via Monte-Carlo rollouts under avg strategies.
    # When root is a chance node, value is estimated from players[0]'s POV
    # by convention.
    root_perspective = players[0] if root_is_chance else root.to_move
    value_sum = 0.0
    for _ in range(value_estimate_samples):
        value_sum += _rollout_with_avg_strategy(root, state, root_perspective)
    value_at_root = value_sum / max(1, value_estimate_samples)

    return SubgameSolution(
        strategy_at_root=strategy_at_root,
        value_at_root=value_at_root,
        iterations=iterations,
        info_set_count=len(state.strategy_sum),
    )
