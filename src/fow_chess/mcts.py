"""Monte Carlo Tree Search for fog-of-war chess.

Architecture
------------
One sampled world per rollout (determinization).

At the root, the engine holds a full BeliefState (particle cloud). Each MCTS
rollout samples one particle — P_true — from the belief distribution weighted
by particle weights. For the duration of that rollout, P_true is treated as the
true board. This is open-loop MCTS: no belief filter updates during rollout.

Selection phase (depth 0 → selection_depth):
    Walk the existing tree using UCB1. At each node, the move from parent was
    applied to P_true. The node stores accumulated statistics (visits,
    total_value, catastrophe_count) across all rollouts that passed through it.

Expansion:
    At the first unvisited child, create a new MCTSNode.

Rollout (depth selection_depth → rollout_depth):
    Continue forward on P_true using a fast rollout policy:
      - Own move: pick from top-K candidates by a cheap heuristic score.
      - Opponent move: sample from the stockfish_shallow prior over the
        opponent's visible board (their moves weighted by model of rationality).
    Evaluate the terminal board with fow_evaluate().

Synthesis (root node, choosing the final move):
    After N rollouts, each candidate move has a distribution of rollout values.
    Selection uses risk-adjusted scoring:
        score(M) = (1 - risk_lambda) × mean + risk_lambda × CVaR_alpha
    where CVaR_alpha is the expected value of the worst-alpha fraction of
    rollouts. This penalises moves with rare catastrophic outcomes without
    fully ignoring their upside.

Hardcoded vetoes applied post-synthesis (certainty facts, not probabilities):
    - Own king capture in next move: always take it.
    - Capture defended by visible opponent king (losing material): always veto.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import chess

from .belief import BeliefState
from .engine import Evaluator
from .evaluator import _KING_CAPTURE_SCORE, _PIECE_VALUES, material_score

if TYPE_CHECKING:
    from .selfplay import PerspectiveView

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UCB_C = 1.41          # exploration constant (√2)
_CATASTROPHE_THRESHOLD = -200.0   # rollout value below this = catastrophe
_DRAW_VALUE = 0.0                  # value assigned to a draw / max-depth truncation


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """One node in the MCTS tree.

    Statistics are accumulated across all rollouts that pass through this node.
    Values are in centipawns, from the root perspective.
    """

    move: chess.Move | None  # move from parent to this node (None at root)

    visits: int = 0
    total_value: float = 0.0

    # Risk-adjusted synthesis: track the tail separately.
    # catastrophe_count / visits = P(catastrophe) for this node.
    catastrophe_count: int = 0
    catastrophe_total: float = 0.0

    children: dict[int, "MCTSNode"] = field(default_factory=dict)  # keyed by move.uci hash

    def q(self) -> float:
        """Mean rollout value."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb(self, parent_visits: int, c: float = _UCB_C) -> float:
        """UCB1 selection score (higher = prefer this child)."""
        if self.visits == 0:
            return float("inf")
        return self.q() + c * math.sqrt(math.log(parent_visits) / self.visits)

    def risk_adjusted_score(self, risk_lambda: float, cvar_alpha: float = 0.10) -> float:
        """Risk-adjusted selection score used at the ROOT for final move choice.

        score = (1 - risk_lambda) × mean + risk_lambda × CVaR_estimate

        CVaR is estimated as: expected value of the worst cvar_alpha fraction.
        We approximate CVaR from (catastrophe_count, catastrophe_total): the
        catastrophe_total / catastrophe_count gives E[value | catastrophe], and
        we blend it proportionally with the mean.

        When no catastrophes have been observed, CVaR ≈ mean (no tail risk).
        """
        if self.visits == 0:
            return float("-inf")
        mean = self.q()
        if self.catastrophe_count == 0 or risk_lambda == 0.0:
            return mean
        # Approximate CVaR from catastrophe statistics.
        p_cat = self.catastrophe_count / self.visits
        e_cat = self.catastrophe_total / self.catastrophe_count
        cvar_est = (1.0 - p_cat) * mean + p_cat * e_cat
        return (1.0 - risk_lambda) * mean + risk_lambda * cvar_est

    def update(self, value: float) -> None:
        self.visits += 1
        self.total_value += value
        if value < _CATASTROPHE_THRESHOLD:
            self.catastrophe_count += 1
            self.catastrophe_total += value


# ---------------------------------------------------------------------------
# Rollout policy
# ---------------------------------------------------------------------------

def _rollout_policy_own(
    board: chess.Board,
    perspective: chess.Color,
    rng: random.Random,
    top_k: int = 8,
) -> chess.Move | None:
    """Pick own move during rollout.

    Priority:
      1. King capture (always).
      2. King defense: if own king is under direct attack, restrict to
         moves that resolve the attack (capture attacker, block, flee).
         This prevents the engine from ignoring an opponent bishop
         converging on the king while making irrelevant moves elsewhere.
      3. Safe captures: prefer taking material that isn't defended cheaply.
      4. Quiet moves away from attacked squares.
    """
    moves = list(board.pseudo_legal_moves)
    if not moves:
        return None

    opp = not perspective

    # Pre-compute opponent attacks.
    board.turn = opp
    opp_attack_squares: set[chess.Square] = {m.to_square for m in board.pseudo_legal_moves}
    board.turn = perspective

    # Check if own king is directly attacked.
    own_king_sq = board.king(perspective)
    king_under_attack = (
        own_king_sq is not None and own_king_sq in opp_attack_squares
    )

    opp_king_sq = board.king(opp)
    king_zone = (
        chess.BB_KING_ATTACKS[opp_king_sq] | chess.BB_SQUARES[opp_king_sq]
        if opp_king_sq is not None
        else 0
    )

    def _score(m: chess.Move) -> float:
        target = board.piece_at(m.to_square)
        if target is not None and target.piece_type == chess.KING:
            return 1e9

        mover = board.piece_at(m.from_square)
        mover_val = _PIECE_VALUES.get(mover.piece_type, 0) if mover else 0
        target_val = _PIECE_VALUES.get(target.piece_type, 0) if target else 0

        # King defense priority: heavily reward moves that resolve own king attack.
        defense_bonus = 0.0
        if king_under_attack and own_king_sq is not None:
            # Simulate: does this move get the king out of check?
            sim = board.copy()
            sim.push(m)
            sim.turn = opp
            new_king_sq = sim.king(perspective)
            if new_king_sq is not None:
                still_attacked = any(
                    rm.to_square == new_king_sq for rm in sim.pseudo_legal_moves
                )
                if not still_attacked:
                    defense_bonus = 800.0  # large bonus for resolving king attack

        # Penalty for moving into opponent-attacked square.
        in_danger = m.to_square in opp_attack_squares
        if in_danger and target_val < mover_val:
            danger_penalty = -(mover_val - target_val) * 1.5
        elif in_danger and target_val == 0:
            danger_penalty = -mover_val * 0.8
        else:
            danger_penalty = 0.0

        capture_gain = target_val
        king_hit = 25 if chess.BB_SQUARES[m.to_square] & king_zone else 0
        return defense_bonus + capture_gain + king_hit + danger_penalty

    scored = sorted(moves, key=_score, reverse=True)
    candidates = scored[:top_k]
    return rng.choice(candidates)


def _rollout_policy_opp(
    board: chess.Board,
    opp: chess.Color,
    rng: random.Random,
    top_k: int = 6,
) -> chess.Move | None:
    """Sample opponent move during rollout.

    The opponent prioritises capturing OUR hanging pieces — pieces that moved
    to squares the opponent attacks. This is the critical fix: if our own
    rollout policy leaves material hanging, the opponent punishes it here,
    making the MCTS correctly penalise reckless moves.
    """
    moves = list(board.pseudo_legal_moves)
    if not moves:
        return None

    perspective = not opp
    my_king_sq = board.king(perspective)
    king_zone = (
        chess.BB_KING_ATTACKS[my_king_sq] | chess.BB_SQUARES[my_king_sq]
        if my_king_sq is not None
        else 0
    )

    def _score(m: chess.Move) -> float:
        target = board.piece_at(m.to_square)
        if target is not None and target.piece_type == chess.KING:
            return 1e9
        # Weight captures heavily — opponent punishes hanging pieces.
        capture = _PIECE_VALUES.get(target.piece_type, 0) if target else 0
        king_hit = 40 if chess.BB_SQUARES[m.to_square] & king_zone else 0
        return capture * 1.5 + king_hit + rng.random() * 8

    scored = sorted(moves, key=_score, reverse=True)
    candidates = scored[:top_k]
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def _run_rollout(
    root: MCTSNode,
    p_true: chess.Board,
    perspective: chess.Color,
    evaluator: Evaluator,
    rng: random.Random,
    selection_depth: int,
    rollout_depth: int,
) -> float:
    """Run one MCTS rollout: selection → expansion → simulation → backprop.

    Returns the rollout value from `perspective`'s point of view.
    """
    opp = not perspective
    board = p_true.copy()
    path: list[MCTSNode] = [root]
    depth = 0

    # --- Selection ---
    node = root
    while depth < selection_depth and node.children:
        parent_visits = node.visits or 1
        best_child = max(
            node.children.values(),
            key=lambda c: c.ucb(parent_visits),
        )
        move = best_child.move
        assert move is not None

        # Apply move to p_true
        if not board.is_pseudo_legal(move):
            # Particle diverged from tree path — abort cleanly
            break

        board.push(move)
        node = best_child
        path.append(node)
        depth += 1

        # Alternate between own and opp moves in the tree
        # (tree alternates: my move at even depth from root, opp at odd)

    # --- Expansion ---
    if not _is_terminal(board):
        board.turn = perspective if depth % 2 == 0 else opp
        legal = list(board.pseudo_legal_moves)
        if legal:
            # Find an unvisited move
            visited_keys = set(node.children.keys())
            unvisited = [m for m in legal if hash(m) not in visited_keys]
            if unvisited:
                expand_move = rng.choice(unvisited)
                child = MCTSNode(move=expand_move)
                node.children[hash(expand_move)] = child
                board.push(expand_move)
                node = child
                path.append(node)
                depth += 1

    # --- Simulation (open-loop rollout on p_true board) ---
    sim = board.copy()
    sim_depth = 0
    remaining = rollout_depth - depth

    while sim_depth < remaining:
        if _is_terminal(sim):
            break
        is_own_turn = (depth + sim_depth) % 2 == 0
        if is_own_turn:
            sim.turn = perspective
            move = _rollout_policy_own(sim, perspective, rng)
        else:
            sim.turn = opp
            move = _rollout_policy_opp(sim, opp, rng)

        if move is None:
            break
        if not sim.is_pseudo_legal(move):
            break
        sim.push(move)
        sim_depth += 1

        # Check terminal after each move
        if _is_terminal(sim):
            break

    value = _terminal_or_eval(sim, perspective, evaluator)

    # --- Backpropagation ---
    for n in path:
        n.update(value)

    return value


def _is_terminal(board: chess.Board) -> bool:
    return board.king(chess.WHITE) is None or board.king(chess.BLACK) is None


def _terminal_or_eval(
    board: chess.Board, perspective: chess.Color, evaluator: Evaluator
) -> float:
    """Score the terminal/leaf board from perspective's POV."""
    if board.king(perspective) is None:
        return -_KING_CAPTURE_SCORE
    opp = not perspective
    if board.king(opp) is None:
        return _KING_CAPTURE_SCORE
    # Non-terminal leaf: evaluate via chess.Move.null() so evaluators see a
    # genuine null move (board.push handles Move(0,0) specially — just flips
    # the turn, no piece movement). The earlier Move(own_king, own_king) trick
    # tripped every evaluator's king-capture short-circuit (target piece on
    # to_square == own king → return -KING_CAPTURE_SCORE), making every MCTS
    # leaf return -100000 regardless of evaluator. Caught 2026-05-17.
    try:
        return evaluator(board, chess.Move.null(), perspective)
    except Exception:
        return material_score(board, perspective)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mcts_pick_move(
    belief: BeliefState,
    view: "PerspectiveView",
    evaluator: Evaluator,
    rng: random.Random,
    *,
    n_rollouts: int = 200,
    selection_depth: int = 3,
    rollout_depth: int = 8,
    risk_lambda: float = 0.25,
    deadline_monotonic: float | None = None,
    out_chosen_q: list[float] | None = None,
    out_root_visits: dict[str, int] | None = None,
) -> chess.Move:
    """Pick a move via MCTS from the current belief state.

    Args:
        belief:            Current particle belief state.
        view:              What the perspective player can see.
        evaluator:         Leaf evaluator (fow_evaluator recommended).
        rng:               Random source.
        n_rollouts:        Target number of rollouts (may stop early if deadline hit).
        selection_depth:   How deep to walk the existing tree before rollout.
        rollout_depth:     Total depth per rollout (selection + simulation).
        risk_lambda:       CVaR blend weight for final move selection.
        deadline_monotonic: Optional wall-clock deadline (time.monotonic()).
    """
    legal = view.own_legal_moves
    if not legal:
        raise ValueError("no legal moves")
    if len(legal) == 1:
        return legal[0]

    perspective = view.perspective
    root = MCTSNode(move=None)

    # Pre-populate root children so every legal move gets at least 1 visit
    for move in legal:
        root.children[hash(move)] = MCTSNode(move=move)

    # Run rollouts
    particles = belief.particles
    weights = belief.weights
    total_weight = sum(weights)

    rollouts_done = 0
    for _ in range(n_rollouts):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            break

        # Sample one particle proportional to weights
        r = rng.random() * total_weight
        cumulative = 0.0
        p_true = particles[-1].copy()
        for particle, weight in zip(particles, weights, strict=False):
            cumulative += weight
            if r <= cumulative:
                p_true = particle.copy()
                break

        p_true.turn = perspective
        _run_rollout(
            root, p_true, perspective, evaluator, rng,
            selection_depth=selection_depth,
            rollout_depth=rollout_depth,
        )
        rollouts_done += 1

    # Select best move by risk-adjusted score
    best_move = legal[0]
    best_score = float("-inf")
    best_child: MCTSNode | None = None
    for move in legal:
        child = root.children.get(hash(move))
        if child is None or child.visits == 0:
            continue
        score = child.risk_adjusted_score(risk_lambda)
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child

    if out_chosen_q is not None:
        # Mean rollout value of the chosen move's subtree, from perspective's
        # POV (centipawns). Distillation labeler reads this to train an eval
        # that matches MCTS-amplified judgment, not just raw game outcomes.
        out_chosen_q.append(best_child.q() if best_child is not None else 0.0)

    if out_root_visits is not None:
        # MCTS root visit distribution — the policy target for AlphaZero-style
        # training. Each legal move's visit count reflects MCTS's relative
        # preference. Saved per ply so the corpus can train a policy net
        # later without re-running self-play.
        for move in legal:
            child = root.children.get(hash(move))
            if child is not None:
                out_root_visits[move.uci()] = child.visits

    return best_move
