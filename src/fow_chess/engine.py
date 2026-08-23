"""Tier-1 fog of war engine: belief tracker + per-particle evaluator + weighted vote."""

from __future__ import annotations

import random
import time
import math
from typing import TYPE_CHECKING, Callable, Protocol

import chess

from .belief import BeliefState

if TYPE_CHECKING:
    from .selfplay import PerspectiveView


class Evaluator(Protocol):
    """Scores a candidate move on a concrete (perfect-info) board for `perspective`."""

    def __call__(
        self,
        board: chess.Board,
        move: chess.Move,
        perspective: chess.Color,
    ) -> float: ...


# An EvaluatorBuilder produces a per-move Evaluator, optionally informed by the
# current PerspectiveView. Strategies that need visibility-grounded heuristics
# (threats counted from observed truth rather than particle hypotheses) build a
# fresh evaluator per move; view-independent evaluators wrap with `static_builder`.
EvaluatorBuilder = Callable[["PerspectiveView"], Evaluator]


def static_builder(evaluator: Evaluator) -> EvaluatorBuilder:
    """Wrap a view-independent Evaluator as an EvaluatorBuilder."""

    def build(view: "PerspectiveView") -> Evaluator:
        return evaluator

    return build


def best_action(
    belief: BeliefState,
    evaluator: Evaluator,
    own_legal_moves: list[chess.Move],
    *,
    max_particles: int | None = 16,
    risk_aversion: float = 0.0,
    rng: random.Random | None = None,
    deadline_monotonic: float | None = None,
    out_scored_moves: list[tuple[chess.Move, float, float]] | None = None,
) -> chess.Move:
    """Pick a move by weighted vote across particles.

    Evaluation order is **particle-major** so useful partial work is available
    quickly: particle[0] is spread across candidate moves before particle[1].
    The deadline is checked after each evaluated move cell so expensive leaf
    evaluators cannot force a full first particle round before interruption.

    `final = (1 - risk_aversion) * mean + risk_aversion * worst` where
    `mean` is the particle-weight-weighted average and `worst` is the
    minimum across legal particles. `risk_aversion` ∈ [0, 1] interpolates:
    0 is pure mean; 1 is CVaR-style worst-case.

    Ties are broken uniformly-at-random within an epsilon band of the best
    score, weighted by the move's particle support.

    `max_particles` caps how many particles are scored per move (sampled by
    weight when belief has more). `deadline_monotonic` is a `time.monotonic()`
    target after which the algorithm returns the best move found so far —
    None = no deadline (regime-1 / untimed).
    """
    if not own_legal_moves:
        raise ValueError("best_action called with empty own_legal_moves")
    if not belief.particles:
        return own_legal_moves[0]
    if not 0.0 <= risk_aversion <= 1.0:
        raise ValueError(f"risk_aversion must be in [0, 1], got {risk_aversion}")

    rng = rng or random.Random(0)
    sampled_particles, sampled_weights = _sample_particles(belief, max_particles, rng)

    n_moves = len(own_legal_moves)
    weighted_sum = [0.0] * n_moves
    total_weight = [0.0] * n_moves
    worst = [float("inf")] * n_moves

    deadline_hit = False
    for _round_idx, (particle, weight) in enumerate(
        zip(sampled_particles, sampled_weights, strict=False)
    ):
        for move_idx, move in enumerate(own_legal_moves):
            if not particle.is_pseudo_legal(move):
                continue
            score = evaluator(particle, move, belief.perspective)
            weighted_sum[move_idx] += weight * score
            total_weight[move_idx] += weight
            if score < worst[move_idx]:
                worst[move_idx] = score
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                deadline_hit = True
                break
        if deadline_hit:
            break
        # Secondary check between particle rounds for cheap evaluators that may
        # finish the inner loop just as the deadline expires.
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            break

    move_scores: list[tuple[chess.Move, float, float]] = []
    best_score = float("-inf")
    for move_idx, move in enumerate(own_legal_moves):
        if total_weight[move_idx] <= 0.0:
            continue
        mean = weighted_sum[move_idx] / total_weight[move_idx]
        if risk_aversion == 0.0:
            final = mean
        else:
            final = (1.0 - risk_aversion) * mean + risk_aversion * worst[move_idx]
        move_scores.append((move, final, total_weight[move_idx]))
        if final > best_score:
            best_score = final

    if out_scored_moves is not None:
        out_scored_moves.extend(move_scores)

    if not move_scores:
        return own_legal_moves[0]

    epsilon = 1e-6
    candidates = [
        (move, support)
        for move, score, support in move_scores
        if score >= best_score - epsilon
    ]
    moves = [m for m, _ in candidates]
    weights = [s for _, s in candidates]
    return rng.choices(moves, weights=weights, k=1)[0]


def _sample_particles(
    belief: BeliefState,
    max_particles: int | None,
    rng: random.Random,
) -> tuple[list[chess.Board], list[float]]:
    if max_particles is None or len(belief.particles) <= max_particles:
        return list(belief.particles), list(belief.weights)

    total = sum(belief.weights)
    if total <= 0:
        return [], []
    if len(belief.particles) <= max_particles:
        return list(belief.particles), list(belief.weights)

    probs = [w / total for w in belief.weights]
    keyed = [
        (-math.log(max(rng.random(), 1e-12)) / prob, idx)
        for idx, prob in enumerate(probs)
        if prob > 0.0
    ]
    keyed.sort()
    indices = [idx for _, idx in keyed[:max_particles]]
    particles = [belief.particles[i] for i in indices]
    selected_total = sum(belief.weights[i] for i in indices)
    if selected_total <= 0:
        return [], []
    weights = [belief.weights[i] / selected_total for i in indices]
    return particles, weights


def _opp_best_score(
    particle: chess.Board,
    opp_color: chess.Color,
    evaluator: Evaluator,
) -> float:
    """Best score opp can achieve from this position, scored from opp's POV.

    Within a particle, the world is fully observed — opp picks the move that
    maximizes its own eval. Used as the depth-1 response inside PIMC.
    """
    best = -float("inf")
    for opp_move in particle.pseudo_legal_moves:
        s = evaluator(particle, opp_move, opp_color)
        if s > best:
            best = s
    return 0.0 if best == -float("inf") else best


def pimc_best_action(
    belief: BeliefState,
    evaluator: Evaluator,
    own_legal_moves: list[chess.Move],
    *,
    max_particles: int | None = 8,
    search_depth: int = 2,
    rng: random.Random | None = None,
    deadline_monotonic: float | None = None,
    out_scored_moves: list[tuple[chess.Move, float, float]] | None = None,
) -> chess.Move:
    """Belief-weighted EV move selection with bounded-depth lookahead per world.

    For each candidate move m:
      - Sample N worlds from belief (weighted by particle weights).
      - In each world, simulate "me plays m → opp plays best response → ..."
        to `search_depth` plies. Leaf evaluated with the supplied evaluator.
      - Mean across worlds, weighted by particle weight.
    Return argmax_m mean_EV(m).

    search_depth=1 → leaf eval of "after my move" (= best_action with cleaner
    semantics); search_depth=2 → my move + opp's best response; search_depth>=3
    not yet implemented (would need recursive minimax — straightforward to add
    but skipped for the v1 launch to keep latency bounded).

    Returns argmax. Ties broken uniformly within an epsilon band.
    """
    if not own_legal_moves:
        raise ValueError("pimc_best_action called with empty own_legal_moves")
    if not belief.particles:
        return own_legal_moves[0]
    if search_depth < 1:
        raise ValueError(f"search_depth must be >= 1, got {search_depth}")

    rng = rng or random.Random(0)
    own_color = belief.perspective
    opp_color = not own_color
    sampled_particles, sampled_weights = _sample_particles(belief, max_particles, rng)

    n_moves = len(own_legal_moves)
    weighted_sum = [0.0] * n_moves
    total_weight = [0.0] * n_moves

    deadline_hit = False
    for _round_idx, (particle, weight) in enumerate(zip(sampled_particles, sampled_weights, strict=False)):
        for move_idx, move in enumerate(own_legal_moves):
            if not particle.is_pseudo_legal(move):
                continue
            # Depth 1: leaf eval after my move.
            if search_depth == 1:
                score = evaluator(particle, move, own_color)
            else:
                # Depth 2: push my move, ask "what's opp's best score from
                # there?" — opp picks max from their POV; we negate to get
                # the score from our POV.
                next_board = particle.copy()
                next_board.push(move)
                if next_board.king(own_color) is None or next_board.king(opp_color) is None:
                    # Terminal — leaf eval handles the king-capture sentinel.
                    score = evaluator(particle, move, own_color)
                else:
                    # Switch turn so opp's pseudo_legal_moves enumerate correctly.
                    next_board.turn = opp_color
                    opp_eval = _opp_best_score(next_board, opp_color, evaluator)
                    # Negamax: our score = -opp's score from the post-move position.
                    score = -opp_eval
            weighted_sum[move_idx] += weight * score
            total_weight[move_idx] += weight
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                deadline_hit = True
                break
        if deadline_hit:
            break
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            break

    move_scores: list[tuple[chess.Move, float, float]] = []
    best_score = -float("inf")
    for move_idx, move in enumerate(own_legal_moves):
        if total_weight[move_idx] <= 0.0:
            continue
        mean = weighted_sum[move_idx] / total_weight[move_idx]
        move_scores.append((move, mean, total_weight[move_idx]))
        if mean > best_score:
            best_score = mean

    if out_scored_moves is not None:
        out_scored_moves.extend(move_scores)

    if not move_scores:
        return own_legal_moves[0]

    epsilon = 1e-6
    candidates = [
        (move, support)
        for move, score, support in move_scores
        if score >= best_score - epsilon
    ]
    moves = [m for m, _ in candidates]
    weights = [s for _, s in candidates]
    return rng.choices(moves, weights=weights, k=1)[0]
