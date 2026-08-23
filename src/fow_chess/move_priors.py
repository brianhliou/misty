"""Opponent move priors used by the particle filter."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Protocol

import chess


class OpponentMovePrior(Protocol):
    """Probability over an opponent's legal moves given a candidate true board."""

    def __call__(
        self, board: chess.Board, legal: list[chess.Move]
    ) -> list[float]: ...


def uniform_prior(board: chess.Board, legal: list[chess.Move]) -> list[float]:
    """Uniform distribution over legal moves; the simplest baseline."""
    n = len(legal)
    return [1.0 / n] * n if n else []


_PIECE_INDEX_PRIOR = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}


def learned_policy_prior(weights_path: str, temperature: float = 1.0) -> OpponentMovePrior:
    """Small MLP policy net trained on production self-play.

    Predicts the move distribution the engine would play from this position.
    Plugged into the belief filter as the opponent model — should outperform
    uniform_prior when modeling a known engine because real opponents play
    purposefully, not uniformly.

    The path must be a .npz with state_dict tensors (fc{1,2,3}.weight,
    fc{1,2,3}.bias). Pure numpy inference — keeps torch out of the runtime
    deploy. Train with scripts/train_policy_net.py which writes both .pt
    (torch checkpoint, for retraining) and .npz (for serving).
    """
    import numpy as np

    weights = np.load(weights_path)
    W1, b1 = weights["fc1.weight"], weights["fc1.bias"]
    W2, b2 = weights["fc2.weight"], weights["fc2.bias"]
    W3, b3 = weights["fc3.weight"], weights["fc3.bias"]
    in_dim = W1.shape[1]
    buf = np.zeros(in_dim, dtype=np.float32)

    def _encode(board: chess.Board, perspective: chess.Color):
        buf[:] = 0.0
        for sq, piece in board.piece_map().items():
            pi = _PIECE_INDEX_PRIOR[piece.piece_type]
            if piece.color == perspective:
                buf[pi * 64 + sq] = 1.0
            else:
                buf[384 + pi * 64 + sq] = 1.0
        return buf

    def _forward(x):
        h1 = np.maximum(0.0, W1 @ x + b1)
        h2 = np.maximum(0.0, W2 @ h1 + b2)
        return W3 @ h2 + b3

    # The belief filter invokes this once per particle on the same opp ply.
    # Many particles share canonical boards after observation filtering — the
    # cache collapses 100+ calls per opp ply into a handful of forward passes.
    # CACHE_CAP guards against unbounded growth across long games.
    logits_cache: dict[str, np.ndarray] = {}
    CACHE_CAP = 4096

    def prior(board: chess.Board, legal: list[chess.Move]) -> list[float]:
        if not legal:
            return []
        # The prior is called from the BELIEF FILTER for an OPPONENT move:
        # the board passed in is from the opponent's perspective (board.turn
        # is the opponent we're modeling). Encode from their POV.
        fen = board.fen()
        logits = logits_cache.get(fen)
        if logits is None:
            x = _encode(board, board.turn)
            logits = _forward(x)
            if len(logits_cache) < CACHE_CAP:
                logits_cache[fen] = logits
        scores = np.fromiter(
            (logits[m.from_square * 64 + m.to_square] for m in legal),
            dtype=np.float32, count=len(legal),
        )
        scores = scores / max(temperature, 1e-6)
        scores = scores - scores.max()  # numerical stability
        ws = np.exp(scores)
        total = float(ws.sum())
        if total <= 0:
            return [1.0 / len(legal)] * len(legal)
        return (ws / total).tolist()

    return prior


def stockfish_shallow_prior(
    *,
    path: str = "stockfish",
    depth: int = 4,
    movetime_ms: int = 50,
    top_k: int = 8,
    softmax_temperature_cp: float = 100.0,
    uniform_blend: float = 0.3,
    threads: int = 1,
) -> tuple[OpponentMovePrior, Callable[[], None]]:
    """Stockfish-shallow prior — top-K moves at depth 4, softmaxed cp scores.

    The returned prior is a closure over a persistent Stockfish subprocess
    configured with `MultiPV=top_k`. Each call sends `position fen ... ; go
    depth D movetime T`, parses multipv info lines for top-K candidate moves
    and their cp scores, and returns a distribution over `legal`.

    Distribution shape: softmax(cp_score / temperature) over top-K moves,
    blended with uniform via `uniform_blend` so non-top-K moves retain
    enough mass that truth-particles don't die when Stockfish-shallow misses
    truth's actual move. With uniform_blend=0.3, top-K moves get ~9-10x the
    weight of non-top-K moves over typical legal-move counts — significant
    pruning without extinction.

    Returns (prior_callable, close_callable). The caller MUST invoke close()
    when done; use `stockfish_shallow_prior_ctx` for automatic cleanup.

    Falls back to uniform on engine crash, timeout, or no parsable scores.
    """
    from .evaluator import _UCIEngine  # local import; evaluator does not import this module

    engine = _UCIEngine(path=path, threads=threads)
    engine.setoption("MultiPV", str(top_k))

    # Particle expansion calls the prior once per particle; many particles
    # share boards after observation filtering, so this cache typically
    # collapses 100+ calls per opp ply into a small number of unique fens.
    # Cap is a safety against unbounded growth across long games.
    cache: dict[str, dict[str, float] | None] = {}
    CACHE_CAP = 4096

    def prior(board: chess.Board, legal: list[chess.Move]) -> list[float]:
        n = len(legal)
        if n == 0:
            return []
        fen = board.fen()
        if fen in cache:
            candidates = cache[fen]
        else:
            try:
                candidates = engine.analyze_fen_multipv(
                    board.fen(),
                    depth=depth,
                    movetime_ms=movetime_ms,
                    slack_seconds=0.3,
                )
            except Exception:
                candidates = None
            if len(cache) < CACHE_CAP:
                cache[fen] = candidates

        if not candidates:
            return [1.0 / n] * n

        # Softmax over top-K. Subtract max for numerical stability.
        topk_scores: list[float | None] = [candidates.get(mv.uci()) for mv in legal]
        present = [s for s in topk_scores if s is not None]
        if not present:
            return [1.0 / n] * n
        max_score = max(present)
        sf_weights: list[float] = [
            math.exp((s - max_score) / softmax_temperature_cp) if s is not None else 0.0
            for s in topk_scores
        ]
        sf_total = sum(sf_weights)
        if sf_total <= 0:
            return [1.0 / n] * n
        sf_normalized = [w / sf_total for w in sf_weights]

        # Blend with uniform so non-top-K moves keep enough mass to survive.
        uniform_share = 1.0 / n
        return [
            (1.0 - uniform_blend) * sw + uniform_blend * uniform_share
            for sw in sf_normalized
        ]

    def close() -> None:
        engine.close()

    return prior, close


@contextmanager
def stockfish_shallow_prior_ctx(**kwargs: Any) -> Iterator[OpponentMovePrior]:
    """Context-manager wrapper around `stockfish_shallow_prior`."""
    prior, close = stockfish_shallow_prior(**kwargs)
    try:
        yield prior
    finally:
        close()
