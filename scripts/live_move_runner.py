"""Pick one live PvE move for the TypeScript server.

Protocol (Phase 3c — protocol-only):
  stdin: JSON request of shape {
    "engineTurnRequest": <EngineTurnRequest, see
      packages/game/src/engine-protocol.ts>,
    "watchdogTimeoutMs": <int or null>,
    "stockfishPath": <str, optional>,
  }
  stdout: JSON response of shape {
    "roomId": str, "engine": {...}, "decisionSource": str,
    "move": {"from": str, "to": str, "promotion"?: str}
  }

The server owns HTTP, rooms, legality validation, and fallback. This
runner is the subprocess-per-move fallback used when the persistent
worker pool isn't initialized; it shares the same protocol surface as
live_move_worker.py.

The engine has access ONLY to the redacted EngineTurnRequest — no
canonical events, GameState, master seed, or opp clock. The redaction
boundary is enforced server-side at
apps/server/src/engine-protocol/build.ts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("PYTHON_ENGINE_LAB_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC = ROOT / "src"
VENDOR = ROOT / "vendor"
for path in (VENDOR, SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import chess

from fow_chess.engine_protocol import request_from_json
from fow_chess.protocol_adapter import (
    board_from_request,
    build_perspective_view,
    replay_transcript_into_strategy,
)
from fow_chess.selfplay import PerspectiveView
from fow_chess.strategies import RandomStrategy
try:
    from fow_chess.tournament.config import canonical_hash, load_config
    from fow_chess.tournament.runtime import bot_runtime
except ImportError:  # chess-only build: legacy tier1 snapshot engines absent
    canonical_hash = load_config = bot_runtime = None  # type: ignore[assignment]

TIER1_CONFIG_HASH = "b22f29dd73f5"
DEADLINE_GUARD_MS = int(os.environ.get("PYTHON_LIVE_DEADLINE_GUARD_MS", "1200"))
PICK_DEADLINE_GUARD_MS = int(os.environ.get("PYTHON_LIVE_PICK_DEADLINE_GUARD_MS", "250"))
MIN_PICK_BUDGET_MS = 50
MIN_STRATEGY_PICK_BUDGET_MS = int(
    os.environ.get("PYTHON_LIVE_MIN_STRATEGY_PICK_BUDGET_MS", "3000")
)
DEFAULT_LIVE_STOCKFISH_TIME_CAP_SECONDS = "0.05"
DEFAULT_LIVE_STOCKFISH_SLACK_SECONDS = "0.05"
MATERIAL_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 100_000,
}
TIER1_LIVE_ENGINES: dict[str, dict[str, str]] = {
    "python-tier1-v0.9.5": {
        "tier1Version": "0.9.5",
        "playSignature": "372b4bb6c064",
        "engineVersion": "v0.9.5-tactical-patches@372b4bb6c064",
    },
    "python-tier1-v0.9.1": {
        "tier1Version": "0.9.1",
        "playSignature": "8918f287499f",
        "engineVersion": "v0.9.1-pawn-shield-diagonal@8918f287499f",
    },
    "python-tier1-v0.7.22": {
        "tier1Version": "0.7.22",
        "playSignature": "5d3ddffa74f6",
        "engineVersion": "v0.7.22-king-risk@5d3ddffa74f6",
    },
    "python-tier1-v0.8.9": {
        "tier1Version": "0.8.9",
        "playSignature": "2c010d792075",
        "engineVersion": "v0.8.9-repair-caps@2c010d792075",
    },
    # Live current-src variant; empty engineVersion → load from src/fow_chess.
    "python-tier1-current": {
        "tier1Version": "current",
        "playSignature": "current",
        "engineVersion": "",
    },
}


def main() -> int:
    started = time.monotonic()
    request = json.load(sys.stdin)
    req = request_from_json(request["engineTurnRequest"])
    # Subprocess-per-move: derive the strategy seed from the protocol's
    # engineSeed (which the server derives per-turn from a per-engine
    # secret + game + ply). Matches the worker-pool behavior of resetting
    # state per request — there's no cross-request state to preserve here.
    seed = req.engine_seed
    stockfish_path = str(request.get("stockfishPath") or "stockfish")
    engine_spec: dict[str, Any] = {"id": req.engine_id}

    _configure_live_stockfish_budget()
    _debug(
        "request-loaded",
        started,
        roomId=req.game_id,
        engineId=req.engine_id,
        color=req.color,
        ply=req.ply,
        legalCount=len(req.legal_moves),
        seed=seed,
        clockRemainingMs=req.clock.remaining_ms,
        incrementMs=req.clock.increment_ms,
    )

    if not req.legal_moves:
        raise RuntimeError("no legal moves available")
    view = build_perspective_view(req)

    deadline = _deadline_monotonic(started, request)
    if _deadline_expired(deadline):
        guard_board = board_from_request(req)
        move = _deadline_guard_move(guard_board, view)
        _debug("deadline-guard", started, phaseBefore="runtime-ready", move=move.uci())
        _print_response(req, engine_spec, move, "deadline-guard")
        return 0

    with strategy_runtime(engine_spec, seed, stockfish_path) as strategy:
        _debug("runtime-ready", started)
        # Cold-start: replay the full observation transcript from the
        # protocol through the strategy's observe_own_move /
        # observe_opp_move hooks. `replay_transcript_into_strategy`
        # calls strategy.reset(perspective) first.
        replay_transcript_into_strategy(strategy, req)
        _debug(
            "transcript-replayed",
            started,
            transcriptLen=(
                len(req.observation_transcript) if req.observation_transcript else 0
            ),
        )
        if _deadline_expired(deadline):
            guard_board = board_from_request(req)
            move = _deadline_guard_move(guard_board, view)
            _debug(
                "deadline-guard",
                started,
                phaseBefore="pick-started",
                move=move.uci(),
            )
            _print_response(req, engine_spec, move, "deadline-guard")
            return 0
        pick_view, pick_budget_ms = _budgeted_pick_view(view, deadline)
        if pick_view is None:
            guard_board = board_from_request(req)
            move = _deadline_guard_move(guard_board, view)
            _debug(
                "deadline-guard",
                started,
                phaseBefore="pick-started",
                move=move.uci(),
                pickBudgetMs=pick_budget_ms,
                minStrategyPickBudgetMs=MIN_STRATEGY_PICK_BUDGET_MS,
            )
            _print_response(req, engine_spec, move, "deadline-guard")
            return 0
        _debug(
            "pick-started",
            started,
            ownLegalCount=len(pick_view.own_legal_moves),
            visibleSquareCount=len(pick_view.visible_squares),
            visiblePieceCount=len(pick_view.visible_piece_map),
            pickBudgetMs=pick_budget_ms,
            strategyClockRemainingMs=pick_view.clock_remaining_ms,
            strategyIncrementMs=pick_view.increment_ms,
        )
        move = strategy.pick_move(pick_view)
        _debug("pick-finished", started, move=move.uci())
        if move not in view.own_legal_moves:
            raise RuntimeError(f"engine returned illegal move: {move.uci()}")

    decision_source = "random" if isinstance(strategy, RandomStrategy) else "tier1"
    _print_response(req, engine_spec, move, decision_source)
    return 0


def _print_response(
    req: Any,
    engine_spec: dict[str, Any],
    move: chess.Move,
    decision_source: str,
) -> None:
    """Response shape matches apps/server PythonPoolResponse.

    Castling uses king-destination (e1→g1), not rook-square — variants.ts
    accepts both forms via alias generation (variants.ts:589).
    """
    promo_letter = None
    if move.promotion is not None:
        promo_letter = {
            chess.QUEEN: "queen", chess.ROOK: "rook",
            chess.BISHOP: "bishop", chess.KNIGHT: "knight",
        }[move.promotion]
    print(json.dumps({
        "roomId": req.game_id,
        "engine": engine_metadata(engine_spec),
        "decisionSource": decision_source,
        "move": {
            "from": chess.SQUARE_NAMES[move.from_square],
            "to": chess.SQUARE_NAMES[move.to_square],
            **({"promotion": promo_letter} if promo_letter else {}),
        },
    }, separators=(",", ":")))


def _debug(phase: str, started: float, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "kind": "python_live_engine_debug",
                "phase": phase,
                "elapsedMs": round((time.monotonic() - started) * 1000),
                **fields,
            },
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


def _deadline_monotonic(started: float, request: dict[str, Any]) -> float | None:
    watchdog_timeout_ms = _parse_optional_int(request.get("watchdogTimeoutMs"))
    if watchdog_timeout_ms is None:
        return None
    budget_ms = max(1, watchdog_timeout_ms - DEADLINE_GUARD_MS)
    return started + budget_ms / 1000.0


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _budgeted_pick_view(
    view: PerspectiveView,
    deadline: float | None,
) -> tuple[PerspectiveView | None, int | None]:
    """Constrain legacy strategies with the worker's external compute deadline.

    Tier-1 v0.9.5 only accepts a clock-bearing PerspectiveView; it does not
    accept an explicit deadline parameter. The live worker's top-level
    watchdogTimeoutMs is already the external compute budget, so translate the
    remaining worker deadline into clock fields that the strategy uses only for
    its internal deadline calculation.
    """
    if deadline is None:
        return view, None
    remaining_ms = int((deadline - time.monotonic()) * 1000) - PICK_DEADLINE_GUARD_MS
    if view.clock_remaining_ms is not None:
        remaining_ms = min(remaining_ms, max(0, view.clock_remaining_ms))
    if remaining_ms < MIN_STRATEGY_PICK_BUDGET_MS:
        return None, max(0, remaining_ms)

    target_ms = max(MIN_PICK_BUDGET_MS, min(10_000, remaining_ms))
    # Tier-1 computes budget ~= ((clock - 400) / 40) + increment, then clamps
    # to the usable clock. Choose synthetic clock/increment fields that produce
    # a budget near target_ms while preserving all non-clock view fields.
    strategy_clock_ms = target_ms + 400
    strategy_increment_ms = max(0, target_ms - target_ms // 40)
    return (
        replace(
            view,
            clock_remaining_ms=strategy_clock_ms,
            increment_ms=strategy_increment_ms,
        ),
        target_ms,
    )


def _deadline_guard_move(board: chess.Board, view: PerspectiveView) -> chess.Move:
    return max(
        sorted(view.own_legal_moves, key=lambda move: move.uci()),
        key=lambda move: _deadline_guard_score(board, view, move),
    )


def _deadline_guard_score(
    board: chess.Board,
    view: PerspectiveView,
    move: chess.Move,
) -> tuple[int, int, int, int, int]:
    """Score a fallback move. Highest tuple wins. See the twin in
    `live_move_worker._deadline_guard_score` for the full rationale.

    `board` carries only the side-to-move's *visible* pieces, so the
    attacker/defender queries are "as far as we can see" — the best a fast
    fallback can do under fog. Priority: (1) king-safety — never leave/place
    the own king on a square a visible enemy attacks if avoidable; (2) net
    capture material, where a free capture scores full value and a capture
    into a visible recapture costs the mover's full value (the old
    `mover // 20` term made KING captures score ~-5000, so the king would
    never take even an undefended queen — it threw the king in room d860f498);
    (3) castle/promotion, central destination, then a deterministic tiebreak.
    """
    perspective = view.perspective
    mover = view.visible_piece_map.get(move.from_square)
    target = view.visible_piece_map.get(move.to_square)

    after = board.copy(stack=False)
    after.remove_piece_at(move.from_square)
    placed = chess.Piece(move.promotion, perspective) if move.promotion else mover
    if placed is not None:
        after.set_piece_at(move.to_square, placed)

    king_sq = after.king(perspective)
    king_safe = king_sq is None or not after.attackers(not perspective, king_sq)

    capture_score = 0
    if target is not None and target.color != perspective:
        capture_score = MATERIAL_VALUE.get(target.piece_type, 0)
        if mover is not None and after.attackers(not perspective, move.to_square):
            capture_score -= MATERIAL_VALUE.get(mover.piece_type, 0)

    castle_score = 80 if board.is_castling(move) else 0
    promotion_score = 70 if move.promotion is not None else 0
    center_score = 10 if move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5} else 0
    return (
        1 if king_safe else 0,
        capture_score,
        castle_score + promotion_score,
        center_score,
        -_move_sort_value(move),
    )


def _move_sort_value(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square + (move.promotion or 0)


class strategy_runtime:
    def __init__(self, spec: dict[str, Any], seed: int, stockfish_path: str) -> None:
        self.spec = spec
        self.seed = seed
        self.stockfish_path = stockfish_path
        self._runtime = None
        self._strategy = None

    def __enter__(self):
        engine_id = str(self.spec.get("id") or "")
        if engine_id in {"python-random-legal", "builtin-random-legal"}:
            self._strategy = RandomStrategy(seed=self.seed)
            return self._strategy
        tier1 = TIER1_LIVE_ENGINES.get(engine_id)
        if tier1 is not None:
            if bot_runtime is None:
                raise RuntimeError(
                    f"engine {engine_id!r} needs fow_chess.tournament (legacy "
                    "tier1 snapshots), which is not present in this build"
                )
            config = load_config(ROOT / "configs" / "tier1-v1.json")
            if canonical_hash(config) != TIER1_CONFIG_HASH:
                raise RuntimeError("tier1-v1 config hash mismatch")
            if tier1.get("engineVersion"):
                config = replace(config, engine_version=tier1["engineVersion"])
            self._runtime = bot_runtime(config, stockfish_path=self.stockfish_path)
            factory = self._runtime.__enter__()
            self._strategy = factory(self.seed)
            return self._strategy
        raise RuntimeError(f"unsupported Python live engine: {engine_id}")

    def __exit__(self, exc_type, exc, tb):
        if self._runtime is not None:
            return self._runtime.__exit__(exc_type, exc, tb)
        return False


def engine_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    engine_id = str(spec.get("id") or "")
    tier1 = TIER1_LIVE_ENGINES.get(engine_id)
    if tier1 is not None:
        return {
            "id": engine_id,
            "tier1Version": tier1["tier1Version"],
            "configHash": TIER1_CONFIG_HASH,
            "playSignature": tier1["playSignature"],
            "engineVersion": tier1["engineVersion"],
        }
    return {"id": engine_id}


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _configure_live_stockfish_budget() -> None:
    os.environ.setdefault(
        "PYTHON_ENGINE_STOCKFISH_TIME_CAP_SECONDS",
        os.environ.get(
            "PYTHON_LIVE_STOCKFISH_TIME_CAP_SECONDS",
            DEFAULT_LIVE_STOCKFISH_TIME_CAP_SECONDS,
        ),
    )
    os.environ.setdefault(
        "PYTHON_ENGINE_STOCKFISH_SLACK_SECONDS",
        os.environ.get(
            "PYTHON_LIVE_STOCKFISH_SLACK_SECONDS",
            DEFAULT_LIVE_STOCKFISH_SLACK_SECONDS,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
