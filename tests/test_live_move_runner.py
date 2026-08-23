from __future__ import annotations

import importlib.util
from pathlib import Path

import chess


def _load_runner():
    script = Path(__file__).resolve().parents[1] / "scripts" / "live_move_runner.py"
    spec = importlib.util.spec_from_file_location("live_move_runner", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tier1_live_engines_includes_current_prod_version() -> None:
    """Guard against silently falling back to random when prod engine ID changes.

    Every version registered in the TypeScript registry's PROD_PLAYABLE_ENGINE_IDS
    must appear in TIER1_LIVE_ENGINES. Failure here means live games will fall
    back to builtin-random-legal (as happened with v0.9.1 at launch).
    """
    runner = _load_runner()
    # Keep this list in sync with PROD_PLAYABLE_ENGINE_IDS in registry.ts.
    # Update both together whenever the prod engine version changes.
    prod_tier1_ids = ["python-tier1-v0.9.5"]
    for engine_id in prod_tier1_ids:
        assert engine_id in runner.TIER1_LIVE_ENGINES, (
            f"{engine_id} missing from TIER1_LIVE_ENGINES in live_move_runner.py — "
            "live games will silently fall back to random. Add the entry."
        )


def test_deadline_guard_prefers_visible_king_capture() -> None:
    runner = _load_runner()
    board = chess.Board.empty()
    board.turn = chess.BLACK
    board.set_piece_at(chess.D8, chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(chess.E7, chess.Piece(chess.KING, chess.WHITE))

    king_capture = chess.Move.from_uci("d8e7")
    quiet = chess.Move.from_uci("d8a5")
    view = runner.PerspectiveView(
        perspective=chess.BLACK,
        own_legal_moves=[quiet, king_capture],
        visible_squares={chess.D8, chess.E7, chess.A5},
        visible_piece_map={
            chess.D8: chess.Piece(chess.QUEEN, chess.BLACK),
            chess.E7: chess.Piece(chess.KING, chess.WHITE),
        },
        clock_remaining_ms=8_000,
        increment_ms=2_000,
    )

    assert runner._deadline_guard_move(board, view) == king_capture


def test_deadline_uses_watchdog_budget_with_guard_band() -> None:
    runner = _load_runner()

    assert runner._deadline_monotonic(10.0, {"watchdogTimeoutMs": 5_000}) == 13.8
    assert runner._deadline_monotonic(10.0, {}) is None


def test_budgeted_pick_view_translates_worker_deadline_to_strategy_clock() -> None:
    runner = _load_runner()
    view = runner.PerspectiveView(
        perspective=chess.WHITE,
        own_legal_moves=[chess.Move.from_uci("e2e4")],
        visible_squares={chess.E2, chess.E4},
        visible_piece_map={chess.E2: chess.Piece(chess.PAWN, chess.WHITE)},
        clock_remaining_ms=180_000,
        increment_ms=2_000,
    )

    original_monotonic = runner.time.monotonic
    runner.time.monotonic = lambda: 10.0
    try:
        pick_view, budget_ms = runner._budgeted_pick_view(view, 14.0)
    finally:
        runner.time.monotonic = original_monotonic

    assert pick_view is not None
    assert budget_ms == 3_750
    assert pick_view.clock_remaining_ms == 4_150
    assert pick_view.increment_ms == 3_657
    assert pick_view.own_legal_moves == view.own_legal_moves
    assert pick_view.visible_piece_map == view.visible_piece_map


def test_budgeted_pick_view_guards_when_deadline_is_too_close() -> None:
    runner = _load_runner()
    view = runner.PerspectiveView(
        perspective=chess.WHITE,
        own_legal_moves=[chess.Move.from_uci("e2e4")],
        visible_squares={chess.E2, chess.E4},
        visible_piece_map={chess.E2: chess.Piece(chess.PAWN, chess.WHITE)},
        clock_remaining_ms=180_000,
        increment_ms=2_000,
    )

    original_monotonic = runner.time.monotonic
    runner.time.monotonic = lambda: 10.0
    try:
        pick_view, budget_ms = runner._budgeted_pick_view(view, 10.2)
    finally:
        runner.time.monotonic = original_monotonic

    assert pick_view is None
    assert budget_ms == 0


def test_budgeted_pick_view_guards_when_strategy_budget_is_too_small() -> None:
    runner = _load_runner()
    view = runner.PerspectiveView(
        perspective=chess.WHITE,
        own_legal_moves=[chess.Move.from_uci("e2e4")],
        visible_squares={chess.E2, chess.E4},
        visible_piece_map={chess.E2: chess.Piece(chess.PAWN, chess.WHITE)},
        clock_remaining_ms=180_000,
        increment_ms=2_000,
    )

    original_monotonic = runner.time.monotonic
    runner.time.monotonic = lambda: 10.0
    try:
        pick_view, budget_ms = runner._budgeted_pick_view(view, 12.5)
    finally:
        runner.time.monotonic = original_monotonic

    assert pick_view is None
    assert budget_ms == 2_250


def test_live_stockfish_budget_defaults_are_configurable() -> None:
    runner = _load_runner()
    env = runner.os.environ
    keys = [
        "PYTHON_ENGINE_STOCKFISH_TIME_CAP_SECONDS",
        "PYTHON_ENGINE_STOCKFISH_SLACK_SECONDS",
        "PYTHON_LIVE_STOCKFISH_TIME_CAP_SECONDS",
        "PYTHON_LIVE_STOCKFISH_SLACK_SECONDS",
    ]
    previous = {key: env.get(key) for key in keys}
    try:
        for key in keys:
            env.pop(key, None)
        runner._configure_live_stockfish_budget()
        assert env["PYTHON_ENGINE_STOCKFISH_TIME_CAP_SECONDS"] == "0.05"
        assert env["PYTHON_ENGINE_STOCKFISH_SLACK_SECONDS"] == "0.05"

        for key in keys:
            env.pop(key, None)
        env["PYTHON_LIVE_STOCKFISH_TIME_CAP_SECONDS"] = "0.08"
        env["PYTHON_LIVE_STOCKFISH_SLACK_SECONDS"] = "0.02"
        runner._configure_live_stockfish_budget()
        assert env["PYTHON_ENGINE_STOCKFISH_TIME_CAP_SECONDS"] == "0.08"
        assert env["PYTHON_ENGINE_STOCKFISH_SLACK_SECONDS"] == "0.02"
    finally:
        for key, value in previous.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
