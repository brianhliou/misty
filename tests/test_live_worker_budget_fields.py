"""The worker's wall deadline comes from the TRANSPORT bound, not the compute budget.

Regression for the deadline-guard cliff. The TS caller used to send
`watchdogTimeoutMs: computeBudgetMs`, so the worker built its wall deadline from
the engine's per-move compute allowance. Because the usable pick window is
`deadline - DEADLINE_GUARD_MS - PICK_DEADLINE_GUARD_MS` and `_budgeted_pick_view`
vetoes below MIN_STRATEGY_PICK_BUDGET_MS, any compute budget under ~4.45s vetoed
EVERY move to an unsearched deadline-guard move. At 3+2 that is any clock below
~29.8s — prod game 8d08b93a ended at 32.6s with 233ms of slack.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path


def _load_worker():
    script = Path(__file__).resolve().parents[1] / "scripts" / "live_move_worker.py"
    spec = importlib.util.spec_from_file_location("live_move_worker", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick_window_ms(worker, request: dict[str, object]) -> float:
    """The budget `_budgeted_pick_view` gets to compare against its floor."""
    started = time.monotonic()
    deadline = worker._deadline_monotonic(started, request)
    assert deadline is not None
    return (deadline - time.monotonic()) * 1000 - worker.PICK_DEADLINE_GUARD_MS


def test_wall_deadline_uses_transport_bound_not_compute_budget() -> None:
    worker = _load_worker()
    # The prod values at game 8d08b93a ply 79, where the guard fired in 916ms.
    request = {"workerDeadlineMs": 15_100, "computeBudgetMs": 5_100}
    window = _pick_window_ms(worker, request)
    assert window > worker.MIN_STRATEGY_PICK_BUDGET_MS, (
        f"pick window {window:.0f}ms must clear the {worker.MIN_STRATEGY_PICK_BUDGET_MS}ms "
        "floor when the transport allows 15.1s; sourcing it from the 5.1s compute "
        "budget is what vetoed searched moves to the deadline-guard"
    )
    # Sanity: the old sourcing really was marginal — that is the bug, not a nit.
    legacy_window = _pick_window_ms(worker, {"watchdogTimeoutMs": 5_100})
    assert legacy_window - worker.MIN_STRATEGY_PICK_BUDGET_MS < 1_000


def test_compute_budget_below_the_old_cliff_still_searches() -> None:
    """A 4.0s compute budget (3+2 at ~25s on the clock) used to veto every move."""
    worker = _load_worker()
    request = {"workerDeadlineMs": 14_000, "computeBudgetMs": 4_000}
    assert _pick_window_ms(worker, request) > worker.MIN_STRATEGY_PICK_BUDGET_MS


def test_legacy_payload_still_resolves_both_quantities() -> None:
    """An old TS caller sends only watchdogTimeoutMs; behavior must not change."""
    worker = _load_worker()
    legacy = {"watchdogTimeoutMs": 9_000}
    assert worker._compute_budget_ms(legacy) == 9_000
    started = time.monotonic()
    deadline = worker._deadline_monotonic(started, legacy)
    assert deadline is not None
    expected = started + (9_000 - worker.DEADLINE_GUARD_MS) / 1000.0
    assert abs(deadline - expected) < 1e-6


def test_missing_fields_leave_the_deadline_unset() -> None:
    worker = _load_worker()
    assert worker._deadline_monotonic(time.monotonic(), {}) is None
    assert worker._compute_budget_ms({}) is None


def test_explicit_compute_budget_wins_over_legacy_key() -> None:
    worker = _load_worker()
    assert (
        worker._compute_budget_ms({"computeBudgetMs": 3_000, "watchdogTimeoutMs": 12_000}) == 3_000
    )


def test_compute_budget_bounds_the_synthetic_clock_without_vetoing() -> None:
    """The compute budget must SHRINK the tier-1 budget, never trigger the guard:
    2s of compute left is a reason to search for 2s, not to skip searching."""
    worker = _load_worker()

    # dataclasses.replace() (used inside _budgeted_pick_view) requires a real
    # dataclass instance, so stub the two clock fields it rewrites.
    @dataclass
    class _View:
        clock_remaining_ms: int = 120_000
        increment_ms: int = 2_000

    deadline = time.monotonic() + 12.0
    generous, _ = worker._budgeted_pick_view(_View(), deadline, 9_000)
    tight, _ = worker._budgeted_pick_view(_View(), deadline, 2_000)
    assert generous is not None
    assert tight is not None, "a small compute budget must not veto the search"
    assert tight.clock_remaining_ms < generous.clock_remaining_ms
