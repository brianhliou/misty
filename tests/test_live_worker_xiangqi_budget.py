"""fdx is TIME-bounded from v1.2, and v1.0/v1.1 stay exactly as they shipped.

v1.1 ran a 64-iteration cap, so the deadline `break` inside
`solve_multiroot_rust_tree` was dead code: 64 iterations cost ~150ms against a
16-21s budget. Measured in prod over a full 5+5 game, Misty played 33 moves in
24.4s total and finished with MORE clock than it started with.

Two things had to change together, and this file pins both plus the blast
radius:

  * the iteration cap goes out of reach on the v1.2 bundle, so wall time is what
    actually stops the search;
  * a per-move ceiling caps the clock formula, which is solvency-shaped (it
    answers "can I afford this?") and therefore hands a blitz move 16-21s.

The cap alone would be a hang: `_mini_budget_seconds` returns None in untimed
dev with no watchdog, and an unreachable iteration cap then spins forever.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest


def _load_worker():
    script = Path(__file__).resolve().parents[1] / "scripts" / "live_move_worker.py"
    spec = importlib.util.spec_from_file_location("live_move_worker", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Clock:
    remaining_ms: int | None
    increment_ms: int | None


@dataclass
class _Req:
    clock: _Clock


def _req(remaining_ms: int | None, increment_ms: int | None = 5_000) -> _Req:
    return _Req(clock=_Clock(remaining_ms=remaining_ms, increment_ms=increment_ms))


# ── Iteration caps ───────────────────────────────────────────────────────────


def test_v12_iteration_cap_is_out_of_reach(monkeypatch):
    monkeypatch.delenv("FOW_XIANGQI_ITERS", raising=False)
    monkeypatch.delenv("FOW_XIANGQI_PROFILE", raising=False)
    worker = _load_worker()
    prof = worker._xiangqi_profile("python-fdx-v1.2")
    assert prof["iterations"] >= 1_000_000, "time must bind, not the iteration cap"
    assert prof["max_budget_s"] == pytest.approx(4.0)


def test_shipped_versions_keep_their_64_iteration_cap(monkeypatch):
    """Never edit a shipped version in place: v1.0 and v1.1 are frozen."""
    monkeypatch.delenv("FOW_XIANGQI_ITERS", raising=False)
    monkeypatch.delenv("FOW_XIANGQI_PROFILE", raising=False)
    worker = _load_worker()
    for engine_id in ("python-fdx-v1.0", "python-fdx-v1.1"):
        prof = worker._xiangqi_profile(engine_id)
        assert prof["iterations"] == 64, engine_id
        assert prof["max_budget_s"] is None, engine_id


def test_env_still_overrides_the_bundle_iteration_cap(monkeypatch):
    monkeypatch.setenv("FOW_XIANGQI_ITERS", "250")
    monkeypatch.delenv("FOW_XIANGQI_PROFILE", raising=False)
    worker = _load_worker()
    assert worker._xiangqi_profile("python-fdx-v1.2")["iterations"] == 250


# ── The per-move ceiling ─────────────────────────────────────────────────────


def test_uncapped_budget_is_unchanged_for_existing_callers():
    """DMX passes no ceiling; its human-validated shape must not move."""
    worker = _load_worker()
    request = {"watchdogTimeoutMs": 34_100}
    # 5+5, move 1: 300s bank -> 300*0.04 + 5*0.8 = 16.0s, under the 32.9s watchdog.
    assert worker._mini_budget_seconds(_req(300_000), request) == pytest.approx(16.0)


def test_ceiling_caps_the_blitz_budget():
    worker = _load_worker()
    request = {"watchdogTimeoutMs": 34_100}
    assert worker._mini_budget_seconds(_req(300_000), request, max_seconds=4.0) == pytest.approx(
        4.0
    )
    # Late game the bank has GROWN (the engine banks its increment), so the raw
    # budget rises to ~21.5s while the ceiling holds the reply steady.
    assert worker._mini_budget_seconds(_req(437_500), request, max_seconds=4.0) == pytest.approx(
        4.0
    )


def test_ceiling_is_inert_at_genuinely_fast_controls():
    """The anti-flag property of the clock formula must survive the ceiling."""
    worker = _load_worker()
    request = {"watchdogTimeoutMs": 6_100}
    # 3+2 with 20s banked: 20*0.04 + 2*0.8 = 2.4s, already under a 4s ceiling.
    uncapped = worker._mini_budget_seconds(_req(20_000, 2_000), request)
    capped = worker._mini_budget_seconds(_req(20_000, 2_000), request, max_seconds=4.0)
    assert uncapped == pytest.approx(2.4)
    assert capped == pytest.approx(uncapped)


def test_untimed_dev_with_a_ceiling_never_returns_none():
    """The hang guard: an unreachable iteration cap + a None budget spins forever."""
    worker = _load_worker()
    no_watchdog: dict[str, object] = {}
    assert worker._mini_budget_seconds(_req(None), no_watchdog) is None, "prior behavior"
    assert worker._mini_budget_seconds(_req(None), no_watchdog, max_seconds=4.0) == pytest.approx(
        4.0
    )


def test_untimed_dev_with_a_watchdog_takes_the_tighter_of_the_two():
    worker = _load_worker()
    request = {"watchdogTimeoutMs": 2_200}
    # hard_s = (2200 - guard)/1000, clamped by the 2.0s untimed cap, then by ours.
    assert worker._mini_budget_seconds(_req(None), request, max_seconds=0.5) == pytest.approx(0.5)
