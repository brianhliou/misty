"""The belief-update (observe) cost projection must fail closed, loudly.

Regression for the prod seat-forfeit in game 12c8ff99 (issue #11): one opp-move
belief update at |P|~6M ran 20s+ on the prod vCPUs, straight through the wall
deadline — the worker's deadline checks only ran BETWEEN phases, so the server
watchdog forfeited the seat while the worker was still inside the update, with
nothing in the logs. _feed_transcript_budgeted projects each observe's cost
(|P| x learned us-per-world rate) BEFORE starting it and raises
BeliefUpdateOverBudget when it cannot fit: an explicit engine failure the
server can see, NOT a fallback move — a belief that can't afford this turn's
update can't afford the next one either, so a fallback bot would keep playing
junk while looking alive (the fail-open class the 2026-06-27 mandate
eliminates).
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


def _load_worker():
    script = Path(__file__).resolve().parents[1] / "scripts" / "live_move_worker.py"
    spec = importlib.util.spec_from_file_location("live_move_worker_obs", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The feeder converts observations only AFTER the projection check; identity
    # conversion lets these tests use plain fake observation objects.
    module.observation_from_protocol = lambda obs: obs
    module.move_from_protocol = lambda m: m
    return module


class _Enumerator:
    def __init__(self, size: int) -> None:
        self.size = size


class _Engine:
    def __init__(self, size: int) -> None:
        self.enumerator = _Enumerator(size)


class _Strategy:
    def __init__(self, worlds: int, opp_sleep_s: float = 0.0) -> None:
        self._engine = _Engine(worlds)
        self.opp_sleep_s = opp_sleep_s
        self.calls: list[tuple[str, Any]] = []

    def reset(self, perspective: Any, game_id: str | None = None) -> None:
        self.calls.append(("reset", game_id))

    def observe_own_move(self, move: Any, obs: Any) -> None:
        self.calls.append(("own", obs))

    def observe_opp_move(self, obs: Any) -> None:
        if self.opp_sleep_s:
            time.sleep(self.opp_sleep_s)
        self.calls.append(("opp", obs))


@dataclass
class _Obs:
    kind: str
    ply: int = 0
    own_move: Any = "e2e4"


@dataclass
class _Req:
    observation_transcript: list[_Obs] = field(default_factory=list)
    color: str = "black"
    game_id: str = "game-1"


def test_unaffordable_observe_raises_and_never_starts() -> None:
    worker = _load_worker()
    strategy = _Strategy(worlds=6_000_000)
    req = _Req([_Obs("opp_move", ply=83)])
    deadline = time.monotonic() + 5.0  # prior 6us/world -> ~36s projected
    with pytest.raises(worker.BeliefUpdateOverBudget) as exc:
        worker._feed_transcript_budgeted(strategy, req, 0, deadline, time.monotonic())
    assert strategy.calls == [], "the expensive observe must never start"
    # The refusal must carry the forensics inline — no replay rig required.
    msg = str(exc.value)
    assert "|P|=6000000" in msg and "ply 83" in msg


def test_small_belief_feeds_everything_despite_tight_deadline() -> None:
    worker = _load_worker()
    strategy = _Strategy(worlds=1_000)  # below OBS_PROJECTION_MIN_WORLDS
    req = _Req([_Obs("own_move", ply=1), _Obs("opp_move", ply=2)])
    processed = worker._feed_transcript_budgeted(
        strategy, req, 0, time.monotonic() + 2.0, time.monotonic()
    )
    assert processed == 2
    assert [kind for kind, _ in strategy.calls] == ["own", "opp"]


def test_no_deadline_means_no_projection() -> None:
    worker = _load_worker()
    strategy = _Strategy(worlds=50_000_000)
    req = _Req([_Obs("opp_move", ply=9)])
    assert worker._feed_transcript_budgeted(strategy, req, 0, None, time.monotonic()) == 1
    assert [kind for kind, _ in strategy.calls] == ["opp"]


def test_kill_switch_restores_grind_behavior() -> None:
    worker = _load_worker()
    worker.OBS_PROJECTION = False
    strategy = _Strategy(worlds=6_000_000)
    req = _Req([_Obs("opp_move", ply=83)])
    assert (
        worker._feed_transcript_budgeted(
            strategy, req, 0, time.monotonic() + 5.0, time.monotonic()
        )
        == 1
    )


def test_rate_is_learned_from_measured_updates() -> None:
    worker = _load_worker()
    # 100k worlds (above the learning floor), 50ms observe -> ~0.5 us/world.
    strategy = _Strategy(worlds=100_000, opp_sleep_s=0.05)
    req = _Req([_Obs("opp_move", ply=10)])
    worker._feed_transcript_budgeted(strategy, req, 0, None, time.monotonic())
    rate = worker._OBS_RATE_US_PER_WORLD.get("opp")
    assert rate is not None and 0.3 < rate < 5.0
    # The learned (cheap) rate now beats the conservative prior: the same 6M
    # belief that the prior would refuse is projected affordable and runs.
    fast = _Strategy(worlds=6_000_000)
    assert (
        worker._feed_transcript_budgeted(
            fast, req, 0, time.monotonic() + 5.0, time.monotonic()
        )
        == 1
    )


def test_feed_stats_recorded_for_diagnostics() -> None:
    """The observe-phase cost must land in _LAST_FEED_STATS (merged into the
    persisted per-move diagnostics) — the update was the invisible phase."""
    worker = _load_worker()
    strategy = _Strategy(worlds=1_000)
    req = _Req([_Obs("own_move", ply=1), _Obs("opp_move", ply=2)])
    worker._feed_transcript_budgeted(strategy, req, 0, None, time.monotonic())
    stats = worker._LAST_FEED_STATS
    assert stats["obsPlies"] == 2
    assert stats["obsMaxWorlds"] == 1_000
    assert set(stats["obsMsByMode"]) == {"own", "opp"}
    assert stats["obsMs"] >= 0


def test_partial_feed_resumes_from_processed_len() -> None:
    """Feeding [0:n] then [n:m] must equal a full feed of [0:m] — session
    continuity is processed_len-based, not per-turn."""
    worker = _load_worker()
    strategy = _Strategy(worlds=1_000)
    obs = [_Obs("own_move", ply=1), _Obs("opp_move", ply=2), _Obs("opp_move", ply=3)]
    req = _Req(obs)
    assert worker._feed_transcript_budgeted(strategy, req, 0, None, time.monotonic()) == 3
    strategy2 = _Strategy(worlds=1_000)
    mid = worker._feed_transcript_budgeted(strategy2, _Req(obs[:2]), 0, None, time.monotonic())
    assert mid == 2
    assert worker._feed_transcript_budgeted(strategy2, req, mid, None, time.monotonic()) == 3
    assert [k for k, _ in strategy2.calls] == [k for k, _ in strategy.calls]
