"""Clock-aware, time-control-aware per-move time budgeting — a core engine concern.

Time management belongs IN the engine, driven by the clock the harness already
puts on the PerspectiveView (`clock_remaining_ms`, `increment_ms`). The SAME code
governs both the bakeoff/selfplay harness (which enforces a decrementing clock +
flag-on-time) and live PvE (the worker, which feeds the real game clock). So
"EvE/bakeoff serves PvE" is literal: the engine budgets time the same way in
both, and the bakeoff measures strength AS PvE will experience it.

SOLVENCY FIRST. The budget must not flag — across every official time control
(1+1 bullet, 3+2 blitz, 5+5 rapid) AND the long tail of Fog-of-War game lengths.
FoW games run much longer than chess: 70+ moves/side happens, and a 146-ply game
flagged under the old formula (bakeoff 2026-06-03, game g0027 at 1+1).

The solvent model (FOW_V2_TIME_SOLVENT=1, default) is time-control-aware WITHOUT
being told the time control — it self-adapts from (remaining, increment) alone:

    reserve   = a protected buffer never spent (absorbs per-move OVERSHOOT at the
                low-clock end: actual wall ≥ budget because belief enumeration +
                anytime granularity have an irreducible floor, so an isolated
                heavy-fog spike must not flag us)
    spendable = max(0, remaining - reserve)
    budget    = inc_fraction * increment + spendable / moves_to_go
    budget    = clamp(min_think, budget, hard_cap)

Key property: as the bank drains, the per-move budget falls toward
`inc_fraction * increment`, which is BELOW the increment (inc_fraction < 1). So
even with overshoot the bank stops draining and converges to a stable floor near
`reserve` — it cannot run out. At 1+1 the budget self-limits to ~2s (never reaches
the 5s cap); at 3+2 / 5+5 the cap binds in the opening, as before. There is no
moves-to-go GUESS that can be wrong because the formula is self-correcting, not a
one-shot division of the whole bank.

The legacy formula (`increment + remaining/divisor`, no reserve) targeted ≥ the
increment at every point, so accumulated overshoot drained long games to a flag.
Recover it with FOW_V2_TIME_SOLVENT=0 (kept for A/B + rollback).

When there is no clock (clock_remaining_ms is None — untimed, or a fixed-budget
bakeoff), `budget_for` returns the static fallback, preserving prior behavior.

Position-awareness (spend MORE on heavy fog / close margins, less on forced
moves) is future work — the `hint` parameter is reserved for it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeManager:
    """Solvency-first per-move budget. Defaults validated across 1+1 / 3+2."""

    hard_cap_s: float = 5.0       # upper clamp; binds for blitz/rapid openings only
    min_think_s: float = 0.5      # always search at least this
    # solvent-model params (the default path)
    moves_to_go: float = 40.0     # FoW horizon for drawing down the bank (chess ~30)
    reserve_s: float = 3.0        # protected buffer, never spent (overshoot guard)
    inc_fraction: float = 0.5     # sustainable spend = this * increment (< 1 ⇒ overshoot-safe)
    solvent: bool = True          # FOW_V2_TIME_SOLVENT; False ⇒ legacy formula
    # legacy-model param (FOW_V2_TIME_SOLVENT=0 only)
    divisor: float = 30.0

    @classmethod
    def from_env(cls) -> "TimeManager":
        def _f(name: str, default: float) -> float:
            v = os.environ.get(name)
            return float(v) if v is not None else default

        return cls(
            hard_cap_s=_f("FOW_V2_TIME_HARD_CAP_S", 5.0),
            min_think_s=_f("FOW_V2_TIME_FLOOR_S", 0.5),
            moves_to_go=_f("FOW_V2_TIME_MOVES_TO_GO", 40.0),
            reserve_s=_f("FOW_V2_TIME_RESERVE_S", 3.0),
            inc_fraction=_f("FOW_V2_TIME_INC_FRACTION", 0.5),
            solvent=os.environ.get("FOW_V2_TIME_SOLVENT", "1") == "1",
            divisor=_f("FOW_V2_TIME_DIVISOR", 30.0),
        )

    def budget_for(
        self,
        clock_remaining_ms: int | None,
        increment_ms: int = 0,
        static_fallback_s: float | None = None,
        hint: object | None = None,  # reserved: position-aware budgeting
    ) -> float | None:
        """Per-move wall budget in seconds.

        No clock (None) → `static_fallback_s` (the engine's configured fixed
        budget, possibly None = iters bind). With a clock → the solvent formula
        (default) or the legacy formula (FOW_V2_TIME_SOLVENT=0).
        """
        if clock_remaining_ms is None:
            return static_fallback_s
        inc_s = (increment_ms or 0) / 1000.0
        rem_s = max(0.0, clock_remaining_ms / 1000.0)
        if self.solvent:
            spendable = max(0.0, rem_s - self.reserve_s)
            budget = self.inc_fraction * inc_s + spendable / self.moves_to_go
        else:
            budget = inc_s + rem_s / self.divisor
        return max(self.min_think_s, min(self.hard_cap_s, budget))
