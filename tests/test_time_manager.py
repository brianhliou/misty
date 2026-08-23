"""TimeManager budget tests — solvency first.

The headline test is a clock SIMULATION at 1+1: it reproduces the failure the
2026-06-03 bakeoff caught (game g0027 flagged at 146 plies) and asserts the
solvent model survives it. A budget that passes a unit "≤ cap" check but still
flags over a long game is useless — so we simulate the whole game.
"""
from fow_chess.cfr.time_manager import TimeManager


def _simulate(tm, *, initial_s, inc_s, moves, overshoot_s, spike_s=0.0, spike_every=0):
    """Play `moves` ply on one side at (initial, inc); return min bank (s) seen.

    Per-move actual wall = budget + overshoot (the irreducible belief-enum +
    anytime-granularity cost the budget can't cut), with an occasional heavy-fog
    `spike_s` every `spike_every` moves. Bank ≤ 0 at any point == a flag.
    """
    bank = initial_s
    min_bank = bank
    for i in range(moves):
        budget = tm.budget_for(int(bank * 1000), int(inc_s * 1000))
        actual = budget + overshoot_s
        if spike_every and i % spike_every == spike_every - 1:
            actual = max(actual, spike_s)
        bank -= actual
        min_bank = min(min_bank, bank)
        if bank <= 0:
            return min_bank  # flagged
        bank += inc_s
    return min_bank


def test_solvent_1plus1_survives_very_long_game():
    """146-ply (73-move/side) game at 1+1 with overshoot must NOT flag."""
    tm = TimeManager.from_env()  # defaults: solvent on
    assert tm.solvent
    min_bank = _simulate(tm, initial_s=60, inc_s=1, moves=90, overshoot_s=0.4)
    assert min_bank > 0, f"bank hit {min_bank:.2f}s — would flag at 1+1"


def test_solvent_1plus1_survives_isolated_heavy_fog_spikes():
    """Isolated 5.5s heavy-fog moves (|P|>10M floor) absorbed by the reserve."""
    tm = TimeManager.from_env()
    min_bank = _simulate(
        tm, initial_s=60, inc_s=1, moves=80, overshoot_s=0.3, spike_s=5.5, spike_every=25
    )
    assert min_bank > 0, f"bank hit {min_bank:.2f}s — spike flagged at 1+1"


def test_legacy_formula_flags_the_long_1plus1_game():
    """The old formula is why we're here: it drains the long game. Guards the A/B."""
    tm = TimeManager(solvent=False, divisor=30.0)
    min_bank = _simulate(tm, initial_s=60, inc_s=1, moves=90, overshoot_s=0.4)
    assert min_bank <= 0, "legacy expected to flag the long game (the regression)"


def test_solvent_bullet_self_limits_below_cap():
    """At 1+1 the budget never approaches the 5s cap — it self-limits."""
    tm = TimeManager.from_env()
    opening = tm.budget_for(60_000, 1_000)
    assert opening < 2.5, f"1+1 opening budget {opening:.2f}s too high"


def test_solvent_blitz_opening_uses_the_cap():
    """At 3+2 the opening still spends a full think (cap binds)."""
    tm = TimeManager.from_env()
    opening = tm.budget_for(180_000, 2_000)
    assert opening == tm.hard_cap_s


def test_budget_respects_min_think_and_cap_bounds():
    tm = TimeManager.from_env()
    for rem in (200_000, 60_000, 10_000, 1_000, 0):
        b = tm.budget_for(rem, 1_000)
        assert tm.min_think_s <= b <= tm.hard_cap_s


def test_no_clock_returns_static_fallback():
    tm = TimeManager.from_env()
    assert tm.budget_for(None, 0, static_fallback_s=5.0) == 5.0
    assert tm.budget_for(None, 0, static_fallback_s=None) is None
