"""Guards for the single-source-of-truth engine profile (engine_profile.py).

The profile exists because the strength knobs were split across constructor
kwargs and process-global leaf flags, and consumers drifted (the live worker
had king-aware on; play_human did not — different engines). These tests lock:
  - the STRONGEST values (the source of truth),
  - that build_strategy() forwards them AND turns king-aware on,
  - the anti-drift invariant (deployment knobs don't change the strength config),
  - that the BARE default stays king-blind/gadget-off (so parity/reproducibility
    guards are unaffected — the profile is explicit opt-in),
  - the bakeoff's --profile bridge (_apply_profile) pins the v2 arm.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from fow_chess.cfr.leaf_eval import (
    king_aware_leaf_enabled,
    set_king_aware_leaf,
    set_tanh_scale_cp,
    tanh_scale_cp,
)
from fow_chess.engine_profile import DEFAULT_ITER_CAP, STRONGEST
from fow_chess.engine_v2 import EngineV2Strategy


@pytest.fixture(autouse=True)
def _restore_leaf_flags():
    """build_strategy / apply_process_flags mutate process-global leaf flags;
    save and restore them so these tests don't leak king-aware ON into the
    parity/reproducibility tests sharing this process."""
    ka, ts = king_aware_leaf_enabled(), tanh_scale_cp()
    try:
        yield
    finally:
        set_king_aware_leaf(ka)
        set_tanh_scale_cp(ts)


def test_strongest_values_are_the_campaign_verdict():
    assert STRONGEST.i_sample_size == 32
    assert STRONGEST.kluss_k == 2
    # Gadget OFF (2026-06-02 verdict): no head-to-head edge (47.5% vs OFF over 40
    # games) yet ~10-30x memory + ~2x game length; king_aware_leaf stays ON.
    assert STRONGEST.resolve_gadget is False
    assert STRONGEST.king_aware_leaf is True
    assert STRONGEST.tanh_scale_cp == 500.0


def test_apply_process_flags_enables_king_aware():
    set_king_aware_leaf(False)
    set_tanh_scale_cp(2000.0)
    STRONGEST.apply_process_flags()
    assert king_aware_leaf_enabled() is True
    assert tanh_scale_cp() == 500.0


def test_build_strategy_forwards_strength_knobs_and_sets_king_aware():
    set_king_aware_leaf(False)  # the very bug we're guarding against
    strat = STRONGEST.build_strategy(
        seed=1, time_budget_seconds=5.0, p_max_size=5_000_000
    )
    # king-aware is on now — building the strongest engine cannot be king-blind.
    assert king_aware_leaf_enabled() is True
    # strength knobs come from the profile...
    assert strat._i_sample_size == 32
    assert strat._kluss_k == 2
    assert strat._resolve_gadget is False  # gadget-off verdict (2026-06-02)
    assert strat._resolve_cvar_q == 0.1
    # ...deployment knobs are caller-supplied.
    assert strat._time_budget == 5.0
    assert strat._p_max_size == 5_000_000
    assert strat._iterations == DEFAULT_ITER_CAP


def test_deployment_knobs_do_not_change_strength_config():
    """The anti-drift invariant: same profile, different deployment knobs (as
    the worker @64M/5s and play_human @5M/budget legitimately differ) ->
    identical strength config. This is the whole point of the module."""
    a = STRONGEST.build_strategy(seed=1, time_budget_seconds=5.0, p_max_size=64_000_000)
    b = STRONGEST.build_strategy(seed=99, time_budget_seconds=0.5, p_max_size=5_000_000)
    for attr in ("_i_sample_size", "_kluss_k", "_resolve_gadget", "_resolve_cvar_q"):
        assert getattr(a, attr) == getattr(b, attr)


def test_replace_gadget_off_keeps_king_aware():
    """play_human's --no-gadget path varies ONE knob off the canonical profile
    via dataclasses.replace — it must still be king-aware."""
    prof = replace(STRONGEST, resolve_gadget=False)
    assert prof.resolve_gadget is False
    assert prof.king_aware_leaf is True
    assert prof.i_sample_size == 32


def test_bare_engine_default_stays_prior_behavior():
    """Guard: the profile is opt-in; the bare EngineV2Strategy default must stay
    king-blind + gadget-off so parity/reproducibility guards are unaffected and
    'default matches prior behavior' holds."""
    set_king_aware_leaf(False)
    bare = EngineV2Strategy(seed=0)
    assert king_aware_leaf_enabled() is False  # construction didn't flip it
    assert bare._resolve_gadget is None  # None = read env (off by default)
    assert bare._kluss_k is None


# --- bakeoff --profile bridge (scripts/run_v2_bakeoff.py:_apply_profile) ---

def _import_bakeoff():
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_v2_bakeoff  # script module, imported on demand

    return run_v2_bakeoff


def _bakeoff_args(**over) -> argparse.Namespace:
    base = dict(
        profile="none", v2_i=8, v2_kluss_k=0, v2_resolve_gadget=None,
        v2_cvar_q=None, v2_iters=500, king_aware=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_bakeoff_profile_strongest_pins_v2_arm():
    bakeoff = _import_bakeoff()
    args = _bakeoff_args(profile="strongest")
    bakeoff._apply_profile(args)
    assert args.v2_i == 32
    assert args.v2_kluss_k == 2
    assert args.v2_resolve_gadget is False  # gadget-off verdict (2026-06-02)
    assert args.v2_cvar_q == 0.1
    assert args.v2_iters >= DEFAULT_ITER_CAP
    assert args.king_aware is True  # defaulted on, since not explicitly set


def test_bakeoff_profile_strongest_respects_explicit_no_king_aware():
    """--no-king-aware must win over the profile so the king-aware A/B is
    possible (prod config vs the same config king-blind)."""
    bakeoff = _import_bakeoff()
    args = _bakeoff_args(profile="strongest", king_aware=False)
    bakeoff._apply_profile(args)
    assert args.v2_resolve_gadget is False  # gadget-off verdict; other knobs still applied
    assert args.king_aware is False  # explicit choice preserved


def test_bakeoff_v2_no_resolve_gadget_override_forces_gadget_off():
    """--v2-no-resolve-gadget forces the gadget OFF even if the profile (or an
    explicit --v2-resolve-gadget) would enable it — the gadget-isolation arm."""
    bakeoff = _import_bakeoff()
    args = _bakeoff_args(profile="strongest", v2_resolve_gadget=True,
                         v2_no_resolve_gadget=True)
    bakeoff._apply_profile(args)
    assert args.v2_resolve_gadget is False
    assert args.v2_kluss_k == 2  # rest of the profile still applied


def test_bakeoff_profile_none_is_noop():
    bakeoff = _import_bakeoff()
    args = _bakeoff_args(profile="none")
    bakeoff._apply_profile(args)
    assert args.v2_i == 8  # untouched
    assert args.v2_kluss_k == 0
    assert args.king_aware is None
