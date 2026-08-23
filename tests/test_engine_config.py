"""Tests for the resolved engine-config dump (drift detector) + profile seeding."""

from __future__ import annotations

import os

import pytest

import fow_chess.engine_config as ec


@pytest.fixture
def isolate_frozen_env():
    """Snapshot/clear/restore the frozen-toggle env vars so a test that calls
    apply_process_flags (which writes os.environ directly) can't leak."""
    keys = ("FOW_BOTTOMK_EXPANSION", "FOW_V2_CLOCK_TIME", "FOW_V2_EARLY_STOP")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k in keys:
        if saved[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = saved[k]


@pytest.fixture
def restore_leaf_globals():
    """apply_process_flags also mutates leaf-eval process globals — save/restore."""
    from fow_chess.cfr import leaf_eval as le

    saved = (le._KING_AWARE_LEAF, le._TANH_SCALE_CP, le._KING_BAND_FLOOR)
    yield
    le.set_king_aware_leaf(saved[0])
    le.set_tanh_scale_cp(saved[1])
    le.set_king_band_floor(saved[2])


# ---- the hash covers only the hand-set toggles -----------------------------


def test_toggle_values_cover_only_hand_set():
    vals = ec.toggle_values()
    assert {t.key for t in ec.TOGGLES} == set(vals)
    # the frozen (profile-owned) flags are NOT in the hand-set toggle set —
    # opening_book moved there once it became profile-owned (v1.3+).
    assert "bottomk_expansion" not in vals
    assert "opening_book" not in vals
    assert "opening_book" in {t.key for t in ec.FROZEN}


def test_hash_is_stable_and_toggle_sensitive(monkeypatch):
    monkeypatch.delenv("FOW_EQ_MERGED", raising=False)
    h_off = ec.config_hash()
    assert h_off == ec.config_hash()
    monkeypatch.setenv("FOW_EQ_MERGED", "1")
    assert ec.config_hash() != h_off


def test_lean_uci_declared_default_matches_code(monkeypatch):
    # The Toggle default must mirror _lean_uci_enabled()'s unset behavior —
    # the dump/hash once declared "0" while the code defaulted ON.
    from fow_chess.cfr.leaf_eval_stockfish import _lean_uci_enabled

    monkeypatch.delenv("FOW_LEAN_UCI", raising=False)
    declared = next(t.default for t in ec.TOGGLES if t.key == "lean_uci")
    assert (declared == "1") == _lean_uci_enabled()


def test_frozen_flags_are_not_in_the_hash(monkeypatch):
    # A frozen flag is profile-owned -> flipping its env must NOT move the hash
    # (it's shown in the table, but not part of the drift signal).
    monkeypatch.delenv("FOW_BOTTOMK_EXPANSION", raising=False)
    h = ec.config_hash()
    monkeypatch.setenv("FOW_BOTTOMK_EXPANSION", "1")
    assert ec.config_hash() == h


def test_hash_excludes_profile_knobs():
    from dataclasses import replace

    from fow_chess.engine_profile import STRONGEST

    swept = replace(STRONGEST, i_sample_size=64)
    assert ec.format_dump(STRONGEST).splitlines()[0] == ec.format_dump(swept).splitlines()[0]


def test_dump_sections(monkeypatch):
    monkeypatch.delenv("FOW_OPENING_BOOK", raising=False)
    full = ec.format_dump(include_profile=True)
    toggles_only = ec.format_dump(include_profile=False)
    assert "profile." in full and "profile." not in toggles_only
    # frozen + hand-set toggles appear in both
    for d in (full, toggles_only):
        assert "frozen.bottomk_expansion" in d
        assert "frozen.opening_book" in d
        assert "toggle.lean_uci" in d


# ---- the profile OWNS the frozen toggles (Phase 2) -------------------------


def test_profile_owns_frozen_toggles():
    from fow_chess.engine_profile import STRONGEST

    assert STRONGEST.bottomk_expansion is True
    assert STRONGEST.clock_time is True
    assert STRONGEST.early_stop is True


def test_apply_process_flags_seeds_frozen_when_unset(isolate_frozen_env, restore_leaf_globals):
    from fow_chess.engine_profile import STRONGEST

    # env unset (fixture) -> apply seeds them from the profile
    STRONGEST.apply_process_flags()
    assert os.environ["FOW_BOTTOMK_EXPANSION"] == "1"
    assert os.environ["FOW_V2_CLOCK_TIME"] == "1"
    assert os.environ["FOW_V2_EARLY_STOP"] == "1"


def test_apply_process_flags_does_not_override_explicit_env(isolate_frozen_env, restore_leaf_globals):
    from fow_chess.engine_profile import STRONGEST

    # an explicit env wins (setdefault) -> per-arm override + prod cutover is a no-op
    os.environ["FOW_BOTTOMK_EXPANSION"] = "0"
    STRONGEST.apply_process_flags()
    assert os.environ["FOW_BOTTOMK_EXPANSION"] == "0"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
