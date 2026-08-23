"""CANDIDATE profile: the faithful stack exists by NAME, fully, or not at all."""
import os

import pytest

from fow_chess.engine_profile import CANDIDATE, PROFILES, STRONGEST


def test_profiles_registry():
    assert PROFILES["strongest"] is STRONGEST
    assert PROFILES["candidate"] is CANDIDATE


def test_candidate_owns_the_full_faithful_stack():
    assert CANDIDATE.resolve_gadget and CANDIDATE.gadget_iterative
    assert CANDIDATE.gadget_alpha and CANDIDATE.resolve_blueprint == "carryover"
    assert CANDIDATE.carryover_subtree and CANDIDATE.structural_carry
    assert CANDIDATE.gadget_merged and CANDIDATE.gadget_wexp
    assert CANDIDATE.c1_fallback and CANDIDATE.king_aware_leaf
    assert CANDIDATE.expansion_budget == 2000 and CANDIDATE.i_sample_size == 200


def test_candidate_apply_process_flags_seeds_envs(monkeypatch):
    for k in ("FOW_GADGET_MERGED", "FOW_GADGET_WEXP", "FOW_GADGET_WEXP_MIX",
              "FOW_GADGET_C1_FALLBACK", "FOW_GADGET_ITER_INTERVAL",
              "FOW_V2_EXPANSION_BUDGET", "FOW_BOTTOMK_EXPANSION",
              "FOW_V2_CLOCK_TIME", "FOW_V2_EARLY_STOP"):
        monkeypatch.delenv(k, raising=False)
    # save/restore the process-global leaf flags around the mutation
    from fow_chess.cfr.leaf_eval import king_aware_leaf_enabled, set_king_aware_leaf
    prev = king_aware_leaf_enabled()
    try:
        CANDIDATE.apply_process_flags()
        assert os.environ["FOW_GADGET_MERGED"] == "1"
        assert os.environ["FOW_GADGET_WEXP"] == "1"
        assert os.environ["FOW_GADGET_WEXP_MIX"] == "0.0"
        assert os.environ["FOW_GADGET_C1_FALLBACK"] == "1"
        assert os.environ["FOW_GADGET_ITER_INTERVAL"] == "1"
        # expansion_budget is ARM-scoped (constructor kwarg), deliberately NOT
        # env-seeded — the env would leak to a different-i opponent arm.
        assert "FOW_V2_EXPANSION_BUDGET" not in os.environ
        assert king_aware_leaf_enabled()
    finally:
        set_king_aware_leaf(prev)


def test_candidate_apply_respects_explicit_env(monkeypatch):
    monkeypatch.setenv("FOW_GADGET_WEXP_MIX", "0.25")
    from fow_chess.cfr.leaf_eval import king_aware_leaf_enabled, set_king_aware_leaf
    prev = king_aware_leaf_enabled()
    try:
        CANDIDATE.apply_process_flags()
        assert os.environ["FOW_GADGET_WEXP_MIX"] == "0.25"  # setdefault: env wins
    finally:
        set_king_aware_leaf(prev)


def test_strongest_unchanged_by_new_fields():
    assert STRONGEST.gadget_iterative is False
    assert STRONGEST.resolve_blueprint is None
    assert STRONGEST.expansion_budget == 0


def test_bakeoff_profile_choices_track_the_registry():
    """The --profile choices must derive from PROFILES — a hand-copied tuple
    rejected candidate-i32 with rc=2 and the runner abandoned the ticket
    (2026-06-12). If this fails, someone re-hardcoded the list."""
    import argparse
    import importlib.util
    from pathlib import Path
    from fow_chess.engine_profile import PROFILES

    spec = importlib.util.spec_from_file_location(
        "run_v2_bakeoff", Path(__file__).parent.parent / "scripts" / "run_v2_bakeoff.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ap = mod.build_arg_parser() if hasattr(mod, "build_arg_parser") else None
    if ap is None:
        import re
        srctext = (Path(__file__).parent.parent / "scripts" / "run_v2_bakeoff.py").read_text()
        assert 'choices=("none", *_PROFILES)' in srctext
        return
    action = next(a for a in ap._actions if "--profile" in a.option_strings)
    assert set(action.choices) == {"none", *PROFILES}
