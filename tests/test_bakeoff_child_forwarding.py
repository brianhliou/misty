"""The per-game child subprocess must receive EVERY per-arm strength flag.

2026-06-10: _spawn_game's cmd never forwarded the faithful-stack per-arm flags
(--v2-gadget-iterative/-alpha/-faithful, --v2-resolve-blueprint,
--v2-carryover-subtree, --v2-structural-carry, opponent equivalents), so every
cloud probe's child fell back to env-reads (= OFF) and silently ran the
read-only stub gadget instead of the stack the ticket claimed to test. This
test constructs the orchestrator's child cmd with all flags set and asserts
each one survives the subprocess boundary.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_bakeoff_module():
    spec = importlib.util.spec_from_file_location(
        "run_v2_bakeoff", ROOT / "scripts" / "run_v2_bakeoff.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _full_args(**overrides) -> argparse.Namespace:
    """A base_args namespace with every field _spawn_game reads."""
    ns = argparse.Namespace(
        out_dir="/tmp/x", max_plies=300, v2_iters=100, v2_i=8,
        v2_time_budget=0.1, time_control="", v2_p_max=1000, v2_kluss_k=0,
        v2_max_actions=1, base_seed=0, stockfish="stockfish",
        opponent_mode="v2", opponent_kluss_k=0, opponent_max_actions=1,
        opponent_i=0, opponent_iters=0, opponent_time_budget=0.0,
        v2_expansion_budget=2000, opponent_expansion_budget=400,
        v2_use_rust_eq=False, opponent_use_rust_eq=False,
        v2_use_rust_state=True, opponent_use_rust_state=True,
        v2_use_rust_tree=True, opponent_use_rust_tree=True,
        v2_resolve_gadget=True, v2_cvar_q=0.1,
        v2_opening_book=False, opponent_opening_book=False,
        king_aware=True,
        v2_gadget_faithful=True, v2_gadget_alpha=True, v2_gadget_iterative=True,
        v2_carryover_subtree=True, v2_structural_carry=True, v2_lean_uci=True,
        v2_resolve_blueprint="carryover",
        opponent_resolve_gadget=True, opponent_gadget_faithful=True,
        opponent_gadget_alpha=True, opponent_gadget_iterative=True,
        opponent_carryover_subtree=True, opponent_structural_carry=True,
        opponent_lean_uci=True, opponent_resolve_blueprint="stub",
        v2_kluss_soft=True, opponent_kluss_soft=True,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _captured_cmd(mod, base_args):
    captured = {}

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = ""

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeProc()

    orig = mod.subprocess.run
    mod.subprocess.run = _fake_run
    try:
        mod._spawn_game(game_idx=0, base_args=base_args, timeout_s=1.0)
    finally:
        mod.subprocess.run = orig
    return captured["cmd"]


REQUIRED_FLAGS = [
    "--v2-resolve-gadget", "--king-aware",
    "--v2-gadget-faithful", "--v2-gadget-alpha", "--v2-gadget-iterative",
    "--v2-carryover-subtree", "--v2-structural-carry", "--v2-lean-uci",
    "--opponent-resolve-gadget", "--opponent-gadget-faithful",
    "--opponent-gadget-alpha", "--opponent-gadget-iterative",
    "--opponent-carryover-subtree", "--opponent-structural-carry",
    "--opponent-lean-uci",
    "--v2-kluss-soft", "--opponent-kluss-soft",
]


def test_child_cmd_carries_every_per_arm_flag():
    mod = _load_bakeoff_module()
    cmd = _captured_cmd(mod, _full_args())
    missing = [f for f in REQUIRED_FLAGS if f not in cmd]
    assert not missing, f"child cmd dropped per-arm flags: {missing}"
    # value-carrying flags
    assert "--v2-resolve-blueprint" in cmd
    assert cmd[cmd.index("--v2-resolve-blueprint") + 1] == "carryover"
    assert "--opponent-resolve-blueprint" in cmd
    assert cmd[cmd.index("--opponent-resolve-blueprint") + 1] == "stub"
    assert cmd[cmd.index("--v2-expansion-budget") + 1] == "2000"
    assert cmd[cmd.index("--opponent-expansion-budget") + 1] == "400"


def test_child_cmd_tri_state_none_omits_flags():
    """None (= let the child env-read) must not emit the flag at all."""
    mod = _load_bakeoff_module()
    cmd = _captured_cmd(mod, _full_args(
        v2_gadget_iterative=None, v2_gadget_alpha=None, v2_gadget_faithful=None,
        v2_resolve_blueprint=None, opponent_resolve_blueprint=None,
        v2_carryover_subtree=None, v2_structural_carry=None, v2_lean_uci=None,
        opponent_resolve_gadget=None, opponent_gadget_faithful=None,
        opponent_gadget_alpha=None, opponent_gadget_iterative=None,
        opponent_carryover_subtree=None, opponent_structural_carry=None,
        opponent_lean_uci=None,
    ))
    for f in ("--v2-gadget-iterative", "--v2-gadget-alpha",
              "--v2-resolve-blueprint", "--opponent-resolve-blueprint"):
        assert f not in cmd
