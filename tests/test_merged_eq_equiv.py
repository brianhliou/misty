"""Convergence-equivalence guard for the merged single-walk eq pass (FOW_EQ_MERGED,
commit f525ad9).

Merged is NOT byte-identical to the two-pass full-CFV eq (it's a different but
valid CFR update scheme — one walk updating both players' regrets instead of two).
So we can't assert bit-equality; what we CAN assert is that both converge to the
SAME equilibrium: at a fixed seed + iteration count, the root action-values must
agree within a small tolerance. A real bug in the merge (e.g. a wrong sign on the
black-to-move regret, or a mis-weighted node value) would shift values by O(1) and
trip this. Only active under full_cfv_backprop (the gadget regime), so we force
that via FOW_FULL_CFV_BACKPROP.
"""
from __future__ import annotations

import random
import shutil

import chess
import pytest

from fow_chess.engine_v2 import EngineV2
from fow_chess.observation import observation_from_transition

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not on PATH",
)


def _root_values(merged: bool, monkeypatch, seed: int = 42, warmup: int = 6,
                 iters: int = 1500, i_sample: int = 8) -> dict[str, float]:
    monkeypatch.setenv("FOW_FULL_CFV_BACKPROP", "1")  # merge only fires under full-CFV
    monkeypatch.setenv("FOW_EQ_MERGED", "1" if merged else "0")
    persp = chess.WHITE
    eng = EngineV2(persp, rng=random.Random(seed))
    rng = random.Random(seed)
    sim = chess.Board()
    for ply in range(warmup):
        legal = list(sim.pseudo_legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        prev = sim.copy()
        sim.push(mv)
        obs = observation_from_transition(prev, sim, persp)
        if ply % 2 == 0:
            eng.observe_own_move(mv, obs)
        else:
            eng.observe_opp_move(obs)
    eng.choose_move(iterations=iters, i_sample_size=i_sample, time_budget_seconds=None)
    av = {m.uci(): v for m, v in (eng.last_solution.action_values_at_root or {}).items()}
    eng.close()
    return av


def test_merged_eq_converges_like_two_pass(monkeypatch):
    two_pass = _root_values(merged=False, monkeypatch=monkeypatch)
    merged = _root_values(merged=True, monkeypatch=monkeypatch)
    common = set(two_pass) & set(merged)
    assert len(common) >= 2, "expected a non-trivial root with multiple actions"
    # Both schemes converge to the same equilibrium -> root values agree closely.
    # Tolerance is loose enough for the per-iterate scheme difference + the
    # half-as-many-walks-per-iter convergence gap, tight enough to catch an O(1)
    # bug (wrong sign / mis-weighting).
    worst = max(abs(two_pass[m] - merged[m]) for m in common)
    assert worst < 0.15, (
        f"merged eq diverged from two-pass by {worst:.3f} (>0.15) — likely a bug, "
        f"not the expected scheme difference.\ntwo-pass={two_pass}\nmerged={merged}"
    )
    # The argmax should agree (the move we'd actually play); both are well-converged.
    assert max(two_pass, key=two_pass.get) == max(merged, key=merged.get), (
        f"merged picks a different best move: two-pass={max(two_pass, key=two_pass.get)} "
        f"vs merged={max(merged, key=merged.get)}"
    )
