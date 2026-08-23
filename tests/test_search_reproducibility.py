"""Same-seed search reproducibility — the regression guard for the belief-sampling
determinism fix (commit 62f94b9).

Before that fix, ``PEnumState.positions`` was collected from a PARALLEL hash-set
build (rayon reduce / DashSet), so its Vec order was non-reproducible across runs.
The seeded ``rng.sample(range(sz), k)`` root sampling drew reproducible *indices*,
but ``get_by_index`` then mapped them onto DIFFERENT worlds each run → a different
move at the SAME seed (the bug was silent — no test guarded it; this is that test).

We assert: same seed + same FIXED iteration count (no time budget, so it's fully
deterministic) → identical move AND identical action-values, with |P|>1 so the
root-sampling path (where the bug lived) is actually exercised.
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


def _run_once(seed: int, warmup: int, iters: int, i_sample: int):
    """One full belief-build + pick_move at a fixed iteration count. Returns
    (move_uci, sorted action-values tuple, |P|)."""
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
    move = eng.choose_move(iterations=iters, i_sample_size=i_sample,
                           time_budget_seconds=None)
    sol = eng.last_solution
    av = tuple(sorted((m.uci(), v) for m, v in (sol.action_values_at_root or {}).items()))
    psize = eng.enumerator.size
    eng.close()
    return move.uci(), av, psize


@pytest.mark.parametrize("seed", [42, 7])
def test_same_seed_reproducible(seed: int):
    m1, av1, p1 = _run_once(seed, warmup=8, iters=400, i_sample=16)
    m2, av2, p2 = _run_once(seed, warmup=8, iters=400, i_sample=16)
    # The bug only manifests when root sampling actually chooses among >1 world.
    assert p1 == p2 and p1 > 1, f"warmup must grow |P|>1 to exercise sampling (got {p1}, {p2})"
    assert m1 == m2, f"same seed picked DIFFERENT moves ({m1} vs {m2}) — sampling not reproducible"
    assert av1 == av2, "same seed produced different action-values — search not reproducible"
