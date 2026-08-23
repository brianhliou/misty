"""Bottom-K (KMV) bounded belief expansion — Phase 1 of the
memory-bounded-expansion track.

Covers ``fow_rust.PEnumState.set_bottomk_cap`` + the cap-aware ``opp_move_core``:
  - uncapped (default) builds the full consistent set M (baseline),
  - a cap below M yields a uniform SUBSET of the true consistent set M,
  - the cap is respected (peak bounded) and the downsample is flagged,
  - a cap above M is identical to uncapped (no spurious downsample),
  - the bottom-K result is deterministic across runs (thread-timing independent),
  - the kept set is monotone in K (the KMV structural guarantee of uniformity),
  - a degenerate cap (k=0/1) does not panic.

The fixture self-plays a fixed random game (seed 2) through an uncapped
rust-state ``PEnumerator``, capturing the belief snapshot + observation of the
opp-move that produced the largest M (~12k from ~1.2k prev worlds). That gives a
genuinely large consistent set to exercise the bounded build — the realistic
shape, not the hard-filtered handful a single synthetic transition admits.

The *scale* property (peak RSS ∝ K at M~10^8) is validated by the Phase 1d
bakeoff on the recovered explosion games, not here.

Runs on the PRODUCTION PEnumState path (``--v2-use-rust-state``).
"""
from __future__ import annotations

import random

import chess
import pytest

pytest.importorskip("fow_rust")
import fow_rust

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import PEnumerator
from fow_chess.p_enum.enumerator import _obs_piece_bitmasks


def _capture_large_M(seed: int, steps: int):
    """Self-play a fixed random game; return (belief snapshot, observation) of
    the opp-move with the largest resulting consistent set M."""
    rng = random.Random(seed)
    board = chess.Board()
    pe = PEnumerator(chess.WHITE, use_rust_state=True)
    best = None
    for _ in range(steps):
        ms = list(board.legal_moves)
        if not ms:
            break
        mv = rng.choice(ms)
        prev = board.copy()
        board.push(mv)
        wtm_before = prev.turn == chess.WHITE
        obs = observation_from_transition(prev, board, chess.WHITE)
        try:
            if wtm_before:
                pe.update_own_move(mv, obs)
            else:
                before = sorted(pe.positions)  # snapshot BEFORE applying
                pe.update_opp_move(obs)
                if best is None or pe.size > best[2]:
                    best = (before, obs, pe.size)
        except RuntimeError:
            break
    assert best is not None and best[2] > 500, "fixture did not produce a large M"
    return best[0], best[1]


@pytest.fixture(scope="module")
def scenario():
    fens, obs = _capture_large_M(seed=2, steps=12)
    return fens, obs


def _run(fens, obs, cap):
    w, b = _obs_piece_bitmasks(obs)
    own = -1 if obs.own_capture_square is None else int(obs.own_capture_square)
    opp = -1 if obs.opp_capture_landing_square is None else int(obs.opp_capture_landing_square)
    ps = fow_rust.PEnumState(fens)
    if cap is not None:
        ps.set_bottomk_cap(cap)
    sz = ps.update_opp_move(
        False, True, int(obs.visibility_mask),
        w[0], w[1], w[2], w[3], w[4], w[5],
        b[0], b[1], b[2], b[3], b[4], b[5], own, opp,
    )
    return sz, set(ps.all_positions()), ps.last_pre_cap_count, ps.last_was_downsampled


def test_uncapped_is_baseline(scenario):
    fens, obs = scenario
    sz, m_set, pre, ds = _run(fens, obs, None)
    assert sz == len(m_set)
    assert ds is False
    assert pre == sz       # exact M, not an estimate
    assert sz > 2000       # the opp move expands the belief (test is meaningful)


def test_cap_below_M_is_uniform_subset(scenario):
    fens, obs = scenario
    _, m_set, _, _ = _run(fens, obs, None)
    cap = len(m_set) // 4
    sz, capped, pre_est, ds = _run(fens, obs, cap)
    assert len(m_set) > cap, "test invalid: M not above cap"
    assert capped <= m_set, "kept a world not in the true consistent set M"
    assert sz <= cap + 5, f"cap not respected: {sz} > {cap}"  # +slack for hash ties
    assert ds is True
    # KMV cardinality estimate should be in the right ballpark for M
    assert 0.5 * len(m_set) <= pre_est <= 2.0 * len(m_set)


def test_cap_above_M_identical_to_uncapped(scenario):
    fens, obs = scenario
    _, m_set, _, _ = _run(fens, obs, None)
    _, capped, _, ds = _run(fens, obs, len(m_set) + 100)
    assert capped == m_set, "cap above M must not change the set"
    assert ds is False, "cap above M must not flag a downsample"


def test_bottomk_deterministic(scenario):
    fens, obs = scenario
    _, m_set, _, _ = _run(fens, obs, None)
    cap = len(m_set) // 4
    _, a, _, _ = _run(fens, obs, cap)
    _, b, _, _ = _run(fens, obs, cap)
    assert a == b, "bottom-K result must be deterministic across runs"


def test_bottomk_monotone_in_k(scenario):
    """Nested caps: the smaller-K set must be a subset of the larger-K set.
    This is the KMV structural guarantee that makes the sample uniform."""
    fens, obs = scenario
    _, m_set, _, _ = _run(fens, obs, None)
    sz_small, s_small, _, _ = _run(fens, obs, len(m_set) // 8)
    sz_big, s_big, _, _ = _run(fens, obs, len(m_set) // 2)
    assert s_small <= s_big, "bottom-K not monotone in K (breaks KMV uniformity)"
    assert sz_small < sz_big <= len(m_set)


def test_degenerate_cap_does_not_panic(scenario):
    """k=0 / k=1 must not panic (select_nth underflow guard)."""
    fens, obs = scenario
    for cap in (0, 1):
        sz, _kept, _, ds = _run(fens, obs, cap)
        assert sz <= max(cap, 1) + 1
        assert ds is True
