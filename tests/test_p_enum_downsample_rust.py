"""Unit coverage for the PRODUCTION downsample path: the Rust
``PEnumState.downsample`` (seeded partial Fisher-Yates).

The existing cap tests in ``test_p_enum_unit.py`` only exercise the Python
``_maybe_downsample`` (set-of-strings) fallback. The path that actually serves
under ``--v2-use-rust-state`` is ``PEnumState.downsample(max_size, mt_words,
mt_index)`` — previously exercised only in cloud bakeoffs, never unit-tested.
This guards size-bound, no-op, the ``last_was_downsampled`` flag, seeded
reproducibility, and statistical uniformity (the property soundness relies on:
a uniform sample can't systematically evict the truth-bearing world).
"""
from __future__ import annotations

import random

import chess
import pytest

pytest.importorskip("fow_rust")
import fow_rust


def _distinct_fens(n: int, seed: int) -> list[str]:
    """n distinct, valid standard-chess FENs via short random playouts."""
    rng = random.Random(seed)
    fens: set[str] = set()
    guard = 0
    while len(fens) < n and guard < n * 50:
        guard += 1
        b = chess.Board()
        for _ in range(rng.randint(1, 24)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        fens.add(b.fen())
    assert len(fens) == n, f"only generated {len(fens)}/{n} distinct fens"
    return list(fens)


def _mt_state(seed: int) -> tuple[list[int], int]:
    """Mirror PEnumerator._rust_downsample: extract CPython MT words+index
    from a seeded random.Random so the Rust eviction is reproducible."""
    r = random.Random(seed)
    st = r.getstate()[1]  # 625-tuple: 624 MT words + index
    return list(st[:624]), st[624]


def _state(fens: list[str]) -> "fow_rust.PEnumState":
    return fow_rust.PEnumState(fens)


def test_downsample_caps_size_and_sets_flag():
    ps = _state(_distinct_fens(200, seed=1))
    assert ps.size() == 200
    mt_words, mt_index = _mt_state(42)
    ps.downsample(50, mt_words, mt_index)
    assert ps.size() == 50
    assert ps.last_was_downsampled is True
    # every survivor came from the input
    survivors = set(ps.all_positions())
    assert survivors.issubset(set(_distinct_fens(200, seed=1)))


def test_downsample_noop_when_under_cap():
    ps = _state(_distinct_fens(80, seed=2))
    before = set(ps.all_positions())
    mt_words, mt_index = _mt_state(7)
    ps.downsample(500, mt_words, mt_index)
    assert ps.size() == 80
    assert ps.last_was_downsampled is False
    assert set(ps.all_positions()) == before


def test_downsample_reproducible_same_seed():
    fens = _distinct_fens(300, seed=3)
    a, b = _state(fens), _state(fens)
    mw, mi = _mt_state(99)
    a.downsample(64, mw, mi)
    mw, mi = _mt_state(99)
    b.downsample(64, mw, mi)
    assert sorted(a.all_positions()) == sorted(b.all_positions())


def test_downsample_different_seed_changes_sample():
    fens = _distinct_fens(300, seed=4)
    a, b = _state(fens), _state(fens)
    mw, mi = _mt_state(1)
    a.downsample(64, mw, mi)
    mw, mi = _mt_state(2)
    b.downsample(64, mw, mi)
    # Overwhelmingly likely to differ for a uniform 64-of-300 draw; a tie would
    # signal the seed isn't actually steering eviction.
    assert sorted(a.all_positions()) != sorted(b.all_positions())


def test_downsample_is_statistically_uniform():
    """No world is systematically kept or evicted — the soundness-critical
    property (uniform => truth survives with probability K/N, not biased away)."""
    fens = _distinct_fens(200, seed=5)
    keep, trials = 50, 300
    counts = {f: 0 for f in fens}
    for t in range(trials):
        ps = _state(fens)
        mw, mi = _mt_state(10_000 + t)
        ps.downsample(keep, mw, mi)
        for f in ps.all_positions():
            counts[f] += 1
    rates = [c / trials for c in counts.values()]
    mean = sum(rates) / len(rates)
    assert keep / len(fens) - 0.01 <= mean <= keep / len(fens) + 0.01  # ~0.25
    # no world starved or forced (loose, non-flaky bounds)
    assert min(rates) > 0.05
    assert max(rates) < 0.55
