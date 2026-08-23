"""Unit tests for the Resolve gadget decision logic (cfr/gadget.py).

Pure, engine-free: validates the per-world follow/exit RM+ mechanism against
hand-computed expectations before it is wired into the eq loop.
"""
from __future__ import annotations

import pytest

from fow_chess.cfr.gadget import ResolveGadget


def _run(gadget: ResolveGadget, world_values, iters: int = 200) -> None:
    for _ in range(iters):
        gadget.update(world_values)


def test_construction_errors() -> None:
    with pytest.raises(ValueError):
        ResolveGadget([])
    with pytest.raises(ValueError):
        ResolveGadget([0.0, 0.0], alpha=[1.0])  # length mismatch
    with pytest.raises(ValueError):
        ResolveGadget([0.0], alpha=[0.0])  # zero mass


def test_initial_follow_prob_is_uniform() -> None:
    g = ResolveGadget([0.0, 0.0])
    assert g.follow_probs() == [0.5, 0.5]


def test_one_step_matches_hand_computation() -> None:
    # World: follow util +0.5, gift -0.1. p_follow=0.5 initially.
    # node = 0.5*0.5 + 0.5*(-0.1) = 0.20; r_follow=+0.30, r_exit=-0.30->0.
    g = ResolveGadget([-0.1])  # gift = -0.1 (margin 0)
    g.update([0.5])
    # After one step regret_follow=0.30, regret_exit=0 -> p_follow=1.0
    assert g.follow_probs()[0] == pytest.approx(1.0)


def test_exploitable_world_opponent_follows() -> None:
    # Following (+0.5) strictly beats the gift (-0.1): opponent enters.
    g = ResolveGadget([-0.1])
    _run(g, [0.5])
    assert g.follow_probs()[0] == pytest.approx(1.0, abs=1e-9)
    assert not g.is_safe()
    # World keeps ~full prior weight (alpha=1.0 for a single world).
    assert g.world_weights()[0] == pytest.approx(1.0, abs=1e-9)


def test_safe_world_opponent_exits() -> None:
    # Following (-0.5) is worse than the gift (+0.1): opponent takes the gift.
    g = ResolveGadget([0.1])
    _run(g, [-0.5])
    assert g.follow_probs()[0] == pytest.approx(0.0, abs=1e-9)
    assert g.is_safe()
    # An exited world exerts ~no pressure on our strategy.
    assert g.world_weights()[0] == pytest.approx(0.0, abs=1e-9)


def test_margin_raises_the_bar_to_exit() -> None:
    # follow=+0.05. With gift=opp_cfv(0.0)-margin(0.2) = -0.2, follow wins -> follow.
    g = ResolveGadget([0.0], margin=0.2)
    _run(g, [0.05])
    assert g.follow_probs()[0] == pytest.approx(1.0, abs=1e-9)
    # A negative margin (more generous gift) flips it: gift = 0.0-(-0.2)=+0.2 > 0.05.
    g2 = ResolveGadget([0.0], margin=-0.2)
    _run(g2, [0.05])
    assert g2.follow_probs()[0] == pytest.approx(0.0, abs=1e-9)
    assert g2.is_safe()


def test_mixed_worlds_not_safe_and_weights_track_follow() -> None:
    # World 0 exploitable (follow +0.6 > gift -0.1); world 1 safe (follow -0.6).
    g = ResolveGadget([-0.1, -0.1], alpha=[0.5, 0.5])
    _run(g, [0.6, -0.6])
    fp = g.follow_probs()
    assert fp[0] == pytest.approx(1.0, abs=1e-9)
    assert fp[1] == pytest.approx(0.0, abs=1e-9)
    assert not g.is_safe()  # at least one world is followed
    w = g.world_weights()
    assert w[0] == pytest.approx(0.5, abs=1e-9)  # alpha 0.5 * follow 1.0
    assert w[1] == pytest.approx(0.0, abs=1e-9)  # exited


def test_alpha_is_normalized() -> None:
    g = ResolveGadget([0.0, 0.0, 0.0], alpha=[2.0, 1.0, 1.0])
    # All exploitable so follow=1 everywhere; weights == normalized alpha.
    _run(g, [1.0, 1.0, 1.0])
    w = g.world_weights()
    assert w == pytest.approx([0.5, 0.25, 0.25], abs=1e-9)


def test_breakeven_world_is_treated_as_safe() -> None:
    # follow == gift exactly: no incentive to follow; RM+ leaves it at exit-ish.
    g = ResolveGadget([0.0])
    _run(g, [0.0])
    # Neither action accrues positive regret -> stays uniform; is_safe with the
    # default threshold should be False (0.5 > 1e-3), documenting the tie.
    assert g.follow_probs()[0] == pytest.approx(0.5, abs=1e-9)
    assert not g.is_safe()


def test_maxmargin_weights_pick_the_adversarys_best_world() -> None:
    # gifts: world0 expects -0.5, world1 expects +0.2; opponent values 0.1 / 0.1
    # -> world0 gains 0.6 by following (vs gift), world1 loses 0.1 -> pick 0.
    g = ResolveGadget([-0.5, 0.2])
    w = g.maxmargin_weights([0.1, 0.1])
    assert w == [1.0, 0.0]
    # flip: world1 becomes the best entry point
    w = g.maxmargin_weights([-0.9, 0.5])
    assert w == [0.0, 1.0]


def test_maxmargin_weights_length_check() -> None:
    g = ResolveGadget([0.0, 0.0])
    with pytest.raises(ValueError):
        g.maxmargin_weights([0.0])
