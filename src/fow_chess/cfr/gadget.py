"""Resolve gadget decision logic (per-opponent-infoset follow/exit).

This is the correctness-critical core of the Resolve refinement, factored out
as a pure, engine-free component so it can be unit-tested in isolation before
being wired into the hot-path eq loop (see
``docs/engine/gadget-mvp-build-notes-2026-05-28.md`` Slice 1).

Construction (Obscuro Appendix B.2 + C.1-C.3, Figs 9-10). Each opponent root
infoset ``J`` — in our multi-root setting, each sampled belief world (a
singleton perfect-information opponent infoset, Fig 9 lines 11-14) — gets a
two-action choice solved by its own regret minimizer:

  * **follow** (enter the subgame): utility = the opponent's value in world J
    under the current re-solved strategy (opponent POV).
  * **exit** (take the gift): utility = ``gift(J) = blueprint.opp_cfv(J) -
    margin`` — a fixed alternate value.

The opponent follows iff following beats the gift (it can exploit us there),
which pressures our shared strategy to keep every world's opponent value at or
below the blueprint baseline — the per-opp-infoset worst-case constraint that
defeats aggregation-dilution (a uniform average over worlds does not, see the
build notes). Worlds the opponent exits contribute a constant gift (no gradient
on our strategy); worlds it follows pressure us, weighted by ``alpha(J) *
P(follow | J)``.

MVP scope (clearly bounded; refinements deferred):
  * Regret-matching+ (RM+), not the paper's predictive PRM+. RM+ converges to
    the same place; the predictive term is a Slice-3+ refinement.
  * Gift is whatever the blueprint returns (constant for StubBlueprint); the
    reach-gift term ``g_hat(J)`` (§C.2) is deferred to the real blueprint.
  * ``alpha`` defaults to uniform (StubBlueprint gives uniform opponent reach);
    the non-uniform ``alpha(J) = 1/2(y(J)/Sum y + 1/m)`` is Slice-4 work.
  * The continuous Maxmargin<->Resolve blend (Fig 10 p_max) is exposed only as
    :meth:`ResolveGadget.is_safe`; the mixing itself is Slice 3.
"""
from __future__ import annotations

from collections.abc import Sequence


def _rm_plus_follow_prob(regret_follow: float, regret_exit: float) -> float:
    """RM+ strategy probability of the *follow* action given the two
    (already nonneg-truncated) cumulative regrets. Uniform 0.5 when both are
    zero (no information yet)."""
    total = regret_follow + regret_exit
    if total <= 0.0:
        return 0.5
    return regret_follow / total


class ResolveGadget:
    """Per-world follow/exit RM+ gadget.

    One instance per ``choose_move``; ``n_worlds`` = number of sampled roots
    (opponent infosets). Each iteration the caller supplies the opponent-POV
    value of each world under the current strategy; the gadget updates its
    regrets and exposes per-world weights for the next eq pass.

    All values are OPPONENT POV (positive = good for the opponent), in
    ``[-1, 1]`` to match the leaf-eval value space.
    """

    def __init__(
        self,
        gifts: Sequence[float],
        *,
        margin: float = 0.0,
        alpha: Sequence[float] | None = None,
    ) -> None:
        n = len(gifts)
        if n == 0:
            raise ValueError("ResolveGadget needs at least one world")
        # Exit (gift) value per world: blueprint opp_cfv minus the safety margin.
        self._gift = [float(g) - float(margin) for g in gifts]
        self._regret_follow = [0.0] * n
        self._regret_exit = [0.0] * n
        if alpha is None:
            self._alpha = [1.0 / n] * n
        else:
            if len(alpha) != n:
                raise ValueError("alpha length must match gifts length")
            s = float(sum(alpha))
            if s <= 0.0:
                raise ValueError("alpha must have positive mass")
            self._alpha = [float(a) / s for a in alpha]
        self.n_worlds = n

    def follow_probs(self) -> list[float]:
        """Current RM+ probability the opponent *follows* (enters) each world."""
        return [
            _rm_plus_follow_prob(rf, re)
            for rf, re in zip(self._regret_follow, self._regret_exit, strict=False)
        ]

    def world_weights(self) -> list[float]:
        """Effective root distribution for the next eq pass: ``alpha(J) *
        P(follow | J)``. Worlds the opponent exits get ~0 weight (their gift is
        a constant that exerts no gradient on our strategy); worlds it follows
        keep their prior weight scaled by the follow probability. NOT
        renormalized — absolute scale is carried into the regret update by the
        caller (an all-exit profile correctly yields ~zero pressure)."""
        fp = self.follow_probs()
        return [a * p for a, p in zip(self._alpha, fp, strict=False)]

    def update(self, world_opp_values: Sequence[float]) -> None:
        """One RM+ step. ``world_opp_values[j]`` = the opponent's value in world
        ``j`` under the current strategy (opponent POV). For each world the two
        actions are follow (util = that value) and exit (util = gift)."""
        if len(world_opp_values) != self.n_worlds:
            raise ValueError(
                f"expected {self.n_worlds} world values, got {len(world_opp_values)}"
            )
        for j, v in enumerate(world_opp_values):
            u_follow = float(v)
            u_exit = self._gift[j]
            p_follow = _rm_plus_follow_prob(
                self._regret_follow[j], self._regret_exit[j]
            )
            node = p_follow * u_follow + (1.0 - p_follow) * u_exit
            # RM+: accumulate instantaneous regret, truncate at 0.
            self._regret_follow[j] = max(0.0, self._regret_follow[j] + (u_follow - node))
            self._regret_exit[j] = max(0.0, self._regret_exit[j] + (u_exit - node))

    def maxmargin_weights(self, world_opp_values) -> list[float]:
        """Maxmargin-mode pass weights: full weight on the adversary's best
        world — argmax over worlds of (follow value − gift) = the infoset where
        the opponent gains most (or loses least) by entering vs the blueprint.

        Obscuro's Maxmargin construction (Appendix B.2): "▼ first selects the
        infoset J" — the adversary picks; optimizing against the per-iteration
        adversarial pick is best-response dynamics for max-min margin. Used by
        the in-solve Maxmargin↔Resolve switch (C.1): when every world EXITS the
        Resolve gadget (is_safe), the Resolve world-weights α·P(follow) collapse
        to ~0 and the weighted eq pass carries no gradient — the solve goes
        DARK and the root strategy freezes at early noise (measured: the
        2026-06-11 H2H lost 12/13 games to 1-ply-fatal moves in believed-bad
        endgames; the |P|=1 game ran 5M gradient-free iterations). Switching to
        the worst-world weight keeps optimization pressure alive — "prevents
        the agent from being too pessimistic" per the paper, and here prevents
        it from being UNGOVERNED.
        """
        if len(world_opp_values) != self.n_worlds:
            raise ValueError(
                f"expected {self.n_worlds} world values, got {len(world_opp_values)}"
            )
        j_star = max(range(self.n_worlds),
                     key=lambda j: float(world_opp_values[j]) - self._gift[j])
        w = [0.0] * self.n_worlds
        w[j_star] = 1.0
        return w

    def is_safe(self, *, follow_threshold: float = 1e-3) -> bool:
        """True iff the opponent (effectively) always exits — i.e. following
        never beats the gift, so our strategy is provably safe (all margins
        nonneg). This is Obscuro's Maxmargin<->Resolve switch criterion (§C.1):
        when safe, the engine may mix (Maxmargin); otherwise it stays Resolve.
        The mixing itself is a later slice; this just exposes the predicate."""
        return all(p <= follow_threshold for p in self.follow_probs())
