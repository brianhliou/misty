"""Blueprint policies for the Resolve/Maxmargin gadget.

The Resolve gadget (Burch et al. 2014; Obscuro Appendix E) bounds re-solving
against a *blueprint* — a baseline policy + value for the opponent's
information sets. At each opponent infoset J the gadget offers the opponent a
choice between entering the re-solved subgame and taking a fixed "gift" value
derived from the blueprint: ``gift(J) = opp_cfv(J) - margin``. Solving the
gadget game guarantees the re-solved strategy never makes any opponent infoset
worse than the blueprint baseline — the safety property that defeats the
aggregation-dilution and deterministic-replay failures (see
``docs/engine/gadget-build-plan-2026-05-28.md`` and
``belief-retention-scoping-2026-05-28.md``).

This module defines the :class:`Blueprint` protocol plus :class:`StubBlueprint`
for verifying the gadget MECHANISM before a real (Stockfish / policy-net)
blueprint exists. The MVP gadget uses ``StubBlueprint`` with a constant
opponent CFV; if the stub gadget fixes the dilution diags at paper-faithful
``i=200``, the mechanism is correct and we graduate the blueprint
(gadget-build-plan Phase 3). The stub carries no positional knowledge, so it
will over-defend — that is expected, and it is exactly what the real blueprint
later corrects.

All values are from the OPPONENT's point of view (positive = good for the
opponent), clamped to ``[-1, 1]`` to match the leaf-eval value space.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import chess
import numpy as np


@runtime_checkable
class Blueprint(Protocol):
    """A baseline opponent policy + value, indexed by opponent infoset.

    ``opp_infoset_id`` identifies an opponent decision point in the search
    tree — the interned infoset id from ``EqEngine.node_infoset`` for the MVP,
    or a richer position-aware key once a real blueprint exists. The gadget is
    the only caller; this protocol is the seam between "how we bound
    re-solving" and "what baseline we bound against."
    """

    def opp_cfv(self, opp_infoset_id: int) -> float:
        """Blueprint counterfactual value of the opponent at this infoset, in
        ``[-1, 1]`` (opponent POV). The gadget's gift action pays
        ``opp_cfv - margin``."""
        ...

    def opp_strategy(
        self, opp_infoset_id: int, actions: Sequence[chess.Move]
    ) -> dict[chess.Move, float]:
        """Blueprint action distribution for the opponent at this infoset over
        ``actions``. Seeds the opponent's gadget strategy and the non-uniform
        root distribution ``alpha(J)``. Sums to 1 over ``actions`` (empty dict
        iff ``actions`` is empty)."""
        ...


class StubBlueprint:
    """Constant-CFV, uniform-policy blueprint for gadget MECHANISM
    verification (gadget-build-plan Phase 1).

    ``opp_cfv`` returns a fixed constant for every infoset and ``opp_strategy``
    is uniform. With a constant gift value the gadget reduces to "the opponent
    can guarantee itself ``opp_cfv`` at every infoset" — sufficient to verify
    the per-opp-infoset worst-case constraint actually fires (the dilution
    diags should jump toward ~100% defense). It encodes no positional
    knowledge, so it will over-defend; the real blueprint (Phase 3) fixes that
    via realistic ``opp_cfv`` + the gadget margin.
    """

    def __init__(self, opp_cfv: float = -0.1) -> None:
        # Default -0.1: a small negative baseline ("opponent expects to be
        # slightly worse than even"). The exact value is a Phase-1 tuning knob
        # the diag sweep settles; -0.1 is a neutral starting point well inside
        # [-1, 1] so the gift never dominates a genuine terminal (+/-1).
        self._cfv = float(opp_cfv)

    def opp_cfv(self, opp_infoset_id: int) -> float:
        return self._cfv

    def opp_strategy(
        self, opp_infoset_id: int, actions: Sequence[chess.Move]
    ) -> dict[chess.Move, float]:
        n = len(actions)
        if n == 0:
            return {}
        p = 1.0 / n
        return {a: p for a in actions}


class StockfishBlueprint:
    """Position-aware blueprint: ``opp_cfv(world)`` = Stockfish's evaluation of
    that world from the OPPONENT's POV.

    **NEGATIVE RESULT (2026-05-28): do not use this as-is.** A static depth-1
    Stockfish eval of the ROOT position is in a different *calibration* than the
    gadget's per-world ``value(a, j)`` (which is the CFR-solved value with the
    opponent best-responding through the tree). The static eval is
    systematically optimistic relative to the solved value, so every margin
    ``M(a,j) = value − bp_value`` skews negative → no action looks safe → the
    gadget drops to the Resolve regime → aggregation-dilution returns. On
    `a84dbaf9` the stub flips Qxc4 to the safe retreat, but this blueprint does
    not. A coherent blueprint must produce baselines in the SAME value space as
    the solve (e.g. the previous move's solved per-infoset values, or a learned
    value head) — not a free static eval. Kept as a flag option
    (FOW_RESOLVE_BLUEPRINT=stockfish) for the record; the StubBlueprint
    (``bp_value=0`` ≈ maximin over the belief) is the working MVP reference.
    ``opp_cfv`` takes a ``chess.Board``; the StubBlueprint ignores its argument.
    """

    def __init__(self, stockfish_eval, opponent_color: chess.Color) -> None:
        self._sf = stockfish_eval
        self._opp = opponent_color

    def opp_cfv(self, world: chess.Board) -> float:
        # Stockfish depth-1 eval in [-1, 1], opponent POV. The eval is cached by
        # EPD inside StockfishLeafEval, so per-world calls are cheap.
        return self._sf.evaluate(world, self._opp)

    def opp_strategy(
        self, opp_infoset_id: int, actions: Sequence[chess.Move]
    ) -> dict[chess.Move, float]:
        n = len(actions)
        if n == 0:
            return {}
        p = 1.0 / n
        return {a: p for a in actions}


class NetBlueprint:
    """Learned-value (CFV-proxy) blueprint: ``opp_cfv(world)`` = a value net's
    ``E[outcome]`` for the OPPONENT at ``world``, in ``[-1, 1]`` (tanh) — a baseline
    IN the solve's value range (the DeepStack/ReBeL route), unlike
    ``StockfishBlueprint``'s miscalibrated static centipawn eval.

    **v1 caveat (Obscuro-Parity Phase 1):** the net (``scripts/train_value_net.py``)
    is trained on self-play OUTCOME via MSE, so it is an *outcome-expectation* — an
    approximation of the CFR-solved counterfactual value, not the solved value
    itself. The Phase-1 question (see ``obscuro-parity-charter.md``) is whether even
    this DE-FATALIZES the gadget (the ply-197 repro no longer scores every move
    ≈ −1.1). A solve-target / carryover-CFV baseline is the Phase-2 calibration
    upgrade.

    Pure-numpy forward (768→256→256→1, ReLU, tanh), so no torch in the runtime.
    Weights = a ``.npz`` with ``fc{1,2,3}.{weight,bias}`` (the ``train_value_net``
    serving format); path via ``FOW_RESOLVE_NET_WEIGHTS``. ``opp_strategy`` is
    uniform for the MVP (same as the other blueprints).
    """

    _PIDX = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }

    def __init__(self, weights_path: str, opponent_color: chess.Color) -> None:
        w = np.load(weights_path)
        self._W1, self._b1 = w["fc1.weight"], w["fc1.bias"]
        self._W2, self._b2 = w["fc2.weight"], w["fc2.bias"]
        self._W3, self._b3 = w["fc3.weight"], w["fc3.bias"]
        self._opp = opponent_color
        self._buf = np.zeros(self._W1.shape[1], dtype=np.float32)
        self._cache: dict[str, float] = {}

    def opp_cfv(self, world: chess.Board) -> float:
        fen = world.fen()
        v = self._cache.get(fen)
        if v is None:
            buf = self._buf
            buf[:] = 0.0
            for sq, piece in world.piece_map().items():
                pi = self._PIDX[piece.piece_type]
                base = pi if piece.color == self._opp else 6 + pi
                buf[base * 64 + sq] = 1.0
            h1 = np.maximum(0.0, self._W1 @ buf + self._b1)
            h2 = np.maximum(0.0, self._W2 @ h1 + self._b2)
            v = float(np.tanh(self._W3 @ h2 + self._b3)[0])
            if len(self._cache) < 4096:
                self._cache[fen] = v
        return v

    def opp_strategy(
        self, opp_infoset_id: int, actions: Sequence[chess.Move]
    ) -> dict[chess.Move, float]:
        n = len(actions)
        if n == 0:
            return {}
        p = 1.0 / n
        return {a: p for a in actions}


class CarryoverBlueprint:
    """Continual-resolving blueprint (Obscuro-Parity Phase 2, Slices 0+1).

    ``opp_cfv(world)`` returns the opponent's solved counterfactual value
    ``u(x, y | J)`` at the opponent infoset ``J`` that *leads to* ``world``,
    carried from the PREVIOUS move's solve. This is the Obscuro-faithful
    blueprint: *"the blueprint strategy profile (x, y) is simply the saved
    strategy from the computation on the previous move"* (paper B.2; C.6.1a).
    No learned net, no static eval.

    **Why this is the right baseline (and the other two were not).** The carried
    value is computed by the SAME ``eq_eval`` machinery the gadget uses for its
    per-world ``value(a, j)`` (``EqEngine.root_child_values`` /
    ``current_strategy`` over the carried tree), so it lives in the *same value
    space* as the values the gadget caps. That calibration is exactly what
    ``StockfishBlueprint`` lacked — a static depth-1 centipawn eval is
    systematically optimistic relative to the CFR-solved value, so every margin
    ``M(a, j) = value − bp_value`` skewed negative, the gadget dropped to the
    Resolve regime, and aggregation-dilution returned. ``StubBlueprint``'s
    constant baseline carries no positional knowledge and over-defends. The
    carried ``u(x, y | J)`` fixes both: it is positional AND in-space.

    The engine populates the per-move map via :meth:`set_values` from
    ``engine_v2.choose_move`` (Slice 0), keyed by the resulting world's
    ``rules.board_fen`` — the same FEN normalization the subtree-carryover
    discovery matches on (``node_pos_to_fen`` == ``rules.board_fen`` for a given
    position). Values are in ``[-1, 1]`` (opponent POV) by construction (a convex
    combination of leaf/terminal values already in that range).

    **Dependency:** meaningful only with ``FOW_CARRYOVER_SUBTREE=1``. Without it
    the prior tree is wiped by ``reset_tree`` each move, the Slice-0 extraction
    finds nothing, and every world falls back to ``fallback`` — i.e. this
    degrades to a ``StubBlueprint`` (the correct, safe degradation, not a bug).
    The ``fallback`` also covers genuinely-uncarried worlds within a normal move:
    the first move of a game, and worlds reached via a move the prior search
    never explored.

    **Slice 2 (non-uniform ``alpha``, FOW_GADGET_ALPHA).** The carried opponent
    strategy ``y(J)`` — the prior solve's probability of the opponent move that
    *reaches* each world — is exposed via :meth:`set_reach` / :meth:`reach` and
    feeds the gadget's non-uniform root distribution
    ``alpha(J) = ½(y(J)/Σy + 1/m)`` (Obscuro B.2, the 53.3% ablation; gadget-mvp
    notes Slice 4 line). ``reach`` is a *probability* (no POV / sign flip) and
    defaults to ``0.0`` for uncarried worlds — those get only the ``1/m`` uniform
    floor, never a negative baseline. ``opp_strategy`` stays uniform (it seeds the
    opponent's in-gadget strategy, distinct from the root distribution ``alpha``).
    """

    def __init__(
        self,
        rules,
        fallback: float = -0.1,
        stockfish=None,
        opponent_color: chess.Color | None = None,
    ) -> None:
        self._rules = rules
        self._fallback = float(fallback)
        self._vals: dict[str, float] = {}
        self._reach: dict[str, float] = {}
        # FOW_GADGET_C1_FALLBACK (Obscuro Appendix C.1): the alternate value for
        # an UNCARRIED world is min{ṽ(h), v*} in OUR POV (ṽ = Stockfish's eval
        # of the world, v* = our previous solve's root value) -> opponent-POV
        # gift = max(sf_opp(h), -v*). The constant `fallback` makes the gadget
        # try to hold the opponent below -0.1 in EVERY uncovered world — usually
        # impossible -> permanent follow pressure -> over-defense (and free
        # exits / under-defense in worlds the opponent is genuinely losing).
        # A position-aware gift fixes both directions. Unlike StockfishBlueprint
        # (the 2026-05-28 negative result: static eval as the baseline for
        # CARRIED worlds is miscalibrated vs solved values), this fills only
        # worlds with NO solved value, and the -v* floor bounds the pessimism.
        # Default OFF -> constant fallback -> byte-identical.
        import os as _os
        self._c1 = (
            _os.environ.get("FOW_GADGET_C1_FALLBACK") == "1"
            and stockfish is not None
            and opponent_color is not None
        )
        self._sf = stockfish
        self._opp = opponent_color
        self._prev_value: float | None = None  # OUR-POV root value, prev solve

    def set_prev_value(self, value: float | None) -> None:
        """Record v* — OUR-POV root value of the just-finished solve — for the
        C.1 fallback's min{ṽ(h), v*} term on the NEXT move. None clears it
        (fresh game)."""
        self._prev_value = None if value is None else float(value)

    def set_values(self, vals: dict[str, float]) -> None:
        """Replace the per-move FEN -> ``u(x, y | J)`` map (opponent POV).

        Called once per move from ``engine_v2.choose_move`` before the solve,
        after the carried (x, y) has been read off the prior tree. Replaces
        rather than merges, so a stale value from two moves ago can never leak
        into the current solve."""
        self._vals = vals

    def set_reach(self, reach: dict[str, float]) -> None:
        """Replace the per-move FEN -> ``y(J)`` map (opponent reach probability).

        The carried opponent strategy's probability of the move that reaches each
        world, read off the prior tree alongside :meth:`set_values`. Feeds the
        gadget's non-uniform ``alpha(J)`` (Slice 2). Replace-not-merge, same as
        ``set_values`` — no stale reach leaks across moves."""
        self._reach = reach

    def opp_cfv(self, world: chess.Board) -> float:
        v = self._vals.get(self._rules.board_fen(world))
        if v is not None:
            return v
        if self._c1:
            # C.1 alternate value for an uncarried world (see __init__ comment):
            # opp-POV gift = max(sf_opp(h), -v*). Stockfish evals are EPD-cached
            # inside StockfishLeafEval, so per-world calls are cheap.
            sf_opp = self._sf.evaluate(world, self._opp)
            if self._prev_value is not None:
                return max(sf_opp, -self._prev_value)
            return sf_opp
        return self._fallback

    def reach(self, world: chess.Board) -> float:
        """Carried opponent reach ``y(J)`` for ``world`` (0.0 if uncarried)."""
        return self._reach.get(self._rules.board_fen(world), 0.0)

    def opp_strategy(
        self, opp_infoset_id: int, actions: Sequence[chess.Move]
    ) -> dict[chess.Move, float]:
        n = len(actions)
        if n == 0:
            return {}
        p = 1.0 / n
        return {a: p for a in actions}
