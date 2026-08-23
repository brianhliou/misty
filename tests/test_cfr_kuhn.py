"""CFR correctness gate: Kuhn poker.

Kuhn poker is the standard tabular-CFR validation game. 3-card deck (J, Q, K),
1 card each to two players, 1 card unseen. Ante 1 each, then a single round
of betting. Equilibrium has been worked out by hand; player 0's expected
value under equilibrium is -1/18 (player 0 is slightly disadvantaged).

This test exercises the imperfect-info parts of our CFR implementation
that the FoW smoke test does not — chance nodes, information set sharing
across different chance outcomes, and convergence to a known Nash value.
"""

from __future__ import annotations

from dataclasses import dataclass

from fow_chess.cfr.tabular import solve_subgame


# -- Kuhn poker fixture ------------------------------------------------------


@dataclass(frozen=True)
class KuhnDecisionNode:
    """A decision point in Kuhn poker.

    ``history`` is a tuple of action strings. Depth counts decisions only
    (chance node doesn't increment).
    """

    card_p0: int
    card_p1: int
    history: tuple
    depth: int

    is_chance: bool = False

    @property
    def to_move(self) -> int:
        # P0 acts at depth 0; P1 at depth 1; P0 at depth 2 (only check-bet path)
        return self.depth % 2

    @property
    def is_terminal(self) -> bool:
        h = self.history
        if h == ("check", "check"):
            return True
        if len(h) >= 2 and h[-1] in ("fold", "call"):
            return True
        return False

    def terminal_value(self, perspective: int) -> float:
        h = self.history
        # Determine pot size and winner
        if h == ("check", "check"):
            pot = 1
            winner = 0 if self.card_p0 > self.card_p1 else 1
        elif h[-1] == "fold":
            # Folder is the player who just acted = (depth - 1) % 2 of the
            # decision that produced this terminal. We tracked depth on
            # apply, so the folder is the player at depth - 1.
            folder = (self.depth - 1) % 2
            winner = 1 - folder
            pot = 1
        elif h[-1] == "call":
            pot = 2
            winner = 0 if self.card_p0 > self.card_p1 else 1
        else:
            raise ValueError(f"Not terminal: {h}")
        return float(pot) if winner == perspective else -float(pot)

    def legal_moves(self) -> list[str]:
        if self.is_terminal:
            return []
        h = self.history
        if not h:
            return ["check", "bet"]
        if h == ("check",):
            return ["check", "bet"]
        if h == ("bet",):
            return ["fold", "call"]
        if h == ("check", "bet"):
            return ["fold", "call"]
        return []

    def info_set_id(self):
        own_card = self.card_p0 if self.to_move == 0 else self.card_p1
        return (self.to_move, own_card, self.history)

    def apply(self, action: str) -> "KuhnDecisionNode":
        return KuhnDecisionNode(
            card_p0=self.card_p0,
            card_p1=self.card_p1,
            history=self.history + (action,),
            depth=self.depth + 1,
        )


@dataclass(frozen=True)
class KuhnRoot:
    """Chance node at the root: deal two distinct cards."""

    is_chance: bool = True
    is_terminal: bool = False
    depth: int = 0
    to_move: str = "chance"

    def chance_outcomes(self):
        outs = []
        for c0 in range(3):
            for c1 in range(3):
                if c0 != c1:
                    outs.append(((c0, c1), 1.0 / 6.0))
        return outs

    def apply(self, deal):
        c0, c1 = deal
        return KuhnDecisionNode(
            card_p0=c0,
            card_p1=c1,
            history=(),
            depth=0,
        )


# -- Test --------------------------------------------------------------------


def _leaf_eval_never_used(_truth, _perspective):
    raise AssertionError("Kuhn poker should never hit the depth bound")


def test_kuhn_poker_value_converges_to_neg_1_over_18():
    """CFR's value-at-root estimate should approach -1/18 for player 0."""
    root = KuhnRoot()
    solution = solve_subgame(
        root,
        depth=100,  # large; Kuhn never reaches it
        leaf_eval=_leaf_eval_never_used,
        iterations=5000,
        value_estimate_samples=5000,
        players=(0, 1),
    )
    target = -1.0 / 18.0
    assert abs(solution.value_at_root - target) < 0.05, (
        f"value_at_root={solution.value_at_root:.4f} target={target:.4f} "
        f"info_sets={solution.info_set_count}"
    )


def test_kuhn_info_set_count_is_12():
    """Sanity: Kuhn has exactly 12 information sets.

    P0 acts at depth 0 (3 info sets, one per card) and depth 2 (3 info sets).
    P1 acts at depth 1 (6 info sets: 3 cards × 2 histories: 'check' or 'bet').
    Total = 3 + 3 + 6 = 12.
    """
    root = KuhnRoot()
    solution = solve_subgame(
        root,
        depth=100,
        leaf_eval=_leaf_eval_never_used,
        iterations=200,  # don't need full convergence to count info sets
        value_estimate_samples=10,
        players=(0, 1),
    )
    assert solution.info_set_count == 12, (
        f"expected 12 info sets, got {solution.info_set_count}"
    )
