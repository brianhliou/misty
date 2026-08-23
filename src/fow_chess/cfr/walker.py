"""Depth-bounded subgame walker for CFR over Fog of War chess.

The walker does not carry a full belief state through nodes. Belief is
implicit in each player's observation history: tabular CFR (and the
forthcoming Obscuro-style replication) identifies information sets by
``(to_move, observation_history)``.

Pure tree-walking + observation-history tracking. No belief-state
machinery, no factored marginals, no neural-net feature encoding —
those were Phase 2 substrate, removed in Phase A0 of the Obscuro
replication.

Mechanics-correctness contract: see ``lab/diag/cfr-walker-test-plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import chess

from ..observation import (
    Observation,
    observation_from_transition,
    observation_from_transition_both,
)

try:
    import fow_rust as _fow_rust
    _HAS_RUST_KEYS = hasattr(_fow_rust, "obs_keys_both_bb")
except ImportError:
    _HAS_RUST_KEYS = False


def _key_from_components(
    visibility: int,
    piece_masks: list,
    own_capture_square,
    opp_capture_landing_square,
    game_over_winner,
    game_over_reason: str,
) -> tuple:
    """Assemble an _obs_key tuple from the raw components Rust returns. MUST
    match `_obs_key` exactly: visibility int, 12-tuple of per-(color,type)
    piece bitmasks, the two capture squares, then winner|None and reason|None
    (empty reason => no game_over => None)."""
    return (
        visibility,
        tuple(piece_masks),
        own_capture_square,
        opp_capture_landing_square,
        game_over_winner if game_over_reason else None,
        game_over_reason or None,
    )


def obs_keys_both(
    prev_board: chess.Board, next_board: chess.Board
) -> tuple[tuple, tuple]:
    """Both perspectives' infoset keys ``(white_key, black_key)`` for a
    transition, built DIRECTLY from Rust components — skips constructing the
    Python ``Observation`` (chess.Piece dict + SquareSet) on the hot expand_leaf
    path. Each key is identical to ``_obs_key(observation_from_transition(...))``.
    """
    if _HAS_RUST_KEYS:
        next_ep_idx = next_board.ep_square if next_board.ep_square is not None else 64
        white_c, black_c = _fow_rust.obs_keys_both_bb(
            prev_board.occupied_co[chess.WHITE],
            prev_board.occupied_co[chess.BLACK],
            prev_board.kings,
            next_board.pawns, next_board.knights, next_board.bishops,
            next_board.rooks, next_board.queens, next_board.kings,
            next_board.occupied_co[chess.WHITE],
            next_board.occupied_co[chess.BLACK],
            next_board.castling_rights,
            next_ep_idx,
        )
        return _key_from_components(*white_c), _key_from_components(*black_c)
    ow, ob = observation_from_transition_both(prev_board, next_board)
    return _obs_key(ow), _obs_key(ob)


def _obs_key(obs: Observation) -> tuple:
    """Hashable canonical key for an Observation.

    Two Observations that compare equal as `Observation` instances must
    produce equal keys. We sort visible-piece entries and the visibility
    mask so dict-ordering is irrelevant.
    """
    # Canonicalize visible pieces as 12 per-(color, piece-type) bitmasks (white
    # P..K, then black P..K). A square holds at most one piece, so this is a
    # bijection with the {square: piece} dict — same equivalence classes as the
    # old sorted-tuple of (sq, color, type), but built with O(n) bit-ORs instead
    # of an O(n log n) sort + tuple-of-tuples on every node.
    pm = [0] * 12
    for sq, p in obs.visible_pieces.items():
        pm[(p.piece_type - 1) + (0 if p.color else 6)] |= 1 << sq
    visible_pieces = tuple(pm)
    # The visibility mask is a bitset: its int value is already a canonical,
    # injective key for the set of visible squares (same squares <=> same int).
    # Using it directly avoids iterating + sorting + tuple-building the squares
    # on every infoset-key construction (the hot path — _obs_key was ~14% of a
    # pick_move via two sorted() calls per node). Equivalence classes are
    # identical to the old sorted-tuple, so infoset identity / strategy / move
    # are unchanged.
    visibility = int(obs.visibility_mask)
    return (
        visibility,
        visible_pieces,
        obs.own_capture_square,
        obs.opp_capture_landing_square,
        obs.game_over.winner if obs.game_over else None,
        obs.game_over.reason if obs.game_over else None,
    )


@dataclass(frozen=True)
class SubgameNode:
    """One node in a CFR subgame tree.

    Holds the truth board (used to enumerate legal actions and detect
    terminals) and each player's observation history from the subgame
    root. The to-move player's history identifies its information set
    for CFR regret-table lookup.

    Frozen by design: ``apply`` returns a new node rather than mutating,
    so sibling branches stay independent.
    """

    truth: chess.Board
    to_move: chess.Color
    obs_history_white: tuple
    obs_history_black: tuple
    depth: int

    @classmethod
    def root(
        cls,
        truth: chess.Board,
        to_move: chess.Color | None = None,
    ) -> "SubgameNode":
        """Construct the root node from a known truth board.

        If ``to_move`` is None, falls back to the board's own turn field.
        Observation histories start empty — info-set IDs in the subgame
        are unique within the subgame so long as histories diverge as
        the tree branches.
        """
        if to_move is None:
            to_move = truth.turn
        return cls(
            truth=truth.copy(),
            to_move=to_move,
            obs_history_white=(),
            obs_history_black=(),
            depth=0,
        )

    @property
    def is_terminal(self) -> bool:
        """True when either king has been captured."""
        return (
            self.truth.king(chess.WHITE) is None
            or self.truth.king(chess.BLACK) is None
        )

    def legal_moves(self) -> list[chess.Move]:
        """FoW legal moves for the to-move player.

        FoW has no check restriction, so legal == pseudo-legal.
        """
        if self.is_terminal:
            return []
        return list(self.truth.pseudo_legal_moves)

    def info_set_id(self) -> Hashable:
        """Identifier for the to-move player's information set.

        Same observation history → same info-set ID, regardless of truth.
        This is what CFR uses to look up regret tables.
        """
        history = (
            self.obs_history_white
            if self.to_move == chess.WHITE
            else self.obs_history_black
        )
        return (self.to_move, history)

    def apply(self, move: chess.Move) -> "SubgameNode":
        """Apply ``move`` (played by ``self.to_move``) and return the next node.

        Both players' observation histories extend with what they each
        would observe of the transition. The next node's to_move flips.
        """
        next_truth = self.truth.copy()
        next_truth.push(move)
        obs_for_white = observation_from_transition(
            self.truth, next_truth, chess.WHITE
        )
        obs_for_black = observation_from_transition(
            self.truth, next_truth, chess.BLACK
        )
        return SubgameNode(
            truth=next_truth,
            to_move=not self.to_move,
            obs_history_white=(*self.obs_history_white, _obs_key(obs_for_white)),
            obs_history_black=(*self.obs_history_black, _obs_key(obs_for_black)),
            depth=self.depth + 1,
        )

    def terminal_value(self, perspective: chess.Color) -> float:
        """Value at a terminal node from ``perspective``'s POV in [-1, 1].

        +1 if ``perspective`` won (opp king captured), -1 if lost, 0 if
        both kings captured (degenerate). Only meaningful when
        ``is_terminal``.
        """
        own_king = self.truth.king(perspective)
        opp_king = self.truth.king(not perspective)
        if own_king is None and opp_king is None:
            return 0.0
        if own_king is None:
            return -1.0
        if opp_king is None:
            return 1.0
        return 0.0
