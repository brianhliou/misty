"""Particle-filter belief state for fog of war chess."""

from __future__ import annotations

from collections import defaultdict
import math
import random
import time
from dataclasses import dataclass, field, replace

import chess

from .constraints import (
    STANDARD_OPP_COUNTS as _STANDARD_OPP_COUNTS,
    opp_piece_counts as _opp_piece_counts,
    is_light_square as _is_light_square,
    opp_bishop_color_counts as _opp_bishop_color_counts,
)
from .move_priors import OpponentMovePrior
from .observation import Observation, consistent_with
from .visibility import piece_map_for_squares, visible_squares


@dataclass(frozen=True)
class RepairDiagnostics:
    """Observability for continuity repair quality.

    Repair is allowed to break strict legal-path lineage, so the engine needs
    to know when it did something expensive or implausible. These metrics do
    not change behavior yet; they make the next tuning target visible.
    """

    cost: int
    moved_piece_count: int
    max_piece_distance: int
    long_move_count: int
    teleport_like_count: int
    forced_visible_square_count: int
    unpaired_added_count: int
    unpaired_removed_count: int
    worst_piece: str | None = None
    worst_from: str | None = None
    worst_to: str | None = None
    worst_distance: int = 0
    worst_one_move_legal: bool | None = None
    strict_unreachable_count: int = 0


def _piece_fact_name(piece: chess.Piece) -> str:
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}-{chess.piece_name(piece.piece_type)}"


def _violates_count_constraint(
    counts: dict[chess.PieceType, int],
    bound: dict[chess.PieceType, int],
) -> bool:
    # Captures reduce `bound`; promotions change piece type but not total
    # material. A particle with fewer opponent pieces than the remaining
    # material ledger is under-modeling the board, even if no per-type upper
    # bound is exceeded.
    if sum(counts.values()) < sum(bound.values()):
        return True

    promotion_credit = max(
        0, bound.get(chess.PAWN, 0) - counts.get(chess.PAWN, 0)
    )
    promoted_excess = 0
    for piece_type, n in counts.items():
        if piece_type == chess.PAWN:
            if n > bound.get(piece_type, 0):
                return True
            continue
        if piece_type == chess.KING:
            if n > bound.get(piece_type, 0):
                return True
            continue
        if n > bound.get(piece_type, 0):
            promoted_excess += n - bound.get(piece_type, 0)
    return promoted_excess > promotion_credit


def _violates_upper_count_constraint(
    counts: dict[chess.PieceType, int],
    bound: dict[chess.PieceType, int],
) -> bool:
    """True when a particle has more material than capture/promotion can allow."""
    promotion_credit = max(
        0, bound.get(chess.PAWN, 0) - counts.get(chess.PAWN, 0)
    )
    promoted_excess = 0
    for piece_type, n in counts.items():
        if piece_type == chess.PAWN:
            if n > bound.get(piece_type, 0):
                return True
            continue
        if piece_type == chess.KING:
            if n > bound.get(piece_type, 0):
                return True
            continue
        if n > bound.get(piece_type, 0):
            promoted_excess += n - bound.get(piece_type, 0)
    return promoted_excess > promotion_credit


@dataclass(frozen=True)
class BeliefHardFacts:
    """Strict facts a belief update must satisfy for one observation.

    This is the first explicit boundary for the particle sub-engine's
    validator contract. Repair and reseed may use soft priors later, but they
    should route strict facts through this object instead of reassembling them
    ad hoc in each recovery path.

    The first implemented durable fact family is square occupancy, but the
    container is intentionally broader than squares. Future strict facts should
    include individual piece-token identities, castling rights, en-passant
    state, promotion/accounting state, and other legal-reachability facts.
    """

    observation: Observation
    perspective: chess.Color
    opp_remaining_counts: dict[chess.PieceType, int]
    opp_bishop_colors_remaining: dict[bool, int]
    hard_opp_occupancy_squares: frozenset[chess.Square] = frozenset()
    hard_opp_piece_facts: dict[chess.Square, chess.Piece] = field(
        default_factory=dict
    )

    @property
    def visibility_set(self) -> set[chess.Square]:
        return set(self.observation.visibility_mask)

    @property
    def visible_pieces(self) -> dict[chess.Square, chess.Piece]:
        return self.observation.visible_pieces

    @property
    def opp(self) -> chess.Color:
        return not self.perspective

    def required_hidden_opp_squares(self) -> set[chess.Square]:
        required = set(
            _required_hidden_opp_blockers_from_pawn_affordance(
                self.observation, self.perspective
            )
        )
        landing = self.observation.opp_capture_landing_square
        if landing is not None and landing not in self.visible_pieces:
            required.add(landing)
        required.update(
            sq
            for sq in self.hard_opp_occupancy_squares
            if sq not in self.visibility_set
        )
        required.update(
            sq for sq in self.hard_opp_piece_facts if sq not in self.visibility_set
        )
        return required

    def matches_visible_board(self, board: chess.Board) -> bool:
        visible = visible_squares(board, self.perspective)
        return visible == self.observation.visibility_mask and (
            piece_map_for_squares(board, visible) == self.visible_pieces
        )

    def matches_visible_squares_exactly(self, board: chess.Board) -> bool:
        for sq in self.observation.visibility_mask:
            if board.piece_at(sq) != self.visible_pieces.get(sq):
                return False
        return True

    def matches_hard_transition(
        self,
        next_board: chess.Board,
        prev_board: chess.Board,
        opp_move: chess.Move | None = None,
    ) -> bool:
        if not self.matches_visible_squares_exactly(next_board):
            return False

        own_before = {
            sq for sq, p in prev_board.piece_map().items() if p.color == self.perspective
        }
        own_after = {
            sq for sq, p in next_board.piece_map().items() if p.color == self.perspective
        }
        captures = own_before - own_after

        if self.observation.own_capture_square is None:
            if captures:
                return False
        elif captures != {self.observation.own_capture_square}:
            return False

        landing = self.observation.opp_capture_landing_square
        if landing is not None:
            landing_piece = next_board.piece_at(landing)
            if landing_piece is None or landing_piece.color == self.perspective:
                return False

        for sq, piece in self.hard_opp_piece_facts.items():
            if sq in self.visibility_set:
                continue
            if (
                opp_move is not None
                and opp_move.from_square == sq
                and prev_board.piece_at(sq) == piece
            ):
                continue
            if next_board.piece_at(sq) != piece:
                return False

        if (
            self.observation.game_over is not None
            and next_board.king(self.perspective) is not None
        ):
            return False

        return True

    def counts_valid(self, board: chess.Board) -> bool:
        return not _violates_count_constraint(
            _opp_piece_counts(board, self.perspective), self.opp_remaining_counts
        )

    def bishop_colors_valid(self, board: chess.Board) -> bool:
        return not _violates_bishop_color_constraint(
            board, self.perspective, self.opp_bishop_colors_remaining
        )

    def piece_facts_valid(self, board: chess.Board) -> bool:
        return all(
            board.piece_at(sq) == piece
            for sq, piece in self.hard_opp_piece_facts.items()
            if sq not in self.visibility_set
        )

    def hidden_facts_valid(self, board: chess.Board) -> bool:
        for sq in self.hard_opp_occupancy_squares:
            if sq in self.visibility_set:
                continue
            piece = board.piece_at(sq)
            if piece is None or piece.color != self.opp:
                return False
        return self.piece_facts_valid(board)

    def with_piece_fact_moved_by(
        self, prev_board: chess.Board, opp_move: chess.Move
    ) -> BeliefHardFacts:
        """Drop exactly the prior piece fact(s) that this opponent move relocates.

        Exact visible-piece facts should not evaporate just because the opponent
        got a turn. They expire only when a surviving transition actually moves
        the same piece from the fact square.

        Castling moves both the king AND a rook at once — only the king's from
        square is on opp_move. Drop both facts for castling so the downstream
        matches_hard_transition check doesn't reject the castled particle by
        looking up a rook on its original square. Without this, hard rook facts
        on a1/h1/a8/h8 silently kill every castled particle in the cloud (q0
        ply 13 annotation).
        """
        next_piece_facts = dict(self.hard_opp_piece_facts)
        changed = False

        piece = next_piece_facts.get(opp_move.from_square)
        if piece is not None and prev_board.piece_at(opp_move.from_square) == piece:
            del next_piece_facts[opp_move.from_square]
            changed = True

        if prev_board.is_castling(opp_move):
            king_rank = chess.square_rank(opp_move.from_square)
            # Kingside (king goes to file g): rook comes from h-file.
            # Queenside (king goes to file c): rook comes from a-file.
            kingside = chess.square_file(opp_move.to_square) == 6
            rook_from = chess.square(7 if kingside else 0, king_rank)
            rook_fact = next_piece_facts.get(rook_from)
            if (
                rook_fact is not None
                and rook_fact.piece_type == chess.ROOK
                and prev_board.piece_at(rook_from) == rook_fact
            ):
                del next_piece_facts[rook_from]
                changed = True

        if not changed:
            return self
        return replace(self, hard_opp_piece_facts=next_piece_facts)


@dataclass
class BeliefState:
    """A weighted particle distribution over true boards consistent with observation history."""

    perspective: chess.Color
    move_prior: OpponentMovePrior
    target_n: int = 256
    particles: list[chess.Board] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)
    # Hard upper bound on opp piece counts. Decremented by `register_capture`
    # whenever we (the perspective) capture a visible enemy piece. Used as a
    # Stage-B post-expansion filter to prune particles that hallucinate extra
    # pieces of types we've already captured.
    opp_remaining_counts: dict[chess.PieceType, int] = field(
        default_factory=lambda: dict(_STANDARD_OPP_COUNTS)
    )
    # v0.7.0: bishop-color preservation. Opp bishops never change square
    # color (impossible in standard chess; under-promotion to bishop is the
    # only counter-example). Tracked separately from opp_remaining_counts
    # because the CSP reseed needs to assign bishops to color-correct
    # hidden squares. Initial: {True: 1, False: 1} for canonical start.
    opp_bishop_colors_remaining: dict[bool, int] = field(
        default_factory=lambda: {True: 1, False: 1}
    )
    # Diagnostics: count of particles dropped by the count constraint on the
    # most recent Stage-B update. Surfaced in the trace for observability.
    last_constraint_pruned: int = 0
    # v0.7.0: count of times CSP reseed fired across the most recent update
    # cycle (Stage A + Stage B). Surfaced in the trace.
    last_csp_reseed_fired: int = 0
    last_csp_reseed_count: int = 0
    # v0.9.2: jitter augmentation fires when CSP reseed yields too few unique
    # particles (< _JITTER_UNIQUE_FLOOR). Diversifies by swapping same-group
    # hidden opponent pieces rather than random-refilling from scratch.
    last_jitter_fired: int = 0
    last_jitter_count: int = 0
    # v0.7.2: count of times continuity repair avoided generic CSP reseed.
    # Generic CSP is deliberately treated as a last resort because it preserves
    # hard facts but can scramble previously good hidden-piece tracks.
    last_repair_fired: int = 0
    last_repair_count: int = 0
    # v0.7.30: repair-quality diagnostics. These are telemetry only; high
    # values mean repair preserved belief liveness by making a large or
    # non-one-move edit, which should enter the review queue.
    last_repair_cost_max: int = 0
    last_repair_cost_total: int = 0
    last_repair_moved_piece_count_max: int = 0
    last_repair_max_piece_distance: int = 0
    last_repair_long_move_count: int = 0
    last_repair_teleport_like_count: int = 0
    last_repair_forced_visible_square_count: int = 0
    last_repair_unpaired_added_count: int = 0
    last_repair_unpaired_removed_count: int = 0
    last_repair_worst_cost: int = 0
    last_repair_worst_piece: str | None = None
    last_repair_worst_from: str | None = None
    last_repair_worst_to: str | None = None
    last_repair_worst_distance: int = 0
    last_repair_worst_one_move_legal: bool | None = None
    last_repair_strict_rejected_count: int = 0
    last_repair_strict_fallback_count: int = 0
    last_stage_a_pushed_count: int = 0
    last_stage_a_pushed_unique: int = 0
    last_stage_a_consistent_count: int = 0
    last_stage_a_consistent_unique: int = 0
    last_stage_a_repair_supplement_count: int = 0
    last_stage_a_elapsed_ms: float = 0.0
    last_stage_a_filter_ms: float = 0.0
    last_stage_a_repair_ms: float = 0.0
    last_stage_a_csp_ms: float = 0.0
    last_stage_a_resample_ms: float = 0.0
    last_stage_a_reject_illegal: int = 0
    last_stage_a_reject_observation: int = 0
    last_stage_a_reject_hard: int = 0
    last_stage_b_primary_count: int = 0
    last_stage_b_primary_unique: int = 0
    last_stage_b_constraint_count: int = 0
    last_stage_b_constraint_unique: int = 0
    last_stage_b_repair_supplement_count: int = 0
    last_stage_b_repair_supplement_considered_count: int = 0
    last_stage_b_repair_supplement_dropped_count: int = 0
    last_stage_b_elapsed_ms: float = 0.0
    last_stage_b_expand_ms: float = 0.0
    last_stage_b_repair_ms: float = 0.0
    last_stage_b_csp_ms: float = 0.0
    last_stage_b_resample_ms: float = 0.0
    last_stage_b_expanded_count: int = 0
    last_stage_b_obs_checked_count: int = 0
    last_stage_b_reject_observation: int = 0
    last_stage_b_reject_hard: int = 0
    last_stage_b_reject_count: int = 0
    # Hidden squares known to contain an opponent piece from prior hard
    # observations, most commonly "opponent captured our piece on this hidden
    # landing square." These facts must survive our own move repair: opponent
    # pieces cannot vanish during our move just because the square is still in
    # fog. The ledger is pruned after updates if particles no longer
    # unanimously support the occupancy or if a visible observation disproves
    # it.
    hard_opp_occupancy_squares: set[chess.Square] = field(default_factory=set)
    # Exact opponent piece facts learned by direct visibility. If we see a
    # black rook on e4, that piece cannot change during our own move even if e4
    # later falls into fog. The fact becomes soft after the opponent has a move
    # opportunity, then survives only if every legal surviving particle still
    # supports the exact square+piece identity.
    hard_opp_piece_facts: dict[chess.Square, chess.Piece] = field(
        default_factory=dict
    )
    # v0.7.31: recent high-health belief checkpoint. When current repair fails,
    # this gives the particle engine a continuity-preserving recovery source
    # before generic CSP random-fills from facts. It is intentionally bounded
    # and local; full legal-path reconstruction is a later subtrack.
    checkpoint_particles: list[chess.Board] = field(default_factory=list)
    checkpoint_weights: list[float] = field(default_factory=list)
    checkpoint_update_index: int = 0
    update_index: int = 0
    last_checkpoint_repair_fired: int = 0
    last_checkpoint_repair_count: int = 0
    last_checkpoint_repair_age: int = 0
    last_checkpoint_repair_unique: int = 0

    @classmethod
    def initial(
        cls,
        perspective: chess.Color,
        move_prior: OpponentMovePrior,
        target_n: int = 256,
        start_board: chess.Board | None = None,
        rng: random.Random | None = None,
        *,
        start_fen: str | None = None,
        chess960: bool = False,
    ) -> "BeliefState":
        """Build a belief seeded with a single known starting board.

        Forward-compat for Draft960 (engine-roadmap "Capability Tracks"):
        `start_fen` and `chess960` are accepted now so callers can pass
        them once Draft960 plumbing lands. Today both are stubs:

        - `start_fen` is honored if provided (overrides start_board), but
          the existing path (start_board / canonical) is preserved
          bit-identically when start_fen is None.
        - `chess960` is propagated to the seeded board's `.chess960` flag
          so future move-generation respects X-FEN castling, but no
          current callers set it. Default False = standard chess.
        """
        if start_fen is not None:
            seed_board = chess.Board(start_fen, chess960=chess960)
        else:
            seed_board = (start_board or chess.Board(chess960=chess960)).copy()
            if chess960:
                seed_board.chess960 = True
        # Seed opp_remaining_counts and opp_bishop_colors from the actual seed
        # board so non-canonical starts (FENs, Draft960) initialize correctly.
        opp_counts = _opp_piece_counts(seed_board, perspective)
        opp_bishop_colors = _opp_bishop_color_counts(seed_board, perspective)
        return cls(
            perspective=perspective,
            move_prior=move_prior,
            target_n=target_n,
            particles=[seed_board],
            weights=[1.0],
            rng=rng or random.Random(),
            opp_remaining_counts=opp_counts,
            opp_bishop_colors_remaining=opp_bishop_colors,
        )

    def register_capture(
        self,
        piece_type: chess.PieceType,
        square: chess.Square | None = None,
    ) -> None:
        """We just captured an opp piece of `piece_type`. Decrement the bound.

        Called by the strategy in `observe_own_move` when the move just played
        landed on a visible enemy piece (or was an en-passant capture). The
        bound is a hard truth: opponent now has one fewer piece of this type
        on the board (modulo promotion, which v0.6.0 does not track).

        v0.7.0: when `square` is provided and the captured piece is a bishop,
        decrement the matching color's count in `opp_bishop_colors_remaining`.
        Used by the CSP reseed to assign hidden bishops to color-correct
        squares. (For en-passant captures of pawns, the square is the pawn's
        actual square, not the move's to-square — pass that.)
        """
        if piece_type in self.opp_remaining_counts:
            self.opp_remaining_counts[piece_type] = max(
                0, self.opp_remaining_counts[piece_type] - 1
            )
        if piece_type == chess.BISHOP and square is not None:
            color_light = _is_light_square(square)
            self.opp_bishop_colors_remaining[color_light] = max(
                0, self.opp_bishop_colors_remaining.get(color_light, 0) - 1
            )
        if square is not None:
            # Captures invalidate any prior fact that an opponent piece still
            # occupies that square. For en passant, callers pass the captured
            # pawn's actual square, not the move destination.
            self.hard_opp_occupancy_squares.discard(square)
            self.hard_opp_piece_facts.pop(square, None)

    def _reset_repair_diagnostics(self) -> None:
        self.last_repair_fired = 0
        self.last_repair_count = 0
        self.last_checkpoint_repair_fired = 0
        self.last_checkpoint_repair_count = 0
        self.last_checkpoint_repair_age = 0
        self.last_checkpoint_repair_unique = 0
        self.last_repair_cost_max = 0
        self.last_repair_cost_total = 0
        self.last_repair_moved_piece_count_max = 0
        self.last_repair_max_piece_distance = 0
        self.last_repair_long_move_count = 0
        self.last_repair_teleport_like_count = 0
        self.last_repair_forced_visible_square_count = 0
        self.last_repair_unpaired_added_count = 0
        self.last_repair_unpaired_removed_count = 0
        self.last_repair_worst_cost = 0
        self.last_repair_worst_piece = None
        self.last_repair_worst_from = None
        self.last_repair_worst_to = None
        self.last_repair_worst_distance = 0
        self.last_repair_worst_one_move_legal = None
        self.last_repair_strict_rejected_count = 0
        self.last_repair_strict_fallback_count = 0

    def _record_repair_diagnostics(self, diag: RepairDiagnostics) -> None:
        self.last_repair_cost_max = max(self.last_repair_cost_max, diag.cost)
        self.last_repair_cost_total += diag.cost
        self.last_repair_moved_piece_count_max = max(
            self.last_repair_moved_piece_count_max, diag.moved_piece_count
        )
        self.last_repair_max_piece_distance = max(
            self.last_repair_max_piece_distance, diag.max_piece_distance
        )
        self.last_repair_long_move_count += diag.long_move_count
        self.last_repair_teleport_like_count += diag.teleport_like_count
        self.last_repair_forced_visible_square_count += (
            diag.forced_visible_square_count
        )
        self.last_repair_unpaired_added_count += diag.unpaired_added_count
        self.last_repair_unpaired_removed_count += diag.unpaired_removed_count
        if diag.cost > self.last_repair_worst_cost:
            self.last_repair_worst_cost = diag.cost
            self.last_repair_worst_piece = diag.worst_piece
            self.last_repair_worst_from = diag.worst_from
            self.last_repair_worst_to = diag.worst_to
            self.last_repair_worst_distance = diag.worst_distance
            self.last_repair_worst_one_move_legal = diag.worst_one_move_legal

    def _repair_candidate_weight_from_diag(
        self,
        diag: RepairDiagnostics,
        base_weight: float,
    ) -> float:
        self._record_repair_diagnostics(diag)
        return base_weight

    def _repair_candidate_weight(
        self,
        before: chess.Board,
        after: chess.Board,
        facts: BeliefHardFacts,
        base_weight: float,
    ) -> float:
        diag = _repair_diagnostics(before, after, facts.visibility_set)
        return self._repair_candidate_weight_from_diag(diag, base_weight)

    def _store_checkpoint(self) -> None:
        self.checkpoint_particles = [particle.copy() for particle in self.particles]
        self.checkpoint_weights = list(self.weights)
        self.checkpoint_update_index = self.update_index

    def _maybe_store_good_checkpoint(self) -> None:
        if not self.particles:
            return
        unique = len({particle.fen() for particle in self.particles})
        min_unique = min(16, max(2, self.target_n // 16))
        min_count = min(32, max(2, self.target_n // 8))
        if len(self.particles) < min_count or unique < min_unique:
            return
        if self.last_csp_reseed_fired:
            return
        if self.last_repair_teleport_like_count:
            return
        self._store_checkpoint()

    def _repair_from_checkpoint(
        self,
        facts: BeliefHardFacts,
        *,
        side_to_move: chess.Color,
        max_age: int = 8,
    ) -> tuple[list[chess.Board], list[float]]:
        if not self.checkpoint_particles:
            return [], []
        age = self.update_index - self.checkpoint_update_index
        if age < 0 or age > max_age:
            return [], []

        repaired: list[chess.Board] = []
        repaired_weights: list[float] = []
        seen: set[str] = set()
        for board, weight in zip(self.checkpoint_particles, self.checkpoint_weights, strict=False):
            repaired_board = _repair_particle_to_observation(
                board,
                facts,
                side_to_move=side_to_move,
                rng=self.rng,
            )
            if repaired_board is None:
                continue
            diag = _repair_diagnostics(board, repaired_board, facts.visibility_set)
            if not _repair_passes_strict_reachability(diag):
                self.last_repair_strict_rejected_count += 1
                continue
            fen = repaired_board.fen()
            if fen in seen:
                continue
            seen.add(fen)
            repaired.append(repaired_board)
            repaired_weights.append(
                self._repair_candidate_weight_from_diag(diag, weight)
            )

        if repaired:
            self.last_checkpoint_repair_fired += 1
            self.last_checkpoint_repair_count += len(repaired)
            self.last_checkpoint_repair_age = age
            self.last_checkpoint_repair_unique = len(
                {particle.fen() for particle in repaired}
            )
        return repaired, repaired_weights

    def update_after_own_move(
        self,
        my_move: chess.Move,
        observation: Observation | None = None,
    ) -> None:
        """Apply perspective's own move to every particle; drop particles where the move is illegal.

        Two-stage update:
          1. Push `my_move` on every particle, drop where the move wasn't
             pseudo-legal on that particle. (Bug-class: belief was never
             consistent with reality.)
          2. If `observation` provided, drop particles whose post-move
             visibility / piece map disagree with what we now actually see.
             (Pruning class: belief was over-broad; observation evidence narrows.)

        If step 2 wipes every particle, the pushed particles are stale against
        hard post-own-move evidence. Reseed from the observation instead of
        carrying an impossible belief forward; post-move visible pieces and
        move-affordance facts are current truth, not optional hints.
        """
        update_start = time.perf_counter()
        self.update_index += 1
        # Reset per-update CSP diagnostics; they're set if reseed fires below.
        self.last_csp_reseed_fired = 0
        self.last_csp_reseed_count = 0
        self.last_jitter_fired = 0
        self.last_jitter_count = 0
        self._reset_repair_diagnostics()
        self.last_stage_a_pushed_count = 0
        self.last_stage_a_pushed_unique = 0
        self.last_stage_a_consistent_count = 0
        self.last_stage_a_consistent_unique = 0
        self.last_stage_a_repair_supplement_count = 0
        self.last_stage_a_elapsed_ms = 0.0
        self.last_stage_a_filter_ms = 0.0
        self.last_stage_a_repair_ms = 0.0
        self.last_stage_a_csp_ms = 0.0
        self.last_stage_a_resample_ms = 0.0
        self.last_stage_a_reject_illegal = 0
        self.last_stage_a_reject_observation = 0
        self.last_stage_a_reject_hard = 0
        facts: BeliefHardFacts | None = None
        if observation is not None:
            self._update_hard_opp_facts_from_observation(observation)
            facts = self._hard_facts(observation)

        pushed: list[chess.Board] = []
        pushed_weights: list[float] = []
        consistent: list[chess.Board] = []
        consistent_weights: list[float] = []
        filter_start = time.perf_counter()
        for board, weight in zip(self.particles, self.weights, strict=False):
            if not board.is_pseudo_legal(my_move):
                self.last_stage_a_reject_illegal += 1
                continue
            advanced = board.copy()
            advanced.push(my_move)
            pushed.append(advanced)
            pushed_weights.append(weight)
            obs_ok = observation is None or consistent_with(
                advanced, board, observation, self.perspective
            )
            hard_ok = facts is None or facts.hidden_facts_valid(advanced)
            if obs_ok and hard_ok:
                consistent.append(advanced)
                consistent_weights.append(weight)
            else:
                if not obs_ok:
                    self.last_stage_a_reject_observation += 1
                if not hard_ok:
                    self.last_stage_a_reject_hard += 1
        self.last_stage_a_filter_ms = (
            time.perf_counter() - filter_start
        ) * 1000.0

        self.last_stage_a_pushed_count = len(pushed)
        self.last_stage_a_pushed_unique = len({board.fen() for board in pushed})
        self.last_stage_a_consistent_count = len(consistent)
        self.last_stage_a_consistent_unique = len(
            {board.fen() for board in consistent}
        )

        if consistent:
            if (
                observation is not None
                and facts is not None
                and pushed
                and _needs_repair_supplement(consistent, self.target_n)
            ):
                seen = {board.fen() for board in consistent}
                supplemented = list(consistent)
                supplemented_weights = list(consistent_weights)
                added = 0
                repair_start = time.perf_counter()
                for board, weight in zip(pushed, pushed_weights, strict=False):
                    repaired_board = _repair_particle_to_observation(
                        board,
                        facts,
                        side_to_move=not self.perspective,
                        rng=self.rng,
                    )
                    if repaired_board is None:
                        continue
                    fen = repaired_board.fen()
                    if fen in seen:
                        continue
                    diag = _repair_diagnostics(
                        board, repaired_board, facts.visibility_set
                    )
                    if not _repair_passes_strict_reachability(diag):
                        self.last_repair_strict_rejected_count += 1
                        continue
                    seen.add(fen)
                    supplemented.append(repaired_board)
                    supplemented_weights.append(
                        self._repair_candidate_weight_from_diag(diag, weight)
                    )
                    added += 1
                self.last_stage_a_repair_ms += (
                    time.perf_counter() - repair_start
                ) * 1000.0
                if added:
                    resample_start = time.perf_counter()
                    consistent, consistent_weights = _resample(
                        supplemented, supplemented_weights, self.target_n, self.rng
                    )
                    self.last_stage_a_resample_ms += (
                        time.perf_counter() - resample_start
                    ) * 1000.0
                    self.last_repair_fired += 1
                    self.last_repair_count += added
                    self.last_stage_a_repair_supplement_count = added
                    self.last_stage_a_consistent_count = len(consistent)
                    self.last_stage_a_consistent_unique = len(
                        {board.fen() for board in consistent}
                    )
            self.particles = consistent
            self.weights = consistent_weights
        elif pushed and observation is not None:
            # v0.7.0: step 2 would wipe belief. Older builds rolled back to
            # the pushed particles, but that meant belief could contradict
            # what our own move just revealed (e.g. a newly visible rook still
            # represented as a queen). First try an identity-preserving repair:
            # force current hard observation facts into each pushed particle,
            # keep hidden history that remains legal, and validate by
            # recomputing fog. Fall back to generic CSP only if repair fails.
            repaired: list[tuple[chess.Board, float, RepairDiagnostics]] = []
            repair_start = time.perf_counter()
            for board, weight in zip(pushed, pushed_weights, strict=False):
                repaired_board = _repair_particle_to_observation(
                    board,
                    facts,
                    side_to_move=not self.perspective,
                    rng=self.rng,
                )
                if repaired_board is not None:
                    diag = _repair_diagnostics(
                        board, repaired_board, facts.visibility_set
                    )
                    if not _repair_passes_strict_reachability(diag):
                        self.last_repair_strict_rejected_count += 1
                        continue
                    repaired.append((repaired_board, weight, diag))
            self.last_stage_a_repair_ms += (
                time.perf_counter() - repair_start
            ) * 1000.0

            if repaired:
                chosen_repairs = _select_repair_candidates(repaired, self.target_n)
                repaired_particles = [board for board, _, _ in chosen_repairs]
                repaired_weights = [
                    self._repair_candidate_weight_from_diag(diag, weight)
                    for _, weight, diag in chosen_repairs
                ]
                resample_start = time.perf_counter()
                self.particles, self.weights = _resample(
                    repaired_particles, repaired_weights, self.target_n, self.rng
                )
                self.last_stage_a_resample_ms += (
                    time.perf_counter() - resample_start
                ) * 1000.0
                self.last_repair_fired += 1
                self.last_repair_count = len(repaired_particles)
            else:
                checkpoint_repaired, checkpoint_weights = self._repair_from_checkpoint(
                    facts,
                    side_to_move=not self.perspective,
                )
                if checkpoint_repaired:
                    resample_start = time.perf_counter()
                    self.particles, self.weights = _resample(
                        checkpoint_repaired,
                        checkpoint_weights,
                        self.target_n,
                        self.rng,
                    )
                    self.last_stage_a_resample_ms += (
                        time.perf_counter() - resample_start
                    ) * 1000.0
                    self.last_repair_fired += 1
                    self.last_repair_count = len(checkpoint_repaired)
                else:
                    csp_start = time.perf_counter()
                    self.particles, self.weights = _csp_reseed_from_facts(
                        facts,
                        side_to_move=not self.perspective,
                        n=min(self.target_n, 64),
                        rng=self.rng,
                    )
                    self.last_stage_a_csp_ms += (
                        time.perf_counter() - csp_start
                    ) * 1000.0
                    self.last_csp_reseed_fired += 1
                    self.last_csp_reseed_count = len(self.particles)
                    _maybe_jitter(self, facts, not self.perspective)
        elif pushed:
            self.particles = pushed
            self.weights = pushed_weights
        elif observation is not None:
            # v0.7.0: step 1 wiped everything — `my_move` was not pseudo-legal
            # in any particle. Reseed via CSP using observation + opp piece-
            # count + bishop-color constraints. Replaces v0.6.3's degenerate
            # single-particle visibility-only seed with N rich particles that
            # have plausible hidden-square hypotheses, so Stage B can expand
            # opp moves immediately and per-particle eval sees realistic
            # boards.
            checkpoint_repaired, checkpoint_weights = self._repair_from_checkpoint(
                facts,
                side_to_move=not self.perspective,
            )
            if checkpoint_repaired:
                resample_start = time.perf_counter()
                self.particles, self.weights = _resample(
                    checkpoint_repaired, checkpoint_weights, self.target_n, self.rng
                )
                self.last_stage_a_resample_ms += (
                    time.perf_counter() - resample_start
                ) * 1000.0
                self.last_repair_fired += 1
                self.last_repair_count = len(checkpoint_repaired)
            else:
                csp_start = time.perf_counter()
                self.particles, self.weights = _csp_reseed_from_facts(
                    facts,
                    side_to_move=not self.perspective,
                    n=min(self.target_n, 64),
                    rng=self.rng,
                )
                self.last_stage_a_csp_ms += (
                    time.perf_counter() - csp_start
                ) * 1000.0
                self.last_csp_reseed_fired += 1
                self.last_csp_reseed_count = len(self.particles)
                _maybe_jitter(self, facts, not self.perspective)
        else:
            self.particles = []
            self.weights = []
        self._prune_hard_opp_facts_to_particles()
        self.last_stage_a_elapsed_ms = (
            time.perf_counter() - update_start
        ) * 1000.0
        self._maybe_store_good_checkpoint()

    def update_after_opp_move(self, obs: Observation) -> None:
        """Expand each particle by opp's pseudo-legal moves, filter by `obs` + count constraint, then resample.

        Filter priority (each layer is a fallback if the previous wipes belief):
          1. obs_pass AND constraint_pass: best — visibility match + correct
             opp piece counts.
          2. constraint_pass only: relax visibility, keep piece-count truth.
             Better than allowing phantom captured pieces back into belief.
          3. repair count-valid expansions into hard-observation compliance.
          4. generic CSP reseed from the current observation.

        The count constraint (v0.6.0): a particle whose opp piece count for
        any type exceeds `opp_remaining_counts` is hallucinating pieces we
        know we've captured. Drop it.
        """
        update_start = time.perf_counter()
        self.update_index += 1
        # Reset per-update CSP diagnostics; they're set if Trigger-B fires below.
        self.last_csp_reseed_fired = 0
        self.last_csp_reseed_count = 0
        self.last_jitter_fired = 0
        self.last_jitter_count = 0
        self._reset_repair_diagnostics()
        self.last_stage_b_primary_count = 0
        self.last_stage_b_primary_unique = 0
        self.last_stage_b_constraint_count = 0
        self.last_stage_b_constraint_unique = 0
        self.last_stage_b_repair_supplement_count = 0
        self.last_stage_b_repair_supplement_considered_count = 0
        self.last_stage_b_repair_supplement_dropped_count = 0
        self.last_stage_b_elapsed_ms = 0.0
        self.last_stage_b_expand_ms = 0.0
        self.last_stage_b_repair_ms = 0.0
        self.last_stage_b_csp_ms = 0.0
        self.last_stage_b_resample_ms = 0.0
        self.last_stage_b_expanded_count = 0
        self.last_stage_b_obs_checked_count = 0
        self.last_stage_b_reject_observation = 0
        self.last_stage_b_reject_hard = 0
        self.last_stage_b_reject_count = 0
        self._update_hard_opp_facts_from_observation(obs)
        # Prior hidden-occupancy facts are not strict across the opponent's
        # move: the opponent may have moved that hidden piece away. Exact prior
        # piece facts are source-aware: they expire only on branches whose
        # opponent move actually starts from that fact square with that piece.
        # Current observation facts (visible squares, captures, counts) remain
        # strict; after resampling, the ledger is retained only for facts every
        # surviving particle still supports.
        facts = self._hard_facts(
            obs,
            include_hard_opp_occupancy=False,
            include_hard_opp_piece_facts=True,
        )

        expanded: list[
            tuple[chess.Board, chess.Board, float, bool, bool, bool, BeliefHardFacts]
        ] = []
        expand_start = time.perf_counter()
        for prev_board, prev_weight in zip(self.particles, self.weights, strict=False):
            legal = list(prev_board.pseudo_legal_moves)
            if not legal:
                continue
            priors = self.move_prior(prev_board, legal)
            for mv, p in zip(legal, priors, strict=False):
                if p <= 0.0:
                    continue
                next_board = prev_board.copy()
                next_board.push(mv)
                transition_facts = facts.with_piece_fact_moved_by(prev_board, mv)
                hard_obs_ok = transition_facts.matches_hard_transition(
                    next_board, prev_board
                )
                count_ok = transition_facts.counts_valid(next_board)
                obs_ok = False
                if hard_obs_ok and count_ok:
                    self.last_stage_b_obs_checked_count += 1
                    obs_ok = consistent_with(
                        next_board, prev_board, obs, self.perspective
                    )
                expanded.append(
                    (
                        prev_board,
                        next_board,
                        prev_weight * p,
                        obs_ok,
                        hard_obs_ok,
                        count_ok,
                        transition_facts,
                    )
                )
        self.last_stage_b_expand_ms = (
            time.perf_counter() - expand_start
        ) * 1000.0
        self.last_stage_b_expanded_count = len(expanded)
        self.last_stage_b_reject_observation = sum(
            1 for _, _, _, obs_ok, _, _, _ in expanded if not obs_ok
        )
        self.last_stage_b_reject_hard = sum(
            1 for _, _, _, _, hard_ok, _, _ in expanded if not hard_ok
        )
        self.last_stage_b_reject_count = sum(
            1 for _, _, _, _, _, count_ok, _ in expanded if not count_ok
        )

        # Tier 1: obs + constraint
        primary_p = [
            b
            for _, b, _, obs_ok, hard_ok, c_ok, _ in expanded
            if obs_ok and hard_ok and c_ok
        ]
        primary_w = [
            w
            for _, _, w, obs_ok, hard_ok, c_ok, _ in expanded
            if obs_ok and hard_ok and c_ok
        ]
        # Tier 2: hard observation + constraint. Relax only the soft visibility
        # mask shape; never relax visible pieces, own captures, or game-over.
        constraint_p = [
            b for _, b, _, _, hard_ok, c_ok, _ in expanded if hard_ok and c_ok
        ]
        constraint_w = [
            w for _, _, w, _, hard_ok, c_ok, _ in expanded if hard_ok and c_ok
        ]
        self.last_stage_b_primary_count = len(primary_p)
        self.last_stage_b_primary_unique = len({board.fen() for board in primary_p})
        self.last_stage_b_constraint_count = len(constraint_p)
        self.last_stage_b_constraint_unique = len(
            {board.fen() for board in constraint_p}
        )

        # Diagnostic: how many particles the constraint pruned (regardless of obs match).
        self.last_constraint_pruned = sum(
            1 for _, _, _, _, _, c_ok, _ in expanded if not c_ok
        )

        if primary_p:
            chosen_particles = primary_p
            chosen_weights = primary_w
        elif constraint_p:
            chosen_particles = constraint_p
            chosen_weights = constraint_w
        elif expanded:
            # v0.7.2 Trigger B: all expanded opponent moves missed hard
            # observation. Before generic CSP, try the same continuity repair
            # Stage A uses: force current hard facts into expanded worlds,
            # preserve hidden history that still fits, fill missing remaining
            # material, then recompute fog exactly. This is allowed to break
            # exact opponent-move reachability, but it keeps stable pawn/piece
            # tracks instead of random-filling from scratch.
            repaired: list[tuple[chess.Board, float, RepairDiagnostics]] = []
            strict_repaired: list[tuple[chess.Board, float, RepairDiagnostics]] = []
            repair_start = time.perf_counter()
            recovery_sources = _select_repair_recovery_sources(
                expanded,
                _repair_recovery_source_limit(self.target_n),
            )
            for (
                prev_board,
                board,
                weight,
                _,
                _,
                _,
                _transition_facts,
            ) in recovery_sources:
                repaired_board = _repair_particle_to_observation(
                    board,
                    facts,
                    side_to_move=self.perspective,
                    rng=self.rng,
                    prev_board=prev_board,
                )
                if repaired_board is not None:
                    diag = _repair_diagnostics(
                        board, repaired_board, facts.visibility_set
                    )
                    candidate = (repaired_board, weight, diag)
                    repaired.append(candidate)
                    if _repair_passes_strict_reachability(diag):
                        strict_repaired.append(candidate)
                    else:
                        self.last_repair_strict_rejected_count += 1
            self.last_stage_b_repair_ms += (
                time.perf_counter() - repair_start
            ) * 1000.0

            if strict_repaired:
                chosen_repairs = _select_repair_candidates(
                    strict_repaired, self.target_n
                )
                repaired_particles = [board for board, _, _ in chosen_repairs]
                repaired_weights = [
                    self._repair_candidate_weight_from_diag(diag, weight)
                    for _, weight, diag in chosen_repairs
                ]
                resample_start = time.perf_counter()
                self.particles, self.weights = _resample(
                    repaired_particles, repaired_weights, self.target_n, self.rng
                )
                self.last_stage_b_resample_ms += (
                    time.perf_counter() - resample_start
                ) * 1000.0
                self.last_repair_fired += 1
                self.last_repair_count = len(repaired_particles)
                self._prune_hard_opp_facts_to_particles()
                self.last_stage_b_elapsed_ms = (
                    time.perf_counter() - update_start
                ) * 1000.0
                self._maybe_store_good_checkpoint()
                return
            if repaired:
                self.last_repair_strict_fallback_count += len(repaired)

            # Generic CSP remains the final emergency path. It preserves hard
            # facts but discards identity continuity, so repeated rows should
            # still enter the annotation queue.
            checkpoint_repaired, checkpoint_weights = self._repair_from_checkpoint(
                facts,
                side_to_move=self.perspective,
            )
            if checkpoint_repaired:
                resample_start = time.perf_counter()
                self.particles, self.weights = _resample(
                    checkpoint_repaired, checkpoint_weights, self.target_n, self.rng
                )
                self.last_stage_b_resample_ms += (
                    time.perf_counter() - resample_start
                ) * 1000.0
                self.last_repair_fired += 1
                self.last_repair_count = len(checkpoint_repaired)
            else:
                csp_start = time.perf_counter()
                self.particles, self.weights = _csp_reseed_from_facts(
                    facts,
                    side_to_move=self.perspective,
                    n=min(self.target_n, 64),
                    rng=self.rng,
                )
                self.last_stage_b_csp_ms += (
                    time.perf_counter() - csp_start
                ) * 1000.0
                self.last_csp_reseed_fired += 1
                self.last_csp_reseed_count = len(self.particles)
                _maybe_jitter(self, facts, self.perspective)
            self._prune_hard_opp_facts_to_particles()
            self.last_stage_b_elapsed_ms = (
                time.perf_counter() - update_start
            ) * 1000.0
            self._maybe_store_good_checkpoint()
            return
        else:
            # No particle had any expandable opponent move. Keeping empty
            # belief makes the next decision contradict every visible hard
            # fact, so recover from the observation directly.
            checkpoint_repaired, checkpoint_weights = self._repair_from_checkpoint(
                facts,
                side_to_move=self.perspective,
            )
            if checkpoint_repaired:
                resample_start = time.perf_counter()
                self.particles, self.weights = _resample(
                    checkpoint_repaired, checkpoint_weights, self.target_n, self.rng
                )
                self.last_stage_b_resample_ms += (
                    time.perf_counter() - resample_start
                ) * 1000.0
                self.last_repair_fired += 1
                self.last_repair_count = len(checkpoint_repaired)
            else:
                csp_start = time.perf_counter()
                self.particles, self.weights = _csp_reseed_from_facts(
                    facts,
                    side_to_move=self.perspective,
                    n=min(self.target_n, 64),
                    rng=self.rng,
                )
                self.last_stage_b_csp_ms += (
                    time.perf_counter() - csp_start
                ) * 1000.0
                self.last_csp_reseed_fired += 1
                self.last_csp_reseed_count = len(self.particles)
                _maybe_jitter(self, facts, self.perspective)
            self._prune_hard_opp_facts_to_particles()
            self.last_stage_b_elapsed_ms = (
                time.perf_counter() - update_start
            ) * 1000.0
            self._maybe_store_good_checkpoint()
            return

        if (
            chosen_particles
            and _needs_repair_supplement(chosen_particles, self.target_n)
            and expanded
        ):
            seen = {board.fen() for board in chosen_particles}
            supplemented = list(chosen_particles)
            supplemented_weights = list(chosen_weights)
            repair_candidates: list[
                tuple[tuple[int, int, int, float], chess.Board, float, RepairDiagnostics]
            ] = []
            supplement_limit = _repair_supplement_limit(
                chosen_particles, self.target_n
            )
            source_limit = _repair_supplement_source_limit(
                chosen_particles, self.target_n
            )
            repair_sources = _select_repair_supplement_sources(
                expanded,
                seen,
                source_limit,
            )
            repair_start = time.perf_counter()
            for prev_board, board, weight, _, _, _, _ in repair_sources:
                repaired_board = _repair_particle_to_observation(
                    board,
                    facts,
                    side_to_move=self.perspective,
                    rng=self.rng,
                    prev_board=prev_board,
                )
                if repaired_board is None:
                    continue
                fen = repaired_board.fen()
                if fen in seen:
                    continue
                diag = _repair_diagnostics(board, repaired_board, facts.visibility_set)
                if not _repair_passes_strict_reachability(diag):
                    self.last_repair_strict_rejected_count += 1
                    continue
                seen.add(fen)
                repair_candidates.append(
                    (
                        (
                            diag.cost,
                            diag.teleport_like_count,
                            diag.long_move_count,
                            -weight,
                        ),
                        repaired_board,
                        weight,
                        diag,
                    )
                )
            self.last_stage_b_repair_ms += (
                time.perf_counter() - repair_start
            ) * 1000.0
            self.last_stage_b_repair_supplement_considered_count = len(
                repair_candidates
            )
            repair_candidates.sort(key=lambda candidate: candidate[0])
            selected_repairs = repair_candidates[:supplement_limit]
            self.last_stage_b_repair_supplement_dropped_count = max(
                0, len(repair_candidates) - len(selected_repairs)
            )
            for _, repaired_board, weight, diag in selected_repairs:
                supplemented.append(repaired_board)
                supplemented_weights.append(
                    self._repair_candidate_weight_from_diag(diag, weight)
                )
            added = len(selected_repairs)
            if added:
                chosen_particles = supplemented
                chosen_weights = supplemented_weights
                self.last_repair_fired += 1
                self.last_repair_count += added
                self.last_stage_b_repair_supplement_count = added

        resample_start = time.perf_counter()
        self.particles, self.weights = _resample(
            chosen_particles, chosen_weights, self.target_n, self.rng
        )
        self.last_stage_b_resample_ms += (
            time.perf_counter() - resample_start
        ) * 1000.0
        self._prune_hard_opp_facts_to_particles()
        self.last_stage_b_elapsed_ms = (
            time.perf_counter() - update_start
        ) * 1000.0
        self._maybe_store_good_checkpoint()

    def marginal_piece_at(
        self, square: chess.Square
    ) -> dict[chess.Piece | None, float]:
        """Marginal distribution over what occupies `square` (None = empty)."""
        if not self.particles:
            return {}
        total = sum(self.weights)
        if total <= 0:
            return {}
        result: dict[chess.Piece | None, float] = {}
        for board, weight in zip(self.particles, self.weights, strict=False):
            piece = board.piece_at(square)
            result[piece] = result.get(piece, 0.0) + weight / total
        return result

    def marginal_piece_field(
        self, min_prob: float = 0.05
    ) -> dict[chess.Square, list[tuple[chess.Piece | None, float]]]:
        """Sparse per-square marginal distributions.

        For normal debug capture, include only squares where belief assigns at
        least `min_prob` probability to a non-empty piece. Once a square is
        included, keep all distribution entries at or above `min_prob`,
        including `None`, so the UI can show uncertainty vs emptiness. Passing
        `min_prob=0.0` returns every square's full distribution.
        """
        field: dict[chess.Square, list[tuple[chess.Piece | None, float]]] = {}
        if min_prob < 0:
            min_prob = 0.0
        for sq in chess.SQUARES:
            marginal = self.marginal_piece_at(sq)
            if not marginal:
                continue
            non_empty_peak = max(
                (prob for piece, prob in marginal.items() if piece is not None),
                default=0.0,
            )
            if min_prob > 0 and non_empty_peak < min_prob:
                continue
            entries = [
                (piece, prob)
                for piece, prob in marginal.items()
                if min_prob <= 0 or prob >= min_prob
            ]
            entries.sort(
                key=lambda item: (
                    item[0] is None,
                    -item[1],
                    "" if item[0] is None else item[0].symbol(),
                )
            )
            field[sq] = entries
        return field

    def top_k_clusters(self, k: int = 5) -> list[tuple[str, float, int]]:
        """Top-K unique particle worlds as `(fen, normalized_weight, count)`."""
        if k <= 0 or not self.particles:
            return []
        total = sum(self.weights)
        if total <= 0:
            return []
        weights_by_fen: dict[str, float] = defaultdict(float)
        counts_by_fen: dict[str, int] = defaultdict(int)
        for board, weight in zip(self.particles, self.weights, strict=False):
            fen = board.fen()
            weights_by_fen[fen] += weight
            counts_by_fen[fen] += 1
        clusters = [
            (fen, weight / total, counts_by_fen[fen])
            for fen, weight in weights_by_fen.items()
        ]
        clusters.sort(key=lambda item: (-item[1], item[0]))
        return clusters[:k]

    def particle_weight_profile(self, k: int = 8) -> dict:
        """Diagnostic profile of distinct-world particle weighting.

        The current scalar `weights` are a posterior-like mixture of move
        priors, observation filtering, resampling, repair, and CSP proposal
        artifacts. The forest lane needs to inspect that mass separately from
        raw appearance frequency, so this returns both channels per distinct
        FEN:

        - `posterior_mass`: normalized sum of particle weights for the FEN.
        - `appearance_mass`: count of appearances divided by total particles.

        This is diagnostic-only for now; decision code still consumes
        `self.weights` exactly as before.
        """
        particle_count = len(self.particles)
        if k <= 0 or particle_count <= 0:
            return {
                "summary": {
                    "particle_count": particle_count,
                    "unique_count": 0,
                    "effective_cluster_count": 0.0,
                    "posterior_top1_mass": 0.0,
                    "appearance_top1_mass": 0.0,
                    "posterior_entropy": 0.0,
                    "posterior_entropy_norm": 0.0,
                },
                "clusters": [],
            }

        total = sum(self.weights)
        if total <= 0:
            return {
                "summary": {
                    "particle_count": particle_count,
                    "unique_count": 0,
                    "effective_cluster_count": 0.0,
                    "posterior_top1_mass": 0.0,
                    "appearance_top1_mass": 0.0,
                    "posterior_entropy": 0.0,
                    "posterior_entropy_norm": 0.0,
                },
                "clusters": [],
            }

        weights_by_fen: dict[str, float] = defaultdict(float)
        counts_by_fen: dict[str, int] = defaultdict(int)
        for board, weight in zip(self.particles, self.weights, strict=False):
            fen = board.fen()
            weights_by_fen[fen] += weight
            counts_by_fen[fen] += 1

        unique_count = len(weights_by_fen)
        posterior_by_fen = {
            fen: weight / total for fen, weight in weights_by_fen.items()
        }
        appearance_by_fen = {
            fen: counts_by_fen[fen] / particle_count for fen in weights_by_fen
        }
        posterior_order = sorted(
            weights_by_fen,
            key=lambda fen: (-posterior_by_fen[fen], fen),
        )
        appearance_order = sorted(
            weights_by_fen,
            key=lambda fen: (-appearance_by_fen[fen], fen),
        )
        posterior_rank = {fen: idx + 1 for idx, fen in enumerate(posterior_order)}
        appearance_rank = {fen: idx + 1 for idx, fen in enumerate(appearance_order)}

        posterior_entropy = -sum(
            mass * math.log(mass) for mass in posterior_by_fen.values() if mass > 0
        )
        posterior_entropy_norm = (
            posterior_entropy / math.log(unique_count) if unique_count > 1 else 0.0
        )
        effective_cluster_count = 1.0 / sum(
            mass * mass for mass in posterior_by_fen.values()
        )

        clusters = [
            {
                "fen": fen,
                "particle_count": counts_by_fen[fen],
                "posterior_mass": posterior_by_fen[fen],
                "appearance_mass": appearance_by_fen[fen],
                "posterior_rank": posterior_rank[fen],
                "appearance_rank": appearance_rank[fen],
                "posterior_minus_appearance": (
                    posterior_by_fen[fen] - appearance_by_fen[fen]
                ),
            }
            for fen in posterior_order[:k]
        ]
        return {
            "summary": {
                "particle_count": particle_count,
                "unique_count": unique_count,
                "effective_cluster_count": effective_cluster_count,
                "posterior_top1_mass": (
                    posterior_by_fen[posterior_order[0]] if posterior_order else 0.0
                ),
                "appearance_top1_mass": (
                    appearance_by_fen[appearance_order[0]] if appearance_order else 0.0
                ),
                "posterior_entropy": posterior_entropy,
                "posterior_entropy_norm": posterior_entropy_norm,
            },
            "clusters": clusters,
        }

    def collapsed(self) -> bool:
        """True if no particle survived the most recent update; signals a tracker bug or rule mismatch."""
        return not self.particles

    def hard_fact_summary(self) -> dict[str, list[str]]:
        """Human/debug-facing strict facts currently carried by belief."""
        piece_facts = sorted(
            f"{chess.square_name(sq)}:{_piece_fact_name(piece)}"
            for sq, piece in self.hard_opp_piece_facts.items()
        )
        return {
            "square_facts": sorted(
                f"{chess.square_name(sq)}:hidden-opp-occupancy"
                for sq in self.hard_opp_occupancy_squares
            ),
            "piece_facts": piece_facts,
            "state_facts": [],
            # Back-compat for current Engine Lab panels and old scripts.
            "hidden_opp_occupancy": sorted(
                chess.square_name(sq) for sq in self.hard_opp_occupancy_squares
            ),
        }

    def _update_hard_opp_facts_from_observation(
        self, observation: Observation
    ) -> None:
        opp = not self.perspective
        visibility_set = set(observation.visibility_mask)
        for sq in list(self.hard_opp_occupancy_squares):
            if sq not in visibility_set:
                continue
            piece = observation.visible_pieces.get(sq)
            if piece is None or piece.color != opp:
                self.hard_opp_occupancy_squares.discard(sq)

        for sq in list(self.hard_opp_piece_facts):
            if sq not in visibility_set:
                continue
            piece = observation.visible_pieces.get(sq)
            if piece != self.hard_opp_piece_facts[sq]:
                del self.hard_opp_piece_facts[sq]

        for sq, piece in observation.visible_pieces.items():
            if piece.color == opp:
                self.hard_opp_piece_facts[sq] = piece

        landing = observation.opp_capture_landing_square
        if landing is None:
            return
        piece = observation.visible_pieces.get(landing)
        if landing not in visibility_set or piece is None or piece.color == opp:
            self.hard_opp_occupancy_squares.add(landing)

    def _prune_hard_opp_facts_to_particles(self) -> None:
        if (
            not self.particles
            or (
                not self.hard_opp_occupancy_squares
                and not self.hard_opp_piece_facts
            )
        ):
            return
        opp = not self.perspective
        for sq in list(self.hard_opp_occupancy_squares):
            if not all(
                (piece := particle.piece_at(sq)) is not None and piece.color == opp
                for particle in self.particles
            ):
                self.hard_opp_occupancy_squares.discard(sq)
        for sq, expected in list(self.hard_opp_piece_facts.items()):
            if not all(particle.piece_at(sq) == expected for particle in self.particles):
                del self.hard_opp_piece_facts[sq]

    def _hard_facts(
        self,
        observation: Observation,
        *,
        include_hard_opp_occupancy: bool = True,
        include_hard_opp_piece_facts: bool = True,
    ) -> BeliefHardFacts:
        return BeliefHardFacts(
            observation=observation,
            perspective=self.perspective,
            opp_remaining_counts=self.opp_remaining_counts,
            opp_bishop_colors_remaining=self.opp_bishop_colors_remaining,
            hard_opp_occupancy_squares=(
                frozenset(self.hard_opp_occupancy_squares)
                if include_hard_opp_occupancy
                else frozenset()
            ),
            hard_opp_piece_facts=(
                dict(self.hard_opp_piece_facts)
                if include_hard_opp_piece_facts
                else {}
            ),
        )


def _matches_hard_observation(
    next_board: chess.Board,
    prev_board: chess.Board,
    obs: Observation,
    perspective: chess.Color,
) -> bool:
    """True iff non-relaxable observation facts match.

    The full visibility mask can be noisy for belief recovery because it
    encodes many move-affordance negatives. Visible pieces are different:
    if the player sees a black pawn on b6, a particle without that pawn is
    impossible. Same for own pieces that disappeared and game-over.
    """
    facts = BeliefHardFacts(
        observation=obs,
        perspective=perspective,
        opp_remaining_counts=dict(_STANDARD_OPP_COUNTS),
        opp_bishop_colors_remaining={True: 1, False: 1},
    )
    return facts.matches_hard_transition(next_board, prev_board)


def _required_hidden_opp_squares_from_observation(
    observation: Observation,
    perspective: chess.Color,
    extra_required_opp_squares: set[chess.Square] | None = None,
) -> set[chess.Square]:
    """Hidden squares that hard observation says must contain opp pieces."""
    facts = BeliefHardFacts(
        observation=observation,
        perspective=perspective,
        opp_remaining_counts=dict(_STANDARD_OPP_COUNTS),
        opp_bishop_colors_remaining={True: 1, False: 1},
        hard_opp_occupancy_squares=frozenset(extra_required_opp_squares or set()),
    )
    return facts.required_hidden_opp_squares()


def _forced_capture_transition_from_observation(
    prev_board: chess.Board,
    observation: Observation,
    perspective: chess.Color,
) -> tuple[chess.Square, chess.Square, chess.Piece] | None:
    """Infer a strict hidden capture identity from a visible vacated source.

    If our piece disappeared on a hidden landing square and exactly one
    opponent piece from a now-visible-empty source could have captured it,
    repair must preserve that identity. This covers the game-14 class where a
    previously visible pawn moved from d5 to e4 in fog; the landing square is
    hidden, but the vacated visible source makes the capturer's identity hard
    evidence, not a soft CSP guess.
    """
    landing = observation.opp_capture_landing_square
    if landing is None:
        return None
    captured_piece = prev_board.piece_at(landing)
    if captured_piece is None or captured_piece.color != perspective:
        return None

    visibility_set = set(observation.visibility_mask)
    opp = not perspective
    candidates: list[tuple[chess.Square, chess.Square, chess.Piece]] = []
    for move in prev_board.pseudo_legal_moves:
        if move.to_square != landing:
            continue
        piece = prev_board.piece_at(move.from_square)
        if piece is None or piece.color != opp:
            continue
        # This is only a hard identity fact when the source is visible after
        # the move and observed empty. Hidden sources remain plausible but not
        # forced, so they stay in the soft particle-ranking problem.
        if move.from_square not in visibility_set:
            continue
        if observation.visible_pieces.get(move.from_square) is not None:
            continue
        landed_type = move.promotion or piece.piece_type
        candidates.append(
            (move.from_square, landing, chess.Piece(landed_type, opp))
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _required_hidden_opp_blockers_from_pawn_affordance(
    observation: Observation, perspective: chess.Color
) -> set[chess.Square]:
    """Squares that must contain hidden opp pieces due to pawn move affordance.

    Fog visibility includes destinations of pseudo-legal own moves. Therefore,
    if an own pawn's forward square is not visible, and no own piece sits there,
    the square is not merely unknown: it is occupied by a hidden opponent piece
    blocking the pawn. Same for a starting-rank double push when the one-step
    square is visibly empty but the two-step destination is not visible.
    """
    visibility_set = set(observation.visibility_mask)
    visible_pieces = observation.visible_pieces
    direction = 8 if perspective == chess.WHITE else -8
    start_rank = 1 if perspective == chess.WHITE else 6
    required: set[chess.Square] = set()

    for sq, piece in visible_pieces.items():
        if piece.color != perspective or piece.piece_type != chess.PAWN:
            continue

        one_step = sq + direction
        if not 0 <= one_step < 64:
            continue
        one_step_piece = visible_pieces.get(one_step)
        if one_step_piece is not None and one_step_piece.color == perspective:
            continue
        if one_step not in visibility_set:
            required.add(one_step)
            continue

        rank = chess.square_rank(sq)
        two_step = sq + 2 * direction
        if rank != start_rank or not 0 <= two_step < 64:
            continue
        one_step_empty = one_step in visibility_set and one_step not in visible_pieces
        two_step_piece = visible_pieces.get(two_step)
        own_piece_on_two_step = (
            two_step_piece is not None and two_step_piece.color == perspective
        )
        if one_step_empty and not own_piece_on_two_step and two_step not in visibility_set:
            required.add(two_step)

    return required


def _choose_required_blocker_piece_type(
    square: chess.Square,
    remaining_counts: dict[chess.PieceType, int],
    remaining_bishops_by_color: dict[bool, int],
    rng: random.Random,
) -> chess.PieceType | None:
    candidates: list[chess.PieceType] = []
    for pt, count in remaining_counts.items():
        if count <= 0:
            continue
        if pt == chess.PAWN and chess.square_rank(square) in {0, 7}:
            continue
        if pt == chess.BISHOP:
            if remaining_bishops_by_color.get(_is_light_square(square), 0) <= 0:
                continue
        candidates.extend([pt] * count)
    if not candidates:
        return None
    return rng.choice(candidates)


def _repair_diagnostics(
    before: chess.Board,
    after: chess.Board,
    visibility_set: set[chess.Square],
) -> RepairDiagnostics:
    forced_visible_square_count = sum(
        1 for sq in visibility_set if before.piece_at(sq) != after.piece_at(sq)
    )
    removed_by_key: dict[tuple[chess.Color, chess.PieceType], list[chess.Square]] = (
        defaultdict(list)
    )
    added_by_key: dict[tuple[chess.Color, chess.PieceType], list[chess.Square]] = (
        defaultdict(list)
    )
    for sq in chess.SQUARES:
        before_piece = before.piece_at(sq)
        after_piece = after.piece_at(sq)
        if before_piece == after_piece:
            continue
        if before_piece is not None:
            removed_by_key[(before_piece.color, before_piece.piece_type)].append(sq)
        if after_piece is not None:
            added_by_key[(after_piece.color, after_piece.piece_type)].append(sq)

    moved_piece_count = 0
    max_piece_distance = 0
    total_piece_distance = 0
    long_move_count = 0
    teleport_like_count = 0
    worst_piece: chess.Piece | None = None
    worst_source: chess.Square | None = None
    worst_target: chess.Square | None = None
    worst_distance = 0
    worst_one_move_legal: bool | None = None
    strict_unreachable_count = 0

    def record_moved_piece(
        piece: chess.Piece,
        source: chess.Square,
        target: chess.Square,
        one_move_legal: bool,
    ) -> None:
        nonlocal moved_piece_count
        nonlocal max_piece_distance
        nonlocal total_piece_distance
        nonlocal long_move_count
        nonlocal teleport_like_count
        nonlocal worst_piece
        nonlocal worst_source
        nonlocal worst_target
        nonlocal worst_distance
        nonlocal worst_one_move_legal
        nonlocal strict_unreachable_count

        distance = _square_chebyshev_distance(source, target)
        moved_piece_count += 1
        max_piece_distance = max(max_piece_distance, distance)
        total_piece_distance += distance
        if distance > worst_distance:
            worst_piece = piece
            worst_source = source
            worst_target = target
            worst_distance = distance
            worst_one_move_legal = one_move_legal
        if distance >= 4:
            long_move_count += 1
        if not one_move_legal:
            teleport_like_count += 1
            strict_unreachable_count += 1

    for key, added_squares in added_by_key.items():
        removed_squares = removed_by_key.get(key, [])
        piece = chess.Piece(key[1], key[0])

        # First preserve genuine one-move continuity. These are the only paired
        # moves that repair should treat as legal piece identity movement.
        while added_squares and removed_squares:
            legal_pairs = [
                (source, target)
                for source in removed_squares
                for target in added_squares
                if _piece_can_reach_in_one_move(before, source, target, piece)
            ]
            if not legal_pairs:
                break
            source, target = min(
                legal_pairs,
                key=lambda pair: _square_chebyshev_distance(pair[0], pair[1]),
            )
            removed_squares.remove(source)
            added_squares.remove(target)
            record_moved_piece(piece, source, target, True)

        # Remaining hidden-to-hidden same-piece changes are genuine teleport-like
        # continuity claims. Remaining changes that touch a visible square are
        # better interpreted as hard observation correction plus unknown material
        # add/remove, so leave them unpaired for the unpaired counters below.
        while added_squares and removed_squares:
            hidden_pairs = [
                (source, target)
                for source in removed_squares
                for target in added_squares
                if source not in visibility_set and target not in visibility_set
            ]
            if not hidden_pairs:
                break
            source, target = min(
                hidden_pairs,
                key=lambda pair: _square_chebyshev_distance(pair[0], pair[1]),
            )
            removed_squares.remove(source)
            added_squares.remove(target)
            record_moved_piece(piece, source, target, False)

    unpaired_added_count = sum(len(squares) for squares in added_by_key.values())
    unpaired_removed_count = sum(len(squares) for squares in removed_by_key.values())
    cost = (
        moved_piece_count * 4
        + total_piece_distance * 2
        + long_move_count * 8
        + teleport_like_count * 25
        + (unpaired_added_count + unpaired_removed_count) * 20
        + forced_visible_square_count * 3
    )
    return RepairDiagnostics(
        cost=cost,
        moved_piece_count=moved_piece_count,
        max_piece_distance=max_piece_distance,
        long_move_count=long_move_count,
        teleport_like_count=teleport_like_count,
        forced_visible_square_count=forced_visible_square_count,
        unpaired_added_count=unpaired_added_count,
        unpaired_removed_count=unpaired_removed_count,
        worst_piece=worst_piece.symbol() if worst_piece is not None else None,
        worst_from=(
            chess.square_name(worst_source) if worst_source is not None else None
        ),
        worst_to=chess.square_name(worst_target) if worst_target is not None else None,
        worst_distance=worst_distance,
        worst_one_move_legal=worst_one_move_legal,
        strict_unreachable_count=strict_unreachable_count,
    )


def _repair_passes_strict_reachability(diag: RepairDiagnostics) -> bool:
    return diag.strict_unreachable_count == 0


def _square_chebyshev_distance(a: chess.Square, b: chess.Square) -> int:
    return max(
        abs(chess.square_file(a) - chess.square_file(b)),
        abs(chess.square_rank(a) - chess.square_rank(b)),
    )


def _piece_can_reach_in_one_move(
    board: chess.Board,
    source: chess.Square,
    target: chess.Square,
    piece: chess.Piece,
) -> bool:
    if source == target:
        return True
    file_delta = chess.square_file(target) - chess.square_file(source)
    rank_delta = chess.square_rank(target) - chess.square_rank(source)
    abs_file = abs(file_delta)
    abs_rank = abs(rank_delta)
    if piece.piece_type == chess.KNIGHT:
        return (abs_file, abs_rank) in {(1, 2), (2, 1)}
    if piece.piece_type == chess.KING:
        return max(abs_file, abs_rank) == 1
    if piece.piece_type == chess.BISHOP:
        return abs_file == abs_rank and _line_clear(board, source, target)
    if piece.piece_type == chess.ROOK:
        return (file_delta == 0 or rank_delta == 0) and _line_clear(
            board, source, target
        )
    if piece.piece_type == chess.QUEEN:
        return (
            file_delta == 0 or rank_delta == 0 or abs_file == abs_rank
        ) and _line_clear(board, source, target)
    if piece.piece_type == chess.PAWN:
        direction = 1 if piece.color == chess.WHITE else -1
        start_rank = 1 if piece.color == chess.WHITE else 6
        if rank_delta == direction and abs_file <= 1:
            return True
        if (
            file_delta == 0
            and rank_delta == 2 * direction
            and chess.square_rank(source) == start_rank
        ):
            mid = chess.square(
                chess.square_file(source),
                chess.square_rank(source) + direction,
            )
            return board.piece_at(mid) is None
    return False


def _line_clear(board: chess.Board, source: chess.Square, target: chess.Square) -> bool:
    file_delta = chess.square_file(target) - chess.square_file(source)
    rank_delta = chess.square_rank(target) - chess.square_rank(source)
    step_file = 0 if file_delta == 0 else file_delta // abs(file_delta)
    step_rank = 0 if rank_delta == 0 else rank_delta // abs(rank_delta)
    file_cursor = chess.square_file(source) + step_file
    rank_cursor = chess.square_rank(source) + step_rank
    while (file_cursor, rank_cursor) != (
        chess.square_file(target),
        chess.square_rank(target),
    ):
        if board.piece_at(chess.square(file_cursor, rank_cursor)) is not None:
            return False
        file_cursor += step_file
        rank_cursor += step_rank
    return True


def _repair_particle_to_observation(
    board: chess.Board,
    facts: BeliefHardFacts,
    side_to_move: chess.Color,
    rng: random.Random,
    prev_board: chess.Board | None = None,
) -> chess.Board | None:
    """Minimally repair a pushed particle against current hard observation.

    Used before generic CSP reseed for Stage A post-own-move mismatches. The
    repair is intentionally conservative:

    - all own pieces are forced to the observed visible-piece map;
    - every visible square is forced to its observed piece or observed empty;
    - hidden opponent pieces are preserved unless they violate count/color
      bounds or are needed as movement-affordance blockers;
    - the candidate must exactly recompute the observation before it survives.

    This is not a full legal-reachability solver. It is a local continuity
    repair that keeps good hidden history when hard current facts changed.
    """
    repaired = board.copy()
    visibility_set = facts.visibility_set
    visible_pieces = facts.visible_pieces

    # Own pieces are always visible to the player. Any perspective-colored
    # piece missing from the visible map is stale.
    for sq, piece in list(repaired.piece_map().items()):
        if piece.color == facts.perspective and visible_pieces.get(sq) != piece:
            repaired.remove_piece_at(sq)

    # Visible squares are hard facts: either an exact piece or exact emptiness.
    for sq in visibility_set:
        expected = visible_pieces.get(sq)
        if expected is None:
            repaired.remove_piece_at(sq)
        else:
            repaired.set_piece_at(sq, expected)

    for sq, piece in visible_pieces.items():
        repaired.set_piece_at(sq, piece)

    forced_capture = (
        _forced_capture_transition_from_observation(
            prev_board, facts.observation, facts.perspective
        )
        if prev_board is not None
        else None
    )
    forced_squares: set[chess.Square] = set()
    if forced_capture is not None:
        source, landing, landed_piece = forced_capture
        if source != landing:
            repaired.remove_piece_at(source)
        repaired.set_piece_at(landing, landed_piece)
        forced_squares.add(landing)

    for sq, piece in facts.hard_opp_piece_facts.items():
        if sq in visibility_set:
            continue
        repaired.set_piece_at(sq, piece)

    required_hidden_opp_squares = facts.required_hidden_opp_squares()
    required_hidden_opp_squares |= forced_squares
    if not _repair_required_blockers(
        repaired,
        required_hidden_opp_squares,
        visibility_set,
        facts.opp,
        rng,
    ):
        return None

    if not _trim_opp_excess_hidden_pieces(
        repaired,
        visible_pieces,
        facts.opp_remaining_counts,
        facts.opp_bishop_colors_remaining,
        facts.perspective,
        required_hidden_opp_squares,
        rng,
    ):
        return None

    if not _fill_opp_missing_hidden_pieces(
        repaired,
        facts.opp_remaining_counts,
        facts.opp_bishop_colors_remaining,
        facts.perspective,
        visibility_set,
        rng,
    ):
        return None

    repaired.turn = side_to_move
    if not facts.counts_valid(repaired):
        return None
    if not facts.bishop_colors_valid(repaired):
        return None
    if not facts.piece_facts_valid(repaired):
        return None
    if not facts.matches_visible_board(repaired):
        return None
    return repaired


def _repair_required_blockers(
    board: chess.Board,
    required_blockers: set[chess.Square],
    visibility_set: set[chess.Square],
    opp: chess.Color,
    rng: random.Random,
) -> bool:
    """Ensure hidden movement-affordance blockers are occupied by opp pieces."""
    blocker_squares = list(required_blockers)
    rng.shuffle(blocker_squares)
    for sq in blocker_squares:
        current = board.piece_at(sq)
        if current is not None and current.color == opp:
            if _piece_can_occupy_hidden_square(current, sq):
                continue
            board.remove_piece_at(sq)
        elif current is not None:
            board.remove_piece_at(sq)

        candidates = [
            cand_sq
            for cand_sq, piece in board.piece_map().items()
            if piece.color == opp
            and cand_sq not in visibility_set
            and cand_sq not in required_blockers
            and _piece_can_occupy_hidden_square(piece, sq)
        ]
        if not candidates:
            return False
        from_sq = rng.choice(candidates)
        piece = board.remove_piece_at(from_sq)
        if piece is None:
            return False
        board.set_piece_at(sq, piece)
    return True


def _piece_can_occupy_hidden_square(piece: chess.Piece, square: chess.Square) -> bool:
    if piece.piece_type == chess.PAWN and chess.square_rank(square) in {0, 7}:
        return False
    if piece.piece_type == chess.BISHOP:
        # A bishop's color complex is identity-preserving in ordinary play.
        return True
    return True


def _trim_opp_excess_hidden_pieces(
    board: chess.Board,
    visible_pieces: dict[chess.Square, chess.Piece],
    opp_remaining_counts: dict[chess.PieceType, int],
    opp_bishop_colors_remaining: dict[bool, int],
    perspective: chess.Color,
    required_blockers: set[chess.Square],
    rng: random.Random,
) -> bool:
    """Remove hidden opp pieces that exceed hard count/color bounds."""
    opp = not perspective
    protected = set(visible_pieces) | set(required_blockers)

    for piece_type, count in _opp_piece_counts(board, perspective).items():
        excess = count - opp_remaining_counts.get(piece_type, 0)
        if excess <= 0:
            continue
        candidates = [
            sq
            for sq, piece in board.piece_map().items()
            if piece.color == opp
            and piece.piece_type == piece_type
            and sq not in protected
        ]
        if len(candidates) < excess:
            return False
        rng.shuffle(candidates)
        for sq in candidates[:excess]:
            board.remove_piece_at(sq)

    for color_light, allowed in opp_bishop_colors_remaining.items():
        squares = [
            sq
            for sq, piece in board.piece_map().items()
            if piece.color == opp
            and piece.piece_type == chess.BISHOP
            and _is_light_square(sq) == color_light
        ]
        excess = len(squares) - allowed
        if excess <= 0:
            continue
        candidates = [sq for sq in squares if sq not in protected]
        if len(candidates) < excess:
            return False
        rng.shuffle(candidates)
        for sq in candidates[:excess]:
            board.remove_piece_at(sq)

    return True


def _fill_opp_missing_hidden_pieces(
    board: chess.Board,
    opp_remaining_counts: dict[chess.PieceType, int],
    opp_bishop_colors_remaining: dict[bool, int],
    perspective: chess.Color,
    visibility_set: set[chess.Square],
    rng: random.Random,
) -> bool:
    """Place missing known-remaining opponent material on hidden squares."""

    opp = not perspective
    missing = _missing_opp_piece_types_for_material_ledger(
        _opp_piece_counts(board, perspective),
        opp_remaining_counts,
    )
    if not missing:
        return True

    current_bishop_colors = _opp_bishop_color_counts(board, perspective)
    hidden_empty = [
        sq
        for sq in chess.SQUARES
        if sq not in visibility_set and board.piece_at(sq) is None
    ]
    rng.shuffle(hidden_empty)

    # Tightest first: kings/bishops/pawns have more placement restrictions
    # than sliding/minor promoted material.
    priority = {
        chess.KING: 0,
        chess.BISHOP: 1,
        chess.PAWN: 2,
        chess.ROOK: 3,
        chess.KNIGHT: 3,
        chess.QUEEN: 3,
    }
    missing.sort(key=lambda pt: priority.get(pt, 9))

    for pt in missing:
        placed = False
        for sq in list(hidden_empty):
            piece = chess.Piece(pt, opp)
            if not _piece_can_occupy_hidden_square(piece, sq):
                continue
            if pt == chess.BISHOP:
                color_light = _is_light_square(sq)
                if (
                    current_bishop_colors.get(color_light, 0)
                    >= opp_bishop_colors_remaining.get(color_light, 0)
                ):
                    continue
            board.set_piece_at(sq, piece)
            hidden_empty.remove(sq)
            if pt == chess.BISHOP:
                current_bishop_colors[_is_light_square(sq)] += 1
            placed = True
            break
        if not placed:
            return False
    return True


def _missing_opp_piece_types_for_material_ledger(
    counts: dict[chess.PieceType, int],
    bound: dict[chess.PieceType, int],
) -> list[chess.PieceType]:
    """Return piece instances needed to satisfy remaining-material lower bounds."""

    missing: list[chess.PieceType] = []
    for pt, expected in bound.items():
        if pt == chess.PAWN:
            continue
        deficit = max(0, expected - counts.get(pt, 0))
        missing.extend([pt] * deficit)

    pawn_deficit = max(0, bound.get(chess.PAWN, 0) - counts.get(chess.PAWN, 0))
    promoted_excess = 0
    for pt, n in counts.items():
        if pt in {chess.PAWN, chess.KING}:
            continue
        promoted_excess += max(0, n - bound.get(pt, 0))
    pawn_deficit = max(0, pawn_deficit - promoted_excess)
    missing.extend([chess.PAWN] * pawn_deficit)
    return missing


def _violates_bishop_color_constraint(
    board: chess.Board,
    perspective: chess.Color,
    bound: dict[bool, int],
) -> bool:
    counts = _opp_bishop_color_counts(board, perspective)
    return any(count > bound.get(color_light, 0) for color_light, count in counts.items())


def _csp_reseed(
    observation: Observation,
    opp_remaining_counts: dict[chess.PieceType, int],
    opp_bishop_colors_remaining: dict[bool, int],
    perspective: chess.Color,
    side_to_move: chess.Color,
    n: int,
    rng: random.Random,
    extra_required_opp_squares: set[chess.Square] | None = None,
) -> tuple[list[chess.Board], list[float]]:
    facts = BeliefHardFacts(
        observation=observation,
        perspective=perspective,
        opp_remaining_counts=opp_remaining_counts,
        opp_bishop_colors_remaining=opp_bishop_colors_remaining,
        hard_opp_occupancy_squares=frozenset(extra_required_opp_squares or set()),
    )
    return _csp_reseed_from_facts(facts, side_to_move, n, rng)


_JITTER_UNIQUE_FLOOR: int = 4
"""Unique-particle threshold below which jitter augmentation fires after CSP reseed.

A 1-particle CSP result causes a belief-collapse spiral: Stage A/B pruning
wipes the lone particle, triggering another CSP that also yields 1, and so on.
Jitter breaks the cycle by deriving diverse variants from the survivors instead
of random-filling from scratch.
"""


def _jitter_particles_from_seeds(
    seeds: list[chess.Board],
    facts: "BeliefHardFacts",
    n: int,
    rng: random.Random,
    *,
    max_attempts: int = 200,
) -> tuple[list[chess.Board], list[float]]:
    """Diversify a sparse CSP-reseed result by swapping hidden opponent piece positions.

    For each seed particle, generates variants by randomly swapping pairs of
    hidden opponent pieces that belong to the same swap-group. Bishops are
    grouped by square colour so the bishop-colour constraint is always
    preserved; all other same-type swaps are unconditionally valid (counts,
    pawn-rank, and king-uniqueness constraints hold by construction).

    Only non-locked squares participate: squares in hard_opp_piece_facts
    (pinned by direct prior visibility) and required_hidden_opp_squares
    (forced occupancy from move-affordance evidence) are left untouched.

    Returns the original seeds plus jittered variants, up to n total, with
    equal weights. Returns (seeds, equal-weights) unchanged if no valid swap
    exists.
    """
    if not seeds:
        return [], []

    visibility_set = facts.visibility_set
    opp = facts.opp
    # Only exclude squares with exact piece-identity facts (type+color known from
    # direct sighting). required_hidden_opp_squares are occupancy-only ("some piece
    # must be here") — swapping between them still satisfies the occupancy constraint.
    locked: set[chess.Square] = set(facts.hard_opp_piece_facts.keys()) - visibility_set

    result: list[chess.Board] = list(seeds)
    seen: set[str] = {b.fen() for b in seeds}

    attempts = 0
    seed_idx = 0
    while len(result) < n and attempts < max_attempts:
        attempts += 1
        source = seeds[seed_idx % len(seeds)]
        seed_idx += 1

        # Group swappable hidden opp pieces. Bishops are split by square colour
        # so a swap never moves a bishop onto the wrong colour.
        by_group: dict[tuple[chess.PieceType, bool | None], list[chess.Square]] = defaultdict(list)
        for sq in chess.SQUARES:
            if sq in visibility_set or sq in locked:
                continue
            piece = source.piece_at(sq)
            if piece is not None and piece.color == opp:
                if piece.piece_type == chess.BISHOP:
                    key: tuple[chess.PieceType, bool | None] = (chess.BISHOP, _is_light_square(sq))
                else:
                    key = (piece.piece_type, None)
                by_group[key].append(sq)

        swappable = [g for g, sqs in by_group.items() if len(sqs) >= 2]
        if not swappable:
            continue

        group = rng.choice(swappable)
        sq_a, sq_b = rng.sample(by_group[group], 2)

        board = source.copy()
        piece_a = board.piece_at(sq_a)
        piece_b = board.piece_at(sq_b)
        board.set_piece_at(sq_a, piece_b)
        board.set_piece_at(sq_b, piece_a)

        fen = board.fen()
        if fen not in seen:
            seen.add(fen)
            result.append(board)

    if not result:
        return [], []
    w = 1.0 / len(result)
    return result, [w] * len(result)


def _maybe_jitter(
    belief: "BeliefState",
    facts: "BeliefHardFacts",
    side_to_move: chess.Color,
) -> None:
    """Augment belief in-place with jittered particles if CSP reseed was too sparse."""
    unique = len({p.fen() for p in belief.particles})
    if unique >= _JITTER_UNIQUE_FLOOR:
        return
    jittered, jitter_weights = _jitter_particles_from_seeds(
        belief.particles, facts, belief.target_n, belief.rng
    )
    if len(jittered) > len(belief.particles):
        belief.particles = jittered
        belief.weights = jitter_weights
        belief.last_jitter_fired += 1
        belief.last_jitter_count = len(jittered)


def _csp_reseed_from_facts(
    facts: BeliefHardFacts,
    side_to_move: chess.Color,
    n: int,
    rng: random.Random,
) -> tuple[list[chess.Board], list[float]]:
    """Generate up to `n` particles satisfying hard constraints from observation.

    v0.7.0 belief-recovery primitive. Replaces v0.6.3's degenerate visibility-
    only reseed when Stage A's step 1 wipes belief. See
    `docs/build-log/2026-05-07-fow-csp-reseed-design.md`.

    Hard constraints:
      - Visible squares match `observation.visible_pieces` exactly.
      - Per-type opp piece count ≤ `opp_remaining_counts[type]`.
      - Hidden bishops assigned to squares of the right color (per
        `opp_bishop_colors_remaining`).
      - Pawns never on rank 1 or 8 (no on-board promotion artifact).
      - One opp king total (visible or hidden).

    Algorithm: random-fill with rejection (Option A from the design doc).
    Each particle is generated independently; visible pieces are placed first,
    then hidden pieces are assigned to shuffled hidden squares respecting
    constraints. If a particle can't be completed (no valid square for some
    piece), reject and retry. Caps total attempts at `n * 10` to bound
    worst-case work.

    Returns equal-weight particles. If zero particles can be generated under
    the constraints (rare — usually means observation is already inconsistent
    with `opp_remaining_counts`), falls back to a single visibility-only
    particle so belief stays alive.
    """
    visibility_set = facts.visibility_set
    visible_pieces = facts.visible_pieces
    hidden_squares = [sq for sq in chess.SQUARES if sq not in visibility_set]
    required_blockers = facts.required_hidden_opp_squares()
    opp = facts.opp

    # Tally what's visible so we know what's left to place on hidden squares.
    visible_opp_by_type: dict[chess.PieceType, int] = defaultdict(int)
    visible_bishop_colors: dict[bool, int] = {True: 0, False: 0}
    for sq, piece in visible_pieces.items():
        if piece.color == opp:
            visible_opp_by_type[piece.piece_type] += 1
            if piece.piece_type == chess.BISHOP:
                visible_bishop_colors[_is_light_square(sq)] += 1

    # Pieces to assign to hidden squares.
    hidden_counts: dict[chess.PieceType, int] = {}
    for pt, total in facts.opp_remaining_counts.items():
        deficit = max(0, total - visible_opp_by_type[pt])
        hidden_counts[pt] = deficit

    # Bishops by color — placed first because their constraint is tightest.
    hidden_bishops_light = max(
        0, facts.opp_bishop_colors_remaining.get(True, 0) - visible_bishop_colors[True]
    )
    hidden_bishops_dark = max(
        0, facts.opp_bishop_colors_remaining.get(False, 0) - visible_bishop_colors[False]
    )

    particles: list[chess.Board] = []
    max_attempts = n * 10
    attempts = 0

    while len(particles) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board.empty()
        for sq, piece in visible_pieces.items():
            board.set_piece_at(sq, piece)

        # Per-attempt shuffle so each particle samples a different hidden layout.
        squares_shuffled = list(hidden_squares)
        rng.shuffle(squares_shuffled)
        used: set[chess.Square] = set()
        remaining_counts = dict(hidden_counts)
        remaining_bishops_by_color = {
            True: hidden_bishops_light,
            False: hidden_bishops_dark,
        }
        valid = True

        # 0. Exact hidden opponent piece facts from prior direct visibility.
        # These are stronger than generic occupancy blockers: if we saw a
        # black rook on e4 and e4 later fell into fog before the opponent could
        # move, every particle must still carry that rook on e4.
        for sq, piece in facts.hard_opp_piece_facts.items():
            if sq in visibility_set:
                continue
            if sq not in hidden_squares or piece.color != opp or sq in used:
                valid = False
                break
            if piece.piece_type == chess.PAWN and chess.square_rank(sq) in {0, 7}:
                valid = False
                break
            if remaining_counts.get(piece.piece_type, 0) <= 0:
                valid = False
                break
            if piece.piece_type == chess.BISHOP:
                color_light = _is_light_square(sq)
                if remaining_bishops_by_color.get(color_light, 0) <= 0:
                    valid = False
                    break
                remaining_bishops_by_color[color_light] -= 1
            board.set_piece_at(sq, piece)
            used.add(sq)
            remaining_counts[piece.piece_type] -= 1
        if not valid:
            continue

        # 1. Required hidden blockers from move-affordance evidence.
        # Example: if our pawn cannot push one square forward, and our own
        # piece is not on that square, the square must contain a hidden opp
        # piece. Put those pieces down before random fill so CSP reseed does
        # not erase hard "fog from movement restriction" facts.
        blocker_squares = list(required_blockers)
        rng.shuffle(blocker_squares)
        for sq in blocker_squares:
            if sq in used:
                continue
            pt = _choose_required_blocker_piece_type(
                sq, remaining_counts, remaining_bishops_by_color, rng
            )
            if pt is None:
                valid = False
                break
            board.set_piece_at(sq, chess.Piece(pt, opp))
            used.add(sq)
            remaining_counts[pt] -= 1
            if pt == chess.BISHOP:
                remaining_bishops_by_color[_is_light_square(sq)] -= 1
        if not valid:
            continue

        # 2. Light-square bishops.
        for _ in range(remaining_bishops_by_color[True]):
            placed = False
            for sq in squares_shuffled:
                if sq in used or not _is_light_square(sq):
                    continue
                board.set_piece_at(sq, chess.Piece(chess.BISHOP, opp))
                used.add(sq)
                remaining_counts[chess.BISHOP] -= 1
                placed = True
                break
            if not placed:
                valid = False
                break
        if not valid:
            continue

        # 3. Dark-square bishops.
        for _ in range(remaining_bishops_by_color[False]):
            placed = False
            for sq in squares_shuffled:
                if sq in used or _is_light_square(sq):
                    continue
                board.set_piece_at(sq, chess.Piece(chess.BISHOP, opp))
                used.add(sq)
                remaining_counts[chess.BISHOP] -= 1
                placed = True
                break
            if not placed:
                valid = False
                break
        if not valid:
            continue

        # 4. Other pieces (pawns + non-bishop). Pawns get rank constraint.
        hidden_to_place_non_bishop = [
            pt
            for pt, count in remaining_counts.items()
            if pt != chess.BISHOP
            for _ in range(count)
        ]
        rng.shuffle(hidden_to_place_non_bishop)
        for pt in hidden_to_place_non_bishop:
            placed = False
            for sq in squares_shuffled:
                if sq in used:
                    continue
                if pt == chess.PAWN:
                    rank = chess.square_rank(sq)
                    if rank == 0 or rank == 7:
                        continue
                board.set_piece_at(sq, chess.Piece(pt, opp))
                used.add(sq)
                placed = True
                break
            if not placed:
                valid = False
                break
        if not valid:
            continue

        board.turn = side_to_move
        if not facts.matches_visible_board(board):
            continue
        if not facts.counts_valid(board):
            continue
        if not facts.bishop_colors_valid(board):
            continue
        if not facts.piece_facts_valid(board):
            continue
        particles.append(board)

    if not particles:
        # Couldn't generate any constraint-satisfying particle. Fall back to
        # relaxed, diverse worlds instead of a single visibility-only board.
        # Exact fog-mask reconstruction is strict; in tight capture chains it
        # can reject every sampled full board. The old singleton fallback kept
        # belief alive but immediately collapsed decision input. Relax only the
        # full-mask check here: visible squares, hard facts, count bounds,
        # bishop colors, and pawn-rank constraints still hold.
        particles = _relaxed_csp_reseed_from_facts(
            facts,
            side_to_move=side_to_move,
            n=n,
            rng=rng,
        )
    elif len(particles) < n:
        # Full visibility validation is intentionally stricter than the early
        # v0.7.0 CSP fill. In tight positions it may find only a few valid
        # worlds inside the attempt budget. Keep the engine's expected particle
        # count stable by resampling those valid worlds with replacement; the
        # unique-particle diagnostic still exposes low diversity.
        particles = [rng.choice(particles).copy() for _ in range(n)]

    weights = [1.0 / len(particles)] * len(particles)
    return particles, weights


def _relaxed_csp_reseed_from_facts(
    facts: BeliefHardFacts,
    side_to_move: chess.Color,
    n: int,
    rng: random.Random,
) -> list[chess.Board]:
    """Generate diverse emergency particles when exact CSP finds zero worlds."""
    visibility_set = facts.visibility_set
    visible_pieces = facts.visible_pieces
    hidden_squares = [sq for sq in chess.SQUARES if sq not in visibility_set]
    required_blockers = facts.required_hidden_opp_squares()
    opp = facts.opp

    visible_opp_by_type: dict[chess.PieceType, int] = defaultdict(int)
    visible_bishop_colors: dict[bool, int] = {True: 0, False: 0}
    for sq, piece in visible_pieces.items():
        if piece.color == opp:
            visible_opp_by_type[piece.piece_type] += 1
            if piece.piece_type == chess.BISHOP:
                visible_bishop_colors[_is_light_square(sq)] += 1

    hidden_counts_template = {
        pt: max(0, total - visible_opp_by_type[pt])
        for pt, total in facts.opp_remaining_counts.items()
    }
    hidden_bishop_template = {
        True: max(
            0,
            facts.opp_bishop_colors_remaining.get(True, 0)
            - visible_bishop_colors[True],
        ),
        False: max(
            0,
            facts.opp_bishop_colors_remaining.get(False, 0)
            - visible_bishop_colors[False],
        ),
    }

    particles: list[chess.Board] = []
    attempts = 0
    max_attempts = max(n * 20, 1)
    while len(particles) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board.empty()
        for sq, piece in visible_pieces.items():
            board.set_piece_at(sq, piece)

        hidden_counts = dict(hidden_counts_template)
        hidden_bishops = dict(hidden_bishop_template)
        hidden_available = list(hidden_squares)
        rng.shuffle(hidden_available)
        used: set[chess.Square] = set()
        valid = True

        for sq, piece in facts.hard_opp_piece_facts.items():
            if sq in visibility_set:
                continue
            if sq not in hidden_squares or sq in used or piece.color != opp:
                valid = False
                break
            if piece.piece_type == chess.PAWN and chess.square_rank(sq) in {0, 7}:
                valid = False
                break
            if hidden_counts.get(piece.piece_type, 0) <= 0:
                valid = False
                break
            if piece.piece_type == chess.BISHOP:
                color_light = _is_light_square(sq)
                if hidden_bishops.get(color_light, 0) <= 0:
                    valid = False
                    break
                hidden_bishops[color_light] -= 1
            hidden_counts[piece.piece_type] -= 1
            board.set_piece_at(sq, piece)
            used.add(sq)
        if not valid:
            continue

        blocker_squares = list(required_blockers)
        rng.shuffle(blocker_squares)
        for sq in blocker_squares:
            if sq in visibility_set or sq in used:
                continue
            pt = _choose_required_blocker_piece_type(
                sq, hidden_counts, hidden_bishops, rng
            )
            if pt is None:
                valid = False
                break
            board.set_piece_at(sq, chess.Piece(pt, opp))
            used.add(sq)
            hidden_counts[pt] -= 1
            if pt == chess.BISHOP:
                hidden_bishops[_is_light_square(sq)] -= 1
        if not valid:
            continue

        for color_light, count in list(hidden_bishops.items()):
            for _ in range(count):
                placed_sq = _pop_random_square(
                    hidden_available,
                    used,
                    lambda sq, want=color_light: _is_light_square(sq) == want,
                )
                if placed_sq is None:
                    valid = False
                    break
                board.set_piece_at(placed_sq, chess.Piece(chess.BISHOP, opp))
                used.add(placed_sq)
                hidden_counts[chess.BISHOP] -= 1
            if not valid:
                break
        if not valid:
            continue

        pieces_to_place = [
            pt
            for pt, count in hidden_counts.items()
            if pt != chess.BISHOP
            for _ in range(count)
        ]
        rng.shuffle(pieces_to_place)
        for pt in pieces_to_place:
            placed_sq = _pop_random_square(
                hidden_available,
                used,
                lambda sq, piece_type=pt: piece_type != chess.PAWN
                or chess.square_rank(sq) not in {0, 7},
            )
            if placed_sq is None:
                valid = False
                break
            board.set_piece_at(placed_sq, chess.Piece(pt, opp))
            used.add(placed_sq)
        if not valid:
            continue

        board.turn = side_to_move
        if not facts.matches_visible_squares_exactly(board):
            continue
        if not facts.hidden_facts_valid(board):
            continue
        if not facts.counts_valid(board):
            continue
        if not facts.bishop_colors_valid(board):
            continue
        particles.append(board)

    if particles:
        return particles

    fallback = chess.Board.empty()
    for sq, piece in visible_pieces.items():
        fallback.set_piece_at(sq, piece)
    for sq, piece in facts.hard_opp_piece_facts.items():
        if sq not in visibility_set:
            fallback.set_piece_at(sq, piece)
    fallback.turn = side_to_move
    return [fallback]


def _pop_random_square(
    squares: list[chess.Square],
    used: set[chess.Square],
    predicate,
) -> chess.Square | None:
    for sq in squares:
        if sq in used or not predicate(sq):
            continue
        return sq
    return None


def _resample(
    particles: list[chess.Board],
    weights: list[float],
    target_n: int,
    rng: random.Random,
) -> tuple[list[chess.Board], list[float]]:
    total = sum(weights)
    if total <= 0:
        return [], []
    if len(particles) <= target_n:
        new_particles = [particle.copy() for particle in particles]
        new_weights = [weight / total for weight in weights]
        return new_particles, new_weights

    # Particle hypotheses are more valuable when diverse. Classic replacement
    # resampling duplicates high-weight boards and can collapse FOW belief into
    # a few top worlds even when many viable hypotheses survived the update.
    # Sample without replacement and preserve normalized posterior weights for
    # the selected boards.
    keyed: list[tuple[float, int]] = []
    for idx, weight in enumerate(weights):
        if weight <= 0:
            continue
        keyed.append((-_random_log(rng) / weight, idx))
    keyed.sort()
    indices = [idx for _, idx in keyed[:target_n]]
    selected_weight_total = sum(weights[idx] for idx in indices)
    if selected_weight_total <= 0:
        return [], []
    new_particles = [particles[i].copy() for i in indices]
    new_weights = [weights[i] / selected_weight_total for i in indices]
    return new_particles, new_weights


def _needs_repair_supplement(
    particles: list[chess.Board], target_n: int
) -> bool:
    unique = len({particle.fen() for particle in particles})
    min_unique = min(target_n, max(8, target_n // 8))
    return unique < min_unique


def _repair_supplement_limit(
    particles: list[chess.Board], target_n: int
) -> int:
    """Bound repair supplement mass to the diversity deficit.

    Repair supplements are meant to rescue a narrow but valid posterior, not
    let thousands of repaired alternatives overwhelm the sampled particles.
    """

    unique = len({particle.fen() for particle in particles})
    min_unique = min(target_n, max(8, target_n // 8))
    deficit = max(0, min_unique - unique)
    if deficit == 0:
        return 0
    cap = max(8, target_n // 4)
    return min(cap, max(8, deficit * 2))


def _repair_supplement_source_limit(
    particles: list[chess.Board], target_n: int
) -> int:
    """Bound expensive repair attempts before producing the supplement cloud."""

    supplement_limit = _repair_supplement_limit(particles, target_n)
    if supplement_limit == 0:
        return 0
    return min(max(32, target_n), max(32, supplement_limit * 4))


def _select_repair_supplement_sources(
    expanded: list[
        tuple[chess.Board, chess.Board, float, bool, bool, bool, BeliefHardFacts]
    ],
    existing_fens: set[str],
    limit: int,
    *,
    require_count_valid: bool = True,
) -> list[
    tuple[chess.Board, chess.Board, float, bool, bool, bool, BeliefHardFacts]
]:
    """Choose a small, likely-useful source pool before expensive repair.

    Stage-B supplement only needs a diversity top-up. Repairing every expanded
    world and sorting afterward wastes work, especially when thousands of
    near-duplicate count-valid branches collapse to a few selected repairs.
    Prefer branches already close to hard observation, then higher prior weight,
    and dedupe exact expanded worlds before invoking repair.
    """

    if limit <= 0:
        return []

    best_by_fen: dict[
        str,
        tuple[chess.Board, chess.Board, float, bool, bool, bool, BeliefHardFacts],
    ] = {}
    for candidate in expanded:
        _, board, _weight, _obs_ok, _hard_ok, count_ok, _ = candidate
        if require_count_valid and not count_ok:
            continue
        fen = board.fen()
        if fen in existing_fens:
            continue
        current = best_by_fen.get(fen)
        if current is None or _repair_source_sort_key(candidate) < (
            _repair_source_sort_key(current)
        ):
            best_by_fen[fen] = candidate

    selected = sorted(best_by_fen.values(), key=_repair_source_sort_key)
    return selected[:limit]


def _repair_recovery_source_limit(target_n: int) -> int:
    """Bound full Stage-B recovery repair while keeping room for failures."""

    return max(128, target_n * 4)


def _select_repair_recovery_sources(
    expanded: list[
        tuple[chess.Board, chess.Board, float, bool, bool, bool, BeliefHardFacts]
    ],
    limit: int,
) -> list[
    tuple[chess.Board, chess.Board, float, bool, bool, bool, BeliefHardFacts]
]:
    """Choose bounded full-recovery repair sources from count-valid expansions."""

    count_valid = [
        candidate
        for candidate in expanded
        if not _violates_upper_count_constraint(
            _opp_piece_counts(candidate[1], candidate[6].perspective),
            candidate[6].opp_remaining_counts,
        )
    ]
    return _select_repair_supplement_sources(
        count_valid,
        set(),
        limit,
        require_count_valid=False,
    )


def _repair_source_sort_key(
    candidate: tuple[
        chess.Board, chess.Board, float, bool, bool, bool, BeliefHardFacts
    ],
) -> tuple[bool, bool, float]:
    _, _, weight, obs_ok, hard_ok, _, _ = candidate
    return (not hard_ok, not obs_ok, -weight)


def _select_repair_candidates(
    candidates: list[tuple[chess.Board, float, RepairDiagnostics]],
    target_n: int,
) -> list[tuple[chess.Board, float, RepairDiagnostics]]:
    """Pick the best bounded repair set before resampling.

    Repair can produce thousands of strict candidates from expanded opponent
    moves. Resampling will ultimately keep at most `target_n`; sorting first
    prevents high-cost correction clouds from dominating runtime and diagnostics.
    """

    best_by_fen: dict[str, tuple[chess.Board, float, RepairDiagnostics]] = {}
    for candidate in candidates:
        board, _, _ = candidate
        fen = board.fen()
        current = best_by_fen.get(fen)
        if current is None or _repair_candidate_sort_key(candidate) < (
            _repair_candidate_sort_key(current)
        ):
            best_by_fen[fen] = candidate

    selected = sorted(best_by_fen.values(), key=_repair_candidate_sort_key)
    return selected[: max(1, target_n)]


def _repair_candidate_sort_key(
    candidate: tuple[chess.Board, float, RepairDiagnostics],
) -> tuple[int, int, int, int, int, float]:
    _, weight, diag = candidate
    return (
        diag.teleport_like_count,
        diag.long_move_count,
        diag.cost,
        diag.unpaired_added_count + diag.unpaired_removed_count,
        diag.forced_visible_square_count,
        -weight,
    )


def _random_log(rng: random.Random) -> float:
    """Return log(U) for U in (0, 1], avoiding log(0)."""
    return math.log(max(rng.random(), 1e-12))
