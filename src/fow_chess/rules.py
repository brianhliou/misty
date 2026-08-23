"""The ``Rules`` seam — the game interface the solver orchestration depends on.

This is the **first slice of Phase 1** of the engine-verticalization track
(see ``docs/engine/mini-xiangqi-verticalization-track.md``). It
introduces the coarse game interface that ``gt_cfr.py`` / ``engine_v2.py`` /
``p_enum/enumerator.py`` will eventually depend on instead of importing
``chess`` directly, with ``ChessRules`` forwarding to today's exact behavior.

**Nothing is rewired yet.** This module is a pure addition: it lands the
interface + a faithful chess adapter, characterized by ``test_rules_chess_parity``
against the canonical incumbents. The per-module rewiring (driver → enumerator →
gt_cfr) happens in later slices, each gated by ``test_pick_move_golden_trace``.

**Why reimplement instead of import:** the modules that own the canonical
helpers (``gt_cfr._mk``, ``engine_v2._upgrade_dominated_promotion``,
``enumerator._canonicalize_castling``) are exactly the ones that will later
import ``Rules`` — importing them here would create a cycle. So ``ChessRules``
reimplements the small pure helpers inline, and the parity test proves each is
**byte-identical** to its incumbent. That equality is the contract that lets the
later rewiring swap call-sites without changing behavior.

**The coarse-seam rule (load-bearing):** the generic solver calls these methods
at tree/belief boundaries, NEVER per-P-position inside a hot loop. Chess keeps
its Rust belief/visibility hot paths; mini keeps its own. The interface must not
force native↔generic conversion inside the ``P × pseudo-legal × observation``
loop. Keep this surface coarse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

# Type aliases kept game-agnostic on the abstract interface; concrete adapters
# pin them (ChessRules uses chess.Board / chess.Move / chess.Color).
Board = Any
Move = Any
Color = Any


class Rules(ABC):
    """The game-specific surface the solver orchestration reaches for.

    Coarse by design (see module docstring). Implemented by ``ChessRules``
    (dark chess, today's behavior) and — next on the roadmap — a Rust-backed
    ``MiniXiangqiRules`` (dark mini xiangqi 7×7).
    """

    #: Short game identifier, e.g. ``"chess"`` / ``"mini-xiangqi"``.
    name: str

    # --- setup / identity ---------------------------------------------------
    @abstractmethod
    def start_position(self) -> Board:
        """The initial board for a fresh game."""

    @abstractmethod
    def board_from_fen(self, fen: str) -> Board:
        """Reconstruct a board from its serialized form."""

    @abstractmethod
    def board_fen(self, board: Board) -> str:
        """Serialize a board's piece placement (the belief-set key)."""

    @abstractmethod
    def to_move(self, board: Board) -> Color:
        """Side to move on ``board``."""

    @abstractmethod
    def is_first_player(self, color: Color) -> bool:
        """True iff ``color`` is the first mover (chess: white; xiangqi: red)."""

    @property
    @abstractmethod
    def first_player(self) -> Color:
        """The first mover's color (chess: white; xiangqi: red)."""

    @property
    @abstractmethod
    def second_player(self) -> Color:
        """The second mover's color (chess: black)."""

    @abstractmethod
    def opponent(self, color: Color) -> Color:
        """The other side's color."""

    # --- move space ---------------------------------------------------------
    @abstractmethod
    def pseudo_legal_moves(self, board: Board) -> Iterable[Move]:
        """Pseudo-legal moves (FoW has no check, so legality is pseudo-legal)."""

    @abstractmethod
    def apply(self, board: Board, move: Move) -> Board:
        """Return a NEW board with ``move`` applied. Does not mutate ``board``."""

    @abstractmethod
    def action_key(self, move: Move) -> int:
        """Collision-free int key for per-(infoset, action) state dicts."""

    @abstractmethod
    def canonicalize_move(self, move: Move, board: Board) -> Move:
        """Rewrite a move into the engine's canonical encoding for ``board``
        (chess: king-takes-rook castling → king two-square)."""

    @abstractmethod
    def normalize_committed_move(self, move: Move) -> Move:
        """Last-mile normalization of the move the engine commits to play
        (chess: upgrade a dominated rook/bishop promotion to queen)."""

    @abstractmethod
    def decode_action_key(self, key: int) -> Move:
        """Inverse of ``action_key``: rebuild the Move from its packed int key
        (the Rust tree returns action keys; the readout decodes them)."""

    @abstractmethod
    def make_move(self, from_square: int, to_square: int, promotion: int) -> Move:
        """Build a Move from the raw (from, to, promotion) components the Rust
        tree's expand_node returns (promotion is 0 when absent)."""

    @abstractmethod
    def root_fen(self, board: Board) -> str:
        """Serialized board WITH side-to-move for the Rust tree's
        ``add_root_from_fen`` — chess: full FEN; mini: self-describing board_fen.
        (Distinct from ``board_fen``, which is the placement-only belief key.)"""

    @abstractmethod
    def is_search_valid(self, board: Board) -> bool:
        """Whether the leaf evaluator can score this board's children directly
        (chess: a fully-legal position; mini: always true)."""

    @abstractmethod
    def material_leaf_eval(self, board: Board, perspective: Color) -> float:
        """Material-only leaf value — the fallback for children the leaf
        evaluator declines (chess: FoW-legal-but-illegal positions)."""

    # --- terminal -----------------------------------------------------------
    @abstractmethod
    def is_terminal(self, board: Board) -> bool:
        """True iff the position is terminal (royal capture / game-specific)."""

    @abstractmethod
    def terminal_value(self, board: Board, perspective: Color) -> float:
        """Value in [-1, 1] from ``perspective`` at a terminal position."""

    @abstractmethod
    def royal_square(self, board: Board, color: Color):
        """Square of ``color``'s royal piece (chess: king; xiangqi: general), or
        None if absent. Used by the win-fast tiebreak to spot an immediate
        royal capture among equal-value moves."""

    def moves_equivalent(self, a: Move, b: Move) -> bool:
        """True when two move objects denote the same action.

        This default intentionally uses the python-chess attribute surface that
        MiniMove mirrors, so generic commit guards can match an action chosen
        from one sampled world against pseudo-legal moves in another world.
        """
        return (
            getattr(a, "from_square", None) == getattr(b, "from_square", None)
            and getattr(a, "to_square", None) == getattr(b, "to_square", None)
            and getattr(a, "promotion", None) == getattr(b, "promotion", None)
            and getattr(a, "drop", None) == getattr(b, "drop", None)
        )

    def matching_pseudo_legal_move(self, board: Board, move: Move) -> Move | None:
        """Return board-local pseudo-legal ``move`` if present, else ``None``."""
        for cand in self.pseudo_legal_moves(board):
            if self.moves_equivalent(cand, move):
                return cand
        return None

    def royal_capture_imminent(self, board: Board, perspective: Color) -> float | None:
        """If the side to move can capture the enemy royal now, return the
        terminal-scale value from ``perspective``'s POV; otherwise ``None``.

        The default works for games whose pseudo-legal move generator emits
        royal captures directly (DMX). Chess overrides this because
        python-chess represents king exposure through attacks rather than
        actual king-capture moves.
        """
        target = self.royal_square(board, self.opponent(self.to_move(board)))
        if target is None:
            return None
        for move in self.pseudo_legal_moves(board):
            if getattr(move, "to_square", None) == target:
                return 1.0 if perspective == self.to_move(board) else -1.0
        return None

    # --- fog ----------------------------------------------------------------
    @abstractmethod
    def visible_squares(self, board: Board, color: Color) -> Any:
        """Squares ``color`` can observe under FoW visibility."""

    @abstractmethod
    def observation_keys(self, prev_board: Board, next_board: Board) -> tuple:
        """Both players' infoset-history keys ``(first_player_key,
        second_player_key)`` for the transition ``prev_board`` -> ``next_board``.
        Folded into the obs-history that identifies a CFR infoset. Coarse: called
        once per expanded child, not per-P-position."""

    # --- belief -------------------------------------------------------------
    @abstractmethod
    def make_belief(
        self,
        perspective: Color,
        *,
        starting_board: Board | None = None,
        max_size: int | None = None,
        rng: Any = None,
        use_rust_state: bool = True,
    ) -> Any:
        """Construct the belief-state backend (the P enumerator) for this game.

        The COARSE belief boundary: the solver calls ``update_own_move`` /
        ``update_opp_move`` on the returned object and samples roots from it,
        but its internals (chess: the Rust hot paths; mini: its own backend)
        never cross this interface per-position. ``max_size`` / ``use_rust_state``
        are backend hints a game may honor or ignore."""


class ChessRules(Rules):
    """Dark chess adapter — forwards to today's exact behavior.

    Every method is characterized byte-for-byte against its canonical incumbent
    in ``test_rules_chess_parity``; this class must never diverge from them.
    """

    name = "chess"

    def __init__(self) -> None:
        import chess  # local import keeps the abstract interface chess-free
        self._chess = chess

    # --- setup / identity ---
    def start_position(self) -> Any:
        return self._chess.Board()

    def board_from_fen(self, fen: str) -> Any:
        return self._chess.Board(fen)

    def board_fen(self, board: Any) -> str:
        return board.board_fen()

    def to_move(self, board: Any) -> Any:
        return board.turn

    def is_first_player(self, color: Any) -> bool:
        return color == self._chess.WHITE

    @property
    def first_player(self) -> Any:
        return self._chess.WHITE

    @property
    def second_player(self) -> Any:
        return self._chess.BLACK

    def opponent(self, color: Any) -> Any:
        return not color

    # --- move space ---
    def pseudo_legal_moves(self, board: Any) -> Iterable[Any]:
        # FoW move space (2026-06-20 castle-into-check fix): python-chess excludes
        # castling onto an attacked square, but FoW *allows* it (the server's
        # fog-castle-through-check rule) — the king can't see the hidden attacker.
        # Add those castles so the search + guards see (and devalue) a castle that
        # walks the king into a fog-hidden capture. Mirrors the Rust
        # gen_fow_pseudo_legal_moves order (fog-castles sort into the castling group),
        # so Python/Rust tree parity holds. No-castle / safe-castle positions are
        # untouched (extras empty -> base returned unchanged).
        base = list(board.pseudo_legal_moves)
        extras = self._fow_castles_into_check(board, base)
        if not extras:
            return base
        moves = base + extras
        moves.sort(key=lambda m: self._pychess_order_key(board, m))
        return moves

    def _fow_castles_into_check(self, board: Any, base: list) -> list:
        """Structurally-available castles that python-chess dropped for landing on an
        attacked square (standard king-on-e + corner rook). FoW allows them."""
        chess = self._chess
        c = board.turn
        rank = 0 if c == chess.WHITE else 7
        e = chess.square(4, rank)
        if board.king(c) != e:
            return []  # chess960 / non-home king: handled by the server but not here
        rook = chess.Piece(chess.ROOK, c)
        h, g, f = chess.square(7, rank), chess.square(6, rank), chess.square(5, rank)
        a, b_, cc, d = (chess.square(0, rank), chess.square(1, rank),
                        chess.square(2, rank), chess.square(3, rank))
        cand = []
        if (board.has_kingside_castling_rights(c) and board.piece_at(h) == rook
                and board.piece_at(f) is None and board.piece_at(g) is None):
            cand.append(chess.Move(e, g))
        if (board.has_queenside_castling_rights(c) and board.piece_at(a) == rook
                and board.piece_at(b_) is None and board.piece_at(cc) is None
                and board.piece_at(d) is None):
            cand.append(chess.Move(e, cc))
        if not cand:
            return []
        have = {(m.from_square, m.to_square) for m in base}
        return [m for m in cand if (m.from_square, m.to_square) not in have]

    def _pychess_order_key(self, board: Any, m: Any):
        """Byte-equal to the Rust gen_..._pychess_order sort key (lib.rs ~763):
        (group, -from, -to, promo_order). Groups: 0 piece, 1 castling, 2 capture,
        3 single push, 4 double push, 5 en passant."""
        chess = self._chess
        f_sq, t_sq = m.from_square, m.to_square
        p = m.promotion or 0
        ff, tf = chess.square_file(f_sq), chess.square_file(t_sq)
        role = board.piece_type_at(f_sq)
        if role == chess.KING and abs(ff - tf) == 2:
            group = 1
        elif role == chess.PAWN:
            if ff == tf:
                group = 4 if abs(chess.square_rank(f_sq) - chess.square_rank(t_sq)) == 2 else 3
            elif board.piece_at(t_sq) is not None:
                group = 2
            else:
                group = 5
        else:
            group = 0
        promo_order = 0 if p == 0 else 5 - p
        return (group, -f_sq, -t_sq, promo_order)

    def apply(self, board: Any, move: Any) -> Any:
        nxt = board.copy()
        nxt.push(move)
        return nxt

    def action_key(self, move: Any) -> int:
        # Byte-identical to cfr.gt_cfr._mk (asserted in test_rules_chess_parity).
        return (
            move.from_square
            | (move.to_square << 6)
            | ((move.promotion or 0) << 12)
            | ((move.drop or 0) << 16)
        )

    def canonicalize_move(self, move: Any, board: Any) -> Any:
        # Byte-identical to p_enum.enumerator._canonicalize_castling.
        chess = self._chess
        fs, ts = move.from_square, move.to_square
        king_mask = board.kings & board.occupied_co[board.turn]
        if fs == chess.E1 and king_mask & (1 << fs):
            if ts == chess.H1:
                return chess.Move(fs, chess.G1)
            if ts == chess.A1:
                return chess.Move(fs, chess.C1)
        elif fs == chess.E8 and king_mask & (1 << fs):
            if ts == chess.H8:
                return chess.Move(fs, chess.G8)
            if ts == chess.A8:
                return chess.Move(fs, chess.C8)
        return move

    def normalize_committed_move(self, move: Any) -> Any:
        # Byte-identical to engine_v2._upgrade_dominated_promotion.
        chess = self._chess
        if move.promotion in (chess.ROOK, chess.BISHOP):
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return move

    def decode_action_key(self, key: int) -> Any:
        # Byte-identical to gt_cfr._decode_mk (from | to<<6 | promo<<12).
        chess = self._chess
        return chess.Move(key & 0x3F, (key >> 6) & 0x3F, promotion=((key >> 12) & 0x7) or None)

    def make_move(self, from_square: int, to_square: int, promotion: int) -> Any:
        return self._chess.Move(from_square, to_square, promotion=promotion or None)

    def root_fen(self, board: Any) -> str:
        return board.fen()

    def is_search_valid(self, board: Any) -> bool:
        return board.is_valid()

    def material_leaf_eval(self, board: Any, perspective: Any) -> float:
        from .cfr.leaf_eval import material_leaf_eval
        return material_leaf_eval(board, perspective)

    # --- terminal ---
    def is_terminal(self, board: Any) -> bool:
        # Byte-identical to GTCFRTreeNode.is_terminal (king capture).
        chess = self._chess
        return (
            board.king(chess.WHITE) is None
            or board.king(chess.BLACK) is None
        )

    def terminal_value(self, board: Any, perspective: Any) -> float:
        # Byte-identical to GTCFRTreeNode.terminal_value.
        own_king = board.king(perspective)
        opp_king = board.king(not perspective)
        if own_king is None and opp_king is None:
            return 0.0
        if own_king is None:
            return -1.0
        if opp_king is None:
            return 1.0
        return 0.0

    def royal_square(self, board: Any, color: Any):
        return board.king(color)

    def royal_capture_imminent(self, board: Any, perspective: Any) -> float | None:
        from .cfr.leaf_eval import king_capture_imminent
        return king_capture_imminent(board, perspective)

    # --- fog ---
    def visible_squares(self, board: Any, color: Any) -> Any:
        from .visibility import visible_squares  # leaf module, no cycle
        return visible_squares(board, color)

    def observation_keys(self, prev_board: Any, next_board: Any) -> tuple:
        # Forwards to today's exact obs-history key builder, so routing
        # expand_leaf through this seam is byte-identical (golden gate).
        from .cfr.walker import obs_keys_both
        return obs_keys_both(prev_board, next_board)

    # --- belief ---
    def make_belief(
        self,
        perspective: Any,
        *,
        starting_board: Any = None,
        max_size: int | None = None,
        rng: Any = None,
        use_rust_state: bool = True,
    ) -> Any:
        # Lazy import: p_enum does not depend on rules, so rules -> p_enum is a
        # safe one-way edge, but keep it lazy for symmetry with the other forwards.
        from .p_enum import PEnumerator
        return PEnumerator(
            perspective,
            starting_board=starting_board,
            max_size=max_size,
            rng=rng,
            use_rust_state=use_rust_state,
        )
