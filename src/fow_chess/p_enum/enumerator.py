"""Eager enumeration of positions consistent with observation history."""

from __future__ import annotations

import logging
import os
import random
from typing import Callable, Iterator, Sequence

import chess

from ..observation import Observation, consistent_with

_logger = logging.getLogger(__name__)


def _weighted_sample_without_replacement(
    items: Sequence[str], weights: Sequence[float], n: int, rng: random.Random
) -> list[str]:
    """Efraimidis-Spirakis A-Res: key_j = u_j**(1/w_j), keep the n largest keys.

    Weighted sampling WITHOUT replacement, deterministic for a given ``rng``.
    Non-positive weights get a tiny floor so a zero-weight world is improbable
    but never crashes the exponent. Consumes ``len(items)`` rng draws."""
    keyed: list[tuple[float, str]] = []
    for it, w in zip(items, weights, strict=True):
        w = w if w > 0.0 else 1e-12
        u = rng.random()
        keyed.append((u ** (1.0 / w), it))
    keyed.sort(key=lambda kv: kv[0], reverse=True)
    return [it for _, it in keyed[:n]]

try:
    import fow_rust as _fow_rust
    # See visibility.py: a namespace-package shadow (unbuilt extension) imports
    # cleanly but has no symbols. Probe both functions we call.
    _HAS_RUST = hasattr(_fow_rust, "update_own_move_rust") and hasattr(
        _fow_rust, "update_opp_move_rust"
    )
except ImportError:
    _HAS_RUST = False


class PEnumerator:
    """Maintains the set P of positions consistent with observation history.

    Eager strategy: enumerate P fully on every update. Memoryless in
    the sense that P at time t depends only on P at time t-1 + the
    update (own move or opp observation). No repair, no fallback —
    if P becomes empty (a soundness leak), updates fail loud rather
    than silently inserting positions.

    Usage::

        e = PEnumerator(chess.WHITE)
        # ... at every ply ...
        if mover_color == perspective:
            e.update_own_move(move)            # known transition
        else:
            obs = observation_from_transition(prev, nxt, perspective)
            e.update_opp_move(obs)             # filtered by obs

    Internal representation: ``self._positions`` is a set of board FEN
    strings (placement + active color + castling + ep + halfmove +
    fullmove) — same dedup key python-chess uses for board equality.

    Args:
        perspective: which color this player IS. The enumerator tracks
            P from their POV; opp moves are filtered through their
            observation.
        starting_board: the canonical game-start board (defaults to
            standard chess starting position).
        max_size: optional cap on |P|. When the post-update set exceeds
            this, we downsample to ``max_size`` uniformly at random
            (reservoir-style — every element has equal probability of
            being kept). When set, the truth-in-P guarantee is no
            longer strict: if the truth happens to be among the dropped
            positions, downstream search reasons over a P that doesn't
            include reality. Trade-off for tractability when |P|
            explodes (per A3 benchmark, real games can hit |P|>200K).
            ``None`` (default) keeps the exact-enumeration guarantee
            from A3.
        rng: deterministic RNG for downsampling. Only used when
            ``max_size`` is set. Defaults to a fresh ``random.Random()``.
    """

    def __init__(
        self,
        perspective: chess.Color,
        *,
        starting_board: chess.Board | None = None,
        max_size: int | None = None,
        rng: random.Random | None = None,
        use_rust_state: bool = False,
    ) -> None:
        self.perspective = perspective
        if starting_board is None:
            starting_board = chess.Board()
        # Resident-Rust belief set (PEnumState): keeps P packed in Rust across
        # plies so the millions of FENs never round-trip the FFI boundary or
        # get rebuilt into a Python set() each move. Only the |I| sampled roots
        # cross out (via sample_root_fens -> get_by_index). Flag-gated, default
        # OFF so the set[str] path stays byte-identical until bakeoff-validated.
        self._use_rust_state = (
            use_rust_state and _HAS_RUST and hasattr(_fow_rust, "PEnumState")
        )
        self.max_size = max_size
        # Bottom-K (KMV) bounded expansion: apply the |P| cap DURING the Rust
        # opp-move build (peak ~2*cap) instead of building the full consistent
        # set M (up to ~4.2x the cap on explosion plies) then MT-downsampling.
        # Flag-gated, default OFF (byte-identical legacy build) until bakeoff-
        # validated. Only meaningful on the rust-state path with a finite cap.
        self._bottomk = bool(os.environ.get("FOW_BOTTOMK_EXPANSION"))
        if self._use_rust_state:
            self._pstate = _fow_rust.PEnumState([starting_board.fen()])
            self._positions = None  # not maintained in this mode
            if self._bottomk and max_size is not None and hasattr(
                self._pstate, "set_bottomk_cap"
            ):
                self._pstate.set_bottomk_cap(max_size)
            else:
                self._bottomk = False
        else:
            self._pstate = None
            self._positions = {starting_board.fen()}
            self._bottomk = False
        self._rng = rng if rng is not None else random.Random()
        # Counter — incremented each time downsampling fires.
        self.downsample_count = 0
        # Per-call telemetry, set on every update_*_move. Lets the engine
        # / bakeoff strategy capture pre-dedup and pre-cap |P| sizes
        # without instrumenting inside the Rust hot path.
        #   last_raw_count: size of the Rust hot-path output BEFORE
        #     Python's set() dedup. Equals total (prev, move) pairs that
        #     survived consistency check — likely-large for opp moves,
        #     equals last_pre_cap_count for own moves (1-to-1 mapping).
        #   last_pre_cap_count: size of new P AFTER dedup, BEFORE
        #     _maybe_downsample. The "natural" |P| we'd carry if cap
        #     were infinite.
        #   last_was_downsampled: bool, True iff the cap fired this call.
        self.last_raw_count: int = 0
        self.last_pre_cap_count: int = 1
        self.last_was_downsampled: bool = False

    @property
    def positions(self) -> frozenset[str]:
        """Frozen snapshot of the current P, as board FEN strings.

        Copies the internal set on every access — safe for callers that
        need immutable-snapshot semantics (engine truth-in-P checks,
        debugging, tests). NOT recommended for downstream consumers
        that just want to iterate; use ``iter_positions()`` for that.
        """
        if self._use_rust_state:
            return frozenset(self._pstate.all_positions())
        return frozenset(self._positions)

    def iter_positions(self) -> Iterator[str]:
        """Stream over the current P without copying.

        Yields each board FEN string in P one at a time. No
        materialization beyond the existing internal set. Use this for
        downstream consumers that aggregate or filter (e.g.,
        ``lab/mining/`` puzzle-mining stats) where building a 10⁶-board
        frozenset snapshot would be wasteful.

        Mutation contract: do NOT call ``update_own_move`` /
        ``update_opp_move`` while iterating; doing so invalidates the
        iterator (RuntimeError: set changed size during iteration).

        In resident-Rust mode this decodes all packed positions to FENs
        (defeats the point of keeping P in Rust) — use ``sample_root_fens``
        on the hot path; ``iter_positions`` is for tests/debug only.
        """
        if self._use_rust_state:
            return iter(self._pstate.all_positions())
        return iter(self._positions)

    def __iter__(self) -> Iterator[str]:
        """Same as ``iter_positions()``. Lets ``for fen in enumerator:``
        work without going through the snapshot-copy ``positions``
        property."""
        return self.iter_positions()

    @property
    def uses_rust_state(self) -> bool:
        """True iff P is held resident in Rust (PEnumState) rather than as a
        Python ``set[str]``. Lets callers pick the index-sampling fast path."""
        return self._use_rust_state

    def sample_root_fens(
        self,
        *,
        n: int,
        rng: random.Random,
        weight_fn: Callable[[Sequence[str]], list[float]] | None = None,
        pool_size: int | None = None,
    ) -> list[str]:
        """Sample ``min(n, |P|)`` FENs — the |I| search roots.

        Default (``weight_fn is None``): uniform without replacement. In
        resident-Rust mode this samples indices and decodes only those positions
        (the only FENs that ever cross out of Rust), instead of
        streaming/decoding all of P. Distributionally identical to the reservoir
        sample over the full set. This path is byte-identical to before — same
        rng draws — so reproducibility/parity guards are unaffected.

        Route-B weighted path (``weight_fn`` given): draw a uniform POOL of
        ``pool_size`` FENs, score them with ``weight_fn``, and importance-resample
        ``n`` distinct roots toward high weight. Biases *which* worlds search
        sees toward plausible ones without changing the belief itself.
        """
        if weight_fn is None:
            if self._use_rust_state:
                sz = self._pstate.size()
                k = min(n, sz)
                idxs = rng.sample(range(sz), k)
                return self._pstate.get_by_index(idxs)
            pool = list(self._positions)
            k = min(n, len(pool))
            return rng.sample(pool, k)

        # Weighted: uniform pool draw, then weighted-without-replacement resample.
        sz = self.size
        pool_k = min(pool_size or n, sz)
        if self._use_rust_state:
            pool_fens = self._pstate.get_by_index(rng.sample(range(sz), pool_k))
        else:
            pool_fens = rng.sample(list(self._positions), pool_k)
        if len(pool_fens) <= n:
            return pool_fens  # pool already <= target -> nothing to resample
        weights = weight_fn(pool_fens)
        return _weighted_sample_without_replacement(pool_fens, weights, n, rng)

    @property
    def size(self) -> int:
        if self._use_rust_state:
            return self._pstate.size()
        return len(self._positions)

    def update_own_move(
        self, move: chess.Move, observation: Observation | None = None
    ) -> None:
        """Apply ``move`` (made by the perspective player) to every position
        in P. Positions where the move is not pseudo-legal are dropped.

        Two-step belief update: if ``observation`` (the perspective's view AFTER
        its own move) is given, ALSO filter P by it — pruning positions
        inconsistent with squares the move just revealed. This keeps P sound
        between the own move and the next opponent observation (the own move can
        reveal an enemy piece the prior belief didn't account for). Search-time P
        is unchanged (the next opp-observation prunes the same positions), but the
        belief is correct at every step and the next opp-enumeration starts from a
        smaller, clean P. ``observation=None`` preserves the prior apply-only
        behavior exactly.

        Raises:
            RuntimeError: if no position in P admits this move
                (soundness violation — the move couldn't have been
                played from any candidate truth).
        """
        if self._use_rust_state:
            if self._pstate.size():
                sample = chess.Board(self._pstate.get_by_index([0])[0])
                move = _canonicalize_castling(move, sample)
            pw = self.perspective == chess.WHITE
            if observation is not None:
                obs_w, obs_b = _obs_piece_bitmasks(observation)
                sz = self._pstate.update_own_move_obs(
                    pw, move.from_square, move.to_square, move.promotion or 0,
                    int(observation.visibility_mask),
                    obs_w[0], obs_w[1], obs_w[2], obs_w[3], obs_w[4], obs_w[5],
                    obs_b[0], obs_b[1], obs_b[2], obs_b[3], obs_b[4], obs_b[5],
                )
            else:
                sz = self._pstate.update_own_move(
                    pw, move.from_square, move.to_square, move.promotion or 0,
                )
            self.last_raw_count = self._pstate.last_raw_count
            self.last_pre_cap_count = self._pstate.last_pre_cap_count
            if sz == 0:
                raise RuntimeError(
                    f"P became empty after own move {move.uci()}; no candidate "
                    f"position admitted it. This is a soundness violation."
                )
            self._rust_downsample()
            self.last_was_downsampled = self._pstate.last_was_downsampled
            return

        if _HAS_RUST:
            # Canonicalize castling encoding before crossing into Rust.
            # python-chess's `move in board.pseudo_legal_moves` is fuzzy:
            # it accepts BOTH standard (e1g1) and Chess960/Shredder
            # (e1h1) castling encodings via `is_pseudo_legal`. Our Rust
            # check is direct tuple equality and only matches the
            # standard encoding that gen_pseudo_legal_moves emits.
            # Without this normalization, replaying historical game
            # files (which use e1h1) crashes with "P empty" at the
            # first castle. Live bakeoff isn't affected (strategies
            # emit standard encoding) but offline diff infra is.
            if self._positions:
                sample = chess.Board(next(iter(self._positions)))
                move = _canonicalize_castling(move, sample)
            kept, raw = _fow_rust.update_own_move_rust(
                list(self._positions),
                self.perspective == chess.WHITE,
                move.from_square,
                move.to_square,
                move.promotion or 0,
            )
            self.last_raw_count = raw
            new_positions: set[str] = set(kept)
        else:
            new_positions = set()
            for fen in self._positions:
                board = chess.Board(fen)
                if board.turn != self.perspective:
                    continue
                if move not in board.pseudo_legal_moves:
                    continue
                board.push(move)
                new_positions.add(board.fen())
            self.last_raw_count = len(new_positions)

        self.last_pre_cap_count = len(new_positions)
        if not new_positions:
            raise RuntimeError(
                f"P became empty after own move {move.uci()}; no candidate "
                f"position admitted it. This is a soundness violation."
            )
        prev_dc = self.downsample_count
        self._positions = self._maybe_downsample(new_positions)
        self.last_was_downsampled = self.downsample_count > prev_dc

    def update_opp_move(self, observation: Observation) -> None:
        """Apply an opponent move: for each p in P, enumerate opp's
        pseudo-legal moves, push each, filter by consistency with
        ``observation``.

        Raises:
            RuntimeError: if no (position, move) pair in any current p
                produces a position consistent with the observation
                (soundness violation).
        """
        opp = not self.perspective
        perspective_white = self.perspective == chess.WHITE

        if self._use_rust_state:
            obs_w, obs_b = _obs_piece_bitmasks(observation)
            obs_own_idx = (
                -1 if observation.own_capture_square is None
                else int(observation.own_capture_square)
            )
            obs_opp_idx = (
                -1 if observation.opp_capture_landing_square is None
                else int(observation.opp_capture_landing_square)
            )
            sz = self._pstate.update_opp_move(
                opp == chess.WHITE,
                perspective_white,
                int(observation.visibility_mask),
                obs_w[0], obs_w[1], obs_w[2], obs_w[3], obs_w[4], obs_w[5],
                obs_b[0], obs_b[1], obs_b[2], obs_b[3], obs_b[4], obs_b[5],
                obs_own_idx, obs_opp_idx,
            )
            self.last_raw_count = self._pstate.last_raw_count
            self.last_pre_cap_count = self._pstate.last_pre_cap_count
            if sz == 0:
                raise RuntimeError(
                    "P became empty after opp move; no (predecessor, move) pair "
                    "produced an observation-consistent position. This is a "
                    "soundness violation."
                )
            if self._bottomk:
                # Bound was applied DURING the Rust build; reflect its outcome
                # and SKIP the post-hoc MT downsample (already <= cap, a no-op).
                self.last_was_downsampled = self._pstate.last_was_downsampled
                if self.last_was_downsampled:
                    self.downsample_count += 1
                    _logger.warning(
                        "belief downsample fired (bottom-k): |P|->%d (cap=%d, "
                        "M_est=%d, total=%d)",
                        sz, self.max_size, self.last_pre_cap_count,
                        self.downsample_count,
                    )
            else:
                self._rust_downsample()
                self.last_was_downsampled = self._pstate.last_was_downsampled
            return

        if _HAS_RUST:
            obs_w, obs_b = _obs_piece_bitmasks(observation)
            obs_visibility = int(observation.visibility_mask)
            obs_own_idx = (
                -1 if observation.own_capture_square is None
                else int(observation.own_capture_square)
            )
            obs_opp_idx = (
                -1 if observation.opp_capture_landing_square is None
                else int(observation.opp_capture_landing_square)
            )
            # Rust dedups successors internally and returns (unique_fens,
            # raw_count). raw_count is the pre-dedup consistent-successor count
            # — heavy duplication here (~3x on explosion plies) is exactly what
            # used to balloon Python memory before dedup moved into Rust.
            kept, raw = _fow_rust.update_opp_move_rust(
                list(self._positions),
                opp == chess.WHITE,
                perspective_white,
                obs_visibility,
                obs_w[0], obs_w[1], obs_w[2], obs_w[3], obs_w[4], obs_w[5],
                obs_b[0], obs_b[1], obs_b[2], obs_b[3], obs_b[4], obs_b[5],
                obs_own_idx, obs_opp_idx,
            )
            self.last_raw_count = raw
            new_positions: set[str] = set(kept)
        else:
            new_positions = set()
            raw = 0
            for fen in self._positions:
                prev = chess.Board(fen)
                if prev.turn != opp:
                    continue
                for move in prev.pseudo_legal_moves:
                    nxt = prev.copy()
                    nxt.push(move)
                    if consistent_with(nxt, prev, observation, self.perspective):
                        raw += 1
                        new_positions.add(nxt.fen())
            self.last_raw_count = raw

        self.last_pre_cap_count = len(new_positions)
        if not new_positions:
            raise RuntimeError(
                "P became empty after opp move; no (predecessor, move) pair "
                "produced an observation-consistent position. This is a "
                "soundness violation."
            )
        prev_dc = self.downsample_count
        self._positions = self._maybe_downsample(new_positions)
        self.last_was_downsampled = self.downsample_count > prev_dc

    def _rust_downsample(self) -> None:
        """Cap |P| to max_size IN RUST, deterministically, by transplanting this
        enumerator's MT state into Rust's CPython-faithful MT (so eviction is
        reproducible from the Python RNG). No-op when uncapped or |P| fits.

        The cap is a tractability fallback that effectively never fires at the
        production p_max (observed max |P| ~950K << 5M). Exact RNG-stream parity
        with the set-path ``_rng.sample`` is not preserved — order is already
        per-process nondeterministic — so we just advance ``_rng`` afterward to
        avoid replaying the same draws on a subsequent cap-fire."""
        if self.max_size is None or self._pstate.size() <= self.max_size:
            return
        pre = self._pstate.size()
        state = self._rng.getstate()[1]
        mt_words = list(state[:624])
        mt_index = state[624]
        self._pstate.downsample(self.max_size, mt_words, mt_index)
        self.downsample_count += 1
        self._rng.getrandbits(64)  # nudge the stream forward
        _logger.warning(
            "belief downsample fired (rust): |P| %d -> %d (cap=%d, total=%d)",
            pre, self.max_size, self.max_size, self.downsample_count,
        )

    def _maybe_downsample(self, positions: set[str]) -> set[str]:
        """If max_size is set and |positions| > max_size, uniformly
        downsample. Otherwise return positions unchanged."""
        if self.max_size is None or len(positions) <= self.max_size:
            return positions
        # random.sample on a set converts to list internally; we do the
        # same explicitly so the conversion is visible.
        kept = self._rng.sample(list(positions), self.max_size)
        self.downsample_count += 1
        _logger.warning(
            "belief downsample fired (python): |P| %d -> %d (cap=%d, total=%d)",
            len(positions), self.max_size, self.max_size, self.downsample_count,
        )
        return set(kept)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, fen: str) -> bool:
        if self._use_rust_state:
            return fen in self._pstate.all_positions()
        return fen in self._positions


def _canonicalize_castling(move: chess.Move, sample_board: chess.Board) -> chess.Move:
    """Normalize king→rook castling encoding (e.g., e1h1) into the
    king→destination encoding (e.g., e1g1) that python-chess's standard
    pseudo_legal_moves emits. Only rewrites when the king is actually
    on its starting square in the sample board (which means it's on
    that square in EVERY P entry at this ply, since own piece positions
    are deterministic). Pass-through for all other moves."""
    if move.promotion or move.drop:
        return move
    fs, ts = move.from_square, move.to_square
    king_mask = sample_board.kings & sample_board.occupied_co[sample_board.turn]
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


def _obs_piece_bitmasks(observation: Observation) -> tuple[list[int], list[int]]:
    """Extract observation.visible_pieces into two 6-element bitmask lists
    indexed by (piece_type - 1): [pawn, knight, bishop, rook, queen, king].
    Returned as (white_masks, black_masks). One-time cost per
    update_opp_move call — avoids per-(prev, move) dict iteration."""
    obs_w = [0] * 6
    obs_b = [0] * 6
    for sq, piece in observation.visible_pieces.items():
        bb = 1 << sq
        if piece.color:
            obs_w[piece.piece_type - 1] |= bb
        else:
            obs_b[piece.piece_type - 1] |= bb
    return obs_w, obs_b
