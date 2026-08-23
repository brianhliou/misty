"""Stockfish-backed leaf evaluator for CFR over Fog of War chess.

Replaces ``hybrid_fog_leaf_eval`` with a perfect-information chess engine.
Obscuro (Zhang & Sandholm 2026) showed that Stockfish — despite knowing
nothing about FoW — works better than a hand-tuned FoW eval as the leaf
evaluator inside KLUSS subgame solving. Their ablation: hand-tuned eval
costs ~20pp vs Stockfish at the same compute budget. Per Obscuro:
"regular chess is not so different from FoW chess in terms of what
positions are good or bad."

The class manages a single Stockfish subprocess via
``chess.engine.SimpleEngine``. Spawning Stockfish per call is expensive
(~50ms cold start), so callers should reuse one instance for many
evaluations.

Two evaluation modes:

* ``evaluate(board, perspective)`` returns a single value in [-1, 1].
  Drop-in replacement for the existing ``leaf_eval`` interface
  (material, hybrid_fog).
* ``evaluate_children(board, perspective)`` returns ``{Move: float}``
  for every legal move using MultiPV at depth 1 — Obscuro's batched
  evaluation pattern. This is the call shape GT-CFR (Phase A4) will
  use when expanding a leaf and adding all its children at once.

Note on FoW semantics: Stockfish evaluates as standard chess
(check restrictions enforced, no king-capture). FoW has no check
restriction, so legal-move counts differ. ``evaluate_children`` only
returns values for moves Stockfish considers legal — FoW-legal-but-
chess-illegal moves (e.g., walking through check) get no Stockfish
evaluation. Callers must handle the gap (e.g., fall back to a cheap
heuristic, or skip those moves in expansion).
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from types import TracebackType

import chess
import chess.engine

from .leaf_eval import (
    king_aware_leaf_enabled,
    king_capture_imminent,
    material_leaf_eval,
    tanh_scale_cp as _current_tanh_scale_cp,
)


logger = logging.getLogger(__name__)


# Stockfish hash size. At depth=1 (the only depth we use for leaf eval),
# the transposition table is barely touched — 16 MB was historical safety
# margin; 1 MB has proven sufficient in production. Lower default saves
# init cost + container RSS without affecting per-call quality.
_DEFAULT_HASH_MB = 1
_DEFAULT_THREADS = 1
_DEFAULT_TANH_SCALE_CP = 500.0  # legacy default; new code inherits via _TANH_SCALE_CP_INHERIT
# Sentinel meaning "inherit the current process-wide scale at construct time"
# (FOW_TANH_SCALE_CP env var + set_tanh_scale_cp runtime setter). Pass an
# explicit number to override.
_TANH_SCALE_CP_INHERIT = -1.0
_MATE_SCORE_CP = 10_000  # converts mate to a large centipawn value for normalization
_DEFAULT_CACHE_SIZE = 100_000


def _find_stockfish() -> str:
    """Locate the Stockfish binary.

    Resolution order: the ``FOW_STOCKFISH`` env var (an absolute path or a
    name on PATH), then ``stockfish`` on PATH. The env var is the robust seam
    for environments where the binary isn't named ``stockfish`` or isn't on
    PATH — e.g. the cloud worker at ``/usr/games/stockfish`` — and it reaches
    this evaluator even when a caller's ``--stockfish`` flag doesn't thread all
    the way into the engine's leaf-eval config.
    """
    env_path = os.environ.get("FOW_STOCKFISH")
    if env_path:
        resolved = shutil.which(env_path) or (
            env_path if os.path.isfile(env_path) and os.access(env_path, os.X_OK)
            else None
        )
        if resolved is None:
            raise FileNotFoundError(
                f"FOW_STOCKFISH={env_path!r} does not resolve to an executable "
                "(not on PATH and not an executable file)."
            )
        return resolved
    path = shutil.which("stockfish")
    if path is None:
        raise FileNotFoundError(
            "Stockfish binary not found on PATH. Install via Homebrew "
            "(`brew install stockfish`), set the FOW_STOCKFISH env var to the "
            "binary path, or pass the `path` argument to StockfishLeafEval."
        )
    return path


def _score_to_eval(
    info_score: chess.engine.PovScore,
    perspective: chess.Color,
    tanh_scale_cp: float,
) -> float:
    """Convert a python-chess PovScore to a tanh-normalized [-1, 1] value
    from ``perspective``'s POV.

    Mate scores are mapped to ±_MATE_SCORE_CP before tanh — they saturate
    near ±1 cleanly.
    """
    pov = info_score.pov(perspective)
    cp = pov.score(mate_score=_MATE_SCORE_CP)
    if cp is None:
        return 0.0
    return math.tanh(cp / tanh_scale_cp)


class StockfishLeafEval:
    """Persistent Stockfish process used as a CFR leaf evaluator.

    Use as a context manager to guarantee subprocess cleanup, or call
    ``close()`` explicitly. Not thread-safe — give each CFR worker its
    own instance.

    Args:
        path: Path to the Stockfish binary. Defaults to the first
            ``stockfish`` on ``PATH``.
        hash_mb: Stockfish hash table size. 16 MB is the Stockfish
            default and is sufficient at depth 1.
        threads: Stockfish thread count. 1 is sufficient at depth 1.
        tanh_scale_cp: Centipawn divisor inside ``tanh`` normalization.
            500 matches the existing ``material_leaf_eval`` convention
            (rook advantage ≈ 0.76, queen advantage ≈ 0.95).
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        hash_mb: int = _DEFAULT_HASH_MB,
        threads: int = _DEFAULT_THREADS,
        tanh_scale_cp: float = _TANH_SCALE_CP_INHERIT,
        cache_size: int = _DEFAULT_CACHE_SIZE,
        use_lean: bool | None = None,
    ) -> None:
        self.path = str(path) if path is not None else _find_stockfish()
        # Byte-parity lean UCI path (FOW_LEAN_UCI). None = read env; explicit
        # bool overrides (per-arm bakeoff wiring). Byte-identical to python-chess,
        # so flipping it on is a pure-throughput change with no strength delta.
        self.use_lean = _lean_uci_enabled() if use_lean is None else use_lean
        # Inherit the current process-wide tanh scale (env var + runtime
        # setter) unless the caller passed an explicit override. Keeps the
        # two leaf-eval paths (material + Stockfish) on the same scale.
        self.tanh_scale_cp = (
            _current_tanh_scale_cp()
            if tanh_scale_cp == _TANH_SCALE_CP_INHERIT
            else tanh_scale_cp
        )
        self.hash_mb = hash_mb
        self.threads = threads
        # Counters for observability — incremented when Stockfish rejects
        # a position or when we restart the engine after a crash.
        self.fallback_count = 0
        self.restart_count = 0
        # King-aware shim: `king_capture_hits` fires when the narrow rule
        # (side-to-move can capture opp's king next ply) returns ±1.0. With
        # FOW_KING_AWARE_LEAF=1 enabled, fallback_count is then strictly the
        # invalid-but-narrow-rule-didn't-fire population — broad-vs-narrow
        # mining signal for whether the narrow rule misses any real cases.
        self.king_capture_hits = 0
        # Position cache. Key = (epd, perspective) — epd() drops the
        # halfmove/fullmove counters that don't matter at depth=1 search,
        # maximizing hit rate across paths-to-same-position. Per-instance
        # so memory stays bounded to one game / one Stockfish process.
        # Stockfish at depth=1 is deterministic, so cached evals are
        # exact replays of what Stockfish would return.
        self._cache_size = cache_size
        self._eval_cache: OrderedDict[tuple, float] = OrderedDict()
        self._children_cache: OrderedDict[
            tuple, dict[chess.Move, float]
        ] = OrderedDict()
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0
        self.children_cache_hits = 0
        self.children_cache_misses = 0
        # Cumulative wall-time in evaluate() + evaluate_children() in ns.
        # Read as a snapshot; callers (engine_v2.choose_move) take a delta
        # across a pick_move to attribute Stockfish leaf-eval cost. Always-on
        # because the overhead is one time.monotonic_ns() per call.
        self.eval_wall_ns = 0
        self.children_wall_ns = 0
        self._spawn_engine()

    def _spawn_engine(self) -> None:
        if self.use_lean:
            self._engine = _FaithfulUCIClient(self.path, self.hash_mb, self.threads)
        else:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"Hash": self.hash_mb, "Threads": self.threads})
        self._closed = False

    def _restart_engine(self) -> None:
        try:
            self._engine.quit()  # both SimpleEngine and _FaithfulUCIClient expose quit()
        except Exception:
            pass
        self._spawn_engine()
        self.restart_count += 1

    def __enter__(self) -> "StockfishLeafEval":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        # Idempotent: a second close must not write to the dead pipe at all —
        # a swallowed BrokenPipeError still leaves a buffered write that fails
        # again (unraisably) when the transport's file objects are finalized.
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._engine.quit()
        except (chess.engine.EngineError, OSError):
            pass

    def evaluate(
        self,
        board: chess.Board,
        perspective: chess.Color,
    ) -> float:
        """Stockfish position evaluation at depth 1, tanh-normalized to [-1, 1]
        from ``perspective``'s POV.

        Fog of War positions can be standard-chess-invalid (kings walked
        through check, multiple-king states, etc.). If Stockfish rejects
        or chokes on a position, this falls back to ``material_leaf_eval``
        and restarts the Stockfish subprocess. The ``fallback_count``
        attribute records how often this fired so callers can audit how
        much of their CFR traversal actually got Stockfish vs material.

        Implementation note: python-chess serializes the board to UCI as
        ``position startpos moves <move-stack>`` when the board has
        history, which forces Stockfish to replay all FoW-illegal moves
        and frequently corrupts its state. We send a fresh
        ``chess.Board(fen)`` to bypass move history — Stockfish receives
        ``position fen <fen>`` and evaluates the position directly.
        """
        _t0 = time.monotonic_ns()
        try:
            return self._evaluate(board, perspective)
        finally:
            self.eval_wall_ns += time.monotonic_ns() - _t0

    def _evaluate(
        self,
        board: chess.Board,
        perspective: chess.Color,
    ) -> float:
        if king_aware_leaf_enabled():
            v = king_capture_imminent(board, perspective)
            if v is not None:
                self.king_capture_hits += 1
                return v
        if not board.is_valid():
            self.fallback_count += 1
            return material_leaf_eval(board, perspective)
        # Cache lookup: position-only EPD (no move counters) + perspective.
        cache_key = (board.epd(), perspective)
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            self._eval_cache.move_to_end(cache_key)
            self.eval_cache_hits += 1
            return cached
        self.eval_cache_misses += 1
        # Lever 4 (Stockfish per-call): bypass move history without the
        # board.fen() → chess.Board(fen) round-trip. board.copy(stack=False)
        # drops move_stack so python-chess engine sends `position fen <fen>`
        # (not `position startpos moves ...`), but avoids the FEN
        # serialize+reparse. Same Stockfish input; less Python work.
        fen_board = board.copy(stack=False)
        try:
            if self.use_lean:
                # en_passant="fen" matches UciProtocol._position's FEN exactly.
                raw = self._engine.single_eval(fen_board.fen(en_passant="fen"))
                if raw is None:
                    result = 0.0  # no scored line (mate/stalemate); python-chess
                    # returns empty info → score().score() is None → _score_to_eval 0.0
                else:
                    result = _lean_score_to_eval(
                        raw[0], raw[1], board.turn, perspective, self.tanh_scale_cp)
            else:
                info = self._engine.analyse(
                    fen_board,
                    chess.engine.Limit(depth=1, time=0.5),
                )
                result = _score_to_eval(info["score"], perspective, self.tanh_scale_cp)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError,
                chess.IllegalMoveError) as exc:
            logger.debug("stockfish eval failed on %s: %s", board.fen(), exc)
            self.fallback_count += 1
            self._restart_engine()
            return material_leaf_eval(board, perspective)
        self._eval_cache[cache_key] = result
        if len(self._eval_cache) > self._cache_size:
            self._eval_cache.popitem(last=False)
        return result

    def evaluate_children(
        self,
        board: chess.Board,
        perspective: chess.Color,
    ) -> dict[chess.Move, float]:
        _t0 = time.monotonic_ns()
        try:
            return self._evaluate_children(board, perspective)
        finally:
            self.children_wall_ns += time.monotonic_ns() - _t0

    def _evaluate_children(
        self,
        board: chess.Board,
        perspective: chess.Color,
    ) -> dict[chess.Move, float]:
        """MultiPV evaluation at depth 1 of every chess-legal move.

        Returns ``{move: eval}`` where ``eval`` is the post-move position's
        tanh-normalized score from ``perspective``'s POV (NOT the score
        for the side now to move). Callers expanding a leaf in GT-CFR
        will index this dict by child move.

        Moves that are FoW-legal but chess-illegal are not in the result.
        On Stockfish error this returns an empty dict; the caller can
        fall back to per-child ``evaluate`` calls (with their own
        material fallback) if needed.
        """
        if not board.is_valid():
            self.fallback_count += 1
            return {}
        n_moves = board.legal_moves.count()
        if n_moves == 0:
            return {}
        # Cache lookup: same key strategy as evaluate(). MultiPV result
        # for a position is deterministic at depth=1, so the cached dict
        # is exact. Hot in practice — the same FEN appears at multiple
        # tree leaves across CFR iterations and across pick_move calls
        # in a game.
        cache_key = (board.epd(), perspective)
        cached = self._children_cache.get(cache_key)
        if cached is not None:
            self._children_cache.move_to_end(cache_key)
            self.children_cache_hits += 1
            return cached
        self.children_cache_misses += 1
        # Lever 4: same FEN-roundtrip avoidance as evaluate(); board.copy(
        # stack=False) preserves position + castling rights + ep + clocks but
        # drops move history so python-chess engine sends `position fen ...`.
        fen_board = board.copy(stack=False)
        out: dict[chess.Move, float] = {}
        try:
            if self.use_lean:
                raw = self._engine.children_eval(
                    fen_board.fen(en_passant="fen"), n_moves)
                out = {
                    chess.Move.from_uci(mv): _lean_score_to_eval(
                        kind, val, board.turn, perspective, self.tanh_scale_cp)
                    for mv, (kind, val) in raw.items()
                }
            else:
                info_list = self._engine.analyse(
                    fen_board,
                    chess.engine.Limit(depth=1, time=0.5),
                    multipv=n_moves,
                )
                for info in info_list:
                    pv = info.get("pv") or []
                    if not pv:
                        continue
                    move = pv[0]
                    out[move] = _score_to_eval(
                        info["score"], perspective, self.tanh_scale_cp
                    )
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError,
                chess.IllegalMoveError) as exc:
            logger.debug("stockfish multipv failed on %s: %s", board.fen(), exc)
            self.fallback_count += 1
            self._restart_engine()
            return {}
        self._children_cache[cache_key] = out
        if len(self._children_cache) > self._cache_size:
            self._children_cache.popitem(last=False)
        return out


def _mate_to_cp(val: int) -> int:
    """Match python-chess Mate.score(mate_score=_MATE_SCORE_CP): a "mate in N"
    UCI score maps to (M - N) when mating, (-M - N) when being mated. Keeps the
    lean client byte-identical to SimpleEngine.analyse, including mate scores."""
    return (_MATE_SCORE_CP - val) if val > 0 else (-_MATE_SCORE_CP - val)


def _lean_score_to_eval(kind: str, val: int, board_turn: chess.Color,
                        perspective: chess.Color, tanh_scale_cp: float) -> float:
    """Convert a raw UCI score (side-to-move POV) to a tanh-normalized eval from
    ``perspective``'s POV — the lean-client equivalent of ``_score_to_eval``."""
    cp = _mate_to_cp(val) if kind == "mate" else val
    if board_turn != perspective:
        cp = -cp
    return math.tanh(cp / tanh_scale_cp)


def _lean_uci_enabled() -> bool:
    """Process-wide toggle for the byte-parity lean UCI leaf-eval path.

    DEFAULT ON since 2026-05-29 (flipped after the 962-comparison byte-parity gate
    in tests/test_lean_uci_parity.py): ``_FaithfulUCIClient`` is bit-identical to
    python-chess ``SimpleEngine.analyse`` but skips its asyncio + full-PV parse
    overhead (~1.9x/eval). Opt OUT with ``FOW_LEAN_UCI=0`` to fall back to
    python-chess (the escape hatch). Byte-identical, so the flip changes throughput
    only, not strength; only the v2 path uses StockfishLeafEval (legacy v0.9.5 does
    not), so production serving is unaffected.
    """
    return os.environ.get("FOW_LEAN_UCI", "").strip().lower() not in ("0", "false", "no", "off")


class _FaithfulUCIClient:
    """Synchronous UCI client byte-identical to ``chess.engine.SimpleEngine.analyse``
    for depth-1 (single + MultiPV) leaf eval, minus the asyncio loop and the
    full-PV parse python-chess does per info line.

    Byte-parity is load-bearing (a prior lean attempt broke it by sending
    ``ucinewgame`` per eval, clearing Stockfish's hash → different depth-1 evals;
    see memory ``leaf-eval-pool-findings-2026-05-27``). This client mirrors
    python-chess's *exact* command stream (captured empirically 2026-05-29):

      init:  uci / uciok → setoption Hash → setoption Threads → ucinewgame → isready / readyok
      eval:  [setoption MultiPV value N  — only when N changes]
             position startpos | position fen <fen>      (en_passant="fen" style)
             go depth 1 movetime 500
             read `info` lines until `bestmove`

    The single ``ucinewgame`` at init (then never again) reproduces python-chess's
    first-game-only behaviour, so the transposition table evolves identically →
    identical depth-1 scores over the *same ordered position sequence*. Parses
    ONLY ``score cp|mate <v>`` and the first ``pv`` token (the move); never builds
    a Board for the discarded PV. NOT thread-safe — one client per worker.

    Contrast: a pooled client would DELIBERATELY send ``ucinewgame`` per eval
    for pool-order determinism — that's the parity-breaker, correct for a pool,
    wrong here.
    """

    def __init__(self, path: str, hash_mb: int, threads: int) -> None:
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        # Leaf search depth (FOW_SF_LEAF_DEPTH, default 1 = the shipped leaf). Bumping
        # it trades throughput for eval quality — used by the leaf-sensitivity probe
        # to ask whether a deeper leaf changes the engine's move (lab/leaf_sensitivity.py).
        self._leaf_depth = int(os.environ.get("FOW_SF_LEAF_DEPTH", "1"))
        self._send("uci"); self._wait("uciok")
        # python-chess SimpleEngine.configure sends these (Hash != SF default 16 →
        # sent; Threads == default 1 → python-chess dedupes, but a redundant
        # Threads=1 is a no-op so we send both harmlessly). Engine state is identical.
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name Threads value {threads}")
        # python-chess emits ucinewgame + isready exactly once (first analyse,
        # first_game=True). Do it here so the TT is clean once; NEVER again.
        self._send("ucinewgame")
        self._send("isready"); self._wait("readyok")
        # python-chess tracks MultiPV from the parsed option default (1 for SF) and
        # only re-sends setoption when it changes. Start at 1 to match the dedupe.
        self._multipv = 1

    def _send(self, s: str) -> None:
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _wait(self, token: str) -> None:
        for line in self.p.stdout:
            parts = line.split()
            if parts and parts[0] == token:
                return
        raise chess.engine.EngineTerminatedError("stockfish closed before %r" % token)

    def _set_multipv(self, n: int) -> None:
        if n != self._multipv:
            self._send(f"setoption name MultiPV value {n}")
            self._multipv = n

    @staticmethod
    def _position_line(fen: str) -> str:
        # Mirrors UciProtocol._position: `position startpos` for the start FEN,
        # else `position fen <fen>`. Callers pass fen(en_passant="fen").
        return "position startpos" if fen == chess.STARTING_FEN else f"position fen {fen}"

    def _go_read(self) -> dict[int, tuple[str | None, str, int]]:
        """Send ``go depth 1 movetime 500`` and read ``info`` lines until
        ``bestmove``. Return ``{multipv_index: (move_uci, score_kind, score_val)}``,
        keeping the LAST scored line per index (python-chess merges info per index;
        the final depth-1 line wins)."""
        self._send(f"go depth {self._leaf_depth} movetime 500")
        per_mpv: dict[int, tuple[str | None, str, int]] = {}
        saw_bestmove = False
        for line in self.p.stdout:
            if line.startswith("bestmove"):
                saw_bestmove = True
                break
            if not line.startswith("info "):
                continue
            t = line.split()
            if "score" not in t:  # e.g. `info depth 1 currmove ...` (no score)
                continue
            try:
                si = t.index("score")
                kind, val = t[si + 1], int(t[si + 2])
                # MultiPV=1 -> SF omits the `multipv` token; treat as index 1.
                mpv = int(t[t.index("multipv") + 1]) if "multipv" in t else 1
                move = t[t.index("pv") + 1] if "pv" in t else None
            except (ValueError, IndexError):
                continue
            per_mpv[mpv] = (move, kind, val)
        if not saw_bestmove:
            raise chess.engine.EngineTerminatedError("stockfish closed mid-search")
        return per_mpv

    def single_eval(self, fen: str) -> tuple[str, int] | None:
        """Raw side-to-move score ``(kind, val)`` for the position at depth 1,
        or None if Stockfish returned no scored line (e.g. checkmate/stalemate —
        python-chess returns an empty info dict there too)."""
        self._set_multipv(1)
        self._send(self._position_line(fen))
        per_mpv = self._go_read()
        if not per_mpv:
            return None
        # python-chess single analyse = merged info; last scored line wins. With
        # MultiPV=1 that's index 1 (or the lone entry if SF labelled it otherwise).
        entry = per_mpv.get(1) or next(reversed(per_mpv.values()))
        return (entry[1], entry[2])

    def children_eval(self, fen: str, n_moves: int) -> dict[str, tuple[str, int]]:
        """``{move_uci: (kind, val)}`` for the depth-1 MultiPV pass (one entry per
        chess-legal move Stockfish returns), matching ``analyse(multipv=n)``."""
        self._set_multipv(n_moves)
        self._send(self._position_line(fen))
        per_mpv = self._go_read()
        return {mv: (kind, val) for (mv, kind, val) in per_mpv.values() if mv is not None}

    def quit(self) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._send("quit"); self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def stockfish_leaf_eval_factory(
    **kwargs,
):
    """Convenience: build a closure suitable for tabular CFR's
    ``leaf_eval`` parameter, plus a teardown handle.

    Returns ``(eval_fn, eval_instance)``. The caller must call
    ``eval_instance.close()`` (or use it as a context manager elsewhere)
    when done.

    Usage::

        eval_fn, sf = stockfish_leaf_eval_factory()
        try:
            soln = solve_subgame(root, leaf_eval=eval_fn, depth=3, iters=100)
        finally:
            sf.close()
    """
    sf = StockfishLeafEval(**kwargs)
    return sf.evaluate, sf
