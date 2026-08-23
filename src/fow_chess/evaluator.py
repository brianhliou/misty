"""Evaluators for Tier-1 fog-of-war engines.

Evaluators exposed:

- `material_evaluator()` — fast, pure-Python centipawn material balance.
  Always usable, no external dependencies, no quirks. The Tier-1 baseline
  uses this.

- `visibility_threat_evaluator(threat_lambda)` — material minus threats from
  visible opp pieces only (observed truth, no particle aggregation). Builder
  form; closes over `PerspectiveView` per move.

- `stockfish_evaluator(...)` — Stockfish via raw UCI subprocess. Sends
  positions, parses `info`-line scores, never reads or validates `bestmove`
  — that's the failure path in python-chess's wrapper, which rejects moves
  Stockfish emits for FOW positions where side-to-move is in check (FOW
  doesn't enforce check escape). Falls back to material on timeout or
  subprocess errors.

- `fow_evaluator(...)` — FoW-native evaluator: material + piece safety +
  king pressure + visibility advantage + fog risk. Operates on the full
  particle board (no subprocess, ~0.5ms/call). Designed as the leaf
  evaluator for MCTS rollouts; also usable as a drop-in for the current
  1-ply architecture.
"""

from __future__ import annotations

import os
import pty
import select
import subprocess
import time
from contextlib import contextmanager
from typing import Iterator

import chess

from .engine import Evaluator, EvaluatorBuilder
from .selfplay import PerspectiveView

_KING_CAPTURE_SCORE = 100_000.0  # Bigger than any centipawn eval Stockfish returns.

_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # Kings handled by the king-capture short-circuit.
}


def material_score(board: chess.Board, perspective: chess.Color) -> float:
    """Centipawn material balance from `perspective`'s POV."""
    total = 0
    for piece in board.piece_map().values():
        sign = 1 if piece.color == perspective else -1
        total += sign * _PIECE_VALUES[piece.piece_type]
    return float(total)


def fog_discount_term(board: chess.Board, perspective: chess.Color) -> float:
    """Static fog-discount penalty: sum over `perspective`'s non-king pieces
    in opp territory of (depth into enemy) × (1 if undefended) × piece_value.

    Captures the FOW-implicit risk that an exposed piece in opp territory
    could be captured by a hidden attacker we don't see. Independent of
    particle hypotheses — doesn't dilute under uniform-prior dispersion the
    way per-particle threat aggregation does.

    Depth: rank distance past midfield (1-4). Pieces on our own half
    contribute zero. Defended-ness checked against own-color attackers
    targeting the piece's square (recapture availability).
    """
    penalty = 0.0
    for square, piece in board.piece_map().items():
        if piece.color != perspective:
            continue
        if piece.piece_type == chess.KING:
            continue
        rank = chess.square_rank(square)
        if perspective == chess.WHITE:
            depth = max(0, rank - 3)
        else:
            depth = max(0, 4 - rank)
        if depth == 0:
            continue
        if board.attackers(perspective, square):
            continue
        penalty += depth * _PIECE_VALUES[piece.piece_type]
    return penalty


def king_safety_evaluator(base: Evaluator) -> Evaluator:
    """Wrap an evaluator with a game-loss-magnitude penalty for own-king-attacked.

    Stockfish-shallow evaluates each particle's post-move board under standard-
    chess rules: opp can't capture the king (only checkmate ends the game), so
    "own king attacked by enemy piece" returns a near-neutral score. Under FOW
    the next move *will* capture the king. This wrapper bakes the FOW rule in:
    any (move, particle) cell that leaves own king attacked by an enemy piece
    on that particle's board is penalized by ½×_KING_CAPTURE_SCORE — large
    enough to dominate normal eval, small enough that king-capture (a true
    +_KING_CAPTURE_SCORE) still beats it.

    Per particle. So aggregation across the belief weights the penalty by
    particle support — if 30% of particles put a hidden bishop attacking our
    king after move M, M's mean score takes a 30% × ½K hit. Moves that resolve
    the threat on those particles aren't penalized there; aggregation surfaces
    them.

    Companion to the visible-king-defense short-circuit. Defense fires when the
    threat is visible (cheap, deterministic). This wrapper handles the
    *believed-but-not-visible* threat case the corpus surfaced in v0.4.0.
    """

    def evaluate(
        board: chess.Board, move: chess.Move, perspective: chess.Color
    ) -> float:
        target = board.piece_at(move.to_square)
        if (
            target is not None
            and target.color != perspective
            and target.piece_type == chess.KING
        ):
            return base(board, move, perspective)

        base_score = base(board, move, perspective)
        advanced = board.copy()
        advanced.push(move)
        own_king = advanced.king(perspective)
        if own_king is None:
            return base_score
        if advanced.attackers(not perspective, own_king):
            return min(base_score - _KING_CAPTURE_SCORE, -_KING_CAPTURE_SCORE / 2)
        return base_score

    return evaluate


def fog_aware_evaluator(base: Evaluator, fog_lambda: float) -> Evaluator:
    """Wrap `base` with a fog-discount penalty on the post-move position.

    The wrapped evaluator returns base_score - fog_lambda * fog_discount_term.
    King-capture scores from the base (±_KING_CAPTURE_SCORE) pass through
    unchanged — winning the game outweighs any exposure penalty.
    """

    def evaluate(
        board: chess.Board, move: chess.Move, perspective: chess.Color
    ) -> float:
        base_score = base(board, move, perspective)
        if abs(base_score) >= _KING_CAPTURE_SCORE / 2:
            return base_score
        advanced = board.copy()
        advanced.push(move)
        if advanced.king(chess.WHITE) is None or advanced.king(chess.BLACK) is None:
            return base_score
        return base_score - fog_lambda * fog_discount_term(advanced, perspective)

    return evaluate


def material_evaluator() -> Evaluator:
    """Evaluator that scores a candidate move by post-move material balance.

    Includes the king-capture short-circuit so Tier-1 always grabs an
    available king capture without needing Stockfish.
    """

    def evaluate(
        board: chess.Board, move: chess.Move, perspective: chess.Color
    ) -> float:
        target = board.piece_at(move.to_square)
        if target is not None and target.piece_type == chess.KING:
            return (
                _KING_CAPTURE_SCORE
                if target.color != perspective
                else -_KING_CAPTURE_SCORE
            )

        advanced = board.copy()
        advanced.push(move)
        return material_score(advanced, perspective)

    return evaluate


def threat_aware_evaluator(threat_lambda: float = 0.3) -> Evaluator:
    """Material balance minus a discount for `perspective`'s pieces opp can capture.

    For each candidate post-move position, sums the values of `perspective`'s
    non-king pieces that opp's pseudo-legal moves could capture (each piece
    counted once even if multiply attacked, since opp only captures one per
    turn). Subtracts `threat_lambda * threatened_value` from the material
    balance. With a lambda around 0.3, this approximates the expected loss
    from a moderately-active opponent without over-penalizing every threat.

    Threat counting depends on opponent piece positions, which differ across
    belief particles — this is the first evaluator that returns
    particle-dependent scores for non-capture moves, making per-particle
    voting and risk_aversion meaningful.
    """

    def evaluate(
        board: chess.Board, move: chess.Move, perspective: chess.Color
    ) -> float:
        target = board.piece_at(move.to_square)
        if target is not None and target.piece_type == chess.KING:
            return (
                _KING_CAPTURE_SCORE
                if target.color != perspective
                else -_KING_CAPTURE_SCORE
            )

        advanced = board.copy()
        advanced.push(move)
        if advanced.king(chess.WHITE) is None or advanced.king(chess.BLACK) is None:
            return 0.0

        base = material_score(advanced, perspective)

        # advanced.turn is now opp; iterate their pseudo-legal moves and
        # collect the squares of `perspective`'s pieces under attack.
        threatened_squares: set[int] = set()
        for m in advanced.pseudo_legal_moves:
            tgt = advanced.piece_at(m.to_square)
            if tgt is not None and tgt.color == perspective and tgt.piece_type != chess.KING:
                threatened_squares.add(m.to_square)

        threat_value = sum(
            _PIECE_VALUES[advanced.piece_at(sq).piece_type] for sq in threatened_squares
        )
        return base - threat_lambda * threat_value

    return evaluate


def visibility_threat_evaluator(threat_lambda: float = 0.3) -> EvaluatorBuilder:
    """Material balance minus a threat discount counted from visible opp pieces only.

    Counter to `threat_aware_evaluator`, which counts threats from every
    particle's hypothesized opp positions (and hallucinates because particles
    disperse opp pieces across many plausible squares), this builder uses the
    PerspectiveView's `visible_piece_map` to count threats only from opp
    pieces the perspective actually sees.

    Implementation: per particle, evaluate as

        base = material_score(advanced, perspective)
        threats = pseudo-legal-moves on a synthetic board containing
                  (own pieces post-move) ∪ (visible opp pieces still on the
                  board after our move)
        score = base - threat_lambda * sum(threatened own piece values)

    Material balance is still computed on the particle (so capture moves on
    hidden squares retain particle sensitivity). Threat counting uses observed
    truth and ignores hidden hypothesized opp pieces, sidestepping the
    hallucination problem.

    Per-particle voting becomes near-degenerate with this evaluator for non-
    capture moves — that's intentional. The structural fix for belief-noise-
    driven heuristics is to not aggregate over noisy particles; aggregate
    over observed truth instead.
    """

    def build(view: PerspectiveView) -> Evaluator:
        perspective = view.perspective
        visible_opp_pieces: dict[chess.Square, chess.Piece] = {
            sq: piece
            for sq, piece in view.visible_piece_map.items()
            if piece.color != perspective
        }

        def evaluate(
            board: chess.Board, move: chess.Move, perspective_: chess.Color
        ) -> float:
            target = board.piece_at(move.to_square)
            if target is not None and target.piece_type == chess.KING:
                return (
                    _KING_CAPTURE_SCORE
                    if target.color != perspective_
                    else -_KING_CAPTURE_SCORE
                )

            advanced = board.copy()
            advanced.push(move)
            if (
                advanced.king(chess.WHITE) is None
                or advanced.king(chess.BLACK) is None
            ):
                return 0.0

            base = material_score(advanced, perspective_)

            visibility_board = chess.Board.empty()
            for sq in chess.SquareSet(advanced.occupied_co[perspective_]):
                piece = advanced.piece_at(sq)
                if piece is not None:
                    visibility_board.set_piece_at(sq, piece)
            for sq, piece in visible_opp_pieces.items():
                if advanced.piece_at(sq) is None:
                    continue  # captured by `move`, no longer threatens
                visibility_board.set_piece_at(sq, piece)
            visibility_board.turn = not perspective_

            threatened: set[int] = set()
            for m in visibility_board.pseudo_legal_moves:
                tgt = visibility_board.piece_at(m.to_square)
                if (
                    tgt is not None
                    and tgt.color == perspective_
                    and tgt.piece_type != chess.KING
                ):
                    threatened.add(m.to_square)
            threat_value = sum(
                _PIECE_VALUES[visibility_board.piece_at(sq).piece_type]
                for sq in threatened
            )

            return base - threat_lambda * threat_value

        return evaluate

    return build


class _UCIEngine:
    """Minimal UCI client over a Stockfish subprocess.

    Sends `position` and `go`, parses `info`-line scores, ignores `bestmove`.
    Bypasses python-chess's `engine.analyse` because that path validates
    bestmove against the position and crashes on FOW positions where
    side-to-move is in check (Stockfish emits a move that doesn't reconcile
    with the position python-chess validates against).

    Reads stdout via `select` so the analyse loop has a real wall-clock
    timeout — Stockfish has been observed to deadlock on some FOW positions,
    and python-chess's `Limit(time=...)` only bounds Stockfish's compute, not
    its response time.
    """

    def __init__(self, path: str = "stockfish", threads: int = 1) -> None:
        self.path = path
        self.threads = threads
        self.proc: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._buffer: str = ""
        # Tracked so they get re-applied automatically when _ensure_alive
        # restarts the subprocess after a crash. Threads is special-cased
        # in _open; everything else (e.g. MultiPV) goes here.
        self._extra_options: list[tuple[str, str]] = []
        self._open()

    def _open(self) -> None:
        # Stockfish block-buffers stdout when its output is a pipe; give it a
        # pty so it line-buffers. Stdin stays a regular pipe (we control that
        # side and flush after every command).
        master_fd, slave_fd = pty.openpty()
        self._master_fd = master_fd
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=slave_fd,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        os.close(slave_fd)
        self._send("uci")
        self._wait_for_token("uciok", timeout=5.0)
        self._send(f"setoption name Threads value {self.threads}")
        # Forward-compat for Draft960 (engine-roadmap "Capability Tracks").
        # Stockfish in 960 mode handles canonical starts identically — the
        # option only changes castling-encoding semantics. Flipping it on
        # unconditionally now removes a future footgun where a Draft960
        # position would be evaluated with standard-chess castling rules.
        self._send("setoption name UCI_Chess960 value true")
        for name, value in self._extra_options:
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for_token("readyok", timeout=5.0)

    def setoption(self, name: str, value: str) -> None:
        """Set a UCI option and remember it so restart-on-crash reapplies it."""
        self._extra_options = [(n, v) for n, v in self._extra_options if n != name]
        self._extra_options.append((name, value))
        self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for_token("readyok", timeout=5.0)

    def _send(self, cmd: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise BrokenPipeError("UCI engine not running")
        self.proc.stdin.write((cmd + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _readline(self, timeout: float) -> str | None:
        if self._master_fd is None:
            return None
        deadline = time.monotonic() + timeout
        while True:
            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                return line + "\n"
            rem = deadline - time.monotonic()
            if rem <= 0:
                return None
            ready, _, _ = select.select([self._master_fd], [], [], rem)
            if not ready:
                return None
            try:
                chunk = os.read(self._master_fd, 4096)
            except OSError:
                return None
            if not chunk:
                return None
            self._buffer += chunk.decode("utf-8", errors="replace")

    def _wait_for_token(self, token: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"UCI: timed out waiting for {token}")
            line = self._readline(remaining)
            if line is None:
                raise TimeoutError(f"UCI: timed out waiting for {token}")
            stripped = line.strip()
            if stripped == token or stripped.startswith(token + " "):
                return

    def evaluate_fen(
        self, fen: str, depth: int, movetime_ms: int, slack_seconds: float = 0.5
    ) -> float | None:
        """Score a position from side-to-move's POV in centipawns, or None on timeout."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth} movetime {movetime_ms}")

        last_score: float | None = None
        deadline = time.monotonic() + (movetime_ms / 1000.0) + slack_seconds
        got_bestmove = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            line = self._readline(remaining)
            if line is None:
                break
            stripped = line.strip()
            if stripped.startswith("info "):
                parsed = _parse_info_score(stripped)
                if parsed is not None:
                    last_score = parsed
            elif stripped.startswith("bestmove"):
                got_bestmove = True
                break

        if not got_bestmove:
            # Bail Stockfish out so it's ready for the next position.
            self._send("stop")
            drain_deadline = time.monotonic() + 0.5
            while True:
                rem = drain_deadline - time.monotonic()
                if rem <= 0:
                    break
                line = self._readline(rem)
                if line is None:
                    break
                stripped = line.strip()
                if stripped.startswith("info "):
                    parsed = _parse_info_score(stripped)
                    if parsed is not None:
                        last_score = parsed
                elif stripped.startswith("bestmove"):
                    break
        return last_score

    def _ensure_alive(self) -> None:
        """Restart the Stockfish subprocess if it has died.

        Stockfish has been observed to crash on some FOW-edge positions
        (the same UB surface that broke python-chess in P2.0). When the
        subprocess exits, stdin writes silently succeed into a closed pipe
        and stdout reads return EOF, so every subsequent call returns
        (None, None) instantly. Self-heal so one bad position doesn't
        zero out analyzer success across the rest of the run.
        """
        if self.proc is None or self.proc.poll() is not None:
            try:
                self.close()
            except Exception:
                pass
            self._buffer = ""
            self._open()

    def analyze_fen_for_move(
        self, fen: str, depth: int, movetime_ms: int, slack_seconds: float = 1.5
    ) -> tuple[str | None, float | None]:
        """Best move (UCI string) + score from side-to-move POV. (None, None) on timeout/no-bestmove.

        Drains any leftover output from a previous (possibly interrupted)
        `go` with isready/readyok before issuing the next position. Without
        this, a single timeout cascades — leftover info/bestmove lines from
        the prior call get parsed as the current call's response and
        Stockfish ends up out of sync with our buffer.
        """
        self._ensure_alive()
        # Drain pending output from any previous go that may not have
        # cleanly ended. `stop` is a no-op if no search is running; the
        # subsequent isready/readyok handshake guarantees Stockfish is idle
        # and our buffer is drained of stale info/bestmove lines.
        try:
            self._send("stop")
            self._send("isready")
            self._wait_for_token("readyok", timeout=2.0)
        except TimeoutError:
            self._ensure_alive()
            return None, None
        except (BrokenPipeError, OSError):
            self._ensure_alive()
            return None, None

        self._send(f"position fen {fen}")
        self._send(f"go depth {depth} movetime {movetime_ms}")

        last_score: float | None = None
        bestmove: str | None = None
        deadline = time.monotonic() + (movetime_ms / 1000.0) + slack_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            line = self._readline(remaining)
            if line is None:
                break
            stripped = line.strip()
            if stripped.startswith("info "):
                parsed = _parse_info_score(stripped)
                if parsed is not None:
                    last_score = parsed
            elif stripped.startswith("bestmove"):
                tokens = stripped.split()
                if len(tokens) >= 2 and tokens[1] not in ("(none)", "0000"):
                    bestmove = tokens[1]
                break

        if bestmove is None:
            self._send("stop")
            drain_deadline = time.monotonic() + 0.5
            while True:
                rem = drain_deadline - time.monotonic()
                if rem <= 0:
                    break
                line = self._readline(rem)
                if line is None:
                    break
                stripped = line.strip()
                if stripped.startswith("info "):
                    parsed = _parse_info_score(stripped)
                    if parsed is not None:
                        last_score = parsed
                elif stripped.startswith("bestmove"):
                    tokens = stripped.split()
                    if len(tokens) >= 2 and tokens[1] not in ("(none)", "0000"):
                        bestmove = tokens[1]
                    break
        return bestmove, last_score

    def analyze_fen_multipv(
        self, fen: str, depth: int, movetime_ms: int, slack_seconds: float = 1.5
    ) -> dict[str, float] | None:
        """Top-K candidate moves with cp scores from side-to-move POV.

        Returns dict mapping UCI move string to cp score, with at most K
        entries (K = the current MultiPV setoption value). Caller should
        configure MultiPV via setoption before calling. Returns None on
        timeout / no bestmove / engine crash; caller should fall back to
        uniform when None.
        """
        self._ensure_alive()
        try:
            self._send("stop")
            self._send("isready")
            # Tight handshake: under 3-Stockfish load, healthy SF responds
            # in <50ms. If we don't see readyok in 300ms, the process is
            # thrashing — kill+restart and fall back, don't sit on it.
            self._wait_for_token("readyok", timeout=0.3)
        except (TimeoutError, BrokenPipeError, OSError):
            self._ensure_alive()
            return None

        try:
            self._send(f"position fen {fen}")
            self._send(f"go depth {depth} movetime {movetime_ms}")
        except (BrokenPipeError, OSError):
            self._ensure_alive()
            return None

        candidates: dict[int, tuple[str, float]] = {}
        deadline = time.monotonic() + (movetime_ms / 1000.0) + slack_seconds
        got_bestmove = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            line = self._readline(remaining)
            if line is None:
                break
            stripped = line.strip()
            if stripped.startswith("info "):
                parsed = _parse_info_multipv(stripped)
                if parsed is not None:
                    idx, move_uci, cp = parsed
                    candidates[idx] = (move_uci, cp)
            elif stripped.startswith("bestmove"):
                got_bestmove = True
                break

        if not got_bestmove:
            try:
                self._send("stop")
            except (BrokenPipeError, OSError):
                self._ensure_alive()
                return None
            drain_deadline = time.monotonic() + 0.5
            while True:
                rem = drain_deadline - time.monotonic()
                if rem <= 0:
                    break
                line = self._readline(rem)
                if line is None:
                    break
                stripped = line.strip()
                if stripped.startswith("info "):
                    parsed = _parse_info_multipv(stripped)
                    if parsed is not None:
                        idx, move_uci, cp = parsed
                        candidates[idx] = (move_uci, cp)
                elif stripped.startswith("bestmove"):
                    break

        if not candidates:
            return None
        return {move: cp for (move, cp) in candidates.values()}

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self._send("quit")
        except Exception:
            pass
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=2.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None


def _parse_info_multipv(line: str) -> tuple[int, str, float] | None:
    """Parse (multipv_index, first_pv_move_uci, cp_score) from an info line.

    Returns None when the info line lacks any of the three fields (e.g. the
    initial low-depth chatter Stockfish emits before scoring is stable).
    """
    tokens = line.split()
    multipv: int | None = None
    cp: float | None = None
    move_uci: str | None = None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "multipv" and i + 1 < len(tokens):
            try:
                multipv = int(tokens[i + 1])
            except ValueError:
                return None
            i += 2
        elif t == "score" and i + 2 < len(tokens):
            kind = tokens[i + 1]
            try:
                value = int(tokens[i + 2])
            except ValueError:
                return None
            if kind == "cp":
                cp = float(value)
            elif kind == "mate":
                cp = float(_KING_CAPTURE_SCORE if value > 0 else -_KING_CAPTURE_SCORE)
            i += 3
        elif t == "pv" and i + 1 < len(tokens):
            move_uci = tokens[i + 1]
            break
        else:
            i += 1
    if multipv is None or cp is None or move_uci is None:
        return None
    return multipv, move_uci, cp


def _parse_info_score(line: str) -> float | None:
    tokens = line.split()
    for i, t in enumerate(tokens):
        if t == "score" and i + 2 < len(tokens):
            kind = tokens[i + 1]
            try:
                value = int(tokens[i + 2])
            except ValueError:
                return None
            if kind == "cp":
                return float(value)
            if kind == "mate":
                return float(
                    _KING_CAPTURE_SCORE if value > 0 else -_KING_CAPTURE_SCORE
                )
    return None


@contextmanager
def stockfish_evaluator(
    *,
    path: str = "stockfish",
    depth: int = 4,
    time_cap_seconds: float = 0.5,
    slack_seconds: float = 0.5,
    threads: int = 1,
) -> Iterator[Evaluator]:
    """Yield a Stockfish-backed Evaluator using raw UCI.

    Bypasses python-chess's `engine.analyse` so FOW positions that produce
    illegal bestmoves don't crash the evaluator pipeline. We never parse
    `bestmove` — only `info`-line scores. Mate scores are clamped to
    ±_KING_CAPTURE_SCORE.

    Falls back to material on timeout, subprocess error, or no parsable score.
    The engine subprocess is restarted on hard errors but kept alive across
    "no score parsed" misses to avoid restart thrashing.
    """
    engine_holder: dict[str, _UCIEngine | None] = {
        "engine": _UCIEngine(path=path, threads=threads)
    }
    movetime_ms = int(time_cap_seconds * 1000)

    def _restart() -> None:
        if engine_holder["engine"] is not None:
            try:
                engine_holder["engine"].close()
            except Exception:
                pass
        engine_holder["engine"] = None

    def _ensure() -> _UCIEngine:
        if engine_holder["engine"] is None:
            engine_holder["engine"] = _UCIEngine(path=path, threads=threads)
        return engine_holder["engine"]

    try:

        def evaluate(
            board: chess.Board, move: chess.Move, perspective: chess.Color
        ) -> float:
            target = board.piece_at(move.to_square)
            if target is not None and target.piece_type == chess.KING:
                return (
                    _KING_CAPTURE_SCORE
                    if target.color != perspective
                    else -_KING_CAPTURE_SCORE
                )

            advanced = board.copy()
            advanced.push(move)

            if advanced.king(chess.WHITE) is None or advanced.king(chess.BLACK) is None:
                return 0.0

            try:
                eng = _ensure()
                score_from_stm = eng.evaluate_fen(
                    advanced.fen(),
                    depth,
                    movetime_ms,
                    slack_seconds=slack_seconds,
                )
            except (TimeoutError, OSError, BrokenPipeError):
                _restart()
                return material_score(advanced, perspective)

            if score_from_stm is None:
                return material_score(advanced, perspective)

            # Stockfish reports from advanced.turn (= side to move on `advanced`).
            return (
                score_from_stm
                if advanced.turn == perspective
                else -score_from_stm
            )

        yield evaluate
    finally:
        _restart()


def _piece_protection_penalty(board: chess.Board, perspective: chess.Color) -> float:
    """Per-piece penalty for own non-pawn non-king pieces with NO own defender.

    Encourages "stacked defense" formations. Per the user's annotation on
    vs-brian-game-2: "the engine doesn't have a great sense of safety and
    keeping its formation well positioned... things protected." Standalone
    from the safety term — safety penalizes pieces under opp attack, this
    penalizes pieces without own defenders regardless of opp attack.

    Scaled by 5% of piece value: knight 16cp, bishop ~17cp, rook 25cp,
    queen 45cp per undefended instance. Modest scale so existing tactical
    terms still dominate; this just biases choice when material is otherwise
    equal.
    """
    penalty = 0.0
    for sq, piece in board.piece_map().items():
        if piece.color != perspective or piece.piece_type in (chess.PAWN, chess.KING):
            continue
        if not board.attackers(perspective, sq):
            penalty += _PIECE_VALUES[piece.piece_type] * 0.05
    return penalty


def _rook_file_counts(board: chess.Board, perspective: chess.Color) -> tuple[int, int]:
    """Returns (open_file_rook_count, semi_open_file_rook_count) for own rooks.

    Open file: no own AND no opp pawns on the file.
    Semi-open: no own pawns on the file (but opp pawns may exist).
    """
    open_count = 0
    semi_count = 0
    for sq, piece in board.piece_map().items():
        if piece.color != perspective or piece.piece_type != chess.ROOK:
            continue
        f = chess.square_file(sq)
        own_pawn = False
        opp_pawn = False
        for r in range(8):
            other = board.piece_at(chess.square(f, r))
            if other is None or other.piece_type != chess.PAWN:
                continue
            if other.color == perspective:
                own_pawn = True
            else:
                opp_pawn = True
        if not own_pawn and not opp_pawn:
            open_count += 1
        elif not own_pawn:
            semi_count += 1
    return open_count, semi_count


def _rook_doubled_or_connected(board: chess.Board, perspective: chess.Color) -> int:
    """Number of additional own rooks beyond the first that share a file or rank
    with another own rook. 0 for one-or-fewer rooks; 1 if two rooks aligned;
    higher possible with three rooks (rare). Bonus signal for rook coordination.
    """
    rook_squares = [
        sq for sq, p in board.piece_map().items()
        if p.color == perspective and p.piece_type == chess.ROOK
    ]
    if len(rook_squares) < 2:
        return 0
    aligned = 0
    for i, sq_i in enumerate(rook_squares):
        for sq_j in rook_squares[i + 1:]:
            if (chess.square_file(sq_i) == chess.square_file(sq_j)
                    or chess.square_rank(sq_i) == chess.square_rank(sq_j)):
                aligned += 1
    return aligned


def _rook_on_7th_count(board: chess.Board, perspective: chess.Color) -> int:
    """Number of own rooks on opp's second rank (rank 7 for white, rank 2 for black)."""
    target_rank = 6 if perspective == chess.WHITE else 1
    count = 0
    for sq, piece in board.piece_map().items():
        if (piece.color == perspective
                and piece.piece_type == chess.ROOK
                and chess.square_rank(sq) == target_rank):
            count += 1
    return count


def fow_evaluator(
    *,
    material_weight: float = 1.0,
    safety_weight: float = 0.8,
    king_pressure_weight: float = 25.0,
    king_threat_weight: float = 200.0,
    king_proximity_weight: float = 40.0,
    visibility_weight: float = 6.0,
    fog_risk_weight: float = 0.2,
    # v0.9.6: positional terms addressing "formation / safety" feedback from
    # vs-brian asymmetric games. Modest weights so they nudge without
    # dominating; together they max out around ~150cp in late middlegames
    # which is comparable to a centralized minor piece in the material scale.
    piece_protection_weight: float = 1.0,
    rook_open_file_weight: float = 30.0,
    rook_semi_open_file_weight: float = 15.0,
    rook_doubled_weight: float = 15.0,
    rook_on_7th_weight: float = 25.0,
) -> Evaluator:
    """FoW-native evaluator combining six position signals on the particle board.

    Designed as a fast (no subprocess), drop-in replacement for Stockfish in
    both the current 1-ply architecture and future MCTS rollouts. All terms
    operate on the complete particle board — the particle provides the hidden
    piece positions that make each term particle-sensitive.

    Terms and centipawn scale:
      material         — centipawn balance of all pieces on the particle board.

      safety           — discount for own non-king pieces under opp pseudo-legal
                         attack. Penalised at piece_value × safety_weight.

      king_pressure    — bonus for own attacks targeting opp king zone. Offensive
                         signal; intentionally small (25cp max) so it doesn't
                         override the defensive terms.

      king_threat      — penalty per opponent piece directly attacking own king
                         square via pseudo-legal move. The defensive symmetric of
                         king_pressure. Set large (200cp) so the evaluator
                         strongly prefers moves that resolve direct king attacks.

      king_proximity   — penalty for each opponent slider/piece within 3 squares
                         of own king, scaled by proximity (closer = worse). Catches
                         the "bishop converging on the king" pattern before it
                         becomes a direct attack; scaled by piece value weight.

      visibility       — bonus for net board squares visible vs. opponent.

      fog_risk         — penalty for own pieces deep in enemy territory without
                         defensive support.
    """
    from .visibility import visible_squares as _vis

    def evaluate(
        board: chess.Board, move: chess.Move, perspective: chess.Color
    ) -> float:
        target = board.piece_at(move.to_square)
        if target is not None and target.piece_type == chess.KING:
            return (
                _KING_CAPTURE_SCORE
                if target.color != perspective
                else -_KING_CAPTURE_SCORE
            )

        advanced = board.copy()
        advanced.push(move)
        if advanced.king(chess.WHITE) is None or advanced.king(chess.BLACK) is None:
            return 0.0

        opp = not perspective

        # --- Term 1: material ---
        mat = material_weight * material_score(advanced, perspective)

        # --- Terms 2 + 3 + 4 share one opp pseudo-legal pass ---
        advanced.turn = opp
        opp_moves = list(advanced.pseudo_legal_moves)
        opp_targets: set[chess.Square] = {m.to_square for m in opp_moves}

        # Term 2: piece safety
        safety_penalty = safety_weight * sum(
            _PIECE_VALUES[p.piece_type]
            for sq, p in advanced.piece_map().items()
            if p.color == perspective
            and p.piece_type != chess.KING
            and sq in opp_targets
        )

        # Term 3 (defensive): king threat — direct attacks on own king
        own_king_sq = advanced.king(perspective)
        king_threat_penalty = 0.0
        if own_king_sq is not None:
            direct_attackers = sum(
                1 for m in opp_moves if m.to_square == own_king_sq
            )
            king_threat_penalty = king_threat_weight * direct_attackers

        # Term 4 (defensive): king proximity — opponent pieces converging on king
        king_proximity_penalty = 0.0
        if own_king_sq is not None:
            for sq, piece in advanced.piece_map().items():
                if piece.color != opp:
                    continue
                if piece.piece_type == chess.KING:
                    continue
                dist = chess.square_distance(sq, own_king_sq)
                if dist <= 3:
                    # Closer = worse; weight by piece mobility (sliders more threatening)
                    slider = piece.piece_type in (chess.QUEEN, chess.BISHOP, chess.ROOK)
                    piece_weight = 1.5 if slider else 1.0
                    king_proximity_penalty += king_proximity_weight * piece_weight * (4 - dist) / 3.0

        # --- Term 5 (offensive): king pressure ---
        opp_king_sq = advanced.king(opp)
        king_pressure_bonus = 0.0
        if opp_king_sq is not None:
            king_zone = chess.BB_KING_ATTACKS[opp_king_sq] | chess.BB_SQUARES[opp_king_sq]
            advanced.turn = perspective
            hits = sum(
                1 for m in advanced.pseudo_legal_moves
                if chess.BB_SQUARES[m.to_square] & king_zone
            )
            king_pressure_bonus = king_pressure_weight * min(hits, 8) / 8.0

        # --- Term 6: visibility advantage ---
        my_vis = _vis(advanced, perspective)
        opp_vis = _vis(advanced, opp)
        vis_bonus = visibility_weight * (len(my_vis) - len(opp_vis))

        # --- Term 7: fog risk ---
        fog_penalty = fog_risk_weight * fog_discount_term(advanced, perspective)

        # --- Term 8: piece protection (undefended own piece penalty) ---
        prot_penalty = piece_protection_weight * _piece_protection_penalty(advanced, perspective)

        # --- Terms 9 + 10: rook on open / semi-open file ---
        open_count, semi_count = _rook_file_counts(advanced, perspective)
        rook_file_bonus = (
            rook_open_file_weight * open_count
            + rook_semi_open_file_weight * semi_count
        )

        # --- Term 11: rook coordination (doubled or connected) ---
        rook_aligned = _rook_doubled_or_connected(advanced, perspective)
        rook_coord_bonus = rook_doubled_weight * rook_aligned

        # --- Term 12: rook on opp's 2nd rank ---
        rook_7_count = _rook_on_7th_count(advanced, perspective)
        rook_7_bonus = rook_on_7th_weight * rook_7_count

        return (
            mat
            - safety_penalty
            - king_threat_penalty
            - king_proximity_penalty
            + king_pressure_bonus
            + vis_bonus
            - fog_penalty
            - prot_penalty
            + rook_file_bonus
            + rook_coord_bonus
            + rook_7_bonus
        )

    return evaluate


_PSQT_PIECE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def psqt_evaluator(weights_path: str) -> Evaluator:
    """Linear piece-square-table evaluator loaded from an .npz checkpoint.

    The .npz must contain:
      w_own: float array of shape (6, 64) — own-piece coefficients (cp)
      w_opp: float array of shape (6, 64) — opp-piece coefficients (cp)
      bias:  float scalar offset

    Trained by scripts/train_psqt.py on a distillation corpus. Output is in
    centipawn-like units (corpus labels are rescaled by 1000 during training).
    """
    import numpy as np

    data = np.load(weights_path)
    w_own = data["w_own"].astype(np.float32)  # (6, 64)
    w_opp = data["w_opp"].astype(np.float32)
    bias = float(data["bias"])

    def evaluate(
        board: chess.Board, move: chess.Move, perspective: chess.Color
    ) -> float:
        target = board.piece_at(move.to_square)
        if target is not None and target.piece_type == chess.KING:
            return (
                _KING_CAPTURE_SCORE
                if target.color != perspective
                else -_KING_CAPTURE_SCORE
            )

        advanced = board.copy()
        advanced.push(move)
        if (
            advanced.king(chess.WHITE) is None
            or advanced.king(chess.BLACK) is None
        ):
            return 0.0

        score = bias
        for sq, piece in advanced.piece_map().items():
            pi = _PSQT_PIECE_INDEX[piece.piece_type]
            if piece.color == perspective:
                score += float(w_own[pi, sq])
            else:
                score += float(w_opp[pi, sq])
        return score

    return evaluate


def value_net_evaluator(
    weights_path: str,
    *,
    score_scale: float = 1000.0,
) -> Evaluator:
    """Pure-numpy value-net evaluator.

    Loads a small MLP V(state, mover) from .npz weights and scores moves by
    pushing the move on a copy of the particle board, then computing
    -V(next_board, opp_mover) — the negamax-shaped expected outcome for
    `perspective` after the move and an immediate opp response.

    The net outputs in [-1, 1] (tanh head). `score_scale` maps that into
    centipawn-ish units so the value net's scores mix sensibly with
    fow_evaluator's centipawn output and with the v0.9.4 capture-risk
    soft penalties. Default 1000.0 → a "winning position" reads as +1000cp.

    Trained by scripts/train_value_net.py on a self-play corpus with
    outcome labels in {-1, 0, +1} from each ply's mover's POV.
    """
    import numpy as np

    weights = np.load(weights_path)
    W1, b1 = weights["fc1.weight"], weights["fc1.bias"]
    W2, b2 = weights["fc2.weight"], weights["fc2.bias"]
    W3, b3 = weights["fc3.weight"], weights["fc3.bias"]
    in_dim = W1.shape[1]
    buf = np.zeros(in_dim, dtype=np.float32)

    def _encode(board: chess.Board, perspective: chess.Color):
        buf[:] = 0.0
        for sq, piece in board.piece_map().items():
            pi = _PSQT_PIECE_INDEX[piece.piece_type]
            if piece.color == perspective:
                buf[pi * 64 + sq] = 1.0
            else:
                buf[384 + pi * 64 + sq] = 1.0
        return buf

    def _forward(x):
        h1 = np.maximum(0.0, W1 @ x + b1)
        h2 = np.maximum(0.0, W2 @ h1 + b2)
        out = W3 @ h2 + b3
        return float(np.tanh(out[0]))

    def evaluate(
        board: chess.Board, move: chess.Move, perspective: chess.Color
    ) -> float:
        target = board.piece_at(move.to_square)
        if target is not None and target.piece_type == chess.KING:
            return (
                _KING_CAPTURE_SCORE
                if target.color != perspective
                else -_KING_CAPTURE_SCORE
            )

        advanced = board.copy()
        advanced.push(move)
        if (
            advanced.king(chess.WHITE) is None
            or advanced.king(chess.BLACK) is None
        ):
            return 0.0

        # Negamax: after my move, opp is to move. V(next, opp) is opp's
        # expected outcome; my expected outcome = -V(next, opp).
        x = _encode(advanced, not perspective)
        v_opp = _forward(x)
        return -v_opp * score_scale

    return evaluate


def mlp_evaluator(weights_path: str) -> Evaluator:
    """Small MLP value evaluator loaded from a torch state_dict checkpoint.

    Input is the same 768-d piece-square indicator vector PSQT uses; capacity
    comes from the hidden layers. The checkpoint must include both the
    state_dict and an 'arch' dict with in_dim/h1/h2 so the model can be
    reconstructed at load time.
    """
    import torch
    import torch.nn as nn

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    arch = ckpt["arch"]

    class _MLP(nn.Module):
        def __init__(self, in_dim, h1, h2):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x).squeeze(-1)

    model = _MLP(arch["in_dim"], arch["h1"], arch["h2"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # Persistent input buffer — single position at a time, no batch.
    buf = torch.zeros(1, arch["in_dim"], dtype=torch.float32)

    def evaluate(
        board: chess.Board, move: chess.Move, perspective: chess.Color
    ) -> float:
        target = board.piece_at(move.to_square)
        if target is not None and target.piece_type == chess.KING:
            return (
                _KING_CAPTURE_SCORE
                if target.color != perspective
                else -_KING_CAPTURE_SCORE
            )

        advanced = board.copy()
        advanced.push(move)
        if (
            advanced.king(chess.WHITE) is None
            or advanced.king(chess.BLACK) is None
        ):
            return 0.0

        buf.zero_()
        for sq, piece in advanced.piece_map().items():
            pi = _PSQT_PIECE_INDEX[piece.piece_type]
            if piece.color == perspective:
                buf[0, pi * 64 + sq] = 1.0
            else:
                buf[0, 384 + pi * 64 + sq] = 1.0
        with torch.no_grad():
            v = model(buf).item()
        return float(v)

    return evaluate
