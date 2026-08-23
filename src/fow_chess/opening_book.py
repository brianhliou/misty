"""Observation-keyed opening book — a surgical patch over known opening holes.

The v2 engine has a systematic opening weakness (the EV-dilution / catastrophe
class, post-launch-track S1/S2): it hangs material or the king to a threat that
exists in only a small fraction of belief worlds, and expected-value selection
averages the catastrophe away. The worst cases are deterministic opening traps
that a human can replay at will on the live site:

  * White pushes the d2-pawn while the king is on e1, opening the a5-e1 diagonal
    to a queen it cannot see -> ``...Qxe1`` captures the king (lost in 3 moves).
  * Black plays an early ``...Qh4`` into an ``Nf3`` it cannot see -> ``Nxh4``
    hangs the queen.

The fix is NOT a general rule (that would change how the engine plays in every
game, a real strength cost). It is a tiny lookup keyed on the engine's *own
observed view* — the only thing it can reliably recognize under fog. In Fog of
War the engine cannot see the enemy move that sets the trap (no white piece
reaches a5; black never sees the f3-knight), so we cannot key on "enemy queen on
a5". We key on what the side-to-move actually observes — its own pieces plus the
squares/pieces it can see — and the book fires *only* when that fingerprint
exactly matches a recorded one. Outside those few positions it is inert, so it
costs ~nothing in general play (validated by the neutrality bakeoff).

Two action kinds:

  * ``force``  — play this move instead of searching. Used for the White safe-
    development line (``e4`` -> ``Nf3`` -> ``Nc3``), which keeps the d2-pawn home
    and parks a knight on c3 (a second blocker of the king's diagonal), so the
    ``Qxe1`` trap is structurally impossible. Short-circuits the search (also
    faster + saves clock in the opening).
  * ``block`` — let the engine search, but if its top choice is one of the
    recorded bad moves, drop it and take the next-best legal move. Used for
    early queen-exposure lines like ``...Qh4``.

View fingerprints are *path-agnostic*: the selfplay harness and the live
protocol adapter both build the same ``PerspectiveView`` for a given true
position (redaction parity is enforced cross-language), so a fingerprint
captured offline from a canonical board matches the live worker's view
byte-for-byte. Entries may also carry an optional observation-history
fingerprint, computed only from legal observations the player received; those
entries fire only when both the current view and observation history match.

The book is opt-in (``FOW_OPENING_BOOK=1`` or the ``opening_book`` constructor
arg); the bare engine is unchanged when it is off — "default matches prior
behavior", so parity/reproducibility guards are unaffected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess

from .observation import Observation
from .move_policy import next_best_legal

# What we fingerprint on. Anything with these three attributes works — a real
# PerspectiveView (live + selfplay) or a lightweight stand-in (the capture
# script). Kept structural on purpose so capture and runtime share one function.


def view_fingerprint(view) -> str:
    """Canonical string for the side-to-move's observed view under fog.

    Built from exactly what the player can see: its perspective, the visible
    piece map (own pieces + any visible opponent pieces), and the visibility
    square set (which empty squares it can see — distinguishes e.g. a blocked vs
    open file even when no enemy piece is on it). Deterministic and order-stable
    (pieces sorted by square), so the same observed position always hashes the
    same, regardless of how it was reached.

    Two true positions that LOOK identical to the side-to-move collapse to one
    fingerprint — that is the point: every invisible opponent reply to ``1.e4``
    produces the same White view, so one recorded fingerprint covers them all.
    """
    persp = "w" if view.perspective == chess.WHITE else "b"
    pieces = ",".join(
        f"{sq}:{piece.symbol()}"
        for sq, piece in sorted(view.visible_piece_map.items())
    )
    vis = int(view.visible_squares)
    return f"{persp}|{pieces}|{vis}"


def observation_fingerprint(observation: Observation) -> str:
    """Canonical string for one legal FoW observation.

    Unlike a true move-history key, this only uses information the perspective
    player was allowed to receive: visible squares/pieces, whether one of its
    own pieces disappeared, any known opponent landing square, and game-over
    signal. It intentionally excludes hidden board state.
    """
    pieces = ",".join(
        f"{sq}:{piece.symbol()}"
        for sq, piece in sorted(observation.visible_pieces.items())
    )
    own_capture = "-" if observation.own_capture_square is None else str(observation.own_capture_square)
    opp_landing = (
        "-"
        if observation.opp_capture_landing_square is None
        else str(observation.opp_capture_landing_square)
    )
    if observation.game_over is None:
        game_over = "-"
    else:
        winner = (
            "-"
            if observation.game_over.winner is None
            else ("w" if observation.game_over.winner == chess.WHITE else "b")
        )
        game_over = f"{winner}:{observation.game_over.reason}"
    return (
        f"obs|vis={int(observation.visibility_mask)}|pieces={pieces}|"
        f"owncap={own_capture}|oppland={opp_landing}|game={game_over}"
    )


def observation_event_fingerprint(
    kind: str,
    observation: Observation,
    *,
    move: chess.Move | None = None,
) -> str:
    """Canonical string for one observation-history event.

    ``kind`` is ``own`` or ``opp``. Own moves include the move UCI because the
    player knows what it chose; opponent moves do not include hidden UCI because
    the player only receives an observation.
    """
    if kind not in ("own", "opp"):
        raise ValueError(f"unknown observation event kind: {kind!r}")
    if kind == "own":
        if move is None:
            raise ValueError("own observation events require move")
        prefix = f"own:{move.uci()}"
    else:
        if move is not None:
            raise ValueError("opp observation events cannot include hidden move")
        prefix = "opp"
    return f"{prefix}|{observation_fingerprint(observation)}"


def observation_history_fingerprint(events: Iterable[str]) -> str:
    """Stable digest of the observation event stream.

    The digest is path-sensitive, but only over observation-event strings built
    from legal player information. Length-prefix each event so concatenations
    cannot collide structurally (``ab`` + ``c`` vs ``a`` + ``bc``).
    """
    h = hashlib.sha256()
    h.update(b"fow-observation-history-v1\0")
    for event in events:
        encoded = event.encode("utf-8")
        h.update(str(len(encoded)).encode("ascii"))
        h.update(b":")
        h.update(encoded)
        h.update(b"\0")
    return f"obsh:v1:{h.hexdigest()}"


@dataclass(frozen=True)
class BookAction:
    kind: str  # "force" | "block"
    move: chess.Move
    note: str = ""
    history_fingerprint: str | None = None


class OpeningBook:
    """An observation-keyed lookup: fingerprint -> BookAction(s).

    Implements the ``MovePolicy`` protocol (move_policy.py): a FORCE entry is a
    ``pre_move`` (play it without searching), a BLOCK entry is a ``post_move``
    (drop the search's pick if it is the booked bad move, take the next-best).
    """

    name = "opening_book"

    def __init__(self, entries: dict[str, BookAction | list[BookAction]]) -> None:
        self._entries = entries

    def __len__(self) -> int:
        total = 0
        for actions in self._entries.values():
            total += len(actions) if isinstance(actions, list) else 1
        return total

    def _matching_actions(
        self,
        view,
        history_fingerprint: str | None = None,
    ) -> list[BookAction]:
        if history_fingerprint is None:
            history_fingerprint = getattr(view, "observation_history_fingerprint", None)
        actions = self._entries.get(view_fingerprint(view))
        if actions is None:
            return []
        if not isinstance(actions, list):
            actions = [actions]
        exact = []
        fallback = []
        for action in actions:
            if action.history_fingerprint is None:
                fallback.append(action)
            elif action.history_fingerprint == history_fingerprint:
                exact.append(action)
        return exact + fallback

    def lookup(self, view, history_fingerprint: str | None = None) -> BookAction | None:
        """Return the first action for this view, or None if the view isn't booked.

        Use the policy hooks for behavior; they consider every matching action.
        This method is kept for tests, diagnostics, and the existing scan script.
        """
        actions = self._matching_actions(view, history_fingerprint)
        return actions[0] if actions else None

    def pre_move(self, view) -> chess.Move | None:
        for action in self._matching_actions(view):
            if action.kind == "force" and action.move in view.own_legal_moves:
                return action.move
        return None

    def post_move(self, view, chosen: chess.Move, solution) -> chess.Move | None:
        blocked = {
            action.move
            for action in self._matching_actions(view)
            if action.kind == "block"
        }
        if chosen in blocked:
            return next_best_legal(solution, view, exclude=blocked)
        return None


def default_path() -> Path:
    """Bundled book shipped with the engine (cloned verbatim by the worker)."""
    return Path(__file__).resolve().parent / "data" / "opening_book.json"


def load(path: str | Path | None = None) -> OpeningBook | None:
    """Load the book from JSON. Returns None when the file is absent or empty,
    so a missing book degrades to "no book" rather than erroring."""
    p = Path(path) if path is not None else default_path()
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    entries: dict[str, list[BookAction]] = {}
    for e in data.get("entries", []):
        action = BookAction(
            kind=e["action"],
            move=chess.Move.from_uci(e["move"]),
            note=e.get("note", ""),
            history_fingerprint=e.get("history_fingerprint"),
        )
        entries.setdefault(e["fingerprint"], []).append(action)
    return OpeningBook(entries) if entries else None
