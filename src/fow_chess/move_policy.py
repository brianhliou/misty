"""Move-policy layer — deterministic overrides around the GT-CFR search.

This is the general mechanism the opening book is the first instance of. A
``MovePolicy`` gets two hooks per decision:

  * ``pre_move(view) -> Move | None`` — play this move WITHOUT searching (None =
    fall through to search). Used by the book's FORCE entries; also instant, so
    it saves clock.
  * ``post_move(view, chosen, solution) -> Move | None`` — given the search's
    pick, return a replacement (None = keep ``chosen``). Used by the book's BLOCK
    entries.

``EngineV2Strategy`` holds an ordered list of policies, each independently
flag-gated. Per move it runs every ``pre_move`` first (the first to return a move
wins; search is skipped); otherwise it searches, then threads the result through
each ``post_move`` (so policies chain — a later one sees the running choice).

Why this seam exists
--------------------
The opening book (``FOW_OPENING_BOOK``) is policy #1. The planned post-launch S1
strength fix — a catastrophe-averse filter that vetoes any move losing
material/king in more than a threshold fraction of belief worlds — is a natural
policy #2: a ``post_move`` that, when ``chosen`` is catastrophic, returns the
best surviving move. A blueprint-anchored override would be a #3. Building those
means writing a new ``MovePolicy`` + its own flag and registering it here — not
threading more special-cases through ``pick_move``.

Policies are deterministic overrides, NOT the search itself. Keep belief-aware
reasoning in the CFR/leaf-eval path; use a policy only for rules that are
correct to apply on the observed view (or the solved tree) regardless of search.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import chess


@runtime_checkable
class MovePolicy(Protocol):
    """A deterministic override around the search. See module docstring."""

    name: str

    def pre_move(self, view) -> chess.Move | None:
        """Return a move to play without searching, or None to fall through."""
        ...

    def post_move(self, view, chosen: chess.Move, solution) -> chess.Move | None:
        """Return a replacement for the search's ``chosen`` move, or None to keep
        it. ``solution`` is the engine's last MultiRoot solution (read
        ``action_values_at_root`` for the move ranking)."""
        ...


def next_best_legal(solution, view, exclude: chess.Move | Iterable[chess.Move]) -> chess.Move:
    """Highest-valued legal move other than ``exclude`` from the search ranking.

    Pure (no promotion tiebreak — the caller applies that uniformly). Falls back
    to any other legal move when the ranking is missing/exhausted, and to
    an excluded move only when there is no alternative to take. This is the
    shared "drop these moves, take the next-best" primitive every vetoing policy
    uses.
    """
    excluded = {exclude} if isinstance(exclude, chess.Move) else set(exclude)
    legal = set(view.own_legal_moves)
    av = getattr(solution, "action_values_at_root", None) if solution else None
    if av:
        for mv, _v in sorted(av.items(), key=lambda kv: -kv[1]):
            if mv not in excluded and mv in legal:
                return mv
    for mv in view.own_legal_moves:
        if mv not in excluded:
            return mv
    return view.own_legal_moves[0] if view.own_legal_moves else next(iter(excluded))
