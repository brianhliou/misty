"""Self-play harness tests."""

from __future__ import annotations

from fow_chess.selfplay import play_game
from fow_chess.strategies import RandomStrategy


def test_random_vs_random_finishes_in_finite_plies() -> None:
    result = play_game(
        RandomStrategy(seed=1),
        RandomStrategy(seed=2),
        max_plies=300,
        room_id="test-rvr",
    )

    assert result.plies > 0
    assert result.plies <= 300
    # First event is room-created; the rest are move-played.
    assert result.events[0]["type"] == "room-created"
    assert all(e["type"] == "move-played" for e in result.events[1:])
    # Random play should normally end in king capture, but truncation is allowed.
    assert result.end_reason in {"king-captured", "truncated", "no-legal-moves"}
