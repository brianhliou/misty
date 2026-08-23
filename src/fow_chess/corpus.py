"""Load mistboard corpus directories produced by `scripts/generate-fow-corpus.mjs`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class GameEntry:
    """One game's manifest entry plus path resolution."""

    seed: int
    plies: int
    winner: str | None
    end_reason: str
    ended_by_king_capture: bool
    truncated: bool
    has_capture: bool
    has_promotion: bool
    has_en_passant: bool
    has_castling: bool
    path: Path  # absolute path to the events JSONL file
    views_path: Path | None  # absolute path to the per-ply TS view JSONL, if produced


@dataclass(frozen=True)
class Corpus:
    """A loaded corpus directory."""

    root: Path
    generator: str
    bias: str | None
    variant: str
    base_seed: int
    game_count: int
    max_plies: int
    totals: dict[str, int]
    games: list[GameEntry]


def load_corpus(corpus_dir: Path | str) -> Corpus:
    """Read a corpus directory's manifest.json and resolve game file paths."""
    root = Path(corpus_dir).resolve()
    manifest_path = root / "manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)

    games = [
        GameEntry(
            seed=entry["seed"],
            plies=entry["plies"],
            winner=entry["winner"],
            end_reason=entry["end_reason"],
            ended_by_king_capture=entry["ended_by_king_capture"],
            truncated=entry["truncated"],
            has_capture=entry["has_capture"],
            has_promotion=entry["has_promotion"],
            has_en_passant=entry["has_en_passant"],
            has_castling=entry["has_castling"],
            path=root / entry["path"],
            views_path=(root / entry["views_path"]) if entry.get("views_path") else None,
        )
        for entry in manifest["games"]
    ]

    return Corpus(
        root=root,
        generator=manifest["generator"],
        bias=manifest.get("bias"),
        variant=manifest["variant"],
        base_seed=manifest["base_seed"],
        game_count=manifest["game_count"],
        max_plies=manifest["max_plies"],
        totals=manifest["totals"],
        games=games,
    )


def read_events(game: GameEntry) -> list[dict[str, Any]]:
    """Read all GameEvent records from a game's JSONL file."""
    with game.path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_views(game: GameEntry) -> list[dict[str, Any]]:
    """Read per-ply TS PlayerView snapshots. Each record: {ply, white: [...], black: [...]}."""
    if game.views_path is None:
        raise FileNotFoundError(
            f"game seed={game.seed} has no views_path; corpus may predate view dumping"
        )
    with game.views_path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_games(corpus: Corpus) -> Iterator[tuple[GameEntry, list[dict[str, Any]]]]:
    """Yield (GameEntry, events) for every game in the corpus, in manifest order."""
    for game in corpus.games:
        yield game, read_events(game)
