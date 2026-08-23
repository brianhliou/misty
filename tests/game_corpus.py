"""The real-game corpus the Rust↔Python parity suites replay.

Two layers:

- ``tests/fixtures/games/<family>/game-*.jsonl`` — a small curated subset
  COMMITTED to the repo. This is the layer that guarantees the parity suites
  can never go silently vacuous: :func:`corpus_game_paths` RAISES at
  collection time if it's missing, instead of returning an empty parametrize
  list (the historical failure mode — an exported/partial tree ran the
  "parity gate" with zero assertions and looked green).
- ``feedback/mirror-*/games/*.jsonl`` — the full 84-game research corpus,
  present only in the private working repo. When it exists, its games are
  ADDED to the parametrization (deduped against the fixture copies), so the
  private repo keeps its broad coverage.

Fixture family names mirror the feedback dirs minus the ``mirror-`` prefix
(``fow-eval-seed1`` ↔ ``mirror-fow-eval-seed1``).
"""

from __future__ import annotations

from pathlib import Path

_TESTS = Path(__file__).parent
FIXTURE_GAMES = _TESTS / "fixtures" / "games"
FEEDBACK = _TESTS.parent / "feedback"


def _family_of(path: Path) -> str:
    # fixtures: fixtures/games/<family>/game-x.jsonl -> <family>
    # feedback: feedback/mirror-<family>/games/game-x.jsonl -> <family>
    if path.parent.name == "games":
        return path.parent.parent.name.removeprefix("mirror-")
    return path.parent.name


def corpus_game_paths(
    family: str | None = None, limit: int | None = None
) -> list[Path]:
    """Sorted game paths: committed fixtures first, then any additional
    private-corpus games (deduped by (family, filename)).

    ``family`` filters to one source family; ``limit`` caps the result.
    Raises if the committed fixture layer is missing — an empty corpus must
    fail the suite loudly, never shrink it to zero tests.
    """
    fixtures = sorted(FIXTURE_GAMES.glob("*/game-*.jsonl"))
    if not fixtures:
        raise RuntimeError(
            f"committed game corpus missing at {FIXTURE_GAMES} — the parity "
            "suites would collect zero tests and pass vacuously"
        )
    seen = {(_family_of(p), p.name) for p in fixtures}
    paths = list(fixtures)
    if FEEDBACK.exists():
        for p in sorted(FEEDBACK.glob("mirror-*/games/*.jsonl")):
            if (_family_of(p), p.name) not in seen:
                paths.append(p)
    if family is not None:
        paths = [p for p in paths if _family_of(p) == family]
        if not paths:
            raise RuntimeError(f"no corpus games for family {family!r}")
    return paths[:limit] if limit else paths


def corpus_id(path: Path) -> str:
    """Stable parametrize id: <family>/<stem>."""
    return f"{_family_of(path)}/{path.stem}"


def resolve_source_game(recorded: str) -> Path:
    """Resolve a golden trace's recorded ``source_game`` path.

    Traces regenerated in the private repo record ``feedback/...`` paths;
    in a tree without ``feedback/`` the same game exists as a committed
    fixture. Missing everywhere = hard error (the trace test must fail,
    not skip — that skip was part of the vacuous-parity hole).
    """
    p = _TESTS.parent / recorded
    if p.exists():
        return p
    name = Path(recorded).name
    hits = sorted(FIXTURE_GAMES.glob(f"*/{name}"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"golden-trace source game {recorded!r} not found in the repo or "
        f"under {FIXTURE_GAMES}"
    )
