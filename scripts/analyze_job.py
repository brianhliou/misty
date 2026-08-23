"""One analysis job: publication JSON in, per-ply analysis JSON out.

The server-side integration seam for fog-chess game analysis. The
platform's analysis job queue spawns this per finished game with the
frozen publication JSON (game-export shape: plies of {ply, mover, uci});
it emits one JSON document on stdout:

  - ``evals``: the standard white-POV eval track per ply cursor
    (Stockfish-on-truth at fixed depth) — feeds the advantage chart and
    move judgments like every other variant's analysis.
  - ``seats.white/.black``: the fog layer — per-ply belief context,
    engine-solve context, verdicts, and the seat's error budget
    (belief / sample / decision).

Runs ONLY on finished games from their published move list; it never
touches live state, so full information here is not a redaction concern
(the same boundary statement as fow_chess.analysis).

Usage:
    python scripts/analyze_job.py --pub game.json [--seat both]
        [--sf-depth 18] [--iterations 200] [--i-sample 8]
        [--time-budget SECONDS] [--no-search]
    cat game.json | python scripts/analyze_job.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess

from fow_chess.analysis import (
    DEFAULT_MISTAKE_CP,
    TruthGrader,
    aggregate_rows,
    analyze_game,
    analyze_game_deep,
    row_to_json,
)

SCHEMA_VERSION = "misty-analysis/1"

_PROMO = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}


def moves_from_publication(pub: dict) -> list[chess.Move]:
    moves: list[chess.Move] = []
    for p in pub.get("plies", []):
        uci = p["uci"]
        # game-export emits plain UCI already; tolerate long promotion names.
        # (Plain UCI is at most 5 chars — and "queen" itself ends in a valid
        # promotion letter, so length is the only safe discriminator.)
        if len(uci) > 5:
            for name, letter in _PROMO.items():
                if uci.endswith(name):
                    uci = uci[: -len(name)] + letter
                    break
        moves.append(chess.Move.from_uci(uci))
    return moves


def evals_from_rows(all_rows: list) -> list[dict]:
    """White-POV eval per ply cursor, from the graded rows of both seats.

    Cursor k is the position AFTER k plies: cursor k-1 uses ply k's
    before-eval; the final cursor uses the last graded ply's after-eval.
    Ungradeable positions (FoW-reachable, standard-illegal) emit cp=None.
    """
    graded = {r.ply: r for r in all_rows if r.grade is not None}
    n = max((r.ply for r in all_rows), default=0)
    out: list[dict] = []
    for cursor in range(n):
        row = graded.get(cursor + 1)
        if row is None:
            out.append({"ply": cursor, "cp": None, "mate": None, "best": None})
            continue
        cp = row.grade.sf_before_cp
        if row.color == "black":
            cp = -cp
        out.append(
            {"ply": cursor, "cp": cp, "mate": None, "best": row.grade.sf_best_uci}
        )
    last = graded.get(n)
    if last is not None:
        cp = last.grade.sf_after_played_cp
        if last.color == "black":
            cp = -cp
        out.append({"ply": n, "cp": cp, "mate": None, "best": None})
    else:
        out.append({"ply": n, "cp": None, "mate": None, "best": None})
    return out


def run_job(
    pub: dict,
    *,
    seat: str = "both",
    sf_depth: int = 18,
    iterations: int = 200,
    i_sample: int = 8,
    time_budget: float | None = None,
    mistake_cp: int = DEFAULT_MISTAKE_CP,
    search: bool = True,
) -> dict:
    moves = moves_from_publication(pub)
    seats = ["white", "black"] if seat == "both" else [seat]
    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "game_id": pub.get("game_id"),
        "variant": pub.get("variant"),
        "sf_depth": sf_depth,
        "mistake_cp": mistake_cp,
        "search": {
            "enabled": search,
            "iterations": iterations,
            "i_sample": i_sample,
            "time_budget_seconds": time_budget,
        },
        "seats": {},
    }
    all_rows = []
    with TruthGrader(depth=sf_depth) as grader:
        for s in seats:
            color = chess.WHITE if s == "white" else chess.BLACK
            if search:
                rows = analyze_game_deep(
                    moves,
                    color,
                    grader=grader,
                    mistake_cp=mistake_cp,
                    iterations=iterations,
                    i_sample_size=i_sample,
                    time_budget_seconds=time_budget,
                )
            else:
                rows = analyze_game(
                    moves, color, grader=grader, mistake_cp=mistake_cp
                )
            all_rows.extend(rows)
            result["seats"][s] = {
                "rows": [
                    row_to_json(r) for r in rows if r.belief_size is not None
                ],
                "budget": aggregate_rows(rows, mistake_cp=mistake_cp),
            }
    result["evals"] = evals_from_rows(all_rows)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pub", type=Path, default=None, help="publication JSON (default: stdin)")
    ap.add_argument("--seat", choices=("white", "black", "both"), default="both")
    ap.add_argument("--sf-depth", type=int, default=18)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--i-sample", type=int, default=8)
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--mistake-cp", type=int, default=DEFAULT_MISTAKE_CP)
    ap.add_argument("--no-search", action="store_true", help="belief + grading only (no per-ply solve)")
    args = ap.parse_args()

    pub = json.loads(args.pub.read_text() if args.pub else sys.stdin.read())
    result = run_job(
        pub,
        seat=args.seat,
        sf_depth=args.sf_depth,
        iterations=args.iterations,
        i_sample=args.i_sample,
        time_budget=args.time_budget,
        mistake_cp=args.mistake_cp,
        search=not args.no_search,
    )
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
