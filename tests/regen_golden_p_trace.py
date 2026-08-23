"""Regenerate the golden P-set hash trace for test_p_enum_golden_trace.

Replays one representative game through PEnumerator and records, at each
ply, a SHA256 hash of the sorted-FEN concatenation of the entire P set.
The trace lives at tests/fixtures/golden_p_trace.json and is committed
to the repo.

When to regen:
  - INTENTIONAL semantic change to PEnumerator (e.g., a new constraint
    in consistent_with). Verify the new trace is correct first.
  - NEVER as a way to "fix" a failing test — that defeats the regression.

Run::

    PYTHONPATH=src .venv/bin/python tests/regen_golden_p_trace.py

Pick: the first 18 plies of game-0007 (the profile game). This range
keeps |P| under ~300K per perspective — fast enough for CI replay but
exercises the meaningful belief-update paths.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import PEnumerator


_GAME = ROOT / "feedback/mirror-fow-eval-seed1/games/game-0007-L-tier1-black.jsonl"
_FIXTURE = ROOT / "tests/fixtures/golden_p_trace.json"
_MAX_PLY = 14
_PROMO_LETTER = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}


def _load_moves(path: Path) -> list[chess.Move]:
    moves: list[chess.Move] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "move-played":
            continue
        mv = e["move"]
        uci = f"{mv['from']}{mv['to']}"
        if mv.get("promotion"):
            uci += _PROMO_LETTER[mv["promotion"].lower()]
        moves.append(chess.Move.from_uci(uci))
    return moves


def _hash_p(positions) -> str:
    """SHA256 of newline-joined sorted FENs. Order-independent."""
    joined = "\n".join(sorted(positions))
    return hashlib.sha256(joined.encode()).hexdigest()


def main() -> int:
    if not _GAME.exists():
        print(f"ERROR: source game not found: {_GAME}", file=sys.stderr)
        return 2

    moves = _load_moves(_GAME)[:_MAX_PLY]
    board = chess.Board()
    penum_w = PEnumerator(chess.WHITE)
    penum_b = PEnumerator(chess.BLACK)

    trace = []
    trace.append({
        "ply": 0,
        "P_white_size": penum_w.size,
        "P_black_size": penum_b.size,
        "P_white_hash": _hash_p(penum_w.iter_positions()),
        "P_black_hash": _hash_p(penum_b.iter_positions()),
    })

    for ply, mv in enumerate(moves, start=1):
        prev = board.copy()
        if prev.king(chess.WHITE) is None or prev.king(chess.BLACK) is None:
            break
        if mv not in prev.pseudo_legal_moves:
            break
        board.push(mv)
        mover = prev.turn  # color that just moved
        if mover == chess.WHITE:
            penum_w.update_own_move(mv)
            obs = observation_from_transition(prev, board, chess.BLACK)
            penum_b.update_opp_move(obs)
        else:
            penum_b.update_own_move(mv)
            obs = observation_from_transition(prev, board, chess.WHITE)
            penum_w.update_opp_move(obs)
        trace.append({
            "ply": ply,
            "mover": "white" if mover == chess.WHITE else "black",
            "uci": mv.uci(),
            "P_white_size": penum_w.size,
            "P_black_size": penum_b.size,
            "P_white_hash": _hash_p(penum_w.iter_positions()),
            "P_black_hash": _hash_p(penum_b.iter_positions()),
        })
        print(
            f"ply {ply:2d} mover={mover} |P_w|={penum_w.size:>6d} "
            f"|P_b|={penum_b.size:>6d}",
            flush=True,
        )

    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_game": str(_GAME.relative_to(ROOT)),
        "max_ply": _MAX_PLY,
        "schema_version": 1,
        "trace": trace,
    }
    _FIXTURE.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {_FIXTURE} ({len(trace)} ply rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
