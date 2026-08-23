"""Golden-trace regression test for PEnumerator.

Replays one representative game and asserts the EXACT P set at every
ply matches a committed golden trace (SHA256 of sorted FENs per
perspective). This is the strongest possible PEnumerator regression
guard: any semantic change to ``update_own_move``,
``update_opp_move``, ``consistent_with``, ``apply_move``,
``pseudo_legal_moves``, ``visible_squares``, or the FEN serializer —
in Python OR Rust — will fail this test.

What the existing ``test_p_enum_replay.py`` asserts:
  - Truth-in-P at every ply (catches drop-truth bugs)
  - |P| ≤ a soft ceiling (catches runaway growth)

What it does NOT catch:
  - P contains EXTRA junk (bloat from FEN normalization mismatches)
  - P contains FEWER valid alternatives
  - FEN string canonicalization changes

This test catches all of the above by comparing the exact P set
membership (via hash) at every ply against the golden.

Regen via ``PYTHONPATH=src python tests/regen_golden_p_trace.py`` —
ONLY do this when an intentional semantic change to PEnumerator
warrants a new baseline, NEVER as a way to silence a failing test.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fow_chess.observation import observation_from_transition
from fow_chess.p_enum import PEnumerator


_FIXTURE = ROOT / "tests/fixtures/golden_p_trace.json"
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
    joined = "\n".join(sorted(positions))
    return hashlib.sha256(joined.encode()).hexdigest()


def test_p_enum_matches_golden_trace():
    if not _FIXTURE.exists():
        pytest.fail(
            f"golden trace fixture missing: {_FIXTURE} "
            f"(regen via tests/regen_golden_p_trace.py)"
        )
    payload = json.loads(_FIXTURE.read_text())
    from game_corpus import resolve_source_game

    source = resolve_source_game(payload["source_game"])
    max_ply = payload["max_ply"]
    expected = payload["trace"]

    moves = _load_moves(source)[:max_ply]
    board = chess.Board()
    penum_w = PEnumerator(chess.WHITE)
    penum_b = PEnumerator(chess.BLACK)

    mismatches: list[str] = []

    def _check(ply_idx: int) -> None:
        row = expected[ply_idx]
        actual_w_size = penum_w.size
        actual_b_size = penum_b.size
        actual_w_hash = _hash_p(penum_w.iter_positions())
        actual_b_hash = _hash_p(penum_b.iter_positions())
        if actual_w_size != row["P_white_size"]:
            mismatches.append(
                f"ply {row['ply']} |P_white| size: expected {row['P_white_size']}, "
                f"got {actual_w_size}"
            )
        if actual_b_size != row["P_black_size"]:
            mismatches.append(
                f"ply {row['ply']} |P_black| size: expected {row['P_black_size']}, "
                f"got {actual_b_size}"
            )
        if actual_w_hash != row["P_white_hash"]:
            mismatches.append(
                f"ply {row['ply']} P_white hash mismatch "
                f"(sizes match: {actual_w_size == row['P_white_size']}) — "
                f"set contents differ"
            )
        if actual_b_hash != row["P_black_hash"]:
            mismatches.append(
                f"ply {row['ply']} P_black hash mismatch "
                f"(sizes match: {actual_b_size == row['P_black_size']}) — "
                f"set contents differ"
            )

    _check(0)

    for ply, mv in enumerate(moves, start=1):
        prev = board.copy()
        if prev.king(chess.WHITE) is None or prev.king(chess.BLACK) is None:
            break
        if mv not in prev.pseudo_legal_moves:
            break
        board.push(mv)
        mover = prev.turn
        if mover == chess.WHITE:
            penum_w.update_own_move(mv)
            obs = observation_from_transition(prev, board, chess.BLACK)
            penum_b.update_opp_move(obs)
        else:
            penum_b.update_own_move(mv)
            obs = observation_from_transition(prev, board, chess.WHITE)
            penum_w.update_opp_move(obs)
        _check(ply)

    if mismatches:
        raise AssertionError(
            f"PEnumerator state diverged from golden trace at "
            f"{len(mismatches)} check(s):\n  " + "\n  ".join(mismatches[:10])
        )
