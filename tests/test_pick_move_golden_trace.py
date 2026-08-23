"""Golden-trace regression test for the FULL pick_move decision path.

The **Phase 0.5 characterization gate** for engine verticalization. Replays the
real game-0007 line through ``EngineV2(WHITE)`` and asserts that, at every one of
WHITE's turns, the engine reproduces the committed decision: exact top move,
exact |P|, and a hash of both the action-value map and the purified strategy.

This is the headline gate for the Phase 1 no-op ``Rules`` refactor (see
``docs/engine/mini-xiangqi-verticalization-track.md``): the refactor
routes ~115 chess call-sites through a ``Rules`` seam without changing behavior,
and *passing this test without regen* is the proof that dark-chess strength is
unchanged. It complements ``test_p_enum_golden_trace`` (which pins belief
membership) by pinning the decision the solver builds on that belief.

Determinism is the same mechanism ``test_search_reproducibility`` guards
(commit 62f94b9): canonical-sorted P + seeded sampling + fixed iteration count
(no time budget) => same seed reproduces the same game.

SAME-MACHINE CONTRACT — regen and assert on the same machine/build. The engine
has documented knife-edge float sensitivity (rust-vs-python near-ties; the
arm64-vs-x86 Bg4 divergence), so this is a refactor characterization gate, not a
cross-arch CI oracle. The top *move* is asserted exactly (the decision must not
change); action values are hashed at 6dp to absorb last-bit noise while still
catching any real semantic change (which moves values far more than 1e-6).

Regen via ``tests/regen_golden_pick_trace.py`` — ONLY for an intentional,
verified change, NEVER to silence a failing test.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fow_chess.engine_v2 import EngineV2
from fow_chess.observation import observation_from_transition

_FIXTURE = ROOT / "tests/fixtures/golden_pick_trace.json"
_PROMO_LETTER = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}

pytestmark = pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not on PATH (leaf eval unavailable)",
)


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


def _hash_action_map(amap) -> str:
    items = sorted((m.uci(), f"{float(v):.6f}") for m, v in (amap or {}).items())
    joined = "\n".join(f"{u}:{v}" for u, v in items)
    return hashlib.sha256(joined.encode()).hexdigest()


def test_pick_move_matches_golden_trace():
    if not _FIXTURE.exists():
        pytest.fail(
            f"golden pick trace fixture missing: {_FIXTURE} "
            f"(regen via tests/regen_golden_pick_trace.py)"
        )
    payload = json.loads(_FIXTURE.read_text())
    from game_corpus import resolve_source_game

    source = resolve_source_game(payload["source_game"])

    persp = chess.WHITE if payload["perspective"] == "white" else chess.BLACK
    seed = payload["seed"]
    max_ply = payload["max_ply"]
    iters = payload["iters"]
    i_sample = payload["i_sample"]
    use_rust_tree = payload["use_rust_tree"]
    expected = {row["ply"]: row for row in payload["trace"]}

    moves = _load_moves(source)[:max_ply]
    eng = EngineV2(persp, rng=random.Random(seed))
    board = chess.Board()
    mismatches: list[str] = []

    try:
        for ply, mv in enumerate(moves, start=1):
            prev = board.copy()
            if prev.king(chess.WHITE) is None or prev.king(chess.BLACK) is None:
                break
            if mv not in prev.pseudo_legal_moves:
                break
            mover = prev.turn
            if mover == persp:
                decision = eng.choose_move(
                    iterations=iters,
                    i_sample_size=i_sample,
                    time_budget_seconds=None,
                    use_rust_tree=use_rust_tree,
                )
                sol = eng.last_solution
                row = expected.get(ply)
                if row is None:
                    mismatches.append(f"ply {ply}: no golden row (trace shape changed)")
                else:
                    if decision.uci() != row["move"]:
                        mismatches.append(
                            f"ply {ply} DECISION: expected {row['move']}, "
                            f"got {decision.uci()} (top-move regression)"
                        )
                    if eng.enumerator.size != row["p_size"]:
                        mismatches.append(
                            f"ply {ply} |P|: expected {row['p_size']}, "
                            f"got {eng.enumerator.size}"
                        )
                    if _hash_action_map(sol.action_values_at_root) != row["av_hash"]:
                        mismatches.append(
                            f"ply {ply} action-value hash differs "
                            f"(decision match: {decision.uci() == row['move']})"
                        )
                    if _hash_action_map(sol.strategy_at_root) != row["strategy_hash"]:
                        mismatches.append(f"ply {ply} strategy hash differs")
                board.push(mv)
                obs = observation_from_transition(prev, board, persp)
                eng.observe_own_move(mv, obs)
            else:
                board.push(mv)
                obs = observation_from_transition(prev, board, persp)
                eng.observe_opp_move(obs)
    finally:
        eng.close()

    if mismatches:
        raise AssertionError(
            f"pick_move decision path diverged from golden trace at "
            f"{len(mismatches)} check(s):\n  " + "\n  ".join(mismatches[:12])
        )
