"""Regenerate the golden pick_move trace for test_pick_move_golden_trace.

This is the **Phase 0.5 characterization baseline** for the engine-
verticalization track (the in-place ``Rules`` seam, see
``docs/engine/mini-xiangqi-verticalization-track.md``). It pins the
FULL pick_move output of the dark-chess engine on a fixed line so the Phase 1
no-op refactor can prove it is byte-identical (the headline of the six parity
gates). Where ``test_p_enum_golden_trace`` pins belief-state membership, this
pins the *decision*: top move + action-value hash + strategy hash + |P| + |I|
at every engine-to-move ply.

It drives ``EngineV2(WHITE)`` down the real game-0007 line, feeding the ACTUAL
game moves to keep belief on the rails, and at each of WHITE's turns records
what the engine WOULD choose (before pushing the game move). Determinism rests
on the same mechanism ``test_search_reproducibility`` guards (commit 62f94b9:
canonical-sorted P, seeded sampling, fixed iteration count, no time budget).

SAME-MACHINE CONTRACT. This is a refactor characterization gate, not a
cross-arch CI gate. Regenerate and assert on the SAME machine/build: the engine
has documented knife-edge float sensitivity (rust-vs-python tree near-ties;
arm64-vs-x86 divergence, cf. the Bg4 finding). Action-value hashes are over
6-decimal-rounded values to absorb last-bit noise, but a genuine semantic change
moves values far more than 1e-6 AND flips the top move (which is asserted
exactly).

When to regen:
  - INTENTIONAL change that the gate should now accept (e.g. an intended
    algorithm tweak), AFTER verifying the new decisions are correct.
  - NEVER to silence a failing test — that defeats the regression. A Phase 1
    no-op refactor must reproduce this trace WITHOUT regen.

Run::

    PYTHONPATH=src .venv/bin/python tests/regen_golden_pick_trace.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fow_chess.engine_v2 import EngineV2
from fow_chess.observation import observation_from_transition

# --- Fixed, deterministic harness parameters (the contract). Changing any of
#     these requires a regen + a conscious note about why. ---
_GAME = ROOT / "feedback/mirror-fow-eval-seed1/games/game-0007-L-tier1-black.jsonl"
_FIXTURE = ROOT / "tests/fixtures/golden_pick_trace.json"
_PERSPECTIVE = chess.WHITE
_SEED = 42
_MAX_PLY = 12          # WHITE moves at plies 1,3,5,7,9,11 -> ~6 decisions
_ITERS = 160           # fixed iteration count => fully deterministic (no clock)
_I_SAMPLE = 16
_USE_RUST_TREE = True  # the live default path (WS2); gate the authoritative core
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


def _hash_action_map(amap) -> str:
    """SHA256 of sorted ``uci:6dp`` pairs. Order- and last-bit-noise-stable."""
    items = sorted((m.uci(), f"{float(v):.6f}") for m, v in (amap or {}).items())
    joined = "\n".join(f"{u}:{v}" for u, v in items)
    return hashlib.sha256(joined.encode()).hexdigest()


def _topk(amap, k: int = 3):
    items = sorted(
        ((m.uci(), round(float(v), 6)) for m, v in (amap or {}).items()),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return items[:k]


def build_trace() -> list[dict]:
    import random

    moves = _load_moves(_GAME)[:_MAX_PLY]
    eng = EngineV2(_PERSPECTIVE, rng=random.Random(_SEED))
    board = chess.Board()
    trace: list[dict] = []

    for ply, mv in enumerate(moves, start=1):
        prev = board.copy()
        if prev.king(chess.WHITE) is None or prev.king(chess.BLACK) is None:
            break
        if mv not in prev.pseudo_legal_moves:
            break
        mover = prev.turn
        if mover == _PERSPECTIVE:
            # Capture the engine's decision at THIS position before advancing.
            decision = eng.choose_move(
                iterations=_ITERS,
                i_sample_size=_I_SAMPLE,
                time_budget_seconds=None,
                use_rust_tree=_USE_RUST_TREE,
            )
            sol = eng.last_solution
            trace.append({
                "ply": ply,
                "p_size": eng.enumerator.size,
                "n_roots": getattr(sol, "n_roots", None),
                "move": decision.uci(),
                "av_hash": _hash_action_map(sol.action_values_at_root),
                "strategy_hash": _hash_action_map(sol.strategy_at_root),
                "av_top3": _topk(sol.action_values_at_root),
            })
            print(
                f"ply {ply:2d} decision={decision.uci()} |P|={eng.enumerator.size:>6d} "
                f"top3={trace[-1]['av_top3']}",
                flush=True,
            )
            # Advance the real game line; record our own move into belief.
            board.push(mv)
            obs = observation_from_transition(prev, board, _PERSPECTIVE)
            eng.observe_own_move(mv, obs)
        else:
            board.push(mv)
            obs = observation_from_transition(prev, board, _PERSPECTIVE)
            eng.observe_opp_move(obs)

    eng.close()
    return trace


def main() -> int:
    if shutil.which("stockfish") is None:
        print("ERROR: stockfish not on PATH (leaf eval unavailable)", file=sys.stderr)
        return 2
    if not _GAME.exists():
        print(f"ERROR: source game not found: {_GAME}", file=sys.stderr)
        return 2

    trace = build_trace()
    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_game": str(_GAME.relative_to(ROOT)),
        "perspective": "white",
        "seed": _SEED,
        "max_ply": _MAX_PLY,
        "iters": _ITERS,
        "i_sample": _I_SAMPLE,
        "use_rust_tree": _USE_RUST_TREE,
        "schema_version": 1,
        "trace": trace,
    }
    _FIXTURE.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {_FIXTURE} ({len(trace)} decision rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
