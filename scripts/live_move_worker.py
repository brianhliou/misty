"""Long-lived Python worker for live PvE moves.

Sister script to live_move_runner.py — same logic, but holds the strategy
runtime open across many requests instead of spinning up a fresh interpreter
per move. Designed for the Node-side python-pool to keep N of these warm.

Protocol (one JSON object per line on stdin and stdout):

  Worker is launched with CLI args:
    --engine-id <id>           e.g. python-random-legal, python-tier1-v0.9.1
    --seed <int>               worker-lifetime seed used to construct the strategy
    [--stockfish <path>]       Stockfish binary (defaults to env / "stockfish")

  Once strategy init succeeds, the worker emits a single ready line:
    {"kind": "ready", "engineId": "...", "pid": <int>}

  Then it accepts request lines from stdin, one JSON object per line:
    {
      "requestId": "<opaque>",
      "engineTurnRequest": <EngineTurnRequest JSON, see
        packages/game/src/engine-protocol.ts>,
      "workerDeadlineMs": <int or null>,   # transport bound -> wall deadline
      "computeBudgetMs": <int or null>,    # per-move compute allowance
      "watchdogTimeoutMs": <int or null>   # LEGACY: held computeBudgetMs; used
                                           # as the fallback for BOTH above
    }
  `workerDeadlineMs` and `computeBudgetMs` are separate because they are
  different quantities: the first is how long the caller will wait, the second is
  how long the engine may think. They were the same field, set to the compute
  budget, which made the wall deadline far tighter than intended — see
  _deadline_monotonic.
  Only `engineTurnRequest` (the redacted protocol payload) carries
  game state. Worker has no access to canonical events, GameState,
  master seed, or opp clock — the redaction-tested boundary
  (apps/server/src/engine-protocol/build.ts).

  Each request yields exactly one response line:
    {"requestId": "...", "ok": true,  "response": {...same shape as one-shot...}}
    {"requestId": "...", "ok": false, "error": "..."}

  EOF on stdin → strategy cleanup → exit 0.

Per-request state (e.g. Tier-1 belief filter) is reset via strategy.reset()
before each request so games never leak state between turns. The expensive
*construction* (torch imports, weight loading) happens once at worker startup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("PYTHON_ENGINE_LAB_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC = ROOT / "src"
VENDOR = ROOT / "vendor"
for path in (VENDOR, SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import chess

from fow_chess.engine_protocol import request_from_json
from fow_chess.protocol_adapter import (
    board_from_request,
    build_perspective_view,
    feed_transcript_tail,
    replay_transcript_into_strategy,
)
from fow_chess import rust_health
from fow_chess.selfplay import PerspectiveView
from fow_chess.strategies import RandomStrategy
from fow_chess.visibility import visible_piece_map, visible_squares
try:
    from fow_chess.tournament.config import canonical_hash, load_config
    from fow_chess.tournament.runtime import bot_runtime
except ImportError:  # chess-only build: legacy tier1 snapshot engines absent
    canonical_hash = load_config = bot_runtime = None  # type: ignore[assignment]

TIER1_CONFIG_HASH = "b22f29dd73f5"
DEADLINE_GUARD_MS = int(os.environ.get("PYTHON_LIVE_DEADLINE_GUARD_MS", "1200"))
PICK_DEADLINE_GUARD_MS = int(os.environ.get("PYTHON_LIVE_PICK_DEADLINE_GUARD_MS", "250"))
MIN_PICK_BUDGET_MS = 50
MIN_STRATEGY_PICK_BUDGET_MS = int(
    os.environ.get("PYTHON_LIVE_MIN_STRATEGY_PICK_BUDGET_MS", "3000")
)
# Stateful-session opt-in for the tier1 path. v2 is always stateful (delta-feed,
# commit c203304); tier1 historically reset+replayed the full transcript every
# move, which blows the budget late-game and drops to the deadline-guard. Set
# this to enable delta-feed for tier1 too. OFF by default: the change evolves
# the particle filter's RNG differently than per-move re-seed, so it shifts play
# and must be bakeoff-validated before flipping on in prod. See
# docs/engine/anytime-search-contract.md.
TIER1_STATEFUL_SESSION = os.environ.get("PYTHON_LIVE_STATEFUL_SESSION", "0") not in (
    "",
    "0",
    "false",
    "False",
)
DEFAULT_LIVE_STOCKFISH_TIME_CAP_SECONDS = "0.05"
DEFAULT_LIVE_STOCKFISH_SLACK_SECONDS = "0.05"
MATERIAL_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 100_000,
}
TIER1_LIVE_ENGINES: dict[str, dict[str, str]] = {
    "python-tier1-v0.9.5": {
        "tier1Version": "0.9.5",
        "playSignature": "372b4bb6c064",
        "engineVersion": "v0.9.5-tactical-patches@372b4bb6c064",
    },
    "python-tier1-v0.9.1": {
        "tier1Version": "0.9.1",
        "playSignature": "8918f287499f",
        "engineVersion": "v0.9.1-pawn-shield-diagonal@8918f287499f",
    },
    "python-tier1-v0.7.22": {
        "tier1Version": "0.7.22",
        "playSignature": "5d3ddffa74f6",
        "engineVersion": "v0.7.22-king-risk@5d3ddffa74f6",
    },
    "python-tier1-v0.8.9": {
        "tier1Version": "0.8.9",
        "playSignature": "2c010d792075",
        "engineVersion": "v0.8.9-repair-caps@2c010d792075",
    },
    # Use current src/fow_chess/ (v0.9.5-equivalent with info_reveal_bonus_coef=25
    # and the rest of the post-eval layer enabled). Empty engineVersion → runtime
    # skips the snapshot load and uses live source. Local-only via the
    # MISTBOARD_EXTRA_PLAYABLE_ENGINES env var; not in PROD_PLAYABLE_ENGINE_IDS.
    "python-tier1-current": {
        "tier1Version": "current",
        "playSignature": "current",
        "engineVersion": "",
    },
}


# v2 GT-CFR engine (the EngineV2 / Obscuro-architecture engine), current live
# source. Local-only — register via MISTBOARD_EXTRA_PLAYABLE_ENGINES and the
# server-side engine registry; NOT in PROD_PLAYABLE_ENGINE_IDS. Used for human
# benchmarking + bringing v2 onto the live path. NOTE: the worker resets+replays
# the transcript per request (stateless; Phase-4 stateful session is future), so
# v2's belief enumeration re-runs each move — late-game moves get slow. The
# server entry must set a generous livePolicy.timeoutMs to avoid the deadline
# guard substituting a move.
# python-v2-v1.0 = "Misty 1.0" (launched as "Misty Max") — the FROZEN shipped v2 (gadget-off + early-stop +
# king-aware + clock + bottom-K, validated 2026-06-02). python-v2-current stays
# the moving dev target (local-only). Both route to the same STRONGEST build; the
# registry pins which config each id represents (mirrors the tier1-vX.Y pattern).
#
# GROUND TRUTH: the "SHIPPED/LIVE/frozen" tags in the comments below are HISTORICAL
# and NOT kept in sync — v1.0, v1.1, and v1.2 are each called "shipped" here, all
# stale. Do NOT infer the live engine or its config from these comments. Resolve
# config with `scripts/show_engine_config.py` (see docs/engine/engine-config-truth.md);
# WHICH engine-id is live is chosen PLATFORM-side (mistboard server), not here.
V2_LIVE_ENGINES = {
    "python-v2-current", "python-v2-v1.0",
    # ★ v1.1 SHIPPED (2026-06-16): the player-facing release that supersedes v1.0.
    "python-v2-v1.1",
    # ★ v1.2 SHIPPED + LIVE (2026-06-19): v1.1 + carryover fix (prune + book DORMANT).
    "python-v2-v1.2",
    # ★ v1.3 OPENING-HARDENING candidate (2026-06-20): v1.2 + adaptive prune ON +
    # curated opening book ON. Buildable/bakeoffable; NOT offered to prod until the
    # human-anchored gate clears (the prune ships gated on a human game, north-star).
    "python-v2-v1.3",
    # ★ v1.4 CASTLE-INTO-CHECK fix (2026-06-20): v1.3 profile + search move-gen now
    # generates fog-castles, so the engine devalues a castle that walks the king onto a
    # fog-attacked square (prod a6f2e491 O-O->Qxg8). Base-code fix, not a profile flag.
    "python-v2-v1.4",
    # ★ v1.5 OPENING-BOOK update (2026-06-21): v1.4 profile + curated book (drop redundant
    # Nc3 forces, force ...dxe4 for the move-2 c6 slip). Base-data change, not a profile flag.
    "python-v2-v1.5",
    # Local-only A/B for the human gadget match (2026-06-14): same v2 build path,
    # different engine_profile. Mapped below; default (and v1.0/current) = strongest.
    "python-v2-strongest", "python-v2-faithful",
    # Local-only A/B for the king-safe human gate (2026-06-15): v1.0 + king-only
    # commit backstop (engine_profile v1.1-rc2). Mapped below.
    "python-v2-kingsafe",
    # Local-only A/B for the adaptive-prune human gate (2026-06-16): SHIPPED v1.1 +
    # the |P|-adaptive catastrophe prune (engine_profile v1.1-rc3). Mapped below.
    "python-v2-adaptive",
    # Local-only A/B for the carryover-fix human gate (2026-06-19): v1.1 + carryover
    # OFF (engine_profile v1.1-rc6) — opening corruption fix + warm==cold. Mapped below.
    "python-v2-nocarry",
}
# engine-id -> engine_profile.PROFILES VERSION (1:1). Explicit, no silent
# fall-through: python-v2-v1.0 is pinned to the frozen v1.0, so a future change to
# the v1.0/strongest constant can't silently change what the shipped engine serves.
# "v1.0" and "strongest" resolve to the same frozen config (dev alias); python-v2-
# faithful serves the v1.1 release candidate for the local A/B.
_V2_PROFILE_BY_ID = {
    "python-v2-v1.0": "v1.0",          # frozen first engine ("Misty 1.0", historical)
    "python-v2-v1.1": "v1.1",          # python-v2-v1.1 ("Misty 1.1") — superseded by v1.2
    "python-v2-v1.2": "v1.2",          # gadget-on + carryover fix (v1.3-v1.5 supersede; old "SHIPPED" tag was stale)
    "python-v2-v1.3": "v1.3",          # v1.2 + adaptive prune + book
    "python-v2-v1.4": "v1.4",          # v1.3 profile + fog-castle move-gen (base-code fix)
    "python-v2-v1.5": "v1.5",          # v1.4 profile + curated book (latest tagged; prod selection is platform-side)
    "python-v2-current": "v1.1",       # dev alias -> v1.1 (NOT the latest; v1.5 is newer). "tracks shipped" was stale
    "python-v2-strongest": "v1.0",     # dev alias of v1.0 (gadget-off)
    "python-v2-faithful": "v1.1-rc1",  # v1.1 release candidate (faithful/gadget, local A/B)
    "python-v2-kingsafe": "v1.1-rc2",  # v1.1 release candidate (king-safe distillation, local A/B)
    "python-v2-adaptive": "v1.1-rc5",  # v1.1 + adaptive prune + carryover gate + king-step floor (local A/B human gate)
    "python-v2-nocarry": "v1.1-rc6",   # v1.1 + carryover OFF (opening corruption fix + warm==cold) (local A/B human gate)
}

# Clock-aware per-move budgeting (the 3+2-flagship fix) now lives in the ENGINE
# (cfr/time_manager.py, used by EngineV2Strategy.pick_move when FOW_V2_CLOCK_TIME=1),
# so the SAME logic governs the selfplay/bakeoff harness and the live worker —
# the bakeoff measures time-management as PvE experiences it. The worker just
# (a) sets FOW_V2_CLOCK_TIME=1 at build time and (b) feeds the REAL game clock on
# the pick view (see the v2 clock-restore at the pick site). Tunable via
# FOW_V2_TIME_* env. See docs/engine/v2-release-prep-2026-06-01.md.

# Per-worker stateful session for v2: the worker reuses one strategy across
# requests, so we keep its belief alive between moves and feed only new
# observations. Resetting + replaying the whole transcript each move re-runs
# v2's (exploding) belief enumeration and blows the per-move watchdog late-game.
# Single-session (a serial local game always lands on the same pool worker);
# cold-replays on a new game / shrunk transcript / fresh worker process.
_LIVE_SESSION: dict[str, Any] = {"game_id": None, "processed_len": 0}

# python-chess encodes castling as king-to-destination-file (e1g1/e1c1/e8g8/
# e8c8); the FoW protocol's legal-move list encodes it as king-to-rook-square
# (e1h1/e1a1/e8h8/e8a8). EngineV2 generates moves via python-chess on its belief
# boards, so its castle comes out in the king-dest form and fails the
# own_legal_moves match. Remap to the server's form at the boundary. (Standard
# start squares; chess960 castles aren't remapped — live games are standard.)
_CASTLE_REMAP = {
    (chess.E1, chess.G1): (chess.E1, chess.H1),
    (chess.E1, chess.C1): (chess.E1, chess.A1),
    (chess.E8, chess.G8): (chess.E8, chess.H8),
    (chess.E8, chess.C8): (chess.E8, chess.A8),
}


def _to_server_castle_encoding(move: chess.Move) -> chess.Move:
    if move.promotion is None and move.drop is None:
        mapped = _CASTLE_REMAP.get((move.from_square, move.to_square))
        if mapped is not None:
            return chess.Move(mapped[0], mapped[1])
    return move


def _selftest_move(strategy: Any) -> str:
    """Run ONE real move on the opening position before signalling ready.

    R1-prevent: the worker's `ready` line must mean "I can actually serve a
    move", not just "the process booted". rust_health is already checked above,
    but Stockfish is resolved LAZILY in the leaf eval — so a worker with a
    missing/unlaunchable Stockfish booted `ready` and only failed on the first
    LIVE move (forfeit, room 81e7b246). Exercising a full pick_move here — rust
    belief enumeration + Stockfish leaf eval + GT-CFR search + purification —
    surfaces that at startup instead. Raises on any failure; caller turns it
    into a `ready_error` so the pool refuses the worker.

    Gated by FOW_WORKER_SELFTEST=1 (set by the live pool spawn); bakeoff/
    experiment spawns leave it off so they don't pay the warmup cost.
    """
    board = chess.Board()
    budget_ms = int(os.environ.get("FOW_WORKER_SELFTEST_BUDGET_MS", "3000"))
    # Initialize belief for a fresh game (white to move) — same as the cold-start
    # path (replay_transcript_into_strategy) and selfplay.play_game's reset().
    strategy.reset(chess.WHITE)
    view = PerspectiveView(
        perspective=chess.WHITE,
        own_legal_moves=list(board.pseudo_legal_moves),
        visible_squares=visible_squares(board, chess.WHITE),
        visible_piece_map=visible_piece_map(board, chess.WHITE),
        clock_remaining_ms=budget_ms,
        increment_ms=0,
    )
    move = strategy.pick_move(view)
    if move not in view.own_legal_moves:
        raise RuntimeError(f"selftest move {move.uci()} illegal at opening position")
    return move.uci()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stockfish", default=None)
    parser.add_argument("--game", default="dark-chess")
    args = parser.parse_args()

    # Xiangqi workers are fully separate paths (their own engines + handlers), so
    # the chess worker below is byte-untouched. Dedicated variant pools spawn
    # these with --game.
    if args.game == "dark-mini-xiangqi":
        return _main_mini(args)
    if args.game == "dark-xiangqi":
        return _main_xiangqi(args)

    engine_id = args.engine_id
    seed = args.seed
    stockfish_path = args.stockfish or os.environ.get("PYTHON_ENGINE_STOCKFISH_PATH") or os.environ.get("STOCKFISH_PATH") or "stockfish"
    spec: dict[str, Any] = {"id": engine_id}

    # Visibility, NOT a hard fail: the live worker must keep serving moves even
    # on the Python fallback (a hard exit on a missing .so would be a worse
    # outage than slow play). Warn loudly to stderr so a degraded deploy is
    # diagnosable; the ~500x slowdown otherwise looks like a strength bug and
    # blows the late-game watchdog. (Bakeoffs DO hard-fail, via FOW_REQUIRE_RUST.)
    if not rust_health.available():
        print(
            "WARNING: fow_rust extension unavailable/incomplete — serving on the "
            "~500x-slower Python fallback; late-game timeouts likely.\n"
            + rust_health.report(),
            file=sys.stderr,
            flush=True,
        )

    _configure_live_stockfish_budget()
    runtime = _StrategyRuntime(spec, seed, stockfish_path)
    try:
        strategy = runtime.enter()
    except Exception as exc:
        _emit({"kind": "ready_error", "engineId": engine_id, "error": str(exc)})
        return 2

    # Boot-time resolved-config dump: log exactly which engine this worker runs
    # (STRONGEST profile knobs + env toggles) + a comparable hash, so prod-vs-
    # bakeoff flag drift is visible at a glance instead of silent. The bakeoff
    # logs the same line; mismatched hashes => the bakeoff isn't testing prod.
    from fow_chess import engine_config

    engine_config.dump(lambda s: print(s, file=sys.stderr, flush=True))

    # R1-prevent: prove we can serve a move (Stockfish + rust + search) before
    # signalling ready, so a broken worker fails at startup, not on the first
    # live move. Gated to the live pool spawn (FOW_WORKER_SELFTEST=1).
    if os.environ.get("FOW_WORKER_SELFTEST") == "1":
        try:
            selftest_move = _selftest_move(strategy)
        except Exception as exc:
            _emit(
                {
                    "kind": "ready_error",
                    "engineId": engine_id,
                    "error": f"selftest failed: {exc}",
                }
            )
            runtime.exit()
            return 2
        print(
            f"selftest ok: {engine_id} produced {selftest_move} at opening position",
            file=sys.stderr,
            flush=True,
        )

    _emit({"kind": "ready", "engineId": engine_id, "pid": os.getpid()})

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request_started = time.monotonic()
            request_id: str | None = None
            try:
                request = json.loads(line)
                request_id = str(request.get("requestId") or "")
                response = _handle_request(strategy, spec, request, request_started)
                _emit({"requestId": request_id, "ok": True, "response": response})
            except Exception as exc:
                _debug("request-error", request_started, requestId=request_id, error=str(exc))
                _emit({"requestId": request_id or "", "ok": False, "error": str(exc)})
                # Desync hardening: a stateful request can partially advance the
                # belief (feed_transcript_tail) and then throw, while processed_len
                # is NOT updated — re-feeding the tail next turn would double-apply
                # observations and silently corrupt the belief for the rest of the
                # game. Invalidate the session so the next turn cold-starts (full
                # replay from the authoritative transcript sent every turn).
                _LIVE_SESSION["game_id"] = None
    finally:
        runtime.exit()
    return 0


# ---------------------------------------------------------------------------
# Dark Mini Xiangqi worker path
#
# Fully separate from the chess worker above (own engine, handler, session) so
# the chess serving path stays byte-identical. A dedicated DMX pool spawns this
# with --game dark-mini-xiangqi; engine_id is the DMX engine id (e.g.
# python-dmx-v1.0). The mini engine is plain EngineV2(rules=MiniXiangqiRules())
# and the move ↔ wire bridge is fow_chess.mini_xiangqi.protocol_adapter.
# ---------------------------------------------------------------------------

# Per-game session for the DMX engine. EngineV2's perspective is fixed at
# construction (chess's strategy.reset(perspective) has no mini equivalent), so a
# new game / perspective / shortened transcript rebuilds the engine; the
# steady-state path append-feeds only the transcript tail (belief kept warm).
_MINI_SESSION: dict[str, Any] = {
    "engine": None,
    "game_id": None,
    "perspective": None,
    "processed_len": 0,
}
_DMX_PROFILE_BY_ID = {
    # Public DMX engine id. Pin explicitly instead of relying on the profile
    # module's moving default, mirroring _V2_PROFILE_BY_ID for dark chess.
    "python-dmx-v1.0": "recommended",
}


def _mini_profile_from_env(engine_id: str | None = None) -> Any:
    from fow_chess.mini_xiangqi.profile import mini_profile

    if os.environ.get("FOW_DMX_PROFILE"):
        return mini_profile(env=True)
    return mini_profile(_DMX_PROFILE_BY_ID.get(engine_id), env=True)


def _mini_fsf_available() -> bool:
    from fow_chess.mini_xiangqi.fsf_leaf_eval import fsf_available

    return fsf_available()


def _make_mini_leaf_eval() -> Any:
    """FSF (real-xiangqi eval, strongest) when its binary is present; otherwise
    the material stub. Prod (no FSF binary) serves on the stub — acceptable for
    the current-strength launch; the FSF/blueprint upgrade is the strength push."""
    from fow_chess.mini_xiangqi.fsf_leaf_eval import MiniFSFLeafEval, fsf_available
    from fow_chess.mini_xiangqi.leaf_eval import MiniMaterialLeafEval

    return MiniFSFLeafEval() if fsf_available() else MiniMaterialLeafEval()


def _build_mini_engine(perspective: str, seed: int, engine_id: str | None = None) -> Any:
    import random

    from fow_chess.engine_v2 import EngineV2
    from fow_chess.mini_xiangqi.rules import MiniXiangqiRules

    prof = _mini_profile_from_env(engine_id)
    return EngineV2(
        perspective,
        rules=MiniXiangqiRules(),
        stockfish=_make_mini_leaf_eval(),
        use_rust_state=prof.use_rust_state,
        rng=random.Random(seed),
        p_max_size=prof.max_size,
        win_fast=True,
        resolve_gadget=prof.resolve_gadget,
        resolve_cvar_q=prof.resolve_cvar_q,
        gadget_faithful=prof.gadget_faithful,
        gadget_iterative=prof.gadget_iterative,
        gadget_alpha=prof.gadget_alpha,
        expansion_budget=prof.expansion_budget,
        commit_royal_guard=prof.commit_royal_guard,
        commit_royal_gap=prof.commit_royal_gap,
        commit_royal_floor=prof.commit_royal_floor,
        commit_royal_threat_guard=prof.commit_royal_threat_guard,
        commit_material_guard=prof.commit_material_guard,
        commit_material_gap=prof.commit_material_gap,
        commit_material_floor=prof.commit_material_floor,
        commit_material_min_value=prof.commit_material_min_value,
    )


def _mini_budget_seconds(req: Any, request: dict[str, Any]) -> float | None:
    """Per-move wall budget. Hard-capped by the worker watchdog (a slow move must
    never trip the room timeout) and clock-aware (so the engine can't flag): a
    small fraction of the bank plus most of the increment keeps per-move spend
    below the increment as the bank drains. Returns None only in untimed dev with
    no watchdog (engine then runs to its iteration cap)."""
    watchdog_ms = _parse_optional_int(request.get("watchdogTimeoutMs"))
    hard_s = max(0.1, (watchdog_ms - DEADLINE_GUARD_MS) / 1000.0) if watchdog_ms else None
    rem = req.clock.remaining_ms
    if rem is None:
        return min(hard_s, 2.0) if hard_s is not None else hard_s
    inc = req.clock.increment_ms or 0
    soft_s = max(0.1, rem / 1000.0 * 0.04 + (inc / 1000.0) * 0.8)
    return min(hard_s, soft_s) if hard_s is not None else soft_s


def _mini_selftest_move(seed: int) -> str:
    """R1-prevent for DMX: prove a full move (rust belief + leaf eval + GT-CFR)
    on the opening position before signalling ready. Raises on any failure."""
    from fow_chess.mini_xiangqi.board import RED, MiniBoard

    eng = _build_mini_engine(RED, seed)
    try:
        prof = _mini_profile_from_env()
        budget_s = int(os.environ.get("FOW_WORKER_SELFTEST_BUDGET_MS", "3000")) / 1000.0
        move = eng.choose_move(
            iterations=prof.iterations,
            i_sample_size=prof.i_sample_size,
            time_budget_seconds=budget_s,
            kluss_k=prof.kluss_k,
            use_rust_tree=prof.use_rust_tree,
        )
        legal = {(m.from_square, m.to_square) for m in MiniBoard().pseudo_legal_moves()}
        if (move.from_square, move.to_square) not in legal:
            raise RuntimeError(
                f"selftest move {move.from_square}->{move.to_square} illegal at opening"
            )
        return f"{move.from_square}->{move.to_square}"
    finally:
        eng.close()


def _main_mini(args: argparse.Namespace) -> int:
    engine_id = args.engine_id
    seed = args.seed

    # Visibility, not a hard fail (mirrors the chess path) — keep serving on the
    # slow Python fallback rather than turning a missing .so into an outage.
    prof = _mini_profile_from_env(engine_id)
    if (prof.use_rust_state or prof.use_rust_tree) and not rust_health.available():
        print(
            "WARNING: fow_rust extension unavailable/incomplete — DMX serving on "
            "the ~500x-slower Python fallback; late-game timeouts likely.\n"
            + rust_health.report(),
            file=sys.stderr,
            flush=True,
        )

    leaf_kind = "fsf" if _mini_fsf_available() else "material"
    print(
        f"dmx worker: engine_id={engine_id} leaf={leaf_kind} "
        f"profile={prof}",
        file=sys.stderr,
        flush=True,
    )

    if os.environ.get("FOW_WORKER_SELFTEST") == "1":
        try:
            selftest_move = _mini_selftest_move(seed)
        except Exception as exc:
            _emit({"kind": "ready_error", "engineId": engine_id, "error": f"selftest failed: {exc}"})
            return 2
        print(
            f"selftest ok: {engine_id} produced {selftest_move} at opening position",
            file=sys.stderr,
            flush=True,
        )

    _emit({"kind": "ready", "engineId": engine_id, "pid": os.getpid()})

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request_started = time.monotonic()
            request_id: str | None = None
            try:
                request = json.loads(line)
                request_id = str(request.get("requestId") or "")
                response = _handle_mini_request(engine_id, request, request_started)
                _emit({"requestId": request_id, "ok": True, "response": response})
            except Exception as exc:
                _debug("request-error", request_started, requestId=request_id, error=str(exc))
                _emit({"requestId": request_id or "", "ok": False, "error": str(exc)})
                # Desync hardening (see chess loop): invalidate the session on any
                # error so the next turn cold-starts instead of re-feeding a tail
                # onto a partially-advanced belief.
                _MINI_SESSION["game_id"] = None
    finally:
        eng = _MINI_SESSION.get("engine")
        if eng is not None:
            eng.close()
            _MINI_SESSION["engine"] = None
    return 0


def _handle_mini_request(
    engine_id: str,
    request: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """DMX analogue of _handle_request. Drives EngineV2(mini) directly from the
    (variant-aware) protocol request via the mini adapter — no chess board, no
    Stockfish, no castle remap. Response is the PythonPoolResponse shape with
    7-wide square names."""
    from fow_chess import engine_protocol as proto
    from fow_chess.mini_xiangqi import protocol_adapter as adapter

    req = proto.request_from_json(request["engineTurnRequest"])
    _debug(
        "request-loaded",
        started,
        roomId=req.game_id,
        engineId=req.engine_id,
        gameSpecId=req.game_spec_id,
        color=req.color,
        ply=req.ply,
        legalCount=len(req.legal_moves),
        clockRemainingMs=req.clock.remaining_ms,
        incrementMs=req.clock.increment_ms,
        workerBudgetMs=_parse_optional_int(request.get("watchdogTimeoutMs")),
    )

    if not req.legal_moves:
        raise RuntimeError("no legal moves available")

    perspective = adapter.color_from_protocol(req.color)
    transcript_len = len(req.observation_transcript) if req.observation_transcript else 0

    eng = _MINI_SESSION.get("engine")
    cold = (
        eng is None
        or _MINI_SESSION["game_id"] != req.game_id
        or _MINI_SESSION["perspective"] != perspective
        or transcript_len < _MINI_SESSION["processed_len"]
    )
    if cold:
        if eng is not None:
            eng.close()
        eng = _build_mini_engine(perspective, req.engine_seed, req.engine_id)
        adapter.replay_transcript_into_engine(eng, req)
        _MINI_SESSION.update(
            engine=eng, game_id=req.game_id, perspective=perspective, processed_len=transcript_len
        )
        _debug("transcript-replayed", started, mode="cold", transcriptLen=transcript_len)
    else:
        _MINI_SESSION["processed_len"] = adapter.feed_transcript_tail_into_engine(
            eng, req, _MINI_SESSION["processed_len"]
        )
        _debug("transcript-replayed", started, mode="delta", transcriptLen=transcript_len)

    budget_s = _mini_budget_seconds(req, request)
    prof = _mini_profile_from_env(req.engine_id)
    _debug("pick-started", started, beliefSize=eng.enumerator.size,
           budgetMs=round((budget_s or 0) * 1000), profile=prof.name,
           useRustState=prof.use_rust_state, useRustTree=prof.use_rust_tree)
    move = eng.choose_move(
        iterations=prof.iterations,
        i_sample_size=prof.i_sample_size,
        time_budget_seconds=budget_s,
        kluss_k=prof.kluss_k,
        use_rust_tree=prof.use_rust_tree,
    )
    _debug("pick-finished", started, move=f"{move.from_square}->{move.to_square}")

    # Belief-legal can differ from truth-legal under fog. The platform validates
    # and substitutes, but return a truth-legal move when ours isn't so a rare
    # fog mismatch can't cost a forfeit.
    legal_keys = {(m.from_square, m.to_square) for m in req.legal_moves}
    decision_source = "v2"
    if (move.from_square, move.to_square) not in legal_keys:
        _debug("belief-illegal-substitute", started, move=f"{move.from_square}->{move.to_square}")
        move = adapter.move_from_protocol(req.legal_moves[0])
        decision_source = "belief-illegal-substitute"

    return {
        "roomId": req.game_id,
        "engine": {"id": engine_id},
        "decisionSource": decision_source,
        "move": {
            "from": proto._int_to_square(move.from_square, 7),
            "to": proto._int_to_square(move.to_square, 7),
        },
    }


# ---------------------------------------------------------------------------
# Full Dark Xiangqi worker path
#
# Same long-lived EngineV2 shape as DMX, but with the 9x10 Xiangqi rules and
# Pikafish leaf eval. Profile is the frozen local candidate from the strength
# climb: 64x12, 20M belief cap, native rust state/tree, Pikafish depth 1, and
# the royal commit guards.
# ---------------------------------------------------------------------------

_XIANGQI_SESSION: dict[str, Any] = {
    "engine": None,
    "game_id": None,
    "perspective": None,
    "processed_len": 0,
}
_XIANGQI_PROFILE_BY_ID = {
    "python-fdx-v1.0": "frozen-64x12-20m",
    "python-fdx-v1.1": "guarded-64x32-20m",
}

# Named-profile bundle defaults — the code-legible source of truth (issue #6).
# Individual FOW_XIANGQI_* env vars still override any field; a profile name absent
# here falls back to the stripped legacy bundle (= python-fdx-v1.0 prior behavior).
# `material_guard`/`material_adaptive` use tri-state: None = "defer to env" (legacy,
# off unless Railway sets it); a bool pins it in code.
_XIANGQI_PROFILE_BUNDLES: dict[str, dict[str, Any]] = {
    # Legacy served play-bot: stripped belief (|I|=12), no KLUSS/gadget/veto, and the
    # material-catastrophe stack deferred to env -> byte-identical to prior v1.0.
    "frozen-64x12-20m": {
        "faithful": False,
        "material_guard": None,
        "material_adaptive": None,
        "material_tau": None,
    },
    # v1.1 (2026-07-15): faithful coverage (|I|=32, KLUSS=2, Resolve gadget + alpha)
    # PLUS the material-catastrophe stack — adaptive material prune (tau=0.15) +
    # general veto — baked ON in code, no env required. Human-validated: this config
    # beat the author 3/3 in live PvE (the opening cannon-hang class did not recur).
    "guarded-64x32-20m": {
        "faithful": True,
        "material_guard": True,
        "material_adaptive": True,
        "material_tau": 0.15,
    },
}


def _xiangqi_profile(engine_id: str | None = None) -> dict[str, Any]:
    name = os.environ.get("FOW_XIANGQI_PROFILE") or _XIANGQI_PROFILE_BY_ID.get(
        engine_id, "frozen-64x12-20m"
    )
    bundle = _XIANGQI_PROFILE_BUNDLES.get(name, _XIANGQI_PROFILE_BUNDLES["frozen-64x12-20m"])

    def _flag(var: str, default_on: bool) -> bool:
        return os.environ.get(var, "1" if default_on else "0") not in ("", "0", "false", "False")

    def _tri(var: str, bundle_val: bool | None) -> bool | None:
        # env wins; else the profile's pinned bool; else None = defer to the
        # engine's own env fallback (legacy behavior).
        raw = os.environ.get(var)
        if raw is not None:
            return raw not in ("", "0", "false", "False")
        return bundle_val

    # FAITHFUL = the Dark-Chess v1.1 arm ported to Full Dark Xiangqi: bigger |I|,
    # KLUSS, and the Resolve gadget + non-uniform alpha (CVaR off). The default now
    # comes from the resolved profile bundle (guarded -> on, frozen -> off); an
    # explicit FOW_XIANGQI_FAITHFUL still overrides. The guards/search machinery run
    # per-move regardless; only the bundle toggles what is actually active.
    faithful = _flag("FOW_XIANGQI_FAITHFUL", bundle["faithful"])
    kluss_env = int(os.environ.get("FOW_XIANGQI_KLUSS_K", "2" if faithful else "0"))
    tau_env = os.environ.get("FOW_XIANGQI_MATERIAL_TAU")
    material_tau = float(tau_env) if tau_env is not None else bundle.get("material_tau")
    # Suffix only when faithful deviates from the bundle default (an env toggle), so
    # a guarded profile stays "guarded-64x32-20m" not "...-faithful".
    suffix = "-faithful" if (faithful and not bundle["faithful"]) else (
        "-stripped" if (not faithful and bundle["faithful"]) else ""
    )
    return {
        "name": name + suffix,
        "faithful": faithful,
        "iterations": int(os.environ.get("FOW_XIANGQI_ITERS", "64")),
        "i_sample_size": int(os.environ.get("FOW_XIANGQI_I_SAMPLE", "32" if faithful else "12")),
        "max_size": int(os.environ.get("FOW_XIANGQI_P_MAX", "20000000")),
        "pikafish_depth": int(os.environ.get("FOW_XIANGQI_PIKAFISH_DEPTH", "1")),
        "use_rust_state": _flag("FOW_XIANGQI_RUST_STATE", True),
        "use_rust_tree": _flag("FOW_XIANGQI_RUST_TREE", True),
        "commit_royal_gap": float(os.environ.get("FOW_XIANGQI_ROYAL_GAP", "0.2")),
        "kluss_k": kluss_env if kluss_env > 0 else None,
        "resolve_gadget": _flag("FOW_XIANGQI_RESOLVE_GADGET", faithful),
        "resolve_cvar_q": float(os.environ.get("FOW_XIANGQI_RESOLVE_CVAR_Q", "0.0")),
        "gadget_alpha": _flag("FOW_XIANGQI_GADGET_ALPHA", faithful),
        "gadget_iterative": _flag("FOW_XIANGQI_GADGET_ITERATIVE", faithful),
        # Hard general-safety veto (terminal, exact over a large belief sample) — the
        # gadget/gapped-royal-guard are value-blind to a next-ply general capture, so
        # this is the terminal backstop. On by default with the faithful stack.
        "general_veto": _flag("FOW_XIANGQI_GENERAL_VETO", faithful),
        # Material-catastrophe commit stack (issue #6: pinned in code for guarded
        # profiles, tri-state None = defer to env for the legacy frozen bot).
        "commit_material_guard": _tri("FOW_XIANGQI_COMMIT_MATERIAL_GUARD", bundle.get("material_guard")),
        "commit_material_adaptive": _tri("FOW_XIANGQI_MATERIAL_ADAPTIVE", bundle.get("material_adaptive")),
        "material_tau": material_tau,
    }


def _xiangqi_pikafish_available() -> bool:
    from fow_chess.xiangqi.pikafish_leaf_eval import pikafish_available

    return pikafish_available()


def _make_xiangqi_leaf_eval(depth: int) -> Any:
    from fow_chess.xiangqi.leaf_eval import XiangqiMaterialLeafEval
    from fow_chess.xiangqi.pikafish_leaf_eval import PikafishLeafEval, pikafish_available

    return PikafishLeafEval(depth=depth) if pikafish_available() else XiangqiMaterialLeafEval()


def _build_xiangqi_engine(perspective: str, seed: int, engine_id: str | None = None) -> Any:
    import random

    from fow_chess.engine_v2 import EngineV2
    from fow_chess.xiangqi.rules import XiangqiRules

    prof = _xiangqi_profile(engine_id)
    eng = EngineV2(
        perspective,
        rules=XiangqiRules(),
        stockfish=_make_xiangqi_leaf_eval(prof["pikafish_depth"]),
        use_rust_state=prof["use_rust_state"],
        rng=random.Random(seed),
        p_max_size=prof["max_size"],
        win_fast=True,
        commit_royal_guard=True,
        commit_royal_gap=prof["commit_royal_gap"],
        commit_royal_threat_guard=True,
        resolve_gadget=prof["resolve_gadget"],
        resolve_cvar_q=prof["resolve_cvar_q"],
        gadget_alpha=prof["gadget_alpha"],
        gadget_iterative=prof["gadget_iterative"],
        commit_general_veto=prof["general_veto"],
        commit_material_guard=prof["commit_material_guard"],
    )
    # Material adaptive-prune knobs are read off the instance (None -> the engine's
    # own env fallback), so a named profile can bake them without FOW_XIANGQI_* vars.
    eng._commit_material_adaptive = prof["commit_material_adaptive"]
    eng._material_tau = prof["material_tau"]
    # EngineV2 only owns internally-created Stockfish evals; this variant passes
    # a Pikafish/material eval through the same slot, so mark it owned for close().
    eng._owns_stockfish = True
    return eng


def _xiangqi_selftest_move(seed: int, engine_id: str) -> str:
    from fow_chess.xiangqi.board import RED, XiangqiBoard

    eng = _build_xiangqi_engine(RED, seed, engine_id)
    try:
        prof = _xiangqi_profile(engine_id)
        budget_s = int(os.environ.get("FOW_WORKER_SELFTEST_BUDGET_MS", "3000")) / 1000.0
        move = eng.choose_move(
            iterations=prof["iterations"],
            i_sample_size=prof["i_sample_size"],
            kluss_k=prof["kluss_k"],
            time_budget_seconds=budget_s,
            use_rust_tree=prof["use_rust_tree"],
        )
        legal = {(m.from_square, m.to_square) for m in XiangqiBoard().pseudo_legal_moves()}
        if (move.from_square, move.to_square) not in legal:
            raise RuntimeError(
                f"selftest move {move.from_square}->{move.to_square} illegal at opening"
            )
        return f"{move.from_square}->{move.to_square}"
    finally:
        eng.close()


def _main_xiangqi(args: argparse.Namespace) -> int:
    engine_id = args.engine_id
    seed = args.seed
    prof = _xiangqi_profile(engine_id)

    if (prof["use_rust_state"] or prof["use_rust_tree"]) and not rust_health.available():
        print(
            "WARNING: fow_rust extension unavailable/incomplete — full Dark Xiangqi serving on "
            "the ~500x-slower Python fallback; late-game timeouts likely.\n"
            + rust_health.report(),
            file=sys.stderr,
            flush=True,
        )

    leaf_kind = "pikafish" if _xiangqi_pikafish_available() else "material"
    print(
        f"xiangqi worker: engine_id={engine_id} leaf={leaf_kind} profile={prof}",
        file=sys.stderr,
        flush=True,
    )

    if os.environ.get("FOW_WORKER_SELFTEST") == "1":
        try:
            selftest_move = _xiangqi_selftest_move(seed, engine_id)
        except Exception as exc:
            _emit({"kind": "ready_error", "engineId": engine_id, "error": f"selftest failed: {exc}"})
            return 2
        print(
            f"selftest ok: {engine_id} produced {selftest_move} at opening position",
            file=sys.stderr,
            flush=True,
        )

    _emit({"kind": "ready", "engineId": engine_id, "pid": os.getpid()})

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request_started = time.monotonic()
            request_id: str | None = None
            try:
                request = json.loads(line)
                request_id = str(request.get("requestId") or "")
                response = _handle_xiangqi_request(engine_id, request, request_started)
                _emit({"requestId": request_id, "ok": True, "response": response})
            except Exception as exc:
                _debug("request-error", request_started, requestId=request_id, error=str(exc))
                _emit({"requestId": request_id or "", "ok": False, "error": str(exc)})
                _XIANGQI_SESSION["game_id"] = None
    finally:
        eng = _XIANGQI_SESSION.get("engine")
        if eng is not None:
            eng.close()
            _XIANGQI_SESSION["engine"] = None
    return 0


def _handle_xiangqi_request(
    engine_id: str,
    request: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    from fow_chess import engine_protocol as proto
    from fow_chess.xiangqi import protocol_adapter as adapter

    req = proto.request_from_json(request["engineTurnRequest"])
    _debug(
        "request-loaded",
        started,
        roomId=req.game_id,
        engineId=req.engine_id,
        gameSpecId=req.game_spec_id,
        color=req.color,
        ply=req.ply,
        legalCount=len(req.legal_moves),
        clockRemainingMs=req.clock.remaining_ms,
        incrementMs=req.clock.increment_ms,
        workerBudgetMs=_parse_optional_int(request.get("watchdogTimeoutMs")),
    )

    if not req.legal_moves:
        raise RuntimeError("no legal moves available")

    perspective = adapter.color_from_protocol(req.color)
    transcript_len = len(req.observation_transcript) if req.observation_transcript else 0

    eng = _XIANGQI_SESSION.get("engine")
    cold = (
        eng is None
        or _XIANGQI_SESSION["game_id"] != req.game_id
        or _XIANGQI_SESSION["perspective"] != perspective
        or transcript_len < _XIANGQI_SESSION["processed_len"]
    )
    if cold:
        if eng is not None:
            eng.close()
        eng = _build_xiangqi_engine(perspective, req.engine_seed, req.engine_id)
        adapter.replay_transcript_into_engine(eng, req)
        _XIANGQI_SESSION.update(
            engine=eng, game_id=req.game_id, perspective=perspective, processed_len=transcript_len
        )
        _debug("transcript-replayed", started, mode="cold", transcriptLen=transcript_len)
    else:
        _XIANGQI_SESSION["processed_len"] = adapter.feed_transcript_tail_into_engine(
            eng, req, _XIANGQI_SESSION["processed_len"]
        )
        _debug("transcript-replayed", started, mode="delta", transcriptLen=transcript_len)

    budget_s = _mini_budget_seconds(req, request)
    prof = _xiangqi_profile(req.engine_id)
    _debug(
        "pick-started",
        started,
        beliefSize=eng.enumerator.size,
        budgetMs=round((budget_s or 0) * 1000),
        profile=prof["name"],
        useRustState=prof["use_rust_state"],
        useRustTree=prof["use_rust_tree"],
    )
    legal_moves = [adapter.move_from_protocol(m) for m in req.legal_moves]
    move = eng.choose_move(
        iterations=prof["iterations"],
        i_sample_size=prof["i_sample_size"],
        kluss_k=prof["kluss_k"],
        time_budget_seconds=budget_s,
        use_rust_tree=prof["use_rust_tree"],
        legal_moves=legal_moves,
    )
    _debug("pick-finished", started, move=f"{move.from_square}->{move.to_square}")

    legal_keys = {(m.from_square, m.to_square) for m in req.legal_moves}
    decision_source = "v2"
    if (move.from_square, move.to_square) not in legal_keys:
        _debug("belief-illegal-substitute", started, move=f"{move.from_square}->{move.to_square}")
        move = adapter.move_from_protocol(req.legal_moves[0])
        decision_source = "belief-illegal-substitute"

    return {
        "roomId": req.game_id,
        "engine": {"id": engine_id},
        "decisionSource": decision_source,
        "move": {
            "from": proto._int_to_square(move.from_square, 9),
            "to": proto._int_to_square(move.to_square, 9),
        },
        "diagnostics": {
            "beliefSize": eng.enumerator.size,
            "profile": prof["name"],
            "leaf": "pikafish" if _xiangqi_pikafish_available() else "material",
        },
    }


def _handle_request(
    strategy: Any,
    spec: dict[str, Any],
    request: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """Protocol-only handler — engine consumes EngineTurnRequest.

    The redaction-tested boundary: nothing here reads canonical events,
    raw GameState, or any field outside the protocol. All engine inputs
    come from the parsed protocol request.

    Phase 3c (commit following 4cc0319) dropped the legacy events-mode
    handler after the TS payload stopped sending `events`.
    """
    req = request_from_json(request["engineTurnRequest"])
    _debug(
        "request-loaded",
        started,
        roomId=req.game_id,
        engineId=req.engine_id,
        color=req.color,
        ply=req.ply,
        legalCount=len(req.legal_moves),
        clockRemainingMs=req.clock.remaining_ms,
        incrementMs=req.clock.increment_ms,
        workerBudgetMs=_parse_optional_int(request.get("watchdogTimeoutMs")),
    )

    if not req.legal_moves:
        raise RuntimeError("no legal moves available")
    view = build_perspective_view(req)

    deadline = _deadline_monotonic(started, request)
    if _deadline_expired(deadline):
        # No canonical board here; reconstruct a chess.Board view from
        # the protocol observation for the fallback move generator.
        guard_board = board_from_request(req)
        move = _deadline_guard_move(guard_board, view)
        _debug("deadline-guard", started, phaseBefore="transcript-replay", move=move.uci())
        return _move_response(spec, move, req, "deadline-guard")

    # Bring the strategy's belief up to date. Stateful path: feed only the
    # delta since last turn (keeps work across moves). Stateless path: reset and
    # replay the full transcript every move — O(plies) per move, blows the
    # budget late-game and drops to the deadline-guard (room d860f498).
    transcript_len = len(req.observation_transcript) if req.observation_transcript else 0
    _debug("transcript-replay-started", started, transcriptLen=transcript_len)
    engine_id = str(spec.get("id") or "")
    stateful = engine_id in V2_LIVE_ENGINES or (
        TIER1_STATEFUL_SESSION and not isinstance(strategy, RandomStrategy)
    )
    if stateful:
        # Feed only the delta unless continuity is broken (different game, or a
        # shorter transcript than we've already processed → not append-only).
        gid = req.game_id
        if _LIVE_SESSION["game_id"] != gid or transcript_len < _LIVE_SESSION["processed_len"]:
            replay_transcript_into_strategy(strategy, req)  # cold start (resets)
            _LIVE_SESSION["game_id"] = gid
            _LIVE_SESSION["processed_len"] = transcript_len
            _debug("transcript-replayed", started, mode="cold", transcriptLen=transcript_len)
        else:
            feed_transcript_tail(strategy, req, _LIVE_SESSION["processed_len"])
            _debug("transcript-replayed", started, mode="delta",
                   fromIdx=_LIVE_SESSION["processed_len"], transcriptLen=transcript_len)
            _LIVE_SESSION["processed_len"] = transcript_len
    else:
        replay_transcript_into_strategy(strategy, req)
        _debug("transcript-replayed", started, transcriptLen=transcript_len)
    if _deadline_expired(deadline):
        guard_board = board_from_request(req)
        move = _deadline_guard_move(guard_board, view)
        _debug("deadline-guard", started, phaseBefore="pick-started", move=move.uci())
        return _move_response(spec, move, req, "deadline-guard")

    pick_view, pick_budget_ms = _budgeted_pick_view(view, deadline, _compute_budget_ms(request))
    if pick_view is None:
        guard_board = board_from_request(req)
        move = _deadline_guard_move(guard_board, view)
        _debug(
            "deadline-guard",
            started,
            phaseBefore="pick-started",
            move=move.uci(),
            pickBudgetMs=pick_budget_ms,
            minStrategyPickBudgetMs=MIN_STRATEGY_PICK_BUDGET_MS,
        )
        return _move_response(spec, move, req, "deadline-guard")

    _debug(
        "pick-started",
        started,
        ownLegalCount=len(pick_view.own_legal_moves),
        visibleSquareCount=len(pick_view.visible_squares),
        visiblePieceCount=len(pick_view.visible_piece_map),
        pickBudgetMs=pick_budget_ms,
        strategyClockRemainingMs=pick_view.clock_remaining_ms,
        strategyIncrementMs=pick_view.increment_ms,
    )
    # v2 clock-aware budgeting lives INSIDE the engine (pick_move reads
    # view.clock_remaining_ms when FOW_V2_CLOCK_TIME=1) — the SAME code path as
    # the selfplay/bakeoff harness, so the bakeoff measures time-management as
    # PvE experiences it. But _budgeted_pick_view OVERWROTE clock_remaining_ms
    # with a tier-1-synthetic watchdog value; v2 needs the REAL game clock to
    # budget correctly, so restore it from the request. The watchdog `deadline`
    # (+ deadline-guard above) remains the hard safety net.
    engine_pick_view = pick_view
    if engine_id in V2_LIVE_ENGINES:
        engine_pick_view = replace(
            pick_view,
            clock_remaining_ms=req.clock.remaining_ms,
            increment_ms=req.clock.increment_ms,
        )
    _assert_belief_consistent(strategy, req, started)
    move = strategy.pick_move(engine_pick_view)
    telemetry = _v2_decision_telemetry(strategy)
    _debug("pick-finished", started, move=move.uci(), **telemetry)
    if move not in view.own_legal_moves:
        # The engine may have produced a castle in python-chess's king-dest form;
        # remap to the server's king-to-rook-square encoding and re-check.
        remapped = _to_server_castle_encoding(move)
        if remapped in view.own_legal_moves:
            move = remapped
    if move not in view.own_legal_moves:
        raise RuntimeError(f"engine returned illegal move: {move.uci()}")
    if isinstance(strategy, RandomStrategy):
        decision_source = "random"
    elif str(spec.get("id") or "") in V2_LIVE_ENGINES:
        decision_source = "v2"
    else:
        decision_source = "tier1"
    _write_decision_provenance(
        request, req, strategy, move, decision_source, telemetry, engine_id, started
    )
    return _move_response(spec, move, req, decision_source, diagnostics=telemetry or None)


# Config env snapshotted into decision telemetry so a live decision can be
# reproduced offline at the SAME config. The repro-fidelity gap (telemetry had
# |P|/iters/moveRanking but NOT the config) is what made the d5944456 and
# 56961913 live-vs-replay divergences un-diagnosable: an offline replay at a
# slightly different config silently ranks moves differently, and there was no
# way to know. Config flags ONLY — keys containing any deny-token are dropped so
# no secret (DATABASE_URL, API keys) can ever land in the artifact.
_CONFIG_ENV_PREFIXES = ("FOW_", "PYTHON_ENGINE_", "PYTHON_LIVE_")
_CONFIG_ENV_DENY = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASS", "DSN", "URL", "CRED")


def _engine_config_snapshot(strategy: Any, eng: Any) -> dict[str, Any]:
    """Capture the engine's EFFECTIVE config (leaf flags, search knobs, relevant
    env) so a live decision can be replayed offline faithfully. Best-effort;
    never raises — the caller's telemetry must not break the move path."""
    snap: dict[str, Any] = {}
    try:
        from fow_chess.cfr import leaf_eval as _le

        snap["leaf"] = {
            "kingAware": bool(_le._KING_AWARE_LEAF),
            "tanhScaleCp": float(_le._TANH_SCALE_CP),
            "kingBandFloor": float(_le._KING_BAND_FLOOR),
        }
    except Exception:
        pass
    try:
        enum = getattr(eng, "enumerator", None)
        snap["search"] = {
            "iSampleSize": getattr(strategy, "_i_sample_size", None),
            "klussK": getattr(strategy, "_kluss_k", None),
            "resolveGadget": getattr(strategy, "_resolve_gadget", None),
            "useRustTree": getattr(strategy, "_use_rust_tree", None),
            "useLean": getattr(strategy, "_use_lean", None),
            "iterations": getattr(strategy, "_iterations", None),
            "queenPromoTiebreak": getattr(eng, "queen_promo_tiebreak", None),
            "pMax": getattr(enum, "max_size", None),
        }
    except Exception:
        pass
    try:
        env = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(_CONFIG_ENV_PREFIXES) and not any(d in k for d in _CONFIG_ENV_DENY)
        }
        if env:
            snap["env"] = env
    except Exception:
        pass
    return snap


def _rounded(value: Any, digits: int) -> float | None:
    """Round a float-ish telemetry value, or None if it isn't one."""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _rounded_component_ms(sol: Any) -> dict[str, float] | None:
    """gt_cfr's per-component wall-time split, rounded for the artifact payload.

    Returns None when the solve path did not produce one (the non-rust-tree
    solvers), so a missing split is distinguishable from an all-zero one.
    """
    raw = getattr(sol, "component_ms", None)
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for key, value in raw.items():
        rounded = _rounded(value, 1)
        if rounded is not None:
            out[str(key)] = rounded
    return out or None


def _v2_decision_telemetry(strategy: Any) -> dict[str, Any]:
    """Best-effort introspection of a v2 strategy's last decision for the debug
    log: belief size |P|, GT-CFR iterations completed, the full move ranking
    (action_values_at_root, sorted), AND the effective config (so the decision
    can be reproduced offline). Lets us compare the LIVE belief + values to an
    offline full-replay reconstruction — the only way to diagnose live-vs-replay
    divergences (e.g. the d5944456 rook-capture, where replay robustly prefers
    the pawn but live took with the rook; and 56961913 ply 19, where replay
    ranked Bg4 ~6th but live ranked it #1 at the SAME |P|=311). Returns {} for
    non-v2 strategies or on any error: this MUST never affect the move path."""
    try:
        eng = getattr(strategy, "_engine", None)
        sol = getattr(eng, "last_solution", None)
        if eng is None or sol is None:
            return {}
        av = getattr(sol, "action_values_at_root", None) or {}
        ranked = sorted(
            ((m.uci(), round(float(v), 4)) for m, v in av.items()),
            key=lambda kv: -kv[1],
        )
        return {
            "beliefSize": eng.enumerator.size,
            # Downsample observability: cumulative cap-fires this game + the
            # last pre-cap |P| (M). In prod, beliefSize is post-cap, so without
            # these a downsample is invisible (looks like a normal large belief).
            "downsampleCount": getattr(eng.enumerator, "downsample_count", 0),
            "beliefPreCap": getattr(eng.enumerator, "last_pre_cap_count", 0),
            "iters": getattr(sol, "iterations", None),
            # Where the think time actually went. gt_cfr already computes this
            # per decision and it was being dropped on the floor, so every
            # "why is Misty only getting 1-3k iterations" question had to be
            # answered by inference from clock deltas. componentMs splits the
            # search itself (sf_eval / sf_children / eq_pass / select_leaf /
            # kluss / expand_seed); searchSeconds vs the caller's think time
            # exposes the UNBUDGETED remainder — belief enumeration and
            # transport — which is the ~5s floor in mistboard#283.
            "componentMs": _rounded_component_ms(sol),
            "searchSeconds": _rounded(getattr(sol, "elapsed_seconds", None), 3),
            "treeNodes": getattr(sol, "total_tree_nodes", None),
            "moveRanking": ranked,
            # Effective config for faithful offline replay (the repro-fidelity
            # gap behind every prior live-vs-replay mystery).
            "config": _engine_config_snapshot(strategy, eng),
        }
    except Exception:
        return {}


def _assert_belief_consistent(strategy: Any, req: Any, started: float) -> None:
    """Tripwire: the engine's belief MUST agree with the latest observation's
    visible_pieces. Every square the side-to-move can see is a hard fact — present
    identically in every belief world. If a sampled world disagrees, the belief has
    silently diverged from what the engine was TOLD it can see (the exact failure
    mode behind the live-vs-replay mysteries: a move that looks fine against a
    corrupt belief, e.g. capturing a defended pawn because the defender went
    missing from the belief). Log-only by default; FOW_BELIEF_GUARD_RECOVER=1 also
    cold-restarts the belief from the full transcript. Never raises — must not
    affect the move path."""
    try:
        if not req.observation_transcript:
            return
        last = req.observation_transcript[-1]
        if not last.visible_pieces:
            return
        eng = getattr(strategy, "_engine", None)
        size = getattr(getattr(eng, "enumerator", None), "size", 0) if eng else 0
        if size <= 0:
            return
        import random as _random

        fens = eng.enumerator.sample_root_fens(n=min(size, 256), rng=_random.Random(0))
        boards = [chess.Board(f) for f in fens]
        n = len(boards)
        diverging: list[tuple[str, int]] = []
        for sq, vp in last.visible_pieces:
            expected = vp.type if vp.color == "white" else vp.type.lower()
            ok = sum(
                1
                for b in boards
                if (pc := b.piece_at(sq)) is not None and pc.symbol() == expected
            )
            if ok < n:
                diverging.append((chess.square_name(sq), n - ok))
        if diverging:
            diverging.sort(key=lambda kv: -kv[1])
            _debug(
                "belief-inconsistent",
                started,
                ply=req.ply,
                beliefSize=size,
                sampled=n,
                divergingSquares=[s for s, _ in diverging[:8]],
                worstMissing=diverging[0][1],
            )
            # Durable alert (independent of _debug routing): a belief that
            # disagrees with what the engine can see is THE bug we couldn't catch.
            _alert_dir = os.environ.get("FOW_DECISION_LOG_DIR")
            if _alert_dir:
                try:
                    os.makedirs(_alert_dir, exist_ok=True)
                    with open(os.path.join(_alert_dir, "belief_alerts.jsonl"), "a") as _fh:
                        _fh.write(
                            json.dumps(
                                {
                                    "ts_ms": int(time.time() * 1000),
                                    "game_id": req.game_id,
                                    "ply": req.ply,
                                    "beliefSize": size,
                                    "sampled": n,
                                    "divergingSquares": [
                                        {"sq": s, "missingWorlds": c} for s, c in diverging[:16]
                                    ],
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            if os.environ.get("FOW_BELIEF_GUARD_RECOVER") == "1":
                replay_transcript_into_strategy(strategy, req)
                _LIVE_SESSION["processed_len"] = len(req.observation_transcript)
                _debug("belief-guard-recovered", started, ply=req.ply)
    except Exception as ex:  # pragma: no cover - guard must never break the move
        _debug("belief-guard-failed", started, error=str(ex)[:120])


def _write_decision_provenance(
    request: dict[str, Any],
    req: Any,
    strategy: Any,
    move: chess.Move,
    decision_source: str,
    telemetry: dict[str, Any],
    engine_id: str,
    started: float,
) -> None:
    """Durably persist everything needed to REPLAY a live decision offline: the
    exact protocol request (its observation_transcript IS the engine's full input),
    the base seed, the profile, and the resulting move + telemetry (|P|, ranking,
    config). Replay with scripts/replay_decision.py. Gated by FOW_DECISION_LOG_DIR
    (unset => disabled). Secret-safe: the protocol request carries no secrets and
    the config snapshot is deny-filtered. Never raises — must not affect the move
    path. This is the durable audit trail whose absence made every prior
    live-vs-replay divergence un-diagnosable."""
    log_dir = os.environ.get("FOW_DECISION_LOG_DIR")
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        rec = {
            "ts_ms": int(time.time() * 1000),
            "game_id": req.game_id,
            "engine_id": engine_id,
            "profile": _V2_PROFILE_BY_ID.get(engine_id),
            "ply": req.ply,
            "color": req.color,
            "base_seed": getattr(strategy, "_seed", None),
            "chosen_move": move.uci(),
            "decision_source": decision_source,
            "telemetry": telemetry,
            "engineTurnRequest": request.get("engineTurnRequest"),
        }
        path = os.path.join(log_dir, f"{req.game_id}.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as ex:  # pragma: no cover - provenance must never break moves
        _debug("provenance-write-failed", started, error=str(ex)[:120])


def _move_response(
    spec: dict[str, Any], move: chess.Move, req: Any, decision_source: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Worker response shape matching apps/server PythonPoolResponse.

    Castling note: `to` is the king's destination (e1→g1), not the
    rook square. variants.ts:589 explicitly accepts both forms via
    alias generation, so this aligns with python-chess's default
    output without needing a special castling-rewrite pass.

    ``diagnostics`` carries the v2 per-move telemetry (belief size, GT-CFR
    iters, full move ranking — see _v2_decision_telemetry) so the server can
    persist it with the live-engine-decision artifact (observability: diagnose a
    prod blunder from the engine's belief + ranking at that ply, vs only stderr).
    Omitted for the deadline-guard / tier1 / random paths (telemetry empty).
    """
    promo_letter = None
    if move.promotion is not None:
        promo_letter = {
            chess.QUEEN: "queen", chess.ROOK: "rook",
            chess.BISHOP: "bishop", chess.KNIGHT: "knight",
        }[move.promotion]
    return {
        "roomId": req.game_id,
        "engine": _engine_metadata(spec),
        "decisionSource": decision_source,
        "move": {
            "from": chess.SQUARE_NAMES[move.from_square],
            "to": chess.SQUARE_NAMES[move.to_square],
            **({"promotion": promo_letter} if promo_letter else {}),
        },
        **({"diagnostics": diagnostics} if diagnostics else {}),
    }


def _engine_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    engine_id = str(spec.get("id") or "")
    tier1 = TIER1_LIVE_ENGINES.get(engine_id)
    if tier1 is not None:
        return {
            "id": engine_id,
            "tier1Version": tier1["tier1Version"],
            "configHash": TIER1_CONFIG_HASH,
            "playSignature": tier1["playSignature"],
            "engineVersion": tier1["engineVersion"],
        }
    return {"id": engine_id}


class _StrategyRuntime:
    """Holds Tier-1's bot_runtime context manager open across requests."""

    def __init__(self, spec: dict[str, Any], seed: int, stockfish_path: str) -> None:
        self.spec = spec
        self.seed = seed
        self.stockfish_path = stockfish_path
        self._runtime: Any | None = None
        self._strategy: Any | None = None

    def enter(self) -> Any:
        engine_id = str(self.spec.get("id") or "")
        if engine_id in {"python-random-legal", "builtin-random-legal"}:
            self._strategy = RandomStrategy(seed=self.seed)
            return self._strategy
        if engine_id in V2_LIVE_ENGINES:
            # The GT-CFR v2 engine at the INTENDED-STRONGEST profile (the single
            # source of truth — see fow_chess.engine_profile.STRONGEST: i=32 +
            # KLUSS k=2 + CVaR Resolve gadget + king-aware leaf, scale 500).
            # build_strategy() also sets the process-global king-aware flag —
            # this worker serves only v2, so that's scoped to this engine.
            # Deployment knobs are ours: a flat 5s/move budget (the high iter
            # cap inside the profile means TIME binds, not iters), and the |P|
            # cap + bottom-K bound below.
            #
            # MEMORY (validated 2026-05-31, bottom-K cap sweep, 80 games):
            # bottom-K (FOW_BOTTOMK_EXPANSION) bounds the opp-move belief
            # expansion DURING the Rust build instead of materializing the full
            # consistent set M (which reached ~134M = 17x the cap on the worst
            # ply, the OOM driver that SIGKILL'd the 24 GB box in the 2026-05-30
            # mirror run). At a 16M cap: 0 OOM, 0 empty-P collapse, max RSS
            # 8.9 GB / 40 games (fits 24 GB with margin). Caveat: a downsample
            # keeps P NON-EMPTY but may drop the true world (silent strength dip,
            # not a crash) — crash-safety is proven, truth-retention is not
            # measured. See docs/engine/memory-bounded-expansion-and-recovery-2026-05-31.md.
            os.environ.setdefault("FOW_BOTTOMK_EXPANSION", "1")
            # Clock-aware per-move budgeting (3+2 anti-flag): the engine reads the
            # real game clock off the pick view (restored below) and budgets
            # solvently instead of a flat 5s. time_budget_seconds=5.0 remains the
            # static fallback when no clock is present (untimed).
            os.environ.setdefault("FOW_V2_CLOCK_TIME", "1")
            # early_stop is NOT forced here anymore: the profile owns it via
            # apply_process_flags() (STRONGEST=on, FAITHFUL=off — early-stop is
            # unvalidated under the gadget regime). Forcing it would break the
            # faithful arm. strongest/v1.0 are unchanged (profile sets it on).
            from fow_chess import engine_profile
            profile_name = _V2_PROFILE_BY_ID.get(engine_id, "v1.0")
            # Local-dev play-test override: FOW_V2_PROFILE_OVERRIDE forces a profile
            # for EVERY engine id (so any Misty tier picked in the UI plays it). Unset
            # = unchanged production routing. Used to play an unregistered candidate
            # (e.g. v1.1-rc6) locally without touching the server registry / picker.
            _override = os.environ.get("FOW_V2_PROFILE_OVERRIDE")
            if _override:
                if _override not in engine_profile.PROFILES:
                    raise RuntimeError(f"unknown FOW_V2_PROFILE_OVERRIDE: {_override}")
                profile_name = _override
            profile = engine_profile.PROFILES[profile_name]
            self._strategy = profile.build_strategy(
                seed=self.seed,
                time_budget_seconds=5.0,
                p_max_size=16_000_000,
            )
            return self._strategy
        tier1 = TIER1_LIVE_ENGINES.get(engine_id)
        if tier1 is None:
            raise RuntimeError(f"unsupported Python live engine: {engine_id}")
        if bot_runtime is None:
            raise RuntimeError(
                f"engine {engine_id!r} needs fow_chess.tournament (legacy tier1 "
                "snapshots), which is not present in this build"
            )
        config = load_config(ROOT / "configs" / "tier1-v1.json")
        if canonical_hash(config) != TIER1_CONFIG_HASH:
            raise RuntimeError("tier1-v1 config hash mismatch")
        if tier1.get("engineVersion"):
            config = replace(config, engine_version=tier1["engineVersion"])
        # else: leave config.engine_version=None → runtime loads live src/fow_chess
        self._runtime = bot_runtime(config, stockfish_path=self.stockfish_path)
        factory = self._runtime.__enter__()
        self._strategy = factory(self.seed)
        return self._strategy

    def exit(self) -> None:
        if self._runtime is not None:
            try:
                self._runtime.__exit__(None, None, None)
            except Exception:
                pass


def _deadline_monotonic(started: float, request: dict[str, Any]) -> float | None:
    """Wall deadline for producing a move: the TRANSPORT bound, not the compute bound.

    Prefers `workerDeadlineMs`. The legacy `watchdogTimeoutMs` key carried the
    engine's per-move COMPUTE budget, not the transport watchdog (the TS caller
    passed `watchdogTimeoutMs: computeBudgetMs`), which made this deadline several
    times tighter than intended. Since the usable pick window is
    `deadline - DEADLINE_GUARD_MS - PICK_DEADLINE_GUARD_MS` and
    `_budgeted_pick_view` vetoes below MIN_STRATEGY_PICK_BUDGET_MS, a compute
    budget under ~4.45s vetoed EVERY move to the unsearched deadline-guard. At
    3+2 that is any clock below ~29.8s: prod game 8d08b93a finished at 32.6s with
    233ms of slack, and its one guard fire (ply 79) followed the game's largest
    belief (|P|=654,573), whose delta-feed ate that slack.

    Falling back to the legacy key keeps an old TS caller working unchanged.
    """
    deadline_ms = _parse_optional_int(request.get("workerDeadlineMs"))
    if deadline_ms is None:
        deadline_ms = _parse_optional_int(request.get("watchdogTimeoutMs"))
    if deadline_ms is None:
        return None
    budget_ms = max(1, deadline_ms - DEADLINE_GUARD_MS)
    return started + budget_ms / 1000.0


def _compute_budget_ms(request: dict[str, Any]) -> int | None:
    """Per-move compute allowance, used to bound the synthetic clock handed to
    strategies that self-budget from a clock (tier-1). Prefers the explicit
    `computeBudgetMs`; the legacy `watchdogTimeoutMs` held this value, so falling
    back to it preserves the old compute cap when talking to an old TS caller.
    v2 ignores this entirely — it budgets from the REAL game clock, restored in
    _handle_request right after _budgeted_pick_view."""
    explicit = _parse_optional_int(request.get("computeBudgetMs"))
    if explicit is not None:
        return explicit
    return _parse_optional_int(request.get("watchdogTimeoutMs"))


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _budgeted_pick_view(
    view: PerspectiveView,
    deadline: float | None,
    compute_budget_ms: int | None = None,
) -> tuple[PerspectiveView | None, int | None]:
    """Constrain legacy strategies with the worker's external compute deadline.

    Tier-1 v0.9.5 only accepts a clock-bearing PerspectiveView; it does not
    accept an explicit deadline parameter, so translate the allowance into clock
    fields the strategy uses only for its internal deadline calculation.

    The VETO (returning None -> deadline-guard) is a wall-clock question only: it
    fires when there is not enough time left to produce any move. The compute
    budget and the game clock BOUND the synthetic budget but must not veto —
    having 2s of compute left is a reason to search for 2s, not a reason to skip
    searching. Conflating the two is what sent well-provisioned moves to the
    deadline-guard (see _deadline_monotonic).
    """
    if deadline is None:
        return view, None
    remaining_ms = int((deadline - time.monotonic()) * 1000) - PICK_DEADLINE_GUARD_MS
    if remaining_ms < MIN_STRATEGY_PICK_BUDGET_MS:
        return None, max(0, remaining_ms)

    budget_ms = remaining_ms
    if compute_budget_ms is not None:
        budget_ms = min(budget_ms, max(0, compute_budget_ms))
    if view.clock_remaining_ms is not None:
        budget_ms = min(budget_ms, max(0, view.clock_remaining_ms))
    target_ms = max(MIN_PICK_BUDGET_MS, min(10_000, budget_ms))
    # Tier-1 computes budget ~= ((clock - 400) / 40) + increment, then clamps
    # to the usable clock. Choose synthetic clock/increment fields that produce
    # a budget near target_ms while preserving all non-clock view fields.
    strategy_clock_ms = target_ms + 400
    strategy_increment_ms = max(0, target_ms - target_ms // 40)
    return (
        replace(
            view,
            clock_remaining_ms=strategy_clock_ms,
            increment_ms=strategy_increment_ms,
        ),
        target_ms,
    )


def _deadline_guard_move(board: chess.Board, view: PerspectiveView) -> chess.Move:
    return max(
        sorted(view.own_legal_moves, key=lambda move: move.uci()),
        key=lambda move: _deadline_guard_score(board, view, move),
    )


def _deadline_guard_score(
    board: chess.Board,
    view: PerspectiveView,
    move: chess.Move,
) -> tuple[int, int, int, int, int]:
    """Score a fallback move. Highest tuple wins.

    `board` carries only the side-to-move's *visible* pieces (see
    `board_from_request`), so attacker/defender queries here are
    "as far as we can see" — the best a fast fallback can do under fog.

    Priority order:
      1. king-safety — never leave/place the own king on a square a visible
         enemy attacks if a legal move avoids it. This is what stops the
         fallback from throwing the king (room d860f498: it had `Kxg7` for a
         free, king-saving queen but ranked it LAST and played `h5h4`).
      2. net capture material — a free capture scores full target value; a
         capture into a visible recapture costs the mover's full value. (The
         old `mover // 20` term made KING captures score ~-5000, so the king
         would never take even an undefended queen.)
      3. castle / promotion, then central destination, then a deterministic
         move-order tiebreak.
    """
    perspective = view.perspective
    mover = view.visible_piece_map.get(move.from_square)
    target = view.visible_piece_map.get(move.to_square)

    # Apply the move on the visibility-only board (approximate: ignores en
    # passant; promotion handled coarsely as a queen) to ask "is my king safe
    # after this, and is the landing square covered by a visible enemy?"
    after = board.copy(stack=False)
    after.remove_piece_at(move.from_square)
    placed = chess.Piece(move.promotion, perspective) if move.promotion else mover
    if placed is not None:
        after.set_piece_at(move.to_square, placed)

    king_sq = after.king(perspective)
    king_safe = king_sq is None or not after.attackers(not perspective, king_sq)

    capture_score = 0
    if target is not None and target.color != perspective:
        capture_score = MATERIAL_VALUE.get(target.piece_type, 0)
        # Likely lost to a visible recapture on the landing square?
        if mover is not None and after.attackers(not perspective, move.to_square):
            capture_score -= MATERIAL_VALUE.get(mover.piece_type, 0)

    castle_score = 80 if board.is_castling(move) else 0
    promotion_score = 70 if move.promotion is not None else 0
    center_score = 10 if move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5} else 0
    return (
        1 if king_safe else 0,
        capture_score,
        castle_score + promotion_score,
        center_score,
        -_move_sort_value(move),
    )


def _move_sort_value(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square + (move.promotion or 0)


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _configure_live_stockfish_budget() -> None:
    os.environ.setdefault(
        "PYTHON_ENGINE_STOCKFISH_TIME_CAP_SECONDS",
        os.environ.get(
            "PYTHON_LIVE_STOCKFISH_TIME_CAP_SECONDS",
            DEFAULT_LIVE_STOCKFISH_TIME_CAP_SECONDS,
        ),
    )
    os.environ.setdefault(
        "PYTHON_ENGINE_STOCKFISH_SLACK_SECONDS",
        os.environ.get(
            "PYTHON_LIVE_STOCKFISH_SLACK_SECONDS",
            DEFAULT_LIVE_STOCKFISH_SLACK_SECONDS,
        ),
    )


def _debug(phase: str, started: float, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "kind": "python_live_engine_debug",
                "phase": phase,
                "elapsedMs": round((time.monotonic() - started) * 1000),
                **fields,
            },
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
