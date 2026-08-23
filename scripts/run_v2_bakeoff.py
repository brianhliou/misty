"""v2 (EngineV2) vs v0.9.5 (Tier-1) production bakeoff runner.

Sharded via ``--start-index`` (modeled on scripts/run_bakeoff.py). Each
game runs in its own subprocess for crash + memory isolation. Per-game
results stream to a shard jsonl log; per-game events stream straight to
``<out>/games/<game_id>.jsonl`` (viewer-compatible). Resume is automatic:
games whose game_ids already appear in the shard log are skipped.

Operational guards:
  - Per-game subprocess: crash/OOM in one game doesn't kill the shard.
  - Per-game timeout: parent kills + logs if the game exceeds
    ``--per-game-timeout`` (default 1800s = 30 min).
  - |P| soft-cap (default 1,000,000 via ``--v2-p-max``; pass 0 for truly
    uncapped at your own OOM risk).
  - Peak RSS captured per game via the subprocess's ``resource.getrusage``.
  - Idempotent resume: re-running with the same ``--out-dir`` skips
    already-logged game_ids.

Output layout::

    <out-dir>/
    ├── spec.json              # bakeoff settings (one per shard, identical)
    ├── shard-NN.jsonl         # one line per game (result + RSS + timings)
    ├── games/
    │   └── game-NNNN-{W|L|D}-tier1-{white|black}.jsonl   # viewer-compatible
    └── manifest.json          # written at shard end; merge across shards post-hoc

Usage (single shard):
    PYTHONPATH=src python scripts/run_v2_bakeoff.py \\
        --out-dir lab/runs/v2-vs-v095-baseline-2026-05-24 \\
        --games 50 --start-index 0 \\
        --v2-iters 500 --v2-i 32 \\
        --shard-id 0

Multi-shard: launch N processes with disjoint --start-index ranges (see
scripts/launch_v2_bakeoff.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Worker mode imports the heavy engine modules. Orchestrator mode doesn't
# need them, so defer.

_TIER1_CONFIG = ROOT / "configs" / "tier1-v1.json"


def _color_label(c: chess.Color) -> str:
    return "white" if c == chess.WHITE else "black"


def _outcome_letter(result_winner: str | None, subject_color: chess.Color) -> str:
    """v2-centric W/L/D. W = v2 won, L = v0.9.5 won, D = draw."""
    if result_winner is None:
        return "D"
    subject_label = "white" if subject_color == chess.WHITE else "black"
    return "W" if result_winner == subject_label else "L"


# ---------------------------------------------------------------------------
# Worker mode: play ONE game in this process, print JSON to stdout, exit.
# ---------------------------------------------------------------------------


def _events_to_jsonl(events: list, room_id: str, variant: str) -> str:
    """Render events as JSONL matching the bakeoff viewer's schema.
    Mirrors scripts/bakeoff_publish_to_viewer.py:_events_to_jsonl so the
    output drops directly into apps/web/public/-style viewer dirs."""
    lines: list[str] = []
    has_room_created = False
    for event in events:
        if event.get("type") == "room-created":
            event = {**event, "variant": variant}
            has_room_created = True
        lines.append(json.dumps(event, separators=(",", ":")))
    if not has_room_created:
        head = json.dumps(
            {
                "type": "room-created",
                "at": 0,
                "roomId": room_id,
                "variant": variant,
                "offer": [],
            },
            separators=(",", ":"),
        )
        lines.insert(0, head)
    return "\n".join(lines) + "\n"


def _peak_rss_mb() -> float:
    """Peak RSS for this process, in MB. ru_maxrss is bytes on macOS,
    kilobytes on Linux — normalize."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def _play_one_in_process(args: argparse.Namespace) -> int:
    """Worker entry point. Plays exactly one game, writes its event jsonl
    to <out-dir>/games/, prints a single JSON line to stdout, exits.

    Communication contract with the parent:
      - stdout: exactly one JSON object on the last line (the result row)
      - stderr: free-form logs; not parsed
      - exit code: 0 success, 1 internal error
    """
    from fow_chess.engine_v2 import EngineV2Strategy
    from fow_chess.selfplay import TimeControlSpec, play_game
    from fow_chess.tournament.config import load_config
    from fow_chess.tournament.runtime import bot_runtime

    # CLOCK-TRUE mode: parse 'BASE+INC' (seconds) into a real decrementing clock.
    # When set, the engine budgets per-move from it (needs FOW_V2_CLOCK_TIME=1);
    # the harness flags a side whose clock hits 0. Empty = fixed-budget (legacy).
    time_control = None
    if getattr(args, "time_control", ""):
        base_s, _, inc_s = args.time_control.partition("+")
        time_control = TimeControlSpec(
            initial_seconds=float(base_s), increment_seconds=float(inc_s or 0)
        )

    out_dir = Path(args.out_dir)
    games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    game_idx = args.game_idx
    v2_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
    seed = args.base_seed + game_idx

    v2 = EngineV2Strategy(
        seed=seed + 7,
        iterations=args.v2_iters,
        i_sample_size=args.v2_i,
        time_budget_seconds=args.v2_time_budget if args.v2_time_budget > 0 else None,
        p_max_size=args.v2_p_max if args.v2_p_max > 0 else None,
        capture_telemetry=True,
        kluss_k=args.v2_kluss_k if args.v2_kluss_k > 0 else None,
        kluss_soft=args.v2_kluss_soft,
        max_actions=args.v2_max_actions,
        use_rust_eq=args.v2_use_rust_eq,
        use_rust_state=args.v2_use_rust_state,
        use_rust_tree=args.v2_use_rust_tree,
        resolve_gadget=args.v2_resolve_gadget,
        resolve_cvar_q=args.v2_cvar_q,
        use_lean=args.v2_lean_uci,
        opening_book=args.v2_opening_book,
        gadget_faithful=args.v2_gadget_faithful,
        gadget_alpha=args.v2_gadget_alpha,
        gadget_iterative=args.v2_gadget_iterative,
        structural_carry=args.v2_structural_carry,
        carryover_subtree=args.v2_carryover_subtree,
        resolve_blueprint=args.v2_resolve_blueprint,
        expansion_budget=args.v2_expansion_budget if args.v2_expansion_budget > 0 else None,
    )
    # Opponent: either v0.9.5 Tier-1 (historical baseline) or a second v2
    # with a different kluss setting (for A/B self-play).
    runtime_cm = None
    opponent_v2 = None
    if args.opponent_mode == "tier1":
        config = load_config(_TIER1_CONFIG)
        runtime_cm = bot_runtime(config, stockfish_path=args.stockfish)
        factory = runtime_cm.__enter__()
        v095 = factory(seed)
    else:
        # opponent_mode == "v2": second EngineV2Strategy; differ via kluss_k.
        # Different seed shift so the two engines don't coordinate sampling.
        opponent_v2 = EngineV2Strategy(
            seed=seed + 13,
            iterations=args.opponent_iters if args.opponent_iters > 0 else args.v2_iters,
            i_sample_size=args.opponent_i if args.opponent_i > 0 else args.v2_i,
            # --opponent-time-budget: per-arm override (0 = inherit --v2-time-budget).
            # A sparring opponent (e.g. i=32) doesn't need the test arm's budget;
            # 30s/move BOTH sides made probe games hit --per-game-timeout unfinished.
            time_budget_seconds=(
                args.opponent_time_budget if args.opponent_time_budget > 0
                else (args.v2_time_budget if args.v2_time_budget > 0 else None)
            ),
            p_max_size=args.v2_p_max if args.v2_p_max > 0 else None,
            capture_telemetry=False,  # only telemetry from the "v2" side
            kluss_k=args.opponent_kluss_k if args.opponent_kluss_k > 0 else None,
            kluss_soft=args.opponent_kluss_soft,
            max_actions=args.opponent_max_actions,
            use_rust_eq=args.opponent_use_rust_eq,
            use_rust_state=args.opponent_use_rust_state,
            use_rust_tree=args.opponent_use_rust_tree,
            resolve_gadget=args.opponent_resolve_gadget,
            use_lean=args.opponent_lean_uci,
            opening_book=args.opponent_opening_book,
            gadget_faithful=args.opponent_gadget_faithful,
            gadget_alpha=args.opponent_gadget_alpha,
            gadget_iterative=args.opponent_gadget_iterative,
            structural_carry=args.opponent_structural_carry,
            carryover_subtree=args.opponent_carryover_subtree,
            resolve_blueprint=args.opponent_resolve_blueprint,
            expansion_budget=(args.opponent_expansion_budget
                              if args.opponent_expansion_budget > 0 else None),
        )
        v095 = opponent_v2  # name kept so the rest of the function flows

    # Label for output filenames: identifies the opponent this game was played
    # against. Was hardcoded "tier1" even for v2-vs-v2 self-play A/B runs, which
    # mislabeled every game in (e.g.) an |I|-sweep run.
    if args.opponent_mode == "tier1":
        opp_label = "tier1"
    else:
        _opp_i = args.opponent_i if args.opponent_i > 0 else args.v2_i
        opp_label = f"v2i{_opp_i}" + (
            f"k{args.opponent_kluss_k}" if args.opponent_kluss_k > 0 else ""
        )

    try:
        white_s = v2 if v2_color == chess.WHITE else v095
        black_s = v095 if v2_color == chess.WHITE else v2
        room_id = f"v2bakeoff-g{game_idx:04d}"

        # Pre-allocate the events list so on PEnumerator soundness errors (or
        # any mid-game crash) we still hold every move played up to the crash.
        # Written below to <game>-CRASH-*.jsonl alongside the would-be game
        # file, with the perply telemetry to its perply sibling.
        # Live per-ply telemetry sink: stream + flush each row during play so an
        # OOM / per-game-timeout SIGKILL (which skips BOTH the end-of-game write
        # below AND the except handler) still leaves the |P| trajectory up to the
        # kill on disk. The games that explode |P| are exactly the ones that get
        # killed before they can report — this is what makes them legible. Survives
        # play_game's internal strategy reset.
        perply_filename = f"game-{game_idx:04d}-perply.jsonl"
        perply_path = games_dir / perply_filename
        v2.set_telemetry_sink(str(perply_path))
        events_sink: list = []
        t0 = time.monotonic()
        try:
            result = play_game(
                white_s, black_s,
                max_plies=args.max_plies,
                room_id=room_id,
                seed=seed,
                events_sink=events_sink,
                time_control=time_control,
            )
        except Exception as crash_exc:
            wall = time.monotonic() - t0
            crash_filename = f"game-{game_idx:04d}-CRASH.jsonl"
            crash_path = games_dir / crash_filename
            crash_path.write_text(_events_to_jsonl(events_sink, room_id, "dark-chess"))
            # Per-ply telemetry was already streamed live to perply_path by the
            # sink (set before play_game) — no end-of-game write needed.
            print(
                f"  g{game_idx:04d} CRASH-DUMP {len(events_sink)} events -> "
                f"{crash_path.name}: {type(crash_exc).__name__}: {crash_exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        wall = time.monotonic() - t0

        outcome = _outcome_letter(result.winner, v2_color)
        game_filename = f"game-{game_idx:04d}-{outcome}-{opp_label}-{_color_label(v2_color)}.jsonl"
        game_path = games_dir / game_filename
        game_path.write_text(_events_to_jsonl(result.events, room_id, "dark-chess"))

        # Per-ply telemetry (perply_filename/perply_path) was streamed live to disk
        # during play by the telemetry sink (set before play_game) and flushed per
        # row, so it's already complete here AND survives an OOM/timeout SIGKILL.

        # |P| explosion early-warning: surface the peak |P| seen this
        # game. Easier than greping the perply jsonl after the fact.
        p_peak = max((row.get("p_post", 0) for row in v2.telemetry), default=0)
        p_pick_max = max(
            (row.get("p_pre", 0) for row in v2.telemetry if row.get("kind") == "pick_move"),
            default=0,
        )

        record = {
            "game_idx": game_idx,
            "game_id": room_id,
            "v2_color": _color_label(v2_color),
            "outcome": outcome,
            "winner": result.winner,
            "end_reason": result.end_reason,
            "plies": result.plies,
            "truncated": result.truncated,
            "wall_seconds": round(wall, 2),
            "peak_rss_mb": round(_peak_rss_mb(), 1),
            "p_peak": p_peak,
            "p_peak_at_pick": p_pick_max,
            "seed_v2": seed + 7,
            "seed_v095": seed,
            "game_path": f"games/{game_filename}",
            "perply_path": f"games/{perply_filename}",
        }
        # Single-line JSON on the LAST stdout line is the parent's contract.
        sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()
        return 0
    finally:
        try:
            v2.close()
            if opponent_v2 is not None:
                opponent_v2.close()
        finally:
            if runtime_cm is not None:
                runtime_cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Orchestrator mode: spawn one subprocess per game in this shard's range.
# ---------------------------------------------------------------------------


def _load_completed_game_ids(out_dir: Path) -> set[str]:
    """Resume support: scan ALL shard-*.jsonl files in ``out_dir`` and
    return the set of game_ids whose lines parse cleanly. Partial /
    errored entries are also counted as "done" — we won't auto-retry
    them; the operator decides.

    Global (across all shards), not per-shard: a ladder rung can
    re-assign game indices to different shards (rung 1 → 4 shards × 1
    game each; rung 2 → 4 shards × 2 games each shifts shard
    membership). Per-shard resume would re-run games already completed
    by a different shard. Global resume skips correctly across rungs."""
    done: set[str] = set()
    for shard_log in sorted(out_dir.glob("shard-*.jsonl")):
        with shard_log.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gid = row.get("game_id")
                if gid:
                    done.add(gid)
    return done


def _apply_profile(args: argparse.Namespace) -> None:
    """Resolve --profile into the v2-ARM knobs in place.

    'strongest' pins the v2 arm to engine_profile.STRONGEST (the single source
    of truth) so a bakeoff scores the exact engine prod serves. Idempotent and
    safe to call in both the orchestrator (so labels/spec reflect the resolved
    config) and — harmlessly, as a no-op since the forwarded child cmd carries
    the resolved per-knob values, not --profile — the child. The opponent arm
    is intentionally left as specified, so you can A/B prod-vs-variant.
    """
    _pname = getattr(args, "profile", "none")
    if _pname and _pname != "none":
        from fow_chess.engine_profile import PROFILES, DEFAULT_ITER_CAP

        if _pname not in PROFILES:
            raise SystemExit(
                f"--profile {_pname!r} unknown; available: {sorted(PROFILES)}")
        prof = PROFILES[_pname]
        args.v2_i = prof.i_sample_size
        args.v2_kluss_k = prof.kluss_k or 0  # 0 = off in the bakeoff's > 0 gate
        args.v2_resolve_gadget = prof.resolve_gadget
        args.v2_cvar_q = prof.resolve_cvar_q
        # The profile's iter cap is intentionally high so the per-move TIME budget
        # binds; never lower an explicit larger --v2-iters the caller passed.
        args.v2_iters = max(args.v2_iters, DEFAULT_ITER_CAP)
        if args.king_aware is None:
            args.king_aware = prof.king_aware_leaf
        # Faithful-stack constructor knobs (None = leave to env; the profile is
        # authoritative when it sets them).
        if prof.gadget_iterative:
            args.v2_gadget_iterative = True
        if prof.gadget_alpha:
            args.v2_gadget_alpha = True
        if prof.resolve_blueprint is not None:
            args.v2_resolve_blueprint = prof.resolve_blueprint
        if prof.carryover_subtree:
            args.v2_carryover_subtree = True
        if prof.structural_carry:
            args.v2_structural_carry = True
        # Process-global env toggles: the profile seeds ALL of them (gadget
        # stack + bottom-K/clock/early-stop) via setdefault — the whole point:
        # a ticket that says `--profile candidate` cannot miss a flag. The
        # children inherit the seeded env. Explicit env in the setup-command
        # still wins (setdefault). Also fixes the unseeded-FOW_BOTTOMK gap that
        # let probe children run uncapped belief expansion (13.9GB RSS spikes).
        prof.apply_process_flags()
        # Per-arm: the profile's expansion budget binds the v2 ARM only (the
        # env-seeded copy above would leak to the opponent arm — at i=32 the
        # same eb is ~6x deeper per world, silently favoring the sparring
        # partner). Explicit --v2-expansion-budget still wins.
        if getattr(args, "v2_expansion_budget", 0) == 0 and prof.expansion_budget:
            args.v2_expansion_budget = prof.expansion_budget
    # Override: force the gadget OFF even under --profile (which otherwise pins it
    # ON). Lets the gadget-isolation arm be "strongest profile MINUS the gadget".
    # Runs AFTER the profile block so it wins.
    if getattr(args, "v2_no_resolve_gadget", False):
        args.v2_resolve_gadget = False


def _spawn_game(
    *, game_idx: int, base_args: argparse.Namespace, timeout_s: float
) -> dict:
    """Spawn one subprocess to play one game. Returns the parsed JSON
    record on success, or an error dict on crash/timeout."""
    game_id = f"v2bakeoff-g{game_idx:04d}"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--play-one",
        "--game-idx", str(game_idx),
        "--out-dir", str(base_args.out_dir),
        "--max-plies", str(base_args.max_plies),
        "--v2-iters", str(base_args.v2_iters),
        "--v2-i", str(base_args.v2_i),
        "--v2-time-budget", str(base_args.v2_time_budget),
        "--time-control", base_args.time_control,
        "--v2-p-max", str(base_args.v2_p_max),
        "--v2-kluss-k", str(base_args.v2_kluss_k),
        "--v2-max-actions", str(base_args.v2_max_actions),
        "--base-seed", str(base_args.base_seed),
        "--stockfish", base_args.stockfish,
        "--opponent-mode", base_args.opponent_mode,
        "--opponent-kluss-k", str(base_args.opponent_kluss_k),
        "--opponent-max-actions", str(base_args.opponent_max_actions),
        "--opponent-i", str(base_args.opponent_i),
        "--opponent-iters", str(base_args.opponent_iters),
        "--opponent-time-budget", str(base_args.opponent_time_budget),
        "--v2-expansion-budget", str(base_args.v2_expansion_budget),
        "--opponent-expansion-budget", str(base_args.opponent_expansion_budget),
    ]
    # store_true flags: only append when set (per-game subprocess relaunch).
    if base_args.v2_use_rust_eq:
        cmd.append("--v2-use-rust-eq")
    if base_args.opponent_use_rust_eq:
        cmd.append("--opponent-use-rust-eq")
    if base_args.v2_use_rust_state:
        cmd.append("--v2-use-rust-state")
    if base_args.opponent_use_rust_state:
        cmd.append("--opponent-use-rust-state")
    if not base_args.v2_use_rust_tree:
        cmd.append("--v2-no-rust-tree")
    if not base_args.opponent_use_rust_tree:
        cmd.append("--opponent-no-rust-tree")
    # The v2-arm gadget/cvar were historically NOT forwarded (the child read the
    # process env instead). Forward them so --profile / explicit --v2-resolve-*
    # actually reach the child. king-aware is forwarded as a tri-state.
    if base_args.v2_resolve_gadget:
        cmd.append("--v2-resolve-gadget")
    if base_args.v2_cvar_q is not None:
        cmd.extend(["--v2-cvar-q", str(base_args.v2_cvar_q)])
    if base_args.v2_opening_book:
        cmd.append("--v2-opening-book")
    if base_args.opponent_opening_book:
        cmd.append("--opponent-opening-book")
    if base_args.king_aware is True:
        cmd.append("--king-aware")
    elif base_args.king_aware is False:
        cmd.append("--no-king-aware")
    # Faithful-stack per-arm flags (2026-06-10): these were NEVER forwarded to
    # the per-game child, so every cloud probe's child fell back to env-reads
    # (= OFF) and silently ran the read-only stub gadget instead of the
    # iterative/alpha/carryover stack it claimed to test. Forward ALL of them,
    # both arms, tri-state-aware (None = let the child env-read).
    for _flag, _val in (
        ("--v2-gadget-faithful", base_args.v2_gadget_faithful),
        ("--v2-gadget-alpha", base_args.v2_gadget_alpha),
        ("--v2-gadget-iterative", base_args.v2_gadget_iterative),
        ("--v2-carryover-subtree", base_args.v2_carryover_subtree),
        ("--v2-structural-carry", base_args.v2_structural_carry),
        ("--v2-lean-uci", base_args.v2_lean_uci),
        ("--opponent-resolve-gadget", base_args.opponent_resolve_gadget),
        ("--opponent-gadget-faithful", base_args.opponent_gadget_faithful),
        ("--opponent-gadget-alpha", base_args.opponent_gadget_alpha),
        ("--opponent-gadget-iterative", base_args.opponent_gadget_iterative),
        ("--opponent-carryover-subtree", base_args.opponent_carryover_subtree),
        ("--opponent-structural-carry", base_args.opponent_structural_carry),
        ("--opponent-lean-uci", base_args.opponent_lean_uci),
        ("--v2-kluss-soft", base_args.v2_kluss_soft),
        ("--opponent-kluss-soft", base_args.opponent_kluss_soft),
    ):
        if _val:
            cmd.append(_flag)
    if base_args.v2_resolve_blueprint is not None:
        cmd.extend(["--v2-resolve-blueprint", base_args.v2_resolve_blueprint])
    if base_args.opponent_resolve_blueprint is not None:
        cmd.extend(["--opponent-resolve-blueprint", base_args.opponent_resolve_blueprint])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}"

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "game_idx": game_idx,
            "game_id": game_id,
            "error": f"timeout after {timeout_s}s",
            "wall_seconds": round(time.monotonic() - t0, 2),
            "stdout_tail": (e.stdout or b"")[-2000:].decode("utf-8", "replace") if e.stdout else "",
            "stderr_tail": (e.stderr or b"")[-2000:].decode("utf-8", "replace") if e.stderr else "",
        }

    if proc.returncode != 0:
        return {
            "game_idx": game_idx,
            "game_id": game_id,
            "error": f"exit {proc.returncode}",
            "wall_seconds": round(time.monotonic() - t0, 2),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    # Parse the LAST non-empty stdout line as the result record.
    last_line = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            last_line = line
    try:
        record = json.loads(last_line)
    except json.JSONDecodeError:
        return {
            "game_idx": game_idx,
            "game_id": game_id,
            "error": "bad worker stdout (could not parse final JSON line)",
            "wall_seconds": round(time.monotonic() - t0, 2),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    return record


def _write_spec(out_dir: Path, args: argparse.Namespace) -> None:
    """Per-shard spec.json — same content across shards of one bakeoff.
    Last writer wins, which is fine since it's deterministic from args."""
    spec_path = out_dir / "spec.json"
    spec = {
        "kind": f"v2-vs-{args.opponent_mode}",
        "max_plies": args.max_plies,
        "v2_iters": args.v2_iters,
        "v2_i": args.v2_i,
        "v2_time_budget": args.v2_time_budget,
        "v2_p_max": args.v2_p_max,
        "v2_kluss_k": args.v2_kluss_k,
        "v2_max_actions": args.v2_max_actions,
        "opponent_mode": args.opponent_mode,
        "opponent_kluss_k": args.opponent_kluss_k,
        "opponent_max_actions": args.opponent_max_actions,
        "opponent_i": args.opponent_i,
        "opponent_iters": args.opponent_iters,
        "base_seed": args.base_seed,
        "stockfish": args.stockfish,
        "per_game_timeout": args.per_game_timeout,
    }
    spec_path.write_text(json.dumps(spec, indent=2))


def _write_manifest(out_dir: Path, args: argparse.Namespace) -> None:
    """Rebuild manifest.json from all shard logs. Idempotent — call at
    end of each shard; the last finisher wins and includes everyone's
    games. Compatible with apps/web bakeoff viewer.

    Cross-shard dedup by game_idx: if the same game ran in multiple
    shards (e.g., from a pre-fix ladder rung re-assignment), the LAST
    entry across the sorted-shard-log scan wins. Higher-shard-id
    overrides lower for a given game_idx — a deterministic tie-break."""
    by_idx: dict[int, dict] = {}
    for shard_log in sorted(out_dir.glob("shard-*.jsonl")):
        with shard_log.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in row:
                    continue
                by_idx[row["game_idx"]] = row

    games_for_manifest: list[dict] = []
    record = {"wins": 0, "losses": 0, "draws": 0}
    for idx in sorted(by_idx):
        row = by_idx[idx]
        outcome = row.get("outcome", "D")
        if outcome == "W":
            record["wins"] += 1
        elif outcome == "L":
            record["losses"] += 1
        else:
            record["draws"] += 1
        games_for_manifest.append({
            "index": row["game_idx"],
            "tier1_color": row["v2_color"],
            "outcome": outcome,
            "plies": row.get("plies", 0),
            "end_reason": row.get("end_reason", "unknown"),
            "truncated": row.get("truncated", False),
            "tier1_seed": row.get("seed_v2"),
            "random_seed": row.get("seed_v095"),
            "path": row["game_path"],
        })

    manifest = {
        "tier1_version": "engine-v2 (rust-port 551cbaf)",
        "tier1_commit": "current-src-fow-chess",
        "opponent": "v0.9.5-equivalent",
        "evaluator": "stockfish",
        "depth": -1,
        "max_particles": args.v2_p_max,
        "target_n": args.v2_i,
        "risk_aversion": 0.0,
        "verbose_belief": False,
        "threat_lambda": 0.0,
        "max_plies": args.max_plies,
        "base_seed": args.base_seed,
        "games_total": len(games_for_manifest),
        "games_saved": len(games_for_manifest),
        "save_only": "all",
        "tier1_record": record,
        "games": games_for_manifest,
        "v2_iters": args.v2_iters,
        "v2_i_sample": args.v2_i,
        "v2_time_budget": args.v2_time_budget,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _run_orchestrator(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_spec(out_dir, args)

    shard_log = out_dir / f"shard-{args.shard_id:02d}.jsonl"
    completed = _load_completed_game_ids(out_dir)

    start = args.start_index
    target = start + args.games
    print(
        f"v2 bakeoff shard {args.shard_id:02d}: games [{start}, {target}) "
        f"v2-iters={args.v2_iters} |I|={args.v2_i} p_max={args.v2_p_max} "
        f"per-game-timeout={args.per_game_timeout}s out={out_dir}",
        flush=True,
    )
    # Resolved env-toggle dump (same source as the live worker): the toggles-hash
    # here must match the worker's when validating prod (--profile strongest +
    # matching env). A mismatch = the bakeoff isn't testing what prod runs.
    from fow_chess import engine_config

    engine_config.dump(lambda s: print(s, flush=True), include_profile=False)
    if completed:
        print(f"  resume: {len(completed)} game(s) already in shard log; skipping", flush=True)

    skipped = 0
    completed_now = 0
    errors_now = 0
    t_start = time.monotonic()
    with shard_log.open("a") as log_fh:
        for game_idx in range(start, target):
            game_id = f"v2bakeoff-g{game_idx:04d}"
            if game_id in completed:
                skipped += 1
                continue
            t_game = time.monotonic()
            row = _spawn_game(
                game_idx=game_idx, base_args=args, timeout_s=args.per_game_timeout
            )
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()
            if "error" in row:
                errors_now += 1
                print(
                    f"  g{game_idx:04d} ERROR {row['error']} "
                    f"({time.monotonic() - t_game:.1f}s)",
                    flush=True,
                )
            else:
                completed_now += 1
                print(
                    f"  g{game_idx:04d} {row['outcome']} "
                    f"{row['end_reason']:18s} plies={row['plies']:3d} "
                    f"wall={row['wall_seconds']:6.1f}s "
                    f"rss={row['peak_rss_mb']:6.0f}MB "
                    f"|P|peak={row.get('p_peak', 0):>7d}",
                    flush=True,
                )

    total_wall = time.monotonic() - t_start
    print(
        f"\nshard {args.shard_id:02d} done: "
        f"{completed_now} completed, {errors_now} errors, "
        f"{skipped} pre-existing in {total_wall:.0f}s",
        flush=True,
    )
    _write_manifest(out_dir, args)
    print(f"manifest: {out_dir / 'manifest.json'}", flush=True)
    return 0 if errors_now == 0 else 3


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--play-one", action="store_true",
                    help="worker mode: play exactly one game, print JSON, exit")
    ap.add_argument("--game-idx", type=int,
                    help="(worker) which game index to play")
    ap.add_argument("--out-dir", required=True,
                    help="bakeoff output directory (shared by all shards of one bakeoff)")
    ap.add_argument("--games", type=int, default=1,
                    help="(orchestrator) number of games for this shard")
    ap.add_argument("--start-index", type=int, default=0,
                    help="(orchestrator) global game-index for this shard's first game")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="(orchestrator) shard identifier (used in shard log filename)")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--v2-iters", type=int, default=500)
    ap.add_argument("--v2-i", type=int, default=32,
                    help="|I| sample size from P per v2 move")
    ap.add_argument("--v2-time-budget", type=float, default=5.0,
                    help="per-move wall budget for v2 (seconds); 0 = unlimited. "
                    "Static fallback only when --time-control is unset; with a "
                    "clock, the engine budgets per-move from it (FOW_V2_CLOCK_TIME).")
    ap.add_argument("--time-control", type=str, default="",
                    help="CLOCK-TRUE mode: 'BASE+INC' in seconds, e.g. '180+2' for "
                    "3+2. Enables a real decrementing clock with flag-on-time (the "
                    "PvE-faithful regime), and the engine budgets per-move from it. "
                    "Empty = fixed-budget mode (no clock; legacy). Set "
                    "FOW_V2_CLOCK_TIME=1 in the env for the engine to honor it.")
    ap.add_argument("--v2-p-max", type=int, default=5_000_000,
                    help="cap on PEnumerator |P| (0 = truly uncapped, OOM risk). "
                    "Default 5M: cap-probe 2026-05-24 showed 1M had 3.1%% crash "
                    "rate from soundness violations; 5M had 0 crashes, 0 downsample "
                    "events, max 1.3 GB RSS across 3 cap-hitting seeds.")
    ap.add_argument("--v2-kluss-k", type=int, default=0,
                    help="KLUSS k-restriction for GT-CFR subgame (0 = off; 2 = Obscuro's choice)")
    ap.add_argument("--v2-max-actions", type=int, default=1,
                    help="A6.2 purification regime for the v2 side: 0 = auto "
                    "(select_regime by margin), 1 = Resolve (deterministic top), "
                    "2-3 = Maxmargin (mix top-m). Default 1 preserves the pre-A6.2 "
                    "behavior. Use 0 to enable A6.2 auto-mode for A/B testing.")
    ap.add_argument("--v2-use-rust-eq", action="store_true",
                    help="run the v2 side's equilibrium pass in Rust (EqEngine).")
    ap.add_argument("--opponent-use-rust-eq", action="store_true",
                    help="run the opponent v2 side's equilibrium pass in Rust.")
    # Per-arm Resolve gadget (overrides the process env FOW_RESOLVE_GADGET so the
    # gadget can be enabled on ONE arm only in a shared-process v2-vs-v2 run).
    # Absent = inherit the env (default OFF). Implies full-CFV on that arm.
    ap.add_argument("--v2-resolve-gadget", action="store_true", default=None,
                    help="enable the CVaR Resolve gadget on the v2 side only.")
    ap.add_argument("--v2-no-resolve-gadget", action="store_true", default=False,
                    help="force the gadget OFF even under --profile (gadget-isolation arm).")
    ap.add_argument("--v2-cvar-q", type=float, default=None,
                    help="CVaR tail fraction for the v2 gadget (default 0.1).")
    ap.add_argument("--opponent-resolve-gadget", action="store_true", default=None,
                    help="enable the CVaR Resolve gadget on the opponent v2 side only.")
    # Per-arm opening book (overrides FOW_OPENING_BOOK). Surgical, observation-
    # keyed patch over known opening traps — fires only on recorded fog-of-war
    # fingerprints, inert elsewhere. Default OFF (explicit, not env) so a v2-vs-v2
    # neutrality run enables it on exactly one arm. See fow_chess/opening_book.py.
    ap.add_argument("--v2-opening-book", action="store_true", default=False,
                    help="enable the opening book on the v2 side only.")
    ap.add_argument("--opponent-opening-book", action="store_true", default=False,
                    help="enable the opening book on the opponent v2 side only.")
    # Per-arm lean-UCI leaf eval (overrides FOW_LEAN_UCI). Byte-identical to the
    # python-chess path (tests/test_lean_uci_parity.py), so this isolates the
    # ~1.9x/eval throughput gain: more iters/move at the same budget. Absent =
    # inherit the env (default OFF).
    ap.add_argument("--v2-lean-uci", action="store_true", default=None,
                    help="use the lean UCI leaf eval on the v2 side only.")
    ap.add_argument("--opponent-lean-uci", action="store_true", default=None,
                    help="use the lean UCI leaf eval on the opponent v2 side only.")
    # Per-arm continual-resolve stack (overrides the process FOW_* env so ONE arm
    # can be faithful-gadget-on and the other gadget-off in a shared-process
    # v2-vs-v2 run). Absent = inherit the env (default OFF). The faithful gadget +
    # non-uniform alpha both also need --v2-resolve-gadget on that arm + a
    # carryover blueprint; structural carry needs carryover-subtree on.
    ap.add_argument("--v2-gadget-faithful", action="store_true", default=None,
                    help="faithful gadget aggregation (min/mean, no CVaR) on the v2 side.")
    ap.add_argument("--opponent-gadget-faithful", action="store_true", default=None)
    ap.add_argument("--v2-gadget-alpha", action="store_true", default=None,
                    help="non-uniform Obscuro alpha(J) on the v2 side (needs faithful + carryover).")
    ap.add_argument("--opponent-gadget-alpha", action="store_true", default=None)
    ap.add_argument("--v2-structural-carry", action="store_true", default=None,
                    help="structural Γ̂-carry root set on the v2 side (needs carryover-subtree).")
    ap.add_argument("--opponent-structural-carry", action="store_true", default=None)
    # PROPER (iterative) Resolve gadget — couples follow/exit INTO the eq loop
    # (proper-gadget Step 1) vs the read-only post-hoc cap. Needs --v2-resolve-gadget
    # on the same arm. Step-4 crux cell: PROPER vs OFF at i=200/un-starved, judged by
    # the per-ply |P| trajectory (p_pre/p_post telemetry) + H2H. Absent = env default OFF.
    ap.add_argument("--v2-gadget-iterative", action="store_true", default=None,
                    help="iterative (in-solve) Resolve gadget on the v2 side (needs --v2-resolve-gadget).")
    ap.add_argument("--opponent-gadget-iterative", action="store_true", default=None)
    ap.add_argument("--v2-carryover-subtree", action="store_true", default=None,
                    help="preserve+reuse the prior search tree on the v2 side.")
    ap.add_argument("--opponent-carryover-subtree", action="store_true", default=None)
    ap.add_argument("--v2-resolve-blueprint", type=str, default=None,
                    choices=["stub", "stockfish", "net", "carryover"],
                    help="gadget blueprint on the v2 side (overrides FOW_RESOLVE_BLUEPRINT).")
    ap.add_argument("--opponent-resolve-blueprint", type=str, default=None,
                    choices=["stub", "stockfish", "net", "carryover"])
    # WS2 rust-tree is the DEFAULT (since 2026-05-27): the v2 side's whole GT-CFR
    # loop runs on the authoritative Rust tree (strategy byte-identical to the
    # Python path, ~1.9x iters/move). Use these opt-OUTs to A/B the old Python
    # path (e.g. a clean belief-stack certification that isolates throughput).
    ap.add_argument("--v2-no-rust-tree", action="store_false", dest="v2_use_rust_tree",
                    help="run the v2 side on the legacy Python GTCFRTreeNode path "
                    "instead of the default Rust tree.")
    ap.add_argument("--opponent-no-rust-tree", action="store_false",
                    dest="opponent_use_rust_tree",
                    help="same as --v2-no-rust-tree for the opponent v2 side.")
    ap.add_argument("--v2-use-rust-state", action="store_true",
                    help="hold the v2 side's belief set P resident in Rust "
                    "(PEnumState, packed keys) instead of a Python set[str]. "
                    "Belief set is byte-identical; this is a throughput/memory "
                    "A/B (expect ~50%% result, lower RSS + per-move wall).")
    ap.add_argument("--opponent-use-rust-state", action="store_true",
                    help="same as --v2-use-rust-state for the opponent v2 side.")
    ap.add_argument("--opponent-max-actions", type=int, default=1,
                    help="Same as --v2-max-actions but for the opponent when "
                    "--opponent-mode=v2. Ignored otherwise.")
    ap.add_argument("--opponent-mode", choices=("tier1", "v2"), default="tier1",
                    help="What the v2 engine plays against. tier1 = v0.9.5 Tier-1 (the historical baseline); "
                    "v2 = another EngineV2Strategy (use --opponent-kluss-k to differ from --v2-kluss-k for "
                    "self-play A/B). Default tier1 preserves the original v2-vs-v0.9.5 bakeoff semantics.")
    ap.add_argument("--opponent-kluss-k", type=int, default=0,
                    help="KLUSS k for the OPPONENT v2 engine when --opponent-mode=v2. Ignored otherwise.")
    ap.add_argument("--opponent-i", type=int, default=0,
                    help="|I| sample size for the OPPONENT v2 engine when "
                    "--opponent-mode=v2 (0 = inherit --v2-i). Lets you A/B the "
                    "belief-sample size head-to-head (F2: does more belief "
                    "coverage beat more iters at a fixed time budget?).")
    ap.add_argument("--opponent-iters", type=int, default=0,
                    help="GT-CFR iteration cap for the OPPONENT v2 engine when "
                    "--opponent-mode=v2 (0 = inherit --v2-iters). Lets you A/B "
                    "the search budget head-to-head (M1: iteration-starvation).")
    ap.add_argument("--v2-kluss-soft", action="store_true", default=None,
                    help="Soft KLUSS for the v2 arm: when the keep-restricted "
                         "expansion walk deadlocks, retry unrestricted "
                         "(FOW_KLUSS_SOFT per-arm; None = env-read).")
    ap.add_argument("--opponent-kluss-soft", action="store_true", default=None,
                    help="Soft KLUSS for the opponent arm.")
    ap.add_argument("--v2-expansion-budget", type=int, default=0,
                    help="per-arm expansion budget for the v2 arm (0 = env "
                    "FOW_V2_EXPANSION_BUDGET / unlimited). Per-arm because the "
                    "env is process-wide and leaks to the opponent.")
    ap.add_argument("--opponent-expansion-budget", type=int, default=0,
                    help="per-arm expansion budget for the OPPONENT arm (0 = env).")
    ap.add_argument("--opponent-time-budget", type=float, default=0.0,
                    help="per-move wall budget for the OPPONENT v2 engine "
                    "(0 = inherit --v2-time-budget). A sparring opponent doesn't "
                    "need the test arm's budget; 30s/move BOTH sides made probe "
                    "games hit --per-game-timeout unfinished.")
    ap.add_argument("--base-seed", type=int, default=12345)
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--per-game-timeout", type=float, default=1800.0,
                    help="(orchestrator) seconds before killing a hung game")
    # --profile pins the v2 ARM to the single source of truth
    # (engine_profile.STRONGEST) so a bakeoff scores the EXACT engine prod
    # serves, instead of a hand-assembled subset that drifts from it. It
    # overrides the individual --v2-* knobs on the v2 arm; the per-knob flags
    # stay for A/B sweeps OFF the profile. The opponent arm is untouched.
    # choices derive from the PROFILES registry — a hand-copied tuple here
    # silently rejected new registry profiles (candidate-i32, rc=2 x3 ->
    # ticket abandoned, 2026-06-12). One registry, one list.
    from fow_chess.engine_profile import PROFILES as _PROFILES
    ap.add_argument("--profile", choices=("none", *_PROFILES), default="none",
                    help="'strongest' = engine_profile.STRONGEST (i=32, KLUSS k=2, "
                    "CVaR gadget q=0.1, king-aware leaf, high iter cap) on the v2 arm.")
    # King-aware is PROCESS-GLOBAL (a module flag in cfr.leaf_eval, no per-arm
    # override) — so in a v2-vs-v2 run it applies to BOTH arms; A/B it across two
    # separate invocations, not two arms. This was previously not wireable at all,
    # so every bakeoff scored a king-blind engine even though the live worker runs
    # king-aware. Absent = inherit FOW_KING_AWARE_LEAF (default OFF, prior behavior).
    ka = ap.add_mutually_exclusive_group()
    ka.add_argument("--king-aware", dest="king_aware", action="store_true", default=None,
                    help="enable the king-aware leaf shim (process-global, both arms).")
    ka.add_argument("--no-king-aware", dest="king_aware", action="store_false",
                    help="force the king-aware shim OFF (overrides --profile / env).")
    args = ap.parse_args()

    _apply_profile(args)
    # King-aware is a process-global flag; set it once here (covers the --play-one
    # child, which re-enters main() with the forwarded flag) before any engine is
    # built. None = leave the env/global default untouched.
    if args.king_aware is not None:
        from fow_chess.cfr.leaf_eval import set_king_aware_leaf
        set_king_aware_leaf(args.king_aware)

    # A bakeoff run against a stale or missing Rust extension silently measures
    # the wrong code (old .so) or the ~500x-slower Python fallback — worthless,
    # trust-eroding numbers. Under FOW_REQUIRE_RUST=1 this hard-fails before any
    # game runs; it is a no-op otherwise. Set the env in bakeoff invocations.
    from fow_chess import rust_health
    rust_health.require()

    if args.play_one:
        if args.game_idx is None:
            print("ERROR: --game-idx required with --play-one", file=sys.stderr)
            return 2
        return _play_one_in_process(args)

    if shutil.which(args.stockfish) is None:
        print(f"ERROR: stockfish binary not found ({args.stockfish!r})", file=sys.stderr)
        return 2
    return _run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
