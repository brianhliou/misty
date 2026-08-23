# Misty

A fog-of-war chess engine.

Fog of war (dark) chess is chess where you only see the squares your own
pieces could move to. You never see your opponent's moves, only their
consequences: a piece of yours disappears, a square you watch changes.
Playing it well means reasoning over every position consistent with what
you've observed so far, and mid-game that set can run into the millions.

Misty follows the architecture of Obscuro (Zhang & Sandholm, ICLR 2026),
the first superhuman fog-of-war chess AI:

| concern | approach |
|---|---|
| belief | exact enumeration of all positions consistent with the observation history |
| search scope | knowledge-limited subgames over sampled information sets |
| search | one-sided growing-tree CFR with PCFR+ |
| evaluation | Stockfish at depth 1 on subgame leaves |
| commit | purification plus a resolve gadget at move selection |

The hot paths (belief updates, visibility, observation derivation) run in
Rust, measured at roughly 500x over the Python reference on the
enumeration loop. The Python implementations stay maintained as the
oracle: the test suite replays real games and requires Rust and Python
to agree byte-for-byte at every ply.

Misty plays live at [mistboard.com](https://mistboard.com). Strength is
gauged against humans; engine-vs-engine results only gate development.

## Install

```
pip install misty-chess
brew install stockfish   # or: apt-get install stockfish
```

The package imports as `fow_chess`. Python 3.11+. Stockfish is invoked as
a subprocess for leaf evaluation and must be on PATH (or set
`FOW_STOCKFISH`).

The PyPI package is pure Python and runs everywhere; results are
identical to the accelerated build but the belief hot path is ~500x
slower, and the engine logs a warning saying so. For real strength at
real time controls, build the Rust extension (binary wheels are planned):

```
git clone <repo> && cd misty
pip install -e '.[dev]'
cd fow_rust && maturin develop --release
```

## Play a game

```python
from fow_chess import EngineV2Strategy, RandomStrategy
from fow_chess.selfplay import play_game

result = play_game(
    EngineV2Strategy(seed=0, time_budget_seconds=5.0),
    RandomStrategy(seed=1),
)
print(result.winner, result.end_reason, result.plies)
```

Strength configurations are named profiles in
`fow_chess.engine_profile.PROFILES`. The served engine is a named
profile, so a bakeoff and production run identical configuration by
construction.

## Analyze a game

The engine ships with a post-game analyzer built for imperfect
information. `TruthGrader` runs Stockfish at a fixed depth on the true
board of a finished game; `analyze_game_deep` replays the game through
the engine's exact belief and its own search, then classifies each
mistake three ways: the true position was missing from the belief, it
was absent from the sampled search set, or it was seen and the engine
still chose badly. That split (belief / sample / decision) says which
lever would have prevented the mistake.

```python
import chess
from fow_chess import TruthGrader, analyze_game_deep

moves = [chess.Move.from_uci(u) for u in ("f2f3", "e7e5", "g2g4")]
with TruthGrader(depth=12) as grader:
    rows = analyze_game_deep(moves, chess.WHITE, grader=grader)
for r in rows:
    if r.verdict:
        print(r.ply, r.uci, f"-{r.grade.cp_loss}cp", r.verdict)
```

## Serving

The engine speaks a JSON protocol over stdio, documented in
[docs/protocol/engine-protocol.md](docs/protocol/engine-protocol.md).
`scripts/live_move_worker.py` is the long-lived worker;
`scripts/live_move_runner.py` is the one-shot fallback. The server sends
redacted observations only. The engine never receives hidden opponent
state, and post-game full-information analysis is a separate request
type, never a flag on the live one.

## Development

```
pip install -e '.[dev]'
cd fow_rust && maturin develop --release && cd ..
pytest -n auto                                  # full suite, ~40s
cargo test --manifest-path fow_rust/Cargo.toml  # native parity pins
```

`FOW_REQUIRE_RUST=1` makes a missing or stale Rust extension fatal
instead of silently falling back to the slow Python path. Every
architectural change ships behind a flag defaulting to prior behavior,
then gets validated by bakeoff before the default flips.

## License

GPL-3.0-or-later.
