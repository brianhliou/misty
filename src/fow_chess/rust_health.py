"""Rust-extension health checks: is ``fow_rust`` built, complete, and fresh?

The hot path silently falls back to pure Python when ``fow_rust`` is missing,
incomplete (namespace-package shadow with no symbols), or — the subtle one —
*stale* (Rust source edited but ``maturin develop`` not re-run). Silent
fallback is a measurement landmine: a bakeoff can run against ~500x-slower
Python, or against the *previous* Rust build, with no visible signal.

This module makes those states inspectable and, under ``FOW_REQUIRE_RUST=1``,
fatal. Call :func:`require` at the start of any run whose results depend on the
Rust path being live and current (CI gate, bakeoffs, the cloud worker).

It does NOT touch the per-module ``_HAS_RUST`` import guards — those keep their
graceful-fallback contract. This is an opt-in tripwire layered on top.
"""

from __future__ import annotations

import os
from pathlib import Path

# Symbols the engine actually calls on the Rust path. If any are absent the
# extension is unbuilt or stale-without-recompile; treat it as "not really
# there". Mirrors the per-module ``hasattr`` probes in observation.py /
# visibility.py / enumerator.py so this check can't disagree with them.
_REQUIRED_SYMBOLS = (
    "observation_from_transition_bb",
    "visible_squares_bb",
    "update_own_move_rust",
    "update_opp_move_rust",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUST_SRC = _REPO_ROOT / "fow_rust" / "src"


def _module():
    try:
        # Lazy/local import: this is a probe — the module may legitimately be
        # absent, and importing at top level would defeat the health check.
        import fow_rust
    except ImportError:
        return None
    return fow_rust


def available() -> bool:
    """True iff ``fow_rust`` imports AND exposes every symbol we call."""
    mod = _module()
    return mod is not None and all(hasattr(mod, s) for s in _REQUIRED_SYMBOLS)


_WARNED_UNAVAILABLE = False


def warn_once_if_unavailable() -> None:
    """One loud process-wide warning when the Rust extension is absent.

    A pip install of the pure-Python wheel runs the fallback belief path
    (~500x slower) with identical results — fine for trying the engine,
    wrong for measuring it. Called from EngineV2 construction so a user
    who never reads the docs still finds out which path they're on."""
    global _WARNED_UNAVAILABLE
    if _WARNED_UNAVAILABLE or available():
        return
    _WARNED_UNAVAILABLE = True
    import logging

    logging.getLogger("fow_chess").warning(
        "fow_rust extension not available: using the pure-Python belief path "
        "(identical results, ~500x slower). Build it with "
        "`maturin develop --release` in fow_rust/ (needs a Rust toolchain), "
        "or set FOW_REQUIRE_RUST=1 to make this fatal."
    )


def missing_symbols() -> list[str]:
    """Required symbols absent from the imported module (empty if all present
    or the module is unimportable)."""
    mod = _module()
    if mod is None:
        return []
    return [s for s in _REQUIRED_SYMBOLS if not hasattr(mod, s)]


def staleness() -> tuple[bool, float, float]:
    """Compare the newest ``fow_rust/src/*.rs`` mtime to the built module's.

    Returns ``(is_stale, newest_src_mtime, module_mtime)``. ``is_stale`` is
    True when a Rust source file is newer than the installed extension — i.e.
    ``maturin develop`` is overdue. Returns ``(False, 0, 0)`` when the module
    isn't importable or sources can't be located (nothing to compare).
    """
    mod = _module()
    if mod is None or not getattr(mod, "__file__", None):
        return (False, 0.0, 0.0)
    if not _RUST_SRC.is_dir():
        return (False, 0.0, 0.0)
    srcs = list(_RUST_SRC.rglob("*.rs"))
    if not srcs:
        return (False, 0.0, 0.0)
    newest_src = max(p.stat().st_mtime for p in srcs)
    module_mtime = Path(mod.__file__).stat().st_mtime
    return (newest_src > module_mtime, newest_src, module_mtime)


def report() -> str:
    """One-screen human summary of Rust-extension health."""
    mod = _module()
    lines = ["fow_rust health:"]
    if mod is None:
        lines.append("  import:   FAIL (module not importable)")
        return "\n".join(lines)
    lines.append(f"  import:   OK ({getattr(mod, '__file__', '?')})")
    miss = missing_symbols()
    lines.append(
        "  symbols:  OK (all present)" if not miss
        else f"  symbols:  MISSING {miss}"
    )
    is_stale, src_m, mod_m = staleness()
    if src_m == 0.0:
        lines.append("  freshness: n/a (sources not found)")
    elif is_stale:
        lines.append(
            f"  freshness: STALE — src newer than build by "
            f"{src_m - mod_m:.0f}s; run `maturin develop --release`"
        )
    else:
        lines.append("  freshness: OK (build newer than sources)")
    return "\n".join(lines)


def require(*, strict: bool | None = None, check_freshness: bool = True) -> None:
    """Raise unless the Rust extension is built, complete, and (optionally) fresh.

    ``strict`` defaults to the truthiness of ``FOW_REQUIRE_RUST`` in the
    environment, so callers can unconditionally invoke ``require()`` and let
    the env decide whether it's fatal. When ``strict`` is False this is a no-op
    (graceful Python fallback stays in force).
    """
    if strict is None:
        strict = bool(os.environ.get("FOW_REQUIRE_RUST"))
    if not strict:
        return
    if _module() is None:
        raise RuntimeError(
            "FOW_REQUIRE_RUST set but fow_rust is not importable. "
            "Build it with `maturin develop --release` (cwd fow_rust/)."
        )
    miss = missing_symbols()
    if miss:
        raise RuntimeError(
            f"FOW_REQUIRE_RUST set but fow_rust is missing symbols {miss} — "
            "likely an unbuilt namespace shadow or a partial build. "
            "Rebuild with `maturin develop --release`."
        )
    if check_freshness:
        is_stale, src_m, mod_m = staleness()
        if is_stale:
            raise RuntimeError(
                f"FOW_REQUIRE_RUST set but fow_rust is STALE: a source file is "
                f"{src_m - mod_m:.0f}s newer than the built extension. "
                "Rerun `maturin develop --release` so you measure current code."
            )


if __name__ == "__main__":
    import sys

    print(report())
    # Exit nonzero on any problem so `just check` / CI can gate on it,
    # independent of FOW_REQUIRE_RUST (the script is the explicit gate).
    try:
        require(strict=True)
    except RuntimeError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
