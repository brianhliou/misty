"""Resolved engine-config inventory + a boot-time dump.

The engine's "which engine am I" config is genuinely split across two mechanisms:

  * **Profile knobs** — single-sourced in ``engine_profile.STRONGEST`` (i,
    kluss, gadget, king-aware, tanh scale, queen-promo). Passed as constructor
    kwargs / process-global leaf flags. These can't drift: every consumer builds
    from the one profile.
  * **Env toggles** — set per-environment (``FOW_BOTTOMK_EXPANSION``,
    ``FOW_V2_CLOCK_TIME``, ``FOW_V2_EARLY_STOP``, ``FOW_OPENING_BOOK``, ...).
    These are hand-set on the prod worker (Railway env) AND in each bakeoff
    setup-command — two copies of one list, which is exactly where prod-vs-bakeoff
    drift creeps in (the bug class behind the king-aware live blunder).

This module gathers BOTH into one resolved view, plus a short ``config_hash``
over the **toggles** (the drift-prone part). The live worker and the bakeoff each
dump it at startup; if their toggle-hashes differ, the bakeoff is not validating
the engine prod runs. It's the "consciously see what's running + catch a mismatch
loudly" mechanism — not a config framework.

NOTE: this is an inventory that RE-READS the env at dump time; it does not yet
own the reads (the engine still reads each flag at its own callsite). Keeping the
toggle list here in sync with those callsites is manual for now — the next step
is to make the profile own the frozen ones so this list shrinks to real toggles.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

# Profile knobs to surface (read off the V2Profile — single-sourced, can't drift).
_PROFILE_KNOBS = (
    "name",
    "i_sample_size",
    "kluss_k",
    "resolve_gadget",
    "resolve_cvar_q",
    "king_aware_leaf",
    "tanh_scale_cp",
    "king_band_floor",
    "queen_promo_tiebreak",
)


@dataclass(frozen=True)
class Toggle:
    """An env-set flag that defines WHICH engine (not deployment/debug)."""

    key: str
    env: str
    default: str
    note: str


# FROZEN serving toggles — now OWNED by STRONGEST (engine_profile.apply_process_flags
# seeds them via setdefault). Shown for completeness (effective value read from
# os.environ), but kept OUT of the drift hash: they're profile-single-sourced, so
# they shouldn't differ across envs. Listing them here keeps the dump complete
# even though the hand-maintained surface (below) no longer includes them.
FROZEN: tuple[Toggle, ...] = (
    Toggle("bottomk_expansion", "FOW_BOTTOMK_EXPANSION", "0", "bottom-K belief-expansion bound (profile-owned)"),
    Toggle("clock_time", "FOW_V2_CLOCK_TIME", "0", "clock-aware per-move budget (profile-owned)"),
    Toggle("early_stop", "FOW_V2_EARLY_STOP", "0", "convergence-based early stop (profile-owned)"),
    # Profile-owned since v1.3 (apply_process_flags seeds it); keeping it in the
    # drift hash made a legitimate v1.0-vs-v1.3+ profile difference fire the
    # toggle-drift alarm.
    Toggle("opening_book", "FOW_OPENING_BOOK", "0", "observation-keyed opening book (profile-owned)"),
)

# The remaining drift-prone, per-environment, HAND-SET engine toggles — the
# genuine drift surface (these go into the hash). (Deployment/debug env vars —
# FOW_STOCKFISH path, FOW_WORKER_SELFTEST, FOW_DEBUG_VERBOSE — are intentionally
# NOT here: they're legitimately per-box and don't define the engine.)
TOGGLES: tuple[Toggle, ...] = (
    # Default "1" mirrors leaf_eval_stockfish._lean_uci_default(): lean UCI is
    # ON when the env is unset. Declaring "0" here made the dump report
    # lean_uci=0 on boxes where it was actually on, and hashed the wrong value.
    Toggle("lean_uci", "FOW_LEAN_UCI", "1", "lean UCI leaf eval (byte-identical; ON unless opted out)"),
    Toggle("eq_merged", "FOW_EQ_MERGED", "0", "merged-eq throughput experiment"),
    Toggle("require_rust", "FOW_REQUIRE_RUST", "0", "hard-fail if rust ext missing/stale"),
)


def _read(specs: tuple[Toggle, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in specs:
        raw = os.environ.get(t.env)
        out[t.key] = {
            "value": raw if raw is not None else t.default,
            "source": "env" if raw is not None else "default",
        }
    return out


def toggle_values() -> dict[str, dict]:
    """``{key: {"value", "source"}}`` for the hand-set env toggles."""
    return _read(TOGGLES)


def frozen_values() -> dict[str, dict]:
    """Effective values of the profile-owned frozen toggles (for the dump)."""
    return _read(FROZEN)


def config_hash() -> str:
    """Short stable hash over the env TOGGLES only — the comparable drift signal.

    Toggles are the per-environment, hand-set flags (the drift-prone ones). The
    profile knobs are deliberately excluded: they're single-sourced via
    ``STRONGEST`` / ``--profile`` and the bakeoff varies them on purpose, so
    putting them in the hash would make a sweep falsely look like prod. The
    worker and a ``--profile strongest`` bakeoff print the SAME hash iff their
    toggles match — that's the mismatch alarm.
    """
    t = toggle_values()
    blob = ";".join(f"{k}={t[k]['value']}" for k in sorted(t))
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def format_dump(profile=None, include_profile: bool = True) -> str:
    """Resolved-config table. ``include_profile`` shows the STRONGEST profile
    knobs for context (the worker, which builds from STRONGEST); the bakeoff
    passes ``False`` since it sets those from its own args."""
    lines = [f"engine-config toggles-hash={config_hash()}"]
    if include_profile:
        if profile is None:
            from .engine_profile import STRONGEST

            profile = STRONGEST
        for knob in _PROFILE_KNOBS:
            lines.append(f"  profile.{knob:18} = {getattr(profile, knob)!s:>8}")
    for k, v in frozen_values().items():
        lines.append(f"  frozen.{k:19} = {v['value']!s:>8}  [{v['source']}]")
    for k, v in toggle_values().items():
        lines.append(f"  toggle.{k:19} = {v['value']!s:>8}  [{v['source']}]")
    return "\n".join(lines)


def dump(emit=print, profile=None, include_profile: bool = True) -> None:
    """Emit the resolved-config table line-by-line via ``emit`` (default print)."""
    for line in format_dump(profile, include_profile).splitlines():
        emit(line)
