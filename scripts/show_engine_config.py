#!/usr/bin/env python3
"""Resolve and print the ACTUAL config for any engine-id or version — ground truth,
not inline comments.

WHY THIS EXISTS: the engine config is inheritance-based (v1.5 = deltas on v1.4 =
... = v1.0), so no single line shows a version's full config. And the "SHIPPED /
LIVE" comments in live_move_worker.py are STALE and CONTRADICTORY (v1.0, v1.1, AND
v1.2 are each labeled "shipped" in different places). DO NOT trust those comments.
Run this instead.

WHAT THIS CANNOT TELL YOU: which version PRODUCTION actually serves. That is chosen
platform-side — the mistboard server requests an engine-id (see
~/projects/mistboard, the engine-protocol / worker-spawn path, and Railway vars).
This repo only maps engine-id -> config; it does not decide which id prod requests.

Usage:
  scripts/show_engine_config.py                  # summary table of every engine-id
  scripts/show_engine_config.py python-v2-v1.5   # full resolved config (chess id)
  scripts/show_engine_config.py v1.5             # full resolved config (by version)
  scripts/show_engine_config.py python-fdx-v1.0  # xiangqi engine-id (env-resolved)
"""
from __future__ import annotations
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # so `import live_move_worker` works

from fow_chess import engine_profile as ep
import live_move_worker as lmw

CHESS_ID_TO_VERSION = lmw._V2_PROFILE_BY_ID
XIANGQI_ID_TO_PROFILE = getattr(lmw, "_XIANGQI_PROFILE_BY_ID", {})

# The fields most likely to be misremembered — shown first in the summary.
KEY_FIELDS = ["resolve_gadget", "gadget_alpha", "gadget_iterative", "resolve_cvar_q",
              "hv_prune_adaptive", "hv_prune_king_floor", "i_sample_size", "kluss_k",
              "king_aware_leaf", "early_stop"]


def chess_config(version: str) -> dict:
    prof = ep.PROFILES[version]
    return dataclasses.asdict(prof) if dataclasses.is_dataclass(prof) else dict(vars(prof))


def print_full(title: str, cfg: dict) -> None:
    print(f"\n=== {title} ===")
    for k in KEY_FIELDS:
        if k in cfg:
            print(f"  {k:22} = {cfg[k]}")
    print("  --- other fields ---")
    for k in sorted(cfg):
        if k not in KEY_FIELDS:
            print(f"  {k:22} = {cfg[k]}")


def resolve_one(arg: str) -> None:
    if arg in CHESS_ID_TO_VERSION:
        version = CHESS_ID_TO_VERSION[arg]
        print_full(f"chess engine-id {arg}  ->  engine_profile version '{version}'", chess_config(version))
    elif arg in ep.PROFILES:
        print_full(f"chess engine_profile version '{arg}'", chess_config(arg))
    elif arg in XIANGQI_ID_TO_PROFILE or arg.startswith(("python-fdx", "python-dxq", "python-dmx")):
        # xiangqi is env-flag-resolved at call time; show the resolved dict + which env flags feed it.
        cfg = lmw._xiangqi_profile(arg)
        print_full(f"xiangqi engine-id {arg}  (env-resolved via _xiangqi_profile)", cfg)
        print("  NOTE: xiangqi config is ENV-FLAG-DRIVEN. Above reflects the CURRENT process env")
        print("        (FOW_XIANGQI_* + FOW_XIANGQI_FAITHFUL). In prod the env is set by Railway.")
    else:
        print(f"unknown engine-id/version: {arg!r}")
        print(f"  known chess ids: {sorted(CHESS_ID_TO_VERSION)}")
        print(f"  known versions:  {sorted(ep.PROFILES)}")


def print_summary() -> None:
    cols = ["resolve_gadget", "gadget_alpha", "resolve_cvar_q", "hv_prune_adaptive",
            "i_sample_size", "kluss_k", "king_aware_leaf", "early_stop"]
    print("CHESS engine-id -> version -> resolved config (RUN, don't read comments):\n")
    hdr = f"{'engine-id':22} {'ver':8} " + " ".join(f"{c[:9]:>9}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for eid, ver in CHESS_ID_TO_VERSION.items():
        cfg = chess_config(ver)
        row = f"{eid:22} {ver:8} " + " ".join(f"{cfg.get(c)!s:>9}" for c in cols)
        print(row)
    print("\nWhich id is SHIPPED/LIVE is chosen platform-side (mistboard server), NOT here.")
    print("Inline 'SHIPPED' comments in live_move_worker.py are stale/contradictory — ignore them.")
    if XIANGQI_ID_TO_PROFILE:
        print(f"\nXIANGQI engine-ids: {sorted(XIANGQI_ID_TO_PROFILE)}")
    print("Xiangqi config is ENV-DRIVEN — run e.g. `show_engine_config.py python-fdx-v1.0` "
          "(optionally with FOW_XIANGQI_* set) to resolve it.")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print_summary()
    else:
        for a in args:
            resolve_one(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
