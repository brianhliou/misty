#!/usr/bin/env python3
"""Provision the official Stockfish 18 x86-64 release binary on the bakeoff box
and print its absolute path to stdout (diagnostics -> stderr, so stdout stays a
clean path for `FOW_STOCKFISH="$(python3 scripts/ensure_sf18.py)"`).

Why this exists
---------------
The Railway box installs apt Stockfish (~SF14) via the railpack `aptPackages`
list, while every local rig runs homebrew SF18. Two different eval functions
made live engine picks irreproducible under local replay (the "live-vs-replay
residual"). Pinning the SAME SF18 the rigs use makes the depth-1 leaf eval
identical (NNUE is deterministic across x86-64/arm64) and closes that gap.

Why Python (not a shell script)
-------------------------------
The DEPLOY container is minimal: it has apt `stockfish` + `git` + a mise python,
but NOT `curl` (curl exists only in the BUILD image) — and `tar`/`flock`/`find`
aren't guaranteed either. The first cut was bash+curl and died with
`curl: command not found`. python3 is the one interpreter guaranteed present (the
runner runs on it), and urllib+tarfile are stdlib, so this has zero external-tool
dependencies and can't hit that class of failure.

This is the RUNTIME pin for the bakeoff measurement track: it lands via the
runner's `git pull` (no image rebuild) and a ticket opts in by resolving
FOW_STOCKFISH through it. The DURABLE, image-baked pin (a railpack download into
/app/bin/stockfish, which also fixes prod engine-worker serving on SF14) ships
with the Misty 1.1 checkpoint, validated.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TAG = "sf_18"
CACHE = Path(os.environ.get("SF18_CACHE_DIR", "/tmp/sf18"))
BIN = CACHE / "stockfish"


def log(msg: str) -> None:
    print(f"[ensure_sf18] {msg}", file=sys.stderr, flush=True)


def verify(p: Path) -> bool:
    """True iff the binary runs and self-reports Stockfish 18."""
    if not (p.exists() and os.access(p, os.X_OK)):
        return False
    try:
        out = subprocess.run(
            [str(p)], input="uci\nquit\n", capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:
        return False
    return "Stockfish 18" in out


def pick_variant() -> str:
    """Best microarch the CPU advertises; fall back conservatively."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                if "avx2" in line:
                    return "avx2"
                if "sse4_1" in line:
                    return "sse41-popcnt"
                break
    except Exception:
        pass
    return "x86-64"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if verify(BIN):  # already provisioned in this container
        print(BIN)
        return

    variant = pick_variant()
    asset = f"stockfish-ubuntu-x86-64-{variant}.tar"
    url = f"https://github.com/official-stockfish/Stockfish/releases/download/{TAG}/{asset}"
    log(f"provisioning {asset}")

    with tempfile.TemporaryDirectory() as td:
        tarpath = Path(td) / "sf.tar"
        req = urllib.request.Request(url, headers={"User-Agent": "ensure_sf18"})
        with urllib.request.urlopen(req, timeout=180) as r, open(tarpath, "wb") as f:
            shutil.copyfileobj(r, f)
        with tarfile.open(tarpath) as tf:
            tf.extractall(td)  # trusted official release tar
        srcs = [p for p in Path(td).rglob("stockfish-ubuntu-x86-64-*") if p.is_file()]
        if not srcs:
            log(f"FATAL: binary not found in {asset}")
            sys.exit(1)
        os.chmod(srcs[0], 0o755)
        # atomic publish: rename within the same filesystem
        tmpbin = CACHE / ".stockfish.tmp"
        shutil.move(str(srcs[0]), str(tmpbin))
        os.replace(str(tmpbin), str(BIN))

    if not verify(BIN):
        log("FATAL: provisioned binary failed the SF18 check")
        sys.exit(1)
    log(f"ready: {BIN} ({variant})")
    print(BIN)


if __name__ == "__main__":
    main()
