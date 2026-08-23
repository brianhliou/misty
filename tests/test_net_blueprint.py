"""Unit test for NetBlueprint (Obscuro-Parity Phase 1 learned-CFV blueprint).

Self-contained: builds a tiny synthetic .npz (the real value nets live under the
gitignored lab/), so this runs in CI without the weights. Verifies the pure-numpy
forward produces a value in [-1, 1], caches by FEN, and that opp_strategy is a
valid distribution. The de-fatalization behavior itself is validated by the
ply-197 gate (lab/repro_ply197_kingsafety.py), not here.
"""

import chess
import numpy as np

from fow_chess.cfr.blueprint import Blueprint, NetBlueprint


def _write_net(path):
    rng = np.random.default_rng(0)
    np.savez(
        path,
        **{
            "fc1.weight": (rng.standard_normal((256, 768)) * 0.01).astype(np.float32),
            "fc1.bias": np.zeros(256, np.float32),
            "fc2.weight": (rng.standard_normal((256, 256)) * 0.01).astype(np.float32),
            "fc2.bias": np.zeros(256, np.float32),
            "fc3.weight": (rng.standard_normal((1, 256)) * 0.01).astype(np.float32),
            "fc3.bias": np.zeros(1, np.float32),
        },
    )


def test_net_blueprint_forward_and_cache(tmp_path):
    p = tmp_path / "w.npz"
    _write_net(p)
    bp = NetBlueprint(str(p), chess.BLACK)

    v = bp.opp_cfv(chess.Board())
    assert isinstance(v, float)
    assert -1.0 <= v <= 1.0
    # Cache: identical board → identical value (and exercises the cache hit path).
    assert bp.opp_cfv(chess.Board()) == v
    # A different position gives a (generally) different value.
    other = bp.opp_cfv(chess.Board("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"))
    assert -1.0 <= other <= 1.0


def test_net_blueprint_satisfies_protocol_and_uniform_strategy(tmp_path):
    p = tmp_path / "w.npz"
    _write_net(p)
    bp = NetBlueprint(str(p), chess.WHITE)
    assert isinstance(bp, Blueprint)  # runtime_checkable Protocol

    moves = list(chess.Board().legal_moves)[:4]
    strat = bp.opp_strategy(0, moves)
    assert abs(sum(strat.values()) - 1.0) < 1e-9
    assert bp.opp_strategy(0, []) == {}
