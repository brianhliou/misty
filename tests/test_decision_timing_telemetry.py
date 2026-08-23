"""Per-decision timing telemetry survives the worker's telemetry boundary.

gt_cfr computes a wall-time split for every decision (`component_ms`) plus the
search's own elapsed time and tree size, and the live worker dropped all three.
That left "why does Misty only get 1-3k iterations per move" answerable only by
inference from clock deltas, and left the unbudgeted remainder (belief
enumeration + transport, the ~5s floor in mistboard#283) unmeasured in prod.

`_v2_decision_telemetry` is best-effort by contract: it must never raise into
the move path. So these pin BOTH that the fields come through on the happy path
AND that a solver without them degrades to None instead of throwing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import live_move_worker as worker


class FakeEnumerator:
    size = 171
    downsample_count = 0
    last_pre_cap_count = 171


class FakeSolution:
    iterations = 2675
    elapsed_seconds = 5.2814
    total_tree_nodes = 4096
    component_ms = {
        "sf_eval": 3011.44,
        "sf_children": 812.06,
        "eq_pass": 940.2,
        "select_leaf": 201.55,
        "kluss": 88.0,
        "expand_seed": 140.9,
    }
    action_values_at_root = {
        chess.Move.from_uci("c3b4"): 0.3512,
        chess.Move.from_uci("d5b4"): 0.1741,
    }


class FakeEngine:
    enumerator = FakeEnumerator()
    last_solution = FakeSolution()


class FakeStrategy:
    _engine = FakeEngine()


def _telemetry(monkeypatch, strategy=None):
    monkeypatch.setattr(worker, "_engine_config_snapshot", lambda *_: {})
    return worker._v2_decision_telemetry(strategy or FakeStrategy())


def test_component_split_is_reported(monkeypatch):
    got = _telemetry(monkeypatch)["componentMs"]
    assert got["sf_eval"] == 3011.4
    assert set(got) == {
        "sf_eval",
        "sf_children",
        "eq_pass",
        "select_leaf",
        "kluss",
        "expand_seed",
    }


def test_search_time_and_tree_size_are_reported(monkeypatch):
    got = _telemetry(monkeypatch)
    # searchSeconds is what makes the unbudgeted remainder computable: the
    # caller's think time minus this is belief enumeration + transport.
    assert got["searchSeconds"] == 5.281
    assert got["treeNodes"] == 4096


def test_existing_fields_are_unchanged(monkeypatch):
    got = _telemetry(monkeypatch)
    assert got["beliefSize"] == 171
    assert got["iters"] == 2675
    assert got["moveRanking"] == [("c3b4", 0.3512), ("d5b4", 0.1741)]


def test_a_solver_without_the_split_degrades_to_none(monkeypatch):
    class Bare(FakeSolution):
        component_ms = None
        elapsed_seconds = None
        total_tree_nodes = None

    class BareEngine(FakeEngine):
        last_solution = Bare()

    class BareStrategy:
        _engine = BareEngine()

    got = _telemetry(monkeypatch, BareStrategy())
    # None, not {} or 0.0 — a missing split must stay distinguishable from a
    # measured all-zero one.
    assert got["componentMs"] is None
    assert got["searchSeconds"] is None
    assert got["iters"] == 2675


def test_a_junk_split_does_not_raise_into_the_move_path(monkeypatch):
    class Junk(FakeSolution):
        component_ms = {"sf_eval": "not-a-number", "eq_pass": 12.34}

    class JunkEngine(FakeEngine):
        last_solution = Junk()

    class JunkStrategy:
        _engine = JunkEngine()

    got = _telemetry(monkeypatch, JunkStrategy())
    assert got["componentMs"] == {"eq_pass": 12.3}
