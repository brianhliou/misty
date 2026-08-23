"""The analysis job runner: publication JSON in, analysis JSON out."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from game_corpus import corpus_game_paths

_SPEC = importlib.util.spec_from_file_location(
    "analyze_job", Path(__file__).parents[1] / "scripts" / "analyze_job.py"
)
analyze_job = importlib.util.module_from_spec(_SPEC)
sys.modules["analyze_job"] = analyze_job
_SPEC.loader.exec_module(analyze_job)

_PROMO = {"queen": "queen", "rook": "rook"}


def _fixture_publication(limit_plies: int) -> dict:
    path = corpus_game_paths(family="fow-eval-seed1", limit=1)[0]
    plies = []
    with path.open() as f:
        for line in f:
            event = json.loads(line)
            if event.get("type") != "move-played":
                continue
            mv = event["move"]
            plies.append(
                {
                    "ply": len(plies) + 1,
                    "mover": event["color"],
                    "uci": mv["from"] + mv["to"],
                }
            )
            if len(plies) >= limit_plies:
                break
    return {
        "schema_version": "test",
        "game_id": "fixture-0000",
        "variant": "fog-of-war",
        "plies": plies,
    }


@pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="job needs Stockfish"
)
def test_job_produces_full_document():
    pub = _fixture_publication(8)
    result = analyze_job.run_job(
        pub, sf_depth=6, iterations=40, i_sample=4
    )
    assert result["schema_version"] == "misty-analysis/1"
    assert result["game_id"] == "fixture-0000"
    # evals: one per cursor 0..N
    assert len(result["evals"]) == 9
    assert result["evals"][0]["ply"] == 0
    # both seats analyzed, 4 own plies each
    for seat in ("white", "black"):
        seat_doc = result["seats"][seat]
        assert len(seat_doc["rows"]) == 4
        budget = seat_doc["budget"]
        assert budget["plies"] == 4
        assert budget["graded"] + budget["ungradeable"] == 4
        for row in seat_doc["rows"]:
            assert row["belief"]["truth_in_p"] is True
            assert "i_size" in row["belief"]
            assert "search" in row
    # the whole document is JSON-serializable
    json.dumps(result)


@pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="job needs Stockfish"
)
def test_job_no_search_mode_is_lighter():
    pub = _fixture_publication(6)
    result = analyze_job.run_job(pub, sf_depth=6, search=False)
    assert result["search"]["enabled"] is False
    row = result["seats"]["white"]["rows"][0]
    assert "search" not in row
    assert "i_size" not in row["belief"]


def test_moves_from_publication_promotion_tolerance():
    pub = {"plies": [{"ply": 1, "mover": "white", "uci": "e2e4"},
                     {"ply": 2, "mover": "black", "uci": "a7a8queen"}]}
    moves = analyze_job.moves_from_publication(pub)
    assert moves[0].uci() == "e2e4"
    assert moves[1].uci() == "a7a8q"
