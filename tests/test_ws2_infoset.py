"""WS2 slice 2: the Rust tree interns infosets natively (obs-history rolling
hash) so its infoset PARTITION matches Python's info_set_id grouping — including
FoW merging, where opponent moves the side-to-move can't see collapse into one
infoset. This is the prerequisite for moving leaf-selection + the eq pass onto
the authoritative Rust tree (they key off infosets).

Validated by EQUIVALENCE (not a bakeoff): build a tree with add_root_from_fen +
expand_node, and assert the map rust_infoset <-> python_info_set_id is a
bijection over every node (catches both over-merging — a hash collision — and
over-splitting).
"""
import random

import chess
import pytest

import fow_rust
from fow_chess.cfr.walker import obs_keys_both

pytestmark = pytest.mark.skipif(
    not hasattr(fow_rust, "EqEngine") or not hasattr(fow_rust.EqEngine, "node_infoset"),
    reason="fow_rust EqEngine.node_infoset not built",
)


def _engine():
    st = random.Random(0).getstate()[1]
    return fow_rust.EqEngine(list(st[:624]), st[624])


def _info_set_id(node):
    """Mirror GTCFRTreeNode.info_set_id: (to_move, obs_history_of_side_to_move)."""
    history = node["obs_w"] if node["to_move"] == chess.WHITE else node["obs_b"]
    return (node["to_move"], history)


def _expand_both(eng, node, perspective_white=True):
    """Expand one node on the Rust tree AND mirror it in the Python oracle.
    Both iterate pseudo-legal moves in python-chess order (pinned by
    test_expand_node_order_vs_pychess), so the i-th children correspond."""
    rust_children = eng.expand_node(node["rust_id"], perspective_white)
    board = node["board"]
    moves = list(board.pseudo_legal_moves)
    assert len(rust_children) == len(moves)
    out = []
    for rc, mv in zip(rust_children, moves):
        r_from, r_to, r_promo, r_cid = rc[0], rc[1], rc[2], rc[3]
        assert (r_from, r_to, r_promo) == (mv.from_square, mv.to_square, mv.promotion or 0)
        nb = board.copy()
        nb.push(mv)
        kw, kb = obs_keys_both(board, nb)
        out.append(
            {
                "rust_id": r_cid,
                "board": nb,
                "obs_w": node["obs_w"] + (kw,),
                "obs_b": node["obs_b"] + (kb,),
                "to_move": not node["to_move"],
            }
        )
    return out


def _assert_bijection(eng, nodes):
    rust2py: dict = {}
    py2rust: dict = {}
    for n in nodes:
        ri = eng.node_infoset(n["rust_id"])
        pi = _info_set_id(n)
        prev_py = rust2py.setdefault(ri, pi)
        assert prev_py == pi, f"rust infoset {ri} maps to two python ids (over-merge)"
        prev_ru = py2rust.setdefault(pi, ri)
        assert prev_ru == ri, f"python id maps to two rust infosets (over-split)"
    return len(rust2py)


def test_roots_share_infoset():
    """All white-to-move roots have info_set_id (WHITE, ()) → one infoset."""
    eng = _engine()
    starts = [chess.Board().fen()]
    b = chess.Board()
    b.push_san("e4")
    b.push_san("e5")
    starts.append(b.fen())  # also white to move
    ids = {eng.node_infoset(eng.add_root_from_fen(f)) for f in starts}
    assert len(ids) == 1, "white-to-move roots must collapse to one infoset"


def test_infoset_partition_matches_python_with_fow_merging():
    eng = _engine()
    # Start position, white to move → children are black-to-move. Most of white's
    # 20 opening moves are invisible to black → their black observations collide →
    # the children MERGE into few black infosets. The bijection check verifies the
    # Rust merging matches Python info_set_id exactly.
    root = {
        "rust_id": eng.add_root_from_fen(chess.Board().fen()),
        "board": chess.Board(),
        "obs_w": (),
        "obs_b": (),
        "to_move": chess.WHITE,
    }
    nodes = [root]
    kids = _expand_both(eng, root)
    nodes += kids
    # go two more plies deep on a few branches for shared-history coverage
    for c in kids[:6]:
        gk = _expand_both(eng, c)
        nodes += gk
        if gk:
            nodes += _expand_both(eng, gk[0])

    n_infosets = _assert_bijection(eng, nodes)
    # sanity: the start-position children really did merge (far fewer infosets
    # than nodes), i.e. we actually exercised FoW merging, not all-distinct.
    assert n_infosets < len(nodes), "expected FoW merging (fewer infosets than nodes)"
    # and specifically: white's 20 opening moves collapse for black's view
    child_infosets = {eng.node_infoset(c["rust_id"]) for c in kids}
    assert len(child_infosets) < len(kids), "opening moves should merge for black"
