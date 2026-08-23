"""WS2 slice 2 (prep): the Rust tree (EqEngine.expand_node) reproduces the
Stockfish-FREE structural half of gt_cfr.expand_leaf — pseudo-legal move set in
python-chess order, the child boards, the (white, black) FoW observation keys,
and king-capture terminal detection with the exact perspective-POV value.

This is the riskiest piece of the eventual coherent Rust-tree build, so it is
pinned offline against the two primitives expand_leaf actually composes
(``board.pseudo_legal_moves`` + ``walker.obs_keys_both``) — no Stockfish, no
live-search wiring. The Stockfish leaf value / regret seed are deliberately out
of scope (they enter later at the FFI batch boundary).
"""
import random

import chess
import pytest

import fow_rust
from fow_chess.cfr.walker import _key_from_components, obs_keys_both

pytestmark = pytest.mark.skipif(
    not hasattr(fow_rust, "EqEngine") or not hasattr(fow_rust.EqEngine, "expand_node"),
    reason="fow_rust EqEngine.expand_node not built",
)


def _engine():
    st = random.Random(0).getstate()[1]
    return fow_rust.EqEngine(list(st[:624]), st[624])


def _random_positions(n, seed=11):
    """Random positions, biased to include some with a captured king (terminal
    children) by occasionally letting the walk run into king captures."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(2, 40)):
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(rng.choice(legal))
            if b.king(chess.WHITE) is None or b.king(chess.BLACK) is None:
                break
        # only expand non-terminal positions (expand_leaf bails on terminal)
        if b.king(chess.WHITE) is not None and b.king(chess.BLACK) is not None:
            out.append(b.fen())
    return out


def _expected_children(fen, perspective_white):
    """The structural half of expand_leaf, computed from python-chess +
    walker.obs_keys_both — the oracle the Rust expand_node must match."""
    b = chess.Board(fen)
    perspective = chess.WHITE if perspective_white else chess.BLACK
    out = []
    for mv in b.pseudo_legal_moves:
        nb = b.copy()
        nb.push(mv)
        kw, kb = obs_keys_both(b, nb)
        white_gone = nb.king(chess.WHITE) is None
        black_gone = nb.king(chess.BLACK) is None
        is_terminal = white_gone or black_gone
        if is_terminal:
            own_gone = white_gone if perspective_white else black_gone
            opp_gone = black_gone if perspective_white else white_gone
            if own_gone and opp_gone:
                lv = 0.0
            elif own_gone:
                lv = -1.0
            else:
                lv = 1.0
        else:
            lv = 0.0  # Stockfish-free: placeholder, matches expand_node's
        out.append(
            (mv.from_square, mv.to_square, mv.promotion or 0, nb.fen(), is_terminal, lv, kw, kb)
        )
    return out


@pytest.mark.parametrize("perspective_white", [True, False])
def test_expand_node_matches_expand_leaf_structure(perspective_white):
    """Per-move SEMANTIC equivalence (matched by move identity, not position):
    same move set, and for each move the same child board, terminal flag,
    perspective-POV terminal value, and both FoW observation keys.

    Child ORDER is validated separately in test_expand_node_order_vs_pychess.
    """
    eng = _engine()
    for fen in _random_positions(40):
        nid = eng.add_root_from_fen(fen)
        rust = {
            (r[0], r[1], r[2]): (eng.node_fen(r[3]), r[4], r[5], r[6], r[7])
            for r in eng.expand_node(nid, perspective_white)
        }
        expected = {
            (e[0], e[1], e[2]): (e[3], e[4], e[5], e[6], e[7])
            for e in _expected_children(fen, perspective_white)
        }
        assert rust.keys() == expected.keys(), f"move set differs: fen={fen!r}"
        for mv, (e_fen, e_term, e_lv, e_wk, e_bk) in expected.items():
            r_fen, r_term, r_lv, r_wk, r_bk = rust[mv]
            ctx = f"fen={fen!r} move={mv}"
            assert r_fen == e_fen, f"child board: {ctx}"
            assert r_term == e_term, f"terminal flag: {ctx}"
            assert r_lv == e_lv, f"leaf value: {ctx}"
            assert _key_from_components(*r_wk) == e_wk, f"white key: {ctx}"
            assert _key_from_components(*r_bk) == e_bk, f"black key: {ctx}"


def test_expand_node_order_vs_pychess():
    """expand_node emits children in EXACTLY python-chess pseudo_legal_moves
    order — so the eventual coherent WS2 build is equivalence-checkable (a Rust
    tree byte-identical to the Python reference), not just bakeoff-validated.
    External-sampling maps an RNG draw to a child by index, so order is
    load-bearing. Pins all of: piece/castling/capture/advance/ep grouping and
    the Q,R,B,N promotion sub-order."""
    eng = _engine()
    for fen in _random_positions(60, seed=23):
        nid = eng.add_root_from_fen(fen)
        rust_moves = [(r[0], r[1], r[2]) for r in eng.expand_node(nid, True)]
        exp_moves = [(e[0], e[1], e[2]) for e in _expected_children(fen, True)]
        assert rust_moves == exp_moves, f"order differs: fen={fen!r}"


# Hand-picked positions exercising the ordering edge cases the random walk
# rarely hits: castling (both sides), pawn promotion by capture AND by advance
# (Q,R,B,N sub-order), and en passant (must sort LAST, after advances).
_EDGE_FENS = [
    # White: both-side castling available + pieces + pawns (groups 0,1,3,4).
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    # White pawn one step from promotion, with a capture-promo available too.
    "n1n5/PPP5/8/8/8/8/8/4K2k w - - 0 1",
    # En passant available for white (black just played d7-d5).
    "rnbqkbnr/ppp1pppp/8/2PpP3/8/8/PP1P1PPP/RNBQKBNR w KQkq d6 0 1",
    # En passant available for black (white just played e2-e4).
    "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # Black castling + promotion-rank pawns (mirror of the white cases).
    "r3k2r/4p3/8/8/8/8/pppPPPPP/4K3 b kq - 0 1",
]


@pytest.mark.parametrize("fen", _EDGE_FENS)
def test_expand_node_order_edge_cases(fen):
    eng = _engine()
    perspective_white = chess.Board(fen).turn == chess.WHITE
    nid = eng.add_root_from_fen(fen)
    rust_moves = [(r[0], r[1], r[2]) for r in eng.expand_node(nid, perspective_white)]
    exp_moves = [(e[0], e[1], e[2]) for e in _expected_children(fen, perspective_white)]
    assert rust_moves == exp_moves


def test_expand_node_links_children_and_flips_to_move():
    eng = _engine()
    # white to move at the start → children are black-to-move
    nid = eng.add_root_from_fen(chess.Board().fen())
    children = eng.expand_node(nid, True)
    assert len(children) == 20  # 20 pseudo-legal opening moves
    for _f, _t, _p, cid, _term, _lv, _wk, _bk in children:
        # each child holds its own board and is black-to-move ("... b ...")
        cfen = eng.node_fen(cid)
        assert cfen is not None
        assert cfen.split()[1] == "b"


def test_expand_node_rejects_boardless_node():
    eng = _engine()
    nid = eng.add_node(True, False, 0.0, 0.0, 0)  # mirror-built, pos=None
    with pytest.raises(ValueError):
        eng.expand_node(nid, True)
