"""FoW castle-into-check: the engine's move space must include castling onto an
attacked square, which python-chess forbids but the FoW rules allow (the server's
`fog-castle-through-check` rule, packages/game/src/variants.ts).

Regression for prod game a6f2e491 (2026-06-20): Misty 1.3 castled its king onto an
attacked g8 (fog-hidden white queen on b3) and lost the king. Root cause: the search
generated castles with python-chess legality (no into-check), so it never produced —
and therefore never evaluated/devalued — the fatal castle in the belief worlds where
it loses. The fix points the search move-gen at the FoW generator (gen_fow in Rust,
ChessRules.pseudo_legal_moves in Python) that the belief admission already used.
"""
import chess
from fow_chess.rules import ChessRules

# Black: Ke8, Rh8, kingside rights. White Qb3 attacks g8 (the castle destination)
# down an empty b3-g8 diagonal -> kingside castling walks the king into check.
FOG_CASTLE_FEN = "4k2r/8/8/8/8/1Q6/8/4K3 b k - 0 1"
OO = chess.Move.from_uci("e8g8")


def test_python_chess_baseline_excludes_castle_into_check():
    # The bug's origin: the standard reference refuses the castle.
    b = chess.Board(FOG_CASTLE_FEN)
    assert b.attackers(chess.WHITE, chess.G8), "fixture must have g8 attacked"
    assert OO not in b.pseudo_legal_moves


def test_rules_pseudo_legal_includes_fog_castle_into_check():
    # The fix: the engine's FoW move space includes it.
    rules = ChessRules()
    moves = list(rules.pseudo_legal_moves(chess.Board(FOG_CASTLE_FEN)))
    assert OO in moves


def test_safe_castle_position_is_unchanged():
    # No regression in normal positions: a safe castle is generated exactly as before
    # (extras empty -> base returned untouched, same order/object set as python-chess).
    rules = ChessRules()
    b = chess.Board("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1")  # g8 NOT attacked
    fow = list(rules.pseudo_legal_moves(b))
    assert fow == list(b.pseudo_legal_moves)


def test_pychess_order_key_is_idempotent_on_standard_moves():
    # The sort key must reproduce python-chess's native order (so re-sorting when a
    # fog-castle is added doesn't reorder the standard moves -> WS2 parity).
    rules = ChessRules()
    b = chess.Board()  # startpos
    base = list(b.pseudo_legal_moves)
    assert sorted(base, key=lambda m: rules._pychess_order_key(b, m)) == base
