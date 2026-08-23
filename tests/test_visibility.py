import chess

from fow_chess.visibility import visible_piece_map, visible_squares


def test_side_sees_own_pieces_from_initial_position() -> None:
    board = chess.Board()

    visible = visible_squares(board, chess.WHITE)

    assert chess.E1 in visible
    assert chess.A2 in visible
    assert chess.H2 in visible


def test_side_does_not_see_hidden_back_rank_from_initial_position() -> None:
    board = chess.Board()

    visible = visible_squares(board, chess.WHITE)

    assert chess.E8 not in visible
    assert chess.A8 not in visible


def test_visible_piece_map_includes_enemy_piece_on_attacked_square() -> None:
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E4, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.F6, chess.Piece(chess.QUEEN, chess.BLACK))

    pieces = visible_piece_map(board, chess.WHITE)

    assert pieces[chess.E4] == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert pieces[chess.F6] == chess.Piece(chess.QUEEN, chess.BLACK)


def test_pawn_diagonals_invisible_when_empty() -> None:
    board = chess.Board()
    board.push_uci("e2e4")

    visible = visible_squares(board, chess.WHITE)

    assert chess.E5 in visible
    assert chess.D5 not in visible
    assert chess.F5 not in visible


def test_pawn_diagonal_visible_when_capture_legal() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")

    visible = visible_squares(board, chess.WHITE)

    assert chess.D5 in visible


def test_visibility_independent_of_whose_turn_it_is() -> None:
    board = chess.Board()
    board.push_uci("e2e4")

    white_after_e4 = visible_squares(board, chess.WHITE)
    board.push_uci("e7e5")
    white_after_black_move = visible_squares(board, chess.WHITE)

    assert chess.E4 in white_after_e4
    assert chess.E4 in white_after_black_move


def test_chess960_castling_visibility_rejects_occupied_final_rook_square() -> None:
    board = chess.Board.empty(chess960=True)
    board.set_piece_at(chess.B1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.C1, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.F1, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    board.castling_rights = chess.BB_C1

    visible = visible_squares(board, chess.WHITE)

    assert chess.G1 not in visible
    assert not any(board.is_castling(move) for move in board.pseudo_legal_moves)
