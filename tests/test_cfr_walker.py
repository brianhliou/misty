"""Mechanics-correctness tests for the CFR subgame walker.

Verifies the properties in ``lab/diag/cfr-walker-test-plan.md``: no truth
leakage in info-set IDs, observation-history parity, legal-move parity,
terminal detection, and cross-branch independence.
"""

from __future__ import annotations

import chess

from fow_chess.cfr.walker import SubgameNode, _obs_key
from fow_chess.observation import observation_from_transition


def _start_board() -> chess.Board:
    return chess.Board()


# -- Basic construction ------------------------------------------------------


def test_root_initializes_from_truth_board():
    board = _start_board()
    root = SubgameNode.root(board)
    assert root.truth.fen() == board.fen()
    assert root.to_move == chess.WHITE
    assert root.depth == 0
    assert root.obs_history_white == ()
    assert root.obs_history_black == ()


def test_root_to_move_explicit_override():
    board = _start_board()
    root = SubgameNode.root(board, to_move=chess.BLACK)
    assert root.to_move == chess.BLACK


def test_root_truth_is_copy_not_reference():
    board = _start_board()
    root = SubgameNode.root(board)
    board.push(chess.Move.from_uci("e2e4"))
    # Mutating the source board must not change the root's truth.
    assert root.truth.fen() != board.fen()
    assert root.truth.fen() == _start_board().fen()


# -- Apply mechanics ---------------------------------------------------------


def test_apply_pushes_move_and_swaps_actor():
    root = SubgameNode.root(_start_board())
    move = chess.Move.from_uci("e2e4")
    child = root.apply(move)
    assert child.truth.move_stack[-1] == move
    assert child.to_move == chess.BLACK
    assert child.depth == 1


def test_apply_does_not_mutate_parent():
    root = SubgameNode.root(_start_board())
    original_fen = root.truth.fen()
    original_stack_len = len(root.truth.move_stack)
    _ = root.apply(chess.Move.from_uci("e2e4"))
    assert root.truth.fen() == original_fen
    assert len(root.truth.move_stack) == original_stack_len


def test_apply_extends_both_observation_histories():
    root = SubgameNode.root(_start_board())
    child = root.apply(chess.Move.from_uci("e2e4"))
    assert len(child.obs_history_white) == 1
    assert len(child.obs_history_black) == 1


# -- P3: Legal-move parity ---------------------------------------------------


def test_legal_moves_match_pseudo_legal_at_root():
    board = _start_board()
    root = SubgameNode.root(board)
    expected = sorted(m.uci() for m in board.pseudo_legal_moves)
    actual = sorted(m.uci() for m in root.legal_moves())
    assert actual == expected


def test_legal_moves_match_through_depth_3():
    """P3: legal-move parity at every visited node up to depth 3."""
    root = SubgameNode.root(_start_board())

    def check(node: SubgameNode, depth_remaining: int) -> None:
        expected = sorted(m.uci() for m in node.truth.pseudo_legal_moves)
        actual = sorted(m.uci() for m in node.legal_moves())
        assert actual == expected, f"mismatch at depth {node.depth}"
        if depth_remaining == 0 or node.is_terminal:
            return
        # Sample first 3 moves at each branch to keep the test fast.
        for move in list(node.truth.pseudo_legal_moves)[:3]:
            check(node.apply(move), depth_remaining - 1)

    check(root, 3)


# -- P4: Terminal detection --------------------------------------------------


def test_is_terminal_false_at_start():
    root = SubgameNode.root(_start_board())
    assert root.is_terminal is False


def test_is_terminal_when_black_king_captured():
    # Minimal board with white pieces only and a white king
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    board.remove_piece_at(chess.E8)
    root = SubgameNode.root(board)
    assert root.is_terminal is True
    assert root.legal_moves() == []


def test_is_terminal_when_white_king_captured():
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    board.remove_piece_at(chess.E1)
    root = SubgameNode.root(board)
    assert root.is_terminal is True
    assert root.legal_moves() == []


# -- P1: No truth leakage in info-set ID -------------------------------------


def test_info_set_id_does_not_depend_on_truth():
    """P1: identical observation histories produce identical info-set IDs.

    The walker's contract: info_set_id is a pure function of
    (to_move, obs_history_of_to_move). We assert this structurally — two
    SubgameNodes with different truths but identical observation histories
    return identical IDs.
    """
    board_a = _start_board()
    board_b = _start_board()
    board_b.push(chess.Move.from_uci("a2a3"))  # different truth

    fake_obs_history = ("fake-obs-step-1", "fake-obs-step-2")
    n_a = SubgameNode(
        truth=board_a,
        to_move=chess.WHITE,
        obs_history_white=fake_obs_history,
        obs_history_black=("other-1", "other-2"),
        depth=2,
    )
    n_b = SubgameNode(
        truth=board_b,  # different truth
        to_move=chess.WHITE,
        obs_history_white=fake_obs_history,  # same history
        obs_history_black=("other-1", "other-2"),
        depth=2,
    )
    assert n_a.info_set_id() == n_b.info_set_id()


def test_info_set_id_picks_history_by_to_move():
    """info_set_id uses the to-move player's history, not the other player's."""
    n = SubgameNode(
        truth=_start_board(),
        to_move=chess.BLACK,
        obs_history_white=("w-1",),
        obs_history_black=("b-1",),
        depth=1,
    )
    assert n.info_set_id() == (chess.BLACK, ("b-1",))


# -- P2: Observation history parity ------------------------------------------


def test_observation_history_parity_through_depth_3():
    """P2: walker's stored obs histories match stepwise observation_from_transition."""
    root = SubgameNode.root(_start_board())
    moves = [chess.Move.from_uci(s) for s in ("e2e4", "e7e5", "g1f3")]
    node = root
    expected_white: tuple = ()
    expected_black: tuple = ()
    for mv in moves:
        prev_truth = node.truth.copy()
        node = node.apply(mv)
        obs_w = observation_from_transition(prev_truth, node.truth, chess.WHITE)
        obs_b = observation_from_transition(prev_truth, node.truth, chess.BLACK)
        expected_white = expected_white + (_obs_key(obs_w),)
        expected_black = expected_black + (_obs_key(obs_b),)
        assert node.obs_history_white == expected_white
        assert node.obs_history_black == expected_black


# -- P5: Cross-branch independence -------------------------------------------


def test_cross_branch_independence():
    """P5: walking from one child must not mutate parent or sibling."""
    root = SubgameNode.root(_start_board())
    child_a = root.apply(chess.Move.from_uci("e2e4"))
    child_b = root.apply(chess.Move.from_uci("d2d4"))

    parent_fen_before = root.truth.fen()
    sibling_fen_before = child_b.truth.fen()
    sibling_stack_before = list(child_b.truth.move_stack)
    sibling_history_before = child_b.obs_history_white

    # Walk one more ply from child_a
    _ = child_a.apply(chess.Move.from_uci("e7e5"))

    assert root.truth.fen() == parent_fen_before
    assert root.obs_history_white == ()
    assert child_b.truth.fen() == sibling_fen_before
    assert list(child_b.truth.move_stack) == sibling_stack_before
    assert child_b.obs_history_white == sibling_history_before


def test_two_children_have_different_truths():
    root = SubgameNode.root(_start_board())
    child_a = root.apply(chess.Move.from_uci("e2e4"))
    child_b = root.apply(chess.Move.from_uci("d2d4"))
    assert child_a.truth.fen() != child_b.truth.fen()


# -- P6: End-to-end smoke from a real position -------------------------------


def test_smoke_depth_3_walk_after_opening_sequence():
    """P6 (light): walk depth 3 from a mid-opening position; legal-move parity throughout."""
    board = _start_board()
    for mv in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"):
        board.push(chess.Move.from_uci(mv))
    root = SubgameNode.root(board)

    visited = 0

    def walk(node: SubgameNode, d: int) -> None:
        nonlocal visited
        visited += 1
        expected = sorted(m.uci() for m in node.truth.pseudo_legal_moves)
        actual = sorted(m.uci() for m in node.legal_moves())
        assert actual == expected
        if d == 0 or node.is_terminal:
            return
        for move in list(node.truth.pseudo_legal_moves)[:3]:
            walk(node.apply(move), d - 1)

    walk(root, 3)
    assert visited > 1  # we actually walked something
