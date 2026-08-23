"""Tests for Tier1Strategy short-circuits (king-capture, queen-capture, queen-save)."""

from __future__ import annotations

import chess

from fow_chess.belief import BeliefState
from fow_chess.engine import static_builder
from fow_chess.evaluator import material_evaluator
from fow_chess.move_priors import uniform_prior
from fow_chess.selfplay import PerspectiveView
from fow_chess.strategies import (
    Tier1Strategy,
    _categorize_king_defense_moves,
    _castle_moves,
    _king_defense_moves,
    _king_shelter_moves,
    _latent_ray_danger_probes,
    _latent_king_slider_block_moves,
    _prefer_higher_value_capture,
    _prefer_lower_value_attacker,
    _prefer_lower_value_same_target_capture,
    _prefer_queen_promotion,
    _queen_save_moves,
    _queen_save_tiers,
    _safe_visible_minor_or_rook_captures,
    _squares_attacked_by_visible_enemy,
)
from fow_chess.visibility import visible_piece_map, visible_squares


def _build_view(
    board: chess.Board,
    perspective: chess.Color,
    *,
    visible_pieces: dict[chess.Square, chess.Piece] | None = None,
) -> PerspectiveView:
    """When `visible_pieces` is set, override visibility — useful for tests
    that want to exercise short-circuit logic without coupling to FOW
    visibility rules."""
    pieces = (
        visible_pieces
        if visible_pieces is not None
        else visible_piece_map(board, perspective)
    )
    return PerspectiveView(
        perspective=perspective,
        own_legal_moves=list(board.pseudo_legal_moves) if board.turn == perspective else [],
        visible_squares=visible_squares(board, perspective),
        visible_piece_map=pieces,
    )


def _strategy(seed: int = 0) -> Tier1Strategy:
    s = Tier1Strategy(
        evaluator_builder=static_builder(material_evaluator()),
        move_prior=uniform_prior,
        target_n=4,
        max_eval_particles=4,
        seed=seed,
    )
    return s


def test_latent_ray_danger_probe_flags_fogged_queen_diagonal_to_king() -> None:
    # Regression for v0.8.0 g16 ply 7: the visible board is quiet, but the
    # fogged a5-e1 diagonal is tactically important. The diagnostic should
    # flag a low-belief hidden queen/bishop line and name available blockers.
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.C1, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.B1, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.D3, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E3, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    view = _build_view(board, chess.WHITE)
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=4,
        start_board=board,
    )

    probes = _latent_ray_danger_probes(view, belief, limit=16)

    queen_probe = next(
        probe
        for probe in probes
        if probe["target_square"] == "e1"
        and probe["danger_square"] == "a5"
        and probe["danger_piece"] == "q"
    )
    assert queen_probe["belief_mass"] == 0.0
    assert "b1c3" in queen_probe["blocking_moves"]
    assert "c1d2" in queen_probe["blocking_moves"]
    assert "b1c3" in queen_probe["actionable_blocking_moves"]
    assert "c1d2" in queen_probe["actionable_blocking_moves"]


def test_latent_king_slider_block_short_circuit_blocks_missing_queen_ray() -> None:
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.C1, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.B1, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.D3, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E3, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    view = _build_view(board, chess.WHITE)
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=4,
        start_board=board,
    )
    blockers = _latent_king_slider_block_moves(view, belief)

    assert {move.uci() for move in blockers} <= {"b1c3", "b1d2", "c1d2"}
    assert blockers

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    s._belief = belief
    chosen = s.pick_move(view)

    assert chosen.uci() in {move.uci() for move in blockers}
    assert s.trace_log[-1]["decision_path"] == "latent-king-slider-block"


def test_queen_capture_fires_when_visible() -> None:
    # White queen on c5 attacked by black bishop on a3. Queen does NOT attack
    # black king on h8 (so king-defense doesn't pre-empt). Capture should fire.
    board = chess.Board.empty()
    board.set_piece_at(chess.A3, chess.Piece(chess.BISHOP, chess.BLACK))
    board.set_piece_at(chess.C5, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.turn = chess.BLACK

    s = _strategy()
    s.reset(perspective=chess.BLACK)
    view = _build_view(board, chess.BLACK)
    chosen = s.pick_move(view)
    assert chosen.from_square == chess.A3
    assert chosen.to_square == chess.C5, f"expected queen capture, got {chosen}"


def test_queen_capture_prefers_least_valuable_attacker() -> None:
    # White can capture a visible black queen on d4 with either Qxd4 or Nxd4.
    # In fog, choose the knight first because d4 may be defended by hidden
    # pieces. Regression for q10 ply 11 from v0.7.0 affordance-check.
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.F3, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.BLACK))
    board.turn = chess.WHITE
    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.D1: chess.Piece(chess.QUEEN, chess.WHITE),
        chess.F3: chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.D4: chess.Piece(chess.QUEEN, chess.BLACK),
    }

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    chosen = s.pick_move(view)

    assert chosen.uci() == "f3d4"


def test_same_target_capture_helper_prefers_cheapest_attacker() -> None:
    # White can capture a visible black pawn on d4 with either Qxd4 or Nxd4.
    # If a selector already chose Qxd4, the helper should spend the knight
    # first because d4 may be defended by fog-hidden material.
    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.D1: chess.Piece(chess.QUEEN, chess.WHITE),
        chess.F3: chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.D4: chess.Piece(chess.PAWN, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, piece in pieces.items():
        board.set_piece_at(sq, piece)
    board.turn = chess.WHITE
    view = _build_view(board, chess.WHITE, visible_pieces=pieces)

    chosen = _prefer_lower_value_same_target_capture(
        chess.Move.from_uci("d1d4"),
        view.own_legal_moves,
        view,
    )

    assert chosen.uci() == "f3d4"


def test_main_eval_capture_spends_lower_value_attacker_on_same_target() -> None:
    # Integration shape for the same blindspot: even when main-eval ranks Qxd4
    # above Nxd4, Tier-1 rewrites the chosen visible capture to the cheaper
    # same-target attacker.
    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.D1: chess.Piece(chess.QUEEN, chess.WHITE),
        chess.F3: chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.D4: chess.Piece(chess.PAWN, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, piece in pieces.items():
        board.set_piece_at(sq, piece)
    board.turn = chess.WHITE

    def queen_biased_evaluator(
        _board: chess.Board, move: chess.Move, _perspective: chess.Color
    ) -> float:
        return 100.0 if move.uci() == "d1d4" else 0.0

    strategy = Tier1Strategy(
        evaluator_builder=static_builder(queen_biased_evaluator),
        move_prior=uniform_prior,
        target_n=1,
        max_eval_particles=1,
        seed=0,
    )
    strategy.reset(perspective=chess.WHITE)
    strategy._observed_ply = 20
    strategy._belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=1,
        particles=[board.copy()],
        weights=[1.0],
    )

    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    chosen = strategy.pick_move(view)

    assert chosen.uci() == "f3d4"
    assert strategy.trace_log[-1]["decision_path"] == "main-eval-lva-capture"


def test_main_eval_trace_reports_weight_mode_disagreement() -> None:
    # Posterior mass strongly trusts particle A, while uniform distinct worlds
    # give particle B equal voice. The trace should surface that disagreement
    # without changing the posterior-selected move.
    particle_a = chess.Board.empty()
    particle_a.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    particle_a.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    particle_a.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    particle_a.set_piece_at(chess.H7, chess.Piece(chess.PAWN, chess.BLACK))
    particle_a.turn = chess.WHITE

    particle_b = particle_a.copy()
    particle_b.remove_piece_at(chess.H7)
    particle_b.set_piece_at(chess.G7, chess.Piece(chess.PAWN, chess.BLACK))

    def world_sensitive_evaluator(
        board: chess.Board, move: chess.Move, _perspective: chess.Color
    ) -> float:
        if board.piece_at(chess.H7) is not None:
            return 100.0 if move.uci() == "a1a2" else 0.0
        return 200.0 if move.uci() == "a1b1" else 0.0

    strategy = Tier1Strategy(
        evaluator_builder=static_builder(world_sensitive_evaluator),
        move_prior=uniform_prior,
        target_n=2,
        max_eval_particles=2,
        seed=0,
    )
    strategy.reset(perspective=chess.WHITE)
    strategy._observed_ply = 20
    strategy._belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=2,
        particles=[particle_a, particle_b],
        weights=[9.0, 1.0],
    )

    view = _build_view(particle_a, chess.WHITE)
    chosen = strategy.pick_move(view)
    modes = strategy.trace_log[-1]["decision_weight_modes"]

    assert chosen.uci() == "a1a2"
    assert modes["winner_disagreement"] is True
    assert modes["mode_winners"]["posterior"] == "a1a2"
    assert modes["mode_winners"]["appearance"] == "a1b1"
    assert modes["mode_winners"]["uniform_distinct"] == "a1b1"


def test_king_capture_beats_queen_capture() -> None:
    # Black has both a king-capture (rook on g8 → enemy king on g1) AND a
    # queen-capture (knight on f3 → enemy queen on h2). King-capture must win.
    board = chess.Board.empty()
    board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H2, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.F3, chess.Piece(chess.KNIGHT, chess.BLACK))
    board.set_piece_at(chess.G8, chess.Piece(chess.ROOK, chess.BLACK))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.BLACK

    s = _strategy()
    s.reset(perspective=chess.BLACK)
    view = _build_view(board, chess.BLACK)
    chosen = s.pick_move(view)
    assert chosen.to_square == chess.G1, f"expected king capture, got {chosen}"


def test_queen_save_fires_when_attacked_with_safe_square() -> None:
    # White queen on d8 attacked by black pawn on e7 (pawn captures e7→d8).
    # White queen has safe square c7 (not attacked by visible black pieces).
    board = chess.Board.empty()
    board.set_piece_at(chess.D8, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    view = _build_view(board, chess.WHITE)
    chosen = s.pick_move(view)
    # The queen must move; the destination must not be attacked by the e7 pawn.
    assert chosen.from_square == chess.D8, f"expected queen move, got {chosen}"
    # e7 pawn attacks d8 (where queen was) and f8. d8 is queen's start, so any
    # destination not on f8 (and not staying) is safe in this stripped position.
    assert chosen.to_square != chess.D8


def test_queen_save_skips_when_queen_not_visibly_attacked() -> None:
    # Queen safe on d4; no enemy attacks it.
    board = chess.Board.empty()
    board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    view = _build_view(board, chess.WHITE)
    saves = _queen_save_moves(view)
    assert saves == [], f"expected empty (queen not under visible attack) but got {saves}"


def test_king_shelter_prefers_bishop_on_e_file_annotation_position() -> None:
    # Regression for annotation replay gate, hardobs g12 ply 23. White's
    # e-pawn is gone, king is still on e1, and both Bf1-e2 and Ng1-e2 can
    # shelter the king. Prefer bishop before main eval grabs material.
    board = chess.Board("r1b2rk1/pp3ppp/2n5/3p4/5P2/2P5/P4PPP/2RQKBNR w - - 0 1")

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    view = _build_view(board, chess.WHITE)
    shelter = _king_shelter_moves(view)
    chosen = s.pick_move(view)

    assert {move.uci() for move in shelter} == {"f1e2"}
    assert chosen.uci() == "f1e2"
    assert s.trace_log[-1]["decision_path"] == "king-shelter"


def test_king_shelter_prefers_knight_over_retracting_developed_bishop() -> None:
    # Follow-up annotation replay case, hardobs g12 ply 25. Once the bishop
    # has already developed to d3, don't pull it back to e2 when the knight can
    # provide the same shelter from g1.
    board = chess.Board("r1b2rk1/pp3ppp/2n5/8/3p1P2/2PB4/P4PPP/2RQK1NR w - - 0 1")

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    view = _build_view(board, chess.WHITE)
    shelter = _king_shelter_moves(view)
    chosen = s.pick_move(view)

    assert {move.uci() for move in shelter} == {"g1e2"}
    assert chosen.uci() == "g1e2"
    assert s.trace_log[-1]["decision_path"] == "king-shelter"


def test_king_shelter_skips_visibly_attacked_shelter_square() -> None:
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.F1, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.G1, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.H5, chess.Piece(chess.BISHOP, chess.BLACK))
    board.turn = chess.WHITE
    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.F1: chess.Piece(chess.BISHOP, chess.WHITE),
        chess.G1: chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.H5: chess.Piece(chess.BISHOP, chess.BLACK),
    }

    view = _build_view(board, chess.WHITE, visible_pieces=pieces)

    assert _king_shelter_moves(view) == []


def test_king_defense_picks_king_flight() -> None:
    # Black king on e8 attacked by white bishop on a4 (a4-e8 diagonal). Black
    # king should flee to d8 or e7 (not on the diagonal).
    board = chess.Board.empty()
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A4, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.turn = chess.BLACK

    pieces = {
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.A4: chess.Piece(chess.BISHOP, chess.WHITE),
        chess.A1: chess.Piece(chess.KING, chess.WHITE),
    }

    s = _strategy()
    s.reset(perspective=chess.BLACK)
    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    chosen = s.pick_move(view)
    assert chosen.from_square == chess.E8, f"expected king move, got {chosen}"


def test_king_defense_captures_attacker() -> None:
    # White king on e1, black knight on f3 attacking e1. White rook on f1 can
    # capture f3. King-defense should include that capture.
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.F3, chess.Piece(chess.KNIGHT, chess.BLACK))
    board.set_piece_at(chess.F1, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.F3: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.F1: chess.Piece(chess.ROOK, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    moves = _king_defense_moves(view)
    move_set = {m.uci() for m in moves}
    assert 'f1f3' in move_set, f"expected f1f3 (rook captures knight) in {move_set}"


def test_king_defense_blocks_sliding_attack() -> None:
    # White king on e1, black rook on e8 attacking down e-file. White knight on
    # g1 → e2 blocks the e-file.
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.ROOK, chess.BLACK))
    board.set_piece_at(chess.G1, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.ROOK, chess.BLACK),
        chess.G1: chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.A8: chess.Piece(chess.KING, chess.BLACK),
    }
    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    moves = _king_defense_moves(view)
    move_set = {m.uci() for m in moves}
    assert 'g1e2' in move_set, f"expected g1e2 (knight blocks e-file) in {move_set}"


def test_king_defense_skips_when_king_not_attacked() -> None:
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D4, chess.Piece(chess.KNIGHT, chess.BLACK))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE

    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.D4: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    assert _king_defense_moves(view) == []


def test_recapture_exempt_from_bad_capture_trade_veto() -> None:
    """When opp just captured one of our pieces, _belief_veto_bad_capture_trade
    must NOT drop our recapture even when belief sees a possible counter-recapture.
    Otherwise the engine refuses to recapture and bleeds material — see
    vs-brian-game-1/2 ply 12 (white plays exd5 against black's d-pawn, engine
    fails to retake e6d5).
    """
    # Position: black to move; e6 black pawn can recapture white pawn on d5.
    # White knight on c3 can recapture the d5 pawn → without the exemption,
    # the equal-value-trade veto fires and drops e6d5.
    board = chess.Board.empty()
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E6, chess.Piece(chess.PAWN, chess.BLACK))
    board.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.C3, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.turn = chess.BLACK
    pieces = {
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.A1: chess.Piece(chess.KING, chess.WHITE),
        chess.E6: chess.Piece(chess.PAWN, chess.BLACK),
        chess.D5: chess.Piece(chess.PAWN, chess.WHITE),
        chess.C3: chess.Piece(chess.KNIGHT, chess.WHITE),
    }
    s = _strategy()
    s.reset(perspective=chess.BLACK)
    # Tell the strategy that opp just captured on d5 (this is what
    # observe_opp_move would set when white played exd5).
    s._opp_last_capture_target = chess.D5
    # Seed belief with a particle that matches truth so the veto's recapture
    # check sees a counter-recapture (c3 knight → d5).
    assert s._belief is not None
    s._belief.particles = [board.copy() for _ in range(8)]
    s._belief.weights = [1.0] * 8

    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    # The veto without the exemption would drop e6d5; with the exemption it stays.
    survivors = s._belief_veto_bad_capture_trade(
        [chess.Move.from_uci('e6d5'), chess.Move.from_uci('e6e5')],
        view,
    )
    survivor_ucis = {m.uci() for m in survivors}
    assert 'e6d5' in survivor_ucis, f"recapture must survive the bad-capture-trade veto; got {survivor_ucis}"


def test_bad_capture_trade_still_filters_non_recapture() -> None:
    """The recapture exemption must not collapse the veto's normal behavior:
    when our capture is NOT a recapture, the equal-value trade veto should
    still fire if belief says opp recaptures.
    """
    board = chess.Board.empty()
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E6, chess.Piece(chess.PAWN, chess.BLACK))
    board.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.C3, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.turn = chess.BLACK
    pieces = {
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.A1: chess.Piece(chess.KING, chess.WHITE),
        chess.E6: chess.Piece(chess.PAWN, chess.BLACK),
        chess.D5: chess.Piece(chess.PAWN, chess.WHITE),
        chess.C3: chess.Piece(chess.KNIGHT, chess.WHITE),
    }
    s = _strategy()
    s.reset(perspective=chess.BLACK)
    # Opp did NOT just capture on d5 — this is a fresh capture initiation.
    s._opp_last_capture_target = None
    assert s._belief is not None
    s._belief.particles = [board.copy() for _ in range(8)]
    s._belief.weights = [1.0] * 8

    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    survivors = s._belief_veto_bad_capture_trade(
        [chess.Move.from_uci('e6d5'), chess.Move.from_uci('e6e5')],
        view,
    )
    survivor_ucis = {m.uci() for m in survivors}
    assert 'e6d5' not in survivor_ucis, f"non-recapture must still be filtered; got {survivor_ucis}"


def test_phantom_check_guard_dismisses_blocked_slider_attack() -> None:
    """When a visible enemy slider APPEARS to attack our king on the visibility-only
    board, but the belief has majority weight on a piece blocking the ray, the
    king-defense tier must not fire on a phantom check. This is the ply-36 class
    from the vs-brian-game-1 diagnostic: white bishop on b3 is visible, but a
    hidden white knight on d5 (in truth, and in most belief particles) blocks
    the b3-g8 long diagonal. The engine should NOT treat the bishop as a real
    attacker; downstream, it can pick a non-defensive move.
    """
    board = chess.Board.empty()
    board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A5, chess.Piece(chess.KNIGHT, chess.BLACK))
    board.set_piece_at(chess.D5, chess.Piece(chess.KNIGHT, chess.WHITE))  # hidden blocker
    board.set_piece_at(chess.B3, chess.Piece(chess.BISHOP, chess.WHITE))  # visible "attacker"
    board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
    board.turn = chess.BLACK
    # Black sees b3 (knight a5 attacks b3) and own pieces; d5 is NOT visible.
    pieces = {
        chess.G8: chess.Piece(chess.KING, chess.BLACK),
        chess.A5: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.B3: chess.Piece(chess.BISHOP, chess.WHITE),
        # d5 hidden from black; g1 hidden too
    }
    s = _strategy()
    s.reset(perspective=chess.BLACK)
    # Replace belief with particles that DO have the d5 knight (matching truth).
    # Use multiple identical copies so weights and ESS are non-degenerate.
    assert s._belief is not None
    s._belief.particles = [board.copy() for _ in range(8)]
    s._belief.weights = [1.0] * 8

    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    # The phantom-check guard should mark the b3 bishop as phantom (path
    # blocked by d5 knight in 100% of particles), and king-defense should NOT
    # be the decision path.
    real = s._real_king_threat_attackers(view)
    assert real == set(), f"expected b3 to be filtered as phantom; got {real}"


def test_phantom_check_guard_keeps_real_visible_threat() -> None:
    """When the belief AGREES the attack ray is clear, the attacker stays real
    and king-defense should fire. Otherwise the guard would defang all visible
    checks.
    """
    board = chess.Board.empty()
    board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.B3, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
    board.turn = chess.BLACK
    pieces = {
        chess.G8: chess.Piece(chess.KING, chess.BLACK),
        chess.B3: chess.Piece(chess.BISHOP, chess.WHITE),
    }
    s = _strategy()
    s.reset(perspective=chess.BLACK)
    # Belief has the b3 bishop AND no blockers on the b3-g8 diagonal.
    assert s._belief is not None
    s._belief.particles = [board.copy() for _ in range(8)]
    s._belief.weights = [1.0] * 8

    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    real = s._real_king_threat_attackers(view)
    assert chess.B3 in real, f"expected b3 to remain real; got {real}"


def test_phantom_check_guard_trusts_visibility_when_belief_stale() -> None:
    """If the belief doesn't even know about the visible attacker (stale belief
    from reset+custom view, or unseeded belief), the guard must trust
    visibility — otherwise it would dismiss every visible threat as phantom.
    """
    board = chess.Board.empty()
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A4, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.turn = chess.BLACK
    pieces = {
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.A4: chess.Piece(chess.BISHOP, chess.WHITE),
    }
    s = _strategy()
    s.reset(perspective=chess.BLACK)
    # Belief is the default initial-position belief from reset() — has nothing
    # on a4. The guard should fall back to trusting visibility.
    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    real = s._real_king_threat_attackers(view)
    assert chess.A4 in real, f"expected stale-belief fallback; got {real}"


def test_king_defense_beats_queen_capture_when_king_attacked() -> None:
    # Black king on e8 attacked by white bishop on a4 (a4-e8 diagonal). Black
    # also has a queen-capture available (rook on h8 can take queen on h1
    # straight down the h-file). King-defense must dominate.
    board = chess.Board.empty()
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A4, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.ROOK, chess.BLACK))
    board.set_piece_at(chess.H1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.turn = chess.BLACK

    pieces = {
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.A4: chess.Piece(chess.BISHOP, chess.WHITE),
        chess.A1: chess.Piece(chess.KING, chess.WHITE),
        chess.H8: chess.Piece(chess.ROOK, chess.BLACK),
        chess.H1: chess.Piece(chess.QUEEN, chess.WHITE),
    }
    s = _strategy()
    s.reset(perspective=chess.BLACK)
    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    chosen = s.pick_move(view)
    assert chosen.uci() != 'h8h1', f"engine took queen instead of defending king: {chosen}"


def test_prefer_queen_promotion_filters_to_queen() -> None:
    moves = [
        chess.Move.from_uci('d2e1q'),
        chess.Move.from_uci('d2e1r'),
        chess.Move.from_uci('d2e1b'),
        chess.Move.from_uci('d2e1n'),
    ]
    filtered = _prefer_queen_promotion(moves)
    assert len(filtered) == 1
    assert filtered[0].promotion == chess.QUEEN


def test_prefer_queen_promotion_passthrough_when_no_promotions() -> None:
    moves = [
        chess.Move.from_uci('e2e4'),
        chess.Move.from_uci('d2d4'),
    ]
    assert _prefer_queen_promotion(moves) == moves


def test_queen_save_includes_attacker_capture_by_other_piece() -> None:
    # Black queen on d3 attacked by white knight on e5. Black pawn on d6 can
    # capture the knight (d6e5; black-pawn diagonal capture toward rank 1).
    # Old queen-save only considered queen-moves; new general version includes
    # captures of the attacker by other pieces.
    pieces = {
        chess.D3: chess.Piece(chess.QUEEN, chess.BLACK),
        chess.E5: chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.D6: chess.Piece(chess.PAWN, chess.BLACK),
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.BLACK

    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    saves = _queen_save_moves(view)
    save_set = {m.uci() for m in saves}
    assert 'd6e5' in save_set, f"expected d6e5 (pawn captures attacker) in {save_set}"


def test_queen_save_prefers_attacker_capture_over_block() -> None:
    """When queen-save has capture and block options, capture the attacker.

    Regression for v0.7.27 g0 ply 113: white queen on b3 was attacked by a
    visible rook on b8. The engine played Ra5-b5 to block instead of Qxb8.
    """
    pieces = {
        chess.F2: chess.Piece(chess.KING, chess.WHITE),
        chess.B3: chess.Piece(chess.QUEEN, chess.WHITE),
        chess.A5: chess.Piece(chess.ROOK, chess.WHITE),
        chess.B8: chess.Piece(chess.ROOK, chess.BLACK),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.WHITE

    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    save_set = {move.uci() for move in _queen_save_moves(view)}
    assert "b3b8" in save_set
    assert "a5b5" in save_set
    assert [[move.uci() for move in tier] for tier in _queen_save_tiers(view)][0] == [
        "b3b8"
    ]

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [board.copy() for _ in range(4)]
    strategy._belief.weights = [1.0] * 4

    chosen = strategy.pick_move(view)

    assert chosen.uci() == "b3b8"
    assert strategy.trace_log[-1]["decision_path"] == "queen-save"


def test_squares_attacked_by_visible_enemy_basic() -> None:
    # Hand-build a PerspectiveView with an explicit visible_piece_map so we can
    # exercise the helper without relying on Mistboard visibility rules — black
    # rook on e4 + white king on e1, pretending both are mutually visible.
    pieces = {
        chess.E4: chess.Piece(chess.ROOK, chess.BLACK),
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
    }
    view = PerspectiveView(
        perspective=chess.WHITE,
        own_legal_moves=[],
        visible_squares=chess.SquareSet(),
        visible_piece_map=pieces,
    )
    attacked = _squares_attacked_by_visible_enemy(view)
    assert chess.E1 in attacked  # rook attacks down e-file to white king
    assert chess.A4 in attacked  # rook attacks across 4th rank
    assert chess.D5 not in attacked  # rook doesn't move diagonally


def test_capture_detection_decrements_opp_count_after_visible_capture() -> None:
    """When pick_move chooses a move landing on a visible enemy piece, the
    next observe_own_move should register the capture on belief.opp_remaining_counts."""
    # White queen on c5 captures black bishop on a3.
    board = chess.Board.empty()
    board.set_piece_at(chess.A3, chess.Piece(chess.BISHOP, chess.BLACK))
    board.set_piece_at(chess.C5, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.turn = chess.BLACK  # black moves; can capture white queen

    s = _strategy()
    s.reset(perspective=chess.BLACK)
    view = _build_view(board, chess.BLACK)
    chosen = s.pick_move(view)
    # Queen-capture short-circuit fires on a3xc5.
    assert chosen.from_square == chess.A3
    assert chosen.to_square == chess.C5
    assert s._pending_capture_type == chess.QUEEN

    # Drive observe_own_move to consume the pending capture. We need a real
    # observation, so build one from the board transition.
    from fow_chess.observation import observation_from_transition
    prev = board.copy()
    next_board = board.copy()
    next_board.push(chosen)
    obs = observation_from_transition(prev, next_board, chess.BLACK)

    # White-queen count was 1; after capture it should be 0.
    pre = s._belief.opp_remaining_counts[chess.QUEEN]
    s.observe_own_move(chosen, obs)
    post = s._belief.opp_remaining_counts[chess.QUEEN]
    assert pre == 1 and post == 0
    # Pending capture cleared after consumption.
    assert s._pending_capture_type is None


def test_capture_detection_skips_non_capture_move() -> None:
    """A quiet move should not flag a pending capture."""
    board = chess.Board()
    s = _strategy()
    s.reset(perspective=chess.WHITE)
    view = _build_view(board, chess.WHITE)
    s.pick_move(view)
    # Standard opening: nothing visibly captured.
    assert s._pending_capture_type is None


# ============================================================================
# v0.6.1 Pattern A: rank king-defense as captures > blocks > flights.
# ============================================================================


def test_king_defense_prefers_attacker_capture_over_flight() -> None:
    """White king on e1 attacked by black knight on f3. White pawn on g2 can
    capture the knight (g2xf3). King also has flight squares (e2, d2). The
    capture must win — flight was the v0.6.0-mirror bug."""
    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.G2: chess.Piece(chess.PAWN, chess.WHITE),
        chess.F3: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.WHITE

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    chosen = s.pick_move(view)
    assert chosen.from_square == chess.G2 and chosen.to_square == chess.F3, (
        f"expected attacker-capture g2xf3, got {chosen}"
    )


def test_king_defense_prefers_flight_over_king_capture_of_material() -> None:
    """Regression for v0.7.10 g12: don't walk king into fog for a pawn."""
    board = chess.Board("rnbq1rk1/p3nppp/1p1ppb2/8/P4P1P/2pPP3/2PK2P1/R1BQ1BNR w - - 0 11")
    risky = chess.Move.from_uci("d2c3")
    assert risky in board.pseudo_legal_moves

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]
    view = _build_view(board, chess.WHITE)

    chosen = strategy.pick_move(view)

    assert chosen != risky
    assert strategy.trace_log[-1]["decision_path"] == "king-defense-flight"


def test_king_defense_forced_risk_fallback_keeps_lowest_risk_flights() -> None:
    """If every king-defense flight is risky, keep the least-risk flights.

    Regression for v0.7.24 g0 ply 185: king-defense applied the king-risk
    veto, got no survivors, then fell back to the whole risky flight tier and
    discarded the relative risk signal.
    """
    visible = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.ROOK, chess.BLACK),
        chess.H8: chess.Piece(chess.KING, chess.BLACK),
    }
    base = chess.Board.empty()
    for sq, p in visible.items():
        base.set_piece_at(sq, p)
    base.turn = chess.WHITE

    lower_risk = base.copy()
    lower_risk.set_piece_at(chess.D8, chess.Piece(chess.ROOK, chess.BLACK))
    higher_risk = base.copy()
    higher_risk.set_piece_at(chess.F8, chess.Piece(chess.ROOK, chess.BLACK))

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [lower_risk] + [higher_risk] * 3
    strategy._belief.weights = [1.0] * 4
    view = _build_view(base, chess.WHITE, visible_pieces=visible)

    chosen = strategy.pick_move(view)

    assert chosen.uci() in {"e1d1", "e1d2"}
    assert strategy.trace_log[-1]["decision_path"] == "king-defense-flight"


def test_king_defense_safe_king_capture_of_rook_beats_flight() -> None:
    """A safe king capture of a visible rook attacker should beat flight.

    Regression for v0.7.25 g0 ply 195: white saw the black rook on g1
    attacking its king on f1, but king-defense treated all king captures as
    lower priority than flight and walked away from the rook.
    """
    visible = {
        chess.F1: chess.Piece(chess.KING, chess.WHITE),
        chess.G1: chess.Piece(chess.ROOK, chess.BLACK),
        chess.H8: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, p in visible.items():
        board.set_piece_at(sq, p)
    board.turn = chess.WHITE

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [board.copy() for _ in range(4)]
    strategy._belief.weights = [1.0] * 4
    view = _build_view(board, chess.WHITE, visible_pieces=visible)

    chosen = strategy.pick_move(view)

    assert chosen.uci() == "f1g1"
    assert strategy.trace_log[-1]["decision_path"] == "king-defense-king-capture"


def test_king_defense_prefers_higher_material_attacker_capture() -> None:
    """King attacked by visible queen; both a pawn and a rook can capture the
    queen. Pawn-takes-queen wins on material — but attacker-capture preference
    + max-material rule should pick whichever pawn or rook capture. We just
    verify a queen capture is selected (not a king flight).
    """
    # Black king on e8 in check from white queen on e4 (down e-file).
    # Black rook on a4 can capture queen (a4xe4). Black pawn on f5 can NOT
    # diagonally take e4 (it'd take whatever's on e4 from f5? f5 captures e4 OK).
    pieces = {
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.E4: chess.Piece(chess.QUEEN, chess.WHITE),
        chess.A4: chess.Piece(chess.ROOK, chess.BLACK),
        chess.F5: chess.Piece(chess.PAWN, chess.BLACK),
        chess.H1: chess.Piece(chess.KING, chess.WHITE),  # off rook's file/rank
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.BLACK

    s = _strategy()
    s.reset(perspective=chess.BLACK)
    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    chosen = s.pick_move(view)
    # Either a4xe4 or f5xe4 — both capture the queen. Reject king flight.
    assert chosen.to_square == chess.E4, f"expected attacker capture, got {chosen}"


def test_categorize_king_defense_moves_partitions_correctly() -> None:
    """Hand-built position: white king on e1 attacked by black bishop on a5
    (a5-e1 diagonal). Resolutions:
      - capture: white knight on b4 takes a5 (Nxa5)
      - block: white pawn on c2 doesn't block; white pawn on d2 plays d3 to
        block on d2's path? Actually the diagonal a5-b4-c3-d2-e1 — a knight
        on b4 already breaks it; need a different setup.
    Simpler: white king e1, bishop attacks via a5-b4-c3-d2-e1 with empty
    diagonal. Block by interposing on c3, d2, or b4.
    """
    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.A5: chess.Piece(chess.BISHOP, chess.BLACK),
        chess.G1: chess.Piece(chess.ROOK, chess.WHITE),  # for blocking via Rd1, etc.
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.WHITE

    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    captures, blocks, flights = _categorize_king_defense_moves(view)
    # The lone bishop is the only attacker. King flights to non-diagonal squares
    # exist (e.g., e2, f1). Blocks: nothing here can interpose since rook on
    # g1 can't reach d2/c3/b4 in one move along the rank? Rg1-d1 doesn't
    # interpose. So expect captures=[] (rook can't reach a5), blocks=[] (no
    # piece can interpose), flights non-empty.
    capture_squares = {m.to_square for m in captures}
    flight_squares = {m.to_square for m in flights}
    # No piece can capture a5.
    assert chess.A5 not in capture_squares
    # King can flee to e2 (not on diagonal).
    assert chess.E2 in flight_squares


# ============================================================================
# v0.6.1 Pattern A: belief-grounded king-attack veto.
# ============================================================================


def test_belief_veto_drops_candidate_when_majority_of_particles_attacked() -> None:
    """If most particles place a hidden bishop on a discovered-check line,
    veto the candidate move that exposes it."""
    # White king on e1, white knight on c3 (only piece blocking a black bishop's
    # check from a5 along a5-b4-c3-d2-e1 diagonal). If knight moves, king is
    # checked. We construct particles where the bishop on a5 is hypothesized;
    # verify candidate Nc3-Nd5 (which moves the knight off c3) gets vetoed.
    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.C3: chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    # Build a particle that has the bishop on a5 (hidden from white).
    particle = chess.Board.empty()
    for sq, p in pieces.items():
        particle.set_piece_at(sq, p)
    particle.set_piece_at(chess.A5, chess.Piece(chess.BISHOP, chess.BLACK))
    particle.turn = chess.WHITE

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    s._belief.particles = [particle.copy(), particle.copy()]
    s._belief.weights = [1.0, 1.0]
    move = chess.Move.from_uci("c3d5")  # knight off c3 → bishop checks e1
    view_board = chess.Board.empty()
    for sq, p in pieces.items():
        view_board.set_piece_at(sq, p)
    view_board.turn = chess.WHITE
    view = _build_view(view_board, chess.WHITE, visible_pieces=pieces)
    survivors = s._belief_veto_king_attack([move], view)
    assert survivors == [], "all particles agree on hidden discovered check; veto must fire"


def test_belief_veto_rejects_candidate_when_minority_terminal_risk() -> None:
    """Terminal king risk is not ordinary minority material uncertainty."""
    pieces = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.C3: chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    base = chess.Board.empty()
    for sq, p in pieces.items():
        base.set_piece_at(sq, p)
    base.turn = chess.WHITE
    # 1 of 3 particles hallucinates bishop on a5; 2 don't.
    p_with = base.copy()
    p_with.set_piece_at(chess.A5, chess.Piece(chess.BISHOP, chess.BLACK))

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    s._belief.particles = [p_with, base.copy(), base.copy()]
    s._belief.weights = [1.0, 1.0, 1.0]
    move = chess.Move.from_uci("c3d5")
    view = _build_view(base, chess.WHITE, visible_pieces=pieces)
    survivors = s._belief_veto_king_attack([move], view)
    assert survivors == [], "1/3 particles is still too much immediate king risk"


def test_belief_veto_uses_lower_risk_tolerance_for_king_moves() -> None:
    """A low-probability immediate king capture is still too risky.

    Regression for v0.7.13 g16: White's belief assigned a small probability to
    a hidden rook on f8, then moved Kg4-f3 into that rook's file. Ordinary
    discovered-check filtering tolerates minority hallucinations; voluntary
    king moves need a stricter bar because the downside is terminal.
    """
    visible = {
        chess.G4: chess.Piece(chess.KING, chess.WHITE),
        chess.A8: chess.Piece(chess.KING, chess.BLACK),
    }
    base = chess.Board.empty()
    for sq, p in visible.items():
        base.set_piece_at(sq, p)
    base.turn = chess.WHITE

    risky = base.copy()
    risky.set_piece_at(chess.F8, chess.Piece(chess.ROOK, chess.BLACK))

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    # 1/16 particles attack f3 after Kg4-f3: below the old majority threshold
    # but above the king-move risk budget.
    s._belief.particles = [risky] + [base.copy() for _ in range(15)]
    s._belief.weights = [1.0] * 16

    move = chess.Move.from_uci("g4f3")
    view = _build_view(base, chess.WHITE, visible_pieces=visible)
    survivors = s._belief_veto_king_attack([move], view)
    assert survivors == [], "king move into a plausible hidden-rook capture must be vetoed"


def test_forced_king_risk_fallback_keeps_lowest_risk_moves() -> None:
    """When all moves exceed terminal-risk budget, keep the least-bad subset.

    Regression for the v0.7.22 rung: the king-risk veto correctly detected
    danger, but an empty survivor set made main eval fall back to every legal
    move and throw away the relative risk signal.
    """
    visible = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    base = chess.Board.empty()
    for sq, p in visible.items():
        base.set_piece_at(sq, p)
    base.turn = chess.WHITE

    lower_risk = base.copy()
    lower_risk.set_piece_at(chess.F8, chess.Piece(chess.ROOK, chess.BLACK))
    higher_risk = lower_risk.copy()
    higher_risk.set_piece_at(chess.D8, chess.Piece(chess.ROOK, chess.BLACK))

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    s._belief.particles = [higher_risk, lower_risk]
    s._belief.weights = [1.0, 1.0]

    move_d1 = chess.Move.from_uci("e1d1")
    move_f1 = chess.Move.from_uci("e1f1")
    view = _build_view(base, chess.WHITE, visible_pieces=visible)

    assert s._belief_veto_king_attack([move_d1, move_f1], view) == []
    assert s._belief_lowest_king_attack_risk([move_d1, move_f1], view) == [
        move_d1
    ]


def test_forced_king_risk_fallback_prefers_near_best_non_king_move() -> None:
    """Do not voluntarily move the king for a tiny risk-model edge.

    Regression for the v0.7.23 target rerun: late king moves were selected
    because they were a few percentage points lower immediate-risk than
    non-king alternatives, but each king step created fresh hidden exposure.
    """
    visible = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.A2: chess.Piece(chess.ROOK, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    base = chess.Board.empty()
    for sq, p in visible.items():
        base.set_piece_at(sq, p)
    base.turn = chess.WHITE

    # Both candidate moves are risky. e1-f1 is slightly lower immediate risk,
    # but a2-a3 keeps the king still and is within the tiebreak band.
    king_risky = base.copy()
    king_risky.set_piece_at(chess.F8, chess.Piece(chess.ROOK, chess.BLACK))
    still_risky = base.copy()
    still_risky.set_piece_at(chess.E8, chess.Piece(chess.ROOK, chess.BLACK))

    s = _strategy()
    s.reset(perspective=chess.WHITE)
    s._belief.particles = [king_risky] + [still_risky] * 2
    s._belief.weights = [1.0, 1.0, 1.0]

    king_move = chess.Move.from_uci("e1f1")
    non_king_move = chess.Move.from_uci("a2a3")
    view = _build_view(base, chess.WHITE, visible_pieces=visible)

    assert s._belief_veto_king_attack([king_move, non_king_move], view) == []
    assert s._belief_lowest_king_attack_risk(
        [king_move, non_king_move],
        view,
        king_move_tiebreak_band=0.34,
    ) == [non_king_move]


# ============================================================================
# v0.6.1 Pattern B: safe-visible-minor-or-rook capture short-circuit.
# ============================================================================


def test_safe_visible_capture_fires_on_undefended_bishop() -> None:
    """Black knight on e3 can capture white bishop on f1; f1 isn't attacked
    by any visible white piece. Expected: short-circuit picks Nxf1."""
    pieces = {
        chess.E3: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.F1: chess.Piece(chess.BISHOP, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.A1: chess.Piece(chess.KING, chess.WHITE),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.BLACK

    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    captures = _safe_visible_minor_or_rook_captures(view)
    assert any(m.to_square == chess.F1 for m in captures), (
        f"expected Nxf1 in {[m.uci() for m in captures]}"
    )


def test_safe_visible_capture_skips_when_destination_visibly_attacked() -> None:
    """Bishop on f1 visibly defended by white king on e1 — destination is
    attacked by enemy king, so this is not a 'free' capture. Short-circuit
    must NOT fire (let main-eval decide whether the trade is worth it)."""
    pieces = {
        chess.E3: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.F1: chess.Piece(chess.BISHOP, chess.WHITE),
        chess.E1: chess.Piece(chess.KING, chess.WHITE),  # defends f1
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.BLACK

    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    captures = _safe_visible_minor_or_rook_captures(view)
    assert captures == [], (
        f"expected no safe captures (king defends f1); got {[m.uci() for m in captures]}"
    )


def test_safe_visible_capture_skips_king_as_material_attacker() -> None:
    """Regression for v0.7.2 g11: black king captured a visible knight on c7
    and was immediately captured. Generic material shortcuts should not use
    the king as the attacker for non-terminal captures."""
    board = chess.Board(
        "2kn1b2/ppN2ppp/2p5/8/4NB2/8/PPP3PP/3K3R b - - 1 18"
    )
    unsafe = chess.Move.from_uci("c8c7")
    assert unsafe in board.pseudo_legal_moves

    # Black sees its own pieces and the knight on c7, but not the bishop on f4.
    visible_pieces = {
        sq: piece
        for sq, piece in board.piece_map().items()
        if piece.color == chess.BLACK or sq == chess.C7
    }
    view = _build_view(board, chess.BLACK, visible_pieces=visible_pieces)
    assert unsafe not in _safe_visible_minor_or_rook_captures(view)

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    chosen = strategy.pick_move(view)

    assert chosen != unsafe
    assert strategy.trace_log[-1]["decision_path"] != "visible-minor-rook-capture"


def test_safe_visible_capture_vetoes_belief_defended_bad_trade() -> None:
    """Regression for v0.7.4 g14 ply 31: white auto-played Rxd8 even though
    belief worlds showed the knight was defended and the rook would be lost."""
    board = chess.Board("2rn1rk1/1p2b1pp/p3bn2/8/4P3/2P1B3/P3BPPP/3R1RK1 w - - 0 16")
    capture = chess.Move.from_uci("d1d8")
    assert capture in board.pseudo_legal_moves

    # White sees its own pieces and the visible knight on d8, but not the rook
    # on f8/c8 that belief carries as hidden defender candidates.
    visible_pieces = {
        sq: piece
        for sq, piece in board.piece_map().items()
        if piece.color == chess.WHITE or sq == chess.D8
    }
    view = _build_view(board, chess.WHITE, visible_pieces=visible_pieces)
    assert capture in _safe_visible_minor_or_rook_captures(view)

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    assert strategy._belief_veto_bad_capture_trade([capture], view) == []
    chosen = strategy.pick_move(view)

    assert strategy.trace_log[-1]["decision_path"] != "visible-minor-rook-capture"
    assert chosen != capture


def test_safe_visible_capture_shortcut_respects_queen_fog_risk() -> None:
    """Do not auto-spend the queen on a minor if belief says recapture is live.

    Regression for v0.7.26 g0 ply 55: Qd1xa4 won a visible knight, but the
    queen was immediately recapturable by a hidden rook in enough particles
    that the move should fall through to the normal decision path.
    """
    visible = {
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.D1: chess.Piece(chess.QUEEN, chess.WHITE),
        chess.A4: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.H8: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, p in visible.items():
        board.set_piece_at(sq, p)
    board.turn = chess.WHITE
    capture = chess.Move.from_uci("d1a4")
    assert capture in board.pseudo_legal_moves

    risky = board.copy()
    risky.set_piece_at(chess.A8, chess.Piece(chess.ROOK, chess.BLACK))

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [risky] + [board.copy() for _ in range(3)]
    strategy._belief.weights = [1.0] * 4
    view = _build_view(board, chess.WHITE, visible_pieces=visible)
    assert capture in _safe_visible_minor_or_rook_captures(view)

    chosen = strategy.pick_move(view)

    assert strategy.trace_log[-1]["decision_path"] != "visible-minor-rook-capture"
    assert chosen != capture


def test_queen_fog_risk_vetoes_hidden_recapture_square() -> None:
    """Regression for annotation replay g2 ply 6: black should not move the
    queen to e4 when belief carries a hidden white knight on c3 attacking e4."""
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.C3, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.D5, chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(chess.C7, chess.Piece(chess.PAWN, chess.BLACK))
    board.turn = chess.BLACK

    unsafe = chess.Move.from_uci("d5e4")
    safe_queen_move = chess.Move.from_uci("d5e6")
    non_queen_move = chess.Move.from_uci("c7c6")
    assert unsafe in board.pseudo_legal_moves
    assert safe_queen_move in board.pseudo_legal_moves
    assert non_queen_move in board.pseudo_legal_moves

    visible_pieces = {
        sq: piece for sq, piece in board.piece_map().items() if piece.color == chess.BLACK
    }
    view = _build_view(board, chess.BLACK, visible_pieces=visible_pieces)

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    filtered = strategy._belief_veto_queen_fog_risk(
        [unsafe, safe_queen_move, non_queen_move],
        view,
    )

    assert unsafe not in filtered
    assert safe_queen_move in filtered
    assert non_queen_move in filtered


def test_queen_fog_risk_vetoes_visible_enemy_attack_square() -> None:
    """Do not send the queen to a square attacked by visible enemy material."""
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A5, chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.turn = chess.WHITE

    unsafe = chess.Move.from_uci("d1a4")
    assert unsafe in board.pseudo_legal_moves
    visible_pieces = dict(board.piece_map())
    view = _build_view(board, chess.WHITE, visible_pieces=visible_pieces)

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    assert strategy._belief_veto_queen_fog_risk([unsafe], view) == []


def test_queen_fog_risk_vetoes_low_value_visible_capture_recapture() -> None:
    """Regression for v0.7.8 g13 ply 32: Qxh4 wins a pawn but loses queen."""
    board = chess.Board("r1bqk2r/p6p/npp1p3/7p/4P2P/3P4/P1P1NP2/b4K1R b kq - 1 16")
    capture = chess.Move.from_uci("d8h4")
    assert capture in board.pseudo_legal_moves

    view = _build_view(
        board,
        chess.BLACK,
        visible_pieces={
            sq: piece for sq, piece in board.piece_map().items() if piece.color == chess.BLACK or sq == chess.H4
        },
    )
    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    assert strategy._belief_veto_queen_fog_risk([capture], view) == []


def test_queen_king_pressure_prefers_safe_belief_attack() -> None:
    """After unsafe Qe4 is filtered, Qe6 should be recognized as safe pressure
    on the believed white king instead of falling through to quiet development."""
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.C3, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.D5, chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(chess.C7, chess.Piece(chess.PAWN, chess.BLACK))
    board.turn = chess.BLACK

    unsafe = chess.Move.from_uci("d5e4")
    pressure = chess.Move.from_uci("d5e6")
    quiet = chess.Move.from_uci("c7c6")
    visible_pieces = {
        sq: piece for sq, piece in board.piece_map().items() if piece.color == chess.BLACK
    }
    view = _build_view(board, chess.BLACK, visible_pieces=visible_pieces)

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    particles = []
    for sq in (chess.A2, chess.B2, chess.C2, chess.D2):
        particle = board.copy()
        particle.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
        particles.append(particle)
    strategy._belief.particles = particles
    strategy._belief.weights = [1.0] * len(particles)

    safe_moves = strategy._belief_veto_queen_fog_risk([unsafe, pressure, quiet], view)
    assert unsafe not in safe_moves
    assert strategy._belief_queen_king_pressure_moves(safe_moves, view) == [pressure]


def test_queen_king_pressure_skips_after_generic_csp() -> None:
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.D5, chess.Piece(chess.QUEEN, chess.BLACK))
    board.turn = chess.BLACK
    pressure = chess.Move.from_uci("d5e6")
    view = _build_view(
        board,
        chess.BLACK,
        visible_pieces={
            sq: piece for sq, piece in board.piece_map().items() if piece.color == chess.BLACK
        },
    )

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    particles = []
    for sq in (chess.A2, chess.B2, chess.C2, chess.D2):
        particle = board.copy()
        particle.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
        particles.append(particle)
    strategy._belief.particles = particles
    strategy._belief.weights = [1.0] * len(particles)
    strategy._pending_belief_steps["csp_reseed_stage_b"] = 1

    assert strategy._belief_queen_king_pressure_moves([pressure], view) == []


def test_castle_preferred_over_flat_material_tie() -> None:
    """Regression for v0.7.8 g12: with castling legal and no tactic, castle."""
    board = chess.Board("rn2kbnr/pp2pppp/2p5/3p4/1qP3b1/N3PN2/PP1PBPPP/R1BQK2R w KQkq - 3 6")
    castle = chess.Move.from_uci("e1g1")
    assert castle in board.pseudo_legal_moves
    view = _build_view(board, chess.WHITE)

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    assert castle in _castle_moves(view)
    chosen = strategy.pick_move(view)
    assert chosen == castle
    assert strategy.trace_log[-1]["decision_path"] == "castle"


def test_high_value_piece_save_blocks_rook_skewer() -> None:
    """Regression for v0.7.8 g12 ply 31: block believed Qe4-h1 rook threat."""
    board = chess.Board("r3k1nr/3nb1pp/1p1p1p2/pp6/4q2N/6Pb/PP1PBP1P/R1BQK2R w kq - 0 16")
    block = chess.Move.from_uci("f2f3")
    assert block in board.pseudo_legal_moves
    view = _build_view(board, chess.WHITE)
    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]
    strategy._observed_ply = 30

    saves = strategy._belief_high_value_piece_save_moves(view.own_legal_moves, view)

    assert block in saves
    chosen = strategy.pick_move(view)
    assert chosen == block
    assert strategy.trace_log[-1]["decision_path"] in (
        "belief-high-value-save",
        "belief-piece-save",
    )


def test_early_development_prefers_e_pawn_over_rook_pawn_drift() -> None:
    """Regression for v0.7.8 g13 ply 6: develop before pushing h-pawn."""
    board = chess.Board("rnbqkb1r/ppppp1pp/5n2/5P2/8/7N/PPPPPP1P/RNBQKB1R b KQkq - 2 3")
    develop = chess.Move.from_uci("e7e6")
    assert develop in board.pseudo_legal_moves
    view = _build_view(board, chess.BLACK)

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    chosen = strategy.pick_move(view)

    assert chosen == develop
    assert strategy.trace_log[-1]["decision_path"] == "early-development"


def test_advanced_minor_retreats_before_more_pawn_drift() -> None:
    """Regression for v0.7.8 g13 ply 8: pull the loose knight back to f6."""
    board = chess.Board("rnbqkb1r/ppppp1pp/8/5P2/4P1n1/7N/PPPP1P1P/RNBQKB1R b KQkq - 0 4")
    retreat = chess.Move.from_uci("g4f6")
    assert retreat in board.pseudo_legal_moves
    view = _build_view(board, chess.BLACK)

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    chosen = strategy.pick_move(view)

    assert chosen == retreat
    assert strategy.trace_log[-1]["decision_path"] in (
        "advanced-minor-retreat",
        "belief-piece-save",
    )


def test_deep_advanced_bishop_retreat_beats_castling() -> None:
    """Regression for v0.7.9 g13 ply 18: bank the rook win by saving bishop."""
    board = chess.Board("r1b1k2r/pppqnppp/n2pp3/B7/8/1P1PPP1N/P1P1B1PP/bN1Q1RK1 b kq - 1 9")
    retreat = chess.Move.from_uci("a1f6")
    assert retreat in board.pseudo_legal_moves
    view = _build_view(board, chess.BLACK)

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]
    strategy._observed_ply = 17

    chosen = strategy.pick_move(view)

    assert chosen == retreat
    assert strategy.trace_log[-1]["decision_path"] == "advanced-minor-retreat"


def test_visible_piece_save_moves_attacked_bishop_before_capture() -> None:
    """Regression for v0.7.9 g13 ply 42: save bishop attacked by visible pawn."""
    board = chess.Board("r4rk1/p5pp/2pp4/p4b2/qn4P1/N2PP3/2P1B2P/4Q1K1 b - - 0 21")
    save = chess.Move.from_uci("f5e6")
    assert save in board.pseudo_legal_moves
    view = _build_view(board, chess.BLACK)

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]
    strategy._observed_ply = 41

    chosen = strategy.pick_move(view)

    assert chosen == save
    assert strategy.trace_log[-1]["decision_path"] == "visible-piece-save"


def test_belief_piece_save_moves_attacked_knight_from_observed_capture() -> None:
    """Regression for v0.7.9 g13 ply 36: belief says pawn attacks knight."""
    board = chess.Board("r1b2rk1/p1p3pp/3pp3/p4n2/qn4P1/N2PP3/2P1B1PP/4QRK1 b - - 0 18")
    save = chess.Move.from_uci("f5e7")
    assert save in board.pseudo_legal_moves
    view = _build_view(board, chess.BLACK)

    strategy = _strategy()
    strategy.reset(perspective=chess.BLACK)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]
    strategy._observed_ply = 35

    chosen = strategy.pick_move(view)

    assert chosen != chess.Move.from_uci("c7c6")
    assert strategy.trace_log[-1]["decision_path"] == "belief-piece-save"


def test_piece_fog_risk_vetoes_minor_jump_to_pawn_defended_square() -> None:
    """Regression for v0.7.9 g14 ply 23: don't leap knight onto defended d5."""
    board = chess.Board("r1bq1rk1/1p1n1ppp/p2pp1n1/7B/Pp3N1b/1N1PP3/2P2PPP/R1BQ1RK1 w - - 2 12")
    jump = chess.Move.from_uci("f4d5")
    assert jump in board.pseudo_legal_moves
    view = _build_view(board, chess.WHITE)

    strategy = _strategy()
    strategy.reset(perspective=chess.WHITE)
    strategy._belief.particles = [board.copy(), board.copy()]
    strategy._belief.weights = [1.0, 1.0]

    filtered = strategy._belief_veto_piece_fog_risk([jump], view)

    assert filtered == []


def test_safe_visible_capture_prefers_higher_material() -> None:
    """Knight can capture either a bishop on f1 or a rook on h6, both safe.
    Rook (5) > bishop (3) — short-circuit should restrict to rook capture."""
    pieces = {
        chess.G4: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.F1: chess.Piece(chess.BISHOP, chess.WHITE),  # could be reachable from h2; skip realism
        chess.H6: chess.Piece(chess.ROOK, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.A1: chess.Piece(chess.KING, chess.WHITE),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.BLACK

    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    captures = _safe_visible_minor_or_rook_captures(view)
    capture_squares = {m.to_square for m in captures}
    # Only the rook capture should remain (max material).
    assert chess.H6 in capture_squares
    assert chess.F1 not in capture_squares


def test_safe_visible_capture_skips_pawns() -> None:
    """A visible undefended pawn capture is not auto-fired — main-eval can decide."""
    pieces = {
        chess.E3: chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.C2: chess.Piece(chess.PAWN, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
        chess.A1: chess.Piece(chess.KING, chess.WHITE),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.BLACK
    view = _build_view(board, chess.BLACK, visible_pieces=pieces)
    captures = _safe_visible_minor_or_rook_captures(view)
    assert captures == []


def test_prefer_higher_value_capture_helper() -> None:
    """Rxqueen beats Rxpawn."""
    pieces = {
        chess.A1: chess.Piece(chess.ROOK, chess.WHITE),
        chess.A8: chess.Piece(chess.QUEEN, chess.BLACK),
        chess.B1: chess.Piece(chess.PAWN, chess.BLACK),
        chess.E1: chess.Piece(chess.KING, chess.WHITE),
        chess.E8: chess.Piece(chess.KING, chess.BLACK),
    }
    board = chess.Board.empty()
    for sq, p in pieces.items():
        board.set_piece_at(sq, p)
    board.turn = chess.WHITE
    view = _build_view(board, chess.WHITE, visible_pieces=pieces)
    captures = [
        chess.Move.from_uci("a1a8"),  # Rxqueen
        chess.Move.from_uci("a1b1"),  # Rxpawn
    ]
    filtered = _prefer_higher_value_capture(captures, view)
    assert filtered == [chess.Move.from_uci("a1a8")]
