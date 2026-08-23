import chess

from fow_chess.belief import (
    BeliefState,
    RepairDiagnostics,
    _csp_reseed,
    _repair_diagnostics,
    _repair_passes_strict_reachability,
    _repair_recovery_source_limit,
    _repair_supplement_source_limit,
    _select_repair_candidates,
    _select_repair_recovery_sources,
    _repair_supplement_limit,
    _select_repair_supplement_sources,
)
from fow_chess.move_priors import uniform_prior
from fow_chess.observation import Observation, observation_from_transition
from fow_chess.visibility import visible_piece_map, visible_squares


def test_initial_belief_holds_a_single_seeded_particle() -> None:
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=64,
    )

    assert len(belief.particles) == 1
    assert belief.particles[0].fen() == chess.Board().fen()


def test_own_move_advances_every_particle() -> None:
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=64,
    )
    move = chess.Move.from_uci("e2e4")

    belief.update_after_own_move(move)

    expected = chess.Board()
    expected.push(move)
    assert len(belief.particles) == 1
    assert belief.particles[0].fen() == expected.fen()


def test_repair_diagnostics_flags_king_teleport() -> None:
    before = chess.Board.empty()
    before.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    before.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))

    after = chess.Board.empty()
    after.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    after.set_piece_at(chess.E4, chess.Piece(chess.KING, chess.BLACK))

    diag = _repair_diagnostics(before, after, visibility_set=set())

    assert diag.moved_piece_count == 1
    assert diag.max_piece_distance == 4
    assert diag.long_move_count == 1
    assert diag.teleport_like_count == 1
    assert diag.cost >= 40
    assert diag.worst_piece == "k"
    assert diag.worst_from == "h8"
    assert diag.worst_to == "e4"
    assert diag.worst_distance == 4
    assert diag.worst_one_move_legal is False
    assert diag.strict_unreachable_count == 1
    assert _repair_passes_strict_reachability(diag) is False


def test_repair_diagnostics_treats_queen_teleport_as_strict_unreachable() -> None:
    before = chess.Board.empty()
    before.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    before.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    before.set_piece_at(chess.H4, chess.Piece(chess.QUEEN, chess.BLACK))
    before.set_piece_at(chess.D6, chess.Piece(chess.BISHOP, chess.BLACK))

    after = chess.Board.empty()
    after.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    after.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    after.set_piece_at(chess.B8, chess.Piece(chess.QUEEN, chess.BLACK))
    after.set_piece_at(chess.D6, chess.Piece(chess.BISHOP, chess.BLACK))

    diag = _repair_diagnostics(before, after, visibility_set=set())

    assert diag.teleport_like_count == 1
    assert diag.worst_piece == "q"
    assert diag.worst_from == "h4"
    assert diag.worst_to == "b8"
    assert diag.worst_one_move_legal is False
    assert diag.strict_unreachable_count == 1
    assert _repair_passes_strict_reachability(diag) is False


def test_repair_supplement_limit_tracks_diversity_deficit() -> None:
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))

    assert _repair_supplement_limit([board], target_n=256) == 62

    particles = []
    for idx in range(32):
        particle = board.copy()
        particle.set_piece_at(idx, chess.Piece(chess.PAWN, chess.WHITE))
        particles.append(particle)

    assert _repair_supplement_limit(particles, target_n=256) == 0


def test_repair_supplement_source_limit_bounds_expensive_repair_pool() -> None:
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))

    assert _repair_supplement_source_limit([board], target_n=256) == 248


def test_select_repair_supplement_sources_prefers_hard_near_high_weight() -> None:
    facts = BeliefState.initial(chess.WHITE, uniform_prior)._hard_facts(
        Observation(visibility_mask=set(), visible_pieces={})
    )
    prev = chess.Board.empty()

    hard_near = prev.copy()
    hard_near.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    hard_near.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    soft_only = hard_near.copy()
    soft_only.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.BLACK))
    duplicate_lower_weight = hard_near.copy()

    expanded = [
        (prev, soft_only, 0.9, False, False, True, facts),
        (prev, hard_near, 0.2, False, True, True, facts),
        (prev, duplicate_lower_weight, 0.1, False, True, True, facts),
    ]

    selected = _select_repair_supplement_sources(expanded, set(), limit=2)

    assert [board.fen() for _, board, *_ in selected] == [
        hard_near.fen(),
        soft_only.fen(),
    ]


def test_repair_recovery_sources_bound_full_stage_b_repair() -> None:
    facts = BeliefState.initial(chess.WHITE, uniform_prior)._hard_facts(
        Observation(visibility_mask=set(), visible_pieces={})
    )
    prev = chess.Board.empty()
    expanded = []
    for idx in range(10):
        board = prev.copy()
        board.set_piece_at(
            chess.square(idx % 8, idx // 8), chess.Piece(chess.PAWN, chess.BLACK)
        )
        expanded.append((prev, board, float(idx), False, False, True, facts))

    selected = _select_repair_recovery_sources(expanded, limit=3)

    assert _repair_recovery_source_limit(16) == 128
    assert [weight for _, _, weight, *_ in selected] == [9.0, 8.0, 7.0]


def test_select_repair_candidates_dedupes_and_prefers_lower_cost() -> None:
    board_a = chess.Board.empty()
    board_a.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board_a.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board_b = board_a.copy()
    board_b.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))

    bad_duplicate = RepairDiagnostics(
        cost=50,
        moved_piece_count=1,
        max_piece_distance=1,
        long_move_count=0,
        teleport_like_count=0,
        forced_visible_square_count=5,
        unpaired_added_count=1,
        unpaired_removed_count=1,
    )
    good_duplicate = RepairDiagnostics(
        cost=10,
        moved_piece_count=0,
        max_piece_distance=0,
        long_move_count=0,
        teleport_like_count=0,
        forced_visible_square_count=1,
        unpaired_added_count=0,
        unpaired_removed_count=0,
    )
    other = RepairDiagnostics(
        cost=20,
        moved_piece_count=0,
        max_piece_distance=0,
        long_move_count=0,
        teleport_like_count=0,
        forced_visible_square_count=2,
        unpaired_added_count=0,
        unpaired_removed_count=0,
    )

    selected = _select_repair_candidates(
        [
            (board_a, 1.0, bad_duplicate),
            (board_a, 0.1, good_duplicate),
            (board_b, 1.0, other),
        ],
        target_n=1,
    )

    assert len(selected) == 1
    assert selected[0][2].cost == 10


def test_stage_b_uses_checkpoint_repair_before_generic_csp() -> None:
    import random

    checkpoint = chess.Board.empty()
    checkpoint.turn = chess.BLACK
    checkpoint.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    checkpoint.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    checkpoint.set_piece_at(chess.B6, chess.Piece(chess.PAWN, chess.BLACK))

    stale = chess.Board.empty()
    stale.turn = chess.BLACK
    stale.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))

    truth = checkpoint.copy()
    truth.turn = chess.WHITE
    obs = Observation(
        visibility_mask=visible_squares(truth, chess.WHITE),
        visible_pieces=visible_piece_map(truth, chess.WHITE),
    )

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {chess.KING: 1, chess.PAWN: 1}
    belief.opp_bishop_colors_remaining = {True: 0, False: 0}
    belief.checkpoint_particles = [checkpoint]
    belief.checkpoint_weights = [1.0]
    belief.checkpoint_update_index = 0

    belief.update_after_opp_move(obs)

    assert belief.last_checkpoint_repair_fired == 1
    assert belief.last_checkpoint_repair_count == 1
    assert belief.last_csp_reseed_fired == 0
    assert belief.particles
    assert all(
        particle.piece_at(chess.H8) == chess.Piece(chess.KING, chess.BLACK)
        and particle.piece_at(chess.B6) == chess.Piece(chess.PAWN, chess.BLACK)
        for particle in belief.particles
    )


def test_canonical_truth_survives_opp_move_update() -> None:
    seed = chess.Board()
    seed.push_uci("e2e4")

    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=128,
        start_board=seed,
    )

    truth = seed.copy()
    truth.push_uci("d7d5")
    obs = observation_from_transition(seed, truth, chess.WHITE)

    belief.update_after_opp_move(obs)

    assert not belief.collapsed()
    truth_fen = truth.fen()
    assert any(p.fen() == truth_fen for p in belief.particles)


def test_observation_filter_rejects_inconsistent_particle_branches() -> None:
    seed = chess.Board()
    seed.push_uci("e2e4")

    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=256,
        start_board=seed,
    )

    truth = seed.copy()
    truth.push_uci("d7d5")
    obs = observation_from_transition(seed, truth, chess.WHITE)

    belief.update_after_opp_move(obs)

    truth_visibility = visible_squares(truth, chess.WHITE)
    truth_pieces = visible_piece_map(truth, chess.WHITE)
    for particle in belief.particles:
        assert visible_squares(particle, chess.WHITE) == truth_visibility
        assert visible_piece_map(particle, chess.WHITE) == truth_pieces


def test_opp_move_update_reseeds_when_no_particle_can_expand() -> None:
    """Stage B must not leave belief empty when every particle has no legal
    opponent expansion; current hard observations are still available."""
    stale = chess.Board.empty()
    stale.turn = chess.BLACK
    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
        particles=[stale],
        weights=[1.0],
    )
    own_king = chess.Piece(chess.KING, chess.WHITE)
    obs = Observation(
        visibility_mask=chess.SquareSet([chess.E1]),
        visible_pieces={chess.E1: own_king},
    )

    belief.update_after_opp_move(obs)

    assert not belief.collapsed()
    assert belief.last_csp_reseed_fired == 1
    assert belief.last_csp_reseed_count == belief.target_n
    assert len({particle.fen() for particle in belief.particles}) > 1
    assert belief.marginal_piece_at(chess.E1) == {own_king: 1.0}


def test_marginals_sum_to_one_when_belief_is_alive() -> None:
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=16,
    )

    marginal = belief.marginal_piece_at(chess.E8)

    assert marginal
    total = sum(marginal.values())
    assert abs(total - 1.0) < 1e-9


def test_marginal_piece_field_exposes_sparse_piece_distribution() -> None:
    knight_f3 = chess.Board()
    knight_f3.remove_piece_at(chess.G1)
    knight_f3.set_piece_at(chess.F3, chess.Piece(chess.KNIGHT, chess.WHITE))
    knight_g1 = chess.Board()
    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        particles=[knight_f3, knight_g1],
        weights=[1.0, 1.0],
    )

    sparse = belief.marginal_piece_field(min_prob=0.05)

    assert sparse[chess.F3] == [
        (chess.Piece(chess.KNIGHT, chess.WHITE), 0.5),
        (None, 0.5),
    ]
    assert sparse[chess.G1] == [
        (chess.Piece(chess.KNIGHT, chess.WHITE), 0.5),
        (None, 0.5),
    ]


def test_top_k_clusters_are_weighted_and_deterministically_ordered() -> None:
    e4 = chess.Board()
    e4.push_uci("e2e4")
    d4 = chess.Board()
    d4.push_uci("d2d4")
    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        particles=[e4, d4],
        weights=[1.0, 1.0],
    )

    clusters = belief.top_k_clusters(k=2)

    assert clusters == sorted(clusters, key=lambda item: (-item[1], item[0]))
    assert clusters[0][1] == 0.5
    assert clusters[0][2] == 1


def test_particle_weight_profile_separates_posterior_from_appearance() -> None:
    e4 = chess.Board()
    e4.push_uci("e2e4")
    d4 = chess.Board()
    d4.push_uci("d2d4")
    d4_duplicate = d4.copy()
    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        particles=[e4, d4, d4_duplicate],
        weights=[8.0, 1.0, 1.0],
    )

    profile = belief.particle_weight_profile(k=2)

    assert profile["summary"]["particle_count"] == 3
    assert profile["summary"]["unique_count"] == 2
    assert profile["summary"]["posterior_top1_mass"] == 0.8
    assert profile["summary"]["appearance_top1_mass"] == 2 / 3
    clusters = profile["clusters"]
    assert clusters[0]["fen"] == e4.fen()
    assert clusters[0]["posterior_mass"] == 0.8
    assert clusters[0]["appearance_mass"] == 1 / 3
    assert clusters[0]["posterior_rank"] == 1
    assert clusters[0]["appearance_rank"] == 2
    assert clusters[1]["fen"] == d4.fen()
    assert clusters[1]["posterior_mass"] == 0.2
    assert clusters[1]["appearance_mass"] == 2 / 3
    assert clusters[1]["posterior_rank"] == 2
    assert clusters[1]["appearance_rank"] == 1


def test_stage_a_repairs_when_post_own_observation_kills_all_particles() -> None:
    """Post-own-move visible pieces are hard facts.

    Regression for g13 ply 34 from v0.7.0 hardobs rung 2: black moved Bc8-g4,
    which revealed a white rook on d1 and an empty f3 square. Stage A used to
    roll back to pushed particles that still believed d1 could be a queen.
    """
    import random

    stale = chess.Board.empty()
    stale.turn = chess.BLACK
    stale.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    stale.set_piece_at(chess.C8, chess.Piece(chess.BISHOP, chess.BLACK))
    stale.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    stale.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
    stale.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))

    truth_pre = stale.copy()
    truth_pre.set_piece_at(chess.D1, chess.Piece(chess.ROOK, chess.WHITE))
    truth_post = truth_pre.copy()
    move = chess.Move.from_uci("c8g4")
    truth_post.push(move)
    obs = observation_from_transition(truth_pre, truth_post, chess.BLACK)
    assert obs.visible_pieces[chess.D1] == chess.Piece(chess.ROOK, chess.WHITE)
    assert chess.F3 in obs.visibility_mask
    assert chess.F3 not in obs.visible_pieces

    belief = BeliefState(
        perspective=chess.BLACK,
        move_prior=uniform_prior,
        target_n=16,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {chess.KING: 1, chess.ROOK: 1, chess.PAWN: 1}
    belief.opp_bishop_colors_remaining = {True: 0, False: 0}

    belief.update_after_own_move(move, obs)

    assert belief.last_csp_reseed_fired == 0
    assert belief.particles
    assert all(
        particle.piece_at(chess.D1) == chess.Piece(chess.ROOK, chess.WHITE)
        for particle in belief.particles
    )
    assert all(particle.piece_at(chess.F3) is None for particle in belief.particles)
    assert all(
        particle.piece_at(chess.A2) == chess.Piece(chess.PAWN, chess.WHITE)
        for particle in belief.particles
    )
    assert all(
        visible_piece_map(particle, chess.BLACK) == obs.visible_pieces
        for particle in belief.particles
    )


def test_initial_opp_remaining_counts_match_standard_start() -> None:
    """Standard chess start: 8 pawns, 2 of N/B/R, 1 Q, 1 K for the opponent."""
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
    )
    assert belief.opp_remaining_counts == {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
        chess.KING: 1,
    }


def test_register_capture_decrements_count() -> None:
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
    )
    belief.register_capture(chess.KNIGHT)
    assert belief.opp_remaining_counts[chess.KNIGHT] == 1
    belief.register_capture(chess.KNIGHT)
    assert belief.opp_remaining_counts[chess.KNIGHT] == 0
    # Floor at zero, never negative.
    belief.register_capture(chess.KNIGHT)
    assert belief.opp_remaining_counts[chess.KNIGHT] == 0


def test_register_capture_clears_prior_piece_fact_on_captured_square() -> None:
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
    )
    belief.hard_opp_occupancy_squares.add(chess.C5)
    belief.hard_opp_piece_facts[chess.C5] = chess.Piece(chess.PAWN, chess.BLACK)

    belief.register_capture(chess.PAWN, chess.C5)

    assert chess.C5 not in belief.hard_opp_occupancy_squares
    assert chess.C5 not in belief.hard_opp_piece_facts


def test_register_bishop_capture_decrements_matching_square_color() -> None:
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
    )

    assert belief.opp_bishop_colors_remaining == {True: 1, False: 1}

    belief.register_capture(chess.BISHOP, chess.C8)

    assert belief.opp_remaining_counts[chess.BISHOP] == 1
    assert belief.opp_bishop_colors_remaining == {True: 0, False: 1}


def test_stage_b_constraint_prunes_phantom_pieces() -> None:
    """If we've captured an opp knight, no surviving particle should have 2 opp knights.

    Hand-construct two particles to exercise pruning: one canonical (2 black
    knights — phantom under our bound) and one with one knight already removed
    (consistent). After Stage B, only the 1-knight particle's expansions can
    survive primary filtering.
    """
    import random
    seed_canonical = chess.Board()
    seed_canonical.push_uci("e2e4")  # black to move

    seed_one_knight = seed_canonical.copy()
    seed_one_knight.remove_piece_at(chess.B8)  # opp now has 1 knight

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=64,
        particles=[seed_canonical, seed_one_knight],
        weights=[1.0, 1.0],
        rng=random.Random(0),
    )
    # Bound says opp has only 1 knight remaining (we captured one).
    belief.opp_remaining_counts[chess.KNIGHT] = 1

    # Construct an observation consistent with the canonical line e2e4 e7e5.
    truth = seed_canonical.copy()
    truth.push_uci("e7e5")
    obs = observation_from_transition(seed_canonical, truth, chess.WHITE)
    belief.update_after_opp_move(obs)

    for particle in belief.particles:
        knight_count = sum(
            1
            for p in particle.piece_map().values()
            if p.color == chess.BLACK and p.piece_type == chess.KNIGHT
        )
        assert knight_count <= 1


def test_stage_b_count_constraint_allows_promotion_excess() -> None:
    """A promoted queen is legal when it is compensated by a missing pawn.

    Regression for v0.7.6 rung2 game 13 ply 76/77: black promoted after the
    original queen had been captured. The old count constraint rejected every
    expanded particle because queen count exceeded the captured-queen bound,
    even though black had one fewer pawn.
    """
    seed = chess.Board.empty()
    seed.turn = chess.BLACK
    seed.set_piece_at(chess.G2, chess.Piece(chess.KING, chess.WHITE))
    seed.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    seed.set_piece_at(chess.G7, chess.Piece(chess.KING, chess.BLACK))
    seed.set_piece_at(chess.D2, chess.Piece(chess.PAWN, chess.BLACK))

    truth = seed.copy()
    truth.push(chess.Move.from_uci("d2d1q"))
    obs = observation_from_transition(seed, truth, chess.WHITE)

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=16,
        particles=[seed],
        weights=[1.0],
    )
    belief.opp_remaining_counts = {
        chess.KING: 1,
        chess.QUEEN: 0,
        chess.ROOK: 0,
        chess.BISHOP: 0,
        chess.KNIGHT: 0,
        chess.PAWN: 1,
    }
    belief.opp_bishop_colors_remaining = {True: 0, False: 0}

    belief.update_after_opp_move(obs)

    assert belief.last_csp_reseed_fired == 0
    assert any(
        particle.piece_at(chess.D1) == chess.Piece(chess.QUEEN, chess.BLACK)
        for particle in belief.particles
    )


def test_count_constraint_rejects_missing_remaining_material() -> None:
    """Remaining material is an exact ledger, not only an upper bound.

    Promotions may move mass from pawn counts to non-pawn pieces, but a particle
    cannot simply omit known-remaining opponent material.
    """
    import random

    stale = chess.Board.empty()
    stale.turn = chess.BLACK
    stale.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    stale.set_piece_at(chess.F2, chess.Piece(chess.KING, chess.WHITE))

    truth = stale.copy()
    truth.set_piece_at(chess.C3, chess.Piece(chess.PAWN, chess.WHITE))
    truth.set_piece_at(chess.C4, chess.Piece(chess.PAWN, chess.WHITE))
    obs = Observation(
        visibility_mask=visible_squares(truth, chess.BLACK),
        visible_pieces=visible_piece_map(truth, chess.BLACK),
    )

    belief = BeliefState(
        perspective=chess.BLACK,
        move_prior=uniform_prior,
        target_n=16,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {chess.KING: 1, chess.PAWN: 2}
    belief.opp_bishop_colors_remaining = {True: 0, False: 0}

    belief.update_after_opp_move(obs)

    assert belief.last_csp_reseed_fired == 0
    assert belief.last_repair_fired == 1
    assert belief.particles
    for particle in belief.particles:
        white_counts = {
            pt: sum(
                1
                for piece in particle.piece_map().values()
                if piece.color == chess.WHITE and piece.piece_type == pt
            )
            for pt in (chess.KING, chess.PAWN)
        }
        assert white_counts[chess.KING] == 1
        assert white_counts[chess.PAWN] == 2


def test_stage_a_reseed_when_step1_wipes_all_particles() -> None:
    """v0.7.0: when no particle has my_move pseudo-legal, reseed from the
    post-move observation rather than collapsing to zero particles.

    Construct a contrived case: belief has one particle where the move
    isn't pseudo-legal (a piece is missing from from_square in the
    particle). With CSP reseed enabled, post-update belief should have
    particles reflecting the visible post-move state.
    """
    import random
    # Belief seeded with an empty board (no piece on e2). The move e2e4
    # therefore has no pseudo-legal support.
    empty_board = chess.Board.empty()
    empty_board.turn = chess.WHITE
    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=4,
        particles=[empty_board],
        weights=[1.0],
        rng=random.Random(0),
    )
    move = chess.Move.from_uci("e2e4")
    # Build a post-move observation showing pawn on e4 (the canonical post-move
    # state from a visibility perspective).
    truth_pre = chess.Board()
    truth_post = truth_pre.copy()
    truth_post.push(move)
    obs = observation_from_transition(truth_pre, truth_post, chess.WHITE)
    belief.update_after_own_move(move, obs)
    assert len(belief.particles) == belief.target_n
    assert belief.last_csp_reseed_fired == 1
    assert belief.last_csp_reseed_count == belief.target_n
    # Every reseeded particle reflects the visible post-move state.
    assert all(
        particle.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
        for particle in belief.particles
    )
    assert all(particle.turn == chess.BLACK for particle in belief.particles)
    for particle in belief.particles:
        black_counts: dict[chess.PieceType, int] = {}
        bishop_colors = {True: 0, False: 0}
        for sq, piece in particle.piece_map().items():
            if piece.color != chess.BLACK:
                continue
            black_counts[piece.piece_type] = black_counts.get(piece.piece_type, 0) + 1
            if piece.piece_type == chess.BISHOP:
                bishop_colors[(chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1] += 1
            if piece.piece_type == chess.PAWN:
                assert chess.square_rank(sq) not in (0, 7)
        assert black_counts == belief.opp_remaining_counts
        assert bishop_colors == belief.opp_bishop_colors_remaining


def test_stage_b_csp_reseed_uses_post_opp_side_to_move() -> None:
    seed = chess.Board()
    seed.push_uci("e2e4")
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
        start_board=seed,
    )
    # Make every ordinary expansion violate the count constraint so Stage B's
    # old all-expansions rollback is replaced by CSP reseed.
    belief.opp_remaining_counts[chess.KNIGHT] = 0

    truth = seed.copy()
    truth.push_uci("d7d5")
    obs = observation_from_transition(seed, truth, chess.WHITE)
    belief.update_after_opp_move(obs)

    assert belief.last_csp_reseed_fired == 1
    assert belief.last_csp_reseed_count == belief.target_n
    assert all(particle.turn == chess.WHITE for particle in belief.particles)
    assert all(
        not any(
            piece.color == chess.BLACK and piece.piece_type == chess.KNIGHT
            for piece in particle.piece_map().values()
        )
        for particle in belief.particles
    )


def test_stage_b_reseeds_when_own_piece_capture_observation_would_be_relaxed() -> None:
    """Own-piece captures are hard observations, not visibility noise.

    Regression for game 0008 ply 22 from v0.7.0 mirror: black played Re8xe2,
    capturing a visible white bishop. White's belief had no particle where that
    rook move matched the observation, so the old constraint-only fallback kept
    particles with the white bishop still on e2.
    """
    import random
    from fow_chess.observation import Observation

    stale = chess.Board.empty()
    stale.turn = chess.BLACK
    stale.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    stale.set_piece_at(chess.E2, chess.Piece(chess.BISHOP, chess.WHITE))
    stale.set_piece_at(chess.A8, chess.Piece(chess.ROOK, chess.BLACK))
    stale.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=8,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )

    obs = Observation(
        visibility_mask=chess.SquareSet([chess.E1, chess.E2]),
        visible_pieces={
            chess.E1: chess.Piece(chess.KING, chess.WHITE),
            chess.E2: chess.Piece(chess.ROOK, chess.BLACK),
        },
        own_capture_square=chess.E2,
    )

    belief.update_after_opp_move(obs)

    assert belief.last_csp_reseed_fired == 1
    assert belief.last_csp_reseed_count > 0
    assert all(particle.piece_at(chess.E2) == chess.Piece(chess.ROOK, chess.BLACK)
               for particle in belief.particles)
    assert all(particle.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE)
               for particle in belief.particles)


def test_stage_b_does_not_relax_visible_opponent_piece() -> None:
    """Visible opponent pieces are hard facts, not soft visibility noise.

    Regression for q10 moveselect-check ply 54/59: white should have seen a
    black pawn on b6, but Stage B's old constraint-only fallback kept particles
    that did not contain that visible pawn.
    """
    import random

    stale = chess.Board.empty()
    stale.turn = chess.BLACK
    stale.set_piece_at(chess.B5, chess.Piece(chess.KING, chess.WHITE))
    stale.set_piece_at(chess.D1, chess.Piece(chess.ROOK, chess.WHITE))
    stale.set_piece_at(chess.A6, chess.Piece(chess.PAWN, chess.WHITE))
    stale.set_piece_at(chess.B4, chess.Piece(chess.PAWN, chess.WHITE))
    stale.set_piece_at(chess.E7, chess.Piece(chess.KING, chess.BLACK))
    stale.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.BLACK))

    truth_pre = stale.copy()
    truth_pre.set_piece_at(chess.B7, chess.Piece(chess.PAWN, chess.BLACK))
    truth_post = truth_pre.copy()
    truth_post.push(chess.Move.from_uci("b7b6"))
    obs = observation_from_transition(truth_pre, truth_post, chess.WHITE)
    assert obs.visible_pieces[chess.B6] == chess.Piece(chess.PAWN, chess.BLACK)

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=16,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {chess.KING: 1, chess.PAWN: 2}
    belief.opp_bishop_colors_remaining = {True: 0, False: 0}

    belief.update_after_opp_move(obs)

    assert belief.last_csp_reseed_fired == 0
    assert belief.last_repair_fired == 1
    assert belief.last_repair_count > 0
    assert belief.particles
    assert all(
        particle.piece_at(chess.B6) == chess.Piece(chess.PAWN, chess.BLACK)
        for particle in belief.particles
    )


def test_stage_b_repairs_hidden_capture_landing_square() -> None:
    """If our piece is captured in fog, the capturer occupies that square.

    Regression for v0.7.4 rung2 game 13 plies 43/45: black knew its pawn on c7
    disappeared, but belief did not put any white piece on c7 afterward. The
    player may not know the capturer type, but ordinary captures still give a
    hard occupancy fact for the captured square.
    """
    import random

    stale = chess.Board.empty()
    stale.turn = chess.WHITE
    stale.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    stale.set_piece_at(chess.C7, chess.Piece(chess.PAWN, chess.BLACK))
    stale.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))

    truth_post = chess.Board.empty()
    truth_post.turn = chess.BLACK
    truth_post.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    truth_post.set_piece_at(chess.C7, chess.Piece(chess.ROOK, chess.WHITE))

    obs = Observation(
        visibility_mask=visible_squares(truth_post, chess.BLACK),
        visible_pieces=visible_piece_map(truth_post, chess.BLACK),
        own_capture_square=chess.C7,
        opp_capture_landing_square=chess.C7,
    )

    belief = BeliefState(
        perspective=chess.BLACK,
        move_prior=uniform_prior,
        target_n=16,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {chess.ROOK: 1}
    belief.opp_bishop_colors_remaining = {True: 0, False: 0}

    belief.update_after_opp_move(obs)

    assert belief.last_repair_fired == 1
    assert all(
        (piece := particle.piece_at(chess.C7)) is not None
        and piece.color == chess.WHITE
        for particle in belief.particles
    )
    assert abs(
        belief.marginal_piece_at(chess.C7)[chess.Piece(chess.ROOK, chess.WHITE)] - 1.0
    ) < 1e-9


def test_stage_b_repair_preserves_forced_visible_source_capture_identity() -> None:
    """A visible source vacating into a hidden capture fixes capturer identity.

    Regression for v0.7.11 rung2 game 14 ply 28: white had seen a black pawn on
    d5. After black captured the white pawn on e4, d5 was still visible and
    empty while e4 was hidden. Belief repair must infer d5xe4 as a black pawn,
    not random-fill e4 with another hidden opponent piece.
    """
    import random

    truth_pre = chess.Board.empty()
    truth_pre.turn = chess.BLACK
    truth_pre.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
    truth_pre.set_piece_at(chess.D1, chess.Piece(chess.ROOK, chess.WHITE))
    truth_pre.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    truth_pre.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    truth_pre.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.BLACK))

    truth_post = truth_pre.copy()
    truth_post.push(chess.Move.from_uci("d5e4"))
    obs = observation_from_transition(truth_pre, truth_post, chess.WHITE)
    assert chess.D5 in obs.visibility_mask
    assert chess.D5 not in obs.visible_pieces
    assert chess.E4 not in obs.visibility_mask
    assert obs.opp_capture_landing_square == chess.E4

    stale = truth_pre.copy()
    # Extra hidden capturer candidate that can also capture e4, plus a stale
    # piece on a visible-empty square to force the Stage-B repair path. Before
    # the forced-source rule, repair could keep the rook on e4 and diffuse the
    # marginal away from the known d5 pawn identity.
    stale.set_piece_at(chess.E8, chess.Piece(chess.ROOK, chess.BLACK))
    stale.set_piece_at(chess.D6, chess.Piece(chess.BISHOP, chess.BLACK))

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=32,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {
        chess.KING: 1,
        chess.ROOK: 1,
        chess.BISHOP: 1,
        chess.PAWN: 1,
    }
    belief.opp_bishop_colors_remaining = {True: 0, False: 1}

    belief.update_after_opp_move(obs)

    assert belief.last_repair_fired == 1
    assert belief.last_csp_reseed_fired == 0
    assert abs(
        belief.marginal_piece_at(chess.E4)[chess.Piece(chess.PAWN, chess.BLACK)] - 1.0
    ) < 1e-9
    assert all(particle.piece_at(chess.D5) is None for particle in belief.particles)
    assert all(
        visible_squares(particle, chess.WHITE) == obs.visibility_mask
        for particle in belief.particles
    )


def test_stage_b_does_not_relax_visible_empty_square() -> None:
    """Visible empty squares are hard facts too.

    Regression for v0.7.1-g12-check: black's belief kept a white bishop on g5
    even though black's current observation saw g5 as empty. The old hard-
    observation fallback checked visible pieces but allowed pieces to remain on
    squares that were visible-empty in the true observation when the particle's
    own visibility mask differed.
    """
    import random

    truth_pre = chess.Board(
        "2kr1bnr/ppp2ppp/4p3/8/8/P2b1N2/1P2NPPP/R1B2RK1 w - - 1 12"
    )
    truth_post = truth_pre.copy()
    truth_post.push(chess.Move.from_uci("f3e5"))
    obs = observation_from_transition(truth_pre, truth_post, chess.BLACK)
    assert chess.G5 in obs.visibility_mask
    assert chess.G5 not in obs.visible_pieces

    stale = truth_pre.copy()
    stale.set_piece_at(chess.G5, chess.Piece(chess.BISHOP, chess.WHITE))

    belief = BeliefState(
        perspective=chess.BLACK,
        move_prior=uniform_prior,
        target_n=16,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {
        chess.KING: 1,
        chess.QUEEN: 0,
        chess.ROOK: 2,
        chess.BISHOP: 2,
        chess.KNIGHT: 2,
        chess.PAWN: 5,
    }
    belief.opp_bishop_colors_remaining = {True: 1, False: 1}

    belief.update_after_opp_move(obs)

    assert all(particle.piece_at(chess.G5) is None for particle in belief.particles)


def test_csp_reseed_preserves_pawn_blocker_from_move_affordance() -> None:
    """If an own pawn cannot push, CSP reseed must infer a hidden blocker.

    The square directly in front of a pawn is visible when the push is pseudo-
    legal. If it is not visible and no own piece occupies it, a hidden opponent
    piece must be blocking the pawn. Generic random-fill reseed used to miss
    this and assign zero belief to the blocker square.
    """
    import random

    truth = chess.Board.empty()
    truth.turn = chess.WHITE
    truth.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
    truth.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
    truth.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
    truth.set_piece_at(chess.E3, chess.Piece(chess.PAWN, chess.BLACK))
    obs = Observation(
        visibility_mask=visible_squares(truth, chess.WHITE),
        visible_pieces=visible_piece_map(truth, chess.WHITE),
    )

    particles, _ = _csp_reseed(
        obs,
        opp_remaining_counts={chess.KING: 1, chess.PAWN: 1},
        opp_bishop_colors_remaining={True: 0, False: 0},
        perspective=chess.WHITE,
        side_to_move=chess.WHITE,
        n=16,
        rng=random.Random(0),
    )

    assert len(particles) == 16
    for particle in particles:
        blocker = particle.piece_at(chess.E3)
        assert blocker is not None
        assert blocker.color == chess.BLACK
        assert visible_squares(particle, chess.WHITE) == obs.visibility_mask
        assert visible_piece_map(particle, chess.WHITE) == obs.visible_pieces


def test_stage_a_repair_preserves_prior_hidden_capture_landing() -> None:
    """Our own move cannot erase a prior hidden opponent occupancy fact.

    Regression for v0.7.12 rung2 game 16: black captured White's h-pawn on h4.
    White correctly knew h4 contained a black piece. Two plies later, White's
    b2-b4 revealed the black queen on a5; Stage-A repair fixed the queen
    identity but trimmed the hidden h4 piece out of most particles, producing
    a 73% empty belief on a square that should still be occupied.
    """
    import random

    truth_pre = chess.Board(
        "1nb1rkn1/p5p1/3ppp2/qp2P2p/2p3Pb/2PP4/PP1NNP2/R1B1QRKB w - - 0 18"
    )
    truth_post = truth_pre.copy()
    move = chess.Move.from_uci("b2b4")
    truth_post.push(move)
    obs = observation_from_transition(truth_pre, truth_post, chess.WHITE)
    assert chess.H4 not in obs.visibility_mask
    assert obs.visible_pieces[chess.A5] == chess.Piece(chess.QUEEN, chess.BLACK)

    cluster_fens = [
        "1nb4r/3k2b1/1n1pppp1/pp2P2p/2p3Pq/2PP4/PP1NNP2/R1B1QRKB w - - 0 18",
        "1nbq3r/3k4/1n1pppp1/pp2P2p/2p3Pb/2PP4/PP1NNP2/R1B1QRKB w - - 0 18",
        "1n3b1r/3k4/bn1pppp1/pp2P2p/2p3Pq/2PP4/PP1NNP2/R1B1QRKB w - - 0 18",
        "1nb2b1r/3k4/1n1pppp1/pp2P2p/2p3Pq/2PP4/PP1NNP2/R1B1QRKB w - - 0 18",
        "2b3nr/p2k2p1/nb1ppp2/1p2P2p/2p3Pq/2PP4/PP1NNP2/R1B1QRKB w - - 0 18",
    ]
    particles = [chess.Board(fen) for fen in cluster_fens]

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=64,
        particles=particles,
        weights=[1.0] * len(particles),
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {
        chess.KING: 1,
        chess.QUEEN: 1,
        chess.ROOK: 1,
        chess.BISHOP: 2,
        chess.KNIGHT: 2,
        chess.PAWN: 8,
    }
    belief.opp_bishop_colors_remaining = {True: 1, False: 1}
    belief.hard_opp_occupancy_squares.add(chess.H4)

    belief.update_after_own_move(move, obs)

    assert belief.last_repair_fired == 1
    assert belief.last_csp_reseed_fired == 0
    assert chess.H4 in belief.hard_opp_occupancy_squares
    assert all(
        (piece := particle.piece_at(chess.H4)) is not None
        and piece.color == chess.BLACK
        for particle in belief.particles
    )
    assert all(
        visible_squares(particle, chess.WHITE) == obs.visibility_mask
        for particle in belief.particles
    )
    assert all(
        visible_piece_map(particle, chess.WHITE) == obs.visible_pieces
        for particle in belief.particles
    )


def test_visible_opp_piece_enters_piece_fact_ledger() -> None:
    """Directly seen opponent pieces are strict facts, not only marginals."""
    truth_pre = chess.Board.empty()
    truth_pre.turn = chess.WHITE
    truth_pre.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    truth_pre.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))
    truth_pre.set_piece_at(chess.E1, chess.Piece(chess.ROOK, chess.WHITE))
    truth_pre.set_piece_at(chess.E4, chess.Piece(chess.ROOK, chess.BLACK))
    truth_pre.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))

    truth_post = truth_pre.copy()
    move = chess.Move.from_uci("a2a3")
    truth_post.push(move)
    obs = observation_from_transition(truth_pre, truth_post, chess.WHITE)
    assert obs.visible_pieces[chess.E4] == chess.Piece(chess.ROOK, chess.BLACK)

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=16,
        particles=[truth_pre.copy()],
        weights=[1.0],
    )
    belief.opp_remaining_counts = {chess.KING: 1, chess.ROOK: 1}

    belief.update_after_own_move(move, obs)

    assert belief.hard_opp_piece_facts == {
        chess.E4: chess.Piece(chess.ROOK, chess.BLACK)
    }
    assert "e4:black-rook" in belief.hard_fact_summary()["piece_facts"]


def test_prior_visible_piece_fact_survives_own_move_when_hidden() -> None:
    """Our own move cannot relocate an opponent piece we previously saw."""
    import random

    truth_pre = chess.Board.empty()
    truth_pre.turn = chess.WHITE
    truth_pre.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    truth_pre.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))
    truth_pre.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    truth_pre.set_piece_at(chess.H4, chess.Piece(chess.BISHOP, chess.BLACK))

    truth_post = truth_pre.copy()
    move = chess.Move.from_uci("a2a3")
    truth_post.push(move)
    obs = observation_from_transition(truth_pre, truth_post, chess.WHITE)
    assert chess.H4 not in obs.visibility_mask

    stale = truth_pre.copy()
    stale.remove_piece_at(chess.H4)
    stale.set_piece_at(chess.F6, chess.Piece(chess.BISHOP, chess.BLACK))

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=16,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {chess.KING: 1, chess.BISHOP: 1}
    belief.opp_bishop_colors_remaining = {True: 0, False: 1}
    belief.hard_opp_piece_facts[chess.H4] = chess.Piece(chess.BISHOP, chess.BLACK)

    belief.update_after_own_move(move, obs)

    assert belief.hard_opp_piece_facts == {
        chess.H4: chess.Piece(chess.BISHOP, chess.BLACK)
    }
    assert all(
        particle.piece_at(chess.H4) == chess.Piece(chess.BISHOP, chess.BLACK)
        for particle in belief.particles
    )


def test_prior_hidden_capture_landing_can_expire_on_opp_move() -> None:
    """A hidden occupancy fact is strict until the opponent can move it.

    Once the opponent gets a turn, a previously known hidden occupant may have
    moved away. If that move is consistent with the new observation, the square
    should no longer remain in the strict hard-fact ledger by fiat.
    """
    import random

    seed = chess.Board.empty()
    seed.turn = chess.BLACK
    seed.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    seed.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    seed.set_piece_at(chess.H4, chess.Piece(chess.BISHOP, chess.BLACK))

    truth = seed.copy()
    truth.push(chess.Move.from_uci("h4g3"))
    obs = observation_from_transition(seed, truth, chess.WHITE)
    assert chess.H4 not in obs.visibility_mask
    assert chess.G3 not in obs.visibility_mask

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=16,
        particles=[seed],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {chess.KING: 1, chess.BISHOP: 1}
    belief.opp_bishop_colors_remaining = {True: 0, False: 1}
    belief.hard_opp_occupancy_squares.add(chess.H4)

    belief.update_after_opp_move(obs)

    assert chess.H4 not in belief.hard_opp_occupancy_squares
    assert any(particle.piece_at(chess.H4) is None for particle in belief.particles)


def test_prior_visible_piece_fact_can_expire_on_opp_move() -> None:
    """After the opponent can move, a prior visible-piece fact is no longer strict."""
    import random

    seed = chess.Board.empty()
    seed.turn = chess.BLACK
    seed.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    seed.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    seed.set_piece_at(chess.H4, chess.Piece(chess.BISHOP, chess.BLACK))

    truth = seed.copy()
    truth.push(chess.Move.from_uci("h4g3"))
    obs = observation_from_transition(seed, truth, chess.WHITE)
    assert chess.H4 not in obs.visibility_mask
    assert chess.G3 not in obs.visibility_mask

    belief = BeliefState(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=16,
        particles=[seed],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {chess.KING: 1, chess.BISHOP: 1}
    belief.opp_bishop_colors_remaining = {True: 0, False: 1}
    belief.hard_opp_piece_facts[chess.H4] = chess.Piece(chess.BISHOP, chess.BLACK)

    belief.update_after_opp_move(obs)

    assert chess.H4 not in belief.hard_opp_piece_facts
    assert any(particle.piece_at(chess.H4) is None for particle in belief.particles)


def test_prior_visible_piece_fact_survives_opp_repair_for_different_move() -> None:
    """Repair must not erase exact facts unrelated to the observed opponent move.

    Regression for v0.7.16 g19: black knew a white rook was on h1, then White's
    c6xd7 observation forced Stage-B repair. The repair recovered the capture
    but let unrelated h1-rook hypotheses drift, demoting a hard fact into a
    soft marginal.
    """
    import random

    truth_pre = chess.Board.empty()
    truth_pre.turn = chess.WHITE
    truth_pre.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    truth_pre.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
    truth_pre.set_piece_at(chess.C6, chess.Piece(chess.PAWN, chess.WHITE))
    truth_pre.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
    truth_pre.set_piece_at(chess.D7, chess.Piece(chess.BISHOP, chess.BLACK))

    truth_post = truth_pre.copy()
    truth_post.push(chess.Move.from_uci("c6d7"))
    obs = observation_from_transition(truth_pre, truth_post, chess.BLACK)

    stale = truth_pre.copy()
    stale.remove_piece_at(chess.C6)

    belief = BeliefState(
        perspective=chess.BLACK,
        move_prior=uniform_prior,
        target_n=16,
        particles=[stale],
        weights=[1.0],
        rng=random.Random(0),
    )
    belief.opp_remaining_counts = {
        chess.KING: 1,
        chess.ROOK: 1,
        chess.PAWN: 1,
    }
    belief.opp_bishop_colors_remaining = {True: 0, False: 0}
    belief.hard_opp_piece_facts[chess.H1] = chess.Piece(chess.ROOK, chess.WHITE)

    belief.update_after_opp_move(obs)

    assert belief.last_repair_strict_rejected_count > 0
    assert belief.last_repair_fired == 0
    assert belief.last_csp_reseed_fired == 1
    assert belief.hard_opp_piece_facts[chess.H1] == chess.Piece(
        chess.ROOK, chess.WHITE
    )
    assert all(
        particle.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE)
        for particle in belief.particles
    )


def test_stage_b_constraint_pruned_diagnostic_increments() -> None:
    """`last_constraint_pruned` should be > 0 when the constraint actually
    rejects expanded particles."""
    seed = chess.Board()
    seed.push_uci("e2e4")  # white move first so opp (black) can move next
    belief = BeliefState.initial(
        perspective=chess.WHITE,
        move_prior=uniform_prior,
        target_n=32,
        start_board=seed,
    )
    # Pretend we've captured all queens — opp has 0 queens. Every expansion
    # still has the queen on d8, so the constraint should fire on each.
    belief.opp_remaining_counts[chess.QUEEN] = 0

    truth = seed.copy()
    truth.push_uci("e7e5")
    obs = observation_from_transition(seed, truth, chess.WHITE)
    belief.update_after_opp_move(obs)

    assert belief.last_constraint_pruned > 0
