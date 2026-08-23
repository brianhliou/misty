"""Concrete fog-of-war strategies for the self-play harness."""

from __future__ import annotations

import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# Bumped when Tier-1's decision behaviour materially changes. Major = new
# architectural layer; minor = behavioural change (new short-circuit,
# evaluator tweak, prior change); patch = refactor with no behaviour delta.
# Written into bake-off manifests so we can A/B across versions.
TIER1_VERSION = "0.9.5"


def tier1_commit() -> str:
    """Short git SHA of the working tree, suffixed with `-dirty` if uncommitted edits exist."""
    repo_root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        )
        return f"{sha}-dirty" if status.strip() else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

import chess

from .belief import BeliefState
from .engine import EvaluatorBuilder, best_action
from .mcts import mcts_pick_move
from .move_priors import OpponentMovePrior, uniform_prior
from .observation import Observation
from .selfplay import PerspectiveView
from .visibility import visible_squares


def _piece_color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _marginal_field_for_json(
    field: dict[chess.Square, list[tuple[chess.Piece | None, float]]]
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for sq, entries in field.items():
        result[chess.square_name(sq)] = [
            (
                {"piece": None, "prob": prob}
                if piece is None
                else {
                    "piece": piece.symbol(),
                    "color": _piece_color_name(piece.color),
                    "prob": prob,
                }
            )
            for piece, prob in entries
        ]
    return result


def _visibility_board(view: PerspectiveView) -> chess.Board:
    """Synthesize a board containing only the pieces this perspective can see."""
    board = chess.Board.empty()
    for sq, piece in view.visible_piece_map.items():
        board.set_piece_at(sq, piece)
    return board


def _ray_squares_between(from_sq: chess.Square, to_sq: chess.Square) -> list[chess.Square] | None:
    """Squares strictly between `from_sq` and `to_sq` on a rank/file/diagonal ray.

    Returns None if the two squares are not collinear (so a slider can't reach
    one from the other in a single move). Returns [] for adjacent squares.
    """
    f1, r1 = chess.square_file(from_sq), chess.square_rank(from_sq)
    f2, r2 = chess.square_file(to_sq), chess.square_rank(to_sq)
    df = f2 - f1
    dr = r2 - r1
    if df == 0 and dr == 0:
        return None
    if df != 0 and dr != 0 and abs(df) != abs(dr):
        return None
    sf = 0 if df == 0 else (1 if df > 0 else -1)
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    steps = max(abs(df), abs(dr))
    return [chess.square(f1 + sf * i, r1 + sr * i) for i in range(1, steps)]


def _squares_attacked_by_visible_enemy(view: PerspectiveView) -> set[chess.Square]:
    """Squares the visible enemy pieces could move to next turn.

    Built from a synthetic board with only visible pieces, with side-to-move
    flipped to enemy. Intentionally ignores hidden enemies — the short-circuit
    fires only on threats the perspective can directly observe.
    """
    board = _visibility_board(view)
    board.turn = not view.perspective
    return {move.to_square for move in board.pseudo_legal_moves}


def _squares_attacked_by_visible_enemy_full(view: PerspectiveView) -> set[chess.Square]:
    """Squares attacked by visible enemy pieces, *including* squares blocked
    by enemy's own pieces.

    Uses `chess.Board.attacks(square)` per enemy piece, which returns all
    squares the piece attacks regardless of own-piece blockade. Differs from
    `_squares_attacked_by_visible_enemy`, which uses pseudo-legal moves and
    omits enemy-own-piece-blocked squares (so misses defender detection: a
    bishop on f1 defended by king on e1 wouldn't show f1 as 'attacked' under
    the pseudo-legal version, since the king can't move onto its own bishop).

    Used by Pattern B's safe-capture check: after we capture the bishop on
    f1, is our piece now sitting on a square the enemy attacks? That requires
    the full attack set, not just the moves-to set.
    """
    board = _visibility_board(view)
    own = view.perspective
    attacked: set[chess.Square] = set()
    for sq, piece in view.visible_piece_map.items():
        if piece.color == own:
            continue
        attacked.update(board.attacks(sq))
    return attacked


def _king_defense_moves(view: PerspectiveView) -> list[chess.Move]:
    """If own king is on a square attacked by a visible enemy piece, return moves that resolve the attack.

    A resolving move ends with own king on a square that no visible enemy piece
    can capture next turn. This generalises across king-flight, attacker-
    capture, and sliding-piece blocking — each is detected by simulating the
    move on the visibility-only board and re-checking attacks on the king's
    new square. Returns [] when the king isn't visibly attacked OR no
    resolving move exists (don't fire the short-circuit then).
    """
    own_color = view.perspective
    own_king_squares = [
        sq
        for sq, piece in view.visible_piece_map.items()
        if piece.color == own_color and piece.piece_type == chess.KING
    ]
    if not own_king_squares:
        return []

    pre_attacked = _squares_attacked_by_visible_enemy(view)
    if not any(sq in pre_attacked for sq in own_king_squares):
        return []

    resolving: list[chess.Move] = []
    for own_move in view.own_legal_moves:
        sim = _visibility_board(view)
        sim.turn = own_color  # _visibility_board defaults to WHITE; push needs the perspective to-move
        if not sim.is_pseudo_legal(own_move):
            # En-passant or castling-edge cases that don't reconcile on a
            # visibility-only board. Fall through to main eval on those.
            continue
        sim.push(own_move)
        king_after: chess.Square | None = None
        for sq, piece in sim.piece_map().items():
            if piece.color == own_color and piece.piece_type == chess.KING:
                king_after = sq
                break
        if king_after is None:
            continue
        sim.turn = not own_color
        post_attacked = {m.to_square for m in sim.pseudo_legal_moves}
        if king_after not in post_attacked:
            resolving.append(own_move)
    return resolving


# Material values used for ranking captures within short-circuits. Numbers
# match the Stockfish-shallow evaluator's piece values closely enough; ties
# (B = N) are intentionally unbroken so the rng picks among equivalent options.
_MATERIAL_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000,  # short-circuited elsewhere; included for completeness
}

_RAY_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _slider_types_for_direction(df: int, dr: int) -> tuple[chess.PieceType, ...]:
    if df == 0 or dr == 0:
        return (chess.QUEEN, chess.ROOK)
    return (chess.QUEEN, chess.BISHOP)


def _latent_ray_danger_probes(
    view: PerspectiveView,
    belief: BeliefState,
    *,
    min_missing_mass: float = 0.05,
    limit: int = 8,
) -> list[dict]:
    """Find underrepresented hidden slider attacks on own king/queen.

    Diagnostic-only. This deliberately asks a different question from belief
    probability: "what low-mass hidden line threat would materially change
    move choice if true?" The first use case is a fogged queen/bishop/rook
    line toward our king or queen where current particles put little or no
    mass on the dangerous square.
    """
    own = view.perspective
    enemy = not own
    visible = set(view.visible_squares)
    visibility_board = _visibility_board(view)
    targets = [
        (sq, piece)
        for sq, piece in view.visible_piece_map.items()
        if piece.color == own and piece.piece_type in {chess.KING, chess.QUEEN}
    ]
    probes: list[dict] = []
    seen: set[tuple[chess.Square, chess.Square, chess.PieceType]] = set()
    visible_attacked = _squares_attacked_by_visible_enemy_full(view)

    for target_sq, target_piece in targets:
        target_file = chess.square_file(target_sq)
        target_rank = chess.square_rank(target_sq)
        for df, dr in _RAY_DIRECTIONS:
            blockers: list[chess.Square] = []
            file_cursor = target_file + df
            rank_cursor = target_rank + dr
            while 0 <= file_cursor <= 7 and 0 <= rank_cursor <= 7:
                sq = chess.square(file_cursor, rank_cursor)
                visible_piece = visibility_board.piece_at(sq)
                if visible_piece is not None:
                    break

                if sq not in visible:
                    for piece_type in _slider_types_for_direction(df, dr):
                        key = (target_sq, sq, piece_type)
                        if key in seen:
                            continue
                        seen.add(key)
                        enemy_piece = chess.Piece(piece_type, enemy)
                        belief_mass = belief.marginal_piece_at(sq).get(enemy_piece, 0.0)
                        if belief_mass >= min_missing_mass:
                            continue
                        blocking_moves = sorted(
                            {
                                move.uci()
                                for move in view.own_legal_moves
                                if move.to_square in blockers
                            }
                        )
                        actionable_blocking_moves = sorted(
                            {
                                move.uci()
                                for move in view.own_legal_moves
                                if move.to_square in blockers
                                and move.to_square not in visible_attacked
                                and (
                                    mover := view.visible_piece_map.get(move.from_square)
                                )
                                is not None
                                and mover.color == own
                                and mover.piece_type
                                not in (chess.KING, chess.QUEEN)
                            }
                        )
                        probes.append(
                            {
                                "target_square": chess.square_name(target_sq),
                                "target_piece": target_piece.symbol(),
                                "danger_square": chess.square_name(sq),
                                "danger_piece": chess.piece_symbol(piece_type),
                                "belief_mass": belief_mass,
                                "ray": [
                                    chess.square_name(path_sq)
                                    for path_sq in [*blockers, sq]
                                ],
                                "blocking_squares": [
                                    chess.square_name(path_sq) for path_sq in blockers
                                ],
                                "blocking_moves": blocking_moves[:8],
                                "actionable_blocking_moves": (
                                    actionable_blocking_moves[:8]
                                ),
                            }
                        )

                blockers.append(sq)
                file_cursor += df
                rank_cursor += dr

    def sort_key(probe: dict) -> tuple[int, float, int, str, str]:
        target_priority = 0 if probe["target_piece"].upper() == "K" else 1
        piece_priority = {"q": 0, "r": 1, "b": 1}.get(probe["danger_piece"], 9)
        return (
            target_priority,
            float(probe["belief_mass"]),
            piece_priority,
            str(probe["target_square"]),
            str(probe["danger_square"]),
        )

    probes.sort(key=sort_key)
    return probes[:limit]


def _latent_king_slider_block_moves(
    view: PerspectiveView,
    belief: BeliefState,
    *,
    min_missing_mass: float = 0.05,
) -> list[chess.Move]:
    """Prophylactic blockers for missing-belief queen rays to our king.

    This is deliberately narrower than `_latent_ray_danger_probes`: it only
    consumes queen-ray-to-king probes, and it only returns moves that interpose
    on the ray. It is the first bridge from "diagnose absent dangerous worlds"
    to "make a quiet move that prices them."
    """
    probes = _latent_ray_danger_probes(
        view,
        belief,
        min_missing_mass=min_missing_mass,
    )
    legal_by_uci = {move.uci(): move for move in view.own_legal_moves}
    visible_attacked = _squares_attacked_by_visible_enemy_full(view)
    candidates: list[tuple[chess.Move, tuple[int, int, int, int, str]]] = []
    for probe in probes:
        if str(probe.get("target_piece") or "").upper() != "K":
            continue
        if str(probe.get("danger_piece") or "").lower() != "q":
            continue
        if float(probe.get("belief_mass") or 0.0) >= min_missing_mass:
            continue
        blocking_squares = [
            chess.parse_square(name)
            for name in probe.get("blocking_squares", [])
            if isinstance(name, str)
        ]
        block_distance = {
            sq: idx for idx, sq in enumerate(blocking_squares)
        }
        for uci in (
            probe.get("actionable_blocking_moves")
            or probe.get("blocking_moves", [])
        ):
            move = legal_by_uci.get(str(uci))
            if move is None:
                continue
            mover = view.visible_piece_map.get(move.from_square)
            if mover is None or mover.color != view.perspective:
                continue
            if mover.piece_type in (chess.KING, chess.QUEEN):
                continue
            piece_pref = {
                chess.BISHOP: 0,
                chess.KNIGHT: 1,
                chess.PAWN: 2,
                chess.ROOK: 3,
            }.get(mover.piece_type, 9)
            attacked_pref = 1 if move.to_square in visible_attacked else 0
            distance_pref = block_distance.get(move.to_square, 99)
            home_rank = 0 if view.perspective == chess.WHITE else 7
            development_pref = (
                0
                if mover.piece_type in (chess.KNIGHT, chess.BISHOP)
                and chess.square_rank(move.from_square) == home_rank
                else 1
            )
            candidates.append(
                (
                    move,
                    (
                        piece_pref,
                        attacked_pref,
                        development_pref,
                        distance_pref,
                        move.uci(),
                    ),
                )
            )

    if not candidates:
        return []
    best_pref = min(pref for _, pref in candidates)
    return sorted({move for move, pref in candidates if pref == best_pref}, key=lambda m: m.uci())


def _categorize_king_defense_moves(
    view: PerspectiveView,
) -> tuple[list[chess.Move], list[chess.Move], list[chess.Move]]:
    """Partition king-defense moves into (attacker_captures, blocks, flights).

    Same legality criterion as `_king_defense_moves` (move resolves the visible
    check on the visibility-only board), but classified by mechanism. The
    pick_move ranking — captures > blocks > flights — is what v0.6.1 added in
    response to the v0.6.0-mirror corpus showing the engine choosing flight
    when a free attacker-capture was available (g4 ply 17, g12 ply 17 majors).

    `attacker_captures`: own move lands on a square holding a visible enemy
    piece that was attacking our king pre-move. Material gain is the captured
    piece's value (we don't separately rank within captures here; callers
    apply `_prefer_higher_value_capture`).

    `blocks`: own move lands on a square BETWEEN the visible attacker and our
    king, breaking the line of attack. Includes interpose moves only when the
    attacker is a sliding piece — knights and pawns can't be blocked.

    `flights`: own king moves to a square not attacked by any visible enemy.
    """
    own_color = view.perspective
    own_king_squares = [
        sq
        for sq, piece in view.visible_piece_map.items()
        if piece.color == own_color and piece.piece_type == chess.KING
    ]
    if not own_king_squares:
        return [], [], []

    pre_attacked = _squares_attacked_by_visible_enemy(view)
    if not any(sq in pre_attacked for sq in own_king_squares):
        return [], [], []

    # Visible enemy attackers of our king.
    attacker_squares = _visible_attackers_of_squares(view, set(own_king_squares))

    captures: list[chess.Move] = []
    blocks: list[chess.Move] = []
    flights: list[chess.Move] = []
    for own_move in view.own_legal_moves:
        sim = _visibility_board(view)
        sim.turn = own_color
        if not sim.is_pseudo_legal(own_move):
            continue
        sim.push(own_move)
        king_after: chess.Square | None = None
        for sq, piece in sim.piece_map().items():
            if piece.color == own_color and piece.piece_type == chess.KING:
                king_after = sq
                break
        if king_after is None:
            continue
        sim.turn = not own_color
        post_attacked = {m.to_square for m in sim.pseudo_legal_moves}
        if king_after in post_attacked:
            continue  # didn't resolve the check

        # Categorize. Order matters: a king move CAN also be a capture
        # (king-takes-attacker), and we treat that as capture, not flight.
        if own_move.to_square in attacker_squares:
            captures.append(own_move)
        elif (
            view.visible_piece_map.get(own_move.from_square) is not None
            and view.visible_piece_map[own_move.from_square].piece_type == chess.KING
        ):
            flights.append(own_move)
        else:
            blocks.append(own_move)

    return captures, blocks, flights


def _visible_attackers_of_squares(
    view: PerspectiveView, target_squares: set[chess.Square]
) -> set[chess.Square]:
    """Return squares of visible enemy pieces that attack any of `target_squares`.

    Used by king-defense to identify "the attacker(s) we want to capture."
    Built from a synthetic visibility-only board with the enemy as side-to-move.
    """
    sim = _visibility_board(view)
    sim.turn = not view.perspective
    attackers: set[chess.Square] = set()
    for move in sim.pseudo_legal_moves:
        if move.to_square in target_squares:
            attackers.add(move.from_square)
    return attackers


def _prefer_higher_value_capture(
    moves: list[chess.Move], view: PerspectiveView
) -> list[chess.Move]:
    """Among capture moves, restrict to those that capture the highest-material piece.

    Captures here are identified by `view.visible_piece_map[move.to_square]`
    being an enemy piece. Moves where to_square has no visible piece (or is
    own piece) are passed through unchanged at the END, after the max-tier is
    chosen — meaning a list of mixed captures and non-captures is reduced to
    only the highest-material captures. Used inside king-defense's
    attacker-capture set to prefer Rxattacker over Pxattacker, etc.
    """
    own = view.perspective
    valued: list[tuple[chess.Move, int]] = []
    for m in moves:
        target = view.visible_piece_map.get(m.to_square)
        if target is not None and target.color != own:
            valued.append((m, _MATERIAL_VALUE.get(target.piece_type, 0)))
    if not valued:
        return moves
    max_value = max(v for _, v in valued)
    return [m for m, v in valued if v == max_value]


def _prefer_lower_value_attacker(
    moves: list[chess.Move], view: PerspectiveView
) -> list[chess.Move]:
    """Among captures of the same target, prefer the least valuable attacker.

    In fog, a visible high-value target may still sit on a square defended by
    hidden pieces. When multiple own pieces can capture it, spend the cheapest
    attacker first. This prevents avoidable queen-first captures on contested
    squares when a knight/pawn capture is also available.
    """
    own = view.perspective
    valued: list[tuple[chess.Move, int]] = []
    for move in moves:
        attacker = view.visible_piece_map.get(move.from_square)
        target = view.visible_piece_map.get(move.to_square)
        if (
            attacker is None
            or attacker.color != own
            or target is None
            or target.color == own
        ):
            continue
        valued.append((move, _MATERIAL_VALUE.get(attacker.piece_type, 0)))
    if not valued:
        return moves
    min_value = min(v for _, v in valued)
    return [m for m, v in valued if v == min_value]


def _prefer_lower_value_same_target_capture(
    chosen: chess.Move,
    moves: list[chess.Move],
    view: PerspectiveView,
) -> chess.Move:
    """Replace a chosen visible capture with a cheaper attacker to the same square.

    This is deliberately narrower than a capture short-circuit: it does not
    force the engine to capture. It only says that once search/eval picked a
    capture on a visible target, and another non-king own piece can take that
    same target, spend the least valuable attacker first.
    """
    own = view.perspective
    target = view.visible_piece_map.get(chosen.to_square)
    chosen_attacker = view.visible_piece_map.get(chosen.from_square)
    if (
        target is None
        or target.color == own
        or chosen_attacker is None
        or chosen_attacker.color != own
        or chosen_attacker.piece_type == chess.KING
    ):
        return chosen

    candidates: list[tuple[chess.Move, int]] = []
    for move in moves:
        if move.to_square != chosen.to_square:
            continue
        attacker = view.visible_piece_map.get(move.from_square)
        if (
            attacker is None
            or attacker.color != own
            or attacker.piece_type == chess.KING
        ):
            continue
        candidates.append((move, _MATERIAL_VALUE.get(attacker.piece_type, 0)))
    if not candidates:
        return chosen

    min_value = min(value for _, value in candidates)
    chosen_value = _MATERIAL_VALUE.get(chosen_attacker.piece_type, 0)
    if min_value >= chosen_value:
        return chosen
    cheapest = [move for move, value in candidates if value == min_value]
    # Stable choice keeps replay deterministic and avoids consuming RNG outside
    # the main engine selector.
    return sorted(cheapest, key=lambda move: move.uci())[0]


def _queen_save_moves(view: PerspectiveView) -> list[chess.Move]:
    """If own queen is on a square attacked by a visible enemy, return moves that resolve the threat.

    Mirrors the king-defense logic: simulate every legal move on the
    visibility-only board; a resolving move ends with all own queens on
    squares no visible enemy can capture next turn. This generalises across
    queen-flight, attacker-capture, and sliding-piece blocking — captures of
    the attacker by any of our pieces fall out for free, which the v0.4.0
    corpus showed was the missed case (e.g., g24 p22: pawn could have taken
    the attacking knight, but the old queen-save only considered queen-moves).

    Returns [] when the queen isn't visibly attacked OR no resolving move
    exists (don't fire the short-circuit then).
    """
    own_color = view.perspective
    own_queen_squares = [
        sq
        for sq, piece in view.visible_piece_map.items()
        if piece.color == own_color and piece.piece_type == chess.QUEEN
    ]
    if not own_queen_squares:
        return []

    pre_attacked = _squares_attacked_by_visible_enemy(view)
    if not any(sq in pre_attacked for sq in own_queen_squares):
        return []

    resolving: list[chess.Move] = []
    for own_move in view.own_legal_moves:
        sim = _visibility_board(view)
        sim.turn = own_color
        if not sim.is_pseudo_legal(own_move):
            continue
        sim.push(own_move)
        queens_after = [
            sq
            for sq, piece in sim.piece_map().items()
            if piece.color == own_color and piece.piece_type == chess.QUEEN
        ]
        if not queens_after:
            # Our move somehow lost the queen — not a save.
            continue
        sim.turn = not own_color
        post_attacked = {m.to_square for m in sim.pseudo_legal_moves}
        if all(q not in post_attacked for q in queens_after):
            resolving.append(own_move)
    return resolving


def _queen_save_tiers(view: PerspectiveView) -> list[list[chess.Move]]:
    """Rank queen-save moves as attacker-captures before blocks/flights."""
    saves = _queen_save_moves(view)
    if not saves:
        return []

    own = view.perspective
    queen_squares = {
        sq
        for sq, piece in view.visible_piece_map.items()
        if piece.color == own and piece.piece_type == chess.QUEEN
    }
    attacker_squares = _visible_attackers_of_squares(view, queen_squares)
    attacker_captures = [
        move
        for move in saves
        if move.to_square in attacker_squares
        and (target := view.visible_piece_map.get(move.to_square)) is not None
        and target.color != own
    ]
    tiers: list[list[chess.Move]] = []
    if attacker_captures:
        tiers.append(
            _prefer_lower_value_attacker(
                _prefer_higher_value_capture(attacker_captures, view),
                view,
            )
        )
    remaining = [move for move in saves if move not in set(attacker_captures)]
    if remaining:
        tiers.append(remaining)
    return tiers


def _high_value_piece_save_moves(view: PerspectiveView) -> list[chess.Move]:
    """Resolve visible attacks on own queens/rooks using the cheapest mover.

    Generalizes queen-save to rooks. v0.7.8 rung annotations showed White
    ignoring a visible queen line on the h1 rook (`...Qe4xh1`) when a pawn
    block (`f2f3`) was available. In Fog, visible attacks on queen/rook are
    tactical facts; do not leave them to flat material-eval tie breaks.
    """
    own = view.perspective
    high_value_squares = [
        sq
        for sq, piece in view.visible_piece_map.items()
        if piece.color == own and piece.piece_type in (chess.QUEEN, chess.ROOK)
    ]
    if not high_value_squares:
        return []

    pre_attacked = _squares_attacked_by_visible_enemy(view)
    threatened = [sq for sq in high_value_squares if sq in pre_attacked]
    if not threatened:
        return []

    resolving: list[tuple[chess.Move, int]] = []
    for own_move in view.own_legal_moves:
        sim = _visibility_board(view)
        sim.turn = own
        if not sim.is_pseudo_legal(own_move):
            continue
        sim.push(own_move)
        still_valuable = [
            sq
            for sq, piece in sim.piece_map().items()
            if piece.color == own and piece.piece_type in (chess.QUEEN, chess.ROOK)
        ]
        sim.turn = not own
        post_attacked = {move.to_square for move in sim.pseudo_legal_moves}
        if any(sq in post_attacked for sq in still_valuable):
            continue
        mover = view.visible_piece_map.get(own_move.from_square)
        mover_value = (
            _MATERIAL_VALUE.get(mover.piece_type, 1000)
            if mover is not None and mover.color == own
            else 1000
        )
        resolving.append((own_move, mover_value))

    if not resolving:
        return []
    best_value = min(value for _, value in resolving)
    return [move for move, value in resolving if value == best_value]


def _castle_moves(view: PerspectiveView) -> list[chess.Move]:
    """Return safe castling moves available in the perspective move list."""
    own = view.perspective
    attacked = _squares_attacked_by_visible_enemy_full(view)
    candidates: list[chess.Move] = []
    for move in view.own_legal_moves:
        piece = view.visible_piece_map.get(move.from_square)
        if (
            piece is None
            or piece.color != own
            or piece.piece_type != chess.KING
            or abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)) != 2
        ):
            continue
        if move.from_square in attacked or move.to_square in attacked:
            continue
        candidates.append(move)
    return candidates


def _fog_depth(square: chess.Square, perspective: chess.Color) -> int:
    rank = chess.square_rank(square)
    if perspective == chess.WHITE:
        return max(0, rank - 3)
    return max(0, 4 - rank)


def _center_file_distance(square: chess.Square) -> float:
    return abs(chess.square_file(square) - 3.5)


def _advanced_minor_retreat_moves(
    view: PerspectiveView, *, min_from_depth: int = 1
) -> list[chess.Move]:
    """Early FOW retreat for minors that have drifted into enemy territory."""
    own = view.perspective
    attacked = _squares_attacked_by_visible_enemy_full(view)
    candidates: list[tuple[chess.Move, int, int, float]] = []
    for move in view.own_legal_moves:
        piece = view.visible_piece_map.get(move.from_square)
        if (
            piece is None
            or piece.color != own
            or piece.piece_type not in (chess.BISHOP, chess.KNIGHT)
        ):
            continue
        if view.visible_piece_map.get(move.to_square) is not None:
            continue
        from_depth = _fog_depth(move.from_square, own)
        to_depth = _fog_depth(move.to_square, own)
        if from_depth < min_from_depth or to_depth >= from_depth:
            continue
        if move.to_square in attacked:
            continue
        to_rank = chess.square_rank(move.to_square)
        home_pref = to_rank if own == chess.WHITE else 7 - to_rank
        candidates.append(
            (move, to_depth, home_pref, _center_file_distance(move.to_square))
        )

    if not candidates:
        return []
    best_depth = min(depth for _, depth, _, _ in candidates)
    best = [row for row in candidates if row[1] == best_depth]
    best_home = min(home for _, _, home, _ in best)
    best = [row for row in best if row[2] == best_home]
    best_center = min(center for _, _, _, center in best)
    return [move for move, _, _, center in best if center == best_center]


def _visible_piece_save_moves(
    view: PerspectiveView,
    belief: "BeliefState | None" = None,
    *,
    piece_types: tuple[chess.PieceType, ...] = (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
    ),
) -> list[chess.Move]:
    """Resolve visible attacks on own non-queen material.

    Queen/rook high-value saves have dedicated handling, but v0.7.9 review
    showed the same blind spot for bishops and knights: when a visible enemy
    pawn or minor attacks an own piece, main-eval may still prefer unrelated
    material. Simulate candidate moves on the visibility board and keep only
    moves that reduce the threatened own material value.
    """
    own = view.perspective
    pre_attacked = _squares_attacked_by_visible_enemy(view)
    threatened_before = {
        sq
        for sq, piece in view.visible_piece_map.items()
        if piece.color == own and piece.piece_type in piece_types and sq in pre_attacked
    }
    if not threatened_before:
        return []

    before_value = sum(
        _MATERIAL_VALUE[view.visible_piece_map[sq].piece_type]
        for sq in threatened_before
    )
    candidates: list[tuple[chess.Move, int, int, float]] = []
    for move in view.own_legal_moves:
        sim = _visibility_board(view)
        sim.turn = own
        if not sim.is_pseudo_legal(move):
            continue
        sim.push(move)
        sim.turn = not own
        post_attacked = {reply.to_square for reply in sim.pseudo_legal_moves}
        post_value = sum(
            _MATERIAL_VALUE[piece.piece_type]
            for sq, piece in sim.piece_map().items()
            if piece.color == own
            and piece.piece_type in piece_types
            and sq in post_attacked
        )
        if post_value >= before_value:
            continue
        mover = view.visible_piece_map.get(move.from_square)
        mover_value = (
            _MATERIAL_VALUE.get(mover.piece_type, 1000)
            if mover is not None and mover.color == own
            else 1000
        )
        moved_piece_depth = _fog_depth(move.to_square, own)
        # Prefer recaptures over retreats when the recapture destination is
        # belief-safe. Only credit capture_gain when particles confirm < 30%
        # chance of an immediate recapture — prevents preferring captures
        # defended by hidden pieces (e.g. a bishop outside the visibility set).
        captured = view.visible_piece_map.get(move.to_square)
        capture_gain = 0
        if captured is not None and captured.color != own and belief is not None and belief.particles:
            total_w = 0.0
            recapturable_w = 0.0
            for particle, weight in zip(belief.particles, belief.weights, strict=False):
                if not particle.is_pseudo_legal(move):
                    continue
                total_w += weight
                sim = particle.copy()
                sim.push(move)
                sim.turn = not own
                if any(r.to_square == move.to_square for r in sim.pseudo_legal_moves):
                    recapturable_w += weight
            if total_w > 0 and recapturable_w / total_w < 0.3:
                capture_gain = _MATERIAL_VALUE.get(captured.piece_type, 0)
        candidates.append(
            (move, post_value, mover_value, moved_piece_depth, capture_gain)
        )

    if not candidates:
        return []
    best_post = min(post for _, post, _, _, _ in candidates)
    best = [row for row in candidates if row[1] == best_post]
    best_gain = max(gain for _, _, _, _, gain in best)
    best = [row for row in best if row[4] == best_gain]
    best_mover = min(mover for _, _, mover, _, _ in best)
    best = [row for row in best if row[2] == best_mover]
    best_depth = min(depth for _, _, _, depth, _ in best)
    return [move for move, _, _, depth, _ in best if depth == best_depth]


def _early_development_moves(view: PerspectiveView) -> list[chess.Move]:
    """Quiet central development while the king is still on its home square.

    The material evaluator treats many opening quiet moves as equivalent, which
    lets random tie breaks pick rook-pawn pushes while pieces remain undeveloped.
    In FOW that creates avoidable loose-piece and king-safety problems. Keep the
    policy narrow: only home-king positions, only non-capturing d/e pawn moves,
    and prefer one-square e-pawn development before broader central pushes.

    Pawn-shield diagonal tier: among candidates, prefer pawns whose home square
    does NOT sit on a diagonal from the own king. Moving such a pawn vacates the
    blocking square and opens a diagonal toward the uncastled king — a hidden
    enemy slider (queen, bishop) could then arrive in one move. The e-file pawn
    on e2/e7 is on the king's file (not a diagonal), so it's always tier-0.
    The d-file pawn on d2/d7 IS on the king's a5-e1 / e8-a5 diagonal, so it
    ranks below e-pawn when both are available. When only d-pawn remains (e-pawn
    already advanced) the tier falls back gracefully — development still fires.
    """
    own = view.perspective
    king_home = chess.E1 if own == chess.WHITE else chess.E8
    king = view.visible_piece_map.get(king_home)
    if king is None or king.color != own or king.piece_type != chess.KING:
        return []

    pawn_rank = 1 if own == chess.WHITE else 6
    direction = 1 if own == chess.WHITE else -1
    attacked = _squares_attacked_by_visible_enemy_full(view)
    # (move, diag_pref, file_pref, step_pref)
    candidates: list[tuple[chess.Move, int, int, int]] = []
    for move in view.own_legal_moves:
        piece = view.visible_piece_map.get(move.from_square)
        if piece is None or piece.color != own or piece.piece_type != chess.PAWN:
            continue
        if chess.square_rank(move.from_square) != pawn_rank:
            continue
        from_file = chess.square_file(move.from_square)
        if from_file not in (3, 4):
            continue
        if view.visible_piece_map.get(move.to_square) is not None:
            continue
        if move.to_square in attacked:
            continue

        rank_delta = chess.square_rank(move.to_square) - chess.square_rank(
            move.from_square
        )
        steps = rank_delta * direction
        if steps not in (1, 2):
            continue

        file_pref = 0 if from_file == 4 else 1
        step_pref = 0 if steps == 1 else 1

        # Pawn-shield diagonal penalty: pawn on a diagonal from the king (|df|==|dr|,
        # df≠0) blocks that diagonal; moving it forward (same file) opens it.
        # Prefer non-diagonal pawns (diag_pref=0) so the engine doesn't voluntarily
        # open king diagonals when a safer pawn advance exists.
        df = chess.square_file(move.from_square) - chess.square_file(king_home)
        dr = chess.square_rank(move.from_square) - chess.square_rank(king_home)
        diag_pref = 1 if (df != 0 and abs(df) == abs(dr)) else 0

        candidates.append((move, diag_pref, file_pref, step_pref))

    if not candidates:
        return []
    best_diag = min(d for _, d, _, _ in candidates)
    best = [row for row in candidates if row[1] == best_diag]
    best_file = min(f for _, _, f, _ in best)
    best = [row for row in best if row[2] == best_file]
    best_step = min(s for _, _, _, s in best)
    return [move for move, _, _, s in best if s == best_step]


def _safe_visible_minor_or_rook_captures(
    view: PerspectiveView,
) -> list[chess.Move]:
    """Visible captures of bishop/knight/rook on squares not attacked by other visible enemies.

    v0.6.1 Pattern B fix: v0.6.0-mirror corpus (g4 p18 major) showed the
    engine choosing a pawn capture over a free bishop capture, because main-
    eval's per-particle Stockfish vote diluted the material delta. Catching
    the obvious case here — a hanging visible minor/rook — bypasses the dilution.

    Conditions:
      - Move captures a visible enemy piece of type B/N/R (queen/king covered
        by their own short-circuits; pawns let through to main-eval since the
        material delta is small).
      - Destination square is NOT attacked by any other visible enemy piece
        (so we don't trade our piece into a visible defender).

    Returns highest-material captures only (R > B = N), so a knight-takes-rook
    beats a knight-takes-bishop when both are safe.
    """
    own = view.perspective
    # Use the FULL attack set (includes squares defended through own pieces),
    # not pseudo_legal_moves. The captured target is itself an enemy piece, so
    # any enemy attacker behind it (that would be blocked by it pre-capture) is
    # invisible to a pseudo-legal-moves check; we'd misclassify a defended
    # bishop as 'safe'.
    pre_attacked = _squares_attacked_by_visible_enemy_full(view)
    candidates: list[tuple[chess.Move, int]] = []
    for move in view.own_legal_moves:
        target = view.visible_piece_map.get(move.to_square)
        if target is None or target.color == own:
            continue
        if target.piece_type not in (chess.BISHOP, chess.KNIGHT, chess.ROOK):
            continue
        attacker = view.visible_piece_map.get(move.from_square)
        if attacker is not None and attacker.piece_type == chess.KING:
            continue
        if move.to_square in pre_attacked:
            # Visible defender on the destination — main-eval can decide
            # whether the trade is worth it. Don't auto-fire.
            continue
        candidates.append((move, _MATERIAL_VALUE[target.piece_type]))
    if not candidates:
        return []
    max_value = max(v for _, v in candidates)
    return [m for m, v in candidates if v == max_value]


def _nonterminal_king_material_capture(
    move: chess.Move,
    view: PerspectiveView,
) -> bool:
    """True for king captures of visible non-king material.

    In Fog of War, using the king as a generic material attacker is especially
    dangerous: the target square may be defended by hidden pieces, and king
    capture is terminal. Keep terminal king captures and king-defense moves
    available through their dedicated short-circuits, but remove ordinary
    king-takes-piece candidates from material and main-eval paths when there
    are alternatives.
    """
    attacker = view.visible_piece_map.get(move.from_square)
    target = view.visible_piece_map.get(move.to_square)
    return (
        attacker is not None
        and attacker.piece_type == chess.KING
        and target is not None
        and target.color != view.perspective
        and target.piece_type != chess.KING
    )


def _king_shelter_moves(view: PerspectiveView) -> list[chess.Move]:
    """Minor-piece interpositions that shelter an uncastled king on the e-file.

    FOW-specific move-selection guardrail from the annotation replay gate:
    when our king is still on e1/e8, the e-pawn is gone, and the direct shelter
    square e2/e7 is empty, moving a minor piece there is often worth more than
    grabbing a pawn or making a generic material move. It reduces immediate
    central-file exposure without requiring us to know every hidden attacker.

    This is deliberately conservative:
      - only e1/e8 kings;
      - only bishop/knight moves to e2/e7;
      - only when the shelter square is not visibly attacked;
      - home-square bishop is preferred; otherwise knight is preferred over
        pulling back an already-developed bishop.
    """
    own = view.perspective
    king_sq = chess.E1 if own == chess.WHITE else chess.E8
    shelter_sq = chess.E2 if own == chess.WHITE else chess.E7

    king = view.visible_piece_map.get(king_sq)
    if king is None or king.color != own or king.piece_type != chess.KING:
        return []
    if shelter_sq in view.visible_piece_map:
        return []
    if shelter_sq in _squares_attacked_by_visible_enemy_full(view):
        return []

    candidates: list[tuple[chess.Move, int]] = []
    for move in view.own_legal_moves:
        if move.to_square != shelter_sq:
            continue
        piece = view.visible_piece_map.get(move.from_square)
        if piece is None or piece.color != own:
            continue
        if piece.piece_type == chess.BISHOP:
            bishop_home = chess.F1 if own == chess.WHITE else chess.F8
            pref = 0 if move.from_square == bishop_home else 2
        elif piece.piece_type == chess.KNIGHT:
            pref = 1
        else:
            continue
        candidates.append((move, pref))

    if not candidates:
        return []
    best_pref = min(pref for _, pref in candidates)
    return [move for move, pref in candidates if pref == best_pref]


def _prefer_queen_promotion(moves: list[chess.Move]) -> list[chess.Move]:
    """Among a set of capture moves, when promotions are present, restrict to queen-promotions.

    Used inside the king-capture and queen-capture short-circuits where the
    `_rng.choice` over capture moves was randomly producing under-promotions
    (g22 p26 minor: rook promotion; g24 p14 major: knight promotion in
    middlegame). Queen is strictly best-or-equal except for niche stalemate
    avoidance — accept that asymmetry.
    """
    if not any(m.promotion is not None for m in moves):
        return moves
    queen_promos = [m for m in moves if m.promotion == chess.QUEEN]
    if queen_promos:
        return queen_promos
    return moves


def _compute_per_move_budget_ms(
    clock_remaining_ms: int,
    increment_ms: int,
    *,
    safety_ms: int = 200,
    moves_remaining_estimate: int = 40,
    soft_cap_ms: int = 10_000,
) -> int:
    """Standard chess time-management heuristic.

    `budget = (clock - safety) / moves_remaining + increment`, clamped to
    [50, soft_cap]. Using 1/40 of the bank per move ensures the engine
    can sustain ~40 moves before the bank depletes, while the increment
    refills the bank so steady-state thinking is bounded by the increment.

    soft_cap protects against single-move blowups when the bank is huge
    (e.g., 60+2 control on the first move would otherwise allow a 4-second
    think; 10s cap is generous and lets even deep positions run).
    """
    usable = max(0, clock_remaining_ms - safety_ms)
    bank_share = usable // moves_remaining_estimate
    budget = bank_share + increment_ms
    return max(50, min(soft_cap_ms, budget))


class RandomStrategy:
    """Picks uniformly at random from legal moves. Baseline opponent."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def reset(self, perspective: chess.Color) -> None:
        self.perspective = perspective

    def observe_own_move(self, move: chess.Move, observation: Observation) -> None:
        pass

    def observe_opp_move(self, observation: Observation) -> None:
        pass

    def pick_move(self, view: PerspectiveView) -> chess.Move:
        return self.rng.choice(view.own_legal_moves)


class LegalGreedy:
    """Capture-if-can-else-random-legal. No belief, no Stockfish.

    Ladder-floor reference: every bot above this is provably doing belief work.
    A "capture" here is a move whose target square holds a *visible* opp
    piece — moves into hidden squares are random (the bot doesn't know).
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def reset(self, perspective: chess.Color) -> None:
        self.perspective = perspective

    def observe_own_move(self, move: chess.Move, observation: Observation) -> None:
        pass

    def observe_opp_move(self, observation: Observation) -> None:
        pass

    def pick_move(self, view: PerspectiveView) -> chess.Move:
        captures: list[chess.Move] = []
        for move in view.own_legal_moves:
            target = view.visible_piece_map.get(move.to_square)
            if target is not None and target.color != view.perspective:
                if target.piece_type == chess.KING:
                    return move
                captures.append(move)
        return self.rng.choice(captures or view.own_legal_moves)


@dataclass
class Tier1Strategy:
    """Belief tracker + per-particle evaluator vote.

    The evaluator is taken in via an `EvaluatorBuilder`: a callable that
    produces a per-move Evaluator given the current PerspectiveView. View-
    independent evaluators (material, Stockfish) wrap with
    `engine.static_builder`. Visibility-grounded evaluators close over the
    view to compute threats from observed truth.
    """

    evaluator_builder: EvaluatorBuilder
    move_prior: OpponentMovePrior = field(default=uniform_prior)
    target_n: int = 256
    max_eval_particles: int = 16
    risk_aversion: float = 0.0
    seed: int = 0
    # MCTS: when mcts_rollouts > 0, use Monte Carlo Tree Search instead of the
    # 1-ply short-circuit hierarchy after hard certainty vetoes.
    mcts_rollouts: int = 0
    mcts_rollout_depth: int = 8
    mcts_selection_depth: int = 3
    mcts_risk_lambda: float = 0.25
    # Per-pick observability. Populated by pick_move; the bake-off harness
    # reads this after each game to dump a per-Tier-1-ply trace JSONL. Each
    # entry: {tier1_move_count, ply, decision_path, particle_count_pre_sample,
    # move_chosen_uci, top_k_scores}. top_k_scores is empty when a short-circuit
    # fired (no main-eval scoring happened).
    trace_log: list[dict] = field(default_factory=list)
    verbose_belief_capture: bool = False
    belief_log: list[dict] = field(default_factory=list)
    # Pillar 1 — PIMC search depth. 0 disables (use legacy best_action 1-ply
    # leaf eval, the v0.9.4 path). 2 enables depth-2 minimax per particle:
    # me plays m → opp plays best response → leaf eval. Higher depths not
    # yet supported. ~5x slower per ply at depth=2 with default N particles.
    pimc_search_depth: int = 0
    # v0.9.4-pre1 stopgap: belief-weighted soft penalties applied AFTER best_action
    # scores the legal moves. Addresses annotation classes from q0:
    #   - moves into known-attacker squares (plies 22, 94, 116, 170) -> capture_risk
    #   - shuffle when ahead (ply 190) -> anti_shuffle
    # Set to 0 to disable; production v0.9.3 behavior is recovered by zero values.
    # First calibration (coef=100, anti_shuffle=200, thresholds=0.10/0.12)
    # was catastrophically aggressive: both engines stopped making any non-king
    # moves and games died to 50-move rule. Tuned values below leave existing
    # vetoes near their production thresholds and add a SOFT nudge on top.
    capture_risk_penalty_coef: float = 10.0
    anti_shuffle_penalty: float = 20.0
    anti_shuffle_window: int = 4
    # Veto thresholds: keep production defaults (0.20 / 0.25). The soft
    # penalty above subsumes the "tighten vetoes" use case.
    queen_fog_risk_threshold: float = 0.20
    piece_fog_risk_threshold: float = 0.25
    # v0.9.5 draw-reduction knobs. Addresses the 67% draw rate observed in
    # 100-game self-play corpus (the "shuffles instead of pushing when ahead"
    # pathology q0 ply 190 annotated). Three additive bonuses applied in the
    # same post-score loop as capture-risk/anti-shuffle:
    push_when_ahead_bonus: float = 200.0    # cp per "advances toward opp side" move when material edge > threshold
    push_when_ahead_min_edge: float = 3.0   # require material edge > this (in pawn units) for push bonus
    info_reveal_bonus_coef: float = 25.0    # cp per square of new visibility unlocked by the move
    anti_shuffle_penalty_strong: float = 250.0  # used in place of anti_shuffle_penalty when material edge > push_when_ahead_min_edge

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._belief: BeliefState | None = None
        self._tier1_move_count = 0
        self._observed_ply = 0
        # Belief-step diagnostics for the next emit_trace. Captured by
        # observe_own_move (Stage A) and observe_opp_move (Stage B) and
        # consumed (cleared) by the next pick_move's _emit_trace.
        self._pending_belief_steps: dict[str, int] = {}
        # v0.6.0: pending capture detected in pick_move from the pre-move
        # visible_piece_map. Consumed by observe_own_move to update belief's
        # opp_remaining_counts.
        self._pending_capture_type: chess.PieceType | None = None
        self._pending_capture_square: chess.Square | None = None
        # Pre-move visibility snapshot, captured at the start of pick_move and
        # used during capture detection. Avoids needing to plumb the view
        # through observe_own_move.
        self._last_view_visible: dict[chess.Square, chess.Piece] = {}
        self._last_decision_view: PerspectiveView | None = None
        # v0.9.4-pre1: rolling window of own (from_sq, to_sq) used by anti-shuffle.
        self._recent_own_moves: list[tuple[int, int]] = []
        # v0.9.6-pre: square where opp captured one of our pieces on opp's most
        # recent move. Read by _belief_veto_bad_capture_trade to exempt
        # recaptures from the equal-value-trade veto — recapturing material
        # opp just took is fundamentally different from initiating a fresh
        # equal trade.
        self._opp_last_capture_target: chess.Square | None = None

    def reset(self, perspective: chess.Color) -> None:
        self._belief = BeliefState.initial(
            perspective=perspective,
            move_prior=self.move_prior,
            target_n=self.target_n,
            rng=random.Random(self.seed + (1 if perspective == chess.BLACK else 0)),
        )
        self.trace_log.clear()
        self.belief_log.clear()
        self._tier1_move_count = 0
        self._observed_ply = 0
        self._pending_belief_steps = {}
        self._pending_capture_type = None
        self._pending_capture_square = None
        self._last_view_visible = {}
        self._last_decision_view = None
        self._recent_own_moves = []
        self._opp_last_capture_target = None

    def _emit_trace(
        self,
        decision_path: str,
        particle_count_pre: int,
        chosen: chess.Move,
        top_k_scores: list[tuple[str, float, float]] | None = None,
        decision_weight_modes: dict | None = None,
    ) -> None:
        self._tier1_move_count += 1
        # Decision snapshot is for the move about to be played. Observed-ply
        # tracking stays correct even if a random opening ran before control
        # reached this strategy.
        assert self._belief is not None
        ply = self._observed_ply + 1
        belief_unique = len({p.fen() for p in self._belief.particles}) if self._belief else 0
        # Snapshot opp_remaining_counts as a string-keyed dict (chess.PieceType
        # is an int alias, so JSON serializes — but readability suffers).
        piece_type_names = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }
        opp_counts = {
            piece_type_names[pt]: n
            for pt, n in self._belief.opp_remaining_counts.items()
        }
        support_weight, support_count, support_unique = self._belief_move_support_stats(
            chosen
        )
        pending_steps = dict(self._pending_belief_steps)
        csp_reseed_fired = bool(
            pending_steps.get("csp_reseed_stage_a", 0)
            or pending_steps.get("csp_reseed_stage_b", 0)
        )
        csp_reseed_count = max(
            pending_steps.get("csp_reseed_count_stage_a", 0),
            pending_steps.get("csp_reseed_count_stage_b", 0),
        )
        jitter_fired = bool(
            pending_steps.get("jitter_stage_a", 0)
            or pending_steps.get("jitter_stage_b", 0)
        )
        jitter_count = max(
            pending_steps.get("jitter_count_stage_a", 0),
            pending_steps.get("jitter_count_stage_b", 0),
        )
        repair_fired = bool(
            pending_steps.get("repair_stage_a", 0)
            or pending_steps.get("repair_stage_b", 0)
        )
        repair_count = max(
            pending_steps.get("repair_count_stage_a", 0),
            pending_steps.get("repair_count_stage_b", 0),
        )
        repair_cost_max = max(
            pending_steps.get("repair_cost_max_stage_a", 0),
            pending_steps.get("repair_cost_max_stage_b", 0),
        )
        repair_cost_total = (
            pending_steps.get("repair_cost_total_stage_a", 0)
            + pending_steps.get("repair_cost_total_stage_b", 0)
        )
        repair_teleport_like_count = (
            pending_steps.get("repair_teleport_like_count_stage_a", 0)
            + pending_steps.get("repair_teleport_like_count_stage_b", 0)
        )
        repair_long_move_count = (
            pending_steps.get("repair_long_move_count_stage_a", 0)
            + pending_steps.get("repair_long_move_count_stage_b", 0)
        )
        repair_forced_visible_square_count = (
            pending_steps.get("repair_forced_visible_square_count_stage_a", 0)
            + pending_steps.get("repair_forced_visible_square_count_stage_b", 0)
        )
        repair_strict_rejected_count = (
            pending_steps.get("repair_strict_rejected_count_stage_a", 0)
            + pending_steps.get("repair_strict_rejected_count_stage_b", 0)
        )
        repair_strict_fallback_count = (
            pending_steps.get("repair_strict_fallback_count_stage_a", 0)
            + pending_steps.get("repair_strict_fallback_count_stage_b", 0)
        )
        if pending_steps.get("repair_worst_cost_stage_b", 0) >= pending_steps.get(
            "repair_worst_cost_stage_a", 0
        ):
            repair_worst_stage = "stage_b"
            repair_worst_cost = pending_steps.get("repair_worst_cost_stage_b", 0)
            repair_worst_piece = pending_steps.get("repair_worst_piece_stage_b")
            repair_worst_from = pending_steps.get("repair_worst_from_stage_b")
            repair_worst_to = pending_steps.get("repair_worst_to_stage_b")
            repair_worst_distance = pending_steps.get(
                "repair_worst_distance_stage_b", 0
            )
            repair_worst_one_move_legal = pending_steps.get(
                "repair_worst_one_move_legal_stage_b"
            )
        else:
            repair_worst_stage = "stage_a"
            repair_worst_cost = pending_steps.get("repair_worst_cost_stage_a", 0)
            repair_worst_piece = pending_steps.get("repair_worst_piece_stage_a")
            repair_worst_from = pending_steps.get("repair_worst_from_stage_a")
            repair_worst_to = pending_steps.get("repair_worst_to_stage_a")
            repair_worst_distance = pending_steps.get(
                "repair_worst_distance_stage_a", 0
            )
            repair_worst_one_move_legal = pending_steps.get(
                "repair_worst_one_move_legal_stage_a"
            )
        checkpoint_repair_fired = bool(
            pending_steps.get("checkpoint_repair_stage_a", 0)
            or pending_steps.get("checkpoint_repair_stage_b", 0)
        )
        checkpoint_repair_count = max(
            pending_steps.get("checkpoint_repair_count_stage_a", 0),
            pending_steps.get("checkpoint_repair_count_stage_b", 0),
        )
        checkpoint_repair_age = max(
            pending_steps.get("checkpoint_repair_age_stage_a", 0),
            pending_steps.get("checkpoint_repair_age_stage_b", 0),
        )
        latent_danger_probes: list[dict] = []
        if self._last_decision_view is not None:
            latent_danger_probes = _latent_ray_danger_probes(
                self._last_decision_view,
                self._belief,
            )
        record = {
            "tier1_move_count": self._tier1_move_count,
            "ply": ply,
            "decision_path": decision_path,
            "particle_count_pre_sample": particle_count_pre,
            "belief_unique_count": belief_unique,
            "particle_weight_profile": self._belief.particle_weight_profile(k=5),
            "move_chosen_uci": chosen.uci(),
            "chosen_move_belief_support": support_weight,
            "chosen_move_belief_support_count": support_count,
            "chosen_move_belief_support_unique": support_unique,
            "top_k_scores": [
                {"uci": uci, "score": score, "support": support}
                for uci, score, support in (top_k_scores or [])
            ],
            "decision_weight_modes": decision_weight_modes,
            "latent_danger_probe_count": len(latent_danger_probes),
            "latent_danger_probes": latent_danger_probes,
            "opp_remaining_counts": opp_counts,
            "csp_reseed_fired": csp_reseed_fired,
            "csp_reseed_count": csp_reseed_count,
            "jitter_fired": jitter_fired,
            "jitter_count": jitter_count,
            "repair_fired": repair_fired,
            "repair_count": repair_count,
            "repair_cost_max": repair_cost_max,
            "repair_cost_total": repair_cost_total,
            "repair_teleport_like_count": repair_teleport_like_count,
            "repair_long_move_count": repair_long_move_count,
            "repair_forced_visible_square_count": repair_forced_visible_square_count,
            "repair_strict_rejected_count": repair_strict_rejected_count,
            "repair_strict_fallback_count": repair_strict_fallback_count,
            "repair_worst_stage": repair_worst_stage if repair_worst_cost else None,
            "repair_worst_cost": repair_worst_cost,
            "repair_worst_piece": repair_worst_piece,
            "repair_worst_from": repair_worst_from,
            "repair_worst_to": repair_worst_to,
            "repair_worst_distance": repair_worst_distance,
            "repair_worst_one_move_legal": repair_worst_one_move_legal,
            "checkpoint_repair_fired": checkpoint_repair_fired,
            "checkpoint_repair_count": checkpoint_repair_count,
            "checkpoint_repair_age": checkpoint_repair_age,
        }
        if self._last_decision_view is not None:
            record.update(self._decision_audit(chosen, self._last_decision_view))
        # Carry over belief-step diagnostics from the most recent Stage A/B
        # observation updates, then clear so the next pick_move starts fresh.
        record.update(pending_steps)
        self._pending_belief_steps = {}
        self.trace_log.append(record)
        if self.verbose_belief_capture:
            self._append_belief_snapshot(
                ply=ply,
                snapshot_kind="decision",
                decision_path=decision_path,
                move=chosen,
                csp_reseed_fired=csp_reseed_fired,
                csp_reseed_count=csp_reseed_count,
                repair_fired=repair_fired,
                repair_count=repair_count,
                decision_weight_modes=decision_weight_modes,
                latent_danger_probes=latent_danger_probes,
            )

    def _belief_move_support_stats(self, move: chess.Move) -> tuple[float, int, int]:
        assert self._belief is not None
        total = sum(self._belief.weights)
        if total <= 0:
            return 0.0, 0, 0
        support_weight = 0.0
        support_count = 0
        support_fens: set[str] = set()
        for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False):
            if not particle.is_pseudo_legal(move):
                continue
            support_weight += weight
            support_count += 1
            support_fens.add(particle.fen())
        return support_weight / total, support_count, len(support_fens)

    def _decision_audit(
        self, chosen: chess.Move, view: PerspectiveView
    ) -> dict[str, float | int | str | None]:
        """Cheap per-move audit fields for belief consumption debugging."""
        own = view.perspective
        chosen_capture_value = 0
        chosen_target = view.visible_piece_map.get(chosen.to_square)
        if chosen_target is not None and chosen_target.color != own:
            chosen_capture_value = _MATERIAL_VALUE.get(chosen_target.piece_type, 0)

        best_capture: tuple[chess.Move, int] | None = None
        for move in view.own_legal_moves:
            target = view.visible_piece_map.get(move.to_square)
            if target is None or target.color == own:
                continue
            value = _MATERIAL_VALUE.get(target.piece_type, 0)
            if best_capture is None or value > best_capture[1]:
                best_capture = (move, value)

        mover = view.visible_piece_map.get(chosen.from_square)
        king_risk, risk_support, risk_unique = self._belief_immediate_king_risk(
            chosen, own
        )
        piece_risk = self._belief_immediate_piece_risk(chosen, own)
        best_capture_value = best_capture[1] if best_capture is not None else 0
        return {
            "chosen_piece": mover.symbol() if mover is not None else None,
            "chosen_piece_value": (
                _MATERIAL_VALUE.get(mover.piece_type, 0) if mover is not None else 0
            ),
            "chosen_visible_capture_value": chosen_capture_value,
            "best_visible_capture_uci": (
                best_capture[0].uci() if best_capture is not None else None
            ),
            "best_visible_capture_value": best_capture_value,
            "visible_capture_value_missed": max(
                0, best_capture_value - chosen_capture_value
            ),
            "chosen_move_king_capture_risk": king_risk,
            "chosen_move_piece_capture_risk": piece_risk,
            "chosen_move_risk_support_count": risk_support,
            "chosen_move_risk_support_unique": risk_unique,
        }

    def _decision_weight_mode_scores(
        self,
        evaluator,
        moves: list[chess.Move],
        *,
        max_clusters: int = 32,
        top_n: int = 5,
    ) -> dict:
        """Compare candidate move scores under multiple particle weight modes.

        Diagnostic-only bridge toward the forest evaluator. Current move choice
        still comes from `best_action`; this answers whether the same candidate
        list would pick a different move if distinct-world appearance or
        uniform-cluster voting replaced posterior mass.
        """
        assert self._belief is not None
        if not moves or not self._belief.particles:
            return {
                "sample": {
                    "selected_clusters": 0,
                    "total_unique_clusters": 0,
                    "max_clusters": max_clusters,
                },
                "mode_winners": {},
                "winner_disagreement": False,
                "modes": {},
            }

        total_weight = sum(self._belief.weights)
        if total_weight <= 0:
            return {
                "sample": {
                    "selected_clusters": 0,
                    "total_unique_clusters": 0,
                    "max_clusters": max_clusters,
                },
                "mode_winners": {},
                "winner_disagreement": False,
                "modes": {},
            }

        board_by_fen: dict[str, chess.Board] = {}
        weights_by_fen: dict[str, float] = {}
        counts_by_fen: dict[str, int] = {}
        for board, weight in zip(self._belief.particles, self._belief.weights, strict=False):
            fen = board.fen()
            board_by_fen.setdefault(fen, board)
            weights_by_fen[fen] = weights_by_fen.get(fen, 0.0) + weight
            counts_by_fen[fen] = counts_by_fen.get(fen, 0) + 1

        total_unique = len(board_by_fen)
        posterior_order = sorted(
            board_by_fen,
            key=lambda fen: (-(weights_by_fen[fen] / total_weight), fen),
        )
        appearance_order = sorted(
            board_by_fen,
            key=lambda fen: (-(counts_by_fen[fen] / len(self._belief.particles)), fen),
        )
        selected: list[str] = []
        seen: set[str] = set()
        for order in (posterior_order, appearance_order):
            for fen in order:
                if fen in seen:
                    continue
                seen.add(fen)
                selected.append(fen)
                if len(selected) >= max_clusters:
                    break
            if len(selected) >= max_clusters:
                break

        if not selected:
            return {
                "sample": {
                    "selected_clusters": 0,
                    "total_unique_clusters": total_unique,
                    "max_clusters": max_clusters,
                },
                "mode_winners": {},
                "winner_disagreement": False,
                "modes": {},
            }

        posterior_total = sum(weights_by_fen[fen] for fen in selected)
        appearance_total = sum(counts_by_fen[fen] for fen in selected)
        modes = {
            "posterior": {
                fen: weights_by_fen[fen] / posterior_total for fen in selected
            },
            "appearance": {
                fen: counts_by_fen[fen] / appearance_total for fen in selected
            },
            "uniform_distinct": {fen: 1.0 / len(selected) for fen in selected},
        }

        mode_rows: dict[str, list[dict]] = {}
        winners: dict[str, str | None] = {}
        for mode_name, masses in modes.items():
            scored: list[dict] = []
            for move in moves:
                support_mass = 0.0
                support_clusters = 0
                weighted_score = 0.0
                for fen, mass in masses.items():
                    board = board_by_fen[fen]
                    if not board.is_pseudo_legal(move):
                        continue
                    support_mass += mass
                    support_clusters += 1
                    weighted_score += mass * evaluator(
                        board, move, self._belief.perspective
                    )
                if support_mass <= 0:
                    continue
                scored.append(
                    {
                        "uci": move.uci(),
                        "score": weighted_score / support_mass,
                        "support_mass": support_mass,
                        "support_clusters": support_clusters,
                    }
                )
            scored.sort(key=lambda row: (-row["score"], row["uci"]))
            mode_rows[mode_name] = scored[:top_n]
            winners[mode_name] = scored[0]["uci"] if scored else None

        non_null_winners = {winner for winner in winners.values() if winner is not None}
        return {
            "sample": {
                "selected_clusters": len(selected),
                "total_unique_clusters": total_unique,
                "max_clusters": max_clusters,
            },
            "mode_winners": winners,
            "winner_disagreement": len(non_null_winners) > 1,
            "modes": mode_rows,
        }

    def _belief_immediate_king_risk(
        self, move: chess.Move, own: chess.Color
    ) -> tuple[float, int, int]:
        assert self._belief is not None
        total_weight = 0.0
        risk_weight = 0.0
        support_count = 0
        support_fens: set[str] = set()
        for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False):
            if not particle.is_pseudo_legal(move):
                continue
            support_count += 1
            support_fens.add(particle.fen())
            total_weight += weight
            sim = particle.copy()
            sim.push(move)
            king_sq = sim.king(own)
            if king_sq is None:
                risk_weight += weight
                continue
            sim.turn = not own
            if any(reply.to_square == king_sq for reply in sim.pseudo_legal_moves):
                risk_weight += weight
        if total_weight <= 0:
            return 0.0, support_count, len(support_fens)
        return risk_weight / total_weight, support_count, len(support_fens)

    def _belief_immediate_piece_risk(self, move: chess.Move, own: chess.Color) -> float:
        assert self._belief is not None
        total_weight = 0.0
        risk_weight = 0.0
        for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False):
            if not particle.is_pseudo_legal(move):
                continue
            sim = particle.copy()
            sim.push(move)
            moved = sim.piece_at(move.to_square)
            if moved is None or moved.color != own or moved.piece_type == chess.KING:
                continue
            total_weight += weight
            sim.turn = not own
            if any(reply.to_square == move.to_square for reply in sim.pseudo_legal_moves):
                risk_weight += weight
        if total_weight <= 0:
            return 0.0
        return risk_weight / total_weight

    def _opp_counts_for_json(self) -> dict[str, int]:
        assert self._belief is not None
        piece_type_names = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }
        return {
            piece_type_names[pt]: n
            for pt, n in self._belief.opp_remaining_counts.items()
        }

    def _append_belief_snapshot(
        self,
        *,
        ply: int,
        snapshot_kind: str,
        decision_path: str,
        move: chess.Move | None = None,
        csp_reseed_fired: bool | None = None,
        csp_reseed_count: int | None = None,
        repair_fired: bool | None = None,
        repair_count: int | None = None,
        decision_weight_modes: dict | None = None,
        latent_danger_probes: list[dict] | None = None,
    ) -> None:
        assert self._belief is not None
        if not self.verbose_belief_capture:
            return
        belief_unique = len({p.fen() for p in self._belief.particles})
        csp_fired = (
            bool(self._belief.last_csp_reseed_fired)
            if csp_reseed_fired is None
            else csp_reseed_fired
        )
        csp_count = (
            self._belief.last_csp_reseed_count
            if csp_reseed_count is None
            else csp_reseed_count
        )
        repair_did_fire = (
            bool(self._belief.last_repair_fired)
            if repair_fired is None
            else repair_fired
        )
        repair_n = (
            self._belief.last_repair_count
            if repair_count is None
            else repair_count
        )
        self.belief_log.append(
            {
                "ply": ply,
                "snapshot_kind": snapshot_kind,
                "decision_path": decision_path,
                "particle_count": len(self._belief.particles),
                "particle_count_unique": belief_unique,
                "move_chosen_uci": move.uci() if move is not None else None,
                "opp_remaining_counts": self._opp_counts_for_json(),
                "last_constraint_pruned": self._belief.last_constraint_pruned,
                "csp_reseed_fired": csp_fired,
                "csp_reseed_count": csp_count,
                "repair_fired": repair_did_fire,
                "repair_count": repair_n,
                "repair_cost_max": self._belief.last_repair_cost_max,
                "repair_cost_total": self._belief.last_repair_cost_total,
                "repair_moved_piece_count_max": (
                    self._belief.last_repair_moved_piece_count_max
                ),
                "repair_max_piece_distance": (
                    self._belief.last_repair_max_piece_distance
                ),
                "repair_long_move_count": self._belief.last_repair_long_move_count,
                "repair_teleport_like_count": (
                    self._belief.last_repair_teleport_like_count
                ),
                "repair_forced_visible_square_count": (
                    self._belief.last_repair_forced_visible_square_count
                ),
                "repair_unpaired_added_count": (
                    self._belief.last_repair_unpaired_added_count
                ),
                "repair_unpaired_removed_count": (
                    self._belief.last_repair_unpaired_removed_count
                ),
                "repair_worst_cost": self._belief.last_repair_worst_cost,
                "repair_worst_piece": self._belief.last_repair_worst_piece,
                "repair_worst_from": self._belief.last_repair_worst_from,
                "repair_worst_to": self._belief.last_repair_worst_to,
                "repair_worst_distance": self._belief.last_repair_worst_distance,
                "repair_worst_one_move_legal": (
                    self._belief.last_repair_worst_one_move_legal
                ),
                "repair_strict_rejected_count": (
                    self._belief.last_repair_strict_rejected_count
                ),
                "repair_strict_fallback_count": (
                    self._belief.last_repair_strict_fallback_count
                ),
                "checkpoint_repair_fired": bool(
                    self._belief.last_checkpoint_repair_fired
                ),
                "checkpoint_repair_count": (
                    self._belief.last_checkpoint_repair_count
                ),
                "checkpoint_repair_age": self._belief.last_checkpoint_repair_age,
                "checkpoint_repair_unique": (
                    self._belief.last_checkpoint_repair_unique
                ),
                "stage_b_repair_supplement_considered_count": (
                    self._belief.last_stage_b_repair_supplement_considered_count
                ),
                "stage_b_repair_supplement_dropped_count": (
                    self._belief.last_stage_b_repair_supplement_dropped_count
                ),
                "jitter_fired": bool(self._belief.last_jitter_fired),
                "jitter_count": self._belief.last_jitter_count,
                "hard_facts": self._belief.hard_fact_summary(),
                "marginal_field": _marginal_field_for_json(
                    self._belief.marginal_piece_field()
                ),
                "top_k_clusters": [
                    {"fen": fen, "weight": weight, "particle_count": count}
                    for fen, weight, count in self._belief.top_k_clusters()
                ],
                "particle_weight_profile": self._belief.particle_weight_profile(k=8),
                "decision_weight_modes": decision_weight_modes,
                "latent_danger_probe_count": len(latent_danger_probes or []),
                "latent_danger_probes": latent_danger_probes or [],
            }
        )

    def observe_own_move(self, move: chess.Move, observation: Observation) -> None:
        assert self._belief is not None
        # v0.9.4-pre1: track recent own (from, to) for anti-shuffle scoring.
        self._recent_own_moves.append((move.from_square, move.to_square))
        if len(self._recent_own_moves) > self.anti_shuffle_window:
            self._recent_own_moves = self._recent_own_moves[-self.anti_shuffle_window:]
        # v0.6.0: register the capture detected in pick_move (visible enemy at
        # the destination, or en-passant). Decrements opp_remaining_counts so
        # the next Stage B can prune particles hallucinating extra pieces.
        if self._pending_capture_type is not None:
            self._belief.register_capture(
                self._pending_capture_type,
                self._pending_capture_square,
            )
            self._pending_capture_type = None
            self._pending_capture_square = None
        before = len(self._belief.particles)
        before_unique = len({p.fen() for p in self._belief.particles})
        self._belief.update_after_own_move(move, observation)
        after = len(self._belief.particles)
        after_unique = len({p.fen() for p in self._belief.particles})
        self._pending_belief_steps["belief_pre_stage_a"] = before
        self._pending_belief_steps["belief_pre_stage_a_unique"] = before_unique
        self._pending_belief_steps["belief_post_stage_a"] = after
        self._pending_belief_steps["belief_post_stage_a_unique"] = after_unique
        self._pending_belief_steps["stage_a_pushed_count"] = (
            self._belief.last_stage_a_pushed_count
        )
        self._pending_belief_steps["stage_a_pushed_unique"] = (
            self._belief.last_stage_a_pushed_unique
        )
        self._pending_belief_steps["stage_a_consistent_count"] = (
            self._belief.last_stage_a_consistent_count
        )
        self._pending_belief_steps["stage_a_consistent_unique"] = (
            self._belief.last_stage_a_consistent_unique
        )
        self._pending_belief_steps["stage_a_repair_supplement_count"] = (
            self._belief.last_stage_a_repair_supplement_count
        )
        self._pending_belief_steps["stage_a_elapsed_ms"] = (
            self._belief.last_stage_a_elapsed_ms
        )
        self._pending_belief_steps["stage_a_filter_ms"] = (
            self._belief.last_stage_a_filter_ms
        )
        self._pending_belief_steps["stage_a_repair_ms"] = (
            self._belief.last_stage_a_repair_ms
        )
        self._pending_belief_steps["stage_a_csp_ms"] = (
            self._belief.last_stage_a_csp_ms
        )
        self._pending_belief_steps["stage_a_resample_ms"] = (
            self._belief.last_stage_a_resample_ms
        )
        self._pending_belief_steps["stage_a_reject_illegal"] = (
            self._belief.last_stage_a_reject_illegal
        )
        self._pending_belief_steps["stage_a_reject_observation"] = (
            self._belief.last_stage_a_reject_observation
        )
        self._pending_belief_steps["stage_a_reject_hard"] = (
            self._belief.last_stage_a_reject_hard
        )
        self._pending_belief_steps["csp_reseed_stage_a"] = (
            self._belief.last_csp_reseed_fired
        )
        self._pending_belief_steps["csp_reseed_count_stage_a"] = (
            self._belief.last_csp_reseed_count
        )
        self._pending_belief_steps["jitter_stage_a"] = self._belief.last_jitter_fired
        self._pending_belief_steps["jitter_count_stage_a"] = self._belief.last_jitter_count
        self._pending_belief_steps["repair_stage_a"] = self._belief.last_repair_fired
        self._pending_belief_steps["repair_count_stage_a"] = (
            self._belief.last_repair_count
        )
        self._pending_belief_steps["repair_cost_max_stage_a"] = (
            self._belief.last_repair_cost_max
        )
        self._pending_belief_steps["repair_cost_total_stage_a"] = (
            self._belief.last_repair_cost_total
        )
        self._pending_belief_steps["repair_moved_piece_count_max_stage_a"] = (
            self._belief.last_repair_moved_piece_count_max
        )
        self._pending_belief_steps["repair_max_piece_distance_stage_a"] = (
            self._belief.last_repair_max_piece_distance
        )
        self._pending_belief_steps["repair_long_move_count_stage_a"] = (
            self._belief.last_repair_long_move_count
        )
        self._pending_belief_steps["repair_teleport_like_count_stage_a"] = (
            self._belief.last_repair_teleport_like_count
        )
        self._pending_belief_steps["repair_forced_visible_square_count_stage_a"] = (
            self._belief.last_repair_forced_visible_square_count
        )
        self._pending_belief_steps["repair_unpaired_added_count_stage_a"] = (
            self._belief.last_repair_unpaired_added_count
        )
        self._pending_belief_steps["repair_unpaired_removed_count_stage_a"] = (
            self._belief.last_repair_unpaired_removed_count
        )
        self._pending_belief_steps["repair_worst_cost_stage_a"] = (
            self._belief.last_repair_worst_cost
        )
        self._pending_belief_steps["repair_worst_piece_stage_a"] = (
            self._belief.last_repair_worst_piece
        )
        self._pending_belief_steps["repair_worst_from_stage_a"] = (
            self._belief.last_repair_worst_from
        )
        self._pending_belief_steps["repair_worst_to_stage_a"] = (
            self._belief.last_repair_worst_to
        )
        self._pending_belief_steps["repair_worst_distance_stage_a"] = (
            self._belief.last_repair_worst_distance
        )
        self._pending_belief_steps["repair_worst_one_move_legal_stage_a"] = (
            self._belief.last_repair_worst_one_move_legal
        )
        self._pending_belief_steps["repair_strict_rejected_count_stage_a"] = (
            self._belief.last_repair_strict_rejected_count
        )
        self._pending_belief_steps["repair_strict_fallback_count_stage_a"] = (
            self._belief.last_repair_strict_fallback_count
        )
        self._pending_belief_steps["checkpoint_repair_stage_a"] = (
            self._belief.last_checkpoint_repair_fired
        )
        self._pending_belief_steps["checkpoint_repair_count_stage_a"] = (
            self._belief.last_checkpoint_repair_count
        )
        self._pending_belief_steps["checkpoint_repair_age_stage_a"] = (
            self._belief.last_checkpoint_repair_age
        )
        self._pending_belief_steps["checkpoint_repair_unique_stage_a"] = (
            self._belief.last_checkpoint_repair_unique
        )
        self._observed_ply += 1
        self._append_belief_snapshot(
            ply=self._observed_ply,
            snapshot_kind="after-own-move",
            decision_path="after-own-move",
            move=move,
        )

    def _real_king_threat_attackers(
        self,
        view: PerspectiveView,
        *,
        clear_threshold: float = 0.5,
    ) -> set[chess.Square]:
        """Visible enemy attackers of our king whose attack ray is plausibly clear in belief.

        The visibility-only board used by `_categorize_king_defense_moves`
        treats every visible enemy slider as if its attack ray were clear,
        because hidden blockers are absent from the synthetic board. In
        positions where a hidden defender (e.g., an opp piece the perspective
        hasn't seen yet) blocks the ray in the TRUE position, the king-defense
        tier fires on a phantom check and overrides better moves (queen-save,
        material grab, etc.).

        This method filters the raw set of visible attackers: for each
        slider-class attacker (bishop/rook/queen), if `clear_threshold`
        fraction of belief weight says the ray to the king is BLOCKED, the
        attacker is treated as phantom and excluded. Non-sliders (knight,
        pawn, king) can't be blocked and pass through unchanged.

        When the returned set is empty AND `_categorize_king_defense_moves`
        returned non-empty lists, the caller should skip the king-defense
        tier — the apparent check is phantom.
        """
        own_king_squares = {
            sq for sq, piece in view.visible_piece_map.items()
            if piece.color == view.perspective and piece.piece_type == chess.KING
        }
        if not own_king_squares:
            return set()
        raw_attackers = _visible_attackers_of_squares(view, own_king_squares)
        if not raw_attackers:
            return set()
        if self._belief is None or not self._belief.particles:
            return raw_attackers

        total_weight = sum(self._belief.weights) or 1.0
        # Belief-staleness threshold: if the visible attacker isn't present in
        # at least this fraction of belief particles (on the correct square,
        # correct piece type, correct color), the belief is out of sync with
        # visibility and we should NOT trust its blocker reasoning. Trust
        # visibility instead.
        attacker_in_belief_threshold = 0.5
        real: set[chess.Square] = set()
        for attacker_sq in raw_attackers:
            piece = view.visible_piece_map.get(attacker_sq)
            if piece is None:
                continue
            # Non-sliders can't be blocked by an intervening square — trust visibility.
            if piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                real.add(attacker_sq)
                continue
            # Stale-belief guard: if the belief doesn't even know the attacker
            # is on this square (e.g., test fixtures with reset+custom view,
            # or a position where belief hasn't propagated the latest opp
            # move), don't try to reason about hidden blockers. Trust the
            # visible attacker as a real threat.
            attacker_weight = 0.0
            for particle, w in zip(self._belief.particles, self._belief.weights, strict=False):
                p = particle.piece_at(attacker_sq)
                if p is not None and p.color == piece.color and p.piece_type == piece.piece_type:
                    attacker_weight += w
            if (attacker_weight / total_weight) < attacker_in_belief_threshold:
                real.add(attacker_sq)
                continue
            # Belief is in sync with the visible attacker. Now check belief
            # for blockers on the attack ray.
            for king_sq in own_king_squares:
                between = _ray_squares_between(attacker_sq, king_sq)
                if between is None:
                    continue
                clear_weight = 0.0
                for particle, w in zip(self._belief.particles, self._belief.weights, strict=False):
                    if all(particle.piece_at(sq) is None for sq in between):
                        clear_weight += w
                clear_frac = clear_weight / total_weight
                if clear_frac >= clear_threshold:
                    real.add(attacker_sq)
                    break
        return real

    def _belief_supports_move(self, move: chess.Move, threshold: float = 0.5) -> bool:
        """True iff `move` is pseudo-legal in at least `threshold` fraction of particles.

        Used to filter short-circuit candidates that would wipe belief on the
        next Stage A update (because no particle has `my_move` pseudo-legal,
        BeliefState.update_after_own_move drops every particle in step 1 — an
        unrecoverable collapse). Catches cases where the new visibility-grounded
        short-circuits pick moves that diverge sharply from belief's stale view
        of opp positions, e.g., sliding-piece moves through squares particles
        hallucinate as occupied.

        Threshold defaults to 0.5: only filter when a clear minority of
        particles agree. With <50% support the move is risky for belief
        survival; with ≥50% support belief will recover.

        Returns True when belief is empty (no signal).
        """
        if self._belief is None or not self._belief.particles:
            return True
        ok = sum(1 for p in self._belief.particles if p.is_pseudo_legal(move))
        return ok / len(self._belief.particles) >= threshold

    def _belief_veto_king_attack(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
        threshold: float = 0.05,
        king_move_threshold: float = 0.05,
    ) -> list[chess.Move]:
        """Drop king-defense candidates where >threshold fraction of particles
        place our king under attack after the move.

        Catches hidden discovered checks the visibility-only board can't see —
        e.g., capturing the visible attacker but unblocking a hidden bishop.
        For non-king moves, v0.7.22 uses a terminal-risk threshold. If more
        than 5% of supporting particles say the move leaves our king
        immediately capturable, prefer safer alternatives. A small hidden king
        line is not ordinary material uncertainty.

        For voluntary king moves, use a much lower threshold. g16/v0.7.13
        showed why: a king step into a low-probability hidden rook line can be
        immediately terminal, and should lose to safer alternatives instead of
        being treated like ordinary material uncertainty.

        Returns the subset of `candidates` not vetoed. Caller falls back to
        the full set when this returns empty (better to make some defense
        than none).
        """
        if self._belief is None or not self._belief.particles:
            return candidates
        own = view.perspective
        survivors: list[chess.Move] = []
        for move in candidates:
            mover = view.visible_piece_map.get(move.from_square)
            is_own_king_move = (
                mover is not None
                and mover.color == own
                and mover.piece_type == chess.KING
            )
            allowed_risk = king_move_threshold if is_own_king_move else threshold
            attacked = 0
            total = 0
            for particle in self._belief.particles:
                if not particle.is_pseudo_legal(move):
                    continue
                total += 1
                sim = particle.copy()
                sim.push(move)
                king_sq = sim.king(own)
                if king_sq is None:
                    attacked += 1
                    continue
                sim.turn = not own
                if any(m.to_square == king_sq for m in sim.pseudo_legal_moves):
                    attacked += 1
            if total == 0 or attacked / total <= allowed_risk:
                survivors.append(move)
        return survivors

    def _belief_lowest_king_attack_risk(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
        *,
        king_move_tiebreak_band: float = 0.03,
    ) -> list[chess.Move]:
        """Return the candidate subset with the lowest terminal king risk.

        The ordinary king-risk veto returns an empty list when every candidate
        is above budget. Older code then fell back to the full candidate set,
        which discarded the only useful signal belief had. In that forced-risk
        case, keep the least-bad moves instead. If a voluntary king move is
        only marginally lower-risk than a non-king move, prefer the non-king
        move: fog risk estimates at this granularity are noisy, and walking the
        king through hidden space creates sequential terminal exposure.
        """
        if self._belief is None or not self._belief.particles or not candidates:
            return candidates

        own = view.perspective
        scored: list[tuple[chess.Move, float, int, bool]] = []
        unsupported: list[chess.Move] = []
        for move in candidates:
            risk, support_count, _ = self._belief_immediate_king_risk(move, own)
            if support_count <= 0:
                unsupported.append(move)
                continue
            mover = view.visible_piece_map.get(move.from_square)
            is_king_move = (
                mover is not None
                and mover.color == own
                and mover.piece_type == chess.KING
            )
            scored.append((move, risk, support_count, is_king_move))

        if not scored:
            return unsupported or candidates

        best_risk = min(risk for _, risk, _, _ in scored)
        eps = 1e-9
        near_best = [
            row
            for row in scored
            if row[1] <= best_risk + king_move_tiebreak_band + eps
        ]
        non_king_near_best = [row for row in near_best if not row[3]]
        if non_king_near_best:
            best_non_king_risk = min(risk for _, risk, _, _ in non_king_near_best)
            return [
                move
                for move, risk, _, _ in non_king_near_best
                if risk <= best_non_king_risk + eps
            ]
        return [move for move, risk, _, _ in scored if risk <= best_risk + eps]

    def _belief_veto_bad_capture_trade(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
        threshold: float = 0.35,
    ) -> list[chess.Move]:
        """Drop material-shortcut captures that belief says are likely bad trades.

        `visible-minor-rook-capture` is meant for hanging material. In fog,
        "not visibly defended" is not enough: particles may strongly believe a
        hidden rook/bishop/king can recapture the destination. If the captured
        piece is not more valuable than the attacker, and >threshold of
        supporting particles allow an immediate recapture, let main-eval decide
        instead of auto-firing the shortcut.
        """
        if self._belief is None or not self._belief.particles:
            return candidates

        survivors: list[chess.Move] = []
        for move in candidates:
            attacker = view.visible_piece_map.get(move.from_square)
            target = view.visible_piece_map.get(move.to_square)
            if (
                attacker is None
                or target is None
                or attacker.color != view.perspective
                or target.color == view.perspective
            ):
                survivors.append(move)
                continue

            attacker_value = _MATERIAL_VALUE.get(attacker.piece_type, 0)
            target_value = _MATERIAL_VALUE.get(target.piece_type, 0)
            if target_value > attacker_value:
                survivors.append(move)
                continue

            # Recapture exemption — narrowed to EQUAL-value trades only.
            #
            # If opp just captured one of our pieces on this square, recapturing
            # *at equal value* is recovering material — the alternative (don't
            # recapture) leaves us strictly down material. Fixes game-1/game-2
            # ply-12 (P captures P with possible-N-recapture: veto wrongly fired).
            #
            # For DOWN-trade recaptures (B captures defended P, etc.), the math
            # flips: recapturing trades a higher-value piece for a lower-value
            # one, *increasing* our loss. Game-3 ply-23 caught this: the bishop
            # took a pawn defended by a knight, losing the bishop. The veto
            # should still fire on down-trades even when they're recaptures.
            attacker_color = attacker.color  # type: ignore[union-attr]
            _ = attacker_color  # silence unused warning if optimizer complains
            if (
                target_value == attacker_value
                and move.to_square == self._opp_last_capture_target
            ):
                survivors.append(move)
                continue

            # Tighten threshold by attacker/target value gap: a down-trade
            # bishop-for-pawn at "only" 10% recapture probability has positive
            # EV (+0.7) but high variance (1 vs −2) — a strong human declines.
            # Game-3 ply-23 caught this with belief at 10% (below the default
            # 0.35 threshold); the engine took Bxe4 into a hidden knight and
            # lost the bishop. Equal-value trades stay at the default — only
            # down-trades (attacker > target) tighten progressively.
            value_gap = max(0, attacker_value - target_value)
            if value_gap == 0:
                effective_threshold = threshold  # 0.35 default — fine for P-P, N-B, etc.
            else:
                # Approximately EV-neutral divided by 2 for a half-Kelly-ish stance.
                # _MATERIAL_VALUE is in pawn units (1-9), so gap is small ints.
                # Gap 2 (B/N for P) → ~0.07. Gap 4 (R for P) → ~0.04. Gap 8 (Q for P) → ~0.02.
                effective_threshold = max(0.02, 0.35 / (1.0 + 2.0 * value_gap))

            recapturable = 0
            total = 0
            for particle in self._belief.particles:
                if not particle.is_pseudo_legal(move):
                    continue
                total += 1
                sim = particle.copy()
                sim.push(move)
                sim.turn = not view.perspective
                if any(reply.to_square == move.to_square for reply in sim.pseudo_legal_moves):
                    recapturable += 1

            if total == 0 or recapturable / total <= effective_threshold:
                survivors.append(move)
        return survivors

    def _veto_captures_defended_by_visible_king(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
    ) -> list[chess.Move]:
        """Drop captures where the destination is defended by the visible opponent king.

        In FoW, particles may not consistently model the king's position, so the
        aggregated Stockfish score underweights the certain king-recapture. If the
        destination is visibly attacked by the opponent king and moving there would
        be an unfavorable exchange (mover > captured), veto the move.
        """
        own = view.perspective
        opp = not own
        king_sq: chess.Square | None = None
        for sq, piece in view.visible_piece_map.items():
            if piece.color == opp and piece.piece_type == chess.KING:
                king_sq = sq
                break
        if king_sq is None:
            return candidates

        king_attacks = chess.BB_KING_ATTACKS[king_sq]
        survivors: list[chess.Move] = []
        for move in candidates:
            captured = view.visible_piece_map.get(move.to_square)
            if captured is None or captured.color == own:
                survivors.append(move)
                continue
            if not (chess.BB_SQUARES[move.to_square] & king_attacks):
                survivors.append(move)
                continue
            mover = view.visible_piece_map.get(move.from_square)
            mover_val = (
                _MATERIAL_VALUE.get(mover.piece_type, 1000)
                if mover is not None and mover.color == own
                else 1000
            )
            captured_val = _MATERIAL_VALUE.get(captured.piece_type, 0)
            if mover_val <= captured_val:
                survivors.append(move)  # even or winning exchange — allow
        return survivors or candidates

    def _belief_veto_queen_fog_risk(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
        threshold: float = 0.20,
    ) -> list[chess.Move]:
        """Drop queen moves that belief says are likely immediately capturable.

        Regression shape: early v0.7 replay gate still chose `Qd5-e4`, moving
        the queen onto a square controlled by a hidden knight in a material
        minority of particles. For queens, even a 20% immediate-loss risk is
        enough to reject a speculative fog move. Keep this guardrail narrow:
        non-queen moves are handled by existing paths. Visible queen captures
        of low-value material still go through this filter because `Qx pawn`
        can be a losing tactic in fog.
        """
        if self._belief is None or not self._belief.particles:
            return candidates

        survivors: list[chess.Move] = []
        for move in candidates:
            mover = view.visible_piece_map.get(move.from_square)
            if (
                mover is None
                or mover.color != view.perspective
                or mover.piece_type != chess.QUEEN
            ):
                survivors.append(move)
                continue

            target = view.visible_piece_map.get(move.to_square)
            if target is not None and target.color != view.perspective:
                target_value = _MATERIAL_VALUE.get(target.piece_type, 0)
                if target_value >= _MATERIAL_VALUE[chess.QUEEN]:
                    # Queen-vs-queen captures are handled by the dedicated
                    # queen-capture path before this fallback filter.
                    survivors.append(move)
                    continue
            if move.to_square in _squares_attacked_by_visible_enemy_full(view):
                continue

            recapturable = 0
            total = 0
            for particle in self._belief.particles:
                if not particle.is_pseudo_legal(move):
                    continue
                total += 1
                sim = particle.copy()
                sim.push(move)
                landed = sim.piece_at(move.to_square)
                if (
                    landed is None
                    or landed.color != view.perspective
                    or landed.piece_type != chess.QUEEN
                ):
                    continue
                sim.turn = not view.perspective
                if any(reply.to_square == move.to_square for reply in sim.pseudo_legal_moves):
                    recapturable += 1

            if total == 0 or recapturable / total <= threshold:
                survivors.append(move)
        return survivors

    def _belief_queen_king_pressure_moves(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
        *,
        min_pressure: float = 0.70,
        max_recapture_risk: float = 0.15,
        min_unique_particles: int = 4,
    ) -> list[chess.Move]:
        """Queen moves that safely attack the believed opponent king.

        This is a narrow Fog-of-War pressure rule for positions where the
        opponent king is not visible but belief strongly preserves its home or
        tracked square. A queen move that attacks that king in most particles
        and is not immediately recapturable deserves to beat quiet development.
        Skip immediately after generic CSP and on near-singleton belief sets:
        those rows need main-eval or review, not another tactical override.
        """
        if self._belief is None or not self._belief.particles:
            return []
        if (
            self._pending_belief_steps.get("csp_reseed_stage_a", 0)
            or self._pending_belief_steps.get("csp_reseed_stage_b", 0)
        ):
            return []
        if len({particle.fen() for particle in self._belief.particles}) < min_unique_particles:
            return []

        scored: list[tuple[chess.Move, float, float]] = []
        for move in candidates:
            mover = view.visible_piece_map.get(move.from_square)
            target = view.visible_piece_map.get(move.to_square)
            if (
                mover is None
                or mover.color != view.perspective
                or mover.piece_type != chess.QUEEN
                or (target is not None and target.color != view.perspective)
            ):
                continue

            total = 0
            pressures = 0
            recaptures = 0
            for particle in self._belief.particles:
                if not particle.is_pseudo_legal(move):
                    continue
                total += 1
                sim = particle.copy()
                sim.push(move)
                opp_king = sim.king(not view.perspective)
                if opp_king is not None and move.to_square in sim.attackers(
                    view.perspective, opp_king
                ):
                    pressures += 1
                sim.turn = not view.perspective
                if any(reply.to_square == move.to_square for reply in sim.pseudo_legal_moves):
                    recaptures += 1

            if total == 0:
                continue
            pressure = pressures / total
            recapture_risk = recaptures / total
            if pressure >= min_pressure and recapture_risk <= max_recapture_risk:
                scored.append((move, pressure, recapture_risk))

        if not scored:
            return []
        best_pressure = max(pressure for _, pressure, _ in scored)
        pressure_top = [
            row for row in scored if row[1] >= best_pressure - 1e-9
        ]
        best_risk = min(risk for _, _, risk in pressure_top)
        return [move for move, _, risk in pressure_top if risk <= best_risk + 1e-9]

    def _belief_high_value_piece_save_moves(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
        *,
        min_before_risk: float = 4.0,
        min_improvement: float = 1.0,
    ) -> list[chess.Move]:
        """Moves that reduce belief-weighted attacks on own queens/rooks.

        Visible queen-save handles directly observed attacks. This catches the
        FOW/belief version: particles place an enemy queen, bishop, rook, or
        knight attacking an own rook/queen even when the attacker is not in the
        current visibility map. Rank by lowest remaining threatened value, then
        cheapest mover, so pawn blocks like `f2f3` beat shuffling material.
        """
        if self._belief is None or not self._belief.particles:
            return []

        own = view.perspective

        def threatened_value(board: chess.Board) -> float:
            sim = board.copy()
            sim.turn = not own
            threatened: set[chess.Square] = set()
            for reply in sim.pseudo_legal_moves:
                target = sim.piece_at(reply.to_square)
                if (
                    target is not None
                    and target.color == own
                    and target.piece_type in (chess.QUEEN, chess.ROOK)
                ):
                    threatened.add(reply.to_square)
            return float(
                sum(
                    _MATERIAL_VALUE[sim.piece_at(sq).piece_type]
                    for sq in threatened
                    if sim.piece_at(sq) is not None
                )
            )

        before = sum(
            threatened_value(particle) * weight
            for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False)
        )
        if before < min_before_risk:
            return []

        scored: list[tuple[chess.Move, float, int]] = []
        for move in candidates:
            total_weight = 0.0
            post = 0.0
            for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False):
                if not particle.is_pseudo_legal(move):
                    continue
                sim = particle.copy()
                sim.push(move)
                post += threatened_value(sim) * weight
                total_weight += weight
            if total_weight <= 0:
                continue
            # Normalize when only a subset of particles support the move.
            post = post / total_weight
            if before - post < min_improvement:
                continue
            mover = view.visible_piece_map.get(move.from_square)
            mover_value = (
                _MATERIAL_VALUE.get(mover.piece_type, 1000)
                if mover is not None and mover.color == own
                else 1000
            )
            scored.append((move, post, mover_value))

        if not scored:
            return []
        best_post = min(post for _, post, _ in scored)
        best = [row for row in scored if row[1] <= best_post + 1e-9]
        best_mover = min(mover_value for _, _, mover_value in best)
        return [move for move, _, mover_value in best if mover_value == best_mover]

    def _belief_piece_save_moves(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
        *,
        piece_types: tuple[chess.PieceType, ...] = (
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
        ),
        min_before_risk: float = 3.0,
        min_improvement: float = 1.0,
    ) -> list[chess.Move]:
        """Moves that reduce belief-weighted attacks on own material.

        This is the non-terminal sibling of king safety and queen/rook saves.
        The perspective may not literally see the attacker, but if particles
        consistently say a knight, bishop, or rook is hanging, prioritize
        saving it before unrelated material moves.

        Landing-square safety is enforced in two layers:

        1. Hard filter — never escape to a square a visible enemy piece attacks,
           unless that square holds a visible enemy piece worth at least as much
           as the piece being moved (a favorable or even capture). This catches
           the ply-65 class: queen "saved" to a pawn-defended square.

        2. Soft ranking — among surviving candidates, compute an expected-loss
           score from particle-inferred recaptures:
               landing_risk = Σ weight_i × (mover_value − attacker_value_i)
                              clamped to bad exchanges only (attacker_val < mover_val)
           Particle pawns are weighted 1.5× because pawn positions are
           geometrically stable across particles (pawns move one square
           forward only) and a pawn recapture is always a losing exchange
           for a non-pawn piece. Candidates are tiered by risk before the
           existing (post, mover_value, fog_depth) criteria are applied.
        """
        if self._belief is None or not self._belief.particles:
            return []

        own = view.perspective
        visible_attacks = _squares_attacked_by_visible_enemy_full(view)

        def threatened_value(board: chess.Board) -> float:
            sim = board.copy()
            sim.turn = not own
            threatened: set[chess.Square] = set()
            for reply in sim.pseudo_legal_moves:
                target = sim.piece_at(reply.to_square)
                if (
                    target is not None
                    and target.color == own
                    and target.piece_type in piece_types
                ):
                    threatened.add(reply.to_square)
            return float(
                sum(
                    _MATERIAL_VALUE[sim.piece_at(sq).piece_type]
                    for sq in threatened
                    if sim.piece_at(sq) is not None
                )
            )

        before = sum(
            threatened_value(particle) * weight
            for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False)
        )
        if before < min_before_risk:
            return []

        # scored: (move, post_threat, mover_value, fog_depth, landing_risk)
        scored: list[tuple[chess.Move, float, int, int, float]] = []
        for move in candidates:
            mover = view.visible_piece_map.get(move.from_square)
            mover_value = (
                _MATERIAL_VALUE.get(mover.piece_type, 1000)
                if mover is not None and mover.color == own
                else 1000
            )

            # Layer 1: hard filter on visible attacks.
            if move.to_square in visible_attacks:
                target = view.visible_piece_map.get(move.to_square)
                # Allow only if we're capturing a piece worth at least as much
                # as what we're moving (even or favorable exchange).
                if (
                    target is None
                    or target.color == own
                    or _MATERIAL_VALUE.get(target.piece_type, 0) < mover_value
                ):
                    continue

            total_weight = 0.0
            post = 0.0
            landing_risk_weight = 0.0

            for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False):
                if not particle.is_pseudo_legal(move):
                    continue
                sim = particle.copy()
                sim.push(move)
                post += threatened_value(sim) * weight
                total_weight += weight

                # Layer 2: particle recapture risk on the landing square.
                moved_piece = sim.piece_at(move.to_square)
                if moved_piece is None or moved_piece.color != own:
                    continue
                sim.turn = not own
                for reply in sim.pseudo_legal_moves:
                    if reply.to_square != move.to_square:
                        continue
                    attacker = sim.piece_at(reply.from_square)
                    if attacker is None:
                        continue
                    attacker_val = _MATERIAL_VALUE.get(attacker.piece_type, 0)
                    if attacker_val >= mover_value:
                        break  # even/favorable capture — no expected loss
                    loss = mover_value - attacker_val
                    # Pawn attacks weighted 1.5×: pawn positions are geometrically
                    # stable in particles and always produce losing exchanges.
                    pawn_mult = 1.5 if attacker.piece_type == chess.PAWN else 1.0
                    landing_risk_weight += weight * loss * pawn_mult
                    break  # one recapture per particle suffices

            if total_weight <= 0:
                continue
            post /= total_weight
            if before - post < min_improvement:
                continue
            landing_risk = landing_risk_weight / total_weight
            # Prefer captures over retreats: credit visible enemy piece on
            # the landing square so Nxh4 ranks above Ng6h8 when threat
            # reduction and risk are otherwise equal. Mirrors _visible_piece_save_moves.
            captured = view.visible_piece_map.get(move.to_square)
            capture_gain = (
                _MATERIAL_VALUE.get(captured.piece_type, 0)
                if captured is not None and captured.color != own
                else 0
            )
            scored.append((move, post, mover_value, _fog_depth(move.to_square, own), landing_risk, capture_gain))

        if not scored:
            return []

        # Tier by landing risk, then apply existing criteria within each tier.
        # Thresholds: ≤0.3 = safe, ≤1.5 = moderate, >1.5 = risky.
        # Tightened from (0.5, 2.0): the old bounds let the engine save a
        # rook to a ~100%-bishop-attacked square (risk≈2.0 → "moderate") or
        # a knight to a 30%-pawn-attacked square (risk≈0.9 → "moderate").
        def _risk_tier(risk: float) -> int:
            if risk <= 0.3:
                return 0
            if risk <= 1.5:
                return 1
            return 2

        best_tier = min(_risk_tier(r) for _, _, _, _, r, _ in scored)
        in_tier = [row for row in scored if _risk_tier(row[4]) == best_tier]

        best_post = min(post for _, post, _, _, _, _ in in_tier)
        best = [row for row in in_tier if row[1] <= best_post + 1e-9]
        # Captures beat retreats among equal-quality saves.
        best_gain = max(gain for _, _, _, _, _, gain in best)
        best = [row for row in best if row[5] == best_gain]
        best_mover = min(mv for _, _, mv, _, _, _ in best)
        best = [row for row in best if row[2] == best_mover]
        best_depth = min(d for _, _, _, d, _, _ in best)
        return [move for move, _, _, d, _, _ in best if d == best_depth]

    def _material_edge_estimate(self, view: PerspectiveView) -> float:
        """Approximate (own_material - expected_opp_material) in pawn units.

        Own material is exact (we see our pieces). Opp material is the
        belief-weighted mean across particles — each particle is a full-board
        hypothesis, so summing opp material on each and weighting by particle
        weight gives the belief-implied opp material. This is much sharper
        than opp_remaining_counts (which is a never-decreased-by-attrition
        upper bound that says ~39 forever if we haven't captured pieces).
        """
        own = sum(
            _MATERIAL_VALUE.get(p.piece_type, 0)
            for p in view.visible_piece_map.values()
            if p.color == view.perspective and p.piece_type != chess.KING
        )
        opp = 0.0
        if self._belief is not None and self._belief.particles:
            total_w = 0.0
            opp_w = 0.0
            opp_color = not view.perspective
            for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False):
                pmat = sum(
                    _MATERIAL_VALUE.get(p.piece_type, 0)
                    for p in particle.piece_map().values()
                    if p.color == opp_color and p.piece_type != chess.KING
                )
                opp_w += weight * pmat
                total_w += weight
            if total_w > 0:
                opp = opp_w / total_w
        return own - opp

    def _is_forward_progress(
        self, move: chess.Move, perspective: chess.Color
    ) -> bool:
        """Heuristic: does this move advance the piece toward opp's side?

        From white's POV, ranks increase toward black's home; from black's POV,
        ranks decrease. The check is rank-monotonic — captures and pawn pushes
        toward the opp side count; lateral moves and retreats don't.
        """
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        if perspective == chess.WHITE:
            return to_rank > from_rank
        return to_rank < from_rank

    def _cached_visibility_count(self, view: PerspectiveView) -> int:
        """Pre-move count of squares visible to our perspective. Used as the
        baseline for info-reveal deltas in the post-score loop."""
        return len(view.visible_squares)

    def _move_visibility_delta(
        self,
        move: chess.Move,
        view: PerspectiveView,
        pre_count: int,
    ) -> int:
        """Number of newly-visible squares this move would unlock.

        Visibility depends only on OWN piece positions (squares attacked /
        occupied by them). We construct a partial board from the visible
        piece map, apply the move, and measure the new visibility count.
        Sliding pieces into fog reveal the most; pawn pushes reveal one row.
        """
        partial = chess.Board.empty()
        for sq, piece in view.visible_piece_map.items():
            partial.set_piece_at(sq, piece)
        partial.turn = view.perspective
        if not partial.is_pseudo_legal(move):
            return 0
        partial.push(move)
        new_count = len(visible_squares(partial, view.perspective))
        return max(0, new_count - pre_count)

    def _compute_capture_risk_map(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
    ) -> dict[chess.Move, float]:
        """Belief-weighted P(landed own piece is recapturable) per move.

        v0.9.4-pre1 stopgap helper. Scoped to non-pawn / non-king moves; for
        pawns/kings the existing king-safety + en-passant vetoes already do
        the work and pawn recapture is a normal trade. Returns {} for moves
        the strategy doesn't apply the soft penalty to, so callers default
        to 0.0 risk via .get().
        """
        if self._belief is None or not self._belief.particles:
            return {}
        own = view.perspective
        risk_map: dict[chess.Move, float] = {}
        for move in candidates:
            mover = view.visible_piece_map.get(move.from_square)
            if (
                mover is None
                or mover.color != own
                or mover.piece_type in (chess.PAWN, chess.KING)
            ):
                continue
            # Captures where the target is equal-or-higher value are trades
            # the existing tier handles; skip them in the soft penalty too.
            target = view.visible_piece_map.get(move.to_square)
            if (
                target is not None
                and target.color != own
                and _MATERIAL_VALUE.get(target.piece_type, 0)
                >= _MATERIAL_VALUE.get(mover.piece_type, 0)
            ):
                continue
            risk_weight = 0.0
            support = 0.0
            for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False):
                if not particle.is_pseudo_legal(move):
                    continue
                sim = particle.copy()
                sim.push(move)
                landed = sim.piece_at(move.to_square)
                if landed is None or landed.color != own:
                    continue
                support += weight
                sim.turn = not own
                if any(reply.to_square == move.to_square for reply in sim.pseudo_legal_moves):
                    risk_weight += weight
            if support > 0:
                risk_map[move] = risk_weight / support
        return risk_map

    def _belief_veto_piece_fog_risk(
        self,
        candidates: list[chess.Move],
        view: PerspectiveView,
        *,
        max_capture_risk: float = 0.25,
    ) -> list[chess.Move]:
        """Drop non-pawn piece moves into likely immediate capture.

        Generalizes the queen-specific fog-risk veto to rooks/minors. It is
        intentionally a veto, not a chooser: if every candidate is risky the
        caller can fall back to the original candidate set.
        """
        if self._belief is None or not self._belief.particles:
            return candidates

        own = view.perspective
        visible_attacks = _squares_attacked_by_visible_enemy_full(view)
        survivors: list[chess.Move] = []
        for move in candidates:
            mover = view.visible_piece_map.get(move.from_square)
            if (
                mover is None
                or mover.color != own
                or mover.piece_type in (chess.PAWN, chess.KING)
            ):
                survivors.append(move)
                continue

            target = view.visible_piece_map.get(move.to_square)
            if (
                target is not None
                and target.color != own
                and _MATERIAL_VALUE.get(target.piece_type, 0)
                >= _MATERIAL_VALUE.get(mover.piece_type, 0)
            ):
                survivors.append(move)
                continue

            if move.to_square in visible_attacks:
                continue

            risk_weight = 0.0
            support = 0.0
            for particle, weight in zip(self._belief.particles, self._belief.weights, strict=False):
                if not particle.is_pseudo_legal(move):
                    continue
                sim = particle.copy()
                sim.push(move)
                moved = sim.piece_at(move.to_square)
                if moved is None or moved.color != own:
                    continue
                support += weight
                sim.turn = not own
                if any(reply.to_square == move.to_square for reply in sim.pseudo_legal_moves):
                    risk_weight += weight
            if support <= 0:
                survivors.append(move)
                continue
            if risk_weight / support <= max_capture_risk:
                survivors.append(move)

        return survivors

    def _detect_capture(
        self,
        move: chess.Move,
        view: PerspectiveView,
    ) -> tuple[chess.PieceType, chess.Square] | None:
        """Return opp piece type/square captured by `move`, or None.

        Two cases (FOW visibility-grounded — only fires on captures we observe):
          1. Direct capture: pre-move `view.visible_piece_map[move.to_square]`
             is an enemy piece. Squares we see are guaranteed accurate, so any
             enemy piece there is what the move captures.
          2. En-passant: own pawn moves diagonally to an empty (per visibility)
             square. The captured pawn is on the file of `to_square`, on the
             rank we came from — and is always visible (adjacent diagonal).
             Captured piece type is PAWN.

        Hidden captures (moving into fog and hitting a hidden enemy) return
        None: no observation evidence, so v0.6.0 leaves the bound conservative.
        """
        own_color = view.perspective
        target = view.visible_piece_map.get(move.to_square)
        if target is not None and target.color != own_color:
            return target.piece_type, move.to_square

        from_piece = view.visible_piece_map.get(move.from_square)
        if (
            from_piece is not None
            and from_piece.piece_type == chess.PAWN
            and chess.square_file(move.from_square) != chess.square_file(move.to_square)
            and target is None
        ):
            ep_capture_sq = chess.square(
                chess.square_file(move.to_square),
                chess.square_rank(move.from_square),
            )
            ep_target = view.visible_piece_map.get(ep_capture_sq)
            if (
                ep_target is not None
                and ep_target.color != own_color
                and ep_target.piece_type == chess.PAWN
            ):
                return chess.PAWN, ep_capture_sq

        return None

    def _stage_pending_capture(self, move: chess.Move, view: PerspectiveView) -> None:
        capture = self._detect_capture(move, view)
        if capture is None:
            self._pending_capture_type = None
            self._pending_capture_square = None
            return
        self._pending_capture_type, self._pending_capture_square = capture

    def observe_opp_move(self, observation: Observation) -> None:
        assert self._belief is not None
        # Track where opp just captured one of our pieces — read by
        # _belief_veto_bad_capture_trade to exempt recaptures from the
        # equal-value-trade veto.
        if observation.own_capture_square is not None:
            self._opp_last_capture_target = observation.own_capture_square
        elif observation.opp_capture_landing_square is not None:
            # En-passant-like edge: capture target differs from landing.
            # Treat the landing square as the recapture target.
            self._opp_last_capture_target = observation.opp_capture_landing_square
        else:
            self._opp_last_capture_target = None
        before = len(self._belief.particles)
        before_unique = len({p.fen() for p in self._belief.particles})
        self._belief.update_after_opp_move(observation)
        after = len(self._belief.particles)
        after_unique = len({p.fen() for p in self._belief.particles})
        self._pending_belief_steps["belief_pre_stage_b"] = before
        self._pending_belief_steps["belief_pre_stage_b_unique"] = before_unique
        self._pending_belief_steps["belief_post_stage_b"] = after
        self._pending_belief_steps["belief_post_stage_b_unique"] = after_unique
        self._pending_belief_steps["stage_b_primary_count"] = (
            self._belief.last_stage_b_primary_count
        )
        self._pending_belief_steps["stage_b_primary_unique"] = (
            self._belief.last_stage_b_primary_unique
        )
        self._pending_belief_steps["stage_b_constraint_count"] = (
            self._belief.last_stage_b_constraint_count
        )
        self._pending_belief_steps["stage_b_constraint_unique"] = (
            self._belief.last_stage_b_constraint_unique
        )
        self._pending_belief_steps["stage_b_repair_supplement_count"] = (
            self._belief.last_stage_b_repair_supplement_count
        )
        self._pending_belief_steps["stage_b_repair_supplement_considered_count"] = (
            self._belief.last_stage_b_repair_supplement_considered_count
        )
        self._pending_belief_steps["stage_b_repair_supplement_dropped_count"] = (
            self._belief.last_stage_b_repair_supplement_dropped_count
        )
        self._pending_belief_steps["stage_b_elapsed_ms"] = (
            self._belief.last_stage_b_elapsed_ms
        )
        self._pending_belief_steps["stage_b_expand_ms"] = (
            self._belief.last_stage_b_expand_ms
        )
        self._pending_belief_steps["stage_b_repair_ms"] = (
            self._belief.last_stage_b_repair_ms
        )
        self._pending_belief_steps["stage_b_csp_ms"] = (
            self._belief.last_stage_b_csp_ms
        )
        self._pending_belief_steps["stage_b_resample_ms"] = (
            self._belief.last_stage_b_resample_ms
        )
        self._pending_belief_steps["stage_b_expanded_count"] = (
            self._belief.last_stage_b_expanded_count
        )
        self._pending_belief_steps["stage_b_obs_checked_count"] = (
            self._belief.last_stage_b_obs_checked_count
        )
        self._pending_belief_steps["stage_b_reject_observation"] = (
            self._belief.last_stage_b_reject_observation
        )
        self._pending_belief_steps["stage_b_reject_hard"] = (
            self._belief.last_stage_b_reject_hard
        )
        self._pending_belief_steps["stage_b_reject_count"] = (
            self._belief.last_stage_b_reject_count
        )
        # v0.6.0: how many expanded particles the count-constraint filter killed.
        self._pending_belief_steps["constraint_pruned_stage_b"] = (
            self._belief.last_constraint_pruned
        )
        self._pending_belief_steps["csp_reseed_stage_b"] = (
            self._belief.last_csp_reseed_fired
        )
        self._pending_belief_steps["csp_reseed_count_stage_b"] = (
            self._belief.last_csp_reseed_count
        )
        self._pending_belief_steps["jitter_stage_b"] = self._belief.last_jitter_fired
        self._pending_belief_steps["jitter_count_stage_b"] = self._belief.last_jitter_count
        self._pending_belief_steps["repair_stage_b"] = self._belief.last_repair_fired
        self._pending_belief_steps["repair_count_stage_b"] = (
            self._belief.last_repair_count
        )
        self._pending_belief_steps["repair_cost_max_stage_b"] = (
            self._belief.last_repair_cost_max
        )
        self._pending_belief_steps["repair_cost_total_stage_b"] = (
            self._belief.last_repair_cost_total
        )
        self._pending_belief_steps["repair_moved_piece_count_max_stage_b"] = (
            self._belief.last_repair_moved_piece_count_max
        )
        self._pending_belief_steps["repair_max_piece_distance_stage_b"] = (
            self._belief.last_repair_max_piece_distance
        )
        self._pending_belief_steps["repair_long_move_count_stage_b"] = (
            self._belief.last_repair_long_move_count
        )
        self._pending_belief_steps["repair_teleport_like_count_stage_b"] = (
            self._belief.last_repair_teleport_like_count
        )
        self._pending_belief_steps["repair_forced_visible_square_count_stage_b"] = (
            self._belief.last_repair_forced_visible_square_count
        )
        self._pending_belief_steps["repair_unpaired_added_count_stage_b"] = (
            self._belief.last_repair_unpaired_added_count
        )
        self._pending_belief_steps["repair_unpaired_removed_count_stage_b"] = (
            self._belief.last_repair_unpaired_removed_count
        )
        self._pending_belief_steps["repair_worst_cost_stage_b"] = (
            self._belief.last_repair_worst_cost
        )
        self._pending_belief_steps["repair_worst_piece_stage_b"] = (
            self._belief.last_repair_worst_piece
        )
        self._pending_belief_steps["repair_worst_from_stage_b"] = (
            self._belief.last_repair_worst_from
        )
        self._pending_belief_steps["repair_worst_to_stage_b"] = (
            self._belief.last_repair_worst_to
        )
        self._pending_belief_steps["repair_worst_distance_stage_b"] = (
            self._belief.last_repair_worst_distance
        )
        self._pending_belief_steps["repair_worst_one_move_legal_stage_b"] = (
            self._belief.last_repair_worst_one_move_legal
        )
        self._pending_belief_steps["repair_strict_rejected_count_stage_b"] = (
            self._belief.last_repair_strict_rejected_count
        )
        self._pending_belief_steps["repair_strict_fallback_count_stage_b"] = (
            self._belief.last_repair_strict_fallback_count
        )
        self._pending_belief_steps["checkpoint_repair_stage_b"] = (
            self._belief.last_checkpoint_repair_fired
        )
        self._pending_belief_steps["checkpoint_repair_count_stage_b"] = (
            self._belief.last_checkpoint_repair_count
        )
        self._pending_belief_steps["checkpoint_repair_age_stage_b"] = (
            self._belief.last_checkpoint_repair_age
        )
        self._pending_belief_steps["checkpoint_repair_unique_stage_b"] = (
            self._belief.last_checkpoint_repair_unique
        )
        self._observed_ply += 1
        self._append_belief_snapshot(
            ply=self._observed_ply,
            snapshot_kind="after-opp-move",
            decision_path="after-opp-move",
        )

    def pick_move(self, view: PerspectiveView) -> chess.Move:
        assert self._belief is not None
        particle_count_pre = len(self._belief.particles)
        # Snapshot pre-move visibility for capture detection in observe_own_move.
        self._last_view_visible = dict(view.visible_piece_map)
        self._last_decision_view = view

        king_captures = [
            move
            for move in view.own_legal_moves
            if (piece := view.visible_piece_map.get(move.to_square)) is not None
            and piece.color != view.perspective
            and piece.piece_type == chess.KING
        ]
        if king_captures:
            chosen = self._rng.choice(_prefer_queen_promotion(king_captures))
            self._stage_pending_capture(chosen, view)
            self._emit_trace("king-capture", particle_count_pre, chosen)
            return chosen

        # MCTS path: when mcts_rollouts > 0, delegate to Monte Carlo Tree Search
        # after hard certainty vetoes (king-capture above, king-defender veto below
        # is applied inside mcts_pick_move via the evaluator's -∞ scoring).
        # All other short-circuits are replaced by the search tree.
        if self.mcts_rollouts > 0 and self._belief is not None and self._belief.particles:
            deadline_monotonic: float | None = None
            if view.clock_remaining_ms is not None:
                usable_ms = max(0, view.clock_remaining_ms - 300)
                budget_ms = _compute_per_move_budget_ms(usable_ms, view.increment_ms)
                budget_ms = min(budget_ms, usable_ms) if usable_ms > 0 else 100
                deadline_monotonic = time.monotonic() + budget_ms / 1000.0
            evaluator = self.evaluator_builder(view)
            q_out: list[float] = []
            visits_out: dict[str, int] = {}
            chosen = mcts_pick_move(
                self._belief,
                view,
                evaluator,
                self._rng,
                n_rollouts=self.mcts_rollouts,
                selection_depth=self.mcts_selection_depth,
                rollout_depth=self.mcts_rollout_depth,
                risk_lambda=self.mcts_risk_lambda,
                deadline_monotonic=deadline_monotonic,
                out_chosen_q=q_out,
                out_root_visits=visits_out,
            )
            # Distillation hooks. Set every MCTS call so callers can read them
            # before the next pick_move overwrites them. Used by corpus
            # generation to train value evals (q) and policy nets (visits)
            # without re-running self-play.
            self.last_mcts_root_q: float | None = q_out[0] if q_out else None
            self.last_mcts_root_visits: dict[str, int] = visits_out
            # Belief summary for FoW-native architectures that may want to
            # condition on uncertainty as an input feature. Shannon entropy
            # over normalized particle weights; n is particle count.
            _b = self._belief
            _n = len(_b.particles)
            _ent = 0.0
            if _n > 0 and _b.weights:
                _total = sum(_b.weights)
                if _total > 0:
                    import math as _math
                    for _w in _b.weights:
                        _p = _w / _total
                        if _p > 0:
                            _ent -= _p * _math.log(_p)
            self.last_belief_summary: dict | None = {"n_particles": _n, "entropy": _ent}
            self._stage_pending_capture(chosen, view)
            self._emit_trace("mcts", particle_count_pre, chosen)
            return chosen

        # King-defense short-circuit. Symmetric to king-capture: if our king is
        # on a square a visible enemy can capture next turn, restrict to moves
        # that resolve the attack (king flight, attacker capture, or block).
        # Stockfish-eval underweights "own king visibly attacked" because under
        # standard-chess rules it expects opp to be unable to capture the king;
        # in FOW opp will, so we have to bake the rule in here.
        # v0.6.1 Pattern A fix: rank king-defense as captures > blocks > flights.
        # Within attacker-captures, prefer max-material capture. Within each
        # tier, apply belief-grounded king-attack veto (drop candidates >50% of
        # particles agree leave the king attacked, which catches hidden
        # discovered checks). v0.6.0-mirror corpus showed two majors (g4 p17,
        # g12 p17) where the engine fled the king when a free pawn-takes-knight
        # was available.
        kd_captures, kd_blocks, kd_flights = _categorize_king_defense_moves(view)
        # Phantom-check guard: _categorize_king_defense_moves uses a visibility-
        # only board which treats every visible slider's ray as clear. If the
        # belief has majority weight on "the ray is blocked," the check is
        # phantom and king-defense should not fire (it would override better
        # moves like queen-save in a position where the king isn't actually in
        # danger). See engine_blunder_taxonomy memory for the ply-36 case.
        if (kd_captures or kd_blocks or kd_flights) and not self._real_king_threat_attackers(view):
            kd_captures, kd_blocks, kd_flights = [], [], []
        if kd_captures or kd_blocks or kd_flights:
            tier: list[chess.Move]
            tier_label: str
            non_king_captures = [
                move
                for move in kd_captures
                if (piece := view.visible_piece_map.get(move.from_square)) is not None
                and piece.piece_type != chess.KING
            ]
            king_captures = [
                move
                for move in kd_captures
                if (piece := view.visible_piece_map.get(move.from_square)) is not None
                and piece.piece_type == chess.KING
            ]
            high_value_king_captures = [
                move
                for move in king_captures
                if (
                    target := view.visible_piece_map.get(move.to_square)
                ) is not None
                and _MATERIAL_VALUE.get(target.piece_type, 0)
                >= _MATERIAL_VALUE[chess.ROOK]
            ]
            if high_value_king_captures:
                king_capture_tier = _prefer_higher_value_capture(
                    high_value_king_captures, view
                )
                belief_ok = [
                    m for m in king_capture_tier if self._belief_supports_move(m)
                ]
                safe_king_captures = self._belief_veto_king_attack(
                    belief_ok or king_capture_tier, view
                )
            else:
                safe_king_captures = []

            # Prefer piece captures over king captures. A non-king piece taking
            # the attacker keeps the king off exposed squares and avoids walking
            # into a hidden second attacker on the same file/rank/diagonal
            # (ply-90 class: Kxe8 into Re1xe8 because particles missed the
            # back-rank rook). King-takes-high-value is only chosen when no
            # piece can resolve the check.
            if non_king_captures:
                tier = _prefer_higher_value_capture(non_king_captures, view)
                tier_label = "king-defense-capture"
            elif safe_king_captures:
                tier = safe_king_captures
                tier_label = "king-defense-king-capture"
            elif kd_blocks:
                tier = kd_blocks
                tier_label = "king-defense-block"
            elif kd_flights:
                tier = kd_flights
                tier_label = "king-defense-flight"
            else:
                tier = king_captures
                tier_label = "king-defense-king-capture"
            # v0.6.2 fix: filter to belief-supported candidates first (avoid
            # picking a move that would wipe belief in Stage A). Then apply
            # belief-grounded king-attack veto. Either filter falling back to
            # the unfiltered tier preserves "make some defense" over none.
            belief_ok = [m for m in tier if self._belief_supports_move(m)]
            tier_filtered = belief_ok or tier
            survivors = self._belief_veto_king_attack(tier_filtered, view)
            candidates = survivors or self._belief_lowest_king_attack_risk(
                tier_filtered, view
            )
            chosen = self._rng.choice(candidates)
            self._stage_pending_capture(chosen, view)
            self._emit_trace(tier_label, particle_count_pre, chosen)
            return chosen

        # Queen-capture short-circuit. Same shape as king-capture: if any legal
        # move lands on a visible enemy queen, take it. Visibility-grounded so
        # it doesn't depend on belief aggregation (which the 2026-05-07 replay-
        # eval showed routinely loses these tactics to vote dilution).
        queen_captures = [
            move
            for move in view.own_legal_moves
            if (piece := view.visible_piece_map.get(move.to_square)) is not None
            and piece.color != view.perspective
            and piece.piece_type == chess.QUEEN
        ]
        if queen_captures:
            candidates = _prefer_lower_value_attacker(queen_captures, view)
            candidates = self._belief_veto_king_attack(candidates, view)
            if candidates:
                chosen = self._rng.choice(_prefer_queen_promotion(candidates))
                self._stage_pending_capture(chosen, view)
                self._emit_trace("queen-capture", particle_count_pre, chosen)
                return chosen

        # Queen-save short-circuit. If our queen is on a square a visible enemy
        # piece could capture next turn AND we have a queen move to a square
        # not attacked by any visible enemy piece, take one of those moves.
        # "Safe" is measured against a visibility-only synthesized board —
        # hidden attackers don't fire this.
        queen_save_tiers = _queen_save_tiers(view)
        for queen_save in queen_save_tiers:
            candidates = self._belief_veto_king_attack(queen_save, view)
            candidates = self._belief_veto_queen_fog_risk(candidates, view)
            if candidates:
                chosen = self._rng.choice(candidates)
                self._stage_pending_capture(chosen, view)
                self._emit_trace("queen-save", particle_count_pre, chosen)
                return chosen

        high_value_save = _high_value_piece_save_moves(view)
        if high_value_save:
            candidates = self._belief_veto_king_attack(high_value_save, view)
            if candidates:
                chosen = self._rng.choice(candidates)
                self._stage_pending_capture(chosen, view)
                self._emit_trace("high-value-save", particle_count_pre, chosen)
                return chosen

        piece_save = _visible_piece_save_moves(view, self._belief)
        if piece_save:
            candidates = self._belief_veto_king_attack(piece_save, view)
            # Also drop down-trade captures the belief says will be recaptured —
            # "saving" a bishop by capturing a defended pawn loses more than it
            # saves. Confirmed in vs-brian-game-3 ply-23 (Bxe4 into hidden knight).
            # If ALL candidates fail the bad-trade veto, fall through to main-eval
            # — don't force a bad save. The save tier is opt-in; main-eval can
            # pick a non-save move (retreat, develop) that's globally better.
            candidates = self._belief_veto_bad_capture_trade(candidates, view)
            if candidates:
                chosen = self._rng.choice(candidates)
                self._stage_pending_capture(chosen, view)
                self._emit_trace("visible-piece-save", particle_count_pre, chosen)
                return chosen

        belief_piece_save = (
            []
            if _castle_moves(view)
            else self._belief_piece_save_moves(view.own_legal_moves, view)
        )
        if belief_piece_save:
            candidates = self._belief_veto_king_attack(belief_piece_save, view)
            if candidates:
                chosen = self._rng.choice(candidates)
                self._stage_pending_capture(chosen, view)
                self._emit_trace("belief-piece-save", particle_count_pre, chosen)
                return chosen

        deep_minor_retreat = _advanced_minor_retreat_moves(view, min_from_depth=2)
        if deep_minor_retreat:
            belief_ok = [m for m in deep_minor_retreat if self._belief_supports_move(m)]
            candidates = belief_ok or deep_minor_retreat
            candidates = self._belief_veto_king_attack(candidates, view)
            if candidates:
                chosen = self._rng.choice(candidates)
                self._stage_pending_capture(chosen, view)
                self._emit_trace("advanced-minor-retreat", particle_count_pre, chosen)
                return chosen

        # v0.6.1 Pattern B: a visible bishop/knight/rook on a square not
        # attacked by any visible enemy is a free piece. Capture it before
        # main-eval gets a chance to dilute the material delta.
        safe_minor_rook = _safe_visible_minor_or_rook_captures(view)
        if safe_minor_rook:
            # v0.6.2: filter to belief-supported candidates so we don't wipe
            # belief by picking a move particles can't accommodate. v0.7.3:
            # also drop captures that belief says leave our king capturable;
            # material shortcuts are not allowed to override terminal FOW risk.
            belief_ok = [m for m in safe_minor_rook if self._belief_supports_move(m)]
            candidates = belief_ok or safe_minor_rook
            candidates = self._belief_veto_king_attack(candidates, view)
            candidates = self._belief_veto_bad_capture_trade(candidates, view)
            candidates = self._belief_veto_queen_fog_risk(candidates, view)
            if candidates:
                chosen = self._rng.choice(_prefer_queen_promotion(candidates))
                self._stage_pending_capture(chosen, view)
                self._emit_trace("visible-minor-rook-capture", particle_count_pre, chosen)
                return chosen

        castle = _castle_moves(view)
        if castle:
            belief_ok = [m for m in castle if self._belief_supports_move(m)]
            candidates = belief_ok or castle
            candidates = self._belief_veto_king_attack(candidates, view)
            if candidates:
                chosen = self._rng.choice(candidates)
                self._stage_pending_capture(chosen, view)
                self._emit_trace("castle", particle_count_pre, chosen)
                return chosen

        king_shelter = _king_shelter_moves(view)
        if king_shelter:
            belief_ok = [m for m in king_shelter if self._belief_supports_move(m)]
            candidates = belief_ok or king_shelter
            candidates = self._belief_veto_king_attack(candidates, view)
            if candidates:
                chosen = self._rng.choice(candidates)
                self._stage_pending_capture(chosen, view)
                self._emit_trace("king-shelter", particle_count_pre, chosen)
                return chosen

        visible_capture_available = any(
            (target := view.visible_piece_map.get(move.to_square)) is not None
            and target.color != view.perspective
            and target.piece_type != chess.KING
            for move in view.own_legal_moves
        )
        latent_block = (
            []
            if visible_capture_available
            else _latent_king_slider_block_moves(view, self._belief)
        )
        if latent_block:
            belief_ok = [m for m in latent_block if self._belief_supports_move(m)]
            candidates = belief_ok or latent_block
            candidates = self._belief_veto_king_attack(candidates, view)
            if candidates:
                chosen = self._rng.choice(candidates)
                self._stage_pending_capture(chosen, view)
                self._emit_trace("latent-king-slider-block", particle_count_pre, chosen)
                return chosen

        if self._observed_ply + 1 <= 12:
            minor_retreat = _advanced_minor_retreat_moves(view)
            if minor_retreat:
                belief_ok = [m for m in minor_retreat if self._belief_supports_move(m)]
                candidates = belief_ok or minor_retreat
                candidates = self._belief_veto_king_attack(candidates, view)
                if candidates:
                    chosen = self._rng.choice(candidates)
                    self._stage_pending_capture(chosen, view)
                    self._emit_trace(
                        "advanced-minor-retreat", particle_count_pre, chosen
                    )
                    return chosen

            development = _early_development_moves(view)
            if development:
                belief_ok = [m for m in development if self._belief_supports_move(m)]
                candidates = belief_ok or development
                candidates = self._belief_veto_king_attack(candidates, view)
                if candidates:
                    chosen = self._rng.choice(candidates)
                    self._stage_pending_capture(chosen, view)
                    self._emit_trace("early-development", particle_count_pre, chosen)
                    return chosen

        if not self._belief.particles:
            # Belief filter collapsed (no particles consistent with observation).
            # Without this fallback, best_action returns own_legal_moves[0] —
            # deterministic first-alphabetical, worse than random. Score moves
            # on a visibility-only synthesized board instead so visible captures
            # and threats still drive selection.
            chosen = self._fallback_pick_move(view)
            self._stage_pending_capture(chosen, view)
            self._emit_trace("fallback", particle_count_pre, chosen)
            return chosen

        # Compute deadline from clock if we're in a timed regime. Leave a
        # 200ms buffer so we don't accidentally spend the entire clock on
        # this move (the harness debits actual elapsed wall, including
        # python overhead between deadline check and return).
        deadline_monotonic: float | None = None
        if view.clock_remaining_ms is not None:
            usable_clock_ms = max(0, view.clock_remaining_ms - 200)
            budget_ms = _compute_per_move_budget_ms(
                usable_clock_ms, view.increment_ms
            )
            budget_ms = min(budget_ms, usable_clock_ms) if usable_clock_ms > 0 else 50
            deadline_monotonic = time.monotonic() + budget_ms / 1000.0

        evaluator = self.evaluator_builder(view)
        scored: list[tuple[chess.Move, float, float]] = []
        safe_legal_moves = self._belief_veto_king_attack(view.own_legal_moves, view)
        legal_moves = safe_legal_moves or self._belief_lowest_king_attack_risk(
            view.own_legal_moves, view
        )
        non_king_capture_moves = [
            move
            for move in legal_moves
            if not _nonterminal_king_material_capture(move, view)
        ]
        legal_moves = non_king_capture_moves or legal_moves
        trade_safe_moves = self._belief_veto_bad_capture_trade(legal_moves, view)
        legal_moves = trade_safe_moves or legal_moves
        king_defended_safe = self._veto_captures_defended_by_visible_king(legal_moves, view)
        legal_moves = king_defended_safe or legal_moves
        queen_fog_safe_moves = self._belief_veto_queen_fog_risk(
            legal_moves, view, threshold=self.queen_fog_risk_threshold
        )
        legal_moves = queen_fog_safe_moves or legal_moves
        piece_fog_safe_moves = self._belief_veto_piece_fog_risk(
            legal_moves, view, max_capture_risk=self.piece_fog_risk_threshold
        )
        legal_moves = piece_fog_safe_moves or legal_moves
        belief_piece_save = self._belief_piece_save_moves(legal_moves, view)
        if belief_piece_save:
            chosen = self._rng.choice(belief_piece_save)
            self._stage_pending_capture(chosen, view)
            self._emit_trace("belief-piece-save", particle_count_pre, chosen)
            return chosen
        high_value_belief_save = self._belief_high_value_piece_save_moves(
            legal_moves, view
        )
        if high_value_belief_save:
            chosen = self._rng.choice(high_value_belief_save)
            self._stage_pending_capture(chosen, view)
            self._emit_trace("belief-high-value-save", particle_count_pre, chosen)
            return chosen
        queen_pressure = self._belief_queen_king_pressure_moves(legal_moves, view)
        if queen_pressure:
            chosen = self._rng.choice(queen_pressure)
            self._stage_pending_capture(chosen, view)
            self._emit_trace("queen-king-pressure", particle_count_pre, chosen)
            return chosen
        if self.pimc_search_depth > 0:
            from .engine import pimc_best_action
            chosen = pimc_best_action(
                self._belief,
                evaluator,
                legal_moves,
                max_particles=self.max_eval_particles,
                search_depth=self.pimc_search_depth,
                rng=self._rng,
                deadline_monotonic=deadline_monotonic,
                out_scored_moves=scored,
            )
        else:
            chosen = best_action(
                self._belief,
                evaluator,
                legal_moves,
                max_particles=self.max_eval_particles,
                risk_aversion=self.risk_aversion,
                rng=self._rng,
                deadline_monotonic=deadline_monotonic,
                out_scored_moves=scored,
            )
        decision_weight_modes = self._decision_weight_mode_scores(
            evaluator,
            legal_moves,
            max_clusters=max(16, self.max_eval_particles),
        )
        # v0.9.4-pre1: soft post-score adjustments that the binary fog-risk
        # vetoes miss (the vetoes only fire above their threshold; below they
        # leave the move scored as if it were safe). Two adjustments:
        #   - capture_risk: belief-weighted P(landed piece is recapturable) ×
        #                   piece_value × coef. So even sub-threshold risk
        #                   shows up as a proportional score penalty.
        #   - anti_shuffle: flat penalty if (from, to) was played in the
        #                   last N own plies. Forces conversion progress.
        if scored and (
            self.capture_risk_penalty_coef > 0
            or self.anti_shuffle_penalty > 0
            or self.push_when_ahead_bonus > 0
            or self.info_reveal_bonus_coef > 0
        ):
            risk_map = self._compute_capture_risk_map(
                [m for m, _, _ in scored], view
            )
            recent = set(self._recent_own_moves)
            # v0.9.5: material edge from belief gates push-when-ahead. Use the
            # visible piece map (truth from our POV) for own material, and the
            # belief's posterior expected count for opp.
            edge = self._material_edge_estimate(view)
            ahead = edge > self.push_when_ahead_min_edge
            shuffle_pen = (
                self.anti_shuffle_penalty_strong if ahead else self.anti_shuffle_penalty
            )
            # Precompute pre-move visibility for info-reveal calculations.
            pre_vis = self._cached_visibility_count(view)
            adjusted = []
            for move, score, support in scored:
                mover = view.visible_piece_map.get(move.from_square)
                piece_val = _MATERIAL_VALUE.get(
                    mover.piece_type, 0
                ) if mover and mover.piece_type not in (chess.PAWN, chess.KING) else 0
                risk = risk_map.get(move, 0.0)
                risk_penalty = risk * piece_val * self.capture_risk_penalty_coef
                shuffle_penalty = (
                    shuffle_pen
                    if (move.from_square, move.to_square) in recent
                    else 0.0
                )
                # Push-when-ahead bonus: any forward-progress move (rank
                # increases toward opp side) when materially up. Pawns and
                # sliding pieces contribute; piece values are not scaled (the
                # bonus is the same regardless of mover) — the idea is to
                # break ties toward forward motion, not to overrule major
                # tactical considerations.
                push_bonus = 0.0
                if ahead and mover is not None:
                    if self._is_forward_progress(move, view.perspective):
                        push_bonus = self.push_when_ahead_bonus
                # Info-reveal bonus: count newly visible squares after this
                # move (sliding pieces into fog reveal more). Linear cp per
                # square; capped implicitly by the small per-square coef.
                reveal_bonus = 0.0
                if self.info_reveal_bonus_coef > 0:
                    delta = self._move_visibility_delta(move, view, pre_vis)
                    reveal_bonus = delta * self.info_reveal_bonus_coef
                final = (
                    score
                    - risk_penalty
                    - shuffle_penalty
                    + push_bonus
                    + reveal_bonus
                )
                adjusted.append((move, final, support))
            adjusted.sort(key=lambda r: -r[1])
            new_chosen = adjusted[0][0]
            if new_chosen != chosen:
                chosen = new_chosen
            scored = adjusted
        # Top 5 moves by aggregated score; only surface what the trace actually needs.
        scored.sort(key=lambda r: -r[1])
        top_k = [(m.uci(), s, support) for m, s, support in scored[:5]]
        lva_chosen = _prefer_lower_value_same_target_capture(chosen, legal_moves, view)
        decision_path = "main-eval"
        if lva_chosen != chosen and self._belief_supports_move(lva_chosen):
            chosen = lva_chosen
            decision_path = "main-eval-lva-capture"
        self._stage_pending_capture(chosen, view)
        self._emit_trace(
            decision_path,
            particle_count_pre,
            chosen,
            top_k_scores=top_k,
            decision_weight_modes=decision_weight_modes,
        )
        return chosen

    def _fallback_pick_move(self, view: PerspectiveView) -> chess.Move:
        """Score moves on a visibility-only board. Used when belief has collapsed."""
        synthesized = chess.Board.empty()
        for sq, piece in view.visible_piece_map.items():
            synthesized.set_piece_at(sq, piece)
        synthesized.turn = view.perspective

        evaluator = self.evaluator_builder(view)
        scored: list[tuple[chess.Move, float]] = []
        best_score = float("-inf")
        for move in view.own_legal_moves:
            try:
                score = evaluator(synthesized, move, view.perspective)
            except (ValueError, AssertionError):
                continue
            scored.append((move, score))
            if score > best_score:
                best_score = score
        if not scored:
            return view.own_legal_moves[0]
        eps = 1e-6
        top = [m for m, s in scored if s >= best_score - eps]
        return self._rng.choice(top)
