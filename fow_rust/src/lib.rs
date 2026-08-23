//! Native Rust hot-path for Fog of War chess belief-state enumeration.
//!
//! Scope (Phase A perf): `visible_squares` + `consistent_with` +
//! the `update_opp_move` inner loop. Everything else stays in Python.
//!
//! Built on shakmaty for FoW-tolerant FEN parsing (kings can walk into
//! attack in FoW; other chess libraries reject those positions) and its
//! attack-bitboard primitives.

use core::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};
use dashmap::DashSet;

use pyo3::prelude::*;
use rayon::prelude::*;

// EXPERIMENT (2026-05-29): faster global allocator. The eq pass allocates a
// per-node Vec (action_values + eq_current_strategy) per traverse per iter —
// O(100M) small allocs/move at i=200. mimalloc is a measurement (is eq_pass
// allocator-bound?) AND a potential free byte-identical win. Revert if no gain.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use shakmaty::{
    attacks, fen::Fen, Bitboard, Board as ShakBoard, Color, File, Piece, Rank, Role, Setup, Square,
};

/// Hello-world ping; verifies the wheel is loaded.
#[pyfunction]
fn ping() -> &'static str {
    "fow_rust 0.1.0 alive"
}

/// FEN parse-then-serialize. Used as a parity check against python-chess
/// to validate that shakmaty's `Fen` serializer produces byte-identical
/// output to `chess.Board.fen()`. Required for set-based dedup to work
/// across the Python/Rust boundary.
#[pyfunction]
fn fen_roundtrip(fen: &str) -> PyResult<String> {
    let parsed = Fen::from_ascii(fen.as_bytes()).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("bad FEN: {e}"))
    })?;
    let setup = parsed.into_setup();
    Ok(Fen(setup).to_string())
}

/// Bitmask of squares visible to `color_bool` (true=white, false=black)
/// under FoW. Returns u64. Matches Python `fow_chess.visibility.visible_squares`.
///
/// FEN-input variant — convenient for tests, slower for hot path because
/// Python has to serialize board → FEN. Use `visible_squares_bb` for the
/// hot path.
#[pyfunction]
fn visible_squares(fen: &str, color_bool: bool) -> PyResult<u64> {
    let setup = parse_fen_lenient(fen)?;
    let color = if color_bool { Color::White } else { Color::Black };
    Ok(visible_squares_from_setup(&setup, color))
}

/// Bitboard-input variant for hot-path callers. Skips FEN serialization.
///
/// Inputs are exactly the bitboards python-chess exposes on `chess.Board`:
///   - pawns, knights, bishops, rooks, queens, kings — bitboard per piece type
///   - occupied_white, occupied_black — bitboard per color
///   - castling_rights — bitboard of rook origins with castling rights
///   - ep_square_idx — 0..63 if ep set, or 64 if no ep (avoids Option in PyO3)
///   - color_bool — true = white, false = black
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn visible_squares_bb(
    pawns: u64,
    knights: u64,
    bishops: u64,
    rooks: u64,
    queens: u64,
    kings: u64,
    occupied_white: u64,
    occupied_black: u64,
    castling_rights: u64,
    ep_square_idx: u32,
    color_bool: bool,
) -> u64 {
    let setup = setup_from_bb(
        pawns, knights, bishops, rooks, queens, kings,
        occupied_white, occupied_black, castling_rights, ep_square_idx,
    );
    let color = if color_bool { Color::White } else { Color::Black };
    visible_squares_from_setup(&setup, color)
}

/// Apply a pseudo-legal move and return the resulting FEN.
///
/// Mirror semantics of `chess.Board.push(move)` for FoW-tolerant positions
/// (kings may be captured, may walk into check). The caller passes the
/// move as `(from_sq, to_sq, promotion)` where `promotion` is 0 for none or
/// shakmaty's Role number (2=Knight, 3=Bishop, 4=Rook, 5=Queen).
///
/// Special-move detection (castling vs normal king move, en passant vs
/// pawn capture) is recovered from board context, mirroring python-chess.
#[pyfunction]
fn apply_move(fen: &str, from_idx: u8, to_idx: u8, promo: u8) -> PyResult<String> {
    let setup = parse_fen_lenient(fen)?;
    let from = unsafe { Square::new_unchecked(from_idx as u32) };
    let to = unsafe { Square::new_unchecked(to_idx as u32) };
    let promo_role = role_from_int(promo);
    let new_setup = apply_move_to_setup(&setup, from, to, promo_role);
    Ok(Fen(new_setup).to_string())
}

#[inline]
fn role_from_int(n: u8) -> Option<Role> {
    match n {
        0 => None,
        2 => Some(Role::Knight),
        3 => Some(Role::Bishop),
        4 => Some(Role::Rook),
        5 => Some(Role::Queen),
        _ => None, // invalid promotion ints are treated as no-promo for defensiveness
    }
}

#[inline]
fn role_code(r: Role) -> u8 {
    match r {
        Role::Pawn => 1, Role::Knight => 2, Role::Bishop => 3,
        Role::Rook => 4, Role::Queen => 5, Role::King => 6,
    }
}

#[inline]
fn role_from_code(c: u8) -> Role {
    match c {
        1 => Role::Pawn, 2 => Role::Knight, 3 => Role::Bishop,
        4 => Role::Rook, 5 => Role::Queen, _ => Role::King,
    }
}

/// Fixed-size, lossless packed position key. Replaces the FEN String as the
/// belief-set dedup key + resident storage. Hashing a 56-byte POD struct via
/// FxHash is far cheaper than SipHash over a 60-byte heap String, and it
/// allocates nothing — the dominant cost on explosion plies was hashing +
/// allocating millions of FEN strings. Bijective with the standard-chess FEN
/// (`Fen(unpack(pack(s))) == Fen(s)`), so dedup membership — and thus PHASH
/// parity + strength — is identical to the old `HashSet<String>`.
///
/// Layout: 64 squares × 4-bit piece code (0=empty, 1-6 white P..K, 7-12 black)
/// in `board`; raw castling-rights rook-square bitboard; halfmove/fullmove
/// clocks; ep square (0=none else sq+1); side to move.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PackedPos {
    board: [u64; 4],
    castling: u64,
    halfmoves: u32,
    fullmoves: u32,
    ep: u8,
    turn_white: bool,
}

/// Stable 64-bit hash of a packed world. Drives the bottom-K (KMV) belief
/// bound: keeping the K worlds with the smallest hash is an exactly-uniform
/// K-subset of the distinct worlds (the hash induces a uniform random
/// permutation of distinct keys). FxHash is fast and fine here — collisions at
/// M~10^8 over 64 bits are negligible, and ties at the threshold tie-break on
/// the full PackedPos key (Ord). NOT used for dedup membership (DashSet handles
/// that on the full key); only for the order statistic.
#[inline]
fn hash_packed(p: &PackedPos) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = FxHasher::default();
    p.hash(&mut h);
    h.finish()
}

/// How many inserts a worker does between bottom-K compaction checks. Avoids
/// calling DashSet::len() (a sum over shards) on every insert. Per-thread.
const BOTTOMK_CHECK_INTERVAL: usize = 8192;

/// Compact a bottom-K DashSet down to its K smallest-hash worlds.
///
/// Periodic (final_pass=false): only fires when |kept| > 2K, the memory
/// high-watermark; one worker claims it via `compacting`, the rest skip and
/// keep inserting against the (lowered) threshold. Final (final_pass=true):
/// post-loop, single-threaded, fires when |kept| > K so the result lands at
/// EXACTLY the bottom-K. `threshold` only ever decreases (fetch_min), so any
/// world dropped early — pre-insert (hash > threshold) or by a prior retain —
/// has hash ≥ the final threshold and could not be in the final bottom-K
/// (uniformity preserved). Sets `downsampled` whenever it removes worlds.
fn compact_bottom_k(
    kept: &DashSet<PackedPos, FxBuildHasher>,
    k: usize,
    threshold: &AtomicU64,
    compacting: &AtomicBool,
    downsampled: &AtomicBool,
    final_pass: bool,
) {
    let trigger = if final_pass { k } else { 2 * k };
    if kept.len() <= trigger {
        return;
    }
    // Periodic compactions are mutually exclusive (memory mgmt under contention);
    // the final pass is post-loop so it always wins the guard uncontended.
    if compacting
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    if kept.len() > k {
        let mut hashes: Vec<u64> = kept.iter().map(|r| hash_packed(r.key())).collect();
        if hashes.len() > k {
            // saturating_sub: k=0 (a degenerate cap) keeps 1 world rather than
            // underflowing to usize::MAX and panicking in select_nth_unstable.
            // For k>=1 this is the k-th smallest (0-indexed k-1) as intended.
            let idx = k.saturating_sub(1);
            let (_, kth, _) = hashes.select_nth_unstable(idx);
            let tau = *kth;
            kept.retain(|w| hash_packed(w) <= tau);
            threshold.fetch_min(tau, Ordering::Relaxed);
            downsampled.store(true, Ordering::Relaxed);
        }
    }
    compacting.store(false, Ordering::Release);
}

#[inline]
fn pack(setup: &Setup) -> PackedPos {
    let mut board = [0u64; 4];
    let b = &setup.board;
    for sq in b.occupied() {
        let piece = b.piece_at(sq).expect("occupied square has a piece");
        let code = role_code(piece.role)
            + if piece.color == Color::White { 0 } else { 6 };
        let idx = u8::from(sq) as usize;
        board[idx >> 4] |= (code as u64) << ((idx & 15) * 4);
    }
    PackedPos {
        board,
        castling: setup.castling_rights.0,
        halfmoves: setup.halfmoves,
        fullmoves: setup.fullmoves.get(),
        ep: setup.ep_square.map(|s| u8::from(s) + 1).unwrap_or(0),
        turn_white: setup.turn == Color::White,
    }
}

#[inline]
fn unpack(p: &PackedPos) -> Setup {
    let mut setup = Setup::empty();
    for idx in 0..64usize {
        let code = ((p.board[idx >> 4] >> ((idx & 15) * 4)) & 0xF) as u8;
        if code == 0 {
            continue;
        }
        let (color, rc) = if code <= 6 {
            (Color::White, code)
        } else {
            (Color::Black, code - 6)
        };
        let sq = unsafe { Square::new_unchecked(idx as u32) };
        setup.board.set_piece_at(sq, Piece { color, role: role_from_code(rc) });
    }
    setup.turn = if p.turn_white { Color::White } else { Color::Black };
    setup.castling_rights = Bitboard(p.castling);
    setup.ep_square = if p.ep == 0 {
        None
    } else {
        Some(unsafe { Square::new_unchecked((p.ep - 1) as u32) })
    };
    setup.halfmoves = p.halfmoves;
    setup.fullmoves = NonZeroU32::new(p.fullmoves).unwrap_or(NonZeroU32::MIN);
    setup
}

fn apply_move_to_setup(setup: &Setup, from: Square, to: Square, promo: Option<Role>) -> Setup {
    let stm = setup.turn;
    let board = &setup.board;
    let moving_role = board
        .role_at(from)
        .expect("apply_move called with empty from-square");
    let captured_role = board.role_at(to);

    let is_castling = moving_role == Role::King
        && (from.file() as i32 - to.file() as i32).abs() == 2;
    let is_ep = moving_role == Role::Pawn
        && captured_role.is_none()
        && setup.ep_square == Some(to);

    let mut next = setup.clone();
    let nb = &mut next.board;

    if is_castling {
        let kingside = (to.file() as u8) > (from.file() as u8);
        // Find the matching rook with castling rights on this side
        let rook_from = setup
            .castling_rights
            .into_iter()
            .find(|r| {
                board.color_at(*r) == Some(stm)
                    && r.rank() == from.rank()
                    && ((r.file() as u8) > (from.file() as u8)) == kingside
            })
            .expect("castling without matching rook in castling_rights");
        let rook_to_file = if kingside { File::F } else { File::D };
        let rook_to = Square::from_coords(rook_to_file, from.rank());

        nb.discard_piece_at(from);
        nb.discard_piece_at(rook_from);
        nb.set_piece_at(to, Piece { color: stm, role: Role::King });
        nb.set_piece_at(rook_to, Piece { color: stm, role: Role::Rook });
    } else if is_ep {
        let captured_sq = Square::from_coords(to.file(), from.rank());
        nb.discard_piece_at(from);
        nb.discard_piece_at(captured_sq);
        nb.set_piece_at(to, Piece { color: stm, role: Role::Pawn });
    } else {
        nb.discard_piece_at(from);
        // set_piece_at internally discards target square, so no explicit capture-clear needed
        let placed_role = promo.unwrap_or(moving_role);
        nb.set_piece_at(to, Piece { color: stm, role: placed_role });
    }

    // En passant square: emit only when at least one ep capture is LEGAL
    // (capturer's king not left in check). Mirrors python-chess's default
    // `EnPassantMode.LEGAL` behavior in `fen()` — required for FEN-equality
    // dedup of next-positions.
    next.ep_square = if moving_role == Role::Pawn
        && (from.rank() as i32 - to.rank() as i32).abs() == 2
    {
        let mid_rank = Rank::new(((from.rank() as u32) + (to.rank() as u32)) / 2);
        let ep_cand = Square::from_coords(from.file(), mid_rank);
        let enemy_color = stm.other();
        // Use the post-push board (`nb`) so the just-pushed pawn at `to` is
        // present (and gets removed in the ep simulation).
        if has_legal_ep_capture(nb, enemy_color, ep_cand, to) {
            Some(ep_cand)
        } else {
            None
        }
    } else {
        None
    };

    // Halfmove clock: reset on pawn move or capture; else +1
    let was_capture = captured_role.is_some() || is_ep;
    next.halfmoves = if moving_role == Role::Pawn || was_capture {
        0
    } else {
        setup.halfmoves + 1
    };

    // Castling rights update
    next.castling_rights = update_castling_rights(setup.castling_rights, from, to, moving_role, stm);

    // FoW: when a king is captured, the captured color loses ALL castling
    // rights (they can't castle without a king). Mirrors python-chess.
    if captured_role == Some(Role::King) {
        let captured_color = stm.other();
        let backrank = match captured_color {
            Color::White => Rank::First,
            Color::Black => Rank::Eighth,
        };
        next.castling_rights = next.castling_rights & !Bitboard::from(backrank);
    }

    // Turn toggle + fullmove increment after black moves
    next.turn = stm.other();
    if stm == Color::Black {
        next.fullmoves = NonZeroU32::new(setup.fullmoves.get() + 1)
            .expect("fullmoves+1 is never zero");
    }

    next
}

/// Does at least one en-passant capture exist for `capturer` that doesn't
/// leave the capturer's king in check? Used to mirror python-chess's
/// FEN `EnPassantMode.LEGAL` filter — emit ep_square only if a legal ep
/// capture exists from this position.
fn has_legal_ep_capture(
    board: &ShakBoard,
    capturer: Color,
    ep_target: Square,
    captured_pawn_sq: Square,
) -> bool {
    let pawn_rank = captured_pawn_sq.rank();
    let capturer_pawns = board.by_piece(capturer.pawn());
    for adj in adjacent_files(ep_target.file()) {
        let pawn_sq = Square::from_coords(adj, pawn_rank);
        if !capturer_pawns.contains(pawn_sq) {
            continue;
        }
        // Simulate ep: capturer pawn moves pawn_sq → ep_target, captured pawn removed.
        // python-chess returns False if capturer has no king (can't be in check).
        let king_bb = board.by_piece(capturer.king());
        if king_bb.count() != 1 {
            continue;
        }
        let king_sq = king_bb.first().unwrap();

        // Build the post-ep occupancy from the input board (post-push state):
        // remove capturer pawn (pawn_sq) AND captured pawn (captured_pawn_sq);
        // add the capturer pawn at ep_target.
        let removed = Bitboard::from_square(pawn_sq) | Bitboard::from_square(captured_pawn_sq);
        let added = Bitboard::from_square(ep_target);
        let post_occ = (board.occupied() & !removed) | added;

        // For the attacker scan we also need to exclude the captured enemy
        // pawn from the enemy-pawn bitboard (it can no longer attack), and
        // exclude the moved capturer pawn from the friendly-pawn bitboard
        // (irrelevant for enemy attacks on our king, but mirrored for safety).
        let enemy = capturer.other();
        if !square_attacked_with_occ(board, king_sq, enemy, post_occ, captured_pawn_sq) {
            return true;
        }
    }
    false
}

/// Like `is_square_attacked`, but uses an explicit occupancy bitmask and
/// also excludes the source-square pawn from being an attacker (since it
/// just moved away). Pawn attacks are computed from board-recorded pawn
/// positions minus the moved pawn.
fn square_attacked_with_occ(
    board: &ShakBoard,
    sq: Square,
    by_color: Color,
    occupied: Bitboard,
    moved_pawn_sq: Square,
) -> bool {
    let pawn_attackers = attacks::pawn_attacks(by_color.other(), sq)
        & (board.by_piece(by_color.pawn()) & !Bitboard::from_square(moved_pawn_sq));
    if pawn_attackers.any() {
        return true;
    }
    if (attacks::knight_attacks(sq) & board.by_piece(by_color.knight())).any() {
        return true;
    }
    if (attacks::king_attacks(sq) & board.by_piece(by_color.king())).any() {
        return true;
    }
    let bq = board.by_piece(by_color.bishop()) | board.by_piece(by_color.queen());
    if (attacks::bishop_attacks(sq, occupied) & bq).any() {
        return true;
    }
    let rq = board.by_piece(by_color.rook()) | board.by_piece(by_color.queen());
    if (attacks::rook_attacks(sq, occupied) & rq).any() {
        return true;
    }
    false
}

fn update_castling_rights(
    rights: Bitboard,
    from: Square,
    to: Square,
    role: Role,
    stm: Color,
) -> Bitboard {
    let mut new = rights;
    // King moved: lose all rights for this color
    if role == Role::King {
        let backrank = match stm {
            Color::White => Rank::First,
            Color::Black => Rank::Eighth,
        };
        new = new & !Bitboard::from(backrank);
    }
    // Rook (or anything) moved from a castling-rights square
    if new.contains(from) {
        new = new & !Bitboard::from_square(from);
    }
    // Castling-rights square captured (a rook there is lost)
    if new.contains(to) {
        new = new & !Bitboard::from_square(to);
    }
    new
}

/// Pseudo-legal moves matching python-chess `Board.pseudo_legal_moves`.
///
/// Returns `Vec<(from_sq, to_sq, promotion)>` where `promotion` is 0 for
/// non-promotion moves or shakmaty's Role number (knight=2..queen=5) for
/// promotions. Special cases:
///   - Castling: encoded as king's from→to (e.g., e1→g1 for white kingside)
///   - En passant: encoded as pawn's from→ep_target
///
/// FoW-tolerant: includes king moves into attacked squares, captures of
/// opp king, and moves that leave own king in check. Castling moves are
/// included ONLY when fully legal (rights present, path clear, transit
/// squares safe, king not in check) — same as python-chess.
#[pyfunction]
fn pseudo_legal_moves(fen: &str) -> PyResult<Vec<(u8, u8, u8)>> {
    let setup = parse_fen_lenient(fen)?;
    Ok(gen_pseudo_legal_moves(&setup, setup.turn))
}

fn gen_pseudo_legal_moves(setup: &Setup, color: Color) -> Vec<(u8, u8, u8)> {
    let board = &setup.board;
    let own = board.by_color(color);
    let opp = board.by_color(color.other());
    let all = own | opp;
    let mut moves: Vec<(u8, u8, u8)> = Vec::with_capacity(64);

    // Pawns: pushes + double pushes + diagonal captures + promotions
    for from in board.by_piece(color.pawn()) {
        // Single push
        if let Some(to) = pawn_push(from, color) {
            if !all.contains(to) {
                if is_promotion_rank(to, color) {
                    push_promotions(&mut moves, from, to);
                } else {
                    moves.push((from as u8, to as u8, 0));
                    // Double push from starting rank, both squares empty
                    if is_pawn_start_rank(from, color) {
                        if let Some(to2) = pawn_push(to, color) {
                            if !all.contains(to2) {
                                moves.push((from as u8, to2 as u8, 0));
                            }
                        }
                    }
                }
            }
        }
        // Diagonal captures
        for to in attacks::pawn_attacks(color, from) & opp {
            if is_promotion_rank(to, color) {
                push_promotions(&mut moves, from, to);
            } else {
                moves.push((from as u8, to as u8, 0));
            }
        }
    }

    // En passant — only when adjacent pawn exists AND color matches ep rank direction
    if let Some(ep_target) = setup.ep_square {
        let ep_rank_idx = ep_target.rank() as u8;
        let valid_for_color = match color {
            Color::White => ep_rank_idx == 5,
            Color::Black => ep_rank_idx == 2,
        };
        if valid_for_color {
            let pawn_rank_idx = match color {
                Color::White => ep_rank_idx - 1,
                Color::Black => ep_rank_idx + 1,
            };
            let pawn_rank = Rank::new(pawn_rank_idx as u32);
            for adj_file in adjacent_files(ep_target.file()) {
                let pawn_sq = Square::from_coords(adj_file, pawn_rank);
                if board.by_piece(color.pawn()).contains(pawn_sq) {
                    moves.push((pawn_sq as u8, ep_target as u8, 0));
                }
            }
        }
    }

    // Knights
    for from in board.by_piece(color.knight()) {
        for to in attacks::knight_attacks(from) & !own {
            moves.push((from as u8, to as u8, 0));
        }
    }
    // Bishops
    for from in board.by_piece(color.bishop()) {
        for to in attacks::bishop_attacks(from, all) & !own {
            moves.push((from as u8, to as u8, 0));
        }
    }
    // Rooks
    for from in board.by_piece(color.rook()) {
        for to in attacks::rook_attacks(from, all) & !own {
            moves.push((from as u8, to as u8, 0));
        }
    }
    // Queens
    for from in board.by_piece(color.queen()) {
        for to in (attacks::bishop_attacks(from, all) | attacks::rook_attacks(from, all)) & !own {
            moves.push((from as u8, to as u8, 0));
        }
    }
    // King (regular moves only; castling emitted separately below)
    for from in board.by_piece(color.king()) {
        for to in attacks::king_attacks(from) & !own {
            moves.push((from as u8, to as u8, 0));
        }
    }

    // Castling — only when fully legal (matches python-chess pseudo_legal_moves)
    push_castling_moves(setup, color, &mut moves);

    moves
}

#[inline]
fn is_promotion_rank(sq: Square, color: Color) -> bool {
    match color {
        Color::White => sq.rank() == Rank::Eighth,
        Color::Black => sq.rank() == Rank::First,
    }
}

#[inline]
fn is_pawn_start_rank(sq: Square, color: Color) -> bool {
    match color {
        Color::White => sq.rank() == Rank::Second,
        Color::Black => sq.rank() == Rank::Seventh,
    }
}

#[inline]
fn push_promotions(moves: &mut Vec<(u8, u8, u8)>, from: Square, to: Square) {
    // python-chess emits in order: knight, bishop, rook, queen (low → high promotion role).
    // Setting up to match its iteration order so a sorted-by-tuple comparison can
    // optionally tighten beyond set equality.
    for promo in [Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
        moves.push((from as u8, to as u8, promo as u8));
    }
}

fn push_castling_moves(setup: &Setup, color: Color, moves: &mut Vec<(u8, u8, u8)>) {
    push_castling_moves_impl(setup, color, moves, /* fow_rules */ false);
}

/// FoW-rules variant: castling is legal even when king is in check, transits
/// attacked squares, or moves into check (the king doesn't "see" the
/// attacker). Used by PEnumerator's own_move_core / opp_move_core for belief
/// admission so the truth — which CAN legally castle in FoW even under hidden
/// check — never drops from P. See `gen_fow_pseudo_legal_moves` for the
/// composed entry point.
fn push_fow_castling_moves(setup: &Setup, color: Color, moves: &mut Vec<(u8, u8, u8)>) {
    push_castling_moves_impl(setup, color, moves, /* fow_rules */ true);
}

fn push_castling_moves_impl(
    setup: &Setup,
    color: Color,
    moves: &mut Vec<(u8, u8, u8)>,
    fow_rules: bool,
) {
    let castle_rights = setup.castling_rights;
    if castle_rights.is_empty() {
        return;
    }
    let board = &setup.board;
    let king_bb = board.by_piece(color.king());
    if king_bb.count() != 1 {
        return;
    }
    let king_sq = king_bb.first().unwrap();
    let opp = color.other();
    let all = board.occupied();

    // Standard-chess: refuse if king is in check. FoW: skip this filter
    // (king doesn't know about a hidden attacker, can still castle).
    if !fow_rules && is_square_attacked(board, king_sq, opp, all) {
        return;
    }

    for rook_sq in castle_rights {
        if board.color_at(rook_sq) != Some(color) {
            continue;
        }
        if board.role_at(rook_sq) != Some(Role::Rook) {
            continue;
        }
        if rook_sq.rank() != king_sq.rank() {
            continue;
        }
        let kingside = (rook_sq.file() as u8) > (king_sq.file() as u8);
        let (king_dest_file, rook_dest_file) = if kingside {
            (File::G, File::F)
        } else {
            (File::C, File::D)
        };
        let king_dest = Square::from_coords(king_dest_file, king_sq.rank());
        let rook_dest = Square::from_coords(rook_dest_file, king_sq.rank());

        let king_path = between_inclusive(king_sq, king_dest);
        let rook_path = between_inclusive(rook_sq, rook_dest);
        let all_path = king_path | rook_path;
        let must_be_clear = all_path
            & !(Bitboard::from_square(king_sq) | Bitboard::from_square(rook_sq));
        if (must_be_clear & all) != Bitboard::EMPTY {
            // Structural: path blocked by own piece — always rejects.
            continue;
        }

        if !fow_rules {
            // Standard-chess: refuse if any transit square is attacked. FoW:
            // skip — king can transit hidden-attacked squares.
            let mut transit_safe = true;
            for sq in king_path {
                if is_square_attacked(board, sq, opp, all) {
                    transit_safe = false;
                    break;
                }
            }
            if !transit_safe {
                continue;
            }
        }

        // python-chess emits castling as king from→to (king_dest = G or C file)
        moves.push((king_sq as u8, king_dest as u8, 0));
    }
}

/// FoW-pseudo-legal move generator: identical to `gen_pseudo_legal_moves`
/// except castling ignores the in-check / through-check / into-check filters
/// (those are standard-chess rules that don't apply in FoW because the king
/// doesn't see hidden attackers). Used by PEnumerator's belief admission so
/// the truth — which may legally castle through hidden check in FoW — stays in
/// P. The original `gen_pseudo_legal_moves` is unchanged so the WS2 search
/// tree's expansion (which uses python-chess as the parity reference) keeps
/// its byte-equivalent move set.
fn gen_fow_pseudo_legal_moves(setup: &Setup, color: Color) -> Vec<(u8, u8, u8)> {
    let mut moves = gen_pseudo_legal_moves_noncastle(setup, color);
    push_fow_castling_moves(setup, color, &mut moves);
    moves
}

/// Helper: emits every pseudo-legal non-castling move (pawns, knights,
/// bishops, rooks, queens, plain king moves, en passant). Identical to the
/// non-castling portion of `gen_pseudo_legal_moves`. Factored out so both
/// the standard and FoW variants share the same non-castling generator.
fn gen_pseudo_legal_moves_noncastle(setup: &Setup, color: Color) -> Vec<(u8, u8, u8)> {
    let mut full = gen_pseudo_legal_moves(setup, color);
    // Castling moves emitted by push_castling_moves are king-move-2-files; strip
    // them and let the caller re-add via its preferred variant. (Cheaper than
    // duplicating the entire generator body.)
    let board = &setup.board;
    let king_bb = board.by_piece(color.king());
    if king_bb.count() == 1 {
        let king_sq = king_bb.first().unwrap() as u8;
        full.retain(|(f, t, _)| {
            !(*f == king_sq && {
                let from_file = (king_sq & 7) as i32;
                let to_file = (*t & 7) as i32;
                (from_file - to_file).abs() == 2
            })
        });
    }
    full
}

/// Pseudo-legal moves in python-chess `generate_pseudo_legal_moves` ORDER.
///
/// `gen_pseudo_legal_moves` produces the correct SET but in its own order (it
/// was only ever used for set-membership — own-move admission, explosion dedup).
/// The WS2 Rust tree must iterate a node's children in python-chess order
/// because external-sampling maps an RNG draw to a child by INDEX, so a
/// different order samples a different child. We reorder the proven set rather
/// than re-deriving generation: classify each move into python-chess's emission
/// groups, then sort by (group, from desc, to desc, promotion Q<R<B<N).
///
/// python-chess emits, in order: (0) non-pawn piece moves [incl. normal king],
/// (1) castling, (2) normal pawn captures, (3) single pawn advances, (4) double
/// pawn advances, (5) en passant — each `scan_reversed` (high square first),
/// promotions Q,R,B,N. Within every group that is exactly (from desc, to desc),
/// and castling/ep/advances have a constant from or to, so one key orders all.
fn gen_fow_pseudo_legal_moves_pychess_order(setup: &Setup) -> Vec<(u8, u8, u8)> {
    // FoW move space (2026-06-20 castle-into-check fix): use gen_fow_pseudo_legal_moves
    // so the WS2 search sees fog-castling-into-check like belief admission already does
    // (gen_fow at 715/917/1105). Was gen_pseudo_legal_moves (python-chess parity), which
    // excluded castle-into-check and blinded the search to castle-into-hidden-attack
    // king-losses (game a6f2e491). Fog-castles are king-2-file moves so they land in the
    // castling group (1) of the sort below — same order convention, parity preserved.
    let mut moves = gen_fow_pseudo_legal_moves(setup, setup.turn);
    let board = &setup.board;
    moves.sort_by_key(|&(f, t, p)| -> (u8, i16, i16, u8) {
        let from = unsafe { Square::new_unchecked(f as u32) };
        let to = unsafe { Square::new_unchecked(t as u32) };
        let from_file = from.file() as i32;
        let to_file = to.file() as i32;
        let from_rank = from.rank() as i32;
        let to_rank = to.rank() as i32;
        let group: u8 = match board.role_at(from) {
            // castling: king moves two files (push_castling_moves emits king→G/C)
            Some(Role::King) if (from_file - to_file).abs() == 2 => 1,
            Some(Role::Pawn) => {
                if from_file == to_file {
                    if (from_rank - to_rank).abs() == 2 { 4 } else { 3 }
                } else if board.piece_at(to).is_some() {
                    2 // normal capture (target occupied)
                } else {
                    5 // en passant (diagonal to an empty square)
                }
            }
            _ => 0, // non-pawn piece move (incl. normal 1-square king moves)
        };
        // python-chess yields promotions Q,R,B,N; codes are Q=5,R=4,B=3,N=2, so
        // 5 - code maps Q→0 (first) .. N→3. No promotion (code 0) → 0 (no tie).
        let promo_order = if p == 0 { 0 } else { 5u8.saturating_sub(p) };
        (group, -(f as i16), -(t as i16), promo_order)
    });
    moves
}

fn parse_fen_lenient(fen: &str) -> PyResult<Setup> {
    let fen_parsed = Fen::from_ascii(fen.as_bytes()).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("bad FEN: {e}"))
    })?;
    Ok(fen_parsed.into_setup())
}

/// Build a Setup from the same 10 bitboard-style inputs `visible_squares_bb`
/// takes. Factored out so `consistent_with_bb` can reuse the same parsing
/// path before calling `visible_squares_from_setup`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn setup_from_bb(
    pawns: u64,
    knights: u64,
    bishops: u64,
    rooks: u64,
    queens: u64,
    kings: u64,
    occupied_white: u64,
    occupied_black: u64,
    castling_rights: u64,
    ep_square_idx: u32,
) -> Setup {
    let mut setup = Setup::empty();
    setup.board = ShakBoard::from_bitboards(
        shakmaty::ByRole {
            pawn: Bitboard(pawns),
            knight: Bitboard(knights),
            bishop: Bitboard(bishops),
            rook: Bitboard(rooks),
            queen: Bitboard(queens),
            king: Bitboard(kings),
        },
        shakmaty::ByColor {
            white: Bitboard(occupied_white),
            black: Bitboard(occupied_black),
        },
    );
    setup.castling_rights = Bitboard(castling_rights);
    setup.ep_square = if ep_square_idx < 64 {
        Some(unsafe { Square::new_unchecked(ep_square_idx) })
    } else {
        None
    };
    setup
}

/// Drives the full `update_own_move` inner loop in Rust.
///
/// For each FEN in `prev_fens` where the side-to-move is the perspective
/// player:
///   1. Check that `(from, to, promo)` is pseudo-legal from this prev
///   2. If legal, apply the move to get the next position
///   3. Serialize the next position back to FEN
///
/// Returns the union of resulting next-FENs across all consistent prev
/// positions. The caller dedups via `set()`. 1-to-1 mapping per prev
/// (no move branching, no observation filter) — much simpler than
/// `update_opp_move_rust` but with the same parallel structure.
///
/// Drops prevs where:
///   - Turn ≠ perspective (defensive; shouldn't happen if caller alternates)
///   - The given move isn't pseudo-legal from this prev (the truth must
///     have admitted this move, so any prev that doesn't is dropped)
/// Core of update_own_move, operating on a slice (reused by the stateless
/// #[pyfunction] and the stateful PEnumState). Returns (unique_next_fens, raw).
/// Visibility-only consistency (conditions 1+2 of consistent_with_setup): the
/// next position's visible squares + visible pieces match the observation. Used
/// to filter the belief by the PERSPECTIVE's OWN post-move observation — own
/// moves are deterministic (no enemy-capture conditions apply), so this is the
/// right filter (the capture conditions in consistent_with_setup are for the
/// OPPONENT's move and would misfire on the moved-own-piece's vacated square).
fn visible_consistent(
    next: &Setup, perspective: Color, obs_vis: u64,
    obs_w: &[u64; 6], obs_b: &[u64; 6],
) -> bool {
    if visible_squares_from_setup(next, perspective) != obs_vis {
        return false;
    }
    let board = &next.board;
    let v = obs_vis;
    let white = board.by_color(Color::White).0;
    let black = board.by_color(Color::Black).0;
    let role = [
        board.by_role(Role::Pawn).0, board.by_role(Role::Knight).0,
        board.by_role(Role::Bishop).0, board.by_role(Role::Rook).0,
        board.by_role(Role::Queen).0, board.by_role(Role::King).0,
    ];
    for i in 0..6 {
        if (role[i] & white & v) != obs_w[i] { return false; }
        if (role[i] & black & v) != obs_b[i] { return false; }
    }
    true
}

fn own_move_core(
    prev: &[PackedPos],
    perspective_white: bool,
    from_idx: u8,
    to_idx: u8,
    promo: u8,
    // Some => ALSO filter each successor by the perspective's post-own-move
    // observation (visibility, white piece masks, black piece masks). The
    // two-step belief update: apply the deterministic own move, then prune
    // positions inconsistent with the squares the move just revealed.
    obs_filter: Option<(u64, [u64; 6], [u64; 6])>,
) -> PyResult<(Vec<PackedPos>, usize)> {
    let perspective = if perspective_white { Color::White } else { Color::Black };
    let from = unsafe { Square::new_unchecked(from_idx as u32) };
    let to = unsafe { Square::new_unchecked(to_idx as u32) };
    let promo_role = role_from_int(promo);
    let merged: Result<(FxHashSet<PackedPos>, usize), PyErr> = prev
        .par_iter()
        .map(|pp| -> Result<(FxHashSet<PackedPos>, usize), PyErr> {
            let prev_setup = unpack(pp);
            let mut local: FxHashSet<PackedPos> = FxHashSet::default();
            if prev_setup.turn != perspective {
                return Ok((local, 0));
            }
            // FoW-rules move generator: standard chess refuses castle-out-of-
            // check / through-check, but in FoW those are legal because the
            // king doesn't see hidden attackers. Using the FoW variant keeps
            // the truth in P when the perspective played a FoW-only-legal
            // castle (caught by the seed_idx=7 capture, see commit msg).
            let moves = gen_fow_pseudo_legal_moves(&prev_setup, perspective);
            let move_admitted = moves
                .iter()
                .any(|(f, t, p)| *f == from_idx && *t == to_idx && *p == promo);
            if !move_admitted {
                return Ok((local, 0));
            }
            let next_setup = apply_move_to_setup(&prev_setup, from, to, promo_role);
            if let Some((vis, ow, ob)) = obs_filter {
                if !visible_consistent(&next_setup, perspective, vis, &ow, &ob) {
                    return Ok((local, 0));
                }
            }
            local.insert(pack(&next_setup));
            Ok((local, 1))
        })
        .try_reduce(
            || (FxHashSet::default(), 0usize),
            |(mut a_set, a_n), (b_set, b_n)| {
                let n = a_n + b_n;
                if b_set.len() > a_set.len() {
                    let mut b_set = b_set;
                    b_set.extend(a_set);
                    Ok((b_set, n))
                } else {
                    a_set.extend(b_set);
                    Ok((a_set, n))
                }
            },
        );
    let (set, raw) = merged?;
    Ok((set.into_iter().collect(), raw))
}

#[pyfunction]
fn update_own_move_rust(
    prev_fens: Vec<String>,
    perspective_white: bool,
    from_idx: u8,
    to_idx: u8,
    promo: u8,
) -> PyResult<(Vec<String>, usize)> {
    // FEN-boundary wrapper (legacy PEnumerator path): pack inputs, run the
    // packed core, decode the deduped result back to FENs. PEnumState calls the
    // core directly and keeps positions packed (no per-ply FEN round-trip).
    let prev: Vec<PackedPos> = prev_fens
        .par_iter()
        .map(|f| Ok(pack(&parse_fen_lenient(f)?)))
        .collect::<PyResult<_>>()?;
    let (next, raw) = own_move_core(&prev, perspective_white, from_idx, to_idx, promo, None)?;
    let fens: Vec<String> = next.into_par_iter().map(|p| Fen(unpack(&p)).to_string()).collect();
    Ok((fens, raw))
}

/// Drives the full `update_opp_move` inner loop in Rust.
///
/// For each FEN in `prev_fens` where the side-to-move is the opponent:
///   1. Enumerate the opponent's pseudo-legal moves
///   2. For each move, apply it to get the next position
///   3. Check observation-consistency (visibility, pieces, captures)
///   4. If consistent, serialize the next position back to FEN
///
/// Returns the union of consistent next-FENs across all prev positions.
/// The caller dedups via set membership on the returned Vec.
///
/// Observation data is passed as 12 piece-color bitmasks (pre-extracted by
/// the Python caller from `observation.visible_pieces`) + the visibility
/// mask + two capture-square ints (i32 with -1 = None). Matches
/// `consistent_with_bb` semantics; pre-extraction keeps the per-call arg
/// list flat and avoids repeated dict iteration per (prev, move) pair.
#[allow(clippy::too_many_arguments)]
fn opp_move_core(
    prev: &[PackedPos],
    // Bottom-K (KMV) belief bound. None => exact: build the full unique
    // consistent set M (today's behavior, byte-identical). Some(k) => keep only
    // the k smallest-hash worlds, bounding peak memory at ~2k DURING the build
    // instead of materializing all M (which reaches 4.2x the cap on explosion
    // plies). The kept set is an exactly-uniform k-subset either way.
    cap: Option<usize>,
    opp_white: bool,
    perspective_white: bool,
    obs_visibility_mask: u64,
    obs_white_pawns: u64,
    obs_white_knights: u64,
    obs_white_bishops: u64,
    obs_white_rooks: u64,
    obs_white_queens: u64,
    obs_white_kings: u64,
    obs_black_pawns: u64,
    obs_black_knights: u64,
    obs_black_bishops: u64,
    obs_black_rooks: u64,
    obs_black_queens: u64,
    obs_black_kings: u64,
    obs_own_capture_idx: i32,
    obs_opp_capture_landing_idx: i32,
    // Returns (positions, raw, pre_cap_count, was_downsampled). pre_cap_count is
    // exact |M| when uncapped or not downsampled, else the KMV cardinality
    // estimate (M is never built under the cap).
) -> PyResult<(Vec<PackedPos>, usize, usize, bool)> {
    let opp = if opp_white { Color::White } else { Color::Black };
    let perspective = if perspective_white { Color::White } else { Color::Black };

    // Parallel outer loop. Each prev_fen is independent: parse, generate
    // pseudo-legal moves, apply each, check consistency, collect the
    // kept-next-FENs into a per-prev local Vec. rayon merges the per-prev
    // Vecs into the final result. Order is not preserved (Python wraps the
    // result in `set()`, so order doesn't matter for downstream correctness)
    // and there is no shared mutable state — the work is embarrassingly
    // parallel.
    //
    // Errors (bad FEN parse) propagate via `Result` — rayon's
    // `collect::<Result<_, _>>` short-circuits on the first error.
    // Dedup INSIDE Rust: different (prev, move) pairs frequently land on the
    // same next position, so the consistent-successor multiset has heavy
    // duplication (observed ~3x: 13.2M raw vs 4.28M unique on an explosion
    // ply). Accumulating into a per-thread HashSet and merging via try_reduce
    // means the duplicated multiset is never materialized — only the unique
    // set crosses the FFI boundary. Previously each thread built a Vec and the
    // full duplicated list was handed to Python, which then `set()`-deduped it,
    // double-holding ~13.2M FEN strings (the source of the ~8 GB RSS spike).
    //
    // The dedup key is the FEN string — identical to Python's prior `set()`
    // semantics — so the returned set is byte-for-byte the same membership the
    // old path produced. Verified by tests/test_rust_update_opp_dedup_diff.py.
    // Each thread accumulates (unique-set, raw-count): raw counts every
    // observation-consistent successor BEFORE dedup, so callers keep the
    // duplication-factor diagnostic (raw / unique) that quantifies the win on
    // every explosion ply. try_reduce unions the sets and sums the counts.
    // Observation-guided pre-pruning: derive cheap NECESSARY conditions from
    // the move (from, to, promo) and the observation, hoisted BEFORE the
    // expensive `apply_move_to_setup` (full Setup clone) + `consistent_with_setup`
    // (from-scratch visibility recompute). Each check only ever skips a
    // candidate the full check would also reject, so the kept set is
    // byte-identical (validated by FEN-hash parity). The win is fewer of the
    // ~68M per-explosion candidate applies/visibility-recomputes — it cuts the
    // candidate COUNT, not the per-candidate cost.
    let obs_w = [obs_white_pawns, obs_white_knights, obs_white_bishops,
                 obs_white_rooks, obs_white_queens, obs_white_kings];
    let obs_b = [obs_black_pawns, obs_black_knights, obs_black_bishops,
                 obs_black_rooks, obs_black_queens, obs_black_kings];
    let all_obs_occ = obs_w.iter().chain(obs_b.iter()).fold(0u64, |a, b| a | b);
    let opp_obs = if opp == Color::White { &obs_w } else { &obs_b };
    let role_idx = |r: Role| -> usize {
        match r {
            Role::Pawn => 0, Role::Knight => 1, Role::Bishop => 2,
            Role::Rook => 3, Role::Queen => 4, Role::King => 5,
        }
    };
    let record_stats = std::env::var_os("FOW_PRUNE_STATS").is_some();
    let do_prune = std::env::var_os("FOW_NO_PRUNE").is_none();

    // Single CONCURRENT sharded set (DashSet): threads dedup IMMEDIATELY across
    // each other. The old design had each of ~N threads build its own
    // FxHashSet then merge via try_reduce — which held the full multi-million
    // CONSISTENT multiset spread across the N per-thread sets (cross-thread
    // duplicates aren't removed until the merge), the dominant explosion-ply
    // transient (e.g. 137M consistent -> only 27M unique, but ~137M held before
    // merge). The DashSet stores only the UNIQUE kept set as it's built, with no
    // separate merge pass. raw / pre-pruned counts via atomics. Membership is
    // identical to the old set -> byte-parity (FEN-hash / suite).
    let kept: DashSet<PackedPos, FxBuildHasher> =
        DashSet::with_hasher(FxBuildHasher::default());
    let raw = AtomicUsize::new(0);
    let pruned = AtomicUsize::new(0);
    // Bottom-K state (only active when `cap` is Some). `threshold` is the
    // current bottom-K hash ceiling (monotone non-increasing); `compacting`
    // serializes the retain pass; `downsampled` records whether any world was
    // ever dropped by the cap.
    let threshold = AtomicU64::new(u64::MAX);
    let compacting = AtomicBool::new(false);
    let downsampled = AtomicBool::new(false);
    prev.par_iter().for_each_init(
        || 0usize,
        |since_check: &mut usize, pp| {
        let prev_setup = unpack(pp);
        if prev_setup.turn != opp {
            return;
        }
        let prev_own_occ = prev_setup.board.by_color(perspective).0;
        let prev_occ = prev_setup.board.occupied().0;
        let mut local_raw = 0usize;
        let mut local_pruned = 0usize;

        // FoW-rules move generator: in FoW, opp may legally castle while in
        // check / through attacked squares (they don't see our hidden pieces).
        // gen_pseudo_legal_moves would refuse, dropping the truth's predecessor
        // from P and corrupting belief downstream. See gen_fow_pseudo_legal_moves.
        let moves = gen_fow_pseudo_legal_moves(&prev_setup, opp);
        for (from_idx, to_idx, promo) in moves {
            let from = unsafe { Square::new_unchecked(from_idx as u32) };
            let to = unsafe { Square::new_unchecked(to_idx as u32) };
            let from_bb = 1u64 << (from_idx as u32);
            let to_bb = 1u64 << (to_idx as u32);
            let promo_role = role_from_int(promo);

            if do_prune {
                let moving_role = match prev_setup.board.role_at(from) {
                    Some(r) => r,
                    None => continue, // empty from-square: not a real move
                };
                let placed_role = promo_role.unwrap_or(moving_role);
                let is_castling = moving_role == Role::King
                    && (from.file() as i32 - to.file() as i32).abs() == 2;

                // (A) Own-capture — EXACT for consistency condition (3).
                let captured_idx: i32 = if is_castling {
                    -1
                } else if moving_role == Role::Pawn
                    && prev_setup.ep_square == Some(to)
                    && (prev_occ & to_bb) == 0
                {
                    u8::from(Square::from_coords(to.file(), from.rank())) as i32
                } else if (prev_own_occ & to_bb) != 0 {
                    to_idx as i32
                } else {
                    -1
                };
                if captured_idx != obs_own_capture_idx {
                    local_pruned += 1;
                    continue;
                }
                // (B) Visible landing — necessary.
                if (to_bb & obs_visibility_mask) != 0
                    && (to_bb & opp_obs[role_idx(placed_role)]) == 0
                {
                    local_pruned += 1;
                    continue;
                }
                // (C) Vacated from-square — necessary.
                if (from_bb & obs_visibility_mask) != 0 && (from_bb & all_obs_occ) != 0 {
                    local_pruned += 1;
                    continue;
                }
            }

            let next_setup = apply_move_to_setup(&prev_setup, from, to, promo_role);

            if !consistent_with_setup(
                &next_setup,
                perspective,
                prev_own_occ,
                obs_visibility_mask,
                obs_white_pawns, obs_white_knights, obs_white_bishops,
                obs_white_rooks, obs_white_queens, obs_white_kings,
                obs_black_pawns, obs_black_knights, obs_black_bishops,
                obs_black_rooks, obs_black_queens, obs_black_kings,
                obs_own_capture_idx,
                obs_opp_capture_landing_idx,
            ) {
                continue;
            }

            local_raw += 1;
            let packed = pack(&next_setup);
            match cap {
                None => {
                    kept.insert(packed);
                }
                Some(k) => {
                    // Lock-free reject of worlds that can't be in the bottom-K.
                    if hash_packed(&packed) >= threshold.load(Ordering::Relaxed) {
                        continue;
                    }
                    kept.insert(packed);
                    *since_check += 1;
                    if *since_check >= BOTTOMK_CHECK_INTERVAL {
                        *since_check = 0;
                        compact_bottom_k(&kept, k, &threshold, &compacting,
                                         &downsampled, false);
                    }
                }
            }
        }
        raw.fetch_add(local_raw, Ordering::Relaxed);
        pruned.fetch_add(local_pruned, Ordering::Relaxed);
    });

    // Final compaction: lands the result at EXACTLY the bottom-K (periodic
    // passes only fire above 2k, so |kept| can sit in (k, 2k] here).
    if let Some(k) = cap {
        compact_bottom_k(&kept, k, &threshold, &compacting, &downsampled, true);
    }

    let raw = raw.into_inner();
    let pruned = pruned.into_inner();
    let was_downsampled = downsampled.into_inner();
    if record_stats {
        let candidates = raw + pruned;
        let pct = if candidates > 0 {
            100.0 * (pruned as f64) / (candidates as f64)
        } else {
            0.0
        };
        eprintln!(
            "[prune] |P_prev|={} candidates={} pre_pruned={} ({:.1}%) consistent={} kept={}",
            prev.len(), candidates, pruned, pct, raw, kept.len()
        );
    }
    let final_len = kept.len();
    // pre_cap_count = exact |M| when we didn't downsample; else the KMV
    // cardinality estimate M_est ~= k * 2^64 / tau (the standard bottom-k
    // estimator). M itself is never materialized under the cap.
    let pre_cap_count = if was_downsampled {
        let tau = threshold.load(Ordering::Relaxed);
        if tau == 0 {
            final_len
        } else {
            ((final_len as u128) * (u64::MAX as u128) / (tau as u128)) as usize
        }
    } else {
        final_len
    };
    Ok((kept.into_iter().collect(), raw, pre_cap_count, was_downsampled))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn update_opp_move_rust(
    prev_fens: Vec<String>,
    opp_white: bool,
    perspective_white: bool,
    obs_visibility_mask: u64,
    obs_white_pawns: u64, obs_white_knights: u64, obs_white_bishops: u64,
    obs_white_rooks: u64, obs_white_queens: u64, obs_white_kings: u64,
    obs_black_pawns: u64, obs_black_knights: u64, obs_black_bishops: u64,
    obs_black_rooks: u64, obs_black_queens: u64, obs_black_kings: u64,
    obs_own_capture_idx: i32,
    obs_opp_capture_landing_idx: i32,
) -> PyResult<(Vec<String>, usize)> {
    // FEN-boundary wrapper (legacy PEnumerator path); see update_own_move_rust.
    let prev: Vec<PackedPos> = prev_fens
        .par_iter()
        .map(|f| Ok(pack(&parse_fen_lenient(f)?)))
        .collect::<PyResult<_>>()?;
    // Legacy FEN path: uncapped (cap=None); the Python PEnumerator caps via
    // _maybe_downsample. Bottom-K is wired through the PEnumState path only.
    let (next, raw, _pre_cap, _ds) = opp_move_core(
        &prev, None, opp_white, perspective_white, obs_visibility_mask,
        obs_white_pawns, obs_white_knights, obs_white_bishops,
        obs_white_rooks, obs_white_queens, obs_white_kings,
        obs_black_pawns, obs_black_knights, obs_black_bishops,
        obs_black_rooks, obs_black_queens, obs_black_kings,
        obs_own_capture_idx, obs_opp_capture_landing_idx,
    )?;
    let fens: Vec<String> = next.into_par_iter().map(|p| Fen(unpack(&p)).to_string()).collect();
    Ok((fens, raw))
}

/// Shared consistency check that works on a `Setup` rather than raw
/// bitboards. Used by the top-level Rust loop driver to avoid extracting
/// bitboards from Setup just to re-pack them as function args.
#[inline]
#[allow(clippy::too_many_arguments)]
fn consistent_with_setup(
    next: &Setup,
    perspective: Color,
    prev_own_occ: u64,
    obs_visibility_mask: u64,
    obs_white_pawns: u64,
    obs_white_knights: u64,
    obs_white_bishops: u64,
    obs_white_rooks: u64,
    obs_white_queens: u64,
    obs_white_kings: u64,
    obs_black_pawns: u64,
    obs_black_knights: u64,
    obs_black_bishops: u64,
    obs_black_rooks: u64,
    obs_black_queens: u64,
    obs_black_kings: u64,
    obs_own_capture_idx: i32,
    obs_opp_capture_landing_idx: i32,
) -> bool {
    let visible = visible_squares_from_setup(next, perspective);
    if visible != obs_visibility_mask {
        return false;
    }

    let board = &next.board;
    let v = obs_visibility_mask;
    let white = board.by_color(Color::White).0;
    let black = board.by_color(Color::Black).0;

    let pawns = board.by_role(Role::Pawn).0;
    let knights = board.by_role(Role::Knight).0;
    let bishops = board.by_role(Role::Bishop).0;
    let rooks = board.by_role(Role::Rook).0;
    let queens = board.by_role(Role::Queen).0;
    let kings = board.by_role(Role::King).0;

    if (pawns & white & v) != obs_white_pawns { return false; }
    if (knights & white & v) != obs_white_knights { return false; }
    if (bishops & white & v) != obs_white_bishops { return false; }
    if (rooks & white & v) != obs_white_rooks { return false; }
    if (queens & white & v) != obs_white_queens { return false; }
    if (kings & white & v) != obs_white_kings { return false; }
    if (pawns & black & v) != obs_black_pawns { return false; }
    if (knights & black & v) != obs_black_knights { return false; }
    if (bishops & black & v) != obs_black_bishops { return false; }
    if (rooks & black & v) != obs_black_rooks { return false; }
    if (queens & black & v) != obs_black_queens { return false; }
    if (kings & black & v) != obs_black_kings { return false; }

    let next_own_occ = if perspective == Color::White { white } else { black };
    let captures = prev_own_occ & !next_own_occ;
    if obs_own_capture_idx < 0 {
        if captures != 0 {
            return false;
        }
    } else {
        let expected = 1u64 << (obs_own_capture_idx as u32);
        if captures != expected {
            return false;
        }
    }

    if obs_opp_capture_landing_idx >= 0 {
        let sq_bb = 1u64 << (obs_opp_capture_landing_idx as u32);
        let next_opp_occ = if perspective == Color::White { black } else { white };
        if (next_opp_occ & sq_bb) == 0 {
            return false;
        }
    }

    true
}

/// Hot-path observation-consistency check.
///
/// Equivalent to `fow_chess.observation.consistent_with(next, prev, obs, perspective)`,
/// but takes raw bitboards so the caller never has to serialize boards to FEN
/// or build piece maps in Python. The four observation properties are
/// pre-extracted by the Python caller and passed as 12 piece-color bitmasks
/// + two capture-square ints + visibility mask.
///
/// Returns true iff every condition holds:
///   1. `visible_squares(next, perspective) == obs.visibility_mask`
///   2. For each (piece-type, color), the next-board pieces masked by visibility
///      equal `obs.visible_pieces` restricted to that (piece-type, color)
///   3. Captures of perspective's pieces equal expectation (None → no captures;
///      Some(sq) → exactly that one square was emptied)
///   4. If `opp_capture_landing_idx >= 0`, an opponent piece is on that square
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn consistent_with_bb(
    next_pawns: u64,
    next_knights: u64,
    next_bishops: u64,
    next_rooks: u64,
    next_queens: u64,
    next_kings: u64,
    next_occ_white: u64,
    next_occ_black: u64,
    next_castling_rights: u64,
    next_ep_square_idx: u32,
    prev_own_occ: u64,
    obs_visibility_mask: u64,
    obs_white_pawns: u64,
    obs_white_knights: u64,
    obs_white_bishops: u64,
    obs_white_rooks: u64,
    obs_white_queens: u64,
    obs_white_kings: u64,
    obs_black_pawns: u64,
    obs_black_knights: u64,
    obs_black_bishops: u64,
    obs_black_rooks: u64,
    obs_black_queens: u64,
    obs_black_kings: u64,
    obs_own_capture_idx: i32,
    obs_opp_capture_landing_idx: i32,
    perspective_white: bool,
) -> bool {
    let perspective = if perspective_white { Color::White } else { Color::Black };

    // (1) Visibility match
    let setup = setup_from_bb(
        next_pawns,
        next_knights,
        next_bishops,
        next_rooks,
        next_queens,
        next_kings,
        next_occ_white,
        next_occ_black,
        next_castling_rights,
        next_ep_square_idx,
    );
    let visible = visible_squares_from_setup(&setup, perspective);
    if visible != obs_visibility_mask {
        return false;
    }

    // (2) Visible pieces: every (piece-type, color) masked by visibility equals obs
    let nw_pawn = next_pawns & next_occ_white;
    let nw_knight = next_knights & next_occ_white;
    let nw_bishop = next_bishops & next_occ_white;
    let nw_rook = next_rooks & next_occ_white;
    let nw_queen = next_queens & next_occ_white;
    let nw_king = next_kings & next_occ_white;
    let nb_pawn = next_pawns & next_occ_black;
    let nb_knight = next_knights & next_occ_black;
    let nb_bishop = next_bishops & next_occ_black;
    let nb_rook = next_rooks & next_occ_black;
    let nb_queen = next_queens & next_occ_black;
    let nb_king = next_kings & next_occ_black;

    let v = obs_visibility_mask;
    if (nw_pawn & v) != obs_white_pawns { return false; }
    if (nw_knight & v) != obs_white_knights { return false; }
    if (nw_bishop & v) != obs_white_bishops { return false; }
    if (nw_rook & v) != obs_white_rooks { return false; }
    if (nw_queen & v) != obs_white_queens { return false; }
    if (nw_king & v) != obs_white_kings { return false; }
    if (nb_pawn & v) != obs_black_pawns { return false; }
    if (nb_knight & v) != obs_black_knights { return false; }
    if (nb_bishop & v) != obs_black_bishops { return false; }
    if (nb_rook & v) != obs_black_rooks { return false; }
    if (nb_queen & v) != obs_black_queens { return false; }
    if (nb_king & v) != obs_black_kings { return false; }

    // (3) Captures of perspective's own pieces
    let next_own_occ = if perspective_white { next_occ_white } else { next_occ_black };
    let captures = prev_own_occ & !next_own_occ;
    if obs_own_capture_idx < 0 {
        if captures != 0 {
            return false;
        }
    } else {
        let expected = 1u64 << (obs_own_capture_idx as u32);
        if captures != expected {
            return false;
        }
    }

    // (4) Opp capture landing — opponent piece must sit there
    if obs_opp_capture_landing_idx >= 0 {
        let sq_bb = 1u64 << (obs_opp_capture_landing_idx as u32);
        let next_opp_occ = if perspective_white { next_occ_black } else { next_occ_white };
        if (next_opp_occ & sq_bb) == 0 {
            return false;
        }
    }

    true
}

fn visible_squares_from_setup(setup: &Setup, color: Color) -> u64 {
    let board: &ShakBoard = &setup.board;
    let own = board.by_color(color);
    let opp = board.by_color(color.other());
    let all = own | opp;

    let mut visible: Bitboard = own;

    // Pawns — pushes (if empty) + diagonal attacks (if enemy)
    for sq in board.by_piece(color.pawn()) {
        if let Some(p1) = pawn_push(sq, color) {
            if !all.contains(p1) {
                visible.add(p1);
                let on_start_rank = match color {
                    Color::White => sq.rank() == Rank::Second,
                    Color::Black => sq.rank() == Rank::Seventh,
                };
                if on_start_rank {
                    if let Some(p2) = pawn_push(p1, color) {
                        if !all.contains(p2) {
                            visible.add(p2);
                        }
                    }
                }
            }
        }
        let attacks_mask = attacks::pawn_attacks(color, sq);
        for atk_sq in attacks_mask {
            if opp.contains(atk_sq) {
                visible.add(atk_sq);
            }
        }
    }

    // Knights
    for sq in board.by_piece(color.knight()) {
        visible = visible | attacks::knight_attacks(sq);
    }

    // Bishops + queens (diagonals)
    for sq in board.by_piece(color.bishop()) {
        visible = visible | attacks::bishop_attacks(sq, all);
    }
    for sq in board.by_piece(color.queen()) {
        visible = visible | attacks::bishop_attacks(sq, all);
        visible = visible | attacks::rook_attacks(sq, all);
    }

    // Rooks
    for sq in board.by_piece(color.rook()) {
        visible = visible | attacks::rook_attacks(sq, all);
    }

    // King (attacks; castling handled below)
    for sq in board.by_piece(color.king()) {
        visible = visible | attacks::king_attacks(sq);
    }

    // En passant — add landing square + captured pawn square.
    // setup.ep_square is the FEN ep target (LANDING square). The ep RIGHT
    // belongs to whichever color can move forward to that square:
    //   - ep_target on rank 6 → only WHITE can capture
    //   - ep_target on rank 3 → only BLACK can capture
    // Other ranks (set on weird FENs) mean no real ep right for either side.
    if let Some(ep_target) = setup.ep_square {
        let ep_rank_idx = ep_target.rank() as u8;
        let valid_for_color = match color {
            Color::White => ep_rank_idx == 5, // rank 6 (0-indexed = 5)
            Color::Black => ep_rank_idx == 2, // rank 3 (0-indexed = 2)
        };
        if valid_for_color {
            // Capturing pawn must be on the rank OF the captured pawn.
            // For white capturing: ep_target rank 6, captured pawn rank 5, capturing pawn rank 5.
            // For black capturing: ep_target rank 3, captured pawn rank 4, capturing pawn rank 4.
            let pawn_rank_idx = match color {
                Color::White => ep_rank_idx - 1,
                Color::Black => ep_rank_idx + 1,
            };
            let pawn_rank = Rank::new(pawn_rank_idx as u32);
            for adj_file in adjacent_files(ep_target.file()) {
                let pawn_sq = Square::from_coords(adj_file, pawn_rank);
                if board.by_piece(color.pawn()).contains(pawn_sq) {
                    visible.add(ep_target); // landing square (Python: move.to_square)
                    let captured = Square::from_coords(ep_target.file(), pawn_rank);
                    visible.add(captured); // captured pawn's square
                }
            }
        }
    }

    add_castling_visibility(&mut visible, setup, color);

    visible.into()
}

fn pawn_push(from: Square, color: Color) -> Option<Square> {
    let new_rank_idx = match color {
        Color::White => (from.rank() as u8).checked_add(1)?,
        Color::Black => (from.rank() as u8).checked_sub(1)?,
    };
    if new_rank_idx > 7 {
        return None;
    }
    let rank = Rank::new(new_rank_idx as u32);
    Some(Square::from_coords(from.file(), rank))
}

fn adjacent_files(file: File) -> impl Iterator<Item = File> {
    let idx = file as i32;
    [idx - 1, idx + 1]
        .into_iter()
        .filter(|f| (0..8).contains(f))
        .map(|f| File::new(f as u32))
}

fn add_castling_visibility(visible: &mut Bitboard, setup: &Setup, color: Color) {
    let castle_rights = setup.castling_rights;
    if castle_rights.is_empty() {
        return;
    }
    let board = &setup.board;
    let king_bb = board.by_piece(color.king());
    if king_bb.count() != 1 {
        return; // 0 or 2+ kings — skip
    }
    let king_sq = king_bb.first().unwrap();
    let opp = color.other();
    let all = board.occupied();

    // Castle out of check is illegal
    if is_square_attacked(board, king_sq, opp, all) {
        return;
    }

    for rook_sq in castle_rights {
        if board.color_at(rook_sq) != Some(color) {
            continue;
        }
        if board.role_at(rook_sq) != Some(Role::Rook) {
            continue;
        }
        if rook_sq.rank() != king_sq.rank() {
            continue;
        }
        let kingside = (rook_sq.file() as u8) > (king_sq.file() as u8);
        let (king_dest_file, rook_dest_file) = if kingside {
            (File::G, File::F)
        } else {
            (File::C, File::D)
        };
        let king_dest = Square::from_coords(king_dest_file, king_sq.rank());
        let rook_dest = Square::from_coords(rook_dest_file, king_sq.rank());

        let king_path = between_inclusive(king_sq, king_dest);
        let rook_path = between_inclusive(rook_sq, rook_dest);
        let all_path = king_path | rook_path;
        let must_be_clear = all_path
            & !(Bitboard::from_square(king_sq) | Bitboard::from_square(rook_sq));
        if (must_be_clear & all) != Bitboard::EMPTY {
            continue;
        }

        let mut transit_safe = true;
        for sq in king_path {
            if is_square_attacked(board, sq, opp, all) {
                transit_safe = false;
                break;
            }
        }
        if !transit_safe {
            continue;
        }

        visible.add(rook_sq);
    }
}

fn between_inclusive(a: Square, b: Square) -> Bitboard {
    let rank = a.rank();
    let f_lo = (a.file() as u8).min(b.file() as u8);
    let f_hi = (a.file() as u8).max(b.file() as u8);
    let mut bb = Bitboard::EMPTY;
    for f in f_lo..=f_hi {
        let file = File::new(f as u32);
        bb.add(Square::from_coords(file, rank));
    }
    bb
}

fn is_square_attacked(
    board: &ShakBoard,
    sq: Square,
    by_color: Color,
    occupied: Bitboard,
) -> bool {
    let pawn_attackers = attacks::pawn_attacks(by_color.other(), sq)
        & board.by_piece(by_color.pawn());
    if pawn_attackers.any() {
        return true;
    }
    if (attacks::knight_attacks(sq) & board.by_piece(by_color.knight())).any() {
        return true;
    }
    if (attacks::king_attacks(sq) & board.by_piece(by_color.king())).any() {
        return true;
    }
    let bishops_queens = board.by_piece(by_color.bishop()) | board.by_piece(by_color.queen());
    if (attacks::bishop_attacks(sq, occupied) & bishops_queens).any() {
        return true;
    }
    let rooks_queens = board.by_piece(by_color.rook()) | board.by_piece(by_color.queen());
    if (attacks::rook_attacks(sq, occupied) & rooks_queens).any() {
        return true;
    }
    false
}

/// Construct an Observation for `color_bool` from a (prev → next) transition.
///
/// Mirrors `fow_chess.observation.observation_from_transition`. RP10
/// (2026-05-25): profile showed this function was called 10K+ times
/// per pick_move and ate 25% of wall in pure Python via python-chess
/// piece_map and visibility scans. Bitboard inputs avoid the FEN
/// round-trip overhead per RP3 pattern.
///
/// Returns a tuple Python decodes back into an Observation:
///
///   (visibility_mask: u64,
///    visible_pieces:  Vec<(square_idx, role_int, color_bool)>,
///    own_capture_square: Option<u8>,
///    opp_capture_landing_square: Option<u8>,
///    game_over_winner: Option<bool>,   // None if no game-over
///    game_over_reason: &'static str)   // "" if no game-over
///
/// `role_int` is shakmaty Role's enum value (1=Pawn .. 6=King), which
/// the Python wrapper maps to `chess.PieceType` (1=PAWN .. 6=KING).
/// Same numeric mapping — direct pass-through.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn observation_from_transition_bb(
    // PREV board — only fields needed to detect captures + king-loss.
    prev_occupied_white: u64,
    prev_occupied_black: u64,
    prev_kings: u64,
    // NEXT board — full state for visibility computation.
    next_pawns: u64,
    next_knights: u64,
    next_bishops: u64,
    next_rooks: u64,
    next_queens: u64,
    next_kings: u64,
    next_occupied_white: u64,
    next_occupied_black: u64,
    next_castling_rights: u64,
    next_ep_square_idx: u32,
    color_bool: bool,
) -> (u64, Vec<(u8, u8, bool)>, Option<u8>, Option<u8>, Option<bool>, &'static str) {
    // Visibility + visible_pieces are computed from next_board (perspective-
    // independent setup build); the rest is cheap per-color bitops. Factored
    // into obs_core_for_color so the both-perspectives entry point shares the
    // single setup_from_bb build.
    let setup = setup_from_bb(
        next_pawns, next_knights, next_bishops, next_rooks, next_queens, next_kings,
        next_occupied_white, next_occupied_black, next_castling_rights, next_ep_square_idx,
    );
    obs_core_for_color(
        &setup, color_bool,
        prev_occupied_white, prev_occupied_black, prev_kings,
        next_occupied_white, next_occupied_black, next_kings,
    )
}

/// Per-color observation core, given a pre-built `next` setup. Shared by the
/// single- and both-perspective entry points so the (perspective-independent)
/// `setup_from_bb` build happens once. Byte-identical to the prior inline body.
#[allow(clippy::too_many_arguments)]
fn obs_core_for_color(
    setup: &Setup,
    color_bool: bool,
    prev_occupied_white: u64,
    prev_occupied_black: u64,
    prev_kings: u64,
    next_occupied_white: u64,
    next_occupied_black: u64,
    next_kings: u64,
) -> (u64, Vec<(u8, u8, bool)>, Option<u8>, Option<u8>, Option<bool>, &'static str) {
    // own_capture: a square that had own piece on prev but no own piece on next.
    let (own_before, own_after) = if color_bool {
        (prev_occupied_white, next_occupied_white)
    } else {
        (prev_occupied_black, next_occupied_black)
    };
    let captures_mask = own_before & !own_after;
    let captured: Option<u8> = if captures_mask != 0 {
        Some(captures_mask.trailing_zeros() as u8)
    } else {
        None
    };

    let opp_capture_landing_square: Option<u8> = match captured {
        Some(sq) => {
            let mask = 1u64 << sq;
            let opp_occ = if color_bool { next_occupied_black } else { next_occupied_white };
            if opp_occ & mask != 0 {
                Some(sq)
            } else {
                None
            }
        }
        None => None,
    };

    let own_kings_before = prev_kings & if color_bool { prev_occupied_white } else { prev_occupied_black };
    let own_kings_after = next_kings & if color_bool { next_occupied_white } else { next_occupied_black };
    let (game_over_winner, game_over_reason): (Option<bool>, &'static str) =
        if own_kings_before != 0 && own_kings_after == 0 {
            (Some(!color_bool), "king-captured")
        } else {
            (None, "")
        };

    let color = if color_bool { Color::White } else { Color::Black };
    let visibility_mask = visible_squares_from_setup(setup, color);

    let mut visible_pieces: Vec<(u8, u8, bool)> = Vec::new();
    let board = &setup.board;
    let mut mask = visibility_mask;
    while mask != 0 {
        let sq_idx = mask.trailing_zeros() as u8;
        mask &= mask - 1; // pop lowest bit
        let sq = unsafe { Square::new_unchecked(sq_idx as u32) };
        if let Some(piece) = board.piece_at(sq) {
            let role_int: u8 = match piece.role {
                Role::Pawn => 1,
                Role::Knight => 2,
                Role::Bishop => 3,
                Role::Rook => 4,
                Role::Queen => 5,
                Role::King => 6,
            };
            let piece_color_white = matches!(piece.color, Color::White);
            visible_pieces.push((sq_idx, role_int, piece_color_white));
        }
    }

    (
        visibility_mask,
        visible_pieces,
        captured,
        opp_capture_landing_square,
        game_over_winner,
        game_over_reason,
    )
}

/// Per-color INFOSET-KEY core: like obs_core_for_color, but emits the components
/// of `walker._obs_key` directly — visibility as a raw u64 and visible pieces as
/// 12 per-(color, piece-type) bitmasks — so the hot expand_leaf path never builds
/// a Python Observation (dict of chess.Piece + SquareSet) just to reduce it to a
/// key. Byte-identical to `_obs_key(observation_from_transition(...))`.
#[allow(clippy::too_many_arguments)]
fn obs_key_core_for_color(
    setup: &Setup,
    color_bool: bool,
    prev_occupied_white: u64,
    prev_occupied_black: u64,
    prev_kings: u64,
    next_occupied_white: u64,
    next_occupied_black: u64,
    next_kings: u64,
) -> (u64, Vec<u64>, Option<u8>, Option<u8>, Option<bool>, &'static str) {
    let (own_before, own_after) = if color_bool {
        (prev_occupied_white, next_occupied_white)
    } else {
        (prev_occupied_black, next_occupied_black)
    };
    let captures_mask = own_before & !own_after;
    let captured: Option<u8> = if captures_mask != 0 {
        Some(captures_mask.trailing_zeros() as u8)
    } else {
        None
    };
    let opp_capture_landing_square: Option<u8> = match captured {
        Some(sq) => {
            let mask = 1u64 << sq;
            let opp_occ = if color_bool { next_occupied_black } else { next_occupied_white };
            if opp_occ & mask != 0 { Some(sq) } else { None }
        }
        None => None,
    };
    let own_kings_before = prev_kings & if color_bool { prev_occupied_white } else { prev_occupied_black };
    let own_kings_after = next_kings & if color_bool { next_occupied_white } else { next_occupied_black };
    let (game_over_winner, game_over_reason): (Option<bool>, &'static str) =
        if own_kings_before != 0 && own_kings_after == 0 {
            (Some(!color_bool), "king-captured")
        } else {
            (None, "")
        };

    let color = if color_bool { Color::White } else { Color::Black };
    let visibility_mask = visible_squares_from_setup(setup, color);

    // 12 per-(color, piece-type) bitmasks: white P..K (0..5), black P..K (6..11).
    let mut piece_masks = vec![0u64; 12];
    let board = &setup.board;
    let mut mask = visibility_mask;
    while mask != 0 {
        let sq_idx = mask.trailing_zeros();
        mask &= mask - 1;
        let sq = unsafe { Square::new_unchecked(sq_idx) };
        if let Some(piece) = board.piece_at(sq) {
            let role_idx = match piece.role {
                Role::Pawn => 0, Role::Knight => 1, Role::Bishop => 2,
                Role::Rook => 3, Role::Queen => 4, Role::King => 5,
            };
            let base = if matches!(piece.color, Color::White) { 0 } else { 6 };
            piece_masks[base + role_idx] |= 1u64 << sq_idx;
        }
    }
    (visibility_mask, piece_masks, captured, opp_capture_landing_square,
     game_over_winner, game_over_reason)
}

/// Both perspectives' infoset-key components in one call (see
/// obs_key_core_for_color). Returns (white, black); each tuple maps directly to
/// `walker._obs_key`'s output: (visibility_int, [12 piece masks], own_cap,
/// opp_landing, winner|None, reason|"").
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn obs_keys_both_bb(
    prev_occupied_white: u64,
    prev_occupied_black: u64,
    prev_kings: u64,
    next_pawns: u64,
    next_knights: u64,
    next_bishops: u64,
    next_rooks: u64,
    next_queens: u64,
    next_kings: u64,
    next_occupied_white: u64,
    next_occupied_black: u64,
    next_castling_rights: u64,
    next_ep_square_idx: u32,
) -> (
    (u64, Vec<u64>, Option<u8>, Option<u8>, Option<bool>, &'static str),
    (u64, Vec<u64>, Option<u8>, Option<u8>, Option<bool>, &'static str),
) {
    let setup = setup_from_bb(
        next_pawns, next_knights, next_bishops, next_rooks, next_queens, next_kings,
        next_occupied_white, next_occupied_black, next_castling_rights, next_ep_square_idx,
    );
    let white = obs_key_core_for_color(
        &setup, true,
        prev_occupied_white, prev_occupied_black, prev_kings,
        next_occupied_white, next_occupied_black, next_kings,
    );
    let black = obs_key_core_for_color(
        &setup, false,
        prev_occupied_white, prev_occupied_black, prev_kings,
        next_occupied_white, next_occupied_black, next_kings,
    );
    (white, black)
}

/// Both perspectives in one call: build the `next` setup once, compute the
/// observation tuple for white and for black. Saves one setup_from_bb build,
/// one FFI crossing, and one Python-side board-attribute extraction per child
/// expansion (expand_leaf needs both perspectives' observations). Each returned
/// tuple is byte-identical to `observation_from_transition_bb` for that color.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn observation_from_transition_both_bb(
    prev_occupied_white: u64,
    prev_occupied_black: u64,
    prev_kings: u64,
    next_pawns: u64,
    next_knights: u64,
    next_bishops: u64,
    next_rooks: u64,
    next_queens: u64,
    next_kings: u64,
    next_occupied_white: u64,
    next_occupied_black: u64,
    next_castling_rights: u64,
    next_ep_square_idx: u32,
) -> (
    (u64, Vec<(u8, u8, bool)>, Option<u8>, Option<u8>, Option<bool>, &'static str),
    (u64, Vec<(u8, u8, bool)>, Option<u8>, Option<u8>, Option<bool>, &'static str),
) {
    let setup = setup_from_bb(
        next_pawns, next_knights, next_bishops, next_rooks, next_queens, next_kings,
        next_occupied_white, next_occupied_black, next_castling_rights, next_ep_square_idx,
    );
    let white = obs_core_for_color(
        &setup, true,
        prev_occupied_white, prev_occupied_black, prev_kings,
        next_occupied_white, next_occupied_black, next_kings,
    );
    let black = obs_core_for_color(
        &setup, false,
        prev_occupied_white, prev_occupied_black, prev_kings,
        next_occupied_white, next_occupied_black, next_kings,
    );
    (white, black)
}

// ---------------------------------------------------------------------------
// MT19937 — byte-identical to CPython's `random.Random`.
//
// The Rust equilibrium pass owns the equilibrium RNG stream so it never crosses
// the FFI boundary per sample. To stay byte-equal with the Python reference,
// this reproduces CPython's Mersenne Twister exactly: the same tempering, the
// same 53-bit `random()` transform (two u32 draws per float), and the same
// state layout so a Python `random.Random.getstate()` snapshot can be loaded
// directly. We carry the 624-word state + index rather than re-seeding, which
// sidesteps reproducing CPython's `init_by_array` seeding entirely.
// ---------------------------------------------------------------------------

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908b0df;
const MT_UPPER: u32 = 0x80000000;
const MT_LOWER: u32 = 0x7fffffff;

pub struct Mt19937 {
    mt: [u32; MT_N],
    index: usize,
}

impl Mt19937 {
    /// Build from a CPython `getstate()` snapshot: `words` = the 624 state
    /// ints, `index` = the trailing `mti` (the 625th element of getstate's
    /// tuple).
    fn from_state(words: &[u32], index: usize) -> Self {
        let mut mt = [0u32; MT_N];
        mt.copy_from_slice(&words[..MT_N]);
        Mt19937 { mt, index }
    }

    /// CPython's genrand_int32: regenerate (twist) when the block is consumed,
    /// then temper.
    fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            // Twist: regenerate the whole 624-word block.
            for kk in 0..MT_N {
                let y = (self.mt[kk] & MT_UPPER) | (self.mt[(kk + 1) % MT_N] & MT_LOWER);
                let mut next = self.mt[(kk + MT_M) % MT_N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= MT_MATRIX_A;
                }
                self.mt[kk] = next;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// CPython's genrand_res53: 53-bit float in [0, 1). Consumes two u32 draws.
    fn next_f64(&mut self) -> f64 {
        let a = self.next_u32() >> 5; // 27 bits
        let b = self.next_u32() >> 6; // 26 bits
        (a as f64 * 67108864.0 + b as f64) * (1.0 / 9007199254740992.0)
    }
}

/// Parity de-risk: load a CPython `getstate()` snapshot and emit `n` `random()`
/// draws. Must match `random.Random` byte-for-byte. Temporary — drops once the
/// EqEngine subsumes it.
#[pyfunction]
fn mt_res53_check(words: Vec<u32>, index: usize, n: usize) -> Vec<f64> {
    let mut rng = Mt19937::from_state(&words, index);
    (0..n).map(|_| rng.next_f64()).collect()
}

// ---------------------------------------------------------------------------
// EqEngine — native PCFR+ equilibrium pass over a flattened tree mirror.
//
// Mirror of `_equilibrium_traverse` (gt_cfr.py). Python keeps the expansion
// pass (Stockfish MultiPV) and pushes the growing tree topology + leaf values
// into this engine; the engine owns the per-(infoset, action) CFR state and
// runs the recursive PCFR+ walk natively, eliminating the Python per-visit cost
// the 2026-05-26 profile pinned (chess.Move hashing, _mk, recursion overhead,
// the 6 state-dict ops).
//
// V1 is byte-equality-first: it reproduces Python's six separate per-infoset
// dicts and its left-to-right f64 folds exactly, so results match the
// RNG-split Python reference bit-for-bit. The speed collapse (one struct per
// action instead of six HashMaps) comes only AFTER parity is proven.
// ---------------------------------------------------------------------------

/// One node of the flattened tree. Children are stored in registration order,
/// which Python guarantees equals `list(node.children.keys())` order (the
/// order expand_leaf iterates pseudo_legal_moves) — the order the strategy and
/// regret folds depend on for byte-equality.
/// Which game an EqEngine is solving. Default Chess (set in `new`); the chess
/// path stays byte-identical to before this seam. Mini opts in via `set_mini`.
/// Only the board-specific methods (add_root_from_fen / node_fen / expand_node)
/// branch on this — the CFR core (select/seed/equilibrium) is game-agnostic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GameKind {
    Chess,
    Mini,
    Xiangqi,
}

/// A tree node's packed board: chess = the 56-byte shakmaty PackedPos, mini =
/// the 49-square MiniSetup. Kept separate from PEnumState's PackedPos (belief)
/// so the chess belief hot path is untouched.
#[derive(Clone)]
enum NodePos {
    Chess(PackedPos),
    #[cfg(feature = "mini")]
    Mini(MiniSetup),
    #[cfg(feature = "xiangqi")]
    Xiangqi(XiangqiSetup),
}

fn node_pos_to_fen(p: &NodePos) -> String {
    match p {
        NodePos::Chess(pp) => Fen(unpack(pp)).to_string(),
        #[cfg(feature = "mini")]
        NodePos::Mini(ms) => mini_board_fen(ms),
        #[cfg(feature = "xiangqi")]
        NodePos::Xiangqi(xs) => xq_board_fen(xs),
    }
}

struct EqNode {
    to_move_white: bool,
    is_terminal: bool,
    /// Leaf value in the PERSPECTIVE player's POV (as Python stores it).
    leaf_value: f64,
    /// terminal_value(WHITE); terminal_value(traverser) negates this for black.
    terminal_val_white: f64,
    infoset: u32,
    child_keys: Vec<u32>,
    child_nodes: Vec<u32>,
    /// WS2 (slice 1): the node's truth board, packed. None for nodes built by the
    /// Python mirror (which doesn't pass a board yet); Some for nodes the Rust
    /// tree builds itself (roots from PEnumState, and — later — expanded
    /// children). The prerequisite for moving expansion/selection into Rust.
    pos: Option<NodePos>,
    /// WS2 (slice 2): running hash of each side's observation history along the
    /// path to this node. info_set_id = (to_move, obs_history_of_side_to_move),
    /// so the node's infoset is interned from (to_move_white, hist_of_to_move).
    /// O(1)/node rolling hashes instead of storing the full nested tuple. 0 = the
    /// empty history (roots) — matches Python's info_set_id == (to_move, ()).
    hist_white: u64,
    hist_black: u64,
}

/// WS2: stable (within-process) hash of an observation-key — the components
/// `obs_key_core_for_color` returns — so an infoset can be rolled up without
/// storing the full nested tuple. Only the PARTITION matters (equal obs-keys
/// hash equal); the value need not match Python's, only group nodes identically.
fn hash_obs_key(k: &(u64, Vec<u64>, Option<u8>, Option<u8>, Option<bool>, &str)) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = FxHasher::default();
    k.0.hash(&mut h);
    k.1.hash(&mut h);
    k.2.hash(&mut h);
    k.3.hash(&mut h);
    k.4.hash(&mut h);
    k.5.hash(&mut h);
    h.finish()
}

/// Global move-identity key — mirrors gt_cfr._mk: from | to<<6 | promo<<12
/// (drop is always 0 for standard chess). A bijection over (from,to,promo), so
/// the SAME move from different roots shares one key in a shared infoset's
/// regret table (cross-truth sharing) and distinct moves never conflate.
fn mk_move_key(from: u8, to: u8, promo: u8) -> u32 {
    (from as u32) | ((to as u32) << 6) | ((promo as u32) << 12)
}

/// Full Xiangqi move key. The board has 90 squares, so from/to need 7 bits.
fn mk_xiangqi_move_key(from: u8, to: u8) -> u32 {
    (from as u32) | ((to as u32) << 7)
}

/// Roll one observation-key hash into a running observation-history hash
/// (order-dependent: equal histories produce equal hashes).
fn roll_hist(prev: u64, key: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = FxHasher::default();
    prev.hash(&mut h);
    key.hash(&mut h);
    h.finish()
}

/// Per-action stats: regret, last_regret, last_strategy, visit_count, value_sum,
/// value_sq_sum. Was 6 separate HashMaps per EqInfoset; consolidated so the hot
/// path in eq_traverse does ONE map lookup instead of 6+ per action. Memory
/// layout stays packed since this is a Copy struct.
#[derive(Default, Clone, Copy)]
struct ActionStats {
    regret: f64,
    last_regret: f64,
    last_strategy: f64,
    visit_count: u64,
    value_sum: f64,
    value_sq_sum: f64,
}

/// Per-infoset CFR state — one map of per-action stats, mirroring the six
/// GTCFRState dicts exactly but consolidated for cache locality. FxHashMap
/// is significantly faster than std HashMap on u32 keys (no SipHash
/// linear-feedback shift; ~10ns/lookup vs ~30ns).
#[derive(Default)]
struct EqInfoset {
    visits: u64,
    actions: FxHashMap<u32, ActionStats>,
}

#[pyclass]
pub struct EqEngine {
    nodes: Vec<EqNode>,
    /// Which game this engine solves (Chess default; Mini via set_mini). Set
    /// once after construction and preserved across reset_tree.
    game: GameKind,
    infosets: Vec<EqInfoset>,
    rng: Mt19937,
    /// WS2: (to_move_white, obs-history hash) -> dense infoset id. The
    /// Rust-authoritative tree's analogue of _RustEqMirror's Python intern map.
    infoset_intern: FxHashMap<(bool, u64), u32>,
    /// WS2: the leaf-selection RNG stream — gt_cfr's `rng` (separate from the eq
    /// pass's `eq_rng`). Set via seed_select_rng before driving the grow loop.
    sel_rng: Mt19937,
    /// KLUSS keep-set: node ids within I^(k+1) of the source infoset(s). Set per
    /// iter via set_kluss_keep_from before select_leaf. None = no filter (default
    /// behavior). Mirrors gt_cfr.py:1061's per-iter keep_ids dance.
    kluss_keep: Option<FxHashSet<u32>>,
    /// Lever 3 (efficiency campaign): cached KLUSS BFS state so
    /// set_kluss_keep_from can incrementally update when the source infoset is
    /// unchanged and the tree has only grown. The full-BFS path stays as the
    /// fallback (source changed, first call). Byte-equivalent to full BFS —
    /// test_ws2_kluss_equiv keeps it honest.
    kluss_dist: Option<FxHashMap<u32, u32>>,
    kluss_by_white: FxHashMap<u64, Vec<u32>>,
    kluss_by_black: FxHashMap<u64, Vec<u32>>,
    /// Parent map (child_id → parent_id) populated incrementally in expand_node.
    /// Used by the connectivity-graph BFS's backward-tree-edge hop.
    kluss_parent_of: FxHashMap<u32, u32>,
    /// Watermark: tree size at the last full or incremental update. Nodes with
    /// id ≥ this count are "new" and need to be folded into kluss_dist.
    kluss_last_node_count: usize,
    /// Source infoset cached from the last call. When the next call's source
    /// differs, we full-rebuild rather than incrementally update.
    kluss_source_infoset: Option<u32>,
    /// Cached `k+1` cutoff. When the next call's `k` differs, we rebuild the
    /// keep-set from kluss_dist without recomputing distances.
    kluss_cutoff: u32,
}

/// PCFR+ current strategy: x = [z + last_regret]^+ / ||·||_1, with uniform
/// fallback. Mirrors _current_strategy's get-with-0.0 + left fold exactly.
fn eq_current_strategy(iset: &EqInfoset, keys: &[u32]) -> Vec<f64> {
    let mut positive: Vec<f64> = Vec::with_capacity(keys.len());
    for &k in keys {
        let stats = iset.actions.get(&k);
        let z = stats.map(|s| s.regret).unwrap_or(0.0);
        let prev = stats.map(|s| s.last_regret).unwrap_or(0.0);
        positive.push((z + prev).max(0.0));
    }
    let mut total = 0.0f64;
    for &p in &positive {
        total += p;
    }
    if total > 0.0 {
        for p in positive.iter_mut() {
            *p /= total;
        }
        positive
    } else {
        let n = keys.len();
        vec![1.0 / n as f64; n]
    }
}

/// External-sampling index draw — mirrors gt_cfr._sample.
fn eq_sample(probs: &[f64], rng: &mut Mt19937) -> usize {
    let r = rng.next_f64();
    let mut cum = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

// ---- WS2 slice 2: PUCT scoring + CPython-RNG primitives for leaf selection ----
// Mirror gt_cfr's puct_score / _q_value / _empirical_variance and random.Random
// exactly, so the Rust selection walk is byte-identical to _select_leaf_for_expansion.

const PRIOR_VARIANCE: f64 = 1.0;
const PUCT_C: f64 = 1.0;

/// Q̄(I,a): mean action value (0 if unvisited). Mirrors _q_value.
fn eq_q_value(iset: &EqInfoset, key: u32) -> f64 {
    match iset.actions.get(&key) {
        Some(s) if s.visit_count > 0 => s.value_sum / s.visit_count as f64,
        _ => 0.0,
    }
}

/// σ̂²(I,a): empirical variance with two ±1 prior samples. Mirrors
/// _empirical_variance (n_total = real + 2; sum_x2 += PRIOR_VARIANCE*2).
fn eq_empirical_variance(iset: &EqInfoset, key: u32) -> f64 {
    let (n_real, sum_x, sum_x2_raw) = match iset.actions.get(&key) {
        Some(s) => (s.visit_count, s.value_sum, s.value_sq_sum),
        None => (0, 0.0, 0.0),
    };
    let n_total = n_real + 2;
    let sum_x2 = sum_x2_raw + PRIOR_VARIANCE * 2.0;
    if n_total <= 1 {
        return PRIOR_VARIANCE;
    }
    let mean = sum_x / n_total as f64;
    (sum_x2 / n_total as f64 - mean * mean).max(0.0)
}

/// PUCT: Q̄ + C·σ̂·√N(I)/(1+N(I,a)). Mirrors puct_score.
fn eq_puct(iset: &EqInfoset, key: u32, c: f64) -> f64 {
    let q = eq_q_value(iset, key);
    let sigma = eq_empirical_variance(iset, key).sqrt();
    let n_infoset = iset.visits.max(1);
    let n_action = iset.actions.get(&key).map(|s| s.visit_count).unwrap_or(0);
    let explore = c * sigma * (n_infoset as f64).sqrt() / (1.0 + n_action as f64);
    q + explore
}

/// CPython `random.Random._randbelow_with_getrandbits(n)`: k = n.bit_length();
/// reject getrandbits(k) >= n. For our small action counts k <= 32 so
/// getrandbits(k) = genrand_uint32() >> (32 - k). Byte-identical to rng.choice's
/// index draw (rng.choice(seq) = seq[_randbelow(len(seq))]).
fn mt_randbelow(rng: &mut Mt19937, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let k = 64 - (n as u64).leading_zeros(); // n.bit_length()
    loop {
        let r = (rng.next_u32() >> (32 - k)) as usize;
        if r < n {
            return r;
        }
    }
}

/// Recursive PCFR+ traversal. Free function (not a method) so the borrow
/// checker can see `nodes` (shared) and `infosets`/`rng` (exclusive) are
/// disjoint across the recursion.
///
/// When `full_cfv_backprop` is false (default): the opponent branch uses
/// external sampling — pick ONE child per visit. This matches the prior
/// (pre-PCFR+ fix) behavior and keeps WS2 byte-parity with the Python
/// reference. When true: the opponent branch sums over ALL children
/// weighted by strategy, the Obscuro/PCFR+ "full counterfactual-value
/// backprop" path. Per-iter cost goes up but per-iter regret quality
/// goes up too — converges to the same equilibrium with less variance.
fn eq_traverse(
    nodes: &[EqNode],
    infosets: &mut [EqInfoset],
    rng: &mut Mt19937,
    node_id: u32,
    traverser_white: bool,
    perspective_white: bool,
    full_cfv_backprop: bool,
    // Gadget world-weight (Resolve follow/exit, Step 1): a constant multiplier on
    // this world's traverser regret contribution. 1.0 = unweighted (byte-identical
    // to the pre-gadget pass; `r * 1.0 == r` in IEEE). The Resolve gadget supplies
    // `alpha(J)·P(follow|J)` per root so worlds the opponent would EXIT exert ~zero
    // gradient on our shared root strategy — the anti-over-caution mechanism. Held
    // CONSTANT down the subtree (matches this CFR scheme's no-per-step-reach
    // convention): it only changes the relative weighting of worlds that SHARE an
    // infoset (the root infoset is shared across all sampled roots), and is a no-op
    // at world-specific infosets (scaling all actions equally there changes nothing).
    reach_weight: f64,
) -> f64 {
    let nid = node_id as usize;
    if nodes[nid].is_terminal {
        let tw = nodes[nid].terminal_val_white;
        return if traverser_white { tw } else { -tw };
    }
    if nodes[nid].child_nodes.is_empty() {
        // Leaf (not yet expanded). leaf_value is in perspective POV.
        let v = nodes[nid].leaf_value;
        return if traverser_white == perspective_white { v } else { -v };
    }
    let iset = nodes[nid].infoset as usize;
    let to_move_white = nodes[nid].to_move_white;
    // strategy from the CURRENT regrets (before this visit's update).
    let keys = &nodes[nid].child_keys;
    let strategy = eq_current_strategy(&infosets[iset], keys);
    infosets[iset].visits += 1;

    if to_move_white == traverser_white {
        let n = nodes[nid].child_nodes.len();
        let mut action_values = vec![0.0f64; n];
        for i in 0..n {
            let child = nodes[nid].child_nodes[i];
            action_values[i] = eq_traverse(
                nodes, infosets, rng, child, traverser_white, perspective_white,
                full_cfv_backprop, reach_weight,
            );
        }
        // node_value = Σ strategy[i] * action_values[i]  (left fold from 0.0)
        let mut node_value = 0.0f64;
        for i in 0..n {
            node_value += strategy[i] * action_values[i];
        }
        // PCFR+ regret update + PUCT bookkeeping (per action, Python order).
        // Gadget: the instantaneous regret is scaled by `reach_weight` (the world's
        // follow weight) so down-weighted worlds barely move the shared strategy.
        // PUCT bookkeeping (visit_count/value_sum/value_sq_sum) stays UNWEIGHTED so
        // leaf selection / expansion still cover every world and the read-out values
        // are undistorted — only the strategy (regret) carries the gadget weight.
        let im = &mut infosets[iset];
        for i in 0..n {
            let k = nodes[nid].child_keys[i];
            let av = action_values[i];
            let r = (av - node_value) * reach_weight;
            let stats = im.actions.entry(k).or_default();
            stats.regret = (stats.regret + r).max(0.0);
            stats.last_regret = r;
            stats.last_strategy = strategy[i];
            stats.visit_count += 1;
            stats.value_sum += av;
            stats.value_sq_sum += av * av;
        }
        node_value
    } else if full_cfv_backprop {
        // Opponent node, FULL CFV backprop: visit ALL children, weight by
        // strategy. Symmetric with the traverser branch but does NOT update
        // regret (it's the opponent's infoset; regret accumulates only for
        // the side currently traversing as the traverser). last_strategy is
        // still recorded for purification + analysis at the root.
        let n = nodes[nid].child_nodes.len();
        let mut node_value = 0.0f64;
        for i in 0..n {
            let child = nodes[nid].child_nodes[i];
            let child_value = eq_traverse(
                nodes, infosets, rng, child, traverser_white, perspective_white,
                full_cfv_backprop, reach_weight,
            );
            node_value += strategy[i] * child_value;
        }
        {
            let im = &mut infosets[iset];
            for i in 0..n {
                let k = nodes[nid].child_keys[i];
                im.actions.entry(k).or_default().last_strategy = strategy[i];
            }
        }
        node_value
    } else {
        // Opponent node, external sampling: pick ONE child by strategy.
        let chosen = eq_sample(&strategy, rng);
        {
            let im = &mut infosets[iset];
            for i in 0..nodes[nid].child_keys.len() {
                let k = nodes[nid].child_keys[i];
                im.actions.entry(k).or_default().last_strategy = strategy[i];
            }
        }
        let child = nodes[nid].child_nodes[chosen];
        eq_traverse(
            nodes, infosets, rng, child, traverser_white, perspective_white,
            full_cfv_backprop, reach_weight,
        )
    }
}

/// MERGED single-walk PCFR+ traversal (full-CFV only): updates BOTH players'
/// regrets in ONE post-order walk, instead of the two separate `eq_traverse`
/// walks (one per traverser). Returns the node value in WHITE POV.
///
/// At every internal node it recurses ALL children (full-width — no sampling, no
/// RNG) and updates the regret of the node's to-move player. White's action
/// values are the children's white-POV values directly; black's are negated
/// (zero-sum), so the regret delta is `sign * (child_white - node_value_white)`
/// with `sign = +1` for white-to-move, `-1` for black.
///
/// NOT byte-identical to the two-pass `equilibrium_pass_with(.., true)`: the two
/// passes use different-vintage opponent strategies (pass A sees the opponent's
/// pre-iter regret; pass B sees A's just-updated regret), a barrier a single
/// interleaved walk can't reproduce. This is a DIFFERENT (still valid) CFR update
/// scheme — same equilibrium in the limit, different per-iterate values — so it's
/// flag-gated (default off) and validated by STRENGTH (gadget defence + bakeoff),
/// not byte-parity. ~2x fewer node visits than the two-pass full-CFV path.
fn eq_traverse_merged(
    nodes: &[EqNode],
    infosets: &mut [EqInfoset],
    node_id: u32,
    perspective_white: bool,
    // Gadget world-weight (Resolve follow/exit): scales ONLY the PERSPECTIVE
    // player's regret update — in the two-pass weighted scheme the perspective
    // traverser carries the weight and the opponent pass runs at 1.0, so the
    // merged walk mirrors that split per to-move player. 1.0 = unweighted
    // (byte-identical to the pre-weight merged pass; `r * 1.0 == r` in IEEE).
    reach_weight: f64,
) -> f64 {
    let nid = node_id as usize;
    if nodes[nid].is_terminal {
        return nodes[nid].terminal_val_white; // already white POV
    }
    if nodes[nid].child_nodes.is_empty() {
        // leaf_value is stored in perspective POV; convert to white POV.
        let v = nodes[nid].leaf_value;
        return if perspective_white { v } else { -v };
    }
    let iset = nodes[nid].infoset as usize;
    let to_move_white = nodes[nid].to_move_white;
    let keys = &nodes[nid].child_keys;
    let strategy = eq_current_strategy(&infosets[iset], keys);
    infosets[iset].visits += 1;
    let n = nodes[nid].child_nodes.len();
    let mut child_white = vec![0.0f64; n];
    for i in 0..n {
        let child = nodes[nid].child_nodes[i];
        child_white[i] =
            eq_traverse_merged(nodes, infosets, child, perspective_white, reach_weight);
    }
    let mut node_value_white = 0.0f64;
    for i in 0..n {
        node_value_white += strategy[i] * child_white[i];
    }
    // Regret update for the to-move player, in that player's POV. The gadget
    // weight applies only when the PERSPECTIVE player is to move (see param
    // comment); bookkeeping (visits/value sums) stays unweighted, mirroring
    // eq_traverse.
    let sign = if to_move_white { 1.0 } else { -1.0 };
    let w = if to_move_white == perspective_white { reach_weight } else { 1.0 };
    let im = &mut infosets[iset];
    for i in 0..n {
        let k = nodes[nid].child_keys[i];
        let av = sign * child_white[i]; // action value in to-move POV
        let r = sign * (child_white[i] - node_value_white) * w; // av - node_value(to-move POV)
        let stats = im.actions.entry(k).or_default();
        stats.regret = (stats.regret + r).max(0.0);
        stats.last_regret = r;
        stats.last_strategy = strategy[i];
        stats.visit_count += 1;
        stats.value_sum += av;
        stats.value_sq_sum += av * av;
    }
    node_value_white
}

/// Read-only value of a node under the CURRENT strategy profile of both
/// players — no regret/visit/value mutation, so it never disturbs the solved
/// tree. Mirrors `eq_traverse`'s value computation with full-CFV semantics
/// (sum over all children weighted by the current regret-matched strategy) and
/// returns the value in the PERSPECTIVE player's POV. Used by the Resolve
/// gadget to read per-world action values after the solve (see
/// docs/engine/gadget-mvp-build-notes-2026-05-28.md, Slice 1).
fn eq_eval(nodes: &[EqNode], infosets: &[EqInfoset], node_id: u32, perspective_white: bool) -> f64 {
    let node = &nodes[node_id as usize];
    if node.is_terminal {
        let tw = node.terminal_val_white;
        return if perspective_white { tw } else { -tw };
    }
    if node.child_nodes.is_empty() {
        // Leaf value is already stored in the perspective player's POV.
        return node.leaf_value;
    }
    let strategy = eq_current_strategy(&infosets[node.infoset as usize], &node.child_keys);
    let mut value = 0.0f64;
    for (i, &child) in node.child_nodes.iter().enumerate() {
        value += strategy[i] * eq_eval(nodes, infosets, child, perspective_white);
    }
    value
}

#[pymethods]
impl EqEngine {
    /// Construct from a CPython `random.Random.getstate()` snapshot: `words` =
    /// the 624 state ints, `index` = the trailing mti.
    #[new]
    fn new(words: Vec<u32>, index: usize) -> Self {
        EqEngine {
            nodes: Vec::new(),
            game: GameKind::Chess,
            infosets: Vec::new(),
            rng: Mt19937::from_state(&words, index),
            infoset_intern: FxHashMap::default(),
            // Placeholder until seed_select_rng; same stream as the eq rng.
            sel_rng: Mt19937::from_state(&words, index),
            kluss_keep: None,
            kluss_dist: None,
            kluss_by_white: FxHashMap::default(),
            kluss_by_black: FxHashMap::default(),
            kluss_parent_of: FxHashMap::default(),
            kluss_last_node_count: 0,
            kluss_source_infoset: None,
            kluss_cutoff: 0,
        }
    }

    /// Select the game this engine solves. Chess (default) keeps byte-identical
    /// behavior; Mini routes the board-specific methods (add_root_from_fen /
    /// node_fen / expand_node) to the mini port. Persists across reset_tree
    /// (which only clears nodes/rng/kluss state).
    fn set_mini(&mut self, on: bool) {
        self.game = if on { GameKind::Mini } else { GameKind::Chess };
    }

    /// Select full 9x10 Xiangqi board-specific tree methods. The CFR core is
    /// unchanged; only add_root_from_fen / expand_node / node_fen branch.
    fn set_xiangqi(&mut self, on: bool) {
        self.game = if on { GameKind::Xiangqi } else { GameKind::Chess };
    }

    /// Transplant gt_cfr's selection RNG (`rng`) so the Rust leaf-selection walk
    /// consumes the identical stream — same getstate() snapshot the eq rng uses.
    fn seed_select_rng(&mut self, words: Vec<u32>, index: usize) {
        self.sel_rng = Mt19937::from_state(&words, index);
    }

    /// Lever 1 (efficiency campaign): full reset of the search-tree state in
    /// place, so the engine instance can be reused across choose_move calls
    /// without re-constructing. Clears nodes, infosets, the intern table, all
    /// KLUSS caches, and re-seeds the equilibrium RNG from `words` + `index`
    /// (caller seeds sel_rng separately via seed_select_rng).
    ///
    /// In Phase 1 this is called per pick_move and preserves byte-parity with
    /// the previous (per-call construction) behavior — same rng consumption
    /// pattern, same starting state. Phase 2 will replace some of these calls
    /// with `prune_to_subgame(own_move, opp_obs)` to actually carry tree state
    /// forward.
    fn reset_tree(&mut self, words: Vec<u32>, index: usize) {
        self.nodes.clear();
        self.infosets.clear();
        self.infoset_intern.clear();
        self.rng = Mt19937::from_state(&words, index);
        self.kluss_keep = None;
        self.kluss_dist = None;
        self.kluss_by_white.clear();
        self.kluss_by_black.clear();
        self.kluss_parent_of.clear();
        self.kluss_last_node_count = 0;
        self.kluss_source_infoset = None;
        self.kluss_cutoff = 0;
    }

    /// Lever 1 Phase 2 (variant b): clear nodes + KLUSS caches but PRESERVE
    /// infoset_intern and infosets across the call. Same RNG re-seed as
    /// reset_tree. Accumulated CFR state (regret, last_regret, visit counts,
    /// value sums) at every infoset survives — so when the next pick_move's
    /// search reaches the same infoset via a different tree (e.g. the next
    /// move's root infoset matches a depth-N infoset visited in the prior
    /// search by (to_move, hist) hash), it picks up warm regret data instead
    /// of starting from zero.
    ///
    /// INTENTIONALLY DIVERGES from reset_tree. Use only when the caller has
    /// opted into carryover semantics (engine_v2 carryover_infosets=True).
    /// Byte-parity vs reset_tree only on the FIRST call of a fresh engine; on
    /// every subsequent call, the search's regrets at re-visited infosets
    /// differ (warm-started instead of zero), which is the whole point.
    fn reset_tree_keep_infosets(&mut self, words: Vec<u32>, index: usize) {
        self.nodes.clear();
        // infosets + infoset_intern: KEPT (the whole point).
        self.rng = Mt19937::from_state(&words, index);
        self.kluss_keep = None;
        self.kluss_dist = None;
        self.kluss_by_white.clear();
        self.kluss_by_black.clear();
        self.kluss_parent_of.clear();
        self.kluss_last_node_count = 0;
        self.kluss_source_infoset = None;
        self.kluss_cutoff = 0;
    }

    /// Lever 1 Phase 2 (variant a): preserve nodes + infosets + intern
    /// completely. Only reset RNG + KLUSS caches (the KLUSS caches are
    /// tree-dependent and must be rebuilt against the new search's root
    /// frontier). Used when the caller will discover carryover candidates and
    /// pass surviving grandchildren as the next search's root_ids.
    ///
    /// The cleared `kluss_parent_of` is intentional: KLUSS BFS would otherwise
    /// walk backward from carryover roots into their old parent chain (which
    /// belongs to the prior move's tree and isn't relevant to the new
    /// search). KLUSS will re-populate parent edges as expand_node runs on
    /// new children.
    fn reset_for_carryover(&mut self, words: Vec<u32>, index: usize) {
        // Nodes + infosets + infoset_intern: KEPT for subtree reuse.
        self.rng = Mt19937::from_state(&words, index);
        self.kluss_keep = None;
        self.kluss_dist = None;
        self.kluss_by_white.clear();
        self.kluss_by_black.clear();
        self.kluss_parent_of.clear();
        self.kluss_last_node_count = 0;
        self.kluss_source_infoset = None;
        self.kluss_cutoff = 0;
    }

    /// Lever 1 Phase 2a: discover carryover candidates. For each prior root,
    /// find the child under `action_key` (= the move the perspective played),
    /// enumerate its grandchildren, return (fen, node_id) pairs. The caller
    /// builds a FEN→node_id dict and matches new-belief root truths against
    /// it to reuse subtrees.
    fn discover_carryover_candidates(
        &self,
        prev_root_ids: Vec<u32>,
        action_key: u32,
    ) -> Vec<(String, u32)> {
        let mut out: Vec<(String, u32)> = Vec::new();
        for &rid in &prev_root_ids {
            let root = match self.nodes.get(rid as usize) {
                Some(n) => n,
                None => continue,
            };
            let mut child_id: Option<u32> = None;
            for (i, &k) in root.child_keys.iter().enumerate() {
                if k == action_key {
                    child_id = Some(root.child_nodes[i]);
                    break;
                }
            }
            let cid = match child_id {
                Some(c) => c,
                None => continue, // root didn't explore this action
            };
            let child = match self.nodes.get(cid as usize) {
                Some(n) => n,
                None => continue,
            };
            for &gc_id in &child.child_nodes {
                if let Some(gc) = self.nodes.get(gc_id as usize) {
                    if let Some(pp) = &gc.pos {
                        out.push((node_pos_to_fen(pp), gc_id));
                    }
                }
            }
        }
        out
    }

    /// Phase 1 (structural carry): SELECT the new search root set from the
    /// carried tree, instead of re-sampling the belief. Walk the grandchildren
    /// of the played-move reply node (same path as
    /// `discover_carryover_candidates`), KEEP those still consistent with the
    /// perspective's post-opponent-move observation (`visible_consistent` —
    /// O(|GC|), independent of |P|), DEDUP by EPD (a carried subtree backs <=1
    /// root; EPD twins that differ only in move-counters collapse to one, which
    /// keeps the downstream duplicate-root assertion happy), and CAP at
    /// `budget`. Returns surviving carried node ids to reuse as roots; the
    /// Python caller tops up to the budget with fresh samples for the new
    /// worlds (Obscuro's Gamma_hat ∪ I). Chess-only (mini has no carryover).
    fn build_carryover_roots(
        &self,
        prev_root_ids: Vec<u32>,
        action_key: u32,
        perspective_white: bool,
        obs_vis: u64,
        obs_w: Vec<u64>,
        obs_b: Vec<u64>,
        budget: usize,
    ) -> Vec<u32> {
        if obs_w.len() < 6 || obs_b.len() < 6 || budget == 0 {
            return Vec::new();
        }
        let perspective = if perspective_white { Color::White } else { Color::Black };
        let ow: [u64; 6] = [obs_w[0], obs_w[1], obs_w[2], obs_w[3], obs_w[4], obs_w[5]];
        let ob: [u64; 6] = [obs_b[0], obs_b[1], obs_b[2], obs_b[3], obs_b[4], obs_b[5]];
        let mut seen: FxHashSet<String> = FxHashSet::default();
        let mut out: Vec<u32> = Vec::new();
        for &rid in &prev_root_ids {
            let root = match self.nodes.get(rid as usize) {
                Some(n) => n,
                None => continue,
            };
            let mut child_id: Option<u32> = None;
            for (i, &k) in root.child_keys.iter().enumerate() {
                if k == action_key {
                    child_id = Some(root.child_nodes[i]);
                    break;
                }
            }
            let cid = match child_id {
                Some(c) => c,
                None => continue, // prior root never explored the move we played
            };
            let child = match self.nodes.get(cid as usize) {
                Some(n) => n,
                None => continue,
            };
            for &gc_id in &child.child_nodes {
                if out.len() >= budget {
                    return out;
                }
                let gc = match self.nodes.get(gc_id as usize) {
                    Some(n) => n,
                    None => continue,
                };
                let pp = match &gc.pos {
                    Some(NodePos::Chess(pp)) => pp,
                    _ => continue, // chess-only carryover
                };
                let setup = unpack(pp);
                // Membership in the new belief = consistent with the perspective's
                // post-opp-move observation (visible squares + visible pieces match).
                if !visible_consistent(&setup, perspective, obs_vis, &ow, &ob) {
                    continue;
                }
                let full = Fen(setup).to_string();
                let epd: String =
                    full.split_whitespace().take(4).collect::<Vec<_>>().join(" ");
                if seen.insert(epd) {
                    out.push(gc_id);
                }
            }
        }
        out
    }

    /// Grow the infoset table so `id` is addressable.
    fn ensure_infoset(&mut self, id: u32) {
        let need = id as usize + 1;
        if self.infosets.len() < need {
            self.infosets.resize_with(need, EqInfoset::default);
        }
    }

    /// Intern (to_move_white, obs-history hash) -> dense infoset id, allocating +
    /// sizing the infoset table on first sight. The Rust-authoritative analogue
    /// of _RustEqMirror.intern_iset: groups tree nodes into infosets natively
    /// (info_set_id = (to_move, obs_history_of_side_to_move)). Distinct from the
    /// mirror path's Python-passed ids — an engine is used in one mode or the
    /// other, so the two id spaces never mix.
    fn intern_infoset(&mut self, to_move_white: bool, hist: u64) -> u32 {
        let next = self.infoset_intern.len() as u32;
        let id = *self.infoset_intern.entry((to_move_white, hist)).or_insert(next);
        self.ensure_infoset(id);
        id
    }

    /// Register one node; returns its id. `leaf_value` is in perspective POV;
    /// `terminal_val_white` is terminal_value(WHITE) (ignored unless terminal).
    fn add_node(
        &mut self,
        to_move_white: bool,
        is_terminal: bool,
        leaf_value: f64,
        terminal_val_white: f64,
        infoset: u32,
    ) -> u32 {
        self.ensure_infoset(infoset);
        let id = self.nodes.len() as u32;
        self.nodes.push(EqNode {
            to_move_white,
            is_terminal,
            leaf_value,
            terminal_val_white,
            infoset,
            child_keys: Vec::new(),
            child_nodes: Vec::new(),
            pos: None,
            // Mirror path (Python owns interning) — histories unused here.
            hist_white: 0,
            hist_black: 0,
        });
        id
    }

    // ---- WS2 slice 1: Rust tree holds the board + builds roots ----
    // Foundation for moving expansion/selection into Rust. Unused by the live
    // (Python-driven) path yet; exercised + verified by tests so later slices
    // build on a proven base. `add_root_from_fen` mirrors gt_cfr.root_node:
    // to_move = board's side, is_terminal = a king missing, empty children.

    fn add_root_from_fen(&mut self, fen: &str) -> PyResult<u32> {
        // Game-specific: parse the board, derive side-to-move + terminal, pack.
        // Chess to_move_white = white-to-move; mini to_move_white = red-to-move
        // (red is the first player, the "white" analog the CFR core keys on).
        let (to_move_white, is_terminal, pos) = match self.game {
            GameKind::Chess => {
                let setup = parse_fen_lenient(fen)?;
                let tmw = setup.turn == Color::White;
                let term = setup.board.king_of(Color::White).is_none()
                    || setup.board.king_of(Color::Black).is_none();
                (tmw, term, NodePos::Chess(pack(&setup)))
            }
            #[cfg(feature = "mini")]
            GameKind::Mini => {
                let setup = mini_parse_fen(fen)?;
                let tmw = setup.red_to_move;
                let term = mini_general_sq(&setup, true).is_none()
                    || mini_general_sq(&setup, false).is_none();
                (tmw, term, NodePos::Mini(setup))
            }
            #[cfg(feature = "xiangqi")]
            GameKind::Xiangqi => {
                let setup = xq_parse_fen(fen)?;
                let tmw = setup.red_to_move;
                let term = xq_general_sq(&setup, true).is_none()
                    || xq_general_sq(&setup, false).is_none();
                (tmw, term, NodePos::Xiangqi(setup))
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "fow_rust built without support for this game (variant \
                     cargo feature disabled)",
                ))
            }
        };
        // Empty obs history (hist 0) → infoset (to_move, ()) — all roots of the
        // same side-to-move share an infoset, matching gt_cfr's multiroot.
        let infoset = self.intern_infoset(to_move_white, 0);
        let id = self.nodes.len() as u32;
        self.nodes.push(EqNode {
            to_move_white,
            is_terminal,
            leaf_value: 0.0,
            terminal_val_white: 0.0,
            infoset,
            child_keys: Vec::new(),
            child_nodes: Vec::new(),
            pos: Some(pos),
            hist_white: 0,
            hist_black: 0,
        });
        Ok(id)
    }

    /// Read a node's board back as a FEN (None if the node has no board). For
    /// equivalence-checking the Rust tree's board against the Python truth.
    fn node_fen(&self, node_id: u32) -> Option<String> {
        self.nodes
            .get(node_id as usize)
            .and_then(|n| n.pos.as_ref())
            .map(node_pos_to_fen)
    }

    /// The node's interned infoset id (None if the node doesn't exist). For
    /// equivalence-checking the Rust tree's infoset partition against Python's
    /// info_set_id grouping.
    fn node_infoset(&self, node_id: u32) -> Option<u32> {
        self.nodes.get(node_id as usize).map(|n| n.infoset)
    }

    // ---- WS2 slice 2: leaf selection in Rust ----
    // Port of gt_cfr._select_leaf_for_expansion (non-KLUSS path): walk from a
    // root, exploring-player = PUCT-mixture (½ uniform-over-support, ½ argmax
    // PUCT), non-exploring = sample current strategy; stop at the first
    // non-terminal leaf. Consumes sel_rng identically to random.Random.

    /// Walk `root_id` to a leaf to expand; None if the reachable subtree is
    /// terminal/exhausted. exploring_white = the exploring player this iter.
    ///
    /// When `self.kluss_keep` is Some, the walk is restricted to nodes in the
    /// keep set. If the current root isn't in keep, return None immediately. At
    /// each step, child candidates are filtered to keep-only; if all children
    /// are outside keep, also return None. A leaf in keep is the only valid
    /// terminator. Mirrors gt_cfr._select_leaf_for_expansion's keep_ids dance.
    fn select_leaf(&mut self, root_id: u32, exploring_white: bool) -> Option<u32> {
        let mut node_id = root_id;
        loop {
            let (is_terminal, leaf, infoset, to_move_white, keys, children) = {
                let node = self.nodes.get(node_id as usize)?;
                (
                    node.is_terminal,
                    node.child_nodes.is_empty(),
                    node.infoset,
                    node.to_move_white,
                    node.child_keys.clone(),
                    node.child_nodes.clone(),
                )
            };
            if is_terminal {
                return None;
            }
            if leaf {
                // Python: a leaf outside keep returns None (gt_cfr.py:571-572).
                if let Some(ref keep) = self.kluss_keep {
                    if !keep.contains(&node_id) {
                        return None;
                    }
                }
                return Some(node_id);
            }
            // KLUSS: filter child candidates to those in keep. If none remain,
            // halt this iter for this root. Indexing through `kept_indices`
            // preserves PUCT/strategy semantics on the filtered support.
            let kept_indices: Vec<usize> = match self.kluss_keep {
                Some(ref keep) => (0..children.len())
                    .filter(|&i| keep.contains(&children[i]))
                    .collect(),
                None => (0..children.len()).collect(),
            };
            if kept_indices.is_empty() {
                return None;
            }
            let kept_keys: Vec<u32> = kept_indices.iter().map(|&i| keys[i]).collect();
            let strategy = eq_current_strategy(&self.infosets[infoset as usize], &kept_keys);
            let kept_choice: usize = if to_move_white == exploring_white {
                // exploring player: ½ uniform-over-support + ½ argmax-PUCT
                if self.sel_rng.next_f64() < 0.5 {
                    let mut support: Vec<usize> = (0..kept_keys.len())
                        .filter(|&i| strategy[i] > 0.0)
                        .collect();
                    if support.is_empty() {
                        support = (0..kept_keys.len()).collect();
                    }
                    support[mt_randbelow(&mut self.sel_rng, support.len())]
                } else {
                    let iset = &self.infosets[infoset as usize];
                    let mut best_i = 0usize;
                    let mut best = f64::NEG_INFINITY;
                    for (i, &k) in kept_keys.iter().enumerate() {
                        let s = eq_puct(iset, k, PUCT_C);
                        if s > best {
                            best = s;
                            best_i = i;
                        }
                    }
                    best_i
                }
            } else {
                eq_sample(&strategy, &mut self.sel_rng)
            };
            node_id = children[kept_indices[kept_choice]];
        }
    }

    /// Compute the k-KLUSS keep-set (nodes within I^(k+1)) and store it as
    /// `self.kluss_keep`. Source = the infoset of `root_ids[0]` (mirrors
    /// gt_cfr.py:1064 — all roots share an infoset, that's the source).
    /// Recompute per iter; the tree grows so the set drifts.
    ///
    /// Lever 3 fast path: when (a) the source infoset matches the last call,
    /// (b) `k` is unchanged, and (c) the tree only grew (no node deletions),
    /// the distance map is updated incrementally — fold the new nodes into
    /// their best distance via connectivity-graph neighbors, then BFS to
    /// propagate any shortcut distance reductions to old nodes. Otherwise
    /// fall back to a full BFS.
    ///
    /// Byte-equivalent to the full-BFS path; verified by test_ws2_kluss_equiv.
    fn set_kluss_keep_from(&mut self, root_ids: Vec<u32>, k: u32) -> PyResult<()> {
        if root_ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_kluss_keep_from: root_ids empty",
            ));
        }
        let source_infoset = match self.nodes.get(root_ids[0] as usize) {
            Some(n) => n.infoset,
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "set_kluss_keep_from: root_ids[0] not in tree",
                ))
            }
        };
        let cutoff = k + 1; // I^(k+1) boundary
        let cur_node_count = self.nodes.len();

        // Incremental needs an existing dist AND keep-set to fold into, the same
        // source + cutoff (else keep membership shifts wholesale), and a tree that
        // only grew. Otherwise full-rebuild (source/cutoff changed, or first call).
        let can_incremental = self.kluss_dist.is_some()
            && self.kluss_keep.is_some()
            && self.kluss_source_infoset == Some(source_infoset)
            && self.kluss_cutoff == cutoff
            && cur_node_count >= self.kluss_last_node_count;

        if can_incremental {
            // Lever 3 keep-set fix: fold ONLY the touched nodes into the existing
            // keep-set. Incremental dist updates only ever DECREASE distances, so
            // the keep-set only GROWS — a touched node enters iff its new dist <=
            // cutoff; nothing leaves. Byte-identical to the full rebuild given
            // keep was correct before (full rebuild establishes it; this preserves
            // it). O(touched) per call instead of O(nodes) — kills the per-iter
            // quadratic that made KLUSS ~40% of pick_move.
            let touched = self.kluss_incremental_update(source_infoset);
            let dist = self.kluss_dist.as_ref().expect("dist on incremental path");
            let keep = self.kluss_keep.as_mut().expect("keep on incremental path");
            for nid in touched {
                if dist.get(&nid).map_or(false, |&d| d <= cutoff) {
                    keep.insert(nid);
                }
            }
        } else {
            // Full rebuild: recompute dist, then build the keep-set from scratch.
            // Rare (source/cutoff change, first call) — not the per-iter hot path.
            self.kluss_full_rebuild(&root_ids, source_infoset);
            self.kluss_cutoff = cutoff;
            let dist = self.kluss_dist.as_ref().expect("dist populated above");
            let keep: FxHashSet<u32> = dist
                .iter()
                .filter_map(|(&nid, &d)| if d <= cutoff { Some(nid) } else { None })
                .collect();
            self.kluss_keep = Some(keep);
        }
        self.kluss_last_node_count = cur_node_count;
        self.kluss_source_infoset = Some(source_infoset);
        Ok(())
    }

    /// Drop the KLUSS filter; subsequent select_leaf calls explore the full tree.
    fn clear_kluss_keep(&mut self) {
        self.kluss_keep = None;
    }

    /// Test helper: snapshot the current keep-set as a sorted Vec for parity
    /// tests against the Python reference.
    fn kluss_keep_snapshot(&self) -> Option<Vec<u32>> {
        self.kluss_keep.as_ref().map(|s| {
            let mut v: Vec<u32> = s.iter().copied().collect();
            v.sort_unstable();
            v
        })
    }

    /// best_root index = min over roots by (subtree size, sel_rng draw), one draw
    /// per root in order — matching gt_cfr's `min(roots, key=(root_sizes,
    /// rng.random()))`. Returned as an index so the driver can bump that root's
    /// size; the caller then select_leaf's it (the next sel_rng draws), so the
    /// consumption order matches gt_cfr's single `rng` stream (min then select).
    fn pick_best_root(&mut self, root_sizes: Vec<u32>) -> usize {
        let mut best_i = 0usize;
        let mut best_key = (u32::MAX, f64::INFINITY);
        for i in 0..root_sizes.len() {
            let r = self.sel_rng.next_f64(); // drawn for EVERY root (Python computes key per root)
            if root_sizes[i] < best_key.0 || (root_sizes[i] == best_key.0 && r < best_key.1) {
                best_key = (root_sizes[i], r);
                best_i = i;
            }
        }
        best_i
    }

    /// (child action keys, child node ids) for a node — both empty for a leaf.
    /// For the equivalence-test reference walk.
    fn node_children(&self, node_id: u32) -> Option<(Vec<u32>, Vec<u32>)> {
        self.nodes
            .get(node_id as usize)
            .map(|n| (n.child_keys.clone(), n.child_nodes.clone()))
    }

    /// Resolve gadget (read-only): for each root, the per-child
    /// `(action_key, value)` under the current strategy, in the perspective
    /// player's POV. `value(action a, root j)` = the value of playing `a` at
    /// world `j`, the input the gadget caps against the blueprint baseline.
    /// Does not mutate the tree (uses `eq_eval`), so it's safe to call after
    /// the solve without affecting any subsequent pass.
    fn root_child_values(
        &self,
        root_ids: Vec<u32>,
        perspective_white: bool,
    ) -> Vec<Vec<(u32, f64)>> {
        root_ids
            .iter()
            .map(|&r| match self.nodes.get(r as usize) {
                None => Vec::new(),
                Some(node) => node
                    .child_keys
                    .iter()
                    .zip(node.child_nodes.iter())
                    .map(|(&k, &c)| {
                        (k, eq_eval(&self.nodes, &self.infosets, c, perspective_white))
                    })
                    .collect(),
            })
            .collect()
    }

    fn node_is_terminal(&self, node_id: u32) -> Option<bool> {
        self.nodes.get(node_id as usize).map(|n| n.is_terminal)
    }

    fn node_to_move_white(&self, node_id: u32) -> Option<bool> {
        self.nodes.get(node_id as usize).map(|n| n.to_move_white)
    }

    /// PUCT score for (infoset, action key) — for the reference walk's argmax.
    fn puct_get(&self, infoset: u32, key: u32, c: f64) -> f64 {
        match self.infosets.get(infoset as usize) {
            Some(iset) => eq_puct(iset, key, c),
            None => 0.0,
        }
    }

    // ---- WS2 slice 2/3: seed an expansion's leaf values + smart regret ----
    // The Rust side of the Stockfish FFI batch boundary (design option A): the
    // Rust tree expands a node (expand_node), Python evaluates the new leaves
    // with Stockfish and passes back per-child values (perspective POV; terminal
    // children keep expand_node's exact value, FoW-illegal children use the
    // material fallback), and this seeds them — the analogue of expand_leaf's
    // post-MultiPV step. `values` is in child (= python-chess move) order.

    /// Set each child's leaf_value from `values` and seed the parent infoset's
    /// regret to all-weight-on-the-best-child (smart-regret init), mirroring
    /// expand_leaf: best = argmax leaf_value if the parent's to_move == perspective
    /// else argmin (first on ties).
    fn seed_expansion(
        &mut self,
        parent_id: u32,
        values: Vec<f64>,
        perspective_white: bool,
    ) -> PyResult<()> {
        let (infoset, keys, children, to_move_white) = {
            let p = self.nodes.get(parent_id as usize).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("seed_expansion: no node {parent_id}"))
            })?;
            (p.infoset, p.child_keys.clone(), p.child_nodes.clone(), p.to_move_white)
        };
        if values.len() != children.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "seed_expansion: {} values for {} children",
                values.len(),
                children.len()
            )));
        }
        if children.is_empty() {
            return Ok(());
        }
        for (i, &cid) in children.iter().enumerate() {
            self.nodes[cid as usize].leaf_value = values[i];
        }
        // best child: maximize if parent is the perspective player to move, else
        // minimize. Strict comparison => first index on ties (matches Python max/min).
        let maximize = to_move_white == perspective_white;
        let mut best_i = 0usize;
        for i in 1..values.len() {
            let better = if maximize { values[i] > values[best_i] } else { values[i] < values[best_i] };
            if better {
                best_i = i;
            }
        }
        let iset = &mut self.infosets[infoset as usize];
        for (i, &k) in keys.iter().enumerate() {
            iset.actions.entry(k).or_default().regret = if i == best_i { 1.0 } else { 0.0 };
        }
        Ok(())
    }

    /// A node's current leaf value (perspective POV). For equivalence-checking
    /// the seeded values.
    fn node_leaf_value(&self, node_id: u32) -> Option<f64> {
        self.nodes.get(node_id as usize).map(|n| n.leaf_value)
    }

    // ---- WS2 slice 2 (prep): structural expansion in Rust ----
    // The Stockfish-FREE half of gt_cfr.expand_leaf, Rust-native. The riskiest
    // piece of the eventual coherent tree build (move order + FoW keys + terminal
    // handling must be byte-identical), de-risked here offline. NOT wired into the
    // live search — exercised + verified by tests so the coherent build stands on
    // a proven base (the slice-1 discipline).

    /// Expand a board-holding node into its children, reproducing expand_leaf's
    /// structural output: for each pseudo-legal move (python-chess order), apply
    /// it to get the child board, compute the (white, black) FoW observation-key
    /// components (reusing obs_key_core_for_color — byte-identical to
    /// walker.obs_keys_both), and detect king-capture terminals with their exact
    /// perspective-POV value (-1/0/+1). Creates + links child EqNodes (pos set,
    /// to_move flipped, is_terminal, terminal_val_white set; leaf_value carries
    /// the terminal value or 0.0 placeholder for non-terminal children — the
    /// Stockfish leaf value + smart-regret seed are applied later from Python at
    /// the batch boundary, design option A). Each child's infoset is interned
    /// natively from its rolled obs-history (info_set_id parity, slice 2).
    ///
    /// Returns one tuple per child in expansion order:
    ///   (from, to, promo, child_id, is_terminal, leaf_value_persp,
    ///    white_key_components, black_key_components)
    /// where each *_key_components mirrors walker._obs_key's inputs:
    ///   (visibility, [12 piece masks], own_cap, opp_landing, winner, reason).
    #[allow(clippy::type_complexity)]
    fn expand_node(
        &mut self,
        node_id: u32,
        perspective_white: bool,
    ) -> PyResult<
        Vec<(
            u8,
            u8,
            u8,
            u32,
            bool,
            f64,
            (u64, Vec<u64>, Option<u8>, Option<u8>, Option<bool>, &'static str),
            (u64, Vec<u64>, Option<u8>, Option<u8>, Option<bool>, &'static str),
        )>,
    > {
        #[cfg(feature = "mini")]
        if let GameKind::Mini = self.game {
            return self.expand_node_mini(node_id, perspective_white);
        }
        #[cfg(feature = "xiangqi")]
        if let GameKind::Xiangqi = self.game {
            return self.expand_node_xiangqi(node_id, perspective_white);
        }
        let parent = self.nodes.get(node_id as usize).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("expand_node: no node {node_id}"))
        })?;
        let parent_pos = match parent.pos.clone() {
            Some(NodePos::Chess(pp)) => pp,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "expand_node: chess node has no chess board (pos)",
                ))
            }
        };
        let parent_to_move_white = parent.to_move_white;
        let parent_hist_white = parent.hist_white;
        let parent_hist_black = parent.hist_black;
        let child_to_move_white = !parent_to_move_white;

        let prev_setup = unpack(&parent_pos);
        let prev_ow = prev_setup.board.by_color(Color::White).0;
        let prev_ob = prev_setup.board.by_color(Color::Black).0;
        let prev_kings = prev_setup.board.by_role(Role::King).0;

        let moves = gen_fow_pseudo_legal_moves_pychess_order(&prev_setup);
        let mut out = Vec::with_capacity(moves.len());
        let mut child_ids: Vec<u32> = Vec::with_capacity(moves.len());
        let mut child_keys: Vec<u32> = Vec::with_capacity(moves.len());

        for (f, t, p) in &moves {
            let from = unsafe { Square::new_unchecked(*f as u32) };
            let to = unsafe { Square::new_unchecked(*t as u32) };
            let promo_role = role_from_int(*p);
            let next_setup = apply_move_to_setup(&prev_setup, from, to, promo_role);

            let next_ow = next_setup.board.by_color(Color::White).0;
            let next_ob = next_setup.board.by_color(Color::Black).0;
            let next_kings = next_setup.board.by_role(Role::King).0;
            let wkey = obs_key_core_for_color(
                &next_setup, true, prev_ow, prev_ob, prev_kings, next_ow, next_ob, next_kings,
            );
            let bkey = obs_key_core_for_color(
                &next_setup, false, prev_ow, prev_ob, prev_kings, next_ow, next_ob, next_kings,
            );

            // Terminal (king-capture) detection — mirrors expand_leaf's
            // terminal-FIRST branch. leaf_value is from the PERSPECTIVE POV;
            // terminal_val_white is terminal_value(WHITE).
            let white_king_gone = next_setup.board.king_of(Color::White).is_none();
            let black_king_gone = next_setup.board.king_of(Color::Black).is_none();
            let is_terminal = white_king_gone || black_king_gone;
            let (leaf_value, terminal_val_white) = if is_terminal {
                let (own_gone, opp_gone) = if perspective_white {
                    (white_king_gone, black_king_gone)
                } else {
                    (black_king_gone, white_king_gone)
                };
                let lv = if own_gone && opp_gone {
                    0.0
                } else if own_gone {
                    -1.0
                } else {
                    1.0
                };
                let tvw = if perspective_white { lv } else { -lv };
                (lv, tvw)
            } else {
                (0.0, 0.0)
            };

            // Roll the child's obs-history hashes and intern its infoset
            // (info_set_id = (to_move, obs_history_of_side_to_move)).
            let child_hist_white = roll_hist(parent_hist_white, hash_obs_key(&wkey));
            let child_hist_black = roll_hist(parent_hist_black, hash_obs_key(&bkey));
            let child_hist_for_to_move = if child_to_move_white {
                child_hist_white
            } else {
                child_hist_black
            };
            let infoset = self.intern_infoset(child_to_move_white, child_hist_for_to_move);

            let child_id = self.nodes.len() as u32;
            self.nodes.push(EqNode {
                to_move_white: child_to_move_white,
                is_terminal,
                leaf_value,
                terminal_val_white,
                infoset,
                child_keys: Vec::new(),
                child_nodes: Vec::new(),
                pos: Some(NodePos::Chess(pack(&next_setup))),
                hist_white: child_hist_white,
                hist_black: child_hist_black,
            });
            child_ids.push(child_id);
            child_keys.push(mk_move_key(*f, *t, *p));
            out.push((*f, *t, *p, child_id, is_terminal, leaf_value, wkey, bkey));
            // Lever 3: maintain parent map for KLUSS incremental BFS so the
            // backward-tree-edge hop doesn't need a full tree walk each call.
            // Always populated regardless of whether KLUSS is currently active.
            self.kluss_parent_of.insert(child_id, node_id);
        }

        // Link children into the parent, keyed by global move identity (_mk) so a
        // shared infoset's regret table shares per-move across roots (cross-truth).
        let parent = &mut self.nodes[node_id as usize];
        parent.child_keys = child_keys;
        parent.child_nodes = child_ids;

        Ok(out)
    }

    /// Attach already-registered children to `parent` in registration order.
    fn link_children(&mut self, parent: u32, keys: Vec<u32>, child_ids: Vec<u32>) {
        let p = &mut self.nodes[parent as usize];
        p.child_keys = keys;
        p.child_nodes = child_ids;
    }

    /// Smart-regret seed at expansion (expand_leaf sets regret=1.0 on the best
    /// Stockfish child, 0.0 otherwise).
    fn seed_regret(&mut self, infoset: u32, key: u32, value: f64) {
        self.ensure_infoset(infoset);
        self.infosets[infoset as usize]
            .actions
            .entry(key)
            .or_default()
            .regret = value;
    }

    /// One full equilibrium pass: for each traverser in (perspective, !persp),
    /// walk every root. Mirrors the multiroot coordinator's inner double loop.
    fn equilibrium_pass(&mut self, root_ids: Vec<u32>, perspective_white: bool) {
        self.equilibrium_pass_with(root_ids, perspective_white, /* full_cfv_backprop */ false);
    }

    /// PCFR+ variant: when `full_cfv_backprop` is true, the opponent branch
    /// in `eq_traverse` sums over all children (Obscuro/PCFR+) instead of
    /// external-sampling one. Default `equilibrium_pass` keeps external
    /// sampling for byte-parity with the Python reference.
    fn equilibrium_pass_with(
        &mut self,
        root_ids: Vec<u32>,
        perspective_white: bool,
        full_cfv_backprop: bool,
    ) {
        for &traverser in &[perspective_white, !perspective_white] {
            for &r in &root_ids {
                eq_traverse(
                    &self.nodes,
                    &mut self.infosets,
                    &mut self.rng,
                    r,
                    traverser,
                    perspective_white,
                    full_cfv_backprop,
                    /* reach_weight */ 1.0,
                );
            }
        }
    }

    /// Resolve-gadget per-world-weighted equilibrium pass (proper-gadget Step 1).
    ///
    /// Like `equilibrium_pass_with(.., full_cfv_backprop=true)` but the PERSPECTIVE
    /// traverser pass scales each root's regret contribution by `weights[i]` (the
    /// Resolve gadget's `alpha(J)·P(follow|J)` for that world). Worlds the opponent
    /// would EXIT get ~0 weight → they exert ~no gradient on our shared root
    /// strategy, so the engine stops defending worlds the opponent will not enter —
    /// the anti-over-caution mechanism the read-only post-hoc cap lacks.
    ///
    /// The OPPONENT traverser pass is UNWEIGHTED (weight 1.0): the opponent's
    /// within-world best response is independent of how likely it is to enter — the
    /// follow/exit decision is the gadget's, solved separately by the Python
    /// `ResolveGadget` RM+ and fed back here as `weights`. (Uniform alpha over
    /// uniformly-sampled worlds would be a constant scale = no relative change
    /// anyway.)
    ///
    /// `weights` must align with `root_ids` (same length, same order). The pass is
    /// full-CFV (Obscuro/PCFR+) — the gadget regime always runs full-CFV.
    fn equilibrium_pass_weighted(
        &mut self,
        root_ids: Vec<u32>,
        perspective_white: bool,
        weights: Vec<f64>,
    ) -> PyResult<()> {
        if weights.len() != root_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "equilibrium_pass_weighted: {} weights for {} roots",
                weights.len(),
                root_ids.len()
            )));
        }
        // Perspective traverser: per-root gadget weight.
        for (i, &r) in root_ids.iter().enumerate() {
            eq_traverse(
                &self.nodes,
                &mut self.infosets,
                &mut self.rng,
                r,
                perspective_white,
                perspective_white,
                /* full_cfv_backprop */ true,
                weights[i],
            );
        }
        // Opponent traverser: unweighted.
        for &r in &root_ids {
            eq_traverse(
                &self.nodes,
                &mut self.infosets,
                &mut self.rng,
                r,
                !perspective_white,
                perspective_white,
                /* full_cfv_backprop */ true,
                1.0,
            );
        }
        Ok(())
    }

    /// FUSED variant of `equilibrium_pass_weighted` (FOW_GADGET_FUSED_EVAL): same
    /// two traverser passes, but RETURNS the perspective-pass per-root values so the
    /// Resolve gadget's RM+ update needs no separate `root_node_values` traversal
    /// (~1 of 3 full tree walks per iteration eliminated at interval=1). The values
    /// are the traversal's own returns — computed under each infoset's pre-update
    /// strategy this iteration, but roots are walked sequentially, so root i+1 sees
    /// regrets root i just updated (a slight intra-pass skew vs `eq_eval`'s clean
    /// snapshot). The caller therefore steps the gadget AFTER the pass: weights lag
    /// the values by ONE iteration (still lockstep cadence, unlike interval=10's
    /// ten-iter lag) — validated on the king-risk rig before use.
    fn equilibrium_pass_weighted_values(
        &mut self,
        root_ids: Vec<u32>,
        perspective_white: bool,
        weights: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        if weights.len() != root_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "equilibrium_pass_weighted_values: {} weights for {} roots",
                weights.len(),
                root_ids.len()
            )));
        }
        let mut values = Vec::with_capacity(root_ids.len());
        for (i, &r) in root_ids.iter().enumerate() {
            values.push(eq_traverse(
                &self.nodes,
                &mut self.infosets,
                &mut self.rng,
                r,
                perspective_white,
                perspective_white,
                /* full_cfv_backprop */ true,
                weights[i],
            ));
        }
        for &r in &root_ids {
            eq_traverse(
                &self.nodes,
                &mut self.infosets,
                &mut self.rng,
                r,
                !perspective_white,
                perspective_white,
                /* full_cfv_backprop */ true,
                1.0,
            );
        }
        Ok(values)
    }

    /// Diagnostics: (total expanded nodes, KLUSS keep-set size or 0 when no
    /// filter). Read-only; used by the KLUSS eq-prune audit to size the gap
    /// between the tree the eq pass WALKS (all nodes) and the subgame KLUSS
    /// would bound it to (keep).
    fn tree_kluss_sizes(&self) -> (usize, usize) {
        (
            self.nodes.len(),
            self.kluss_keep.as_ref().map_or(0, |k| k.len()),
        )
    }

    /// Per-root node value (perspective POV) under the CURRENT strategy profile —
    /// read-only (`eq_eval`, no mutation). The Resolve gadget negates this to get
    /// each world's OPPONENT-POV value `u(x,y|J)` for its follow/exit RM+ update.
    fn root_node_values(&self, root_ids: Vec<u32>, perspective_white: bool) -> Vec<f64> {
        root_ids
            .iter()
            .map(|&r| eq_eval(&self.nodes, &self.infosets, r, perspective_white))
            .collect()
    }

    /// MERGED full-CFV equilibrium pass: ONE walk per root updating both players'
    /// regrets (vs `equilibrium_pass_with`'s two walks). Valid CFR, different
    /// per-iterate values than the two-pass — flag-gated, strength-validated.
    /// See `eq_traverse_merged`. ~2x fewer node visits at the full-CFV regime.
    fn equilibrium_pass_merged(&mut self, root_ids: Vec<u32>, perspective_white: bool) {
        for &r in &root_ids {
            eq_traverse_merged(&self.nodes, &mut self.infosets, r, perspective_white, 1.0);
        }
    }

    /// Weighted MERGED pass for the iterative Resolve gadget: ONE walk per root
    /// updating both players' regrets, with the perspective player's regret
    /// scaled by the gadget world-weight `alpha(J)·P(follow|J)` (the opponent's
    /// in-world best response stays unweighted, mirroring the two-pass split).
    /// 2 full walks per iteration (this + the clean `root_node_values` read)
    /// instead of the two-pass path's 3 — the LAG-FREE throughput lever (the
    /// fused-eval variant's one-iter weight lag flipped a knife-edge king-hang
    /// back on; this keeps the gadget update on a pre-pass snapshot).
    fn equilibrium_pass_merged_weighted(
        &mut self,
        root_ids: Vec<u32>,
        perspective_white: bool,
        weights: Vec<f64>,
    ) -> PyResult<()> {
        if weights.len() != root_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "equilibrium_pass_merged_weighted: {} weights for {} roots",
                weights.len(),
                root_ids.len()
            )));
        }
        for (i, &r) in root_ids.iter().enumerate() {
            eq_traverse_merged(&self.nodes, &mut self.infosets, r, perspective_white, weights[i]);
        }
        Ok(())
    }

    // --- solution / snapshot accessors (read-side, for parity + extraction) ---

    fn current_strategy(&self, infoset: u32, keys: Vec<u32>) -> Vec<f64> {
        if (infoset as usize) >= self.infosets.len() {
            let n = keys.len();
            return vec![1.0 / n as f64; n];
        }
        eq_current_strategy(&self.infosets[infoset as usize], &keys)
    }

    fn last_strategy_get(&self, infoset: u32, key: u32) -> f64 {
        self.infosets
            .get(infoset as usize)
            .and_then(|s| s.actions.get(&key))
            .map(|a| a.last_strategy)
            .unwrap_or(0.0)
    }

    fn value_sum_get(&self, infoset: u32, key: u32) -> f64 {
        self.infosets
            .get(infoset as usize)
            .and_then(|s| s.actions.get(&key))
            .map(|a| a.value_sum)
            .unwrap_or(0.0)
    }

    fn visit_count_get(&self, infoset: u32, key: u32) -> u64 {
        self.infosets
            .get(infoset as usize)
            .and_then(|s| s.actions.get(&key))
            .map(|a| a.visit_count)
            .unwrap_or(0)
    }

    fn value_sq_sum_get(&self, infoset: u32, key: u32) -> f64 {
        self.infosets
            .get(infoset as usize)
            .and_then(|s| s.actions.get(&key))
            .map(|a| a.value_sq_sum)
            .unwrap_or(0.0)
    }

    fn visits_get(&self, infoset: u32) -> u64 {
        self.infosets.get(infoset as usize).map(|s| s.visits).unwrap_or(0)
    }

    fn regret_get(&self, infoset: u32, key: u32) -> f64 {
        self.infosets
            .get(infoset as usize)
            .and_then(|s| s.actions.get(&key))
            .map(|a| a.regret)
            .unwrap_or(0.0)
    }

    fn last_regret_get(&self, infoset: u32, key: u32) -> f64 {
        self.infosets
            .get(infoset as usize)
            .and_then(|s| s.actions.get(&key))
            .map(|a| a.last_regret)
            .unwrap_or(0.0)
    }

    /// Estimated per-component byte counts. Heuristic — uses capacity, not
    /// allocator-reported high water — but close enough to attribute Rust
    /// retention vs the Python cache. Memo-profile-monotonic identified this as
    /// the unmeasured "non-Python residual" lever; this gives caller code an
    /// honest first-order accounting.
    fn memory_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        let node_bytes = std::mem::size_of::<EqNode>();
        let node_capacity = self.nodes.capacity() * node_bytes;
        // Variable child Vec storage per node.
        let mut node_child_bytes: usize = 0;
        for n in &self.nodes {
            node_child_bytes += n.child_keys.capacity() * std::mem::size_of::<u32>();
            node_child_bytes += n.child_nodes.capacity() * std::mem::size_of::<u32>();
        }
        // Each infoset has one actions HashMap of u32 → ActionStats.
        // ActionStats is 6 × 8 = 48 bytes; hash control adds ~8; so ~64B/entry.
        const ISET_ENTRY_BYTES: usize = 64;
        let mut iset_bytes: usize = 0;
        for is in &self.infosets {
            iset_bytes += is.actions.capacity() * ISET_ENTRY_BYTES;
        }
        let intern_bytes = self.infoset_intern.capacity() * (16 + 4 + 12);
        let keep_bytes = self
            .kluss_keep
            .as_ref()
            .map(|s| s.capacity() * (4 + 8))
            .unwrap_or(0);
        d.set_item("n_nodes", self.nodes.len())?;
        d.set_item("n_infosets", self.infosets.len())?;
        d.set_item("nodes_struct_bytes", node_capacity)?;
        d.set_item("nodes_child_bytes", node_child_bytes)?;
        d.set_item("infosets_bytes", iset_bytes)?;
        d.set_item("infoset_intern_bytes", intern_bytes)?;
        d.set_item("kluss_keep_bytes", keep_bytes)?;
        d.set_item(
            "total_bytes",
            node_capacity + node_child_bytes + iset_bytes + intern_bytes + keep_bytes,
        )?;
        Ok(d)
    }
}

// -----------------------------------------------------------------
// KLUSS — knowledge-limited subgame keep-set (Obscuro §3.1 / App. C.6)
// -----------------------------------------------------------------
// Port of src/fow_chess/cfr/kluss.py. BFS over the connectivity graph
// G (vertices = tree nodes; edges = same-white-infoset, same-black-
// infoset, parent↔child) starting from any node in the source infoset
// set. Keep-set = nodes at graph-distance ≤ k+1 (Obscuro uses k=2 →
// I^3 boundary). Helpers live in a non-pymethods impl so they can pass
// Rust-native types (&FxHashSet, &[u32]) that PyO3 can't marshal.

/// One expanded child as returned by expand_node. The trailing two tuples are
/// the chess-shaped obs keys; mini leaves them empty since Python's
/// _rust_expand_and_seed ignores them (the tree interns infosets internally).
type RustExpandChild = (
    u8,
    u8,
    u8,
    u32,
    bool,
    f64,
    (u64, Vec<u64>, Option<u8>, Option<u8>, Option<bool>, &'static str),
    (u64, Vec<u64>, Option<u8>, Option<u8>, Option<bool>, &'static str),
);

impl EqEngine {
    /// Mini-xiangqi expansion — the dark-chess expand_node ported to the mini
    /// board. Generates pseudo-legal children, applies them, interns each
    /// child's infoset from the mini obs-history (hist_white = red, the first
    /// player), and detects the general-capture terminal. The returned obs-key
    /// tuples are empty (Python ignores them).
    #[cfg(feature = "mini")]
    fn expand_node_mini(
        &mut self,
        node_id: u32,
        perspective_white: bool,
    ) -> PyResult<Vec<RustExpandChild>> {
        let parent = self.nodes.get(node_id as usize).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("expand_node: no node {node_id}"))
        })?;
        let prev_setup = match parent.pos.clone() {
            Some(NodePos::Mini(ms)) => ms,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "expand_node: mini node has no mini board (pos)",
                ))
            }
        };
        let parent_to_move_white = parent.to_move_white;
        let parent_hist_white = parent.hist_white;
        let parent_hist_black = parent.hist_black;
        let child_to_move_white = !parent_to_move_white;

        let moves = mini_pseudo_legal(&prev_setup);
        let mut out: Vec<RustExpandChild> = Vec::with_capacity(moves.len());
        let mut child_ids: Vec<u32> = Vec::with_capacity(moves.len());
        let mut child_keys: Vec<u32> = Vec::with_capacity(moves.len());

        for (f, t) in &moves {
            let (f, t) = (*f, *t);
            let next_setup = mini_apply(&prev_setup, f, t);

            // Per-side obs keys → infoset interning (red = "white"/first player).
            let rk = mini_obs_key_core(&prev_setup, &next_setup, true);
            let bk = mini_obs_key_core(&prev_setup, &next_setup, false);

            // Terminal = a general captured. leaf_value from perspective POV
            // (perspective_white = red); terminal_val_white = value to red.
            let red_gone = mini_general_sq(&next_setup, true).is_none();
            let black_gone = mini_general_sq(&next_setup, false).is_none();
            let is_terminal = red_gone || black_gone;
            let (leaf_value, terminal_val_white) = if is_terminal {
                let (own_gone, opp_gone) = if perspective_white {
                    (red_gone, black_gone)
                } else {
                    (black_gone, red_gone)
                };
                let lv = if own_gone && opp_gone {
                    0.0
                } else if own_gone {
                    -1.0
                } else {
                    1.0
                };
                let tvw = if perspective_white { lv } else { -lv };
                (lv, tvw)
            } else {
                (0.0, 0.0)
            };

            let child_hist_white = roll_hist(parent_hist_white, hash_mini_obs_key(&rk));
            let child_hist_black = roll_hist(parent_hist_black, hash_mini_obs_key(&bk));
            let child_hist_for_to_move = if child_to_move_white {
                child_hist_white
            } else {
                child_hist_black
            };
            let infoset = self.intern_infoset(child_to_move_white, child_hist_for_to_move);

            let child_id = self.nodes.len() as u32;
            self.nodes.push(EqNode {
                to_move_white: child_to_move_white,
                is_terminal,
                leaf_value,
                terminal_val_white,
                infoset,
                child_keys: Vec::new(),
                child_nodes: Vec::new(),
                pos: Some(NodePos::Mini(next_setup)),
                hist_white: child_hist_white,
                hist_black: child_hist_black,
            });
            child_ids.push(child_id);
            child_keys.push(mk_move_key(f, t, 0));
            out.push((
                f,
                t,
                0,
                child_id,
                is_terminal,
                leaf_value,
                (0, Vec::new(), None, None, None, ""),
                (0, Vec::new(), None, None, None, ""),
            ));
            self.kluss_parent_of.insert(child_id, node_id);
        }

        let parent = &mut self.nodes[node_id as usize];
        parent.child_keys = child_keys;
        parent.child_nodes = child_ids;
        Ok(out)
    }

    /// Full-Xiangqi expansion: same native-tree contract as mini, but with a
    /// 90-square board and 7-bit action keys. Red is the first player and maps
    /// to the CFR core's "white" bool.
    #[cfg(feature = "xiangqi")]
    fn expand_node_xiangqi(
        &mut self,
        node_id: u32,
        perspective_white: bool,
    ) -> PyResult<Vec<RustExpandChild>> {
        let parent = self.nodes.get(node_id as usize).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("expand_node: no node {node_id}"))
        })?;
        let prev_setup = match parent.pos.clone() {
            Some(NodePos::Xiangqi(xs)) => xs,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "expand_node: xiangqi node has no xiangqi board (pos)",
                ))
            }
        };
        let parent_to_move_white = parent.to_move_white;
        let parent_hist_white = parent.hist_white;
        let parent_hist_black = parent.hist_black;
        let child_to_move_white = !parent_to_move_white;

        let moves = xq_pseudo_legal(&prev_setup);
        let mut out: Vec<RustExpandChild> = Vec::with_capacity(moves.len());
        let mut child_ids: Vec<u32> = Vec::with_capacity(moves.len());
        let mut child_keys: Vec<u32> = Vec::with_capacity(moves.len());

        for (f, t) in &moves {
            let (f, t) = (*f, *t);
            let next_setup = xq_apply(&prev_setup, f, t);

            let rk = xq_obs_key_core(&prev_setup, &next_setup, true);
            let bk = xq_obs_key_core(&prev_setup, &next_setup, false);

            let red_gone = xq_general_sq(&next_setup, true).is_none();
            let black_gone = xq_general_sq(&next_setup, false).is_none();
            let is_terminal = red_gone || black_gone;
            let (leaf_value, terminal_val_white) = if is_terminal {
                let (own_gone, opp_gone) = if perspective_white {
                    (red_gone, black_gone)
                } else {
                    (black_gone, red_gone)
                };
                let lv = if own_gone && opp_gone {
                    0.0
                } else if own_gone {
                    -1.0
                } else {
                    1.0
                };
                let tvw = if perspective_white { lv } else { -lv };
                (lv, tvw)
            } else {
                (0.0, 0.0)
            };

            let child_hist_white = roll_hist(parent_hist_white, hash_xq_obs_key(&rk));
            let child_hist_black = roll_hist(parent_hist_black, hash_xq_obs_key(&bk));
            let child_hist_for_to_move = if child_to_move_white {
                child_hist_white
            } else {
                child_hist_black
            };
            let infoset = self.intern_infoset(child_to_move_white, child_hist_for_to_move);

            let child_id = self.nodes.len() as u32;
            self.nodes.push(EqNode {
                to_move_white: child_to_move_white,
                is_terminal,
                leaf_value,
                terminal_val_white,
                infoset,
                child_keys: Vec::new(),
                child_nodes: Vec::new(),
                pos: Some(NodePos::Xiangqi(next_setup)),
                hist_white: child_hist_white,
                hist_black: child_hist_black,
            });
            child_ids.push(child_id);
            child_keys.push(mk_xiangqi_move_key(f, t));
            out.push((
                f,
                t,
                0,
                child_id,
                is_terminal,
                leaf_value,
                (0, Vec::new(), None, None, None, ""),
                (0, Vec::new(), None, None, None, ""),
            ));
            self.kluss_parent_of.insert(child_id, node_id);
        }

        let parent = &mut self.nodes[node_id as usize];
        parent.child_keys = child_keys;
        parent.child_nodes = child_ids;
        Ok(out)
    }

    /// Full rebuild of kluss_dist + kluss_by_white + kluss_by_black from
    /// scratch. Used when the source infoset changed or this is the first
    /// call. After: kluss_dist holds full BFS distances; bucket maps are
    /// refreshed from all current nodes.
    fn kluss_full_rebuild(&mut self, root_ids: &[u32], source_infoset: u32) {
        let mut source_set: FxHashSet<u32> = FxHashSet::default();
        source_set.insert(source_infoset);
        let dist = self.compute_knowledge_distances(root_ids, &source_set);
        // Rebuild bucket maps to mirror current tree contents.
        self.kluss_by_white.clear();
        self.kluss_by_black.clear();
        for (nid, node) in self.nodes.iter().enumerate() {
            self.kluss_by_white
                .entry(node.hist_white)
                .or_default()
                .push(nid as u32);
            self.kluss_by_black
                .entry(node.hist_black)
                .or_default()
                .push(nid as u32);
        }
        self.kluss_dist = Some(dist);
    }

    /// Lever 3: fold nodes added since the last call into kluss_dist without
    /// re-traversing the whole tree. For each new node, compute its initial
    /// distance from existing neighbors (parent + same-infoset peers) and
    /// itself-as-source if its infoset matches; then BFS-propagate any
    /// shortcut distance reductions to old nodes.
    ///
    /// Worst case is O(V + E) (same as a full BFS) when a deeply-shorter path
    /// opens; typical case is O(|new_nodes| + small frontier) since each
    /// expand_node call only adds ~legal-moves children, and shortcuts are
    /// rare on quiescent stretches.
    fn kluss_incremental_update(&mut self, source_infoset: u32) -> Vec<u32> {
        let prev_count = self.kluss_last_node_count;
        let cur_count = self.nodes.len();
        if cur_count == prev_count {
            return Vec::new(); // Nothing new — dist already current.
        }
        let mut dist = self.kluss_dist.take().expect("dist exists on incremental path");
        let mut queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        // Lever 3 keep-set fix: collect every node whose dist is set/lowered, so
        // set_kluss_keep_from folds just these into the keep-set instead of an
        // O(nodes) rebuild per call. dist only ever DECREASES here, so a touched
        // node can only ENTER the keep-set, never leave.
        let mut touched: Vec<u32> = Vec::new();

        // Stage 1: initial distance for each new node from EXISTING bucket peers
        // and from its parent (via kluss_parent_of). Add the new node to its
        // buckets, so subsequent new nodes can see each other.
        for nid in prev_count..cur_count {
            let nid_u32 = nid as u32;
            let node = match self.nodes.get(nid) {
                Some(n) => n,
                None => continue,
            };
            let mut best: u32 = u32::MAX;
            if node.infoset == source_infoset {
                best = 0;
            }
            // Parent (backward tree edge)
            if let Some(&p) = self.kluss_parent_of.get(&nid_u32) {
                if let Some(&d) = dist.get(&p) {
                    if d != u32::MAX && d + 1 < best {
                        best = d + 1;
                    }
                }
            }
            // Same-white-infoset peers
            if let Some(peers) = self.kluss_by_white.get(&node.hist_white) {
                for &v in peers {
                    if let Some(&d) = dist.get(&v) {
                        if d != u32::MAX && d + 1 < best {
                            best = d + 1;
                        }
                    }
                }
            }
            // Same-black-infoset peers
            if let Some(peers) = self.kluss_by_black.get(&node.hist_black) {
                for &v in peers {
                    if let Some(&d) = dist.get(&v) {
                        if d != u32::MAX && d + 1 < best {
                            best = d + 1;
                        }
                    }
                }
            }
            if best != u32::MAX {
                dist.insert(nid_u32, best);
                queue.push_back(nid_u32);
                touched.push(nid_u32);
            }
            // Add new node to buckets (after computing its initial distance, so
            // it doesn't see itself as a peer).
            self.kluss_by_white
                .entry(node.hist_white)
                .or_default()
                .push(nid_u32);
            self.kluss_by_black
                .entry(node.hist_black)
                .or_default()
                .push(nid_u32);
        }

        // Stage 2: BFS-propagate distance reductions. Same edge set as the full
        // BFS — white-infoset / black-infoset / forward + backward tree.
        while let Some(u) = queue.pop_front() {
            let du = *dist.get(&u).expect("queued node has distance");
            let (hist_w, hist_b, child_nodes) = {
                let node = match self.nodes.get(u as usize) {
                    Some(n) => n,
                    None => continue,
                };
                (node.hist_white, node.hist_black, node.child_nodes.clone())
            };
            let new_d = du + 1;
            // white-infoset peers
            if let Some(peers) = self.kluss_by_white.get(&hist_w) {
                for &v in peers {
                    let cur = dist.get(&v).copied();
                    if cur.map_or(true, |c| new_d < c) {
                        dist.insert(v, new_d);
                        queue.push_back(v);
                        touched.push(v);
                    }
                }
            }
            // black-infoset peers
            if let Some(peers) = self.kluss_by_black.get(&hist_b) {
                for &v in peers {
                    let cur = dist.get(&v).copied();
                    if cur.map_or(true, |c| new_d < c) {
                        dist.insert(v, new_d);
                        queue.push_back(v);
                        touched.push(v);
                    }
                }
            }
            // forward tree edges
            for c in child_nodes {
                let cur = dist.get(&c).copied();
                if cur.map_or(true, |x| new_d < x) {
                    dist.insert(c, new_d);
                    queue.push_back(c);
                    touched.push(c);
                }
            }
            // backward tree edge
            if let Some(&p) = self.kluss_parent_of.get(&u) {
                let cur = dist.get(&p).copied();
                if cur.map_or(true, |c| new_d < c) {
                    dist.insert(p, new_d);
                    queue.push_back(p);
                    touched.push(p);
                }
            }
        }

        self.kluss_dist = Some(dist);
        touched
    }

    /// BFS the tree from `root_ids` via parent→children, returning node ids
    /// in BFS order plus a (child_id → parent_id) map for reverse-tree edges
    /// (the connectivity-graph BFS needs parent hops as well as children).
    fn enumerate_tree_nodes_with_parents(
        &self,
        root_ids: &[u32],
    ) -> (Vec<u32>, FxHashMap<u32, u32>) {
        let mut order: Vec<u32> = Vec::new();
        let mut parent_of: FxHashMap<u32, u32> = FxHashMap::default();
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        let mut q: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        for &r in root_ids {
            if seen.insert(r) {
                q.push_back(r);
            }
        }
        while let Some(u) = q.pop_front() {
            order.push(u);
            if let Some(node) = self.nodes.get(u as usize) {
                for &c in &node.child_nodes {
                    if seen.insert(c) {
                        parent_of.insert(c, u);
                        q.push_back(c);
                    }
                }
            }
        }
        (order, parent_of)
    }

    /// Compute graph-distance from any node whose `infoset` is in
    /// `source_infoset_ids` to every other reachable node. Returns
    /// (node_id → distance). Unreachable nodes are absent. Mirrors
    /// kluss.knowledge_distances exactly:
    ///   • white-infoset edges: nodes sharing `hist_white`
    ///   • black-infoset edges: nodes sharing `hist_black`
    ///   • tree edges: parent ↔ child (both directions)
    /// Source nodes are at distance 0.
    fn compute_knowledge_distances(
        &self,
        root_ids: &[u32],
        source_infoset_ids: &FxHashSet<u32>,
    ) -> FxHashMap<u32, u32> {
        let (order, parent_of) = self.enumerate_tree_nodes_with_parents(root_ids);
        let mut dist: FxHashMap<u32, u32> = FxHashMap::default();
        let mut q: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        // Bucket nodes by hist_white and hist_black for O(1) neighbor lookup.
        let mut by_white: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
        let mut by_black: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
        for &nid in &order {
            if let Some(node) = self.nodes.get(nid as usize) {
                by_white.entry(node.hist_white).or_default().push(nid);
                by_black.entry(node.hist_black).or_default().push(nid);
                if source_infoset_ids.contains(&node.infoset) {
                    dist.insert(nid, 0);
                    q.push_back(nid);
                }
            }
        }
        while let Some(u) = q.pop_front() {
            let du = *dist.get(&u).expect("queued node has distance");
            let (hist_w, hist_b, child_nodes) = {
                let node = match self.nodes.get(u as usize) {
                    Some(n) => n,
                    None => continue,
                };
                (node.hist_white, node.hist_black, node.child_nodes.clone())
            };
            // white-infoset peers
            if let Some(peers) = by_white.get(&hist_w) {
                for &v in peers {
                    if !dist.contains_key(&v) {
                        dist.insert(v, du + 1);
                        q.push_back(v);
                    }
                }
            }
            // black-infoset peers
            if let Some(peers) = by_black.get(&hist_b) {
                for &v in peers {
                    if !dist.contains_key(&v) {
                        dist.insert(v, du + 1);
                        q.push_back(v);
                    }
                }
            }
            // forward tree edges
            for c in child_nodes {
                if !dist.contains_key(&c) {
                    dist.insert(c, du + 1);
                    q.push_back(c);
                }
            }
            // backward tree edge
            if let Some(&p) = parent_of.get(&u) {
                if !dist.contains_key(&p) {
                    dist.insert(p, du + 1);
                    q.push_back(p);
                }
            }
        }
        dist
    }
}

// ---------------------------------------------------------------------------
// PEnumState (E2) — stateful belief-set owner.
//
// Keeps the belief set P in Rust across calls so the millions of FENs never
// round-trip the FFI boundary every move (the transient that dominated
// explosion memory). Python holds a handle and mutates P in place via
// update_own_move / update_opp_move (which reuse own_move_core / opp_move_core);
// only the |I| sampled roots ever cross out, via get_by_index. Because the
// kept (deduped) set is small and never double-held across the boundary, P can
// stay large before any cap, so the truth-dropping downsample rarely/never
// fires → no R1 "P empty" soundness crash. The cap, when set, downsamples in
// Rust via the CPython-faithful MT (seeded from a getstate snapshot) so it
// stays deterministic.
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PEnumState {
    positions: Vec<PackedPos>,
    last_raw: usize,
    last_pre_cap: usize,
    last_downsampled: bool,
    // Bottom-K (KMV) belief bound, set from Python when FOW_BOTTOMK_EXPANSION is
    // on. None => exact build then Python-side MT downsample (legacy, default).
    // Some(k) => bound the opp-move expansion to k DURING the build (peak ~2k,
    // not the full M which reaches 4.2x the cap on explosion plies).
    bottomk_cap: Option<usize>,
}

impl PEnumState {
    /// Impose a canonical, reproducible order on `positions`.
    ///
    /// The explosion update collects the kept set from a PARALLEL/concurrent
    /// build (rayon reduce over per-thread `FxHashSet`s, or `DashSet` inserts),
    /// so `into_iter().collect()` order is NOT reproducible across runs — and
    /// that silently defeated the seeded root sampling in
    /// `PEnumerator.sample_root_fens`: the `rng.sample(range(sz), k)` INDICES
    /// were reproducible, but `get_by_index` mapped them onto a different world
    /// each run → different CFR roots → non-deterministic search/move/strength
    /// (diagnosed 2026-05-29; `PYTHONHASHSEED` couldn't fix it — Rust hasher).
    /// Sorting after every rebuild makes index→world stable, so a fixed seed
    /// reproduces the same game.
    ///
    /// Safe: PHASH parity is unaffected (the golden trace hashes SORTED FENs;
    /// `positions` crosses to Python as a frozenset), and a uniform index sample
    /// is uniform regardless of order, so the sampling distribution is identical.
    #[inline]
    fn canonicalize_order(&mut self) {
        self.positions.par_sort_unstable();
    }
}

#[pymethods]
impl PEnumState {
    #[new]
    fn new(initial_fens: Vec<String>) -> PyResult<Self> {
        let positions: Vec<PackedPos> = initial_fens
            .iter()
            .map(|f| Ok(pack(&parse_fen_lenient(f)?)))
            .collect::<PyResult<_>>()?;
        let mut state = PEnumState { positions, last_raw: 0, last_pre_cap: 0,
                                     last_downsampled: false, bottomk_cap: None };
        state.canonicalize_order();
        Ok(state)
    }

    /// Enable the bottom-K (KMV) belief bound on opp-move expansion. `cap=None`
    /// (default) restores the exact build (legacy, Python-side MT downsample).
    /// Wired behind FOW_BOTTOMK_EXPANSION; when on, the Python enumerator passes
    /// max_size here and SKIPS its post-hoc _rust_downsample (the bound is
    /// applied during the build instead).
    fn set_bottomk_cap(&mut self, cap: Option<usize>) {
        self.bottomk_cap = cap;
    }

    fn size(&self) -> usize {
        self.positions.len()
    }

    #[getter]
    fn last_raw_count(&self) -> usize {
        self.last_raw
    }

    #[getter]
    fn last_pre_cap_count(&self) -> usize {
        self.last_pre_cap
    }

    #[getter]
    fn last_was_downsampled(&self) -> bool {
        self.last_downsampled
    }

    /// Apply the perspective player's own move in place. Returns the new |P|.
    fn update_own_move(
        &mut self, perspective_white: bool, from_idx: u8, to_idx: u8, promo: u8,
    ) -> PyResult<usize> {
        let (next, raw) = own_move_core(&self.positions, perspective_white,
                                        from_idx, to_idx, promo, None)?;
        self.last_raw = raw;
        self.last_pre_cap = next.len();
        self.positions = next;
        self.canonicalize_order();  // reproducible get_by_index → deterministic sampling
        Ok(self.positions.len())
    }

    /// Two-step own move: apply the move AND filter by the perspective's
    /// post-own-move observation (visibility + visible pieces), pruning positions
    /// inconsistent with squares the move just revealed. Returns the new |P|.
    #[allow(clippy::too_many_arguments)]
    fn update_own_move_obs(
        &mut self, perspective_white: bool, from_idx: u8, to_idx: u8, promo: u8,
        obs_visibility_mask: u64,
        obs_white_pawns: u64, obs_white_knights: u64, obs_white_bishops: u64,
        obs_white_rooks: u64, obs_white_queens: u64, obs_white_kings: u64,
        obs_black_pawns: u64, obs_black_knights: u64, obs_black_bishops: u64,
        obs_black_rooks: u64, obs_black_queens: u64, obs_black_kings: u64,
    ) -> PyResult<usize> {
        let obs = Some((
            obs_visibility_mask,
            [obs_white_pawns, obs_white_knights, obs_white_bishops,
             obs_white_rooks, obs_white_queens, obs_white_kings],
            [obs_black_pawns, obs_black_knights, obs_black_bishops,
             obs_black_rooks, obs_black_queens, obs_black_kings],
        ));
        let (next, raw) = own_move_core(&self.positions, perspective_white,
                                        from_idx, to_idx, promo, obs)?;
        self.last_raw = raw;
        self.last_pre_cap = next.len();
        self.positions = next;
        self.canonicalize_order();  // reproducible get_by_index → deterministic sampling
        Ok(self.positions.len())
    }

    /// Apply an opponent move (observation-filtered) in place. Returns new |P|.
    #[allow(clippy::too_many_arguments)]
    fn update_opp_move(
        &mut self,
        opp_white: bool,
        perspective_white: bool,
        obs_visibility_mask: u64,
        obs_white_pawns: u64, obs_white_knights: u64, obs_white_bishops: u64,
        obs_white_rooks: u64, obs_white_queens: u64, obs_white_kings: u64,
        obs_black_pawns: u64, obs_black_knights: u64, obs_black_bishops: u64,
        obs_black_rooks: u64, obs_black_queens: u64, obs_black_kings: u64,
        obs_own_capture_idx: i32,
        obs_opp_capture_landing_idx: i32,
    ) -> PyResult<usize> {
        let (next, raw, pre_cap, was_ds) = opp_move_core(
            &self.positions, self.bottomk_cap,
            opp_white, perspective_white, obs_visibility_mask,
            obs_white_pawns, obs_white_knights, obs_white_bishops,
            obs_white_rooks, obs_white_queens, obs_white_kings,
            obs_black_pawns, obs_black_knights, obs_black_bishops,
            obs_black_rooks, obs_black_queens, obs_black_kings,
            obs_own_capture_idx, obs_opp_capture_landing_idx,
        )?;
        self.last_raw = raw;
        self.last_pre_cap = pre_cap;
        // When bottom-K bounded the build, record the downsample here (the
        // Python side then skips its MT _rust_downsample). Uncapped path leaves
        // last_downsampled for the Python MT path to set, as before.
        if self.bottomk_cap.is_some() {
            self.last_downsampled = was_ds;
        }
        self.positions = next;
        self.canonicalize_order();  // reproducible get_by_index → deterministic sampling
        Ok(self.positions.len())
    }

    /// Downsample to `max_size` uniformly at random IN RUST, deterministically,
    /// using an MT seeded from a CPython getstate() snapshot (so the eviction is
    /// reproducible). Partial Fisher-Yates: swap the first max_size slots with a
    /// random later index, then truncate. No-op when |P| <= max_size.
    fn downsample(&mut self, max_size: usize, mt_words: Vec<u32>, mt_index: usize) {
        let n = self.positions.len();
        if n <= max_size {
            self.last_downsampled = false;
            return;
        }
        let mut rng = Mt19937::from_state(&mt_words, mt_index);
        for i in 0..max_size {
            // uniform j in [i, n)
            let span = (n - i) as f64;
            let j = i + (rng.next_f64() * span) as usize;
            let j = if j >= n { n - 1 } else { j };
            self.positions.swap(i, j);
        }
        self.positions.truncate(max_size);
        self.last_downsampled = true;
    }

    /// Return the FENs at `indices` — the only positions that cross to Python
    /// (the |I| sampled roots). Python picks indices with its own RNG so root
    /// sampling stays reproducible and decoupled from belief storage.
    fn get_by_index(&self, indices: Vec<usize>) -> Vec<String> {
        indices
            .iter()
            .filter_map(|&i| self.positions.get(i).map(|p| Fen(unpack(p)).to_string()))
            .collect()
    }

    /// Snapshot all positions (escape hatch for invariants/debug; avoid on the
    /// hot path — defeats the point of keeping P in Rust).
    fn all_positions(&self) -> Vec<String> {
        self.positions.par_iter().map(|p| Fen(unpack(p)).to_string()).collect()
    }

    /// Per-component byte counts for the belief representation. Lets the
    /// caller (memory profiler / bakeoff telemetry) attribute the
    /// non-Python residual — the dominant unmeasured RSS contributor per
    /// memory-profile-monotonic. `positions_capacity_bytes` is the packed
    /// belief; `positions_size_bytes` is the live |P| × PackedPos size.
    fn memory_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        let entry = std::mem::size_of::<PackedPos>();
        d.set_item("n_positions", self.positions.len())?;
        d.set_item("positions_size_bytes", self.positions.len() * entry)?;
        d.set_item("positions_capacity_bytes", self.positions.capacity() * entry)?;
        d.set_item("packed_pos_bytes", entry)?;
        d.set_item("last_raw", self.last_raw)?;
        d.set_item("last_pre_cap", self.last_pre_cap)?;
        Ok(d)
    }
}

// Variant game modules (Dark Mini Xiangqi, Full Dark Xiangqi): compiled only
// with their cargo features (both default-ON in this repo so prod builds are
// unchanged). The public chess-only export excludes the files and builds with
// default-features off; `#[cfg]`-gated `mod` declarations tolerate the files
// being absent.
#[cfg(feature = "mini")]
mod mini;
#[cfg(feature = "mini")]
use mini::*;
#[cfg(feature = "xiangqi")]
mod xiangqi;
#[cfg(feature = "xiangqi")]
use xiangqi::*;

#[pymodule]
fn fow_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(mt_res53_check, m)?)?;
    m.add_class::<EqEngine>()?;
    m.add_class::<PEnumState>()?;
    #[cfg(feature = "mini")]
    m.add_class::<MiniPEnumState>()?;
    #[cfg(feature = "xiangqi")]
    m.add_class::<XiangqiPEnumState>()?;
    m.add_function(wrap_pyfunction!(fen_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(apply_move, m)?)?;
    m.add_function(wrap_pyfunction!(update_opp_move_rust, m)?)?;
    m.add_function(wrap_pyfunction!(update_own_move_rust, m)?)?;
    m.add_function(wrap_pyfunction!(visible_squares, m)?)?;
    m.add_function(wrap_pyfunction!(visible_squares_bb, m)?)?;
    m.add_function(wrap_pyfunction!(consistent_with_bb, m)?)?;
    m.add_function(wrap_pyfunction!(observation_from_transition_bb, m)?)?;
    m.add_function(wrap_pyfunction!(observation_from_transition_both_bb, m)?)?;
    m.add_function(wrap_pyfunction!(obs_keys_both_bb, m)?)?;
    #[cfg(feature = "mini")]
    m.add_function(wrap_pyfunction!(mini_pseudo_legal_moves, m)?)?;
    #[cfg(feature = "mini")]
    m.add_function(wrap_pyfunction!(mini_fen_roundtrip, m)?)?;
    #[cfg(feature = "mini")]
    m.add_function(wrap_pyfunction!(mini_visible_squares, m)?)?;
    #[cfg(feature = "mini")]
    m.add_function(wrap_pyfunction!(mini_obs_keys_both, m)?)?;
    #[cfg(feature = "xiangqi")]
    m.add_function(wrap_pyfunction!(xiangqi_pseudo_legal_moves, m)?)?;
    #[cfg(feature = "xiangqi")]
    m.add_function(wrap_pyfunction!(xiangqi_fen_roundtrip, m)?)?;
    #[cfg(feature = "xiangqi")]
    m.add_function(wrap_pyfunction!(xiangqi_visible_squares, m)?)?;
    #[cfg(feature = "xiangqi")]
    m.add_function(wrap_pyfunction!(xiangqi_obs_keys_both, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Native-level parity pins. The heavyweight oracle is the Python test
    //! suite (tests/test_rust_diff_* replays real games against python-chess
    //! byte-for-byte); these tests exist so `cargo test` alone — with no
    //! Python environment — still verifies the crate against golden values
    //! computed from the Python reference implementation.

    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn fen_roundtrip_startpos() {
        assert_eq!(fen_roundtrip(STARTPOS).unwrap(), STARTPOS);
    }

    #[test]
    fn pseudo_legal_startpos_has_20_moves() {
        assert_eq!(pseudo_legal_moves(STARTPOS).unwrap().len(), 20);
    }

    #[test]
    fn apply_move_double_push_matches_python_chess() {
        // python-chess golden: Board().push(e2e4).fen()
        let after = apply_move(STARTPOS, 12, 28, 0).unwrap(); // e2 -> e4
        assert_eq!(
            after,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        );
    }

    #[test]
    fn apply_move_castling_matches_python_chess() {
        // Italian-ish position; python-chess golden after e1g1 (O-O).
        let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
        let after = apply_move(fen, 4, 6, 0).unwrap(); // e1 -> g1
        assert_eq!(
            after,
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4"
        );
    }

    #[test]
    fn apply_move_promotion_matches_python_chess() {
        // python-chess golden after a7a8=Q.
        let fen = "8/P7/8/8/8/7k/8/7K w - - 0 1";
        let after = apply_move(fen, 48, 56, 5).unwrap(); // a7 -> a8, Role 5 = Queen
        assert_eq!(after, "Q7/8/8/8/8/7k/8/7K b - - 0 1");
    }

    #[test]
    fn visible_squares_matches_python_reference() {
        // Goldens computed with fow_chess.visibility._visible_squares_py
        // (the maintained pure-Python reference), 2026-08-22.
        assert_eq!(visible_squares(STARTPOS, true).unwrap(), 0xffffffff);
        assert_eq!(
            visible_squares(STARTPOS, false).unwrap(),
            0xffffffff00000000
        );
        let italian =
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
        assert_eq!(visible_squares(italian, true).unwrap(), 0x20115adfefffff);
    }
}
