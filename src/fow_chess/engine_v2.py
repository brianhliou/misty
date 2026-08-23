"""V2 Fog of War engine — assembled from A1-A6 components.

Stateful per-game engine that combines:
- A1 Stockfish leaf eval (StockfishLeafEval)
- A2 PCFR+ inner solver (in tabular.py, also used by GT-CFR)
- A3 exact P enumeration (PEnumerator)
- A4 GT-CFR substrate (GTCFRTreeNode, expand_leaf, etc.)
- A5.1 multi-root shared-regret coordinator (solve_multiroot_growing_subgame)
- A6.1 purification + stable-actions filter (purify_strategy)

Per game, a single EngineV2 instance is alive for one perspective
player. It maintains the PEnumerator across moves:

* ``observe_opp_move(obs)`` — opponent just moved; update P via the
  observation we received.
* ``observe_own_move(move)`` — we just played ``move``; update P
  deterministically.
* ``choose_move(*, iterations, time_budget_seconds)`` — sample I⊆P,
  run multi-root GT-CFR with time budget, purify, return the top
  action. Default ``max_actions=1`` (Resolve regime) until A6.2
  ships the Maxmargin gadget.

This module does NOT touch the v0.9.5 substrate (engine.py,
strategies.py, belief.py) — those stay alive as the bakeoff baseline
per the post-A7 delete plan.
"""

from __future__ import annotations

import os
import random
from types import SimpleNamespace
from typing import Any, Iterable

import chess

from .cfr.gt_cfr import (
    sample_roots_from_P,
    solve_multiroot_growing_subgame,
    solve_multiroot_rust_tree,
)
from .cfr.blueprint import (
    CarryoverBlueprint,
    NetBlueprint,
    StockfishBlueprint,
    StubBlueprint,
)
from .cfr.leaf_eval_stockfish import StockfishLeafEval
from .cfr.purification import PurifiedStrategy, purify_strategy, select_regime
from .cfr.time_manager import TimeManager
from .observation import Observation
from .opening_book import (
    load as _load_opening_book,
    observation_event_fingerprint,
    observation_history_fingerprint,
)
from . import rust_health
from .rules import ChessRules, Rules
from .variant_hooks import for_rules as _variant_hooks_for

# Distinct per-guard RNG seeds: each commit guard samples belief worlds with its
# own seed so the guards' draws stay independent. Keep these DISTINCT — collapsing
# them to a single seed would correlate the samples across guards.
_GUARD_SEED_RISK = 424242      # _commit_risk_check
_GUARD_SEED_MATERIAL = 424243  # _commit_material_check
_GUARD_SEED_PRUNE = 424244     # _catastrophe_prune (HV-prune)


_DEFAULT_I_SAMPLE_SIZE = 16
_DEFAULT_ITERATIONS = 500
_DEFAULT_MAX_ACTIONS = 1
# |P| cap: None = UNCAPPED (exact enumeration, truth-in-P always holds). The
# old default of 10_000 was pathological — it caps BELOW Obscuro's measured
# average |P| (~17K), so it downsampled typical belief sets and could evict the
# true position (the cap is fundamentally unsound: in FoW you can't identify
# which P-member is real, so any random downsample can drop reality -> P loses
# the truth or empties -> R1 crash). With the packed PackedPos + concurrent
# DashSet dedup, real-play |P| (~1M, matching Obscuro's worst case) costs ~1-1.5
# GB, so uncapped is affordable and sound. Callers that must bound a pathological
# explosion can pass an explicit high never-fire guard, but the DEFAULT is sound.
_DEFAULT_P_MAX_SIZE = None


def _upgrade_dominated_promotion(move: chess.Move) -> chess.Move:
    """Break a promotion tie toward the dominant piece.

    In FoW there is no stalemate, so a queen STRICTLY dominates a rook or
    bishop — promoting to either is never better than to a queen. CFR can't
    distinguish them when both lines reach a king-capture and hit the
    material-blind +1 ceiling, leaving the engine indifferent (root-caused in
    ``lab/diag_king_capture_leaves.py``: the rook line's en-prise leaves get the
    same flat +1 as the queen's, erasing the material gradient; the choice is
    within search noise — the rust and python trees disagree on it). This
    commits the dominant piece when the value function is indifferent. A KNIGHT
    promotion is left ALONE — it is not dominated (it can reach king-capture
    squares the queen can't).

    Upgrades UNCONDITIONALLY: a queen promotion to the same square is always
    legal whenever the rook/bishop promotion is (the piece is a free choice),
    so no root-action membership check is needed — and such a check would
    spuriously fail here anyway, since the CFR strategy dict omits the
    zero-probability queen action exactly when the engine committed the rook.
    """
    if move.promotion in (chess.ROOK, chess.BISHOP):
        return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move


def _carryover_pos_key(full_fen: str) -> str:
    """Position identity for subtree-carryover matching: the FEN's first four
    fields (pieces, side-to-move, castling, en passant) — i.e. EPD, dropping the
    path-dependent halfmove/fullmove counters.

    The rust `node_pos_to_fen` and python `rules.root_fen` both emit a FULL FEN;
    they AGREE on these four fields for the same game state but their move
    counters can differ by path, so matching on the full FEN never hits. The old
    code matched `rules.board_fen` (piece-placement ONLY) against full-FEN keys —
    which never matched either (silent no-op: carryover never reused a subtree).
    Piece-only would be UNSOUND anyway (it conflates castling/ep — could reuse a
    wrong-rights node). EPD is the correct identity: same epd ⇒ same state.
    Assumes chess-style FEN field order (carryover is chess-only today)."""
    return " ".join(full_fen.split()[:4])


def _apply_legal_move_mask(
    rules: Rules,
    move: Any,
    legal_moves: Iterable[Any] | None,
    action_values: dict[Any, float],
    strategy: dict[Any, float],
) -> Any:
    """Constrain a committed action to an externally supplied legal set.

    Imperfect-information roots can contain actions that are legal in sampled
    belief worlds but illegal in the actual hidden state. Live clients already
    know the true legal move set, so callers may pass it as a final commit mask.
    """
    if legal_moves is None:
        return move
    legal = list(legal_moves)
    if not legal:
        return move

    legal_by_key = {rules.action_key(candidate): candidate for candidate in legal}
    if rules.action_key(move) in legal_by_key:
        return legal_by_key[rules.action_key(move)]

    legal_action_keys = set(legal_by_key)
    legal_solved = [
        candidate
        for candidate in action_values
        if rules.action_key(candidate) in legal_action_keys
    ]
    if legal_solved:
        best = max(
            legal_solved,
            key=lambda candidate: (
                action_values.get(candidate, float("-inf")),
                strategy.get(candidate, 0.0),
            ),
        )
        return legal_by_key[rules.action_key(best)]
    return legal[0]


class EngineV2:
    """Stateful v2 engine for one perspective in one game.

    Args:
        perspective: which color this engine plays.
        starting_board: starting position (defaults to standard chess
            starting position).
        stockfish: optional pre-built StockfishLeafEval. If None, a
            fresh subprocess is spawned. Caller-supplied lets multiple
            EngineV2 instances share a single Stockfish in tests.
        rng: deterministic RNG for sampling. Defaults to random.Random().
    """

    def __init__(
        self,
        perspective: chess.Color,
        *,
        starting_board: chess.Board | None = None,
        stockfish: StockfishLeafEval | None = None,
        rng: random.Random | None = None,
        p_max_size: int | None = _DEFAULT_P_MAX_SIZE,
        use_rust_state: bool = True,
        resolve_gadget: bool | None = None,
        resolve_cvar_q: float | None = None,
        use_lean: bool | None = None,
        queen_promo_tiebreak: bool | None = None,
        rules: Rules | None = None,
        win_fast: bool | None = None,
        gadget_faithful: bool | None = None,
        gadget_alpha: bool | None = None,
        gadget_iterative: bool | None = None,
        structural_carry: bool | None = None,
        carryover_subtree: bool | None = None,
        resolve_blueprint: str | None = None,
        expansion_budget: int | None = None,
        kluss_soft: bool | None = None,
        hv_prune_frac: float | None = None,
        hv_prune_adaptive: bool | None = None,
        hv_prune_tau: float | None = None,
        hv_prune_pref: float | None = None,
        hv_prune_pmax: float | None = None,
        hv_prune_king_floor: float | None = None,
        commit_royal_guard: bool | None = None,
        commit_royal_gap: float | None = None,
        commit_royal_floor: float | None = None,
        commit_royal_threat_guard: bool | None = None,
        commit_royal_flight_guard: bool | None = None,
        commit_royal_flight_gap: float | None = None,
        commit_royal_flight_extra_risk: float | None = None,
        commit_royal_flight_min_frac: float | None = None,
        commit_material_guard: bool | None = None,
        commit_material_gap: float | None = None,
        commit_material_floor: float | None = None,
        commit_material_min_value: float | None = None,
        commit_general_veto: bool | None = None,
        carryover_min_p: int | None = None,
    ) -> None:
        self.perspective = perspective
        # The game seam (Phase 1 of verticalization). ChessRules forwards to
        # today's exact behavior; a Rust MiniXiangqiRules adapter slots in here
        # later. This driver routes its behavioral chess calls through self.rules;
        # the enumerator / gt_cfr hot paths are rewired in later slices.
        self.rules: Rules = rules if rules is not None else ChessRules()
        # Variant-specific engine hooks (commit guards etc.), registered by the
        # variant's rules module at import time. None for chess.
        self._variant_hooks = _variant_hooks_for(self.rules)
        # Pure-python-wheel installs run the fallback belief path; say so once,
        # loudly (identical results, ~500x slower — a measurement landmine).
        rust_health.warn_once_if_unavailable()
        # Per-arm expansion budget (None = read FOW_V2_EXPANSION_BUDGET per move).
        # The env is process-wide and LEAKED to the opponent arm in bakeoffs —
        # at i=32 the same eb gives ~6x deeper per-world trees than at i=200,
        # silently favoring the sparring partner. Constructor wins over env.
        self._expansion_budget = expansion_budget
        # Per-arm soft-KLUSS (None = read FOW_KLUSS_SOFT per solve). Same
        # per-arm rationale as expansion_budget: constructor wins over env.
        self._kluss_soft = kluss_soft
        # Per-arm catastrophe-prune knobs (None = read FOW_HV_PRUNE_* per move).
        # MUST be per-arm: the prune is read at commit time, so a shared process
        # env can't differ between two in-process engines (a v2-vs-v2 canary) —
        # the gadget/expansion-budget leak, fixed the same way (constructor wins).
        self._hv_prune_frac = hv_prune_frac
        self._hv_prune_adaptive = hv_prune_adaptive
        self._hv_prune_tau = hv_prune_tau
        self._hv_prune_pref = hv_prune_pref
        self._hv_prune_pmax = hv_prune_pmax
        # King-step severity (None = read FOW_HV_PRUNE_KING_FLOOR; 0 = legacy threshold-free
        # king redirect; >0 = cost-benefit redirect, value conceded vs king-risk x sev).
        self._hv_prune_king_floor = hv_prune_king_floor
        # Game-generic royal safety guard (king/general): the chess commit guards
        # below are still material/SEE-specific, so DMX needs the terminal-risk
        # part lifted through Rules. Constructor wins over env for per-game pools.
        self._commit_royal_guard = commit_royal_guard
        self._commit_royal_gap = commit_royal_gap
        self._commit_royal_floor = commit_royal_floor
        self._commit_royal_threat_guard = commit_royal_threat_guard
        self._commit_royal_flight_guard = commit_royal_flight_guard
        self._commit_royal_flight_gap = commit_royal_flight_gap
        self._commit_royal_flight_extra_risk = commit_royal_flight_extra_risk
        self._commit_royal_flight_min_frac = commit_royal_flight_min_frac
        self._commit_material_guard = commit_material_guard
        self._commit_material_gap = commit_material_gap
        self._commit_material_floor = commit_material_floor
        self._commit_material_min_value = commit_material_min_value
        self._commit_general_veto = commit_general_veto
        # Carryover |P|-gate (None = read FOW_CARRYOVER_MIN_P, default 0 = no gate =
        # byte-identical). When |P| <= this, SKIP carryover/structural-carry and solve
        # fresh: at small |P| a fresh re-sample is the COMPLETE exact belief, which the
        # search-biased carried subset would corrupt (it under-weights rare threat
        # worlds -> wrong opening moves, e.g. 1.c4 d5 2.Nc3 Nc6?? lab/nc6_carryover.py).
        # Carryover only earns its keep at large |P| where you can only ever sample.
        self._carryover_min_p = carryover_min_p
        self.rng = rng if rng is not None else random.Random()
        # Never commit a rook/bishop underpromotion when the queen promo is
        # legal (queen strictly dominates them in FoW) — see
        # _upgrade_dominated_promotion. Constructor arg overrides the env
        # (per-arm bakeoff control); None = read FOW_QUEEN_PROMO_TIEBREAK.
        self.queen_promo_tiebreak: bool = (
            (os.environ.get("FOW_QUEEN_PROMO_TIEBREAK") == "1")
            if queen_promo_tiebreak is None
            else bool(queen_promo_tiebreak)
        )
        # Win-fast: among equal-top-value moves, prefer a CONFIDENT immediate
        # capture of the enemy royal (king/general). The value saturates at +1
        # for ALL winning moves, so without this the engine dawdles in won
        # positions (the game-11 shuffle). Per-instance + default OFF, so dark
        # chess is byte-identical (the golden gate stays green); a mini engine
        # opts in. None = read FOW_WIN_FAST.
        self.win_fast: bool = (
            (os.environ.get("FOW_WIN_FAST") == "1")
            if win_fast is None
            else bool(win_fast)
        )
        self.enumerator = self.rules.make_belief(
            perspective,
            starting_board=starting_board,
            max_size=p_max_size,
            rng=self.rng,
            use_rust_state=use_rust_state,
        )
        # use_lean=None -> StockfishLeafEval reads FOW_LEAN_UCI; an explicit bool
        # (from a per-arm bakeoff flag) overrides it. Ignored when a pre-built
        # stockfish is supplied (that instance already chose its path).
        self._stockfish = (
            stockfish if stockfish is not None
            else StockfishLeafEval(use_lean=use_lean)
        )
        self._owns_stockfish = stockfish is None
        # Lever 1 Phase 1: persistent EqEngine across choose_move. Constructed
        # lazily on first rust-tree pick to avoid pulling in fow_rust when only
        # the python-tree path is used. RNG state at construction is a
        # placeholder; reset_tree() per pick re-seeds from self.rng so the
        # per-call rng consumption matches the prior per-call-construction
        # behavior (byte-parity preserved). Phase 2 will swap reset_tree() for
        # actual prune-to-subtree calls.
        self._eq_engine = None
        # Lever 1 Phase 2 (variant b): when True, between-move resets preserve
        # infoset_intern + per-infoset CFR state. Next move's search warm-starts
        # at infosets that match a prior search's (to_move, hist) hash. Opt-in,
        # default False to preserve byte-parity vs prior behavior.
        self.carryover_infosets: bool = bool(
            os.environ.get("FOW_CARRYOVER_INFOSETS") == "1"
        )
        # Lever 1 Phase 2a (subtree reuse): additionally preserve the prior
        # search's nodes and reuse depth-2 grandchildren as new roots when
        # their truth FEN matches a newly-sampled root truth. Strictly
        # stronger than carryover_infosets alone (it implies it). Opt-in;
        # FOW_CARRYOVER_SUBTREE=1.
        # Constructor arg overrides the env (per-arm bakeoff control); None = env.
        self.carryover_subtree: bool = (
            (os.environ.get("FOW_CARRYOVER_SUBTREE") == "1")
            if carryover_subtree is None
            else bool(carryover_subtree)
        )
        # The continual-resolve stack flags — faithful gadget aggregation,
        # non-uniform alpha, structural Γ̂-carry. Same per-arm override pattern
        # (arg overrides FOW_* env; None = read env, default OFF). Always set so
        # they're defined regardless of resolve_gadget. gadget_faithful/_alpha
        # thread to gt_cfr's _apply_resolve_gadget via solve_multiroot_rust_tree;
        # structural_carry is read in choose_move's root-set construction.
        self.gadget_faithful: bool = (
            (os.environ.get("FOW_GADGET_FAITHFUL") == "1")
            if gadget_faithful is None
            else bool(gadget_faithful)
        )
        self.gadget_alpha: bool = (
            (os.environ.get("FOW_GADGET_ALPHA") == "1")
            if gadget_alpha is None
            else bool(gadget_alpha)
        )
        self.structural_carry: bool = (
            (os.environ.get("FOW_STRUCTURAL_CARRY") == "1")
            if structural_carry is None
            else bool(structural_carry)
        )
        # PROPER (iterative) Resolve gadget (proper-gadget Step 1): couple the
        # follow/exit gadget INTO the eq loop rather than the read-only post-hoc
        # cap. Threads to gt_cfr.solve_multiroot_rust_tree; only meaningful with
        # resolve_gadget on. Same per-arm override (arg wins over FOW_GADGET_ITERATIVE
        # env; None = env, default OFF → byte-identical).
        self.gadget_iterative: bool = (
            (os.environ.get("FOW_GADGET_ITERATIVE") == "1")
            if gadget_iterative is None
            else bool(gadget_iterative)
        )
        # Phase 2a state: the previous choose_move's root_ids (in fow_rust
        # node-id space) and the action_key the perspective played from those
        # roots. Used by next choose_move's discovery walk.
        self._prev_root_ids: list[int] | None = None
        self._prev_played_action_key: int | None = None
        # Concern 3 of the parity audit: full counterfactual-value backprop on
        # the opponent branch (vs the default external sampling). Obscuro-
        # faithful. Per-iter cost goes up; per-iter regret quality goes up.
        # Convergence target same; intermediate strategies differ.
        # FOW_FULL_CFV_BACKPROP=1.
        self.full_cfv_backprop: bool = bool(
            os.environ.get("FOW_FULL_CFV_BACKPROP") == "1"
        )
        # Resolve gadget (MVP, Slice 1): after the solve, cap each sampled
        # world's value by a blueprint baseline and pick the per-world
        # worst-case-safe move — the structural fix for aggregation-dilution
        # (a uniform belief-mean cannot escape it; see
        # docs/engine/belief-retention-scoping-2026-05-28.md). Read-only
        # over the solved tree. Implies full-CFV so per-world opponent values
        # are the lower-variance Obscuro-faithful estimates. Opt-in;
        # FOW_RESOLVE_GADGET=1, FOW_RESOLVE_MARGIN / FOW_RESOLVE_OPP_CFV tune it.
        # An explicit constructor arg (not None) overrides the env var — lets a
        # harness put the gadget on ONE arm only when both engines share a
        # process env (e.g. the v2-vs-v2 bakeoff). None = read the env (default).
        self.resolve_gadget: bool = (
            (os.environ.get("FOW_RESOLVE_GADGET") == "1")
            if resolve_gadget is None
            else bool(resolve_gadget)
        )
        if self.resolve_gadget:
            self.full_cfv_backprop = True
            self.gadget_margin: float = float(os.environ.get("FOW_RESOLVE_MARGIN", "0.0"))
            # CVaR tail fraction for the gadget objective (worst-q-fraction mean,
            # robust to per-world value noise — see gt_cfr._apply_resolve_gadget).
            # 0.1 default; -> 0 is pure worst-case, 1 is the full-belief mean.
            # Explicit arg overrides the env (per-arm control), same as above.
            self.gadget_cvar_q: float = (
                float(os.environ.get("FOW_RESOLVE_CVAR_Q", "0.1"))
                if resolve_cvar_q is None
                else float(resolve_cvar_q)
            )
            # FOW_RESOLVE_BLUEPRINT=stub (constant opp_cfv, default — mechanism
            # only, over-defends) | stockfish (per-world Stockfish eval, the
            # miscalibrated baseline that re-dilutes) | net (learned CFV-proxy)
            # | carryover (the Obscuro-faithful continual-resolving baseline:
            # the previous move's solved u(x,y|J), in the solve's value space —
            # populated per-move by choose_move's Slice-0 extraction; needs
            # FOW_CARRYOVER_SUBTREE=1 to have a prior tree to read).
            _bp = (resolve_blueprint if resolve_blueprint is not None
                   else os.environ.get("FOW_RESOLVE_BLUEPRINT", "stub"))
            if _bp == "stockfish":
                self.gadget_blueprint = StockfishBlueprint(self._stockfish, not self.perspective)
            elif _bp == "net":
                # Obscuro-Parity Phase 1: learned-value CFV-proxy baseline.
                self.gadget_blueprint = NetBlueprint(
                    os.environ["FOW_RESOLVE_NET_WEIGHTS"], not self.perspective
                )
            elif _bp == "carryover":
                # Obscuro-Parity Phase 2: continual-resolve baseline. Empty until
                # the first set_values; the fallback doubles as the stub baseline
                # for uncarried worlds (move 1, unexplored branches). With
                # FOW_GADGET_C1_FALLBACK=1 uncarried worlds instead get the
                # paper-C.1 alternate value min{ṽ(h), v*} (stockfish + prev root
                # value handed here; see CarryoverBlueprint).
                self.gadget_blueprint = CarryoverBlueprint(
                    self.rules,
                    fallback=float(os.environ.get("FOW_RESOLVE_OPP_CFV", "-0.1")),
                    stockfish=self._stockfish,
                    opponent_color=not self.perspective,
                )
            else:
                self.gadget_blueprint = StubBlueprint(
                    opp_cfv=float(os.environ.get("FOW_RESOLVE_OPP_CFV", "-0.1"))
                )
        else:
            self.gadget_margin = 0.0
            self.gadget_blueprint = None
            self.gadget_cvar_q = 0.1
        # Diagnostic counters
        self.moves_chosen = 0
        self.last_solution = None  # MultiRootGTCFRSolution
        # FENs of the roots the last solve actually sampled (the I ⊆ P the
        # search reasoned over). Telemetry for analysis (truth-in-I); does not
        # affect play.
        self.last_root_fens: list[str] | None = None
        self.last_purified: PurifiedStrategy | None = None
        # A6.2 telemetry: True if the last choose_move ran regime
        # auto-selection (max_actions=0), False if the caller pinned
        # a specific regime.
        self.last_regime_auto: bool = False

    def observe_opp_move(self, observation: Observation) -> None:
        """Opp just played; update P via what we observed."""
        self.enumerator.update_opp_move(observation)
        self._observed_plies = getattr(self, "_observed_plies", 0) + 1

    def observe_own_move(
        self, move: chess.Move, observation: Observation | None = None
    ) -> None:
        """We just played ``move``; update P. Two-step: if ``observation`` (our
        view after the move) is given, also prune positions inconsistent with the
        squares the move revealed (see PEnumerator.update_own_move)."""
        self.enumerator.update_own_move(move, observation)
        self._observed_plies = getattr(self, "_observed_plies", 0) + 1
        # Phase 2a: record the action_key of the played move so the next
        # choose_move can discover carryover candidates among depth-2
        # grandchildren of the previous search's roots.
        self._prev_played_action_key = self.rules.action_key(move)

    def _carryover_blueprint_values(self) -> dict[str, float]:
        """Obscuro-Parity Phase 2 Slice 0: per-world ``u(x, y | J)`` (opponent
        POV) carried from the previous move's solved tree, keyed by the
        resulting world's ``rules.board_fen``.

        Continual re-solving anchors the gadget to the value the PREVIOUS solve
        assigned to each of THIS move's opponent infosets. At our move the gadget
        treats each sampled world (an opponent-distinguishable true state where
        we are to move) as one opponent infoset ``J`` at the subgame root, so
        ``opp_cfv(world)`` must be the carried value *at that world's state* —
        ``eq_eval`` of the world's node in the prior tree — NOT the value at the
        opponent's earlier decision node one ply up. (Mapping every world to the
        parent node's value is an off-by-one-ply error that mis-calibrates the
        gadget baseline.)

        Path in the prior tree: ``prev_root`` (us to move, move N) --[the move we
        played]--> ``J`` (opponent to move, move N's reply node) --[an opponent
        move]--> ``gc`` (us to move, move N+1) == one of THIS move's worlds. So
        for each prior root we take ``J``, then map each child ``gc`` of ``J`` to
        the opponent-POV value at ``gc``, aligned with ``node_children(J)`` for
        the node ids → FENs.

        **POV (subtle — read our POV and negate, do NOT pass the opponent
        perspective).** ``EqEngine.root_child_values`` / ``eq_eval`` flips
        *terminal* values by ``perspective_white`` (``tw`` vs ``-tw``) but returns
        *leaf* (Stockfish-evaluated) values UNFLIPPED — leaf values are stored in
        the solve's perspective (ours). Our worlds are leaf-bottomed, so calling
        ``eq_eval(gc, opp_perspective)`` returns the value in OUR POV, not the
        opponent's — a sign-inverted baseline that anchors the gadget backwards
        (measured: 19-39-2 / 33% H2H). So we read ``eq_eval(gc, OUR perspective)``
        (clean: leaves as-is = our POV, terminals as ``tw`` = our POV) and NEGATE
        it to get the opponent-POV ``opp_cfv``.

        **Coverage:** ``J``'s subgame is only populated when the previous search
        actually EXPANDED the move we played — i.e. on-policy, the engine's own
        searched continuation. A move the prior search visited at the root but
        never expanded (an off-policy / forced move) leaves ``J`` childless, so
        those worlds get no carried value and fall back. In real play we commit
        the move we just searched, so its subtree is the one that carries.

        Pure read — no tree mutation — over the prior tree, which
        ``reset_for_carryover`` preserves. Last-wins on duplicate FENs, mirroring
        ``discover_carryover_candidates``'s ``dict(pairs)``. Empty when there is
        no prior tree (first move / subtree-carryover off); the blueprint absorbs
        that via its fallback.
        """
        eng = self._eq_engine
        if eng is None or self._prev_root_ids is None or self._prev_played_action_key is None:
            return {}
        persp_white = self.rules.is_first_player(self.perspective)
        vals: dict[str, float] = {}
        for prev_root in self._prev_root_ids:
            nc = eng.node_children(prev_root)
            if nc is None:
                continue
            keys, child_nodes = nc
            try:
                j_idx = keys.index(self._prev_played_action_key)
            except ValueError:
                continue  # this prior root never explored the move we played
            j_node = child_nodes[j_idx]
            # eq_eval(gc) for each child gc of J, in J's child order, in OUR POV
            # (see POV note above: leaf values don't flip, so read our perspective
            # and negate for the opponent-POV opp_cfv).
            gc_vals = eng.root_child_values([j_node], persp_white)[0]
            gnc = eng.node_children(j_node)
            if not gc_vals or gnc is None:
                continue  # J unexpanded (off-policy move) — no worlds carried here
            for (_gkey, gval), gc in zip(gc_vals, gnc[1], strict=False):
                nf = eng.node_fen(gc)
                if nf is not None:
                    # Key by the SAME canonical form opp_cfv looks up:
                    # rules.board_fen (piece placement only). node_fen is a FULL
                    # FEN (side-to-move / castling / counters), so keying by it
                    # NEVER matches the gadget's rules.board_fen(world) lookup
                    # (0% coverage — the blueprint silently never engages). Route
                    # both through rules so the keys match by construction.
                    key = self.rules.board_fen(self.rules.board_from_fen(nf))
                    vals[key] = -gval  # opp_cfv(world gc) = -eq_eval(gc, our POV)
        return vals

    def _carryover_blueprint_reach(self) -> dict[str, float]:
        """Slice 2 (FOW_GADGET_ALPHA): per-world opponent reach ``y(J)`` carried
        from the previous solve, keyed like :meth:`_carryover_blueprint_values`.

        Same prior-tree walk (``prev_root`` --[the move we played]--> ``J`` (opp
        reply) --[an opponent move]--> ``gc`` == one of THIS move's worlds), but
        instead of the world's value we read ``y(world gc)`` = the prior solve's
        opponent strategy probability at ``J`` for the move leading to ``gc``
        (``current_strategy`` over ``J``'s child keys — the PRM+ current iterate,
        the same strategy readout the non-gadget root uses at
        ``gt_cfr.py``). A probability, so NO POV / sign flip (unlike the value
        carry). Feeds the gadget's non-uniform ``alpha(J) = ½(y/Σy + 1/m)``.
        Empty when there is no prior tree (first move / carryover-subtree off) ->
        the gadget falls back to uniform alpha. Pure read over the prior tree
        (preserved by ``reset_for_carryover``).

        Duplicate FENs ACCUMULATE (sum, not last-wins): the paper's ``y(J)`` is
        the probability the opponent blueprint strategy *generates* infoset J,
        i.e. a sum over all histories reaching it. A world reachable from
        several prior roots collects each root's strategy mass — last-wins
        (the value carry's rule, fine for a value, wrong for a distribution)
        would keep one arbitrary path's probability and understate exactly the
        worlds the blueprint considers most likely.
        """
        eng = self._eq_engine
        if eng is None or self._prev_root_ids is None or self._prev_played_action_key is None:
            return {}
        reach: dict[str, float] = {}
        for prev_root in self._prev_root_ids:
            nc = eng.node_children(prev_root)
            if nc is None:
                continue
            keys, child_nodes = nc
            try:
                j_idx = keys.index(self._prev_played_action_key)
            except ValueError:
                continue  # this prior root never explored the move we played
            j_node = child_nodes[j_idx]
            gnc = eng.node_children(j_node)
            if gnc is None:
                continue
            gkeys, gc_nodes = gnc
            if not gkeys:
                continue  # J unexpanded (off-policy move) — no worlds carried here
            inf = eng.node_infoset(j_node)
            if inf is None:
                continue
            # y(J): opponent's solved strategy over J's moves, aligned with gkeys
            # (node_children returns (keys, nodes) in one order; current_strategy
            # returns probabilities aligned with the keys it's handed).
            y = eng.current_strategy(inf, gkeys)
            for yj, gc in zip(y, gc_nodes, strict=False):
                nf = eng.node_fen(gc)
                if nf is not None:
                    key = self.rules.board_fen(self.rules.board_from_fen(nf))
                    # Sum over histories (see docstring) — y(J) is a reach
                    # probability, not a value.
                    reach[key] = reach.get(key, 0.0) + float(yj)
        return reach

    def _structural_carry_roots(self, budget: int):
        """FOW_STRUCTURAL_CARRY (Phase 1): build the search root set from the
        carried tree's surviving grandchildren instead of a fresh re-sample —
        Obscuro continual re-solving (``Γ̂ ∪ I``). Carried worlds (deduped,
        capped) become roots that REUSE their prior subtree (warm CFR state);
        fresh top-up roots cover the new worlds. Returns
        ``(roots, root_carryover_ids)`` or ``None`` to fall back to the normal
        re-sample (first move / nothing carried). Reads the prior tree, so MUST
        run BEFORE solve_multiroot_rust_tree's reset_for_carryover.

        The belief-membership filter (is a grandchild still in P?) runs in Rust
        via the perspective's post-opp-move observation — every belief world
        agrees on what we see, so a single sampled world yields the visibility
        bitboards. O(|GC|), independent of |P|.
        """
        eng = self._eq_engine
        if (eng is None or self._prev_root_ids is None
                or self._prev_played_action_key is None):
            return None
        from .visibility import visible_squares
        ref = self.enumerator.sample_root_fens(n=1, rng=self.rng)
        if not ref:
            return None
        rb = chess.Board(ref[0])
        persp_white = self.rules.is_first_player(self.perspective)
        vis = int(visible_squares(rb, self.perspective))
        ow = [rb.pieces_mask(r, chess.WHITE) & vis for r in range(1, 7)]
        ob = [rb.pieces_mask(r, chess.BLACK) & vis for r in range(1, 7)]
        carried_ids = eng.build_carryover_roots(
            self._prev_root_ids, self._prev_played_action_key, persp_white,
            vis, ow, ob, budget,
        )
        # Align node ids with their FENs (skip any node with no position).
        carried_fens: list[str] = []
        valid_ids: list[int] = []
        for nid in carried_ids:
            f = eng.node_fen(nid)
            if f is not None:
                carried_fens.append(f)
                valid_ids.append(int(nid))
        if not carried_fens:
            return None  # nothing carried -> normal re-sample
        carried_epds = {_carryover_pos_key(f) for f in carried_fens}
        # Top up to the budget with fresh worlds NOT already carried (so a fresh
        # root never duplicates a carried world -> no duplicate-world double-count).
        n_top = budget - len(carried_fens)
        fresh_fens: list[str] = []
        if n_top > 0:
            pool_n = min(self.enumerator.size, max(n_top * 4, budget))
            for f in self.enumerator.sample_root_fens(n=pool_n, rng=self.rng):
                if len(fresh_fens) >= n_top:
                    break
                if _carryover_pos_key(self.rules.root_fen(chess.Board(f))) not in carried_epds:
                    fresh_fens.append(f)
        ordered = carried_fens + fresh_fens
        # n == len(ordered) -> sample_roots_from_P fills the reservoir in order
        # (no replacement), so roots align 1:1 with `ordered`.
        roots = sample_roots_from_P(
            iter(ordered), to_move=self.perspective, n=len(ordered),
            rng=self.rng, rules=self.rules,
        )
        rcids: list[int | None] = list(valid_ids) + [None] * len(fresh_fens)
        return roots, rcids

    def _commit_material_check(self, move, action_values):
        """Material-catastrophe commit guard (FOW_COMMIT_MATERIAL_GUARD): if the
        committed move hangs material — our king capturable, or the opponent can
        win >= ``cp`` (default 300) by static exchange next ply — in >= ``frac``
        (default 0.12) of belief worlds, switch to the highest-VALUE move whose
        material-hang fraction is below ``frac``.

        Unlike the value-banded king guard above, this OVERRIDES the value band:
        the action value of a hanging move is unreliable (the i-limited solve
        doesn't see the next-ply capture, so it over-rates the active move), and a
        queen/rook hang is worth a passive move to avoid. Fires ONLY on a genuine
        material hang, so normal aggressive play is untouched (no Bd3->Bf1
        over-defense). The gadget already handles ordinary king-safety; this is the
        material catastrophe it misses on uncarried fog worlds."""
        try:
            # KING is terminal -> a far lower bar than MATERIAL. The recurring
            # opening hangs (unseen Qa4+/Qh4) sit in a SINGLE small-belief world
            # (~5-10%); a 12% material bar misses them, so split the thresholds.
            kfrac = float(os.environ.get("FOW_COMMIT_KING_FRAC", "0.05"))
            mfrac = float(os.environ.get("FOW_COMMIT_MATERIAL_FRAC", "0.12"))
            cp = float(os.environ.get("FOW_COMMIT_MATERIAL_CP", "300"))
            # KING-ONLY mode (DORMANT, off by default): would drop the SEE material
            # veto and keep only the king-capture backstop, on the theory that the
            # in-gadget severity boost (B', FOW_GADGET_SEVERITY_BOOST) handles
            # material catastrophes faithfully/sac-aware. TRIED and REJECTED for the
            # `faithful` profile: B' is nondeterministically marginal (hangs the
            # queen on 945dc208 in ~1/3 of solves — lab/bprime_sweep.py), so the
            # FULL material+king guard stays as the deterministic backstop. Flag is
            # retained for the future where deeper per-world search makes the
            # material worlds reliably value-visible and B' can stand alone.
            king_only = os.environ.get("FOW_COMMIT_KING_ONLY") == "1"
            nmax = int(os.environ.get("FOW_COMMIT_RISK_WORLDS", "1000"))
            if not action_values:
                return move
            n = min(self.enumerator.size, nmax)
            if n == 0:
                return move
            from .cfr.gt_cfr import _max_opponent_material_gain as _mmg, _SEE_VAL
            from .cfr.leaf_eval import king_capture_imminent
            import chess as _c
            fens = self.enumerator.sample_root_fens(n=n, rng=random.Random(_GUARD_SEED_MATERIAL))
            worlds = [_c.Board(f) for f in fens]

            _risk_cache: dict[str, tuple[float, bool]] = {}

            def _risk(m):
                """(king_frac, material_unsafe) for m over the belief sample.
                king_frac = fraction of legal-in-belief worlds where m leaves the
                king capturable next ply; material_unsafe = m hangs >= cp by SEE in
                >= mfrac worlds. Memoized — the king-safety scan below calls it
                across many candidate moves."""
                key = m.uci()
                if key in _risk_cache:
                    return _risk_cache[key]
                kf = mf = used = 0
                for w in worlds:
                    if m not in w.pseudo_legal_moves and m not in w.legal_moves:
                        continue
                    used += 1
                    # Value of the piece m CAPTURES in this world (0 if none). The
                    # material check must NET this against the opponent's recapture:
                    # Rxf2 winning a queen then losing the rook is +9-5, NOT a hang.
                    # _max_opponent_material_gain only sees the recapture on the
                    # post-move board, so without this credit it vetoes winning
                    # captures (the engine declined Rxf2/Qxc2, games 56f30e52/57d77bdc).
                    _victim = w.piece_at(m.to_square)
                    _cap_val = _SEE_VAL[_victim.piece_type] if _victim else 0
                    w2 = w.copy()
                    try:
                        w2.push(m)
                    except Exception:
                        pc = w2.piece_at(m.from_square)
                        if pc is None:
                            used -= 1
                            continue
                        w2.remove_piece_at(m.from_square)
                        w2.set_piece_at(m.to_square, pc)
                        w2.turn = not w2.turn
                    if king_capture_imminent(w2, self.perspective) is not None:
                        kf += 1
                    elif (not king_only) and (
                        _mmg(w2, self.perspective, cp) - _cap_val
                    ) >= cp:
                        mf += 1
                r = (1.0, True) if used == 0 else (
                    kf / used, (not king_only) and (mf / used) >= mfrac)
                _risk_cache[key] = r
                return r

            best_v = max(action_values.values())
            gap = float(os.environ.get("FOW_COMMIT_VALUE_GAP", "99"))
            kgap = float(os.environ.get("FOW_COMMIT_KING_GAP", "99"))
            dbg = os.environ.get("FOW_DEBUG_COMMIT") == "1"

            # --- KING-SAFETY (threshold-free, fine resolution) -------------------
            # Among moves within `kgap` of the best gadget value, switch to the one
            # with the LOWEST king-capture risk over the 200-world sample. The i=32
            # gadget can't resolve a sub-3% king-risk (1/32) and the king-death worlds
            # are value-BLIND (the shallow per-world tree misses the next-ply king
            # grab, so neither the gadget nor B' down-weights them) — that let a 3%
            # king-risk d7d5 suicide ship in ~30% of seeds (game 16a78780). A flat
            # threshold can't fix it: kfrac=5% misses 3%, kfrac=2% over-flags until no
            # safe move remains. Choosing the least-king-risky move among VALUE-
            # COMPARABLE ones concedes no value -> bounded over-defense. kgap default
            # 99 = OFF (prior behavior, byte-identical for non-faithful callers).
            if kgap < 90.0:
                cand = [m for m, v in action_values.items() if best_v - v <= kgap]
                kr = {m: _risk(m)[0] for m in cand}
                if kr:
                    min_kr = min(kr.values())
                    if kr.get(move, 0.0) > min_kr + 1e-9:
                        new = max((m for m in cand if kr[m] <= min_kr + 1e-9),
                                  key=lambda m: action_values[m])
                        if dbg:
                            import sys as _s
                            print(f"[COMMIT-KING] {move.uci()}(kr={kr.get(move, 0.0):.3f})"
                                  f" -> {new.uci()}(kr={min_kr:.3f}) among {len(cand)} cand",
                                  file=_s.stderr, flush=True)
                        move = new

            # --- KING hard veto (>= kfrac) + MATERIAL veto (value-gated) ---------
            # Backstops the fine king-safety step: a >= kfrac king-risk is switched
            # HARD (gadget value-blind to king captures — c6c53a42); a material hang
            # is switched only if a material-safe move is within `gap` of its value
            # (the gadget's EV already prices material fog-risk, so double-counting via
            # SEE caused the Qxd5 -> passive over-defense, game 5d413d32; a real
            # catastrophe like 945 has a comparable-value safe move, so it still fires).
            mk, m_unsafe = _risk(move)
            if not (m_unsafe or mk >= kfrac):
                return move  # safe — leave normal play untouched
            safe = [(m, v) for m, v in action_values.items()
                    if not (lambda r: r[1] or r[0] >= kfrac)(_risk(m))]
            if safe:
                best_m, best_v2 = max(safe, key=lambda mv: mv[1])
                flagged_v = action_values.get(move, best_v2)
                do_switch = (mk >= kfrac) or (flagged_v - best_v2 <= gap)
                if dbg:
                    import sys as _s
                    print(
                        f"[COMMIT-MAT] flagged={move.uci()} mk={mk:.3f} "
                        f"v={flagged_v:+.3f} safe={best_m.uci()} v={best_v2:+.3f} "
                        f"switch={do_switch}",
                        file=_s.stderr, flush=True,
                    )
                if do_switch:
                    move = best_m  # best-VALUE safe move
        except Exception:
            pass  # the guard must never break a commit
        return move

    def _catastrophe_prune(self, move, action_values, frac, strategy):
        """Keep the search's MIX, but never commit a move that loses the KING or
        QUEEN/ROOK to capture in >= ``frac`` of belief worlds.

        The clean successor to the king-only / value-gated commit guards (which
        missed the Qxd5/d7d5 class — the king backstop is material-blind, the
        material guard only fires when a safe move is value-comparable, but a fog
        catastrophe is usually rated EV-BEST). Two design choices fix that:

          * HIGH-VALUE ONLY (king + queen/rook, cp>=500) — ordinary pawn/minor risk
            is left alone, so it does NOT over-defend on normal moves.
          * NOT value-gated — a queen/king hang is never worth its expected value,
            so we redirect regardless of how good the move looks on average.

        Crucially it preserves mixing: if the search's own pick is already safe (the
        common case), it is returned UNCHANGED. Only a high-value-catastrophe pick is
        redirected — in a sampling regime, re-sampled among the safe moves by their
        strategy weight; otherwise the highest-value safe move. If every move is
        catastrophic (cornered), nothing is changed."""
        try:
            from .cfr.gt_cfr import _max_opponent_material_gain as _mmg, _SEE_VAL
            from .cfr.leaf_eval import king_capture_imminent
            import chess as _c
            nmax = int(os.environ.get("FOW_HV_PRUNE_WORLDS", "1000"))
            n = min(self.enumerator.size, nmax)
            if n == 0:
                return move
            fens = self.enumerator.sample_root_fens(n=n, rng=random.Random(_GUARD_SEED_PRUNE))
            worlds = [_c.Board(f) for f in fens]
            cp = float(os.environ.get("FOW_HV_PRUNE_CP", "500"))  # Q/R-level loss
            # King is handled threshold-FREE (a 3% king-suicide is below any flat bar
            # yet above the safe moves' ~0%; a flat bar over-prunes the opening's
            # incidental king-risk). king_gap = the value-band the relative step
            # compares king-risk within.
            king_gap = float(os.environ.get("FOW_HV_PRUNE_KING_GAP", "0.2"))
            # NET-HANG floor (2026-08-21, prod game 42b652b6 move 24): the legacy
            # hang test nets the victim credit against the recapture
            # (loss = gross - cap_val >= cp), so queen-takes-DEFENDED-rook nets
            # 400cp and slips UNDER the 500cp floor — the prune scored the
            # game-losing Qxe8 at 0.0cp risk, and worse, its STEP-1 redirect moved
            # the commit BACK onto Qxe8 after _commit_material_check had correctly
            # switched away (the guard saves, the prune un-saves; the prune runs
            # last by design so its blind spot wins). With the floor set (>0):
            # ALSO flag a move when a policed piece is at stake (gross >= cp) and
            # the net SEE deficit is >= net_floor. 300 mirrors the material
            # guard's own FOW_COMMIT_MATERIAL_CP bar. Winning captures stay safe
            # (RxQ recaptured nets -400 < floor); exchange sacs stay playable
            # (R-for-minor nets ~170 < 300). 0 = OFF, byte-identical legacy.
            net_floor = float(os.environ.get("FOW_HV_PRUNE_NET_FLOOR", "0"))
            cache: dict[str, tuple[float, float, float]] = {}

            def hv_risk(m):
                key = m.uci()
                if key in cache:
                    return cache[key]
                kf = mf = used = 0
                em = 0.0  # expected centipawn loss (severity), >= cp hangs only
                for w in worlds:
                    if m not in w.pseudo_legal_moves and m not in w.legal_moves:
                        continue
                    used += 1
                    victim = w.piece_at(m.to_square)
                    cap_val = _SEE_VAL[victim.piece_type] if victim else 0
                    w2 = w.copy()
                    try:
                        w2.push(m)
                    except Exception:
                        pc = w2.piece_at(m.from_square)
                        if pc is None:
                            used -= 1
                            continue
                        w2.remove_piece_at(m.from_square)
                        w2.set_piece_at(m.to_square, pc)
                        w2.turn = not w2.turn
                    if king_capture_imminent(w2, self.perspective) is not None:
                        kf += 1
                    else:
                        gross = _mmg(w2, self.perspective, cp)
                        loss = gross - cap_val
                        if loss >= cp or (
                            net_floor > 0 and gross >= cp and loss >= net_floor
                        ):
                            mf += 1
                            em += loss
                r = (kf / used, mf / used, em / used) if used else (1.0, 1.0, 1.0e4)
                cache[key] = r
                return r

            def _resample(cands):
                """Keep the mix: re-sample among cands by their strategy weight
                (sampling regime), else the highest-value candidate."""
                if len(strategy) > 1:
                    wts = [max(strategy.get(m, 0.0), 0.0) for m in cands]
                    tot = sum(wts)
                    if tot > 0:
                        r = self.rng.random() * tot
                        cum = 0.0
                        for m, wt in zip(cands, wts, strict=False):
                            cum += wt
                            if r < cum:
                                return m
                return max(cands, key=lambda m: action_values.get(m, -2.0))

            # STEP 1 — MATERIAL. Default: flat hard veto on hang-FRACTION (>= frac).
            # Adaptive (FOW_HV_PRUNE_ADAPTIVE=1): veto on severity-weighted EXPECTED
            # loss (prob x material centipawns) against a threshold that LOOSENS with
            # |P| — tight when belief is sharp (opening/endgame, where the catastrophes
            # cluster), loose in heavy fog (large |P|) where every move hangs in some
            # worlds and a flat bar would paralyze. NOT value-gated: the opening hangs
            # are +EV (a 3% queen loss barely dents EV), so only a hard veto on
            # prob x severity catches them. Two measured signals (severity, |P|); one
            # base scalar (tau). Default OFF -> static path, byte-identical.
            _adaptive = (self._hv_prune_adaptive if self._hv_prune_adaptive is not None
                         else os.environ.get("FOW_HV_PRUNE_ADAPTIVE") == "1")
            if _adaptive:
                tau = (self._hv_prune_tau if self._hv_prune_tau is not None
                       else float(os.environ.get("FOW_HV_PRUNE_TAU", "20")))     # base cp floor
                pref = (self._hv_prune_pref if self._hv_prune_pref is not None
                        else float(os.environ.get("FOW_HV_PRUNE_PREF", "100")))  # |P| reference
                pmax = (self._hv_prune_pmax if self._hv_prune_pmax is not None
                        else float(os.environ.get("FOW_HV_PRUNE_PMAX", "8")))    # max loosening
                tau_eff = tau * min(max(self.enumerator.size / pref, 1.0), pmax)
                mat_safe = [m for m in action_values if hv_risk(m)[2] < tau_eff]
                if mat_safe and hv_risk(move)[2] >= tau_eff:
                    move = _resample(mat_safe)
            else:
                mat_safe = [m for m in action_values if hv_risk(m)[1] < frac]
                if mat_safe and hv_risk(move)[1] >= frac:
                    move = _resample(mat_safe)
            pool = mat_safe or list(action_values)
            # STEP 2 — KING. Redirect to the lowest-king-risk value-comparable move ONLY
            # if the picked move's king-risk exceeds the FLOOR. The legacy threshold-free
            # step (floor=0) shaved a negligible 1.2% king-risk for a materially-worse move
            # (exd5 -> Qxd5, game 81fa6bda, a LOSS). Above the floor (the ~3% d7d5 suicide)
            # the validated threshold-free redirect still fires. floor=0 = byte-identical.
            king_floor = (self._hv_prune_king_floor if self._hv_prune_king_floor is not None
                          else float(os.environ.get("FOW_HV_PRUNE_KING_FLOOR", "0")))
            best_v = max(action_values.values())
            band = [m for m in pool if best_v - action_values.get(m, -2.0) <= king_gap]
            if band and hv_risk(move)[0] > king_floor:
                min_kr = min(hv_risk(m)[0] for m in band)
                if hv_risk(move)[0] > min_kr + 1e-9:
                    move = _resample([m for m in band if hv_risk(m)[0] <= min_kr + 1e-9])
            return move
        except Exception:
            return move  # never break a commit

    def choose_move(
        self,
        *,
        iterations: int = _DEFAULT_ITERATIONS,
        time_budget_seconds: float | None = None,
        i_sample_size: int = _DEFAULT_I_SAMPLE_SIZE,
        max_actions: int = _DEFAULT_MAX_ACTIONS,
        kluss_k: int | None = None,
        use_rust_eq: bool = False,
        use_rust_tree: bool = True,
        legal_moves: Iterable[Any] | None = None,
    ) -> chess.Move:
        """Pick a move using multi-root GT-CFR + purification.

        Pipeline — a 5-stage sequence; each stage has a matching
        ``# === Stage N`` marker in the body below:
          1. Sample |I| belief roots from P.
          2. Carryover: reuse the prior search's surviving subtrees
             (structural-carry / EPD-matched discovery; default off; the
             discovery step is nested in the rust-tree branch of stage 3).
          3. Search: multi-root GT-CFR (``solve_multiroot_rust_tree``),
             with the opening-phase gadget switch.
          4. Extract: purify + A6.2 regime select -> one move.
          5. Commit guards: risk / material / royal / mini / win-fast /
             catastrophe-prune vetoes (env- or profile-gated). The prune
             runs LAST by design (see its marker).

        Behavior is pinned by tests/test_pick_move_golden_trace.py (fixed
        iters, no time budget) — keep it green across edits here.

        Args:
            iterations: equilibrium passes (only an upper bound when
                time_budget_seconds is set — anytime cuts off earlier).
            time_budget_seconds: if set, stops as soon as wall time
                exceeds budget. Real-play default for live games.
            i_sample_size: |I| roots to sample from P. Smaller = faster
                per iter, less belief coverage.
            max_actions: support size after purification. 1 = Resolve
                regime (deterministic top); ≤3 = Maxmargin regime
                (mixing). Defaults to 1 until A6.2 ships Maxmargin.
            legal_moves: optional externally known true-legal move set used as
                a final commit mask. Existing callers that omit it keep the
                historical behavior.

        Returns:
            One chess.Move to play. Always non-None.

        Raises:
            RuntimeError: if P is empty (shouldn't happen — would
                indicate a soundness violation upstream).
        """
        # === Stage 1: sample |I| belief roots from P ===
        if self.enumerator.size == 0:
            raise RuntimeError("P is empty; cannot choose a move")
        if self.enumerator.uses_rust_state:
            # Resident-Rust P: sample |I| root indices and decode only those —
            # the whole point of keeping P packed in Rust. sample_roots_from_P
            # over the pre-sampled FENs just builds the root nodes (no further
            # rng draws since the stream length <= n).
            fens = self.enumerator.sample_root_fens(n=i_sample_size, rng=self.rng)
            roots = sample_roots_from_P(
                iter(fens), to_move=self.perspective, n=i_sample_size, rng=self.rng,
                rules=self.rules,
            )
        else:
            roots = sample_roots_from_P(
                self.enumerator.iter_positions(),
                to_move=self.perspective,
                n=i_sample_size,
                rng=self.rng,
                rules=self.rules,
            )
        if not roots:
            raise RuntimeError("sample_roots_from_P returned 0 roots")
        if (
            self._variant_hooks is not None
            and not getattr(
                self._variant_hooks, "standalone_rust_eq_compatible", True
            )
            and not use_rust_tree
        ):
            # The variant's Python tree keys aren't chess-shaped, so the
            # standalone Rust-equilibrium mirror can't consume them (full
            # Xiangqi's native path is the full Rust tree).
            use_rust_eq = False

        # === Stage 2: carryover — reuse the prior search's surviving subtrees
        # (default off; the rust-tree discovery half lives inside Stage 3) ===
        # FOW_STRUCTURAL_CARRY (Phase 1): replace the fresh re-sample with the
        # carried tree's surviving worlds + fresh top-up (continual re-solving).
        # Default OFF -> skipped -> byte-identical. Only on the Rust-tree path
        # with carryover-subtree on and a resident-Rust belief.
        # Carryover |P|-gate: at small |P| a fresh re-sample IS the complete exact
        # belief; the carried subset would corrupt it. Below the threshold, force the
        # fresh full solve (no structural-carry, no subtree warm-start). 0 = off.
        _carry_min = (self._carryover_min_p if self._carryover_min_p is not None
                      else int(os.environ.get("FOW_CARRYOVER_MIN_P", "0")))
        _carry_ok = self.enumerator.size > _carry_min

        structural_root_ids: list[int | None] | None = None
        if (_carry_ok and use_rust_tree and self.carryover_subtree
                and self.enumerator.uses_rust_state
                and self.structural_carry):
            _sc = self._structural_carry_roots(i_sample_size)
            if _sc is not None:
                roots, structural_root_ids = _sc

        # === Stage 3: search — multi-root GT-CFR (carryover discovery + gadget
        # phase-switch nested here, then the solve call) ===
        if use_rust_tree:
            # WS2 (DEFAULT since 2026-05-27): the authoritative Rust tree drives
            # the whole loop (select / expand / eq / seed in Rust; Stockfish at the
            # FFI boundary). Reuses the same sampled root boards. Strategy is
            # byte-identical to the Python path (test_ws2_full_loop_equiv); ~1.9x
            # iters/move. KLUSS is now ported to the Rust tree (lib.rs
            # set_kluss_keep_from + filtered select_leaf), so kluss_k threads
            # through here instead of falling back to the Python path.
            # Lever 5: skip per-iter strategy_history snapshots when only the
            # final strategy will be read (Resolve regime, max_actions=1).
            # Auto-mode (max_actions=0 → A6.2 select_regime) might pick
            # Maxmargin (>1) post-hoc; keep history for that case.
            need_history = max_actions != 1
            # Lever 1 Phase 1: lazily construct + reuse one EqEngine instance
            # across choose_move calls. solve_multiroot_rust_tree calls
            # reset_tree() to clear state at the start of each call, so
            # behavior matches the prior per-call construction (byte-parity).
            if self._eq_engine is None:
                import fow_rust as _fow_rust
                # Placeholder seed; reset_tree re-seeds per call from self.rng.
                self._eq_engine = _fow_rust.EqEngine([0] * 624, 0)
            # Phase 2a: discover carryover candidates from prior search.
            # Discovery walks the prior tree BEFORE reset_for_carryover —
            # the reset only clears KLUSS, not nodes, so ids stay valid.
            root_carryover_ids: list[int | None] | None = None
            if structural_root_ids is not None:
                # Structural carry (Γ̂ ∪ I) already selected the carried root
                # node ids in roots-order; skip the re-sample-and-match discovery.
                root_carryover_ids = structural_root_ids
                if isinstance(self.gadget_blueprint, CarryoverBlueprint):
                    self.gadget_blueprint.set_values(
                        self._carryover_blueprint_values()
                    )
                    if self.gadget_alpha:
                        self.gadget_blueprint.set_reach(
                            self._carryover_blueprint_reach()
                        )
            elif (
                _carry_ok
                and self.carryover_subtree
                and self._prev_root_ids is not None
                and self._prev_played_action_key is not None
            ):
                pairs = self._eq_engine.discover_carryover_candidates(
                    self._prev_root_ids, self._prev_played_action_key,
                )
                # Match on EPD (pos identity), not piece-only board_fen vs the
                # full-FEN keys — that mismatch made carryover a silent no-op.
                fen_to_node: dict[str, int] = {
                    _carryover_pos_key(f): nid for f, nid in pairs
                }
                root_carryover_ids = [
                    fen_to_node.get(_carryover_pos_key(self.rules.root_fen(r.truth)))
                    for r in roots
                ]
                # Dedup: a carried node is ONE subtree → it can back at most one
                # root. EPD keys are NOT injective on the belief (two worlds that
                # differ only in move counters share an EPD), so distinct roots can
                # map to the same carried node → duplicate root_ids → the
                # double-counted-regrets invariant fires (the real bakeoff crash).
                # First claimant keeps the subtree; later collisions get a fresh
                # root (a leaf subtree carries nothing anyway, and counters don't
                # affect the game tree, so reusing it for one of the twins is fine).
                _seen: set[int] = set()
                _deduped: list[int | None] = []
                for _nid in root_carryover_ids:
                    if _nid is None or _nid in _seen:
                        _deduped.append(None)
                    else:
                        _seen.add(_nid)
                        _deduped.append(_nid)
                root_carryover_ids = _deduped
                # Phase 2 Slice 0: if the gadget runs the continual-resolve
                # blueprint, read the previous move's solved u(x,y|J) off the
                # still-intact prior tree (same condition + timing as discovery
                # above: BEFORE solve_multiroot_rust_tree's reset, so move N's
                # CFR state is pristine) and hand it to the blueprint for this
                # move. No-op for every other blueprint type.
                if isinstance(self.gadget_blueprint, CarryoverBlueprint):
                    self.gadget_blueprint.set_values(
                        self._carryover_blueprint_values()
                    )
                    if self.gadget_alpha:
                        self.gadget_blueprint.set_reach(
                            self._carryover_blueprint_reach()
                        )
            # FOW_V2_EXPANSION_BUDGET: decouple tree growth from iteration count.
            # The default (None) expands one leaf per iteration, so the tree —
            # and the eq pass that walks ALL of it every iteration — grows
            # linearly with iters (total solve cost quadratic; measured: tree
            # 20K→157K nodes over an 8K-iter solve, eq_pass 77-81% of move
            # time). Obscuro decouples these (CFR thread free-runs; expansion
            # threads grow the tree at their own wall-bounded rate). A fixed
            # budget caps the walked-tree size -> constant per-iter cost ->
            # several-fold more iterations per wall-second (the alpha safe-flip
            # convergence the probes showed we lack). 0/unset = None = prior
            # behavior, byte-identical.
            _eb = (self._expansion_budget if self._expansion_budget is not None
                   else int(os.environ.get("FOW_V2_EXPANSION_BUDGET", "0")) or None)
            _eb = _eb or None
            # --- PHASE switch: gadget in the opening, gadget-off (EV-max) after ---
            # The gadget's worst-case robustness wins in the OPENING (high fog, can't
            # calculate — the first ~N plies are uniformly uncertain across games,
            # lab/uncertainty_trajectory.py); once the position is concrete (mid/end)
            # EV-max wins and the gadget's hedging reads as passivity (Brian's games
            # 83e36e03/372110f1). So engage the gadget only while ply <= N; after that
            # run plain GT-CFR. The commit guards (king-safety, material) stay on every
            # move regardless. FOW_GADGET_OPENING_PLIES=0 = OFF (gadget all game).
            _open_plies = int(os.environ.get("FOW_GADGET_OPENING_PLIES", "0"))
            _cur_ply = getattr(self, "_observed_plies", 0) + 1
            _use_gadget = self.resolve_gadget and (
                _open_plies <= 0 or _cur_ply <= _open_plies)
            if os.environ.get("FOW_DEBUG_VERBOSE") == "1" and _open_plies > 0:
                print(f"[FOW_PHASE] ply~{_cur_ply} gadget={'ON' if _use_gadget else 'OFF'}"
                      f" (opening<={_open_plies})", file=__import__("sys").stderr, flush=True)
            solution = solve_multiroot_rust_tree(
                [r.truth for r in roots],
                stockfish_eval=self._stockfish,
                perspective=self.perspective,
                expansion_budget=_eb,
                iterations=iterations,
                rng=self.rng,
                time_budget_seconds=time_budget_seconds,
                kluss_k=kluss_k,
                kluss_soft=self._kluss_soft,
                record_strategy_history=need_history,
                eq_engine=self._eq_engine,
                carryover_infosets=self.carryover_infosets,
                carryover_subtree=self.carryover_subtree,
                root_carryover_ids=root_carryover_ids,
                full_cfv_backprop=self.full_cfv_backprop,
                resolve_gadget=_use_gadget,
                gadget_blueprint=self.gadget_blueprint,
                gadget_margin=self.gadget_margin,
                gadget_cvar_q=self.gadget_cvar_q,
                gadget_faithful=self.gadget_faithful,
                gadget_alpha=self.gadget_alpha,
                gadget_iterative=(self.gadget_iterative if _use_gadget else None),
                rules=self.rules,
            )
            # Phase 2a: record this search's root_ids for the NEXT pick's
            # discovery walk. solution.root_ids holds the rust-tree node ids
            # used by the just-completed search (mix of fresh + carryover).
            self._prev_root_ids = list(solution.root_ids) if solution.root_ids else None
        else:
            solution = solve_multiroot_growing_subgame(
                roots,
                stockfish_eval=self._stockfish,
                perspective=self.perspective,
                iterations=iterations,
                rng=self.rng,
                time_budget_seconds=time_budget_seconds,
                kluss_k=kluss_k,
                use_rust_eq=use_rust_eq,
            )
        self.last_solution = solution
        # Chess truths are chess.Board; variant boards don't expose .fen(), so
        # the truth-in-I telemetry is chess-only for now.
        if roots and hasattr(roots[0].truth, "fen"):
            self.last_root_fens = [r.truth.fen() for r in roots]
        else:
            self.last_root_fens = None
        # C.1 fallback's v* term: record OUR-POV root value of this solve for
        # the NEXT move's uncarried-world alternate values. Only on the
        # iterative-gadget (or gadget-off) readout — the READ-ONLY gadget's
        # value_at_root is a blueprint-relative MARGIN, not an absolute value,
        # and would corrupt v*. (No-op for other blueprint types / flag off.)
        if isinstance(self.gadget_blueprint, CarryoverBlueprint) and (
            not self.resolve_gadget or self.gadget_iterative
        ):
            self.gadget_blueprint.set_prev_value(solution.value_at_root)

        if not solution.strategy_at_root:
            raise RuntimeError("GT-CFR returned empty strategy at root")

        # --- diagnostics (FOW_DEBUG_VERBOSE; pure stderr, no effect on the move) ---
        # FOW_DEBUG_VERBOSE: per-move solution diagnostic. Top-K strategy +
        # action values + KLUSS-relevant metadata to stderr. Off by default;
        # gate kept narrow because a full game prints ~100 such blocks.
        if os.environ.get("FOW_DEBUG_VERBOSE") == "1":
            import sys as _sys
            self._dbg_move_count = getattr(self, "_dbg_move_count", 0) + 1
            persp = "white" if self.rules.is_first_player(self.perspective) else "black"
            sorted_actions = sorted(
                solution.strategy_at_root.items(), key=lambda kv: -kv[1]
            )
            print(
                f"[FOW_DBG] move#{self._dbg_move_count} {persp}: "
                f"iters={solution.iterations} nodes={solution.total_tree_nodes} "
                f"roots={solution.n_roots} elapsed={solution.elapsed_seconds:.2f}s "
                f"n_actions={len(sorted_actions)}",
                file=_sys.stderr,
                flush=True,
            )
            # Component-time breakdown (Step 0 of the efficiency campaign).
            # Empty when the Python-tree path was used; populated by
            # solve_multiroot_rust_tree (the WS2 default).
            cm = solution.component_ms
            if cm:
                total_ms = solution.elapsed_seconds * 1000.0
                sf_total = cm.get("sf_eval", 0.0) + cm.get("sf_children", 0.0)
                non_sf = (
                    cm.get("eq_pass", 0.0)
                    + cm.get("select_leaf", 0.0)
                    + cm.get("kluss", 0.0)
                    + cm.get("expand_seed", 0.0)
                    - sf_total
                )
                # expand_seed contains the Stockfish call (via _rust_expand_and_seed),
                # so the non_sf above is the rust-tree orchestration cost
                # excluding Stockfish. May go slightly negative on tiny budgets.
                def _pct(x: float) -> str:
                    return f"{(x / total_ms * 100):.1f}%" if total_ms > 0 else "—"
                print(
                    f"[FOW_DBG]   time: total={total_ms:.0f}ms "
                    f"sf_eval={cm['sf_eval']:.0f}ms({_pct(cm['sf_eval'])}) "
                    f"sf_children={cm['sf_children']:.0f}ms({_pct(cm['sf_children'])}) "
                    f"sf_total={sf_total:.0f}ms({_pct(sf_total)})",
                    file=_sys.stderr,
                    flush=True,
                )
                print(
                    f"[FOW_DBG]   time: eq={cm['eq_pass']:.0f}ms({_pct(cm['eq_pass'])}) "
                    f"select={cm['select_leaf']:.0f}ms({_pct(cm['select_leaf'])}) "
                    f"kluss={cm['kluss']:.0f}ms({_pct(cm['kluss'])}) "
                    f"expand={cm['expand_seed']:.0f}ms({_pct(cm['expand_seed'])}) "
                    f"non_sf_rust≈{max(0, non_sf):.0f}ms",
                    file=_sys.stderr,
                    flush=True,
                )
            for mv, p in sorted_actions[:12]:
                v = solution.action_values_at_root.get(mv, float("nan"))
                print(
                    f"[FOW_DBG]   {mv.uci():>6}  p={p:.4f}  v={v:+.4f}",
                    file=_sys.stderr,
                    flush=True,
                )

        # === Stage 4: extract — purify + A6.2 regime select -> one move ===
        # A6.2 regime selection: if caller passed max_actions=0 (the
        # explicit "auto" sentinel), derive it from the action-value
        # margin (Resolve regime = top-1, Maxmargin regime = top-≤3).
        # Any positive integer overrides — preserves old "force
        # max_actions=1" callsites and lets the bakeoff harness pin a
        # regime for A/B testing.
        effective_max_actions = max_actions
        if max_actions == 0:
            effective_max_actions = select_regime(solution.action_values_at_root)
        self.last_regime_auto = (max_actions == 0)
        purified = purify_strategy(
            solution.strategy_at_root,
            solution.strategy_history_at_root,
            solution.t_half,
            max_actions=effective_max_actions,
        )
        self.last_purified = purified

        if effective_max_actions == 1:
            # Resolve regime: pick the single top action.
            move = next(iter(purified.strategy.keys()))
        else:
            # Maxmargin regime: sample from the purified mix.
            actions = list(purified.strategy.keys())
            probs = list(purified.strategy.values())
            r = self.rng.random()
            cum = 0.0
            move = actions[-1]
            for a, p in zip(actions, probs, strict=False):
                cum += p
                if r < cum:
                    move = a
                    break

        # === Stage 5: commit guards — material / royal / variant hooks /
        # win-fast, then the catastrophe-prune veto LAST (ordering is
        # load-bearing) ===
        if (os.environ.get("FOW_COMMIT_MATERIAL_GUARD") == "1"
                and solution.action_values_at_root):
            move = self._commit_material_check(move, solution.action_values_at_root)
        # Variant commit guards — the royal guard, material checks, and vetoes
        # live in the variant packages' engine_hooks (backed by
        # variants_common.royal_guard); chess registers none.
        if self._variant_hooks is not None and hasattr(
            self._variant_hooks, "commit_guards"
        ):
            move = self._variant_hooks.commit_guards(
                self, move, solution.action_values_at_root
            )
        if self.win_fast:
            move = self._prefer_immediate_royal_capture(
                move, roots, solution.action_values_at_root)
        # Catastrophe prune (FOW_HV_PRUNE_FRAC / adaptive; default off): the FINAL
        # commit veto — never ship a KING/QUEEN/ROOK hang. Runs LAST, after the
        # commit guards, because the king-safety step in _commit_material_check is
        # material-BLIND: it switches to the highest-value king-safe move, which can
        # re-introduce a queen-hang the prune just avoided (prodQ 1281103c@10: prune
        # picks safe c6b4 → king step swaps back to d8d5, the +0.6 equalizer that
        # hangs the queen to Nc3). Last word = no later guard can undo the avoidance.
        # win_fast only ever picks a winning royal capture (hv_risk ~0), so ordering
        # it before the prune is safe.
        _hv_frac = (self._hv_prune_frac if self._hv_prune_frac is not None
                    else float(os.environ.get("FOW_HV_PRUNE_FRAC", "0")))
        _hv_adaptive = (self._hv_prune_adaptive if self._hv_prune_adaptive is not None
                        else os.environ.get("FOW_HV_PRUNE_ADAPTIVE") == "1")
        if (_hv_frac > 0.0 or _hv_adaptive) and solution.action_values_at_root:
            move = self._catastrophe_prune(
                move, solution.action_values_at_root, _hv_frac, purified.strategy)
        if self.queen_promo_tiebreak:
            move = self.rules.normalize_committed_move(move)
        move = _apply_legal_move_mask(
            self.rules,
            move,
            legal_moves,
            solution.action_values_at_root,
            purified.strategy,
        )
        self.moves_chosen += 1
        return move

    def _prefer_immediate_royal_capture(self, move, roots, action_values):
        """Win-fast tiebreak (gated by self.win_fast). If the enemy royal's
        square is CONFIDENTLY known (identical across every sampled root) and a
        max-value root action captures it, return that immediate capture instead
        of an equal-value dawdle. Confidence guard = don't gamble on an unseen
        royal; max-value guard = never trade down for the capture."""
        if not action_values or not roots:
            return move
        opp = self.rules.opponent(self.perspective)
        royal_sqs = {self.rules.royal_square(r.truth, opp) for r in roots}
        if len(royal_sqs) != 1:
            return move  # uncertain where the enemy royal is — no gamble
        target = next(iter(royal_sqs))
        if target is None:
            return move
        best = max(action_values.values())
        captures = [m for m, v in action_values.items()
                    if m.to_square == target and v >= best - 1e-9]
        return captures[0] if captures else move

    def close(self) -> None:
        """Release the Stockfish subprocess if this engine owns it."""
        if self._owns_stockfish:
            self._stockfish.close()

    def __enter__(self) -> "EngineV2":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Strategy-protocol adapter (selfplay.Strategy compatible)
# ---------------------------------------------------------------------------


class EngineV2Strategy:
    """Adapter making EngineV2 conform to ``selfplay.Strategy`` protocol.

    Lets v2 play through the existing ``selfplay.play_game`` harness
    against any other Strategy (Tier-1 v0.9.5 baseline, random, etc.).
    Constructs the EngineV2 lazily on reset() so the same instance can
    be reused across games (close() between resets).

    Args:
        seed: RNG seed (separate from Stockfish PRNG).
        iterations: GT-CFR equilibrium passes per move (upper bound
            when time_budget_seconds is set).
        i_sample_size: |I| roots sampled from P per move.
        time_budget_seconds: optional per-move wall budget.
        p_max_size: cap on PEnumerator |P| (None for unbounded).
        max_actions: purification regime (1 = Resolve, ≤3 = Maxmargin).
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        iterations: int = 200,
        i_sample_size: int = 8,
        time_budget_seconds: float | None = None,
        p_max_size: int | None = _DEFAULT_P_MAX_SIZE,
        max_actions: int = _DEFAULT_MAX_ACTIONS,
        capture_telemetry: bool = False,
        kluss_k: int | None = None,
        use_rust_eq: bool = False,
        use_rust_state: bool = True,
        use_rust_tree: bool = True,
        resolve_gadget: bool | None = None,
        resolve_cvar_q: float | None = None,
        use_lean: bool | None = None,
        queen_promo_tiebreak: bool | None = None,
        opening_book: bool | None = None,
        gadget_faithful: bool | None = None,
        gadget_alpha: bool | None = None,
        gadget_iterative: bool | None = None,
        structural_carry: bool | None = None,
        carryover_subtree: bool | None = None,
        resolve_blueprint: str | None = None,
        expansion_budget: int | None = None,
        kluss_soft: bool | None = None,
        hv_prune_frac: float | None = None,
        hv_prune_adaptive: bool | None = None,
        hv_prune_tau: float | None = None,
        hv_prune_pref: float | None = None,
        hv_prune_pmax: float | None = None,
        hv_prune_king_floor: float | None = None,
        carryover_min_p: int | None = None,
    ) -> None:
        self._seed = seed
        self._expansion_budget = expansion_budget
        self._kluss_soft = kluss_soft
        # Per-arm catastrophe-prune knobs forwarded to EngineV2 (None = env).
        self._hv_prune_frac = hv_prune_frac
        self._hv_prune_adaptive = hv_prune_adaptive
        self._hv_prune_tau = hv_prune_tau
        self._hv_prune_pref = hv_prune_pref
        self._hv_prune_pmax = hv_prune_pmax
        self._hv_prune_king_floor = hv_prune_king_floor
        self._carryover_min_p = carryover_min_p
        self._iterations = iterations
        self._i_sample_size = i_sample_size
        self._time_budget = time_budget_seconds
        # Clock-aware time management (core engine responsibility). When ON
        # (FOW_V2_CLOCK_TIME=1) AND the PerspectiveView carries a clock,
        # pick_move budgets per-move from the remaining clock (solvency) instead
        # of the static time_budget. Default OFF → uses _time_budget as before,
        # so bakeoffs/tests at fixed budget are unchanged. Same code governs
        # selfplay (bakeoff) and the live worker — see cfr/time_manager.py.
        self._clock_time_enabled = os.environ.get("FOW_V2_CLOCK_TIME") == "1"
        self._time_manager = TimeManager.from_env()
        self._p_max_size = p_max_size
        self._max_actions = max_actions
        self._kluss_k = kluss_k
        self._use_rust_eq = use_rust_eq
        self._use_rust_state = use_rust_state
        self._use_rust_tree = use_rust_tree
        # Per-arm gadget override (None = read the process env). Lets a v2-vs-v2
        # bakeoff enable the gadget on one arm only.
        self._resolve_gadget = resolve_gadget
        self._resolve_cvar_q = resolve_cvar_q
        # Per-arm lean-UCI override (None = read FOW_LEAN_UCI). Byte-identical to
        # the python-chess path, so a v2-vs-v2 bakeoff isolates the throughput
        # gain (more iters/move at the same budget), not a strength change.
        self._use_lean = use_lean
        # Per-arm queen-promotion tiebreak override (None = read the env).
        self._queen_promo_tiebreak = queen_promo_tiebreak
        # Per-arm continual-resolve stack overrides (None = read the FOW_* env).
        # Lets a v2-vs-v2 bakeoff put the faithful gadget / non-uniform alpha /
        # structural carry / carryover blueprint on ONE arm despite a shared
        # process env. Forwarded straight to EngineV2 (same None=env semantics).
        self._gadget_faithful = gadget_faithful
        self._gadget_alpha = gadget_alpha
        self._gadget_iterative = gadget_iterative
        self._structural_carry = structural_carry
        self._carryover_subtree = carryover_subtree
        self._resolve_blueprint = resolve_blueprint
        # Move-policy layer (move_policy.py): an ordered list of deterministic
        # overrides applied around the search. The opening book is policy #1;
        # future strength fixes (e.g. the post-launch S1 catastrophe filter)
        # register here behind their OWN flags. Each policy is independently
        # gated; the layer is inert (a no-op loop over []) when none are enabled,
        # so a bare strategy stays byte-identical until opted in.
        #
        # FOW_OPENING_BOOK (or the opening_book kwarg; None = read the env, default
        # off) enables the observation-keyed opening book. load() returns None if
        # the bundled data file is absent -> the book is silently skipped.
        if opening_book is None:
            opening_book = os.environ.get("FOW_OPENING_BOOK") == "1"
        self._policies: list = []
        if opening_book:
            book = _load_opening_book()
            if book is not None:
                self._policies.append(book)
        # Name of the policy that overrode the last pick (e.g. "opening_book:pre"),
        # else None. Introspection hook for telemetry/debugging; reset per pick.
        self.last_policy_action: str | None = None
        self._engine: EngineV2 | None = None
        self.perspective: chess.Color | None = None
        # Per-ply telemetry. Append one row per observe_* / pick_move call.
        # Off by default — long offline runs (200+ games) want it on so
        # post-mortem analysis on |P| trajectory + per-ply wall is possible.
        self._capture_telemetry = capture_telemetry
        self.telemetry: list[dict] = []
        self._ply_seen = 0
        # Optional LIVE per-ply telemetry sink (set via set_telemetry_sink). When
        # open, each telemetry row is written + flushed AS IT IS PRODUCED, so a
        # SIGKILL (OOM / per-game timeout) — which never runs the end-of-game write
        # or the except handler — still leaves the |P| trajectory up to the crash on
        # disk. The whole point of the crux measurement: the games that explode |P|
        # are exactly the ones that get killed before they can report. Default None →
        # no file I/O, byte-identical to every caller that doesn't opt in.
        self._telemetry_fh = None
        self._observation_history_events: list[str] = []

    def reset(self, perspective: chess.Color, game_id: str | None = None) -> None:
        # Close any prior engine to release Stockfish + reset state.
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
        self.perspective = perspective
        # Per-game seed: salt the base seed with the game id so EV-margin mixing
        # produces a DIFFERENT line each game (opening-replay variety). Without
        # this, reset() always re-seeds to the constant self._seed → identical
        # opening every game. game_id=None (bakeoffs, tests) → bare self._seed,
        # so same-seed reproducibility guards are unaffected.
        if game_id:
            import hashlib

            salt = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
            engine_seed = (self._seed ^ salt) & 0xFFFFFFFF
        else:
            engine_seed = self._seed
        self._engine = EngineV2(
            perspective,
            rng=random.Random(engine_seed),
            p_max_size=self._p_max_size,
            use_rust_state=self._use_rust_state,
            resolve_gadget=self._resolve_gadget,
            resolve_cvar_q=self._resolve_cvar_q,
            use_lean=self._use_lean,
            queen_promo_tiebreak=self._queen_promo_tiebreak,
            gadget_faithful=self._gadget_faithful,
            gadget_alpha=self._gadget_alpha,
            gadget_iterative=self._gadget_iterative,
            expansion_budget=self._expansion_budget,
            kluss_soft=self._kluss_soft,
            structural_carry=self._structural_carry,
            carryover_subtree=self._carryover_subtree,
            resolve_blueprint=self._resolve_blueprint,
            hv_prune_frac=self._hv_prune_frac,
            hv_prune_adaptive=self._hv_prune_adaptive,
            hv_prune_tau=self._hv_prune_tau,
            hv_prune_pref=self._hv_prune_pref,
            hv_prune_pmax=self._hv_prune_pmax,
            hv_prune_king_floor=self._hv_prune_king_floor,
            carryover_min_p=self._carryover_min_p,
        )
        self.telemetry = []
        self._ply_seen = 0
        self._observation_history_events = []

    def set_telemetry_sink(self, path: str | None) -> None:
        """Open a LIVE per-ply telemetry file (truncating) so rows are flushed as
        they happen — survives an OOM/timeout SIGKILL that skips the end-of-game
        write. Call once per game (before play), after construction; survives the
        per-game ``reset`` (which clears the in-memory list but not this handle).
        ``path=None`` closes/clears the sink. Implies capture (no-op otherwise)."""
        if self._telemetry_fh is not None:
            try:
                self._telemetry_fh.close()
            except Exception:
                pass
            self._telemetry_fh = None
        if path is not None:
            self._capture_telemetry = True
            self._telemetry_fh = open(path, "w")

    def observe_own_move(self, move: chess.Move, observation) -> None:
        # Two-step belief: forward the post-own-move observation so the engine
        # prunes positions inconsistent with squares the move revealed.
        if self._engine is None:
            raise RuntimeError("reset() must be called before observe_own_move")
        self._record(
            "observe_own_move",
            lambda: self._engine.observe_own_move(move, observation),  # type: ignore[union-attr]
        )
        self._observation_history_events.append(
            observation_event_fingerprint("own", observation, move=move)
        )

    def observe_opp_move(self, observation) -> None:
        if self._engine is None:
            raise RuntimeError("reset() must be called before observe_opp_move")
        self._record(
            "observe_opp_move",
            lambda: self._engine.observe_opp_move(observation),  # type: ignore[union-attr]
        )
        self._observation_history_events.append(
            observation_event_fingerprint("opp", observation)
        )

    def set_time_budget(self, seconds: float | None) -> None:
        """Override the per-move wall budget for the NEXT pick_move. Lets the
        live worker compute a clock-proportional budget per move (3+2 etc.)
        instead of a fixed budget. None restores 'no time bound' (iters bind)."""
        self._time_budget = seconds

    def pick_move(self, view) -> chess.Move:
        if self._engine is None:
            raise RuntimeError("reset() must be called before pick_move")
        self.last_policy_action = None
        policy_view = self._policy_view(view)

        # Move-policy PRE hooks: a policy may play a move WITHOUT searching (the
        # book's FORCE entries; also instant, so it saves opening clock). First
        # non-None wins. Belief stays consistent — the harness calls
        # observe_own_move(move) next regardless, and choose_move never mutates
        # belief. last_solution is cleared so post-move telemetry honestly reports
        # "no search ran" rather than a stale prior-move ranking.
        for policy in self._policies:
            forced = policy.pre_move(policy_view)
            if forced is not None:
                self.last_policy_action = f"{policy.name}:pre"
                self._engine.last_solution = None
                return forced

        chosen: list[chess.Move] = []

        # Clock-aware budget: when enabled and the view carries a clock, budget
        # per-move from the remaining clock (keeps 3+2 solvent); else the static
        # configured budget. Works identically in selfplay + the live worker
        # because both put clock_remaining_ms/increment_ms on the view.
        if self._clock_time_enabled:
            budget = self._time_manager.budget_for(
                getattr(view, "clock_remaining_ms", None),
                getattr(view, "increment_ms", 0) or 0,
                static_fallback_s=self._time_budget,
            )
        else:
            budget = self._time_budget

        def _do() -> None:
            chosen.append(
                self._engine.choose_move(  # type: ignore[union-attr]
                    iterations=self._iterations,
                    i_sample_size=self._i_sample_size,
                    time_budget_seconds=budget,
                    max_actions=self._max_actions,
                    kluss_k=self._kluss_k,
                    use_rust_eq=self._use_rust_eq,
                    use_rust_tree=self._use_rust_tree,
                )
            )

        self._record("pick_move", _do)
        move = chosen[0]

        # Move-policy POST hooks: a policy may replace the search's pick (the
        # book's BLOCK entries; later, the catastrophe filter). Policies chain —
        # each sees the running choice. The queen-promo tiebreak is re-applied to
        # any replacement (idempotent on the unchanged search pick, which already
        # got it inside choose_move).
        for policy in self._policies:
            replacement = policy.post_move(policy_view, move, self._engine.last_solution)
            if replacement is not None and replacement != move:
                move = replacement
                if self._engine.queen_promo_tiebreak:
                    move = self._engine.rules.normalize_committed_move(move)
                self.last_policy_action = f"{policy.name}:post"
        return move

    def _policy_view(self, view):
        """Policy-only view with the observation-history digest attached.

        Search receives the original view. Policies may opt into path-sensitive
        matching by reading ``observation_history_fingerprint``; current-view
        policies ignore the extra attribute and behave as before.
        """
        return SimpleNamespace(
            perspective=view.perspective,
            own_legal_moves=view.own_legal_moves,
            visible_squares=view.visible_squares,
            visible_piece_map=view.visible_piece_map,
            clock_remaining_ms=getattr(view, "clock_remaining_ms", None),
            increment_ms=getattr(view, "increment_ms", 0),
            observation_history_fingerprint=observation_history_fingerprint(
                self._observation_history_events
            ),
        )

    def _record(self, kind: str, fn) -> None:
        """Run ``fn`` with timing + |P| capture; append a telemetry row.
        No-op timing path when capture is off (still calls ``fn``)."""
        if not self._capture_telemetry:
            fn()
            return
        import time as _time

        eng = self._engine
        p_pre = eng.enumerator.size if eng is not None else 0
        t0 = _time.monotonic()
        fn()
        wall_ms = (_time.monotonic() - t0) * 1000.0
        p_post = eng.enumerator.size if eng is not None else 0
        self._ply_seen += 1
        row: dict = {
            "ply": self._ply_seen,
            "kind": kind,
            "p_pre": p_pre,
            "p_post": p_post,
            "wall_ms": round(wall_ms, 2),
        }
        # Capture cap-probe fields from the enumerator if the call touched
        # it (update_own_move / update_opp_move set these; pick_move does not).
        if eng is not None and kind in ("observe_own_move", "observe_opp_move"):
            row["p_raw"] = eng.enumerator.last_raw_count
            row["p_pre_cap"] = eng.enumerator.last_pre_cap_count
            row["downsampled"] = eng.enumerator.last_was_downsampled
        # pick_move search telemetry: iters actually completed (vs the
        # iteration cap) is the iteration-starvation signal — if it sits
        # well below the cap at a fixed time budget, the engine is time-
        # bound and more search speed buys strength. ``margin`` is the
        # top-1 vs top-2 action-value gap (the A6.2 regime input).
        if eng is not None and kind == "pick_move":
            sol = eng.last_solution
            if sol is not None:
                row["iters"] = sol.iterations
                vals = sorted(sol.action_values_at_root.values(), reverse=True)
                row["n_actions"] = len(vals)
                row["margin"] = round(vals[0] - vals[1], 4) if len(vals) >= 2 else None
                # Live-vs-replay residual instrumentation (2026-06-12): every
                # controlled replay of a live blunder picks correctly at the
                # same config/iters — the divergence must be in the live-solve
                # VALUES. Log the top-3 root action values per pick so cloud
                # runs compare directly against local replays.
                top3 = sorted(sol.action_values_at_root.items(),
                              key=lambda kv: -kv[1])[:3]
                row["top3"] = [[m.uci(), round(v, 4)] for m, v in top3]
            # Stockfish leaf-eval cache stats (cumulative across the game). The
            # cache is a bounded LRU (default 100K entries); tracking the per-move
            # delta + current size lets us validate the cap is right: hit-rate
            # should stay high, and sf_ch_size should plateau well below the cap
            # in steady state. Surfaces internal data already tracked but unlogged.
            sf = getattr(eng, "_stockfish", None)
            if sf is not None:
                row["sf_ch_hits"] = getattr(sf, "children_cache_hits", 0)
                row["sf_ch_misses"] = getattr(sf, "children_cache_misses", 0)
                row["sf_ch_size"] = len(getattr(sf, "_children_cache", {}))
                # Nonzero = Stockfish errored and a material fallback leaked
                # into leaf values that move (residual suspect elimination).
                row["sf_fb"] = getattr(sf, "fallback_count", 0)
        self.telemetry.append(row)
        # Live sink: write + flush this row now so a later SIGKILL can't erase the
        # trajectory up to this ply (see set_telemetry_sink).
        if self._telemetry_fh is not None:
            import json as _json
            try:
                self._telemetry_fh.write(_json.dumps(row, separators=(",", ":")) + "\n")
                self._telemetry_fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._telemetry_fh is not None:
            try:
                self._telemetry_fh.close()
            except Exception:
                pass
            self._telemetry_fh = None
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None
