"""Single source of truth for the intended-strongest v2 engine config.

To READ a resolved config (do NOT infer from the inheritance chain or the inline
"SHIPPED" comments — they go stale): run ``scripts/show_engine_config.py`` and see
``docs/engine/engine-config-truth.md``. Which engine-id is live is platform-side.

Why this module exists
----------------------
The strength-defining knobs of EngineV2 are split across two mechanisms:

  1. Constructor kwargs  — ``i_sample_size``, ``kluss_k``, ``resolve_gadget``,
     ``resolve_cvar_q`` (per-instance).
  2. PROCESS-GLOBAL flags read at import — king-aware leaf emission and the
     tanh normalization scale, both module constants in ``cfr.leaf_eval``
     with no per-instance override (``set_king_aware_leaf`` /
     ``set_tanh_scale_cp`` mutate the process).

Before this module each consumer (the live worker, ``play_human``, the bakeoff
harness) hand-assembled the set, and they DRIFTED: the live worker enabled
king-aware but ``play_human`` did not, so the two served *different* engines —
the same fix-in-one-path-absent-in-another bug class that produced the live
king blunder in room 8221c5ab (Kd2-c3 into a believed knight while +5). And the
bakeoff harness could not turn king-aware on at all, so any bakeoff "validating"
the prod config was scoring a king-blind engine that prod no longer runs.

This module is the ONE place that defines "the strongest engine" so every
consumer serves the identical thing and bakeoffs validate the real prod config.

What's in / out
---------------
The profile captures only STRENGTH knobs (which engine). Deployment knobs that
legitimately vary per caller — seed, per-move time budget, |P| cap, iteration
cap — are supplied at build time, not baked into the profile.

Merged-eq (``FOW_EQ_MERGED``) is intentionally NOT in the profile: it's a
throughput experiment (non-byte-identical, strength-unvalidated by bakeoff) and
stays an env opt-in.

The bare ``EngineV2`` / ``EngineV2Strategy`` defaults are deliberately left
king-blind / gadget-off — "default matches prior behavior" — so the parity and
reproducibility test guards are unaffected. This profile is explicit opt-in.
"""

from __future__ import annotations

import os
import dataclasses
from dataclasses import dataclass

from .cfr.leaf_eval import (
    set_king_aware_leaf,
    set_king_band_floor,
    set_tanh_scale_cp,
)

# A high iteration cap so the per-move TIME budget is what binds, not the iter
# cap. A low cap is silently reached first and leaves search on the table —
# this bit eval_position_suite, the parity sweep, and play_human (see 6a8d98a).
DEFAULT_ITER_CAP = 10_000_000


@dataclass(frozen=True)
class V2Profile:
    """Strength-defining configuration of the v2 engine.

    Frozen so it can be a module-level constant, hashed, and logged for the
    ``engine_versions/`` checkpoint freeze. Vary one knob off the canonical
    profile with :func:`dataclasses.replace` (e.g. a king-aware A/B), never by
    hand-reassembling the whole set.
    """

    name: str
    i_sample_size: int
    kluss_k: int | None
    resolve_gadget: bool
    resolve_cvar_q: float
    king_aware_leaf: bool
    tanh_scale_cp: float = 500.0
    # 1.0 = prior hard ±1 king-capture clamp; <1.0 bands the value so material
    # orders ties within the band. A king-safety↔material dial, NOT a validated
    # underpromotion fix (measured negative — lab/probe_king_band.py). Default
    # 1.0 keeps STRONGEST byte-identical until a bakeoff validates a band floor.
    king_band_floor: float = 1.0
    # Never commit a rook/bishop underpromotion when the queen promo is legal.
    # Queen strictly dominates them in FoW (no stalemate); the engine is only
    # ever indifferent here because the king-capture +1 is material-blind
    # (root-cause: lab/diag_king_capture_leaves.py). A principled tiebreak, not
    # a strength gamble — ON by default; verify no-regression in bakeoff.
    queen_promo_tiebreak: bool = True
    # Win-fast: among equal-top-value moves, prefer an immediate capture of a
    # CONFIDENTLY-located enemy king instead of dawdling (the value saturates at
    # +1 for every winning move, so the search alone shuffles in won positions —
    # observed live: declined Re1xh1 on a visible king, game f0554c46 ply72).
    # Default OFF keeps v1.0/STRONGEST byte-identical (golden gate); v1.1-rc2+ opt
    # in. apply_process_flags seeds FOW_WIN_FAST; EngineV2 reads it (build_strategy
    # passes win_fast=None, so the env is the channel).
    win_fast: bool = False
    # Frozen serving toggles — part of the engine's identity, not per-run knobs.
    # These were historically hand-set as env vars on the prod worker AND in each
    # bakeoff setup-command (two copies of one list → drift). Now the profile owns
    # them: apply_process_flags() seeds the env from these via setdefault, so a
    # consumer that calls it no longer needs them hand-set. An explicit env still
    # wins (setdefault), preserving per-arm override + making the cutover a no-op
    # for prod (where the env is already set). bottom-K bounds the belief
    # expansion (memory safety); clock_time = clock-aware per-move budget;
    # early_stop = convergence-based early stop.
    bottomk_expansion: bool = True
    clock_time: bool = True
    early_stop: bool = True
    # Opening book (MovePolicy, FOW_OPENING_BOOK): a forced-move stopgap for the
    # opening (~one 2.Nc3 + block Qh4). LAST RESORT — default OFF and we'd rather
    # not maintain it; real opening strength should come from the gadget/search,
    # not this. Kept as a profile field only so a version can DECLARE it explicitly
    # instead of it floating as an orphan env flag.
    opening_book: bool = False
    # --- Faithful-stack knobs (the Obscuro program; 2026-06-10) ---------------
    # Added after the king-aware-leaf instrument mismatch cost a week: every
    # local rig ran flags the cloud probes never set ("every strength flag OFF
    # by default" — the 2026-05-29 trap, again, through the bakeoff path). A
    # profile that OWNS the full stack means a config exists by NAME everywhere
    # — `--profile candidate` in a ticket, CANDIDATE in a rig — or not at all.
    # Constructor kwargs (per-instance):
    gadget_iterative: bool = False     # PROPER in-solve Resolve gadget
    gadget_alpha: bool = False         # non-uniform alpha(J)=½(y/Σy+1/m)
    resolve_blueprint: str | None = None  # "carryover" = continual re-solving
    carryover_subtree: bool = False
    structural_carry: bool = False
    # Process-global env toggles (seeded by apply_process_flags, setdefault):
    gadget_merged: bool = False        # FOW_GADGET_MERGED — lag-free 1-walk pass
    gadget_wexp: bool = False          # FOW_GADGET_WEXP — weighted expansion roots
    wexp_mix: float = 0.0              # FOW_GADGET_WEXP_MIX — uniform-floor dial
    c1_fallback: bool = False          # FOW_GADGET_C1_FALLBACK — min{ṽ(h),v*} gifts
    gadget_iter_interval: int = 1      # FOW_GADGET_ITER_INTERVAL — lockstep cadence
    # Commit-layer material-catastrophe floor (2026-06-15): the queen/material
    # generalization of the king-aware leaf. ``commit_cvar`` runs the read-only
    # gadget at commit ON TOP of the iterative solve; ``defensive_danger`` makes
    # it floor a move's per-world value to the post-loss value when the opponent
    # can win >= FOW_GADGET_DEFENSIVE_MIN_CP (default 300cp) by static exchange
    # next ply. This lets the worst-case objective SEE the material hang that the
    # i-limited carryover misses — the danger worlds are uncarried, so they fall
    # to the C1 static fallback which is blind to the next-ply capture. Validated:
    # saves the queen on 945dc208/c6c53a42, preserves king-safety on
    # 0eaedfb1/c5a9eb83. Both needed (the floor only runs inside the commit gadget).
    commit_cvar: bool = False          # FOW_GADGET_COMMIT_CVAR — run commit gadget
    defensive_danger: bool = False     # FOW_GADGET_DEFENSIVE_DANGER — material floor
    # Commit-time material-safety guard (the surgical, no-over-defense fix):
    # generalizes the king commit-risk-check (engine_v2._commit_risk_check) to
    # material — POST-purification, if the chosen move hangs material (>= 300cp by
    # SEE next ply) in clearly more belief worlds than an in-VALUE-BAND alternative,
    # switch to the safe one. Value-banded + fires only on clear excess => never
    # shuffles safe positions (cf. the commit_cvar broad worst-case re-derive,
    # which over-defended: Bd3->Bf1, Brian's game 05c6e43c). `faithful` uses this.
    material_commit_guard: bool = False  # FOW_COMMIT_MATERIAL_GUARD + FOW_COMMIT_MATERIAL_CP=300
    # KING-ONLY commit guard (pairs with B'): drop the SEE material veto (B' now
    # owns material, sac-aware) and keep only the hard king-capture backstop for
    # the value-BLIND minority worlds B' can't weight. Requires material_commit_guard.
    commit_king_only: bool = False     # FOW_COMMIT_KING_ONLY — king veto only
    # Commit-guard VALUE-GAP (2026-06-15): the over-defense fix. The guard only
    # overrides a flagged-unsafe move if a safe move is within this much of its
    # gadget value. The gadget's EV already prices the fog-risk over all belief
    # worlds, so when it confidently prefers the flagged move (gap > this), the SEE
    # count is double-counting and switching concedes real value — the bug that
    # made the engine swap the equalizer Qxd5 (+0.6 over any safe move) for a
    # passive move and lose game 5d413d32. A real catastrophe (945) has a safe move
    # of comparable value (gap < 0.1), so the guard still fires. Validated: opening
    # keeps Qxd5, corpus catastrophes still caught at 0.3. 99 = always switch (prior).
    commit_value_gap: float = 99.0     # FOW_COMMIT_VALUE_GAP — trust the gadget when gap exceeds this
    # Threshold-free KING-SAFETY (2026-06-15): among moves within this much of the
    # best gadget value, commit the LOWEST king-risk one (200-world sample). Catches
    # sub-i-resolution king-risk the i=32 gadget can't see and B' can't down-weight
    # (the value-blind 3% d7d5 king-suicide, game 16a78780) WITHOUT a flat threshold
    # (5% misses 3%; 2% over-flags). Concedes no value (chooses among near-equals).
    commit_king_gap: float = 99.0      # FOW_COMMIT_KING_GAP — 99 = off
    # IN-GADGET catastrophe weighting (B', 2026-06-14): the faithful, no-commit-
    # guard fix. The iterative Resolve gadget already DETECTS a hang world (the
    # opponent follows it, follow_p=1) but reach-weighted averaging dilutes that
    # one catastrophe under many small passive-negative margins, so the worst-case
    # objective doesn't avoid it (Phase-1 diagnostic, plan §5c). This multiplies a
    # followed world's gadget weight by how badly we're losing it (opp-POV value),
    # so a real catastrophe DOMINATES the worst-case instead of being diluted. 0 =
    # off = faithful Resolve. Only fires on worlds we're losing (opp_v > 0) → safe
    # positions untouched (no over-defense, unlike commit_cvar). The principled
    # alternative to the material_commit_guard — keeps the fix inside the objective.
    severity_boost: float = 0.0        # FOW_GADGET_SEVERITY_BOOST — catastrophe up-weight
    # PHASE switch (2026-06-15): engage the gadget only in the OPENING (ply <= this),
    # run gadget-off EV-max after. The gadget's worst-case robustness wins under
    # opening fog (can't calculate -> avoid exploitable/committal moves) but hedges
    # into passivity once the position is concrete (mid/end), where EV-max is sharper
    # — Brian's complementary-profiles read, confirmed by the |P|/visibility trace
    # (lab/uncertainty_trajectory.py: openings uniformly uncertain; concreteness
    # tracks winning). Commit guards stay on every move. 0 = OFF (gadget all game).
    gadget_opening_plies: int = 0      # FOW_GADGET_OPENING_PLIES — gadget only for ply <= N
    expansion_budget: int = 0          # FOW_V2_EXPANSION_BUDGET — 0 = 1/iter (legacy)
    # ADAPTIVE catastrophe prune (2026-06-16): the |P|-adaptive successor to the
    # static FOW_HV_PRUNE_FRAC. Hard-veto a commit whose severity-weighted EXPECTED
    # loss (prob × material centipawns) over the belief exceeds tau_eff = tau ×
    # clamp(|P|/pref, 1, pmax) — TIGHT when belief is sharp (opening/endgame, where
    # the +EV fog-catastrophes cluster), loosening up to pmax× in heavy fog so it
    # never paralyzes the midgame. NOT value-gated: the opening hangs are genuinely
    # +EV (a 3% queen-loss barely dents EV), so only a hard prob×severity veto
    # catches them — measured: CVaR-q swept 0.1→0.02 is inert, value-gating misses
    # them, this catches Qxd5/Qa4/Qf3 with captures intact (lab/adaptive_sweep.py,
    # lab/frac_sweep.py). Broad: 24/29 mined catastrophes vs static 21/29, controls
    # flat (lab/validate_mined.py adaptive_tau=12). 0/off = static path, byte-identical.
    hv_prune_adaptive: bool = False    # FOW_HV_PRUNE_ADAPTIVE
    hv_prune_cp: float = 500.0         # FOW_HV_PRUNE_CP — min piece value the prune polices
    hv_prune_tau: float = 12.0         # FOW_HV_PRUNE_TAU — base cp expected-loss floor
    hv_prune_pref: float = 100.0       # FOW_HV_PRUNE_PREF — |P| reference for loosening
    hv_prune_pmax: float = 8.0         # FOW_HV_PRUNE_PMAX — max loosening multiple
    # Prune king-step FLOOR (2026-06-17): the king redirect only engages when the picked
    # move's king-risk exceeds this. 0 = legacy threshold-free step, which shaved a
    # negligible 1.2% king-risk for a 0.17-worse move (exd5 -> Qxd5, game 81fa6bda, a
    # LOSS). A small floor (~0.02) skips that nothing while still firing on the ~3% d7d5
    # suicide via the validated threshold-free redirect. lab/qa5_*.py, lab/rc5_*.py.
    hv_prune_king_floor: float = 0.0     # FOW_HV_PRUNE_KING_FLOOR
    # Prune NET-HANG floor (2026-08-21, game 42b652b6): the legacy hang test nets the
    # victim credit against the recapture, so queen-takes-defended-ROOK (net 400cp)
    # slips under the 500cp gross floor — the prune scored the game-losing Qxe8 as
    # 0-risk AND its redirect moved the commit back onto it after the material guard
    # had switched away. >0 = also flag gross >= cp hangs whose net SEE deficit is
    # >= this floor (300 mirrors the material guard's bar; winning captures and
    # exchange sacs stay unflagged). 0 = OFF, byte-identical legacy.
    hv_prune_net_floor: float = 0.0      # FOW_HV_PRUNE_NET_FLOOR
    # Carryover |P|-gate (2026-06-17): skip carryover/structural-carry when |P| <= this
    # — at small |P| a fresh re-sample is the COMPLETE exact belief, which the
    # search-biased carried subset corrupts (under-weights rare threat worlds -> bad
    # opening moves, 1.c4 d5 2.Nc3 Nc6??; lab/nc6_carryover.py: carry-on plays Nc6 2/4,
    # off plays the correct ...c6 4/4). 0 = no gate = byte-identical to prior behavior.
    carryover_min_p: int = 0           # FOW_CARRYOVER_MIN_P

    def apply_process_flags(self) -> None:
        """Set the PROCESS-GLOBAL flags this profile requires.

        Two kinds:
          * leaf-eval module constants (king-aware, tanh-scale, band floor) — set
            authoritatively via their setters (no per-instance override exists).
          * frozen serving toggles (bottom-K / clock / early-stop) — seeded into
            the env via ``setdefault`` so an explicit env still overrides. These
            are read from ``os.environ`` at their own callsites (PEnumerator,
            EngineV2Strategy, gt_cfr), which run AFTER this; seeding here means a
            consumer that calls apply_process_flags no longer hand-sets them.

        :meth:`build_strategy` calls this; call it directly if you construct
        ``EngineV2`` by hand. Mutates global state — in a shared process (pytest)
        save/restore around it.
        """
        set_king_aware_leaf(self.king_aware_leaf)
        set_tanh_scale_cp(self.tanh_scale_cp)
        set_king_band_floor(self.king_band_floor)
        os.environ.setdefault("FOW_BOTTOMK_EXPANSION", "1" if self.bottomk_expansion else "0")
        os.environ.setdefault("FOW_V2_CLOCK_TIME", "1" if self.clock_time else "0")
        os.environ.setdefault("FOW_V2_EARLY_STOP", "1" if self.early_stop else "0")
        os.environ.setdefault("FOW_OPENING_BOOK", "1" if self.opening_book else "0")
        # Faithful-stack env toggles (same setdefault contract as above).
        os.environ.setdefault("FOW_GADGET_MERGED", "1" if self.gadget_merged else "0")
        os.environ.setdefault("FOW_GADGET_WEXP", "1" if self.gadget_wexp else "0")
        os.environ.setdefault("FOW_GADGET_WEXP_MIX", str(self.wexp_mix))
        os.environ.setdefault("FOW_GADGET_C1_FALLBACK", "1" if self.c1_fallback else "0")
        os.environ.setdefault("FOW_GADGET_ITER_INTERVAL", str(self.gadget_iter_interval))
        os.environ.setdefault("FOW_GADGET_COMMIT_CVAR", "1" if self.commit_cvar else "0")
        os.environ.setdefault("FOW_GADGET_DEFENSIVE_DANGER", "1" if self.defensive_danger else "0")
        if self.material_commit_guard:
            os.environ.setdefault("FOW_COMMIT_MATERIAL_GUARD", "1")
        if self.commit_king_only:
            os.environ.setdefault("FOW_COMMIT_KING_ONLY", "1")
        os.environ.setdefault("FOW_COMMIT_VALUE_GAP", str(self.commit_value_gap))
        os.environ.setdefault("FOW_COMMIT_KING_GAP", str(self.commit_king_gap))
        os.environ.setdefault("FOW_WIN_FAST", "1" if self.win_fast else "0")
        os.environ.setdefault("FOW_GADGET_SEVERITY_BOOST", str(self.severity_boost))
        os.environ.setdefault("FOW_GADGET_OPENING_PLIES", str(self.gadget_opening_plies))
        if self.hv_prune_adaptive:
            os.environ.setdefault("FOW_HV_PRUNE_ADAPTIVE", "1")
            os.environ.setdefault("FOW_HV_PRUNE_TAU", str(self.hv_prune_tau))
            os.environ.setdefault("FOW_HV_PRUNE_PREF", str(self.hv_prune_pref))
            os.environ.setdefault("FOW_HV_PRUNE_PMAX", str(self.hv_prune_pmax))
        if self.hv_prune_king_floor:
            os.environ.setdefault("FOW_HV_PRUNE_KING_FLOOR", str(self.hv_prune_king_floor))
        if self.hv_prune_net_floor:
            os.environ.setdefault("FOW_HV_PRUNE_NET_FLOOR", str(self.hv_prune_net_floor))
        if self.hv_prune_adaptive and self.hv_prune_cp != 500.0:
            os.environ.setdefault("FOW_HV_PRUNE_CP", str(self.hv_prune_cp))
        if self.carryover_min_p:
            os.environ.setdefault("FOW_CARRYOVER_MIN_P", str(self.carryover_min_p))
        # expansion_budget is deliberately NOT env-seeded: the env is
        # process-wide and would leak the budget to a different-i opponent arm
        # (same eb at i=32 = ~6x deeper per-world trees than at i=200). It's an
        # ARM property — passed as a constructor kwarg by build_strategy and
        # resolved per-arm by the bakeoff's --profile handling.

    def build_strategy(
        self,
        *,
        seed: int,
        time_budget_seconds: float | None,
        p_max_size: int | None,
        iterations: int = DEFAULT_ITER_CAP,
        capture_telemetry: bool = False,
    ):
        """Construct an ``EngineV2Strategy`` at this profile.

        Applies the process-global flags first, then builds with the strength
        kwargs. Deployment knobs (seed, budget, |P| cap, iter cap, telemetry)
        are caller-supplied — they don't define the engine. ``use_rust_tree``
        is left at its default (on); ``use_rust_eq`` is unused (the rust tree
        supersedes the rust-eq pass).
        """
        self.apply_process_flags()
        from .engine_v2 import EngineV2Strategy

        return EngineV2Strategy(
            seed=seed,
            iterations=iterations,
            i_sample_size=self.i_sample_size,
            time_budget_seconds=time_budget_seconds,
            p_max_size=p_max_size,
            kluss_k=self.kluss_k,
            resolve_gadget=self.resolve_gadget,
            resolve_cvar_q=self.resolve_cvar_q,
            capture_telemetry=capture_telemetry,
            queen_promo_tiebreak=self.queen_promo_tiebreak,
            gadget_iterative=self.gadget_iterative or None,
            gadget_alpha=self.gadget_alpha or None,
            resolve_blueprint=self.resolve_blueprint,
            carryover_subtree=self.carryover_subtree or None,
            structural_carry=self.structural_carry or None,
            expansion_budget=self.expansion_budget or None,
            hv_prune_adaptive=self.hv_prune_adaptive or None,
            hv_prune_tau=self.hv_prune_tau,
            hv_prune_pref=self.hv_prune_pref,
            hv_prune_pmax=self.hv_prune_pmax,
            hv_prune_king_floor=self.hv_prune_king_floor or None,
            carryover_min_p=self.carryover_min_p or None,
        )


# THE single source of truth. Campaign verdict (2026-05-29): i=32 + KLUSS k=2 +
# CVaR Resolve gadget (q=0.1) + king-aware leaf, tanh scale 500. Every live and
# bakeoff consumer of "the strongest engine" builds from this constant.
STRONGEST = V2Profile(
    name="v1.0-gadget-off",
    i_sample_size=32,
    kluss_k=2,
    # Gadget OFF (2026-06-02): the 3-arm + head-to-head bakeoff showed the
    # Resolve/Maxmargin gadget gives NO strength edge (gadget-ON+TB vs OFF = 47.5%
    # head-to-head over 40 games) yet costs ~10-30x memory and ~2x game length —
    # its worst-case selection plays passively into heavy-fog belief blowups (hit
    # the 16M cap 11/30 games vs 0/30 for OFF) and, untuned, threw won games
    # (27-2-1 vs OFF's 30-0-0 vs tier1). king_aware_leaf stays ON — that's the
    # leaf-level king-safety signal, and it's why OFF has no king blunders.
    # resolve_cvar_q kept for when the gadget is toggled on for experiments.
    resolve_gadget=False,
    resolve_cvar_q=0.1,
    king_aware_leaf=True,
    tanh_scale_cp=500.0,
    # The served config's frozen toggles, now owned here instead of hand-set as
    # Railway env + bakeoff env-prefix. apply_process_flags seeds them (setdefault
    # → explicit env still wins), so dropping the env vars is safe.
    bottomk_expansion=True,
    clock_time=True,
    early_stop=True,
    opening_book=False,  # v1.0 ships WITHOUT the book (last-resort stopgap)
)

#: ===== VERSIONING (2026-06-15) =====================================================
#: A VERSION is a frozen, named profile. The ``name`` field IS the version, so the
#: boot dump's toggles-hash verifies served == intended. Engine-id -> version is 1:1
#: (live_move_worker._V2_PROFILE_BY_ID). RULE: never edit a shipped version in place;
#: the next engine is a NEW constant. ``strongest``/``faithful`` are DEV ALIASES, not
#: shipped names.
#:
#: v1.0 — the shipped player-facing engine ("Misty 1.0", id python-v2-v1.0): gadget
#: OFF, i=32, king-aware leaf, bottom-K, clock, early-stop, tanh-500, no book. FROZEN.
V1_0 = STRONGEST


# The Obscuro-faithful CANDIDATE (frozen 2026-06-10, ticket
# 2026-06-10-kingaware-wexp-pbound, engine 333c72c): the first config to clear
# every king-risk rig gate at a 30s-compatible budget AND run clean at scale
# (3W-3L vs the i=32/5s sparring partner, zero crashes/OOMs/pathological
# king-captures, |P| median 84.8K). NOT yet strength-validated vs STRONGEST —
# that H2H is the next phase; this constant exists so the H2H ticket, the rig,
# and any consumer reference the identical stack BY NAME (`--profile candidate`)
# instead of hand-assembling 12 flags (the king-aware instrument-mismatch trap).
# Deployment knobs (30s/move budget, 16M p_max, seed) stay caller-supplied.
CANDIDATE = V2Profile(
    name="candidate-2026-06-10-faithful",
    i_sample_size=200,
    kluss_k=2,
    resolve_gadget=True,
    resolve_cvar_q=0.1,
    king_aware_leaf=True,
    tanh_scale_cp=500.0,
    bottomk_expansion=True,
    clock_time=True,
    early_stop=False,  # early-stop unvalidated under the gadget regime
    gadget_iterative=True,
    gadget_alpha=True,
    resolve_blueprint="carryover",
    carryover_subtree=True,
    structural_carry=True,
    gadget_merged=True,
    gadget_wexp=True,
    wexp_mix=0.0,
    c1_fallback=True,
    gadget_iter_interval=1,
    expansion_budget=2000,
)

# The i=32 ATTRIBUTION arm (2026-06-12): identical faithful stack, only the
# belief-sample size changes. Motivated by the H2H ladder — 41.7/37.5/35.4/
# 33.3/31.2% with every mechanical fix landed (soft KLUSS un-deadlocked,
# eb=500 sized to box SF throughput, 90K iters/move verified healthy) — and
# by budget dilution: eb=500 over i=200 worlds is ~2.5 expansions per world
# vs ~15/world at i=32, and the eq pass walks 6x fewer roots per iteration.
# i=200 came from the paper; i=32 is the locally-validated optimum. A
# registry profile (not --v2-i) because _apply_profile overwrites args.v2_i
# and argparse can't distinguish an explicit 32 from the default.
CANDIDATE_I32 = dataclasses.replace(
    CANDIDATE, name="candidate-i32-2026-06-12", i_sample_size=32,
)

# The FAITHFUL Resolve candidate (frozen 2026-06-14, the strategy zoom-out):
# CANDIDATE_I32 with `resolve_cvar_q=0.0` (CVaR appears nowhere in Obscuro — paper
# re-read 2026-06-14: their Resolve objective is expected-margin, not a worst-case/
# tail blend — the one unambiguous non-Obscuro contaminant in the OBJECTIVE) plus
# the catastrophe-aversion fix, split along the value-visibility line:
#   * severity_boost=8 (B') — the IN-GADGET, faithful material fix. The iterative
#     Resolve gadget already DETECTS a material hang (the opponent follows that
#     world) but reach-weighted averaging dilutes it under many small passive
#     margins; B' up-weights a followed losing world so a real catastrophe
#     dominates the worst-case. Weights by the SF-rooted search value => sac-aware
#     (won't over-defend a sound sacrifice), unlike the SEE material guard.
#     Validated: 945dc208 e7e5/M=18% -> d5a5/M=0%, control 05c6e43c stays b1c3.
#   * material_commit_guard (FULL king+material) — B' is nondeterministically
#     MARGINAL: at boost=8 it fixes 945dc208 in ~2/3 of solves but hangs the queen
#     in ~1/3 (the 2.5s budget + world sampling flip the committed move), and no
#     boost 8..32 reliably fixes it without destabilizing other positions
#     (lab/bprime_sweep.py). A ~1/3 queen-hang rate is disqualifying, so the
#     deterministic post-hoc guard stays as the FULL (not king-only) backstop. B'
#     reduces how often it must override (keeping the committed move closer to the
#     gadget's own equilibrium); the guard guarantees no catastrophe ships. Known
#     residual: the guard's SEE material veto is sac-blind (passivity risk) — but
#     the over-defense control (05c6e43c) holds b1c3 in every test, and the verdict
#     is the human match, not these micro-positions.
# Everything else (alpha, KLUSS, one-sided GT-CFR, carryover blueprint, king-aware
# leaf, tanh scale) stays MATCHED to CANDIDATE_I32 / STRONGEST. The VERDICT is a
# human match + the catastrophe rigs, not H2H vs the prod-shape bot (see
# campaign-north-star memory / gadget-track journal).
FAITHFUL = dataclasses.replace(
    CANDIDATE_I32, name="v1.1-rc1-phased", resolve_cvar_q=0.0,
    severity_boost=8.0, material_commit_guard=True, commit_value_gap=0.3,
    commit_king_gap=0.2, gadget_opening_plies=14, opening_book=False,
    win_fast=True,  # same won-position short-circuit as v1.1-rc2 (fair A/B)
)

#: v1.1 RELEASE CANDIDATE (this session, 2026-06-15): v1.0's belief/leaf + opening-
#: only Resolve gadget (gadget_opening_plies=14) + B' + sac-aware commit guards
#: (value-gap, king/material decouple, threshold-free king-safety, net-material).
#: NOT a shipped version until the human benchmark clears it — it stays a candidate
#: (a "-rc" name), and only then gets frozen as v1.1. FAITHFUL is the working alias.
V1_1_RC = FAITHFUL

#: v1.1-rc2 KING-SAFE candidate (2026-06-15, the measurement-driven distillation):
#: SHIPPED v1.0 + ONLY the king-only threshold-free commit backstop. No gadget, no
#: B', no SEE material veto, no value-gap — i.e. NONE of the parts that caused the
#: faithful arm's over-defense + declined-capture bugs (those all came from the SEE
#: material veto, which `commit_king_only` keeps OFF). Motivated by a measurement
#: that REVERSED the "v1.0 has no catastrophes" assumption: shipped v1.0 plays the
#: d7d5 king-suicide 31% (5/16) at game 16a78780 ply 8 (lab/v1_suicide_rate.py) —
#: the king-aware LEAF is a soft signal the noisy i=32 EV-averaging overrides ~1/3
#: of the time, invisible to H2H (tier1 never sets the Qa4 trap; a human does). The
#: king-only backstop drops that to 0% (redistributing to v1.0's OWN preferred safe
#: moves, no over-defense) and still takes the winning queen captures f1f2/f5c2
#: (lab/v1_decline_check.py). This is the distilled essence of the whole gadget arm:
#: veto the rare king-suicide, nothing else. Everything else byte-identical to v1.0.
#: NOT shipped until a human game clears it (north-star gate); stays an -rc name.
V1_1_RC2 = dataclasses.replace(
    STRONGEST, name="v1.1-rc2-kingsafe",
    material_commit_guard=True, commit_king_only=True, commit_king_gap=0.2,
    win_fast=True,  # short-circuit a confident enemy-king capture (no won-position dawdle)
)

#: ★ v1.1 SHIPPED (2026-06-16) — the faithful/Resolve arm, frozen as the
#: player-facing release that SUPERSEDES v1.0. It is the ONLY config 0% on BOTH
#: catastrophe rigs (king-suicide d7d5 + queen-hang Qxd5; lab/catastrophe_rig.py)
#: and a 40-position move-divergence study (lab/divergence_scale.py) showed it
#: plays ≈ v1.1-rc2/v1.0 (85% identical moves, mean EV gap ~0.03) — i.e. the
#: catastrophe-completeness is essentially free, no strength regression. Frozen
#: from FAITHFUL (= CANDIDATE_I32 + cvar=0 + B'=8 + commit guards + opening-only
#: gadget at ply≤14 + win_fast). NEVER edit in place — a future engine = a NEW
#: constant (the locked versioning rule). Served as engine-id python-v2-v1.1.
V1_1 = dataclasses.replace(FAITHFUL, name="v1.1-faithful-resolve")

#: v1.1-rc3 ADAPTIVE-PRUNE candidate (2026-06-16): SHIPPED v1.1 + the |P|-adaptive
#: catastrophe prune (tau=12). Motivated by a five-way sweep proving the OTHER knobs
#: can't touch the opening +EV queen-hangs (Qxd5/Qa4/Qf3): CVaR-q 0.1→0.02 inert,
#: soft-KLUSS no-change, leaf-depth ruled out, and v1.1's own material_commit_guard
#: is value-gated (gap=0.3) so it MISSES them (Qa4 gap 0.31 > 0.3 → trusts the gadget
#: → hangs). The adaptive prune is the one mechanism that grips: a hard prob×severity
#: veto, tight at small |P| (opening), loosening in midgame fog so it adds no
#: over-defense. Broad mined corpus: 24/29 catastrophes (vs v1.1-on-candidate-i32's
#: 21/29 static), controls flat. NOT shipped until the human gate clears it (stays
#: an -rc name, per the campaign's instrument rule). Verdict = human match + rigs.
V1_1_RC3 = dataclasses.replace(
    V1_1, name="v1.1-rc3-adaptive-prune", hv_prune_adaptive=True, hv_prune_tau=12.0,
)

#: v1.1-rc4 CARRYOVER-GATE candidate (2026-06-17): rc3 + carryover gated OFF at small
#: |P| (carryover_min_p=512). The human gate found rc3 still playing weak openings
#: (1.c4 d5 2.Nc3 Nc6?? -> lost a knight); the diagnosis was NOT opening theory but a
#: carryover bug — at the tiny opening |P| (16) the carried search-biased belief
#: under-weights the Qa4/Nc3 threat worlds, so the engine mis-evaluates. A clean
#: (carryover-off) belief plays the correct ...c6 4/4; carryover-on plays Nc6 2/4
#: (lab/nc6_carryover.py). Gating carryover below |P|=512 restores the complete exact
#: opening belief while keeping carryover for the large-|P| midgame where it helps.
V1_1_RC4 = dataclasses.replace(
    V1_1_RC3, name="v1.1-rc4-carrygate", carryover_min_p=512,
)

#: v1.1-rc5 KING-FLOOR candidate (2026-06-17): rc4 + a floor on the prune's king step
#: (hv_prune_king_floor=0.02). The human gate found rc4 LOSING a game because the
#: threshold-free king step redirected the search's correct exd5 (+0.051) to Qxd5
#: (-0.115) just to shave a negligible 1.2% king-risk (game 81fa6bda). The floor skips
#: that (1.2% < 2%) while the ~3% d7d5 suicide still triggers the validated threshold-
#: free redirect. (A cost-benefit variant was tried first and let d7d5 slip 1/4 — too
#: value-sensitive; the floor is robust.) lab/qa5_deepdive.py, lab/rc5_exd5_check.py.
V1_1_RC5 = dataclasses.replace(
    V1_1_RC4, name="v1.1-rc5-kingfloor", hv_prune_king_floor=0.02,
)

#: NOTE (2026-06-17): an rc6 with hv_prune_cp=300 (police MINOR-piece hangs too) was
#: tried and DROPPED. The motivating case (a hung knight) was a SYMPTOM of an earlier
#: bad opening (…Nc6 into the Qa4 fork), not an avoidable minor blunder — and cp<500
#: reverses the deliberate K/Q/R scoping, risking over-defense on sound minor
#: development. The right lever for opening-exploitation is the carryover gate (done)
#: + the net, not patching minor hangs. The hv_prune_cp knob is retained (default 500
#: = rook+, byte-identical) but no shipped profile lowers it.

#: v1.1-rc6 NO-CARRYOVER candidate (2026-06-18): rc5 + carryover FULLY OFF
#: (structural_carry=False, carryover_subtree=False). The rc4 carryover_min_p=512
#: GATE was INCOMPLETE — it gates the Python-side belief/subtree discovery but NOT
#: the Rust solver's reset choice (carryover_subtree=True → reset_tree_keep_infosets,
#: which PRESERVES the prior move's CFR regrets across moves). So the carryover
#: corruption kept biting through the gate: at |P|=1 the warm Rust tree warm-starts
#: from move N's regrets and commits a worse move (game 0a0e6961 ply8 Bb5 vs the
#: correct exd5; demonstrated cold=exd5 / warm=Bb5, toggling carryover flips it).
#: Isolated audit (|P|=1 opening): carryover cost ~125cp (plies 8,10 recovered by
#: carry-off → SF best); the residual ~84cp (ply14) is baseline eval weakness, NOT
#: carryover. Disabling carryover also RESTORES reproducibility (warm==cold by
#: construction) — the property that made every live-vs-replay ghost undiagnosable.
#: NOTE the blueprint feed goes quiet too: set_values() (engine_v2.py) is gated by
#: the SAME structural_carry / carryover_subtree flags, so with both off the gadget's
#: Resolve falls back to the StubBlueprint baseline (constant -0.1) rather than a live
#: CarryoverBlueprint. That's acceptable — the SOUND within-move calibrated blueprint
#: is the NET (NetBlueprint), the proper future feed; the carryover VALUE carry was
#: only ever a stand-in, and it rode the same unsound regret reuse. Carryover is
#: unsound in principle (reuses a prior subgame's regrets in a different subgame), the
#: prior faithful-bundle H2H dragged ~21pp, AND it's a measured throughput PESSIMIZATION
#: (2026-06-19, warm 5s/move over game 0a0e6961's first 9 moves: rc6 does ~21% MORE
#: iters than rc5 — at |P|=1 the eq pass walks the whole accumulated tree every iter,
#: and the carried tree bloats across moves, outweighing the node-reuse savings; only
#: the |P|=13 move-2 case favored rc5). So "carryover helps speed" is falsified for the
#: opening, and "it helps the midgame" stays unverified folklore — pending a clean A/B.
V1_1_RC6 = dataclasses.replace(
    V1_1_RC5, name="v1.1-rc6-nocarry",
    structural_carry=False, carryover_subtree=False,
)

#: v1.2 — SHIP CANDIDATE (2026-06-19): v1.1 + carryover OFF *only*. The minimal,
#: lowest-risk fix = "v1.1 minus the opening regret-reuse corruption" (which also
#: restores warm==cold reproducibility); it changes NOTHING else vs the live v1.1.
#: DELIBERATELY does NOT enable the adaptive prune (hv_prune_adaptive stays off, as in
#: v1.1): the prune is validated but it ships as its OWN gated version, not bundled
#: into this carryover hotfix. (The prune CODE rides along in the same commit only
#: because it's file-entangled in engine_v2.py — but v1.2 leaves it OFF.) The ~3%
#: Re7 commit-variance blunder is separate (in v1.1 too) — the faithful-i fix.
V1_2 = dataclasses.replace(
    V1_1, name="v1.2", structural_carry=False, carryover_subtree=False,
)

#: ★ v1.3 — OPENING-HARDENING ship candidate (2026-06-20): v1.2 + the two opening-
#: catastrophe layers it deliberately shipped OFF, turned ON together.
#:   * the |P|-adaptive catastrophe PRUNE (hv_prune_adaptive) with the rc5 king-floor
#:     fix (hv_prune_king_floor=0.02) — i.e. the fully-iterated rc6 prune config. This
#:     is the GENERAL fix for the belief-dilution hang class (queen/king hangs that
#:     live in a minority of belief worlds and the EV-average can't see): a hard
#:     severity×prob veto, tight in the opening (small |P|), loose in midgame fog so
#:     it adds no over-defense. Validated 24/29 on the mined catastrophe corpus.
#:   * the curated opening BOOK (opening_book=True) — the exact-match belt-and-
#:     suspenders: queen-hang BLOCKs (Qh4/Qa5/Qxc5 still live in v1.2) + a dxe4 FORCE.
#:     Re-curated 2026-06-20 to drop the king-trap forces the v1.2 king guard now
#:     supersedes (lab/book_redundancy_probe.py: king safe 8/8 without them).
#: carryover stays OFF (inherited from v1.2). Expressed as a delta from the shipped
#: v1.2 so the diff is exactly "prune on + king-floor + book on".
#: ★ VERDICT GATE: a human-anchored match + the catastrophe rigs, NOT internal H2H
#: (north-star rule). Do NOT flip prod PROD_PLAYABLE to python-v2-v1.3 until that
#: clears — this constant + the worker mapping only make v1.3 buildable/bakeoffable.
V1_3 = dataclasses.replace(
    V1_2, name="v1.3", hv_prune_adaptive=True, hv_prune_king_floor=0.02,
    opening_book=True,
)

#: ★ v1.4 — CASTLE-INTO-CHECK fix (2026-06-20): identical PROFILE to v1.3 — every knob
#: is the same. The behavioral delta is in the base CODE, not the profile: the WS2
#: search move-gen now generates fog-castles (gen_fow_pseudo_legal_moves) instead of
#: python-chess legality, so the engine can finally SEE and devalue a castle that walks
#: the king onto a fog-attacked square. v1.3 never generated that move in the losing
#: belief worlds -> never scored the king-loss -> committed it (prod game a6f2e491:
#: O-O onto an attacked g8, then Qxg8). Validated: no-regression on 8/8 normal prod
#: positions (byte-identical to v1.3 at the same seed); a6f2 O-O picks 3/8 -> 0/8;
#: WS2 Python<->Rust byte-parity preserved. Because the fix is move-gen-level (not a
#: flag), v1.4 ships as its OWN frozen sha off the live engine sha 653aa33 (= v1.3
#: dark-chess serving + DMX/mini-xiangqi) + the castle fix; 653aa33 stays the rollback
#: target (revert engine.ref). Expressed as a pure rename so the only profile diff vs
#: v1.3 is the name; the only engine diff vs live is the castle fix.
V1_4 = dataclasses.replace(V1_3, name="v1.4")

#: ★ v1.5 — OPENING-BOOK update (2026-06-21): identical PROFILE to v1.4 (book ON).
#: The behavioral delta is the curated book DATA (data/opening_book.json), not the
#: profile: drop the now-redundant 2.Nc3 FORCE entries (v1.2's king guard covers the
#: a5-e1 diagonal on its own — book_redundancy_probe.py N=8) and add a forced ...dxe4
#: after 1.Nf3 d5 2.e4, killing the ~6% move-2 commit-slip to c6 (deep-dive 9ed7d9a5).
#: Both soundness-asserted. Ships as its OWN frozen sha off the live engine sha 3ae331c
#: (= v1.4 castle fix + DMX hardening) + the new book; 3ae331c stays the rollback target.
V1_5 = dataclasses.replace(V1_4, name="v1.5")

#: ★ v1.6-rc1 NET-HANG-PRUNE candidate (2026-08-21): v1.5 + the catastrophe
#: prune's net-hang floor (hv_prune_net_floor=300). Prod game 42b652b6 move 24:
#: Qa4xe8 (queen for DEFENDED rook, e8 covered in 9/13 belief worlds) nets
#: 400cp — under the prune's 500cp gross floor — so the prune scored it 0-risk,
#: then its STEP-1 redirect committed BACK onto it after _commit_material_check
#: had correctly switched away (compensating defects: guard saves, prune
#: un-saves, prune runs last). The floor closes the blind spot; validated by
#: tests/test_catastrophe_prune.py + the game repro (lab/). NOT shipped until
#: the mined-catastrophe corpus + declined-capture controls + bakeoff clear it
#: (north-star rule: strength claims need bakeoff evidence).
V1_6_RC1 = dataclasses.replace(
    V1_5, name="v1.6-rc1-net-prune", hv_prune_net_floor=300.0,
)

#: Profiles addressable by name. Versions first (canonical), then dev aliases.
PROFILES = {
    "v1.0": V1_0,            # frozen — python-v2-v1.0 ("Misty 1.0", historical)
    "v1.1": V1_1,            # python-v2-v1.1 ("Misty 1.1") — superseded by v1.2 2026-06-19
    "v1.2": V1_2,            # SHIPPED + LIVE — python-v2-v1.2 ("Misty 1.2", carryover fix; prune+book OFF)
    "v1.3": V1_3,            # v1.2 + prune ON + curated book ON — superseded by v1.4 (castle fix) 2026-06-20
    "v1.4": V1_4,            # CASTLE-INTO-CHECK fix — v1.3 profile + search move-gen sees fog-castles
    "v1.5": V1_5,            # OPENING-BOOK update — v1.4 profile + curated book (drop Nc3 forces, force ...dxe4)
    "v1.6-rc1": V1_6_RC1,    # release candidate (prune net-hang floor, game 42b652b6 Qxe8) — not shipped
    "v1.1-rc1": V1_1_RC,     # release candidate (faithful/gadget) — frozen as v1.1
    "v1.1-rc2": V1_1_RC2,    # release candidate (king-safe distillation) — not shipped
    "v1.1-rc3": V1_1_RC3,    # release candidate (adaptive prune) — not shipped
    "v1.1-rc4": V1_1_RC4,    # release candidate (adaptive prune + carryover gate) — not shipped
    "v1.1-rc5": V1_1_RC5,    # release candidate (+ king-step floor) — not shipped
    "v1.1-rc6": V1_1_RC6,    # release candidate (carryover OFF — fixes opening corruption + reproducibility) — not shipped
    "strongest": STRONGEST,  # dev alias of v1.0
    "faithful": FAITHFUL,    # dev alias of v1.1-rc1 / v1.1
    "kingsafe": V1_1_RC2,    # dev alias of v1.1-rc2
    "candidate": CANDIDATE,
    "candidate-i32": CANDIDATE_I32,
    "adaptive": V1_1_RC3,    # dev alias of v1.1-rc3
    "carrygate": V1_1_RC4,   # dev alias of v1.1-rc4
    "kingcb": V1_1_RC5,      # dev alias of v1.1-rc5
}
