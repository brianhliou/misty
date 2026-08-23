"""Exact enumeration of the set P of opponent-positions consistent with
observation history.

Obscuro's first architectural component. Replaces the particle-filter
``BeliefState`` for v2 engine work. From the perspective player's POV:

  P_t = { board state at time t : board is consistent with every
          observation the player has seen so far }

Obscuro reports P_t typically has |P| ≈ 17K, max ~10⁶. Storage is the
only cost; no heuristic repair, no jitter, no reseed — exact
enumeration sidesteps the failure modes of particle approximation.

This package is correctness-critical for v2: KLUSS, GT-CFR, PCFR+ all
reason over P. A bug here doesn't surface as a flaky test — it
surfaces as the engine confidently playing for the wrong belief
state. Hence: test harness comes first (tests/test_p_enum_*.py);
implementation is driven against it.
"""

from .enumerator import PEnumerator
from .invariants import (
    assert_all_consistent_with_observation,
    assert_cardinality_bound,
    assert_truth_in_P,
)

__all__ = [
    "PEnumerator",
    "assert_all_consistent_with_observation",
    "assert_cardinality_bound",
    "assert_truth_in_P",
]
