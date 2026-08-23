"""Registry for variant-specific engine hooks.

EngineV2's search/commit pipeline is variant-agnostic; anything a variant
needs to bolt into it (commit-stage safety guards, capability quirks) is
registered here keyed by ``rules.name``. Chess registers nothing — the lookup
returning ``None`` IS the chess path, byte-identical to a hard-coded
name-check that never matches.

Registration happens as a side effect of importing the variant's rules
module, which any caller must already do to construct that variant's Rules —
so the engine core never imports variant packages, and a tree stripped of a
variant package still imports cleanly.

Hook surface (all optional; duck-typed, probed with ``getattr``):

- ``commit_guards(engine, move, action_values) -> move`` — runs in the commit
  stage between the chess material guard and ``win_fast``/catastrophe-prune,
  exactly where the variant guard dispatch historically sat (variants run
  their royal guard here too; see ``variants_common.royal_guard``).
- ``standalone_rust_eq_compatible: bool`` (default True) — False when the
  standalone Rust-equilibrium mirror can't consume this variant's Python tree
  keys, forcing ``use_rust_eq`` off on the Python-tree path.
"""

from __future__ import annotations

_HOOKS: dict[str, object] = {}


def register(rules_name: str, hooks: object) -> None:
    """Register (or replace) the hooks object for ``rules_name``."""
    _HOOKS[rules_name] = hooks


def for_rules(rules: object) -> object | None:
    """The hooks registered for ``rules.name``, or None (the chess path)."""
    return _HOOKS.get(getattr(rules, "name", None))
