"""Rule sets: Normal (free) Gomoku and, later, Renju (T24).

``get_rules(name)`` is the single factory the rest of the codebase uses;
``configs/*.yaml`` selects the name. ``gomoku/`` must not import torch --
rules stay pure numpy so MCTS can call them millions of times cheaply.
"""

from __future__ import annotations

from gomoku.rules.base import Rules
from gomoku.rules.normal import NormalRules

_REGISTRY = {
    "normal": NormalRules,
    # "renju": RenjuRules,  # registered in T24
}


def get_rules(name: str) -> Rules:
    """Instantiate the rule set by config name."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown ruleset '{name}'; available: {sorted(_REGISTRY)}"
        ) from None
    return cls()


__all__ = ["Rules", "NormalRules", "get_rules"]
