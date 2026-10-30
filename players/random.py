"""Random baseline player.

Picks uniformly among ``rules.legal_moves(board)`` -- never hardcodes rule
knowledge, so it stays correct if the ruleset gains forbidden moves later.
"""

from __future__ import annotations

import random

from gomoku.board import SIZE, Board
from gomoku.rules.base import Rules
from players.base import Player


class RandomPlayer(Player):
    """Chooses a uniformly random legal move."""

    def __init__(self, rules: Rules, rng: random.Random | None = None) -> None:
        self._rules = rules
        self._rng = rng if rng is not None else random.Random()

    def select_move(self, board: Board) -> tuple[int, int]:
        legal = self._rules.legal_moves(board)
        if not legal:
            raise RuntimeError("RandomPlayer asked to move on a terminal board")
        idx = self._rng.choice(legal)
        return divmod(idx, SIZE)
