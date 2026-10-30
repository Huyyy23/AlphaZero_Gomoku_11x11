"""Board state for the 11x11 Gomoku game.

Pure game state only -- no rule-set logic (win detection, forbidden moves)
lives here. The board is stored as a flat numpy int8 array of size
``SIZE * SIZE`` with plane index ``r * SIZE + c``.

``SIZE = 11`` is the single source of truth for the board dimension; there is
no size parameter anywhere in the project (plan.md decision #1).
"""

from __future__ import annotations

import numpy as np

# Single source of truth for the board size (hardcoded by design).
SIZE = 11

EMPTY = 0
BLACK = 1
WHITE = 2


class Board:
    """Mutable 11x11 board state with play/undo and move history."""

    __slots__ = ("_grid", "_current_player", "_history")

    def __init__(self) -> None:
        self._grid = np.zeros(SIZE * SIZE, dtype=np.int8)
        self._current_player = BLACK
        self._history: list[tuple[int, int]] = []

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def grid(self) -> np.ndarray:
        """Flat int8 view of length ``SIZE * SIZE`` (index ``r*SIZE + c``)."""
        return self._grid

    @property
    def current_player(self) -> int:
        """BLACK or WHITE -- the player to move next."""
        return self._current_player

    @property
    def history(self) -> list[tuple[int, int]]:
        """Copy of the move history as ``(row, col)`` tuples, oldest first."""
        return list(self._history)

    @property
    def num_moves(self) -> int:
        """Number of stones currently on the board."""
        return len(self._history)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @staticmethod
    def idx(r: int, c: int) -> int:
        """Flat index for cell ``(r, c)``."""
        return r * SIZE + c

    def cell(self, r: int, c: int) -> int:
        """Return EMPTY/BLACK/WHITE at ``(r, c)``; raises if out of range."""
        self._check_range(r, c)
        return int(self._grid[self.idx(r, c)])

    def is_empty(self, r: int, c: int) -> bool:
        self._check_range(r, c)
        return self._grid[self.idx(r, c)] == EMPTY

    def is_full(self) -> bool:
        """True when every cell holds a stone."""
        return len(self._history) == SIZE * SIZE

    # ------------------------------------------------------------------ #
    # Mutators
    # ------------------------------------------------------------------ #
    def play(self, r: int, c: int) -> None:
        """Place a stone for the current player at ``(r, c)``, append to
        history and switch turns. Raises ``ValueError`` for out-of-range
        coordinates or an occupied cell."""
        self._check_range(r, c)
        i = self.idx(r, c)
        if self._grid[i] != EMPTY:
            raise ValueError(f"cell ({r}, {c}) is already occupied")
        self._grid[i] = self._current_player
        self._history.append((r, c))
        self._current_player = _other(self._current_player)

    def undo(self) -> tuple[int, int]:
        """Remove the last stone, restore the previous turn and return the
        undone ``(r, c)``. Raises ``IndexError`` on an empty history."""
        if not self._history:
            raise IndexError("undo called on an empty history")
        r, c = self._history.pop()
        self._grid[self.idx(r, c)] = EMPTY
        # The player who just moved now has to move again.
        self._current_player = _other(self._current_player)
        return (r, c)

    def reset(self) -> None:
        """Clear the board back to the initial state."""
        self._grid.fill(EMPTY)
        self._current_player = BLACK
        self._history.clear()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_range(r: int, c: int) -> None:
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            raise ValueError(f"move ({r}, {c}) is outside the {SIZE}x{SIZE} board")

    def copy(self) -> "Board":
        """Independent duplicate of the current state."""
        new = Board.__new__(Board)
        new._grid = self._grid.copy()
        new._current_player = self._current_player
        new._history = list(self._history)
        return new

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Board(moves={len(self._history)}, "
            f"to_move={'BLACK' if self._current_player == BLACK else 'WHITE'})"
        )


def _other(player: int) -> int:
    """Return the opposite colour of ``player``."""
    return BLACK if player == WHITE else WHITE
