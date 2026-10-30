"""Normal (free) Gomoku rules for the 11x11 board.

Win condition: five or more stones of one colour in a row, horizontally,
vertically, or along either diagonal. Overlines (6+) count as a win -- this
is "Normal/Gomoku free", not Renju (Renju forbids Black overlines and some
3-3/4-4 shapes; that lives in ``renju.py``, T24).

Performance contract (todo T4 acceptance criterion): no Python loop scanning
all 121 cells per call.

* ``legal_moves`` -- one numpy mask over the flat grid.
* ``winner``/``is_win`` -- checks only the four lines through the *last*
  played move (at most 4 directions x <=8 neighbour probes), which is sound
  because a new winning line can only appear through the stone just placed.
* ``has_win_on_grid`` -- fully vectorized sliding-window scan used for
  externally constructed boards (tests, debug tools); no per-cell loop.
"""

from __future__ import annotations

import numpy as np

from gomoku.board import BLACK, EMPTY, SIZE, Board
from gomoku.rules.base import Rules

# The four line directions as (dr, dc).
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

WIN_LENGTH = 5


def has_win_on_grid(grid_2d: np.ndarray, player: int) -> bool:
    """Vectorized check: does ``player`` have >= WIN_LENGTH contiguous
    stones along any of the 4 directions on an (SIZE, SIZE) int grid?
    Implemented with stride-trick windows -- no Python loop over cells."""
    mask = grid_2d == player
    # Pad with a border wide enough that any 5-step walk from an original
    # cell stays inside the padded array (border cells are always False).
    pad = WIN_LENGTH - 1
    padded = np.zeros((SIZE + 2 * pad, SIZE + 2 * pad), dtype=bool)
    padded[pad:-pad, pad:-pad] = mask
    for dr, dc in _DIRECTIONS:
        # Window view over the padded array: element (i, j, k) is the k-th
        # cell of the line starting at original coord (i, j), i.e. padded
        # coord (i + pad, j + pad). Passing the sliced view to as_strided
        # sets the base pointer correctly; the last axis steps by (dr, dc).
        shifted = padded[pad:pad + SIZE, pad:pad + SIZE]
        win = np.lib.stride_tricks.as_strided(
            shifted,
            shape=(SIZE, SIZE, WIN_LENGTH),
            strides=(padded.strides[0], padded.strides[1],
                     padded.strides[0] * dr + padded.strides[1] * dc),
            writeable=False,
        )
        if bool(np.any(win.all(axis=2))):
            return True
    return False


class NormalRules(Rules):
    """Free-style Gomoku: first to line up 5+ stones wins."""

    name = "normal"

    # ------------------------------------------------------------------ #
    # Legality
    # ------------------------------------------------------------------ #
    def legal_moves(self, board: Board) -> list[int]:
        """All empty cells as flat indices (vectorized, single numpy pass)."""
        return np.flatnonzero(board.grid == EMPTY).tolist()

    def forbidden_moves(self, board: Board) -> list[int]:
        # Normal Gomoku has no forbidden moves (interface kept for Renju).
        return []

    # ------------------------------------------------------------------ #
    # Win detection
    # ------------------------------------------------------------------ #
    def winner(self, board: Board) -> int:
        """BLACK/WHITE if a 5+ line exists on the board, else EMPTY.

        Fast path: only the last move can create a win during normal play,
        so check the four lines through it. If history is empty (board built
        directly by tests/tools), fall back to the vectorized full scan.
        """
        hist = board.history
        if not hist:
            g = board.grid.reshape(SIZE, SIZE)
            if has_win_on_grid(g, BLACK):
                return BLACK
            if has_win_on_grid(g, WHITE):
                return WHITE
            return EMPTY
        r, c = hist[-1]
        player = int(board.grid[r * SIZE + c])
        if player != EMPTY and self._counts_through(board, r, c, player):
            return player
        return EMPTY

    def is_win(self, board: Board) -> bool:
        return self.winner(board) != EMPTY

    def is_terminal(self, board: Board) -> bool:
        return self.is_win(board) or board.is_full()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _counts_through(board: Board, r: int, c: int, player: int) -> bool:
        """True if any of the 4 lines through (r, c) holds >= WIN_LENGTH
        contiguous ``player`` stones."""
        grid = board.grid
        for dr, dc in _DIRECTIONS:
            run = 1
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < SIZE and 0 <= cc < SIZE and \
                        grid[rr * SIZE + cc] == player:
                    run += 1
                    rr += sign * dr
                    cc += sign * dc
            if run >= WIN_LENGTH:
                return True
        return False


RULES = NormalRules()
