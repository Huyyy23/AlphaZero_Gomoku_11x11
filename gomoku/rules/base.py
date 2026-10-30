"""Rule-set interface used by MCTS, self-play and the UI.

A rule set is the *only* place win detection and move legality live.
The rest of the program (``az/``, ``players/``, ``ui/``) talks to this
interface and never inspects raw board planes for rule logic.

Vectorization contract: implementations must not scan all 121 cells with a
Python loop per call. The supported pattern is: take the last played move
from ``board.history[-1]`` and check only the four lines through it (at most
31 cells), or operate on numpy arrays directly. ``legal_moves`` uses one
numpy boolean mask over the flat grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gomoku.board import Board


class Rules(ABC):
    """Abstract rule set for a two-colour connection game."""

    name: str = "base"

    @abstractmethod
    def legal_moves(self, board: Board) -> list[int]:
        """Return flat indices (``r*SIZE + c``) of currently playable cells.

        Must exclude occupied cells (and forbidden cells in Renju). Order is
        unspecified; callers that need determinism sort themselves.
        """

    @abstractmethod
    def is_win(self, board: Board) -> bool:
        """True if the position contains a winning line for some player.

        After ``play``/``undo`` the caller can rely on the last-move shortcut,
        but the method must be correct for any reachable board state.
        """

    @abstractmethod
    def winner(self, board: Board) -> int:
        """Return BLACK/WHITE who has a winning line, or 0 (EMPTY) if none."""

    @abstractmethod
    def is_terminal(self, board: Board) -> bool:
        """True when the game is over: someone won or the board is full."""

    def forbidden_moves(self, board: Board) -> list[int]:
        """Flat indices of moves the *current player* may not play.

        Default: empty list (Normal Gomoku has no restrictions). Renju
        overrides this for Black only.
        """
        return []

    # ------------------------------------------------------------------ #
    # Optional hook used by MCTS leaf expansion; default derives from
    # legal_moves so rule sets only override what they need.
    # ------------------------------------------------------------------ #
    def is_legal(self, board: Board, idx: int) -> bool:
        """Convenience membership test for a single flat index."""
        return idx in self.legal_moves(board)
