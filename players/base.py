"""Player interface -- the single contract between the runner and any agent.

``select_move(board) -> (row, col)`` is the *only* method the game runner
ever calls (plan.md T6 acceptance criterion).  Random / Human / MCTSPure /
AzPlayer all implement it, so they are interchangeable in arenas and CLI.

This layer must stay torch-free; heavy agents live in ``az/`` and import
their own dependencies lazily inside their constructors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gomoku.board import Board


class Player(ABC):
    """Base class for every agent that can play a move on a board."""

    @abstractmethod
    def select_move(self, board: Board) -> tuple[int, int]:
        """Return ``(row, col)`` of the chosen move for ``board.current_player``.

        The move must be legal under the ruleset governing ``board``; the
        runner validates before playing and raises ``IllegalMoveError``
        otherwise.  Implementations must not mutate ``board``.
        """

    @property
    def name(self) -> str:
        """Human-readable identifier used by CLI output and arena logs."""
        return type(self).__name__


class IllegalMoveError(RuntimeError):
    """Raised when a player returns a move that is not legal on the board."""
