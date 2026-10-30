"""Game runner -- drives a full game between two players under a ruleset.

Torch-free by design (plan.md T6): this module only knows the ``Player``
interface and the ``Rules`` interface, so it works identically for Random,
Human, MCTSPure and AzPlayer.  The runner validates every move before
playing it; an agent that returns an illegal move raises
:class:`~players.base.IllegalMoveError` rather than corrupting the board.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules.base import Rules
from players.base import IllegalMoveError, Player

REASON_WIN = "win"
REASON_DRAW = "draw"


@dataclass
class GameResult:
    """Outcome of one finished game."""

    winner: int  # BLACK / WHITE / EMPTY (0) for a draw
    reason: str  # REASON_WIN or REASON_DRAW
    moves: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_draw(self) -> bool:
        return self.winner == EMPTY


def render_board(board: Board) -> str:
    """ASCII rendering with 0-based row/col rulers (matches CLI docs)."""
    symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
    header = "    " + " ".join(f"{c:>1}" for c in range(SIZE))
    lines = [header]
    for r in range(SIZE):
        cells = " ".join(symbols[board.cell(r, c)] for c in range(SIZE))
        lines.append(f"{r:>2}  {cells}")
    return "\n".join(lines)


class Runner:
    """Plays complete games between two :class:`Player` instances."""

    def __init__(self, rules: Rules) -> None:
        self._rules = rules

    def play(
        self,
        black: Player,
        white: Player,
        *,
        on_move=None,
        max_moves: int | None = None,
    ) -> GameResult:
        """Run one game: ``black`` moves first. Returns a ``GameResult``.

        ``on_move(board, player, side)`` is called *after* each accepted
        move (used by the CLI to print the board); exceptions from the
        callback propagate -- it is presentation code, not game logic.
        ``max_moves`` caps game length defensively (None = no cap; a full
        11x11 board is 121 moves anyway).
        """
        board = Board()
        players = {BLACK: black, WHITE: white}
        played: list[tuple[int, int]] = []

        while True:
            side = board.current_player
            mover = players[side]
            move = mover.select_move(board)
            self._validate(board, move, mover)

            r, c = move
            board.play(r, c)
            played.append((r, c))
            if on_move is not None:
                on_move(board, mover, side)

            # Terminal check happens right after the move that ends the game,
            # so the winner's last stone is never missed and the losing side
            # is never asked for another move.
            if self._rules.is_win(board):
                return GameResult(
                    winner=self._rules.winner(board),
                    reason=REASON_WIN,
                    moves=played,
                )
            if self._rules.is_terminal(board):
                # No winner and no further play possible -> draw (board full).
                return GameResult(
                    winner=EMPTY, reason=REASON_DRAW, moves=played
                )
            if max_moves is not None and len(played) >= max_moves:
                raise RuntimeError(
                    f"game exceeded max_moves={max_moves} without terminal state"
                )

    def _validate(self, board: Board, move, mover: Player) -> None:
        if not (isinstance(move, tuple) and len(move) == 2):
            raise IllegalMoveError(
                f"{mover.name} returned malformed move {move!r}"
            )
        r, c = move
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            raise IllegalMoveError(
                f"{mover.name} moved out of range: ({r}, {c})"
            )
        idx = r * SIZE + c
        if not board.is_empty(r, c):
            raise IllegalMoveError(
                f"{mover.name} played occupied cell ({r}, {c})"
            )
        if not self._rules.is_legal(board, idx):
            raise IllegalMoveError(
                f"{mover.name} played illegal cell ({r}, {c}) under ruleset"
            )
