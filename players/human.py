"""Human player driven by an injected input callback.

The player itself never touches ``input()`` directly -- the CLI passes a
callback so tests can script moves and future UIs (web, T?) can reuse the
same class.  Invalid or illegal coordinates are rejected with a prompt to
retry until a legal move is produced.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from gomoku.board import SIZE, Board
from gomoku.rules.base import Rules
from players.base import Player

# Callback signature: given the board and a hint message (None on the first
# attempt, an error description on retries), return a raw user string or None
# if no more input is available.
InputCallback = Callable[[Board, str | None], str | None]

# A hyphen that sits between two digits and is NOT part of a negative-number
# token (i.e. preceded by a non-digit or start, like "5-3") acts as separator.
_HYPHEN_SEP_RE = re.compile(r"(?<=\d)-(?=\d)")


def _is_number_token(tok: str) -> bool:
    """True for things like '-1', '10', '+3' -- tokens int() accepts."""
    try:
        int(tok)
        return True
    except ValueError:
        return False


class HumanPlayer(Player):
    """Asks the attached callback for coordinates, validating legality."""

    def __init__(self, rules: Rules, prompt: InputCallback) -> None:
        self._rules = rules
        self._prompt = prompt

    def select_move(self, board: Board) -> tuple[int, int]:
        hint: str | None = None
        while True:
            raw = self._prompt(board, hint)
            if raw is None:
                raise EOFError("no input available for HumanPlayer")
            hint = self._explain_reject(board, raw)
            if hint is None:
                r, c = _parse_rc(raw)
                return (r, c)

    def _explain_reject(self, board: Board, raw: str) -> str | None:
        """Return None if ``raw`` parses to a legal move, else an error msg."""
        try:
            r, c = _parse_rc(raw)
        except ValueError as exc:
            return str(exc)
        if not board.is_empty(r, c):
            return f"({r}, {c}) đã có đá"
        if not self._rules.is_legal(board, r * SIZE + c):
            return f"({r}, {c}) không hợp lệ theo luật hiện hành"
        return None


def _parse_rc(raw: str) -> tuple[int, int]:
    """Parse ``'r,c'`` / ``'r c'`` / ``'r-c'`` into a zero-based (row, col).

    Coordinates are documented as 0..SIZE-1 everywhere (CLI header, board
    ruler labels); anything outside that range is rejected outright rather
    than silently reinterpreted as 1-based -- no ambiguous zones.
    """
    cleaned = raw.strip().lower()
    if any(ch.isdigit() for ch in cleaned):
        # Normalize separators only when digits are present, so pure-garbage
        # input ("abc") fails with a format error instead of an int error.
        # Space-separated pairs ("5 3") become "5,3"; hyphens used as
        # separators ("5-3") become "5,3", but a hyphen directly between two
        # digits is kept (it may be a negative number like "-1" or "10-3"-style
        # ambiguity is resolved by the range check afterwards).
        chunks = cleaned.split()
        if len(chunks) == 2:
            # Space-separated pair ("5 3", "-1 0") -> comma form.
            cleaned = ",".join(chunks)
        cleaned = _HYPHEN_SEP_RE.sub(",", cleaned)
    parts = [p for p in cleaned.split(",") if p != ""]
    if len(parts) != 2:
        raise ValueError(f"'{raw}' không phải tọa độ dạng 'r,c'")
    try:
        r, c = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"'{raw}' chứa thành phần không phải số") from None
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        raise ValueError(
            f"({r}, {c}) ngoài bàn cờ -- dùng chỉ số 0..{SIZE - 1}"
        )
    return (r, c)
