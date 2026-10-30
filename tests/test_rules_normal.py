"""Tests for T4: Normal (free) Gomoku rules on the 11x11 board.

Covers the todo acceptance criteria: horizontal/vertical/diagonal wins,
6+ overline win, draw when full, legal_moves excludes occupied cells,
forbidden_moves is empty, and the vectorized full-scan agrees with the
incremental last-move fast path.
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules import get_rules
from gomoku.rules.base import Rules
from gomoku.rules.normal import NormalRules, has_win_on_grid


@pytest.fixture()
def rules() -> NormalRules:
    return NormalRules()


def make_board(moves: list[tuple[int, int]]) -> Board:
    """Build a board by replaying moves through play() (turn order honoured)."""
    b = Board()
    for r, c in moves:
        b.play(r, c)
    return b


def manual_board(black_cells: list[tuple[int, int]],
                 white_cells: list[tuple[int, int]]) -> Board:
    """Board built directly on the grid (no history) to exercise the scan."""
    b = Board()
    g = b.grid.reshape(SIZE, SIZE)
    for r, c in black_cells:
        g[r, c] = BLACK
    for r, c in white_cells:
        g[r, c] = WHITE
    return b


# ---------------------------------------------------------------------- #
# Interface / factory
# ---------------------------------------------------------------------- #
def test_factory_returns_normal_rules():
    r = get_rules("normal")
    assert isinstance(r, Rules)
    assert r.name == "normal"


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError):
        get_rules("renju-not-yet")


# ---------------------------------------------------------------------- #
# legal_moves
# ---------------------------------------------------------------------- #
def test_legal_moves_empty_board_is_all_cells(rules):
    lm = rules.legal_moves(Board())
    assert sorted(lm) == list(range(SIZE * SIZE))


def test_legal_moves_excludes_occupied(rules):
    b = make_board([(5, 5), (5, 6), (4, 4)])  # B, W, B
    lm = set(rules.legal_moves(b))
    assert {b.idx(5, 5), b.idx(5, 6), b.idx(4, 4)}.isdisjoint(lm)
    assert len(lm) == SIZE * SIZE - 3
    # No stone values anywhere in the returned cells.
    assert all(b.grid[i] == EMPTY for i in lm)


def test_forbidden_moves_empty_for_normal(rules):
    assert rules.forbidden_moves(make_board([(5, 5)])) == []


# ---------------------------------------------------------------------- #
# Win detection through normal play (fast path).
# Fixture helper: Black builds a line while White fills far-away cells that
# never form >=2 adjacent stones (so White can never win accidentally).
# ---------------------------------------------------------------------- #
_FILLERS = [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8), (0, 10),
            (2, 0), (2, 2), (2, 4), (2, 6), (2, 8), (2, 10),
            (4, 0), (4, 2), (4, 4), (4, 6), (4, 8), (4, 10)]


def _black_line(line: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Interleave Black's line moves with isolated White fillers."""
    assert len(_FILLERS) >= len(line) - 1
    moves: list[tuple[int, int]] = []
    fi = 0
    for mv in line:
        moves.append(mv)
        if fi < len(line) - 1:
            moves.append(_FILLERS[fi])
            fi += 1
    return moves


def test_win_horizontal(rules):
    b = make_board(_black_line([(5, 3), (5, 4), (5, 5), (5, 6), (5, 7)]))
    assert rules.is_win(b)
    assert rules.winner(b) == BLACK
    assert rules.is_terminal(b)


def test_win_vertical(rules):
    b = make_board(_black_line([(3, 5), (4, 5), (5, 5), (6, 5), (7, 5)]))
    assert rules.winner(b) == BLACK


def test_win_diagonal_backslash(rules):
    b = make_board(_black_line([(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]))
    assert rules.winner(b) == BLACK


def test_win_diagonal_slash(rules):
    # anti-diagonal: row +1, col -1
    b = make_board(_black_line([(3, 8), (4, 7), (5, 6), (6, 5), (7, 4)]))
    assert rules.winner(b) == BLACK


def test_white_can_win(rules):
    # Black plays isolated cells; White completes a vertical five.
    moves = [(0, 0), (9, 0), (2, 2), (9, 1), (4, 4), (9, 2),
             (6, 6), (9, 3), (8, 8), (9, 4)]
    b = make_board(moves)
    assert rules.winner(b) == WHITE


def test_overline_six_counts_as_win_in_normal(rules):
    # Six in a row horizontally for Black (Normal allows overlines).
    b = make_board(_black_line(
        [(5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)]))
    assert rules.winner(b) == BLACK


def test_four_in_row_is_not_a_win(rules):
    b = make_board(_black_line([(5, 3), (5, 4), (5, 5), (5, 6)]))
    assert not rules.is_win(b)
    assert not rules.is_terminal(b)


def test_split_run_not_a_win(rules):
    # B at (5,3),(5,4),(5,5),(5,7) -- gap at (5,6): only 4 contiguous max.
    moves = [(5, 3), (0, 0), (5, 4), (0, 2), (5, 5), (0, 4), (5, 7), (0, 6)]
    b = make_board(moves)
    assert not rules.is_win(b)


# ---------------------------------------------------------------------- #
# Terminal / draw
# ---------------------------------------------------------------------- #
def _draw_pattern() -> np.ndarray:
    """Full-board two-colour pattern with no 5-in-a-row in any direction.

    Found by randomized search (seed 3) and cross-verified against a brute
    force scanner; hardcoded here so the draw test is deterministic.
    """
    rows = [
        [1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2],
        [1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2],
        [1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1],
        [1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2],
        [2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2],
        [1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1],
        [2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1],
        [1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2],
        [1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1],
        [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
        [1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2],
    ]
    return np.array(rows, dtype=np.int8)


def test_draw_pattern_really_has_no_five():
    g = _draw_pattern()
    assert not has_win_on_grid(g, BLACK)
    assert not has_win_on_grid(g, WHITE)


def test_draw_when_board_full_without_win(rules):
    g = _draw_pattern()
    b = Board()
    b.grid[:] = g.ravel()
    # Fake a full history so is_full() reports True without replaying 121
    # strictly alternating turns (the pattern parity would not match).
    b._history = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    assert b.is_full()
    assert rules.is_terminal(b)
    # winner() fast path looks at the last history cell only; verify via the
    # scan helper that nobody actually has five.
    assert not has_win_on_grid(b.grid.reshape(SIZE, SIZE), BLACK)
    assert not has_win_on_grid(b.grid.reshape(SIZE, SIZE), WHITE)


# ---------------------------------------------------------------------- #
# Vectorized scan vs incremental agreement
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("cells", [
    [(5, 3), (5, 4), (5, 5), (5, 6), (5, 7)],          # horizontal
    [(3, 5), (4, 5), (5, 5), (6, 5), (7, 5)],          # vertical
    [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],          # backslash
    [(3, 8), (4, 7), (5, 6), (6, 5), (7, 4)],          # slash
])
def test_full_scan_matches_incremental(cells):
    g = np.zeros((SIZE, SIZE), dtype=np.int8)
    for r, c in cells:
        g[r, c] = BLACK
    assert has_win_on_grid(g, BLACK)
    b = manual_board(cells, [])
    # Manual board has no history -> winner() uses the full scan.
    assert NormalRules().winner(b) == BLACK


def test_full_scan_no_false_positive():
    g = np.zeros((SIZE, SIZE), dtype=np.int8)
    for r, c in [(5, 3), (5, 4), (5, 5), (5, 6), (5, 8)]:  # gap at (5,7)
        g[r, c] = BLACK
    assert not has_win_on_grid(g, BLACK)
    assert not has_win_on_grid(g, WHITE)


def test_full_scan_detects_edge_line():
    g = np.zeros((SIZE, SIZE), dtype=np.int8)
    for c in range(5):
        g[0, c] = WHITE
    assert has_win_on_grid(g, WHITE)
    g2 = np.zeros((SIZE, SIZE), dtype=np.int8)
    for r in range(SIZE):
        g2[r, 10] = WHITE
    assert has_win_on_grid(g2, WHITE)


def test_full_scan_random_against_bruteforce():
    """Randomised cross-check of the stride-window scan against a plain
    Python brute force (allowed in tests, forbidden in production paths)."""
    rng = np.random.default_rng(12345)
    for _ in range(200):
        g = rng.integers(0, 3, size=(SIZE, SIZE)).astype(np.int8)
        for player in (BLACK, WHITE):
            expected = False
            for r in range(SIZE):
                for c in range(SIZE):
                    if g[r, c] != player:
                        continue
                    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                        ok = True
                        for k in range(5):
                            rr, cc = r + dr * k, c + dc * k
                            if not (0 <= rr < SIZE and 0 <= cc < SIZE) or \
                                    g[rr, cc] != player:
                                ok = False
                                break
                        if ok:
                            expected = True
            assert has_win_on_grid(g, player) == expected


def test_undo_restores_non_terminal(rules):
    b = make_board(_black_line([(5, 3), (5, 4), (5, 5), (5, 6), (5, 7)]))
    assert rules.is_win(b)
    b.undo()  # remove winning black stone
    assert not rules.is_win(b)
    assert b.current_player == BLACK
