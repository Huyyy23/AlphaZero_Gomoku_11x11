"""Unit tests for ``gomoku.board`` (Task 3).

Covers: play/undo/history, turn switching, out-of-range errors, occupied-cell
errors, flat int8 grid layout and the torch-free import rule.
"""

import sys

import numpy as np
import pytest

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board


# --------------------------------------------------------------------------- #
# Constants / layout
# --------------------------------------------------------------------------- #
def test_size_is_11_and_constants():
    assert SIZE == 11
    assert (EMPTY, BLACK, WHITE) == (0, 1, 2)


def test_grid_is_flat_int8_of_length_121():
    b = Board()
    assert isinstance(b.grid, np.ndarray)
    assert b.grid.dtype == np.int8
    assert b.grid.shape == (SIZE * SIZE,)
    assert np.all(b.grid == EMPTY)


def test_flat_index_convention_r_times_size_plus_c():
    b = Board()
    r, c = 7, 3
    b.play(r, c)
    assert b.grid[r * SIZE + c] == BLACK
    # Sanity: a different cell with the same "sum" is untouched.
    assert b.grid[b.idx(3, 7)] == EMPTY


# --------------------------------------------------------------------------- #
# play / undo / history
# --------------------------------------------------------------------------- #
def test_play_places_current_player_stone_and_switches_turn():
    b = Board()
    assert b.current_player == BLACK
    b.play(5, 5)
    assert b.grid[Board.idx(5, 5)] == BLACK
    assert b.current_player == WHITE
    b.play(0, 0)
    assert b.grid[Board.idx(0, 0)] == WHITE
    assert b.current_player == BLACK
    assert b.history == [(5, 5), (0, 0)]
    assert b.num_moves == 2


def test_history_is_a_copy_not_internal_reference():
    b = Board()
    b.play(1, 1)
    h = b.history
    h.append((9, 9))
    assert b.history == [(1, 1)]


def test_undo_returns_move_and_restores_state():
    b = Board()
    snapshot = b.grid.copy()
    b.play(2, 3)
    b.play(4, 6)
    undone = b.undo()
    assert undone == (4, 6)
    assert b.grid[Board.idx(4, 6)] == EMPTY
    assert b.current_player == WHITE
    assert b.history == [(2, 3)]
    undone = b.undo()
    assert undone == (2, 3)
    assert np.array_equal(b.grid, snapshot)
    assert b.current_player == BLACK
    assert b.history == []


def test_undo_on_empty_history_raises():
    b = Board()
    with pytest.raises(IndexError):
        b.undo()


def test_play_then_undo_full_cycle_keeps_turn_correct():
    b = Board()
    for i in range(5):
        b.play(i, i)
    assert b.current_player == WHITE  # 5 moves played, white to move
    for i in reversed(range(5)):
        assert b.undo() == (i, i)
    assert b.current_player == BLACK


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("r,c", [(-1, 0), (0, -1), (11, 0), (0, 11), (11, 11), (99, 99)])
def test_out_of_range_raises_value_error(r, c):
    b = Board()
    with pytest.raises(ValueError):
        b.play(r, c)
    # Accessors validate too.
    with pytest.raises(ValueError):
        b.cell(r, c)


def test_occupied_cell_raises_and_leaves_state_untouched():
    b = Board()
    b.play(5, 5)
    before_player = b.current_player
    before_history = b.history
    with pytest.raises(ValueError):
        b.play(5, 5)
    assert b.current_player == before_player
    assert b.history == before_history
    # Also raises for a cell taken by the other colour.
    b.play(6, 6)
    with pytest.raises(ValueError):
        b.play(6, 6)


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #
def test_is_empty_and_is_full_and_reset():
    b = Board()
    assert b.is_empty(0, 0)
    b.play(0, 0)
    assert not b.is_empty(0, 0)
    assert not b.is_full()
    b.reset()
    assert b.num_moves == 0
    assert b.current_player == BLACK
    assert np.all(b.grid == EMPTY)


def test_copy_is_independent():
    b = Board()
    b.play(3, 4)
    c = b.copy()
    c.play(5, 5)
    assert b.grid[Board.idx(5, 5)] == EMPTY
    assert b.history == [(3, 4)]
    assert c.history == [(3, 4), (5, 5)]
    assert b.current_player == WHITE  # one move played by black
    assert c.current_player == BLACK  # two moves played, back to black


def test_gomoku_package_does_not_import_torch():
    """Hard rule: game core must stay torch-free (AGENTS.md architecture)."""
    code = (
        "import sys; "
        "assert 'torch' not in sys.modules; "
        "import gomoku.board; "
        "assert 'torch' not in sys.modules; "
        "print('ok')"
    )
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
