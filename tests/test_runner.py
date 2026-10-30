"""T6 tests: player interface, RandomPlayer, HumanPlayer, Runner, CLI wiring."""

from __future__ import annotations

import random
import subprocess
import sys

import pytest

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules import get_rules
from gomoku.runner import REASON_DRAW, REASON_WIN, GameResult, Runner, render_board
from players.base import IllegalMoveError, Player
from players.human import HumanPlayer
from players.random import RandomPlayer


@pytest.fixture()
def rules():
    return get_rules("normal")


# --------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------- #


def test_select_move_is_the_only_abstract_method():
    class Bad(Player):
        pass

    with pytest.raises(TypeError):
        Bad()  # type: ignore[abstract]

    class Good(Player):
        def select_move(self, board):
            return (0, 0)

    assert Good().name == "Good"


# --------------------------------------------------------------------- #
# RandomPlayer
# --------------------------------------------------------------------- #


def test_random_player_returns_legal_moves(rules):
    rng = random.Random(7)
    p = RandomPlayer(rules, rng=rng)
    board = Board()
    for _ in range(30):
        if rules.is_terminal(board):
            break
        r, c = p.select_move(board)
        assert 0 <= r < SIZE and 0 <= c < SIZE
        assert board.is_empty(r, c)
        board.play(r, c)


def test_random_player_skips_forbidden_via_rules(rules):
    """RandomPlayer must consult rules.legal_moves, not raw emptiness."""

    class StrictRules(type(rules)):
        def legal_moves(self, board):
            return [SIZE * SIZE - 1]  # only corner allowed

    p = RandomPlayer(StrictRules(), rng=random.Random(0))
    assert p.select_move(Board()) == (SIZE - 1, SIZE - 1)


# --------------------------------------------------------------------- #
# HumanPlayer
# --------------------------------------------------------------------- #


def script_prompts(*answers):
    """Callback factory: feeds canned answers, records retry hints."""
    it = iter(answers)
    seen_hints: list[str | None] = []

    def prompt(board, hint):
        seen_hints.append(hint)
        try:
            return next(it)
        except StopIteration:
            return None

    prompt.hints = seen_hints
    return prompt


def test_human_accepts_valid_coordinates(rules):
    cb = script_prompts("5,5")
    hp = HumanPlayer(rules, prompt=cb)
    assert hp.select_move(Board()) == (5, 5)
    assert cb.hints == [None]  # accepted on first attempt


def test_human_rejects_occupied_cell_with_retry_hint(rules):
    board = Board()
    board.play(3, 4)  # Black stone at (3,4)
    cb = script_prompts("3,4", "0,0")
    hp = HumanPlayer(rules, prompt=cb)
    assert hp.select_move(board.copy()) == (0, 0)
    assert cb.hints[0] is None
    assert "đã có đá" in cb.hints[1]


def test_human_rejects_garbage_and_out_of_range(rules):
    board = Board()
    board.play(2, 2)
    cb = script_prompts("abc", "99,99", "-1,0", "10,10")
    hp = HumanPlayer(rules, prompt=cb)
    assert hp.select_move(board) == (10, 10)
    assert "không phải tọa độ" in cb.hints[1]
    assert "ngoài bàn cờ" in cb.hints[2]


def test_human_rejects_illegal_under_rule_interface(rules):
    class BanCorners(type(rules)):
        def is_legal(self, board, idx):
            return idx not in (0, SIZE * SIZE - 1) and board.grid[idx] == EMPTY

    cb = script_prompts("0,0", "5,6")
    hp = HumanPlayer(BanCorners(), prompt=cb)
    assert hp.select_move(Board()) == (5, 6)
    assert "không hợp lệ" in cb.hints[1]


def test_human_raises_on_exhausted_input(rules):
    cb = script_prompts("junk")  # one bad answer then EOF
    hp = HumanPlayer(rules, prompt=cb)
    with pytest.raises(EOFError):
        hp.select_move(Board())


# --------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------- #


class ScriptedPlayer(Player):
    """Plays a fixed move list; raises if asked beyond its script."""

    def __init__(self, moves):
        self._moves = list(moves)
        self.asked = 0

    def select_move(self, board):
        i = self.asked
        self.asked += 1
        if i >= len(self._moves):
            raise AssertionError("ScriptedPlayer asked beyond script")
        return self._moves[i]


def test_runner_random_vs_random_terminates_consistently(rules):
    rng = random.Random(42)
    result = Runner(rules).play(RandomPlayer(rules, rng), RandomPlayer(rules, rng))
    assert isinstance(result, GameResult)
    assert result.reason in (REASON_WIN, REASON_DRAW)
    assert len(result.moves) == len(set(result.moves))  # no repeats
    assert len(result.moves) <= SIZE * SIZE
    # Replay the recorded moves; winner must match rules on final board.
    board = Board()
    for r, c in result.moves:
        board.play(r, c)
    if result.reason == REASON_WIN:
        assert rules.winner(board) == result.winner
        assert result.winner in (BLACK, WHITE)
    else:
        assert result.winner == EMPTY
        assert board.is_full() or rules.is_terminal(board)


def test_runner_detects_black_win_on_fifth_move(rules):
    line = [(5, 3), (0, 0), (5, 4), (0, 1), (5, 5), (0, 2), (5, 6), (0, 3)]
    black = ScriptedPlayer(line[0::2] + [(9, 9)])  # spare never played
    white = ScriptedPlayer(line[1::2])
    result = Runner(rules).play(black, white)
    assert result.reason == REASON_WIN
    assert result.winner == BLACK
    assert result.moves == line[:9]  # stops right after winning move


def test_runner_calls_on_move_after_each_accepted_move(rules):
    seen = []

    def on_move(board, player, side):
        seen.append((len(board.history), side, tuple(board.history[-1])))

    line = [(5, 3), (0, 0), (5, 4), (0, 1), (5, 5), (0, 2), (5, 6), (0, 3)]
    Runner(rules).play(ScriptedPlayer(line[0::2] + [(9, 9)]), ScriptedPlayer(line[1::2]),
                       on_move=on_move)
    assert seen[0] == (1, BLACK, (5, 3))
    assert seen[-1] == (9, BLACK, (5, 7))
    assert [n for n, _, _ in seen] == list(range(1, 10))


def test_runner_rejects_illegal_move_from_player(rules):
    dup = [(5, 5), (0, 0), (5, 6), (0, 1), (5, 5)]  # black repeats (5,5)
    with pytest.raises(IllegalMoveError):
        Runner(rules).play(ScriptedPlayer(dup[0::2]), ScriptedPlayer(dup[1::2]))


def test_runner_rejects_out_of_range_move(rules):
    with pytest.raises(IllegalMoveError):
        Runner(rules).play(ScriptedPlayer([(11, 0)]), ScriptedPlayer([]))


def test_runner_respects_max_moves_guard(rules):
    rng = random.Random(1)
    with pytest.raises(RuntimeError, match="max_moves"):
        Runner(rules).play(
            RandomPlayer(rules, rng), RandomPlayer(rules, rng), max_moves=5
        )


def test_render_board_shape():
    board = Board()
    board.play(0, 0)
    board.play(10, 10)
    text = render_board(board)
    lines = text.splitlines()
    assert len(lines) == SIZE + 1
    assert lines[1].startswith(" 0 ") and "X" in lines[1]
    assert "O" in lines[SIZE]


# --------------------------------------------------------------------- #
# Layering + CLI smoke tests
# --------------------------------------------------------------------- #


def test_runner_does_not_import_torch():
    code = (
        "import gomoku.runner, sys;"
        "assert 'torch' not in sys.modules, 'runner pulled torch in'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=".")


def test_cli_ai_vs_ai_runs_to_completion(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/play.py", "--mode", "ai-vs-ai"],
        capture_output=True, text=True, timeout=120, cwd=".",
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "Kết quả:" in out
    assert ("THẮNG sau" in out) or ("HOÀ" in out)


def test_cli_human_vs_random_scripts_a_game():
    # Human (Black) wins with a straight five while random White flails.
    stdin = "\n".join(["5,3", "5,4", "5,5", "5,6", "5,7"]) + "\n"
    proc = subprocess.run(
        [sys.executable, "scripts/play.py", "--mode", "human-vs-random"],
        input=stdin, capture_output=True, text=True, timeout=120, cwd=".",
    )
    assert proc.returncode == 0, proc.stderr
    assert "ĐEN (X) THẮNG" in proc.stdout
