"""CLI: play a game on the terminal.

Usage examples::

    python scripts/play.py --mode human-vs-random        # you (Black) vs random
    python scripts/play.py --mode ai-vs-ai               # random vs random
    python scripts/play.py --mode human-vs-random --ruleset normal

Coordinates are 0-based (rows/cols 0..10), entered as ``r,c`` or ``r c``.
This script must stay torch-free so the base game layer can be exercised on
machines without ML dependencies.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")  # allow running from repo root without install

from gomoku.board import BLACK, EMPTY  # noqa: E402
from gomoku.rules import get_rules  # noqa: E402
from gomoku.runner import REASON_DRAW, Runner, render_board  # noqa: E402
from players.human import HumanPlayer  # noqa: E402
from players.random import RandomPlayer  # noqa: E402


def _input_prompt(board, hint):
    if hint is not None:
        print(f"  ! {hint} -- thử lại:")
    try:
        return input("  bạn đi (r,c): ")
    except EOFError:
        return None


def _make_players(mode: str, ruleset: str):
    """Return (black_player, white_player) for the requested mode."""
    rules = get_rules(ruleset)
    if mode == "ai-vs-ai":
        return RandomPlayer(rules), RandomPlayer(rules)
    if mode == "human-vs-random":
        human = HumanPlayer(rules, prompt=_input_prompt)
        return human, RandomPlayer(rules)  # human plays Black
    raise SystemExit(f"chưa hỗ trợ mode '{mode}' (human-vs-random | ai-vs-ai)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", required=True, choices=["human-vs-random", "ai-vs-ai"]
    )
    parser.add_argument("--ruleset", default="normal")
    args = parser.parse_args(argv)

    black, white = _make_players(args.mode, args.ruleset)
    runner = Runner(get_rules(args.ruleset))

    show_board = args.mode != "ai-vs-ai"

    def on_move(board, player, side):
        if show_board:
            sym = "X" if side == BLACK else "O"
            r, c = board.history[-1]
            print(f"\n{player.name} ({sym}) đi ({r}, {c}):")
            print(render_board(board))

    result = runner.play(black, white, on_move=on_move)

    print()
    if result.reason == REASON_DRAW:
        print("Kết quả: HOÀ (bàn đầy)")
    else:
        winner = "ĐEN (X)" if result.winner == BLACK else "TRẮNG (O)"
        if result.winner == EMPTY:  # pragma: no cover - defensive
            print("Kết quả: không xác định")
        else:
            print(f"Kết quả: {winner} THẮNG sau {len(result.moves)} nước")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
