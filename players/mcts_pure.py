"""Pure MCTS baseline -- UCT tree search with random rollouts, no network.

Evaluation opponent for the AlphaZero agent (plan.md build order step 3).
It talks to the game *only* through the ``Rules`` interface
(``legal_moves`` / ``winner`` / ``is_win`` / ``is_terminal``, hard rule 1),
so it stays correct under any rule set -- Renju forbidden moves are simply
never returned by ``legal_moves`` and therefore never explored.

Design notes
------------
* Each node stores win statistics from the point of view of the player
  *to move at that node*.  During backup a rollout result is negated
  (here: re-attributed to the alternating colour) at every hop toward
  the root -- the sign trap called out as "the most common bug" in
  AGENTS.md (hard rule 2).
* The simulation loop is iterative so the exact ply parity between root
  and leaf is known; the child of the root that was visited gets the
  raw result, its ancestors alternate from there.
* Rollouts stop on ``rules.is_win`` only (plus a depth cap); a board
  where someone already has five in a row must never score as a draw.
* A win-in-1 scan runs before the search: a tactical shot needs no
  simulations, which makes the win-in-1 acceptance criterion exact
  instead of probabilistic.
* ``n_rollouts`` defaults to the configured MCTS budget (200) so arenas
  can give this baseline and the AlphaZero agent the same per-move
  simulation budget (plan.md decision 20).

Torch-free by contract: everything in ``players/`` imports without any
ML dependency.
"""

from __future__ import annotations

import math
import random

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules.base import Rules
from players.base import Player


def _other(player: int) -> int:
    return BLACK if player == WHITE else WHITE


class _Node:
    """One position in the MCTS tree.

    ``wins`` counts rollouts won by the player to move at this node.
    ``children`` maps flat move index -> child node (expanded lazily).
    ``moves`` snapshots the legal list at creation (for stats reporting).
    """

    __slots__ = ("parent", "move_into", "player_to_move", "children",
                 "visits", "wins")

    def __init__(self, parent: "_Node | None", move_into: int,
                 player_to_move: int) -> None:
        self.parent = parent
        self.move_into = move_into
        self.player_to_move = player_to_move
        self.children: dict[int, _Node] = {}
        self.visits = 0
        self.wins = 0


class MCTSPure(Player):
    """UCT + random-rollout agent implementing ``Player.select_move``."""

    def __init__(
        self,
        rules: Rules,
        n_rollouts: int = 200,
        *,
        c_uct: float = math.sqrt(2.0),
        max_playouts: int = 121,
        rng: random.Random | None = None,
    ) -> None:
        """``n_rollouts`` = simulations per move (config ``n_simulations``).

        ``c_uct`` is the exploration constant; ``max_playouts`` caps
        rollout depth (full-board fallback); ``rng`` makes play reproducible.
        """
        if n_rollouts <= 0:
            raise ValueError("n_rollouts must be positive")
        if c_uct < 0:
            raise ValueError("c_uct must be non-negative")
        self._rules = rules
        self._n_rollouts = int(n_rollouts)
        self._c = float(c_uct)
        self._max_playouts = int(max_playouts)
        self._rng = rng if rng is not None else random.Random()

    # ------------------------------------------------------------------ #
    # Player interface
    # ------------------------------------------------------------------ #
    def select_move(self, board: Board) -> tuple[int, int]:
        """Return ``(row, col)`` of the most-visited move after search.

        ``board`` is never mutated.
        """
        idx = self.search(board)[0]
        return divmod(idx, SIZE)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, board: Board) -> tuple[int, list[tuple[int, int]]]:
        """Run ``n_rollouts`` simulations from ``board``.

        Returns ``(chosen_flat_index, [(flat_index, visits), ...])`` in
        legal-move order; the visit table is exposed for tests/debugging.
        """
        rules = self._rules
        legal = rules.legal_moves(board)
        if not legal:
            raise RuntimeError("MCTSPure asked to move on a terminal board")
        if len(legal) == 1:
            return legal[0], [(legal[0], self._n_rollouts)]

        # Tactical shortcut: take an immediate win without searching.
        mover = board.current_player
        for idx in legal:
            board.play(*divmod(idx, SIZE))
            won = rules.winner(board) == mover
            board.undo()
            if won:
                return idx, [(idx, self._n_rollouts)]

        root = _Node(parent=None, move_into=-1, player_to_move=mover)
        work = Board()
        work._grid[:] = board.grid
        work._current_player = mover
        # Fresh (empty) history: NormalRules.winner falls back to a full
        # vectorized scan when history is empty, so copied-in stones are
        # still detected. The search replays only its own moves and undoes
        # them, keeping `work` consistent with the tree path at all times.

        root_legal = legal  # root's legal moves (the loop below rebinds `legal`)

        for _ in range(self._n_rollouts):
            # --- Selection: descend until a node has an untried move ---
            node = root
            plies = 0
            while True:
                legal = rules.legal_moves(work)
                untried = [m for m in legal if m not in node.children]
                if untried:
                    idx = self._rng.choice(untried)
                    work.play(*divmod(idx, SIZE))
                    child = _Node(node, idx, _other(work.current_player))
                    node.children[idx] = child
                    node = child
                    plies += 1
                    break
                # Fully expanded -> UCB1 over children.
                best_idx, best_score = self._uct_pick(node, legal)
                work.play(*divmod(best_idx, SIZE))
                node = node.children[best_idx]
                plies += 1

            # --- Rollout from the leaf, then undo the tree path ---
            result = self._rollout(work)
            for _ in range(plies):
                work.undo()

            # --- Backup: value alternates perspective each ply up ------
            # `result` is the winning colour of the final position; the
            # player to move at `node` just played into it, so they won
            # iff result == node.player_to_move. Going up one hop flips
            # which side a fixed result favours (hard rule 2).
            winner_for_node = result
            cur: _Node | None = node
            while cur is not None:
                cur.visits += 1
                if winner_for_node != EMPTY and \
                        winner_for_node == cur.player_to_move:
                    cur.wins += 1
                winner_for_node = (_other(winner_for_node)
                                   if winner_for_node != EMPTY else EMPTY)
                cur = cur.parent

        # A move reaches the stats only if it was expanded as a child of
        # the root during some simulation; unvisited legal moves count as
        # zero visits (the tree is sparse -- each rollout adds one leaf).
        assert root.children, "search ran zero simulations"
        visited = [i for i in root_legal if i in root.children]
        best = max(visited, key=lambda i: root.children[i].visits)
        table = [(i, root.children[i].visits if i in root.children else 0)
                 for i in root_legal]
        return best, table

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _uct_pick(self, node: _Node, legal: list[int]) -> tuple[int, float]:
        """UCB1 child selection; ties broken by first legal-move order."""
        log_n = math.log(node.visits + 1.0)
        best_idx, best_score = -1, -math.inf
        for idx in legal:
            child = node.children[idx]
            v = child.visits
            exploit = child.wins / v if v else 0.0
            explore = self._c * math.sqrt(log_n / (v + 1.0))
            score = exploit + explore
            if score > best_score:
                best_idx, best_score = idx, score
        return best_idx, best_score

    def _rollout(self, work: Board) -> int:
        """Random playout from ``work`` (mutates it, restores on exit).

        Stops when ``rules.is_win`` or when no legal move remains; capped
        at ``max_playouts`` random moves as a depth guard.  Returns the
        winning colour or EMPTY for a draw/cutoff.
        """
        rules = self._rules
        start_moves = work.num_moves
        while work.num_moves - start_moves < self._max_playouts:
            if rules.is_win(work):
                break
            legal = rules.legal_moves(work)
            if not legal:
                break
            work.play(*divmod(self._rng.choice(legal), SIZE))
        winner = rules.winner(work) if rules.is_win(work) else EMPTY
        while work.num_moves > start_moves:
            work.undo()
        return winner
