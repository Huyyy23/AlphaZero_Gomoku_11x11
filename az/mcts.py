"""PUCT MCTS for AlphaZero (11x11).

Implements the AlphaZero-style search:
- Each simulation goes down to a leaf and evaluates with EXACTLY one neural
  network inference (batch size 1 at leaf time in the simple case, or
  node-batched across parallel games; the interface supports both).
- The network returns policy logits/probs over all 121 moves and a value
  from the leaf's perspective (current player to move).
- Dirichlet noise is added only at the root.
- Temperature: use tau=1 for first `temp_moves`, else tau -> 0 (argmax).
- Value backup negates at each level (child value is opponent's view).
- Forbidden moves (Renju) are never expanded.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules.base import Rules


def _other(player: int) -> int:
    return BLACK if player == WHITE else WHITE


@dataclass
class MCTSNode:
    parent: "MCTSNode | None"
    move: int = -1  # move that led to this node
    player: int = 0  # player to move
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    is_expanded: bool = False


class MCTS:
    """AlphaZero MCTS with PUCT."""

    def __init__(
        self,
        rules: Rules,
        model: Any = None,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.15,
        dirichlet_eps: float = 0.25,
        temp_moves: int = 15,
        device: str = "cpu",
    ):
        self.rules = rules
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temp_moves = temp_moves
        self.device = device

    def search(
        self,
        board: Board,
        add_noise: bool = True,
        temperature: float | None = None,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Run MCTS from board. Returns (policy_pi, visit_counts)."""
        root = MCTSNode(parent=None, move=-1, player=board.current_player)
        
        # Add Dirichlet noise at root if requested
        if add_noise:
            legal = self.rules.legal_moves(board)
            if legal:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
                for i, move in enumerate(legal):
                    root.children[move] = MCTSNode(parent=root, move=move, player=_other(board.current_player))
                    root.children[move].prior = (1 - self.dirichlet_eps) * self._get_prior(root, move) + self.dirichlet_eps * noise[i]

        for _ in range(self.n_simulations):
            node = root
            path = [node]
            current_board = self._copy_board(board)
            
            # Selection
            while node.is_expanded and len(node.children) > 0:
                # Check if terminal
                if self.rules.is_terminal(current_board):
                    break
                move, node = self._select_child(node)
                current_board.play(*divmod(move, SIZE))
                path.append(node)
            
            # Check if terminal
            if self.rules.is_terminal(current_board):
                value = self._get_value(current_board, node.player)
            else:
                # Expansion and evaluation - ONE neural network inference
                value = self._expand_and_evaluate(node, current_board)
            
            # Backup
            self._backup(path, value)
        
        # Compute policy
        pi = self._get_policy(root, temperature)
        visit_counts = {move: child.visits for move, child in root.children.items()}
        return pi, visit_counts

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_move = -1
        best_child = None
        
        for move, child in node.children.items():
            score = self._ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child

    def _ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """PUCT formula: Q + U"""
        if child.visits == 0:
            q = 0
        else:
            q = child.value_sum / child.visits  # Q from child's perspective
        u = self.c_puct * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
        return q + u

    def _expand_and_evaluate(self, node: MCTSNode, board: Board) -> float:
        """Expand node with ONE neural network inference."""
        legal_moves = self.rules.legal_moves(board)
        if not legal_moves:
            value = self._get_value(board, node.player)
            node.is_expanded = True
            return value
        
        # Get prediction from model - ONE inference for all moves
        if self.model is not None:
            policy_probs, value = self._evaluate_with_model(board)
        else:
            # Fallback for testing without model
            policy_probs = np.ones(121) / 121
            value = np.array([0.0])
        
        # Mask forbidden moves
        forbidden = set(self.rules.forbidden_moves(board))
        for move in forbidden:
            policy_probs[move] = 0.0
        
        # Normalize
        policy_sum = policy_probs.sum()
        if policy_sum > 0:
            policy_probs = policy_probs / policy_sum
        else:
            # Fallback to uniform over legal moves
            policy_probs = np.zeros(121)
            for move in legal_moves:
                if move not in forbidden:
                    policy_probs[move] = 1.0 / len([m for m in legal_moves if m not in forbidden])
        
        # Create children
        for move in legal_moves:
            if move in forbidden:
                continue
            if move not in node.children:
                child = MCTSNode(
                    parent=node,
                    move=move,
                    player=_other(board.current_player)
                )
                child.prior = float(policy_probs[move])
                node.children[move] = child
        
        node.is_expanded = True
        return float(value[0]) if hasattr(value, '__len__') else float(value)

    def _evaluate_with_model(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate board with model - returns (policy_probs, value)."""
        from gomoku.features import board_to_planes
        planes = board_to_planes(board, self.rules, board.current_player)
        # Add batch dimension
        x = planes[np.newaxis, ...]
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            # Assume model is callable
            import torch
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.from_numpy(x).float()
                if hasattr(self.model, 'to') and next(self.model.parameters()).device.type != 'cpu':
                    x_tensor = x_tensor.to(next(self.model.parameters()).device)
                policy_logits, value = self.model(x_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1)
                return policy_probs.cpu().numpy()[0], value.cpu().numpy()[0]

    def _get_prior(self, node: MCTSNode, move: int) -> float:
        """Get prior for a move (uniform fallback)."""
        return 1.0

    def _get_value(self, board: Board, player: int) -> float:
        """Get value for terminal position from player's perspective."""
        if self.rules.is_win(board):
            winner = self.rules.winner(board)
            if winner == player:
                return 1.0
            elif winner == _other(player):
                return -1.0
        return 0.0  # draw

    def _backup(self, path: list[MCTSNode], value: float):
        """Backup value up the tree, negating at each level."""
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value  # negate for parent (opponent's perspective)

    def _get_policy(self, root: MCTSNode, temperature: float | None = None) -> np.ndarray:
        """Get policy from visit counts with temperature."""
        pi = np.zeros(121)
        if not root.children:
            return pi
        
        if temperature is None:
            # Use config-based temperature
            # For now use temperature based on move number - we'll handle this externally
            temperature = 1.0
        
        if temperature == 0:
            # Greedy
            best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
            pi[best_move] = 1.0
        else:
            # Temperature sampling
            visits = np.array([root.children[move].visits for move in root.children.keys()])
            moves = list(root.children.keys())
            visits_temp = visits ** (1.0 / temperature)
            pi_sum = visits_temp.sum()
            if pi_sum > 0:
                for i, move in enumerate(moves):
                    pi[move] = visits_temp[i] / pi_sum
            else:
                for move in moves:
                    pi[move] = 1.0 / len(moves)
        
        return pi

    def _copy_board(self, board: Board) -> Board:
        """Create a copy of board for simulation."""
        new_board = Board()
        new_board._grid[:] = board.grid
        new_board._current_player = board.current_player
        # Copy history if it exists
        if hasattr(board, '_history'):
            new_board._history = list(board._history)
        else:
            # Try to reconstruct or leave empty
            try:
                new_board._history = list(board.history)
            except:
                new_board._history = []
        return new_board
