"""Player layer -- interchangeable agents behind ``Player.select_move``.

Only ``base.Player`` is part of the public contract; concrete agents are
imported from their modules (``players.random``, ``players.human``, ...).
Nothing here may import torch -- GPU-heavy agents live in ``az/``.
"""

from players.base import IllegalMoveError, Player
from players.human import HumanPlayer
from players.random import RandomPlayer

__all__ = ["Player", "IllegalMoveError", "RandomPlayer", "HumanPlayer"]
