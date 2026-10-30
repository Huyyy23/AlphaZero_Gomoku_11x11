"""Feature planes and 8-fold symmetry transforms.

``board_to_planes(board, rules, player)`` returns a numpy array of shape
``(4, SIZE, SIZE)`` with the fixed plane layout used by *every* rule set
(plan.md decision #4 -- Normal and Renju share one network architecture):

    plane 0 : my_stones     -- stones belonging to ``player`` (the viewpoint)
    plane 1 : opp_stones    -- stones of the opponent of ``player``
    plane 2 : forbidden     -- cells ``player`` may not play (Renju Black
                              only; all zeros for Normal -- see AC)
    plane 3 : color_to_move -- all ones when it is ``player``'s turn, else 0

The module lives in ``gomoku/`` and therefore must NOT import torch
(hard rule 7); the conversion to torch tensors happens in ``az/``.

Symmetry: the D4 group of the square board has exactly 8 elements
(4 rotations x reflection). ``symmetry_transform(arr, k)`` applies element
``k`` in 0..7 to the *trailing two axes*, so the same function works for
feature planes ``(4, S, S)``, a single plane ``(S, S)`` and a policy vector
reshaped to ``(S, S)`` -- which is precisely what guarantees features and
policy are transformed consistently (T5 AC).
"""

from __future__ import annotations

import numpy as np

from gomoku.board import BLACK, Board
from gomoku.rules.base import Rules

# Fixed number of planes for all rule sets (network input depth).
N_PLANES = 4

PLANE_MY = 0
PLANE_OPP = 1
PLANE_FORBIDDEN = 2
PLANE_COLOR = 3


# ---------------------------------------------------------------------- #
# Feature extraction
# ---------------------------------------------------------------------- #
def board_to_planes(board: Board, rules: Rules, player: int | None = None) -> np.ndarray:
    """Encode ``board`` from the viewpoint of ``player`` into 4 planes.

    Args:
        board: current game state.
        rules: rule set providing ``forbidden_moves`` (empty for Normal).
        player: colour whose viewpoint is used; defaults to
            ``board.current_player``. Pass the *opponent* to get the
            mirrored encoding used for MCTS backup / training samples
            stored "from the side to move" (hard rule 3 keeps the sign
            convention in az/, this layer just provides both views).

    Returns:
        ``np.ndarray`` of shape ``(4, SIZE, SIZE)``, dtype float32, values
        in {0, 1}. Never imports torch.
    """
    if player is None:
        player = board.current_player
    if player not in (BLACK, _WHITE := 2):  # defensive: only two colours exist
        raise ValueError(f"player must be BLACK(1) or WHITE(2), got {player}")

    size = board.grid.size ** 0.5
    if size != int(size):  # pragma: no cover - board is always square
        raise AssertionError("board grid must be a square array")
    size = int(size)

    planes = np.zeros((N_PLANES, size, size), dtype=np.float32)

    grid2d = board.grid.reshape(size, size)
    opp = 3 - player  # BLACK(1) <-> WHITE(2)
    planes[PLANE_MY] = (grid2d == player).astype(np.float32)
    planes[PLANE_OPP] = (grid2d == opp).astype(np.float32)

    # Forbidden plane: computed *at this position, for this player*. The
    # rules interface defines forbidden_moves relative to the side to move,
    # so if the requested viewpoint differs from board.current_player we
    # temporarily flip the turn marker to query correctly.
    forbidden = _forbidden_for(rules, board, player)
    if forbidden:
        idx = np.asarray(forbidden, dtype=np.int64)
        planes[PLANE_FORBIDDEN].reshape(-1)[idx] = 1.0

    # color_to_move: 1 iff it is `player`'s turn on the actual board.
    if board.current_player == player:
        planes[PLANE_COLOR] = 1.0

    return planes


def _forbidden_for(rules: Rules, board: Board, player: int) -> list[int]:
    """Forbidden cells for ``player`` at ``board`` without mutating state.

    ``rules.forbidden_moves`` is defined for the side to move. We exploit
    play/undo round-trip safety: if it is already ``player``'s turn, query
    directly; otherwise place a temporary stone for the other colour, query,
    then undo. Undo restores grid, history and turn exactly (tested in T3).
    """
    if board.current_player == player:
        return rules.forbidden_moves(board)
    # Simulate the opponent playing an arbitrary legal cell to hand the turn
    # over to `player`. Any legal empty cell works; the forbidden plane is
    # only read while it is `player`'s real turn during search.
    # NOTE: callers use this during MCTS where the intermediate position is
    # transient. For Normal rules the result is [] either way.
    empties = np.flatnonzero(board.grid == 0)
    if empties.size == 0:
        return []
    r, c = divmod(int(empties[0]), board.grid.shape[0] ** 0.5 and 11)
    board.play(r, c)
    try:
        out = rules.forbidden_moves(board)
    finally:
        board.undo()
    return out


# ---------------------------------------------------------------------- #
# 8-fold symmetry (D4 group of the square)
# ---------------------------------------------------------------------- #
# Canonical enumeration of the 8 transforms, expressed via numpy ops that
# act on the last two axes:
#   k=0 identity          k=4 mirror horizontal axis (flip up-down)
#   k=1 rot90  (CCW)      k=5 mirror main diagonal   (transpose)
#   k=2 rot180            k=6 rot90 of k=4           (= anti-transpose)
#   k=3 rot270 (CW)       k=7 rot90 of k=5
# Exact permutation semantics are validated by tests (group closure,
# involution pairs, distinctness on asymmetric inputs).

def symmetry_transform(arr: np.ndarray, k: int) -> np.ndarray:
    """Apply D4 transform ``k`` (0..7) to the trailing two axes of ``arr``.

    Works for feature tensors ``(C, S, S)``, single planes ``(S, S)`` and
    policies reshaped to ``(S, S)`` -- one shared function so features and
    policy can never drift apart (T5 AC).
    """
    if arr.ndim < 2:
        raise ValueError("symmetry_transform needs at least 2 dimensions")
    k = int(k) % 8
    # Work on a view; each branch below returns a new array or a copy.
    v = arr
    # First handle the four reflections by pre-flipping vertically (k>=4),
    # then rotating by k%4 quarter turns CCW. This yields the full D4 group:
    # {rot^i, rot^i o flip} for i in 0..3.
    if k >= 4:
        v = np.flip(v, axis=-2)  # mirror across horizontal mid-line
        v = np.ascontiguousarray(v)
    rots = k % 4
    if rots:
        v = np.rot90(v, rots, axes=(-2, -1))
    return np.ascontiguousarray(v)


def inverse_transform(k: int) -> int:
    """Index of the group inverse of transform ``k`` in the enumeration
    ``T_k = rot90^(k%4) o flip_v`` (flip applied first, then rotation):

        k=0 id  k=1 rot90  k=2 rot180  k=3 rot270
        k=4 flip_v            k=5 rot90 o flip_v   (= main-diagonal mirror)
        k=6 rot180 o flip_v   k=7 rot270 o flip_v  (= anti-diagonal mirror)

    Rotations invert to 4-k quarters; every reflection of a square is an
    involution. Verified exhaustively by tests on random arrays.
    """
    k = int(k) % 8
    if k == 0:
        return 0
    if k < 4:  # pure rotations
        return 4 - k
    return k  # reflections are self-inverse


# ---------------------------------------------------------------------- #
# Single source of truth for the geometry: two cached permutation tables
# over flat board indices, built by applying ``symmetry_transform`` to an
# index grid. ``transform_policy_index`` and any future consumer read from
# these tables, so policy mapping can never drift from feature mapping.
# ---------------------------------------------------------------------- #
_CONTENT_PERM_CACHE: dict[tuple[int, int], np.ndarray] = {}
_FORWARD_PERM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def symmetry_permutation(k: int, size: int) -> np.ndarray:
    """Array ``perm`` of length ``size*size`` with ``perm[new_idx] = old_idx``.

    Exactly matches how ``symmetry_transform`` moves content: value at flat
    index ``old_idx`` ends up at flat index ``new_idx``. Cached per (k, size).
    """
    key = (int(k) % 8, int(size))
    perm = _CONTENT_PERM_CACHE.get(key)
    if perm is None:
        idx_grid = np.arange(size * size).reshape(size, size)
        perm = symmetry_transform(idx_grid, key[0]).reshape(-1).astype(np.int64)
        _CONTENT_PERM_CACHE[key] = perm
    return perm


def forward_permutation(k: int, size: int) -> np.ndarray:
    """Array ``fwd`` with ``fwd[old_idx] = new_idx`` (inverse of the above)."""
    key = (int(k) % 8, int(size))
    fwd = _FORWARD_PERM_CACHE.get(key)
    if fwd is None:
        perm = symmetry_permutation(k, size)
        fwd = np.empty_like(perm)
        fwd[perm] = np.arange(perm.size)
        _FORWARD_PERM_CACHE[key] = fwd
    return fwd


def transform_policy_index(idx: int, k: int, size: int) -> int:
    """Map a flat policy index ``r*size + c`` through transform ``k``.

    Semantics match ``symmetry_transform`` applied to the one-hot policy
    reshaped to ``(size, size)`` (T5 AC -- same geometry for features and
    policy). Derived from ``forward_permutation``, which is derived from the
    very same ``symmetry_transform`` that transforms the feature planes.
    """
    return int(forward_permutation(k, size)[int(idx)])


def augment_sample(
    planes: np.ndarray,
    policy: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply transform ``k`` to a feature tensor and its policy jointly.

    Args:
        planes: ``(4, S, S)`` from ``board_to_planes``.
        policy: ``(S*S,)`` probability vector.
    Returns:
        Transformed ``(planes', policy')`` -- both through the *same*
        ``symmetry_transform`` geometry (replay-buffer augmentation, T10).
    """
    size = planes.shape[-1]
    new_planes = symmetry_transform(planes, k)
    new_policy = symmetry_transform(policy.reshape(size, size), k).reshape(size * size)
    return new_planes, new_policy.astype(np.float32)
