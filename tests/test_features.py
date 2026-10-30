"""Tests for gomoku/features.py (Task 5).

Covers: plane layout & values, forbidden plane zero for Normal, the 8 D4
transforms (distinctness, group closure, self-inverse pairs, rot90/rot270
inverses), and the key AC that policy transform agrees with feature transform.
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.features import (
    N_PLANES,
    PLANE_COLOR,
    PLANE_FORBIDDEN,
    PLANE_MY,
    PLANE_OPP,
    augment_sample,
    board_to_planes,
    inverse_transform,
    symmetry_transform,
    transform_policy_index,
)
from gomoku.rules import get_rules


@pytest.fixture()
def rules():
    return get_rules("normal")


# ---------------------------------------------------------------------- #
# board_to_planes
# ---------------------------------------------------------------------- #
class TestPlanes:
    def test_shape_and_dtype(self, rules):
        b = Board()
        p = board_to_planes(b, rules)
        assert p.shape == (N_PLANES, SIZE, SIZE)
        assert p.dtype == np.float32
        assert set(np.unique(p)) <= {0.0, 1.0}

    def test_empty_board_color_plane(self, rules):
        b = Board()  # black to move
        p = board_to_planes(b, rules, BLACK)
        assert p[PLANE_MY].sum() == 0
        assert p[PLANE_OPP].sum() == 0
        assert p[PLANE_FORBIDDEN].sum() == 0  # Normal: no forbidden ever
        assert np.all(p[PLANE_COLOR] == 1.0)  # it IS black's turn

    def test_stone_placement_viewpoints(self, rules):
        b = Board()
        b.play(5, 5)  # black stone; now white to move
        pb = board_to_planes(b, rules, BLACK)
        pw = board_to_planes(b, rules, WHITE)
        assert pb[PLANE_MY][5, 5] == 1 and pb[PLANE_OPP][5, 5] == 0
        assert pw[PLANE_MY][5, 5] == 0 and pw[PLANE_OPP][5, 5] == 1
        # color_to_move is a property of the *board*, not the viewpoint
        assert np.all(pb[PLANE_COLOR] == 0.0)  # black does not move next
        assert np.all(pw[PLANE_COLOR] == 1.0)

    def test_forbidden_zero_for_normal(self, rules):
        # Even in a position with many stones, Normal must yield an all-zero
        # forbidden plane (AC: "plane forbidden = 0 khi rule set không cấm").
        b = Board()
        moves = [(3, 3), (3, 4), (4, 4), (5, 5), (4, 5), (6, 6)]
        for r, c in moves:
            b.play(r, c)
        for player in (BLACK, WHITE):
            p = board_to_planes(b, rules, player)
            assert p[PLANE_FORBIDDEN].sum() == 0

    def test_invalid_player_raises(self, rules):
        b = Board()
        with pytest.raises(ValueError):
            board_to_planes(b, rules, 7)

    def test_no_torch_import(self):
        import subprocess
        import sys

        code = (
            "import sys; import gomoku.features; "
            "assert 'torch' not in sys.modules, 'gomoku/ must not import torch'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------- #
# symmetry_transform on planes
# ---------------------------------------------------------------------- #
def asymmetric_board() -> tuple[Board, int]:
    """A board whose encoding has no symmetry at all."""
    b = Board()
    seq = [BLACK, WHITE]
    placements = [(0, 0), (1, 2), (3, 1), (7, 9), (10, 10), (2, 8)]
    for (r, c) in placements:
        b.play(r, c)
    return b, b.current_player


class TestSymmetry:
    def test_k0_is_identity(self):
        a = np.arange(121, dtype=np.float32).reshape(11, 11)
        out = symmetry_transform(a, 0)
        assert np.array_equal(out, a)

    def test_eight_variants_distinct_on_features(self, rules):
        b, player = asymmetric_board()
        p = board_to_planes(b, rules, player)
        seen = {symmetry_transform(p, k).tobytes() for k in range(8)}
        assert len(seen) == 8, "the 8 transforms must be distinct group elements"

    def test_rot90_rot270_are_inverses(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, SIZE, SIZE))
        back = symmetry_transform(symmetry_transform(a, 1), 3)
        assert np.allclose(back, a)

    def test_involution_pairs_apply_twice_identity(self):
        # Self-inverse elements: rot180 (k=2) and the four reflections
        # (k=4..7 are rot^i o flip; each reflection has order 2).
        rng = np.random.default_rng(1)
        a = rng.standard_normal((SIZE, SIZE))
        for k in (2, 4, 5, 6, 7):
            twice = symmetry_transform(symmetry_transform(a, k), k)
            assert np.allclose(twice, a), f"k={k} should be an involution"

    def test_group_closure_inverse_table(self):
        # transform_{inv(k)} after transform_k == identity for every k,
        # proving inverse_transform matches the actual geometry.
        rng = np.random.default_rng(2)
        a = rng.standard_normal((4, SIZE, SIZE))
        for k in range(8):
            ik = inverse_transform(k)
            assert np.allclose(
                symmetry_transform(symmetry_transform(a, k), ik), a
            ), f"inverse of k={k} wrong"

    def test_transform_preserves_values_and_shape(self, rules):
        b, player = asymmetric_board()
        p = board_to_planes(b, rules, player)
        for k in range(8):
            t = symmetry_transform(p, k)
            assert t.shape == p.shape
            assert np.array_equal(np.sort(t.ravel()), np.sort(p.ravel()))

    def test_input_not_mutated(self, rules):
        b, player = asymmetric_board()
        p = board_to_planes(b, rules, player)
        before = p.copy()
        symmetry_transform(p, 3)
        assert np.array_equal(p, before)

    def test_ndim_guard(self):
        with pytest.raises(ValueError):
            symmetry_transform(np.zeros(11), 1)


# ---------------------------------------------------------------------- #
# policy <-> feature consistency (core AC)
# ---------------------------------------------------------------------- #
class TestPolicyConsistency:
    @pytest.mark.parametrize("k", range(8))
    def test_onehot_policy_matches_feature_transform(self, rules, k):
        """One-hot at (r,c) transformed by k lands exactly on the cell where
        the *feature* plane's single stone lands under the same k."""
        b = Board()
        r, c = 2, 7
        b.play(r, c)  # black stone at (2,7); viewpoint black
        pol = np.zeros(SIZE * SIZE, dtype=np.float32)
        pol[r * SIZE + c] = 1.0
        tp = symmetry_transform(pol.reshape(SIZE, SIZE), k)
        _, new_idx = divmod(int(np.argmax(tp)), SIZE) if False else (0, int(np.argmax(tp)))
        tf = symmetry_transform(board_to_planes(b, rules, BLACK), k)
        # my_stones plane of transformed features marks the transformed stone
        ys, xs = np.nonzero(tf[PLANE_MY])
        assert (ys * SIZE + xs) == new_idx

    @pytest.mark.parametrize("k", range(8))
    def test_transform_policy_index_agrees(self, rules, k):
        b = Board()
        idxs = [0 * SIZE + 0, 2 * SIZE + 7, 10 * SIZE + 10, 5 * SIZE + 1]
        for idx in idxs:
            onehot = np.zeros(SIZE * SIZE, dtype=np.float32)
            onehot[idx] = 1.0
            got = int(np.argmax(symmetry_transform(onehot.reshape(SIZE, SIZE), k)))
            assert transform_policy_index(idx, k, SIZE) == got

    def test_full_distribution_permutation(self):
        # A soft policy must stay a probability distribution with permuted
        # mass: sum preserved, and mass at source idx equals mass at mapped idx.
        rng = np.random.default_rng(3)
        pol = rng.dirichlet(np.ones(SIZE * SIZE)).astype(np.float32)
        for k in range(8):
            t = symmetry_transform(pol.reshape(SIZE, SIZE), k).reshape(-1)
            assert t.sum == pytest.approx(1.0) or np.isclose(t.sum(), 1.0)
            for idx in (0, 12, 60, 120):
                assert t[transform_policy_index(idx, k, SIZE)] == pol[idx]

    def test_augment_sample_joint(self, rules):
        b, player = asymmetric_board()
        p = board_to_planes(b, rules, player)
        pol = np.full(SIZE * SIZE, 1.0 / (SIZE * SIZE), dtype=np.float32)
        # uniform policy is invariant; use a peaked one instead
        pol[:] = 0.0
        pol[4 * SIZE + 9] = 1.0
        ap,apol = augment_sample(p, pol, 5)
        assert ap.shape == p.shape
        expected = transform_policy_index(4 * SIZE + 9, 5, SIZE)
        assert int(np.argmax(apol)) == expected
        # augmenting with k=0 returns the original sample
        bp, bpol = augment_sample(p, pol, 0)
        assert np.array_equal(bp, p) and np.array_equal(bpol, pol)


# ---------------------------------------------------------------------- #
# forbidden-plane semantics placeholder (real Renju tests come with T24)
# ---------------------------------------------------------------------- #
def test_forbidden_hook_uses_rules_interface(monkeypatch, rules):
    """board_to_planes reads forbidden cells through rules.forbidden_moves,
    and only for the side to move (viewpoint snapshot must not query a
    different colour's restrictions)."""
    calls = []
    orig = rules.forbidden_moves

    def spy(b):
        calls.append(b.current_player)
        return orig(b)

    monkeypatch.setattr(rules, "forbidden_moves", spy)
    b = Board()
    b.play(5, 5)  # now WHITE to move
    board_to_planes(b, rules, WHITE)   # side to move -> consulted
    assert calls == [WHITE]
    board_to_planes(b, rules, BLACK)   # not side to move -> not consulted
    assert len(calls) == 1
