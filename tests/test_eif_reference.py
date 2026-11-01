"""Verify genuine oblique splits and path scoring independently of SHP outcomes."""

import numpy as np
import pytest

from scripts.eif_reference import ExtendedForest, Node, correction


def test_analytical_oblique_tree_and_leaf_correction():
    tree = Node(0, 4, np.array([1.0, 1.0]), np.zeros(2), Node(1, 1), Node(1, 3))
    values = np.array([[-2.0, 1.0], [-2.0, 3.0], [1.0, -2.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        ExtendedForest.tree_paths(values, tree), [1, 1 + correction(3), 1, 1 + correction(3)]
    )
    assert correction(0) == correction(1) == 0
    assert correction(2) == 1


def test_forest_determinism_prefix_outliers_and_full_extension():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 6))
    model = ExtendedForest(ntrees=25, sample_size=64).fit(x)
    other = ExtendedForest(ntrees=25, sample_size=64).fit(x)
    np.testing.assert_array_equal(model.score(x), other.score(x))
    np.testing.assert_allclose(model.score(x)[:20], model.score(x[:20]), atol=1e-12)
    assert all(np.count_nonzero(t.normal) == 6 for t in model.trees)
    assert model.score(np.full((1, 6), 20)).item() > np.median(model.score(x))
    assert np.all((model.score(x) > 0) & (model.score(x) <= 1))
    with pytest.raises(ValueError):
        model.score(np.zeros((3, 5)))


def test_duplicate_points_and_empty_branches_remain_finite():
    model = ExtendedForest(ntrees=3, sample_size=8).fit(np.ones((8, 2)))
    assert np.isfinite(model.score(np.array([[1.0, 1.0], [100.0, -10.0]]))).all()
