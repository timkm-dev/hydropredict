"""Small NumPy implementation of fully extended isolation trees.

Algorithm: Hariri et al., https://arxiv.org/abs/1811.02141 and SHP Algorithms 2/3.
Gaussian hyperplane normals, uniform bounding-box intercepts, full extension.
Independent implementation; not an exact reproduction of isotree 0.1.16.
"""

from dataclasses import dataclass

import numpy as np


def correction(size):
    if size <= 1:
        return 0.0
    if size == 2:
        return 1.0
    return 2 * (np.log(size - 1) + np.euler_gamma) - 2 * (size - 1) / size


@dataclass
class Node:
    depth: int
    size: int
    normal: object = None
    intercept: object = None
    left: object = None
    right: object = None


class ExtendedForest:
    def __init__(self, ntrees=500, sample_size=2048, seed=42):
        self.ntrees = ntrees
        self.sample_size = sample_size
        self.seed = seed

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Finite two-dimensional training data with at least two rows required")
        if self.ntrees < 1 or self.sample_size < 2:
            raise ValueError("Invalid tree/sample count")
        self.dim = values.shape[1]
        self.sample = min(self.sample_size, len(values))
        self.limit = int(np.ceil(np.log2(self.sample)))
        rng = np.random.default_rng(self.seed)

        def build(x, depth):
            node = Node(depth, len(x))
            if depth < self.limit and len(x) > 1:
                node.normal = rng.normal(size=self.dim)
                node.intercept = rng.uniform(x.min(axis=0), x.max(axis=0))
                mask = ((x - node.intercept) @ node.normal) < 0
                node.left = build(x[mask], depth + 1)
                node.right = build(x[~mask], depth + 1)
            return node

        self.trees = [
            build(values[rng.choice(len(values), self.sample, replace=False)], 0)
            for _ in range(self.ntrees)
        ]
        return self

    @staticmethod
    def tree_paths(values, tree):
        paths = np.empty(len(values))
        stack = [(tree, np.arange(len(values)))]
        while stack:
            node, indices = stack.pop()
            if not len(indices):
                continue
            if node.normal is None:
                paths[indices] = node.depth + correction(node.size)
            else:
                mask = ((values[indices] - node.intercept) @ node.normal) < 0
                stack.extend(((node.left, indices[mask]), (node.right, indices[~mask])))
        return paths

    def score(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.dim or not np.isfinite(values).all():
            raise ValueError("Invalid scoring dimensions or values")
        paths = np.zeros(len(values))
        for tree in self.trees:
            paths += self.tree_paths(values, tree)
        return np.exp2(-paths / (self.ntrees * correction(self.sample)))
