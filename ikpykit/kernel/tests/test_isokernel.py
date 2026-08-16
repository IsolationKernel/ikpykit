"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from ikpykit import IsoKernel

method = ["inne", "anne", "iforest"]


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize("method", method)
def test_IsoKernel_fit(data, method):
    X = data[0]
    ik = IsoKernel(method=method, n_estimators=200)
    ik.fit(X)
    assert ik.is_fitted_


@pytest.mark.parametrize("method", method)
def test_IsoKernel_similarity(data, method):
    X = data[0]
    ik = IsoKernel(method=method, n_estimators=200)
    ik.fit(X)
    similarity = ik.similarity(X)
    assert similarity.shape == (X.shape[0], X.shape[0])


@pytest.mark.parametrize("method", method)
def test_IsoKernel_transform(data, method):
    X = data[0]
    max_samples = 16
    ik = IsoKernel(method=method, max_samples=max_samples)
    ik.fit(X)
    transformed_X = ik.transform(X)
    assert transformed_X.shape == (X.shape[0], ik.n_estimators * max_samples)


@pytest.mark.parametrize("method", method)
def test_IsoKernel_transform_is_one_hot_per_estimator(data, method):
    """Every estimator must place a sample in exactly one of its cells.

    `inne` is excluded from the lower bound: a point outside every hypersphere
    falls in no cell, which is that method's way of saying "unlike anything seen".
    """
    X = data[0]
    max_samples, n_estimators = 8, 20
    ik = IsoKernel(
        method=method,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=0,
    ).fit(X)
    blocks = ik.transform(X).toarray().reshape(X.shape[0], n_estimators, max_samples)
    per_block = blocks.sum(axis=2)

    assert per_block.max() == 1
    if method != "inne":
        assert per_block.min() == 1


@pytest.mark.parametrize("method", method)
def test_IsoKernel_similarity_is_higher_in_sparse_regions(method):
    """The property the kernel exists for: isolation is easier where it is empty.

    Two neighbours out on their own should score as more similar than two
    neighbours inside a crowd, even though the sparse pair is further apart.
    """
    rng = np.random.RandomState(0)
    crowd = rng.randn(200, 2) * 0.3
    outliers = np.array([[6.0, 6.0], [6.4, 6.4]])
    X = np.vstack([crowd, outliers])

    similarity = (
        IsoKernel(method=method, n_estimators=300, max_samples=16, random_state=0)
        .fit(X)
        .similarity(X)
    )

    assert similarity[-1, -2] > similarity[0, 1]


@pytest.mark.parametrize("method", method)
def test_IsoKernel_is_reproducible(data, method):
    X = data[0]

    def embed(random_state):
        ik = IsoKernel(
            method=method, n_estimators=20, max_samples=8, random_state=random_state
        )
        return ik.fit(X).transform(X)

    assert (embed(42) != embed(42)).nnz == 0
    assert (embed(42) != embed(7)).nnz > 0


def test_IsoKernel_rejects_unknown_method(data):
    with pytest.raises(ValueError, match="is not supported"):
        IsoKernel(method="nope").fit(data[0])
