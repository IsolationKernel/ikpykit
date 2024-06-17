"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

from sklearn.datasets import load_iris
from isoml.kernel._isokernel import IsoKernel
import pytest


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_IsoKernel_fit(data):
    X = data[0]
    ik = IsoKernel(method="anne", n_estimators=200, max_samples="auto")
    ik.fit(X)
    assert ik.is_fitted_


def test_IsoKernel_similarity(data):
    X = data[0]
    ik = IsoKernel(method="anne", n_estimators=200, max_samples="auto")
    ik.fit(X)
    similarity = ik.similarity(X)
    assert similarity.shape == (X.shape[0], X.shape[0])


def test_IsoKernel_transform(data):
    X = data[0]
    ik = IsoKernel(method="anne", n_estimators=200, max_samples="auto")
    ik.fit(X)
    transformed_X = ik.transform(X)
    assert transformed_X.shape == (X.shape[0], ik.n_estimators * ik.max_samples_)
