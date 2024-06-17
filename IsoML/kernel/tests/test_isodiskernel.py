"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

from sklearn.datasets import load_iris
from isoml.kernel._isodiskernel import IsoDisKernel
import pytest


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_IsoDisKernel_fit(data):
    X = data[0]
    idk = IsoDisKernel(method="anne", n_estimators=200, max_samples="auto")
    idk.fit(X)
    assert idk.is_fitted_


def test_IsoDisKernel_similarity(data):
    X = data[0]
    idk = IsoDisKernel(method="anne", n_estimators=200, max_samples="auto")
    idk.fit(X)
    D_i = X[:10]
    D_j = X[-10:]
    similarity = idk.similarity(D_i, D_j)
    assert similarity.shape == (10, 10)


def test_IsoDisKernel_transform(data):
    X = data[0]
    idk = IsoDisKernel(method="anne", n_estimators=200, max_samples="auto")
    idk.fit(X)
    D_i = X[:10]
    D_j = X[-10:]
    transformed_D_i, transformed_D_j = idk.transform(D_i, D_j)
    assert transformed_D_i.shape == (10, idk.n_estimators * idk.max_samples_)
    assert transformed_D_j.shape == (10, idk.n_estimators * idk.max_samples_)
