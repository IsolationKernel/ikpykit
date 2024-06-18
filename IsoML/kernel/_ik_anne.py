"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IK_ANNE(TransformerMixin, BaseEstimator):
    """Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version uses Voronoi diagrams to split the data space and calculate Isolation
    kernel Similarity. Based on this implementation, the feature
    in the Isolation kernel space is the index of the cell in Voronoi diagrams. Each
    point is represented as a binary vector such that only the cell the point falling
    into is 1.

    Parameters
    ----------

    n_estimators : int
        The number of base estimators in the ensemble.


    max_samples : int
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    References
    ----------
    .. [1] Qin, X., Ting, K.M., Zhu, Y. and Lee, V.C.
    "Nearest-neighbour-induced isolation similarity and its impact on density-based clustering".
    In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, 2019, July, pp. 4755-4762
    """

    def __init__(self, n_estimators, max_samples, random_state=None) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model on data X.
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        Returns
        -------
        self : object
        """

        X = check_array(X)
        self.max_samples_ = self.max_samples
        n_samples = X.shape[0]
        self.max_samples = min(self.max_samples_, n_samples)
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        self.center_index_set = np.empty(
            (self.n_estimators, self.max_samples_), dtype=int
        )
        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            center_index = rnd.choice(n_samples, self.max_samples_, replace=False)
            self.center_index_set[i] = center_index

        self.unique_index = np.unique(self.center_index_set)
        self.center_data = X[self.unique_index]

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Compute the isolation kernel feature of X.
        Parameters
        ----------
        X: array-like of shape (n_instances, n_features)
            The input instances.
        Returns
        -------
        The finite binary features based on the kernel feature map.
        The features are organized as a n_instances by psi*t matrix.
        """

        check_is_fitted(self)
        X = check_array(X)
        n, m = X.shape
        X_dists = euclidean_distances(X, self.center_data)
        embedding = np.zeros([n, self.max_samples_ * self.n_estimators])

        for i in range(n):
            mapping_array = np.zeros(self.unique_index.max() + 1, dtype=X_dists.dtype)
            mapping_array[self.unique_index] = X_dists[i]
            x_center_dist_mat = mapping_array[self.center_index_set]
            nearest_center_index = np.argmin(x_center_dist_mat, axis=1)
            flatten_index = nearest_center_index + self.max_samples_ * np.arange(
                self.n_estimators
            )
            embedding[i, flatten_index] = 1.0
        return embedding
