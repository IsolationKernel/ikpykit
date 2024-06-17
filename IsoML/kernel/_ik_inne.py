"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

import numbers
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IK_INNE(TransformerMixin, BaseEstimator):
    """Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version uses iforest to split the data space and calculate Isolation
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

    import numpy as np


from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix


class iNN_IK:
    data = None
    centroid = []

    def __init__(self, psi, t):
        self.psi = psi
        self.t = t

    # def fit_transform(self, data):
    #     self.data = data
    #     self.centroid = []
    #     self.centroids_radius = []
    #     sn = self.data.shape[0]
    #     n, d = self.data.shape
    #     IDX = np.array([])  #column index
    #     V = []
    #     for i in range(self.t):
    #         subIndex = sample(range(sn), self.psi)
    #         self.centroid.append(subIndex)
    #         tdata = self.data[subIndex, :]
    #         tt_dis = cdist(tdata, tdata)
    #         radius = [] #restore centroids' radius
    #         for r_idx in range(self.psi):
    #             r = tt_dis[r_idx]
    #             r[r<0] = 0
    #             r = np.delete(r,r_idx)
    #             radius.append(np.min(r))
    #         self.centroids_radius.append(radius)
    #         nt_dis = cdist(tdata, self.data)
    #         centerIdx = np.argmin(nt_dis, axis=0)
    #         for j in range(n):
    #             V.append(int(nt_dis[centerIdx[j],j] <= radius[centerIdx[j]]))
    #         IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
    #     IDR = np.tile(range(n), self.t) #row index
    #     #V = np.ones(self.t * n) #value
    #     ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
    #     return ndata

    def fit(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        for i in range(self.t):
            subIndex = sample(range(sn), self.psi)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = []  # restore centroids' radius
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r < 0] = 0
                r = np.delete(r, r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)

    def transform(self, newdata):
        assert self.centroid != None, "invoke fit() first!"
        n, d = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata

    def fit(self, X, y=None):
        """Fit the model with the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        self._validate_hyperparameters()
        self._fit(X)
        return self

    def _fit(self, X):
        pass

    def transform(self, X):
        """Transform the given data into the Isolation Kernel feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features)
            The transformed data.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._transform(X)

    def _transform(self, X):
        pass
