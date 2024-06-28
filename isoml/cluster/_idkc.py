"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

import numpy as np
from isoml.kernel import IsoKernel
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.utils import check_array
from collections.abc import Iterable
from scipy import sparse as sp


class IKDC(BaseEstimator, ClusterMixin):
    """Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version uses anne to split the data space and calculate Isolation
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

    method : str
        The method used to calculate the Isolation Kernel. Possible values are 'inne','anne' and 'iforest'.

    k : int
        The number of clusters to form.

    kn : int
        The number of nearest neighbors to consider when calculating the local contrast.

    v : float
        The decay factor for reducing the threshold value.

    n_init_samples : int
        The number of samples to use for initializing the cluster centers.

    init_center : int or None, default=None
        The index of the initial cluster center. If None, the center will be automatically selected.

    is_post_process : bool, default=True
        Whether to perform post-processing to refine the clusters.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

    References
    ----------
    .. [1] Ye Zhu, Kai Ming Ting (2023). Kernel-based Clustering via Isolation Distributional Kernel. Information Systems.
    """

    def __init__(
        self,
        n_estimators,
        max_samples,
        method,
        k,
        kn,
        v,
        n_init_samples,
        init_center=None,
        is_post_process=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.method = method
        self.k = k
        self.v = v
        self.kn = kn
        self.n_init_samples = n_init_samples
        self.is_post_process = is_post_process
        self.init_center = init_center
        self.random_state = random_state
        self.clusters_ = []
        self.it_ = 0
        self.labels_ = None

    @property
    def n_it(self):
        return self.it_

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
        isokernel = IsoKernel(
            method=self.method,
            max_samples=self.max_samples,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        data_ik = isokernel.fit_transform(X)
        self._fit(data_ik)
        if self.is_post_process:
            self._post_process(data_ik)
        self.is_fitted_ = True
        self.labels_ = self._get_labels(X)
        return self

    def _fit(self, X):
        n_samples, n_features = X.shape
        data_index = np.array(range(n_samples))
        if self.init_center is None:
            # find modes based on sample
            rnd = check_random_state(self.random_state)
            samples_index = rnd.choice(n_samples, self.n_init_samples, replace=False)
            init_center_id = self._find_mode(X[samples_index])
            init_center = samples_index[init_center_id]
        else:
            init_center = self.init_center

        self.clusters_ = [KCluster(i) for i in range(self.k)]
        for i in range(self.k):
            self.clusters_[i].add_points(init_center[i], X[init_center[i]])
        data_index = np.delete(data_index, init_center)

        # linking points
        while len(data_index) > 0:
            c_mean = sp.vstack([c.kernel_mean for c in self.clusters_])
            similarity = safe_sparse_dot(X[data_index], c_mean.T)
            tmp_labels = np.argmax(similarity, axis=1)
            tmp_similarity = similarity[:, tmp_labels]
            if self.it_ == 0:
                r = np.max(tmp_similarity)
            r = self.v * r
            if np.sum(tmp_similarity) == 0 or r < 0.00001:
                break

            DI = np.zeros_like(tmp_labels)
            for i in range(self.k):
                I = np.logical_and(tmp_labels == i, tmp_similarity > r)
                if np.sum(I) > 0:
                    self.clusters_[i].add_points(data_index[I], X[data_index][I])
                    DI += I

            self._update_centers(X)
            data_index = np.delete(data_index, np.where(DI > 0)[0])
            self.it_ += 1
        return self

    def _post_process(self, X):
        th = np.ceil(X.shape[0] * 0.01)
        for _ in range(100):
            original_labels = self._get_labels(X)
            data_index = np.array(range(X.shape[0]))
            c_mean = sp.vstack([c_k.kernel_mean for c_k in self.clusters_])
            new_labels = np.argmax(safe_sparse_dot(X, c_mean.T), axis=1)
            change_id = new_labels != original_labels
            if np.sum(change_id) < th or len(np.unique(new_labels)) < self.k:
                break
            or_label, new_label = original_labels[change_id], new_labels[change_id]
            data_index = data_index[change_id]
            for l in range(len(or_label)):
                self._change_points(
                    self.clusters_[or_label[l]],
                    self.clusters_[new_label[l]],
                    data_index[l],
                    X,
                )
            self._update_centers(X)

        return self

    def _find_mode(self, X):
        density = safe_sparse_dot(X, X.mean(axis=0).T)
        dists = euclidean_distances(X, X, squared=True)

        density = self._get_lc(dists, density)

        maxd = np.max(dists)
        n_samples = dists.shape[1]
        min_dist = np.zeros_like(density)
        sort_density = np.argsort(density)[::-1]

        min_dist[sort_density[0]] = -1
        nneigh = np.zeros_like(sort_density)

        for i in range(1, n_samples):  # TODO: check this fuction
            min_dist[sort_density[i]] = maxd
            for j in range(i):
                dist = dists[sort_density[i], sort_density[j]]
                if dist < min_dist[sort_density[i]]:
                    min_dist[sort_density[i]] = dist
                    nneigh[sort_density[i]] = sort_density[j]

        min_dist[sort_density[0]] = np.max(min_dist)

        density = np.argsort(density) + 0.0000000001
        min_dist = np.argsort(min_dist) + 0.0000000001

        Mult = density * min_dist
        ISortMult = np.argsort(Mult)[::-1]
        ID = ISortMult[: self.k]
        return ID

    def _get_lc(self, dist, density):
        # input:
        # dist: distance matrix (N*N) of a dataset
        # density: density vector (N*1) of the same dataset
        # k: k parameter for KNN

        # output:
        # LC: Local Contrast

        n = density.shape[0]
        lc = np.zeros(n)
        for i in range(n):
            inx = np.argsort(dist[i])
            knn = inx[: self.kn]  # K-nearest-neighbors of instance i
            lc[i] = np.sum(density[i] > density[knn])
        return lc

    def _get_labels(self, X):
        check_is_fitted(self)
        n = X.shape[0]
        labels = np.zeros(n)
        for i, c in enumerate(self.clusters_):
            for p in c.points_:
                labels[p] = i
        return labels

    def _change_points(self, c1, c2, p, X):
        c2.add_points(p, X[p])
        c1.delete_points(p, X[p])
        return self

    def _update_centers(self, X):
        for ci in self.clusters_:
            c_i_index = ci.points
            x = np.argmax(
                safe_sparse_dot(
                    X[c_i_index],
                    ci.kernel_mean.T,
                )
            )
            ci.set_center(c_i_index[x])
        return self


class KCluster(object):
    def __init__(self, id: int) -> None:
        self.id = id
        self.center = None
        self.kernel_mean_ = None
        self.points_ = []
        self.center = None

    def add_points(self, ids, X):
        self.increment_kernel_mean_(X)
        if isinstance(ids, np.integer):
            if self.center is None:
                self.center = ids
            self.points_.append(ids)
        elif isinstance(ids, Iterable):
            if self.center is None:
                raise ValueError("Cluster is not initialized.")
            self.points_.extend(ids)

    def set_center(self, center):
        self.center = center

    def delete_points(self, points):
        self.delete_kernel_mean_(points)
        if isinstance(points, np.integer):
            self.points_.remove(points)
        elif isinstance(points, Iterable):
            for p in points:
                self.points_.remove(p)

    def delete_kernel_mean_(self, X):
        if self.kernel_mean_ is None:
            raise ValueError("Kernel mean is not initialized.")
        else:
            self.kernel_mean_ = (self.kernel_mean_ * self.n_points - X.sum(axis=0)).sum(
                axis=0
            ) / (self.n_points - X.shape[0])

    def increment_kernel_mean_(self, X):
        if self.kernel_mean_ is None:
            self.kernel_mean_ = X
        self.kernel_mean_ = sp.vstack((self.kernel_mean_ * self.n_points, X)).sum(
            axis=0
        ) / (self.n_points + X.shape[0])

    @property
    def n_points(self):
        return len(self.points_)

    @property
    def points(self):
        return self.points_

    @property
    def kernel_mean(self):
        return self.kernel_mean_
