import numpy as np
import time


import numpy as np
from isoml.kernel import IsoKernel
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.utils import check_array
from collections.abc import Iterable
from scipy import sparse as sp

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IKDC(BaseEstimator, ClusterMixin):
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

    tau : float
        The threshold value for stopping the clustering process.

    v : float
        The decay factor for reducing the threshold value.

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
        is_post_process=True,
        init_id=None,
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
        self.init_center_id = init_id
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
        GP = []
        self.clusters_ = []
        n_samples, n_features = X.shape
        data_index = np.array(range(n_samples))
        if self.init_center_id is None:
            # find modes based on sample
            rnd = check_random_state(self.random_state)
            samples_index = rnd.choice(n_samples, self.n_init_samples, replace=False)
            init_center_id = self._find_mode(X[samples_index])
            init_center_id = samples_index[init_center_id]
        else:
            init_center_id = self.init_center_id

        self.clusters_ = [KCluster(i, init_center_id[i]) for i in range(self.k)]
        for i in range(self.k):
            self.clusters_[i].add_points(init_center_id[i], X[init_center_id[i]])

        data_index = np.delete(data_index, init_center_id)

        C_mean = sp.vstack([c.kernel_mean for c in self.clusters_])
        S = np.max(safe_sparse_dot(X[data_index], C_mean.T), axis=1)

        r = np.max(S)

        # linking points
        while len(data_index) > 0:
            C_mean = sp.vstack([c.kernel_mean for c in self.clusters_])
            similarity = safe_sparse_dot(X[data_index], C_mean.T)
            T = np.argmax(similarity, axis=1)
            S = similarity[:, T]
            r = self.v * r

            if np.sum(S) == 0 or r < 0.00001:
                break

            self.it_ += 1

            DI = np.zeros_like(T)
            for i in range(self.k):
                I = np.logical_and(T == i, S > r)
                if np.sum(I) > 0:
                    self.clusters_[i].add_points(data_index[I], X[data_index][I])
                    DI += I

            for ci in self.clusters_:
                c_i_index = ci.points
                x = np.argmax(
                    safe_sparse_dot(
                        X[data_index][c_i_index],
                        X[data_index][c_i_index].sum().T,
                    )
                )
                ci.set_center(data_index[x])

            data_index = np.delete(data_index, np.where(DI > 0)[0])

    def _get_labels(self, X):
        check_is_fitted(self)
        n = X.shape[0]
        labels = np.zeros(n)
        for i, c in enumerate(self.clusters_):
            for p in c.points_:
                labels[p] = i
        return labels

    def _post_process(self, X):
        Th = np.ceil(X.shape[0] * 0.01)
        for _ in range(100):
            original_labels = self._get_labels(X)
            c_mean = sp.vstack([c.kernel_mean for c in self.clusters_])
            new_labels = np.argmax(safe_sparse_dot(X, c_mean.T), axis=1)
            if np.sum() < Th or len(np.unique(new_labels)) < self.k:
                break
            or_label = original_labels[new_labels != original_labels]
            new_label = new_labels[new_labels != original_labels]
            for x in range(len(or_label)):
                self._change_points(
                    self.clusters_[or_label[x]],
                    self.clusters_[new_label[x]],
                    or_label[x],
                    X,
                )
            self._update_centers(X)

        return self

    def _find_mode(self, X):
        density = safe_sparse_dot(X, X.mean(axis=0).T)
        ik_dist = euclidean_distances(X, X, squared=True)

        # Density = Density.flatten()
        # IKDist = IKDist.flatten()

        density = self._get_lc(ik_dist, density)

        maxd = np.max(ik_dist)
        n_samples = ik_dist.shape[1]
        min_dist = np.zeros_like(density)
        sort_density = np.argsort(density)[::-1]

        min_dist[sort_density[0]] = -1
        nneigh = np.zeros_like(sort_density)

        for ii in range(1, n_samples):
            min_dist[sort_density[ii]] = maxd
            for jj in range(ii):
                if (
                    ik_dist[sort_density[ii], sort_density[jj]]
                    < min_dist[sort_density[ii]]
                ):
                    min_dist[sort_density[ii]] = ik_dist[
                        sort_density[ii], sort_density[jj]
                    ]
                    nneigh[sort_density[ii]] = sort_density[jj]

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

        N = density.shape[0]
        LC = np.zeros(N)
        for i in range(N):
            inx = np.argsort(dist[i])
            knn = inx[: self.kn]  # K-nearest-neighbors of instance i
            LC[i] = np.sum(density[i] > density[knn])

        return LC

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
    def __init__(self, id: int, center: int) -> None:
        self.id = id
        self.center = center
        self.kernel_mean_ = None
        self.points_ = []

    def add_points(self, points, X):
        self.increment_kernel_mean_(X)
        if isinstance(points, np.integer):
            self.points_.append(points)
        elif isinstance(points, Iterable):
            self.points_.extend(points)

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
