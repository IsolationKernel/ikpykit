"""
isoml (c) by Xin Han

isoml is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from isoml.kernel import IsoKernel
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from collections.abc import Iterable
from scipy import sparse as sp

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IKAT(BaseEstimator, ClusterMixin):
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
    .. [1] Wang, Y., Wang, Z., Ting, K. M., & Shang, Y. (2024).
    A Principled Distributional Approach to Trajectory Similarity Measurement and
    its Application to Anomaly Detection. Journal of Artificial Intelligence Research, 79, 865-893.
    """

    def __init__(self, n_estimators, max_samples, method, tau, v, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.method = method
        self.tau = tau
        self.v = v
        self.random_state = random_state
        self.clusters_ = []
        self.labels_ = None

    @property
    def clusters(self):
        check_is_fitted(self)
        return self.clusters_

    @property
    def centers(self):
        check_is_fitted(self)
        return [c.center for c in self.clusters_]

    @property
    def n_classes(self):
        check_is_fitted(self)
        return len(self.clusters_)

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
            max_samples=self.max_samples,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            method=self.method,
        )
        ndata = isokernel.fit_transform(X)
        self._fit(ndata)
        self.is_fitted_ = True
        self.labels_ = self._get_labels(X)
        return self

    def _fit(self, X):
        k = 1
        n = X.shape[0]
        data_index = np.array(range(n))
        while len(data_index) > 0:
            center_id = np.argmax(
                safe_sparse_dot(X[data_index], X[data_index].mean(axis=0).T)
            )
            c_k = KCluster(k, center_id)
            c_k.add_points(data_index[center_id], X[data_index][center_id])
            self.clusters_.append(c_k)
            data_index = np.delete(data_index, center_id)
            assert self._get_n_points() == n - len(data_index)

            if len(data_index) == 0:
                break

            nn_dists = (
                safe_sparse_dot(X[data_index], X[data_index].mean(axis=0).T)
                / self.n_estimators
            )
            nn_index = np.argmax(nn_dists)
            nn_dist = nn_dists[nn_index]
            c_k.add_points(data_index[nn_index], X[data_index][nn_index])
            data_index = np.delete(data_index, nn_index)
            assert self._get_n_points() == n - len(data_index)

            r = (1 - self.v) * nn_dist
            if r <= self.tau:
                print("break")
                break

            while r > self.tau:
                S = (
                    safe_sparse_dot(X[data_index], c_k.kernel_mean.T)
                    / self.n_estimators
                )
                x = np.where(S > r)[0]  # Use [0] to get the indices as a 1D array
                if len(x) == 0:
                    break
                c_k.add_points([data_index[j] for j in x], X[data_index][x])
                data_index = np.delete(data_index, x)
                r = (1 - self.v) * r
            assert self._get_n_points() == n - len(data_index)
            k += 1
        return self

    def _get_labels(self, X):
        check_is_fitted(self)
        n = X.shape[0]
        labels = np.zeros(n)
        for i, c in enumerate(self.clusters_):
            for p in c.points_:
                labels[p] = i
        return labels

    def _get_n_points(self):
        check_is_fitted(self)
        n_points = sum([c.n_points for c in self.clusters_])
        return n_points


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
    def kernel_mean(self):
        return self.kernel_mean_
