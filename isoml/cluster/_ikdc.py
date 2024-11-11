"""
isoml (c) by Xin Han

isoml is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, check_random_state, check_array
from sklearn.metrics._pairwise_distances_reduction import ArgKmin
from isoml.kernel import IsoKernel
from ._kcluster import KCluster


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
        self.kn = kn
        self.v = v
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
        n_samples, _ = X.shape
        data_index = np.arange(n_samples)
        init_center = self._initialize_centers(X, data_index)
        self.clusters_ = [KCluster(i) for i in range(self.k)]
        for i in range(self.k):
            self.clusters_[i].add_points(init_center[i], X[init_center[i]])
        data_index = np.delete(data_index, init_center)

        while data_index.size > 0:
            c_mean = np.vstack([c.kernel_mean for c in self.clusters_])
            similarity = safe_sparse_dot(X[data_index], c_mean.T)
            tmp_labels = np.argmax(similarity, axis=1).A1
            tmp_similarity = np.max(similarity, axis=1).A1
            if self.it_ == 0:
                r = np.max(tmp_similarity)
            r *= self.v
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

    def _initialize_centers(self, X, data_index):
        if self.init_center is None:
            rnd = check_random_state(self.random_state)
            samples_index = rnd.choice(
                data_index.size, self.n_init_samples, replace=False
            )
            seeds_id = self._get_seeds(X[samples_index])
            return samples_index[seeds_id]
        return self.init_center

    def _post_process(self, X):
        th = np.ceil(X.shape[0] * 0.01)
        for _ in range(100):
            old_labels = self._get_labels(X)
            data_index = np.arange(X.shape[0])
            c_mean = np.vstack([c_k.kernel_mean for c_k in self.clusters_])
            new_labels = np.argmax(safe_sparse_dot(X, c_mean.T), axis=1).A1
            change_id = new_labels != old_labels
            if np.sum(change_id) < th or len(np.unique(new_labels)) < self.k:
                break
            old_label, new_label = old_labels[change_id], new_labels[change_id]
            data_index = data_index[change_id]
            for l in range(len(old_label)):
                self._change_points(
                    self.clusters_[old_label[l]],
                    self.clusters_[new_label[l]],
                    data_index[l],
                    X,
                )
            self._update_centers(X)
        return self

    def _get_seeds(self, X):
        dists = 1 - safe_sparse_dot(X, X.T, dense_output=True) / self.n_estimators
        density = self._get_klc(X)
        filter_index = density < density.T
        tmp_dists = np.ones_like(dists)
        tmp_dists[filter_index] = dists[filter_index]
        min_dist = np.min(tmp_dists, axis=1)
        mult = density.A1 * min_dist
        sort_mult = np.argpartition(mult, -self.k, axis=1)[-self.k :]
        return sort_mult

    def _get_klc(self, X):
        density = safe_sparse_dot(X, X.mean(axis=0).T)
        n = density.shape[0]
        knn_index = ArgKmin.compute(
            X=X,
            Y=X,
            k=self.kn + 1,
            metric="sqeuclidean",
            metric_kwargs={},
            strategy="auto",
            return_distance=False,
        )
        nn_density = density[knn_index[:, 1:]].reshape(n, self.kn)
        lc = np.sum(density > nn_density, axis=1)
        return lc

    def _get_labels(self, X):
        check_is_fitted(self)
        n = X.shape[0]
        labels = np.zeros(n, dtype=int)
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
            x = np.argmax(safe_sparse_dot(X[ci.points], ci.kernel_mean.T))
            ci.set_center(ci.points[x])
        return self
