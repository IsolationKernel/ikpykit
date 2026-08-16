"""
ikpykit (c) by Xin Han

ikpykit is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from ikpykit.cluster._kcluster import KCluster
from ikpykit.kernel import IsoKernel


class PSKC(BaseEstimator, ClusterMixin):
    """Point-Set Kernel Clustering algorithm using Isolation Kernels.

    PSKC is a clustering algorithm that leverages Isolation Kernels to create
    feature vector representations of data points. It adaptively captures the
    characteristics of local data distributions by using data-dependent kernels.
    The algorithm forms clusters by identifying points with high similarity in
    the transformed kernel space.

    The clustering process works by iteratively:
    1. Selecting a center point with maximum similarity to the mean
    2. Forming a cluster around this center
    3. Removing these points from consideration
    4. Continuing until stopping criteria are met

    n_estimators : int, default=200
        The number of base estimators (trees) in the isolation ensemble.

    max_samples : int or str, default="auto"
        - If int, then draw `max_samples` samples.
        - If "auto", then `max_samples=min(256, n_samples)`.

    method : {'inne', 'anne'}, default='inne'
        The method used for building the isolation kernel.

    tau : float, default=0.1
        Lower values result in more clusters.

    v : float, default=0.1
        The decay factor for reducing the similarity threshold.
        Controls the expansion of clusters.

        Controls the pseudo-randomness of the algorithm for reproducibility.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    clusters_ : list
        List of KCluster objects representing the identified clusters.

    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset.

    centers : list
        Centers of each cluster in the transformed feature space.

    n_classes : int
        Number of clusters found.

    Examples
    --------
    >>> from ikpykit.cluster import PSKC
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [10, 2], [10, 10],  [1, 0], [1, 1]])
    >>> pskc = PSKC(n_estimators=100, max_samples=2, tau=0.3, v=0.1, random_state=24)
    >>> pskc.fit_predict(X)
    array([0, 0, 1, 1, 0, 0])

    References
    ----------
    1. Kai Ming Ting, Jonathan R. Wells, Ye Zhu (2023) "Point-set Kernel Clustering".
       IEEE Transactions on Knowledge and Data Engineering. Vol.35, 5147-5158.
    """

    def __init__(
        self,
        n_estimators=200,
        max_samples="auto",
        method="inne",
        tau=0.1,
        v=0.1,
        random_state=None,
    ):
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
        cluster_id = 1
        point_indices = np.arange(X.shape[0])
        while len(point_indices) > 0:
            cluster, point_indices = self._create_seed_cluster(
                X,
                point_indices,
                cluster_id,
            )
            self.clusters_.append(cluster)
            if len(point_indices) == 0:
                break

            cluster, point_indices, threshold = self._attach_nearest_global_point(
                cluster,
                X,
                point_indices,
            )

            if threshold <= self.tau:
                break

            cluster, point_indices = self._expand_cluster(
                cluster,
                X,
                point_indices,
                threshold,
            )
            assert self._get_n_points() == X.shape[0] - len(point_indices)
            cluster_id += 1
        return self

    def _create_seed_cluster(self, X, point_indices, cluster_id):
        """Create a new cluster by selecting the point most similar to global mean."""
        center_idx = np.argmax(self._similarity_to_global_mean(X, point_indices))
        cluster = KCluster(cluster_id)
        return self._update_cluster(cluster, X, point_indices, center_idx)

    def _attach_nearest_global_point(self, cluster, X, point_indices):
        """Attach the current best remaining point and compute initial threshold."""
        nn_dists = self._similarity_to_global_mean(X, point_indices) / self.n_estimators
        nn_index = np.argmax(nn_dists)
        nn_dist = nn_dists[nn_index]
        cluster, point_indices = self._update_cluster(
            cluster, X, point_indices, nn_index
        )
        threshold = (1 - self.v) * nn_dist
        return cluster, point_indices, threshold

    def _expand_cluster(self, cluster, X, point_indices, threshold):
        """Expand a cluster by iteratively adding points above decayed threshold."""
        r = threshold
        while r > self.tau:
            similarities = (
                safe_sparse_dot(X[point_indices], cluster.kernel_mean.T)
                / self.n_estimators
            )
            selected = np.where(r < similarities)[0]
            if len(selected) == 0:
                break
            cluster, point_indices = self._update_cluster(
                cluster, X, point_indices, selected
            )
            r = (1 - self.v) * r
        return cluster, point_indices

    def _similarity_to_global_mean(self, X, point_indices):
        """Compute similarity of points to mean of the current active set."""
        return safe_sparse_dot(X[point_indices], X[point_indices].mean(axis=0).T)

    def _update_cluster(
        self,
        c_k,
        X,
        point_indices,
        x_id,
    ):
        c_k.add_points(point_indices[x_id], X[point_indices][x_id])
        point_indices = np.delete(point_indices, x_id)
        return c_k, point_indices

    def _get_labels(self, X):
        """Get cluster labels for all points in the dataset."""
        return KCluster.build_labels(self.clusters_, X.shape[0])

    def _get_n_points(self):
        check_is_fitted(self)
        return KCluster.total_points(self.clusters_)
