"""
ikpykit (c) by Xin Han

ikpykit is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics._pairwise_distances_reduction import ArgKmin
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

    This version splits the data space with hyperspheres: each estimator draws
    `max_samples` points and puts a ball around every one of them, reaching out
    to that point's nearest neighbour among the draw. The cells are therefore
    balls, where `anne` has Voronoi cells and `iforest` has axis-parallel boxes.
    The feature in the Isolation kernel space is the index of the ball a point
    falls into, so each point is represented as a binary vector such that only
    the cell the point falling into is 1. A point outside every ball falls into
    no cell and is represented by zeros.

    Parameters
    ----------

    n_estimators : int
        The number of base estimators in the ensemble.

    max_samples : int
        The number of samples to draw from X to train each base estimator.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

    References
    ----------
    1. Qin, X., Ting, K.M., Zhu, Y. and Lee, V.C.
       "Nearest-neighbour-induced isolation similarity and its impact on density-based clustering".
       In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, 2019, July, pp. 4755-4762

    2. T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells.
       "Isolation-based anomaly detection using nearest-neighbor ensembles." In Computational
       Intelligence, vol. 34, 2018, pp. 968-998.
    """

    def __init__(self, n_estimators, max_samples, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_samples_ = None
        self._seeds = None
        self._radius = None
        self._centroids = None

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
        n_samples, n_features = X.shape
        self.max_samples_ = min(self.max_samples_, n_samples)

        self._centroids = np.empty((self.n_estimators, self.max_samples_, n_features))
        self._radius = np.empty((self.n_estimators, self.max_samples_))
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            centroid_index = rnd.choice(n_samples, self.max_samples_, replace=False)
            self._centroids[i] = X[centroid_index]
            # radius of each hypersphere is the Nearest Neighbors distance of centroid.
            nn_neighbors, _ = ArgKmin.compute(
                X=self._centroids[i],
                Y=self._centroids[i],
                k=2,
                metric="sqeuclidean",
                metric_kwargs={},
                strategy="auto",
                return_distance=True,
            )
            self._radius[i] = nn_neighbors[:, 1]

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
        The features are organized as a n_instances by n_estimators*t matrix.
        """
        check_is_fitted(self)
        X = check_array(X)
        n, _m = X.shape
        embedding = None
        for i in range(self.n_estimators):
            nearest_index, nearest_values = pairwise_distances_argmin_min(
                X, self._centroids[i], metric="euclidean", axis=1
            )
            # filter out of ball
            out_index = np.array(range(n))[
                nearest_values > self._radius[i][nearest_index]
            ]
            ik_value = np.eye(self.max_samples)[nearest_index]
            ik_value[out_index] = 0

            ik_value_sparse = sparse.csr_matrix(ik_value)
            if embedding is None:
                embedding = ik_value_sparse
            else:
                embedding = sparse.hstack((embedding, ik_value_sparse))
        return embedding
