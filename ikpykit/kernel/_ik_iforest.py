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
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max


class IK_IForest(TransformerMixin, BaseEstimator):
    """Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version splits the data space with isolation trees: each tree draws
    `max_samples` points and cuts them apart with axis-parallel splits at random
    thresholds, so the cells are boxes rather than the Voronoi cells of `anne` or
    the hyperspheres of `inne`. The feature in the Isolation kernel space is the
    index of the leaf a point falls into. Each point is represented as a binary
    vector such that only the cell the point falls into is 1.

    Parameters
    ----------

    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : int, default=256
        The number of samples to draw from X to train each base estimator.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    max_samples_ : int
        The number of samples actually drawn, capped at the size of X.

    trees_ : list of ExtraTreeRegressor
        The isolation trees, one per estimator.

    leaf_indices_ : list of ndarray
        For each tree, a lookup from the tree's own node ids to a dense cell
        index. A tree grown on `max_samples_` points has at most that many
        leaves, since every leaf holds at least one of them, so the cell index
        always fits the same block width the other methods use.

    is_fitted_ : bool
        Whether the estimator has been fitted.

    References
    ----------
    1. Kai Ming Ting, Yue Zhu, Zhi-Hua Zhou (2018).
       "Isolation Kernel and Its Effect on SVM".
       Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2329-2337.
    """

    def __init__(self, n_estimators=100, max_samples=256, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        n_samples = X.shape[0]
        self.max_samples_ = min(self.max_samples, n_samples)
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        # The height an isolation tree is grown to. Beyond this the tree can
        # only separate points that are already rare, which is why the standard
        # isolation forest stops here as well.
        max_depth = int(np.ceil(np.log2(max(self.max_samples_, 2))))

        self.trees_ = []
        self.leaf_indices_ = []
        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            subsample = rnd.choice(n_samples, self.max_samples_, replace=False)
            tree = ExtraTreeRegressor(
                max_features=1,
                splitter="random",
                max_depth=max_depth,
                random_state=rnd.randint(MAX_INT),
            )
            # The target is noise: an isolation tree splits at random and never
            # consults it, but the regressor needs one to fit against.
            tree.fit(X[subsample], rnd.uniform(size=self.max_samples_))
            self.trees_.append(tree)
            self.leaf_indices_.append(self._dense_leaf_index(tree))

        self.is_fitted_ = True
        return self

    @staticmethod
    def _dense_leaf_index(tree):
        """Map a tree's leaf node ids onto a contiguous range starting at 0.

        The ids the tree assigns are positions in its node array, so they run
        past the number of leaves and cannot be used as feature columns.
        """
        inner = tree.tree_
        lookup = np.zeros(inner.node_count, dtype=np.int32)
        leaves = np.flatnonzero(inner.children_left == -1)
        lookup[leaves] = np.arange(len(leaves), dtype=np.int32)
        return lookup

    def transform(self, X):
        """Compute the isolation kernel feature of X.

        Parameters
        ----------
        X: array-like of shape (n_instances, n_features)
            The input instances.

        Returns
        -------
        sparse matrix: The finite binary features based on the kernel feature map.
            The features are organized as a n_instances by (n_estimators * max_samples_) matrix.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        n_samples = X.shape[0]
        n_features = self.n_estimators * self.max_samples_

        rows = np.tile(np.arange(n_samples), self.n_estimators)
        cols = np.empty(n_samples * self.n_estimators, dtype=np.int32)
        data = np.ones(n_samples * self.n_estimators, dtype=np.float64)

        for est_idx, (tree, leaf_index) in enumerate(
            zip(self.trees_, self.leaf_indices_, strict=True)
        ):
            cells = leaf_index[tree.apply(X)]
            start_idx = est_idx * n_samples
            end_idx = (est_idx + 1) * n_samples
            cols[start_idx:end_idx] = cells + (est_idx * self.max_samples_)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))
