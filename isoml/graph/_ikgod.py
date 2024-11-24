"""
isoml (c) by Xin Han

isoml is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numbers
import copy
from warnings import warn
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from isoml.graph.utils import get_degrees, get_neighbors, check_format

from isoml.kernel import IsoKernel


class IKGOD(BaseEstimator):
    """Isolation-based graph anomaly detection using nearest-neighbor ensembles.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of base estimators in the ensemble.
    max_samples : int, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.
    method: {"inne", "anne", "auto"}, default="inne"
        isolation method to use. The original algorithm in paper is `"inne"`.
    contamination : "auto" or float, default="auto"
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
            - If "auto", the threshold is determined as in the original paper.
            - If float, the contamination should be in the range (0, 0.5].
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    References
    ----------
    .. [1] Zhong Zhuang, Kai Ming Ting, Guansong Pang, Shuaibin Song (2023).
    Subgraph Centralization: A Necessary Step for Graph Anomaly Detection.
    Proceedings of The SIAM Conference on Data Mining.

    Examples
    --------
    >>> from isoml.graph import IKGOD
    >>> import numpy as np
    >>> X =  [[-1.1], [0.3], [0.5], [100]]
    >>> clf = IKGOD().fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])
    """

    def __init__(
        self,
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        method="inne",
        random_state=None,
        h=3,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.contamination = contamination
        self.method = method
        self.h = h

    def fit(self, adjacency, features, h, y=None):
        """
        Fit estimator.
        Parameters
        ----------
        adjacency : array-like of shape (n_samples, n_samples)
        features : array-like of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency.
        h: int
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Check data
        adjacency = check_format(adjacency)
        features = check_array(features, accept_sparse=False)

        n_samples = features.shape[0]
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(16, n_samples)
            else:
                raise ValueError(
                    "max_samples (%s) is not supported."
                    'Valid choices are: "auto", int or'
                    "float" % self.max_samples
                )

        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError(
                    "max_samples must be in (0, 1], got %r" % self.max_samples
                )
            max_samples = int(self.max_samples * features.shape[0])

        self.max_samples_ = max_samples
        self._fit(adjacency, features, h)
        self.is_fitted_ = True

        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], got: %f" % self.contamination
                )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
        else:
            # else, define offset_ wrt contamination parameter
            self.offset_ = np.percentile(
                self.score_samples(features), 100.0 * self.contamination
            )

        return self

    def _fit(self, adjacency, features, h):

        iso_kernel = IsoKernel(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples_,
            random_state=self.random_state,
            method=self.method,
        )
        features_trans = iso_kernel.fit_transform(features)
        h_index = self._get_h_nodes_n_dict(adjacency, h)
        self.embedding_ = self._subgraph_embeddings(adjacency, features_trans, h_index)
        self.is_fitted_ = True
        return self

    def _get_h_nodes_n_dict(self, adj, h):
        adj_h = sp.eye(adj.shape[0])
        M = [{i: 0} for i in range(adj.shape[0])]
        h_index = [[i] for i in range(adj.shape[0])]
        for k in range(h):
            adj_h = sp.coo_matrix(adj_h * adj)
            for i, j in zip(adj_h.row, adj_h.col):
                if j not in M[i]:
                    M[i][j] = k + 1
                    h_index[i].append(j)
        return h_index

    def _subgraph_embeddings(self, adjacency, features, subgraph_index):
        n_nodes = adjacency.shape[0]
        subgraph_embedding = None
        for i in range(n_nodes):
            source_feat = features[i, :]
            subgraph_feat = features[subgraph_index[i]]
            subgraph_feat = subgraph_feat - np.tile(
                source_feat, (len(subgraph_index[i]), 1)
            )
            adj_i = adjacency[subgraph_index[i], :][:, subgraph_index[i]]
            graph_embed = self._wlembedding(adj_i, subgraph_feat, self.h)
            if subgraph_embedding is None:
                subgraph_embedding = graph_embed
            else:
                subgraph_embedding = sp.vstack((subgraph_embedding, graph_embed))
        return subgraph_embedding

    def _wlembedding(self, adjacency, X, h):
        n_nodes = adjacency.shape[0]
        degrees = get_degrees(adjacency)
        tmp_embedding = X
        embedding = copy.deepcopy(X)
        for it in range(h + 1)[1:]:
            updated_embedding = np.empty(X.shape)
            for i in range(n_nodes):  # TODO: Add weights
                neighbors = get_neighbors(adjacency, i)
                updated_embedding[i] = (
                    (
                        tmp_embedding[neighbors].sum(axis=0) / degrees[i]
                        + tmp_embedding[i]
                    )
                    / 2
                ).A1
            tmp_embedding = check_format(updated_embedding)
            embedding = sp.hstack((embedding, tmp_embedding))

        embedding = check_format(embedding.mean(axis=0))
        return embedding

    def _kernel_mean_embedding(self, X):
        return np.mean(X, axis=0) / self.max_samples_

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """

        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """
        Average anomaly score of X of the base classifiers.
        The anomaly score of an input sample is computed as
        the mean anomaly score of the .
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier.

        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """
        Opposite of the anomaly score defined in the original paper.
        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """

        check_is_fitted(self, "is_fitted_")
        # Check data
        X = check_array(X, accept_sparse=False)
        kme = self._kernel_mean_embedding(self.embedding_)
        scores = safe_sparse_dot(self.embedding_, kme.T).A1
        return -scores


# # embedding of a graph
# def createWlEmbedding(node_features, adj_mat, h):
#     graph_feat = []
#     for it in range(h + 1):
#         if it == 0:
#             graph_feat.append(node_features)
#         else:
#             adj_cur = adj_mat + np.identity(adj_mat.shape[0])

#             adj_cur = create_adj_avg(adj_cur)

#             np.fill_diagonal(adj_cur, 0)
#             graph_feat_cur = 0.5 * (
#                 np.dot(adj_cur, graph_feat[it - 1]) + graph_feat[it - 1]
#             )
#             graph_feat.append(graph_feat_cur)
#     return np.mean(np.concatenate(graph_feat, axis=1), axis=0)
