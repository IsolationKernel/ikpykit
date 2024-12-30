"""
pyike (c) by Xin Han

pyike is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import copy
from warnings import warn
from typing import Union
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pyike.kernel import IsoKernel
from pyike.graph.utils import get_degrees, get_neighbors, check_format


class IsoGraphKernel(BaseEstimator):
    """Isolation Graph Kernel is a new way to measure the similarity between two graphs.

    It addresses two key issues of kernel mean embedding, where the kernel employed has:
    (i) a feature map with intractable dimensionality which leads to high computational cost;
    and (ii) data independency which leads to poor detection accuracy in anomaly detection.

    Parameters
    ----------
    method : str, default="anne"
        The method to compute the isolation kernel feature. The available methods are: `anne`, `inne`, and `iforest`.

    n_estimators : int, default=200
        The number of base estimators in the ensemble.

    max_samples : int, default="auto"
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
    .. [1] Bi-Cun Xu, Kai Ming Ting and Yuan Jiang. 2021. "Isolation Graph Kernel".
    In Proceedings of The Thirty-Fifth AAAI Conference on Artificial Intelligence. 10487-10495.

    Examples
    --------
    >>> from pyike.graph import IsoGraphKernel
    >>> import numpy as np
    >>> X = [[0.4,0.3], [0.3,0.8], [0.5,0.4], [0.5,0.1]]
    >>> igk = IsoGraphKernel.fit(X)
    >>> D_i = [[0.4,0.3], [0.3,0.8]]
    >>> D_j = [[0.5, 0.4], [0.5, 0.1]]
    >>> igk.similarity(D_j, D_j)
    """

    def __init__(
        self, method="anne", n_estimators=200, max_samples="auto", random_state=None
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.method = method

    def fit(
        self,
        features: Union[sp.csr_matrix, np.ndarray],
    ):
        """Fit the model on data X.

        Parameters
        ----------
        features : sparse.csr_matrix, np.ndarray
            Features, array of shape (n_nodes, n_features).

        Returns
        -------
        self
        """
        features = check_array(features)
        self.iso_kernel_ = IsoKernel(
            self.method, self.n_estimators, self.max_samples, self.random_state
        )
        self.iso_kernel_ = self.iso_kernel_.fit(features)
        self.is_fitted_ = True
        return self

    # def similarity(self, X, dense_output=True):
    #     """Compute the isolation kernel similarity matrix of X.
    #     Parameters
    #     ----------
    #     X: array-like of shape (n_instances, n_features)
    #         The input instances.
    #     dense_output: bool, default=True
    #         Whether to return dense matrix of output.
    #     Returns
    #     -------
    #     The similarity matrix organized as an n_instances * n_instances matrix.
    #     """
    #     check_is_fitted(self)
    #     X = check_array(X)
    #     embed_X = self.transform(X)
    #     return (
    #         safe_sparse_dot(embed_X, embed_X.T, dense_output=dense_output)
    #         / self.n_estimators
    #     )

    def transform(
        self,
        adjacency: Union[sp.csr_matrix, np.ndarray],
        # weights: Union[list, np.ndarray],
        features: Union[sp.csr_matrix, np.ndarray],
        h: int,
        dense_output=False,
    ) -> Union[sp.csr_matrix, np.ndarray]:
        """Compute the isolation kernel feature of G.
        Parameters
        ----------
        adjacency : Union[list, sparse.csr_matrix]
            Adjacency matrix or list of sampled adjacency matrices.
        weights : list, np.array
            The weights of the adjacency matrix.
        features : sparse.csr_matrix, np.ndarray
            Features, array of shape (n_nodes, n_features).
        h : int
            The iteration of Weisfeiler–Lehman embedding.

        Returns
        -------
        The finite binary features based on the kernel feature map.
        The features are organized as an n_instances by h+1*psi*t matrix.
        """

        check_is_fitted(self)
        X = check_array(features)
        # weights = check_array(weights)
        adjacency = check_format(adjacency)
        X_trans = self.iso_kernel_.transform(X)
        embedding = self._wlembedding(adjacency, X_trans, h)

        if dense_output:
            if sp.issparse(embedding) and hasattr(embedding, "toarray"):
                return embedding.toarray()
            else:
                warn("The IsoKernel transform output is already dense.")
        return embedding

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

    def fit_transform(
        self,
        adjacency: Union[np.ndarray, sp.csr_matrix],
        features: Union[sp.csr_matrix, np.ndarray],
        **trans_params,
    ) -> np.ndarray:
        """Fit the model on data X and transform X.

        Parameters
        ----------
        adjacency : Union[list, sparse.csr_matrix]
            Adjacency matrix or list of sampled adjacency matrices.
        features : sparse.csr_matrix, np.ndarray
            Features, array of shape (n_nodes, n_features).

        Returns
        -------
        X_new : np.ndarray
            Transformed array.
        """
        self.fit(features)
        return self.transform(adjacency, features, **trans_params)
