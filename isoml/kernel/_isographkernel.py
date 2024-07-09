# Copyright 2024 Xin Han. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import copy
from warnings import warn
from typing import Union
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from isoml.kernel import IsoKernel


def _get_degrees(
        input_matrix: sp.csr_matrix, transpose: bool = False
) -> np.ndarray:
    """Get the vector of degrees of a graph.

    If the graph is directed, returns the out-degrees (number of successors). Set ``transpose=True``
    to get the in-degrees (number of predecessors).

    For a biadjacency matrix, returns the degrees of rows. Set ``transpose=True`` to get the degrees of columns.

    Parameters
    ----------
    input_matrix : sparse.csr_matrix
        Adjacency or biadjacency matrix.
    transpose :
        If ``True``, transpose the input matrix.
    Returns
    -------
    degrees : np.ndarray
        Array of degrees.
    """
    if transpose:
        matrix = sp.csr_matrix(input_matrix.T)
    else:
        matrix = input_matrix
    degrees = matrix.indptr[1:] - matrix.indptr[:-1]
    return degrees


def _get_neighbors(
        input_matrix: sp.csr_matrix, node: int, transpose: bool = False
) -> np.ndarray:
    """Get the neighbors of a node.

    If the graph is directed, returns the vector of successors. Set ``transpose=True``
    to get the predecessors.

    For a biadjacency matrix, returns the neighbors of a row node. Set ``transpose=True``
    to get the neighbors of a column node.

    """
    if transpose:
        matrix = sp.csr_matrix(input_matrix.T)
    else:
        matrix = input_matrix
    neighbors = matrix.indices[matrix.indptr[node] : matrix.indptr[node + 1]]
    return neighbors


def _check_format(
        input_matrix: Union[
        sp.csr_matrix,
        sp.csc_matrix,
        sp.coo_matrix,
        sp.lil_matrix,
        np.ndarray,
        np.matrix,
    ],
    allow_empty: bool = False,
) -> sp.csr_matrix:
    """Check whether the matrix is a NumPy array or a Scipy sparse matrix and return
    the corresponding Scipy CSR matrix.
    """
    formats = {
        sp.csr_matrix,
        sp.csc_matrix,
        sp.coo_matrix,
        sp.lil_matrix,
        np.ndarray,
        np.matrix,
    }
    if type(input_matrix) not in formats:
        raise TypeError(
            "The input matrix must be in Scipy sparse format or Numpy ndarray format."
        )
    input_matrix = sp.csr_matrix(input_matrix)
    if not allow_empty and input_matrix.nnz == 0:
        raise ValueError("The input matrix is empty.")
    return input_matrix


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
    >>> from isoml.kernel import IsoGraphKernel
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
        #weights: Union[list, np.ndarray],
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
        n_nodes = X.shape[0]
        #weights = check_array(weights)
        adjacency = _check_format(adjacency)
        X_trans = self.iso_kernel_.transform(X)
        degrees = _get_degrees(adjacency)
        tmp_emd = X_trans
        embedding = copy.deepcopy(X_trans)
        for it in range(h + 1)[1:]:
            tmp_new = np.empty(X_trans.shape)
            for i in range(n_nodes):  # TODO: Add weights
                neighbors = _get_neighbors(adjacency, i)
                tmp_new[i] = ((tmp_emd[neighbors].sum(axis=0) / degrees[i] + tmp_emd[i]) / 2).A1
            tmp_emd = _check_format(tmp_new)
            embedding = sp.hstack((embedding, tmp_new))

        embedding = _check_format(embedding.mean(axis=0))

        if dense_output:
            if sp.issparse(embedding) and hasattr(embedding, "toarray"):
                return embedding.toarray()
            else:
                warn("The IsoKernel transform output is already dense.")
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
