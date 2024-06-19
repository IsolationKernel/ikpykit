# Copyright 2024 Xin Han. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright 2022 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.base import BaseEstimator

from isoml.kernel import IsolationKernel


class IsoKHC(BaseEstimator):
    """IsoKHC is a novel hierarchical clustering algorithm.
    It uses a data-dependent kernel called Isolation Kernel to measure the the similarity between clusters.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of base estimators in the ensemble.

    max_samples : int, default="auto"
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.

    ik_method: {"inne", "anne"}, default="inne"
        isolation method to use. The original algorithm in paper is `"anne"`.

    lk_method : str, default="single"
        The linkage algorithm to use. The supported  Linkage Methods are 'single', 'complete', 'average' and
        'weighted'.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the samples to
        fit the Isolation Kernel.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    References
    ----------
    .. [1] Xin Han, Ye Zhu, Kai Ming Ting, and Gang Li,
           "The Impact of Isolation Kernel on Agglomerative Hierarchical Clustering Algorithms",
           arXiv e-prints, 2020.

    Examples
    --------
    >>> from IsoKAHC import IsoKAHC
    >>> import numpy as np
    >>> X = [[0.4,0.3], [0.3,0.8], [0.5, 0.4], [0.5, 0.1]]
    >>> clf = IsoKAHC(n_estimators=200, max_samples=2, method='single')
    >>> dendrogram  = clf.fit_transform(X)
    """

    def __init__(
        self,
        n_estimators=200,
        max_samples="auto",
        lk_method="single",
        ik_method="anne",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.ik_method = ik_method
        self.lk_method = lk_method
        self.random_state = random_state

    def fit(self, X):
        # Check data
        X = self._validate_data(X, accept_sparse=False)
        self.isokernel_ = IsolationKernel(
            self.ik_method, self.n_estimators, self.max_samples, self.random_state
        )
        self.isokernel_ = self.isokernel_.fit(X)
        similarity_matrix = self.isokernel_.similarity(X)
        self.dendrogram_ = linkage(1 - similarity_matrix, method=self.lk_method)
        return self

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to data and return the dendrogram. Same parameters as the ``fit`` method.
        Returns
        -------
        dendrogram : np.ndarray
            Dendrogram.
        """
        self.fit(*args, **kwargs)
        return self.dendrogram_
