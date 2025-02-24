"""
pyiks (c) by Xin Han

pyiks is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numbers
from warnings import warn
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from pyiks.kernel import IsoKernel
import numpy as np

import numpy as np
from scipy.spatial.distance import pdist, squareform


class ICID(OutlierMixin, BaseEstimator):
    """Isolation-based anomaly detection using nearest-neighbor ensembles.
    The INNE algorithm uses the nearest neighbour ensemble to isolate anomalies.
    It partitions the data space into regions using a subsample and determines an
    isolation score for each region. As each region adapts to local distribution,
    the calculated isolation score is a local measure that is relative to the local
    neighbourhood, enabling it to detect both global and local anomalies. INNE has
    linear time complexity to efficiently handle large and high-dimensional datasets
    with complex distributions.
    Parameters
    ----------
    n_estimators_1 : int, default=200
        The number of base estimators in the ensemble of first step.
    max_samples_1 : int, default="auto"
        The number of samples to draw from X to train each base estimator in the first step.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.
    n_estimators_2 : int, default=200
        The number of base estimators in the ensemble of secound step.
    max_samples_2 : int, default="auto"
        The number of samples to draw from X to train each base estimator in the secound step.
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
    .. [1]  Y. Cao, Y. Zhu, K. M. Ting, F. D. Salim, H. X. Li, L. Yang, G. Li (2024). Detecting change intervals with isolation distributional kernel. Journal of Artificial Intelligence Research, 79:273–306.
    Examples
    --------
    >>> from pyiks.group import ICID
    >>> import numpy as np
    >>> X =  [[[-1.1], [0.3], [0.5], [100]]] : 3D array-like of shape (n_groups , n_samples, n_features)
    >>> clf = ICID().fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])
    """

    def point_score(Y, psi, window):
        # calculate each point dissimilarity score
        # input should be data, psi value and window size

        # Normalisation
        Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
        Y[np.isnan(Y)] = 0.5

        type = "NormalisedKernel"

        Sdata = Y
        data = Y

        t = 200

        ndata = IsoKernel.fit_transform(Sdata, data, psi, t)

        # index each segmentation
        index = list(range(0, len(Y), window))

        if index[-1] != len(Y):
            index.append(len(Y))

        # kernel mean embedding
        mdata = []

        for i in range(len(index) - 1):
            cdata = ndata[index[i] : index[i + 1], :]
            mdata.append(np.mean(cdata, axis=0))

        mdata = np.array(mdata)

        k = 1  # knn

        score = []

        if type == "NormalisedKernel":
            for i in range(k + 1, len(mdata)):
                Cscore = []
                for j in range(1, k + 1):
                    Cscore.append(
                        np.dot(mdata[i], mdata[i - j])
                        / (np.linalg.norm(mdata[i]) * np.linalg.norm(mdata[i - j]))
                    )  # normalised inner product
                score.append(1 - np.mean(Cscore))

        elif type == "MMD":
            for i in range(k + 1, len(mdata)):
                score.append(
                    np.mean(pdist(mdata[i - k : i, :], "euclidean"))
                )  # MMD (euclidean distance)

        # assign score to segmentation
        Pscore = np.zeros(len(Y))

        for i in range(len(index) - 1):
            Pscore[index[i] : index[i + 1]] = score[i]

        # Normalisation
        Pscore = (Pscore - np.min(Pscore)) / (np.max(Pscore) - np.min(Pscore))

        return Pscore

    def aNNEspace(Sdata, data, psi, t):
        # Placeholder function for aNNEspace
        # This function needs to be implemented or imported if available
        pass

    def best_threshold(Pscore, a):
        # a is alpha, Pscore is point dissimilarity score
        # Adjust the threshold by changing the value of a

        best_threshold = np.mean(Pscore) + a * np.std(Pscore)

        # the point will be labelled as 1 if Pscore larger than threshold.
        # the point will be labelled as 0 if Pscore smaller than threshold.
        result = np.zeros_like(Pscore)

        for i in range(len(Pscore)):
            if best_threshold < Pscore[i]:
                result[i] = 1
            else:
                result[i] = 0

        return best_threshold, result

    def best_psi(Y, window):
        # select the best psi by using approximate Entropy

        Psi_list = [2, 4, 8, 16, 32, 64]  # range of psi

        Pscore = []
        ent = np.zeros(len(Psi_list))

        for i in range(len(Psi_list)):
            Pscore.append(point_score(Y, Psi_list[i], window))
            ent[i] = approximate_entropy(Pscore[i])
            # Approximate entropy is a measure to quantify the amount of regularity
            # and unpredictability of fluctuations over a time series.

        best_ent = np.min(ent)
        idx = np.argmin(ent)
        best_Pscore = Pscore[idx]
        best_psi = Psi_list[idx]

        return best_Pscore, best_ent, best_psi

    # Placeholder functions for point_score and approximate_entropy
    def point_score(Y, psi, window):
        # Implement the point_score function here
        pass

    def approximate_entropy(data):
        # Implement the approximate_entropy function here
        pass
