"""
pyike (c) by Xin Han

pyike is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

"""
pyike (c) by Xin Han

pyike is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

from sklearn.datasets import make_blobs
import numpy as np
from pyike.cluster import IKHC
from sklearn import metrics


def test_IsoKHC():
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    n_samples = 100
    n_clusters, n_features = centers.shape
    X, true_labels = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )

    v = 0.1
    psi = 8
    tau = 0.0001

    # clus = IsoKHC(n_estimators=200, max_samples=psi, method="anne", tau=tau, v=v)
    clus = IKHC()
    labels_pred = clus.fit_predict(X)

    # Check performance
    print(metrics.adjusted_mutual_info_score(true_labels, labels_pred))


test_IsoKHC()
