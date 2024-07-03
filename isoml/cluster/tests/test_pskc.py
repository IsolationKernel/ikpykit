# Copyright 2024 Xin Han. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from sklearn.datasets import make_blobs
import numpy as np
from isoml.cluster import PSKC
from sklearn import metrics

def test_PSKC():
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

    clus = PSKC(n_estimators=200, max_samples=psi, method="anne", tau=tau, v=v)
    labels_pred = clus.fit_predict(X)

    # Check performance
    print(metrics.adjusted_mutual_info_score(true_labels, labels_pred))

test_PSKC()