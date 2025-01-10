import pytest
import logging

import numpy as np

from pyike.anomaly import IsolationNNE

from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score




@pytest.mark.parametrize("n_estimators", [100])
@pytest.mark.parametrize("max_samples", [2, 4, 8])
def test_inne(n_estimators, max_samples):
    N = 100
    anomaly_ratio = 0.01

    anomaly_num = int(N * anomaly_ratio)
    normal_num = N - anomaly_num

    normal, _ = make_blobs(n_samples=normal_num, centers=1, cluster_std=0.60, random_state=42)
    anomaly = np.random.uniform(low=-6, high=6, size=(anomaly_num, normal.shape[1]))

    data = np.vstack((normal, anomaly))
    label = np.concatenate((np.zeros(normal_num), np.ones(anomaly_num)))

    model = IsolationNNE(n_estimators=n_estimators, max_samples=max_samples)
    model.fit(data)

    score = model.score_samples(data)

    logging.info(f"[test_inne.py]")
    logging.info(f"n_estimators: {n_estimators}, max_samples: {max_samples}, AUC: {roc_auc_score(label, score)}")

