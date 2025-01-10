import pytest
import logging

import numpy as np

from pyike.cluster import IDKC

from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score




@pytest.mark.parametrize("n_estimators", [100])
@pytest.mark.parametrize("max_samples", [4, 8, 16, 32])
@pytest.mark.parametrize("method", ["inne"])
@pytest.mark.parametrize("k", [3])
@pytest.mark.parametrize("kn", [5])
@pytest.mark.parametrize("v", [0.1])
@pytest.mark.parametrize("n_init_samples", [30])
def test_idkc(n_estimators, max_samples, method, k, kn, v, n_init_samples):
    n_samples = 1000  # 样本数量
    n_features = 2    # 特征数量
    n_clusters = 3    # 聚类数量
    random_state = 42 # 随机种子，以便结果可重复

    # 使用 make_blobs 生成数据
    data, label = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=random_state)

    model = IDKC(n_estimators=n_estimators,
                 max_samples=max_samples,
                 method=method,
                 k=k,
                 kn=kn,
                 v=v,
                 n_init_samples=n_init_samples)

    predict = model.fit_predict(data)

    ami = adjusted_mutual_info_score(label, predict)


    logging.info(f"[test_idkc.py]")
    logging.info(f"n_estimators: {n_estimators}," \
                 f"max_samples: {max_samples}," \
                 f"method: {method}," \
                 f"k: {k}," \
                 f"kn: {kn}," \
                 f"v: {v}," \
                 f"n_init_samples: {n_init_samples}," \
                 f"AMI: {ami}")

