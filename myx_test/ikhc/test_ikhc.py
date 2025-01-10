import pytest
import logging

import numpy as np

from pyike.cluster import IKHC

from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score




@pytest.mark.parametrize("n_estimators", [100])
@pytest.mark.parametrize("max_samples", [4, 8, 16, 32])
def test_ikhc(n_estimators, max_samples):
    n_samples = 1000  # 样本数量
    n_features = 2    # 特征数量
    n_clusters = 3    # 聚类数量
    random_state = 42 # 随机种子，以便结果可重复

    # 使用 make_blobs 生成数据
    data, label = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=random_state)

    model = IKHC(n_estimators=n_estimators,
                 max_samples=max_samples,
                 )

    predict = model.fit_predict(data)

    ami = adjusted_mutual_info_score(label, predict)


    logging.info(f"[test_ikhc.py]")
    logging.info(f"n_estimators: {n_estimators}," \
                 f"max_samples: {max_samples}," \
                 f"AMI: {ami}")

