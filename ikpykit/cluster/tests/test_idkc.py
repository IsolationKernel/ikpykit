"""
ikpykit (c) by Xin Han

ikpykit is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import pytest
import numpy as np
from sklearn import metrics
from sklearn.datasets import make_blobs, make_circles
from ikpykit.cluster import IDKC


def test_idkc_basic():
    # Generate sample data with 3 clusters
    centers = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    X, true_labels = make_blobs(
        n_samples=300, centers=centers, cluster_std=1.0, random_state=42
    )

    # Initialize IDKC
    idkc = IDKC(
        n_estimators=100,
        max_samples=10,
        method="anne",
        k=3,
        kn=5,
        v=0.1,
        n_init_samples=30,
        is_post_process=True,
        random_state=42,
    )

    # Fit and predict
    pred_labels = idkc.fit_predict(X)

    # Basic assertions
    assert len(pred_labels) == len(true_labels)
    assert len(idkc.clusters_) == 3
    assert idkc.n_it > 0

    # Check that clustering has reasonable performance
    ami_score = metrics.adjusted_mutual_info_score(true_labels, pred_labels)
    assert ami_score > 0.5, f"AMI score is too low: {ami_score}"


def test_idkc_parameter_validation():
    X = np.random.rand(50, 5)

    # Test invalid n_init_samples
    with pytest.raises(ValueError):
        idkc = IDKC(
            n_estimators=100,
            max_samples=10,
            method="anne",
            k=2,
            kn=5,
            v=0.1,
            n_init_samples=-1,
            random_state=42,
        )
        idkc.fit(X)

    # Test valid float n_init_samples
    idkc = IDKC(
        n_estimators=100,
        max_samples=10,
        method="anne",
        k=2,
        kn=5,
        v=0.1,
        n_init_samples=0.5,
        random_state=42,
    )
    idkc.fit(X)
    assert idkc.n_init_samples == 25

    # Test invalid float n_init_samples
    with pytest.raises(ValueError):
        idkc = IDKC(
            n_estimators=100,
            max_samples=10,
            method="anne",
            k=2,
            kn=5,
            v=0.1,
            n_init_samples=2.0,
            random_state=42,
        )
        idkc.fit(X)


def test_idkc_methods():
    # Test with different isolation methods
    X, true_labels = make_blobs(n_samples=100, centers=3, random_state=42)

    methods = ["anne", "inne"]
    for method in methods:
        idkc = IDKC(
            n_estimators=100,
            max_samples=10,
            method=method,
            k=3,
            kn=5,
            v=0.1,
            n_init_samples=30,
            random_state=42,
        )
        labels = idkc.fit_predict(X)
        assert len(labels) == len(X)
        assert len(np.unique(labels)) <= 3


def test_idkc_init_center():
    # Test with explicit init_center
    X, true_labels = make_blobs(n_samples=100, centers=3, random_state=42)

    # Choose specific points as initial centers
    init_centers = [0, 33, 66]

    idkc = IDKC(
        n_estimators=100,
        max_samples=10,
        method="anne",
        k=3,
        kn=5,
        v=0.1,
        n_init_samples=30,
        init_center=init_centers,
        random_state=42,
    )

    labels = idkc.fit_predict(X)
    assert len(labels) == len(X)


def test_idkc_post_processing():
    # Test with and without post-processing
    X, true_labels = make_blobs(n_samples=100, centers=3, random_state=42)

    # With post-processing
    idkc_with_pp = IDKC(
        n_estimators=100,
        max_samples=10,
        method="anne",
        k=3,
        kn=5,
        v=0.1,
        n_init_samples=30,
        is_post_process=True,
        random_state=42,
    )

    labels_with_pp = idkc_with_pp.fit_predict(X)

    # Without post-processing
    idkc_without_pp = IDKC(
        n_estimators=100,
        max_samples=10,
        method="anne",
        k=3,
        kn=5,
        v=0.1,
        n_init_samples=30,
        is_post_process=False,
        random_state=42,
    )

    labels_without_pp = idkc_without_pp.fit_predict(X)

    # Different results are expected, but both should produce valid clusterings
    assert len(labels_with_pp) == len(X)
    assert len(labels_without_pp) == len(X)


def test_idkc_non_linear_clusters():
    # Test on a dataset with non-linearly separable clusters
    X, true_labels = make_circles(
        n_samples=200, factor=0.5, noise=0.05, random_state=42
    )

    idkc = IDKC(
        n_estimators=150,
        max_samples=15,
        method="anne",
        k=2,
        kn=10,
        v=0.2,
        n_init_samples=50,
        is_post_process=True,
        random_state=42,
    )

    labels = idkc.fit_predict(X)
    assert len(labels) == len(X)

    # Test clusters are formed
    assert len(idkc.clusters_) == 2
    for cluster in idkc.clusters_:
        assert cluster.n_points > 0


def test_idkc_empty_clusters():
    # Generate imbalanced data where one cluster might end up empty
    X = np.vstack(
        [
            np.random.randn(95, 2),  # Cluster 1: 95 points
            np.random.randn(5, 2) + 10,  # Cluster 2: 5 points
        ]
    )

    idkc = IDKC(
        n_estimators=100,
        max_samples=10,
        method="anne",
        k=3,  # Asking for 3 clusters when there are only 2
        kn=5,
        v=0.1,
        n_init_samples=20,
        random_state=42,
    )

    labels = idkc.fit_predict(X)

    # Should still have k=3 clusters in the model
    assert len(idkc.clusters_) == 3
