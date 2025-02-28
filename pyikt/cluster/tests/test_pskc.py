import pytest
import numpy as np
from sklearn.datasets import make_blobs
from pyikt.cluster import PSKC


def test_pskc_initialization():
    """Test PSKC initialization with parameters."""
    # Initialize with parameters
    model = PSKC(
        n_estimators=100,
        max_samples=0.8,
        method="anne",
        tau=0.1,
        v=0.2,
        random_state=42,
    )

    # Check if parameters are correctly set
    assert model.n_estimators == 100
    assert model.max_samples == 0.8
    assert model.method == "anne"
    assert model.tau == 0.1
    assert model.v == 0.2
    assert model.random_state == 42
    assert model.clusters_ == []
    assert model.labels_ is None


def test_pskc_with_synthetic_data():
    """Test PSKC with synthetic blob data."""
    # Generate synthetic data
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    # Initialize and fit the model
    model = PSKC(
        n_estimators=100,
        max_samples=0.8,
        method="anne",
        tau=0.5,
        v=0.1,
        random_state=42,
    )
    model.fit(X)

    # Check if the model is fitted
    assert hasattr(model, "is_fitted_")
    assert model.is_fitted_ is True

    # Check if clusters were found
    assert len(model.clusters_) > 0

    # Check label assignment
    assert model.labels_.shape == (100,)
    assert len(np.unique(model.labels_)) == len(model.clusters_)


def test_pskc_properties():
    """Test PSKC properties after fitting."""
    X, _ = make_blobs(n_samples=50, centers=2, random_state=42)

    model = PSKC(
        n_estimators=30,
        max_samples=0.8,
        method="anne",
        tau=0.5,
        v=0.1,
        random_state=42,
    )
    model.fit(X)

    # Test properties
    assert len(model.clusters) > 0
    assert len(model.centers) == len(model.clusters)
    assert model.n_classes == len(model.clusters)


def test_pskc_not_fitted():
    """Test error raised when accessing properties before fitting."""
    model = PSKC(n_estimators=100, max_samples=0.8, method="anne", tau=0.5, v=0.1)

    with pytest.raises(Exception):
        _ = model.clusters

    with pytest.raises(Exception):
        _ = model.centers

    with pytest.raises(Exception):
        _ = model.n_classes


def test_pskc_parameter_effect():
    """Test effect of parameters on clustering results."""
    X, _ = make_blobs(n_samples=80, centers=2, random_state=42)

    # Model with high tau (should produce fewer clusters)
    model_high_tau = PSKC(
        n_estimators=30,
        max_samples=0.8,
        method="anne",
        tau=0.9,
        v=0.1,
        random_state=42,
    )
    model_high_tau.fit(X)

    # Model with lower tau (should produce more clusters)
    model_low_tau = PSKC(
        n_estimators=30,
        max_samples=0.8,
        method="anne",
        tau=0.1,
        v=0.1,
        random_state=42,
    )
    model_low_tau.fit(X)

    # Check if the number of clusters is different (or at least not decreasing)
    assert len(model_low_tau.clusters) >= len(model_high_tau.clusters)
