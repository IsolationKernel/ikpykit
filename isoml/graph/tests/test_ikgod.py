import numpy as np
from sklearn.datasets import make_blobs
from isoml.graph import IKGOD


def test_ikgod():
    # Generate a random dataset
    X, _ = make_blobs(n_samples=100, centers=2, random_state=0)

    # Initialize the IKGOD model
    model = IKGOD()

    # Fit the model to the dataset
    model.fit(X)

    # Predict the labels for the dataset
    labels = model.predict(X)

    # Check if the predicted labels have the correct shape
    assert labels.shape == (100,)

    # Check if the predicted labels are either 1 or -1
    assert np.all(np.logical_or(labels == 1, labels == -1))
