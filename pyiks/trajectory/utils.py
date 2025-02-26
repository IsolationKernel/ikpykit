"""
pyiks (c) by Xin Han

pyiks is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
import scipy.sparse as sp
from typing import Union


def check_format(
    X,
    allow_empty: bool = False,
) -> np.ndarray:
    # Validate input shape (3D array for trajectories)
    if not isinstance(X, (list, np.ndarray)):
        raise ValueError(
            "X should be array-like with 3 dimensions (n_trajectories, n_samples, n_features)"
        )

    if isinstance(X, list):
        X = np.array(X)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")

    # Check all trajectories have the same number of features
    for trajectory in X:
        if len(trajectory) == 0:
            raise ValueError("All trajectories must have at least one sample")
    n_trajactory_features = set([len(trajectory[0]) for trajectory in X])
    if len(n_trajactory_features) > 1:
        raise ValueError("All trajectories must have the same number of features")
    return X
