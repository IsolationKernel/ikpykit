"""
pyiks (c) by Xin Han

pyiks is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import numpy as np
from typing import Union, List, Any


def check_format(
    X: Union[List[Any], np.ndarray],
    allow_empty: bool = False,
) -> np.ndarray:
    """
    Validates trajectory data format.

    Parameters:
    -----------
    X : array-like
        Trajectory data with shape (n_trajectories, n_samples, n_features)
    allow_empty : bool, default=False
        Whether to allow trajectories with no samples

    Returns:
    --------
    np.ndarray
        Validated trajectory data
    """
    # Validate input shape (3D array for trajectories)
    if not isinstance(X, (list, np.ndarray)):
        raise ValueError(
            "X should be array-like with 3 dimensions (n_trajectories, n_samples, n_features)"
        )

    if isinstance(X, list):
        X = np.array(X)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")

    # Check for empty trajectories
    empty_trajectories = [i for i, trajectory in enumerate(X) if len(trajectory) == 0]
    if not allow_empty and empty_trajectories:
        raise ValueError(
            f"Trajectories at indices {empty_trajectories} have no samples"
        )

    # Only check non-empty trajectories
    non_empty_trajectories = [t for t in X if len(t) > 0]
    if not non_empty_trajectories:
        if allow_empty:
            return X
        raise ValueError("No valid trajectories found")

    # Check all trajectories have the same length
    trajectory_lengths = np.array(
        [len(trajectory) for trajectory in non_empty_trajectories]
    )
    if len(np.unique(trajectory_lengths)) != 1:
        raise ValueError(
            f"All trajectories must have same length. Found lengths: {sorted(np.unique(trajectory_lengths))}"
        )

    # Check all trajectories have the same number of features
    feature_counts = np.array(
        [
            trajectory.shape[1] if trajectory.size > 0 else 0
            for trajectory in non_empty_trajectories
        ]
    )
    if len(np.unique(feature_counts)) > 1:
        raise ValueError("All trajectories must have the same number of features")
    elif np.unique(feature_counts)[0] != 2:
        raise ValueError("All trajectories must have 2 features (longitude, latitude)")

    return X
