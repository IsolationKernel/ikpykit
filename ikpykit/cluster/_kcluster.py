"""
ikpykit (c) by Xin Han

ikpykit is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

from collections.abc import Iterable
from numbers import Integral

import numpy as np
from scipy import sparse as sp


class KCluster:
    def __init__(self, id: int) -> None:
        self.id = id
        self.center = None
        self.kernel_mean_ = None
        self.points_ = []
        self.center = None

    def set_center(self, center):
        self.center = center

    def add_points(self, ids, X):
        self.increment_kernel_mean_(X)
        if isinstance(ids, Integral):
            self.points_.append(ids)
        elif isinstance(ids, Iterable):
            self.points_.extend(ids)

    def delete_points(self, points, X):
        if isinstance(points, Integral):
            if points not in self.points_:
                raise ValueError(f"Point {points} not in cluster {self.id}")
            self.points_.remove(points)
            self.reduce_kernel_mean_(X)
        elif isinstance(points, Iterable):
            missing_points = [p for p in points if p not in self.points_]
            if missing_points:
                raise ValueError(f"Points {missing_points} not in cluster {self.id}")
            for p in points:
                self.points_.remove(p)
            self.reduce_kernel_mean_(X)

    def reduce_kernel_mean_(self, X):
        if self.kernel_mean_ is None:
            raise ValueError("Kernel mean is not initialized.")
        else:
            self.kernel_mean_ = (self.kernel_mean_ * self.n_points - X.sum(axis=0)).sum(
                axis=0
            ) / (self.n_points - X.shape[0])

    def increment_kernel_mean_(self, X):
        if self.kernel_mean_ is None:
            self.kernel_mean_ = X
        else:
            self.kernel_mean_ = sp.vstack((self.kernel_mean_ * self.n_points, X)).sum(
                axis=0
            ) / (self.n_points + X.shape[0])

    @property
    def n_points(self):
        return len(self.points_)

    @property
    def points(self):
        return self.points_

    @property
    def kernel_mean(self):
        return self.kernel_mean_

    @staticmethod
    def build_labels(clusters, n_samples):
        """Build label array from a sequence of clusters.

        Unassigned samples are marked as -1.
        """
        labels = np.full(n_samples, -1, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            labels[cluster.points_] = cluster_id
        return labels

    @staticmethod
    def total_points(clusters):
        """Return total number of points currently assigned across clusters."""
        return sum(cluster.n_points for cluster in clusters)
