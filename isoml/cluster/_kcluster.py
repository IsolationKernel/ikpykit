# Copyright 2024 Xin Han. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from collections.abc import Iterable
from scipy import sparse as sp
import numpy as np


class KCluster(object):
    def __init__(self, id: int) -> None:
        self.id = id
        self.center = None
        self.kernel_mean_ = None
        self.points_ = []
        self.center = None

    def add_points(self, ids, X):
        self.increment_kernel_mean_(X)
        if isinstance(ids, np.integer):
            if self.center is None:
                self.center = ids
            self.points_.append(ids)
        elif isinstance(ids, Iterable):
            if self.center is None:
                raise ValueError("Cluster is not initialized.")
            self.points_.extend(ids)

    def set_center(self, center):
        self.center = center

    def delete_points(self, points, X):
        self.delete_kernel_mean_(X)
        if isinstance(points, np.integer):
            self.points_.remove(points)
        elif isinstance(points, Iterable):
            for p in points:
                self.points_.remove(p)

    def delete_kernel_mean_(self, X):
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
