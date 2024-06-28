import numpy as np
import time


import numpy as np
from isoml.kernel import IsoKernel
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.utils import check_array
from collections.abc import Iterable
from scipy import sparse as sp

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IKDC(BaseEstimator, ClusterMixin):
    """Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version uses iforest to split the data space and calculate Isolation
    kernel Similarity. Based on this implementation, the feature
    in the Isolation kernel space is the index of the cell in Voronoi diagrams. Each
    point is represented as a binary vector such that only the cell the point falling
    into is 1.

    Parameters
    ----------

    n_estimators : int
        The number of base estimators in the ensemble.

    max_samples : int
        The number of samples to draw from X to train each base estimator.

    tau : float
        The threshold value for stopping the clustering process.

    v : float
        The decay factor for reducing the threshold value.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

    References
    ----------
    .. [1] Ye Zhu, Kai Ming Ting (2023). Kernel-based Clustering via Isolation Distributional Kernel. Information Systems.
    """

    def __init__(
        self,
        n_estimators,
        max_samples,
        method,
        k,
        kn,
        v,
        n_init_samples,
        init_id=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.method = method
        self.k = k
        self.v = v
        self.kn = kn
        self.n_init_samples = n_init_samples
        self.init_center_id = init_id
        self.random_state = random_state
        self.clusters_ = []
        self.it_ = 0
        self.labels_ = None

    def fit(self, X, y=None):
        """Fit the model on data X.
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        Returns
        -------
        self : object
        """
        X = check_array(X)
        isokernel = IsoKernel(
            method=self.method,
            max_samples=self.max_samples,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        data_ik = isokernel.fit_transform(X)
        self._fit(data_ik)
        self.is_fitted_ = True
        self.labels_ = self._get_labels(X)
        return self

    def _fit(self, ndata):
        GP = []
        clusters = []
        n_samples, n_features = ndata.shape
        data_index = np.array(range(n_samples))
        if self.init_center_id is None:
            # find modes based on sample
            rnd = check_random_state(self.random_state)
            samples_index = rnd.choice(n_samples, self.n_init_samples, replace=False)
            init_center_id = self.find_mode(ndata[samples_index], self.k, self.Kn)
            init_center_id = samples_index[init_center_id]
        else:
            init_center_id = self.init_center_id

        clusters = [KCluster(i, init_center_id[i]) for i in range(self.k)]
        for i in range(self.k):
            clusters[i].add_points(init_center_id[i], ndata[init_center_id[i]])

        data_index = np.delete(data_index, init_center_id)

        GP.append(np.column_stack((init_center_id, np.arange(1, self.k + 1))))

        C_mean = sp.vstack([c.kernel_mean for c in clusters])
        S = np.max(safe_sparse_dot(ndata[data_index], C_mean.T), axis=1)

        r = np.max(S)

        # linking points
        while len(data_index) > 0:
            C_mean = sp.vstack([c.kernel_mean for c in clusters])
            similarity = safe_sparse_dot(ndata[data_index], C_mean.T)
            T = np.argmax(similarity, axis=1)
            S = similarity[:, T]
            r = self.v * r

            if np.sum(S) == 0 or r < 0.00001:
                break

            self.it_ += 1

            DI = np.zeros_like(T)
            for i in range(self.k):
                I = np.logical_and(T == i, S > r)
                if np.sum(I) > 0:
                    clusters[i].add_points(data_index[I], ndata[data_index][I])
                    DI += I

            for ci in clusters:
                c_i_index = ci.points
                x = np.argmax(
                    safe_sparse_dot(
                        ndata[data_index][c_i_index],
                        ndata[data_index][c_i_index].sum().T,
                    )
                )
                ci.set_centre(data_index[x])

            data_index = np.delete(data_index, np.where(DI > 0)[0])

        Tclass = np.zeros((ndata.shape[0], 1))

        for i in range(len(clusters)):
            Tclass[clusters[i][:, 0] - 1] = i + 1

        # postprocessing
        t2 = time.time()

        Th = np.ceil(ndata.shape[0] * 0.01)
        OTclass = Tclass.copy()

        for _ in range(100):
            Cmean = np.zeros((k, ndata.shape[1]))
            for i in range(k):
                Cmean[i, :] = np.mean(ndata[Tclass[:, 0] == i + 1, :], axis=0)
            _, Tclass2 = np.argmax(np.dot(ndata, Cmean.T), axis=1)

            if np.sum(Tclass2 != Tclass) < Th or len(np.unique(Tclass2)) < k:
                break
            Tclass = Tclass2

        # update centres
        Centre.append([])

        for i in range(k):
            I = np.where(Tclass == i + 1)[0]
            CD = ndata[I, :]
            x = np.argmax(np.dot(CD, np.sum(CD, axis=0).T))
            Centre[it + 1] = np.concatenate((Centre[it + 1], [I[x]]))

        tr = (time.time() - t2) / (time.time() - t1)
        GP.append(np.column_stack((np.arange(1, ndata.shape[0] + 1), Tclass)))

        return Tclass, Centre, GP, it, OTclass, tr

        pass

    def find_mode(self, ndata, k, Kn):
        density = safe_sparse_dot(ndata, ndata.mean(axis=0).T)
        ik_dist = euclidean_distances(ndata, ndata, squared=True)

        # Density = Density.flatten()
        # IKDist = IKDist.flatten()

        density = self.get_lc(ik_dist, density, Kn)

        maxd = np.max(ik_dist)
        n_samples = ik_dist.shape[1]
        min_dist = np.zeros_like(density)
        sort_density = np.argsort(density)[::-1]

        min_dist[sort_density[0]] = -1
        nneigh = np.zeros_like(sort_density)

        for ii in range(1, n_samples):
            min_dist[sort_density[ii]] = maxd
            for jj in range(ii):
                if (
                    ik_dist[sort_density[ii], sort_density[jj]]
                    < min_dist[sort_density[ii]]
                ):
                    min_dist[sort_density[ii]] = ik_dist[
                        sort_density[ii], sort_density[jj]
                    ]
                    nneigh[sort_density[ii]] = sort_density[jj]

        min_dist[sort_density[0]] = np.max(min_dist)

        density = np.argsort(density) + 0.0000000001
        min_dist = np.argsort(min_dist) + 0.0000000001

        Mult = density * min_dist
        ISortMult = np.argsort(Mult)[::-1]

        ID = ISortMult[:k]

        return ID

    def get_lc(self, dist, density, k):
        # input:
        # dist: distance matrix (N*N) of a dataset
        # density: density vector (N*1) of the same dataset
        # k: k parameter for KNN

        # output:
        # LC: Local Contrast

        N = density.shape[0]
        LC = np.zeros(N)
        for i in range(N):
            inx = np.argsort(dist[i])
            knn = inx[1 : k + 1]  # K-nearest-neighbors of instance i
            LC[i] = np.sum(density[i] > density[knn])

        return LC

    @property
    def n_it(self):
        return self.it_


def DKC(ndata, k, Kn, v, s, ID):
    # Distributional Kernel Clustering

    # Input
    # ndata is the kernel feature matrix
    # k is the number of clusters
    # Kn is the kNN size
    # v is the learning rate
    # s is the sample size for mode seletcion
    # ID are the index for manually picked initial modes

    # Output
    # Tclass are the final cluster labels
    # GP are grouped points with labels in each iteration
    # Centre are cluster modes in each iteration
    # it is the iteration times
    # OTclass are cluster labels without refinement
    # tr measures the refinement time

    t1 = time.time()
    C = []
    dID = np.arange(1, ndata.shape[0] + 1)
    GP = []
    D = np.column_stack((dID, ndata))  # add index in the first column

    if ID is None:
        # find modes based on sample
        sID = np.random.choice(ndata.shape[0], s, replace=False)
        ID = find_mode(ndata[sID, :], k, Kn)
        ID = sID[ID]

    Centre = [ID]
    GP.append(np.column_stack((ID, np.arange(1, k + 1))))

    # initializing clusters
    L = ndata.shape[1]
    Csum = np.zeros((k, L))
    Csize = np.zeros((k, 1))
    for i in range(k):
        C.append(D[ID[i] - 1, :])
        Csum[i, :] = np.sum(C[i][:, 1:], axis=0)
        Csize[i, 0] = C[i][:, 1:].shape[0]
    D = np.delete(D, ID - 1, axis=0)
    it = 1

    Cmean = Csum / np.tile(Csize, (1, L))

    S, T = np.argmax(np.dot(D[:, 1:], Cmean.T), axis=1)

    r = np.max(S)

    # linking points
    while D.shape[0] > 0:
        Cmean = Csum / np.tile(Csize, (1, L))

        S, T = np.argmax(np.dot(D[:, 1:], Cmean.T), axis=1)

        r = v * r

        if np.sum(S) == 0 or r < 0.00001:
            break

        it += 1

        DI = T - T
        for i in range(k):
            I = np.logical_and(T == i + 1, S > r)
            if np.sum(I) > 0:
                C[i] = np.concatenate((C[i], D[I, :]), axis=0)
                Csum[i, :] += np.sum(D[I, 1:], axis=0)
                Csize[i, 0] += np.sum(I)
                DI += I

        Centre.append([])
        GP.append([])
        for jj in range(k):
            CD = C[jj]
            x = np.argmax(np.dot(CD[:, 1:], np.sum(CD[:, 1:], axis=0).T))
            Centre[it] = np.concatenate((Centre[it], [CD[x, 0]]))
            GP[it] = np.concatenate(
                (
                    GP[it],
                    np.column_stack((CD[:, 0], np.zeros((CD.shape[0], 1)) + jj + 1)),
                ),
                axis=0,
            )

        D = np.delete(D, np.where(DI > 0)[0], axis=0)

    Tclass = np.zeros((ndata.shape[0], 1))

    for i in range(len(C)):
        Tclass[C[i][:, 0] - 1] = i + 1

    # postprocessing
    t2 = time.time()

    Th = np.ceil(ndata.shape[0] * 0.01)
    OTclass = Tclass.copy()

    for _ in range(100):
        Cmean = np.zeros((k, ndata.shape[1]))
        for i in range(k):
            Cmean[i, :] = np.mean(ndata[Tclass[:, 0] == i + 1, :], axis=0)
        _, Tclass2 = np.argmax(np.dot(ndata, Cmean.T), axis=1)

        if np.sum(Tclass2 != Tclass) < Th or len(np.unique(Tclass2)) < k:
            break
        Tclass = Tclass2

    # update centres
    Centre.append([])

    for i in range(k):
        I = np.where(Tclass == i + 1)[0]
        CD = ndata[I, :]
        x = np.argmax(np.dot(CD, np.sum(CD, axis=0).T))
        Centre[it + 1] = np.concatenate((Centre[it + 1], [I[x]]))

    tr = (time.time() - t2) / (time.time() - t1)
    GP.append(np.column_stack((np.arange(1, ndata.shape[0] + 1), Tclass)))

    return Tclass, Centre, GP, it, OTclass, tr


class KCluster(object):
    def __init__(self, id: int, center: int) -> None:
        self.id = id
        self.center = center
        self.kernel_mean_ = None
        self.points_ = []

    def set_centre(self, center):
        self.center = center

    def add_points(self, points, X):
        self.increment_kernel_mean_(X)
        if isinstance(points, np.integer):
            self.points_.append(points)
        elif isinstance(points, Iterable):
            self.points_.extend(points)

    def increment_kernel_mean_(self, X):
        if self.kernel_mean_ is None:
            self.kernel_mean_ = X
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
