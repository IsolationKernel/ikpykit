## ikpykit.cluster.PSKC

```python
PSKC(
    n_estimators=200,
    max_samples="auto",
    method="inne",
    tau=0.1,
    v=0.1,
    random_state=None,
)
```

Bases: `BaseEstimator`, `ClusterMixin`

Point-Set Kernel Clustering algorithm using Isolation Kernels.

PSKC is a clustering algorithm that leverages Isolation Kernels to create feature vector representations of data points. It adaptively captures the characteristics of local data distributions by using data-dependent kernels. The algorithm forms clusters by identifying points with high similarity in the transformed kernel space.

The clustering process works by iteratively:

1. Selecting a center point with maximum similarity to the mean
1. Forming a cluster around this center
1. Removing these points from consideration
1. Continuing until stopping criteria are met

n_estimators : int, default=200 The number of base estimators (trees) in the isolation ensemble.

max_samples : int or str, default="auto"

- If int, then draw `max_samples` samples.
- If "auto", then `max_samples=min(256, n_samples)`.

method : {'inne', 'anne'}, default='inne' The method used for building the isolation kernel.

tau : float, default=0.1 Lower values result in more clusters.

v : float, default=0.1 The decay factor for reducing the similarity threshold. Controls the expansion of clusters.

```
Controls the pseudo-randomness of the algorithm for reproducibility.
Pass an int for reproducible results across multiple function calls.
```

Attributes clusters\_ : list List of KCluster objects representing the identified clusters.

labels\_ : ndarray of shape (n_samples,) Cluster labels for each point in the dataset.

centers : list Centers of each cluster in the transformed feature space.

n_classes : int Number of clusters found.

Examples:

```pycon
>>> from ikpykit.cluster import PSKC
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [10, 2], [10, 10],  [1, 0], [1, 1]])
>>> pskc = PSKC(n_estimators=100, max_samples=2, tau=0.3, v=0.1, random_state=24)
>>> pskc.fit_predict(X)
array([0, 0, 1, 1, 0, 0])
```

References

.. [1] Kai Ming Ting, Jonathan R. Wells, Ye Zhu (2023) "Point-set Kernel Clustering". IEEE Transactions on Knowledge and Data Engineering. Vol.35, 5147-5158.

Methods:

| Name  | Description              |
| ----- | ------------------------ |
| `fit` | Fit the model on data X. |

Source code in `ikpykit/cluster/_pskc.py`

```python
def __init__(
    self,
    n_estimators=200,
    max_samples="auto",
    method="inne",
    tau=0.1,
    v=0.1,
    random_state=None,
):
    self.n_estimators = n_estimators
    self.max_samples = max_samples
    self.method = method
    self.tau = tau
    self.v = v
    self.random_state = random_state
    self.clusters_ = []
    self.labels_ = None
```

### fit

```python
fit(X, y=None)
```

Fit the model on data X.

Parameters:

| Name | Type                                        | Description          | Default    |
| ---- | ------------------------------------------- | -------------------- | ---------- |
| `X`  | `np.array of shape (n_samples, n_features)` | The input instances. | *required* |

Returns:

| Name   | Type     | Description |
| ------ | -------- | ----------- |
| `self` | `object` |             |

Source code in `ikpykit/cluster/_pskc.py`

```python
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
        max_samples=self.max_samples,
        n_estimators=self.n_estimators,
        random_state=self.random_state,
        method=self.method,
    )
    ndata = isokernel.fit_transform(X)
    self._fit(ndata)
    self.is_fitted_ = True
    self.labels_ = self._get_labels(X)
    return self
```
