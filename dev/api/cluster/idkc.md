## ikpykit.cluster.IDKC

```python
IDKC(
    n_estimators,
    max_samples,
    method,
    k,
    kn,
    v,
    n_init_samples,
    init_center=None,
    is_post_process=True,
    force_assign_unassigned=False,
    random_state=None,
)
```

Bases: `BaseEstimator`, `ClusterMixin`

Isolation Distributional Kernel Clustering.

A clustering algorithm that leverages Isolation Kernels to transform data into a feature space where cluster structures are more distinguishable. The algorithm first constructs Isolation Kernel representations, then performs clustering in this transformed space using a threshold-based assignment mechanism.

Parameters:

| Name                      | Type                                | Description                                                                                                                                                                                                                                                                                             | Default    |
| ------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `n_estimators`            | `int`                               | Number of base estimators in the ensemble for the Isolation Kernel. Higher values generally lead to more stable results but increase computation time.                                                                                                                                                  | *required* |
| `max_samples`             | `int`                               | Number of samples to draw from X to train each base estimator in the Isolation Kernel. Controls the granularity of the kernel representation.                                                                                                                                                           | *required* |
| `method`                  | `(inne, anne, iforest)`             | Method used to calculate the Isolation Kernel: - 'inne': Isolation Nearest Neighbor Ensemble - 'anne': Approximate Nearest Neighbor Ensemble - 'iforest': Isolation Forest                                                                                                                              | `'inne'`   |
| `k`                       | `int`                               | Number of clusters to form in the dataset.                                                                                                                                                                                                                                                              | *required* |
| `kn`                      | `int`                               | Number of nearest neighbors used for local contrast density calculation during initialization. Higher values consider more neighbors when determining density.                                                                                                                                          | *required* |
| `v`                       | `float`                             | Decay factor (0 < v < 1) for reducing the similarity threshold during clustering. Smaller values cause faster decay, leading to more aggressive cluster assignments.                                                                                                                                    | *required* |
| `n_init_samples`          | `int or float`                      | If int, number of samples to consider when initializing cluster centers. If float, fraction of total samples to consider when initializing cluster centers. Number of samples to consider when initializing cluster centers. Larger values may produce better initial centers but increase computation. | *required* |
| `init_center`             | `int or array-like of shape (k,)`   | Index or indices of initial cluster centers. If None, centers are selected automatically based on density and distance considerations.                                                                                                                                                                  | `None`     |
| `is_post_process`         | `bool`                              | Whether to perform post-processing refinement of clusters through iterative reassignment. Improves cluster quality but adds computational overhead.                                                                                                                                                     | `True`     |
| `force_assign_unassigned` | `bool`                              | Whether to force-assign all remaining unassigned points (label -1) to their nearest existing cluster center at the end of training.                                                                                                                                                                     | `False`    |
| `random_state`            | `int, RandomState instance or None` | Controls the randomness of the algorithm. Pass an int for reproducible results.                                                                                                                                                                                                                         | `None`     |

Attributes:

| Name         | Type                            | Description                                                                                 |
| ------------ | ------------------------------- | ------------------------------------------------------------------------------------------- |
| `clusters_`  | `list of KCluster objects`      | The cluster objects containing assignment and centroid information.                         |
| `it_`        | `int`                           | Number of iterations performed during the initial clustering phase.                         |
| `labels_`    | `ndarray of shape (n_samples,)` | Cluster labels for each point. Points not assigned to any cluster have label -1 (outliers). |
| `is_fitted_` | `bool`                          | Whether the model has been fitted to data.                                                  |

Examples:

```pycon
>>> from ikpykit.cluster import IDKC
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [5, 2], [5, 5],  [1, 0], [5, 0]])
>>> clustering = IDKC(
...     n_estimators=100, max_samples=3, method='anne',
...     k=2, kn=5, v=0.5, n_init_samples=4, random_state=42
... )
>>> clustering.fit_predict(X)
array([1, 1, 0, 0, 1, 0])
```

References

.. [1] Ye Zhu, Kai Ming Ting (2023). Kernel-based Clustering via Isolation Distributional Kernel. Information Systems.

Methods:

| Name      | Description                                     |
| --------- | ----------------------------------------------- |
| `fit`     | Fit the IDKC clustering model on data X.        |
| `predict` | Predict the cluster labels for each point in X. |

Source code in `ikpykit/cluster/_idkc.py`

```python
def __init__(
    self,
    n_estimators,
    max_samples,
    method,
    k,
    kn,
    v,
    n_init_samples,
    init_center=None,
    is_post_process=True,
    force_assign_unassigned=False,
    random_state=None,
):
    self.n_estimators = n_estimators
    self.max_samples = max_samples
    self.method = method
    self.k = k
    self.kn = kn
    self.v = v
    self.n_init_samples = n_init_samples
    self.is_post_process = is_post_process
    self.force_assign_unassigned = force_assign_unassigned
    self.init_center = init_center
    self.random_state = random_state
    self.clusters_ = []
    self.it_ = 0
    self.labels_ = None
    self.data_index = None
```

### n_it

```python
n_it
```

Get number of iterations performed during clustering.

### fit

```python
fit(X, y=None)
```

Fit the IDKC clustering model on data X.

Parameters:

| Name | Type                                       | Description                                          | Default    |
| ---- | ------------------------------------------ | ---------------------------------------------------- | ---------- |
| `X`  | `ndarray of shape (n_samples, n_features)` | The input instances to cluster.                      | *required* |
| `y`  | `Ignored`                                  | Not used, present for API consistency by convention. | `None`     |

Returns:

| Name   | Type     | Description       |
| ------ | -------- | ----------------- |
| `self` | `object` | Fitted estimator. |

Source code in `ikpykit/cluster/_idkc.py`

```python
def fit(self, X, y=None):
    """Fit the IDKC clustering model on data X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input instances to cluster.
    y : Ignored
        Not used, present for API consistency by convention.

    Returns
    -------
    self : object
        Fitted estimator.
    """
    X = check_array(X)
    if self.n_init_samples <= 0:
        raise ValueError(
            f"Number of initial samples n_init_samples={self.n_init_samples} must be greater than 0"
        )
    elif isinstance(self.n_init_samples, numbers.Integral):
        if self.n_init_samples > X.shape[0]:
            self.n_init_samples = X.shape[0]
            raise warn(
                f"Number of initial samples n_init_samples={self.n_init_samples} is greater than the number of samples in the dataset. Setting n_init_samples to {X.shape[0]}",
                stacklevel=2,
            )
        else:
            self.n_init_samples = int(self.n_init_samples)
    elif isinstance(self.n_init_samples, float):
        if not (0 < self.n_init_samples <= 1):
            raise ValueError(
                f"Fraction of initial samples n_init_samples={self.n_init_samples} must be between 0 and 1"
            )
        self.n_init_samples = int(self.n_init_samples * X.shape[0])
    self.data_index = np.arange(X.shape[0])
    isokernel = IsoKernel(
        method=self.method,
        max_samples=self.max_samples,
        n_estimators=self.n_estimators,
        random_state=self.random_state,
    )
    data_ik = isokernel.fit_transform(X)
    self._fit(data_ik)

    # Apply post-processing if requested
    if self.is_post_process:
        self._post_process(data_ik)

    if self.force_assign_unassigned:
        self._force_assign_unassigned_points(data_ik)

    self.is_fitted_ = True
    self.labels_ = self._get_labels(data_ik)
    return self
```

### predict

```python
predict(X)
```

Predict the cluster labels for each point in X.

Parameters:

| Name | Type                                       | Description                                        | Default    |
| ---- | ------------------------------------------ | -------------------------------------------------- | ---------- |
| `X`  | `ndarray of shape (n_samples, n_features)` | The input instances to predict cluster labels for. | *required* |

Returns:

| Name     | Type                            | Description                                                                                 |
| -------- | ------------------------------- | ------------------------------------------------------------------------------------------- |
| `labels` | `ndarray of shape (n_samples,)` | Cluster labels for each point. Points not assigned to any cluster have label -1 (outliers). |

Source code in `ikpykit/cluster/_idkc.py`

```python
def predict(self, X):
    """Predict the cluster labels for each point in X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input instances to predict cluster labels for.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point. Points not assigned to any cluster
        have label -1 (outliers).
    """
    X = check_array(X)

    return self._get_labels
```
