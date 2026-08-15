## ikpykit.cluster.IKAHC

```python
IKAHC(
    n_estimators=200,
    max_samples="auto",
    lk_method="single",
    ik_method="anne",
    return_flat=False,
    t=None,
    n_clusters=None,
    criterion="distance",
    random_state=None,
)
```

Bases: `BaseEstimator`, `ClusterMixin`

IKAHC is a novel hierarchical clustering algorithm. It uses a data-dependent kernel called Isolation Kernel to measure the similarity between clusters.

Parameters:

| Name           | Type                                    | Description                                                                                                                                                                                                                 | Default      |
| -------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `n_estimators` | `int`                                   | The number of base estimators in the ensemble.                                                                                                                                                                              | `200`        |
| `max_samples`  | `int or float or str`                   | The number of samples to draw from X to train each base estimator. - If int, then draw `max_samples` samples. - If float, then draw `max_samples \* X.shape[0]` samples. - If "auto", then `max_samples=min(8, n_samples)`. | `"auto"`     |
| `ik_method`    | `Literal['inne', 'anne']`               | Isolation method to use. The original algorithm in paper is "anne".                                                                                                                                                         | `'anne'`     |
| `lk_method`    | `(single, complete, average, weighted)` | The linkage algorithm to use. The supported Linkage Methods are 'single', 'complete', 'average' and 'weighted'.                                                                                                             | `"single"`   |
| `return_flat`  | `bool`                                  | Whether to return flat clusters that extract from the fitted dendrogram.                                                                                                                                                    | `False`      |
| `t`            | `float`                                 | The threshold to apply when forming flat clusters. Either t or n_clusters should be provided.                                                                                                                               | `None`       |
| `n_clusters`   | `int`                                   | The number of flat clusters to form. Either t or n_clusters should be provided.                                                                                                                                             | `None`       |
| `criterion`    | `str`                                   | The criterion to use in forming flat clusters. Valid options are 'distance', 'inconsistent', 'maxclust', or 'monocrit'.                                                                                                     | `'distance'` |
| `random_state` | `int, RandomState instance or None`     | Controls the pseudo-randomness of the selection of the samples to fit the Isolation Kernel. Pass an int for reproducible results across multiple function calls. See :term:Glossary \<random_state>.                        | `None`       |

Attributes:

| Name         | Type        | Description                                                       |
| ------------ | ----------- | ----------------------------------------------------------------- |
| `isokernel`  | `IsoKernel` | Fitted isolation kernel.                                          |
| `dendrogram` | `ndarray`   | Cluster hierarchy as computed by scipy.cluster.hierarchy.linkage. |

References

.. [1] Xin Han, Ye Zhu, Kai Ming Ting, and Gang Li, "The Impact of Isolation Kernel on Agglomerative Hierarchical Clustering Algorithms", Pattern Recognition, 2023, 139: 109517.

Examples:

```pycon
>>> from ikpykit.cluster import IKAHC
>>> import numpy as np
>>> X = [[0.4,0.3], [0.3,0.8], [0.5, 0.4], [0.5, 0.1]]
>>> clf = IKAHC(n_estimators=200, max_samples=2, lk_method='single', n_clusters=2, return_flat=True)
>>> clf.fit_predict(X)
array([1, 2, 1, 1], dtype=int32)
```

Methods:

| Name            | Description                                          |
| --------------- | ---------------------------------------------------- |
| `fit`           | Fit the IKAHC clustering model.                      |
| `fit_transform` | Fit algorithm to data and return the dendrogram.     |
| `fit_predict`   | Fit algorithm to data and return the cluster labels. |

Source code in `ikpykit/cluster/_ikahc.py`

```python
def __init__(
    self,
    n_estimators: int = 200,
    max_samples: int | float | str = "auto",
    lk_method: Literal["single", "complete", "average", "weighted"] = "single",
    ik_method: Literal["inne", "anne"] = "anne",
    return_flat: bool = False,
    t: float | None = None,
    n_clusters: int | None = None,
    criterion: str = "distance",
    random_state: int | np.random.RandomState | None = None,
):
    self.n_estimators = n_estimators
    self.max_samples = max_samples
    self.ik_method = ik_method
    self.lk_method = lk_method
    self.return_flat = return_flat
    self.t = t
    self.n_clusters = n_clusters
    self.criterion = criterion
    self.random_state = random_state
    self.labels_ = None
```

### dendrogram

```python
dendrogram
```

Get the dendrogram of the hierarchical clustering.

Returns:

| Name          | Type      | Description                                              |
| ------------- | --------- | -------------------------------------------------------- |
| `dendrogram_` | `ndarray` | The dendrogram representing the hierarchical clustering. |

### isokernel

```python
isokernel
```

Get the fitted isolation kernel.

Returns:

| Name         | Type        | Description                  |
| ------------ | ----------- | ---------------------------- |
| `isokernel_` | `IsoKernel` | The fitted isolation kernel. |

### fit

```python
fit(X)
```

Fit the IKAHC clustering model.

Parameters:

| Name | Type                                          | Description        | Default    |
| ---- | --------------------------------------------- | ------------------ | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples. | *required* |

Returns:

| Name   | Type     | Description       |
| ------ | -------- | ----------------- |
| `self` | `object` | Fitted estimator. |

Source code in `ikpykit/cluster/_ikahc.py`

```python
def fit(self, X: np.ndarray) -> "IKAHC":
    """Fit the IKAHC clustering model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    self : object
        Fitted estimator.
    """
    # Check data
    X = check_array(X, accept_sparse=False)

    # Validate parameters
    if self.lk_method not in ["single", "complete", "average", "weighted"]:
        raise ValueError(
            f"lk_method must be one of 'single', 'complete', 'average', 'weighted', got {self.lk_method}"
        )

    if self.ik_method not in ["inne", "anne"]:
        raise ValueError(
            f"ik_method must be one of 'inne', 'anne', got {self.ik_method}"
        )

    if self.n_estimators <= 0:
        raise ValueError(f"n_estimators must be positive, got {self.n_estimators}")

    # Check if both t and n_clusters are provided at initialization
    if self.return_flat and self.t is not None and self.n_clusters is not None:
        raise ValueError(
            "Specify either a distance threshold t or n_clusters, not both."
        )

    # Fit isolation kernel
    self.isokernel_ = IsoKernel(
        method=self.ik_method,
        n_estimators=self.n_estimators,
        max_samples=self.max_samples,
        random_state=self.random_state,
    )
    self.isokernel_ = self.isokernel_.fit(X)

    # Calculate similarity matrix and convert to distance matrix (1-similarity)
    similarity_matrix = self.isokernel_.similarity(X)
    self.dendrogram_ = linkage(1 - similarity_matrix, method=self.lk_method)

    if self.return_flat:
        self.labels_ = self._extract_flat_cluster()

    return self
```

### fit_transform

```python
fit_transform(X, y=None)
```

Fit algorithm to data and return the dendrogram.

Parameters:

| Name | Type                                          | Description                                          | Default    |
| ---- | --------------------------------------------- | ---------------------------------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples.                                   | *required* |
| `y`  | `Ignored`                                     | Not used, present for API consistency by convention. | `None`     |

Returns:

| Name         | Type      | Description                                          |
| ------------ | --------- | ---------------------------------------------------- |
| `dendrogram` | `ndarray` | Dendrogram representing the hierarchical clustering. |

Source code in `ikpykit/cluster/_ikahc.py`

```python
def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
    """Fit algorithm to data and return the dendrogram.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    y : Ignored
        Not used, present for API consistency by convention.

    Returns
    -------
    dendrogram : np.ndarray
        Dendrogram representing the hierarchical clustering.
    """
    self.fit(X)
    return self.dendrogram
```

### fit_predict

```python
fit_predict(X, y=None)
```

Fit algorithm to data and return the cluster labels.

Source code in `ikpykit/cluster/_ikahc.py`

```python
def fit_predict(self, X, y=None):
    """Fit algorithm to data and return the cluster labels."""
    return super().fit_predict(X, y)
```
