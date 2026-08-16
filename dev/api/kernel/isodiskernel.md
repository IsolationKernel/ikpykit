## ikpykit.kernel.IsoDisKernel

```python
IsoDisKernel(
    method="anne",
    n_estimators=200,
    max_samples="auto",
    random_state=None,
)
```

Bases: `BaseEstimator`, `TransformerMixin`

Isolation Distributional Kernel is a new way to measure the similarity between two distributions.

It addresses two key issues of kernel mean embedding, where the kernel employed has: (i) a feature map with intractable dimensionality which leads to high computational cost; and (ii) data independency which leads to poor detection accuracy in anomaly detection.

Parameters:

| Name           | Type                                | Description                                                                                                                                                                                                                                   | Default  |
| -------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `method`       | `str`                               | The method to compute the isolation kernel feature. The available methods are: anne, inne, and iforest.                                                                                                                                       | `"anne"` |
| `n_estimators` | `int`                               | The number of base estimators in the ensemble.                                                                                                                                                                                                | `200`    |
| `max_samples`  | `int`                               | The number of samples to draw from X to train each base estimator. - If int, then draw `max_samples` samples. - If float, then draw `max_samples` * X.shape[0]`samples. - If "auto", then`max_samples=min(8, n_samples)\`.                    | `"auto"` |
| `random_state` | `int, RandomState instance or None` | Controls the pseudo-randomness of the selection of the feature and split values for each branching step and each tree in the forest. Pass an int for reproducible results across multiple function calls. See :term:Glossary \<random_state>. | `None`   |

References

1. Kai Ming Ting, Bi-Cun Xu, Takashi Washio, and Zhi-Hua Zhou. 2020. "Isolation Distributional Kernel: A New Tool for Kernel based Anomaly Detection". In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 198-206.

Examples:

```pycon
>>> from ikpykit.kernel import IsoDisKernel
>>> import numpy as np
>>> X = [[0.4,0.3], [0.3,0.8], [0.5,0.4], [0.5,0.1]]
>>> idk = IsoDisKernel(max_samples=3,).fit(X)
>>> D_i = [[0.4,0.3], [0.3,0.8]]
>>> D_j = [[0.5, 0.4], [0.5, 0.1]]
>>> idk.similarity(D_j, D_j)
1.0
```

Methods:

| Name          | Description                                               |
| ------------- | --------------------------------------------------------- |
| `fit`         | Fit the model on data X.                                  |
| `kernel_mean` | Compute the kernel mean embedding of X.                   |
| `similarity`  | Compute the isolation distribution kernel of D_i and D_j. |
| `transform`   | Compute the isolation kernel feature of D_i and D_j.      |

Source code in `ikpykit/kernel/_isodiskernel.py`

```python
def __init__(
    self, method="anne", n_estimators=200, max_samples="auto", random_state=None
) -> None:
    self.n_estimators = n_estimators
    self.max_samples = max_samples
    self.random_state = random_state
    self.method = method
```

### fit

```python
fit(X)
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

Source code in `ikpykit/kernel/_isodiskernel.py`

```python
def fit(self, X):
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
    iso_kernel = IsoKernel(
        self.method, self.n_estimators, self.max_samples, self.random_state
    )
    self.iso_kernel_ = iso_kernel.fit(X)
    self.is_fitted_ = True
    return self
```

### kernel_mean

```python
kernel_mean(X)
```

Compute the kernel mean embedding of X.

Source code in `ikpykit/kernel/_isodiskernel.py`

```python
def kernel_mean(self, X):
    """Compute the kernel mean embedding of X."""
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.mean(X, axis=0)
```

### similarity

```python
similarity(D_i, D_j, is_normalize=True)
```

Compute the isolation distribution kernel of D_i and D_j.

Parameters:

| Name           | Type | Description          | Default    |
| -------------- | ---- | -------------------- | ---------- |
| `D_i`          |      | The input instances. | *required* |
| `D_j`          |      | The input instances. | *required* |
| `is_normalize` |      |                      | `True`     |

Returns:

| Type                                                          | Description |
| ------------------------------------------------------------- | ----------- |
| `The Isolation distribution similarity of given two dataset.` |             |

Source code in `ikpykit/kernel/_isodiskernel.py`

```python
def similarity(self, D_i, D_j, is_normalize=True):
    """Compute the isolation distribution kernel of D_i and D_j.
    Parameters
    ----------
    D_i: array-like of shape (n_instances, n_features)
        The input instances.
    D_j: array-like of shape (n_instances, n_features)
        The input instances.
    is_normalize: whether return the normalized similarity matrix ranged of [0,1]. Default: False
    Returns
    -------
    The Isolation distribution similarity of given two dataset.
    """
    emb_D_i, emb_D_j = self.transform(D_i, D_j)
    kme_D_i, kme_D_j = self.kernel_mean(emb_D_i), self.kernel_mean(emb_D_j)
    return self.kme_similarity(kme_D_i, kme_D_j, is_normalize=is_normalize)
```

### transform

```python
transform(D_i, D_j)
```

Compute the isolation kernel feature of D_i and D_j.

Parameters:

| Name  | Type | Description          | Default    |
| ----- | ---- | -------------------- | ---------- |
| `D_i` |      | The input instances. | *required* |
| `D_j` |      | The input instances. | *required* |

Returns:

| Type                                                           | Description |
| -------------------------------------------------------------- | ----------- |
| `The finite binary features based on the kernel feature map.`  |             |
| `The features are organised as a n_instances by psi*t matrix.` |             |

Source code in `ikpykit/kernel/_isodiskernel.py`

```python
def transform(self, D_i, D_j):
    """Compute the isolation kernel feature of D_i and D_j.
    Parameters
    ----------
    D_i: array-like of shape (n_instances, n_features)
        The input instances.
    D_j: array-like of shape (n_instances, n_features)
        The input instances.
    Returns
    -------
    The finite binary features based on the kernel feature map.
    The features are organised as a n_instances by psi*t matrix.
    """
    check_is_fitted(self)
    D_i = check_array(D_i)
    D_j = check_array(D_j)
    return self.iso_kernel_.transform(D_i), self.iso_kernel_.transform(D_j)
```
