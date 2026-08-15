## ikpykit.kernel.IsoKernel

```python
IsoKernel(
    method="anne",
    n_estimators=200,
    max_samples="auto",
    random_state=None,
)
```

Bases: `TransformerMixin`, `BaseEstimator`

Isolation Kernel.

Build Isolation Kernel feature vector representations via the feature map for a given dataset.

Isolation kernel is a data dependent kernel measure that is adaptive to local data distribution and has more flexibility in capturing the characteristics of the local data distribution. It has been shown promising performance on density and distance-based classification and clustering problems.

Parameters:

| Name           | Type                                | Description                                                                                                                                                                                                                                   | Default  |
| -------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `method`       | `str`                               | The method to compute the isolation kernel feature. The available methods are: anne, inne, and iforest.                                                                                                                                       | `"anne"` |
| `n_estimators` | `int`                               | The number of base estimators in the ensemble.                                                                                                                                                                                                | `200`    |
| `max_samples`  | `int`                               | The number of samples to draw from X to train each base estimator. - If int, then draw `max_samples` samples. - If float, then draw `max_samples` * X.shape[0]`samples. - If "auto", then`max_samples=min(8, n_samples)\`.                    | `"auto"` |
| `random_state` | `int, RandomState instance or None` | Controls the pseudo-randomness of the selection of the feature and split values for each branching step and each tree in the forest. Pass an int for reproducible results across multiple function calls. See :term:Glossary \<random_state>. | `None`   |

References

.. [1] Qin, X., Ting, K.M., Zhu, Y. and Lee, V.C. "Nearest-neighbour-induced isolation similarity and its impact on density-based clustering". In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, 2019, July, pp. 4755-4762

Examples:

```pycon
>>> from ikpykit.kernel import IsoKernel
>>> import numpy as np
>>> X = [[0.4,0.3], [0.3,0.8], [0.5, 0.4], [0.5, 0.1]]
>>> ik = IsoKernel().fit(X)
>>> X_trans = ik.transform(X)
>>> X_sim = ik.similarity(X)
```

Methods:

| Name         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `fit`        | Fit the model on data X.                             |
| `similarity` | Compute the isolation kernel similarity matrix of X. |
| `transform`  | Compute the isolation kernel feature of X.           |

Source code in `ikpykit/kernel/_isokernel.py`

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

Source code in `ikpykit/kernel/_isokernel.py`

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
    n_samples = X.shape[0]
    if isinstance(self.max_samples, str):
        if self.max_samples == "auto":
            max_samples = min(16, n_samples)
        else:
            raise ValueError(
                f"max_samples ({self.max_samples}) is not supported."
                'Valid choices are: "auto", int or'
                "float"
            )
    elif isinstance(self.max_samples, numbers.Integral):
        if self.max_samples > n_samples:
            warn(
                f"max_samples ({self.max_samples}) is greater than the "
                f"total number of samples ({n_samples}). max_samples "
                "will be set to n_samples for estimation.",
                stacklevel=2,
            )
            max_samples = n_samples
        else:
            max_samples = self.max_samples
    else:  # float
        if not 0.0 < self.max_samples <= 1.0:
            raise ValueError(
                f"max_samples must be in (0, 1], got {self.max_samples!r}"
            )
        max_samples = int(self.max_samples * X.shape[0])
    self.max_samples_ = max_samples

    if self.method == "anne":
        self.iso_kernel_ = IK_ANNE(
            self.n_estimators, self.max_samples_, self.random_state
        )
    elif self.method == "inne":
        self.iso_kernel_ = IK_INNE(
            self.n_estimators, self.max_samples_, self.random_state
        )
    elif self.method == "iforest":
        self.iso_kernel_ = IK_IForest(
            self.n_estimators, self.max_samples_, self.random_state
        )
    else:
        raise ValueError(
            f"method ({self.method}) is not supported."
            'Valid choices are: "anne", "inne" or "iforest"'
        )

    self.iso_kernel_.fit(X)
    self.is_fitted_ = True
    return self
```

### similarity

```python
similarity(X, dense_output=True)
```

Compute the isolation kernel similarity matrix of X.

Parameters:

| Name           | Type | Description                               | Default    |
| -------------- | ---- | ----------------------------------------- | ---------- |
| `X`            |      | The input instances.                      | *required* |
| `dense_output` |      | Whether to return dense matrix of output. | `True`     |

Returns:

| Type                                                                         | Description |
| ---------------------------------------------------------------------------- | ----------- |
| `The simalarity matrix are organised as a n_instances * n_instances matrix.` |             |

Source code in `ikpykit/kernel/_isokernel.py`

```python
def similarity(self, X, dense_output=True):
    """Compute the isolation kernel similarity matrix of X.
    Parameters
    ----------
    X: array-like of shape (n_instances, n_features)
        The input instances.
    dense_output: bool, default=True
        Whether to return dense matrix of output.
    Returns
    -------
    The simalarity matrix are organised as a n_instances * n_instances matrix.
    """
    check_is_fitted(self)
    X = check_array(X)
    embed_X = self.transform(X)
    return (
        safe_sparse_dot(embed_X, embed_X.T, dense_output=dense_output)
        / self.n_estimators
    )
```

### transform

```python
transform(X, dense_output=False)
```

Compute the isolation kernel feature of X.

Parameters:

| Name           | Type | Description                               | Default    |
| -------------- | ---- | ----------------------------------------- | ---------- |
| `X`            |      | The input instances.                      | *required* |
| `dense_output` |      | Whether to return dense matrix of output. | `False`    |

Returns:

| Type                                                           | Description |
| -------------------------------------------------------------- | ----------- |
| `The finite binary features based on the kernel feature map.`  |             |
| `The features are organised as a n_instances by psi*t matrix.` |             |

Source code in `ikpykit/kernel/_isokernel.py`

```python
def transform(self, X, dense_output=False):
    """Compute the isolation kernel feature of X.
    Parameters
    ----------
    X: array-like of shape (n_instances, n_features)
        The input instances.
    dense_output: bool, default=False
        Whether to return dense matrix of output.
    Returns
    -------
    The finite binary features based on the kernel feature map.
    The features are organised as a n_instances by psi*t matrix.
    """

    check_is_fitted(self)
    X = check_array(X)
    X_trans = self.iso_kernel_.transform(X)
    if dense_output:
        if sp.issparse(X_trans) and hasattr(X_trans, "toarray"):
            return X_trans.toarray()
        else:
            warn("The IsoKernel transform output is already dense.", stacklevel=2)
    return X_trans
```
