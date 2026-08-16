## ikpykit.graph.IsoGraphKernel

```python
IsoGraphKernel(
    method="anne",
    n_estimators=200,
    max_samples="auto",
    random_state=None,
)
```

Bases: `BaseEstimator`

Isolation Graph Kernel is a new way to measure the similarity between two graphs.

It addresses two key issues of kernel mean embedding, where the kernel employed has: (i) a feature map with intractable dimensionality which leads to high computational cost; and (ii) data independency which leads to poor detection accuracy in anomaly detection.

Parameters:

| Name           | Type                                | Description                                                                                                                                                                                                                                   | Default  |
| -------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `method`       | `str`                               | The method to compute the isolation kernel feature. The available methods are: anne, inne, and iforest.                                                                                                                                       | `"anne"` |
| `n_estimators` | `int`                               | The number of base estimators in the ensemble.                                                                                                                                                                                                | `200`    |
| `max_samples`  | `int or float or str`               | The number of samples to draw from X to train each base estimator. - If int, then draw `max_samples` samples. - If float, then draw `max_samples` * X.shape[0]`samples. - If "auto", then`max_samples=min(8, n_samples)\`.                    | `"auto"` |
| `random_state` | `int, RandomState instance or None` | Controls the pseudo-randomness of the selection of the feature and split values for each branching step and each tree in the forest. Pass an int for reproducible results across multiple function calls. See :term:Glossary \<random_state>. | `None`   |

References

1. Bi-Cun Xu, Kai Ming Ting and Yuan Jiang. 2021. "Isolation Graph Kernel". In Proceedings of The Thirty-Fifth AAAI Conference on Artificial Intelligence. 10487-10495.

Examples:

```pycon
>>> from ikpykit.graph import IsoGraphKernel
>>> import numpy as np
>>> X = np.array([[0.4, 0.3], [0.3, 0.8], [0.5, 0.4], [0.5, 0.1]])
>>> adjacency = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
>>> igk = IsoGraphKernel()
>>> igk = igk.fit(X)
>>> embedding = igk.transform(adjacency, X, h=2)
```

Methods:

| Name            | Description                                          |
| --------------- | ---------------------------------------------------- |
| `fit`           | Fit the model on data X.                             |
| `similarity`    | Compute the isolation kernel similarity matrix of X. |
| `transform`     | Compute the isolation kernel feature of a graph.     |
| `fit_transform` | Fit the model on data X and transform X.             |

Source code in `ikpykit/graph/_isographkernel.py`

```python
def __init__(
    self,
    method: str = "anne",
    n_estimators: int = 200,
    max_samples: int | float | str = "auto",
    random_state: int | None = None,
) -> None:
    self.n_estimators = n_estimators
    self.max_samples = max_samples
    self.random_state = random_state
    self.method = method
```

### fit

```python
fit(features)
```

Fit the model on data X.

Parameters:

| Name       | Type                    | Description                                     | Default    |
| ---------- | ----------------------- | ----------------------------------------------- | ---------- |
| `features` | `(csr_matrix, ndarray)` | Features, array of shape (n_nodes, n_features). | *required* |

Returns:

| Name   | Type             | Description           |
| ------ | ---------------- | --------------------- |
| `self` | `IsoGraphKernel` | The fitted estimator. |

Source code in `ikpykit/graph/_isographkernel.py`

```python
def fit(
    self,
    features: sp.csr_matrix | np.ndarray,
):
    """Fit the model on data X.

    Parameters
    ----------
    features : sparse.csr_matrix, np.ndarray
        Features, array of shape (n_nodes, n_features).

    Returns
    -------
    self : IsoGraphKernel
        The fitted estimator.
    """
    features = check_array(features)
    self.iso_kernel_ = IsoKernel(
        self.method, self.n_estimators, self.max_samples, self.random_state
    )
    self.iso_kernel_ = self.iso_kernel_.fit(features)
    self.is_fitted_ = True
    return self
```

### similarity

```python
similarity(X, dense_output=True)
```

Compute the isolation kernel similarity matrix of X.

Parameters:

| Name           | Type         | Description                               | Default                                         |
| -------------- | ------------ | ----------------------------------------- | ----------------------------------------------- |
| `X`            | \`csr_matrix | ndarray\`                                 | The input instances or pre-computed embeddings. |
| `dense_output` | `bool`       | Whether to return dense matrix of output. | `True`                                          |

Returns:

| Name         | Type                                             | Description                                                             |
| ------------ | ------------------------------------------------ | ----------------------------------------------------------------------- |
| `similarity` | `array-like of shape (n_instances, n_instances)` | The similarity matrix organized as an n_instances * n_instances matrix. |

Source code in `ikpykit/graph/_isographkernel.py`

```python
def similarity(
    self, X: sp.csr_matrix | np.ndarray, dense_output: bool = True
) -> sp.csr_matrix | np.ndarray:
    """Compute the isolation kernel similarity matrix of X.

    Parameters
    ----------
    X: array-like of shape (n_instances, n_features)
        The input instances or pre-computed embeddings.
    dense_output: bool, default=True
        Whether to return dense matrix of output.

    Returns
    -------
    similarity : array-like of shape (n_instances, n_instances)
        The similarity matrix organized as an n_instances * n_instances matrix.
    """
    check_is_fitted(self)
    X = check_array(X)

    return safe_sparse_dot(X, X.T, dense_output=dense_output) / self.n_estimators
```

### transform

```python
transform(adjacency, features, h, dense_output=False)
```

Compute the isolation kernel feature of a graph.

Parameters:

| Name           | Type                         | Description                                               | Default    |
| -------------- | ---------------------------- | --------------------------------------------------------- | ---------- |
| `adjacency`    | `Union[csr_matrix, ndarray]` | Adjacency matrix of the graph.                            | *required* |
| `features`     | `(csr_matrix, ndarray)`      | Features, array of shape (n_nodes, n_features).           | *required* |
| `h`            | `int`                        | The number of iterations for Weisfeiler-Lehman embedding. | *required* |
| `dense_output` | `bool`                       | Whether to return a dense array.                          | `False`    |

Returns:

| Type                                                                | Description |
| ------------------------------------------------------------------- | ----------- |
| `The finite binary features based on the kernel feature map.`       |             |
| `The features are organized as an n_instances by h+1*psi*t matrix.` |             |

Source code in `ikpykit/graph/_isographkernel.py`

```python
def transform(
    self,
    adjacency: sp.csr_matrix | np.ndarray,
    features: sp.csr_matrix | np.ndarray,
    h: int,
    dense_output: bool = False,
) -> sp.csr_matrix | np.ndarray:
    """Compute the isolation kernel feature of a graph.

    Parameters
    ----------
    adjacency : Union[sp.csr_matrix, np.ndarray]
        Adjacency matrix of the graph.
    features : sparse.csr_matrix, np.ndarray
        Features, array of shape (n_nodes, n_features).
    h : int
        The number of iterations for Weisfeiler-Lehman embedding.
    dense_output : bool, default=False
        Whether to return a dense array.

    Returns
    -------
    The finite binary features based on the kernel feature map.
    The features are organized as an n_instances by h+1*psi*t matrix.
    """
    check_is_fitted(self)
    features = check_array(features)
    adjacency = check_format(adjacency)
    X_trans = self.iso_kernel_.transform(features)
    embedding = self._wlembedding(adjacency, X_trans, h)

    if dense_output:
        if sp.issparse(embedding) and hasattr(embedding, "toarray"):
            return embedding.toarray()
        else:
            warn("The IsoKernel transform output is already dense.", stacklevel=2)
    return embedding
```

### fit_transform

```python
fit_transform(adjacency, features, h, dense_output=False)
```

Fit the model on data X and transform X.

Parameters:

| Name           | Type                         | Description                                               | Default    |
| -------------- | ---------------------------- | --------------------------------------------------------- | ---------- |
| `adjacency`    | `Union[csr_matrix, ndarray]` | Adjacency matrix of the graph.                            | *required* |
| `features`     | `(csr_matrix, ndarray)`      | Features, array of shape (n_nodes, n_features).           | *required* |
| `h`            | `int`                        | The number of iterations for Weisfeiler-Lehman embedding. | *required* |
| `dense_output` | `bool`                       | Whether to return a dense array.                          | `False`    |

Returns:

| Name        | Type                         | Description        |
| ----------- | ---------------------------- | ------------------ |
| `embedding` | `Union[csr_matrix, ndarray]` | Transformed array. |

Source code in `ikpykit/graph/_isographkernel.py`

```python
def fit_transform(
    self,
    adjacency: np.ndarray | sp.csr_matrix,
    features: sp.csr_matrix | np.ndarray,
    h: int,
    dense_output: bool = False,
) -> sp.csr_matrix | np.ndarray:
    """Fit the model on data X and transform X.

    Parameters
    ----------
    adjacency : Union[sp.csr_matrix, np.ndarray]
        Adjacency matrix of the graph.
    features : sparse.csr_matrix, np.ndarray
        Features, array of shape (n_nodes, n_features).
    h : int
        The number of iterations for Weisfeiler-Lehman embedding.
    dense_output : bool, default=False
        Whether to return a dense array.

    Returns
    -------
    embedding : Union[sp.csr_matrix, np.ndarray]
        Transformed array.
    """
    self.fit(features)
    return self.transform(adjacency, features, h, dense_output)
```
