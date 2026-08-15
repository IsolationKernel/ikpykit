## ikpykit.stream.STREAMKHC

```python
STREAMKHC(
    method="anne",
    n_estimators=200,
    max_samples="auto",
    max_leaf=5000,
    random_state=None,
)
```

Bases: `BaseEstimator`, `ClusterMixin`

Streaming Hierarchical Clustering Based on Point-Set Kernel.

This algorithm performs hierarchical clustering on streaming data using isolation kernel techniques. It builds a tree structure that adapts as new data points arrive, allowing for efficient online clustering.

Parameters:

| Name           | Type                                | Description                                                                                                                                                                                                          | Default  |
| -------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `method`       | `str`                               | The method used to calculate the Isolation Kernel. Possible values are 'inne' and 'anne'.                                                                                                                            | `"anne"` |
| `n_estimators` | `int`                               | The number of base estimators in the isolation kernel.                                                                                                                                                               | `200`    |
| `max_samples`  | `(str, int or float)`               | The number of samples to draw from X to train each base estimator. - If int, then draw max_samples samples. - If float, then draw max_samples * X.shape[0] samples. - If "auto", then max_samples=min(8, n_samples). | `"auto"` |
| `max_leaf`     | `int`                               | Maximum number of data points to maintain in the clustering tree. When exceeded, the oldest points will be removed.                                                                                                  | `5000`   |
| `random_state` | `int, RandomState instance or None` | Controls the randomness of the estimator.                                                                                                                                                                            | `None`   |

Attributes:

| Name             | Type        | Description                                            |
| ---------------- | ----------- | ------------------------------------------------------ |
| `tree_`          | `INODE`     | The root node of the hierarchical clustering tree.     |
| `iso_kernel_`    | `IsoKernel` | The isolation kernel used for data transformation.     |
| `point_counter_` | `int`       | Counter tracking the total number of points processed. |
| `n_features_in_` | `int`       | Number of features seen during fit.                    |

Examples:

```pycon
>>> from ikpykit.stream import STREAMKHC
>>> import numpy as np
>>> # Generate sample data
>>> X = np.random.rand(100, 10)  # 100 samples with 10 features
>>> y = np.random.randint(0, 3, size=100)  # Optional class labels
>>> # Initialize and fit the model with a batch
>>> clusterer = STREAMKHC(n_estimators=100, random_state=42)
>>> clusterer =  clusterer.fit(X, y)
>>> # Process new streaming data
>>> new_data = np.random.rand(1, 10)  # 10 new samples
>>> new_labels = np.random.randint(0, 3, size=1)  # Optional class labels
>>> clusterer = clusterer.fit_online(new_data, new_labels)
>>> # Calculate clustering purity (if labels were provided)
>>> purity = clusterer.get_purity()
```

References

.. [1] Xin Han, Ye Zhu, Kai Ming Ting, De-Chuan Zhan, Gang Li (2022) Streaming Hierarchical Clustering Based on Point-Set Kernel. Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

Methods:

| Name             | Description                                   |
| ---------------- | --------------------------------------------- |
| `fit`            | Fit the model with a batch of data points.    |
| `fit_online`     | Fit the model with a stream of data points.   |
| `get_purity`     | Calculate the purity of the clustering tree.  |
| `serialize_tree` | Serialize the clustering tree to a file.      |
| `visualize_tree` | Visualize the clustering tree using Graphviz. |

Source code in `ikpykit/stream/cluster/_streakhc.py`

```python
def __init__(
    self,
    method: Literal["inne", "anne"] = "anne",
    n_estimators: int = 200,
    max_samples: Literal["auto"] | int | float = "auto",
    max_leaf: int = 5000,
    random_state: int | np.random.RandomState | None = None,
):
    self.method = method
    self.n_estimators = n_estimators
    self.max_samples = max_samples
    self.max_leaf = max_leaf
    self.random_state = random_state
    self.tree_ = None
    self.point_counter_ = 0
    self.iso_kernel_ = None
    self.n_features_in_ = None
```

### fit

```python
fit(X, y=None)
```

Fit the model with a batch of data points.

Parameters:

| Name | Type                                                        | Description                                                                                                                                                         | Default    |
| ---- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)`               | The input data points.                                                                                                                                              | *required* |
| `y`  | `array-like of shape (n_samples,), optional (default=None)` | The labels of the data points. Not used in clustering processing, just for calculating purity. If not provided, the model will generate a tree with a single label. | `None`     |

Returns:

| Name   | Type        | Description   |
| ------ | ----------- | ------------- |
| `self` | `STREAMKHC` | Returns self. |

Raises:

| Type         | Description                                            |
| ------------ | ------------------------------------------------------ |
| `ValueError` | If parameters are invalid or data has incorrect shape. |

Source code in `ikpykit/stream/cluster/_streakhc.py`

```python
def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> STREAMKHC:
    """Fit the model with a batch of data points.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data points.
    y : array-like of shape (n_samples,), optional (default=None)
        The labels of the data points.
        Not used in clustering processing, just for calculating purity.
        If not provided, the model will generate a tree with a single label.

    Returns
    -------
    self : STREAMKHC
        Returns self.

    Raises
    ------
    ValueError
        If parameters are invalid or data has incorrect shape.
    """

    if isinstance(self.max_leaf, int) and self.max_leaf <= 0:
        raise ValueError(f"max_leaf must be positive, got {self.max_leaf}")

    # Process input data
    X = check_array(X, accept_sparse=False)
    if y is None:
        y = np.ones(X.shape[0], dtype=np.int64)
    else:
        X, y = check_X_y(X, y, accept_sparse=False)

    self.n_features_in_ = X.shape[1]
    self._initialize_tree(X, y)
    return self
```

### fit_online

```python
fit_online(X, y=None)
```

Fit the model with a stream of data points.

Parameters:

| Name | Type                                                        | Description                                                                                                                                                         | Default    |
| ---- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)`               | The input data points.                                                                                                                                              | *required* |
| `y`  | `array-like of shape (n_samples,), optional (default=None)` | The labels of the data points. Not used in clustering processing, just for calculating purity. If not provided, the model will generate a tree with a single label. | `None`     |

Returns:

| Name   | Type        | Description   |
| ------ | ----------- | ------------- |
| `self` | `STREAMKHC` | Returns self. |

Raises:

| Type             | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `NotFittedError` | If the model has not been initialized with fit.               |
| `ValueError`     | If X has a different number of features than seen during fit. |

Source code in `ikpykit/stream/cluster/_streakhc.py`

```python
def fit_online(self, X: np.ndarray, y: np.ndarray | None = None) -> STREAMKHC:
    """Fit the model with a stream of data points.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data points.
    y : array-like of shape (n_samples,), optional (default=None)
        The labels of the data points.
        Not used in clustering processing, just for calculating purity.
        If not provided, the model will generate a tree with a single label.

    Returns
    -------
    self : STREAMKHC
        Returns self.

    Raises
    ------
    NotFittedError
        If the model has not been initialized with fit.
    ValueError
        If X has a different number of features than seen during fit.
    """
    # Check if model is fitted
    check_is_fitted(self, ["tree_", "iso_kernel_", "n_features_in_"])

    # Process input data
    X = check_array(X, accept_sparse=False)
    if y is None:
        y = np.ones(X.shape[0], dtype=np.int64)
    else:
        X, y = check_X_y(X, y, accept_sparse=False)

    # Check feature consistency
    if X.shape[1] != self.n_features_in_:
        raise ValueError(
            f"X has {X.shape[1]} features, but STREAMKHC was trained with {self.n_features_in_} features."
        )

    # Transform and process data
    X_ikv = self.iso_kernel_.transform(X, dense_output=True)
    self._process_batch(X_ikv, y)
    return self
```

### get_purity

```python
get_purity()
```

Calculate the purity of the clustering tree.

Returns:

| Type    | Description                              |
| ------- | ---------------------------------------- |
| `float` | The purity score of the clustering tree. |

Raises:

| Type             | Description                            |
| ---------------- | -------------------------------------- |
| `NotFittedError` | If the model has not been initialized. |

Source code in `ikpykit/stream/cluster/_streakhc.py`

```python
def get_purity(self) -> float:
    """Calculate the purity of the clustering tree.

    Returns
    -------
    float
        The purity score of the clustering tree.

    Raises
    ------
    NotFittedError
        If the model has not been initialized.
    """
    check_is_fitted(self, ["tree_"])
    if self.tree_ is None:
        return 0.0
    return dendrogram_purity(self.tree_)
```

### serialize_tree

```python
serialize_tree(path)
```

Serialize the clustering tree to a file.

Parameters:

| Name   | Type  | Description                                | Default    |
| ------ | ----- | ------------------------------------------ | ---------- |
| `path` | `str` | The file path to save the serialized tree. | *required* |

Raises:

| Type             | Description                            |
| ---------------- | -------------------------------------- |
| `NotFittedError` | If the model has not been initialized. |

Source code in `ikpykit/stream/cluster/_streakhc.py`

```python
def serialize_tree(self, path: str) -> None:
    """Serialize the clustering tree to a file.

    Parameters
    ----------
    path : str
        The file path to save the serialized tree.

    Raises
    ------
    NotFittedError
        If the model has not been initialized.
    """
    check_is_fitted(self, ["tree_"])
    serliaze_tree_to_file(self.tree_, path)
```

### visualize_tree

```python
visualize_tree(path)
```

Visualize the clustering tree using Graphviz.

Parameters:

| Name   | Type  | Description                              | Default    |
| ------ | ----- | ---------------------------------------- | ---------- |
| `path` | `str` | The file path to save the visualization. | *required* |

Raises:

| Type             | Description                            |
| ---------------- | -------------------------------------- |
| `NotFittedError` | If the model has not been initialized. |

Source code in `ikpykit/stream/cluster/_streakhc.py`

```python
def visualize_tree(self, path: str) -> None:
    """Visualize the clustering tree using Graphviz.

    Parameters
    ----------
    path : str
        The file path to save the visualization.

    Raises
    ------
    NotFittedError
        If the model has not been initialized.
    """
    check_is_fitted(self, ["tree_"])
    Graphviz.write_tree(self.tree_, path)
```
