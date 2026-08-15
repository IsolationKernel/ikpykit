## ikpykit.graph.IKGOD

```python
IKGOD(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    method="inne",
    random_state=None,
    h=3,
)
```

Bases: `BaseEstimator`

Isolation-based Graph Anomaly Detection using kernel embeddings.

This algorithm detects anomalies in graphs by using isolation kernels on subgraph features. It combines graph structure and node features to identify outliers.

Parameters:

| Name            | Type                         | Description                                                                                                                                                                          | Default  |
| --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| `n_estimators`  | `int`                        | Number of isolation estimators in the ensemble.                                                                                                                                      | `200`    |
| `max_samples`   | `(int, float or auto)`       | Number of samples to draw for training each base estimator: - If int, draw max_samples samples - If float, draw max_samples * X.shape[0] samples - If "auto", use min(16, n_samples) | `"auto"` |
| `contamination` | `float or auto`              | Expected proportion of outliers in the data: - If "auto", threshold is set at -0.5 as in the original paper - If float, must be in range (0, 0.5\]                                   | `"auto"` |
| `method`        | `(inne, anne, auto)`         | Isolation method to use. The original algorithm uses "inne".                                                                                                                         | `"inne"` |
| `random_state`  | `(int, RandomState or None)` | Controls randomness for reproducibility.                                                                                                                                             | `None`   |
| `h`             | `int`                        | Maximum hop distance for subgraph extraction.                                                                                                                                        | `3`      |

Attributes:

| Name           | Type                                     | Description                        |
| -------------- | ---------------------------------------- | ---------------------------------- |
| `max_samples_` | `int`                                    | Actual number of samples used      |
| `embedding_`   | `array of shape (n_samples, n_features)` | Learned subgraph embeddings        |
| `offset_`      | `float`                                  | Threshold for determining outliers |
| `is_fitted_`   | `bool`                                   | Whether the model has been fitted  |

References

.. [1] Zhong Zhuang, Kai Ming Ting, Guansong Pang, Shuaibin Song (2023). Subgraph Centralization: A Necessary Step for Graph Anomaly Detection. Proceedings of The SIAM Conference on Data Mining.

Examples:

```pycon
>>> from ikpykit.graph import IKGOD
>>> import scipy.sparse as sp
>>> import numpy as np
>>> # Create adjacency matrix and features
>>> adj = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
>>> features = np.array([[0.1, 0.2], [0.3, 0.4], [5.0, 6.0]])
>>> # Fit model
>>> model = IKGOD(n_estimators=100, h=2).fit(adj, features)
>>> # Predict outliers
>>> labels = model.predict(features)
```

Methods:

| Name                | Description                         |
| ------------------- | ----------------------------------- |
| `fit`               | Fit the IKGOD model.                |
| `predict`           | Predict outliers in X.              |
| `decision_function` | Compute decision function.          |
| `score_samples`     | Compute anomaly scores for samples. |

Source code in `ikpykit/graph/_ikgod.py`

```python
def __init__(
    self,
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    method="inne",
    random_state=None,
    h=3,
):
    self.n_estimators = n_estimators
    self.max_samples = max_samples
    self.random_state = random_state
    self.contamination = contamination
    self.method = method
    self.h = h
```

### fit

```python
fit(adjacency, features, y=None)
```

Fit the IKGOD model.

Parameters:

| Name        | Type                                                          | Description                            | Default    |
| ----------- | ------------------------------------------------------------- | -------------------------------------- | ---------- |
| `adjacency` | `array-like or sparse matrix of shape (n_samples, n_samples)` | Adjacency matrix of the graph          | *required* |
| `features`  | `array-like of shape (n_samples, n_features)`                 | Node features                          | *required* |
| `y`         | `Ignored`                                                     | Not used, present for API consistency. | `None`     |

Returns:

| Name   | Type     | Description       |
| ------ | -------- | ----------------- |
| `self` | `object` | Fitted estimator. |

Source code in `ikpykit/graph/_ikgod.py`

```python
def fit(self, adjacency, features, y=None):
    """Fit the IKGOD model.

    Parameters
    ----------
    adjacency : array-like or sparse matrix of shape (n_samples, n_samples)
        Adjacency matrix of the graph

    features : array-like of shape (n_samples, n_features)
        Node features

    y : Ignored
        Not used, present for API consistency.

    Returns
    -------
    self : object
        Fitted estimator.
    """
    # Check and format inputs
    adjacency = check_format(adjacency)
    features = check_array(features, accept_sparse=False)

    n_samples = features.shape[0]

    # Determine max_samples
    if isinstance(self.max_samples, str):
        if self.max_samples == "auto":
            max_samples = min(16, n_samples)
        else:
            raise ValueError(
                f"max_samples '{self.max_samples}' is not supported. "
                f'Valid choices are: "auto", int or float'
            )
    elif isinstance(self.max_samples, numbers.Integral):
        if self.max_samples > n_samples:
            warn(
                f"max_samples ({self.max_samples}) is greater than the "
                f"total number of samples ({n_samples}). max_samples "
                f"will be set to n_samples for estimation.",
                stacklevel=2,
            )
            max_samples = n_samples
        else:
            max_samples = self.max_samples
    else:  # float
        if not 0.0 < self.max_samples <= 1.0:
            raise ValueError(
                f"max_samples must be in (0, 1], got {self.max_samples}"
            )
        max_samples = int(self.max_samples * n_samples)

    self.max_samples_ = max_samples

    # Fit the model
    self._fit(adjacency, features)
    self.is_fitted_ = True

    # Set contamination threshold
    if self.contamination != "auto":
        if not (0.0 < self.contamination <= 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5], got: {self.contamination}"
            )

    if self.contamination == "auto":
        # 0.5 plays a special role as described in the original paper
        self.offset_ = -0.5
    else:
        # Set threshold based on contamination parameter
        self.offset_ = np.percentile(
            self.score_samples(features), 100.0 * self.contamination
        )

    return self
```

### predict

```python
predict(X)
```

Predict outliers in X.

Parameters:

| Name | Type                                          | Description       | Default    |
| ---- | --------------------------------------------- | ----------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples | *required* |

Returns:

| Name        | Type                            | Description                     |
| ----------- | ------------------------------- | ------------------------------- |
| `is_inlier` | `ndarray of shape (n_samples,)` | +1 for inliers, -1 for outliers |

Source code in `ikpykit/graph/_ikgod.py`

```python
def predict(self, X):
    """Predict outliers in X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples

    Returns
    -------
    is_inlier : ndarray of shape (n_samples,)
        +1 for inliers, -1 for outliers
    """
    check_is_fitted(self, "is_fitted_")
    decision_func = self.decision_function(X)
    is_inlier = np.ones_like(decision_func, dtype=int)
    is_inlier[decision_func < 0] = -1
    return is_inlier
```

### decision_function

```python
decision_function(X)
```

Compute decision function.

Parameters:

| Name | Type                                          | Description       | Default    |
| ---- | --------------------------------------------- | ----------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples | *required* |

Returns:

| Name     | Type                            | Description                                          |
| -------- | ------------------------------- | ---------------------------------------------------- |
| `scores` | `ndarray of shape (n_samples,)` | Decision scores. Negative scores represent outliers. |

Source code in `ikpykit/graph/_ikgod.py`

```python
def decision_function(self, X):
    """Compute decision function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Decision scores. Negative scores represent outliers.
    """
    return self.score_samples(X) - self.offset_
```

### score_samples

```python
score_samples(X)
```

Compute anomaly scores for samples.

Lower scores indicate more anomalous points.

Parameters:

| Name | Type                                          | Description       | Default    |
| ---- | --------------------------------------------- | ----------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples | *required* |

Returns:

| Name     | Type                            | Description                                                  |
| -------- | ------------------------------- | ------------------------------------------------------------ |
| `scores` | `ndarray of shape (n_samples,)` | Anomaly scores. Lower values indicate more anomalous points. |

Source code in `ikpykit/graph/_ikgod.py`

```python
def score_samples(self, X):
    """Compute anomaly scores for samples.

    Lower scores indicate more anomalous points.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Anomaly scores. Lower values indicate more anomalous points.
    """
    check_is_fitted(self, "is_fitted_")
    X = check_array(X, accept_sparse=False)
    kme = self._kernel_mean_embedding(self.embedding_)
    scores = safe_sparse_dot(self.embedding_, kme.T).A1
    return -scores
```
