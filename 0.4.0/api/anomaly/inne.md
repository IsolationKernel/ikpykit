## ikpykit.anomaly.INNE

```python
INNE(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=None,
)
```

Bases: `OutlierMixin`, `BaseEstimator`

Isolation-based anomaly detection using nearest-neighbor ensembles.

The INNE algorithm uses the nearest neighbour ensemble to isolate anomalies. It partitions the data space into regions using a subsample and determines an isolation score for each region. As each region adapts to local distribution, the calculated isolation score is a local measure that is relative to the local neighbourhood, enabling it to detect both global and local anomalies. INNE has linear time complexity to efficiently handle large and high-dimensional datasets with complex distributions.

Parameters:

| Name            | Type                                | Description                                                                                                                                                                                                                                                                                              | Default  |
| --------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `n_estimators`  | `int`                               | The number of base estimators in the ensemble.                                                                                                                                                                                                                                                           | `200`    |
| `max_samples`   | `int`                               | The number of samples to draw from X to train each base estimator. - If int, then draw `max_samples` samples. - If float, then draw `max_samples` * X.shape[0]`samples. - If "auto", then`max_samples=min(8, n_samples)\`.                                                                               | `"auto"` |
| `contamination` | `auto or float`                     | The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples. - If "auto", the threshold is determined as in the original paper. - If float, the contamination should be in the range (0, 0.5\]. | `"auto"` |
| `random_state`  | `int, RandomState instance or None` | Controls the pseudo-randomness of the selection of the feature and split values for each branching step and each tree in the forest. Pass an int for reproducible results across multiple function calls. See :term:Glossary \<random_state>.                                                            | `None`   |

References

1. T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells. "Isolation-based anomaly detection using nearest-neighbor ensembles." In Computational Intelligence, vol. 34, 2018, pp. 968-998.

Examples:

```pycon
>>> from ikpykit.anomaly import INNE
>>> import numpy as np
>>> X = np.array([[-1.1, 0.2], [0.3, 0.5], [0.5, 1.1], [100, 90]])
>>> clf = INNE(contamination=0.25).fit(X)
>>> clf.predict([[0.1, 0.3], [0, 0.7], [90, 85]])
array([ 1,  1, -1])
```

Methods:

| Name                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `fit`               | Fit estimator.                                               |
| `predict`           | Predict if a particular sample is an outlier or not.         |
| `decision_function` | Average anomaly score of X of the base classifiers.          |
| `score_samples`     | Opposite of the anomaly score defined in the original paper. |

Source code in `ikpykit/anomaly/_inne.py`

```python
def __init__(
    self,
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=None,
):
    self.n_estimators = n_estimators
    self.max_samples = max_samples
    self.random_state = random_state
    self.contamination = contamination
```

### fit

```python
fit(X, y=None)
```

Fit estimator.

Parameters:

| Name | Type                                          | Description                                                     | Default    |
| ---- | --------------------------------------------- | --------------------------------------------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples. Use dtype=np.float32 for maximum efficiency. | *required* |
| `y`  | `Ignored`                                     | Not used, present for API consistency by convention.            | `None`     |

Returns:

| Name   | Type     | Description       |
| ------ | -------- | ----------------- |
| `self` | `object` | Fitted estimator. |

Source code in `ikpykit/anomaly/_inne.py`

```python
def fit(self, X, y=None):
    """
    Fit estimator.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples. Use ``dtype=np.float32`` for maximum
        efficiency.

    y : Ignored
        Not used, present for API consistency by convention.

    Returns
    -------
    self : object
        Fitted estimator.
    """

    # Check data
    X = check_array(X, accept_sparse=False)

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
    self._fit(X)
    self.is_fitted_ = True

    if self.contamination != "auto":
        if not (0.0 < self.contamination <= 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5], got: {self.contamination:f}"
            )

    if self.contamination == "auto":
        # 0.5 plays a special role as described in the original paper.
        # we take the opposite as we consider the opposite of their score.
        self.offset_ = -0.5
    else:
        # else, define offset_ wrt contamination parameter
        self.offset_ = np.percentile(
            self.score_samples(X), 100.0 * self.contamination
        )

    return self
```

### predict

```python
predict(X)
```

Predict if a particular sample is an outlier or not.

Parameters:

| Name | Type                                          | Description                                                                                                                        | Default    |
| ---- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples. Internally, it will be converted to dtype=np.float32 and if a sparse matrix is provided to a sparse csr_matrix. | *required* |

Returns:

| Name        | Type                            | Description                                                                                                               |
| ----------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `is_inlier` | `ndarray of shape (n_samples,)` | For each observation, tells whether or not (+1 or -1) it should be considered as an inlier according to the fitted model. |

Source code in `ikpykit/anomaly/_inne.py`

```python
def predict(self, X):
    """
    Predict if a particular sample is an outlier or not.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples. Internally, it will be converted to
        ``dtype=np.float32`` and if a sparse matrix is provided
        to a sparse ``csr_matrix``.

    Returns
    -------
    is_inlier : ndarray of shape (n_samples,)
        For each observation, tells whether or not (+1 or -1) it should
        be considered as an inlier according to the fitted model.
    """

    check_is_fitted(self)
    decision_func = self.decision_function(X)
    is_inlier = np.ones_like(decision_func, dtype=int)
    is_inlier[decision_func < 0] = -1
    return is_inlier
```

### decision_function

```python
decision_function(X)
```

Average anomaly score of X of the base classifiers.

The anomaly score of an input sample is computed as the mean anomaly score of the .

Parameters:

| Name | Type                                          | Description                                                              | Default    |
| ---- | --------------------------------------------- | ------------------------------------------------------------------------ | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples. Internally, it will be converted to dtype=np.float32. | *required* |

Returns:

| Name     | Type                            | Description                                                                                                                                  |
| -------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `scores` | `ndarray of shape (n_samples,)` | The anomaly score of the input samples. The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers. |

Source code in `ikpykit/anomaly/_inne.py`

```python
def decision_function(self, X):
    """
    Average anomaly score of X of the base classifiers.

    The anomaly score of an input sample is computed as
    the mean anomaly score of the .

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples. Internally, it will be converted to
        ``dtype=np.float32``.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        The anomaly score of the input samples.
        The lower, the more abnormal. Negative scores represent outliers,
        positive scores represent inliers.
    """
    # We subtract self.offset_ to make 0 be the threshold value for being
    # an outlier.

    return self.score_samples(X) - self.offset_
```

### score_samples

```python
score_samples(X)
```

Opposite of the anomaly score defined in the original paper. The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.

Parameters:

| Name | Type                                          | Description        | Default    |
| ---- | --------------------------------------------- | ------------------ | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | The input samples. | *required* |

Returns:

| Name     | Type                            | Description                                                           |
| -------- | ------------------------------- | --------------------------------------------------------------------- |
| `scores` | `ndarray of shape (n_samples,)` | The anomaly score of the input samples. The lower, the more abnormal. |

Source code in `ikpykit/anomaly/_inne.py`

```python
def score_samples(self, X):
    """
    Opposite of the anomaly score defined in the original paper.
    The anomaly score of an input sample is computed as
    the mean anomaly score of the trees in the forest.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        The anomaly score of the input samples.
        The lower, the more abnormal.
    """

    check_is_fitted(self, "is_fitted_")
    # Check data
    X = check_array(X, accept_sparse=False)

    isolation_scores = np.ones([self.n_estimators, X.shape[0]])
    # each test instance is evaluated against n_estimators sets of hyperspheres
    for i in range(self.n_estimators):
        x_dists = euclidean_distances(X, self._centroids[i], squared=True)
        # find instances that are covered by at least one hypersphere.
        cover_radius = np.where(
            x_dists <= self._centroids_radius[i], self._centroids_radius[i], np.nan
        )
        x_covered = np.where(~np.isnan(cover_radius).all(axis=1))
        # the centroid of the hypersphere covering x and having the smallest radius
        cnn_x = np.nanargmin(cover_radius[x_covered], axis=1)
        isolation_scores[i][x_covered] = self._ratio[i][cnn_x]
    # the isolation scores are averaged to produce the anomaly score
    scores = np.mean(isolation_scores, axis=0)
    return -scores
```
