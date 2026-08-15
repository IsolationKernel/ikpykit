## ikpykit.stream.ICID

```python
ICID(
    n_estimators=200,
    max_samples_list=None,
    method="inne",
    stability_method="entropy",
    adjust_rate=0.1,
    contamination="auto",
    window_size=10,
    random_state=None,
)
```

Bases: `BaseEstimator`

Isolate Change Interval Detection for monitoring data stream distribution changes.

ICID (Isolate Change Interval Detection) is designed to detect intervals in a data stream where significant distribution changes occur. It leverages isolation-based methods to measure similarity between consecutive data windows, identifying points where the underlying distribution shifts. The algorithm adaptively selects the best sampling parameters for isolation kernels based on stability metrics.

Parameters:

| Name               | Type                                | Description                                                                                                                                                                                                       | Default                 |
| ------------------ | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `n_estimators`     | `int`                               | The number of base estimators in the isolation distribution kernel.                                                                                                                                               | `200`                   |
| `max_samples_list` | `list of int`                       | List of candidate values for max_samples parameter. The algorithm will select the value that yields the most stable isolation kernel.                                                                             | `[2, 4, 8, 16, 32, 64]` |
| `method`           | `(inne, anne)`                      | The isolation method to use for the kernel. - 'inne': Isolation-based Nearest Neighbor Ensemble - 'anne': Approximate Nearest Neighbor Ensemble                                                                   | `'inne'`                |
| `stability_method` | `(entropy, variance, mean)`         | Method used to evaluate the stability of interval scores. - 'entropy': Use information entropy as stability measure - 'variance': Use variance as stability measure - 'mean': Use mean value as stability measure | `'entropy'`             |
| `window_size`      | `int`                               | The size of the sliding window for batch detection.                                                                                                                                                               | `10`                    |
| `adjust_rate`      | `float`                             | Rate to adjust the threshold for anomaly detection based on standard deviation of interval scores.                                                                                                                | `0.1`                   |
| `contamination`    | `auto or float`                     | The proportion of outliers in the data set. Used when fitting to define the threshold on interval scores.                                                                                                         | `'auto'`                |
| `random_state`     | `int, RandomState instance or None` | Controls the randomness of the estimator.                                                                                                                                                                         | `None`                  |

Attributes:

| Name                    | Type                                 | Description                                                           |
| ----------------------- | ------------------------------------ | --------------------------------------------------------------------- |
| `best_iso_kernel_`      | `IsoDisKernel`                       | The fitted isolation kernel with the best stability score.            |
| `best_stability_score_` | `float`                              | The stability score of the best isolation kernel.                     |
| `interval_score_`       | `array-like of shape (n_intervals,)` | The dissimilarity scores between consecutive intervals.               |
| `best_max_samples_`     | `int`                                | The max_samples parameter of the best isolation kernel.               |
| `pre_interval_`         | `array - like`                       | The last interval from the training data, used for online prediction. |

References

.. [1] Y. Cao, Y. Zhu, K. M. Ting, F. D. Salim, H. X. Li, L. Yang, G. Li (2024). Detecting change intervals with isolation distributional kernel. Journal of Artificial Intelligence Research, 79:273-306.

Examples:

```pycon
>>> from ikpykit.stream import ICID
>>> import numpy as np
>>> np.random.seed(42)
>>> X_normal1 = np.random.randn(50, 2)
>>> X_anomaly = np.random.randn(10, 2) * 5 + 10  # Different distribution
>>> X_normal2 = np.random.randn(20, 2)
>>> X = np.vstack([X_normal1, X_anomaly, X_normal2])
>>> icid = ICID(n_estimators=50, max_samples_list=[4, 8], window_size=10, random_state=42)
>>> # Batch predictions
>>> icid.fit_predict_batch(X)
array([ 1,  1,  1,  1,  -1,  -1,  1])
>>> X_anomaly = np.random.randn(10, 2) * 5 + 10
>>> X_normal = np.random.randn(10, 2)
>>> # Predict on new data online
>>> icid.predict_online(X_normal)
1
>>> icid.predict_online(X_anomaly)
-1
```

Methods:

| Name                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `fit`               | Fit the model on data X in batch mode.                                  |
| `fit_predict_batch` | Fit the model on data X and predict anomalies in batch mode.            |
| `predict_online`    | Predict if the new data represents a change from the previous interval. |

Source code in `ikpykit/stream/changedetect/_icid.py`

```python
def __init__(
    self,
    n_estimators=200,
    max_samples_list=None,
    method="inne",
    stability_method="entropy",
    adjust_rate=0.1,
    contamination="auto",
    window_size=10,
    random_state=None,
):
    if max_samples_list is None:
        max_samples_list = [2, 4, 8, 16, 32, 64]
    self.n_estimators = n_estimators
    self.max_samples_list = max_samples_list
    self.method = method
    self.stability_method = stability_method
    self.contamination = contamination
    self.window_size = window_size
    self.random_state = random_state
    self.adjust_rate = adjust_rate
    self.best_iso_kernel_ = None
    self.pre_interval_ = None
    self.interval_score_ = None
    self.best_stability_score_ = float("inf")
```

### best_stability_score

```python
best_stability_score
```

Get the best stability score found during fitting.

### best_iso_kernel

```python
best_iso_kernel
```

Get the isolation kernel with the best stability.

### best_max_samples

```python
best_max_samples
```

Get the max_samples parameter of the best isolation kernel.

### fit

```python
fit(X, y=None)
```

Fit the model on data X in batch mode.

Parameters:

| Name | Type                                        | Description          | Default    |
| ---- | ------------------------------------------- | -------------------- | ---------- |
| `X`  | `np.array of shape (n_samples, n_features)` | The input instances. | *required* |

Returns:

| Name   | Type     | Description |
| ------ | -------- | ----------- |
| `self` | `object` |             |

Source code in `ikpykit/stream/changedetect/_icid.py`

```python
def fit(self, X, y=None):
    """Fit the model on data X in batch mode.

    Parameters
    ----------
    X : np.array of shape (n_samples, n_features)
        The input instances.
    Returns
    -------
    self : object
    """
    X = check_array(X)
    for max_samples in self.max_samples_list:
        isodiskernel = IsoDisKernel(
            n_estimators=self.n_estimators,
            max_samples=max_samples,
            random_state=self.random_state,
            method=self.method,
        )
        isodiskernel.fit(X)
        interval_scores = self._interval_score(X, isodiskernel, self.window_size)
        stability_score = self._stability_score(interval_scores)
        if stability_score < self.best_stability_score_:
            self.best_iso_kernel_ = isodiskernel
            self.best_stability_score_ = stability_score
            self.interval_score_ = interval_scores
    self.is_fitted_ = True
    return self
```

### fit_predict_batch

```python
fit_predict_batch(X)
```

Fit the model on data X and predict anomalies in batch mode.

Parameters:

| Name | Type                                        | Description          | Default    |
| ---- | ------------------------------------------- | -------------------- | ---------- |
| `X`  | `np.array of shape (n_samples, n_features)` | The input instances. | *required* |

Returns:

| Name        | Type                               | Description                                |
| ----------- | ---------------------------------- | ------------------------------------------ |
| `is_inlier` | `np.array of shape (n_intervals,)` | Returns 1 for inliers and -1 for outliers. |

Source code in `ikpykit/stream/changedetect/_icid.py`

```python
def fit_predict_batch(self, X):
    """Fit the model on data X and predict anomalies in batch mode.

    Parameters
    ----------
    X : np.array of shape (n_samples, n_features)
        The input instances.

    Returns
    -------
    is_inlier : np.array of shape (n_intervals,)
        Returns 1 for inliers and -1 for outliers.
    """
    self.fit(X)
    is_inlier = np.ones(len(self.interval_score_), dtype=int)
    threshold = self._determine_anomaly_bounds()
    is_inlier[self.interval_score_ > threshold] = (
        -1
    )  # Higher scores indicate change
    return is_inlier
```

### predict_online

```python
predict_online(X)
```

Predict if the new data represents a change from the previous interval.

Parameters:

| Name | Type                                        | Description                        | Default    |
| ---- | ------------------------------------------- | ---------------------------------- | ---------- |
| `X`  | `np.array of shape (n_samples, n_features)` | The new data interval to evaluate. | *required* |

Returns:

| Name  | Type                                                      | Description |
| ----- | --------------------------------------------------------- | ----------- |
| `int` | `1 for normal (inlier), -1 for change detected (outlier)` |             |

Source code in `ikpykit/stream/changedetect/_icid.py`

```python
def predict_online(self, X):
    """Predict if the new data represents a change from the previous interval.

    Parameters
    ----------
    X : np.array of shape (n_samples, n_features)
        The new data interval to evaluate.

    Returns
    -------
    int : 1 for normal (inlier), -1 for change detected (outlier)
    """
    check_is_fitted(self, ["best_iso_kernel_", "pre_interval_", "interval_score_"])
    X = check_array(X)
    anomaly_score = 1.0 - self.best_iso_kernel_.similarity(self.pre_interval_, X)
    self.interval_score_.append(anomaly_score)
    self.pre_interval_ = X

    threshold = self._determine_anomaly_bounds()
    return 1 if anomaly_score <= threshold else -1
```
