## ikpykit.timeseries.IKTOD

```python
IKTOD(
    n_estimators_1=100,
    max_samples_1="auto",
    n_estimators_2=100,
    max_samples_2="auto",
    method="inne",
    period_length=10,
    contamination="auto",
    random_state=None,
)
```

Bases: `OutlierMixin`, `BaseEstimator`

Isolation Kernel-based Time series Subsequence Anomaly Detection.

IKTOD implements a distribution-based approach for anomaly time series subsequence detection. Unlike traditional time or frequency domain approaches that rely on sliding windows, IKTOD treats time series subsequences as distributions in R domain, enabling more effective similarity measurements with linear time complexity.

This approach uses Isolation Distributional Kernel (IDK) to measure similarities between subsequences, resulting in better detection accuracy compared to sliding-window-based detectors.

Parameters:

| Name             | Type                                | Description                                                                                                                                                                   | Default  |
| ---------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `n_estimators_1` | `int`                               | Number of base estimators in the first-level ensemble.                                                                                                                        | `100`    |
| `max_samples_1`  | `int, float, or "auto"`             | Number of samples for training each first-level base estimator: - int: exactly max_samples_1 samples - float: max_samples_1 * X.shape[0] samples - "auto": min(8, n_samples)  | `"auto"` |
| `n_estimators_2` | `int`                               | Number of base estimators in the second-level ensemble.                                                                                                                       | `100`    |
| `max_samples_2`  | `int, float, or "auto"`             | Number of samples for training each second-level base estimator: - int: exactly max_samples_2 samples - float: max_samples_2 * X.shape[0] samples - "auto": min(8, n_samples) | `"auto"` |
| `method`         | `(inne, anne)`                      | Isolation method to use: - "inne": original Isolation Forest approach - "anne": approximate nearest neighbor ensemble                                                         | `"inne"` |
| `period_length`  | `int`                               | Length of subsequences to split the time series.                                                                                                                              | `10`     |
| `contamination`  | `auto or float`                     | Proportion of outliers in the dataset: - "auto": threshold determined as in the original paper - float: must be in range (0, 0.5\]                                            | `"auto"` |
| `random_state`   | `int, RandomState instance or None` | Controls randomization for reproducibility.                                                                                                                                   | `None`   |

Attributes:

| Name         | Type    | Description                                      |
| ------------ | ------- | ------------------------------------------------ |
| `ikgad_`     | `IKGAD` | Trained Isolation Kernel Group Anomaly Detector. |
| `offset_`    | `float` | Decision threshold for outlier detection.        |
| `is_fitted_` | `bool`  | Indicates if the model has been fitted.          |

References

1. Ting, K.M., Liu, Z., Zhang, H., Zhu, Y. (2022). A New Distributional Treatment for Time Series and An Anomaly Detection Investigation. Proceedings of The Very Large Data Bases (VLDB) Conference.

Examples:

```pycon
>>> from ikpykit.timeseries import IKTOD
>>> import numpy as np
>>> # Time series with length 40 (4 periods of length 10)
>>> X = np.sin(np.linspace(0, 8*np.pi, 40)).reshape(-1, 1)
>>> # Add anomaly
>>> X[25:30] = X[25:30] + 5.0
>>> detector = IKTOD(max_samples_1=2, max_samples_2=2, contamination=0.1, random_state=42)
>>> detector = detector.fit(X)
>>> detector.predict(X)
array([ 1,  1, -1,  1])
```

Methods:

| Name                | Description                               |
| ------------------- | ----------------------------------------- |
| `fit`               | Fit the IKTOD model.                      |
| `predict`           | Predict if subsequences contain outliers. |
| `decision_function` | Compute decision scores for subsequences. |
| `score_samples`     | Compute anomaly scores for subsequences.  |

Source code in `ikpykit/timeseries/anomaly/_iktod.py`

```python
def __init__(
    self,
    n_estimators_1: int = 100,
    max_samples_1: int | float | str = "auto",
    n_estimators_2: int = 100,
    max_samples_2: int | float | str = "auto",
    method: str = "inne",
    period_length: int = 10,
    contamination: str | float = "auto",
    random_state: int | np.random.RandomState | None = None,
):
    self.n_estimators_1 = n_estimators_1
    self.max_samples_1 = max_samples_1
    self.n_estimators_2 = n_estimators_2
    self.max_samples_2 = max_samples_2
    self.period_length = period_length
    self.random_state = random_state
    self.contamination = contamination
    self.method = method
```

### fit

```python
fit(X)
```

Fit the IKTOD model.

Parameters:

| Name | Type                                          | Description                                                                                                                       | Default    |
| ---- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | Input time series data where: - n_samples: length of the time series - n_features: number of variables (default 1 for univariate) | *required* |

Returns:

| Name   | Type     | Description       |
| ------ | -------- | ----------------- |
| `self` | `object` | Fitted estimator. |

Raises:

| Type         | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `ValueError` | If time series length is less than or equal to period_length. |

Source code in `ikpykit/timeseries/anomaly/_iktod.py`

```python
def fit(self, X) -> "IKTOD":
    """Fit the IKTOD model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input time series data where:
        - n_samples: length of the time series
        - n_features: number of variables (default 1 for univariate)

    Returns
    -------
    self : object
        Fitted estimator.

    Raises
    ------
    ValueError
        If time series length is less than or equal to period_length.
    """
    # Validate input data
    X = check_array(X)

    if len(X) <= self.period_length:
        raise ValueError(
            f"Time series length ({X.shape[0]}) must be greater than "
            f"period_length ({self.period_length})."
        )

    # Check if time series length is compatible with period_length
    rest_samples = X.shape[0] % self.period_length
    if rest_samples != 0:
        warnings.warn(
            f"The last sequence of series has {rest_samples} samples, "
            f"which are less than other sequence.",
            stacklevel=2,
        )

    # Fit the model
    self._fit(X)
    self.is_fitted_ = True

    # Set threshold
    if self.contamination != "auto":
        if not (0.0 < self.contamination <= 0.5):
            raise ValueError(
                f"contamination must be in (0, 0.5], got: {self.contamination}"
            )
        # Define threshold based on contamination parameter
        scores = self.score_samples(X)
        self.offset_ = np.percentile(scores, 100.0 * self.contamination)
    else:
        # Use default threshold as described in the original paper
        self.offset_ = -0.5

    return self
```

### predict

```python
predict(X)
```

Predict if subsequences contain outliers.

Parameters:

| Name | Type                                          | Description                  | Default    |
| ---- | --------------------------------------------- | ---------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | Time series data to evaluate | *required* |

Returns:

| Name     | Type                                 | Description                                                      |
| -------- | ------------------------------------ | ---------------------------------------------------------------- |
| `labels` | `ndarray of shape (n_subsequences,)` | Returns +1 for inliers and -1 for outliers for each subsequence. |

Source code in `ikpykit/timeseries/anomaly/_iktod.py`

```python
def predict(self, X) -> np.ndarray:
    """Predict if subsequences contain outliers.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Time series data to evaluate

    Returns
    -------
    labels : ndarray of shape (n_subsequences,)
        Returns +1 for inliers and -1 for outliers for each subsequence.
    """
    check_is_fitted(self, "is_fitted_")
    X = check_array(X)
    X_sep = self._split_to_subsequences(X)
    return self.ikgad_.predict(X_sep)
```

### decision_function

```python
decision_function(X)
```

Compute decision scores for subsequences.

Parameters:

| Name | Type                                          | Description                  | Default    |
| ---- | --------------------------------------------- | ---------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | Time series data to evaluate | *required* |

Returns:

| Name     | Type                                 | Description                                                                             |
| -------- | ------------------------------------ | --------------------------------------------------------------------------------------- |
| `scores` | `ndarray of shape (n_subsequences,)` | Decision scores. Negative scores represent outliers, positive scores represent inliers. |

Source code in `ikpykit/timeseries/anomaly/_iktod.py`

```python
def decision_function(self, X) -> np.ndarray:
    """Compute decision scores for subsequences.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Time series data to evaluate

    Returns
    -------
    scores : ndarray of shape (n_subsequences,)
        Decision scores. Negative scores represent outliers,
        positive scores represent inliers.
    """
    return self.score_samples(X) - self.offset_
```

### score_samples

```python
score_samples(X)
```

Compute anomaly scores for subsequences.

Parameters:

| Name | Type                                          | Description                  | Default    |
| ---- | --------------------------------------------- | ---------------------------- | ---------- |
| `X`  | `array-like of shape (n_samples, n_features)` | Time series data to evaluate | *required* |

Returns:

| Name     | Type                                 | Description                                                             |
| -------- | ------------------------------------ | ----------------------------------------------------------------------- |
| `scores` | `ndarray of shape (n_subsequences,)` | Anomaly scores where lower values indicate more anomalous subsequences. |

Source code in `ikpykit/timeseries/anomaly/_iktod.py`

```python
def score_samples(self, X) -> np.ndarray:
    """Compute anomaly scores for subsequences.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Time series data to evaluate

    Returns
    -------
    scores : ndarray of shape (n_subsequences,)
        Anomaly scores where lower values indicate more anomalous subsequences.
    """
    check_is_fitted(self, "is_fitted_")
    X = check_array(X)
    X_sep = self._split_to_subsequences(X)
    return self.ikgad_.score_samples(X_sep)
```
