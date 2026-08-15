## ikpykit.trajectory.IKAT

```python
IKAT(
    n_estimators_1=100,
    max_samples_1="auto",
    n_estimators_2=100,
    max_samples_2="auto",
    contamination="auto",
    method="inne",
    random_state=None,
)
```

Bases: `OutlierMixin`, `BaseEstimator`

Isolation-based anomaly detection for trajectory data.

IKAT is a trajectory anomaly detection algorithm that leverages the Isolation Distribution Kernel. Trajectory data is a sequence of points in a multi-dimensional space. It leverages a two-step approach: first transforming the data using an isolation kernel, then calculating kernel mean embeddings for each trajectory data to detect anomalous trajectory. The algorithm is effective for detecting both global and local trajectory anomalies.

Parameters:

| Name             | Type                         | Description                                                                                                                                                                                                               | Default  |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `n_estimators_1` | `int`                        | Number of base estimators in the first step ensemble.                                                                                                                                                                     | `200`    |
| `max_samples_1`  | `(int, float or auto)`       | Number of samples to draw for training each base estimator in first step: - If int, draws exactly max_samples_1 samples - If float, draws max_samples_1 * n_samples samples - If "auto", draws min(8, n_samples) samples  | `"auto"` |
| `n_estimators_2` | `int`                        | Number of base estimators in the second step ensemble.                                                                                                                                                                    | `200`    |
| `max_samples_2`  | `(int, float or auto)`       | Number of samples to draw for training each base estimator in second step: - If int, draws exactly max_samples_2 samples - If float, draws max_samples_2 * n_samples samples - If "auto", draws min(8, n_samples) samples | `"auto"` |
| `method`         | `(inne, anne)`               | Isolation method to use. "inne" is the original algorithm from the paper.                                                                                                                                                 | `"inne"` |
| `contamination`  | `auto or float`              | Proportion of outliers in the dataset: - If "auto", threshold is determined as in the original paper - If float, must be in range (0, 0.5\]                                                                               | `"auto"` |
| `random_state`   | `(int, RandomState or None)` | Controls randomness for reproducibility.                                                                                                                                                                                  | `None`   |

Attributes:

| Name         | Type    | Description                                                      |
| ------------ | ------- | ---------------------------------------------------------------- |
| `offset_`    | `float` | Offset used to define the decision function from the raw scores. |
| `ikgod_`     | `IKGAD` | The fitted IKGAD object.                                         |
| `is_fitted_` | `bool`  | Flag indicating if the estimator is fitted.                      |

References

.. [1] Wang, Y., Wang, Z., Ting, K. M., & Shang, Y. (2024). A Principled Distributional Approach to Trajectory Similarity Measurement and its Application to Anomaly Detection. Journal of Artificial Intelligence Research, 79, 865-893.

Examples:

```pycon
>>> from ikpykit.trajectory import IKAT
>>> from ikpykit.trajectory.dataloader import SheepDogs
>>> sheepdogs = SheepDogs()
>>> X, y = sheepdogs.load(return_X_y=True)
>>> clf = IKAT().fit(X)
>>> predictions = clf.predict(X)
>>> anomaly_scores = clf.score_samples(X)
```

Methods:

| Name                | Description                                        |
| ------------------- | -------------------------------------------------- |
| `fit`               | Fit the anomaly detector.                          |
| `predict`           | Predict if trajectories are outliers or not.       |
| `decision_function` | Compute the decision function for each trajectory. |
| `score_samples`     | Compute the anomaly scores for each trajectory.    |

Source code in `ikpykit/trajectory/anomaly/_ikat.py`

```python
def __init__(
    self,
    n_estimators_1: int = 100,
    max_samples_1: int | float | str = "auto",
    n_estimators_2: int = 100,
    max_samples_2: int | float | str = "auto",
    contamination: str | float = "auto",
    method: Literal["inne", "anne", "auto"] = "inne",
    random_state: int | np.random.RandomState | None = None,
):
    self.n_estimators_1 = n_estimators_1
    self.max_samples_1 = max_samples_1
    self.n_estimators_2 = n_estimators_2
    self.max_samples_2 = max_samples_2
    self.random_state = random_state
    self.contamination = contamination
    self.method = method
```

### fit

```python
fit(X, y=None)
```

Fit the anomaly detector.

Parameters:

| Name | Type                                                         | Description                            | Default    |
| ---- | ------------------------------------------------------------ | -------------------------------------- | ---------- |
| `X`  | `array-like of shape (n_trajectories, n_points, n_features)` | The input trajectories to train on.    | *required* |
| `y`  | `Ignored`                                                    | Not used, present for API consistency. | `None`     |

Returns:

| Name   | Type     | Description       |
| ------ | -------- | ----------------- |
| `self` | `object` | Fitted estimator. |

Raises:

| Type         | Description                                                            |
| ------------ | ---------------------------------------------------------------------- |
| `ValueError` | If contamination is outside of (0, 0.5\] range or method is not valid. |

Source code in `ikpykit/trajectory/anomaly/_ikat.py`

```python
def fit(self, X: list, y: Any = None) -> "IKAT":
    """Fit the anomaly detector.

    Parameters
    ----------
    X : array-like of shape (n_trajectories, n_points, n_features)
        The input trajectories to train on.

    y : Ignored
        Not used, present for API consistency.

    Returns
    -------
    self : object
        Fitted estimator.

    Raises
    ------
    ValueError
        If contamination is outside of (0, 0.5] range or method is not valid.
    """
    X = check_format(X, n_features=2)

    # Validate method parameter
    if self.method not in ["inne", "anne"]:
        raise ValueError(
            f"method must be one of 'inne', 'anne', got: {self.method}"
        )

    # Validate contamination parameter
    if self.contamination != "auto" and not (0.0 < self.contamination <= 0.5):
        raise ValueError(
            f"contamination must be in (0, 0.5], got: {self.contamination}"
        )

    # Fit the model
    self._fit(X)
    self.is_fitted_ = True

    # Set the offset for decision function
    if self.contamination == "auto":
        self.offset_ = -0.5
    else:
        # Set threshold based on contamination level
        self.offset_ = np.percentile(
            self.score_samples(X), 100.0 * self.contamination
        )

    return self
```

### predict

```python
predict(X)
```

Predict if trajectories are outliers or not.

Parameters:

| Name | Type                                                         | Description             | Default    |
| ---- | ------------------------------------------------------------ | ----------------------- | ---------- |
| `X`  | `array-like of shape (n_trajectories, n_points, n_features)` | The input trajectories. | *required* |

Returns:

| Name     | Type                                 | Description                                             |
| -------- | ------------------------------------ | ------------------------------------------------------- |
| `labels` | `ndarray of shape (n_trajectories,)` | The predicted labels: - 1 for inliers - -1 for outliers |

Source code in `ikpykit/trajectory/anomaly/_ikat.py`

```python
def predict(self, X: list) -> list:
    """Predict if trajectories are outliers or not.

    Parameters
    ----------
    X : array-like of shape (n_trajectories, n_points, n_features)
        The input trajectories.

    Returns
    -------
    labels : ndarray of shape (n_trajectories,)
        The predicted labels:
        - 1 for inliers
        - -1 for outliers
    """
    check_is_fitted(self, "is_fitted_")
    X = check_format(X, n_features=2)
    return self.ikgad_.predict(X)
```

### decision_function

```python
decision_function(X)
```

Compute the decision function for each trajectory.

Parameters:

| Name | Type                                                         | Description             | Default    |
| ---- | ------------------------------------------------------------ | ----------------------- | ---------- |
| `X`  | `array-like of shape (n_trajectories, n_points, n_features)` | The input trajectories. | *required* |

Returns:

| Name     | Type                                 | Description                                                                                                           |
| -------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `scores` | `ndarray of shape (n_trajectories,)` | The decision function value for each trajectory. Negative values indicate outliers, positive values indicate inliers. |

Source code in `ikpykit/trajectory/anomaly/_ikat.py`

```python
def decision_function(self, X: list) -> list:
    """Compute the decision function for each trajectory.

    Parameters
    ----------
    X : array-like of shape (n_trajectories, n_points, n_features)
        The input trajectories.

    Returns
    -------
    scores : ndarray of shape (n_trajectories,)
        The decision function value for each trajectory.
        Negative values indicate outliers, positive values indicate inliers.
    """
    check_is_fitted(self, "is_fitted_")
    X = check_format(X, n_features=2)  # Add format check for consistency
    return self.ikgad_.decision_function(X)
```

### score_samples

```python
score_samples(X)
```

Compute the anomaly scores for each trajectory.

Parameters:

| Name | Type                                                         | Description             | Default    |
| ---- | ------------------------------------------------------------ | ----------------------- | ---------- |
| `X`  | `array-like of shape (n_trajectories, n_points, n_features)` | The input trajectories. | *required* |

Returns:

| Name     | Type                                 | Description                                                                                |
| -------- | ------------------------------------ | ------------------------------------------------------------------------------------------ |
| `scores` | `ndarray of shape (n_trajectories,)` | The anomaly scores for each trajectory. Lower scores indicate more anomalous trajectories. |

Source code in `ikpykit/trajectory/anomaly/_ikat.py`

```python
def score_samples(self, X: list) -> list:
    """Compute the anomaly scores for each trajectory.

    Parameters
    ----------
    X : array-like of shape (n_trajectories, n_points, n_features)
        The input trajectories.

    Returns
    -------
    scores : ndarray of shape (n_trajectories,)
        The anomaly scores for each trajectory.
        Lower scores indicate more anomalous trajectories.
    """
    check_is_fitted(self, "is_fitted_")
    X = check_format(X, n_features=2)  # Use check_format consistently

    return self.ikgad_.score_samples(X)
```
