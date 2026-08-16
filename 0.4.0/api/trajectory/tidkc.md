## ikpykit.trajectory.TIDKC

```python
TIDKC(
    k,
    kn,
    v,
    n_init_samples,
    n_estimators_1=100,
    max_samples_1="auto",
    n_estimators_2=100,
    max_samples_2="auto",
    method="anne",
    is_post_process=True,
    random_state=None,
)
```

Bases: `BaseEstimator`, `ClusterMixin`

Trajectory Isolation Distributional Kernel Clustering (TIDKC).

TIDKC identifies non-linearly separable clusters with irregular shapes and varied densities in trajectory data using distributional kernels. It operates in linear time, does not rely on random initialization, and is robust to outliers.

Parameters:

| Name              | Type                                | Description                                                                                                                                                                                                               | Default    |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `k`               | `int`                               | The number of clusters to form.                                                                                                                                                                                           | *required* |
| `kn`              | `int`                               | The number of nearest neighbors to consider when calculating the local contrast.                                                                                                                                          | *required* |
| `v`               | `float`                             | The decay factor for reducing the threshold value.                                                                                                                                                                        | *required* |
| `n_init_samples`  | `int`                               | The number of samples to use for initializing the cluster centers.                                                                                                                                                        | *required* |
| `n_estimators_1`  | `int`                               | Number of base estimators in the first step ensemble.                                                                                                                                                                     | `100`      |
| `max_samples_1`   | `(int, float or auto)`              | Number of samples to draw for training each base estimator in first step: - If int, draws exactly max_samples_1 samples - If float, draws max_samples_1 * n_samples samples - If "auto", draws min(8, n_samples) samples  | `"auto"`   |
| `n_estimators_2`  | `int`                               | Number of base estimators in the second step ensemble.                                                                                                                                                                    | `100`      |
| `max_samples_2`   | `(int, float or auto)`              | Number of samples to draw for training each base estimator in second step: - If int, draws exactly max_samples_2 samples - If float, draws max_samples_2 * n_samples samples - If "auto", draws min(8, n_samples) samples | `"auto"`   |
| `method`          | `(inne, anne)`                      | Isolation method to use. "anne" is the original algorithm from the paper.                                                                                                                                                 | `"inne"`   |
| `is_post_process` | `bool`                              | Whether to perform post-processing to refine the clusters.                                                                                                                                                                | `True`     |
| `random_state`    | `int, RandomState instance or None` | Controls the pseudo-randomness of the selection of the feature and split values for each branching step and each tree in the forest.                                                                                      | `None`     |

Attributes:

| Name          | Type                            | Description                                   |
| ------------- | ------------------------------- | --------------------------------------------- |
| `labels_`     | `ndarray of shape (n_samples,)` | Cluster labels for each point in the dataset. |
| `iso_kernel_` | `IsoKernel`                     | The fitted isolation kernel.                  |
| `idkc_`       | `IDKC`                          | The fitted IDKC clustering model.             |

References

1. Z. J. Wang, Y. Zhu and K. M. Ting, "Distribution-Based Trajectory Clustering," 2023 IEEE International Conference on Data Mining (ICDM).

Examples:

```pycon
>>> from ikpykit.trajectory import TIDKC
>>> from ikpykit.trajectory.dataloader import SheepDogs
>>> sheepdogs = SheepDogs()
>>> X, y = sheepdogs.load(return_X_y=True)
>>> clf = TIDKC(k=2, kn=5, v=0.5, n_init_samples=10).fit(X)
>>> predictions = clf.fit_predict(X)
```

Methods:

| Name          | Description                               |
| ------------- | ----------------------------------------- |
| `fit`         | Fit the trajectory cluster model.         |
| `fit_predict` | Fit the model and predict clusters for X. |

Source code in `ikpykit/trajectory/cluster/_tidkc.py`

```python
def __init__(
    self,
    k: int,
    kn: int,
    v: float,
    n_init_samples: int,
    n_estimators_1: int = 100,
    max_samples_1: int | float | str = "auto",
    n_estimators_2: int = 100,
    max_samples_2: int | float | str = "auto",
    method: Literal["inne", "anne"] = "anne",
    is_post_process: bool = True,
    random_state: int | np.random.RandomState | None = None,
):
    self.n_estimators_1 = n_estimators_1
    self.max_samples_1 = max_samples_1
    self.n_estimators_2 = n_estimators_2
    self.max_samples_2 = max_samples_2
    self.method = method
    self.k = k
    self.kn = kn
    self.v = v
    self.n_init_samples = n_init_samples
    self.is_post_process = is_post_process
    self.random_state = random_state
```

### fit

```python
fit(X, y=None)
```

Fit the trajectory cluster model.

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

| Type         | Description             |
| ------------ | ----------------------- |
| `ValueError` | If method is not valid. |

Source code in `ikpykit/trajectory/cluster/_tidkc.py`

```python
def fit(self, X: list, y: Any = None) -> "TIDKC":
    """Fit the trajectory cluster model.

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
        If method is not valid.
    """
    X = check_format(X, n_features=2)

    # Validate method parameter
    if self.method not in ["inne", "anne"]:
        raise ValueError(
            f"method must be one of 'inne', 'anne', got: {self.method}"
        )

    # Fit the model
    self._fit(X)
    self.is_fitted_ = True
    return self
```

### fit_predict

```python
fit_predict(X, y=None)
```

Fit the model and predict clusters for X.

Parameters:

| Name | Type                                                         | Description                            | Default    |
| ---- | ------------------------------------------------------------ | -------------------------------------- | ---------- |
| `X`  | `array-like of shape (n_trajectories, n_points, n_features)` | The input trajectories.                | *required* |
| `y`  | `Ignored`                                                    | Not used, present for API consistency. | `None`     |

Returns:

| Name     | Type                            | Description     |
| -------- | ------------------------------- | --------------- |
| `labels` | `ndarray of shape (n_samples,)` | Cluster labels. |

Source code in `ikpykit/trajectory/cluster/_tidkc.py`

```python
def fit_predict(self, X, y=None):
    """Fit the model and predict clusters for X.

    Parameters
    ----------
    X : array-like of shape (n_trajectories, n_points, n_features)
        The input trajectories.

    y : Ignored
        Not used, present for API consistency.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    """
    return super().fit_predict(X, y)
```
